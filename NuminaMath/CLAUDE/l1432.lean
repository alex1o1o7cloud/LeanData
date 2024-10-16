import Mathlib

namespace NUMINAMATH_CALUDE_A_not_lose_probability_l1432_143267

/-- The probability of player A winning -/
def prob_A_win : ℝ := 0.30

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := 0.25

/-- The probability that player A does not lose -/
def prob_A_not_lose : ℝ := prob_A_win + prob_draw

theorem A_not_lose_probability : prob_A_not_lose = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_A_not_lose_probability_l1432_143267


namespace NUMINAMATH_CALUDE_hundredth_term_is_one_l1432_143241

/-- Defines the sequence term at position n -/
def sequenceTerm (n : ℕ) : ℕ :=
  sorry

/-- The number of elements in the first n groups -/
def elementsInGroups (n : ℕ) : ℕ :=
  n^2

theorem hundredth_term_is_one :
  sequenceTerm 100 = 1 :=
sorry

end NUMINAMATH_CALUDE_hundredth_term_is_one_l1432_143241


namespace NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l1432_143219

theorem frog_grasshopper_jump_difference :
  let grasshopper_jump : ℕ := 9
  let frog_jump : ℕ := 12
  frog_jump - grasshopper_jump = 3 := by sorry

end NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l1432_143219


namespace NUMINAMATH_CALUDE_calculation_proof_l1432_143206

theorem calculation_proof :
  (((3 * Real.sqrt 48) - (2 * Real.sqrt 27)) / Real.sqrt 3 = 6) ∧
  ((Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + (1 / (2 - Real.sqrt 5)) = -3 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1432_143206


namespace NUMINAMATH_CALUDE_neighborhood_glass_panels_l1432_143251

/-- Represents the number of houses of each type -/
def num_houses_A : ℕ := 4
def num_houses_B : ℕ := 3
def num_houses_C : ℕ := 3

/-- Represents the number of glass panels in each type of house -/
def panels_per_house_A : ℕ := 
  4 * 6 + -- double windows downstairs
  8 * 3 + -- single windows upstairs
  2 * 6 + -- bay windows
  1 * 2 + -- front door
  1 * 3   -- back door

def panels_per_house_B : ℕ := 
  8 * 5 + -- double windows downstairs
  6 * 4 + -- single windows upstairs
  1 * 7 + -- bay window
  1 * 4   -- front door

def panels_per_house_C : ℕ := 
  5 * 4 + -- double windows downstairs
  10 * 2 + -- single windows upstairs
  3 * 1   -- skylights

/-- The total number of glass panels in the neighborhood -/
def total_panels : ℕ := 
  num_houses_A * panels_per_house_A +
  num_houses_B * panels_per_house_B +
  num_houses_C * panels_per_house_C

/-- Theorem stating that the total number of glass panels in the neighborhood is 614 -/
theorem neighborhood_glass_panels : total_panels = 614 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_glass_panels_l1432_143251


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1432_143237

/-- The ratio of a man's age to his son's age in two years -/
def age_ratio (man_age son_age : ℕ) : ℚ :=
  (man_age + 2) / (son_age + 2)

/-- Theorem stating the age ratio of a man to his son in two years -/
theorem man_son_age_ratio (son_age : ℕ) (h1 : son_age = 22) :
  age_ratio (son_age + 24) son_age = 2 := by
  sorry

#check man_son_age_ratio

end NUMINAMATH_CALUDE_man_son_age_ratio_l1432_143237


namespace NUMINAMATH_CALUDE_arithmetic_sum_problem_l1432_143270

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₃ = 2 and a₃ + a₅ = 4, prove a₅ + a₇ = 6. -/
theorem arithmetic_sum_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum1 : a 1 + a 3 = 2) 
  (h_sum2 : a 3 + a 5 = 4) : 
  a 5 + a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_problem_l1432_143270


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1432_143233

/-- Parametric equation of a line in 2D space -/
structure ParametricLine2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point lies on a parametric line -/
def pointOnLine (p : ℝ × ℝ) (l : ParametricLine2D) : Prop :=
  ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

theorem intersection_of_lines (line1 line2 : ParametricLine2D)
    (h1 : line1 = ParametricLine2D.mk (5, 1) (3, -2))
    (h2 : line2 = ParametricLine2D.mk (2, 8) (5, -3)) :
    ∃! p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 ∧ p = (-73, 53) := by
  sorry

#check intersection_of_lines

end NUMINAMATH_CALUDE_intersection_of_lines_l1432_143233


namespace NUMINAMATH_CALUDE_gcd_set_divisors_l1432_143242

theorem gcd_set_divisors (a b c d : ℕ+) (h1 : a * d ≠ b * c) (h2 : Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd a d = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd b d = 1 ∧ Nat.gcd c d = 1) :
  ∃ k : ℕ, {x : ℕ | ∃ n : ℕ+, x = Nat.gcd (a * n + b) (c * n + d)} = {x : ℕ | x ∣ k} := by
  sorry


end NUMINAMATH_CALUDE_gcd_set_divisors_l1432_143242


namespace NUMINAMATH_CALUDE_side_length_of_five_cubes_l1432_143247

/-- Given five equal cubes placed adjacent to each other forming a new solid with volume 625 cm³,
    prove that the side length of each cube is 5 cm. -/
theorem side_length_of_five_cubes (n : ℕ) (v : ℝ) (s : ℝ) : 
  n = 5 → v = 625 → v = n * s^3 → s = 5 := by sorry

end NUMINAMATH_CALUDE_side_length_of_five_cubes_l1432_143247


namespace NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l1432_143226

theorem conditions_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → |x| ≤ 1 ∧ |y| ≤ 1) ∧
  ¬(∀ x y : ℝ, |x| ≤ 1 ∧ |y| ≤ 1 → x^2 + y^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l1432_143226


namespace NUMINAMATH_CALUDE_danielle_travel_time_l1432_143259

-- Define the speeds and times
def chase_speed : ℝ := 1 -- Normalized speed
def chase_time : ℝ := 180 -- Minutes
def cameron_speed : ℝ := 2 * chase_speed
def danielle_speed : ℝ := 3 * cameron_speed

-- Define the distance (constant for all travelers)
def distance : ℝ := chase_speed * chase_time

-- Theorem to prove
theorem danielle_travel_time : 
  (distance / danielle_speed) = 30 := by
sorry

end NUMINAMATH_CALUDE_danielle_travel_time_l1432_143259


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1432_143200

/-- The perimeter of a semicircle with radius 6.83 cm is approximately 35.12 cm. -/
theorem semicircle_perimeter_approx : 
  let r : Real := 6.83
  let perimeter : Real := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 35.12) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l1432_143200


namespace NUMINAMATH_CALUDE_half_triangles_isosceles_l1432_143223

/-- A function that returns the number of pairwise non-congruent triangles
    that can be formed from N points on a circle. -/
def totalTriangles (N : ℕ) : ℕ := N * (N - 1) * (N - 2) / 6

/-- A function that returns the number of isosceles triangles
    that can be formed from N points on a circle. -/
def isoscelesTriangles (N : ℕ) : ℕ := N * (N - 2) / 3

/-- The theorem stating that exactly half of the triangles are isosceles
    if and only if N is 10 or 11, for N > 2. -/
theorem half_triangles_isosceles (N : ℕ) (h : N > 2) :
  2 * isoscelesTriangles N = totalTriangles N ↔ N = 10 ∨ N = 11 :=
sorry

end NUMINAMATH_CALUDE_half_triangles_isosceles_l1432_143223


namespace NUMINAMATH_CALUDE_total_age_problem_l1432_143207

theorem total_age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 10 →
  a + b + c = 27 :=
by sorry

end NUMINAMATH_CALUDE_total_age_problem_l1432_143207


namespace NUMINAMATH_CALUDE_stratified_sample_fourth_unit_l1432_143239

/-- Represents a stratified sample from four units -/
structure StratifiedSample :=
  (total : ℕ)
  (unit_samples : Fin 4 → ℕ)
  (is_arithmetic : ∃ d : ℤ, ∀ i : Fin 3, (unit_samples i.succ : ℤ) - (unit_samples i) = d)
  (sum_to_total : (Finset.univ.sum unit_samples) = total)

/-- The theorem statement -/
theorem stratified_sample_fourth_unit 
  (sample : StratifiedSample)
  (total_collected : ℕ)
  (h_total : sample.total = 150)
  (h_collected : total_collected = 1000)
  (h_second_unit : sample.unit_samples 1 = 30) :
  sample.unit_samples 3 = 60 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_fourth_unit_l1432_143239


namespace NUMINAMATH_CALUDE_anns_age_l1432_143234

theorem anns_age (A B t T : ℕ) : 
  A + B = 44 →
  B = A - t →
  B - t = A - T →
  B - T = A / 2 →
  A = 24 :=
by sorry

end NUMINAMATH_CALUDE_anns_age_l1432_143234


namespace NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l1432_143224

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

-- Theorem for angle A
theorem angle_A_measure (t : Triangle) (h : given_condition t) : t.A = π / 3 :=
sorry

-- Theorem for perimeter range
theorem perimeter_range (t : Triangle) (h1 : given_condition t) (h2 : t.a = 6) :
  12 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_angle_A_measure_perimeter_range_l1432_143224


namespace NUMINAMATH_CALUDE_ellipse_y_axis_l1432_143248

/-- The equation represents an ellipse with focal points on the y-axis -/
theorem ellipse_y_axis (x y : ℝ) : 
  (x^2 / (Real.sin (Real.sqrt 2) - Real.sin (Real.sqrt 3))) + 
  (y^2 / (Real.cos (Real.sqrt 2) - Real.cos (Real.sqrt 3))) = 1 →
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_l1432_143248


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1432_143229

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1432_143229


namespace NUMINAMATH_CALUDE_kangaroo_meeting_count_l1432_143297

def kangaroo_a_period : ℕ := 9
def kangaroo_b_period : ℕ := 6
def total_jumps : ℕ := 2017

def meeting_count (a_period b_period total_jumps : ℕ) : ℕ :=
  let lcm := Nat.lcm a_period b_period
  let meetings_per_cycle := 2  -- They meet twice in each LCM cycle
  let complete_cycles := total_jumps / lcm
  let remainder := total_jumps % lcm
  let meetings_in_complete_cycles := complete_cycles * meetings_per_cycle
  let initial_meeting := 1  -- They start at the same point
  let extra_meeting := if remainder ≥ 1 then 1 else 0
  meetings_in_complete_cycles + initial_meeting + extra_meeting

theorem kangaroo_meeting_count :
  meeting_count kangaroo_a_period kangaroo_b_period total_jumps = 226 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_meeting_count_l1432_143297


namespace NUMINAMATH_CALUDE_value_of_a_l1432_143220

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- State the theorem
theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 3 * a * x^2 + 6 * x) → 
  deriv (f a) (-1) = 3 → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1432_143220


namespace NUMINAMATH_CALUDE_magnitude_of_c_l1432_143299

def vector_a : Fin 2 → ℝ := ![1, -1]
def vector_b : Fin 2 → ℝ := ![2, 1]

def vector_c : Fin 2 → ℝ := λ i => 2 * vector_a i + vector_b i

theorem magnitude_of_c :
  Real.sqrt ((vector_c 0) ^ 2 + (vector_c 1) ^ 2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_c_l1432_143299


namespace NUMINAMATH_CALUDE_max_volume_smaller_pyramid_l1432_143204

/-- Regular square pyramid with base side length 2 and height 3 -/
structure SquarePyramid where
  base_side : ℝ
  height : ℝ
  base_side_eq : base_side = 2
  height_eq : height = 3

/-- Smaller pyramid formed by intersecting the main pyramid with a parallel plane -/
structure SmallerPyramid (p : SquarePyramid) where
  intersection_height : ℝ
  volume : ℝ
  height_bounds : 0 < intersection_height ∧ intersection_height < p.height
  volume_eq : volume = (4 / 27) * intersection_height^3 - (8 / 9) * intersection_height^2 + (4 / 3) * intersection_height

/-- The maximum volume of the smaller pyramid is 16/27 -/
theorem max_volume_smaller_pyramid (p : SquarePyramid) : 
  ∃ (sp : SmallerPyramid p), ∀ (other : SmallerPyramid p), sp.volume ≥ other.volume ∧ sp.volume = 16/27 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_smaller_pyramid_l1432_143204


namespace NUMINAMATH_CALUDE_manuscript_cost_example_l1432_143271

def manuscript_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) (revised_thrice : ℕ) 
  (initial_cost : ℕ) (revision_cost : ℕ) : ℕ :=
  let no_revision := total_pages - (revised_once + revised_twice + revised_thrice)
  let cost_no_revision := no_revision * initial_cost
  let cost_revised_once := revised_once * (initial_cost + revision_cost)
  let cost_revised_twice := revised_twice * (initial_cost + 2 * revision_cost)
  let cost_revised_thrice := revised_thrice * (initial_cost + 3 * revision_cost)
  cost_no_revision + cost_revised_once + cost_revised_twice + cost_revised_thrice

theorem manuscript_cost_example : 
  manuscript_cost 300 55 35 25 8 6 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_example_l1432_143271


namespace NUMINAMATH_CALUDE_sign_of_a_l1432_143290

theorem sign_of_a (a b c : ℝ) (n : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ((-2)^8 * a^3 * b^3 * c^(n-1)) * ((-3)^3 * a^2 * b^5 * c^(n+1)) > 0) : 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_sign_of_a_l1432_143290


namespace NUMINAMATH_CALUDE_bens_income_l1432_143265

/-- Represents the state income tax calculation and Ben's specific case -/
theorem bens_income (q : ℝ) : 
  ∃ (A : ℝ), 
    (A > 35000) ∧ 
    (0.01 * q * 35000 + 0.01 * (q + 4) * (A - 35000) = (0.01 * (q + 0.5)) * A) ∧ 
    (A = 40000) := by
  sorry

end NUMINAMATH_CALUDE_bens_income_l1432_143265


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1432_143232

theorem diophantine_equation_solution :
  ∀ x y z : ℕ,
    x ≤ y →
    x^2 + y^2 = 3 * 2016^z + 77 →
    ((x = 4 ∧ y = 8 ∧ z = 0) ∨
     (x = 14 ∧ y = 49 ∧ z = 1) ∨
     (x = 35 ∧ y = 70 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1432_143232


namespace NUMINAMATH_CALUDE_inequality_proof_l1432_143261

theorem inequality_proof (x : ℝ) : (x - 4) / 2 - (x - 1) / 4 < 1 → x < 11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1432_143261


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l1432_143217

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, y)
  collinear a b → y = -6 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l1432_143217


namespace NUMINAMATH_CALUDE_problem_solution_l1432_143202

theorem problem_solution : (88 * 707 - 38 * 707) / 1414 = 25 := by
  have h : 1414 = 707 * 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1432_143202


namespace NUMINAMATH_CALUDE_managers_salary_l1432_143266

/-- Given 100 employees with an average monthly salary of 3500 rupees,
    if adding one more person (the manager) increases the average salary by 800 rupees,
    then the manager's salary is 84300 rupees. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 100 →
  avg_salary = 3500 →
  salary_increase = 800 →
  (num_employees * avg_salary + 84300) / (num_employees + 1) = avg_salary + salary_increase :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l1432_143266


namespace NUMINAMATH_CALUDE_fabric_still_needed_l1432_143283

def fabric_per_pair : ℝ := 8.5
def pairs_needed : ℕ := 7
def fabric_available_yards : ℝ := 3.5
def yards_to_feet : ℝ := 3

def fabric_needed (fabric_per_pair : ℝ) (pairs_needed : ℕ) : ℝ :=
  fabric_per_pair * (pairs_needed : ℝ)

def fabric_available_feet (fabric_available_yards : ℝ) (yards_to_feet : ℝ) : ℝ :=
  fabric_available_yards * yards_to_feet

theorem fabric_still_needed :
  fabric_needed fabric_per_pair pairs_needed - fabric_available_feet fabric_available_yards yards_to_feet = 49 := by
  sorry

end NUMINAMATH_CALUDE_fabric_still_needed_l1432_143283


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l1432_143255

theorem unique_root_of_equation :
  ∃! x : ℝ, Real.sqrt (x + 25) - 7 / Real.sqrt (x + 25) = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l1432_143255


namespace NUMINAMATH_CALUDE_operational_probability_independent_of_root_l1432_143214

/-- Represents a computer network -/
structure ComputerNetwork where
  servers : Type
  channels : servers → servers → Prop
  failure_prob : ℝ
  failure_prob_nonneg : 0 ≤ failure_prob
  failure_prob_le_one : failure_prob ≤ 1

/-- Predicate to check if a server can reach another server using operating channels -/
def can_reach (G : ComputerNetwork) (s t : G.servers) : Prop :=
  sorry

/-- Predicate to check if a network is operational with respect to a root server -/
def is_operational (G : ComputerNetwork) (r : G.servers) : Prop :=
  ∀ s : G.servers, can_reach G s r

/-- The probability that a network is operational -/
noncomputable def operational_probability (G : ComputerNetwork) (r : G.servers) : ℝ :=
  sorry

/-- Theorem stating that the operational probability is independent of the choice of root server -/
theorem operational_probability_independent_of_root (G : ComputerNetwork) 
  (r₁ r₂ : G.servers) (h : r₁ ≠ r₂) : 
  operational_probability G r₁ = operational_probability G r₂ :=
sorry

end NUMINAMATH_CALUDE_operational_probability_independent_of_root_l1432_143214


namespace NUMINAMATH_CALUDE_emberly_walks_l1432_143205

theorem emberly_walks (total_days : Nat) (miles_per_walk : Nat) (total_miles : Nat) :
  total_days = 31 →
  miles_per_walk = 4 →
  total_miles = 108 →
  total_days - (total_miles / miles_per_walk) = 4 :=
by sorry

end NUMINAMATH_CALUDE_emberly_walks_l1432_143205


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1432_143209

-- Define the two fixed circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the trajectory hyperbola
def trajectory_hyperbola (x y : ℝ) : Prop := 4*(x + 2)^2 - y^2 = 1

-- Define the concept of a moving circle being externally tangent to two fixed circles
def externally_tangent (x y : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (∃ (x_m y_m : ℝ), circle_M x_m y_m ∧ (x - x_m)^2 + (y - y_m)^2 = (r + 1)^2) ∧
  (∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = (r + 2)^2)

-- The main theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), externally_tangent x y → trajectory_hyperbola x y ∧ x < -2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1432_143209


namespace NUMINAMATH_CALUDE_point_on_line_m_value_l1432_143291

/-- A point with coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

theorem point_on_line_m_value :
  ∀ (m : ℝ),
  let A : Point := ⟨2, m⟩
  let L : Line := ⟨-2, 3⟩
  pointOnLine A L → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_m_value_l1432_143291


namespace NUMINAMATH_CALUDE_target_distribution_l1432_143238

def target_parts : Nat := 10

def arrange_decreasing (n : Nat) (center : Nat) (middle : Nat) (outer : Nat) : Nat :=
  sorry

def arrange_equal_sum (n : Nat) (center : Nat) (middle : Nat) (outer : Nat) : Nat :=
  sorry

theorem target_distribution :
  (Nat.factorial target_parts = 3628800) ∧
  (arrange_decreasing target_parts 1 3 6 = 4320) ∧
  (arrange_equal_sum target_parts 1 3 6 = 34560) := by
  sorry

end NUMINAMATH_CALUDE_target_distribution_l1432_143238


namespace NUMINAMATH_CALUDE_heart_value_is_three_l1432_143275

/-- Represents a digit in base 9 and base 10 notation -/
def Heart : ℕ → Prop :=
  λ n => 0 ≤ n ∧ n ≤ 9

theorem heart_value_is_three :
  ∀ h : ℕ,
  Heart h →
  (h * 9 + 6 = h * 10 + 3) →
  h = 3 := by
sorry

end NUMINAMATH_CALUDE_heart_value_is_three_l1432_143275


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1432_143245

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, a * x^2 + b * x + c = 0 ↔ 3 * x^2 - 4 * x + 1 = 0) →
  a = 3 ∧ b = -4 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1432_143245


namespace NUMINAMATH_CALUDE_subtract_negative_l1432_143210

theorem subtract_negative : -2 - (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1432_143210


namespace NUMINAMATH_CALUDE_soccer_team_beverage_consumption_l1432_143274

theorem soccer_team_beverage_consumption 
  (team_size : ℕ) 
  (total_beverage : ℕ) 
  (h1 : team_size = 36) 
  (h2 : total_beverage = 252) :
  total_beverage / team_size = 7 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_beverage_consumption_l1432_143274


namespace NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l1432_143215

-- Define the four lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 15
def line2 (x y : ℚ) : Prop := -3 * x - 4 * y = 15
def line3 (x y : ℚ) : Prop := 4 * y + 3 * x = 15
def line4 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Define perpendicularity
def perpendicular (f g : ℚ → ℚ → Prop) : Prop :=
  ∃ m1 m2 : ℚ, (∀ x y, f x y ↔ y = m1 * x + (15 / 4)) ∧
             (∀ x y, g x y ↔ y = m2 * x + 5) ∧
             m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_and_intersection :
  perpendicular line1 line4 ∧
  line1 (15/32) (35/8) ∧
  line4 (15/32) (35/8) := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_and_intersection_l1432_143215


namespace NUMINAMATH_CALUDE_shopping_mall_pricing_l1432_143289

/-- Shopping mall pricing problem -/
theorem shopping_mall_pricing
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_monthly_sales : ℝ)
  (sales_increase_rate : ℝ)
  (target_monthly_profit : ℝ)
  (h1 : purchase_price = 280)
  (h2 : initial_selling_price = 360)
  (h3 : initial_monthly_sales = 60)
  (h4 : sales_increase_rate = 5)
  (h5 : target_monthly_profit = 7200) :
  ∃ (price_reduction : ℝ),
    price_reduction = 60 ∧
    (initial_selling_price - price_reduction - purchase_price) *
    (initial_monthly_sales + sales_increase_rate * price_reduction) =
    target_monthly_profit :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_pricing_l1432_143289


namespace NUMINAMATH_CALUDE_rational_function_value_l1432_143262

-- Define the function types
def linear_function (α : Type*) [Ring α] := α → α
def quadratic_function (α : Type*) [Ring α] := α → α
def rational_function (α : Type*) [Field α] := α → α

-- Define the properties of the rational function
def has_vertical_asymptotes (f : rational_function ℝ) (a b : ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ a ∧ x ≠ b → f x ≠ 0

def passes_through (f : rational_function ℝ) (x y : ℝ) : Prop :=
  f x = y

-- Main theorem
theorem rational_function_value
  (p : linear_function ℝ)
  (q : quadratic_function ℝ)
  (f : rational_function ℝ)
  (h1 : ∀ (x : ℝ), f x = p x / q x)
  (h2 : has_vertical_asymptotes f (-1) 4)
  (h3 : passes_through f 0 0)
  (h4 : passes_through f 1 (-3)) :
  p (-2) / q (-2) = -6 :=
sorry

end NUMINAMATH_CALUDE_rational_function_value_l1432_143262


namespace NUMINAMATH_CALUDE_probability_three_heads_five_coins_l1432_143211

theorem probability_three_heads_five_coins :
  let n : ℕ := 5  -- number of coins
  let k : ℕ := 3  -- number of heads we want
  let p : ℚ := 1/2  -- probability of getting heads on a single coin toss
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = 5/16 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_five_coins_l1432_143211


namespace NUMINAMATH_CALUDE_mitzi_remaining_money_l1432_143287

/-- Proves that Mitzi has $9 left after her amusement park expenses -/
theorem mitzi_remaining_money (initial_amount ticket_cost food_cost tshirt_cost : ℕ) 
  (h1 : initial_amount = 75)
  (h2 : ticket_cost = 30)
  (h3 : food_cost = 13)
  (h4 : tshirt_cost = 23) :
  initial_amount - (ticket_cost + food_cost + tshirt_cost) = 9 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_remaining_money_l1432_143287


namespace NUMINAMATH_CALUDE_z_value_proof_l1432_143260

theorem z_value_proof : 
  ∃ z : ℝ, (12 / 20 = (z / 20) ^ (1/3)) ∧ z = 4.32 :=
by
  sorry

end NUMINAMATH_CALUDE_z_value_proof_l1432_143260


namespace NUMINAMATH_CALUDE_large_font_pages_l1432_143258

/-- Represents the number of words per page for large font -/
def large_font_words_per_page : ℕ := 1800

/-- Represents the number of words per page for small font -/
def small_font_words_per_page : ℕ := 2400

/-- Represents the total number of pages allowed -/
def total_pages : ℕ := 21

/-- Represents the ratio of large font pages to small font pages -/
def font_ratio : Rat := 2 / 3

theorem large_font_pages : ℕ :=
  let large_pages : ℕ := 8
  let small_pages : ℕ := total_pages - large_pages
  have h1 : large_pages + small_pages = total_pages := by sorry
  have h2 : (large_pages : Rat) / (small_pages : Rat) = font_ratio := by sorry
  have h3 : large_pages * large_font_words_per_page + small_pages * small_font_words_per_page ≤ 48000 := by sorry
  large_pages

end NUMINAMATH_CALUDE_large_font_pages_l1432_143258


namespace NUMINAMATH_CALUDE_larger_number_proof_l1432_143279

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 17) (h2 : x - y = 7) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1432_143279


namespace NUMINAMATH_CALUDE_function_increasing_l1432_143288

theorem function_increasing (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁) : 
  StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_l1432_143288


namespace NUMINAMATH_CALUDE_volumeAsFractionOfLitre_l1432_143250

-- Define the conversion factor from litres to millilitres
def litreToMl : ℝ := 1000

-- Define the volume in millilitres
def volumeMl : ℝ := 30

-- Theorem to prove
theorem volumeAsFractionOfLitre : (volumeMl / litreToMl) = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_volumeAsFractionOfLitre_l1432_143250


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l1432_143295

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

def probability_mixed_committee : ℚ :=
  1 - (Nat.choose boys committee_size + Nat.choose girls committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 284 / 295 := by
  sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l1432_143295


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l1432_143284

/-- Systematic sampling interval calculation -/
theorem systematic_sampling_interval
  (total : Nat) (sample : Nat) (h1 : total = 2005) (h2 : sample = 20) :
  (total - 5) / sample = 100 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l1432_143284


namespace NUMINAMATH_CALUDE_driveways_shoveled_is_9_l1432_143273

/-- The number of driveways Jimmy shoveled -/
def driveways_shoveled : ℕ :=
  let candy_bar_price : ℚ := 75/100
  let candy_bar_discount : ℚ := 20/100
  let candy_bars_bought : ℕ := 2
  let lollipop_price : ℚ := 25/100
  let lollipops_bought : ℕ := 4
  let sales_tax : ℚ := 5/100
  let snow_shoveling_fraction : ℚ := 1/6
  let driveway_price : ℚ := 3/2

  let discounted_candy_price := candy_bar_price * (1 - candy_bar_discount)
  let total_candy_cost := (discounted_candy_price * candy_bars_bought)
  let total_lollipop_cost := (lollipop_price * lollipops_bought)
  let subtotal := total_candy_cost + total_lollipop_cost
  let total_with_tax := subtotal * (1 + sales_tax)
  let total_earned := total_with_tax / snow_shoveling_fraction
  let driveways := (total_earned / driveway_price).floor

  driveways.toNat

theorem driveways_shoveled_is_9 :
  driveways_shoveled = 9 := by sorry

end NUMINAMATH_CALUDE_driveways_shoveled_is_9_l1432_143273


namespace NUMINAMATH_CALUDE_greatest_power_of_three_dividing_nine_to_seven_l1432_143282

theorem greatest_power_of_three_dividing_nine_to_seven : 
  (∃ x : ℕ+, (3 : ℕ) ^ (x : ℕ) ∣ 9^7 ∧ ∀ y : ℕ+, (3 : ℕ) ^ (y : ℕ) ∣ 9^7 → y ≤ x) ∧ 
  (∀ x : ℕ+, (3 : ℕ) ^ (x : ℕ) ∣ 9^7 ∧ (∀ y : ℕ+, (3 : ℕ) ^ (y : ℕ) ∣ 9^7 → y ≤ x) → x = 14) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_dividing_nine_to_seven_l1432_143282


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1432_143298

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define the intersection condition
def intersect (a : ℝ) : Prop := ∃ x y : ℝ, circle1 a x y ∧ circle2 x y

-- Define the range of a
def valid_range (a : ℝ) : Prop := (a > -6 ∧ a < -4) ∨ (a > 4 ∧ a < 6)

-- Theorem statement
theorem circle_intersection_range :
  ∀ a : ℝ, intersect a ↔ valid_range a := by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1432_143298


namespace NUMINAMATH_CALUDE_geometric_progression_property_l1432_143230

def geometric_progression (b : ℕ → ℝ) := 
  ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1

theorem geometric_progression_property (b : ℕ → ℝ) 
  (h₁ : geometric_progression b) 
  (h₂ : ∀ n : ℕ, b n > 0) : 
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) ^ (1/6) = (b 3 * b 4) ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_property_l1432_143230


namespace NUMINAMATH_CALUDE_monotonicity_condition_necessary_not_sufficient_l1432_143293

def f (a : ℝ) (x : ℝ) : ℝ := |a - 3*x|

theorem monotonicity_condition (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y) ↔ a ≤ 3 :=
sorry

theorem necessary_not_sufficient :
  (∀ a : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y) → a = 3) ∧
  (∃ a : ℝ, a = 3 ∧ ¬(∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_necessary_not_sufficient_l1432_143293


namespace NUMINAMATH_CALUDE_melody_dogs_count_l1432_143296

def dogs_food_problem (daily_consumption : ℚ) (initial_amount : ℚ) (remaining_amount : ℚ) (days : ℕ) : ℚ :=
  (initial_amount - remaining_amount) / (daily_consumption * days)

theorem melody_dogs_count :
  dogs_food_problem 1 30 9 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_melody_dogs_count_l1432_143296


namespace NUMINAMATH_CALUDE_probability_sum_five_l1432_143246

/-- The probability of obtaining a sum of 5 when rolling two dice of different sizes simultaneously -/
theorem probability_sum_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) : 
  total_outcomes = 36 → favorable_outcomes = 4 → (favorable_outcomes : ℚ) / total_outcomes = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l1432_143246


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_prime_factorization_property_l1432_143231

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def prime_factorization_sum_property (n : ℕ) : Prop :=
  ∃ (factors : List ℕ) (exponents : List ℕ),
    n = (factors.zip exponents).foldl (λ acc (p, e) => acc * p^e) 1 ∧
    factors.all Nat.Prime ∧
    factors.sum = exponents.sum

theorem four_digit_numbers_with_prime_factorization_property :
  {n : ℕ | is_four_digit n ∧ prime_factorization_sum_property n} =
  {1792, 2000, 3125, 3840, 5000, 5760, 6272, 8640, 9600} := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_prime_factorization_property_l1432_143231


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1432_143249

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1432_143249


namespace NUMINAMATH_CALUDE_regular_15gon_symmetry_sum_l1432_143243

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry (in degrees) for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

/-- Theorem: For a regular 15-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 39 -/
theorem regular_15gon_symmetry_sum :
  ∀ (p : RegularPolygon 15),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 39 := by
  sorry

end NUMINAMATH_CALUDE_regular_15gon_symmetry_sum_l1432_143243


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1432_143285

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1432_143285


namespace NUMINAMATH_CALUDE_basil_planter_problem_l1432_143256

theorem basil_planter_problem (total_seeds : ℕ) (num_large_planters : ℕ) (large_planter_capacity : ℕ) (small_planter_capacity : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : num_large_planters = 4)
  (h3 : large_planter_capacity = 20)
  (h4 : small_planter_capacity = 4) :
  (total_seeds - num_large_planters * large_planter_capacity) / small_planter_capacity = 30 := by
  sorry

end NUMINAMATH_CALUDE_basil_planter_problem_l1432_143256


namespace NUMINAMATH_CALUDE_sum_first_ten_enhanced_nice_l1432_143212

def is_prime (n : ℕ) : Prop := sorry

def proper_divisors (n : ℕ) : Set ℕ := sorry

def product_of_set (s : Set ℕ) : ℕ := sorry

def prime_factors (n : ℕ) : List ℕ := sorry

def is_enhanced_nice (n : ℕ) : Prop :=
  (n > 1) ∧
  ((product_of_set (proper_divisors n) = n) ∨
   (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p * q) ∨
   (∃ p : ℕ, is_prime p ∧ n = p^3))

def first_ten_enhanced_nice_under_100 : List ℕ :=
  [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

theorem sum_first_ten_enhanced_nice :
  (List.sum first_ten_enhanced_nice_under_100 = 182) ∧
  (∀ n ∈ first_ten_enhanced_nice_under_100, is_enhanced_nice n) ∧
  (∀ n < 100, is_enhanced_nice n → n ∈ first_ten_enhanced_nice_under_100) :=
sorry

end NUMINAMATH_CALUDE_sum_first_ten_enhanced_nice_l1432_143212


namespace NUMINAMATH_CALUDE_chess_group_players_l1432_143277

/-- The number of players in the chess group -/
def n : ℕ := 10

/-- The total number of games played -/
def total_games : ℕ := 45

/-- Theorem: Given the conditions, the number of players in the chess group is 10 -/
theorem chess_group_players :
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < total_games) ∧
  (∀ (game : ℕ), game < total_games → ∃! (i j : ℕ), i < n ∧ j < n ∧ i ≠ j) ∧
  (n * (n - 1) / 2 = total_games) →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_players_l1432_143277


namespace NUMINAMATH_CALUDE_product_square_theorem_l1432_143227

theorem product_square_theorem : (10 * 0.2 * 3 * 0.1)^2 = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_product_square_theorem_l1432_143227


namespace NUMINAMATH_CALUDE_inscribed_circles_radius_l1432_143286

/-- Given a circle segment with radius R and central angle α, 
    this theorem proves the radius of two equal inscribed circles 
    that touch each other, the arc, and the chord. -/
theorem inscribed_circles_radius 
  (R : ℝ) 
  (α : ℝ) 
  (h_α_pos : 0 < α) 
  (h_α_lt_pi : α < π) : 
  ∃ x : ℝ, 
    x = R * Real.sin (α / 4) ^ 2 ∧ 
    x > 0 ∧
    (∀ y : ℝ, y = R * Real.sin (α / 4) ^ 2 → y = x) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circles_radius_l1432_143286


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1432_143268

theorem regular_hexagon_perimeter (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 / 2) * s^2 = s → 6 * s = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l1432_143268


namespace NUMINAMATH_CALUDE_flour_for_one_loaf_l1432_143294

/-- The amount of flour required for one loaf of bread -/
def flour_per_loaf (total_flour : ℕ) (num_loaves : ℕ) : ℕ := total_flour / num_loaves

/-- Theorem: Given 400g of total flour and the ability to make 2 loaves, 
    prove that one loaf requires 200g of flour -/
theorem flour_for_one_loaf : 
  flour_per_loaf 400 2 = 200 := by
sorry

end NUMINAMATH_CALUDE_flour_for_one_loaf_l1432_143294


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1432_143244

theorem sum_of_roots_quadratic (x : ℝ) : 
  (∃ r1 r2 : ℝ, r1 + r2 = 5 ∧ x^2 - 5*x + 6 = (x - r1) * (x - r2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1432_143244


namespace NUMINAMATH_CALUDE_seven_books_arrangement_l1432_143281

/-- The number of distinct arrangements of books on a shelf -/
def book_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2)

/-- Theorem stating the number of distinct arrangements for the given book configuration -/
theorem seven_books_arrangement :
  book_arrangements 7 3 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_seven_books_arrangement_l1432_143281


namespace NUMINAMATH_CALUDE_kids_ticket_price_l1432_143253

/-- Proves that the price of a kid's ticket is $12 given the specified conditions --/
theorem kids_ticket_price (total_people : ℕ) (adult_price : ℕ) (total_sales : ℕ) (num_kids : ℕ) :
  total_people = 254 →
  adult_price = 28 →
  total_sales = 3864 →
  num_kids = 203 →
  ∃ (kids_price : ℕ), kids_price = 12 ∧ 
    total_sales = (total_people - num_kids) * adult_price + num_kids * kids_price :=
by sorry


end NUMINAMATH_CALUDE_kids_ticket_price_l1432_143253


namespace NUMINAMATH_CALUDE_height_comparison_l1432_143276

theorem height_comparison (ashis_height babji_height : ℝ) 
  (h : ashis_height = babji_height * 1.25) : 
  (ashis_height - babji_height) / ashis_height = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l1432_143276


namespace NUMINAMATH_CALUDE_ribbon_cutting_l1432_143216

-- Define the lengths of the two ribbons
def ribbon1_length : ℕ := 28
def ribbon2_length : ℕ := 16

-- Define the function to calculate the maximum length of shorter ribbons
def max_short_ribbon_length (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the function to calculate the total number of shorter ribbons
def total_short_ribbons (a b c : ℕ) : ℕ := (a + b) / c

-- Theorem statement
theorem ribbon_cutting :
  (max_short_ribbon_length ribbon1_length ribbon2_length = 4) ∧
  (total_short_ribbons ribbon1_length ribbon2_length (max_short_ribbon_length ribbon1_length ribbon2_length) = 11) :=
by sorry

end NUMINAMATH_CALUDE_ribbon_cutting_l1432_143216


namespace NUMINAMATH_CALUDE_henrys_score_l1432_143222

theorem henrys_score (june patty josh henry : ℕ) : 
  june = 97 → patty = 85 → josh = 100 →
  (june + patty + josh + henry) / 4 = 94 →
  henry = 94 := by
sorry

end NUMINAMATH_CALUDE_henrys_score_l1432_143222


namespace NUMINAMATH_CALUDE_sum_of_x_intercepts_is_14_l1432_143228

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Theorem statement
theorem sum_of_x_intercepts_is_14 :
  ∃ a b : ℝ, 
    transformed_parabola a = 0 ∧ 
    transformed_parabola b = 0 ∧ 
    a + b = 14 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_intercepts_is_14_l1432_143228


namespace NUMINAMATH_CALUDE_y_value_proof_l1432_143269

theorem y_value_proof : ∀ y : ℚ, (1/4 - 1/5 = 4/y) → y = 80 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1432_143269


namespace NUMINAMATH_CALUDE_range_of_M_l1432_143213

theorem range_of_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  let M := 1 / (1 + a) + 1 / (1 + b)
  1 < M ∧ M < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_M_l1432_143213


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_two_l1432_143252

/-- The opposite of a real number -/
def opposite (a : ℝ) : ℝ := -a

/-- The property that defines the opposite of a number -/
theorem opposite_def (a : ℝ) : a + opposite a = 0 := by sorry

/-- Proof that the opposite of -2 is 2 -/
theorem opposite_of_neg_two : opposite (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_two_l1432_143252


namespace NUMINAMATH_CALUDE_range_of_a_l1432_143257

open Set

theorem range_of_a (p : ∀ x ∈ Icc 1 2, x^2 - a ≥ 0) 
                   (q : ∃ x₀ : ℝ, x₀ + 2*a*x₀ + 2 - a = 0) : 
  a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1432_143257


namespace NUMINAMATH_CALUDE_gas_supply_equilibrium_l1432_143254

/-- The distance between points A and B in kilometers -/
def total_distance : ℝ := 500

/-- The amount of gas extracted from reservoir A in cubic meters per minute -/
def gas_from_A : ℝ := 10000

/-- The rate of gas leakage in cubic meters per kilometer -/
def leakage_rate : ℝ := 4

/-- The distance between point A and city C in kilometers -/
def distance_AC : ℝ := 100

theorem gas_supply_equilibrium :
  let gas_to_C_from_A := gas_from_A - leakage_rate * distance_AC
  let gas_to_C_from_B := (gas_from_A * 1.12) - leakage_rate * (total_distance - distance_AC)
  gas_to_C_from_A = gas_to_C_from_B :=
by sorry

end NUMINAMATH_CALUDE_gas_supply_equilibrium_l1432_143254


namespace NUMINAMATH_CALUDE_tree_growth_rate_l1432_143280

/-- Proves that a tree growing from 52 feet to 92 feet in 8 years has an annual growth rate of 5 feet --/
theorem tree_growth_rate (initial_height : ℝ) (final_height : ℝ) (years : ℕ) 
  (h1 : initial_height = 52)
  (h2 : final_height = 92)
  (h3 : years = 8) :
  (final_height - initial_height) / years = 5 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l1432_143280


namespace NUMINAMATH_CALUDE_math_competition_probability_l1432_143264

/-- The number of students in the math competition team -/
def num_students : ℕ := 4

/-- The number of comprehensive questions -/
def num_questions : ℕ := 4

/-- The probability that each student solves a different question -/
def prob_different_questions : ℚ := 3/32

theorem math_competition_probability :
  (num_students.factorial : ℚ) / (num_students ^ num_students : ℕ) = prob_different_questions :=
sorry

end NUMINAMATH_CALUDE_math_competition_probability_l1432_143264


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1432_143292

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem tangent_slope_at_one :
  (deriv f) 1 = 1 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1432_143292


namespace NUMINAMATH_CALUDE_equation_solution_l1432_143221

theorem equation_solution : 
  ∃! x : ℝ, |Real.sqrt ((x - 2)^2) - 1| = x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1432_143221


namespace NUMINAMATH_CALUDE_gracies_height_l1432_143263

/-- Given the heights of Griffin, Grayson, and Gracie, prove Gracie's height -/
theorem gracies_height 
  (griffin_height : ℕ) 
  (grayson_height : ℕ) 
  (gracie_height : ℕ)
  (h1 : griffin_height = 61)
  (h2 : grayson_height = griffin_height + 2)
  (h3 : gracie_height = grayson_height - 7) : 
  gracie_height = 56 := by sorry

end NUMINAMATH_CALUDE_gracies_height_l1432_143263


namespace NUMINAMATH_CALUDE_female_democrats_count_l1432_143240

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 660 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1432_143240


namespace NUMINAMATH_CALUDE_isosceles_triangle_unique_range_l1432_143235

theorem isosceles_triangle_unique_range (a : ℝ) :
  (∃ (x y : ℝ), x^2 - 6*x + a = 0 ∧ y^2 - 6*y + a = 0 ∧ 
   x ≠ y ∧ 
   (x < y → 2*x ≤ y) ∧
   (y < x → 2*y ≤ x)) ↔
  (0 < a ∧ a ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_unique_range_l1432_143235


namespace NUMINAMATH_CALUDE_divide_money_l1432_143225

theorem divide_money (total : ℝ) (a b c : ℝ) : 
  total = 364 →
  a = (1/2) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 208 := by
sorry

end NUMINAMATH_CALUDE_divide_money_l1432_143225


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1432_143236

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + a| ≥ 3) ↔ (a ≤ -5 ∨ a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1432_143236


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5_l1432_143201

theorem unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5 :
  ∃! n : ℕ+, (15 ∣ n) ∧ (28 < (n : ℝ).sqrt) ∧ ((n : ℝ).sqrt < 28.5) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_15_with_sqrt_between_28_and_28_5_l1432_143201


namespace NUMINAMATH_CALUDE_adult_meals_sold_l1432_143203

theorem adult_meals_sold (kids_meals : ℕ) (adult_meals : ℕ) : 
  (10 : ℚ) / 7 = kids_meals / adult_meals →
  kids_meals = 70 →
  adult_meals = 49 := by
sorry

end NUMINAMATH_CALUDE_adult_meals_sold_l1432_143203


namespace NUMINAMATH_CALUDE_no_perfect_square_solution_l1432_143272

theorem no_perfect_square_solution :
  ¬ ∃ (x y z t : ℕ+), 
    (x * y - z * t = x + y) ∧
    (x + y = z + t) ∧
    (∃ (a b : ℕ+), x * y = a * a ∧ z * t = b * b) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_solution_l1432_143272


namespace NUMINAMATH_CALUDE_layla_phone_probability_l1432_143208

def first_segment_choices : ℕ := 3
def last_segment_digits : ℕ := 4

theorem layla_phone_probability :
  (1 : ℚ) / (first_segment_choices * Nat.factorial last_segment_digits) = 1 / 72 :=
by sorry

end NUMINAMATH_CALUDE_layla_phone_probability_l1432_143208


namespace NUMINAMATH_CALUDE_lucas_book_purchase_l1432_143278

theorem lucas_book_purchase (total_money : ℚ) (total_books : ℕ) (book_price : ℚ) 
    (h1 : total_money > 0)
    (h2 : total_books > 0)
    (h3 : book_price > 0)
    (h4 : (1 / 4 : ℚ) * total_money = (1 / 2 : ℚ) * total_books * book_price) : 
  total_money - (total_books * book_price) = (1 / 2 : ℚ) * total_money := by
sorry

end NUMINAMATH_CALUDE_lucas_book_purchase_l1432_143278


namespace NUMINAMATH_CALUDE_complex_sum_zero_l1432_143218

theorem complex_sum_zero : 
  let z : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  1 + z + z^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l1432_143218
