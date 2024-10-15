import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_inequality_holds_for_interval_iff_m_in_range_l2301_230197

-- Part 1
theorem solution_set_nonempty_iff_m_in_range (m : ℝ) :
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 :=
sorry

-- Part 2
theorem inequality_holds_for_interval_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → m * x^2 - m * x < -m + 2) ↔ m < 2/7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_inequality_holds_for_interval_iff_m_in_range_l2301_230197


namespace NUMINAMATH_CALUDE_hash_seven_three_l2301_230176

-- Define the # operation
def hash (a b : ℤ) : ℚ := 2 * a + a / b + 3

-- Theorem statement
theorem hash_seven_three : hash 7 3 = 19 + 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hash_seven_three_l2301_230176


namespace NUMINAMATH_CALUDE_line_intersects_plane_not_perpendicular_implies_not_parallel_l2301_230163

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define the relationships
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

def perpendicular (l : Line3D) (α : Plane3D) : Prop :=
  sorry

def plane_through_line (l : Line3D) : Plane3D :=
  sorry

def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

-- State the theorem
theorem line_intersects_plane_not_perpendicular_implies_not_parallel 
  (l : Line3D) (α : Plane3D) :
  intersects l α ∧ ¬perpendicular l α →
  ∀ p : Plane3D, p = plane_through_line l → ¬parallel_planes p α :=
sorry

end NUMINAMATH_CALUDE_line_intersects_plane_not_perpendicular_implies_not_parallel_l2301_230163


namespace NUMINAMATH_CALUDE_number_333_less_than_600_l2301_230120

theorem number_333_less_than_600 : 600 - 333 = 267 := by sorry

end NUMINAMATH_CALUDE_number_333_less_than_600_l2301_230120


namespace NUMINAMATH_CALUDE_largest_valid_number_l2301_230113

def is_valid_number (a b c d e : Nat) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
  d > e ∧
  c > d + e ∧
  b > c + d + e ∧
  a > b + c + d + e

def number_value (a b c d e : Nat) : Nat :=
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem largest_valid_number :
  ∀ a b c d e : Nat,
    is_valid_number a b c d e →
    number_value a b c d e ≤ 95210 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2301_230113


namespace NUMINAMATH_CALUDE_properties_of_negative_three_l2301_230174

theorem properties_of_negative_three :
  (- (-3) = 3) ∧
  (((-3)⁻¹ : ℚ) = -1/3) ∧
  (abs (-3) = 3) := by
sorry

end NUMINAMATH_CALUDE_properties_of_negative_three_l2301_230174


namespace NUMINAMATH_CALUDE_initial_group_size_l2301_230167

theorem initial_group_size (initial_avg : ℝ) (new_people : ℕ) (new_avg : ℝ) (final_avg : ℝ) : 
  initial_avg = 16 → 
  new_people = 20 → 
  new_avg = 15 → 
  final_avg = 15.5 → 
  ∃ N : ℕ, N = 20 ∧ 
    (N * initial_avg + new_people * new_avg) / (N + new_people) = final_avg :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l2301_230167


namespace NUMINAMATH_CALUDE_num_bottles_is_four_l2301_230156

-- Define the weight of a bag of chips
def bag_weight : ℕ := 400

-- Define the weight difference between a bag of chips and a bottle of juice
def weight_difference : ℕ := 350

-- Define the total weight of 5 bags of chips and some bottles of juice
def total_weight : ℕ := 2200

-- Define the number of bags of chips
def num_bags : ℕ := 5

-- Define the weight of a bottle of juice
def bottle_weight : ℕ := bag_weight - weight_difference

-- Define the function to calculate the number of bottles
def num_bottles : ℕ :=
  (total_weight - num_bags * bag_weight) / bottle_weight

-- Theorem statement
theorem num_bottles_is_four :
  num_bottles = 4 :=
sorry

end NUMINAMATH_CALUDE_num_bottles_is_four_l2301_230156


namespace NUMINAMATH_CALUDE_journey_distance_l2301_230103

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 20)
  (h2 : speed1 = 10)
  (h3 : speed2 = 15) : 
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 240 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2301_230103


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2301_230140

theorem cubic_equation_solution (t s : ℝ) : t = 8 * s^3 ∧ t = 64 → s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2301_230140


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l2301_230130

theorem shopkeeper_profit (C : ℝ) (h : C > 0) : 
  ∃ N : ℝ, N > 0 ∧ 12 * C + 0.2 * (N * C) = N * C ∧ N = 15 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l2301_230130


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2301_230166

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x + 1 = (x - 4) * (x - 2)^3 * (21 / (8 * (x - 4)) + 19 / (4 * (x - 2)) + (-11) / (2 * (x - 2)^3)) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2301_230166


namespace NUMINAMATH_CALUDE_increased_speed_calculation_l2301_230170

/-- Proves that given a distance of 100 km, a usual speed of 20 km/hr,
    and a travel time reduction of 1 hour with increased speed,
    the increased speed is 25 km/hr. -/
theorem increased_speed_calculation (distance : ℝ) (usual_speed : ℝ) (time_reduction : ℝ) :
  distance = 100 ∧ usual_speed = 20 ∧ time_reduction = 1 →
  (distance / (distance / usual_speed - time_reduction)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_increased_speed_calculation_l2301_230170


namespace NUMINAMATH_CALUDE_ticket_distribution_theorem_l2301_230182

/-- The number of ways to distribute 3 different tickets to 3 students out of a group of 10 -/
def ticket_distribution_ways : ℕ := 10 * 9 * 8

/-- Theorem: The number of ways to distribute 3 different tickets to 3 students out of a group of 10 is 720 -/
theorem ticket_distribution_theorem : ticket_distribution_ways = 720 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_theorem_l2301_230182


namespace NUMINAMATH_CALUDE_division_chain_l2301_230112

theorem division_chain : (180 / 6) / 3 = 10 := by sorry

end NUMINAMATH_CALUDE_division_chain_l2301_230112


namespace NUMINAMATH_CALUDE_perimeter_difference_l2301_230138

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Represents Figure 1: a 3x6 rectangle --/
def figure1 : ℕ × ℕ := (3, 6)

/-- Represents Figure 2: a 2x7 rectangle with an additional square --/
def figure2 : ℕ × ℕ := (2, 7)

/-- The additional perimeter contributed by the extra square in Figure 2 --/
def extraSquarePerimeter : ℕ := 3

theorem perimeter_difference :
  rectanglePerimeter figure2.1 figure2.2 + extraSquarePerimeter -
  rectanglePerimeter figure1.1 figure1.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l2301_230138


namespace NUMINAMATH_CALUDE_point_d_theorem_l2301_230151

-- Define the triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

-- Define point D
def PointD (x y : ℝ) : Prod ℝ ℝ := (x, y)

-- Define the condition for point D
def SatisfiesCondition (t : RightTriangle) (d : Prod ℝ ℝ) : Prop :=
  let (x, y) := d
  let ad := Real.sqrt ((x - t.a)^2 + y^2)
  let bc := t.a
  let ac := Real.sqrt (t.a^2 + t.b^2)
  let bd := Real.sqrt (x^2 + (y - t.b)^2)
  let cd := Real.sqrt (x^2 + y^2)
  ad * bc = ac * bd ∧ ac * bd = (Real.sqrt (t.a^2 + t.b^2) * cd) / Real.sqrt 2

-- Theorem statement
theorem point_d_theorem (t : RightTriangle) :
  ∀ x y : ℝ, SatisfiesCondition t (PointD x y) ↔ 
  (x = t.a * t.b / (t.a + t.b) ∧ y = t.a * t.b / (t.a + t.b)) ∨
  (x = t.a * t.b / (t.a - t.b) ∧ y = t.a * t.b / (t.a - t.b)) :=
sorry

end NUMINAMATH_CALUDE_point_d_theorem_l2301_230151


namespace NUMINAMATH_CALUDE_min_value_theorem_l2301_230181

-- Define the condition function
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x * x ≥ 0

-- State the theorem
theorem min_value_theorem (a b : ℝ) (h : condition a b) : 
  ∃ (min : ℝ), min = 1 ∧ ∀ (c : ℝ), b / a ≥ c := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2301_230181


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l2301_230141

theorem three_digit_sum_property (a b c d e f : Nat) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0) →
  (100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000) →
  (a + b + c + d + e + f = 28) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l2301_230141


namespace NUMINAMATH_CALUDE_range_of_a_l2301_230168

def A : Set ℝ := {x | x^2 + 4*x = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2301_230168


namespace NUMINAMATH_CALUDE_m_minus_n_equals_eighteen_l2301_230135

theorem m_minus_n_equals_eighteen :
  ∀ m n : ℤ,
  (∀ k : ℤ, k < 0 → k ≤ -m) →  -- m's opposite is the largest negative integer
  (-n = 17) →                  -- n's opposite is 17
  m - n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_eighteen_l2301_230135


namespace NUMINAMATH_CALUDE_figurine_cost_l2301_230145

/-- The cost of a single figurine given Annie's purchase details -/
theorem figurine_cost (tv_count : ℕ) (tv_price : ℕ) (figurine_count : ℕ) (total_spent : ℕ) : 
  tv_count = 5 → 
  tv_price = 50 → 
  figurine_count = 10 → 
  total_spent = 260 → 
  (total_spent - tv_count * tv_price) / figurine_count = 1 :=
by
  sorry

#check figurine_cost

end NUMINAMATH_CALUDE_figurine_cost_l2301_230145


namespace NUMINAMATH_CALUDE_semicircles_in_rectangle_l2301_230185

theorem semicircles_in_rectangle (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₁ > r₂) :
  let height := 2 * Real.sqrt (r₁ * r₂)
  let rectangle_area := height * (r₁ + r₂)
  let semicircles_area := π / 2 * (r₁^2 + r₂^2)
  semicircles_area / rectangle_area = (π / 2 * (r₁^2 + r₂^2)) / (2 * Real.sqrt (r₁ * r₂) * (r₁ + r₂)) :=
by sorry

end NUMINAMATH_CALUDE_semicircles_in_rectangle_l2301_230185


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2301_230164

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0

theorem smallest_number_divisible : 
  (∀ m : ℕ, m < 6303 → ¬(is_divisible_by_all m)) ∧ 
  is_divisible_by_all 6303 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2301_230164


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2301_230107

theorem minimum_value_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + 1/c^2 ≥ Real.sqrt 3 + 2 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ c₀ ≠ 0 ∧
    a₀^2 + b₀^2 + c₀^2 + 1/a₀^2 + b₀/a₀ + 1/c₀^2 = Real.sqrt 3 + 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2301_230107


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l2301_230144

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_unique :
  ∀ a b c : ℝ,
  (f a b c (-1) = 0) →
  (∀ x : ℝ, x ≤ f a b c x) →
  (∀ x : ℝ, f a b c x ≤ (x^2 + 1) / 2) →
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l2301_230144


namespace NUMINAMATH_CALUDE_division_problem_l2301_230161

theorem division_problem (smaller larger : ℕ) : 
  larger - smaller = 1395 →
  larger = 1656 →
  larger % smaller = 15 →
  larger / smaller = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2301_230161


namespace NUMINAMATH_CALUDE_complex_real_condition_l2301_230127

theorem complex_real_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  ((2 + Complex.I) * (1 - m * Complex.I)).im = 0 →
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2301_230127


namespace NUMINAMATH_CALUDE_action_figures_added_l2301_230136

theorem action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : 
  initial = 15 → removed = 7 → final = 10 → initial - removed + (final - (initial - removed)) = 2 := by
sorry

end NUMINAMATH_CALUDE_action_figures_added_l2301_230136


namespace NUMINAMATH_CALUDE_linear_decreasing_slope_l2301_230134

/-- For a linear function y = (m-2)x + 1, if y is decreasing as x increases, then m < 2. -/
theorem linear_decreasing_slope (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((m - 2) * x₁ + 1) > ((m - 2) * x₂ + 1)) →
  m < 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_decreasing_slope_l2301_230134


namespace NUMINAMATH_CALUDE_parallel_condition_theorem_l2301_230172

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition_theorem 
  (a b : Line) (α : Plane) 
  (h_different : a ≠ b) 
  (h_contained : contained_in b α) :
  (∀ x y : Line, ∀ p : Plane, 
    contained_in y p → 
    parallel_lines x y → 
    parallel_line_plane x p) ∧
  (∃ x y : Line, ∃ p : Plane,
    contained_in y p ∧ 
    parallel_line_plane x p ∧ 
    ¬parallel_lines x y) →
  (parallel_line_plane a α → parallel_lines a b) ∧
  ¬(parallel_lines a b → parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_theorem_l2301_230172


namespace NUMINAMATH_CALUDE_root_equation_problem_l2301_230122

theorem root_equation_problem (p q : ℝ) 
  (h1 : ∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    (∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) ∧
    (∀ x : ℝ, x ≠ -4))
  (h2 : ∃! (s1 s2 : ℝ), s1 ≠ s2 ∧
    (∀ x : ℝ, (x + 2*p) * (x + 4) * (x + 9) = 0 ↔ x = s1 ∨ x = s2) ∧
    (∀ x : ℝ, x ≠ -q ∧ x ≠ -15)) :
  100 * p + q = -191 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l2301_230122


namespace NUMINAMATH_CALUDE_sector_radius_l2301_230149

theorem sector_radius (r : ℝ) (h1 : r > 0) : 
  (r = r) →  -- radius equals arc length
  ((3 * r) / ((1/2) * r^2) = 2) →  -- ratio of perimeter to area is 2
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_sector_radius_l2301_230149


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l2301_230132

theorem sunshine_orchard_pumpkins (moonglow_pumpkins : ℕ) (sunshine_pumpkins : ℕ) : 
  moonglow_pumpkins = 14 →
  sunshine_pumpkins = 3 * moonglow_pumpkins + 12 →
  sunshine_pumpkins = 54 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l2301_230132


namespace NUMINAMATH_CALUDE_average_cost_per_stadium_l2301_230152

def number_of_stadiums : ℕ := 30
def savings_per_year : ℕ := 1500
def years_to_accomplish : ℕ := 18

theorem average_cost_per_stadium :
  (savings_per_year * years_to_accomplish) / number_of_stadiums = 900 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_per_stadium_l2301_230152


namespace NUMINAMATH_CALUDE_circle_motion_speeds_l2301_230110

/-- Represents the state of two circles moving towards the vertex of a right angle -/
structure CircleMotion where
  r1 : ℝ  -- radius of first circle
  r2 : ℝ  -- radius of second circle
  d1 : ℝ  -- initial distance of first circle from vertex
  d2 : ℝ  -- initial distance of second circle from vertex
  t_external : ℝ  -- time when circles touch externally
  t_internal : ℝ  -- time when circles touch internally

/-- Represents a pair of speeds for the two circles -/
structure SpeedPair where
  s1 : ℝ  -- speed of first circle
  s2 : ℝ  -- speed of second circle

/-- Checks if a given speed pair satisfies the conditions for the circle motion -/
def satisfiesConditions (cm : CircleMotion) (sp : SpeedPair) : Prop :=
  let d1_external := cm.d1 - sp.s1 * cm.t_external
  let d2_external := cm.d2 - sp.s2 * cm.t_external
  let d1_internal := cm.d1 - sp.s1 * cm.t_internal
  let d2_internal := cm.d2 - sp.s2 * cm.t_internal
  d1_external^2 + d2_external^2 = (cm.r1 + cm.r2)^2 ∧
  d1_internal^2 + d2_internal^2 = (cm.r1 - cm.r2)^2

/-- The main theorem stating that given the conditions, only two speed pairs satisfy the motion -/
theorem circle_motion_speeds (cm : CircleMotion)
  (h_r1 : cm.r1 = 9)
  (h_r2 : cm.r2 = 4)
  (h_d1 : cm.d1 = 48)
  (h_d2 : cm.d2 = 14)
  (h_t_external : cm.t_external = 9)
  (h_t_internal : cm.t_internal = 11) :
  ∃ (sp1 sp2 : SpeedPair),
    satisfiesConditions cm sp1 ∧
    satisfiesConditions cm sp2 ∧
    ((sp1.s1 = 4 ∧ sp1.s2 = 1) ∨ (sp1.s1 = 3.9104 ∧ sp1.s2 = 1.3072)) ∧
    ((sp2.s1 = 4 ∧ sp2.s2 = 1) ∨ (sp2.s1 = 3.9104 ∧ sp2.s2 = 1.3072)) ∧
    sp1 ≠ sp2 ∧
    ∀ (sp : SpeedPair), satisfiesConditions cm sp → (sp = sp1 ∨ sp = sp2) := by
  sorry

end NUMINAMATH_CALUDE_circle_motion_speeds_l2301_230110


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l2301_230169

/-- Represents a rectangular piece of land -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: For a rectangle with area 215.6 m² and width 14 m, the length is 15.4 m -/
theorem rectangle_length_calculation (r : Rectangle) 
  (h_area : area r = 215.6) 
  (h_width : r.width = 14) : 
  r.length = 15.4 := by
  sorry

#check rectangle_length_calculation

end NUMINAMATH_CALUDE_rectangle_length_calculation_l2301_230169


namespace NUMINAMATH_CALUDE_min_four_digit_with_different_remainders_l2301_230121

theorem min_four_digit_with_different_remainders :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n ≤ 9999 ∧
    (∀ i j, i ≠ j → n % (i + 2) ≠ n % (j + 2)) ∧
    (∀ i, n % (i + 2) ≠ 0) ∧
    (∀ m, 1000 ≤ m ∧ m < n →
      ¬(∀ i j, i ≠ j → m % (i + 2) ≠ m % (j + 2)) ∨
      ¬(∀ i, m % (i + 2) ≠ 0)) ∧
    n = 1259 :=
by sorry

end NUMINAMATH_CALUDE_min_four_digit_with_different_remainders_l2301_230121


namespace NUMINAMATH_CALUDE_hiking_club_boys_count_l2301_230188

theorem hiking_club_boys_count :
  ∀ (total_members attendance boys girls : ℕ),
  total_members = 32 →
  attendance = 22 →
  boys + girls = total_members →
  boys + (2 * girls) / 3 = attendance →
  boys = 2 := by
  sorry

end NUMINAMATH_CALUDE_hiking_club_boys_count_l2301_230188


namespace NUMINAMATH_CALUDE_max_value_F_unique_s_for_H_l2301_230165

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x

def F (x : ℝ) : ℝ := x^2 - Real.log x

def H (s x : ℝ) : ℝ := 
  if x ≥ s then x / (2 * Real.exp 1) else f x

theorem max_value_F :
  ∃ (x : ℝ), x ∈ Set.Icc (1/2) 2 ∧ 
  ∀ (y : ℝ), y ∈ Set.Icc (1/2) 2 → F y ≤ F x ∧
  F x = 4 - Real.log 2 :=
sorry

theorem unique_s_for_H :
  ∃! (s : ℝ), s > 0 ∧ 
  (∀ (k : ℝ), ∃ (x : ℝ), H s x = k) ∧
  s = Real.sqrt (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_F_unique_s_for_H_l2301_230165


namespace NUMINAMATH_CALUDE_class_score_theorem_l2301_230187

def average_score : ℕ := 90

def is_valid_total_score (total : ℕ) : Prop :=
  1000 ≤ total ∧ total ≤ 9999 ∧ total % 10 = 0

def construct_number (A B : ℕ) : ℕ :=
  A * 1000 + 800 + 60 + B

theorem class_score_theorem (A B : ℕ) :
  A < 10 → B < 10 →
  is_valid_total_score (construct_number A B) →
  (construct_number A B) / (construct_number A B / average_score) = average_score →
  A = 4 ∧ B = 0 := by
sorry

end NUMINAMATH_CALUDE_class_score_theorem_l2301_230187


namespace NUMINAMATH_CALUDE_intersection_M_N_l2301_230137

-- Define set M
def M : Set ℝ := {x | x^2 ≥ x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3^x + 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2301_230137


namespace NUMINAMATH_CALUDE_pen_to_book_ratio_l2301_230155

theorem pen_to_book_ratio (pencils pens books : ℕ) 
  (h1 : pencils = 140)
  (h2 : books = 30)
  (h3 : pencils * 4 = pens * 14)
  (h4 : pencils * 3 = books * 14) : 
  4 * books = 3 * pens := by
  sorry

end NUMINAMATH_CALUDE_pen_to_book_ratio_l2301_230155


namespace NUMINAMATH_CALUDE_birthday_cake_theorem_l2301_230196

/-- Represents a rectangular cake with dimensions length, width, and height -/
structure Cake where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of unit cubes with exactly two iced sides in a cake -/
def count_two_sided_iced_pieces (c : Cake) : ℕ :=
  sorry

/-- The main theorem stating that a 5 × 3 × 4 cake with five faces iced
    has 25 pieces with exactly two iced sides -/
theorem birthday_cake_theorem :
  let cake : Cake := { length := 5, width := 3, height := 4 }
  count_two_sided_iced_pieces cake = 25 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cake_theorem_l2301_230196


namespace NUMINAMATH_CALUDE_min_value_theorem_l2301_230102

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b = 1 →
  (∀ x y : ℝ, 0 < x → x < 2 → y = 1 + Real.sin (π * x) → a * x + b * y = 1) →
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2301_230102


namespace NUMINAMATH_CALUDE_min_side_diff_is_one_l2301_230126

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  AB : ℕ
  BC : ℕ
  AC : ℕ

/-- The perimeter of the triangle -/
def Triangle.perimeter (t : Triangle) : ℕ := t.AB + t.BC + t.AC

/-- The difference between the longest and second longest sides -/
def Triangle.sideDiff (t : Triangle) : ℕ := t.AC - t.BC

/-- Predicate for a valid triangle satisfying the given conditions -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.AB ≤ t.BC ∧ t.BC < t.AC ∧ t.perimeter = 2020

theorem min_side_diff_is_one :
  ∃ (t : Triangle), t.isValid ∧
    ∀ (t' : Triangle), t'.isValid → t.sideDiff ≤ t'.sideDiff :=
by sorry

end NUMINAMATH_CALUDE_min_side_diff_is_one_l2301_230126


namespace NUMINAMATH_CALUDE_multiply_by_conjugate_equals_one_l2301_230189

theorem multiply_by_conjugate_equals_one :
  let x : ℝ := (3 - Real.sqrt 5) / 4
  x * (3 + Real.sqrt 5) = 1 := by
sorry

end NUMINAMATH_CALUDE_multiply_by_conjugate_equals_one_l2301_230189


namespace NUMINAMATH_CALUDE_darnel_sprint_jog_difference_l2301_230193

theorem darnel_sprint_jog_difference : 
  let sprint_distance : ℝ := 0.88
  let jog_distance : ℝ := 0.75
  sprint_distance - jog_distance = 0.13 := by sorry

end NUMINAMATH_CALUDE_darnel_sprint_jog_difference_l2301_230193


namespace NUMINAMATH_CALUDE_degree_not_determined_by_characteristic_l2301_230180

/-- A type representing a characteristic of a polynomial -/
def PolynomialCharacteristic := Type

/-- A function that computes a characteristic of a polynomial -/
noncomputable def compute_characteristic (P : Polynomial ℝ) : PolynomialCharacteristic :=
  sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from its characteristic -/
theorem degree_not_determined_by_characteristic :
  ∃ (P1 P2 : Polynomial ℝ), 
    P1.degree ≠ P2.degree ∧ 
    compute_characteristic P1 = compute_characteristic P2 := by
  sorry

end NUMINAMATH_CALUDE_degree_not_determined_by_characteristic_l2301_230180


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2301_230194

theorem fixed_point_on_line (m : ℝ) : 
  m * (-2) - 1 + 2 * m + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2301_230194


namespace NUMINAMATH_CALUDE_smallest_valid_student_count_l2301_230157

def is_valid_student_count (n : ℕ) : Prop :=
  20 ∣ n ∧ 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card ≥ 15 ∧
  ¬(10 ∣ n) ∧ ¬(25 ∣ n) ∧ ¬(50 ∣ n)

theorem smallest_valid_student_count :
  is_valid_student_count 120 ∧ 
  ∀ m < 120, ¬is_valid_student_count m :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_student_count_l2301_230157


namespace NUMINAMATH_CALUDE_camp_distribution_correct_l2301_230123

/-- Represents a summer camp with three sub-camps -/
structure SummerCamp where
  totalStudents : Nat
  sampleSize : Nat
  firstDrawn : Nat
  campIEnd : Nat
  campIIEnd : Nat

/-- Calculates the number of students drawn from each camp -/
def campDistribution (camp : SummerCamp) : (Nat × Nat × Nat) :=
  sorry

/-- Theorem stating the correct distribution of sampled students across camps -/
theorem camp_distribution_correct (camp : SummerCamp) 
  (h1 : camp.totalStudents = 720)
  (h2 : camp.sampleSize = 60)
  (h3 : camp.firstDrawn = 4)
  (h4 : camp.campIEnd = 360)
  (h5 : camp.campIIEnd = 640) :
  campDistribution camp = (30, 24, 6) := by
  sorry

end NUMINAMATH_CALUDE_camp_distribution_correct_l2301_230123


namespace NUMINAMATH_CALUDE_tangent_product_approximation_l2301_230114

theorem tangent_product_approximation :
  let A : Real := 30 * π / 180
  let B : Real := 40 * π / 180
  ∃ ε > 0, |(1 + Real.tan A) * (1 + Real.tan B) - 2.9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_product_approximation_l2301_230114


namespace NUMINAMATH_CALUDE_decreasing_condition_direct_proportion_condition_l2301_230158

-- Define the linear function
def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (m^2 - 4)

-- Theorem for part 1
theorem decreasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function m x₁ > linear_function m x₂) ↔ m < 2 :=
sorry

-- Theorem for part 2
theorem direct_proportion_condition (m : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, linear_function m x = k * x) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_decreasing_condition_direct_proportion_condition_l2301_230158


namespace NUMINAMATH_CALUDE_find_other_number_l2301_230148

theorem find_other_number (A B : ℕ) (h1 : A = 24) (h2 : Nat.gcd A B = 15) (h3 : Nat.lcm A B = 312) : B = 195 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2301_230148


namespace NUMINAMATH_CALUDE_equation_solution_l2301_230198

theorem equation_solution :
  ∃! x : ℚ, (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 :=
by
  use (-2/3)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2301_230198


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2301_230131

/-- Represents a shape created by joining seven unit cubes -/
structure SevenCubeShape where
  /-- The volume of the shape in cubic units -/
  volume : ℕ
  /-- The surface area of the shape in square units -/
  surface_area : ℕ
  /-- The shape is composed of seven unit cubes -/
  is_seven_cubes : volume = 7
  /-- The surface area is calculated based on the configuration of the seven cubes -/
  surface_area_calc : surface_area = 30

/-- Theorem stating that the ratio of volume to surface area for the SevenCubeShape is 7:30 -/
theorem volume_to_surface_area_ratio (shape : SevenCubeShape) :
  (shape.volume : ℚ) / shape.surface_area = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l2301_230131


namespace NUMINAMATH_CALUDE_expression_evaluation_l2301_230192

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  ((x + 2*y) * (x - 2*y) - (x - y)^2) = -24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2301_230192


namespace NUMINAMATH_CALUDE_expression_equals_49_l2301_230117

theorem expression_equals_49 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_49_l2301_230117


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2301_230191

def concatenate_range (a b : ℕ) : ℕ :=
  sorry

def M : ℕ := concatenate_range 25 87

theorem highest_power_of_three_dividing_M :
  ∃ (k : ℕ), M % 3 = 0 ∧ M % (3^2) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2301_230191


namespace NUMINAMATH_CALUDE_project_time_ratio_l2301_230146

theorem project_time_ratio (kate mark pat : ℕ) : 
  kate + mark + pat = 144 →
  pat = 2 * kate →
  mark = kate + 80 →
  pat / mark = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_project_time_ratio_l2301_230146


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2301_230177

/-- An arithmetic sequence {a_n} with given conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ a 5 = -3 ∧ ∃ n : ℕ, a n = -27

/-- The common difference of the arithmetic sequence -/
def common_difference (a : ℕ → ℤ) : ℤ := (a 5 - a 1) / 4

/-- The theorem stating that n = 17 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_value (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  ∃ n : ℕ, n = 17 ∧ a n = -27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2301_230177


namespace NUMINAMATH_CALUDE_division_of_decimals_l2301_230118

theorem division_of_decimals : (0.045 : ℚ) / (0.009 : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2301_230118


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l2301_230119

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem unique_function_satisfying_conditions :
  (∀ (x : ℝ), x ≠ 0 → f x = x * f (1 / x)) ∧
  (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x ≠ -y → f x + f y = 1 + f (x + y)) ∧
  (∀ (g : ℝ → ℝ), 
    ((∀ (x : ℝ), x ≠ 0 → g x = x * g (1 / x)) ∧
     (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x ≠ -y → g x + g y = 1 + g (x + y)))
    → (∀ (x : ℝ), x ≠ 0 → g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l2301_230119


namespace NUMINAMATH_CALUDE_candy_game_solution_l2301_230195

/-- Represents the game state and rules --/
structure CandyGame where
  totalCandies : Nat
  xiaomingEat : Nat
  xiaomingKeep : Nat
  xiaoliangEat : Nat
  xiaoliangKeep : Nat

/-- Represents the result of the game --/
structure GameResult where
  xiaomingWins : Nat
  xiaoliangWins : Nat
  xiaomingPocket : Nat
  xiaoliangPocket : Nat
  totalEaten : Nat

/-- The theorem to prove --/
theorem candy_game_solution (game : CandyGame)
  (h1 : game.totalCandies = 50)
  (h2 : game.xiaomingEat + game.xiaomingKeep = 5)
  (h3 : game.xiaoliangEat + game.xiaoliangKeep = 5)
  (h4 : game.xiaomingKeep = 1)
  (h5 : game.xiaoliangKeep = 2)
  : ∃ (result : GameResult),
    result.xiaomingWins + result.xiaoliangWins = game.totalCandies / 5 ∧
    result.xiaomingPocket = result.xiaomingWins * game.xiaomingKeep ∧
    result.xiaoliangPocket = result.xiaoliangWins * game.xiaoliangKeep ∧
    result.xiaoliangPocket = 3 * result.xiaomingPocket ∧
    result.totalEaten = result.xiaomingWins * game.xiaomingEat + result.xiaoliangWins * game.xiaoliangEat ∧
    result.totalEaten = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_candy_game_solution_l2301_230195


namespace NUMINAMATH_CALUDE_females_wearing_glasses_l2301_230186

theorem females_wearing_glasses (total_population : ℕ) (male_population : ℕ) (female_glasses_percentage : ℚ) :
  total_population = 5000 →
  male_population = 2000 →
  female_glasses_percentage = 30 / 100 →
  (total_population - male_population) * female_glasses_percentage = 900 := by
  sorry

end NUMINAMATH_CALUDE_females_wearing_glasses_l2301_230186


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2301_230106

/-- Prove that given a journey of 225 km completed in 10 hours, 
    where the first half is traveled at 21 km/hr, 
    the speed for the second half of the journey is approximately 24.23 km/hr. -/
theorem journey_speed_calculation 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 225) 
  (h2 : total_time = 10) 
  (h3 : first_half_speed = 21) : 
  ∃ (second_half_speed : ℝ), 
    (abs (second_half_speed - 24.23) < 0.01) ∧ 
    (total_distance / 2 / first_half_speed + total_distance / 2 / second_half_speed = total_time) :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l2301_230106


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2301_230115

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct intersection points of diagonals in the interior of a regular decagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points n = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2301_230115


namespace NUMINAMATH_CALUDE_stickers_given_to_alex_l2301_230150

theorem stickers_given_to_alex (initial_stickers : ℕ) (stickers_to_lucy : ℕ) (remaining_stickers : ℕ)
  (h1 : initial_stickers = 99)
  (h2 : stickers_to_lucy = 42)
  (h3 : remaining_stickers = 31) :
  initial_stickers - remaining_stickers - stickers_to_lucy = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_stickers_given_to_alex_l2301_230150


namespace NUMINAMATH_CALUDE_items_not_washed_l2301_230154

theorem items_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (pants : ℕ) (jackets : ℕ) (washed : ℕ)
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : pants = 15)
  (h4 : jackets = 8)
  (h5 : washed = 43) :
  short_sleeve + long_sleeve + pants + jackets - washed = 10 := by
  sorry

end NUMINAMATH_CALUDE_items_not_washed_l2301_230154


namespace NUMINAMATH_CALUDE_largest_root_cubic_equation_l2301_230159

theorem largest_root_cubic_equation (a₂ a₁ a₀ : ℝ) 
  (h₂ : |a₂| < 2) (h₁ : |a₁| < 2) (h₀ : |a₀| < 2) :
  ∃ r : ℝ, r > 0 ∧ r^3 + a₂*r^2 + a₁*r + a₀ = 0 ∧
  (∀ x : ℝ, x > 0 ∧ x^3 + a₂*x^2 + a₁*x + a₀ = 0 → x ≤ r) ∧
  (5/2 < r ∧ r < 3) := by
  sorry

end NUMINAMATH_CALUDE_largest_root_cubic_equation_l2301_230159


namespace NUMINAMATH_CALUDE_racing_cars_lcm_l2301_230160

theorem racing_cars_lcm (lap_time_A lap_time_B : ℕ) 
  (h1 : lap_time_A = 28) 
  (h2 : lap_time_B = 24) : 
  Nat.lcm lap_time_A lap_time_B = 168 := by
  sorry

end NUMINAMATH_CALUDE_racing_cars_lcm_l2301_230160


namespace NUMINAMATH_CALUDE_rational_reciprocal_power_smallest_positive_integer_main_result_l2301_230190

theorem rational_reciprocal_power (a : ℚ) : 
  (a ≠ 0 ∧ a = a⁻¹) → a^2014 = (1 : ℚ) := by sorry

theorem smallest_positive_integer : 
  ∀ n : ℤ, n > 0 → (1 : ℤ) ≤ n := by sorry

theorem main_result (a : ℚ) :
  (a ≠ 0 ∧ a = a⁻¹) → 
  (∃ (n : ℤ), (n : ℚ) = a^2014 ∧ ∀ m : ℤ, m > 0 → n ≤ m) := by sorry

end NUMINAMATH_CALUDE_rational_reciprocal_power_smallest_positive_integer_main_result_l2301_230190


namespace NUMINAMATH_CALUDE_dihedral_angle_perpendicular_halfplanes_l2301_230147

-- Define dihedral angle
def DihedralAngle : Type := sorry

-- Define half-plane of a dihedral angle
def halfPlane (α : DihedralAngle) : Type := sorry

-- Define perpendicularity of half-planes
def perpendicular (p q : Type) : Prop := sorry

-- Define equality of dihedral angles
def equal (α β : DihedralAngle) : Prop := sorry

-- Define complementary dihedral angles
def complementary (α β : DihedralAngle) : Prop := sorry

-- The theorem
theorem dihedral_angle_perpendicular_halfplanes 
  (α β : DihedralAngle) : 
  perpendicular (halfPlane α) (halfPlane β) → 
  equal α β ∨ complementary α β := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_perpendicular_halfplanes_l2301_230147


namespace NUMINAMATH_CALUDE_P_equals_Q_l2301_230179

-- Define set P
def P : Set ℝ := {m : ℝ | -1 < m ∧ m ≤ 0}

-- Define set Q
def Q : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

-- Theorem statement
theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l2301_230179


namespace NUMINAMATH_CALUDE_sixth_root_of_68968845601_l2301_230124

theorem sixth_root_of_68968845601 :
  51^6 = 68968845601 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_68968845601_l2301_230124


namespace NUMINAMATH_CALUDE_number_of_boys_l2301_230105

theorem number_of_boys (M W B : ℕ) : 
  M = W → 
  W = B → 
  M * 8 = 120 → 
  B = 15 := by sorry

end NUMINAMATH_CALUDE_number_of_boys_l2301_230105


namespace NUMINAMATH_CALUDE_black_hen_day_probability_l2301_230199

/-- Represents the color of a hen -/
inductive HenColor
| Black
| White

/-- Represents a program type -/
inductive ProgramType
| Day
| Evening

/-- Represents the state of available spots -/
structure AvailableSpots :=
  (day : Nat)
  (evening : Nat)

/-- Represents a hen's application -/
structure Application :=
  (color : HenColor)
  (program : ProgramType)

/-- The probability of at least one black hen in the daytime program -/
def prob_black_hen_day (total_spots : Nat) (day_spots : Nat) (evening_spots : Nat) 
                       (black_hens : Nat) (white_hens : Nat) : ℚ :=
  sorry

theorem black_hen_day_probability :
  let total_spots := 5
  let day_spots := 2
  let evening_spots := 3
  let black_hens := 3
  let white_hens := 1
  prob_black_hen_day total_spots day_spots evening_spots black_hens white_hens = 59 / 64 :=
by sorry

end NUMINAMATH_CALUDE_black_hen_day_probability_l2301_230199


namespace NUMINAMATH_CALUDE_infinitely_many_palindromes_l2301_230111

/-- Arithmetic progression term -/
def a (n : ℕ+) : ℕ := 18 + 19 * (n - 1)

/-- Repunit -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- k values -/
def k (t : ℕ) : ℕ := 18 * t + 6

theorem infinitely_many_palindromes :
  ∀ m : ℕ, ∃ t : ℕ, t > m ∧ ∃ n : ℕ+, R (k t) = a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_palindromes_l2301_230111


namespace NUMINAMATH_CALUDE_singer_hourly_rate_l2301_230116

/-- Given a singer hired for 3 hours, with a 20% tip, and a total payment of $54, 
    the hourly rate for the singer is $15. -/
theorem singer_hourly_rate (hours : ℕ) (tip_percentage : ℚ) (total_payment : ℚ) :
  hours = 3 →
  tip_percentage = 1/5 →
  total_payment = 54 →
  ∃ (hourly_rate : ℚ), 
    hourly_rate * hours * (1 + tip_percentage) = total_payment ∧
    hourly_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_singer_hourly_rate_l2301_230116


namespace NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_not_zero_l2301_230133

-- Define the complex number i
def i : ℂ := Complex.I

-- Define x and y
def x : ℂ := i
def y : ℂ := -i

-- State the theorem
theorem x_fourth_plus_y_fourth_not_zero : x^4 + y^4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_not_zero_l2301_230133


namespace NUMINAMATH_CALUDE_min_points_on_circle_l2301_230173

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for circles in a plane
def Circle : Type := Point × ℝ

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Function to count points on a circle
def countPointsOnCircle (points : List Point) (c : Circle) : Nat := sorry

-- Main theorem
theorem min_points_on_circle 
  (points : List Point) 
  (h1 : points.length = 10)
  (h2 : ∀ (sublist : List Point), sublist ⊆ points → sublist.length = 5 → 
        ∃ (c : Circle), (countPointsOnCircle sublist c) ≥ 4) :
  ∃ (c : Circle), (countPointsOnCircle points c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l2301_230173


namespace NUMINAMATH_CALUDE_inequality_solution_l2301_230184

theorem inequality_solution (x : ℝ) : 
  |((3 * x - 2) / (x^2 - x - 2))| > 3 ↔ 
  (x > -1 ∧ x < -2/3) ∨ (x > 1/3 ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2301_230184


namespace NUMINAMATH_CALUDE_proposition_false_negation_true_l2301_230108

theorem proposition_false_negation_true :
  (¬ (∀ x y : ℝ, x + y > 0 → x > 0 ∧ y > 0)) ∧
  (∃ x y : ℝ, x + y > 0 ∧ (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_false_negation_true_l2301_230108


namespace NUMINAMATH_CALUDE_total_food_consumption_theorem_l2301_230143

/-- The total amount of food consumed daily by both sides in a war --/
def total_food_consumption (first_side_soldiers : ℕ) (food_per_soldier_first : ℕ) 
  (soldier_difference : ℕ) (food_difference : ℕ) : ℕ :=
  let second_side_soldiers := first_side_soldiers - soldier_difference
  let food_per_soldier_second := food_per_soldier_first - food_difference
  (first_side_soldiers * food_per_soldier_first) + 
  (second_side_soldiers * food_per_soldier_second)

/-- Theorem stating the total food consumption for both sides --/
theorem total_food_consumption_theorem :
  total_food_consumption 4000 10 500 2 = 68000 := by
  sorry

end NUMINAMATH_CALUDE_total_food_consumption_theorem_l2301_230143


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2301_230100

theorem vector_difference_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 10) → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2301_230100


namespace NUMINAMATH_CALUDE_digit_five_occurrences_l2301_230128

/-- The number of occurrences of a digit in a specific place value when writing numbers from 1 to n -/
def occurrences_in_place (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10 ^ place)) * (10 ^ (place - 1))

/-- The total number of occurrences of the digit 5 when writing all integers from 1 to n -/
def total_occurrences (n : ℕ) : ℕ :=
  occurrences_in_place n 0 + occurrences_in_place n 1 + 
  occurrences_in_place n 2 + occurrences_in_place n 3

theorem digit_five_occurrences :
  total_occurrences 10000 = 4000 := by sorry

end NUMINAMATH_CALUDE_digit_five_occurrences_l2301_230128


namespace NUMINAMATH_CALUDE_lucas_52_mod_5_l2301_230142

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas (n + 1) + lucas n

theorem lucas_52_mod_5 : lucas 51 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucas_52_mod_5_l2301_230142


namespace NUMINAMATH_CALUDE_fraction_difference_equals_difference_over_product_l2301_230178

theorem fraction_difference_equals_difference_over_product 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = (y - x) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_difference_over_product_l2301_230178


namespace NUMINAMATH_CALUDE_novel_writing_stats_l2301_230153

-- Define the given conditions
def total_words : ℕ := 50000
def total_hours : ℕ := 100
def hours_per_day : ℕ := 5

-- Theorem to prove
theorem novel_writing_stats :
  (total_words / total_hours = 500) ∧
  (total_hours / hours_per_day = 20) := by
  sorry

end NUMINAMATH_CALUDE_novel_writing_stats_l2301_230153


namespace NUMINAMATH_CALUDE_max_y_coordinate_l2301_230125

theorem max_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) + y = 0 →
  y ≤ (-19 + Real.sqrt 325) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l2301_230125


namespace NUMINAMATH_CALUDE_line_slope_l2301_230109

/-- The slope of the line given by the equation x/4 + y/5 = 1 is -5/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l2301_230109


namespace NUMINAMATH_CALUDE_cuboid_volume_error_percentage_l2301_230162

/-- The error percentage in volume calculation for a cuboid with specific measurement errors -/
theorem cuboid_volume_error_percentage :
  let length_error := 1.08  -- 8% excess
  let breadth_error := 0.95 -- 5% deficit
  let height_error := 0.90  -- 10% deficit
  let volume_error := length_error * breadth_error * height_error
  let error_percentage := (volume_error - 1) * 100
  error_percentage = -2.74 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_error_percentage_l2301_230162


namespace NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l2301_230171

theorem temperature_difference (B : ℝ) (N : ℝ) : 
  (∃ A : ℝ, A = B + N) → 
  (|((B + N) - 4) - (B + 5)| = 1) →
  (N = 10 ∨ N = 8) :=
by sorry

theorem product_of_N_values :
  ∃ N₁ N₂ : ℝ, 
    (∃ B : ℝ, (∃ A : ℝ, A = B + N₁) ∧ |((B + N₁) - 4) - (B + 5)| = 1) ∧
    (∃ B : ℝ, (∃ A : ℝ, A = B + N₂) ∧ |((B + N₂) - 4) - (B + 5)| = 1) ∧
    N₁ * N₂ = 80 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_of_N_values_l2301_230171


namespace NUMINAMATH_CALUDE_dice_probability_l2301_230101

/-- A fair 10-sided die -/
def ten_sided_die : Finset ℕ := Finset.range 10

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- The event that the number on the 10-sided die is less than or equal to the number on the 6-sided die -/
def favorable_event : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ≤ p.2) (ten_sided_die.product six_sided_die)

/-- The probability of the event -/
def probability : ℚ :=
  (favorable_event.card : ℚ) / ((ten_sided_die.card * six_sided_die.card) : ℚ)

theorem dice_probability : probability = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2301_230101


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2301_230139

theorem min_value_quadratic (x y : ℝ) (h : x^2 + x*y + y^2 = 3) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ z, z = x^2 - x*y + y^2 → z ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2301_230139


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l2301_230175

theorem license_plate_palindrome_probability :
  let prob_4digit_palindrome : ℚ := 1 / 100
  let prob_3letter_palindrome : ℚ := 1 / 26
  let prob_at_least_one_palindrome : ℚ := 
    prob_3letter_palindrome + prob_4digit_palindrome - (prob_3letter_palindrome * prob_4digit_palindrome)
  prob_at_least_one_palindrome = 5 / 104 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l2301_230175


namespace NUMINAMATH_CALUDE_correct_payments_l2301_230104

/-- Represents the weekly payments to three employees --/
structure EmployeePayments where
  total : ℕ
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given payments satisfy the problem conditions --/
def isValidPayment (p : EmployeePayments) : Prop :=
  p.total = 1500 ∧
  p.a = (150 * p.b) / 100 ∧
  p.c = (80 * p.b) / 100 ∧
  p.a + p.b + p.c = p.total

/-- The theorem stating the correct payments --/
theorem correct_payments :
  ∃ (p : EmployeePayments), isValidPayment p ∧ p.a = 682 ∧ p.b = 454 ∧ p.c = 364 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_payments_l2301_230104


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2301_230129

theorem solve_linear_equation :
  ∃ x : ℝ, -2 * x - 7 = 7 * x + 2 ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2301_230129


namespace NUMINAMATH_CALUDE_binomial_20_5_l2301_230183

theorem binomial_20_5 : Nat.choose 20 5 = 15504 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_5_l2301_230183
