import Mathlib

namespace NUMINAMATH_CALUDE_baseball_cards_cost_theorem_l560_56079

/-- The cost of a baseball card deck given the total spent and the cost of Digimon card packs -/
def baseball_card_cost (total_spent : ℝ) (digimon_pack_cost : ℝ) (num_digimon_packs : ℕ) : ℝ :=
  total_spent - (digimon_pack_cost * num_digimon_packs)

/-- Theorem: The cost of the baseball cards is $6.06 -/
theorem baseball_cards_cost_theorem (total_spent : ℝ) (digimon_pack_cost : ℝ) (num_digimon_packs : ℕ) :
  total_spent = 23.86 ∧ digimon_pack_cost = 4.45 ∧ num_digimon_packs = 4 →
  baseball_card_cost total_spent digimon_pack_cost num_digimon_packs = 6.06 := by
  sorry

#eval baseball_card_cost 23.86 4.45 4

end NUMINAMATH_CALUDE_baseball_cards_cost_theorem_l560_56079


namespace NUMINAMATH_CALUDE_francine_work_weeks_francine_work_weeks_solution_l560_56010

theorem francine_work_weeks 
  (daily_distance : ℕ) 
  (workdays_per_week : ℕ) 
  (total_distance : ℕ) : ℕ :=
  let weekly_distance := daily_distance * workdays_per_week
  total_distance / weekly_distance

#check francine_work_weeks 140 4 2240

theorem francine_work_weeks_solution :
  francine_work_weeks 140 4 2240 = 4 := by
  sorry

end NUMINAMATH_CALUDE_francine_work_weeks_francine_work_weeks_solution_l560_56010


namespace NUMINAMATH_CALUDE_harry_marbles_distribution_l560_56047

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Harry's distribution -/
theorem harry_marbles_distribution (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 45) : 
  min_additional_marbles num_friends initial_marbles = 33 := by
  sorry

end NUMINAMATH_CALUDE_harry_marbles_distribution_l560_56047


namespace NUMINAMATH_CALUDE_regular_quad_pyramid_theorem_l560_56087

/-- A regular quadrilateral pyramid with a plane drawn through the diagonal of the base and the height -/
structure RegularQuadPyramid where
  /-- The ratio of the area of the cross-section to the lateral surface -/
  k : ℝ
  /-- The ratio k is positive -/
  k_pos : k > 0

/-- The cosine of the angle between slant heights of opposite lateral faces -/
def slant_height_angle_cos (p : RegularQuadPyramid) : ℝ := 16 * p.k^2 - 1

/-- The theorem stating the cosine of the angle between slant heights and the permissible values of k -/
theorem regular_quad_pyramid_theorem (p : RegularQuadPyramid) :
  slant_height_angle_cos p = 16 * p.k^2 - 1 ∧ p.k < 0.25 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_regular_quad_pyramid_theorem_l560_56087


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l560_56043

/-- For a line with equation 2x + y + 1 = 0, its slope is -2 and y-intercept is -1 -/
theorem line_slope_and_intercept :
  ∀ (x y : ℝ), 2*x + y + 1 = 0 → 
  ∃ (k b : ℝ), k = -2 ∧ b = -1 ∧ y = k*x + b := by
sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l560_56043


namespace NUMINAMATH_CALUDE_lake_bright_population_is_16000_l560_56016

-- Define the total population
def total_population : ℕ := 80000

-- Define Gordonia's population as a fraction of the total
def gordonia_population : ℕ := total_population / 2

-- Define Toadon's population as a percentage of Gordonia's
def toadon_population : ℕ := (gordonia_population * 60) / 100

-- Define Lake Bright's population
def lake_bright_population : ℕ := total_population - gordonia_population - toadon_population

-- Theorem statement
theorem lake_bright_population_is_16000 :
  lake_bright_population = 16000 := by sorry

end NUMINAMATH_CALUDE_lake_bright_population_is_16000_l560_56016


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l560_56051

def A : Set ℝ := {x | |x - 2| ≤ 3}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x : ℝ | x ≤ 1 ∨ x > 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l560_56051


namespace NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l560_56040

/-- Represents a right circular cone --/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones --/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- Calculates the maximum squared radius of a sphere fitting inside the intersecting cones --/
def maxSquaredSphereRadius (ic : IntersectingCones) : ℝ := sorry

/-- Theorem statement --/
theorem max_sphere_in_intersecting_cones :
  let ic : IntersectingCones := {
    cone1 := { baseRadius := 4, height := 10 },
    cone2 := { baseRadius := 4, height := 10 },
    intersectionDistance := 4
  }
  maxSquaredSphereRadius ic = 144 := by sorry

end NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l560_56040


namespace NUMINAMATH_CALUDE_triple_a_student_distribution_l560_56075

theorem triple_a_student_distribution (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 36 :=
sorry

end NUMINAMATH_CALUDE_triple_a_student_distribution_l560_56075


namespace NUMINAMATH_CALUDE_exam_exemption_logic_l560_56027

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (score_above_90 : Student → Prop)
variable (exempted : Student → Prop)

-- State the theorem
theorem exam_exemption_logic (s : Student) 
  (h : ∀ x, score_above_90 x → exempted x) :
  ¬(exempted s) → ¬(score_above_90 s) := by
  sorry

end NUMINAMATH_CALUDE_exam_exemption_logic_l560_56027


namespace NUMINAMATH_CALUDE_parabola_symmetry_and_form_l560_56048

/-- A parabola with equation y = x^2 - 5x + 2 -/
def parabola (x : ℝ) : ℝ := x^2 - 5*x + 2

/-- The symmetry point of the parabola -/
def symmetry_point : ℝ × ℝ := (3, 2)

/-- The general form of a quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a*x^2 + b*x + c

/-- Theorem stating that the parabola y = x^2 - 5x + 2 is symmetric about (3, 2)
    and can be written as y = ax^2 + bx + c, where 3a + 3c + b = -8 -/
theorem parabola_symmetry_and_form :
  ∃ (a b c : ℝ),
    (∀ x : ℝ, parabola x = quadratic a b c x) ∧
    (∀ x y : ℝ, parabola x = y ↔ parabola (2*symmetry_point.1 - x) = y) ∧
    3*a + 3*c + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_and_form_l560_56048


namespace NUMINAMATH_CALUDE_parabola_equation_l560_56002

-- Define the parabola C
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions of the problem
axiom vertex_at_origin : ∃ C : Parabola, C.equation 0 0
axiom focus_on_x_axis : ∃ C : Parabola, ∃ f : ℝ, C.equation f 0
axiom line_intersects_parabola : ∃ C : Parabola, ∃ A B : Point, 
  C.equation A.x A.y ∧ C.equation B.x B.y ∧ A.x - A.y = 0 ∧ B.x - B.y = 0
axiom midpoint_condition : ∃ C : Parabola, ∃ A B : Point, 
  C.equation A.x A.y ∧ C.equation B.x B.y ∧ (A.x + B.x) / 2 = 1 ∧ (A.y + B.y) / 2 = 1

-- The theorem to prove
theorem parabola_equation : 
  ∃ C : Parabola, ∀ x y : ℝ, C.equation x y ↔ x^2 = 2*y :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l560_56002


namespace NUMINAMATH_CALUDE_max_sectional_area_of_cone_l560_56021

/-- The maximum sectional area of a cone --/
theorem max_sectional_area_of_cone (θ : Real) (l : Real) : 
  θ = π / 3 → l = 3 → (∀ α, 0 ≤ α ∧ α ≤ 2*π/3 → (1/2) * l^2 * Real.sin α ≤ 9/2) ∧ 
  ∃ α, 0 ≤ α ∧ α ≤ 2*π/3 ∧ (1/2) * l^2 * Real.sin α = 9/2 :=
by sorry

#check max_sectional_area_of_cone

end NUMINAMATH_CALUDE_max_sectional_area_of_cone_l560_56021


namespace NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_result_l560_56036

theorem multiply_subtract_distribute (a b c : ℕ) :
  a * c - b * c = (a - b) * c :=
by sorry

theorem computation_result : 72 * 1313 - 32 * 1313 = 52520 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_distribute_computation_result_l560_56036


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l560_56029

/-- Proves that the ratio of girls to boys is 4:5 given the class conditions -/
theorem girls_to_boys_ratio (total_students : ℕ) (boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  total_students = 30 →
  boys = 10 →
  cups_per_boy = 5 →
  total_cups = 90 →
  (total_students - boys, boys) = (20, 10) ∧ 
  (20 : ℚ) / 10 = 4 / 5 := by
  sorry

#check girls_to_boys_ratio

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l560_56029


namespace NUMINAMATH_CALUDE_problem_statement_l560_56019

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (h : b * Real.log a - a * Real.log b = a - b) : 
  (a + b - a * b > 1) ∧ (a + b > 2) ∧ (1 / a + 1 / b > 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l560_56019


namespace NUMINAMATH_CALUDE_base7_multiplication_l560_56082

/-- Converts a number from base 7 to base 10 --/
def toBase10 (n : ℕ) : ℕ :=
  sorry

/-- Converts a number from base 10 to base 7 --/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Multiplies two numbers in base 7 --/
def multiplyBase7 (a b : ℕ) : ℕ :=
  toBase7 (toBase10 a * toBase10 b)

theorem base7_multiplication :
  multiplyBase7 325 6 = 2624 :=
sorry

end NUMINAMATH_CALUDE_base7_multiplication_l560_56082


namespace NUMINAMATH_CALUDE_smallest_c_value_l560_56005

-- Define the polynomial
def polynomial (c d x : ℤ) : ℤ := x^3 - c*x^2 + d*x - 2310

-- Define the property that the polynomial has three positive integer roots
def has_three_positive_integer_roots (c d : ℤ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ x, polynomial c d x = (x - r₁) * (x - r₂) * (x - r₃)

-- State the theorem
theorem smallest_c_value (c d : ℤ) :
  has_three_positive_integer_roots c d →
  (∀ c' d', has_three_positive_integer_roots c' d' → c ≤ c') →
  c = 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l560_56005


namespace NUMINAMATH_CALUDE_distribute_9_4_l560_56024

/-- The number of ways to distribute n identical items into k boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 220 ways to distribute 9 identical items into 4 boxes -/
theorem distribute_9_4 : distribute 9 4 = 220 := by
  sorry

end NUMINAMATH_CALUDE_distribute_9_4_l560_56024


namespace NUMINAMATH_CALUDE_inscribed_cylinder_height_l560_56012

theorem inscribed_cylinder_height (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 8 → c = 3 →
  h^2 = r^2 - c^2 →
  h = Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_height_l560_56012


namespace NUMINAMATH_CALUDE_count_five_digit_with_four_or_five_l560_56000

/-- The number of five-digit positive integers. -/
def total_five_digit_integers : ℕ := 90000

/-- The number of five-digit positive integers without 4 or 5. -/
def five_digit_without_four_or_five : ℕ := 28672

/-- The number of five-digit positive integers containing either 4 or 5 at least once. -/
def five_digit_with_four_or_five : ℕ := total_five_digit_integers - five_digit_without_four_or_five

theorem count_five_digit_with_four_or_five :
  five_digit_with_four_or_five = 61328 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_with_four_or_five_l560_56000


namespace NUMINAMATH_CALUDE_volume_third_number_l560_56008

/-- Given a volume that is the product of three numbers, where two numbers are 12 and 18,
    and 48 cubes of edge 3 can be inserted into it, the third number in the product is 6. -/
theorem volume_third_number (volume : ℕ) (x : ℕ) : 
  volume = 12 * 18 * x →
  volume = 48 * 3^3 →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_volume_third_number_l560_56008


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_N_l560_56069

-- Define the number N as described in the problem
def N : ℕ := sorry

-- Define a function to calculate the sum of digits of N
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem highest_power_of_three_in_N :
  ∃ (k : ℕ), k = 3 ∧ 
  (N % (3^k) = 0) ∧ 
  (∀ m > k, N % (3^m) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_N_l560_56069


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l560_56023

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l560_56023


namespace NUMINAMATH_CALUDE_equal_intersection_areas_l560_56011

/-- A tetrahedron with specific properties -/
structure Tetrahedron where
  opposite_edges_perpendicular : Bool
  vertical_segment : ℝ
  midplane_area : ℝ

/-- A sphere with a specific radius -/
structure Sphere where
  radius : ℝ

/-- The configuration of a tetrahedron and a sphere -/
structure Configuration where
  tetra : Tetrahedron
  sphere : Sphere
  radius_condition : sphere.radius^2 * π = tetra.midplane_area
  vertical_segment_condition : tetra.vertical_segment = 2 * sphere.radius

/-- The area of intersection of a shape with a plane -/
def intersection_area (height : ℝ) : Configuration → ℝ
  | _ => sorry

/-- The main theorem stating that the areas of intersection are equal for all heights -/
theorem equal_intersection_areas (config : Configuration) :
  ∀ h : ℝ, 0 ≤ h ∧ h ≤ config.tetra.vertical_segment →
    intersection_area h config = intersection_area (config.tetra.vertical_segment - h) config :=
  sorry

end NUMINAMATH_CALUDE_equal_intersection_areas_l560_56011


namespace NUMINAMATH_CALUDE_vector_on_line_l560_56068

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors p and q
variable (p q : V)

-- Define the condition that p and q are distinct
variable (h_distinct : p ≠ q)

-- Define the line passing through p and q
def line_through (p q : V) := {x : V | ∃ t : ℝ, x = p + t • (q - p)}

-- Define the theorem
theorem vector_on_line (m : ℝ) (h_m : m = 5/8) :
  ∃ k : ℝ, k = 3/8 ∧ (k • p + m • q) ∈ line_through p q :=
sorry

end NUMINAMATH_CALUDE_vector_on_line_l560_56068


namespace NUMINAMATH_CALUDE_polynomial_sum_l560_56090

theorem polynomial_sum (a b c d : ℝ) : 
  (fun x : ℝ => (4*x^2 - 3*x + 2)*(5 - x)) = 
  (fun x : ℝ => a*x^3 + b*x^2 + c*x + d) → 
  5*a + 3*b + 2*c + d = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l560_56090


namespace NUMINAMATH_CALUDE_system_solution_l560_56097

theorem system_solution (x y k : ℝ) : 
  (2 * x - y = 5 * k + 6) → 
  (4 * x + 7 * y = k) → 
  (x + y = 2023) → 
  (k = 2022) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l560_56097


namespace NUMINAMATH_CALUDE_race_speed_ratio_l560_56055

theorem race_speed_ratio (race_distance : ℕ) (head_start : ℕ) (win_margin : ℕ) :
  race_distance = 500 →
  head_start = 140 →
  win_margin = 20 →
  (race_distance - head_start : ℚ) / (race_distance - win_margin : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l560_56055


namespace NUMINAMATH_CALUDE_standard_deviation_decreases_after_correction_l560_56073

/-- Represents a class with test scores -/
structure TestScores where
  size : ℕ
  average : ℝ
  standardDev : ℝ

/-- Represents a score correction -/
structure ScoreCorrection where
  oldScore : ℝ
  newScore : ℝ

/-- The main theorem stating that the original standard deviation is greater than the new one after corrections -/
theorem standard_deviation_decreases_after_correction 
  (original : TestScores)
  (correction1 correction2 : ScoreCorrection)
  (new_std_dev : ℝ)
  (h_size : original.size = 50)
  (h_avg : original.average = 70)
  (h_correction1 : correction1.oldScore = 50 ∧ correction1.newScore = 80)
  (h_correction2 : correction2.oldScore = 100 ∧ correction2.newScore = 70)
  : original.standardDev > new_std_dev := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_decreases_after_correction_l560_56073


namespace NUMINAMATH_CALUDE_sine_area_theorem_l560_56026

open Set
open MeasureTheory
open Interval

-- Define the sine function
noncomputable def f (x : ℝ) := Real.sin x

-- Define the interval
def I : Set ℝ := Icc (-Real.pi) (2 * Real.pi)

-- State the theorem
theorem sine_area_theorem :
  (∫ x in I, |f x| ∂volume) = 6 := by sorry

end NUMINAMATH_CALUDE_sine_area_theorem_l560_56026


namespace NUMINAMATH_CALUDE_translation_result_l560_56080

/-- Represents a 2D point with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Translates a point by given x and y offsets -/
def translate (p : Point) (dx dy : Int) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_result :
  let initial_point : Point := { x := -5, y := 1 }
  let final_point : Point := translate (translate initial_point 2 0) 0 (-4)
  final_point = { x := -3, y := -3 } := by sorry

end NUMINAMATH_CALUDE_translation_result_l560_56080


namespace NUMINAMATH_CALUDE_new_library_capacity_l560_56066

theorem new_library_capacity 
  (M : ℚ) -- Millicent's total number of books
  (H : ℚ) -- Harold's total number of books
  (h1 : H = (1 : ℚ) / 2 * M) -- Harold has 1/2 as many books as Millicent
  (h2 : (1 : ℚ) / 3 * H + (1 : ℚ) / 2 * M > 0) -- New home's capacity is positive
  : ((1 : ℚ) / 3 * H + (1 : ℚ) / 2 * M) / M = (2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_new_library_capacity_l560_56066


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l560_56046

theorem polynomial_division_theorem (x : ℚ) : 
  let dividend := 10 * x^4 + 5 * x^3 - 8 * x^2 + 7 * x - 3
  let divisor := 3 * x + 2
  let quotient := 10/3 * x^3 - 5/9 * x^2 - 31/27 * x + 143/81
  let remainder := 88/9
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l560_56046


namespace NUMINAMATH_CALUDE_difference_of_squares_650_550_l560_56059

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_650_550_l560_56059


namespace NUMINAMATH_CALUDE_right_triangle_count_l560_56018

/-- A point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- Definition of the rectangle ABCD and points E, F, G -/
def rectangle_setup :=
  let A := Point.mk 0 0
  let B := Point.mk 6 0
  let C := Point.mk 6 4
  let D := Point.mk 0 4
  let E := Point.mk 3 0
  let F := Point.mk 3 4
  let G := Point.mk 2 4
  (A, B, C, D, E, F, G)

/-- Function to count right triangles -/
def count_right_triangles (points : Point × Point × Point × Point × Point × Point × Point) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of right triangles is 16 -/
theorem right_triangle_count :
  count_right_triangles rectangle_setup = 16 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_count_l560_56018


namespace NUMINAMATH_CALUDE_expression_factorization_l560_56060

theorem expression_factorization (b : ℝ) :
  (8 * b^3 + 120 * b^2 - 14) - (9 * b^3 - 2 * b^2 + 14) = -1 * (b^3 - 122 * b^2 + 28) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l560_56060


namespace NUMINAMATH_CALUDE_regular_polygon_with_four_to_one_angle_ratio_l560_56083

/-- A regular polygon where the interior angle is exactly 4 times the exterior angle has 10 sides -/
theorem regular_polygon_with_four_to_one_angle_ratio (n : ℕ) : 
  n > 2 → 
  (360 / n : ℚ) * 4 = (180 - 360 / n : ℚ) → 
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_four_to_one_angle_ratio_l560_56083


namespace NUMINAMATH_CALUDE_all_star_seating_arrangements_l560_56070

/-- Represents the number of ways to arrange All-Stars from different teams in a row --/
def allStarArrangements (total : Nat) (team1 : Nat) (team2 : Nat) (team3 : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial team1 * Nat.factorial team2 * Nat.factorial team3

/-- Theorem stating the number of arrangements for 8 All-Stars from 3 teams --/
theorem all_star_seating_arrangements :
  allStarArrangements 8 3 3 2 = 432 := by
  sorry

#eval allStarArrangements 8 3 3 2

end NUMINAMATH_CALUDE_all_star_seating_arrangements_l560_56070


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l560_56085

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (a % 2 = 1) →                 -- a is odd
  (a + (a + 4) = 150) →         -- sum of first and third is 150
  (a + (a + 2) + (a + 4) = 225) -- sum of all three is 225
  := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l560_56085


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l560_56041

theorem max_value_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 9) :
  Real.sqrt (x + 15) + Real.sqrt (9 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 143 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l560_56041


namespace NUMINAMATH_CALUDE_sector_area_l560_56074

/-- The area of a circular sector with a central angle of 60° and a radius of 10 cm is 50π/3 cm². -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 60 * π / 180) (h2 : r = 10) :
  (θ / (2 * π)) * (π * r^2) = 50 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l560_56074


namespace NUMINAMATH_CALUDE_find_g_value_l560_56035

theorem find_g_value (x g : ℝ) (h1 : x = 0.3) 
  (h2 : (10 * x + 2) / 4 - (3 * x - 6) / 18 = (g * x + 4) / 3) : g = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_g_value_l560_56035


namespace NUMINAMATH_CALUDE_parallel_transitivity_l560_56009

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "in plane" relation for a line
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (a b : Line) (α : Plane) :
  parallel_line a b →
  ¬ in_plane a α →
  ¬ in_plane b α →
  parallel_line_plane a α →
  parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l560_56009


namespace NUMINAMATH_CALUDE_jake_brought_six_balloons_l560_56062

/-- The number of balloons Jake brought to the park -/
def jakes_balloons (allans_initial_balloons allans_bought_balloons : ℕ) : ℕ :=
  allans_initial_balloons + allans_bought_balloons + 1

/-- Theorem stating that Jake brought 6 balloons to the park -/
theorem jake_brought_six_balloons :
  jakes_balloons 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jake_brought_six_balloons_l560_56062


namespace NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_milk_powder_l560_56007

/-- Represents different types of statistical charts -/
inductive ChartType
  | Line
  | Bar
  | Pie

/-- Represents a substance in milk powder -/
structure Substance where
  name : String
  percentage : Float

/-- Represents the composition of milk powder -/
def MilkPowderComposition := List Substance

/-- Determines if a chart type is appropriate for displaying percentage composition -/
def is_appropriate_for_percentage_composition (chart : ChartType) (composition : MilkPowderComposition) : Prop :=
  chart = ChartType.Pie

/-- Theorem stating that a pie chart is the most appropriate for displaying milk powder composition -/
theorem pie_chart_most_appropriate_for_milk_powder (composition : MilkPowderComposition) :
  is_appropriate_for_percentage_composition ChartType.Pie composition :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_most_appropriate_for_milk_powder_l560_56007


namespace NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l560_56063

def expansion (x : ℝ) : ℝ := (x^2 + 2) * (3*x^3 - x^2 + 4) - 2 * (x^4 - 3*x^3 + x^2)

def count_nonzero_terms (p : ℝ → ℝ) : ℕ := sorry

theorem expansion_has_four_nonzero_terms :
  count_nonzero_terms expansion = 4 := by sorry

end NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l560_56063


namespace NUMINAMATH_CALUDE_problem_solution_l560_56054

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ) 
  (h_xavier : p_xavier = 1/4)
  (h_yvonne : p_yvonne = 1/3)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l560_56054


namespace NUMINAMATH_CALUDE_cylinder_min_surface_area_l560_56017

/-- Given a cone with base radius 4 and slant height 5, and a cylinder with equal volume,
    the surface area of the cylinder is minimized when its base radius is 2. -/
theorem cylinder_min_surface_area (r h : ℝ) : 
  r > 0 → h > 0 → 
  (π * r^2 * h = (1/3) * π * 4^2 * 3) →
  (∀ r' h' : ℝ, r' > 0 → h' > 0 → π * r'^2 * h' = (1/3) * π * 4^2 * 3 → 
    2 * π * r * (r + h) ≤ 2 * π * r' * (r' + h')) →
  r = 2 := by sorry

end NUMINAMATH_CALUDE_cylinder_min_surface_area_l560_56017


namespace NUMINAMATH_CALUDE_chairs_to_hall_l560_56095

theorem chairs_to_hall (num_students : ℕ) (chairs_per_trip : ℕ) (num_trips : ℕ) :
  num_students = 5 → chairs_per_trip = 5 → num_trips = 10 →
  num_students * chairs_per_trip * num_trips = 250 := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_hall_l560_56095


namespace NUMINAMATH_CALUDE_product_equals_720_l560_56014

theorem product_equals_720 (n : ℕ) (h : n = 5) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_720_l560_56014


namespace NUMINAMATH_CALUDE_max_volume_box_l560_56015

/-- The maximum volume of a box created from a rectangular metal sheet --/
theorem max_volume_box (sheet_length sheet_width : ℝ) (h_length : sheet_length = 16)
  (h_width : sheet_width = 12) :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < sheet_length / 2 ∧ 
    x < sheet_width / 2 ∧
    ∀ (y : ℝ), 
      0 < y ∧ 
      y < sheet_length / 2 ∧ 
      y < sheet_width / 2 → 
      y * (sheet_length - 2*y) * (sheet_width - 2*y) ≤ 128 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_box_l560_56015


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l560_56088

/-- Given a parabola y² = 2px (p > 0) and a point A(m, 2√2) on it,
    if a circle centered at A with radius |AF| intersects the y-axis
    with a chord of length 2√7, then m = (2√3)/3 -/
theorem parabola_circle_intersection (p m : ℝ) (hp : p > 0) :
  2 * p * m = 8 →
  let f := (2 / m, 0)
  let r := m + 2 / m
  (r^2 - m^2 = 7) →
  m = (2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l560_56088


namespace NUMINAMATH_CALUDE_system_solution_l560_56042

theorem system_solution (a : ℝ) (x y z : ℝ) :
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = a^2) →
  (x^3 + y^3 + z^3 = a^3) →
  ((x = 0 ∧ y = 0 ∧ z = a) ∨ (x = 0 ∧ y = a ∧ z = 0) ∨ (x = a ∧ y = 0 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l560_56042


namespace NUMINAMATH_CALUDE_front_axle_wheels_count_l560_56022

/-- Represents a truck with a specific wheel configuration -/
structure Truck where
  total_wheels : ℕ
  wheels_per_axle : ℕ
  front_axle_wheels : ℕ
  toll : ℚ

/-- Calculates the number of axles for a given truck -/
def num_axles (t : Truck) : ℕ :=
  (t.total_wheels - t.front_axle_wheels) / t.wheels_per_axle + 1

/-- Calculates the toll for a given number of axles -/
def toll_formula (x : ℕ) : ℚ :=
  (3/2) + (3/2) * (x - 2)

/-- Theorem stating that a truck with the given specifications has 2 wheels on its front axle -/
theorem front_axle_wheels_count (t : Truck) 
    (h1 : t.total_wheels = 18)
    (h2 : t.wheels_per_axle = 4)
    (h3 : t.toll = 6)
    (h4 : t.toll = toll_formula (num_axles t)) :
    t.front_axle_wheels = 2 := by
  sorry

end NUMINAMATH_CALUDE_front_axle_wheels_count_l560_56022


namespace NUMINAMATH_CALUDE_min_ab_for_line_through_point_l560_56065

/-- Given a line equation (x/a) + (y/b) = 1 where a > 0 and b > 0,
    and the line passes through the point (1,1),
    the minimum value of ab is 4. -/
theorem min_ab_for_line_through_point (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → (∀ x y : ℝ, x / a + y / b = 1 → (x, y) = (1, 1)) → 
  ∀ c d : ℝ, c > 0 → d > 0 → (1 / c + 1 / d = 1) → c * d ≥ 4 := by
  sorry

#check min_ab_for_line_through_point

end NUMINAMATH_CALUDE_min_ab_for_line_through_point_l560_56065


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l560_56057

theorem exactly_one_greater_than_one (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (prod_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l560_56057


namespace NUMINAMATH_CALUDE_paco_cookies_l560_56098

def cookies_eaten (initial : ℕ) (given : ℕ) (left : ℕ) : ℕ :=
  initial - given - left

theorem paco_cookies : cookies_eaten 36 14 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l560_56098


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l560_56058

theorem trigonometric_inequality : ∀ (a b c : ℝ),
  a = Real.sin (4/5) →
  b = Real.cos (4/5) →
  c = Real.tan (4/5) →
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l560_56058


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l560_56031

theorem log_equality_implies_ratio (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (Real.log p / Real.log 4 = Real.log q / Real.log 8) ∧
  (Real.log p / Real.log 4 = Real.log (p + q) / Real.log 18) →
  q / p = Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l560_56031


namespace NUMINAMATH_CALUDE_ceiling_plus_one_l560_56033

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_plus_one (x : ℝ) : ceiling (x + 1) = ceiling x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_plus_one_l560_56033


namespace NUMINAMATH_CALUDE_intersection_equality_implies_range_l560_56071

theorem intersection_equality_implies_range (a : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) ↔ (1 ≤ x ∧ x ≤ 2 ∧ 2 - a ≤ x ∧ x ≤ 1 + a)) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_range_l560_56071


namespace NUMINAMATH_CALUDE_factor_x6_minus_x4_minus_x2_plus_1_l560_56089

theorem factor_x6_minus_x4_minus_x2_plus_1 (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_x4_minus_x2_plus_1_l560_56089


namespace NUMINAMATH_CALUDE_paragraphs_per_page_is_twenty_l560_56003

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 50

/-- Represents the total time taken to read the book in hours -/
def total_reading_time : ℕ := 50

/-- Calculates the number of paragraphs per page in the book -/
def paragraphs_per_page : ℕ :=
  (reading_speed * total_reading_time) / (sentences_per_paragraph * total_pages)

/-- Theorem stating that the number of paragraphs per page is 20 -/
theorem paragraphs_per_page_is_twenty :
  paragraphs_per_page = 20 := by
  sorry

end NUMINAMATH_CALUDE_paragraphs_per_page_is_twenty_l560_56003


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l560_56025

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x > 1 ∨ x < -6}

theorem set_intersection_and_union (a : ℝ) :
  (A a ∩ B = ∅ → a ∈ Set.Icc (-6) (-2)) ∧
  (A a ∪ B = B → a ∈ Set.Ioi 1 ∪ Set.Iio (-9)) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l560_56025


namespace NUMINAMATH_CALUDE_tessa_initial_apples_l560_56094

theorem tessa_initial_apples :
  ∀ (initial_apples : ℕ),
    (initial_apples + 5 = 9) →
    initial_apples = 4 := by
  sorry

end NUMINAMATH_CALUDE_tessa_initial_apples_l560_56094


namespace NUMINAMATH_CALUDE_donation_distribution_l560_56076

/-- Proves that donating 80% of $2500 to 8 organizations results in each organization receiving $250 --/
theorem donation_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) :
  total_amount = 2500 →
  donation_percentage = 0.8 →
  num_organizations = 8 →
  (total_amount * donation_percentage) / num_organizations = 250 := by
sorry

end NUMINAMATH_CALUDE_donation_distribution_l560_56076


namespace NUMINAMATH_CALUDE_two_pi_irrational_l560_56086

theorem two_pi_irrational : Irrational (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_two_pi_irrational_l560_56086


namespace NUMINAMATH_CALUDE_pauls_license_plate_earnings_l560_56028

theorem pauls_license_plate_earnings 
  (total_states : ℕ) 
  (pauls_states : ℕ) 
  (total_earnings : ℚ) :
  total_states = 50 →
  pauls_states = 40 →
  total_earnings = 160 →
  (total_earnings / (pauls_states / total_states * 100 : ℚ)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pauls_license_plate_earnings_l560_56028


namespace NUMINAMATH_CALUDE_wage_cut_and_raise_l560_56072

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.75 * original_wage
  let raise_percentage := 1 / 3
  reduced_wage * (1 + raise_percentage) = original_wage := by sorry

end NUMINAMATH_CALUDE_wage_cut_and_raise_l560_56072


namespace NUMINAMATH_CALUDE_largest_after_three_operations_obtainable_1999_l560_56038

-- Define the expansion operation
def expand (a b : ℕ) : ℕ := a * b + a + b

-- Theorem for the largest number after three operations
theorem largest_after_three_operations :
  let step1 := expand 1 4
  let step2 := expand 4 step1
  let step3 := expand step1 step2
  step3 = 499 := by sorry

-- Theorem for the obtainability of 1999
theorem obtainable_1999 :
  ∃ (m n : ℕ), 2000 = 2^m * 5^n := by sorry

end NUMINAMATH_CALUDE_largest_after_three_operations_obtainable_1999_l560_56038


namespace NUMINAMATH_CALUDE_set_union_problem_l560_56061

theorem set_union_problem (a b : ℝ) :
  let M : Set ℝ := {a, b}
  let N : Set ℝ := {a + 1, 3}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l560_56061


namespace NUMINAMATH_CALUDE_square_congruent_neg_one_mod_prime_l560_56050

theorem square_congruent_neg_one_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℤ, k^2 ≡ -1 [ZMOD p]) ↔ p = 2 ∨ p ≡ 1 [ZMOD 4] :=
sorry

end NUMINAMATH_CALUDE_square_congruent_neg_one_mod_prime_l560_56050


namespace NUMINAMATH_CALUDE_right_triangle_check_l560_56064

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a)

theorem right_triangle_check :
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 1 2 2 ∧
  ¬ is_right_triangle (Real.sqrt 2) (Real.sqrt 2) (Real.sqrt 2) ∧
  is_right_triangle 6 8 10 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_check_l560_56064


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l560_56078

theorem cubic_equation_solution :
  ∃! x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l560_56078


namespace NUMINAMATH_CALUDE_inequality_proof_l560_56077

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a^2 / b) + Real.sqrt (b^2 / a) ≥ Real.sqrt a + Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l560_56077


namespace NUMINAMATH_CALUDE_total_spent_calculation_l560_56096

/-- Calculates the total amount spent on t-shirts given the prices, quantities, discount, and tax rate -/
def total_spent (price_a price_b price_c : ℚ) (qty_a qty_b qty_c : ℕ) (discount_b tax_rate : ℚ) : ℚ :=
  let subtotal_a := price_a * qty_a
  let subtotal_b := price_b * qty_b * (1 - discount_b)
  let subtotal_c := price_c * qty_c
  let total_before_tax := subtotal_a + subtotal_b + subtotal_c
  total_before_tax * (1 + tax_rate)

/-- Theorem stating that given the specific conditions, the total amount spent is $695.21 -/
theorem total_spent_calculation :
  total_spent 9.95 12.50 14.95 18 23 15 0.1 0.05 = 695.21 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l560_56096


namespace NUMINAMATH_CALUDE_xyz_value_l560_56037

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 9)
  (eq5 : x + y + z = 6) :
  x * y * z = 14 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l560_56037


namespace NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l560_56032

theorem cos_x_plus_pi_sixth (x : ℝ) (h : Real.sin (π / 3 - x) = -3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l560_56032


namespace NUMINAMATH_CALUDE_phone_package_comparison_l560_56044

/-- Represents the monthly bill for a phone package as a function of call duration. -/
structure PhonePackage where
  monthly_fee : ℝ
  call_fee : ℝ
  bill : ℝ → ℝ

/-- Package A with a monthly fee of 15 yuan and a call fee of 0.1 yuan per minute. -/
def package_a : PhonePackage :=
  { monthly_fee := 15
    call_fee := 0.1
    bill := λ x => 0.1 * x + 15 }

/-- Package B with no monthly fee and a call fee of 0.15 yuan per minute. -/
def package_b : PhonePackage :=
  { monthly_fee := 0
    call_fee := 0.15
    bill := λ x => 0.15 * x }

theorem phone_package_comparison :
  ∃ (x : ℝ),
    (x > 0) ∧
    (package_a.bill x = package_b.bill x) ∧
    (x = 300) ∧
    (∀ y : ℝ, y > x → package_a.bill y < package_b.bill y) :=
by sorry

end NUMINAMATH_CALUDE_phone_package_comparison_l560_56044


namespace NUMINAMATH_CALUDE_sum_of_digits_n_l560_56053

/-- The least 7-digit number that leaves a remainder of 4 when divided by 5, 850, 35, 27, and 90 -/
def n : ℕ := sorry

/-- Condition: n is a 7-digit number -/
axiom n_seven_digits : 1000000 ≤ n ∧ n < 10000000

/-- Condition: n leaves a remainder of 4 when divided by 5 -/
axiom n_mod_5 : n % 5 = 4

/-- Condition: n leaves a remainder of 4 when divided by 850 -/
axiom n_mod_850 : n % 850 = 4

/-- Condition: n leaves a remainder of 4 when divided by 35 -/
axiom n_mod_35 : n % 35 = 4

/-- Condition: n leaves a remainder of 4 when divided by 27 -/
axiom n_mod_27 : n % 27 = 4

/-- Condition: n leaves a remainder of 4 when divided by 90 -/
axiom n_mod_90 : n % 90 = 4

/-- Condition: n is the least number satisfying all the above conditions -/
axiom n_least : ∀ m : ℕ, (1000000 ≤ m ∧ m < 10000000 ∧ 
                          m % 5 = 4 ∧ m % 850 = 4 ∧ m % 35 = 4 ∧ m % 27 = 4 ∧ m % 90 = 4) → n ≤ m

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of n is 22 -/
theorem sum_of_digits_n : sum_of_digits n = 22 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_l560_56053


namespace NUMINAMATH_CALUDE_det_is_zero_l560_56045

variables {α : Type*} [Field α]
variables (s p q : α)

-- Define the polynomial
def f (x : α) := x^3 - s*x^2 + p*x + q

-- Define the roots
structure Roots (s p q : α) where
  a : α
  b : α
  c : α
  root_a : f s p q a = 0
  root_b : f s p q b = 0
  root_c : f s p q c = 0

-- Define the matrix
def matrix (r : Roots s p q) : Matrix (Fin 3) (Fin 3) α :=
  ![![r.a, r.b, r.c],
    ![r.c, r.a, r.b],
    ![r.b, r.c, r.a]]

-- Theorem statement
theorem det_is_zero (r : Roots s p q) : 
  Matrix.det (matrix s p q r) = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_is_zero_l560_56045


namespace NUMINAMATH_CALUDE_emails_left_in_inbox_l560_56039

theorem emails_left_in_inbox (initial_emails : ℕ) : 
  initial_emails = 400 → 
  (initial_emails / 2 - (initial_emails / 2 * 40 / 100) : ℕ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_emails_left_in_inbox_l560_56039


namespace NUMINAMATH_CALUDE_line_through_point_l560_56004

/-- Given a line equation -3/4 - 3kx = 7y and a point (1/3, -8) on this line,
    prove that k = 55.25 is the unique value satisfying these conditions. -/
theorem line_through_point (k : ℝ) : 
  (-3/4 - 3*k*(1/3) = 7*(-8)) ↔ k = 55.25 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l560_56004


namespace NUMINAMATH_CALUDE_arctan_inequality_implies_a_nonnegative_l560_56067

theorem arctan_inequality_implies_a_nonnegative (a : ℝ) : 
  (∀ x : ℝ, Real.arctan (Real.sqrt (x^2 + x + 13/4)) ≥ π/3 - a) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_inequality_implies_a_nonnegative_l560_56067


namespace NUMINAMATH_CALUDE_least_number_divisible_by_3_4_5_7_8_l560_56020

theorem least_number_divisible_by_3_4_5_7_8 : ∀ n : ℕ, n > 0 → (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) → n ≥ 840 := by
  sorry

#check least_number_divisible_by_3_4_5_7_8

end NUMINAMATH_CALUDE_least_number_divisible_by_3_4_5_7_8_l560_56020


namespace NUMINAMATH_CALUDE_luke_fillets_l560_56093

/-- Calculates the total number of fish fillets Luke has after fishing for a given number of days -/
def total_fillets (fish_per_day : ℕ) (days : ℕ) (fillets_per_fish : ℕ) : ℕ :=
  fish_per_day * days * fillets_per_fish

/-- Proves that Luke has 120 fillets after fishing for 30 days, catching 2 fish per day, with 2 fillets per fish -/
theorem luke_fillets :
  total_fillets 2 30 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_luke_fillets_l560_56093


namespace NUMINAMATH_CALUDE_divisor_and_equation_solution_l560_56049

theorem divisor_and_equation_solution :
  ∃ (k : ℕ) (base : ℕ+),
    (929260 : ℕ) % (base : ℕ)^k = 0 ∧
    3^k - k^3 = 1 ∧
    base = 17 ∧
    k = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_and_equation_solution_l560_56049


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l560_56056

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b)
  (h4 : (a + b) / 2 = 7 / 2) (h5 : Real.sqrt (a * b) = 2 * Real.sqrt 3) :
  let c := Real.sqrt (a^2 + b^2)
  Real.sqrt (c^2 - b^2) / b = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l560_56056


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l560_56013

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 - 4*α - 3 = 0) → 
  (β^2 - 4*β - 3 = 0) → 
  (α - 3) * (β - 3) = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l560_56013


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l560_56001

theorem min_sum_reciprocals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (2 * a + 1) + 1 / (2 * b + 1) + 1 / (2 * c + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l560_56001


namespace NUMINAMATH_CALUDE_tan_30_plus_4sin_30_l560_56034

theorem tan_30_plus_4sin_30 : Real.tan (π / 6) + 4 * Real.sin (π / 6) = (Real.sqrt 3 + 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_4sin_30_l560_56034


namespace NUMINAMATH_CALUDE_students_not_picked_l560_56052

theorem students_not_picked (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 58)
  (h2 : num_groups = 8)
  (h3 : students_per_group = 6) : 
  total_students - (num_groups * students_per_group) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_not_picked_l560_56052


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l560_56091

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relationship between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l560_56091


namespace NUMINAMATH_CALUDE_difference_of_numbers_l560_56084

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares : x^2 - y^2 = 32) : 
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l560_56084


namespace NUMINAMATH_CALUDE_deepak_age_l560_56092

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l560_56092


namespace NUMINAMATH_CALUDE_baker_cakes_l560_56099

/-- Calculates the final number of cakes a baker has after selling some and buying new ones. -/
def final_cakes (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

/-- Proves that for the given numbers, the baker ends up with 186 cakes. -/
theorem baker_cakes : final_cakes 121 105 170 = 186 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l560_56099


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l560_56030

theorem quadratic_coefficient (b m : ℝ) : 
  b > 0 ∧ 
  (∀ x, x^2 + b*x + 72 = (x + m)^2 + 12) →
  b = 4 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l560_56030


namespace NUMINAMATH_CALUDE_dice_hidden_sum_l560_56006

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 4

/-- The sum of visible numbers -/
def visible_sum : ℕ := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6

/-- The number of visible faces -/
def num_visible : ℕ := 8

theorem dice_hidden_sum :
  (num_dice * die_sum) - visible_sum = 54 :=
by sorry

end NUMINAMATH_CALUDE_dice_hidden_sum_l560_56006


namespace NUMINAMATH_CALUDE_max_area_inscribed_isosceles_triangle_l560_56081

/-- An isosceles triangle inscribed in a circle --/
structure InscribedIsoscelesTriangle where
  /-- The radius of the circle --/
  radius : ℝ
  /-- The height of the triangle to its base --/
  height : ℝ

/-- The area of an inscribed isosceles triangle --/
def area (t : InscribedIsoscelesTriangle) : ℝ := sorry

/-- Theorem: The area of an isosceles triangle inscribed in a circle with radius 6
    is maximized when the height to the base is 9 --/
theorem max_area_inscribed_isosceles_triangle :
  ∀ t : InscribedIsoscelesTriangle,
  t.radius = 6 →
  area t ≤ area { radius := 6, height := 9 } :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_isosceles_triangle_l560_56081
