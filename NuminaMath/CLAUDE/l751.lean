import Mathlib

namespace average_weight_increase_l751_75104

theorem average_weight_increase (group_size : ℕ) (original_weight new_weight : ℝ) :
  group_size = 4 →
  original_weight = 65 →
  new_weight = 71 →
  (new_weight - original_weight) / group_size = 1.5 := by
  sorry

end average_weight_increase_l751_75104


namespace greatest_third_side_length_l751_75189

theorem greatest_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 11) :
  ∃ (c : ℕ), c = 15 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  ∀ (d : ℕ), d > c → ¬((a + b > d) ∧ (a + d > b) ∧ (b + d > a)) :=
by sorry

end greatest_third_side_length_l751_75189


namespace complex_fraction_simplification_l751_75172

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by
  sorry

end complex_fraction_simplification_l751_75172


namespace two_digit_numbers_property_l751_75190

-- Define a function to calculate the truncated square of a number
def truncatedSquare (n : ℕ) : ℕ := n * n + n * (n % 10) + (n % 10) * (n % 10)

-- Define the property for a two-digit number
def satisfiesProperty (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n = truncatedSquare (n / 10 + n % 10)

-- Theorem statement
theorem two_digit_numbers_property :
  satisfiesProperty 13 ∧
  satisfiesProperty 63 ∧
  63 - 13 = 50 ∧
  (∃ (x : ℕ), satisfiesProperty x ∧ x ≠ 13 ∧ x ≠ 63) :=
by sorry

end two_digit_numbers_property_l751_75190


namespace elevator_capacity_l751_75164

/-- Proves that the number of people in an elevator is 20, given the weight limit,
    average weight, and excess weight. -/
theorem elevator_capacity
  (weight_limit : ℝ)
  (average_weight : ℝ)
  (excess_weight : ℝ)
  (h1 : weight_limit = 1500)
  (h2 : average_weight = 80)
  (h3 : excess_weight = 100)
  : (weight_limit + excess_weight) / average_weight = 20 := by
  sorry

#check elevator_capacity

end elevator_capacity_l751_75164


namespace textbook_weight_difference_l751_75101

/-- The weight difference between Kelly's chemistry and geometry textbooks -/
theorem textbook_weight_difference :
  let chemistry_weight : ℚ := 712 / 100
  let geometry_weight : ℚ := 62 / 100
  chemistry_weight - geometry_weight = 650 / 100 :=
by sorry

end textbook_weight_difference_l751_75101


namespace age_of_17th_student_l751_75145

theorem age_of_17th_student
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_students_group1 : Nat)
  (average_age_group1 : ℝ)
  (num_students_group2 : Nat)
  (average_age_group2 : ℝ)
  (h1 : total_students = 17)
  (h2 : average_age_all = 17)
  (h3 : num_students_group1 = 5)
  (h4 : average_age_group1 = 14)
  (h5 : num_students_group2 = 9)
  (h6 : average_age_group2 = 16) :
  ℝ := by
  sorry

#check age_of_17th_student

end age_of_17th_student_l751_75145


namespace fruit_bag_probabilities_l751_75154

theorem fruit_bag_probabilities (apples oranges : ℕ) (h1 : apples = 7) (h2 : oranges = 1) :
  let total := apples + oranges
  (apples : ℚ) / total = 7 / 8 ∧ (oranges : ℚ) / total = 1 / 8 := by
sorry


end fruit_bag_probabilities_l751_75154


namespace parabola_intersection_l751_75142

/-- Prove that for a parabola y² = 2px (p > 0) with focus F on the x-axis, 
    if a line with slope angle π/4 passes through F and intersects the parabola at points A and B, 
    and the perpendicular bisector of AB passes through (0, 2), then p = 4/5. -/
theorem parabola_intersection (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∃ F : ℝ × ℝ, F.2 = 0 ∧ F.1 = p / 2) →
  (∀ x y : ℝ, y ^ 2 = 2 * p * x) →
  (∃ m b : ℝ, m = 1 ∧ A.2 = m * A.1 + b ∧ B.2 = m * B.1 + b) →
  (A.2 ^ 2 = 2 * p * A.1 ∧ B.2 ^ 2 = 2 * p * B.1) →
  ((A.1 + B.1) / 2 = 3 * p / 2 ∧ (A.2 + B.2) / 2 = p) →
  (∃ m' b' : ℝ, m' = -1 ∧ 2 = m' * 0 + b') →
  p = 4 / 5 := by
sorry

end parabola_intersection_l751_75142


namespace min_value_of_sum_reciprocals_l751_75161

theorem min_value_of_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 3/4 := by
  sorry

end min_value_of_sum_reciprocals_l751_75161


namespace fewer_heads_probability_l751_75137

/-- The number of coins being flipped -/
def n : ℕ := 8

/-- The probability of getting the same number of heads and tails -/
def p_equal : ℚ := (n.choose (n / 2)) / 2^n

/-- The probability of getting fewer heads than tails -/
def p_fewer_heads : ℚ := (1 - p_equal) / 2

theorem fewer_heads_probability :
  p_fewer_heads = 93 / 256 := by sorry

end fewer_heads_probability_l751_75137


namespace one_fifth_of_number_l751_75150

theorem one_fifth_of_number (x : ℚ) : (3/10 : ℚ) * x = 12 → (1/5 : ℚ) * x = 8 := by
  sorry

end one_fifth_of_number_l751_75150


namespace six_students_three_groups_arrangements_l751_75197

/-- The number of ways to divide n students into k equal groups -/
def divide_into_groups (n k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k topics -/
def assign_topics (k : ℕ) : ℕ := sorry

/-- The total number of arrangements for n students divided into k equal groups 
    and assigned to k different topics -/
def total_arrangements (n k : ℕ) : ℕ :=
  divide_into_groups n k * assign_topics k

theorem six_students_three_groups_arrangements :
  total_arrangements 6 3 = 540 := by sorry

end six_students_three_groups_arrangements_l751_75197


namespace inequality_proof_l751_75115

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq : 1/a + 1/b + 1/c ≥ a + b + c) :
  a + b + c ≥ 3 * a * b * c := by
  sorry

end inequality_proof_l751_75115


namespace dodgeball_team_size_l751_75167

/-- Given a dodgeball team with the following conditions:
  * The team scored 39 points total
  * One player (Emily) scored 23 points
  * Everyone else scored 2 points each
  This theorem proves that the total number of players on the team is 9. -/
theorem dodgeball_team_size :
  ∀ (total_points : ℕ) (emily_points : ℕ) (points_per_other : ℕ),
    total_points = 39 →
    emily_points = 23 →
    points_per_other = 2 →
    ∃ (team_size : ℕ),
      team_size = (total_points - emily_points) / points_per_other + 1 ∧
      team_size = 9 :=
by sorry

end dodgeball_team_size_l751_75167


namespace triangle_angle_measure_l751_75139

theorem triangle_angle_measure (D E F : ℝ) : 
  D = E →                         -- Two angles are congruent
  F = D + 40 →                    -- One angle is 40 degrees more than the congruent angles
  D + E + F = 180 →               -- Sum of angles in a triangle is 180 degrees
  F = 86.67 :=                    -- The measure of angle F is 86.67 degrees
by
  sorry

end triangle_angle_measure_l751_75139


namespace cone_volume_from_cylinder_l751_75135

/-- Given a cylinder with volume 81π cm³, prove that a cone with the same base radius
    and twice the height of the cylinder has a volume of 54π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 81 * π) → 
  ((1/3) * π * r^2 * (2*h) = 54 * π) :=
by sorry

end cone_volume_from_cylinder_l751_75135


namespace inscribed_rhombus_rectangle_perimeter_l751_75144

/-- A rhombus inscribed in a rectangle -/
structure InscribedRhombus where
  /-- The length of PB -/
  pb : ℝ
  /-- The length of BQ -/
  bq : ℝ
  /-- The length of PR (diagonal) -/
  pr : ℝ
  /-- The length of QS (diagonal) -/
  qs : ℝ
  /-- PB is positive -/
  pb_pos : pb > 0
  /-- BQ is positive -/
  bq_pos : bq > 0
  /-- PR is positive -/
  pr_pos : pr > 0
  /-- QS is positive -/
  qs_pos : qs > 0
  /-- PR ≠ QS (to ensure the rhombus is not a square) -/
  diag_neq : pr ≠ qs

/-- The perimeter of the rectangle containing the inscribed rhombus -/
def rectanglePerimeter (r : InscribedRhombus) : ℝ := sorry

/-- Theorem stating the perimeter of the rectangle for the given measurements -/
theorem inscribed_rhombus_rectangle_perimeter :
  let r : InscribedRhombus := {
    pb := 15,
    bq := 20,
    pr := 30,
    qs := 40,
    pb_pos := by norm_num,
    bq_pos := by norm_num,
    pr_pos := by norm_num,
    qs_pos := by norm_num,
    diag_neq := by norm_num
  }
  rectanglePerimeter r = 672 / 5 := by sorry

end inscribed_rhombus_rectangle_perimeter_l751_75144


namespace work_completion_time_l751_75179

/-- Proves the time taken to complete a work when two people work together -/
theorem work_completion_time (rahul_rate meena_rate : ℚ) 
  (hrahul : rahul_rate = 1 / 5)
  (hmeena : meena_rate = 1 / 10) :
  1 / (rahul_rate + meena_rate) = 10 / 3 := by
  sorry

#check work_completion_time

end work_completion_time_l751_75179


namespace investment_return_calculation_l751_75112

theorem investment_return_calculation (total_investment : ℝ) (combined_return_rate : ℝ) 
  (investment_1 : ℝ) (return_rate_1 : ℝ) (investment_2 : ℝ) :
  total_investment = 2000 →
  combined_return_rate = 0.22 →
  investment_1 = 500 →
  return_rate_1 = 0.07 →
  investment_2 = 1500 →
  let total_return := combined_return_rate * total_investment
  let return_1 := return_rate_1 * investment_1
  let return_2 := total_return - return_1
  return_2 / investment_2 = 0.27 := by sorry

end investment_return_calculation_l751_75112


namespace cube_volume_from_surface_area_l751_75130

-- Define the surface area of the cube
def surface_area : ℝ := 150

-- Theorem stating that a cube with surface area 150 has volume 125
theorem cube_volume_from_surface_area :
  ∃ (s : ℝ), s > 0 ∧ 6 * s^2 = surface_area ∧ s^3 = 125 :=
by sorry

end cube_volume_from_surface_area_l751_75130


namespace number_and_percentage_l751_75194

theorem number_and_percentage (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 35) : 
  (40/100) * N = 420 := by
sorry

end number_and_percentage_l751_75194


namespace clock_hand_overlaps_in_day_l751_75110

/-- Represents the number of overlaps between clock hands in a given time period -/
def clockHandOverlaps (hourRotations minuteRotations : ℕ) : ℕ :=
  minuteRotations - hourRotations

theorem clock_hand_overlaps_in_day :
  clockHandOverlaps 2 24 = 22 := by
  sorry

#eval clockHandOverlaps 2 24

end clock_hand_overlaps_in_day_l751_75110


namespace parallelogram_base_length_l751_75156

/-- Theorem: For a parallelogram with area 216 cm² and height 18 cm, the base length is 12 cm. -/
theorem parallelogram_base_length
  (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 216)
  (h_height : height = 18)
  (h_parallelogram : area = base * height) :
  base = 12 :=
by sorry

end parallelogram_base_length_l751_75156


namespace reflection_x_axis_example_l751_75119

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- The theorem stating that the reflection of (3, 4, -5) across the x-axis is (3, -4, 5) -/
theorem reflection_x_axis_example : 
  reflect_x_axis { x := 3, y := 4, z := -5 } = { x := 3, y := -4, z := 5 } := by
  sorry

end reflection_x_axis_example_l751_75119


namespace tangent_line_sum_l751_75182

/-- Given a function f: ℝ → ℝ with a tangent line at x=1 described by the equation 3x+y-4=0,
    prove that f(1) + f'(1) = -2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y : ℝ, y = f x → (3 * 1 + f 1 - 4 = 0 ∧ 3 * x + y - 4 = 0)) : 
    f 1 + (deriv f) 1 = -2 := by
  sorry

end tangent_line_sum_l751_75182


namespace ellipse_eccentricity_minimized_l751_75151

/-- The eccentricity of an ellipse passing through (3, 2) when a² + b² is minimized -/
theorem ellipse_eccentricity_minimized (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : (3:ℝ)^2 / a^2 + (2:ℝ)^2 / b^2 = 1) :
  let e := Real.sqrt (1 - b^2 / a^2)
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' > b' → (3:ℝ)^2 / a'^2 + (2:ℝ)^2 / b'^2 = 1 →
    a^2 + b^2 ≤ a'^2 + b'^2) →
  e = Real.sqrt 3 / 3 := by
sorry

end ellipse_eccentricity_minimized_l751_75151


namespace longest_length_is_three_smallest_square_is_1444_l751_75152

/-- A number is a perfect square with n identical non-zero last digits if it's
    a square and its last n digits in base 10 are the same and non-zero. -/
def is_perfect_square_with_n_identical_last_digits (x n : ℕ) : Prop :=
  ∃ k : ℕ, x = k^2 ∧
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧
  ∀ i : ℕ, i < n → (x / 10^i) % 10 = d

/-- The longest possible length for which a perfect square ends with
    n identical non-zero digits is 3. -/
theorem longest_length_is_three :
  (∀ n : ℕ, ∃ x : ℕ, is_perfect_square_with_n_identical_last_digits x n) →
  (∀ m : ℕ, m > 3 → ¬∃ x : ℕ, is_perfect_square_with_n_identical_last_digits x m) :=
sorry

/-- The smallest perfect square with 3 identical non-zero last digits is 1444. -/
theorem smallest_square_is_1444 :
  is_perfect_square_with_n_identical_last_digits 1444 3 ∧
  ∀ x : ℕ, x < 1444 → ¬is_perfect_square_with_n_identical_last_digits x 3 :=
sorry

end longest_length_is_three_smallest_square_is_1444_l751_75152


namespace min_value_of_a_l751_75149

theorem min_value_of_a (a : ℝ) (h_a : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) → 
  a ≥ 4 ∧ ∀ b : ℝ, b > 0 → (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + b / y) ≥ 9) → b ≥ a :=
by sorry

end min_value_of_a_l751_75149


namespace chloe_carrot_problem_l751_75111

theorem chloe_carrot_problem :
  ∀ (initial_carrots picked_next_day final_carrots thrown_out : ℕ),
    initial_carrots = 48 →
    picked_next_day = 42 →
    final_carrots = 45 →
    initial_carrots - thrown_out + picked_next_day = final_carrots →
    thrown_out = 45 := by
  sorry

end chloe_carrot_problem_l751_75111


namespace optimal_walking_distance_ratio_l751_75113

-- Define the problem setup
structure TravelProblem where
  totalDistance : ℝ
  speedA : ℝ
  speedB : ℝ
  speedC : ℝ
  mk_travel_problem : totalDistance > 0 ∧ speedA > 0 ∧ speedB > 0 ∧ speedC > 0

-- Define the optimal solution
def OptimalSolution (p : TravelProblem) :=
  ∃ (x : ℝ),
    0 < x ∧ x < p.totalDistance ∧
    (p.totalDistance - x) / p.speedA = x / (2 * p.speedC) + (p.totalDistance - x) / p.speedC

-- Theorem statement
theorem optimal_walking_distance_ratio 
  (p : TravelProblem) 
  (h_speeds : p.speedA = 4 ∧ p.speedB = 5 ∧ p.speedC = 12) 
  (h_optimal : OptimalSolution p) : 
  ∃ (distA distB : ℝ),
    distA > 0 ∧ distB > 0 ∧
    distA / distB = 17 / 10 ∧
    distA + distB = p.totalDistance :=
  sorry

end optimal_walking_distance_ratio_l751_75113


namespace fold_equilateral_triangle_l751_75196

-- Define an equilateral triangle
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the folding operation
def FoldTriangle (A B C P Q : ℝ × ℝ) : Prop :=
  P.1 = B.1 + 7 * (A.1 - B.1) / 10 ∧
  P.2 = B.2 + 7 * (A.2 - B.2) / 10 ∧
  Q.1 = C.1 + 7 * (A.1 - C.1) / 10 ∧
  Q.2 = C.2 + 7 * (A.2 - C.2) / 10

theorem fold_equilateral_triangle :
  ∀ (A B C P Q : ℝ × ℝ),
  EquilateralTriangle A B C →
  dist A B = 10 →
  FoldTriangle A B C P Q →
  (dist P Q)^2 = 9 := by
  sorry

end fold_equilateral_triangle_l751_75196


namespace sqrt_expressions_l751_75132

theorem sqrt_expressions :
  (2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2) ∧
  (Real.sqrt ((-3)^2) ≠ -3) ∧
  (Real.sqrt 24 / Real.sqrt 6 ≠ 4) ∧
  (Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5) :=
by sorry

end sqrt_expressions_l751_75132


namespace negation_of_universal_quantifier_negation_of_proposition_l751_75174

theorem negation_of_universal_quantifier (P : ℝ → Prop) :
  (¬ ∀ x ∈ Set.Ici 1, P x) ↔ (∃ x ∈ Set.Ici 1, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x ∈ Set.Ici 1, x^2 - 2*x + 1 ≥ 0) ↔ (∃ x ∈ Set.Ici 1, x^2 - 2*x + 1 < 0) :=
by sorry

end negation_of_universal_quantifier_negation_of_proposition_l751_75174


namespace zoo_trip_result_l751_75168

def zoo_trip (initial_students_class1 initial_students_class2 parent_chaperones teachers students_left chaperones_left : ℕ) : ℕ :=
  let total_initial_students := initial_students_class1 + initial_students_class2
  let total_initial_adults := parent_chaperones + teachers
  let remaining_students := total_initial_students - students_left
  let remaining_chaperones := parent_chaperones - chaperones_left
  remaining_students + remaining_chaperones + teachers

theorem zoo_trip_result :
  zoo_trip 10 10 5 2 10 2 = 15 := by
  sorry

end zoo_trip_result_l751_75168


namespace plane_equation_l751_75138

theorem plane_equation (s t x y z : ℝ) : 
  (∃ (s t : ℝ), x = 3 + 2*s - 3*t ∧ y = 1 + s ∧ z = 4 - 3*s + t) ↔ 
  (x - 7*y + 3*z - 8 = 0) :=
by sorry

end plane_equation_l751_75138


namespace total_matches_is_seventeen_l751_75165

/-- Calculates the number of matches in a round-robin tournament for n teams -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents a football competition with the given structure -/
structure FootballCompetition where
  totalTeams : ℕ
  groupSize : ℕ
  numGroups : ℕ
  semiFinalistPerGroup : ℕ
  semiFinalsLegs : ℕ
  finalMatches : ℕ

/-- Calculates the total number of matches in the competition -/
def totalMatches (comp : FootballCompetition) : ℕ :=
  (comp.numGroups * roundRobinMatches comp.groupSize) +
  (comp.numGroups * comp.semiFinalistPerGroup * comp.semiFinalsLegs / 2) +
  comp.finalMatches

/-- The specific football competition described in the problem -/
def specificCompetition : FootballCompetition :=
  { totalTeams := 8
  , groupSize := 4
  , numGroups := 2
  , semiFinalistPerGroup := 2
  , semiFinalsLegs := 2
  , finalMatches := 1 }

theorem total_matches_is_seventeen :
  totalMatches specificCompetition = 17 := by
  sorry

end total_matches_is_seventeen_l751_75165


namespace election_winner_margin_l751_75134

theorem election_winner_margin (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (992 : ℚ) / total_votes = 62 / 100) : 
  992 - (total_votes - 992) = 384 := by
sorry

end election_winner_margin_l751_75134


namespace final_surface_area_l751_75147

/-- Represents the cube structure after modifications --/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCornerCubes : Nat
  removedCentralCube : Nat
  removedCenterUnits : Bool

/-- Calculates the surface area of the modified cube structure --/
def surfaceArea (cube : ModifiedCube) : Nat :=
  let totalSmallCubes := (cube.initialSize / cube.smallCubeSize) ^ 3
  let remainingCubes := totalSmallCubes - cube.removedCornerCubes - cube.removedCentralCube
  let initialSurfaceArea := remainingCubes * (6 * cube.smallCubeSize ^ 2)
  let additionalInternalSurface := if cube.removedCenterUnits then remainingCubes * 6 else 0
  initialSurfaceArea + additionalInternalSurface

/-- The main theorem stating the surface area of the final structure --/
theorem final_surface_area :
  let cube : ModifiedCube := {
    initialSize := 12,
    smallCubeSize := 3,
    removedCornerCubes := 8,
    removedCentralCube := 1,
    removedCenterUnits := true
  }
  surfaceArea cube = 3300 := by
  sorry

end final_surface_area_l751_75147


namespace intersecting_lines_angles_l751_75136

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function for alternate interior angles
def alternate_interior_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Define a function for corresponding angles
def corresponding_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Define a function for consecutive interior angles
def consecutive_interior_angles (l1 l2 : Line) (t : Line) : Angle × Angle :=
  sorry

-- Main theorem
theorem intersecting_lines_angles (l1 l2 t : Line) 
  (h : ¬ are_parallel l1 l2) : 
  ∃ (a1 a2 : Angle), 
    (alternate_interior_angles l1 l2 t = (a1, a2) ∧ a1.measure ≠ a2.measure) ∨
    (corresponding_angles l1 l2 t = (a1, a2) ∧ a1.measure ≠ a2.measure) ∨
    (consecutive_interior_angles l1 l2 t = (a1, a2) ∧ a1.measure + a2.measure ≠ 180) :=
  sorry

end intersecting_lines_angles_l751_75136


namespace quadratic_function_b_range_l751_75105

/-- Given a quadratic function f(x) = x^2 + 2bx + c where b and c are real numbers,
    if f(1) = 0 and the equation f(x) + x + b = 0 has two real roots
    in the intervals (-3,-2) and (0,1), then b is in the open interval (1/5, 5/7). -/
theorem quadratic_function_b_range (b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*b*x + c
  (f 1 = 0) →
  (∃ x₁ x₂, -3 < x₁ ∧ x₁ < -2 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
    f x₁ + x₁ + b = 0 ∧ f x₂ + x₂ + b = 0) →
  1/5 < b ∧ b < 5/7 :=
by sorry

end quadratic_function_b_range_l751_75105


namespace exactly_one_true_l751_75116

-- Define the three propositions
def prop1 : Prop := ∀ x : ℝ, x^4 > x^2

def prop2 : Prop := (∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q))

def prop3 : Prop := (¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0))

-- Theorem statement
theorem exactly_one_true : (prop1 ∨ prop2 ∨ prop3) ∧ ¬(prop1 ∧ prop2) ∧ ¬(prop1 ∧ prop3) ∧ ¬(prop2 ∧ prop3) :=
sorry

end exactly_one_true_l751_75116


namespace tower_surface_area_l751_75180

/-- Represents a cube in the tower --/
structure Cube where
  volume : ℕ
  sideLength : ℕ
  deriving Repr

/-- Represents the tower of cubes --/
def Tower : List Cube := [
  { volume := 343, sideLength := 7 },
  { volume := 125, sideLength := 5 },
  { volume := 27,  sideLength := 3 },
  { volume := 64,  sideLength := 4 },
  { volume := 1,   sideLength := 1 }
]

/-- Calculates the visible surface area of a cube in the tower --/
def visibleSurfaceArea (cube : Cube) (aboveCube : Option Cube) : ℕ := sorry

/-- Calculates the total visible surface area of the tower --/
def totalVisibleSurfaceArea (tower : List Cube) : ℕ := sorry

/-- Theorem stating that the total visible surface area of the tower is 400 square units --/
theorem tower_surface_area : totalVisibleSurfaceArea Tower = 400 := by sorry

end tower_surface_area_l751_75180


namespace average_age_problem_l751_75177

theorem average_age_problem (a c : ℝ) : 
  (a + c) / 2 = 29 →
  ((a + c) + 26) / 3 = 28 := by
sorry

end average_age_problem_l751_75177


namespace initial_money_calculation_l751_75126

theorem initial_money_calculation (initial_money : ℚ) : 
  (2/5 : ℚ) * initial_money = 200 → initial_money = 500 := by
  sorry

end initial_money_calculation_l751_75126


namespace arithmetic_fraction_difference_l751_75175

theorem arithmetic_fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end arithmetic_fraction_difference_l751_75175


namespace oldest_sister_clothing_amount_l751_75141

/-- Proves that the oldest sister's clothing amount is the difference between Nicole's final amount and the sum of the younger sisters' amounts. -/
theorem oldest_sister_clothing_amount 
  (nicole_initial : ℕ) 
  (nicole_final : ℕ) 
  (first_older_sister : ℕ) 
  (next_oldest_sister : ℕ) 
  (h1 : nicole_initial = 10)
  (h2 : first_older_sister = nicole_initial / 2)
  (h3 : next_oldest_sister = nicole_initial + 2)
  (h4 : nicole_final = 36) :
  nicole_final - (nicole_initial + first_older_sister + next_oldest_sister) = 9 := by
sorry

end oldest_sister_clothing_amount_l751_75141


namespace problem_solution_l751_75128

noncomputable section

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ, Real.cos θ)
def O : ℝ × ℝ := (0, 0)

def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem problem_solution (θ : ℝ) :
  (vec_length (vec A (C θ)) = vec_length (vec B (C θ)) → Real.tan θ = 1/2) ∧
  (dot_product (vec O A + 2 • vec O B) (vec O (C θ)) = 1 → Real.sin θ * Real.cos θ = -3/8) := by
  sorry

end

end problem_solution_l751_75128


namespace factorization_theorem_1_factorization_theorem_2_l751_75140

-- Theorem 1
theorem factorization_theorem_1 (m n : ℝ) : m^3*n - 9*m*n = m*n*(m+3)*(m-3) := by
  sorry

-- Theorem 2
theorem factorization_theorem_2 (a : ℝ) : a^3 + a - 2*a^2 = a*(a-1)^2 := by
  sorry

end factorization_theorem_1_factorization_theorem_2_l751_75140


namespace field_ratio_l751_75199

/-- Given a rectangular field with perimeter 240 meters and width 50 meters,
    prove that the ratio of length to width is 7:5 -/
theorem field_ratio (perimeter width length : ℝ) : 
  perimeter = 240 ∧ width = 50 ∧ perimeter = 2 * (length + width) →
  length / width = 7 / 5 := by
  sorry

end field_ratio_l751_75199


namespace marco_new_cards_l751_75129

/-- Given a total number of cards, calculate the number of new cards obtained by trading
    one-fifth of the duplicate cards, where duplicates are one-fourth of the total. -/
def new_cards (total : ℕ) : ℕ :=
  let duplicates := total / 4
  duplicates / 5

theorem marco_new_cards :
  new_cards 500 = 25 := by sorry

end marco_new_cards_l751_75129


namespace valid_numbers_divisible_by_36_l751_75162

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 52000 + a * 100 + 20 + b

def is_divisible_by_36 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 36 * k

theorem valid_numbers_divisible_by_36 :
  ∀ n : ℕ, is_valid_number n ∧ is_divisible_by_36 n ↔ 
    n = 52524 ∨ n = 52128 ∨ n = 52020 ∨ n = 52920 :=
by sorry

end valid_numbers_divisible_by_36_l751_75162


namespace unused_streetlights_l751_75166

/-- Given the number of streetlights bought by the New York City Council, 
    the number of squares in New York, and the number of streetlights per square, 
    calculate the number of unused streetlights. -/
theorem unused_streetlights (total : ℕ) (squares : ℕ) (per_square : ℕ) 
    (h1 : total = 200) (h2 : squares = 15) (h3 : per_square = 12) : 
  total - squares * per_square = 20 := by
  sorry

end unused_streetlights_l751_75166


namespace S_value_S_approx_l751_75122

/-- Define the sum S as a function of n, where n is the number of terms -/
def S (n : ℕ) : ℚ :=
  let rec aux (k : ℕ) : ℚ :=
    if k = 0 then 5005
    else (5005 - k : ℚ) + (1/2) * aux (k-1)
  aux n

/-- The main theorem stating that S(5000) is equal to 5009 - (1/2^5000) -/
theorem S_value : S 5000 = 5009 - (1/2)^5000 := by
  sorry

/-- Corollary stating that S(5000) is approximately equal to 5009 -/
theorem S_approx : abs (S 5000 - 5009) < 1 := by
  sorry

end S_value_S_approx_l751_75122


namespace specific_arithmetic_sequence_sum_l751_75153

/-- The sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a1 : ℤ) (an : ℤ) (d : ℤ) : ℤ :=
  let n : ℤ := (an - a1) / d + 1
  n * (a1 + an) / 2

/-- Theorem: The sum of the specific arithmetic sequence is -440 -/
theorem specific_arithmetic_sequence_sum :
  arithmeticSequenceSum (-41) 1 2 = -440 := by
  sorry

end specific_arithmetic_sequence_sum_l751_75153


namespace defeat_dragon_l751_75158

def dragonHeads (n : ℕ) : ℕ → ℕ
  | 0 => n
  | m + 1 => 
    let remaining := dragonHeads n m - 5
    if remaining ≤ 5 then 0
    else remaining + (remaining % 9)

theorem defeat_dragon (initialHeads : ℕ) (swings : ℕ) : 
  initialHeads = 198 →
  (∀ k < swings, dragonHeads initialHeads k > 5) →
  dragonHeads initialHeads swings ≤ 5 →
  swings = 40 :=
sorry

#check defeat_dragon

end defeat_dragon_l751_75158


namespace carla_cooking_time_l751_75184

def total_time (sharpening_time peeling_time chopping_time first_break fruits_time second_break salad_time : ℝ) : ℝ :=
  sharpening_time + peeling_time + chopping_time + first_break + fruits_time + second_break + salad_time

theorem carla_cooking_time : ∃ (total : ℝ),
  let sharpening_time : ℝ := 15
  let peeling_time : ℝ := 3 * sharpening_time
  let chopping_time : ℝ := (1 / 4) * peeling_time
  let first_break : ℝ := 5
  let fruits_time : ℝ := 2 * chopping_time
  let second_break : ℝ := 10
  let previous_activities_time : ℝ := sharpening_time + peeling_time + chopping_time + first_break + fruits_time + second_break
  let salad_time : ℝ := (3 / 5) * previous_activities_time
  total = total_time sharpening_time peeling_time chopping_time first_break fruits_time second_break salad_time ∧
  total = 174.6 := by
    sorry

end carla_cooking_time_l751_75184


namespace triangle_perimeter_ratio_l751_75185

theorem triangle_perimeter_ratio (X Y Z D J : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : 
  let XZ := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let YZ := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let XY := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  -- XYZ is a right triangle with hypotenuse XY
  (X.1 - Z.1) * (Y.1 - Z.1) + (X.2 - Z.2) * (Y.2 - Z.2) = 0 →
  -- XZ = 8, YZ = 15
  XZ = 8 →
  YZ = 15 →
  -- ZD is the altitude to XY
  (X.1 - Y.1) * (D.1 - Z.1) + (X.2 - Y.2) * (D.2 - Z.2) = 0 →
  -- ω is the circle with ZD as diameter
  ω = {P : ℝ × ℝ | (P.1 - Z.1)^2 + (P.2 - Z.2)^2 = (D.1 - Z.1)^2 + (D.2 - Z.2)^2} →
  -- J is outside XYZ
  (J.1 - X.1) * (Y.2 - X.2) - (J.2 - X.2) * (Y.1 - X.1) ≠ 0 →
  -- XJ and YJ are tangent to ω
  ∃ P ∈ ω, (J.1 - X.1) * (P.1 - X.1) + (J.2 - X.2) * (P.2 - X.2) = 0 →
  ∃ Q ∈ ω, (J.1 - Y.1) * (Q.1 - Y.1) + (J.2 - Y.2) * (Q.2 - Y.2) = 0 →
  -- The ratio of the perimeter of XYJ to XY is 30/17
  (Real.sqrt ((X.1 - J.1)^2 + (X.2 - J.2)^2) + 
   Real.sqrt ((Y.1 - J.1)^2 + (Y.2 - J.2)^2) + XY) / XY = 30/17 :=
by sorry

end triangle_perimeter_ratio_l751_75185


namespace largest_square_tile_size_l751_75146

theorem largest_square_tile_size 
  (length width : ℕ) 
  (h_length : length = 378) 
  (h_width : width = 595) : 
  ∃ (tile_size : ℕ), 
    tile_size = Nat.gcd length width ∧ 
    tile_size = 7 ∧
    length % tile_size = 0 ∧ 
    width % tile_size = 0 ∧
    ∀ (larger_size : ℕ), larger_size > tile_size → 
      length % larger_size ≠ 0 ∨ width % larger_size ≠ 0 :=
by sorry

end largest_square_tile_size_l751_75146


namespace hyperbola_y_coordinate_comparison_l751_75127

/-- Given two points on a hyperbola, prove that the y-coordinate of the point with smaller x-coordinate is greater -/
theorem hyperbola_y_coordinate_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h_positive : k > 0)
  (h_point_A : y₁ = k / 2)
  (h_point_B : y₂ = k / 3) :
  y₁ > y₂ := by
  sorry

end hyperbola_y_coordinate_comparison_l751_75127


namespace total_spent_with_tip_l751_75160

def lunch_cost : ℝ := 60.50
def tip_percentage : ℝ := 0.20

theorem total_spent_with_tip : 
  lunch_cost * (1 + tip_percentage) = 72.60 := by
  sorry

end total_spent_with_tip_l751_75160


namespace playlist_song_length_l751_75183

theorem playlist_song_length 
  (total_songs : Nat) 
  (song1_length : Nat) 
  (song2_length : Nat) 
  (total_playtime : Nat) 
  (playlist_repeats : Nat) 
  (h1 : total_songs = 3) 
  (h2 : song1_length = 3) 
  (h3 : song2_length = 3) 
  (h4 : total_playtime = 40) 
  (h5 : playlist_repeats = 5) :
  ∃ (song3_length : Nat), 
    song1_length + song2_length + song3_length = total_playtime / playlist_repeats ∧ 
    song3_length = 2 := by
  sorry

end playlist_song_length_l751_75183


namespace BD_range_l751_75163

/-- Triangle ABC with median AD to side BC -/
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)
  (BD : ℝ)
  (is_median : BD = BC / 2)
  (AB_eq : AB = 5)
  (AC_eq : AC = 7)

/-- The length of BD in a triangle ABC with median AD to side BC, 
    where AB = 5 and AC = 7, satisfies 1 < BD < 6 -/
theorem BD_range (t : Triangle) : 1 < t.BD ∧ t.BD < 6 := by
  sorry

end BD_range_l751_75163


namespace article_cost_price_l751_75118

theorem article_cost_price 
  (C M : ℝ) 
  (h1 : 0.95 * M = 1.4 * C) 
  (h2 : 0.95 * M = 70) : 
  C = 50 := by
sorry

end article_cost_price_l751_75118


namespace two_lines_exist_l751_75143

-- Define the lines given in the problem
def line_l1 (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0
def line_l2 (x y : ℝ) : Prop := x + y + 2 = 0
def line_perp (x y : ℝ) : Prop := 2 * x - y + 7 = 0

-- Define the intersection point of l1 and l2
def intersection_point : ℝ × ℝ := (-1, -1)

-- Define the given point
def given_point : ℝ × ℝ := (-3, 1)

-- Define the equations of the lines we need to prove
def line_L1 (x y : ℝ) : Prop := x + 2 * y + 3 = 0
def line_L2 (x y : ℝ) : Prop := x - 3 * y + 6 = 0

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define the theorem
theorem two_lines_exist :
  ∃ (L1 L2 : ℝ → ℝ → Prop),
    (∀ x y, line_l1 x y ∧ line_l2 x y → L1 x y) ∧
    (∃ m1 m2, perpendicular m1 m2 ∧
      (∀ x y, line_perp x y ↔ y = m1 * x + 7/2) ∧
      (∀ x y, L1 x y ↔ y = m2 * x + (intersection_point.2 - m2 * intersection_point.1))) ∧
    L1 = line_L1 ∧
    L2 given_point.1 given_point.2 ∧
    (∃ a b, a + b = -4 ∧ ∀ x y, L2 x y ↔ x / a + y / b = 1) ∧
    L2 = line_L2 :=
  sorry

end two_lines_exist_l751_75143


namespace circle_radii_l751_75195

theorem circle_radii (r R : ℝ) (hr : r > 0) (hR : R > 0) : 
  ∃ (circumscribed_radius inscribed_radius : ℝ),
    circumscribed_radius = Real.sqrt (r * R) ∧
    inscribed_radius = (Real.sqrt (r * R) * (Real.sqrt R + Real.sqrt r - Real.sqrt (R + r))) / Real.sqrt (R + r) := by
  sorry

end circle_radii_l751_75195


namespace imaginary_part_of_z_l751_75170

open Complex

theorem imaginary_part_of_z : ∃ (z : ℂ), z = (1 + I)^2 + I^2010 ∧ z.im = 2 := by sorry

end imaginary_part_of_z_l751_75170


namespace leanna_money_l751_75125

/-- The amount of money Leanna has to spend -/
def total_money : ℕ := 37

/-- The price of a CD -/
def cd_price : ℕ := 14

/-- The price of a cassette -/
def cassette_price : ℕ := 9

/-- Leanna can spend all her money on two CDs and a cassette -/
axiom scenario1 : 2 * cd_price + cassette_price = total_money

/-- Leanna can buy one CD and two cassettes and have $5 left over -/
axiom scenario2 : cd_price + 2 * cassette_price + 5 = total_money

theorem leanna_money : total_money = 37 := by
  sorry

end leanna_money_l751_75125


namespace point_above_with_distance_l751_75192

/-- Given two points P(3, a) and Q(3, 4) in a Cartesian coordinate system,
    if P is above Q and the distance between P and Q is 3,
    then the y-coordinate of P (which is a) equals 7. -/
theorem point_above_with_distance (a : ℝ) :
  a > 4 →  -- P is above Q
  (3 - 3)^2 + (a - 4)^2 = 3^2 →  -- Distance formula
  a = 7 := by
sorry

end point_above_with_distance_l751_75192


namespace tank_filling_time_l751_75191

/-- The time it takes for A, B, and C to fill the tank together -/
def combined_time : ℝ := 17.14285714285714

/-- The time it takes for B to fill the tank alone -/
def b_time : ℝ := 20

/-- The time it takes for C to empty the tank -/
def c_time : ℝ := 40

/-- The time it takes for A to fill the tank alone -/
def a_time : ℝ := 30

theorem tank_filling_time :
  (1 / a_time + 1 / b_time - 1 / c_time) = (1 / combined_time) := by sorry

end tank_filling_time_l751_75191


namespace balls_per_color_l751_75108

theorem balls_per_color 
  (total_balls : ℕ) 
  (num_colors : ℕ) 
  (h1 : total_balls = 350) 
  (h2 : num_colors = 10) 
  (h3 : total_balls % num_colors = 0) : 
  total_balls / num_colors = 35 := by
sorry

end balls_per_color_l751_75108


namespace total_profit_calculation_total_profit_is_630_l751_75176

/-- Calculates the total profit given investment conditions and A's share of profit -/
theorem total_profit_calculation (a_initial : ℕ) (b_initial : ℕ) (a_withdrawal : ℕ) (b_addition : ℕ) (months : ℕ) (a_share : ℕ) : ℕ :=
  let a_investment_months := a_initial * 8 + (a_initial - a_withdrawal) * 4
  let b_investment_months := b_initial * 8 + (b_initial + b_addition) * 4
  let total_ratio_parts := a_investment_months + b_investment_months
  let total_profit := a_share * total_ratio_parts / a_investment_months
  total_profit

/-- The total profit at the end of the year is 630 Rs -/
theorem total_profit_is_630 :
  total_profit_calculation 3000 4000 1000 1000 12 240 = 630 := by
  sorry

end total_profit_calculation_total_profit_is_630_l751_75176


namespace sum_of_xyz_l751_75188

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 40) (hxz : x * z = 80) (hyz : y * z = 120) :
  x + y + z = 22 * Real.sqrt 15 / 3 := by
sorry

end sum_of_xyz_l751_75188


namespace transistors_2010_count_l751_75157

/-- The number of transistors in a CPU triples every two years -/
def tripling_period : ℕ := 2

/-- The initial number of transistors in 1990 -/
def initial_transistors : ℕ := 500000

/-- The number of years between 1990 and 2010 -/
def years_passed : ℕ := 20

/-- Calculate the number of transistors in 2010 -/
def transistors_2010 : ℕ := initial_transistors * (3 ^ (years_passed / tripling_period))

/-- Theorem stating that the number of transistors in 2010 is 29,524,500,000 -/
theorem transistors_2010_count : transistors_2010 = 29524500000 := by
  sorry

end transistors_2010_count_l751_75157


namespace largest_fraction_sum_inequality_l751_75114

theorem largest_fraction_sum_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≥ b) (hac : a ≥ c)
  (h_eq : a / b = c / d) : 
  a + d > b + c := by
sorry

end largest_fraction_sum_inequality_l751_75114


namespace problem_statements_l751_75109

theorem problem_statements :
  (∀ x : ℝ, (x^2 - 4*x + 3 = 0 → x = 3) ↔ (x ≠ 3 → x^2 - 4*x + 3 ≠ 0)) ∧
  (¬(∀ x : ℝ, x^2 - x + 2 > 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≤ 0)) ∧
  (∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  (∀ x : ℝ, x > -1 → x^2 + 4*x + 3 > 0) ∧
  (∃ x : ℝ, x^2 + 4*x + 3 > 0 ∧ x ≤ -1) :=
by sorry

end problem_statements_l751_75109


namespace find_n_l751_75193

theorem find_n : ∃ n : ℤ, 5^2 - 7 = 3^3 + n ∧ n = -9 := by sorry

end find_n_l751_75193


namespace orthocenter_of_specific_triangle_l751_75100

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC is (13/7, 41/14, 55/7) -/
theorem orthocenter_of_specific_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (13/7, 41/14, 55/7) := by sorry

end orthocenter_of_specific_triangle_l751_75100


namespace problem_l751_75133

theorem problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1/a + 1/b) :
  (a + b ≥ 2) ∧ ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end problem_l751_75133


namespace intersection_line_passes_through_circles_l751_75102

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4*x
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

-- Define the line
def line (x y : ℝ) : Prop := y = -x

-- Theorem statement
theorem intersection_line_passes_through_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end intersection_line_passes_through_circles_l751_75102


namespace shara_shell_collection_l751_75121

/-- Calculates the total number of shells Shara has after her vacations -/
def total_shells (initial : ℕ) (vacation1 : ℕ) (vacation2 : ℕ) (vacation3 : ℕ) : ℕ :=
  initial + vacation1 + vacation2 + vacation3

/-- The number of shells Shara collected during her first vacation -/
def first_vacation : ℕ := 5 * 3 + 6

/-- The number of shells Shara collected during her second vacation -/
def second_vacation : ℕ := 4 * 2 + 7

/-- The number of shells Shara collected during her third vacation -/
def third_vacation : ℕ := 8 + 4 + 3 * 2

theorem shara_shell_collection :
  total_shells 20 first_vacation second_vacation third_vacation = 74 := by
  sorry

end shara_shell_collection_l751_75121


namespace vasya_fool_count_l751_75103

theorem vasya_fool_count (misha petya kolya vasya : ℕ) : 
  misha + petya + kolya + vasya = 16 →
  misha ≥ 1 → petya ≥ 1 → kolya ≥ 1 → vasya ≥ 1 →
  petya + kolya = 9 →
  misha > petya → misha > kolya → misha > vasya →
  vasya = 1 := by
sorry

end vasya_fool_count_l751_75103


namespace combined_average_marks_l751_75187

/-- Given two classes with the specified number of students and average marks,
    calculate the combined average mark of all students in both classes. -/
theorem combined_average_marks
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ)
  (h1 : n1 = 55) (h2 : n2 = 48) (h3 : avg1 = 60) (h4 : avg2 = 58) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℝ) = (55 * 60 + 48 * 58) / (55 + 48 : ℝ) :=
by sorry

end combined_average_marks_l751_75187


namespace equation_solution_l751_75107

theorem equation_solution : ∃ x : ℝ, (x / 3 + (30 - x) / 2 = 5) ∧ (x = 60) := by
  sorry

end equation_solution_l751_75107


namespace sum_of_medians_l751_75171

def player_A_median : ℝ := 36
def player_B_median : ℝ := 27

theorem sum_of_medians : player_A_median + player_B_median = 63 := by
  sorry

end sum_of_medians_l751_75171


namespace angle_bisector_theorem_l751_75123

theorem angle_bisector_theorem (a b : Real) (h : b - a = 100) :
  (b / 2) - (a / 2) = 50 := by sorry

end angle_bisector_theorem_l751_75123


namespace lucy_groceries_l751_75155

/-- The number of packs of groceries Lucy bought -/
def total_groceries (cookies cake chocolate : ℕ) : ℕ :=
  cookies + cake + chocolate

/-- Theorem stating that Lucy bought 42 packs of groceries in total -/
theorem lucy_groceries : total_groceries 4 22 16 = 42 := by
  sorry

end lucy_groceries_l751_75155


namespace circle_center_coordinate_sum_l751_75159

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 15, 
    prove that the sum of the x and y coordinates of its center is 7. -/
theorem circle_center_coordinate_sum : 
  ∀ (h k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 8*k + 15)) →
  h + k = 7 := by
sorry

end circle_center_coordinate_sum_l751_75159


namespace min_sum_given_product_min_sum_equality_case_l751_75186

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 8*b = a*b → a + b ≥ 18 := by
  sorry

-- The equality case
theorem min_sum_equality_case (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + 8*b = a*b → (a + b = 18 ↔ a = 12 ∧ b = 6) := by
  sorry

end min_sum_given_product_min_sum_equality_case_l751_75186


namespace vector_magnitude_l751_75178

/-- Given plane vectors a, b, and c, if (a + b) is parallel to c, then the magnitude of c is 2√17. -/
theorem vector_magnitude (a b c : ℝ × ℝ) (h : ∃ (t : ℝ), a + b = t • c) : 
  a = (-1, 1) → b = (2, 3) → c.1 = -2 → ‖c‖ = 2 * Real.sqrt 17 := by
  sorry

end vector_magnitude_l751_75178


namespace sphere_cross_section_area_l751_75181

theorem sphere_cross_section_area (R d : ℝ) (h1 : R = 3) (h2 : d = 2) :
  let r := (R^2 - d^2).sqrt
  π * r^2 = 5 * π :=
by sorry

end sphere_cross_section_area_l751_75181


namespace woodworker_extra_parts_l751_75169

/-- A woodworker's production scenario -/
structure WoodworkerProduction where
  normal_days : ℕ
  normal_parts : ℕ
  productivity_increase : ℕ
  new_days : ℕ

/-- Calculate the extra parts made by the woodworker -/
def extra_parts (w : WoodworkerProduction) : ℕ :=
  let normal_daily := w.normal_parts / w.normal_days
  let new_daily := normal_daily + w.productivity_increase
  new_daily * w.new_days - w.normal_parts

/-- Theorem stating the extra parts made by the woodworker -/
theorem woodworker_extra_parts :
  ∀ (w : WoodworkerProduction),
    w.normal_days = 24 ∧
    w.normal_parts = 360 ∧
    w.productivity_increase = 5 ∧
    w.new_days = 22 →
    extra_parts w = 80 := by
  sorry

end woodworker_extra_parts_l751_75169


namespace unique_solution_logarithmic_equation_l751_75148

theorem unique_solution_logarithmic_equation :
  ∃! (x : ℝ), x > 0 ∧ x^(Real.log x / Real.log 10) = x^4 / 10000 := by
  sorry

end unique_solution_logarithmic_equation_l751_75148


namespace two_red_one_blue_probability_l751_75120

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 3

theorem two_red_one_blue_probability :
  (Nat.choose red_marbles 2 * blue_marbles) / Nat.choose total_marbles selected_marbles = 44 / 95 :=
by sorry

end two_red_one_blue_probability_l751_75120


namespace monkey_peach_division_l751_75198

theorem monkey_peach_division (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, 100 = n * k + 10) →
  (∃ m : ℕ, 1000 = n * m + 10) :=
by sorry

end monkey_peach_division_l751_75198


namespace triangle_min_ab_value_l751_75131

theorem triangle_min_ab_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (2 * c * Real.cos B = 2 * a + b) →
  (1 / 2 * c = 1 / 2 * a * b * Real.sin C) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' * b' ≥ a * b) →
  a * b = 4 :=
sorry

end triangle_min_ab_value_l751_75131


namespace convenience_store_choices_l751_75124

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1 : Nat) (set2 : Nat) : Nat :=
  set1 * set2

/-- Theorem: Choosing one item from a set of 4 and one from a set of 3 results in 12 possibilities -/
theorem convenience_store_choices :
  choose_one_from_each 4 3 = 12 := by
  sorry

end convenience_store_choices_l751_75124


namespace division_scaling_l751_75173

theorem division_scaling (a b c : ℝ) (h : a / b = c) : (a / 100) / (b / 100) = c := by
  sorry

end division_scaling_l751_75173


namespace parabola_intersection_locus_l751_75106

/-- Given a parabola y² = 2px with vertex at the origin, 
    prove that the locus of intersection points forms another parabola -/
theorem parabola_intersection_locus (p : ℝ) (h : p > 0) :
  ∃ (f : ℝ → ℝ), 
    (∀ x y : ℝ, y ^ 2 = 2 * p * x → 
      ∃ (x₁ y₁ : ℝ), 
        y₁ ^ 2 = 2 * p * x₁ ∧ 
        (y - y₁) = -(y₁ / p) * (x - x₁) ∧
        y = (p / y₁) * (x - p / 2) ∧
        f x = y) ∧
    (∀ x : ℝ, (f x) ^ 2 = (p / 2) * (x - p / 2)) := by
  sorry


end parabola_intersection_locus_l751_75106


namespace investment_growth_l751_75117

/-- Represents the initial investment amount in dollars -/
def initial_investment : ℝ := 295097.57

/-- Represents the future value in dollars -/
def future_value : ℝ := 600000

/-- Represents the annual interest rate as a decimal -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of compounding periods per year -/
def compounding_periods_per_year : ℕ := 2

/-- Represents the number of years for the investment -/
def investment_years : ℕ := 12

/-- Theorem stating that the initial investment grows to the future value 
    under the given conditions -/
theorem investment_growth (ε : ℝ) (h_ε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  |future_value - initial_investment * (1 + annual_interest_rate / compounding_periods_per_year) ^ (compounding_periods_per_year * investment_years)| < δ ∧
  δ < ε :=
sorry

end investment_growth_l751_75117
