import Mathlib

namespace NUMINAMATH_CALUDE_june_score_l391_39116

theorem june_score (april_may_avg : ℝ) (april_may_june_avg : ℝ) (june_score : ℝ) :
  april_may_avg = 89 →
  april_may_june_avg = 88 →
  june_score = 3 * april_may_june_avg - 2 * april_may_avg →
  june_score = 86 := by
sorry

end NUMINAMATH_CALUDE_june_score_l391_39116


namespace NUMINAMATH_CALUDE_tan_half_sum_right_triangle_angles_l391_39143

theorem tan_half_sum_right_triangle_angles (A B : ℝ) : 
  0 < A → A < π/2 → 0 < B → B < π/2 → A + B = π/2 → 
  Real.tan ((A + B) / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_half_sum_right_triangle_angles_l391_39143


namespace NUMINAMATH_CALUDE_product_of_four_integers_l391_39175

theorem product_of_four_integers (A B C D : ℕ) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)
  (h_sum : A + B + C + D = 64)
  (h_relation : A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 3) :
  A * B * C * D = 19440 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_integers_l391_39175


namespace NUMINAMATH_CALUDE_milk_container_problem_l391_39160

theorem milk_container_problem (A : ℝ) 
  (hB : ℝ) (hC : ℝ) 
  (hB_initial : hB = 0.375 * A) 
  (hC_initial : hC = A - hB) 
  (h_equal_after_transfer : hB + 150 = hC - 150) : A = 1200 :=
by sorry

end NUMINAMATH_CALUDE_milk_container_problem_l391_39160


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l391_39123

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l391_39123


namespace NUMINAMATH_CALUDE_brian_drove_200_miles_more_l391_39101

/-- Represents the driving scenario of Mike, Steve, and Brian --/
structure DrivingScenario where
  t : ℝ  -- Mike's driving time
  s : ℝ  -- Mike's driving speed
  d : ℝ  -- Mike's driving distance
  steve_distance : ℝ  -- Steve's driving distance
  brian_distance : ℝ  -- Brian's driving distance

/-- The conditions of the driving scenario --/
def scenario_conditions (scenario : DrivingScenario) : Prop :=
  scenario.d = scenario.s * scenario.t ∧  -- Mike's distance equation
  scenario.steve_distance = (scenario.s + 6) * (scenario.t + 1.5) ∧  -- Steve's distance equation
  scenario.brian_distance = (scenario.s + 12) * (scenario.t + 3) ∧  -- Brian's distance equation
  scenario.steve_distance = scenario.d + 90  -- Steve drove 90 miles more than Mike

/-- The theorem stating that Brian drove 200 miles more than Mike --/
theorem brian_drove_200_miles_more (scenario : DrivingScenario) 
  (h : scenario_conditions scenario) : 
  scenario.brian_distance = scenario.d + 200 := by
  sorry


end NUMINAMATH_CALUDE_brian_drove_200_miles_more_l391_39101


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_F_powers_of_two_l391_39168

-- Define the function F recursively
def F : ℕ → ℚ
  | 0 => 0
  | 1 => 3/2
  | (n+2) => 5/2 * F (n+1) - F n

-- Define the series
def series_sum : ℕ → ℚ
  | 0 => 1 / F (2^0)
  | (n+1) => series_sum n + 1 / F (2^(n+1))

-- State the theorem
theorem sum_of_reciprocal_F_powers_of_two :
  ∃ (L : ℚ), L = 1 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |series_sum n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_F_powers_of_two_l391_39168


namespace NUMINAMATH_CALUDE_electrons_gained_by_oxidizing_agent_l391_39124

-- Define the redox reaction components
structure RedoxReaction where
  cu_io3_2 : ℕ
  ki : ℕ
  h2so4 : ℕ
  cui : ℕ
  i2 : ℕ
  k2so4 : ℕ
  h2o : ℕ

-- Define the valence changes
structure ValenceChanges where
  cu_initial : ℤ
  cu_final : ℤ
  i_initial : ℤ
  i_final : ℤ

-- Define the function to calculate electron moles gained
def electronMolesGained (vc : ValenceChanges) : ℤ :=
  (vc.cu_initial - vc.cu_final) + 2 * (vc.i_initial - vc.i_final)

-- Theorem statement
theorem electrons_gained_by_oxidizing_agent 
  (reaction : RedoxReaction)
  (valence_changes : ValenceChanges)
  (h1 : reaction.cu_io3_2 = 2)
  (h2 : reaction.ki = 24)
  (h3 : reaction.h2so4 = 12)
  (h4 : reaction.cui = 2)
  (h5 : reaction.i2 = 13)
  (h6 : reaction.k2so4 = 12)
  (h7 : reaction.h2o = 12)
  (h8 : valence_changes.cu_initial = 2)
  (h9 : valence_changes.cu_final = 1)
  (h10 : valence_changes.i_initial = 5)
  (h11 : valence_changes.i_final = 0) :
  electronMolesGained valence_changes = 11 := by
  sorry

end NUMINAMATH_CALUDE_electrons_gained_by_oxidizing_agent_l391_39124


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l391_39198

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 48) 
  (h2 : Nat.gcd a b = 8) : 
  a * b = 384 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l391_39198


namespace NUMINAMATH_CALUDE_function_difference_implies_m_value_l391_39108

theorem function_difference_implies_m_value :
  ∀ (m : ℝ),
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 3 * x + 5
  let g : ℝ → ℝ := λ x ↦ x^2 - m * x - 8
  (f 5 - g 5 = 15) → (m = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_function_difference_implies_m_value_l391_39108


namespace NUMINAMATH_CALUDE_algebraic_simplification_l391_39118

theorem algebraic_simplification (a : ℝ) : (-2*a)^3 * a^3 + (-3*a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l391_39118


namespace NUMINAMATH_CALUDE_equation_solution_l391_39186

theorem equation_solution : ∃ x : ℝ, (10 - 2*x)^2 = 4*x^2 + 16 ∧ x = 21/10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l391_39186


namespace NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l391_39185

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and the x-axis -/
def areaRegionBetweenCirclesAndXAxis (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_between_circles_and_x_axis :
  let c1 : Circle := { center := (6, 5), radius := 3 }
  let c2 : Circle := { center := (14, 5), radius := 3 }
  areaRegionBetweenCirclesAndXAxis c1 c2 = 40 - 9 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l391_39185


namespace NUMINAMATH_CALUDE_b_share_is_302_l391_39144

/-- Given a division of money among five people A, B, C, D, and E, 
    prove that B's share is 302 rupees. -/
theorem b_share_is_302 
  (total : ℕ) 
  (share_a share_b share_c share_d share_e : ℕ) 
  (h_total : total = 1540)
  (h_a : share_a = share_b + 40)
  (h_c : share_c = share_a + 30)
  (h_d : share_d = share_b - 50)
  (h_e : share_e = share_d + 20)
  (h_sum : share_a + share_b + share_c + share_d + share_e = total) : 
  share_b = 302 := by
  sorry


end NUMINAMATH_CALUDE_b_share_is_302_l391_39144


namespace NUMINAMATH_CALUDE_visitors_scientific_notation_l391_39178

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_scientific_notation :
  toScientificNotation 564200 = ScientificNotation.mk 5.642 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_visitors_scientific_notation_l391_39178


namespace NUMINAMATH_CALUDE_root_in_interval_l391_39138

def f (x : ℝ) := 3 * x^2 + 3 * x - 8

theorem root_in_interval :
  (f 1.25 < 0) → (f 1.5 > 0) →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l391_39138


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l391_39164

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l391_39164


namespace NUMINAMATH_CALUDE_union_of_sets_l391_39169

theorem union_of_sets (S R : Set ℕ) : 
  S = {1} → R = {1, 2} → S ∪ R = {1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l391_39169


namespace NUMINAMATH_CALUDE_cube_root_simplification_l391_39156

theorem cube_root_simplification : 
  (80^3 + 100^3 + 120^3 : ℝ)^(1/3) = 20 * 405^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l391_39156


namespace NUMINAMATH_CALUDE_bathroom_extension_l391_39177

/-- Given a rectangular bathroom with area and width, calculate the new area after extension --/
theorem bathroom_extension (area : ℝ) (width : ℝ) (extension : ℝ) :
  area = 96 →
  width = 8 →
  extension = 2 →
  (area / width + extension) * (width + extension) = 140 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_extension_l391_39177


namespace NUMINAMATH_CALUDE_greifswald_schools_l391_39172

-- Define the schools
inductive School
| A
| B
| C

-- Define the student type
structure Student where
  id : Nat
  school : School

-- Define the knowing relation
def knows (s1 s2 : Student) : Prop := sorry

-- Define the set of all students
def AllStudents : Set Student := sorry

-- State the conditions
axiom non_empty_schools :
  ∃ (a b c : Student), a.school = School.A ∧ b.school = School.B ∧ c.school = School.C

axiom knowing_condition :
  ∀ (a b c : Student),
    a.school = School.A → b.school = School.B → c.school = School.C →
    ((knows a b ∧ knows a c ∧ ¬knows b c) ∨
     (knows a b ∧ ¬knows a c ∧ knows b c) ∨
     (¬knows a b ∧ knows a c ∧ knows b c))

-- State the theorem to be proved
theorem greifswald_schools :
  (∃ (a : Student), a.school = School.A ∧ ∀ (b : Student), b.school = School.B → knows a b) ∨
  (∃ (b : Student), b.school = School.B ∧ ∀ (c : Student), c.school = School.C → knows b c) ∨
  (∃ (c : Student), c.school = School.C ∧ ∀ (a : Student), a.school = School.A → knows c a) :=
by
  sorry

end NUMINAMATH_CALUDE_greifswald_schools_l391_39172


namespace NUMINAMATH_CALUDE_cubic_factor_identity_l391_39183

theorem cubic_factor_identity (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_factor_identity_l391_39183


namespace NUMINAMATH_CALUDE_real_number_in_set_l391_39193

theorem real_number_in_set (a : ℝ) : a ∈ ({a^2 - a, 0} : Set ℝ) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_number_in_set_l391_39193


namespace NUMINAMATH_CALUDE_road_repair_group_size_l391_39120

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 15

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- The theorem stating that the first group size is 39 -/
theorem road_repair_group_size :
  first_group_size * first_group_days * first_group_hours =
  second_group_size * second_group_days * second_group_hours :=
by
  sorry

#check road_repair_group_size

end NUMINAMATH_CALUDE_road_repair_group_size_l391_39120


namespace NUMINAMATH_CALUDE_max_factors_bound_l391_39105

/-- The number of positive factors of b^n, where b and n are positive integers with b ≤ 20 and n ≤ 20 -/
def max_factors (b n : ℕ+) : ℕ :=
  if b ≤ 20 ∧ n ≤ 20 then
    -- Placeholder for the actual calculation of factors
    0
  else
    0

/-- The maximum number of positive factors of b^n is 861, where b and n are positive integers with b ≤ 20 and n ≤ 20 -/
theorem max_factors_bound :
  ∃ (b n : ℕ+), b ≤ 20 ∧ n ≤ 20 ∧ max_factors b n = 861 ∧
  ∀ (b' n' : ℕ+), b' ≤ 20 → n' ≤ 20 → max_factors b' n' ≤ 861 :=
sorry

end NUMINAMATH_CALUDE_max_factors_bound_l391_39105


namespace NUMINAMATH_CALUDE_max_zero_point_quadratic_l391_39104

theorem max_zero_point_quadratic (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  let f := fun x : ℝ => a * x^2 + (3 + 1/b) * x - a
  let zero_points := {x : ℝ | f x = 0}
  ∃ (x : ℝ), x ∈ zero_points ∧ ∀ (y : ℝ), y ∈ zero_points → y ≤ x ∧ x = (-9 + Real.sqrt 85) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_zero_point_quadratic_l391_39104


namespace NUMINAMATH_CALUDE_tree_ratio_is_13_3_l391_39194

/-- The ratio of trees planted to fallen Mahogany trees -/
def tree_ratio (initial_mahogany : ℕ) (initial_narra : ℕ) (total_fallen : ℕ) (final_count : ℕ) : ℚ :=
  let mahogany_fallen := (total_fallen + 1) / 2
  let trees_planted := final_count - (initial_mahogany + initial_narra - total_fallen)
  (trees_planted : ℚ) / mahogany_fallen

/-- The ratio of trees planted to fallen Mahogany trees is 13:3 -/
theorem tree_ratio_is_13_3 :
  tree_ratio 50 30 5 88 = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tree_ratio_is_13_3_l391_39194


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l391_39106

theorem book_sale_gain_percentage (initial_sale_price : ℝ) (loss_percentage : ℝ) (desired_sale_price : ℝ) : 
  initial_sale_price = 810 →
  loss_percentage = 10 →
  desired_sale_price = 990 →
  let cost_price := initial_sale_price / (1 - loss_percentage / 100)
  let gain := desired_sale_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l391_39106


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l391_39180

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 4 →                     -- first term condition
  q ≠ 1 →                       -- common ratio condition
  2 * a 5 = 4 * a 1 - 2 * a 3 → -- arithmetic sequence condition
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l391_39180


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l391_39154

/-- Parabola C defined by y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line l defined by x = ty + m where m > 0 -/
structure Line where
  t : ℝ
  m : ℝ
  h_m_pos : m > 0

/-- Point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Intersection points of line and parabola -/
structure Intersection (C : Parabola) (l : Line) where
  A : ParabolaPoint C
  B : ParabolaPoint C
  h_A_on_line : A.x = l.t * A.y + l.m
  h_B_on_line : B.x = l.t * B.y + l.m

/-- Main theorem -/
theorem parabola_and_line_properties
  (C : Parabola)
  (P : ParabolaPoint C)
  (h_P_x : P.x = 2)
  (h_P_dist : (P.x - C.p/2)^2 + P.y^2 = 4^2)
  (l : Line)
  (i : Intersection C l)
  (h_circle : i.A.x * i.B.x + i.A.y * i.B.y = 0) :
  (C.p = 4 ∧ ∀ (y : ℝ), y^2 = 8 * (l.t * y + l.m)) ∧
  (l.m = 8 ∧ ∀ (y : ℝ), l.t * y + l.m = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l391_39154


namespace NUMINAMATH_CALUDE_problem_one_l391_39171

theorem problem_one : Real.sqrt 12 + (-2024)^0 - 4 * Real.sin (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_l391_39171


namespace NUMINAMATH_CALUDE_unique_integer_angle_geometric_progression_l391_39107

theorem unique_integer_angle_geometric_progression :
  ∃! (a b c : ℕ+), a + b + c = 180 ∧ 
  ∃ (r : ℚ), r > 1 ∧ b = a * (r : ℚ) ∧ c = b * (r : ℚ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_angle_geometric_progression_l391_39107


namespace NUMINAMATH_CALUDE_unique_composite_with_special_divisor_property_l391_39102

theorem unique_composite_with_special_divisor_property :
  ∃! (n : ℕ), 
    n > 1 ∧ 
    ¬(Nat.Prime n) ∧
    (∃ (k : ℕ) (d : ℕ → ℕ), 
      d 1 = 1 ∧ d k = n ∧
      (∀ i, 1 ≤ i → i < k → d i < d (i+1)) ∧
      (∀ i, 1 ≤ i → i < k → d i ∣ n) ∧
      (∀ i, 1 < i → i ≤ k → (d i - d (i-1)) = i * (d 2 - d 1))) ∧
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_composite_with_special_divisor_property_l391_39102


namespace NUMINAMATH_CALUDE_tesseract_parallel_edge_pairs_l391_39197

/-- A tesseract is a 4-dimensional hypercube -/
structure Tesseract where
  dim : Nat
  dim_eq : dim = 4

/-- The number of pairs of parallel edges in a tesseract -/
def parallel_edge_pairs (t : Tesseract) : Nat := 36

/-- Theorem: A tesseract has 36 pairs of parallel edges -/
theorem tesseract_parallel_edge_pairs (t : Tesseract) : 
  parallel_edge_pairs t = 36 := by sorry

end NUMINAMATH_CALUDE_tesseract_parallel_edge_pairs_l391_39197


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_product_l391_39129

theorem min_sum_reciprocal_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_product_l391_39129


namespace NUMINAMATH_CALUDE_original_number_proof_l391_39161

theorem original_number_proof (y : ℚ) : 1 + 1 / y = 8 / 3 → y = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l391_39161


namespace NUMINAMATH_CALUDE_zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two_l391_39137

theorem zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two :
  (∃ x : ℝ, 0 < x ∧ x < 2 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(0 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two_l391_39137


namespace NUMINAMATH_CALUDE_integral_absolute_value_l391_39191

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

theorem integral_absolute_value : ∫ x in (0)..(4), f x = 10 := by sorry

end NUMINAMATH_CALUDE_integral_absolute_value_l391_39191


namespace NUMINAMATH_CALUDE_square_diagonal_half_l391_39141

/-- Given a square with side length 6 cm and AE = 8 cm, prove that OB = 4.5 cm -/
theorem square_diagonal_half (side_length : ℝ) (AE : ℝ) (OB : ℝ) :
  side_length = 6 →
  AE = 8 →
  OB = side_length * Real.sqrt 2 / 2 →
  OB = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_half_l391_39141


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l391_39166

/-- The surface area of a cuboid given its dimensions -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 3, length 4, and height 5 is 94 -/
theorem cuboid_surface_area_example : cuboid_surface_area 3 4 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l391_39166


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l391_39182

/-- A geometric sequence with positive terms and common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l391_39182


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l391_39140

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0. The least significant bit is at the head of the list. -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : ℕ :=
  b.enum.foldr (λ (i, bit) acc => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : BinaryNum :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : ℕ) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

theorem binary_multiplication_theorem :
  let a : BinaryNum := [true, true, false, true, true]  -- 11011₂
  let b : BinaryNum := [true, false, true]              -- 101₂
  let result : BinaryNum := [true, true, true, false, true, true, false, false, true]  -- 100110111₂
  binary_multiply a b = result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l391_39140


namespace NUMINAMATH_CALUDE_smallest_integer_cube_root_l391_39158

theorem smallest_integer_cube_root (m n : ℕ) (r : ℝ) : 
  (∀ k < n, ¬∃ (m' : ℕ) (r' : ℝ), m' < m ∧ 0 < r' ∧ r' < 1/500 ∧ m'^(1/3 : ℝ) = k + r') →
  0 < r →
  r < 1/500 →
  m^(1/3 : ℝ) = n + r →
  n = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_cube_root_l391_39158


namespace NUMINAMATH_CALUDE_julia_bought_496_balls_l391_39190

/-- The number of balls Julia bought -/
def total_balls : ℕ :=
  let red_packs : ℕ := 3
  let yellow_packs : ℕ := 10
  let green_packs : ℕ := 8
  let blue_packs : ℕ := 5
  let red_balls_per_pack : ℕ := 22
  let yellow_balls_per_pack : ℕ := 19
  let green_balls_per_pack : ℕ := 15
  let blue_balls_per_pack : ℕ := 24
  red_packs * red_balls_per_pack +
  yellow_packs * yellow_balls_per_pack +
  green_packs * green_balls_per_pack +
  blue_packs * blue_balls_per_pack

theorem julia_bought_496_balls : total_balls = 496 := by
  sorry

end NUMINAMATH_CALUDE_julia_bought_496_balls_l391_39190


namespace NUMINAMATH_CALUDE_inverse_functions_values_l391_39128

-- Define the inverse function relationship
def are_inverse_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Define the two linear functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- State the theorem
theorem inverse_functions_values :
  ∀ a b : ℝ, are_inverse_functions (f a) (g b) → a = 1/3 ∧ b = -6 :=
by sorry

end NUMINAMATH_CALUDE_inverse_functions_values_l391_39128


namespace NUMINAMATH_CALUDE_percentage_difference_l391_39109

theorem percentage_difference (C : ℝ) (A B : ℝ) 
  (hA : A = 0.75 * C) 
  (hB : B = 0.63 * C) : 
  (A - B) / A = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l391_39109


namespace NUMINAMATH_CALUDE_hexagon_triangle_count_l391_39111

/-- Regular hexagon with area 6 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : area = 6)

/-- Equilateral triangle with area 4 -/
structure EquilateralTriangle :=
  (area : ℝ)
  (is_equilateral : area = 4)

/-- Configuration of four regular hexagons -/
def HexagonConfiguration := Fin 4 → RegularHexagon

/-- Count of equilateral triangles formed by vertices of hexagons -/
def count_triangles (config : HexagonConfiguration) : ℕ := sorry

/-- Main theorem: There are 12 equilateral triangles with area 4 -/
theorem hexagon_triangle_count (config : HexagonConfiguration) :
  count_triangles config = 12 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_count_l391_39111


namespace NUMINAMATH_CALUDE_marble_distribution_l391_39188

theorem marble_distribution (total_marbles : ℕ) (num_friends : ℕ) 
  (h1 : total_marbles = 60) (h2 : num_friends = 4) :
  total_marbles / num_friends = 15 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l391_39188


namespace NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l391_39142

/-- Converts an octal digit to its decimal equivalent -/
def octal_to_decimal (digit : ℕ) : ℕ := digit

/-- Represents an octal number as a list of natural numbers -/
def octal_number : List ℕ := [1, 2, 3]

/-- Converts an octal number to its decimal equivalent -/
def octal_to_decimal_conversion (octal : List ℕ) : ℕ :=
  octal.enum.foldr (fun (i, digit) acc => acc + octal_to_decimal digit * 8^i) 0

theorem octal_123_equals_decimal_83 :
  octal_to_decimal_conversion octal_number = 83 := by
  sorry

end NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l391_39142


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l391_39113

theorem divisibility_equivalence (x y : ℤ) : 
  (7 ∣ (2*x + 3*y)) ↔ (7 ∣ (5*x + 4*y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l391_39113


namespace NUMINAMATH_CALUDE_sqrt_equality_l391_39100

theorem sqrt_equality (x : ℝ) (h : x < -1) :
  Real.sqrt ((x + 2) / (1 - (x - 2) / (x + 1))) = Real.sqrt ((x^2 + 3*x + 2) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l391_39100


namespace NUMINAMATH_CALUDE_emmanuel_jelly_beans_l391_39147

theorem emmanuel_jelly_beans (total : ℕ) (thomas_percent : ℚ) (barry_ratio : ℕ) (emmanuel_ratio : ℕ) : 
  total = 200 →
  thomas_percent = 1/10 →
  barry_ratio = 4 →
  emmanuel_ratio = 5 →
  (emmanuel_ratio * (total - thomas_percent * total)) / (barry_ratio + emmanuel_ratio) = 100 := by
sorry

end NUMINAMATH_CALUDE_emmanuel_jelly_beans_l391_39147


namespace NUMINAMATH_CALUDE_tree_planting_group_size_l391_39145

theorem tree_planting_group_size :
  ∀ x : ℕ,
  (7 * x + 9 > 9 * (x - 1)) →
  (7 * x + 9 < 9 * (x - 1) + 3) →
  x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_group_size_l391_39145


namespace NUMINAMATH_CALUDE_square_minus_floor_product_l391_39125

/-- The floor function, which returns the greatest integer less than or equal to a given real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- Theorem stating that for A = 50 + 19√7, A^2 - A⌊A⌋ = 27 -/
theorem square_minus_floor_product (A : ℝ) (h : A = 50 + 19 * Real.sqrt 7) :
  A^2 - A * (floor A) = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_floor_product_l391_39125


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l391_39150

-- Define a sequence of real numbers
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be geometric
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the condition given in the problem
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l391_39150


namespace NUMINAMATH_CALUDE_total_packs_bought_l391_39165

/-- The number of index card packs John buys for each student -/
def packs_per_student : ℕ := 3

/-- The number of students in the first class -/
def class1_students : ℕ := 20

/-- The number of students in the second class -/
def class2_students : ℕ := 25

/-- The number of students in the third class -/
def class3_students : ℕ := 18

/-- The number of students in the fourth class -/
def class4_students : ℕ := 22

/-- The number of students in the fifth class -/
def class5_students : ℕ := 15

/-- The total number of students across all classes -/
def total_students : ℕ := class1_students + class2_students + class3_students + class4_students + class5_students

/-- Theorem: The total number of index card packs bought by John is 300 -/
theorem total_packs_bought : packs_per_student * total_students = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_bought_l391_39165


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l391_39121

theorem quadratic_form_sum (x : ℝ) : ∃ (b c : ℝ), 
  2*x^2 - 28*x + 50 = (x + b)^2 + c ∧ b + c = -55 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l391_39121


namespace NUMINAMATH_CALUDE_simplification_order_l391_39112

-- Define the power operations
inductive PowerOperation
| MultiplicationOfPowers
| PowerOfPower
| PowerOfProduct

-- Define a function to simplify the expression
def simplify (a : ℕ) : ℕ := (a^2 * a^3)^2

-- Define a function to get the sequence of operations
def operationSequence : List PowerOperation :=
  [PowerOperation.PowerOfProduct, PowerOperation.PowerOfPower, PowerOperation.MultiplicationOfPowers]

-- State the theorem
theorem simplification_order :
  simplify a = a^10 ∧ operationSequence = [PowerOperation.PowerOfProduct, PowerOperation.PowerOfPower, PowerOperation.MultiplicationOfPowers] :=
sorry

end NUMINAMATH_CALUDE_simplification_order_l391_39112


namespace NUMINAMATH_CALUDE_goldbach_extension_l391_39139

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_two_prime_pairs (N : ℕ) : Prop :=
  ∃ (p₁ q₁ p₂ q₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ is_prime p₂ ∧ is_prime q₂ ∧
    p₁ + q₁ = N ∧ p₂ + q₂ = N ∧
    (p₁ ≠ p₂ ∨ q₁ ≠ q₂) ∧
    ∀ (p q : ℕ), is_prime p → is_prime q → p + q = N →
      ((p = p₁ ∧ q = q₁) ∨ (p = p₂ ∧ q = q₂) ∨ (p = q₁ ∧ q = p₁) ∨ (p = q₂ ∧ q = p₂))

theorem goldbach_extension :
  ∀ N : ℕ, N ≥ 10 →
    (N = 10 ↔ (N % 2 = 0 ∧ has_two_prime_pairs N ∧
      ∀ M : ℕ, M < N → M % 2 = 0 → M > 2 → ¬has_two_prime_pairs M)) :=
sorry

end NUMINAMATH_CALUDE_goldbach_extension_l391_39139


namespace NUMINAMATH_CALUDE_fraction_meaningful_l391_39151

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (3 - x)) ↔ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l391_39151


namespace NUMINAMATH_CALUDE_major_axis_length_is_eight_l391_39149

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- The length of the major axis of an ellipse with given properties -/
def major_axis_length (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating that the length of the major axis is 8 for the given ellipse -/
theorem major_axis_length_is_eight :
  let e : Ellipse := {
    tangent_to_axes := true,
    focus1 := (5, -4 + 2 * Real.sqrt 3),
    focus2 := (5, -4 - 2 * Real.sqrt 3)
  }
  major_axis_length e = 8 :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_is_eight_l391_39149


namespace NUMINAMATH_CALUDE_extreme_point_value_bound_l391_39155

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - x^2

-- Define the derivative of f
def f_deriv (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - 2 * x

-- Theorem statement
theorem extreme_point_value_bound 
  (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : f_deriv k x₁ = 0) 
  (h3 : f_deriv k x₂ = 0) 
  (h4 : ∀ x, x₁ < x → x < x₂ → f_deriv k x ≠ 0) : 
  0 < f k x₁ ∧ f k x₁ < 1 := by
sorry

end

end NUMINAMATH_CALUDE_extreme_point_value_bound_l391_39155


namespace NUMINAMATH_CALUDE_barneys_restock_order_l391_39127

/-- Represents the number of items in Barney's grocery store --/
structure GroceryStore where
  sold : Nat        -- Number of items sold that day
  left : Nat        -- Number of items left in the store
  storeroom : Nat   -- Number of items in the storeroom

/-- Calculates the total number of items ordered to restock the shelves --/
def items_ordered (store : GroceryStore) : Nat :=
  store.sold + store.left + store.storeroom

/-- Theorem stating that for Barney's grocery store, the number of items
    ordered to restock the shelves is 5608 --/
theorem barneys_restock_order :
  let store : GroceryStore := {
    sold := 1561,
    left := 3472,
    storeroom := 575
  }
  items_ordered store = 5608 := by
  sorry

end NUMINAMATH_CALUDE_barneys_restock_order_l391_39127


namespace NUMINAMATH_CALUDE_female_listeners_l391_39153

theorem female_listeners (total_listeners male_listeners : ℕ) 
  (h1 : total_listeners = 180) 
  (h2 : male_listeners = 80) : 
  total_listeners - male_listeners = 100 := by
  sorry

end NUMINAMATH_CALUDE_female_listeners_l391_39153


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_1000_l391_39146

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (λ e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents_for_1000 :
  ∀ exponents : List ℕ,
    is_sum_of_distinct_powers_of_two 1000 exponents →
    exponents.length ≥ 3 →
    exponents.sum ≥ 38 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_1000_l391_39146


namespace NUMINAMATH_CALUDE_product_of_y_values_l391_39196

theorem product_of_y_values (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, 
    (|2 * y₁ * 3| + 5 = 47) ∧ 
    (|2 * y₂ * 3| + 5 = 47) ∧ 
    (y₁ ≠ y₂) ∧
    (y₁ * y₂ = -49)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_y_values_l391_39196


namespace NUMINAMATH_CALUDE_ratio_of_a_to_c_l391_39174

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 7 / 3)
  (hdb : d / b = 5 / 4) :
  a / c = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_c_l391_39174


namespace NUMINAMATH_CALUDE_cone_volume_l391_39170

/-- The volume of a cone with base radius 1 and unfolded side surface area 2π -/
theorem cone_volume (r : Real) (side_area : Real) (h : Real) : 
  r = 1 → side_area = 2 * Real.pi → h = Real.sqrt 3 → 
  (1 / 3 : Real) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 := by
  sorry

#check cone_volume

end NUMINAMATH_CALUDE_cone_volume_l391_39170


namespace NUMINAMATH_CALUDE_complex_magnitude_l391_39162

theorem complex_magnitude (z : ℂ) (h : (1 - 2*I)*z = 3 + I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l391_39162


namespace NUMINAMATH_CALUDE_weightlifting_total_capacity_l391_39122

/-- Represents a weightlifter's lifting capacities -/
structure LiftingCapacity where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Calculates the new lifting capacity after applying percentage increases -/
def newCapacity (initial : LiftingCapacity) (cjIncrease snatchIncrease : ℝ) : LiftingCapacity :=
  { cleanAndJerk := initial.cleanAndJerk * (1 + cjIncrease)
  , snatch := initial.snatch * (1 + snatchIncrease) }

/-- Calculates the total lifting capacity for a weightlifter -/
def totalCapacity (capacity : LiftingCapacity) : ℝ :=
  capacity.cleanAndJerk + capacity.snatch

/-- The theorem to be proved -/
theorem weightlifting_total_capacity : 
  let john_initial := LiftingCapacity.mk 80 50
  let alice_initial := LiftingCapacity.mk 90 55
  let mark_initial := LiftingCapacity.mk 100 65
  
  let john_final := newCapacity john_initial 1 0.8
  let alice_final := newCapacity alice_initial 0.5 0.9
  let mark_final := newCapacity mark_initial 0.75 0.7
  
  totalCapacity john_final + totalCapacity alice_final + totalCapacity mark_final = 775 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_total_capacity_l391_39122


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l391_39131

/-- The line (2a+b)x + (a+b)y + a - b = 0 passes through (-2, 3) for all real a and b -/
theorem line_passes_through_fixed_point :
  ∀ (a b x y : ℝ), (2*a + b)*x + (a + b)*y + a - b = 0 ↔ (x = -2 ∧ y = 3) ∨ (x ≠ -2 ∨ y ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l391_39131


namespace NUMINAMATH_CALUDE_tan_alpha_neg_three_l391_39136

theorem tan_alpha_neg_three (α : ℝ) (h : Real.tan α = -3) :
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_neg_three_l391_39136


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l391_39152

/-- Given a line segment with one endpoint at (10, 2) and midpoint at (4, -6),
    the sum of the coordinates of the other endpoint is -16. -/
theorem endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (x + 10) / 2 = 4 →
  (y + 2) / 2 = -6 →
  x + y = -16 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l391_39152


namespace NUMINAMATH_CALUDE_sine_cosine_shift_l391_39189

/-- The shift amount between two trigonometric functions -/
def shift_amount (f g : ℝ → ℝ) : ℝ :=
  sorry

theorem sine_cosine_shift :
  let f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x
  let g (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x
  let φ := shift_amount f g
  0 < φ ∧ φ < 2 * Real.pi → φ = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_shift_l391_39189


namespace NUMINAMATH_CALUDE_anthony_total_pencils_l391_39187

/-- The total number of pencils Anthony has after receiving pencils from others -/
def total_pencils (initial : ℕ) (from_kathryn : ℕ) (from_greg : ℕ) (from_maria : ℕ) : ℕ :=
  initial + from_kathryn + from_greg + from_maria

/-- Theorem stating that Anthony's total pencils is 287 -/
theorem anthony_total_pencils :
  total_pencils 9 56 84 138 = 287 := by
  sorry

end NUMINAMATH_CALUDE_anthony_total_pencils_l391_39187


namespace NUMINAMATH_CALUDE_price_reduction_l391_39159

theorem price_reduction (original_price reduced_price : ℝ) : 
  reduced_price = original_price * 0.5 ∧ reduced_price = 620 → original_price = 1240 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l391_39159


namespace NUMINAMATH_CALUDE_tree_count_after_planting_l391_39134

theorem tree_count_after_planting (road_length : ℕ) (original_spacing : ℕ) (additional_trees : ℕ) : 
  road_length = 7200 → 
  original_spacing = 120 → 
  additional_trees = 5 → 
  (road_length / original_spacing * (additional_trees + 1) + 1) * 2 = 722 := by
sorry

end NUMINAMATH_CALUDE_tree_count_after_planting_l391_39134


namespace NUMINAMATH_CALUDE_area_of_intersection_l391_39117

-- Define the square ABCD
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_unit : A = (0, 1) ∧ B = (1, 1) ∧ C = (1, 0) ∧ D = (0, 0))

-- Define the rotation
def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

-- Define the rotated square A'B'C'D'
def rotated_square (S : Square) (angle : ℝ) : Square := sorry

-- Define the intersection quadrilateral DALC'
structure Quadrilateral :=
  (D A L C' : ℝ × ℝ)

-- Define the area function for a quadrilateral
def area (Q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem area_of_intersection (S : Square) (α : ℝ) :
  let S' := rotated_square S α
  let Q := Quadrilateral.mk S.D S.A (Real.cos α, 1) (Real.cos α, Real.sin α)
  area Q = 1/2 * (1 - Real.sin α * Real.cos α) :=
sorry

end NUMINAMATH_CALUDE_area_of_intersection_l391_39117


namespace NUMINAMATH_CALUDE_no_integer_solution_for_2006_l391_39167

theorem no_integer_solution_for_2006 : ¬∃ (x y : ℤ), x^2 - y^2 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_2006_l391_39167


namespace NUMINAMATH_CALUDE_basket_weight_is_20_l391_39110

/-- The weight of the basket in kilograms -/
def basket_weight : ℝ := 20

/-- The lifting capacity of one standard balloon in kilograms -/
def balloon_capacity : ℝ := 60

/-- One standard balloon can lift a basket with contents weighing not more than 80 kg -/
axiom one_balloon_limit : basket_weight + balloon_capacity ≤ 80

/-- Two standard balloons can lift the same basket with contents weighing not more than 180 kg -/
axiom two_balloon_limit : basket_weight + 2 * balloon_capacity ≤ 180

theorem basket_weight_is_20 : basket_weight = 20 := by sorry

end NUMINAMATH_CALUDE_basket_weight_is_20_l391_39110


namespace NUMINAMATH_CALUDE_untouchable_area_of_cube_l391_39132

-- Define the cube and sphere
def cube_edge_length : ℝ := 4
def sphere_radius : ℝ := 1

-- Theorem statement
theorem untouchable_area_of_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) 
  (h1 : cube_edge_length = 4) (h2 : sphere_radius = 1) : 
  (6 * (cube_edge_length ^ 2 - (cube_edge_length - 2 * sphere_radius) ^ 2)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_untouchable_area_of_cube_l391_39132


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_l391_39163

/-- The function f(x) = x^2 - 2ax is increasing on [1, +∞) if and only if a ≤ 1 -/
theorem increasing_quadratic_function (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => x^2 - 2*a*x)) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_l391_39163


namespace NUMINAMATH_CALUDE_square_area_after_cut_l391_39115

theorem square_area_after_cut (x : ℝ) : 
  x > 0 → x * (x - 3) = 40 → x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_after_cut_l391_39115


namespace NUMINAMATH_CALUDE_count_rectangles_3x6_grid_l391_39179

/-- The number of rectangles in a 3 × 6 grid with vertices at grid points -/
def num_rectangles : ℕ :=
  let horizontal_lines := 4
  let vertical_lines := 7
  let horizontal_vertical_rectangles := (horizontal_lines.choose 2) * (vertical_lines.choose 2)
  let diagonal_sqrt2 := 5 * 2
  let diagonal_2sqrt2 := 4 * 2
  let diagonal_sqrt5 := 4 * 2
  horizontal_vertical_rectangles + diagonal_sqrt2 + diagonal_2sqrt2 + diagonal_sqrt5

theorem count_rectangles_3x6_grid :
  num_rectangles = 152 :=
sorry

end NUMINAMATH_CALUDE_count_rectangles_3x6_grid_l391_39179


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l391_39184

theorem algebraic_expression_simplification (x : ℝ) :
  x = 2 * Real.cos (45 * π / 180) + 1 →
  (1 / (x - 1) - (x - 3) / (x^2 - 2*x + 1)) / (2 / (x - 1)) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l391_39184


namespace NUMINAMATH_CALUDE_article_cost_l391_39195

/-- The cost of an article given specific selling prices and gains -/
theorem article_cost (sp1 sp2 : ℝ) (gain_increase : ℝ) :
  sp1 = 500 →
  sp2 = 570 →
  gain_increase = 0.15 →
  ∃ (cost gain : ℝ),
    cost + gain = sp1 ∧
    cost + gain * (1 + gain_increase) = sp2 ∧
    cost = 100 / 3 :=
sorry

end NUMINAMATH_CALUDE_article_cost_l391_39195


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l391_39173

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

-- Define the length of the real axis
def real_axis_length : ℝ := 4

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola_equation x y → real_axis_length = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l391_39173


namespace NUMINAMATH_CALUDE_simplify_fraction_l391_39192

theorem simplify_fraction : (144 : ℚ) / 216 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l391_39192


namespace NUMINAMATH_CALUDE_complement_of_intersection_l391_39148

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_of_intersection (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4}) 
  (hM : M = {1, 2, 3}) 
  (hN : N = {2, 3, 4}) : 
  (M ∩ N)ᶜ = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l391_39148


namespace NUMINAMATH_CALUDE_min_k_value_l391_39119

/-- A special number is a three-digit number with all digits different and non-zero -/
def is_special_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10 ∧
  ∀ i, (n / 10^i) % 10 ≠ 0

/-- F(n) is the sum of three new numbers obtained by swapping digits of n, divided by 111 -/
def F (n : ℕ) : ℚ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ((d2 * 100 + d1 * 10 + d3) + (d3 * 100 + d2 * 10 + d1) + (d1 * 100 + d3 * 10 + d2)) / 111

theorem min_k_value (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 1 ≤ y ∧ y ≤ 9)
  (hs : is_special_number (100 * x + 32)) (ht : is_special_number (150 + y))
  (h_sum : F (100 * x + 32) + F (150 + y) = 19) :
  let s := 100 * x + 32
  let t := 150 + y
  let k := F s - F t
  ∃ k₀, k ≥ k₀ ∧ k₀ = -7 :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l391_39119


namespace NUMINAMATH_CALUDE_sum_m_n_equals_19_l391_39133

theorem sum_m_n_equals_19 (m n : ℕ+) 
  (h1 : (m.val.choose n.val) * 2 = 272)
  (h2 : (m.val.factorial / (m.val - n.val).factorial) = 272) :
  m + n = 19 := by sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_19_l391_39133


namespace NUMINAMATH_CALUDE_range_of_m_l391_39126

/-- A function f is decreasing on (0, +∞) -/
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ 0 < y ∧ x < y → f y < f x

/-- The solution set of (x-1)² > m is ℝ -/
def SolutionSetIsReals (m : ℝ) : Prop :=
  ∀ x, (x - 1)^2 > m

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : IsDecreasingOn f)
  (h2 : SolutionSetIsReals m)
  (h3 : (IsDecreasingOn f) ∨ (SolutionSetIsReals m))
  (h4 : ¬((IsDecreasingOn f) ∧ (SolutionSetIsReals m))) :
  0 ≤ m ∧ m < (1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l391_39126


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l391_39157

/-- A parabola with equation y = x^2 + 6x + c has its vertex on the x-axis if and only if c = 9 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ ∀ y : ℝ, y^2 + 6*y + c ≥ x^2 + 6*x + c) ↔ c = 9 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l391_39157


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l391_39130

/-- Represents the number of coins in the final distribution step -/
def x : ℕ := sorry

/-- Pete's coin distribution pattern -/
def petes_coins (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Polly's final coin count -/
def pollys_coins : ℕ := x

/-- Pete's final coin count -/
def petes_final_coins : ℕ := 3 * x

theorem pirate_treasure_distribution :
  petes_coins x = petes_final_coins ∧
  pollys_coins + petes_final_coins = 20 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l391_39130


namespace NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l391_39103

def fair_six_sided_die : Fin 6 → ℚ
  | _ => 1 / 6

def is_odd (n : Fin 6) : Bool :=
  n.val % 2 = 1

def prob_exactly_k_odd (k : Nat) (n : Nat) : ℚ :=
  (Nat.choose n k) * (1/2)^k * (1/2)^(n-k)

theorem prob_five_odd_in_six_rolls :
  prob_exactly_k_odd 5 6 = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l391_39103


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l391_39114

/-- Given a quadratic expression x^2 - 16x + 15, when rewritten in the form (x+d)^2 + e,
    the sum of d and e is -57. -/
theorem quadratic_rewrite_sum (d e : ℝ) : 
  (∀ x, x^2 - 16*x + 15 = (x+d)^2 + e) → d + e = -57 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l391_39114


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l391_39135

theorem digit_puzzle_solution :
  ∀ (E F G H : ℕ),
  (E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10) →
  (10 * E + F) + (10 * G + E) = 10 * H + E →
  (10 * E + F) - (10 * G + E) = E →
  H = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l391_39135


namespace NUMINAMATH_CALUDE_pedro_extra_squares_l391_39199

theorem pedro_extra_squares (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end NUMINAMATH_CALUDE_pedro_extra_squares_l391_39199


namespace NUMINAMATH_CALUDE_black_pens_count_l391_39176

/-- The number of black pens initially in the jar -/
def initial_black_pens : ℕ := 21

theorem black_pens_count :
  let initial_blue_pens : ℕ := 9
  let initial_red_pens : ℕ := 6
  let removed_blue_pens : ℕ := 4
  let removed_black_pens : ℕ := 7
  let remaining_pens : ℕ := 25
  initial_blue_pens + initial_black_pens + initial_red_pens - 
    (removed_blue_pens + removed_black_pens) = remaining_pens →
  initial_black_pens = 21 := by
sorry

end NUMINAMATH_CALUDE_black_pens_count_l391_39176


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l391_39181

theorem quadratic_equation_conversion :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) ↔ x^2 - 3*x - 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l391_39181
