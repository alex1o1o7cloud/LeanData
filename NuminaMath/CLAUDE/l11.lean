import Mathlib

namespace NUMINAMATH_CALUDE_square_root_squared_l11_1187

theorem square_root_squared (x : ℝ) (hx : x = 49) : (Real.sqrt x)^2 = x := by
  sorry

end NUMINAMATH_CALUDE_square_root_squared_l11_1187


namespace NUMINAMATH_CALUDE_six_valid_cuts_l11_1157

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertex : Point3D
  base : (Point3D × Point3D × Point3D)

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  vertex1 : Point3D
  vertex2 : Point3D
  vertex3 : Point3D

/-- Function to check if a plane cuts a tetrahedron such that 
    the first projection is an isosceles right triangle -/
def validCut (t : Tetrahedron) (p : Plane) : Bool :=
  sorry

/-- Function to count the number of valid cutting planes -/
def countValidCuts (t : Tetrahedron) : Nat :=
  sorry

/-- Theorem stating that there are exactly 6 valid cutting planes -/
theorem six_valid_cuts (t : Tetrahedron) : 
  countValidCuts t = 6 := by sorry

end NUMINAMATH_CALUDE_six_valid_cuts_l11_1157


namespace NUMINAMATH_CALUDE_negation_equivalence_l11_1104

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l11_1104


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l11_1172

theorem arithmetic_expression_evaluation : 2 + (4 * 3 - 2) / 2 * 3 + 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l11_1172


namespace NUMINAMATH_CALUDE_line_chart_for_weekly_temperature_l11_1171

/-- A type representing different chart types -/
inductive ChartType
  | Bar
  | Line
  | Pie
  | Scatter

/-- A structure representing data over time -/
structure TimeSeriesData where
  time_period : String
  has_continuous_change : Bool

/-- A function to determine the most appropriate chart type for a given data set -/
def most_appropriate_chart (data : TimeSeriesData) : ChartType :=
  if data.has_continuous_change then ChartType.Line else ChartType.Bar

/-- Theorem stating that a line chart is most appropriate for weekly temperature data -/
theorem line_chart_for_weekly_temperature :
  let weekly_temp_data : TimeSeriesData := { time_period := "Week", has_continuous_change := true }
  most_appropriate_chart weekly_temp_data = ChartType.Line :=
by
  sorry


end NUMINAMATH_CALUDE_line_chart_for_weekly_temperature_l11_1171


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1100_for_23_divisibility_least_addition_is_4_l11_1102

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_to_1100_for_23_divisibility :
  ∃ (x : ℕ), x < 23 ∧ (1100 + x) % 23 = 0 ∧ ∀ (y : ℕ), y < x → (1100 + y) % 23 ≠ 0 :=
by
  apply least_addition_for_divisibility 1100 23
  norm_num

#eval (1100 + 4) % 23  -- This should evaluate to 0

theorem least_addition_is_4 :
  4 < 23 ∧ (1100 + 4) % 23 = 0 ∧ ∀ (y : ℕ), y < 4 → (1100 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1100_for_23_divisibility_least_addition_is_4_l11_1102


namespace NUMINAMATH_CALUDE_sqrt_80_bound_l11_1189

theorem sqrt_80_bound (k : ℤ) : k < Real.sqrt 80 ∧ Real.sqrt 80 < k + 1 → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_80_bound_l11_1189


namespace NUMINAMATH_CALUDE_range_of_m_l11_1124

theorem range_of_m (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) → 
  m ∈ Set.Ici ((Real.sqrt 5 - 1) / 2) ∩ Set.Iio 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l11_1124


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l11_1103

/-- An infinite arithmetic progression of natural numbers -/
structure ArithmeticProgression :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℕ) : Prop := ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b

/-- Predicate to check if a number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ (k : ℕ), n = k * k

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℕ := ap.a + n * ap.d

theorem arithmetic_progression_properties (ap : ArithmeticProgression) :
  (∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ IsComposite (nthTerm ap n)) ∧
  ((∀ (n : ℕ), ¬IsPerfectSquare (nthTerm ap n)) ∨
   (∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ IsPerfectSquare (nthTerm ap n))) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l11_1103


namespace NUMINAMATH_CALUDE_tangent_circle_area_l11_1188

/-- A circle passing through two given points with tangent lines intersecting on x-axis --/
structure TangentCircle where
  /-- The center of the circle --/
  center : ℝ × ℝ
  /-- The radius of the circle --/
  radius : ℝ
  /-- The circle passes through point A --/
  passes_through_A : (center.1 - 7)^2 + (center.2 - 14)^2 = radius^2
  /-- The circle passes through point B --/
  passes_through_B : (center.1 - 13)^2 + (center.2 - 12)^2 = radius^2
  /-- The tangent lines at A and B intersect on the x-axis --/
  tangents_intersect_x_axis : ∃ x : ℝ, 
    (x - 7) * (center.2 - 14) = (center.1 - 7) * 14 ∧
    (x - 13) * (center.2 - 12) = (center.1 - 13) * 12

/-- The theorem stating that the area of the circle is 196π --/
theorem tangent_circle_area (ω : TangentCircle) : π * ω.radius^2 = 196 * π :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_area_l11_1188


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l11_1194

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- sum of squares condition
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l11_1194


namespace NUMINAMATH_CALUDE_a_5_equals_one_l11_1154

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem a_5_equals_one
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a 2)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_a_5_equals_one_l11_1154


namespace NUMINAMATH_CALUDE_triangle_side_length_l11_1135

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = π / 3) :
  ∃ b : ℝ, b > 0 ∧ b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l11_1135


namespace NUMINAMATH_CALUDE_reading_completion_time_l11_1186

/-- Represents a reader with their reading speed and number of books to read -/
structure Reader where
  speed : ℕ  -- hours per book
  books : ℕ

/-- Represents the reading schedule constraints -/
structure ReadingConstraints where
  hours_per_day : ℕ

/-- Calculate the total reading time for a reader -/
def total_reading_time (reader : Reader) : ℕ :=
  reader.speed * reader.books

/-- Calculate the number of days needed to finish reading -/
def days_to_finish (reader : Reader) (constraints : ReadingConstraints) : ℕ :=
  (total_reading_time reader + constraints.hours_per_day - 1) / constraints.hours_per_day

theorem reading_completion_time 
  (peter kristin : Reader) 
  (constraints : ReadingConstraints) 
  (h1 : peter.speed = 12)
  (h2 : kristin.speed = 3 * peter.speed)
  (h3 : peter.books = 20)
  (h4 : kristin.books = 20)
  (h5 : constraints.hours_per_day = 16) :
  kristin.speed = 36 ∧ 
  days_to_finish peter constraints = days_to_finish kristin constraints ∧
  days_to_finish kristin constraints = 45 := by
  sorry

end NUMINAMATH_CALUDE_reading_completion_time_l11_1186


namespace NUMINAMATH_CALUDE_line_CR_tangent_to_circumcircle_l11_1175

-- Define the square ABCD
structure Square (A B C D : ℝ × ℝ) : Prop where
  is_square : A = (0, 0) ∧ B = (0, 1) ∧ C = (1, 1) ∧ D = (1, 0)

-- Define point P on BC
def P (k : ℝ) : ℝ × ℝ := (k, 1)

-- Define square APRS
structure SquareAPRS (A P R S : ℝ × ℝ) : Prop where
  is_square : A = (0, 0) ∧ P.1 = k ∧ P.2 = 1 ∧
              S = (1, -k) ∧ R = (1+k, 1-k)

-- Define the circumcircle of triangle ABC
def CircumcircleABC (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 0.5)^2 + (p.2 - 0.5)^2 = 0.5^2}

-- Define the line CR
def LineCR (C R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - C.2) = -1 * (p.1 - C.1)}

-- Theorem statement
theorem line_CR_tangent_to_circumcircle 
  (A B C D : ℝ × ℝ) 
  (k : ℝ) 
  (P R S : ℝ × ℝ) 
  (h1 : Square A B C D) 
  (h2 : 0 ≤ k ∧ k ≤ 1) 
  (h3 : P = (k, 1)) 
  (h4 : SquareAPRS A P R S) :
  ∃ (x : ℝ × ℝ), x ∈ CircumcircleABC A B C ∧ x ∈ LineCR C R ∧
  ∀ (y : ℝ × ℝ), y ≠ x → y ∈ CircumcircleABC A B C → y ∉ LineCR C R :=
sorry


end NUMINAMATH_CALUDE_line_CR_tangent_to_circumcircle_l11_1175


namespace NUMINAMATH_CALUDE_trigonometric_evaluations_l11_1144

open Real

theorem trigonometric_evaluations :
  (∃ (x : ℝ), x = sin (18 * π / 180) ∧ x = (Real.sqrt 5 - 1) / 4) ∧
  sin (18 * π / 180) * sin (54 * π / 180) = 1 / 4 ∧
  sin (36 * π / 180) * sin (72 * π / 180) = Real.sqrt 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_evaluations_l11_1144


namespace NUMINAMATH_CALUDE_dexter_sam_same_team_l11_1114

/-- The number of students in the dodgeball league -/
def total_students : ℕ := 12

/-- The number of players in each team -/
def team_size : ℕ := 6

/-- The number of students not including Dexter and Sam -/
def other_students : ℕ := total_students - 2

/-- The number of additional players needed to form a team with Dexter and Sam -/
def additional_players : ℕ := team_size - 2

theorem dexter_sam_same_team :
  (Nat.choose other_students additional_players) = 210 :=
sorry

end NUMINAMATH_CALUDE_dexter_sam_same_team_l11_1114


namespace NUMINAMATH_CALUDE_theater_casting_theorem_l11_1153

/-- Represents the number of ways to fill roles in a theater company. -/
def theater_casting_combinations (
  female_roles : Nat
) (male_roles : Nat) (
  neutral_roles : Nat
) (auditioning_men : Nat) (
  auditioning_women : Nat
) (qualified_lead_actresses : Nat) : Nat :=
  auditioning_men *
  qualified_lead_actresses *
  (auditioning_women - qualified_lead_actresses) *
  (auditioning_women - qualified_lead_actresses - 1) *
  (auditioning_men + auditioning_women - female_roles - male_roles) *
  (auditioning_men + auditioning_women - female_roles - male_roles - 1) *
  (auditioning_men + auditioning_women - female_roles - male_roles - 2)

/-- Theorem stating the number of ways to fill roles in the specific theater casting scenario. -/
theorem theater_casting_theorem :
  theater_casting_combinations 3 1 3 6 7 3 = 108864 := by
  sorry

end NUMINAMATH_CALUDE_theater_casting_theorem_l11_1153


namespace NUMINAMATH_CALUDE_negation_of_implication_l11_1140

theorem negation_of_implication (x : ℝ) :
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 ∧ x ≠ 1 ∧ x ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l11_1140


namespace NUMINAMATH_CALUDE_least_integer_abs_value_l11_1193

theorem least_integer_abs_value (y : ℤ) : 
  (∀ z : ℤ, 3 * |z| + 2 < 20 → y ≤ z) ↔ y = -5 := by sorry

end NUMINAMATH_CALUDE_least_integer_abs_value_l11_1193


namespace NUMINAMATH_CALUDE_solve_for_y_l11_1178

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l11_1178


namespace NUMINAMATH_CALUDE_parabola_properties_l11_1127

/-- A parabola is defined by its coefficients a, h, and k in the equation y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A parabola opens downwards if its 'a' coefficient is negative -/
def opens_downwards (p : Parabola) : Prop := p.a < 0

/-- The axis of symmetry of a parabola is the line x = h -/
def axis_of_symmetry (p : Parabola) (x : ℝ) : Prop := x = p.h

theorem parabola_properties (p : Parabola) :
  opens_downwards p ∧ axis_of_symmetry p 3 → p.a < 0 ∧ p.h = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l11_1127


namespace NUMINAMATH_CALUDE_cement_price_per_bag_l11_1148

theorem cement_price_per_bag 
  (cement_bags : ℕ) 
  (sand_lorries : ℕ) 
  (sand_tons_per_lorry : ℕ) 
  (sand_price_per_ton : ℕ) 
  (total_payment : ℕ) 
  (h1 : cement_bags = 500)
  (h2 : sand_lorries = 20)
  (h3 : sand_tons_per_lorry = 10)
  (h4 : sand_price_per_ton = 40)
  (h5 : total_payment = 13000) :
  (total_payment - sand_lorries * sand_tons_per_lorry * sand_price_per_ton) / cement_bags = 10 :=
by sorry

end NUMINAMATH_CALUDE_cement_price_per_bag_l11_1148


namespace NUMINAMATH_CALUDE_correct_sum_calculation_l11_1132

theorem correct_sum_calculation (tens_digit : Nat) : 
  let original_number := tens_digit * 10 + 9
  let mistaken_number := tens_digit * 10 + 6
  mistaken_number + 57 = 123 →
  original_number + 57 = 126 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_calculation_l11_1132


namespace NUMINAMATH_CALUDE_pig_count_l11_1199

theorem pig_count (initial_pigs additional_pigs : Float) 
  (h1 : initial_pigs = 64.0)
  (h2 : additional_pigs = 86.0) :
  initial_pigs + additional_pigs = 150.0 := by
sorry

end NUMINAMATH_CALUDE_pig_count_l11_1199


namespace NUMINAMATH_CALUDE_total_reams_is_five_l11_1176

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := 3

/-- The total number of reams of paper bought by Haley's mom -/
def total_reams : ℕ := reams_for_haley + reams_for_sister

theorem total_reams_is_five : total_reams = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_reams_is_five_l11_1176


namespace NUMINAMATH_CALUDE_circle_area_ratio_l11_1160

theorem circle_area_ratio (R : ℝ) (h : R > 0) : 
  let total_area := π * R^2
  let part_area := total_area / 8
  let shaded_area := 2 * part_area
  let unshaded_area := total_area - shaded_area
  shaded_area / unshaded_area = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l11_1160


namespace NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l11_1168

theorem tetrahedron_fits_in_box : ∃ (x y z : ℝ),
  (x^2 + y^2 = 100) ∧
  (x^2 + z^2 = 81) ∧
  (y^2 + z^2 = 64) ∧
  (x < 8) ∧ (y < 8) ∧ (z < 5) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_fits_in_box_l11_1168


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_810_l11_1170

theorem sin_n_equals_cos_810 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * Real.pi / 180) = Real.cos (810 * Real.pi / 180) →
  n = -180 ∨ n = 0 ∨ n = 180 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_810_l11_1170


namespace NUMINAMATH_CALUDE_power_sum_theorem_l11_1197

theorem power_sum_theorem (k : ℕ) :
  (∃ (n m : ℕ), m ≥ 2 ∧ 3^k + 5^k = n^m) → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l11_1197


namespace NUMINAMATH_CALUDE_stratified_sample_size_l11_1181

/-- Given a population with three groups in the ratio 2:3:5, 
    if a stratified sample contains 16 items from the first group, 
    then the total sample size is 80. -/
theorem stratified_sample_size 
  (population_ratio : Fin 3 → ℕ)
  (h_ratio : population_ratio = ![2, 3, 5])
  (sample_size : ℕ)
  (first_group_sample : ℕ)
  (h_first_group : first_group_sample = 16)
  (h_stratified : (population_ratio 0 : ℚ) / (population_ratio 0 + population_ratio 1 + population_ratio 2) 
                = first_group_sample / sample_size) :
  sample_size = 80 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l11_1181


namespace NUMINAMATH_CALUDE_circle_op_proof_l11_1185

def circle_op (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

theorem circle_op_proof (M N : Set ℕ) 
  (hM : M = {0, 2, 4, 6, 8, 10}) 
  (hN : N = {0, 3, 6, 9, 12, 15}) : 
  (circle_op (circle_op M N) M) = N := by
  sorry

#check circle_op_proof

end NUMINAMATH_CALUDE_circle_op_proof_l11_1185


namespace NUMINAMATH_CALUDE_unique_solution_l11_1149

/-- Represents the arithmetic operations and equality --/
inductive Operation
| Add
| Sub
| Mul
| Div
| Eq

/-- The set of equations given in the problem --/
def Equations (A B C D E : Operation) : Prop :=
  (4 / 2 = 2) ∧
  (8 = 4 * 2) ∧
  (2 + 3 = 5) ∧
  (4 = 5 - 1) ∧
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧
  (C ≠ D) ∧ (C ≠ E) ∧
  (D ≠ E)

/-- The theorem stating the unique solution to the problem --/
theorem unique_solution :
  ∃! (A B C D E : Operation),
    Equations A B C D E ∧
    A = Operation.Div ∧
    B = Operation.Eq ∧
    C = Operation.Mul ∧
    D = Operation.Add ∧
    E = Operation.Sub := by sorry

end NUMINAMATH_CALUDE_unique_solution_l11_1149


namespace NUMINAMATH_CALUDE_river_depth_calculation_l11_1117

/-- Proves that given a river with specified width, flow rate, and discharge,
    the depth of the river is as calculated. -/
theorem river_depth_calculation
  (width : ℝ)
  (flow_rate_kmph : ℝ)
  (discharge_per_minute : ℝ)
  (h1 : width = 25)
  (h2 : flow_rate_kmph = 8)
  (h3 : discharge_per_minute = 26666.666666666668) :
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  let depth := discharge_per_minute / (width * flow_rate_mpm)
  depth = 8 := by sorry

end NUMINAMATH_CALUDE_river_depth_calculation_l11_1117


namespace NUMINAMATH_CALUDE_equation_solution_l11_1109

theorem equation_solution (x : ℚ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l11_1109


namespace NUMINAMATH_CALUDE_infinite_rational_points_in_circle_l11_1137

theorem infinite_rational_points_in_circle : 
  ∀ ε > 0, ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ x^2 + y^2 ≤ 25 ∧ 
  ∀ (x' y' : ℚ), x' > 0 → y' > 0 → x'^2 + y'^2 ≤ 25 → (x - x')^2 + (y - y')^2 < ε^2 :=
sorry

end NUMINAMATH_CALUDE_infinite_rational_points_in_circle_l11_1137


namespace NUMINAMATH_CALUDE_hryzka_nuts_theorem_l11_1179

/-- Represents the two types of days in Hryzka's eating schedule -/
inductive DayType
  | Diet
  | Normal

/-- Calculates the number of nuts eaten on a given day type -/
def nutsEaten (d : DayType) : ℕ :=
  match d with
  | DayType.Diet => 1
  | DayType.Normal => 3

/-- Represents a sequence of day types -/
def Schedule := List DayType

/-- Generates an alternating schedule of the given length -/
def generateSchedule (startWithDiet : Bool) (length : ℕ) : Schedule :=
  sorry

/-- Calculates the total nuts eaten for a given schedule -/
def totalNutsEaten (s : Schedule) : ℕ :=
  sorry

theorem hryzka_nuts_theorem :
  let dietFirst := generateSchedule true 19
  let normalFirst := generateSchedule false 19
  (totalNutsEaten dietFirst = 37 ∧ totalNutsEaten normalFirst = 39) ∧
  (∀ (s : Schedule), s.length = 19 → totalNutsEaten s ≥ 37 ∧ totalNutsEaten s ≤ 39) :=
  sorry

#check hryzka_nuts_theorem

end NUMINAMATH_CALUDE_hryzka_nuts_theorem_l11_1179


namespace NUMINAMATH_CALUDE_min_value_quadratic_l11_1108

theorem min_value_quadratic (x y : ℝ) : 
  3 ≤ 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l11_1108


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l11_1121

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) 
  (container_capacity : ℕ) (eggs_in_deficient : ℕ) : 
  total_containers > 10 ∧ 
  deficient_containers = 3 ∧ 
  container_capacity = 15 ∧ 
  eggs_in_deficient = 13 →
  container_capacity * total_containers - 
    deficient_containers * (container_capacity - eggs_in_deficient) = 159 ∧
  159 > 150 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l11_1121


namespace NUMINAMATH_CALUDE_pencil_count_multiple_of_ten_l11_1162

/-- Given that 1230 pens and some pencils are distributed among students, 
    with each student receiving the same number of pens and pencils, 
    and the maximum number of students is 10, 
    prove that the total number of pencils is a multiple of 10. -/
theorem pencil_count_multiple_of_ten (total_pens : ℕ) (total_pencils : ℕ) (num_students : ℕ) :
  total_pens = 1230 →
  num_students ≤ 10 →
  num_students ∣ total_pens →
  num_students ∣ total_pencils →
  num_students = 10 →
  10 ∣ total_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_multiple_of_ten_l11_1162


namespace NUMINAMATH_CALUDE_complex_equality_l11_1192

theorem complex_equality (z : ℂ) : z = -1 + (7/2) * I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z + I) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l11_1192


namespace NUMINAMATH_CALUDE_insufficient_blue_points_l11_1119

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A tetrahedron formed by four points in 3D space -/
structure Tetrahedron where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Checks if a point is inside a tetrahedron -/
def isInside (t : Tetrahedron) (p : Point3D) : Prop := sorry

/-- The set of all tetrahedra formed by n red points -/
def allTetrahedra (redPoints : Finset Point3D) : Finset Tetrahedron := sorry

/-- Theorem: There exists a configuration of n red points such that 3n blue points
    are not sufficient to cover all tetrahedra formed by the red points -/
theorem insufficient_blue_points (n : ℕ) :
  ∃ (redPoints : Finset Point3D),
    redPoints.card = n ∧
    ∀ (bluePoints : Finset Point3D),
      bluePoints.card = 3 * n →
      ∃ (t : Tetrahedron),
        t ∈ allTetrahedra redPoints ∧
        ∀ (p : Point3D), p ∈ bluePoints → ¬isInside t p :=
sorry

end NUMINAMATH_CALUDE_insufficient_blue_points_l11_1119


namespace NUMINAMATH_CALUDE_four_digit_numbers_two_repeated_l11_1161

/-- The number of ways to choose 3 different digits from 0 to 9 -/
def three_digit_choices : ℕ := 10 * 9 * 8

/-- The number of ways to arrange 3 different digits with one repeated (forming a 4-digit number) -/
def repeated_digit_arrangements : ℕ := 6

/-- The number of four-digit numbers with exactly two repeated digits, including those starting with 0 -/
def total_with_leading_zero : ℕ := three_digit_choices * repeated_digit_arrangements

/-- The number of three-digit numbers with exactly two repeated digits (those starting with 0) -/
def starting_with_zero : ℕ := 9 * 8 * repeated_digit_arrangements

/-- The number of four-digit numbers with exactly two repeated digits -/
def four_digit_repeated : ℕ := total_with_leading_zero - starting_with_zero

theorem four_digit_numbers_two_repeated : four_digit_repeated = 3888 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_two_repeated_l11_1161


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l11_1169

theorem complex_number_magnitude_squared :
  ∀ (z : ℂ), z + Complex.abs z = 4 + 5*I → Complex.abs z^2 = 1681/64 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l11_1169


namespace NUMINAMATH_CALUDE_crease_line_equation_l11_1174

/-- Given a circle with radius R and a point A inside the circle at distance a from the center,
    the set of all points (x, y) on the crease lines formed by folding the paper so that any point
    on the circumference coincides with A satisfies the equation:
    (2x - a)^2 / R^2 + 4y^2 / (R^2 - a^2) = 1 -/
theorem crease_line_equation (R a x y : ℝ) (h1 : R > 0) (h2 : 0 ≤ a) (h3 : a < R) :
  (∃ (A' : ℝ × ℝ), (A'.1^2 + A'.2^2 = R^2) ∧
   ((x - A'.1)^2 + (y - A'.2)^2 = (x - a)^2 + y^2)) ↔
  (2*x - a)^2 / R^2 + 4*y^2 / (R^2 - a^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_crease_line_equation_l11_1174


namespace NUMINAMATH_CALUDE_int_coord_triangle_area_rational_l11_1173

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a triangle with three integer points
structure IntTriangle where
  p1 : IntPoint
  p2 : IntPoint
  p3 : IntPoint

-- Function to calculate the area of a triangle
def triangleArea (t : IntTriangle) : ℚ :=
  let x1 := t.p1.x
  let y1 := t.p1.y
  let x2 := t.p2.x
  let y2 := t.p2.y
  let x3 := t.p3.x
  let y3 := t.p3.y
  (1/2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

-- Theorem stating that the area of a triangle with integer coordinates is rational
theorem int_coord_triangle_area_rational (t : IntTriangle) : 
  ∃ q : ℚ, triangleArea t = q :=
sorry

end NUMINAMATH_CALUDE_int_coord_triangle_area_rational_l11_1173


namespace NUMINAMATH_CALUDE_expand_polynomial_l11_1190

theorem expand_polynomial (x : ℝ) : 
  (x + 3) * (4 * x^2 - 2 * x - 5) = 4 * x^3 + 10 * x^2 - 11 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l11_1190


namespace NUMINAMATH_CALUDE_complex_equation_sum_l11_1138

def complex_power (z : ℂ) (n : ℕ) := z ^ n

theorem complex_equation_sum (a b : ℝ) :
  (↑a + ↑b * Complex.I : ℂ) = complex_power Complex.I 2019 →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l11_1138


namespace NUMINAMATH_CALUDE_karen_took_one_sixth_l11_1147

/-- 
Given:
- Sasha added 48 cards to a box
- There were originally 43 cards in the box
- There are now 83 cards in the box

Prove that the fraction of cards Karen took out is 1/6
-/
theorem karen_took_one_sixth (cards_added : ℕ) (original_cards : ℕ) (final_cards : ℕ) 
  (h1 : cards_added = 48)
  (h2 : original_cards = 43)
  (h3 : final_cards = 83) :
  (cards_added + original_cards - final_cards : ℚ) / cards_added = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_karen_took_one_sixth_l11_1147


namespace NUMINAMATH_CALUDE_puppy_cost_first_year_l11_1136

def adoption_fee : ℝ := 150.00
def dog_food : ℝ := 40.00
def treats : ℝ := 3 * 5.00
def toys : ℝ := 2 * 25.00
def crate : ℝ := 120.00
def bed : ℝ := 80.00
def collar_leash : ℝ := 35.00
def grooming_tools : ℝ := 45.00
def training_classes : ℝ := 55.00 + 60.00 + 60.00 + 70.00 + 70.00
def discount_rate : ℝ := 0.12
def dog_license : ℝ := 25.00
def pet_insurance_first_half : ℝ := 6 * 25.00
def pet_insurance_second_half : ℝ := 6 * 30.00

def discountable_items : ℝ := dog_food + treats + toys + crate + bed + collar_leash + grooming_tools

theorem puppy_cost_first_year :
  let total_initial := adoption_fee + dog_food + treats + toys + crate + bed + collar_leash + grooming_tools + training_classes
  let discount := discount_rate * discountable_items
  let total_after_discount := total_initial - discount
  let total_insurance := pet_insurance_first_half + pet_insurance_second_half
  total_after_discount + dog_license + total_insurance = 1158.80 := by
sorry

end NUMINAMATH_CALUDE_puppy_cost_first_year_l11_1136


namespace NUMINAMATH_CALUDE_line_equation_sum_l11_1166

/-- Given two points on a line and the general form of the line equation,
    prove that the sum of the slope and y-intercept equals 7. -/
theorem line_equation_sum (m b : ℝ) : 
  (1 = m * (-3) + b) →   -- Point (-3,1) satisfies the equation
  (7 = m * 1 + b) →      -- Point (1,7) satisfies the equation
  m + b = 7 := by
sorry


end NUMINAMATH_CALUDE_line_equation_sum_l11_1166


namespace NUMINAMATH_CALUDE_max_mice_two_kittens_max_mice_two_males_l11_1184

/-- Represents the production possibility frontier (PPF) for a kitten --/
structure KittenPPF where
  maxMice : ℕ  -- Maximum number of mice caught when K = 0
  slope : ℚ    -- Rate of decrease in mice caught per hour of therapy

/-- Calculates the number of mice caught given hours of therapy --/
def micesCaught (ppf : KittenPPF) (therapyHours : ℚ) : ℚ :=
  ppf.maxMice - ppf.slope * therapyHours

/-- Male kitten PPF --/
def malePPF : KittenPPF := { maxMice := 80, slope := 4 }

/-- Female kitten PPF --/
def femalePPF : KittenPPF := { maxMice := 16, slope := 1/4 }

/-- Theorem: The maximum number of mice caught by 2 kittens is 160 --/
theorem max_mice_two_kittens :
  ∀ (k1 k2 : KittenPPF), ∀ (h1 h2 : ℚ),
    micesCaught k1 h1 + micesCaught k2 h2 ≤ 160 :=
by sorry

/-- Corollary: The maximum is achieved with two male kittens and zero therapy hours --/
theorem max_mice_two_males :
  micesCaught malePPF 0 + micesCaught malePPF 0 = 160 :=
by sorry

end NUMINAMATH_CALUDE_max_mice_two_kittens_max_mice_two_males_l11_1184


namespace NUMINAMATH_CALUDE_student_number_problem_l11_1139

theorem student_number_problem (x : ℝ) : 2 * x - 140 = 102 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l11_1139


namespace NUMINAMATH_CALUDE_friday_to_thursday_ratio_l11_1112

/-- Represents the study time for each day of the week -/
structure StudyTime where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  weekend : ℝ

/-- The study time satisfies the given conditions -/
def valid_study_time (st : StudyTime) : Prop :=
  st.wednesday = 2 ∧
  st.thursday = 3 * st.wednesday ∧
  st.weekend = st.wednesday + st.thursday + st.friday ∧
  st.wednesday + st.thursday + st.friday + st.weekend = 22

/-- The theorem to be proved -/
theorem friday_to_thursday_ratio (st : StudyTime) 
  (h : valid_study_time st) : st.friday / st.thursday = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_friday_to_thursday_ratio_l11_1112


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l11_1116

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxesEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxesEllipse) : ℝ := sorry

theorem ellipse_foci_distance 
  (e : ParallelAxesEllipse) 
  (h1 : e.x_tangent = (8, 0)) 
  (h2 : e.y_tangent = (0, 2)) : 
  foci_distance e = 4 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l11_1116


namespace NUMINAMATH_CALUDE_daisy_toys_count_l11_1126

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_toys (initial : ℕ) (bought_tuesday : ℕ) (bought_wednesday : ℕ) : ℕ :=
  initial + bought_tuesday + bought_wednesday

/-- Theorem stating the total number of Daisy's toys if all were found -/
theorem daisy_toys_count :
  total_toys 5 3 5 = 13 :=
by sorry

end NUMINAMATH_CALUDE_daisy_toys_count_l11_1126


namespace NUMINAMATH_CALUDE_no_x_exists_rational_l11_1195

theorem no_x_exists_rational : ¬ ∃ (x : ℝ), (∃ (a b : ℚ), (x + Real.sqrt 2 = a) ∧ (x^3 + Real.sqrt 2 = b)) := by
  sorry

end NUMINAMATH_CALUDE_no_x_exists_rational_l11_1195


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l11_1156

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ), ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
    (x^2 + 2) / ((x - 1) * (x - 4) * (x - 6)) = 
    P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
    P = 1/5 ∧ Q = -3 ∧ R = 19/5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l11_1156


namespace NUMINAMATH_CALUDE_press_conference_arrangement_l11_1151

/-- Number of reporters in each station -/
def n : ℕ := 5

/-- Total number of reporters to be selected -/
def k : ℕ := 4

/-- Number of ways to arrange questioning when selecting 1 from A and 3 from B -/
def case1 : ℕ := Nat.choose n 1 * Nat.choose n 3 * Nat.choose k 1 * (Nat.factorial 3)

/-- Number of ways to arrange questioning when selecting 2 from A and 2 from B -/
def case2 : ℕ := Nat.choose n 2 * Nat.choose n 2 * (2 * (Nat.factorial 2) * (Nat.factorial 2) + (Nat.factorial 2) * (Nat.factorial 2))

/-- Total number of ways to arrange the questioning -/
def total_ways : ℕ := case1 + case2

theorem press_conference_arrangement :
  total_ways = 2400 := by sorry

end NUMINAMATH_CALUDE_press_conference_arrangement_l11_1151


namespace NUMINAMATH_CALUDE_max_player_salary_l11_1123

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 18 →
  min_salary = 20000 →
  total_cap = 600000 →
  (∃ (salaries : Fin n → ℕ), 
    (∀ i, salaries i ≥ min_salary) ∧ 
    (Finset.sum Finset.univ salaries ≤ total_cap) ∧
    (∀ i, salaries i ≤ 260000) ∧
    (∃ j, salaries j = 260000)) ∧
  ¬(∃ (salaries : Fin n → ℕ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ total_cap) ∧
    (∃ j, salaries j > 260000)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_l11_1123


namespace NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l11_1146

theorem cinnamon_swirls_distribution (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → 
  num_people = 3 → 
  total_pieces = num_people * pieces_per_person →
  pieces_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l11_1146


namespace NUMINAMATH_CALUDE_yogurt_cases_l11_1131

theorem yogurt_cases (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_case : ℕ) 
  (h1 : total_cups = 960) 
  (h2 : cups_per_box = 6) 
  (h3 : boxes_per_case = 8) : 
  (total_cups / cups_per_box) / boxes_per_case = 20 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_cases_l11_1131


namespace NUMINAMATH_CALUDE_largest_integer_solution_l11_1122

theorem largest_integer_solution : 
  (∀ x : ℤ, 10 - 3*x > 25 → x ≤ -6) ∧ (10 - 3*(-6) > 25) := by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l11_1122


namespace NUMINAMATH_CALUDE_binomial_100_3_l11_1105

theorem binomial_100_3 : Nat.choose 100 3 = 161700 := by
  sorry

end NUMINAMATH_CALUDE_binomial_100_3_l11_1105


namespace NUMINAMATH_CALUDE_average_MTWT_is_48_l11_1167

/-- The average temperature for some days -/
def average_some_days : ℝ := 48

/-- The average temperature for Tuesday, Wednesday, Thursday, and Friday -/
def average_TWTF : ℝ := 46

/-- The temperature on Monday -/
def temp_Monday : ℝ := 42

/-- The temperature on Friday -/
def temp_Friday : ℝ := 34

/-- The number of days in the TWTF group -/
def num_days_TWTF : ℕ := 4

/-- The number of days in the MTWT group -/
def num_days_MTWT : ℕ := 4

/-- Theorem: The average temperature for Monday, Tuesday, Wednesday, and Thursday is 48 degrees -/
theorem average_MTWT_is_48 : 
  (temp_Monday + (average_TWTF * num_days_TWTF - temp_Friday)) / num_days_MTWT = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_MTWT_is_48_l11_1167


namespace NUMINAMATH_CALUDE_asterisk_value_for_solution_l11_1133

theorem asterisk_value_for_solution (x : ℝ) (asterisk : ℝ) :
  (2 * x - 7)^2 + (5 * x - asterisk)^2 = 0 → asterisk = 35/2 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_value_for_solution_l11_1133


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l11_1196

/-- Given two vectors in R², prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l11_1196


namespace NUMINAMATH_CALUDE_average_height_problem_l11_1142

/-- Given the heights of four people with specific relationships, prove their average height. -/
theorem average_height_problem (reese daisy parker giselle : ℝ) : 
  reese = 60 →
  daisy = reese + 8 →
  parker = daisy - 4 →
  giselle = parker - 2 →
  (reese + daisy + parker + giselle) / 4 = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_average_height_problem_l11_1142


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l11_1101

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept) + 0)

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l := Line.mk (-3) 7
  y_intercept l = (0, 21) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l11_1101


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l11_1150

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (-(1 : ℝ) / (1 + m) = -(m / 2))

/-- The first line equation -/
def line1 (m x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- The second line equation -/
def line2 (m x y : ℝ) : Prop :=
  m * x + 2 * y + 8 = 0

/-- The theorem stating that the lines are parallel if and only if m = 1 -/
theorem lines_parallel_iff_m_eq_one :
  ∀ m : ℝ, (∃ x y : ℝ, line1 m x y ∧ line2 m x y) →
    (parallel_lines m ↔ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l11_1150


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l11_1155

theorem product_of_sum_and_difference (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l11_1155


namespace NUMINAMATH_CALUDE_cosine_value_problem_l11_1129

theorem cosine_value_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 6)
  (h3 : Real.sin α ^ 6 + Real.cos α ^ 6 = 7 / 12) : 
  1998 * Real.cos α = 333 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_problem_l11_1129


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l11_1165

/-- Given three equally spaced parallel lines intersecting a circle and creating chords of lengths 40, 36, and 32, 
    the distance between two adjacent parallel lines is √(576/31). -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ), 
    chord1 = 40 ∧ 
    chord2 = 36 ∧ 
    chord3 = 32 ∧ 
    400 + (5/4) * d^2 = r^2 ∧ 
    256 + (36/4) * d^2 = r^2) → 
  d = Real.sqrt (576/31) := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l11_1165


namespace NUMINAMATH_CALUDE_product_sequence_sum_l11_1134

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l11_1134


namespace NUMINAMATH_CALUDE_first_field_rows_l11_1130

/-- Represents a corn field with a certain number of rows -/
structure CornField where
  rows : ℕ

/-- Represents a farm with two corn fields -/
structure Farm where
  field1 : CornField
  field2 : CornField

def corn_cobs_per_row : ℕ := 4

def total_corn_cobs (f : Farm) : ℕ :=
  (f.field1.rows + f.field2.rows) * corn_cobs_per_row

theorem first_field_rows (f : Farm) :
  f.field2.rows = 16 → total_corn_cobs f = 116 → f.field1.rows = 13 := by
  sorry

end NUMINAMATH_CALUDE_first_field_rows_l11_1130


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l11_1158

theorem soccer_ball_cost (total_cost : ℕ) (num_soccer_balls : ℕ) (num_volleyballs : ℕ) (volleyball_cost : ℕ) :
  total_cost = 980 ∧ num_soccer_balls = 5 ∧ num_volleyballs = 4 ∧ volleyball_cost = 65 →
  ∃ (soccer_ball_cost : ℕ), soccer_ball_cost = 144 ∧ 
    total_cost = num_soccer_balls * soccer_ball_cost + num_volleyballs * volleyball_cost :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l11_1158


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l11_1141

noncomputable def smallest_sum (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ x : ℝ, x^2 + (Real.sqrt 2 * a) * x + (Real.sqrt 2 * b) = 0) ∧
  (∃ x : ℝ, x^2 + (2 * b) * x + (Real.sqrt 2 * a) = 0) ∧
  a + b = (4 * Real.sqrt 2)^(2/3) / Real.sqrt 2 + (4 * Real.sqrt 2)^(1/3)

theorem smallest_sum_proof (a b : ℝ) :
  smallest_sum a b ↔ 
  (∀ c d : ℝ, c > 0 ∧ d > 0 ∧ 
   (∃ x : ℝ, x^2 + (Real.sqrt 2 * c) * x + (Real.sqrt 2 * d) = 0) ∧
   (∃ x : ℝ, x^2 + (2 * d) * x + (Real.sqrt 2 * c) = 0) →
   c + d ≥ (4 * Real.sqrt 2)^(2/3) / Real.sqrt 2 + (4 * Real.sqrt 2)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l11_1141


namespace NUMINAMATH_CALUDE_pipe_A_fill_time_l11_1182

-- Define the time it takes for Pipe A to fill the tank
variable (A : ℝ)

-- Define the time it takes for Pipe B to empty the tank
def B : ℝ := 24

-- Define the total time to fill the tank when both pipes are used
def total_time : ℝ := 30

-- Define the time Pipe B is open
def B_open_time : ℝ := 24

-- Define the theorem
theorem pipe_A_fill_time :
  (1 / A - 1 / B) * B_open_time + (1 / A) * (total_time - B_open_time) = 1 →
  A = 15 := by
sorry

end NUMINAMATH_CALUDE_pipe_A_fill_time_l11_1182


namespace NUMINAMATH_CALUDE_pizza_size_increase_l11_1110

theorem pizza_size_increase (r : ℝ) (hr : r > 0) : 
  let medium_area := π * r^2
  let large_radius := 1.1 * r
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area = 0.21
  := by sorry

end NUMINAMATH_CALUDE_pizza_size_increase_l11_1110


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l11_1120

theorem power_sum_equals_two : (-1)^2 + (1/3)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l11_1120


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_600_degrees_l11_1145

theorem angle_with_same_terminal_side_as_600_degrees :
  ∀ α : ℝ, (∃ k : ℤ, α = 600 + k * 360) → (∃ k : ℤ, α = k * 360 + 240) :=
by sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_600_degrees_l11_1145


namespace NUMINAMATH_CALUDE_second_number_proof_l11_1100

theorem second_number_proof (a b : ℝ) (h1 : a = 50) (h2 : 0.6 * a - 0.3 * b = 27) : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l11_1100


namespace NUMINAMATH_CALUDE_product_inequality_l11_1180

theorem product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l11_1180


namespace NUMINAMATH_CALUDE_tim_has_five_marbles_l11_1118

/-- The number of blue marbles Fred has -/
def fred_marbles : ℕ := 110

/-- The ratio of Fred's marbles to Tim's marbles -/
def ratio : ℕ := 22

/-- The number of blue marbles Tim has -/
def tim_marbles : ℕ := fred_marbles / ratio

theorem tim_has_five_marbles : tim_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_five_marbles_l11_1118


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l11_1159

theorem integer_pairs_satisfying_equation :
  {(x, y) : ℤ × ℤ | 8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y} =
  {(0, 0), (1, 1), (-1, -1)} :=
by sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l11_1159


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l11_1111

theorem product_pure_imaginary (b : ℝ) : 
  let Z1 : ℂ := 3 - 4*I
  let Z2 : ℂ := 4 + b*I
  (∃ (y : ℝ), Z1 * Z2 = y*I) → b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l11_1111


namespace NUMINAMATH_CALUDE_section_area_theorem_l11_1191

/-- Regular quadrilateral pyramid with given properties -/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  -- Distance from apex to cutting plane
  apex_distance : ℝ

/-- Area of the section formed by a plane in the pyramid -/
def section_area (p : RegularPyramid) : ℝ := sorry

/-- Theorem stating the area of the section for the given pyramid -/
theorem section_area_theorem (p : RegularPyramid) 
  (h1 : p.base_side = 8 / Real.sqrt 7)
  (h2 : p.apex_distance = 2 / 3) : 
  section_area p = 6 := by sorry

end NUMINAMATH_CALUDE_section_area_theorem_l11_1191


namespace NUMINAMATH_CALUDE_vehicles_separation_time_l11_1125

/-- Given two vehicles moving in opposite directions, calculate the time taken to reach a specific distance apart. -/
theorem vehicles_separation_time
  (initial_distance : ℝ)
  (speed1 speed2 : ℝ)
  (final_distance : ℝ)
  (h1 : initial_distance = 5)
  (h2 : speed1 = 60)
  (h3 : speed2 = 40)
  (h4 : final_distance = 85) :
  (final_distance - initial_distance) / (speed1 + speed2) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_vehicles_separation_time_l11_1125


namespace NUMINAMATH_CALUDE_sine_function_midpoint_l11_1177

/-- Given a sine function y = A sin(Bx + C) + D that oscillates between 6 and 2, prove that D = 4 -/
theorem sine_function_midpoint (A B C D : ℝ) 
  (h_oscillation : ∀ x, 2 ≤ A * Real.sin (B * x + C) + D ∧ A * Real.sin (B * x + C) + D ≤ 6) : 
  D = 4 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_midpoint_l11_1177


namespace NUMINAMATH_CALUDE_saturday_visitors_count_l11_1107

def friday_visitors : ℕ := 12315
def saturday_multiplier : ℕ := 7

theorem saturday_visitors_count : friday_visitors * saturday_multiplier = 86205 := by
  sorry

end NUMINAMATH_CALUDE_saturday_visitors_count_l11_1107


namespace NUMINAMATH_CALUDE_smallest_n_for_no_real_roots_l11_1163

theorem smallest_n_for_no_real_roots :
  ∀ n : ℤ, (∀ x : ℝ, 3 * x * (n * x + 3) - 2 * x^2 - 9 ≠ 0) →
  n ≥ -1 ∧ ∀ m : ℤ, m < -1 → ∃ x : ℝ, 3 * x * (m * x + 3) - 2 * x^2 - 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_no_real_roots_l11_1163


namespace NUMINAMATH_CALUDE_inverse_function_decomposition_l11_1198

noncomputable section

def PeriodOn (h : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, h (x + d) = h x

def IsPeriodic (h : ℝ → ℝ) : Prop :=
  ∃ d ≠ 0, PeriodOn h d

def MutuallyInverse (f g : ℝ → ℝ) : Prop :=
  (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_function_decomposition
  (f g : ℝ → ℝ)
  (h : ℝ → ℝ)
  (k : ℝ)
  (h_inv : MutuallyInverse f g)
  (h_periodic : IsPeriodic h)
  (h_decomp : ∀ x, f x = k * x + h x) :
  ∃ p : ℝ → ℝ, (IsPeriodic p) ∧ (∀ y, g y = (1/k) * y + p y) :=
sorry

end NUMINAMATH_CALUDE_inverse_function_decomposition_l11_1198


namespace NUMINAMATH_CALUDE_account_balance_difference_l11_1152

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between Angela's and Bob's account balances after 15 years -/
theorem account_balance_difference : 
  let angela_balance := compound_interest 9000 0.05 15
  let bob_balance := simple_interest 11000 0.06 15
  ⌊|bob_balance - angela_balance|⌋ = 2189 := by
sorry

end NUMINAMATH_CALUDE_account_balance_difference_l11_1152


namespace NUMINAMATH_CALUDE_calculate_expression_l11_1113

theorem calculate_expression : (-2)^3 + Real.sqrt 12 + (1/3)⁻¹ = 2 * Real.sqrt 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l11_1113


namespace NUMINAMATH_CALUDE_total_pencils_l11_1115

theorem total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) :
  initial_pencils = 37 → additional_pencils = 17 →
  initial_pencils + additional_pencils = 54 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l11_1115


namespace NUMINAMATH_CALUDE_sum_of_specific_digits_l11_1164

/-- A sequence where each positive integer n is repeated n times in increasing order -/
def special_sequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => sorry

/-- The nth digit of the special sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the 4501st and 4052nd digits of the special sequence is 13 -/
theorem sum_of_specific_digits :
  nth_digit 4501 + nth_digit 4052 = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_digits_l11_1164


namespace NUMINAMATH_CALUDE_sherlock_lock_combination_l11_1183

def is_valid_solution (d : ℕ) (S E N D R : ℕ) : Prop :=
  S < d ∧ E < d ∧ N < d ∧ D < d ∧ R < d ∧
  S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ R ∧
  E ≠ N ∧ E ≠ D ∧ E ≠ R ∧
  N ≠ D ∧ N ≠ R ∧
  D ≠ R ∧
  (S * d^3 + E * d^2 + N * d + D) +
  (E * d^2 + N * d + D) +
  (R * d^2 + E * d + D) =
  (D * d^3 + E * d^2 + E * d + R)

theorem sherlock_lock_combination :
  ∃ (d : ℕ), ∃ (S E N D R : ℕ),
    is_valid_solution d S E N D R ∧
    R * d^2 + E * d + D = 879 :=
sorry

end NUMINAMATH_CALUDE_sherlock_lock_combination_l11_1183


namespace NUMINAMATH_CALUDE_quadratic_downwards_condition_l11_1128

/-- A quadratic function of the form y = (2a-6)x^2 + 4 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (2*a - 6)*x^2 + 4

/-- The condition for a quadratic function to open downwards -/
def opens_downwards (a : ℝ) : Prop := 2*a - 6 < 0

theorem quadratic_downwards_condition (a : ℝ) :
  opens_downwards a → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_downwards_condition_l11_1128


namespace NUMINAMATH_CALUDE_triangle_inequalities_l11_1106

/-- For any triangle ABC with exradii r_a, r_b, r_c, inradius r, and circumradius R -/
theorem triangle_inequalities (r_a r_b r_c r R : ℝ) (h_positive : r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0 ∧ R > 0) :
  r_a^2 + r_b^2 + r_c^2 ≥ 27 * r^2 ∧ 4 * R < r_a + r_b + r_c ∧ r_a + r_b + r_c ≤ 9/2 * R := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l11_1106


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l11_1143

theorem smallest_number_divisible (h : ℕ) : 
  (∀ n : ℕ, n < 259 → ¬(((n + 5) % 8 = 0) ∧ ((n + 5) % 11 = 0) ∧ ((n + 5) % 24 = 0))) ∧
  ((259 + 5) % 8 = 0) ∧ ((259 + 5) % 11 = 0) ∧ ((259 + 5) % 24 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l11_1143
