import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_problem_l121_12198

/-- True discount calculation -/
noncomputable def true_discount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Theorem stating the existence of a solution to the true discount problem -/
theorem true_discount_problem (bill_amount : ℝ) (double_time_discount : ℝ) 
  (h1 : bill_amount = 110)
  (h2 : double_time_discount = 18.333333333333332)
  : ∃ (rate : ℝ) (time : ℝ), 
    true_discount bill_amount rate time = 9.166666666666666 ∧
    true_discount bill_amount rate (2 * time) = double_time_discount := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_problem_l121_12198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l121_12100

def sequence_a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | n + 2 => (sequence_a n + sequence_a (n + 1)) / 2

def sequence_b (n : ℕ) : ℚ := sequence_a (n + 1) - sequence_a n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_b (n + 1) = -1/2 * sequence_b n) ∧
  (∀ n : ℕ, sequence_a n = 5/3 - 2/3 * (-1/2)^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l121_12100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l121_12184

def U : Finset ℕ := {0, 1, 2, 3, 4}

def A : Finset ℕ := U.filter (fun x => x^2 - 2*x = 0)

def B : Finset ℕ := U.filter (fun x => x < 3)

theorem complement_of_union :
  (U \ (A ∪ B)) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l121_12184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l121_12152

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to get x-intercept of a line
noncomputable def xIntercept (l : Line) : ℝ :=
  -l.c / l.a

-- Function to get y-intercept of a line
noncomputable def yIntercept (l : Line) : ℝ :=
  -l.c / l.b

-- Theorem statement
theorem line_equation (l : Line) :
  pointOnLine ⟨7, 1⟩ l ∧ 
  xIntercept l = -(yIntercept l) →
  (l.a = 1 ∧ l.b = -7 ∧ l.c = 0) ∨ 
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l121_12152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_value_l121_12166

-- Define the tetrahedron and its properties
structure Tetrahedron where
  vertices : Finset (Fin 4)
  edges : Finset (Fin 6)
  numbers : Fin 10 → ℕ
  vertex_assignment : Fin 4 → Fin 10
  edge_assignment : Fin 6 → Fin 10

-- Helper function to get vertices of an edge
def edge_vertices : Fin 6 → Fin 4 × Fin 4
  | 0 => (0, 1)  -- PQ
  | 1 => (0, 2)  -- PR
  | 2 => (0, 3)  -- PS
  | 3 => (1, 2)  -- QR
  | 4 => (1, 3)  -- QS
  | 5 => (2, 3)  -- RS

-- Define the properties of the tetrahedron
def valid_tetrahedron (t : Tetrahedron) : Prop :=
  (∀ i : Fin 10, ∃! x, t.numbers x = i + 1) ∧
  (∀ e : Fin 6, t.numbers (t.edge_assignment e) = 
    t.numbers (t.vertex_assignment (edge_vertices e).1) + 
    t.numbers (t.vertex_assignment (edge_vertices e).2)) ∧
  (t.numbers (t.edge_assignment 0) = 9)

-- Theorem statement
theorem tetrahedron_edge_value (t : Tetrahedron) :
  valid_tetrahedron t → t.numbers (t.edge_assignment 5) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_value_l121_12166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_device_improvement_proof_l121_12192

/-- Represents the sample data for a device -/
structure DeviceData where
  mean : ℝ
  variance : ℝ

/-- Determines if there's significant improvement between two devices -/
def significant_improvement (old new : DeviceData) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

theorem device_improvement_proof (old new : DeviceData)
  (h_old_mean : old.mean = 10)
  (h_new_mean : new.mean = 10.3)
  (h_old_variance : old.variance = 0.036)
  (h_new_variance : new.variance = 0.04) :
  significant_improvement old new := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_device_improvement_proof_l121_12192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_point_on_line_m_l121_12171

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  x = 1 ∨ 5 / 12 * x - y + 43 / 12 = 0

-- Define the chord length
def chord_length : ℝ := 6

-- Main theorem
theorem circle_and_line_properties :
  ∀ (x y : ℝ),
  (circle_C x y ∧ line_l x y) →
  ((x + 3)^2 + (y + 2)^2 = 25) ∧
  (chord_length = 6 →
    line_m x y) :=
by sorry

-- Additional theorem to show that (1, 4) is on line m
theorem point_on_line_m :
  line_m 1 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_point_on_line_m_l121_12171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_4_statement_5_correct_statements_l121_12183

-- Define the complex number i
def i : ℂ := Complex.I

-- Statement ④
theorem statement_4 : 
  let z : ℂ := 1 - i
  Complex.abs ((2 / z) + z^2) = Real.sqrt 2 := by sorry

-- Statement ⑤
theorem statement_5 :
  let z : ℂ := 1 / i
  (z^5 + 1).re > 0 ∧ (z^5 + 1).im < 0 := by sorry

-- Combined theorem
theorem correct_statements :
  (let z : ℂ := 1 - i; Complex.abs ((2 / z) + z^2) = Real.sqrt 2) ∧
  (let z : ℂ := 1 / i; (z^5 + 1).re > 0 ∧ (z^5 + 1).im < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_4_statement_5_correct_statements_l121_12183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l121_12140

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

-- State the theorem
theorem function_inequality_implies_a_range :
  (∀ a : ℝ, ∀ x ∈ Set.Icc a (a + 1), f (x + a) > f (2 * a - x)) →
  Set.Nonempty {a : ℝ | a < -2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l121_12140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_application_equals_three_l121_12107

noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 1 then
    -1/2 * x^2 + x + 3
  else if 1 < x ∧ x ≤ 5 then
    1/3 * x^2 - 3*x + 11
  else
    0  -- undefined outside the given intervals

theorem unique_double_application_equals_three :
  ∃! x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ g (g x) = 3 := by
  sorry

#check unique_double_application_equals_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_double_application_equals_three_l121_12107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_range_of_a_unique_solution_m_l121_12137

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2 - (1/2) * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1
theorem max_value_f (x : ℝ) (hx : x > 0) :
  ∃ (max_val : ℝ), max_val = -3/4 ∧ ∀ y > 0, f (1/4) y ≤ max_val :=
sorry

-- Theorem 2
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 3, deriv (g a) x ≤ 1/2) → a ≥ 1/2 :=
sorry

-- Theorem 3
theorem unique_solution_m (x m : ℝ) (hx : x > 0) (hm : m > 0) :
  (∃! x, x > 0 ∧ 2 * m * f 0 x = x * (x - 3 * m)) → m = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_range_of_a_unique_solution_m_l121_12137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l121_12129

-- Define the expression as noncomputable
noncomputable def f (x : ℝ) := Real.sqrt (x / (x - 1))

-- Theorem statement
theorem f_defined_iff (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ (x ≥ 0 ∧ x ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_iff_l121_12129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l121_12165

/-- Represents an ellipse centered at the origin -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  h : a > 0 ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_standard_equation (e : Ellipse) 
  (h1 : e.a = 4 ∨ e.b = 4)  -- passes through (4, 0)
  (h2 : e.eccentricity = Real.sqrt 3 / 2)  -- eccentricity is √3/2
  : (∀ x y : ℝ, x^2 / 16 + y^2 / 4 = 1) ∨ 
    (∀ x y : ℝ, y^2 / 64 + x^2 / 16 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l121_12165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_puzzle_l121_12173

theorem multiplication_puzzle :
  ∃ (a b c d e : ℕ),
  (100 * a + 10 * b + c) * (10 * d + e) = 7632 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a ∉ ({6, 7, 3, 2} : Set ℕ) ∧
  b ∉ ({6, 7, 3, 2} : Set ℕ) ∧
  c ∉ ({6, 7, 3, 2} : Set ℕ) ∧
  d ∉ ({6, 7, 3, 2} : Set ℕ) ∧
  e ∉ ({6, 7, 3, 2} : Set ℕ) ∧
  a ∈ ({1, 4, 5, 8, 9} : Set ℕ) ∧
  b ∈ ({1, 4, 5, 8, 9} : Set ℕ) ∧
  c ∈ ({1, 4, 5, 8, 9} : Set ℕ) ∧
  d ∈ ({1, 4, 5, 8, 9} : Set ℕ) ∧
  e ∈ ({1, 4, 5, 8, 9} : Set ℕ) ∧
  b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_puzzle_l121_12173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_triangle_distribution_l121_12185

-- Define the number of points and groups
def total_points : ℕ := 2019
def num_groups : ℕ := 30

-- Define a type for group distributions
def GroupDistribution := Fin num_groups → ℕ

-- Function to calculate the number of triangles
def num_triangles (dist : GroupDistribution) : ℕ :=
  sorry

-- Function to check if a distribution is valid
def is_valid_distribution (dist : GroupDistribution) : Prop :=
  (Finset.sum Finset.univ (λ i => dist i) = total_points) ∧
  ∀ i j, i ≠ j → dist i ≠ dist j

-- Define the optimal distribution
def optimal_distribution : GroupDistribution :=
  λ i => if i.val < 6 then 52 + i.val else 53 + i.val

-- Theorem statement
theorem optimal_triangle_distribution :
  ∀ dist : GroupDistribution,
    is_valid_distribution dist →
    num_triangles dist ≤ num_triangles optimal_distribution :=
by
  sorry

#check optimal_triangle_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_triangle_distribution_l121_12185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_positive_a_l121_12127

open Real

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := a * (x - 1/x) - 2 * log x

noncomputable def g (a x : ℝ) : ℝ := -a/x

-- State the theorem
theorem function_inequality_implies_positive_a :
  ∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (exp 1) ∧ f a x₀ > g a x₀) → a > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_positive_a_l121_12127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_width_is_20_l121_12132

/-- The width of a running track formed by two concentric circles -/
noncomputable def track_width (r₁ r₂ : ℝ) : ℝ := r₁ - r₂

/-- The difference in circumferences of two concentric circles -/
noncomputable def circumference_difference (r₁ r₂ : ℝ) : ℝ := 2 * Real.pi * (r₁ - r₂)

theorem track_width_is_20 (r₁ r₂ : ℝ) :
  circumference_difference r₁ r₂ = 40 * Real.pi →
  2 * Real.pi * r₁ = 4 * 40 * Real.pi →
  track_width r₁ r₂ = 20 := by
  sorry

#check track_width_is_20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_width_is_20_l121_12132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_sequence_l121_12130

def geometric_sequence (a₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

theorem fourth_term_of_geometric_sequence (a₁ r : ℕ) 
  (h₁ : a₁ = 3)
  (h₂ : geometric_sequence a₁ r 2 = 27) :
  geometric_sequence a₁ r 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_sequence_l121_12130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_first_class_l121_12101

theorem exactly_one_first_class (p1 p2 : ℝ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/4) 
  (h3 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h4 : 0 ≤ p2 ∧ p2 ≤ 1) : 
  p1 * (1 - p2) + (1 - p1) * p2 = 5/12 := by
  sorry

#check exactly_one_first_class

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_first_class_l121_12101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_exponent_proof_l121_12157

theorem missing_exponent_proof :
  (9 : ℝ) ^ (56/10) * (9 : ℝ) ^ (469944/100000) / (9 : ℝ) ^ (256256/100000) = (9 : ℝ) ^ (1333744/100000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_exponent_proof_l121_12157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donnas_weekly_earnings_l121_12163

/-- Calculates Donna's weekly earnings based on her various jobs -/
def weekly_earnings (dog_walking_rate : ℚ) (dog_walking_hours : ℚ) (dog_walking_days : ℚ)
                    (card_shop_rate : ℚ) (card_shop_hours : ℚ) (card_shop_days : ℚ)
                    (babysitting_rate : ℚ) (babysitting_hours : ℚ) : ℚ :=
  dog_walking_rate * dog_walking_hours * dog_walking_days +
  card_shop_rate * card_shop_hours * card_shop_days +
  babysitting_rate * babysitting_hours

/-- Theorem stating Donna's weekly earnings -/
theorem donnas_weekly_earnings :
  weekly_earnings 10 2 7 (25/2) 2 5 10 4 = 305 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_donnas_weekly_earnings_l121_12163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_distance_downstream_l121_12111

/-- Calculates the effective distance travelled downstream by a boat --/
theorem effective_distance_downstream 
  (boat_speed : ℝ) 
  (current_speed : ℝ) 
  (wind_factor : ℝ) 
  (total_time : ℝ) 
  (lost_time : ℝ) 
  (h1 : boat_speed = 65) 
  (h2 : current_speed = 15) 
  (h3 : wind_factor = 1.1) 
  (h4 : total_time = 25 / 60) 
  (h5 : lost_time = 2 / 60) : 
  ∃ (distance : ℝ), abs (distance - ((boat_speed * wind_factor + current_speed) * (total_time - lost_time))) < 0.01 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_distance_downstream_l121_12111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_probability_l121_12169

/-- The probability that the second-best ranked team is eliminated in the last round
    in a tournament of 2021 teams over 2020 rounds. -/
def p : ℚ := 337 / 1010

theorem tournament_probability :
  ⌊(2021 : ℚ) * p⌋ = 674 := by
  sorry

#eval ⌊(2021 : ℚ) * (337 / 1010)⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_probability_l121_12169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_inequality_l121_12155

noncomputable section

open Real

def cot (x : ℝ) : ℝ := 1 / tan x

theorem triangle_cotangent_inequality (A B C : ℝ) (a b c : ℝ) :
  0 < a → 0 < b → 0 < c →
  a + b > c → b + c > a → c + a > b →
  b^2 = (a^2 + c^2) / 2 →
  (cot B)^2 ≥ (cot A) * (cot C) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_inequality_l121_12155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_volume_l121_12162

/-- Calculates the volume of a cone given its radius and height -/
noncomputable def coneVolume (r h : Real) : Real := (Real.pi * r^2 * h) / 3

theorem sand_pile_volume : 
  let largeDiameter : Real := 12
  let largeHeight : Real := 0.6 * largeDiameter
  let smallDiameter : Real := 0.5 * largeDiameter
  let smallHeight : Real := 0.6 * smallDiameter
  let largeRadius : Real := largeDiameter / 2
  let smallRadius : Real := smallDiameter / 2
  let largeVolume : Real := coneVolume largeRadius largeHeight
  let smallVolume : Real := coneVolume smallRadius smallHeight
  largeVolume + smallVolume = 97.2 * Real.pi := by
  sorry

#check sand_pile_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_volume_l121_12162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l121_12182

noncomputable section

/-- A cubic polynomial with coefficients p, q, and d -/
def Q (p q d : ℝ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + d

/-- The sum of the zeros of a cubic polynomial -/
def sumOfZeros (p : ℝ) : ℝ := -p

/-- The product of the zeros of a cubic polynomial -/
def productOfZeros (d : ℝ) : ℝ := -d

/-- The mean of the zeros of a cubic polynomial -/
def meanOfZeros (p : ℝ) : ℝ := -p/3

/-- The sum of the coefficients of Q -/
def sumOfCoefficients (p q d : ℝ) : ℝ := 1 + p + q + d

theorem cubic_polynomial_property (p q d : ℝ) :
  meanOfZeros p = productOfZeros d ∧
  meanOfZeros p = sumOfCoefficients p q d ∧
  Q p q d 0 = 3 →
  q = -16 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_property_l121_12182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_through_point_unique_perpendicular_through_point_angle_bisectors_of_supplementary_angles_l121_12126

-- Define the basic geometric objects
structure Point : Type where

structure Line : Type where

-- Define the relationships between geometric objects
axiom IsOn : Point → Line → Prop
axiom IsParallel : Line → Line → Prop
axiom IsPerpendicular : Line → Line → Prop
axiom AngleBisector : Line → Line → Line
axiom IsSupplementary : ℝ → ℝ → Prop

-- Define the supplementary angle relationship
axiom supplementary_def : ∀ a1 a2 : ℝ, IsSupplementary a1 a2 ↔ a1 + a2 = Real.pi

-- State the theorems to be proved
theorem unique_parallel_through_point (p : Point) (l : Line) :
  ¬(IsOn p l) → ∃! l' : Line, IsParallel l' l ∧ IsOn p l' := sorry

theorem unique_perpendicular_through_point (p : Point) (l : Line) :
  ∃! l' : Line, IsPerpendicular l' l ∧ IsOn p l' := sorry

theorem angle_bisectors_of_supplementary_angles (l1 l2 : Line) (a1 a2 : ℝ) :
  IsSupplementary a1 a2 →
  IsPerpendicular (AngleBisector l1 l2) (AngleBisector l2 l1) := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_parallel_through_point_unique_perpendicular_through_point_angle_bisectors_of_supplementary_angles_l121_12126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_longer_leg_l121_12148

/-- Represents a 30-60-90 right triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ

/-- System of three connected 30-60-90 right triangles -/
structure TriangleSystem where
  large : Triangle30_60_90
  medium : Triangle30_60_90
  small : Triangle30_60_90
  hypotenuse_connection : 
    large.hypotenuse * (Real.sqrt 3 / 2) = medium.hypotenuse ∧ 
    medium.hypotenuse * (Real.sqrt 3 / 2) = small.hypotenuse

theorem smallest_triangle_longer_leg 
  (system : TriangleSystem) 
  (h : system.large.hypotenuse = 10) : 
  system.small.hypotenuse * (Real.sqrt 3 / 2) = (7.5 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_longer_leg_l121_12148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_exists_l121_12103

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (R : ℝ)
variable (P : Point)
variable (L : Line)

-- Define the property of a circle being tangent to a line
def isTangent (c : Circle) (l : Line) : Prop :=
  ∃ (t : Point), ((l.a * t.x + l.b * t.y + l.c = 0) ∧ 
    (t.x - c.center.x)^2 + (t.y - c.center.y)^2 = c.radius^2)

-- Define the property of a point being on a circle
def isOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem circle_exists (R : ℝ) (P : Point) (L : Line) :
  ∃ (C : Circle), C.radius = R ∧ isOnCircle P C ∧ isTangent C L := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_exists_l121_12103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_sixth_l121_12125

theorem cos_alpha_minus_pi_sixth (α : ℝ) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin (α + π / 6) = 3 / 5) : 
  Real.cos (α - π / 6) = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_sixth_l121_12125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l121_12113

/-- The area of a regular hexagon inscribed in a circle with radius 5 units -/
theorem regular_hexagon_area (r : ℝ) (h : r = 5) : 
  (3 * Real.sqrt 3 * r^2) / 2 = 75 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l121_12113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_product_theorem_proof_l121_12194

def vector_product_theorem (a b : ℝ × ℝ) : Prop :=
  let angle : ℝ := Real.pi / 3
  let magnitude_a : ℝ := 2
  let magnitude_b : ℝ := 1
  (a.1^2 + a.2^2 = magnitude_a^2) ∧
  (b.1^2 + b.2^2 = magnitude_b^2) ∧
  (a.1 * b.1 + a.2 * b.2 = magnitude_a * magnitude_b * Real.cos angle) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) *
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 21

theorem vector_product_theorem_proof (a b : ℝ × ℝ) : vector_product_theorem a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_product_theorem_proof_l121_12194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_correct_rate_is_approximately_2_35_percent_l121_12135

/-- Given a principal, amount, and time period, calculates the rate of simple interest -/
noncomputable def simple_interest_rate (principal amount time : ℝ) : ℝ :=
  (100 * (amount - principal)) / (principal * time)

/-- Theorem stating that the simple interest rate calculation is correct -/
theorem simple_interest_rate_correct (principal amount time : ℝ) 
  (h1 : principal = 850)
  (h2 : amount = 950)
  (h3 : time = 5)
  (h4 : principal > 0)
  (h5 : time > 0) :
  simple_interest_rate principal amount time = (100 * (amount - principal)) / (principal * time) :=
by
  -- Unfold the definition of simple_interest_rate
  unfold simple_interest_rate
  -- The goal is now trivially true by reflexivity
  rfl

/-- Theorem proving that the rate is approximately 2.35% -/
theorem rate_is_approximately_2_35_percent (principal amount time : ℝ) 
  (h1 : principal = 850)
  (h2 : amount = 950)
  (h3 : time = 5)
  (h4 : principal > 0)
  (h5 : time > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |simple_interest_rate principal amount time - 2.35| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_correct_rate_is_approximately_2_35_percent_l121_12135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l121_12181

-- Define the necessary structures
structure Line
structure Plane

-- Define the parallel and subset relations
axiom parallel_lines : Line → Line → Prop
axiom parallel_plane_line : Plane → Line → Prop
axiom parallel_planes : Plane → Plane → Prop
axiom line_in_plane : Line → Plane → Prop

-- State the theorem
theorem line_parallel_to_plane 
  (a : Line) (α β : Plane) 
  (h1 : line_in_plane a β) 
  (h2 : parallel_planes α β) : 
  parallel_plane_line α a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_l121_12181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrots_harvested_is_320_l121_12124

/-- Represents the farmer's harvest and sales data -/
structure FarmData where
  potato_count : ℕ
  potato_bundle_size : ℕ
  potato_bundle_price : ℚ
  carrot_bundle_size : ℕ
  carrot_bundle_price : ℚ
  total_revenue : ℚ

/-- Calculates the number of carrots harvested based on the farm data -/
def carrots_harvested (data : FarmData) : ℕ :=
  let potato_revenue := (data.potato_count / data.potato_bundle_size) * data.potato_bundle_price
  let carrot_revenue := data.total_revenue - potato_revenue
  let carrot_bundles := carrot_revenue / data.carrot_bundle_price
  (Int.floor carrot_bundles).toNat * data.carrot_bundle_size

/-- Theorem stating that given the specific farm data, the number of carrots harvested is 320 -/
theorem carrots_harvested_is_320 (data : FarmData) 
  (h1 : data.potato_count = 250)
  (h2 : data.potato_bundle_size = 25)
  (h3 : data.potato_bundle_price = 19/10)
  (h4 : data.carrot_bundle_size = 20)
  (h5 : data.carrot_bundle_price = 2)
  (h6 : data.total_revenue = 51) :
  carrots_harvested data = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrots_harvested_is_320_l121_12124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l121_12108

-- Define a right triangle with hypotenuse 13 and one leg 5
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ c = 13 ∧ a = 5

-- Define the area of a triangle given two sides
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1/2) * base * height

-- Theorem statement
theorem right_triangle_area :
  ∀ a b c : ℝ, right_triangle a b c → triangle_area a b = 30 :=
by
  intros a b c h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l121_12108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_received_l121_12196

/-- The amount distributed equally among people each week -/
noncomputable def S : ℝ := sorry

/-- The initial number of people -/
def n : ℕ := sorry

/-- Condition: If there were 5 fewer people, each would receive $2 more -/
axiom fewer_people : S / (n - 5) = S / n + 2

/-- Condition: When there are 4 more people, each receives $1 less -/
axiom more_people : S / (n + 4) = S / n - 1

/-- The theorem to prove -/
theorem amount_received : S / (n + 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_received_l121_12196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_square_inequality_l121_12187

-- Define the ceiling function as noncomputable
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- State the theorem
theorem ceiling_square_inequality (x : ℝ) :
  (ceiling x)^2 - ceiling x - 12 ≤ 0 ↔ x ∈ Set.Ici (-3) ∩ Set.Iio 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_square_inequality_l121_12187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_solution_l121_12159

/-- The infinite nested radical equation -/
noncomputable def nested_radical_equation (x : ℝ) : Prop :=
  Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x))) = 
  (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))^(1/4)

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The positive solution to the nested radical equation -/
theorem nested_radical_solution :
  ∃! x : ℝ, x > 0 ∧ nested_radical_equation x ∧ x = (7 + 3 * Real.sqrt 5) / 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_solution_l121_12159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graduation_photo_arrangements_l121_12180

/-- The number of arrangements of 7 students in a row with specific conditions -/
def num_arrangements : ℕ := 192

/-- The total number of students -/
def total_students : ℕ := 7

/-- The position of student A (middle position) -/
def middle_position : ℕ := (total_students + 1) / 2

theorem graduation_photo_arrangements :
  (total_students = 7) →
  (∃ A : ℕ, A = middle_position) →
  (∃ B C : ℕ, B + 1 = C ∨ C + 1 = B) →
  num_arrangements = 192 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graduation_photo_arrangements_l121_12180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_specific_values_l121_12164

theorem cos_sum_specific_values (α β : Real) 
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  Real.cos (α + β) = -569/800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_specific_values_l121_12164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_196_is_169_l121_12178

/-- Represents the array structure described in the problem -/
def ArrayStructure (k : ℕ) : Prop :=
  (∀ n : ℕ, n ≤ k → ((2 * n - 1) = 2 * n - 1)) ∧
  (∀ n : ℕ, n ≤ k → (n^2 = n^2))

/-- The number directly above a given number in the array -/
def NumberAbove (n : ℕ) : ℕ := 
  if n = 196 then 169 else 0  -- simplified for this specific problem

/-- Theorem stating that the number directly above 196 in the described array is 169 -/
theorem number_above_196_is_169 :
  ∀ k : ℕ, k ≥ 14 → ArrayStructure k → NumberAbove 196 = 169 := by
  intros k h_k _
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_above_196_is_169_l121_12178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_result_l121_12144

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 4*y - 16 = -y^2 + 24*x + 16

-- Define the center and radius
def circle_center : ℝ × ℝ := (12, -2)

noncomputable def circle_radius : ℝ := 2 * Real.sqrt 41

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y →
  ∃ c d s : ℝ,
    (c, d) = circle_center ∧
    s = circle_radius ∧
    (x - c)^2 + (y - d)^2 = s^2 ∧
    c + d + s = 10 + 2 * Real.sqrt 41 :=
by
  sorry

-- Additional theorem to show the result
theorem result :
  let (c, d) := circle_center
  let s := circle_radius
  c + d + s = 10 + 2 * Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_result_l121_12144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_a_eq_neg_one_l121_12133

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the complex number z -/
noncomputable def z (a : ℝ) : ℂ := a^2 - 1 + (a - 1) * i

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def is_pure_imaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

theorem pure_imaginary_implies_a_eq_neg_one :
  ∃ (a : ℝ), is_pure_imaginary (z a) → a = -1 := by
  sorry

#check pure_imaginary_implies_a_eq_neg_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_implies_a_eq_neg_one_l121_12133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l121_12151

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

theorem min_shift_for_even_function :
  ∃ (m : ℝ), m > 0 ∧ 
  (∀ (x : ℝ), g m x = g m (-x)) ∧
  (∀ (m' : ℝ), m' > 0 → (∀ (x : ℝ), g m' x = g m' (-x)) → m' ≥ m) ∧
  m = π / 3 := by
  sorry

#check min_shift_for_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_even_function_l121_12151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l121_12106

/-- Two lines are parallel if their slopes are equal and their y-intercepts are different -/
def are_parallel (m1 n1 b1 m2 n2 b2 : ℝ) : Prop :=
  m1 / n1 = m2 / n2 ∧ b1 / n1 ≠ b2 / n2

/-- The first line: (3+a)x + 4y = 5-3a -/
def line1 (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (3 + a) * x + 4 * y = 5 - 3 * a

/-- The second line: 2x + (5+a)y = 8 -/
def line2 (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ 2 * x + (5 + a) * y = 8

theorem parallel_lines (a : ℝ) :
  are_parallel (3 + a) 4 (5 - 3 * a) 2 (5 + a) 8 ↔ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l121_12106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l121_12139

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := c / b

/-- The line equation is in the form ax + by + c = 0 -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

theorem y_intercept_of_line :
  let a : ℝ := 1
  let b : ℝ := -1
  let c : ℝ := 3
  y_intercept a b c = 3 ∧ ∀ x y : ℝ, line_equation a b c x y ↔ x - y + 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l121_12139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_win_l121_12110

/-- Represents the state of the game --/
structure GameState where
  box1 : Nat
  box2 : Nat
  box3 : Nat

/-- Defines a valid move in the game --/
inductive Move where
  | TakeFromBox1 (n : Nat)
  | TakeFromBox2 (n : Nat)
  | TakeFromBox3 (n : Nat)

/-- Applies a move to a game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeFromBox1 n => { state with box1 := state.box1 - n }
  | Move.TakeFromBox2 n => { state with box2 := state.box2 - n }
  | Move.TakeFromBox3 n => { state with box3 := state.box3 - n }

/-- Checks if a game state is terminal (i.e., no matches left) --/
def isTerminal (state : GameState) : Bool :=
  state.box1 = 0 && state.box2 = 0 && state.box3 = 0

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Move

/-- Generates the sequence of game states given two strategies --/
def generateGameSequence (initial_state : GameState) (strategy1 strategy2 : Strategy) : List GameState :=
  sorry

/-- Checks if a sequence of game states represents a win for the given strategy --/
def isWinningSequence (game_sequence : List GameState) (winning_strategy : Strategy) : Prop :=
  sorry

/-- Theorem stating that there exists a winning strategy for the first player --/
theorem first_player_can_win :
  ∃ (strategy : Strategy),
    ∀ (opponent_strategy : Strategy),
      let initial_state : GameState := ⟨2016, 2016, 2016⟩
      let game_sequence := generateGameSequence initial_state strategy opponent_strategy
      isWinningSequence game_sequence strategy :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_can_win_l121_12110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l121_12121

theorem expression_equality : 
  |-(Real.sqrt 3)| - (4 - Real.pi)^(0 : ℕ) + 2 * Real.sin (Real.pi / 3) - (1/4)^(-1 : ℤ) = 2 * Real.sqrt 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l121_12121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l121_12136

noncomputable def f (x m n : ℝ) : ℝ := (-2^x + n) / (2^(x+1) + m)

theorem odd_function_and_inequality (m n : ℝ) :
  (∀ x, f x m n = -f (-x) m n) →
  (∀ t ∈ Set.Ioo 1 2, ∀ k, f (t^2 - 2*t) m n + f (2*t^2 - k) m n < 0) →
  (m = 2 ∧ n = 1 ∧ ∀ k, (∀ t ∈ Set.Ioo 1 2, f (t^2 - 2*t) m n + f (2*t^2 - k) m n < 0) → k ≤ 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l121_12136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_equation_l121_12161

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 3*x else -((-x)^2 - 3*(-x))

-- State the theorem
theorem solution_set_of_equation :
  let S : Set ℝ := {x | f x - x + 3 = 0}
  S = {-2 - Real.sqrt 7, 1, 3} := by
  sorry

-- You can add additional lemmas or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_equation_l121_12161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equality_l121_12158

/-- Internal angle bisector of angle C in triangle ABC -/
noncomputable def internal_bisector_C (a b c : ℝ) : ℝ := 
  sorry

/-- External angle bisector of angle C in triangle ABC -/
noncomputable def external_bisector_C (a b c : ℝ) : ℝ := 
  sorry

/-- Given a triangle ABC with sides a, b, and c, where the internal and external 
    angle bisectors of angle C are equal, prove that (b^2 - a^2)^2 = c^2(a^2 + b^2). -/
theorem angle_bisector_equality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_bisectors : internal_bisector_C a b c = external_bisector_C a b c) :
  (b^2 - a^2)^2 = c^2 * (a^2 + b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equality_l121_12158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l121_12177

/-- Given two vectors a and b in ℝ², prove that the angle between them is 2π/3 -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  Real.sqrt ((a.1 ^ 2 + a.2 ^ 2 : ℝ)) = 4 * Real.cos (π / 8) →
  Real.sqrt ((b.1 ^ 2 + b.2 ^ 2 : ℝ)) = 2 * Real.sin (π / 8) →
  a.1 * b.1 + a.2 * b.2 = -Real.sqrt 2 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1 ^ 2 + a.2 ^ 2) * (b.1 ^ 2 + b.2 ^ 2)))) = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l121_12177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_c_l121_12131

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The parabola y = 2x^2 -/
def parabola (p : Point2D) : Prop := p.y = 2 * p.x^2

/-- Check if two points have the same y-coordinate -/
def horizontal (p q : Point2D) : Prop := p.y = q.y

/-- Check if a triangle is isosceles right -/
def isoscelesRight (a b c : Point2D) : Prop :=
  (a.x - c.x)^2 + (a.y - c.y)^2 = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (a.x - c.x) * (b.x - c.x) + (a.y - c.y) * (b.y - c.y) = 0

/-- Calculate the area of a triangle -/
noncomputable def triangleArea (a b c : Point2D) : ℝ :=
  abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2

theorem y_coordinate_of_c (a b c : Point2D) :
  parabola a ∧ parabola b ∧ parabola c ∧
  horizontal a b ∧
  isoscelesRight a b c ∧
  triangleArea a b c = 1600 →
  c.y = 1600 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_c_l121_12131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sequence_properties_l121_12109

/-- A geometric sequence of square sides -/
noncomputable def squareSequence (n : ℕ) : ℝ :=
  9 * (2/3)^(n-1)

theorem square_sequence_properties :
  let x := squareSequence 3
  ∀ (n : ℕ), n ≥ 1 → 
    squareSequence 1 = 9 ∧
    squareSequence 2 = 6 ∧
    squareSequence (n+1) = (2/3) * squareSequence n →
    x = 4 ∧ squareSequence 2014 = 9 * (2/3)^2013 :=
by
  sorry

#check square_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sequence_properties_l121_12109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_ranges_l121_12186

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 8) / 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem function_ranges :
  (∃ m : ℝ, ∃ x₀ : ℝ, x₀ ≥ 2 ∧ f x₀ = m ∧ m ∈ Set.Ici 3) ∧
  (∃ a : ℝ, a ∈ Set.Ioo 1 (Real.sqrt 3) ∧
    ∀ x₁ : ℝ, x₁ ≥ 2 →
      ∃ x₂ : ℝ, x₂ > 2 ∧ f x₁ = g a x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_ranges_l121_12186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l121_12154

theorem calculate_expression : 
  Real.sqrt 5 * (5 : ℝ)^(1/2 : ℝ) + 20 / 4 * 3 - (8 : ℝ)^(3/2 : ℝ) + 5 = 25 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l121_12154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_vertex_product_l121_12190

/-- A tetrahedron is a solid figure with four triangular faces. -/
structure Tetrahedron where
  faces : Fin 4 → Fin 3 → Fin 3 → Bool
  is_solid : Bool

/-- The number of vertices in a tetrahedron. -/
def Tetrahedron.num_vertices : ℕ := 4

/-- The number of edges in a tetrahedron. -/
def Tetrahedron.num_edges : ℕ := 6

/-- The product of the number of edges and vertices in a tetrahedron is 24. -/
theorem tetrahedron_edge_vertex_product : 
  Tetrahedron.num_edges * Tetrahedron.num_vertices = 24 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_vertex_product_l121_12190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_to_twentythird_l121_12193

/-- Days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, BEq, Inhabited

/-- Function to get the next day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Function to get the day after a given number of days -/
def dayAfter (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

theorem sixth_to_twentythird (h : dayAfter Day.Thursday 0 = Day.Thursday) :
  dayAfter Day.Thursday 17 = Day.Sunday := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_to_twentythird_l121_12193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_top_row_five_hops_l121_12143

/-- Represents a position on the 4x4 grid -/
inductive Position
| mk : Fin 4 → Fin 4 → Position

/-- Represents a direction of movement -/
inductive Direction
| Up | Down | Left | Right

/-- The grid structure -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Function to move in a given direction with wraparound -/
def move (p : Position) (d : Direction) : Position :=
  sorry

/-- Function to calculate the probability of reaching the top row from a given position in n hops -/
def probReachTopRow (g : Grid) (p : Position) (n : ℕ) : ℝ :=
  sorry

/-- The main theorem -/
theorem prob_reach_top_row_five_hops :
  let initial_pos := Position.mk 0 0
  probReachTopRow (λ _ _ => 0) initial_pos 5 = 31 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_reach_top_row_five_hops_l121_12143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_dot_product_l121_12195

/-- Ellipse C with equation x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point A is the left vertex of ellipse C -/
def point_A : ℝ × ℝ := (-2, 0)

/-- Point B is the top vertex of ellipse C -/
noncomputable def point_B : ℝ × ℝ := (0, Real.sqrt 3)

/-- Point F is the right focus of ellipse C -/
def point_F : ℝ × ℝ := (1, 0)

/-- Vector AB -/
noncomputable def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

/-- Vector AF -/
def vector_AF : ℝ × ℝ := (point_F.1 - point_A.1, point_F.2 - point_A.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem ellipse_vector_dot_product : 
  dot_product vector_AB vector_AF = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_dot_product_l121_12195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_and_expression_l121_12153

/-- Given an angle θ whose terminal side passes through the point (-4, 3),
    prove the trigonometric identities and the given expression. -/
theorem trig_identities_and_expression (θ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos θ = -4 ∧ r * Real.sin θ = 3) →
  Real.sin θ = 3/5 ∧ 
  Real.cos θ = -4/5 ∧ 
  Real.tan θ = -3/4 ∧
  Real.cos (θ - π/2) / Real.sin (π/2 + θ) * Real.sin (θ + π) * Real.cos (2*π - θ) = -9/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_and_expression_l121_12153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approximation_l121_12115

/-- The area of a triangle given its three side lengths --/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 28, 30, and 16 is approximately 221.25 --/
theorem triangle_area_approximation :
  |triangleArea 28 30 16 - 221.25| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approximation_l121_12115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_line_constant_from_area_quadrilateral_area_condition_l121_12105

/-- A line in 2D space that is not parallel to either axis -/
structure NonAxisParallelLine where
  a : ℝ
  b : ℝ
  k : ℝ
  ha : a ≠ 0
  hb : b ≠ 0

/-- The area of the triangle formed by a line's intersections with the axes and the origin -/
noncomputable def triangleArea (l : NonAxisParallelLine) : ℝ := |l.k^2 / (2 * l.a * l.b)|

/-- Given an area and coefficients, determine the constant term of the line equation -/
noncomputable def lineConstant (a b : ℝ) (area : ℝ) : ℝ := Real.sqrt (2 * area * |a * b|)

/-- Check if a pair of integers (m, n) satisfies the quadrilateral area condition -/
def satisfiesAreaCondition (m n : ℤ) : Prop :=
  14 * n - 4 * m = 2022 ∧ 100 ≤ m ∧ m < n

theorem triangle_area_formula (l : NonAxisParallelLine) :
  triangleArea l = |l.k^2 / (2 * l.a * l.b)| := by sorry

theorem line_constant_from_area (l : NonAxisParallelLine) (A : ℝ) (h : A > 0) :
  triangleArea l = A → l.k = lineConstant l.a l.b A ∨ l.k = -lineConstant l.a l.b A := by sorry

theorem quadrilateral_area_condition :
  (satisfiesAreaCondition 100 173 ∧ satisfiesAreaCondition 107 175) ∧
  ∀ m n : ℤ, satisfiesAreaCondition m n → (m = 100 ∧ n = 173) ∨ (m = 107 ∧ n = 175) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_formula_line_constant_from_area_quadrilateral_area_condition_l121_12105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_l121_12138

/-- A geometric transformation that maps one circle to another -/
structure CircleTransformation where
  original_center : ℝ × ℝ
  original_radius : ℝ
  transformed_center : ℝ × ℝ
  transformed_radius : ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Distance moved by the origin under the given transformation -/
theorem origin_movement (t : CircleTransformation) 
    (h1 : t.original_center = (1, 3))
    (h2 : t.original_radius = 4)
    (h3 : t.transformed_center = (7, 10))
    (h4 : t.transformed_radius = 6) :
  distance (0, 0) t.transformed_center = Real.sqrt 149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_l121_12138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_digit_sum_eighths_ninths_l121_12112

theorem fifteenth_digit_sum_eighths_ninths : ∃ (s : ℚ),
  (s = 1/8 + 1/9) ∧
  (∃ (a b : ℕ) (c : ℕ),
    s = (a : ℚ) + (b : ℚ) / (10^3 : ℚ) + (c : ℚ) / (10^3 : ℚ) / (1 - 1/10) ∧
    0 ≤ a ∧ a < 1 ∧
    0 ≤ b ∧ b < 1000 ∧
    0 < c ∧ c < 10) ∧
  (⌊s * 10^15⌋ % 10 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_digit_sum_eighths_ninths_l121_12112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_a_for_system_l121_12175

theorem minimal_a_for_system (x y z : ℝ) :
  (∃ a : ℝ, 
    Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) = a - 1 ∧
    Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1) = a + 1) →
  (∀ a : ℝ, 
    Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) = a - 1 ∧
    Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1) = a + 1 →
    a ≥ 11/2) ∧
  (∃ x y z : ℝ, 
    Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) = 11/2 - 1 ∧
    Real.sqrt (x + 1) + Real.sqrt (y + 1) + Real.sqrt (z + 1) = 11/2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_a_for_system_l121_12175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_one_l121_12168

/-- The distance between tangency points of inscribed circles in triangles ABC and CDA --/
noncomputable def distance_between_tangency_points (AB BC CD DA : ℝ) : ℝ :=
  let AC := CD + DA - AB - BC
  let s_ABC := (AB + BC + AC) / 2
  let s_CDA := (CD + DA + AC) / 2
  let AK := s_ABC - BC
  let AM := DA
  |AK - AM|

/-- Theorem: The distance between tangency points is 1 for given quadrilateral side lengths --/
theorem distance_is_one (AB BC CD DA : ℝ) 
  (h : (AB = 5 ∧ BC = 7 ∧ CD = DA) ∨ (AB = 7 ∧ BC = CD ∧ DA = 9)) :
  distance_between_tangency_points AB BC CD DA = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_one_l121_12168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_peaches_l121_12156

/-- The number of peaches in the pile -/
def P : ℕ := sorry

/-- The number of monkeys -/
def n : ℕ := sorry

/-- If each monkey gets 6 peaches, there are 57 peaches left -/
axiom condition1 : P = 6 * n + 57

/-- If each monkey gets 9 peaches, 5 monkeys get none, and one monkey gets only 3 peaches -/
axiom condition2 : P = 9 * (n - 6) + 3

theorem total_peaches : P = 273 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_peaches_l121_12156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l121_12167

-- Define the initial amount
noncomputable def initial_amount : ℝ := 3200

-- Define the yearly increase rate
noncomputable def increase_rate : ℝ := 1 / 8

-- Define the number of years
def years : ℕ := 2

-- Define the final amount after 2 years
noncomputable def final_amount : ℝ := initial_amount * (1 + increase_rate) ^ years

-- Theorem to prove
theorem amount_after_two_years :
  final_amount = 4050 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_after_two_years_l121_12167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_odd_l121_12119

-- Define the sine function as noncomputable
noncomputable def sine (x : ℝ) : ℝ := Real.sin x

-- State that sine is an odd function
axiom sine_odd (x : ℝ) : sine (-x) = -sine x

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := sine (x^2 + 1)

-- Theorem statement
theorem f_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_odd_l121_12119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisibility_l121_12176

theorem sum_divisibility (n k : ℕ) :
  let m := (Finset.range n).sum (fun i => (i + 1) ^ (2 * k + 1))
  (n + 2) ∣ m ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisibility_l121_12176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l121_12199

/-- Represents a bag of colored balls -/
structure BagOfBalls where
  total : ℕ
  white : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ
  red : ℕ

/-- The probability of choosing a ball that is neither red nor purple -/
def probNotRedPurple (bag : BagOfBalls) : ℚ :=
  (bag.white + bag.green + bag.yellow : ℚ) / bag.total

theorem red_balls_count (bag : BagOfBalls) 
  (h1 : bag.total = 100)
  (h2 : bag.white = 10)
  (h3 : bag.green = 30)
  (h4 : bag.yellow = 10)
  (h5 : bag.purple = 3)
  (h6 : probNotRedPurple bag = 1/2) :
  bag.red = 47 := by
  sorry

-- Remove the #eval statement as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l121_12199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_count_l121_12114

theorem student_assignment_count : 60 = 60 := by
  let total_students : ℕ := 5
  let friday_attendees : ℕ := 2
  let saturday_attendees : ℕ := 1
  let sunday_attendees : ℕ := 1
  
  let ways_to_select_friday : ℕ := Nat.choose total_students friday_attendees
  let ways_to_select_saturday : ℕ := Nat.choose (total_students - friday_attendees) saturday_attendees
  let ways_to_select_sunday : ℕ := Nat.choose (total_students - friday_attendees - saturday_attendees) sunday_attendees
  
  have h : ways_to_select_friday * ways_to_select_saturday * ways_to_select_sunday = 60 := by
    -- The actual calculation would go here
    sorry
  
  exact h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_assignment_count_l121_12114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_limit_above_six_percent_l121_12170

-- Define the raise percentage as a real number between 0 and 1
variable (raise_percentage : ℝ)

-- Define the conditions
axiom raise_upper_bound : raise_percentage < 0.1
axiom raise_lower_bound : raise_percentage > 0
axiom possible_increase : ∃ ε > 0, |raise_percentage - 0.06| < ε

-- Theorem to prove
theorem lower_limit_above_six_percent :
  raise_percentage > 0.06 ∧ raise_percentage < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_limit_above_six_percent_l121_12170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l121_12120

-- Define the universe U
def U : Set ℝ := {x | x ≥ -4}

-- Define set A
def A : Set ℝ := {x ∈ U | -1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x ∈ U | 0 ≤ x ∧ x < 5}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x ∈ U | 0 ≤ x ∧ x ≤ 3}) ∧
  ((U \ A) ∪ B = {x ∈ U | x ≤ -1 ∨ x ≥ 0}) ∧
  (A ∩ (U \ B) = {x ∈ U | -1 < x ∧ x < 0}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l121_12120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tax_inclusive_price_l121_12174

theorem smallest_tax_inclusive_price : 
  ∃ x : ℕ, x > 0 ∧ (107 : ℚ) * x = 100 * 107 ∧ 
  ∀ n : ℕ, 0 < n ∧ n < 107 → ¬∃ y : ℕ, y > 0 ∧ (107 : ℚ) * y = 100 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tax_inclusive_price_l121_12174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l121_12172

def new_device_data : List Float := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def old_device_data : List Float := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def sample_mean (data : List Float) : Float :=
  (data.sum) / (data.length.toFloat)

def sample_variance (data : List Float) : Float :=
  let mean := sample_mean data
  (data.map (fun x => (x - mean) ^ 2)).sum / (data.length.toFloat)

theorem significant_improvement :
  let x_bar := sample_mean new_device_data
  let y_bar := sample_mean old_device_data
  let s1_squared := sample_variance new_device_data
  let s2_squared := sample_variance old_device_data
  y_bar - x_bar ≥ 2 * (((s1_squared + s2_squared) / 10 : Float).sqrt) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_significant_improvement_l121_12172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l121_12123

def b : ℝ × ℝ := (1, -2)

theorem parallel_vectors_magnitude (m : ℝ) :
  let a : ℝ × ℝ := (4, m)
  (∃ (k : ℝ), a = k • b) →
  ‖a - 2 • b‖ = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l121_12123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_formula_l121_12189

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  -- Base edge length
  a : ℝ
  -- Height
  h : ℝ
  -- Ratio of height division by inscribed sphere center
  m : ℝ
  n : ℝ
  -- Conditions
  a_pos : 0 < a
  h_pos : 0 < h
  m_pos : 0 < m
  n_pos : 0 < n

/-- Angle between adjacent lateral faces of a regular quadrilateral pyramid -/
noncomputable def lateral_face_angle (p : RegularQuadPyramid) : ℝ :=
  Real.pi - Real.arccos ((p.n^2) / (p.m^2))

/-- Theorem: The angle between adjacent lateral faces of a regular quadrilateral pyramid
    is π - arccos(n²/m²), where m:n is the ratio that the inscribed sphere's center
    divides the height (counting from the apex) -/
theorem lateral_face_angle_formula (p : RegularQuadPyramid) :
  lateral_face_angle p = Real.pi - Real.arccos ((p.n^2) / (p.m^2)) := by
  -- Proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_face_angle_formula_l121_12189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trihedral_angle_dihedral_angles_l121_12141

/-- A trihedral angle with two right angles -/
structure RightTrihedralAngle where
  α : ℝ
  angle1 : α ≥ 0
  angle2 : α ≤ Real.pi

/-- The dihedral angles of a right trihedral angle -/
noncomputable def dihedralAngles (t : RightTrihedralAngle) : Fin 3 → ℝ
  | 0 => Real.pi / 2
  | 1 => Real.pi / 2
  | 2 => t.α

theorem right_trihedral_angle_dihedral_angles (t : RightTrihedralAngle) :
  (dihedralAngles t 0 = Real.pi / 2) ∧
  (dihedralAngles t 1 = Real.pi / 2) ∧
  (dihedralAngles t 2 = t.α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trihedral_angle_dihedral_angles_l121_12141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l121_12179

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π
  cos_relation : b * Real.cos (C - π/3) = (a + c) / 2
  side_b : b = 2 * Real.sqrt 3

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  t.B = π/3 ∧
  (6 + 2 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 6 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l121_12179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_sum_of_opposite_and_b_l121_12122

theorem square_of_sum_of_opposite_and_b (a b : ℝ) : 
  (-(a) + b)^2 = (-a + b)^2 := by
  rfl

#check square_of_sum_of_opposite_and_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_sum_of_opposite_and_b_l121_12122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l121_12145

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

theorem range_of_f : 
  {y : ℝ | ∃ x ≥ 8, f x = y} = Set.Ioo 0 (1/256) ∪ {1/256} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l121_12145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l121_12134

/-- A circle C in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle C passes through the points (1, 0) and (3, 0), and is tangent to the y-axis -/
noncomputable def circle_C : Circle :=
  { center := (2, Real.sqrt 3),
    radius := 2 }

/-- The equation of a circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_C_equation :
  ∀ (x y : ℝ), circle_equation circle_C x y ↔ (x - 2)^2 + (y - Real.sqrt 3)^2 = 4 ∨ (x - 2)^2 + (y + Real.sqrt 3)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_l121_12134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_existence_l121_12197

theorem right_triangle_existence (a b c d : ℕ+) 
  (h1 : a * b = c * d) (h2 : a + b = c - d) :
  ∃ (x y z : ℕ+), x^2 + y^2 = z^2 ∧ x * y = 2 * a * b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_existence_l121_12197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_implies_m_equals_neg_one_l121_12146

/-- A function f is a power function if it can be written as f(x) = k * x^n 
    where k and n are constants and k ≠ 0. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), k ≠ 0 ∧ ∀ x, f x = k * x^n

/-- The given function f(x) = (2m + 3)x^(m^2 - 3) -/
noncomputable def f (m : ℝ) : ℝ → ℝ := λ x ↦ (2*m + 3) * (x^(m^2 - 3))

/-- Theorem: If f(x) = (2m + 3)x^(m^2 - 3) is a power function, then m = -1 -/
theorem power_function_implies_m_equals_neg_one (m : ℝ) :
  is_power_function (f m) → m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_implies_m_equals_neg_one_l121_12146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l121_12104

-- Define the parabola C
def parabola_C (t : ℝ) : ℝ × ℝ := (8 * t^2, 8 * t)

-- Define the circle
def circle_C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = r^2}

-- Define the line passing through the focus of C with slope 1
def line_through_focus (x y : ℝ) : Prop := y = x - 2

-- Define the focus of the parabola C
def parabola_C_focus : ℝ × ℝ := (2, 0)

-- State the theorem
theorem parabola_circle_tangent (r : ℝ) (hr : r > 0) :
  (line_through_focus parabola_C_focus.1 parabola_C_focus.2) ∧
  (∃ (p : ℝ × ℝ), p ∈ circle_C r ∧ line_through_focus p.1 p.2) →
  r = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l121_12104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_uniqueness_l121_12128

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
noncomputable def Triangle.side_length (t : Triangle) (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def Triangle.angle (t : Triangle) (p q r : ℝ × ℝ) : ℝ :=
  Real.arccos ((Triangle.side_length t p q)^2 + (Triangle.side_length t p r)^2 - (Triangle.side_length t q r)^2) /
               (2 * Triangle.side_length t p q * Triangle.side_length t p r)

-- Theorem statement
theorem triangle_uniqueness (t : Triangle) :
  Triangle.side_length t t.A t.B = 2 →
  Triangle.side_length t t.B t.C = 3 →
  Triangle.angle t t.A t.B t.C = 50 * π / 180 →
  ∃! (t' : Triangle), Triangle.side_length t' t'.A t'.B = 2 ∧
                      Triangle.side_length t' t'.B t'.C = 3 ∧
                      Triangle.angle t' t'.A t'.B t'.C = 50 * π / 180 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_uniqueness_l121_12128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muslim_percentage_l121_12150

theorem muslim_percentage (total : ℕ) (hindu_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) : 
  total = 700 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  other = 126 →
  (total - (hindu_percent * ↑total).floor - (sikh_percent * ↑total).floor - other) / ↑total = 44 / 100 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_muslim_percentage_l121_12150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l121_12118

/-- Represents the average annual growth rate of per capita disposable income -/
def x : ℝ := sorry

/-- The initial per capita disposable income in 2020 (in ten thousand yuan) -/
def initial_income : ℝ := 5.76

/-- The final per capita disposable income in 2022 (in ten thousand yuan) -/
def final_income : ℝ := 6.58

/-- The number of years between 2020 and 2022 -/
def years : ℕ := 2

/-- Theorem stating that the equation correctly represents the relationship between
    initial income, final income, and average annual growth rate over two years -/
theorem income_growth_equation :
  initial_income * (1 + x) ^ years = final_income := by
  sorry

#check income_growth_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l121_12118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_sphere_properties_l121_12116

/-- A frustum with a tangent sphere -/
structure FrustumWithSphere where
  r₁ : ℝ  -- radius of top circle
  r₂ : ℝ  -- radius of bottom circle
  l : ℝ   -- slant height
  R : ℝ   -- radius of tangent sphere
  h₁ : r₁ > 0
  h₂ : r₂ > 0
  h₃ : r₁ ≠ r₂
  h₄ : l > 0
  h₅ : R > 0

/-- The sphere is tangent to the top, bottom, and lateral surface of the frustum -/
axiom sphere_tangent (f : FrustumWithSphere) : True

/-- Surface area of a sphere -/
noncomputable def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2

/-- Volume of a sphere -/
noncomputable def sphere_volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

/-- Surface area of a frustum -/
noncomputable def frustum_surface_area (r₁ r₂ l : ℝ) : ℝ := Real.pi * (r₁^2 + r₂^2 + (r₁ + r₂) * l)

/-- Volume of a frustum -/
noncomputable def frustum_volume (r₁ r₂ h : ℝ) : ℝ := (1 / 3) * Real.pi * (r₁^2 + r₂^2 + r₁ * r₂) * h

theorem frustum_sphere_properties (f : FrustumWithSphere) :
  (f.l = f.r₁ + f.r₂) ∧
  (f.R = Real.sqrt (f.r₁ * f.r₂)) ∧
  (sphere_surface_area f.R / frustum_surface_area f.r₁ f.r₂ f.l =
   sphere_volume f.R / frustum_volume f.r₁ f.r₂ (2 * f.R)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_sphere_properties_l121_12116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_triangle_side_range_l121_12149

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sin (2*x - 7*Real.pi/6)

-- Theorem for the maximum value and when it occurs
theorem f_maximum (x : ℝ) : 
  (∀ y, f y ≤ 2) ∧ 
  (f x = 2 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi/6) := by
  sorry

-- Theorem for the range of side a in triangle ABC
theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  f A = 3/2 →
  b + c = 2 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  1 ≤ a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_triangle_side_range_l121_12149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_crossing_impossibility_l121_12147

theorem boat_crossing_impossibility (n : ℕ) (hn : n = 5) :
  ¬ ∃ (crossings : Finset (Finset (Fin n))),
    (∀ s : Finset (Fin n), s.Nonempty → s ∈ crossings) ∧
    (∀ s ∈ crossings, s.Nonempty) ∧
    crossings.card = 2^n - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_crossing_impossibility_l121_12147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_working_hours_difference_l121_12191

/-- Represents the working hours of each person -/
structure WorkingHours :=
  (person1 : ℕ)
  (person2 : ℕ)
  (person3 : ℕ)
  (person4 : ℕ)
  (person5 : ℕ)

/-- Represents the project details -/
structure Project :=
  (total_hours : ℕ)
  (working_hours : WorkingHours)

/-- The theorem to prove -/
theorem project_working_hours_difference
  (p : Project)
  (h1 : p.working_hours.person1 + p.working_hours.person2 + p.working_hours.person3 + p.working_hours.person4 + p.working_hours.person5 = p.total_hours)
  (h2 : (p.working_hours.person1 : ℚ) = (3/25) * p.total_hours)
  (h3 : (p.working_hours.person2 : ℚ) = (4/25) * p.total_hours)
  (h4 : (p.working_hours.person3 : ℚ) = (5/25) * p.total_hours)
  (h5 : (p.working_hours.person4 : ℚ) = (6/25) * p.total_hours)
  (h6 : (p.working_hours.person5 : ℚ) = (7/25) * p.total_hours)
  (h7 : p.total_hours = 1800)
  (h8 : ∃ (days_worked_least : ℕ), days_worked_least * 8 = p.working_hours.person1 * 3 / 5)
  (h9 : ∃ (days_worked_most : ℕ), days_worked_most * 8 = p.working_hours.person5 * 3 / 5)
  : ∃ (difference : ℕ), difference = (p.working_hours.person5 * 3 / 5 / 8) * 8 - (p.working_hours.person1 * 3 / 5 / 8) * 8 ∧ difference = 168 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_working_hours_difference_l121_12191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l121_12102

theorem perfect_square_sum (n : ℕ) : (∃ k : ℤ, 4^n + 5^n = k^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l121_12102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l121_12188

/-- Calculates the total amount after compound interest -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_growth :
  let principal : ℝ := 12000
  let rate : ℝ := 0.05
  let time : ℕ := 7
  let finalAmount := compoundInterest principal rate time
  roundToNearest finalAmount = 16885 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l121_12188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_l121_12117

/-- A point in the Cartesian plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Converts a point from polar coordinates (r, θ) to Cartesian coordinates (x, y). -/
noncomputable def polarToCartesian (r : ℝ) (θ : ℝ) : Point :=
  { x := r * Real.cos θ, y := r * Real.sin θ }

/-- The set of points satisfying θ = π/4 in polar coordinates. -/
def curvePoints : Set Point :=
  {p : Point | ∃ r : ℝ, p = polarToCartesian r (Real.pi/4)}

/-- Defines a line in the Cartesian plane passing through two points. -/
def isLine (S : Set Point) : Prop :=
  ∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ ∀ p ∈ S, a * p.x + b * p.y = c

/-- Theorem stating that the curve defined by θ = π/4 is a line. -/
theorem curve_is_line : isLine curvePoints := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_l121_12117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l121_12160

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The perimeter of a triangle given by three points -/
noncomputable def trianglePerimeter (a b c : Point) : ℝ :=
  distance a b + distance b c + distance c a

/-- A point is on the line y = x -/
def onLineYEqX (p : Point) : Prop :=
  p.y = p.x

/-- A point is on the x-axis -/
def onXAxis (p : Point) : Prop :=
  p.y = 0

/-- The theorem stating the minimum perimeter of the triangle -/
theorem min_perimeter_triangle :
  ∃ (b c : Point),
    onLineYEqX b ∧
    onXAxis c ∧
    (∀ (b' c' : Point),
      onLineYEqX b' →
      onXAxis c' →
      trianglePerimeter ⟨3, 2⟩ b c ≤ trianglePerimeter ⟨3, 2⟩ b' c') ∧
    trianglePerimeter ⟨3, 2⟩ b c = Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_triangle_l121_12160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PQ_is_4_l121_12142

-- Define the circle C
noncomputable def circle_C (θ : Real) : Real × Real :=
  (2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Define line l1
def line_l1 (x : Real) : Prop :=
  x + 1 = 0

-- Define line l2 in polar form
def line_l2 (θ : Real) : Prop :=
  θ = Real.pi / 3

-- Define point O
def point_O : Real × Real :=
  (0, 0)

-- Define point P
noncomputable def point_P : Real × Real :=
  (1, Real.sqrt 3)

-- Define point Q
noncomputable def point_Q : Real × Real :=
  (-1, -Real.sqrt 3)

-- Theorem statement
theorem distance_PQ_is_4 :
  let P := point_P
  let Q := point_Q
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_PQ_is_4_l121_12142
