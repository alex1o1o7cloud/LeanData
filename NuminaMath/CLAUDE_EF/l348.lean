import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_pure_imaginary_implies_a_eq_one_l348_34806

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z (a : ℝ) : ℂ := (a + i) / (1 - i)

/-- A complex number is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

/-- Main theorem: If z is pure imaginary, then a = 1 -/
theorem z_pure_imaginary_implies_a_eq_one (a : ℝ) :
  is_pure_imaginary (z a) → a = 1 := by
  sorry

#check z_pure_imaginary_implies_a_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_pure_imaginary_implies_a_eq_one_l348_34806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_four_l348_34886

/-- Represents a monic polynomial of degree 3 with no z^2 term -/
structure MonicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : c > 0

/-- The product of two MonicPolynomials -/
noncomputable def product (p q : MonicPolynomial) : Polynomial ℝ :=
  Polynomial.X^6 + 2 * Polynomial.X^5 + 4 * Polynomial.X^3 + 9 * Polynomial.X + 16

theorem constant_term_is_four (p q : MonicPolynomial) 
  (h_product : product p q = Polynomial.X^6 + 2 * Polynomial.X^5 + 4 * Polynomial.X^3 + 9 * Polynomial.X + 16) :
  p.c = 4 ∧ q.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_four_l348_34886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_13th_innings_l348_34839

-- Define the initial number of innings
def initial_innings : ℕ := 12

-- Define the score in the 13th innings
def new_score : ℕ := 96

-- Define the increase in average
def average_increase : ℕ := 5

-- Theorem statement
theorem new_average_after_13th_innings :
  ∀ (initial_average : ℚ),
  (initial_average * initial_innings + new_score) / (initial_innings + 1 : ℚ) = initial_average + average_increase →
  (initial_average * initial_innings + new_score) / (initial_innings + 1 : ℚ) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_13th_innings_l348_34839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l348_34816

/-- The circle C with equation x^2 + y^2 - 6x - 8y + 17 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 8*y + 17 = 0

/-- The point of tangency (1, 2) -/
def tangent_point : ℝ × ℝ := (1, 2)

/-- The radius of the new circle -/
noncomputable def new_circle_radius : ℝ := 5 * Real.sqrt 2 / 2

/-- The equation of the new circle -/
def new_circle (x y : ℝ) : Prop :=
  (x - 7/2)^2 + (y - 9/2)^2 = (5 * Real.sqrt 2 / 2)^2

theorem circle_tangency :
  (∀ x y, new_circle x y → (x, y) = tangent_point ∨ ¬circle_C x y) ∧
  new_circle (tangent_point.1) (tangent_point.2) ∧
  circle_C (tangent_point.1) (tangent_point.2) ∧
  (∀ x y, new_circle x y → (x - 7/2)^2 + (y - 9/2)^2 = new_circle_radius^2) :=
by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l348_34816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_l348_34894

/-- The number of radars --/
noncomputable def n : ℝ := 8

/-- The radius of each radar's coverage area in km --/
noncomputable def r : ℝ := 15

/-- The width of the coverage ring in km --/
noncomputable def w : ℝ := 18

/-- The central angle between two adjacent radars in radians --/
noncomputable def θ : ℝ := 2 * Real.pi / n

theorem radar_placement_and_coverage :
  let d := r * Real.cos (θ / 2) / Real.sin (θ / 2)
  let inner_radius := d - w / 2
  let outer_radius := d + w / 2
  (d = 12 / Real.sin (22.5 * Real.pi / 180)) ∧ 
  (Real.pi * (outer_radius^2 - inner_radius^2) = 432 * Real.pi / Real.tan (22.5 * Real.pi / 180)) := by
  sorry

#check radar_placement_and_coverage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_placement_and_coverage_l348_34894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l348_34872

theorem remainder_theorem (x : ℕ) (h : x % 17 = 7) :
  (5 * x^2 + 3 * x + 2) % 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l348_34872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l348_34869

/-- Two linear functions with parallel non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- Condition for tangency of two functions -/
def is_tangent (f g : ℝ → ℝ) : Prop :=
  ∃! x, f x = g x

/-- Main theorem -/
theorem tangent_condition (funcs : ParallelLinearFunctions) :
  is_tangent (λ x ↦ (funcs.f x)^2) (λ x ↦ -8 * funcs.g x) →
  (∀ A : ℝ, is_tangent (λ x ↦ (funcs.g x)^2) (λ x ↦ A * funcs.f x) ↔ A = 0 ∨ A = 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l348_34869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_f_values_l348_34827

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x / (x - 1)

theorem compare_f_values (m a b : ℝ) (h1 : a > b) (h2 : b > 1) :
  (m > 0 → f m a < f m b) ∧
  (m = 0 → f m a = f m b) ∧
  (m < 0 → f m a > f m b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compare_f_values_l348_34827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l348_34823

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := c * x / (2 * x + 3)

-- State the theorem
theorem function_composition_identity (c : ℝ) :
  (∀ x : ℝ, x ≠ -3/2 → f c (f c x) = x) → c = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l348_34823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_count_l348_34890

/-- Represents the number of students in a group -/
def total_students : ℕ := 8

/-- Represents the number of boys in the group -/
def num_boys : ℕ := sorry

/-- Represents the number of girls in the group -/
def num_girls : ℕ := sorry

/-- The total number of students is the sum of boys and girls -/
axiom total_sum : num_boys + num_girls = total_students

/-- There are more girls than boys -/
axiom more_girls : num_girls > num_boys

/-- Theorem: The number of boys in the group is either 1, 2, or 3 -/
theorem boys_count : num_boys = 1 ∨ num_boys = 2 ∨ num_boys = 3 := by
  sorry

#check boys_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_count_l348_34890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_min_value_g_l348_34848

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x + 2|
noncomputable def g (x : ℝ) : ℝ := |5/6 * x - 1| + |1/2 * x - 1| + |2/3 * x - 1|

-- Theorem for the minimum value of f
theorem min_value_f : ∀ x : ℝ, f x ≥ 6 ∧ ∃ y : ℝ, f y = 6 := by
  sorry

-- Theorem for the minimum value of g
theorem min_value_g : ∀ x : ℝ, g x ≥ 1/2 ∧ ∃ y : ℝ, g y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_min_value_g_l348_34848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l348_34832

-- Define the plane
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the point O and the quadrilateral ABCD
variable (O A B C D : V)

-- Define the vector equation
axiom vector_equation : (A - O) + (C - O) = (B - O) + (D - O)

-- Define what it means for a quadrilateral to be a parallelogram
def is_parallelogram (A B C D : V) : Prop :=
  (B - A = D - C) ∧ (C - B = A - D)

-- Theorem statement
theorem quadrilateral_is_parallelogram :
  is_parallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_l348_34832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_average_rainfall_l348_34875

/-- Represents the rainfall during a storm -/
structure StormRainfall where
  first_30min : ℚ
  next_30min : ℚ
  last_60min : ℚ

/-- Calculates the average rainfall per hour for a given storm -/
def average_rainfall (storm : StormRainfall) : ℚ :=
  (storm.first_30min + storm.next_30min + storm.last_60min) / 2

/-- Theorem stating that the average rainfall for the given storm conditions is 4 inches per hour -/
theorem storm_average_rainfall :
  let storm : StormRainfall := {
    first_30min := 5,
    next_30min := 5 / 2,
    last_60min := 1 / 2
  }
  average_rainfall storm = 4 := by
  -- Proof goes here
  sorry

#eval average_rainfall { first_30min := 5, next_30min := 5 / 2, last_60min := 1 / 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_average_rainfall_l348_34875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_exists_l348_34836

/-- Represents a point in a 2D Euclidean plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction in the plane -/
inductive Direction
  | North
  | Other

/-- Represents a path in the plane -/
structure SmurfPath where
  points : List Point
  directions : List Direction

/-- A function to check if a path is valid according to the rules -/
def isValidPath (p : SmurfPath) : Prop :=
  p.points.length > 1 ∧
  p.directions.length = p.points.length - 1 ∧
  p.directions.head? = some Direction.North ∧
  (∀ i, i % 2 = 1 → i < p.directions.length → p.directions.get? i = some Direction.North) ∧
  p.points.head? = p.points.getLast? ∧
  (∀ i j, i ≠ j → i < p.points.length → j < p.points.length →
    (p.points.get? i ≠ p.points.get? j ∨ i = 0 ∧ j = p.points.length - 1))

/-- The main theorem stating that no valid path exists -/
theorem no_valid_path_exists : ¬∃ p : SmurfPath, isValidPath p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_exists_l348_34836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_bank_account_balance_l348_34850

/-- Proves that given Robert's spending conditions, the amount in his second bank account is $232.50 --/
theorem roberts_bank_account_balance 
  (raw_materials_cost : ℝ) 
  (machinery_cost : ℝ)
  (raw_materials_tax_rate : ℝ)
  (machinery_tax_rate : ℝ)
  (machinery_discount_rate : ℝ)
  (savings_usage_rate : ℝ)
  (first_account_balance : ℝ)
  (total_spending_rate : ℝ)
  (h1 : raw_materials_cost = 100)
  (h2 : machinery_cost = 125)
  (h3 : raw_materials_tax_rate = 0.05)
  (h4 : machinery_tax_rate = 0.08)
  (h5 : machinery_discount_rate = 0.10)
  (h6 : savings_usage_rate = 0.10)
  (h7 : first_account_balance = 900)
  (h8 : total_spending_rate = 0.20)
  : ∃ (second_account_balance : ℝ),
    (let total_raw_materials_cost := raw_materials_cost * (1 + raw_materials_tax_rate)
     let discounted_machinery_cost := machinery_cost * (1 - machinery_discount_rate)
     let total_machinery_cost := discounted_machinery_cost * (1 + machinery_tax_rate)
     let total_spending := total_raw_materials_cost + total_machinery_cost
     let total_savings := first_account_balance + second_account_balance
     total_spending = total_spending_rate * total_savings) ∧ 
    second_account_balance = 232.5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_bank_account_balance_l348_34850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_47_values_l348_34817

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 5 * x^2 - 3

noncomputable def g (y : ℝ) : ℝ := 
  let x := Real.sqrt ((y + 3) / 5)
  x^2 - x + 2

-- State the theorem
theorem sum_of_g_47_values : 
  (g 47 + g 47) = 24 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_47_values_l348_34817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeters_ascending_order_l348_34833

/-- The perimeter of shape P formed by five congruent right-angled triangles -/
noncomputable def perimeter_P (r : ℝ) : ℝ := (2 + 3 * Real.sqrt 2) * r

/-- The perimeter of shape Q formed by five congruent right-angled triangles -/
noncomputable def perimeter_Q (r : ℝ) : ℝ := (6 + Real.sqrt 2) * r

/-- The perimeter of shape R formed by five congruent right-angled triangles -/
noncomputable def perimeter_R (r : ℝ) : ℝ := (4 + 3 * Real.sqrt 2) * r

/-- Theorem stating that the perimeters are in ascending order P < Q < R -/
theorem perimeters_ascending_order (r : ℝ) (h : r > 0) :
  perimeter_P r < perimeter_Q r ∧ perimeter_Q r < perimeter_R r := by
  sorry

#check perimeters_ascending_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeters_ascending_order_l348_34833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_l348_34892

noncomputable def x (t : ℝ) : ℝ := Real.log t

noncomputable def y (t : ℝ) : ℝ := t^3 + 2*t + 4

noncomputable def dxdt (t : ℝ) : ℝ := 1 / t

noncomputable def dydt (t : ℝ) : ℝ := 3*t^2 + 2

noncomputable def dydx (t : ℝ) : ℝ := (dydt t) / (dxdt t)

noncomputable def d2ydx2 (t : ℝ) : ℝ := (9*t^2 + 2) / (dxdt t)

theorem second_derivative_parametric (t : ℝ) (h : t > 0) : 
  d2ydx2 t = 9*t^3 + 2*t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_derivative_parametric_l348_34892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l348_34825

/-- Given a function f: ℝ → ℝ, if its graph is tangent to the line y = -x + 8 at the point (5, f(5)),
    then f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (f 5 = -5 + 8 ∧ deriv f 5 = -1) →
  f 5 + deriv f 5 = 2 := by
  intro h
  have h1 : f 5 = 3 := by
    rw [h.1]
    norm_num
  have h2 : deriv f 5 = -1 := h.2
  calc
    f 5 + deriv f 5 = 3 + (-1) := by rw [h1, h2]
    _ = 2 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l348_34825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_condition_l348_34884

def sequenceM (M : ℕ+) : ℕ → ℚ
  | 0 => M + 1/2
  | n + 1 => let a := sequenceM M n; a * ⌊a⌋

def contains_integer_term (M : ℕ+) : Prop :=
  ∃ n : ℕ, ∃ k : ℤ, sequenceM M n = k

theorem sequence_integer_condition (M : ℕ+) :
  contains_integer_term M ↔ M > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_condition_l348_34884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_zero_l348_34877

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (3, 2)

def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)

def vector_a (x : ℝ) : ℝ × ℝ := (2*x + 3, x^2 - 4)

theorem vector_angle_zero (x : ℝ) :
  (∃ k : ℝ, vector_a x = (vector_AB.1 * k, vector_AB.2 * k) ∧ k > 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_zero_l348_34877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_750m_90kmh_l348_34887

/-- Calculates the time (in seconds) for a train to cross a platform of equal length -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let platform_length : ℝ := train_length
  let total_distance : ℝ := train_length + platform_length
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train with length 750 meters and speed 90 km/h takes 60 seconds to cross a platform of equal length -/
theorem train_crossing_time_750m_90kmh :
  train_crossing_time 750 90 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_750m_90kmh_l348_34887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_numbered_cube_existence_l348_34804

/-- Represents a cube with numbers 1 to 8 on its vertices -/
structure NumberedCube where
  vertices : Fin 8 → Fin 8
  bijective : Function.Bijective vertices

/-- Checks if a face of the cube satisfies the sum condition -/
def face_sum_condition (c : NumberedCube) (v1 v2 v3 v4 : Fin 8) : Prop :=
  c.vertices v1 = c.vertices v2 + c.vertices v3 + c.vertices v4 ∨
  c.vertices v2 = c.vertices v1 + c.vertices v3 + c.vertices v4 ∨
  c.vertices v3 = c.vertices v1 + c.vertices v2 + c.vertices v4 ∨
  c.vertices v4 = c.vertices v1 + c.vertices v2 + c.vertices v3

/-- Checks if three faces of the cube satisfy the sum condition -/
def three_faces_sum_condition (c : NumberedCube) : Prop :=
  ∃ (f1 f2 f3 : Fin 8 × Fin 8 × Fin 8 × Fin 8),
    face_sum_condition c f1.1 f1.2.1 f1.2.2.1 f1.2.2.2 ∧
    face_sum_condition c f2.1 f2.2.1 f2.2.2.1 f2.2.2.2 ∧
    face_sum_condition c f3.1 f3.2.1 f3.2.2.1 f3.2.2.2

/-- Checks if the vertex with 6 is adjacent to the given set of numbers -/
def adjacent_to_six (c : NumberedCube) (s : Finset (Fin 8)) : Prop :=
  ∃ v : Fin 8, c.vertices v = 6 ∧ 
    ∃ (v1 v2 v3 : Fin 8), s = {c.vertices v1, c.vertices v2, c.vertices v3} ∧
      v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3

/-- The main theorem stating the existence of a valid cube configuration -/
theorem numbered_cube_existence : 
  ∃ (c : NumberedCube), 
    three_faces_sum_condition c ∧
    (adjacent_to_six c {2, 3, 5} ∨ 
     adjacent_to_six c {3, 5, 7} ∨ 
     adjacent_to_six c {2, 3, 7}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_numbered_cube_existence_l348_34804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contests_path_count_is_255_l348_34842

/-- Represents the diamond-shaped grid of letters -/
structure Grid where
  -- We don't need to define the full grid structure for this statement
  dummy : Unit

/-- Represents a path in the grid -/
structure GridPath (g : Grid) where
  -- Simplified representation of a path
  dummy : Unit

/-- Predicate to check if a path spells "CONTESTS" -/
def spells_contests (g : Grid) (p : GridPath g) : Prop := sorry

/-- Count of paths spelling "CONTESTS" in the grid -/
def contest_paths_count (g : Grid) : ℕ := sorry

/-- The main theorem stating that the number of paths spelling "CONTESTS" is 255 -/
theorem contests_path_count_is_255 (g : Grid) : contest_paths_count g = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contests_path_count_is_255_l348_34842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrews_age_is_72_over_23_l348_34844

-- Define variables
variable (andrew_age : ℚ)
variable (father_age : ℚ)
variable (grandfather_age : ℚ)

-- Define relationships
axiom father_age_relation : father_age = 8 * andrew_age
axiom grandfather_age_relation : grandfather_age = 3 * father_age
axiom age_difference : grandfather_age - andrew_age = 72

-- Theorem to prove
theorem andrews_age_is_72_over_23 : andrew_age = 72 / 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrews_age_is_72_over_23_l348_34844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l348_34860

/-- Represents the train's journey with an accident -/
structure TrainJourney where
  initial_speed : ℝ
  total_distance : ℝ
  time_before_accident : ℝ
  detention_time : ℝ
  speed_reduction_factor : ℝ
  delay : ℝ
  accident_distance_shift : ℝ
  shifted_delay : ℝ

/-- Calculates the total time taken for the journey -/
noncomputable def total_time (tj : TrainJourney) : ℝ :=
  tj.time_before_accident + tj.detention_time +
  (tj.total_distance - tj.initial_speed * tj.time_before_accident) /
  (tj.initial_speed * tj.speed_reduction_factor)

/-- Theorem stating that under the given conditions, the total distance is 640 miles -/
theorem train_journey_distance : ∃ (tj : TrainJourney),
  tj.time_before_accident = 2 ∧
  tj.detention_time = 1 ∧
  tj.speed_reduction_factor = 2/3 ∧
  tj.delay = 4 ∧
  tj.accident_distance_shift = 120 ∧
  tj.shifted_delay = 3.5 ∧
  total_time tj = tj.total_distance / tj.initial_speed + tj.delay ∧
  (total_time { tj with
    total_distance := tj.total_distance,
    time_before_accident := (tj.initial_speed * tj.time_before_accident + tj.accident_distance_shift) / tj.initial_speed
  }) = tj.total_distance / tj.initial_speed + tj.shifted_delay ∧
  tj.total_distance = 640 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l348_34860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleSegmentArrangement_l348_34810

-- Define a segment on a number line
structure Segment where
  start : ℤ
  length : ℕ

-- Define the property of even length
def isEvenLength (s : Segment) : Prop :=
  ∃ k : ℕ, s.length = 2 * k

-- Define the intersection of two segments
def intersection (s1 s2 : Segment) : Option Segment :=
  let s1End := s1.start + s1.length
  let s2End := s2.start + s2.length
  if s1.start ≤ s2.start ∧ s2.start < s1End then
    some ⟨s2.start, (min s1End s2End - s2.start).toNat⟩
  else if s2.start ≤ s1.start ∧ s1.start < s2End then
    some ⟨s1.start, (min s1End s2End - s1.start).toNat⟩
  else
    none

-- Define the property of odd length
def isOddLength (s : Option Segment) : Prop :=
  match s with
  | some seg => ∃ k : ℕ, seg.length = 2 * k + 1
  | none => false

-- Theorem statement
theorem impossibleSegmentArrangement :
  ¬∃ (s1 s2 s3 : Segment),
    isEvenLength s1 ∧ isEvenLength s2 ∧ isEvenLength s3 ∧
    isOddLength (intersection s1 s2) ∧
    isOddLength (intersection s2 s3) ∧
    isOddLength (intersection s1 s3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleSegmentArrangement_l348_34810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l348_34897

def y : ℕ → ℝ
  | 0 => 200  -- Add this case to handle Nat.zero
  | 1 => 200
  | k + 2 => 3 * y (k + 1) + 4

theorem series_sum : ∑' k : ℕ, 1 / (y k + 1) = 1 / 201 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l348_34897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_duration_decrease_l348_34891

/-- Represents the rate of decrease in homework duration -/
def x : Real := Real.mk 0  -- Placeholder value, can be changed as needed

/-- The initial homework duration in minutes -/
def initial_duration : ℝ := 100

/-- The final homework duration in minutes -/
def final_duration : ℝ := 70

/-- The number of adjustments made -/
def num_adjustments : ℕ := 2

/-- Theorem stating the relationship between initial duration, rate of decrease, and final duration -/
theorem homework_duration_decrease :
  (initial_duration * (1 - x)^num_adjustments = final_duration) ↔
  (100 * (1 - x)^2 = 70) := by
  sorry

#check homework_duration_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_duration_decrease_l348_34891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l348_34805

noncomputable def f (x : ℝ) := -2 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 1

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ k : ℤ, ∀ x : ℝ, f ((-π/4 + k * π) + x) = f ((-π/4 + k * π) - x)) ∧
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ x : ℝ, f x ≥ -1) ∧
  (∃ x : ℝ, f x = 2) ∧
  (∃ x : ℝ, f x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l348_34805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l348_34828

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  Real.log (4/5) / Real.log a < 1 ↔ a ∈ Set.Ioo 0 (4/5) ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l348_34828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_are_knights_l348_34849

-- Define the inhabitants
inductive Inhabitant : Type
| A
| B
| C

-- Define the possible statuses
inductive Status : Type
| Knight
| Liar

-- Function to determine if a statement is true based on the speaker's status
def isTrueStatement (speakerStatus : Status) (statement : Prop) : Prop :=
  match speakerStatus with
  | Status.Knight => statement
  | Status.Liar => ¬statement

-- A's statement: B is a knight
def statementA (statusB : Status) : Prop :=
  statusB = Status.Knight

-- B's statement: If A is a knight, then C is a knight
def statementB (statusA statusC : Status) : Prop :=
  (statusA = Status.Knight) → (statusC = Status.Knight)

-- Theorem to prove
theorem all_are_knights :
  ∀ (statusA statusB statusC : Status),
    (isTrueStatement statusA (statementA statusB)) →
    (isTrueStatement statusB (statementB statusA statusC)) →
    (statusA = Status.Knight ∧ statusB = Status.Knight ∧ statusC = Status.Knight) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_are_knights_l348_34849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_speed_increase_l348_34818

/-- Ivan's usual travel time in minutes -/
noncomputable def T : ℝ := 520 / 3

/-- Ivan's usual speed (arbitrary units) -/
noncomputable def v : ℝ := 1

/-- The distance to work (same units as v * T) -/
noncomputable def D : ℝ := v * T

/-- The time Ivan arrives when he increases his speed by 60% and leaves 40 minutes late -/
def early_arrival : ℝ := 25

theorem ivan_speed_increase (late_departure : ℝ) (speed_increase : ℝ) 
  (h1 : late_departure = 40)
  (h2 : speed_increase = 0.3)
  (h3 : D / ((1 + 0.6) * v) = T - (late_departure + early_arrival)) :
  D / ((1 + speed_increase) * v) = T - late_departure := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_speed_increase_l348_34818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_convoy_time_l348_34881

/-- The time for a convoy to pass through a tunnel -/
noncomputable def convoy_time (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then
    3480 / x
  else if 12 < x ∧ x ≤ 20 then
    5 * x + 2880 / x + 10
  else
    0

/-- The theorem stating the minimum time for the convoy to pass through the tunnel -/
theorem min_convoy_time :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 20 ∧
  convoy_time x = 254 ∧
  ∀ (y : ℝ), 0 < y ∧ y ≤ 20 → convoy_time y ≥ convoy_time x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_convoy_time_l348_34881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l348_34800

/-- A number is square-free if it's not divisible by the square of any prime number. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

/-- Main theorem: If c is square-free and b^2 * c is divisible by a^2, then b is divisible by a. -/
theorem divisibility_theorem (a b c : ℕ) (h1 : IsSquareFree c) (h2 : a^2 ∣ b^2 * c) : a ∣ b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_theorem_l348_34800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_l_l348_34885

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the line l
def l (x y : ℝ) : Prop := 2*x - y - 6 = 0

-- Distance function from a point (x, y) to the line l
noncomputable def distance_to_l (x y : ℝ) : ℝ := 
  |2*x - y - 6| / Real.sqrt 5

-- Theorem stating the maximum distance
theorem max_distance_C₁_to_l :
  ∃ (d : ℝ), d = 2 * Real.sqrt 5 ∧
  (∀ (x y : ℝ), C₁ x y → distance_to_l x y ≤ d) ∧
  (∃ (x y : ℝ), C₁ x y ∧ distance_to_l x y = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₁_to_l_l348_34885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_counts_l348_34829

/-- The set of natural numbers less than 100 -/
def N : Finset Nat := Finset.range 100

/-- Numbers in N divisible by 2 but not by 3 -/
def A : Finset Nat := N.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0)

/-- Numbers in N divisible by 2 or by 3 -/
def B : Finset Nat := N.filter (λ n => n % 2 = 0 ∨ n % 3 = 0)

/-- Numbers in N not divisible by either 2 or 3 -/
def C : Finset Nat := N.filter (λ n => n % 2 ≠ 0 ∧ n % 3 ≠ 0)

theorem divisibility_counts :
  (Finset.card A = 33) ∧
  (Finset.card B = 66) ∧
  (Finset.card C = 33) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_counts_l348_34829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_lines_l348_34813

/-- Definition of the ellipse E -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

/-- Definition of the circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2 = 0

/-- Condition for a line to be tangent to the circle -/
def is_tangent_to_circle (x₀ y₀ k : ℝ) : Prop :=
  ((2 - x₀)^2 - 2) * k^2 + 2*(2 - x₀)*y₀*k + y₀^2 - 2 = 0

/-- Main theorem -/
theorem ellipse_and_tangent_lines :
  (∀ x y, ellipse x y → (x^2 + y^2 = 16 ∨ x^2 + y^2 = 12)) ∧
  (∀ x₀ y₀, ellipse x₀ y₀ →
    (x₀ = -2 ∧ (y₀ = 3 ∨ y₀ = -3)) ∨
    (x₀ = 18/5 ∧ (y₀ = Real.sqrt 57/5 ∨ y₀ = -Real.sqrt 57/5)) →
    ∃ k₁ k₂, k₁ * k₂ = 1/2 ∧
      is_tangent_to_circle x₀ y₀ k₁ ∧
      is_tangent_to_circle x₀ y₀ k₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_lines_l348_34813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l348_34868

theorem right_triangle_condition (A B C : ℝ) (h : Real.sin A * Real.cos B = 1 - Real.cos A * Real.sin B) :
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_condition_l348_34868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l348_34882

theorem complex_absolute_value : 
  Complex.abs (2 + Complex.I^2 + 2*Complex.I^3) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l348_34882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yearly_savings_l348_34802

def monthly_salary : ℝ := 7920
def consumable_percentage : ℝ := 0.6
def clothes_transport_percentage : ℝ := 0.5
def clothes_transport_amount : ℝ := 1584
def months_in_year : ℝ := 12

theorem yearly_savings : 
  (1 - consumable_percentage) * (1 - clothes_transport_percentage) * monthly_salary * months_in_year = 19008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yearly_savings_l348_34802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l348_34801

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1

-- Define the circle
def circle' (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 8 = 0

-- Statement: The minimum length of the tangent is √7
theorem min_tangent_length :
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧
    (∀ (x y : ℝ), circle' x y →
      (x - x₀)^2 + (y - y₀)^2 ≥ 7) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l348_34801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l348_34814

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x - a|)

-- State the theorem
theorem sufficient_not_necessary_condition :
  (∀ a : ℝ, a ≤ 0 → ∀ x y : ℝ, 1 < x → x < y → f a x ≤ f a y) ∧
  (∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, 1 < x → x < y → f a x ≤ f a y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l348_34814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_together_l348_34855

/-- The time it takes Sawyer to clean the entire house alone, in hours -/
noncomputable def sawyer_time : ℝ := 6

/-- The time it takes Nick to clean the entire house alone, in hours -/
noncomputable def nick_time : ℝ := sawyer_time * 3 / 2

/-- The rate at which Sawyer cleans the house, in house per hour -/
noncomputable def sawyer_rate : ℝ := 1 / sawyer_time

/-- The rate at which Nick cleans the house, in house per hour -/
noncomputable def nick_rate : ℝ := 1 / nick_time

/-- The combined rate at which Sawyer and Nick clean the house, in house per hour -/
noncomputable def combined_rate : ℝ := sawyer_rate + nick_rate

/-- The time it takes Sawyer and Nick to clean the entire house together, in hours -/
noncomputable def combined_time : ℝ := 1 / combined_rate

theorem cleaning_time_together : combined_time = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_together_l348_34855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_result_l348_34809

/-- Definition of x as a continued fraction --/
noncomputable def x : ℝ := 1 + Real.sqrt 3 / (1 + Real.sqrt 3 / (1 + Real.sqrt 3 / (1 + Real.sqrt 3)))

/-- The main theorem --/
theorem continued_fraction_result :
  1 / ((x + 2) * (x - 3)) = (6 + Real.sqrt 3) / -33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_result_l348_34809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_second_quadrant_l348_34846

theorem cos_value_second_quadrant (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_value_second_quadrant_l348_34846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_conditions_l348_34865

-- Define a real-valued function on the entire real line
variable (f : ℝ → ℝ)

-- Assume f is differentiable everywhere
variable (hf : Differentiable ℝ f)

-- Define the derivative of f
noncomputable def f' : ℝ → ℝ := deriv f

-- Define what it means for a function to be odd
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define what it means for a function to be even
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define what it means for a function to be periodic with period T
def IsPeriodic (g : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, g (x + T) = g x

-- Define what it means for a function to have an extremum at a point
def HasExtremumAt (g : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), g y ≤ g x ∨ g y ≥ g x

theorem sufficient_conditions (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (IsOdd f → IsEven (λ x ↦ f' f x)) ∧
  (∀ T : ℝ, IsPeriodic f T → IsPeriodic (λ x ↦ f' f x) T) ∧
  (∀ x : ℝ, HasExtremumAt f x → f' f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_conditions_l348_34865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l348_34878

/-- Given a circle C₁ and a parabola C₂ that intersect at points A and B,
    prove that the equation of C₂ is y² = 32x/5 -/
theorem parabola_equation (x y : ℝ) (p : ℝ) (A B : ℝ × ℝ) :
  (x ^ 2 + (y - 2) ^ 2 = 4) →  -- Circle C₁
  (y ^ 2 = 2 * p * x) →        -- Parabola C₂
  (p > 0) →                    -- p is positive
  (A ∈ Set.univ) →             -- A is a point in ℝ²
  (B ∈ Set.univ) →             -- B is a point in ℝ²
  (A ≠ B) →                    -- A and B are distinct points
  (Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 8 * Real.sqrt 5 / 5) →  -- |AB| = 8√5/5
  (y ^ 2 = 32 * x / 5) :=      -- Equation of C₂
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l348_34878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l348_34820

noncomputable def f (a b x : ℝ) : ℝ :=
  if x < 3 then a * x^2 + b else 10 - 2 * x

theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) ↔ (a = -1/6 ∧ b = 14/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l348_34820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_same_axis_of_symmetry_l348_34893

/-- Convex quadrilateral -/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
sorry

/-- Tangent line to a function at a point -/
def IsTangentLine (f : ℝ → ℝ) (x y : ℝ) : Prop :=
sorry

/-- Inscribed circle in a quadrilateral -/
def IsInscribedCircle (center : ℝ × ℝ) (radius : ℝ) (A B C D : ℝ × ℝ) : Prop :=
sorry

/-- Two quadratic functions with specific properties have the same axis of symmetry -/
theorem quadratic_functions_same_axis_of_symmetry
  (p₁ q₁ r₁ p₂ q₂ r₂ : ℝ)
  (h₁ : p₁ > 0)
  (h₂ : p₂ < 0)
  (f₁ : ℝ → ℝ)
  (f₂ : ℝ → ℝ)
  (hf₁ : f₁ = fun x ↦ p₁ * x^2 + q₁ * x + r₁)
  (hf₂ : f₂ = fun x ↦ p₂ * x^2 + q₂ * x + r₂)
  (hIntersect : ∃ (a b : ℝ), a ≠ b ∧ f₁ a = f₂ a ∧ f₁ b = f₂ b)
  (hTangentQuadrilateral : ∃ (A B C D : ℝ × ℝ),
    ConvexQuadrilateral A B C D ∧
    IsTangentLine f₁ A.1 A.2 ∧
    IsTangentLine f₁ B.1 B.2 ∧
    IsTangentLine f₂ C.1 C.2 ∧
    IsTangentLine f₂ D.1 D.2)
  (hInscribedCircle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    IsInscribedCircle center radius A B C D) :
  -q₁ / (2 * p₁) = -q₂ / (2 * p₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_same_axis_of_symmetry_l348_34893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l348_34880

/-- Given a cubic function f(x) = ax³ - bx + 2 where a and b are real numbers,
    if f(-3) = -1, then f(3) = 5. -/
theorem cubic_function_property (a b : ℝ) :
  let f := λ x : ℝ ↦ a * x^3 - b * x + 2
  f (-3) = -1 → f 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_property_l348_34880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_or_one_l348_34858

noncomputable def f (x a : ℝ) : ℝ := 1 / (x - 1) + a / (x + a - 1) + 1 / (x + 1)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_implies_a_zero_or_one (a : ℝ) :
  is_odd (f · a) → (a = 0 ∨ a = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_or_one_l348_34858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_common_point_to_pole_l348_34835

/-- The distance from the common point of two polar curves to the pole --/
noncomputable def distance_to_pole : ℝ := 1 + Real.sqrt 3

/-- First curve equation in polar coordinates --/
def curve1 (ρ θ : ℝ) : Prop := ρ = Real.sin θ + 2

/-- Second curve equation in polar coordinates --/
def curve2 (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 2

/-- Theorem stating that the distance from the common point of the curves to the pole is 1 + √3 --/
theorem distance_from_common_point_to_pole :
  ∃ (ρ θ : ℝ), curve1 ρ θ ∧ curve2 ρ θ ∧ ρ = distance_to_pole := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_common_point_to_pole_l348_34835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_lambda_theorem_l348_34824

/-- Parabola with vertex at origin and focus on x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  focus_on_x_axis : eq p 0

/-- Triangle inscribed in the parabola -/
structure InscribedTriangle (par : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  on_parabola : par.eq A.1 A.2 ∧ par.eq B.1 B.2 ∧ par.eq C.1 C.2
  centroid_is_focus : (A.1 + B.1 + C.1) / 3 = par.p ∧ (A.2 + B.2 + C.2) / 3 = 0

/-- Line equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Tangent line to the parabola -/
noncomputable def TangentLine (par : Parabola) (M : ℝ × ℝ) : Line := sorry

/-- Point on parabola -/
def OnParabola (par : Parabola) (P : ℝ × ℝ) : Prop := par.eq P.1 P.2

/-- Perpendicular lines -/
def Perpendicular (l1 l2 : Line) : Prop := sorry

/-- Angle between two lines -/
noncomputable def AngleBetween (l1 l2 : Line) : ℝ := sorry

theorem parabola_and_lambda_theorem (par : Parabola) (tri : InscribedTriangle par) :
  (∃ (BC : Line), BC.a = 4 ∧ BC.b = 1 ∧ BC.c = -20) →
  (par.eq = fun x y ↦ y^2 = 16*x) ∧
  (∀ (M : ℝ × ℝ) (N : ℝ × ℝ) (E : ℝ × ℝ),
    OnParabola par M →
    OnParabola par N →
    let l := TangentLine par M
    let MN := Line.mk (N.1 - M.1) (N.2 - M.2) 0
    let ME := Line.mk 1 0 0
    let MF := Line.mk (par.p - M.1) (-M.2) 0
    Perpendicular l MN →
    E.2 = M.2 →
    E.1 > M.1 →
    (∃ (lambda : ℝ), AngleBetween MF MN = lambda * AngleBetween MN ME) →
    lambda = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_lambda_theorem_l348_34824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l348_34898

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then x - 3/x - 2
  else if x < 0 then x - 3/x + 2
  else 0

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x - 3/x - 2) ∧  -- given condition
  (f 0 = 0) ∧  -- property of odd function
  (∀ x < 0, f x = x - 3/x + 2) ∧  -- derived from odd property
  (f (-3) = 0 ∧ f 0 = 0 ∧ f 3 = 0) ∧  -- zeros of f
  (∀ x, f x = 0 → x = -3 ∨ x = 0 ∨ x = 3)  -- only zeros of f
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l348_34898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l348_34859

/-- Represents the number of days it takes for a person to complete a work alone -/
structure WorkDays where
  days : ℚ
  days_positive : days > 0

/-- Represents the portion of work completed in a given number of days -/
def work_completed (w : WorkDays) (days : ℚ) : ℚ := days / w.days

theorem work_completion_time 
  (a : WorkDays) 
  (b : WorkDays)
  (h1 : b.days = 40)
  (h2 : work_completed a 9 + work_completed b 9 + work_completed b 23 = 1) :
  a.days = 45 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l348_34859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l348_34845

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ (a d : ℝ), 
    Real.sqrt (49 + k) = a ∧ 
    Real.sqrt (225 + k) = a + d ∧ 
    Real.sqrt (441 + k) = a + 2*d) → 
  k = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l348_34845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l348_34803

/-- The minimum distance from any point on the circle ρ=2 to the line ρ(cos θ + √3 sin θ) = 6 is 1 -/
theorem min_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 = 6}
  ∃ (d : ℝ), d = 1 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ circle → 
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l348_34803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l348_34867

theorem trajectory_equation (OA OB OC : ℝ × ℝ) (l m : ℝ) :
  (OA.1^2 + OA.2^2 = 1) →
  (OB.1^2 + OB.2^2 = 1) →
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) →
  (OC = (l * OA.1 + m * OB.1, l * OA.2 + m * OB.2)) →
  let M := ((OA.1 + OB.1) / 2, (OA.2 + OB.2) / 2)
  (OC.1 - M.1)^2 + (OC.2 - M.2)^2 = 1 →
  (l - 1/2)^2 + (m - 1/2)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l348_34867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l348_34812

/-- The probability of drawing two balls of the same color from a bag containing
    5 green balls, 7 red balls, and 3 blue balls, with replacement. -/
theorem same_color_probability : (83 : ℚ) / 225 = 
  let green_balls : ℕ := 5
  let red_balls : ℕ := 7
  let blue_balls : ℕ := 3
  let total_balls : ℕ := green_balls + red_balls + blue_balls
  let prob_green : ℚ := (green_balls : ℚ) / (total_balls : ℚ)
  let prob_red : ℚ := (red_balls : ℚ) / (total_balls : ℚ)
  let prob_blue : ℚ := (blue_balls : ℚ) / (total_balls : ℚ)
  prob_green ^ 2 + prob_red ^ 2 + prob_blue ^ 2
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l348_34812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l348_34811

-- Define the logarithm base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Define the condition from the problem
def condition (x y : ℝ) : Prop := log4 (x + 2*y) + log4 (x - 2*y) = 1

-- State the theorem
theorem min_value_theorem (x y : ℝ) (h : condition x y) : 
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ ∀ (a b : ℝ), condition a b → |a| - |b| ≥ m := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l348_34811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_divisible_by_13_l348_34853

theorem count_three_digit_numbers_divisible_by_13 : ∃ n : ℕ, n = 69 :=
  let lower_bound := 100
  let upper_bound := 999
  let divisor := 13
  let smallest := (lower_bound + divisor - 1) / divisor * divisor
  let largest := upper_bound / divisor * divisor
  let count := (largest - smallest) / divisor + 1
  ⟨count, by sorry⟩

#eval (999 / 13 - 99 / 13 : ℕ)  -- This should output 69

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_divisible_by_13_l348_34853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_giselle_badge_number_l348_34831

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def consecutive_primes (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ is_prime p₄ ∧
  p₂ = Nat.minFac (p₁ + 1) ∧
  p₃ = Nat.minFac (p₂ + 1) ∧
  p₄ = Nat.minFac (p₃ + 1)

theorem giselle_badge_number 
  (p₁ p₂ p₃ p₄ : ℕ) 
  (h1 : consecutive_primes p₁ p₂ p₃ p₄) 
  (h2 : p₃ + p₄ = 2025) : 
  p₄ = 1014 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_giselle_badge_number_l348_34831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_bound_l348_34896

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of points
def PointSet := Set Point

-- Define a function to calculate the angle between three points
noncomputable def angle (a b c : Point) : ℝ := sorry

-- Define a function to find the largest angle in a triangle formed by three points
noncomputable def largestAngle (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem largest_angle_bound (n : ℕ) (points : Finset Point) 
  (h1 : n = 4 ∨ n = 5 ∨ n = 6) 
  (h2 : points.card = n) : 
  ∃ a b c : Point, a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ 
  largestAngle a b c ≥ 360 / n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_bound_l348_34896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l348_34826

/-- The equation of the tangent line to y = 1/x^2 at (2, 1/4) is x + 4y - 3 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => 1 / t^2
  let P : ℝ × ℝ := (2, 1/4)
  let tangent_line : ℝ → ℝ := λ t => -(1/4) * t + 3/4
  (∀ t, (t, f t) ∈ Set.range (λ p : ℝ × ℝ => (p.1, f p.1))) →
  (2, 1/4) ∈ Set.range (λ p : ℝ × ℝ => (p.1, f p.1)) →
  (HasDerivAt f (-1/4) 2) →
  (∀ t, tangent_line t = -(1/4) * t + 3/4) →
  x + 4 * y - 3 = 0 ↔ y = tangent_line x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l348_34826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l348_34851

-- Define the complex number w
noncomputable def w (θ : ℝ) : ℂ := 3 * Complex.exp (θ * Complex.I)

-- Define the function f(w) = w + 2/w
noncomputable def f (w : ℂ) : ℂ := w + 2 / w

-- Theorem statement
theorem locus_is_ellipse (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (Complex.re (f (w θ)))^2 / a^2 + (Complex.im (f (w θ)))^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l348_34851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_circumscribed_sphere_l348_34807

/-- Given a right square prism with volume 8, the minimum volume of its circumscribed sphere is 4√3π -/
theorem min_volume_circumscribed_sphere (a h : ℝ) : 
  a > 0 → h > 0 → a^2 * h = 8 → 
  (4 / 3) * Real.pi * (Real.sqrt 3)^3 ≤ (4 / 3) * Real.pi * ((a^2 + a^2 + h^2) / 4)^(3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_volume_circumscribed_sphere_l348_34807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l348_34830

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1 ∧ x ≠ -1

-- Define the line l
def l (x y : ℝ) : Prop := 2*x + Real.sqrt 3*y + 11 = 0

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) : ℝ :=
  abs (2*x + Real.sqrt 3*y + 11) / Real.sqrt 7

-- Theorem statement
theorem min_distance_C_to_l :
  ∃ (d : ℝ), d = Real.sqrt 7 ∧
  (∀ (x y : ℝ), C x y → distance_point_to_line x y ≥ d) ∧
  (∃ (x y : ℝ), C x y ∧ distance_point_to_line x y = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l348_34830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_projection_l348_34874

theorem min_ratio_projection :
  (∀ m : ℝ, m > 0 → (2^m - 2^(8/(2*m+1))) / (2^(-m) - 2^(-8/(2*m+1))) ≥ 8 * Real.sqrt 2) ∧
  (∃ m : ℝ, m > 0 ∧ (2^m - 2^(8/(2*m+1))) / (2^(-m) - 2^(-8/(2*m+1))) = 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_projection_l348_34874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypersphere_measure_l348_34847

-- Define the dimension of the space
def dimension : ℕ := 4

-- Define the radius
variable (r : ℝ)

-- Define the three-dimensional measure (volume) of the hypersphere
noncomputable def volume (r : ℝ) : ℝ := 12 * Real.pi * r^3

-- Define the four-dimensional measure of the hypersphere
noncomputable def hypervolume (r : ℝ) : ℝ := 3 * Real.pi * r^4

-- Theorem statement
theorem hypersphere_measure (r : ℝ) : 
  deriv hypervolume r = volume r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypersphere_measure_l348_34847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_caseC_has_two_solutions_l348_34843

-- Define the structure for a triangle case
structure TriangleCase where
  a : ℝ
  b : ℝ
  A : ℝ

-- Define the four cases
noncomputable def caseA : TriangleCase := { a := 5, b := 5, A := 50 * Real.pi / 180 }
noncomputable def caseB : TriangleCase := { a := 5, b := 10, A := 30 * Real.pi / 180 }
noncomputable def caseC : TriangleCase := { a := 3, b := 4, A := 30 * Real.pi / 180 }
noncomputable def caseD : TriangleCase := { a := 12, b := 10, A := 135 * Real.pi / 180 }

-- Function to check if a case has two solutions
def hasTwoSolutions (t : TriangleCase) : Prop :=
  let sinB := t.b * Real.sin t.A / t.a
  sinB < 1 ∧ t.b > t.a

-- Theorem stating that only case C has two solutions
theorem only_caseC_has_two_solutions :
  hasTwoSolutions caseC ∧
  ¬hasTwoSolutions caseA ∧
  ¬hasTwoSolutions caseB ∧
  ¬hasTwoSolutions caseD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_caseC_has_two_solutions_l348_34843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l348_34837

-- Define the function types
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom domain_f : ∀ x, f x ∈ Set.univ
axiom sum_one : ∀ x, g x + f x = 1
axiom g_odd : ∀ x, g (x + 1) = -g (-x + 1)
axiom f_odd : ∀ x, f (2 - x) = -f (2 + x)

-- Theorem to prove
theorem g_values (f g : ℝ → ℝ) 
  (hf : ∀ x, f x ∈ Set.univ)
  (h1 : ∀ x, g x + f x = 1)
  (h2 : ∀ x, g (x + 1) = -g (-x + 1))
  (h3 : ∀ x, f (2 - x) = -f (2 + x)) :
  g 0 = -1 ∧ g 1 = 0 ∧ g 2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l348_34837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_squared_relationship_l348_34841

-- Define the coefficient of determination (R²)
noncomputable def R_squared (sum_squares_residuals : ℝ) (total_sum_squares : ℝ) : ℝ :=
  1 - sum_squares_residuals / total_sum_squares

-- Define the relationship between R² and sum of squares of residuals
theorem r_squared_relationship 
  (sum_squares_residuals₁ sum_squares_residuals₂ total_sum_squares : ℝ)
  (h_positive : total_sum_squares > 0)
  (h_less : sum_squares_residuals₁ < sum_squares_residuals₂)
  (h_bound₁ : sum_squares_residuals₁ ≥ 0)
  (h_bound₂ : sum_squares_residuals₂ ≤ total_sum_squares) :
  R_squared sum_squares_residuals₁ total_sum_squares > 
  R_squared sum_squares_residuals₂ total_sum_squares := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_squared_relationship_l348_34841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fourth_root_l348_34822

theorem simplified_fourth_root (a b : ℕ+) :
  (↑a * (b : ℝ).rpow (1/4) : ℝ) = (2^6 * 5^5 : ℝ).rpow (1/4) →
  a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fourth_root_l348_34822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_commute_l348_34870

/-- The distance from Steve's house to work in kilometers. -/
def D : ℝ := sorry

/-- Steve's speed on the way to work in km/h. -/
def V : ℝ := sorry

/-- Time spent on the road in hours. -/
def total_time : ℝ := 6

/-- Speed on the way back from work in km/h. -/
def return_speed : ℝ := 15

theorem steves_commute : 
  (D / V + D / (2 * V) = total_time) ∧ 
  (2 * V = return_speed) → 
  D = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_commute_l348_34870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_difference_exists_l348_34815

theorem divisible_difference_exists (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ (a i - a j) % (n : ℤ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_difference_exists_l348_34815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l348_34861

theorem complex_modulus (z : ℂ) (h : (1 + 2*Complex.I)*z = (1 - Complex.I)) : 
  Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_l348_34861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_m_bound_l348_34864

open Set Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1/2) * x + m else x - Real.log x

-- State the theorem
theorem monotone_f_implies_m_bound (m : ℝ) :
  Monotone (f m) → m ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_m_bound_l348_34864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proof_l348_34852

-- Define the walking and jogging speeds
noncomputable def linda_speed : ℝ := 4
noncomputable def tom_speed : ℝ := 9

-- Define the time difference between Linda's start and Tom's start
noncomputable def time_difference : ℝ := 1

-- Function to calculate the time Tom takes to cover a multiple of Linda's distance
noncomputable def tom_time (distance_multiple : ℝ) : ℝ :=
  (distance_multiple * linda_speed * time_difference) / tom_speed

-- Theorem statement
theorem time_difference_proof :
  tom_time 2 - tom_time (1/2) = 2/3 := by
  sorry

-- Use #eval with rational numbers instead of real numbers
#eval (((2 : ℚ) * 4 * 1) / 9 - ((1/2 : ℚ) * 4 * 1) / 9) * 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_proof_l348_34852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_conditions_l348_34873

theorem count_divisors_with_conditions : 
  let b_set := {b : ℕ | b > 0 ∧ 6 ∣ b ∧ b ∣ 24}
  Finset.card (Finset.filter (λ b => b > 0 ∧ 6 ∣ b ∧ b ∣ 24) (Finset.range 25)) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_with_conditions_l348_34873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l348_34899

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x^2

theorem range_of_f :
  Set.range f = {y : ℝ | y > 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l348_34899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_l348_34857

/-- The radius of the outer circle -/
def R : ℝ := 15

/-- The radius of circle B -/
def r_B : ℝ := 5

/-- The radius of circles C and D -/
def r_CD : ℝ := 3

/-- The radius of circle E -/
noncomputable def r_E (p q : ℕ) : ℚ := p / q

/-- The radius of circle F -/
noncomputable def r_F : ℝ := sorry

/-- The theorem stating the relationship between the radii -/
theorem circle_configuration (p q : ℕ) (r : ℝ) 
  (h_pq : Nat.Coprime p q) 
  (h_positive : p > 0 ∧ q > 0) :
  p + q + Int.floor r = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_l348_34857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l348_34854

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l348_34854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_neg_half_implies_expression_eq_one_fourteenth_l348_34871

theorem tan_alpha_neg_half_implies_expression_eq_one_fourteenth (α : ℝ) :
  Real.tan α = -1/2 →
  (Real.sin (2*α) + 2 * Real.cos (2*α)) / (4 * Real.cos (2*α) - 4 * Real.sin (2*α)) = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_neg_half_implies_expression_eq_one_fourteenth_l348_34871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_percentage_gain_l348_34876

/-- Calculates the percentage gain after applying multiple salary changes -/
theorem salary_percentage_gain (S : ℝ) (h : S > 0) : 
  let increase := 1.50
  let decrease := 0.90
  let bonus := 1.05
  let tax := 0.97
  let final_salary := S * increase * decrease * bonus * tax
  let percentage_gain := (final_salary - S) / S * 100
  |percentage_gain - 37.5| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_percentage_gain_l348_34876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_l348_34840

/-- Helper function to define a line through two points -/
def line_through (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (t : ℝ), (x, y) = ((1 - t) * P.1 + t * Q.1, (1 - t) * P.2 + t * Q.2)}

/-- Given a triangle ABC with vertices A(0,5), B(1,-2), and C(-7,4),
    prove that the equation of the line passing through A and the midpoint of BC
    is 4x-3y+15=0 -/
theorem median_equation (A B C D : ℝ × ℝ) : 
  A = (0, 5) → 
  B = (1, -2) → 
  C = (-7, 4) → 
  D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) → 
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ line_through A D ↔ 4*x - 3*y + 15 = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_equation_l348_34840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_meeting_time_l348_34866

/-- Alice's lap time in seconds -/
def t : ℝ := sorry

/-- Bob's lap time in seconds -/
def bob_lap_time : ℝ := 60

/-- Number of laps Alice completes when they first meet -/
def alice_laps : ℕ := 30

/-- Possible number of laps Bob completes when they first meet -/
def bob_laps : Set ℝ := {29.5, 30.5}

theorem alice_bob_meeting_time :
  (∃ l ∈ bob_laps, t * (alice_laps : ℝ) = l * bob_lap_time) →
  t = 59 ∨ t = 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_meeting_time_l348_34866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l348_34862

/-- Given a triangle with base b and altitude 3h, and a rectangle inscribed in it
    with height x and base along the triangle's base, the area of the rectangle
    is (b*x/3h)*(3h-x) -/
theorem inscribed_rectangle_area (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxh : x < 3*h) :
  (b * x / (3 * h)) * (3 * h - x) = (b * x / (3 * h)) * (3 * h - x) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l348_34862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sport_formulation_corn_syrup_amount_l348_34888

/-- Represents the ratio of ingredients in a flavored drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

/-- Theorem stating that in the sport formulation, 
    if there are 15 ounces of water, there are also 15 ounces of corn syrup -/
theorem sport_formulation_corn_syrup_amount 
  (water_amount : ℚ) (h : water_amount = 15) :
  (sport_ratio standard_ratio).corn_syrup * (water_amount / (sport_ratio standard_ratio).water) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sport_formulation_corn_syrup_amount_l348_34888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_129_l348_34821

/-- Represents a segment of the journey with speed and duration -/
structure Segment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled for a given segment -/
def distance (s : Segment) : ℝ := s.speed * s.duration

/-- Represents the entire journey as a list of segments -/
def journey : List Segment := [
  ⟨5, 4⟩, ⟨3, 2⟩, ⟨4, 3⟩,  -- Day 1
  ⟨6, 3⟩, ⟨2, 1⟩, ⟨6, 3⟩, ⟨3, 4⟩,  -- Day 2
  ⟨4, 2⟩, ⟨2, 1⟩, ⟨7, 3⟩, ⟨5, 2⟩   -- Day 3
]

/-- The theorem to be proved -/
theorem total_distance_is_129 : 
  (journey.map distance).sum = 129 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_129_l348_34821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_son_counts_in_base_seven_unique_base_l348_34819

theorem son_counts_in_base_seven :
  ∃ (a : ℕ), a > 1 ∧ 2 * a^2 + 5 * a + 3 = 136 :=
by
  use 7
  constructor
  · simp -- proves 7 > 1
  · ring -- proves 2 * 7^2 + 5 * 7 + 3 = 136

#eval 2 * 7^2 + 5 * 7 + 3 -- Evaluates to 136

theorem unique_base :
  ∀ (a : ℕ), a > 1 → 2 * a^2 + 5 * a + 3 = 136 → a = 7 :=
by
  intro a ha heq
  -- The full proof would involve solving the quadratic equation
  -- For now, we'll use sorry to skip the detailed proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_son_counts_in_base_seven_unique_base_l348_34819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sinusoid_l348_34838

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

noncomputable def period : ℝ := 2 * π / 2

noncomputable def shift : ℝ := period / 4

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 3)

theorem shift_sinusoid :
  ∀ x : ℝ, f (x + shift) = g x := by
  intro x
  -- The proof steps would go here
  sorry

#check shift_sinusoid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sinusoid_l348_34838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_string_length_l348_34808

-- Define the properties of each pole
def pole1_circumference : ℝ := 6
def pole1_height : ℝ := 20
def pole1_loops : ℕ := 5

def pole2_circumference : ℝ := 3
def pole2_height : ℝ := 10
def pole2_loops : ℕ := 3

-- Define the function to calculate the length of string for a single pole
noncomputable def string_length_for_pole (circumference height : ℝ) (loops : ℕ) : ℝ :=
  (loops : ℝ) * Real.sqrt (circumference^2 + (height / (loops : ℝ))^2)

-- Theorem statement
theorem total_string_length :
  string_length_for_pole pole1_circumference pole1_height pole1_loops +
  string_length_for_pole pole2_circumference pole2_height pole2_loops =
  5 * Real.sqrt 52 + 3 * Real.sqrt 19.89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_string_length_l348_34808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l348_34856

-- Define the vectors
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 + Real.sin x, 1)
def b : ℝ × ℝ := (2, -2)
noncomputable def c (x : ℝ) : ℝ × ℝ := (Real.sin x - 3, 1)
def d (k : ℝ) : ℝ × ℝ := (1, k)

-- Define the dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (t : ℝ), v = (t * w.1, t * w.2)

theorem vector_problem (x k : ℝ) :
  -- Part 1
  (x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧ 
   parallel (a x) (b + c x)) → x = -Real.pi/6 ∧

  -- Part 2
  (∀ y : ℝ, dot (a y) b ≥ 0) ∧

  -- Part 3
  (∃ k : ℝ, dot (a x + d k) (b + c x) = 0) ∧
  (∀ k : ℝ, dot (a x + d k) (b + c x) = 0 → k ∈ Set.Icc (-5) (-1))
  :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l348_34856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_isosceles_area_theorem_l348_34834

/-- Represents a planar figure composed of four congruent isosceles triangles -/
structure QuadIsoscelesShape where
  /-- Length of each side in the figure -/
  side_length : ℝ
  /-- Vertex angle of each isosceles triangle in degrees -/
  vertex_angle : ℝ
  /-- Assumption that side length is positive -/
  side_positive : side_length > 0
  /-- Assumption that vertex angle is between 0 and 180 degrees -/
  angle_range : 0 < vertex_angle ∧ vertex_angle < 180

/-- Calculates the area of the QuadIsoscelesShape -/
noncomputable def area (shape : QuadIsoscelesShape) : ℝ :=
  4 * shape.side_length^2 * Real.sin (shape.vertex_angle * Real.pi / 180) / 2

/-- Theorem stating that a QuadIsoscelesShape with side length 2 and vertex angle 70° has area 8 * sin(70°) -/
theorem quad_isosceles_area_theorem (shape : QuadIsoscelesShape)
  (h1 : shape.side_length = 2)
  (h2 : shape.vertex_angle = 70) :
  area shape = 8 * Real.sin (70 * Real.pi / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quad_isosceles_area_theorem_l348_34834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l348_34863

def runner1_time : ℚ := 2
def runner2_time : ℚ := 4
def runner3_time : ℚ := 11/2

def lcm_of_times (t1 t2 t3 : ℚ) : ℚ := 
  (t1 * t2 * t3).num / (t1.den.lcm t2.den).lcm t3.den

theorem runners_meet_time (t1 t2 t3 : ℚ) :
  t1 = runner1_time → t2 = runner2_time → t3 = runner3_time →
  lcm_of_times t1 t2 t3 = 44 := by
  sorry

#check runners_meet_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_meet_time_l348_34863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l348_34883

-- Define the line
def line (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Define the circle
def circle' (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define points A and B as intersections of line and circle
def is_intersection_point (x y : ℝ) : Prop := line x y ∧ circle' x y

-- Define point C as intersection of line and x-axis
def point_C : ℝ × ℝ := (3, 0)

-- Define point D as intersection of line and y-axis
def point_D : ℝ × ℝ := (0, 4)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem line_circle_intersection :
  ∃ (A B : ℝ × ℝ),
    is_intersection_point A.1 A.2 ∧
    is_intersection_point B.1 B.2 ∧
    2 * distance point_C point_D = 5 * distance A B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l348_34883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_theorem_l348_34895

theorem distinct_remainders_theorem (p : ℕ) (hp : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, Finset.card (Finset.image (λ i : Fin p ↦ (a i + i.val * k) % ↑p) Finset.univ) ≥ p / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_theorem_l348_34895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_ways_l348_34889

def num_balls : ℕ := 4
def num_boxes : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the function for the number of ways to distribute balls into boxes
def number_of_ways_to_distribute_balls_into_boxes (balls boxes : ℕ) : ℕ :=
  (choose balls (boxes - 1)) * (perm boxes boxes)

theorem ball_distribution_ways :
  number_of_ways_to_distribute_balls_into_boxes num_balls num_boxes =
  (choose num_balls (num_boxes - 1)) * (perm num_boxes num_boxes) :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_ways_l348_34889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_eq_l348_34879

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + x^2011

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => deriv (f_n n)

theorem f_2012_eq (x : ℝ) : f_n 2012 x = Real.sin x + Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2012_eq_l348_34879
