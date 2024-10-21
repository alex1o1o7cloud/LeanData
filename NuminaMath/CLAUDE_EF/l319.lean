import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_2018_225th_number_l319_31964

/-- A function that calculates the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that generates the nth number in the sequence of natural numbers with digit sum 2018 -/
def nthNumberWithDigitSum2018 (n : ℕ) : ℕ := sorry

/-- The 225th number in the sequence -/
def number225 : ℕ := nthNumberWithDigitSum2018 225

/-- Function to convert a natural number to a list of its digits -/
def toDigitList (n : ℕ) : List ℕ := sorry

theorem digit_sum_2018_225th_number :
  digitSum number225 = 2018 ∧ 
  toDigitList number225 = 3 :: List.replicate 223 9 ++ [8] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_2018_225th_number_l319_31964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l319_31946

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 15) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → |f x - 7/4| < ε :=
by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l319_31946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_bisector_property_l319_31986

/-- Given a triangle with sides a, b, and c, and angle bisector fc corresponding to side c,
    if 1/a + 1/b = 1/fc, then the angle opposite to side c is 120°. -/
theorem triangle_angle_bisector_property (a b c fc : ℝ) (γ : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ fc > 0) 
    (h_triangle : γ > 0 ∧ γ < Real.pi) (h_bisector : fc * Real.sin (γ/2) = (a * b * Real.sin γ) / (a + b)) 
    (h_relation : 1/a + 1/b = 1/fc) : γ = 2*Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_bisector_property_l319_31986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_area_and_cost_l319_31992

/-- Represents the dimensions and properties of a rectangular field with a surrounding path. -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  north_south_path_width : ℝ
  east_west_path_width_max : ℝ
  east_west_path_width_min : ℝ
  first_area_rate : ℝ
  subsequent_area_rate : ℝ
  first_area_threshold : ℝ

/-- Calculates the area of the path surrounding the field. -/
noncomputable def path_area (f : FieldWithPath) : ℝ :=
  2 * (f.field_length * f.north_south_path_width) +
  2 * (0.5 * (f.east_west_path_width_max + f.east_west_path_width_min) * f.field_width)

/-- Calculates the cost of constructing the path. -/
noncomputable def construction_cost (f : FieldWithPath) (area : ℝ) : ℝ :=
  min area f.first_area_threshold * f.first_area_rate +
  max (area - f.first_area_threshold) 0 * f.subsequent_area_rate

/-- Theorem stating the area of the path and its construction cost. -/
theorem path_area_and_cost (f : FieldWithPath) 
  (h1 : f.field_length = 75)
  (h2 : f.field_width = 55)
  (h3 : f.north_south_path_width = 4)
  (h4 : f.east_west_path_width_max = 4)
  (h5 : f.east_west_path_width_min = 2.5)
  (h6 : f.first_area_rate = 2)
  (h7 : f.subsequent_area_rate = 1.5)
  (h8 : f.first_area_threshold = 250) :
  path_area f = 957.5 ∧ construction_cost f (path_area f) = 1561.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_area_and_cost_l319_31992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l319_31920

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : abc.a > abc.c)
  (h2 : abc.b * abc.b * (Real.cos abc.B) = 2)  -- Represents BA · BC = 2
  (h3 : Real.cos abc.B = 1/3)
  (h4 : abc.b = 3) :
  abc.a = 3 ∧ abc.c = 2 ∧ Real.cos (abc.B - abc.C) = 23/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l319_31920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_variables_l319_31947

theorem order_of_variables (a b c : ℝ) : 
  a = Real.log 3 / Real.log (1/2) → 
  b = (1/3 : ℝ) ^ (0.2 : ℝ) → 
  c = (1/2 : ℝ) ^ (-(0.5 : ℝ)) → 
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_variables_l319_31947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_rule_implies_three_or_nine_l319_31900

/-- A divisibility rule that doesn't depend on digit order -/
def digit_order_independent_divisibility (M : ℕ) : Prop :=
  ∀ (n : ℕ) (digits : List ℕ),
    digits.length = n →
    (∀ d ∈ digits, d < 10) →
    (∀ perm : List ℕ, perm.isPerm digits → 
      (digits.foldl (fun acc d ↦ acc * 10 + d) 0) % M = 0 ↔ 
      (perm.foldl (fun acc d ↦ acc * 10 + d) 0) % M = 0)

theorem divisibility_rule_implies_three_or_nine (M : ℕ) :
  M ≠ 1 →
  digit_order_independent_divisibility M →
  M = 3 ∨ M = 9 := by
  sorry

#check divisibility_rule_implies_three_or_nine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_rule_implies_three_or_nine_l319_31900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_inequality_l319_31929

theorem power_sum_inequality (x y : ℝ) (h : x > y) : (2 : ℝ)^x + (2 : ℝ)^(-y) > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_inequality_l319_31929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_representable_set_infinite_l319_31949

/-- The type of expressions of the form a^3 + b^5 + c^7 + d^9 + e^11 where a, b, c, d, e are positive integers -/
def SumOfPowers (a b c d e : ℕ+) : ℕ :=
  a^3 + b^5 + c^7 + d^9 + e^11

/-- The set of all numbers that can be represented as SumOfPowers -/
def RepresentableSet : Set ℕ :=
  {n : ℕ | ∃ (a b c d e : ℕ+), SumOfPowers a b c d e = n}

/-- The complement of RepresentableSet -/
def NonRepresentableSet : Set ℕ :=
  (Set.univ : Set ℕ) \ RepresentableSet

/-- The theorem stating that NonRepresentableSet is infinite -/
theorem non_representable_set_infinite : Set.Infinite NonRepresentableSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_representable_set_infinite_l319_31949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l319_31907

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_odd : ∀ x, g (-x) = -g x
axiom second_derivative_condition : ∀ x, x < 0 → (deriv^[2] f) x * g x + f x * (deriv^[2] g) x > 0
axiom f_zero_at_neg_two : f (-2) = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x > 2}

-- State the theorem
theorem solution_set_correct : 
  {x : ℝ | f x * g x < 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l319_31907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_number_not_prime_l319_31953

/-- Represents the assignment of digits to letters in "ЗЕМЛЕТРЯСЕНИЕ" -/
def DigitAssignment := Fin 10 → Fin 10

/-- The number of times 'E' appears in "ЗЕМЛЕТРЯСЕНИЕ" -/
def e_count : Nat := 4

/-- The number of other unique letters in "ЗЕМЛЕТРЯСЕНИЕ" -/
def other_letters_count : Nat := 9

theorem earthquake_number_not_prime (assignment : DigitAssignment) 
  (h_injective : Function.Injective assignment) :
  ∃ (n : Nat), n > 3 ∧ 
    (∀ (i : Fin 10), (i.val = assignment 4 → (n.digits 10).count i = e_count) ∧ 
                     (i.val ≠ assignment 4 → (n.digits 10).count i ≤ 1)) ∧
    n % 3 = 0 ∧ ¬ Nat.Prime n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_number_not_prime_l319_31953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l319_31932

theorem triangle_angle_B (a b c : ℝ) (h1 : a = 3) (h2 : b = Real.sqrt 7) (h3 : c = 2) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  Real.arccos cos_B = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l319_31932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_of_C_range_of_x_plus_y_l319_31948

-- Define the curve C
noncomputable def C (φ : ℝ) : ℝ × ℝ := (4 * Real.cos φ, 3 * Real.sin φ)

-- Theorem for the standard equation of C
theorem standard_equation_of_C :
  ∀ (x y : ℝ), (∃ φ, C φ = (x, y)) → x^2/16 + y^2/9 = 1 := by sorry

-- Theorem for the range of x + y
theorem range_of_x_plus_y :
  ∀ (x y : ℝ), (∃ φ, C φ = (x, y)) → -5 ≤ x + y ∧ x + y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_of_C_range_of_x_plus_y_l319_31948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_finish_mowing_l319_31939

/-- Represents the lawn mowing scenario -/
structure LawnMowing where
  length : ℚ  -- length of the lawn in feet
  width : ℚ   -- width of the lawn in feet
  swath : ℚ   -- mower swath width in inches
  overlap : ℚ  -- overlap in inches
  speed : ℚ   -- mowing speed in feet per hour
  time : ℚ    -- available time in hours

/-- Calculates the time required to mow the lawn -/
def timeRequired (scenario : LawnMowing) : ℚ :=
  let effectiveSwath := (scenario.swath - scenario.overlap) / 12  -- Convert to feet
  let strips := scenario.width / effectiveSwath
  let totalDistance := strips * scenario.length
  totalDistance / scenario.speed

/-- Theorem stating that Moe cannot finish mowing the lawn in time -/
theorem cannot_finish_mowing (scenario : LawnMowing) 
  (h1 : scenario.length = 120)
  (h2 : scenario.width = 180)
  (h3 : scenario.swath = 30)
  (h4 : scenario.overlap = 6)
  (h5 : scenario.speed = 4000)
  (h6 : scenario.time = 2) :
  timeRequired scenario > scenario.time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_finish_mowing_l319_31939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l319_31903

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 0 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l319_31903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l319_31989

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0

noncomputable def hypotenuse_length (t : Triangle) : ℝ :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

def median_A_slope (t : Triangle) : ℝ := 1

def median_B_slope (t : Triangle) : ℝ := -1

noncomputable def triangle_area (t : Triangle) : ℝ :=
  let (x₁, y₁) := t.A
  let (x₂, y₂) := t.B
  let (x₃, y₃) := t.C
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₃) - (x₂ - x₃) * (y₁ - y₃))

-- Theorem statement
theorem right_triangle_area (t : Triangle) :
  is_right_triangle t ∧
  hypotenuse_length t = 50 ∧
  median_A_slope t = 1 ∧
  median_B_slope t = -1 →
  triangle_area t = 1250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l319_31989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_square_l319_31991

theorem divisors_of_square (n : ℕ+) (h : (Nat.divisors n.val).card = 4) : 
  (Nat.divisors (n^2).val).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_square_l319_31991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_pace_improvement_l319_31957

-- Define constants
noncomputable def bob_distance : ℝ := 5  -- in kilometers
noncomputable def bob_time : ℝ := 26.5  -- in minutes
noncomputable def sister_distance : ℝ := 3  -- in miles
noncomputable def sister_time : ℝ := 23.75  -- in minutes
noncomputable def km_to_mile : ℝ := 0.621371  -- conversion factor

-- Define functions
noncomputable def convert_km_to_miles (km : ℝ) : ℝ := km * km_to_mile

noncomputable def calculate_pace (distance : ℝ) (time : ℝ) : ℝ := time / distance

noncomputable def calculate_improvement_percentage (current_pace : ℝ) (target_pace : ℝ) : ℝ :=
  ((current_pace - target_pace) / current_pace) * 100

-- Theorem statement
theorem bob_pace_improvement :
  let bob_distance_miles := convert_km_to_miles bob_distance
  let bob_pace := calculate_pace bob_distance_miles bob_time
  let sister_pace := calculate_pace sister_distance sister_time
  let improvement_needed := calculate_improvement_percentage bob_pace sister_pace
  ∃ (ε : ℝ), abs (improvement_needed - 7.16) < ε ∧ ε > 0 ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_pace_improvement_l319_31957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_T_l319_31977

noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ f x = y}

def is_upper_bound (N : ℝ) : Prop := ∀ y ∈ T, y ≤ N

def is_lower_bound (n : ℝ) : Prop := ∀ y ∈ T, y ≥ n

theorem bounds_of_T :
  ∃ N n : ℝ, is_upper_bound N ∧ is_lower_bound n ∧ n ∈ T ∧ N ∉ T := by
  -- We'll use N = 3 and n = 4/3
  let N : ℝ := 3
  let n : ℝ := 4/3
  
  have h_upper : is_upper_bound N := by sorry
  have h_lower : is_lower_bound n := by sorry
  have h_n_in_T : n ∈ T := by sorry
  have h_N_not_in_T : N ∉ T := by sorry

  exact ⟨N, n, h_upper, h_lower, h_n_in_T, h_N_not_in_T⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounds_of_T_l319_31977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_correctness_l319_31914

theorem calculation_correctness : 
  (11 + (-14) + 19 - (-6) ≠ 10) ∧ 
  ((-2/3 : ℚ) - (1/5 : ℚ) + (-1/3 : ℚ) = -6/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_correctness_l319_31914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l319_31909

-- Define the quadratic function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - 2 * k * x

theorem quadratic_function_properties
  (a b : ℝ)
  (h1 : f a b (-1) = 0)
  (h2 : ∀ x, f a b x ≥ 0)
  : 
  (∃ (k : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, StrictMono (g a b k) ∨ StrictAnti (g a b k)) ↔ k ≤ -1 ∨ k ≥ 3) ∧
  (∃ (k : ℝ), IsMinOn (g a b k) (Set.Icc (-2 : ℝ) 2) (-15) ↔ k = -4 ∨ k = 6) ∧
  f a b = f 1 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l319_31909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_complement_l319_31944

theorem angle_complement (α : Real) : 
  Real.sin (28 * π / 180) = Real.cos α → α = 62 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_complement_l319_31944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l319_31994

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 3)

theorem f_minimum_value (x : ℝ) (h : x > 3) : f x ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l319_31994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_length_is_1_85_l319_31959

/-- A quadrangular pyramid with equal edge lengths -/
structure QuadrangularPyramid where
  edge_length : ℝ
  sum_of_edges : ℝ
  edges_equal : sum_of_edges = 8 * edge_length

/-- The length of one edge in a quadrangular pyramid with equal edges and sum of edges 14.8 -/
noncomputable def edge_length_of_pyramid (pyramid : QuadrangularPyramid) : ℝ :=
  pyramid.sum_of_edges / 8

/-- Theorem: The length of one edge in a quadrangular pyramid with equal edges and sum of edges 14.8 is 1.85 -/
theorem edge_length_is_1_85 (pyramid : QuadrangularPyramid) 
    (h : pyramid.sum_of_edges = 14.8) : edge_length_of_pyramid pyramid = 1.85 := by
  sorry

#check edge_length_is_1_85

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_length_is_1_85_l319_31959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_round_trip_distance_l319_31956

-- Define the positions
def river_y : ℝ := 0
def cowboy_position : ℝ × ℝ := (0, 6)
def cabin_position : ℝ × ℝ := (-12, 11)
def trading_post_position : ℝ × ℝ := (-9, 9)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the reflected cowboy position
def reflected_cowboy_position : ℝ × ℝ := (0, -6)

-- State the theorem
theorem shortest_round_trip_distance :
  let dist_to_river := cowboy_position.2 - river_y
  let dist_reflected_to_cabin := distance reflected_cowboy_position cabin_position
  let dist_cabin_to_post := distance cabin_position trading_post_position
  dist_to_river + dist_reflected_to_cabin + 2 * dist_cabin_to_post = 6 + Real.sqrt 433 + 2 * Real.sqrt 13 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_round_trip_distance_l319_31956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_fixed_circle_l319_31997

/-- The curve C: y^2 + 4x^2 = 1 -/
def C (x y : ℝ) : Prop := y^2 + 4*x^2 = 1

/-- Two points M and N on curve C -/
structure PointPair where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : C x₁ y₁
  h₂ : C x₂ y₂

/-- OM ⊥ ON condition -/
def perpendicular (p : PointPair) : Prop := p.x₁ * p.x₂ + p.y₁ * p.y₂ = 0

/-- The fixed circle: x^2 + y^2 = 1/5 -/
def fixedCircle (x y : ℝ) : Prop := x^2 + y^2 = 1/5

/-- Distance from origin to line MN -/
noncomputable def distanceToLine (p : PointPair) : ℝ :=
  Real.sqrt ((p.x₁^2 + p.y₁^2) * (p.x₂^2 + p.y₂^2)) / Real.sqrt ((p.x₁ - p.x₂)^2 + (p.y₁ - p.y₂)^2)

/-- Main theorem -/
theorem tangent_to_fixed_circle (p : PointPair) (h : perpendicular p) :
  distanceToLine p = Real.sqrt (1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_fixed_circle_l319_31997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_of_m_l319_31922

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g : ℝ → ℝ := sorry

-- Theorem for f
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1) 1 → f (-x) = -f x) →  -- f is odd on (-1, 1)
  (∀ x y, x ∈ Set.Ioo (-1) 1 → y ∈ Set.Ioo (-1) 1 → x < y → f x > f y) →  -- f is decreasing on (-1, 1)
  (f (1 - a) + f (1 - a^2) > 0) →
  a ∈ Set.Ioo 1 (Real.sqrt 2) :=
by sorry

-- Theorem for g
theorem range_of_m (g : ℝ → ℝ) (m : ℝ) :
  (∀ x, x ∈ Set.Icc (-2) 2 → g (-x) = g x) →  -- g is even on [-2, 2]
  (∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → g x > g y) →  -- g is decreasing on [0, 2]
  (g (1 - m) < g m) →
  m ∈ Set.Ico (-1) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_range_of_m_l319_31922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anniversary_book_solution_l319_31982

/-- The sum of an arithmetic sequence with n terms, first term a, and common difference d -/
noncomputable def arithmetic_sum (n : ℕ) (a d : ℝ) : ℝ := (n : ℝ) * (2 * a + (n - 1) * d) / 2

/-- Given conditions of the anniversary book problem -/
def anniversary_book_problem (first_day_books : ℝ) : Prop :=
  let n : ℕ := 19  -- number of days
  let d : ℝ := 1   -- common difference (increase in books per day)
  let total_books : ℝ := 190
  arithmetic_sum n first_day_books d = total_books

/-- Theorem stating the solution to the anniversary book problem -/
theorem anniversary_book_solution :
  ∃ (x : ℝ), anniversary_book_problem x ∧ x = 1 := by
  sorry

#check anniversary_book_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anniversary_book_solution_l319_31982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_solutions_l319_31940

noncomputable def g (x : ℝ) : ℝ := -3 * Real.cos (Real.pi * x)

theorem at_least_two_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  -1 ≤ x₁ ∧ x₁ ≤ 1 ∧ 
  -1 ≤ x₂ ∧ x₂ ≤ 1 ∧
  g (g (g x₁)) = g x₁ ∧
  g (g (g x₂)) = g x₂ := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_solutions_l319_31940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_shortest_chord_line_l319_31935

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

-- Define the line l
def line_l (m x y : ℝ) : Prop := 2*m*x - 3*m*y + x - y - 1 = 0

-- Theorem 1: The line always intersects the circle
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l m x y :=
by
  sorry

-- Theorem 2: The line with the shortest chord
theorem shortest_chord_line :
  ∀ m : ℝ, (∀ x y : ℝ, line_l m x y → (x = 3 ∧ y = 2)) →
  (∀ x y : ℝ, line_l m x y ↔ y = x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_shortest_chord_line_l319_31935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_eight_terms_l319_31993

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => sequence_a n / (1 + 3 * sequence_a n)

def sequence_inverse_a (n : ℕ) : ℚ := 1 / sequence_a n

def T (n : ℕ) : ℚ := (List.range n).map sequence_inverse_a |>.sum

theorem sum_of_first_eight_terms : T 8 = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_eight_terms_l319_31993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_is_twenty_l319_31966

/-- Represents a cricket game scenario -/
structure CricketGame where
  initial_run_rate : ℚ
  remaining_overs : ℕ
  remaining_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the number of overs played initially in a cricket game -/
def initial_overs (game : CricketGame) : ℚ :=
  (game.target_runs - game.remaining_overs * game.remaining_run_rate) / game.initial_run_rate

/-- Theorem stating that the number of overs played initially is 20 -/
theorem initial_overs_is_twenty (game : CricketGame) 
  (h1 : game.initial_run_rate = 24/5)
  (h2 : game.remaining_overs = 30)
  (h3 : game.remaining_run_rate = 6866666666666666/1000000000000000)
  (h4 : game.target_runs = 302) :
  initial_overs game = 20 := by
  sorry

def example_game : CricketGame := {
  initial_run_rate := 24/5,
  remaining_overs := 30,
  remaining_run_rate := 6866666666666666/1000000000000000,
  target_runs := 302
}

#eval initial_overs example_game

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_is_twenty_l319_31966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l319_31912

/-- Given a right triangle XYZ where XY = XZ = 5, the length of the hypotenuse YZ is 5√2. -/
theorem right_triangle_hypotenuse (X Y Z : EuclideanSpace ℝ (Fin 2)) : 
  (dist X Y = 5) →
  (dist X Z = 5) →
  (dist Y X)^2 + (dist Z X)^2 = (dist Y Z)^2 →
  dist Y Z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l319_31912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_white_equals_formula_l319_31906

/-- The number of white balls in bag A -/
def white_balls_A : ℕ := 8

/-- The number of red balls in bag A -/
def red_balls_A : ℕ := 4

/-- The number of white balls in bag B -/
def white_balls_B : ℕ := 6

/-- The number of red balls in bag B -/
def red_balls_B : ℕ := 5

/-- The total number of balls in bag A -/
def total_balls_A : ℕ := white_balls_A + red_balls_A

/-- The total number of balls in bag B -/
def total_balls_B : ℕ := white_balls_B + red_balls_B

/-- The probability of drawing exactly one white ball when randomly drawing one ball from each bag -/
noncomputable def prob_one_white : ℚ :=
  (Nat.choose white_balls_A 1 * Nat.choose red_balls_B 1 + Nat.choose red_balls_A 1 * Nat.choose white_balls_B 1) /
  (Nat.choose total_balls_A 1 * Nat.choose total_balls_B 1)

theorem prob_one_white_equals_formula :
  prob_one_white = (Nat.choose white_balls_A 1 * Nat.choose red_balls_B 1 + Nat.choose red_balls_A 1 * Nat.choose white_balls_B 1) /
                   (Nat.choose total_balls_A 1 * Nat.choose total_balls_B 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_white_equals_formula_l319_31906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approximation_l319_31954

noncomputable def selling_price_1 : ℝ := 120
noncomputable def selling_price_2 : ℝ := 225
noncomputable def selling_price_3 : ℝ := 450

noncomputable def profit_percentage_1 : ℝ := 0.25
noncomputable def profit_percentage_2 : ℝ := 0.40
noncomputable def profit_percentage_3 : ℝ := 0.20

noncomputable def cost_price (selling_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  selling_price / (1 + profit_percentage)

noncomputable def total_cost : ℝ :=
  cost_price selling_price_1 profit_percentage_1 +
  cost_price selling_price_2 profit_percentage_2 +
  cost_price selling_price_3 profit_percentage_3

theorem total_cost_approximation :
  abs (total_cost - 631.71) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approximation_l319_31954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l319_31960

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that (n+i)^6 is an integer
def is_integer_power (n : ℤ) : Prop :=
  ∃ m : ℤ, (n : ℂ) + i ^ 6 = m

-- Theorem statement
theorem unique_integer_power :
  ∃! n : ℤ, is_integer_power n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_power_l319_31960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_plus_theorem_l319_31942

/-- Custom operation ⊕ -/
noncomputable def circle_plus (a b : ℝ) : ℝ := 1/a + 1/b

/-- Main theorem -/
theorem circle_plus_theorem (a b : ℝ) (h : circle_plus a (-b) = 2) :
  (3 * a * b) / (2 * a - 2 * b) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_plus_theorem_l319_31942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_colorings_l319_31919

/-- Represents a color in the grid -/
inductive Color
| Red
| Green
| Blue

/-- Represents a position in the 3x3 grid -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- A coloring of the 3x3 grid -/
def Coloring := Position → Color

/-- Checks if two positions are adjacent -/
def adjacent (p q : Position) : Prop :=
  (p.row = q.row ∧ |p.col - q.col| = 1) ∨
  (p.col = q.col ∧ |p.row - q.row| = 1)

/-- A valid coloring satisfies the adjacency constraint -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ p q : Position, adjacent p q → c p ≠ c q

/-- The main theorem: there are exactly 3 valid colorings of the 3x3 grid -/
theorem three_valid_colorings :
  ∃! (n : Nat), (∃ (s : Finset Coloring), s.card = n ∧ ∀ c ∈ s, valid_coloring c) ∧ n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_valid_colorings_l319_31919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_true_l319_31921

theorem p_and_q_true : 
  (∃ x : ℝ, Real.sin x ≥ 1) ∧ (∀ x : ℝ, x > 0 → Real.exp x > Real.log x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_and_q_true_l319_31921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l319_31996

theorem arcsin_equation_solution (x : ℝ) : 
  Real.arcsin x + Real.arcsin (3 * x) = π / 4 → 
  (x = Real.sqrt ((39 + Real.sqrt 77) / 722) ∨ 
   x = Real.sqrt ((39 - Real.sqrt 77) / 722) ∨ 
   x = -Real.sqrt ((39 + Real.sqrt 77) / 722) ∨ 
   x = -Real.sqrt ((39 - Real.sqrt 77) / 722)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l319_31996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l319_31941

/-- A geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) : ℕ → ℝ :=
  fun n => a * q^(n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h : q ≠ 1) (h_sum : geometric_sum a q 3 = a + 3 * (a * q)) : 
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l319_31941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_l319_31901

noncomputable section

/-- Curve C in the Cartesian plane -/
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - Real.sqrt 2)^2 + p.2^2 = 2}

/-- Curve C₁ in the Cartesian plane -/
def C₁ : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = 3 - Real.sqrt 2 + 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ}

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- The transformation that maps M to P -/
def transform (M : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 + Real.sqrt 2 * (M.1 - A.1), A.2 + Real.sqrt 2 * (M.2 - A.2))

theorem no_common_points : C ∩ C₁ = ∅ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_l319_31901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_calculation_l319_31910

theorem average_score_calculation (total_students : ℕ) 
  (score_100 score_95 score_85 score_75 score_65 score_55 score_45 : ℕ) 
  (h1 : total_students = 130)
  (h2 : score_100 = 10)
  (h3 : score_95 = 15)
  (h4 : score_85 = 30)
  (h5 : score_75 = 35)
  (h6 : score_65 = 25)
  (h7 : score_55 = 10)
  (h8 : score_45 = 5)
  (h9 : total_students = score_100 + score_95 + score_85 + score_75 + score_65 + score_55 + score_45) :
  let total_score := 100 * score_100 + 95 * score_95 + 85 * score_85 + 75 * score_75 + 
                     65 * score_65 + 55 * score_55 + 45 * score_45
  ∃ (average : ℚ), abs (average - (total_score : ℚ) / total_students) < 0.01 ∧ 
                   76 < average ∧ average < 77 := by
  sorry

#eval 10000 / 130

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_score_calculation_l319_31910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_three_a_minus_c_is_zero_l319_31961

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- Vector m defined as (2sin(A+C), -√3) -/
noncomputable def m (t : AcuteTriangle) : ℝ × ℝ := (2 * Real.sin (t.A + t.C), -Real.sqrt 3)

/-- Vector n defined as (1-2cos²(B/2), cos(2B)) -/
noncomputable def n (t : AcuteTriangle) : ℝ × ℝ := (1 - 2 * (Real.cos (t.B / 2))^2, Real.cos (2 * t.B))

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem angle_B_is_pi_over_three (t : AcuteTriangle) 
  (h : dot_product (m t) (n t) = 0) : t.B = π/3 := by sorry

theorem a_minus_c_is_zero (t : AcuteTriangle) 
  (h1 : Real.sin t.A * Real.sin t.C = (Real.sin t.B)^2) 
  (h2 : t.B = π/3) : t.a - t.c = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_three_a_minus_c_is_zero_l319_31961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_product_l319_31971

/-- The angle of intersection between two lines in radians -/
noncomputable def intersection_angle : ℝ := Real.pi / 3

/-- The relationship between the slopes of the two lines -/
def slope_relationship (m₁ m₂ : ℝ) : Prop := m₂ = 3 * m₁

/-- The tangent of the angle between two lines with slopes m₁ and m₂ -/
noncomputable def tan_angle (m₁ m₂ : ℝ) : ℝ := |m₂ - m₁| / (1 + m₁ * m₂)

/-- The product of the slopes -/
def slope_product (m₁ m₂ : ℝ) : ℝ := m₁ * m₂

theorem max_slope_product :
  ∀ m₁ m₂ : ℝ,
    slope_relationship m₁ m₂ →
    tan_angle m₁ m₂ = Real.tan intersection_angle →
    slope_product m₁ m₂ ≤ 3/2 := by
  sorry

#check max_slope_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_product_l319_31971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_theorem_l319_31976

-- Define the points of the hexagon
def p1 : ℝ × ℝ := (0, 2)
def p2 : ℝ × ℝ := (1, 3)
def p3 : ℝ × ℝ := (2, 3)
def p4 : ℝ × ℝ := (2, 2)
def p5 : ℝ × ℝ := (3, 0)
def p6 : ℝ × ℝ := (2, -1)

-- Function to calculate distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter of the hexagon
noncomputable def hexagon_perimeter : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p4 +
  distance p4 p5 + distance p5 p6 + distance p6 p1

-- Theorem statement
theorem hexagon_perimeter_theorem :
  ∃ (a b c : ℤ),
    hexagon_perimeter = 2 + 2 * Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 13 ∧
    hexagon_perimeter = a + b * Real.sqrt 2 + c * Real.sqrt 5 ∧
    a + b + c = 5 := by
  sorry

#eval "Hexagon perimeter theorem is stated."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_theorem_l319_31976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vector_sum_magnitude_l319_31923

-- Define the circle C
def is_on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Define the line l
def is_on_line (x y : ℝ) : Prop := x + y = 1

-- Define the vector OP + OQ
def vector_sum (px py qx qy : ℝ) : ℝ × ℝ := (px + qx, py + qy)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem min_vector_sum_magnitude :
  ∃ (min_val : ℝ), min_val = (5 * Real.sqrt 2 - 2) / 2 ∧
  ∀ (px py qx qy : ℝ),
    is_on_circle px py →
    is_on_line qx qy →
    magnitude (vector_sum px py qx qy) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_vector_sum_magnitude_l319_31923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l319_31978

-- Define the circle
def circle_set (a : ℝ) : Set (ℝ × ℝ) := {(x, y) | (x - a)^2 + y^2 = 1}

-- Define the line (axis of symmetry)
def axis_of_symmetry : Set (ℝ × ℝ) := {(x, y) | 2*x + y - 1 = 0}

-- Define reflection across a line
noncomputable def reflect_line (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_symmetry (a : ℝ) :
  (∀ (p : ℝ × ℝ), p ∈ circle_set a ↔ reflect_line axis_of_symmetry p ∈ circle_set a) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_l319_31978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_6_or_7_base_9_l319_31943

-- Define the set of integers from 1 to 729
def S : Set Nat := {n : Nat | 1 ≤ n ∧ n ≤ 729}

-- Define a function that checks if a number contains 6 or 7 in base 9
def contains_6_or_7_base_9 (n : Nat) : Bool := 
  let digits := Nat.digits 9 n
  digits.any (λ d => d = 6 ∨ d = 7)

-- Define the set of numbers in S that contain 6 or 7 in base 9
def T : Set Nat := {n ∈ S | contains_6_or_7_base_9 n = true}

-- The theorem to prove
theorem count_numbers_with_6_or_7_base_9 : Finset.card (Finset.filter (λ n => contains_6_or_7_base_9 n = true) (Finset.range 729)) = 386 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_6_or_7_base_9_l319_31943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_area_ratio_l319_31975

/-- The area of a circle with radius r -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of a ring between two concentric circles with radii r1 and r2 -/
noncomputable def ring_area (r1 r2 : ℝ) : ℝ := circle_area r2 - circle_area r1

theorem concentric_circles_area_ratio :
  let radii : List ℝ := [1, 3, 5, 7, 9]
  let black_areas : List ℝ := [circle_area 1, ring_area 3 5, ring_area 7 9]
  let white_areas : List ℝ := [ring_area 1 3, ring_area 5 7]
  (black_areas.sum / white_areas.sum) = 49 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_area_ratio_l319_31975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l319_31979

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 3*x + 4

-- State the theorem
theorem f_minimum_value :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l319_31979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_toy_production_time_l319_31945

/-- The time taken to make one toy, given the total time and number of toys made -/
noncomputable def time_per_toy (total_time : ℝ) (num_toys : ℝ) : ℝ :=
  total_time / num_toys

/-- Theorem stating that it takes 2 hours to make one toy -/
theorem worker_toy_production_time :
  time_per_toy 120 60 = 2 := by
  -- Unfold the definition of time_per_toy
  unfold time_per_toy
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_toy_production_time_l319_31945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_km_to_miles_approx_l319_31930

/-- Conversion factor from kilometers to miles -/
noncomputable def km_to_miles : ℝ := 5 / 8

/-- Given approximation: 8 km ≈ 5 miles -/
axiom km_miles_approx : ∀ (ε : ℝ), ε > 0 → |km_to_miles - 5 / 8| < ε

/-- The value to be converted: 1.2 km -/
def kilometers : ℝ := 1.2

/-- The target approximation in miles -/
def target_miles : ℝ := 0.75

/-- Theorem stating that 1.2 km is approximately 0.75 miles -/
theorem km_to_miles_approx : 
  ∀ (ε : ℝ), ε > 0 → |kilometers * km_to_miles - target_miles| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_km_to_miles_approx_l319_31930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_to_plane_perpendicular_transitivity_l319_31911

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem parallel_line_to_plane 
  (α β : Plane) (m : Line) 
  (h1 : parallel_plane α β) (h2 : subset m α) : 
  parallel_line_plane m β :=
sorry

-- Theorem 2
theorem perpendicular_transitivity 
  (α β : Plane) (m n : Line) 
  (h1 : perpendicular n α) (h2 : perpendicular n β) (h3 : perpendicular m α) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_to_plane_perpendicular_transitivity_l319_31911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_product_l319_31928

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 3 then |Real.log x / Real.log 3|
  else if 3 ≤ x ∧ x ≤ 9 then -Real.cos (Real.pi/3 * x)
  else 0  -- undefined for other values, set to 0 for completeness

theorem range_of_product (x₁ x₂ x₃ x₄ : ℝ) :
  (0 < x₁ ∧ x₁ < 3) →
  (3 ≤ x₃ ∧ x₃ ≤ 9) →
  (3 ≤ x₄ ∧ x₄ ≤ 9) →
  f 1 = x₂ ∧ x₂ = f x₃ ∧ f x₃ = f x₄ →
  ∃ y, 27 < y ∧ y < 135/4 ∧ y = x₁ * 2 * x₃ * x₄ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_product_l319_31928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_force_l319_31938

/-- Work done by a force on a particle moving between two points -/
theorem work_done_by_force (A B F : ℝ × ℝ × ℝ) : 
  A = (1, 1, -2) → 
  B = (3, 4, -2 + Real.sqrt 2) → 
  F = (2, 2, 2 * Real.sqrt 2) → 
  let S := (B.fst - A.fst, B.snd.fst - A.snd.fst, B.snd.snd - A.snd.snd)
  let W := F.fst * S.fst + F.snd.fst * S.snd.fst + F.snd.snd * S.snd.snd
  W = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_force_l319_31938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_solution_set_f_greater_than_three_l319_31973

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - 1)

-- Theorem 1: If f is an odd function, then a = 1
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 1 := by sorry

-- Theorem 2: The solution set of f(x) > 3 when a = 1
theorem solution_set_f_greater_than_three :
  {x : ℝ | f 1 x > 3} = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_solution_set_f_greater_than_three_l319_31973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_comparisons_l319_31962

-- Define the theorem
theorem logarithm_comparisons :
  -- (1)
  (Real.log 0.8 / Real.log 6 < Real.log 9.1 / Real.log 6) ∧
  -- (2)
  (Real.log 7 / Real.log 0.1 > Real.log 9 / Real.log 0.1) ∧
  -- (3)
  (Real.log 5 / Real.log 0.1 < Real.log 5 / Real.log 2.3) ∧
  -- (4)
  (∀ a : ℝ, a > 0 → a ≠ 1 →
    ((0 < a ∧ a < 1 → Real.log 4 / Real.log a > Real.log 6 / Real.log a) ∧
     (a > 1 → Real.log 4 / Real.log a < Real.log 6 / Real.log a))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_comparisons_l319_31962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_theorem_l319_31924

/-- The cost per foot of a fence around a square plot -/
noncomputable def cost_per_foot (area : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (4 * Real.sqrt area)

/-- Theorem: The cost per foot of a fence around a square plot with area 144 sq ft
    and total cost Rs. 2784 is Rs. 58 -/
theorem fence_cost_theorem (area : ℝ) (total_cost : ℝ) 
    (h1 : area = 144) (h2 : total_cost = 2784) : 
    cost_per_foot area total_cost = 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_theorem_l319_31924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_smaller_cone_altitude_l319_31952

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  upper_base_area : ℝ
  lower_base_area : ℝ

/-- Calculates the altitude of the smaller cone removed from a frustum -/
noncomputable def smaller_cone_altitude (f : Frustum) : ℝ :=
  f.altitude / 2

/-- Theorem: The altitude of the smaller cone removed from the given frustum is 20 cm -/
theorem frustum_smaller_cone_altitude 
  (f : Frustum) 
  (h1 : f.altitude = 40)
  (h2 : f.upper_base_area = 36 * Real.pi)
  (h3 : f.lower_base_area = 144 * Real.pi) :
  smaller_cone_altitude f = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_smaller_cone_altitude_l319_31952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_transformation_distance_l319_31980

noncomputable section

def original_radius : ℝ := 3
def original_center : ℝ × ℝ := (3, 3)
def transformed_radius : ℝ := 4.5
def dilated_center : ℝ × ℝ := (6, 9)
def rotation_angle : ℝ := Real.pi / 4  -- 45 degrees in radians
def point_p : ℝ × ℝ := (1, 1)

def dilation_factor : ℝ := transformed_radius / original_radius

-- Function to apply dilation
def apply_dilation (p : ℝ × ℝ) : ℝ × ℝ :=
  (dilation_factor * p.1, dilation_factor * p.2)

-- Function to apply rotation
def apply_rotation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos rotation_angle - p.2 * Real.sin rotation_angle,
   p.1 * Real.sin rotation_angle + p.2 * Real.cos rotation_angle)

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem composite_transformation_distance :
  let transformed_p := apply_rotation (apply_dilation point_p)
  distance point_p transformed_p = Real.sqrt (1 + (1 - 1.5 * Real.sqrt 2)^2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_transformation_distance_l319_31980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l319_31951

/-- Ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line l with equation y = kx + m -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Circle passing through three points -/
def circle_through (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

theorem ellipse_intersection_fixed_point (k m : ℝ) (hk : k ≠ 0)
  (x1 y1 x2 y2 : ℝ) (hM : ellipse_C x1 y1) (hN : ellipse_C x2 y2)
  (hl1 : line_l k m x1 y1) (hl2 : line_l k m x2 y2)
  (hdistinct : (x1, y1) ≠ (x2, y2))
  (hnot_vertex1 : (x1, y1) ≠ (-2, 0)) (hnot_vertex2 : (x1, y1) ≠ (2, 0))
  (hnot_vertex3 : (x2, y2) ≠ (-2, 0)) (hnot_vertex4 : (x2, y2) ≠ (2, 0))
  (hcircle : circle_through x1 y1 x2 y2 2 0) :
  line_l k m (6/5) 0 := by
  sorry

#check ellipse_intersection_fixed_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_fixed_point_l319_31951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_cost_l319_31933

/-- Represents the cost of a pizza order --/
structure PizzaOrder where
  base_price : ℚ
  premium_topping_price : ℚ
  regular_topping_price : ℚ
  extra_cheese_price : ℚ
  tip : ℚ
  pepperoni_pizza_count : ℕ
  sausage_onion_pizza_count : ℕ
  olive_mushroom_pepper_pizza_count : ℕ
  veggie_pizza_count : ℕ

/-- Calculates the total cost of a pizza order --/
def total_cost (order : PizzaOrder) : ℚ :=
  let pepperoni_cost := order.base_price + order.premium_topping_price
  let sausage_onion_cost := order.base_price + order.premium_topping_price + order.regular_topping_price
  let olive_mushroom_pepper_cost := order.base_price + 3 * order.regular_topping_price
  let veggie_cost := order.base_price + 2 * order.regular_topping_price + order.extra_cheese_price
  (pepperoni_cost * order.pepperoni_pizza_count +
   sausage_onion_cost * order.sausage_onion_pizza_count +
   olive_mushroom_pepper_cost * order.olive_mushroom_pepper_pizza_count +
   veggie_cost * order.veggie_pizza_count) +
  order.tip

/-- The theorem stating the total cost of the specific pizza order --/
theorem pizza_order_cost :
  ∃ (order : PizzaOrder),
    order.base_price = 10 ∧
    order.premium_topping_price = 3/2 ∧
    order.regular_topping_price = 1 ∧
    order.extra_cheese_price = 2 ∧
    order.tip = 5 ∧
    order.pepperoni_pizza_count = 1 ∧
    order.sausage_onion_pizza_count = 1 ∧
    order.olive_mushroom_pepper_pizza_count = 1 ∧
    order.veggie_pizza_count = 1 ∧
    total_cost order = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_cost_l319_31933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoe_candy_bar_sales_l319_31969

/-- The number of candy bars Zoe needs to sell for her trip -/
def candy_bars_to_sell (trip_cost : ℚ) (grandma_contribution : ℚ) (earnings_per_bar : ℚ) : ℕ :=
  (((trip_cost - grandma_contribution) / earnings_per_bar).ceil).toNat

/-- Theorem stating the number of candy bars Zoe needs to sell -/
theorem zoe_candy_bar_sales : candy_bars_to_sell 485 250 (5/4) = 188 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoe_candy_bar_sales_l319_31969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_rate_l319_31984

/-- Calculates the annual percentage increase rate given initial population, 
    final population, and number of years. -/
noncomputable def annual_percentage_increase (initial_population : ℝ) 
                               (final_population : ℝ) 
                               (years : ℝ) : ℝ :=
  ((final_population / initial_population) ^ (1 / years) - 1) * 100

/-- Theorem stating that the annual percentage increase for the given population data
    is approximately 18.03% -/
theorem population_increase_rate : 
  let initial_population : ℝ := 7800
  let final_population : ℝ := 10860.72
  let years : ℝ := 2
  let rate := annual_percentage_increase initial_population final_population years
  abs (rate - 18.03) < 0.01 := by
  sorry

-- Use #eval only for computable functions
def approximate_rate : Float :=
  ((10860.72 / 7800) ^ (1 / 2) - 1) * 100

#eval approximate_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_rate_l319_31984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_third_in_second_quadrant_l319_31981

-- Define the concept of quadrants
def in_second_quadrant (θ : ℝ) : Prop := Real.pi / 2 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi

-- State the theorem
theorem angle_third_in_second_quadrant (α : ℝ) 
  (h1 : in_second_quadrant α) 
  (h2 : |Real.cos (α / 3)| = -Real.cos (α / 3)) : 
  in_second_quadrant (α / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_third_in_second_quadrant_l319_31981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l319_31970

-- Define the basic structures
structure Line
structure Plane
structure Point

-- Define the relationships
def parallel (a b : Line) : Prop := sorry
def parallelToPlane (l : Line) (p : Plane) : Prop := sorry
def formEqualAngles (a b : Line) (p : Plane) : Prop := sorry
def perpendicularProjection (a b : Line) (p : Plane) : Prop := sorry
def perpendicular (a b : Line) : Prop := sorry
def distanceToPlane (A : Point) (p : Plane) : ℝ := sorry

-- Define the propositions
def proposition1 (a b : Line) (α : Plane) : Prop :=
  formEqualAngles a b α → parallel a b

def proposition2 (a b : Line) (α : Plane) : Prop :=
  parallel a b → parallelToPlane a α → parallelToPlane b α

def proposition3 (a b : Line) (α : Plane) : Prop :=
  perpendicularProjection a b α → perpendicular a b

def proposition4 (A B : Point) (α : Plane) : Prop :=
  distanceToPlane A α = distanceToPlane B α → parallelToPlane (Line.mk) α

-- Theorem stating that all propositions are false
theorem all_propositions_false :
  (∃ a b α, ¬proposition1 a b α) ∧
  (∃ a b α, ¬proposition2 a b α) ∧
  (∃ a b α, ¬proposition3 a b α) ∧
  (∃ A B α, ¬proposition4 A B α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l319_31970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_specific_point_l319_31913

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_specific_point :
  cylindrical_to_rectangular 5 (Real.pi / 4) 2 = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) := by
  unfold cylindrical_to_rectangular
  simp [Real.cos_pi_div_four, Real.sin_pi_div_four]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_specific_point_l319_31913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l319_31918

/-- A function that checks if a two-digit number is divisible by 17 or 29 -/
def isDivisibleBy17Or29 (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 29 = 0)

/-- A function that represents the string of digits -/
def digitString : ℕ → ℕ := sorry

/-- The length of the digit string -/
def stringLength : ℕ := 2024

/-- The condition that any two consecutive digits form a number divisible by 17 or 29 -/
def consecutiveDigitsDivisible : Prop :=
  ∀ i, i < stringLength - 1 → isDivisibleBy17Or29 (digitString i * 10 + digitString (i + 1))

/-- The theorem stating that the largest possible last digit is 8 -/
theorem largest_last_digit :
  digitString 0 = 1 →
  consecutiveDigitsDivisible →
  ∀ n, n ≤ 9 → digitString (stringLength - 1) ≤ n → n ≤ 8 :=
by
  sorry

#check largest_last_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_last_digit_l319_31918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_43_between_6_and_7_l319_31904

theorem sqrt_43_between_6_and_7 :
  (∀ x ∈ ({Real.sqrt 28, Real.sqrt 35, Real.sqrt 58} : Set ℝ), x < 6 ∨ x > 7) ∧
  (6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_43_between_6_and_7_l319_31904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_fraction_is_one_fourth_l319_31925

/-- Represents the properties of a tank with a mixture of water and sodium chloride -/
structure Tank where
  capacity : ℝ
  initialSodiumChlorideRatio : ℝ
  waterEvaporationRate : ℝ
  elapsedTime : ℝ
  finalWaterConcentration : ℝ

/-- Calculates the initial mixture volume as a fraction of the tank's capacity -/
noncomputable def initialMixtureFraction (t : Tank) : ℝ :=
  let initialVolume := t.waterEvaporationRate * t.elapsedTime / (t.finalWaterConcentration - t.initialSodiumChlorideRatio)
  initialVolume / t.capacity

/-- Theorem stating that for the given conditions, the initial mixture fraction is 1/4 -/
theorem initial_mixture_fraction_is_one_fourth :
  let t : Tank := {
    capacity := 24,
    initialSodiumChlorideRatio := 0.3,
    waterEvaporationRate := 0.4,
    elapsedTime := 6,
    finalWaterConcentration := 0.5
  }
  initialMixtureFraction t = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_mixture_fraction_is_one_fourth_l319_31925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recruitment_probabilities_l319_31955

-- Define the probabilities as real numbers between 0 and 1
variable (P_A P_B P_C : ℝ)
variable (h_A : 0 ≤ P_A ∧ P_A ≤ 1)
variable (h_B : 0 ≤ P_B ∧ P_B ≤ 1)
variable (h_C : 0 ≤ P_C ∧ P_C ≤ 1)

-- Given conditions
variable (h1 : P_A = 2/3)
variable (h2 : (1 - P_A) * (1 - P_B) = 1/12)
variable (h3 : P_B * P_C = 3/8)

-- Independence assumption (simplified)
variable (h4 : P_A * P_B = P_A * P_B ∧ P_A * P_C = P_A * P_C ∧ P_B * P_C = P_B * P_C)

-- Theorem to prove
theorem recruitment_probabilities :
  P_B = 3/4 ∧ P_C = 1/2 ∧
  P_A * P_B * P_C + (1 - P_A) * P_B * P_C + P_A * (1 - P_B) * P_C + P_A * P_B * (1 - P_C) = 17/24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recruitment_probabilities_l319_31955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l319_31968

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 3) / (abs (x + 1) - 5)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc 3 4 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l319_31968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l319_31937

-- Define the universal set U
def U : Set ℝ := Set.Icc 0 1

-- Define the set A
def A : Set ℝ := {1}

-- State the theorem
theorem complement_of_A_in_U : Set.diff U A = Set.Ico 0 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l319_31937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_exists_l319_31974

theorem circular_arrangement_exists :
  ∃ (p : Fin 2015 → Fin 2015), Function.Bijective p ∧
    ∀ i : Fin 2015,
      (((p i).val + 1 + (p (i.succ)).val + 1) % 4 = 0) ∨
      (((p i).val + 1 + (p (i.succ)).val + 1) % 7 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_arrangement_exists_l319_31974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_construction_condition_l319_31936

/-- A rhombus with side length a can be constructed if and only if
    the given difference between the diagonals d satisfies d < 2a. -/
theorem rhombus_construction_condition (a d : ℝ) (h_pos : a > 0) :
  ∃ (e f : ℝ), e > f ∧ e - f = d ∧ a^2 = (e/2)^2 + (f/2)^2 ↔ d < 2*a :=
by
  sorry

#check rhombus_construction_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_construction_condition_l319_31936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_polar_coords_l319_31926

/-- The polar equation of the circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ + 2 * Real.sin θ

/-- The center of the circle in rectangular coordinates -/
def center : ℝ × ℝ := (1, 1)

/-- Conversion from rectangular to polar coordinates -/
noncomputable def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt (x^2 + y^2), Real.arctan (y / x))

theorem center_polar_coords :
  rect_to_polar center.1 center.2 = (Real.sqrt 2, π/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_polar_coords_l319_31926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_evaluation_l319_31927

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then -2 * x^2 + 1 else 2 * x - 3

-- State the theorem
theorem nested_g_evaluation :
  g (g (g (g (g 2)))) = -1003574820536187457 := by
  -- Implement the proof steps here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_evaluation_l319_31927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_apartment_size_is_720_l319_31995

/-- The cost per square foot for apartment rentals in Ridgewood -/
noncomputable def cost_per_sqft : ℚ := 6/5

/-- Michael's monthly budget for rent -/
noncomputable def budget : ℚ := 864

/-- The maximum apartment size that can be rented given the cost per square foot and budget -/
noncomputable def max_apartment_size : ℚ := budget / cost_per_sqft

theorem max_apartment_size_is_720 :
  max_apartment_size = 720 :=
by
  -- Unfold the definitions
  unfold max_apartment_size budget cost_per_sqft
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_apartment_size_is_720_l319_31995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_positive_l319_31998

theorem sine_sum_positive (α : Real) (h : 0 < α ∧ α < Real.pi) : 
  Real.sin α + (1/2) * Real.sin (2*α) + (1/3) * Real.sin (3*α) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_positive_l319_31998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l319_31908

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - x else (-x)^2 - (-x)

-- State the theorem
theorem solution_set_of_inequality (hf : ∀ x, f (-x) = f x) :
  {x : ℝ | f (x + 2) < 6} = {x : ℝ | -5 < x ∧ x < 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l319_31908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l319_31905

theorem angle_in_fourth_quadrant (θ : ℝ) : 
  Real.sin θ < Real.cos θ → Real.sin θ * Real.cos θ < 0 → 
  θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l319_31905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l319_31985

-- Define the function (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := (x + 1)^0 / Real.sqrt (abs x - x)

-- Define the domain
def domain : Set ℝ := {x | x < 0 ∧ x ≠ -1}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, x ∈ domain ↔ (∃ y : ℝ, f x = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l319_31985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l319_31967

noncomputable def plane1 (x y z : ℝ) : Prop := x + 2*y - 2*z + 1 = 0

noncomputable def plane2 (x y z : ℝ) : Prop := 2*x + 5*y - 4*z + 8 = 0

noncomputable def distance_between_planes : ℝ := 2 * Real.sqrt 5 / 5

theorem distance_between_given_planes :
  ∃ (d : ℝ), d = distance_between_planes ∧
  ∀ (p1 p2 : ℝ × ℝ × ℝ), 
    plane1 p1.1 p1.2.1 p1.2.2 → 
    plane2 p2.1 p2.2.1 p2.2.2 → 
    d ≤ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2.1 - p2.2.1)^2 + (p1.2.2 - p2.2.2)^2) :=
by
  sorry

#check distance_between_given_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l319_31967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l319_31931

/-- The vertex of a quadratic equation ax^2 + bx + c -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let x := -b / (2 * a)
  (x, a * x^2 + b * x + c)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_vertices :
  let C := vertex 1 (-6) 5
  let D := vertex 1 2 4
  distance C D = Real.sqrt 65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l319_31931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_correct_propositions_l319_31915

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := sorry

def is_geometric_sequence (a c b : ℝ) : Prop := sorry

theorem count_correct_propositions : 
  let prop1 := (¬ ∃ x : ℝ, x^3 + 1 < 0) ↔ (∀ x : ℝ, x^3 + 1 > 0)
  let prop2 := ∀ a b : ℝ, a > 0 → b > 0 → 
    (∃ F A B : ℝ × ℝ, 
      B = (0, b) ∧ 
      (A.1 - B.1) * (F.1 - B.1) + (A.2 - B.2) * (F.2 - B.2) = 0 → 
      hyperbola_eccentricity a b = (Real.sqrt 5 + 1) / 2)
  let prop3 := ∀ A B C a b c : ℝ,
    Real.cos (2*B) + Real.cos B + Real.cos (A-C) = 1 →
    is_geometric_sequence a c b
  let prop4 := ∀ a b : ℝ × ℝ, 
    ‖a‖ = 1 → ‖b‖ = 1 → a.1 * b.1 + a.2 * b.2 = -1/2 →
    (∃ lambda : ℝ, ∀ t : ℝ × ℝ, 
      t = (lambda * a.1 + b.1, lambda * a.2 + b.2) →
      t.1 * (a.1 - 2*b.1) + t.2 * (a.2 - 2*b.2) = 0) ↔
    (5 : ℝ)/4 = 5/4
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_correct_propositions_l319_31915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_three_halves_f_composition_solution_set_characterization_l319_31934

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1/2 then x^2 - 4*x else Real.log (2*x + 1) / Real.log (1/2)

-- Theorem statements
theorem f_at_three_halves : f (3/2) = -2 := by sorry

theorem f_composition : f (f (1/2)) = 5 := by sorry

-- Define the solution set
def solution_set := {x : ℝ | f x > -3}

theorem solution_set_characterization : 
  solution_set = {x : ℝ | x < 7/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_three_halves_f_composition_solution_set_characterization_l319_31934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tetrahedron_possible_l319_31917

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- Checks if a triangle is obtuse-angled -/
def is_obtuse_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

/-- The set of available triangles -/
noncomputable def available_triangles : List Triangle :=
  [
    { a := 3, b := 4, c := 5 },
    { a := 3, b := 4, c := 5 },
    { a := 4, b := 5, c := Real.sqrt 41 },
    { a := 4, b := 5, c := Real.sqrt 41 },
    { a := 4, b := 5, c := Real.sqrt 41 },
    { a := 4, b := 5, c := Real.sqrt 41 },
    { a := (5/6) * Real.sqrt 2, b := 4, c := 5 },
    { a := (5/6) * Real.sqrt 2, b := 4, c := 5 },
    { a := (5/6) * Real.sqrt 2, b := 4, c := 5 },
    { a := (5/6) * Real.sqrt 2, b := 4, c := 5 },
    { a := (5/6) * Real.sqrt 2, b := 4, c := 5 },
    { a := (5/6) * Real.sqrt 2, b := 4, c := 5 }
  ]

/-- Checks if a set of four triangles can form a tetrahedron -/
def can_form_tetrahedron (t1 t2 t3 t4 : Triangle) : Prop :=
  sorry -- Definition of conditions for forming a tetrahedron

/-- Theorem stating that no tetrahedron can be formed from the available triangles -/
theorem no_tetrahedron_possible : ∀ (t1 t2 t3 t4 : Triangle), 
  t1 ∈ available_triangles → 
  t2 ∈ available_triangles → 
  t3 ∈ available_triangles → 
  t4 ∈ available_triangles → 
  ¬(can_form_tetrahedron t1 t2 t3 t4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tetrahedron_possible_l319_31917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_solution_alcohol_mixture_proof_l319_31987

/-- Calculates the volume of solution y needed to achieve a desired alcohol concentration --/
noncomputable def volume_y (x_volume : ℝ) (x_concentration : ℝ) (y_concentration : ℝ) (desired_concentration : ℝ) : ℝ :=
  (desired_concentration * x_volume - x_concentration * x_volume) / (y_concentration - desired_concentration)

/-- Verifies if the calculated volume of solution y results in the desired alcohol concentration --/
theorem verify_solution (x_volume : ℝ) (x_concentration : ℝ) (y_concentration : ℝ) (desired_concentration : ℝ) :
  let y_volume := volume_y x_volume x_concentration y_concentration desired_concentration
  (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = desired_concentration :=
by
  sorry

/-- Proves that adding 450 mL of 30% solution to 300 mL of 10% solution results in 22% solution --/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let desired_concentration : ℝ := 0.22
  let y_volume := volume_y x_volume x_concentration y_concentration desired_concentration
  y_volume = 450 ∧
  (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = desired_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_verify_solution_alcohol_mixture_proof_l319_31987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_trajectory_l319_31965

/-- Trajectory of a particle under gravity and perpendicular force -/
theorem particle_trajectory
  (g k : ℝ) -- gravity and force constant
  (k_pos : k > 0) -- ensure k is positive
  (x y : ℝ → ℝ) -- position functions
  (v : ℝ → ℝ × ℝ) -- velocity function
  (h_initial : x 0 = 0 ∧ y 0 = 0 ∧ v 0 = (0, 0)) -- initial conditions
  (h_eom_x : ∀ t, ((deriv^[2] x) t) = k * (deriv y t)) -- equation of motion for x
  (h_eom_y : ∀ t, ((deriv^[2] y) t) = g - k * (deriv x t)) -- equation of motion for y
  (h_perp : ∀ t, (deriv x t) * (deriv x t) + (deriv y t) * (deriv y t) ≠ 0 →
    (deriv x t) * k * (deriv y t) - (deriv y t) * k * (deriv x t) = 0) -- force perpendicular to velocity
  : ∀ t, x t = (g / k^2) * (Real.sin (k*t) - k*t) ∧
           y t = (g / k^2) * (1 - Real.cos (k*t)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_trajectory_l319_31965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l319_31972

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_property :
  ∀ d : ℝ, d ≠ 0 →
  let a := arithmetic_sequence (-2) d
  (a 15 = a 3 * a 5) →
  ((∀ n : ℕ, a n = 3 * n - 5) ∨ (∀ n : ℕ, a n = (1/4) * n - 9/4)) ∧
  ((∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, m > n → sum_arithmetic_sequence (-2) 3 m ≥ arithmetic_sequence (-2) 3 m) ∨
   (∃ n : ℕ, n = 17 ∧ ∀ m : ℕ, m > n → sum_arithmetic_sequence (-2) (1/4) m ≥ arithmetic_sequence (-2) (1/4) m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l319_31972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l319_31916

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x/2 + 1/(2*a)| + |x/2 - a/2|

theorem function_properties (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≥ 1) ∧
  (f a 6 < 5 → 1 + Real.sqrt 2 < a ∧ a < 5 + 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l319_31916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l319_31958

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The octagon formed by joining midpoints of consecutive sides of a regular octagon -/
noncomputable def midpointOctagon (o : RegularOctagon) : RegularOctagon where
  vertices := λ i => ((o.vertices i).1 + (o.vertices ((i + 1) % 8)).1, 
                      (o.vertices i).2 + (o.vertices ((i + 1) % 8)).2) / 2

/-- The area of a RegularOctagon -/
noncomputable def area (o : RegularOctagon) : ℝ := sorry

/-- Theorem: The area of the midpoint octagon is 3/4 of the area of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (3/4) * area o := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_octagon_area_ratio_l319_31958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_condition_l319_31988

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

theorem function_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  a ∈ Set.Icc (-3) (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_condition_l319_31988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l319_31983

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 1/2)^0 + |x^2 - 1| / Real.sqrt (x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > -2 ∧ x ≠ 1/2}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | IsRegular (f x)} = domain_f :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l319_31983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_10ring_is_02_l319_31963

/-- Represents the outcome of a shot --/
inductive ShotOutcome
  | Ring10
  | Ring9
  | Ring8
  | Miss

/-- Represents the results of a shooting practice --/
structure ShootingPractice where
  total_shots : Nat
  ring10_hits : Nat
  ring9_hits : Nat
  ring8_hits : Nat
  misses : Nat

/-- Calculates the probability of hitting the 10-ring in a single shot --/
noncomputable def probability_10ring (practice : ShootingPractice) : Real :=
  (practice.ring10_hits : Real) / (practice.total_shots : Real)

/-- The theorem to be proved --/
theorem probability_10ring_is_02 (practice : ShootingPractice)
  (h1 : practice.total_shots = 10)
  (h2 : practice.ring10_hits = 2)
  (h3 : practice.ring9_hits = 3)
  (h4 : practice.ring8_hits = 4)
  (h5 : practice.misses = 1)
  : probability_10ring practice = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_10ring_is_02_l319_31963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l319_31990

theorem triangle_area_inequality (a b c : ℝ) (l m n : ℝ) (h_pos : l > 0 ∧ m > 0 ∧ n > 0) :
  let Δ := Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4
  Δ ≤ (l * a^2 + m * b^2 + n * c^2) * (m * l + l * n + n * m) / (12 * Real.sqrt 3 * m * n * l) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_inequality_l319_31990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l319_31902

-- Define the points
variable (A B C D E F N₁ N₂ N₃ : ℝ × ℝ)

-- Define the ratios
noncomputable def ratio_CD : ℝ := 1/4
noncomputable def ratio_AE : ℝ := 1/4
noncomputable def ratio_BF : ℝ := 1/4

noncomputable def ratio_AN₂ : ℝ := 4/8
noncomputable def ratio_N₂N₁ : ℝ := 1/8
noncomputable def ratio_N₁D : ℝ := 3/8

-- Define the areas
noncomputable def area_ABC : ℝ := sorry
noncomputable def area_N₁N₂N₃ : ℝ := sorry

-- State the theorem
theorem area_ratio :
  area_N₁N₂N₃ = (11/32) * area_ABC := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l319_31902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l319_31999

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 4*x + 4) / 8

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -2

/-- A point is symmetric about a line if the line is the perpendicular bisector of the segment joining the point and its reflection -/
def IsSymmetricAbout (p : ℝ × ℝ) (l : ℝ × ℝ) : Prop :=
  let (px, py) := p
  let (lx, ly) := l
  (px - lx)^2 + (py - ly)^2 = (px - lx)^2 + (-py - ly)^2

/-- Theorem: The directrix of the given parabola is y = -2 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ y₀ : ℝ, directrix y₀ ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → IsSymmetricAbout p (0, y₀)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l319_31999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_circle_theorem_l319_31950

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A circle with center (0, 0) and diameter 1 -/
def unitDiameterCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 ≤ (1/4)

theorem six_points_in_circle_theorem (points : Fin 6 → Point) 
  (h : ∀ i, unitDiameterCircle (points i)) :
  ∃ i j, i ≠ j ∧ distance (points i) (points j) ≤ 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_in_circle_theorem_l319_31950
