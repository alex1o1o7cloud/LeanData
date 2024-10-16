import Mathlib

namespace NUMINAMATH_CALUDE_power_division_equivalence_l1722_172273

theorem power_division_equivalence : 8^15 / 64^5 = 32768 := by
  have h1 : 8 = 2^3 := by sorry
  have h2 : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_division_equivalence_l1722_172273


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_180_choose_90_l1722_172266

theorem largest_two_digit_prime_factor_of_180_choose_90 : 
  ∃ (p : ℕ), 
    Prime p ∧ 
    10 ≤ p ∧ 
    p < 100 ∧ 
    p ∣ Nat.choose 180 90 ∧ 
    ∀ (q : ℕ), Prime q → 10 ≤ q → q < 100 → q ∣ Nat.choose 180 90 → q ≤ p ∧
    p = 59 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_180_choose_90_l1722_172266


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l1722_172205

theorem cubic_polynomial_roots (P : ℝ → ℝ) (x y z : ℝ) :
  P = (fun t ↦ t^3 - 2*t^2 - 10*t - 3) →
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 0 ∧ P b = 0 ∧ P c = 0) →
  x + y + z = 2 →
  x*y + x*z + y*z = -10 →
  x*y*z = 3 →
  let u := x^2 * y^2 * z
  let v := x^2 * z^2 * y
  let w := y^2 * z^2 * x
  let R := fun t ↦ t^3 - (u + v + w)*t^2 + (u*v + u*w + v*w)*t - u*v*w
  R = fun t ↦ t^3 + 30*t^2 + 54*t - 243 := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l1722_172205


namespace NUMINAMATH_CALUDE_number_of_refills_l1722_172285

/-- Proves that the number of refills is 4 given the total spent and cost per refill -/
theorem number_of_refills (total_spent : ℕ) (cost_per_refill : ℕ) 
  (h1 : total_spent = 40) 
  (h2 : cost_per_refill = 10) : 
  total_spent / cost_per_refill = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_refills_l1722_172285


namespace NUMINAMATH_CALUDE_a4_value_l1722_172244

theorem a4_value (aₙ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = aₙ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₄ = 5 := by
sorry

end NUMINAMATH_CALUDE_a4_value_l1722_172244


namespace NUMINAMATH_CALUDE_problem_solution_l1722_172262

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | (x - m + 2)*(x - m - 2) ≤ 0}

theorem problem_solution :
  (∀ m : ℝ, (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2) ∧
  (∀ m : ℝ, (A ⊆ (Set.univ \ B m)) → (m < -3 ∨ m > 5)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1722_172262


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1722_172240

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (x - 3 * (x - 2) ≤ 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ (x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1722_172240


namespace NUMINAMATH_CALUDE_train_crossing_time_l1722_172280

/-- Proves that a train 300 meters long, traveling at 90 km/hr, will take 12 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 300 →
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1722_172280


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1722_172237

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1722_172237


namespace NUMINAMATH_CALUDE_division_multiplication_equivalence_l1722_172225

theorem division_multiplication_equivalence : 
  (5.8 / 0.001) = (5.8 * 1000) := by sorry

end NUMINAMATH_CALUDE_division_multiplication_equivalence_l1722_172225


namespace NUMINAMATH_CALUDE_max_area_at_P_l1722_172213

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := 4 - x^2

/-- Line equation -/
def line (x : ℝ) : ℝ := 4 * x

/-- Point P on the parabola -/
def P : ℝ × ℝ := (-2, 0)

/-- Theorem stating that P maximizes the area of triangle PAB -/
theorem max_area_at_P :
  ∀ x : ℝ, x ≠ -2 →
  (parabola x - line x)^2 + (x - P.1)^2 ≤ (parabola P.1 - line P.1)^2 + (P.1 - x)^2 :=
sorry

end NUMINAMATH_CALUDE_max_area_at_P_l1722_172213


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1722_172257

theorem fraction_subtraction (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) - 1 / (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1722_172257


namespace NUMINAMATH_CALUDE_ellipse_focus_m_l1722_172212

/-- Given an ellipse with equation x²/m + y²/4 = 1 and one focus at (0,1), prove that m = 3 -/
theorem ellipse_focus_m (m : ℝ) : 
  (∀ x y : ℝ, x^2/m + y^2/4 = 1) →  -- Ellipse equation
  (∃ x y : ℝ, x^2/m + y^2/4 = 1 ∧ (0, 1) = (x, y)) →  -- One focus at (0,1)
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_m_l1722_172212


namespace NUMINAMATH_CALUDE_line_intercepts_l1722_172230

/-- Given a line with equation x - 2y - 2 = 0, prove that its x-intercept is 2 and y-intercept is -1 -/
theorem line_intercepts :
  let line := {(x, y) : ℝ × ℝ | x - 2*y - 2 = 0}
  let x_intercept := {x : ℝ | ∃ y, (x, y) ∈ line ∧ y = 0}
  let y_intercept := {y : ℝ | ∃ x, (x, y) ∈ line ∧ x = 0}
  x_intercept = {2} ∧ y_intercept = {-1} := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_l1722_172230


namespace NUMINAMATH_CALUDE_x_eq_zero_necessary_not_sufficient_l1722_172251

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem x_eq_zero_necessary_not_sufficient :
  (∀ x y : ℝ, is_purely_imaginary (Complex.mk x y) → x = 0) ∧
  ¬(∀ x y : ℝ, x = 0 → is_purely_imaginary (Complex.mk x y)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_zero_necessary_not_sufficient_l1722_172251


namespace NUMINAMATH_CALUDE_factorization_proof_l1722_172202

theorem factorization_proof (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1722_172202


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l1722_172203

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 1 = 0

-- Define the condition of external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m

-- State the theorem
theorem tangent_circles_m_value (m : ℝ) :
  externally_tangent m → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l1722_172203


namespace NUMINAMATH_CALUDE_largest_value_l1722_172265

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def digits_85_9 : List Nat := [8, 5]
def digits_111111_2 : List Nat := [1, 1, 1, 1, 1, 1]
def digits_1000_4 : List Nat := [1, 0, 0, 0]
def digits_210_6 : List Nat := [2, 1, 0]

theorem largest_value :
  let a := to_decimal digits_85_9 9
  let b := to_decimal digits_111111_2 2
  let c := to_decimal digits_1000_4 4
  let d := to_decimal digits_210_6 6
  d > a ∧ d > b ∧ d > c := by sorry

end NUMINAMATH_CALUDE_largest_value_l1722_172265


namespace NUMINAMATH_CALUDE_freddy_age_l1722_172235

theorem freddy_age (F M R : ℕ) 
  (sum_ages : F + M + R = 35)
  (matthew_rebecca : M = R + 2)
  (freddy_matthew : F = M + 4) :
  F = 15 := by
sorry

end NUMINAMATH_CALUDE_freddy_age_l1722_172235


namespace NUMINAMATH_CALUDE_function_value_problem_l1722_172219

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f x = 2^x - 5)
  (h2 : f m = 3) : 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_function_value_problem_l1722_172219


namespace NUMINAMATH_CALUDE_total_cost_mangoes_l1722_172270

def prices : List Nat := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def boxes : Nat := 36

theorem total_cost_mangoes :
  (List.sum prices) * boxes = 3060 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_mangoes_l1722_172270


namespace NUMINAMATH_CALUDE_reactions_not_usable_in_primary_cell_l1722_172297

-- Define the types of reactions
inductive ReactionType
| Neutralization
| Redox
| Endothermic

-- Define a structure for chemical reactions
structure ChemicalReaction where
  id : Nat
  reactionType : ReactionType
  isExothermic : Bool

-- Define the condition for a reaction to be used in a primary cell
def canBeUsedInPrimaryCell (reaction : ChemicalReaction) : Prop :=
  reaction.reactionType = ReactionType.Redox ∧ reaction.isExothermic

-- Define the given reactions
def reaction1 : ChemicalReaction :=
  { id := 1, reactionType := ReactionType.Neutralization, isExothermic := true }

def reaction2 : ChemicalReaction :=
  { id := 2, reactionType := ReactionType.Redox, isExothermic := true }

def reaction3 : ChemicalReaction :=
  { id := 3, reactionType := ReactionType.Redox, isExothermic := true }

def reaction4 : ChemicalReaction :=
  { id := 4, reactionType := ReactionType.Endothermic, isExothermic := false }

-- Theorem to prove
theorem reactions_not_usable_in_primary_cell :
  ¬(canBeUsedInPrimaryCell reaction1) ∧ ¬(canBeUsedInPrimaryCell reaction4) :=
sorry

end NUMINAMATH_CALUDE_reactions_not_usable_in_primary_cell_l1722_172297


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1722_172218

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- State the theorem
theorem interest_rate_calculation (P : ℝ) (r : ℝ) 
  (h1 : compound_interest P r 6 = 6000)
  (h2 : compound_interest P r 7 = 7500) :
  r = 0.25 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1722_172218


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1722_172241

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, 4)

-- Define vector a as a function of x
def a (x : ℝ) : ℝ × ℝ := (2*x - 1, x^2 + 3*x - 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_equation_solution :
  ∃ x : ℝ, a x = AB ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1722_172241


namespace NUMINAMATH_CALUDE_solution_pairs_l1722_172238

theorem solution_pairs : 
  {(x, y) : ℝ × ℝ | (x^2 + y + 1) * (y^2 + x + 1) = 4 ∧ (x^2 + y)^2 + (y^2 + x)^2 = 2} = 
  {(0, 1), (1, 0), ((Real.sqrt 5 - 1) / 2, (Real.sqrt 5 - 1) / 2), 
   (-(Real.sqrt 5 + 1) / 2, -(Real.sqrt 5 + 1) / 2)} := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l1722_172238


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1722_172210

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (1 + i) * z = 1 - 2 * i^3

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1722_172210


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1722_172287

/-- The function f(x) = a^(x-1) - 2 passes through the point (1, -1) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f := fun x : ℝ => a^(x - 1) - 2
  f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1722_172287


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l1722_172281

/-- Represents a rectangular prism with dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total number of faces of unit cubes in the prism. -/
def totalFaces (p : RectangularPrism) : ℕ := 6 * p.a * p.b * p.c

/-- Calculates the number of red faces in the prism. -/
def redFaces (p : RectangularPrism) : ℕ := 2 * (p.a * p.b + p.b * p.c + p.a * p.c)

/-- Theorem stating the conditions and result for the rectangular prism problem. -/
theorem rectangular_prism_problem (p : RectangularPrism) :
  p.a + p.b + p.c = 12 →
  3 * redFaces p = totalFaces p →
  p.a = 3 ∧ p.b = 4 ∧ p.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_prism_problem_l1722_172281


namespace NUMINAMATH_CALUDE_mo_tea_consumption_l1722_172275

/-- Represents the drinking habits of Mo --/
structure MoDrinkingHabits where
  n : ℕ  -- number of hot chocolate cups on rainy days
  t : ℕ  -- number of tea cups on non-rainy days
  total_cups : ℕ  -- total cups drunk in a week
  tea_chocolate_diff : ℕ  -- difference between tea and hot chocolate cups
  rainy_days : ℕ  -- number of rainy days in a week

/-- Theorem stating Mo's tea consumption on non-rainy days --/
theorem mo_tea_consumption (mo : MoDrinkingHabits) 
  (h1 : mo.total_cups = 36)
  (h2 : mo.tea_chocolate_diff = 14)
  (h3 : mo.rainy_days = 2)
  (h4 : mo.rainy_days * mo.n + (7 - mo.rainy_days) * mo.t = mo.total_cups)
  (h5 : (7 - mo.rainy_days) * mo.t = mo.rainy_days * mo.n + mo.tea_chocolate_diff) :
  mo.t = 5 := by
  sorry

#check mo_tea_consumption

end NUMINAMATH_CALUDE_mo_tea_consumption_l1722_172275


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1722_172282

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 9)
  (h_6 : a 6 = 15) :
  a 12 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1722_172282


namespace NUMINAMATH_CALUDE_volume_of_specific_triangular_pyramid_l1722_172283

-- Define the regular triangular prism
structure RegularTriangularPrism where
  baseEdgeLength : ℝ
  sideEdgeLength : ℝ

-- Define the triangular pyramid
structure TriangularPyramid where
  base : RegularTriangularPrism
  apexPoint : Point

-- Define the volume function for the triangular pyramid
def volumeTriangularPyramid (pyramid : TriangularPyramid) : ℝ := sorry

-- Theorem statement
theorem volume_of_specific_triangular_pyramid 
  (prism : RegularTriangularPrism)
  (pyramid : TriangularPyramid)
  (h1 : prism.baseEdgeLength = 2)
  (h2 : prism.sideEdgeLength = Real.sqrt 3)
  (h3 : pyramid.base = prism)
  : volumeTriangularPyramid pyramid = 1 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_triangular_pyramid_l1722_172283


namespace NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l1722_172247

/-- The number of apps Dave initially had on his phone -/
def initial_apps : ℕ := 16

/-- The number of apps Dave had left after deletion -/
def remaining_apps : ℕ := 5

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := initial_apps - remaining_apps

theorem dave_deleted_eleven_apps : deleted_apps = 11 := by
  sorry

end NUMINAMATH_CALUDE_dave_deleted_eleven_apps_l1722_172247


namespace NUMINAMATH_CALUDE_town_distance_proof_l1722_172201

/-- Given a map distance and a scale, calculates the actual distance between two towns. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem stating that for a map distance of 7.5 inches and a scale of 1 inch = 8 miles,
    the actual distance between two towns is 60 miles. -/
theorem town_distance_proof :
  let map_distance : ℝ := 7.5
  let scale : ℝ := 8
  actual_distance map_distance scale = 60 := by
  sorry

end NUMINAMATH_CALUDE_town_distance_proof_l1722_172201


namespace NUMINAMATH_CALUDE_min_value_sum_l1722_172206

theorem min_value_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 19 / x + 98 / y = 1) :
  x + y ≥ 117 + 14 * Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l1722_172206


namespace NUMINAMATH_CALUDE_brendas_blisters_l1722_172263

/-- The number of blisters Brenda has on each arm -/
def blisters_per_arm : ℕ := 60

/-- The number of blisters Brenda has on the rest of her body -/
def blisters_on_body : ℕ := 80

/-- The number of arms Brenda has -/
def number_of_arms : ℕ := 2

/-- The total number of blisters Brenda has -/
def total_blisters : ℕ := blisters_per_arm * number_of_arms + blisters_on_body

theorem brendas_blisters : total_blisters = 200 := by
  sorry

end NUMINAMATH_CALUDE_brendas_blisters_l1722_172263


namespace NUMINAMATH_CALUDE_volume_cube_sphere_region_l1722_172279

/-- The volume of the region within a cube of side length 4 cm, outside an inscribed sphere
    tangent to the cube, and closest to one vertex of the cube. -/
theorem volume_cube_sphere_region (π : ℝ) (h : π = Real.pi) :
  let a : ℝ := 4
  let cube_volume : ℝ := a ^ 3
  let sphere_radius : ℝ := a / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  let outside_sphere_volume : ℝ := cube_volume - sphere_volume
  let region_volume : ℝ := (1 / 8) * outside_sphere_volume
  region_volume = 8 * (1 - π / 6) :=
by sorry

end NUMINAMATH_CALUDE_volume_cube_sphere_region_l1722_172279


namespace NUMINAMATH_CALUDE_fractional_method_experiments_l1722_172228

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The number of experimental points -/
def num_points : ℕ := 12

/-- The maximum number of additional experiments needed -/
def max_additional_experiments : ℕ := 5

/-- Theorem: Given 12 experimental points and using the fractional method
    to find the optimal point of a unimodal function, the maximum number
    of additional experiments needed is 5. -/
theorem fractional_method_experiments :
  ∃ k : ℕ, num_points = fib (k + 1) - 1 ∧ max_additional_experiments = k :=
sorry

end NUMINAMATH_CALUDE_fractional_method_experiments_l1722_172228


namespace NUMINAMATH_CALUDE_valid_assignment_example_l1722_172248

def is_variable (s : String) : Prop := s.length > 0 ∧ s.all Char.isAlpha

def is_expression (s : String) : Prop := s.length > 0

def is_valid_assignment (s : String) : Prop :=
  ∃ (lhs rhs : String),
    s = lhs ++ " = " ++ rhs ∧
    is_variable lhs ∧
    is_expression rhs

theorem valid_assignment_example :
  is_valid_assignment "A = A*A + A - 2" := by sorry

end NUMINAMATH_CALUDE_valid_assignment_example_l1722_172248


namespace NUMINAMATH_CALUDE_opposite_sides_m_range_l1722_172290

/-- Given two points on opposite sides of a line, prove the range of m -/
theorem opposite_sides_m_range :
  ∀ (m : ℝ),
  (2 * 1 + 3 + m) * (2 * (-4) + (-2) + m) < 0 →
  m ∈ Set.Ioo (-5 : ℝ) 10 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_m_range_l1722_172290


namespace NUMINAMATH_CALUDE_middle_number_proof_l1722_172216

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
    (h3 : a + b = 15) (h4 : a + c = 18) (h5 : b + c = 21) : b = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1722_172216


namespace NUMINAMATH_CALUDE_parabola_distance_l1722_172258

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance (A : ℝ × ℝ) :
  parabola A.1 A.2 →
  ‖A - focus‖ = ‖point_B - focus‖ →
  ‖A - point_B‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_distance_l1722_172258


namespace NUMINAMATH_CALUDE_max_value_ab_l1722_172299

theorem max_value_ab (a b : ℝ) : 
  (∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) → 
  a * b ≤ (1/2) * Real.exp 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_ab_l1722_172299


namespace NUMINAMATH_CALUDE_prob_club_then_heart_l1722_172208

/-- The number of cards in a standard deck --/
def standard_deck_size : ℕ := 52

/-- The number of clubs in a standard deck --/
def num_clubs : ℕ := 13

/-- The number of hearts in a standard deck --/
def num_hearts : ℕ := 13

/-- Probability of drawing a club first and then a heart from a standard 52-card deck --/
theorem prob_club_then_heart : 
  (num_clubs : ℚ) / standard_deck_size * num_hearts / (standard_deck_size - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_prob_club_then_heart_l1722_172208


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1722_172286

def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 1) - f x = 2 * x + 1) ∧
  (∀ x, f (-x) = f x) ∧
  (f 0 = 1)

theorem quadratic_function_properties (f : ℝ → ℝ) (h : QuadraticFunction f) :
  (∀ x, f x = x^2 + 1) ∧
  (∀ x ∈ Set.Icc (-2) 1, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc (-2) 1, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2) 1, f x = 5) ∧
  (∃ x ∈ Set.Icc (-2) 1, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1722_172286


namespace NUMINAMATH_CALUDE_divisibility_of_polynomial_l1722_172232

theorem divisibility_of_polynomial (x : ℤ) :
  (x^2 + 1) * (x^8 - x^6 + x^4 - x^2 + 1) = x^10 + 1 →
  ∃ k : ℤ, x^2030 + 1 = k * (x^8 - x^6 + x^4 - x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_polynomial_l1722_172232


namespace NUMINAMATH_CALUDE_train_length_l1722_172207

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 70) (h2 : t = 13.884603517432893) (h3 : bridge_length = 150) :
  ∃ (train_length : ℝ), abs (train_length - 120) < 1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1722_172207


namespace NUMINAMATH_CALUDE_inequality_solution_and_function_property_l1722_172220

def f (x : ℝ) : ℝ := |x + 1|

theorem inequality_solution_and_function_property :
  (∃ (S : Set ℝ), S = {x : ℝ | x ≤ -10 ∨ x ≥ 0} ∧
    ∀ x, x ∈ S ↔ f (x + 8) ≥ 10 - f x) ∧
  (∀ x y, |x| > 1 → |y| < 1 → f y < |x| * f (y / x^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_and_function_property_l1722_172220


namespace NUMINAMATH_CALUDE_length_FG_is_20_l1722_172274

/-- Triangle PQR with points F and G -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Point F on PQ -/
  F : ℝ
  /-- Point G on PR -/
  G : ℝ
  /-- FG is parallel to QR -/
  FG_parallel_QR : Bool
  /-- G divides PR in ratio 2:1 -/
  G_divides_PR : G = (2/3) * PR

/-- The length of FG in the given triangle configuration -/
def length_FG (t : TrianglePQR) : ℝ := sorry

/-- Theorem stating that the length of FG is 20 under the given conditions -/
theorem length_FG_is_20 (t : TrianglePQR) 
  (h1 : t.PQ = 24) 
  (h2 : t.PR = 26) 
  (h3 : t.QR = 30) 
  (h4 : t.FG_parallel_QR = true) : 
  length_FG t = 20 := by sorry

end NUMINAMATH_CALUDE_length_FG_is_20_l1722_172274


namespace NUMINAMATH_CALUDE_star_seven_three_l1722_172245

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 5*a + 3*b - 2*a*b

-- Theorem statement
theorem star_seven_three : star 7 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l1722_172245


namespace NUMINAMATH_CALUDE_fifth_month_sale_proof_l1722_172233

/-- Calculates the sale in the fifth month given the sales of other months and the average -/
def fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) : ℕ :=
  6 * avg - (m1 + m2 + m3 + m4 + m6)

/-- Proves that the sale in the fifth month is 3562 given the specified conditions -/
theorem fifth_month_sale_proof :
  fifth_month_sale 3435 3927 3855 4230 1991 3500 = 3562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_proof_l1722_172233


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l1722_172278

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l1722_172278


namespace NUMINAMATH_CALUDE_gcd_m_l1722_172288

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_m_l1722_172288


namespace NUMINAMATH_CALUDE_abs_x_leq_2_necessary_not_sufficient_l1722_172227

theorem abs_x_leq_2_necessary_not_sufficient :
  (∃ x : ℝ, |x + 1| ≤ 1 ∧ ¬(|x| ≤ 2)) = False ∧
  (∃ x : ℝ, |x| ≤ 2 ∧ ¬(|x + 1| ≤ 1)) = True :=
by sorry

end NUMINAMATH_CALUDE_abs_x_leq_2_necessary_not_sufficient_l1722_172227


namespace NUMINAMATH_CALUDE_monkey_percentage_after_events_l1722_172267

/-- Represents the counts of animals in the tree --/
structure AnimalCounts where
  monkeys : ℕ
  birds : ℕ
  squirrels : ℕ
  cats : ℕ

/-- Calculates the total number of animals --/
def totalAnimals (counts : AnimalCounts) : ℕ :=
  counts.monkeys + counts.birds + counts.squirrels + counts.cats

/-- Applies the events described in the problem --/
def applyEvents (initial : AnimalCounts) : AnimalCounts :=
  { monkeys := initial.monkeys,
    birds := initial.birds - 2 - 2,  -- 2 eaten by monkeys, 2 chased away
    squirrels := initial.squirrels - 1,  -- 1 chased away
    cats := initial.cats }

/-- Calculates the percentage of monkeys after the events --/
def monkeyPercentage (initial : AnimalCounts) : ℚ :=
  let final := applyEvents initial
  (final.monkeys : ℚ) / (totalAnimals final : ℚ) * 100

theorem monkey_percentage_after_events :
  let initial : AnimalCounts := { monkeys := 6, birds := 9, squirrels := 3, cats := 5 }
  monkeyPercentage initial = 100/3 := by sorry

end NUMINAMATH_CALUDE_monkey_percentage_after_events_l1722_172267


namespace NUMINAMATH_CALUDE_grade_11_count_l1722_172293

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  sample_size : ℕ
  grade_10_sample : ℕ
  grade_12_sample : ℕ

/-- Calculates the number of Grade 11 students in the school -/
def grade_11_students (s : School) : ℕ :=
  ((s.sample_size - s.grade_10_sample - s.grade_12_sample) * s.total_students) / s.sample_size

/-- Theorem stating the number of Grade 11 students in the given school -/
theorem grade_11_count (s : School) 
  (h1 : s.total_students = 900)
  (h2 : s.sample_size = 45)
  (h3 : s.grade_10_sample = 20)
  (h4 : s.grade_12_sample = 10) :
  grade_11_students s = 300 := by
  sorry

#eval grade_11_students ⟨900, 45, 20, 10⟩

end NUMINAMATH_CALUDE_grade_11_count_l1722_172293


namespace NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l1722_172264

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

def C_R_B : Set ℝ := (Set.univ : Set ℝ) \ B

theorem intersection_equals_open_closed_interval : (C_R_B ∩ A) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_closed_interval_l1722_172264


namespace NUMINAMATH_CALUDE_log_sum_property_l1722_172289

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_log : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_log_sum_property_l1722_172289


namespace NUMINAMATH_CALUDE_original_page_count_l1722_172294

/-- Represents a book with numbered pages -/
structure Book where
  pages : ℕ

/-- Calculates the total number of digits in the page numbers of remaining pages after removing even-numbered sheets -/
def remainingDigits (b : Book) : ℕ := sorry

/-- Theorem stating the possible original page counts given the remaining digit count -/
theorem original_page_count (b : Book) : 
  remainingDigits b = 845 → b.pages = 598 ∨ b.pages = 600 := by sorry

end NUMINAMATH_CALUDE_original_page_count_l1722_172294


namespace NUMINAMATH_CALUDE_vasya_driving_distance_l1722_172234

theorem vasya_driving_distance 
  (total_distance : ℝ) 
  (anton_distance vasya_distance sasha_distance dima_distance : ℝ) 
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance)
  : vasya_distance = (2 / 5) * total_distance :=
by sorry

end NUMINAMATH_CALUDE_vasya_driving_distance_l1722_172234


namespace NUMINAMATH_CALUDE_product_sum_equality_l1722_172253

theorem product_sum_equality (x y z : ℕ) 
  (h1 : 2014 + y = 2015 + x)
  (h2 : 2015 + x = 2016 + z)
  (h3 : y * x * z = 504) :
  y * x + x * z = 128 := by
sorry

end NUMINAMATH_CALUDE_product_sum_equality_l1722_172253


namespace NUMINAMATH_CALUDE_trig_inequality_l1722_172243

theorem trig_inequality : ∃ (a b c : ℝ), 
  a = Real.cos 1 ∧ 
  b = Real.sin 1 ∧ 
  c = Real.tan 1 ∧ 
  a < b ∧ b < c :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1722_172243


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l1722_172211

theorem quadratic_equation_integer_roots (a : ℝ) : 
  (a > 0 ∧ ∃ x y : ℤ, x ≠ y ∧ 
    a^2 * (x : ℝ)^2 + a * (x : ℝ) + 1 - 7 * a^2 = 0 ∧
    a^2 * (y : ℝ)^2 + a * (y : ℝ) + 1 - 7 * a^2 = 0) ↔ 
  (a = 1 ∨ a = 1/2 ∨ a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l1722_172211


namespace NUMINAMATH_CALUDE_alexa_vacation_fraction_is_three_fourths_l1722_172236

/-- The number of days Alexa spent on vacation -/
def alexa_vacation_days : ℕ := 7 + 2

/-- The number of days it took Joey to learn swimming -/
def joey_swimming_days : ℕ := 6

/-- The number of days it took Ethan to learn fencing tricks -/
def ethan_fencing_days : ℕ := 2 * joey_swimming_days

/-- The fraction of time Alexa spent on vacation compared to Ethan's fencing learning time -/
def alexa_vacation_fraction : ℚ := alexa_vacation_days / ethan_fencing_days

theorem alexa_vacation_fraction_is_three_fourths :
  alexa_vacation_fraction = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_alexa_vacation_fraction_is_three_fourths_l1722_172236


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1722_172271

theorem complex_equation_solution (x : ℝ) (y : ℂ) 
  (h1 : y.re = 0)  -- y is purely imaginary
  (h2 : (3 * x + 1 : ℂ) - 2 * Complex.I = y) : 
  x = -1/3 ∧ y = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1722_172271


namespace NUMINAMATH_CALUDE_suyeong_run_distance_l1722_172254

/-- The circumference of the playground in meters -/
def playground_circumference : ℝ := 242.7

/-- The number of laps Suyeong ran -/
def laps_run : ℕ := 5

/-- The total distance Suyeong ran in meters -/
def total_distance : ℝ := playground_circumference * (laps_run : ℝ)

theorem suyeong_run_distance : total_distance = 1213.5 := by
  sorry

end NUMINAMATH_CALUDE_suyeong_run_distance_l1722_172254


namespace NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l1722_172204

theorem acute_triangle_tangent_inequality (A B C : Real) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (1 / (1 + Real.tan A) + 1 / (1 + Real.tan B)) < (Real.tan A / (1 + Real.tan A) + Real.tan B / (1 + Real.tan B)) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l1722_172204


namespace NUMINAMATH_CALUDE_at_least_one_passes_probability_l1722_172252

/-- Probability of A answering a single question correctly -/
def prob_A : ℚ := 2/3

/-- Probability of B answering a single question correctly -/
def prob_B : ℚ := 1/2

/-- Number of questions in the test -/
def num_questions : ℕ := 3

/-- Number of correct answers required to pass -/
def pass_threshold : ℕ := 2

/-- Probability of at least one of A and B passing the test -/
def prob_at_least_one_passes : ℚ := 47/54

theorem at_least_one_passes_probability :
  prob_at_least_one_passes = 1 - (1 - (Nat.choose num_questions pass_threshold * prob_A^pass_threshold * (1-prob_A)^(num_questions-pass_threshold) + prob_A^num_questions)) *
                                 (1 - (Nat.choose num_questions pass_threshold * prob_B^pass_threshold * (1-prob_B)^(num_questions-pass_threshold) + prob_B^num_questions)) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_passes_probability_l1722_172252


namespace NUMINAMATH_CALUDE_fraction_simplification_l1722_172277

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1722_172277


namespace NUMINAMATH_CALUDE_sequence_length_is_602_l1722_172269

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The number of terms in the specified arithmetic sequence is 602 -/
theorem sequence_length_is_602 :
  arithmetic_sequence_length 3 3008 5 = 602 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_is_602_l1722_172269


namespace NUMINAMATH_CALUDE_octagon_perimeter_is_six_l1722_172214

/-- A pentagon formed by removing a right-angled isosceles triangle from a unit square -/
structure Pentagon where
  /-- The side length of the original square -/
  squareSide : ℝ
  /-- The length of the leg of the removed right-angled isosceles triangle -/
  triangleLeg : ℝ
  /-- Assertion that the square is a unit square -/
  squareIsUnit : squareSide = 1
  /-- Assertion that the removed triangle is right-angled isosceles with leg length equal to the square side -/
  triangleIsRightIsosceles : triangleLeg = squareSide

/-- An octagon formed by fitting together two congruent pentagons -/
structure Octagon where
  /-- The first pentagon used to form the octagon -/
  pentagon1 : Pentagon
  /-- The second pentagon used to form the octagon -/
  pentagon2 : Pentagon
  /-- Assertion that the two pentagons are congruent -/
  pentagonsAreCongruent : pentagon1 = pentagon2

/-- The perimeter of the octagon -/
def octagonPerimeter (o : Octagon) : ℝ :=
  -- Definition of perimeter calculation goes here
  sorry

/-- Theorem: The perimeter of the octagon is 6 -/
theorem octagon_perimeter_is_six (o : Octagon) : octagonPerimeter o = 6 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_is_six_l1722_172214


namespace NUMINAMATH_CALUDE_largest_pentagon_angle_l1722_172250

/-- Represents the measures of interior angles of a convex pentagon --/
structure PentagonAngles where
  a1 : ℝ
  a2 : ℝ
  a3 : ℝ
  a4 : ℝ
  a5 : ℝ

/-- Theorem: The largest angle in a convex pentagon with specific angle measures is 162° --/
theorem largest_pentagon_angle (x : ℝ) (angles : PentagonAngles) 
  (h1 : angles.a1 = 2*x - 8)
  (h2 : angles.a2 = 3*x + 12)
  (h3 : angles.a3 = 4*x + 8)
  (h4 : angles.a4 = 5*x - 18)
  (h5 : angles.a5 = x + 6)
  (h_sum : angles.a1 + angles.a2 + angles.a3 + angles.a4 + angles.a5 = 540) :
  max angles.a1 (max angles.a2 (max angles.a3 (max angles.a4 angles.a5))) = 162 := by
  sorry


end NUMINAMATH_CALUDE_largest_pentagon_angle_l1722_172250


namespace NUMINAMATH_CALUDE_pigeonhole_multiples_of_five_l1722_172298

theorem pigeonhole_multiples_of_five (n : ℕ) (h : n = 200) : 
  ∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card ≥ 82 → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a + b) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_multiples_of_five_l1722_172298


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1722_172260

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem: Given the specified conditions, the man's speed against the current is 9.4 km/hr -/
theorem mans_speed_against_current :
  speedAgainstCurrent 15 2.8 = 9.4 := by
  sorry

#eval speedAgainstCurrent 15 2.8

end NUMINAMATH_CALUDE_mans_speed_against_current_l1722_172260


namespace NUMINAMATH_CALUDE_last_term_formula_l1722_172231

def u (n : ℕ) : ℕ := 2 + 5 * ((n - 1) % (3 * ((n - 1).sqrt + 1) - 1))

def f (n : ℕ) : ℕ := (15 * n^2 + 10 * n + 4) / 2

theorem last_term_formula (n : ℕ) : 
  u ((n^2 + n) / 2) = f n :=
sorry

end NUMINAMATH_CALUDE_last_term_formula_l1722_172231


namespace NUMINAMATH_CALUDE_candy_bar_profit_l1722_172200

def candy_bars_bought : ℕ := 1500
def buying_price : ℚ := 3 / 8
def selling_price : ℚ := 2 / 3
def booth_setup_cost : ℚ := 50

def total_cost : ℚ := candy_bars_bought * buying_price
def total_revenue : ℚ := candy_bars_bought * selling_price
def net_profit : ℚ := total_revenue - total_cost - booth_setup_cost

theorem candy_bar_profit : net_profit = 387.5 := by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l1722_172200


namespace NUMINAMATH_CALUDE_prime_square_difference_l1722_172222

theorem prime_square_difference (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (hp_form : ∃ k, p = 4*k + 3) (hq_form : ∃ k, q = 4*k + 3)
  (h_exists : ∃ (x y : ℤ), x^2 - p*q*y^2 = 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (p*a^2 - q*b^2 = 1 ∨ q*b^2 - p*a^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_difference_l1722_172222


namespace NUMINAMATH_CALUDE_tangent_product_l1722_172221

theorem tangent_product (α β : Real) 
  (h1 : Real.cos (α + β) = 1/3)
  (h2 : Real.cos (α - β) = 1/5) :
  Real.tan α * Real.tan β = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l1722_172221


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1722_172292

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 4 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 4 * Real.sqrt 6) ∧ 
  (θ = π / 8) ∧ 
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * π) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1722_172292


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_l1722_172261

/-- The smallest positive integer x such that 2520x is a perfect cube -/
def smallest_x : ℕ := 3675

/-- 2520 as a natural number -/
def given_number : ℕ := 2520

theorem smallest_x_is_correct :
  (∀ y : ℕ, y > 0 ∧ y < smallest_x → ¬∃ M : ℕ, given_number * y = M^3) ∧
  ∃ M : ℕ, given_number * smallest_x = M^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_is_correct_l1722_172261


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1722_172229

theorem inequality_equivalence (x y : ℝ) :
  y - x > Real.sqrt (x^2 + 9) ↔ y > x + Real.sqrt (x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1722_172229


namespace NUMINAMATH_CALUDE_simplify_expression_l1722_172255

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1722_172255


namespace NUMINAMATH_CALUDE_hyperbola_and_line_theorem_l1722_172215

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - (4 * y^2 / 33) = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define point P
def P : ℝ × ℝ := (7, 12)

-- Define the asymptotes
def asymptote_positive (x y : ℝ) : Prop := y = (Real.sqrt 33 / 2) * x
def asymptote_negative (x y : ℝ) : Prop := y = -(Real.sqrt 33 / 2) * x

-- Define line l
def line_l (x y t : ℝ) : Prop := y = x + t

-- Define perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_and_line_theorem :
  -- Hyperbola C passes through P
  hyperbola_C P.1 P.2 →
  -- There exist points A and B on C and l
  ∃ (A B : ℝ × ℝ) (t : ℝ),
    hyperbola_C A.1 A.2 ∧ 
    hyperbola_C B.1 B.2 ∧
    line_l A.1 A.2 t ∧
    line_l B.1 B.2 t ∧
    -- A and B are perpendicular from the origin
    perpendicular A.1 A.2 B.1 B.2 →
  -- Then the equation of line l is y = x ± √(66/29)
  t = Real.sqrt (66 / 29) ∨ t = -Real.sqrt (66 / 29) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_theorem_l1722_172215


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1722_172284

-- Define the regular quadrilateral pyramid
structure RegularQuadPyramid where
  a : ℝ  -- base edge
  h : ℝ  -- height
  lateral_edge : ℝ
  lateral_edge_eq : lateral_edge = (5/2) * a

-- Define the cylinder
structure Cylinder (P : RegularQuadPyramid) where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cylinder

-- Theorem statement
theorem cylinder_lateral_surface_area 
  (P : RegularQuadPyramid) 
  (C : Cylinder P) :
  ∃ (S : ℝ), S = (π * P.a^2 * Real.sqrt 46) / 9 ∧ 
  S = 2 * π * C.r * C.h :=
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1722_172284


namespace NUMINAMATH_CALUDE_bagel_bakery_bound_l1722_172242

/-- Definition of a bagel -/
def Bagel (a b : ℕ) : ℕ := 2 * a + 2 * b + 4

/-- Definition of a bakery of order n -/
def Bakery (n : ℕ) : Set (ℕ × ℕ) := sorry

/-- The smallest possible number of cells in a bakery of order n -/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem bagel_bakery_bound :
  ∃ (α : ℝ), ∀ (n : ℕ), n ≥ 8 → Even n →
    ∃ (N : ℕ), ∀ (m : ℕ), m ≥ N →
      (1 / 100 : ℝ) < (f m : ℝ) / m ^ α ∧ (f m : ℝ) / m ^ α < 100 :=
by sorry

end NUMINAMATH_CALUDE_bagel_bakery_bound_l1722_172242


namespace NUMINAMATH_CALUDE_total_cost_for_cakes_l1722_172276

/-- The number of cakes Claire wants to make -/
def num_cakes : ℕ := 2

/-- The number of packages of flour required for one cake -/
def packages_per_cake : ℕ := 2

/-- The cost of one package of flour in dollars -/
def cost_per_package : ℕ := 3

/-- Theorem: The total cost of flour for making 2 cakes is $12 -/
theorem total_cost_for_cakes : num_cakes * packages_per_cake * cost_per_package = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_cakes_l1722_172276


namespace NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l1722_172246

theorem pascal_triangle_51st_row_third_number : 
  let n : ℕ := 50  -- The row with 51 numbers corresponds to (x+y)^50
  let k : ℕ := 2   -- The third number (0-indexed) corresponds to k=2
  Nat.choose n k = 1225 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l1722_172246


namespace NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l1722_172249

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := sorry

-- Define what it means for two angles of a triangle to be equal
def twoAnglesEqual (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_angles (t : Triangle) :
  (¬(isIsosceles t) → ¬(twoAnglesEqual t)) ↔ (twoAnglesEqual t → isIsosceles t) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l1722_172249


namespace NUMINAMATH_CALUDE_alice_grading_papers_l1722_172209

/-- Given that Ms. Alice can grade 296 papers in 8 hours, prove that she can grade 407 papers in 11 hours. -/
theorem alice_grading_papers : 
  let papers_in_8_hours : ℕ := 296
  let hours_initial : ℕ := 8
  let hours_new : ℕ := 11
  let papers_in_11_hours : ℕ := 407
  (papers_in_8_hours : ℚ) / hours_initial * hours_new = papers_in_11_hours :=
by sorry

end NUMINAMATH_CALUDE_alice_grading_papers_l1722_172209


namespace NUMINAMATH_CALUDE_left_seats_count_l1722_172217

/-- Represents the seating configuration of a bus -/
structure BusSeats where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusConfig (bus : BusSeats) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeat = 8 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 89 ∧
  bus.leftSeats * bus.seatCapacity + bus.rightSeats * bus.seatCapacity + bus.backSeat = bus.totalCapacity

/-- The number of seats on the left side of the bus is 15 -/
theorem left_seats_count (bus : BusSeats) (h : validBusConfig bus) : bus.leftSeats = 15 := by
  sorry

end NUMINAMATH_CALUDE_left_seats_count_l1722_172217


namespace NUMINAMATH_CALUDE_unique_composite_with_sum_power_of_two_l1722_172224

theorem unique_composite_with_sum_power_of_two :
  ∃! m : ℕ+, 
    (1 < m) ∧ 
    (∀ a b : ℕ+, a * b = m → ∃ k : ℕ, a + b = 2^k) ∧
    m = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_composite_with_sum_power_of_two_l1722_172224


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1722_172272

/-- Given that x is inversely proportional to y, this function represents their relationship -/
def inverse_proportion (x y : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_ratio 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0)
  (hxy₁ : inverse_proportion x₁ y₁)
  (hxy₂ : inverse_proportion x₂ y₂)
  (hx_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1722_172272


namespace NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l1722_172268

/-- Right prism with equilateral triangle base -/
structure RightPrism :=
  (height : ℝ)
  (base_side_length : ℝ)

/-- Point on an edge of the prism -/
structure EdgePoint :=
  (distance_ratio : ℝ)

/-- Solid formed by slicing off part of the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (point1 : EdgePoint)
  (point2 : EdgePoint)
  (point3 : EdgePoint)

/-- Surface area of the sliced solid -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

theorem surface_area_of_sliced_prism (solid : SlicedSolid) 
  (h1 : solid.prism.height = 24)
  (h2 : solid.prism.base_side_length = 18)
  (h3 : solid.point1.distance_ratio = 1/3)
  (h4 : solid.point2.distance_ratio = 1/3)
  (h5 : solid.point3.distance_ratio = 1/3) :
  surface_area solid = 96 + 9 * Real.sqrt 3 + 9 * Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_sliced_prism_l1722_172268


namespace NUMINAMATH_CALUDE_nn_plus_one_prime_l1722_172239

theorem nn_plus_one_prime (n : ℕ) : n ∈ Finset.range 16 \ {0} →
  Nat.Prime (n^n + 1) ↔ n = 1 ∨ n = 2 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_nn_plus_one_prime_l1722_172239


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l1722_172296

/-- Given that the line x + y = b is a perpendicular bisector of the line segment from (2,4) to (6,8), prove that b = 10. -/
theorem perpendicular_bisector_b_value : 
  ∀ b : ℝ, 
  (∀ x y : ℝ, x + y = b ↔ 
    (x - 4)^2 + (y - 6)^2 = (2 - 4)^2 + (4 - 6)^2 ∧ 
    (x - 4)^2 + (y - 6)^2 = (6 - 4)^2 + (8 - 6)^2) → 
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l1722_172296


namespace NUMINAMATH_CALUDE_bottom_row_bricks_l1722_172295

/-- Represents a brick wall with decreasing number of bricks in each row -/
structure BrickWall where
  totalRows : Nat
  totalBricks : Nat
  bottomRowBricks : Nat
  decreaseRate : Nat
  rowsDecreasing : bottomRowBricks ≥ (totalRows - 1) * decreaseRate

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 : Nat) (an : Nat) (n : Nat) : Nat :=
  n * (a1 + an) / 2

/-- Theorem stating that a brick wall with given properties has 43 bricks in the bottom row -/
theorem bottom_row_bricks (wall : BrickWall)
  (h1 : wall.totalRows = 10)
  (h2 : wall.totalBricks = 385)
  (h3 : wall.decreaseRate = 1)
  : wall.bottomRowBricks = 43 := by
  sorry

end NUMINAMATH_CALUDE_bottom_row_bricks_l1722_172295


namespace NUMINAMATH_CALUDE_exactly_fourteen_numbers_l1722_172226

/-- A function that reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- The property that a number satisfies the given condition -/
def satisfies_condition (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ k : ℕ, (reverse_two_digit n - n) = k^2

/-- The theorem stating that there are exactly 14 numbers satisfying the condition -/
theorem exactly_fourteen_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 14 :=
sorry

end NUMINAMATH_CALUDE_exactly_fourteen_numbers_l1722_172226


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l1722_172291

/-- Given a sinusoidal function y = a * sin(b * x + c) + d that oscillates between 5 and -3,
    prove that the amplitude a is equal to 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l1722_172291


namespace NUMINAMATH_CALUDE_tree_ratio_l1722_172223

/-- The number of streets in the neighborhood -/
def num_streets : ℕ := 18

/-- The number of plum trees planted -/
def num_plum_trees : ℕ := 3

/-- The number of pear trees planted -/
def num_pear_trees : ℕ := 3

/-- The number of apricot trees planted -/
def num_apricot_trees : ℕ := 3

/-- Theorem stating that the ratio of plum trees to pear trees to apricot trees is 1:1:1 -/
theorem tree_ratio : 
  num_plum_trees = num_pear_trees ∧ num_pear_trees = num_apricot_trees :=
sorry

end NUMINAMATH_CALUDE_tree_ratio_l1722_172223


namespace NUMINAMATH_CALUDE_two_possible_values_for_D_l1722_172256

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct_digits (A B C D E : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
  C ≠ D ∧ C ≠ E ∧
  D ≠ E

def addition_equation (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + B) +
  (B * 10000 + C * 1000 + A * 100 + D * 10 + E) =
  (E * 10000 + D * 1000 + D * 100 + E * 10 + E)

theorem two_possible_values_for_D :
  ∃ (D₁ D₂ : ℕ), D₁ ≠ D₂ ∧
  (∀ (A B C D E : ℕ), distinct_digits A B C D E → addition_equation A B C D E →
    D = D₁ ∨ D = D₂) ∧
  (∀ (A B C E : ℕ), ∃ (D : ℕ), distinct_digits A B C D E ∧ addition_equation A B C D E) :=
by sorry

end NUMINAMATH_CALUDE_two_possible_values_for_D_l1722_172256


namespace NUMINAMATH_CALUDE_travel_time_difference_l1722_172259

/-- Proves that the difference in travel time between a 400-mile trip and a 360-mile trip,
    when traveling at a constant speed of 40 miles per hour, is 60 minutes. -/
theorem travel_time_difference (speed : ℝ) (dist1 : ℝ) (dist2 : ℝ) :
  speed = 40 → dist1 = 400 → dist2 = 360 →
  (dist1 / speed - dist2 / speed) * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_difference_l1722_172259
