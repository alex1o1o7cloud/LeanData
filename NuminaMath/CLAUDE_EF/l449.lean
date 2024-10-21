import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_two_difference_l449_44914

theorem gcd_power_two_difference (m n : ℕ+) (h : Nat.Coprime m n) :
  (Nat.gcd (2^(m:ℕ) - 2^(n:ℕ)) (2^((m:ℕ)^2 + (m:ℕ)*(n:ℕ) + (n:ℕ)^2) - 1) = 1) ∨
  (Nat.gcd (2^(m:ℕ) - 2^(n:ℕ)) (2^((m:ℕ)^2 + (m:ℕ)*(n:ℕ) + (n:ℕ)^2) - 1) = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_power_two_difference_l449_44914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_circle_area_l449_44945

/-- Given a rectangle with area 180 and one side three times the length of the other,
    the area of the circle formed by a string with length equal to the rectangle's perimeter,
    rounded to the nearest whole number, is 306 square units. -/
theorem rectangle_to_circle_area : ∀ x y : ℝ,
  x * y = 180 →
  y = 3 * x →
  Int.floor (π * ((2 * (x + y)) / (2 * π))^2 + 0.5) = 306 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_to_circle_area_l449_44945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_theorem_l449_44911

/-- Represents the properties of a river --/
structure River where
  depth : ℝ
  flowRate : ℝ
  discharge : ℝ

/-- Calculates the width of a river given its properties --/
noncomputable def riverWidth (r : River) : ℝ :=
  r.discharge / (r.flowRate * r.depth / 3600 * 1000 * r.depth)

/-- Theorem stating that a river with given properties has a width of 45 meters --/
theorem river_width_theorem (r : River) 
  (h1 : r.depth = 2)
  (h2 : r.flowRate = 4)
  (h3 : r.discharge = 6000) :
  riverWidth r = 45 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_width_theorem_l449_44911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_circle_l449_44933

open Real

-- Define the curve
noncomputable def curve (a x : ℝ) : ℝ := exp x * (x^2 + a*x + 1 - 2*a)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + 2*x + y^2 - 12 = 0

-- Define the tangent line
def tangent_line (a x : ℝ) : ℝ := (1 - a) * x + (1 - 2*a)

-- State the theorem
theorem tangent_intersects_circle (a : ℝ) :
  ∃ x y : ℝ, circle_eq x y ∧ y = tangent_line a x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersects_circle_l449_44933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_given_A_l449_44951

-- Define the dice
def Die : Type := Fin 6

-- Define events A and B
def event_A (blue : Die) : Prop := blue.val = 3 ∨ blue.val = 5
def event_B (red blue : Die) : Prop := red.val + blue.val + 2 > 8

-- Define the probability space
def Ω : Type := Die × Die

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem prob_B_given_A :
  P {ω : Ω | event_B ω.1 ω.2 ∧ event_A ω.2} / P {ω : Ω | event_A ω.2} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_given_A_l449_44951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l449_44937

-- Define the speeds for each hour
noncomputable def speed_hour1 : ℝ := 95
noncomputable def speed_hour2 : ℝ := 60

-- Define the total time
noncomputable def total_time : ℝ := 2

-- Define the average speed calculation
noncomputable def average_speed : ℝ := (speed_hour1 + speed_hour2) / total_time

-- Theorem statement
theorem car_average_speed :
  average_speed = 77.5 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Unfold the definitions of speed_hour1, speed_hour2, and total_time
  unfold speed_hour1 speed_hour2 total_time
  -- Simplify the arithmetic
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l449_44937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l449_44985

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point :
  ∃ (α : ℝ), (power_function α) (1/2) = Real.sqrt 2 / 2 ∧ 
  ∀ (x : ℝ), x ≥ 0 → (power_function α) x = Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l449_44985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_200_l449_44916

theorem closest_integer_to_cube_root_200 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (200 : ℝ) ^ (1/3)| ≤ |m - (200 : ℝ) ^ (1/3)| ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_200_l449_44916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l449_44967

/-- Given a line passing through (a, 0) that forms a triangle with the x-axis and y-axis in the first quadrant,
    with area T and angle θ with the positive x-axis, prove that the equation of the line is
    tan(θ)x - y - a*tan(θ) = 0, where a > 0 and θ ≠ 0. -/
theorem line_equation_proof (a T θ : ℝ) (h1 : a > 0) (h2 : θ ≠ 0) : 
  ∃ (x y : ℝ), Real.tan θ * x - y - a * Real.tan θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l449_44967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_score_le_seven_l449_44953

def total_balls : ℕ := 7
def red_balls : ℕ := 4
def black_balls : ℕ := 3
def drawn_balls : ℕ := 4
def red_score : ℕ := 1
def black_score : ℕ := 3

def score (r b : ℕ) : ℕ := r * red_score + b * black_score

theorem probability_score_le_seven :
  (Nat.choose red_balls drawn_balls * Nat.choose black_balls 0 +
   Nat.choose red_balls 3 * Nat.choose black_balls 1) /
  Nat.choose total_balls drawn_balls = 13 / 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_score_le_seven_l449_44953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l449_44907

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, 6)
  parallel a b → x = 4 :=
by
  intro h
  sorry

#check parallel_vectors_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l449_44907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_properties_l449_44923

/-- The volume of a tetrahedron with one edge length x and the remaining edge lengths 1 -/
noncomputable def F (x : ℝ) : ℝ := (1 / 12) * Real.sqrt (3 * x^2 - x^4)

/-- The domain of the function F -/
def F_domain : Set ℝ := {x : ℝ | 0 < x ∧ x < Real.sqrt 3}

theorem tetrahedron_volume_properties :
  ∃ (x_max : ℝ), x_max ∈ F_domain ∧
  (∀ x, x ∈ F_domain → F x ≤ F x_max) ∧
  (∃ x₁ x₂, x₁ ∈ F_domain ∧ x₂ ∈ F_domain ∧ x₁ < x₂ ∧ F x₁ > F x₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_properties_l449_44923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l449_44956

noncomputable def f (x : ℝ) := Real.cos (x + Real.pi / 6)

theorem f_properties :
  (∀ x, f (x - 2*Real.pi) = f x) ∧
  (∀ x, f (5*Real.pi/3 - x) = f (5*Real.pi/3 + x)) ∧
  (f (Real.pi/3 + Real.pi) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l449_44956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_sum_l449_44919

theorem tangent_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y - Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y - Real.cos y / Real.sin x = 3) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_sum_l449_44919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_zero_l449_44938

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 4

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := 4*x - y + 1 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem min_distance_is_zero :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
    ∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ → 
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄ ∧
      distance x₁ y₁ x₂ y₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_zero_l449_44938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_green_stones_l449_44943

/-- The number of stones in Brian's white and black stone collection -/
def total_wb_stones : ℕ := 100

/-- The number of white stones Brian has -/
def white_stones : ℕ := 60

/-- The number of grey stones Brian has -/
def grey_stones : ℕ := 40

/-- The number of green stones Brian has -/
def green_stones : ℕ := 27

/-- The ratio of white to black stones is equal to the ratio of grey to green stones -/
axiom ratio_equality : (white_stones : ℚ) / (total_wb_stones - white_stones) = (grey_stones : ℚ) / green_stones

/-- Brian has more white stones than black stones -/
axiom more_white_than_black : white_stones > total_wb_stones - white_stones

/-- Theorem stating that given the conditions, Brian has 27 green stones -/
theorem brian_green_stones : green_stones = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_green_stones_l449_44943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_triangle_l449_44971

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A configuration of 6 points in a plane -/
structure Configuration where
  points : Fin 6 → Point
  eight_unit_distances : ∃ (pairs : Finset (Fin 6 × Fin 6)), pairs.card = 8 ∧
    ∀ (p q : Fin 6), (p, q) ∈ pairs → distance (points p) (points q) = 1

/-- An equilateral triangle formed by three points -/
def isEquilateralTriangle (p q r : Point) : Prop :=
  distance p q = 1 ∧ distance q r = 1 ∧ distance r p = 1

/-- The main theorem -/
theorem exists_equilateral_triangle (config : Configuration) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    isEquilateralTriangle (config.points i) (config.points j) (config.points k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_triangle_l449_44971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l449_44927

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ y, ∃ x, f x = y) ∧   -- range of f is ℝ
  (¬ ∃ T > 0, ∀ x, f (x + T) = f x) -- f is not periodic
  := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l449_44927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_leaders_selected_l449_44910

def numTeams : ℕ := 4
def teamSizes : Fin numTeams → ℕ := ![6, 9, 7, 10]
def leadersPerTeam : ℕ := 2

theorem probability_both_leaders_selected :
  let probSelectTeam := 1 / numTeams
  let probBothLeadersFromTeam (n : ℕ) := 2 / (n * (n - 1))
  let totalProb := (Finset.univ.sum (fun i => probSelectTeam * probBothLeadersFromTeam (teamSizes i)))
  totalProb = 29 / 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_leaders_selected_l449_44910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l449_44941

noncomputable def a : ℝ × ℝ := (3, 1)
noncomputable def b : ℝ × ℝ := (-2, 4)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

theorem projection_a_onto_b :
  projection a b = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l449_44941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_cube_equation_l449_44924

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem no_solutions_for_cube_equation : ¬∃ (x y z : ℕ), x^3 + 2*y^3 + 4*z^3 = factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_cube_equation_l449_44924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l449_44959

/-- A function that returns true if a number is a positive three-digit integer with each digit greater than 4, ending in 5 or 0, and divisible by 5 -/
def validNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧
  (n % 10 = 5 ∨ n % 10 = 0) ∧
  (n / 100 > 4) ∧ ((n / 10) % 10 > 4) ∧ (n % 10 > 4) ∧
  n % 5 = 0

/-- The count of numbers satisfying the validNumber condition is 25 -/
theorem count_valid_numbers : 
  (Finset.filter (fun n => validNumber n = true) (Finset.range 1000)).card = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l449_44959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_2x1_minus_x2_l449_44935

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Define the function g as a shift of f
def g (x : ℝ) : ℝ := f (x + π / 12) + 1

-- Theorem statement
theorem max_value_of_2x1_minus_x2 :
  ∃ (x₁ x₂ : ℝ),
    x₁ ∈ Set.Icc (-2 * π) (2 * π) ∧
    x₂ ∈ Set.Icc (-2 * π) (2 * π) ∧
    g x₁ * g x₂ = 9 ∧
    ∀ (y₁ y₂ : ℝ),
      y₁ ∈ Set.Icc (-2 * π) (2 * π) →
      y₂ ∈ Set.Icc (-2 * π) (2 * π) →
      g y₁ * g y₂ = 9 →
      2 * y₁ - y₂ ≤ 2 * x₁ - x₂ ∧
      2 * x₁ - x₂ = 49 * π / 12 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_2x1_minus_x2_l449_44935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_12_not_combinable_with_sqrt_2_l449_44975

-- Define the square roots we're working with
noncomputable def sqrt_half : ℝ := Real.sqrt (1/2)
noncomputable def sqrt_8 : ℝ := Real.sqrt 8
noncomputable def sqrt_12 : ℝ := Real.sqrt 12
noncomputable def neg_sqrt_18 : ℝ := -Real.sqrt 18

-- Define what it means for a real number to be a rational multiple of √2
def is_rational_multiple_of_sqrt_2 (x : ℝ) : Prop :=
  ∃ q : ℚ, x = q * Real.sqrt 2

-- State the theorem
theorem sqrt_12_not_combinable_with_sqrt_2 :
  is_rational_multiple_of_sqrt_2 sqrt_half ∧
  is_rational_multiple_of_sqrt_2 sqrt_8 ∧
  is_rational_multiple_of_sqrt_2 neg_sqrt_18 ∧
  ¬ is_rational_multiple_of_sqrt_2 sqrt_12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_12_not_combinable_with_sqrt_2_l449_44975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_quadratic_l449_44944

/-- Given a quadratic function f(x) = x² + ax + b, if the tangent line
    at the point (0, b) is x + y = 1, then a = -1 and b = 1. -/
theorem tangent_line_quadratic (a b : ℝ) : 
  (∀ x, HasDerivAt (fun x ↦ x^2 + a*x + b) (2*x + a) x) →
  HasDerivAt (fun x ↦ x^2 + a*x + b) a 0 →
  (∀ x, x + (x^2 + a*x + b) = 1) →
  (a = -1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_quadratic_l449_44944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_foci_to_asymptotes_l449_44942

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

-- Define the foci
def focus (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Theorem statement
theorem distance_from_foci_to_asymptotes :
  ∀ (x_f y_f x_a y_a : ℝ),
    focus x_f y_f →
    asymptote x_a y_a →
    hyperbola x_f y_f →
    (abs (y_a - (Real.sqrt 3 / 3) * x_a) / Real.sqrt (1 + (Real.sqrt 3 / 3)^2)) = 1 :=
by
  sorry

#check distance_from_foci_to_asymptotes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_foci_to_asymptotes_l449_44942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_imaginary_z_purely_imaginary_l449_44976

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := (2 + Complex.I) * m^2 - 3 * m * (1 + Complex.I) - 2 * (1 - Complex.I)

/-- z is imaginary iff m ≠ 1 and m ≠ 2 -/
theorem z_imaginary (m : ℝ) : (∃ (y : ℝ), z m = y * Complex.I) ↔ m ≠ 1 ∧ m ≠ 2 := by sorry

/-- z is purely imaginary iff m = -1/2 -/
theorem z_purely_imaginary (m : ℝ) : (∃ (y : ℝ), y ≠ 0 ∧ z m = y * Complex.I) ↔ m = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_imaginary_z_purely_imaginary_l449_44976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_condition_necessary_not_sufficient_l449_44979

/-- Given two unit vectors a₁ and a₂ in ℝ², prove that a₁ = (√3/2, 1/2) is a necessary 
but not sufficient condition for a₁ + a₂ = (√3, 1) -/
theorem vector_condition_necessary_not_sufficient 
  (a₁ a₂ : ℝ × ℝ) 
  (h_unit_a₁ : ‖a₁‖ = 1) 
  (h_unit_a₂ : ‖a₂‖ = 1) : 
  (a₁ = (Real.sqrt 3 / 2, 1 / 2) → a₁ + a₂ = (Real.sqrt 3, 1)) ∧
  ¬(a₁ = (Real.sqrt 3 / 2, 1 / 2) ↔ a₁ + a₂ = (Real.sqrt 3, 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_condition_necessary_not_sufficient_l449_44979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_l449_44929

/-- Represents the cost and lifespan of a pair of shoes -/
structure ShoeOption where
  cost : ℝ
  lifespan : ℝ

/-- Calculates the average cost per year for a shoe option -/
noncomputable def averageCostPerYear (shoe : ShoeOption) : ℝ :=
  shoe.cost / shoe.lifespan

/-- Calculates the percentage increase between two values -/
noncomputable def percentageIncrease (original : ℝ) (new : ℝ) : ℝ :=
  (new - original) / original * 100

theorem shoe_cost_comparison : 
  let repairedShoes : ShoeOption := ⟨13.50, 1⟩
  let newShoes : ShoeOption := ⟨32.00, 2⟩
  let repairedCostPerYear := averageCostPerYear repairedShoes
  let newCostPerYear := averageCostPerYear newShoes
  abs (percentageIncrease repairedCostPerYear newCostPerYear - 18.52) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_l449_44929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_32_68_60_l449_44926

/-- The area of a triangle given its three sides using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 32, 68, and 60 is 960 square centimeters -/
theorem triangle_area_32_68_60 :
  triangle_area 32 68 60 = 960 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval triangle_area 32 68 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_32_68_60_l449_44926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l449_44966

/-- A regular polygon with perimeter 150 and side length 15 has 10 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_regular : p ≥ 3)
  (h_perimeter : perimeter = 150)
  (h_side_length : side_length = 15)
  (h_relation : perimeter = p * side_length) : p = 10 := by
  sorry

#check regular_polygon_sides

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l449_44966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l449_44947

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def focal_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

def focus1 (c : ℝ) : ℝ × ℝ := (-c, 0)
def focus2 (c : ℝ) : ℝ × ℝ := (c, 0)

def perpendicular_line (x₀ : ℝ) (x : ℝ) : Prop := x = x₀

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem ellipse_focus_distance :
  let a : ℝ := 2
  let b : ℝ := 1
  let c : ℝ := focal_distance a b
  let f₁ : ℝ × ℝ := focus1 c
  let f₂ : ℝ × ℝ := focus2 c
  ∀ x y : ℝ,
    ellipse x y →
    perpendicular_line f₁.1 x →
    distance (x, y) f₂ = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l449_44947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_l449_44948

/-- The planned number of masks to be produced -/
def total_masks : ℕ := 3000

/-- The number of hours ahead of schedule the task was completed -/
def hours_saved : ℕ := 5

/-- The planned production rate in masks per hour -/
def planned_rate : ℝ → ℝ := λ x ↦ x

/-- The actual production rate is twice the planned rate -/
def actual_rate : ℝ → ℝ := λ x ↦ 2 * x

/-- The theorem stating the relationship between the planned rate and the given conditions -/
theorem production_equation (x : ℝ) (h : x > 0) :
  (total_masks : ℝ) / (planned_rate x) - (total_masks : ℝ) / (actual_rate x) = hours_saved := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_equation_l449_44948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_perp_to_centers_AB_equation_perpendicular_bisector_equation_l449_44912

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define the intersection points A and B
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ
axiom A_on_circles : circle1 A.1 A.2 ∧ circle2 A.1 A.2
axiom B_on_circles : circle1 B.1 B.2 ∧ circle2 B.1 B.2

-- Define centers of circles
def C₁ : ℝ × ℝ := (1, 0)
def C₂ : ℝ × ℝ := (-1, 2)

-- Define line AB
def line_AB (x y : ℝ) : Prop := 4*x - 4*y + 1 = 0

-- Define perpendicular bisector of AB
def perp_bisector_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statements
theorem line_AB_perp_to_centers : 
  ∃ (k : ℝ), (C₂.1 - C₁.1) * (B.1 - A.1) + (C₂.2 - C₁.2) * (B.2 - A.2) = k * ((B.1 - A.1)^2 + (B.2 - A.2)^2) :=
by sorry

theorem AB_equation : 
  ∀ (x y : ℝ), (∃ t : ℝ, (x, y) = ((1-t)*A.1 + t*B.1, (1-t)*A.2 + t*B.2)) → line_AB x y :=
by sorry

theorem perpendicular_bisector_equation : 
  ∀ (x y : ℝ), perp_bisector_AB x y ↔ (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2)/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_perp_to_centers_AB_equation_perpendicular_bisector_equation_l449_44912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l449_44995

noncomputable def f (x : ℝ) := 7 * Real.sin (x - Real.pi / 6)

theorem f_monotone_increasing :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y := by
  intros x y h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l449_44995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_regular_polygons_are_isosceles_triangles_regular_polygons_can_have_more_than_three_sides_l449_44991

-- Define a polygon
structure Polygon where
  vertices : List (ℝ × ℝ)
  is_closed : vertices.head? = vertices.getLast?

-- Define properties of polygons
def Polygon.isRegular (p : Polygon) : Prop := sorry
def Polygon.isIsosceles (p : Polygon) : Prop := sorry
def Polygon.numSides (p : Polygon) : ℕ := p.vertices.length - 1

-- Theorem statement
theorem not_all_regular_polygons_are_isosceles_triangles :
  ∃ p : Polygon, p.isRegular ∧ ¬p.isIsosceles := by
  -- Proof sketch: Consider a regular square
  sorry

-- Additional theorem to show that regular polygons can have more than 3 sides
theorem regular_polygons_can_have_more_than_three_sides :
  ∃ p : Polygon, p.isRegular ∧ p.numSides > 3 := by
  -- Proof sketch: Again, consider a regular square
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_regular_polygons_are_isosceles_triangles_regular_polygons_can_have_more_than_three_sides_l449_44991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l449_44906

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus F₁ of the parabola
def F₁ : ℝ × ℝ := (0, 1)

-- Define F₂ as the point (0, -1)
def F₂ : ℝ × ℝ := (0, -1)

-- Define point A
def A : ℝ × ℝ := (2, 1)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b c : ℝ × ℝ) : ℝ :=
  (distance a c - distance a b) / (distance b c)

theorem hyperbola_eccentricity :
  parabola A.1 A.2 →
  eccentricity A F₁ F₂ = Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l449_44906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_b_range_l449_44909

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + b

-- State the theorem
theorem unique_zero_implies_b_range :
  (∃! x, x ∈ Set.Ioo 2 4 ∧ f b x = 0) →
  b ∈ Set.Ioo (-8) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_b_range_l449_44909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_slope_angle_l449_44954

/-- Parabola type representing y² = 3x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 3 * x

/-- Focus of the parabola -/
noncomputable def focus : ℝ × ℝ := (3/4, 0)

/-- Point on the parabola -/
def point_on_parabola (p : Parabola) : ℝ × ℝ := (p.x, p.y)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Slope angle of a line -/
noncomputable def slope_angle (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.arctan ((p2.2 - p1.2) / (p2.1 - p1.1))

theorem parabola_focus_slope_angle (p : Parabola) :
  let A := point_on_parabola p
  distance A focus = 3 →
  slope_angle A focus = π/3 ∨ slope_angle A focus = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_slope_angle_l449_44954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_faces_same_edge_count_l449_44989

-- Define a structure for a convex polyhedron
structure ConvexPolyhedron where
  faces : Finset (Finset ℕ)
  edges : Finset (Fin 2 → ℕ)
  convex : Bool
  min_edges : ∀ f ∈ faces, f.card ≥ 3

-- Theorem statement
theorem two_faces_same_edge_count (P : ConvexPolyhedron) : 
  ∃ f₁ f₂, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.card = f₂.card :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_faces_same_edge_count_l449_44989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_MH_equals_MK_sometimes_but_not_always_l449_44913

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of the geometric problem -/
structure GeometricConfig where
  H : Point
  K : Point
  B : Point
  C : Point
  M : Point
  θ : ℝ
  HK : Line
  BC : Line

/-- Conditions of the geometric problem -/
def satisfiesConditions (config : GeometricConfig) : Prop :=
  -- HK and BC intersect at angle θ
  ∃ (intersectionPoint : Point),
    config.HK.a * intersectionPoint.x + config.HK.b * intersectionPoint.y + config.HK.c = 0 ∧
    config.BC.a * intersectionPoint.x + config.BC.b * intersectionPoint.y + config.BC.c = 0 ∧
  -- M is the midpoint of BC
  config.M.x = (config.B.x + config.C.x) / 2 ∧
  config.M.y = (config.B.y + config.C.y) / 2 ∧
  -- BH and CK are perpendicular to HK
  config.HK.a * (config.B.y - config.H.y) = config.HK.b * (config.H.x - config.B.x) ∧
  config.HK.a * (config.C.y - config.K.y) = config.HK.b * (config.K.x - config.C.x) ∧
  -- θ is not 90°
  config.θ ≠ 90

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem to prove -/
theorem MH_equals_MK_sometimes_but_not_always :
  ∃ (config1 config2 : GeometricConfig),
    satisfiesConditions config1 ∧
    satisfiesConditions config2 ∧
    distance config1.M config1.H = distance config1.M config1.K ∧
    distance config2.M config2.H ≠ distance config2.M config2.K := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_MH_equals_MK_sometimes_but_not_always_l449_44913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_square_l449_44946

/-- Square structure with basic properties -/
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

/-- Inner square relation -/
def Square.isInnerSquareOf (inner outer : Square) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ d < outer.side / 2 ∧ inner.side = outer.side - 2 * d

/-- Representing points on the square -/
noncomputable def Square.AX (s : Square) : ℝ := s.side / 4
noncomputable def Square.BW (s : Square) : ℝ := s.side / 4
noncomputable def Square.CZ (s : Square) : ℝ := s.side / 4
noncomputable def Square.DY (s : Square) : ℝ := s.side / 4

/-- Given a square ABCD with area 64 and AX = BW = CZ = DY = 2,
    prove that the area of square WXYZ is 40 -/
theorem area_of_inner_square (ABCD : Square) (WXYZ : Square) 
  (h1 : ABCD.area = 64)
  (h2 : ABCD.AX = 2 ∧ ABCD.BW = 2 ∧ ABCD.CZ = 2 ∧ ABCD.DY = 2)
  (h3 : WXYZ.isInnerSquareOf ABCD) : WXYZ.area = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_square_l449_44946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_small_chessboard_exists_l449_44940

/-- Represents a chessboard -/
structure Chessboard where
  size : ℕ

/-- Represents a small chessboard placed on the large chessboard -/
structure SmallChessboard where
  x : ℕ
  y : ℕ

/-- Checks if two small chessboards overlap -/
def overlap (a b : SmallChessboard) : Prop :=
  (a.x ≤ b.x + 2 ∧ b.x ≤ a.x + 2) ∧ (a.y ≤ b.y + 2 ∧ b.y ≤ a.y + 2)

/-- The main theorem -/
theorem additional_small_chessboard_exists 
  (board : Chessboard)
  (small_boards : Finset SmallChessboard)
  (h1 : board.size = 48)
  (h2 : small_boards.card = 99)
  (h3 : ∀ a b, a ∈ small_boards → b ∈ small_boards → a ≠ b → ¬overlap a b)
  (h4 : ∀ s, s ∈ small_boards → s.x < 46 ∧ s.y < 46) :
  ∃ new_board : SmallChessboard, 
    (new_board.x < 46 ∧ new_board.y < 46) ∧
    ∀ s, s ∈ small_boards → ¬overlap new_board s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_small_chessboard_exists_l449_44940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l449_44982

-- Define the cone
structure Cone where
  r : ℝ  -- radius of the base
  l : ℝ  -- slant height
  h : ℝ  -- height

-- Define the properties of the cone
def is_valid_cone (c : Cone) : Prop :=
  c.l = 2 * c.r ∧                    -- lateral surface is a semicircle
  c.h = c.r * Real.sqrt 3 ∧          -- height relation
  (1/3) * Real.pi * c.r^2 * c.h = 9 * Real.sqrt 3 * Real.pi  -- volume condition

-- Theorem statement
theorem cone_surface_area (c : Cone) (h : is_valid_cone c) :
  Real.pi * c.r^2 + Real.pi * c.r * c.l = 27 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l449_44982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_subjects_l449_44984

theorem basketball_team_subjects (total_players physics_players both_subjects : ℕ) : 
  total_players = 25 →
  physics_players = 10 →
  both_subjects = 5 →
  (∀ player, player ≤ total_players → 
    (player ≤ physics_players ∨ 
     (player > physics_players - both_subjects ∧ player ≤ total_players))) →
  (Finset.filter (λ s : ℕ ↦ s > physics_players - both_subjects ∧ s ≤ total_players) (Finset.range (total_players + 1))).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_subjects_l449_44984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_K_equals_one_plus_two_ln_two_l449_44936

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- Define f_K
noncomputable def f_K (K : ℝ) (x : ℝ) : ℝ :=
  if f x ≤ K then K else f x

-- State the theorem
theorem integral_f_K_equals_one_plus_two_ln_two :
  let K := (1 : ℝ)
  ∫ x in (1/4)..(2), f_K K x = 1 + 2 * Real.log 2 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_K_equals_one_plus_two_ln_two_l449_44936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_formula_product_form_l449_44930

theorem quadratic_formula_product_form (a b c : ℝ) (h : b^2 ≥ 4*a*c) :
  ∃ θ : ℝ, 
    (a*c ≥ 0 → (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) = -b/a * Real.sin (θ/2)^2) ∧
    (a*c < 0 → (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) = (b * Real.sin (θ/2)^2) / (a * Real.cos θ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_formula_product_form_l449_44930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_annoyingness_l449_44949

/-- The annoyingness of a permutation -/
def annoyingness {n : ℕ} (p : Equiv.Perm (Fin n)) : ℕ :=
  sorry

/-- The expected value of the annoyingness of a random permutation -/
theorem expected_annoyingness (n : ℕ) : 
  Finset.sum (Finset.univ : Finset (Equiv.Perm (Fin n))) (fun p => (annoyingness p : ℚ)) / 
  Finset.card (Finset.univ : Finset (Equiv.Perm (Fin n))) = (n + 1) / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_annoyingness_l449_44949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l449_44952

theorem inequality_condition (a : ℝ) : 
  (∀ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) → 
    Real.sqrt 2 * (2 * a + 3) * Real.cos (θ - Real.pi / 4) + 
    6 / (Real.sin θ + Real.cos θ) - 
    2 * Real.sin (2 * θ) < 3 * a + 6) ↔ 
  a > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l449_44952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_ABNM_l449_44905

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of a point being in the third quadrant -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- The area of quadrilateral ABNM for a given point P on the ellipse -/
noncomputable def area_ABNM (x₀ y₀ : ℝ) : ℝ := 
  (2 + x₀/(y₀ - 1)) * (1 + 2*y₀/(x₀ - 2)) / 2

theorem constant_area_ABNM :
  ∀ x₀ y₀ : ℝ, 
  ellipse x₀ y₀ → third_quadrant x₀ y₀ → 
  area_ABNM x₀ y₀ = 2 := by
  sorry

#check constant_area_ABNM

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_area_ABNM_l449_44905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_C_range_PA_PB_l449_44993

noncomputable section

-- Define the curve C
def curve_C (a b : ℝ) := {p : ℝ × ℝ | ∃ α : ℝ, p.1 = a * Real.cos α ∧ p.2 = b * Real.sin α}

-- Define point M
def point_M : ℝ × ℝ := (1, Real.sqrt 2 / 2)

-- Define point P in polar coordinates
def point_P_polar : ℝ × ℝ := (Real.sqrt 2, Real.pi / 2)

-- Convert P to Cartesian coordinates
def point_P : ℝ × ℝ := (0, Real.sqrt 2)

-- Theorem for the standard equation of curve C
theorem standard_equation_C (a b : ℝ) (h1 : point_M ∈ curve_C a b) (h2 : ∃ α, α = Real.pi / 4 ∧ point_M = (a * Real.cos α, b * Real.sin α)) :
  curve_C a b = {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1} := by sorry

-- Theorem for the range of |PA| * |PB|
theorem range_PA_PB :
  ∀ A B : ℝ × ℝ, A ∈ curve_C (Real.sqrt 2) 1 → B ∈ curve_C (Real.sqrt 2) 1 →
  ∃ l : Set (ℝ × ℝ), point_P ∈ l ∧ A ∈ l ∧ B ∈ l →
  1 ≤ ‖A - point_P‖ * ‖B - point_P‖ ∧ ‖A - point_P‖ * ‖B - point_P‖ ≤ 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_C_range_PA_PB_l449_44993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_knight_moves_l449_44974

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a knight's move -/
def is_knight_move (p1 p2 : Position) : Prop :=
  (Int.natAbs (p1.x.val - p2.x.val) == 2 ∧ Int.natAbs (p1.y.val - p2.y.val) == 1) ∨
  (Int.natAbs (p1.x.val - p2.x.val) == 1 ∧ Int.natAbs (p1.y.val - p2.y.val) == 2)

/-- Minimum number of knight's moves between two positions -/
noncomputable def min_knight_moves (p1 p2 : Position) : ℕ := sorry

/-- Theorem: Given 17 marked squares on an 8x8 chessboard, 
    there exist at least two marked squares such that 
    the minimum number of knight's moves between them is at least 3 -/
theorem chessboard_knight_moves :
  ∀ (marked_squares : Finset Position),
  marked_squares.card = 17 →
  ∃ p1 p2 : Position, p1 ∈ marked_squares ∧ p2 ∈ marked_squares ∧ p1 ≠ p2 ∧ min_knight_moves p1 p2 ≥ 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_knight_moves_l449_44974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_to_return_to_sixteen_l449_44965

noncomputable def reciprocal (x : ℝ) : ℝ := 1 / x
noncomputable def square (x : ℝ) : ℝ := x ^ 2

noncomputable def calc_sequence (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => if n % 2 = 0 then reciprocal (calc_sequence n x) else square (calc_sequence n x)

theorem min_operations_to_return_to_sixteen :
  (∀ k < 6, calc_sequence k 16 ≠ 16) ∧ calc_sequence 6 16 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_operations_to_return_to_sixteen_l449_44965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bret_in_three_l449_44961

-- Define the type for people
inductive Person : Type
  | Abby : Person
  | Bret : Person
  | Carl : Person
  | Dana : Person

-- Define the type for seat numbers
inductive Seat : Type
  | one : Seat
  | two : Seat
  | three : Seat
  | four : Seat

-- Define the seating arrangement
def seating : Person → Seat := sorry

-- Carl is in seat #2
axiom carl_in_two : seating Person.Carl = Seat.two

-- Dana is not next to Abby
axiom dana_not_next_to_abby :
  ¬((seating Person.Dana = Seat.one ∧ seating Person.Abby = Seat.two) ∨
    (seating Person.Dana = Seat.two ∧ seating Person.Abby = Seat.one) ∨
    (seating Person.Dana = Seat.two ∧ seating Person.Abby = Seat.three) ∨
    (seating Person.Dana = Seat.three ∧ seating Person.Abby = Seat.two) ∨
    (seating Person.Dana = Seat.three ∧ seating Person.Abby = Seat.four) ∨
    (seating Person.Dana = Seat.four ∧ seating Person.Abby = Seat.three))

-- Carl is not between Dana and Abby
axiom carl_not_between :
  ¬((seating Person.Dana = Seat.one ∧ seating Person.Abby = Seat.three) ∨
    (seating Person.Dana = Seat.three ∧ seating Person.Abby = Seat.one))

-- All seats are occupied
axiom all_seats_occupied :
  ∃ (p1 p2 p3 p4 : Person),
    seating p1 = Seat.one ∧
    seating p2 = Seat.two ∧
    seating p3 = Seat.three ∧
    seating p4 = Seat.four

-- Theorem: Bret is in seat #3
theorem bret_in_three : seating Person.Bret = Seat.three := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bret_in_three_l449_44961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l449_44957

noncomputable def f (x : ℝ) : ℝ := 
  Matrix.det !![Real.sin x, Real.cos x; Real.cos x, Real.sin x]

theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ 
  T = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l449_44957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_p_condition_q_condition_l449_44986

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0

def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (m - 6) - y^2 / (m + 3) = 1

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (¬(p m ∧ q m)) → m ≥ -3 := by
  sorry

-- Helper theorems
theorem p_condition (m : ℝ) : p m → m < 5 := by
  sorry

theorem q_condition (m : ℝ) : q m → (m < -3 ∨ m > 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_p_condition_q_condition_l449_44986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l449_44997

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For a parabola y^2 = 4x, if a line through the focus intersects
    the parabola at P and Q with x₁ + x₂ = 6, then |PQ| = 8 -/
theorem parabola_intersection_distance 
  (P Q : Point) 
  (hP : P ∈ Parabola) 
  (hQ : Q ∈ Parabola) 
  (h_line : ∃ (t : ℝ), P = ⟨(1 - t) * focus.x + t * Q.x, (1 - t) * focus.y + t * Q.y⟩) 
  (h_sum : P.x + Q.x = 6) :
  distance P Q = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l449_44997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_2_3_6_is_obtuse_l449_44973

theorem triangle_with_angle_ratio_2_3_6_is_obtuse :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 2 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 6 →
  ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 6*x ∧ c > 90 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_2_3_6_is_obtuse_l449_44973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l449_44903

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def H (x : ℝ) : ℝ := ((Real.exp 1 + 1) / 2) * x^2 - 2 * Real.exp 1 * x * Real.log x

theorem problem_statement :
  (∀ x > 0, f 1 x ≤ x - 1) ∧
  (∀ a > 0, (∀ x > 0, f a x ≤ x) → a ≤ Real.exp 1) ∧
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (H x₁ - H x₂) / (x₁ - x₂) > -Real.exp 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l449_44903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l449_44934

theorem equation_roots (n : ℕ+) :
  ∃ (roots : Finset ℂ), 
    (∀ x ∈ roots, x^(4*n.val) - 4*x^n.val - 1 = 0) ∧ 
    (roots.card = 4*n.val) ∧
    (∀ x ∈ roots, x.re > 0 → x.im ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l449_44934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_with_discounts_l449_44962

def puppy_prices : List Float := [72, 78]
def kitten_prices : List Float := [48, 52]
def parakeet_prices : List Float := [10, 12, 14]

def puppy_discount : Float := 0.05
def kitten_discount : Float := 0.10

def apply_discount (prices : List Float) (discount : Float) : Float :=
  (prices.sum) * (1 - discount)

noncomputable def parakeet_discounted_price (prices : List Float) : Float :=
  prices.sum - (prices.minimum?).getD 0 / 2

theorem total_cost_with_discounts : 
  apply_discount puppy_prices puppy_discount +
  apply_discount kitten_prices kitten_discount +
  parakeet_discounted_price parakeet_prices = 263.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_with_discounts_l449_44962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lipschitz_distribution_implies_bounded_density_l449_44972

-- Define the distribution function and Lipschitz condition
def is_lipschitz_distribution_function (F : ℝ → ℝ) (L : ℝ) : Prop :=
  L > 0 ∧ ∀ x y : ℝ, |F x - F y| ≤ L * |x - y|

-- Define the theorem
theorem lipschitz_distribution_implies_bounded_density
  (F : ℝ → ℝ) (L : ℝ) (hF : is_lipschitz_distribution_function F L) :
  ∃ f : ℝ → ℝ, (∀ x, F x = ∫ y in Set.Iic x, f y) ∧
               (∀ᵐ x, f x ≤ L) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lipschitz_distribution_implies_bounded_density_l449_44972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_approx_l449_44922

/-- The volume of lemon juice in a cylindrical glass -/
noncomputable def lemon_juice_volume (glass_height : ℝ) (glass_diameter : ℝ) (fill_ratio : ℝ) (lemon_ratio : ℝ) : ℝ :=
  let glass_radius : ℝ := glass_diameter / 2
  let liquid_height : ℝ := glass_height * fill_ratio
  let total_volume : ℝ := Real.pi * glass_radius^2 * liquid_height
  total_volume * lemon_ratio

/-- Theorem stating the volume of lemon juice in the glass -/
theorem lemon_juice_volume_approx :
  ∃ ε > 0, |lemon_juice_volume 9 3 (1/3) (1/3) - 7.07| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemon_juice_volume_approx_l449_44922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sine_curve_l449_44964

-- Define the function for the curve
noncomputable def f (x : ℝ) := Real.sin x

-- Define the interval
noncomputable def a : ℝ := 0
noncomputable def b : ℝ := Real.pi

-- State the theorem
theorem area_under_sine_curve : 
  ∫ x in a..b, f x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_sine_curve_l449_44964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_after_operations_l449_44925

/-- Represents the operation that can be performed on the board -/
def boardOperation (a b : ℕ) : (ℕ × ℕ) :=
  (a^2 - 2011*b^2, a*b)

/-- Represents the state of the board after any number of operations -/
def BoardState : Type := List ℕ

/-- Checks if a list contains 10 consecutive natural numbers -/
def containsConsecutiveNumbers (l : List ℕ) : Prop :=
  ∃ n : ℕ, l = (List.range 10).map (· + n)

/-- Represents a single operation on the board -/
def performOperation (state : BoardState) (i j : ℕ) : BoardState :=
  sorry

/-- Represents any number of operations on the board -/
def performOperations (initial : BoardState) : BoardState :=
  sorry

theorem no_consecutive_after_operations :
  ∀ (initial : BoardState),
    (containsConsecutiveNumbers initial) →
    ¬∃ (final : BoardState),
      (performOperations initial = final) ∧
      (containsConsecutiveNumbers final) :=
by
  sorry

#check no_consecutive_after_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_after_operations_l449_44925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_sixteen_l449_44983

theorem series_sum_equals_sixteen (x : ℝ) :
  (∑' n, (2 * n + 1) * x^n) = 16 → x = (33 - Real.sqrt 129) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_sixteen_l449_44983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l449_44901

theorem quartic_equation_solutions : 
  {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -Complex.I * Real.sqrt 2, Complex.I * Real.sqrt 2, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l449_44901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_of_equation_l449_44932

theorem smallest_root_of_equation (x : ℝ) :
  (∀ y < 0, Real.sin (π * y) + Real.tan χ ≠ y + y^3) ∧
  Real.sin (π * 0) + Real.tan χ = 0 + 0^3 →
  0 ≤ x ∧ Real.sin (π * x) + Real.tan χ = x + x^3 →
  x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_root_of_equation_l449_44932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l449_44918

noncomputable def f (x : ℝ) : ℝ := -x * Real.exp (abs x)

noncomputable def f' (x : ℝ) : ℝ := 
  if x < 0 then (x - 1) * Real.exp (-x)
  else -(x + 1) * Real.exp x

theorem f_satisfies_conditions :
  (∀ x : ℝ, f x + f (-x) = 0) ∧
  (∀ x : ℝ, HasDerivAt f (f' x) x ∧ f' x ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l449_44918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_t_value_l449_44928

/-- Given two vectors a and b in ℝ³, where a is (2, -1, 1) and b is (t, 1, -1) for some real t,
    if a is parallel to b, then t = -2. -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : Fin 3 → ℝ := ![2, -1, 1]
  let b : Fin 3 → ℝ := ![t, 1, -1]
  (∃ (k : ℝ), a = k • b) → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_t_value_l449_44928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_implies_a_range_l449_44900

theorem function_zero_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ 4^x - 2^x - a = 0) → a ∈ Set.Icc (-1/4 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_implies_a_range_l449_44900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l449_44988

theorem trig_identities (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin (π / 4 + α) = -Real.sqrt 10 / 10 ∧ 
  Real.cos (5 * π / 6 - 2 * α) = -(4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l449_44988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l449_44960

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * m^2 - 2 * m) * x^(m - 1/2)
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := -3^x + t

-- Define the interval [1, 4]
def I : Set ℝ := Set.Icc 1 4

-- State the theorem
theorem problem_solution :
  ∃ (m : ℝ), (∀ x > 0, Monotone (f m)) ∧
  ∃ (t : ℝ), t ∈ Set.Icc 5 82 ∧
  (Set.range (f m ∘ (fun x => x)) ⊆ Set.range (g t ∘ (fun x => x))) ∧
  (Set.range (f m ∘ (fun x => x)) ≠ Set.range (g t ∘ (fun x => x))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l449_44960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_force_calculation_wrench_force_example_l449_44980

/-- Given a force and length that satisfy the inverse relationship, 
    calculate the force for a new length -/
noncomputable def calculate_force (F1 : ℝ) (L1 : ℝ) (L2 : ℝ) : ℝ :=
  (F1 * L1) / L2

theorem inverse_force_calculation (F1 L1 L2 : ℝ) 
  (h1 : F1 > 0) (h2 : L1 > 0) (h3 : L2 > 0) :
  calculate_force F1 L1 L2 * L2 = F1 * L1 :=
by sorry

theorem wrench_force_example :
  calculate_force 240 12 20 = 144 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_force_calculation_wrench_force_example_l449_44980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l449_44958

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, a*x^2 - (1+a)*x + b ≥ 0 ↔ -1/5 ≤ x ∧ x ≤ 1) → 
  a + b = -4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l449_44958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_decomposition_l449_44996

theorem tan_22_5_decomposition :
  ∃ (x y z w : ℕ),
    (Real.tan (22.5 * Real.pi / 180) = Real.sqrt (x : ℝ) - Real.sqrt (y : ℝ) + Real.sqrt (z : ℝ) - (w : ℝ)) ∧
    (x ≥ y) ∧ (y ≥ z) ∧ (z ≥ w) ∧
    (x > 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ (w > 0) ∧
    (x + y + z + w = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_decomposition_l449_44996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l449_44981

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f2x1 : Set ℝ := Set.Icc (-1) 0

-- State the theorem
theorem domain_of_composite_function :
  (∀ x ∈ domain_f2x1, ∃ y, f (2*x+1) = y) →
  {x : ℝ | ∃ y, (f (x+1))/(2^x - 1) = y} = Set.Ico (-2) 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composite_function_l449_44981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_eleven_average_l449_44915

theorem last_eleven_average (numbers : List ℝ) : 
  numbers.length = 21 →
  numbers.sum / 21 = 44 →
  (numbers.take 11).sum / 11 = 48 →
  numbers.get? 10 = some 55 →
  (numbers.drop 10).sum / 11 = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_eleven_average_l449_44915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_increase_year_l449_44939

def years : List Nat := [1998, 2000, 2002, 2004, 2006]
def profits : List Nat := [20, 40, 70, 80, 100]

def profit_increases : List Nat :=
  List.zipWith (fun a b => b - a) profits (List.tail profits)

theorem max_profit_increase_year :
  (List.argmax id profit_increases).isSome →
  (List.argmax id profit_increases).get! = 1 →
  years[2]! = 2002 ∧
  ∀ i, i < profit_increases.length →
    profit_increases[i]! ≤ profit_increases[1]! := by
  sorry

#eval profit_increases
#eval List.argmax id profit_increases

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_increase_year_l449_44939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_three_element_set_l449_44908

theorem proper_subsets_of_three_element_set :
  let S : Finset Nat := {1, 2, 3}
  (Finset.powerset S).erase S |>.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_three_element_set_l449_44908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_time_l449_44999

/-- Represents a point on the highway -/
structure Point where
  position : ℝ

/-- Represents a bus stop -/
structure BusStop where
  point : Point

/-- Represents the bus journey -/
structure BusJourney where
  speed : ℝ
  stopA : BusStop
  stopB : BusStop
  stopC : BusStop
  firstSpecialPoint : Point
  secondSpecialPoint : Point
  timeFromSecondToB : ℝ
  stopTimeAtB : ℝ

/-- The main theorem to prove -/
theorem bus_journey_time (journey : BusJourney) : 
  journey.stopA.point.position < journey.stopB.point.position →
  journey.stopB.point.position < journey.stopC.point.position →
  journey.speed > 0 →
  journey.timeFromSecondToB = 25 →
  journey.stopTimeAtB = 5 →
  (abs (journey.firstSpecialPoint.position - journey.stopC.point.position) = 
   abs (journey.firstSpecialPoint.position - journey.stopA.point.position) + 
   abs (journey.firstSpecialPoint.position - journey.stopB.point.position)) →
  (abs (journey.secondSpecialPoint.position - journey.stopC.point.position) = 
   abs (journey.secondSpecialPoint.position - journey.stopA.point.position) + 
   abs (journey.secondSpecialPoint.position - journey.stopB.point.position)) →
  (journey.stopC.point.position - journey.stopA.point.position) / journey.speed + journey.stopTimeAtB = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_time_l449_44999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_squares_existence_l449_44990

theorem polynomial_squares_existence :
  ∃ (P Q : Polynomial ℤ) (R S T : Polynomial ℤ),
    P - Q = R^2 ∧
    P = S^2 ∧
    P + Q = T^2 ∧
    ¬∃ (c : ℤ), Q = c • P :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_squares_existence_l449_44990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_sum_l449_44969

-- Define the piecewise function f
noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 5 then c * x + d else 10 - 2 * x

-- State the theorem
theorem function_property_implies_sum (c d : ℝ) :
  (∀ x, f c d (f c d x) = x) → c + d = 6.5 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_implies_sum_l449_44969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l449_44955

/-- A geometric sequence with first term a and common ratio q -/
noncomputable def geometricSequence (a q : ℝ) : ℕ → ℝ := fun n ↦ a * q ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a : ℝ) :
  ∃ q : ℝ, (geometricSum a q 3 + 4 * geometricSum a q 2 + a = 0) ∧ (q = -2 ∨ q = -3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l449_44955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_minor_axis_equals_focal_length_l449_44917

/-- Eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Focal length of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

theorem ellipse_eccentricity_when_minor_axis_equals_focal_length 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = focal_length a b) :
  eccentricity a b = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_minor_axis_equals_focal_length_l449_44917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l449_44921

variable (a b : ℝ × ℝ)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos (dot_product v w / (magnitude v * magnitude w))

theorem angle_between_vectors (h1 : magnitude a = 1) 
  (h2 : magnitude b = Real.sqrt 2) 
  (h3 : magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = Real.sqrt 5) : 
  angle a b = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l449_44921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smartest_brain_theorem_l449_44950

/-- Represents the number of winners in the "Smartest Brain" competition -/
def smartest_brain_winners : ℕ → Prop := sorry

/-- Represents the total number of winners -/
def total_winners : ℕ := 25

/-- Represents the condition that the number of winners in the "Comprehensive Ability Competition" 
    is not less than 5 times the number of winners in the "Smartest Brain" competition -/
def comprehensive_condition (x : ℕ) : Prop :=
  total_winners - x ≥ 5 * x

/-- Represents the prize amount for each winner in the "Smartest Brain" competition -/
def smartest_brain_prize : ℕ := 15

/-- Represents the prize amount for each winner in the "Comprehensive Ability Competition" -/
def comprehensive_prize : ℕ := 30

/-- Calculates the total prize amount given the number of "Smartest Brain" winners -/
def total_prize (x : ℕ) : ℕ :=
  smartest_brain_prize * x + comprehensive_prize * (total_winners - x)

/-- States that the maximum number of winners in the "Smartest Brain" competition is 4,
    and that the minimum total prize amount is 690 yuan when there are 4 winners 
    in the "Smartest Brain" competition -/
theorem smartest_brain_theorem :
  (∀ x, smartest_brain_winners x → x ≤ 4) ∧
  (∀ x, smartest_brain_winners x → total_prize x ≥ 690) ∧
  (smartest_brain_winners 4 ∧ total_prize 4 = 690) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smartest_brain_theorem_l449_44950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l449_44992

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*x - 2 else 2^(-abs (1-x)) - 2

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := abs (a - 1) * Real.cos x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, f x₁ ≤ g a x₂) → a ∈ Set.Icc 0 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l449_44992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_cos_between_zero_and_half_l449_44963

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi * x / 2)

-- Define the event
def E : Set ℝ := {x ∈ I | 0 < f x ∧ f x < 1/2}

-- State the theorem
theorem probability_cos_between_zero_and_half :
  MeasureTheory.volume E / MeasureTheory.volume I = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_cos_between_zero_and_half_l449_44963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_l449_44968

/-- Given a function h such that h(5x-2) = 3x + 10 for all x,
    prove that the unique solution to h(x) = x is x = 28. -/
theorem unique_fixed_point_of_h :
  ∀ h : ℝ → ℝ, (∀ x, h (5 * x - 2) = 3 * x + 10) →
  (∃! x, h x = x ∧ x = 28) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_l449_44968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_is_2030_l449_44994

def sequenceNum : Fin 5 → ℤ
  | 0 => 1370
  | 1 => 1310
  | 2 => 1070
  | 3 => 2030
  | 4 => -6430

theorem fourth_number_is_2030 :
  (∀ i : Fin 3, sequenceNum (i + 1) - sequenceNum i = 4 * (sequenceNum i - sequenceNum (i - 1))) →
  sequenceNum 3 = 2030 := by
  sorry

#eval sequenceNum 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_is_2030_l449_44994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_to_c_l449_44987

open Real MeasureTheory

-- Define the function f and constant c
variable (f : ℝ → ℝ) (c : ℝ)

-- State the conditions
axiom c_gt_one : c > 1
axiom f_symmetry : ∀ (x : ℝ), x > 0 → f x = f (c / x)
axiom integral_sqrt_c : ∫ x in Set.Icc 1 (Real.sqrt c), f x / x = 3

-- State the theorem
theorem integral_to_c :
  ∫ x in Set.Icc 1 c, f x / x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_to_c_l449_44987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l449_44998

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 3 * x + 1

theorem range_of_a (a : ℝ) : 
  (f (6 - a^2) > f (5 * a)) ↔ (-6 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l449_44998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_rings_l449_44977

theorem tree_rings (fat_rings : ℕ) : 
  (∀ (group : ℕ), group + fat_rings + 4 = group + fat_rings + 4) →
  (first_tree_groups : ℕ) →
  (second_tree_groups : ℕ) →
  (first_tree_groups = 70) →
  (second_tree_groups = 40) →
  (first_tree_groups * (fat_rings + 4) = second_tree_groups * (fat_rings + 4) + 180) →
  fat_rings = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_rings_l449_44977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_bounds_triangle_expression_range_attainable_l449_44931

/-- A triangle with sides a ≤ b ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_b : a ≤ b
  b_le_c : b ≤ c
  triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a

/-- The expression (a+b+c)^2 / (b*c) for a triangle -/
noncomputable def triangle_expression (t : Triangle) : ℝ :=
  (t.a + t.b + t.c)^2 / (t.b * t.c)

theorem triangle_expression_bounds (t : Triangle) :
  4 < triangle_expression t ∧ triangle_expression t ≤ 9 := by
  sorry

theorem triangle_expression_range_attainable :
  ∀ x, 4 < x ∧ x ≤ 9 → ∃ t : Triangle, triangle_expression t = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_bounds_triangle_expression_range_attainable_l449_44931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l449_44920

def sequence_property (s : List ℕ) : Prop :=
  s.length = 8 ∧
  (∀ i, i < 6 → s.get! i + s.get! (i+1) + s.get! (i+2) = 50) ∧
  s.head? = some 11 ∧
  s.getLast? = some 12

theorem unique_sequence : 
  ∀ s : List ℕ, sequence_property s → s = [11, 12, 27, 11, 12, 27, 11, 12] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l449_44920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_relations_and_marriages_l449_44904

structure Person where
  name : String
  age : ℕ

inductive Married : Person → Person → Prop where
  | pair : (p1 p2 : Person) → Married p1 p2

def is_married (p1 p2 : Person) : Prop := Married p1 p2 ∨ Married p2 p1

theorem age_relations_and_marriages 
  (andrew boris svetlana larisa : Person)
  (h1 : is_married andrew svetlana ∨ is_married andrew larisa)
  (h2 : is_married boris svetlana ∨ is_married boris larisa)
  (h3 : ¬(is_married andrew boris ∨ is_married svetlana larisa))
  (h4 : ∃ p, p ∈ [andrew, boris] ∧ 
       (is_married p larisa ∧ ∀ q, q ∈ [andrew, boris, svetlana, larisa] → p.age ≥ q.age))
  (h5 : andrew.age < svetlana.age ∧ andrew.age > larisa.age) :
  (larisa.age < andrew.age ∧ larisa.age < svetlana.age ∧ larisa.age < boris.age) ∧ 
  is_married boris larisa ∧ 
  ¬is_married boris svetlana := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_relations_and_marriages_l449_44904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_condition_l449_44978

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

theorem sin_symmetry_condition (φ : ℝ) :
  (φ = -π/6 → is_symmetric_about (fun x ↦ Real.sin (2*x - φ)) (π/6)) ∧
  (∃ ψ : ℝ, ψ ≠ -π/6 ∧ is_symmetric_about (fun x ↦ Real.sin (2*x - ψ)) (π/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_symmetry_condition_l449_44978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l449_44970

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  f a 2 = Real.sqrt 2 / 2 → f a 4 = 1 / 2 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l449_44970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_in_range_l449_44902

noncomputable section

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x * |x + a| - 5 else a / x

-- State the theorem
theorem f_monotonic_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-3) (-2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_iff_a_in_range_l449_44902
