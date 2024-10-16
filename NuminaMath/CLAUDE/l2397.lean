import Mathlib

namespace NUMINAMATH_CALUDE_polar_to_rectangular_transform_l2397_239729

/-- Given a point with rectangular coordinates (8, 6) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r³, 3θ) has rectangular coordinates (480, 9360) -/
theorem polar_to_rectangular_transform (r θ : Real) : 
  (r * Real.cos θ = 8 ∧ r * Real.sin θ = 6) → 
  (r^3 * Real.cos (3 * θ) = 480 ∧ r^3 * Real.sin (3 * θ) = 9360) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transform_l2397_239729


namespace NUMINAMATH_CALUDE_george_monthly_income_l2397_239743

def monthly_income : ℝ := 240

theorem george_monthly_income :
  let half_income := monthly_income / 2
  let remaining_after_groceries := half_income - 20
  remaining_after_groceries = 100 → monthly_income = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_george_monthly_income_l2397_239743


namespace NUMINAMATH_CALUDE_min_value_theorem_l2397_239761

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (z : ℝ), (y/x) + (3/y) ≥ z → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2397_239761


namespace NUMINAMATH_CALUDE_opposite_expressions_imply_y_value_l2397_239781

theorem opposite_expressions_imply_y_value :
  ∀ y : ℚ, (4 * y + 8) = -(8 * y - 7) → y = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_imply_y_value_l2397_239781


namespace NUMINAMATH_CALUDE_least_divisible_by_240_cubed_l2397_239778

theorem least_divisible_by_240_cubed (a : ℕ) : 
  (∀ n : ℕ, n < 60 → ¬(240 ∣ n^3)) ∧ (240 ∣ 60^3) := by
sorry

end NUMINAMATH_CALUDE_least_divisible_by_240_cubed_l2397_239778


namespace NUMINAMATH_CALUDE_die_roll_invariant_l2397_239799

/-- Represents the faces of a tetrahedral die -/
inductive DieFace
  | one
  | two
  | three
  | four

/-- Represents a position in the triangular grid -/
structure GridPosition where
  x : ℕ
  y : ℕ

/-- Represents the state of the die on the grid -/
structure DieState where
  position : GridPosition
  faceDown : DieFace

/-- Represents a single roll of the die -/
inductive DieRoll
  | rollLeft
  | rollRight
  | rollUp
  | rollDown

/-- Defines the starting corner of the grid -/
def startCorner : GridPosition :=
  { x := 0, y := 0 }

/-- Defines the opposite corner of the grid -/
def endCorner : GridPosition :=
  { x := 1, y := 1 }  -- Simplified for demonstration; actual values depend on grid size

/-- Function to perform a single roll -/
def performRoll (state : DieState) (roll : DieRoll) : DieState :=
  sorry  -- Implementation details omitted

/-- Theorem stating that regardless of the path taken, the die will end with face 1 down -/
theorem die_roll_invariant (path : List DieRoll) :
  let initialState : DieState := { position := startCorner, faceDown := DieFace.four }
  let finalState := path.foldl performRoll initialState
  finalState.position = endCorner → finalState.faceDown = DieFace.one :=
by sorry

end NUMINAMATH_CALUDE_die_roll_invariant_l2397_239799


namespace NUMINAMATH_CALUDE_y_can_take_any_real_value_l2397_239732

-- Define the equation
def equation (x y : ℝ) : Prop := 2 * x * abs x + y^2 = 1

-- Theorem statement
theorem y_can_take_any_real_value :
  ∀ y : ℝ, ∃ x : ℝ, equation x y :=
by
  sorry

end NUMINAMATH_CALUDE_y_can_take_any_real_value_l2397_239732


namespace NUMINAMATH_CALUDE_equation_solution_l2397_239754

theorem equation_solution : 
  ∃ x : ℝ, (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + x + (0.9 : ℝ)^2 = 0.2999999999999999 ∧ 
  x = -1.73175 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2397_239754


namespace NUMINAMATH_CALUDE_sixth_term_value_l2397_239760

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  a_1_eq_4 : a 1 = 4
  a_3_eq_prod : a 3 = a 2 * a 4

/-- The sixth term of the geometric sequence is either 1/8 or -1/8 -/
theorem sixth_term_value (seq : GeometricSequence) : 
  seq.a 6 = 1/8 ∨ seq.a 6 = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2397_239760


namespace NUMINAMATH_CALUDE_max_value_condition_l2397_239769

theorem max_value_condition (x y : ℝ) : 
  (2 * x^2 - y^2 + 3/2 ≤ 1 ∧ y^4 + 4*x + 2 ≤ 1) ↔ 
  ((x = -1/2 ∧ y = 1) ∨ (x = -1/2 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_condition_l2397_239769


namespace NUMINAMATH_CALUDE_amount_lent_to_C_l2397_239780

/-- The amount of money A lent to B in rupees -/
def amount_B : ℝ := 5000

/-- The duration of B's loan in years -/
def duration_B : ℝ := 2

/-- The duration of C's loan in years -/
def duration_C : ℝ := 4

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.07000000000000001

/-- The total interest received from both B and C in rupees -/
def total_interest : ℝ := 1540

/-- The amount of money A lent to C in rupees -/
def amount_C : ℝ := 3000

/-- Theorem stating that given the conditions, A lent 3000 rupees to C -/
theorem amount_lent_to_C : 
  amount_B * interest_rate * duration_B + 
  amount_C * interest_rate * duration_C = total_interest :=
by sorry

end NUMINAMATH_CALUDE_amount_lent_to_C_l2397_239780


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l2397_239737

theorem unique_number_with_properties : ∃! x : ℕ, 
  (x ≥ 10000 ∧ x < 100000) ∧ 
  (x * (x - 1) % 100000 = 0) ∧
  ((x / 1000) % 10 = 0) ∧
  ((x % 3125 = 0 ∧ (x - 1) % 32 = 0) ∨ ((x - 1) % 3125 = 0 ∧ x % 32 = 0)) :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l2397_239737


namespace NUMINAMATH_CALUDE_marly_soup_bags_l2397_239775

/-- Calculates the number of bags needed to hold Marly's soup -/
def bags_needed (milk : ℚ) (chicken_stock_ratio : ℚ) (vegetables : ℚ) (bag_capacity : ℚ) : ℚ :=
  let total_volume := milk + (chicken_stock_ratio * milk) + vegetables
  total_volume / bag_capacity

/-- Proves that Marly needs 3 bags for his soup -/
theorem marly_soup_bags :
  bags_needed 2 3 1 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marly_soup_bags_l2397_239775


namespace NUMINAMATH_CALUDE_minimum_in_interval_implies_a_range_l2397_239783

open Real

/-- The function f(x) = x³ - 2ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x + a

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a

theorem minimum_in_interval_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (f a) x) →
  (∀ x ∈ Set.Ioo 0 1, ¬IsLocalMax (f a) x) →
  a ∈ Set.Ioo 0 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_minimum_in_interval_implies_a_range_l2397_239783


namespace NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l2397_239740

theorem function_always_positive_implies_x_range (f : ℝ → ℝ) :
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x = x^2 + (a - 4)*x + 4 - 2*a) →
  (∀ a ∈ Set.Icc (-1) 1, ∀ x, f x > 0) →
  ∀ x, f x > 0 → x ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by sorry

end NUMINAMATH_CALUDE_function_always_positive_implies_x_range_l2397_239740


namespace NUMINAMATH_CALUDE_square_area_problem_l2397_239738

theorem square_area_problem (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 3) (h₂ : s₂ = 7) (h₃ : s₃ = 22) :
  s₁ + s₂ + s₃ + s₄ + s₅ = s₃ + s₅ → s₄ = 18 :=
by sorry

end NUMINAMATH_CALUDE_square_area_problem_l2397_239738


namespace NUMINAMATH_CALUDE_circle_radius_range_l2397_239721

/-- The set of points (x, y) satisfying the given equation -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sin (3 * p.1 + 4 * p.2) = Real.sin (3 * p.1) + Real.sin (4 * p.2)}

/-- A circle with center c and radius r -/
def Circle (c : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2}

/-- The theorem stating the range of possible radii for non-intersecting circles -/
theorem circle_radius_range (c : ℝ × ℝ) (r : ℝ) :
  (∀ p ∈ M, p ∉ Circle c r) → 0 < r ∧ r < Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_range_l2397_239721


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l2397_239702

theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, x ≠ 0 → f x = k / x) ∧ f (-2) = 3) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l2397_239702


namespace NUMINAMATH_CALUDE_open_box_volume_l2397_239710

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_square_side : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_square_side = 8) : 
  (sheet_length - 2 * cut_square_side) * 
  (sheet_width - 2 * cut_square_side) * 
  cut_square_side = 5120 := by
sorry

end NUMINAMATH_CALUDE_open_box_volume_l2397_239710


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2397_239759

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let new_length := 1.3 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.495 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2397_239759


namespace NUMINAMATH_CALUDE_analysis_time_proof_l2397_239764

/-- The number of bones in a human body -/
def num_bones : ℕ := 206

/-- The time spent analyzing each bone (in hours) -/
def time_per_bone : ℕ := 1

/-- The total time needed to analyze all bones in a human body -/
def total_analysis_time : ℕ := num_bones * time_per_bone

theorem analysis_time_proof : total_analysis_time = 206 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_proof_l2397_239764


namespace NUMINAMATH_CALUDE_range_of_m_l2397_239787

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q,
    prove that the range of values for m is m > 2/3. -/
theorem range_of_m (p q : ℝ → Prop) (m : ℝ) 
  (h_p : ∀ x, p x ↔ (x + 1) * (x - 1) ≤ 0)
  (h_q : ∀ x, q x ↔ (x + 1) * (x - (3 * m - 1)) ≤ 0)
  (h_m_pos : m > 0)
  (h_sufficient : ∀ x, p x → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬p x) :
  m > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2397_239787


namespace NUMINAMATH_CALUDE_max_true_statements_l2397_239763

theorem max_true_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (s1 s2 s3 s4 s5 : Bool),
    s1 = (1 / a > 1 / b) ∧
    s2 = (a^2 > b^2) ∧
    s3 = (a > b) ∧
    s4 = (|a| > 1) ∧
    s5 = (b < 1) ∧
    s1 + s2 + s3 + s4 + s5 ≤ 4) ∧
  (∃ (s1 s2 s3 s4 s5 : Bool),
    s1 = (1 / a > 1 / b) ∧
    s2 = (a^2 > b^2) ∧
    s3 = (a > b) ∧
    s4 = (|a| > 1) ∧
    s5 = (b < 1) ∧
    s1 + s2 + s3 + s4 + s5 = 4) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l2397_239763


namespace NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l2397_239772

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents a box containing balls -/
structure Box where
  red : Nat
  black : Nat

/-- Represents the state of the ball distribution -/
structure BallDistribution where
  totalBalls : Nat
  boxA : Box
  boxB : Box
  boxC : Box

/-- The main theorem stating that the number of red balls in Box B
    is equal to the number of black balls in Box C -/
theorem red_in_B_equals_black_in_C
  (dist : BallDistribution)
  (h1 : dist.totalBalls % 2 = 0)
  (h2 : dist.totalBalls = dist.boxA.red + dist.boxA.black + dist.boxB.red + dist.boxB.black + dist.boxC.red + dist.boxC.black)
  (h3 : dist.totalBalls / 2 = dist.boxA.red + dist.boxB.red + dist.boxC.red)
  (h4 : dist.totalBalls / 2 = dist.boxA.black + dist.boxB.black + dist.boxC.black)
  (h5 : dist.boxA.red + dist.boxA.black = dist.totalBalls / 2)
  : dist.boxB.red = dist.boxC.black := by
  sorry

end NUMINAMATH_CALUDE_red_in_B_equals_black_in_C_l2397_239772


namespace NUMINAMATH_CALUDE_korona_division_l2397_239792

theorem korona_division (total : ℕ) (a b c d : ℝ) :
  total = 9246 →
  (2 * a = 3 * b) →
  (5 * b = 6 * c) →
  (3 * c = 4 * d) →
  (a + b + c + d = total) →
  ∃ (k : ℝ), k > 0 ∧ a = 1380 * k ∧ b = 2070 * k ∧ c = 2484 * k ∧ d = 3312 * k :=
by sorry

end NUMINAMATH_CALUDE_korona_division_l2397_239792


namespace NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_l2397_239730

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_2_minus_6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  (∀ x y, y = m * x + b ↔ y - y₀ = m * (x - x₀)) ∧ 
  y₀ = -6 ∧ 
  m = 13 ∧ 
  b = -32 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_minus_6_l2397_239730


namespace NUMINAMATH_CALUDE_max_consecutive_irreducible_five_digit_l2397_239701

/-- A number is irreducible if it cannot be expressed as a product of two three-digit numbers -/
def irreducible (n : ℕ) : Prop :=
  ∀ a b : ℕ, 100 ≤ a ∧ a ≤ 999 ∧ 100 ≤ b ∧ b ≤ 999 → n ≠ a * b

/-- The set of five-digit numbers -/
def five_digit_numbers : Set ℕ := {n | 10000 ≤ n ∧ n ≤ 99999}

/-- A function that returns the length of the longest sequence of consecutive irreducible numbers in a set -/
def max_consecutive_irreducible (s : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating that the maximum number of consecutive irreducible five-digit numbers is 99 -/
theorem max_consecutive_irreducible_five_digit :
  max_consecutive_irreducible five_digit_numbers = 99 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_irreducible_five_digit_l2397_239701


namespace NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l2397_239704

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2021, month := 1, day := 1, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 1453

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { year := 2021, month := 1, day := 2, hour := 0, minute := 13 }

theorem minutes_after_midnight_theorem :
  addMinutes startTime minutesToAdd = expectedResult :=
sorry

end NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l2397_239704


namespace NUMINAMATH_CALUDE_B_subset_M_M_closed_under_mult_l2397_239706

-- Define the set M
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Define the set B
def B : Set ℤ := {b | ∃ n : ℕ, b = 2*n + 1}

-- Theorem 1: B is a subset of M
theorem B_subset_M : B ⊆ M := by sorry

-- Theorem 2: M is closed under multiplication
theorem M_closed_under_mult : ∀ a₁ a₂ : ℤ, a₁ ∈ M → a₂ ∈ M → (a₁ * a₂) ∈ M := by sorry

end NUMINAMATH_CALUDE_B_subset_M_M_closed_under_mult_l2397_239706


namespace NUMINAMATH_CALUDE_brother_catchup_l2397_239789

/-- The time it takes for the older brother to catch up with the younger brother -/
def catchup_time (older_time younger_time delay : ℚ) : ℚ :=
  let relative_speed := 1 / older_time - 1 / younger_time
  let distance_covered := delay / younger_time
  delay + distance_covered / relative_speed

theorem brother_catchup :
  let older_time : ℚ := 12
  let younger_time : ℚ := 20
  let delay : ℚ := 5
  catchup_time older_time younger_time delay = 25/2 := by sorry

end NUMINAMATH_CALUDE_brother_catchup_l2397_239789


namespace NUMINAMATH_CALUDE_smallest_m_is_correct_l2397_239786

/-- The smallest positive value of m for which 10x^2 - mx + 180 = 0 has integral solutions -/
def smallest_m : ℕ := 90

/-- A function representing the quadratic equation 10x^2 - mx + 180 = 0 -/
def quadratic (m : ℕ) (x : ℤ) : ℤ := 10 * x^2 - m * x + 180

theorem smallest_m_is_correct : 
  (∃ x y : ℤ, x ≠ y ∧ quadratic smallest_m x = 0 ∧ quadratic smallest_m y = 0) ∧ 
  (∀ m : ℕ, m < smallest_m → 
    ¬∃ x y : ℤ, x ≠ y ∧ quadratic m x = 0 ∧ quadratic m y = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_correct_l2397_239786


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2397_239785

/-- Proves that the speed of a boat in still water is 12 km/hr given the conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 4) 
  (h2 : downstream_distance = 4.8) 
  (h3 : downstream_time = 18 / 60) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 12 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2397_239785


namespace NUMINAMATH_CALUDE_stratified_sampling_major_c_l2397_239762

/-- Represents the number of students to be sampled from a major -/
def sampleSize (totalStudents : ℕ) (sampleTotal : ℕ) (majorStudents : ℕ) : ℕ :=
  (sampleTotal * majorStudents) / totalStudents

/-- Proves that the number of students to be drawn from major C is 40 -/
theorem stratified_sampling_major_c :
  let totalStudents : ℕ := 1200
  let sampleTotal : ℕ := 120
  let majorAStudents : ℕ := 380
  let majorBStudents : ℕ := 420
  let majorCStudents : ℕ := totalStudents - majorAStudents - majorBStudents
  sampleSize totalStudents sampleTotal majorCStudents = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_major_c_l2397_239762


namespace NUMINAMATH_CALUDE_trajectory_equation_l2397_239714

theorem trajectory_equation (x y : ℝ) :
  ((x + 3)^2 + y^2) + ((x - 3)^2 + y^2) = 38 → x^2 + y^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2397_239714


namespace NUMINAMATH_CALUDE_symmetry_center_of_f_l2397_239708

/-- Given a function f(x) and a constant θ, prove that (0,0) is one of the symmetry centers of the graph of f(x). -/
theorem symmetry_center_of_f (θ : ℝ) :
  let f : ℝ → ℝ := λ x => 2 * Real.cos (2 * x + θ) * Real.sin θ - Real.sin (2 * (x + θ))
  (0, 0) ∈ {p : ℝ × ℝ | ∀ x, f (p.1 + x) = f (p.1 - x)} :=
by sorry

end NUMINAMATH_CALUDE_symmetry_center_of_f_l2397_239708


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l2397_239756

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l2397_239756


namespace NUMINAMATH_CALUDE_maria_receives_no_funds_main_result_l2397_239747

/-- Represents the deposit insurance system in rubles -/
def deposit_insurance_threshold : ℕ := 1600000

/-- Represents Maria's deposit amount in rubles -/
def maria_deposit : ℕ := 0  -- We don't know the exact amount, so we use 0 as a placeholder

/-- Theorem stating that Maria will not receive any funds -/
theorem maria_receives_no_funds (h : maria_deposit < deposit_insurance_threshold) :
  maria_deposit = 0 := by
  sorry

/-- Main theorem combining the conditions and the result -/
theorem main_result : 
  maria_deposit < deposit_insurance_threshold → maria_deposit = 0 := by
  sorry

end NUMINAMATH_CALUDE_maria_receives_no_funds_main_result_l2397_239747


namespace NUMINAMATH_CALUDE_polynomial_roots_l2397_239790

theorem polynomial_roots : 
  let p (x : ℝ) := x^4 - 2*x^3 - 7*x^2 + 14*x - 6
  ∃ (a b c d : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = (-1 + Real.sqrt 13) / 2 ∧ d = (-1 - Real.sqrt 13) / 2) ∧
    (∀ x : ℝ, p x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2397_239790


namespace NUMINAMATH_CALUDE_lines_properties_l2397_239793

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem statement
theorem lines_properties (a : ℝ) :
  -- 1. Perpendicularity condition
  (a ≠ 0 → (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = -1)) ∧
  -- 2. Fixed points condition
  (l₁ a 0 1 ∧ l₂ a (-1) 0) ∧
  -- 3. Maximum distance condition
  (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y → x^2 + y^2 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_lines_properties_l2397_239793


namespace NUMINAMATH_CALUDE_watermelon_banana_weights_l2397_239705

theorem watermelon_banana_weights :
  ∀ (watermelon_weight banana_weight : ℕ),
    2 * watermelon_weight + banana_weight = 8100 →
    2 * watermelon_weight + 3 * banana_weight = 8300 →
    watermelon_weight = 4000 ∧ banana_weight = 100 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_banana_weights_l2397_239705


namespace NUMINAMATH_CALUDE_mean_temperature_is_85_point_6_l2397_239750

def temperatures : List ℝ := [85, 84, 85, 83, 82, 84, 86, 88, 90, 89]

theorem mean_temperature_is_85_point_6 :
  (List.sum temperatures) / (List.length temperatures) = 85.6 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_85_point_6_l2397_239750


namespace NUMINAMATH_CALUDE_optimal_garden_length_l2397_239776

/-- Represents a rectangular garden with one side along a house wall. -/
structure Garden where
  parallel_length : ℝ  -- Length of the side parallel to the house
  perpendicular_length : ℝ  -- Length of the sides perpendicular to the house
  house_length : ℝ  -- Length of the house wall
  fence_cost_per_foot : ℝ  -- Cost of the fence per foot
  total_fence_cost : ℝ  -- Total cost of the fence

/-- The area of the garden. -/
def garden_area (g : Garden) : ℝ :=
  g.parallel_length * g.perpendicular_length

/-- The total length of the fence. -/
def fence_length (g : Garden) : ℝ :=
  g.parallel_length + 2 * g.perpendicular_length

/-- Theorem stating that the optimal garden length is 100 feet. -/
theorem optimal_garden_length (g : Garden) 
    (h1 : g.house_length = 500)
    (h2 : g.fence_cost_per_foot = 10)
    (h3 : g.total_fence_cost = 2000)
    (h4 : fence_length g = g.total_fence_cost / g.fence_cost_per_foot) :
  ∃ (optimal_length : ℝ), 
    optimal_length = 100 ∧ 
    ∀ (other_length : ℝ), 
      0 < other_length → 
      other_length ≤ fence_length g / 2 →
      garden_area { g with parallel_length := other_length, 
                           perpendicular_length := (fence_length g - other_length) / 2 } ≤ 
      garden_area { g with parallel_length := optimal_length, 
                           perpendicular_length := (fence_length g - optimal_length) / 2 } :=
sorry

end NUMINAMATH_CALUDE_optimal_garden_length_l2397_239776


namespace NUMINAMATH_CALUDE_classroom_desks_l2397_239724

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of desks needed given the number of students and students per desk -/
def calculateDesks (students : ℕ) (studentsPerDesk : ℕ) : ℕ := sorry

theorem classroom_desks :
  let studentsBase6 : ℕ := 305
  let studentsPerDesk : ℕ := 3
  let studentsBase10 : ℕ := base6ToBase10 studentsBase6
  calculateDesks studentsBase10 studentsPerDesk = 38 := by sorry

end NUMINAMATH_CALUDE_classroom_desks_l2397_239724


namespace NUMINAMATH_CALUDE_integral_value_l2397_239709

-- Define the inequality
def inequality (x a : ℝ) : Prop := 1 - 3 / (x + a) < 0

-- Define the solution set
def solution_set : Set ℝ := Set.Ioo (-1) 2

-- Theorem statement
theorem integral_value (a : ℝ) 
  (h1 : ∀ x, x ∈ solution_set ↔ inequality x a) :
  ∫ x in a..3, (1 - 3 / (x + a)) = 2 - 3 * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_value_l2397_239709


namespace NUMINAMATH_CALUDE_right_triangle_area_l2397_239794

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 26) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 120 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2397_239794


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l2397_239774

theorem arccos_cos_eleven : 
  ∃! x : ℝ, x ∈ Set.Icc 0 π ∧ (x - 11) ∈ Set.range (λ n : ℤ => 2 * π * n) ∧ x = Real.arccos (Real.cos 11) := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l2397_239774


namespace NUMINAMATH_CALUDE_area_difference_zero_l2397_239755

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  contains_origin : (center.1)^2 + (center.2)^2 < radius^2
  radius : ℝ

-- Define the areas S+ and S-
def S_plus (c : Circle) : ℝ := sorry
def S_minus (c : Circle) : ℝ := sorry

-- Theorem statement
theorem area_difference_zero (c : Circle) : S_plus c - S_minus c = 0 := by sorry

end NUMINAMATH_CALUDE_area_difference_zero_l2397_239755


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_squared_l2397_239725

theorem complex_magnitude_sum_squared : (Complex.abs (3 - 6*Complex.I) + Complex.abs (3 + 6*Complex.I))^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_squared_l2397_239725


namespace NUMINAMATH_CALUDE_system_solution_l2397_239739

theorem system_solution :
  ∃ (x y : ℝ), (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ (x = 0.5) ∧ (y = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2397_239739


namespace NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l2397_239712

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2 * p + 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l2397_239712


namespace NUMINAMATH_CALUDE_tom_initial_money_l2397_239745

/-- Tom's initial amount of money -/
def initial_money : ℕ := sorry

/-- Cost of the game Tom bought -/
def game_cost : ℕ := 49

/-- Cost of each toy -/
def toy_cost : ℕ := 4

/-- Number of toys Tom could buy after purchasing the game -/
def num_toys : ℕ := 2

/-- Theorem stating that Tom's initial money was $57 -/
theorem tom_initial_money : 
  initial_money = game_cost + num_toys * toy_cost :=
sorry

end NUMINAMATH_CALUDE_tom_initial_money_l2397_239745


namespace NUMINAMATH_CALUDE_meat_distribution_l2397_239728

/-- Proves the correct distribution of meat between two pots -/
theorem meat_distribution (pot1 pot2 total_meat : ℕ) 
  (h1 : pot1 = 645)
  (h2 : pot2 = 237)
  (h3 : total_meat = 1000) :
  ∃ (meat1 meat2 : ℕ),
    meat1 + meat2 = total_meat ∧
    pot1 + meat1 = pot2 + meat2 ∧
    meat1 = 296 ∧
    meat2 = 704 := by
  sorry

end NUMINAMATH_CALUDE_meat_distribution_l2397_239728


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2397_239791

/-- The line y = kx + 1 always intersects with the circle x^2 + y^2 - 2ax + a^2 - 2a - 4 = 0 
    for any real k if and only if -1 ≤ a ≤ 3 -/
theorem line_circle_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 + y^2 - 2*a*x + a^2 - 2*a - 4 = 0) ↔ 
  -1 ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2397_239791


namespace NUMINAMATH_CALUDE_part_one_part_two_l2397_239742

-- Define the sets A, B, and C
def A : Set ℝ := {1, 4, 7, 10}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 9}
def C : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

-- Part 1
theorem part_one :
  (A ∪ B 1 = {x | 1 ≤ x ∧ x ≤ 10}) ∧
  (A ∩ Cᶜ = {1, 7, 10}) := by sorry

-- Part 2
theorem part_two :
  ∀ m : ℝ, (B m ∩ C = C) → (-3 < m ∧ m < 3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2397_239742


namespace NUMINAMATH_CALUDE_rationalize_and_product_l2397_239713

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (2 : ℝ) - Real.sqrt 5 / (3 + Real.sqrt 5) = A + B * Real.sqrt C ∧
  A * B * C = -50 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l2397_239713


namespace NUMINAMATH_CALUDE_tangent_slope_point_M_l2397_239744

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

theorem tangent_slope_point_M :
  ∀ x y : ℝ, f y = f x → f' x = -4 → x = -1 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_point_M_l2397_239744


namespace NUMINAMATH_CALUDE_smallest_product_l2397_239748

def digits : List Nat := [7, 8, 9, 10]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 63990 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l2397_239748


namespace NUMINAMATH_CALUDE_last_integer_is_768_l2397_239779

/-- A sequence of 10 distinct positive integers where each (except the first) is a multiple of the previous one -/
def IntegerSequence : Type := Fin 10 → ℕ+

/-- The property that each integer (except the first) is a multiple of the previous one -/
def IsMultipleSequence (seq : IntegerSequence) : Prop :=
  ∀ i : Fin 9, ∃ k : ℕ+, seq (i.succ) = k * seq i

/-- The property that all integers in the sequence are distinct -/
def IsDistinct (seq : IntegerSequence) : Prop :=
  ∀ i j : Fin 10, i ≠ j → seq i ≠ seq j

/-- The last integer is between 600 and 1000 -/
def LastIntegerInRange (seq : IntegerSequence) : Prop :=
  600 < seq 9 ∧ seq 9 < 1000

theorem last_integer_is_768 (seq : IntegerSequence) 
  (h1 : IsMultipleSequence seq) 
  (h2 : IsDistinct seq) 
  (h3 : LastIntegerInRange seq) : 
  seq 9 = 768 := by
  sorry

end NUMINAMATH_CALUDE_last_integer_is_768_l2397_239779


namespace NUMINAMATH_CALUDE_max_sum_of_products_l2397_239707

/-- The maximum sum of products for four distinct values from {3, 4, 5, 6} -/
theorem max_sum_of_products : 
  ∀ (f g h j : ℕ), 
    f ∈ ({3, 4, 5, 6} : Set ℕ) → 
    g ∈ ({3, 4, 5, 6} : Set ℕ) → 
    h ∈ ({3, 4, 5, 6} : Set ℕ) → 
    j ∈ ({3, 4, 5, 6} : Set ℕ) → 
    f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
    f * g + g * h + h * j + j * f ≤ 80 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_products_l2397_239707


namespace NUMINAMATH_CALUDE_society_member_numbers_l2397_239723

theorem society_member_numbers (n : ℕ) (k : ℕ) (members : Fin n → Fin k) :
  n = 1978 →
  k = 6 →
  (∀ i : Fin n, (members i).val + 1 = i.val) →
  ∃ i j l : Fin n,
    (members i = members j ∧ members i = members l ∧ i.val = j.val + l.val) ∨
    (members i = members j ∧ i.val = 2 * j.val) :=
by sorry

end NUMINAMATH_CALUDE_society_member_numbers_l2397_239723


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2397_239777

theorem inequality_solution_set :
  {x : ℝ | |x - 1| + 2*x > 4} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2397_239777


namespace NUMINAMATH_CALUDE_cos_plus_one_is_pseudo_even_l2397_239703

-- Define the concept of a pseudo-even function
def isPseudoEven (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = f (2 * a - x)

-- State the theorem
theorem cos_plus_one_is_pseudo_even :
  isPseudoEven (λ x => Real.cos (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_cos_plus_one_is_pseudo_even_l2397_239703


namespace NUMINAMATH_CALUDE_root_conditions_imply_a_range_l2397_239727

/-- Given a quadratic function f(x) = x² + (a² - 1)x + (a - 2) where 'a' is a real number,
    if one root of f(x) is greater than 1 and the other root is less than 1,
    then 'a' is in the open interval (-2, 1). -/
theorem root_conditions_imply_a_range (a : ℝ) :
  let f := fun x : ℝ => x^2 + (a^2 - 1)*x + (a - 2)
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ > 1 ∧ r₂ < 1) →
  -2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_root_conditions_imply_a_range_l2397_239727


namespace NUMINAMATH_CALUDE_log_sum_equals_zero_l2397_239773

theorem log_sum_equals_zero (a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (h_log : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_log_sum_equals_zero_l2397_239773


namespace NUMINAMATH_CALUDE_dividend_calculation_l2397_239741

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2397_239741


namespace NUMINAMATH_CALUDE_b_cubed_is_zero_l2397_239757

theorem b_cubed_is_zero (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_cubed_is_zero_l2397_239757


namespace NUMINAMATH_CALUDE_large_rectangle_length_is_40cm_l2397_239758

/-- The length of a longer side of a large rectangle formed by four identical small rectangles --/
def large_rectangle_length (small_rectangle_short_side : ℝ) : ℝ :=
  2 * small_rectangle_short_side + 2 * small_rectangle_short_side

/-- Theorem stating that the length of a longer side of the large rectangle is 40 cm --/
theorem large_rectangle_length_is_40cm :
  large_rectangle_length 10 = 40 := by
  sorry

#eval large_rectangle_length 10

end NUMINAMATH_CALUDE_large_rectangle_length_is_40cm_l2397_239758


namespace NUMINAMATH_CALUDE_range_of_f_l2397_239718

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -18 ≤ y ∧ y ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2397_239718


namespace NUMINAMATH_CALUDE_existence_of_integers_l2397_239752

theorem existence_of_integers (m : ℕ) (hm : m > 0) :
  ∃ (a b : ℤ),
    (abs a ≤ m) ∧
    (abs b ≤ m) ∧
    (0 < a + b * Real.sqrt 2) ∧
    (a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_integers_l2397_239752


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l2397_239720

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + (t.c - 2 * t.b) * Real.cos t.A = 0

-- Theorem 1: If the condition is satisfied, then A = π/3
theorem angle_A_value (t : Triangle) (h : satisfies_condition t) : t.A = π / 3 :=
sorry

-- Theorem 2: If a = 2 and the condition is satisfied, the maximum area is √3
theorem max_area (t : Triangle) (h1 : satisfies_condition t) (h2 : t.a = 2) :
  (∀ t' : Triangle, satisfies_condition t' → t'.a = 2 → 
    1 / 2 * t'.b * t'.c * Real.sin t'.A ≤ Real.sqrt 3) ∧
  (∃ t' : Triangle, satisfies_condition t' ∧ t'.a = 2 ∧
    1 / 2 * t'.b * t'.c * Real.sin t'.A = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l2397_239720


namespace NUMINAMATH_CALUDE_root_difference_quadratic_nonnegative_difference_roots_l2397_239726

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * 1^2 + b * 1 + c = 0 → |r₁ - r₂| = Real.sqrt (b^2 - 4*a*c) / a :=
by sorry

theorem nonnegative_difference_roots :
  let f (x : ℝ) := x^2 + 34*x + 274
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| = 6 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_nonnegative_difference_roots_l2397_239726


namespace NUMINAMATH_CALUDE_square_in_S_l2397_239722

def S : Set ℕ := {n | ∃ a b c d e f : ℕ, 
  (n - 1 = a^2 + b^2) ∧ 
  (n = c^2 + d^2) ∧ 
  (n + 1 = e^2 + f^2) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0)}

theorem square_in_S (n : ℕ) (h : n ∈ S) : n^2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_square_in_S_l2397_239722


namespace NUMINAMATH_CALUDE_sequence_problem_l2397_239700

def arithmetic_sequence (a b c d : ℚ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d e : ℚ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_problem (a₁ a₂ b₁ b₂ b₃ : ℚ) 
  (h1 : arithmetic_sequence (-1) a₁ a₂ (-4))
  (h2 : geometric_sequence (-1) b₁ b₂ b₃ (-4)) :
  (a₂ - a₁) / b₂ = 1/2 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l2397_239700


namespace NUMINAMATH_CALUDE_triangle_roots_condition_l2397_239719

/-- Given a cubic polynomial x^3 - ux^2 + vx - w with roots a, b, and c forming a triangle, 
    prove that uv > 2w -/
theorem triangle_roots_condition (u v w a b c : ℝ) : 
  (∀ x, x^3 - u*x^2 + v*x - w = (x - a)*(x - b)*(x - c)) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  uv > 2*w :=
sorry

end NUMINAMATH_CALUDE_triangle_roots_condition_l2397_239719


namespace NUMINAMATH_CALUDE_bike_route_total_length_l2397_239736

/-- The total length of a rectangular bike route -/
def bike_route_length (h1 h2 h3 v1 v2 : ℝ) : ℝ :=
  2 * ((h1 + h2 + h3) + (v1 + v2))

/-- Theorem stating the total length of the specific bike route -/
theorem bike_route_total_length :
  bike_route_length 4 7 2 6 7 = 52 := by
  sorry

#eval bike_route_length 4 7 2 6 7

end NUMINAMATH_CALUDE_bike_route_total_length_l2397_239736


namespace NUMINAMATH_CALUDE_security_deposit_percentage_l2397_239782

/-- Security deposit calculation for a mountain cabin rental --/
theorem security_deposit_percentage
  (daily_rate : ℚ)
  (duration_days : ℕ)
  (pet_fee : ℚ)
  (service_fee_rate : ℚ)
  (security_deposit : ℚ)
  (h1 : daily_rate = 125)
  (h2 : duration_days = 14)
  (h3 : pet_fee = 100)
  (h4 : service_fee_rate = 1/5)
  (h5 : security_deposit = 1110) :
  security_deposit / (daily_rate * duration_days + pet_fee + service_fee_rate * (daily_rate * duration_days + pet_fee)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_security_deposit_percentage_l2397_239782


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l2397_239715

/-- Calculates the overall profit percentage for a retailer selling three items -/
theorem retailer_profit_percentage
  (radio_cost watch_cost phone_cost : ℚ)
  (radio_overhead watch_overhead phone_overhead : ℚ)
  (radio_sp watch_sp phone_sp : ℚ)
  (h_radio_cost : radio_cost = 225)
  (h_watch_cost : watch_cost = 425)
  (h_phone_cost : phone_cost = 650)
  (h_radio_overhead : radio_overhead = 15)
  (h_watch_overhead : watch_overhead = 20)
  (h_phone_overhead : phone_overhead = 30)
  (h_radio_sp : radio_sp = 300)
  (h_watch_sp : watch_sp = 525)
  (h_phone_sp : phone_sp = 800) :
  let total_cp := radio_cost + watch_cost + phone_cost + radio_overhead + watch_overhead + phone_overhead
  let total_sp := radio_sp + watch_sp + phone_sp
  let profit := total_sp - total_cp
  let profit_percentage := (profit / total_cp) * 100
  ∃ ε > 0, |profit_percentage - 19.05| < ε :=
by sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l2397_239715


namespace NUMINAMATH_CALUDE_small_cubes_count_l2397_239751

/-- Given a cube with edge length 9 cm cut into smaller cubes with edge length 3 cm,
    the number of small cubes obtained is 27. -/
theorem small_cubes_count (large_edge : ℕ) (small_edge : ℕ) : 
  large_edge = 9 → small_edge = 3 → (large_edge / small_edge) ^ 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_small_cubes_count_l2397_239751


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2397_239734

theorem quadratic_roots_difference_squared :
  ∀ a b : ℝ, (2 * a^2 - 8 * a + 6 = 0) → (2 * b^2 - 8 * b + 6 = 0) → (a - b)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2397_239734


namespace NUMINAMATH_CALUDE_discount_difference_l2397_239753

def bill_amount : ℝ := 12000
def single_discount : ℝ := 0.42
def first_successive_discount : ℝ := 0.35
def second_successive_discount : ℝ := 0.05

def single_discounted_amount : ℝ := bill_amount * (1 - single_discount)
def successive_discounted_amount : ℝ := bill_amount * (1 - first_successive_discount) * (1 - second_successive_discount)

theorem discount_difference :
  successive_discounted_amount - single_discounted_amount = 450 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l2397_239753


namespace NUMINAMATH_CALUDE_ratio_equality_l2397_239749

theorem ratio_equality (n m : ℚ) (h1 : 3 * n = 4 * m) (h2 : m ≠ 0) (h3 : n ≠ 0) : n / m = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2397_239749


namespace NUMINAMATH_CALUDE_expansion_coefficients_l2397_239795

theorem expansion_coefficients (n : ℕ) : 
  (2^(2*n) = 2^n + 240) → 
  (∃ k, k = (Nat.choose 8 4) ∧ k = 70) ∧ 
  (∃ m, m = (2^4) ∧ m = 16) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l2397_239795


namespace NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l2397_239768

/-- Proof of the original ratio of boarders to day students -/
theorem original_ratio_of_boarders_to_day_students :
  let initial_boarders : ℕ := 120
  let new_boarders : ℕ := 30
  let total_boarders : ℕ := initial_boarders + new_boarders
  let day_students : ℕ := 2 * total_boarders
  (initial_boarders : ℚ) / day_students = 1 / (5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_of_boarders_to_day_students_l2397_239768


namespace NUMINAMATH_CALUDE_subsets_and_proper_subsets_of_S_l2397_239767

def S : Set ℕ := {0, 1, 2}

theorem subsets_and_proper_subsets_of_S :
  (Finset.powerset {0, 1, 2} = {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}) ∧
  (Finset.powerset {0, 1, 2} \ {{0, 1, 2}} = {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}) := by
  sorry

end NUMINAMATH_CALUDE_subsets_and_proper_subsets_of_S_l2397_239767


namespace NUMINAMATH_CALUDE_even_increasing_ordering_l2397_239717

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f is increasing on (0, +∞) if x < y implies f(x) < f(y) for all x, y > 0 -/
def IncreasingOnPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem even_increasing_ordering (f : ℝ → ℝ) 
  (h_even : IsEven f) (h_incr : IncreasingOnPositive f) :
  f 3 < f (-Real.pi) ∧ f (-Real.pi) < f (-4) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_ordering_l2397_239717


namespace NUMINAMATH_CALUDE_triangle_side_length_l2397_239771

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  c = 4 * Real.sqrt 2 →
  B = π / 4 →
  S = 2 →
  S = (1 / 2) * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2397_239771


namespace NUMINAMATH_CALUDE_max_vertex_sum_l2397_239770

def parabola (a b c : ℤ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem max_vertex_sum (a T : ℤ) (h : T ≠ 0) :
  ∃ b c : ℤ,
    (parabola a b c 0 = 0) ∧
    (parabola a b c (3 * T) = 0) ∧
    (parabola a b c (3 * T + 1) = 36) →
    ∃ x y : ℚ,
      (∀ t : ℚ, parabola a b c t ≤ parabola a b c x) ∧
      y = parabola a b c x ∧
      x + y ≤ 62 :=
by sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l2397_239770


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_factorials_l2397_239788

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 5 + factorial 6 + factorial 7

theorem largest_prime_factor_of_sum_of_factorials :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_factorials ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_factorials → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_factorials_l2397_239788


namespace NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l2397_239784

theorem rational_numbers_include_integers_and_fractions : 
  (∀ n : ℤ, ∃ q : ℚ, (n : ℚ) = q) ∧ 
  (∀ a b : ℤ, b ≠ 0 → ∃ q : ℚ, (a : ℚ) / (b : ℚ) = q) :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_include_integers_and_fractions_l2397_239784


namespace NUMINAMATH_CALUDE_answer_key_combinations_l2397_239796

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- Represents the total number of possible true-false combinations -/
def total_true_false_combinations : ℕ := 2^true_false_questions

/-- Represents the number of invalid true-false combinations (all true or all false) -/
def invalid_true_false_combinations : ℕ := 2

/-- The main theorem stating the number of ways to create the answer key -/
theorem answer_key_combinations : 
  (total_true_false_combinations - invalid_true_false_combinations) * 
  (multiple_choice_options^multiple_choice_questions) = 96 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l2397_239796


namespace NUMINAMATH_CALUDE_megan_carrots_second_day_l2397_239798

/-- Calculates the number of carrots Megan picked on the second day -/
def carrots_picked_second_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proves that Megan picked 46 carrots on the second day -/
theorem megan_carrots_second_day : 
  carrots_picked_second_day 19 4 61 = 46 := by
  sorry

end NUMINAMATH_CALUDE_megan_carrots_second_day_l2397_239798


namespace NUMINAMATH_CALUDE_volume_ratio_in_partitioned_cube_l2397_239735

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  edgeLength : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- Calculates the volume of the part of the cube on one side of a plane -/
noncomputable def volumePartition (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem: The ratio of volumes in a cube partitioned by a specific plane -/
theorem volume_ratio_in_partitioned_cube (c : Cube) (e f : Point3D) : 
  let p := Plane.mk 1 1 1 0  -- Placeholder plane, actual coefficients would depend on B, E, F
  volumePartition c p / cubeVolume c = 25 / 72 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_in_partitioned_cube_l2397_239735


namespace NUMINAMATH_CALUDE_visited_neither_country_l2397_239716

theorem visited_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 250 →
  iceland = 125 →
  norway = 95 →
  both = 80 →
  total - ((iceland + norway) - both) = 110 :=
by sorry

end NUMINAMATH_CALUDE_visited_neither_country_l2397_239716


namespace NUMINAMATH_CALUDE_center_square_area_ratio_l2397_239766

/-- Represents a square flag with a symmetric cross -/
structure SquareFlag where
  side : ℝ
  cross_width : ℝ
  center_side : ℝ
  cross_area_ratio : ℝ
  cross_symmetric : Bool
  cross_uniform : Bool

/-- The theorem stating that if a symmetric cross occupies 50% of a square flag's area, 
    the center square occupies 6.25% of the total area -/
theorem center_square_area_ratio (flag : SquareFlag) 
  (h1 : flag.cross_area_ratio = 0.5)
  (h2 : flag.cross_symmetric = true)
  (h3 : flag.cross_uniform = true)
  (h4 : flag.center_side = flag.side / 4) :
  (flag.center_side ^ 2) / (flag.side ^ 2) = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_center_square_area_ratio_l2397_239766


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_fifth_power_l2397_239746

theorem imaginary_part_of_one_plus_i_fifth_power (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 : ℂ) + i)^5 = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_fifth_power_l2397_239746


namespace NUMINAMATH_CALUDE_expand_expression_l2397_239731

theorem expand_expression (x : ℝ) : 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2397_239731


namespace NUMINAMATH_CALUDE_sum_of_integers_5_to_20_l2397_239797

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_of_integers_5_to_20 :
  sum_of_integers 5 20 = 200 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_integers_5_to_20_l2397_239797


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2397_239733

/-- Given n = 3, prove that r = 177136, where r = 3^s - s and s = 2^n + n -/
theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + n
  let r : ℕ := 3^s - s
  r = 177136 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2397_239733


namespace NUMINAMATH_CALUDE_M_union_N_eq_l2397_239711

def M : Set ℤ := {x | |x| < 2}
def N : Set ℤ := {-2, -1, 0}

theorem M_union_N_eq : M ∪ N = {-2, -1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_M_union_N_eq_l2397_239711


namespace NUMINAMATH_CALUDE_subtraction_and_divisibility_imply_sum_l2397_239765

/-- A number is divisible by 11 if and only if the alternating sum of its digits is divisible by 11 -/
def divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

/-- Returns the hundreds digit of a three-digit number -/
def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

/-- Returns the tens digit of a three-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Returns the ones digit of a three-digit number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem subtraction_and_divisibility_imply_sum (c d : ℕ) :
  (745 - (300 + c * 10 + 4) = 400 + d * 10 + 1) →
  divisible_by_11 (400 + d * 10 + 1) →
  c + d = 14 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_divisibility_imply_sum_l2397_239765
