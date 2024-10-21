import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_range_l1163_116311

theorem intersection_points_and_range (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  let f := λ x : ℝ => a * x + b
  let g := λ x : ℝ => a * x^2 + b * x + c
  let h := λ x : ℝ => a * x^2 + (b - a) * x + (c - b)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ 
  (3 / 2 : ℝ) < |x₂ - x₁| ∧ |x₂ - x₁| < 2 * Real.sqrt 3 ∧
  ∀ x : ℝ, x ≤ -Real.sqrt 3 → 0 < h x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_range_l1163_116311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_l1163_116319

/-- A function satisfying the given symmetry conditions -/
structure SymmetricFunction where
  f : ℝ → ℝ
  symmetry_3 : ∀ x, f (3 + x) = f (3 - x)
  symmetry_8 : ∀ x, f (8 + x) = f (8 - x)
  f_0 : f 0 = 2

/-- The number of roots of f(x) = 2 in [-1010, 1010] is at least 203 -/
theorem root_count (sf : SymmetricFunction) : 
  ∃ (S : Finset ℝ), S.card ≥ 203 ∧ (∀ x ∈ S, sf.f x = 2 ∧ x ∈ Set.Icc (-1010) 1010) := by
  sorry

#check root_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_count_l1163_116319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_in_expansion_l1163_116334

theorem coefficient_x3y3_in_expansion : ℕ := by
  -- Define the binomial coefficient
  let binomial (n k : ℕ) : ℕ := Nat.choose n k

  -- Define the expansion of (x+y)^5 using binomial theorem
  let expansion_xy5 (x y : ℚ) : ℚ := 
    binomial 5 0 * x^5 + binomial 5 1 * x^4 * y + binomial 5 2 * x^3 * y^2 + 
    binomial 5 3 * x^2 * y^3 + binomial 5 4 * x * y^4 + binomial 5 5 * y^5

  -- Define the full expansion
  let full_expansion (x y : ℚ) : ℚ := (2*x - 1) * expansion_xy5 x y

  -- The coefficient of x³y³ in the full expansion
  let coefficient_x3y3 : ℕ := 2 * binomial 5 3

  -- The theorem statement
  have : coefficient_x3y3 = 20 := by
    -- Proof goes here
    sorry

  exact 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_in_expansion_l1163_116334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1163_116347

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (f' : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_bound : ∀ x, f' x < 2) :
  {x : ℝ | f (x + 1) - Real.log (x + 2) - 2 > Real.exp (x + 1) + 3 * x} = Set.Ioo (-2) (-1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1163_116347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l1163_116398

/-- Given a 3x3 matrix with elements x, y, z that is not invertible,
    prove that (x/(y+z)) + (y/(x+z)) + (z/(x+y)) can only equal 2 or 3/2 -/
theorem matrix_sum_values (x y z : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![x, y, z; y, z, x; z, x, y]
  ¬(IsUnit (Matrix.det M)) →
  (x / (y + z) + y / (x + z) + z / (x + y) = 2) ∨
  (x / (y + z) + y / (x + z) + z / (x + y) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_values_l1163_116398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1163_116313

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = 1
  sum_3_5 : a 3 + a 5 = 14
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  ∃ n : ℕ, sum_n_terms seq n = 100 ∧ n = 10 := by
  sorry

/-- Helper lemma: The common difference of the sequence -/
lemma common_difference (seq : ArithmeticSequence) : seq.a 2 - seq.a 1 = 2 := by
  sorry

/-- Helper lemma: The nth term of the sequence -/
lemma nth_term (seq : ArithmeticSequence) (n : ℕ) : seq.a n = 1 + 2 * (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1163_116313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_four_stations_valid_l1163_116337

/-- Represents a station on the circular railway -/
structure Station where
  id : ℕ

/-- Represents a circular railway with n stations -/
structure CircularRailway where
  n : ℕ
  stations : Fin n → Station

/-- Represents a radio session between two stations -/
inductive RadioSession where
  | session : Station → Station → RadioSession

/-- Represents the communication order for a station -/
inductive CommunicationOrder where
  | clockwise : CommunicationOrder
  | counterclockwise : CommunicationOrder

/-- A function that returns true if the communication pattern is valid for a given n -/
def validCommunicationPattern (railway : CircularRailway) : Prop :=
  ∀ (i j : Fin railway.n), i ≠ j →
    ∃! (session : RadioSession),
      ∃ (order : CommunicationOrder),
        (order = CommunicationOrder.clockwise ∨ order = CommunicationOrder.counterclockwise) ∧
        (session = RadioSession.session (railway.stations i) (railway.stations j) ∨
         session = RadioSession.session (railway.stations j) (railway.stations i))

/-- The main theorem stating that 4 is the only valid number of stations -/
theorem only_four_stations_valid :
  ∀ (n : ℕ), (∃ (railway : CircularRailway), railway.n = n ∧ validCommunicationPattern railway) ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_four_stations_valid_l1163_116337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_recycling_earnings_l1163_116324

/-- The number of cans in each bag -/
def c : ℕ := sorry

/-- The price paid per can in dollars -/
def price_per_can : ℚ := 1 / 10

/-- The total number of bags collected over three days -/
def total_bags : ℕ := 10 + 3 + 7

/-- The total amount of money earned in dollars -/
def total_money_earned : ℚ := 2 * (c : ℚ)

theorem tiffany_recycling_earnings :
  (total_bags * c : ℚ) * price_per_can = total_money_earned := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiffany_recycling_earnings_l1163_116324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1163_116379

def sequence_a : ℕ → ℚ
  | 0 => 2/3  -- Add a case for 0 to cover all natural numbers
  | n+1 => let an := sequence_a n; an / (1 - an)  -- Rewrite the recursion to avoid termination issues

theorem sequence_a_formula (n : ℕ) : 
  sequence_a n = 2 / (5 - 2*(n+1)) := by
  sorry  -- Use 'by' and 'sorry' to skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1163_116379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_eleven_point_twenty_five_l1163_116362

-- Define the functions t and s
def t (x : ℝ) : ℝ := 4 * x - 9

noncomputable def s (x : ℝ) : ℝ := 
  let y := (x + 9) / 4
  y^2 + 4 * y - 5

-- State the theorem
theorem s_of_one_equals_eleven_point_twenty_five :
  s 1 = 11.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_of_one_equals_eleven_point_twenty_five_l1163_116362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_binomial_expansion_l1163_116388

/-- Given a geometric sequence {a_n} where the 5th term is equal to the constant term
    in the expansion of (x + 1/x)^4, prove that a_3 * a_7 = 36 -/
theorem geometric_sequence_and_binomial_expansion 
  (a : ℕ → ℝ) -- a is the geometric sequence
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) -- geometric sequence condition
  (h_fifth_term : a 5 = (Nat.choose 4 2 : ℝ)) -- 5th term is the constant term of (x + 1/x)^4
  : a 3 * a 7 = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_binomial_expansion_l1163_116388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_in_sport_formulation_l1163_116332

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The sport formulation of the drink -/
def sport_formulation : Formulation where
  flavoring := 1
  corn_syrup := 4
  water := 60

/-- Calculates the amount of water given the amount of corn syrup -/
def water_amount (c : ℚ) : ℚ :=
  (c / sport_formulation.corn_syrup) * sport_formulation.water

/-- Theorem: If there are 6 ounces of corn syrup in the sport formulation,
    then there are 90 ounces of water -/
theorem water_in_sport_formulation :
  water_amount 6 = 90 := by
  -- Unfold the definition of water_amount
  unfold water_amount
  -- Simplify the arithmetic
  simp [sport_formulation]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_in_sport_formulation_l1163_116332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_bound_l1163_116335

noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

theorem arithmetic_sequence_sum_bound (n : ℕ) :
  arithmetic_sum 22 (-3) n ≥ 52 ↔ 3 ≤ n ∧ n ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_bound_l1163_116335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unplaced_piece_is_five_l1163_116336

/-- Represents the distribution of numbers on the pieces -/
def piece_distribution : Finset (ℕ × ℕ) :=
  {(1, 2), (2, 8), (3, 12), (4, 4), (5, 5)}

/-- The total number of pieces -/
def total_pieces : ℕ := 31

/-- The number of pieces placed on the chessboard -/
def placed_pieces : ℕ := 30

/-- The dimensions of the chessboard -/
def chessboard_dimensions : ℕ × ℕ := (5, 6)

/-- Calculates the total sum of all numbers on the pieces -/
def total_sum : ℕ :=
  (piece_distribution.sum fun p => p.1 * p.2)

/-- The number on the piece not placed on the chessboard -/
def unplaced_piece : ℕ := 5

theorem unplaced_piece_is_five :
  (total_sum - unplaced_piece) % (chessboard_dimensions.1 * chessboard_dimensions.2) = 0 ∧
  ∀ n : ℕ, n ∈ piece_distribution.image Prod.fst →
    n ≠ unplaced_piece →
    (total_sum - n) % (chessboard_dimensions.1 * chessboard_dimensions.2) ≠ 0 :=
by
  sorry

#eval total_sum
#eval unplaced_piece

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unplaced_piece_is_five_l1163_116336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_polar_form_complex_division_result_l1163_116333

noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

theorem complex_division_polar_form (r₁ r₂ θ₁ θ₂ : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) :
  (r₁ * cis θ₁) / (r₂ * cis θ₂) = (r₁ / r₂) * cis (θ₁ - θ₂) := by
  sorry

theorem complex_division_result :
  let z := (4 * cis (30 * π / 180)) / (5 * cis (45 * π / 180))
  ∃ (r θ : ℝ), z = r * cis θ ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π ∧ r = 4/5 ∧ θ = 345 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_polar_form_complex_division_result_l1163_116333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1163_116312

/-- The circle on which point A moves -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The fixed point B -/
def pointB : ℝ × ℝ := (3, 0)

/-- The midpoint M of line AB -/
def midpointEq (xM yM xA yA : ℝ) : Prop :=
  xM = (xA + 3) / 2 ∧ yM = yA / 2

/-- The trajectory equation of the midpoint M -/
def trajectoryEq (x y : ℝ) : Prop := (2*x - 3)^2 + 4*y^2 = 1

/-- Theorem stating that the midpoint M of AB, where A is on the circle and B is fixed,
    follows the trajectory equation -/
theorem midpoint_trajectory (xM yM xA yA : ℝ) :
  circleEq xA yA → midpointEq xM yM xA yA → trajectoryEq xM yM := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l1163_116312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l1163_116346

/-- Calculates the time (in seconds) for a train to pass a platform -/
noncomputable def time_to_pass_platform (train_length : ℝ) (train_speed_kmh : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train 360 m long running at 45 km/hr takes 40 seconds to pass a 140 m long platform -/
theorem train_passing_platform :
  time_to_pass_platform 360 45 140 = 40 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_pass_platform 360 45 140

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_platform_l1163_116346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1163_116370

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y z : ℝ, y > 0 ∧ z > 0 ∧ x = y * Real.sqrt z → y = 1 ∧ Real.sqrt z = z

theorem simplest_quadratic_radical :
  let options := [Real.sqrt 17, Real.sqrt 12, Real.sqrt 24, Real.sqrt (1/3)]
  ∀ x ∈ options, is_simplest_quadratic_radical (Real.sqrt 17) ∧
    (is_simplest_quadratic_radical x → x = Real.sqrt 17) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1163_116370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1163_116366

theorem complex_fraction_equality : 
  (1 + Complex.I) / (3 - Complex.I) = (1 + 2*Complex.I) / 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l1163_116366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1163_116321

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x^2 ≥ 1 → x ≥ 1

-- Define the converse
def converse (x : ℝ) : Prop := x ≥ 1 → x^2 ≥ 1

-- Define the inverse
def inverse (x : ℝ) : Prop := x^2 < 1 → x < 1

-- Define the contrapositive
def contrapositive (x : ℝ) : Prop := x < 1 → x^2 < 1

-- Theorem stating that exactly two of these are true
theorem two_true_propositions :
  ∃ (p₁ p₂ : ℝ → Prop) (p₃ : ℝ → Prop),
    (p₁ = converse ∨ p₁ = inverse ∨ p₁ = contrapositive) ∧
    (p₂ = converse ∨ p₂ = inverse ∨ p₂ = contrapositive) ∧
    (p₃ = converse ∨ p₃ = inverse ∨ p₃ = contrapositive) ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    (∀ x : ℝ, p₁ x) ∧
    (∀ x : ℝ, p₂ x) ∧
    (∃ x : ℝ, ¬(p₃ x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1163_116321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1163_116306

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

-- Define the domain of f
def domain (x : ℝ) : Prop := x < -1 ∨ x > 3

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x y, domain x → domain y → x > 3 → y > 3 → x < y → f x > f y :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1163_116306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1163_116361

/-- The area of a triangle with vertices at (2, 2), (9, 2), and (5, 10) is 28 square units. -/
theorem triangle_area : ∃ area : ℝ, area = 28 := by
  -- Define the vertices
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (9, 2)
  let C : ℝ × ℝ := (5, 10)

  -- Calculate the area using the formula: 1/2 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
  let area := (1/2 : ℝ) * abs (
    A.1 * (B.2 - C.2) +
    B.1 * (C.2 - A.2) +
    C.1 * (A.2 - B.2)
  )

  -- Prove that the calculated area equals 28
  use area
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1163_116361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1163_116316

-- Define the constants
noncomputable def cone_radius : ℝ := 10
noncomputable def cone_height : ℝ := 15
noncomputable def cylinder_radius : ℝ := 20

-- Define the volume of the cone
noncomputable def cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height

-- Define the height of water in the cylinder
noncomputable def cylinder_height : ℝ := cone_volume / (Real.pi * cylinder_radius^2)

-- Theorem statement
theorem water_height_in_cylinder :
  cylinder_height = 1.25 := by
  -- Expand the definitions
  unfold cylinder_height cone_volume cone_radius cone_height cylinder_radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cylinder_l1163_116316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_linked_rings_a4_l1163_116302

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => if (n + 1) % 2 = 0 then 2 * a n - 1 else 2 * a n + 2

theorem nine_linked_rings_a4 : a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_linked_rings_a4_l1163_116302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_in_sphere_l1163_116348

theorem cube_in_sphere (V : ℝ) (a : ℝ) :
  V = (9 / 16) * Real.pi →
  V = (4 / 3) * Real.pi * ((Real.sqrt 3 * a) / 2)^3 →
  a = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_in_sphere_l1163_116348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoint_distance_l1163_116331

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 16 * (x + 2)^2 + 4 * y^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (-2, 0)

-- Define an endpoint of the major axis
def major_endpoint : ℝ × ℝ := (-2, 4)

-- Define an endpoint of the minor axis
def minor_endpoint : ℝ × ℝ := (0, 0)

-- Calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem ellipse_axis_endpoint_distance :
  distance major_endpoint minor_endpoint = 2 * Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoint_distance_l1163_116331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bed_purchase_discount_l1163_116356

/-- Calculates the discount percentage given the bed frame price, bed price multiplier, and final price paid -/
noncomputable def discount_percentage (bed_frame_price : ℝ) (bed_price_multiplier : ℝ) (final_price : ℝ) : ℝ :=
  let original_price := bed_frame_price * (1 + bed_price_multiplier)
  let discount_amount := original_price - final_price
  (discount_amount / original_price) * 100

/-- Proves that the discount percentage is 20% for the given conditions -/
theorem bed_purchase_discount :
  discount_percentage 75 10 660 = 20 := by
  unfold discount_percentage
  -- The actual proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bed_purchase_discount_l1163_116356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_interval_containment_l1163_116329

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  let a := 1
  ∃ S : Set ℝ, S = {x | f a x ≥ g x} ∧ S = Set.Icc (-1) ((Real.sqrt 17 - 1) / 2) :=
sorry

-- Theorem for part 2
theorem range_of_a_for_interval_containment :
  ∃ A : Set ℝ, A = {a | ∀ x ∈ Set.Icc (-1) 1, f a x ≥ g x} ∧ A = Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_for_interval_containment_l1163_116329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_needed_to_grandma_l1163_116327

/-- The fuel efficiency of the car in miles per gallon -/
noncomputable def fuel_efficiency : ℚ := 20

/-- The distance to Grandma's house in miles -/
noncomputable def distance_to_grandma : ℚ := 100

/-- The number of gallons of gas needed to reach Grandma's house -/
noncomputable def gallons_needed : ℚ := distance_to_grandma / fuel_efficiency

theorem gas_needed_to_grandma : gallons_needed = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_needed_to_grandma_l1163_116327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_valid_l1163_116328

-- Define the set of valid x values
def valid_x : Set ℝ := {2, 3}

-- Define the four expressions
noncomputable def expr_A (x : ℝ) : Option ℝ := if x ≠ 2 then some (1 / (x - 2)) else none
noncomputable def expr_B (x : ℝ) : Option ℝ := if x ≠ 3 then some (1 / (x - 3)) else none
noncomputable def expr_C (x : ℝ) : Option ℝ := if x ≥ 2 then some (Real.sqrt (x - 2)) else none
noncomputable def expr_D (x : ℝ) : Option ℝ := if x ≥ 3 then some (Real.sqrt (x - 3)) else none

-- Theorem stating that only expr_C is valid for both x values
theorem only_C_valid :
  (∀ x ∈ valid_x, expr_A x = none) ∧
  (∀ x ∈ valid_x, expr_B x = none) ∧
  (∀ x ∈ valid_x, expr_C x ≠ none) ∧
  ¬(∀ x ∈ valid_x, expr_D x ≠ none) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_valid_l1163_116328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1163_116364

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := m * (1/2)^(abs x) + n

noncomputable def g (m n : ℝ) (x : ℝ) : ℝ := (1/4)^x + f m n x

theorem f_and_g_properties :
  ∃ (m n : ℝ),
    (∀ x, f m n x ≤ 1) ∧  -- f approaches y = 1 asymptotically
    (f m n 0 = 0) ∧  -- f passes through the origin
    (∀ x, f m n x = -(1/2)^(abs x) + 1) ∧  -- analytical expression of f
    (∀ x ∈ Set.Icc 0 2, g m n x ≥ 3/4) ∧  -- minimum value of g on [0, 2]
    (∃ x ∈ Set.Icc 0 2, g m n x = 3/4)  -- g attains its minimum
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l1163_116364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_chocolate_depth_l1163_116385

-- Define the constants
def sphere_radius : ℝ := 3
def cylinder_radius : ℝ := 10

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- Define the volume of a cylinder
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- State the theorem
theorem melted_chocolate_depth :
  ∃ h : ℝ, h = 9/25 ∧ 
  sphere_volume sphere_radius = cylinder_volume cylinder_radius h := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_chocolate_depth_l1163_116385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_majority_owner_share_l1163_116369

/-- Represents the profit distribution in a business --/
structure BusinessProfit where
  totalProfit : ℝ
  majorityOwnerShare : ℝ
  partnerShare : ℝ
  numPartners : ℕ

/-- The conditions of the problem --/
def businessConditions (x : ℝ) : BusinessProfit where
  totalProfit := 80000
  majorityOwnerShare := x
  partnerShare := 0.25 * (1 - x)
  numPartners := 4

/-- Theorem stating that the majority owner's share is 25% --/
theorem majority_owner_share (x : ℝ) :
  let b := businessConditions x
  b.majorityOwnerShare + 2 * b.partnerShare * b.totalProfit = 50000 →
  b.majorityOwnerShare = 0.25 := by
  sorry

#check majority_owner_share

end NUMINAMATH_CALUDE_ERRORFEEDBACK_majority_owner_share_l1163_116369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1163_116391

theorem trigonometric_identity (α : ℝ) :
  (Real.cos (4 * α - 9 * π / 2)) / 
  (Real.tan (π / 4 - 2 * α) * (1 - Real.cos (5 * π / 2 + 4 * α))) = 
  Real.tan (4 * α) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1163_116391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_for_c_l1163_116301

/-- Represents the time (in days) it takes for a worker to complete the job alone -/
structure WorkerRate where
  days : ℚ
  days_pos : days > 0

/-- Represents the payment for the job in Rupees -/
def TotalPayment : ℚ := 3600

/-- Calculate the daily work rate of a worker -/
def daily_rate (w : WorkerRate) : ℚ := 1 / w.days

theorem payment_for_c 
  (a : WorkerRate)
  (b : WorkerRate)
  (total_days : ℚ)
  (ha : a.days = 6)
  (hb : b.days = 8)
  (htotal : total_days = 3)
  : ∃ (c_payment : ℚ), c_payment = 450 ∧ 
    c_payment = TotalPayment * (1/total_days - daily_rate a - daily_rate b) / (1/total_days) := by
  sorry

#check payment_for_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_for_c_l1163_116301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1163_116307

theorem power_function_properties : 
  (∀ n : ℝ, (1 : ℝ) ^ n = 1) ∧ 
  (∀ x : ℝ, x > 0 → ∀ n : ℝ, x ^ n > 0) := by
  sorry

#check power_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l1163_116307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_spherical_coords_l1163_116304

/-- The radius of the circle formed by points with spherical coordinates (2, 0, φ) is 2 -/
theorem circle_radius_from_spherical_coords : 
  ∀ φ : ℝ, 
  let r : ℝ := Real.sqrt ((2 * Real.sin φ) ^ 2 + 0 ^ 2)
  ∃ φ_max : ℝ, r ≤ 2 ∧ r = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_spherical_coords_l1163_116304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1163_116360

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
noncomputable def distance_between_planes (p1 p2 : Plane) : ℝ :=
  |p1.d - p2.d| / Real.sqrt (p1.a^2 + p1.b^2 + p1.c^2)

theorem distance_between_specific_planes :
  let p1 : Plane := ⟨3, 2, -6, 12⟩
  let p2 : Plane := ⟨6, 4, -12, 18⟩
  distance_between_planes p1 p2 = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l1163_116360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_four_less_than_negative_three_l1163_116363

theorem only_negative_four_less_than_negative_three :
  ∀ x : ℤ, x ∈ ({-4, -2, 0, 3} : Set ℤ) → (x < -3 ↔ x = -4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_four_less_than_negative_three_l1163_116363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_scaling_transformation_l1163_116380

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  scale_x : ℝ
  scale_y : ℝ

/-- Apply a scaling transformation to a point -/
def apply_scaling (t : ScalingTransformation) (p : ℝ × ℝ) : ℝ × ℝ :=
  (t.scale_x * p.1, t.scale_y * p.2)

theorem correct_scaling_transformation :
  ∃ (t : ScalingTransformation), 
    apply_scaling t (-2, 2) = (-6, 1) ∧ 
    t.scale_x = 3 ∧ 
    t.scale_y = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_scaling_transformation_l1163_116380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_deviation_optimal_min_deviation_exact_l1163_116386

/-- The quadratic polynomial with minimum deviation from zero on [-1, 1] -/
noncomputable def min_deviation_poly (x : ℝ) : ℝ := x^2 - 1/2

/-- The maximum absolute value of the minimum deviation polynomial on [-1, 1] -/
noncomputable def min_deviation_value : ℝ := 1/2

/-- Theorem stating that min_deviation_poly has the smallest maximum absolute value on [-1, 1] among all quadratic polynomials -/
theorem min_deviation_optimal (p q : ℝ) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, 
    |min_deviation_poly x| ≤ min_deviation_value ∧
    ∃ y ∈ Set.Icc (-1 : ℝ) 1, |x^2 + p*x + q| ≥ min_deviation_value :=
by sorry

/-- Theorem stating that the maximum absolute value of min_deviation_poly on [-1, 1] is exactly min_deviation_value -/
theorem min_deviation_exact :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |min_deviation_poly x| = min_deviation_value :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_deviation_optimal_min_deviation_exact_l1163_116386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_one_l1163_116314

theorem max_min_sum_equals_one :
  ∃ (M m : ℝ), 
    (∀ (x y z : ℝ), 4 * (x + y + z) = 2 * (x^2 + y^2 + z^2) → 
      x * y + x * z + y * z ≤ M ∧ m ≤ x * y + x * z + y * z) ∧ 
    M + 10 * m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_one_l1163_116314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1163_116355

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → a ∈ Set.Icc (-3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1163_116355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_problem_l1163_116318

theorem milk_production_problem (a b c d e : ℝ) (h : a > 0 ∧ c > 0) :
  let original_rate := b / (a * c)
  let new_rate := d * original_rate * 0.9
  let total_production := new_rate * e
  total_production = 0.9 * b * d * e / (a * c) := by
  sorry

#check milk_production_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_production_problem_l1163_116318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_ratio_after_two_iterations_l1163_116368

/-- Represents the composition of a milk and water mixture -/
structure Mixture where
  total : ℝ
  milk : ℝ
  water : ℝ
  milk_nonneg : milk ≥ 0
  water_nonneg : water ≥ 0
  sum_parts : milk + water = total

/-- Performs one iteration of removing and replacing mixture with pure milk -/
noncomputable def remove_and_replace (m : Mixture) (amount_removed : ℝ) : Mixture :=
  let milk_ratio := m.milk / m.total
  let water_ratio := m.water / m.total
  let milk_removed := milk_ratio * amount_removed
  let water_removed := water_ratio * amount_removed
  { total := m.total
    milk := m.milk - milk_removed + amount_removed
    water := m.water - water_removed
    milk_nonneg := by sorry
    water_nonneg := by sorry
    sum_parts := by sorry }

/-- The main theorem to be proved -/
theorem final_ratio_after_two_iterations (initial_mixture : Mixture) 
    (h_total : initial_mixture.total = 20)
    (h_ratio : initial_mixture.milk / initial_mixture.water = 3 / 2)
    (h_remove : ∀ i : Fin 2, (remove_and_replace (remove_and_replace initial_mixture 10) 10).total = 20) :
    let final_mixture := remove_and_replace (remove_and_replace initial_mixture 10) 10
    final_mixture.milk / final_mixture.water = 9 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_ratio_after_two_iterations_l1163_116368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_24_l1163_116330

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  downstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ
  current_speed : ℝ

/-- Calculates the upstream distance given a swimming scenario -/
noncomputable def upstream_distance (s : SwimmingScenario) : ℝ :=
  let swimming_speed := s.downstream_distance / s.downstream_time - s.current_speed
  (swimming_speed - s.current_speed) * s.upstream_time

/-- Theorem stating that under the given conditions, the upstream distance is 24 km -/
theorem upstream_distance_is_24 (s : SwimmingScenario) 
  (h1 : s.downstream_distance = 64)
  (h2 : s.downstream_time = 8)
  (h3 : s.upstream_time = 8)
  (h4 : s.current_speed = 2.5) : 
  upstream_distance s = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_24_l1163_116330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_squares_5x5_l1163_116390

/-- A square with side length n -/
structure Square where
  side : ℕ

/-- A covering of a larger square by smaller squares -/
structure Covering where
  squares : List Square
  no_overlap : Bool
  covers_5x5 : Bool

def valid_square_sizes : List ℕ := [1, 2, 3, 4]

/-- Check if a covering is valid for the 5x5 square problem -/
def is_valid_covering (c : Covering) : Prop :=
  c.no_overlap ∧ 
  c.covers_5x5 ∧ 
  (c.squares.map (λ s => s.side)).all (· ∈ valid_square_sizes)

/-- The number of squares in a covering -/
def num_squares (c : Covering) : ℕ := c.squares.length

theorem min_squares_5x5 :
  ∃ (c : Covering), is_valid_covering c ∧ 
    (∀ (c' : Covering), is_valid_covering c' → num_squares c ≤ num_squares c') ∧
    num_squares c = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_squares_5x5_l1163_116390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_sum_l1163_116384

theorem last_two_digits_of_sum (N : ℕ) : 
  N = (Finset.range 2015).sum (fun i => 2^(5^(i + 1))) → N % 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_sum_l1163_116384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1163_116338

/-- The eccentricity of a hyperbola with equation x²/3 - y²/6 = 1 is √3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 3 ∧ 
  ∀ x y : ℝ, x^2 / 3 - y^2 / 6 = 1 → 
  e = Real.sqrt ((x^2 / 3 + y^2 / 6) / (x^2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1163_116338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_minimum_perimeter_l1163_116367

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- Calculates the length of a line segment between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Calculates the perimeter of a quadrilateral -/
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

/-- Calculates the length of a diagonal of a quadrilateral -/
noncomputable def diagonal_length (p q : ℝ × ℝ) : ℝ :=
  distance p q

/-- Calculates the angle between two vectors -/
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- Theorem: Among all quadrilaterals with given diagonal lengths and angle between them,
    the parallelogram has the smallest perimeter -/
theorem parallelogram_minimum_perimeter
  (q : Quadrilateral)
  (d1 d2 : ℝ)
  (θ : ℝ)
  (h1 : diagonal_length q.A q.C = d1)
  (h2 : diagonal_length q.B q.D = d2)
  (h3 : angle_between (q.C.1 - q.A.1, q.C.2 - q.A.2) (q.D.1 - q.B.1, q.D.2 - q.B.2) = θ)
  : ∀ (q' : Quadrilateral),
    diagonal_length q'.A q'.C = d1 →
    diagonal_length q'.B q'.D = d2 →
    angle_between (q'.C.1 - q'.A.1, q'.C.2 - q'.A.2) (q'.D.1 - q'.B.1, q'.D.2 - q'.B.2) = θ →
    perimeter q ≤ perimeter q' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_minimum_perimeter_l1163_116367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_theorem_l1163_116345

-- Define a circle with center O and radius R
structure Circle where
  O : ℝ × ℝ
  R : ℝ

-- Define a chord AB of the circle
structure Chord (c : Circle) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_circle : (A.1 - c.O.1)^2 + (A.2 - c.O.2)^2 = c.R^2 ∧
              (B.1 - c.O.1)^2 + (B.2 - c.O.2)^2 = c.R^2

-- Function to calculate the length of a chord
noncomputable def chordLength (c : Circle) (ab : Chord c) : ℝ :=
  Real.sqrt ((ab.A.1 - ab.B.1)^2 + (ab.A.2 - ab.B.2)^2)

-- Function to check if a chord is a diameter
def isDiameter (c : Circle) (ab : Chord c) : Prop :=
  (ab.A.1 - c.O.1)^2 + (ab.A.2 - c.O.2)^2 = c.R^2 ∧
  (ab.B.1 - c.O.1)^2 + (ab.B.2 - c.O.2)^2 = c.R^2 ∧
  (ab.A.1 - ab.B.1)^2 + (ab.A.2 - ab.B.2)^2 = (2 * c.R)^2

-- Theorem statement
theorem chord_length_theorem (c : Circle) (ab : Chord c) :
  chordLength c ab ≤ 2 * c.R ∧
  (chordLength c ab = 2 * c.R ↔ isDiameter c ab) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_theorem_l1163_116345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_given_point_l1163_116320

theorem sin_alpha_for_given_point :
  ∀ α : ℝ,
  ∃ (x y : ℝ),
  x = 1/2 ∧ y = Real.sqrt 3/2 ∧
  (∃ r : ℝ, r > 0 ∧ x^2 + y^2 = r^2) →
  Real.sin α = Real.sqrt 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_for_given_point_l1163_116320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_global_max_implies_m_ge_one_l1163_116353

/-- The piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ m then -x^2 - 2*x else -x + 2

/-- Theorem stating that if f has a global maximum, then m ≥ 1 -/
theorem global_max_implies_m_ge_one (m : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f m x ≤ f m x₀) → m ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_global_max_implies_m_ge_one_l1163_116353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1163_116352

theorem relationship_abc : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.log 3 / Real.log π
  let c : ℝ := Real.log (Real.sqrt 2 / 2) / Real.log 2
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1163_116352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_even_sum_l1163_116350

theorem existence_of_even_sum (n : ℕ+) (A B : Finset ℕ) 
  (hA : ∀ a ∈ A, a ≤ n)
  (hB : ∀ b ∈ B, b ≤ n)
  (hSum : A.card + B.card ≥ n + 2) :
  ∃ a b, a ∈ A ∧ b ∈ B ∧ Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_even_sum_l1163_116350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_relation_l1163_116308

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define function g
variable (g : ℝ → ℝ)

-- Define symmetry with respect to the origin
def symmetric_to_origin (g h : ℝ → ℝ) : Prop :=
  ∀ x, g x = -h (-x)

-- State the theorem
theorem symmetry_implies_relation :
  symmetric_to_origin g f_inv → g = λ x ↦ -f_inv (-x) :=
by
  intro h
  ext x
  exact h x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_relation_l1163_116308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l1163_116317

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 1) / (n(n + 1)(n + 3)) -/
noncomputable def infiniteSeriesSum : ℝ := ∑' n, (3 * n - 1) / (n * (n + 1) * (n + 3))

/-- Theorem stating that the sum of the infinite series is equal to -73/12 -/
theorem infiniteSeriesSumValue : infiniteSeriesSum = -73/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l1163_116317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l1163_116357

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (h1 : (A.1 - B.1) * (E.1 - A.1) + (A.2 - B.2) * (E.2 - A.2) = 0)  -- Angle EAB is right
variable (h2 : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)  -- Angle ABC is right
variable (h3 : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)  -- AB = 5
variable (h4 : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 8)  -- BC = 8
variable (h5 : Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = 10) -- AE = 10
variable (h6 : ∃ t : ℝ, D = (1 - t) • A + t • C)  -- D is on AC
variable (h7 : ∃ s : ℝ, D = (1 - s) • B + s • E)  -- D is on BE

-- Define the areas of triangles
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

-- State the theorem
theorem area_difference (A B C D E : ℝ × ℝ) 
  (h1 : (A.1 - B.1) * (E.1 - A.1) + (A.2 - B.2) * (E.2 - A.2) = 0)
  (h2 : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)
  (h3 : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (h4 : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 8)
  (h5 : Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = 10)
  (h6 : ∃ t : ℝ, D = (1 - t) • A + t • C)
  (h7 : ∃ s : ℝ, D = (1 - s) • B + s • E) :
  area_triangle A D E - area_triangle B D C = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_l1163_116357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_implies_a_leq_neg_one_solution_set_when_a_eq_one_l1163_116309

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * abs (x - 2) + x

-- Theorem 1: If f(x) has a maximum value, then a ≤ -1
theorem f_max_implies_a_leq_neg_one (a : ℝ) :
  (∃ M : ℝ, ∀ x : ℝ, f a x ≤ M) → a ≤ -1 :=
by sorry

-- Theorem 2: When a = 1, the solution set of f(x) > |2x-3| is {x | x > 1/2}
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x > abs (2*x - 3)} = {x : ℝ | x > 1/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_implies_a_leq_neg_one_solution_set_when_a_eq_one_l1163_116309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_grandsons_is_half_l1163_116397

/-- The number of grandchildren Mr. Lee has -/
def num_grandchildren : ℕ := 12

/-- The probability of a child being male or female -/
noncomputable def gender_probability : ℚ := 1 / 2

/-- Represents the possible outcomes for the number of grandsons -/
def grandson_outcomes : Finset ℕ := Finset.range (num_grandchildren + 1)

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- The probability of having an even number of grandsons -/
noncomputable def prob_even_grandsons : ℚ := 
  (grandson_outcomes.filter (fun n => n % 2 = 0)).card / 2^num_grandchildren

/-- Theorem stating that the probability of having an even number of grandsons is 1/2 -/
theorem prob_even_grandsons_is_half : 
  prob_even_grandsons = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_grandsons_is_half_l1163_116397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_calculation_l1163_116392

/-- Calculates the original cost of a car given repair cost, selling price, and profit percentage --/
noncomputable def originalCost (repairCost sellPrice profitPercent : ℝ) : ℝ :=
  (sellPrice - repairCost) / (1 + profitPercent / 100)

/-- Theorem stating that given the conditions, the original cost is approximately 50075 --/
theorem car_cost_calculation (repairCost sellPrice profitPercent : ℝ)
  (h1 : repairCost = 14000)
  (h2 : sellPrice = 72900)
  (h3 : profitPercent = 17.580645161290324) :
  ‖originalCost repairCost sellPrice profitPercent - 50075‖ < 1 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_calculation_l1163_116392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intersecting_circle_l1163_116310

/-- Circle C with center (-6, 0) and radius 5 -/
def circle_C (x y : ℝ) : Prop := (x + 6)^2 + y^2 = 25

/-- Line l with angle α to the x-axis -/
def line_l (α : ℝ) (x y : ℝ) : Prop := ∃ t : ℝ, x = t * Real.cos α ∧ y = t * Real.sin α

/-- The distance between two points on the xy-plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The slope of line l intersecting circle C with chord length √10 is ±√(15)/3 -/
theorem line_slope_intersecting_circle (α : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l α x₁ y₁ ∧ line_l α x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 10) →
  Real.tan α = Real.sqrt (5 / 3) ∨ Real.tan α = -Real.sqrt (5 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intersecting_circle_l1163_116310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l1163_116382

-- Define Convex and interior_angles as parameters
variable (Convex : Prop) (interior_angles : ℕ → Set ℕ)

theorem polygon_sides_count (n : ℕ) (h_convex : Convex) 
  (h_one_angle : ∃ a : ℕ, a ∈ interior_angles n ∧ a = 160) 
  (h_other_angles : ∀ a : ℕ, a ∈ interior_angles n → a ≠ 160 → a = 112) : n = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l1163_116382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_difference_l1163_116372

-- Define the given conditions
def terry_daily_income : ℚ := 24
def jordan_daily_income : ℚ := 30
def terry_work_days : ℕ := 7
def jordan_work_days : ℕ := 6
def tax_rate : ℚ := 1/10

-- Define the function to calculate weekly gross income
def weekly_gross_income (daily_income : ℚ) (work_days : ℕ) : ℚ :=
  daily_income * work_days

-- Define the function to calculate weekly net income after tax
def weekly_net_income (gross_income : ℚ) : ℚ :=
  gross_income - (gross_income * tax_rate)

-- Theorem statement
theorem income_difference :
  weekly_net_income (weekly_gross_income jordan_daily_income jordan_work_days) -
  weekly_net_income (weekly_gross_income terry_daily_income terry_work_days) = 54/5 := by
  sorry

#eval weekly_net_income (weekly_gross_income jordan_daily_income jordan_work_days) -
      weekly_net_income (weekly_gross_income terry_daily_income terry_work_days)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_difference_l1163_116372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_triangle_is_equilateral_l1163_116340

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  Real.tan t.A * Real.tan t.C = 1 / (2 * Real.cos t.A * Real.cos t.C) + 1

/-- The dot product condition -/
def DotProductCondition (t : Triangle) : Prop :=
  t.a * t.c * Real.cos t.B = 1/2 * t.b^2

/-- Theorem 1: If the triangle conditions are met, then angle B is π/3 -/
theorem angle_B_is_pi_third (t : Triangle) (h : TriangleConditions t) : t.B = Real.pi/3 := by
  sorry

/-- Theorem 2: If both conditions are met, then the triangle is equilateral -/
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : TriangleConditions t) (h2 : DotProductCondition t) : 
  t.a = t.b ∧ t.b = t.c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_third_triangle_is_equilateral_l1163_116340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_12_l1163_116394

-- Define the cone properties
noncomputable def cone_volume : ℝ := 1728 * Real.pi
def vertex_angle : ℝ := 90

-- Define the relationship between radius and height for a 90-degree vertex angle
def radius_height_relation (r h : ℝ) : Prop := h = r

-- Define the volume formula for a cone
noncomputable def cone_volume_formula (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_height_is_12 (r h : ℝ) :
  cone_volume_formula r h = cone_volume ∧ 
  radius_height_relation r h ∧ 
  vertex_angle = 90 →
  h = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_12_l1163_116394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersections_is_two_l1163_116399

-- Define the lines
noncomputable def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
noncomputable def line2 (x y : ℝ) : Prop := x + 3 * y = 3
noncomputable def line3 (x y : ℝ) : Prop := 6 * x - 8 * y = 6

-- Define the intersection points
noncomputable def intersection12 : ℝ × ℝ := (18/13, 11/13)
noncomputable def intersection23 : ℝ × ℝ := (-3/5, 6/5)

-- Theorem statement
theorem num_intersections_is_two :
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (∃! p : ℝ × ℝ, line2 p.1 p.2 ∧ line3 p.1 p.2) ∧
  (∀ p : ℝ × ℝ, ¬(line1 p.1 p.2 ∧ line3 p.1 p.2)) ∧
  intersection12 ≠ intersection23 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_intersections_is_two_l1163_116399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_five_expansion_l1163_116365

theorem coefficient_x_five_expansion (x : ℝ) : 
  (Polynomial.coeff ((Polynomial.X^2 + 1)^2 * (Polynomial.X - 1)^6) 5) = -52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_five_expansion_l1163_116365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l1163_116339

/-- Given a point Q in the first octant of 3D space, if the cosines of the angles between OQ
    and the x- and y-axes are 4/5 and 2/5 respectively, then the cosine of the angle between
    OQ and the z-axis is 1/√5. -/
theorem cosine_third_angle (Q : ℝ × ℝ × ℝ) 
    (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0) 
    (h_cos_alpha : Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 4/5)
    (h_cos_beta : Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 2/5) :
  Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_third_angle_l1163_116339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_proof_l1163_116375

/-- The number of sides in a convex polygon with interior angles in arithmetic progression -/
def polygon_sides : ℕ :=
  28

/-- The common difference of the arithmetic progression of interior angles -/
noncomputable def common_difference : ℝ :=
  3

/-- The largest interior angle of the polygon -/
noncomputable def largest_angle : ℝ :=
  150

/-- The sum of interior angles of a polygon with n sides -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ :=
  180 * (n - 2)

/-- The sum of n terms in arithmetic progression with first term a and last term l -/
noncomputable def sum_arithmetic_progression (n : ℕ) (a l : ℝ) : ℝ :=
  n * (a + l) / 2

theorem polygon_sides_proof :
  let n := polygon_sides
  let d := common_difference
  let largest := largest_angle
  let smallest := largest - (n - 1) * d
  sum_arithmetic_progression n smallest largest = sum_interior_angles n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_proof_l1163_116375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_co_l1163_116351

/-- The atomic mass of carbon in atomic mass units (amu) -/
noncomputable def carbon_mass : ℝ := 12.01

/-- The atomic mass of oxygen in atomic mass units (amu) -/
noncomputable def oxygen_mass : ℝ := 16.00

/-- The molecular mass of carbon monoxide in atomic mass units (amu) -/
noncomputable def co_mass : ℝ := carbon_mass + oxygen_mass

/-- The mass percentage of carbon in carbon monoxide -/
noncomputable def carbon_percentage : ℝ := (carbon_mass / co_mass) * 100

/-- Theorem stating that the mass percentage of carbon in carbon monoxide is approximately 42.91% -/
theorem carbon_percentage_in_co :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |carbon_percentage - 42.91| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_co_l1163_116351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1163_116322

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (-3^x + a) / (3^(x+1) + b)

-- Part 1
theorem part_one : ∃! x : ℝ, f 1 1 x = 3^x := by
  sorry

-- Part 2
theorem part_two (h1 : ∀ x : ℝ, f 1 3 x = -f 1 3 (-x)) 
                 (h2 : ∃ t : ℝ, f 1 3 (t^2 - 2*t) < f 1 3 (2*t^2 - k)) : 
  k > -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1163_116322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_word_generation_count_l1163_116395

/-- Represents a word as a list of characters -/
def Word := List Char

/-- The process of generating new words -/
def generate_words (initial_word : Word) : List Word :=
  sorry

/-- All characters in a word are distinct -/
def all_distinct (w : Word) : Prop :=
  sorry

theorem word_generation_count (initial_word : Word) :
  all_distinct initial_word →
  (generate_words initial_word).length = Nat.factorial initial_word.length :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_word_generation_count_l1163_116395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1163_116396

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 3) / (x - 1)

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = 9/2 ∧
  ∀ x : ℝ, x ≥ 3 → f x ≥ min_val := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1163_116396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l1163_116344

theorem exponential_monotonicity (a b : ℝ) : a > b ↔ (2 : ℝ)^a > (2 : ℝ)^b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_monotonicity_l1163_116344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_two_part_journey_l1163_116326

/-- Calculates the average speed of a two-part journey -/
theorem average_speed_two_part_journey (D : ℝ) : 
  D > 0 → 
  (let train_distance := 0.8 * D
   let car_distance := 0.2 * D
   let train_speed := 80
   let car_speed := 20
   let total_time := train_distance / train_speed + car_distance / car_speed
   D / total_time) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_two_part_journey_l1163_116326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_perimeter_l1163_116373

/-- 
Given a right triangle with sides 9, 12, and 15, prove that an inscribed triangle
formed by points 2 units away from each side has a perimeter of 24 units.
-/
theorem inscribed_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_a : a = 9) (h_b : b = 12) (h_c : c = 15) : 
  (a - 4) + (b - 4) + (c - 4) = 24 := by
  -- Substitute the given values
  rw [h_a, h_b, h_c]
  -- Simplify the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_perimeter_l1163_116373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_q_right_triangle_l1163_116358

theorem tan_q_right_triangle (PQ PR : ℝ) (h_pq : PQ = 40) (h_pr : PR = 41) :
  let QR := Real.sqrt (PR^2 - PQ^2)
  Real.tan (Real.arcsin (QR / PR)) = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_q_right_triangle_l1163_116358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_triangle_in_different_disks_side_length_greater_than_96_l1163_116341

-- Define the disk radius
noncomputable def disk_radius : ℝ := 1 / 1000

-- Define a structure for a point in the plane
structure Point where
  x : ℤ
  y : ℤ

-- Define a structure for an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

-- Function to check if two points are in different disks
def in_different_disks (p q : Point) : Prop :=
  (p.x - q.x)^2 + (p.y - q.y)^2 > (2 * disk_radius)^2

-- Function to calculate the side length of an equilateral triangle
noncomputable def side_length (t : EquilateralTriangle) : ℝ :=
  Real.sqrt ((t.a.x - t.b.x)^2 + (t.a.y - t.b.y)^2 : ℝ)

-- Theorem 1: Existence of an equilateral triangle with vertices in different disks
theorem exists_equilateral_triangle_in_different_disks :
  ∃ t : EquilateralTriangle, 
    in_different_disks t.a t.b ∧ 
    in_different_disks t.b t.c ∧ 
    in_different_disks t.c t.a := by
  sorry

-- Theorem 2: Side length of equilateral triangle with vertices in different disks is greater than 96
theorem side_length_greater_than_96 (t : EquilateralTriangle) :
  in_different_disks t.a t.b ∧ 
  in_different_disks t.b t.c ∧ 
  in_different_disks t.c t.a →
  side_length t > 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_equilateral_triangle_in_different_disks_side_length_greater_than_96_l1163_116341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_condition_sufficient_not_necessary_l1163_116376

theorem sin_condition_sufficient_not_necessary :
  (∀ k : ℤ, Real.sin (2 * k * Real.pi + Real.pi / 6) = 1 / 2) ∧
  (∃ x : ℝ, Real.sin x = 1 / 2 ∧ ∀ k : ℤ, x ≠ 2 * k * Real.pi + Real.pi / 6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_condition_sufficient_not_necessary_l1163_116376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_12_to_2050_l1163_116378

theorem closest_multiple_of_12_to_2050 :
  ∃ (n : ℤ), n * 12 = 2052 ∧ 
  ∀ (m : ℤ), m * 12 ≠ 2052 → |m * 12 - 2050| > |2052 - 2050| := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_12_to_2050_l1163_116378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l1163_116303

/-- The circle in the problem -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

/-- The line passing through the origin -/
def line_eq (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- The chord length formed by the intersection of the line and the circle -/
noncomputable def chord_length (k : ℝ) : ℝ := sorry

theorem line_satisfies_conditions :
  ∃ k : ℝ, 
    (∀ x y : ℝ, line_eq k x y → (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0)) ∧ 
    (∃ x y : ℝ, line_eq k x y ∧ circle_eq x y) ∧
    chord_length k = 2 ∧
    k = 2 := by sorry

#check line_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_satisfies_conditions_l1163_116303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rotation_surface_ratio_l1163_116342

/-- The surface area of the solid formed by rotating a regular hexagon 
    with side length 1 around the axis passing through midpoints of opposite sides -/
noncomputable def surface_area_midpoint_axis : ℝ := (7/2) * Real.pi

/-- The surface area of the solid formed by rotating a regular hexagon 
    with side length 1 around the axis passing through opposite vertices -/
noncomputable def surface_area_vertex_axis : ℝ := 2 * Real.sqrt 3 * Real.pi

/-- The ratio of surface areas of the two solids formed by rotating 
    a regular hexagon with side length 1 around its two types of symmetry axes -/
noncomputable def surface_area_ratio : ℝ := surface_area_midpoint_axis / surface_area_vertex_axis

theorem hexagon_rotation_surface_ratio :
  surface_area_ratio = 7 / (4 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_rotation_surface_ratio_l1163_116342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_to_base10_l1163_116354

/-- Converts a digit from base 7 to base 10 -/
def toBase10 (digit : Nat) : Nat :=
  digit

/-- Calculates the value of a digit in a specific position in base 7 -/
def digitValue (digit : Nat) (position : Nat) : Nat :=
  digit * (7 ^ position)

/-- Theorem: The base 10 representation of 6423 in base 7 is equal to 2271 -/
theorem base7_to_base10 :
  toBase10 3 + digitValue 2 1 + digitValue 4 2 + digitValue 6 3 = 2271 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_to_base10_l1163_116354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_a_and_b_values_l1163_116374

def z : ℂ := 1 + Complex.I

theorem omega_value :
  z^2 + 3*(1 - Complex.I) - 4 = -1 - Complex.I :=
sorry

theorem a_and_b_values (a b : ℝ) :
  (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I →
  a = -1 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_a_and_b_values_l1163_116374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1163_116389

/-- The area of the region in the xy-plane satisfying x^6 - x^2 + y^2 ≤ 0 is equal to π/2 -/
theorem area_of_region : 
  MeasureTheory.volume (Set.Iio 0 ∩ {p : ℝ × ℝ | p.1^6 - p.1^2 + p.2^2 ≤ 0}) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1163_116389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_points_outside_l1163_116300

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define a relation for a point being on a line
variable (isOn : Point → Line → Prop)

-- Define a relation for a point being outside a plane
variable (isOutside : Point → Plane → Prop)

-- Theorem statement
theorem infinitely_many_points_outside 
  (l : Line) (p : Plane) :
  (∃ x : Point, isOn x l ∧ isOutside x p) →
  ∃ S : Set Point, (∀ y ∈ S, isOn y l ∧ isOutside y p) ∧ Set.Infinite S :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_points_outside_l1163_116300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l1163_116325

-- Define f and g as functions from reals to reals
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = f (x + 2)

-- Define the domain of g
def dom_g : Set ℝ := Set.Ioo 0 1

-- Define the domain of f
def dom_f : Set ℝ := Set.Ioo 2 3

-- Theorem statement
theorem domain_shift :
  (∀ x ∈ dom_g, g x = f (x + 2)) →
  dom_f = {x | ∃ y ∈ dom_g, x = y + 2} :=
by
  sorry

#check domain_shift

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l1163_116325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_finish_day_l1163_116359

/-- Represents the amount of food eaten by the cat each day -/
def daily_consumption : ℚ := 2/5 + 1/5

/-- Represents the initial number of cans -/
def initial_cans : ℕ := 10

/-- Calculates the day on which the cat finishes all the food -/
def days_to_finish : ℕ := 
  Int.toNat ((initial_cans : ℚ) / daily_consumption).ceil

/-- Converts the number of days to a day of the week -/
def day_of_week (n : ℕ) : ℕ := 
  (n - 1) % 7 + 1

theorem cat_food_finish_day :
  day_of_week days_to_finish = 3 := by
  sorry

#eval day_of_week days_to_finish

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_finish_day_l1163_116359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_5m_diameter_l1163_116343

/-- The area of a circle with diameter 5 meters in square centimeters -/
theorem circle_area_5m_diameter (π : ℝ) : ℝ := by
  -- Define the diameter in meters
  let diameter : ℝ := 5
  
  -- Define the radius in meters (half the diameter)
  let radius : ℝ := diameter / 2
  
  -- Define the area formula for a circle (π * r^2)
  let area_m2 : ℝ := π * radius^2
  
  -- Define the conversion factor from m² to cm²
  let m2_to_cm2 : ℝ := 10000
  
  -- Calculate the area in square centimeters
  let area_cm2 : ℝ := area_m2 * m2_to_cm2
  
  -- Prove that the area is equal to 62500π square centimeters
  exact area_cm2

#check circle_area_5m_diameter

-- The proof is omitted
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_5m_diameter_l1163_116343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l1163_116349

/-- Ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Angle in degrees -/
def Angle := ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- Function to calculate the angle between two vectors -/
noncomputable def Angle.between (A B C : Point) : ℝ := sorry

theorem ellipse_eccentricity_special_case (e : Ellipse) (F A Q : Point) :
  let O : Point := ⟨0, 0⟩
  F.x < 0 →
  F.y = 0 →
  A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1 →
  Q.x = 0 →
  A.x * Q.x / e.a^2 + A.y * Q.y / e.b^2 = 1 →
  Angle.between F Q O = 45 →
  Angle.between F Q A = 30 →
  eccentricity e = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l1163_116349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1163_116381

-- Define the constants
noncomputable def a : ℝ := Real.exp (-2)
noncomputable def b : ℝ := a ^ a
noncomputable def c : ℝ := a ^ (a ^ a)

-- State the theorem
theorem size_relationship : c < b ∧ b < a := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l1163_116381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1163_116315

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- State the theorem
theorem range_of_a (a : ℝ) : B a ⊆ (A ∩ B a) → a ∈ Set.Iic (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1163_116315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_theorem_l1163_116377

theorem homework_time_theorem (J : ℝ) 
  (greg_time : ℝ)
  (patrick_time : ℝ)
  (h1 : greg_time = J - 6)
  (h2 : patrick_time = 2 * (J - 6) - 4)
  (h3 : J + greg_time + patrick_time = 50) :
  J = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_theorem_l1163_116377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_squares_l1163_116371

theorem cubic_roots_sum_squares (p q : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h : ∀ x, x^3 + p*x + q = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) :
  x₂^2 + x₂*x₃ + x₃^2 = x₁^2 + x₁*x₃ + x₃^2 ∧
  x₁^2 + x₁*x₃ + x₃^2 = x₁^2 + x₁*x₂ + x₂^2 ∧
  x₁^2 + x₁*x₂ + x₂^2 = -p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_sum_squares_l1163_116371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1163_116393

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + Real.log (x + Real.sqrt (x^2 + 1))

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 2, f (x^2 + 2) + f (-2*a*x) ≥ 0) → 
  -3/2 ≤ a ∧ a ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1163_116393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alley_width_problem_l1163_116383

/-- The width of an alley given a ladder's position -/
noncomputable def alleyWidth (ladderLength : ℝ) (angleQ angleR : ℝ) : ℝ :=
  ladderLength * (Real.cos (angleQ * Real.pi / 180) + Real.cos (angleR * Real.pi / 180))

/-- Theorem stating the width of the alley for the given problem -/
theorem alley_width_problem :
  let ladderLength : ℝ := 10
  let angleQ : ℝ := 60
  let angleR : ℝ := 70
  ∃ ε > 0, |alleyWidth ladderLength angleQ angleR - 8.42| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alley_width_problem_l1163_116383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_quarter_necessary_not_sufficient_l1163_116305

theorem pi_quarter_necessary_not_sufficient :
  (∀ α, α = π / 4 → Real.sin α = Real.cos α) ∧
  ¬(∀ α, Real.sin α = Real.cos α → α = π / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_quarter_necessary_not_sufficient_l1163_116305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1163_116387

-- Define the line
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle as a predicate
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Theorem statement
theorem line_circle_intersection (k : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    is_on_circle x₁ y₁ ∧ is_on_circle x₂ y₂ ∧ 
    y₁ = line k x₁ ∧ y₂ = line k x₂) ∧ 
  (line k 0 ≠ 0) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1163_116387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1163_116323

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the centers and radii
noncomputable def O₁ : ℝ × ℝ := sorry
noncomputable def O₂ : ℝ × ℝ := sorry
noncomputable def O₃ : ℝ × ℝ := sorry
def r₁ : ℝ := 5
def r₂ : ℝ := 9

-- Define the circles
noncomputable def C₁ : Set (ℝ × ℝ) := Circle O₁ r₁
noncomputable def C₂ : Set (ℝ × ℝ) := Circle O₂ r₂
noncomputable def C₃ : Set (ℝ × ℝ) := Circle O₃ (r₁ + r₂)

-- State the theorem
theorem chord_length : 
  -- C₁ and C₂ are externally tangent
  (∃ p : ℝ × ℝ, p ∈ C₁ ∧ p ∈ C₂) →
  -- C₁ and C₂ are both internally tangent to C₃
  (∃ q : ℝ × ℝ, q ∈ C₁ ∧ q ∈ C₃) ∧ (∃ r : ℝ × ℝ, r ∈ C₂ ∧ r ∈ C₃) →
  -- Centers are collinear
  (∃ t : ℝ, O₂.1 = O₁.1 + t * (O₃.1 - O₁.1) ∧ O₂.2 = O₁.2 + t * (O₃.2 - O₁.2)) →
  -- A chord of C₃ is a common external tangent of C₁ and C₂
  (∃ A B : ℝ × ℝ, A ∈ C₃ ∧ B ∈ C₃ ∧ 
    (∀ p : ℝ × ℝ, p ∈ C₁ → ((A.1 - p.1) * (B.2 - p.2) = (B.1 - p.1) * (A.2 - p.2))) ∧
    (∀ q : ℝ × ℝ, q ∈ C₂ → ((A.1 - q.1) * (B.2 - q.2) = (B.1 - q.1) * (A.2 - q.2)))) →
  -- The length of the chord
  ∃ A B : ℝ × ℝ, A ∈ C₃ ∧ B ∈ C₃ ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = (10 * Real.sqrt 426) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1163_116323
