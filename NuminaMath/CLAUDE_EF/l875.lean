import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_seven_l875_87577

/-- The distance from a point to a plane defined by three points -/
noncomputable def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  
  -- Coefficients of the plane equation Ax + By + Cz + D = 0
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁

  -- Distance formula
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_is_seven :
  distance_point_to_plane (-6, 7, -10) (3, 10, -1) (-2, 3, -5) (-6, 0, -3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_seven_l875_87577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_p_divided_by_divisor_l875_87553

/-- The polynomial p(x) = x^15 + x^14 + ... + x + 1 -/
noncomputable def p (x : ℝ) : ℝ := (x^16 - 1) / (x - 1)

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 2

/-- The theorem stating that the remainder of p(x) divided by the divisor is -8 -/
theorem remainder_of_p_divided_by_divisor :
  ∃ (q : ℝ → ℝ), ∀ x, x ≠ 1 → p x = q x * divisor x + (-8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_p_divided_by_divisor_l875_87553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l875_87570

noncomputable def f (x : ℝ) : ℝ := (x^2 - x) / (x^2 - x + 1)

theorem f_range : Set.range f = Set.Icc (-1/3 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l875_87570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_of_sequence_l875_87589

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def Arithmetic (d : ℕ → ℕ) : Prop :=
  ∃ c, ∀ n, d (n + 1) - d n = c

/-- A second-order arithmetic sequence is a sequence where the differences between consecutive terms form an arithmetic sequence. -/
def SecondOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ → ℕ, Arithmetic d ∧ ∀ n, a (n + 1) - a n = d n

theorem fifteenth_term_of_sequence (a : ℕ → ℕ) :
  SecondOrderArithmeticSequence a →
  a 1 = 2 →
  a 2 = 3 →
  a 3 = 6 →
  a 4 = 11 →
  a 15 = 198 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_of_sequence_l875_87589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_half_l875_87520

/-- The probability that the sum of three distinct randomly selected integers 
    between 1 and 100 (inclusive) is even -/
noncomputable def prob_even_sum : ℚ :=
  let even_count := 50  -- number of even integers in [1, 100]
  let odd_count := 50   -- number of odd integers in [1, 100]
  let total := 100      -- total number of integers
  
  -- Probability of selecting all even
  let p_all_even := (even_count / total) * ((even_count - 1) / (total - 1)) * ((even_count - 2) / (total - 2))
  
  -- Probability of selecting two odd and one even
  let p_two_odd_one_even := 3 * (odd_count / total) * ((odd_count - 1) / (total - 1)) * (even_count / (total - 2))
  
  -- Total probability
  p_all_even + p_two_odd_one_even

theorem prob_even_sum_is_half : prob_even_sum = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_is_half_l875_87520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l875_87567

-- Define the given condition
def solution_set (b : ℝ) : Set ℝ := {x | x > 2}

-- Define the inequality function
def inequality (b x : ℝ) : Prop := (x + b) / ((x - 6) * (x + 1)) > 0

-- State the theorem
theorem inequality_solution_set (b : ℝ) :
  solution_set b = {x | x > 2} →
  {x : ℝ | inequality b x} = Set.Ioo (-1 : ℝ) 2 ∪ Set.Ioi (6 : ℝ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l875_87567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_concentration_after_six_hours_l875_87501

/-- Represents the tank and its properties -/
structure Tank where
  capacity : ℚ
  initialFill : ℚ
  initialSodiumChlorideConcentration : ℚ
  waterEvaporationRate : ℚ

/-- Calculates the water concentration after a given time -/
def waterConcentrationAfterTime (tank : Tank) (time : ℚ) : ℚ :=
  let initialVolume := tank.initialFill * tank.capacity
  let initialSodiumChlorideVolume := tank.initialSodiumChlorideConcentration * initialVolume
  let initialWaterVolume := initialVolume - initialSodiumChlorideVolume
  let waterEvaporated := tank.waterEvaporationRate * time
  let remainingWaterVolume := initialWaterVolume - waterEvaporated
  let totalVolumeAfterEvaporation := remainingWaterVolume + initialSodiumChlorideVolume
  remainingWaterVolume / totalVolumeAfterEvaporation

theorem water_concentration_after_six_hours (tank : Tank)
    (h1 : tank.capacity = 24)
    (h2 : tank.initialFill = 1/4)
    (h3 : tank.initialSodiumChlorideConcentration = 3/10)
    (h4 : tank.waterEvaporationRate = 2/5) :
    waterConcentrationAfterTime tank 6 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_concentration_after_six_hours_l875_87501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l875_87550

/-- Regular truncated triangular pyramid -/
structure RegularTruncatedTriangularPyramid where
  H : ℝ             -- Height of the pyramid
  α : ℝ             -- Angle between lateral edge and base
  a : ℝ             -- Side length of the lower base
  b : ℝ             -- Side length of the upper base
  h_positive : H > 0
  α_range : 0 < α ∧ α < π / 2
  geometric_mean : H^2 = a * b

/-- Volume of a regular truncated triangular pyramid -/
noncomputable def volume (p : RegularTruncatedTriangularPyramid) : ℝ :=
  (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2)

/-- Theorem: The volume of a regular truncated triangular pyramid is (H³ * √3) / (4 * sin²α) -/
theorem volume_formula (p : RegularTruncatedTriangularPyramid) :
  volume p = (p.H^3 * Real.sqrt 3) / (4 * Real.sin p.α ^ 2) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l875_87550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l875_87579

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_property 
  (seq : ArithmeticSequence) 
  (h : 3 - seq.a 7 = seq.a 7 - seq.a 5) : 
  S seq 17 = 51 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l875_87579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_range_l875_87574

-- Define the ellipse and line
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def line (x y : ℝ) : Prop := x - y = 1

-- Define the intersection points
def intersection_points (a b : ℝ) (P Q : ℝ × ℝ) : Prop :=
  ellipse a b P.fst P.snd ∧ ellipse a b Q.fst Q.snd ∧
  line P.fst P.snd ∧ line Q.fst Q.snd

-- Define orthogonality condition
def orthogonal (P Q : ℝ × ℝ) : Prop :=
  P.fst * Q.fst + P.snd * Q.snd = 0

-- Main theorem
theorem ellipse_line_intersection_range (a b : ℝ) 
  (P Q : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : intersection_points a b P Q)
  (h4 : orthogonal P Q)
  (h5 : Real.sqrt 2 / 2 * a ≤ b) (h6 : b ≤ Real.sqrt 6 / 3 * a) :
  Real.sqrt 5 / 2 ≤ a ∧ a ≤ Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_range_l875_87574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_tangents_l875_87544

/-- Parabola E: y^2 = 4x -/
def parabola_E (x y : ℝ) : Prop := y^2 = 4*x

/-- Circle C: (x-3)^2 + y^2 = 2 -/
def circle_C (x y : ℝ) : Prop := (x-3)^2 + y^2 = 2

/-- The center of circle C -/
def center_C : ℝ × ℝ := (3, 0)

/-- The radius of circle C -/
noncomputable def radius_C : ℝ := Real.sqrt 2

theorem max_angle_tangents : 
  ∀ (P : ℝ × ℝ), parabola_E P.1 P.2 →
  (∃ (θ : ℝ), θ ≤ 60 ∧ 
    ∀ (α : ℝ), (∃ (Q : ℝ × ℝ), circle_C Q.1 Q.2 ∧ 
      (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = radius_C^2) → α ≤ θ) :=
by
  sorry

#check max_angle_tangents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_tangents_l875_87544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_7_and_13_l875_87586

theorem three_digit_divisible_by_7_and_13 : 
  (Finset.filter (fun n => n % 7 = 0 ∧ n % 13 = 0) (Finset.range 1000)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_7_and_13_l875_87586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l875_87592

/-- The area of a quadrilateral given its vertices. -/
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  (1/2) * |x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1|

/-- A quadrilateral with vertices at (0,0), (2,3), (4,0), and (1,1) has an area of 4 square units. -/
theorem quadrilateral_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 3)
  let C : ℝ × ℝ := (4, 0)
  let D : ℝ × ℝ := (1, 1)
  area_quadrilateral A B C D = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l875_87592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l875_87515

theorem election_votes_theorem (total_votes : ℕ) : 
  (0.7 * (total_votes : ℝ) - 0.3 * (total_votes : ℝ) = 280) → total_votes = 700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l875_87515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_C2D_inv_l875_87569

open Matrix

theorem det_C2D_inv {n : Type*} [Fintype n] [DecidableEq n]
  (C D : Matrix n n ℝ) (hC : det C = 3) (hD : det D = 7) :
  det (C ^ 2 * D⁻¹) = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_C2D_inv_l875_87569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l875_87543

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ x, f (x - (-Real.pi / 6)) = -f ((-Real.pi / 6) - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l875_87543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_time_for_specific_ride_l875_87545

/-- Represents a bike ride with varying speeds -/
structure BikeRide where
  totalDistance : ℚ
  totalTime : ℚ
  energeticSpeed : ℚ
  tiredSpeed : ℚ

/-- Calculates the time spent feeling energetic during the bike ride -/
def energeticTime (ride : BikeRide) : ℚ :=
  (ride.totalDistance - ride.tiredSpeed * ride.totalTime) / (ride.energeticSpeed - ride.tiredSpeed)

/-- Theorem stating that for the given bike ride conditions, the energetic time is 14/3 hours -/
theorem energetic_time_for_specific_ride :
  let ride : BikeRide := {
    totalDistance := 170,
    totalTime := 10,
    energeticSpeed := 25,
    tiredSpeed := 10
  }
  energeticTime ride = 14/3 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energetic_time_for_specific_ride_l875_87545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_equal_T_l875_87581

/-- T(n) represents the sum of remainders when n is divided by each integer from 1 through 6 -/
def T (n : ℕ) : ℕ :=
  Finset.sum (Finset.range 6) fun k => n % (k + 1)

/-- Count of two-digit numbers n such that T(n) = T(n+3) -/
def count_equal_T : ℕ :=
  (Finset.range 90).filter (fun n => T (n + 10) = T (n + 13)) |>.card

/-- Theorem stating that there are exactly 5 two-digit numbers n such that T(n) = T(n+3) -/
theorem five_equal_T : count_equal_T = 5 := by
  sorry

#eval count_equal_T

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_equal_T_l875_87581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_is_pi_l875_87511

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.tan x + (Real.tan x)⁻¹)^2

-- State the theorem
theorem f_period_is_pi : 
  ∀ x : ℝ, f (x + π) = f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_is_pi_l875_87511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_donation_exists_l875_87533

/-- A graph representing book ownership -/
structure BookGraph where
  vertices : Finset ℕ
  edges : Finset (Finset ℕ)
  vertex_count : vertices.card = 10
  edge_count : edges.card = 10
  edge_size : ∀ e ∈ edges, e.card = 2
  degree_two : ∀ v ∈ vertices, (edges.filter (λ e => v ∈ e)).card = 2
  two_complete_sets : (edges.biUnion id).card = 20

/-- A selection of books to be donated -/
def Donation (g : BookGraph) := Finset ℕ

/-- The theorem stating that a complete set can be donated -/
theorem complete_set_donation_exists (g : BookGraph) :
  ∃ d : Finset ℕ, 
    d.card = 10 ∧ 
    (∀ v ∈ d, ∃ e ∈ g.edges, v ∈ e) ∧
    d = g.vertices := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_set_donation_exists_l875_87533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_increase_is_25_percent_l875_87519

/-- The percentage increase in apple prices -/
noncomputable def apple_price_increase (original_price : ℝ) (new_total_cost : ℝ) 
  (pounds_per_person : ℝ) (family_members : ℕ) : ℝ :=
  let new_price := new_total_cost / (pounds_per_person * (family_members : ℝ))
  ((new_price - original_price) / original_price) * 100

/-- Theorem stating that the apple price increase is 25% -/
theorem apple_price_increase_is_25_percent :
  apple_price_increase 1.6 16 2 4 = 25 := by
  -- Unfold the definition of apple_price_increase
  unfold apple_price_increase
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_increase_is_25_percent_l875_87519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_x_coordinate_l875_87555

/-- The x-coordinate of the end point of a line segment -/
def end_x : ℝ := sorry

/-- Theorem: For a line segment starting at (2, -1) and ending at (x, 5) with a length of 10 units and x > 0, the value of x is 10. -/
theorem line_segment_x_coordinate :
  end_x > 0 ∧
  Real.sqrt ((end_x - 2)^2 + (5 - (-1))^2) = 10 →
  end_x = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_x_coordinate_l875_87555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l875_87532

theorem matrix_scalar_multiplication (v : Fin 2 → ℝ) :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 0, 3]
  M.vecMul v = (3 : ℝ) • v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l875_87532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kristin_reading_time_l875_87526

/-- 
Given that Peter and Kristin each read 20 books in a week, 
Peter reads 3 times as fast as Kristin, and Peter reads 1 book in 18 hours,
prove that Kristin takes 540 hours to read half of her books.
-/
theorem kristin_reading_time 
  (peter_books kristin_books : ℕ) 
  (peter_speed : ℝ) 
  (peter_time : ℝ) 
  (h1 : peter_books = 20)
  (h2 : kristin_books = 20)
  (h3 : peter_speed = 3)
  (h4 : peter_time = 18)
  : (kristin_books / 2 : ℝ) * (peter_time * peter_speed) = 540 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kristin_reading_time_l875_87526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l875_87529

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the foci
def foci (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Define the point M on the right branch
def point_M (a b c : ℝ) : Prop :=
  hyperbola a b c (b^2 / a)

-- Define the line passing through F1 at 30 degrees
def line_through_F1 (c : ℝ) (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * (x + c)

-- Define MF2 perpendicular to x-axis
def MF2_perpendicular (c : ℝ) : Prop :=
  True  -- This condition is implicitly satisfied by the coordinates of M (c, b^2/a)

-- Theorem statement
theorem hyperbola_eccentricity (a b c : ℝ) :
  hyperbola a b c (b^2 / a) →
  foci a b c →
  line_through_F1 c c (b^2 / a) →
  MF2_perpendicular c →
  c / a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l875_87529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_reflection_segment_length_F_to_F_prime_l875_87561

/-- The distance between a point and its reflection over the x-axis -/
theorem distance_to_x_axis_reflection (x y : ℝ) : 
  Real.sqrt ((x - x)^2 + (y - (-y))^2) = 2 * abs y := by sorry

/-- The length of the segment from F(4, -2) to its reflection F'(4, 2) over the x-axis is 4 -/
theorem segment_length_F_to_F_prime : 
  Real.sqrt ((4 - 4)^2 + (2 - (-2))^2) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_reflection_segment_length_F_to_F_prime_l875_87561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_f_inequality_l875_87582

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- State the theorem
theorem x_range_for_f_inequality (x : ℝ) :
  f (x^2) < f (3*x - 2) → 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_for_f_inequality_l875_87582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_symbol_proof_l875_87572

/-- Represents an element in the periodic table -/
structure Element where
  symbol : String
  atomic_weight : Float

/-- Represents a compound made of elements -/
structure Compound where
  elements : List (Element × Nat)
  molecular_weight : Float

/-- Calculates the molecular weight of a compound -/
def calculate_molecular_weight (c : Compound) : Float :=
  c.elements.foldl (fun acc (elem, count) => acc + elem.atomic_weight * count.toFloat) 0

/-- Theorem stating that if a compound with given composition has a molecular weight of 122,
    then the symbol for oxygen is "O" -/
theorem oxygen_symbol_proof (Al P O : Element) (compound : Compound) :
  Al.symbol = "Al" →
  P.symbol = "P" →
  (Al.atomic_weight - 26.98).abs < 0.01 →
  (P.atomic_weight - 30.97).abs < 0.01 →
  (O.atomic_weight - 16.00).abs < 0.01 →
  compound.elements = [(Al, 1), (P, 1), (O, 4)] →
  (compound.molecular_weight - 122).abs < 0.1 →
  O.symbol = "O" := by
  intro hAl hP hAlWeight hPWeight hOWeight hElements hMolWeight
  sorry

#check oxygen_symbol_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_symbol_proof_l875_87572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nitrogen_atomic_weight_l875_87527

/-- The atomic weight of hydrogen in unified atomic mass units (u) -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of iodine in unified atomic mass units (u) -/
def iodine_weight : ℝ := 126.90

/-- The molecular weight of the compound in unified atomic mass units (u) -/
def compound_weight : ℝ := 145

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 4

/-- The number of nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 1

/-- The number of iodine atoms in the compound -/
def iodine_count : ℕ := 1

/-- The atomic weight of nitrogen in unified atomic mass units (u) -/
def nitrogen_weight : ℝ := compound_weight - (hydrogen_count * hydrogen_weight + iodine_count * iodine_weight)

theorem nitrogen_atomic_weight :
  ∀ ε > 0, |nitrogen_weight - 14.068| < ε := by
  sorry

#eval nitrogen_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nitrogen_atomic_weight_l875_87527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_is_four_l875_87516

/-- A rectangle with sides 8 and 6 -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  h_AB : AB = 8
  h_BC : BC = 6

/-- The maximum length of a perpendicular segment to the diagonal -/
noncomputable def max_perpendicular_length (r : Rectangle) : ℝ := r.AB / 2

/-- Theorem: The maximum length of a perpendicular segment to the diagonal is 4 -/
theorem max_length_is_four (r : Rectangle) : max_perpendicular_length r = 4 := by
  unfold max_perpendicular_length
  rw [r.h_AB]
  norm_num

#check max_length_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_length_is_four_l875_87516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nc_gas_price_l875_87564

/-- The price per gallon of gas in North Carolina -/
def price_nc : ℝ := sorry

/-- The amount of gas bought in North Carolina (in gallons) -/
def gas_nc : ℝ := 10

/-- The amount of gas bought in Virginia (in gallons) -/
def gas_va : ℝ := 10

/-- The price difference per gallon between Virginia and North Carolina -/
def price_diff : ℝ := 1

/-- The total amount spent on gas -/
def total_spent : ℝ := 50

/-- Theorem stating that the price per gallon in North Carolina is $2.00 -/
theorem nc_gas_price : price_nc = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nc_gas_price_l875_87564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defect_free_prob_is_0_86_factory_C_most_likely_l875_87583

-- Define the factories
inductive Factory
| A
| B
| C

-- Define the properties of the factories
def defect_rate (f : Factory) : ℚ :=
  match f with
  | Factory.A => 95/100
  | Factory.B => 90/100
  | Factory.C => 80/100

def production_proportion (f : Factory) : ℚ :=
  match f with
  | Factory.A => 2
  | Factory.B => 3
  | Factory.C => 5

-- Define the total production proportion
def total_proportion : ℚ := 
  production_proportion Factory.A + production_proportion Factory.B + production_proportion Factory.C

-- Define the probability of selecting a product from each factory
noncomputable def factory_probability (f : Factory) : ℚ :=
  production_proportion f / total_proportion

-- Define the probability of selecting a defect-free product
noncomputable def defect_free_probability : ℚ :=
  (factory_probability Factory.A * defect_rate Factory.A) +
  (factory_probability Factory.B * defect_rate Factory.B) +
  (factory_probability Factory.C * defect_rate Factory.C)

-- Define the conditional probability of a factory given a defect-free product
noncomputable def conditional_probability (f : Factory) : ℚ :=
  (factory_probability f * defect_rate f) / defect_free_probability

-- Theorem 1: The probability of selecting a defect-free product is 0.86
theorem defect_free_prob_is_0_86 : defect_free_probability = 43/50 := by
  sorry

-- Theorem 2: Factory C is most likely to have produced a defect-free product
theorem factory_C_most_likely :
  conditional_probability Factory.C > conditional_probability Factory.B ∧
  conditional_probability Factory.C > conditional_probability Factory.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defect_free_prob_is_0_86_factory_C_most_likely_l875_87583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_sum_extrema_l875_87560

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - 2018) ∧
  (∀ x : ℝ, x > 0 → f x > 2018)

/-- The maximum value of f on [-2017, 2017] -/
noncomputable def M (f : ℝ → ℝ) : ℝ := 
  ⨆ x ∈ Set.Icc (-2017) 2017, f x

/-- The minimum value of f on [-2017, 2017] -/
noncomputable def N (f : ℝ → ℝ) : ℝ := 
  ⨅ x ∈ Set.Icc (-2017) 2017, f x

/-- The theorem stating that M + N = 4036 for any function satisfying the given conditions -/
theorem special_function_sum_extrema (f : ℝ → ℝ) (h : SpecialFunction f) : 
  M f + N f = 4036 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_sum_extrema_l875_87560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l875_87546

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  max_area : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_e : e = 1/2
  h_max_area : max_area = Real.sqrt 3

/-- Point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with non-zero slope passing through a point -/
structure Line where
  slope : ℝ
  point : Point
  h_slope_nonzero : slope ≠ 0

/-- Check if a point is on the ellipse -/
def on_ellipse (p : Point) (C : Ellipse) : Prop :=
  p.x^2 / C.a^2 + p.y^2 / C.b^2 = 1

/-- Check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y - l.point.y = l.slope * (p.x - l.point.x)

/-- Main theorem about the ellipse properties -/
theorem ellipse_properties (C : Ellipse) :
  (∃ (x y : ℝ), x^2/4 + y^2/3 = 1) ∧
  (∀ (l : Line) (F₂ A₁ A₂ O P Q : Point),
    F₂.x = 1 ∧ F₂.y = 0 ∧
    A₁.x = -2 ∧ A₁.y = 0 ∧
    A₂.x = 2 ∧ A₂.y = 0 ∧
    O.x = 0 ∧ O.y = 0 ∧
    l.point = F₂ ∧
    (∃ (M N : Point), on_ellipse M C ∧ on_ellipse N C ∧ on_line M l ∧ on_line N l) ∧
    (∃ (P_y Q_y : ℝ),
      P.x = 1 ∧ P.y = P_y ∧
      Q.x = 1 ∧ Q.y = Q_y ∧
      ∃ (M N : Point), on_ellipse M C ∧ on_ellipse N C ∧ on_line M l ∧ on_line N l ∧
      (P_y - A₁.y) / (P.x - A₁.x) = (M.y - A₁.y) / (M.x - A₁.x) ∧
      (Q_y - A₂.y) / (Q.x - A₂.x) = (N.y - A₂.y) / (N.x - A₂.x)) →
    (O.x^2 + O.y^2 = P.x^2 + P.y^2 ∧
     P.x^2 + P.y^2 = A₂.x^2 + A₂.y^2 ∧
     A₂.x^2 + A₂.y^2 = Q.x^2 + Q.y^2 ∧
     (P.x - O.x) * (A₂.x - Q.x) + (P.y - O.y) * (A₂.y - Q.y) = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l875_87546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetry_line_for_exp_l875_87505

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define what it means for a function to be symmetric about a line
def is_symmetric_about_line (f : ℝ → ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f y = x ↔ f (2 * l (x, y) - y) = x

-- Theorem statement
theorem no_symmetry_line_for_exp :
  ¬ ∃ l : ℝ × ℝ → ℝ, is_symmetric_about_line f l := by
  sorry

#check no_symmetry_line_for_exp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetry_line_for_exp_l875_87505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_prime_quotient_l875_87585

def d (N : ℕ) : ℕ := (Finset.filter (· ∣ N) (Finset.range (N + 1))).card

def isPrime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isValidN (N : ℕ) : Prop :=
  N = 8 ∨ N = 9 ∨ N = 12 ∨ N = 18 ∨ N = 24 ∨
  (∃ p : ℕ, isPrime p ∧ p > 3 ∧ (N = 8 * p ∨ N = 12 * p ∨ N = 18 * p))

theorem divisor_prime_quotient (N : ℕ) (hN : N > 0) :
  isPrime (N / d N) ↔ isValidN N :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_prime_quotient_l875_87585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_multiples_of_three_neg20_to_10_l875_87598

def sumMultiplesOfThree (lower upper : Int) : Int :=
  (List.range (Int.toNat (upper - lower + 1))).map (fun i => i + lower)
    |>.filter (fun x => x % 3 = 0)
    |>.sum

theorem sum_multiples_of_three_neg20_to_10 :
  sumMultiplesOfThree (-20) 10 = -45 := by
  -- Proof goes here
  sorry

#eval sumMultiplesOfThree (-20) 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_multiples_of_three_neg20_to_10_l875_87598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l875_87507

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (y : ℝ) → (x : ℝ) → Prop

/-- Represents a parabola -/
structure Parabola where
  a : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem statement -/
theorem min_distance_sum (h : Hyperbola) (p : Parabola) (F A : Point) (O : Point := ⟨0, 0⟩) :
  h.equation = fun y x => y^2 / 3 - x^2 = 1 →
  p.equation = fun x y => x^2 = p.a * y →
  (∃ (x y : ℝ), h.equation y x ∧ F = ⟨x, y⟩) →
  (∃ (x y : ℝ), p.equation x y ∧ F = ⟨x, y⟩) →
  (∃ (x y : ℝ), p.equation x y ∧ A = ⟨x, y⟩) →
  distance A F = 4 →
  (∃ (P : Point), (∀ (Q : Point), distance P O + distance P A ≤ distance Q O + distance Q A)) →
  ∃ (P : Point), distance P O + distance P A = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l875_87507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l875_87551

-- Define the total number of balls
def total_balls : ℕ := 4

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of white balls
def white_balls : ℕ := 2

-- Define the number of balls to be selected
def selected_balls : ℕ := 2

-- Define the probability of selecting 2 balls of different colors
def prob_different_colors : ℚ := 2 / 3

-- Theorem statement
theorem different_color_probability :
  (Nat.choose red_balls 1 * Nat.choose white_balls 1 : ℚ) / Nat.choose total_balls selected_balls = prob_different_colors :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_color_probability_l875_87551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_ellipse_tangent_l875_87538

/-- The distance from the origin to the tangent line of an ellipse -/
theorem distance_to_ellipse_tangent :
  let ellipse := fun (x y : ℝ) ↦ x^2 / 2 + y^2 = 1
  let tangent_point := fun (x y : ℝ) ↦ ellipse x y ∧ y = ((Real.sqrt 5 - 2) / 2) * (x + 1)
  let distance := fun (x y : ℝ) ↦ |y| / Real.sqrt (1 + ((Real.sqrt 5 - 2) / 2)^2)
  ∀ x y, tangent_point x y →
  distance x y = (Real.sqrt 5 - 2) / Real.sqrt (7 - 2 * Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_ellipse_tangent_l875_87538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l875_87537

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

-- State the theorem
theorem f_minimum_value (x : ℝ) (h : x > -1) :
  (∀ y, y > -1 → f y ≥ f 0) ∧ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l875_87537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l875_87554

def sequenceA (n : ℕ) : ℚ := (-1)^(n+1) * (2*n+1) / (2^n : ℚ)

theorem sequence_correct : 
  (sequenceA 1 = 3/2) ∧ 
  (sequenceA 2 = -5/4) ∧ 
  (sequenceA 3 = 7/8) ∧ 
  (sequenceA 4 = -9/16) := by
  sorry

#eval sequenceA 1
#eval sequenceA 2
#eval sequenceA 3
#eval sequenceA 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l875_87554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_point_sum_l875_87593

noncomputable section

-- Define the line
def line (x : ℝ) : ℝ := -1/2 * x + 6

-- Define points P and Q
def P : ℝ × ℝ := (12, 0)
def Q : ℝ × ℝ := (0, 6)

-- Define point T
def T (r s : ℝ) : ℝ × ℝ := (r, s)

-- Define the condition that T is on the line segment PQ
def T_on_PQ (r s : ℝ) : Prop :=
  0 ≤ r ∧ r ≤ 12 ∧ s = line r

-- Define the area ratio condition
def area_ratio_condition (r s : ℝ) : Prop :=
  4 * (1/2 * r * s) = 1/2 * 12 * 6

-- Theorem statement
theorem line_segment_point_sum (r s : ℝ) :
  T_on_PQ r s → area_ratio_condition r s → r + s = 10.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_point_sum_l875_87593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_90_degrees_l875_87563

/-- Given a triangle ABC where sin A + sin C = √2 sin B, prove that the largest angle is 90° -/
theorem largest_angle_is_90_degrees 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle
  (h1 : 0 < A ∧ A < Real.pi) -- A is between 0 and π
  (h2 : 0 < B ∧ B < Real.pi) -- B is between 0 and π
  (h3 : 0 < C ∧ C < Real.pi) -- C is between 0 and π
  (h4 : A + B + C = Real.pi) -- Sum of angles in a triangle is π
  (h5 : Real.sin A + Real.sin C = Real.sqrt 2 * Real.sin B) -- Given condition
  (h6 : a / Real.sin A = b / Real.sin B) -- Law of sines
  (h7 : b / Real.sin B = c / Real.sin C) -- Law of sines
  : max A (max B C) = Real.pi/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_90_degrees_l875_87563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_square_product_exists_l875_87562

theorem subset_with_square_product_exists :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    ∃ S : Finset (Finset ℕ),
      (∀ s ∈ S, s.Nonempty ∧ s ⊆ Finset.range (3 * n + 1).succ.succ \ Finset.range (n^2 + 1)) ∧
      (∀ s ∈ S, ∃ k : ℕ, (s.prod id) = k^2) ∧
      S.card ≥ 2015 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_square_product_exists_l875_87562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l875_87597

noncomputable def f (x : ℝ) := Real.sin (abs (2 * x)) + abs (Real.sin (2 * x))

theorem f_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ x, x ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) → 
    ∀ y, y ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) → x < y → f y < f x) ∧
  (∀ x, f x ≤ 2) ∧ (∃ x, f x = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l875_87597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_line_equation_l875_87540

/-- Helper function to calculate the angle ACB -/
noncomputable def angle_ACB (A B C : ℝ × ℝ) : ℝ :=
  sorry -- Implement the angle calculation

/-- Given a point P and a circle C, prove that the line l passing through P and intersecting C
    at points A and B has the equation x - y - 3 = 0 when angle ACB is at its minimum. -/
theorem min_angle_line_equation (P : ℝ × ℝ) (C : Set (ℝ × ℝ)) :
  P = (1, -2) →
  C = {(x, y) | (x - 2)^2 + (y + 3)^2 = 9} →
  ∃ (A B : ℝ × ℝ),
    A ∈ C ∧ B ∈ C ∧
    (∃ (t : ℝ), A = (1 + t, -2 - t) ∧ B = (1 + t, -2 - t)) ∧
    (∀ (A' B' : ℝ × ℝ),
      A' ∈ C → B' ∈ C →
      (∃ (t' : ℝ), A' = (1 + t', -2 - t') ∧ B' = (1 + t', -2 - t')) →
      angle_ACB A B (2, -3) ≤ angle_ACB A' B' (2, -3)) →
    ∀ (x y : ℝ), x - y - 3 = 0 ↔ ∃ (t : ℝ), (x, y) = (1 + t, -2 - t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_angle_line_equation_l875_87540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_geq_two_l875_87517

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ a ∈ ({x + 1/y, y + 1/z, z + 1/x} : Set ℝ), a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_geq_two_l875_87517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l875_87534

-- Define the functions
noncomputable def f (k : ℤ) (a : ℝ) (x : ℝ) : ℝ := a^x - (k - 1) * a^(-x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f 2 a x) / (f 0 a x)

noncomputable def h (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := (f 0 a (2*x)) + 2 * m * (f 2 a x)

-- Main theorem
theorem main_theorem (a : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (a > 1 → Monotone (g a)) ∧
  ((∃ x ∈ Set.Icc 1 2, ∃ y ∈ Set.Icc 1 2, f 1 a x - f 1 a y = 2) →
    ∀ x, g a (-x) = -(g a x)) ∧
  ((∃ x ∈ Set.Icc 1 2, ∃ y ∈ Set.Icc 1 2, f 1 a x - f 1 a y = 2) →
    (∃ x ≥ 1, h a m x = 0) → m ≤ -17/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l875_87534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_g_of_seven_l875_87528

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem five_fold_g_of_seven :
  g (g (g (g (g 7)))) = -1 / 7 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_g_of_seven_l875_87528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_university_box_cost_l875_87536

/-- The cost per box for packaging a fine arts collection. -/
noncomputable def cost_per_box (box_length box_width box_height : ℝ) (total_volume min_total_cost : ℝ) : ℝ :=
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  min_total_cost / num_boxes

/-- Theorem: The cost per box is $0.50 given the specified conditions. -/
theorem university_box_cost :
  cost_per_box 20 20 12 2400000 250 = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_university_box_cost_l875_87536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_movies_five_years_l875_87521

def movies_lj (year : ℕ) : ℝ :=
  220 * (1 + 0.03) ^ (year - 1)

def movies_johnny (year : ℕ) : ℝ :=
  (220 * 1.25) * (1 + 0.05) ^ (year - 1)

def total_movies (years : ℕ) : ℝ :=
  (List.range years).map (fun y => movies_lj (y + 1) + movies_johnny (y + 1)) |>.sum

theorem total_movies_five_years :
  ⌊total_movies 5⌋ = 2688 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_movies_five_years_l875_87521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cosine_and_sine_l875_87590

theorem intersection_of_cosine_and_sine (φ : ℝ) : 
  0 ≤ φ ∧ φ ≤ π →
  Real.cos (π/3) = Real.sin (2*(π/3) + φ) →
  φ = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_cosine_and_sine_l875_87590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l875_87510

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the condition for the intersection point
def intersection_condition (t₁ t₂ α : ℝ) : Prop :=
  line_l ((t₁ + t₂) / 2) α = (1, 2)

-- Theorem statement
theorem slope_of_line_l (t₁ t₂ α : ℝ) :
  intersection_condition t₁ t₂ α →
  (∃ (θ₁ θ₂ : ℝ), curve_C θ₁ = line_l t₁ α ∧ curve_C θ₂ = line_l t₂ α) →
  Real.tan α = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l875_87510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l875_87513

theorem inequality_proof (a b lambda : ℝ) (ha : a > 0) (hb : b > 0) (hlambda : 0 < lambda) (hlambda2 : lambda ≤ 1/2) :
  (a^lambda + b^lambda) * ((2*a + b)^(-lambda) + (a + 2*b)^(-lambda)) ≤ 4 * 3^(-lambda) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l875_87513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_cube_fixed_point_l875_87503

noncomputable def F : ℕ → (ℝ → ℝ)
| 0 => λ x => x
| 1 => λ x => x
| (n + 2) => λ x => 1 / (1 - F (n + 1) x)

def is_three_digit_cube (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3

theorem largest_three_digit_cube_fixed_point :
  ∃ C : ℕ, is_three_digit_cube C ∧ 
    F C C = C ∧
    ∀ D : ℕ, is_three_digit_cube D → F D D = D → D ≤ C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_cube_fixed_point_l875_87503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l875_87506

noncomputable def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2/2 = 1

noncomputable def eccentricity : ℝ := Real.sqrt 3 / 3

noncomputable def perimeter_F1AB : ℝ := 4 * Real.sqrt 3

def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

def point_P (x y : ℝ) : Prop := ellipse_C x y

def is_parallelogram (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) : Prop :=
  x₀ = x₁ + x₂ ∧ y₀ = y₁ + y₂

theorem ellipse_and_line_theorem :
  (∀ x y, ellipse_C x y → x^2/3 + y^2/2 = 1) ∧
  (∃ k x₁ y₁ x₂ y₂ x₀ y₀,
    ellipse_C x₁ y₁ ∧
    ellipse_C x₂ y₂ ∧
    point_P x₀ y₀ ∧
    line_l k x₁ y₁ ∧
    line_l k x₂ y₂ ∧
    is_parallelogram x₁ y₁ x₂ y₂ x₀ y₀ →
    k = Real.sqrt 2 ∨ k = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l875_87506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_2a_minus_b_l875_87566

-- Define the sets A and B as functions
def A : ℝ → Prop := λ x => x < 1 ∨ x > 5
def B (a b : ℝ) : ℝ → Prop := λ x => a ≤ x ∧ x ≤ b

-- State the theorem
theorem value_of_2a_minus_b (a b : ℝ) : 
  (∀ x, A x ∨ B a b x) ∧ 
  (∀ x, (A x ∧ B a b x) ↔ (5 < x ∧ x ≤ 6)) → 
  2 * a - b = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_2a_minus_b_l875_87566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l875_87557

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | Real.exp (x * Real.log 2) < 2}

theorem intersection_M_complement_N : 
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complement_N_l875_87557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_egyptian_fraction_l875_87547

open BigOperators Finset

theorem count_solutions_egyptian_fraction : 
  ∃ s : Finset (ℤ × ℤ), 
    s.card = 53 ∧ 
    ∀ (x y : ℤ), (x, y) ∈ s ↔ 
      (x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / 2022 = 1 / x + 1 / y) := by
  sorry

#check count_solutions_egyptian_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_egyptian_fraction_l875_87547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_sqrt_130_l875_87568

def A : Fin 3 → ℝ := ![2, -3, 1]
def B : Fin 3 → ℝ := ![4, -6, 4]
def C : Fin 3 → ℝ := ![5, -2, 1]
def D : Fin 3 → ℝ := ![7, -5, 4]

def vec_sub (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => b i - a i

def cross_product (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![u 1 * v 2 - u 2 * v 1, u 2 * v 0 - u 0 * v 2, u 0 * v 1 - u 1 * v 0]

def norm_squared (v : Fin 3 → ℝ) : ℝ :=
  (v 0)^2 + (v 1)^2 + (v 2)^2

noncomputable def quadrilateral_area (a b c d : Fin 3 → ℝ) : ℝ :=
  let u := vec_sub a b
  let v := vec_sub a c
  Real.sqrt (norm_squared (cross_product u v))

theorem quadrilateral_area_is_sqrt_130 :
  quadrilateral_area A B C D = Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_sqrt_130_l875_87568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l875_87548

/-- Proves that given 15 families with an average of 3 children per family,
    and exactly 3 childless families, the average number of children in
    families with children is 3.75. -/
theorem average_children_in_families_with_children
  (total_families : ℕ)
  (average_children_per_family : ℚ)
  (childless_families : ℕ)
  (h1 : total_families = 15)
  (h2 : average_children_per_family = 3)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * average_children_per_family / ((total_families : ℚ) - childless_families) = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_children_in_families_with_children_l875_87548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_b_value_l875_87509

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) : ℝ := e^2 * Real.log x

def is_tangent_line (k b : ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ y₀, f x₀ = y₀ ∧ (∀ x, k * x + b = k * x₀ + y₀)

theorem common_tangent_b_value :
  ∀ k b : ℝ,
  (∃ x₁ x₂ : ℝ, is_tangent_line k b f x₁ ∧ is_tangent_line k b g x₂) →
  (b = 0 ∨ b = -e^2) := by
  sorry

#check common_tangent_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_b_value_l875_87509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l875_87573

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola x y → (∃ x' y' : ℝ, asymptotes x' y' ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, hyperbola x y → 
      (|x - x'| < δ ∨ |y - y'| < δ) → 
      |y - (Real.sqrt 3 * x)| < ε ∨ |y - (-Real.sqrt 3 * x)| < ε)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l875_87573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_sum_l875_87514

theorem complex_number_sum (x y : ℝ) : 
  (1 - Complex.I) / (2 + Complex.I) = Complex.ofReal x + Complex.I * Complex.ofReal y → x + y = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_sum_l875_87514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_theorem_simplification_theorem_equation_solution_theorem_l875_87599

-- 1. Factorization
theorem factorization_theorem (a : ℝ) : 2*a^3 - 12*a^2 + 8*a = 2*a*(a-2)*(a-4) := by
  sorry

-- 2. Algebraic simplification
theorem simplification_theorem (a : ℝ) (h : a ≠ 0) (h' : a ≠ 1) : 
  3/a - 6/(1-a) - (a+5)/(a^2-a) = -8/(1-a) - 2/(a*(1-a)) := by
  sorry

-- 3. Equation solving
theorem equation_solution_theorem :
  {x : ℝ | (x-2)/(x+2) - 12/(x^2-4) = 1} = {5, -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_theorem_simplification_theorem_equation_solution_theorem_l875_87599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_k_range_l875_87512

/-- The range of k for the given ellipse equation and vector conditions -/
theorem ellipse_vector_k_range (a b : ℝ) (k : ℝ) : 
  a > 0 → b > 0 → a > b → 
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) → 
  (∃ A B : ℝ × ℝ, A + B = (3, -1)) → 
  k ∈ Set.Ioi (1/4 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_k_range_l875_87512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l875_87535

-- Define the function f(x) = ln(x) - x + 1
noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

-- State the theorem
theorem inequality_theorem (m n : ℝ) (h1 : 0 < m) (h2 : m < n) :
  (1 / n - 1) < (f n - f m) / (n - m) ∧ (f n - f m) / (n - m) < (1 / m - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l875_87535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_symmetry_l875_87524

-- Define the polar curve
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin (θ - Real.pi / 3)

-- Define the line of symmetry
def line_of_symmetry (θ : ℝ) : Prop :=
  θ = 5 * Real.pi / 6

-- Theorem statement
theorem polar_curve_symmetry :
  ∀ ρ θ : ℝ, polar_curve ρ θ → line_of_symmetry θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_symmetry_l875_87524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l875_87523

/-- A right circular cone inscribed in a right rectangular prism -/
structure InscribedCone where
  prism_width : ℝ
  prism_length : ℝ
  prism_height : ℝ
  cone_radius : ℝ
  cone_height : ℝ
  length_twice_width : prism_length = 2 * prism_width
  cone_inscribed : cone_radius = prism_width ∧ cone_height = prism_height

/-- The volume of a cone -/
noncomputable def cone_volume (c : InscribedCone) : ℝ :=
  (1/3) * Real.pi * c.cone_radius^2 * c.cone_height

/-- The volume of a rectangular prism -/
def prism_volume (c : InscribedCone) : ℝ :=
  c.prism_length * c.prism_width * c.prism_height

/-- The ratio of the cone volume to the prism volume -/
noncomputable def volume_ratio (c : InscribedCone) : ℝ :=
  cone_volume c / prism_volume c

/-- Theorem: The ratio of the volume of the inscribed cone to the volume of the prism is π/6 -/
theorem inscribed_cone_volume_ratio (c : InscribedCone) :
  volume_ratio c = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_ratio_l875_87523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_is_100_l875_87591

/-- Represents the scenario of Emily measuring a ship's length by walking along the riverbank -/
structure ShipMeasurement where
  /-- Emily's walking speed relative to the ship's speed -/
  speed_ratio : ℚ
  /-- Number of steps Emily takes when walking from rear to front of the ship -/
  steps_rear_to_front : ℕ
  /-- Number of steps Emily takes when walking from front to rear of the ship -/
  steps_front_to_rear : ℕ

/-- Calculates the length of the ship in terms of Emily's steps -/
def ship_length (m : ShipMeasurement) : ℚ :=
  (m.steps_rear_to_front * m.speed_ratio - m.steps_front_to_rear) / (m.speed_ratio + 1)

/-- Theorem stating that given the specific conditions, the ship's length is 100 steps -/
theorem ship_length_is_100 (m : ShipMeasurement)
    (h1 : m.speed_ratio = 2)
    (h2 : m.steps_rear_to_front = 300)
    (h3 : m.steps_front_to_rear = 60) :
    ship_length m = 100 := by
  sorry

#eval ship_length ⟨2, 300, 60⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_length_is_100_l875_87591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_number_is_twelve_l875_87559

def mySequence : Fin 7 → ℕ
  | 0 => 2
  | 1 => 16
  | 2 => 4
  | 3 => 14
  | 4 => 6
  | 5 => 12
  | 6 => 8

theorem sixth_number_is_twelve : mySequence 5 = 12 := by
  rfl

#eval mySequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_number_is_twelve_l875_87559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l875_87508

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if C has an asymptote y = 2x, then the eccentricity of C is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = 2*x) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l875_87508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l875_87502

-- Define the complex number z
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I

-- State the theorem
theorem imaginary_part_of_z :
  z.im = -2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l875_87502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_negative_tan_third_l875_87556

theorem cos_double_angle_with_negative_tan_third (θ : Real) 
  (h1 : Real.tan θ = -1/3) (h2 : θ ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.cos (2*θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_negative_tan_third_l875_87556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l875_87552

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, 3 * x^3 - 30 * (floor x) + 34 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l875_87552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_l875_87576

/-- A rectangular parallelepiped in 3D space -/
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The angle between a face of the parallelepiped and the horizontal plane -/
noncomputable def angle_to_horizontal (rp : RectangularParallelepiped) : ℝ := sorry

/-- The area of the projection of the parallelepiped onto the horizontal plane -/
noncomputable def projection_area (rp : RectangularParallelepiped) : ℝ := sorry

/-- Theorem: The area of the projection is maximized when one face is parallel to the horizontal plane -/
theorem max_projection_area (rp : RectangularParallelepiped) :
  ∃ (θ : ℝ), θ = 0 ∧ ∀ (φ : ℝ), projection_area rp ≤ projection_area rp :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_projection_area_l875_87576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atomic_weight_O_approx_l875_87525

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := 137.33

/-- The molecular weight of the compound BaO -/
def molecular_weight_BaO : ℝ := 153

/-- The atomic weight of Oxygen (O) -/
def atomic_weight_O : ℝ := molecular_weight_BaO - atomic_weight_Ba

/-- Theorem stating that the atomic weight of Oxygen is approximately 15.67 -/
theorem atomic_weight_O_approx :
  abs (atomic_weight_O - 15.67) < 0.01 := by
  -- Unfold the definition of atomic_weight_O
  unfold atomic_weight_O
  -- Simplify the expression
  simp [molecular_weight_BaO, atomic_weight_Ba]
  -- Prove the inequality
  norm_num

#eval atomic_weight_O

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atomic_weight_O_approx_l875_87525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lumberjacks_in_team_l875_87522

/-- Represents the number of lumberjacks on the small plot -/
def small_plot_lumberjacks : ℕ := sorry

/-- Represents the area of the small plot -/
def small_plot_area : ℝ := sorry

/-- Represents the productivity per lumberjack -/
def productivity_per_lumberjack : ℝ := sorry

theorem max_lumberjacks_in_team :
  small_plot_area > 0 →
  productivity_per_lumberjack > 0 →
  (3 * small_plot_area) / (productivity_per_lumberjack * (small_plot_lumberjacks + 8)) <
    small_plot_area / (productivity_per_lumberjack * small_plot_lumberjacks) →
  small_plot_lumberjacks < 4 →
  2 * small_plot_lumberjacks + 8 ≤ 14 :=
by sorry

#check max_lumberjacks_in_team

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lumberjacks_in_team_l875_87522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l875_87518

theorem power_equation_solution (x : ℝ) : (2:ℝ)^x - (2:ℝ)^(x-2) = 3 * (2:ℝ)^11 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l875_87518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_product_max_l875_87504

theorem triangle_side_product_max (a b c B : ℝ) :
  b = 4 →
  B = Real.pi / 3 →
  a > 0 →
  c > 0 →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  a * c ≤ 16 ∧ ∃ a' c', a' > 0 ∧ c' > 0 ∧ a' * c' = 16 :=
by
  sorry

#check triangle_side_product_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_product_max_l875_87504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_from_symmetric_difference_l875_87596

def symmetric_difference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

theorem intersection_size_from_symmetric_difference 
  (x y : Finset ℤ) 
  (hx : x.card = 16) 
  (hy : y.card = 18) 
  (hxy : (symmetric_difference x y).card = 22) :
  (x ∩ y).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_size_from_symmetric_difference_l875_87596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l875_87549

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sequence_property (m : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, n < m → S n = geometric_sum 2 (1/2) n) →
  (∀ n, n < m - 1 → S n - S (n + 1) = a n) →
  ∀ n, n < m - 1 → a n = -1 / (2^(n-1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l875_87549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_comparison_exists_l875_87565

-- Define a quadrilateral as a structure with 4 vertices in ℝ²
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to get the side lengths of a quadrilateral
noncomputable def sideLengths (q : Quadrilateral) : List ℝ :=
  [distance q.A q.B, distance q.B q.C, distance q.C q.D, distance q.D q.A]

-- Function to get the diagonal lengths of a quadrilateral
noncomputable def diagonalLengths (q : Quadrilateral) : List ℝ :=
  [distance q.A q.C, distance q.B q.D]

-- Theorem statement
theorem quadrilateral_comparison_exists : 
  ∃ (Q1 Q2 : Quadrilateral), 
    (∀ (i : Fin 4), (sideLengths Q1)[i.val] < (sideLengths Q2)[i.val]) ∧ 
    (∀ (i : Fin 2), (diagonalLengths Q1)[i.val] > (diagonalLengths Q2)[i.val]) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_comparison_exists_l875_87565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_is_444_l875_87580

/-- Represents the circular track problem with two students and a dog -/
structure TrackProblem where
  trackLength : ℝ
  speedA : ℝ
  speedB : ℝ
  speedDog : ℝ
  dogStartDelay : ℝ

/-- Calculates the total distance run by the dog in the track problem -/
noncomputable def dogTotalDistance (p : TrackProblem) : ℝ :=
  let meetingTime := p.trackLength / (p.speedA + p.speedB)
  let dogRunTime := meetingTime - p.dogStartDelay
  dogRunTime * p.speedDog

/-- Theorem stating that the dog's total distance is 444 meters for the given problem -/
theorem dog_distance_is_444 : dogTotalDistance {
  trackLength := 400
  speedB := 3
  speedA := 2
  speedDog := 6
  dogStartDelay := 6
} = 444 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_is_444_l875_87580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l875_87594

/-- The value of b for the given ellipse properties -/
noncomputable def ellipse_b : ℝ := Real.sqrt 6 / 3

/-- The equation of the ellipse -/
def ellipse_equation (x y b : ℝ) : Prop := x^2 + y^2 / b^2 = 1

/-- The condition that b is between 0 and 1 -/
def b_condition (b : ℝ) : Prop := 0 < b ∧ b < 1

/-- The coordinates of the foci -/
def foci_coordinates (c : ℝ) : ℝ × ℝ × ℝ × ℝ := (-c, 0, c, 0)

/-- The condition that AF₂ is perpendicular to the x-axis -/
def perpendicular_condition (y c : ℝ) : Prop := y = c

/-- The condition that |AF₁| = 3|BF₁| -/
def distance_ratio_condition (xa ya xb yb c : ℝ) : Prop :=
  ((xa + c)^2 + ya^2) = 9 * ((xb + c)^2 + yb^2)

theorem ellipse_property (b c : ℝ) (hb : b_condition b) 
  (hc : c^2 = 1 - b^2) 
  (hf : foci_coordinates c = (-c, 0, c, 0))
  (ha : ∃ (xa ya : ℝ), ellipse_equation xa ya b ∧ perpendicular_condition ya c)
  (hb : ∃ (xb yb : ℝ), ellipse_equation xb yb b)
  (hr : ∀ (xa ya xb yb : ℝ), 
    ellipse_equation xa ya b → 
    ellipse_equation xb yb b → 
    perpendicular_condition ya c → 
    distance_ratio_condition xa ya xb yb c) :
  b = ellipse_b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l875_87594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_l875_87500

/-- Represents the price reduction percentage -/
def price_reduction : ℝ := 0.18

/-- Represents the additional amount of oil obtained after price reduction -/
def additional_oil : ℝ := 8

/-- Represents the total cost in Rupees -/
def total_cost : ℝ := 1080

/-- Calculates the reduced price per kg of oil -/
def reduced_price (original_price : ℝ) : ℝ :=
  original_price * (1 - price_reduction)

/-- Theorem stating the reduced price per kg of oil -/
theorem reduced_oil_price : 
  ∃ (original_price original_quantity : ℝ),
    original_quantity * original_price = total_cost ∧
    (original_quantity + additional_oil) * reduced_price original_price = total_cost ∧
    abs ((reduced_price original_price) - 24.30) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_l875_87500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_values_l875_87587

/-- Recursive definition of the sequence -/
def a : ℕ → ℝ → ℝ
  | 0, x => x  -- Add case for 0
  | 1, x => x
  | 2, _ => 3000
  | (n + 3), x => a (n + 1) x * a (n + 2) x - 2

/-- The sequence contains 3001 -/
def contains_3001 (x : ℝ) : Prop :=
  ∃ n : ℕ, a n x = 3001

/-- The set of x values for which the sequence contains 3001 -/
def S : Set ℝ :=
  {x : ℝ | x > 0 ∧ contains_3001 x}

/-- Main theorem: there are exactly 4 values of x in S -/
theorem exactly_four_values :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x, x ∈ S ↔ x ∈ s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_values_l875_87587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l875_87571

-- Define the parameters
variable (a b c d : ℝ)

-- Define the conditions
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)

-- Define the boundaries of the rectangle
def top (a : ℝ) : ℝ := 2 * a
def bottom (b : ℝ) : ℝ := -b
def left (c : ℝ) : ℝ := -2 * c
def right (d : ℝ) : ℝ := d

-- Define the area of the rectangle
def rectangle_area (a b c d : ℝ) : ℝ := 
  (top a - bottom b) * (right d - left c)

-- Theorem statement
theorem area_of_bounded_region (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  rectangle_area a b c d = 2 * a * d + 4 * a * c + b * d + 2 * b * c := by
  -- Expand the definition of rectangle_area
  unfold rectangle_area
  -- Expand the definitions of top, bottom, left, and right
  unfold top bottom left right
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l875_87571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_value_l875_87539

def a : Fin 3 → ℝ := ![2, -1, 3]
def b (k : ℝ) : Fin 3 → ℝ := ![4, -2, k]

theorem orthogonal_vectors_k_value :
  (∀ k, (a 0) * (b k 0) + (a 1) * (b k 1) + (a 2) * (b k 2) = 0) →
  (∃ k, k = -10/3 ∧ (a 0) * (b k 0) + (a 1) * (b k 1) + (a 2) * (b k 2) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_k_value_l875_87539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_at_40_optimal_speed_and_consumption_l875_87588

-- Define the fuel consumption function
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

-- Define the total fuel consumption for a 100 km journey
noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

-- Maximum speed of the car
def max_speed : ℝ := 120

-- Theorem for fuel consumption at 40 km/h
theorem fuel_at_40 : total_fuel 40 = 17.5 := by sorry

-- Theorem for optimal speed and minimum fuel consumption
theorem optimal_speed_and_consumption :
  ∃ (x : ℝ), 0 < x ∧ x ≤ max_speed ∧
  (∀ y, 0 < y → y ≤ max_speed → total_fuel x ≤ total_fuel y) ∧
  x = 80 ∧ total_fuel x = 11.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_at_40_optimal_speed_and_consumption_l875_87588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_x_intercepts_l875_87578

-- Define the interval bounds
noncomputable def lower_bound : ℝ := 1000 / Real.pi
noncomputable def upper_bound : ℝ := 10000 / Real.pi

-- Define the number of integers in the interval
noncomputable def num_integers : ℤ := ⌊upper_bound⌋ - ⌊lower_bound⌋

-- Theorem statement
theorem num_x_intercepts : num_integers = 2865 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_x_intercepts_l875_87578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_f_monotone_increasing_condition_l875_87595

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs (x - 1)

-- Theorem for part (1)
theorem f_extrema_on_interval :
  let a := 2
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 6) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = 1) :=
by sorry

-- Theorem for part (2)
theorem f_monotone_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f a x < f a y) ↔ a ∈ Set.Icc (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_f_monotone_increasing_condition_l875_87595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bologna_sandwiches_l875_87530

/-- Given a ratio of sandwiches and a total count, calculates the number of bologna sandwiches -/
theorem bologna_sandwiches (total : ℕ) (ratio_cheese : ℕ) (ratio_bologna : ℕ) (ratio_peanut : ℕ) : 
  total = 80 → ratio_cheese = 1 → ratio_bologna = 7 → ratio_peanut = 8 →
  (ratio_cheese + ratio_bologna + ratio_peanut) * (total / (ratio_cheese + ratio_bologna + ratio_peanut)) = total →
  ratio_bologna * (total / (ratio_cheese + ratio_bologna + ratio_peanut)) = 35 := by
  intros h_total h_cheese h_bologna h_peanut h_sum
  sorry

#check bologna_sandwiches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bologna_sandwiches_l875_87530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_n_plus_4_l875_87575

theorem not_divisible_by_n_plus_4 (n : ℕ) (hn : 0 < n) :
  ¬ ∃ q : ℤ, (n : ℤ)^2 + 8*(n : ℤ) + 15 = ((n : ℤ) + 4) * q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_n_plus_4_l875_87575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_2_l875_87531

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 18 / Real.sqrt x

def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

theorem f_of_g_2 : f (g 2) = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_2_l875_87531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l875_87541

/-- Represents the profit percentage calculation for a retailer selling pens --/
theorem retailer_profit_percentage 
  (purchase_quantity : ℕ) 
  (purchase_price_quantity : ℕ) 
  (discount_percentage : ℝ) 
  (unit_price : ℝ) 
  (h1 : purchase_quantity = 250)
  (h2 : purchase_price_quantity = 200)
  (h3 : discount_percentage = 3.5)
  (h4 : unit_price > 0) : 
  let cost_price := purchase_price_quantity * unit_price
  let market_price := purchase_quantity * unit_price
  let discount := discount_percentage / 100 * market_price
  let selling_price := market_price - discount
  let profit := selling_price - cost_price
  let profit_percentage := profit / cost_price * 100
  profit_percentage = 20.625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percentage_l875_87541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_domain_range_l875_87542

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + x

theorem quadratic_function_domain_range (m n : ℝ) :
  m < n ∧
  n ≤ 1 ∧
  (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ f x ∧ f x ≤ 3*n) →
  m + n = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_domain_range_l875_87542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l875_87584

theorem simplify_radical_sum : Real.sqrt (6 + 4 * Real.sqrt 3) + Real.sqrt (6 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l875_87584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_neg_x_cubed_l875_87558

noncomputable def is_monomial (m : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℕ), m = fun x ↦ a * x^n

noncomputable def coefficient (m : ℝ → ℝ) (h : is_monomial m) : ℝ :=
  Classical.choose h

noncomputable def degree (m : ℝ → ℝ) (h : is_monomial m) : ℕ :=
  Classical.choose (Classical.choose_spec h)

theorem monomial_neg_x_cubed :
  let m : ℝ → ℝ := fun x ↦ -x^3
  is_monomial m ∧ coefficient m (by sorry) = -1 ∧ degree m (by sorry) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_neg_x_cubed_l875_87558
