import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_sum_2013_l141_14100

/-- The ones digit of the sum of odd numbers from 1 to n, each raised to the power of m -/
def ones_digit_sum (n m : ℕ) : ℕ :=
  (Finset.sum (Finset.range ((n + 1) / 2)) (fun i => (2 * i + 1) ^ m)) % 10

/-- The theorem statement -/
theorem ones_digit_sum_2013 : ones_digit_sum 2013 2013 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ones_digit_sum_2013_l141_14100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_and_centroid_l141_14176

-- Define the triangle ABC and points O and N
variable (A B C O N : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : dist O A = dist O B)
variable (h2 : dist O B = dist O C)
variable (h3 : (N -ᵥ A) + (N -ᵥ B) + (N -ᵥ C) = 0)

-- Define what it means for O to be the circumcenter
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

-- Define what it means for N to be the centroid
def is_centroid (N A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  (N -ᵥ A) + (N -ᵥ B) + (N -ᵥ C) = 0

-- State the theorem
theorem circumcenter_and_centroid (A B C O N : EuclideanSpace ℝ (Fin 2))
  (h1 : dist O A = dist O B) (h2 : dist O B = dist O C)
  (h3 : (N -ᵥ A) + (N -ᵥ B) + (N -ᵥ C) = 0) :
  is_circumcenter O A B C ∧ is_centroid N A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_and_centroid_l141_14176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bruno_pens_count_l141_14153

/-- The number of items in one dozen -/
def dozen : ℕ := 12

/-- The number of pens Bruno wants to buy, expressed in dozens -/
def brunos_order : ℚ := 5/2

/-- The number of pens Bruno will have -/
def pens_count : ℕ := 30

/-- Theorem stating that Bruno's order of two and one-half dozens of pens results in 30 pens -/
theorem bruno_pens_count : 
  (brunos_order * ↑dozen).floor = pens_count := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bruno_pens_count_l141_14153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l141_14177

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y + 1 = 0

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = x ∨ y = -x

-- Define the intersection points of asymptotes and circle
def intersection_points (x y : ℝ) : Prop := asymptotes x y ∧ circle_eq x y

-- Theorem statement
theorem chord_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection_points x₁ y₁ ∧ 
    intersection_points x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧ 
    y₁ ≠ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l141_14177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l141_14182

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.log x + (x + 1) / x

/-- The inequality condition -/
def inequality_holds (k : ℝ) : Prop :=
  ∀ x > 1, f x > ((x - k) * Real.log x) / (x - 1)

/-- The main theorem -/
theorem inequality_condition (k : ℝ) :
  inequality_holds k ↔ k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l141_14182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_c_drain_rate_is_25_l141_14147

/-- Represents the water tank system with three pipes --/
structure WaterTankSystem where
  tankCapacity : ℝ
  pipeAFillRate : ℝ
  pipeBFillRate : ℝ
  pipeAOpenTime : ℝ
  pipeBOpenTime : ℝ
  pipeCDrainTime : ℝ
  totalFillTime : ℝ

/-- Calculates the drain rate of Pipe C --/
noncomputable def calculatePipeCDrainRate (system : WaterTankSystem) : ℝ :=
  let cycleTime := system.pipeAOpenTime + system.pipeBOpenTime + system.pipeCDrainTime
  let numCycles := system.totalFillTime / cycleTime
  let netFillPerCycle := system.tankCapacity / numCycles
  let fillPerCycle := system.pipeAFillRate * system.pipeAOpenTime + system.pipeBFillRate * system.pipeBOpenTime
  (fillPerCycle - netFillPerCycle) / system.pipeCDrainTime

theorem pipe_c_drain_rate_is_25 (system : WaterTankSystem) 
  (h1 : system.tankCapacity = 5000)
  (h2 : system.pipeAFillRate = 200)
  (h3 : system.pipeBFillRate = 50)
  (h4 : system.pipeAOpenTime = 1)
  (h5 : system.pipeBOpenTime = 2)
  (h6 : system.pipeCDrainTime = 2)
  (h7 : system.totalFillTime = 100) :
  calculatePipeCDrainRate system = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_c_drain_rate_is_25_l141_14147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_2000_identity_l141_14136

/-- A linear function of the form f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- The n-th iteration of a function f -/
def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- Theorem: Characterization of linear functions that are the identity after 2000 iterations -/
theorem linear_function_2000_identity (a b : ℝ) :
  (∀ x, iterate (linear_function a b) 2000 x = x) ↔ 
  ((a = 1 ∧ b = 0) ∨ (a = -1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_2000_identity_l141_14136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_squares_sum_power_of_two_l141_14192

theorem odd_squares_sum_power_of_two (n : ℕ) :
  ∃ (x y : ℕ), Odd x ∧ Odd y ∧ x^2 + 7*y^2 = 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_squares_sum_power_of_two_l141_14192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grey_rectangles_area_l141_14132

theorem grey_rectangles_area (blue_area red_area rectangle_area : ℝ) 
  (h_blue : blue_area = 36)
  (h_red : red_area = 49)
  (h_rectangle : rectangle_area = 78) : ℝ := by
  -- Canvas side length
  let canvas_side := Real.sqrt blue_area + Real.sqrt red_area
  -- Canvas area
  let canvas_area := canvas_side ^ 2
  -- Area of the right half of the canvas
  let right_half_area := canvas_area / 2
  -- Sum of grey rectangles area
  let grey_area := right_half_area - rectangle_area + blue_area
  -- Theorem statement
  have : grey_area = 42 := by sorry
  exact grey_area


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grey_rectangles_area_l141_14132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_road_time_l141_14195

/-- Calculates the total time Steve spends on the roads each day. -/
noncomputable def total_time_on_roads (distance : ℝ) (speed_back : ℝ) : ℝ :=
  let speed_to := speed_back / 2
  let time_to := distance / speed_to
  let time_back := distance / speed_back
  time_to + time_back

/-- Proves that Steve spends 6 hours on the roads each day. -/
theorem steve_road_time : total_time_on_roads 30 15 = 6 := by
  -- Unfold the definition of total_time_on_roads
  unfold total_time_on_roads
  -- Simplify the expression
  simp
  -- The proof is completed using numerical computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_road_time_l141_14195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_area_implies_angle_l141_14139

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S = (1/4)(b^2 + c^2 - a^2), then angle A = π/4 -/
theorem triangle_special_area_implies_angle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → C > 0 →
  A + B + C = π →
  a = b * Real.sin C →
  b = c * Real.sin A →
  c = a * Real.sin B →
  ((b^2 + c^2 - a^2) / 4 = (1/2) * b * c * Real.sin A) →
  A = π/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_area_implies_angle_l141_14139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_70_factorial_l141_14131

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define a function to get the last two nonzero digits
def lastTwoNonzeroDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let nonzeroDigits := digits.filter (· ≠ 0)
  match nonzeroDigits.reverse with
  | [] => 0
  | [a] => a
  | a :: b :: _ => b * 10 + a

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial :
  lastTwoNonzeroDigits (factorial 70) = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_of_70_factorial_l141_14131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l141_14104

def S : ℕ → ℕ 
  | 1 => 1
  | 3 => 15
  | 5 => 65
  | _ => 0  -- placeholder for other values

theorem sequence_property (a b c : ℚ) :
  (∀ n : ℕ+, S (2*n - 1) = (2*n - 1) * (a*(n : ℚ)^2 + b*(n : ℚ) + c)) →
  3*a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l141_14104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_identities_l141_14185

theorem algebraic_identities :
  (∀ a : ℝ, a^2 * (-a)^3 * (-a^4) = a^9) ∧
  (∀ a : ℝ, -2*a^6 - (-3*a^2)^3 = 25*a^6) ∧
  (3^(0:ℝ) - 2^(-3:ℝ) + (-3)^2 - (1/4:ℝ)^(-1:ℝ) = 5 + 7/8) ∧
  (∀ p q : ℝ, (p-q)^4 / (q-p)^3 * (p-q)^2 = -(p-q)^3) := by
  sorry

#check algebraic_identities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebraic_identities_l141_14185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_fraction_is_one_fourth_l141_14140

/-- Represents the contents of a cup --/
structure CupContents where
  tea : ℚ
  sugar : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : CupState :=
  { cup1 := { tea := 6, sugar := 0 },
    cup2 := { tea := 0, sugar := 4 } }

def transfer_tea (state : CupState) (amount : ℚ) : CupState :=
  { cup1 := { tea := state.cup1.tea - amount, sugar := state.cup1.sugar },
    cup2 := { tea := state.cup2.tea + amount, sugar := state.cup2.sugar } }

def add_tea_to_cup1 (state : CupState) (amount : ℚ) : CupState :=
  { cup1 := { tea := state.cup1.tea + amount, sugar := state.cup1.sugar },
    cup2 := state.cup2 }

def transfer_back (state : CupState) (amount : ℚ) : CupState :=
  let total_cup2 := state.cup2.tea + state.cup2.sugar
  let tea_ratio := state.cup2.tea / total_cup2
  let sugar_ratio := state.cup2.sugar / total_cup2
  { cup1 := { tea := state.cup1.tea + amount * tea_ratio,
              sugar := state.cup1.sugar + amount * sugar_ratio },
    cup2 := { tea := state.cup2.tea - amount * tea_ratio,
              sugar := state.cup2.sugar - amount * sugar_ratio } }

def final_state : CupState :=
  transfer_back (add_tea_to_cup1 (transfer_tea initial_state 2) 2) 3

theorem sugar_fraction_is_one_fourth :
  let total_liquid := final_state.cup1.tea + final_state.cup1.sugar
  final_state.cup1.sugar / total_liquid = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_fraction_is_one_fourth_l141_14140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_is_63_l141_14172

/-- The sequence of partial sums of {aₙ} -/
def S : ℕ → ℝ := sorry

/-- The sequence {aₙ} -/
def a : ℕ → ℝ := sorry

/-- The property that aₙ, n+1/2, aₙ₊₁ form an arithmetic sequence for n ∈ ℕ₊ -/
axiom arithmetic_property : ∀ n : ℕ+, 2 * (n + 1/2 : ℝ) = a n + a (n + 1)

/-- The sum of the first n terms is 2020 -/
axiom sum_2020 : ∃ n : ℕ+, S n = 2020

/-- The second term is less than 3 -/
axiom a2_lt_3 : a 2 < 3

/-- The theorem to be proved -/
theorem max_n_is_63 : ∃ n : ℕ+, S n = 2020 ∧ ∀ m : ℕ+, S m = 2020 → m ≤ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_is_63_l141_14172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_periodic_l141_14178

/-- A word is a list of characters from the lowercase alphabet --/
def Word := List Char

/-- A word is periodic if it's formed by repeating a shorter word at least twice --/
def IsPeriodic (w : Word) : Prop :=
  ∃ (subword : Word), subword ≠ [] ∧ subword.length < w.length ∧
    w = (List.replicate (w.length / subword.length) subword).join

/-- Two words are equal after removing their first letters --/
def EqualWithoutFirst (w1 w2 : Word) : Prop :=
  w1.tail = w2.tail

theorem not_both_periodic (w1 w2 : Word) :
  w1.length = w2.length →
  w1.head? ≠ w2.head? →
  EqualWithoutFirst w1 w2 →
  ¬(IsPeriodic w1 ∧ IsPeriodic w2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_both_periodic_l141_14178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_properties_l141_14151

def SpecialSet (A : Set ℕ) : Prop :=
  (∀ m n, m ∈ A → n ∈ A → m + n ∈ A) ∧
  (¬ ∃ p, Nat.Prime p ∧ ∀ a ∈ A, p ∣ a)

theorem special_set_properties (A : Set ℕ) (hA : SpecialSet A) :
  (∀ n₁ n₂, n₁ ∈ A → n₂ ∈ A → n₂ > n₁ + 1 → ∃ m₁ m₂, m₁ ∈ A ∧ m₂ ∈ A ∧ 0 < m₂ - m₁ ∧ m₂ - m₁ < n₂ - n₁) ∧
  (∃ n₀, n₀ ∈ A ∧ n₀ + 1 ∈ A) ∧
  (∀ n₀, n₀ ∈ A → n₀ + 1 ∈ A → ∀ n, n ≥ n₀^2 → n ∈ A) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_properties_l141_14151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l141_14167

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 3) / (|x - 2| + |x + 2|)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x < -2 ∨ (-2 < x ∧ x < 2) ∨ 2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l141_14167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_minimum_l141_14116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem extreme_value_and_minimum (a : ℝ) (h1 : a > 1/2) :
  (∃ (x : ℝ), x = 3 ∧ DifferentiableAt ℝ (f a) x ∧ deriv (f a) x = 0) →
  (∃ (x : ℝ), x ∈ Set.Icc 0 (2 * a) ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 (2 * a) → f a x ≤ f a y) →
  (∃ (x : ℝ), x ∈ Set.Icc 0 (2 * a) ∧ f a x = -a^2) →
  (HasDerivAt (f a) 18 0) ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_minimum_l141_14116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_width_calculation_l141_14186

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.thickness

/-- Theorem stating that the width of each brick is approximately 11.21 cm -/
theorem brick_width_calculation 
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (num_bricks : ℕ) :
  brick.length = 25 ∧ 
  brick.height = 6 ∧ 
  wall.length = 850 ∧ 
  wall.height = 600 ∧ 
  wall.thickness = 22.5 ∧ 
  num_bricks = 6800 →
  ∃ ε > 0, |brick.width - 11.21| < ε := by
  sorry

#check brick_width_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_width_calculation_l141_14186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_evaluation_l141_14171

-- Define the propositions
def prop1 (X Y : ℝ) : Prop := X + Y = 0 → (X = -Y ∧ Y = -X)

-- We'll use 'Triangle' instead of 'Set ℝ × ℝ' for better representation
structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

def prop2 (T1 T2 : Triangle) : Prop := sorry -- Placeholder for congruence and area

def prop3 (q : ℝ) : Prop := q ≤ 1 → ∃ x : ℝ, x^2 + 2*x + q = 0

def prop4 (T : Triangle) : Prop := sorry -- Placeholder for scalene and equal angles

-- Define the logical transformations
def inverse (P Q : Prop) : Prop := (Q → P)
def contrapositive (P Q : Prop) : Prop := (¬Q → ¬P)
def negation (P : Prop) : Prop := ¬P

-- State the theorem
theorem proposition_evaluation :
  (∀ X Y : ℝ, inverse (prop1 X Y) (X = -Y ∧ Y = -X)) ∧
  (∀ q : ℝ, contrapositive (prop3 q) (¬∃ x : ℝ, x^2 + 2*x + q = 0)) ∧
  ¬(∀ T1 T2 : Triangle, negation (prop2 T1 T2)) ∧
  ¬(∀ T : Triangle, inverse (prop4 T) sorry) := -- Placeholder for EqualAngles T
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_evaluation_l141_14171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_sum_l141_14160

theorem greatest_power_sum (c d : ℕ) : 
  d > 1 ∧ 
  c^d < 500 ∧ 
  (∀ (x y : ℕ), y > 1 ∧ x^y < 500 → x^y ≤ c^d) → 
  c + d = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_sum_l141_14160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangents_theorem_l141_14155

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Given two circles are tangent if they touch at exactly one point -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- A line is tangent to a circle if it touches the circle at exactly one point -/
def is_tangent_to_circle (l : Line) (c : Circle) : Prop := sorry

/-- Two lines are parallel if they have the same direction or opposite directions -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- A point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Two lines intersect at a point -/
def lines_intersect_at (l1 l2 : Line) (p : ℝ × ℝ) : Prop := sorry

/-- A point is equidistant from three other points -/
def equidistant (p p1 p2 p3 : ℝ × ℝ) : Prop := sorry

theorem circle_tangents_theorem (C C' C'' : Circle) (L' L'' : Line) 
  (A B X Y Z Q : ℝ × ℝ) :
  are_parallel L' L'' →
  is_tangent_to_circle L' C →
  is_tangent_to_circle L'' C →
  is_tangent_to_circle L' C' →
  point_on_line A L' →
  are_tangent C C' →
  point_on_circle X C →
  point_on_circle X C' →
  is_tangent_to_circle L'' C'' →
  point_on_line B L'' →
  are_tangent C C'' →
  point_on_circle Y C →
  point_on_circle Y C'' →
  are_tangent C' C'' →
  point_on_circle Z C' →
  point_on_circle Z C'' →
  lines_intersect_at (Line.mk A Y) (Line.mk B X) Q →
  equidistant Q X Y Z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangents_theorem_l141_14155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l141_14138

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ A < Real.pi →
  Real.cos A = 4/5 →
  b = 2 →
  S = 3 →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l141_14138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_notation_l141_14161

/-- Represents temperature in Celsius -/
structure Temperature where
  value : Int
  unit : String
  deriving Repr

/-- Notation for temperature above or below zero -/
def notate (temp : Temperature) : String :=
  if temp.value > 0 then
    s!"+{temp.value}{temp.unit}"
  else if temp.value < 0 then
    s!"{temp.value}{temp.unit}"
  else
    s!"0{temp.unit}"

/-- Given: 5°C above zero is denoted as +5°C -/
axiom above_zero_notation : notate { value := 5, unit := "°C" } = "+5°C"

/-- Theorem: 3°C below zero is denoted as -3°C -/
theorem below_zero_notation : notate { value := -3, unit := "°C" } = "-3°C" := by
  rfl

#eval notate { value := 5, unit := "°C" }
#eval notate { value := -3, unit := "°C" }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_below_zero_notation_l141_14161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l141_14188

theorem problem_statement 
  (a b c d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (hab : a^2 + b^2 = a*b + 1) 
  (hcd : c*d > 1) : 
  (a + b ≤ 2) ∧ (Real.sqrt (a*c) + Real.sqrt (b*d) < c + d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l141_14188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_6_l141_14127

/-- Represents the total distance in kilometers --/
def D : ℝ := 6

/-- The equation representing the relationship between distance and time --/
axiom distance_time_equation : D / 6 + D / 15 = 1.4

/-- Theorem stating that the total distance is 6 km --/
theorem total_distance_is_6 : D = 6 := by
  -- The proof is omitted for brevity
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_6_l141_14127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heaviest_minus_lightest_is_seven_l141_14128

/-- Represents the weights of four boxes -/
structure BoxWeights where
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  a_lt_b : a < b
  b_lt_c : b < c
  c_lt_d : c < d

/-- The sum of weights for each pair of boxes -/
def pairSums (w : BoxWeights) : Finset ℚ :=
  {w.a + w.b, w.a + w.c, w.a + w.d, w.b + w.c, w.b + w.d, w.c + w.d}

theorem heaviest_minus_lightest_is_seven (w : BoxWeights) 
  (h1 : pairSums w ⊆ {22, 23, 25, 27, 29, 30})
  (h2 : (pairSums w).card = 6) : 
  w.d - w.a = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heaviest_minus_lightest_is_seven_l141_14128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l141_14179

def sample : List ℝ := [1, 3, 2, 5]

noncomputable def mean (xs : List ℝ) (x : ℝ) : ℝ := (xs.sum + x) / (xs.length + 1 : ℝ)

noncomputable def variance (xs : List ℝ) (x : ℝ) (m : ℝ) : ℝ :=
  ((xs.map (λ y => (y - m)^2)).sum + (x - m)^2) / (xs.length + 1 : ℝ)

noncomputable def standardDeviation (xs : List ℝ) (x : ℝ) (m : ℝ) : ℝ :=
  Real.sqrt (variance xs x m)

theorem sample_standard_deviation :
  ∃ x : ℝ, mean sample x = 3 ∧
           standardDeviation sample x 3 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_standard_deviation_l141_14179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_necessary_not_sufficient_l141_14181

-- Define custom types for Plane and Line
structure Plane : Type
structure Line : Type

variable (α β γ : Plane) (l m : Line)

-- α and β are two different planes
axiom different_planes : α ≠ β

-- Define a custom subset relation for Line and Plane
def LineInPlane (l : Line) (p : Plane) : Prop := sorry

-- Line l is in plane β
axiom l_in_β : LineInPlane l β

-- Define parallelism for planes
def plane_parallel (p q : Plane) : Prop := sorry

-- Define parallelism between a line and a plane
def line_parallel_plane (m : Line) (p : Plane) : Prop := sorry

theorem parallel_necessary_not_sufficient :
  (plane_parallel α β → line_parallel_plane l α) ∧
  ∃ (γ : Plane) (m : Line), LineInPlane m γ ∧ line_parallel_plane m α ∧ ¬plane_parallel α γ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_necessary_not_sufficient_l141_14181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_three_l141_14141

def left_spinner : Finset ℕ := {5, 6, 7, 8}
def right_spinner : Finset ℕ := {1, 2, 3, 4, 5}

def is_multiple_of_three (n : ℕ) : Bool :=
  n % 3 = 0

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (left_spinner.product right_spinner).filter (fun p => is_multiple_of_three (p.1 * p.2))

theorem probability_multiple_of_three :
  (favorable_outcomes.card : ℚ) / (left_spinner.card * right_spinner.card) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_three_l141_14141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_area_approx_l141_14102

/-- Rectangle EFGH with given dimensions and circles -/
structure Rectangle :=
  (ef : ℝ)
  (fg : ℝ)
  (circle_e_radius : ℝ)
  (circle_f_radius : ℝ)
  (circle_g_radius : ℝ)
  (external_circle_radius : ℝ)

/-- Calculate the unoccupied area in the rectangle -/
noncomputable def unoccupied_area (r : Rectangle) : ℝ :=
  let rectangle_area := r.ef * r.fg
  let quarter_circles_area := (r.circle_e_radius^2 + r.circle_f_radius^2 + r.circle_g_radius^2) * Real.pi / 4
  let external_quarter_circle_area := r.external_circle_radius^2 * Real.pi / 4
  rectangle_area - (quarter_circles_area + external_quarter_circle_area)

/-- Theorem stating the unoccupied area is approximately 7.5 square units -/
theorem unoccupied_area_approx (r : Rectangle) 
  (h1 : r.ef = 4)
  (h2 : r.fg = 6)
  (h3 : r.circle_e_radius = 2)
  (h4 : r.circle_f_radius = 3)
  (h5 : r.circle_g_radius = 1.5)
  (h6 : r.external_circle_radius = 2.5) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |unoccupied_area r - 7.5| < ε :=
by
  sorry

#check unoccupied_area_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_area_approx_l141_14102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_inequality_l141_14117

def sequence_a : ℕ → ℝ
  | 0 => 1  -- We define a_0 = 1 to handle the case when n = 0
  | 1 => 1
  | 2 => 3
  | 3 => 6
  | n + 4 => 3 * sequence_a (n + 3) - sequence_a (n + 2) - 2 * sequence_a (n + 1)

theorem sequence_a_inequality : ∀ n : ℕ, n > 3 → sequence_a n > 3 * 2^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_inequality_l141_14117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l141_14134

-- Problem 1
theorem problem_one : 
  (0.008 : ℝ) ^ (1/3 : ℝ) + (Real.sqrt 2 - Real.pi) ^ (0 : ℝ) - (125/64 : ℝ) ^ (-1/3 : ℝ) = 2/5 := by
  sorry

-- Problem 2
theorem problem_two : 
  (((Real.log 2 / Real.log 3) + (Real.log 2 / Real.log 9)) * 
   ((Real.log 3 / Real.log 4) + (Real.log 3 / Real.log 8))) / 
  (Real.log 600 - (1/2 : ℝ) * Real.log 0.036 - (1/2 : ℝ) * Real.log 0.1) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l141_14134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_88_l141_14158

def count_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then count_sequence n + 2 else count_sequence n + 1

theorem thirtieth_term_is_88 : count_sequence 29 = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirtieth_term_is_88_l141_14158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_determination_l141_14148

/-- A device that reports the sum of two weights when given 10 weights -/
def Device (α : Type*) := Finset α → ℕ

/-- The property that all weights in a set are distinct -/
def DistinctWeights (S : Finset ℕ) : Prop := ∀ x y, x ∈ S → y ∈ S → x = y → x = y

theorem weight_determination
  (n : ℕ)
  (S : Finset ℕ)
  (h_card : S.card = n)
  (h_distinct : DistinctWeights S)
  (device : Device ℕ)
  (h_device : ∀ T : Finset ℕ, T ⊆ S → T.card = 10 → ∃ x y, x ∈ T ∧ y ∈ T ∧ x ≠ y ∧ device T = x + y) :
  ∃ w, w ∈ S ∧ ∃ k : ℕ, k = w :=
sorry

#check weight_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_determination_l141_14148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_doubling_red_l141_14149

theorem marble_fraction_after_doubling_red (total : ℝ) (h : total > 0) :
  let initial_blue := (4 / 9 : ℝ) * total
  let initial_red := (1 / 3 : ℝ) * total
  let initial_green := total - initial_blue - initial_red
  let new_red := 2 * initial_red
  let new_total := initial_blue + new_red + initial_green
  new_red / new_total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_doubling_red_l141_14149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_PFQ_l141_14130

-- Define the trajectory curve
noncomputable def trajectory (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ y ≠ 0

-- Define points A, B, and F
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
def F : ℝ × ℝ := (1, 0)

-- Define the tangent line at a point on the curve
noncomputable def tangent_line (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  let (px, py) := p
  py / 3 * (x - px) + py

-- Define point Q as the intersection of tangent line with x = 4
noncomputable def Q (p : ℝ × ℝ) : ℝ × ℝ :=
  (4, tangent_line p 4)

-- State the theorem
theorem right_angle_PFQ (p : ℝ × ℝ) :
  trajectory p.1 p.2 →
  let q := Q p
  (p.1 - F.1) * (q.1 - F.1) + (p.2 - F.2) * (q.2 - F.2) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_PFQ_l141_14130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_inclination_angle_range_l141_14168

open Real

noncomputable def inclination_angle_range (k : ℝ) : Set ℝ :=
  {α | (0 ≤ α ∧ α ≤ π/4) ∨ (3*π/4 ≤ α ∧ α < π)}

theorem slope_to_inclination_angle_range :
  ∀ k : ℝ, |k| ≤ 1 →
  ∀ α : ℝ, α ∈ inclination_angle_range k ↔ k = tan α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_inclination_angle_range_l141_14168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_and_line_in_plane_l141_14135

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relationships
def parallel_to_plane (l : Line) (α : Plane) : Prop :=
  sorry

def contained_in_plane (a : Line) (α : Plane) : Prop :=
  sorry

def parallel_or_skew (l1 l2 : Line) : Prop :=
  sorry

-- State the theorem
theorem line_parallel_to_plane_and_line_in_plane 
  (l a : Line) (α : Plane) 
  (h1 : parallel_to_plane l α) 
  (h2 : contained_in_plane a α) : 
  parallel_or_skew l a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_plane_and_line_in_plane_l141_14135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_competition_results_l141_14183

/-- Represents a class with scores for costume, pitch, and innovation -/
structure ClassScore where
  costume : ℚ
  pitch : ℚ
  innovation : ℚ

/-- Calculate the average score of a class -/
def average_score (c : ClassScore) : ℚ :=
  (c.costume + c.pitch + c.innovation) / 3

/-- Calculate the weighted score of a class with weights 1:7:2 -/
def weighted_score (c : ClassScore) : ℚ :=
  (c.costume + 7 * c.pitch + 2 * c.innovation) / 10

/-- The two classes in the competition -/
def class1 : ClassScore := ⟨90, 77, 85⟩
def class2 : ClassScore := ⟨74, 95, 80⟩

theorem choir_competition_results :
  (average_score class1 > average_score class2) ∧
  (weighted_score class2 > weighted_score class1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_competition_results_l141_14183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_preimage_of_one_eighth_l141_14174

-- Define the set A
def A : Set ℝ := {x | x ≤ 0}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Theorem statement
theorem unique_preimage_of_one_eighth :
  ∃! (x : ℝ), x ∈ A ∧ f x = 1/8 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_preimage_of_one_eighth_l141_14174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_a_l141_14165

/-- A function f: ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The function f(x) = (1/3)x³ - ax² + 2x + 3 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotone_increasing_range_a :
  (∀ a : ℝ, MonotonicallyIncreasing (f a)) ↔ (∀ a : ℝ, -Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2) := by
  sorry

#check monotone_increasing_range_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_range_a_l141_14165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_on_tires_l141_14152

-- Define the given conditions
def original_price : ℚ := 60
def discount_rate : ℚ := 15 / 100
def tax_rate : ℚ := 8 / 100
def num_tires : ℕ := 4

-- Define the theorem
theorem total_spent_on_tires : 
  (original_price * (1 - discount_rate) * (1 + tax_rate) * num_tires : ℚ) = 220.32 := by
  -- Proof goes here
  sorry

#eval (original_price * (1 - discount_rate) * (1 + tax_rate) * num_tires : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_spent_on_tires_l141_14152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasper_win_prob_l141_14107

/-- The probability of Jasper's coin landing heads -/
def jasper_prob : ℚ := 2/7

/-- The probability of Kira's coin landing heads -/
def kira_prob : ℚ := 1/4

/-- The probability of both players getting tails in one round -/
def both_tails_prob : ℚ := (1 - kira_prob) * (1 - jasper_prob)

/-- The game where Jasper and Kira alternately toss coins until someone gets a head -/
def coin_game (jasper_prob kira_prob : ℚ) : Prop :=
  jasper_prob ∈ Set.Icc (0 : ℚ) 1 ∧ 
  kira_prob ∈ Set.Icc (0 : ℚ) 1 ∧
  kira_prob ≠ 1 ∧
  jasper_prob ≠ 1

/-- The theorem stating the probability of Jasper winning the game -/
theorem jasper_win_prob (h : coin_game jasper_prob kira_prob) : 
  (jasper_prob * (1 / (1 - both_tails_prob))) = 30/91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasper_win_prob_l141_14107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_i_over_one_plus_i_l141_14144

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.im (i / (1 + i)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_i_over_one_plus_i_l141_14144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sequence_exists_l141_14129

-- Define a sequence of natural numbers
def Sequence := ℕ → ℕ

-- Define what it means for a number to be square-free
def IsSquareFree (n : ℕ) : Prop := ∀ p : ℕ, Nat.Prime p → ¬(p ^ 2 ∣ n)

-- Define what it means for a sequence to satisfy the given condition
def SatisfiesCondition (s : Sequence) : Prop :=
  ∃ N : ℕ, ∀ k ≥ N, ∀ start : ℕ,
    IsSquareFree (Finset.sum (Finset.range k) (λ i => s (start + i))) ↔ IsSquareFree k

-- State the theorem
theorem no_sequence_exists :
  ¬ ∃ s : Sequence,
    (Function.Injective s) ∧  -- distinct terms
    (∀ n m : ℕ, n ≠ m → s n ≠ s m) ∧  -- non-constant
    SatisfiesCondition s :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sequence_exists_l141_14129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_green_is_seven_fiftieths_l141_14137

/-- A tile is green if its number is congruent to 3 mod 7 -/
def is_green (n : ℕ) : Prop := n % 7 = 3

/-- The set of all tiles -/
def all_tiles : Finset ℕ := Finset.range 100

/-- The set of green tiles -/
def green_tiles : Finset ℕ := all_tiles.filter (fun n => n % 7 = 3)

/-- The probability of choosing a green tile -/
noncomputable def prob_green : ℚ := (green_tiles.card : ℚ) / (all_tiles.card : ℚ)

theorem prob_green_is_seven_fiftieths : prob_green = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_green_is_seven_fiftieths_l141_14137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l141_14124

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- State the theorem
theorem range_of_a (a : ℝ) : f a < 1 ↔ -3 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l141_14124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l141_14191

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = (4/3)*(x - 1)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define points A and B
variable (A B : ℝ × ℝ)

-- State that A and B are on both the parabola and the line
axiom A_on_parabola : parabola A.1 A.2
axiom A_on_line : line A.1 A.2
axiom B_on_parabola : parabola B.1 B.2
axiom B_on_line : line B.1 B.2

-- Define lambda (using 'lambda' instead of 'λ' to avoid syntax issues)
variable (lambda : ℝ)

-- State that lambda > 1
axiom lambda_gt_one : lambda > 1

-- Define the vector equality condition
def vector_condition (A B : ℝ × ℝ) (lambda : ℝ) : Prop :=
  A.1 - focus.1 = lambda * (focus.1 - B.1) ∧ A.2 - focus.2 = lambda * (focus.2 - B.2)

-- Theorem statement
theorem parabola_line_intersection (A B : ℝ × ℝ) (lambda : ℝ) :
  vector_condition A B lambda → lambda = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l141_14191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l141_14169

/-- Given two vectors a and b in R², and a real number lambda satisfying 
    |a - lambda*b| = √5, prove that lambda = -1/5 or lambda = 1 -/
theorem vector_equation_solution (a b : Fin 2 → ℝ) 
    (h1 : a = ![2, 0])
    (h2 : b = ![1, 2])
    (lambda : ℝ) 
    (h3 : ‖a - lambda • b‖ = Real.sqrt 5) :
    lambda = -1/5 ∨ lambda = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l141_14169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karsyn_phone_percentage_l141_14146

/-- The percentage of the initial price that Karsyn paid for the phone -/
noncomputable def percentagePaid (initialPrice paidPrice : ℝ) : ℝ :=
  (paidPrice / initialPrice) * 100

theorem karsyn_phone_percentage : 
  let initialPrice : ℝ := 600
  let paidPrice : ℝ := 480
  percentagePaid initialPrice paidPrice = 80 := by
  -- Unfold the definition of percentagePaid
  unfold percentagePaid
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_karsyn_phone_percentage_l141_14146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l141_14166

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (1, -1)

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vector_diff (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem angle_between_vectors :
  let v1 := vector_sum (scalar_mult 2 a) b
  let v2 := vector_diff a b
  Real.arccos ((dot_product v1 v2) / (vector_magnitude v1 * vector_magnitude v2)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l141_14166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_fifth_eq_one_l141_14154

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- Definition of f(x) for x > 0 -/
noncomputable def f_pos (x : ℝ) : ℝ := 32^x + Real.log x / Real.log 5

/-- Main theorem -/
theorem f_neg_one_fifth_eq_one
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_pos : ∀ x > 0, f x = f_pos x) :
  f (-1/5) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_fifth_eq_one_l141_14154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l141_14119

theorem solve_exponential_equation :
  ∃ x : ℚ, (27 : ℝ)^(x : ℝ) * (27 : ℝ)^(x : ℝ) * (27 : ℝ)^(x : ℝ) * (27 : ℝ)^(x : ℝ) = (243 : ℝ)^4 ∧ x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l141_14119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l141_14198

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ Real.pi then Real.sin x
  else if -Real.pi < x ∧ x < 0 then Real.cos x
  else 0  -- This else case is added to make the function total

theorem periodic_function_value (f : ℝ → ℝ) :
  (∀ x, f (x + 2 * Real.pi) = f x) →  -- Smallest positive period is 2π
  (∀ x, 0 ≤ x → x ≤ Real.pi → f x = Real.sin x) →  -- Definition for 0 ≤ x ≤ π
  (∀ x, -Real.pi < x → x < 0 → f x = Real.cos x) →  -- Definition for -π < x < 0
  f (-13 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l141_14198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_counter_impossible_l141_14123

/-- Represents the state of a cell on the checkerboard -/
inductive CellState
  | Empty
  | Counter

/-- Represents the infinite checkerboard -/
def Checkerboard := ℤ → ℤ → CellState

/-- Represents a position on the checkerboard -/
structure Position where
  x : ℤ
  y : ℤ

/-- Defines a valid jump on the checkerboard -/
def is_valid_jump (board : Checkerboard) (start finish : Position) : Prop :=
  (board start.x start.y = CellState.Counter) ∧
  (board finish.x finish.y = CellState.Empty) ∧
  ((abs (start.x - finish.x) = 2 ∧ start.y = finish.y) ∨
   (abs (start.y - finish.y) = 2 ∧ start.x = finish.x))

/-- Defines the initial state of the board -/
def initial_board (k n : ℕ) : Checkerboard :=
  λ x y ↦ if 0 ≤ x ∧ x < 3 * k ∧ 0 ≤ y ∧ y < n then CellState.Counter else CellState.Empty

/-- Counts the number of counters on the board -/
noncomputable def count_counters (board : Checkerboard) : ℕ := sorry

/-- The main theorem to be proved -/
theorem one_counter_impossible (k n : ℕ) :
  ¬∃ (final_board : Checkerboard),
    (∃ (moves : List (Position × Position)),
      (∀ (move : Position × Position), move ∈ moves → is_valid_jump (initial_board k n) move.fst move.snd) ∧
      (count_counters final_board = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_counter_impossible_l141_14123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_product_minus_constant_l141_14115

theorem derivative_of_product_minus_constant (f g : ℝ → ℝ) (x : ℝ) :
  let h := λ x ↦ f x * g x - 1
  f 6 = 5 →
  g 6 = 4 →
  deriv f 6 = 3 →
  deriv g 6 = 1 →
  deriv h 6 = 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_product_minus_constant_l141_14115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_pi_over_8_l141_14196

noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 4) * Real.sin x

theorem f_symmetry_about_pi_over_8 :
  ∀ x : ℝ, f (Real.pi / 4 - x) = f (Real.pi / 4 + x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_about_pi_over_8_l141_14196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_increases_with_submerged_block_l141_14109

/-- Represents the pressure at the bottom of a container filled with liquid -/
def pressure (h : ℝ) (ρ : ℝ) (g : ℝ) : ℝ := ρ * g * h

theorem pressure_increases_with_submerged_block 
  (h₀ h₁ : ℝ) (ρ : ℝ) (g : ℝ) (P₀ : ℝ) 
  (h₀_positive : h₀ > 0)
  (h₁_greater : h₁ > h₀)
  (ρ_positive : ρ > 0)
  (g_positive : g > 0) :
  pressure h₁ ρ g + P₀ > pressure h₀ ρ g + P₀ := by
  -- Proof goes here
  sorry

#check pressure_increases_with_submerged_block

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_increases_with_submerged_block_l141_14109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l141_14120

theorem tan_alpha_value (α : ℝ) (h1 : Real.cos α = -(1/2)) (h2 : π/2 < α ∧ α < π) : 
  Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l141_14120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l141_14170

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem tangent_point_x_coordinate 
  (a : ℝ) 
  (h_even : ∀ x, f a x = f a (-x))
  (h_slope : ∃ x, f_derivative a x = 3/2) :
  ∃ x, f_derivative a x = 3/2 ∧ x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_x_coordinate_l141_14170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rotation_path_length_l141_14103

/-- The length of the path traveled by a vertex of a square when rotated around its opposite vertex -/
noncomputable def rotationPathLength (sideLength : ℝ) (rotationAngle : ℝ) : ℝ :=
  sideLength * (rotationAngle * Real.pi / 180)

/-- Theorem: The length of the path traveled by vertex B of a square ABCD with side length 4 cm, 
    when rotated 180° around vertex A, is equal to 4π cm -/
theorem square_rotation_path_length :
  let sideLength : ℝ := 4
  let rotationAngle : ℝ := 180
  rotationPathLength sideLength rotationAngle = 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rotation_path_length_l141_14103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_propositions_l141_14122

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

-- Define geometric mean
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

-- Define a linear function
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m k : ℝ, ∀ x : ℝ, f x = m * x + k

theorem sequence_propositions :
  (¬ ∀ (a : ℕ → ℝ) (p q r : ℕ), 
    is_arithmetic_sequence a → p + q = r → a p + a q = a r) ∧
  (¬ ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) = 2 * a n) → is_geometric_sequence a 2) ∧
  (geometric_mean 2 8 = 4 ∨ geometric_mean 2 8 = -4) ∧
  (¬ ∀ (a : ℕ → ℝ) (f : ℝ → ℝ), 
    is_arithmetic_sequence a → (∀ n : ℕ, a n = f n) → is_linear_function f) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_propositions_l141_14122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_equalization_l141_14118

noncomputable def equalize_operation (a b : ℝ) : ℝ × ℝ :=
  let avg := (a + b) / 2
  (avg, avg)

theorem water_equalization (glasses : Fin 8 → ℝ) :
  ∃ (result : Fin 8 → ℝ), (∀ i j : Fin 8, result i = result j) ∧
  (∃ (n : ℕ) (seq : Fin n → (Fin 8 × Fin 8)),
    (∀ k : Fin n, let (i, j) := seq k
      ∀ l : Fin 8, l ≠ i ∧ l ≠ j →
        result l = glasses l) ∧
    (∀ k : Fin n, let (i, j) := seq k
      let (new_i, new_j) := equalize_operation (glasses i) (glasses j)
      result i = new_i ∧ result j = new_j)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_equalization_l141_14118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l141_14199

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  2 * x^2 - y^2 - 8 * x + 4 * y - 4 = 0

/-- The focus coordinates -/
noncomputable def focus : ℝ × ℝ := (2 + 2 * Real.sqrt 3, 2)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola :
  let (x, y) := focus
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x' y' : ℝ), hyperbola_eq x' y' ↔
      ((x' - 2)^2 / a^2) - ((y' - 2)^2 / b^2) = 1) ∧
    x = 2 + Real.sqrt (a^2 + b^2) ∧
    y = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_focus_of_hyperbola_l141_14199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_l141_14173

theorem cube_root_of_product : (2^9 * 5^3 * 7^3 : ℝ)^(1/3) = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_product_l141_14173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_gross_profit_l141_14112

/-- Calculates the gross profit for a furniture sale given the purchase price and markup percentage. -/
theorem furniture_gross_profit (purchase_price markup_percent : ℚ) : 
  purchase_price = 150 →
  markup_percent = 1/2 →
  let selling_price := purchase_price / (1 - markup_percent)
  selling_price - purchase_price = 150 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_gross_profit_l141_14112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l141_14197

/-- The problem statement --/
theorem tangent_line_problem (m n c : ℝ) : 
  (∀ x, x^3 + m*x + c = 2*x + 1 → x = 1) → -- Tangent line condition
  1^3 + m*1 + c = n → -- Point (1,n) lies on the curve
  m + n + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_problem_l141_14197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l141_14106

/-- The distance between two planes in R³ --/
noncomputable def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  |a₂ * 0 + b₂ * 0 + c₂ * (d₁ / c₁) + d₂| / Real.sqrt (a₂^2 + b₂^2 + c₂^2)

/-- Theorem: The distance between the planes 2x - 3y + 6z - 4 = 0 and 4x - 6y + 12z + 9 = 0 is 17/14 --/
theorem distance_between_specific_planes :
  distance_between_planes 2 (-3) 6 (-4) 4 (-6) 12 9 = 17/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l141_14106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_and_function_properties_l141_14159

-- Define the sets B and C
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def C : Set ℝ := {y | ∃ x ∈ B, y = x^2 + x - 1}

-- Define the function f and its domain A
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ Real.sqrt (4 * x - a)
def A (a : ℝ) : Set ℝ := {x | 4 * x - a ≥ 0}

-- State the theorem
theorem sets_and_function_properties :
  ∀ a : ℝ,
  (B ⊆ (Set.univ \ A a)) →
  ((B ∩ C) = {x | -5/4 ≤ x ∧ x < 2} ∧
   (B ∪ C) = {x | -3 < x ∧ x < 5} ∧
   a ∈ {x | 8 ≤ x}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_and_function_properties_l141_14159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l141_14110

-- Define a real-valued function g
variable (g : ℝ → ℝ)

-- Define the property of g being invertible
def IsInvertible (g : ℝ → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, (∀ x, h (g x) = x) ∧ (∀ y, g (h y) = y)

-- Define the intersection points of g(x^3) and g(x^6)
def IntersectionPoints (g : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | g (x^3) = g (x^6)}

-- Theorem statement
theorem intersection_count (hg : IsInvertible g) :
  ∃ s : Finset ℝ, s.card = 2 ∧ ∀ x : ℝ, x ∈ IntersectionPoints g ↔ x ∈ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l141_14110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_present_age_l141_14189

-- Define the present age of the son
def son_age : ℕ → ℕ := sorry

-- Define the present age of the father
def father_age : ℕ → ℕ := sorry

-- Condition 1: The present age of the father is 4 years more than 4 times the age of his son
axiom condition1 : ∀ n : ℕ, father_age n = 4 * son_age n + 4

-- Condition 2: 4 years hence, the father's age will be 20 years more than twice the age of the son
axiom condition2 : ∀ n : ℕ, father_age n + 4 = 2 * (son_age n + 4) + 20

-- Theorem: The present age of the father is 44 years
theorem father_present_age : ∀ n : ℕ, father_age n = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_present_age_l141_14189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l141_14150

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a * x^2) / Real.exp x

theorem max_value_implies_a (a : ℝ) :
  (∃ x : ℝ, f a x = 5 ∧ ∀ y : ℝ, f a y ≤ f a x) →
  a = -Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l141_14150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_zero_l141_14101

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

def f_iter (a b M : ℕ) : ℕ → (ℤ → ℤ)
  | 0 => id
  | n + 1 => λ x => f a b M (f_iter a b M n x)

theorem smallest_k_for_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  (∀ k : ℕ, k > 0 → f_iter a b M k 0 = 0 ↔ k ≥ a + b) ∧
  f_iter a b M (a + b) 0 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_zero_l141_14101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_2020_integers_with_property_l141_14105

theorem exist_2020_integers_with_property : 
  ∃ (S : Finset ℕ), S.card = 2020 ∧ 
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → a > 0 ∧ b > 0 ∧ |Int.ofNat a - Int.ofNat b| = Nat.gcd a b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_2020_integers_with_property_l141_14105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_problem_l141_14143

/-- Represents the car rental company's revenue function -/
noncomputable def revenue (x : ℝ) : ℝ := -1/5 * x^2 + 162*x - 2100

/-- The number of cars owned by the company -/
def total_cars : ℕ := 100

/-- The base rent at which all cars are rented -/
def base_rent : ℝ := 300

/-- The increment in rent that causes one car to remain unrented -/
def rent_increment : ℝ := 5

/-- Calculates the number of rented cars given a rent amount -/
noncomputable def rented_cars (rent : ℝ) : ℝ := total_cars - (rent - base_rent) / rent_increment

theorem car_rental_problem :
  (rented_cars 360 = 88) ∧
  (∀ x : ℝ, revenue x ≤ revenue 405) ∧
  (revenue 405 = 30705) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_problem_l141_14143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l141_14108

-- Define the function f
noncomputable def f (m n : ℝ) (x : ℝ) : ℝ := (-2^x + n) / (2^(x+1) + m)

-- State the theorem
theorem odd_function_properties (m n : ℝ) :
  (∀ x, f m n x = -f m n (-x)) →
  (m = 2 ∧ n = 1) ∧
  (∀ t ∈ Set.Ioo 1 2, (∀ k, f m n (t^2 + 2*t) + f m n (2*t^2 - k) < 0 → k ≤ 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l141_14108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_max_l141_14175

/-- Represents a triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Area of a triangle -/
noncomputable def area_triangle (t : Triangle) : ℝ := (1/2) * t.base * t.height

/-- Area of a rectangle -/
noncomputable def area_rectangle (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the configuration of rectangles R and S in triangle T -/
structure Configuration where
  T : Triangle
  R : Rectangle
  S : Rectangle
  h_acute : T.height < T.base -- T is acute
  h_R_inscribed : R.width ≤ T.base ∧ R.height ≤ T.height
  h_S_inscribed : S.width ≤ R.width ∧ S.height ≤ T.height - R.height

/-- The ratio of areas (A(R) + A(S)) / A(T) -/
noncomputable def area_ratio (c : Configuration) : ℝ :=
  (area_rectangle c.R + area_rectangle c.S) / area_triangle c.T

/-- Theorem stating that the maximum value of the area ratio is 2/3 -/
theorem area_ratio_max :
  ∃ (max : ℝ), max = 2/3 ∧ ∀ (c : Configuration), area_ratio c ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_max_l141_14175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_has_min_diagonal_l141_14113

/-- A parallelogram with side lengths a and b, angle α between sides, and area S -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  α : ℝ
  S : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < α ∧ 0 < S
  h_angle : 0 < α ∧ α < π
  h_area : S = a * b * Real.sin α

/-- The length of the longest diagonal of a parallelogram -/
noncomputable def longest_diagonal (p : Parallelogram) : ℝ :=
  Real.sqrt (p.a ^ 2 + p.b ^ 2 + 2 * p.a * p.b * Real.cos p.α)

/-- Theorem: Among all parallelograms with a fixed area, the square has the minimum longest diagonal -/
theorem square_has_min_diagonal {S : ℝ} (h_S : 0 < S) :
    ∀ (p : Parallelogram), p.S = S →
      longest_diagonal p ≥ Real.sqrt (2 * S) ∧
      (longest_diagonal p = Real.sqrt (2 * S) ↔ p.a = p.b ∧ p.α = π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_has_min_diagonal_l141_14113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l141_14125

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  sides : Fin 6 → ℕ
  distinct_lengths : ∃ (a b : ℕ), a ≠ b ∧ (∀ i, sides i = a ∨ sides i = b)
  convex : True  -- We don't model convexity explicitly here

/-- The perimeter of a hexagon -/
def perimeter (h : ConvexHexagon) : ℕ := (Finset.sum Finset.univ h.sides)

theorem hexagon_side_count (h : ConvexHexagon) 
  (side_7 : ∃ i, h.sides i = 7)
  (side_9 : ∃ i, h.sides i = 9)
  (perim_50 : perimeter h = 50) :
  (Finset.card (Finset.filter (λ i => h.sides i = 9) Finset.univ)) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l141_14125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EPHQ_is_42_l141_14187

/-- Rectangle EFGH with dimensions 12 cm by 6 cm -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Point P on FG such that FP is one-third of FG -/
noncomputable def P (rect : Rectangle) : ℝ := rect.width / 3

/-- Point Q as the midpoint of GH -/
noncomputable def Q (rect : Rectangle) : ℝ := rect.height / 2

/-- Area of region EPHQ -/
noncomputable def area_EPHQ (rect : Rectangle) : ℝ :=
  rect.width * rect.height - (rect.height * (rect.width / 3) / 2) - (rect.width * (rect.height / 2) / 2)

/-- Theorem stating that the area of region EPHQ is 42 square centimeters -/
theorem area_EPHQ_is_42 (rect : Rectangle) (h1 : rect.width = 12) (h2 : rect.height = 6) :
  area_EPHQ rect = 42 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EPHQ_is_42_l141_14187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_symmetric_intersection_is_spiral_homothety_center_l141_14145

/-- Representation of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Representation of a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point
  isParallelogram : Prop  -- This is a placeholder for the parallelogram property
  notRhombus : Prop       -- This is a placeholder for the not-rhombus property

/-- The center of a parallelogram -/
noncomputable def centerOfParallelogram (p : Parallelogram) : Point :=
  { x := (p.A.x + p.C.x) / 2, y := (p.A.y + p.C.y) / 2 }

/-- Point of intersection of symmetric lines -/
noncomputable def symmetricIntersection (p : Parallelogram) : Point :=
  { x := 0, y := 0 }  -- This is a placeholder for the actual calculation

/-- Definition of spiral homothety -/
def isSpiralHomothetyCenter (Q A O D : Point) : Prop :=
  True  -- This is a placeholder for the actual condition

/-- Main theorem -/
theorem parallelogram_symmetric_intersection_is_spiral_homothety_center 
  (p : Parallelogram) : 
  let O := centerOfParallelogram p
  let Q := symmetricIntersection p
  isSpiralHomothetyCenter Q p.A O p.D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_symmetric_intersection_is_spiral_homothety_center_l141_14145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_areas_l141_14180

/-- A regular hexagon inscribed in a circle -/
structure RegularHexagonInCircle where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- The radius is equal to the side length -/
  radius_eq_side : radius = side_length

/-- Calculate the area of a sector in a circle -/
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

/-- Calculate the area of an equilateral triangle -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

theorem hexagon_areas (h : RegularHexagonInCircle) (h_side : h.side_length = 5) :
  let sector_area := sector_area h.radius (π/3)
  let triangle_area := equilateral_triangle_area h.side_length
  sector_area = 25*π/6 ∧ triangle_area = 25*Real.sqrt 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_areas_l141_14180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l141_14114

/-- The circle equation -/
def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + x - 3*y + 5/4*(m^2 + m) = 0

/-- The line equation -/
def line_eq (m : ℝ) (x y : ℝ) : Prop :=
  (m + 3/2)*x - y + m - 1/2 = 0

/-- Condition 1: The origin is outside the circle -/
def origin_outside_circle (m : ℝ) : Prop :=
  ∀ x y, circle_eq m x y → x^2 + y^2 > 0

/-- Condition 2: The line does not pass through the second quadrant -/
def line_not_in_second_quadrant (m : ℝ) : Prop :=
  ∀ x y, line_eq m x y → ¬(x < 0 ∧ y > 0)

/-- The main theorem -/
theorem range_of_m :
  ∀ m : ℝ, (origin_outside_circle m ∧ line_not_in_second_quadrant m) ↔
    (-3/2 ≤ m ∧ m < -1) ∨ (0 < m ∧ m ≤ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l141_14114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cosine_sum_zero_implies_special_shape_l141_14190

/-- A quadrilateral is represented by its four angles -/
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real

/-- The property that the sum of cosines of angles is zero -/
def sumCosinesZero (q : Quadrilateral) : Prop :=
  Real.cos q.α + Real.cos q.β + Real.cos q.γ + Real.cos q.δ = 0

/-- The property that the sum of angles is 2π -/
def sumAngles2Pi (q : Quadrilateral) : Prop :=
  q.α + q.β + q.γ + q.δ = 2 * Real.pi

/-- The quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  (q.α + q.γ = Real.pi) ∧ (q.β + q.δ = Real.pi)

/-- The quadrilateral is cyclic -/
def isCyclic (q : Quadrilateral) : Prop :=
  (q.α + q.γ = Real.pi) ∨ (q.β + q.δ = Real.pi)

/-- The quadrilateral is a trapezoid -/
def isTrapezoid (q : Quadrilateral) : Prop :=
  (q.α + q.β = Real.pi) ∨ (q.β + q.γ = Real.pi) ∨ (q.γ + q.δ = Real.pi) ∨ (q.δ + q.α = Real.pi)

/-- Main theorem -/
theorem quadrilateral_cosine_sum_zero_implies_special_shape (q : Quadrilateral) 
  (h1 : sumCosinesZero q) (h2 : sumAngles2Pi q) :
  isParallelogram q ∨ isCyclic q ∨ isTrapezoid q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cosine_sum_zero_implies_special_shape_l141_14190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l141_14184

theorem coefficient_implies_a_value (a : ℝ) : 
  (∃ f : ℝ → ℝ, f = (λ x ↦ (1 + x) * (2*x^2 + a*x + 1)) ∧ 
   (∃ b c d : ℝ, ∀ x, f x = b*x^3 + (-4)*x^2 + c*x + d)) → 
  a = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_implies_a_value_l141_14184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_P_value_l141_14156

def is_valid_arrangement (p₁ p₂ p₃ q₁ q₂ q₃ r₁ r₂ r₃ : ℕ) : Prop :=
  Finset.toSet {p₁, p₂, p₃, q₁, q₂, q₃, r₁, r₂, r₃} = Finset.toSet {2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
  p₁ < p₂ ∧ p₂ < p₃ ∧
  q₁ < q₂ ∧ q₂ < q₃ ∧
  r₁ < r₂ ∧ r₂ < r₃

def P (p₁ p₂ p₃ q₁ q₂ q₃ r₁ r₂ r₃ : ℕ) : ℕ :=
  p₁ * p₂ * p₃ + q₁ * q₂ * q₃ + r₁ * r₂ * r₃

theorem smallest_P_value :
  ∀ p₁ p₂ p₃ q₁ q₂ q₃ r₁ r₂ r₃ : ℕ,
  is_valid_arrangement p₁ p₂ p₃ q₁ q₂ q₃ r₁ r₂ r₃ →
  P p₁ p₂ p₃ q₁ q₂ q₃ r₁ r₂ r₃ ≥ 954 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_P_value_l141_14156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_condition_l141_14163

theorem prime_power_condition (p : ℕ) (k : ℕ) : 
  Prime p → ((p - 1)^p + 1 = p^k ↔ p = 2 ∨ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_condition_l141_14163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_points_theorem_l141_14194

noncomputable def point_A : ℝ × ℝ := (Real.pi / 6, Real.sqrt 3 / 2)
noncomputable def point_B : ℝ × ℝ := (Real.pi / 4, 1)
noncomputable def point_C : ℝ × ℝ := (Real.pi / 2, 0)

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

def possible_ω_set : Set ℝ :=
  {ω | ω > 0 ∧ (∃ k : ℕ, ω = 8 * k + 2) ∧
    ((∃ k : ℕ, ω = 12 * k + 2) ∨ (∃ k : ℕ, ω = 12 * k + 4))} ∪ {2, 4}

theorem sine_points_theorem :
  ∀ ω : ℝ, ω > 0 →
    (f ω (point_A.1) = point_A.2 ∧
     f ω (point_B.1) = point_B.2 ∧
     f ω (point_C.1) = point_C.2) ↔
    ω ∈ possible_ω_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_points_theorem_l141_14194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l141_14133

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := Real.exp x

-- Define g and h as functions satisfying the given conditions
noncomputable def g (x : ℝ) : ℝ := (F x + F (-x)) / 2
noncomputable def h (x : ℝ) : ℝ := (F x - F (-x)) / 2

-- State the theorem
theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 2, g (2 * x) - a * h x ≥ 0) →
  a ≤ 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l141_14133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l141_14164

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x - 5)

-- Theorem statement
theorem vertical_asymptote_at_five :
  ∃ (L : ℝ), ∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x - 5| ∧ |x - 5| < δ → |f x| > L :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_five_l141_14164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_theorem_l141_14121

noncomputable section

open Real

/-- Given an ellipse with equation b^2 x^2 + a^2 y^2 = a^2 b^2 -/
def ellipse_equation (a b x y : ℝ) : Prop :=
  b^2 * x^2 + a^2 * y^2 = a^2 * b^2

/-- Volume of the ellipsoid formed by rotating the ellipse around its minor axis -/
noncomputable def ellipsoid_volume (a b : ℝ) : ℝ :=
  (4/3) * Real.pi * a^2 * b

/-- Volume of the largest inscribed cylinder in the ellipsoid -/
noncomputable def max_cylinder_volume (a b : ℝ) : ℝ :=
  (4/3) * Real.pi * a^2 * b * Real.sqrt (1/3)

/-- Sum of volumes of infinite sequence of ellipsoids -/
noncomputable def sum_ellipsoid_volumes (a b : ℝ) : ℝ :=
  (36/23) * a^2 * b * Real.pi * (1 + 2 * Real.sqrt 3 / 9)

/-- Sum of volumes of infinite sequence of cylinders -/
noncomputable def sum_cylinder_volumes (a b : ℝ) : ℝ :=
  sum_ellipsoid_volumes a b * Real.sqrt (1/3)

/-- The main theorem stating the ratio of sums of volumes -/
theorem volume_ratio_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  sum_cylinder_volumes a b / sum_ellipsoid_volumes a b = Real.sqrt 3 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_theorem_l141_14121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_l141_14111

/-- The line segment endpoints -/
noncomputable def p1 : ℝ × ℝ := (-4, 5)
noncomputable def p2 : ℝ × ℝ := (5, -1)

/-- The given point -/
noncomputable def p : ℝ × ℝ := (3, 4)

/-- Calculate trisection points -/
noncomputable def trisection_points : List (ℝ × ℝ) :=
  let x1 := p1.1
  let y1 := p1.2
  let x2 := p2.1
  let y2 := p2.2
  let dx := (x2 - x1) / 3
  let dy := (y2 - y1) / 3
  [(x1 + dx, y1 + dy), (x1 + 2*dx, y1 + 2*dy)]

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x - 4*y + 13 = 0

theorem line_passes_through_points :
  line_equation p.1 p.2 ∧ 
  ∃ t ∈ trisection_points, line_equation t.1 t.2 :=
by sorry

#check line_passes_through_points

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_l141_14111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_zero_conditions_l141_14142

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let f₁ := f 1
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b) = ((deriv f₁) 0) * x + f₁ 0) ∧
  (deriv f₁) 0 = 2 ∧ f₁ 0 = 0 :=
sorry

-- Theorem for the range of a
theorem zero_conditions (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a x = 0) ∧
  (∃! x : ℝ, x ∈ Set.Ioi 0 ∧ f a x = 0) ↔
  a < -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_zero_conditions_l141_14142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l141_14126

/-- The volume of a pyramid with a rectangular base and four equal edges to the apex. -/
theorem pyramid_volume
  (base_length : ℝ)
  (base_width : ℝ)
  (edge_length : ℝ)
  (h_base : base_length = 5 ∧ base_width = 7)
  (h_edge : edge_length = 15) :
  (1/3) * (base_length * base_width) *
    Real.sqrt (edge_length^2 - ((Real.sqrt (base_length^2 + base_width^2))/2)^2) =
    (70 * Real.sqrt 47) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l141_14126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l141_14162

theorem triangle_angle_relation (A B C : ℝ) (a b c : ℝ) (p : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute-angled triangle
  A + B + C = π →  -- sum of angles in a triangle
  Real.sqrt 3 * Real.sin B - c * Real.sin B * (Real.sqrt 3 * Real.sin C - c * Real.sin C) = 4 * Real.cos B * Real.cos C →
  Real.sin B = p * Real.sin C →
  1/2 < p ∧ p < 2 := by
  sorry

#check triangle_angle_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l141_14162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_subset_C_iff_l141_14157

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 1/2 < Real.exp (x - 1) * Real.log 2 ∧ Real.exp (x - 1) * Real.log 2 < Real.log 8}
def C (m : ℝ) : Set ℝ := {x | (x+2)*(x-m) < 0}

-- Theorem for part (I)
theorem intersection_A_B : A ∩ B = Set.Ioo 1 2 := by sorry

-- Theorem for part (II)
theorem union_subset_C_iff (m : ℝ) : A ∪ B ⊆ C m ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_union_subset_C_iff_l141_14157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l141_14193

/-- Polar coordinate point -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The polar equation of a line passing through two points -/
def polarLineEquation (A B : PolarPoint) : ℝ → ℝ → Prop :=
  fun ρ θ => ρ * Real.sin (θ + Real.pi / 6) = 2

theorem line_through_points (A B : PolarPoint) 
  (hA : A = ⟨4, 2 * Real.pi / 3⟩) 
  (hB : B = ⟨2, Real.pi / 3⟩) : 
  polarLineEquation A B = fun ρ θ => ρ * Real.sin (θ + Real.pi / 6) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l141_14193
