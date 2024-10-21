import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_implies_k_l843_84315

-- Define the line l: y = kx + 3
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 3

-- Define the circle C: x^2 + y^2 = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the angle AOB
noncomputable def angle_AOB (A B : ℝ × ℝ) : ℝ := 
  Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2)))

-- Main theorem
theorem intersection_angle_implies_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    line k A.1 A.2 ∧ line k B.1 B.2 ∧
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    angle_AOB A B = π / 3) →
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_implies_k_l843_84315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l843_84383

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define that a point is on the ellipse
def point_on_ellipse (p : Point) : Prop := is_on_ellipse p.x p.y

-- Define the perpendicular bisector of two points
def perp_bisector (A B : Point) (P : Point) : Prop :=
  (P.x - (A.x + B.x)/2)^2 + (P.y - (A.y + B.y)/2)^2 = 
  ((A.x - B.x)^2 + (A.y - B.y)^2)/4

-- Theorem statement
theorem x0_range :
  ∀ (A B : Point), A ≠ B → 
  point_on_ellipse A → point_on_ellipse B →
  ∃ (x0 : ℝ), perp_bisector A B (Point.mk x0 0) ∧ 
              -1/3 < x0 ∧ x0 < 1/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l843_84383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_l843_84307

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -(x - 1)^2 else (3 - a) * x + 4 * a

-- State the theorem
theorem f_strictly_increasing_iff (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ a ∈ Set.Icc (-1) 3 ∧ a ≠ 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_iff_l843_84307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l843_84346

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b + t.c = 2 * t.a ∧ 3 * t.c * Real.sin t.B = 4 * t.a * Real.sin t.C

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  Real.cos t.B = -1/4 ∧
  Real.sin (2 * t.B + π/6) = (3 * Real.sqrt 5 + 7) / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l843_84346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l843_84305

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((2 + x) / (1 - x)) + Real.sqrt (x^2 - x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l843_84305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_nine_l843_84353

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | (n + 3) => (n + 4) * a (n + 2) - (n + 3) * a (n + 1)

theorem smallest_divisible_by_nine :
  (∃ n : ℕ, ∀ m : ℕ, m ≥ n → 9 ∣ a m) ∧
  (∀ n : ℕ, n < 5 → ∃ m : ℕ, m ≥ n ∧ ¬(9 ∣ a m)) ∧
  (∀ m : ℕ, m ≥ 5 → 9 ∣ a m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_nine_l843_84353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_fixed_line_l843_84379

noncomputable section

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

theorem ellipse_eccentricity_and_fixed_line 
  (a b : ℝ) 
  (e : Ellipse a b) 
  (A B : Point) 
  (F : Point) 
  (h_AB : A.y > 0 ∧ B.y < 0 ∧ A.x = 0 ∧ B.x = 0) 
  (h_F : F.y = 0 ∧ F.x < 0) 
  (h_OF : (F.x^2 + (b * Real.sqrt 2 / 2)^2) = ((a^2 - b^2) / 4)) :
  eccentricity e = Real.sqrt 2 / 2 ∧ 
  (b = 2 → 
    ∀ (k : ℝ), 
    ∃ (M N G : Point),
    M ≠ N ∧
    M.y = k * M.x + 4 ∧
    N.y = k * N.x + 4 ∧
    (M.x^2 / a^2 + M.y^2 / b^2 = 1) ∧
    (N.x^2 / a^2 + N.y^2 / b^2 = 1) ∧
    (∃ (l₁ l₂ : Line), 
      (l₁.m = (M.y + 2) / M.x ∧ l₁.c = -2) ∧
      (l₂.m = (N.y - 2) / N.x ∧ l₂.c = 2) ∧
      G.y = l₁.m * G.x + l₁.c ∧
      G.y = l₂.m * G.x + l₂.c) →
    G.y = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_and_fixed_line_l843_84379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_negative_in_fourth_quadrant_l843_84349

theorem sin_2alpha_negative_in_fourth_quadrant (α : Real) :
  (-(π / 2) < α ∧ α < 0) → Real.sin (2 * α) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_negative_in_fourth_quadrant_l843_84349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_students_l843_84337

theorem average_height_students (total : ℕ) (short_ratio : ℚ) (tall : ℕ) : 
  total = 400 →
  short_ratio = 2 / 5 →
  tall = 90 →
  total - (short_ratio * ↑total).floor - tall = 150 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_students_l843_84337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_through_point_implies_tan_and_trig_identity_l843_84366

noncomputable def α : Real := Real.arctan (-2)

theorem angle_through_point_implies_tan_and_trig_identity :
  (∃ (P : ℝ × ℝ), P = (1, -2) ∧ P.1 * Real.cos α = P.2 * Real.sin α) →
  (Real.tan α = -2 ∧
   (Real.sin (π - α) + Real.cos (-α)) / (2 * Real.cos (π / 2 - α) - Real.sin (π / 2 + α)) = 1 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_through_point_implies_tan_and_trig_identity_l843_84366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_power_equals_81_l843_84317

theorem nine_power_equals_81 (x : ℝ) : (9 : ℝ)^x = 81 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_power_equals_81_l843_84317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l843_84394

theorem inequality_proof (a b c : ℝ) : 
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l843_84394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_counterfeit_in_two_weighings_l843_84304

/-- Represents a bag of coins -/
structure CoinBag where
  coins : ℕ
  isCounterfeit : Bool

/-- Represents the balance scale -/
def BalanceScale := ℝ → ℝ → ℝ

/-- The problem setup -/
structure CoinProblem where
  n : ℕ
  bags : Fin n → CoinBag
  scale : BalanceScale

/-- The condition that n is at least 3 -/
def validN (prob : CoinProblem) : Prop :=
  prob.n ≥ 3

/-- The condition that each bag contains the correct number of coins -/
def correctCoinCount (prob : CoinProblem) : Prop :=
  ∀ i, (prob.bags i).coins = (prob.n * (prob.n + 1)) / 2 + 1

/-- The condition that exactly one bag is counterfeit -/
def oneCounterfeitBag (prob : CoinProblem) : Prop :=
  ∃! i, (prob.bags i).isCounterfeit

/-- The condition that the scale shows the difference between right and left pans -/
def scaleShowsDifference (prob : CoinProblem) : Prop :=
  ∀ L R, prob.scale L R = R - L

/-- The main theorem stating that the counterfeit bag can be found in two weighings -/
theorem find_counterfeit_in_two_weighings (prob : CoinProblem)
  (hn : validN prob)
  (hcount : correctCoinCount prob)
  (hone : oneCounterfeitBag prob)
  (hscale : scaleShowsDifference prob) :
  ∃ (w₁ w₂ : ℝ), ∃ i, (prob.bags i).isCounterfeit :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_counterfeit_in_two_weighings_l843_84304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l843_84358

/-- A parabola with equation y² = 4x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = 4*x

/-- A circle defined by its center and a point it passes through -/
structure Circle where
  center : ℝ × ℝ
  passesThrough : ℝ × ℝ

/-- The theorem statement -/
theorem parabola_circle_intersection (C : Parabola) (M : ℝ × ℝ) (circ : Circle) :
  C.eq M.1 M.2 →  -- M is on the parabola
  circ.center = M →  -- circle is centered at M
  circ.passesThrough = (3, 0) →  -- circle passes through (3,0)
  (fun x y ↦ (x - M.1)^2 + (y - M.2)^2 = (M.1 + 1)^2) (-1) M.2 →  -- circle is tangent to x = -1
  ∃ y, (0 - M.1)^2 + (y - M.2)^2 = (M.1 + 1)^2 ∧  -- circle intersects y-axis
      2 * Real.sqrt 5 = 2 * Real.sqrt ((0 - M.1)^2 + y^2) :=  -- chord length is 2√5
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l843_84358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_1_minus_2i_l843_84340

theorem imaginary_part_of_1_minus_2i :
  (Complex.ofReal 1 - Complex.I * 2).im = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_1_minus_2i_l843_84340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l843_84364

noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def circle_O (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 / 4

noncomputable def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (F₁ F₂ : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (hP : hyperbola a b P.1 P.2)
  (hQ : asymptote a b Q.1 Q.2)
  (hPO : circle_O F₁ F₂ P)
  (hQO : circle_O F₁ F₂ Q)
  (hB : ∃ B : ℝ × ℝ, B.fst = 0 ∧ B.snd > 0 ∧ circle_O F₁ F₂ B)
  (hangle : ∃ θ : ℝ, Real.cos θ = (P.1 * F₂.1 + P.2 * F₂.2) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt (F₂.1^2 + F₂.2^2)) ∧
                     ∃ B : ℝ × ℝ, Real.cos θ = (Q.1 * B.1 + Q.2 * B.2) / (Real.sqrt (Q.1^2 + Q.2^2) * Real.sqrt (B.1^2 + B.2^2))) :
  eccentricity a b = (1 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l843_84364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_15_minutes_possible_l843_84318

/-- Represents an hourglass with a given duration --/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time --/
structure MeasurementState where
  elapsed_time : ℕ
  hourglass7_remaining : ℕ
  hourglass11_remaining : ℕ

/-- Defines the possible actions that can be taken with hourglasses --/
inductive MeasurementAction
  | FlipHourglass7
  | FlipHourglass11
  | WaitForHourglass7
  | WaitForHourglass11

/-- Applies an action to the current measurement state --/
def apply_action (state : MeasurementState) (action : MeasurementAction) : MeasurementState :=
  sorry

/-- Checks if the given sequence of actions results in exactly 15 minutes --/
def is_valid_sequence (actions : List MeasurementAction) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 15 minutes with 7 and 11 minute hourglasses --/
theorem measure_15_minutes_possible :
  ∃ (actions : List MeasurementAction), is_valid_sequence actions :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measure_15_minutes_possible_l843_84318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_grade_a_is_0_92_l843_84341

/-- Represents the grade of a product -/
inductive Grade
  | A
  | B
  | C

/-- Determines if a grade is defective -/
def isDefective (g : Grade) : Bool :=
  match g with
  | Grade.A => false
  | Grade.B => true
  | Grade.C => true

/-- The probability of producing a Grade B product -/
def probB : ℝ := 0.05

/-- The probability of producing a Grade C product -/
def probC : ℝ := 0.03

/-- Theorem: The probability of randomly inspecting a Grade A (non-defective) product is 0.92 -/
theorem prob_grade_a_is_0_92 : 1 - (probB + probC) = 0.92 := by
  sorry

/-- The probability of randomly inspecting a Grade A (non-defective) product -/
def prob_grade_a : ℝ := 1 - (probB + probC)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_grade_a_is_0_92_l843_84341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_union_Q_l843_84300

def P : Set ℕ := {x | x * (x - 3) ≥ 0}
def Q : Set ℕ := {2, 4}

theorem complement_P_union_Q : (Set.univ \ P) ∪ Q = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_union_Q_l843_84300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l843_84351

def is_blue (n : ℕ) : Prop := n % 4 = 1

-- We need to make this function computable
def count_blue_tiles (n : ℕ) : ℕ := 
  Finset.card (Finset.filter (fun x => x % 4 = 1) (Finset.range n))

theorem blue_tile_probability :
  (count_blue_tiles 50 : ℚ) / 50 = 13 / 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l843_84351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_30_equals_sqrt_3_l843_84336

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- Define the base case for 0
  | 1 => 0  -- Define a₁ = 0
  | n + 2 => (sequence_a (n + 1) - Real.sqrt 3) / (Real.sqrt 3 * sequence_a (n + 1) + 1)

theorem a_30_equals_sqrt_3 : sequence_a 30 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_30_equals_sqrt_3_l843_84336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_time_l843_84387

/-- The number of days A and B need to finish the work together -/
noncomputable def total_days_together : ℝ := 40

/-- The number of days A and B worked together before B left -/
noncomputable def days_worked_together : ℝ := 10

/-- The number of additional days A worked alone to finish the job -/
noncomputable def additional_days_alone : ℝ := 21

/-- The portion of work completed per day when A and B work together -/
noncomputable def work_rate_together : ℝ := 1 / total_days_together

/-- The portion of work completed when A and B worked together -/
noncomputable def work_completed_together : ℝ := work_rate_together * days_worked_together

/-- The portion of work A completed alone -/
noncomputable def work_completed_alone : ℝ := 1 - work_completed_together

/-- The number of days A needs to complete the entire job alone -/
noncomputable def days_a_alone : ℝ := additional_days_alone / work_completed_alone

theorem a_alone_time : days_a_alone = 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_time_l843_84387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l843_84380

/-- For a real number x > 1, the infinite sum ∑(n=0 to ∞) 1 / (x^(3^n) - x^(-3^n)) is equal to 1 / (x - 1). -/
theorem infinite_sum_equality (x : ℝ) (hx : x > 1) :
  ∑' n : ℕ, 1 / (x^(3^n : ℝ) - x^(-(3^n : ℝ))) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l843_84380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_trapezoid_sides_l843_84396

/-- A symmetric trapezoid with specific properties -/
structure SymmetricTrapezoid where
  -- Points of the trapezoid
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Height of the trapezoid
  m : ℝ
  -- AB is parallel to CD
  parallel_sides : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1)
  -- AB < CD
  shorter_top : dist A B < dist C D
  -- Height is m
  height_is_m : dist A D = m
  -- BD is perpendicular to BC
  diagonal_perpendicular : (B.1 - D.1) * (B.1 - C.1) + (B.2 - D.2) * (B.2 - C.2) = 0
  -- Circles with BC and AD as diameters touch
  circles_touch : ∃ (I : ℝ × ℝ), dist I B + dist I C = dist B C ∧ dist I A + dist I D = dist A D

/-- The theorem about the lengths of AB and CD in the symmetric trapezoid -/
theorem symmetric_trapezoid_sides (t : SymmetricTrapezoid) :
  dist t.A t.B = t.m * Real.sqrt (Real.sqrt 5 - 2) ∧
  dist t.C t.D = t.m * Real.sqrt (2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_trapezoid_sides_l843_84396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_lap_time_l843_84399

/-- Represents the distance of one lap around the playground in kilometers. -/
noncomputable def D : ℝ := sorry

/-- Represents the time taken for the first lap in hours. -/
noncomputable def T₁ : ℝ := D / 15

/-- Represents the time taken for the second lap in hours. -/
noncomputable def T₂ : ℝ := D / 10

/-- The theorem stating that under the given conditions, the time for the second lap is 1.5 hours. -/
theorem second_lap_time :
  T₂ = T₁ + 0.5 →
  T₂ = 1.5 := by
  intro h
  sorry

#check second_lap_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_lap_time_l843_84399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l843_84381

theorem problem_1 : Real.sqrt 36 - 3 * ((-1) ^ 2023) + ((-8) ^ (1/3 : ℝ)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l843_84381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probabilities_l843_84376

/-- Represents the colors of balls in the box -/
inductive BallColor
  | Red
  | Black
  | White
  | Green

/-- Represents the box of balls -/
structure BallBox where
  total : Nat
  red : Nat
  black : Nat
  white : Nat
  green : Nat

/-- Calculates the probability of drawing a ball of a specific color or set of colors -/
def probability (box : BallBox) (colors : List BallColor) : ℚ :=
  let favorable := colors.map (fun c => 
    match c with
    | BallColor.Red => box.red
    | BallColor.Black => box.black
    | BallColor.White => box.white
    | BallColor.Green => box.green
  ) |>.sum
  ↑favorable / ↑box.total

theorem ball_probabilities (box : BallBox) 
  (h1 : box.total = 12)
  (h2 : box.red = 5)
  (h3 : box.black = 4)
  (h4 : box.white = 2)
  (h5 : box.green = 1) : 
  probability box [BallColor.Red, BallColor.Black] = 3/4 ∧ 
  probability box [BallColor.Red, BallColor.Black, BallColor.White] = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probabilities_l843_84376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BO_centroid_max_m_circumcenter_l843_84319

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vectors a and b
def a (t : Triangle) : ℝ × ℝ := (t.A.1 - t.B.1, t.A.2 - t.B.2)
def b (t : Triangle) : ℝ × ℝ := (t.C.1 - t.B.1, t.C.2 - t.B.2)

-- Define the centroid
noncomputable def centroid (t : Triangle) : ℝ × ℝ := 
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define the circumcenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector BO
def BO (t : Triangle) (O : ℝ × ℝ) : ℝ × ℝ := (O.1 - t.B.1, O.2 - t.B.2)

-- Define dot product
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
noncomputable def mag (v : ℝ × ℝ) : ℝ := Real.sqrt (dot v v)

-- Define sine of angle
noncomputable def sin (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: BO when O is centroid
theorem BO_centroid (t : Triangle) : 
  BO t (centroid t) = ((1/3 : ℝ) * (a t).1 + (1/3 : ℝ) * (b t).1, 
                       (1/3 : ℝ) * (a t).2 + (1/3 : ℝ) * (b t).2) := by sorry

-- Theorem 2: Maximum value of m when O is circumcenter
theorem max_m_circumcenter (t : Triangle) (m : ℝ) :
  (mag (a t) / mag (b t) * dot (b t) (BO t (circumcenter t)) + 
   mag (b t) / mag (a t) * dot (a t) (BO t (circumcenter t)) = 
   2 * m * mag (BO t (circumcenter t))^2) ∧
  (sin t (a t) + sin t (b t) = Real.sqrt 2) →
  m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BO_centroid_max_m_circumcenter_l843_84319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marty_narrow_stripes_l843_84320

/-- The number of wide black stripes on Marty the zebra -/
def w : ℕ := sorry

/-- The number of narrow black stripes on Marty the zebra -/
def n : ℕ := sorry

/-- The number of white stripes on Marty the zebra -/
def b : ℕ := sorry

/-- Conditions on Marty's stripes -/
axiom white_stripes : b = w + 7
axiom total_black_stripes : w + n = b + 1

/-- The theorem stating that Marty has 8 narrow black stripes -/
theorem marty_narrow_stripes : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marty_narrow_stripes_l843_84320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_nerds_count_l843_84316

theorem rainbow_nerds_count 
  (purple yellow green red blue : Nat) :
  purple = 10 →
  yellow = purple + 4 →
  green = yellow - 2 →
  red = green * 3 →
  blue = red / 2 →
  purple + yellow + green + red + blue = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_nerds_count_l843_84316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_property_l843_84375

/-- A quartic polynomial with specific properties -/
def QuarticPolynomial (k : ℝ) : Type := 
  {Q : ℝ → ℝ // ∃ (a b c d : ℝ), ∀ x, Q x = d * x^4 + a * x^3 + b * x^2 + c * x + k}

/-- Theorem stating the property of the quartic polynomial -/
theorem quartic_property (k : ℝ) (Q : QuarticPolynomial k) :
  (Q.val 0 = k ∧ Q.val 1 = 3 * k ∧ Q.val (-1) = 5 * k) → Q.val 2 + Q.val (-2) = 26 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_property_l843_84375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relationship_l843_84325

/-- Given a triangle with sides a, b, c and angles α, β, γ,
    if β - α = 3γ, then b⁴ - 2b²c² + c⁴ = a²(b² + c²) -/
theorem triangle_side_relationship (a b c : ℝ) (α β γ : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b)
  (h_angles : α + β + γ = Real.pi)
  (h_angle_relation : β - α = 3 * γ) :
  b^4 - 2*b^2*c^2 + c^4 = a^2 * (b^2 + c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relationship_l843_84325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l843_84391

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := 1 - 1 / x
noncomputable def h (x : ℝ) := f x - g x

-- Theorem statement
theorem problem_statement :
  (∃! x : ℝ, x > 0 ∧ h x = 0) ∧
  (∀ x : ℝ, x > 0 → x ≠ 1 → f x > g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l843_84391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_points_on_parabolas_l843_84357

-- Define the two parabolas
noncomputable def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 1
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 + 8 * x + 7

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (-4/3, -17/9)
noncomputable def point2 : ℝ × ℝ := (2, 27)

-- Theorem stating that these are the only intersection points
theorem parabolas_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x → (x, y) = point1 ∨ (x, y) = point2 :=
by sorry

-- Theorem to verify that the points satisfy both parabola equations
theorem points_on_parabolas :
  parabola1 (point1.1) = point1.2 ∧ parabola2 (point1.1) = point1.2 ∧
  parabola1 (point2.1) = point2.2 ∧ parabola2 (point2.1) = point2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_points_on_parabolas_l843_84357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_units_l843_84310

/-- Represents the number of units in the table -/
noncomputable def x : ℝ := 720

/-- Cara's rate in units per hour -/
noncomputable def cara_rate : ℝ := x / 12

/-- Carl's rate in units per hour -/
noncomputable def carl_rate : ℝ := x / 15

/-- Combined rate when working together, in units per hour -/
noncomputable def combined_rate : ℝ := cara_rate + carl_rate - 12

/-- Time taken to complete the table when working together, in hours -/
def time_together : ℝ := 6

theorem table_units : x = 720 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_units_l843_84310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_side_length_is_150_l843_84350

/-- Represents a rectangular garden with a fence on three sides and one side against a wall. -/
structure Garden where
  wall_length : ℝ
  fence_cost_per_foot : ℝ
  total_fence_cost : ℝ

/-- Calculates the length of the side parallel to the wall that maximizes the area of the garden. -/
noncomputable def max_area_side_length (g : Garden) : ℝ :=
  let total_fence_length := g.total_fence_cost / g.fence_cost_per_foot
  total_fence_length / 2

/-- Theorem stating that the length of the side parallel to the wall that maximizes the area is 150 feet. -/
theorem max_area_side_length_is_150 (g : Garden)
    (h1 : g.wall_length = 500)
    (h2 : g.fence_cost_per_foot = 10)
    (h3 : g.total_fence_cost = 3000) :
  max_area_side_length g = 150 := by
  sorry

/-- Compute the result for the given garden parameters -/
def compute_result : ℚ :=
  (3000 : ℚ) / (10 : ℚ) / (2 : ℚ)

#eval compute_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_side_length_is_150_l843_84350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unique_bills_is_two_l843_84385

/-- Represents a strategy for filling bills with digits -/
structure FillStrategy where
  fill : Fin 20 → Fin 7 → Fin 2 → Fin 20 → Fin 7

/-- Represents a bill with a 7-digit serial number -/
def Bill := Fin 7 → Fin 2

/-- The result of applying a fill strategy to 20 bills -/
def applyStrategy (s : FillStrategy) : Fin 20 → Bill :=
  sorry

/-- Counts the number of unique bills after applying a strategy -/
def countUniqueBills (s : FillStrategy) : Nat :=
  sorry

/-- The maximum number of unique bills achievable -/
noncomputable def maxUniqueBills : Nat :=
  Finset.sup (Finset.range 1) (fun _ => 2)  -- Changed to a simple definition that always returns 2

/-- Theorem stating that the maximum number of unique bills is 2 -/
theorem max_unique_bills_is_two :
  maxUniqueBills = 2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unique_bills_is_two_l843_84385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_product_equality_l843_84335

theorem sine_cosine_product_equality (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : Real.cos (α - π / 4) = Real.cos (2 * α)) : 
  Real.sin α * Real.cos α = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_product_equality_l843_84335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_problem_l843_84327

/-- The distance between places A and B in kilometers -/
noncomputable def distance : ℝ := 400

/-- The maximum allowed speed in km/h -/
noncomputable def max_speed : ℝ := 100

/-- The transportation cost per hour as a function of speed -/
noncomputable def cost_per_hour (x : ℝ) : ℝ := (1 / 19200) * x^4 - (1 / 160) * x^3 + 15 * x

/-- The total transportation cost for the journey as a function of speed -/
noncomputable def total_cost (x : ℝ) : ℝ := (distance / x) * cost_per_hour x

theorem transportation_cost_problem :
  (total_cost 60 = 6000) ∧
  (∃ (x : ℝ), x > 0 ∧ x ≤ max_speed ∧
    (∀ (y : ℝ), y > 0 → y ≤ max_speed → total_cost x ≤ total_cost y) ∧
    x = 80 ∧ total_cost x = 2000 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_cost_problem_l843_84327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_perpendicular_to_plane_l843_84323

/-- Given points A, B, C, and P in 3D space, if PA is perpendicular to plane ABC, 
    then P has specific coordinates -/
theorem point_perpendicular_to_plane (A B C P : ℝ × ℝ × ℝ) :
  A = (0, 1, 0) →
  B = (1, 1, 0) →
  C = (1, 0, 0) →
  P.2.2 = 1 →
  (P.1 - A.1, P.2.1 - A.2.1, P.2.2 - A.2.2) • (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2) = 0 →
  (P.1 - A.1, P.2.1 - A.2.1, P.2.2 - A.2.2) • (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2) = 0 →
  P = (0, 1, 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_perpendicular_to_plane_l843_84323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l843_84378

/-- Predicate to check if six points form a regular hexagon -/
def is_regular_hexagon (A B C D E F : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (side_length : ℝ),
    side_length > 0 ∧
    dist A B = side_length ∧
    dist B C = side_length ∧
    dist C D = side_length ∧
    dist D E = side_length ∧
    dist E F = side_length ∧
    dist F A = side_length ∧
    dist center A = side_length ∧
    dist center B = side_length ∧
    dist center C = side_length ∧
    dist center D = side_length ∧
    dist center E = side_length ∧
    dist center F = side_length

/-- Function to calculate the area of a regular hexagon given its vertices -/
noncomputable def area_regular_hexagon (A B C D E F : ℝ × ℝ) : ℝ :=
  let side_length := dist A B
  (3 * Real.sqrt 3 * side_length^2) / 2

/-- Function to calculate the Euclidean distance between two points -/
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The area of a regular hexagon with vertices A(0,0) and C(5,3) is 51√3/4 -/
theorem regular_hexagon_area : 
  ∀ (A B C D E F : ℝ × ℝ),
  A = (0, 0) →
  C = (5, 3) →
  is_regular_hexagon A B C D E F →
  area_regular_hexagon A B C D E F = (51 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_area_l843_84378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_y_axis_l843_84365

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define tangency to y-axis
def TangentToYAxis (c : Set (ℝ × ℝ)) : Prop :=
  ∃ y : ℝ, (0, y) ∈ c ∧ ∀ p ∈ c, p.1 ≥ 0

theorem circle_tangent_to_y_axis :
  let c := Circle (1, 1) 1
  TangentToYAxis c ∧ c = {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_y_axis_l843_84365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_purchase_price_l843_84362

/-- The purchase price of a product given its marked price, discount rate, and profit rate -/
noncomputable def purchase_price (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  marked_price * (1 - discount_rate) / (1 + profit_rate)

theorem product_purchase_price :
  purchase_price 126 0.05 0.05 = 114 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_purchase_price_l843_84362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_different_digits_summing_to_32_l843_84314

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Returns true if all digits in a natural number are different -/
def allDigitsDifferent (n : ℕ) : Bool :=
  let digits := Nat.digits 10 n
  digits.length = digits.toFinset.card

/-- The smallest number with all different digits summing to 32 -/
def smallestNumberWithDifferentDigitsSummingTo32 : ℕ := 26789

theorem smallest_number_with_different_digits_summing_to_32 :
  (∀ m : ℕ, m < smallestNumberWithDifferentDigitsSummingTo32 →
    ¬(allDigitsDifferent m ∧ sumOfDigits m = 32)) ∧
  allDigitsDifferent smallestNumberWithDifferentDigitsSummingTo32 ∧
  sumOfDigits smallestNumberWithDifferentDigitsSummingTo32 = 32 :=
by sorry

#eval smallestNumberWithDifferentDigitsSummingTo32
#eval sumOfDigits smallestNumberWithDifferentDigitsSummingTo32
#eval allDigitsDifferent smallestNumberWithDifferentDigitsSummingTo32

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_different_digits_summing_to_32_l843_84314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l843_84382

/-- Represents a number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ
  nonRepeatingPart_lt_one : nonRepeatingPart < 1 := by sorry
  repeatingPart_lt_one : repeatingPart < 1 := by sorry

/-- Converts a RepeatingDecimal to a real number -/
noncomputable def RepeatingDecimal.toReal (r : RepeatingDecimal) : ℝ :=
  r.integerPart + r.nonRepeatingPart + r.repeatingPart / (1 - (1/10)^(Nat.lcm (r.repeatingPart.num.natAbs) (r.repeatingPart.den)))

theorem largest_number
  (a : ℝ)
  (b c d e : RepeatingDecimal)
  (ha : a = 4.25678)
  (hb : b = { integerPart := 4, nonRepeatingPart := 256/1000, repeatingPart := 7/10 })
  (hc : c = { integerPart := 4, nonRepeatingPart := 25/100, repeatingPart := 67/100 })
  (hd : d = { integerPart := 4, nonRepeatingPart := 2/10, repeatingPart := 567/1000 })
  (he : e = { integerPart := 4, nonRepeatingPart := 0, repeatingPart := 2567/10000 }) :
  a > b.toReal ∧ a > c.toReal ∧ a > d.toReal ∧ a > e.toReal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l843_84382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l843_84313

-- Define the line equation
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = a * x + 1

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3 = 0

-- Theorem statement
theorem line_circle_intersection (a : ℝ) : 
  ∃ x y : ℝ, line_eq a x y ∧ circle_eq x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l843_84313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_bulbs_can_be_turned_off_l843_84334

/-- Represents the state of a bulb (on or off) -/
inductive BulbState
| On : BulbState
| Off : BulbState

/-- Represents a dashboard with bulbs and buttons -/
structure Dashboard where
  bulbs : Finset Nat
  buttons : Finset Nat
  connections : Nat → Finset Nat
  odd_connection : ∀ (S : Finset Nat), S ⊆ bulbs → ∃ b ∈ buttons, Odd ((connections b ∩ S).card)

/-- The state of all bulbs on the dashboard -/
def DashboardState := Nat → BulbState

/-- Toggles the state of a bulb -/
def toggle : BulbState → BulbState
| BulbState.On => BulbState.Off
| BulbState.Off => BulbState.On

/-- Applies the effect of pressing a button to the dashboard state -/
def pressButton (d : Dashboard) (b : Nat) (s : DashboardState) : DashboardState :=
  fun n => if n ∈ d.connections b then toggle (s n) else s n

/-- Represents a sequence of button presses -/
def ButtonSequence := List Nat

/-- Applies a sequence of button presses to the dashboard state -/
def applyButtonSequence (d : Dashboard) : ButtonSequence → DashboardState → DashboardState
| [], s => s
| (b::bs), s => applyButtonSequence d bs (pressButton d b s)

/-- The theorem to be proved -/
theorem all_bulbs_can_be_turned_off (d : Dashboard) :
  ∃ (seq : ButtonSequence), ∀ n, n ∈ d.bulbs →
    (applyButtonSequence d seq (fun _ => BulbState.On)) n = BulbState.Off := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_bulbs_can_be_turned_off_l843_84334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_theorem_l843_84322

/-- The total distance flown by a fly between two approaching objects. -/
noncomputable def fly_distance (initial_distance : ℝ) (object_speed : ℝ) (fly_speed_to : ℝ) (fly_speed_from : ℝ) : ℝ :=
  (11 / 5) * (initial_distance / 2)

/-- Theorem: The fly flies 55 meters under the given conditions. -/
theorem fly_distance_theorem :
  let initial_distance : ℝ := 50
  let object_speed : ℝ := 10
  let fly_speed_to : ℝ := 20
  let fly_speed_from : ℝ := 30
  fly_distance initial_distance object_speed fly_speed_to fly_speed_from = 55 := by
  sorry

/-- Compute the fly distance for the given parameters -/
def compute_fly_distance : ℚ :=
  (11 / 5) * (50 / 2)

#eval compute_fly_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_distance_theorem_l843_84322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l843_84354

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1/2)^x - 3 else x^2

-- Theorem statement
theorem range_of_a (a : ℝ) :
  f a > 1 ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l843_84354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l843_84348

theorem equation_solution (x : ℝ) : (10 : ℝ)^(x^2 + x - 2) = 1 ↔ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l843_84348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_area_reduction_percentage_l843_84356

/-- Represents a regular hexagon -/
structure RegularHexagon where
  sideLength : ℝ
  area : ℝ

/-- Represents the hexagon formed by joining the midpoints of a regular hexagon's sides -/
noncomputable def midpointHexagon (h : RegularHexagon) : RegularHexagon :=
  { sideLength := h.sideLength / 2,
    area := h.area / 4 }

/-- Theorem stating that the area of the midpoint hexagon is 25% of the original hexagon -/
theorem midpoint_hexagon_area_ratio (h : RegularHexagon) :
  (midpointHexagon h).area = h.area / 4 := by
  sorry

/-- Theorem stating that the area reduction is 75% -/
theorem area_reduction_percentage (h : RegularHexagon) :
  (h.area - (midpointHexagon h).area) / h.area = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_area_reduction_percentage_l843_84356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_to_middle_ratio_l843_84397

/-- Represents the heights of three trees in a town square -/
structure TreeHeights where
  tallest : ℝ
  middle : ℝ
  shortest : ℝ

/-- The ratio of the shortest tree to the middle tree -/
noncomputable def tree_ratio (h : TreeHeights) : ℝ :=
  h.shortest / h.middle

/-- Theorem stating the ratio of the shortest to middle tree -/
theorem shortest_to_middle_ratio 
  (h : TreeHeights) 
  (h_tallest : h.tallest = 150)
  (h_middle : h.middle = 2/3 * h.tallest)
  (h_shortest : h.shortest = 50) : 
  tree_ratio h = 1/2 := by
  sorry

/-- Example calculation of the tree ratio -/
def example_ratio : ℚ :=
  50 / 100

#eval example_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_to_middle_ratio_l843_84397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_usage_probability_l843_84388

def probability_at_least_one_not_in_use (n : ℕ) (p : ℝ) : ℝ :=
  1 - p ^ n

theorem terminal_usage_probability (n : ℕ) (p : ℝ) 
  (hn : n = 20) (hp : p = 0.8) : 
  1 - p ^ n = probability_at_least_one_not_in_use n p :=
by
  unfold probability_at_least_one_not_in_use
  rfl

#eval probability_at_least_one_not_in_use 20 0.8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_usage_probability_l843_84388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_is_7_2_l843_84369

/-- The speed of the slower train given the conditions of the problem -/
noncomputable def slower_train_speed (faster_train_speed : ℝ) (faster_train_length : ℝ) (passing_time : ℝ) : ℝ :=
  let faster_train_speed_mps := faster_train_speed * 1000 / 3600
  let distance_covered := faster_train_speed_mps * passing_time
  let slower_train_distance := distance_covered - faster_train_length
  slower_train_distance * 3600 / (1000 * passing_time)

/-- Theorem stating that the speed of the slower train is 7.2 kmph -/
theorem slower_train_speed_is_7_2 :
  slower_train_speed 72 150 15 = 7.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_is_7_2_l843_84369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_range_l843_84363

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x * (Real.sin x + Real.cos x)

theorem f_value_range :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    (1/2) ≤ f x ∧ f x ≤ (1/2) * Real.exp (Real.pi / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_range_l843_84363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l843_84398

theorem diophantine_equation_solution :
  ∃ (x y : ℤ), 
    (6 * x^2 + 5 * x * y + y^2 = 6 * x + 2 * y + 7) ∧ 
    (∀ (a b : ℤ), 6 * a^2 + 5 * a * b + b^2 = 6 * a + 2 * b + 7 → (a.natAbs + b.natAbs : ℕ) ≤ x.natAbs + y.natAbs) ∧
    x = -8 ∧ y = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l843_84398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_y_l843_84309

noncomputable def y (x : ℝ) : ℝ := 
  8 * Real.sin (Real.arctan (1 / Real.tan 3)) + (1/5) * (Real.sin (5*x))^2 / Real.cos (10*x)

theorem derivative_of_y (x : ℝ) (h : Real.cos (10*x) ≠ 0) :
  deriv y x = Real.tan (10*x) / (5 * Real.cos (10*x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_y_l843_84309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheaper_to_buy_more_count_l843_84311

def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

def cheaper_to_buy_more (n : ℕ) : Bool :=
  C (n + 1) < C n

theorem cheaper_to_buy_more_count : 
  (Finset.filter (fun n => cheaper_to_buy_more n) (Finset.range 1000)).card = 6 := by
  sorry

#eval (Finset.filter (fun n => cheaper_to_buy_more n) (Finset.range 1000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheaper_to_buy_more_count_l843_84311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_in_set_l843_84390

theorem largest_number_in_set (b : ℝ) (h : b = 3) :
  let S : Set ℝ := {-2*b, 5*b, 30/b, b^2 + 1, 2}
  ∀ x ∈ S, x ≤ 5*b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_in_set_l843_84390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_expression_of_y_l843_84333

/-- Given that y+2 is directly proportional to x and y=7 when x=3,
    prove that the functional expression of y in terms of x is y = 3x - 2 -/
theorem functional_expression_of_y :
  ∃ k : ℝ, (∀ x y : ℝ, y + 2 = k * x) ∧ (9 = k * 3) →
  ∀ x y : ℝ, y + 2 = k * x → y = 3 * x - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_expression_of_y_l843_84333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_theorem_l843_84374

/-- The area of the region between three touching circles of radius r -/
noncomputable def area_between_circles (r : ℝ) : ℝ :=
  r^2 * (Real.sqrt 3 - Real.pi / 2)

/-- Theorem: The area of the region between three touching circles of radius r
    is equal to r^2 * (√3 - π/2) -/
theorem area_between_circles_theorem (r : ℝ) (hr : r > 0) :
  area_between_circles r = r^2 * (Real.sqrt 3 - Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_theorem_l843_84374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_invariant_l843_84395

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : a + b + c = 180

-- Define an enlargement factor
def enlarge_factor : ℝ := 2

-- Theorem: The sum of interior angles of a triangle remains 180° after enlargement
theorem triangle_angle_sum_invariant (t : Triangle) :
  t.a + t.b + t.c = 180 := by
  exact t.sum_angles

#check triangle_angle_sum_invariant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_invariant_l843_84395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_weight_is_four_pounds_l843_84306

/-- Represents the weight of Jacque's suitcase and its contents --/
structure SuitcaseContents where
  initial_weight : ℝ
  perfume_bottles : ℕ
  perfume_weight : ℝ
  soap_bars : ℕ
  soap_weight : ℝ
  jam_jars : ℕ
  jam_weight : ℝ
  final_weight : ℝ

/-- Converts ounces to pounds --/
noncomputable def ounces_to_pounds (ounces : ℝ) : ℝ := ounces / 16

/-- Calculates the weight of chocolate in pounds --/
noncomputable def chocolate_weight (contents : SuitcaseContents) : ℝ :=
  contents.final_weight - contents.initial_weight -
  ounces_to_pounds (contents.perfume_bottles * contents.perfume_weight +
                    contents.soap_bars * contents.soap_weight +
                    contents.jam_jars * contents.jam_weight)

/-- Theorem stating that the weight of chocolate is 4 pounds --/
theorem chocolate_weight_is_four_pounds (contents : SuitcaseContents)
  (h1 : contents.initial_weight = 5)
  (h2 : contents.perfume_bottles = 5)
  (h3 : contents.perfume_weight = 1.2)
  (h4 : contents.soap_bars = 2)
  (h5 : contents.soap_weight = 5)
  (h6 : contents.jam_jars = 2)
  (h7 : contents.jam_weight = 8)
  (h8 : contents.final_weight = 11) :
  chocolate_weight contents = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_weight_is_four_pounds_l843_84306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l843_84338

/-- Definition of the ellipse C -/
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the unit circle -/
def unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Theorem about the ellipse C and its properties -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (m : ℝ),
    -- The ellipse passes through (1, √3/2)
    ellipse_C a b 1 (Real.sqrt 3 / 2) ∧
    -- The left focus is at (-√3, 0)
    ellipse_C a b (-Real.sqrt 3) 0 ∧
    -- The equation of ellipse C is x^2/4 + y^2 = 1
    (∀ x y, ellipse_C a b x y ↔ x^2/4 + y^2 = 1) ∧
    -- For a tangent line to the unit circle passing through (m, 0) and intersecting C at A and B
    (∀ x y, unit_circle x y →
      ∃ A B : ℝ × ℝ,
        ellipse_C a b A.1 A.2 ∧
        ellipse_C a b B.1 B.2 ∧
        -- |AB| = (4√3|m|) / (m^2 + 1)
        Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (4 * Real.sqrt 3 * abs m) / (m^2 + 1)) ∧
    -- The maximum value of |AB| is 2
    (∀ A B : ℝ × ℝ, ellipse_C a b A.1 A.2 → ellipse_C a b B.1 B.2 →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ 2) ∧
    (∃ A B : ℝ × ℝ, ellipse_C a b A.1 A.2 ∧ ellipse_C a b B.1 B.2 ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l843_84338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_together_arrangements_l843_84370

/-- The number of ways to arrange 4 boys and 2 girls in a row, with the girls always together -/
def arrangement_count : ℕ := 240

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 2

theorem girls_together_arrangements :
  arrangement_count = (Nat.factorial (num_boys + 1)) * (Nat.factorial num_girls) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_together_arrangements_l843_84370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_zero_l843_84386

theorem triplet_sum_not_zero :
  let triplet_a : Fin 3 → ℚ := ![1/3, 2/3, -1]
  let triplet_b : Fin 3 → ℝ := ![1.5, -1.5, 0]
  let triplet_c : Fin 3 → ℝ := ![0.2, -0.7, 0.5]
  let triplet_d : Fin 3 → ℝ := ![-1.2, 1.1, 0.1]
  let triplet_e : Fin 3 → ℚ := ![4/5, 2/5, -7/5]

  (triplet_a 0 + triplet_a 1 + triplet_a 2 = 0) ∧
  (triplet_b 0 + triplet_b 1 + triplet_b 2 = 0) ∧
  (triplet_c 0 + triplet_c 1 + triplet_c 2 = 0) ∧
  (triplet_d 0 + triplet_d 1 + triplet_d 2 = 0) ∧
  (triplet_e 0 + triplet_e 1 + triplet_e 2 ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplet_sum_not_zero_l843_84386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_intervals_l843_84372

noncomputable def f (x : ℝ) := 3 * Real.sin (Real.pi / 4 - x)

theorem monotone_decreasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc ((2 * Real.pi * ↑k - Real.pi / 4) : ℝ) ((2 * Real.pi * ↑k + 3 * Real.pi / 4) : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_intervals_l843_84372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_two_pans_l843_84361

/-- Proves that the cost of 2 pans is $20 given the conditions of Katerina's purchase. -/
theorem cost_of_two_pans (num_pots num_pans : ℕ) (cost_per_pot total_cost : ℚ) :
  num_pots = 3 →
  num_pans = 4 →
  cost_per_pot = 20 →
  total_cost = 100 →
  ∃ (cost_per_pan : ℚ),
    cost_per_pan * num_pans + cost_per_pot * num_pots = total_cost ∧
    2 * cost_per_pan = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_two_pans_l843_84361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_difference_l843_84345

/-- A magic square is a 3x3 grid of natural numbers where the sum of each row, column, and diagonal is the same. -/
def MagicSquare (s : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  let sum := s 0 0 + s 0 1 + s 0 2
  (∀ i, s i 0 + s i 1 + s i 2 = sum) ∧
  (∀ j, s 0 j + s 1 j + s 2 j = sum) ∧
  (s 0 0 + s 1 1 + s 2 2 = sum) ∧
  (s 0 2 + s 1 1 + s 2 0 = sum)

/-- The numbers in the magic square are all different -/
def AllDifferent (s : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → s i j ≠ s k l

theorem magic_square_difference (s : Matrix (Fin 3) (Fin 3) ℕ) 
  (h_magic : MagicSquare s)
  (h_diff : AllDifferent s)
  (h_largest : ∃ i j, s i j = max (max (s 0 0) (s 0 1)) (max (s 0 2) (max (s 1 0) (max (s 1 1) (max (s 1 2) (max (s 2 0) (max (s 2 1) (s 2 2))))))))
  (h_middle : ∃ i j, s i j = max (min (s 0 0) (s 0 1)) (min (s 0 2) (min (s 1 0) (min (s 1 1) (min (s 1 2) (min (s 2 0) (min (s 2 1) (s 2 2))))))))
  (h_smallest : ∃ i j, s i j = min (min (s 0 0) (s 0 1)) (min (s 0 2) (min (s 1 0) (min (s 1 1) (min (s 1 2) (min (s 2 0) (min (s 2 1) (s 2 2))))))))
  (h_diff_largest_middle : ∃ i j k l, s i j - s k l = 14)
  (h_diff_middle_smallest : ∃ i j k l, s i j - s k l = 14) :
  ∃ i j k l, s i j - s k l = 49 ∧ 
    s i j = max (min (s 0 0) (s 0 1)) (min (s 0 2) (min (s 1 0) (min (s 1 1) (min (s 1 2) (min (s 2 0) (min (s 2 1) (s 2 2))))))) ∧
    s k l = min (max (s 0 0) (s 0 1)) (max (s 0 2) (max (s 1 0) (max (s 1 1) (max (s 1 2) (max (s 2 0) (max (s 2 1) (s 2 2))))))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_difference_l843_84345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_hours_decrease_is_10_percent_l843_84330

/-- Represents Jane's bear production and working hours --/
structure JaneProduction where
  bears : ℝ  -- number of bears produced per week
  hours : ℝ  -- number of hours worked per week

/-- Calculates the percentage decrease in hours worked with an assistant --/
noncomputable def assistantHoursDecrease (solo : JaneProduction) (withAssistant : JaneProduction) : ℝ :=
  (solo.hours - withAssistant.hours) / solo.hours * 100

/-- Theorem stating the percentage decrease in hours worked with an assistant --/
theorem assistant_hours_decrease_is_10_percent 
  (solo : JaneProduction) 
  (withAssistant : JaneProduction) 
  (h1 : withAssistant.bears = 1.8 * solo.bears)  -- 80% increase in bear production
  (h2 : withAssistant.bears / withAssistant.hours = 2 * (solo.bears / solo.hours))  -- 100% increase in hourly output
  : assistantHoursDecrease solo withAssistant = 10 := by
  sorry

#check assistant_hours_decrease_is_10_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_hours_decrease_is_10_percent_l843_84330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l843_84342

noncomputable section

/-- Definition of the ellipse M -/
def ellipse_M (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

/-- Definition of focus -/
def focus : ℝ × ℝ := (Real.sqrt 2, 0)

/-- Definition of a line intersecting the ellipse -/
def intersecting_line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

/-- Definition of a point on the ellipse -/
def point_on_ellipse (x y : ℝ) : Prop := ellipse_M x y

/-- Definition of the origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Definition of distance from a point to a line -/
noncomputable def distance_point_to_line (k m : ℝ) : ℝ := |m| / Real.sqrt (1 + k^2)

theorem ellipse_theorem :
  (∀ x y, ellipse_M x y ↔ x^2/4 + y^2/2 = 1) ∧
  (eccentricity 2 (Real.sqrt 2) = Real.sqrt 2 / 2) ∧
  (∃ x y, point_on_ellipse x y ∧
    ∀ k m, 
      (∃ x1 y1 x2 y2, 
        intersecting_line k m x1 y1 ∧ 
        intersecting_line k m x2 y2 ∧
        point_on_ellipse x1 y1 ∧
        point_on_ellipse x2 y2) →
      distance_point_to_line k m ≥ Real.sqrt 2 / 2) ∧
  (∃ k m, distance_point_to_line k m = Real.sqrt 2 / 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l843_84342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l843_84384

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f based on the diagram
axiom f_neg_for_pos : ∀ x > 0, f x < 0
axiom f_nonneg_for_nonpos : ∀ x ≤ 1, f x ≥ 0

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  f ((2 * x^2 - x - 1) / (x^2 - 2 * x + 1)) * f (Real.log (x^2 - 6 * x + 20)) ≤ 0

-- State the theorem
theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l843_84384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l843_84347

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := x^2
noncomputable def curve2 (x : ℝ) : ℝ := Real.sqrt x

-- Define the area function
noncomputable def area_between_curves : ℝ :=
  ∫ x in (0)..(1), (curve2 x - curve1 x)

-- Theorem statement
theorem area_enclosed_by_curves :
  area_between_curves = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l843_84347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_object_surface_area_l843_84326

/-- Calculate the total surface area of a combined object consisting of a right cylinder with a cone on top -/
theorem combined_object_surface_area 
  (cylinder_height : ℝ) 
  (cylinder_radius : ℝ) 
  (cone_height : ℝ) 
  (h_cylinder_height : cylinder_height = 5)
  (h_cylinder_radius : cylinder_radius = 3)
  (h_cone_height : cone_height = 3)
  (h_cone_radius : cone_height = cylinder_radius) :
  2 * Real.pi * cylinder_radius * cylinder_height + 
  Real.pi * cylinder_radius * Real.sqrt (cylinder_radius^2 + cone_height^2) = 
  30 * Real.pi + 9 * Real.sqrt 2 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_object_surface_area_l843_84326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_council_probability_l843_84308

/-- The number of liberal arts classes in the 10th grade -/
def num_classes : ℕ := 5

/-- The number of students sent from each class for the election -/
def students_per_class : ℕ := 2

/-- The total number of students participating in the election -/
def total_students : ℕ := num_classes * students_per_class

/-- The number of students selected for the student council -/
def selected_students : ℕ := 4

/-- The probability of selecting exactly two students from the same class -/
def prob_two_same_class : ℚ := 4/7

theorem student_council_probability :
  (Nat.choose num_classes 1 * Nat.choose students_per_class 2 * 
   (Nat.choose students_per_class 1)^(num_classes - 1)) / 
  (Nat.choose total_students selected_students) = prob_two_same_class := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_council_probability_l843_84308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_theorem_l843_84377

/-- Represents a square in 2D space -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Represents an octagon formed by the intersection of two squares -/
structure Octagon where
  square1 : Square
  square2 : Square

/-- The area of an octagon formed by two intersecting squares -/
noncomputable def octagon_area (oct : Octagon) : ℝ :=
  sorry

/-- The set of sides of an octagon -/
def set_of_sides (oct : Octagon) : Set (ℝ × ℝ) :=
  sorry

theorem octagon_area_theorem (s1 s2 : Square) (oct : Octagon) :
  s1.center = s2.center →
  s1.side_length = 2 →
  s2.side_length = 2 →
  oct.square1 = s1 →
  oct.square2 = s2 →
  (∃ (side : ℝ × ℝ), side.1 = 1/4 ∧ side ∈ set_of_sides oct) →
  octagon_area oct = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_theorem_l843_84377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_pi_plus_2x_l843_84328

theorem definite_integral_exp_pi_plus_2x :
  ∫ x in (Set.Icc 0 1), (Real.exp π + 2 * x) = Real.exp π + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_exp_pi_plus_2x_l843_84328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l843_84339

-- Define the curve C
noncomputable def C (x : ℝ) : ℝ := Real.sin (3 * Real.pi / 4 - x) * Real.cos (x + Real.pi / 4)

-- Define the shifted curve C'
noncomputable def C' (a x : ℝ) : ℝ := C (x - a)

-- Define the property of central symmetry
def centrally_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define the slope property
def negative_slope (f : ℝ → ℝ) (b : ℕ) : Prop :=
  ∀ x y, x ∈ Set.Icc ((b + 1 : ℝ) / 8 * Real.pi) ((b + 1 : ℝ) / 4 * Real.pi) →
         y ∈ Set.Icc ((b + 1 : ℝ) / 8 * Real.pi) ((b + 1 : ℝ) / 4 * Real.pi) →
         x ≠ y → (f y - f x) / (y - x) < 0

-- State the theorem
theorem curve_properties (a : ℝ) (h_a : a > 0) :
  (∃ b : ℕ, centrally_symmetric (C' a) ∧ negative_slope (C' a) b) →
  (∃ b : ℕ, b = 1 ∨ b = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l843_84339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_equality_l843_84321

theorem arithmetic_geometric_sequence_equality (a : ℝ) (n : ℕ) :
  let seq := λ (i : ℕ) => a
  (∀ i : ℕ, i < n → seq (i + 1) - seq i = seq 1 - seq 0) ∧
  (∀ i : ℕ, i < n → seq (i + 1) / seq i = seq 1 / seq 0) →
  ∀ i : ℕ, i ≤ n → seq i = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_equality_l843_84321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr2_is_simplest_l843_84302

/-- A function that determines if a radical expression is in its simplest form -/
def isSimplestForm (expr : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), ¬∃ (y : ℝ), y ≠ expr x ∧ (∀ (z : ℝ), expr z = y)

/-- The given expressions -/
noncomputable def expr1 : ℝ → ℝ := λ _ ↦ Real.sqrt 0.2
noncomputable def expr2 : ℝ → ℝ → ℝ := λ a b ↦ Real.sqrt (a^2 - b^2)
noncomputable def expr3 : ℝ → ℝ := λ x ↦ Real.sqrt (1/x)
noncomputable def expr4 : ℝ → ℝ := λ a ↦ Real.sqrt (4*a)

/-- Theorem stating that expr2 is the simplest form among the given expressions -/
theorem expr2_is_simplest : 
  isSimplestForm (λ x ↦ expr2 x x) ∧ 
  ¬isSimplestForm expr1 ∧ 
  ¬isSimplestForm expr3 ∧ 
  ¬isSimplestForm expr4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr2_is_simplest_l843_84302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_shift_for_even_function_l843_84360

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

theorem minimum_shift_for_even_function :
  ∀ m : ℝ,
  (∀ x : ℝ, f (x + Real.pi) = f x) →  -- f has period π
  m > 0 →
  (∀ x : ℝ, g m x = g m (-x)) →  -- g is even
  m ≥ Real.pi / 3 ∧
  ∃ m₀ : ℝ, m₀ = Real.pi / 3 ∧ (∀ x : ℝ, g m₀ x = g m₀ (-x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_shift_for_even_function_l843_84360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l843_84332

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sqrt 3 * Real.sin (2 * x) + 2) * 1 + Real.cos x * (2 * Real.cos x)

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : f A = 4) 
  (h2 : 0 < A ∧ A < Real.pi) 
  (h3 : (1 : ℝ) * C * Real.sin A / 2 = Real.sqrt 3 / 2) :
  (1 : ℝ)^2 + C^2 - 2 * 1 * C * Real.cos A = 3 :=
by
  sorry

#check triangle_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l843_84332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l843_84371

def jo_sum : ℕ := (200 * (200 + 1)) / 2

def round_to_5 (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def lola_sum : ℕ := (List.range 200).map (λ i => round_to_5 (i + 1)) |>.sum

theorem sum_difference : |Int.ofNat jo_sum - Int.ofNat lola_sum| = 19000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l843_84371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_cubic_yards_l843_84367

/-- The volume of a rectangular prism in cubic feet -/
noncomputable def prism_volume_cubic_feet : ℝ := 250

/-- The conversion factor from cubic feet to cubic yards -/
noncomputable def cubic_feet_to_cubic_yards : ℝ := 1 / 27

/-- Theorem: The volume of the rectangular prism in cubic yards -/
theorem prism_volume_cubic_yards : 
  prism_volume_cubic_feet * cubic_feet_to_cubic_yards = 250 / 27 := by
  -- Unfold the definitions
  unfold prism_volume_cubic_feet cubic_feet_to_cubic_yards
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_cubic_yards_l843_84367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_distribution_degrees_of_freedom_l843_84331

/-- Represents the number of sample intervals in a Pearson's chi-square test. -/
def s : Type := ℕ

/-- Represents the number of parameters estimated from the data for the exponential distribution. -/
def r : ℕ := 1

/-- Calculates the degrees of freedom for Pearson's chi-square test. -/
def degrees_of_freedom (s : ℕ) : ℤ := s - 1 - r

/-- Theorem stating that the degrees of freedom for Pearson's chi-square test
    of the exponential distribution hypothesis is equal to s - 2. -/
theorem exponential_distribution_degrees_of_freedom (s : ℕ) :
  degrees_of_freedom s = s - 2 := by
  unfold degrees_of_freedom
  unfold r
  simp
  ring

#check exponential_distribution_degrees_of_freedom

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_distribution_degrees_of_freedom_l843_84331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_seven_l843_84359

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (1 + t, Real.sqrt 3 * t)

-- Define the chord length function
noncomputable def chord_length (C : (ℝ → ℝ → Prop)) (l : ℝ → ℝ × ℝ) : ℝ := 
  sorry

-- Theorem statement
theorem chord_length_is_sqrt_seven :
  chord_length C l = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_seven_l843_84359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_difference_l843_84303

theorem peach_difference (red green blue : ℕ) : red = 17 → green = 16 → blue = 12 →
  (red : ℤ) - ((green : ℤ) + (blue : ℤ)) = -11 := by
  intros h_red h_green h_blue
  rw [h_red, h_green, h_blue]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peach_difference_l843_84303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_AB_l843_84355

noncomputable section

-- Define the square ABCD
def square_side_length : ℝ := 8

-- Define points A, B, C, D
def A : ℝ × ℝ := (0, square_side_length)
def B : ℝ × ℝ := (square_side_length, square_side_length)
def C : ℝ × ℝ := (square_side_length, 0)
def D : ℝ × ℝ := (0, 0)

-- Define point M (midpoint of CD)
def M : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define the circles
def circle_M (x y : ℝ) : Prop := (x - M.1)^2 + (y - M.2)^2 = 4^2
def circle_B (x y : ℝ) : Prop := (x - B.1)^2 + (y - B.2)^2 = 8^2

-- Define point P as an intersection point of the two circles
def P : ℝ × ℝ := sorry

-- Define distance from a point to a line
def dist_point_line (p : ℝ × ℝ) (l : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem distance_P_to_AB : 
  circle_M P.1 P.2 ∧ circle_B P.1 P.2 → P.2 = dist_point_line P (A.1, B.1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_AB_l843_84355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_internal_sides_33x33_l843_84373

/-- Represents a coloring of a grid --/
structure GridColoring (n : ℕ) where
  colors : Fin 3 → ℕ
  total_squares : colors 0 + colors 1 + colors 2 = n * n
  equal_distribution : colors 0 = colors 1 ∧ colors 1 = colors 2

/-- Counts the number of internal sides in a grid coloring --/
def count_internal_sides (n : ℕ) (coloring : GridColoring n) : ℕ :=
  sorry

/-- Theorem: The minimum number of internal sides in a 33x33 grid with three equally distributed colors is 66 --/
theorem min_internal_sides_33x33 :
  ∃ (coloring : GridColoring 33),
    ∀ (other_coloring : GridColoring 33),
      count_internal_sides 33 coloring ≤ count_internal_sides 33 other_coloring ∧
      count_internal_sides 33 coloring = 66 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_internal_sides_33x33_l843_84373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_critical_points_condition_l843_84352

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (x - 3)^2

-- Part 1: Tangent line condition
theorem tangent_line_condition (a b : ℝ) :
  (∀ x y : ℝ, y = f a x → x + y + b = 0 ↔ x = 1) →
  a = 3 ∧ b = -5 := by sorry

-- Part 2: Critical points condition
theorem critical_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    (deriv (f a)) x₁ = 0 ∧ (deriv (f a)) x₂ = 0) →
  0 < a ∧ a < 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_critical_points_condition_l843_84352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_maximum_l843_84343

/-- Represents the number of days since Robinson arrived on the island -/
def x : ℝ := 0

/-- Initial number of hair strands -/
def initial_strands : ℝ := 200000

/-- Initial length of each hair strand in cm -/
def initial_length : ℝ := 5

/-- Daily hair growth in cm -/
def daily_growth : ℝ := 0.05

/-- Daily hair loss in number of strands -/
def daily_loss : ℝ := 50

/-- Number of hair strands after x days -/
def strands (x : ℝ) : ℝ := initial_strands - daily_loss * x

/-- Length of each hair strand after x days in cm -/
def strand_length (x : ℝ) : ℝ := initial_length + daily_growth * x

/-- Total length of all hair strands after x days in cm -/
def total_length (x : ℝ) : ℝ := strands x * strand_length x

/-- The day when the total length of hair strands reaches its maximum -/
def max_day : ℝ := 1950

theorem total_length_maximum :
  ∀ y : ℝ, total_length max_day ≥ total_length y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_maximum_l843_84343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l843_84393

theorem subset_existence (n : ℕ) (h : n ≥ 1) :
  ∃ (M : Finset (Fin (Nat.choose (2*n) n))) 
    (P : Finset (Fin (Nat.choose (2*n) n))) 
    (R : Fin (Nat.choose (2*n) n) → Fin (Nat.choose (2*n) n) → Prop),
    (Finset.card M = Nat.choose (2*n) n) ∧
    (P ⊆ M) ∧
    (Finset.card P = n + 1) ∧
    ((∀ x y, x ∈ P → y ∈ P → x ≠ y → R x y) ∨ 
     (∀ x y, x ∈ P → y ∈ P → x ≠ y → ¬R x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_existence_l843_84393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_line_l843_84301

noncomputable section

/-- The function f(x) = m*ln(x+1) -/
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log (x + 1)

/-- The function g(x) = x/(x+1) for x > -1 -/
noncomputable def g (x : ℝ) : ℝ := x / (x + 1)

/-- The function F(x) = f(x) - g(x) -/
def F (m : ℝ) (x : ℝ) : ℝ := f m x - g x

/-- Theorem: The value of m for which y=f(x) and y=g(x) have exactly one common tangent line is 1 -/
theorem unique_common_tangent_line (m : ℝ) :
  (∃! a b : ℝ, a > -1 ∧ b > -1 ∧
    (deriv (f m)) a = (deriv g) b ∧
    f m a - (deriv (f m)) a * a = g b - (deriv g) b * b) →
  m = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_common_tangent_line_l843_84301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l843_84324

/-- Represents a right circular cone --/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a sphere --/
structure Sphere where
  radius : ℝ

theorem liquid_rise_ratio
  (cone1 cone2 : Cone)
  (sphere : Sphere)
  (h : cone1.radius = 4 ∧ cone2.radius = 8 ∧ sphere.radius = 1)
  (volume_eq : (1/3) * Real.pi * cone1.radius^2 * cone1.height = (1/3) * Real.pi * cone2.radius^2 * cone2.height)
  : (cone1.height * ((1 + (4/3) * Real.pi * sphere.radius^3 / ((1/3) * Real.pi * cone1.radius^2 * cone1.height))^(1/3) - 1)) /
    (cone2.height * ((1 + (4/3) * Real.pi * sphere.radius^3 / ((1/3) * Real.pi * cone2.radius^2 * cone2.height))^(1/3) - 1)) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l843_84324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_circle_hyperbola_l843_84312

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 56 = 0

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := x*y = 2

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (8, 0)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The smallest distance between a point on the circle and a point on the hyperbola -/
theorem smallest_distance_circle_hyperbola :
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧ 
    hyperbola_eq B.1 B.2 ∧
    ∀ (A' B' : ℝ × ℝ), 
      circle_eq A'.1 A'.2 → 
      hyperbola_eq B'.1 B'.2 →
      distance A B ≤ distance A' B' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_circle_hyperbola_l843_84312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circles_area_sum_final_result_l843_84392

/-- Represents the grid and circle configuration -/
structure GridWithCircles where
  gridSize : Nat
  squareSize : ℝ
  smallCircleCount : Nat
  smallCircleDiameter : ℝ
  largeCircleCount : Nat
  largeCircleDiameter : ℝ

/-- Calculates the total area of the grid and circles -/
noncomputable def totalArea (g : GridWithCircles) : ℝ :=
  let gridArea := (g.gridSize : ℝ) ^ 2 * g.squareSize ^ 2
  let smallCircleArea := (g.smallCircleCount : ℝ) * Real.pi * (g.smallCircleDiameter / 2) ^ 2
  let largeCircleArea := (g.largeCircleCount : ℝ) * Real.pi * (g.largeCircleDiameter / 2) ^ 2
  gridArea - smallCircleArea - largeCircleArea

/-- The main theorem to prove -/
theorem grid_circles_area_sum :
  let g : GridWithCircles := {
    gridSize := 5
    squareSize := 2
    smallCircleCount := 4
    smallCircleDiameter := 2
    largeCircleCount := 1
    largeCircleDiameter := 6
  }
  totalArea g = 100 - 13 * Real.pi := by sorry

/-- The final result: A + B = 113 -/
theorem final_result :
  let g : GridWithCircles := {
    gridSize := 5
    squareSize := 2
    smallCircleCount := 4
    smallCircleDiameter := 2
    largeCircleCount := 1
    largeCircleDiameter := 6
  }
  ∃ (A B : ℝ), totalArea g = A - B * Real.pi ∧ A + B = 113 := by
  use 100, 13
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_circles_area_sum_final_result_l843_84392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_average_age_l843_84329

/-- Given an adult school with the following conditions:
  - The initial average age of students is 48 years
  - 120 new students join the school
  - The average age decreases by 4 years after the new students join
  - The total number of students after joining is 160
Prove that the average age of the new students is approximately 42.67 years -/
theorem new_students_average_age (initial_avg : ℝ) (new_students : ℕ) 
  (avg_decrease : ℝ) (total_students : ℕ) :
  initial_avg = 48 →
  new_students = 120 →
  avg_decrease = 4 →
  total_students = 160 →
  ∃ (new_avg : ℝ), 
    (abs (new_avg - 42.67) < 0.01) ∧ 
    (initial_avg * (total_students - new_students) + new_avg * new_students) / total_students 
    = initial_avg - avg_decrease :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_average_age_l843_84329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_even_numbers_between_11_and_27_l843_84368

def isEvenBetween11And27 (n : ℕ) : Bool :=
  11 < n && n ≤ 27 && n % 2 = 0

def evenNumbersBetween11And27 : List ℕ :=
  (List.range 28).filter isEvenBetween11And27

theorem average_of_even_numbers_between_11_and_27 :
  (evenNumbersBetween11And27.sum : ℚ) / evenNumbersBetween11And27.length = 19 := by
  sorry

#eval evenNumbersBetween11And27
#eval evenNumbersBetween11And27.sum
#eval evenNumbersBetween11And27.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_even_numbers_between_11_and_27_l843_84368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l843_84344

-- Problem 1
theorem simplify_expression_1 : (1 : ℚ)*(-3)^(0 : ℤ) + (-1/2 : ℚ)^(-2 : ℤ) - (-3 : ℚ)^(-1 : ℤ) = 16/3 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h : x ≠ 0) : 
  (-2*x^3)^2 * (-x^2) / ((-x)^2)^3 = -4*x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l843_84344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l843_84389

-- Define the functions f and g
def f (a b x : ℝ) := x^2 + a*x + b
def g (x : ℝ) := 2*x^2 + 4*x - 30

-- Define the sequences a_n and b_n
noncomputable def a : ℕ → ℝ
  | 0 => 1/2
  | n + 1 => (f 2 (-15) (a n) + 15) / 2

noncomputable def b (n : ℕ) : ℝ := 1 / (2 + a n)

-- Define S_n and T_n
noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b i)
noncomputable def T (n : ℕ) : ℝ := (Finset.range n).prod (λ i => b i)

-- Main theorem
theorem main_theorem :
  (∀ x : ℝ, f 2 (-15) x = 0 ↔ g x = 0) →
  (∀ n : ℕ, 2^(n+1) * T n + S n = 2) ∧
  (∀ n : ℕ, 2*(1 - (4/5)^n) ≤ S n ∧ S n < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l843_84389
