import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_decreasing_condition_l114_11472

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + m * x^2 - 3 * m^2 * x + 1

-- Statement 1: Tangent line equation when m = 1
theorem tangent_line_at_2 :
  let f₁ := f 1
  let tangent_line (x y : ℝ) := 15 * x - 3 * y - 25 = 0
  tangent_line 2 (f₁ 2) ∧ 
  ∀ x, tangent_line x (f₁ 2 + ((deriv f₁) 2) * (x - 2)) :=
by sorry

-- Statement 2: Condition for f(x) to be decreasing on (-2, 3)
theorem decreasing_condition (m : ℝ) :
  (∀ x ∈ Set.Ioo (-2) 3, (deriv (f m)) x < 0) ↔ (m ≥ 3 ∨ m ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_2_decreasing_condition_l114_11472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l114_11447

theorem closest_integer_to_cube_root :
  ∃ (n : ℤ), ∀ (m : ℤ), |((7^3 + 9^3 : ℝ) ^ (1/3 : ℝ)) - n| ≤ |((7^3 + 9^3 : ℝ) ^ (1/3 : ℝ)) - m| ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_l114_11447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_win_percentage_l114_11462

/-- Proves that the required percentage to win an election is 51% given the specified conditions -/
theorem election_win_percentage 
  (total_votes : ℕ) 
  (candidate_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : candidate_percentage = 1 / 100)
  (h3 : additional_votes_needed = 3000) :
  (((candidate_percentage * total_votes + additional_votes_needed : ℚ) / total_votes) * 100 : ℚ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_win_percentage_l114_11462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l114_11463

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x + a) * ln((2x - 1) / (2x + 1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log ((2 * x - 1) / (2 * x + 1))

/-- If f is an even function, then a = 0 -/
theorem f_even_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_zero_l114_11463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l114_11429

noncomputable section

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Circle O centered at origin with radius r -/
def circle_O (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

/-- Line l with slope 1 and y-intercept 2 -/
def line_l (x y : ℝ) : Prop :=
  y = x + 2

/-- Line m with slope k through point (x₀, y₀) -/
def line_m (k x₀ y₀ x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- Main theorem -/
theorem ellipse_and_tangent_line (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∃ (x y : ℝ), ellipse_C a b x y ∧ line_l x y) →
  (∃ (r : ℝ), ∀ (x y : ℝ), circle_O r x y ↔ circle_O b x y) →
  (a^2 - b^2) / a^2 = 3 / 4 →
  (∃ (x₀ y₀ : ℝ), ellipse_C a b x₀ y₀ ∧ x₀ = -a) →
  (∀ k : ℝ, k ≠ 0 →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      line_m k (-a) 0 x₁ y₁ ∧
      line_m k (-a) 0 x₂ y₂ ∧
      circle_O b x₁ y₁ ∧
      circle_O b x₂ y₂ ∧
      x₁ * x₂ + y₁ * y₂ < 0) →
    -Real.sqrt 2/2 < k ∧ k < Real.sqrt 2/2) →
  (ellipse_C 3 2 = ellipse_C a b) ∧
  (∀ k : ℝ, (-Real.sqrt 2/2 < k ∧ k < 0) ∨ (0 < k ∧ k < Real.sqrt 2/2) ↔
    k ≠ 0 ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      line_m k (-a) 0 x₁ y₁ ∧
      line_m k (-a) 0 x₂ y₂ ∧
      circle_O b x₁ y₁ ∧
      circle_O b x₂ y₂ ∧
      x₁ * x₂ + y₁ * y₂ < 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l114_11429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_fourth_l114_11437

theorem cos_three_pi_fourth : Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_fourth_l114_11437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_meeting_sum_l114_11488

/-- Two friends arrive independently at random times between 3 p.m. and 4 p.m. and stay for n minutes. -/
def arrival_time : Set ℝ := Set.Icc 0 60

/-- The probability that either friend arrives while the other is at the park is 25%. -/
def meeting_probability : ℚ := 1/4

/-- n is the number of minutes each friend stays at the park. -/
noncomputable def n : ℕ → ℕ → ℕ → ℝ
  | d, e, f => d - e * Real.sqrt (f : ℝ)

/-- f is not divisible by the square of any prime. -/
def square_free (f : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ f)

theorem friend_meeting_sum (d e f : ℕ) 
  (hd : d > 0) (he : e > 0) (hf : f > 0)
  (hsf : square_free f)
  (hprob : (1 - (60 - n d e f)^2 / 3600) = meeting_probability) :
  d + e + f = 93 := by
  sorry

#eval 60 + 30 + 3  -- This should output 93

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_meeting_sum_l114_11488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l114_11405

-- Define M as a function of a
noncomputable def M (a : ℝ) : ℝ := a + 1 / (a - 2)

-- Define N as a function of x
noncomputable def N (x : ℝ) : ℝ := Real.log (x^2 + 1/16) / Real.log (1/2)

-- Theorem statement
theorem M_greater_than_N (a x : ℝ) (h1 : 2 < a) (h2 : a < 3) :
  M a > N x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_greater_than_N_l114_11405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l114_11470

/-- A partnership business with two partners A and B -/
structure Partnership where
  investment_A : ℚ
  investment_B : ℚ
  period_A : ℚ
  period_B : ℚ
  profit_B : ℚ

/-- The total profit of the partnership -/
def total_profit (p : Partnership) : ℚ :=
  p.profit_B * (p.investment_A * p.period_A + p.investment_B * p.period_B) / (p.investment_B * p.period_B)

theorem partnership_profit (p : Partnership) 
  (h1 : p.investment_A = 3 * p.investment_B)
  (h2 : p.period_A = 2 * p.period_B)
  (h3 : p.profit_B = 7000) :
  total_profit p = 49000 := by
  sorry

#eval total_profit { investment_A := 3, investment_B := 1, period_A := 2, period_B := 1, profit_B := 7000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l114_11470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l114_11402

-- Define the sector
noncomputable def Sector (arcLength centralAngle : ℝ) : Prop :=
  arcLength > 0 ∧ centralAngle > 0

-- Define the area of a sector
noncomputable def sectorArea (arcLength centralAngle : ℝ) : ℝ :=
  (arcLength * arcLength) / (2 * centralAngle)

-- Theorem statement
theorem sector_area_calculation (arcLength centralAngle : ℝ) 
  (h : Sector arcLength centralAngle) 
  (h1 : arcLength = Real.pi/3) 
  (h2 : centralAngle = Real.pi/6) : 
  sectorArea arcLength centralAngle = Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l114_11402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arithmetic_sequence_length_in_P_l114_11440

/-- The set P of numbers in base 7 with coefficients between 1 and 6 -/
def P : Set ℕ :=
  {x | ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧
    x = 7^3 + a * 7^2 + b * 7 + c}

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def IsArithmeticSequence (s : List ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < s.length - 1 → s[i + 1]! - s[i]! = d

/-- The maximum length of an arithmetic sequence in P -/
theorem max_arithmetic_sequence_length_in_P :
  (∃ (s : List ℕ), (∀ x ∈ s, x ∈ P) ∧ IsArithmeticSequence s ∧ s.length = 6) ∧
  (∀ (s : List ℕ), (∀ x ∈ s, x ∈ P) → IsArithmeticSequence s → s.length ≤ 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_arithmetic_sequence_length_in_P_l114_11440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l114_11468

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_difference_magnitude 
  (a b : V) 
  (ha : ‖a‖ = 6) 
  (hb : ‖b‖ = 8) 
  (hab : ‖a + b‖ = ‖a - b‖) : 
  ‖a - b‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l114_11468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_chessboard_l114_11495

/-- A chessboard is represented as an 8x8 grid -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A position on the chessboard -/
def Position := Fin 8 × Fin 8

/-- Check if two positions are adjacent (including diagonally) -/
def are_adjacent (p1 p2 : Position) : Bool :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (x1 = x2 ∧ y1 = y2) ∨ (abs (x1 - x2) ≤ 1 ∧ abs (y1 - y2) ≤ 1)

/-- A valid placement of kings is one where no two kings attack each other -/
def is_valid_placement (kings : List Position) : Prop :=
  ∀ k1 k2, k1 ∈ kings → k2 ∈ kings → k1 ≠ k2 → ¬(are_adjacent k1 k2)

/-- The maximum number of kings that can be placed on the chessboard -/
def max_kings : ℕ := 16

/-- Theorem: The maximum number of kings that can be placed on an 8x8 chessboard
    so that no two of them attack each other is 16 -/
theorem max_kings_on_chessboard :
  ∃ (kings : List Position),
    kings.length = max_kings ∧
    is_valid_placement kings ∧
    ∀ (kings' : List Position),
      is_valid_placement kings' →
      kings'.length ≤ max_kings := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_kings_on_chessboard_l114_11495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l114_11411

noncomputable def f (x α : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x + α) - Real.sin (2 * x + α)

theorem symmetry_condition (α : ℝ) : 
  (∀ x, f x α = f (-x) α) ↔ ∃ k : ℤ, α = k * Real.pi - Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_condition_l114_11411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_mixture_theorem_l114_11451

/-- Represents a liquid with a specific kerosene concentration -/
structure Liquid where
  kerosene_percent : ℚ
  deriving Repr

/-- Represents a mixture of two liquids -/
structure Mixture where
  liquid1 : Liquid
  liquid2 : Liquid
  parts1 : ℚ
  parts2 : ℚ
  deriving Repr

/-- Calculates the kerosene percentage in a mixture of two liquids -/
def mixture_kerosene_percent (m : Mixture) : ℚ :=
  (m.liquid1.kerosene_percent * m.parts1 + m.liquid2.kerosene_percent * m.parts2) / (m.parts1 + m.parts2)

theorem kerosene_mixture_theorem (l1 l2 : Liquid) (p1 p2 : ℚ) :
  let m := Mixture.mk l1 l2 p1 p2
  mixture_kerosene_percent m = 27 / 100 :=
by
  sorry

#eval mixture_kerosene_percent (Mixture.mk (Liquid.mk (25/100)) (Liquid.mk (30/100)) 6 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kerosene_mixture_theorem_l114_11451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_volume_calculation_l114_11401

/-- Represents the dimensions of the sand pile -/
structure SandPile where
  coneDiameter : ℝ
  coneHeightRatio : ℝ
  cylinderHeight : ℝ
  cylinderThickness : ℝ

/-- Calculates the total volume of sand delivered -/
noncomputable def totalSandVolume (pile : SandPile) : ℝ :=
  let coneRadius := pile.coneDiameter / 2
  let coneHeight := pile.coneHeightRatio * pile.coneDiameter
  let coneVolume := (Real.pi * coneRadius^2 * coneHeight) / 3
  let cylinderOuterRadius := coneRadius + pile.cylinderThickness
  let cylinderVolume := Real.pi * pile.cylinderHeight * (cylinderOuterRadius^2 - coneRadius^2)
  coneVolume + cylinderVolume

/-- Theorem stating the total volume of sand delivered -/
theorem sand_volume_calculation (pile : SandPile) 
  (h1 : pile.coneDiameter = 12)
  (h2 : pile.coneHeightRatio = 0.5)
  (h3 : pile.cylinderHeight = 2)
  (h4 : pile.cylinderThickness = 1) :
  totalSandVolume pile = 98 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_volume_calculation_l114_11401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_180m_72kmh_l114_11497

/-- The time (in seconds) taken for a train to cross a stationary point. -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem stating that a 180-meter long train moving at 72 km/h takes 9 seconds to cross a stationary point. -/
theorem train_crossing_time_180m_72kmh :
  train_crossing_time 180 72 = 9 := by
  -- Expand the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_180m_72kmh_l114_11497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_sum_inverse_equal_one_l114_11434

/-- Custom operation * for positive real numbers --/
noncomputable def custom_op (m n : ℝ) : ℝ :=
  if m ≥ n then Real.log n / Real.log m else Real.log m / Real.log n

/-- Theorem statement --/
theorem custom_op_sum_inverse_equal_one
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1)
  (hab : a * b = c) :
  (custom_op a c)⁻¹ + (custom_op b c)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_sum_inverse_equal_one_l114_11434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l114_11430

theorem tan_double_angle_second_quadrant (α : Real) :
  α ∈ Set.Ioo (π / 2) π →  -- α is in the second quadrant
  Real.sin α = 4 / 5 →
  Real.tan (2 * α) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l114_11430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l114_11435

theorem projection_of_b_onto_a (e₁ e₂ a b : ℝ → ℝ → ℝ → ℝ) : 
  (∀ x y z, e₁ x y z • e₂ x y z = 0) →
  (∀ x y z, ‖e₁ x y z‖ = 1) →
  (∀ x y z, ‖e₂ x y z‖ = 1) →
  (∀ x y z, a x y z = e₁ x y z + 2 • e₂ x y z) →
  (∀ x y z, b x y z = 4 • e₁ x y z - e₂ x y z) →
  ∀ x y z, (b x y z • a x y z) / ‖a x y z‖ = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_b_onto_a_l114_11435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_and_area_l114_11439

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 100 + y^2 / 36 = 1

-- Define the condition for P being in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the condition for PF₁ ⊥ PF₂
def perpendicular_to_foci (x y : ℝ) : Prop := x^2 + y^2 = 64

-- Define the point P
noncomputable def point_p : ℝ × ℝ := (5 * Real.sqrt 7 / 2, 9 / 2)

-- Define the area of triangle F₁PF₂
def triangle_area : ℝ := 36

-- State the theorem
theorem ellipse_point_and_area :
  ∀ x y : ℝ,
  is_on_ellipse x y →
  in_first_quadrant x y →
  perpendicular_to_foci x y →
  (x, y) = point_p ∧ triangle_area = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_and_area_l114_11439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounded_l114_11410

noncomputable def f (a b x : ℝ) : ℝ := a / x + b / x^2

noncomputable def m (a b : ℝ) : ℝ := max (max (abs a) (abs b)) 1

theorem f_bounded (a b : ℝ) (hab : a * b ≠ 0) (x : ℝ) (hx : abs x > m a b) :
  abs (f a b x) < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounded_l114_11410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l114_11499

-- Define the four statements
def statement1 : Prop := ∀ a b : ℝ, a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3

def statement2 : Prop := ∀ p q : Prop, (p ∨ q) → (p ∧ q)

def statement3 : Prop := (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*(a-b-1)) ↔ (∃ a b : ℝ, a^2 + b^2 ≤ 2*(a-b-1))

noncomputable def statement4 : Prop := ∀ x y : ℝ, x ≠ 0 ∨ y ≠ 0 → 
  (x * y > -2 → ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi/2 ∧ 
    x * y + 2 = (x^2 + 1)^(1/2) * (y^2 + 4)^(1/2) * Real.cos θ)

-- Theorem stating that exactly two of the statements are true
theorem exactly_two_true : 
  (statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) ∨
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4) ∨
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_l114_11499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_4_equals_16_l114_11491

theorem g_4_equals_16 (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g (g x + y) = g (x^2 - y) + 3 * g x * y + 1) 
  (h_g : ∀ x : ℝ, g x = x^2) : 
  g 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_4_equals_16_l114_11491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l114_11458

theorem triangle_inequality (A B C : ℝ) (p q r : ℝ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0) : 
  p * Real.cos A + q * Real.cos B + r * Real.cos C ≤ 
  (1/2) * p * q * r * (1/p^2 + 1/q^2 + 1/r^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l114_11458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l114_11423

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ 
  (n : ℚ) / d = 5 / 11 ∧ 
  Nat.gcd n d = 1 ∧
  n + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_fraction_sum_l114_11423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_pi_inequality_l114_11442

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - (a n)^2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + (b n)^2) - 1) / (b n)

theorem a_b_pi_inequality : ∀ n : ℕ, 2^(n+2) * (a n) < Real.pi ∧ Real.pi < 2^(n+2) * (b n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_pi_inequality_l114_11442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_l114_11459

/-- The number of divisors of n -/
noncomputable def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The condition that n has exactly 49 divisors -/
def has_49_divisors (n : ℕ) : Prop := num_divisors n = 49

/-- The condition that n has exactly 30 divisors -/
def has_30_divisors (n : ℕ) : Prop := num_divisors n = 30

/-- The condition that n has exactly 36 divisors -/
def has_36_divisors (n : ℕ) : Prop := num_divisors n = 36

theorem find_A : ∃ A : ℕ, 
  has_49_divisors (9 * A) ∧ 
  has_30_divisors (2 * A) ∧ 
  has_36_divisors (3 * A) ∧ 
  A = 1176 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_l114_11459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_rearrangements_count_l114_11431

def is_adjacent (a b : Char) : Bool :=
  (a.toNat + 1 = b.toNat) ∨ (b.toNat + 1 = a.toNat)

def is_valid_rearrangement (s : List Char) : Bool :=
  s.length = 4 ∧ 
  s.toFinset = {'w', 'x', 'y', 'z'} ∧
  ¬(s.zip (s.tail)).any (fun (a, b) => is_adjacent a b)

def count_valid_rearrangements : Nat :=
  (List.permutations ['w', 'x', 'y', 'z']).filter is_valid_rearrangement |>.length

theorem valid_rearrangements_count :
  count_valid_rearrangements = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_rearrangements_count_l114_11431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_discount_l114_11475

/-- Represents the discounted price of a concert ticket -/
def discounted_price : ℚ := 8/5

/-- Represents the full price of a concert ticket -/
def full_price : ℚ := 2

/-- Represents the total number of tickets bought -/
def total_tickets : ℕ := 10

/-- Represents the number of discounted tickets bought -/
def discounted_tickets : ℕ := 4

/-- Represents the total amount spent on all tickets -/
def total_spent : ℚ := 92/5

theorem concert_ticket_discount :
  discounted_price * discounted_tickets +
  full_price * (total_tickets - discounted_tickets) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_discount_l114_11475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_steps_from_one_and_two_five_steps_exponent_sum_l114_11441

/-- The operation that combines two numbers a and b to produce c -/
def combine (a b : ℝ) : ℝ := a * b + a + b

/-- Perform one step of the operation, selecting the two largest numbers -/
noncomputable def step (a b : ℝ) : ℝ :=
  let c := combine a b
  if a ≤ b ∧ b ≤ c then combine b c
  else if b ≤ a ∧ a ≤ c then combine a c
  else combine a b

/-- Perform n steps of the operation -/
noncomputable def iterate (n : ℕ) (a b : ℝ) : ℝ :=
  match n with
  | 0 => max a b
  | n + 1 => step a (iterate n a b)

theorem two_steps_from_one_and_two :
  iterate 2 1 2 = 17 := by sorry

theorem five_steps_exponent_sum (p q : ℝ) (m n : ℕ) (hp : p > q) (hq : q > 0) :
  iterate 5 p q = (q + 1) ^ m * (p + 1) ^ n - 1 →
  m + n = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_steps_from_one_and_two_five_steps_exponent_sum_l114_11441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l114_11456

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)
  (scalar * u.1, scalar * u.2)

theorem projection_property (P : (ℝ × ℝ) → (ℝ × ℝ)) :
  P (3, 3) = (45/13, 9/13) →
  ∃ u : ℝ × ℝ, P = projection u →
  P (1, -1) = (10/13, 2/13) := by
  sorry

#check projection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_property_l114_11456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l114_11413

/-- The speed of the current in kilometers per hour -/
noncomputable def current_speed : ℝ := 4

/-- The distance covered downstream in meters -/
noncomputable def distance_downstream : ℝ := 150

/-- The time taken to cover the distance downstream in seconds -/
noncomputable def time_downstream : ℝ := 17.998560115190784

/-- Converts meters to kilometers -/
noncomputable def meters_to_km (m : ℝ) : ℝ := m / 1000

/-- Converts seconds to hours -/
noncomputable def seconds_to_hours (s : ℝ) : ℝ := s / 3600

theorem mans_speed_in_still_water :
  let downstream_speed := meters_to_km distance_downstream / seconds_to_hours time_downstream
  let speed_in_still_water := downstream_speed - current_speed
  speed_in_still_water = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_in_still_water_l114_11413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l114_11455

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)

-- Theorem statement
theorem triangle_angle_value (t : Triangle) :
  (t.a^2 + t.c^2 - t.b^2) * (Real.tan t.B) = Real.sqrt 3 * t.a * t.c →
  t.B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_value_l114_11455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l114_11473

noncomputable def z : ℂ := -Complex.I / (1 + 2 * Complex.I)

theorem z_in_third_quadrant : 
  Real.sign z.re = -1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l114_11473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_size_from_stratified_sample_l114_11485

/-- Represents a population divided into two strata -/
structure StratifiedPopulation where
  strataA : Finset ℕ
  strataB : Finset ℕ
  disjoint : strataA ∩ strataB = ∅

/-- Represents a stratified sampling method -/
def StratifiedSampling (pop : StratifiedPopulation) (sampleSize : ℕ) : Prop :=
  ∃ (sample : Finset ℕ), sample ⊆ pop.strataA ∪ pop.strataB ∧ sample.card = sampleSize

/-- Represents the probability of an individual being selected -/
def ProbabilityOfSelection (pop : StratifiedPopulation) : ℕ → ℚ :=
  fun _ => 1 / 12

theorem population_size_from_stratified_sample
  (pop : StratifiedPopulation)
  (sampleSize : ℕ)
  (hSample : StratifiedSampling pop sampleSize)
  (hProbB : ∀ x ∈ pop.strataB, ProbabilityOfSelection pop x = 1 / 12)
  (hSampleSize : sampleSize = 10) :
  (pop.strataA ∪ pop.strataB).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_size_from_stratified_sample_l114_11485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_integral_inequality_l114_11422

open Real MeasureTheory

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf_continuous : Continuous f)
variable (hf_positive : ∀ x, f x > 0)
variable (hf_periodic : ∀ x, f (x + 2) = f x)

-- Define the existence of the integral
variable (hf_integrable : Integrable (fun x ↦ f (1 + x) / f x) (volume.restrict (Set.Icc 0 2)))

-- State the theorem
theorem periodic_function_integral_inequality :
  (∫ x in Set.Icc 0 2, f (1 + x) / f x) ≥ 2 ∧
  ((∫ x in Set.Icc 0 2, f (1 + x) / f x) = 2 ↔ ∀ x, f (x + 1) = f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_integral_inequality_l114_11422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l114_11445

/-- Given a hyperbola with equation x²/3 - y²/b² = 1, where the distance from a focus
    to an asymptote is √2, prove that the focal distance is 2√5. -/
theorem hyperbola_focal_distance (b : ℝ) :
  (∃ (x y : ℝ), x^2 / 3 - y^2 / b^2 = 1) →
  (∃ (f : ℝ × ℝ) (a : ℝ × ℝ → ℝ), ∀ p : ℝ × ℝ, a p = Real.sqrt 2) →
  (∃ c : ℝ, c = Real.sqrt 5 ∧ 2 * c = 2 * Real.sqrt 5) :=
by
  intro h1 h2
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l114_11445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quarters_l114_11466

/-- Represents the number of quarters (and nickels) -/
def q : ℕ := sorry

/-- The total value of coins in cents -/
def total_value : ℕ := 480

/-- The value of all coins in terms of q -/
def coin_value (q : ℕ) : ℚ := 0.25 * q + 0.05 * q + 0.20 * q

theorem max_quarters :
  coin_value q = total_value / 100 →
  q ≤ 9 ∧ ∃ (q' : ℕ), q' = 9 ∧ coin_value q' = total_value / 100 := by
  sorry

#check max_quarters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quarters_l114_11466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_difference_is_zero_l114_11443

/-- Calculate the perimeter of a rectangle -/
def rectanglePerimeter (length : ℤ) (width : ℤ) : ℤ :=
  2 * (length + width)

/-- Calculate the perimeter of the second figure -/
def secondFigurePerimeter (length : ℤ) (width : ℤ) (additionalHeight : ℤ) : ℤ :=
  rectanglePerimeter (length + 1) (max width additionalHeight)

/-- The positive difference between the perimeters of the two figures is 0 -/
theorem perimeter_difference_is_zero :
  let figure1Perimeter := rectanglePerimeter 5 1
  let figure2Perimeter := secondFigurePerimeter 3 2 2
  (figure1Perimeter - figure2Perimeter).natAbs = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_difference_is_zero_l114_11443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_trajectory_l114_11403

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the parabola -/
def F : Point := ⟨2, 0⟩

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The distance from a point to the line x+2=0 -/
def distanceToDirectrix (p : Point) : ℝ :=
  |p.x + 2|

/-- Predicate for a point being on the parabola -/
def isOnParabola (p : Point) : Prop :=
  distance p F = distanceToDirectrix p

/-- The equation of the parabola -/
def parabolaEquation (p : Point) : Prop :=
  p.y^2 = 8 * p.x

theorem parabola_trajectory :
  ∀ p : Point, isOnParabola p ↔ parabolaEquation p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_trajectory_l114_11403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l114_11427

open Real

theorem tan_cot_equation_solutions :
  ∃ (S : Finset ℝ), 
    (∀ θ ∈ S, 0 < θ ∧ θ < 3 * π ∧ tan (3 * π * cos θ) = 1 / tan (3 * π * sin θ)) ∧
    Finset.card S = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l114_11427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_properties_l114_11400

def row_start (n : ℕ) : ℕ := 2^(n-1)
def row_length (n : ℕ) : ℕ := 2^(n-1)

theorem pattern_properties :
  ∀ n : ℕ, n > 0 →
  (∃ last_num sum_of_row : ℕ,
    -- 1. Last number of the nth row
    last_num = 2^n - 1 ∧
    -- 2. Sum of all numbers in the nth row
    sum_of_row = 3 * 2^(2*n-3) - 2^(n-2)) ∧
  -- 3. Position of 2010
  (2010 ∈ Finset.range (row_length 12) ∧
   2010 = row_start 12 + 986) ∧
  -- 4. Existence and uniqueness of n for the sum condition
  (∃! n : ℕ, 
    (Finset.sum (Finset.range 10) (λ i => 3 * 2^(2*(n+i)-3) - 2^((n+i)-2)))
    = 2^27 - 2^13 - 120 ∧ n = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_properties_l114_11400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l114_11433

/-- The speed of a wheel in miles per hour -/
noncomputable def wheel_speed (circumference : ℝ) (rotation_time : ℝ) : ℝ :=
  (circumference / 5280) * (3600 / rotation_time)

/-- Theorem stating the original speed of the wheel -/
theorem wheel_speed_theorem (original_rotation_time : ℝ) :
  let circumference : ℝ := 15
  let original_speed := wheel_speed circumference original_rotation_time
  let new_rotation_time := original_rotation_time - 1/4
  let new_speed := wheel_speed circumference new_rotation_time
  new_speed = original_speed + 6 → original_speed = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_speed_theorem_l114_11433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_positive_odd_divisors_of_60_l114_11448

def sum_of_positive_odd_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors n)).sum id

theorem sum_of_positive_odd_divisors_of_60 :
  sum_of_positive_odd_divisors 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_positive_odd_divisors_of_60_l114_11448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_compression_work_l114_11449

/-- Work done during isothermal compression of an ideal gas -/
noncomputable def work_done (p₀ H h R : ℝ) : ℝ :=
  p₀ * Real.pi * R^2 * H * Real.log (H / (H - h))

/-- The problem statement -/
theorem isothermal_compression_work :
  ∀ (p₀ H h R : ℝ),
    p₀ > 0 ∧ H > 0 ∧ h > 0 ∧ R > 0 ∧ h < H →
    p₀ = 103300 ∧ H = 0.8 ∧ h = 0.6 ∧ R = 0.2 →
    Int.floor (work_done p₀ H h R) = 14400 := by
  sorry

/-- The equation of state for the ideal gas -/
axiom ideal_gas_law {ρ V : ℝ} (c : ℝ) : ρ * V = c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_compression_work_l114_11449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l114_11467

theorem no_solution_for_equation : ¬∃ (x y n : ℕ), x > 0 ∧ y > 0 ∧ n > 0 ∧ x^2 + y^2 + 41 = 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_equation_l114_11467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_result_l114_11428

def S : Set ℕ := {7, 11, 13, 17, 19, 23}

def process (a b c : ℕ) : ℕ := max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

theorem smallest_result :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 168 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 168 := by
  sorry

#eval process 7 11 13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_result_l114_11428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_is_40_5_l114_11406

-- Define the two lines
def line1 (x : ℝ) : ℝ := x
def line2 : ℝ := -9

-- Define the intersection point
def intersection : ℝ × ℝ := (line2, line1 line2)

-- Define the triangle
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Define the triangle formed by the intersecting lines and x-axis
def triangleArea : Triangle := {
  a := intersection
  b := (line2, 0)
  c := (0, 0)
}

-- Theorem statement
theorem area_of_triangle_is_40_5 : 
  (1/2 : ℝ) * |intersection.1 - triangleArea.c.1| * |intersection.2 - triangleArea.b.2| = 40.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_is_40_5_l114_11406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l114_11444

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi / 6) + 1

theorem f_properties :
  let T := Real.pi
  let increasing_intervals := ({x | 0 ≤ x ∧ x ≤ Real.pi / 6} : Set ℝ) ∪ ({x | 2 * Real.pi / 3 ≤ x ∧ x ≤ Real.pi} : Set ℝ)
  let range_interval := {y | ∃ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 ∧ f x = y}
  (∀ x, f (x + T) = f x) ∧  -- smallest positive period
  (∀ x ∈ increasing_intervals, ∀ y ∈ increasing_intervals, x < y → f x < f y) ∧  -- monotonically increasing intervals
  (Set.Icc 1 (5/2) = range_interval)  -- range of the function
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l114_11444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_shape_l114_11438

/-- Predicate defining a cylinder in cylindrical coordinates -/
def IsCylinder (r θ z : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ r = R ∧ ∀ h, z = h

/-- The shape described by r = 2 cos θ in cylindrical coordinates, 
    with -π/2 ≤ θ ≤ π/2 and z unrestricted, is a cylinder. -/
theorem cylindrical_shape (r θ z : ℝ) : 
  r = 2 * Real.cos θ → 
  -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2 → 
  IsCylinder r θ z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_shape_l114_11438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l114_11478

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := {P | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define the points
variable (O Q : ℝ × ℝ)
variable (r : ℝ)

-- A is on the circle
def A (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := Circle O r

-- Q is inside the circle, different from O
axiom Q_inside : (Q.1 - O.1)^2 + (Q.2 - O.2)^2 < r^2
axiom Q_not_O : Q ≠ O

-- P is on OA
def P (O A : ℝ × ℝ) : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • O + t • A}

-- Define the locus of P
def Locus_P (O Q : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) := 
  {p | ∃ a ∈ A O r, p ∈ P O a}

-- Theorem statement
theorem locus_is_ellipse (O Q : ℝ × ℝ) (r : ℝ) : 
  ∃ F₁ F₂ : ℝ × ℝ, ∃ k : ℝ, 
    Locus_P O Q r = {P : ℝ × ℝ | dist P F₁ + dist P F₂ = k} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_ellipse_l114_11478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_l114_11425

theorem count_triples (m : ℕ) (h_odd : Odd m) (h_ge_5 : m ≥ 5) : 
  (Finset.filter (fun t : Fin m × Fin m × Fin m => 
    t.1.val + t.2.1.val + t.2.2.val + 3 = m ∧ 
    t.1.val + 1 < t.2.1.val + 1 ∧ 
    t.2.1.val + 1 < t.2.2.val + 1) 
    (Finset.univ.product (Finset.univ.product Finset.univ))).card = m * (m - 3) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_l114_11425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_calculation_l114_11450

/-- Given the average temperatures for three consecutive days and the temperature of the last day,
    calculate the temperature of the first day. -/
theorem temperature_calculation (temp_tue wed thu fri : ℚ) : 
  (temp_tue + wed + thu) / 3 = 42 →
  (wed + thu + fri) / 3 = 44 →
  fri = 43 →
  temp_tue = 37 := by
  sorry

#check temperature_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_calculation_l114_11450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l114_11424

/-- Parametric curve defined by x = 4(t - sin t) and y = 4(1 - cos t) -/
noncomputable def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (4 * (t - Real.sin t), 4 * (1 - Real.cos t))

/-- Line y = 4 -/
def line_y_eq_4 : ℝ → ℝ := fun _ => 4

/-- The domain of x: 0 < x < 8π -/
def x_domain (x : ℝ) : Prop := 0 < x ∧ x < 8 * Real.pi

/-- The condition y ≥ 4 -/
def y_condition (y : ℝ) : Prop := y ≥ 4

/-- The area bounded by the given curves -/
noncomputable def bounded_area : ℝ := 24 * Real.pi + 64

/-- Theorem stating the existence of t₁ and t₂ satisfying the area calculation -/
theorem area_calculation :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := parametric_curve t₁
    let (x₂, y₂) := parametric_curve t₂
    x_domain x₁ ∧ x_domain x₂ ∧
    y₁ = line_y_eq_4 x₁ ∧ y₂ = line_y_eq_4 x₂ ∧
    (∀ t ∈ Set.Icc t₁ t₂, y_condition (parametric_curve t).2) ∧
    (∫ (t : ℝ) in t₁..t₂, (parametric_curve t).2 * (4 * (1 - Real.cos t))) = bounded_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_calculation_l114_11424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l114_11474

/-- The distance from the center of the circle (x+1)^2 + y^2 = 2 to the line y = 2x + 3 is √5/5 -/
theorem distance_circle_center_to_line : ∃ d : ℝ, d = Real.sqrt 5 / 5 := by
  -- Define the circle equation
  let circle_eq : ℝ → ℝ → Prop := λ x y ↦ (x + 1)^2 + y^2 = 2
  
  -- Define the line equation
  let line_eq : ℝ → ℝ → Prop := λ x y ↦ y = 2*x + 3
  
  -- Define the distance function
  let distance : ℝ → ℝ → ℝ → ℝ → ℝ := λ x1 y1 x2 y2 ↦ 
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  
  -- State the theorem
  have dist_circle_center_to_line : 
    ∃ (x_center y_center : ℝ), 
      circle_eq x_center y_center ∧
      (∀ (x y : ℝ), line_eq x y → 
        distance x_center y_center x y = Real.sqrt 5 / 5) := by
    -- Proof
    sorry

  -- Conclude the theorem
  exact ⟨Real.sqrt 5 / 5, rfl⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l114_11474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l114_11482

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/25 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y fx fy : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (f1x f1y f2x f2y : ℝ) 
  (h_foci : f1x^2 + f1y^2 = f2x^2 + f2y^2 ∧ f1x * f2x ≤ 0 ∧ f1y * f2y ≤ 0) 
  (h_distance_f1 : distance_to_focus x y f1x f1y = 6) :
  distance_to_focus x y f2x f2y = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l114_11482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_two_zeros_l114_11484

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- Statement for the maximum value of f(x)
theorem f_max_value : ∃ (x : ℝ), ∀ (y : ℝ), y > 0 → f y ≤ f x ∧ f x = 1 / Real.exp 1 := by
  sorry

-- Statement for the condition on a for g(x) to have two zeros
theorem g_two_zeros (a : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ g a x = 0 ∧ g a y = 0) ↔ a > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_g_two_zeros_l114_11484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_toy_intersection_l114_11492

/-- The point where the robot begins to move away from the toy -/
noncomputable def intersection_point : ℝ × ℝ := (3/17, 135/17)

/-- The coordinates of the toy -/
def toy_position : ℝ × ℝ := (15, 12)

/-- The equation of the line along which the robot moves -/
noncomputable def robot_path (x : ℝ) : ℝ := -4*x + 9

/-- The perpendicular line from the toy to the robot's path -/
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1/4)*x + 33/4

theorem robot_toy_intersection :
  let (c, d) := intersection_point
  (robot_path c = d) ∧
  (perpendicular_line c = d) ∧
  (c + d = 138/17) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robot_toy_intersection_l114_11492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l114_11414

-- Define the domain for both functions (non-zero real numbers)
def NonZeroReals : Type := {x : ℝ // x ≠ 0}

-- Define the two functions
noncomputable def f (x : NonZeroReals) : ℝ := x.val / x.val
noncomputable def g (x : NonZeroReals) : ℝ := 1 / (x.val ^ 0)

-- Theorem stating that f and g are equivalent
theorem f_equals_g : ∀ x : NonZeroReals, f x = g x := by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l114_11414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_squared_distance_is_84_over_pi_l114_11469

/-- Triangle with side lengths 13, 14, and 15 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 13)
  (hb : b = 14)
  (hc : c = 15)

/-- Expected value of squared distance from random interior point to perimeter -/
noncomputable def expected_squared_distance (t : Triangle) : ℝ := 84 / Real.pi

/-- Theorem statement -/
theorem expected_squared_distance_is_84_over_pi (t : Triangle) :
  expected_squared_distance t = 84 / Real.pi := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_squared_distance_is_84_over_pi_l114_11469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_l114_11498

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to a line ax + by + c = 0 -/
noncomputable def distanceToLine (p : Point) (a b c : ℝ) : ℝ :=
  |a * p.x + b * p.y + c| / Real.sqrt (a^2 + b^2)

/-- The point is equally distant from x-axis, y-axis, and line x + y = 4 -/
def isEquidistant (p : Point) : Prop :=
  distanceToLine p 0 1 0 = distanceToLine p 1 0 0 ∧
  distanceToLine p 0 1 0 = distanceToLine p 1 1 (-4)

theorem equidistant_point_x_coordinate :
  ∃ (p : Point), isEquidistant p ∧ p.x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_x_coordinate_l114_11498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l114_11480

-- Define the triangle and its properties
def Triangle (a b c A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C)

-- State the theorem
theorem triangle_problem 
  (a b c A B C : ℝ)
  (h_triangle : Triangle a b c A B C)
  (h_condition : (b - 2*c) * Real.cos A = a - 2*a * (Real.cos (B/2))^2) :
  A = Real.pi/3 ∧ 
  (a = Real.sqrt 3 → Real.sqrt 3/2 < b + c ∧ b + c ≤ Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l114_11480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l114_11436

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3) * Real.cos x

-- Define the theorem
theorem f_properties :
  -- Range of f(x) for 0 ≤ x ≤ π/2
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 3 / 2) ∧
  -- Triangle ABC properties
  (∀ A B C a b c : ℝ,
    -- A is acute
    0 < A ∧ A < Real.pi / 2 →
    -- f(A) = √3/2
    f A = Real.sqrt 3 / 2 →
    -- b = 2, c = 3
    b = 2 ∧ c = 3 →
    -- Triangle inequality
    a + b > c ∧ b + c > a ∧ c + a > b →
    -- Conclusion
    Real.cos (A - B) = 5 * Real.sqrt 7 / 14) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l114_11436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l114_11420

-- Part 1
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

noncomputable def F (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f b x else -(f b x)

theorem part_one (b : ℝ) (h : f b (-1) = 0) : F b 2 + F b (-2) = 8 := by
  sorry

-- Part 2
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

theorem part_two (b : ℝ) (h : ∀ x ∈ Set.Ioo 0 1, -1 ≤ g b x ∧ g b x ≤ 1) :
  b ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l114_11420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_m_equals_2n_minus_1_l114_11464

/-- Definition of the set R_n -/
def R (n : ℕ) : Set ℤ :=
  if n % 2 = 0 then
    let k := n / 2
    {i : ℤ | -k ≤ i ∧ i ≤ k ∧ i ≠ 0}
  else
    let k := n / 2
    {i : ℤ | -k ≤ i ∧ i ≤ k}

/-- Definition of a "good" marking -/
def isGoodMarking {α : Type} (n : ℕ) (marking : α → ℤ) : Prop :=
  ∀ x y : α, x ≠ y → marking x ∈ R n ∧ marking y ∈ R n → marking x ≠ marking y

/-- Definition of an "insightful" marking -/
def isInsightfulMarking {α : Type} (m : ℕ) (marking : α → ℤ) : Prop :=
  (∀ x y : α, x ≠ y → marking x ∈ R m ∧ marking y ∈ R m → marking x ≠ marking y) ∧
  (∀ x y : α, x ≠ y → marking x ∈ R m ∧ marking y ∈ R m → marking x + marking y ≠ 0)

/-- The main theorem -/
theorem smallest_m (n : ℕ) (h : n ≥ 3) :
  (∀ α : Type, ∀ marking : α → ℤ, isGoodMarking n marking → ∃ m : ℕ, isInsightfulMarking m marking) →
  ∃ m : ℕ, (∀ α : Type, ∀ marking : α → ℤ, isGoodMarking n marking → isInsightfulMarking m marking) ∧
           (∀ k < m, ¬(∀ α : Type, ∀ marking : α → ℤ, isGoodMarking n marking → isInsightfulMarking k marking)) :=
by
  sorry

/-- Proof that m = 2n - 1 is the smallest value satisfying the conditions -/
theorem m_equals_2n_minus_1 (n : ℕ) (h : n ≥ 3) :
  ∃ m : ℕ, m = 2*n - 1 ∧
    (∀ α : Type, ∀ marking : α → ℤ, isGoodMarking n marking → isInsightfulMarking m marking) ∧
    (∀ k < m, ¬(∀ α : Type, ∀ marking : α → ℤ, isGoodMarking n marking → isInsightfulMarking k marking)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_m_equals_2n_minus_1_l114_11464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_2023rd_derivative_l114_11489

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def nth_derivative (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => deriv (nth_derivative f n)

noncomputable def tangent_line_y_intercept (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  -(nth_derivative f n 0) / (nth_derivative f (n + 1) 0)

theorem y_intercept_of_2023rd_derivative :
  tangent_line_y_intercept f 2023 = -2023 / 2024 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_2023rd_derivative_l114_11489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_to_g_transformation_l114_11486

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (4 * x - 3 * Real.pi / 4)

theorem f_to_g_transformation (x : ℝ) : 
  g (x + Real.pi / 4) = f (x / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_to_g_transformation_l114_11486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l114_11446

noncomputable def z (m : ℝ) : ℂ := (m^2 - m - 6) / (m + 3) + (m^2 + 5*m + 6) * Complex.I

theorem complex_number_properties :
  (∀ m : ℝ, (z m).im = 0 ↔ m = -2) ∧
  (∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l114_11446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_f₄_l114_11419

noncomputable def f₁ (x : ℝ) := 2018 * Real.sin x
noncomputable def f₂ (x : ℝ) := Real.sin (2018 * x)
noncomputable def f₃ (x : ℝ) := -Real.cos (2 * x)
noncomputable def f₄ (x : ℝ) := Real.sin (4 * x + Real.pi / 4)

def smallest_positive_period (f : ℝ → ℝ) : ℝ := sorry

theorem smallest_period_f₄ :
  smallest_positive_period f₄ = Real.pi / 2 ∧
  smallest_positive_period f₄ < smallest_positive_period f₁ ∧
  smallest_positive_period f₄ < smallest_positive_period f₂ ∧
  smallest_positive_period f₄ < smallest_positive_period f₃ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_f₄_l114_11419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adeline_working_days_l114_11494

/-- Calculates the number of working days per week given hourly rate, daily work hours, total earnings, and total weeks. -/
noncomputable def working_days_per_week (hourly_rate : ℝ) (daily_hours : ℝ) (total_earnings : ℝ) (total_weeks : ℝ) : ℝ :=
  total_earnings / (hourly_rate * daily_hours * total_weeks)

/-- Proves that given the specified conditions, the number of working days per week is 5. -/
theorem adeline_working_days :
  let hourly_rate : ℝ := 12
  let daily_hours : ℝ := 9
  let total_earnings : ℝ := 3780
  let total_weeks : ℝ := 7
  working_days_per_week hourly_rate daily_hours total_earnings total_weeks = 5 := by
  -- Unfold the definition of working_days_per_week
  unfold working_days_per_week
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adeline_working_days_l114_11494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_no_solution_when_greater_than_one_l114_11404

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x^3 else -x^2 + 2*x

-- State the theorem
theorem unique_solution_for_f (a : ℝ) :
  f a = -5/4 → a = -1/2 := by
  sorry

-- Additional theorem to demonstrate the other case
theorem no_solution_when_greater_than_one (a : ℝ) :
  a > 1 → f a ≠ -5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_f_no_solution_when_greater_than_one_l114_11404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_cross_section_figures_l114_11415

-- Define the types for our geometric figures and plane
structure GeometricFigure where
  -- Add some placeholder field to make the structure valid
  dummy : Unit

structure Plane where
  -- Add some placeholder field to make the structure valid
  dummy : Unit

structure Circle where
  -- Add some placeholder field to make the structure valid
  dummy : Unit

-- Define the property of having a circular cross-section
def has_circular_cross_section (figure : GeometricFigure) (plane : Plane) : Prop :=
  ∃ (c : Circle), True  -- We replace the intersection with a trivial proposition

-- Define specific geometric figures
def Cone : GeometricFigure :=
  ⟨()⟩  -- Use unit value to initialize the dummy field

def Cylinder : GeometricFigure :=
  ⟨()⟩

def Sphere : GeometricFigure :=
  ⟨()⟩

-- Theorem stating that if a figure has a circular cross-section, it could be a cone, cylinder, or sphere
theorem circular_cross_section_figures 
  (figure : GeometricFigure) 
  (plane : Plane) 
  (h : has_circular_cross_section figure plane) :
  figure = Cone ∨ figure = Cylinder ∨ figure = Sphere :=
by
  sorry  -- Skip the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_cross_section_figures_l114_11415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_l114_11454

noncomputable section

/-- Parabola with directrix x = -3/2 -/
def parabola_directrix : ℝ := -3/2

/-- Hyperbola passes through (2, 0) -/
def hyperbola_point1 : ℝ × ℝ := (2, 0)

/-- Hyperbola passes through (2√3, √6) -/
noncomputable def hyperbola_point2 : ℝ × ℝ := (2 * Real.sqrt 3, Real.sqrt 6)

/-- The focus of the hyperbola is on the x-axis -/
axiom hyperbola_focus_on_x_axis : Prop

/-- The equation of the parabola is y² = 6x -/
def parabola_equation (x y : ℝ) : Prop := y^2 = 6*x

/-- The equation of the hyperbola is x²/4 - y²/3 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop := x^2/4 - y^2/3 = 1

theorem curve_equations :
  (∀ x y : ℝ, parabola_equation x y) ∧
  (∀ x y : ℝ, hyperbola_equation x y) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_l114_11454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_sum_l114_11452

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (Finset.range 16).sum (λ i => ω^(25 + 3*i)) = ω^7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_sum_l114_11452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electrician_to_worker_salary_ratio_l114_11407

/-- Represents the daily salary of a worker in dollars -/
def worker_salary : ℚ := 100

/-- Represents the number of construction workers -/
def num_construction_workers : ℕ := 2

/-- Represents the plumber's salary as a percentage of a worker's salary -/
def plumber_salary_percentage : ℚ := 250

/-- Represents the total daily labor costs in dollars -/
def total_labor_costs : ℚ := 650

/-- Theorem: The ratio of the electrician's salary to a construction worker's salary is 2:1 -/
theorem electrician_to_worker_salary_ratio :
  (total_labor_costs - (num_construction_workers * worker_salary + (plumber_salary_percentage / 100) * worker_salary)) / worker_salary = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electrician_to_worker_salary_ratio_l114_11407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_equation_solutions_l114_11479

/-- The solutions to the equation ∜(60 - 3x) + ∜(20 + 3x) = 4 are x = 20 and x = -20 -/
theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((60 - 3*x) ^ (1/4 : ℝ) + (20 + 3*x) ^ (1/4 : ℝ)) = 4) ↔ (x = 20 ∨ x = -20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_equation_solutions_l114_11479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_g_l114_11487

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := ((x + 4) / 5) ^ (1/3)

-- State the theorem
theorem unique_solution_for_g :
  ∃! x : ℝ, g (3 * x) = 3 * g x :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_g_l114_11487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l114_11477

theorem angle_sum (α β : Real) : 
  α ∈ Set.Ioo 0 (π/2) →
  β ∈ Set.Ioo 0 (π/2) →
  Real.cos (α - β/2) = Real.sqrt 3/2 →
  Real.sin (α/2 - β) = -1/2 →
  α + β = 2*π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l114_11477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_after_discount_l114_11461

/-- Represents the cost in currency A -/
abbrev CurrencyA := ℝ

/-- Represents the cost in currency B -/
abbrev CurrencyB := ℝ

/-- Cost of 1 kg of mangos in currency A -/
def mango_cost : CurrencyA := 10

/-- Cost of 1 kg of rice in currency B -/
def rice_cost : CurrencyB := 8

/-- Cost of 1 kg of flour in currency A -/
def flour_cost : CurrencyA := 21

/-- Conversion rate from currency B to currency A -/
noncomputable def conversion_rate : ℝ := 1 / 2

/-- Discount rate applied to the total cost -/
def discount_rate : ℝ := 0.1

/-- Theorem stating the total cost after discount -/
theorem total_cost_after_discount :
  let mangos_weight : ℝ := 4
  let rice_weight : ℝ := 3
  let flour_weight : ℝ := 5
  let total_cost : CurrencyA := 
    mangos_weight * mango_cost + 
    rice_weight * rice_cost * conversion_rate + 
    flour_weight * flour_cost
  let discounted_cost : CurrencyA := total_cost * (1 - discount_rate)
  discounted_cost = 141.30 := by
  sorry

#eval mango_cost
#eval rice_cost
#eval flour_cost
#eval discount_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_after_discount_l114_11461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_8_equals_8_l114_11496

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := sumOfDigits ((n : ℕ)^2 + 1)

/-- Recursive definition of f_k -/
def f_k : ℕ → ℕ+ → ℕ
  | 0, n => f n
  | k + 1, n => f_k k (⟨f_k k n, sorry⟩)

theorem f_2010_8_equals_8 : f_k 2010 8 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_8_equals_8_l114_11496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l114_11418

/-- An arithmetic sequence with common difference d -/
noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_d_neg : d < 0)
  (h_prod : a 2 * a 4 = 12)
  (h_sum : a 2 + a 4 = 8) :
  a 1 = 8 ∧ d = -2 ∧ arithmetic_sum a 10 = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l114_11418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_convergence_l114_11408

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = C.1 ∧ B.2 = 0 ∧ C = (0, 0)

-- Define the sequence of points
noncomputable def P (a : ℝ) : ℕ → ℝ × ℝ
  | 0 => (0, a)
  | 1 => (a/2, 0)
  | n+2 => if n % 2 = 0 
           then ((P a n).1/2, ((P a n).2 + a)/2) 
           else (((P a n).1 + a)/2, (P a n).2/2)

-- State the theorem
theorem midpoint_convergence (a : ℝ) (h : a > 0) :
  ∃ (A B C : ℝ × ℝ), Triangle A B C ∧ 
  (∀ ε > 0, ∃ N : ℕ, ∀ k ≥ N, 
    dist (P a (2*k)) (a/3, 2*a/3) < ε ∧ 
    dist (P a (2*k+1)) (2*a/3, a/3) < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_convergence_l114_11408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_satisfies_triangle_inequality_l114_11421

/-- The equation of the conic section --/
noncomputable def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 12

/-- The first focus of the ellipse --/
def focus1 : ℝ × ℝ := (0, 2)

/-- The second focus of the ellipse --/
def focus2 : ℝ × ℝ := (6, -4)

/-- The distance between the two foci --/
noncomputable def foci_distance : ℝ :=
  Real.sqrt ((focus2.1 - focus1.1)^2 + (focus2.2 - focus1.2)^2)

/-- Theorem stating that the given equation represents an ellipse --/
theorem is_ellipse : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focus1.1 + focus2.1) / 2)^2 / a^2 +
    (y - (focus1.2 + focus2.2) / 2)^2 / b^2 = 1 :=
sorry

/-- Theorem stating that the equation satisfies the triangle inequality --/
theorem satisfies_triangle_inequality : 12 > foci_distance :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_satisfies_triangle_inequality_l114_11421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_g_above_l114_11417

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2*|x + a|

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (1/2)*x + b

theorem f_inequality_and_g_above (a b : ℝ) :
  (a = 1/2 → {x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 0}) ∧
  (a ≥ -1 → (∀ x, g b x > f a x) → 2*b - 3*a > 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_g_above_l114_11417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreases_with_abs_value_l114_11476

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x - x^2

-- State the theorem
theorem f_decreases_with_abs_value {x₁ x₂ : ℝ} 
  (h1 : x₁ ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h2 : x₂ ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h3 : |x₁| > |x₂|) : 
  f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreases_with_abs_value_l114_11476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_negative_four_x_squared_l114_11453

/-- The equation of the directrix of a parabola y = ax² --/
noncomputable def directrix_equation (a : ℝ) : ℝ := -1 / (4 * a)

/-- Theorem: The directrix of the parabola y = -4x² is y = 1/16 --/
theorem directrix_of_negative_four_x_squared :
  directrix_equation (-4) = 1/16 := by
  -- Unfold the definition of directrix_equation
  unfold directrix_equation
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_negative_four_x_squared_l114_11453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_perpendicular_l114_11481

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def l₁ (x y : ℝ) : Prop := x - 2*y - 1 = 0

def l₂ (a b x y : ℝ) : Prop := a*x + b*y - 1 = 0

def perpendicular (a b : ℝ) : Prop := a - 2*b = 0

def count_perpendicular : ℕ := 3

theorem probability_perpendicular :
  (∀ a b, a ∈ S ∧ b ∈ S → (perpendicular a b ↔ (a = 2 ∧ b = 1) ∨ (a = 4 ∧ b = 2) ∨ (a = 6 ∧ b = 3))) →
  (Finset.card S * Finset.card S = 36) →
  count_perpendicular = 3 →
  (count_perpendicular : ℚ) / (Finset.card S * Finset.card S : ℚ) = 1 / 12 := by
  sorry

#eval Finset.card S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_perpendicular_l114_11481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cirrus_cloud_count_l114_11460

theorem cirrus_cloud_count 
  (cirrus_count : ℕ) (cumulus_count : ℕ) (cumulonimbus_count : ℕ) :
  cirrus_count = 4 * cumulus_count →
  cumulus_count = 12 * cumulonimbus_count →
  cumulonimbus_count = 3 →
  cirrus_count = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cirrus_cloud_count_l114_11460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_seq_convergence_l114_11465

noncomputable def x_seq (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => 1 + Real.log ((x_seq a n)^2 / (1 + Real.log (x_seq a n)))

theorem x_seq_convergence (a : ℝ) (h : a ≥ 1) :
  ∃ (L : ℝ), L = 1 ∧ Filter.Tendsto (x_seq a) Filter.atTop (nhds L) := by
  sorry

#check x_seq_convergence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_seq_convergence_l114_11465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_telegraph_post_l114_11483

/-- The time (in seconds) it takes for a train to cross a stationary object -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A train 560 m long traveling at 126 km/h takes 16 seconds to cross a telegraph post -/
theorem train_crossing_telegraph_post :
  train_crossing_time 560 126 = 16 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp [div_eq_mul_inv]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_telegraph_post_l114_11483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l114_11493

-- Define the function f as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_value (α : ℝ) :
  f α 2 = Real.sqrt 2 / 2 → f α 4 = 1 / 2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l114_11493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_terms_l114_11432

/-- An arithmetic sequence where a_5 and a_7 are roots of x^2 - 2x - 6 = 0 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n, a (n + 1) = a n + d) ∧
  (a 5)^2 - 2*(a 5) - 6 = 0 ∧
  (a 7)^2 - 2*(a 7) - 6 = 0

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem sum_eleven_terms (a : ℕ → ℚ) (h : arithmetic_sequence a) :
  sum_arithmetic a 11 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_terms_l114_11432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_micheal_work_days_l114_11426

/-- The number of days it takes Micheal and Adam to complete the work together -/
noncomputable def combined_days : ℝ := 20

/-- The number of days Micheal and Adam work together before Micheal stops -/
noncomputable def days_worked_together : ℝ := 11

/-- The fraction of work remaining after Micheal stops -/
noncomputable def remaining_work : ℝ := 9 / 20

/-- The number of days it takes Adam to complete the remaining work -/
noncomputable def adam_completion_days : ℝ := 10

/-- The number of days it would take Micheal to complete the work alone -/
noncomputable def micheal_solo_days : ℝ := 200

theorem micheal_work_days :
  (1 / micheal_solo_days) + (remaining_work / adam_completion_days) = (1 / combined_days) ∧
  micheal_solo_days = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_micheal_work_days_l114_11426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l114_11471

/-- Represents the tax structure of Country X -/
structure TaxStructure where
  lower_bracket : ℚ
  lower_rate : ℚ
  upper_rate : ℚ

/-- Calculates the tax for a given income and tax structure -/
noncomputable def calculate_tax (income : ℚ) (tax : TaxStructure) : ℚ :=
  let lower_tax := min income tax.lower_bracket * tax.lower_rate
  let upper_tax := max (income - tax.lower_bracket) 0 * tax.upper_rate
  lower_tax + upper_tax

/-- Theorem stating that given the tax structure and total tax paid, the citizen's income is $58,000 -/
theorem income_from_tax (tax : TaxStructure) (total_tax : ℚ) :
  tax.lower_bracket = 40000 ∧ 
  tax.lower_rate = 11/100 ∧ 
  tax.upper_rate = 1/5 ∧
  total_tax = 8000 →
  ∃ (income : ℚ), calculate_tax income tax = total_tax ∧ income = 58000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l114_11471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l114_11416

/-- Represents the speed of a train in km/hr -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (3600 / 1000)

/-- Proves that a train with the given length and time to cross a pole has the specified speed -/
theorem train_speed_calculation (length time : ℝ) 
  (h1 : length = 175) 
  (h2 : time = 9) : 
  train_speed length time = 70 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Substitute the given values
  rw [h1, h2]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l114_11416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_when_c_equals_a_BC_BD_over_S_range_l114_11457

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (S : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.b * Real.cos t.C - t.c * Real.cos t.B = 2 * t.a ∧
  t.c ≤ 2 * t.a

-- Theorem 1
theorem angle_B_when_c_equals_a (t : Triangle) 
  (h : triangle_conditions t) (h_c_eq_a : t.c = t.a) : 
  t.B = 2 * Real.pi / 3 :=
sorry

-- Theorem 2
theorem BC_BD_over_S_range (t : Triangle) (h : triangle_conditions t) :
  ∃ (D : ℝ), 8/9 ≤ t.a * D / t.S ∧ t.a * D / t.S ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_when_c_equals_a_BC_BD_over_S_range_l114_11457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_formation_l114_11490

-- Define Quadrilateral as a structure
structure Quadrilateral (α : Type*) :=
(sides : Finset α)
(side_count : sides.card = 4)

theorem quadrilateral_formation (a b c d : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) →
  (a + b + c + d = 2) →
  (∃ (quad : Quadrilateral ℝ), quad.sides = {a, b, c, d}) ↔ 
  ((1/4 : ℝ) ≤ a ∧ a ≤ (1/2 : ℝ) ∧
   (1/4 : ℝ) ≤ b ∧ b ≤ (1/2 : ℝ) ∧
   (1/4 : ℝ) ≤ c ∧ c ≤ (1/2 : ℝ) ∧
   (1/4 : ℝ) ≤ d ∧ d ≤ (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_formation_l114_11490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l114_11409

theorem constant_term_expansion :
  let f (x : ℝ) := (x^2 + 2) * (x - 1/x)^6
  ∃ (c : ℝ), ∀ x ≠ 0, f x = c + x * (f x - c) / x
  ∧ c = -25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l114_11409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brads_pumpkin_weight_l114_11412

/-- The weight of Brad's pumpkin in pounds -/
noncomputable def B : ℝ := 54

/-- The weight of Jessica's pumpkin in pounds -/
noncomputable def J : ℝ := B / 2

/-- The weight of Betty's pumpkin in pounds -/
noncomputable def T : ℝ := 4 * J

/-- The difference between the heaviest and lightest pumpkin is 81 pounds -/
axiom weight_difference : T - J = 81

theorem brads_pumpkin_weight : B = 54 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brads_pumpkin_weight_l114_11412
