import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_integer_l40_4071

def digit_of (n : ℕ) (d : ℕ) : ℕ :=
  (n / (10 ^ d)) % 10

theorem no_special_integer : ¬ ∃ (n : ℕ+), 
  (∀ d : ℕ, digit_of n.val d > 5) ∧ 
  (∀ d : ℕ, digit_of (n.val ^ 2) d < 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_integer_l40_4071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_approximation_l40_4014

theorem complex_calculation_approximation :
  let x := (6859 : ℝ)^(1/3) * Real.sqrt 1024 - Real.sqrt 1764 / (729 : ℝ)^(1/3) + Real.sqrt 1089 * Real.sqrt 484
  ∃ ε > 0, |x - 1329.333| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_approximation_l40_4014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l40_4018

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 6*x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (3/2, 0)

-- Define the line passing through the focus with 45° inclination
noncomputable def line (x y : ℝ) : Prop := y = x - 3/2

-- Define the intersection points of the line and the parabola
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem parabola_intersection_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l40_4018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_f_at_powers_of_ζ_l40_4019

noncomputable def f (x : ℂ) : ℂ := 1 + 2*x + 3*x^2 + 4*x^3 + 5*x^4

noncomputable def ζ : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem product_of_f_at_powers_of_ζ : f ζ * f (ζ^2) * f (ζ^3) * f (ζ^4) = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_f_at_powers_of_ζ_l40_4019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_periodic_sine_l40_4064

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ := Real.sin (a n)

def S (a : ℕ → ℝ) : Set ℝ := {x | ∃ n : ℕ, x = b a n}

theorem arithmetic_sequence_periodic_sine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∃ t : ℕ, t > 0 ∧ t ≤ 8 ∧ ∀ n : ℕ, b a (n + t) = b a n) →
  (∃ s : Finset ℝ, s.card = 4 ∧ S a = s) →
  ∃ T : Finset ℕ, T.card = 4 ∧ ∀ t ∈ T, t > 0 ∧ t ≤ 8 ∧ ∀ n : ℕ, b a (n + t) = b a n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_periodic_sine_l40_4064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_is_21_l40_4076

-- Define the problem parameters
def polygon_sides (a b c : ℕ) : Prop :=
  a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ (a = c ∨ a = b ∨ b = c)

-- Define the angle sum condition
def angle_sum_360 (a b c : ℕ) : Prop :=
  (a - 2) / a + (b - 2) / b + (c - 2) / c = 2

-- Define the perimeter calculation
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c - 6

-- Theorem: The maximum perimeter is 21
theorem max_perimeter_is_21 :
  ∃ a b c : ℕ, polygon_sides a b c ∧ angle_sum_360 a b c ∧
  perimeter a b c = 21 ∧
  ∀ x y z : ℕ, polygon_sides x y z → angle_sum_360 x y z →
  perimeter x y z ≤ 21 :=
sorry

-- Proof sketch
/-
Proof:
1. We can show that (a-4)(b-2) = 8 is equivalent to the angle sum condition.
2. The possible integer solutions for (a-4, b-2) are (1,8), (2,4), (4,2), and (8,1).
3. These correspond to (a,b) pairs: (5,10), (6,6), (8,4), and (12,3).
4. Calculating perimeters for each case (including the third polygon c = a):
   - (5,10,5): 14
   - (6,6,6): 12
   - (8,4,8): 14
   - (12,3,12): 21
5. The maximum perimeter is 21, achieved with (12,3,12).
6. We can prove that no other combination satisfying the conditions can exceed 21.
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_is_21_l40_4076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_one_l40_4095

noncomputable def sequenceLimit (n : ℕ) : ℚ :=
  (Nat.factorial (2*n+1) + Nat.factorial (2*n+2)) /
  (Nat.factorial (2*n+3) - Nat.factorial (2*n+2))

theorem sequence_limit_is_one :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sequenceLimit n - 1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_one_l40_4095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_probabilities_l40_4028

/-- Represents a bag of cards -/
structure Bag where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- The setup of the problem -/
def problem_setup : (Bag × Bag) := 
  ({zeros := 1, ones := 2, twos := 3}, {zeros := 4, ones := 1, twos := 2})

/-- The number of cards drawn from each bag -/
def cards_drawn : (ℕ × ℕ) := (1, 2)

/-- Calculates the probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  (favorable : ℚ) / (total : ℚ)

theorem card_drawing_probabilities 
  (setup : (Bag × Bag) := problem_setup) 
  (drawn : (ℕ × ℕ) := cards_drawn) :
  let (bagA, bagB) := setup
  let (fromA, fromB) := drawn
  let total_outcomes := (Nat.choose (bagA.zeros + bagA.ones + bagA.twos) fromA) * 
                        (Nat.choose (bagB.zeros + bagB.ones + bagB.twos) fromB)
  probability (Nat.choose bagA.zeros fromA * Nat.choose bagB.zeros fromB) total_outcomes = 1 / 21 ∧ 
  probability ((Nat.choose bagA.twos 2 * Nat.choose bagA.ones 1 * Nat.choose bagB.twos 1) + 
               (Nat.choose bagA.twos 1 * Nat.choose bagB.twos 1 * Nat.choose bagB.ones 1)) 
              total_outcomes = 4 / 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_drawing_probabilities_l40_4028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_division_percentage_l40_4078

theorem second_division_percentage
  (total_students : ℕ)
  (first_division_percentage : ℚ)
  (just_passed : ℕ)
  (h1 : total_students = 300)
  (h2 : first_division_percentage = 25 / 100)
  (h3 : just_passed = 63) :
  ∃ second_division_percentage : ℚ,
    second_division_percentage = 54 / 100 ∧
    first_division_percentage * total_students + just_passed + (second_division_percentage * total_students) = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_division_percentage_l40_4078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_turn_point_sum_l40_4017

/-- The point where the fly starts moving away from the spider -/
noncomputable def fly_turn_point (spider_x spider_y fly_line_slope fly_line_intercept : ℝ) : ℝ × ℝ :=
  let perp_slope := -1 / fly_line_slope
  let perp_intercept := spider_y - perp_slope * spider_x
  let x := (fly_line_intercept - perp_intercept) / (perp_slope - fly_line_slope)
  let y := fly_line_slope * x + fly_line_intercept
  (x, y)

/-- Theorem stating that the sum of coordinates of the turn point is 205/37 -/
theorem fly_turn_point_sum :
  let (a, b) := fly_turn_point 15 5 (-6) 15
  a + b = 205 / 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_turn_point_sum_l40_4017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l40_4031

/-- Given a positive geometric sequence with common ratio 3, prove that the minimum value of 2/m + 1/(2n) is 3/4, where m and n satisfy a_m * a_n = 9 * a_2^2 -/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) : 
  (∀ k, a (k + 1) = 3 * a k) →  -- geometric sequence with common ratio 3
  (∀ k, a k > 0) →  -- positive sequence
  a m * a n = 9 * (a 2)^2 →  -- given condition
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 1/(2*y) ≥ 3/4) ∧  -- minimum exists and is at least 3/4
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/(2*y) ≥ 3/4) →  -- for all positive x and y, the expression is at least 3/4
  IsGLB {z : ℝ | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ z = 2/x + 1/(2*y)} (3/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l40_4031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_effort_distribution_l40_4016

/-- Represents the distance from home to the train station in kilometers. -/
noncomputable def distance_to_station : ℝ := 4

/-- Represents the number of suitcases. -/
def num_suitcases : ℕ := 2

/-- Represents the number of people carrying the suitcases. -/
def num_people : ℕ := 3

/-- Calculates the total distance all suitcases need to be carried. -/
noncomputable def total_carrying_distance : ℝ := distance_to_station * num_suitcases

/-- Represents the distance each person should carry a suitcase to ensure equal effort. -/
noncomputable def individual_carrying_distance : ℝ := total_carrying_distance / num_people

theorem equal_effort_distribution :
  individual_carrying_distance = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_effort_distribution_l40_4016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_jacket_price_is_90_l40_4086

-- Define the original price of a shirt
def original_shirt_price : ℚ := 60

-- Define the discount rate
def discount_rate : ℚ := 1/5

-- Define the number of shirts and jackets bought
def num_shirts : ℕ := 5
def num_jackets : ℕ := 10

-- Define the total amount paid
def total_paid : ℚ := 960

-- Theorem to prove
theorem original_jacket_price_is_90 :
  ∃ (original_jacket_price : ℚ),
    original_jacket_price * (1 - discount_rate) * num_jackets +
    original_shirt_price * (1 - discount_rate) * num_shirts = total_paid ∧
    original_jacket_price = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_jacket_price_is_90_l40_4086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_approx_16_67_l40_4094

def eastern_segments : ℕ := 6
def western_segments : ℕ := 8
def southern_segments : ℕ := 7
def northern_segments : ℕ := 9

def percentage_difference (compared_to western : ℕ) : ℚ :=
  (abs (western - compared_to) : ℚ) / (western : ℚ) * 100

def average_percentage_difference : ℚ :=
  (percentage_difference eastern_segments western_segments +
   percentage_difference southern_segments western_segments +
   percentage_difference northern_segments western_segments) / 3

theorem average_difference_approx_16_67 :
  (average_percentage_difference * 100).floor / 100 = 1666 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_approx_16_67_l40_4094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problems_l40_4046

theorem sqrt_problems :
  (Real.sqrt 27 + Real.sqrt 3 = 4 * Real.sqrt 3) ∧ 
  ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_problems_l40_4046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_middle_school_l40_4096

/-- Represents the number of students in each school -/
structure SchoolPopulation where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sample size from each school -/
structure SampleSize where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (x y z : ℕ) : Prop :=
  y - x = z - y

/-- The main theorem -/
theorem stratified_sampling_middle_school
  (pop : SchoolPopulation)
  (sample : SampleSize)
  (total_students : pop.a + pop.b + pop.c = 1200)
  (total_sample : sample.a + sample.b + sample.c = 120)
  (arithmetic_seq : isArithmeticSequence pop.a pop.b pop.c)
  (stratified : ∃ (k : ℚ), 
    sample.a = Int.floor (k * pop.a) ∧
    sample.b = Int.floor (k * pop.b) ∧
    sample.c = Int.floor (k * pop.c)) :
  sample.b = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_middle_school_l40_4096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hallway_length_l40_4035

/-- The length of a hallway where a father and son meet under specific conditions. -/
theorem hallway_length (s : ℝ) (d : ℝ) (t : ℝ) : d = 16 :=
  by
  -- Define the speed of the father as three times the son's speed
  let father_speed : ℝ := 3 * s

  -- Assume the meeting point is 12 meters from the father's end
  have meeting_point : d - 12 = 12 := by sorry

  -- The distance covered by the father is the product of his speed and time
  have father_distance : 3 * s * t = 12 := by sorry

  -- The distance covered by the son is the product of his speed and time
  have son_distance : s * t = d - 12 := by sorry

  -- Prove that the hallway length is 16 meters
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hallway_length_l40_4035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OEF_l40_4025

noncomputable section

open Real

variable (θ : ℝ)

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := y^2 + x^2/4 = 1

/-- The circle equation -/
def circle' (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point P on the ellipse -/
def P : ℝ × ℝ := (2 * cos θ, sin θ)

/-- The line equation containing the chord of tangency -/
def line (x y : ℝ) : Prop := 2 * x * cos θ + y * sin θ = 1

/-- Point E: x-intercept of the line -/
def E : ℝ × ℝ := (1 / (2 * cos θ), 0)

/-- Point F: y-intercept of the line -/
def F : ℝ × ℝ := (0, 1 / sin θ)

/-- Area of triangle OEF -/
def area_OEF : ℝ := (1/2) * |1 / sin (2*θ)|

theorem min_area_OEF :
  ∀ θ : ℝ, ellipse (P θ).1 (P θ).2 → circle' (P θ).1 (P θ).2 → 
  line θ (E θ).1 (E θ).2 → line θ (F θ).1 (F θ).2 →
  area_OEF θ ≥ (1/2) ∧ ∃ θ₀, area_OEF θ₀ = (1/2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OEF_l40_4025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l40_4070

/-- Given a train of length 300 meters that takes 51 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 550 meters. -/
theorem platform_length (train_length time_platform time_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_platform = 51)
  (h3 : time_pole = 18) :
  let speed := train_length / time_pole
  let platform_length := speed * time_platform - train_length
  platform_length = 550 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l40_4070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_theorem_l40_4066

/-- Ant's movement on a number line --/
noncomputable def AntMovement :=
  { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- Probability of moving right --/
noncomputable def prob_right : AntMovement := ⟨2/3, by norm_num⟩

/-- Probability of moving left --/
noncomputable def prob_left : AntMovement := ⟨1/3, by norm_num⟩

/-- Position of the ant after n seconds --/
def position (n : ℕ) : ℝ → Prop := sorry

/-- Probability of being at x after n seconds --/
noncomputable def prob_at (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Expected position after n seconds --/
noncomputable def expected_position (n : ℕ) : ℝ := sorry

theorem ant_movement_theorem :
  (prob_at 0 2 / (1 - prob_at (-1) 2 - prob_at (-2) 2) = 1/2) ∧
  (expected_position 4 = 4/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_theorem_l40_4066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_not_through_origin_l40_4069

/-- The power function f defined by a real parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * m^2 - 2 * m) * x^(3 * m)

/-- Theorem stating that the graph of f does not pass through the origin when m = -1/3 -/
theorem graph_not_through_origin :
  ∃ (x : ℝ), f (-1/3) x ≠ 0 :=
by
  -- We'll use x = 1 as our witness
  use 1
  -- Simplify the expression
  simp [f]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_not_through_origin_l40_4069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_a_l40_4059

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℕ, (1 : ℚ) / (x - 3) + (x + a) / (3 - x) = 1) ∧ 
    (-(a + 6) / (2 * 4) < 0)) ∧
  (∀ a : ℤ, 
    ((∃ x : ℕ, (1 : ℚ) / (x - 3) + (x + a) / (3 - x) = 1) ∧ 
    (-(a + 6) / (2 * 4) < 0)) → a ∈ S) ∧
  (S.sum id = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_a_l40_4059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_half_X_l40_4041

-- Define the variance operator D
noncomputable def D (X : Type) : ℝ := sorry

-- Axiom for the scaling property of variance
axiom D_scale {X : Type} (a : ℝ) : D (X) = a^2 * D X

-- Given condition
axiom D_X {X : Type} : D X = 2

-- Theorem to prove
theorem variance_half_X {X : Type} : D X = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_half_X_l40_4041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_optimal_solutions_l40_4067

-- Define the triangle vertices
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (-2, -2)
def C : ℝ × ℝ := (2, 0)

-- Define the function z
def z (a : ℝ) (p : ℝ × ℝ) : ℝ := p.2 - a * p.1

-- Define the set of vertices
def vertices : Set (ℝ × ℝ) := {A, B, C}

-- Define the theorem
theorem multiple_optimal_solutions :
  ∃ (a : ℝ), ∀ (p q : ℝ × ℝ),
    p ∈ vertices → q ∈ vertices → p ≠ q →
    (z a p = z a q ∧ ∀ (r : ℝ × ℝ), r ∈ vertices → z a r ≤ z a p) →
    a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_optimal_solutions_l40_4067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_example_l40_4011

theorem complex_division_example : (1 + 2*Complex.I) / (1 - Complex.I) = -1/2 + 3/2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_example_l40_4011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_negative_a_l40_4026

theorem power_of_four_negative_a (a : ℝ) (h : a * (Real.log 4 / Real.log 3) = 2) : 
  (4 : ℝ)^(-a) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_four_negative_a_l40_4026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l40_4098

theorem trig_identity (a : ℝ) : 
  (Real.sin (π - a) * Real.cos (2*π - a) * Real.sin ((3*π)/2 - a)) / (Real.sin (π/2 - a) * Real.sin (π + a)) = Real.cos a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l40_4098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersect_x_axis_once_l40_4092

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m + 2) * x + (1/2) * m + 1

theorem intersect_x_axis_once (m : ℝ) : 
  (∃! x, f m x = 0) ↔ m = -2 ∨ m = 0 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersect_x_axis_once_l40_4092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l40_4036

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 3 * x + 1

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (a ≠ 0 ∨ b ≠ 0) ∧
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p.1 = point.1 + t ∧ p.2 = f (point.1 + t)} →
      a * x + b * y + c = 0) ∧
    a = 2 ∧ b = 1 ∧ c = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l40_4036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l40_4032

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def line (x y : ℝ) : Prop := y = -x + 4

-- Define the intersection points A and B
def intersection_points (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2

-- Define the length of AB
noncomputable def chord_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define perpendicularity of OA and OB
def perpendicular (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

-- The main theorem
theorem parabola_intersection_theorem (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  intersection_points p A B →
  chord_length A B = 4 * Real.sqrt 10 →
  p = 2 ∧ perpendicular A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l40_4032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_X_between_1_and_2_l40_4082

-- Define the distribution function F
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ 2 then x / 2
  else 1

-- Define the random variable X
def X : Type := ℝ

-- Theorem statement
theorem probability_X_between_1_and_2 :
  (F 2 - F 1 : ℝ) = 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_X_between_1_and_2_l40_4082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_area_l40_4029

/-- The trajectory of point P -/
def trajectory (x y : ℝ) : Prop :=
  ((x + 3)^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

/-- The area enclosed by the trajectory -/
noncomputable def enclosed_area : ℝ := 16 * Real.pi

/-- Theorem stating that the area enclosed by the trajectory is 16π -/
theorem trajectory_area : 
  ∀ (x y : ℝ), trajectory x y → enclosed_area = 16 * Real.pi :=
by
  intros x y h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_area_l40_4029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l40_4083

/-- The curve C defined by parametric equations x = 4cos(φ) and y = 3sin(φ) -/
noncomputable def C : ℝ → ℝ × ℝ := fun φ ↦ (4 * Real.cos φ, 3 * Real.sin φ)

theorem curve_C_properties :
  ∀ φ : ℝ,
  let (x, y) := C φ
  -- The point satisfies the standard equation
  (x^2 / 16 + y^2 / 9 = 1) ∧
  -- The sum x + y is bounded between -5 and 5
  (-5 ≤ x + y) ∧ (x + y ≤ 5) := by
  sorry

#check curve_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l40_4083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_depth_of_specific_channel_l40_4043

/-- Represents a trapezoidal water channel -/
structure TrapezoidalChannel where
  topWidth : ℝ
  bottomWidth : ℝ
  area : ℝ

/-- Calculates the depth of a trapezoidal channel -/
noncomputable def channelDepth (channel : TrapezoidalChannel) : ℝ :=
  (2 * channel.area) / (channel.topWidth + channel.bottomWidth)

theorem depth_of_specific_channel :
  let channel := TrapezoidalChannel.mk 12 6 630
  channelDepth channel = 70 := by
  -- Unfold the definition of channelDepth
  unfold channelDepth
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num
  -- If norm_num doesn't complete the proof, you can use sorry
  -- sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_depth_of_specific_channel_l40_4043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l40_4030

-- Define arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sequence of odd-numbered terms
def odd_terms (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n ↦ a (2 * n - 1)

-- Define geometric mean
def is_geometric_mean (G a b : ℝ) : Prop :=
  G ^ 2 = a * b

theorem correct_statements_count :
  (∀ a b : ℕ → ℝ, is_arithmetic_seq a → is_arithmetic_seq b → is_arithmetic_seq (λ n ↦ a n + b n)) ∨
  (∀ a b : ℕ → ℝ, is_geometric_seq a → is_geometric_seq b → is_geometric_seq (λ n ↦ a n + b n)) ∨
  (∀ a : ℕ → ℝ, ∀ d : ℝ, is_arithmetic_seq a → is_arithmetic_seq (odd_terms a)) ∨
  (∀ G a b : ℝ, is_geometric_mean G a b ↔ G ^ 2 = a * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_count_l40_4030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ticket_percentage_l40_4087

/-- Represents the percentage of motorists who exceed the speed limit -/
noncomputable def exceed_limit_percent : ℚ := 20

/-- Represents the percentage of motorists who receive speeding tickets -/
noncomputable def receive_ticket_percent : ℚ := 10

/-- Represents the percentage of motorists who exceed the speed limit but do not receive tickets -/
noncomputable def no_ticket_percent : ℚ := (exceed_limit_percent - receive_ticket_percent) / exceed_limit_percent * 100

theorem no_ticket_percentage :
  no_ticket_percent = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_ticket_percentage_l40_4087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l40_4048

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (Real.pi / 2 - x) * Real.cos (x + Real.pi / 3) + Real.sqrt 3 / 2

theorem f_properties :
  let period : ℝ := Real.pi
  let monotonic_decrease (k : ℤ) : Set ℝ := Set.Icc (k * Real.pi + Real.pi / 12) (k * Real.pi + 7 * Real.pi / 12)
  let range : Set ℝ := Set.Icc (-Real.sqrt 3 / 2) 1
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ monotonic_decrease k, ∀ y ∈ monotonic_decrease k, x < y → f y < f x) ∧
  Set.range (fun x ↦ f x) ∩ Set.Icc 0 (Real.pi / 2) = range :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l40_4048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_local_odd_function_l40_4047

-- Define a local odd function
def is_local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (-x₀) = -f x₀

-- Define the specific function f(x) = -a2^x - 4
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * (2 ^ x) - 4

-- State the theorem
theorem range_of_a_for_local_odd_function :
  ∀ a : ℝ, is_local_odd_function (f a) → a ∈ Set.Icc (-4) 0 ∧ a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_local_odd_function_l40_4047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_5_on_1_2_l40_4015

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f
  | n + 1 => λ x => f (f_n n x)

-- Theorem statement
theorem max_f_5_on_1_2 :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 1 2 → f_n 5 y ≤ f_n 5 x ∧
  f_n 5 x = 3^32 - 1 := by
  sorry

#check max_f_5_on_1_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_f_5_on_1_2_l40_4015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l40_4065

/-- The equation has no solutions in the closed interval [0, 2π] -/
theorem no_solutions_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi →
    Real.sin (Real.pi * Real.cos x) ≠ Real.cos (Real.pi * Real.sin (x - Real.pi / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l40_4065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothesTransportEqualsSavings_l40_4073

/-- Represents Mr. Yadav's monthly finances --/
structure MonthlyFinances where
  salary : ℚ
  consumableRate : ℚ
  entertainmentRate : ℚ
  housingRate : ℚ
  clothesTransportRate : ℚ
  yearlySavings : ℚ

/-- Calculates the monthly amount spent on clothes and transport --/
def clothesTransportExpense (mf : MonthlyFinances) : ℚ :=
  mf.salary * (1 - mf.consumableRate) * (1 - mf.entertainmentRate) * (1 - mf.housingRate) * mf.clothesTransportRate

/-- Calculates the monthly savings --/
def monthlySavings (mf : MonthlyFinances) : ℚ :=
  mf.yearlySavings / 12

/-- Theorem stating that the monthly amount spent on clothes and transport
    is equal to the monthly savings --/
theorem clothesTransportEqualsSavings (mf : MonthlyFinances) 
    (h1 : mf.consumableRate = 6/10)
    (h2 : mf.entertainmentRate = 2/10)
    (h3 : mf.housingRate = 3/10)
    (h4 : mf.clothesTransportRate = 1/2)
    (h5 : mf.yearlySavings = 48456) :
    clothesTransportExpense mf = monthlySavings mf := by
  sorry

#eval clothesTransportExpense { salary := 36053.57, consumableRate := 0.6, entertainmentRate := 0.2, housingRate := 0.3, clothesTransportRate := 0.5, yearlySavings := 48456 }
#eval monthlySavings { salary := 36053.57, consumableRate := 0.6, entertainmentRate := 0.2, housingRate := 0.3, clothesTransportRate := 0.5, yearlySavings := 48456 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothesTransportEqualsSavings_l40_4073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_y_l40_4091

noncomputable def y (x : ℝ) := Real.sqrt (Real.sin x) + Real.sqrt (Real.cos x - 1/2)

theorem domain_of_y :
  ∀ x : ℝ, (∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + 2 * k * Real.pi) ↔
    (Real.sin x ≥ 0 ∧ Real.cos x - 1/2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_y_l40_4091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_N_l40_4057

-- Define the points
def F : ℝ × ℝ := (1, 0)
def M (x₀ : ℝ) : ℝ × ℝ := (x₀, 0)
def P (y₀ : ℝ) : ℝ × ℝ := (0, y₀)
def N (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define the vector operations
def vec (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := dot v w = 0

-- State the theorem
theorem trajectory_of_N (x₀ y₀ x y : ℝ) :
  perpendicular (vec (P y₀) (M x₀)) (vec (P y₀) F) ∧
  vec (M x₀) (N x y) = scale 2 (vec (P y₀) (M x₀)) →
  y^2 = 4*x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_N_l40_4057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_sum_sum_reciprocals_pow_l40_4068

/-- Geometric progression with n terms, first term b, and common ratio q -/
structure GeometricProgression where
  n : ℕ
  b : ℝ
  q : ℝ

/-- Product of terms in a geometric progression -/
noncomputable def product (gp : GeometricProgression) : ℝ :=
  gp.b^gp.n * gp.q^(gp.n * (gp.n - 1) / 2)

/-- Sum of terms in a geometric progression -/
noncomputable def sum (gp : GeometricProgression) : ℝ :=
  gp.b * (1 - gp.q^gp.n) / (1 - gp.q)

/-- Sum of reciprocals of terms in a geometric progression -/
noncomputable def sumReciprocals (gp : GeometricProgression) : ℝ :=
  (gp.q^gp.n - 1) / (gp.b * (gp.q - 1))

theorem product_equals_sum_sum_reciprocals_pow (gp : GeometricProgression) :
  product gp = (sum gp * sumReciprocals gp)^(gp.n / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_sum_sum_reciprocals_pow_l40_4068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l40_4050

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity √3, prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eccentricity : Real.sqrt (1 + (b/a)^2) = Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), (x^2/a^2 - y^2/b^2 = 1) → (y = k*x ∨ y = -k*x)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l40_4050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_shifted_l40_4054

open Set Real

noncomputable def f (x : ℝ) := tan (π / 4 - x)

theorem domain_of_tan_shifted :
  {x : ℝ | ¬(∃ k : ℤ, x = k * π + 3 * π / 4)} = {x : ℝ | f x ≠ 0 ∨ f x ≠ 0⁻¹} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_shifted_l40_4054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_sine_solutions_l40_4038

theorem cosine_and_sine_solutions :
  (Real.cos (-π/3) = 1/2) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2*π → (Real.sin x = Real.sqrt 3/2 ↔ x = π/3 ∨ x = 2*π/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_sine_solutions_l40_4038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_ran_24_laps_l40_4037

/-- Represents the race scenario --/
structure RaceScenario where
  duration : ℕ  -- Race duration in minutes
  lap_distance : ℕ  -- Distance of one lap in meters
  award_per_100m : ℚ  -- Award in dollars per 100 meters
  avg_earnings_per_min : ℚ  -- Average earnings in dollars per minute

/-- Calculates the number of laps run by the winner --/
def laps_run (scenario : RaceScenario) : ℕ :=
  let total_earnings := scenario.avg_earnings_per_min * scenario.duration
  let total_100m := total_earnings / scenario.award_per_100m
  (total_100m.num / total_100m.den).natAbs

/-- Theorem stating that given the specific race conditions, the winner ran 24 laps --/
theorem winner_ran_24_laps :
  let scenario : RaceScenario := {
    duration := 12,
    lap_distance := 100,
    award_per_100m := 7/2,
    avg_earnings_per_min := 7
  }
  laps_run scenario = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winner_ran_24_laps_l40_4037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l40_4009

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ 0 < t.B ∧ 0 < t.C ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a * Real.sin t.B = t.b * Real.sin t.A ∧
  t.b * Real.sin t.C = t.c * Real.sin t.B ∧
  t.c * Real.sin t.A = t.a * Real.sin t.C ∧
  2 * t.a * Real.cos t.B = 2 * t.c + t.b ∧
  1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3

-- Theorem statement
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.A = 2 * Real.pi / 3 ∧
  (t.b - t.c = 2 → t.a + t.b + t.c = 6 + 2 * Real.sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l40_4009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptote_slope_l40_4044

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.b / h.a

theorem hyperbola_eccentricity_from_asymptote_slope (h : Hyperbola) 
  (h_slope : asymptote_slope h = 1/2) : eccentricity h = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptote_slope_l40_4044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_property_l40_4062

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 4 / (Real.cos θ^2 + 4 * Real.sin θ^2)

-- Define the perpendicularity condition
def perpendicular (θ₁ θ₂ : ℝ) : Prop := θ₂ = θ₁ + Real.pi/2 ∨ θ₂ = θ₁ - Real.pi/2

theorem curve_C_property (θ₁ θ₂ ρ₁ ρ₂ : ℝ) :
  C ρ₁ θ₁ → C ρ₂ θ₂ → perpendicular θ₁ θ₂ →
  1/ρ₁^2 + 1/ρ₂^2 = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_property_l40_4062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_specific_matrix_l40_4089

open Matrix

theorem determinant_of_specific_matrix (x y z : ℝ) :
  det !![1, x + z, y;
         1, x + y + z, y + z;
         1, x + z, x + y + z] = x * y + y * z + x * z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_specific_matrix_l40_4089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l40_4034

theorem max_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 3 * b < 60) :
  a * b * (60 - 5 * a - 3 * b) ≤ 1600 / 3 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 5 * a₀ + 3 * b₀ < 60 ∧ a₀ * b₀ * (60 - 5 * a₀ - 3 * b₀) = 1600 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l40_4034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l40_4055

/-- Given two vectors a and b in ℝ², prove that if λa + b is parallel to a - 2b, then λ = -1/2 -/
theorem parallel_vectors_lambda (a b : ℝ × ℝ) (h : a = (1, 1) ∧ b = (2, -1)) :
  ∃ l : ℝ, (∃ k : ℝ, k ≠ 0 ∧ (l * a.1 + b.1, l * a.2 + b.2) = k • (a.1 - 2 * b.1, a.2 - 2 * b.2)) →
  l = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l40_4055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_tangent_intersection_points_unique_intersection_l40_4022

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x + y^2 - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define what it means for two circles to be tangent
def are_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  (∀ (u v : ℝ), (u, v) ≠ (x, y) → ¬(C1 u v ∧ C2 u v))

-- Theorem statement
theorem circles_are_tangent :
  are_tangent circle1 circle2 := by
  sorry

-- Helper theorem to show the circles intersect at (0, 2) and (0, -2)
theorem intersection_points :
  ∃ (x y : ℝ), (x = 0 ∧ (y = 2 ∨ y = -2)) ∧ circle1 x y ∧ circle2 x y := by
  sorry

-- Helper theorem to show that (0, 2) is the only intersection point
theorem unique_intersection :
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → (x = 0 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_tangent_intersection_points_unique_intersection_l40_4022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l40_4093

theorem sine_difference_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α = 2 * Real.sqrt 2 / 3)
  (h4 : Real.cos (α + β) = -1 / 3) :
  Real.sin (α - β) = 10 * Real.sqrt 2 / 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_value_l40_4093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l40_4023

/-- Represents an investment scheme with an initial investment and a yield rate. -/
structure Scheme where
  initial_investment : ℝ
  yield_rate : ℝ

/-- Calculates the total amount after a year for a given scheme. -/
def total_after_year (s : Scheme) : ℝ :=
  s.initial_investment * (1 + s.yield_rate)

/-- Theorem stating the difference between two investment schemes after a year. -/
theorem investment_difference (scheme_a scheme_b : Scheme)
  (h1 : scheme_a.initial_investment = 300)
  (h2 : scheme_b.initial_investment = 200)
  (h3 : scheme_a.yield_rate = 0.3)
  (h4 : scheme_b.yield_rate = 0.5) :
  total_after_year scheme_a - total_after_year scheme_b = 90 := by
  sorry

-- Remove the #eval statement as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l40_4023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_twelve_dividing_30_factorial_l40_4075

theorem largest_power_of_twelve_dividing_30_factorial : 
  ∃ n : ℕ, n = 13 ∧ 
  (∀ m : ℕ, 12^m ∣ Nat.factorial 30 → m ≤ n) ∧
  (12^n ∣ Nat.factorial 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_twelve_dividing_30_factorial_l40_4075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_recursive_relation_l40_4056

open BigOperators

def f (n : ℕ) : ℕ := ∑ i in Finset.range (2 * n + 1), i ^ 2

theorem f_recursive_relation (k : ℕ) : 
  f (k + 1) = f k + (2 * k + 1) ^ 2 + (2 * k + 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_recursive_relation_l40_4056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_start_point_l40_4039

/-- The distance between points A and B in meters -/
def total_distance : ℝ := 3000

/-- Ada's remaining distance when Luisa finishes in meters -/
def ada_remaining : ℝ := 120

/-- Luisa's speed in meters per second -/
noncomputable def luisa_speed : ℝ := 1  -- Arbitrary non-zero value

/-- Ada's speed in meters per second -/
noncomputable def ada_speed : ℝ := (total_distance - ada_remaining) / (total_distance / luisa_speed)

/-- The additional distance Luisa needs to run to finish with Ada -/
noncomputable def luisa_additional_distance : ℝ := 
  total_distance * (total_distance / (total_distance - ada_remaining)) - total_distance

theorem luisa_start_point (t : ℝ) :
  t * ada_speed = (t * luisa_speed + luisa_additional_distance) →
  luisa_additional_distance = 125 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luisa_start_point_l40_4039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l40_4004

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths
  (sum_angles : A + B + C = Real.pi)  -- Sum of angles in a triangle is π
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)  -- Side lengths are positive

-- Define the condition given in the problem
def condition (t : Triangle) : Prop :=
  (t.a^2 + t.b^2) * Real.sin (t.A - t.B) = (t.a^2 - t.b^2) * Real.sin (t.A + t.B)

-- Define isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define right-angled triangle
def isRightAngled (t : Triangle) : Prop :=
  t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

-- Theorem statement
theorem triangle_shape (t : Triangle) :
  condition t → isIsosceles t ∨ isRightAngled t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l40_4004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l40_4088

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - x^2) / Real.log (1/2)

-- State the theorem
theorem f_monotone_increasing_interval :
  ∃ (a b : ℝ), a = 1 ∧ b = 2 ∧
  (∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ < f x₂) ∧
  (∀ x, 0 < x → x < 2 → (x < a ∨ b ≤ x → ¬(∀ y, x < y → y < 2 → f x < f y))) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_l40_4088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_value_l40_4074

theorem sin_two_theta_value (θ : ℝ) (h : Real.cos θ + Real.sin θ = 7/5) : 
  Real.sin (2 * θ) = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_theta_value_l40_4074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_450_deg_undefined_l40_4081

open Real

-- Define the secant function
noncomputable def sec (θ : ℝ) : ℝ := 1 / cos θ

-- Define the degree to radian conversion
noncomputable def deg_to_rad (θ : ℝ) : ℝ := θ * (π / 180)

-- Theorem statement
theorem sec_450_deg_undefined :
  ¬∃ (x : ℝ), sec (deg_to_rad (-450)) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_450_deg_undefined_l40_4081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequences_with_01_blocks_l40_4013

/-- The number of binary sequences of length n containing the block 01 exactly m times. -/
def number_of_binary_sequences (n m : ℕ) : ℕ :=
  Nat.choose (n - m) m

/-- 
Given non-negative integers n and m where n ≥ 2m, the number of binary sequences 
of length n containing the block 01 exactly m times is equal to (n - m choose m).
-/
theorem binary_sequences_with_01_blocks (n m : ℕ) (h : n ≥ 2 * m) : 
  number_of_binary_sequences n m = Nat.choose (n - m) m := by
  -- The proof is trivial because of how we defined number_of_binary_sequences
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequences_with_01_blocks_l40_4013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangements_count_l40_4090

/-- The number of ways to distribute n items into k groups with at most m items per group -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to arrange 7 rings out of 10 on 4 fingers with at most 2 rings per finger -/
def ring_arrangements : ℕ :=
  Nat.choose 10 7 * Nat.factorial 7 * distribute 7 4 2

theorem ring_arrangements_count : ring_arrangements = 24192000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_arrangements_count_l40_4090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l40_4027

/-- The function f(x) = ln x - kx + 1 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * x + 1

/-- Sum of logarithmic terms -/
noncomputable def log_sum (n : ℕ) : ℝ :=
  (Finset.range (n - 1)).sum (λ i => Real.log (i + 2) / ((i + 2)^2 - 1))

theorem problem_statement :
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f k x ≤ 0) ↔ k ≥ 1) ∧
  (∀ n : ℕ, n > 1 → log_sum n + (1 + 1 / n)^n < (n^2 + n + 10) / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l40_4027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l40_4008

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := Real.exp (x - 1) + 1

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem tangent_line_through_origin :
  ∃ x₀ : ℝ,
    -- The tangent line passes through a point on the curve
    f x₀ = tangent_line x₀ ∧
    -- The slope of the tangent line equals the derivative at x₀
    f' x₀ = 2 ∧
    -- The tangent line passes through the origin
    tangent_line 0 = 0 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l40_4008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l40_4045

theorem sin_double_angle_special_case (α : Real) 
  (h1 : Real.sin α = 1/5) 
  (h2 : α ∈ Set.Ioo (Real.pi/2) Real.pi) : 
  Real.sin (2*α) = -(4*Real.sqrt 6)/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l40_4045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_cube_gt_3n_floor_a_9n_cube_l40_4085

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a (n + 1) + 1 / (a (n + 1))^2

theorem a_cube_gt_3n (n : ℕ) (h : n ≥ 2) : (a n)^3 > 3 * ↑n := by sorry

theorem floor_a_9n_cube (n : ℕ) (h : n ≥ 4) : ⌊a (9 * n^3)⌋ = 3 * n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_cube_gt_3n_floor_a_9n_cube_l40_4085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l40_4080

noncomputable section

open Real

/-- The function g(x) defined in the problem -/
def g (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) + cos (ω * x + π / 6)

/-- Symmetry condition about the point (2π, 0) -/
def symmetric_about_2π (ω : ℝ) : Prop :=
  ∀ x, g ω (2 * π - x) = g ω (2 * π + x)

/-- Monotonicity condition in the interval [-π/3, π/6] -/
def monotonous_in_interval (ω : ℝ) : Prop :=
  Monotone (fun x ↦ g ω x) ∨ StrictMonoOn (fun x ↦ g ω x) (Set.Icc (-π/3) (π/6))

/-- Main theorem -/
theorem g_properties (ω : ℝ) (h_pos : ω > 0) 
    (h_sym : symmetric_about_2π ω) (h_mono : monotonous_in_interval ω) :
    ω = 1/3 ∨ ω = 5/6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l40_4080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l40_4010

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 : ℝ)^x - m else x^2 - 3*m*x + 2*m^2

-- Define the set of m values
def M : Set ℝ := Set.Icc (1/2) 1 ∪ Set.Ici 6

-- Theorem statement
theorem f_has_two_zeros (m : ℝ) :
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ ∀ z, f m z = 0 → z = x ∨ z = y) ↔ m ∈ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_zeros_l40_4010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l40_4058

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 16

/-- The midpoint of a chord -/
def chord_midpoint : ℝ × ℝ := (1, -1)

/-- A point lies inside the ellipse -/
def inside_ellipse (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + 4*y^2 < 16

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := x - 4*y - 5 = 0

theorem chord_equation :
  inside_ellipse chord_midpoint →
  ∃ (p q : ℝ × ℝ),
    ellipse p.1 p.2 ∧
    ellipse q.1 q.2 ∧
    ((p.1 + q.1) / 2, (p.2 + q.2) / 2) = chord_midpoint ∧
    ∀ (x y : ℝ), line_equation x y ↔ ∃ t : ℝ, (x, y) = (1 - t, -1 - t/4) ∨ (x, y) = (1 + t, -1 + t/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l40_4058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_mod_9_eq_7_l40_4040

def S : ℕ := (List.range 27).map (fun k => Nat.choose 27 (k + 1)) |>.sum

theorem S_mod_9_eq_7 : S % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_mod_9_eq_7_l40_4040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l40_4007

/-- Regular triangular pyramid with base side length a and lateral edge angle 60° --/
structure RegularTriangularPyramid where
  a : ℝ  -- base side length
  h : a > 0  -- side length is positive

/-- Volume of a regular triangular pyramid --/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ :=
  (p.a^3 * Real.sqrt 3) / 12

/-- Theorem: The volume of a regular triangular pyramid with base side length a
    and lateral edge forming a 60° angle with the base plane is (a^3 * √3) / 12 --/
theorem regular_triangular_pyramid_volume (p : RegularTriangularPyramid) :
  volume p = (p.a^3 * Real.sqrt 3) / 12 := by
  -- Proof steps would go here
  sorry

#check regular_triangular_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l40_4007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_expression_sum_of_coefficients_main_result_l40_4084

theorem sum_of_coefficients_expanded_expression (d : ℝ) : 
  let expanded := -(4 - d) * (2*d + 3*(4 - d))
  expanded = -d^2 + 16*d - 48 := by
  sorry

theorem sum_of_coefficients (a b c : ℝ) :
  (a + b + c : ℝ) = -33 → a + b + c = -33 := by
  intro h
  exact h

theorem main_result (d : ℝ) :
  let expanded := -(4 - d) * (2*d + 3*(4 - d))
  expanded = -d^2 + 16*d - 48 ∧ 
  (-1 : ℝ) + 16 + (-48) = -33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_expression_sum_of_coefficients_main_result_l40_4084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_l40_4024

/-- The volume of a cuboid with edges 2 cm, 5 cm, and 8 cm is 80 cubic centimeters. -/
theorem cuboid_volume : ∃ V : ℝ,
  ∃ (l w h : ℝ),
    l = 2 ∧ w = 5 ∧ h = 8 ∧
    V = l * w * h ∧
    V = 80 := by
  -- Proof goes here
  sorry

#check cuboid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_l40_4024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_intersection_l40_4097

-- Define the ellipsoid G
def ellipsoid (x y z : ℝ) : Prop :=
  x^2/16 + y^2/9 + z^2/4 = 1

-- Define the sphere
def sphere (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 4

-- Define the volume of the ellipsoid G
noncomputable def volume_G : ℝ := 32 * Real.pi

-- Define the volume of the sphere
noncomputable def volume_sphere : ℝ := (32/3) * Real.pi

-- Define the volume of region g (the intersection)
noncomputable def volume_g : ℝ := volume_G - volume_sphere

-- The probability theorem
theorem probability_in_intersection :
  (volume_g / volume_G) = 2/3 := by
  -- Expand the definitions
  unfold volume_g volume_G volume_sphere
  -- Perform algebraic simplification
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_intersection_l40_4097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investments_properties_l40_4051

/-- Represents an investment with initial amount p and return amount x -/
structure Investment where
  p : ℝ
  x : ℝ

/-- Calculates the distance between two investments -/
noncomputable def distance (i1 i2 : Investment) : ℝ :=
  Real.sqrt ((i1.x - i2.x)^2 + (i1.p - i2.p)^2)

/-- Theorem stating the properties of the optimal investments -/
theorem optimal_investments_properties :
  ∃ (i1 i2 : Investment),
    i1.p > 0 ∧ i2.p > 0 ∧
    4 * i1.x - 3 * i1.p - 44 = 0 ∧
    i2.p^2 - 12 * i2.p + i2.x^2 - 8 * i2.x + 43 = 0 ∧
    (∀ (j1 j2 : Investment),
      j1.p > 0 ∧ j2.p > 0 ∧
      4 * j1.x - 3 * j1.p - 44 = 0 ∧
      j2.p^2 - 12 * j2.p + j2.x^2 - 8 * j2.x + 43 = 0 →
      distance i1 i2 ≤ distance j1 j2) ∧
    distance i1 i2 = 6.2 ∧
    i1.x + i2.x - i1.p - i2.p = 13.08 := by
  sorry

#check optimal_investments_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_investments_properties_l40_4051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_remaining_average_height_l40_4005

/-- Given a class of students with known average heights, calculate the average height of the remaining students. -/
theorem calculate_remaining_average_height
  (total_students : ℕ)
  (known_group_size : ℕ)
  (known_group_average : ℚ)
  (class_average : ℚ)
  (h_total : total_students = 50)
  (h_known : known_group_size = 40)
  (h_known_avg : known_group_average = 169)
  (h_class_avg : class_average = 168.6)
  : (total_students * class_average - known_group_size * known_group_average) / (total_students - known_group_size) = 167 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_remaining_average_height_l40_4005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suit_price_after_increase_and_discount_l40_4061

theorem suit_price_after_increase_and_discount 
  (original_price : ℝ) 
  (increase_percentage : ℝ) 
  (discount_percentage : ℝ) : 
  original_price = 160 → 
  increase_percentage = 0.25 → 
  discount_percentage = 0.25 → 
  (original_price * (1 + increase_percentage) * (1 - discount_percentage)) = 150 := by
  sorry

#check suit_price_after_increase_and_discount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_suit_price_after_increase_and_discount_l40_4061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_ticket_price_l40_4063

/-- Calculates the new ticket price after a price reduction -/
theorem new_ticket_price (initial_price spectator_increase revenue_increase : ℝ) :
  initial_price = 400 →
  spectator_increase = 0.25 →
  revenue_increase = 0.125 →
  (1 + spectator_increase) * ((1 + revenue_increase) * initial_price) / (1 + spectator_increase) = 360 := by
  intros h1 h2 h3
  sorry

#eval Float.toString ((1 + 0.25) * ((1 + 0.125) * 400) / (1 + 0.25))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_ticket_price_l40_4063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_undetermined_l40_4049

/-- Represents a student in the circle -/
inductive Student : Type
| Ana : Student
| Ben : Student
| Cal : Student
| Dee : Student
| Eli : Student
| Fay : Student
| Gus : Student
| Hal : Student

/-- The initial order of students in the circle -/
def initialOrder : List Student :=
  [Student.Ana, Student.Ben, Student.Cal, Student.Dee, 
   Student.Eli, Student.Fay, Student.Gus, Student.Hal]

/-- Checks if a number contains 8 as a digit or is a multiple of 8 -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 8 = 0 || n.repr.contains '8'

/-- The elimination process -/
def eliminationProcess (students : List Student) : Option Student :=
  sorry

/-- The theorem stating that the last remaining student is undetermined -/
theorem last_student_undetermined :
  ∀ (result : Option Student), eliminationProcess initialOrder = result →
  (¬ ∃ (s : Student), result = some s) ∧ result ≠ none :=
by
  sorry

#eval isEliminationNumber 18
#eval isEliminationNumber 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_student_undetermined_l40_4049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_eq_v_poly_sum_of_coefficients_is_seven_l40_4077

/-- Sequence v_n defined by the given recurrence relation -/
def v : ℕ → ℝ
  | 0 => 7  -- Add base case for 0
  | 1 => 7
  | (n + 1) => v n + (5 + 3 * (n - 1))

/-- Polynomial representation of v_n -/
def v_poly (n : ℕ) : ℝ := 1.5 * n^2 + 0.5 * n + 5

/-- Theorem stating that v_n equals its polynomial representation -/
theorem v_eq_v_poly : ∀ n : ℕ, v n = v_poly n := by sorry

/-- Theorem proving the sum of coefficients is 7 -/
theorem sum_of_coefficients_is_seven :
  ∃ (a b c : ℝ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_eq_v_poly_sum_of_coefficients_is_seven_l40_4077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l40_4020

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2 - x) * (x + 1) / ((x + 1) * (x + 3))

-- Define the domain of the function
def domain (x : ℝ) : Prop := x ≠ -1 ∧ x ≠ -3

-- Define the range of the function
def range (y : ℝ) : Prop := y < -1 ∨ (-1 < y ∧ y < 3/2) ∨ y > 3/2

-- Theorem statement
theorem function_range : 
  ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ range y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l40_4020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_scalar_multiplication_l40_4033

/-- Prove that the sum of (4, -2) and 2 times (-5, 8) equals (-6, 14) -/
theorem vector_addition_and_scalar_multiplication :
  (⟨4, -2⟩ : ℝ × ℝ) + 2 • (⟨-5, 8⟩ : ℝ × ℝ) = (⟨-6, 14⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_and_scalar_multiplication_l40_4033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l40_4002

/-- Parabola with equation y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Focus of the parabola y^2 = 8x -/
def focus : ℝ × ℝ := (2, 0)

/-- Directrix of the parabola y^2 = 8x -/
def directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -2}

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_range (x₀ y₀ : ℝ) :
  (x₀, y₀) ∈ Parabola →
  (∃ p ∈ directrix, distance p focus = distance (x₀, y₀) focus) →
  x₀ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l40_4002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_for_f_f_x_eq_6_l40_4052

noncomputable def f (x : ℝ) : ℝ := if x ≥ -2 then x^2 - 4 else x + 4

theorem four_solutions_for_f_f_x_eq_6 :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 6 := by
  sorry

#check four_solutions_for_f_f_x_eq_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_for_f_f_x_eq_6_l40_4052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_translation_l40_4060

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := sin (2 * x)

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 3)

-- Theorem statement
theorem sin_graph_translation :
  ∀ x : ℝ, g x = f (x - π / 6) :=
by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_translation_l40_4060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_triangle_property_l40_4053

/-- A function that checks if a set of three integers forms a triangle -/
def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if all 3-element subsets of an 11-element set satisfy the triangle property -/
def all_subsets_triangle (s : Finset ℕ) : Prop :=
  s.card = 11 → ∀ a b c, a ∈ s → b ∈ s → c ∈ s → a < b → b < c → is_triangle a b c

/-- The main theorem: 321 is the largest n such that all 11-element subsets of {3,...,n} have the triangle property -/
theorem largest_n_with_triangle_property :
  ∀ n : ℕ, n > 321 →
    ∃ s : Finset ℕ, s ⊆ Finset.Icc 3 n ∧ ¬(all_subsets_triangle s) :=
by
  sorry

#check largest_n_with_triangle_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_triangle_property_l40_4053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l40_4072

theorem chord_length (r d : ℝ) (hr : r = 4) (hd : d = 3) :
  2 * Real.sqrt (r^2 - d^2) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l40_4072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earliest_time_100_degrees_l40_4000

def temperature (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

theorem earliest_time_100_degrees :
  ∃ t : ℝ, t ≥ 0 ∧ t ≤ 24 ∧ temperature t = 100 ∧
  ∀ t' : ℝ, t' ≥ 0 ∧ t' < t → temperature t' ≠ 100 ∧
  t = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earliest_time_100_degrees_l40_4000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_l40_4099

/-- A function f is symmetric about the origin if f(-x) = -f(x) for all x -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The cosine function with amplitude A, angular frequency ω, and phase φ -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.cos (ω * x + φ)

/-- The main theorem: f is symmetric about the origin iff φ = kπ + π/2 for some integer k -/
theorem cosine_symmetry (A ω φ : ℝ) (h_A : A ≠ 0) :
  SymmetricAboutOrigin (f A ω φ) ↔ ∃ k : ℤ, φ = k * Real.pi + Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_l40_4099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_angle_problem_l40_4003

/-- The measure of the interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

theorem polygon_angle_problem (P₁ P₂ : ℕ) (h₁ : P₁ = 16) (h₂ : P₂ = 4) 
  (h₃ : interior_angle P₂ = 2 * interior_angle P₁ - 10) : 
  interior_angle P₁ = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_angle_problem_l40_4003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l40_4021

/-- The longest side of a triangle with vertices at (1,3), (4,7), and (7,3) has a length of 6 units -/
theorem longest_side_of_triangle : ∃ (longest : ℝ), longest = 6 := by
  -- Define the vertices of the triangle
  let v1 : ℝ × ℝ := (1, 3)
  let v2 : ℝ × ℝ := (4, 7)
  let v3 : ℝ × ℝ := (7, 3)

  -- Define the distance function between two points
  let distance (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

  -- Calculate the lengths of all sides
  let side1 : ℝ := distance v1 v2
  let side2 : ℝ := distance v1 v3
  let side3 : ℝ := distance v2 v3

  -- The longest side is the maximum of all sides
  let longest_side : ℝ := max (max side1 side2) side3

  -- Prove that the longest side is 6 units
  have h : longest_side = 6 := by
    sorry

  exact ⟨longest_side, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_triangle_l40_4021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l40_4012

/-- The parabola C defined by x^2 = 4ay passing through (-2, 1) has directrix y = -1 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, x^2 = 4*a*y) →  -- Equation of the parabola
  ((-2)^2 = 4*a*1) →     -- Parabola passes through (-2, 1)
  (∀ x : ℝ, x^2 = 4*a*(-1)) -- Directrix equation
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l40_4012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cindy_cycling_speed_l40_4042

/-- The speed at which Cindy must cycle to arrive home at 5:00 PM -/
noncomputable def required_speed (d : ℝ) (t₁ t₂ t₃ : ℝ) : ℝ :=
  d / t₃

/-- The theorem stating the required speed given the conditions -/
theorem cindy_cycling_speed 
  (d : ℝ) -- distance from school to home
  (t₁ : ℝ) -- time taken at 20 km/h
  (t₂ : ℝ) -- time taken at 10 km/h
  (t₃ : ℝ) -- time to arrive at 5:00 PM
  (h₁ : d = 20 * t₁) -- distance equation at 20 km/h
  (h₂ : d = 10 * t₂) -- distance equation at 10 km/h
  (h₃ : t₂ = t₁ + 3/4) -- time difference between 10 km/h and 20 km/h
  (h₄ : t₃ = t₁ + 1/2) -- time difference to arrive at 5:00 PM
  : required_speed d t₁ t₂ t₃ = 12 := by
  sorry

#check cindy_cycling_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cindy_cycling_speed_l40_4042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_distance_l40_4079

/-- Helper function to calculate the distance between two points -/
noncomputable def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The total distance of a light ray reflecting off two perpendicular mirrors -/
theorem light_ray_distance (x y a b : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0) :
  ∃ d : ℝ, d = Real.sqrt ((a + x)^2 + (b + y)^2) ∧ 
  d = distance_between_points (x, y) (-a, -b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_ray_distance_l40_4079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l40_4006

-- Define the points
noncomputable def A : ℝ × ℝ := (3, 1)
noncomputable def B : ℝ × ℝ := (-4, 6)
noncomputable def C : ℝ × ℝ := (-1/2, 7/2)

-- Define the line equation
def line (x y a : ℝ) : ℝ := 3*x - 2*y + a

-- Define symmetry condition
def symmetric (p1 p2 c : ℝ × ℝ) : Prop :=
  p2.1 = 2*c.1 - p1.1 ∧ p2.2 = 2*c.2 - p1.2

-- Define same side condition
def same_side (p1 p2 : ℝ × ℝ) (a : ℝ) : Prop :=
  (line p1.1 p1.2 a) * (line p2.1 p2.2 a) > 0

-- Theorem statement
theorem a_range (a : ℝ) : 
  symmetric A B C → same_side A B a → 
  (a < -7 ∨ a > 24) ∧ ∀ x, (x < -7 ∨ x > 24) → ∃ y, same_side A B y ∧ y = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l40_4006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l40_4001

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2

-- State the theorem
theorem f_properties (a : ℝ) :
  -- Part 1: Monotonicity
  (∀ x > 0, ∀ y > 0, a ≤ 0 → x < y → f a x < f a y) ∧
  (a > 0 → ∀ x > 0, ∀ y > 0, 
    (x < y ∧ y < Real.sqrt (1/a) → f a x < f a y) ∧
    (Real.sqrt (1/a) < x ∧ x < y → f a y < f a x)) ∧
  -- Part 2: Minimum integer value of a
  (∀ x > 0, f 2 x ≤ x - 1) ∧
  (∀ b : ℤ, (∀ x > 0, f (↑b) x ≤ (↑b - 1) * x - 1) → 2 ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l40_4001
