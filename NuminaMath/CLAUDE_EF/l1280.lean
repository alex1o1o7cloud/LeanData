import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_hit_l1280_128022

/-- The probability that at least one person hits the target -/
def probability_at_least_one_hit (P₁ P₂ : ℝ) : ℝ :=
  1 - (1 - P₁) * (1 - P₂)

/-- The probability that at least one person hits the target when two people shoot independently -/
theorem prob_at_least_one_hit (P₁ P₂ : ℝ) (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  probability_at_least_one_hit P₁ P₂ = 1 - (1 - P₁) * (1 - P₂) :=
by
  -- Unfold the definition of probability_at_least_one_hit
  unfold probability_at_least_one_hit
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_hit_l1280_128022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_one_l1280_128052

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^3 * (a * 2^x - 2^(-x)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * (2:ℝ)^x - (2:ℝ)^(-x))

/-- If f is an even function, then a = 1 -/
theorem even_function_implies_a_eq_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
  sorry

#check even_function_implies_a_eq_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_eq_one_l1280_128052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_and_angle_range_l1280_128043

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Definition of an acute triangle --/
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧ 0 < t.B ∧ t.B < Real.pi/2 ∧ 0 < t.C ∧ t.C < Real.pi/2

theorem triangle_side_range_and_angle_range (t : Triangle) :
  (isAcute t ∧ t.a = 2 ∧ t.b = 3 → Real.sqrt 5 < t.c ∧ t.c < Real.sqrt 13) ∧
  (Real.sin t.A ^ 2 ≤ Real.sin t.B ^ 2 + Real.sin t.C ^ 2 - Real.sin t.B * Real.sin t.C
    → 0 < t.A ∧ t.A ≤ Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_and_angle_range_l1280_128043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1280_128075

/-- The focal length of a hyperbola is twice the distance from the center to a focus -/
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * (a^2 + b^2).sqrt

/-- The hyperbola equation in standard form (x^2/a^2 - y^2/b^2 = 1) -/
def is_hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

theorem hyperbola_focal_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ is_hyperbola (x / (3 : ℝ).sqrt) (y / (3 : ℝ).sqrt) a b ∧ focal_length a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1280_128075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_played_l1280_128002

-- Define the initial run rate
def initial_run_rate : ℝ := 3.2

-- Define the run rate for the remaining overs
def remaining_run_rate : ℝ := 11.363636363636363

-- Define the number of remaining overs
def remaining_overs : ℕ := 22

-- Define the target total runs
def target_runs : ℕ := 282

-- Theorem statement
theorem initial_overs_played : 
  ∃ (x : ℕ), (initial_run_rate * (x : ℝ) + remaining_run_rate * (remaining_overs : ℝ) = (target_runs : ℝ)) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_played_l1280_128002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_cube_sums_l1280_128004

theorem infinitely_many_cube_sums (n : ℕ) :
  ∃ (f : ℕ → ℕ), Function.Injective f ∧ ∀ k, ∃ m : ℕ+, n^6 + 3 * (f k) = m^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_cube_sums_l1280_128004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l1280_128063

/-- The point on the plane 2x - 3y + 4z = 40 closest to (1, 0, 2) -/
noncomputable def closest_point : ℝ × ℝ × ℝ := (89/29, -90/29, 176/29)

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (1, 0, 2)

/-- The plane equation -/
def plane_equation (p : ℝ × ℝ × ℝ) : Prop :=
  2 * p.1 - 3 * p.2.1 + 4 * p.2.2 = 40

/-- Distance between two points in ℝ³ -/
noncomputable def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2)

theorem closest_point_is_closest :
  plane_equation closest_point ∧
  ∀ p : ℝ × ℝ × ℝ, plane_equation p →
    distance closest_point given_point ≤ distance p given_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l1280_128063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_set_price_theorem_l1280_128034

/-- Calculates the final price of a set meal on Wednesday -/
def wednesday_set_price (coffee_price cheesecake_price sandwich_price : ℚ)
  (coffee_discount cheesecake_discount : ℚ)
  (set_discount sales_tax : ℚ) : ℚ :=
  let discounted_coffee := coffee_price * (1 - coffee_discount)
  let discounted_cheesecake := cheesecake_price * (1 - cheesecake_discount)
  let subtotal := discounted_coffee + discounted_cheesecake + sandwich_price - set_discount
  let total := subtotal * (1 + sales_tax)
  (total * 100).floor / 100  -- Rounding down to 2 decimal places

/-- The final price of the Wednesday set meal is $19.43 -/
theorem wednesday_set_price_theorem :
  wednesday_set_price 6 10 8 (1/4) (1/10) 3 (1/20) = 1943/100 := by
  sorry

#eval wednesday_set_price 6 10 8 (1/4) (1/10) 3 (1/20)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_set_price_theorem_l1280_128034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_4_l1280_128013

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

-- State the theorem
theorem derivative_f_at_pi_over_4 :
  deriv f (π / 4) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_4_l1280_128013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1280_128088

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_term : a 1 = 2
  common_diff : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h : S seq 2 = seq.a 3) : 
  seq.a 2 = 4 ∧ S seq 10 = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1280_128088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_derivatives_eq_neg_1012029_l1280_128041

/-- The function f(x) = x cos x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

/-- The nth derivative of f evaluated at 0 -/
noncomputable def f_n_at_0 (n : ℕ) : ℝ :=
  if n = 0 then f 0
  else (deriv^[n] f) 0

/-- The sum of f(0) and its derivatives up to the 2013th order evaluated at x = 0 -/
noncomputable def sum_derivatives : ℝ :=
  Finset.sum (Finset.range 2014) f_n_at_0

theorem sum_derivatives_eq_neg_1012029 : sum_derivatives = -1012029 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_derivatives_eq_neg_1012029_l1280_128041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1280_128016

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
  t.a^2 + t.c^2 - t.b^2 = t.a * t.c ∧
  Real.sqrt 2 * t.b = Real.sqrt 3 * t.c

-- Define the function f(x)
noncomputable def f (x : ℝ) (B : ℝ) : ℝ :=
  1 + Real.cos (2 * x + B) - Real.cos (2 * x)

-- State the theorem
theorem triangle_properties {t : Triangle} (h : triangle_conditions t) :
  t.A = 5 * Real.pi / 12 ∧
  ∀ x, f x t.B ≤ 2 ∧ ∃ x₀, f x₀ t.B = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1280_128016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_one_implies_power_sum_one_l1280_128074

theorem sin_cos_sum_one_implies_power_sum_one (x : ℝ) :
  Real.sin x + Real.cos x = 1 → (Real.sin x)^2018 + (Real.cos x)^2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_one_implies_power_sum_one_l1280_128074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_l1280_128028

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) - x

-- Theorem for part 1
theorem monotonicity_of_f :
  let f₁ := f (-1)
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f₁ x₁ < f₁ x₂ ∧
  ∀ y₁ y₂, 2 < y₁ ∧ y₁ < y₂ → f₁ y₁ > f₁ y₂ := by
  sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → Real.exp (f a x) + (a / 2) * x^2 > 1) ↔
  a > (2 * (Real.exp 1 - 1)) / (Real.exp 1 + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_l1280_128028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_312_l1280_128098

-- Define the triangle
def triangle_DEF (DE EF DF : ℝ) : Prop :=
  DE = 25 ∧ EF = 25 ∧ DF = 36

-- Define Heron's formula for triangle area
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_is_312 :
  ∀ (DE EF DF : ℝ),
  triangle_DEF DE EF DF →
  heron_area DE EF DF = 312 :=
by
  intros DE EF DF h
  sorry -- Proof to be completed later


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_312_l1280_128098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_A_and_D_are_true_l1280_128032

-- Define the propositions
def proposition_A (θ : Real) : Prop :=
  (θ ∈ Set.Icc 0 (Real.pi/2) ∪ Set.Icc (3*Real.pi/2) (2*Real.pi)) → Real.cos θ > 0

def proposition_D (area : Real) (circumference : Real) (central_angle : Real) : Prop :=
  (area = 1 ∧ circumference = 4) → central_angle = 2

-- Theorem statement
theorem propositions_A_and_D_are_true :
  (∀ θ, proposition_A θ) ∧
  (∀ area circumference central_angle, proposition_D area circumference central_angle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_A_and_D_are_true_l1280_128032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_is_252_l1280_128086

/-- The area covered by two congruent squares with side length 12, where the center of one square
    coincides with the midpoint of a side of the other square. -/
noncomputable def area_of_overlapping_squares : ℝ :=
  let square_side : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let square_diagonal : ℝ := square_side * Real.sqrt 2
  let overlap_diagonal : ℝ := square_diagonal / 2
  let overlap_area : ℝ := (1 / 2) * overlap_diagonal * overlap_diagonal
  2 * square_area - overlap_area

/-- Theorem stating that the area covered by the overlapping squares is 252. -/
theorem area_of_overlapping_squares_is_252 :
  area_of_overlapping_squares = 252 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_overlapping_squares_is_252_l1280_128086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1280_128066

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (2 * x - 1) / Real.log 5)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x > 1/2 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1280_128066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l1280_128050

theorem cos_difference (α β : ℝ) 
  (h1 : Real.cos α - Real.cos β = 1/2) 
  (h2 : Real.sin α - Real.sin β = 1/3) : 
  Real.cos (α - β) = 59/72 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l1280_128050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1280_128054

/-- The lateral surface area of a frustum -/
noncomputable def lateral_surface_area (upper_base_area lower_base_diameter slant_height : ℝ) : ℝ :=
  Real.pi * slant_height * (lower_base_diameter / 2 + (upper_base_area / Real.pi).sqrt)

/-- Theorem: The lateral surface area of a frustum with given dimensions -/
theorem frustum_lateral_surface_area :
  lateral_surface_area 25 20 10 = 150 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l1280_128054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_P_on_y_equals_x_l1280_128012

-- Define the points
noncomputable def P : ℝ × ℝ := (1, 1)
noncomputable def Q : ℝ × ℝ := (1, 2)
noncomputable def M : ℝ × ℝ := (2, 3)
noncomputable def N : ℝ × ℝ := (1/2, 1/4)

-- Define the condition for a point to be on y = x
def on_y_equals_x (p : ℝ × ℝ) : Prop := p.1 = p.2

-- Theorem statement
theorem only_P_on_y_equals_x :
  on_y_equals_x P ∧
  ¬on_y_equals_x Q ∧
  ¬on_y_equals_x M ∧
  ¬on_y_equals_x N :=
by
  sorry

#check only_P_on_y_equals_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_P_on_y_equals_x_l1280_128012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_of_revolution_l1280_128027

-- Define the curve y = 2ln x
noncomputable def curve (x : ℝ) : ℝ := 2 * Real.log x

-- Define the region
def region (x y : ℝ) : Prop :=
  0 ≤ x ∧ 0 ≤ y ∧ y ≤ 1 ∧ y ≤ curve x

-- Define the volume of the solid of revolution
noncomputable def volume : ℝ := 4 * Real.pi * (Real.sqrt (Real.exp 1) - 1)

-- Theorem statement
theorem volume_of_solid_of_revolution :
  volume = ∫ y in Set.Icc 0 1, 2 * Real.pi * Real.exp (y / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solid_of_revolution_l1280_128027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axes_l1280_128039

theorem sine_symmetry_axes (ω φ : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∀ x : ℝ, Real.sin (ω * (π/4 + x) + φ) = Real.sin (ω * (π/4 - x) + φ))
  (h5 : ∀ x : ℝ, Real.sin (ω * (5*π/4 + x) + φ) = Real.sin (ω * (5*π/4 - x) + φ)) :
  φ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_symmetry_axes_l1280_128039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_then_red_prob_l1280_128025

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Color of a card -/
inductive Color
  | Black
  | Red
deriving DecidableEq

/-- Given a card number, determine its color -/
def card_color (n : Fin 52) : Color :=
  if n.val < 26 then Color.Black else Color.Red

/-- The probability of drawing a black card first and a red card second from a standard deck -/
theorem black_then_red_prob (d : Deck) : 
  (Finset.filter (λ i => card_color i = Color.Black) d.cards).card * 
  (Finset.filter (λ i => card_color i = Color.Red) d.cards).card / 
  (d.cards.card * (d.cards.card - 1)) = 13 / 51 := by
  sorry

/-- Helper lemma: card_color is decidable -/
lemma card_color_decidable (i : Fin 52) (c : Color) : 
  Decidable (card_color i = c) := by
  cases c
  · exact inferInstance
  · exact inferInstance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_then_red_prob_l1280_128025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1280_128037

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 1 / Real.sqrt (2 - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 < x ∧ x < Real.sqrt 2} :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1280_128037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1280_128095

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
def is_midpoint (M X Y : ℝ × ℝ) : Prop := M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem length_AB (h1 : is_midpoint C A B)
                  (h2 : is_midpoint D B C)
                  (h3 : is_midpoint E C D)
                  (h4 : is_midpoint F D E)
                  (h5 : dist A F = 5) :
  dist A B = 80 / 9 := by
  sorry

-- Helper function to calculate Euclidean distance
noncomputable def dist (X Y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l1280_128095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_threshold_minimizes_wait_time_l1280_128084

/-- The number of crosswalks --/
def num_crosswalks : ℕ := 10

/-- The maximum wait time at a crosswalk --/
def max_wait_time : ℝ := 60

/-- The strategy function that determines whether to cross based on wait time --/
noncomputable def cross_strategy (k : ℝ) (t : ℝ) : Bool :=
  t ≤ k

/-- The expected wait time given a threshold k --/
noncomputable def expected_wait_time (k : ℝ) : ℝ :=
  30 - (30 - k/2) * (1 - (1 - k/60)^9)

/-- The optimal threshold that minimizes expected wait time --/
noncomputable def optimal_threshold : ℝ :=
  60 * (1 - (1/10)^(1/9))

/-- Theorem stating that the optimal_threshold minimizes the expected wait time --/
theorem optimal_threshold_minimizes_wait_time :
  ∀ k : ℝ, k ≥ 0 → k ≤ max_wait_time →
  expected_wait_time optimal_threshold ≤ expected_wait_time k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_threshold_minimizes_wait_time_l1280_128084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_divisibility_conditions_l1280_128014

theorem smallest_integer_with_divisibility_conditions :
  ∃ (M : ℕ), M = 249 ∧
  (∃ (k : ℕ), k ∈ ({M, M+1, M+2} : Set ℕ) ∧ 8 ∣ k) ∧
  (∃ (k : ℕ), k ∈ ({M, M+1, M+2} : Set ℕ) ∧ 27 ∣ k) ∧
  (∃ (k : ℕ), k ∈ ({M, M+1, M+2} : Set ℕ) ∧ 125 ∣ k) ∧
  (∀ (N : ℕ), N < M →
    ¬((∃ (k : ℕ), k ∈ ({N, N+1, N+2} : Set ℕ) ∧ 8 ∣ k) ∧
      (∃ (k : ℕ), k ∈ ({N, N+1, N+2} : Set ℕ) ∧ 27 ∣ k) ∧
      (∃ (k : ℕ), k ∈ ({N, N+1, N+2} : Set ℕ) ∧ 125 ∣ k))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_divisibility_conditions_l1280_128014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_plane_with_60_degree_inclination_l1280_128079

/-- Represents a line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Angle between two vectors in radians -/
noncomputable def angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Projection of a vector onto a plane -/
noncomputable def projection (v : ℝ × ℝ × ℝ) (plane : Plane3D) : ℝ × ℝ × ℝ := sorry

/-- First angle of inclination of a plane -/
noncomputable def firstInclinationAngle (plane : Plane3D) : ℝ := sorry

/-- Check if a point is on a plane -/
def isOnPlane (point : ℝ × ℝ × ℝ) (plane : Plane3D) : Prop := sorry

/-- Main theorem -/
theorem exists_plane_with_60_degree_inclination
  (g : Line3D)
  (h1 : ∃ (proj1 proj2 : Plane3D), projection g.direction proj1 = projection g.direction proj2)
  (h2 : ∀ (proj : Plane3D), angle g.direction (projection g.direction proj) = π / 4) :
  ∃ (p : Plane3D), isOnPlane g.point p ∧ firstInclinationAngle p = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_plane_with_60_degree_inclination_l1280_128079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radiocarbon_dating_age_l1280_128081

/-- The half-life of ^14C in years -/
noncomputable def halfLife : ℝ := 5730

/-- The ratio of remaining ^14C to original amount -/
noncomputable def remainingRatio : ℝ := 1/5

/-- Approximation of (lg 2)^-1 -/
noncomputable def logTwoInvApprox : ℝ := 3.3219

/-- The approximate age of the ancient object in years -/
noncomputable def approximateAge : ℝ := 13304

theorem radiocarbon_dating_age :
  let decayRate := (1/2 : ℝ) ^ (1/halfLife)
  let age := halfLife * (Real.log remainingRatio / Real.log decayRate)
  abs (age - approximateAge) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radiocarbon_dating_age_l1280_128081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_y_l1280_128089

theorem existence_of_x_y (m : ℕ) (x₀ y₀ : ℤ) 
  (h_pos : m > 0)
  (h_coprime : Int.gcd x₀ y₀ = 1)
  (h_div_y : ∃ k : ℤ, x₀^2 + m = k * y₀)
  (h_div_x : ∃ l : ℤ, y₀^2 + m = l * x₀) :
  ∃ x y : ℕ, 
    x > 0 ∧ 
    y > 0 ∧ 
    Int.gcd x y = 1 ∧ 
    (∃ k : ℕ, x^2 + m = k * y) ∧
    (∃ l : ℕ, y^2 + m = l * x) ∧
    x + y ≤ m + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_x_y_l1280_128089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l1280_128019

/-- A pentagon with specific properties -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  triangle_base : ℝ
  triangle_height : ℝ

/-- The area of a pentagon with the given properties -/
noncomputable def pentagon_area (p : Pentagon) : ℝ :=
  p.rectangle_width * p.rectangle_height + (1/2) * p.triangle_base * p.triangle_height

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area :
  let p : Pentagon := {
    side1 := 12, side2 := 17, side3 := 25, side4 := 18, side5 := 17,
    rectangle_width := 18, rectangle_height := 12,
    triangle_base := 18, triangle_height := 13
  }
  pentagon_area p = 333 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l1280_128019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_triangle_max_area_l1280_128053

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sum of angles in a triangle is π --/
axiom angle_sum (t : Triangle) : t.A + t.B + t.C = Real.pi

/-- The law of cosines --/
axiom law_of_cosines (t : Triangle) : t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

/-- The sine law --/
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The area of a triangle --/
noncomputable def triangle_area (t : Triangle) : ℝ := 1/2 * t.b * t.c * Real.sin t.A

/-- The dot product of vectors AB and AC --/
noncomputable def dot_product (t : Triangle) : ℝ := t.b * t.c * Real.cos t.A

theorem triangle_side_sum_range (t : Triangle) (h1 : t.a = 2) (h2 : t.A = Real.pi/3) :
  2 < t.b + t.c ∧ t.b + t.c ≤ 4 := by sorry

theorem triangle_max_area (t : Triangle) (h1 : t.a = 2) (h2 : dot_product t = 1) :
  triangle_area t ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_triangle_max_area_l1280_128053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_prime_square_plus_odd_l1280_128001

theorem unique_solution_for_prime_square_plus_odd (x y : ℕ) :
  Nat.Prime x → Odd y → (x^2 + y = 2007) → ∃! (x y : ℕ), Nat.Prime x ∧ Odd y ∧ x^2 + y = 2007 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_prime_square_plus_odd_l1280_128001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_saved_is_two_hours_l1280_128015

/-- Represents the rowing scenario with changing tide -/
structure RowingScenario where
  initial_distance : ℝ
  initial_time : ℝ
  further_distance : ℝ
  further_time : ℝ

/-- Calculates the time saved if the tide direction had not changed -/
noncomputable def time_saved (scenario : RowingScenario) : ℝ :=
  let speed_with_tide := scenario.initial_distance / scenario.initial_time
  let speed_against_tide := scenario.further_distance / scenario.further_time
  let time_with_constant_tide := scenario.further_distance / speed_with_tide
  scenario.further_time - time_with_constant_tide

/-- Theorem stating that the time saved is 2 hours for the given scenario -/
theorem time_saved_is_two_hours (scenario : RowingScenario) 
    (h1 : scenario.initial_distance = 5)
    (h2 : scenario.initial_time = 1)
    (h3 : scenario.further_distance = 40)
    (h4 : scenario.further_time = 10) :
  time_saved scenario = 2 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_saved_is_two_hours_l1280_128015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_6_max_area_when_c_is_2_l1280_128062

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h_angles : A + B + C = Real.pi
  h_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.b / t.c = Real.sqrt 3 * Real.sin t.A + Real.cos t.A

-- Theorem 1: Angle C = π/6
theorem angle_C_is_pi_over_6 (t : Triangle) (h : given_condition t) : 
  t.C = Real.pi / 6 := by sorry

-- Theorem 2: Maximum area when c = 2
theorem max_area_when_c_is_2 (t : Triangle) (h1 : given_condition t) (h2 : t.c = 2) :
  ∃ (max_area : Real), 
    (∀ (s : Real), s = 1/2 * t.a * t.b * Real.sin t.C → s ≤ max_area) ∧
    max_area = 2 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_6_max_area_when_c_is_2_l1280_128062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_size_l1280_128006

/-- A partition of a set of integers into two subsets -/
def Partition (n : ℕ) := (Finset ℕ) × (Finset ℕ)

/-- Checks if a partition is valid for the given n -/
def is_valid_partition (n : ℕ) (p : Partition n) : Prop :=
  let (A, B) := p
  (∀ x, x ∈ A → x ≥ 2 ∧ x ≤ n) ∧
  (∀ x, x ∈ B → x ≥ 2 ∧ x ≤ n) ∧
  (A ∪ B).card = n - 1 ∧
  A ∩ B = ∅ ∧
  (∀ x y, x ∈ A → y ∈ A → x ≠ y → ¬∃ (z : ℕ), x + y = z^2) ∧
  (∀ x y, x ∈ B → y ∈ B → x ≠ y → ¬∃ (z : ℕ), x + y = z^2)

/-- The existence of a valid partition for a given n -/
def exists_valid_partition (n : ℕ) : Prop :=
  ∃ p : Partition n, is_valid_partition n p

/-- The main theorem stating that 28 is the largest n for which a valid partition exists -/
theorem max_partition_size :
  (∀ n > 28, ¬exists_valid_partition n) ∧ exists_valid_partition 28 :=
sorry

#check max_partition_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_partition_size_l1280_128006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_property_l1280_128097

theorem arithmetic_geometric_sequence_property (a b : ℝ) :
  (2 * a = 1 + b) →  -- arithmetic sequence condition
  (b ^ 2 = a) →      -- geometric sequence condition
  (a ≠ b) →          -- a is not equal to b
  (7 * a * (Real.log (-b) / Real.log a) = 7 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_property_l1280_128097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_technicians_count_l1280_128082

theorem technicians_count (total_workers : ℕ) (avg_salary_all : ℕ) 
  (avg_salary_tech : ℕ) (avg_salary_non_tech : ℕ) 
  (h1 : total_workers = 49)
  (h2 : avg_salary_all = 8000)
  (h3 : avg_salary_tech = 20000)
  (h4 : avg_salary_non_tech = 6000)
  : ∃ (tech_count : ℕ),
    tech_count + (total_workers - tech_count) = total_workers ∧
    avg_salary_all * total_workers = 
      avg_salary_tech * tech_count + avg_salary_non_tech * (total_workers - tech_count) ∧
    tech_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_technicians_count_l1280_128082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l1280_128068

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) (frequency : ℕ) : ℝ :=
  principal * (1 + rate / (frequency : ℝ)) ^ ((frequency : ℝ) * (time : ℝ)) - principal

/-- The compound interest problem -/
theorem compound_interest_problem : 
  let principal : ℝ := 500
  let rate : ℝ := 0.05
  let time : ℕ := 5
  let frequency : ℕ := 1
  abs (compound_interest principal rate time frequency - 138.14) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l1280_128068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_height_after_one_year_l1280_128093

/-- Represents a tree with its growth properties and initial height -/
structure TreeGrowth where
  initialHeight : ℕ  -- in cm
  springGrowthRate : ℕ  -- in cm
  springGrowthPeriod : ℕ  -- in weeks
  winterGrowthRate : ℕ  -- in cm
  winterGrowthPeriod : ℕ  -- in weeks

/-- Calculates the height of a tree after one year -/
def heightAfterOneYear (t : TreeGrowth) : ℕ :=
  let springGrowth := (26 / t.springGrowthPeriod) * t.springGrowthRate
  let winterGrowth := (26 / t.winterGrowthPeriod) * t.winterGrowthRate
  t.initialHeight + springGrowth + winterGrowth

/-- The combined height of all trees after one year -/
def combinedHeightAfterOneYear (trees : List TreeGrowth) : ℕ :=
  trees.map heightAfterOneYear |>.sum

/-- Theorem stating the combined height of the three trees after one year -/
theorem combined_height_after_one_year :
  let treeA : TreeGrowth := {
    initialHeight := 200,
    springGrowthRate := 50,
    springGrowthPeriod := 2,
    winterGrowthRate := 25,
    winterGrowthPeriod := 2
  }
  let treeB : TreeGrowth := {
    initialHeight := 150,
    springGrowthRate := 70,
    springGrowthPeriod := 3,
    winterGrowthRate := 35,
    winterGrowthPeriod := 3
  }
  let treeC : TreeGrowth := {
    initialHeight := 250,
    springGrowthRate := 90,
    springGrowthPeriod := 4,
    winterGrowthRate := 45,
    winterGrowthPeriod := 4
  }
  combinedHeightAfterOneYear [treeA, treeB, treeC] = 3150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_height_after_one_year_l1280_128093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joans_travel_time_l1280_128096

/-- Calculates the total travel time given distance, speed, and break time -/
noncomputable def totalTravelTime (distance : ℝ) (speed : ℝ) (breakTime : ℝ) : ℝ :=
  distance / speed + breakTime

/-- Proves that the total travel time for Joan's journey is 9 hours -/
theorem joans_travel_time :
  let distance : ℝ := 480
  let speed : ℝ := 60
  let lunchBreak : ℝ := 0.5
  let bathroomBreaks : ℝ := 2 * 0.25
  totalTravelTime distance speed (lunchBreak + bathroomBreaks) = 9 := by
  sorry

#check joans_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joans_travel_time_l1280_128096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1280_128046

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1
def C2 (x y : ℝ) : Prop := 4*x + 3*y - 8 = 0

-- Define point M as the intersection of C2 and the x-axis
def M : ℝ × ℝ := (2, 0)

-- N is a point on C1
def N (x y : ℝ) : Prop := C1 x y

-- Distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_MN :
  ∀ x y : ℝ, N x y → distance M (x, y) ≤ Real.sqrt 5 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_MN_l1280_128046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_transfer_height_l1280_128003

/-- Proves that the height of water in a cylinder is 1.25 cm when transferred from a cone -/
theorem water_transfer_height (cone_radius cone_height cylinder_radius : ℝ) 
  (hr : cone_radius = 15)
  (hh : cone_height = 15)
  (hcr : cylinder_radius = 30) : 
  (1/3 * Real.pi * cone_radius^2 * cone_height) / (Real.pi * cylinder_radius^2) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_transfer_height_l1280_128003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l1280_128000

-- Define the points
def D : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (8, 0)
def F : ℝ × ℝ := (2, 4)
def Q : ℝ × ℝ := (3, 1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem sum_of_distances :
  distance D Q + distance E Q + distance F Q = 2 * Real.sqrt 10 + Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l1280_128000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1280_128065

theorem constant_term_expansion : 
  let f (x : ℝ) := x^4 + 2*x^2 + 7
  let g (x : ℝ) := 2*x^5 + 3*x^2 + 20
  (f * g) 0 = 140 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1280_128065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_form_max_t_l1280_128067

/-- A quadratic function f(x) with specific properties -/
noncomputable def f (b c x : ℝ) : ℝ := 2 * x^2 + b * x + c

/-- The solution set of f(x) < 0 is (0,5) -/
axiom solution_set : ∃ b c : ℝ, ∀ x : ℝ, f b c x < 0 ↔ 0 < x ∧ x < 5

/-- The explicit form of f(x) is 2x^2 - 10x -/
theorem explicit_form : ∃ b c : ℝ, ∀ x : ℝ, f b c x = 2 * x^2 - 10 * x := by
  sorry

/-- The maximum value of t for which f(x) + t ≤ 2 holds for all x ∈ [-1,1] is -10 -/
theorem max_t : ∃ b c : ℝ, ∀ t : ℝ, 
  (∀ x : ℝ, x ≥ -1 → x ≤ 1 → f b c x + t ≤ 2) ↔ t ≤ -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_explicit_form_max_t_l1280_128067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1280_128045

theorem gcd_problem (b : ℤ) (h : 1820 ∣ b) : 
  Int.gcd (b^2 + 11*b + 28) (b + 6) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1280_128045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_eq_neg_two_i_l1280_128011

-- Define the complex number i
def i : ℂ := Complex.I

-- Define a as a complex number
noncomputable def a : ℂ := (-3 - i) / (1 + 2*i)

-- Theorem statement
theorem a_squared_eq_neg_two_i : a^2 = -2*i := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_squared_eq_neg_two_i_l1280_128011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_revenue_decrease_l1280_128036

theorem tax_revenue_decrease (T C : ℝ) (h1 : T > 0) (h2 : C > 0) :
  (T * C - T * (1 - 0.4) * C * (1 + 0.25)) / (T * C) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_revenue_decrease_l1280_128036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_exponent_l1280_128061

theorem power_of_two_exponent (n : ℝ) :
  (∃ k : ℝ, n = 2^k) →
  n^(12 : ℝ) = 8 →
  n = 2^(1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_exponent_l1280_128061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_perimeter_l1280_128040

/-- Triangle with vertices A(-2, 1), B(7, -3), and C(4, 6) -/
structure Triangle where
  A : ℝ × ℝ := (-2, 1)
  B : ℝ × ℝ := (7, -3)
  C : ℝ × ℝ := (4, 6)

/-- Calculate the area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Calculate the perimeter of the triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := sorry

/-- Theorem stating the area and perimeter of the specific triangle -/
theorem triangle_area_and_perimeter :
  let t : Triangle := {}
  (area t = 28.5) ∧ (abs (perimeter t - 29.2) < 0.1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_perimeter_l1280_128040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_C₁_C₂_intersection_l1280_128008

-- Define the parametric equations for C₁
noncomputable def C₁ (k : ℕ) (t : ℝ) : ℝ × ℝ := (Real.cos t ^ k, Real.sin t ^ k)

-- Define the Cartesian equation for C₂
def C₂ (x y : ℝ) : Prop := 4 * x - 16 * y + 3 = 0

-- Theorem 1: When k = 1, C₁ is a circle with radius 1 centered at the origin
theorem C₁_is_unit_circle : 
  ∀ (x y : ℝ), (∃ t, C₁ 1 t = (x, y)) ↔ x^2 + y^2 = 1 := by sorry

-- Theorem 2: When k = 4, the intersection of C₁ and C₂ is (1/4, 1/4)
theorem C₁_C₂_intersection : 
  ∃! (x y : ℝ), (∃ t, C₁ 4 t = (x, y)) ∧ C₂ x y ∧ x = 1/4 ∧ y = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₁_is_unit_circle_C₁_C₂_intersection_l1280_128008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_blue_packs_l1280_128085

/-- Represents the number of packs of T-shirts --/
abbrev Packs := ℕ

/-- Represents the cost in dollars --/
abbrev Dollars := ℕ

/-- Represents the problem of Maddie's T-shirt purchase --/
structure TShirtProblem where
  white_packs : Packs
  white_per_pack : ℕ
  blue_per_pack : ℕ
  cost_per_shirt : Dollars
  total_spent : Dollars

/-- Calculates the number of blue T-shirt packs bought --/
def blue_packs_bought (p : TShirtProblem) : Packs :=
  (p.total_spent - p.white_packs * p.white_per_pack * p.cost_per_shirt) / (p.blue_per_pack * p.cost_per_shirt)

/-- The main theorem stating that Maddie bought 4 packs of blue T-shirts --/
theorem maddie_blue_packs : 
  let p : TShirtProblem := {
    white_packs := 2,
    white_per_pack := 5,
    blue_per_pack := 3,
    cost_per_shirt := 3,
    total_spent := 66
  }
  blue_packs_bought p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maddie_blue_packs_l1280_128085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_reciprocal_l1280_128048

theorem imaginary_part_of_reciprocal (a : ℝ) (z : ℂ) : 
  z = a^2 - 1 + (a + 1) * Complex.I →
  z.re = 0 →
  z ≠ 0 →
  (1 / (z + a)).im = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_reciprocal_l1280_128048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1280_128038

noncomputable def f (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x

theorem symmetry_axis_of_f :
  ∃ (k : ℤ), ∀ (x : ℝ), f x = 2 * Real.sin (x - π/3) ∧
  (x - π/3 = k * π + π/2 ↔ x = k * π + 5*π/6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1280_128038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1280_128060

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-1/2 + Real.sqrt 2/2 * t, 1/2 + Real.sqrt 2/2 * t)

-- Define the ellipse C
def ellipse_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define point A
def point_A : ℝ × ℝ := (-1/2, 1/2)

-- Define the standard form of the ellipse
def ellipse_standard (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Theorem statement
theorem intersection_distance_product :
  ∃ (P Q : ℝ × ℝ), 
    (∃ t, line_l t = P) ∧ 
    (∃ θ, ellipse_C θ = P) ∧
    (∃ t, line_l t = Q) ∧ 
    (∃ θ, ellipse_C θ = Q) ∧
    P ≠ Q ∧
    (point_A.1 - P.1)^2 + (point_A.2 - P.2)^2 * 
    ((point_A.1 - Q.1)^2 + (point_A.2 - Q.2)^2) = (41/14)^2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1280_128060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_to_odd_l1280_128099

-- Define a function that is symmetric about a point (h, k)
def symmetric_about (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x : ℝ, f (2*h - x) = 2*k - f x

-- Define an odd function
def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- Theorem statement
theorem symmetric_to_odd (f : ℝ → ℝ) (h k : ℝ) 
  (sym : symmetric_about f h k) : 
  odd_function (fun x ↦ f (x + h) - k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_to_odd_l1280_128099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l1280_128005

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ (x θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ∈ Set.Iic (Real.sqrt 6) ∪ Set.Ici (7/2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_inequality_l1280_128005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l1280_128018

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- The problem statement -/
theorem simple_interest_problem :
  ∃ (principal : ℝ),
    simple_interest principal 8 3 = 
    (1/2) * compound_interest 4000 10 2 ∧
    principal = 1750 := by
  -- Prove the existence of such a principal
  use 1750
  -- Split the goal into two parts
  constructor
  -- Prove the equality of simple and compound interest
  · sorry
  -- Prove that the principal is indeed 1750
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l1280_128018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_2x_squared_l1280_128021

/-- Definition of the focus of a parabola -/
def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  sorry

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem focus_of_parabola_2x_squared :
  focus_of_parabola 2 0 0 = (0, 1/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_parabola_2x_squared_l1280_128021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1280_128057

-- Define the vectors and function
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.sqrt 3 * Real.cos x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_area (t : Triangle) (h1 : f t.A = 3) (h2 : t.B = π / 3) (h3 : t.c = 4) :
  (1 / 2) * t.a * t.b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1280_128057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l1280_128071

-- Define the constants
noncomputable def a : ℝ := (8.1 : ℝ) ^ (51/100 : ℝ)
noncomputable def b : ℝ := (8.1 : ℝ) ^ (1/2 : ℝ)
noncomputable def c : ℝ := Real.log 0.3 / Real.log 3

-- State the theorem
theorem order_of_constants : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l1280_128071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1280_128055

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - (Real.arcsin (x/3))^2 + (Real.pi^2/18) * (x^2 + 12*x + 27)

theorem g_range :
  ∀ y ∈ Set.range g, π^2/4 ≤ y ∧ y ≤ 5*π^2/2 ∧
  ∃ x₁ x₂, x₁ ∈ Set.Icc (-3 : ℝ) 3 ∧ x₂ ∈ Set.Icc (-3 : ℝ) 3 ∧ 
           g x₁ = π^2/4 ∧ g x₂ = 5*π^2/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1280_128055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_country_density_approximation_l1280_128072

/-- Represents the data for a country's population and area --/
structure CountryData where
  population : ℕ
  area_hectares : ℕ
  sq_meters_per_hectare : ℕ

/-- Calculates the average number of square meters per person --/
noncomputable def avg_sq_meters_per_person (data : CountryData) : ℝ :=
  (data.area_hectares * data.sq_meters_per_hectare : ℝ) / data.population

/-- Calculates the population density per hectare --/
noncomputable def population_density_per_hectare (data : CountryData) : ℝ :=
  (data.population : ℝ) / data.area_hectares

/-- Theorem stating the approximations for average square meters per person
    and population density per hectare --/
theorem country_density_approximation (data : CountryData)
    (h_pop : data.population = 350000000)
    (h_area : data.area_hectares = 4500000)
    (h_conv : data.sq_meters_per_hectare = 10000) :
    ⌊avg_sq_meters_per_person data⌋ = 128 ∧
    ⌊population_density_per_hectare data⌋ = 78 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_country_density_approximation_l1280_128072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1280_128080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a - 1) / (x^2 + 1)

theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (a = 1) ∧  -- Part 1: a = 1
  (∀ x y, 1 < x → x < y → f a x > f a y) ∧  -- Part 2: f is monotonically decreasing on (1, +∞)
  (∀ m n, m > 0 → n > 0 → m + n = 5 * a →  -- Part 3: Minimum value of 1/m + 1/(n+3) is 1/2
    ∀ p q, p > 0 → q > 0 → p + q = 5 * a →
      1/m + 1/(n+3) ≤ 1/p + 1/(q+3) ∧
      (1/m + 1/(n+3) = 1/2 ↔ m = 4 ∧ n = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1280_128080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_ratio_l1280_128023

/-- Pascal's Triangle property: each entry is the sum of the two entries above it -/
axiom pascal_triangle_property (n : ℕ) (r : ℕ) : 
  r ≤ n → Nat.choose n r = Nat.choose (n-1) (r-1) + Nat.choose (n-1) r

/-- Definition of the ratio condition for three consecutive entries -/
def ratio_condition (n : ℕ) (r : ℕ) : Prop :=
  (Nat.choose n r : ℚ) / (Nat.choose n (r+1)) = 2/3 ∧ 
  (Nat.choose n (r+1) : ℚ) / (Nat.choose n (r+2)) = 3/4

/-- Theorem: The row in Pascal's Triangle where three consecutive entries 
    are in the ratio 2:3:4 is the 34th row -/
theorem pascal_triangle_ratio : ∃ (r : ℕ), ratio_condition 34 r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_ratio_l1280_128023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l1280_128056

/-- Represents the sum of the first k terms of a geometric sequence --/
noncomputable def GeometricSum (a₁ : ℝ) (q : ℝ) (k : ℕ) : ℝ :=
  a₁ * (1 - q^k) / (1 - q)

/-- Theorem: For a geometric sequence with positive terms, if S_n = 2 and S_{3n} = 14, then S_{4n} = 30 --/
theorem geometric_sum_problem (a₁ : ℝ) (q : ℝ) (n : ℕ) 
  (h_pos : a₁ > 0 ∧ q > 0) 
  (h_n : GeometricSum a₁ q n = 2)
  (h_3n : GeometricSum a₁ q (3*n) = 14) :
  GeometricSum a₁ q (4*n) = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_problem_l1280_128056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_three_l1280_128049

theorem sequence_limit_is_three : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((4*n + 1 : ℝ)^(1/2) - (27*n^3 + 1 : ℝ)^(1/3)) / ((n : ℝ)^(1/4) - (n^5 + n : ℝ)^(1/3)) - 3| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_three_l1280_128049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1280_128090

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2*x) - f (x + 2)

-- Theorem statement
theorem g_properties :
  ∀ x ∈ Set.Icc 0 1,
    (g x = 2^(2*x) - 2^(x + 2)) ∧
    (Set.range (g ∘ (fun y => y : Set.Icc 0 1 → ℝ)) = Set.Icc (-4) (-3)) ∧
    (∃ x₀ ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, g x₀ ≥ g y) ∧
    (∃ x₁ ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, g x₁ ≤ g y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1280_128090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_increase_l1280_128076

/-- Represents the daily increase in weaving --/
def daily_increase : ℚ := sorry

/-- Represents the total amount woven in 30 days in chi --/
def total_woven : ℚ := sorry

/-- Conversion rate from bolt to zhang --/
def bolt_to_zhang : ℚ := 4

/-- Conversion rate from zhang to chi --/
def zhang_to_chi : ℚ := 10

/-- Conversion rate from chi to cun --/
def chi_to_cun : ℚ := 10

/-- The amount woven on the first day in chi --/
def first_day_weaving : ℚ := 5

/-- The total amount woven in one month in bolts --/
def total_bolts : ℚ := 9

/-- The additional amount woven in one month in chi --/
def additional_chi : ℚ := 30

/-- The number of days in a month --/
def days_in_month : ℕ := 30

theorem weaving_increase :
  daily_increase = (5 + 15 / 29) / chi_to_cun ∧
  total_woven = total_bolts * bolt_to_zhang * zhang_to_chi + additional_chi ∧
  total_woven = first_day_weaving * days_in_month + (days_in_month * (days_in_month - 1) / 2) * daily_increase := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaving_increase_l1280_128076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_conditions_imply_one_greater_than_one_l1280_128020

theorem two_conditions_imply_one_greater_than_one (a b : ℝ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (n = (if a + b > 1 then 1 else 0) +
         (if a + b = 2 then 1 else 0) +
         (if a + b > 2 then 1 else 0) +
         (if a^2 + b^2 > 2 then 1 else 0) +
         (if a^3 + b^3 > 2 then 1 else 0) +
         (if a * b > 1 then 1 else 0)) ∧
    (∀ c : ℝ → ℝ → Prop, 
      (c = (λ x y ↦ x + y > 1) ∨
       c = (λ x y ↦ x + y = 2) ∨
       c = (λ x y ↦ x + y > 2) ∨
       c = (λ x y ↦ x^2 + y^2 > 2) ∨
       c = (λ x y ↦ x^3 + y^3 > 2) ∨
       c = (λ x y ↦ x * y > 1)) →
      (c a b → (a > 1 ∨ b > 1)) ↔ (c = (λ x y ↦ x + y > 2) ∨ c = (λ x y ↦ x^3 + y^3 > 2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_conditions_imply_one_greater_than_one_l1280_128020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rows_with_47_l1280_128026

-- Define Pascal's Triangle
def pascal : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n+1, 0 => 1
  | n+1, k+1 => pascal n k + pascal n (k+1)

-- Define a function to check if a number appears in a row of Pascal's Triangle
def appears_in_row (num row : ℕ) : Bool :=
  ∃ k, k ≤ row ∧ pascal row k = num

-- Define a function to count the number of rows containing a number
def count_rows_with_num (num : ℕ) : ℕ :=
  (List.range 100).filter (appears_in_row num) |>.length

-- The main theorem
theorem rows_with_47 : count_rows_with_num 47 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rows_with_47_l1280_128026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_C0E_equals_3086_l1280_128091

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3 | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11 | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.data.reverse.enum.foldl
    (fun acc (i, c) => acc + (hex_to_dec c) * (16 ^ i))
    0

/-- Theorem: The hexadecimal number C0E is equal to 3086 in decimal -/
theorem hex_C0E_equals_3086 : hex_string_to_dec "C0E" = 3086 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_C0E_equals_3086_l1280_128091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_plans_count_l1280_128031

def project_plans (days_A : ℕ) (days_B : ℕ) (max_days : ℕ) : ℕ :=
  let plans := (List.range (max_days + 1)).bind (fun x =>
    (List.range (max_days + 1 - x)).filter (fun y =>
      x + y ≤ max_days ∧
      ((x : ℚ) * (1 / days_A + 1 / days_B) + (y : ℚ) * (1 / days_A) = 1 ∨
       (x : ℚ) * (1 / days_A + 1 / days_B) + (y : ℚ) * (1 / days_B) = 1)
    )
  )
  plans.length

theorem project_plans_count :
  project_plans 12 9 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_plans_count_l1280_128031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_through_C_avoiding_D_l1280_128044

/-- Represents a point on the grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a path on the grid --/
def GridPath := List GridPoint

/-- The probability of choosing a direction at an intersection --/
def direction_probability : ℚ := 1/2

/-- The number of paths from A to B that pass through C and avoid D --/
def paths_through_C : ℕ := 12

/-- The total number of paths from A to B that avoid D --/
def total_paths : ℕ := 21

/-- The theorem to prove --/
theorem probability_through_C_avoiding_D :
  (paths_through_C : ℚ) / total_paths = 12/21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_through_C_avoiding_D_l1280_128044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l1280_128033

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ
  B : ℝ

-- Define the function f
noncomputable def f (A x : ℝ) : ℝ := Real.sin A * Real.cos x ^ 2 - Real.sin (A / 2) ^ 2 * Real.sin (2 * x)

theorem triangle_and_function_properties 
  (ABC : Triangle) 
  (h1 : ABC.b * Real.sin ABC.A ^ 2 = Real.sqrt 3 * ABC.a * Real.cos ABC.A * Real.sin ABC.B) :
  ABC.A = π / 3 ∧ 
  Set.Icc (Real.sqrt 3 / 4 - 1 / 2) (Real.sqrt 3 / 2) = 
    Set.range (fun x ↦ f ABC.A x) ∩ Set.Icc 0 (π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l1280_128033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_chords_sine_l1280_128010

theorem parallel_chords_sine (r : ℝ) (θ φ : ℝ) : 
  0 < r →
  0 < θ →
  0 < φ →
  θ + φ < π →
  5 = 2 * r * Real.sin (θ / 2) →
  12 = 2 * r * Real.sin (φ / 2) →
  13 = 2 * r * Real.sin ((θ + φ) / 2) →
  Real.sin θ = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_chords_sine_l1280_128010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_365_1_equals_sin_5_1_l1280_128058

theorem sin_365_1_equals_sin_5_1 (m : ℝ) (h : Real.sin (5.1 * π / 180) = m) :
  Real.sin (365.1 * π / 180) = Real.sin (5.1 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_365_1_equals_sin_5_1_l1280_128058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l1280_128087

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- Two vectors are non-collinear -/
def NonCollinear (v w : V) : Prop := ∀ (r : ℝ), v ≠ r • w

/-- Two vectors are collinear -/
def AreCollinear (v w : V) : Prop := ∃ (r : ℝ), v = r • w

theorem collinear_vectors_lambda 
  (h_non_collinear : NonCollinear V e₁ e₂)
  (a b : V)
  (h_a : a = 2 • e₁ - 3 • e₂)
  (h_b : ∃ lambda : ℝ, b = lambda • e₁ + 6 • e₂)
  (h_collinear : AreCollinear V a b) :
  ∃ lambda : ℝ, b = lambda • e₁ + 6 • e₂ ∧ lambda = -4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_lambda_l1280_128087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1280_128073

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

-- State the theorem
theorem f_min_value (ω : ℝ) : 
  ∃ (x : ℝ), f ω x = -2 ∧ ∀ (y : ℝ), f ω y ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1280_128073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1280_128007

noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 1)^(x^2 - x - 6)

theorem solution_characterization :
  ∀ x : ℝ, f x = 1 ↔ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1280_128007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l1280_128017

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 2 * x + 7 * y = 35

/-- The y-intercept of the line -/
noncomputable def y_intercept : ℝ × ℝ := (0, 5)

/-- The x-intercept of the line -/
noncomputable def x_intercept : ℝ × ℝ := (35/2, 0)

/-- Theorem: The y-intercept and x-intercept of the line 2x + 7y = 35 are (0, 5) and (35/2, 0) respectively -/
theorem line_intercepts :
  (line_equation y_intercept.1 y_intercept.2) ∧
  (line_equation x_intercept.1 x_intercept.2) :=
by
  constructor
  · -- Proof for y-intercept
    simp [line_equation, y_intercept]
    norm_num
  · -- Proof for x-intercept
    simp [line_equation, x_intercept]
    norm_num

#check line_intercepts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l1280_128017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_area_l1280_128047

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vectors m and n -/
noncomputable def m (t : Triangle) : ℝ × ℝ := (Real.cos t.A, Real.cos t.B)
def n (t : Triangle) : ℝ × ℝ := (t.a, 2 * t.c - t.b)

/-- Parallel vectors condition -/
def parallel (t : Triangle) : Prop :=
  (2 * t.c - t.b) * Real.cos t.A = t.a * Real.cos t.B

theorem triangle_angle_and_area (t : Triangle) 
  (h_parallel : parallel t) 
  (h_positive : 0 < t.A ∧ t.A < Real.pi) :
  t.A = Real.pi / 3 ∧ 
  (∀ (t' : Triangle), t'.a = 4 → t'.A = Real.pi / 3 → 
    (1/2 * t'.b * t'.c * Real.sin t'.A) ≤ 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_area_l1280_128047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_speed_at_vertical_ring_speed_correct_l1280_128070

/-- A pendulum system with a weightless ring and two point masses -/
structure Pendulum where
  l : ℝ  -- Half-length of the rod
  m : ℝ  -- Mass at the middle of the rod
  g : ℝ  -- Gravitational acceleration

/-- The speed of the ring when the pendulum passes through the vertical position -/
noncomputable def ring_speed (p : Pendulum) : ℝ :=
  Real.sqrt (15 * p.g * p.l)

/-- Theorem stating the speed of the ring when the pendulum is vertical -/
theorem ring_speed_at_vertical (p : Pendulum) (h₁ : p.l > 0) (h₂ : p.m > 0) (h₃ : p.g > 0) :
  ring_speed p = Real.sqrt (15 * p.g * p.l) := by
  -- Unfold the definition of ring_speed
  unfold ring_speed
  -- The equality holds by definition
  rfl

/-- Theorem proving the correctness of the ring speed formula -/
theorem ring_speed_correct (p : Pendulum) (h₁ : p.l > 0) (h₂ : p.m > 0) (h₃ : p.g > 0) :
  (ring_speed p) ^ 2 = 15 * p.g * p.l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_speed_at_vertical_ring_speed_correct_l1280_128070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_l1280_128078

variable (U : Type) -- Universal set
variable (A B I : Set U) -- Subsets of U

-- Define the conditions
variable (h1 : A ⊆ I)
variable (h2 : B ⊆ I)

-- Define the complement operation in I
def complementI (X : Set U) : Set U := {x ∈ I | x ∉ X}

-- State the theorem
theorem complement_intersection :
  complementI U A (A ∩ B) = {x ∈ I | x ∉ A ∨ x ∉ B} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_l1280_128078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1280_128069

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 4) ∧ f x = y) ↔ y ∈ Set.Icc 1 (Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1280_128069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l1280_128024

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Reflects a point across the line y = x -/
def reflect (p : Point) : Point :=
  { x := p.y, y := p.x }

/-- The shortest path problem -/
theorem shortest_path (C B : Point) (h1 : C.x = 0 ∧ C.y = -6) (h2 : B.x = 12 ∧ B.y = -16) : 
  ∃ (P : Point), P.y = P.x ∧ distance C P + distance P B = 6 + Real.sqrt 292 := by
  sorry

#check shortest_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l1280_128024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_n_l1280_128030

noncomputable def sum_of_terms : ℝ :=
  1 / (Real.sin (15 * Real.pi / 180) * Real.sin (16 * Real.pi / 180)) +
  1 / (Real.sin (17 * Real.pi / 180) * Real.sin (18 * Real.pi / 180)) +
  -- ... (terms from 15° to 173°)
  1 / (Real.sin (173 * Real.pi / 180) * Real.sin (174 * Real.pi / 180))

theorem least_positive_integer_n :
  ∃ (n : ℕ), n > 0 ∧ sum_of_terms = 1 / Real.sin (n * Real.pi / 180) ∧
  ∀ (m : ℕ), m > 0 → sum_of_terms = 1 / Real.sin (m * Real.pi / 180) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_n_l1280_128030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l1280_128029

open Real

-- Define the radius and central angle
noncomputable def radius : ℝ := π
def central_angle_degrees : ℝ := 120

-- Convert the central angle to radians
noncomputable def central_angle_radians : ℝ := (central_angle_degrees * π) / 180

-- Define the arc length formula
def arc_length (r : ℝ) (θ : ℝ) : ℝ := r * θ

-- State the theorem
theorem sector_arc_length : 
  arc_length radius central_angle_radians = (2 * π^2) / 3 := by
  -- Unfold definitions
  unfold arc_length radius central_angle_radians central_angle_degrees
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- The proof is completed
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_arc_length_l1280_128029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_above_percentile_l1280_128051

/-- Given a group of students and their test scores, this theorem proves the minimum number of students scoring at or above the 80th percentile. -/
theorem min_students_above_percentile (total_students : ℕ) (percentile_score : ℝ) : 
  total_students = 1200 → 
  percentile_score = 103 →
  (↑total_students * (1 - 0.8) : ℝ) = 240 := by
  sorry

#check min_students_above_percentile

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_above_percentile_l1280_128051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_digit_sum_l1280_128059

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem smallest_n_digit_sum (n : ℕ) : 
  (n > 0) →
  ((Nat.factorial (n + 2) + Nat.factorial (n + 3) = Nat.factorial n * 720) →
   ∀ m, m > 0 ∧ m < n → 
   (Nat.factorial (m + 2) + Nat.factorial (m + 3) ≠ Nat.factorial m * 720)) →
  (sumOfDigits n = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_digit_sum_l1280_128059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1280_128042

def sequence_a (n : ℕ) : ℝ := (2 : ℝ) ^ n

def sequence_b (n : ℕ) : ℝ := 2 * n - 1

def sequence_c (n : ℕ) : ℝ := sequence_a n * sequence_b n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sum_T (n : ℕ) : ℝ := (2 * n - 3) * (2 : ℝ) ^ (n + 1) + 6

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sum_S n = 2 * sequence_a n - 2) ∧
  (sequence_b 1 = 1) ∧
  (∀ n : ℕ, n > 0 → sequence_b (n + 1) - sequence_b n = 2) →
  (∀ n : ℕ, n > 0 → sequence_a n = (2 : ℝ) ^ n) ∧
  (∀ n : ℕ, n > 0 → sequence_b n = 2 * n - 1) ∧
  (∀ n : ℕ, n > 0 → sum_T n = (2 * n - 3) * (2 : ℝ) ^ (n + 1) + 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1280_128042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1280_128094

theorem sin_half_angle (α : Real) 
  (h1 : Real.cos α = 3/5) 
  (h2 : α ∈ Set.Ioo (3/2 * Real.pi) (2 * Real.pi)) : 
  Real.sin (α/2) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1280_128094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_slope_is_two_l1280_128009

/-- A circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line that equally divides the total area of the circles -/
structure DividingLine where
  slope : ℝ
  passes_through : ℝ × ℝ

/-- The problem setup -/
def problem_setup : (List Circle) × DividingLine :=
  let circles := [
    { center := (15, 90), radius := 4 },
    { center := (18, 75), radius := 4 },
    { center := (20, 82), radius := 4 },
    { center := (22, 70), radius := 4 }
  ]
  let line := { slope := 2, passes_through := (18, 75) }
  (circles, line)

/-- Function to check if the areas are divided equally -/
def area_divided_equally (circles : List Circle) (line : DividingLine) : Prop :=
  sorry -- Implementation details omitted

/-- The theorem statement -/
theorem dividing_line_slope_is_two :
  let (circles, line) := problem_setup
  (∀ c ∈ circles, c.radius = 4) →
  line.passes_through = (18, 75) →
  area_divided_equally circles line →
  |line.slope| = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividing_line_slope_is_two_l1280_128009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_set_l1280_128077

theorem log_inequality_solution_set (x : ℝ) :
  (Real.log (x - 1) ≤ Real.log 2) ↔ (x ≤ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_set_l1280_128077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_e_minus_three_floor_l1280_128092

-- Define e as a constant representing the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem e_minus_three_floor : Int.floor (e - 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_e_minus_three_floor_l1280_128092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_derivative_l1280_128035

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_derivative_l1280_128035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1280_128083

/-- Given a hyperbola with the following properties:
    - Point P is the intersection of y = (b/3a)x and the left branch of x²/a² - y²/b² = 1
    - a > 0, b > 0
    - F₁ is the left focus
    - PF₁ is perpendicular to the x-axis
    Prove that the eccentricity of the hyperbola is 3√2/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ c : ℝ,
    let P : ℝ × ℝ := (-c, -b*c/(3*a))
    let F₁ : ℝ × ℝ := (-c, 0)
    let hyperbola := fun (x y : ℝ) ↦ x^2/a^2 - y^2/b^2 = 1
    let line := fun (x y : ℝ) ↦ y = b*x/(3*a)
    hyperbola P.1 P.2 ∧
    line P.1 P.2 ∧
    (P.1 - F₁.1) * F₁.2 = (P.2 - F₁.2) * F₁.1 →
    c/a = 3*Real.sqrt 2/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1280_128083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_triangle_area_not_integer_l1280_128064

-- Define a triangle with prime side lengths
structure PrimeTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_prime : Nat.Prime a
  b_prime : Nat.Prime b
  c_prime : Nat.Prime c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the area of a triangle using Heron's formula
noncomputable def triangle_area (t : PrimeTriangle) : ℝ :=
  let s := (t.a + t.b + t.c : ℝ) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Theorem statement
theorem prime_triangle_area_not_integer (t : PrimeTriangle) :
  ¬(∃ n : ℤ, (triangle_area t) = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_triangle_area_not_integer_l1280_128064
