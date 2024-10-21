import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1227_122765

/-- The standard equation of a circle with center (1, 2) and radius 1 -/
theorem circle_equation (x y : ℝ) : 
  (x - 1)^2 + (y - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1227_122765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_relation_gcd_u_n_n_plus_3_l1227_122772

def u : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 1
  | n + 3 => u (n + 2) + 2 * u (n + 1)

theorem u_relation (n p : ℕ) (hp : p > 1) :
  u (n + p) = u (n + 1) * u p + 2 * u n * u (p - 1) := by
  sorry

theorem gcd_u_n_n_plus_3 (n : ℕ) :
  Nat.gcd (u n) (u (n + 3)) = if n % 3 = 0 then 3 else 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_relation_gcd_u_n_n_plus_3_l1227_122772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_tangent_line_through_B_l1227_122724

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := 4 / x

-- Define points A and B
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (2, 0)

-- Theorem for the first part
theorem tangent_line_at_A :
  ∀ x y : ℝ, x + y - 4 = 0 ↔ 
  (∃ m : ℝ, (y - f A.1 = m * (x - A.1)) ∧ 
             (m = - (4 / A.1^2)) ∧
             (y = f x)) := by
  sorry

-- Theorem for the second part
theorem tangent_line_through_B :
  ∀ x y : ℝ, 4*x + y - 8 = 0 ↔
  (∃ m t : ℝ, (y - f t = - (4 / t^2) * (x - t)) ∧
              (B.2 - f t = - (4 / t^2) * (B.1 - t)) ∧
              (y = f x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_tangent_line_through_B_l1227_122724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1227_122706

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition given in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 + Real.sqrt 2 * t.a * t.c

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h : satisfiesCondition t) :
  t.B = π/4 ∧ 
  (∀ A C : ℝ, A + C = 3*π/4 → Real.sqrt 2 * Real.cos A + Real.cos C ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1227_122706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1227_122718

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x = f (-x)
axiom f_decreasing : ∀ x y, x ∈ Set.Icc (0 : ℝ) 2 → y ∈ Set.Icc (0 : ℝ) 2 → x < y → f x > f y
axiom f_domain : ∀ x, f x ≠ 0 → x ∈ Set.Icc (-2 : ℝ) 2

-- State the theorem
theorem range_of_m (m : ℝ) (h : f (1 - m) < f m) : 
  m ∈ Set.Icc (-1 : ℝ) (1/2) ∧ m ≠ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1227_122718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_is_twelve_l1227_122766

/-- Represents the fishing boat scenario --/
structure FishingBoat where
  initial_distance : ℝ
  initial_leak_rate : ℝ
  second_leak_rate : ℝ
  boat_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to avoid sinking --/
noncomputable def minimum_bailing_rate (boat : FishingBoat) : ℝ :=
  let total_time := boat.initial_distance / boat.rowing_speed
  let half_time := total_time / 2
  let initial_water := boat.initial_leak_rate * half_time
  let second_water := (boat.initial_leak_rate + boat.second_leak_rate) * half_time
  let total_water := initial_water + second_water
  let excess_water := total_water - boat.boat_capacity
  excess_water / total_time

/-- The theorem to be proved --/
theorem minimum_bailing_rate_is_twelve :
  let boat := FishingBoat.mk 3 8 10 50 2
  Int.floor (minimum_bailing_rate boat) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_bailing_rate_is_twelve_l1227_122766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l1227_122722

noncomputable def y (a x : ℝ) : ℝ := (x - 1)^2 + a*x + Real.sin (x + Real.pi/2)

theorem even_function_implies_a_equals_two (a : ℝ) :
  (∀ x, y a x = y a (-x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l1227_122722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_baby_ages_after_five_years_l1227_122787

/-- Represents the age and aging characteristics of an animal -/
structure Animal where
  age : ℝ
  agingRate : ℝ
  lifespan : ℝ

/-- Calculates the age of a baby animal given its mother's age -/
noncomputable def babyAge (mother : Animal) : ℝ := mother.age / 2

/-- Calculates the new age of an animal after a given number of human years -/
noncomputable def newAge (animal : Animal) (humanYears : ℝ) : ℝ :=
  animal.age + animal.agingRate * humanYears

/-- Calculates the new age of a baby animal after a given number of human years -/
noncomputable def newBabyAge (mother : Animal) (humanYears : ℝ) : ℝ :=
  babyAge mother + (newAge mother humanYears - mother.age) / 2

/-- Theorem statement for the sum of baby animal ages after 5 human years -/
theorem sum_of_baby_ages_after_five_years 
  (lioness hyena leopard : Animal)
  (h1 : lioness.age = 2 * hyena.age)
  (h2 : leopard.age = 3 * hyena.age)
  (h3 : lioness.age = 12)
  (h4 : lioness.lifespan = 25)
  (h5 : leopard.lifespan = 25)
  (h6 : hyena.lifespan = 22)
  (h7 : lioness.agingRate = 1.5)
  (h8 : leopard.agingRate = 2)
  (h9 : hyena.agingRate = 1.25) :
  newBabyAge lioness 5 + newBabyAge hyena 5 + newBabyAge leopard 5 = 29.875 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_baby_ages_after_five_years_l1227_122787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l1227_122719

theorem constant_term_in_expansion (n : ℕ) 
  (h : (Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose n k)) = 64) : 
  Nat.choose n (n / 2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l1227_122719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_40_l1227_122799

/-- Calculates the angle between hour and minute hands of a clock -/
noncomputable def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  |60 * (hour % 12 : ℝ) - 11 * (minute : ℝ)| / 2

theorem angle_at_3_40 :
  clockAngle 3 40 = 130 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_40_l1227_122799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_circle_radius_l1227_122763

-- Define the number of discs
def n : ℕ := 10

-- Define the radius of the inner circle
def inner_radius : ℝ := 1

-- Define the angle between two adjacent discs
noncomputable def angle : ℝ := 2 * Real.pi / n

-- Define the radius of the outer circle as a function
noncomputable def outer_radius : ℝ := (1 + Real.sin (angle / 2)) / (1 - Real.sin (angle / 2))

-- Theorem statement
theorem outer_circle_radius :
  outer_radius = (1 + Real.sin (Real.pi / 10)) / (1 - Real.sin (Real.pi / 10)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_circle_radius_l1227_122763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_is_difference_of_primes_l1227_122792

def arithmeticSequence (n : ℕ) : ℕ := 4 + 10 * n

def isPrime (p : ℕ) : Prop := Nat.Prime p

def canBeWrittenAsDifferenceOfPrimes (m : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ m = Int.natAbs (p - q)

theorem only_first_term_is_difference_of_primes :
  ∃! k : ℕ, canBeWrittenAsDifferenceOfPrimes (arithmeticSequence k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_is_difference_of_primes_l1227_122792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_rectangle_perimeter_l1227_122723

/-- Given a rectangle with dimensions 12 inches by 9 inches and a right triangular section
    cut from one corner with height 3 inches and area 18 square inches,
    the perimeter of the remaining unshaded rectangle is 24 inches. -/
theorem remaining_rectangle_perimeter : 
  ∀ (original_length original_width triangle_height triangle_area : ℝ),
  original_length = 12 →
  original_width = 9 →
  triangle_height = 3 →
  triangle_area = 18 →
  let triangle_base := 2 * triangle_area / triangle_height
  let new_length := original_length - triangle_base
  let new_width := original_width
  2 * (new_length + new_width) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_rectangle_perimeter_l1227_122723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sum_l1227_122710

theorem reciprocal_of_sum : (3/4 + 1/6 : ℚ)⁻¹ = 12/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sum_l1227_122710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_value_l1227_122713

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- The sequence a_n. -/
def a : Sequence := sorry

/-- The sequence b_n. -/
def b : Sequence := sorry

/-- b is a geometric sequence. -/
axiom b_geometric : ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = q * b n

/-- Relation between a_n and b_n. -/
axiom b_def : ∀ n : ℕ, b n = (a n + 1) / a n

/-- First term of a_n is 1. -/
axiom a_first : a 1 = 1

/-- Product of b_10 and b_11 is 6. -/
axiom b_product : b 10 * b 11 = 6

/-- Main theorem: a_20 equals 1/2. -/
theorem a_20_value : a 20 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_value_l1227_122713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_in_right_triangle_l1227_122758

theorem rectangle_area_in_right_triangle (h c x : ℝ) (h_pos : h > 0) (c_pos : c > 0) (x_pos : x > 0) (h_le_c : h ≤ c) (x_le_h : x ≤ h) :
  let y := x * (Real.sqrt (c^2 - h^2)) / h
  x * y = x^2 * (Real.sqrt (c^2 - h^2)) / h := by
    sorry

#check rectangle_area_in_right_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_in_right_triangle_l1227_122758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l1227_122730

-- Define c and d
noncomputable def c : ℝ := 4 + 3 * Real.sqrt 3
noncomputable def d : ℝ := 4 - 3 * Real.sqrt 3

-- Define the sequence S_n
noncomputable def S (n : ℕ) : ℝ := (c ^ n + d ^ n) / 2

-- Define a function to get the units digit
noncomputable def unitsDigit (x : ℝ) : ℕ := (Int.floor x).natAbs % 10

-- Theorem statement
theorem units_digit_S_6789 : unitsDigit (S 6789) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l1227_122730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1227_122754

/-- Given Alice's current age and the relationship between Alice and Tom's ages 10 years ago,
    prove that Tom is 15 years younger than Alice. -/
theorem age_difference (alice_current_age : ℕ) (h1 : alice_current_age = 30) :
  let tom_current_age := (alice_current_age - 10) / 4 + 10
  alice_current_age - tom_current_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l1227_122754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_unique_numerators_l1227_122741

/-- The set of rational numbers with repeating decimal expansions 0.abcdabcdabcd... -/
def T : Set ℚ :=
  {r : ℚ | 0 < r ∧ r < 1 ∧ ∃ (a b c d : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    r = (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999}

/-- The number of unique numerators required to express elements of T in lowest terms -/
def uniqueNumerators : ℕ := 5800

/-- The main theorem stating that the number of unique numerators is 5800 -/
theorem count_unique_numerators : (Finset.filter (fun n : ℕ => n < 10000 ∧ Nat.Coprime n 9999) (Finset.range 10000)).card = uniqueNumerators := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_unique_numerators_l1227_122741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_teams_is_six_l1227_122784

/-- Represents a team in the tournament -/
def Team : Type := Fin 7

/-- Represents a week in the tournament schedule -/
def Week : Type := Fin 4

/-- A schedule is a function that assigns to each team and week whether it has a home game -/
def Schedule : Type := Team → Week → Bool

/-- A valid schedule satisfies the tournament conditions -/
def ValidSchedule (n : Nat) (s : Schedule) : Prop :=
  -- Each team plays every other team twice (once at home, once away)
  ∀ t1 t2 : Team, t1.val < n → t2.val < n → t1 ≠ t2 →
    (∃! w : Week, s t1 w ∧ ¬s t2 w) ∧
    (∃! w : Week, s t2 w ∧ ¬s t1 w) ∧
  -- A team cannot have both home and away games in the same week
  ∀ t : Team, t.val < n → ∀ w : Week,
    s t w → ∀ t' : Team, t'.val < n → t ≠ t' → ¬s t' w

/-- The maximum number of teams that can participate in the tournament is 6 -/
theorem max_teams_is_six :
  (∃ s : Schedule, ValidSchedule 6 s) ∧
  (∀ n : Nat, n > 6 → ¬∃ s : Schedule, ValidSchedule n s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_teams_is_six_l1227_122784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_2_sqrt_15_l1227_122790

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the point O
def O : ℝ × ℝ := sorry

-- Define the sides a, b, c
noncomputable def a (t : Triangle) : ℝ := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
def b : ℝ := 4
noncomputable def c (t : Triangle) : ℝ := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)

-- Define the angles A, B, C
noncomputable def angle_A (t : Triangle) : ℝ := sorry
noncomputable def angle_B (t : Triangle) : ℝ := sorry
noncomputable def angle_C (t : Triangle) : ℝ := sorry

-- Define the given conditions
axiom triangle_oblique (t : Triangle) : angle_A t ≠ 0 ∧ angle_B t ≠ 0 ∧ angle_C t ≠ 0

axiom condition1 (t : Triangle) : 
  4 * Real.sqrt 6 * a t * Real.sin (2 * angle_C t) = 3 * (a t ^ 2 + b ^ 2 - c t ^ 2) * Real.sin (angle_B t)

axiom condition2 (t : Triangle) :
  2 * (O.1 - t.A.1, O.2 - t.A.2) + (O.1 - t.B.1, O.2 - t.B.2) + (O.1 - t.C.1, O.2 - t.C.2) = (0, 0)

noncomputable def angle_CAO (t : Triangle) : ℝ := sorry

axiom condition3 (t : Triangle) :
  Real.cos (angle_CAO t) = 1/4

-- Define the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ := sorry

-- Theorem to prove
theorem area_is_2_sqrt_15 (t : Triangle) : triangle_area t = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_2_sqrt_15_l1227_122790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l1227_122756

-- Define the function f(x) = x √(4 - x^2)
noncomputable def f (x : ℝ) : ℝ := x * Real.sqrt (4 - x^2)

-- Define the lower and upper bounds of x
def lower_bound : ℝ := 0
def upper_bound : ℝ := 2

-- State the theorem
theorem area_under_curve : 
  (∫ x in lower_bound..upper_bound, f x) = 8/3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l1227_122756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equivalence_l1227_122709

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  m_a : ℝ
  h_a : ℝ
  t_a : ℝ
  R : ℝ
  Δ : ℝ

-- Define the theorem
theorem triangle_equivalence (ABC A'B'C' : Triangle)
  (h1 : ABC.b = A'B'C'.b)
  (h2 : ABC.c = A'B'C'.c) :
  (ABC.a = A'B'C'.a) ∧
  (ABC.m_a = A'B'C'.m_a) ∧
  (ABC.h_a = A'B'C'.h_a) ∧
  (ABC.t_a = A'B'C'.t_a) ∧
  (ABC.R = A'B'C'.R) ∧
  (ABC.Δ = A'B'C'.Δ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equivalence_l1227_122709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_2_sqrt_5_l1227_122702

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def is_intersection (x y : ℝ) : Prop := parabola x y ∧ circle_eq x y

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem intersection_distance_is_2_sqrt_5 :
  ∃ x1 y1 x2 y2 : ℝ,
    is_intersection x1 y1 ∧
    is_intersection x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_2_sqrt_5_l1227_122702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_construction_output_additional_construction_output_l1227_122739

-- Define the variables
variable (x y : ℝ)

-- Define the conditions as functions
def condition1 (x y : ℝ) : Prop := 13 * x + 8 * y = 970
def condition2 (x y : ℝ) : Prop := 2 * x + 20 = 3 * y

-- Define the additional variable for part 2
variable (m : ℝ)

-- Define the theorem for part 1
theorem daily_construction_output 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 50 ∧ y = 40 := by sorry

-- Define the theorem for part 2
theorem additional_construction_output 
  (h1 : condition1 x y) (h2 : condition2 x y) : 
  4 * (x + y) + 8 * (y + m) ≥ 1000 → m ≥ 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_construction_output_additional_construction_output_l1227_122739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1227_122740

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Points on the sides of the octagon -/
structure OctagonPoints (o : RegularOctagon) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  T : ℝ × ℝ
  U : ℝ × ℝ
  V : ℝ × ℝ
  W : ℝ × ℝ

/-- Helper function to check if a point is on a line segment -/
def on_side (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  sorry

/-- Helper function to create a line from two points -/
def line (a : ℝ × ℝ) (b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Helper function to check if lines are parallel and equally spaced -/
def parallel_and_equally_spaced (lines : List (Set (ℝ × ℝ))) : Prop :=
  sorry

/-- Helper function to create an octagon from 8 points -/
def octagon (p1 p2 p3 p4 p5 p6 p7 p8 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Helper function to calculate the area ratio of two shapes -/
noncomputable def area_ratio (s1 s2 : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem octagon_area_ratio 
  (o : RegularOctagon) 
  (p : OctagonPoints o) 
  (h1 : on_side p.P (o.vertices 0) (o.vertices 1))
  (h2 : on_side p.Q (o.vertices 1) (o.vertices 2))
  (h3 : on_side p.R (o.vertices 2) (o.vertices 3))
  (h4 : on_side p.S (o.vertices 3) (o.vertices 4))
  (h5 : on_side p.T (o.vertices 4) (o.vertices 5))
  (h6 : on_side p.U (o.vertices 5) (o.vertices 6))
  (h7 : on_side p.V (o.vertices 6) (o.vertices 7))
  (h8 : on_side p.W (o.vertices 7) (o.vertices 0))
  (h9 : parallel_and_equally_spaced 
    [line (o.vertices 0) (o.vertices 6), 
     line p.P p.W, 
     line p.Q p.T, 
     line p.U p.R, 
     line p.S p.V]) :
  area_ratio (octagon p.P p.Q p.R p.S p.T p.U p.V p.W) 
             (octagon (o.vertices 0) (o.vertices 1) (o.vertices 2) (o.vertices 3) 
                      (o.vertices 4) (o.vertices 5) (o.vertices 6) (o.vertices 7)) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_l1227_122740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1227_122789

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The equation of a circle with center (a, b) and radius r -/
def circle_equation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_tangent_to_line :
  let r := distance_point_to_line 0 0 3 (-4) 5
  ∀ x y : ℝ, circle_equation x y 0 0 r ↔ x^2 + y^2 = 1 := by
  sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1227_122789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_juice_percentage_l1227_122781

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  fruit : String
  quantity : ℕ
  juice_ounces : ℚ

/-- Calculates the percentage of a specific juice in a blend -/
def juice_percentage (yield1 yield2 : JuiceYield) (blend_quantity1 blend_quantity2 : ℕ) : ℚ :=
  let total_juice1 := (yield1.juice_ounces / yield1.quantity) * blend_quantity1
  let total_juice2 := (yield2.juice_ounces / yield2.quantity) * blend_quantity2
  let total_juice := total_juice1 + total_juice2
  (total_juice1 / total_juice) * 100

theorem pear_juice_percentage :
  let pear_yield := JuiceYield.mk "Pear" 3 8
  let orange_yield := JuiceYield.mk "Orange" 2 10
  let blend_pears := 4
  let blend_oranges := 4
  let result := juice_percentage pear_yield orange_yield blend_pears blend_oranges
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |result - 34.78| < ε := by
  sorry

#eval juice_percentage (JuiceYield.mk "Pear" 3 8) (JuiceYield.mk "Orange" 2 10) 4 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_juice_percentage_l1227_122781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_equal_tangents_l1227_122736

-- Define the centers and radii of the two given circles
variable (O₁ O₂ : ℝ × ℝ)
variable (r₁ r₂ : ℝ)

-- Define the lengths of the given segments
variable (l₁ l₂ : ℝ)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the tangent length function
noncomputable def tangentLength (p c : ℝ × ℝ) (r : ℝ) : ℝ :=
  Real.sqrt ((distance p c)^2 - r^2)

-- Theorem statement
theorem exists_point_with_equal_tangents :
  ∃ P : ℝ × ℝ,
    distance P O₁ = Real.sqrt (r₁^2 + l₁^2) ∧
    distance P O₂ = Real.sqrt (r₂^2 + l₂^2) ∧
    tangentLength P O₁ r₁ = l₁ ∧
    tangentLength P O₂ r₂ = l₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_with_equal_tangents_l1227_122736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l1227_122751

theorem sum_of_divisors_143 : (Finset.filter (λ x => 143 % x = 0) (Finset.range 144)).sum id = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l1227_122751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part_1_solution_part_2_l1227_122721

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := |x + 2*a| + |x - 1|

-- Define the function g
noncomputable def g (a : ℝ) : ℝ := f a (1/a)

-- Theorem for part (1)
theorem solution_part_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part (2)
theorem solution_part_2 :
  {a : ℝ | a ≠ 0 ∧ g a ≤ 4} = {a : ℝ | 1/2 ≤ a ∧ a ≤ 3/2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part_1_solution_part_2_l1227_122721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l1227_122748

/-- Represents a parabola with equation ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (- p.b / (2 * p.a), p.c - p.b^2 / (4 * p.a))

/-- Check if a point is on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- The axis of symmetry is vertical if the coefficient of x^2 is non-zero -/
def vertical_axis (p : Parabola) : Prop :=
  p.a ≠ 0

theorem parabola_equation_correct : 
  ∃ (p : Parabola), 
    p.a = -1/3 ∧ p.b = 2 ∧ p.c = 2 ∧
    vertex p = (3, 5) ∧
    vertical_axis p ∧
    contains_point p 0 2 := by
  sorry

#eval "Parabola theorem compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l1227_122748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1227_122762

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x) * Real.sin x

-- State the theorem
theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1227_122762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_sin_4x_plus_pi_div_2_l1227_122720

/-- The phase shift of the function y = sin(4x + π/2) -/
noncomputable def phase_shift (f : ℝ → ℝ) : ℝ :=
  Real.pi / 8

/-- The given function -/
noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (4 * x + Real.pi / 2)

theorem phase_shift_of_sin_4x_plus_pi_div_2 :
  phase_shift f = Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_sin_4x_plus_pi_div_2_l1227_122720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_work_hours_l1227_122737

/-- Represents Thomas's savings scenario over two years --/
structure SavingsScenario where
  allowance_per_week : ℚ
  hourly_wage : ℚ
  car_cost : ℚ
  weekly_spending : ℚ
  remaining_needed : ℚ
  weeks_per_year : ℕ

/-- Calculates the approximate number of weekly work hours --/
noncomputable def calculate_work_hours (s : SavingsScenario) : ℚ :=
  let first_year_savings := s.allowance_per_week * s.weeks_per_year
  let second_year_savings := s.car_cost - first_year_savings - s.remaining_needed
  let total_second_year_earnings := second_year_savings + (s.weekly_spending * s.weeks_per_year)
  let total_hours := total_second_year_earnings / s.hourly_wage
  total_hours / s.weeks_per_year

/-- Thomas's specific savings scenario --/
def thomas_scenario : SavingsScenario := {
  allowance_per_week := 50
  hourly_wage := 9
  car_cost := 15000
  weekly_spending := 35
  remaining_needed := 2000
  weeks_per_year := 52
}

/-- Theorem stating that Thomas works approximately 24 hours per week --/
theorem thomas_work_hours :
  ⌊calculate_work_hours thomas_scenario⌋ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thomas_work_hours_l1227_122737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_scanning_codes_count_independent_squares_count_symmetric_codes_from_independent_squares_l1227_122700

/-- A symmetric scanning code on an 8x8 grid. -/
structure SymmetricScanningCode where
  grid : Fin 8 → Fin 8 → Bool
  symmetry : ∀ (i j : Fin 8),
    grid i j = grid (7 - i) j ∧
    grid i j = grid i (7 - j) ∧
    grid i j = grid j i
  has_both_colors : (∃ i j, grid i j = true) ∧ (∃ i j, grid i j = false)

/-- The number of symmetric scanning codes on an 8x8 grid. -/
def num_symmetric_scanning_codes : ℕ :=
  62 -- We directly define this as 62 based on our calculation

/-- The theorem stating that there are 62 symmetric scanning codes. -/
theorem symmetric_scanning_codes_count :
  num_symmetric_scanning_codes = 62 := by
  -- Proof goes here
  sorry

/-- A function to count the number of independent squares in the grid -/
def count_independent_squares : ℕ := 6

/-- Theorem stating that there are 6 independent squares -/
theorem independent_squares_count :
  count_independent_squares = 6 := by
  -- Proof goes here
  sorry

/-- Theorem relating the number of symmetric scanning codes to independent squares -/
theorem symmetric_codes_from_independent_squares :
  num_symmetric_scanning_codes = 2^count_independent_squares - 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_scanning_codes_count_independent_squares_count_symmetric_codes_from_independent_squares_l1227_122700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_square_field_time_l1227_122712

noncomputable section

-- Define the field side length
def side_length : ℝ := 55

-- Define the speeds for each side (in m/s)
def speed1 : ℝ := 9 * 1000 / 3600
def speed2 : ℝ := 7 * 1000 / 3600 * 0.9
def speed3 : ℝ := 11 * 1000 / 3600
def speed4 : ℝ := 5 * 1000 / 3600

-- Define the additional time for water crossings
def water_crossing_time : ℝ := 30

-- Define the reduced speed for the last 20 meters of the fourth side
def reduced_speed4 : ℝ := speed4 * 0.8

end noncomputable section

-- Define the theorem
theorem run_around_square_field_time :
  let time1 := side_length / speed1
  let time2 := side_length / speed2
  let time3 := side_length / speed3 + water_crossing_time
  let time4 := (side_length - 20) / speed4 + 20 / reduced_speed4
  let total_time := time1 + time2 + time3 + time4
  ∃ ε > 0, |total_time - 144.63| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_square_field_time_l1227_122712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_distribution_α_7_l1227_122733

/-- Exponential distribution with parameter α -/
structure ExponentialDistribution where
  α : ℝ
  α_pos : α > 0

/-- Probability Density Function (PDF) for Exponential Distribution -/
noncomputable def pdf (d : ExponentialDistribution) (x : ℝ) : ℝ :=
  if x < 0 then 0 else d.α * Real.exp (-d.α * x)

/-- Cumulative Distribution Function (CDF) for Exponential Distribution -/
noncomputable def cdf (d : ExponentialDistribution) (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else 1 - Real.exp (-d.α * x)

/-- Theorem: PDF and CDF for Exponential Distribution with α = 7 -/
theorem exponential_distribution_α_7 :
  let d : ExponentialDistribution := ⟨7, by norm_num⟩
  (∀ x, pdf d x = if x < 0 then 0 else 7 * Real.exp (-7 * x)) ∧
  (∀ x, cdf d x = if x ≤ 0 then 0 else 1 - Real.exp (-7 * x)) := by
  sorry

#check exponential_distribution_α_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_distribution_α_7_l1227_122733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_5300_l1227_122774

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a mapping from Chinese characters to digits -/
def ChineseCharToDigit := Fin 7 → Digit

/-- Four-digit number represented by Chinese characters -/
def fourDigitNumber (f : ChineseCharToDigit) : ℕ :=
  5000 + (f 0).val * 100 + (f 0).val * 10 + (f 0).val

/-- Three-digit number represented by Chinese characters -/
def threeDigitNumber (f : ChineseCharToDigit) : ℕ :=
  300 + (f 1).val * 10 + (f 2).val

/-- The sum of the four-digit and three-digit numbers -/
def sum (f : ChineseCharToDigit) : ℕ :=
  fourDigitNumber f + threeDigitNumber f

/-- Theorem stating that the sum equals 5300 -/
theorem sum_equals_5300 (f : ChineseCharToDigit) 
  (h1 : fourDigitNumber f = 5111)
  (h2 : threeDigitNumber f = 189)
  (h3 : ∀ i j, i ≠ j → f i ≠ f j) : 
  sum f = 5300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_5300_l1227_122774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l1227_122768

theorem election_votes_theorem (total_votes : ℕ) (invalid_percent : ℚ) 
  (a_exceed_b_percent : ℚ) (c_percent : ℚ) 
  (h_total : total_votes = 5720)
  (h_invalid : invalid_percent = 1/4)
  (h_a_exceed_b : a_exceed_b_percent = 3/20)
  (h_c : c_percent = 1/10) : 
  let valid_votes := total_votes - (invalid_percent * total_votes).floor
  let c_votes := (c_percent * total_votes).floor
  let b_votes := ((valid_votes - c_votes : ℚ) / 2 - (a_exceed_b_percent * total_votes) / 2).floor
  b_votes + c_votes = 2002 := by
  sorry

#check election_votes_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_theorem_l1227_122768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stella_has_five_glasses_l1227_122727

/-- Represents the inventory and financial data of Stella's antique shop. -/
structure AntiqueShop where
  num_dolls : ℕ
  num_clocks : ℕ
  num_glasses : ℕ
  doll_price : ℕ
  clock_price : ℕ
  glass_price : ℕ
  total_cost : ℕ
  total_profit : ℕ

/-- Calculates the total revenue from selling all items in the shop. -/
def total_revenue (shop : AntiqueShop) : ℕ :=
  shop.num_dolls * shop.doll_price +
  shop.num_clocks * shop.clock_price +
  shop.num_glasses * shop.glass_price

/-- Theorem stating that Stella has 5 glasses for sale. -/
theorem stella_has_five_glasses (shop : AntiqueShop)
  (h1 : shop.num_dolls = 3)
  (h2 : shop.num_clocks = 2)
  (h3 : shop.doll_price = 5)
  (h4 : shop.clock_price = 15)
  (h5 : shop.glass_price = 4)
  (h6 : shop.total_cost = 40)
  (h7 : shop.total_profit = 25)
  (h8 : total_revenue shop = shop.total_cost + shop.total_profit) :
  shop.num_glasses = 5 := by
  sorry

-- Remove the #eval statement as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stella_has_five_glasses_l1227_122727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l1227_122785

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.log x + Real.pi / 4)

-- State the theorem
theorem infinitely_many_zeros :
  ∃ S : Set ℝ, S.Infinite ∧ (∀ x ∈ S, 0 < x ∧ x < 1 ∧ f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_zeros_l1227_122785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_lotto_profit_l1227_122796

/-- Calculates the profit from buying lotto tickets given specific conditions. -/
def lotto_profit (total_tickets : ℕ) (ticket_cost : ℚ) (winner_percentage : ℚ) 
  (five_dollar_winner_percentage : ℚ) (grand_prize : ℚ) (other_winner_average : ℚ) : ℚ :=
  let total_cost := (total_tickets : ℚ) * ticket_cost
  let winning_tickets := ⌊winner_percentage * (total_tickets : ℚ)⌋
  let five_dollar_winners := ⌊five_dollar_winner_percentage * (winning_tickets : ℚ)⌋
  let five_dollar_winnings := (five_dollar_winners : ℚ) * 5
  let remaining_winners := (winning_tickets - five_dollar_winners - 1 : ℚ)
  let other_winnings := remaining_winners * other_winner_average
  let total_winnings := five_dollar_winnings + grand_prize + other_winnings
  total_winnings - total_cost

/-- Proves that James's profit from buying lotto tickets is $4,830 given the specific conditions. -/
theorem james_lotto_profit : 
  lotto_profit 200 2 (1/5) (4/5) 5000 10 = 4830 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_lotto_profit_l1227_122796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_lucas_150_mod_5_l1227_122732

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem modified_lucas_150_mod_5 :
  modifiedLucas 149 % 5 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_lucas_150_mod_5_l1227_122732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sincos_function_l1227_122767

open Real Set

theorem min_value_of_sincos_function (a : ℝ) :
  (∀ x ∈ Ioo 0 (π/4), sin (2*x) + a*cos (2*x) ≥ a) →
  (∃ x ∈ Icc 0 (π/4), sin (2*x) + a*cos (2*x) = a) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sincos_function_l1227_122767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l1227_122708

/-- Probability of getting heads for the unfair coin -/
noncomputable def p : ℝ := 3/4

/-- Number of coin tosses -/
def n : ℕ := 100

/-- Probability of getting an even number of heads after k tosses -/
noncomputable def P (k : ℕ) : ℝ :=
  1/2 * (1 + (1/4)^k)

/-- Main theorem: Probability of even number of heads after n tosses -/
theorem even_heads_probability : P n = 1/2 * (1 + (1/4)^n) := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l1227_122708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_size_theorem_l1227_122745

/-- Represents the size of the candidate pool -/
def k : ℕ → Prop := λ k => k > 2

/-- Represents the new committee size -/
def n : ℕ → Prop := λ n => n > 3

/-- The probability of selecting two specific individuals for a 3-member committee -/
def prob_3 (k : ℕ) : ℚ := 6 / (k * (k - 1))

/-- The probability of selecting two specific individuals for an n-member committee -/
def prob_n (k n : ℕ) : ℚ := (n * (n - 1)) / (k * (k - 1))

/-- The theorem stating that if the probability ratio is 40, then n must be 16 -/
theorem committee_size_theorem {k n : ℕ} (hk : k > 2) (hn : n > 3) 
  (h : prob_n k n = 40 * prob_3 k) : n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_size_theorem_l1227_122745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1227_122738

-- Define opposite numbers
def opposite (x y : ℝ) : Prop := x = -y

-- Define congruent triangles (simplified)
def congruent (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define scalene triangle (simplified)
def scalene (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define equal internal angles (simplified)
def equal_angles (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define area function (simplified)
noncomputable def area (t : Set (ℝ × ℝ)) : ℝ := sorry

theorem problem_solution :
  -- 1. Converse of "If x + y = 0, then x and y are opposite numbers" is true
  (∀ x y : ℝ, opposite x y → x + y = 0) ∧
  -- 2. Contrapositive of "If q ≤ 1, then x^2 + 2x + q = 0 has real roots" is true
  (∀ q : ℝ, (¬∃ x : ℝ, x^2 + 2*x + q = 0) → q > 1) ∧
  -- 3. Negation of "Congruent triangles have equal areas" is false
  (¬(∀ t1 t2 : Set (ℝ × ℝ), ¬(congruent t1 t2) → area t1 ≠ area t2)) ∧
  -- 4. Converse of "A scalene triangle has three equal internal angles" is false
  (¬(∀ t : Set (ℝ × ℝ), equal_angles t → scalene t)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1227_122738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1227_122731

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0), 
    if one of its asymptotes is perpendicular to the line x + 2y + 1 = 0, 
    then its eccentricity is √5. -/
theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_perpendicular : ∃ (k : ℝ), k * (b / a) = -1/2) : 
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1227_122731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l1227_122726

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = 0

-- Define the point through which the chords pass
def P : ℝ × ℝ := (3, 5)

-- Define the longest chord AC
def AC (A C : ℝ × ℝ) : Prop :=
  circle_equation A.1 A.2 ∧ circle_equation C.1 C.2 ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)) ∧
  ∀ X Y : ℝ × ℝ, circle_equation X.1 X.2 → circle_equation Y.1 Y.2 →
    (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (s * X.1 + (1 - s) * Y.1, s * X.2 + (1 - s) * Y.2)) →
    (A.1 - C.1)^2 + (A.2 - C.2)^2 ≥ (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Define the shortest chord BD
def BD (B D : ℝ × ℝ) : Prop :=
  circle_equation B.1 B.2 ∧ circle_equation D.1 D.2 ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)) ∧
  ∀ X Y : ℝ × ℝ, circle_equation X.1 X.2 → circle_equation Y.1 Y.2 →
    (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (s * X.1 + (1 - s) * Y.1, s * X.2 + (1 - s) * Y.2)) →
    (B.1 - D.1)^2 + (B.2 - D.2)^2 ≤ (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  let AC := ((A.1 - C.1)^2 + (A.2 - C.2)^2).sqrt
  let BD := ((B.1 - D.1)^2 + (B.2 - D.2)^2).sqrt
  (1/2) * AC * BD

-- Theorem statement
theorem area_of_quadrilateral (A B C D : ℝ × ℝ) :
  AC A C → BD B D → (area_quadrilateral A B C D = 20 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l1227_122726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_problem_l1227_122798

/-- A binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ

/-- The probability of a binomial random variable being greater than or equal to k -/
noncomputable def P_ge (X : BinomialDistribution) (k : ℕ) : ℝ := sorry

/-- The statement to prove -/
theorem binomial_probability_problem (p : ℝ) 
  (ξ : BinomialDistribution) (η : BinomialDistribution)
  (hξ : ξ = ⟨2, p⟩)
  (hη : η = ⟨3, p⟩)
  (h_prob : P_ge ξ 1 = 5/9) :
  P_ge η 1 = 19/27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_problem_l1227_122798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_X_equals_band_Y_l1227_122788

/-- The length of a band surrounding four touching unit circles in a row -/
noncomputable def band_X : ℝ := 12 + 2 * Real.pi

/-- The length of a band surrounding six touching unit circles in a hexagon -/
noncomputable def band_Y : ℝ := 12 + 2 * Real.pi

/-- Theorem stating that the lengths of band X and band Y are equal -/
theorem band_X_equals_band_Y : band_X = band_Y := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_X_equals_band_Y_l1227_122788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1227_122757

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1) + x * Real.exp x

theorem tangent_line_at_zero_one :
  let p : ℝ × ℝ := (0, 1)
  let tangent_line := λ x : ℝ ↦ 1
  (∀ x : ℝ, HasDerivAt f (tangent_line x - tangent_line 0) x) ∧
  f 0 = 1 ∧
  tangent_line 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_one_l1227_122757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_l1227_122735

/-- Represents a cube with 6 faces -/
structure Cube where
  faces : Fin 6 → ℕ

/-- The set of numbers that can be on the faces of each cube -/
def cube_numbers : Finset ℕ := {1, 3, 9, 27, 81, 243}

/-- A function to calculate the sum of visible faces when stacking cubes -/
def visible_sum (cubes : Fin 5 → Cube) : ℕ := sorry

/-- The theorem stating the maximum sum of visible numbers -/
theorem max_visible_sum :
  ∃ (cubes : Fin 5 → Cube),
    (∀ i : Fin 5, ∀ j : Fin 6, (cubes i).faces j ∈ cube_numbers) ∧
    (∀ i : Fin 5, (Finset.card (Finset.filter (λ j => (cubes i).faces j ∈ cube_numbers) (Finset.univ : Finset (Fin 6))) = 6)) ∧
    visible_sum cubes = 1815 ∧
    ∀ (other_cubes : Fin 5 → Cube),
      (∀ i : Fin 5, ∀ j : Fin 6, (other_cubes i).faces j ∈ cube_numbers) →
      (∀ i : Fin 5, (Finset.card (Finset.filter (λ j => (other_cubes i).faces j ∈ cube_numbers) (Finset.univ : Finset (Fin 6))) = 6)) →
      visible_sum other_cubes ≤ 1815 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_l1227_122735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1227_122761

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (8*x) + Real.sin (4*x)) / (Real.cos (5*x) + Real.cos x) = 6 * |Real.sin (2*x)| ↔ 
  (∃ k : ℤ, x = k * Real.pi) ∨ 
  (∃ k : ℤ, x = Real.arccos ((3 - Real.sqrt 13) / 4) + 2 * k * Real.pi) ∨ 
  (∃ k : ℤ, x = -Real.arccos ((3 - Real.sqrt 13) / 4) + 2 * k * Real.pi) ∨
  (∃ k : ℤ, x = Real.arccos ((Real.sqrt 13 - 3) / 4) + 2 * k * Real.pi) ∨ 
  (∃ k : ℤ, x = -Real.arccos ((Real.sqrt 13 - 3) / 4) + 2 * k * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1227_122761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l1227_122770

def number_of_distributions (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

theorem ball_distribution (n k : ℕ) (h : n > 0 ∧ k > 0) : 
  number_of_distributions n k = Nat.choose (n - 1) (k - 1) :=
by
  unfold number_of_distributions
  rfl

#eval number_of_distributions 7 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_l1227_122770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_trig_function_ratio_l1227_122780

-- Problem 1
theorem sin_cos_relation (α : ℝ) 
  (h : (Real.sin α + 3 * Real.cos α) / (3 * Real.cos α - Real.sin α) = 5) : 
  Real.sin α ^ 2 - Real.sin α * Real.cos α = 2 / 5 := by sorry

-- Problem 2
theorem trig_function_ratio (α : ℝ) 
  (h : ∃ (x y : ℝ), x = -4 ∧ y = 3 ∧ Real.tan α = y / x) :
  (Real.cos (π / 2 + α) * Real.sin (-π - α)) / (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_trig_function_ratio_l1227_122780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_equation_l1227_122749

/-- The product of the first n odd numbers -/
def oddProduct (n : ℕ) : ℕ :=
  List.prod (List.range n |>.map (fun i => 2 * i + 1))

/-- The product of n consecutive integers starting from n+1 -/
def consecutiveProduct (n : ℕ) : ℕ :=
  List.prod (List.range n |>.map (fun i => n + i + 1))

/-- The nth equation in the pattern -/
theorem nth_equation (n : ℕ) : consecutiveProduct n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_equation_l1227_122749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_labeling_l1227_122746

-- Define a tetrahedron as a structure with 4 vertices
structure Tetrahedron :=
  (v1 v2 v3 v4 : ℕ)

-- Define a predicate for valid labeling
def ValidLabeling (t : Tetrahedron) : Prop :=
  t.v1 ∈ Finset.range 4 ∧
  t.v2 ∈ Finset.range 4 ∧
  t.v3 ∈ Finset.range 4 ∧
  t.v4 ∈ Finset.range 4 ∧
  t.v1 ≠ t.v2 ∧ t.v1 ≠ t.v3 ∧ t.v1 ≠ t.v4 ∧
  t.v2 ≠ t.v3 ∧ t.v2 ≠ t.v4 ∧
  t.v3 ≠ t.v4

-- Define a predicate for equal face sums
def EqualFaceSums (t : Tetrahedron) : Prop :=
  t.v1 + t.v2 + t.v3 = t.v1 + t.v2 + t.v4 ∧
  t.v1 + t.v2 + t.v3 = t.v1 + t.v3 + t.v4 ∧
  t.v1 + t.v2 + t.v3 = t.v2 + t.v3 + t.v4

-- Theorem statement
theorem no_valid_labeling :
  ¬ ∃ t : Tetrahedron, ValidLabeling t ∧ EqualFaceSums t :=
by
  intro h
  rcases h with ⟨t, ⟨valid, equal⟩⟩
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_labeling_l1227_122746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_value_l1227_122755

def alphabet_value (n : Nat) : Int :=
  match n % 16 with
  | 1 => 1
  | 3 => 2
  | 5 => 1
  | 7 => -1
  | 9 => -2
  | 11 => -1
  | 13 => 1
  | 15 => 2
  | _ => 0

def letter_position (c : Char) : Nat :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0

def word_value (word : String) : Int :=
  word.toList.map (λ c => alphabet_value (letter_position c)) |>.sum

theorem pattern_value : word_value "PATTERN" = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_value_l1227_122755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_derivative_sum_l1227_122752

open Real BigOperators Finset

theorem binomial_expansion_derivative_sum 
  (x : ℝ) (a : Fin 11 → ℝ) :
  ((1 - 2*x)^10 = ∑ i in range 11, a i * x^i) →
  (∑ i in range 11, i * a i) = 20 := by
  sorry

#check binomial_expansion_derivative_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_derivative_sum_l1227_122752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_iff_q_l1227_122777

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  side_angle_correspondence : 
    a / Real.sin A = b / Real.sin B ∧ 
    b / Real.sin B = c / Real.sin C

-- Define propositions p and q
def p (t : Triangle) : Prop :=
  t.B + t.C = 2 * t.A ∧ t.b + t.c = 2 * t.a

def q (t : Triangle) : Prop :=
  t.A = t.B ∧ t.B = t.C ∧ t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem p_iff_q (t : Triangle) : p t ↔ q t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_iff_q_l1227_122777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_coefficient_sum_l1227_122759

noncomputable def ellipse_x (t : ℝ) : ℝ := (3 * (Real.sin t - 2)) / (3 - Real.cos t)
noncomputable def ellipse_y (t : ℝ) : ℝ := (4 * (Real.cos t - 4)) / (3 - Real.cos t)

def are_integers (A B C D E F : ℤ) : Prop := True

def gcd_is_one (A B C D E F : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D)) (Int.natAbs E)) (Int.natAbs F) = 1

theorem ellipse_coefficient_sum :
  ∃ (A B C D E F : ℤ),
    (∀ (x y : ℝ), (∃ t, x = ellipse_x t ∧ y = ellipse_y t) ↔
      A * x^2 + B * x * y + C * y^2 + D * x + E * y + ↑F = 0) ∧
    are_integers A B C D E F ∧
    gcd_is_one A B C D E F ∧
    Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1383 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_coefficient_sum_l1227_122759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juanico_age_fraction_l1227_122715

/-- The fraction of Gladys' age that Juanico is less than -/
def fraction_of_gladys_age : ℚ := 1/2

theorem juanico_age_fraction (gladys_age : ℕ) (juanico_age : ℕ) : 
  gladys_age = 30 →
  juanico_age = (↑fraction_of_gladys_age : ℚ) * gladys_age - 4 →
  juanico_age + 30 = 41 →
  fraction_of_gladys_age = 1/2 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry

#check juanico_age_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juanico_age_fraction_l1227_122715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_necessary_not_sufficient_l1227_122778

theorem a_gt_b_necessary_not_sufficient (a b : ℝ) :
  (∀ c : ℝ, a * c^2 > b * c^2 → a > b) ∧ 
  (∃ c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_necessary_not_sufficient_l1227_122778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_max_marks_l1227_122791

/-- The maximum marks for an exam, given the passing percentage and a student's performance. -/
theorem exam_max_marks (passing_percentage : ℚ) (student_score shortfall : ℕ) : 
  passing_percentage = 3/10 →
  student_score = 30 →
  shortfall = 36 →
  (passing_percentage * (student_score + shortfall : ℚ)) = (student_score + shortfall : ℚ) →
  (student_score + shortfall : ℚ) / passing_percentage = 220 := by
  sorry

#eval (30 + 36 : ℚ) / (3/10 : ℚ)  -- Expected output: 220

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_max_marks_l1227_122791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_repeated_digits_l1227_122775

theorem divisibility_of_repeated_digits :
  (∃ (k : ℕ) (m : ℕ), (1988 : ℕ) ∣ (1989 * (10^(4*k) - 1) / 9) * 10^m) ∧
  (∃ (n : ℕ), (1989 : ℕ) ∣ (1988 * (10^(4*n) - 1) / 9)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_repeated_digits_l1227_122775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_integer_and_divisibility_l1227_122707

noncomputable def a (n : ℕ) : ℝ := ((2 + Real.sqrt 3) ^ n - (2 - Real.sqrt 3) ^ n) / (2 * Real.sqrt 3)

theorem a_is_integer_and_divisibility (n : ℕ) : 
  (∃ k : ℤ, a n = k) ∧ (3 ∣ (Int.floor (a n)) ↔ 3 ∣ n) := by sorry

#check a_is_integer_and_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_integer_and_divisibility_l1227_122707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tetrahedron_possible_l1227_122703

-- Define the triangle types
noncomputable def Triangle1 : ℝ × ℝ × ℝ := (3, 4, 5)
noncomputable def Triangle2 : ℝ × ℝ × ℝ := (4, 5, Real.sqrt 41)
noncomputable def Triangle3 : ℝ × ℝ × ℝ := ((5/6) * Real.sqrt 2, 4, 5)

-- Define the number of each triangle type
def NumTriangle1 : ℕ := 2
def NumTriangle2 : ℕ := 4
def NumTriangle3 : ℕ := 6

-- Define a tetrahedron as a collection of four triangles
def Tetrahedron := List (ℝ × ℝ × ℝ)

-- Function to check if a list of triangles forms a valid tetrahedron
def isValidTetrahedron (triangles : List (ℝ × ℝ × ℝ)) : Prop :=
  sorry

-- Function to count valid tetrahedra
def countValidTetrahedra (triangles1 : List (ℝ × ℝ × ℝ)) 
                         (triangles2 : List (ℝ × ℝ × ℝ)) 
                         (triangles3 : List (ℝ × ℝ × ℝ)) : ℕ :=
  sorry

-- Theorem statement
theorem one_tetrahedron_possible :
  countValidTetrahedra 
    (List.replicate NumTriangle1 Triangle1)
    (List.replicate NumTriangle2 Triangle2)
    (List.replicate NumTriangle3 Triangle3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_tetrahedron_possible_l1227_122703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1227_122793

noncomputable def f (x : ℝ) : ℝ := (15*x^4 + 5*x^3 + 7*x^2 + 6*x + 2) / (5*x^4 + 3*x^3 + 10*x^2 + 4*x + 1)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l1227_122793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1227_122717

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

def dom : Set ℝ := Set.Ioo (-1) 1

theorem f_properties :
  (∀ x, x ∈ dom → f (-x) = -f x) ∧  -- f is odd
  (f (1/2) = 2/5) ∧  -- f(1/2) = 2/5
  (∀ x y, x ∈ dom → y ∈ dom → x < y → f x < f y) ∧  -- f is increasing on (-1,1)
  (∀ t : ℝ, f (t-1) + f t < 0 ↔ 0 < t ∧ t < 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1227_122717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_bid_problem_l1227_122795

/-- Original price in yuan -/
noncomputable def original_price : ℝ := 25

/-- Original annual sales volume in pieces -/
noncomputable def original_sales : ℝ := 80000

/-- Price increase in yuan -/
noncomputable def price_increase : ℝ := 1

/-- Sales volume decrease per yuan of price increase -/
noncomputable def sales_decrease : ℝ := 2000

/-- Investment cost for innovation in millions of yuan -/
noncomputable def innovation_cost (x : ℝ) : ℝ := (1/6) * (x^2 - 600)

/-- Publicity expenses in millions of yuan -/
noncomputable def publicity_cost (x : ℝ) : ℝ := 50 + 2*x

/-- Revenue function before price change -/
noncomputable def revenue_before (t : ℝ) : ℝ := t * (original_sales - (t - original_price) * sales_decrease)

/-- Revenue function after reform -/
noncomputable def revenue_after (x a : ℝ) : ℝ := a * x

/-- Total investment -/
noncomputable def total_investment (x : ℝ) : ℝ := innovation_cost x + publicity_cost x

theorem olympic_bid_problem :
  (∃ t : ℝ, t ≤ 40 ∧ ∀ s : ℝ, s > 40 → revenue_before s < revenue_before original_price) ∧
  (∃ a : ℝ, a ≥ 12 ∧ ∀ x : ℝ, x > 25 → 
    revenue_after x a ≥ revenue_before original_price + total_investment x) ∧
  (∃ x : ℝ, x = 30 ∧ revenue_after x 12 = revenue_before original_price + total_investment x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_bid_problem_l1227_122795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1227_122797

theorem equation_solution (a : ℝ) (x : ℝ) (k : ℤ) :
  a ≠ 0 →
  x ≠ π / 2 + π * (k : ℝ) →
  (a * (1 / Real.cos x - Real.tan x) = 1) ↔
  (|a| ≥ 1 ∧ x = Real.arccos (a / Real.sqrt (a^2 + 1)) + 2 * π * (k : ℝ)) ∨
  ((a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) ∧ x = -Real.arccos (a / Real.sqrt (a^2 + 1)) + 2 * π * (k : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1227_122797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arcsin_sin_arccos_tan_arccot_cot_arctan_cos_arctan_sin_arctan_cos_arccot_sin_arccot_l1227_122728

-- Define the domain conditions for inverse trigonometric functions
def arcsin_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def arccos_domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def arctan_domain (x : ℝ) : Prop := True  -- arctan is defined for all real numbers
def arccot_domain (x : ℝ) : Prop := True  -- arccot is defined for all real numbers

-- Theorem statements
theorem cos_arcsin (x : ℝ) (h : arcsin_domain x) : 
  Real.cos (Real.arcsin x) = Real.sqrt (1 - x^2) := by sorry

theorem sin_arccos (x : ℝ) (h : arccos_domain x) : 
  Real.sin (Real.arccos x) = Real.sqrt (1 - x^2) := by sorry

theorem tan_arccot (x : ℝ) (h : arccot_domain x) (hx : x ≠ 0) : 
  Real.tan (Real.arccos (1 / Real.sqrt (1 + x^2))) = 1 / x := by sorry

theorem cot_arctan (x : ℝ) (h : arctan_domain x) (hx : x ≠ 0) : 
  (1 / Real.tan (Real.arctan x)) = 1 / x := by sorry

theorem cos_arctan (x : ℝ) (h : arctan_domain x) : 
  Real.cos (Real.arctan x) = 1 / Real.sqrt (1 + x^2) := by sorry

theorem sin_arctan (x : ℝ) (h : arctan_domain x) : 
  Real.sin (Real.arctan x) = x / Real.sqrt (1 + x^2) := by sorry

theorem cos_arccot (x : ℝ) (h : arccot_domain x) : 
  Real.cos (Real.arccos (x / Real.sqrt (1 + x^2))) = x / Real.sqrt (1 + x^2) := by sorry

theorem sin_arccot (x : ℝ) (h : arccot_domain x) : 
  Real.sin (Real.arccos (x / Real.sqrt (1 + x^2))) = 1 / Real.sqrt (1 + x^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_arcsin_sin_arccos_tan_arccot_cot_arctan_cos_arctan_sin_arctan_cos_arccot_sin_arccot_l1227_122728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_velocity_with_5a_propellant_l1227_122734

/-- The maximum velocity formula for a single-stage rocket -/
noncomputable def rocket_velocity (v₀ m₁ m₂ : ℝ) : ℝ := v₀ * Real.log ((m₁ + m₂) / m₁)

/-- The theorem stating the maximum velocity of the rocket with 5a kg of propellant -/
theorem rocket_velocity_with_5a_propellant (a : ℝ) (h₁ : a > 0) :
  ∃ v₀ : ℝ, 
    rocket_velocity v₀ a (3 * a) = 2.8 → 
    rocket_velocity v₀ a (5 * a) = 3.6 := by
  sorry

#check rocket_velocity_with_5a_propellant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_velocity_with_5a_propellant_l1227_122734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_after_shrink_is_twenty_l1227_122771

/-- Represents the shrink ray device that reduces size by 50% -/
def shrink_factor : ℚ := 1/2

/-- Number of coffee cups -/
def num_cups : ℕ := 5

/-- Initial amount of coffee in each cup (in ounces) -/
def coffee_per_cup : ℚ := 8

/-- Calculates the total amount of coffee after shrinking -/
def total_coffee_after_shrink : ℚ := shrink_factor * (num_cups : ℚ) * coffee_per_cup

/-- Theorem stating that the total amount of coffee after shrinking is 20 ounces -/
theorem coffee_after_shrink_is_twenty :
  total_coffee_after_shrink = 20 := by
  -- Unfold the definition of total_coffee_after_shrink
  unfold total_coffee_after_shrink
  -- Simplify the arithmetic
  simp [shrink_factor, num_cups, coffee_per_cup]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_after_shrink_is_twenty_l1227_122771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_proportional_l1227_122773

/-- Represents the number of students in each grade --/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Represents the sample size for each grade --/
structure GradeSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The total number of students --/
def totalStudents (gp : GradePopulation) : ℕ :=
  gp.freshmen + gp.sophomores + gp.juniors + gp.seniors

/-- The sampling fraction --/
def samplingFraction (totalSample : ℕ) (gp : GradePopulation) : ℚ :=
  totalSample / (totalStudents gp)

/-- Calculates the expected sample size for a grade --/
def expectedSampleSize (gradeSize : ℕ) (totalSample : ℕ) (gp : GradePopulation) : ℕ :=
  (gradeSize * totalSample) / (totalStudents gp)

/-- Theorem: The sample sizes are proportional to the grade populations --/
theorem stratified_sampling_proportional
  (gp : GradePopulation)
  (totalSample : ℕ)
  (h_gp : gp = GradePopulation.mk 1600 3200 2000 1200)
  (h_total : totalSample = 400) :
  let expected := GradeSample.mk
    (expectedSampleSize gp.freshmen totalSample gp)
    (expectedSampleSize gp.sophomores totalSample gp)
    (expectedSampleSize gp.juniors totalSample gp)
    (expectedSampleSize gp.seniors totalSample gp)
  expected = GradeSample.mk 80 160 100 60 := by
  sorry

#check stratified_sampling_proportional

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_proportional_l1227_122773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_given_sin_l1227_122750

theorem cos_double_angle_given_sin (θ : Real) (h : Real.sin θ = 3/5) : Real.cos (2*θ) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_given_sin_l1227_122750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1227_122779

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3*2^x + 5

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

theorem problem_solution :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ 137/32) ∧ 
  (f (-2) = 137/32) ∧
  (∃ a > 0, a ≠ 1 ∧ (∀ x ∈ Set.Icc (-2 : ℝ) 2, g a x ≤ 14) ∧ 
   (g a 2 = 14 ∨ g a (-2) = 14) ∧ (a = Real.sqrt 3 ∨ a = Real.sqrt 3 / 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1227_122779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mayo_bulk_savings_l1227_122782

/-- Proves the savings when buying mayo in bulk at Costco compared to a normal store. -/
theorem mayo_bulk_savings (
  costco_price : ℝ)
  (normal_store_price : ℝ)
  (ounces_per_gallon : ℝ)
  (ounces_per_bottle : ℝ)
  (costco_price_condition : costco_price = 8)
  (normal_store_price_condition : normal_store_price = 3)
  (ounces_per_gallon_condition : ounces_per_gallon = 128)
  (ounces_per_bottle_condition : ounces_per_bottle = 16) :
  (ounces_per_gallon / ounces_per_bottle * normal_store_price) - costco_price = 16 := by
  sorry

#check mayo_bulk_savings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mayo_bulk_savings_l1227_122782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_ceiling_sum_5_to_39_l1227_122786

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℕ := Int.toNat (Int.ceil x)

-- Define our sum function
noncomputable def sqrtCeilingSum (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => ceiling (Real.sqrt (a + i)))

-- State the theorem
theorem sqrt_ceiling_sum_5_to_39 :
  sqrtCeilingSum 5 39 = 175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_ceiling_sum_5_to_39_l1227_122786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_spheres_count_l1227_122704

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ
  
/-- Represents a sphere -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents the configuration of spheres in the truncated cone -/
structure SphereConfiguration where
  cone : TruncatedCone
  o1 : Sphere
  o2 : Sphere

/-- Checks if a sphere configuration is valid -/
def is_valid_configuration (config : SphereConfiguration) : Prop :=
  config.cone.height = 8 ∧
  config.o1.radius = 2 ∧
  config.o2.radius = 3 ∧
  -- O₁ is tangent to upper base and side surface
  -- O₁'s center lies on the axis of truncated cone
  -- O₂ is tangent to O₁, lower base, and side surface
  True  -- placeholder for additional conditions

/-- Counts the maximum number of additional spheres that can fit -/
def max_additional_spheres (config : SphereConfiguration) : ℕ :=
  2  -- Implementation details omitted, returning the known result

theorem additional_spheres_count 
  (config : SphereConfiguration) 
  (h : is_valid_configuration config) :
  max_additional_spheres config = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_spheres_count_l1227_122704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l1227_122742

theorem constant_term_in_expansion (n : ℕ) (h1 : n ≥ 5) (h2 : n ≤ 16) :
  (∃ k : ℕ, n = 4 * k) ↔ n ∈ ({8, 12, 16} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l1227_122742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_top_after_removal_l1227_122744

/-- Represents the number of dots on each face of the cube -/
def cube_faces : Finset Nat := {2, 3, 4, 5, 6, 7}

/-- The total number of dots on the cube -/
def total_dots : Nat := Finset.sum cube_faces id

/-- Probability of removing a dot from a specific face -/
def prob_remove_from_face (n : Nat) : ℚ := n / total_dots

/-- Probability of having an odd number of dots on top after removing one dot -/
def prob_odd_top : ℚ := 13 / 27

theorem prob_odd_top_after_removal :
  prob_odd_top = 
    (Finset.sum (cube_faces.filter (fun n => n % 2 = 0)) (fun n => prob_remove_from_face n * 2/3)) +
    (Finset.sum (cube_faces.filter (fun n => n % 2 = 1)) (fun n => prob_remove_from_face n * 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_top_after_removal_l1227_122744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l1227_122729

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((5 + 6 * Complex.I) * z^4 - z^6) ≤ 729 + 81 * Real.sqrt 61 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l1227_122729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1227_122725

theorem triangle_angle_relation (a b : ℝ) (A B : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  Real.sin B = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1227_122725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_integral_l1227_122776

-- Define a linear function f
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem linear_function_integral (a b : ℝ) :
  (∫ x in (0:ℝ)..1, f a b x) = 5 ∧
  (∫ x in (0:ℝ)..1, x * f a b x) = 17/6 →
  f a b = fun x ↦ 4 * x + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_integral_l1227_122776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_system_equations_l1227_122769

theorem solution_system_equations :
  ∀ a b c : ℝ,
  (a + b + c = 0) ∧ 
  (a^2 + b^2 + c^2 = 1) ∧ 
  (a^3 + b^3 + c^3 = 4*a*b*c) →
  ((a = Real.sqrt 2/2 ∧ b = -Real.sqrt 2/2 ∧ c = 0) ∨
   (a = -Real.sqrt 2/2 ∧ b = Real.sqrt 2/2 ∧ c = 0) ∨
   (a = Real.sqrt 2/2 ∧ b = 0 ∧ c = -Real.sqrt 2/2) ∨
   (a = -Real.sqrt 2/2 ∧ b = 0 ∧ c = Real.sqrt 2/2) ∨
   (a = 0 ∧ b = Real.sqrt 2/2 ∧ c = -Real.sqrt 2/2) ∨
   (a = 0 ∧ b = -Real.sqrt 2/2 ∧ c = Real.sqrt 2/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_system_equations_l1227_122769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_l1227_122711

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ) : ℕ := sorry

/-- Theorem stating the smallest possible value of b given the conditions -/
theorem smallest_b (a b : ℕ) 
  (ha : num_factors a = 4)
  (hb : num_factors b = a)
  (hd : b % a = 0) :
  24 ≤ b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_l1227_122711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1227_122783

noncomputable def f (x : ℝ) := Real.sqrt (1 - x^2)

def g (a b x : ℝ) := a * (x + b)

noncomputable def h (a b x : ℝ) := |a * (f x)^2 - (g a b x) / a|

theorem problem_statement 
  (a b : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : b ≤ 0) :
  (b = 0 → ∀ x, f x * g a b x = -(f (-x) * g a b (-x))) ∧ 
  (b = 0 → ∀ x y, -1 < x ∧ x < y ∧ y < 1 → g a b x / (f x)^2 < g a b y / (f y)^2) ∧
  ((∃ M, M = 2 ∧ ∀ x, h a b x ≤ M) → -1 < a + b ∧ a + b ≤ -1/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1227_122783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_routes_from_P_to_Q_l1227_122743

/-- Represents a point in the graph -/
inductive Point : Type
| P : Point
| R : Point
| S : Point
| T : Point
| Q : Point

/-- Represents a direct connection between two points -/
def connected : Point → Point → Prop
| Point.P, Point.R => True
| Point.P, Point.S => True
| Point.R, Point.S => True
| Point.R, Point.Q => True
| Point.S, Point.T => True
| Point.S, Point.Q => True
| Point.T, Point.Q => True
| _, _ => False

/-- Counts the number of routes between two points -/
def routeCount : Point → Point → Nat
| Point.T, Point.Q => 1
| Point.S, Point.Q => 2
| Point.R, Point.Q => 1
| Point.P, Point.Q => 3
| _, _ => 0

/-- The main theorem stating that there are 3 routes from P to Q -/
theorem three_routes_from_P_to_Q : routeCount Point.P Point.Q = 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_routes_from_P_to_Q_l1227_122743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_temp_of_square_sheet_l1227_122794

/-- Represents a square metal sheet with temperature boundary conditions -/
structure MetalSheet where
  side_length : ℝ
  side1_temp : ℝ
  side2_temp : ℝ
  side3_temp : ℝ
  side4_temp : ℝ

/-- The temperature at the center of the metal sheet -/
noncomputable def center_temperature (sheet : MetalSheet) : ℝ :=
  (sheet.side1_temp + sheet.side2_temp + sheet.side3_temp + sheet.side4_temp) / 4

/-- Theorem stating the temperature at the center of the metal sheet -/
theorem center_temp_of_square_sheet (sheet : MetalSheet) 
  (h1 : sheet.side1_temp = 0)
  (h2 : sheet.side2_temp = 0)
  (h3 : sheet.side3_temp = 0)
  (h4 : sheet.side4_temp = 100) :
  center_temperature sheet = 25 := by
  sorry

#check center_temp_of_square_sheet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_temp_of_square_sheet_l1227_122794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_digits_is_38_l1227_122716

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60
  seconds : Fin 60

/-- Calculates the sum of digits for a natural number -/
def sumDigits (n : ℕ) : ℕ :=
  n.repr.data.map (fun c => c.toNat - '0'.toNat) |>.sum

/-- Calculates the sum of digits for a given time -/
def timeSumDigits (t : Time24) : ℕ :=
  sumDigits t.hours.val + sumDigits t.minutes.val + sumDigits t.seconds.val

/-- The maximum sum of digits possible in a 24-hour digital watch display -/
def maxSumDigits : ℕ := 38

theorem max_sum_digits_is_38 :
  ∀ t : Time24, timeSumDigits t ≤ maxSumDigits :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_digits_is_38_l1227_122716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_tangent_intersection_l1227_122760

/-- Predicate defining a tangent line to a circle at a point -/
def IsTangentLine (circle : Set (ℝ × ℝ)) (point : ℝ × ℝ) (line : Set (ℝ × ℝ)) : Prop :=
  point ∈ circle ∧ point ∈ line ∧ 
  ∀ p : ℝ × ℝ, p ∈ line → p = point ∨ p ∉ circle

/-- Function to calculate the area of a circle -/
noncomputable def area (circle : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Implementation details omitted

/-- Given two points A and B on a circle ω, if the tangent lines to ω at A and B 
    intersect at a point on the x-axis, then the area of ω is 234π. -/
theorem circle_area_from_tangent_intersection (ω : Set (ℝ × ℝ)) 
  (A B : ℝ × ℝ) (h1 : A ∈ ω) (h2 : B ∈ ω) :
  A = (8, 15) → 
  B = (14, 9) → 
  (∃ C : ℝ × ℝ, C.2 = 0 ∧ 
    (∃ l1 l2 : Set (ℝ × ℝ), IsTangentLine ω A l1 ∧ IsTangentLine ω B l2 ∧ C ∈ l1 ∩ l2)) →
  area ω = 234 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_tangent_intersection_l1227_122760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_height_l1227_122705

/-- The volume of a right square cone inscribed in a sphere -/
noncomputable def volume_of_cone (sphere_diameter : ℝ) (cone_height : ℝ) : ℝ :=
  let radius := sphere_diameter / 2
  let base_to_center := radius - cone_height
  let base_edge := 2 * Real.sqrt (radius^2 - base_to_center^2)
  (1/3) * base_edge^2 * cone_height

/-- The height of the inscribed right square cone with maximum volume in a sphere of diameter 12 -/
theorem max_volume_cone_height (sphere_diameter : ℝ) (h : sphere_diameter = 12) : 
  ∃ (cone_height : ℝ), 
    cone_height = 8 ∧ 
    ∀ (other_height : ℝ), 
      (0 < other_height) → 
      (other_height ≤ sphere_diameter) →
      (volume_of_cone sphere_diameter other_height ≤ volume_of_cone sphere_diameter cone_height) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_cone_height_l1227_122705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l1227_122753

/-- Configuration of two intersecting circles with a line through their intersection point -/
structure CircleConfiguration where
  r₁ : ℝ
  r₂ : ℝ
  d : ℝ
  chord_length : ℝ

/-- The specific configuration described in the problem -/
noncomputable def problem_config : CircleConfiguration where
  r₁ := 7
  r₂ := 9
  d := 14
  chord_length := Real.sqrt 145

/-- Theorem stating that the given configuration results in chord length squared of 145 -/
theorem chord_length_squared (config : CircleConfiguration) 
  (h₁ : config.r₁ = 7)
  (h₂ : config.r₂ = 9)
  (h₃ : config.d = 14)
  (h₄ : config.chord_length > 0)
  (h₅ : config.chord_length^2 ≤ 4 * config.r₁^2)
  (h₆ : config.chord_length^2 ≤ 4 * config.r₂^2)
  (h₇ : (config.r₁ - config.r₂)^2 < config.d^2)
  (h₈ : config.d^2 < (config.r₁ + config.r₂)^2) :
  config.chord_length^2 = 145 := by
  sorry

#check chord_length_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l1227_122753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l1227_122714

-- Define the fixed points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (1, -1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the locus condition
def locus_condition (P : ℝ × ℝ) (a : ℝ) : Prop :=
  distance P A - distance P B = a

-- Theorem statement
theorem locus_characterization (a : ℝ) (h : a ≥ 0) :
  (∃ P : ℝ × ℝ, locus_condition P a ∧
    (∃ x y : ℝ, P = (x, y) ∧
      ((∃ k b : ℝ, y = k * x + b) ∨  -- Line
       (∃ c d e f g : ℝ, c * x^2 + d * x * y + e * y^2 + f * x + g * y = 1)))) -- Hyperbola
  ∨ (∀ P : ℝ × ℝ, ¬locus_condition P a) -- Empty set
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l1227_122714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_folding_problem_l1227_122747

/-- Represents a rectangle with given perimeter and area -/
structure Rectangle where
  perimeter : ℝ
  area : ℝ

/-- Checks if a rectangle with given perimeter and area exists -/
def rectangleExists (r : Rectangle) : Prop :=
  ∃ (length width : ℝ), 
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = r.perimeter ∧
    length * width = r.area

/-- Finds the dimensions of a rectangle with maximum area given a fixed perimeter -/
noncomputable def maxAreaRectangle (perimeter : ℝ) : Rectangle :=
  { perimeter := perimeter
  , area := (perimeter / 4) ^ 2 }

theorem wire_folding_problem (wire_length : ℝ) 
  (h_wire : wire_length = 22) 
  (area_30 : ℝ) (h_area_30 : area_30 = 30)
  (area_32 : ℝ) (h_area_32 : area_32 = 32) :
  (rectangleExists { perimeter := wire_length, area := area_30 } ∧ 
   (maxAreaRectangle wire_length).area = area_30) ∧
  ¬rectangleExists { perimeter := wire_length, area := area_32 } := by
  sorry

#check wire_folding_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_folding_problem_l1227_122747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l1227_122701

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the propositions
def prop1 (Line Plane : Type) 
  (perpendicular : Line → Plane → Prop) 
  (parallel_line : Line → Line → Prop) : Prop :=
  ∀ (a b : Line) (α : Plane),
    perpendicular a α → perpendicular b α → parallel_line a b

def prop2 (Line Plane : Type) 
  (parallel_line : Line → Line → Prop)
  (parallel_line_plane : Line → Plane → Prop) : Prop :=
  ∀ (a b : Line) (α : Plane),
    parallel_line_plane a α → parallel_line_plane b α → parallel_line a b

def prop3 (Line Plane : Type) 
  (perpendicular : Line → Plane → Prop)
  (parallel_plane : Plane → Plane → Prop) : Prop :=
  ∀ (a : Line) (α β : Plane),
    perpendicular a α → perpendicular a β → parallel_plane α β

def prop4 (Line Plane : Type) 
  (parallel_plane : Plane → Plane → Prop)
  (parallel_line_plane : Line → Plane → Prop) : Prop :=
  ∀ (b : Line) (α β : Plane),
    parallel_line_plane b α → parallel_line_plane b β → parallel_plane α β

-- The theorem to prove
theorem exactly_two_props_true : 
  (prop1 Line Plane perpendicular parallel_line ∧ 
   prop3 Line Plane perpendicular parallel_plane ∧ 
   ¬prop2 Line Plane parallel_line parallel_line_plane ∧ 
   ¬prop4 Line Plane parallel_plane parallel_line_plane) ∨
  (prop1 Line Plane perpendicular parallel_line ∧ 
   ¬prop3 Line Plane perpendicular parallel_plane ∧ 
   prop2 Line Plane parallel_line parallel_line_plane ∧ 
   ¬prop4 Line Plane parallel_plane parallel_line_plane) ∨
  (prop1 Line Plane perpendicular parallel_line ∧ 
   ¬prop3 Line Plane perpendicular parallel_plane ∧ 
   ¬prop2 Line Plane parallel_line parallel_line_plane ∧ 
   prop4 Line Plane parallel_plane parallel_line_plane) ∨
  (¬prop1 Line Plane perpendicular parallel_line ∧ 
   prop3 Line Plane perpendicular parallel_plane ∧ 
   prop2 Line Plane parallel_line parallel_line_plane ∧ 
   ¬prop4 Line Plane parallel_plane parallel_line_plane) ∨
  (¬prop1 Line Plane perpendicular parallel_line ∧ 
   prop3 Line Plane perpendicular parallel_plane ∧ 
   ¬prop2 Line Plane parallel_line parallel_line_plane ∧ 
   prop4 Line Plane parallel_plane parallel_line_plane) ∨
  (¬prop1 Line Plane perpendicular parallel_line ∧ 
   ¬prop3 Line Plane perpendicular parallel_plane ∧ 
   prop2 Line Plane parallel_line parallel_line_plane ∧ 
   prop4 Line Plane parallel_plane parallel_line_plane) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_props_true_l1227_122701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1227_122764

-- Define opposite numbers
def opposite (x y : ℝ) : Prop := x = -y

-- Define congruent triangles (simplified representation)
def congruent (t1 t2 : ℝ × ℝ × ℝ) : Prop := sorry

-- Define area of a triangle (simplified representation)
def area (t : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define even numbers
def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem propositions_truth : 
  (∀ x y : ℝ, opposite x y → x + y = 0) ∧ 
  (∃ t1 t2 : ℝ × ℝ × ℝ, congruent t1 t2 ∧ area t1 ≠ area t2) ∧ 
  (∀ a : ℚ, (a : ℝ) + 5 ∈ Set.range (Rat.cast : ℚ → ℝ)) ∧ 
  ¬(∀ a b : ℤ, even (a + b) → even a ∧ even b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1227_122764
