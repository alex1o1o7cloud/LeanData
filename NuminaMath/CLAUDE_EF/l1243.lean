import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_theorem_l1243_124308

/-- Calculates the total height of a stack of rings with given properties -/
noncomputable def stackHeight (topDiameter : ℝ) (bottomDiameter : ℝ) (thickness : ℝ) : ℝ :=
  let ringCount := (topDiameter - bottomDiameter) / (2 * thickness) + 1
  let insideDiameterSum := ringCount * (topDiameter - thickness + bottomDiameter - thickness) / 2
  insideDiameterSum + thickness

/-- The stack height theorem -/
theorem stack_height_theorem (topDiameter bottomDiameter thickness : ℝ) :
  topDiameter = 30 →
  bottomDiameter = 10 →
  thickness = 2 →
  stackHeight topDiameter bottomDiameter thickness = 200 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval stackHeight 30 10 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_theorem_l1243_124308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_2211_l1243_124348

def product_of_even_integers (n : Nat) : Nat :=
  Finset.prod (Finset.range (n/2 + 1)) (fun i => 2 * (i + 1))

theorem smallest_n_divisible_by_2211 :
  ∀ n : Nat, n % 2 = 0 →
    (product_of_even_integers n % 2211 = 0 ↔ n ≥ 42) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisible_by_2211_l1243_124348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_is_30_l1243_124347

/-- The coefficient of x^4 in the given polynomial expression -/
def coefficient_x4 (p : Polynomial ℚ) : ℚ :=
  p.coeff 4

/-- The polynomial expression from the problem -/
noncomputable def problem_polynomial : Polynomial ℚ :=
  2 * (Polynomial.X^2 - 2*Polynomial.X^4 + 3*Polynomial.X) + 
  4 * (2*Polynomial.X + 3*Polynomial.X^4 - Polynomial.X^2 + 2*Polynomial.X^5 - 2*Polynomial.X^4) - 
  6 * (2 + Polynomial.X - 5*Polynomial.X^4 - 2*Polynomial.X^3 + Polynomial.X^5)

/-- Theorem stating that the coefficient of x^4 in the problem polynomial is 30 -/
theorem coefficient_x4_is_30 : coefficient_x4 problem_polynomial = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_is_30_l1243_124347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1243_124359

noncomputable section

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 6 - y^2 / 12 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Define the point that the hyperbola passes through
def point_on_hyperbola : ℝ × ℝ := (3, -2 * Real.sqrt 3)

-- Define the angle between foci and point P
def angle_FPF : ℝ := 60 * Real.pi / 180

-- Theorem statement
theorem hyperbola_properties :
  -- The hyperbola passes through the given point
  hyperbola point_on_hyperbola.1 point_on_hyperbola.2 ∧
  -- The asymptotes are correct
  ∀ x y, hyperbola x y → asymptotes x y →
  -- The area of triangle PF₁F₂ is 6√3 when ∠F₁PF₂ = 60°
  ∃ P F₁ F₂ : ℝ × ℝ,
    hyperbola P.1 P.2 ∧
    |F₁.1 - F₂.1| = 6 ∧ -- Distance between foci is 2c = 2√18 = 6
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 + (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 -
    2 * Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2) * Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2) * Real.cos angle_FPF = 0 ∧
    1/2 * |F₁.1 * (F₂.2 - P.2) + F₂.1 * (P.2 - F₁.2) + P.1 * (F₁.2 - F₂.2)| = 6 * Real.sqrt 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1243_124359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_ring_radius_change_l1243_124363

/-- Represents the change in external radius of a metal ring -/
noncomputable def change_in_external_radius (initial_inner_circumference final_inner_circumference thickness : ℝ) : ℝ :=
  (final_inner_circumference - initial_inner_circumference) / (2 * Real.pi)

/-- Theorem stating the change in external radius for the given conditions -/
theorem metal_ring_radius_change :
  change_in_external_radius 30 40 1 = 5 / Real.pi :=
by
  -- Unfold the definition of change_in_external_radius
  unfold change_in_external_radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry

#check metal_ring_radius_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_ring_radius_change_l1243_124363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1243_124341

/-- A sequence of consecutive odd numbers -/
structure ConsecutiveOddSequence where
  start : ℕ  -- The first (smallest) number in the sequence
  length : ℕ  -- The number of terms in the sequence
  start_odd : Odd start  -- Ensure the start is odd
  length_pos : length > 0  -- Ensure the length is positive

/-- The sum of a sequence of consecutive odd numbers -/
def sum_consecutive_odd (seq : ConsecutiveOddSequence) : ℕ :=
  seq.length * (seq.start + seq.length - 1)

/-- A sequence is valid if its sum equals 1995 -/
def is_valid_sequence (seq : ConsecutiveOddSequence) : Prop :=
  sum_consecutive_odd seq = 1995

/-- The main theorem -/
theorem count_valid_sequences :
  ∃ (s : Finset ConsecutiveOddSequence), 
    (∀ seq ∈ s, is_valid_sequence seq) ∧ s.card = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1243_124341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1243_124395

-- Define the ellipse equation
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

-- Define the property of foci on x-axis
def foci_on_x_axis (k : ℝ) : Prop := k > 1

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, ellipse_equation x y k ∧ foci_on_x_axis k) ↔ k ∈ Set.Ioi 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_k_range_l1243_124395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1243_124396

noncomputable def f (a b c x : ℝ) : ℝ := a * x - b / x + c

theorem function_properties (a b c : ℝ) :
  f a b c 1 = 0 ∧
  (∀ x, x ≠ 0 → HasDerivAt (f a b c) (-x + 3) 2) →
  (∀ x, x ≠ 0 → f a b c x = -3 * x - 8 / x + 11) ∧
  IsLocalMin (f (-3) 8 11) (-2 * Real.sqrt 6 / 3) ∧
  f (-3) 8 11 (-2 * Real.sqrt 6 / 3) = 11 + 4 * Real.sqrt 6 ∧
  IsLocalMax (f (-3) 8 11) (2 * Real.sqrt 6 / 3) ∧
  f (-3) 8 11 (2 * Real.sqrt 6 / 3) = 11 - 4 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1243_124396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1243_124309

theorem simplify_expression (n : ℕ) : (2^(n+4) - 2*(2^n)) / (2*(2^(n+3))) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1243_124309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_compounding_difference_anthony_loan_difference_l1243_124360

/-- Calculates the compound interest for a given principal, rate, compounding frequency, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- The difference in amount owed between semi-annual and annual compounding -/
theorem loan_compounding_difference (principal : ℝ) (rate : ℝ) (time : ℝ) :
  let semi_annual := compound_interest principal rate 2 time
  let annual := compound_interest principal rate 1 time
  ∃ ε > 0, |semi_annual - annual - 147.04| < ε :=
by
  sorry

/-- The specific case for Anthony's loan -/
theorem anthony_loan_difference :
  let principal := (8000 : ℝ)
  let rate := (0.10 : ℝ)
  let time := (5 : ℝ)
  let semi_annual := compound_interest principal rate 2 time
  let annual := compound_interest principal rate 1 time
  ∃ ε > 0, |semi_annual - annual - 147.04| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_compounding_difference_anthony_loan_difference_l1243_124360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1243_124379

noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin ((ω * x / 2) + (φ / 2) + (Real.pi / 6))

theorem g_properties (ω φ : ℝ) (h1 : 0 < ω) (h2 : 0 < φ) (h3 : φ < Real.pi) :
  (∀ x, g ω φ x = g ω φ (-x)) ∧
  (∀ x, g ω φ (x + Real.pi) = g ω φ x) →
  ω = 4 ∧ φ = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1243_124379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1243_124330

theorem diophantine_equation_solutions :
  {(x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1} =
  {(1,0,0,0), (3,0,0,1), (1,1,1,0), (2,2,1,1)} := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1243_124330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l1243_124342

/-- Calculates the compound interest for a given principal, rate, compounding frequency, and time --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounding : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compounding) ^ (compounding * time)

/-- Calculates the total payment for Plan 1 --/
noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  let half_time := time / 2
  let half_balance := compound_interest principal rate 2 half_time / 2
  half_balance + compound_interest half_balance rate 2 half_time

/-- Calculates the total payment for Plan 2 --/
noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest principal rate 1 time

/-- The main theorem stating the difference between Plan 2 and Plan 1 payments --/
theorem loan_payment_difference (principal : ℝ) (rate : ℝ) (time : ℝ)
  (h_principal : principal = 15000)
  (h_rate : rate = 0.08)
  (h_time : time = 12) :
  ⌊plan2_payment principal rate time - plan1_payment principal rate time⌋ = 6542 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l1243_124342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1243_124383

-- Define the circle
noncomputable def circle_eq (x y : ℝ) : Prop := (x - Real.sqrt 2)^2 + y^2 = 1

-- Define the hyperbola
noncomputable def hyperbola_eq (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), circle_eq x y ∧ hyperbola_eq x y a b) →
  eccentricity a b = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1243_124383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_prime_l1243_124374

theorem at_least_one_prime (n : ℕ) (a : Fin n → ℕ) :
  (∀ i : Fin n, 1 < a i ∧ a i < (2 * n - 1)^2) →
  (∀ i j : Fin n, i ≠ j → Nat.Coprime (a i) (a j)) →
  ∃ i : Fin n, Nat.Prime (a i) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_prime_l1243_124374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_four_max_l1243_124371

noncomputable def f (a b : ℝ) : ℝ :=
  |(|b - a| / |a * b| + (b + a) / (a * b) - 1)| + |b - a| / |a * b| + (b + a) / (a * b) + 1

theorem f_equals_four_max (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f a b = 4 * max (1 / a) (max (1 / b) (1 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_four_max_l1243_124371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_power_of_two_l1243_124328

theorem square_sum_power_of_two (n : ℕ) : (∃ m : ℕ, 2^8 + 2^11 + 2^n = m^2) ↔ n = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_sum_power_of_two_l1243_124328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_250_l1243_124338

theorem closest_integer_to_cube_root_250 : 
  ∃ n : ℤ, (∀ m : ℤ, |n - (250 : ℝ) ^ (1/3)| ≤ |m - (250 : ℝ) ^ (1/3)|) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_integer_to_cube_root_250_l1243_124338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l1243_124345

-- Define the function type
def PositiveRealFunction := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : PositiveRealFunction) : Prop :=
  ∀ (x y z : ℝ), x > y ∧ y > z ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    f (x - y + z) = f x + f y + f z - x * y - y * z + x * z

-- Theorem statement
theorem unique_function_satisfying_equation :
  ∀ (f : PositiveRealFunction), SatisfiesEquation f → ∀ (x : ℝ), x > 0 → f x = x^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_equation_l1243_124345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_f_2_eq_4_implies_p_4_q_0_g_decreasing_on_0_to_2_l1243_124389

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ := (x^2 + p) / (x + q)

theorem odd_function_and_f_2_eq_4_implies_p_4_q_0 (p q : ℝ) :
  (∀ x, f p q x = -f p q (-x)) → f p q 2 = 4 → p = 4 ∧ q = 0 := by sorry

noncomputable def g (x : ℝ) : ℝ := x + 4 / x

theorem g_decreasing_on_0_to_2 :
  ∀ x y, 0 < x → x < y → y < 2 → g x > g y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_f_2_eq_4_implies_p_4_q_0_g_decreasing_on_0_to_2_l1243_124389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hc_je_ratio_l1243_124361

/-- A line segment with a start and end point -/
structure Segment (P : Type) where
  start : P
  finish : P

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Parallel relation between two segments -/
def parallel (s1 s2 : Segment Point) : Prop := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- A point lies on a segment -/
def lies_on (p : Point) (s : Segment Point) : Prop := sorry

/-- Collinear points -/
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem hc_je_ratio 
  (A B C D E F G H J : Point)
  (AF : Segment Point)
  (h1 : AF.start = A ∧ AF.finish = F)
  (h2 : collinear A B C ∧ collinear B C D ∧ collinear C D E ∧ collinear D E F)
  (h3 : distance A B = distance B C ∧ distance B C = distance C D ∧ 
        distance C D = distance D E ∧ distance D E = distance E F ∧ 
        distance E F = 1)
  (h4 : ¬ lies_on G AF)
  (h5 : lies_on H (Segment.mk G D))
  (h6 : lies_on J (Segment.mk G F))
  (h7 : parallel (Segment.mk H C) (Segment.mk J E))
  (h8 : parallel (Segment.mk H C) (Segment.mk A G))
  (h9 : parallel (Segment.mk J E) (Segment.mk A G)) :
  distance H C / distance J E = 5 / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hc_je_ratio_l1243_124361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l1243_124388

-- Define the line (marked as noncomputable due to use of Real.sqrt)
noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + t/2, -3*Real.sqrt 3 + (Real.sqrt 3/2)*t)

-- Define the circle
def circle_eq (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 9

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, line t = p ∧ circle_eq p}

-- State the theorem
theorem midpoint_coordinates :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  (A.1 + B.1)/2 = 3 ∧ (A.2 + B.2)/2 = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l1243_124388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_erasable_records_l1243_124312

-- Define a type for cities
def City : Type := ℝ × ℝ

-- Define a function to calculate distance between two cities
noncomputable def distance (c1 c2 : City) : ℝ := Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)

-- Define a property that no three cities are collinear
def not_collinear (cities : List City) : Prop :=
  ∀ a b c : City, a ∈ cities → b ∈ cities → c ∈ cities →
    a ≠ b → b ≠ c → a ≠ c →
    (b.2 - a.2) * (c.1 - a.1) ≠ (c.2 - a.2) * (b.1 - a.1)

-- Define the theorem
theorem max_erasable_records (n : ℕ) (cities : List City) 
  (h_count : cities.length = n) (h_not_collinear : not_collinear cities) :
  ∃ k : ℕ, k = n - 4 ∧ 
    (∀ m : ℕ, m ≤ k → 
      ∃! (distances : List ℝ), 
        distances.length = n * (n - 1) / 2 - m ∧
        ∀ i j : Fin n, i < j → 
          distance (cities[i.val]) (cities[j.val]) ∈ distances) ∧
    ¬(∃ (distances : List ℝ), 
        distances.length = n * (n - 1) / 2 - (k + 1) ∧
        ∀ i j : Fin n, i < j → 
          distance (cities[i.val]) (cities[j.val]) ∈ distances) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_erasable_records_l1243_124312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l1243_124304

/-- A line passing through points (2, 8), (6, 20), (10, 32), and (50, u) -/
structure Line where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ

/-- Definition of a point lying on the line -/
def liesOn (p : ℝ × ℝ) (l : Line) : Prop :=
  p.2 = l.m * p.1 + l.b

/-- Theorem stating that if (2, 8), (6, 20), (10, 32), and (50, u) lie on the same line, then u = 152 -/
theorem line_through_points (l : Line) (u : ℝ) 
    (h1 : liesOn (2, 8) l)
    (h2 : liesOn (6, 20) l)
    (h3 : liesOn (10, 32) l)
    (h4 : liesOn (50, u) l) : 
  u = 152 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_l1243_124304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l1243_124386

/-- The time (in seconds) for a train to pass a man running in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / (train_speed + man_speed)

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

theorem train_passing_man_time :
  let train_length : ℝ := 110
  let train_speed : ℝ := 84 * km_per_hr_to_m_per_s
  let man_speed : ℝ := 6 * km_per_hr_to_m_per_s
  train_passing_time train_length train_speed man_speed = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l1243_124386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l1243_124372

noncomputable section

-- Define the ellipse C and parabola E
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_parabola_intersection_eccentricity 
  (a b p : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : p > 0) 
  (h4 : ∃ (A B : ℝ × ℝ), 
    ellipse a b A.1 A.2 ∧ 
    ellipse a b B.1 B.2 ∧ 
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2) 
  (h5 : ∃ (F : ℝ × ℝ), 
    F.1 = Real.sqrt (a^2 - b^2) ∧ 
    F.1 = p / 2) 
  (h6 : ∃ (A B F : ℝ × ℝ), 
    ellipse a b A.1 A.2 ∧ 
    ellipse a b B.1 B.2 ∧ 
    parabola p A.1 A.2 ∧ 
    parabola p B.1 B.2 ∧ 
    F.1 = Real.sqrt (a^2 - b^2) ∧ 
    F.1 = p / 2 ∧ 
    (A.2 - F.2) / (A.1 - F.1) = (B.2 - F.2) / (B.1 - F.1)) :
  eccentricity a b = Real.sqrt 2 - 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l1243_124372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1243_124365

-- Define the original function f
def f (x : ℝ) := x^2

-- Define the inverse function g
noncomputable def g (x : ℝ) := -Real.sqrt x

-- Theorem statement
theorem inverse_function_proof :
  ∀ x : ℝ, x < -2 → 
  ∀ y : ℝ, y > 4 →
  (f x = y ↔ g y = x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1243_124365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_ratio_approx_twelve_l1243_124349

/-- The number of muffins Arthur baked -/
noncomputable def arthur_muffins : ℝ := 115.0

/-- The number of muffins James baked -/
noncomputable def james_muffins : ℝ := 9.58333333299999

/-- The ratio of muffins baked by Arthur to muffins baked by James -/
noncomputable def muffin_ratio : ℝ := arthur_muffins / james_muffins

theorem muffin_ratio_approx_twelve : 
  |muffin_ratio - 12| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_ratio_approx_twelve_l1243_124349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1243_124344

def sequenceProperty (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (∀ n : ℕ, n > 0 → a n * a (n + 1) * a (n + 2) ≠ 1) ∧
  (∀ n : ℕ, n > 0 → a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3))

theorem sequence_sum (a : ℕ → ℕ) (h : sequenceProperty a) :
  (Finset.range 100).sum a = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1243_124344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rotation_path_length_l1243_124380

/-- The length of the path traversed by a vertex of an isosceles right triangle
    rotating inside a square. -/
noncomputable def path_length (triangle_leg_length : ℝ) (square_side_length : ℝ) : ℝ :=
  8 * (Real.pi / 2 * triangle_leg_length)

theorem isosceles_triangle_rotation_path_length :
  path_length 3 6 = 12 * Real.pi :=
by
  unfold path_length
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check isosceles_triangle_rotation_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rotation_path_length_l1243_124380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_combined_mpg_l1243_124393

/-- Car represents a car with its fuel efficiency in miles per gallon -/
structure Car where
  mpg : ℚ

/-- TravelInfo represents travel information for a car -/
structure TravelInfo where
  car : Car
  distance : ℚ

/-- CombinedTravel represents combined travel information for two cars -/
structure CombinedTravel where
  alice : TravelInfo
  bob : TravelInfo
  gas_limit : ℚ

/-- Calculate the combined miles per gallon for two cars -/
noncomputable def combined_mpg (travel : CombinedTravel) : ℚ :=
  (travel.alice.distance + travel.bob.distance) / travel.gas_limit

/-- Theorem stating the combined miles per gallon for Alice and Bob's cars -/
theorem alice_bob_combined_mpg :
  let alice_car : Car := ⟨50⟩
  let bob_car : Car := ⟨30⟩
  let bob_distance : ℚ := 160
  let alice_distance : ℚ := 2 * bob_distance
  let gas_limit : ℚ := 10
  let combined_travel : CombinedTravel := ⟨⟨alice_car, alice_distance⟩, ⟨bob_car, bob_distance⟩, gas_limit⟩
  combined_mpg combined_travel = 408/10 := by
  sorry

#eval (408 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_bob_combined_mpg_l1243_124393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_increasing_is_one_fourth_l1243_124300

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

-- Define the condition for f(x) to be increasing on [2, +∞)
def f_increasing (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 2 → (1 - a / x) ≥ 0

-- Define the set of 'a' values satisfying the given inequality
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, 2 * a * x^2 + a * x + 1 > 0}

-- State the theorem
theorem probability_f_increasing_is_one_fourth :
  (MeasureTheory.volume {a ∈ A | f_increasing a}) / (MeasureTheory.volume A) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_increasing_is_one_fourth_l1243_124300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sphere_in_unit_cube_sphere_diameter_equals_cube_edge_sphere_volume_formula_l1243_124392

/-- The volume of the largest sphere that can be carved from a cube with edge length 1 -/
noncomputable def largest_sphere_volume : ℝ := Real.pi / 6

/-- The edge length of the cube -/
def cube_edge : ℝ := 1

/-- Theorem stating that the volume of the largest sphere that can be carved from a cube 
    with edge length 1 is π/6 -/
theorem largest_sphere_in_unit_cube : 
  largest_sphere_volume = Real.pi / 6 := by
  -- Unfold the definition of largest_sphere_volume
  unfold largest_sphere_volume
  -- The left-hand side is now equal to the right-hand side
  rfl

/-- Theorem proving that the largest sphere's diameter equals the cube's edge length -/
theorem sphere_diameter_equals_cube_edge :
  2 * (largest_sphere_volume * 3 / (4 * Real.pi)) ^ (1/3 : ℝ) = cube_edge := by
  sorry

/-- Theorem proving that the sphere volume formula is correct -/
theorem sphere_volume_formula (r : ℝ) :
  4 / 3 * Real.pi * r ^ 3 = 4 * Real.pi * r ^ 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sphere_in_unit_cube_sphere_diameter_equals_cube_edge_sphere_volume_formula_l1243_124392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l1243_124301

/-- Given a square room with an area of 400 square feet and 8-inch by 8-inch tiles,
    prove that the number of tiles in each row is 30. -/
theorem tiles_per_row (room_area : ℝ) (tile_size : ℝ) : 
  room_area = 400 → tile_size = 8 / 12 → ⌊Real.sqrt room_area / tile_size⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_per_row_l1243_124301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1243_124329

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 6*x + 10⌋

theorem domain_of_f :
  let S := {x : ℝ | x < 3 ∨ x > 3}
  (∀ x, x^2 - 6*x + 10 > 0) →
  (3^2 - 6*3 + 10 = 1) →
  (∀ x ≠ 3, x^2 - 6*x + 10 > 1) →
  ∀ x, f x ≠ 0 ↔ x ∈ S :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1243_124329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1243_124373

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → 2 * f x + x^2 - a * x + 3 ≥ 0) → 
  a ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1243_124373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1243_124390

theorem complex_equation_solution (z : ℂ) 
  (h : 12 * (z.re^2 + z.im^2) = 2 * ((z + 2).re^2 + (z + 2).im^2) + ((z^2 + 1).re^2 + (z^2 + 1).im^2) + 31) : 
  z + 6 / z = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l1243_124390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_total_l1243_124399

theorem election_votes_total (candidate1_percentage : ℚ) (candidate2_votes : ℕ) : 
  candidate1_percentage = 4/5 →
  candidate2_votes = 480 →
  ∃ (total_votes : ℕ), 
    (candidate1_percentage * total_votes = total_votes - candidate2_votes) ∧
    (total_votes = 2400) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_total_l1243_124399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_3_range_of_a_empty_solution_l1243_124334

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |x + 1/2| + |x - 3/2|

-- Theorem for part (Ⅰ)
theorem solution_set_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by
  sorry

-- Theorem for part (2)
theorem range_of_a_empty_solution :
  (∀ x : ℝ, f x ≥ 1/2 * |1 - a|) ↔ -3 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_3_range_of_a_empty_solution_l1243_124334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1243_124370

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
  t.a * Real.cos t.C + t.c * Real.cos t.A = 2 * t.b * Real.cos t.A ∧
  t.b + t.c = Real.sqrt 10 ∧
  t.a = 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) : 
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1243_124370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1243_124378

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (9, 0)

-- Define the dividing line
def a : ℝ := 6

-- Function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem equal_area_division :
  let D : ℝ × ℝ := (a, 0)
  let E : ℝ × ℝ := (a, 2*a/9)
  triangleArea A B D + triangleArea A D E = triangleArea A E C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1243_124378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_rate_is_6_kmph_l1243_124353

/-- Represents the properties of a river and its flow --/
structure River where
  depth : ℝ
  width : ℝ
  flow_volume : ℝ

/-- Calculates the flow rate of a river in kilometers per hour --/
noncomputable def flow_rate_kmph (r : River) : ℝ :=
  (r.flow_volume / (r.depth * r.width)) * (1 / 1000) * 60

/-- Theorem stating that for the given river properties, the flow rate is 6 kmph --/
theorem river_flow_rate_is_6_kmph :
  let r : River := { depth := 4, width := 65, flow_volume := 26000 }
  flow_rate_kmph r = 6 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_rate_is_6_kmph_l1243_124353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_equilateral_triangle_characterization_l1243_124318

/-- Predicate to define an equilateral triangle in the complex plane -/
def IsEquilateralTriangle (a b c : ℂ) : Prop :=
  Complex.abs (b - a) = Complex.abs (c - b) ∧ 
  Complex.abs (c - b) = Complex.abs (a - c) ∧
  (c - a) / (b - a) = Complex.exp (Complex.I * Real.pi / 3)

/-- Characterization of a direct equilateral triangle in the complex plane -/
theorem direct_equilateral_triangle_characterization 
  (a b c : ℂ) (h : IsEquilateralTriangle a b c) :
  (c - a) / (b - a) = Complex.exp (Complex.I * Real.pi / 3) ∧ 
  a + Complex.exp (Complex.I * 2 * Real.pi / 3) * b + Complex.exp (Complex.I * 4 * Real.pi / 3) * c = 0 ∧
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direct_equilateral_triangle_characterization_l1243_124318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_count_l1243_124327

/-- Represents a polynomial with real coefficients -/
def MyPolynomial (α : Type*) [Ring α] := List α

/-- Horner's method for polynomial evaluation -/
def horner_eval (p : MyPolynomial ℝ) (x : ℝ) : ℝ :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Counts the number of operations in Horner's method -/
def horner_ops_count (p : MyPolynomial ℝ) : ℕ × ℕ :=
  (p.length - 1, p.length - 1)

theorem horner_method_operations_count :
  let p : MyPolynomial ℝ := [1, 8, 7, 6, 5, 4, 3]  -- coefficients in reverse order
  let (mults, adds) := horner_ops_count p
  mults = 6 ∧ adds = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_count_l1243_124327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_sphere_surface_area_l1243_124356

/-- A right prism with all vertices on a sphere --/
structure PrismOnSphere where
  height : ℝ
  volume : ℝ
  sphereRadius : ℝ

/-- The surface area of a sphere --/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- Theorem stating the relationship between the prism's dimensions and the sphere's surface area --/
theorem prism_sphere_surface_area (p : PrismOnSphere) 
  (h_height : p.height = 4)
  (h_volume : p.volume = 32) :
  sphereSurfaceArea p.sphereRadius = 32 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_sphere_surface_area_l1243_124356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1243_124313

theorem equation_solutions : 
  ∃! (S : Finset ℝ), (∀ x ∈ S, (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)) ∧ 
                   S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1243_124313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_n_l1243_124340

-- Define the property for n
def has_property (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧
  Nat.Coprime a b ∧
  ∀ (p : ℕ), Nat.Prime p → p ≤ Nat.sqrt n → p ∣ (a * b)

-- Theorem statement
theorem characterize_n : 
  ∀ n : ℕ, has_property n ↔ n = 2 ∨ n = 5 ∨ n = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_n_l1243_124340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_quadrant_l1243_124324

theorem complex_multiplication_quadrant :
  ∀ (a b : ℝ), (3 - Complex.I) * Complex.I = Complex.ofReal a + Complex.I * Complex.ofReal b →
  a = 1 ∧ b = 3 ∧ a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_multiplication_quadrant_l1243_124324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_center_l1243_124352

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point
  b : Point
  c : Point

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def isOnCircle (p : Point) (c : Circle) : Prop :=
  distance p c.center = c.radius

def isChord (p1 p2 : Point) (c : Circle) : Prop :=
  isOnCircle p1 c ∧ isOnCircle p2 c

def isEquilateral (t : Triangle) : Prop :=
  distance t.a t.b = distance t.b t.c ∧ distance t.b t.c = distance t.c t.a

theorem max_distance_to_circle_center 
  (c : Circle)
  (t : Triangle)
  (h1 : c.center = Point.mk 1 2)
  (h2 : c.radius = Real.sqrt 2)
  (h3 : isEquilateral t)
  (h4 : isChord t.a t.b c) :
  ∃ (p : Point), isOnCircle p c ∧ 
    ∀ (q : Point), isOnCircle q c → distance q c.center ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_center_l1243_124352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_144_minus_36pi_l1243_124364

noncomputable section

-- Define the side length of the square
def side_length : ℝ := 12

-- Define the area of the square
def square_area : ℝ := side_length ^ 2

-- Define the radius of each quarter circle
def quarter_circle_radius : ℝ := side_length / 2

-- Define the area of one quarter circle
def quarter_circle_area : ℝ := Real.pi * quarter_circle_radius ^ 2 / 4

-- Define the total area of four quarter circles
def total_quarter_circles_area : ℝ := 4 * quarter_circle_area

-- Define the shaded area
def shaded_area : ℝ := square_area - total_quarter_circles_area

-- Theorem statement
theorem shaded_area_equals_144_minus_36pi :
  shaded_area = 144 - 36 * Real.pi := by
  -- The proof is omitted for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_equals_144_minus_36pi_l1243_124364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_blue_cube_possible_l1243_124315

/-- Represents a small cube with some painted faces -/
structure SmallCube where
  painted_faces : Fin 6 → Bool

/-- Represents the large 3x3x3 cube -/
def LargeCube := Fin 3 → Fin 3 → Fin 3 → SmallCube

def total_painted_faces (cube : LargeCube) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 3)) λ i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) λ j =>
      Finset.sum (Finset.univ : Finset (Fin 3)) λ k =>
        Finset.sum (Finset.univ : Finset (Fin 6)) λ f =>
          if (cube i j k).painted_faces f then 1 else 0

/-- Theorem stating that no blue cube can be assembled -/
theorem no_blue_cube_possible (cube : LargeCube) 
    (h : total_painted_faces cube = 54) : 
  ¬∃ (size : Nat) (i j k : Fin 3), 
    (i.val + size ≤ 3 ∧ j.val + size ≤ 3 ∧ k.val + size ≤ 3) ∧
    (∀ x y z, x < size → y < size → z < size →
      ∀ f, (cube ⟨i.val + x, by sorry⟩ ⟨j.val + y, by sorry⟩ ⟨k.val + z, by sorry⟩).painted_faces f = true) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_blue_cube_possible_l1243_124315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_equation_l1243_124323

/-- Represents the monthly growth rate of mask donations -/
def x : Real := sorry

/-- Represents the initial donation in January (in units of 10,000 masks) -/
def initial_donation : Real := 1

/-- Represents the total donation goal by the end of Q1 (in units of 10,000 masks) -/
def total_donation_goal : Real := 4.75

/-- Theorem stating that the sum of donations over three months equals the total donation goal -/
theorem donation_equation : 
  initial_donation + (initial_donation * (1 + x)) + (initial_donation * (1 + x)^2) = total_donation_goal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donation_equation_l1243_124323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_in_ten_digit_number_repeated_digit_in_2_pow_30_l1243_124320

-- Define a type for decimal digits (0-9)
inductive Digit : Type
  | zero | one | two | three | four | five | six | seven | eight | nine

-- Define a function to count the number of digits in a natural number
def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + count_digits (n / 10)

-- Define a function to get the nth digit of a number
def get_digit (n : ℕ) (i : ℕ) : ℕ :=
  (n / (10 ^ i)) % 10

-- Define a proposition that states there are at least two identical digits in a number
def has_repeated_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ) (i j : ℕ), i ≠ j ∧ i < 10 ∧ j < 10 ∧ get_digit n i = d ∧ get_digit n j = d

theorem repeated_digit_in_ten_digit_number :
  ∀ n : ℕ, count_digits n = 10 → has_repeated_digit n :=
by
  intro n h
  -- We'll use the pigeonhole principle here
  -- If we have 10 digits (pigeons) and only 10 possible values (holes),
  -- at least one value must appear twice
  sorry  -- The detailed proof is omitted for brevity

-- Theorem specifically for 2^30
theorem repeated_digit_in_2_pow_30 :
  has_repeated_digit (2^30) :=
by
  have h : count_digits (2^30) = 10 := by
    -- Proof that 2^30 has 10 digits
    sorry  -- The detailed proof is omitted for brevity
  exact repeated_digit_in_ten_digit_number (2^30) h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_digit_in_ten_digit_number_repeated_digit_in_2_pow_30_l1243_124320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_of_special_list_l1243_124335

/-- A list where each integer n from 1 to 100 appears exactly n times -/
def special_list := List ℕ

/-- The property that each integer n appears exactly n times in the list -/
def has_special_property (l : special_list) : Prop :=
  ∀ n : Fin 100, (l.count (n.val.succ) = n.val.succ)

/-- The mode of a list is the element that appears most frequently -/
def is_mode (m : ℕ) (l : special_list) : Prop :=
  ∀ n : Fin 100, l.count m ≥ l.count (n.val.succ)

theorem mode_of_special_list (l : special_list) (h : has_special_property l) :
  is_mode 100 l := by
  sorry

#check mode_of_special_list

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_of_special_list_l1243_124335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l1243_124336

noncomputable def S (n : ℕ) : ℝ := n^2 + 2*n

def a : ℕ → ℝ
  | 0 => 0
  | 1 => 3
  | (n+2) => 2*(n+2) + 1

noncomputable def b (n : ℕ) : ℝ := a n * a (n+1) * Real.cos ((n+1 : ℝ) * Real.pi)

noncomputable def T : ℕ → ℝ
  | 0 => 0
  | (n+1) => T n + b n

theorem max_t_value (t : ℝ) : 
  (∀ n : ℕ, n > 0 → T n ≥ t * n^2) → t ≤ -5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l1243_124336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_toys_sold_l1243_124343

/-- Proves that the percentage of toys sold is 40% given the problem conditions -/
theorem percentage_of_toys_sold
  (initial_toys : ℕ)
  (buy_price sell_price total_profit : ℚ)
  (h1 : initial_toys = 200)
  (h2 : buy_price = 20)
  (h3 : sell_price = 30)
  (h4 : total_profit = 800) :
  (((total_profit / (sell_price - buy_price)) / initial_toys) * 100 : ℚ) = 40 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_toys_sold_l1243_124343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_through_center_angle_bisector_divides_square_equally_l1243_124387

-- Define the square
structure Square where
  side : ℝ
  side_pos : side > 0

-- Define the right triangle
structure RightTriangle where
  base : ℝ
  height : ℝ
  base_pos : base > 0
  height_pos : height > 0

-- Define the configuration
structure Configuration where
  square : Square
  triangle : RightTriangle
  hypotenuse_eq_side : square.side = Real.sqrt (triangle.base^2 + triangle.height^2)

-- Define the angle bisector
def angle_bisector (config : Configuration) : ℝ × ℝ → Prop :=
  fun point => sorry

-- Define the center of the square
noncomputable def square_center (s : Square) : ℝ × ℝ :=
  (s.side / 2, s.side / 2)

-- Theorem statement
theorem angle_bisector_through_center (config : Configuration) :
  angle_bisector config (square_center config.square) := by
  sorry

-- Theorem: The angle bisector divides the square into two equal parts
theorem angle_bisector_divides_square_equally (config : Configuration) :
  ∃ (area1 area2 : ℝ), 
    area1 = area2 ∧ 
    area1 + area2 = config.square.side ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_through_center_angle_bisector_divides_square_equally_l1243_124387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homologous_synapse_definition_l1243_124346

-- Define a structure for chromosomes
structure Chromosome where
  length : ℝ
  genePositions : Set ℕ
  centromereLocation : ℝ
  genes : Set String

-- Define a predicate for chromosome similarity
def similar (c1 c2 : Chromosome) : Prop :=
  c1.length = c2.length ∧
  c1.genePositions = c2.genePositions ∧
  c1.centromereLocation = c2.centromereLocation ∧
  c1.genes = c2.genes

-- Define a predicate for synapsing during meiosis
def synapsesDuringMeiosis (c1 c2 : Chromosome) : Prop := sorry

-- Define homologous chromosomes
def homologous (c1 c2 : Chromosome) : Prop :=
  similar c1 c2 ∧ synapsesDuringMeiosis c1 c2

-- Theorem stating that homologous chromosomes are defined by synapsing during meiosis
theorem homologous_synapse_definition (c1 c2 : Chromosome) :
  homologous c1 c2 ↔ synapsesDuringMeiosis c1 c2 ∧ similar c1 c2 := by
  sorry

#check homologous_synapse_definition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homologous_synapse_definition_l1243_124346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_equals_81_l1243_124376

theorem product_xy_equals_81 
  (x y : ℝ) 
  (h1 : Real.sqrt x + Real.sqrt y = 10) 
  (h2 : (x ^ (1/4)) + (y ^ (1/4)) = 4) : 
  x * y = 81 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_xy_equals_81_l1243_124376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_distance_is_15_8_l1243_124375

/-- A random walk on a line where each move is +1 or -1 unit with equal probability -/
def RandomWalk := ℕ → Bool

/-- The position after n steps in a random walk -/
def position (walk : RandomWalk) (n : ℕ) : ℤ :=
  (List.range n).foldl (λ acc i => acc + if walk i then 1 else -1) 0

/-- The distance from the origin after n steps in a random walk -/
def distance (walk : RandomWalk) (n : ℕ) : ℕ :=
  (position walk n).natAbs

/-- The probability space of all possible 6-step random walks -/
def RandomWalk6 := Fin (2^6)

/-- Convert a Fin (2^6) to a RandomWalk -/
def toRandomWalk (f : Fin (2^6)) : RandomWalk :=
  λ i => if i < 6 then (f.val >>> i) % 2 = 1 else false

/-- The expected distance from the origin after 6 steps in a random walk -/
noncomputable def expected_distance : ℚ :=
  (Finset.sum Finset.univ (λ f => (distance (toRandomWalk f) 6))) / (2^6)

theorem expected_distance_is_15_8 : expected_distance = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_distance_is_15_8_l1243_124375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1243_124366

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + 1 / x

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -1 ∧ x ≠ 0}

-- Theorem stating that the domain of f is [-1,0) ∪ (0,+∞)
theorem domain_of_f :
  domain_f = Set.Icc (-1) 0 ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1243_124366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_symmetric_perpendicular_points_l1243_124398

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points A, B, and C
variable (A B C : ℝ × ℝ)

-- Define that AB is a diameter of the circle
variable (h_diameter : IsDiameter circle A B)

-- Define that C is on line AB
variable (h_C_on_AB : C ∈ Set.Icc A B)

-- Define the property of symmetry with respect to a line
def SymmetricWrtLine (X Y : ℝ × ℝ) (L : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property of perpendicularity
def Perpendicular (L1 L2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define a line through two points
def LineThroughPoints (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem existence_of_symmetric_perpendicular_points :
  ∃ (X Y : ℝ × ℝ),
    X ∈ circle ∧
    Y ∈ circle ∧
    SymmetricWrtLine X Y (LineThroughPoints A B) ∧
    Perpendicular (LineThroughPoints A X) (LineThroughPoints Y C) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_symmetric_perpendicular_points_l1243_124398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_theorem_l1243_124391

/-- Calculates the savings in foreign currency based on given tax rates and fees -/
def calculate_savings (original_tax_rate : ℚ) (reduced_tax_rate : ℚ) 
                      (discount_rate : ℚ) (market_price : ℚ) 
                      (processing_fee_rate : ℚ) (conversion_rate : ℚ) : ℚ :=
  let original_price := market_price * (1 + original_tax_rate) * (1 + processing_fee_rate)
  let discounted_price := market_price * (1 - discount_rate)
  let reduced_price := discounted_price * (1 + reduced_tax_rate) * (1 + processing_fee_rate)
  let savings_in_rs := original_price - reduced_price
  savings_in_rs / conversion_rate

/-- The theorem stating that the savings are approximately 53.65 units of foreign currency -/
theorem savings_theorem : 
  let original_tax_rate : ℚ := 77 / 1000
  let reduced_tax_rate : ℚ := 73 / 1000
  let discount_rate : ℚ := 12 / 100
  let market_price : ℚ := 19800
  let processing_fee_rate : ℚ := 25 / 1000
  let conversion_rate : ℚ := 50
  abs ((calculate_savings original_tax_rate reduced_tax_rate discount_rate 
                     market_price processing_fee_rate conversion_rate) - 53.65) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_theorem_l1243_124391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1243_124369

noncomputable section

/-- The function f(x) = a - 3 / (2^x + 1) -/
def f (a : ℝ) (x : ℝ) : ℝ := a - 3 / (2^x + 1)

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- f is an increasing function -/
def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem f_properties :
  ∃ a : ℝ, (is_odd_function (f a)) ∧ (a = 3/2) ∧ (is_increasing_function (f a)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1243_124369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1243_124316

def Quadrant := Fin 4

def in_third_quadrant (α : ℝ) : Prop :=
  Real.pi < α ∧ α < 3 * Real.pi / 2

theorem angle_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.tan α > 0) : 
  in_third_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l1243_124316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_covering_possible_l1243_124384

/-- Represents a square window -/
structure Window where
  side : ℝ
  area : ℝ

/-- The original window -/
noncomputable def original_window : Window where
  side := 1
  area := 1

/-- The covered area of the window -/
noncomputable def covered_area : ℝ := 1/2

/-- The remaining window after covering -/
noncomputable def remaining_window : Window where
  side := 1
  area := 1/2

/-- Theorem stating that it's possible to cover half the area of a square window
    with side length 1 and still have a square window with side length 1 remaining -/
theorem window_covering_possible :
  ∃ (w : Window), 
    w.side = original_window.side ∧ 
    w.area = original_window.area - covered_area ∧
    w.side = remaining_window.side := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_covering_possible_l1243_124384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_square_wire_frame_l1243_124326

/-- Represents a rectangular wire frame -/
structure RectangularWireFrame where
  length : ℝ
  width : ℝ

/-- Represents a square wire frame -/
structure SquareWireFrame where
  side : ℝ

/-- Transforms a rectangular wire frame into a square wire frame -/
noncomputable def transform_to_square (rect : RectangularWireFrame) : SquareWireFrame :=
  { side := (2 * (rect.length + rect.width)) / 4 }

theorem rectangular_to_square_wire_frame :
  let rect := RectangularWireFrame.mk 12 8
  let square := transform_to_square rect
  square.side = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_square_wire_frame_l1243_124326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_theorem_l1243_124397

/-- The sum of the series 1 + 3x + 5x^2 + 7x^3 + ... -/
noncomputable def S (x : ℝ) : ℝ := (1 + x) / (1 - x)

/-- The theorem states that if S(x) = 16, then x = 15/17 -/
theorem series_sum_theorem (x : ℝ) : S x = 16 → x = 15 / 17 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_theorem_l1243_124397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_five_thirds_l1243_124302

-- Define the function g
noncomputable def g (a b : ℝ) : ℝ :=
  if a^b + b ≤ 5 then
    (a^2 * b - a + 1) / (3 * a)
  else
    (a * b^2 - b + 1) / (-3 * b)

-- State the theorem
theorem g_sum_equals_five_thirds :
  g 1 2 + g 1 3 = 5/3 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_five_thirds_l1243_124302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y4_in_x_plus_y_to_7_l1243_124362

theorem coefficient_x3y4_in_x_plus_y_to_7 (X Y : Polynomial ℚ) :
  (X + Y)^7 =
  35 * X^3 * Y^4 + 
  (Finset.range 8).sum (λ k => 
    if k ≠ 3 then Nat.choose 7 k * X^k * Y^(7 - k) 
    else 0) :=
by sorry

#check coefficient_x3y4_in_x_plus_y_to_7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y4_in_x_plus_y_to_7_l1243_124362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_fraction_l1243_124332

theorem min_value_of_fraction (a b c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + a*b < 0 ↔ 1 < x ∧ x < c) →
  a > b →
  (∃ m : ℝ, (∀ a' b' : ℝ, a' > b' → (a'^2 + b'^2) / (a' - b') ≥ m) ∧ 
    (∃ a₀ b₀ : ℝ, a₀ > b₀ ∧ (a₀^2 + b₀^2) / (a₀ - b₀) = m)) ∧
  (∀ m : ℝ, (∀ a' b' : ℝ, a' > b' → (a'^2 + b'^2) / (a' - b') ≥ m) →
    (∃ a₀ b₀ : ℝ, a₀ > b₀ ∧ (a₀^2 + b₀^2) / (a₀ - b₀) = m) →
    m ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_fraction_l1243_124332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1243_124357

/-- A function f(x) = ax² + bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  lower_bound : ∀ x : ℝ, 2 * x ≤ a * x^2 + b * x + c
  upper_bound : ∀ x : ℝ, a * x^2 + b * x + c ≤ (1/2) * (x + 1)^2

/-- The function g(x) defined on [-2, 2] -/
def g (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c + 2 * f.a * |x - 1|

theorem quadratic_function_properties (f : QuadraticFunction) :
  f.a * 1^2 + f.b * 1 + f.c = 2 ∧
  0 < f.a ∧ f.a < 1/2 ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, g f x ≥ -1) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, g f x = -1) →
  f.a = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1243_124357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_scalar_mult_l1243_124307

def p : Fin 3 → ℝ := ![3, -4, 6]
def q : Fin 3 → ℝ := ![-2, 5, -3]

theorem vector_subtraction_scalar_mult :
  (fun i => p i - 5 * q i) = ![13, -29, 21] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_subtraction_scalar_mult_l1243_124307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1243_124303

theorem inequality_equivalence (x : ℝ) : 
  3 * ((7 : ℝ) ^ (2 * x)) + 37 * ((140 : ℝ) ^ x) < 26 * ((20 : ℝ) ^ (2 * x)) ↔ 
  x ≥ Real.log (2 / 3) / Real.log (7 / 20) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l1243_124303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_prime_power_l1243_124367

theorem cube_sum_prime_power (a b p n : ℕ) (hp : Nat.Prime p) 
  (ha : a > 0) (hb : b > 0) (hp_pos : p > 0) (hn : n > 0)
  (heq : a^3 + b^3 = p^n) :
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
            (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_prime_power_l1243_124367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_grid_problem_l1243_124355

theorem blue_grid_problem : 
  let count_solutions := Finset.filter (fun p : ℕ × ℕ => 
    let (n, m) := p
    n > 0 ∧ m > 0 ∧ (m + n - 1 : ℚ) / (m * n : ℚ) = 1 / 2010) (Finset.range 4020 ×ˢ Finset.range 4020)
  Finset.card count_solutions = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_grid_problem_l1243_124355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_overlap_area_l1243_124354

theorem chessboard_overlap_area (n : ℕ) (h : n = 8) :
  let total_area := (n : ℝ) ^ 2
  let overlap_area := total_area * (Real.sqrt 2 - 1)
  let black_square_area := overlap_area / 4
  black_square_area * 2 = 32 * (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_overlap_area_l1243_124354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_prime_power_solutions_l1243_124333

theorem quadratic_prime_power_solutions (x : ℤ) :
  (∃ (p : ℕ) (k : ℕ), Prime p ∧ 2 * x^2 + x - 6 = p^(k : ℕ)) ↔ x = -3 ∨ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_prime_power_solutions_l1243_124333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_l1243_124314

/-- The function f(x) = sin(x - π/3) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/3)

/-- The axis of symmetry for f(x) -/
noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 5*Real.pi/6 + k*Real.pi

/-- Theorem: The axis of symmetry for f(x) = sin(x - π/3) is x = 5π/6 + kπ, where k ∈ ℤ -/
theorem symmetry_axis_of_sine (x : ℝ) :
  ∃ (k : ℤ), f (axis_of_symmetry k + x) = f (axis_of_symmetry k - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_sine_l1243_124314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_speed_difference_l1243_124317

/-- Represents a runner with their race characteristics -/
structure Runner where
  total_distance : ℚ
  total_time : ℚ
  initial_distance : ℚ
  initial_time : ℚ

/-- Calculates the top speed of a runner -/
def top_speed (r : Runner) : ℚ :=
  (r.total_distance - r.initial_distance) / (r.total_time - r.initial_time)

/-- John's race characteristics -/
def john : Runner :=
  { total_distance := 100
    total_time := 13
    initial_distance := 4
    initial_time := 1 }

/-- James' race characteristics -/
def james : Runner :=
  { total_distance := 100
    total_time := 11
    initial_distance := 10
    initial_time := 2 }

/-- Theorem stating the difference in top speeds -/
theorem top_speed_difference :
  top_speed james - top_speed john = 2 := by
  -- Expand the definitions and simplify
  unfold top_speed john james
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_speed_difference_l1243_124317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probabilities_sum_to_one_probabilities_approximately_correct_l1243_124331

noncomputable section

-- Define the production rates for each machine
def production_rate_a : ℝ := 0.20
def production_rate_b : ℝ := 0.35
def production_rate_c : ℝ := 0.45

-- Define the defect rates for each machine
def defect_rate_a : ℝ := 0.03
def defect_rate_b : ℝ := 0.02
def defect_rate_c : ℝ := 0.04

-- Define the total probability of a defective product
def total_defect_prob : ℝ := 
  production_rate_a * defect_rate_a + 
  production_rate_b * defect_rate_b + 
  production_rate_c * defect_rate_c

-- Define the probability that a defective product was produced by each machine
def prob_defective_a : ℝ := (production_rate_a * defect_rate_a) / total_defect_prob
def prob_defective_b : ℝ := (production_rate_b * defect_rate_b) / total_defect_prob
def prob_defective_c : ℝ := (production_rate_c * defect_rate_c) / total_defect_prob

-- Theorem: The sum of probabilities equals 1
theorem probabilities_sum_to_one : 
  prob_defective_a + prob_defective_b + prob_defective_c = 1 := by
  sorry

-- Theorem: The probabilities are approximately correct
theorem probabilities_approximately_correct : 
  (abs (prob_defective_a - 0.1936) < 0.0001) ∧ 
  (abs (prob_defective_b - 0.2258) < 0.0001) ∧ 
  (abs (prob_defective_c - 0.5806) < 0.0001) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probabilities_sum_to_one_probabilities_approximately_correct_l1243_124331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ds_superset_l1243_124381

/-- A DS-set is a finite set of positive integers where each element divides the sum of all elements. -/
def isDSSet (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, x > 0 → (S.sum id) % x = 0

/-- Every finite set of positive integers is a subset of some DS-set. -/
theorem exists_ds_superset (A : Finset ℕ) (h : ∀ a ∈ A, a > 0) : 
  ∃ B : Finset ℕ, A ⊆ B ∧ isDSSet B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_ds_superset_l1243_124381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1243_124339

/-- The area of a right triangle with hypotenuse h and one leg three times the length of the other is 3h²/20. -/
theorem right_triangle_area (h : ℝ) (hpos : h > 0) : ∃ (a : ℝ), 
  a > 0 ∧ 
  a^2 + (3*a)^2 = h^2 ∧ 
  (1/2 * a * (3*a)) = (3 * h^2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1243_124339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equality_l1243_124325

open Real

theorem tan_alpha_equality (α : ℝ) :
  tan α = 8 * Real.sin (70 * π / 180) * Real.cos (10 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) →
  α = π / 3 ∨ α = 4 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_equality_l1243_124325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_eating_time_l1243_124368

/-- The time it takes for three people to eat a certain number of pizza slices -/
def time_to_eat (fast_rate slow_rate steady_rate : ℚ) (total_slices : ℕ) : ℕ :=
  let combined_rate := fast_rate + slow_rate + steady_rate
  let exact_time := (total_slices : ℚ) / combined_rate
  (exact_time + 1/2).floor.toNat

/-- Theorem stating the time it takes for Mr. Fast, Mr. Slow, and Mr. Steady to eat 24 pizza slices -/
theorem pizza_eating_time : 
  time_to_eat (1/5) (1/10) (1/8) 24 = 56 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_eating_time_l1243_124368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_equals_twelve_implies_n_equals_three_l1243_124310

-- Define the curve
noncomputable def y (n : ℝ) (x : ℝ) : ℝ := x^n

-- Define the derivative of the curve
noncomputable def y_derivative (n : ℝ) (x : ℝ) : ℝ := n * x^(n - 1)

-- Theorem statement
theorem derivative_at_two_equals_twelve_implies_n_equals_three :
  ∀ n : ℝ, y_derivative n 2 = 12 → n = 3 := by
  intro n h
  -- The proof goes here
  sorry

#check derivative_at_two_equals_twelve_implies_n_equals_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_equals_twelve_implies_n_equals_three_l1243_124310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_l1243_124337

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the left focus
def left_focus : ℝ × ℝ := (-3, 0)

-- Define the objective function
noncomputable def objective (x y : ℝ) : ℝ :=
  distance (-2) 2 x y + (5/3) * distance x y (left_focus.1) (left_focus.2)

-- State the theorem
theorem ellipse_minimum :
  ∀ x y : ℝ, is_on_ellipse x y →
  objective x y ≥ objective (-5 * Real.sqrt 3 / 2) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_l1243_124337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1243_124321

variable (a b : ℝ)

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 ∧ y > 0 → f x * f y = y^a * f (x/2) + x^b * f (y/2)

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation a b f →
    (∀ x : ℝ, x > 0 → f x = 0) ∨
    (a = b ∧ ∃ c : ℝ, c = 2^(1-a) ∧ ∀ x : ℝ, x > 0 → f x = c * x^a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l1243_124321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1243_124351

-- Define the circle and chord properties
def circle_radius : ℝ := 10

-- Theorem statement
theorem chord_length (O A C D : ℝ × ℝ) :
  let r := circle_radius
  let OA := Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let M := ((A.1 + O.1) / 2, (A.2 + O.2) / 2)
  (OA = r) →
  (Real.sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) = r) →
  (Real.sqrt ((D.1 - O.1)^2 + (D.2 - O.2)^2) = r) →
  ((M.1 - O.1) * (C.1 - D.1) + (M.2 - O.2) * (C.2 - D.2) = 0) →  -- Perpendicularity
  (M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) →                     -- Bisection
  CD = 10 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1243_124351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_quadratic_with_eight_roots_in_arithmetic_sequence_l1243_124322

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p*x + q

/-- The composition of f with itself three times -/
def f_cubed (p q x : ℝ) : ℝ := f p q (f p q (f p q x))

/-- Predicate to check if a list of real numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (l : List ℝ) : Prop :=
  l.length > 1 ∧ ∀ i, i + 2 < l.length → l.get! (i+2) - l.get! (i+1) = l.get! (i+1) - l.get! i

theorem no_quadratic_with_eight_roots_in_arithmetic_sequence :
  ¬∃ (p q : ℝ), ∃ (roots : List ℝ),
    roots.length = 8 ∧
    roots.Nodup ∧
    is_arithmetic_sequence roots ∧
    ∀ x ∈ roots, f_cubed p q x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_quadratic_with_eight_roots_in_arithmetic_sequence_l1243_124322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_coverage_probability_l1243_124382

/-- The probability of a coin covering part of the black region in a specially colored square. -/
theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (center_circle_radius : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  center_circle_radius = 2 →
  coin_diameter = 2 →
  ∃ (p : ℝ), p = (9 * Real.pi + 18) / 64 ∧ 
    p = (area_of_black_region_plus_buffer / total_possible_area) :=
by
  intro h_square h_triangle h_circle h_coin
  -- Define the areas
  let total_possible_area := (square_side - coin_diameter) ^ 2
  let central_circle_area := Real.pi * (center_circle_radius + coin_diameter / 2) ^ 2
  let triangle_buffer_area := 4 * (triangle_leg ^ 2 / 2 + triangle_leg)
  let area_of_black_region_plus_buffer := central_circle_area + triangle_buffer_area
  -- Provide the probability
  use area_of_black_region_plus_buffer / total_possible_area
  constructor
  · -- Prove the numerical equality
    sorry
  · -- Prove that this is indeed the correct probability
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_coverage_probability_l1243_124382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_value_l1243_124358

/-- s is the positive real solution to x^3 + 3/7 * x - 1 = 0 -/
noncomputable def s : ℝ := Real.sqrt (Real.sqrt (7/3))

/-- The infinite series s^2 + 3s^5 + 5s^8 + 7s^11 + ... -/
noncomputable def T : ℝ := ∑' n, (2*n + 1) * s^(3*n + 2)

/-- Theorem stating that the infinite series equals 7/3 -/
theorem infinite_series_value : T = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_value_l1243_124358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_exp_iff_m_le_half_open_l1243_124385

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x * Real.log (x + 1) + x + 1

/-- The theorem stating the range of m for which f(x) ≤ e^x holds for all x ≥ 0 -/
theorem f_le_exp_iff_m_le_half_open (m : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f m x ≤ Real.exp x) ↔ m ≤ (1/2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_exp_iff_m_le_half_open_l1243_124385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l1243_124306

noncomputable section

-- Define the radius of the original circle
def original_radius : ℝ := 6

-- Define the arc length of the half-sector (which becomes the circumference of the cone's base)
def base_circumference : ℝ := Real.pi * original_radius

-- Define the radius of the cone's base
def base_radius : ℝ := base_circumference / (2 * Real.pi)

-- Define the height of the cone using the Pythagorean theorem
def cone_height : ℝ := Real.sqrt (original_radius^2 - base_radius^2)

-- State the theorem
theorem cone_volume_from_half_sector :
  (1/3) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l1243_124306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l1243_124311

theorem intersection_angle (n m : ℕ) (hn : n > 4) (hm : m > 3) (hmn : m < n) :
  let larger_polygon_angle := (n - 2) * 180 / n
  let smaller_polygon_angle := (m - 2) * 180 / m
  let half_external_angle := 180 / n
  let new_angle_at_intersection := 2 * half_external_angle + smaller_polygon_angle
  new_angle_at_intersection = 360 / n + (m - 2) * 180 / m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l1243_124311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_coins_l1243_124394

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_array_coins :
  ∃ (N : ℕ), triangular_sum N = 2016 ∧ 
  (N.repr.toList.map (λ c => c.toString.toNat!)).sum = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_coins_l1243_124394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_region_area_proof_l1243_124319

/-- The area of a region bounded by three 90-degree arcs of circles with radius 5 units,
    intersecting at points of tangency -/
noncomputable def bounded_region_area : ℝ := 18.75 * Real.pi - 37.5

/-- The radius of each circle -/
def circle_radius : ℝ := 5

/-- The central angle of each arc in radians -/
noncomputable def arc_angle : ℝ := Real.pi / 2

theorem bounded_region_area_proof :
  bounded_region_area = 3 * ((circle_radius^2 * arc_angle / 2) - (circle_radius^2 / 2)) := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval bounded_region_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_region_area_proof_l1243_124319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1243_124305

noncomputable def curve_C (φ : Real) : Real × Real :=
  (Real.sqrt 3 * Real.cos φ, Real.sin φ)

noncomputable def line_l (θ : Real) : Real × Real :=
  let ρ := 4 / (Real.cos θ + Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def distance (p q : Real × Real) : Real :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_curve_to_line :
  ∃ (min_dist : Real),
    ∀ (φ θ : Real),
      distance (curve_C φ) (line_l θ) ≥ min_dist ∧
      ∃ (φ₀ θ₀ : Real),
        distance (curve_C φ₀) (line_l θ₀) = min_dist ∧
        min_dist = |2 * Real.sin (φ₀ + π/3) - 4| / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1243_124305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_one_l1243_124377

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + 1/(2^(x+2))

-- Theorem statement
theorem f_min_at_neg_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_neg_one_l1243_124377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l1243_124350

/-- The slope angle of a line is the angle between the line and the positive x-axis. -/
noncomputable def slope_angle (a b : ℝ) : ℝ :=
  Real.arctan (a / b)

/-- The line equation is in the form ax + by + c = 0 -/
def is_line_equation (a b : ℝ) : Prop :=
  a ≠ 0 ∨ b ≠ 0

theorem slope_angle_of_line (a b : ℝ) (h : is_line_equation a b) :
  slope_angle a b = Real.pi / 4 ↔ a = -b ∧ a ≠ 0 := by
  sorry

#check slope_angle_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l1243_124350
