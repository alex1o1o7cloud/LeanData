import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l738_73811

/-- Represents a rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  c₁ : ℝ  -- y-coordinate of bottom side
  c₂ : ℝ  -- y-coordinate of top side
  d₁ : ℝ  -- x-coordinate of left side
  d₂ : ℝ  -- x-coordinate of right side
  h₁ : c₁ < c₂
  h₂ : d₁ < d₂

/-- Predicate for points satisfying the distance condition -/
def satisfiesCondition (r : Rectangle) (x y : ℝ) : Prop :=
  |y - r.c₁| + |y - r.c₂| = |x - r.d₁| + |x - r.d₂|

/-- The locus of points satisfying the distance condition -/
def locus (r : Rectangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | satisfiesCondition r p.1 p.2}

/-- Main theorem stating the geometric properties of the locus -/
theorem locus_characterization (r : Rectangle) :
  ∃ (s₁ s₂ : Set (ℝ × ℝ)) (l₁ l₂ l₃ l₄ : Set (ℝ × ℝ)),
    locus r = s₁ ∪ s₂ ∪ l₁ ∪ l₂ ∪ l₃ ∪ l₄ ∧
    (∀ p ∈ s₁, p.2 = (r.c₁ + r.c₂) / 2 ∧ r.d₁ ≤ p.1 ∧ p.1 ≤ r.d₂) ∧
    (∀ p ∈ s₂, p.2 = (r.c₁ + r.c₂) / 2 ∧ r.d₁ ≤ p.1 ∧ p.1 ≤ r.d₂) ∧
    (∀ i ∈ ({1, 2, 3, 4} : Set ℕ), ∃ m b,
      (i = 1 → ∀ p ∈ l₁, p.2 = m * p.1 + b ∧ p.1 ≤ r.d₁ ∧ p.2 ≤ r.c₁) ∧
      (i = 2 → ∀ p ∈ l₂, p.2 = m * p.1 + b ∧ p.1 ≥ r.d₂ ∧ p.2 ≤ r.c₁) ∧
      (i = 3 → ∀ p ∈ l₃, p.2 = m * p.1 + b ∧ p.1 ≤ r.d₁ ∧ p.2 ≥ r.c₂) ∧
      (i = 4 → ∀ p ∈ l₄, p.2 = m * p.1 + b ∧ p.1 ≥ r.d₂ ∧ p.2 ≥ r.c₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l738_73811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_Q_equation_no_perpendicular_bisector_l738_73840

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the line l₁
def line_l₁ (x y : ℝ) (m : ℝ) : Prop := y = m * (x - P.1) + P.2

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Statement for part (1)
theorem circle_Q_equation (M N : ℝ × ℝ) (m : ℝ) : 
  circle_C M.1 M.2 ∧ circle_C N.1 N.2 ∧ 
  line_l₁ M.1 M.2 m ∧ line_l₁ N.1 N.2 m ∧ 
  distance M N = 4 →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 ↔ 
    distance (x, y) M = distance (x, y) N ∧ 
    distance M N = distance (x, y) M + distance (x, y) N := 
by sorry

-- Define the line ax - y + 1 = 0
def line_AB (x y a : ℝ) : Prop := a*x - y + 1 = 0

-- Statement for part (2)
theorem no_perpendicular_bisector :
  ¬∃ a : ℝ, ∀ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
    line_AB A.1 A.2 a ∧ line_AB B.1 B.2 a →
    ∃ m : ℝ, line_l₁ ((A.1 + B.1)/2) ((A.2 + B.2)/2) m ∧
    (B.2 - A.2) * (P.1 - A.1) = (A.1 - B.1) * (P.2 - A.2) := 
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_Q_equation_no_perpendicular_bisector_l738_73840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x2_f_lt_g_l738_73872

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (2*a + 1) * x + 2 * Real.log x

/-- The function g(x) defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x

/-- The main theorem to be proved -/
theorem exists_x2_f_lt_g (a : ℝ) (h_a : a > 0) :
  ∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Ioo 0 2, f a x₁ < g x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_x2_f_lt_g_l738_73872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BP_equals_two_l738_73824

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C D P : ℝ × ℝ)

-- Define the conditions
axiom on_circle : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle
axiom intersect : P ∈ Set.inter (Set.range (λ t => (1 - t) • A + t • C))
                                (Set.range (λ t => (1 - t) • B + t • D))
axiom AP_length : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 8
axiom PC_length : Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = 1
axiom BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6
axiom BP_less_DP : Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) < Real.sqrt ((D.1 - P.1)^2 + (D.2 - P.2)^2)

-- Theorem to prove
theorem BP_equals_two : Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BP_equals_two_l738_73824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_t_for_g_defined_range_of_t_for_f_leq_g_l738_73803

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log (x + 1)
noncomputable def g (x t : ℝ) : ℝ := Real.log (2 * x + t)

-- Theorem for the domain of f
theorem domain_of_f :
  {x : ℝ | f x ∈ Set.univ} = {x : ℝ | x > -1} := by sorry

-- Theorem for the range of t when g is defined on [0, 1]
theorem range_of_t_for_g_defined :
  ∀ t : ℝ, (∀ x ∈ Set.Icc 0 1, g x t ∈ Set.univ) ↔ t > -2 := by sorry

-- Theorem for the range of t when f ≤ g on [0, 1]
theorem range_of_t_for_f_leq_g :
  ∀ t : ℝ, (∀ x ∈ Set.Icc 0 1, f x ≤ g x t) ↔ t ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_t_for_g_defined_range_of_t_for_f_leq_g_l738_73803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_3sin_max_l738_73871

theorem cos_plus_3sin_max (x : ℝ) : Real.cos x + 3 * Real.sin x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_3sin_max_l738_73871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_side_line_range_l738_73857

/-- Given a line 3x - 2y - a = 0 and two points (-3, -1) and (4, -6) on the same side of this line,
    the range of values for a is (-∞, -7) ∪ (24, +∞). -/
theorem same_side_line_range (a : ℝ) : 
  let line := fun x y ↦ 3 * x - 2 * y - a
  let p1 := (-3, -1)
  let p2 := (4, -6)
  (line p1.1 p1.2) * (line p2.1 p2.2) > 0 → 
  (a < -7 ∨ a > 24) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_side_line_range_l738_73857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_fourth_l738_73897

/-- The series term for a given n -/
noncomputable def seriesTerm (n : ℕ) : ℝ := 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2))

/-- The sum of the infinite series -/
noncomputable def seriesSum : ℝ := ∑' n, seriesTerm n

/-- Theorem stating that the sum of the infinite series is equal to 1/4 -/
theorem series_sum_is_one_fourth : seriesSum = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_fourth_l738_73897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ratio_from_cosine_sum_diff_l738_73834

theorem sine_ratio_from_cosine_sum_diff (a b : ℝ) 
  (h1 : Real.cos (a + b) = 1/4) 
  (h2 : Real.cos (a - b) = 3/4) : 
  (Real.sin a / Real.sin b = 1) ∨ (Real.sin a / Real.sin b = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_ratio_from_cosine_sum_diff_l738_73834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_123_l738_73809

/-- The repeating decimal 0.123123... is equal to 41/333 -/
theorem repeating_decimal_123 : ∀ x : ℚ, x = 0.123123123 → x = 41 / 333 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_123_l738_73809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l738_73882

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 2

-- Define the inverse function of f
def f_inv (x : ℝ) : ℝ := (x - 2) / 3

-- Theorem statement
theorem sum_of_solutions_is_eight :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  f_inv x₁ = f (x₁⁻¹) ∧
  f_inv x₂ = f (x₂⁻¹) ∧
  x₁ + x₂ = 8 ∧
  ∀ (x : ℝ), f_inv x = f (x⁻¹) → (x = x₁ ∨ x = x₂) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l738_73882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_raffle_winnings_l738_73804

/-- The initial amount won at a charity raffle -/
def initial_amount : ℚ := 418

/-- The amount left after all expenses -/
def final_amount : ℚ := 240

theorem charity_raffle_winnings :
  let after_donation := initial_amount * (3/4)
  let after_lunch := after_donation * (9/10)
  let after_gift := after_lunch * (17/20)
  ⌊after_gift⌋ = final_amount := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_raffle_winnings_l738_73804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_calculation_l738_73855

/-- Calculates the annual income from a stock investment -/
noncomputable def annual_income (investment_amount : ℝ) (dividend_rate : ℝ) (stock_price : ℝ) (face_value : ℝ) : ℝ :=
  let num_stocks := investment_amount / stock_price
  let dividend_per_stock := dividend_rate * face_value
  num_stocks * dividend_per_stock

/-- Theorem: The annual income from investing $6800 in a 30% dividend stock at $136 per share with $100 face value is $1500 -/
theorem investment_income_calculation :
  annual_income 6800 0.3 136 100 = 1500 := by
  -- Unfold the definition of annual_income
  unfold annual_income
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_calculation_l738_73855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_and_function_properties_l738_73892

noncomputable section

open Real

def f (x : ℝ) := sin (2 * x - π / 6) - cos (2 * x)

theorem triangle_angles_and_function_properties
  (A B C : ℝ)  -- Angles of the triangle
  (a b c : ℝ)  -- Sides of the triangle
  (h1 : f (B / 2) = -sqrt 3 / 2)
  (h2 : b = 1)
  (h3 : c = sqrt 3)
  (h4 : a > b) :
  (∃ (T : ℝ), T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧  -- Smallest positive period
  (∃ (M : ℝ), ∀ x, f x ≤ M ∧ ∃ x, f x = M) ∧  -- Maximum value exists
  (∃ (S : Set ℝ), S = {x : ℝ | ∃ k : ℤ, x = k * π + 5 * π / 12} ∧ ∀ x, f x = sqrt 3 ↔ x ∈ S) ∧  -- Set of x for maximum value
  B = π / 6 ∧
  C = π / 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_and_function_properties_l738_73892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l738_73854

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of line l₁: ax + 2y - 1 = 0 -/
noncomputable def slope_l₁ (a : ℝ) : ℝ := -a / 2

/-- The slope of line l₂: x + (a+1)y + 4 = 0 -/
noncomputable def slope_l₂ (a : ℝ) : ℝ := -1 / (a + 1)

/-- The condition "a = 1" is sufficient but not necessary for the parallelism of l₁ and l₂ -/
theorem parallel_condition (a : ℝ) :
  (a = 1 → are_parallel (slope_l₁ a) (slope_l₂ a)) ∧
  ¬(are_parallel (slope_l₁ a) (slope_l₂ a) → a = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l738_73854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_solution_l738_73817

/-- Represents the possible answers to the question -/
inductive Answer
  | A
  | B
  | C
  | D

/-- The correct answer to the question -/
def correct_answer : Answer := Answer.C

/-- A theorem stating that the correct answer is C -/
theorem question_solution : correct_answer = Answer.C := by
  -- The proof is trivial since we defined correct_answer as Answer.C
  rfl

#check question_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_solution_l738_73817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l738_73894

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.log (x + 1) + 2 * x - a)

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 1 ∧ f a (f a x₀) = x₀) →
  a ∈ Set.Icc (-1) (2 + Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l738_73894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_probability_l738_73888

/-- Represents a parking lot with a fixed number of spaces -/
structure ParkingLot where
  total_spaces : ℕ
  occupied_spaces : ℕ
  (occupied_le_total : occupied_spaces ≤ total_spaces)

/-- Calculates the probability of finding two adjacent empty spaces in a parking lot -/
def prob_two_adjacent_empty (p : ParkingLot) : ℚ :=
  1 - (Nat.choose (p.total_spaces - p.occupied_spaces + 5) 5 : ℚ) / (Nat.choose p.total_spaces p.occupied_spaces : ℚ)

/-- The main theorem stating the probability of finding two adjacent empty spaces
    in a specific parking scenario -/
theorem parking_probability :
  let p : ParkingLot := ⟨20, 14, by norm_num⟩
  prob_two_adjacent_empty p = 850 / 922 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_probability_l738_73888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_paper_area_l738_73843

theorem unused_paper_area (π : Real) (h_pi : π > 0) : 
  100 - 25 * π = 
  let square_side : Real := 10
  let square_area : Real := square_side ^ 2
  let circle_diameter : Real := square_side / 3
  let circle_radius : Real := circle_diameter / 2
  let circle_area : Real := π * circle_radius ^ 2
  let total_circles_area : Real := 9 * circle_area
  square_area - total_circles_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unused_paper_area_l738_73843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_modulo_prime_l738_73881

theorem distinct_remainders_modulo_prime (p : ℕ) (hp : Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, (Finset.univ.filter (λ i : Fin p ↦ ∃ j : Fin p, (a i + i.val * k) % p = (a j + j.val * k) % p)).card ≥ (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_modulo_prime_l738_73881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_parabola_conditions_l738_73895

/-- The standard equation of a circle given specific conditions on a parabola --/
theorem circle_equation_from_parabola_conditions (m p : ℝ) (h1 : m > 0) (h2 : p > 0) :
  let A : ℝ × ℝ := (4, m)
  let on_parabola := m^2 = 2 * p * 4
  let F := (p / 2, 0)  -- Focus of the parabola
  let AF_length := ((4 - p / 2)^2 + m^2).sqrt
  let chord_length := 6
  on_parabola → AF_length^2 = 4^2 + (chord_length / 2)^2 →
  (fun x y => (x - 4)^2 + (y - 4)^2 = 25) = 
  (fun x y => (x - A.1)^2 + (y - A.2)^2 = AF_length^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_parabola_conditions_l738_73895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l738_73833

theorem log_inequality_condition (x : ℝ) : 
  (∀ (a b : ℝ), a > 0 → b > 0 → 2 * Real.log ((a + b) / 2) ≤ Real.log a + Real.log b) ↔ 
  (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l738_73833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tan_a8_l738_73866

open Real

theorem arithmetic_sequence_tan_a8 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n / 2 : ℝ) * (a 1 + a n)) →  -- Definition of sum for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 15 = 25 * π →                 -- Given condition
  tan (a 8) = -sqrt 3 :=           -- Conclusion to prove
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tan_a8_l738_73866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumcenter_intersection_l738_73883

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

-- Define the given circles and intersection points
noncomputable def circle1 : Circle := sorry
noncomputable def circle2 : Circle := sorry
noncomputable def circle3 : Circle := sorry

noncomputable def A0 : Point := sorry
noncomputable def A1 : Point := sorry
noncomputable def B0 : Point := sorry
noncomputable def B1 : Point := sorry
noncomputable def C0 : Point := sorry
noncomputable def C1 : Point := sorry

-- Define the circumcenter of a triangle
noncomputable def circumcenter (p1 p2 p3 : Point) : Point := sorry

-- Define the lines through circumcenters
noncomputable def line_through_circumcenters (i j k : ℕ) : Set (ℝ × ℝ) := sorry

-- Define a membership relation for Point in Set (ℝ × ℝ)
instance : Membership Point (Set (ℝ × ℝ)) where
  mem p s := (p.x, p.y) ∈ s

-- Main theorem
theorem circle_circumcenter_intersection :
  (∃ p : Point, ∀ i j k : ℕ, p ∈ line_through_circumcenters i j k) ∨
  (∀ i j k l m n : ℕ, i ≠ l ∨ j ≠ m ∨ k ≠ n →
    line_through_circumcenters i j k = line_through_circumcenters l m n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumcenter_intersection_l738_73883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l738_73829

/-- Represents a line in 2D space with equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculates the distance between two parallel lines -/
noncomputable def distance (l1 l2 : Line) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

theorem parallel_line_equation (l1 : Line) (d : ℝ) :
  l1.a = 5 ∧ l1.b = -12 ∧ l1.c = 6 →
  ∃ l : Line,
    parallel l l1 ∧
    distance l l1 = 2 →
    (l.a = 5 ∧ l.b = -12 ∧ l.c = 32) ∨
    (l.a = 5 ∧ l.b = -12 ∧ l.c = -20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l738_73829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_coprime_exists_l738_73808

def sequenceF (n : ℕ) : ℕ := 2^n - 3

theorem subsequence_coprime_exists :
  ∃ (f : ℕ → ℕ), (∀ i j, i < j → f i < f j) ∧
    (∀ i, 1 < f i) ∧
    (∀ i j, i ≠ j → Nat.Coprime (sequenceF (f i)) (sequenceF (f j))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_coprime_exists_l738_73808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_fixed_line_triangle_area_l738_73810

noncomputable section

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 4 = 1

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop := y = (1/3) * x + (Real.sqrt 2 * (-13/7))

-- Define point P
def P : ℝ × ℝ := (3 * Real.sqrt 2, Real.sqrt 2)

-- Define points A and B as the intersection of line l and ellipse C
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State that P is to the upper left of line l
axiom P_position : line_l P.1 P.2 → False

-- Define necessary auxiliary functions
def is_incenter (I A B P : ℝ × ℝ) : Prop := sorry
def angle (A P B : ℝ × ℝ) : ℝ := sorry
def area_triangle (A B P : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: The center of the incircle of triangle PAB lies on x = 3√2
theorem incenter_on_fixed_line : 
  ∃ (I : ℝ × ℝ), is_incenter I A B P ∧ I.1 = 3 * Real.sqrt 2 := by sorry

-- Theorem 2: If angle APB = 60°, the area of triangle PAB is (117√3)/49
theorem triangle_area (h : angle A P B = π/3) : 
  area_triangle A B P = (117 * Real.sqrt 3) / 49 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_on_fixed_line_triangle_area_l738_73810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l738_73863

open InnerProductSpace

theorem cos_theta_value 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) 
  (ha : ‖a‖ = 4) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a - 2 • b‖ = 4) : 
  inner a b / (‖a‖ * ‖b‖) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l738_73863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_three_element_set_l738_73889

theorem proper_subsets_of_three_element_set :
  ∀ (S : Finset ℕ), Finset.card S = 3 →
  Finset.card (Finset.powerset S \ {S}) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_three_element_set_l738_73889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_inequality_proof_l738_73814

/-- The function f(x) = ln(x) / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

/-- The function g(x) = e^x -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem function_inequalities (m : ℝ) :
  (∀ x > 0, f x ≤ m * x ∧ m * x ≤ g x) ↔ m ∈ Set.Icc (1 / (2 * Real.exp 1)) (Real.exp 1) := by
  sorry

theorem inequality_proof (x₁ x₂ : ℝ) (h : x₁ > x₂ ∧ x₂ > 0) :
  x₁ * f x₁ - x₂ * f x₂ * (x₁^2 + x₂^2) > 2 * x₂ * (x₁ - x₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_inequality_proof_l738_73814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shape_is_cone_l738_73869

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Describes the shape formed by the equation ρ = c * sin(θ₀) in spherical coordinates -/
def ConeShape (c : ℝ) (θ₀ : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.ρ = c * Real.sin θ₀}

/-- Predicate to represent that a set is a cone surface -/
def IsConeSurface (s : Set SphericalPoint) (apex : SphericalPoint) (axis : ℝ × ℝ × ℝ) (angle : ℝ) : Prop :=
  sorry -- We define this as a placeholder for now

/-- Theorem stating that the shape described by ρ = c * sin(θ₀) is a cone -/
theorem cone_shape_is_cone (c : ℝ) (θ₀ : ℝ) (hc : c > 0) :
  ∃ (apex : SphericalPoint) (axis : ℝ × ℝ × ℝ) (angle : ℝ),
    IsConeSurface (ConeShape c θ₀) apex axis angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shape_is_cone_l738_73869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l738_73887

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 5
  else 2^x

-- State the theorem
theorem f_composition_value : f (f (1/25)) = 1/4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l738_73887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l738_73846

/-- Circle O -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Curve C -/
def curve_C (x y t : ℝ) : Prop := y = 3 * abs (x - t)

/-- Point on curve C -/
def point_on_curve (x y t : ℝ) : Prop := curve_C x y t

/-- Distance ratio condition -/
def distance_ratio (m n s p k : ℝ) : Prop :=
  ∀ x y, circle_O x y →
    ((x - m)^2 + (y - n)^2) / ((x - s)^2 + (y - p)^2) = k^2

/-- Main theorem -/
theorem find_t (m n s p : ℕ+) (k : ℝ) (h_k : k > 1)
  (h_A : point_on_curve m n t)
  (h_B : point_on_curve s p t)
  (h_ratio : distance_ratio m n s p k) :
  t = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_t_l738_73846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l738_73867

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define the point that the line l passes through
def point : ℝ × ℝ := (-3, 0)

-- Define perpendicularity of two lines
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define the slope of a line given its equation ax + by + c = 0
noncomputable def slope_from_equation (a b : ℝ) : ℝ := -a / b

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
  (∃ (m : ℝ), perpendicular m (slope_from_equation 2 (-1))) ∧
  line_l point.1 point.2 →
  line_l x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l738_73867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_twenty_l738_73856

theorem multiple_of_twenty (n : ℕ) : ∃ k : ℤ, 4 * 6^n + 5^n + 1 - 9 = 20 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_twenty_l738_73856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_per_wheel_for_given_vehicle_l738_73839

/-- Represents a three-wheeled vehicle with spare wheels -/
structure Vehicle where
  total_distance : ℕ
  num_wheels_in_use : ℕ
  num_spare_wheels : ℕ

/-- Calculates the distance each wheel travels in a vehicle where all wheels are used equally -/
def distance_per_wheel (v : Vehicle) : ℚ :=
  (v.total_distance * v.num_wheels_in_use : ℚ) / (v.num_wheels_in_use + v.num_spare_wheels)

theorem distance_per_wheel_for_given_vehicle :
  let v : Vehicle := {
    total_distance := 100,
    num_wheels_in_use := 3,
    num_spare_wheels := 2
  }
  distance_per_wheel v = 60 := by
  -- Proof steps would go here
  sorry

#eval distance_per_wheel {
  total_distance := 100,
  num_wheels_in_use := 3,
  num_spare_wheels := 2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_per_wheel_for_given_vehicle_l738_73839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l738_73816

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line parameterization -/
structure Parameterization where
  origin : Vector2D
  direction : Vector2D

/-- The line equation y = -7/4x + 21/4 -/
def lineEquation (x y : ℝ) : Prop := y = -7/4 * x + 21/4

/-- Check if a parameterization is valid for the line -/
noncomputable def isValidParameterization (p : Parameterization) : Prop :=
  ∀ t : ℝ, lineEquation (p.origin.x + t * p.direction.x) (p.origin.y + t * p.direction.y)

/-- The parameterizations to be checked -/
noncomputable def paramA : Parameterization := ⟨⟨7, 0⟩, ⟨4, -7⟩⟩
noncomputable def paramB : Parameterization := ⟨⟨3, 6⟩, ⟨8, 14⟩⟩
noncomputable def paramC : Parameterization := ⟨⟨0, 21/4⟩, ⟨-4, 7⟩⟩
noncomputable def paramD : Parameterization := ⟨⟨7, 7⟩, ⟨1, -7/4⟩⟩
noncomputable def paramE : Parameterization := ⟨⟨-1, 6⟩, ⟨28, -49⟩⟩

/-- Theorem stating which parameterizations are valid -/
theorem valid_parameterizations :
  isValidParameterization paramA ∧
  isValidParameterization paramC ∧
  ¬isValidParameterization paramB ∧
  ¬isValidParameterization paramD ∧
  ¬isValidParameterization paramE := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l738_73816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_closest_to_opposite_hands_l738_73806

def minute_hand_angle (t : ℝ) : ℝ := 6 * t
def hour_hand_angle (t : ℝ) : ℝ := 30 + 0.5 * t

def angle_difference (t : ℝ) : ℝ :=
  |minute_hand_angle (t + 6) - hour_hand_angle (t - 3)|

def is_closest_to_180 (t : ℝ) : Prop :=
  ∀ other : ℝ, other ∈ ({5 + 5/11, 7.5, 10, 15, 17.5} : Set ℝ) →
    |angle_difference t - 180| ≤ |angle_difference other - 180|

theorem time_closest_to_opposite_hands :
  is_closest_to_180 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_closest_to_opposite_hands_l738_73806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l738_73828

/-- The function f(x) defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := 2 * a * Real.log x + b * x

/-- The theorem statement -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_slope : deriv (f a b) 1 = 2) :
  ∀ x y, x > 0 ∧ y > 0 → (8 * x + y) / (x * y) ≥ 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l738_73828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_definition_of_locus_correct_definition_A_correct_definition_B_correct_definition_C_correct_definition_E_l738_73821

-- Define a type for points in a space
variable {Point : Type}

-- Define a predicate for conditions that points might satisfy
variable (condition : Point → Prop)

-- Define a set representing the locus
variable (locus : Set Point)

-- Define the correct definition of a locus
def correct_locus (locus : Set Point) (condition : Point → Prop) : Prop :=
  ∀ p, p ∈ locus ↔ condition p

-- Define the incorrect definition (D)
def incorrect_locus (locus : Set Point) (condition : Point → Prop) : Prop :=
  (∀ p, p ∈ locus → condition p) ∧ 
  (∃ p, condition p ∧ p ∉ locus)

-- Theorem stating that the incorrect definition does not imply the correct definition
theorem incorrect_definition_of_locus :
  ∃ (locus : Set Point) (condition : Point → Prop), incorrect_locus locus condition ∧ ¬(correct_locus locus condition) :=
sorry

-- Theorems stating that other definitions (A, B, C, E) imply the correct definition
theorem correct_definition_A {locus : Set Point} {condition : Point → Prop} :
  (∀ p, p ∈ locus → condition p) ∧ (∀ p, ¬condition p → p ∉ locus) →
  correct_locus locus condition :=
sorry

theorem correct_definition_B {locus : Set Point} {condition : Point → Prop} :
  (∀ p, condition p → p ∈ locus) ∧ (∀ p, p ∈ locus → condition p) →
  correct_locus locus condition :=
sorry

theorem correct_definition_C {locus : Set Point} {condition : Point → Prop} :
  (∀ p, p ∉ locus → ¬condition p) ∧ (∀ p, ¬condition p → p ∉ locus) →
  correct_locus locus condition :=
sorry

theorem correct_definition_E {locus : Set Point} {condition : Point → Prop} :
  (∀ p, condition p → p ∈ locus) ∧ (∀ p, p ∉ locus → ¬condition p) →
  correct_locus locus condition :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_definition_of_locus_correct_definition_A_correct_definition_B_correct_definition_C_correct_definition_E_l738_73821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l738_73893

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < Real.pi / 2)
  (h_intersect : ∃ x₁ x₂, x₁ < x₂ ∧ f ω φ x₁ = 0 ∧ f ω φ x₂ = 0 ∧ x₂ - x₁ = Real.pi / 4)
  (h_point : f ω φ (Real.pi / 3) = -1) :
  (∀ x, f ω φ x = Real.sin (4 * x + Real.pi / 6)) ∧
  (∀ k : ℤ, ∀ x, -Real.pi / 6 + k * Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 12 + k * Real.pi / 2 →
    (∀ y, x ≤ y → f ω φ x ≤ f ω φ y)) ∧
  (∀ k : ℝ, ((-Real.sqrt 3 / 2 < k ∧ k ≤ Real.sqrt 3 / 2) ∨ k = -1) ↔
    (∃! x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ Real.sin (2 * x - Real.pi / 3) + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l738_73893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_y_axis_l738_73842

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.exp (x + 1)

-- Define the point P where the curve intersects the y-axis
noncomputable def P : ℝ × ℝ := (0, f 0)

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := -Real.exp (x + 1)

-- Theorem statement
theorem tangent_line_at_y_axis :
  let slope := f_derivative (P.1)
  let tangent_line (x : ℝ) := slope * (x - P.1) + P.2
  ∀ x, tangent_line x = -Real.exp 1 * x - Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_y_axis_l738_73842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_ten_thirds_pi_l738_73875

theorem sin_negative_ten_thirds_pi : 
  Real.sin (-10/3 * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_ten_thirds_pi_l738_73875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_original_price_l738_73880

-- Define the reduced prices
noncomputable def shirt_reduced_price : ℝ := 6
noncomputable def pants_reduced_price : ℝ := 12

-- Define the discount percentages
noncomputable def shirt_discount_percent : ℝ := 25
noncomputable def pants_discount_percent : ℝ := 40

-- Define the original prices
noncomputable def shirt_original_price : ℝ := shirt_reduced_price / (shirt_discount_percent / 100)
noncomputable def pants_original_price : ℝ := pants_reduced_price / (1 - pants_discount_percent / 100)

-- Theorem to prove
theorem combined_original_price :
  shirt_original_price + pants_original_price = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_original_price_l738_73880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cylinder_height_l738_73890

/-- Right circular cylinder with given dimensions -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Smaller cylinder cut from the original cylinder -/
structure SmallerCylinder where
  height : ℝ

/-- Cylindrical frustum remaining after cutting the smaller cylinder -/
structure CylindricalFrustum where
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

/-- Volume of the smaller cylinder -/
noncomputable def smallerCylinderVolume (c : Cylinder) (sc : SmallerCylinder) : ℝ :=
  Real.pi * c.radius^2 * sc.height

/-- Volume of the cylindrical frustum -/
noncomputable def cylindricalFrustumVolume (c : Cylinder) (cf : CylindricalFrustum) : ℝ :=
  Real.pi * c.radius^2 * cf.height

/-- Theorem stating the height of the smaller cylinder -/
theorem smaller_cylinder_height 
  (c : Cylinder) 
  (sc : SmallerCylinder) 
  (cf : CylindricalFrustum) 
  (h_cylinder : c.height = 10 ∧ c.radius = 5) 
  (h_cut : sc.height + cf.height = c.height)
  (h_ratio : smallerCylinderVolume c sc / cylindricalFrustumVolume c cf = 2 / 3) : 
  sc.height = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cylinder_height_l738_73890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l738_73805

-- Define set A
def A : Set ℝ := {x | 1/2 < (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 2}

-- Define set B
def B : Set ℝ := {x | Real.log (x - 1/2) ≤ 0}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo (-1 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l738_73805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_one_of_ghi_odd_l738_73860

theorem min_odd_integers (a b c d e f g h i : ℤ) 
  (sum_first_three : a + b + c = 30)
  (sum_first_six : a + b + c + d + e + f = 48)
  (sum_all_nine : a + b + c + d + e + f + g + h + i = 69) :
  ∃ (odd_count : ℕ), odd_count ≥ 1 ∧ 
  (∃ (even_count : ℕ), even_count + odd_count = 9 ∧
  (Finset.filter (fun x => Odd x) {a, b, c, d, e, f, g, h, i}).card = odd_count) :=
by
  -- We'll use 1 as the minimum number of odd integers
  use 1
  constructor
  · -- Prove that 1 ≥ 1
    exact Nat.le_refl 1
  · -- Prove the existence of even_count
    use 8
    constructor
    · -- Prove that even_count + odd_count = 9
      rfl
    · -- Prove that the number of odd integers is equal to odd_count (1)
      -- This part is complex and requires more detailed proof
      sorry

-- Helper theorem to show that at least one of g, h, i must be odd
theorem one_of_ghi_odd (g h i : ℤ) (sum_ghi : g + h + i = 21) :
  Odd g ∨ Odd h ∨ Odd i :=
by
  -- The sum of three even numbers cannot be odd
  -- So at least one of g, h, i must be odd
  sorry

#check min_odd_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_integers_one_of_ghi_odd_l738_73860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l738_73831

/-- Represents the speed of bicycles in km/min -/
def bicycle_speed : ℝ → ℝ := sorry

/-- Represents the speed of the car in km/min -/
def car_speed : ℝ → ℝ := sorry

/-- The distance to the museum in km -/
def distance_to_museum : ℝ := 12

/-- The time difference between car and bicycle departure in minutes -/
def time_difference : ℝ := 20

theorem car_speed_calculation (x : ℝ) 
  (h1 : bicycle_speed x = x)
  (h2 : car_speed x = 2 * x)
  (h3 : distance_to_museum / (bicycle_speed x) = 
        distance_to_museum / (car_speed x) + time_difference) :
  car_speed x = 0.6 := by
  sorry

#check car_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l738_73831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_specific_triangle_l738_73896

/-- The line equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The triangle formed by a line and the x and y axes -/
structure AxisTriangle where
  line : Line

noncomputable def intersect_x_axis (l : Line) : Point :=
  { x := l.c / l.a, y := 0 }

noncomputable def intersect_y_axis (l : Line) : Point :=
  { x := 0, y := l.c / l.b }

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def perimeter (t : AxisTriangle) : ℝ :=
  let a := intersect_x_axis t.line
  let b := intersect_y_axis t.line
  let origin : Point := { x := 0, y := 0 }
  distance origin a + distance origin b + distance a b

theorem perimeter_of_specific_triangle :
  let l : Line := { a := 1/3, b := 1/4, c := 1 }
  let t : AxisTriangle := { line := l }
  perimeter t = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_specific_triangle_l738_73896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_l738_73819

-- Define the given parameters
noncomputable def train_length : ℝ := 100  -- meters
noncomputable def train_speed : ℝ := 72    -- km/hr
noncomputable def tunnel_length : ℝ := 2.3 -- km

-- Define the function to calculate the time taken
noncomputable def time_to_pass_tunnel (train_length : ℝ) (train_speed : ℝ) (tunnel_length : ℝ) : ℝ :=
  let speed_ms : ℝ := train_speed * 1000 / 3600  -- Convert km/hr to m/s
  let total_distance : ℝ := train_length + tunnel_length * 1000  -- Convert km to m
  (total_distance / speed_ms) / 60  -- Calculate time in minutes

-- Theorem statement
theorem train_tunnel_time :
  time_to_pass_tunnel train_length train_speed tunnel_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_l738_73819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l738_73886

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (5 + 4*x - x^2)

-- State the theorem about the range of the function
theorem f_range : 
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 0 ≤ y ∧ y ≤ 3 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l738_73886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_one_twentyseventh_l738_73841

/-- The vertices of the original tetrahedron -/
def original_vertices : List (Fin 4 → ℝ) := [
  ![1, 0, 0, 0],
  ![0, 1, 0, 0],
  ![0, 0, 1, 0],
  ![0, 0, 0, 1]
]

/-- The center of a face given by three vertices -/
noncomputable def face_center (v1 v2 v3 : Fin 4 → ℝ) : Fin 4 → ℝ :=
  fun i => (v1 i + v2 i + v3 i) / 3

/-- The centers of the faces of the original tetrahedron -/
noncomputable def face_centers : List (Fin 4 → ℝ) :=
  [face_center (original_vertices[0]!) (original_vertices[1]!) (original_vertices[3]!),
   face_center (original_vertices[0]!) (original_vertices[2]!) (original_vertices[3]!),
   face_center (original_vertices[1]!) (original_vertices[2]!) (original_vertices[3]!),
   face_center (original_vertices[0]!) (original_vertices[1]!) (original_vertices[2]!)]

/-- The volume of a tetrahedron given its four vertices -/
noncomputable def tetrahedron_volume (v1 v2 v3 v4 : Fin 4 → ℝ) : ℝ :=
  (1/6) * abs (Matrix.det ![v2 - v1, v3 - v1, v4 - v1, 0])

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio_is_one_twentyseventh :
  let original_volume := tetrahedron_volume (original_vertices[0]!) (original_vertices[1]!) (original_vertices[2]!) (original_vertices[3]!)
  let smaller_volume := tetrahedron_volume (face_centers[0]!) (face_centers[1]!) (face_centers[2]!) (face_centers[3]!)
  smaller_volume / original_volume = 1 / 27 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_one_twentyseventh_l738_73841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_angle_sine_relationship_l738_73836

-- Define the solution set for the inequality
def solution_set := {x : ℝ | 0 < x ∧ x < 1}

-- Define the condition for triangle ABC
def triangle_condition (A B : ℝ) := 0 < A ∧ 0 < B ∧ A + B < Real.pi

-- Statement for the inequality solution
theorem inequality_solution : 
  ∀ x : ℝ, x / (x - 1) < 0 ↔ x ∈ solution_set :=
sorry

-- Statement for the triangle angle-sine relationship
theorem angle_sine_relationship :
  ∀ A B : ℝ, triangle_condition A B → 
    (A > B ↔ Real.sin A > Real.sin B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_angle_sine_relationship_l738_73836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_is_zero_l738_73827

open Matrix Real

-- Define the matrix as noncomputable
noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![sin 1, sin 2, sin 3],
    ![sin 4, sin 5, sin 6],
    ![sin 7, sin 8, sin 9]]

-- State the theorem
theorem det_A_is_zero : det A = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_is_zero_l738_73827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_wins_when_k_gt_l_leva_wins_when_k_le_l_l738_73818

-- Define the lengths of segments
variable (k l : ℝ)

-- Define the division of segments
variable (k1 k2 k3 l1 l2 l3 : ℝ)

-- Define the conditions for Kolya's division
def kolya_division (k k1 k2 k3 : ℝ) : Prop :=
  k1 + k2 + k3 = k ∧ k1 ≥ 0 ∧ k2 ≥ 0 ∧ k3 ≥ 0

-- Define the conditions for Leva's division
def leva_division (l l1 l2 l3 : ℝ) : Prop :=
  l1 + l2 + l3 = l ∧ l1 ≥ 0 ∧ l2 ≥ 0 ∧ l3 ≥ 0

-- Define the condition for forming a triangle
def forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition for Leva's win
def leva_wins (k1 k2 k3 l1 l2 l3 : ℝ) : Prop :=
  ∃ (a1 b1 c1 a2 b2 c2 : ℝ), 
    (Set.toFinset {a1, b1, c1, a2, b2, c2} = Set.toFinset {k1, k2, k3, l1, l2, l3}) ∧
    forms_triangle a1 b1 c1 ∧ forms_triangle a2 b2 c2

-- Theorem for Kolya's win when k > l
theorem kolya_wins_when_k_gt_l (h : k > l) :
  ∃ (k1 k2 k3 : ℝ), kolya_division k k1 k2 k3 ∧
    ∀ (l1 l2 l3 : ℝ), leva_division l l1 l2 l3 → ¬(leva_wins k1 k2 k3 l1 l2 l3) := by
  sorry

-- Theorem for Leva's win when k ≤ l
theorem leva_wins_when_k_le_l (h : k ≤ l) :
  ∀ (k1 k2 k3 : ℝ), kolya_division k k1 k2 k3 →
    ∃ (l1 l2 l3 : ℝ), leva_division l l1 l2 l3 ∧ leva_wins k1 k2 k3 l1 l2 l3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_wins_when_k_gt_l_leva_wins_when_k_le_l_l738_73818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l738_73800

-- Define the line and circle equations
def line_eq (a x y : ℝ) : Prop := a * x + y + a + 1 = 0
def circle_eq (x y b : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + b = 0

-- State the theorem
theorem circle_line_intersection_range (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, line_eq a x y ∧ circle_eq x y b) →
  (∀ x y : ℝ, circle_eq x y b → (x - 1)^2 + (y - 1)^2 = 2 - b) →
  b < -6 ∧ b < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_l738_73800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_perpendicular_distance_problem_solution_l738_73801

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A line in a 2D plane represented by its y-intercept -/
structure Line where
  yIntercept : ℝ

/-- The perpendicular distance from a point to a line -/
def perpendicularDistance (p : ℝ × ℝ) (l : Line) : ℝ :=
  p.2 - l.yIntercept

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

theorem centroid_perpendicular_distance (t : Triangle) (l : Line) :
  perpendicularDistance (centroid t) l =
  (perpendicularDistance t.A l + perpendicularDistance t.B l + perpendicularDistance t.C l) / 3 := by
  sorry

/-- The specific triangle and line from the problem -/
def problemTriangle : Triangle :=
  { A := (0, 12)
    B := (0, 8)
    C := (0, 20) }

def problemLine : Line :=
  { yIntercept := 0 }

theorem problem_solution :
  perpendicularDistance (centroid problemTriangle) problemLine = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_perpendicular_distance_problem_solution_l738_73801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_calculation_l738_73873

noncomputable section

def gallon_to_ounce : ℝ := 128
def whole_milk_weight : ℝ := 8.6
def skim_milk_weight : ℝ := 8.4
def almond_milk_weight : ℝ := 8.3

def initial_whole_milk : ℝ := 3
def initial_skim_milk : ℝ := 2
def initial_almond_milk : ℝ := 1

def consumed_whole_milk : ℝ := 13
def consumed_skim_milk : ℝ := 20
def consumed_almond_milk : ℝ := 25

def remaining_whole_milk : ℝ := initial_whole_milk * gallon_to_ounce - consumed_whole_milk
def remaining_skim_milk : ℝ := initial_skim_milk * gallon_to_ounce - consumed_skim_milk
def remaining_almond_milk : ℝ := initial_almond_milk * gallon_to_ounce - consumed_almond_milk

def total_weight : ℝ :=
  (remaining_whole_milk / gallon_to_ounce * whole_milk_weight) +
  (remaining_skim_milk / gallon_to_ounce * skim_milk_weight) +
  (remaining_almond_milk / gallon_to_ounce * almond_milk_weight)

theorem milk_calculation :
  remaining_whole_milk = 371 ∧
  remaining_skim_milk = 236 ∧
  remaining_almond_milk = 103 ∧
  (total_weight ≥ 47.09 ∧ total_weight < 47.10) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_calculation_l738_73873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l738_73884

-- Define the point P
def P : ℝ × ℝ := (11, 0)

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the angle of inclination
noncomputable def inclination : ℝ := Real.pi/4

-- Define the function for the area of triangle PMN
noncomputable def triangleArea (m : ℝ) : ℝ := 2 * Real.sqrt (1 - m) * |m + 11|

-- State the theorem
theorem max_triangle_area :
  ∃ (maxArea : ℝ), maxArea = 22 ∧
  ∀ (m : ℝ), m < 1 → triangleArea m ≤ maxArea := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l738_73884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_of_given_ellipse_l738_73876

/-- The equation of an ellipse in the form x²/a² + y²/b² = 1, where a and b are positive real numbers -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The foci of an ellipse are two fixed points on its major axis -/
noncomputable def foci (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p = (Real.sqrt (e.a^2 - e.b^2), 0) ∨ p = (-Real.sqrt (e.a^2 - e.b^2), 0)}

/-- The given ellipse with equation x²/2 + y² = 1 -/
noncomputable def given_ellipse : Ellipse where
  a := Real.sqrt 2
  b := 1
  h_pos_a := by sorry
  h_pos_b := by simp

theorem foci_of_given_ellipse :
  foci given_ellipse = {(1, 0), (-1, 0)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foci_of_given_ellipse_l738_73876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_jo_max_sum_l738_73837

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_multiple_of_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def max_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_multiple_of_five (List.range n))

theorem difference_jo_max_sum :
  (jo_sum 60 : ℤ) - (max_sum 60 : ℤ) = 1650 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_jo_max_sum_l738_73837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_range_multiple_of_840_l738_73847

def is_multiple_of_840 (product : ℕ) : Prop :=
  ∃ k : ℕ, product = 840 * k

def consecutive_range (start finish : ℕ) : List ℕ :=
  List.range (finish - start + 1) |>.map (· + start)

def range_product (l : List ℕ) : ℕ :=
  l.prod

theorem smallest_range_multiple_of_840 :
  ∀ start finish : ℕ,
    start ≤ finish →
    is_multiple_of_840 (range_product (consecutive_range start finish)) →
    finish - start + 1 ≥ 3 →
    (start = 5 ∧ finish = 7) ∨ finish - start + 1 > 3 :=
by
  sorry

#eval range_product (consecutive_range 5 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_range_multiple_of_840_l738_73847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l738_73815

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (3 + abs x) - 4 / (1 + x^2)

-- State the theorem
theorem range_of_inequality (x : ℝ) :
  f x - f (3*x + 1) < 0 ↔ x < -1/2 ∨ x > -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_inequality_l738_73815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vishal_investment_percentage_l738_73877

noncomputable section

-- Define the investments
def raghu_investment : ℝ := 2500

def trishul_investment : ℝ := raghu_investment * 0.9

def total_investment : ℝ := 7225

def vishal_investment : ℝ := total_investment - trishul_investment - raghu_investment

-- Define the percentage difference
def percentage_difference : ℝ := (vishal_investment - trishul_investment) / trishul_investment * 100

-- Theorem statement
theorem vishal_investment_percentage :
  percentage_difference = 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vishal_investment_percentage_l738_73877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_seven_sides_l738_73879

/-- A regular polygon is a polygon where all sides have equal length and all interior angles are equal. -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  r : ℝ  -- radius of circumscribed circle
  [positive_r : Fact (r > 0)]

/-- The chord length between two vertices of a regular polygon separated by k sides. -/
noncomputable def chord_length (p : RegularPolygon) (k : ℕ) : ℝ :=
  2 * p.r * Real.sin (k * Real.pi / p.n)

/-- Theorem: If a regular polygon with consecutive vertices A, B, C, D satisfies
    the equation 1/AB = 1/AC + 1/AD, then the polygon has 7 sides. -/
theorem regular_polygon_seven_sides (p : RegularPolygon) :
  (1 / chord_length p 1 = 1 / chord_length p 2 + 1 / chord_length p 3) →
  p.n = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_seven_sides_l738_73879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_is_half_wheat_field_area_fraction_l738_73870

/-- Represents a kite-shaped field -/
structure KiteField where
  long_side : ℝ
  short_side : ℝ
  angle_unequal : ℝ
  angle_equal : ℝ

/-- The kite field configuration -/
def wheat_field : KiteField :=
  { long_side := 120
  , short_side := 80
  , angle_unequal := 120
  , angle_equal := 60 }

/-- The area of the region closest to the longest side -/
noncomputable def area_closest_to_longest_side (k : KiteField) : ℝ :=
  1/2 * k.long_side * k.short_side * Real.sin (k.angle_equal * Real.pi / 180)

/-- The total area of the kite-shaped field -/
noncomputable def total_area (k : KiteField) : ℝ :=
  2 * area_closest_to_longest_side k

/-- Theorem stating that the area closest to the longest side is half of the total area -/
theorem area_fraction_is_half (k : KiteField) :
  area_closest_to_longest_side k / total_area k = 1/2 := by
  sorry

/-- The main theorem applied to the specific wheat field -/
theorem wheat_field_area_fraction :
  area_closest_to_longest_side wheat_field / total_area wheat_field = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_fraction_is_half_wheat_field_area_fraction_l738_73870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_two_l738_73835

open Real Set

theorem inequality_holds_iff_a_leq_two (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Icc 0 (π / 2), f x = x + sin x) →
  (∀ x ∈ Icc 0 (π / 2), f x ≥ a * x * cos x) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_leq_two_l738_73835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_proof_l738_73848

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (a^2 + b^2 ≥ 8) ∧ ((2:ℝ)^a + (2:ℝ)^b ≥ 8) ∧ (1/a + 4/b ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_proof_l738_73848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l738_73832

def a : ℝ × ℝ := (7, 1)
def b : ℝ × ℝ := (1, 3)

theorem vector_properties :
  let cos_angle := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  let lambda := (3 : ℝ) / 7
  cos_angle = Real.sqrt 5 / 5 ∧
  (a.1 + 2*b.1) * (lambda * a.1 - b.1) + (a.2 + 2*b.2) * (lambda * a.2 - b.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l738_73832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l738_73852

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: f is increasing for all a
theorem f_increasing (a : ℝ) : 
  ∀ x y : ℝ, x < y → f a x < f a y := by sorry

-- Theorem 2: f is odd if and only if a = 1
theorem f_odd_iff_a_eq_one : 
  ∃! a : ℝ, ∀ x : ℝ, f a (-x) = -(f a x) ∧ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l738_73852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_implies_double_angle_sum_zero_l738_73862

theorem cosine_sine_sum_implies_double_angle_sum_zero
  (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_implies_double_angle_sum_zero_l738_73862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_parameter_l738_73812

theorem parabola_focal_parameter (p x₀ y₀ : ℝ) : 
  p > 0 →
  y₀^2 = 2 * p * x₀ →
  (x₀ - p / 2)^2 + y₀^2 = 100 →
  y₀^2 = 36 →
  p = 2 ∨ p = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focal_parameter_l738_73812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_143_to_base6_has_three_consecutive_digits_l738_73802

/-- Converts a decimal number to its base 6 representation -/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Checks if a list of digits are consecutive -/
def areConsecutive (digits : List ℕ) : Bool :=
  match digits with
  | [] => true
  | [_] => true
  | x :: y :: rest => (y - x = 1) && areConsecutive (y :: rest)

/-- The main theorem to prove -/
theorem decimal_143_to_base6_has_three_consecutive_digits :
  let base6Repr := toBase6 143
  base6Repr.length = 3 ∧ areConsecutive base6Repr := by
  sorry

#eval toBase6 143  -- To check the result
#eval areConsecutive (toBase6 143)  -- To check if the digits are consecutive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_143_to_base6_has_three_consecutive_digits_l738_73802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_for_horizontal_asymptote_l738_73878

/-- The denominator polynomial -/
def p (x : ℝ) : ℝ := 2*x^5 - 3*x^4 + 2*x^3 - 7*x^2 + x + 1

/-- A rational function with numerator q and denominator p -/
noncomputable def f (q : ℝ → ℝ) (x : ℝ) : ℝ := q x / p x

/-- Definition of having a horizontal asymptote -/
def has_horizontal_asymptote (f : ℝ → ℝ) : Prop :=
  ∃ L : ℝ, ∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - L| < ε

/-- The degree of a polynomial -/
noncomputable def degree (q : ℝ → ℝ) : ℕ := sorry

/-- The main theorem -/
theorem largest_degree_for_horizontal_asymptote :
  (∃ q : ℝ → ℝ, degree q = 5 ∧ has_horizontal_asymptote (f q)) ∧
  (∀ q : ℝ → ℝ, degree q > 5 → ¬ has_horizontal_asymptote (f q)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_for_horizontal_asymptote_l738_73878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_non_real_root_l738_73844

theorem polynomial_has_non_real_root
  (a b c d : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) :
  ∃ z : ℂ, z^6 + a*z^3 + b*z^2 + c*z + d = 0 ∧ ¬(∃ x : ℝ, (z : ℂ) = x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_non_real_root_l738_73844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pairs_theorem_l738_73865

theorem prime_pairs_theorem (n p : ℕ) (h_prime : Nat.Prime p) (h_bound : n ≤ 2 * p) 
  (h_divisible : (p - 1)^n + 1 ∣ n^(p - 1)) :
  ((n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pairs_theorem_l738_73865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l738_73885

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  AB : ℝ
  BC : ℝ
  cosC : ℝ

/-- The triangle ABC with the given conditions -/
noncomputable def triangle_ABC : Triangle where
  AB := Real.sqrt 2
  BC := 1
  cosC := 3/4

/-- Theorem stating the properties of triangle ABC -/
theorem triangle_ABC_properties (T : Triangle) (h : T = triangle_ABC) :
  let sinA := Real.sqrt 14 / 8
  let AC := 2
  sinA = Real.sqrt (1 - T.cosC^2) * T.BC / T.AB ∧ 
  AC^2 = T.AB^2 + T.BC^2 - 2 * T.BC * AC * T.cosC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l738_73885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_domain_of_inverse_l738_73822

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 2 else 2^(-x)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≤ -2 ∨ y > 1} := by
  sorry

-- Theorem stating that the domain of the inverse function is the range of f
theorem domain_of_inverse :
  Set.range f = {y : ℝ | y ≤ -2 ∨ y > 1} := by
  exact range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_domain_of_inverse_l738_73822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_minus_one_pow_2015_l738_73891

theorem remainder_x_minus_one_pow_2015 (x : Polynomial ℤ) :
  (X - 1)^2015 ≡ -1 [ZMOD (X^2 - X + 1)] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_x_minus_one_pow_2015_l738_73891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_relation_l738_73830

/-- Given functions f and g, where f(x) = sin(ωx + φ) and g(x) = 2cos(ωx + φ) - 1,
    if f is symmetric about x = π/4, then g(π/4) = -1 -/
theorem symmetric_sine_cosine_relation (ω φ : ℝ) :
  (∀ x : ℝ, Real.sin (ω * (π/4 - x) + φ) = Real.sin (ω * (π/4 + x) + φ)) →
  2 * Real.cos (ω * π/4 + φ) - 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_cosine_relation_l738_73830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pricing_l738_73851

/-- A product with given wholesale and retail prices, and a linear demand function. -/
structure Product where
  wholesale_price : ℚ
  initial_retail_price : ℚ
  initial_sales_volume : ℚ
  price_sensitivity : ℚ

/-- Calculate the profit for a given price increase. -/
def profit (p : Product) (price_increase : ℚ) : ℚ :=
  (p.initial_retail_price + price_increase - p.wholesale_price) *
  (p.initial_sales_volume - p.price_sensitivity * price_increase)

/-- Find the price increase that maximizes profit. -/
def optimal_price_increase (p : Product) : ℚ :=
  (p.initial_retail_price - p.wholesale_price + p.initial_sales_volume / p.price_sensitivity) / 2

/-- The main theorem stating the optimal retail price and maximum profit. -/
theorem optimal_pricing (p : Product)
  (h1 : p.wholesale_price = 40)
  (h2 : p.initial_retail_price = 50)
  (h3 : p.initial_sales_volume = 50)
  (h4 : p.price_sensitivity = 1) :
  optimal_price_increase p = 20 ∧
  profit p (optimal_price_increase p) = 900 := by
  sorry

#eval optimal_price_increase { wholesale_price := 40, initial_retail_price := 50, initial_sales_volume := 50, price_sensitivity := 1 }
#eval profit { wholesale_price := 40, initial_retail_price := 50, initial_sales_volume := 50, price_sensitivity := 1 } 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pricing_l738_73851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_square_centers_l738_73838

/-- Predicate stating that point P is the center of a square
    constructed on the side AB of a triangle. -/
def is_center_of_square
  (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
sorry

/-- Predicate stating that three points form an acute-angled triangle. -/
def is_acute_angled_triangle
  (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop :=
sorry

/-- Given three points X, Y, and Z in a plane, this theorem states that
    a unique triangle ABC can be constructed with X, Y, and Z as the centers
    of squares on its sides if and only if triangle XYZ is acute-angled. -/
theorem triangle_construction_from_square_centers
  (X Y Z : EuclideanSpace ℝ (Fin 2)) :
  (∃! (A B C : EuclideanSpace ℝ (Fin 2)),
    (is_center_of_square X B C) ∧
    (is_center_of_square Y C A) ∧
    (is_center_of_square Z A B)) ↔
  is_acute_angled_triangle X Y Z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_from_square_centers_l738_73838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtake_time_l738_73849

/-- Represents the time (in hours) it takes for a faster car to overtake a slower car -/
noncomputable def overtakeTime (v1 v2 distance : ℝ) : ℝ :=
  distance / (v2 - v1)

/-- Proves that the time for the black car to overtake the red car is 1 hour -/
theorem black_car_overtake_time :
  let red_speed : ℝ := 30
  let black_speed : ℝ := 50
  let initial_distance : ℝ := 20
  overtakeTime red_speed black_speed initial_distance = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_car_overtake_time_l738_73849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_distance_flash_distance_positive_l738_73825

-- Define the variables and conditions
variable (x y : ℝ)
variable (h_x : x > 1)

-- Define Ace's speed (can be any positive real number)
variable (v : ℝ)
variable (h_v : v > 0)

-- Define Flash's speed in terms of Ace's
noncomputable def flash_speed (x v : ℝ) : ℝ := x^2 * v

-- Define the head start distance
noncomputable def head_start (y : ℝ) : ℝ := y^2

-- Define the function to calculate the distance Flash runs
noncomputable def flash_distance (x y : ℝ) : ℝ := (x^2 * y^2) / (x^2 - 1)

-- Theorem statement
theorem flash_catch_distance (x y : ℝ) (h_x : x > 1) : 
  flash_distance x y = (x^2 * y^2) / (x^2 - 1) := by
  -- Unfold the definition of flash_distance
  unfold flash_distance
  -- The equality is trivial by definition
  rfl

-- Additional theorem to show that the distance is positive
theorem flash_distance_positive (x y : ℝ) (h_x : x > 1) (h_y : y ≠ 0) :
  flash_distance x y > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_distance_flash_distance_positive_l738_73825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l738_73874

-- Define the function
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + Real.log x

-- Define the derivative of the function
noncomputable def f_derivative (k : ℝ) (x : ℝ) : ℝ := 2 * k * x + 1 / x

-- State the theorem
theorem tangent_line_parallel (k : ℝ) :
  (f_derivative k 1 = 2) → k = 1/2 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l738_73874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l738_73859

/-- α(n) is the number of ways to express n as the sum of several 1s and several 2s, 
    with different orders considered different -/
def α : ℕ → ℕ := sorry

/-- β(n) is the number of ways to express n as the sum of several integers greater than 1, 
    with different orders considered different -/
def β : ℕ → ℕ := sorry

/-- For every positive integer n, α(n) = β(n + 2) -/
theorem alpha_beta_equality (n : ℕ) (hn : n > 0) : α n = β (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_equality_l738_73859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l738_73807

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def IsPerp (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line y = mx + c is m -/
def slope_of_line (m c : ℝ) : ℝ := m

/-- The slope of the line ax + by = c is -a/b -/
noncomputable def slope_of_general_line (a b c : ℝ) : ℝ := -a / b

theorem perpendicular_lines (b : ℝ) : 
  IsPerp (slope_of_line 3 7) (slope_of_general_line b 4 8) ↔ b = 4/3 := by
  sorry

#check perpendicular_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l738_73807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_specific_altitude_ratio_l738_73826

/-- Given a triangle with integer side lengths and altitudes in the ratio 3:4:5,
    prove that one of its side lengths must be 12. -/
theorem triangle_with_specific_altitude_ratio (a b c h₁ h₂ h₃ : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 →
  h₁ * 4 = h₂ * 3 →
  h₁ * 5 = h₃ * 3 →
  a * h₁ = b * h₂ →
  a * h₁ = c * h₃ →
  (∃ k : ℕ, k > 0 ∧ a = 20 * k ∧ b = 15 * k ∧ c = 12 * k) →
  12 ∈ ({a, b, c} : Set ℕ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_specific_altitude_ratio_l738_73826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_day_price_is_correct_l738_73864

/-- Represents the lemonade stand problem --/
structure LemonadeStand where
  totalProfit : ℚ
  hotDayPriceIncrease : ℚ
  costPerCup : ℚ
  cupsSoldPerDay : ℕ
  hotDays : ℕ
  totalDays : ℕ

/-- Calculates the price of a cup on a hot day --/
noncomputable def hotDayPrice (stand : LemonadeStand) : ℚ :=
  let regularPrice := (stand.totalProfit + stand.costPerCup * stand.cupsSoldPerDay * stand.totalDays) /
    (stand.cupsSoldPerDay * (stand.totalDays + stand.hotDayPriceIncrease * stand.hotDays))
  regularPrice * (1 + stand.hotDayPriceIncrease)

/-- Theorem stating that the hot day price is $2.50 given the problem conditions --/
theorem hot_day_price_is_correct (stand : LemonadeStand)
  (h1 : stand.totalProfit = 450)
  (h2 : stand.hotDayPriceIncrease = 1/4)
  (h3 : stand.costPerCup = 3/4)
  (h4 : stand.cupsSoldPerDay = 32)
  (h5 : stand.hotDays = 3)
  (h6 : stand.totalDays = 10) :
  hotDayPrice stand = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_day_price_is_correct_l738_73864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l738_73861

-- Define the circles
noncomputable def circle1_center : ℝ × ℝ := (0, 0)
noncomputable def circle1_radius : ℝ := 3

noncomputable def circle2_center : ℝ × ℝ := (18, 0)
noncomputable def circle2_radius : ℝ := 8

-- Define the tangent point on the x-axis
noncomputable def tangent_point : ℝ := 54 / 11

-- Theorem statement
theorem tangent_line_intersection :
  let distance := circle2_center.1 - circle1_center.1
  let slope := (circle2_radius - circle1_radius) / distance
  tangent_point = (distance * circle1_radius) / (circle2_radius - circle1_radius) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l738_73861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_c_relation_l738_73868

def a : ℕ → ℚ
  | 0 => 1/2
  | (n+1) => 2 * a n / (1 + (a n)^2)

def c : ℕ → ℚ
  | 0 => 4
  | (n+1) => c n ^ 2 - 2 * c n + 2

def product_c : ℕ → ℚ 
  | 0 => 1
  | (n+1) => product_c n * c n

theorem a_c_relation (n : ℕ) (h : n ≥ 1) : 
  a n = 2 * product_c (n-1) / c n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_c_relation_l738_73868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_L_l738_73823

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.y^2 - p.x^2 / 3 = 1

/-- Represents a line -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is on the line -/
def isOnLine (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem about the hyperbola and line L -/
theorem hyperbola_and_line_L :
  ∀ (h : Hyperbola) (l : Line) (a m n : Point),
    h.c = 2 →
    h.a / h.b = Real.sqrt 3 / 3 →
    a.x = 1 ∧ a.y = 1/2 →
    isOnHyperbola h a →
    isOnLine l a →
    isOnLine l m →
    isOnLine l n →
    isOnHyperbola h m →
    isOnHyperbola h n →
    a.x = (m.x + n.x) / 2 ∧ a.y = (m.y + n.y) / 2 →
    (h.a = 1 ∧ h.b = Real.sqrt 3) ∧ (l.a = 4 ∧ l.b = -6 ∧ l.c = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_L_l738_73823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_nonpositive_l738_73813

-- Define the function f as noncomputable due to its dependency on Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x * (x - a))

-- State the theorem
theorem f_monotone_iff_a_nonpositive :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ < f a x₂) ↔ a ≤ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_iff_a_nonpositive_l738_73813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_1_theorem_2_l738_73845

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log x - x^2 + 2

-- Define the derivative of f
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m / x - 2 * x

-- Theorem 1: When m = 2, for all x > 0, f(x) - f'(x) ≤ 4x - 3
theorem theorem_1 (x : ℝ) (hx : x > 0) : 
  f 2 x - f_deriv 2 x ≤ 4 * x - 3 := by sorry

-- Theorem 2: When 2 ≤ m ≤ 8, for all x ≥ 1, f(x) - f'(x) ≤ 4x - 3
theorem theorem_2 (m : ℝ) (hm : 2 ≤ m ∧ m ≤ 8) (x : ℝ) (hx : x ≥ 1) :
  f m x - f_deriv m x ≤ 4 * x - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem_1_theorem_2_l738_73845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l738_73899

/-- The function f(x) = cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

/-- Theorem stating that given the conditions, ω = 2/3 -/
theorem omega_value (ω : ℝ) 
  (h1 : ω > 0)
  (h2 : ∀ x, f ω (3 * Real.pi / 4 - x) = f ω (3 * Real.pi / 4 + x))
  (h3 : StrictMonoOn (f ω) (Set.Ioo 0 (2 * Real.pi / 3))) :
  ω = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l738_73899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_specific_circle_l738_73858

/-- The length of the shortest chord passing through a given point on a circle. -/
noncomputable def shortestChordLength (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - (((point.1 - center.1)^2 + (point.2 - center.2)^2) : ℝ))

/-- Theorem: The length of the shortest chord passing through the point (3,1) 
    on the circle (x-2)^2+(y-2)^2=4 is 2√2. -/
theorem shortest_chord_length_specific_circle : 
  shortestChordLength (2, 2) 2 (3, 1) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_specific_circle_l738_73858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_power_sum_l738_73898

theorem two_power_sum (a : ℝ) (h : (4 : ℝ)^a = 3) : 2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_power_sum_l738_73898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_two_l738_73853

theorem reciprocal_of_negative_two :
  (1 : ℝ) / (-2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_two_l738_73853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l738_73850

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.cos (x + Real.pi / 2)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.cos (Real.cos x))

-- Theorem statement
theorem trigonometric_properties :
  (∀ x, f (-x) = -f x) ∧ (∀ x, g x ∈ Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_properties_l738_73850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l738_73820

theorem sum_of_solutions : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 - 6*x + 8)^(x^2 - 8*x + 15) = 1) ∧ 
  (∀ x : ℝ, (x^2 - 6*x + 8)^(x^2 - 8*x + 15) = 1 → x ∈ S) ∧
  (S.sum id) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l738_73820
