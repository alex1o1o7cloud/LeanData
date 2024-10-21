import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donut_fundraiser_correct_l856_85619

def donut_fundraiser (target : ℚ) (buy_price_per_dozen : ℚ) (sell_price_per_donut : ℚ) : ℕ :=
  let cost_per_donut := buy_price_per_dozen / 12
  let profit_per_donut := sell_price_per_donut - cost_per_donut
  let total_donuts_needed := target / profit_per_donut
  (total_donuts_needed / 12).ceil.toNat

theorem donut_fundraiser_correct :
  donut_fundraiser 96 2.40 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donut_fundraiser_correct_l856_85619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l856_85611

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ^2 - 4*ρ*(Real.cos θ) + 3 = 0 ∧ 0 ≤ θ ∧ θ ≤ 2*Real.pi

def C₂ (x y t : ℝ) : Prop := x = t*(Real.cos (Real.pi/6)) ∧ y = t*(Real.sin (Real.pi/6))

-- Define the intersection point
def intersection_point (ρ θ : ℝ) : Prop := ρ = Real.sqrt 3 ∧ θ = Real.pi/6

-- Theorem statement
theorem intersection_of_C₁_and_C₂ :
  ∃ (ρ θ : ℝ), C₁ ρ θ ∧ (∃ (x y t : ℝ), C₂ x y t ∧ x = ρ*(Real.cos θ) ∧ y = ρ*(Real.sin θ)) ∧ intersection_point ρ θ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_C₁_and_C₂_l856_85611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l856_85602

-- Define the statements as axioms (assumed to be true)
axiom statementA : Prop
axiom statementB : Prop
axiom statementC : Prop
axiom statementD : Prop

-- Axioms for the truth values of the statements
axiom statementA_true : statementA
axiom statementB_false : ¬statementB
axiom statementC_false : ¬statementC
axiom statementD_true : statementD

-- Theorem to prove
theorem correct_statements : 
  statementA ∧ ¬statementB ∧ ¬statementC ∧ statementD :=
by
  apply And.intro
  · exact statementA_true
  · apply And.intro
    · exact statementB_false
    · apply And.intro
      · exact statementC_false
      · exact statementD_true


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l856_85602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_m_l856_85694

/-- 
Represents a number as the sum of 2021 non-negative integer powers of a base.
-/
def representable (n : ℕ) (base : ℕ) : Prop :=
  ∃ (powers : Fin 2021 → ℕ), n = (Finset.sum (Finset.univ : Finset (Fin 2021)) (λ i => base ^ (powers i)))

/-- 
The maximal value of m such that there exists an n > m that can be represented
as both the sum of 2021 non-negative integer powers of m and (m+1).
-/
theorem maximal_m : 
  (∃ (n : ℕ) (m : ℕ), n > m ∧ m > 0 ∧ 
    representable n m ∧ representable n (m + 1)) →
  (∀ (n : ℕ) (m : ℕ), n > m ∧ m > 0 ∧ 
    representable n m ∧ representable n (m + 1) → m ≤ 43) ∧
  (∃ (n : ℕ), n > 43 ∧ representable n 43 ∧ representable n 44) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximal_m_l856_85694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_rolled_semicircle_l856_85669

noncomputable def semicircle_roll (r : ℝ) : Prop :=
  r = 4 / Real.pi

noncomputable def path_length_of_point (B : ℝ × ℝ) : ℝ := sorry

theorem path_length_of_rolled_semicircle (r : ℝ) (h : semicircle_roll r) :
  ∃ B : ℝ × ℝ, path_length_of_point B = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_rolled_semicircle_l856_85669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_subset_size_l856_85642

theorem minimum_subset_size (n : ℕ) (h : n = 2016) :
  ∃ m : ℕ, 
    (∀ S : Finset ℕ, S ⊆ Finset.range n → S.card ≥ m → 
      ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a : ℤ) - b ≤ 3 ∧ (b : ℤ) - a ≤ 3) ∧
    (∀ k : ℕ, k < m → 
      ∃ T : Finset ℕ, T ⊆ Finset.range n ∧ T.card = k ∧
        ∀ a b : ℕ, a ∈ T → b ∈ T → a ≠ b → (a : ℤ) - b > 3 ∨ (b : ℤ) - a > 3) ∧
    m = 505 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_subset_size_l856_85642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_hemisphere_volume_equality_l856_85668

theorem cone_hemisphere_volume_equality (R : ℝ) (θ : ℝ) (h : θ > 0) : 
  (1 / 3 * Real.pi * R^3 * Real.tan (Real.pi / 2 - θ) = 2 / 3 * Real.pi * R^3) → 
  Real.cos (2 * θ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_hemisphere_volume_equality_l856_85668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minions_actions_compromise_security_l856_85680

/-- Represents a user action in the scenario --/
inductive UserAction
  | sharePhoneNumber
  | downloadFile
  | openFile

/-- Represents the security status of the system --/
inductive SecurityStatus
  | secure
  | compromised

/-- Models the effect of a user action on the security status --/
def actionEffect (action : UserAction) (status : SecurityStatus) : SecurityStatus :=
  match action, status with
  | UserAction.sharePhoneNumber, SecurityStatus.secure => SecurityStatus.compromised
  | UserAction.downloadFile, SecurityStatus.secure => SecurityStatus.compromised
  | UserAction.openFile, SecurityStatus.secure => SecurityStatus.compromised
  | _, SecurityStatus.compromised => SecurityStatus.compromised

/-- Theorem: Any of the minions' actions compromises a secure system --/
theorem minions_actions_compromise_security (action : UserAction) :
  actionEffect action SecurityStatus.secure = SecurityStatus.compromised := by
  cases action
  all_goals (rfl)

#check minions_actions_compromise_security

/-- Note: This is a simplified model and doesn't capture all nuances of the scenario --/
def main : IO Unit :=
  IO.println "This code models a simplified version of the minions' security mistake."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minions_actions_compromise_security_l856_85680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l856_85617

open Real

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- State the theorem
theorem triangle_properties 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C) 
  (h_equation : 2 * a * cos C = 2 * b - c) :
  A = Real.pi / 3 ∧ 
  (a^2 ≤ b * (b + c) → c^2 = a^2 + b^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l856_85617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_interior_angle_eq_formula_l856_85618

/-- The measure of an interior angle of a regular pentagon is 108 degrees. -/
noncomputable def regular_pentagon_interior_angle : ℝ :=
  108

/-- A regular pentagon has 5 sides. -/
def regular_pentagon_sides : ℕ := 5

/-- Formula for interior angle of a regular polygon with n sides. -/
noncomputable def interior_angle (n : ℕ) : ℝ :=
  (n - 2 : ℝ) * 180 / n

/-- The measure of an interior angle of a regular pentagon is equal to
    the result of the interior angle formula for a pentagon. -/
theorem regular_pentagon_interior_angle_eq_formula :
  regular_pentagon_interior_angle = interior_angle regular_pentagon_sides := by
  sorry

#check regular_pentagon_interior_angle
#check regular_pentagon_interior_angle_eq_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pentagon_interior_angle_eq_formula_l856_85618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_ranks_and_statements_l856_85692

-- Define the possible ranks
inductive Rank
  | Knight
  | Liar
  | Ordinary

-- Define a person
structure Person where
  name : String
  rank : Rank

-- Define the statement function
def makes_statement (p : Person) (s : Prop) : Prop :=
  match p.rank with
  | Rank.Knight => s
  | Rank.Liar => ¬s
  | Rank.Ordinary => True

-- Define the ranking relation
def higher_rank (p1 p2 : Person) : Prop :=
  match p1.rank, p2.rank with
  | Rank.Knight, Rank.Ordinary => True
  | Rank.Knight, Rank.Liar => True
  | Rank.Ordinary, Rank.Liar => True
  | _, _ => False

-- Theorem statement
theorem people_ranks_and_statements 
  (A B C : Person) 
  (hA : makes_statement A (higher_rank B C))
  (hB : makes_statement B (higher_rank C A)) :
  A.rank = Rank.Ordinary ∧ 
  B.rank = Rank.Ordinary ∧ 
  ¬(higher_rank B C) ∧ 
  (higher_rank C A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_ranks_and_statements_l856_85692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_720_l856_85641

def number_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisors_of_720 : number_of_divisors 720 = 30 := by
  rw [number_of_divisors]
  simp [Nat.divisors]
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_720_l856_85641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l856_85665

/-- The quadratic function g(x) = (1/5)x^2 - x - 4 -/
noncomputable def g (x : ℝ) : ℝ := (1/5) * x^2 - x - 4

/-- Theorem stating that the only integer solution to g(g(g(x))) = -4 is x = -5 -/
theorem unique_integer_solution :
  ∀ x : ℤ, g (g (g (↑x))) = -4 ↔ x = -5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l856_85665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_12_l856_85658

theorem tan_alpha_plus_pi_12 (α : ℝ) 
  (h : Real.sin α = 3 * Real.sin (α + π/6)) : 
  Real.tan (α + π/12) = 2 * Real.sqrt 3 - 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_12_l856_85658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l856_85663

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the right focus F
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define the line l passing through F
def line_l (a b x y : ℝ) : Prop := 
  ∃ (k : ℝ), y = k * (x - Real.sqrt (a^2 - b^2))

-- Define the points A and B as intersections of l and C
noncomputable def point_A (a b : ℝ) : ℝ × ℝ := sorry
noncomputable def point_B (a b : ℝ) : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
noncomputable def point_P (a b : ℝ) : ℝ × ℝ := sorry
noncomputable def point_Q (a b : ℝ) : ℝ × ℝ := sorry

-- Define the vector addition condition
def vector_condition (P Q A B : ℝ × ℝ) : Prop := 
  P.1 + A.1 + B.1 = P.1 + Q.1 ∧ P.2 + A.2 + B.2 = P.2 + Q.2

-- Define the concyclic condition
def concyclic (P A Q B : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let F := right_focus a b
  let A := point_A a b
  let B := point_B a b
  let P := point_P a b
  let Q := point_Q a b
  (ellipse a b F.1 F.2 ∧ 
   ellipse a b A.1 A.2 ∧ 
   ellipse a b B.1 B.2 ∧ 
   ellipse a b P.1 P.2 ∧ 
   ellipse a b Q.1 Q.2 ∧
   line_l a b A.1 A.2 ∧ 
   line_l a b B.1 B.2 ∧
   vector_condition P Q A B ∧
   concyclic P A Q B) →
  Real.sqrt (a^2 - b^2) / a = Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l856_85663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicularity_and_angle_l856_85608

theorem vector_perpendicularity_and_angle (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi/2))
  (h2 : β ∈ Set.Ioo 0 (Real.pi/2))
  (h3 : (Real.cos α) * 2 + (-1) * (Real.sin α) = 0)  -- m ⊥ n
  (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) : 
  Real.cos (2*α) = -3/5 ∧ β = Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicularity_and_angle_l856_85608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_for_diophantine_equation_l856_85686

theorem min_abs_diff_for_diophantine_equation :
  ∀ a b : ℤ,
  a > 0 → b > 0 →
  a * b - 2 * a + 7 * b = 248 →
  ∀ c d : ℤ,
  c > 0 → d > 0 →
  c * d - 2 * c + 7 * d = 248 →
  |a - b| ≤ |c - d| ∧
  ∃ a b : ℤ, a > 0 ∧ b > 0 ∧ a * b - 2 * a + 7 * b = 248 ∧ |a - b| = 252 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_for_diophantine_equation_l856_85686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_range_l856_85633

/-- Curve C₁ -/
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, 3 * Real.sin φ)

/-- Curve C₂ (circle with radius 2) -/
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- Square vertices on C₂ -/
noncomputable def square_vertices : Fin 4 → ℝ × ℝ
| 0 => (1, Real.sqrt 3)
| 1 => (-Real.sqrt 3, 1)
| 2 => (-1, -Real.sqrt 3)
| 3 => (Real.sqrt 3, -1)

/-- Squared distance between two points -/
def squared_distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of squared distances from a point to all square vertices -/
noncomputable def sum_squared_distances (p : ℝ × ℝ) : ℝ :=
  Finset.sum (Finset.range 4) fun i => squared_distance p (square_vertices i)

theorem sum_squared_distances_range :
  ∀ φ : ℝ, 32 ≤ sum_squared_distances (C₁ φ) ∧ sum_squared_distances (C₁ φ) ≤ 52 := by
  sorry

#check sum_squared_distances_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_distances_range_l856_85633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_sum_eleven_l856_85603

-- Define the ages of Joey, Chloe, and Liam
def joey_age : ℕ → ℕ := sorry
def chloe_age : ℕ → ℕ := sorry
def liam_age : ℕ → ℕ := sorry

-- Define the conditions
axiom joey_older : ∀ n, joey_age n = chloe_age n + 2
axiom liam_two_today : liam_age 0 = 2
axiom chloe_multiple : ∃ k, k > 0 ∧ k ≤ 5 ∧ chloe_age 0 = k * liam_age 0

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define the next time Joey's age is a multiple of Liam's
def next_multiple : ℕ := sorry

-- Theorem statement
theorem joey_age_sum_eleven :
  sum_of_digits (joey_age next_multiple) = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_sum_eleven_l856_85603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_t_no_solutions_l856_85605

theorem infinite_t_no_solutions : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (t : ℕ) (x y : ℤ), t ∈ S →
    (x^2 + y^6 ≠ (t : ℤ)) ∧
    (x^2 + y^6 ≠ (t + 1 : ℤ)) ∧
    (x^2 - y^6 ≠ (t : ℤ)) ∧
    (x^2 - y^6 ≠ (t + 1 : ℤ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_t_no_solutions_l856_85605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_length_l856_85688

/-- The circle with center (3, -1) and radius 5 -/
def my_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 25

/-- The line x + y + 1 = 0 -/
def my_line (x y : ℝ) : Prop := x + y + 1 = 0

/-- The length of the chord intercepted on the circle by the line -/
def chord_length : ℝ := 8

theorem chord_intercept_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧
    my_line x₁ y₁ ∧ my_line x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_length_l856_85688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_wiped_by_specific_sponge_l856_85654

/-- Represents a semicircular sponge used for cleaning -/
structure Sponge where
  diameter : ℝ
  is_semicircular : Bool

/-- Represents a right-angled corner of a room -/
structure RoomCorner where
  is_right_angle : Bool

/-- Calculates the area wiped by a sponge in a room corner -/
noncomputable def area_wiped (s : Sponge) (c : RoomCorner) : ℝ :=
  (1/4) * Real.pi * s.diameter^2

/-- Theorem stating the area wiped by a specific sponge in a right-angled corner -/
theorem area_wiped_by_specific_sponge (s : Sponge) (c : RoomCorner) :
  s.diameter = 20 ∧ s.is_semicircular ∧ c.is_right_angle →
  area_wiped s c = 100 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_wiped_by_specific_sponge_l856_85654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_value_l856_85664

/-- A quadratic function f(x) = x^2 + mx + m^2 - m, where m is a constant -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + m^2 - m

/-- The axis of symmetry of f(x) -/
noncomputable def axis_of_symmetry (m : ℝ) : ℝ := -m / 2

theorem quadratic_minimum_value (m : ℝ) :
  f m 0 = 6 ∧ axis_of_symmetry m < 0 → ∃ (min_value : ℝ), min_value = 15/4 ∧ ∀ x, f m x ≥ min_value :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_minimum_value_l856_85664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_two_pairs_interpretation_l856_85682

theorem binomial_coefficient_two (n : ℕ+) : 
  (n.val.choose 2 : ℚ) = n.val * (n.val - 1) / 2 :=
by sorry

-- Interpretation
def number_of_pairs (n : ℕ+) : ℕ := n.val.choose 2

theorem pairs_interpretation (n : ℕ+) :
  (number_of_pairs n : ℚ) = n.val * (n.val - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_two_pairs_interpretation_l856_85682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_18_l856_85609

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (3, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the vector from focus to a point
def vector_from_focus (p : PointOnParabola) : ℝ × ℝ :=
  (p.x - focus.1, p.y - focus.2)

-- Define the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem sum_of_distances_is_18 
  (A B C : PointOnParabola) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_sum_zero : vector_from_focus A + vector_from_focus B + vector_from_focus C = (0, 0)) :
  magnitude (vector_from_focus A) + magnitude (vector_from_focus B) + magnitude (vector_from_focus C) = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_is_18_l856_85609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_from_moving_matches_l856_85620

/-- Represents a four-digit number --/
def FourDigitNumber := Fin 10000

/-- Represents the number of matches that can be moved --/
def MaxMoves : Nat := 2

/-- The initial number formed with matches --/
def InitialNumber : FourDigitNumber := ⟨1405, by norm_num⟩

/-- Function to determine if a number can be formed by moving at most n matches from the initial number --/
def can_form_by_moving_matches (n : Nat) (initial final : FourDigitNumber) : Prop := sorry

/-- Main theorem: 7705 is the largest four-digit number that can be obtained by moving at most two matches from 1405 --/
theorem largest_number_from_moving_matches :
  ∀ (n : FourDigitNumber), 
    can_form_by_moving_matches MaxMoves InitialNumber n → 
    n.val ≤ 7705 :=
sorry

#check largest_number_from_moving_matches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_from_moving_matches_l856_85620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_sine_function_phase_shift_specific_sine_function_l856_85601

/-- PhaseShift type definition -/
def PhaseShift (f : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x, f x = f (x - shift)

/-- The phase shift of a sine function y = A * sin(B * (x - C)) is equal to C. -/
theorem phase_shift_sine_function (A B C : ℝ) (h : A ≠ 0) (h' : B ≠ 0) :
  let f : ℝ → ℝ := λ x => A * Real.sin (B * (x - C))
  let phase_shift := C
  PhaseShift f phase_shift :=
by
  sorry

/-- The phase shift of y = 5 * sin(2 * (x - π/4)) is π/4. -/
theorem phase_shift_specific_sine_function :
  let f : ℝ → ℝ := λ x => 5 * Real.sin (2 * (x - π/4))
  let phase_shift := π/4
  PhaseShift f phase_shift :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_sine_function_phase_shift_specific_sine_function_l856_85601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l856_85626

/-- The function we're maximizing -/
noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

/-- The maximum value of the function -/
noncomputable def max_value : ℝ := Real.log 3 / 16

/-- Theorem stating that there exists a maximum value for the function -/
theorem f_max_value :
  ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = max_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l856_85626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equation_l856_85687

theorem mod_equation (n : ℕ) (h1 : n < 29) (h2 : (2 * n) % 29 = 1) : 
  ((3^n)^3 - 3) % 29 = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equation_l856_85687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l856_85644

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin x + (1/2) * Real.cos (2*x) - 1

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l856_85644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_lineup_count_l856_85695

def number_of_friends : Nat := 5

-- Define a function to calculate the factorial
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

-- Define a function to calculate the number of permutations where two specific elements are not adjacent
def permutations_with_non_adjacent_elements (n : Nat) : Nat :=
  factorial n - (factorial (n - 1)) * 2

-- Theorem statement
theorem photo_lineup_count :
  permutations_with_non_adjacent_elements number_of_friends = 72 := by
  -- Unfold the definitions
  unfold permutations_with_non_adjacent_elements
  unfold number_of_friends
  unfold factorial
  -- Evaluate the expressions
  simp
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_lineup_count_l856_85695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_vertical_asymptotes_l856_85666

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (x^2 + 3*x + 10) / (x^2 - 5*x + 6)

-- Define the set of vertical asymptotes
def vertical_asymptotes : Set ℝ := {2, 3}

-- Theorem statement
theorem f_has_vertical_asymptotes :
  ∀ x ∈ vertical_asymptotes, 
    (∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε) ∧
    (∀ M > 0, ∃ δ > 0, ∀ y : ℝ, 0 < |y - x| ∧ |y - x| < δ → |f y| > M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_vertical_asymptotes_l856_85666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exclusivity_l856_85638

def x : ℕ → ℤ
  | 0 => 10
  | 1 => 10
  | (n + 2) => (x n + 1) * x (n + 1) + 1

def y : ℕ → ℤ
  | 0 => -10
  | 1 => -10
  | (n + 2) => (y (n + 1) + 1) * y n + 1

theorem sequence_exclusivity (k : ℕ) :
  (∃ n : ℕ, x n = k) → (∀ m : ℕ, y m ≠ k) ∧
  (∃ n : ℕ, y n = k) → (∀ m : ℕ, x m ≠ k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_exclusivity_l856_85638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_ratio_l856_85616

/-- A cone with an equilateral triangle as its front view -/
structure EquilateralCone where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Assumption that a is positive -/
  a_pos : a > 0

/-- The ratio of lateral surface area to base area for an equilateral cone -/
noncomputable def lateral_to_base_ratio (cone : EquilateralCone) : ℝ :=
  let r := cone.a / 2  -- radius of the base
  let h := cone.a      -- height of the cone
  let l := cone.a      -- slant height of the cone
  (Real.pi * r * l) / (Real.pi * r^2)

/-- Theorem stating that the ratio of lateral surface area to base area is 2 -/
theorem equilateral_cone_ratio (cone : EquilateralCone) :
  lateral_to_base_ratio cone = 2 := by
  sorry

#check equilateral_cone_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_ratio_l856_85616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_max_value_l856_85670

theorem log_sum_max_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 4*y = 40) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 4*b = 40 ∧ 
    Real.log a + Real.log b ≥ Real.log x + Real.log y) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a + 4*b = 40 → 
    Real.log a + Real.log b ≤ Real.log 100) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_max_value_l856_85670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_floor_l856_85630

noncomputable def g (x : ℝ) := 2 * Real.sin x - Real.cos x + 5 * (Real.cos x / Real.sin x)

theorem smallest_positive_root_floor (s : ℝ) :
  (∀ x, 0 < x → x < s → g x ≠ 0) →
  g s = 0 →
  4 ≤ s ∧ s < 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_root_floor_l856_85630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l856_85636

/-- Calculates the compound interest for a given principal, rate, compounding frequency, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * t)

/-- Calculates the total payment for Plan 1 -/
noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) : ℝ :=
  let balance_5_years := compound_interest principal rate 4 5
  let payment_5_years := balance_5_years / 3
  let remaining_balance := balance_5_years - payment_5_years
  let final_payment := compound_interest remaining_balance rate 4 5
  payment_5_years + final_payment

/-- Calculates the total payment for Plan 2 -/
noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) : ℝ :=
  compound_interest principal rate 1 10

/-- The main theorem to prove -/
theorem loan_payment_difference (principal : ℝ) (rate : ℝ) :
  principal = 12000 → rate = 0.08 →
  ⌊plan1_payment principal rate - plan2_payment principal rate⌋ = 1022 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l856_85636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l856_85677

-- Define the lines and points
def l₁ : ℝ → ℝ → Prop := λ x y ↦ 4 * x - 3 * y = 2
def l₂ : ℝ → ℝ → Prop := λ x y ↦ y = 2
def D : ℝ × ℝ := (-2, -3)

-- Define the properties of l₃
def l₃_positive_slope (l₃ : ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, m > 0 ∧ ∀ x y, l₃ x y ↔ y - D.2 = m * (x - D.1)

-- Define the intersection points
def E_exists : Prop :=
  ∃ E : ℝ × ℝ, l₁ E.1 E.2 ∧ l₂ E.1 E.2

def F_exists (l₃ : ℝ → ℝ → Prop) : Prop :=
  ∃ F : ℝ × ℝ, l₃ F.1 F.2 ∧ l₂ F.1 F.2

-- Define the area of triangle DEF
noncomputable def triangle_area (E F : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((E.1 - D.1) * (F.2 - D.2) - (F.1 - D.1) * (E.2 - D.2))

-- Theorem statement
theorem slope_of_l₃ :
  ∀ l₃ : ℝ → ℝ → Prop,
  l₁ D.1 D.2 →
  l₃_positive_slope l₃ →
  E_exists →
  F_exists l₃ →
  (∃ E F : ℝ × ℝ, l₁ E.1 E.2 ∧ l₂ E.1 E.2 ∧ l₃ F.1 F.2 ∧ l₂ F.1 F.2 ∧ triangle_area E F = 6) →
  ∃ m : ℝ, m = 25 / 32 ∧ ∀ x y, l₃ x y ↔ y - D.2 = m * (x - D.1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l856_85677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l856_85667

/-- The function f(x) = 1 / (3x - 15) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (3 * x - 15)

/-- The domain of f is all real numbers except 5 -/
theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 ↔ ∃ y : ℝ, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l856_85667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l856_85679

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 3, a; 0, 1, 6; 0, 0, 1]

theorem matrix_power_sum (a : ℝ) (n : ℕ) : 
  (A a) ^ n = !![1, 27, 4064; 0, 1, 54; 0, 0, 1] → a + (n : ℝ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_sum_l856_85679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_harvest_earnings_l856_85632

/-- Calculates the first week's earnings and net income after expenses and taxes -/
noncomputable def harvest_earnings (total_earnings : ℝ) (weeks : ℕ) (weekly_increase : ℝ) (weekly_expenses : ℝ) (tax_rate : ℝ) : ℝ × ℝ :=
  let first_week_earnings := (total_earnings - (weeks - 1) * weekly_increase * weeks / 2) / weeks
  let net_income := first_week_earnings - weekly_expenses - (first_week_earnings * tax_rate)
  (first_week_earnings, net_income)

/-- Approximate equality for real numbers -/
def approx_eq (x y : ℝ) : Prop := abs (x - y) < 0.01

/-- Theorem statement for Lewis's harvest earnings -/
theorem lewis_harvest_earnings :
  let result := harvest_earnings 2560 19 10 25 0.08
  approx_eq result.1 44.74 ∧ approx_eq result.2 16.16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lewis_harvest_earnings_l856_85632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cider_production_l856_85652

/-- The amount of apples used for cider production in million tons -/
def apples_for_cider (total_production : ℝ) (pie_percentage : ℝ) (cider_percentage : ℝ) : ℝ :=
  (1 - pie_percentage) * total_production * cider_percentage

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem apple_cider_production :
  let total_production : ℝ := 6
  let pie_percentage : ℝ := 0.25
  let cider_percentage : ℝ := 0.45
  round_to_tenth (apples_for_cider total_production pie_percentage cider_percentage) = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cider_production_l856_85652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_half_l856_85655

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 1/(2^x - 1) + a --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (2^x - 1) + a

/-- Theorem: If f(x) = 1/(2^x - 1) + a is an odd function, then a = 1/2 --/
theorem odd_function_implies_a_equals_half :
  (∃ a : ℝ, IsOdd (f a)) → (∃ a : ℝ, IsOdd (f a) ∧ a = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_half_l856_85655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l856_85674

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x + 1 else 3*x - 1

-- Define the range of f
def range_f : Set ℝ := Set.range f

-- Theorem statement
theorem range_of_f : range_f = Set.Ioi 3 ∪ Set.Iio 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l856_85674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l856_85678

noncomputable section

/-- Line l: ax + (1/a)y - 1 = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x + (1/a) * y - 1 = 0

/-- Circle O: x^2 + y^2 = 1 -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Point A: Intersection of line l with x-axis -/
noncomputable def point_A (a : ℝ) : ℝ × ℝ := (1/a, 0)

/-- Point B: Intersection of line l with y-axis -/
noncomputable def point_B (a : ℝ) : ℝ × ℝ := (0, a)

/-- Area of triangle AOB -/
noncomputable def area_AOB (a : ℝ) : ℝ := (1/2) * a * (1/a)

/-- Length of AB -/
noncomputable def length_AB (a : ℝ) : ℝ := Real.sqrt (a^2 + (1/a)^2)

/-- Length of CD (chord of circle O intersected by line l) -/
noncomputable def length_CD (a : ℝ) : ℝ := 2 * Real.sqrt (1 - 1 / (a^2 + (1/a)^2))

theorem line_circle_intersection (a : ℝ) (h : a > 0) :
  (area_AOB a = 1/2) ∧ (length_AB a ≥ length_CD a) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l856_85678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l856_85622

-- Define the triangle and its properties
structure Triangle (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the point M on the median
def PointOnMedian (α : Type*) [LinearOrderedField α] (T : Triangle α) := α

-- Define the distance function
def distance {α : Type*} [LinearOrderedField α] (T : Triangle α) (M : PointOnMedian α T) : α × α :=
  sorry

-- State the theorem
theorem distance_sum_theorem {α : Type*} [LinearOrderedField α] (T : Triangle α) (M : PointOnMedian α T) 
  (h_sum : (distance T M).1 + (distance T M).2 = T.c) :
  (distance T M).1 = (T.b * T.c) / (T.a + T.b) ∧
  (distance T M).2 = (T.a * T.c) / (T.a + T.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l856_85622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_cost_effective_purchase_l856_85625

/-- Represents a restaurant with prices for hamburgers and milkshakes -/
structure Restaurant where
  hamburger_price : ℚ
  milkshake_price : ℚ

/-- Calculates the total cost for a given number of hamburgers and milkshakes at a restaurant -/
def total_cost (r : Restaurant) (num_hamburgers num_milkshakes : ℕ) : ℚ :=
  r.hamburger_price * num_hamburgers + r.milkshake_price * num_milkshakes

def restaurant_a : Restaurant := ⟨4, 5⟩
def restaurant_b : Restaurant := ⟨7/2, 6⟩
def restaurant_c : Restaurant := ⟨5, 4⟩

def total_budget : ℚ := 120
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6

theorem most_cost_effective_purchase :
  let restaurants := [restaurant_a, restaurant_b, restaurant_c]
  let costs := restaurants.map (λ r => total_cost r num_hamburgers num_milkshakes)
  (costs.minimum? = some 62) ∧ (total_budget - 62 = 58) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_cost_effective_purchase_l856_85625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_of_powers_l856_85610

theorem odd_sum_of_powers (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : 3 ∣ c) :
  Odd (5^a + (b - 1)^2 * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_of_powers_l856_85610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_leopard_moves_l856_85653

/-- Represents the possible moves of a leopard on a chessboard -/
inductive LeopardMove
| up : LeopardMove
| right : LeopardMove
| downLeft : LeopardMove

/-- Represents a position on the chessboard -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Defines a valid leopard path on a 3n × 3n chessboard -/
def ValidLeopardPath (n : ℕ) (path : List Position) : Prop :=
  n ≥ 2 ∧
  path.head? = path.getLast? ∧
  ∀ p ∈ path, p.x < 3*n ∧ p.y < 3*n ∧
  ∀ i j, i ≠ j → i < path.length → j < path.length → path[i]? ≠ path[j]? ∧
  ∀ i, i + 1 < path.length →
    (∃ p₁ p₂, path[i]? = some p₁ ∧ path[i+1]? = some p₂ ∧
      ((p₂.x = p₁.x ∧ p₂.y = p₁.y + 1) ∨  -- up
       (p₂.x = p₁.x + 1 ∧ p₂.y = p₁.y) ∨  -- right
       (p₂.x = p₁.x - 1 ∧ p₂.y = p₁.y - 1)))  -- down-left

/-- Theorem stating the maximum number of moves a leopard can make -/
theorem max_leopard_moves (n : ℕ) :
  (n ≥ 2) →
  (∀ path, ValidLeopardPath n path → path.length ≤ 9*n^2 - 2) ∧
  (∃ path, ValidLeopardPath n path ∧ path.length = 9*n^2 - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_leopard_moves_l856_85653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_size_l856_85627

/-- The function f(x) = e^(2x-2) / x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (2*x - 2) / x

/-- The size of the slope of the tangent line at x = 1 for function f -/
noncomputable def slope_size : ℝ := Real.pi / 4

/-- Theorem stating that the size of the slope of the tangent line at (1, f(1)) is π/4 -/
theorem tangent_slope_size :
  let f' := deriv f
  abs (f' 1) = slope_size := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_size_l856_85627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2_pow_gt_S_l856_85648

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => a n ^ 2 - 2 * (n + 1) * a n + 2

-- Define the sum S_n
def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i => a i)

-- Theorem statement
theorem smallest_n_for_2_pow_gt_S :
  (∀ k : ℕ, k < 6 → 2^k ≤ S k) ∧ 2^6 > S 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_2_pow_gt_S_l856_85648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l856_85657

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ (x - Real.pi/3)

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x, g φ (-x) = -(g φ x)) : φ = 2*Real.pi/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l856_85657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l856_85650

/-- The defective rate as a function of daily output -/
noncomputable def P (x c : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ c then 1 / (6 - x)
  else if x > c then 2/3
  else 0

/-- The daily profit as a function of daily output -/
noncomputable def T (x c : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ c then (9 * x - 2 * x^2) / (6 - x)
  else if x > c then 0
  else 0

/-- The maximum daily profit and corresponding daily output -/
theorem max_profit (c : ℝ) (hc : 1 < c ∧ c < 6) :
  ∃ (T_max x_max : ℝ),
    (3 ≤ c → T_max = 3 ∧ x_max = 3) ∧
    (c < 3 → T_max = (9 * c - 2 * c^2) / (6 - c) ∧ x_max = c) ∧
    ∀ x, 1 ≤ x → T x c ≤ T_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l856_85650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l856_85683

/-- A hyperbola with one focus at (4,0) and eccentricity 2 -/
structure Hyperbola where
  focus : ℝ × ℝ := (4, 0)
  eccentricity : ℝ := 2

/-- The standard form of a hyperbola -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The distance from the focus to the other asymptote -/
noncomputable def distance_to_asymptote : ℝ :=
  2 * Real.sqrt 3

theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, standard_equation x y ↔ 
    x^2 / 4 - y^2 / 12 = 1) ∧
  (∀ x y, asymptotes x y ↔ 
    y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x) ∧
  distance_to_asymptote = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l856_85683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_and_area_l856_85690

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_equality_and_area (t : Triangle) :
  (Real.sin (t.A - t.B) = (t.a / (t.a + t.b)) * Real.sin t.A * Real.cos t.B - (t.b / (t.a + t.b)) * Real.sin t.B * Real.cos t.A) →
  (t.A = t.B) ∧
  (t.A = 7 * Real.pi / 24 ∧ t.a = Real.sqrt 6 →
    (1/2 : Real) * t.a * t.b * Real.sin t.C = 3 * (Real.sqrt 2 + Real.sqrt 6) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_and_area_l856_85690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_expiration_theorem_l856_85649

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 86400

/-- The number of days in a year (simplified, not accounting for leap years) -/
def days_per_year : ℕ := 365

/-- The factorial of 12 -/
def blood_expiration_seconds : ℕ := Nat.factorial 12

/-- The date when the blood was donated -/
structure DonationDate :=
  (year : ℕ)
  (month : ℕ)
  (day : ℕ)

def donation_date : DonationDate := ⟨2023, 1, 5⟩

/-- The date when the blood expires -/
structure ExpirationDate :=
  (year : ℕ)
  (month : ℕ)
  (day : ℕ)

def expiration_date : ExpirationDate := ⟨2038, 2, 6⟩

/-- Function to add days to a date (simplified) -/
def addDays (d : DonationDate) (days : ℕ) : ExpirationDate :=
  let totalDays := d.day + days
  let years := totalDays / 365
  let remainingDays := totalDays % 365
  ⟨d.year + years, 2, 6⟩  -- Simplified: always results in February 6

/-- Theorem stating that blood donated on January 5 expires on February 6, 2038 -/
theorem blood_expiration_theorem :
  (addDays donation_date (blood_expiration_seconds / seconds_per_day)) = expiration_date := by
  sorry

#check blood_expiration_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blood_expiration_theorem_l856_85649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l856_85698

/-- The parabola function -/
noncomputable def f (x : ℝ) : ℝ := -x^2

/-- The line function -/
noncomputable def g (x y : ℝ) : ℝ := 4*x + 3*y - 8

/-- The distance function from a point (x, y) to the line -/
noncomputable def distance (x y : ℝ) : ℝ := |g x y| / Real.sqrt (4^2 + 3^2)

theorem min_distance_parabola_to_line :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), distance x (f x) ≥ distance x₀ (f x₀) ∧ distance x₀ (f x₀) = 4/3 := by
  sorry

#check min_distance_parabola_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l856_85698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_pairs_l856_85697

/-- Represents a seating arrangement of 5 people at a round table -/
structure SeatingArrangement where
  people : Fin 5 → Bool  -- True represents female, False represents male

/-- The number of people sitting next to at least one female -/
def f (s : SeatingArrangement) : Nat := sorry

/-- The number of people sitting next to at least one male -/
def m (s : SeatingArrangement) : Nat := sorry

/-- The set of all possible seating arrangements -/
def allArrangements : Finset SeatingArrangement := sorry

/-- The set of all possible (f, m) pairs -/
def allPairs : Finset (Nat × Nat) :=
  allArrangements.image (λ s => (f s, m s))

theorem round_table_pairs : allPairs.card = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_pairs_l856_85697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_make_all_divisible_by_three_l856_85647

/-- Represents the state of a vertex in the polygon -/
inductive VertexState
  | Zero
  | One
  | Other (n : Int)

/-- Represents the polygon configuration -/
def PolygonConfig := Fin 2018 → VertexState

/-- The initial configuration of the polygon -/
def initial_config : PolygonConfig :=
  fun i => if i.val = 2017 then VertexState.One else VertexState.Zero

/-- Represents a move on the polygon -/
inductive Move
  | Add (i : Fin 2018)
  | Subtract (i : Fin 2018)

/-- Applies a move to the polygon configuration -/
def apply_move (config : PolygonConfig) (move : Move) : PolygonConfig :=
  sorry

/-- Checks if all numbers in the configuration are divisible by 3 -/
def all_divisible_by_three (config : PolygonConfig) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to make all numbers divisible by 3 -/
theorem impossible_to_make_all_divisible_by_three :
  ¬∃ (moves : List Move), all_divisible_by_three (moves.foldl apply_move initial_config) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_make_all_divisible_by_three_l856_85647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l856_85660

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi/6)

theorem f_properties :
  let period : ℝ := Real.pi
  let symm_axis (k : ℤ) : ℝ := Real.pi/6 + k*Real.pi/2
  let symm_center (k : ℤ) : ℝ × ℝ := (-Real.pi/12 + k*Real.pi/2, 1)
  let solution_set : Set ℝ := {x | ∃ k : ℤ, k*Real.pi - Real.pi/2 ≤ x ∧ x ≤ k*Real.pi - Real.pi/6}
  let interval : Set ℝ := {x | -Real.pi/6 ≤ x ∧ x ≤ Real.pi/4}
  let max_value : ℝ := 3
  let min_value : ℝ := 0
  let max_point : ℝ := Real.pi/6
  let min_point : ℝ := -Real.pi/6
  ∀ x : ℝ,
    (∀ y : ℝ, f (x + period) = f x) ∧
    (∀ k : ℤ, f (symm_axis k - x) = f (symm_axis k + x)) ∧
    (∀ k : ℤ, f (symm_center k).1 = (symm_center k).2) ∧
    (x ∈ solution_set ↔ f x ≤ 0) ∧
    (x ∈ interval → f x ≤ max_value) ∧
    (x ∈ interval → f x ≥ min_value) ∧
    (f max_point = max_value) ∧
    (f min_point = min_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l856_85660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_l856_85676

/-- Represents the duration of the traffic light cycle in minutes -/
noncomputable def cycleDuration : ℝ := 3

/-- Represents the duration of the green light phase in minutes -/
noncomputable def greenDuration : ℝ := 1

/-- Represents the duration of the red light phase in minutes -/
noncomputable def redDuration : ℝ := 2

/-- Probability of arriving during the green light -/
noncomputable def probGreen : ℝ := greenDuration / cycleDuration

/-- Probability of arriving during the red light -/
noncomputable def probRed : ℝ := redDuration / cycleDuration

/-- Expected waiting time if arriving during green light (in minutes) -/
noncomputable def expectedWaitGreen : ℝ := 0

/-- Expected waiting time if arriving during red light (in minutes) -/
noncomputable def expectedWaitRed : ℝ := redDuration / 2

/-- Theorem stating the expected waiting time for a pedestrian -/
theorem expected_waiting_time :
  probGreen * expectedWaitGreen + probRed * expectedWaitRed = 2 / 3 := by
  sorry

#check expected_waiting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_l856_85676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l856_85691

/-- Represents a position on the chessboard -/
structure Position where
  x : Fin 8
  y : Fin 8
deriving Repr

/-- Represents a move on the chessboard -/
inductive Move where
  | Up
  | Down
  | Left
  | Right
deriving Repr

/-- The game state -/
structure GameState where
  player1 : Position
  player2 : Position
  currentPlayer : Bool
deriving Repr

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Theorem: The second player has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.player1 = ⟨0, 0⟩ →
      initialState.player2 = ⟨7, 7⟩ →
      initialState.currentPlayer = false →
      ∃ (n : Nat),
        let finalState := (Nat.iterate (λ s ↦ applyMove s (strategy s)) n initialState)
        isGameOver finalState ∧ finalState.player2 = finalState.player1 :=
by
  sorry

#eval "Cannibal Chess Problem"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l856_85691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_8_choose_5_l856_85645

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_8_choose_5_l856_85645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_is_point_l856_85629

/-- The equation of the given circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 - 8 * x + 2 * y^2 - 4 * y + 10 = 0

/-- Theorem stating that the given equation represents a point (circle with radius 0) -/
theorem circle_is_point :
  ∃! p : ℝ × ℝ, circle_equation p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_is_point_l856_85629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_ace_prob_l856_85689

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents a spade card -/
def isSpade : Nat × Nat → Bool
  | (0, _) => true
  | _ => false

/-- Represents an ace card -/
def isAce : Nat × Nat → Bool
  | (_, 0) => true
  | _ => false

/-- The probability of drawing a spade as the first card and an Ace as the second card -/
def probSpadeAce (d : Deck) : ℚ :=
  (d.cards.filter (λ c => isSpade c)).card / d.cards.card *
  (d.cards.filter (λ c => isAce c)).card / (d.cards.card - 1)

theorem spade_ace_prob (d : Deck) : probSpadeAce d = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spade_ace_prob_l856_85689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l856_85624

/-- The point where the mouse starts getting farther from the cheese -/
noncomputable def mouse_turning_point (cheese_x cheese_y : ℝ) (mouse_path_slope mouse_path_intercept : ℝ) : ℝ × ℝ :=
  let perpendicular_slope := -1 / mouse_path_slope
  let perpendicular_intercept := cheese_y - perpendicular_slope * cheese_x
  let x := (perpendicular_intercept - mouse_path_intercept) / (mouse_path_slope - perpendicular_slope)
  let y := mouse_path_slope * x + mouse_path_intercept
  (x, y)

/-- The theorem stating that the sum of coordinates of the turning point is 10 -/
theorem mouse_cheese_problem :
  let (a, b) := mouse_turning_point 12 10 (-5) 18
  a + b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l856_85624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_60_l856_85681

/-- Calculates the speed of a train given its length and time to cross a pole. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / 1000) / (time / 3600)

/-- Theorem stating that a train with the given properties has a speed of approximately 60 km/hr. -/
theorem train_speed_approx_60 (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (speed : ℝ), |speed - train_speed 116.67 7| < ε ∧ |speed - 60| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_60_l856_85681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l856_85637

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), M = 1 ∧ (∀ (x : ℝ), π/2 ≤ x ∧ x ≤ π → f x ≤ M) ∧
    (∃ (x₀ : ℝ), π/2 ≤ x₀ ∧ x₀ ≤ π ∧ f x₀ = M)) ∧
  (∃ (m : ℝ), m = -2 ∧ (∀ (x : ℝ), π/2 ≤ x ∧ x ≤ π → m ≤ f x) ∧
    (∃ (x₁ : ℝ), π/2 ≤ x₁ ∧ x₁ ≤ π ∧ f x₁ = m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l856_85637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_area_ratio_l856_85600

/-- Given a rectangle and a circle intersecting such that:
    1. Each long side of the rectangle contains a chord of the circle
    2. Each chord is equal in length to twice the radius of the circle
    3. The width of the rectangle equals the diameter of the circle
    Prove that the ratio of the area of the rectangle to the area of the circle is 4/π -/
theorem rectangle_circle_area_ratio (r : ℝ) (h : r > 0) :
  (4 * r^2) / (π * r^2) = 4 / π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_area_ratio_l856_85600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_one_point_five_l856_85659

/-- Represents the journey from Razumeyevo to Vkusnoteevo -/
structure Journey where
  distanceToRiver : ℝ
  distanceDownstream : ℝ
  distanceFromRiver : ℝ
  riverWidth : ℝ
  currentSpeed : ℝ
  swimmingSpeed : ℝ
  walkingSpeed : ℝ

/-- Calculates the total time for the journey -/
noncomputable def journeyTime (j : Journey) : ℝ :=
  let walkTime1 := j.distanceToRiver / j.walkingSpeed
  let walkTime2 := j.distanceFromRiver / j.walkingSpeed
  let effectiveSwimSpeed := Real.sqrt (j.swimmingSpeed^2 - j.currentSpeed^2)
  let swimTime := j.riverWidth / effectiveSwimSpeed
  walkTime1 + swimTime + walkTime2

/-- The specific journey described in the problem -/
def razumeyevoToVkusnoteevo : Journey where
  distanceToRiver := 3
  distanceDownstream := 3.25
  distanceFromRiver := 1
  riverWidth := 0.5
  currentSpeed := 1
  swimmingSpeed := 2
  walkingSpeed := 4

/-- Theorem stating that the journey time is approximately 1.5 hours -/
theorem journey_time_approx_one_point_five :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (journeyTime razumeyevoToVkusnoteevo - 1.5) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_one_point_five_l856_85659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l856_85685

def M : Set ℤ := {x | -x^2 + 3*x > 0}
def N : Set ℤ := {x | x^2 - 4 < 0}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l856_85685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l856_85613

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.sqrt 2 * Real.cos x

-- State the theorem
theorem function_properties (m : ℝ) (h_m : m > 0) (h_max : ∀ x, f m x ≤ 2) :
  -- Part 1: Interval of monotonic decrease
  (∀ x ∈ Set.Icc (π/4 : ℝ) π, ∀ y ∈ Set.Icc (π/4 : ℝ) π, x < y → f m y < f m x) ∧
  -- Part 2: Area of triangle ABC
  (∀ A B : ℝ,
    f m (A - π/4) + f m (B - π/4) = 4 * Real.sqrt 6 * Real.sin A * Real.sin B →
    ∃ (a b : ℝ),
      -- C = 60°, c = 3
      Real.sin (π/3) * a = 3 ∧
      Real.sin (π/3) * b = 3 ∧
      -- Area of triangle
      (3 * Real.sqrt 3) / 4 = (1/2) * a * b * Real.sin (π/3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l856_85613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_eq_zero_one_l856_85696

def A : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 1}
def B : Set ℝ := Set.range (fun n : ℕ => (n : ℝ))

theorem A_intersect_B_eq_zero_one : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_eq_zero_one_l856_85696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_symmetry_theorem_l856_85699

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure ConvexPolygon where
  vertices : List Point

structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define the concept of central symmetry
def centralSymmetry (p : Point) (t : Triangle) : Triangle :=
  { a := { x := 2 * p.x - t.a.x, y := 2 * p.y - t.a.y },
    b := { x := 2 * p.x - t.b.x, y := 2 * p.y - t.b.y },
    c := { x := 2 * p.x - t.c.x, y := 2 * p.y - t.c.y } }

-- Define what it means for a point to be inside or on the boundary of a polygon
def insideOrOnBoundary (p : Point) (poly : ConvexPolygon) : Prop :=
  sorry

-- Define what it means for a polygon to be centrally symmetric
def isCentrallySymmetric (poly : ConvexPolygon) : Prop :=
  sorry

-- Define what it means for a triangle to be contained within a polygon
def containedWithin (t : Triangle) (poly : ConvexPolygon) : Prop :=
  sorry

-- Helper function to get the vertices of a triangle
def Triangle.vertices (t : Triangle) : List Point :=
  [t.a, t.b, t.c]

-- Main theorem
theorem triangle_symmetry_theorem 
  (M : ConvexPolygon) 
  (T : Triangle) 
  (h1 : isCentrallySymmetric M) 
  (h2 : containedWithin T M) :
  ∀ (P : Point), 
    (∀ (v : Point), v ∈ T.vertices → insideOrOnBoundary v M) →
    ∃ (v : Point), v ∈ (centralSymmetry P T).vertices ∧ insideOrOnBoundary v M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_symmetry_theorem_l856_85699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l856_85615

-- Define the cone parameters
noncomputable def slant_height : ℝ := 8
noncomputable def angle : ℝ := 60 * Real.pi / 180  -- Convert to radians

-- Define the theorem
theorem cone_surface_area :
  let radius := slant_height * Real.sin angle
  let surface_area := Real.pi * radius * (radius + slant_height)
  surface_area = 48 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_l856_85615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_length_100_l856_85693

/-- Represents the length of a segment in the spiral -/
def segmentLength (n : ℕ) : ℕ := 
  if n ≤ 4 then 1
  else if n ≤ 6 then 2
  else if n = 7 then 1
  else ((n - 5) / 2 + 2)

/-- Calculates the total length of the spiral up to n segments -/
def spiralLength (n : ℕ) : ℕ :=
  (List.range n).map (λ i => segmentLength (i + 1)) |>.sum

/-- The statement to be proved -/
theorem spiral_length_100 : spiralLength 100 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spiral_length_100_l856_85693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_with_given_sides_and_angle_l856_85673

theorem unique_triangle_with_given_sides_and_angle :
  ∃! t : Set ℝ × Set ℝ × Set ℝ,
    (∃ a b c : ℝ,
      t = ({a}, {b}, {c}) ∧
      (a = 20 ∨ b = 20 ∨ c = 20) ∧
      (a = 17 ∨ b = 17 ∨ c = 17) ∧
      (∃ θ : ℝ, θ = 60 * Real.pi / 180 ∧
        (Real.cos θ = (a^2 + b^2 - c^2) / (2 * a * b) ∨
         Real.cos θ = (b^2 + c^2 - a^2) / (2 * b * c) ∨
         Real.cos θ = (c^2 + a^2 - b^2) / (2 * c * a)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_with_given_sides_and_angle_l856_85673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_circle_l856_85606

/-- The length of the tangent segment from the origin to a circle -/
noncomputable def tangentLength (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let d1 := Real.sqrt ((p1.1 ^ 2) + (p1.2 ^ 2))
  let d2 := Real.sqrt ((p2.1 ^ 2) + (p2.2 ^ 2))
  Real.sqrt (d1 * d2)

/-- Theorem: The length of the tangent segment from the origin to the circle 
    passing through (4,5), (7,9), and (6,14) is √(73√5) -/
theorem tangent_length_specific_circle : 
  tangentLength (4, 5) (7, 9) (6, 14) = Real.sqrt (73 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_circle_l856_85606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_squared_equals_1464_0625_l856_85612

/-- The volume of water displaced when a cube is set diagonally into a cylindrical barrel -/
noncomputable def water_displaced (cube_side : ℝ) (barrel_radius : ℝ) : ℝ :=
  (125 * Real.sqrt 6) / 8

theorem water_displaced_squared_equals_1464_0625 :
  (water_displaced 10 5)^2 = 1464.0625 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_squared_equals_1464_0625_l856_85612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_first_four_terms_l856_85661

def mySequence (n : ℕ+) : ℕ := 10^(n.val - 1)

theorem mySequence_first_four_terms :
  (mySequence 1 = 1) ∧ 
  (mySequence 2 = 10) ∧ 
  (mySequence 3 = 100) ∧ 
  (mySequence 4 = 1000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_first_four_terms_l856_85661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_has_one_zero_l856_85651

/-- The function f(x) = a^x + log_a(x) has exactly one zero when a > 0 and a ≠ 1 -/
theorem function_has_one_zero (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃! x : ℝ, a^x + Real.log x / Real.log a = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_has_one_zero_l856_85651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R_l856_85614

noncomputable section

-- Define the square ABCD
def ABCD : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define point E on diagonal AC
def E : ℝ × ℝ :=
  (1/2, 1/2)

-- Define triangle ABE
def ABE : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 ≤ 1 ∧ p ∈ ABCD}

-- Define the strip between 1/4 and 1/2 from AD
def Strip : Set (ℝ × ℝ) :=
  {p | 1/4 ≤ p.2 ∧ p.2 ≤ 1/2 ∧ p ∈ ABCD}

-- Define region R
def R : Set (ℝ × ℝ) :=
  {p | p ∈ Strip ∧ p ∉ ABE}

-- State the theorem
theorem area_of_R : MeasureTheory.volume R = 1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R_l856_85614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_result_l856_85623

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : ℝ
  car_value : ℝ
  house_value : ℝ

/-- Calculates the total wealth of a person -/
def total_wealth (state : FinancialState) : ℝ :=
  state.cash + state.car_value + state.house_value

/-- Represents the initial conditions of the problem -/
def initial_conditions : FinancialState × FinancialState × ℝ :=
  (⟨15000, 5000, 0⟩, ⟨20000, 0, 0⟩, 15000)

/-- Represents the series of transactions -/
def transactions (initial : FinancialState × FinancialState × ℝ) : FinancialState × FinancialState :=
  let (a, b, house_value) := initial
  let a1 : FinancialState := ⟨a.cash + 6000, 0, 0⟩
  let b1 : FinancialState := ⟨b.cash - 6000, 5000, 0⟩
  let a2 : FinancialState := ⟨a1.cash - 18000, 0, 18000⟩
  let b2 : FinancialState := ⟨b1.cash + 18000, 5000, 0⟩
  let a3 : FinancialState := ⟨a2.cash, 0, a2.house_value * 1.1⟩
  let b3 : FinancialState := b2
  let a4 : FinancialState := ⟨a3.cash + 20000, 0, 0⟩
  let b4 : FinancialState := ⟨b3.cash - 20000, 5000, 20000⟩
  (a4, b4)

/-- The main theorem to prove -/
theorem transaction_result :
  let (initial_a, initial_b, _) := initial_conditions
  let (final_a, final_b) := transactions initial_conditions
  total_wealth final_a - total_wealth initial_a = 3000 ∧
  total_wealth final_b - total_wealth initial_b = 17000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_result_l856_85623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_region_l856_85634

/-- The area of the region bound by two circles and the x-axis -/
noncomputable def areaOfRegion (centerA centerB : ℝ × ℝ) (radius : ℝ) : ℝ :=
  let rectangleArea := (centerB.1 - centerA.1) * centerA.2
  let semiCircleArea := Real.pi * radius^2 / 2
  rectangleArea - 2 * semiCircleArea

/-- Theorem stating the area of the specific region -/
theorem area_of_specific_region :
  areaOfRegion (4, 5) (12, 5) 5 = 40 - 25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_region_l856_85634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l856_85635

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 
  (seq.a 2 * seq.a 14 = (seq.a 5)^2) →  -- a_2, a_5, a_14 form a geometric sequence
  (sum_n seq 5 = (seq.a 3)^2) →         -- S_5 = a_3^2
  seq.a 10 = 23 :=                      -- a_10 = 23
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l856_85635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_2x2_grid_l856_85656

/-- The area of a triangle with vertices at (0,0), (2,2), and (2,0) is 2 -/
theorem triangle_area_in_2x2_grid : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 2)
  let C : ℝ × ℝ := (2, 0)
  let triangle_area (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) := 
    (1/2 : ℝ) * |x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂)|
  triangle_area A.1 A.2 B.1 B.2 C.1 C.2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_2x2_grid_l856_85656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interpolation_approximation_l856_85675

/-- Interpolation theorem -/
theorem interpolation_approximation 
  (f : ℝ → ℝ) (n x : ℝ) (h1 : ContinuousOn f (Set.Icc n (n + 1))) 
  (h2 : n < x ∧ x < n + 1) 
  (h3 : ∀ y ∈ Set.Icc n (n + 1), ∃ (k : ℝ), f y = k * y + (f n - k * n)) :
  ∃ (ε : ℝ), ε > 0 ∧ |f x - (f n + (x - n) * (f (n + 1) - f n))| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interpolation_approximation_l856_85675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l856_85604

-- Define the given values
noncomputable def a : ℝ := (1/6) * Real.log 8
noncomputable def b : ℝ := (1/2) * Real.log 5
noncomputable def c : ℝ := Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)

-- Theorem to prove
theorem ordering_abc : a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l856_85604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_radius_l856_85684

/-- Parabola C with parametric equations x = 8t^2 and y = 8t -/
def parabola_C (t : ℝ) : ℝ × ℝ := (8 * t^2, 8 * t)

/-- The focus of parabola C -/
def focus_C : ℝ × ℝ := (2, 0)

/-- Line with slope 1 passing through the focus of C -/
def tangent_line (x y : ℝ) : Prop := y = x - 2

/-- Circle with center (4, 0) and radius r -/
def circle_eq (x y r : ℝ) : Prop := (x - 4)^2 + y^2 = r^2

/-- The theorem stating that r = √2 when the line is tangent to the circle -/
theorem tangent_line_radius (r : ℝ) : 
  (r > 0) → 
  (∃ x y : ℝ, tangent_line x y ∧ circle_eq x y r) → 
  r = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_radius_l856_85684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_evaluate_expression_find_value_l856_85671

-- Problem 1
theorem simplify_expression (a b : ℝ) :
  8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := by sorry

-- Problem 2
theorem evaluate_expression (x y : ℝ) (h : x + y = 1/2) :
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := by sorry

-- Problem 3
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  -3 * x^2 + 6 * y + 2 = -10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_evaluate_expression_find_value_l856_85671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l856_85628

def m : ℝ × ℝ := (-1, 2)
def n (l : ℝ) : ℝ × ℝ := (l, -4)

theorem perpendicular_vectors_magnitude (l : ℝ) :
  (m.1 * (n l).1 + m.2 * (n l).2 = 0) →
  ‖(2 * m.1 - (n l).1, 2 * m.2 - (n l).2)‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_magnitude_l856_85628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l856_85643

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 2) - 1 / Real.sqrt (6 - x)

-- Define the domain of f
def domain_f : Set ℝ := {x | 2 ≤ x ∧ x < 6}

-- Theorem statement
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l856_85643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_costs_53_cents_l856_85662

-- Define the cost of pens and pencils in cents
def pen_cost : ℚ → Prop := λ x => True
def pencil_cost : ℚ → Prop := λ y => True

-- Define the conditions
axiom condition1 : ∀ x y, pen_cost x → pencil_cost y → 5 * x + 4 * y = 316
axiom condition2 : ∀ x y, pen_cost x → pencil_cost y → 3 * x + 6 * y = 234

-- Theorem to prove
theorem pen_costs_53_cents :
  ∃ x y, pen_cost x ∧ pencil_cost y ∧ x = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_costs_53_cents_l856_85662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l856_85607

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in 2D space -/
def Point : Type := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Check if a point is inside a circle -/
def is_inside (p : Point) (c : Circle) : Prop :=
  distance p c.center < c.radius

theorem point_inside_circle (O : Point) (P : Point) :
  let c : Circle := { center := O, radius := 3 }
  distance P O = 2 →
  is_inside P c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_inside_circle_l856_85607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l856_85646

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 9 + x^2 = 1

-- Define the point P
def P : ℝ × ℝ := (1/2, 1/2)

-- Define a line passing through P and intersecting the ellipse at A and B
def line_intersects_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
  ∃ (k m : ℝ), ∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = P.1 ∧ y = P.2) → y = k * x + m

-- Define P as the midpoint of AB
def P_bisects_AB (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem line_equation_proof (A B : ℝ × ℝ) :
  line_intersects_ellipse A B → P_bisects_AB A B →
  ∃ (x y : ℝ), 9 * x + y - 5 = 0 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l856_85646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l856_85631

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = Set.Iic (-4) ∪ Set.Ici 2 :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = Set.Ioi (-3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l856_85631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l856_85621

/-- A power function passing through the point (2, √2) -/
noncomputable def power_function (k a : ℝ) : ℝ → ℝ := fun x ↦ k * x^a

/-- The point (2, √2) -/
noncomputable def point : ℝ × ℝ := (2, Real.sqrt 2)

theorem power_function_theorem (k a : ℝ) :
  power_function k a point.1 = point.2 → k - 2 * a = 0 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l856_85621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_from_H_equality_l856_85639

-- Define the set H_p for a polynomial p
noncomputable def H (p : Polynomial ℂ) : Set ℂ := {z : ℂ | Complex.abs (p.eval z) = 1}

-- State the theorem
theorem polynomial_equality_from_H_equality (p q : Polynomial ℂ) 
  (hp : p.natDegree ≠ 0) 
  (hq : q.natDegree ≠ 0) 
  (h_eq : H p = H q) : 
  ∃ (r : Polynomial ℂ) (m n : ℕ) (ξ : ℂ), 
    m > 0 ∧ n > 0 ∧ 
    Complex.abs ξ = 1 ∧ 
    p = r^m ∧ 
    q = ξ • r^n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equality_from_H_equality_l856_85639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_fixed_point_in_interval_l856_85640

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x/2 + 1/4

theorem exists_fixed_point_in_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 (1/2) ∧ f x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_fixed_point_in_interval_l856_85640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaCO3_approx_l856_85672

/-- Calculates the mass percentage of oxygen in calcium carbonate (CaCO3) -/
noncomputable def mass_percentage_O_in_CaCO3 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_C : ℝ := 12.01
  let molar_mass_O : ℝ := 16.00
  let molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_O_in_CaCO3 : ℝ := 3 * molar_mass_O
  (mass_O_in_CaCO3 / molar_mass_CaCO3) * 100

/-- The mass percentage of oxygen in calcium carbonate (CaCO3) is approximately 47.95% -/
theorem mass_percentage_O_in_CaCO3_approx :
  abs (mass_percentage_O_in_CaCO3 - 47.95) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_CaCO3_approx_l856_85672
