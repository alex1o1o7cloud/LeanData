import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_necessary_not_sufficient_l676_67652

-- Define the property of being an arithmetic sequence
def is_arithmetic_sequence (α β γ : ℝ) : Prop := β - α = γ - β

-- Define the condition from the equation
noncomputable def equation_holds (α β γ : ℝ) : Prop := Real.sin (α + γ) = Real.sin (2 * β)

-- Theorem statement
theorem equation_necessary_not_sufficient :
  (∀ α β γ : ℝ, is_arithmetic_sequence α β γ → equation_holds α β γ) ∧
  (∃ α β γ : ℝ, equation_holds α β γ ∧ ¬is_arithmetic_sequence α β γ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_necessary_not_sufficient_l676_67652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l676_67677

-- Define the function (marked as noncomputable due to dependency on Real.sqrt)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / (x - 3)

-- Define the domain of the function
def domain : Set ℝ := {x | x ≥ -1 ∧ x ≠ 3}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l676_67677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_divisors_sum_14133_l676_67614

def has_six_proper_divisors (n : ℕ) : Prop :=
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ n) (Nat.divisors n)).card = 6

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.sum (Nat.divisors n) id)

theorem six_divisors_sum_14133 (n : ℕ) :
  n > 0 → has_six_proper_divisors n ∧ sum_of_divisors n = 14133 →
  n = 16136 ∨ n = 26666 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_divisors_sum_14133_l676_67614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_O_percentage_in_CuCO3_l676_67644

noncomputable section

/-- Molar mass of Copper in g/mol -/
def Cu_mass : ℝ := 63.55

/-- Molar mass of Carbon in g/mol -/
def C_mass : ℝ := 12.01

/-- Molar mass of Oxygen in g/mol -/
def O_mass : ℝ := 16.00

/-- Number of Oxygen atoms in CuCO3 -/
def O_count : ℕ := 3

/-- Molar mass of CuCO3 in g/mol -/
def CuCO3_mass : ℝ := Cu_mass + C_mass + O_count * O_mass

/-- Mass of Oxygen in one mole of CuCO3 in g/mol -/
def O_in_CuCO3 : ℝ := O_count * O_mass

/-- Mass percentage of Oxygen in CuCO3 -/
def O_percentage : ℝ := (O_in_CuCO3 / CuCO3_mass) * 100

theorem O_percentage_in_CuCO3 : 
  38.83 < O_percentage ∧ O_percentage < 38.85 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_O_percentage_in_CuCO3_l676_67644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l676_67693

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

theorem f_properties (a : ℝ) :
  (∃ e : ℝ, Real.exp 1 = e ∧ (deriv (f a)) e = 3) →
  (a = 1 ∧ ∀ x > 1, f a x / (x - 1) > 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l676_67693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perpendicular_tangents_sin_cos_l676_67628

theorem no_perpendicular_tangents_sin_cos : 
  ¬ ∃ (x : ℝ), (Real.sin x = Real.cos x) ∧ 
  ((Real.cos x) * (-Real.sin x) = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perpendicular_tangents_sin_cos_l676_67628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sum_of_roots_l676_67612

theorem simplify_sum_of_roots : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sum_of_roots_l676_67612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_specific_value_l676_67638

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sqrt 3 * Real.sin x + Real.cos x)

-- Theorem for the maximum value of f
theorem f_max_value : ∃ (M : ℝ), M = 3/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

-- Theorem for the specific value of f
theorem f_specific_value (θ : ℝ) (h : f (θ/2) = 3/4) : f (θ + Real.pi/3) = 7/8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_f_specific_value_l676_67638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_animals_l676_67618

/-- The type representing the animals in Lele's family -/
inductive Animal : Type
| Chicken : Animal
| Duck : Animal

/-- The finset of all animals in Lele's family -/
def FamilyAnimals : Finset Animal := sorry

/-- Decidable equality for Animal -/
instance : DecidableEq Animal :=
  fun a b => match a, b with
  | Animal.Chicken, Animal.Chicken => isTrue rfl
  | Animal.Duck, Animal.Duck => isTrue rfl
  | Animal.Chicken, Animal.Duck => isFalse (fun h => Animal.noConfusion h)
  | Animal.Duck, Animal.Chicken => isFalse (fun h => Animal.noConfusion h)

/-- At least 2 out of any 6 animals are not ducks -/
axiom condition1 : ∀ (s : Finset Animal), s ⊆ FamilyAnimals → s.card = 6 → (s.filter (λ a => a ≠ Animal.Duck)).card ≥ 2

/-- At least 1 out of any 9 animals is a duck -/
axiom condition2 : ∀ (s : Finset Animal), s ⊆ FamilyAnimals → s.card = 9 → (s.filter (λ a => a = Animal.Duck)).card ≥ 1

/-- The maximum number of animals in Lele's family is 12 -/
theorem max_animals : FamilyAnimals.card ≤ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_animals_l676_67618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l676_67607

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  h_pos : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ :=
  Real.sqrt ((a^2 + b^2) / a^2)

/-- A line passing through the right focus of the hyperbola with slope angle 60° -/
noncomputable def line_through_focus (h : Hyperbola a b) : ℝ → ℝ → Prop :=
  λ x y ↦ y = Real.sqrt 3 * (x - Real.sqrt (a^2 + b^2))

theorem hyperbola_eccentricity_range (a b : ℝ) (h : Hyperbola a b) :
  (∃! p : ℝ × ℝ, p.1 > 0 ∧ p.1^2 / a^2 - p.2^2 / b^2 = 1 ∧ line_through_focus h p.1 p.2) →
  eccentricity h ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l676_67607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_possibilities_l676_67673

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The setup of the problem -/
def problem_setup : Prop :=
  ∃ (A B C D E F : Point) (l1 l2 : Line),
    -- A, B, C, D are distinct and collinear
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A.x < B.x ∧ B.x < C.x ∧ C.x < D.x ∧
    l1.a * A.x + l1.b * A.y + l1.c = 0 ∧
    l1.a * B.x + l1.b * B.y + l1.c = 0 ∧
    l1.a * C.x + l1.b * C.y + l1.c = 0 ∧
    l1.a * D.x + l1.b * D.y + l1.c = 0 ∧
    -- AB = 1, BC = 2, CD = 3
    (B.x - A.x)^2 + (B.y - A.y)^2 = 1 ∧
    (C.x - B.x)^2 + (C.y - B.y)^2 = 4 ∧
    (D.x - C.x)^2 + (D.y - C.y)^2 = 9 ∧
    -- E and F lie on a second line parallel to ABCD
    l1.a * E.x + l1.b * E.y + l1.c ≠ 0 ∧
    l1.a * F.x + l1.b * F.y + l1.c ≠ 0 ∧
    l2.a / l1.a = l2.b / l1.b ∧
    l2.a * E.x + l2.b * E.y + l2.c = 0 ∧
    l2.a * F.x + l2.b * F.y + l2.c = 0 ∧
    -- EF = 2
    (F.x - E.x)^2 + (F.y - E.y)^2 = 4

/-- The theorem to be proved -/
theorem triangle_area_possibilities : problem_setup →
  ∃ (S : Finset ℝ), (S.card = 5 ∧
    ∀ (area : ℝ), (∃ (P Q R : Point), P ≠ Q ∧ P ≠ R ∧ Q ≠ R ∧
      ({P, Q, R} : Set Point) ⊆ {A, B, C, D, E, F} ∧
      area = abs ((P.x - R.x) * (Q.y - R.y) - (Q.x - R.x) * (P.y - R.y)) / 2) ↔ area ∈ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_possibilities_l676_67673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l676_67631

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^((-3) : ℝ) - b^((-3) : ℝ)) / (a^((-3/2) : ℝ) - b^((-3/2) : ℝ)) = 
  a^((-2) : ℝ) + a^((-1) : ℝ)*b^((-1) : ℝ) + b^((-2) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l676_67631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_theorem_l676_67642

-- Define the dimensions of the paper
def paper_length : ℚ := 20
def paper_width : ℚ := 12

-- Define the function for the number of shapes after n folds
def num_shapes (n : ℕ) : ℕ := n + 1

-- Define the function for the sum of areas after n folds
noncomputable def sum_areas (n : ℕ) : ℚ :=
  240 * (3 - (n + 3 : ℚ) / 2^n)

-- Theorem statement
theorem paper_folding_theorem (n : ℕ) :
  (num_shapes n = n + 1) ∧
  (sum_areas n = 240 * (3 - (n + 3 : ℚ) / 2^n)) :=
by
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_theorem_l676_67642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_point_for_seven_occurrences_l676_67679

/-- Count the occurrences of a digit in a range of numbers -/
def countDigitOccurrences (digit : Nat) (start : Nat) (stop : Nat) : Nat :=
  sorry

/-- The starting point of a list of integers ending at 1000 where the digit 7 appears 300 times -/
theorem starting_point_for_seven_occurrences :
  ∃ start : Nat, start ≤ 1000 ∧ 
    countDigitOccurrences 7 start 1000 = 300 ∧
    ∀ s : Nat, s < start → countDigitOccurrences 7 s 1000 > 300 :=
by
  sorry

#check starting_point_for_seven_occurrences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_point_for_seven_occurrences_l676_67679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_from_square_cuts_l676_67601

/-- Represents the length of a rectangle formed by rearranging small squares cut from a larger square -/
noncomputable def rectangle_length (square_side : ℝ) (small_square_side : ℝ) : ℝ :=
  (square_side / small_square_side) * (square_side / small_square_side) * small_square_side

/-- Theorem stating the length of the rectangle formed by rearranging small squares -/
theorem rectangle_length_from_square_cuts (square_side : ℝ) (small_square_side : ℝ) 
  (h1 : square_side = 1) -- 1 meter
  (h2 : small_square_side = 1 / 1000) -- 1 millimeter
  : rectangle_length square_side small_square_side = 1000000 := by
  sorry

#check rectangle_length_from_square_cuts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_from_square_cuts_l676_67601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_bounds_l676_67646

theorem cos_2x_plus_2sin_x_bounds :
  ∃ (min max : ℝ), min = -3 ∧ max = 3/2 ∧
  (∀ x : ℝ, min ≤ Real.cos (2*x) + 2*Real.sin x) ∧
  (∀ x : ℝ, Real.cos (2*x) + 2*Real.sin x ≤ max) ∧
  (∃ x₁ x₂ : ℝ, Real.cos (2*x₁) + 2*Real.sin x₁ = min ∧ Real.cos (2*x₂) + 2*Real.sin x₂ = max) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_plus_2sin_x_bounds_l676_67646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_solves_diff_eq_l676_67653

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv^[2] y) x + 4 * y x = 12 * Real.cos (2 * x)

-- Define the general solution
noncomputable def general_solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  C₁ * Real.cos (2 * x) + C₂ * Real.sin (2 * x) + 3 * x * Real.sin (2 * x)

-- Theorem statement
theorem general_solution_solves_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (general_solution C₁ C₂) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_solves_diff_eq_l676_67653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l676_67640

-- Define the function f(x) = x + 4/x
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ ≤ 2 → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l676_67640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_condition_l676_67637

/-- Two intersecting circles with diameters d₁ and d₂ and common chord length h -/
structure IntersectingCircles where
  d₁ : ℝ
  d₂ : ℝ
  h : ℝ
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  h_pos : h > 0
  h_le_d₁ : h ≤ d₁
  h_le_d₂ : h ≤ d₂

/-- The perpendicular condition for intersecting lines -/
def perpendicularCondition (circles : IntersectingCircles) : Prop :=
  1 / circles.h^2 = 1 / circles.d₁^2 + 1 / circles.d₂^2

/-- Predicate to check if two line segments are perpendicular -/
def linesArePerpendicular (C₁ D₁ C₂ D₂ : ℝ × ℝ) : Prop :=
  let v₁ := (D₁.1 - C₁.1, D₁.2 - C₁.2)
  let v₂ := (D₂.1 - C₂.1, D₂.2 - C₂.2)
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

/-- The theorem stating the equivalence between the perpendicular lines and the condition -/
theorem perpendicular_iff_condition (circles : IntersectingCircles) :
  (∃ C₁ D₁ C₂ D₂ : ℝ × ℝ, linesArePerpendicular C₁ D₁ C₂ D₂) ↔ perpendicularCondition circles :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_iff_condition_l676_67637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l676_67617

noncomputable section

open Real

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := sin (x/2 + π/12)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := sin (x - π/4)

-- State the theorem
theorem transform_f_to_g : 
  ∀ x : ℝ, g (2*x + π/3) = f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l676_67617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squared_distances_l676_67697

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the points D, E, and F
def point_D : ℝ × ℝ := (0, 1)
def point_E : ℝ × ℝ := (-2, 1)
noncomputable def point_F : ℝ × ℝ := (-1, Real.sqrt 2)

-- Define the lines l₁ and l₂
def line_l1 (x y : ℝ) : Prop := y = x - 2
def line_l2 (x y : ℝ) : Prop := y = x + 1

-- Define the points A and B as intersections of circle C and line l₂
def point_A : ℝ × ℝ := (0, 1)
def point_B : ℝ × ℝ := (-2, -1)

-- Define the function for |PA|² + |PB|²
def sum_of_squared_distances (a : ℝ) : ℝ :=
  let p := (a, a - 2)
  (p.1 - point_A.1)^2 + (p.2 - point_A.2)^2 + 
  (p.1 - point_B.1)^2 + (p.2 - point_B.2)^2

-- Theorem statement
theorem min_sum_of_squared_distances :
  circle_C point_D.1 point_D.2 ∧
  circle_C point_E.1 point_E.2 ∧
  circle_C point_F.1 point_F.2 ∧
  (∀ x y, line_l2 x y → (circle_C x y ↔ (x = point_A.1 ∧ y = point_A.2) ∨ (x = point_B.1 ∧ y = point_B.2))) →
  ∃ (min : ℝ), min = 13 ∧ ∀ a, line_l1 a (a - 2) → sum_of_squared_distances a ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_squared_distances_l676_67697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_dan_theorem_l676_67616

/-- The minimum speed Dan must exceed to arrive before Cara -/
noncomputable def min_speed_dan (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) : ℝ :=
  distance / (distance / cara_speed - dan_delay)

/-- Theorem stating the minimum speed Dan must exceed -/
theorem min_speed_dan_theorem (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) 
  (h1 : distance = 180)
  (h2 : cara_speed = 30)
  (h3 : dan_delay = 1) :
  min_speed_dan distance cara_speed dan_delay > 36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_dan_theorem_l676_67616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l676_67626

-- Define proposition p
def p : Prop := ∀ x : ℝ, x + 1/x ≥ 2

-- Define proposition q
def q : Prop := ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 ∧ Real.sin x + Real.cos x = Real.sqrt 2

-- Theorem statement
theorem correct_proposition : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_proposition_l676_67626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pens_left_calculation_l676_67659

/-- The number of pens Sally initially brought to school -/
def x : ℕ := 8135

/-- The number of pens left after distributing to all classes -/
noncomputable def pens_left : ℕ :=
  let pens_after_first_two := x - 5130 - 4774
  let pens_after_loss := (pens_after_first_two : ℚ) * (95 : ℚ) / 100
  (⌊pens_after_loss⌋ : ℤ).toNat % 71

/-- Theorem stating the number of pens left after distribution -/
theorem pens_left_calculation (h : x > 9904) :
  pens_left = ((⌊((x - 5130 - 4774 : ℚ) * (95 : ℚ) / 100)⌋ : ℤ).toNat % 71) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pens_left_calculation_l676_67659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_l676_67647

theorem sum_of_last_two_digits : (7^30 + 13^30) % 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_last_two_digits_l676_67647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_response_rate_increase_l676_67634

/-- Calculates the response rate given the number of respondents and total customers --/
noncomputable def responseRate (respondents : ℕ) (totalCustomers : ℕ) : ℝ :=
  (respondents : ℝ) / (totalCustomers : ℝ) * 100

/-- Calculates the percentage increase between two values --/
noncomputable def percentageIncrease (original : ℝ) (new : ℝ) : ℝ :=
  (new - original) / original * 100

theorem survey_response_rate_increase : 
  let originalRespondents : ℕ := 7
  let originalTotal : ℕ := 60
  let redesignedRespondents : ℕ := 9
  let redesignedTotal : ℕ := 63
  let originalRate := responseRate originalRespondents originalTotal
  let redesignedRate := responseRate redesignedRespondents redesignedTotal
  let increase := percentageIncrease originalRate redesignedRate
  ∃ ε > 0, |increase - 22.44| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_response_rate_increase_l676_67634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_power_15_l676_67671

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x and y
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

-- State the theorem
theorem x_plus_y_power_15 : x^15 + y^15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_power_15_l676_67671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_coefficient_sum_l676_67649

noncomputable def ellipse_parametric (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t), (2 * (Real.cos t - 4)) / (3 - Real.cos t))

def ellipse_coefficients : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ := (121, 132, 225, 0, -792, -576)

def coefficient_sum (coeffs : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : ℕ :=
  Int.natAbs coeffs.1 + Int.natAbs coeffs.2.1 + Int.natAbs coeffs.2.2.1 +
  Int.natAbs coeffs.2.2.2.1 + Int.natAbs coeffs.2.2.2.2.1 + Int.natAbs coeffs.2.2.2.2.2

theorem ellipse_coefficient_sum :
  let coeffs := ellipse_coefficients
  (∀ t, let (x, y) := ellipse_parametric t
        coeffs.1 * x^2 + coeffs.2.1 * x * y + coeffs.2.2.1 * y^2 +
        coeffs.2.2.2.1 * x + coeffs.2.2.2.2.1 * y + coeffs.2.2.2.2.2 = 0) ∧
  (Nat.gcd (Int.natAbs coeffs.1) (Nat.gcd (Int.natAbs coeffs.2.1) (Nat.gcd (Int.natAbs coeffs.2.2.1)
           (Nat.gcd (Int.natAbs coeffs.2.2.2.1) (Nat.gcd (Int.natAbs coeffs.2.2.2.2.1) (Int.natAbs coeffs.2.2.2.2.2))))) = 1) →
  coefficient_sum coeffs = 1846 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_coefficient_sum_l676_67649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_motion_with_two_fixed_points_l676_67639

/-- A rigid body in 3D space -/
structure RigidBody where
  points : Set (Fin 3 → ℝ)

/-- A motion of a rigid body -/
structure Motion (body : RigidBody) where
  transform : (Fin 3 → ℝ) → (Fin 3 → ℝ)
  preserves_distances : ∀ (p q : Fin 3 → ℝ), p ∈ body.points → q ∈ body.points → 
    ‖transform p - transform q‖ = ‖p - q‖

/-- A point is fixed under a motion if it doesn't move -/
def is_fixed_point (body : RigidBody) (motion : Motion body) (p : Fin 3 → ℝ) : Prop :=
  p ∈ body.points ∧ motion.transform p = p

/-- The theorem stating that there exists a rigid body motion with exactly two fixed points -/
theorem exists_motion_with_two_fixed_points :
  ∃ (body : RigidBody) (motion : Motion body),
    ∃ (p q : Fin 3 → ℝ), p ≠ q ∧ 
      is_fixed_point body motion p ∧ 
      is_fixed_point body motion q ∧
      ∀ r, is_fixed_point body motion r → r = p ∨ r = q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_motion_with_two_fixed_points_l676_67639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_specific_l676_67690

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) : ℝ :=
  4 * (1/2 * base_edge * (Real.sqrt (lateral_edge^2 - (base_edge/2)^2)))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 10 units is equal to 32√21 square units -/
theorem pyramid_face_area_specific : pyramid_face_area 8 10 = 32 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_specific_l676_67690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_roots_quadratic_l676_67625

theorem sin_cos_roots_quadratic (θ : ℝ) (m : ℝ) :
  (4 * (Real.sin θ)^2 + 2 * m * Real.sin θ + m = 0) ∧
  (4 * (Real.cos θ)^2 + 2 * m * Real.cos θ + m = 0) →
  m = 1 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_roots_quadratic_l676_67625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_dot_product_l676_67692

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)

noncomputable def point_M : ℝ × ℝ := (-9 * Real.sqrt 3 / 8, 0)

noncomputable def dot_product (A B : ℝ × ℝ) : ℝ :=
  let M := point_M
  (M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2)

theorem ellipse_constant_dot_product :
  ∀ (l : ℝ → ℝ × ℝ) (t₁ t₂ : ℝ),
    l 0 = left_focus →
    ellipse (l t₁).1 (l t₁).2 →
    ellipse (l t₂).1 (l t₂).2 →
    t₁ ≠ t₂ →
    dot_product (l t₁) (l t₂) = -13/64 :=
by
  sorry

#check ellipse_constant_dot_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_dot_product_l676_67692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_bead_necklace_arrangements_l676_67602

/-- The number of distinct arrangements of n distinct beads on a necklace, 
    considering rotational and reflectional symmetry -/
def necklaceArrangements (n : ℕ) : ℕ := (Nat.factorial n) / (n * 2)

/-- Theorem: The number of distinct arrangements of 7 distinct beads on a necklace, 
    considering rotational and reflectional symmetry, is 360 -/
theorem seven_bead_necklace_arrangements : 
  necklaceArrangements 7 = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_bead_necklace_arrangements_l676_67602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_example_l676_67665

/-- The projection of vector u onto vector v -/
noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scalar * v.1, scalar * v.2)

/-- Theorem: The projection of (3, -4) onto (1, 2) is (-1, -2) -/
theorem projection_example : vector_projection (3, -4) (1, 2) = (-1, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_example_l676_67665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l676_67627

variable (a b : ℝ × ℝ)

noncomputable def angle_between (v w : ℝ × ℝ) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem angle_between_vectors (h1 : a.1 + b.1 = 2 ∧ a.2 + b.2 = -8) 
                              (h2 : -a.1 + 2*b.1 = -11 ∧ -a.2 + 2*b.2 = 5) :
  angle_between a b = Real.pi - Real.arccos (4 * Real.sqrt 185 / 185) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l676_67627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_unique_solution_l676_67657

theorem triangle_unique_solution (a b : ℝ) (A : ℝ) (h1 : a = 30) (h2 : b = 25) (h3 : A = 150 * π / 180) :
  ∃! B : ℝ, 
    0 < B ∧ 
    B < π / 2 ∧ 
    Real.sin A / a = Real.sin B / b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_unique_solution_l676_67657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l676_67663

noncomputable section

-- Define the ellipse and hyperbola
def ellipse (x y a1 b1 : ℝ) : Prop := x^2 / a1^2 + y^2 / b1^2 = 1
def hyperbola (x y a2 b2 : ℝ) : Prop := x^2 / a2^2 - y^2 / b2^2 = 1

-- Define the eccentricity of an ellipse
noncomputable def eccentricity_ellipse (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem eccentricity_range 
  (a1 b1 a2 b2 : ℝ) 
  (h1 : a1 > b1) (h2 : b1 > 0) 
  (h3 : a2 > 0) (h4 : b2 > 0)
  (h5 : ∃ (x y : ℝ), ellipse x y a1 b1 ∧ hyperbola x y a2 b2) 
  (h6 : ∃ (F1 F2 M : ℝ × ℝ), 
    (F1.1^2 + F1.2^2 = a1^2 - b1^2) ∧ 
    (F2.1^2 + F2.2^2 = a1^2 - b1^2) ∧
    (F1.1^2 + F1.2^2 = a2^2 + b2^2) ∧ 
    (F2.1^2 + F2.2^2 = a2^2 + b2^2) ∧
    ellipse M.1 M.2 a1 b1 ∧ 
    hyperbola M.1 M.2 a2 b2 ∧
    M.1 > 0 ∧ M.2 > 0 ∧
    (M.1 - F1.1)^2 + (M.2 - F1.2)^2 + (M.1 - F2.1)^2 + (M.2 - F2.2)^2 = 
    ((M.1 - F1.1)^2 + (M.2 - F1.2)^2) * ((M.1 - F2.1)^2 + (M.2 - F2.2)^2))
  (h7 : eccentricity_ellipse a1 b1 ≥ Real.sqrt 6 / 3)
  (h8 : eccentricity_ellipse a1 b1 < 1) :
  1 < eccentricity_hyperbola a2 b2 ∧ eccentricity_hyperbola a2 b2 ≤ Real.sqrt 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l676_67663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_novels_l676_67672

theorem percentage_of_novels (total_books : ℕ) (graphic_novels : ℕ) (comic_books_percent : ℚ) : 
  total_books = 120 → 
  graphic_novels = 18 → 
  comic_books_percent = 20 / 100 →
  (total_books - graphic_novels - (comic_books_percent * ↑total_books).floor) / ↑total_books * 100 = 65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_novels_l676_67672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l676_67635

/-- The line in standard form Ax + By + C = 0 --/
noncomputable def line (x y : ℝ) : ℝ := 25 * x - 15 * y + 12

/-- The distance from a point (x, y) to the line --/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |line x y| / Real.sqrt (25^2 + 15^2)

/-- Theorem stating the minimum distance from integer points to the line --/
theorem min_distance_to_line :
  ∃ (d : ℝ), d = Real.sqrt 34 / 85 ∧
  ∀ (x y : ℤ), distance_to_line (↑x : ℝ) (↑y : ℝ) ≥ d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l676_67635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meters_examined_l676_67654

/-- Represents the percentage of meters rejected as defective -/
noncomputable def rejection_rate : ℚ := 5 / 10000

/-- Represents the number of meters rejected -/
def rejected_meters : ℕ := 4

/-- Theorem stating that given the rejection rate and number of rejected meters, 
    the total number of meters examined is 8000 -/
theorem meters_examined : 
  ∃ (total_meters : ℕ), 
    (↑rejected_meters : ℚ) / (↑total_meters : ℚ) = rejection_rate ∧ 
    total_meters = 8000 := by
  use 8000
  constructor
  · -- Proof that (4 : ℚ) / (8000 : ℚ) = 5 / 10000
    sorry
  · -- Proof that 8000 = 8000
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meters_examined_l676_67654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_and_range_l676_67660

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := (Real.log x / Real.log 2)^2 - 2*a*(Real.log x / Real.log 2) + b

-- State the theorem
theorem minimum_and_range (a b : ℝ) :
  (∀ x > 0, f x a b ≥ f (1/4) a b) ∧ (f (1/4) a b = -1) →
  (a = -2 ∧ b = 3) ∧
  (∀ x > 0, f x (-2) 3 < 0 ↔ 1/8 < x ∧ x < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_and_range_l676_67660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_function_composition_l676_67658

-- Define the set A
inductive A : Type
| yes : A
| no : A

-- Define decision function
def is_decision_function {n : ℕ} (f : (Fin n → A) → A) : Prop :=
  (∀ x : Fin n → A, f (λ i ↦ match x i with | A.yes => A.no | A.no => A.yes) ≠ f x) ∧
  (∀ x : Fin n → A, ∀ i : Fin n, f (Function.update x i (f x)) = f x)

-- Define dictatoric function
def is_dictatoric_function {n : ℕ} (d : (Fin n → A) → A) : Prop :=
  ∃ i : Fin n, ∀ x : Fin n → A, d x = x i

-- Define democratic function for 3 arguments
def democratic_function (x y z : A) : A :=
  match x, y, z with
  | A.yes, A.yes, _ | A.yes, _, A.yes | _, A.yes, A.yes => A.yes
  | A.no, A.no, _ | A.no, _, A.no | _, A.no, A.no => A.no

-- Theorem statement
theorem decision_function_composition {n : ℕ} (f : (Fin n → A) → A) 
  (hf : is_decision_function f) : 
  ∃ g : (Fin n → A) → A, (is_decision_function g ∨ is_dictatoric_function g) ∧ 
  (∀ x : Fin n → A, f x = g x) := by
  sorry

#check decision_function_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decision_function_composition_l676_67658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deer_distribution_problem_l676_67645

theorem deer_distribution_problem (x : ℚ) : 
  (∃ (y : ℚ), x + y = 100 ∧ y > 0 ∧ (y * 3).num % (y * 3).den = 0) → 
  x + (1/3) * x = 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deer_distribution_problem_l676_67645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_f_to_g_l676_67685

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem shift_f_to_g :
  ∀ x : ℝ, f (x + Real.pi / 6) = g x :=
by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_f_to_g_l676_67685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l676_67615

/-- The distance between two points A and B, given specific conditions about three people's movements --/
theorem distance_AB (v_A v_B v_C : ℝ) (d_AC : ℝ) 
  (h_first_meet : d_AC / v_A = d_AC / v_B)
  (h_C_behind : d_AC / v_A = (d_AC - 100) / v_C)
  (h_A_extra : 108 / v_A = 100 / v_C)
  (h_second_meet : ∃ t : ℝ, d_AC + v_A * t = 3750 - v_B * t ∧ d_AC + v_C * t = 3750)
  : d_AC + (3750 - d_AC) = 3750 := by
  sorry

#check distance_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l676_67615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l676_67623

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f is decreasing on (-1, 1)
def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

-- Define the condition f(1-a) ≤ f(3a-2)
def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (1 - a) ≤ f (3 * a - 2)

-- State the theorem
theorem range_of_a (h1 : is_decreasing_on_interval f) :
  (∀ a, condition f a) → {a : ℝ | 1/3 < a ∧ a ≤ 3/4} = {a : ℝ | condition f a} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l676_67623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_five_from_six_l676_67613

/-- The number of ways to seat n people from a group around a circular table -/
def circular_seating_arrangements (seated_people : ℕ) : ℕ :=
  if seated_people ≤ 1 then 1 else (seated_people - 1).factorial

/-- Theorem stating that the number of ways to seat 5 people from a group of 6 
    around a circular table is 24 -/
theorem circular_seating_five_from_six :
  circular_seating_arrangements 5 = 24 := by
  rfl

#eval circular_seating_arrangements 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_seating_five_from_six_l676_67613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_implies_a_bound_l676_67643

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + 2*a*x - Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := x + 2*a - 1/x

theorem increasing_function_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (1/3 : ℝ) 2, f_deriv a x ≥ 0) → a ≥ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_implies_a_bound_l676_67643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l676_67621

theorem inequality_proof (k l m n : ℕ) 
  (hk : k > 0) (hl : l > 0) (hm : m > 0) (hn : n > 0)
  (h1 : k < l) (h2 : l < m) (h3 : m < n) 
  (h4 : k * n = l * m) : 
  ((n - k : ℚ) / 2)^2 ≥ k + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l676_67621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_for_seven_points_l676_67619

/-- A configuration of points and segments in a plane. -/
structure PointConfiguration where
  points : Finset (ℝ × ℝ)
  segments : Finset ((ℝ × ℝ) × (ℝ × ℝ))

/-- A predicate that checks if a configuration satisfies the condition
    that any three points have at least two connected by a segment. -/
def satisfiesCondition (config : PointConfiguration) : Prop :=
  ∀ p q r, p ∈ config.points → q ∈ config.points → r ∈ config.points →
    p ≠ q → q ≠ r → p ≠ r →
    (p, q) ∈ config.segments ∨ (q, r) ∈ config.segments ∨ (p, r) ∈ config.segments

/-- The main theorem stating that 9 is the minimum number of segments needed. -/
theorem min_segments_for_seven_points :
  ∀ config : PointConfiguration,
    config.points.card = 7 →
    satisfiesCondition config →
    config.segments.card ≥ 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_for_seven_points_l676_67619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_triple_equality_l676_67604

noncomputable def g (x : ℝ) : ℝ := if x ≤ 0 then -x else 3*x - 22

theorem g_triple_equality (a : ℝ) (ha : a < 0) :
  g (g (g 7)) = g (g (g a)) ↔ a = -23/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_triple_equality_l676_67604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_property_l676_67622

/-- Given a function g and its relation to the inverse of f, prove a property of f's coefficients -/
theorem coefficient_property (a b : ℝ) (g f : ℝ → ℝ) : 
  (∀ x, g x = 3 * x + 5) → 
  (∀ x, g x = f⁻¹ x - 1) → 
  (∀ x, f x = a * x + b) → 
  (∀ x, (f ∘ f⁻¹) x = x) → 
  5 * a + 5 * b = -25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_property_l676_67622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_1498_to_1500_is_294_l676_67687

/-- A function that generates the list of digits for positive integers starting with 2 -/
def digitListStartingWith2 : ℕ → List ℕ :=
  sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits in the list -/
def threeDigitNumber (list : List ℕ) : ℕ :=
  sorry

theorem digit_1498_to_1500_is_294 :
  threeDigitNumber (List.take 1500 (digitListStartingWith2 1500)) = 294 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_1498_to_1500_is_294_l676_67687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l676_67662

-- Define the domain M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x + 2^(x+2)

-- Theorem statement
theorem f_min_value : 
  ∃ (min : ℝ), min = 9/4 ∧ ∀ y ∈ M, f y ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l676_67662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l676_67670

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^x + (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (7 : ℝ)^x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l676_67670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_correct_l676_67661

/-- A move is either horizontal (permuting elements within rows) or vertical (permuting elements within columns) -/
inductive Move
| horizontal
| vertical

/-- The minimum number of moves required to achieve any permutation in an m × n table -/
def min_moves (m n : ℕ) : ℕ :=
  if m = 1 ∨ n = 1 then 1 else 3

/-- Predicate stating that a sequence of moves achieves a given permutation -/
def moves_achieve_permutation (m n : ℕ) (moves : List Move) (perm : Fin (m * n) → Fin (m * n)) : Prop :=
  sorry

theorem min_moves_correct (m n : ℕ) :
  (∀ (perm : Fin (m * n) → Fin (m * n)), ∃ (moves : List Move), moves.length = min_moves m n ∧ 
    moves_achieve_permutation m n moves perm) ∧
  (∃ (perm : Fin (m * n) → Fin (m * n)), ∀ (moves : List Move), moves.length < min_moves m n →
    ¬ moves_achieve_permutation m n moves perm) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_correct_l676_67661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_solution_range_inequality_solution_sets_l676_67699

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / (3^x + 1) + a

theorem odd_function_condition (a : ℝ) :
  (∀ x, f a x = -f a (-x)) → a = -1 := by sorry

theorem solution_range (a : ℝ) :
  a = -1 →
  ∀ t, (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f a x + 1 = t) →
  1/2 ≤ t ∧ t ≤ 1 := by sorry

theorem inequality_solution_sets (a m : ℝ) :
  a = -1 →
  (∀ x, f a (x^2 - m*x) ≥ f a (2*x - 2*m)) →
  ((m > 2 → ∀ x, f a (x^2 - m*x) ≥ f a (2*x - 2*m) ↔ 2 ≤ x ∧ x ≤ m) ∧
   (m = 2 → ∀ x, f a (x^2 - m*x) ≥ f a (2*x - 2*m) ↔ x = 2) ∧
   (m < 2 → ∀ x, f a (x^2 - m*x) ≥ f a (2*x - 2*m) ↔ m ≤ x ∧ x ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_solution_range_inequality_solution_sets_l676_67699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l676_67676

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x + 2) / Real.sqrt (x - 1)

-- State the theorem
theorem domain_of_g :
  (∀ x, f x ≠ 0 → 0 < x ∧ x < 4) →
  (∀ x, g x ≠ 0 → 1 < x ∧ x < 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l676_67676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ACB_is_60_degrees_l676_67680

open Real EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define points P, Q, L
variable (P Q L : EuclideanSpace ℝ (Fin 2))

-- Define that BC is the longest side
variable (h_longest : norm (B - C) ≥ max (norm (A - B)) (norm (A - C)))

-- Define that P is on the altitude from A
variable (h_P_on_AA1 : inner (P - A) (B - C) = 0)

-- Define that Q is on the altitude from B
variable (h_Q_on_BB1 : inner (Q - B) (A - C) = 0)

-- Define that P and Q are on the angle bisector of C
variable (h_P_on_bisector : inner (P - C) (A - C) = inner (P - C) (B - C))
variable (h_Q_on_bisector : inner (Q - C) (A - C) = inner (Q - C) (B - C))

-- Define that L is on the circumcircle
variable (h_L_on_circle : norm (L - A) = norm (L - B) ∧ norm (L - B) = norm (L - C))

-- Define that L is on the angle bisector of C
variable (h_L_on_bisector : inner (L - C) (A - C) = inner (L - C) (B - C))

-- Define that AP = LQ
variable (h_AP_eq_LQ : norm (A - P) = norm (L - Q))

-- Theorem statement
theorem angle_ACB_is_60_degrees :
  angle A C B = 60 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ACB_is_60_degrees_l676_67680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_equality_condition_l676_67641

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  3 * cos θ + 2 / cos θ + Real.sqrt 3 * tan θ ≥ 6 :=
by sorry

theorem equality_condition :
  3 * cos (π / 4) + 2 / cos (π / 4) + Real.sqrt 3 * tan (π / 4) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_equality_condition_l676_67641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_doubles_when_radius_doubles_l676_67691

-- Define a cylinder type
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the volume of a cylinder
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

-- Theorem statement
theorem volume_doubles_when_radius_doubles (c : Cylinder) :
  let c' := Cylinder.mk (2 * c.radius) c.height
  volume c' = 4 * volume c := by
  -- Unfold the definitions
  unfold volume
  -- Simplify the expression
  simp [Real.pi, pow_two]
  -- Basic algebra
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_doubles_when_radius_doubles_l676_67691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l676_67674

theorem sum_of_powers (x : ℝ) (h : x = 0.25) : 
  (625 : ℝ)^(-x) + (25 : ℝ)^(-2*x) + (5 : ℝ)^(-4*x) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l676_67674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l676_67656

noncomputable def slope_angle (f : ℝ → ℝ) : ℝ :=
  Real.arctan (deriv f 1)

theorem tangent_slope_angle (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_lim : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (1 + 2*Δx) - f (1 - Δx)) / Δx) - 3| < ε) : 
  slope_angle f = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l676_67656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_value_l676_67655

/-- Theorem: Minimum value of (b^2 + 1) / (3a) for an ellipse with eccentricity 1/2 -/
theorem ellipse_min_value (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt (a^2 - b^2) / a  -- eccentricity
  (e = 1/2) →
  (∀ a' b' : ℝ, a' > b' ∧ b' > 0 ∧ Real.sqrt (a'^2 - b'^2) / a' = 1/2 →
    (b^2 + 1) / (3 * a) ≤ (b'^2 + 1) / (3 * a')) →
  (b^2 + 1) / (3 * a) = Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_min_value_l676_67655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_3_remainder_5_mod_8_l676_67624

def three_digit_probability : ℚ := 37 / 1000

theorem probability_divisible_by_3_remainder_5_mod_8 :
  three_digit_probability = 37 / 1000 := by
  -- Proof goes here
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_3_remainder_5_mod_8_l676_67624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_volume_of_specific_box_l676_67688

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within two units of a box -/
noncomputable def extendedVolume (b : Box) : ℝ :=
  sorry

/-- The main theorem -/
theorem extended_volume_of_specific_box :
  let box : Box := { length := 2, width := 3, height := 4 }
  ∃ (m n p : ℕ), 
    extendedVolume box = (m + n * Real.pi) / p ∧ 
    0 < m ∧ 0 < n ∧ 0 < p ∧
    Nat.Coprime n p ∧
    m + n + p = 337 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_volume_of_specific_box_l676_67688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l676_67630

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x > 0, HasDerivAt (f (-1)) ((Real.log 2) * x - Real.log 2) 1) ∧
  (∀ x > 0, Monotone (f a) ↔ a ≥ (1/2 : ℝ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l676_67630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_36_l676_67651

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define a₀ = 0 to make the sequence 1-indexed
  | 1 => 1  -- Given: a₁ = 1
  | k + 1 => sequence_a k + (k + 1)  -- Given: aₙ - aₙ₋₁ = n

theorem eighth_term_is_36 : sequence_a 8 = 36 := by
  -- Proof steps would go here
  sorry

#eval sequence_a 8  -- This will evaluate the 8th term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_36_l676_67651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_sided_polygon_diagonals_l676_67683

/-- A convex polygon is a polygon where all interior angles are less than 180 degrees. -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  isConvex : Bool
  sidesEq : sides = n

/-- The number of diagonals in a convex polygon with n sides. -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals. -/
theorem fifteen_sided_polygon_diagonals :
  ∀ (p : ConvexPolygon 15), numDiagonals 15 = 90 := by
  intro p
  unfold numDiagonals
  norm_num

#eval numDiagonals 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteen_sided_polygon_diagonals_l676_67683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_coefficients_l676_67605

theorem existence_of_coefficients (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) :
  ∃ l₁ l₂ l₃ : Fin 3, 
    (l₁ ≠ 0 ∨ l₂ ≠ 0 ∨ l₃ ≠ 0) ∧
    (l₁.val * a₁ + l₂.val * a₂ + l₃.val * a₃) % 3 = 0 ∧
    (l₁.val * b₁ + l₂.val * b₂ + l₃.val * b₃) % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_coefficients_l676_67605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_45_degrees_max_area_when_b_is_2_l676_67633

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*Real.cos B

/-- The given condition that a = b cos C + c sin B -/
def special_condition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

theorem angle_B_is_45_degrees (t : Triangle) (h : special_condition t) :
  t.B = Real.pi/4 := by sorry

theorem max_area_when_b_is_2 (t : Triangle) (h1 : special_condition t) (h2 : t.b = 2) :
  (∀ t' : Triangle, special_condition t' → t'.b = 2 → 
    1/2 * t'.a * t'.c * Real.sin t'.B ≤ Real.sqrt 2 + 1) ∧
  (∃ t' : Triangle, special_condition t' ∧ t'.b = 2 ∧
    1/2 * t'.a * t'.c * Real.sin t'.B = Real.sqrt 2 + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_45_degrees_max_area_when_b_is_2_l676_67633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_angle_probability_l676_67609

open Real Set

-- Define the sample space for a single die roll
def DieRoll : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the vector types
def Vec2 := ℝ × ℝ

-- Define the angle between two vectors
noncomputable def angle (v w : Vec2) : ℝ := 
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Define the probability measure for the dice rolls
noncomputable def prob (A : Finset (ℕ × ℕ)) : ℝ := (A.card : ℝ) / 36

-- State the theorem
theorem dice_roll_angle_probability :
  let outcomes := DieRoll.product DieRoll
  let favorable_outcomes := outcomes.filter (fun p => 0 < angle (p.1, p.2) (1, -1) ∧ angle (p.1, p.2) (1, -1) < π/2)
  prob favorable_outcomes = 5/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_angle_probability_l676_67609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l676_67610

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the line
def my_line (k x y : ℝ) : Prop := y - 1 = k * (x - 1)

-- Theorem statement
theorem line_intersects_circle (k : ℝ) :
  ∃ (x y : ℝ), my_circle x y ∧ my_line k x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l676_67610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_positive_integers_l676_67667

-- Define the type for positive integers
def PositiveInt : Type := { n : ℕ // n > 0 }

-- Define what it means for three numbers to form an arithmetic progression
def IsArithmeticProgression (a b c : PositiveInt) : Prop :=
  b.val - a.val = c.val - b.val

-- Define what it means for a set to contain no three-term arithmetic progression
def NoThreeTermAP (S : Set PositiveInt) : Prop :=
  ∀ a b c, a ∈ S → b ∈ S → c ∈ S → ¬IsArithmeticProgression a b c

-- Define what it means for a set to contain no infinite arithmetic progression
def NoInfiniteAP (S : Set PositiveInt) : Prop :=
  ∀ (f : ℕ → PositiveInt), ¬(∀ n : ℕ, f n ∈ S ∧ IsArithmeticProgression (f n) (f (n+1)) (f (n+2)))

-- State the theorem
theorem partition_positive_integers :
  ∃ (A B : Set PositiveInt), 
    (∀ n : PositiveInt, n ∈ A ∨ n ∈ B) ∧
    (A ∩ B = ∅) ∧
    NoThreeTermAP A ∧
    NoInfiniteAP B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_positive_integers_l676_67667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_l676_67675

/-- Given points A, B, and C in 3D space, prove that the distance between C
    and the midpoint of AB is 3. -/
theorem distance_to_midpoint (A B C : ℝ × ℝ × ℝ) : 
  A = (3, 2, 1) → B = (1, 0, 5) → C = (0, 2, 1) → 
  let M := ((A.fst + B.fst) / 2, (A.snd.fst + B.snd.fst) / 2, (A.snd.snd + B.snd.snd) / 2)
  Real.sqrt ((M.fst - C.fst)^2 + (M.snd.fst - C.snd.fst)^2 + (M.snd.snd - C.snd.snd)^2) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_l676_67675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l676_67606

-- Define the points
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (0, 5)
def D : ℝ × ℝ := (-1, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AD : ℝ × ℝ := (D.1 - A.1, D.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def BD : ℝ × ℝ := (D.1 - B.1, D.2 - B.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define magnitude (marked as noncomputable due to use of Real.sqrt)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem rectangle_properties :
  (dot_product AB AD = 0) ∧ 
  (AB = (C.1 - D.1, C.2 - D.2)) ∧
  (dot_product AC BD / (magnitude AC * magnitude BD) = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l676_67606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_promotion_theorem_l676_67603

/-- A box containing white and red balls -/
structure Box where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from the box -/
def prob_red (b : Box) : ℚ :=
  b.red / (b.white + b.red)

/-- The expected number of red balls drawn in n independent draws -/
def expected_red (b : Box) (n : ℕ) : ℚ :=
  n * prob_red b

/-- The probability mass function for binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem mall_promotion_theorem (b : Box) (h1 : b.white = 6) (h2 : b.red = 4) :
  (prob_red b = 2/5) ∧
  (expected_red b 3 = 6/5) ∧
  (∀ k, k ≠ 4 → binomial_pmf 10 (prob_red b) 4 ≥ binomial_pmf 10 (prob_red b) k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mall_promotion_theorem_l676_67603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_7_pow_2012_l676_67636

/-- The function that returns the last two digits of 7^n -/
def lastTwoDigits (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 00  -- This case is mathematically impossible, but needed for exhaustive pattern matching

theorem last_two_digits_of_7_pow_2012 :
  lastTwoDigits 2012 = 01 := by
  rfl  -- reflexivity, since 2012 % 4 = 0

#eval lastTwoDigits 2012  -- This will evaluate to 01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_7_pow_2012_l676_67636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_lengths_l676_67682

/-- The focal length of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- The first ellipse -/
def ellipse1 : Set (ℝ × ℝ) := {(x, y) | x^2 / 25 + y^2 / 9 = 1}

/-- The second ellipse -/
def ellipse2 (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 / (25 - k) + y^2 / (9 - k) = 1}

theorem equal_focal_lengths (k : ℝ) (h : k < 9) :
  focal_length 5 3 = focal_length (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_lengths_l676_67682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l676_67689

/-- The trajectory of point M given the conditions in the problem -/
def trajectory (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 25

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The condition that the ratio of distances is 5 -/
def ratio_condition (x y : ℝ) : Prop :=
  distance x y 26 1 / distance x y 2 1 = 5

/-- A line passing through point (-2, 3) -/
def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y - 3 = k * (x + 2)

/-- The distance from a point to a line -/
noncomputable def distance_to_line (x0 y0 k : ℝ) : ℝ :=
  abs (3 * k + 2) / Real.sqrt (k^2 + 1)

/-- The theorem stating the equations of line l -/
theorem line_equations :
  ∀ x y : ℝ,
  ratio_condition x y →
  trajectory x y →
  (∃ k : ℝ, line_through_point k x y ∧
    (distance_to_line 1 1 k)^2 + 4^2 = 5^2) →
  (x = -2 ∨ 5 * x - 12 * y + 46 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l676_67689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l676_67684

/-- The determinant of a specific 3x3 symmetric matrix is (a^2 + b^2 + c^2 + d^2)^3 -/
theorem det_specific_matrix (a b c d : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![a^2 + b^2 - c^2 - d^2, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
    ![2*b*c + 2*a*d, a^2 - b^2 + c^2 - d^2, 2*c*d - 2*a*b],
    ![2*b*d - 2*a*c, 2*c*d + 2*a*b, a^2 - b^2 - c^2 + d^2]
  ]
  Matrix.det M = (a^2 + b^2 + c^2 + d^2)^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_specific_matrix_l676_67684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l676_67608

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x + (1/2) * x^2 - b * x

theorem function_properties (a b : ℝ) :
  (∀ x : ℝ, x > 0 → (deriv (f a) x = 2 → x = 1)) →
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    deriv (g a b) x₁ = 0 ∧ 
    deriv (g a b) x₂ = 0 ∧
    (∀ x : ℝ, x₁ < x ∧ x < x₂ → deriv (g a b) x ≠ 0)) →
  b ≥ 7/2 →
  (a = 1 ∧ 
   ∀ x₁ x₂ : ℝ, x₁ < x₂ → 
     deriv (g a b) x₁ = 0 → 
     deriv (g a b) x₂ = 0 → 
     g a b x₁ - g a b x₂ ≥ 15/8 - 2 * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l676_67608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_and_profit_properties_l676_67600

/-- Represents the sales and profit model for a product -/
structure SalesModel where
  cost : ℝ  -- Cost price per unit
  price1 : ℝ  -- Initial price point
  sales1 : ℝ  -- Sales at initial price point
  price2 : ℝ  -- Second price point
  sales2 : ℝ  -- Sales at second price point

/-- Calculates the sales volume as a function of price -/
noncomputable def sales_function (model : SalesModel) (x : ℝ) : ℝ :=
  let slope := (model.sales2 - model.sales1) / (model.price2 - model.price1)
  let intercept := model.sales1 - slope * model.price1
  slope * x + intercept

/-- Calculates the daily profit as a function of price -/
noncomputable def profit_function (model : SalesModel) (x : ℝ) : ℝ :=
  (x - model.cost) * (sales_function model x)

/-- Theorem stating the properties of the sales and profit model -/
theorem sales_and_profit_properties (model : SalesModel) 
  (h_cost : model.cost = 34)
  (h_price1 : model.price1 = 48)
  (h_sales1 : model.sales1 = 200)
  (h_price2 : model.price2 = 50)
  (h_sales2 : model.sales2 = 196) :
  (∀ x, sales_function model x = -2 * x + 296) ∧
  (∃ max_price, ∀ x, profit_function model x ≤ profit_function model max_price ∧ max_price = 91) ∧
  (profit_function model 91 = 6498) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_and_profit_properties_l676_67600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l676_67694

-- Define the function f as a set (the domain of f)
def f : Set ℝ := Set.Icc (-6) 9

-- Define the function h
def h (x : ℝ) : ℝ := 3 * x

-- Theorem stating the domain of h
theorem domain_of_h :
  {x : ℝ | h x ∈ f} = Set.Icc (-2) 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l676_67694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_expression_l676_67668

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (120 : ℤ) = Int.gcd 120 ((15*x + 3) * (15*x + 9) * (10*x + 10)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_of_expression_l676_67668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_tan_product_l676_67695

theorem cos_tan_product (α : Real) : 
  ∃ (x y : Real), x = 3/5 ∧ y = -4/5 ∧ x^2 + y^2 = 1 → 
  Real.cos α * Real.tan α = -4/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_tan_product_l676_67695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_no_equal_division_l676_67698

/-- A scalene triangle is a triangle with no two sides of equal length -/
noncomputable def IsScalene (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≠ (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 ≠ (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≠ (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- The area of a triangle given its vertices -/
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Theorem: In a scalene triangle, no line can divide it into two equal areas -/
theorem scalene_triangle_no_equal_division (A B C D : ℝ × ℝ) 
  (h_scalene : IsScalene A B C) 
  (h_on_side : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ D = (B.1 + t*(C.1 - B.1), B.2 + t*(C.2 - B.2))) :
  TriangleArea A B D ≠ TriangleArea A D C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_no_equal_division_l676_67698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_specific_pyramid_l676_67686

/-- A right pyramid with a hexagonal base -/
structure RightPyramid where
  /-- Length of each side of the hexagonal base -/
  base_side : ℝ
  /-- Height of the pyramid (distance from vertex to center of base) -/
  height : ℝ

/-- Calculate the sum of the lengths of all edges of the pyramid -/
noncomputable def sum_of_edges (p : RightPyramid) : ℝ :=
  6 * p.base_side + 6 * Real.sqrt (p.height^2 + (p.base_side * Real.sqrt 3 / 2)^2)

/-- Theorem stating the sum of edges for a specific pyramid -/
theorem sum_of_edges_specific_pyramid :
  ∃ (p : RightPyramid), p.base_side = 8 ∧ p.height = 15 ∧ sum_of_edges p = 48 + 6 * Real.sqrt 273 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_edges_specific_pyramid_l676_67686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_apex_cosine_l676_67611

/-- An isosceles triangle with perimeter 5 times the base length has apex angle cosine of 7/8 -/
theorem isosceles_triangle_apex_cosine (a b c : ℝ) (h_isosceles : a = b) 
  (h_perimeter : a + b + c = 5 * c) : 
  (a^2 + b^2 - c^2) / (2 * a * b) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_apex_cosine_l676_67611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_jelly_bean_next_l676_67681

/-- Represents the number of jelly beans of each color -/
structure JellyBeanCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of jelly beans -/
def total (count : JellyBeanCount) : ℕ :=
  count.red + count.blue + count.green

/-- Represents Jack's actions of eating jelly beans -/
inductive JellyBeanAction
  | eatGreen
  | eatBlue

/-- Applies an action to the jelly bean count -/
def applyAction (count : JellyBeanCount) (action : JellyBeanAction) : JellyBeanCount :=
  match action with
  | JellyBeanAction.eatGreen => ⟨count.red, count.blue, count.green - 1⟩
  | JellyBeanAction.eatBlue => ⟨count.red, count.blue - 1, count.green⟩

/-- Theorem stating the probability of choosing a red jelly bean next -/
theorem prob_red_jelly_bean_next 
  (initial : JellyBeanCount) 
  (h_initial : initial = ⟨15, 20, 16⟩) 
  (actions : List JellyBeanAction)
  (h_actions : actions = [JellyBeanAction.eatGreen, JellyBeanAction.eatBlue]) :
  let final := actions.foldl applyAction initial
  (final.red : ℚ) / (total final : ℚ) = 15 / 49 := by
  sorry

#check prob_red_jelly_bean_next

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_jelly_bean_next_l676_67681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l676_67664

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem stating that a man rowing upstream at 25 kmph and downstream at 45 kmph has a speed of 35 kmph in still water -/
theorem man_rowing_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 25)
  (h2 : downstream_speed = 45) :
  speed_in_still_water upstream_speed downstream_speed = 35 := by
  -- Unfold the definition of speed_in_still_water
  unfold speed_in_still_water
  -- Substitute the given values
  rw [h1, h2]
  -- Simplify the arithmetic
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l676_67664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tonys_weekly_exercise_time_l676_67666

/-- Represents Tony's daily exercise routine -/
structure ExerciseRoutine where
  walkDistance : ℚ
  walkSpeed : ℚ
  runDistance : ℚ
  runSpeed : ℚ

/-- Calculates the total weekly exercise time for a given routine -/
def weeklyExerciseTime (routine : ExerciseRoutine) : ℚ :=
  7 * ((routine.walkDistance / routine.walkSpeed) + (routine.runDistance / routine.runSpeed))

/-- Tony's actual exercise routine -/
def tonysRoutine : ExerciseRoutine :=
  { walkDistance := 3
  , walkSpeed := 3
  , runDistance := 10
  , runSpeed := 5 }

theorem tonys_weekly_exercise_time :
  weeklyExerciseTime tonysRoutine = 21 := by
  -- Unfold the definitions
  unfold weeklyExerciseTime tonysRoutine
  -- Simplify the arithmetic
  simp [Rat.div_def, Rat.mul_def]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tonys_weekly_exercise_time_l676_67666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l676_67648

/-- The number of boys in the class -/
def num_boys : ℕ := 35

/-- The initially calculated average height in cm -/
def initial_avg : ℚ := 182

/-- The wrongly recorded height of one boy in cm -/
def wrong_height : ℚ := 166

/-- The actual height of the boy with the wrongly recorded height in cm -/
def actual_height : ℚ := 106

/-- The actual average height of the boys in the class -/
def actual_avg : ℚ := (num_boys * initial_avg - (wrong_height - actual_height)) / num_boys

theorem actual_average_height :
  (actual_avg * 100).floor / 100 = 18029 / 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_average_height_l676_67648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_theorem_l676_67696

/-- Calculates the overall average runs for a batsman given the number of matches and averages for two sets of matches. -/
def overallAverage (totalMatches : ℕ) (firstSetMatches : ℕ) (firstSetAverage : ℚ) (secondSetAverage : ℚ) : ℚ :=
  let secondSetMatches := totalMatches - firstSetMatches
  let totalRuns := firstSetMatches * firstSetAverage + secondSetMatches * secondSetAverage
  totalRuns / totalMatches

/-- Theorem stating that the overall average for 30 matches is approximately 33.33
    given the specified conditions. -/
theorem batsman_average_theorem :
  let totalMatches : ℕ := 30
  let firstSetMatches : ℕ := 20
  let firstSetAverage : ℚ := 40
  let secondSetAverage : ℚ := 20
  ‖(overallAverage totalMatches firstSetMatches firstSetAverage secondSetAverage : ℝ) - 33.33‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_theorem_l676_67696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l676_67678

def U : Set ℝ := {x | -2 < x ∧ x < 12}
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((U \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l676_67678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_exterior_points_distance_l676_67629

/-- Given a square PQRS with side length 15 and exterior points X and Y,
    prove that XY² = 1394 when QX = RY = 7 and PX = SY = 14 -/
theorem square_exterior_points_distance (P Q R S X Y : ℝ × ℝ) : 
  (∀ (A B : ℝ × ℝ), A ∈ ({P, Q, R, S} : Set (ℝ × ℝ)) ∧ B ∈ ({P, Q, R, S} : Set (ℝ × ℝ)) ∧ A ≠ B → 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 15) →
  Real.sqrt ((Q.1 - X.1)^2 + (Q.2 - X.2)^2) = 7 →
  Real.sqrt ((R.1 - Y.1)^2 + (R.2 - Y.2)^2) = 7 →
  Real.sqrt ((P.1 - X.1)^2 + (P.2 - X.2)^2) = 14 →
  Real.sqrt ((S.1 - Y.1)^2 + (S.2 - Y.2)^2) = 14 →
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 1394 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_exterior_points_distance_l676_67629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l676_67669

theorem pure_imaginary_product (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  (Complex.I * Complex.I = -1) →
  (((3 - 4 * Complex.I) * Complex.ofReal c + (3 - 4 * Complex.I) * (Complex.ofReal d * Complex.I)).re = 0) →
  c / d = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l676_67669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_inverse_in_fourth_quadrant_l676_67632

def z : ℂ := 1 - 2 * Complex.I

theorem z_plus_inverse_in_fourth_quadrant :
  let w := z + 1/z
  (w.re > 0 ∧ w.im < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_inverse_in_fourth_quadrant_l676_67632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixing_solutions_result_l676_67650

/-- Represents a solution with a given weight and initial percentage of Liquid X -/
structure Solution :=
  (weight : ℝ)
  (initialPercentLiquidX : ℝ)
  (temperature : ℝ)

/-- Calculates the adjusted percentage of Liquid X based on temperature difference -/
noncomputable def adjustedPercentLiquidX (s : Solution) (baseTemp : ℝ) : ℝ :=
  s.initialPercentLiquidX + (s.temperature - baseTemp) / 5 * 0.2

/-- Calculates the amount of Liquid X in a solution after temperature adjustment -/
noncomputable def amountLiquidX (s : Solution) (baseTemp : ℝ) : ℝ :=
  s.weight * adjustedPercentLiquidX s baseTemp / 100

/-- Theorem: Mixing solutions results in 1.82% Liquid X -/
theorem mixing_solutions_result (solutionA solutionB : Solution) 
  (hA : solutionA.weight = 300 ∧ solutionA.initialPercentLiquidX = 0.8 ∧ solutionA.temperature = 40)
  (hB : solutionB.weight = 700 ∧ solutionB.initialPercentLiquidX = 1.8 ∧ solutionB.temperature = 20) :
  let baseTemp := 20
  let totalWeight := solutionA.weight + solutionB.weight
  let totalLiquidX := amountLiquidX solutionA baseTemp + amountLiquidX solutionB baseTemp
  totalLiquidX / totalWeight * 100 = 1.82 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixing_solutions_result_l676_67650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_negative_two_l676_67620

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2^x - 3 else 2^(-x) - 3

theorem f_even (x : ℝ) : f x = f (-x) := by sorry

theorem f_negative_two : f (-2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_f_negative_two_l676_67620
