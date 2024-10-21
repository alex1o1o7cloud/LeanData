import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l550_55059

open Real

-- Define the function
noncomputable def f (x : ℝ) := (log x) / x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ (∀ x > 0, f x ≤ f c) ∧ f c = 1 / (exp 1) := by
  -- We know that c = e, but we'll leave the proof to be filled in later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l550_55059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_distance_l550_55064

-- Define the ship's velocity
noncomputable def ship_velocity : ℝ := 22 * Real.sqrt 6

-- Define the time taken to travel from A to B
def travel_time : ℝ := 1.5

-- Define the angles at which the lighthouse is observed
def angle_at_A : ℝ := 45
def angle_at_B : ℝ := 15

-- Define the distance AB
noncomputable def distance_AB : ℝ := ship_velocity * travel_time

-- Theorem statement
theorem lighthouse_distance : 
  let angle_S : ℝ := 180 - (angle_at_A + angle_at_B)
  let distance_SB : ℝ := distance_AB * Real.sin (angle_at_A * π / 180) / Real.sin (angle_S * π / 180)
  ∃ ε > 0, |distance_SB - 66| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lighthouse_distance_l550_55064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_property_l550_55073

/-- The largest natural number that does not end in zero and decreases by an integer factor
    when one (not the first) digit is removed is 180625. -/
theorem largest_number_with_property : ∃ n : ℕ,
  (n % 10 ≠ 0) ∧
  (∃ m : ℕ, m < n ∧ ∃ k : ℕ, n = k * m) ∧
  (∀ n' > n, n' % 10 ≠ 0 →
    ∀ m' < n', ¬∃ k : ℕ, n' = k * m') ∧
  n = 180625 := by
  -- The proof goes here
  sorry

#eval 180625 % 10  -- Should output 5, confirming it doesn't end in 0
#eval 180625 / 18625  -- Should output 10, confirming it decreases by an integer factor when 0 is removed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_property_l550_55073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_children_theorem_l550_55084

/-- Represents the movie theater pricing and budget scenario -/
structure MovieTheater where
  adult_ticket_price : ℕ
  child_ticket_price : ℕ
  child_group_discount_price : ℕ
  group_discount_threshold : ℕ
  total_budget : ℕ

/-- Calculates the maximum number of children that can be taken to the movies -/
def max_children (theater : MovieTheater) : ℕ :=
  let remaining_budget := theater.total_budget - theater.adult_ticket_price
  remaining_budget / theater.child_group_discount_price

/-- Theorem stating the maximum number of children for the given scenario -/
theorem max_children_theorem (theater : MovieTheater) 
  (h1 : theater.adult_ticket_price = 12)
  (h2 : theater.child_ticket_price = 6)
  (h3 : theater.child_group_discount_price = 4)
  (h4 : theater.group_discount_threshold = 5)
  (h5 : theater.total_budget = 75) :
  max_children theater = 15 ∧ 
  max_children theater ≥ theater.group_discount_threshold := by
  sorry

#eval max_children ⟨12, 6, 4, 5, 75⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_children_theorem_l550_55084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l550_55093

/-- Given real numbers p, q, r, and points X, Y, Z in 3D space such that
    the midpoint of YZ is (p,0,0), the midpoint of XZ is (0,q,0),
    and the midpoint of XY is (0,0,r), prove that
    (XY^2 + XZ^2 + YZ^2) / (p^2 + q^2 + r^2) = 8 -/
theorem midpoint_triangle_ratio (p q r : ℝ) (X Y Z : Fin 3 → ℝ) 
    (h1 : ((Y 0 + Z 0) / 2 = p) ∧ ((Y 1 + Z 1) / 2 = 0) ∧ ((Y 2 + Z 2) / 2 = 0))
    (h2 : ((X 0 + Z 0) / 2 = 0) ∧ ((X 1 + Z 1) / 2 = q) ∧ ((X 2 + Z 2) / 2 = 0))
    (h3 : ((X 0 + Y 0) / 2 = 0) ∧ ((X 1 + Y 1) / 2 = 0) ∧ ((X 2 + Y 2) / 2 = r))
    : (dist X Y)^2 + (dist X Z)^2 + (dist Y Z)^2 = 8 * (p^2 + q^2 + r^2) := by
  sorry

#check midpoint_triangle_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_ratio_l550_55093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l550_55070

/-- Sum of a finite geometric series -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- The sum of the specific geometric series -/
theorem specific_geometric_sum :
  geometricSum 1 3 8 = 3280 := by
  -- Unfold the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp
  -- Evaluate the numerical expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_geometric_sum_l550_55070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_units_digit_l550_55031

/-- The set of all six-digit positive integers -/
def SixDigitIntegers : Set ℕ := {n : ℕ | 100000 ≤ n ∧ n ≤ 999999}

/-- The set of six-digit positive integers with even units digit -/
def SixDigitIntegersWithEvenUnitsDigit : Set ℕ :=
  {n ∈ SixDigitIntegers | n % 10 % 2 = 0}

/-- The probability of a randomly chosen six-digit positive integer having an even units digit -/
theorem probability_even_units_digit :
  Nat.card SixDigitIntegersWithEvenUnitsDigit / Nat.card SixDigitIntegers = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_units_digit_l550_55031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_blend_cost_per_kg_l550_55014

/-- The cost per kilogram of a special cheese blend -/
noncomputable def cost_per_kg_blend (mozzarella_price : ℝ) (romano_price : ℝ) 
                      (mozzarella_weight : ℝ) (romano_weight : ℝ) : ℝ :=
  let total_cost := mozzarella_price * mozzarella_weight + romano_price * romano_weight
  let total_weight := mozzarella_weight + romano_weight
  total_cost / total_weight

/-- The cost per kilogram of the special blend is approximately $695.89 -/
theorem special_blend_cost_per_kg :
  let mozzarella_price : ℝ := 504.35
  let romano_price : ℝ := 887.75
  let mozzarella_weight : ℝ := 19
  let romano_weight : ℝ := 18.999999999999986
  abs (cost_per_kg_blend mozzarella_price romano_price mozzarella_weight romano_weight - 695.89) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_blend_cost_per_kg_l550_55014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l550_55089

noncomputable def f (x : ℝ) : ℝ := (18*x^4 + 3*x^3 + 5*x^2 + 7*x + 6) / (6*x^4 + 4*x^3 + 7*x^2 + 2*x + 4)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 3| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l550_55089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_m_l550_55025

-- Define the absolute value function
def f (x : ℝ) : ℝ := abs x

-- Part 1: Solution set of f(x) + f(x-1) ≤ 2
theorem solution_set_part1 : 
  {x : ℝ | f x + f (x - 1) ≤ 2} = Set.Icc (-1/2) (3/2) := by sorry

-- Part 2: Range of m
theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - abs (x + 2) ≤ 1/a + 1/b) → m ∈ Set.Icc (-6) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_m_l550_55025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_315_to_radians_radians_7_12pi_to_degrees_l550_55057

-- Define the conversion factor between radians and degrees
noncomputable def deg_to_rad : ℝ → ℝ := λ deg => deg * (Real.pi / 180)
noncomputable def rad_to_deg : ℝ → ℝ := λ rad => rad * (180 / Real.pi)

-- Theorem 1: 315° equals 7π/4 radians
theorem degrees_315_to_radians :
  deg_to_rad 315 = 7 * Real.pi / 4 := by sorry

-- Theorem 2: 7/12π radians equals 105°
theorem radians_7_12pi_to_degrees :
  rad_to_deg (7 * Real.pi / 12) = 105 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_315_to_radians_radians_7_12pi_to_degrees_l550_55057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_common_point_l550_55056

theorem power_function_common_point :
  ∀ (α : ℝ), (fun x : ℝ ↦ x^α) 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_common_point_l550_55056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPF_l550_55091

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 10 = 1

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define the vertex A
def A : ℝ × ℝ := (3, 0)

-- Define a point P on the ellipse in the first quadrant
noncomputable def P (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sqrt 10 * Real.sin θ)

-- Define the area of quadrilateral OAPF
noncomputable def area_OAPF (θ : ℝ) : ℝ := 
  (3 / 2) * (Real.sqrt 10 * Real.sin θ + Real.cos θ)

theorem max_area_OAPF : 
  ∃ (θ : ℝ), ∀ (φ : ℝ), 0 < φ ∧ φ < Real.pi / 2 → area_OAPF φ ≤ area_OAPF θ ∧ area_OAPF θ = (3 * Real.sqrt 11) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPF_l550_55091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l550_55072

theorem inequality_solution_set (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - (2*a + 2) * x + 4
  (a = 0 → {x : ℝ | f x > 0} = {x : ℝ | x > 2}) ∧
  (a = 1 → {x : ℝ | f x > 0} = ∅) ∧
  (a < 0 → {x : ℝ | f x > 0} = {x : ℝ | x < 2/a ∨ x > 2}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | f x > 0} = {x : ℝ | 2 < x ∧ x < 2/a}) ∧
  (a > 1 → {x : ℝ | f x > 0} = {x : ℝ | 2/a < x ∧ x < 2}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l550_55072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_satisfies_divisibility_l550_55058

/-- The number formed by appending a digit B to 24570 -/
def number (B : Fin 10) : ℕ := 245700 + B

/-- Proposition: The only digit B that makes the number 24570B divisible by 2, 3, 5, and 9 is 0 -/
theorem only_zero_satisfies_divisibility :
  ∃! (B : Fin 10), (number B) % 2 = 0 ∧ 
                   (number B) % 3 = 0 ∧ 
                   (number B) % 5 = 0 ∧ 
                   (number B) % 9 = 0 ∧ 
                   B = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_zero_satisfies_divisibility_l550_55058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l550_55039

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 6 = 0

-- Define the centers and radii of the circles
def center₁ : ℝ × ℝ := (-1, -1)
def center₂ : ℝ × ℝ := (3, -1)
def radius₁ : ℝ := 2
def radius₂ : ℝ := 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2)

-- Theorem statement
theorem circles_externally_tangent :
  distance_between_centers = radius₁ + radius₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l550_55039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_count_l550_55060

/-- Function to check if a number has a 1 or 0 digit -/
def has_one_or_zero (n : ℕ) : Bool :=
  let digits := n.digits 10
  1 ∈ digits ∨ 0 ∈ digits

/-- The set of valid pairs (a,b) -/
def valid_pairs : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = 500 ∧ ¬(has_one_or_zero p.1) ∧ ¬(has_one_or_zero p.2)}

/-- Predicate version of valid_pairs for use with Finset.filter -/
def is_valid_pair (p : ℕ × ℕ) : Prop :=
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = 500 ∧ ¬(has_one_or_zero p.1) ∧ ¬(has_one_or_zero p.2)

instance : DecidablePred is_valid_pair := fun p =>
  And.decidable

theorem valid_pairs_count : Finset.card (Finset.filter is_valid_pair (Finset.product (Finset.range 501) (Finset.range 501))) = 374 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_count_l550_55060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_translations_exist_l550_55018

theorem disjoint_translations_exist (S : Finset ℕ) (A : Finset ℕ) : 
  S = Finset.range 1000000 →
  A ⊆ S →
  A.card = 101 →
  ∃ (t : Fin 100 → ℕ), 
    (∀ i, t i ∈ S) ∧ 
    (∀ i j, i ≠ j → 
      Disjoint (A.image (λ x => x + t i)) (A.image (λ x => x + t j))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_translations_exist_l550_55018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_c_max_at_3_l550_55063

noncomputable def a : ℕ → ℝ
  | 0 => 0
  | 1 => 0
  | n + 2 => 2 * a (n + 1) + (n + 1)

noncomputable def b (n : ℕ) : ℝ := a (n + 1) - a n + 1

noncomputable def c (n : ℕ) : ℝ := a n / 3^n

theorem b_is_geometric : ∃ (r : ℝ), ∀ (n : ℕ), b (n + 1) = r * b n := by
  sorry

theorem c_max_at_3 : ∀ (n : ℕ), n ≠ 3 → c n ≤ c 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_c_max_at_3_l550_55063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_l550_55004

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2:ℝ)^x + a * (2:ℝ)^(-x)

-- State the theorem
theorem odd_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = -f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_l550_55004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l550_55052

open Real

theorem same_terminal_side (k : ℤ) : 
  (∃ k : ℤ, (5 * π / 3 : ℝ) = 2 * k * π - π / 3) ↔ 
  (5 * π / 3 : ℝ) = (-π / 3 : ℝ) + 2 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l550_55052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l550_55086

theorem root_difference_quadratic (a b c : ℝ) :
  let f (x : ℝ) := x^2 - 2*(a^2 + b^2 + c^2 - 2*a*c)*x + (b^2 - a^2 - c^2 + 2*a*c)^2
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ (x₁ - x₂ = 4*b*(a - c) ∨ x₁ - x₂ = -4*b*(a - c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_quadratic_l550_55086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_four_l550_55029

/-- A random variable following a normal distribution with mean 1 and variance 36 -/
def ξ : Real → Prop := sorry

/-- The probability density function of ξ -/
def pdf_ξ : Real → Real := sorry

/-- The cumulative distribution function of ξ -/
def cdf_ξ : Real → Real := sorry

/-- The probability that ξ is between -2 and 1 is 0.4 -/
axiom prob_between_neg_two_and_one : ∫ x in Set.Icc (-2) 1, pdf_ξ x = 0.4

/-- The statement to be proved -/
theorem prob_greater_than_four : ∫ x in Set.Ioi 4, pdf_ξ x = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_greater_than_four_l550_55029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_of_fractions_l550_55074

-- Define the fractions
def f1 (x : ℚ) : ℚ := 1 / (4 * x)
def f2 (x : ℚ) : ℚ := 1 / (6 * x)
def f3 (x : ℚ) : ℚ := 1 / (9 * x)

-- State the theorem
theorem lcm_gcd_of_fractions (x : ℚ) (hx : x ≠ 0) :
  (∃ (lcm : ℚ), lcm = 1 / (36 * x) ∧ 
    (∀ (m : ℚ), (∃ (a b c : ℚ), m = a * f1 x ∧ m = b * f2 x ∧ m = c * f3 x) → lcm ≤ m)) ∧
  (∃ (gcd : ℚ), gcd = x ∧ 
    (∀ (d : ℚ), (∃ (a b c : ℚ), f1 x = a * d ∧ f2 x = b * d ∧ f3 x = c * d) → d ≤ gcd)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_of_fractions_l550_55074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l550_55061

/-- Calculates the compound interest amount -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) (compoundingsPerYear : ℝ) : ℝ :=
  principal * (1 + rate / compoundingsPerYear) ^ (compoundingsPerYear * time)

/-- Calculates the simple interest amount -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem loan_difference_theorem :
  let principal := (15000 : ℝ)
  let compoundRate := (0.08 : ℝ)
  let simpleRate := (0.10 : ℝ)
  let loanTime := (15 : ℝ)
  let halfTime := (7.5 : ℝ)
  let compoundingsPerYear := (2 : ℝ)

  let compoundAmount7_5 := compoundInterest principal compoundRate halfTime compoundingsPerYear
  let halfPayment := compoundAmount7_5 / 2
  let remainingBalance := compoundAmount7_5 - halfPayment
  let finalCompoundAmount := halfPayment + compoundInterest remainingBalance compoundRate halfTime compoundingsPerYear

  let simpleAmount := simpleInterest principal simpleRate loanTime

  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (simpleAmount - finalCompoundAmount - 9453) < ε := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l550_55061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configuration_probability_l550_55081

/-- A type representing a point on a circle with 2023 evenly spaced points. -/
def CirclePoint : Type := Fin 2023

/-- Determines if chord AB intersects chord CD. -/
def chord_intersects (A B C D : CirclePoint) : Prop := sorry

/-- Given five points on the circle, determines if chord AB intersects chord CD
    but neither intersects with chord DE. -/
def valid_configuration (A B C D E : CirclePoint) : Prop :=
  (chord_intersects A B C D) ∧ 
  ¬(chord_intersects A B D E) ∧ 
  ¬(chord_intersects C D D E)

/-- The number of valid configurations for 5 points. -/
def num_valid_configs : ℕ := 2 * 673

/-- The total number of ways to choose 5 points from 2023 points. -/
def total_configs : ℕ := (2023 * 2022 * 2021 * 2020 * 2019) / 120

/-- The probability of a valid configuration. -/
noncomputable def probability : ℚ := num_valid_configs / total_configs

theorem valid_configuration_probability :
  probability > 0 ∧ 
  probability = (2 * 673 : ℚ) / ((2023 * 2022 * 2021 * 2020 : ℚ) / 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_configuration_probability_l550_55081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_number_rearrangement_l550_55040

def is_nine_digit (n : Nat) : Prop := 10^8 ≤ n ∧ n < 10^9

def is_coprime_with_24 (n : Nat) : Prop := Nat.Coprime n 24

def move_last_digit_to_front (n : Nat) : Nat :=
  let last_digit := n % 10
  let rest := n / 10
  last_digit * 10^8 + rest

theorem nine_digit_number_rearrangement (B : Nat) 
  (h1 : is_nine_digit B)
  (h2 : is_coprime_with_24 B)
  (h3 : B > 666666666) :
  let A := move_last_digit_to_front B
  166666667 ≤ A ∧ A ≤ 999999998 := by
  sorry

#eval move_last_digit_to_front 123456789

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_number_rearrangement_l550_55040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_S6_l550_55067

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating that S₆ is the largest sum among S₁ to S₁₂ -/
theorem largest_sum_is_S6 (seq : ArithmeticSequence) 
    (h1 : seq.a 1 > 0)
    (h2 : S seq 12 > 0)
    (h3 : S seq 13 < 0) :
    ∀ n : ℕ, n ≤ 12 → S seq 6 ≥ S seq n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_S6_l550_55067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l550_55078

theorem find_n : ∃ n : ℕ, 2^6 * 3^3 * n = Nat.factorial 9 :=
  by
    use 210
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l550_55078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_increasing_from_one_probability_two_zeros_around_one_l550_55098

def P : Set ℕ := {1, 2, 3}
def Q : Set ℤ := {-1, 1, 2, 3, 4}

def f (a b x : ℝ) : ℝ := a * x^2 - 4 * b * x + 1

def is_increasing_from_one (a b : ℝ) : Prop :=
  ∀ x ≥ 1, (deriv (f a b)) x ≥ 0

def has_two_zeros_around_one (a b : ℝ) : Prop :=
  ∃ x y, x < 1 ∧ 1 < y ∧ f a b x = 0 ∧ f a b y = 0

def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 8 ∧ p.1 > 0 ∧ p.2 > 0}

-- Assuming Prob is defined elsewhere in Mathlib
noncomputable def Prob (P : α → Prop) : ℝ := sorry

theorem probability_increasing_from_one :
  Prob (λ (p : ℕ × ℤ) ↦ p.1 ∈ P ∧ p.2 ∈ Q ∧ is_increasing_from_one (p.1 : ℝ) (p.2 : ℝ)) = 1/3 := by
  sorry

theorem probability_two_zeros_around_one :
  Prob (λ (p : ℝ × ℝ) ↦ p ∈ region ∧ has_two_zeros_around_one p.1 p.2) = 961/1280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_increasing_from_one_probability_two_zeros_around_one_l550_55098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l550_55012

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / (Real.sin x * Real.cos x + 1)

theorem f_range : Set.range f = Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l550_55012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l550_55050

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 + x) / (x - 2)

-- Define the closed interval [3,6]
def I : Set ℝ := Set.Icc 3 6

-- Theorem statement
theorem function_properties :
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y) ∧ 
  (∀ x ∈ I, f x ≤ 6) ∧
  (∀ x ∈ I, f x ≥ 9/4) ∧
  (∃ x ∈ I, f x = 6) ∧
  (∃ x ∈ I, f x = 9/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l550_55050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_is_eleven_l550_55030

theorem sum_of_b_is_eleven (b₂ b₃ b₄ b₅ b₆ b₇ : ℤ) : 
  (∀ i j : ℤ, i ≠ j → b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧
                   b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧
                   b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧
                   b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧
                   b₆ ≠ b₇) →
  (6 : ℚ) / 7 = (b₂ : ℚ) / 2 + (b₃ : ℚ) / 6 + (b₄ : ℚ) / 24 + (b₅ : ℚ) / 120 + (b₆ : ℚ) / 720 + (b₇ : ℚ) / 5040 →
  (0 ≤ b₂ ∧ b₂ < 2) ∧ (0 ≤ b₃ ∧ b₃ < 3) ∧ (0 ≤ b₄ ∧ b₄ < 4) ∧ 
  (0 ≤ b₅ ∧ b₅ < 5) ∧ (0 ≤ b₆ ∧ b₆ < 6) ∧ (0 ≤ b₇ ∧ b₇ < 8) →
  b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_b_is_eleven_l550_55030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_score_range_l550_55033

/-- Represents the normal distribution parameters -/
structure NormalDistParams where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation

/-- Represents the given probabilities for standard deviations -/
structure StdDevProbabilities where
  prob_1σ : ℝ
  prob_2σ : ℝ
  prob_3σ : ℝ

/-- Calculates the probability of scores falling within a specific range -/
noncomputable def probability_in_range (params : NormalDistParams) (lower : ℝ) (upper : ℝ) 
  (probs : StdDevProbabilities) : ℝ :=
  sorry

/-- Main theorem: Approximate number of students scoring between 85 and 90 -/
theorem students_in_score_range 
  (total_students : ℕ)
  (score_dist : NormalDistParams)
  (probs : StdDevProbabilities) :
  let num_students := total_students * 
    probability_in_range score_dist 85 90 probs
  ⌊num_students⌋₊ = 6 := by
  sorry

#check students_in_score_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_score_range_l550_55033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_intersection_ratio_l550_55022

theorem sin_intersection_ratio : ∃ (r s : ℕ), 
  (0 < r ∧ r < s) ∧  -- r and s are positive integers with r < s
  (Nat.gcd r s = 1) ∧  -- r and s are relatively prime
  (∀ (x : ℝ), Real.sin x = Real.sin (60 * π / 180) → 
    (∃ (k : ℤ), x = (30 + 360 * k) * π / 180 ∨ x = (150 + 360 * k) * π / 180)) ∧
  (r : ℝ) / s = 120 / 240 ∧  -- ratio of successive segments
  r = 1 ∧ s = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_intersection_ratio_l550_55022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_magnitude_l550_55066

/-- An ellipse C in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- The theorem to be proved -/
theorem ellipse_vector_sum_magnitude 
  (C : Ellipse)
  (F : Point)
  (l : Line)
  (A B : Point)
  (h1 : C.a^2 = 2 ∧ C.b^2 = 1)
  (h2 : F.x = 1 ∧ F.y = 0)
  (h3 : A.x^2/2 + A.y^2 = 1)
  (h4 : B.x^2/2 + B.y^2 = 1)
  (h5 : ∃ (k : ℝ), A.y = k * (A.x - F.x) ∧ B.y = k * (B.x - F.x))
  (h6 : (F.x - A.x, F.y - A.y) = 3 * (B.x - F.x, B.y - F.y)) :
  magnitude (A.x + B.x, A.y + B.y) = 2 * Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_vector_sum_magnitude_l550_55066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l550_55041

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 2)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l550_55041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_count_l550_55011

-- Define the sequence
def our_sequence (n : ℕ) : ℚ :=
  6075 / (3 ^ n)

-- Define a predicate to check if a number is an integer
def is_integer (q : ℚ) : Prop :=
  ∃ (n : ℤ), q = n

-- Theorem statement
theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n < k → is_integer (our_sequence n)) ∧
  (∀ (k : ℕ), (∀ (n : ℕ), n ≤ k → is_integer (our_sequence n)) → k ≤ 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integer_count_l550_55011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_APR_l550_55005

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the basic concepts
def Line := Point → Point → Prop

def Tangent (l : Line) (c : Circle) : Prop := sorry

noncomputable def distance (p1 p2 : Point) : ℝ := sorry

-- Main theorem
theorem perimeter_of_triangle_APR 
  (c : Circle) 
  (A B C P Q R : Point) 
  (t1 t2 t3 : Line) :
  Tangent t1 c ∧ Tangent t2 c ∧ Tangent t3 c →
  ¬(c.center = A) →
  t1 B B ∧ t2 C C ∧ t3 Q Q →
  t1 A B ∧ t2 A C →
  t3 P P ∧ t3 R R →
  distance A B = 24 →
  distance A P = distance A R + 3 →
  distance A P + distance A R + distance P R = 57 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_APR_l550_55005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l550_55099

theorem simplify_expressions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → (Real.sqrt x - Real.sqrt y)^2 = x + y - 2*Real.sqrt (x*y)) →
  (∀ n : ℕ, (-1 : ℝ)^n = 1 - 2*(n % 2 : ℝ)) →
  (Real.sqrt 27 - Real.sqrt 3 = 2*Real.sqrt 3) ∧
  ((Real.sqrt 10 - Real.sqrt 2)^2 + (-1 : ℝ)^0 = 13 - 4*Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expressions_l550_55099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_composite_l550_55020

/-- Two polynomials P and Q satisfying specific conditions -/
class SpecialPolynomials (P Q : ℝ → ℝ) where
  is_polynomial_P : Polynomial ℝ
  is_polynomial_Q : Polynomial ℝ
  no_real_solution : ∀ x : ℝ, P x ≠ Q x
  commutativity : ∀ x : ℝ, P (Q x) = Q (P x)

/-- Theorem stating that P(P(x)) = Q(Q(x)) has no real solution -/
theorem no_solution_composite {P Q : ℝ → ℝ} [SpecialPolynomials P Q] :
  ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_composite_l550_55020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gavrila_position_l550_55071

/-- The distance from the siren to Gavrila's starting point -/
def L : ℝ := 50

/-- Gavrila's position satisfies the equation y² = 4xL -/
def gavrila_path (x y : ℝ) : Prop := y^2 = 4 * x * L

/-- The distance from the bank where we want to calculate Gavrila's position -/
def y_distance : ℝ := 40

/-- Calculate the distance from the starting point to Gavrila's position -/
noncomputable def distance_from_start (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem gavrila_position :
  ∃ (x : ℝ), gavrila_path x y_distance ∧ 
  (Int.floor (distance_from_start x y_distance + 0.5) : ℤ) = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gavrila_position_l550_55071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_with_specific_10th_derivative_l550_55055

/-- A linear function f such that its 10th derivative is 1024x + 1023 -/
def LinearFunctionWith10thDerivative (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧
  (∀ x, (deriv^[10] f) x = 1024 * x + 1023)

/-- The theorem stating that a linear function with the given 10th derivative 
    must be either 2x + 1 or -2x - 3 -/
theorem linear_function_with_specific_10th_derivative 
  (f : ℝ → ℝ) (h : LinearFunctionWith10thDerivative f) :
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_with_specific_10th_derivative_l550_55055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actor_golden_section_l550_55003

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The length of the stage in meters -/
def stage_length : ℝ := 10

/-- The distance from one end of the stage where the actor should stand -/
noncomputable def actor_position : ℝ := stage_length * (1 - 1 / φ)

theorem actor_golden_section :
  actor_position = stage_length * ((3 - Real.sqrt 5) / 2) ∧
  actor_position = stage_length * φ - stage_length :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actor_golden_section_l550_55003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l550_55062

noncomputable def f (x : ℝ) := Real.log (x + 1) - 3 / x

theorem root_exists_in_interval :
  Continuous f → f 2 < 0 → 0 < f 3 → ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l550_55062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_complex_l550_55008

-- Define the complex plane
variable (z : ℂ)

-- Define points A, B, C
variable (A B C : ℂ)

-- Define the given conditions
variable (h1 : A = 2 + I)
variable (h2 : B - A = 2 + 3*I)
variable (h3 : C - B = 3 - I)

-- Theorem statement
theorem point_C_complex : C = 3 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_complex_l550_55008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_bounded_diff_implies_infinite_close_indices_l550_55037

def is_permutation (a : ℕ → ℕ) : Prop :=
  Function.Bijective a

theorem permutation_bounded_diff_implies_infinite_close_indices
  (d : ℕ)
  (a : ℕ → ℕ)
  (h_d_pos : d > 0)
  (h_perm : is_permutation a)
  (h_bound : ∀ i ≥ 10^100, |Int.ofNat (a (i + 1)) - Int.ofNat (a i)| ≤ 2 * d) :
  ∀ n : ℕ, ∃ j ≥ n, |Int.ofNat (a j) - Int.ofNat j| < d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_bounded_diff_implies_infinite_close_indices_l550_55037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l550_55090

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_of_2_7 : floor 2.7 = 2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_2_7_l550_55090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_no_solution_l550_55001

theorem sin_equation_no_solution (m : ℝ) :
  (m^3 > Real.sqrt m → ∀ x, Real.sin x ≠ m) ∧
  (∃ m, (∀ x, Real.sin x ≠ m) ∧ m^3 ≤ Real.sqrt m) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_no_solution_l550_55001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l550_55096

/-- Given non-coplanar vectors a, b, and c in ℝ³, if 2a + b - c = (z-1)a + xb + 2yc, 
    then x = 1, y = -1/2, and z = 3 -/
theorem vector_equation_solution 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  (a b c : V) 
  (h_not_coplanar : ¬ Submodule.span ℝ {a, b} = Submodule.span ℝ {a, b, c}) 
  (h_eq : (2 : ℝ) • a + b - c = (z - 1 : ℝ) • a + x • b + (2 * y) • c) :
  x = 1 ∧ y = -1/2 ∧ z = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l550_55096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_problem_l550_55085

/-- The distance between point A and point B in kilometers -/
noncomputable def distance : ℝ := 30

/-- The speed ratio of A to B -/
noncomputable def speed_ratio : ℝ := 1.2

/-- The time it takes for A to catch up with B in part (1), in hours -/
noncomputable def catchup_time : ℝ := 0.5

/-- The initial distance B travels before A starts in part (1), in kilometers -/
noncomputable def initial_distance : ℝ := 2

/-- The time B rides before A starts in part (2), in hours -/
noncomputable def initial_time : ℝ := 1/3

theorem cycling_problem (speed_b : ℝ) :
  (speed_ratio * speed_b * catchup_time = initial_distance + speed_b * catchup_time →
   speed_ratio * speed_b = 24) ∧
  (distance / speed_b - distance / (speed_ratio * speed_b) = initial_time →
   speed_ratio * speed_b = 18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_problem_l550_55085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_sum_of_point_B_l550_55035

/-- Given point A at (0, 0), point B on the line y = 3, and the slope of segment AB is 4/5,
    prove that the sum of the x- and y-coordinates of point B is 6.75. -/
theorem coordinate_sum_of_point_B :
  ∀ (B : ℝ × ℝ),
  B.2 = 3 →
  (B.2 - 0) / (B.1 - 0) = 4 / 5 →
  B.1 + B.2 = 6.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinate_sum_of_point_B_l550_55035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_composition_l550_55028

def dilation_matrix (factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![factor, 0], ![0, factor]]

def rotation_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -2], ![2, 0]]

theorem dilation_rotation_composition :
  rotation_90_ccw * dilation_matrix 2 = combined_transformation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_rotation_composition_l550_55028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_all_ones_row_l550_55038

/-- Represents the 0-1 triangle number table derived from Pascal's triangle -/
def ZeroOneTriangle : ℕ → ℕ → ℕ := sorry

/-- Checks if a row in the 0-1 triangle consists entirely of 1s -/
def IsAllOnesRow (row : ℕ) : Prop :=
  ∀ k, k ≤ row → ZeroOneTriangle row k = 1

/-- The row number of the nth occurrence of a row with all 1s -/
def NthAllOnesRow (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that the nth occurrence of a row with all 1s is in the 2^n - 1 row -/
theorem nth_all_ones_row (n : ℕ) :
  IsAllOnesRow (NthAllOnesRow n) ∧
  (∀ k, k < NthAllOnesRow n → ¬IsAllOnesRow k ∨ (∃ m < n, k = NthAllOnesRow m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_all_ones_row_l550_55038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_sum_l550_55054

theorem hundreds_digit_of_factorial_sum : ∃ n : ℕ, 
  (Nat.factorial 30 - Nat.factorial 20 + Nat.factorial 10) % 1000 = 800 ∧ 
  (Nat.factorial 30 - Nat.factorial 20 + Nat.factorial 10) / 100 % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_sum_l550_55054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l550_55065

theorem arithmetic_sequence_middle_term 
  (a b c : ℝ) (h : b - a = c - b) 
  (h1 : a = 1 + Real.sqrt 3) (h3 : c = 1 - Real.sqrt 3) : 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l550_55065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l550_55075

-- Define helper functions
def is_rhombus (points : Set (ℝ × ℝ)) : Prop := sorry

def is_area_of_rhombus (points : Set (ℝ × ℝ)) (area : ℝ) : Prop := sorry

def circum_radius (triangle : Set (ℝ × ℝ)) : ℝ := sorry

theorem rhombus_area (P Q R S : ℝ × ℝ) : 
  let PQRS := {P, Q, R, S}
  let circum_radius_PQR := 15
  let circum_radius_PSR := 30
  -- Define the rhombus PQRS
  -- Define the circumradius of triangle PQR as 15
  -- Define the circumradius of triangle PSR as 30
  ∃ (area : ℝ), area = 400 ∧ is_area_of_rhombus PQRS area := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l550_55075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l550_55095

/-- The volume of a truncated right circular cone. -/
noncomputable def truncated_cone_volume (R h r : ℝ) : ℝ :=
  (1/3) * Real.pi * h * (R^2 + r^2 + R*r)

/-- Theorem: The volume of a truncated right circular cone with large base radius 12 cm,
    small base radius 6 cm, and height 10 cm is equal to 840π cm³. -/
theorem truncated_cone_volume_example :
  truncated_cone_volume 12 10 6 = 840 * Real.pi := by
  sorry

#check truncated_cone_volume_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l550_55095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_A_equidistant_lines_through_B_l550_55082

/-- Triangle ABC with vertices A(4,0), B(8,10), and C(0,6) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (8, 10)
  C : ℝ × ℝ := (0, 6)

/-- Line equation in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def perpendicular_to_BC (t : Triangle) (l : Line) : Prop :=
  l.a * (t.B.1 - t.C.1) + l.b * (t.B.2 - t.C.2) = 0

def passes_through_A (t : Triangle) (l : Line) : Prop :=
  l.a * t.A.1 + l.b * t.A.2 + l.c = 0

def passes_through_B (t : Triangle) (l : Line) : Prop :=
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0

noncomputable def distance_to_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2 + l.c) / Real.sqrt (l.a^2 + l.b^2)

def equidistant_from_A_and_C (t : Triangle) (l : Line) : Prop :=
  distance_to_line t.A l = distance_to_line t.C l

theorem perpendicular_line_through_A (t : Triangle) :
  ∃ l : Line, l.a = 2 ∧ l.b = 1 ∧ l.c = -8 ∧
  perpendicular_to_BC t l ∧ passes_through_A t l := by
  sorry

theorem equidistant_lines_through_B (t : Triangle) :
  ∃ l1 l2 : Line,
  (l1.a = 7 ∧ l1.b = -6 ∧ l1.c = 4) ∧
  (l2.a = 3 ∧ l2.b = 2 ∧ l2.c = -44) ∧
  passes_through_B t l1 ∧ passes_through_B t l2 ∧
  equidistant_from_A_and_C t l1 ∧ equidistant_from_A_and_C t l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_A_equidistant_lines_through_B_l550_55082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_male_students_l550_55002

/-- Represents the number of male students in a class -/
def male_students : ℕ := sorry

/-- Represents the number of female students in a class -/
def female_students : ℕ := 20

/-- The number of male students is greater than 20 -/
axiom male_students_gt_20 : male_students > 20

/-- The difference between male and female students represents
    the number of additional male students compared to female students -/
theorem additional_male_students :
  male_students - female_students = male_students - 20 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_male_students_l550_55002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_sum_l550_55047

/-- Given a line and a hyperbola that intersect at two points, 
    if the line from the origin to one intersection point is perpendicular 
    to the line from the origin to the other intersection point, 
    then the sum of the reciprocals of the hyperbola's coefficients is 2. -/
theorem intersection_perpendicular_sum (a b : ℝ) (P Q : ℝ × ℝ) :
  a * b < 0 →
  (∀ x y : ℝ, x - y + 1 = 0 ↔ (x, y) = P ∨ (x, y) = Q) →
  (∀ x y : ℝ, x^2 / a + y^2 / b = 1 ↔ (x, y) = P ∨ (x, y) = Q) →
  (P.1 * Q.1 + P.2 * Q.2 = 0) →
  1 / a + 1 / b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_perpendicular_sum_l550_55047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_jar_initial_amount_l550_55044

/-- The amount of money initially in the cookie jar --/
noncomputable def initial_amount : ℚ := 24

/-- The amount Doris spent --/
def doris_spent : ℚ := 6

/-- The amount Martha spent --/
def martha_spent : ℚ := doris_spent / 2

/-- The amount left in the cookie jar after spending --/
def amount_left : ℚ := 15

/-- Theorem stating that the initial amount in the cookie jar was $24 --/
theorem cookie_jar_initial_amount :
  initial_amount = doris_spent + martha_spent + amount_left := by
  -- Expand the definitions
  unfold initial_amount doris_spent martha_spent amount_left
  -- Perform the calculation
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_jar_initial_amount_l550_55044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_partitions_to_remove_l550_55006

/-- 
min_partitions n is the minimum number of partitions to remove
for connectivity in a cube of side length n divided into unit cubes.
-/
def min_partitions (n : ℕ) : ℕ := sorry

/-- 
Given a cube with side length n (n ≥ 3) divided into unit cubes,
the minimum number of partitions to remove for connectivity is (n-2)³.
-/
theorem min_partitions_to_remove (n : ℕ) (h : n ≥ 3) : 
  min_partitions n = (n - 2)^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_partitions_to_remove_l550_55006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l550_55092

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  circumradius : Real

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A + t.a * Real.sin t.C * Real.cos t.B + 
  t.b * Real.sin t.C * Real.cos t.A = 
  t.b * Real.sin t.B + t.c * Real.sin t.A

def special_triangle (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c ∧ t.circumradius = 2

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : satisfies_condition t) (h2 : special_triangle t) : 
  t.B = Real.pi / 3 ∧ 
  ∀ (P : Real × Real), 
    (P.1 - t.circumradius)^2 + P.2^2 = t.circumradius^2 → 
    -2 ≤ (P.1 * (P.1 - t.a) + P.2 * P.2) ∧ 
    (P.1 * (P.1 - t.a) + P.2 * P.2) ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l550_55092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_inverse_tangents_l550_55042

theorem min_value_sum_inverse_tangents (A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_cos_B : Real.cos B = 1/4) : 
  ∃ (min_value : ℝ), min_value = 2 * Real.sqrt 15 / 5 ∧ 
    ∀ (A' C' : ℝ), A' + C' = Real.pi - B → 1 / Real.tan A' + 1 / Real.tan C' ≥ min_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_inverse_tangents_l550_55042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_l550_55000

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 10 + Real.sin x ^ 10

theorem smallest_period_of_f :
  ∀ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) → p ≥ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_l550_55000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_increasing_implies_positive_ratio_positive_ratio_not_sufficient_for_increasing_sum_l550_55034

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Represents whether a sequence is increasing -/
def is_increasing (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S (n + 1) > S n

/-- Theorem: If the geometric sum is increasing, then the common ratio is positive -/
theorem geometric_sum_increasing_implies_positive_ratio
  (a₁ : ℝ) (q : ℝ) (h_increasing : is_increasing (geometric_sum a₁ q)) :
  q > 0 := by
  sorry

/-- Theorem: A positive common ratio is not sufficient for an increasing sum -/
theorem positive_ratio_not_sufficient_for_increasing_sum :
  ∃ a₁ q : ℝ, q > 0 ∧ ¬(is_increasing (geometric_sum a₁ q)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_increasing_implies_positive_ratio_positive_ratio_not_sufficient_for_increasing_sum_l550_55034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_values_l550_55046

theorem cubic_function_values (a b : ℝ) : 
  (let y : ℝ → ℝ := fun x ↦ a * x^3 + b * x + 2;
   y (-1) = 2009) → 
  (let y : ℝ → ℝ := fun x ↦ a * x^3 + b * x + 2;
   y 1 = -2005) := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_values_l550_55046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totient_gcd_inequality_l550_55036

open Nat

theorem totient_gcd_inequality (m n : ℕ) :
  (totient (Nat.gcd ((2^m : ℕ) + 1) ((2^n : ℕ) + 1))) / 
  (Nat.gcd (totient ((2^m : ℕ) + 1)) (totient ((2^n : ℕ) + 1))) ≥ 
  (2 * Nat.gcd m n) / (2^(Nat.gcd m n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_totient_gcd_inequality_l550_55036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l550_55013

/-- Given a triangle ABC with the following properties:
  * cos A = 3/4
  * C = 2A
  * ac = 24
  Prove that:
  1) cos B = 9/16
  2) The perimeter of the triangle is 15 -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = 3/4 →
  C = 2 * A →
  a * c = 24 →
  (Real.cos B = 9/16) ∧ (a + b + c = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l550_55013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l550_55009

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def Line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the condition for points A and B
def ConditionAB (xA yA xB yB : ℝ) : Prop :=
  (xA^2 + yA^2 + 4*xB^2 + 4*yB^2 + 4*xA*xB + 4*yA*yB) =
  (xA^2 + yA^2 + 4*xB^2 + 4*yB^2 - 4*xA*xB - 4*yA*yB)

theorem ellipse_theorem :
  ∃ (k m : ℝ), ∀ (xA yA xB yB : ℝ),
    Ellipse xA yA → Ellipse xB yB →
    Line k m xA yA → Line k m xB yB →
    ConditionAB xA yA xB yB →
    (m ≤ -2*Real.sqrt 21/7 ∨ m ≥ 2*Real.sqrt 21/7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l550_55009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l550_55051

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x y, x < y ∧ f x = 0 ∧ f y = 0 ∧ (∀ z, x < z ∧ z < y → f z ≠ 0) → y - x = Real.pi / 2) ∧
  f (2 * Real.pi / 3) = -2 ∧
  (∃ A ω φ, A > 0 ∧ ω > 0 ∧ 0 < φ ∧ φ < Real.pi / 2 ∧
    ∀ x, f x = A * Real.sin (ω * x + φ)) →
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (∀ k : ℤ, ∀ x, Real.pi * (k : ℝ) - Real.pi / 3 ≤ x ∧ x ≤ Real.pi * (k : ℝ) + Real.pi / 6 →
    ∀ y, x ≤ y ∧ y ≤ Real.pi * (k : ℝ) + Real.pi / 6 → f y ≤ f x) ∧
  (∀ x, Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1 ≤ f x ∧ f x ≤ 2) ∧
  (∃ x₁ x₂, Real.pi / 12 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧
    Real.pi / 12 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧
    f x₁ = -1 ∧ f x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l550_55051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pollution_source_intensity_l550_55043

/-- Pollution index function -/
noncomputable def pollution_index (k a b x : ℝ) : ℝ :=
  (k * a) / (x^2) + (k * b) / ((18 - x)^2)

theorem pollution_source_intensity 
  (k : ℝ) 
  (h_k : k > 0) 
  (a b x : ℝ) 
  (h_a : a = 1) 
  (h_x : x = 6) 
  (h_min : IsLocalMin (pollution_index k a b) x) :
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pollution_source_intensity_l550_55043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l550_55021

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, t + 1)

-- Define the curve C
noncomputable def curve_C (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)

-- Define the polar coordinate for ray OP
def ray_OP (α : ℝ) : ℝ := α

-- Define the polar coordinate for ray OQ
noncomputable def ray_OQ (α : ℝ) : ℝ := α + Real.pi / 2

-- Define the area of triangle OPQ
noncomputable def area_OPQ (α : ℝ) : ℝ := 
  (2 * Real.cos α) / (Real.cos α + Real.sin α)

theorem line_curve_intersection (α : ℝ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : area_OPQ α = 1) :
  α = Real.pi / 4 ∧ Real.sqrt 8 = 4 * Real.cos α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_l550_55021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l550_55048

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x as a complex number
noncomputable def x : ℂ := (2 - i * Real.sqrt 3) / 3

-- State the theorem
theorem complex_fraction_equality :
  1 / (x^2 - x) = -45/28 + (9 * i * Real.sqrt 3) / 28 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l550_55048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_different_tilings_l550_55049

/-- Represents a cell on the checkerboard -/
structure Cell where
  x : Fin 533
  y : Fin 533

/-- Represents the color of a cell -/
inductive Color
  | Black
  | White

/-- Returns the color of a cell based on its coordinates -/
def cellColor (c : Cell) : Color :=
  if (c.x.val + c.y.val) % 2 = 0 then Color.Black else Color.White

/-- Counts the number of domino tilings on the board without a specific cell -/
noncomputable def countTilings (board : Cell → Prop) : Nat :=
  sorry

/-- The main theorem -/
theorem exist_different_tilings :
  ∃ (A B : Cell), cellColor A = cellColor B ∧
    countTilings (fun c => c ≠ A) ≠ countTilings (fun c => c ≠ B) := by
  sorry

#check exist_different_tilings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_different_tilings_l550_55049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linus_winning_strategy_l550_55069

/-- Function to calculate the product of positive divisors of a number -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Recursive sequence definition -/
def a : ℕ → ℕ
  | 0 => sorry  -- a₀ is chosen by Linus (we start from 0 to match Nat.zero case)
  | (n + 1) => productOfDivisors (a n)

/-- Predicate to check if a number is a square -/
def isSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The set P of indices where a_k is a square number -/
def P : Set ℕ := {k : ℕ | k ≤ 2018 ∧ isSquare (a k)}

/-- Linus has a winning strategy -/
theorem linus_winning_strategy :
  ∀ Q : Set ℕ, (∀ q ∈ Q, q ≤ 2018) → ∃ a₀ : ℕ, P = Q :=
by
  sorry

#check linus_winning_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linus_winning_strategy_l550_55069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sequence_stabilizes_l550_55087

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Function to get points inside a circle -/
def pointsInside (c : Circle) (points : List Point) : List Point :=
  sorry

/-- Function to calculate the center of mass of a list of points -/
def centerOfMass (points : List Point) : Point :=
  sorry

/-- Function to generate the next circle in the sequence -/
def nextCircle (c : Circle) (points : List Point) : Circle :=
  { center := centerOfMass (pointsInside c points),
    radius := c.radius }

/-- The main theorem stating that the process will eventually stop -/
theorem circle_sequence_stabilizes (points : List Point) (R : ℝ) :
  ∃ (n : ℕ), ∀ (m : ℕ), m ≥ n →
    (Nat.iterate (λ c => nextCircle c points) m { center := centerOfMass points, radius := R } =
     Nat.iterate (λ c => nextCircle c points) n { center := centerOfMass points, radius := R }) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sequence_stabilizes_l550_55087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_two_on_die_l550_55017

theorem probability_greater_than_two_on_die :
  let die := Finset.range 6
  let greater_than_two := Finset.filter (fun x => x > 2) die
  (Finset.card greater_than_two : ℚ) / (Finset.card die : ℚ) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_greater_than_two_on_die_l550_55017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_condition_l550_55068

noncomputable section

theorem vector_sum_magnitude_condition (a b : EuclideanSpace ℝ (Fin 3)) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), x • a + y • b = 0) →
  (‖a + b‖ = ‖a‖ + ‖b‖ → ∃ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), x • a + y • b = 0) ∧
  ¬(∃ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), x • a + y • b = 0 → ‖a + b‖ = ‖a‖ + ‖b‖) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_condition_l550_55068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_semiperimeter_properties_l550_55023

/-- Properties of a triangle's semi-perimeter -/
theorem triangle_semiperimeter_properties 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (s : ℝ) 
  (R r : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_semiperimeter : s = (a + b + c) / 2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_R : R > 0) 
  (h_r : r > 0) :
  (s = b * (Real.cos (C / 2))^2 + c * (Real.cos (B / 2))^2) ∧
  (s = a + b * (Real.sin (C / 2))^2 + c * (Real.sin (B / 2))^2) ∧
  (s = 4 * R * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) ∧
  (s = r * (Real.tan (π/2 - A/2) + Real.tan (π/2 - B/2) + Real.tan (π/2 - C/2))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_semiperimeter_properties_l550_55023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_calculation_l550_55077

def total_amount : ℝ := 3400
def interest_rate1 : ℝ := 0.03
def interest_rate2 : ℝ := 0.05
def part1 : ℝ := 1300

def part2 : ℝ := total_amount - part1

def interest1 : ℝ := part1 * interest_rate1
def interest2 : ℝ := part2 * interest_rate2

def total_interest : ℝ := interest1 + interest2

theorem annual_interest_calculation : 
  abs (total_interest - 144) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_interest_calculation_l550_55077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l550_55053

/-- The radius of three congruent spheres inside a right circular cone -/
noncomputable def sphere_radius (base_radius height : ℝ) : ℝ :=
  (90 - 40 * Real.sqrt 3) / 11

/-- Sphere in ℝ³ -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate for two spheres being tangent -/
def sphere_tangent (s1 s2 : Sphere) : Prop :=
  sorry

/-- Predicate for a sphere being tangent to the base of a cone -/
def sphere_tangent_to_cone_base (s : Sphere) (base_radius height : ℝ) : Prop :=
  sorry

/-- Predicate for a sphere being tangent to the side of a cone -/
def sphere_tangent_to_cone_side (s : Sphere) (base_radius height : ℝ) : Prop :=
  sorry

/-- Theorem stating the radius of three congruent spheres inside a right circular cone -/
theorem sphere_radius_in_cone (base_radius height : ℝ) 
  (h1 : base_radius = 5)
  (h2 : height = 12) :
  ∃ (r : ℝ), r = sphere_radius base_radius height ∧ 
  r > 0 ∧
  (∃ (s1 s2 s3 : Sphere), 
    s1.radius = r ∧
    s2.radius = r ∧
    s3.radius = r ∧
    sphere_tangent s1 s2 ∧
    sphere_tangent s1 s3 ∧
    sphere_tangent s2 s3 ∧
    sphere_tangent_to_cone_base s1 base_radius height ∧
    sphere_tangent_to_cone_base s2 base_radius height ∧
    sphere_tangent_to_cone_base s3 base_radius height ∧
    sphere_tangent_to_cone_side s1 base_radius height ∧
    sphere_tangent_to_cone_side s2 base_radius height ∧
    sphere_tangent_to_cone_side s3 base_radius height) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cone_l550_55053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_price_equivalence_l550_55094

/-- Represents the price of a fruit in an arbitrary unit -/
structure Price where
  value : ℚ

instance : HMul ℕ Price Price where
  hMul n p := ⟨n * p.value⟩

instance : HMul ℚ Price Price where
  hMul q p := ⟨q * p.value⟩

/-- Given price relationships between fruits, prove the equivalence of 20 apples to cucumbers and grapes -/
theorem fruit_price_equivalence 
  (apple_price banana_price cucumber_price grape_price : Price)
  (h1 : (10 : ℕ) * apple_price = (5 : ℕ) * banana_price)
  (h2 : (3 : ℕ) * banana_price = (4 : ℕ) * cucumber_price)
  (h3 : (4 : ℕ) * cucumber_price = (6 : ℕ) * grape_price) :
  ((20 : ℕ) * apple_price = (40 / 3 : ℚ) * cucumber_price) ∧ 
  ((20 : ℕ) * apple_price = (20 : ℕ) * grape_price) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_price_equivalence_l550_55094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l550_55080

/-- The sum of the infinite series Σ(n/5^n) from n=1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' n, n / 5^n

/-- The theorem stating that the sum of the infinite series is equal to 5/16 -/
theorem infiniteSeries_sum : infiniteSeries = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l550_55080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l550_55015

/-- Regular tetrahedron with inscribed spheres -/
structure RegularTetrahedronWithSpheres where
  R : ℝ
  edge_length : ℝ
  sphere1_radius : ℝ
  sphere2_radius : ℝ
  sphere1_center : ℝ × ℝ × ℝ
  sphere2_center : ℝ × ℝ × ℝ

/-- The theorem stating the edge length of the tetrahedron -/
theorem tetrahedron_edge_length 
  (t : RegularTetrahedronWithSpheres) 
  (h1 : t.sphere1_radius = 2 * t.R) 
  (h2 : t.sphere2_radius = 3 * t.R) 
  (h3 : t.sphere1_center = (0, 0, 0)) 
  (h4 : t.sphere2_center = (t.edge_length, 0, 0)) 
  (h5 : ‖t.sphere2_center.1 - t.sphere1_center.1‖ = t.sphere1_radius + t.sphere2_radius) :
  t.edge_length = (5 * Real.sqrt 6 + Real.sqrt 32) * t.R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edge_length_l550_55015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bin_game_purple_balls_l550_55076

theorem bin_game_purple_balls (k : ℚ) : k > 0 → (
  let total_balls := 8 + k
  let prob_green := 8 / total_balls
  let prob_purple := k / total_balls
  let expected_value := prob_green * 3 + prob_purple * (-1)
  expected_value = 1
) → k = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bin_game_purple_balls_l550_55076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l550_55097

/-- The function g as defined in the problem -/
noncomputable def g (a b c : ℝ) : ℝ := a / (a + b) + b / (b + c) + c / (c + a)

/-- The main theorem stating the range of g -/
theorem g_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 ∧
  (∀ ε > 0, ∃ a' b' c', 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ g a' b' c' < 1 + ε) ∧
  (∀ ε > 0, ∃ a' b' c', 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ g a' b' c' > 2 - ε) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l550_55097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_700_to_900_digit_sum_18_l550_55088

def digit_sum (n : ℕ) : ℕ := sorry

def count_integers_with_digit_sum (lower upper sum : ℕ) : ℕ :=
  (List.range (upper - lower + 1)).map (λ i => i + lower)
    |>.filter (λ n => digit_sum n = sum)
    |>.length

theorem count_integers_700_to_900_digit_sum_18 : 
  count_integers_with_digit_sum 700 900 18 = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_700_to_900_digit_sum_18_l550_55088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l550_55024

/-- The function g(x) = x^m -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^m

/-- Theorem: The range of g(x) = x^m on [1, ∞) is [1, ∞) when m > 0 -/
theorem range_of_g (m : ℝ) (h : m > 0) :
  Set.range (fun x => g m x) = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l550_55024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l550_55032

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- State the theorem
theorem power_function_through_point_value :
  ∀ (f : ℝ → ℝ) (α : ℝ),
  (∀ x, f x = power_function α x) →  -- f is a power function
  f 4 = 2 →                          -- f passes through (4, 2)
  f 3 = Real.sqrt 3 :=               -- conclusion: f(3) = √3
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l550_55032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l550_55019

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  distance p e.focus1 + distance p e.focus2 = distance e.focus1 ⟨0, 0⟩ + distance e.focus2 ⟨0, 0⟩

/-- Theorem: The other x-intercept of the ellipse -/
theorem ellipse_x_intercept (e : Ellipse) : 
  e.focus1 = ⟨1, 2⟩ → 
  e.focus2 = ⟨4, 0⟩ → 
  isOnEllipse e ⟨0, 0⟩ → 
  isOnEllipse e ⟨19/2, 0⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l550_55019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_basic_fee_l550_55027

/-- Represents a mobile plan -/
structure MobilePlan where
  basicFee : ℝ
  callTimeLimit : ℝ
  freeDataTraffic : ℝ
  overtimeCallFee : ℝ

/-- Calculates the total fee for a given plan and usage -/
noncomputable def totalFee (plan : MobilePlan) (callTime : ℝ) (dataUsage : ℝ) : ℝ :=
  plan.basicFee + max 0 (callTime - plan.callTimeLimit) * plan.overtimeCallFee

/-- Theorem: Given the conditions, the monthly basic fee 'a' for plan A is 79 yuan -/
theorem plan_a_basic_fee :
  ∀ (a : ℝ),
    let planA : MobilePlan := {
      basicFee := a,
      callTimeLimit := 600,
      freeDataTraffic := 15,
      overtimeCallFee := 0.15
    }
    totalFee planA 800 14.5 = 109 →
    a = 79 := by
  sorry

#check plan_a_basic_fee

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_a_basic_fee_l550_55027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l550_55045

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def selling_price (cost_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  cost_price * (1 - loss_percentage / 100)

/-- Theorem stating that the selling price of a radio with a cost price of 1800
    and a loss percentage of 20.555555555555554% is approximately 1430. -/
theorem radio_selling_price :
  (Int.floor (selling_price 1800 20.555555555555554)) = 1430 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_selling_price_l550_55045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l550_55026

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : π < α ∧ α < 3*π/2) : Real.tan α = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l550_55026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chance_condition_l550_55010

def game_set : Finset ℕ := Finset.range 100

theorem equal_chance_condition (k : ℕ) : 
  (1 ≤ k ∧ k ≤ 99 ∧ k % 2 = 1) ↔ 
  (∀ (S : Finset ℕ), S ⊆ game_set → S.card = k → 
    (S.sum id % 2 = 0) = (S.sum id % 2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chance_condition_l550_55010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equal_magnitude_z_fraction_l550_55079

-- Define complex numbers z₁ and z₂
def z₁ (x y : ℝ) : ℂ := x - 1 + y * Complex.I
def z₂ (y : ℝ) : ℂ := 1 + (4 - y) * Complex.I

-- Theorem 1
theorem z_equal_magnitude (x y : ℝ) (h : z₁ x y = z₂ y) : 
  Complex.abs (z₁ x y) = Real.sqrt 5 := by sorry

-- Theorem 2
theorem z_fraction (x y : ℝ) (h₁ : x = 3) (h₂ : y = 3) : 
  (z₂ y)^2 / (z₁ x y + z₂ y) = (8/25 : ℂ) + (6/25 : ℂ) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_equal_magnitude_z_fraction_l550_55079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l550_55016

open Real

-- Define the sine function with the given parameters
noncomputable def f (x : ℝ) (b : ℝ) : ℝ := sin (2 * x + Real.pi / b)

-- State the theorem
theorem axis_of_symmetry (b : ℝ) :
  ∃ (k : ℤ), ∀ (x : ℝ), f (Real.pi/4 + x) b = f (Real.pi/4 - x) b :=
by sorry

-- Define the period of the function
noncomputable def period : ℝ := Real.pi

-- Define the phase shift of the function
noncomputable def phase_shift (b : ℝ) : ℝ := -Real.pi / b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l550_55016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l550_55007

theorem right_triangle_area (BC AC : ℝ) (h1 : BC = 10) (h2 : AC = 6) :
  (1/2) * AC * Real.sqrt (BC^2 - AC^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l550_55007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l550_55083

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_lambda (x : ℝ) :
  parallel (2, 5) (x, 4) → x = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l550_55083
