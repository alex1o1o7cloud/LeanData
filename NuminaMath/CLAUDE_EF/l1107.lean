import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_selection_probability_l1107_110772

/-- Represents a student in the high school -/
structure Student where
  id : Nat

/-- Represents a sample of students -/
def Sample := List Student

/-- A sampling method is a function that selects a sample from a population -/
def SamplingMethod := (List Student) → Nat → Sample

/-- The probability of a student being selected in a sample -/
noncomputable def probability_of_selection
  (method : SamplingMethod)
  (population : List Student)
  (size : Nat)
  (s : Student) : ℝ :=
sorry

/-- A reasonable sampling method ensures equal probability of selection -/
def IsReasonable (method : SamplingMethod) : Prop :=
  ∀ (population : List Student) (size : Nat) (s : Student),
    s ∈ population →
    (probability_of_selection method population size s =
     1 / (population.length : ℝ))

theorem equal_selection_probability
  (population : List Student)
  (size : Nat)
  (method : SamplingMethod)
  (h_reasonable : IsReasonable method)
  (s : Student)
  (h_in_pop : s ∈ population) :
  probability_of_selection method population size s =
  probability_of_selection method population size s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_selection_probability_l1107_110772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1107_110776

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (Real.sin t.B)^2 = 2 * Real.sin t.A * Real.sin t.C

-- Part I
theorem part_one (t : Triangle) (h : satisfiesConditions t) (h_ab : t.a = t.b) :
  Real.cos t.B = 1/4 := by
  sorry

-- Part II
theorem part_two (t : Triangle) (h : satisfiesConditions t) 
  (h_B : t.B = Real.pi/2) (h_a : t.a = Real.sqrt 2) :
  1/2 * t.a * t.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1107_110776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_universal_truth_l1107_110783

theorem no_universal_truth (p q r : ℝ) 
  (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p > 0) (h5 : p > q * r) :
  ¬(∀ (p q r : ℝ), -p > -q) ∧ 
  ¬(∀ (p q r : ℝ), -p > q) ∧ 
  ¬(∀ (p q r : ℝ), 1 < -q/p) ∧ 
  ¬(∀ (p q r : ℝ), 1 > q/p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_universal_truth_l1107_110783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_spend_before_tax_verify_solution_l1107_110762

/-- Represents the minimum spend before tax in pounds of honey -/
def M : ℝ := 8

/-- The bulk price of honey in dollars per pound -/
def bulk_price : ℝ := 5

/-- The tax rate for honey in dollars per pound -/
def tax_rate : ℝ := 1

/-- The total amount Penny paid in dollars -/
def total_paid : ℝ := 240

/-- The amount by which Penny exceeded the minimum spend in pounds -/
def excess : ℝ := 32

/-- Theorem stating that the minimum spend before tax is 8 pounds of honey -/
theorem minimum_spend_before_tax : M = 8 := by
  -- Unfold the definition of M
  unfold M
  -- This is trivially true by definition
  rfl

/-- Verification of the solution -/
theorem verify_solution : (bulk_price + tax_rate) * (M + excess) = total_paid := by
  -- Substitute the values
  simp [bulk_price, tax_rate, M, excess, total_paid]
  -- Evaluate the expression
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_spend_before_tax_verify_solution_l1107_110762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_fifteen_l1107_110775

theorem cos_product_fifteen : 
  (Real.cos (π / 15)) * (Real.cos (2 * π / 15)) * (Real.cos (3 * π / 15)) * (Real.cos (4 * π / 15)) * 
  (Real.cos (5 * π / 15)) * (Real.cos (6 * π / 15)) * (Real.cos (7 * π / 15)) = 1 / 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_fifteen_l1107_110775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l1107_110751

-- Define the ∇ operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation :
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
  nabla (nabla (nabla a b) c) d = 7/8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l1107_110751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_results_in_option_a_l1107_110732

-- Define the structure of a triangle in a 3x3 grid
structure Triangle where
  row : Fin 3
  col : Fin 3
  shaded : Bool
  deriving Repr, BEq

-- Define the original figure
def original_figure : List Triangle := [
  ⟨0, 1, true⟩,  -- top center
  ⟨1, 0, true⟩,  -- middle left
  ⟨2, 2, true⟩   -- bottom right
]

-- Define the rotated figure (Option A)
def rotated_figure : List Triangle := [
  ⟨0, 2, true⟩,  -- top right
  ⟨0, 0, true⟩,  -- top left
  ⟨2, 2, true⟩   -- bottom right
]

-- Function to rotate a triangle 90 degrees clockwise
def rotate_90_clockwise (t : Triangle) : Triangle :=
  ⟨t.col, 2 - t.row, t.shaded⟩

-- Helper function to check if two lists contain the same elements
def same_elements (l1 l2 : List Triangle) : Prop :=
  ∀ t, t ∈ l1 ↔ t ∈ l2

-- Theorem stating that rotating the original figure results in the rotated figure
theorem rotation_results_in_option_a :
  same_elements (original_figure.map rotate_90_clockwise) rotated_figure :=
by
  -- The proof goes here
  sorry

#eval original_figure.map rotate_90_clockwise
#eval rotated_figure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_results_in_option_a_l1107_110732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunil_investment_l1107_110785

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r / 100) ^ t - P

/-- Calculates the total amount after compound interest -/
noncomputable def total_amount (P : ℝ) (CI : ℝ) : ℝ :=
  P + CI

theorem sunil_investment (P : ℝ) (r : ℝ) (t : ℕ) (CI : ℝ) :
  r = 8 ∧ t = 2 ∧ CI = 2828.80 ∧ compound_interest P r t = CI →
  total_amount P CI = 19828.80 := by
  sorry

#eval Float.toString (19828.80 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunil_investment_l1107_110785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_at_40km_per_hour_minimum_fuel_consumption_l1107_110709

-- Define the fuel consumption function
noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

-- Define the total fuel consumed for a 100 km journey at speed x
noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

-- Theorem for part I
theorem fuel_at_40km_per_hour :
  0 < 40 ∧ 40 ≤ 120 → total_fuel 40 = 17.5 :=
by sorry

-- Theorem for part II
theorem minimum_fuel_consumption :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 120 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ 120 → total_fuel x ≤ total_fuel y) ∧
  x = 80 ∧ total_fuel x = 11.25 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_at_40km_per_hour_minimum_fuel_consumption_l1107_110709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1107_110733

open Real

/-- The angle (in radians) between two vectors -/
noncomputable def angle (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (sqrt (a.1^2 + a.2^2) * sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : sqrt (a.1^2 + a.2^2) = 1)
  (h2 : sqrt (b.1^2 + b.2^2) = 2)
  (h3 : a.1 * b.1 + a.2 * b.2 = 1) :
  angle a b = π / 3 := by
  sorry

#check angle_between_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1107_110733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_platform_approx_39_seconds_l1107_110797

-- Define the given parameters
noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 350
noncomputable def time_to_cross_pole : ℝ := 18

-- Define the speed of the train
noncomputable def train_speed : ℝ := train_length / time_to_cross_pole

-- Define the total distance to cross the platform
noncomputable def total_distance : ℝ := train_length + platform_length

-- Theorem statement
theorem time_to_cross_platform_approx_39_seconds :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  abs ((total_distance / train_speed) - 39) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cross_platform_approx_39_seconds_l1107_110797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_softball_team_ratio_l1107_110738

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ
  total_players : ℕ
  more_women : ℕ
  h1 : women = men + more_women
  h2 : men + women = total_players

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) 
  (h3 : team.more_women = 4) 
  (h4 : team.total_players = 18) : 
  (team.men : ℚ) / team.women = 7 / 11 := by
  sorry

-- Remove the #eval line as it's not necessary and can cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_softball_team_ratio_l1107_110738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1107_110702

noncomputable section

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def line_eq (a b c x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem chord_length 
  (a b c : ℝ) 
  (A B : ℝ × ℝ) 
  (h1 : circle_eq A.1 A.2) 
  (h2 : circle_eq B.1 B.2) 
  (h3 : line_eq a b c A.1 A.2) 
  (h4 : line_eq a b c B.1 B.2) 
  (h5 : A ≠ B) 
  (h6 : dot_product (A.1 - origin.1, A.2 - origin.2) (B.1 - origin.1, B.2 - origin.2) = -1/2) :
  distance A B = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1107_110702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_4_l1107_110708

theorem cos_2alpha_plus_pi_4 (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : π/2 < α ∧ α < π) :
  Real.cos (2*α + π/4) = 17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_4_l1107_110708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_proof_l1107_110754

/-- Represents a number in the octal (base-8) system -/
structure Octal where
  value : ℕ
  isValid : value < 8^64 := by decide

/-- Converts an Octal number to its decimal (base-10) representation -/
def octal_to_decimal (o : Octal) : ℕ := o.value

/-- Converts a decimal (base-10) number to its Octal representation -/
def decimal_to_octal (n : ℕ) : Octal :=
  { value := n, isValid := by sorry }

instance : OfNat Octal n where
  ofNat := decimal_to_octal n

/-- Subtraction operation for Octal numbers -/
def octal_sub (a b : Octal) : Octal :=
  decimal_to_octal (octal_to_decimal a - octal_to_decimal b)

theorem octal_subtraction_proof :
  octal_sub (octal_sub 4325 2377) 122 = 1714 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_subtraction_proof_l1107_110754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1107_110796

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := 2 * x - 1 + Real.log x / Real.log 2

-- State the theorem
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo (1/2 : ℝ) 1, f c = 0 :=
by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1107_110796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cube_sum_l1107_110700

theorem sin_cos_cube_sum (α : ℝ) (a : ℝ) : 
  (Real.sin α)^2 - a * Real.sin α = 0 ∧ (Real.cos α)^2 - a * Real.cos α = 0 → 
  (Real.sin α)^3 + (Real.cos α)^3 = -2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_cube_sum_l1107_110700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_g_at_2_l1107_110799

-- Define the real-valued functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom g_odd : ∀ x : ℝ, g x = -g (-x)

-- Define the relationship between f and g
axiom f_minus_g : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1

-- Theorem to prove
theorem f_plus_g_at_2 : f 2 + g 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_plus_g_at_2_l1107_110799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l1107_110704

/-- Calculates the average speed given distance and time --/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- The journey details --/
noncomputable def journey_distance : ℝ := 210
noncomputable def journey_time : ℝ := 19/3  -- 6 hours and 20 minutes in decimal form

/-- Theorem stating the average speed for the given journey --/
theorem journey_average_speed :
  abs (average_speed journey_distance journey_time - 33.16) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l1107_110704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gggg_is_odd_one_out_l1107_110770

/-- Represents a word as a string -/
def Word := String

/-- Predicate to check if a word is considered "real" -/
def is_real_word (w : Word) : Prop := sorry

/-- The set of words given in the problem -/
def word_set : Finset String := {"ARFA", "BANT", "VOLKODAV", "GGGG", "SOUS"}

/-- Theorem stating that GGGG can be justified as the odd one out -/
theorem gggg_is_odd_one_out :
  ∃ (pattern : String → Prop),
    (∀ w ∈ word_set, w ≠ "GGGG" → pattern w) ∧
    ¬pattern "GGGG" ∧
    ¬is_real_word "GGGG" :=
by
  sorry

#check gggg_is_odd_one_out

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gggg_is_odd_one_out_l1107_110770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_problem_l1107_110768

open Real

theorem proposition_problem : 
  (¬(∀ x : ℝ, (3 : ℝ)^x < (5 : ℝ)^x)) ∧ (∃ x₀ : ℝ, x₀ > 0 ∧ 2 - x₀^2 > 1/x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_problem_l1107_110768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1_equals_3_l1107_110789

def my_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = (1 + a n) / (1 - a n)

theorem a_1_equals_3 (a : ℕ → ℝ) (h1 : my_sequence a) (h2 : a 2017 = 3) : a 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1_equals_3_l1107_110789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l1107_110724

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ d * x + e * y + f = 0) →
  (a / d = b / e)

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 2 * x + (m + 1) * y + 4 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => m * x + 3 * y - 2 = 0

theorem parallel_lines_m_values :
  ∀ m : ℝ, (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) ↔ m = -3 ∨ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_values_l1107_110724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_formula_l1107_110701

noncomputable section

/-- The equation of motion for a point mass -/
def s (t : ℝ) : ℝ := 5 - 3 * t^2

/-- The average velocity during the time interval [1, 1+Δt] -/
def avgVelocity (Δt : ℝ) : ℝ := (s (1 + Δt) - s 1) / Δt

theorem average_velocity_formula (Δt : ℝ) (h : Δt ≠ 0) :
  avgVelocity Δt = -3 * Δt - 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_velocity_formula_l1107_110701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l1107_110752

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_nonzero : α ≠ 0
  β_nonzero : β ≠ 0
  γ_nonzero : γ ≠ 0
  distance_from_origin : 1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4

/-- The centroid of the triangle formed by the intersection points -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.α / 3, plane.β / 3, plane.γ / 3)

theorem centroid_sum (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l1107_110752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_equal_sums_l1107_110759

/-- Sum of digits function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Property that sigma(n) is equal for n, 2n, 3n, ..., n² -/
def has_equal_digit_sums (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → sigma (k * n) = sigma n

/-- Main theorem -/
theorem smallest_three_digit_equal_sums :
  ∃ n : ℕ, n = 999 ∧ n ≥ 100 ∧ n < 1000 ∧ has_equal_digit_sums n ∧
  ∀ m : ℕ, m ≥ 100 → m < 1000 → has_equal_digit_sums m → n ≤ m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_equal_sums_l1107_110759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_triangles_l1107_110718

/-- The area outside the triangles in a circle with an inscribed equilateral triangle -/
theorem area_outside_triangles (R : ℝ) (R_pos : R > 0) : 
  R^2 * (π - Real.sqrt 3) = 
  π * R^2 - 3 * (π * R^2 / 6 - (Real.sqrt 3 / 2) * (R * Real.sqrt 3) * R / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_triangles_l1107_110718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1107_110767

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

def C₂ (x y : ℝ) : Prop := x + y = 4

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Statement to prove
theorem min_distance_between_curves :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), C₁ x₃ y₃ → C₂ x₄ y₄ →
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 ∧
    x₁ = 3/2 ∧ y₁ = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l1107_110767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1107_110725

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 * x - 1)}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (1/4 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1107_110725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1107_110742

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 5

-- Define a, b, and c
noncomputable def a : ℝ := f (-Real.log 5 / Real.log 2)
noncomputable def b : ℝ := f (Real.log 3 / Real.log 2)
def c : ℝ := f (-1)

-- Theorem statement
theorem relationship_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1107_110742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_to_work_l1107_110787

-- Define the parameters
noncomputable def total_time : ℝ := 1  -- Total round trip time in hours
noncomputable def time_to_work : ℝ := 35 / 60  -- Time to work in hours
noncomputable def return_speed : ℝ := 105  -- Return speed in km/h

-- Theorem statement
theorem average_speed_to_work : 
  ∀ (distance : ℝ),
  distance > 0 →
  distance / time_to_work = distance / (total_time - time_to_work) * return_speed →
  distance / time_to_work = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_to_work_l1107_110787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_theorem_l1107_110757

/-- Configuration of circles with specific properties -/
structure CircleConfiguration where
  /-- Centers of larger circles form a square -/
  centers_form_square : Bool
  /-- Radius of larger circles -/
  larger_radius : ℝ
  /-- Each larger circle touches one other larger circle and two smaller circles -/
  touching_property : Bool

/-- Radius of smaller circles in the configuration -/
noncomputable def smaller_circle_radius (config : CircleConfiguration) : ℝ :=
  Real.sqrt 2 - 1

/-- Theorem stating the radius of smaller circles in the given configuration -/
theorem smaller_circle_radius_theorem (config : CircleConfiguration) 
  (h1 : config.centers_form_square = true)
  (h2 : config.larger_radius = 1)
  (h3 : config.touching_property = true) :
  smaller_circle_radius config = Real.sqrt 2 - 1 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_theorem_l1107_110757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_paint_is_200_l1107_110758

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ
  deriving Repr

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℕ := 6 * c.side^2

/-- Represents the initial large cube -/
def largeCube : CubeDimensions := ⟨30⟩

/-- Represents a small cube after cutting -/
def smallCube : CubeDimensions := ⟨10⟩

/-- The amount of paint used for the large cube -/
def initialPaint : ℕ := 100

/-- The number of small cubes after cutting -/
def numSmallCubes : ℕ := 27

/-- Calculates the additional paint needed after cutting -/
def additionalPaintNeeded : ℕ :=
  let totalSmallSurfaceArea := numSmallCubes * surfaceArea smallCube
  let additionalSurfaceArea := totalSmallSurfaceArea - surfaceArea largeCube
  (additionalSurfaceArea / surfaceArea largeCube) * initialPaint

theorem additional_paint_is_200 : additionalPaintNeeded = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_paint_is_200_l1107_110758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_change_is_correct_l1107_110707

-- Define the cost of items
def cheeseburger_cost : ℚ := 365/100
def milkshake_cost : ℚ := 2
def coke_cost : ℚ := 1
def fries_cost : ℚ := 4
def cookie_cost : ℚ := 1/2

-- Define tax and tip rates
def sales_tax_rate : ℚ := 7/100
def tip_rate : ℚ := 15/100

-- Define Toby's initial amount
def toby_initial : ℚ := 15

-- Define a function to round to two decimal places
def round_to_cents (x : ℚ) : ℚ :=
  (x * 100).floor / 100

-- Define the function to calculate Toby's change
noncomputable def toby_change : ℚ :=
  let pre_tax_total := 2 * cheeseburger_cost + milkshake_cost + coke_cost + fries_cost + 3 * cookie_cost
  let sales_tax := round_to_cents (sales_tax_rate * pre_tax_total)
  let tip := round_to_cents (tip_rate * pre_tax_total)
  let final_total := pre_tax_total + sales_tax + tip
  let toby_share := final_total / 2
  round_to_cents (toby_initial - toby_share)

-- Theorem statement
theorem toby_change_is_correct : toby_change = 536/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toby_change_is_correct_l1107_110707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_paper_per_student_l1107_110746

theorem construction_paper_per_student
  (num_students : ℕ)
  (num_glue_bottles : ℕ)
  (dropped_fraction : ℚ)
  (additional_paper : ℕ)
  (final_supplies : ℕ)
  (h1 : num_students = 8)
  (h2 : num_glue_bottles = 6)
  (h3 : dropped_fraction = 1/2)
  (h4 : additional_paper = 5)
  (h5 : final_supplies = 20) :
  ((final_supplies - additional_paper) * 2 - num_glue_bottles) / num_students = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_paper_per_student_l1107_110746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1107_110778

/-- A right triangle with integral sides and perimeter 12 has an area of 6 -/
theorem right_triangle_area (a b c : ℕ) : 
  a + b + c = 12 →  -- perimeter is 12
  a^2 + b^2 = c^2 →  -- right triangle (Pythagorean theorem)
  a * b = 12 →  -- area is 6 (2 * area = a * b = 12)
  (a * b : ℚ) / 2 = 6 := by
    sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1107_110778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1107_110705

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x^2 - 3*x + 5)

-- Define the range of f
def range_f : Set ℝ := Set.range f

-- Theorem statement
theorem range_of_f :
  range_f = { y | y ∈ Set.Icc ((7 - 2*Real.sqrt 15) / 11) ((7 + 2*Real.sqrt 15) / 11) } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1107_110705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1107_110715

open Real

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : (2 * t.c - t.a) * cos t.B - t.b * cos t.A = 0) 
  (h2 : t.A + t.B + t.C = π) 
  (h3 : 0 < t.A ∧ t.A < 2*π/3) : 
  t.B = π/3 ∧ 
  ∃ x, x ∈ Set.Ioo 1 2 ∧ Real.sqrt 3 * sin t.A + sin (t.C - π/6) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1107_110715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_left_turns_opposite_turn_angles_supplementary_l1107_110741

/-- Represents a turn, where the angle is in degrees -/
structure Turn where
  angle : ℚ
  direction : Bool  -- true for left, false for right

/-- Represents the state of the car after turns -/
structure CarState where
  direction : ℚ  -- Angle in degrees relative to initial direction

/-- Applies a turn to the car's current state -/
noncomputable def applyTurn (state : CarState) (turn : Turn) : CarState :=
  { direction := 
      if turn.direction
      then (state.direction + turn.angle) % 360
      else (state.direction - turn.angle) % 360 }

/-- Theorem: Two left turns of 50° and 130° result in opposite direction -/
theorem two_left_turns_opposite (initialState : CarState) :
  let turn1 := { angle := 50, direction := true }
  let turn2 := { angle := 130, direction := true }
  let finalState := applyTurn (applyTurn initialState turn1) turn2
  finalState.direction = (initialState.direction + 180) % 360 := by
  sorry

/-- Theorem: The sum of the two turn angles is 180° -/
theorem turn_angles_supplementary :
  (50 : ℚ) + 130 = 180 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_left_turns_opposite_turn_angles_supplementary_l1107_110741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_eq_18_l1107_110763

theorem sum_of_solutions_eq_18 : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, x^2 - 8*x + 21 = |x - 5| + 4) ∧
  (∀ x ∉ S, x^2 - 8*x + 21 ≠ |x - 5| + 4) ∧
  (S.sum id) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_eq_18_l1107_110763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1107_110734

theorem cosine_identity (α β : ℝ) 
  (h1 : α ∈ Set.Ioo (3*Real.pi/4) Real.pi) 
  (h2 : β ∈ Set.Ioo (3*Real.pi/4) Real.pi) 
  (h3 : Real.cos (α + β) = 4/5) 
  (h4 : Real.sin (α - Real.pi/4) = 12/13) : 
  Real.cos (β + Real.pi/4) = -56/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1107_110734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_sqrt_two_l1107_110743

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 5 else -x^2 + 2 * x + 3

-- Theorem statement
theorem sum_of_solutions_is_sqrt_two :
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ ≥ 0 ∧ f x₁ = 2 ∧ f x₂ = 2 ∧ x₁ + x₂ = Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_sqrt_two_l1107_110743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1107_110714

/-- Calculates the length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
noncomputable def train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + person_speed
  let relative_speed_ms := relative_speed * 1000 / 3600
  relative_speed_ms * passing_time

/-- The length of the train is approximately 119.9904 meters when it passes a person running in the opposite direction. -/
theorem train_length_calculation :
  let train_speed := 65.99424046076315
  let person_speed := 6
  let passing_time := 6
  ∃ ε > 0, |train_length train_speed person_speed passing_time - 119.9904| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1107_110714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_congruence_solutions_l1107_110795

theorem three_digit_congruence_solutions : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun x : ℕ ↦ 100 ≤ x ∧ x < 1000 ∧ (4843 * x + 731) % 29 = 1647 % 29) 
    (Finset.range 1000)).card ∧ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_congruence_solutions_l1107_110795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_k_value_inequality_range_minimum_value_m_l1107_110791

-- Define the function f
noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x + (k-1)*a^(-x) + k^2

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2^(2*x) + 2^(-2*x) - 2*m*(2^x - 2^(-x))

-- Theorem 1
theorem odd_function_k_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ k : ℝ, (∀ x : ℝ, f a k x = -f a k (-x)) → k = 0 := by
  sorry

-- Theorem 2
theorem inequality_range (a : ℝ) (h : a > 1) :
  ∃ t : ℝ, (∀ x : ℝ, f a 0 (x^2 + x) + f a 0 (t - 2*x) > 0) ↔ t > 1/4 := by
  sorry

-- Theorem 3
theorem minimum_value_m (m : ℝ) :
  (∀ x : ℝ, x ≥ 1 → g m x ≥ -1) ∧ (∃ x : ℝ, x ≥ 1 ∧ g m x = -1) → m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_k_value_inequality_range_minimum_value_m_l1107_110791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1107_110798

/-- Predicate to check if three lengths form a triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if R is the circumradius of a triangle with sides a, b, c -/
def IsCircumradius (R a b c : ℝ) : Prop :=
  4 * R^2 * (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c) = 
  (a + b + c)^2 * (a^2 + b^2 + c^2 - 2*a*b - 2*b*c - 2*c*a)

/-- Given a triangle with sides a, b, c and circumradius R, 
    the sum of the reciprocals of the products of pairs of sides 
    is greater than or equal to the reciprocal of the square of the circumradius. -/
theorem triangle_inequality (a b c R : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hR : R > 0)
  (h_triangle : IsTriangle a b c)
  (h_circumradius : IsCircumradius R a b c) : 
  1 / (a * b) + 1 / (b * c) + 1 / (c * a) ≥ 1 / R^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1107_110798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_relation_l1107_110719

theorem angle_sine_relation (B C : Real) (h1 : 0 < B) (h2 : B < π) (h3 : 0 < C) (h4 : C < π) :
  (B > C ↔ Real.sin B > Real.sin C) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_relation_l1107_110719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1107_110711

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle --/
noncomputable def area (t : Triangle) : ℝ := 1/2 * t.b * t.c * Real.sin t.A

theorem triangle_problem (t : Triangle) 
  (h_area : area t = 3 * Real.sqrt 15)
  (h_diff : t.b - t.c = 2)
  (h_cos : Real.cos t.A = -1/4) :
  t.a = 8 ∧ 
  Real.sin t.C = Real.sqrt 15 / 8 ∧ 
  Real.cos (2 * t.A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1107_110711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EG_l1107_110764

noncomputable section

-- Define the square EFGH
def E : ℝ × ℝ := (0, 5)
def F : ℝ × ℝ := (5, 5)
def G : ℝ × ℝ := (5, 0)
def H : ℝ × ℝ := (0, 0)

-- Define N as the midpoint of GH
def N : ℝ × ℝ := ((G.1 + H.1) / 2, (G.2 + H.2) / 2)

-- Define the circles
def circle_N (x y : ℝ) : Prop := (x - N.1)^2 + (y - N.2)^2 = 2.5^2
def circle_E (x y : ℝ) : Prop := (x - E.1)^2 + (y - E.2)^2 = 5^2

-- Define Q as the intersection point
def Q : ℝ × ℝ := (25/7, 10/7)

-- Theorem statement
theorem distance_Q_to_EG :
  circle_N Q.1 Q.2 ∧ circle_E Q.1 Q.2 →
  |5 - Q.2| = 25/7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EG_l1107_110764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_26_l1107_110753

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x + 4) else x * (x - 4)

-- State the theorem
theorem f_sum_equals_26 : f 1 + f (-3) = 26 := by
  -- Evaluate f(1)
  have h1 : f 1 = 5 := by
    simp [f]
    norm_num
  
  -- Evaluate f(-3)
  have h2 : f (-3) = 21 := by
    simp [f]
    norm_num
  
  -- Sum the results
  calc
    f 1 + f (-3) = 5 + 21 := by rw [h1, h2]
    _            = 26     := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_26_l1107_110753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1107_110712

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos x ≤ 1) ↔ (∃ x₀ : ℝ, Real.cos x₀ > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_inequality_l1107_110712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1107_110744

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := -Real.sqrt x

-- State the theorem
theorem inverse_function_theorem :
  (∀ x ≤ -1, f x ≥ 1) ∧
  (∀ y ≥ 1, f (f_inv y) = y) ∧
  (∀ x ≤ -1, f_inv (f x) = x) := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1107_110744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_plan_l1107_110729

/-- Represents the purchase and selling information for dolls --/
structure DollInfo where
  purchase_price : ℕ
  selling_price : ℕ

/-- Represents the store's inventory and constraints --/
structure Store where
  doll_a : DollInfo
  doll_b : DollInfo
  total_dolls : ℕ
  max_a_ratio : ℚ

/-- Calculates the profit for a given purchase plan --/
def calculate_profit (store : Store) (a_count : ℕ) : ℤ :=
  let b_count := store.total_dolls - a_count
  (store.doll_a.selling_price - store.doll_a.purchase_price) * a_count +
  (store.doll_b.selling_price - store.doll_b.purchase_price) * b_count

/-- Checks if a purchase plan satisfies the store's constraints --/
def is_valid_plan (store : Store) (a_count : ℕ) : Prop :=
  a_count ≤ store.total_dolls ∧
  (a_count : ℚ) ≤ store.max_a_ratio * store.total_dolls

/-- Theorem stating that the maximum profit is achieved with the specified purchase plan --/
theorem max_profit_plan (store : Store) :
  store.doll_a.purchase_price = 20 →
  store.doll_a.selling_price = 28 →
  store.doll_b.purchase_price = 15 →
  store.doll_b.selling_price = 20 →
  store.total_dolls = 30 →
  store.max_a_ratio = 1/2 →
  (∀ a_count : ℕ, is_valid_plan store a_count →
    calculate_profit store a_count ≤ calculate_profit store 10) ∧
  calculate_profit store 10 = 180 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_plan_l1107_110729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_theorem_l1107_110727

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A square with side length 2 -/
def Square : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 2 ∧ 0 ≤ p.y ∧ p.y ≤ 2}

theorem seven_points_theorem (points : Finset Point) :
  points.card = 7 → (∀ p ∈ points, p ∈ Square) →
  ∃ p1 p2 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_points_theorem_l1107_110727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_deal_properties_l1107_110774

/-- Represents an investment contract with initial investment p and return x --/
structure Investment where
  p : ℝ
  x : ℝ

/-- The distance between two investments --/
noncomputable def distance (i1 i2 : Investment) : ℝ :=
  Real.sqrt ((i1.x - i2.x)^2 + (i1.p - i2.p)^2)

/-- The profit from two investments --/
def profit (i1 i2 : Investment) : ℝ :=
  i1.x + i2.x - i1.p - i2.p

/-- Theorem stating the minimum distance and profitability of the investment deal --/
theorem investment_deal_properties :
  ∃ (i1 i2 : Investment),
    i1.p > 0 ∧ i2.p > 0 ∧
    3 * i1.x - 4 * i1.p - 30 = 0 ∧
    i2.p^2 - 12 * i2.p + i2.x^2 - 14 * i2.x + 69 = 0 ∧
    (∀ (j1 j2 : Investment),
      j1.p > 0 ∧ j2.p > 0 ∧
      3 * j1.x - 4 * j1.p - 30 = 0 ∧
      j2.p^2 - 12 * j2.p + j2.x^2 - 14 * j2.x + 69 = 0 →
      distance i1 i2 ≤ distance j1 j2) ∧
    distance i1 i2 = 2.6 ∧
    profit i1 i2 = 16.84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_deal_properties_l1107_110774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_M_l1107_110726

/-- A quadratic function f(x) = ax^2 + bx + c where a < b and f(x) ≥ 0 for all real x -/
structure NonNegativeQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a < b
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The expression M = (a + b + c) / (b - a) for a NonNegativeQuadratic -/
noncomputable def M (f : NonNegativeQuadratic) : ℝ :=
  (f.a + f.b + f.c) / (f.b - f.a)

/-- The theorem stating that the minimum value of M for any NonNegativeQuadratic is 3 -/
theorem min_value_of_M :
  ∀ f : NonNegativeQuadratic, M f ≥ 3 ∧ ∃ g : NonNegativeQuadratic, M g = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_M_l1107_110726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_gain_l1107_110748

/-- Calculates the percentage gain from a sale given the original cost, loss percentage, and final selling price -/
noncomputable def percentageGain (originalCost sellPrice : ℝ) (lossPct : ℝ) : ℝ :=
  let intermediatePrice := originalCost * (1 - lossPct / 100)
  ((sellPrice - intermediatePrice) / intermediatePrice) * 100

theorem car_sale_gain :
  percentageGain 52325.58 54000 14 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_gain_l1107_110748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wristwatch_time_to_1pm_l1107_110761

/-- Represents the time discrepancy of a wristwatch in minutes per hour -/
noncomputable def timeDiscrepancyPerHour : ℝ := 5

/-- Represents the number of hours since the wristwatch was set to the correct time -/
noncomputable def hoursSinceCorrectTime : ℝ := 5.5

/-- Calculates the effective time elapsed on the wristwatch given the actual time elapsed -/
noncomputable def effectiveTime (actualTime : ℝ) : ℝ :=
  actualTime * (60 - timeDiscrepancyPerHour) / 60

/-- Theorem stating that it will take 30 minutes for the wristwatch to show 1 PM -/
theorem wristwatch_time_to_1pm :
  30 = (1 - (hoursSinceCorrectTime - effectiveTime hoursSinceCorrectTime)) * 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wristwatch_time_to_1pm_l1107_110761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_circle_l1107_110740

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0

-- Define the parabola
def parabola_equation (x y p : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the directrix of the parabola
noncomputable def directrix (p : ℝ) : ℝ := -p/2

-- Theorem statement
theorem parabola_tangent_to_circle (p : ℝ) :
  (∃ x y : ℝ, circle_equation x y) →
  (∃ x y : ℝ, parabola_equation x y p) →
  (∃ x : ℝ, circle_equation x 0 ∧ x = directrix p) →
  p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_circle_l1107_110740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1107_110790

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem f_increasing_interval (φ : ℝ) 
  (h1 : ∀ x, f x φ ≤ |f (π/4) φ|) 
  (h2 : f (π/2) φ > f π φ) :
  ∃ k : ℤ, ∀ x y, x ∈ Set.Icc (k * π) (k * π + π/4) → 
    y ∈ Set.Icc (k * π) (k * π + π/4) → 
      x < y → f x φ < f y φ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l1107_110790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l1107_110786

/-- Given two lines l₁ and l₂, and a point A, we define a structure to represent the problem setup. -/
structure TriangleSetup where
  /-- Line l₁ has equation 4x - 3y = 2 -/
  l₁ : Set (ℝ × ℝ)
  l₁_eq : l₁ = {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 = 2}

  /-- Point A = (-2, -3) is on l₁ -/
  A : ℝ × ℝ
  A_def : A = (-2, -3)
  A_on_l₁ : A ∈ l₁

  /-- Line l₂ has equation y = 2 -/
  l₂ : Set (ℝ × ℝ)
  l₂_eq : l₂ = {p : ℝ × ℝ | p.2 = 2}

  /-- l₁ and l₂ intersect at B -/
  B : ℝ × ℝ
  B_on_l₁ : B ∈ l₁
  B_on_l₂ : B ∈ l₂

  /-- Line l₃ has positive slope and goes through A -/
  l₃ : Set (ℝ × ℝ)
  l₃_slope_pos : ∃ m : ℝ, m > 0 ∧ l₃ = {p : ℝ × ℝ | p.2 - A.2 = m * (p.1 - A.1)}
  A_on_l₃ : A ∈ l₃

  /-- l₂ and l₃ intersect at C -/
  C : ℝ × ℝ
  C_on_l₂ : C ∈ l₂
  C_on_l₃ : C ∈ l₃

  /-- Area of triangle ABC is 4 -/
  triangle_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 4

/-- The main theorem stating that given the setup, the slope of l₃ is 25/28 -/
theorem slope_of_l₃ (setup : TriangleSetup) : 
  ∃ m : ℝ, m = 25/28 ∧ setup.l₃ = {p : ℝ × ℝ | p.2 - setup.A.2 = m * (p.1 - setup.A.1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l1107_110786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1107_110794

theorem plane_equation (p₁ p₂ p₃ : ℝ × ℝ × ℝ) (h₁ : p₁ = (2, -1, 3)) (h₂ : p₂ = (4, -1, 5)) (h₃ : p₃ = (1, 0, 2)) :
  ∃ (A B C D : ℤ), 
    (∀ (x y z : ℝ), (x, y, z) ∈ {p : ℝ × ℝ × ℝ | p = p₁ ∨ p = p₂ ∨ p = p₃} → A * x + B * y + C * z + D = 0) ∧
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧
    A = 1 ∧ B = 0 ∧ C = -1 ∧ D = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1107_110794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nineteen_switches_eight_lamps_l1107_110781

/-- Represents a switch in the system -/
structure Switch :=
  (id : Nat)

/-- Represents a lamp in the system -/
structure Lamp :=
  (id : Nat)
  (connected_switches : Finset Switch)
  (state : Bool)

/-- The system of switches and lamps -/
structure SwitchLampSystem :=
  (switches : Finset Switch)
  (lamps : Finset Lamp)
  (switch_lamp_connection : Switch → Lamp → Bool)

/-- Axioms for the system -/
axiom total_switches : (s : SwitchLampSystem) → Finset.card s.switches = 70
axiom total_lamps : (s : SwitchLampSystem) → Finset.card s.lamps = 15
axiom lamps_connection : (s : SwitchLampSystem) → ∀ l : Lamp, l ∈ s.lamps → Finset.card l.connected_switches = 35
axiom unique_connections : (s : SwitchLampSystem) → ∀ sw1 sw2 : Switch, sw1 ≠ sw2 → ∃ l : Lamp, s.switch_lamp_connection sw1 l ≠ s.switch_lamp_connection sw2 l
axiom initial_state : (s : SwitchLampSystem) → ∀ l : Lamp, l ∈ s.lamps → l.state = false

/-- Defines the action of pressing a set of switches -/
noncomputable def press_switches (s : SwitchLampSystem) (pressed : Finset Switch) : SwitchLampSystem :=
  sorry

/-- The main theorem to prove -/
theorem exists_nineteen_switches_eight_lamps (s : SwitchLampSystem) : 
  ∃ pressed : Finset Switch, Finset.card pressed = 19 ∧ 
    (Finset.card (Finset.filter (fun l => l.state = true) (press_switches s pressed).lamps) ≥ 8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nineteen_switches_eight_lamps_l1107_110781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_theorem_l1107_110773

/-- Represents a board with hats and trophies -/
structure Board where
  size : ℕ
  hats : Fin size → Bool
  trophies : Fin size → Bool
  empty : Fin size

/-- A valid move on the board -/
inductive Move (n : ℕ) where
  | Left : Fin n → Move n
  | Right : Fin n → Move n

/-- The result of applying a sequence of moves to a board -/
def apply_moves (b : Board) (moves : List (Move b.size)) : Board :=
  sorry

/-- Checks if all hats and trophies have swapped positions -/
def is_swapped (initial : Board) (final : Board) : Prop :=
  sorry

/-- The minimum number of moves required to swap hats and trophies -/
noncomputable def min_moves (b : Board) : ℕ :=
  sorry

/-- Theorem: The minimum number of moves to swap n hats and n trophies is 2n + 1 -/
theorem min_moves_theorem (n : ℕ) (b : Board) (h : b.size = 2 * n + 1) :
  min_moves b = 2 * n + 1 := by
  sorry

#check min_moves_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_theorem_l1107_110773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_extension_l1107_110720

/-- Represents a rectangular bathroom -/
structure Bathroom where
  area : ℕ
  width : ℕ

/-- Calculates the new area of a bathroom after extension -/
def new_area (b : Bathroom) (extension : ℕ) : ℕ :=
  (b.area / b.width) * (b.width + 2 * extension)

/-- Theorem: The new area of the bathroom after extension is 144 sq ft -/
theorem bathroom_extension (b : Bathroom) (h1 : b.area = 96) (h2 : b.width = 8) :
  new_area b 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_extension_l1107_110720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_divisibility_l1107_110717

theorem sum_of_squares_divisibility (a b c d : ℤ) (p : ℕ+) :
  (p : ℤ) = a^2 + b^2 →
  Nat.Prime p →
  (p : ℤ) ∣ (c^2 + d^2) →
  ∃ t s : ℤ, (c^2 + d^2) / (p : ℤ) = t^2 + s^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_divisibility_l1107_110717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_marathon_time_l1107_110750

/-- Given a marathon distance, Jack's finishing time, and the ratio of Jack's speed to Jill's speed,
    calculate Jill's time to finish the marathon. -/
noncomputable def calculate_jill_time (distance : ℝ) (jack_time : ℝ) (speed_ratio : ℝ) : ℝ :=
  distance / (distance / jack_time / speed_ratio)

/-- Theorem stating that under given conditions, Jill's marathon time is approximately 4.3 hours -/
theorem jill_marathon_time :
  let distance : ℝ := 43
  let jack_time : ℝ := 4.5
  let speed_ratio : ℝ := 0.9555555555555555
  let jill_time := calculate_jill_time distance jack_time speed_ratio
  ∃ ε > 0, |jill_time - 4.3| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_marathon_time_l1107_110750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_distance_to_line_l1107_110739

/-- A circle with center in the first quadrant, passing through (2,0), and tangent to both axes -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_in_first_quadrant : 0 < center.1 ∧ 0 < center.2
  passes_through_2_0 : (center.1 - 2)^2 + center.2^2 = radius^2
  tangent_to_axes : center.1 = radius ∧ center.2 = radius

/-- The distance from a point (x,y) to the line Ax + By + C = 0 -/
noncomputable def distance_to_line (x y A B C : ℝ) : ℝ :=
  |A*x + B*y + C| / Real.sqrt (A^2 + B^2)

theorem special_circle_distance_to_line (c : SpecialCircle) :
  distance_to_line c.center.1 c.center.2 2 1 (-11) = Real.sqrt 5 := by
  sorry

#check special_circle_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_distance_to_line_l1107_110739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_number_between_one_and_two_l1107_110755

theorem only_number_between_one_and_two : ∀ x : ℝ, 
  x ∈ ({9/10, 1.8, 2, 1, 11/5} : Set ℝ) ∧ 1 < x ∧ x < 2 → x = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_number_between_one_and_two_l1107_110755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_grid_intersections_l1107_110766

-- Define a rectangle on a grid
structure GridRectangle where
  vertices : Fin 4 → ℝ × ℝ
  is_rectangle : ∀ i j k l : Fin 4, i ≠ j → j ≠ k → k ≠ l → l ≠ i →
    (vertices i).1 + (vertices k).1 = (vertices j).1 + (vertices l).1 ∧
    (vertices i).2 + (vertices k).2 = (vertices j).2 + (vertices l).2
  angle_45 : ∀ i j : Fin 4, i ≠ j →
    (vertices j).1 - (vertices i).1 = (vertices j).2 - (vertices i).2 ∨
    (vertices j).1 - (vertices i).1 = (vertices i).2 - (vertices j).2
  not_on_gridlines : ∀ i : Fin 4, ¬(∃ n : ℤ, (vertices i).1 = n ∨ (vertices i).2 = n)

-- Define the number of grid line intersections for a side
noncomputable def gridIntersections (a b : ℝ × ℝ) : ℕ :=
  (Int.floor b.1 - Int.floor a.1).natAbs + (Int.floor b.2 - Int.floor a.2).natAbs

-- Theorem statement
theorem rectangle_grid_intersections (r : GridRectangle) :
  ¬(∀ i j : Fin 4, i ≠ j → Odd (gridIntersections (r.vertices i) (r.vertices j))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_grid_intersections_l1107_110766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_forest_grove_is_one_fifth_l1107_110722

/-- Represents the commuter rail system between Scottsdale and Sherbourne -/
structure CommuterRail where
  totalLength : ℚ
  roundTripTime : ℚ
  forestGroveToSherbourneTime : ℚ

/-- The fraction of the track's length from Scottsdale to Forest Grove -/
def fractionToForestGrove (rail : CommuterRail) : ℚ :=
  1 - rail.forestGroveToSherbourneTime / (rail.roundTripTime / 2)

/-- Theorem stating that the fraction of the track's length from Scottsdale to Forest Grove is 1/5 -/
theorem fraction_to_forest_grove_is_one_fifth (rail : CommuterRail)
  (h1 : rail.totalLength = 200)
  (h2 : rail.roundTripTime = 5)
  (h3 : rail.forestGroveToSherbourneTime = 2) :
  fractionToForestGrove rail = 1/5 := by
  sorry

#eval fractionToForestGrove { totalLength := 200, roundTripTime := 5, forestGroveToSherbourneTime := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_to_forest_grove_is_one_fifth_l1107_110722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_sum_theorem_l1107_110788

theorem xyz_sum_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2/3 = 25)
  (eq2 : y^2/3 + z^2 = 9)
  (eq3 : z^2 + z*x + x^2 = 16) :
  x*y + 2*y*z + 3*z*x = 24 * Real.sqrt 3 := by
  sorry

#check xyz_sum_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_sum_theorem_l1107_110788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_is_half_l1107_110731

def digits : Finset Nat := {1, 2, 4, 5}

def is_odd (n : Nat) : Bool := n % 2 = 1

def four_digit_number (a b c d : Nat) : Bool :=
  a ∈ digits && b ∈ digits && c ∈ digits && d ∈ digits &&
  a ≠ b && a ≠ c && a ≠ d && b ≠ c && b ≠ d && c ≠ d

def probability_odd : ℚ :=
  let all_numbers := Finset.filter (fun (abcd : Nat × Nat × Nat × Nat) => 
    four_digit_number abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2
  ) (Finset.product digits (Finset.product digits (Finset.product digits digits)))
  let odd_numbers := Finset.filter (fun (abcd : Nat × Nat × Nat × Nat) => 
    four_digit_number abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2 &&
    is_odd abcd.2.2.2
  ) all_numbers
  odd_numbers.card / all_numbers.card

theorem probability_odd_is_half : probability_odd = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_is_half_l1107_110731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l1107_110771

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0 : ℝ)..2, 3 * x^2 + k) = 16 → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l1107_110771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_formula_l1107_110706

-- Define the triangle ABC
structure Triangle where
  a : ℝ -- Length of side AB
  b : ℝ -- Length of side AC
  α : ℝ -- Angle BAC in radians

-- Define the volume of the solid of revolution
noncomputable def volumeOfRevolution (t : Triangle) : ℝ :=
  (Real.pi / 3) * t.a * t.b * (t.a + t.b) * Real.sin t.α * Real.cos (t.α / 2)

-- Theorem statement
theorem volume_of_revolution_formula (t : Triangle) :
  volumeOfRevolution t = (Real.pi / 3) * t.a * t.b * (t.a + t.b) * Real.sin t.α * Real.cos (t.α / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_formula_l1107_110706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1107_110747

theorem triangle_angle_range (a b c : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  b / (b + c) + c / (a + b) ≥ 1 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  0 < A ∧ A ≤ Real.pi/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l1107_110747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_in_3x3_grid_l1107_110703

/-- Represents a 3x3 grid seating arrangement -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions are adjacent in the same row or along a diagonal -/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  ((p1.1.val + 1 = p2.1.val ∧ p1.2.val + 1 = p2.2.val) ∨
   (p2.1.val + 1 = p1.1.val ∧ p2.2.val + 1 = p1.2.val))

/-- The number of possible seating arrangements -/
def totalArrangements : ℕ := Nat.factorial 9

/-- The number of favorable arrangements where two specific students are adjacent -/
def favorableArrangements : ℕ := 13 * 2 * Nat.factorial 7

/-- The probability of two specific students being adjacent in a 3x3 grid -/
noncomputable def adjacentProbability : ℚ := 
  (favorableArrangements : ℚ) / (totalArrangements : ℚ)

theorem adjacent_probability_in_3x3_grid :
  adjacentProbability = 13 / 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_in_3x3_grid_l1107_110703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_6_equals_1_l1107_110710

def G : ℕ → ℤ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 2) => 3 * G (n + 1) - 2

theorem G_6_equals_1 : G 6 = 1 := by
  -- Proof steps would go here
  sorry

#eval G 6  -- Optional: to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_6_equals_1_l1107_110710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l1107_110723

/-- The perimeter of the rectangular garden -/
noncomputable def perimeter : ℝ := 36

/-- The length of the rectangular garden -/
noncomputable def length (l : ℝ) : ℝ := l

/-- The width of the rectangular garden -/
noncomputable def width (l : ℝ) : ℝ := perimeter / 2 - l

/-- The area of the rectangular garden -/
noncomputable def area (l : ℝ) : ℝ := length l * width l

/-- The maximum area of the rectangular garden -/
noncomputable def max_area : ℝ := 81

theorem rectangle_max_area :
  ∃ l : ℝ, 0 < l ∧ l < perimeter / 2 ∧ 
  (∀ x : ℝ, 0 < x → x < perimeter / 2 → area x ≤ area l) ∧
  area l = max_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_l1107_110723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1107_110769

/-- The area of the shaded region in a geometric setup with two squares and a triangle -/
theorem shaded_area_calculation (large_square_side small_square_side : ℝ) 
  (h1 : large_square_side = 12)
  (h2 : small_square_side = 5)
  (h3 : small_square_side > 0)
  (h4 : large_square_side > small_square_side) : 
  (small_square_side^2 - (1/2) * ((small_square_side^2) / (large_square_side + small_square_side)) * small_square_side) = 725 / 34 := by
  sorry

#eval (725 : ℚ) / 34

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1107_110769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ABCD_is_rhombus_l1107_110737

def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (3, 4)
def D : ℝ × ℝ := (-1, 6)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def vector_DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem quadrilateral_ABCD_is_rhombus :
  (vector_AB = vector_DC) ∧ (distance A B = distance A D) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_ABCD_is_rhombus_l1107_110737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_in_X_l1107_110749

def I : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2020}

def W : Set ℕ := {n | n ∈ I ∧ ∃ a b, a ∈ I ∧ b ∈ I ∧ n = (a + b) + a * b}

def Y : Set ℕ := {n | n ∈ I ∧ ∃ a b, a ∈ I ∧ b ∈ I ∧ n = (a + b) * a * b}

def X : Set ℕ := W ∩ Y

theorem sum_of_max_and_min_in_X :
  ∃ min max : ℕ, min ∈ X ∧ max ∈ X ∧
  (∀ x ∈ X, min ≤ x) ∧ (∀ x ∈ X, x ≤ max) ∧
  min + max = 2020 := by
  sorry

#check sum_of_max_and_min_in_X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_in_X_l1107_110749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_l1107_110735

/-- The number of team members in the relay team -/
def team_size : ℕ := 5

/-- The number of fixed positions in the relay team -/
def fixed_positions : ℕ := 2

/-- The number of different orders for the relay team -/
def relay_orders : ℕ := 6

/-- Theorem stating that the number of different orders for the relay team is correct -/
theorem relay_team_orders :
  Nat.factorial (team_size - fixed_positions) = relay_orders := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_team_orders_l1107_110735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1107_110713

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + 2

theorem f_properties :
  let f := f
  -- Smallest positive period
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
  -- Monotonically increasing interval
  (∀ k : ℤ, ∀ x y : ℝ, -3 * Real.pi / 8 + k * Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi / 8 + k * Real.pi → f x < f y) ∧
  -- Maximum value on [0, π/2]
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 2 + Real.sqrt 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2 + Real.sqrt 2) ∧
  -- Minimum value on [0, π/2]
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 1 ≤ f x) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1107_110713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1107_110765

noncomputable def vector_projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1^2 + w.2^2
  (dot_product / magnitude_squared * w.1, dot_product / magnitude_squared * w.2)

theorem projection_problem (u : ℝ × ℝ) :
  vector_projection (2, 5) u = (1, 5/2) →
  vector_projection (3, 2) u = (800/725, 2000/725) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l1107_110765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_dilution_problem_l1107_110745

/-- Calculates the sugar concentration after dilution -/
noncomputable def sugar_concentration_after_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) : ℝ :=
  (initial_volume * initial_concentration) / (initial_volume + added_water)

/-- Theorem: Adding 1 liter of water to 3 liters of 40% sugar solution results in 30% concentration -/
theorem sugar_dilution_problem :
  sugar_concentration_after_dilution 3 0.4 1 = 0.3 := by
  -- Unfold the definition of sugar_concentration_after_dilution
  unfold sugar_concentration_after_dilution
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num

-- We can't use #eval for noncomputable functions, so we'll use #check instead
#check sugar_concentration_after_dilution 3 0.4 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_dilution_problem_l1107_110745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1107_110782

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  center : Point

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Distance between two points -/
noncomputable def dist (A B : Point) : ℝ := sorry

/-- Checks if a point is on the asymptote of a hyperbola -/
def is_on_asymptote (h : Hyperbola) (P : Point) : Prop := sorry

/-- Theorem: Given a hyperbola with specific properties, its asymptotes have the equation y = ± 2√6 x -/
theorem hyperbola_asymptotes 
  (h : Hyperbola) 
  (P : Point) 
  (F₁ F₂ : Point) 
  (h_positive : h.a > 0 ∧ h.b > 0) 
  (h_on_hyperbola : (P.x - h.center.x)^2 / h.a^2 - (P.y - h.center.y)^2 / h.b^2 = 1) 
  (h_foci : F₁.x < F₂.x) 
  (h_right_angle : angle F₁ P F₂ = 90) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ dist P F₁ = 4*k ∧ dist P F₂ = 3*k ∧ dist F₁ F₂ = 5*k) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 6 ∧ 
  (∀ (x y : ℝ), (y = m*x ∨ y = -m*x) ↔ is_on_asymptote h (Point.mk x y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1107_110782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_ratio_l1107_110777

theorem cosine_sum_ratio (A B C : ℝ) (h : Real.cos A + Real.cos B + Real.cos C = 0) :
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C)) / (Real.cos A * Real.cos B * Real.cos C) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_ratio_l1107_110777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l1107_110721

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  3 * x^2 - y^2 + 6 * x - 4 * y + 8 = 0

-- Define the focus coordinates
noncomputable def focus : ℝ × ℝ := (-1, -2 + 2 * Real.sqrt 3 / 3)

-- Theorem statement
theorem focus_of_hyperbola :
  let (h, k) := focus
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), hyperbola_eq x y ↔ 
      ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1) ∧
    a^2 + b^2 = (2 * Real.sqrt 3 / 3)^2 := by
  sorry

#check focus_of_hyperbola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l1107_110721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l1107_110779

-- Define the sequence and its partial sums
def a : ℕ → ℝ := sorry
def S : ℕ → ℝ := sorry

-- Define the property of the sequence
axiom seq_property : ∀ n > 1, 2 * a n = a (n + 1) + a (n - 1)

-- Define the given condition about partial sums
axiom sum_condition : S 3 < S 5 ∧ S 5 < S 4

-- Define the property we want to prove
def property_to_prove (n : ℕ) : Prop := n > 1 ∧ S (n - 1) * S n < 0

-- State the theorem
theorem smallest_n_with_property :
  ∃! n : ℕ, property_to_prove n ∧ ∀ m : ℕ, m < n → ¬property_to_prove m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_property_l1107_110779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1107_110760

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := ((i - 1)^2 + 4) / (i + 1)

theorem imaginary_part_of_z : Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l1107_110760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1107_110793

noncomputable section

def g (x : ℝ) : ℝ := x - 1

def f : ℝ → ℝ := sorry

axiom f_recurrence (x : ℝ) : f (x + 1) = -2 * f x - 1

axiom f_definition (x : ℝ) (h : x ∈ Set.Ioo 0 1) : f x = x^2 - x

theorem min_value_theorem :
  ∃ (min : ℝ), min = 49/128 ∧
  ∀ (x₁ x₂ : ℝ) (h : x₁ ∈ Set.Ioo 1 2),
    (x₁ - x₂)^2 + (f x₁ - g x₂)^2 ≥ min :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1107_110793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_substitution_l1107_110730

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 7
def equation2 (x y : ℝ) : Prop := 3 * x - 4 * y = 3

-- Define the substitution options
def optionA (x y : ℝ) : Prop := x = (7 + y) / 2
def optionB (x y : ℝ) : Prop := y = 2 * x - 7
def optionC (x y : ℝ) : Prop := x = (3 + 4 * y) / 3
def optionD (x y : ℝ) : Prop := y = (3 * x - 3) / 4

-- Theorem stating that option B is the simplest substitution
theorem simplest_substitution (x y : ℝ) :
  equation1 x y ∧ equation2 x y →
  ∀ (f : ℝ → ℝ → Prop), f ∈ [optionA, optionB, optionC, optionD] →
    (∀ (a b : ℝ), f a b ↔ optionB a b) ∨
    (∃ (a b : ℝ), f a b ∧ ¬(optionB a b)) ∨
    (∃ (a b : ℝ), optionB a b ∧ ¬(f a b)) :=
by
  sorry

#check simplest_substitution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_substitution_l1107_110730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1107_110736

theorem fraction_simplification (x k a : ℝ) (h : x^2 + k*a^2 > 0) :
  (Real.sqrt (x^2 + k*a^2) - (k*a^2 - x^2) / Real.sqrt (x^2 + k*a^2)) / (x^2 + k*a^2) = 
  2*x^2 / (x^2 + k*a^2)^(3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1107_110736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_pi_over_4_equals_1_l1107_110780

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12) + 1 / 2

theorem g_pi_over_4_equals_1 : g (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_pi_over_4_equals_1_l1107_110780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_minimum_distance_l1107_110756

/-- The circle ⊙C in the Cartesian plane -/
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + (y - Real.sqrt 3)^2 = 3

/-- The line l in the Cartesian plane -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + (1/2) * t, (Real.sqrt 3 / 2) * t)

/-- The center of circle ⊙C -/
noncomputable def center_C : ℝ × ℝ := (0, Real.sqrt 3)

/-- The distance between two points in the Cartesian plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_equation_and_minimum_distance :
  (∀ θ : ℝ, (2 * Real.sqrt 3 * Real.sin θ)^2 = x^2 + y^2 → circle_C x y) ∧
  (∀ t : ℝ, distance (line_l t) center_C ≥ distance (3, 0) center_C) := by
  sorry

#check circle_equation_and_minimum_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_and_minimum_distance_l1107_110756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1107_110792

theorem remainder_theorem (m n : ℕ) 
  (hm : m % 80 = 49)
  (hn : n % 120 = 103) :
  (3 * m - 4 * n) % 240 = 215 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1107_110792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1107_110728

theorem lambda_range (l : ℝ) : 
  (∀ a b : ℝ, a^2 + 8*b^2 ≥ l*b*(a + b)) → -8 ≤ l ∧ l ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1107_110728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1107_110716

/-- Given a function f(x) = sin(ωx + φ), prove that under certain conditions, 
    the minimum value of ω is 5/2. -/
theorem min_omega_value (ω φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2) : 
  (∀ x, Real.sin (ω * x + φ) = -1/2 → x = 0) →
  (∀ x, Real.sin (ω * (x - π/3) + φ) = -Real.sin (ω * (-x - π/3) + φ)) →
  ω ≥ 5/2 ∧ ∃ (ω₀ : ℝ), ω₀ = 5/2 ∧ 
    (∀ x, Real.sin (ω₀ * x + φ) = -1/2 → x = 0) ∧
    (∀ x, Real.sin (ω₀ * (x - π/3) + φ) = -Real.sin (ω₀ * (-x - π/3) + φ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1107_110716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_range_a_l1107_110784

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x + 1
  else -x^2 + (2*a + 1)*x - 4*a + 2

theorem decreasing_f_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Icc (1/3) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_range_a_l1107_110784
