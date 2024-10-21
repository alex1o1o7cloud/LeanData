import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_concentric_circles_l381_38109

/-- The area of a circle with radius r -/
noncomputable def circleArea (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of a ring formed by two concentric circles with radii r₁ and r₂ -/
noncomputable def ringArea (r₁ r₂ : ℝ) : ℝ := circleArea r₁ - circleArea r₂

theorem ring_area_concentric_circles :
  ringArea 12 7 = 95 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_concentric_circles_l381_38109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l381_38190

/-- The area of a trapezoid given its side lengths. -/
noncomputable def trapezoidArea (a b c d : ℝ) : ℝ :=
  (a + b) / (4 * (a - b)) * Real.sqrt ((a + c + d - b) * (a + d - b - c) * (a + c - b - d) * (b + c + d - a))

/-- Theorem: The area of a trapezoid with parallel sides a and b (a > b) and non-parallel sides c and d
    is equal to (a+b)/(4(a-b)) * √((a+c+d-b)(a+d-b-c)(a+c-b-d)(b+c+d-a)). -/
theorem trapezoid_area_formula (a b c d : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) (h5 : d > 0) :
  trapezoidArea a b c d = (a + b) / (4 * (a - b)) * Real.sqrt ((a + c + d - b) * (a + d - b - c) * (a + c - b - d) * (b + c + d - a)) := by
  sorry

#check trapezoid_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_l381_38190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_amount_invariant_l381_38175

/-- Represents a chemical species in a reaction -/
inductive Species
| Solid : String → Species
| Gas : String → Species

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Species
  products : List Species

/-- Represents the rate of a chemical reaction -/
def ReactionRate := ℝ

/-- Represents the concentration of a species -/
def Concentration := Species → ℝ

/-- Function to determine if a species is a solid -/
def isSolid : Species → Bool
  | Species.Solid _ => true
  | Species.Gas _ => false

/-- Function to determine if a species is a gas -/
def isGas : Species → Bool
  | Species.Gas _ => true
  | Species.Solid _ => false

/-- Function representing the relationship between concentration and reaction rate -/
noncomputable def f : Concentration → ReactionRate := sorry

/-- Theorem stating that increasing the amount of a solid reactant does not change the reaction rate -/
theorem solid_amount_invariant (r : Reaction) (s : Species) (rate : ReactionRate) 
    (c : Concentration)
    (h1 : s ∈ r.reactants) 
    (h2 : isSolid s = true) 
    (h3 : rate = f c) : 
  ∀ Δ : ℝ, Δ > 0 → rate = f (λ g ↦ if isGas g then c g else c g + Δ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_amount_invariant_l381_38175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l381_38167

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing on a set S if
    for all x, y in S, x ≤ y implies f(x) ≤ f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem odd_function_properties
    (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_mono : MonoIncreasing f (Set.Ici 0)) :
    (∀ x, |f x| = |f (-x)|) ∧
    (MonoIncreasing (fun x ↦ f x * f (-x)) (Set.Iic 0)) := by
  sorry

#check odd_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l381_38167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l381_38126

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | (2 : ℝ)^x - (5 : ℝ)^x < 0}
def B : Set ℝ := {x | ∃ y, y = log (x^2 - x - 2)}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Iic (-1) ∪ Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l381_38126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_proof_l381_38187

theorem city_distance_proof (S : ℕ) : 
  (∀ x : ℕ, x ≤ S → Nat.gcd x (S - x) ∈ ({1, 3, 13} : Set ℕ)) →
  (∀ T : ℕ, T < S → ∃ y : ℕ, y ≤ T ∧ Nat.gcd y (T - y) ∉ ({1, 3, 13} : Set ℕ)) →
  S = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_distance_proof_l381_38187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_twelve_exists_l381_38160

theorem sum_twelve_exists (S : Finset ℕ) : 
  S ⊆ Finset.range 12 → S.card = 7 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_twelve_exists_l381_38160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l381_38179

/-- Given a cube where one side is a square with perimeter 52 cm, 
    prove that its surface area is 1014 cm². -/
theorem cube_surface_area (perimeter : ℝ) (h : perimeter = 52) : 
  (6 * (perimeter / 4)^2) = 1014 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l381_38179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l381_38180

noncomputable section

open Real

/-- Given a function f(x) = A * sin(ω * x + π/6) with A > 0 and ω > 0,
    its smallest positive period is 6π, and f(2π) = 2.
    We define g(x) = f(x) + 2. -/
def f (A ω : ℝ) (x : ℝ) : ℝ := A * sin (ω * x + π / 6)

def g (A ω : ℝ) (x : ℝ) : ℝ := f A ω x + 2

/-- The theorem states that under the given conditions,
    f(x) = 4 * sin(x/3 + π/6) and the maximum value of g(x) is 6. -/
theorem function_properties (A ω : ℝ) (h₁ : A > 0) (h₂ : ω > 0)
    (h₃ : ∀ x, f A ω (x + 6 * π) = f A ω x)
    (h₄ : ∀ T, T > 0 → (∀ x, f A ω (x + T) = f A ω x) → T ≥ 6 * π)
    (h₅ : f A ω (2 * π) = 2) :
    (∀ x, f A ω x = 4 * sin (x / 3 + π / 6)) ∧
    (∀ x, g A ω x ≤ 6) ∧
    (∃ x, g A ω x = 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l381_38180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_virus_count_after_5_hours_l381_38111

/-- Represents the growth rate of the virus -/
noncomputable def k : ℝ := Real.log 4

/-- The growth function of the virus -/
noncomputable def virus_growth (t : ℝ) : ℝ := Real.exp (k * t)

/-- The virus doubles every 0.5 hours -/
axiom doubles_in_half_hour : virus_growth 0.5 = 2 * virus_growth 0

/-- The number of viruses after 5 hours -/
def viruses_after_5_hours : ℕ := 1024

theorem virus_count_after_5_hours : 
  ⌊virus_growth 5⌋ = viruses_after_5_hours := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_virus_count_after_5_hours_l381_38111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_winning_probability_l381_38127

def num_questions : ℕ := 5
def num_choices : ℕ := 3
def min_correct : ℕ := 3

def probability_correct_answer : ℚ := 1 / num_choices

noncomputable def probability_winning : ℚ :=
  (Finset.range (num_questions - min_correct + 1)).sum (λ k =>
    (Nat.choose num_questions (num_questions - k)) *
    (probability_correct_answer ^ (num_questions - k : ℕ)) *
    ((1 - probability_correct_answer) ^ k))

theorem quiz_winning_probability :
  probability_winning = 17 / 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_winning_probability_l381_38127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_of_g_l381_38186

-- Define the original function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x) / x

-- Define the new function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x = 0 then 3 else f x

-- State the theorem
theorem continuity_of_g :
  Continuous g ∧ (∀ x ≠ 0, g x = f x) ∧ g 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_of_g_l381_38186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_neg_two_l381_38128

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 / (x^2 + 2) - 1

-- Define the interval
def I : Set ℝ := Set.Icc (-2023) 2023

-- State the theorem
theorem max_min_sum_equals_neg_two :
  ∃ (M m : ℝ), (∀ x ∈ I, f x ≤ M) ∧ 
                (∀ x ∈ I, m ≤ f x) ∧ 
                (∃ x₁ x₂, x₁ ∈ I ∧ x₂ ∈ I ∧ f x₁ = M ∧ f x₂ = m) ∧ 
                (M + m = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_neg_two_l381_38128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_complex_sum_l381_38164

theorem min_value_of_complex_sum (a b c : ℤ) (ω : ℂ) : 
  a * b * c = 60 → 
  ω ≠ 1 → 
  ω^3 = 1 → 
  ∃ (x y z : ℤ), x * y * z = 60 ∧ 
    ∀ (p q r : ℤ), p * q * r = 60 → 
      Complex.abs (x + y*ω + z*ω^2) ≤ Complex.abs (p + q*ω + r*ω^2) ∧
      Complex.abs (x + y*ω + z*ω^2) = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_complex_sum_l381_38164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l381_38170

theorem sum_remainder (a b c : ℕ) 
  (ha : a % 24 = 10)
  (hb : b % 24 = 4)
  (hc : c % 24 = 12) :
  (a + b + c) % 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l381_38170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l381_38139

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem absolute_difference_of_solution (x y : ℝ) 
  (eq1 : (floor x : ℝ) + frac y = (3.7 : ℝ))
  (eq2 : frac x + (floor y : ℝ) = (8.2 : ℝ)) : 
  |x - y| = (5.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l381_38139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l381_38195

-- Define the circle
def circle_center : ℝ × ℝ := (1, 0)
def circle_radius : ℝ := 1

-- Define the polar coordinates
def ρ : ℝ → ℝ := sorry
def θ : ℝ → ℝ := sorry

-- Define the conversion between Cartesian and polar coordinates
axiom polar_to_cartesian :
  ∀ t, (ρ t * Real.cos (θ t), ρ t * Real.sin (θ t)) = (circle_center.1 + circle_radius * Real.cos t, circle_center.2 + circle_radius * Real.sin t)

-- Theorem: The polar equation of the circle is ρ = 2 cos θ
theorem circle_polar_equation : ∀ t, ρ t = 2 * Real.cos (θ t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_equation_l381_38195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_have_circumcenter_l381_38192

/-- A quadrilateral type --/
inductive QuadrilateralType
  | RegularQuad
  | Rectangle3to1
  | NonSquareKite
  | ElongatedSquare
  | RightKite

/-- Whether a quadrilateral type has a circumcenter --/
def hasCircumcenter : QuadrilateralType → Bool
  | QuadrilateralType.RegularQuad => true
  | QuadrilateralType.Rectangle3to1 => true
  | QuadrilateralType.NonSquareKite => false
  | QuadrilateralType.ElongatedSquare => true
  | QuadrilateralType.RightKite => true

/-- The list of all quadrilateral types --/
def allTypes : List QuadrilateralType :=
  [QuadrilateralType.RegularQuad, QuadrilateralType.Rectangle3to1,
   QuadrilateralType.NonSquareKite, QuadrilateralType.ElongatedSquare,
   QuadrilateralType.RightKite]

theorem exactly_four_have_circumcenter :
  (allTypes.filter hasCircumcenter).length = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_have_circumcenter_l381_38192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recess_break_duration_l381_38125

/-- Given a total outside time of 80 minutes, which includes two equal recess breaks,
    a 30-minute lunch, and a 20-minute recess break, prove that the duration of each
    of the first two recess breaks is 15 minutes. -/
theorem recess_break_duration :
  let total_outside_time : ℕ := 80
  let lunch_duration : ℕ := 30
  let third_recess_duration : ℕ := 20
  let first_two_recess_duration : ℕ → ℕ := λ x ↦ 2 * x
  ∀ x : ℕ, first_two_recess_duration x + lunch_duration + third_recess_duration = total_outside_time →
    x = 15 :=
by
  intro x h
  -- Proof steps would go here
  sorry

#check recess_break_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recess_break_duration_l381_38125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_star_48_eq_3_implies_x_eq_60_l381_38122

noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

theorem x_star_48_eq_3_implies_x_eq_60 :
  ∀ x : ℝ, star x 48 = 3 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_star_48_eq_3_implies_x_eq_60_l381_38122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l381_38191

-- Define the constants
noncomputable def a : ℝ := 2^(1.2 : ℝ)
noncomputable def b : ℝ := (1/2)^(-(0.8 : ℝ))
noncomputable def c : ℝ := 2 * Real.log 2 / Real.log 5

-- State the theorem
theorem abc_order : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l381_38191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bahs_equal_to_yahs_l381_38112

/-- Conversion rate between bahs and rahs -/
def bah_to_rah : ℚ := 30 / 18

/-- Conversion rate between rahs and yahs -/
def rah_to_yah : ℚ := 20 / 12

/-- The number of yahs we want to convert -/
def target_yahs : ℕ := 1200

/-- Theorem stating that 1200 yahs are equal to 432 bahs -/
theorem bahs_equal_to_yahs : 
  ⌊(target_yahs : ℚ) / (bah_to_rah * rah_to_yah)⌋ = 432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bahs_equal_to_yahs_l381_38112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_inclination_angle_range_l381_38152

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Theorem for the tangent line with minimum slope
theorem min_slope_tangent : 
  ∃ (x₀ y₀ : ℝ), f x₀ = y₀ ∧ 
  (∀ (x y : ℝ), f x = y → f' x ≥ f' x₀) ∧
  (3 * x₀ + 3 * y₀ - 11 = 0) := by
  sorry

-- Theorem for the range of inclination angles
theorem inclination_angle_range (α : ℝ) :
  (∃ (x y : ℝ), f x = y ∧ α = Real.arctan (f' x)) →
  (0 ≤ α ∧ α < Real.pi / 2) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_tangent_inclination_angle_range_l381_38152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_b_fill_time_l381_38177

/-- Proves that tap B takes 20 minutes to fill a third of the bucket given the problem conditions -/
theorem tap_b_fill_time (tap_a_rate : ℝ) (bucket_volume : ℝ) (total_fill_time : ℝ) : ℝ :=
  let tap_b_fill_time := 20
  let tap_b_rate := (bucket_volume / 3) / tap_b_fill_time
  let tap_a_volume := tap_a_rate * total_fill_time
  let tap_b_volume := tap_b_rate * total_fill_time
  have h1 : tap_a_rate = 3 := by sorry
  have h2 : bucket_volume = 36 := by sorry
  have h3 : total_fill_time = 10 := by sorry
  have h4 : tap_a_volume + tap_b_volume = bucket_volume := by sorry
  tap_b_fill_time

#check tap_b_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_b_fill_time_l381_38177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l381_38155

-- Define the functions
noncomputable def f (x : ℝ) := Real.sqrt (-x^2 + 7*x - 12)
noncomputable def g (a x : ℝ) := a / (x^2 + 1)

-- Define the domain A and range B
def A : Set ℝ := {x | -x^2 + 7*x - 12 ≥ 0}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = g a x}

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : A ⊆ B a) : 
  a ∈ Set.Icc 4 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l381_38155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_attendance_l381_38156

/-- Prom attendance theorem -/
theorem prom_attendance
  (total_students : ℕ)
  (solo_students : ℕ)
  (chaperones : ℕ)
  (h1 : total_students = 123)
  (h2 : solo_students = 3)
  (h3 : chaperones = 7) :
  (total_students - solo_students) / 2 = 60 ∧
  total_students + chaperones = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_attendance_l381_38156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l381_38121

theorem expansion_terms_count (a b : ℝ) : 
  ∃ (c₁ c₂ c₃ c₄ c₅ : ℝ), 
    ((a + 3*b)^2 * (a - 3*b)^2)^2 = c₁*a^8 + c₂*a^6*b^2 + c₃*a^4*b^4 + c₄*a^2*b^6 + c₅*b^8 ∧
    (c₁ ≠ 0 ∧ c₂ ≠ 0 ∧ c₃ ≠ 0 ∧ c₄ ≠ 0 ∧ c₅ ≠ 0) ∧
    ∀ (d : ℝ) (i j : ℕ), i + j = 8 → 
      ((a + 3*b)^2 * (a - 3*b)^2)^2 ≠ 
        c₁*a^8 + c₂*a^6*b^2 + c₃*a^4*b^4 + c₄*a^2*b^6 + c₅*b^8 + d*a^i*b^j := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_terms_count_l381_38121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_degrees_to_radians_l381_38106

/-- Conversion factor from degrees to radians -/
noncomputable def degree_to_radian : ℝ := Real.pi / 180

/-- Converts an angle from degrees to radians -/
noncomputable def to_radians (degrees : ℝ) : ℝ := degrees * degree_to_radian

theorem negative_150_degrees_to_radians :
  to_radians (-150) = -5 * Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_150_degrees_to_radians_l381_38106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_2pi_l381_38105

-- Define the function f(x) = sin x + sec x
noncomputable def f (x : ℝ) : ℝ := Real.sin x + (1 / Real.cos x)

-- Theorem stating that f has a period of 2π
theorem f_period_2pi : 
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_2pi_l381_38105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_scenario_theorem_l381_38183

/-- Represents the fuel efficiency and travel scenarios for a car --/
structure CarScenario where
  highway_efficiency : ℚ  -- Miles per gallon on highway
  city_efficiency : ℚ     -- Miles per gallon in city
  scenario1_highway : ℚ   -- Miles traveled on highway in Scenario 1
  scenario1_city : ℚ      -- Miles traveled in city in Scenario 1
  fuel_increase_factor : ℚ -- Factor by which fuel consumption increases in Scenario 2

/-- Calculates the miles traveled on highway in Scenario 2 --/
def scenario2_highway_miles (c : CarScenario) : ℚ :=
  (c.scenario1_highway / c.highway_efficiency + c.scenario1_city / c.city_efficiency) * c.highway_efficiency / c.fuel_increase_factor

/-- Theorem stating that given the specific conditions, the car travels 8 miles on the highway in Scenario 2 --/
theorem car_scenario_theorem (c : CarScenario) 
  (h1 : c.highway_efficiency = 40)
  (h2 : c.city_efficiency = 20)
  (h3 : c.scenario1_highway = 4)
  (h4 : c.scenario1_city = 4)
  (h5 : c.fuel_increase_factor = 3/2)
  : scenario2_highway_miles c = 8 := by
  sorry

/-- Example calculation --/
def example_scenario : CarScenario := {
  highway_efficiency := 40,
  city_efficiency := 20,
  scenario1_highway := 4,
  scenario1_city := 4,
  fuel_increase_factor := 3/2
}

#eval scenario2_highway_miles example_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_scenario_theorem_l381_38183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_outcome_for_first_player_l381_38154

/-- Represents a quadratic equation of the form x^2 + px + q = 0 --/
structure QuadraticEquation where
  p : ℕ
  q : ℕ

/-- Checks if a quadratic equation has two distinct roots --/
def hasTwoDistinctRoots (eq : QuadraticEquation) : Prop :=
  eq.p^2 > 4 * eq.q

/-- The set of natural numbers from 1 to 10 --/
def validNumbers : Finset ℕ :=
  Finset.range 11 \ {0}

/-- A game state consisting of five quadratic equations --/
def GameState : Type :=
  Fin 5 → QuadraticEquation

/-- The theorem to be proved --/
theorem best_outcome_for_first_player
  (initialState : GameState)
  (strategy : GameState → Fin 5 → ℕ → GameState) :
  ∃ (finalState : GameState),
    (∀ i : Fin 5, (finalState i).p ∈ validNumbers ∧ (finalState i).q ∈ validNumbers) ∧
    (∀ i j : Fin 5, i ≠ j → (finalState i).p ≠ (finalState j).p ∧ (finalState i).q ≠ (finalState j).q) ∧
    (∃ (S : Finset (Fin 5)), S.card = 3 ∧ ∀ i ∈ S, hasTwoDistinctRoots (finalState i)) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_outcome_for_first_player_l381_38154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_cookie_price_theorem_l381_38182

/-- The radius of Art's circular cookies -/
def art_cookie_radius : ℝ := 2

/-- The number of cookies Art makes -/
def art_cookie_count : ℕ := 10

/-- The price of one of Art's cookies in cents -/
def art_cookie_price : ℝ := 50

/-- The total amount of dough used (in square units) -/
noncomputable def total_dough : ℝ := art_cookie_count * Real.pi * art_cookie_radius^2

/-- The total earnings from Art's cookies in cents -/
def total_earnings : ℝ := art_cookie_count * art_cookie_price

/-- The number of cookies Roger makes -/
def roger_cookie_count : ℕ := 16

/-- The price of one of Roger's cookies in cents -/
noncomputable def roger_cookie_price : ℝ := total_earnings / roger_cookie_count

theorem roger_cookie_price_theorem :
  roger_cookie_price = 31.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roger_cookie_price_theorem_l381_38182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_is_square_diagonal_length_l381_38113

/-- The quadratic equation representing the sides of the rectangle -/
def side_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - m*x + m/2 - 1/4 = 0

/-- A rectangle ABCD with sides satisfying the given equation -/
structure Rectangle (m : ℝ) where
  AB : ℝ
  AD : ℝ
  hAB : side_equation m AB
  hAD : side_equation m AD

/-- Theorem: A rectangle is a square if and only if m = 1 -/
theorem rectangle_is_square (m : ℝ) (rect : Rectangle m) :
  rect.AB = rect.AD ↔ m = 1 :=
sorry

/-- Theorem: If AB = 2, then the diagonal length is √17/2 -/
theorem diagonal_length (m : ℝ) (rect : Rectangle m) :
  rect.AB = 2 → Real.sqrt (rect.AB^2 + rect.AD^2) = Real.sqrt 17 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_is_square_diagonal_length_l381_38113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_complex_number_l381_38169

noncomputable def initial_number : ℂ := -4 + 6 * Complex.I

noncomputable def rotation_60_degrees (z : ℂ) : ℂ := z * (1/2 + (Real.sqrt 3 / 2) * Complex.I)

noncomputable def dilation_factor_2 (z : ℂ) : ℂ := 2 * z

noncomputable def combined_transformation (z : ℂ) : ℂ := dilation_factor_2 (rotation_60_degrees z)

theorem transform_complex_number :
  combined_transformation initial_number = -22 + (6 - 4 * Real.sqrt 3) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_complex_number_l381_38169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_proof_l381_38120

noncomputable def x : List ℝ := [12, 13, 14, 15]
noncomputable def y : List ℝ := [26, 29, 28, 31]

noncomputable def x_mean : ℝ := (List.sum x) / (List.length x)
noncomputable def y_mean : ℝ := (List.sum y) / (List.length y)

noncomputable def sum_xy_diff : ℝ := 7

noncomputable def sum_x_diff_sq : ℝ := List.sum (List.map (fun xi => (xi - x_mean)^2) x)

noncomputable def b_hat : ℝ := sum_xy_diff / sum_x_diff_sq
noncomputable def a_hat : ℝ := y_mean - b_hat * x_mean

theorem linear_regression_proof :
  b_hat = 1.4 ∧ 
  a_hat = 9.6 ∧ 
  (∀ n : ℕ, n < 19 → 1.4 * (n : ℝ) + 9.6 ≤ 36) ∧
  1.4 * 19 + 9.6 > 36 := by
  sorry

#check linear_regression_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_proof_l381_38120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_rational_root_of_polynomials_l381_38119

/-- Represents a polynomial with coefficients of type α -/
def MyPolynomial (α : Type*) := List α

/-- Evaluates a polynomial at a given point x -/
def evaluatePolynomial {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (λ acc coeff => acc * x + coeff) 0

/-- Checks if a rational number is a root of a polynomial -/
def isRoot {α : Type*} [Field α] (p : MyPolynomial α) (r : α) : Prop :=
  evaluatePolynomial p r = 0

theorem common_rational_root_of_polynomials 
  (a b c d e f g : ℚ) : 
  ∃! (k : ℚ), 
    isRoot [8, c, b, a, 45] k ∧ 
    isRoot [45, g, f, e, d, 8] k ∧ 
    k < 0 ∧ 
    ¬(∃ (n : ℤ), k = ↑n) ∧
    k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_rational_root_of_polynomials_l381_38119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l381_38150

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point on the plane
def Point : Type := ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the circumcenter of a triangle
noncomputable def circumcenter (t : Triangle) : Point := sorry

-- Define the condition for points M₁ and M₂
def satisfiesCondition (t : Triangle) (M₁ M₂ : Point) : Prop :=
  ∃ (p q r : ℝ),
    distance t.A M₁ / distance t.B M₁ = p / q ∧
    distance t.B M₁ / distance t.C M₁ = q / r ∧
    distance t.C M₁ / distance t.A M₁ = r / p ∧
    distance t.A M₂ / distance t.B M₂ = p / q ∧
    distance t.B M₂ / distance t.C M₂ = q / r ∧
    distance t.C M₂ / distance t.A M₂ = r / p

-- Define a line passing through two points
def line (p q : Point) : Set Point := sorry

-- The main theorem
theorem fixed_point_theorem (t : Triangle) (M₁ M₂ : Point) :
  satisfiesCondition t M₁ M₂ → circumcenter t ∈ line M₁ M₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l381_38150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l381_38104

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem equation_has_two_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, 9 * x^2 - 45 * (floor (x^2 - 1)) + 94 = 0 :=
by
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l381_38104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_l381_38151

theorem sum_of_max_and_min : ℤ := by
  -- Define the set of numbers
  let numbers : List ℤ := [-2^4, 15, -18, (-4)^2]
  
  -- Define the maximum and minimum of the list
  let max_value := numbers.maximum?
  let min_value := numbers.minimum?
  
  -- The theorem
  have sum_of_extremes : 
    (max_value.getD 0) + (min_value.getD 0) = -2 := by sorry
  
  exact -2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_l381_38151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_inequality_l381_38141

/-- Binomial coefficient -/
def binomial (n m : ℕ) : ℕ :=
  if m ≤ n then
    Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))
  else
    0

/-- The sum in the inequality -/
noncomputable def sum (n j : ℕ) : ℚ :=
  ∑' i : ℕ, (-1)^n * (binomial n (3*i + j) : ℚ)

/-- The main theorem -/
theorem binomial_sum_inequality (n j : ℕ) (h : j ∈ ({0, 1, 2} : Set ℕ)) :
  sum n j ≥ (((-2)^n : ℚ) - 1) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_inequality_l381_38141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l381_38157

theorem problem_statement : 
  let p := ∀ x : ℝ, (3 : ℝ)^x ≤ 0
  let q := (∀ x : ℝ, x > 2 → x > 4) ∧ (∃ x : ℝ, x > 4 ∧ x ≤ 2)
  ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l381_38157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l381_38199

-- Define the function f(x) = ln(cos(2x))
noncomputable def f (x : ℝ) := Real.log (Real.cos (2 * x))

-- Theorem statement
theorem f_properties :
  -- 1. f is not an odd function
  (∃ x : ℝ, f (-x) ≠ -f x) ∧
  -- 2. The smallest positive period of f is π
  (∀ y : ℝ, y > 0 ∧ (∀ x : ℝ, f (x + y) = f x) → y ≥ π) ∧
  (∀ x : ℝ, f (x + π) = f x) ∧
  -- 3. f has at least one zero
  (∃ x : ℝ, f x = 0) ∧
  -- 4. f has at least one critical point
  (∃ x : ℝ, deriv f x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l381_38199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_symmetry_axes_l381_38185

/-- A heptagon is a polygon with 7 sides. -/
structure Heptagon where
  sides : Finset ℕ
  card_eq : sides.card = 7

/-- The number of axes of symmetry in a polygon. -/
def numAxesOfSymmetry (h : Heptagon) : ℕ := sorry

/-- Theorem: The number of axes of symmetry in a heptagon is either 0, 1, or 7. -/
theorem heptagon_symmetry_axes (h : Heptagon) : 
  numAxesOfSymmetry h ∈ ({0, 1, 7} : Finset ℕ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_symmetry_axes_l381_38185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l381_38137

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (1/3 * x - Real.pi/6)

theorem problem_solution (α β : ℝ) 
  (h1 : 0 ≤ α ∧ α ≤ Real.pi/2) 
  (h2 : 0 ≤ β ∧ β ≤ Real.pi/2) 
  (h3 : f (3*α + Real.pi/2) = 10/13) 
  (h4 : f (3*β + 2*Real.pi) = 6/5) : 
  f (5*Real.pi/4) = Real.sqrt 2 ∧ Real.cos (α + β) = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l381_38137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l381_38189

/-- Helper function to represent the focus of a parabola -/
noncomputable def focus_of_parabola (x y : ℝ) : ℝ × ℝ :=
  sorry -- Implementation details omitted

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 2 * x^2

/-- The focus of the parabola y = 2x² is at (0, 1/8) -/
theorem parabola_focus (x y : ℝ) : 
  parabola_equation x y → focus_of_parabola x y = (0, 1/8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l381_38189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_tips_fraction_l381_38161

/-- Represents the tips earned by Williams at a resort --/
structure ResortTips where
  /-- Average monthly tips for months other than July and August --/
  avg_other_months : ℝ
  /-- Number of months worked (February to October) --/
  total_months : ℕ
  /-- Number of months excluding July, August, and September --/
  other_months : ℕ
  /-- July tips multiplier --/
  july_multiplier : ℕ
  /-- August tips multiplier --/
  august_multiplier : ℕ
  /-- September tips as a fraction of August tips --/
  september_fraction : ℝ
  /-- Assertion that total months is 9 --/
  total_months_eq : total_months = 9
  /-- Assertion that other months is 6 --/
  other_months_eq : other_months = 6
  /-- Assertion for July multiplier --/
  july_mult_eq : july_multiplier = 10
  /-- Assertion for August multiplier --/
  august_mult_eq : august_multiplier = 15
  /-- Assertion for September fraction --/
  september_frac_eq : september_fraction = 0.75

/-- Theorem stating the fraction of total tips from July, August, and September --/
theorem summer_tips_fraction (tips : ResortTips) :
  (tips.avg_other_months * (tips.july_multiplier + tips.august_multiplier + tips.september_fraction * tips.august_multiplier)) /
  (tips.avg_other_months * (tips.other_months + tips.july_multiplier + tips.august_multiplier + tips.september_fraction * tips.august_multiplier)) =
  36.25 / 44.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_summer_tips_fraction_l381_38161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slanted_roof_surface_area_l381_38144

/-- Represents a rectangular slanted roof -/
structure SlantedRoof where
  width : ℝ
  length : ℝ
  pitch_angle : ℝ
  base_area : ℝ

/-- Calculates the surface area of a slanted roof -/
noncomputable def surface_area (roof : SlantedRoof) : ℝ :=
  roof.length * (roof.width / Real.sin roof.pitch_angle)

/-- Theorem stating the surface area of the specific roof -/
theorem slanted_roof_surface_area :
  ∀ (roof : SlantedRoof),
    roof.length = 4 * roof.width →
    roof.pitch_angle = Real.pi / 6 →
    roof.base_area = 576 →
    surface_area roof = 1152 := by
  intro roof length_eq pitch_eq base_area_eq
  sorry

-- Remove the #eval statement as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slanted_roof_surface_area_l381_38144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_quadratic_roots_l381_38171

variable (a b c : ℝ)

-- Define the original quadratic equation
def original_quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the new quadratic equation
def new_quadratic (x : ℝ) : ℝ := c * x^2 + b * x + a

-- Theorem statement
theorem new_quadratic_roots (x₁ x₂ : ℝ) :
  (∀ x, original_quadratic a b c x = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, new_quadratic a b c x = 0 ↔ x = 1/x₁ ∨ x = 1/x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_quadratic_roots_l381_38171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_roots_range_l381_38115

theorem sin_equation_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ ∈ Set.Icc (π/3) π ∧ 
   x₂ ∈ Set.Icc (π/3) π ∧ 
   Real.sin x₁ = (1-a)/2 ∧ 
   Real.sin x₂ = (1-a)/2) ↔ 
  a ∈ Set.Ioc (-1) (1 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_equation_roots_range_l381_38115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_implies_coefficient_l381_38168

-- Define the integral condition
def integral_condition (n : ℝ) : Prop :=
  ∫ x in (0 : ℝ)..n, |x - 5| = 25

-- Define the coefficient of x^2 in the binomial expansion
def coefficient_x_squared (n : ℕ) : ℕ :=
  n.choose (n - 2) * 2^2

-- Theorem statement
theorem integral_implies_coefficient :
  ∀ n : ℝ, integral_condition n →
    ∃ m : ℕ, (n : ℝ) = m ∧ coefficient_x_squared m = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_implies_coefficient_l381_38168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_foci_implies_k_equals_two_l381_38159

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

/-- A hyperbola with semi-major axis a and parameter k -/
structure Hyperbola (a k : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / k^2 = 1

/-- The foci of an ellipse -/
noncomputable def ellipse_foci (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

/-- The foci of a hyperbola -/
noncomputable def hyperbola_foci (a k : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + k^2), 0)

/-- 
If an ellipse and a hyperbola have the same foci, then k = 2
-/
theorem same_foci_implies_k_equals_two (a b k : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_k : 0 < k)
  (e : Ellipse a b) (h : Hyperbola a k) 
  (h_same_foci : ellipse_foci a b = hyperbola_foci a k) : 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_foci_implies_k_equals_two_l381_38159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l381_38123

theorem solution_pairs (m n : ℤ) : 
  ((-2 : ℝ) ^ (2 * m + n) = (2 : ℝ) ^ (24 - m)) ↔ 
  ((m = 0 ∧ n = 24) ∨ 
   (m = 1 ∧ n = 21) ∨ 
   (m = 2 ∧ n = 18) ∨ 
   (m = 3 ∧ n = 15) ∨ 
   (m = 4 ∧ n = 12) ∨ 
   (m = 5 ∧ n = 9) ∨ 
   (m = 6 ∧ n = 6) ∨ 
   (m = 7 ∧ n = 3) ∨ 
   (m = 8 ∧ n = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l381_38123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l381_38147

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 2 + Real.pi / 3)

theorem function_properties :
  (∃ T : ℝ, T > 0 ∧ T = 4 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x, f (Real.pi / 3 + (Real.pi / 3 - x)) = f x) ∧
  (∀ x ∈ Set.Ioo (2 * Real.pi / 3) (5 * Real.pi / 6), ∀ y ∈ Set.Ioo (2 * Real.pi / 3) (5 * Real.pi / 6),
    x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l381_38147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photographer_choices_l381_38173

theorem photographer_choices (portrait landscape macro_photos street : Nat) :
  portrait = 10 →
  landscape = 8 →
  macro_photos = 5 →
  street = 4 →
  (portrait.choose 2) * (landscape.choose 2) * (macro_photos.choose 1) * (street.choose 1) = 25200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_photographer_choices_l381_38173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_in_interval_l381_38143

/-- Given two vectors a and b in a real inner product space, 
    and a third vector c defined as a linear combination of a and b,
    prove that the cosine of the angle between a and c lies within a specific interval. -/
theorem cosine_angle_in_interval 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (x y : ℝ) :
  let c := x • a + y • b
  norm b = 2 * norm a →
  norm a = 1 →
  inner a b = norm a * norm b * (1 / 2) →
  x ∈ Set.Icc 1 2 →
  y ∈ Set.Icc 1 2 →
  (inner a c / (norm a * norm c) : ℝ) ∈ Set.Icc (Real.sqrt 21 / 7) (Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_in_interval_l381_38143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l381_38196

open Real

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (Real.sqrt 2 * c * Real.sin A * Real.cos B = a * Real.sin C) →
  (B = π/4 ∧
   (area_triangle a b c = a^2 → Real.cos A = 3 * Real.sqrt 10 / 10)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l381_38196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l381_38140

/-- Represents the production of cogs on an assembly line -/
structure CogProduction where
  initial_rate : ℚ
  initial_order : ℚ
  increased_rate : ℚ
  second_batch : ℚ

/-- Calculates the overall average output of cogs per hour -/
noncomputable def average_output (prod : CogProduction) : ℚ :=
  (prod.initial_order + prod.second_batch) / 
  (prod.initial_order / prod.initial_rate + prod.second_batch / prod.increased_rate)

/-- Theorem stating that the average output for the given production scenario is 40 cogs per hour -/
theorem assembly_line_output :
  let prod : CogProduction := {
    initial_rate := 30,
    initial_order := 60,
    increased_rate := 60,
    second_batch := 60
  }
  average_output prod = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l381_38140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l381_38124

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_problem (P : ℝ) (h : simple_interest P 5 2 = 50) :
  compound_interest P 5 2 = 51.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_problem_l381_38124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_b_preferred_l381_38166

-- Define the original setup
def red_balls : ℕ := 2
def white_balls : ℕ := 8
def total_balls : ℕ := red_balls + white_balls

-- Define voucher amounts
noncomputable def voucher_two_red : ℚ := 200
noncomputable def voucher_one_each : ℚ := 80
noncomputable def voucher_two_white : ℚ := 10

-- Define Plan A
def plan_a_red_balls : ℕ := red_balls + 1
def plan_a_white_balls : ℕ := white_balls + 1
def plan_a_total_balls : ℕ := plan_a_red_balls + plan_a_white_balls

-- Define Plan B
noncomputable def plan_b_increase : ℚ := 10

-- Expected value calculation function
noncomputable def expected_value (r : ℕ) (w : ℕ) (t : ℕ) (v_rr v_rw v_ww : ℚ) : ℚ :=
  (v_rr * (r * (r - 1)) + v_rw * (2 * r * w) + v_ww * (w * (w - 1))) / (t * (t - 1))

-- Theorem statement
theorem plan_b_preferred :
  expected_value red_balls white_balls total_balls 
    (voucher_two_red + plan_b_increase) 
    (voucher_one_each + plan_b_increase) 
    (voucher_two_white + plan_b_increase) >
  expected_value plan_a_red_balls plan_a_white_balls plan_a_total_balls 
    voucher_two_red voucher_one_each voucher_two_white :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan_b_preferred_l381_38166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cosine_function_l381_38136

theorem min_value_cosine_function (k : ℝ) (h : k < -4) :
  ∀ x : ℝ, Real.cos (2 * x) + k * (Real.cos x - 1) ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cosine_function_l381_38136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_from_squares_specific_rectangle_area_l381_38184

/-- The area of a rectangle formed by attaching squares in a row -/
theorem rectangle_area_from_squares (n : ℝ) (side : ℝ) : 
  n * side * side = n * side^2 := by
  ring

/-- The area of a rectangle formed by attaching 3.2 squares with sides of 8.5 cm in a row -/
theorem specific_rectangle_area : 
  (3.2 : ℝ) * (8.5 : ℝ) * (8.5 : ℝ) = 231.2 := by
  norm_num

#eval (3.2 : Float) * (8.5 : Float) * (8.5 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_from_squares_specific_rectangle_area_l381_38184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nutrition_requirements_satisfied_l381_38176

theorem nutrition_requirements_satisfied (food_A food_B : ℝ) : 
  food_A = 28 ∧ food_B = 30 →
  0.5 * food_A + 0.7 * food_B = 35 ∧
  1 * food_A + 0.4 * food_B = 40 := by
  intro h
  cases' h with hA hB
  constructor
  · -- Prove the protein requirement
    calc
      0.5 * food_A + 0.7 * food_B = 0.5 * 28 + 0.7 * 30 := by rw [hA, hB]
      _ = 14 + 21 := by norm_num
      _ = 35 := by norm_num
  · -- Prove the iron requirement
    calc
      1 * food_A + 0.4 * food_B = 1 * 28 + 0.4 * 30 := by rw [hA, hB]
      _ = 28 + 12 := by norm_num
      _ = 40 := by norm_num

#check nutrition_requirements_satisfied

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nutrition_requirements_satisfied_l381_38176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rancher_driving_time_l381_38142

/-- Calculates the total driving time to transport cattle to higher ground -/
noncomputable def total_driving_time (total_cattle : ℕ) (distance : ℝ) (truck_capacity : ℕ) (speed : ℝ) : ℝ :=
  let num_trips : ℕ := total_cattle / truck_capacity
  let round_trip_distance : ℝ := 2 * distance
  let total_distance : ℝ := (num_trips : ℝ) * round_trip_distance
  total_distance / speed

/-- Proves that the total driving time for the given conditions is 40 hours -/
theorem rancher_driving_time :
  total_driving_time 400 60 20 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rancher_driving_time_l381_38142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l381_38165

open Real

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := log x + (1/2) * x^2

/-- The function F as defined in the problem -/
noncomputable def F (x : ℝ) : ℝ := f x + x

/-- The main theorem to prove -/
theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : F x₁ = -F x₂) :
  x₁ + x₂ ≥ Real.sqrt 3 - 1 := by
  sorry

#check sum_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lower_bound_l381_38165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_transformation_l381_38117

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_to_geometric_transformation :
  ∀ a : ℕ → ℤ,
  arithmetic_sequence a →
  a 1 = -8 →
  a 2 = -6 →
  ∀ x : ℤ,
  geometric_sequence (λ n ↦ if n = 1 then a 1 + x
                            else if n = 2 then a 4 + x
                            else a 5 + x) →
  x = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_transformation_l381_38117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_and_wage_calculation_l381_38174

def planned_weekly_production : ℕ := 1400
def average_daily_production : ℕ := 200
def daily_deviations : List ℤ := [5, -2, -4, 13, -10, 16, -9]
def base_wage : ℕ := 60
def additional_reward : ℕ := 15

def actual_weekly_production : ℕ := 
  (planned_weekly_production : ℤ) + daily_deviations.sum |>.toNat

def total_wage : ℕ :=
  actual_weekly_production * base_wage +
  ((daily_deviations.filter (λ x => x > 0)).sum * additional_reward).toNat

theorem production_and_wage_calculation :
  actual_weekly_production = 1409 ∧ total_wage = 85050 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_and_wage_calculation_l381_38174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l381_38138

noncomputable def sinFunction (a b c d : ℝ) (x : ℝ) : ℝ := a * Real.sin (b * x + c) + d

theorem smallest_c_value 
  (a b c d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (amplitude : a = 3) 
  (max_value : sinFunction a b c d (π / 6) = 5) 
  (is_max : ∀ x, sinFunction a b c d x ≤ 5) :
  c ≥ π / 2 ∧ ∃ (c_min : ℝ), c_min = π / 2 ∧ 
    sinFunction a b c_min d (π / 6) = 5 ∧ 
    (∀ x, sinFunction a b c_min d x ≤ 5) := by
  sorry

#check smallest_c_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_value_l381_38138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_integer_l381_38110

theorem largest_odd_integer (seq : List ℕ) : 
  (seq.length = 20) →                           -- There are 20 integers
  (∀ n ∈ seq, n % 2 = 1) →                      -- All integers are odd
  (∀ i ∈ Finset.range 19, seq[i + 1]! = seq[i]! + 2) →  -- Consecutive odd integers
  (seq.sum = 2600) →                            -- Sum is 2600
  (seq.maximum = some 149) :=                   -- The largest integer is 149
by
  intro h_length h_odd h_consecutive h_sum
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_integer_l381_38110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_six_l381_38135

theorem polynomial_value_at_six (a b c d e f : ℝ) :
  let P : ℂ → ℂ := λ x => (2*x^4 - 26*x^3 + a*x^2 + b*x + c) * (5*x^4 - 80*x^3 + d*x^2 + e*x + f)
  (∀ z : ℂ, P z = 0 → z ∈ ({1, 2, 3, 4, 5} : Set ℂ)) →
  (P 6 : ℂ) = 2400 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_six_l381_38135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_oil_for_brownies_l381_38146

/-- Represents the recipe and chef's situation for making brownies -/
structure BrownieRecipe where
  butter_per_cup : ℚ  -- Amount of butter needed per cup of baking mix
  coconut_oil_sub : ℚ  -- Amount of coconut oil that can substitute butter
  available_butter : ℚ  -- Amount of butter the chef has
  total_baking_mix : ℚ  -- Total amount of baking mix to be used

/-- Calculates the amount of coconut oil needed for the brownie recipe -/
def coconut_oil_needed (recipe : BrownieRecipe) : ℚ :=
  let total_fat_needed := recipe.butter_per_cup * recipe.total_baking_mix
  let butter_used := min recipe.available_butter total_fat_needed
  max 0 (total_fat_needed - butter_used)

/-- Theorem stating that given the specific recipe conditions, 8 ounces of coconut oil are needed -/
theorem coconut_oil_for_brownies :
  let recipe := BrownieRecipe.mk 2 2 4 6
  coconut_oil_needed recipe = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coconut_oil_for_brownies_l381_38146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l381_38103

noncomputable def average_speed (outbound_speed inbound_speed : ℝ) : ℝ :=
  2 / (1 / outbound_speed + 1 / inbound_speed)

theorem round_trip_average_speed :
  average_speed 60 36 = 45 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l381_38103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athletes_meeting_and_overtakes_l381_38162

/-- Represents an athlete with a given speed -/
structure Athlete where
  speed : ℚ

/-- Represents a circular track with given length -/
structure Track where
  length : ℚ

/-- Calculates the time for two athletes to meet on a track -/
noncomputable def meetTime (a1 a2 : Athlete) (t : Track) : ℚ :=
  t.length / (a2.speed - a1.speed)

/-- Calculates the number of overtakes between two athletes in a given time -/
noncomputable def overtakes (a1 a2 : Athlete) (t : Track) (time : ℚ) : ℕ :=
  (Int.floor (time / meetTime a1 a2 t) - 1).toNat

/-- The main theorem to prove -/
theorem athletes_meeting_and_overtakes 
  (t : Track)
  (a1 a2 a3 : Athlete)
  (h1 : t.length = 400)
  (h2 : a1.speed = 155)
  (h3 : a2.speed = 200)
  (h4 : a3.speed = 275) :
  let minTime : ℚ := 80 / 3
  (minTime = meetTime a1 a2 t ∧ 
   minTime = meetTime a1 a3 t ∧ 
   minTime = meetTime a2 a3 t) ∧
  (overtakes a1 a2 t minTime + 
   overtakes a1 a3 t minTime + 
   overtakes a2 a3 t minTime = 13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athletes_meeting_and_overtakes_l381_38162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l381_38102

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry :
  ∃ (a : ℝ), a = Real.pi / 12 ∧
  ∀ (x : ℝ), f (a + x) = f (a - x) := by
  sorry

#check axis_of_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l381_38102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l381_38198

theorem no_real_solutions : ¬∃ x : ℝ, (2 : ℝ)^(4*x+2) * (8 : ℝ)^(2*x+1) = (32 : ℝ)^(2*x+3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_solutions_l381_38198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transform_l381_38172

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin ((1/3) * x - 3)

-- Theorem statement
theorem sin_transform :
  ∀ x : ℝ, g x = 3 * f ((1/3) * x - 1) :=
by
  intro x
  -- Expand the definitions of g and f
  unfold g f
  -- Simplify the right-hand side
  simp [Real.sin_sub]
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_transform_l381_38172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_equivalence_l381_38149

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 8 * 2^(-x)

-- Define the reference function
noncomputable def g (x : ℝ) : ℝ := (1/2)^x

-- Define the shifted reference function
noncomputable def h (x : ℝ) : ℝ := g (x - 3)

-- Theorem statement
theorem graph_equivalence : ∀ x : ℝ, f x = h x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_equivalence_l381_38149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l381_38178

/-- Given a line l with direction vector (4, 2, m) and a plane α with normal vector (2, 1, -1),
    if l is perpendicular to α, then m = -2. -/
theorem perpendicular_line_plane (m : ℝ) : 
  let l_dir : Fin 3 → ℝ := ![4, 2, m]
  let α_normal : Fin 3 → ℝ := ![2, 1, -1]
  (∀ (t : ℝ), (l_dir 0) * (α_normal 0) + (l_dir 1) * (α_normal 1) + (l_dir 2) * (α_normal 2) = 0) →
  m = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_plane_l381_38178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_BC_equation_l381_38148

noncomputable section

-- Define the line m
def line_m (x y : ℝ) : Prop := y = 2 * x - 16

-- Define the parabola C
def parabola_C (a x y : ℝ) : Prop := y^2 = a * x ∧ a > 0

-- Define the focus of a parabola
def focus (a : ℝ) : ℝ × ℝ := (a / 4, 0)

-- Define a point on a parabola
def point_on_parabola (a x y : ℝ) : Prop := parabola_C a x y

-- Define the centroid of a triangle
def centroid (xA yA xB yB xC yC : ℝ) : ℝ × ℝ :=
  ((xA + xB + xC) / 3, (yA + yB + yC) / 3)

-- The main theorem
theorem line_BC_equation (a xA yA xB yB xC yC : ℝ) :
  parabola_C a xA yA →
  parabola_C a xB yB →
  parabola_C a xC yC →
  yA = 8 →
  line_m (a / 4) 0 →
  centroid xA yA xB yB xC yC = focus a →
  ∃ (x y : ℝ), 4 * x + y - 40 = 0 ∧ 
    ((x = xB ∧ y = yB) ∨ (x = xC ∧ y = yC)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_BC_equation_l381_38148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_table_sum_bounds_l381_38107

def numbers : List ℕ := [2, 3, 5, 7, 11, 17]

def sum_of_products (a b c d e f : ℕ) : ℕ := (a + b + c) * (d + e + f)

theorem multiplication_table_sum_bounds :
  let arrangements := numbers.permutations.filter (λ l => l.length = 6)
  ∀ perm ∈ arrangements,
    let a := perm[0]!
    let b := perm[1]!
    let c := perm[2]!
    let d := perm[3]!
    let e := perm[4]!
    let f := perm[5]!
    sum_of_products a b c d e f ≤ 450 ∧
    sum_of_products a b c d e f ≥ 504 ∧
    (∃ perm₁ perm₂, perm₁ ∈ arrangements ∧ perm₂ ∈ arrangements ∧
      let a₁ := perm₁[0]!
      let b₁ := perm₁[1]!
      let c₁ := perm₁[2]!
      let d₁ := perm₁[3]!
      let e₁ := perm₁[4]!
      let f₁ := perm₁[5]!
      let a₂ := perm₂[0]!
      let b₂ := perm₂[1]!
      let c₂ := perm₂[2]!
      let d₂ := perm₂[3]!
      let e₂ := perm₂[4]!
      let f₂ := perm₂[5]!
      sum_of_products a₁ b₁ c₁ d₁ e₁ f₁ = 450 ∧
      sum_of_products a₂ b₂ c₂ d₂ e₂ f₂ = 504) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_table_sum_bounds_l381_38107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_l381_38158

/-- Represents the game state -/
structure GameState where
  current : ℕ
  is_first_player_turn : Bool

/-- Checks if a number is a proper divisor of another number -/
def is_proper_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ n ∧ d ≠ 0

/-- Defines a valid move in the game -/
def valid_move (s : GameState) (m : ℕ) : Prop :=
  is_proper_divisor m s.current ∧ m < s.current

/-- Applies a move to the game state -/
def apply_move (s : GameState) (m : ℕ) : GameState :=
  { current := s.current - m, is_first_player_turn := ¬s.is_first_player_turn }

/-- Checks if the game is over (current number is 1) -/
def is_game_over (s : GameState) : Prop := s.current = 1

/-- Defines a winning strategy for the second player -/
def second_player_wins (initial : ℕ) : Prop :=
  ∃ (strategy : GameState → ℕ),
    ∀ (game : ℕ → GameState),
      game 0 = { current := initial, is_first_player_turn := true } →
      (∀ n, ¬is_game_over (game n) →
        (game n).is_first_player_turn →
          valid_move (game n) ((game (n+1)).current - (game n).current) ∧
          game (n+1) = apply_move (game n) ((game (n+1)).current - (game n).current)) →
      (∀ n, ¬is_game_over (game n) →
        ¬(game n).is_first_player_turn →
          valid_move (game n) (strategy (game n)) ∧
          game (n+1) = apply_move (game n) (strategy (game n))) →
      ∃ k, is_game_over (game k) ∧ (game k).is_first_player_turn

/-- Theorem stating that the second player (Vasya) wins the game starting with 2017 -/
theorem vasya_wins : second_player_wins 2017 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_l381_38158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l381_38130

/-- Hyperbola with foci F₁ and F₂, and point P on the right branch -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x, y) = P
  foci_dist : dist F₁ F₂ = 2 * dist (0, 0) P
  P_right_branch : P.1 > 0
  PF₁_geq_3PF₂ : dist P F₁ ≥ 3 * dist P F₂

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := dist (0, 0) h.F₁ / h.a

/-- Theorem about the range of eccentricity for a hyperbola satisfying given conditions -/
theorem eccentricity_range (h : Hyperbola) : 
  1 < eccentricity h ∧ eccentricity h ≤ Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l381_38130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l381_38134

theorem division_problem (x y : ℕ) (h1 : x % y = 8) (h2 : (x : ℝ) / y = 76.4) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l381_38134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approx_50_hours_l381_38163

/-- The approximate time (in hours) for a jet to fly around Earth's equator -/
noncomputable def flight_time (earth_radius : ℝ) (jet_speed : ℝ) : ℝ :=
  (2 * Real.pi * earth_radius) / jet_speed

/-- Theorem stating that the flight time is approximately 50 hours -/
theorem flight_time_approx_50_hours :
  let earth_radius : ℝ := 4200
  let jet_speed : ℝ := 525
  |flight_time earth_radius jet_speed - 50| < 1 := by
  sorry

#check flight_time_approx_50_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approx_50_hours_l381_38163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_gravitational_force_l381_38145

/-- Gravitational force as a function of distance -/
noncomputable def gravitational_force (k : ℝ) (d : ℝ) : ℝ := k / (d ^ 2)

theorem satellite_gravitational_force (k : ℝ) :
  gravitational_force k 1000 = 500 →
  gravitational_force k 10000 = (1/2 : ℝ) := by
  intro h
  -- The proof steps would go here
  sorry

#check satellite_gravitational_force

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satellite_gravitational_force_l381_38145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_must_be_squares_l381_38194

theorem rectangles_must_be_squares (n : ℕ) (Q : ℕ) (k l : ℕ)
  (h1 : n > 1)
  (h2 : Nat.Prime Q)
  (h3 : ∃ (cuts : Fin n → ℕ),
    Q = (Finset.univ.sum (λ i => (k / cuts i) * (l / cuts i))) ∧
    ∀ i : Fin n, cuts i ∣ k ∧ cuts i ∣ l) :
  k = l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_must_be_squares_l381_38194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l381_38100

/-- The area of the region bounded by r = 2sec(θ), r = 2csc(θ), x-axis, and y-axis -/
def bounded_region_area : ℝ := 4

/-- The curve r = 2sec(θ) in polar coordinates -/
noncomputable def curve1 (θ : ℝ) : ℝ := 2 / Real.cos θ

/-- The curve r = 2csc(θ) in polar coordinates -/
noncomputable def curve2 (θ : ℝ) : ℝ := 2 / Real.sin θ

/-- Theorem stating that the area of the region bounded by the given curves and axes is 4 -/
theorem area_of_bounded_region :
  bounded_region_area = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l381_38100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_25_16_l381_38101

/-- The area of a triangle with given base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (base * height) / 2

/-- Theorem: The area of a triangle with base 25 cm and height 16 cm is 200 square cm -/
theorem triangle_area_25_16 : triangle_area 25 16 = 200 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the arithmetic
  simp [mul_div_right_comm]
  -- The result should now be obvious to Lean
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_25_16_l381_38101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_difference_l381_38131

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- Represents the first figure -/
structure Figure1 where
  baseLength : ℕ
  baseWidth : ℕ
  extensionLength : ℕ
  extensionWidth : ℕ

/-- Represents the second figure -/
structure Figure2 where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of the first figure -/
def figure1Perimeter (f : Figure1) : ℕ :=
  rectanglePerimeter f.baseLength f.baseWidth + 2 * (f.extensionLength + f.extensionWidth)

/-- Calculates the perimeter of the second figure -/
def figure2Perimeter (f : Figure2) : ℕ :=
  rectanglePerimeter f.length f.width

/-- The main theorem to prove -/
theorem perimeter_difference (f1 : Figure1) (f2 : Figure2)
  (h1 : f1.baseLength = 5 ∧ f1.baseWidth = 1 ∧ f1.extensionLength = 2 ∧ f1.extensionWidth = 1)
  (h2 : f2.length = 5 ∧ f2.width = 2) :
  Int.natAbs (figure1Perimeter f1 - figure2Perimeter f2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_difference_l381_38131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_parabola_l381_38153

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The perpendicular bisector of AB intersects x-axis at D(4,0) -/
def perpendicularBisectorIntersectsAt4 (A B : Point) : Prop :=
  ∃ (M : Point), M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2 ∧
  (M.y / (M.x - 4) = -(B.x - A.x) / (B.y - A.y))

/-- The distance between two points -/
noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- The main theorem -/
theorem max_distance_on_parabola (A B : Point) :
  A ≠ B →
  onParabola A →
  onParabola B →
  perpendicularBisectorIntersectsAt4 A B →
  distance A B ≤ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_parabola_l381_38153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_plane_sufficient_not_necessary_l381_38133

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes and for lines with planes
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define specific lines and planes
variable (m n : Line) (α β : Plane)

-- State the theorem
theorem parallel_plane_sufficient_not_necessary :
  (m ≠ n) →
  (α ≠ β) →
  (subset m α) →
  (subset n α) →
  (∀ l : Line, subset l α → parallel_planes α β → parallel_line_plane l β) ∧
  (∃ l₁ l₂ : Line, l₁ ≠ l₂ ∧ subset l₁ α ∧ subset l₂ α ∧ 
    parallel_line_plane l₁ β ∧ parallel_line_plane l₂ β ∧ ¬parallel_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_plane_sufficient_not_necessary_l381_38133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_l381_38188

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then Real.log x else (1/2) * x + (1/2)

-- State the theorem
theorem min_difference (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  ∃ (diff : ℝ), diff = n - m ∧ diff ≥ 3 - 2 * Real.log 2 ∧
  ∀ (m' n' : ℝ), m' < n' → f m' = f n' → n' - m' ≥ diff := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_l381_38188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph20_has_hamiltonian_path_l381_38129

/-- A graph with 20 vertices where each vertex has degree at least 10 -/
structure Graph20 where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 10

/-- A Hamiltonian path in a graph -/
def IsHamiltonianPath (g : Graph20) (path : List (Fin 20)) : Prop :=
  path.length = 20 ∧ 
  path.Nodup ∧
  ∀ i, i < 19 → (path[i]!, path[i+1]!) ∈ g.edges

/-- Theorem: A graph with 20 vertices where each vertex has degree at least 10 contains a Hamiltonian path -/
theorem graph20_has_hamiltonian_path (g : Graph20) : ∃ path, IsHamiltonianPath g path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph20_has_hamiltonian_path_l381_38129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_and_triangle_perimeter_l381_38114

/-- Represents an equilateral triangle with a circumscribed circle -/
structure TriangleWithCircumcircle where
  side_length : ℝ
  is_positive : 0 < side_length

/-- Calculates the area of the circumscribed circle of an equilateral triangle -/
noncomputable def circle_area (t : TriangleWithCircumcircle) : ℝ :=
  Real.pi * (t.side_length / Real.sqrt 3) ^ 2

/-- Calculates the perimeter of an equilateral triangle -/
def triangle_perimeter (t : TriangleWithCircumcircle) : ℝ :=
  3 * t.side_length

/-- Theorem stating the area of the circumscribed circle and the perimeter of the triangle -/
theorem circle_area_and_triangle_perimeter 
  (t : TriangleWithCircumcircle) 
  (h : t.side_length = 12) : 
  circle_area t = 48 * Real.pi ∧ triangle_perimeter t = 36 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_and_triangle_perimeter_l381_38114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_line_and_plane_l381_38132

-- Define the direction vector of the line
def d : ℝ × ℝ × ℝ := (4, 5, 7)

-- Define the normal vector of the plane
def n : ℝ × ℝ × ℝ := (6, 3, -2)

-- Define the angle θ
noncomputable def θ : ℝ := Real.arcsin (25 / (21 * Real.sqrt 10))

-- Theorem statement
theorem angle_between_line_and_plane :
  Real.sin θ = 25 / (21 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_line_and_plane_l381_38132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l381_38181

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

def C2 (x y : ℝ) : Prop := (x - 1/2)^2 + y^2 = 1/4

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_C1_C2 :
  ∃ (min_dist : ℝ), min_dist = (Real.sqrt 7 - 1) / 2 ∧
    ∀ (x1 y1 x2 y2 : ℝ),
      C1 x1 y1 → C2 x2 y2 →
        distance x1 y1 x2 y2 ≥ min_dist :=
by
  sorry

#check min_distance_C1_C2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l381_38181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_output_l381_38197

theorem algorithm_output (x : ℝ) : 
  let y := ⌊x⌋
  let z := (2 : ℕ) ^ (y.toNat) - y
  z = 27 → x = 5.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_output_l381_38197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_product_squared_l381_38108

/-- Represents a right triangle with legs a and b, and hypotenuse c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The area of a right triangle -/
noncomputable def area (t : RightTriangle) : ℝ := t.a * t.b / 2

theorem hypotenuse_product_squared
  (T₁ T₂ : RightTriangle)
  (area_T₁ : area T₁ = 4)
  (area_T₂ : area T₂ = 9)
  (hyp_leg_congruent : T₁.c = T₂.a) :
  (T₁.c * T₂.c)^2 = 904 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_product_squared_l381_38108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_value_l381_38116

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Defines the function g(Y) for a given tetrahedron and point Y -/
noncomputable def g (t : Tetrahedron) (Y : Point3D) : ℝ :=
  distance t.A Y + distance t.B Y + distance t.C Y + distance t.D Y

/-- States the theorem about the minimum value of g(Y) for a specific tetrahedron -/
theorem min_g_value (t : Tetrahedron) 
  (h1 : distance t.A t.D = 26) 
  (h2 : distance t.B t.C = 26)
  (h3 : distance t.A t.C = 42)
  (h4 : distance t.B t.D = 42)
  (h5 : distance t.A t.B = 50)
  (h6 : distance t.C t.D = 50) :
  ∃ (Y : Point3D), ∀ (Z : Point3D), g t Y ≤ g t Z ∧ g t Y = 4 * Real.sqrt 650 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_g_value_l381_38116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l381_38118

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a > b > 0, and foci F₁ and F₂, if lines AF₁ and AF₂ intersect 
    the ellipse at points B and C respectively, then the sum of the ratios 
    AF₁/F₁B and AF₂/F₂C equals 4a²/b² - 2. -/
theorem ellipse_ratio_sum (a b : ℝ) (A B C F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 →
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ ((x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t))) →
  F₁ ∈ Set.range (λ t : ℝ × ℝ ↦ t) ∧ F₂ ∈ Set.range (λ t : ℝ × ℝ ↦ t) →
  B ∈ Set.range (λ t : ℝ × ℝ ↦ t) ∧ C ∈ Set.range (λ t : ℝ × ℝ ↦ t) →
  (∃ (t : ℝ), A + t • (F₁ - A) = B) →
  (∃ (s : ℝ), A + s • (F₂ - A) = C) →
  (dist A F₁ / dist F₁ B) + (dist A F₂ / dist F₂ C) = 4 * a^2 / b^2 - 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l381_38118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l381_38193

open Real

theorem triangle_properties (A B C : ℝ) (h : sin A + sin B = sin C * (cos A + cos B)) :
  ∃ (a b c r : ℝ),
    -- Triangle ABC exists
    0 < a ∧ 0 < b ∧ 0 < c ∧
    -- A, B, C are angles of a triangle
    0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π ∧
    -- Part 1: Angle C is 90°
    C = π / 2 ∧
    -- Part 2: If c = 1, the inradius r satisfies 0 < r ≤ (√2 - 1)/2
    (c = 1 →
      -- r is the inradius
      r = (a + b - c) / 2 ∧
      0 < r ∧ r ≤ (Real.sqrt 2 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l381_38193
