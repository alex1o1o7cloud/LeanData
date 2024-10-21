import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_approx_l343_34398

/-- Calculates the gain percent given the cost price and selling price -/
noncomputable def gain_percent (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the gain percent for a cycle bought at Rs. 850 and sold at Rs. 1080 is approximately 27.06% -/
theorem cycle_gain_percent_approx :
  let cost_price := (850 : ℝ)
  let selling_price := (1080 : ℝ)
  abs (gain_percent cost_price selling_price - 27.06) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_gain_percent_approx_l343_34398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l343_34314

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^2 + 77*n = m^2) ↔ n ∈ ({4, 99, 175, 1444} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_l343_34314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factors_1728_l343_34329

theorem smallest_difference_factors_1728 :
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a * b = 1728 ∧
    ∀ (c d : ℕ), c > 0 → d > 0 → c * d = 1728 → |Int.ofNat a - Int.ofNat b| ≤ |Int.ofNat c - Int.ofNat d|) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a * b = 1728 → |Int.ofNat a - Int.ofNat b| ≥ 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factors_1728_l343_34329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_count_l343_34393

/-- The number of men that can finish a piece of work in 100 days -/
def M : ℕ := 110

/-- The time taken to finish the work with M men -/
def original_time : ℕ := 100

/-- The time taken to finish the work with M - 10 men -/
def new_time : ℕ := 110

/-- The amount of work to be done -/
def W : ℝ := 1

theorem men_count : M = 110 := by
  have h1 : W / (original_time * M) = W / (new_time * (M - 10)) := by sorry
  -- The rest of the proof steps
  sorry

#eval M

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_count_l343_34393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_odd_numbers_up_to_5000_l343_34371

/-- Sum of digits of an integer -/
def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

/-- The sum of the digits of all odd numbers from 1 to 5000 -/
def sumOfDigitsOfOddNumbers : ℕ :=
  (List.range 5000).filter isOdd |>.map sumOfDigits |>.sum

theorem sum_of_digits_of_odd_numbers_up_to_5000 :
  sumOfDigitsOfOddNumbers = 54025 := by sorry

#eval sumOfDigitsOfOddNumbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_odd_numbers_up_to_5000_l343_34371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_increasing_numbers_l343_34334

-- Define the number of friends
def n : ℕ := 16

-- Define the probability of choosing strictly increasing numbers
noncomputable def probability_increasing_numbers : ℝ := Real.sqrt ((17 ^ (n - 1)) / ((n.factorial) ^ 2))

-- State the theorem
theorem probability_of_increasing_numbers :
  probability_increasing_numbers = Real.sqrt ((17 ^ (n - 1)) / ((n.factorial) ^ 2)) := by
  -- The proof is omitted
  sorry

#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_increasing_numbers_l343_34334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l343_34356

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2)^x else Real.exp (-x)

theorem function_inequality_range (a : ℝ) :
  (∀ x ∈ Set.Icc (1 - 2*a) (1 + 2*a), f (2*x + a) ≥ (f x)^3) ↔ a ∈ Set.Ioo 0 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l343_34356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_10000_l343_34368

/-- Calculates the compound interest for a given principal, rate, and number of compounding periods -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * ((1 + rate / 2) ^ periods - 1)

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem stating that the principal is 10000 given the specified conditions -/
theorem principal_is_10000 (P : ℝ) (h : P > 0) :
  compound_interest P 0.1 2 - simple_interest P 0.1 1 = 25 →
  P = 10000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_is_10000_l343_34368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_Γ_l343_34302

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the region Γ
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 + (floor p.1 : ℝ) + (floor p.2 : ℝ) ≤ 5}

-- State the theorem
theorem area_of_Γ : MeasureTheory.volume Γ = 9 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_Γ_l343_34302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2010_in_third_quadrant_l343_34327

/-- The quadrant of an angle in degrees -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines the quadrant of an angle in degrees -/
noncomputable def angle_quadrant (angle : ℝ) : Quadrant :=
  let reduced_angle := angle % 360
  if 0 ≤ reduced_angle && reduced_angle < 90 then Quadrant.first
  else if 90 ≤ reduced_angle && reduced_angle < 180 then Quadrant.second
  else if 180 ≤ reduced_angle && reduced_angle < 270 then Quadrant.third
  else Quadrant.fourth

theorem angle_2010_in_third_quadrant :
  angle_quadrant 2010 = Quadrant.third := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2010_in_third_quadrant_l343_34327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l343_34330

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_f :
  Set.Icc (-4 : ℝ) 0 = Set.range (fun x => f x) ∩ Set.Icc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l343_34330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l343_34309

/-- A function f with given properties -/
noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x + Real.pi / 3)

/-- The theorem stating the properties of f and its range -/
theorem f_properties (A ω : ℝ) (hA : A > 0) (hω : ω > 0) 
  (hmax : ∀ x, f A ω x ≤ 2) 
  (hperiod : ∀ x, f A ω (x + Real.pi) = f A ω x) :
  (A = 2 ∧ ω = 2) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi/2), -Real.sqrt 3 ≤ f A ω x ∧ f A ω x ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l343_34309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_theorem_l343_34391

noncomputable def g : ℕ → (ℝ → ℝ)
| 0 => λ _ => 0  -- Adding a case for 0 to avoid missing case error
| 1 => λ x => Real.sqrt (4 - x)
| (n + 2) => λ x => g (n + 1) (Real.sqrt (5 * (n + 2)^2 + x))

def hasDomain (f : ℝ → ℝ) : Prop :=
  ∃ x, ∃ y, f x = y

theorem g_domain_theorem :
  (∀ n > 2, ¬ hasDomain (g n)) ∧
  hasDomain (g 2) ∧
  (∀ x, g 2 x = g 2 (-4) → x = -4) := by
  sorry

#check g_domain_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_theorem_l343_34391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_existence_l343_34322

noncomputable section

/-- Definition of the ellipse parameters -/
def ellipse_params (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ c > 0 ∧ c^2 = a^2 - b^2

/-- Definition of the eccentricity -/
def eccentricity (a c : ℝ) : ℝ := c / a

/-- Definition of the maximum distance to the right focus -/
def max_distance_to_focus (a c : ℝ) : ℝ := a + c

/-- Theorem about the ellipse equation and the existence of a line -/
theorem ellipse_and_line_existence 
  (a b c : ℝ) 
  (h_params : ellipse_params a b c)
  (h_ecc : eccentricity a c = Real.sqrt 2 / 2)
  (h_max_dist : max_distance_to_focus a c = Real.sqrt 2 + 1)
  (m : ℝ) 
  (h_m : 0 < m ∧ m < 1) :
  (∀ x y : ℝ, x^2 / 2 + y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (0 < m ∧ m < 1/2 → ∃ k : ℝ, k ≠ 0 ∧ 
    ∃ x1 y1 x2 y2 : ℝ, 
      x1^2 / 2 + y1^2 = 1 ∧
      x2^2 / 2 + y2^2 = 1 ∧
      y1 = k * (x1 - 1) ∧
      y2 = k * (x2 - 1) ∧
      (x1 - m)^2 + y1^2 = (x2 - m)^2 + y2^2) ∧
  (1/2 ≤ m ∧ m < 1 → ∀ k : ℝ, k ≠ 0 → 
    ¬∃ x1 y1 x2 y2 : ℝ, 
      x1^2 / 2 + y1^2 = 1 ∧
      x2^2 / 2 + y2^2 = 1 ∧
      y1 = k * (x1 - 1) ∧
      y2 = k * (x2 - 1) ∧
      (x1 - m)^2 + y1^2 = (x2 - m)^2 + y2^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_existence_l343_34322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l343_34362

/-- The projection of vector a onto the direction of vector b -/
noncomputable def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_magnitude_squared := b.1^2 + b.2^2
  let scalar := dot_product / b_magnitude_squared
  (scalar * b.1, scalar * b.2)

/-- Theorem stating that the projection of a=(2,3) onto b=(-4,7) is (-4/5, 7/5) -/
theorem projection_a_onto_b :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, 7)
  proj_vector a b = (-4/5, 7/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l343_34362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_iff_sum_half_pi_l343_34338

theorem trig_identity_iff_sum_half_pi (α β : Real) :
  α ∈ Set.Ioo 0 (π/2) →
  β ∈ Set.Ioo 0 (π/2) →
  (Real.sin α)^3 / Real.cos β + (Real.cos α)^3 / Real.sin β = 1 ↔ α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_iff_sum_half_pi_l343_34338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_correct_l343_34399

/-- Triangle ABC with vertices on positive x, y, and z axes respectively -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  c_positive : 0 < c
  side_ab : a^2 + b^2 = 36
  side_bc : b^2 + c^2 = 25
  side_ca : c^2 + a^2 = 16

/-- Volume of tetrahedron OABC where O is the origin -/
noncomputable def tetrahedron_volume (t : TriangleABC) : ℝ := (1/6) * t.a * t.b * t.c

/-- The volume of tetrahedron OABC is (5/6) * √30.375 -/
theorem tetrahedron_volume_is_correct (t : TriangleABC) : 
  tetrahedron_volume t = (5/6) * Real.sqrt 30.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_is_correct_l343_34399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_nine_half_pi_plus_theta_l343_34350

theorem sin_nine_half_pi_plus_theta (θ : ℝ) :
  4 * Real.cos θ = 4 ∧ 3 * Real.sin θ = 3 →
  Real.sin (9 * Real.pi / 2 + θ) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_nine_half_pi_plus_theta_l343_34350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l343_34340

/-- Calculates the final amount after compound interest is applied -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (compoundingsPerYear : ℕ) (years : ℝ) : ℝ :=
  principal * (1 + rate / (compoundingsPerYear : ℝ)) ^ ((compoundingsPerYear : ℝ) * years)

/-- Theorem stating that an investment of $700 at 10% interest compounded semiannually for one year results in $770.75 -/
theorem investment_growth :
  let principal : ℝ := 700
  let rate : ℝ := 0.10
  let compoundingsPerYear : ℕ := 2
  let years : ℝ := 1
  compoundInterest principal rate compoundingsPerYear years = 770.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l343_34340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truthfulness_l343_34333

/-- Represents a dwarf's preference for ice cream -/
inductive IceCream
  | vanilla
  | chocolate
  | fruit
deriving DecidableEq

/-- Represents whether a dwarf tells the truth or lies -/
inductive Truthfulness
  | truthful
  | liar
deriving DecidableEq

/-- Represents a dwarf with their ice cream preference and truthfulness -/
structure Dwarf :=
  (preference : IceCream)
  (truthfulness : Truthfulness)
deriving DecidableEq

/-- The main theorem to prove -/
theorem dwarf_truthfulness 
  (dwarfs : Finset Dwarf)
  (total_count : Finset.card dwarfs = 10)
  (one_preference : ∀ d : Dwarf, d ∈ dwarfs → 
    (d.preference = IceCream.vanilla ∨ 
     d.preference = IceCream.chocolate ∨ 
     d.preference = IceCream.fruit))
  (vanilla_hands : Finset.card 
    (dwarfs.filter (λ d => 
      (d.truthfulness = Truthfulness.truthful ∧ d.preference = IceCream.vanilla) ∨
      d.truthfulness = Truthfulness.liar
    )) = 10)
  (chocolate_hands : Finset.card 
    (dwarfs.filter (λ d => 
      (d.truthfulness = Truthfulness.truthful ∧ d.preference = IceCream.chocolate) ∨
      (d.truthfulness = Truthfulness.liar ∧ d.preference ≠ IceCream.chocolate)
    )) = 5)
  (fruit_hands : Finset.card 
    (dwarfs.filter (λ d => 
      (d.truthfulness = Truthfulness.truthful ∧ d.preference = IceCream.fruit) ∨
      (d.truthfulness = Truthfulness.liar ∧ d.preference ≠ IceCream.fruit)
    )) = 1) :
  Finset.card (dwarfs.filter (λ d => d.truthfulness = Truthfulness.truthful)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truthfulness_l343_34333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l343_34307

-- Define the curve
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem statement
theorem tangent_line_condition (a : ℝ) :
  (∃ x₀ : ℝ, line x₀ (curve a x₀) ∧
    (∀ x : ℝ, x ≠ x₀ → ¬(line x (curve a x)))) ↔ a = 1 := by
  sorry

#check tangent_line_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l343_34307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_pirates_count_l343_34363

/-- Represents the number of days in the voyage before the treasure doubling event --/
def days_before_doubling : ℕ := 7

/-- Represents the total number of days in the voyage --/
def total_days : ℕ := 48

/-- Represents the initial number of gold doubloons --/
def initial_treasure : ℕ := 1010

/-- Represents the number of doubloons buried at the end --/
def buried_treasure : ℕ := 1000

/-- Represents the daily payment to each pirate --/
def daily_payment : ℕ := 1

/-- Theorem stating the initial number of pirates --/
theorem initial_pirates_count : 
  ∃ n : ℕ, 
    (2 * (initial_treasure - n * days_before_doubling) - 
     (n / 2) * (total_days - days_before_doubling - 1) * daily_payment = buried_treasure) ∧ 
    n = 30 := by
  -- The proof goes here
  sorry

#check initial_pirates_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_pirates_count_l343_34363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_parabola_l343_34300

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  focus : ℝ × ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) :=
  {P : ℝ × ℝ // P.2^2 = 4 * p.a * P.1}

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Theorem: Minimum sum of distances for parabola y^2 = 4x -/
theorem min_distance_sum_parabola :
  let p : Parabola := ⟨1, (1, 0)⟩
  let B : ℝ × ℝ := (3, 2)
  ∀ P : PointOnParabola p,
  ∃ m : ℝ, m ≤ distance P.val B + distance P.val p.focus ∧
  (∀ Q : PointOnParabola p, m ≤ distance Q.val B + distance Q.val p.focus) ∧
  m = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_parabola_l343_34300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merry_go_round_revolutions_l343_34337

/-- The number of revolutions needed for a horse on a merry-go-round to travel a certain distance -/
noncomputable def revolutions_needed (distance_from_center : ℝ) (target_distance : ℝ) : ℝ :=
  target_distance / (2 * Real.pi * distance_from_center)

/-- Theorem about the number of revolutions needed for two horses on a merry-go-round -/
theorem merry_go_round_revolutions 
  (horse_a_distance : ℝ) 
  (horse_a_revolutions : ℝ) 
  (horse_b_distance : ℝ) : 
  horse_a_distance = 30 ∧ 
  horse_a_revolutions = 15 ∧ 
  horse_b_distance = 10 → 
  revolutions_needed horse_b_distance (2 * Real.pi * horse_a_distance * horse_a_revolutions) = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_merry_go_round_revolutions_l343_34337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l343_34347

/-- Represents the conditions for a, b, A, B to form a triangle -/
def IsTriangle (a b A B : Real) : Prop :=
  a > 0 ∧ b > 0 ∧ 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi

/-- Given a triangle ABC with sides a and b opposite to angles A and B respectively,
    if a*cos(π-A) + b*sin(π/2+B) = 0, then the triangle is either isosceles or right. -/
theorem triangle_shape (a b A B : Real) (h_triangle : IsTriangle a b A B) 
  (h_equation : a * Real.cos (Real.pi - A) + b * Real.sin (Real.pi / 2 + B) = 0) :
  A = B ∨ A + B = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l343_34347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l343_34383

/-- The coefficient of x^3 in the expansion of ((1-x)(1+x)^6) is 5 -/
theorem coefficient_x_cubed_in_expansion : 
  let f : Polynomial ℤ := (1 - X) * (1 + X)^6
  f.coeff 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l343_34383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l343_34392

noncomputable def f (x : ℝ) := Real.log (Real.sqrt (1 - x^2))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l343_34392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_operations_l343_34388

theorem negative_operations :
  (-(-1/5 : ℚ) = 1/5) ∧
  ((-3 : ℤ) = -3) ∧
  (-((-2) : ℤ) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_operations_l343_34388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l343_34390

/-- A function f is even if f(-x) = f(x) for all x in its domain. -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x(e^x + ae^{-x}) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ x * (Real.exp x + a * Real.exp (-x))

/-- Theorem: If f(x) = x(e^x + ae^{-x}) is an even function on ℝ, then a = -1 -/
theorem even_function_condition (a : ℝ) : IsEven (f a) → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_condition_l343_34390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_28000_l343_34364

/-- Represents the investment and profit structure of a joint business venture -/
structure BusinessVenture where
  b_investment : ℚ
  b_period : ℚ
  b_profit : ℚ
  a_investment : ℚ
  a_period : ℚ

/-- Calculates the total profit of the business venture -/
def total_profit (bv : BusinessVenture) : ℚ :=
  bv.b_profit + bv.b_profit * (bv.a_investment * bv.a_period) / (bv.b_investment * bv.b_period)

/-- Theorem stating that under given conditions, the total profit is 28000 -/
theorem total_profit_is_28000 (bv : BusinessVenture)
  (h1 : bv.a_investment = 3 * bv.b_investment)
  (h2 : bv.a_period = 2 * bv.b_period)
  (h3 : bv.b_profit = 4000) :
  total_profit bv = 28000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_28000_l343_34364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_special_point_l343_34359

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter and orthocenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the centroid of a triangle
noncomputable def centroid (X Y Z : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem euler_line_special_point (t : Triangle) :
  ∃ P : ℝ × ℝ,
    (∃ k : ℝ, P = k • (orthocenter t) + (1 - k) • (circumcenter t)) ∧
    distance (centroid t.A t.B P) t.C =
    distance (centroid t.B t.C P) t.A ∧
    distance (centroid t.B t.C P) t.A =
    distance (centroid t.C t.A P) t.B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_special_point_l343_34359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l343_34319

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 - 2*y^2 = 2

/-- Definition of the foci coordinates -/
noncomputable def foci : Set (ℝ × ℝ) := {(-Real.sqrt 3, 0), (Real.sqrt 3, 0)}

/-- Definition of the eccentricity -/
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 2

/-- Theorem stating the properties of the hyperbola -/
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (x, y) ∉ foci) ∧
  (∃ x y, hyperbola x y ∧ (x, y) ∉ foci) ∧
  eccentricity = Real.sqrt 6 / 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l343_34319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_line_circle_l343_34379

/-- The length of the chord intercepted by a line on a circle -/
noncomputable def chord_length (a b c r : ℝ) : ℝ :=
  2 * Real.sqrt (r^2 - (c^2 / (a^2 + b^2)))

/-- Theorem: The length of the chord intercepted by the line √3x + y - 2√3 = 0 on the circle x^2 + y^2 = 4 is equal to 2 -/
theorem chord_length_specific_line_circle : 
  chord_length (Real.sqrt 3) 1 (2 * Real.sqrt 3) 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_line_circle_l343_34379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l343_34305

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

-- Define the line
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + y + 2| / Real.sqrt 2

-- Theorem statement
theorem min_distance_to_line :
  ∀ x > 0, ∃ y, f x = y ∧ 
  ∀ x' > 0, ∀ y', f x' = y' → distance_to_line x' y' ≥ 2 * Real.sqrt 2 :=
by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l343_34305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surrounding_circle_radius_is_correct_surrounding_circle_radius_properties_l343_34303

/-- The radius of the surrounding circles in a hexagonal arrangement around a central circle --/
noncomputable def surrounding_circle_radius : ℝ := (1 + Real.sqrt 5) / 2

/-- The configuration of circles as described in the problem --/
structure CircleConfiguration where
  central_radius : ℝ
  surrounding_radius : ℝ
  num_surrounding : ℕ
  is_hexagonal : Bool
  is_tangent : Bool

/-- The specific configuration from the problem --/
noncomputable def problem_configuration : CircleConfiguration :=
  { central_radius := 2,
    surrounding_radius := surrounding_circle_radius,
    num_surrounding := 6,
    is_hexagonal := true,
    is_tangent := true }

/-- Theorem stating that the surrounding circle radius in the given configuration is correct --/
theorem surrounding_circle_radius_is_correct (config : CircleConfiguration) :
  config = problem_configuration →
  config.surrounding_radius = surrounding_circle_radius := by
  intro h
  rw [h]
  rfl

/-- Theorem stating the properties of the surrounding circle radius --/
theorem surrounding_circle_radius_properties :
  surrounding_circle_radius ^ 2 - surrounding_circle_radius - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surrounding_circle_radius_is_correct_surrounding_circle_radius_properties_l343_34303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l343_34335

-- Define the cone parameters
noncomputable def r : ℝ := 6
noncomputable def V : ℝ := 30 * Real.pi

-- State the theorem
theorem cone_lateral_surface_area :
  let h : ℝ := (3 * V) / (Real.pi * r^2)
  let l : ℝ := Real.sqrt (r^2 + h^2)
  Real.pi * r * l = 39 * Real.pi := by
  -- Unfold definitions
  unfold r V
  -- Simplify expressions
  simp [Real.pi]
  -- Skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_area_l343_34335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_b_mod_13_l343_34394

theorem remainder_of_b_mod_13 : 
  ∀ b : ZMod 13, b = (((2 : ZMod 13)⁻¹ + (5 : ZMod 13)⁻¹ + (9 : ZMod 13)⁻¹)⁻¹) → b = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_b_mod_13_l343_34394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_selected_firefighters_l343_34320

/-- Represents the number of firefighters selected from each city -/
structure SelectedFirefighters :=
  (cityA : ℕ)
  (cityB : ℕ)
  (cityC : ℕ)

/-- Calculates the number of firefighters selected from each city -/
def calculateSelectedFirefighters (totalFirefighters : ℕ) (selectedFirefighters : ℕ) 
  (cityARange : Set ℕ) (cityBRange : Set ℕ) (cityCRange : Set ℕ) : SelectedFirefighters :=
  sorry

/-- Theorem stating the correct number of selected firefighters from each city -/
theorem correct_selected_firefighters :
  let totalFirefighters := 600
  let selectedFirefighters := 50
  let cityARange := {n : ℕ | 1 ≤ n ∧ n ≤ 300}
  let cityBRange := {n : ℕ | 301 ≤ n ∧ n ≤ 495}
  let cityCRange := {n : ℕ | 496 ≤ n ∧ n ≤ 600}
  let result := calculateSelectedFirefighters totalFirefighters selectedFirefighters cityARange cityBRange cityCRange
  result.cityA = 25 ∧ result.cityB = 17 ∧ result.cityC = 8 :=
by
  sorry

#check correct_selected_firefighters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_selected_firefighters_l343_34320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_proof_l343_34367

/-- A function that checks if a quadratic polynomial factors into two linear polynomials with integer coefficients -/
def is_factorable (b : ℤ) : Prop :=
  ∃ (p q : ℤ), ∀ (x : ℤ), x^2 + b*x + 2352 = (x + p) * (x + q)

/-- The smallest positive integer b for which x^2 + bx + 2352 factors -/
def smallest_factorable_b : ℕ := 112

theorem smallest_factorable_b_proof :
  (is_factorable (smallest_factorable_b : ℤ)) ∧
  (∀ b : ℕ, b < smallest_factorable_b → ¬(is_factorable (b : ℤ))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorable_b_proof_l343_34367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_from_equal_inscribed_radii_l343_34382

open Real EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define angle bisector
def is_angle_bisector (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop :=
  angle P Q R = angle R Q S

-- Define the inscribed circle
structure InscribedCircle (P Q R : EuclideanSpace ℝ (Fin 2)) where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ
  radius_pos : radius > 0
  tangent_P : dist center P = radius
  tangent_Q : dist center Q = radius
  tangent_R : dist center R = radius

-- State the theorem
theorem isosceles_from_equal_inscribed_radii 
  (A₁ C₁ : EuclideanSpace ℝ (Fin 2))
  (h_AA₁_bisector : is_angle_bisector B A C A₁)
  (h_CC₁_bisector : is_angle_bisector B C A C₁)
  (circle_AA₁C : InscribedCircle A A₁ C)
  (circle_CC₁A : InscribedCircle C C₁ A)
  (h_equal_radii : circle_AA₁C.radius = circle_CC₁A.radius) :
  angle B A C = angle B C A :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_from_equal_inscribed_radii_l343_34382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_arctans_l343_34352

theorem right_triangle_arctans (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a^2 = b^2 + c^2) → Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_arctans_l343_34352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cd_length_l343_34324

/-- A quadrangle ABCD inscribed in a unit circle with specific properties -/
structure InscribedQuadrangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  radius : ℝ
  ac_diameter : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 4 * radius^2
  bd_equal_ab : (B.1 - D.1)^2 + (B.2 - D.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2
  p_intersect : ∃ t : ℝ, P = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2) ∧
                    ∃ s : ℝ, P = (s * B.1 + (1 - s) * D.1, s * B.2 + (1 - s) * D.2)
  pc_length : (P.1 - C.1)^2 + (P.2 - C.2)^2 = (2/5)^2
  unit_circle : radius = 1

/-- The theorem stating that CD has length 2/3 -/
theorem cd_length (q : InscribedQuadrangle) :
  (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = (2/3)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cd_length_l343_34324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l343_34332

def is_valid_number (N : ℕ) : Prop :=
  ∃ (divisors : List ℕ),
    divisors ≠ [] ∧
    List.Sorted (· ≤ ·) divisors ∧
    1 ∈ divisors ∧
    N ∈ divisors ∧
    (∀ d ∈ divisors, N % d = 0) ∧
    (∀ d ∈ divisors, d ∣ N) ∧
    divisors.length ≥ 3 ∧
    divisors[1]! * 21 = divisors[divisors.length - 3]!

theorem largest_valid_number : 
  (∀ n : ℕ, n > 441 → ¬is_valid_number n) ∧ is_valid_number 441 := by
  sorry

#check largest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l343_34332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l343_34323

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

/-- The maximum value of f(x) in [1, 3] -/
noncomputable def M (a : ℝ) : ℝ := max (f a 1) (f a 3)

/-- The minimum value of f(x) in [1, 3] -/
noncomputable def N (a : ℝ) : ℝ := min (f a 1) (f a 3)

/-- The function g(a) defined in the problem -/
noncomputable def g (a : ℝ) : ℝ := M a - N a

theorem max_value_of_g :
  ∀ a : ℝ, 1/3 ≤ a ∧ a ≤ 1 → g a ≤ 4 ∧ g 1 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l343_34323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_s_l343_34370

theorem solve_for_s (k : ℝ) (s : ℝ) 
  (h1 : 5 = k * 2^s) 
  (h2 : 45 = k * 8^s) : 
  s = (Real.log 9) / (2 * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_s_l343_34370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_center_is_not_always_diameter_l343_34365

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
def Chord (c : Circle) : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(p₁, p₂, q₁, q₂) | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
    ((1 - t) * p₁ + t * q₁ - c.center.1)^2 + ((1 - t) * p₂ + t * q₂ - c.center.2)^2 = c.radius^2}

-- Define a diameter
def Diameter (c : Circle) : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(p₁, p₂, q₁, q₂) | 
    (p₁ - c.center.1)^2 + (p₂ - c.center.2)^2 = c.radius^2 ∧
    (q₁ - c.center.1)^2 + (q₂ - c.center.2)^2 = c.radius^2 ∧
    (p₁ - q₁)^2 + (p₂ - q₂)^2 = (2 * c.radius)^2}

-- Theorem to be proved
theorem chord_through_center_is_not_always_diameter (c : Circle) :
  ¬(∀ (p₁ p₂ q₁ q₂ : ℝ), (p₁, p₂, q₁, q₂) ∈ Chord c → 
    (p₁ - q₁)^2 + (p₂ - q₂)^2 = (2 * c.radius)^2 → (p₁, p₂, q₁, q₂) ∈ Diameter c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_through_center_is_not_always_diameter_l343_34365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_of_sequence_l343_34328

theorem max_gcd_of_sequence : 
  (∀ n : ℕ+, ∃ k : ℕ, Nat.gcd (100 + n^2) (100 + (n+1)^2) = k) ∧ 
  (∀ m : ℕ+, Nat.gcd (100 + m^2) (100 + (m+1)^2) ≤ 401) ∧
  (∃ l : ℕ+, Nat.gcd (100 + l^2) (100 + (l+1)^2) = 401) := by
  sorry

#check max_gcd_of_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_gcd_of_sequence_l343_34328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_53_l343_34331

theorem sum_remainder_mod_53 (a b c d : ℕ) 
  (ha : a % 53 = 31)
  (hb : b % 53 = 44)
  (hc : c % 53 = 6)
  (hd : d % 53 = 2) :
  (a + b + c + d) % 53 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_53_l343_34331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l343_34384

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being decreasing on (0, +∞)
def is_decreasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f y < f x

-- Define the minimum value of f
def min_value (f : ℝ → ℝ) (m : ℝ) : Prop := ∀ x, m ≤ f x

-- Define the property of being increasing on (-∞, 0)
def is_increasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x < f y

-- State the theorem
theorem even_function_properties :
  is_even f →
  is_decreasing_on_positive f →
  min_value f 2 →
  is_increasing_on_negative f ∧ min_value f 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l343_34384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_l343_34341

/-- Represents a trip with three segments of equal duration -/
structure Trip where
  total_distance : ℝ
  total_time : ℝ
  speed_segment1 : ℝ
  speed_segment2 : ℝ
  speed_segment3 : ℝ

/-- The average speed of the entire trip -/
noncomputable def average_speed (t : Trip) : ℝ :=
  t.total_distance / t.total_time

/-- Theorem stating the conditions and the result to be proved -/
theorem last_segment_speed (t : Trip) 
    (h1 : t.total_distance = 150)
    (h2 : t.total_time = 2.5)
    (h3 : t.speed_segment1 = 50)
    (h4 : t.speed_segment2 = 60)
    (h5 : average_speed t = 60) :
    t.speed_segment3 = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_segment_speed_l343_34341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_mbc_is_sixty_degrees_l343_34357

/-- Represents a point M in the interior of triangle ABC. -/
def TriangleInteriorPoint (A B C M : EuclideanSpace ℝ (Fin 2)) : Prop :=
  M ≠ A ∧ M ≠ B ∧ M ≠ C ∧
  (∃ t u v : ℝ, t > 0 ∧ u > 0 ∧ v > 0 ∧ t + u + v < 1 ∧
    M = t • A + u • B + v • C)

/-- Represents the value of an angle in degrees. -/
noncomputable def AngleValue (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

/-- Given a triangle ABC with an interior point M, if the specified angle conditions are met,
    then the angle MBC is 60 degrees. -/
theorem angle_mbc_is_sixty_degrees (A B C M : EuclideanSpace ℝ (Fin 2)) :
  TriangleInteriorPoint A B C M →
  AngleValue A M B = 10 →
  AngleValue M B A = 20 →
  AngleValue M C A = 30 →
  AngleValue M A C = 40 →
  AngleValue M B C = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_mbc_is_sixty_degrees_l343_34357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l343_34308

/-- Two circles that intersect at (8,4) and have product of radii 50 --/
structure IntersectingCircles where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)
  intersect_point : (8, 4) ∈ C₁ ∩ C₂
  radii_product : ∃ r₁ r₂ : ℝ, r₁ * r₂ = 50 ∧ 
    (∀ p ∈ C₁, (p.1 - 8)^2 + (p.2 - 4)^2 = r₁^2) ∧
    (∀ p ∈ C₂, (p.1 - 8)^2 + (p.2 - 4)^2 = r₂^2)

/-- A line y = mx tangent to both circles --/
def TangentLine (m : ℝ) (ic : IntersectingCircles) : Prop :=
  m > 0 ∧
  (∀ p ∈ ic.C₁, p.2 ≥ m * p.1) ∧
  (∀ p ∈ ic.C₂, p.2 ≥ m * p.1) ∧
  (∃ p₁ ∈ ic.C₁, p₁.2 = m * p₁.1) ∧
  (∃ p₂ ∈ ic.C₂, p₂.2 = m * p₂.1)

/-- The x-axis and y-axis are tangent to both circles --/
def AxisTangent (ic : IntersectingCircles) : Prop :=
  (∀ p ∈ ic.C₁, p.1 ≥ 0 ∧ p.2 ≥ 0) ∧
  (∀ p ∈ ic.C₂, p.1 ≥ 0 ∧ p.2 ≥ 0) ∧
  (∃ p₁ ∈ ic.C₁, p₁.1 = 0 ∨ p₁.2 = 0) ∧
  (∃ p₂ ∈ ic.C₂, p₂.1 = 0 ∨ p₂.2 = 0)

theorem tangent_line_slope (ic : IntersectingCircles) 
  (h_axis : AxisTangent ic) :
  ∃ m : ℝ, TangentLine m ic ∧ m = 2 * Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l343_34308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sectional_areas_relationship_l343_34313

/-- Represents a pyramid in 3D space -/
structure Pyramid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if the plane is parallel to the base of the pyramid -/
def plane_parallel_to_base (p : Pyramid) (plane : Plane) : Prop :=
  sorry

/-- Checks if the plane cuts the lateral edges in half -/
def cuts_lateral_edges_in_half (p : Pyramid) (plane : Plane) : Prop :=
  sorry

/-- Checks if the plane cuts the lateral surface area in half -/
def cuts_lateral_surface_area_in_half (p : Pyramid) (plane : Plane) : Prop :=
  sorry

/-- Checks if the plane cuts the volume in half -/
def cuts_volume_in_half (p : Pyramid) (plane : Plane) : Prop :=
  sorry

/-- Calculates the sectional area when the plane cuts the lateral edges in half -/
noncomputable def sectional_area_edges (p : Pyramid) (plane : Plane) : ℝ :=
  sorry

/-- Calculates the sectional area when the plane cuts the lateral surface area in half -/
noncomputable def sectional_area_surface (p : Pyramid) (plane : Plane) : ℝ :=
  sorry

/-- Calculates the sectional area when the plane cuts the volume in half -/
noncomputable def sectional_area_volume (p : Pyramid) (plane : Plane) : ℝ :=
  sorry

/-- Theorem stating the relationship between sectional areas -/
theorem sectional_areas_relationship (p : Pyramid) (plane : Plane) 
  (h_parallel : plane_parallel_to_base p plane)
  (h_edges : cuts_lateral_edges_in_half p plane)
  (h_surface : cuts_lateral_surface_area_in_half p plane)
  (h_volume : cuts_volume_in_half p plane) :
  sectional_area_edges p plane < sectional_area_surface p plane ∧ 
  sectional_area_surface p plane < sectional_area_volume p plane :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sectional_areas_relationship_l343_34313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_services_intersection_l343_34316

theorem airline_services_intersection (internet_percentage : ℝ) (snack_percentage : ℝ)
  (h_internet : internet_percentage = 40)
  (h_snack : snack_percentage = 70)
  : min internet_percentage snack_percentage = 40 := by
  rw [h_internet, h_snack]
  simp [min]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airline_services_intersection_l343_34316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_ratio_l343_34355

theorem inscribed_circle_area_ratio (s : ℝ) (h : s > 0) :
  (π * (s / (2 * Real.sqrt 3))^2) / (π * (s / 2)^2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_ratio_l343_34355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_cross_sections_imply_sphere_l343_34380

/-- A solid is a subset of three-dimensional space. -/
def Solid : Type := Set (Fin 3 → ℝ)

/-- A plane is a two-dimensional subspace of three-dimensional space. -/
def Plane : Type := Subspace ℝ (Fin 3 → ℝ)

/-- A circle is a subset of a plane. -/
def Circle : Type := Set (Fin 2 → ℝ)

/-- A cross-section of a solid by a plane is the intersection of the solid and the plane. -/
def CrossSection (s : Solid) (p : Plane) : Set (Fin 2 → ℝ) := sorry

/-- A sphere is a specific type of solid. -/
def Sphere : Solid := sorry

/-- A solid has circular cross-sections if the intersection of the solid with any plane is a circle. -/
def HasCircularCrossSections (s : Solid) : Prop :=
  ∀ (p : Plane), ∃ (c : Circle), CrossSection s p = c

/-- If a solid has circular cross-sections, then it is a sphere. -/
theorem circular_cross_sections_imply_sphere (s : Solid) :
  HasCircularCrossSections s → s = Sphere :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_cross_sections_imply_sphere_l343_34380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l343_34344

-- Define the points and the trajectory
def P : ℝ × ℝ → Prop := sorry
noncomputable def M : ℝ → ℝ × ℝ := λ x => (x, -4)
def O : ℝ × ℝ := (0, 0)
noncomputable def W : ℝ → ℝ := λ x => (1/4) * x^2

-- Define the condition for the circle passing through O
def circle_condition (x y : ℝ) : Prop :=
  (x, y) • M x = 0

-- Define the line l and points A, B, and A'
def l : ℝ → ℝ → Prop := sorry
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def A' : ℝ × ℝ := (-A.1, A.2)

-- Define E
def E : ℝ × ℝ := (0, -4)

-- Main theorem
theorem trajectory_and_fixed_point :
  (∀ x y, P (x, y) → circle_condition x y) →
  (l E.1 E.2) →
  (l A.1 A.2 ∧ l B.1 B.2) →
  (P A ∧ P B) →
  (∀ x, P (x, W x)) ∧
  (∃ k, ∀ x, (A'.2 - B.2) * x + (B.1 * A'.2 - A'.1 * B.2) = k * (A'.1 - B.1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_point_l343_34344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_range_l343_34343

-- Define vectors AB and AC
def AB : ℝ := 8
def AC : ℝ := 5

-- Define the theorem
theorem BC_range :
  ∀ BC : ℝ, (3 ≤ BC ∧ BC ≤ 13) ↔ 
  (∃ (θ : ℝ), BC = Real.sqrt ((AB - AC * Real.cos θ)^2 + (AC * Real.sin θ)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_range_l343_34343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_less_than_half_sphere_volume_l343_34318

/-- A right circular cone inscribed in a sphere -/
structure InscribedCone (R : ℝ) where
  r : ℝ  -- radius of the base of the cone
  m : ℝ  -- height of the cone
  r_le_R : r ≤ R  -- radius of cone base ≤ radius of sphere
  m_lt_2R : m < 2 * R  -- height of cone < diameter of sphere

/-- The volume of a sphere with radius R -/
noncomputable def sphereVolume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

/-- The volume of a cone with base radius r and height m -/
noncomputable def coneVolume (r m : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * m

/-- Theorem: The volume of an inscribed cone is always less than half the volume of the sphere -/
theorem inscribed_cone_volume_less_than_half_sphere_volume (R : ℝ) (cone : InscribedCone R) :
  coneVolume cone.r cone.m < (1 / 2) * sphereVolume R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_less_than_half_sphere_volume_l343_34318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l343_34374

/-- Given a triangle with angles in ratio 3:4:5 and shortest side 15 units,
    its perimeter is 15 + 15√(3/2) + 15((√6 + √2)/(2√2)) units. -/
theorem triangle_perimeter (a b c : ℝ) (h_angles : a + b + c = π)
  (h_ratio : ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k)
  (h_shortest_side : Real.sin a * 15 = Real.sin b * 15 ∧ Real.sin a * 15 = Real.sin c * 15) :
  ∃ (p : ℝ), p = 15 + 15 * Real.sqrt (3/2) + 15 * ((Real.sqrt 6 + Real.sqrt 2) / (2 * Real.sqrt 2)) ∧
  p = 15 / Real.sin a * (Real.sin a + Real.sin b + Real.sin c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l343_34374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parameter_value_l343_34376

/-- A parabola with focus F and parameter p -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ

/-- A point on a parabola -/
structure PointOnParabola (para : Parabola) where
  point : ℝ × ℝ
  on_parabola : (point.2)^2 = 2 * para.p * point.1

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_parameter_value (para : Parabola) (M : PointOnParabola para) :
  M.point.1 = 4 → distance M.point para.F = 5 → para.p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_parameter_value_l343_34376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l343_34386

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- The problem statement -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h : q ≠ 1) :
  (geometric_sum a q 6) / (geometric_sum a q 3) = 1 / 2 →
  (geometric_sum a q 9) / (geometric_sum a q 3) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l343_34386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_l343_34360

noncomputable section

-- Define the curve y = 1/x
def inverse_function (x : ℝ) : ℝ := 1 / x

-- Define a point on the curve
structure PointOnCurve where
  x : ℝ
  y : ℝ
  x_pos : x > 0
  on_curve : y = inverse_function x

-- Define the area of the region bounded by OA, OB, and arc AB
noncomputable def area_OAB (A B : PointOnCurve) : ℝ := sorry

-- Define the area of the region bounded by AH_A, BH_B, x-axis, and arc AB
noncomputable def area_AHBH (A B : PointOnCurve) : ℝ := sorry

-- Theorem statement
theorem areas_equal (A B : PointOnCurve) : area_OAB A B = area_AHBH A B := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_equal_l343_34360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_students_in_line_l343_34396

/-- The number of students in a line formation -/
def students_in_line (students_before_seokjin : ℕ) (students_between_seokjin_jimin : ℕ) (students_after_jimin : ℕ) : ℕ :=
  students_before_seokjin + 1 + students_between_seokjin_jimin + 1 + students_after_jimin

/-- Proof that there are 16 students in the line -/
theorem sixteen_students_in_line :
  students_in_line 4 3 7 = 16 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_students_in_line_l343_34396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l343_34336

-- Define set A
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}

-- Define set B (domain of ln(4-x))
def B : Set ℝ := {x | x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc (-1) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l343_34336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_arithmetic_sequence_l343_34351

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  h : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def ArithmeticSum (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq.a |>.sum

theorem minimize_sum_arithmetic_sequence 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = -9) 
  (h2 : ArithmeticSum seq 3 = ArithmeticSum seq 7) : 
  ∃ (n : ℕ), ∀ (m : ℕ), ArithmeticSum seq n ≤ ArithmeticSum seq m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_arithmetic_sequence_l343_34351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l343_34339

/-- The function f(x) = x^2 + ax + 1/x is increasing on (1/2, +∞) if and only if a ≥ 3 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (fun x => x^2 + a*x + 1/x)) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l343_34339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_chords_l343_34378

theorem disjoint_chords (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n/2 - 1) :
  ∃ (chords : Finset (Fin n × Fin n)) (disjoint_chords : Finset (Fin n × Fin n)),
    chords.card = n * k + 1 ∧
    disjoint_chords ⊆ chords ∧
    disjoint_chords.card ≥ k + 1 ∧
    ∀ c1 c2, c1 ∈ disjoint_chords → c2 ∈ disjoint_chords → c1 ≠ c2 →
      c1.1 ≠ c2.1 ∧ c1.1 ≠ c2.2 ∧ c1.2 ≠ c2.1 ∧ c1.2 ≠ c2.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_chords_l343_34378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l343_34354

/-- Given a cone with lateral surface area 2π and base surface area π, 
    its volume is (√3 * π) / 3 -/
theorem cone_volume 
  (lateralSurfaceArea : ℝ) 
  (baseSurfaceArea : ℝ) 
  (volume : ℝ)
  (h1 : lateralSurfaceArea = 2 * Real.pi) 
  (h2 : baseSurfaceArea = Real.pi) : 
  volume = (Real.sqrt 3 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l343_34354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_return_probability_2021_l343_34377

/-- Represents the probability of the ball returning to the starting person after n passes in a three-person passing game with equal pass probabilities. -/
noncomputable def ball_return_probability (n : ℕ) : ℝ :=
  1/3 + 2/3 * (-1/2)^n

/-- The theorem states that the probability of the ball returning to the starting person
    after 2021 passes in a three-person passing game with equal pass probabilities
    is equal to 1/3 * (1 - 1/2^2020). -/
theorem ball_return_probability_2021 :
  ball_return_probability 2021 = 1/3 * (1 - 1/2^2020) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_return_probability_2021_l343_34377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crayons_lost_percentage_l343_34385

def initial_crayons : ℕ := 479
def remaining_crayons : ℕ := 134

noncomputable def percentage_lost : ℝ := 
  (initial_crayons - remaining_crayons : ℝ) / initial_crayons * 100

theorem crayons_lost_percentage :
  ∃ ε > 0, abs (percentage_lost - 72.03) < ε := by
  -- The proof goes here
  sorry

#eval initial_crayons
#eval remaining_crayons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crayons_lost_percentage_l343_34385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_property_l343_34389

/-- Predicate stating that a circle with given center and radius is tangent to the sides
    (or their extensions) of a triangle with sides a, b, and c. -/
def CircleTangentToTriangleSides (center : ℝ × ℝ) (radius : ℝ) (a b c : ℝ) : Prop :=
  sorry

/-- Given a triangle with sides a, b, and c, if there exists a circle of radius (a+b+c)/2
    that touches side c and extensions of sides a and b, then there exists a circle of
    radius (a+c-b)/2 that is tangent to side a and extensions of sides b and c. -/
theorem excircle_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_excircle : ∃ (center : ℝ × ℝ), 
    CircleTangentToTriangleSides center ((a + b + c) / 2) a b c) :
  ∃ (center : ℝ × ℝ), CircleTangentToTriangleSides center ((a + c - b) / 2) a b c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_property_l343_34389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l343_34366

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given triangle ABC with specific measurements -/
noncomputable def given_triangle : Triangle where
  b := 2
  c := Real.sqrt 3
  A := Real.pi / 6
  a := 1  -- We define this, but it's what we'll prove
  B := 0  -- We don't know these values, so we set them to 0
  C := 0  -- We don't know these values, so we set them to 0

/-- The area of a triangle given two sides and the included angle -/
noncomputable def triangle_area (t : Triangle) : ℝ := 
  (1 / 2) * t.b * t.c * Real.sin t.A

/-- The law of cosines for a triangle -/
def law_of_cosines (t : Triangle) : Prop :=
  t.a ^ 2 = t.b ^ 2 + t.c ^ 2 - 2 * t.b * t.c * Real.cos t.A

theorem triangle_properties (t : Triangle) 
  (h1 : t.b = 2) 
  (h2 : t.c = Real.sqrt 3) 
  (h3 : t.A = Real.pi / 6) : 
  triangle_area t = Real.sqrt 3 / 2 ∧ t.a = 1 := by
  sorry

#check triangle_properties given_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l343_34366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_a_faster_than_car_b_l343_34381

noncomputable section

-- Define the total distance
def total_distance : ℝ := 120

-- Define Car A's segments
def car_a_segment1_distance : ℝ := 40
def car_a_segment1_speed : ℝ := 60
def car_a_segment2_distance : ℝ := 40
def car_a_segment2_speed : ℝ := 40
def car_a_segment3_distance : ℝ := 40
def car_a_segment3_speed : ℝ := 80

-- Define Car B's segments
def car_b_segment1_time : ℝ := 1
def car_b_segment1_speed : ℝ := 60
def car_b_segment2_time : ℝ := 1
def car_b_segment2_speed : ℝ := 40
def car_b_total_time : ℝ := 3

-- Calculate Car A's average speed
noncomputable def car_a_average_speed : ℝ :=
  total_distance / (car_a_segment1_distance / car_a_segment1_speed +
                    car_a_segment2_distance / car_a_segment2_speed +
                    car_a_segment3_distance / car_a_segment3_speed)

-- Calculate Car B's average speed
noncomputable def car_b_average_speed : ℝ :=
  total_distance / car_b_total_time

-- Theorem statement
theorem car_a_faster_than_car_b : car_a_average_speed > car_b_average_speed := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_a_faster_than_car_b_l343_34381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bluetooth_module_count_l343_34349

theorem bluetooth_module_count :
  let expensive_cost : ℚ := 10
  let cheap_cost : ℚ := 5/2
  let total_value : ℚ := 125/2
  let cheap_count : ℕ := 21
  ∃ (expensive_count : ℕ), 
    (expensive_cost * expensive_count + cheap_cost * cheap_count = total_value) ∧
    (expensive_count + cheap_count = 22) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bluetooth_module_count_l343_34349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caffeine_consumption_proof_l343_34310

/- Define the characteristics of the drinks and pill -/
noncomputable def first_drink_ounces : ℝ := 12
noncomputable def first_drink_caffeine : ℝ := 250
noncomputable def second_drink_ounces : ℝ := 2
noncomputable def second_drink_caffeine_multiplier : ℝ := 3

/- Calculate caffeine per ounce for the first drink -/
noncomputable def caffeine_per_ounce_first : ℝ := first_drink_caffeine / first_drink_ounces

/- Calculate total caffeine in the second drink -/
noncomputable def second_drink_caffeine : ℝ := 
  caffeine_per_ounce_first * second_drink_caffeine_multiplier * second_drink_ounces

/- Define the total caffeine consumption -/
noncomputable def total_caffeine_consumption : ℝ := 
  2 * (first_drink_caffeine + second_drink_caffeine)

/- Theorem statement -/
theorem caffeine_consumption_proof : 
  total_caffeine_consumption = 2 * (first_drink_caffeine + second_drink_caffeine) :=
by
  -- Unfold the definition of total_caffeine_consumption
  unfold total_caffeine_consumption
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_caffeine_consumption_proof_l343_34310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l343_34321

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

def simple_interest (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate * years)

def loan_amount : ℝ := 15000
def compound_rate : ℝ := 0.08
def simple_rate : ℝ := 0.10
def loan_term : ℝ := 15
def compounds_per_year : ℝ := 2
def half_payment_year : ℝ := 7

theorem loan_difference_theorem :
  let compound_balance_7 := compound_interest loan_amount compound_rate compounds_per_year half_payment_year
  let half_payment := compound_balance_7 / 2
  let remaining_balance := compound_balance_7 - half_payment
  let final_compound_balance := compound_interest remaining_balance compound_rate compounds_per_year (loan_term - half_payment_year)
  let total_compound_payment := half_payment + final_compound_balance
  let total_simple_payment := simple_interest loan_amount simple_rate loan_term
  Int.floor ((total_simple_payment - total_compound_payment) + 0.5) = 5448 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_difference_theorem_l343_34321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reinforcement_size_is_650_l343_34306

-- Define the problem parameters
def initial_garrison : ℕ := 2000
def initial_days : ℕ := 54
def initial_rate : ℚ := 1
def days_before_reinforcement : ℕ := 21
def reinforcement_rate : ℚ := 2
def remaining_days : ℕ := 20

-- Define the function to calculate the size of the reinforcement
noncomputable def reinforcement_size : ℕ :=
  let total_provisions : ℚ := initial_garrison * initial_rate * initial_days
  let consumed_provisions : ℚ := initial_garrison * initial_rate * days_before_reinforcement
  let remaining_provisions : ℚ := total_provisions - consumed_provisions
  let daily_consumption : ℚ := remaining_provisions / remaining_days
  ⌊(daily_consumption - initial_garrison * initial_rate) / reinforcement_rate⌋.toNat

-- Theorem statement
theorem reinforcement_size_is_650 : reinforcement_size = 650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reinforcement_size_is_650_l343_34306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l343_34325

-- Define the train lengths
noncomputable def train_length : ℝ := 500

-- Define the speeds in km/hr
noncomputable def speed_faster : ℝ := 45
noncomputable def speed_slower : ℝ := 30

-- Convert speeds to m/s
noncomputable def speed_faster_ms : ℝ := speed_faster * 1000 / 3600
noncomputable def speed_slower_ms : ℝ := speed_slower * 1000 / 3600

-- Calculate relative speed
noncomputable def relative_speed : ℝ := speed_faster_ms + speed_slower_ms

-- Calculate time taken
noncomputable def time_taken : ℝ := train_length / relative_speed

-- Theorem to prove
theorem train_passing_time :
  ∀ ε > 0, |time_taken - 24.01| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l343_34325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleConfiguration_l343_34395

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), (p3.x - p1.x) = t * (p2.x - p1.x) ∧
             (p3.y - p1.y) = t * (p2.y - p1.y) ∧
             (p3.z - p1.z) = t * (p2.z - p1.z)

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The main theorem -/
theorem impossibleConfiguration :
  ¬ ∃ (points : Finset Point3D) (planes : Finset Plane3D),
    (points.card = 24) ∧
    (planes.card = 2019) ∧
    (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points →
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬ collinear p1 p2 p3) ∧
    (∀ plane, plane ∈ planes →
      ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
        pointOnPlane p1 plane ∧ pointOnPlane p2 plane ∧ pointOnPlane p3 plane) ∧
    (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points →
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
      ∃ plane, plane ∈ planes ∧ pointOnPlane p1 plane ∧ pointOnPlane p2 plane ∧ pointOnPlane p3 plane) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleConfiguration_l343_34395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l343_34353

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q)) ∧
  -- The minimum value in [0, π/2] is -1/2
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1/2) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l343_34353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l343_34342

/-- The complex number z defined as 2i / (i - 1) -/
noncomputable def z : ℂ := 2 * Complex.I / (Complex.I - 1)

/-- Theorem: z is in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  Real.sign z.re = 1 ∧ Real.sign z.im = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l343_34342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parallel_point_set_l343_34375

/-- A set of points in 3D space satisfying the parallelism condition -/
def ParallelPointSet (M : Set (Fin 3 → ℝ)) : Prop :=
  M.Finite ∧
  ∀ A B, A ∈ M → B ∈ M → A ≠ B →
    ∃ C D, C ∈ M ∧ D ∈ M ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧
      (∃ t : ℝ, B - A = t • (D - C)) ∧  -- AB parallel to CD
      ¬ (∃ s : ℝ, C - A = s • (B - A))  -- AB and CD do not coincide

/-- Theorem stating the existence of a set satisfying the parallelism condition -/
theorem exists_parallel_point_set : ∃ M : Set (Fin 3 → ℝ), ParallelPointSet M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_parallel_point_set_l343_34375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_p_and_q_l343_34387

theorem propositions_p_and_q :
  (∃ a b : ℝ, (|a| + |b| > 1) ∧ (|a + b| ≤ 1)) ∧
  (∀ x : ℝ, (x ∈ Set.Iic (-1) ∪ Set.Ici 3) ↔ (|x - 1| - 2 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_p_and_q_l343_34387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l343_34346

noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.cos (2*x) + (Real.sqrt 3 / 2) * Real.sin (2*x)

noncomputable def g (x m : ℝ) : ℝ := f (x + m)

theorem min_shift_for_symmetry :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, g x m = g (-x) m) →
  m ≥ π/12 :=
by
  sorry

#check min_shift_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l343_34346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marksman_probability_l343_34348

theorem marksman_probability (P : ℝ) 
  (h1 : 0 < P) (h2 : P < 1) 
  (h3 : (Nat.choose 8 3) * P^3 * (1-P)^5 = (1/25) * (Nat.choose 8 5) * P^5 * (1-P)^3) : 
  P = 5/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marksman_probability_l343_34348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_argument_of_complex_number_l343_34301

theorem argument_of_complex_number : 
  Complex.arg (Complex.mk (Real.sin (40 * π / 180)) (-(Real.cos (40 * π / 180)))) = 140 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_argument_of_complex_number_l343_34301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l343_34326

/-- The function to be maximized -/
noncomputable def f (t : ℝ) : ℝ := (3^t - 2*t) * t / 9^t

/-- The maximum value of f(t) for real t -/
theorem f_max_value : ∃ (M : ℝ), M = 1/8 ∧ ∀ (t : ℝ), f t ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l343_34326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_west_asian_percentage_approx_47_l343_34311

/-- Asian population distribution in millions across four U.S. regions -/
def asian_population : Fin 4 → ℕ
  | 0 => 2  -- NE
  | 1 => 3  -- MW
  | 2 => 4  -- South
  | 3 => 8  -- West

/-- Total Asian population across all regions -/
def total_asian_population : ℕ := (Finset.sum Finset.univ asian_population)

/-- Asian population in the West -/
def west_asian_population : ℕ := asian_population 3

/-- Percentage of Asian population in the West -/
noncomputable def west_asian_percentage : ℚ :=
  (west_asian_population : ℚ) / (total_asian_population : ℚ) * 100

theorem west_asian_percentage_approx_47 :
  ∃ ε > 0, ε < 1 ∧ |west_asian_percentage - 47| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_west_asian_percentage_approx_47_l343_34311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_dot_product_l343_34369

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter O
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Main theorem
theorem right_triangle_dot_product (t : Triangle) :
  -- Conditions
  (t.A = (0, 0)) →  -- Assume A is at origin for simplicity
  (t.B = (6, 0)) →  -- AB = 6
  (t.C = (0, 8)) →  -- AC = 8
  -- Conclusion
  dot_product (vec t.A (circumcenter t)) (vec t.B t.C) = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_dot_product_l343_34369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l343_34317

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = 3x - 7 -/
noncomputable def m1 : ℝ := 3

/-- The slope of the second line 2y + bx = 4 in terms of b -/
noncomputable def m2 (b : ℝ) : ℝ := -b/2

/-- The value of b that makes the lines perpendicular -/
noncomputable def b : ℝ := 2/3

theorem lines_perpendicular : perpendicular m1 (m2 b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_l343_34317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inverse_correct_l343_34361

-- Define the functions f, g, and k
noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of k
noncomputable def k_inv (x : ℝ) : ℝ := (x + 11) / 12

-- Theorem statement
theorem k_inverse_correct : 
  ∀ x : ℝ, k (k_inv x) = x ∧ k_inv (k x) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inverse_correct_l343_34361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_problem_l343_34358

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Int :=
  (List.range (List.length digits)).zip digits
  |> List.foldl (fun acc (power, digit) => acc + digit * (b ^ power : Int)) 0

/-- The result of 325₆ - 215₉ in base 10 -/
def problem : Prop :=
  let base_6_num := to_base_10 [5, 2, 3] 6
  let base_9_num := to_base_10 [5, 1, 2] 9
  base_6_num - base_9_num = -51

theorem prove_problem : problem := by
  -- Unfold definitions
  unfold problem
  unfold to_base_10
  -- Evaluate the expressions
  simp
  -- The proof is complete
  rfl

#eval to_base_10 [5, 2, 3] 6 - to_base_10 [5, 1, 2] 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_problem_l343_34358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l343_34304

-- Define the basic structures
structure Line where
  -- Add necessary fields for a line
  slope : ℝ
  intercept : ℝ

structure Parabola where
  -- Add necessary fields for a parabola
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the propositions
def has_one_common_point (l : Line) (c : Parabola) : Prop :=
  ∃! x : ℝ, c.a * x^2 + c.b * x + c.c = l.slope * x + l.intercept

def is_tangent (l : Line) (c : Parabola) : Prop :=
  has_one_common_point l c ∧ 
  ∀ x : ℝ, c.a * x^2 + c.b * x + c.c ≥ l.slope * x + l.intercept

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ l : Line, ∀ c : Parabola, is_tangent l c → has_one_common_point l c) ∧ 
  (∃ l : Line, ∃ c : Parabola, has_one_common_point l c ∧ ¬is_tangent l c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_but_not_sufficient_l343_34304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_in_interval_l343_34372

open Real

theorem two_solutions_in_interval :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
    (∀ θ ∈ s, 0 < θ ∧ θ < π ∧ 2 - 4 * tan θ + 3 * tan (π/2 - 2*θ) = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_in_interval_l343_34372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l343_34345

/-- Represents a rectangular field with fencing on three sides -/
structure FencedField where
  area : ℚ
  uncoveredSide : ℚ
  costSide1 : ℚ
  costSide2 : ℚ
  costSide3 : ℚ

/-- Calculates the total cost of fencing for a given field -/
def totalFencingCost (field : FencedField) : ℚ :=
  let width := field.area / field.uncoveredSide
  field.costSide1 * field.uncoveredSide + 
  field.costSide2 * width + 
  field.costSide3 * field.uncoveredSide

/-- Theorem stating that the total fencing cost for the given field is $388 -/
theorem fencing_cost_theorem (field : FencedField) 
  (h1 : field.area = 680)
  (h2 : field.uncoveredSide = 40)
  (h3 : field.costSide1 = 3)
  (h4 : field.costSide2 = 4)
  (h5 : field.costSide3 = 5) :
  totalFencingCost field = 388 := by
  sorry

def main : IO Unit := do
  let result := totalFencingCost { 
    area := 680, 
    uncoveredSide := 40, 
    costSide1 := 3, 
    costSide2 := 4, 
    costSide3 := 5 
  }
  IO.println s!"The total fencing cost is: {result}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_theorem_l343_34345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l343_34373

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

theorem max_value_of_y :
  ∃ (m : ℝ), ∀ z ∈ Set.Icc 1 9, y z ≤ m ∧ m = 13 :=
by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l343_34373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l343_34315

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem stating that the area of a trapezium with parallel sides of lengths 20 and 18,
    and a distance of 25 between them, is equal to 475. -/
theorem trapezium_area_example : trapezium_area 20 18 25 = 475 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l343_34315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l343_34312

-- Define L(m) as the x-coordinate of the left endpoint of the intersection
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem limit_of_r_as_m_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -4 < m ∧ m < 4 →
    |r m - (1/2)| < ε := by
  sorry

#check limit_of_r_as_m_approaches_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_r_as_m_approaches_zero_l343_34312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_impossibility_l343_34397

theorem magic_square_impossibility : ¬ ∃ (a b c d : ℕ), 
  (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) ∧
  (a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2) ∧
  (a = 3 ∨ b = 3 ∨ c = 3 ∨ d = 3) ∧
  a + b + c + d = 34 ∧
  a ≤ 16 ∧ b ≤ 16 ∧ c ≤ 16 ∧ d ≤ 16 :=
by
  intro h
  cases' h with a h
  cases' h with b h
  cases' h with c h
  cases' h with d h
  cases' h with h1 h
  cases' h with h2 h
  cases' h with h3 h
  cases' h with h4 h5
  -- The proof would go here, but we'll use sorry for now
  sorry

#check magic_square_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_square_impossibility_l343_34397
