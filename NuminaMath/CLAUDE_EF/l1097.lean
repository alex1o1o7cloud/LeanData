import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_track_length_l1097_109739

/-- The length of a circular track satisfying specific meeting conditions for two runners -/
theorem circular_track_length : ∃ (x : ℝ), 
  x > 0 ∧ 
  (x / 2 - 120) / 120 = (x / 2 + 60) / (x - 120) ∧
  x = 300 + Real.sqrt 61200 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_track_length_l1097_109739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1097_109730

noncomputable def f (x : ℝ) := 2 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T')) ∧
  (∀ x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8),
    ∀ y ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8),
    x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1097_109730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_ordering_l1097_109798

theorem fraction_ordering : 
  (25 / 19 : ℚ) < (23 / 17 : ℚ) ∧ (23 / 17 : ℚ) < (21 / 13 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_ordering_l1097_109798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l1097_109768

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (x - b) / (x - a)

-- State the theorem
theorem increasing_function_condition (a b : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 4 → f a b x < f a b y) →
  4 ≤ a ∧ a < b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l1097_109768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1097_109742

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 2 → f x ≤ f y) ∧
  f x = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1097_109742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_arithmetic_fibonacci_p_th_term_l1097_109705

/-- Definition of p-arithmetic Fibonacci sequence -/
def p_arithmetic_fibonacci (p : ℕ) : ℕ → ℕ := sorry

/-- The p-th term of the p-arithmetic Fibonacci sequence -/
def p_th_term (p : ℕ) : ℕ := p_arithmetic_fibonacci p p

theorem p_arithmetic_fibonacci_p_th_term (p : ℕ) (h : Odd p) :
  p_th_term p = 5^((p - 1) / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_arithmetic_fibonacci_p_th_term_l1097_109705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_42_1376_to_hundredth_l1097_109708

-- Define a function to round a number to the nearest hundredth
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

-- State the theorem
theorem round_42_1376_to_hundredth :
  roundToHundredth 42.1376 = 42.14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_42_1376_to_hundredth_l1097_109708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_eleven_thirds_l1097_109759

theorem greatest_integer_less_than_eleven_thirds :
  Int.floor (11 / 3 : ℚ) = 3 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_less_than_eleven_thirds_l1097_109759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_polynomial_l1097_109764

theorem min_degree_polynomial (p : Polynomial ℚ) (hp : p ≠ 0) : 
  (∃ (a b c d : ℚ), 
    (a^2 = 7 ∧ (3 - a) ∈ p.roots) ∧ 
    (b^2 = 12 ∧ (5 + b) ∈ p.roots) ∧ 
    (c^2 = 40 ∧ (16 - c) ∈ p.roots) ∧ 
    (d^2 = 3 ∧ -d ∈ p.roots)) →
  p.degree ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_polynomial_l1097_109764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_difference_of_primes_l1097_109744

def arithmetic_sequence (start : ℕ) (diff : ℕ) (n : ℕ) : ℕ :=
  start + n * diff

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_difference_of_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = Int.natAbs (p - q)

theorem unique_difference_of_primes :
  ∃! k : ℕ, ∃ n : ℕ, arithmetic_sequence 7 10 n = k ∧ is_difference_of_primes k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_difference_of_primes_l1097_109744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_locus_and_tangent_property_l1097_109757

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 / a)^2 + (p.2 / b)^2 = 1}

-- Define the locus C
def LocusC (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (2 * p.1^2) / a^2 + (2 * p.2^2) / b^2 = 1}

-- Define the vector condition
def VectorCondition (O A B P : ℝ × ℝ) (m n : ℝ) : Prop :=
  P - O = ((m^2 - n^2) / (m^2 + n^2)) • (A - O) + ((2 * m * n) / (m^2 + n^2)) • (B - O)

-- Define a tangent line
def IsTangentAt (t : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ s ∧ p ∈ t ∧ ∀ q ∈ s, q ∈ t → q = p

theorem ellipse_midpoint_locus_and_tangent_property
  (a b : ℝ) (O A B P : ℝ × ℝ) (m n : ℝ)
  (h_ellipse : A ∈ Ellipse a b ∧ B ∈ Ellipse a b ∧ P ∈ Ellipse a b)
  (h_vector : VectorCondition O A B P m n)
  (h_positive : m > n ∧ n > 0) :
  (∃ C, C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ C ∈ LocusC a b) ∧
  (∀ Q ∈ LocusC a b, ∃ E F : ℝ × ℝ,
    E ∈ Ellipse a b ∧ F ∈ Ellipse a b ∧
    (∃ t : Set (ℝ × ℝ), IsTangentAt t (LocusC a b) Q ∧ E ∈ t ∧ F ∈ t) ∧
    dist Q E = dist Q F) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_midpoint_locus_and_tangent_property_l1097_109757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_hyperbola_l1097_109792

/-- The points (x, y) satisfying both equations for real parameter t form a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∃ (A B C D E F : ℝ) (h1 : A ≠ 0 ∨ B ≠ 0) (h2 : C ≠ 0),
    ∀ (x y t : ℝ),
      (2*t*x - 3*y - 4*t = 0) →
      (x - 3*t*y + 4 = 0) →
      A*x^2 + 2*B*x*y + C*y^2 + D*x + E*y + F = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_hyperbola_l1097_109792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_super_sudoku_l1097_109704

/-- Represents a 9x9 grid of integers -/
def Grid := Fin 9 → Fin 9 → Fin 9

/-- Checks if a given grid is a valid super-sudoku -/
def is_super_sudoku (g : Grid) : Prop :=
  -- Each square is filled with an integer between 1 and 9 (implicit in Fin 9)
  -- Each column contains each number 1-9 exactly once
  (∀ j : Fin 9, Function.Injective (λ i ↦ g i j)) ∧
  -- Each row contains each number 1-9 exactly once
  (∀ i : Fin 9, Function.Injective (λ j ↦ g i j)) ∧
  -- Each 3x3 subsquare contains each number 1-9 exactly once
  (∀ k l : Fin 3, Function.Injective (λ (i j : Fin 3) ↦ g (3 * k + i) (3 * l + j)))

/-- Theorem stating that no super-sudoku grid exists -/
theorem no_super_sudoku : ¬∃ (g : Grid), is_super_sudoku g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_super_sudoku_l1097_109704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_properties_l1097_109755

/-- A circle in the polar coordinate system -/
structure PolarCircle where
  /-- The polar equation of the circle -/
  equation : ℝ → ℝ → Prop
  /-- The circle passes through the point (√2, π/4) -/
  passes_through : equation (Real.sqrt 2) (π / 4)
  /-- The center of the circle is at the intersection of ρsin(θ - π/3) = -√3/2 with the polar axis -/
  center_on_intersection : ∃ (ρ : ℝ), equation ρ 0 ∧ ρ * Real.sin (-π / 3) = -Real.sqrt 3 / 2

/-- The theorem to be proved -/
theorem polar_circle_properties (C : PolarCircle) :
  (∀ (ρ θ : ℝ), C.equation ρ θ ↔ ρ = 2 * Real.cos θ) ∧
  (∃ (a b : ℝ), C.equation a (π / 3) ∧ C.equation b (π / 3) ∧ (b - a) = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_properties_l1097_109755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_30_factors_and_18_20_as_factors_l1097_109723

def has_30_factors (x : ℕ) : Prop := (Finset.filter (λ y ↦ x % y = 0) (Finset.range (x + 1))).card = 30

theorem smallest_integer_with_30_factors_and_18_20_as_factors :
  ∃ (x : ℕ), has_30_factors x ∧ 18 ∣ x ∧ 20 ∣ x ∧
  (∀ (y : ℕ), has_30_factors y ∧ 18 ∣ y ∧ 20 ∣ y → x ≤ y) ∧
  x = 720 :=
by
  sorry

#eval 720 % 18
#eval 720 % 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_30_factors_and_18_20_as_factors_l1097_109723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1097_109715

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1/a)*x + 1

-- Part 1
theorem part_one : 
  let a : ℝ := 2
  {x : ℝ | f a (2^x) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 2) :
  {x : ℝ | f a x ≥ 0} = {x : ℝ | x ≤ 1/a ∨ x ≥ a} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1097_109715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_yield_l1097_109781

-- Define the stocks and their properties
noncomputable def stock_a_yield : ℝ := 0.21
noncomputable def stock_a_quote : ℝ := 0.10
noncomputable def stock_b_yield : ℝ := 0.15
noncomputable def stock_b_quote : ℝ := 0.20

-- Define the investments
noncomputable def investment_a : ℝ := 10000
noncomputable def investment_b : ℝ := 15000

-- Define the weighted average yield calculation
noncomputable def weighted_average_yield (yield_a yield_b inv_a inv_b : ℝ) : ℝ :=
  (yield_a * inv_a + yield_b * inv_b) / (inv_a + inv_b)

-- Theorem statement
theorem portfolio_yield :
  weighted_average_yield stock_a_yield stock_b_yield investment_a investment_b = 0.174 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_yield_l1097_109781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_possible_starting_days_l1097_109725

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a 29-day month -/
def Month := Fin 29

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in the month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Checks if the number of Sundays and Tuesdays are equal for a given starting day -/
def hasSameSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  sorry

/-- List of all days of the week -/
def allDaysOfWeek : List DayOfWeek :=
  [DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday,
   DayOfWeek.Friday, DayOfWeek.Saturday, DayOfWeek.Sunday]

/-- The main theorem: there are exactly 4 possible starting days that satisfy the condition -/
theorem four_possible_starting_days :
  (allDaysOfWeek.filter hasSameSundaysAndTuesdays).length = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_possible_starting_days_l1097_109725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_buying_cheat_percentage_approx_l1097_109793

/-- Represents the shop owner's pricing strategy -/
structure ShopOwner where
  buyingCheatPercentage : ℝ
  sellingCheatPercentage : ℝ
  profitPercentage : ℝ

/-- Calculates the actual buying price given the claimed cost price -/
noncomputable def actualBuyingPrice (claimedCost : ℝ) (owner : ShopOwner) : ℝ :=
  claimedCost * (1 - owner.buyingCheatPercentage / 100)

/-- Calculates the actual selling price given the claimed cost price -/
noncomputable def actualSellingPrice (claimedCost : ℝ) (owner : ShopOwner) : ℝ :=
  claimedCost * (1 + owner.sellingCheatPercentage / 100)

/-- Theorem stating the relationship between buying cheat percentage and profit -/
theorem buying_cheat_percentage_approx (owner : ShopOwner) 
  (h1 : owner.sellingCheatPercentage = 30)
  (h2 : owner.profitPercentage = 60) :
  ∃ ε > 0, |owner.buyingCheatPercentage - 37.31| < ε := by
  sorry

#check buying_cheat_percentage_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_buying_cheat_percentage_approx_l1097_109793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_plane_parallel_l1097_109719

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry

def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem line_perpendicular_plane_parallel 
  (x y : Line) (z : Plane) 
  (h1 : perpendicular x z) 
  (h2 : perpendicular y z) : 
  parallel x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_plane_parallel_l1097_109719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_alpha_l1097_109747

theorem cos_seven_alpha (α : ℝ) (h : Real.cos α = 2/3) : 
  Real.cos (7*α) = 53/2187 + 3640/6561 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_alpha_l1097_109747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_sixteen_squares_l1097_109782

/-- The area of 16 congruent squares formed by dividing a square with diagonal 8 cm into a 3x3 grid and removing corner squares -/
theorem area_of_sixteen_squares (diagonal : ℝ) (h_diagonal : diagonal = 8) : 
  (16 * (diagonal / (3 * Real.sqrt 2)) ^ 2) = 512 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_sixteen_squares_l1097_109782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_l1097_109790

/-- A function f(x) = x ln x - ax^2 with two distinct extreme points -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x - 2 * a * x

theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    f_derivative a x₁ = 0 ∧ 
    f_derivative a x₂ = 0 ∧ 
    (∀ x : ℝ, x > 0 → f_derivative a x = 0 → (x = x₁ ∨ x = x₂))) →
  (a > 0 ∧ a < 1/2) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f_derivative a x₁ = 0 → 
    f_derivative a x₂ = 0 → 
    x₁ * x₂ > 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_condition_l1097_109790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l1097_109734

noncomputable section

-- Define the lines
def line1 (y a : ℝ) : ℝ := (1/4) * y + a
def line2 (x b : ℝ) : ℝ := (1/4) * x + b

-- State the theorem
theorem intersection_point_sum (a b : ℝ) 
  (h1 : line1 2 a = 1) 
  (h2 : line2 1 b = 2) : 
  a + b = 9/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l1097_109734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l1097_109780

-- Define the dimensions of the papers
noncomputable def amy_width : ℝ := 5
noncomputable def amy_height : ℝ := 10
noncomputable def belinda_width : ℝ := 7
noncomputable def belinda_height : ℝ := 10

-- Define the volumes of the cylindrical tubes
noncomputable def amy_volume : ℝ := (amy_width^2 * amy_height) / (4 * Real.pi)
noncomputable def belinda_volume : ℝ := (belinda_width^2 * belinda_height) / (4 * Real.pi)

-- Theorem statement
theorem cylinder_volume_difference :
  Real.pi * |belinda_volume - amy_volume| = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l1097_109780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A44_A45_A43_l1097_109728

/-- Represents a point in the triangle sequence -/
structure Point where
  coord : ℝ × ℝ

/-- Defines the triangle sequence -/
def triangle_sequence : ℕ → Point
  | 0 => ⟨(0, 0)⟩
  | n + 1 => ⟨(0, 0)⟩  -- Placeholder definition

/-- The angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := 0  -- Placeholder definition

/-- States that the initial triangle is isosceles with A₁ angle of 100° -/
axiom initial_triangle_isosceles :
  let A1 := (triangle_sequence 1).coord
  let A2 := (triangle_sequence 2).coord
  let A3 := (triangle_sequence 3).coord
  angle A1 A2 A3 = 100 ∧
  (A2.1 - A3.1)^2 + (A2.2 - A3.2)^2 =
  (A3.1 - A1.1)^2 + (A3.2 - A1.2)^2

/-- Defines the midpoint property for the sequence -/
axiom midpoint_property (n : ℕ) :
  let An := (triangle_sequence n).coord
  let An2 := (triangle_sequence (n + 2)).coord
  let An3 := (triangle_sequence (n + 3)).coord
  An3 = ((An.1 + An2.1) / 2, (An.2 + An2.2) / 2)

/-- The theorem to be proved -/
theorem angle_A44_A45_A43 :
  let A43 := (triangle_sequence 43).coord
  let A44 := (triangle_sequence 44).coord
  let A45 := (triangle_sequence 45).coord
  angle A44 A45 A43 = 100 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A44_A45_A43_l1097_109728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_p_implies_m_range_prop_p_and_q_implies_m_range_l1097_109722

noncomputable section

-- Define the curve equation
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x + 9/16) / Real.log 10

-- Proposition p: The curve is an ellipse with foci on the y-axis
def prop_p (m : ℝ) : Prop :=
  ∀ x y, curve_equation m x y → (m > 1 ∧ m < 3 ∧ 3 - m > m - 1)

-- Proposition q: The domain of f(x) is ℝ
def prop_q (m : ℝ) : Prop :=
  ∀ x, x ∈ Set.univ → f m x ∈ Set.univ

-- Theorem 1
theorem prop_p_implies_m_range (m : ℝ) :
  prop_p m → m > 1 ∧ m < 2 :=
by
  sorry

-- Theorem 2
theorem prop_p_and_q_implies_m_range (m : ℝ) :
  prop_p m ∧ prop_q m → m > 1 ∧ m < 3/2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_p_implies_m_range_prop_p_and_q_implies_m_range_l1097_109722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_decreasing_on_interval_l1097_109765

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 1 / (2 * x^2)

-- Theorem for evenness of f
theorem f_is_even : ∀ x : ℝ, x ≠ 0 → f (-x) = f x := by sorry

-- Theorem for monotonicity of f on (0, √2/2)
theorem f_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.sqrt 2 / 2 → f x₁ > f x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_decreasing_on_interval_l1097_109765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1097_109703

theorem sin_half_angle_special_case (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_special_case_l1097_109703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_visitors_november_l1097_109799

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
deriving Repr, DecidableEq

/-- Calculates the number of occurrences of each day in a 30-day month starting with the given day -/
def countDays (startDay : DayOfWeek) : DayOfWeek → Nat
  | DayOfWeek.Sunday => if startDay = DayOfWeek.Sunday then 5 else 4
  | _ => 4

/-- Returns the number of visitors for a given day of the week -/
def visitorsPerDay (day : DayOfWeek) : Nat :=
  match day with
  | DayOfWeek.Sunday => 510
  | DayOfWeek.Monday | DayOfWeek.Tuesday => 240
  | DayOfWeek.Wednesday | DayOfWeek.Thursday => 300
  | DayOfWeek.Friday | DayOfWeek.Saturday => 200

/-- Calculates the total number of visitors for a 30-day month starting with the given day -/
def totalVisitors (startDay : DayOfWeek) : Nat :=
  let f (day : DayOfWeek) := (countDays startDay day) * (visitorsPerDay day)
  f DayOfWeek.Sunday + f DayOfWeek.Monday + f DayOfWeek.Tuesday + f DayOfWeek.Wednesday +
  f DayOfWeek.Thursday + f DayOfWeek.Friday + f DayOfWeek.Saturday

theorem average_visitors_november (h : totalVisitors DayOfWeek.Sunday / 30 = 282) :
  282 = totalVisitors DayOfWeek.Sunday / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_visitors_november_l1097_109799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1097_109776

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = q * a n

def first_term (a : ℕ → ℝ) : ℝ := a 1

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence a q) :
  let b := λ n ↦ a n + a (n + 1) + a (n + 2)
  let c := λ n ↦ 1 / a n
  (geometric_sequence b q ∧ first_term b = first_term a * (1 + q + q^2)) ∧
  (geometric_sequence c (1/q) ∧ first_term c = 1 / first_term a) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1097_109776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_equation_l1097_109706

-- Define the parametric equations of line l
noncomputable def line_l (a t : ℝ) : ℝ × ℝ := (3 + t, 1 + a * t)

-- Define the parametric equations of curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the intersection points A and B
noncomputable def A (a : ℝ) : ℝ × ℝ := sorry

noncomputable def B (a : ℝ) : ℝ × ℝ := sorry

-- Define the length of chord AB
noncomputable def AB_length (a : ℝ) : ℝ := sorry

-- Define the condition for shortest chord
def is_shortest_chord (a : ℝ) : Prop := sorry

-- Theorem statement
theorem shortest_chord_equation (a : ℝ) :
  is_shortest_chord a → (λ (x y : ℝ) => x + y - 4 = 0) = (λ (x y : ℝ) => y - 1 = a * (x - 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_equation_l1097_109706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perpendicular_l1097_109754

-- Define the necessary structures
structure Line : Type
structure Plane : Type

-- Define the relationships
axiom perpendicular_line_plane : Line → Plane → Prop
axiom perpendicular_line_line : Line → Line → Prop
axiom perpendicular_plane_plane : Plane → Plane → Prop

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : perpendicular_line_plane m α) 
  (h4 : perpendicular_line_plane n β) 
  (h5 : perpendicular_line_line m n) : 
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perpendicular_l1097_109754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1097_109772

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 1/x^3

-- State the theorem
theorem range_of_f :
  ∀ x : ℝ, x ≠ 0 → (x + 1/x ≤ 4) →
  (f x ∈ Set.Iic (-2) ∪ Set.Icc 2 52) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1097_109772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_coords_l1097_109788

noncomputable def harry_position : ℝ × ℝ := (10, -3)
noncomputable def sandy_position : ℝ × ℝ := (4, 9)

noncomputable def meeting_point (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let t : ℝ := 1/3
  (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

theorem meeting_point_coords :
  meeting_point harry_position sandy_position = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_point_coords_l1097_109788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_length_l1097_109756

def R : ℝ × ℝ := (10, 4)

def line1 (x y : ℝ) : Prop := 5 * y = 12 * x

def line2 (x y : ℝ) : Prop := 7 * y = 2 * x

def midpoint_of (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem PQ_length :
  ∀ P Q : ℝ × ℝ,
  line1 P.1 P.2 →
  line2 Q.1 Q.2 →
  midpoint_of P Q R →
  distance P Q = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_length_l1097_109756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_theorem_l1097_109784

/-- The horizontal shift required to transform y = sin(2x - 3π/4) to y = sin(2x) -/
noncomputable def horizontal_shift : ℝ := 3 * Real.pi / 8

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 3 * Real.pi / 4)

/-- The target function -/
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem sin_shift_theorem :
  ∀ x : ℝ, f (x + horizontal_shift) = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_theorem_l1097_109784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1097_109711

/-- Given a hyperbola with the following properties:
    1. Its equation is of the form x²/a² - y²/b² = 1 where a > 0 and b > 0
    2. It has an asymptote parallel to y = √3x - 1
    3. One of its foci is on the directrix of the parabola x² = 24y
    Then the equation of the hyperbola is y²/27 - x²/9 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (asymptote : ∃ (k : ℝ), k * Real.sqrt 3 = a / b) 
  (focus : ∃ (x y : ℝ), x^2 = 24*y ∧ x^2/a^2 - y^2/b^2 = 1 ∧ y = -6) :
  a^2 = 27 ∧ b^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1097_109711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_say_will_lie_tomorrow_on_thursday_l1097_109731

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
  deriving Repr, Inhabited, DecidableEq

-- Define the next day function
def next_day : Day → Day
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Define the lying patterns for Chris and Mark
def Chris_lies (d : Day) : Prop :=
  d = Day.Friday ∨ d = Day.Saturday ∨ d = Day.Sunday

def Mark_lies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday

-- Define the statement "Tomorrow, I will lie"
def says_will_lie_tomorrow (person_lies : Day → Prop) (d : Day) : Prop :=
  (person_lies d ∧ ¬person_lies (next_day d)) ∨
  (¬person_lies d ∧ person_lies (next_day d))

-- Theorem stating that Thursday is the only day when both Chris and Mark say "Tomorrow, I will lie"
theorem both_say_will_lie_tomorrow_on_thursday :
  ∀ d : Day, (says_will_lie_tomorrow Chris_lies d ∧ says_will_lie_tomorrow Mark_lies d) ↔ d = Day.Thursday :=
by sorry

#eval Day.Thursday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_say_will_lie_tomorrow_on_thursday_l1097_109731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1097_109760

/-- Given a triangle ABC with the specified properties, prove the angle A and maximum perimeter -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  (Real.sin A)^2 - (Real.sin B)^2 - (Real.sin C)^2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove:
  -- 1) A = 2π/3
  A = 2 * Real.pi / 3 ∧
  -- 2) Maximum perimeter is 3 + 2√3
  (∀ (a' b' c' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' →
    a' + b' + c' ≤ 3 + 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1097_109760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coefficients_binomial_expansion_zero_l1097_109727

theorem sum_coefficients_binomial_expansion_zero :
  let n : ℕ := 7
  let coefficients := List.range (n + 1) |>.map (λ k => (n.choose k : ℤ) * (-1)^(n - k))
  coefficients.sum = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_coefficients_binomial_expansion_zero_l1097_109727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequence_function_property_l1097_109735

/-- The set of all binary sequences of length n -/
def BinarySequence (n : ℕ) := Fin n → Bool

/-- Addition operation on binary sequences -/
def add {n : ℕ} (a b : BinarySequence n) : BinarySequence n :=
  λ i => a i ≠ b i

/-- The zero sequence -/
def zero {n : ℕ} : BinarySequence n := λ _ => false

/-- Hamming distance between two binary sequences -/
def hammingDistance {n : ℕ} (a b : BinarySequence n) : ℕ :=
  (Finset.filter (λ i => a i ≠ b i) (Finset.univ : Finset (Fin n))).card

/-- Main theorem -/
theorem binary_sequence_function_property {n : ℕ} 
  (f : BinarySequence n → BinarySequence n)
  (h1 : f zero = zero)
  (h2 : ∀ a b : BinarySequence n, hammingDistance (f a) (f b) = hammingDistance a b)
  (a b c : BinarySequence n)
  (h3 : add (add a b) c = zero) :
  add (add (f a) (f b)) (f c) = zero := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequence_function_property_l1097_109735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1097_109729

-- Define the complex number z
noncomputable def z : ℂ := (2 * Complex.I) / (1 + Complex.I)

-- Theorem statement
theorem z_in_first_quadrant : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1097_109729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_range_l1097_109741

theorem subset_implies_a_range (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | 1 ≤ x ∧ x ≤ 3} →
  B = {x : ℝ | 0 < x ∧ x < a} →
  A ⊆ B →
  a ∈ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_a_range_l1097_109741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CMN_l1097_109774

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 1

-- Define the intersection points M and N
def intersection_points (M N : ℝ × ℝ) : Prop :=
  line_l M.1 M.2 ∧ circle_C M.1 M.2 ∧
  line_l N.1 N.2 ∧ circle_C N.1 N.2 ∧
  M ≠ N

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, -2)

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_CMN (M N : ℝ × ℝ) :
  intersection_points M N →
  area_triangle circle_center M N = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CMN_l1097_109774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_division_l1097_109778

theorem angle_division (angle : ℝ) (divisions : ℕ) (h1 : angle = 78) (h2 : divisions = 26) :
  angle / (divisions : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_division_l1097_109778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1097_109779

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 + Real.cos (2 * x)) / (4 * Real.sin (Real.pi / 2 + x)) - a * Real.sin (x / 2) * Real.cos (Real.pi - x / 2)

theorem max_value_implies_a (a : ℝ) :
  (∃ M, M = 1 ∧ ∀ x, f a x ≤ M) →
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l1097_109779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_natural_number_with_square_digit_sum_2014_2015_l1097_109791

-- Define sum_of_digits function
def sum_of_digits : ℕ → ℕ
| n => if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem no_natural_number_with_square_digit_sum_2014_2015 :
  ¬ ∃ (n : ℕ), sum_of_digits (n^2) = 2014 * 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_natural_number_with_square_digit_sum_2014_2015_l1097_109791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_l1097_109797

theorem inverse_difference_inverse : ((9 : ℚ)⁻¹ - (6 : ℚ)⁻¹)⁻¹ = -18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_l1097_109797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_collinear_vectors_l1097_109707

/-- Given vectors a and b, if they project to the same vector p on some vector v,
    and a, b, and p are collinear, then p is equal to (4/3, 1/3, 4/3) -/
theorem projection_collinear_vectors (a b p v : Fin 3 → ℝ) :
  a = ![1, -1, 2] →
  b = ![2, 3, 0] →
  (∃ (k : ℝ), p = k • (b - a) + a) →  -- collinearity condition
  (∃ (t : ℝ), p = t • v) →  -- p is a projection onto v
  (∃ (s : ℝ), a = s • v) →  -- a projects onto v
  (∃ (r : ℝ), b = r • v) →  -- b projects onto v
  p = ![4/3, 1/3, 4/3] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_collinear_vectors_l1097_109707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l1097_109721

/-- A polygon in a square -/
structure Polygon where
  area : ℝ

/-- A square containing polygons -/
structure Square where
  area : ℝ
  polygons : List Polygon

/-- The intersection area of two polygons -/
noncomputable def intersection_area (p1 p2 : Polygon) : ℝ := sorry

/-- Theorem: In a square of area 5 with 9 polygons of area 1 each, 
    there exist two polygons with intersection area at least 1/9 -/
theorem intersection_theorem (s : Square) 
  (h1 : s.area = 5)
  (h2 : s.polygons.length = 9)
  (h3 : ∀ p, p ∈ s.polygons → p.area = 1) :
  ∃ p1 p2, p1 ∈ s.polygons ∧ p2 ∈ s.polygons ∧ p1 ≠ p2 ∧ intersection_area p1 p2 ≥ 1/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_theorem_l1097_109721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_caterer_cheaper_at_N_l1097_109713

/-- Caterer pricing models -/
def first_caterer_cost (p : ℕ) : ℚ := 50 * p

noncomputable def second_caterer_cost (p : ℕ) : ℝ :=
  if p ≤ 61 then
    500 + 40 * p
  else
    2500 * Real.log (p / 4 : ℝ)

/-- The number of plates where the second caterer becomes cheaper -/
def N : ℕ := 51

/-- Theorem stating that for N plates, the second caterer is at least as cheap as the first -/
theorem second_caterer_cheaper_at_N :
  (second_caterer_cost N : ℝ) ≤ (first_caterer_cost N : ℝ) := by
  sorry

#check second_caterer_cheaper_at_N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_caterer_cheaper_at_N_l1097_109713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_relationship_polynomial_coefficients_l1097_109701

/-- The polynomial h(x) = x^3 - 2x^2 + 3x - 4 -/
def h (x : ℂ) : ℂ := x^3 - 2*x^2 + 3*x - 4

/-- The polynomial p(x) = x^3 + ax^2 + bx + c -/
def p (x a b c : ℂ) : ℂ := x^3 + a*x^2 + b*x + c

/-- The roots of h(x) -/
def roots_h : Set ℂ :=
  {x | h x = 0}

/-- The roots of p(x) -/
def roots_p (a b c : ℂ) : Set ℂ :=
  {x | p x a b c = 0}

/-- Theorem stating the relationship between the roots of h and p -/
theorem roots_relationship (a b c : ℂ) :
  roots_p a b c = (roots_h.image (λ x ↦ x^2)) → a = -7 ∧ b = 14 ∧ c = -8 := by
  sorry

/-- Main theorem to prove -/
theorem polynomial_coefficients :
  ∃ a b c : ℂ, roots_p a b c = (roots_h.image (λ x ↦ x^2)) ∧ a = -7 ∧ b = 14 ∧ c = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_relationship_polynomial_coefficients_l1097_109701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l1097_109738

theorem test_maximum_marks (passing_percentage : ℚ) (student_score : ℕ) (failing_margin : ℕ) : 
  passing_percentage = 30/100 →
  student_score = 80 →
  failing_margin = 10 →
  ∃ (max_marks : ℕ), max_marks = 300 ∧ 
    (passing_percentage * max_marks = (student_score + failing_margin : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l1097_109738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equilateral_triangle_side_length_l1097_109716

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := y^2 = a * x

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = -3

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d := Real.sqrt 3
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = d^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = d^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = d^2

-- Define a point on the parabola
def point_on_parabola (a : ℝ) (P : ℝ × ℝ) : Prop :=
  parabola a P.1 P.2

theorem parabola_equilateral_triangle_side_length 
  (a : ℝ) 
  (A B C : ℝ × ℝ) :
  a ≠ 0 →
  axis_of_symmetry (-3) →
  point_on_parabola a A ∧ point_on_parabola a B ∧ point_on_parabola a C →
  equilateral_triangle A B C →
  A = (0, 0) →
  let side_length := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  side_length = 24 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equilateral_triangle_side_length_l1097_109716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_combining_theorem_l1097_109766

def canCombineWater (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem water_combining_theorem (n : ℕ) :
  (∃ (initial_amount : ℕ), initial_amount > 0) →
  (∀ (pour : ℕ → ℕ → ℕ), 
    (∀ (i j : ℕ), i ≠ j → i < n → j < n → 
      pour i j = 2 * pour j i) →
    (∃ (final : ℕ), final < n ∧ 
      (∀ (i : ℕ), i < n → i ≠ final → pour i final = 0))) →
  canCombineWater n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_combining_theorem_l1097_109766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_6_l1097_109794

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem derivative_f_at_pi_over_6 :
  deriv f (Real.pi / 6) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_6_l1097_109794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1097_109758

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

-- Define the points and distances
variable (H : Hyperbola) (F₁ F₂ P Q : ℝ × ℝ)

-- Define the conditions
def intersects_hyperbola (H : Hyperbola) (F₁ P Q : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), P = F₁ + t • (Q - F₁) ∧
    (P.2 ^ 2 / H.a ^ 2) - (P.1 ^ 2 / H.b ^ 2) = 1 ∧
    (Q.2 ^ 2 / H.a ^ 2) - (Q.1 ^ 2 / H.b ^ 2) = 1

noncomputable def distance_difference (F₁ P Q : ℝ × ℝ) (a : ℝ) : Prop :=
  Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) -
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 2 * a

def orthogonal_vectors (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2) = 0

-- Define eccentricity
noncomputable def eccentricity (H : Hyperbola) : ℝ :=
  Real.sqrt (H.a^2 + H.b^2) / H.a

-- State the theorem
theorem hyperbola_eccentricity (H : Hyperbola) (F₁ F₂ P Q : ℝ × ℝ) :
  intersects_hyperbola H F₁ P Q →
  distance_difference F₁ P Q H.a →
  orthogonal_vectors P F₁ F₂ →
  eccentricity H = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1097_109758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_bound_l1097_109785

/-- A convex polygon with unit area -/
structure UnitConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)
  unitArea : MeasureTheory.volume (convexHull ℝ vertices) = 1

/-- A centrally symmetric convex hexagon -/
structure CentrallySymmetricConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  convex : Convex ℝ (convexHull ℝ (Set.range vertices))
  centrallySymmetric : ∀ i : Fin 6, vertices i = -(vertices ((i + 3) % 6))

/-- The theorem statement -/
theorem inscribed_hexagon_area_bound (W : UnitConvexPolygon) :
  ∃ (V : CentrallySymmetricConvexHexagon),
    (∀ i : Fin 6, V.vertices i ∈ W.vertices) ∧
    MeasureTheory.volume (convexHull ℝ (Set.range V.vertices)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_bound_l1097_109785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_uvz_perimeter_l1097_109752

/-- A rectangle UVWX with a semi-circular wall -/
structure RectangleWithWall where
  /-- Length of side UV -/
  uv : ℝ
  /-- Length of side VW -/
  vw : ℝ
  /-- Highest point on the semi-circular wall -/
  z : ℝ × ℝ

/-- The perimeter of triangle UVZ in a rectangle with a semi-circular wall -/
noncomputable def trianglePerimeter (r : RectangleWithWall) : ℝ :=
  r.uv + 2 * Real.sqrt 1100

/-- Theorem stating the perimeter of triangle UVZ -/
theorem triangle_uvz_perimeter (r : RectangleWithWall) 
  (h1 : r.uv = 20) 
  (h2 : r.vw = 30) : 
  trianglePerimeter r = 20 + 2 * Real.sqrt 1100 := by
  sorry

/-- Approximation of the perimeter -/
def approxPerimeter : ℚ :=
  20 + 2 * 33.166

#eval approxPerimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_uvz_perimeter_l1097_109752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_total_spent_l1097_109712

/-- Sally's shopping expenses -/
def sally_shopping (peaches_price : ℝ) (coupon : ℝ) (cherries_price : ℝ) : ℝ :=
  (peaches_price + coupon) + cherries_price

theorem sally_total_spent :
  sally_shopping 12.32 3 11.54 = 26.86 := by
  -- Unfold the definition of sally_shopping
  unfold sally_shopping
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_total_spent_l1097_109712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_root_l1097_109762

-- Define the polynomials f and g
def f (r : ℝ) (c : ℝ) (x : ℝ) : ℝ := (x - (r + 2)) * (x - (r + 8)) * (x - c)
def g (r : ℝ) (d : ℝ) (x : ℝ) : ℝ := (x - (r + 5)) * (x - (r + 11)) * (x - d)

-- State the theorem
theorem cubic_polynomials_root :
  ∃ r : ℝ, ∀ c d x : ℝ, f r c x - g r d x = 2 * r → r = 20.25 := by
  sorry

-- Optional: You can add more specific lemmas or propositions if needed
-- For example:
lemma polynomial_difference_constant (r : ℝ) (c d : ℝ) :
  ∀ x : ℝ, f r c x - g r d x = 2 * r → 
  ∃ k : ℝ, (x - (r + 2)) * (x - (r + 8)) * (x - c) - 
           (x - (r + 5)) * (x - (r + 11)) * (x - d) = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_root_l1097_109762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l1097_109700

theorem extreme_value_condition (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (2/3) * x^3 + a * x^2
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, 0 < |x - 2| ∧ |x - 2| < ε → 
    |f x - f 2| ≤ |(2 * x^2 + 2 * a * x) * (x - 2)|) →
  a = -2 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l1097_109700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_corner_probability_l1097_109761

/-- Represents a position on the 4x4 grid -/
inductive Position
| Corner : Position
| Edge : Position
| Inner : Position

/-- Represents the number of hops, up to 5 -/
def Hops : Type := Fin 6

/-- The probability of reaching a corner from a given position within n hops -/
def probability_to_corner (pos : Position) (n : Nat) : ℚ :=
  sorry

/-- The 4x4 grid with wrapping rules -/
def Grid : Type := Unit

/-- The frog's movement rules on the grid -/
def move (g : Grid) (pos : Position) : Position :=
  sorry

theorem frog_corner_probability :
  ∃ (g : Grid), probability_to_corner Position.Edge 5 = 57 / 64 :=
by
  sorry

#check frog_corner_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_corner_probability_l1097_109761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sequences_l1097_109736

noncomputable def a (n : ℕ+) : ℝ := 2^(n.val - 1)
noncomputable def b (n : ℕ+) : ℝ := 3 * n.val - 1

noncomputable def S (n : ℕ+) : ℝ := (b n^2 + 3 * b n + 2) / 6

noncomputable def c (n : ℕ+) : ℝ := b n / a n

theorem geometric_arithmetic_sequences :
  (∀ n : ℕ+, a n > 0) ∧
  (a 1 = 1) ∧
  (a 2 * a 4 = 16) ∧
  (∀ n : ℕ+, b (n + 1) > b n) ∧
  (a 4 = b 3) ∧
  (∀ n : ℕ+, 6 * S n = b n^2 + 3 * b n + 2) →
  (∀ n : ℕ+, a n = 2^(n.val - 1)) ∧
  (∀ n : ℕ+, b n = 3 * n.val - 1) ∧
  (∀ n : ℕ+, c n > 1 ↔ n.val ∈ ({1, 2, 3, 4} : Set ℕ)) ∧
  (∀ p q r : ℕ+, p < q → q < r → 2 * a q ≠ a p + a r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_arithmetic_sequences_l1097_109736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_nontrivial_factor_l1097_109743

def has_exactly_one_nontrivial_factor (n : ℕ) : Prop :=
  ∃! x : ℕ, x ≠ 1 ∧ x ≠ n ∧ n % x = 0

theorem unique_nontrivial_factor :
  ∀ n : ℕ, n ∈ ({6, 8, 13, 19, 25} : Set ℕ) → (has_exactly_one_nontrivial_factor n ↔ n = 25) :=
by
  intro n hn
  sorry

#check unique_nontrivial_factor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_nontrivial_factor_l1097_109743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_not_arithmetic_progression_l1097_109750

/-- The sequence b_n defined recursively -/
def b : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => b (n + 1) * b (n + 2) + 1

/-- A sequence is an arithmetic progression if the difference between consecutive terms is constant -/
def is_arithmetic_progression (s : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, s (n + 1) - s n = d

/-- The main theorem stating that b is not an arithmetic progression -/
theorem b_not_arithmetic_progression : ¬ is_arithmetic_progression b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_not_arithmetic_progression_l1097_109750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_decreasing_omega_range_l1097_109724

theorem sin_monotone_decreasing_omega_range (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x, x ∈ Set.Ioo (π / 2) π → f x = Real.sin (ω * x + π / 4)) →
  (∀ x y, x ∈ Set.Ioo (π / 2) π → y ∈ Set.Ioo (π / 2) π → x < y → f x > f y) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_decreasing_omega_range_l1097_109724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_longevity_theorem_l1097_109786

/-- The probability of an animal living to a certain age -/
noncomputable def prob_live_to (age : ℕ) : ℝ := sorry

/-- The conditional probability of an animal living to age b given it has lived to age a -/
noncomputable def conditional_prob (a b : ℕ) : ℝ := 
  (prob_live_to b) / (prob_live_to a)

theorem animal_longevity_theorem (h1 : prob_live_to 20 = 0.8) (h2 : prob_live_to 25 = 0.4) :
  conditional_prob 20 25 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_longevity_theorem_l1097_109786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_write_numbers_l1097_109751

/-- Calculates the number of digits in a positive integer -/
def countDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else countDigits (n / 10) + 1

/-- Calculates the sum of digits for all numbers in a given range -/
def sumDigitsInRange (start : ℕ) (stop : ℕ) : ℕ :=
  List.sum (List.map countDigits (List.range (stop - start + 1) |>.map (· + start)))

/-- Converts seconds to minutes, rounding to the nearest minute -/
def secondsToRoundedMinutes (seconds : ℕ) : ℕ :=
  (seconds + 30) / 60

theorem time_to_write_numbers (start stop : ℕ) (start_eq : start = 5) (stop_eq : stop = 105) :
  secondsToRoundedMinutes (sumDigitsInRange start stop) = 4 := by
  sorry

#eval secondsToRoundedMinutes (sumDigitsInRange 5 105)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_write_numbers_l1097_109751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_speed_l1097_109709

/-- Proves that a beetle's speed is 2.7 km/h given the conditions -/
theorem beetle_speed : 
  ∀ (ant_distance : ℝ) (time : ℝ) (beetle_distance_ratio : ℝ),
    ant_distance = 600 →
    time = 10 →
    beetle_distance_ratio = 0.75 →
    (ant_distance * beetle_distance_ratio / 1000) / (time / 60) = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_speed_l1097_109709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1097_109775

theorem angle_sum_theorem (θ φ : ℝ) (h1 : 0 < θ ∧ θ < π/2) (h2 : 0 < φ ∧ φ < π/2)
  (h3 : Real.tan θ = 1/3) (h4 : Real.sin φ = 3/5) :
  θ + 3*φ = π - Real.arctan (199/93) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1097_109775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuttlefish_sum_bound_l1097_109795

/-- A graph with 400 vertices and edges assigned values of 1 or -1 -/
structure CuttlefishGraph where
  vertices : Finset (Fin 400)
  edges : Finset (Fin 400 × Fin 400)
  edge_value : (Fin 400 × Fin 400) → Int
  edge_value_range : ∀ e ∈ edges, edge_value e = 1 ∨ edge_value e = -1

/-- The set of edges in a cuttlefish for a given edge -/
def cuttlefish (G : CuttlefishGraph) (e : Fin 400 × Fin 400) : Finset (Fin 400 × Fin 400) :=
  G.edges.filter (fun f => e.1 = f.1 ∨ e.1 = f.2 ∨ e.2 = f.1 ∨ e.2 = f.2)

/-- The sum of edge values in a set of edges -/
def edge_sum (G : CuttlefishGraph) (E : Finset (Fin 400 × Fin 400)) : Int :=
  E.sum G.edge_value

/-- The theorem statement -/
theorem cuttlefish_sum_bound (G : CuttlefishGraph) 
  (h : ∀ e ∈ G.edges, edge_sum G (cuttlefish G e) ≥ 1) :
  edge_sum G G.edges ≥ -10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuttlefish_sum_bound_l1097_109795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1097_109714

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 2*x - 2
  else if x < 0 then -x^2 - 2*x + 2
  else 0  -- undefined at x = 0

theorem f_properties :
  (∀ x ≠ 0, f (-x) = -f x) ∧
  (∀ x < 0, f x = -x^2 - 2*x + 2) ∧
  (∀ x, f x > 1 ↔ (x ∈ Set.Ioo (-1 - Real.sqrt 2) 0 ∨ x ∈ Set.Ioi 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1097_109714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_commission_l1097_109783

theorem new_average_commission
  (total_sales : ℕ)
  (big_sale_commission : ℚ)
  (average_increase : ℚ)
  (h_total_sales : total_sales = 6)
  (h_big_sale_commission : big_sale_commission = 1000)
  (h_average_increase : average_increase = 150) :
  let new_average : ℚ := 250
  (big_sale_commission + (total_sales - 1 : ℕ) * (new_average - average_increase)) / total_sales = new_average :=
by
  -- Introduce the local definition of new_average
  let new_average : ℚ := 250
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_commission_l1097_109783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_earning_members_l1097_109767

theorem initial_earning_members : ∃ n : ℕ, n = 14 ∧ 
  n * 782 = (n - 1) * 650 + 1178 ∧
  n > 0 := by
  sorry

#check initial_earning_members

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_earning_members_l1097_109767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_side_b_is_sqrt_7_l1097_109717

-- Define an acute triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  law_of_sines : a / (Real.sin A) = b / (Real.sin B)
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- Define the specific triangle from the problem
def specific_triangle : Prop :=
  ∃ (t : AcuteTriangle), t.a = 3 * Real.sqrt 3 ∧ t.c = 5

-- Theorem 1: Prove that angle B is 30°
theorem angle_B_is_30_degrees {t : AcuteTriangle} 
  (h : t.a = 2 * t.b * Real.sin t.A) : 
  t.B = π / 6 := by sorry

-- Theorem 2: Prove that side b is √7
theorem side_b_is_sqrt_7 {t : AcuteTriangle} 
  (h1 : t.a = 3 * Real.sqrt 3) 
  (h2 : t.c = 5) 
  (h3 : t.B = π / 6) : 
  t.b = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_side_b_is_sqrt_7_l1097_109717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_positive_l1097_109745

theorem T_is_positive (α : ℝ) (h : ∀ k : ℤ, α ≠ k * π / 2) :
  let T := (Real.sin α + Real.tan α) / (Real.cos α + (1 / Real.tan α))
  T > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_positive_l1097_109745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l1097_109702

/-- Time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length platform_length time_to_cross_pole : ℝ) : ℝ :=
  (train_length + platform_length) / (train_length / time_to_cross_pole)

/-- Theorem: The time taken for a 300 m long train to cross a 150 m long platform,
    given that it takes 26 seconds to cross a signal pole, is (300 + 150) / (300 / 26) seconds -/
theorem train_crossing_platform_time :
  time_to_cross_platform 300 150 26 = (300 + 150) / (300 / 26) := by
  -- Unfold the definition of time_to_cross_platform
  unfold time_to_cross_platform
  -- The rest of the proof is omitted
  sorry

#eval (300 + 150) / (300 / 26)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_time_l1097_109702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_outing_arrangements_l1097_109748

/-- The number of senior classes -/
def num_classes : ℕ := 6

/-- The number of scenic spots -/
def num_spots : ℕ := 6

/-- The number of classes with restrictions -/
def num_restricted_classes : ℕ := 2

/-- The number of spots available for restricted classes -/
def num_spots_for_restricted : ℕ := num_spots - 1

/-- The number of ways to arrange the restricted classes -/
def arrange_restricted : ℕ := (num_spots_for_restricted).choose num_restricted_classes * (num_restricted_classes).factorial

/-- The number of ways to arrange the unrestricted classes -/
def arrange_unrestricted : ℕ := num_spots ^ (num_classes - num_restricted_classes)

/-- The total number of arrangements -/
def total_arrangements : ℕ := arrange_restricted * arrange_unrestricted

theorem spring_outing_arrangements :
  total_arrangements = ((num_spots - 1).choose num_restricted_classes * num_restricted_classes.factorial) * num_spots ^ (num_classes - num_restricted_classes) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_outing_arrangements_l1097_109748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l1097_109737

-- Define the ellipse
def ellipse (x y b : ℝ) : Prop := x^2 / 4 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem ellipse_line_slope :
  ∀ (a b : ℝ) (A B : ℝ × ℝ),
  a > b ∧ b > 0 →
  ellipse A.1 A.2 b ∧ ellipse B.1 B.2 b →
  ((A.1 + B.1) / 2 = midpoint_AB.1 ∧ (A.2 + B.2) / 2 = midpoint_AB.2) →
  (B.2 - A.2) / (B.1 - A.1) = 1 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_l1097_109737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cosine_phi_l1097_109732

theorem odd_cosine_phi (φ : ℝ) (h1 : φ ∈ Set.Ioo 0 π) 
  (h2 : ∀ x, Real.cos (2*x + φ) = -Real.cos (-2*x + φ)) : φ = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cosine_phi_l1097_109732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_QS_ST_TR_l1097_109769

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the angle PQR
noncomputable def anglePQR (t : Triangle) : ℝ := 30 * Real.pi / 180

-- Define the lengths of PQ and PR
def lengthPQ : ℝ := 8
def lengthPR : ℝ := 12

-- Define points S and T
structure Points (t : Triangle) :=
  (S : ℝ × ℝ)
  (T : ℝ × ℝ)

-- Define the theorem
theorem min_sum_QS_ST_TR (t : Triangle) (p : Points t) :
  ∃ (minSum : ℝ), minSum = Real.sqrt (208 + 96 * Real.sqrt 3) ∧
  ∀ (S T : ℝ × ℝ), 
    (∃ (l1 : ℝ), 0 ≤ l1 ∧ l1 ≤ 1 ∧ S = (l1 * t.P.1 + (1 - l1) * t.Q.1, l1 * t.P.2 + (1 - l1) * t.Q.2)) →
    (∃ (l2 : ℝ), 0 ≤ l2 ∧ l2 ≤ 1 ∧ T = (l2 * t.P.1 + (1 - l2) * t.R.1, l2 * t.P.2 + (1 - l2) * t.R.2)) →
    minSum ≤ Real.sqrt ((S.1 - t.Q.1)^2 + (S.2 - t.Q.2)^2) + 
            Real.sqrt ((T.1 - S.1)^2 + (T.2 - S.2)^2) + 
            Real.sqrt ((t.R.1 - T.1)^2 + (t.R.2 - T.2)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_QS_ST_TR_l1097_109769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_obtuse_angles_at_intersection_l1097_109720

-- Define a structure for a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a structure for an intersection point
structure IntersectionPoint where
  x : ℝ
  y : ℝ

-- Define a function to check if two lines are perpendicular
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

-- Define a function to check if an angle is obtuse
def is_obtuse_angle (angle : ℝ) : Prop :=
  angle > Real.pi / 2 ∧ angle < Real.pi

-- Define a function to calculate the angle between two lines
noncomputable def angle_between_lines (l1 l2 : Line) : ℝ :=
  Real.arctan ((l2.slope - l1.slope) / (1 + l1.slope * l2.slope))

-- Theorem statement
theorem two_obtuse_angles_at_intersection 
  (l1 l2 : Line) 
  (intersection : IntersectionPoint) :
  ¬are_perpendicular l1 l2 →
  is_obtuse_angle (angle_between_lines l1 l2) →
  ∃ (count : ℕ), count = 2 ∧ 
    (∀ (angle : ℝ), 
      (angle = angle_between_lines l1 l2 ∨ 
       angle = Real.pi - angle_between_lines l1 l2) → 
      is_obtuse_angle angle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_obtuse_angles_at_intersection_l1097_109720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_seating_arrangements_l1097_109789

theorem dinner_seating_arrangements (n m : ℕ) (hn : n = 8) (hm : m = 7) :
  (n.choose m) * Nat.factorial (m - 1) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dinner_seating_arrangements_l1097_109789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_x_converges_to_3_l1097_109740

def sequence_x : ℕ → ℝ
| 0 => 1
| 1 => 9
| 2 => 9
| 3 => 1
| (n + 4) => (sequence_x n * sequence_x (n + 1) * sequence_x (n + 2) * sequence_x (n + 3)) ^ (1/4)

theorem sequence_x_converges_to_3 :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_x n - 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_x_converges_to_3_l1097_109740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_max_pressure_l1097_109763

/-- Represents the maximum pressure of an ideal gas in a cyclic process -/
noncomputable def max_pressure (R V₀ T₀ a b c : ℝ) : ℝ :=
  (R * T₀ / V₀) * (a * Real.sqrt (a^2 + b^2 - c^2) + b * c) /
  (b * Real.sqrt (a^2 + b^2 - c^2) - a * c)

/-- Theorem stating the maximum pressure for an ideal gas in a cyclic process -/
theorem ideal_gas_max_pressure
  (R V₀ T₀ a b c : ℝ)
  (h_positive : R > 0 ∧ V₀ > 0 ∧ T₀ > 0)
  (h_constraint : c^2 < a^2 + b^2) :
  ∃ (P_max : ℝ), ∀ (V T : ℝ),
    ((V / V₀ - a)^2 + (T / T₀ - b)^2 = c^2) →
    (R * T / V ≤ P_max) ∧
    (P_max = max_pressure R V₀ T₀ a b c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_gas_max_pressure_l1097_109763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_eq_solution_l1097_109787

def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n}

theorem A_intersect_B_eq_solution : A ∩ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_eq_solution_l1097_109787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_avg_math_biology_is_250_div_3_l1097_109746

-- Define the weightages for each subject
noncomputable def physics_weight : ℝ := 1.5
noncomputable def chemistry_weight : ℝ := 2
noncomputable def math_weight : ℝ := 1.25
noncomputable def biology_weight : ℝ := 1.75

-- Define the condition about total weighted marks
def total_weighted_marks_condition (P C M B : ℝ) : Prop :=
  physics_weight * P + chemistry_weight * C + math_weight * M + biology_weight * B =
  (physics_weight * P + chemistry_weight * C) + 250

-- Define the weighted average for math and biology
noncomputable def weighted_avg_math_biology (M B : ℝ) : ℝ :=
  (math_weight * M + biology_weight * B) / (math_weight + biology_weight)

-- Theorem statement
theorem weighted_avg_math_biology_is_250_div_3 
  (P C M B : ℝ) (h : total_weighted_marks_condition P C M B) :
  weighted_avg_math_biology M B = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_avg_math_biology_is_250_div_3_l1097_109746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sales_for_tom_l1097_109718

/-- Represents the least number of sales Tom must make per year to not lose money -/
def min_sales (old_salary : ℕ) (new_base_salary : ℕ) (commission_rate : ℚ) (sale_value : ℕ) : ℕ :=
  let commission_per_sale := (commission_rate * sale_value : ℚ)
  let salary_difference := old_salary - new_base_salary
  Nat.ceil ((salary_difference : ℚ) / commission_per_sale)

/-- Theorem stating the minimum number of sales Tom needs to make -/
theorem min_sales_for_tom :
  min_sales 75000 45000 (15 / 100) 750 = 267 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sales_for_tom_l1097_109718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1097_109770

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6)

theorem function_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (S : ℝ), (0 < S) → (∀ (x : ℝ), f (x + S) = f x) → (T ≤ S)) ∧
    (∀ (x : ℝ), f x = Real.sin (2 * (x + Real.pi / 12))) ∧
    (∀ (x : ℝ), f (Real.pi / 3 - x) = f (Real.pi / 3 + x)) ∧
    ¬(∀ (x y : ℝ), -Real.pi / 4 ≤ x ∧ x < y ∧ y ≤ Real.pi / 4 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1097_109770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_range_of_a_l1097_109726

theorem inequality_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2*x - 1| - |x + a| ≥ a) → a ≤ -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_range_of_a_l1097_109726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1097_109777

/-- The function we're maximizing -/
noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem f_max_value :
  (∀ t : ℝ, f t ≤ 1/16) ∧ (∃ t : ℝ, f t = 1/16) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1097_109777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1097_109753

open Real

/-- The function f(x) = x^2 - 2ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

/-- Theorem stating that the minimum value of f(x) is 1 -/
theorem f_min_value :
  ∀ x > 0, f x ≥ 1 ∧ ∃ x₀ > 0, f x₀ = 1 := by
  sorry

#check f_min_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1097_109753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_adjacent_probability_l1097_109710

/-- The probability of two runners on a regular hexagon being at adjacent vertices after t seconds -/
noncomputable def adjacentProbability (t : ℕ) : ℝ :=
  2/3 + 1/3 * (1/4)^t

/-- The probability that a runner moves left or right -/
noncomputable def moveProbability : ℝ := 1/2

/-- The theorem stating the probability of two runners on a regular hexagon 
    being at adjacent vertices after t seconds -/
theorem runners_adjacent_probability (t : ℕ) :
  let initialAdjacent : Bool := true  -- Runners start at adjacent vertices
  let hexagonVertices : ℕ := 6        -- Regular hexagon has 6 vertices
  adjacentProbability t = 2/3 + 1/3 * (1/4)^t := by
  sorry

#check runners_adjacent_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_adjacent_probability_l1097_109710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_closest_to_015_l1097_109796

theorem sqrt_difference_closest_to_015 :
  let diff := Real.sqrt 122 - Real.sqrt 120
  ∀ x ∈ ({0.14, 0.16, 0.17, 0.18} : Set ℝ), |diff - 0.15| < |diff - x| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_closest_to_015_l1097_109796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_min_value_on_interval_a_range_l1097_109771

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem f_increasing (a : ℝ) (h : a > 0) :
  StrictMonoOn (f a) (Set.Ioi 0) := by sorry

theorem min_value_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧ (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) := by sorry

theorem a_range (a : ℝ) :
  (∀ x > 1, f a x < x^2) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_min_value_on_interval_a_range_l1097_109771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_digit_sum_l1097_109749

/-- sum_of_digits m returns the sum of the digits of the natural number m -/
def sum_of_digits (m : ℕ) : ℕ :=
  sorry

/-- For any positive integer n, there exists a multiple of n whose sum of digits is n. -/
theorem multiple_with_digit_sum (n : ℕ) (hn : n ≥ 1) :
  ∃ m : ℕ, (n ∣ m) ∧ (sum_of_digits m = n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_with_digit_sum_l1097_109749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1097_109773

theorem cos_double_angle_special_case (α : ℝ) (h : Real.sin α = Real.sqrt 3 / 3) :
  Real.cos (2 * α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l1097_109773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1097_109733

/-- A pyramid with a rectangular base -/
structure Pyramid where
  base_length : ℝ
  base_width : ℝ
  slant_height : ℝ

/-- Volume of a pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  (1 / 3) * p.base_length * p.base_width * 
    Real.sqrt ((p.slant_height)^2 - (p.base_width / 2)^2 - (p.base_length / 2)^2)

/-- The theorem stating the volume of the specific pyramid -/
theorem pyramid_volume :
  let p : Pyramid := {
    base_length := 4 * Real.sqrt 2,
    base_width := 2 * Real.sqrt 2,
    slant_height := 6 * Real.sqrt 2
  }
  volume p = 128 / 3 := by
  sorry

#eval "Pyramid volume theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l1097_109733
