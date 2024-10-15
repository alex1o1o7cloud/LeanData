import Mathlib

namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2397_239788

theorem cube_root_equation_solution :
  let x : ℝ := 168 / 5
  (15 * x + (15 * x + 8) ^ (1/3)) ^ (1/3) = 8 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2397_239788


namespace NUMINAMATH_CALUDE_percentage_problem_l2397_239717

theorem percentage_problem (x : ℝ) (h : 0.25 * x = 70) : x = 280 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2397_239717


namespace NUMINAMATH_CALUDE_line_through_origin_and_intersection_l2397_239718

/-- The equation of the line passing through the origin and the intersection point of two given lines -/
theorem line_through_origin_and_intersection (x y : ℝ) : 
  (x - 3*y + 4 = 0 ∧ 2*x + y + 5 = 0) → 
  (3*x + 19*y = 0) := by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_intersection_l2397_239718


namespace NUMINAMATH_CALUDE_add_fractions_three_ninths_seven_twelfths_l2397_239753

theorem add_fractions_three_ninths_seven_twelfths :
  3 / 9 + 7 / 12 = 11 / 12 := by sorry

end NUMINAMATH_CALUDE_add_fractions_three_ninths_seven_twelfths_l2397_239753


namespace NUMINAMATH_CALUDE_arithmetic_progression_sine_squared_l2397_239701

theorem arithmetic_progression_sine_squared (x y z α : Real) : 
  (y = (x + z) / 2) →  -- x, y, z form an arithmetic progression
  (α = Real.arcsin (Real.sqrt 7 / 4)) →  -- α is defined as arcsin(√7/4)
  (8 / Real.sin y = 1 / Real.sin x + 1 / Real.sin z) →  -- 1/sin(x), 4/sin(y), 1/sin(z) form an arithmetic progression
  Real.sin y ^ 2 = 7 / 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sine_squared_l2397_239701


namespace NUMINAMATH_CALUDE_marching_band_weight_is_245_l2397_239763

/-- Represents the total weight carried by the Oprah Winfrey High School marching band. -/
def marching_band_weight : ℕ :=
  let trumpet_clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpet_count := 6
  let clarinet_count := 9
  let trombone_count := 8
  let tuba_count := 3
  let drum_count := 2
  (trumpet_clarinet_weight * (trumpet_count + clarinet_count)) +
  (trombone_weight * trombone_count) +
  (tuba_weight * tuba_count) +
  (drum_weight * drum_count)

/-- Theorem stating that the total weight carried by the marching band is 245 pounds. -/
theorem marching_band_weight_is_245 : marching_band_weight = 245 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_weight_is_245_l2397_239763


namespace NUMINAMATH_CALUDE_two_digit_integers_theorem_l2397_239750

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def share_digit (a b : ℕ) : Prop :=
  (a / 10 = b / 10) ∨ (a / 10 = b % 10) ∨ (a % 10 = b / 10) ∨ (a % 10 = b % 10)

theorem two_digit_integers_theorem (a b : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧
  ((a = b + 12) ∨ (b = a + 12)) ∧
  share_digit a b ∧
  ((digit_sum a = digit_sum b + 3) ∨ (digit_sum b = digit_sum a + 3)) →
  (∃ t : ℕ, 2 ≤ t ∧ t ≤ 8 ∧ a = 11 * t + 10 ∧ b = 11 * t - 2) ∨
  (∃ s : ℕ, 1 ≤ s ∧ s ≤ 6 ∧ a = 11 * s + 1 ∧ b = 11 * s + 13) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_integers_theorem_l2397_239750


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2397_239767

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^3 + 21*y^2 + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2397_239767


namespace NUMINAMATH_CALUDE_ratio_comparison_l2397_239754

theorem ratio_comparison (y : ℕ) (h : y > 4) : (3 : ℚ) / 4 < 4 / y := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_l2397_239754


namespace NUMINAMATH_CALUDE_bologna_sandwich_count_l2397_239728

/-- Represents the number of sandwiches of each type -/
structure SandwichCount where
  cheese : ℕ
  bologna : ℕ
  peanutButter : ℕ

/-- The ratio of sandwiches -/
def sandwichRatio : ℕ → SandwichCount
  | x => { cheese := 1, bologna := x, peanutButter := 8 }

/-- The total number of sandwiches -/
def totalSandwiches : ℕ := 80

theorem bologna_sandwich_count :
  ∃ x : ℕ, 
    let ratio := sandwichRatio x
    (ratio.cheese + ratio.bologna + ratio.peanutButter) * y = totalSandwiches →
    ratio.bologna * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_bologna_sandwich_count_l2397_239728


namespace NUMINAMATH_CALUDE_abs_negative_two_l2397_239746

theorem abs_negative_two : |(-2 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l2397_239746


namespace NUMINAMATH_CALUDE_angle_between_vectors_l2397_239789

theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (Real.sqrt 3, -1) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l2397_239789


namespace NUMINAMATH_CALUDE_chessTeamArrangements_eq_12_l2397_239704

/-- The number of ways to arrange a chess team with 3 boys and 2 girls in a specific order -/
def chessTeamArrangements : ℕ :=
  let numBoys : ℕ := 3
  let numGirls : ℕ := 2
  let girlArrangements : ℕ := Nat.factorial numGirls
  let boyArrangements : ℕ := Nat.factorial numBoys
  girlArrangements * boyArrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem chessTeamArrangements_eq_12 : chessTeamArrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_chessTeamArrangements_eq_12_l2397_239704


namespace NUMINAMATH_CALUDE_smallest_divisible_by_20_and_63_l2397_239727

theorem smallest_divisible_by_20_and_63 : ∀ n : ℕ, n > 0 ∧ 20 ∣ n ∧ 63 ∣ n → n ≥ 1260 := by
  sorry

#check smallest_divisible_by_20_and_63

end NUMINAMATH_CALUDE_smallest_divisible_by_20_and_63_l2397_239727


namespace NUMINAMATH_CALUDE_three_pieces_per_box_l2397_239715

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the tape needed for a single box -/
def tapeForBox (box : BoxDimensions) : ℕ :=
  box.length + 2 * box.width

/-- The number of 15x30 boxes -/
def numSmallBoxes : ℕ := 5

/-- The number of 40x40 boxes -/
def numLargeBoxes : ℕ := 2

/-- The dimensions of the small boxes -/
def smallBox : BoxDimensions :=
  { length := 30, width := 15 }

/-- The dimensions of the large boxes -/
def largeBox : BoxDimensions :=
  { length := 40, width := 40 }

/-- The total amount of tape needed -/
def totalTape : ℕ := 540

/-- Theorem: Each box needs 3 pieces of tape -/
theorem three_pieces_per_box :
  (∃ (n : ℕ), n > 0 ∧
    n * (numSmallBoxes * tapeForBox smallBox + numLargeBoxes * tapeForBox largeBox) = totalTape * n ∧
    n * (numSmallBoxes + numLargeBoxes) = 3 * n * (numSmallBoxes + numLargeBoxes)) := by
  sorry


end NUMINAMATH_CALUDE_three_pieces_per_box_l2397_239715


namespace NUMINAMATH_CALUDE_quadratic_and_line_properties_l2397_239736

/-- Given a quadratic equation with two equal real roots, prove the value of m and the quadrants through which the corresponding line passes -/
theorem quadratic_and_line_properties :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 + (2*m + 1)*x + m^2 + 2 = 0 → (∃! r : ℝ, x = r)) →
  (m = 7/4 ∧ 
   ∀ x y : ℝ, y = (2*m - 3)*x - 4*m + 6 →
   (∃ x₁ y₁ : ℝ, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = (2*m - 3)*x₁ - 4*m + 6) ∧
   (∃ x₂ y₂ : ℝ, x₂ < 0 ∧ y₂ < 0 ∧ y₂ = (2*m - 3)*x₂ - 4*m + 6) ∧
   (∃ x₃ y₃ : ℝ, x₃ > 0 ∧ y₃ < 0 ∧ y₃ = (2*m - 3)*x₃ - 4*m + 6)) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_and_line_properties_l2397_239736


namespace NUMINAMATH_CALUDE_tan_2alpha_value_l2397_239787

theorem tan_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) - 4 * Real.sin α = 1) : 
  Real.tan (2 * α) = -4 * Real.sqrt 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_value_l2397_239787


namespace NUMINAMATH_CALUDE_distance_after_translation_l2397_239731

/-- Given two points A and B in a 2D plane, and a translation vector,
    prove that the distance between A and the translated B is √153. -/
theorem distance_after_translation :
  let A : ℝ × ℝ := (2, -2)
  let B : ℝ × ℝ := (8, 6)
  let translation : ℝ × ℝ := (-3, 4)
  let C : ℝ × ℝ := (B.1 + translation.1, B.2 + translation.2)
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_translation_l2397_239731


namespace NUMINAMATH_CALUDE_system_solution_l2397_239783

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (x^4 / y^2)^(Real.log y) = (-x)^(Real.log (-x*y))

def equation2 (x y : ℝ) : Prop :=
  2*y^2 - x*y - x^2 - 4*x - 8*y = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(-4, 2), (-2, 2), ((Real.sqrt 17 - 9)/2, (Real.sqrt 17 - 1)/2)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2397_239783


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l2397_239796

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- Line l₁ with equation x + (1+m)y = 2 - m -/
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  x + (1+m)*y = 2 - m

/-- Line l₂ with equation 2mx + 4y + 16 = 0 -/
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  2*m*x + 4*y + 16 = 0

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel 1 (1+m) (2*m) 4 → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l2397_239796


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_lines_parallel_l2397_239734

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Axioms for different objects
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem 1: Transitivity of parallel planes
theorem parallel_transitive :
  parallel α β → parallel β γ → parallel α γ :=
sorry

-- Theorem 2: Lines perpendicular to the same plane are parallel
theorem perpendicular_lines_parallel :
  perpendicular m α → perpendicular n α → lineParallel m n :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_lines_parallel_l2397_239734


namespace NUMINAMATH_CALUDE_find_a_l2397_239791

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x + 1/x)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The coefficient of x^k in the expansion of (x + 1/x)^n -/
def coefficient (n k : ℕ) : ℝ := sorry

theorem find_a : ∃ (a : ℝ), 
  (coefficient 6 3 * a + coefficient 6 2) = 30 ∧ 
  ∀ (b : ℝ), (coefficient 6 3 * b + coefficient 6 2) = 30 → b = a :=
sorry

end NUMINAMATH_CALUDE_find_a_l2397_239791


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l2397_239705

-- Define an acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem double_acute_angle_range (θ : ℝ) (h : is_acute_angle θ) :
  0 < 2 * θ ∧ 2 * θ < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l2397_239705


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_l2397_239774

def S : Finset Int := {-5, 30, -2, 15, -4}

theorem smallest_sum_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x + y + z = -11 ∧
  ∀ (d e f : Int), d ∈ S → e ∈ S → f ∈ S → 
  d ≠ e ∧ e ≠ f ∧ d ≠ f → 
  d + e + f ≥ -11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_l2397_239774


namespace NUMINAMATH_CALUDE_mans_downstream_speed_l2397_239747

/-- Proves that given a man's upstream speed of 30 kmph and still water speed of 35 kmph, his downstream speed is 40 kmph. -/
theorem mans_downstream_speed 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_speed = 30) 
  (h2 : still_water_speed = 35) : 
  still_water_speed + (still_water_speed - upstream_speed) = 40 := by
  sorry

#check mans_downstream_speed

end NUMINAMATH_CALUDE_mans_downstream_speed_l2397_239747


namespace NUMINAMATH_CALUDE_initial_sale_percentage_l2397_239793

theorem initial_sale_percentage (P : ℝ) (x : ℝ) (h : x ≥ 0 ∧ x ≤ 1) : 
  ((1 - x) * P * 0.9 = 0.45 * P) → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_initial_sale_percentage_l2397_239793


namespace NUMINAMATH_CALUDE_hyperbola_focus_product_l2397_239733

-- Define the hyperbola C
def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 9 - y^2 / m = 1

-- Define the foci F₁ and F₂
def is_focus (F : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola x y m ∧ 
  ((F.1 - x)^2 + (F.2 - y)^2 = 16 ∨ (F.1 - x)^2 + (F.2 - y)^2 = 16)

-- Define point P on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) (m : ℝ) : Prop :=
  hyperbola P.1 P.2 m

-- Define the dot product condition
def perpendicular_vectors (P F₁ F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Define the directrix condition
def directrix_through_focus (F : ℝ × ℝ) : Prop :=
  F.1 = -4

-- Main theorem
theorem hyperbola_focus_product (m : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_focus F₁ m →
  is_focus F₂ m →
  point_on_hyperbola P m →
  perpendicular_vectors P F₁ F₂ →
  (directrix_through_focus F₁ ∨ directrix_through_focus F₂) →
  ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 14^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_product_l2397_239733


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2397_239710

theorem quadratic_inequality_solution (x : ℝ) : 
  3 * x^2 + 4 * x - 9 < 0 ∧ x ≥ -2 → -2 ≤ x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2397_239710


namespace NUMINAMATH_CALUDE_opposite_numbers_iff_different_sign_l2397_239709

/-- Two real numbers are opposite if and only if they differ only in sign -/
theorem opposite_numbers_iff_different_sign (a b : ℝ) : 
  (a = -b) ↔ (abs a = abs b) :=
sorry

end NUMINAMATH_CALUDE_opposite_numbers_iff_different_sign_l2397_239709


namespace NUMINAMATH_CALUDE_compare_negative_roots_l2397_239792

theorem compare_negative_roots : -3 * Real.sqrt 3 > -2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_roots_l2397_239792


namespace NUMINAMATH_CALUDE_solve_equation_l2397_239770

theorem solve_equation (x : ℚ) (h : 5 * x - 8 = 15 * x + 18) : 3 * (x + 9) = 96 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2397_239770


namespace NUMINAMATH_CALUDE_m_range_theorem_l2397_239778

/-- The equation x^2 + mx + 1 = 0 has two real roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- ∀x ∈ ℝ, 4x^2 + 4(m-2)x + 1 ≠ 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The range of values for m is (1, 2) -/
def range_m : Set ℝ := { m | 1 < m ∧ m < 2 }

theorem m_range_theorem (m : ℝ) :
  (¬(p m ∧ q m)) ∧ (¬¬(q m)) → m ∈ range_m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2397_239778


namespace NUMINAMATH_CALUDE_spending_ratio_l2397_239782

-- Define the amounts spent by each person
def akeno_spent : ℚ := 2985
def lev_spent : ℚ := 995  -- This is derived from the solution, but we'll use it as a given
def ambrocio_spent : ℚ := lev_spent - 177

-- State the theorem
theorem spending_ratio :
  -- Conditions
  (akeno_spent = lev_spent + ambrocio_spent + 1172) →
  (ambrocio_spent = lev_spent - 177) →
  -- Conclusion
  (lev_spent / akeno_spent = 1 / 3) := by
sorry


end NUMINAMATH_CALUDE_spending_ratio_l2397_239782


namespace NUMINAMATH_CALUDE_first_year_interest_rate_l2397_239764

/-- Proves that the first-year interest rate is 4% given the problem conditions --/
theorem first_year_interest_rate (initial_amount : ℝ) (final_amount : ℝ) (second_year_rate : ℝ) :
  initial_amount = 5000 →
  final_amount = 5460 →
  second_year_rate = 0.05 →
  ∃ (first_year_rate : ℝ),
    final_amount = initial_amount * (1 + first_year_rate) * (1 + second_year_rate) ∧
    first_year_rate = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_first_year_interest_rate_l2397_239764


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l2397_239780

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The percentage of mixture X in the final blend -/
def x_percentage : ℝ := 13.333333333333332

/-- The percentage of ryegrass in the final blend -/
def final_ryegrass_percentage : ℝ := 27

/-- Seed mixture X -/
def mixture_x : SeedMixture where
  ryegrass := 40  -- This is what we want to prove
  bluegrass := 60
  fescue := 0

/-- Seed mixture Y -/
def mixture_y : SeedMixture where
  ryegrass := 25
  bluegrass := 0
  fescue := 75

theorem ryegrass_percentage_in_x : 
  (mixture_x.ryegrass * x_percentage + mixture_y.ryegrass * (100 - x_percentage)) / 100 = final_ryegrass_percentage := by
  sorry

#check ryegrass_percentage_in_x

end NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l2397_239780


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2397_239786

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2397_239786


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l2397_239729

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let rec bounceSum (height : ℝ) (n : ℕ) : ℝ :=
    if n = 0 then 0
    else height + height * bounceRatio + bounceSum (height * bounceRatio) (n - 1)
  initialHeight + 2 * bounceSum initialHeight numBounces

/-- The bouncing ball problem -/
theorem bouncing_ball_distance :
  ∃ (d : ℝ), abs (d - totalDistance 20 (2/3) 4) < 0.5 ∧ Int.floor d = 80 := by
  sorry


end NUMINAMATH_CALUDE_bouncing_ball_distance_l2397_239729


namespace NUMINAMATH_CALUDE_largest_square_with_four_interior_lattice_points_l2397_239756

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square in the plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Counts the number of lattice points strictly inside a square -/
def count_interior_lattice_points (s : Square) : ℕ :=
  sorry

/-- The theorem stating the area of the largest square with exactly 4 interior lattice points -/
theorem largest_square_with_four_interior_lattice_points :
  ∃ (s : Square),
    (count_interior_lattice_points s = 4) ∧
    (∀ (t : Square), count_interior_lattice_points t = 4 → t.side_length ≤ s.side_length) ∧
    (9 < s.side_length ^ 2) ∧ (s.side_length ^ 2 < 10) :=
  sorry

end NUMINAMATH_CALUDE_largest_square_with_four_interior_lattice_points_l2397_239756


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2397_239758

theorem consecutive_integers_sum (a b c : ℤ) : 
  b = 19 ∧ c = b + 1 ∧ a = b - 1 → a + b + c = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2397_239758


namespace NUMINAMATH_CALUDE_set_one_two_three_not_triangle_l2397_239795

/-- Triangle Inequality Theorem: A set of three positive real numbers a, b, and c can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set {1, 2, 3} cannot form a triangle. -/
theorem set_one_two_three_not_triangle :
  ¬ can_form_triangle 1 2 3 := by
  sorry

#check set_one_two_three_not_triangle

end NUMINAMATH_CALUDE_set_one_two_three_not_triangle_l2397_239795


namespace NUMINAMATH_CALUDE_max_rented_trucks_is_twenty_l2397_239755

/-- Represents the truck rental scenario for a week -/
structure TruckRental where
  total : ℕ
  returned_percent : ℚ
  saturday_minimum : ℕ

/-- Calculates the maximum number of trucks that could have been rented out -/
def max_rented_trucks (rental : TruckRental) : ℕ :=
  min rental.total (2 * rental.saturday_minimum)

/-- Theorem stating the maximum number of trucks that could have been rented out -/
theorem max_rented_trucks_is_twenty (rental : TruckRental) :
    rental.total = 20 ∧ 
    rental.returned_percent = 1/2 ∧ 
    rental.saturday_minimum = 10 →
    max_rented_trucks rental = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_rented_trucks_is_twenty_l2397_239755


namespace NUMINAMATH_CALUDE_equality_check_l2397_239799

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-2)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-(-2) ≠ -|-2|) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l2397_239799


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2397_239737

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2397_239737


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2397_239720

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  ∃ k : ℕ, k = 3 ∧
  (∀ m : ℕ, 2012^m ∣ factorial 2012 → m ≤ k) ∧
  2012^k ∣ factorial 2012 :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2397_239720


namespace NUMINAMATH_CALUDE_chairs_moved_by_alex_l2397_239700

/-- Given that Carey moves x chairs, Pat moves y chairs, and Alex moves z chairs,
    with a total of 74 chairs to be moved, prove that the number of chairs
    Alex moves is equal to 74 minus the sum of chairs moved by Carey and Pat. -/
theorem chairs_moved_by_alex (x y z : ℕ) (h : x + y + z = 74) :
  z = 74 - x - y := by sorry

end NUMINAMATH_CALUDE_chairs_moved_by_alex_l2397_239700


namespace NUMINAMATH_CALUDE_buttons_solution_l2397_239725

def buttons_problem (mari kendra sue will lea : ℕ) : Prop :=
  mari = 8 ∧
  kendra = 5 * mari + 4 ∧
  sue = kendra / 2 ∧
  will = (5 * (kendra + sue)) / 2 ∧
  lea = will - will / 5

theorem buttons_solution :
  ∃ (mari kendra sue will lea : ℕ),
    buttons_problem mari kendra sue will lea ∧ lea = 132 := by
  sorry

end NUMINAMATH_CALUDE_buttons_solution_l2397_239725


namespace NUMINAMATH_CALUDE_truck_sales_l2397_239719

theorem truck_sales (total_vehicles : ℕ) (car_truck_difference : ℕ) : 
  total_vehicles = 69 → car_truck_difference = 27 → 
  ∃ trucks : ℕ, trucks = 21 ∧ trucks + (trucks + car_truck_difference) = total_vehicles :=
by
  sorry

end NUMINAMATH_CALUDE_truck_sales_l2397_239719


namespace NUMINAMATH_CALUDE_expression_value_l2397_239716

theorem expression_value (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = 3) : 
  10 * a + 10 * b + c * d * x = 3 ∨ 10 * a + 10 * b + c * d * x = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2397_239716


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2397_239738

theorem quadratic_inequality (a b c A B C : ℝ) (ha : a ≠ 0) (hA : A ≠ 0)
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4*a*c| ≤ |B^2 - 4*A*C| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2397_239738


namespace NUMINAMATH_CALUDE_root_sum_squares_l2397_239711

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2397_239711


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l2397_239752

theorem cos_two_alpha_value (α : Real) (h1 : π/8 < α) (h2 : α < 3*π/8) : 
  let f := fun x => Real.cos x * (Real.sin x + Real.cos x) - 1/2
  f α = Real.sqrt 2 / 6 → Real.cos (2 * α) = (Real.sqrt 2 - 4) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l2397_239752


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2397_239797

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem stating that the base k in which 167₈ is written as 315ₖ is equal to 6 -/
theorem base_conversion_theorem : 
  ∃ k : ℕ, k > 1 ∧ base8ToBase10 167 = baseKToBase10 315 k ∧ k = 6 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2397_239797


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2397_239759

def original_price : ℝ := 200
def tuesday_discount : ℝ := 0.40
def thursday_discount : ℝ := 0.25

theorem bicycle_price_after_discounts :
  original_price * (1 - tuesday_discount) * (1 - thursday_discount) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2397_239759


namespace NUMINAMATH_CALUDE_odd_function_minimum_value_l2397_239735

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- F is defined as a linear combination of f and x, plus a constant -/
def F (f : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ :=
  a * f x + b * x + 1

theorem odd_function_minimum_value
    (f : ℝ → ℝ) (a b : ℝ)
    (h_odd : IsOdd f)
    (h_max : ∀ x > 0, F f a b x ≤ 2) :
    ∀ x < 0, F f a b x ≥ 0 :=
  sorry

end NUMINAMATH_CALUDE_odd_function_minimum_value_l2397_239735


namespace NUMINAMATH_CALUDE_direct_proportion_through_points_l2397_239703

/-- A direct proportion function passing through (-1, 2) also passes through (1, -2) -/
theorem direct_proportion_through_points :
  ∀ (f : ℝ → ℝ) (k : ℝ),
    (∀ x, f x = k * x) →  -- f is a direct proportion function
    f (-1) = 2 →          -- f passes through (-1, 2)
    f 1 = -2 :=           -- f passes through (1, -2)
by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_through_points_l2397_239703


namespace NUMINAMATH_CALUDE_function_is_linear_l2397_239760

/-- Given a real number k, we define a function f that satisfies two conditions -/
def satisfies_conditions (k : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) ∧ 
  (∀ x : ℝ, |f x - k*x| ≤ |x^2 - x|)

/-- Theorem stating that if f satisfies the conditions, then f(x) = kx for all x ∈ ℝ -/
theorem function_is_linear (k : ℝ) (f : ℝ → ℝ) 
  (h : satisfies_conditions k f) : 
  ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_function_is_linear_l2397_239760


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l2397_239777

/-- The cost of a luncheon item combination -/
structure LuncheonCost where
  sandwiches : ℕ
  coffees : ℕ
  pies : ℕ
  total : ℚ

/-- The given luncheon costs -/
def givenLuncheons : List LuncheonCost := [
  ⟨5, 9, 2, 595/100⟩,
  ⟨7, 12, 2, 790/100⟩,
  ⟨3, 5, 1, 350/100⟩
]

/-- The theorem to prove -/
theorem luncheon_cost_theorem (s c p : ℚ) 
  (h1 : 5*s + 9*c + 2*p = 595/100)
  (h2 : 7*s + 12*c + 2*p = 790/100)
  (h3 : 3*s + 5*c + p = 350/100) :
  s + c + p = 105/100 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l2397_239777


namespace NUMINAMATH_CALUDE_farmer_apples_l2397_239732

/-- The number of apples the farmer gave away -/
def apples_given_away : ℕ := 88

/-- The number of apples the farmer has left -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given_away + apples_left

theorem farmer_apples : initial_apples = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l2397_239732


namespace NUMINAMATH_CALUDE_range_of_a_l2397_239748

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 < 0) →
  a ∈ Set.Ioc (-8) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2397_239748


namespace NUMINAMATH_CALUDE_fraction_simplification_l2397_239744

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2397_239744


namespace NUMINAMATH_CALUDE_fraction_leading_zeros_l2397_239724

/-- The number of leading zeros in the decimal representation of a rational number -/
def leadingZeros (q : ℚ) : ℕ := sorry

/-- The fraction we're analyzing -/
def fraction : ℚ := 1 / (2^7 * 5^9)

theorem fraction_leading_zeros : leadingZeros fraction = 8 := by sorry

end NUMINAMATH_CALUDE_fraction_leading_zeros_l2397_239724


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2397_239743

/-- The sum of the geometric series with a specific pattern -/
theorem geometric_series_sum : 
  ∑' k : ℕ, (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2397_239743


namespace NUMINAMATH_CALUDE_product_with_seven_zeros_is_odd_l2397_239751

def binary_num (n : ℕ) : Prop := ∀ d : ℕ, d ∈ n.digits 2 → d = 0 ∨ d = 1

def count_zeros (n : ℕ) : ℕ := (n.digits 2).filter (· = 0) |>.length

theorem product_with_seven_zeros_is_odd (m : ℕ) :
  binary_num m →
  count_zeros (17 * m) = 7 →
  Odd (17 * m) :=
by sorry

end NUMINAMATH_CALUDE_product_with_seven_zeros_is_odd_l2397_239751


namespace NUMINAMATH_CALUDE_power_of_special_sum_l2397_239742

theorem power_of_special_sum (a b : ℝ) 
  (h : b = Real.sqrt (1 - 2*a) + Real.sqrt (2*a - 1) + 3) : 
  a ^ b = (1/8 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_special_sum_l2397_239742


namespace NUMINAMATH_CALUDE_find_a_l2397_239784

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x - 1) / (x + 1) > 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | x < -1 ∨ x > 1/2}

-- Theorem statement
theorem find_a : ∃ a : ℝ, ∀ x : ℝ, inequality a x ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_find_a_l2397_239784


namespace NUMINAMATH_CALUDE_exist_two_players_with_eight_points_no_undefeated_pair_from_sixth_round_l2397_239776

/-- Represents a chess tournament with the given rules --/
structure ChessTournament where
  players : ℕ
  rounds : ℕ
  win_points : ℚ
  draw_points : ℚ
  loss_points : ℚ
  bye_points : ℚ
  max_byes : ℕ

/-- Defines the specific tournament in the problem --/
def problem_tournament : ChessTournament :=
  { players := 29
  , rounds := 9
  , win_points := 1
  , draw_points := 1/2
  , loss_points := 0
  , bye_points := 1
  , max_byes := 1 }

/-- Represents the state of a player after a certain number of rounds --/
structure PlayerState where
  wins : ℕ
  losses : ℕ
  byes : ℕ

/-- Calculates the total points for a player --/
def total_points (t : ChessTournament) (p : PlayerState) : ℚ :=
  p.wins * t.win_points + p.losses * t.loss_points + min p.byes t.max_byes * t.bye_points

/-- Theorem stating that two players can have 8 points each before the final round --/
theorem exist_two_players_with_eight_points (t : ChessTournament) :
  t = problem_tournament →
  ∃ (p1 p2 : PlayerState),
    total_points t p1 = 8 ∧
    total_points t p2 = 8 ∧
    p1.wins + p1.losses + p1.byes < t.rounds ∧
    p2.wins + p2.losses + p2.byes < t.rounds :=
  sorry

/-- Theorem stating that from the 6th round, no two undefeated players can meet --/
theorem no_undefeated_pair_from_sixth_round (t : ChessTournament) :
  t = problem_tournament →
  ∀ (r : ℕ), r ≥ 6 →
  ¬∃ (p1 p2 : PlayerState),
    p1.wins = r - 1 ∧
    p2.wins = r - 1 ∧
    p1.losses = 0 ∧
    p2.losses = 0 :=
  sorry

end NUMINAMATH_CALUDE_exist_two_players_with_eight_points_no_undefeated_pair_from_sixth_round_l2397_239776


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l2397_239713

/-- The total expenditure to cover the floor of a rectangular hall with a mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Theorem: The total expenditure to cover the floor of a rectangular hall
    with dimensions 20 m × 15 m × 5 m using a mat that costs Rs. 40 per square meter
    is equal to Rs. 12,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 40 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_hall_mat_expenditure_l2397_239713


namespace NUMINAMATH_CALUDE_figure_to_square_l2397_239745

/-- Represents a figure on a grid --/
structure GridFigure where
  area : ℕ

/-- Represents a cut of the figure --/
inductive Cut
  | Part : Cut

/-- Represents the result of cutting and arranging --/
inductive Arrangement
  | Square : Arrangement

/-- Theorem: If a grid figure's area is a perfect square, 
    it can be cut into three parts and arranged into a square --/
theorem figure_to_square (f : GridFigure) 
  (h : ∃ n : ℕ, f.area = n * n) : 
  ∃ (c1 c2 c3 : Cut) (arr : Arrangement), 
    arr = Arrangement.Square := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l2397_239745


namespace NUMINAMATH_CALUDE_prop_p_or_q_false_iff_a_in_range_l2397_239740

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, a^2 * x^2 + a * x - 2 = 0 ∧ -1 ≤ x ∧ x ≤ 1

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- Theorem statement
theorem prop_p_or_q_false_iff_a_in_range (a : ℝ) :
  (¬(p a ∨ q a)) ↔ ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)) :=
sorry

end NUMINAMATH_CALUDE_prop_p_or_q_false_iff_a_in_range_l2397_239740


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2397_239714

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  e : ℝ  -- last term
  y : ℝ  -- middle term

/-- Theorem: In an arithmetic sequence with 5 terms, where 12 is the first term,
    56 is the last term, and y is the middle term, y equals 34. -/
theorem arithmetic_sequence_middle_term 
  (seq : ArithmeticSequence5) 
  (h1 : seq.a = 12) 
  (h2 : seq.e = 56) : 
  seq.y = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2397_239714


namespace NUMINAMATH_CALUDE_solution_set_l2397_239762

theorem solution_set (x : ℝ) :
  (x - 2) / (x - 4) ≥ 3 ∧ x ≠ 2 → 4 < x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l2397_239762


namespace NUMINAMATH_CALUDE_greater_number_problem_l2397_239726

theorem greater_number_problem (x y : ℝ) : 
  y = 2 * x ∧ x + y = 96 → y = 64 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l2397_239726


namespace NUMINAMATH_CALUDE_bhanu_petrol_expense_l2397_239739

def bhanu_expenditure (total_income : ℝ) : Prop :=
  let petrol_percent : ℝ := 0.30
  let rent_percent : ℝ := 0.30
  let petrol_expense : ℝ := petrol_percent * total_income
  let remaining_after_petrol : ℝ := total_income - petrol_expense
  let rent_expense : ℝ := rent_percent * remaining_after_petrol
  rent_expense = 210 ∧ petrol_expense = 300

theorem bhanu_petrol_expense : 
  ∃ (total_income : ℝ), bhanu_expenditure total_income :=
sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expense_l2397_239739


namespace NUMINAMATH_CALUDE_average_problem_l2397_239785

theorem average_problem (x : ℝ) : (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2397_239785


namespace NUMINAMATH_CALUDE_f_max_on_interval_f_min_on_interval_f_max_attained_f_min_attained_l2397_239765

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f satisfies f(x+y) = f(x) + f(y) for all x, y -/
axiom f_additive : ∀ x y, f (x + y) = f x + f y

/-- f(x) < 0 when x > 0 -/
axiom f_neg_when_pos : ∀ x, x > 0 → f x < 0

/-- f(1) = -2 -/
axiom f_one : f 1 = -2

/-- The maximum value of f on [-3, 3] is 6 -/
theorem f_max_on_interval : 
  ∀ x, x ∈ Set.Icc (-3) 3 → f x ≤ 6 :=
sorry

/-- The minimum value of f on [-3, 3] is -6 -/
theorem f_min_on_interval : 
  ∀ x, x ∈ Set.Icc (-3) 3 → f x ≥ -6 :=
sorry

/-- The maximum value 6 is attained at -3 -/
theorem f_max_attained : f (-3) = 6 :=
sorry

/-- The minimum value -6 is attained at 3 -/
theorem f_min_attained : f 3 = -6 :=
sorry

end NUMINAMATH_CALUDE_f_max_on_interval_f_min_on_interval_f_max_attained_f_min_attained_l2397_239765


namespace NUMINAMATH_CALUDE_unique_intersection_l2397_239706

/-- The curve C in the xy-plane -/
def curve (x y : ℝ) : Prop :=
  y = x^2 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

/-- The line l in the xy-plane -/
def line (x y : ℝ) : Prop :=
  y - x = 2

/-- The intersection point of the curve and the line -/
def intersection_point : ℝ × ℝ := (-1, 1)

theorem unique_intersection :
  ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line p.1 p.2 ∧ p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l2397_239706


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_function_l2397_239775

theorem min_value_of_quadratic_function :
  ∃ (min : ℝ), min = -1 ∧ ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x + 8 * y + 15 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_function_l2397_239775


namespace NUMINAMATH_CALUDE_james_total_earnings_l2397_239773

def january_earnings : ℕ := 4000

def february_earnings : ℕ := 2 * january_earnings

def march_earnings : ℕ := february_earnings - 2000

def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings : total_earnings = 18000 := by
  sorry

end NUMINAMATH_CALUDE_james_total_earnings_l2397_239773


namespace NUMINAMATH_CALUDE_average_difference_l2397_239722

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2397_239722


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2397_239768

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a - 8 > b - 8) → ¬(a > b)) ↔ (a - 8 ≤ b - 8 → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2397_239768


namespace NUMINAMATH_CALUDE_sum_of_squares_l2397_239749

theorem sum_of_squares (x y : ℝ) : 
  x * y = 10 → x^2 * y + x * y^2 + x + y = 80 → x^2 + y^2 = 3980 / 121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2397_239749


namespace NUMINAMATH_CALUDE_polynomial_is_square_of_binomial_l2397_239721

/-- The polynomial 4x^2 + 16x + 16 is the square of a binomial. -/
theorem polynomial_is_square_of_binomial :
  ∃ (r s : ℝ), ∀ x, 4 * x^2 + 16 * x + 16 = (r * x + s)^2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_is_square_of_binomial_l2397_239721


namespace NUMINAMATH_CALUDE_cookie_radius_l2397_239730

theorem cookie_radius (x y : ℝ) :
  x^2 + y^2 + 26 = 6*x + 12*y →
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l2397_239730


namespace NUMINAMATH_CALUDE_problem_solution_l2397_239790

theorem problem_solution (x : ℝ) : (0.65 * x = 0.20 * 422.50) → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2397_239790


namespace NUMINAMATH_CALUDE_angle_trigonometry_l2397_239794

open Real

-- Define the angle θ
variable (θ : ℝ)

-- Define the condition that the terminal side of θ lies on y = 2x (x ≥ 0)
def terminal_side_condition (θ : ℝ) : Prop :=
  ∃ (x : ℝ), x ≥ 0 ∧ tan θ = 2

-- Theorem statement
theorem angle_trigonometry (h : terminal_side_condition θ) :
  (tan θ = 2) ∧
  ((2 * cos θ + 3 * sin θ) / (cos θ - 3 * sin θ) + sin θ * cos θ = -6/5) := by
  sorry

end NUMINAMATH_CALUDE_angle_trigonometry_l2397_239794


namespace NUMINAMATH_CALUDE_february_to_january_sales_ratio_l2397_239702

/-- The ratio of window screens sold in February to January is 2:3 -/
theorem february_to_january_sales_ratio :
  ∀ (january february march : ℕ),
  february = march / 4 →
  march = 8800 →
  january + february + march = 12100 →
  (february : ℚ) / january = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_february_to_january_sales_ratio_l2397_239702


namespace NUMINAMATH_CALUDE_friend_apple_rotations_l2397_239761

/-- Given the conditions of a juggling contest, prove the number of rotations made by each of Toby's friend's apples -/
theorem friend_apple_rotations 
  (toby_baseballs : ℕ)
  (toby_rotations_per_baseball : ℕ)
  (friend_apples : ℕ)
  (winner_total_rotations : ℕ)
  (h1 : toby_baseballs = 5)
  (h2 : toby_rotations_per_baseball = 80)
  (h3 : friend_apples = 4)
  (h4 : winner_total_rotations = 404)
  : (winner_total_rotations - toby_baseballs * toby_rotations_per_baseball) / friend_apples + toby_rotations_per_baseball = 81 := by
  sorry

end NUMINAMATH_CALUDE_friend_apple_rotations_l2397_239761


namespace NUMINAMATH_CALUDE_circle_tangency_l2397_239798

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii. -/
def internally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r2 - r1)^2

theorem circle_tangency (n : ℝ) : 
  internally_tangent (0, 0) (3, 4) 1 (Real.sqrt (25 - n)) → n = -11 := by
  sorry

#check circle_tangency

end NUMINAMATH_CALUDE_circle_tangency_l2397_239798


namespace NUMINAMATH_CALUDE_constant_theta_and_z_forms_line_l2397_239712

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying θ = c and z = d -/
def ConstantThetaAndZ (c d : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = c ∧ p.z = d}

/-- Definition of a line in cylindrical coordinates -/
def IsLine (S : Set CylindricalPoint) : Prop :=
  ∃ (a b : ℝ), ∀ p ∈ S, p.r = a * p.θ + b

theorem constant_theta_and_z_forms_line (c d : ℝ) :
  IsLine (ConstantThetaAndZ c d) := by
  sorry


end NUMINAMATH_CALUDE_constant_theta_and_z_forms_line_l2397_239712


namespace NUMINAMATH_CALUDE_right_triangle_altitude_condition_l2397_239781

theorem right_triangle_altitude_condition 
  (a b m : ℝ) 
  (h_positive : a > 0 ∧ b > 0) 
  (h_right_triangle : a^2 + b^2 = (a + b)^2 / 4) 
  (h_altitude : m = (1/5) * Real.sqrt (9*b^2 - 16*a^2)) : 
  (m = (1/5) * Real.sqrt (9*b^2 - 16*a^2)) ↔ b = 2*a := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_condition_l2397_239781


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l2397_239741

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (total : ℕ) (green : ℕ) (red : ℕ) (blue : ℕ) : ℕ :=
  2 * (Nat.factorial blue) * (Nat.factorial red) * (Nat.factorial green)

/-- Theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 8 2 3 3 = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l2397_239741


namespace NUMINAMATH_CALUDE_smallest_perfect_square_factor_l2397_239779

def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

theorem smallest_perfect_square_factor (k : ℕ) : 
  (k > 0 ∧ ∃ m : ℕ, k * y = m^2 ∧ ∀ n : ℕ, 0 < n ∧ n < k → ¬∃ m : ℕ, n * y = m^2) ↔ k = 110 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_factor_l2397_239779


namespace NUMINAMATH_CALUDE_range_of_g_l2397_239772

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2397_239772


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2397_239757

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2397_239757


namespace NUMINAMATH_CALUDE_division_equality_l2397_239766

theorem division_equality : (786^2 * 74) / 23592 = 1938.8 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l2397_239766


namespace NUMINAMATH_CALUDE_division_problem_l2397_239707

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2397_239707


namespace NUMINAMATH_CALUDE_factorization_equality_l2397_239771

theorem factorization_equality (m a : ℝ) : m * a^2 - m = m * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2397_239771


namespace NUMINAMATH_CALUDE_fixed_point_coordinates_l2397_239723

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation (3+k)x-2y+1-k=0 passes through the point A for any real k -/
def passes_through (A : Point) : Prop :=
  ∀ k : ℝ, (3 + k) * A.x - 2 * A.y + 1 - k = 0

/-- The fixed point A that the line passes through for all k has coordinates (1, 2) -/
theorem fixed_point_coordinates : 
  ∃ A : Point, passes_through A ∧ A.x = 1 ∧ A.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_coordinates_l2397_239723


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_l2397_239708

theorem sqrt_fraction_sum : Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_l2397_239708


namespace NUMINAMATH_CALUDE_g_definition_l2397_239769

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 2*x - 11

-- Theorem statement
theorem g_definition : ∀ x : ℝ, g (x + 2) = 2*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_g_definition_l2397_239769
