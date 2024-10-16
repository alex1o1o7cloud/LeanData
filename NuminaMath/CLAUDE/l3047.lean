import Mathlib

namespace NUMINAMATH_CALUDE_special_checkerboard_black_squares_l3047_304770

/-- A checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  isRed : Fin size → Fin size → Bool

/-- The properties of our specific checkerboard -/
def specialCheckerboard : Checkerboard where
  size := 32
  isRed := fun i j => 
    (i.val + j.val) % 2 = 0 ∨ i.val = 0 ∨ i.val = 31 ∨ j.val = 0 ∨ j.val = 31

/-- Count of black squares on the checkerboard -/
def blackSquareCount (c : Checkerboard) : ℕ :=
  (c.size * c.size) - (Finset.sum (Finset.univ : Finset (Fin c.size × Fin c.size)) 
    fun (i, j) => if c.isRed i j then 1 else 0)

/-- Theorem stating the number of black squares on our special checkerboard -/
theorem special_checkerboard_black_squares :
  blackSquareCount specialCheckerboard = 511 := by sorry

end NUMINAMATH_CALUDE_special_checkerboard_black_squares_l3047_304770


namespace NUMINAMATH_CALUDE_scalar_for_coplanarity_l3047_304721

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points as vectors
variable (O A B C D : V)

-- Define the scalar k
variable (k : ℝ)

-- Define the equation
def equation (O A B C D : V) (k : ℝ) : Prop :=
  2 • (A - O) - 3 • (B - O) + 7 • (C - O) + k • (D - O) = 0

-- Define coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- Theorem statement
theorem scalar_for_coplanarity (O A B C D : V) :
  ∃ (k : ℝ), equation O A B C D k ∧ coplanar A B C D ∧ k = -6 := by sorry

end NUMINAMATH_CALUDE_scalar_for_coplanarity_l3047_304721


namespace NUMINAMATH_CALUDE_min_value_z_l3047_304717

theorem min_value_z (a b : ℝ) : 
  ∃ (m : ℝ), ∀ (x y : ℝ), x^2 + y^2 ≤ 25 ∧ 2*x + y ≤ 5 → 
  x^2 + y^2 - 2*a*x - 2*b*y ≥ m ∧ m ≥ -a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3047_304717


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_462_l3047_304720

theorem sum_of_distinct_prime_factors_462 : 
  (Finset.sum (Nat.factors 462).toFinset id) = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_462_l3047_304720


namespace NUMINAMATH_CALUDE_complex_modulus_of_fraction_l3047_304714

theorem complex_modulus_of_fraction (z : ℂ) : 
  z = (4 - 2*I) / (1 - I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_fraction_l3047_304714


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3047_304728

/-- A random variable following a normal distribution with mean 1 and standard deviation σ > 0 -/
def normal_rv (σ : ℝ) : Type := ℝ

/-- The probability density function of the normal distribution -/
noncomputable def pdf (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that the random variable takes a value in the interval (a, b) -/
noncomputable def prob (σ : ℝ) (a b : ℝ) : ℝ := sorry

/-- Theorem: If P(0 < ξ < 1) = 0.4, then P(0 < ξ < 2) = 0.8 for a normal distribution with mean 1 -/
theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) :
  prob σ 0 1 = 0.4 → prob σ 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3047_304728


namespace NUMINAMATH_CALUDE_circle_max_sum_of_abs_l3047_304709

theorem circle_max_sum_of_abs (x y : ℝ) :
  x^2 + y^2 = 4 → |x| + |y| ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_sum_of_abs_l3047_304709


namespace NUMINAMATH_CALUDE_number_calculation_l3047_304778

theorem number_calculation : 
  let x : Float := 0.17999999999999997
  let number : Float := x * 0.05
  number / x = 0.05 ∧ number = 0.009 := by
sorry

end NUMINAMATH_CALUDE_number_calculation_l3047_304778


namespace NUMINAMATH_CALUDE_twelve_triangles_fit_l3047_304741

/-- Represents a right triangle with integer leg lengths -/
structure RightTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the area of a right triangle -/
def area (t : RightTriangle) : ℕ := t.base * t.height / 2

/-- Counts the number of small triangles that fit into a large triangle -/
def count_triangles (large : RightTriangle) (small : RightTriangle) : ℕ :=
  area large / area small

/-- Theorem stating that 12 small triangles fit into the large triangle -/
theorem twelve_triangles_fit (large small : RightTriangle) 
  (h1 : large.base = 6) (h2 : large.height = 4)
  (h3 : small.base = 2) (h4 : small.height = 1) :
  count_triangles large small = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_triangles_fit_l3047_304741


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3047_304716

theorem complex_modulus_problem (z : ℂ) (h : (z - Complex.I) * Complex.I = 2 + 3 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3047_304716


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3047_304791

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3047_304791


namespace NUMINAMATH_CALUDE_percentage_relation_l3047_304700

theorem percentage_relation (w x y z : ℝ) 
  (hx : x = 1.25 * y)
  (hy : y = 0.4 * z)
  (hz : z = 1.1 * w) :
  x = 0.55 * w := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3047_304700


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3047_304711

theorem fruit_basket_count : 
  let apples_per_basket : ℕ := 9
  let oranges_per_basket : ℕ := 15
  let bananas_per_basket : ℕ := 14
  let num_baskets : ℕ := 4
  let fruits_per_basket : ℕ := apples_per_basket + oranges_per_basket + bananas_per_basket
  let fruits_in_three_baskets : ℕ := 3 * fruits_per_basket
  let fruits_in_fourth_basket : ℕ := fruits_per_basket - 6
  fruits_in_three_baskets + fruits_in_fourth_basket = 70
  := by sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3047_304711


namespace NUMINAMATH_CALUDE_range_of_a_l3047_304760

-- Define the function f(x) = |x+2| - |x-3|
def f (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x, f x ≤ a) → a ∈ Set.Ici (-5) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3047_304760


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l3047_304768

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

theorem tank_insulation_cost :
  let l : ℝ := 6
  let w : ℝ := 3
  let h : ℝ := 2
  let costPerSqFt : ℝ := 20
  insulationCost l w h costPerSqFt = 1440 := by
  sorry

#eval insulationCost 6 3 2 20

end NUMINAMATH_CALUDE_tank_insulation_cost_l3047_304768


namespace NUMINAMATH_CALUDE_max_class_size_is_17_l3047_304776

/-- Represents a school with students and buses for an excursion. -/
structure School where
  total_students : ℕ
  num_buses : ℕ
  seats_per_bus : ℕ

/-- Checks if it's possible to seat all students with the given maximum class size. -/
def can_seat_all (s : School) (max_class_size : ℕ) : Prop :=
  ∀ (class_sizes : List ℕ),
    (class_sizes.sum = s.total_students) →
    (∀ size ∈ class_sizes, size ≤ max_class_size) →
    ∃ (allocation : List (List ℕ)),
      (allocation.length ≤ s.num_buses) ∧
      (∀ bus ∈ allocation, bus.sum ≤ s.seats_per_bus) ∧
      (allocation.join.sum = s.total_students)

/-- The main theorem stating that 17 is the maximum class size for the given school configuration. -/
theorem max_class_size_is_17 (s : School)
    (h1 : s.total_students = 920)
    (h2 : s.num_buses = 16)
    (h3 : s.seats_per_bus = 71) :
    (can_seat_all s 17 ∧ ¬can_seat_all s 18) :=
  sorry


end NUMINAMATH_CALUDE_max_class_size_is_17_l3047_304776


namespace NUMINAMATH_CALUDE_simplify_fraction_l3047_304757

theorem simplify_fraction : (120 : ℚ) / 1800 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3047_304757


namespace NUMINAMATH_CALUDE_lily_pad_half_coverage_l3047_304706

theorem lily_pad_half_coverage (total_days : ℕ) (half_coverage_days : ℕ) : 
  (total_days = 34) → 
  (half_coverage_days = total_days - 1) →
  (half_coverage_days = 33) :=
by sorry

end NUMINAMATH_CALUDE_lily_pad_half_coverage_l3047_304706


namespace NUMINAMATH_CALUDE_inverse_square_problem_l3047_304786

-- Define the relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y * y) ∧ k ≠ 0

-- State the theorem
theorem inverse_square_problem (x₁ x₂ y₁ y₂ : ℝ) :
  inverse_square_relation x₁ y₁ →
  inverse_square_relation x₂ y₂ →
  x₁ = 1 →
  y₁ = 3 →
  y₂ = 6 →
  x₂ = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_problem_l3047_304786


namespace NUMINAMATH_CALUDE_sector_area_l3047_304739

theorem sector_area (α : Real) (r : Real) (h1 : α = 2 * Real.pi / 3) (h2 : r = Real.sqrt 3) : 
  (1/2) * α * r^2 = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3047_304739


namespace NUMINAMATH_CALUDE_A_cubed_is_zero_l3047_304726

open Matrix

theorem A_cubed_is_zero {α : Type*} [Field α] (A : Matrix (Fin 2) (Fin 2) α) 
  (h1 : A ^ 4 = 0)
  (h2 : Matrix.trace A = 0) :
  A ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_A_cubed_is_zero_l3047_304726


namespace NUMINAMATH_CALUDE_solution_to_system_l3047_304784

theorem solution_to_system :
  ∃ (x y : ℝ), x + y = (6 - x) + (6 - y) ∧ x - y = (x - 2) + (y - 2) ∧ x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l3047_304784


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3047_304761

theorem age_ratio_problem (sam drew : ℕ) : 
  sam + drew = 54 → sam = 18 → sam * 2 = drew :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3047_304761


namespace NUMINAMATH_CALUDE_danny_steve_time_ratio_l3047_304754

/-- The time it takes Danny to reach Steve's house, in minutes -/
def danny_time : ℝ := 35

/-- The time it takes Steve to reach Danny's house, in minutes -/
def steve_time : ℝ := 70

/-- The extra time it takes Steve to reach the halfway point compared to Danny, in minutes -/
def extra_time : ℝ := 17.5

theorem danny_steve_time_ratio :
  danny_time / steve_time = 1 / 2 ∧
  steve_time / 2 = danny_time / 2 + extra_time :=
sorry

end NUMINAMATH_CALUDE_danny_steve_time_ratio_l3047_304754


namespace NUMINAMATH_CALUDE_polynomial_equality_l3047_304792

def P (x : ℝ) := -x^2 + 5*x - 3

theorem polynomial_equality : 
  (∀ x, P x + (x^2 - 2*x + 1) = x^2 - 2*x + 1) → 
  (∀ x, P x = 2*x^2 - 7*x + 4) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3047_304792


namespace NUMINAMATH_CALUDE_jonathan_daily_burn_l3047_304781

-- Define Jonathan's daily calorie intake
def daily_intake : ℕ := 2500

-- Define Jonathan's extra calorie intake on Saturday
def saturday_extra : ℕ := 1000

-- Define Jonathan's weekly caloric deficit
def weekly_deficit : ℕ := 2500

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove Jonathan's daily calorie burn
theorem jonathan_daily_burn :
  (6 * daily_intake + (daily_intake + saturday_extra) + weekly_deficit) / days_in_week = 3000 :=
by sorry

end NUMINAMATH_CALUDE_jonathan_daily_burn_l3047_304781


namespace NUMINAMATH_CALUDE_virginia_sweettarts_l3047_304755

/-- The number of Virginia's friends -/
def num_friends : ℕ := 4

/-- The number of Sweettarts each person ate -/
def sweettarts_per_person : ℕ := 3

/-- The initial number of Sweettarts Virginia had -/
def initial_sweettarts : ℕ := num_friends * sweettarts_per_person + sweettarts_per_person

theorem virginia_sweettarts : initial_sweettarts = 15 := by
  sorry

end NUMINAMATH_CALUDE_virginia_sweettarts_l3047_304755


namespace NUMINAMATH_CALUDE_telephone_number_D_is_9_l3047_304727

def TelephoneNumber (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧ D > E ∧ E > F ∧ G > H ∧ H > I ∧ I > J ∧
  A % 2 = 0 ∧ B = A - 2 ∧ C = B - 2 ∧
  D % 2 = 1 ∧ E = D - 2 ∧ F = E - 2 ∧
  H + I + J = 9 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem telephone_number_D_is_9 :
  ∀ A B C D E F G H I J, TelephoneNumber A B C D E F G H I J → D = 9 :=
by sorry

end NUMINAMATH_CALUDE_telephone_number_D_is_9_l3047_304727


namespace NUMINAMATH_CALUDE_platform_length_l3047_304795

/-- Given a train of length 300 meters, which takes 39 seconds to cross a platform
    and 9 seconds to cross a signal pole, the length of the platform is 1000 meters. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 9) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 1000 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3047_304795


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3047_304733

-- Define the conditions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is sufficient but not necessary for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3047_304733


namespace NUMINAMATH_CALUDE_positive_real_inequality_l3047_304766

theorem positive_real_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l3047_304766


namespace NUMINAMATH_CALUDE_program_output_25_l3047_304715

theorem program_output_25 (x : ℝ) : 
  ((x < 0 ∧ (x + 1)^2 = 25) ∨ (x ≥ 0 ∧ (x - 1)^2 = 25)) ↔ (x = 6 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_program_output_25_l3047_304715


namespace NUMINAMATH_CALUDE_gum_distribution_l3047_304707

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) :
  cousins = 4 →
  total_gum = 20 →
  total_gum = cousins * gum_per_cousin →
  gum_per_cousin = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l3047_304707


namespace NUMINAMATH_CALUDE_inverse_prop_is_false_l3047_304789

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval [a,b]
variable (a b : ℝ)

-- State that f is continuous on [a,b]
variable (hf : ContinuousOn f (Set.Icc a b))

-- Define the original proposition
def original_prop : Prop :=
  ∀ x ∈ Set.Ioo a b, f a * f b < 0 → ∃ c ∈ Set.Ioo a b, f c = 0

-- Define the inverse proposition
def inverse_prop : Prop :=
  ∀ x ∈ Set.Ioo a b, (∃ c ∈ Set.Ioo a b, f c = 0) → f a * f b < 0

-- State the theorem
theorem inverse_prop_is_false
  (h : original_prop f a b) : ¬(inverse_prop f a b) := by
  sorry


end NUMINAMATH_CALUDE_inverse_prop_is_false_l3047_304789


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l3047_304758

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs (x + a)

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l3047_304758


namespace NUMINAMATH_CALUDE_smallest_s_for_arithmetic_progression_l3047_304794

open Real

theorem smallest_s_for_arithmetic_progression (β : ℝ) (s : ℝ) :
  0 < β ∧ β < π / 2 →
  (∃ d : ℝ, arcsin (sin (3 * β)) + d = arcsin (sin (5 * β)) ∧
            arcsin (sin (5 * β)) + d = arcsin (sin (10 * β)) ∧
            arcsin (sin (10 * β)) + d = arcsin (sin (s * β))) →
  s ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_s_for_arithmetic_progression_l3047_304794


namespace NUMINAMATH_CALUDE_import_tax_problem_l3047_304725

/-- Calculates the import tax percentage given the total value, tax-free portion, and tax amount -/
def import_tax_percentage (total_value tax_free_portion tax_amount : ℚ) : ℚ :=
  (tax_amount / (total_value - tax_free_portion)) * 100

/-- Proves that the import tax percentage is 7% given the problem conditions -/
theorem import_tax_problem :
  let total_value : ℚ := 2610
  let tax_free_portion : ℚ := 1000
  let tax_amount : ℚ := 1127/10
by
  sorry


end NUMINAMATH_CALUDE_import_tax_problem_l3047_304725


namespace NUMINAMATH_CALUDE_roots_of_equation_l3047_304765

def f (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x) * (x^2 - 1)

theorem roots_of_equation :
  {x : ℝ | f x = 0} = {0, 3, -5, 1, -1} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3047_304765


namespace NUMINAMATH_CALUDE_required_run_rate_is_6_15_l3047_304742

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  targetScore : ℕ
  firstSegmentOvers : ℕ
  firstSegmentRunRate : ℚ
  firstSegmentWicketsLost : ℕ
  maxTotalWicketsLost : ℕ
  personalMilestone : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstSegmentOvers
  let runsScored := game.firstSegmentRunRate * game.firstSegmentOvers
  let runsNeeded := game.targetScore - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given game scenario -/
theorem required_run_rate_is_6_15 (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.targetScore = 282)
    (h3 : game.firstSegmentOvers = 10)
    (h4 : game.firstSegmentRunRate = 3.6)
    (h5 : game.firstSegmentWicketsLost = 2)
    (h6 : game.maxTotalWicketsLost = 5)
    (h7 : game.personalMilestone = 75) :
    requiredRunRate game = 6.15 := by
  sorry


end NUMINAMATH_CALUDE_required_run_rate_is_6_15_l3047_304742


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coordinate_l3047_304737

/-- Given an equilateral triangle with two vertices at (1, 10) and (9, 10),
    and the third vertex in the first quadrant, 
    prove that the y-coordinate of the third vertex is 10 + 4√3 -/
theorem equilateral_triangle_third_vertex_y_coordinate 
  (A B C : ℝ × ℝ) : 
  A = (1, 10) → 
  B = (9, 10) → 
  C.1 ≥ 0 → 
  C.2 ≥ 0 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 → 
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 → 
  C.2 = 10 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coordinate_l3047_304737


namespace NUMINAMATH_CALUDE_reading_time_difference_l3047_304787

/-- Proves that given Xanthia's and Molly's reading speeds and a book's page count,
    the difference in reading time is 240 minutes. -/
theorem reading_time_difference
  (xanthia_speed : ℕ)
  (molly_speed : ℕ)
  (book_pages : ℕ)
  (h1 : xanthia_speed = 80)
  (h2 : molly_speed = 40)
  (h3 : book_pages = 320) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 240 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_l3047_304787


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l3047_304736

def g (x : ℝ) : ℝ := 12 * x^5 - 24 * x^3 + 9 * x

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/2) ∧
  g r = 0 ∧
  ∀ x, g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l3047_304736


namespace NUMINAMATH_CALUDE_min_cost_pool_l3047_304771

/-- Represents the dimensions and cost parameters of a rectangular pool -/
structure PoolParams where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of the pool given its length and width -/
def totalCost (p : PoolParams) (length width : ℝ) : ℝ :=
  p.bottomCost * length * width + p.wallCost * (2 * length * p.depth + 2 * width * p.depth)

/-- Theorem stating the minimum cost and dimensions of the pool -/
theorem min_cost_pool (p : PoolParams) 
    (hv : p.volume = 16)
    (hd : p.depth = 4)
    (hb : p.bottomCost = 110)
    (hw : p.wallCost = 90) :
    ∃ (length width : ℝ),
      length * width * p.depth = p.volume ∧
      length = 2 ∧
      width = 2 ∧
      totalCost p length width = 1880 ∧
      ∀ (l w : ℝ), l * w * p.depth = p.volume → totalCost p l w ≥ totalCost p length width :=
  sorry

end NUMINAMATH_CALUDE_min_cost_pool_l3047_304771


namespace NUMINAMATH_CALUDE_symmetric_origin_correct_symmetric_point_correct_l3047_304732

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the origin
def symmetricOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

-- Define symmetry with respect to another point
def symmetricPoint (p : Point2D) (k : Point2D) : Point2D :=
  { x := 2 * k.x - p.x, y := 2 * k.y - p.y }

-- Theorem for symmetry with respect to the origin
theorem symmetric_origin_correct (m : Point2D) :
  symmetricOrigin m = { x := -m.x, y := -m.y } := by sorry

-- Theorem for symmetry with respect to another point
theorem symmetric_point_correct (m k : Point2D) :
  symmetricPoint m k = { x := 2 * k.x - m.x, y := 2 * k.y - m.y } := by sorry

end NUMINAMATH_CALUDE_symmetric_origin_correct_symmetric_point_correct_l3047_304732


namespace NUMINAMATH_CALUDE_P_intersect_Q_l3047_304713

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem P_intersect_Q : P ∩ Q = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_Q_l3047_304713


namespace NUMINAMATH_CALUDE_tangent_line_parallel_to_3x_minus_y_equals_0_l3047_304773

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_line_parallel_to_3x_minus_y_equals_0 :
  let P : ℝ × ℝ := (1, 0)
  f P.1 = P.2 ∧ f' P.1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_to_3x_minus_y_equals_0_l3047_304773


namespace NUMINAMATH_CALUDE_custom_operation_result_l3047_304701

/-- Custom operation delta -/
def delta (x y : ℝ) : ℝ := x^2 - y

/-- Main theorem -/
theorem custom_operation_result :
  delta (5^(delta 3 2)) (4^(delta 2 7)) = 6103515624.984375 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l3047_304701


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l3047_304705

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle at C
def angle_C (t : Triangle) : ℝ := sorry

-- Define the length of side AB
def side_AB (t : Triangle) : ℝ := sorry

-- Define the sum of sides AC and BC
def sum_AC_BC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_side_sum_range (t : Triangle) :
  angle_C t = π / 3 →  -- 60° in radians
  side_AB t = 2 →
  2 < sum_AC_BC t ∧ sum_AC_BC t ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l3047_304705


namespace NUMINAMATH_CALUDE_nested_cube_root_simplification_l3047_304712

theorem nested_cube_root_simplification (N : ℝ) (h : N > 1) :
  (N^3 * (N^5 * N^3)^(1/3))^(1/3) = N^(5/3) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_simplification_l3047_304712


namespace NUMINAMATH_CALUDE_largest_product_of_three_l3047_304746

def S : Finset Int := {-5, -4, -1, 3, 7, 9}

theorem largest_product_of_three (a b c : Int) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S →
  x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ x * y * z →
  x * y * z ≤ 189 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l3047_304746


namespace NUMINAMATH_CALUDE_smallest_n_for_m_independent_same_color_lines_l3047_304774

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for m lines of the same color with no common endpoints -/
def HasMIndependentSameColorLines (c : TwoColoring n) (m : ℕ) : Prop :=
  ∃ (edges : Fin m → Fin n × Fin n),
    (∀ i j, i ≠ j → (edges i).1 ≠ (edges j).1 ∧ (edges i).1 ≠ (edges j).2 ∧
                    (edges i).2 ≠ (edges j).1 ∧ (edges i).2 ≠ (edges j).2) ∧
    (∀ i j, c (edges i).1 (edges i).2 = c (edges j).1 (edges j).2)

/-- The main theorem -/
theorem smallest_n_for_m_independent_same_color_lines (m : ℕ) :
  (∀ n, n ≥ 3 * m - 1 → ∀ c : TwoColoring n, HasMIndependentSameColorLines c m) ∧
  (∀ n, n < 3 * m - 1 → ∃ c : TwoColoring n, ¬HasMIndependentSameColorLines c m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_m_independent_same_color_lines_l3047_304774


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3047_304710

/-- The complex number z = i(1+i) -/
def z : ℂ := Complex.I * (1 + Complex.I)

/-- The real part of z -/
def real_part : ℝ := z.re

/-- The imaginary part of z -/
def imag_part : ℝ := z.im

/-- Theorem: z is in the second quadrant -/
theorem z_in_second_quadrant : real_part < 0 ∧ imag_part > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3047_304710


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_proof_l3047_304762

/-- The least positive base-10 number that requires seven digits in binary representation -/
def least_seven_digit_binary : ℕ := 64

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binary_digits (n : ℕ) : ℕ := sorry

theorem least_seven_digit_binary_proof :
  (binary_digits least_seven_digit_binary = 7) ∧
  (∀ m : ℕ, m > 0 ∧ m < least_seven_digit_binary → binary_digits m < 7) :=
sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_proof_l3047_304762


namespace NUMINAMATH_CALUDE_jelly_bean_probabilities_l3047_304753

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue + bag.purple

/-- The specific bag of jelly beans described in the problem -/
def ourBag : JellyBeanBag :=
  { red := 10, green := 12, yellow := 15, blue := 18, purple := 5 }

theorem jelly_bean_probabilities :
  let total := totalJellyBeans ourBag
  (ourBag.purple : ℚ) / total = 1 / 12 ∧
  ((ourBag.blue + ourBag.purple : ℚ) / total = 23 / 60) := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probabilities_l3047_304753


namespace NUMINAMATH_CALUDE_regression_independence_correct_statement_l3047_304722

/-- Definition of regression analysis -/
def regression_analysis : Type := Unit

/-- Definition of independence test -/
def independence_test : Type := Unit

/-- Property: Regression analysis studies correlation between two variables -/
axiom regression_studies_correlation : regression_analysis → Prop

/-- Property: Independence test analyzes relationship between two variables -/
axiom independence_analyzes_relationship : independence_test → Prop

/-- Property: Independence test cannot determine relationships with 100% certainty -/
axiom independence_not_certain : independence_test → Prop

/-- The correct statement about regression analysis and independence test -/
def correct_statement : Prop :=
  ∃ (ra : regression_analysis) (it : independence_test),
    regression_studies_correlation ra ∧
    independence_analyzes_relationship it

theorem regression_independence_correct_statement :
  correct_statement :=
sorry

end NUMINAMATH_CALUDE_regression_independence_correct_statement_l3047_304722


namespace NUMINAMATH_CALUDE_cube_root_sum_l3047_304788

theorem cube_root_sum (u v w : ℝ) : 
  (∃ x y z : ℝ, x^3 = 8 ∧ y^3 = 27 ∧ z^3 = 64 ∧
   (u - x) * (u - y) * (u - z) = 1/2 ∧
   (v - x) * (v - y) * (v - z) = 1/2 ∧
   (w - x) * (w - y) * (w - z) = 1/2 ∧
   u ≠ v ∧ u ≠ w ∧ v ≠ w) →
  u^3 + v^3 + w^3 = -42 := by
sorry

end NUMINAMATH_CALUDE_cube_root_sum_l3047_304788


namespace NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l3047_304718

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def number_with_d (d : ℕ) : ℕ := 563000 + d * 100 + 4

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by_9 (number_with_d d) ∧
    ∀ (d' : ℕ), d' < d → ¬(is_divisible_by_9 (number_with_d d')) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_for_divisibility_l3047_304718


namespace NUMINAMATH_CALUDE_b_100_mod_50_l3047_304744

/-- Define the sequence b_n = 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- Theorem: b_100 ≡ 2 (mod 50) -/
theorem b_100_mod_50 : b 100 ≡ 2 [MOD 50] := by
  sorry

end NUMINAMATH_CALUDE_b_100_mod_50_l3047_304744


namespace NUMINAMATH_CALUDE_juan_number_problem_l3047_304780

theorem juan_number_problem (n : ℚ) : 
  ((n + 3) * 3 - 5) / 3 = 10 → n = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_juan_number_problem_l3047_304780


namespace NUMINAMATH_CALUDE_fraction_simplification_l3047_304740

theorem fraction_simplification :
  (3^1006 + 3^1004) / (3^1006 - 3^1004) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3047_304740


namespace NUMINAMATH_CALUDE_harry_seed_purchase_cost_l3047_304730

/-- Given the prices of seed packets and the quantities Harry wants to buy, 
    prove that the total cost is $18.00 -/
theorem harry_seed_purchase_cost : 
  let pumpkin_price : ℚ := 25/10
  let tomato_price : ℚ := 15/10
  let chili_price : ℚ := 9/10
  let pumpkin_quantity : ℕ := 3
  let tomato_quantity : ℕ := 4
  let chili_quantity : ℕ := 5
  (pumpkin_price * pumpkin_quantity + 
   tomato_price * tomato_quantity + 
   chili_price * chili_quantity) = 18
:= by sorry

end NUMINAMATH_CALUDE_harry_seed_purchase_cost_l3047_304730


namespace NUMINAMATH_CALUDE_average_age_of_extreme_new_employees_is_30_l3047_304731

/-- Represents a company with employees and their ages -/
structure Company where
  initialEmployees : ℕ
  group1Size : ℕ
  group1AvgAge : ℕ
  group2Size : ℕ
  group2AvgAge : ℕ
  group3Size : ℕ
  group3AvgAge : ℕ
  newEmployees : ℕ
  newEmployeesTotalAge : ℕ
  ageDifference : ℕ

/-- Calculates the average age of the youngest and oldest new employees -/
def averageAgeOfExtremeNewEmployees (c : Company) : ℚ :=
  let totalAge := c.group1Size * c.group1AvgAge + c.group2Size * c.group2AvgAge + c.group3Size * c.group3AvgAge
  let totalEmployees := c.initialEmployees + c.newEmployees
  let x := (c.newEmployeesTotalAge - (c.newEmployees - 1) * (c.ageDifference / 2)) / c.newEmployees
  (x + x + c.ageDifference) / 2

/-- Theorem stating that for the given company configuration, 
    the average age of the youngest and oldest new employees is 30 -/
theorem average_age_of_extreme_new_employees_is_30 :
  let c : Company := {
    initialEmployees := 50,
    group1Size := 20,
    group1AvgAge := 30,
    group2Size := 20,
    group2AvgAge := 40,
    group3Size := 10,
    group3AvgAge := 50,
    newEmployees := 5,
    newEmployeesTotalAge := 150,
    ageDifference := 20
  }
  averageAgeOfExtremeNewEmployees c = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_extreme_new_employees_is_30_l3047_304731


namespace NUMINAMATH_CALUDE_value_of_a_l3047_304708

theorem value_of_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 6) 
  (eq3 : c = 4) : 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3047_304708


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l3047_304724

theorem solution_implies_a_value (a x y : ℝ) : 
  x = 2 → y = 1 → a * x - 3 * y = 1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l3047_304724


namespace NUMINAMATH_CALUDE_line_plane_relationship_l3047_304783

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (m : Line) (α β : Plane) 
  (h1 : perpPlanes α β) 
  (h2 : perp m α) : 
  para m β ∨ subset m β := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l3047_304783


namespace NUMINAMATH_CALUDE_translation_result_l3047_304764

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a translation in 2D space
structure Translation2D where
  dx : ℝ
  dy : ℝ

-- Define a function to apply a translation to a point
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_result :
  let A : Point2D := { x := -3, y := 2 }
  let right_translation : Translation2D := { dx := 4, dy := 0 }
  let down_translation : Translation2D := { dx := 0, dy := -3 }
  let A' := applyTranslation (applyTranslation A right_translation) down_translation
  A'.x = 1 ∧ A'.y = -1 := by
sorry

end NUMINAMATH_CALUDE_translation_result_l3047_304764


namespace NUMINAMATH_CALUDE_angle_sum_proof_l3047_304769

theorem angle_sum_proof (a b : Real) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : 4 * (Real.cos a)^3 - 3 * (Real.cos b)^3 = 2)
  (h2 : 4 * Real.cos (2*a) + 3 * Real.cos (2*b) = 1) :
  2*a + b = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l3047_304769


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l3047_304703

theorem wicket_keeper_age_difference (team_size : ℕ) (team_average_age : ℝ) 
  (remaining_average_age : ℝ) (h1 : team_size = 11) (h2 : team_average_age = 23) 
  (h3 : remaining_average_age = team_average_age - 1) :
  ∃ (wicket_keeper_age : ℝ),
    wicket_keeper_age = team_average_age + 9 ∧
    (team_size - 2) * remaining_average_age + wicket_keeper_age + team_average_age = 
      team_size * team_average_age :=
by
  sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l3047_304703


namespace NUMINAMATH_CALUDE_problem_solution_l3047_304798

theorem problem_solution (x y : ℝ) : 
  (0.5 * x = 0.05 * 500 - 20) ∧ 
  (0.3 * y = 0.25 * x + 10) → 
  (x = 10 ∧ y = 125/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3047_304798


namespace NUMINAMATH_CALUDE_monotone_function_decomposition_l3047_304704

-- Define the real-valued functions
variable (f g h : ℝ → ℝ)

-- Define the constants
variable (c a b : ℝ)

-- State the theorem
theorem monotone_function_decomposition :
  (Monotone f) →
  (∀ x y : ℝ, f (x + y) = g x + h y) →
  ∃ c a b : ℝ, ∀ x : ℝ, f x = c * x + a + b :=
by sorry

end NUMINAMATH_CALUDE_monotone_function_decomposition_l3047_304704


namespace NUMINAMATH_CALUDE_quartic_integer_roots_l3047_304772

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure QuarticPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  d_nonzero : d ≠ 0

/-- The number of integer roots of a quartic polynomial, counting multiplicities -/
def num_integer_roots (p : QuarticPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem quartic_integer_roots (p : QuarticPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 4 :=
sorry

end NUMINAMATH_CALUDE_quartic_integer_roots_l3047_304772


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3047_304743

theorem division_remainder_proof (D R r : ℕ) : 
  D = 12 * 42 + R →
  D = 21 * 24 + r →
  0 ≤ r →
  r < 21 →
  r = 0 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3047_304743


namespace NUMINAMATH_CALUDE_max_product_sum_300_l3047_304702

theorem max_product_sum_300 :
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l3047_304702


namespace NUMINAMATH_CALUDE_remainder_calculation_l3047_304767

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- Theorem statement
theorem remainder_calculation :
  rem (5/9 : ℚ) (-3/4 : ℚ) = -7/36 := by
  sorry

end NUMINAMATH_CALUDE_remainder_calculation_l3047_304767


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l3047_304729

theorem smallest_integer_satisfying_conditions : ∃ x : ℤ, 
  (∀ y : ℤ, (|3*y - 4| ≤ 25 ∧ 3 ∣ y) → x ≤ y) ∧ 
  |3*x - 4| ≤ 25 ∧ 
  3 ∣ x ∧
  x = -6 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l3047_304729


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l3047_304790

/-- Given an investment with the following properties:
  * Initial investment: $8000
  * Interest rate: y% per annum
  * Time period: 2 years
  * Simple interest earned: $800
Prove that the compound interest earned is $820 -/
theorem compound_interest_calculation (initial_investment : ℝ) (y : ℝ) (time : ℝ) 
  (simple_interest : ℝ) (h1 : initial_investment = 8000)
  (h2 : time = 2) (h3 : simple_interest = 800) 
  (h4 : simple_interest = initial_investment * y * time / 100) :
  initial_investment * ((1 + y / 100) ^ time - 1) = 820 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l3047_304790


namespace NUMINAMATH_CALUDE_rabbit_travel_time_l3047_304752

/-- Proves that a rabbit traveling at 5 miles per hour takes 24 minutes to cover 2 miles -/
theorem rabbit_travel_time :
  let speed : ℝ := 5  -- miles per hour
  let distance : ℝ := 2  -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 24 := by sorry

end NUMINAMATH_CALUDE_rabbit_travel_time_l3047_304752


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l3047_304779

theorem solution_satisfies_equations :
  let a : ℚ := 4/7
  let b : ℚ := 19/7
  let c : ℚ := 29/19
  let d : ℚ := -6/19
  (8*a^2 - 3*b^2 + 5*c^2 + 16*d^2 - 10*a*b + 42*c*d + 18*a + 22*b - 2*c - 54*d = 42) ∧
  (15*a^2 - 3*b^2 + 21*c^2 - 5*d^2 + 4*a*b + 32*c*d - 28*a + 14*b - 54*c - 52*d = -22) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l3047_304779


namespace NUMINAMATH_CALUDE_expression_evaluation_l3047_304756

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3047_304756


namespace NUMINAMATH_CALUDE_rhombus_area_l3047_304797

/-- The area of a rhombus with diagonals of length 3 and 4 is 6 -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : 
  (1 / 2 : ℝ) * d1 * d2 = 6 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l3047_304797


namespace NUMINAMATH_CALUDE_world_cup_2018_21st_edition_l3047_304782

/-- Represents the year of the nth World Cup -/
def worldCupYear (n : ℕ) : ℕ := 1950 + 4 * (n - 4)

/-- Theorem stating that the 2018 World Cup was the 21st edition -/
theorem world_cup_2018_21st_edition :
  ∃ n : ℕ, n = 21 ∧ worldCupYear n = 2018 :=
sorry

end NUMINAMATH_CALUDE_world_cup_2018_21st_edition_l3047_304782


namespace NUMINAMATH_CALUDE_odd_function_sum_condition_l3047_304751

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_condition (f : ℝ → ℝ) (h : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  ¬(∀ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 → x₁ + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_sum_condition_l3047_304751


namespace NUMINAMATH_CALUDE_big_dig_mining_theorem_l3047_304738

/-- Represents a mine with its daily production and ore percentages -/
structure Mine where
  dailyProduction : ℝ
  copperPercentage : ℝ
  ironPercentage : ℝ
  nickelPercentage : ℝ
  zincPercentage : ℝ

/-- Calculates the daily copper production for a given mine -/
def dailyCopperProduction (m : Mine) : ℝ :=
  m.dailyProduction * m.copperPercentage

/-- The Big Dig Mining Company problem -/
theorem big_dig_mining_theorem (mineA mineB mineC : Mine)
  (hA : mineA = { dailyProduction := 3000
                , copperPercentage := 0.05
                , ironPercentage := 0.60
                , nickelPercentage := 0.10
                , zincPercentage := 0.25 })
  (hB : mineB = { dailyProduction := 4000
                , copperPercentage := 0.10
                , ironPercentage := 0.50
                , nickelPercentage := 0.30
                , zincPercentage := 0.10 })
  (hC : mineC = { dailyProduction := 3500
                , copperPercentage := 0.15
                , ironPercentage := 0.45
                , nickelPercentage := 0.20
                , zincPercentage := 0.20 }) :
  dailyCopperProduction mineA + dailyCopperProduction mineB + dailyCopperProduction mineC = 1075 := by
  sorry

end NUMINAMATH_CALUDE_big_dig_mining_theorem_l3047_304738


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_l3047_304785

-- Proposition 1
def p : Prop := ∃ x : ℝ, Real.tan x = 2
def q : Prop := ∀ x : ℝ, x^2 - x + 1/2 > 0

theorem proposition_1 : ¬(p ∧ ¬q) := by sorry

-- Proposition 2
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := x + b * y + 1 = 0

theorem proposition_2 (a b : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ b x y → (a * 1 + 3 * b = 0)) ≠ 
  (∀ x y : ℝ, l₁ a x y → l₂ b x y → (a / b = -3)) := by sorry

-- Proposition 3
def original_statement (a b : ℝ) : Prop := 
  a * b ≥ 2 → a^2 + b^2 > 4

def negation_statement (a b : ℝ) : Prop := 
  a * b < 2 → a^2 + b^2 ≤ 4

theorem proposition_3 : 
  (∀ a b : ℝ, ¬(original_statement a b)) ↔ (∀ a b : ℝ, negation_statement a b) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_l3047_304785


namespace NUMINAMATH_CALUDE_tan_thirty_degrees_l3047_304763

theorem tan_thirty_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirty_degrees_l3047_304763


namespace NUMINAMATH_CALUDE_probability_x_plus_y_even_l3047_304775

def X := Finset.range 5
def Y := Finset.range 4

theorem probability_x_plus_y_even :
  let total_outcomes := X.card * Y.card
  let favorable_outcomes := (X.filter (λ x => x % 2 = 0)).card * (Y.filter (λ y => y % 2 = 0)).card +
                            (X.filter (λ x => x % 2 = 1)).card * (Y.filter (λ y => y % 2 = 1)).card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_even_l3047_304775


namespace NUMINAMATH_CALUDE_triangle_area_l3047_304793

/-- Given a triangle ABC with the following properties:
  - f(x) = sin(2x + B) + √3 cos(2x + B)
  - y = f(x - π/3) is an odd function
  - a = 1
  - b = f(0)
  Prove that the area S of triangle ABC is equal to √3/4. -/
theorem triangle_area (A B C : Real) (a b c : Real) (f : Real → Real) :
  (∀ x, f x = Real.sin (2 * x + B) + Real.sqrt 3 * Real.cos (2 * x + B)) →
  (∀ x, f (x - Real.pi / 3) = -f (-x + Real.pi / 3)) →
  a = 1 →
  b = f 0 →
  let S := (1 / 2) * a * b * Real.sin C
  S = Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3047_304793


namespace NUMINAMATH_CALUDE_school_boys_count_l3047_304735

/-- The number of girls in the school -/
def num_girls : ℕ := 34

/-- The difference between the number of boys and girls -/
def difference : ℕ := 807

/-- The number of boys in the school -/
def num_boys : ℕ := num_girls + difference

theorem school_boys_count : num_boys = 841 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3047_304735


namespace NUMINAMATH_CALUDE_three_possible_medians_l3047_304748

-- Define the set of game scores
def gameScores (x y : ℤ) : Finset ℤ := {x, 11, 13, y, 12}

-- Define the median of a set of integers
def median (s : Finset ℤ) : ℤ := sorry

-- Theorem statement
theorem three_possible_medians :
  ∃ (m₁ m₂ m₃ : ℤ), ∀ (x y : ℤ),
    (∃ (m : ℤ), median (gameScores x y) = m) →
    (m = m₁ ∨ m = m₂ ∨ m = m₃) ∧
    (m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃) :=
  sorry

#check three_possible_medians

end NUMINAMATH_CALUDE_three_possible_medians_l3047_304748


namespace NUMINAMATH_CALUDE_problem_solution_l3047_304750

theorem problem_solution (x : ℝ) (h : x = 6) : (x^6 - 17*x^3 + 72) / (x^3 - 8) = 207 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3047_304750


namespace NUMINAMATH_CALUDE_remaining_average_l3047_304734

theorem remaining_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 5 ∧ subset = 3 ∧ total_avg = 11 ∧ subset_avg = 4 →
  ((total_avg * total) - (subset_avg * subset)) / (total - subset) = 21.5 :=
by sorry

end NUMINAMATH_CALUDE_remaining_average_l3047_304734


namespace NUMINAMATH_CALUDE_intersection_points_count_l3047_304745

/-- The number of distinct intersection points for the given equations -/
def num_intersection_points : ℕ :=
  let eq1 := fun (x y : ℝ) => (x - y + 2) * (2 * x + 3 * y - 6) = 0
  let eq2 := fun (x y : ℝ) => (3 * x - 2 * y - 1) * (x + 2 * y - 4) = 0
  2

/-- Theorem stating that the number of distinct intersection points is 2 -/
theorem intersection_points_count :
  num_intersection_points = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3047_304745


namespace NUMINAMATH_CALUDE_right_angle_constraint_l3047_304799

/-- Given two points A and B on the x-axis, and a point P on a line,
    prove that if ∠APB is a right angle, then the distance between A and B
    is at least 10 units. -/
theorem right_angle_constraint (m : ℝ) (h_m : m > 0) :
  (∃ (x y : ℝ), 3 * x + 4 * y + 25 = 0 ∧
    ((x + m) * (x - m) + y * y = 0)) →
  m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_right_angle_constraint_l3047_304799


namespace NUMINAMATH_CALUDE_jean_thursday_calls_correct_l3047_304747

/-- The number of calls Jean answered on each day of the week --/
structure CallData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days --/
def working_days : ℕ := 5

/-- Jean's actual call data for the week --/
def jean_calls : CallData where
  monday := 35
  tuesday := 46
  wednesday := 27
  thursday := 61  -- This is what we want to prove
  friday := 31

/-- Theorem stating that Jean's Thursday call count is correct --/
theorem jean_thursday_calls_correct :
  jean_calls.thursday = 
    working_days * average_calls - 
    (jean_calls.monday + jean_calls.tuesday + jean_calls.wednesday + jean_calls.friday) := by
  sorry


end NUMINAMATH_CALUDE_jean_thursday_calls_correct_l3047_304747


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3047_304723

-- Define the mixed number addition function
def mixed_number_add (a b c d : ℚ) : ℚ := a + b + c + d

-- Theorem 1
theorem problem_1 : 
  mixed_number_add (-2020 - 2/3) (2019 + 3/4) (-2018 - 5/6) (2017 + 1/2) = -2 - 1/4 := by sorry

-- Theorem 2
theorem problem_2 : 
  mixed_number_add (-1 - 1/2) (-2000 - 5/6) (4000 + 3/4) (-1999 - 2/3) = -5/4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3047_304723


namespace NUMINAMATH_CALUDE_min_hexagon_area_l3047_304796

/-- Represents a configuration of two intersecting triangles -/
structure IntersectingTriangles where
  /-- The total number of disjoint regions formed -/
  num_regions : Nat
  /-- The number of triangular regions -/
  num_triangles : Nat
  /-- The areas of the triangular regions -/
  triangle_areas : Fin num_triangles → ℝ
  /-- The area of the hexagonal region -/
  hexagon_area : ℝ
  /-- Assertion that there are 7 regions in total -/
  region_count : num_regions = 7
  /-- Assertion that 6 of the regions are triangles -/
  triangle_count : num_triangles = 6
  /-- Assertion that all triangular regions have area 1 -/
  unit_triangles : ∀ i, triangle_areas i = 1

/-- The theorem stating the minimum possible area of the hexagonal region -/
theorem min_hexagon_area (config : IntersectingTriangles) :
  config.hexagon_area ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_min_hexagon_area_l3047_304796


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l3047_304719

def complex_mul (a b c d : ℝ) : ℂ :=
  (a * c - b * d : ℝ) + (a * d + b * c : ℝ) * Complex.I

theorem imaginary_part_of_product :
  let z₁ : ℂ := 1 - Complex.I
  let z₂ : ℂ := 2 + 4 * Complex.I
  Complex.im (z₁ * z₂) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l3047_304719


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_value_l3047_304749

/-- Given a geometric sequence {a_n} where a₁a₁₃ + 2a₇² = 4π, prove that tan(a₂a₁₂) = √3 -/
theorem geometric_sequence_tan_value (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  (h_sum : a 1 * a 13 + 2 * (a 7)^2 = 4 * Real.pi)  -- Given equation
  : Real.tan (a 2 * a 12) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_value_l3047_304749


namespace NUMINAMATH_CALUDE_intersection_points_theorem_pair_c_not_solution_pair_a_solution_pair_b_solution_pair_d_solution_pair_e_solution_l3047_304777

theorem intersection_points_theorem : 
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x = 2 ∨ x = 4) :=
by sorry

theorem pair_c_not_solution :
  ¬∃ (x : ℝ), (x - 2 = x - 4) ∧ (x^2 - 6*x + 8 = 0) :=
by sorry

theorem pair_a_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x + 8 = 0 ∧ 0 = 0) :=
by sorry

theorem pair_b_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x = 8) :=
by sorry

theorem pair_d_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x + 9 = 1) :=
by sorry

theorem pair_e_solution :
  ∃ (x : ℝ), (x^2 - 5 = 6*x - 8) ∧ (x^2 - 6*x + 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_pair_c_not_solution_pair_a_solution_pair_b_solution_pair_d_solution_pair_e_solution_l3047_304777


namespace NUMINAMATH_CALUDE_distinct_prime_factors_sum_and_product_l3047_304759

def number : Nat := 420

theorem distinct_prime_factors_sum_and_product :
  (Finset.sum (Nat.factors number).toFinset id = 17) ∧
  (Finset.prod (Nat.factors number).toFinset id = 210) := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_sum_and_product_l3047_304759
