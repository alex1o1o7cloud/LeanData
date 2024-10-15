import Mathlib

namespace NUMINAMATH_CALUDE_family_museum_cost_calculation_l845_84543

/-- Calculates the discounted ticket price based on age --/
def discountedPrice (age : ℕ) (basePrice : ℚ) : ℚ :=
  if age ≥ 65 then basePrice * (1 - 0.2)
  else if age ≥ 12 ∧ age ≤ 18 then basePrice * (1 - 0.3)
  else if age ≥ 0 ∧ age ≤ 11 then basePrice * (1 - 0.5)
  else basePrice

/-- Calculates the total cost for a family museum trip --/
def familyMuseumCost (ages : List ℕ) (regularPrice specialPrice taxRate : ℚ) : ℚ :=
  let totalBeforeTax := (ages.map (fun age => discountedPrice age regularPrice + specialPrice)).sum
  totalBeforeTax * (1 + taxRate)

theorem family_museum_cost_calculation :
  let ages := [15, 10, 40, 42, 65]
  let regularPrice := 10
  let specialPrice := 5
  let taxRate := 0.1
  familyMuseumCost ages regularPrice specialPrice taxRate = 71.5 := by sorry

end NUMINAMATH_CALUDE_family_museum_cost_calculation_l845_84543


namespace NUMINAMATH_CALUDE_total_trips_is_forty_l845_84534

/-- The number of trips Jean makes -/
def jean_trips : ℕ := 23

/-- The difference between Jean's and Bill's trips -/
def trip_difference : ℕ := 6

/-- Calculates the total number of trips made by Bill and Jean -/
def total_trips : ℕ := jean_trips + (jean_trips - trip_difference)

/-- Proves that the total number of trips made by Bill and Jean is 40 -/
theorem total_trips_is_forty : total_trips = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_trips_is_forty_l845_84534


namespace NUMINAMATH_CALUDE_line_equation_l845_84515

/-- Given a line passing through (-a, 0) and forming a triangle in the second quadrant with area T,
    prove that its equation is 2Tx - a²y + 2aT = 0 -/
theorem line_equation (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = -a ∧ y = 0) ∨ (x < 0 ∧ y > 0) →
    (y = m * x + b ↔ 2 * T * x - a^2 * y + 2 * a * T = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l845_84515


namespace NUMINAMATH_CALUDE_quadratic_translation_problem_solution_l845_84568

/-- Represents a horizontal and vertical translation of a quadratic function -/
structure Translation where
  horizontal : ℝ
  vertical : ℝ

/-- Applies a translation to a quadratic function -/
def apply_translation (f : ℝ → ℝ) (t : Translation) : ℝ → ℝ :=
  λ x => f (x + t.horizontal) - t.vertical

theorem quadratic_translation (a : ℝ) (t : Translation) :
  apply_translation (λ x => a * x^2) t =
  λ x => a * (x + t.horizontal)^2 - t.vertical := by
  sorry

/-- The specific translation in the problem -/
def problem_translation : Translation :=
  { horizontal := 3, vertical := 2 }

theorem problem_solution :
  apply_translation (λ x => 2 * x^2) problem_translation =
  λ x => 2 * (x + 3)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_translation_problem_solution_l845_84568


namespace NUMINAMATH_CALUDE_infinite_power_tower_four_l845_84555

/-- The limit of the sequence defined by a_0 = x, a_(n+1) = x^(a_n) --/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := sorry

theorem infinite_power_tower_four (x : ℝ) :
  x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_infinite_power_tower_four_l845_84555


namespace NUMINAMATH_CALUDE_binomial_15_4_l845_84525

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l845_84525


namespace NUMINAMATH_CALUDE_small_circle_radius_l845_84556

/-- Given a large circle with radius 10 meters containing three smaller circles
    that touch each other and are aligned horizontally across its center,
    prove that the radius of each smaller circle is 10/3 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : R = 10 →
  3 * (2 * r) = 2 * R →
  r = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l845_84556


namespace NUMINAMATH_CALUDE_inverse_89_mod_91_l845_84594

theorem inverse_89_mod_91 : ∃ x : ℕ, x < 91 ∧ (89 * x) % 91 = 1 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_89_mod_91_l845_84594


namespace NUMINAMATH_CALUDE_root_in_interval_l845_84558

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + x - 5

-- State the theorem
theorem root_in_interval (a b : ℕ+) (x₀ : ℝ) :
  b - a = 1 →
  ∃ x₀, x₀ ∈ Set.Icc a b ∧ f x₀ = 0 →
  a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l845_84558


namespace NUMINAMATH_CALUDE_choir_group_size_l845_84567

theorem choir_group_size (total : ℕ) (group2 : ℕ) (group3 : ℕ) (h1 : total = 70) (h2 : group2 = 30) (h3 : group3 = 15) :
  total - group2 - group3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_choir_group_size_l845_84567


namespace NUMINAMATH_CALUDE_units_digit_of_42_pow_5_plus_27_pow_5_l845_84522

theorem units_digit_of_42_pow_5_plus_27_pow_5 : (42^5 + 27^5) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_42_pow_5_plus_27_pow_5_l845_84522


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l845_84542

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 9)
  (h_a5 : a 5 = 33) :
  ∃ d : ℝ, d = 8 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l845_84542


namespace NUMINAMATH_CALUDE_matrix_commutation_result_l845_84583

theorem matrix_commutation_result (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ c) → ((a - d) / (c - 4 * b) = -3) := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_result_l845_84583


namespace NUMINAMATH_CALUDE_fraction_always_nonnegative_l845_84573

theorem fraction_always_nonnegative (x : ℝ) : (x^2 + 2*x + 1) / (x^2 + 4*x + 8) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_always_nonnegative_l845_84573


namespace NUMINAMATH_CALUDE_f_of_1_plus_g_of_3_l845_84581

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_of_1_plus_g_of_3 : f (1 + g 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_of_1_plus_g_of_3_l845_84581


namespace NUMINAMATH_CALUDE_inequality_implication_l845_84536

theorem inequality_implication (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
sorry

end NUMINAMATH_CALUDE_inequality_implication_l845_84536


namespace NUMINAMATH_CALUDE_first_number_proof_l845_84541

theorem first_number_proof (y : ℝ) (h1 : y = 48) (h2 : ∃ x : ℝ, x + (1/4) * y = 27) : 
  ∃ x : ℝ, x + (1/4) * y = 27 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l845_84541


namespace NUMINAMATH_CALUDE_tetrahedron_centroid_intersection_sum_l845_84585

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Distance between two points in 3D space -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Intersection point of a line and a face of the tetrahedron -/
def intersectionPoint (l : Line3D) (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

theorem tetrahedron_centroid_intersection_sum (t : Tetrahedron) (l : Line3D) : 
  let G := centroid t
  let M := intersectionPoint l t 0
  let N := intersectionPoint l t 1
  let S := intersectionPoint l t 2
  let T := intersectionPoint l t 3
  1 / distance G M + 1 / distance G N + 1 / distance G S + 1 / distance G T = 0 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_centroid_intersection_sum_l845_84585


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l845_84535

/-- Given an isosceles triangle with base angle α and difference b between
    the radii of its circumscribed and inscribed circles, 
    the length of its base side is (2b * sin(2α)) / (1 - tan²(α/2)) -/
theorem isosceles_triangle_base_length 
  (α : ℝ) 
  (b : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < b) : 
  ∃ (x : ℝ), x = (2 * b * Real.sin (2 * α)) / (1 - Real.tan (α / 2) ^ 2) ∧ 
  x > 0 ∧ 
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R - r = b ∧
  R = x / (2 * Real.sin (2 * α)) ∧
  r = x / 2 * Real.tan (α / 2) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l845_84535


namespace NUMINAMATH_CALUDE_intersection_points_form_equilateral_triangle_l845_84561

/-- The common points of the circle x^2 + (y - 1)^2 = 1 and the ellipse 9x^2 + (y + 1)^2 = 9 form an equilateral triangle -/
theorem intersection_points_form_equilateral_triangle :
  ∀ (A B C : ℝ × ℝ),
  (A ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  (B ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  (C ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  A ≠ B → B ≠ C → A ≠ C →
  dist A B = dist B C ∧ dist B C = dist C A :=
by sorry


end NUMINAMATH_CALUDE_intersection_points_form_equilateral_triangle_l845_84561


namespace NUMINAMATH_CALUDE_train_length_l845_84574

theorem train_length (t_platform : ℝ) (t_pole : ℝ) (l_platform : ℝ) 
  (h1 : t_platform = 36)
  (h2 : t_pole = 18)
  (h3 : l_platform = 300) :
  ∃ l_train : ℝ, l_train = 300 ∧ l_train / t_pole = (l_train + l_platform) / t_platform :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l845_84574


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l845_84562

/-- Represents a repeating decimal with a single repeating digit -/
def single_repeat (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two repeating digits -/
def double_repeat (a b : ℕ) : ℚ := (10 * a + b) / 99

/-- The sum of 0.777... and 0.131313... is equal to 10/11 -/
theorem sum_of_repeating_decimals : 
  single_repeat 7 + double_repeat 1 3 = 10 / 11 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l845_84562


namespace NUMINAMATH_CALUDE_root_of_f_l845_84550

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is indeed the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Given condition: f⁻¹(0) = 2
axiom inverse_intersect_y : f_inv 0 = 2

-- Theorem to prove
theorem root_of_f (h : f_inv 0 = 2) : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_f_l845_84550


namespace NUMINAMATH_CALUDE_f_properties_l845_84537

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_properties :
  (∀ x > 0, f x ≥ -1 / Real.exp 1) ∧
  (∀ t > 0, (∀ x ∈ Set.Icc t (t + 2), f x ≥ min (-1 / Real.exp 1) (f t))) ∧
  (∀ x > 0, Real.log x > 1 / (Real.exp x) - 2 / (Real.exp 1 * x)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l845_84537


namespace NUMINAMATH_CALUDE_regular_price_is_18_l845_84549

/-- The regular price of a medium pizza at Joe's pizzeria -/
def regular_price : ℝ := 18

/-- The cost of 3 medium pizzas with the promotion -/
def promotion_cost : ℝ := 15

/-- The total savings when taking full advantage of the promotion -/
def total_savings : ℝ := 39

/-- Theorem stating that the regular price of a medium pizza is $18 -/
theorem regular_price_is_18 :
  regular_price = (promotion_cost + total_savings) / 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_price_is_18_l845_84549


namespace NUMINAMATH_CALUDE_wildlife_population_estimate_l845_84591

theorem wildlife_population_estimate 
  (tagged_released : ℕ) 
  (later_captured : ℕ) 
  (tagged_in_sample : ℕ) 
  (h1 : tagged_released = 1200)
  (h2 : later_captured = 1000)
  (h3 : tagged_in_sample = 100) :
  (tagged_released * later_captured) / tagged_in_sample = 12000 :=
by sorry

end NUMINAMATH_CALUDE_wildlife_population_estimate_l845_84591


namespace NUMINAMATH_CALUDE_x_value_when_y_is_14_l845_84598

theorem x_value_when_y_is_14 (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 5) 
  (h3 : y = 14) : 
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_14_l845_84598


namespace NUMINAMATH_CALUDE_cubic_sum_from_elementary_symmetric_polynomials_l845_84586

theorem cubic_sum_from_elementary_symmetric_polynomials (p q r : ℝ) 
  (h1 : p + q + r = 7)
  (h2 : p * q + p * r + q * r = 8)
  (h3 : p * q * r = -15) :
  p^3 + q^3 + r^3 = 151 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_from_elementary_symmetric_polynomials_l845_84586


namespace NUMINAMATH_CALUDE_triangle_internal_region_l845_84538

-- Define the three lines that form the triangle
def line1 (x y : ℝ) : Prop := x + 2*y = 2
def line2 (x y : ℝ) : Prop := 2*x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the internal region of the triangle
def internal_region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2*y < 2 ∧ 2*x + y > 2

-- Theorem statement
theorem triangle_internal_region :
  ∀ x y : ℝ, 
    (∃ ε > 0, line1 (x + ε) y ∨ line2 (x + ε) y ∨ line3 (x + ε) y) →
    (∃ ε > 0, line1 (x - ε) y ∨ line2 (x - ε) y ∨ line3 (x - ε) y) →
    (∃ ε > 0, line1 x (y + ε) ∨ line2 x (y + ε) ∨ line3 x (y + ε)) →
    (∃ ε > 0, line1 x (y - ε) ∨ line2 x (y - ε) ∨ line3 x (y - ε)) →
    internal_region x y :=
sorry

end NUMINAMATH_CALUDE_triangle_internal_region_l845_84538


namespace NUMINAMATH_CALUDE_unknowns_and_variables_l845_84553

-- Define a type for equations
structure Equation where
  f : ℝ → ℝ → ℝ
  c : ℝ

-- Define a type for systems of equations
structure SystemOfEquations where
  eq1 : Equation
  eq2 : Equation

-- Define a type for single equations
structure SingleEquation where
  eq : Equation

-- Define a property for being an unknown
def isUnknown (x : ℝ) (y : ℝ) (system : SystemOfEquations) : Prop :=
  ∀ (sol_x sol_y : ℝ), system.eq1.f sol_x sol_y = system.eq1.c ∧ 
                        system.eq2.f sol_x sol_y = system.eq2.c →
                        x = sol_x ∧ y = sol_y

-- Define a property for being a variable
def isVariable (x : ℝ) (y : ℝ) (single : SingleEquation) : Prop :=
  ∀ (val_x : ℝ), ∃ (val_y : ℝ), single.eq.f val_x val_y = single.eq.c

-- Theorem statement
theorem unknowns_and_variables 
  (x y : ℝ) (system : SystemOfEquations) (single : SingleEquation) : 
  (isUnknown x y system) ∧ (isVariable x y single) := by
  sorry

end NUMINAMATH_CALUDE_unknowns_and_variables_l845_84553


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l845_84569

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_sum1 : a 1 + a 3 = 6)
  (h_sum2 : (a 1 + a 2 + a 3 + a 4) + a 2 = (a 1 + a 2 + a 3) + 3)
  : q = (1 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l845_84569


namespace NUMINAMATH_CALUDE_sector_central_angle_l845_84582

/-- Given a sector with radius 6 and area 6π, its central angle measure in degrees is 60. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) : 
  radius = 6 → area = 6 * Real.pi → angle = (area * 360) / (Real.pi * radius ^ 2) → angle = 60 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l845_84582


namespace NUMINAMATH_CALUDE_problem_statement_l845_84548

theorem problem_statement (m n : ℤ) : 
  (∃ k : ℤ, 56786730 * k = m * n * (m^60 - n^60)) ∧ 
  (m^5 + 3*m^4*n - 5*m^3*n^2 - 15*m^2*n^3 + 4*m*n^4 + 12*n^5 ≠ 33) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l845_84548


namespace NUMINAMATH_CALUDE_two_digit_divisors_of_723_with_remainder_30_l845_84516

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divides_with_remainder (d q r : ℕ) : Prop := ∃ k, d * k + r = q

theorem two_digit_divisors_of_723_with_remainder_30 :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S, is_two_digit n ∧ divides_with_remainder n 723 30) ∧
    S.card = 4 ∧
    S = {33, 63, 77, 99} :=
by sorry

end NUMINAMATH_CALUDE_two_digit_divisors_of_723_with_remainder_30_l845_84516


namespace NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l845_84530

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that missing both times is the complement of hitting at least once
theorem complement_of_hit_at_least_once :
  ∀ ω : Ω, miss_both_times ω ↔ ¬(hit_at_least_once ω) :=
sorry

end NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l845_84530


namespace NUMINAMATH_CALUDE_fixed_points_of_quadratic_l845_84504

/-- The quadratic function f(x) always passes through two fixed points -/
theorem fixed_points_of_quadratic (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + (3*a - 1)*x - (10*a + 3)
  (f 2 = -5 ∧ f (-5) = 2) := by sorry

end NUMINAMATH_CALUDE_fixed_points_of_quadratic_l845_84504


namespace NUMINAMATH_CALUDE_interest_rate_equivalence_l845_84588

/-- Given an amount A that produces the same interest in 12 years as Rs 1000 produces in 2 years at 12%,
    prove that the interest rate R for amount A is 12%. -/
theorem interest_rate_equivalence (A : ℝ) (R : ℝ) : A > 0 →
  A * R * 12 = 1000 * 12 * 2 →
  R = 12 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equivalence_l845_84588


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l845_84531

/-- Given vectors a and b in ℝ³, prove that a - 5b equals the expected result. -/
theorem vector_subtraction_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) :
  a = (-5, 3, 2) → b = (2, -1, 4) → a - 5 • b = (-15, 8, -18) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l845_84531


namespace NUMINAMATH_CALUDE_expand_product_l845_84580

theorem expand_product (x : ℝ) : (2 * x + 3) * (x - 4) = 2 * x^2 - 5 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l845_84580


namespace NUMINAMATH_CALUDE_ellipse_equation_l845_84552

/-- The equation of an ellipse with foci at (-2,0) and (2,0) passing through (2, 5/3) -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x₀ y₀ : ℝ), x₀^2/a^2 + y₀^2/b^2 = 1 ↔ 
      (Real.sqrt ((x₀ + 2)^2 + y₀^2) + Real.sqrt ((x₀ - 2)^2 + y₀^2) = 2*a)) ∧
    (2^2/a^2 + (5/3)^2/b^2 = 1)) →
  x^2/9 + y^2/5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l845_84552


namespace NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l845_84589

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_one_billion_scientific_notation :
  toScientificNotation (21000000000 : ℝ) = ScientificNotation.mk 2.1 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l845_84589


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l845_84587

theorem smallest_next_divisor (m : ℕ) : 
  m % 2 = 0 ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 437 = 0 → 
  ∃ (d : ℕ), d > 437 ∧ m % d = 0 ∧ d ≥ 874 ∧ 
  ∀ (d' : ℕ), d' > 437 ∧ m % d' = 0 → d' ≥ 874 :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l845_84587


namespace NUMINAMATH_CALUDE_diff_color_probability_is_three_fourths_l845_84571

/-- The number of color choices for socks -/
def sock_colors : ℕ := 3

/-- The number of color choices for shorts -/
def short_colors : ℕ := 4

/-- The total number of possible combinations -/
def total_combinations : ℕ := sock_colors * short_colors

/-- The number of combinations where socks and shorts have the same color -/
def same_color_combinations : ℕ := 3

/-- The probability of selecting different colors for socks and shorts -/
def diff_color_probability : ℚ := (total_combinations - same_color_combinations : ℚ) / total_combinations

theorem diff_color_probability_is_three_fourths :
  diff_color_probability = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_diff_color_probability_is_three_fourths_l845_84571


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l845_84500

theorem sum_and_equal_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 150)
  (equal_numbers : a - 3 = b + 4 ∧ b + 4 = 4 * c) : 
  a = 631 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l845_84500


namespace NUMINAMATH_CALUDE_paddyfield_warbler_percentage_l845_84507

/-- Represents the composition of birds in a nature reserve -/
structure BirdPopulation where
  total : ℝ
  hawk_percent : ℝ
  other_percent : ℝ
  kingfisher_to_warbler_ratio : ℝ

/-- Theorem about the percentage of paddyfield-warblers among non-hawks -/
theorem paddyfield_warbler_percentage
  (pop : BirdPopulation)
  (h1 : pop.hawk_percent = 0.3)
  (h2 : pop.other_percent = 0.35)
  (h3 : pop.kingfisher_to_warbler_ratio = 0.25)
  : (((1 - pop.hawk_percent - pop.other_percent) * pop.total) / ((1 - pop.hawk_percent) * pop.total)) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_paddyfield_warbler_percentage_l845_84507


namespace NUMINAMATH_CALUDE_equation_solutions_l845_84540

theorem equation_solutions :
  (∃ (s1 s2 : Set ℝ),
    (s1 = {x : ℝ | (x - 1)^2 - 25 = 0} ∧ s1 = {6, -4}) ∧
    (s2 = {x : ℝ | 3*x*(x - 2) = x - 2} ∧ s2 = {2, 1/3})) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l845_84540


namespace NUMINAMATH_CALUDE_central_cell_value_l845_84539

def table_sum (a : ℝ) : ℝ :=
  a + 4*a + 16*a + 3*a + 12*a + 48*a + 9*a + 36*a + 144*a

theorem central_cell_value (a : ℝ) (h : table_sum a = 546) : 12 * a = 24 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l845_84539


namespace NUMINAMATH_CALUDE_train_speed_through_tunnel_l845_84564

/-- Calculates the speed of a train passing through a tunnel -/
theorem train_speed_through_tunnel
  (train_length : ℝ)
  (tunnel_length : ℝ)
  (time_to_pass : ℝ)
  (h1 : train_length = 300)
  (h2 : tunnel_length = 1200)
  (h3 : time_to_pass = 100)
  : (train_length + tunnel_length) / time_to_pass * 3.6 = 54 := by
  sorry

#check train_speed_through_tunnel

end NUMINAMATH_CALUDE_train_speed_through_tunnel_l845_84564


namespace NUMINAMATH_CALUDE_power_sum_of_i_l845_84593

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^22 + i^222 = -2 := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l845_84593


namespace NUMINAMATH_CALUDE_construct_one_degree_angle_l845_84551

-- Define the given angle
def given_angle : ℕ := 19

-- Define the target angle
def target_angle : ℕ := 1

-- Theorem stating that it's possible to construct the target angle from the given angle
theorem construct_one_degree_angle :
  ∃ n : ℕ, (n * given_angle) % 360 = target_angle :=
sorry

end NUMINAMATH_CALUDE_construct_one_degree_angle_l845_84551


namespace NUMINAMATH_CALUDE_gym_towels_l845_84501

theorem gym_towels (first_hour : ℕ) (second_hour_increase : ℚ) 
  (third_hour_increase : ℚ) (fourth_hour_increase : ℚ) 
  (total_towels : ℕ) : 
  first_hour = 50 →
  second_hour_increase = 1/5 →
  third_hour_increase = 1/4 →
  fourth_hour_increase = 1/3 →
  total_towels = 285 →
  let second_hour := first_hour + (first_hour * second_hour_increase).floor
  let third_hour := second_hour + (second_hour * third_hour_increase).floor
  let fourth_hour := third_hour + (third_hour * fourth_hour_increase).floor
  first_hour + second_hour + third_hour + fourth_hour = total_towels :=
by sorry

end NUMINAMATH_CALUDE_gym_towels_l845_84501


namespace NUMINAMATH_CALUDE_intersection_point_with_median_line_l845_84554

open Complex

/-- Given complex numbers and a curve, prove the intersection point with the median line -/
theorem intersection_point_with_median_line 
  (a b c : ℝ) 
  (z₁₁ : ℂ) 
  (z₁ : ℂ) 
  (z₂ : ℂ) 
  (h_z₁₁ : z₁₁ = Complex.I * a) 
  (h_z₁ : z₁ = (1/2 : ℝ) + Complex.I * b) 
  (h_z₂ : z₂ = 1 + Complex.I * c) 
  (h_non_collinear : a + c ≠ 2 * b) 
  (z : ℝ → ℂ) 
  (h_z : ∀ t, z t = z₁ * (cos t)^4 + 2 * z₁ * (cos t)^2 * (sin t)^2 + z₂ * (sin t)^4) :
  ∃! p : ℂ, p ∈ Set.range z ∧ 
    p.re = (1/2 : ℝ) ∧ 
    p.im = (a + c + 2*b) / 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_with_median_line_l845_84554


namespace NUMINAMATH_CALUDE_box_office_scientific_notation_l845_84528

theorem box_office_scientific_notation :
  let billion : ℝ := 10^9
  let box_office : ℝ := 40.25 * billion
  box_office = 4.025 * 10^9 := by sorry

end NUMINAMATH_CALUDE_box_office_scientific_notation_l845_84528


namespace NUMINAMATH_CALUDE_correct_graph_representation_l845_84532

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup -/
def problem (m n : Car) : Prop :=
  m.speed > 0 ∧
  n.speed = 2 * m.speed ∧
  m.distance = n.distance ∧
  m.distance = m.speed * m.time ∧
  n.distance = n.speed * n.time

/-- The theorem to prove -/
theorem correct_graph_representation (m n : Car) 
  (h : problem m n) : n.speed = 2 * m.speed ∧ n.time = m.time / 2 := by
  sorry


end NUMINAMATH_CALUDE_correct_graph_representation_l845_84532


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l845_84510

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes : 
  let total_players : ℕ := 8
  let players_per_team : ℕ := 2
  let total_teams : ℕ := total_players / players_per_team
  let handshakes_per_player : ℕ := total_players - players_per_team
  total_players * handshakes_per_player / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l845_84510


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l845_84521

/-- Represents the number of students to be drawn from a stratum -/
structure SampleSize (total : ℕ) (stratum : ℕ) (drawn : ℕ) where
  size : ℕ
  proportional : size * total = stratum * drawn

/-- The problem statement -/
theorem stratified_sampling_problem :
  let total_students : ℕ := 1400
  let male_students : ℕ := 800
  let female_students : ℕ := 600
  let male_drawn : ℕ := 40
  ∃ (female_sample : SampleSize total_students female_students male_drawn),
    female_sample.size = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l845_84521


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l845_84505

theorem quadratic_inequality_solution_sets (a : ℝ) :
  (∀ x, 6 * x^2 + a * x - a^2 < 0 ↔ 
    (a > 0 ∧ -a/2 < x ∧ x < a/3) ∨
    (a < 0 ∧ a/3 < x ∧ x < -a/2)) ∧
  (a = 0 → ∀ x, ¬(6 * x^2 + a * x - a^2 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l845_84505


namespace NUMINAMATH_CALUDE_buying_problem_equations_l845_84560

theorem buying_problem_equations (x y : ℕ) : 
  x > 0 → y > 0 → (8 * x - y = 3 ∧ y - 7 * x = 4) → True := by
  sorry

end NUMINAMATH_CALUDE_buying_problem_equations_l845_84560


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l845_84597

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y, m * x + (m + 2) * y - 1 = 0 ∧ (m - 1) * x + m * y = 0 → 
    (m * (m - 1) + (m + 2) * m = 0 ∨ m = 0)) → 
  (m = 0 ∨ m = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l845_84597


namespace NUMINAMATH_CALUDE_find_k_l845_84502

-- Define the binary linear equation
def binary_linear_equation (x y t : ℝ) : Prop := 3 * x - 2 * y = t

-- Define the theorem
theorem find_k (m n : ℝ) (h1 : binary_linear_equation m n 5) 
  (h2 : binary_linear_equation (m + 2) (n - 2) k) : k = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l845_84502


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_proof_l845_84527

/-- The measure of an interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let total_interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := total_interior_angle_sum / n
  144

/-- Proof of the theorem -/
theorem regular_decagon_interior_angle_proof :
  regular_decagon_interior_angle = 144 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_proof_l845_84527


namespace NUMINAMATH_CALUDE_tan_and_sin_values_l845_84518

theorem tan_and_sin_values (α : ℝ) (h : Real.tan (α + π / 4) = -3) : 
  Real.tan α = 1 ∧ Real.sin (2 * α + π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_sin_values_l845_84518


namespace NUMINAMATH_CALUDE_sum_floor_equals_126_l845_84546

theorem sum_floor_equals_126 
  (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2008 ∧ c^2 + d^2 = 2008)
  (products : a*c = 1000 ∧ b*d = 1000) : 
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_floor_equals_126_l845_84546


namespace NUMINAMATH_CALUDE_inequality_solution_l845_84577

theorem inequality_solution (x : ℝ) : 
  1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) > 1 / 4 ↔ 
  x < -2 ∨ (0 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l845_84577


namespace NUMINAMATH_CALUDE_ball_selection_ways_l845_84572

/-- Represents the number of ways to select balls from a bag -/
def select_balls (total white red black : ℕ) (select : ℕ) 
  (white_min white_max red_min black_max : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to select balls under given conditions -/
theorem ball_selection_ways : 
  select_balls 20 9 5 6 10 2 8 2 3 = 16 := by sorry

end NUMINAMATH_CALUDE_ball_selection_ways_l845_84572


namespace NUMINAMATH_CALUDE_corn_profit_problem_l845_84596

theorem corn_profit_problem (seeds_per_ear : ℕ) (ear_price : ℚ) (bag_price : ℚ) (seeds_per_bag : ℕ) (total_profit : ℚ) :
  seeds_per_ear = 4 →
  ear_price = 1/10 →
  bag_price = 1/2 →
  seeds_per_bag = 100 →
  total_profit = 40 →
  (total_profit / (ear_price - (bag_price / seeds_per_bag) * seeds_per_ear) : ℚ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_corn_profit_problem_l845_84596


namespace NUMINAMATH_CALUDE_max_airline_services_l845_84578

theorem max_airline_services (internet_percentage : ℝ) (snack_percentage : ℝ) 
  (h1 : internet_percentage = 35) 
  (h2 : snack_percentage = 70) : 
  ∃ (max_both_percentage : ℝ), max_both_percentage ≤ 35 ∧ 
  ∀ (both_percentage : ℝ), 
    (both_percentage ≤ internet_percentage ∧ 
     both_percentage ≤ snack_percentage) → 
    both_percentage ≤ max_both_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_airline_services_l845_84578


namespace NUMINAMATH_CALUDE_symmetric_angles_theorem_l845_84579

-- Define the property of terminal sides being symmetric with respect to x + y = 0
def symmetric_terminal_sides (α β : Real) : Prop := sorry

-- Define the set of angles β
def angle_set : Set Real := {β | ∃ k : Int, β = 2 * k * Real.pi - Real.pi / 6}

-- State the theorem
theorem symmetric_angles_theorem (α β : Real) 
  (h_symmetric : symmetric_terminal_sides α β) 
  (h_alpha : α = -Real.pi / 3) : 
  β ∈ angle_set := by sorry

end NUMINAMATH_CALUDE_symmetric_angles_theorem_l845_84579


namespace NUMINAMATH_CALUDE_total_baseball_cards_l845_84547

/-- The number of people who have baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 3

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_baseball_cards : total_cards = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l845_84547


namespace NUMINAMATH_CALUDE_louise_pencil_boxes_l845_84508

def pencil_problem (box_capacity : ℕ) (red_pencils : ℕ) (yellow_pencils : ℕ) : Prop :=
  let blue_pencils := 2 * red_pencils
  let green_pencils := red_pencils + blue_pencils
  let total_boxes := 
    (red_pencils + blue_pencils + yellow_pencils + green_pencils) / box_capacity
  total_boxes = 8

theorem louise_pencil_boxes : 
  pencil_problem 20 20 40 :=
sorry

end NUMINAMATH_CALUDE_louise_pencil_boxes_l845_84508


namespace NUMINAMATH_CALUDE_number_of_ways_to_buy_three_items_l845_84563

/-- The number of headphones available -/
def num_headphones : ℕ := 9

/-- The number of computer mice available -/
def num_mice : ℕ := 13

/-- The number of keyboards available -/
def num_keyboards : ℕ := 5

/-- The number of "keyboard and mouse" sets available -/
def num_keyboard_mouse_sets : ℕ := 4

/-- The number of "headphones and mouse" sets available -/
def num_headphones_mouse_sets : ℕ := 5

/-- The theorem stating the number of ways to buy three items -/
theorem number_of_ways_to_buy_three_items : 
  num_keyboard_mouse_sets * num_headphones + 
  num_headphones_mouse_sets * num_keyboards + 
  num_headphones * num_mice * num_keyboards = 646 := by
  sorry


end NUMINAMATH_CALUDE_number_of_ways_to_buy_three_items_l845_84563


namespace NUMINAMATH_CALUDE_remainder_sum_l845_84524

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9 = 3) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l845_84524


namespace NUMINAMATH_CALUDE_prop_p_true_prop_q_false_prop_2_true_prop_3_true_l845_84545

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem statements
theorem prop_p_true : p := by sorry

theorem prop_q_false : ¬q := by sorry

theorem prop_2_true : p ∨ q := by sorry

theorem prop_3_true : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_prop_p_true_prop_q_false_prop_2_true_prop_3_true_l845_84545


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l845_84503

/-- The length of the diagonal of a rectangular prism with dimensions 12, 16, and 21 -/
def prism_diagonal : ℝ := 29

/-- Theorem: The diagonal of a rectangular prism with dimensions 12, 16, and 21 is 29 -/
theorem rectangular_prism_diagonal :
  let a : ℝ := 12
  let b : ℝ := 16
  let c : ℝ := 21
  Real.sqrt (a^2 + b^2 + c^2) = prism_diagonal :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l845_84503


namespace NUMINAMATH_CALUDE_subset_partition_with_closure_l845_84513

theorem subset_partition_with_closure (A B C : Set ℕ+) : 
  (A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅) ∧ 
  (A ∪ B ∪ C = Set.univ) ∧
  (∀ a ∈ A, ∀ b ∈ B, ∀ c ∈ C, (a + c : ℕ+) ∈ A ∧ (b + c : ℕ+) ∈ B ∧ (a + b : ℕ+) ∈ C) →
  ((A = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 2} ∧ 
    B = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 1} ∧ 
    C = {n : ℕ+ | ∃ k : ℕ+, n = 3*k}) ∨
   (A = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 1} ∧ 
    B = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 2} ∧ 
    C = {n : ℕ+ | ∃ k : ℕ+, n = 3*k})) :=
by sorry

end NUMINAMATH_CALUDE_subset_partition_with_closure_l845_84513


namespace NUMINAMATH_CALUDE_parking_savings_l845_84526

/-- Calculates the yearly savings when renting a parking space monthly instead of weekly. -/
theorem parking_savings (weekly_rate : ℕ) (monthly_rate : ℕ) : 
  weekly_rate = 10 → monthly_rate = 24 → (52 * weekly_rate) - (12 * monthly_rate) = 232 := by
  sorry

end NUMINAMATH_CALUDE_parking_savings_l845_84526


namespace NUMINAMATH_CALUDE_sample_size_is_fifteen_l845_84506

/-- Represents the stratified sampling scenario -/
structure StratifiedSampling where
  total_employees : ℕ
  young_employees : ℕ
  young_in_sample : ℕ

/-- Calculates the sample size for a given stratified sampling scenario -/
def sample_size (s : StratifiedSampling) : ℕ :=
  s.total_employees / (s.young_employees / s.young_in_sample)

/-- Theorem stating that the sample size is 15 for the given scenario -/
theorem sample_size_is_fifteen :
  let s : StratifiedSampling := {
    total_employees := 75,
    young_employees := 35,
    young_in_sample := 7
  }
  sample_size s = 15 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_fifteen_l845_84506


namespace NUMINAMATH_CALUDE_unique_valid_number_l845_84576

def is_valid_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧
  tens = hundreds + 3 ∧
  units = tens - 4 ∧
  (hundreds + tens + units) / 2 = tens

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 473 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l845_84576


namespace NUMINAMATH_CALUDE_number_wall_x_value_l845_84544

/-- Represents a simplified number wall with given conditions --/
structure NumberWall where
  x : ℤ
  y : ℤ
  -- Define the wall structure based on given conditions
  bottom_row : Vector ℤ 5 := ⟨[x, 7, y, 14, 9], rfl⟩
  second_row_right : Vector ℤ 2 := ⟨[y + 14, 23], rfl⟩
  third_row_right : ℤ := 37
  top : ℤ := 80

/-- The main theorem stating that x must be 12 in the given number wall --/
theorem number_wall_x_value (wall : NumberWall) : wall.x = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_x_value_l845_84544


namespace NUMINAMATH_CALUDE_quadratic_root_property_l845_84595

theorem quadratic_root_property (m : ℝ) : 
  m^2 - 4*m + 1 = 0 → 2023 - m^2 + 4*m = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l845_84595


namespace NUMINAMATH_CALUDE_existence_of_non_divisible_a_l845_84517

theorem existence_of_non_divisible_a (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧
    ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_divisible_a_l845_84517


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_1_l845_84566

theorem binomial_coefficient_n_1 (n : ℕ+) : (n.val : ℕ).choose 1 = n.val := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_1_l845_84566


namespace NUMINAMATH_CALUDE_rachel_colored_pictures_l845_84511

/-- The number of pictures Rachel has colored -/
def pictures_colored (book1_pictures book2_pictures remaining_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - remaining_pictures

theorem rachel_colored_pictures :
  pictures_colored 23 32 11 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rachel_colored_pictures_l845_84511


namespace NUMINAMATH_CALUDE_pencil_length_l845_84529

/-- The length of the purple section of the pencil in centimeters -/
def purple_length : ℝ := 3.5

/-- The length of the black section of the pencil in centimeters -/
def black_length : ℝ := 2.8

/-- The length of the blue section of the pencil in centimeters -/
def blue_length : ℝ := 1.6

/-- The length of the green section of the pencil in centimeters -/
def green_length : ℝ := 0.9

/-- The length of the yellow section of the pencil in centimeters -/
def yellow_length : ℝ := 1.2

/-- The total length of the pencil is the sum of all colored sections -/
theorem pencil_length : 
  purple_length + black_length + blue_length + green_length + yellow_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l845_84529


namespace NUMINAMATH_CALUDE_inverse_inequality_l845_84533

theorem inverse_inequality (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l845_84533


namespace NUMINAMATH_CALUDE_minimum_value_problem_l845_84575

theorem minimum_value_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧
  (∀ (c d : ℝ), c ≠ 0 → d ≠ 0 →
    c^2 + d^2 + 2 / c^2 + d / c + 1 / d^2 ≥ x^2 + y^2 + 2 / x^2 + y / x + 1 / y^2) ∧
  x^2 + y^2 + 2 / x^2 + y / x + 1 / y^2 = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l845_84575


namespace NUMINAMATH_CALUDE_smallest_n_for_seven_numbers_l845_84520

/-- Represents the sequence generation process -/
def generateSequence (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a number is an even square -/
def isEvenSquare (n : ℕ) : Bool :=
  sorry

/-- Finds the largest even square less than or equal to n -/
def largestEvenSquare (n : ℕ) : ℕ :=
  sorry

theorem smallest_n_for_seven_numbers : 
  (∀ m : ℕ, m < 168 → (generateSequence m).length ≠ 7) ∧ 
  (generateSequence 168).length = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_seven_numbers_l845_84520


namespace NUMINAMATH_CALUDE_pi_estimate_l845_84512

theorem pi_estimate (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let p := m / n
  let estimate := (4 * p + 2) / 1
  estimate = 78 / 25 := by
sorry

end NUMINAMATH_CALUDE_pi_estimate_l845_84512


namespace NUMINAMATH_CALUDE_work_completion_time_l845_84570

/-- Given workers a and b, where b completes a work in 7 days, and both a and b
    together complete the work in 4.117647058823529 days, prove that a can
    complete the work alone in 10 days. -/
theorem work_completion_time
  (total_work : ℝ)
  (rate_b : ℝ)
  (rate_combined : ℝ)
  (h1 : rate_b = total_work / 7)
  (h2 : rate_combined = total_work / 4.117647058823529)
  (h3 : rate_combined = rate_b + total_work / 10) :
  ∃ (days_a : ℝ), days_a = 10 ∧ total_work / days_a = total_work / 10 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l845_84570


namespace NUMINAMATH_CALUDE_m_range_proof_l845_84519

/-- Proposition p: The equation x^2+mx+1=0 has two distinct negative roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- Proposition q: The domain of the function f(x)=log_2(4x^2+4(m-2)x+1) is ℝ -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

/-- The range of m given the conditions -/
def m_range : Set ℝ := {m : ℝ | m ≥ 3 ∨ (1 < m ∧ m ≤ 2)}

theorem m_range_proof (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_m_range_proof_l845_84519


namespace NUMINAMATH_CALUDE_systematic_sampling_l845_84557

theorem systematic_sampling (total_products : Nat) (num_samples : Nat) (sampled_second : Nat) : 
  total_products = 100 → 
  num_samples = 5 → 
  sampled_second = 24 → 
  ∃ (interval : Nat) (position : Nat),
    interval = total_products / num_samples ∧
    position = sampled_second % interval ∧
    (position + 3 * interval = 64) :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l845_84557


namespace NUMINAMATH_CALUDE_engineer_is_smith_l845_84559

-- Define the cities
inductive City
| Sheffield
| Leeds
| Halfway

-- Define the occupations
inductive Occupation
| Businessman
| Conductor
| Stoker
| Engineer

-- Define the people
structure Person where
  name : String
  occupation : Occupation
  city : City

-- Define the problem setup
def setup : Prop := ∃ (smith robinson jones : Person) 
  (conductor stoker engineer : Person),
  -- Businessmen
  smith.occupation = Occupation.Businessman ∧
  robinson.occupation = Occupation.Businessman ∧
  jones.occupation = Occupation.Businessman ∧
  -- Railroad workers
  conductor.occupation = Occupation.Conductor ∧
  stoker.occupation = Occupation.Stoker ∧
  engineer.occupation = Occupation.Engineer ∧
  -- Locations
  robinson.city = City.Sheffield ∧
  conductor.city = City.Sheffield ∧
  jones.city = City.Leeds ∧
  stoker.city = City.Leeds ∧
  smith.city = City.Halfway ∧
  engineer.city = City.Halfway ∧
  -- Salary relations
  ∃ (conductor_namesake : Person),
    conductor_namesake.name = conductor.name ∧
    conductor_namesake.occupation = Occupation.Businessman ∧
  -- Billiards game
  (∃ (smith_worker : Person),
    smith_worker.name = "Smith" ∧
    smith_worker.occupation ≠ Occupation.Businessman ∧
    smith_worker ≠ stoker) ∧
  -- Engineer's salary relation
  ∃ (closest_businessman : Person),
    closest_businessman.occupation = Occupation.Businessman ∧
    closest_businessman.city = City.Halfway

-- The theorem to prove
theorem engineer_is_smith (h : setup) : 
  ∃ (engineer : Person), engineer.occupation = Occupation.Engineer ∧ 
  engineer.name = "Smith" := by
  sorry

end NUMINAMATH_CALUDE_engineer_is_smith_l845_84559


namespace NUMINAMATH_CALUDE_constant_ratio_sum_theorem_l845_84565

theorem constant_ratio_sum_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (h_not_all_equal : ¬(x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄))
  (h_constant_ratio : ∃ k : ℝ, 
    (x₁ + x₂) / (x₃ + x₄) = k ∧
    (x₁ + x₃) / (x₂ + x₄) = k ∧
    (x₁ + x₄) / (x₂ + x₃) = k) :
  (∃ k : ℝ, k = -1 ∧ 
    (x₁ + x₂) / (x₃ + x₄) = k ∧
    (x₁ + x₃) / (x₂ + x₄) = k ∧
    (x₁ + x₄) / (x₂ + x₃) = k) ∧
  x₁ + x₂ + x₃ + x₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_sum_theorem_l845_84565


namespace NUMINAMATH_CALUDE_ninety_six_configurations_l845_84523

/-- Represents a configuration of numbers in the grid -/
def Configuration := Fin 6 → Fin 6

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 6) : Prop :=
  sorry

/-- Checks if a configuration is valid according to the rules -/
def valid_configuration (c : Configuration) : Prop :=
  ∀ p1 p2 : Fin 6, adjacent p1 p2 → abs (c p1 - c p2) ≠ 3

/-- The total number of valid configurations -/
def total_valid_configurations : ℕ :=
  sorry

/-- Main theorem: There are 96 valid configurations -/
theorem ninety_six_configurations : total_valid_configurations = 96 :=
  sorry

end NUMINAMATH_CALUDE_ninety_six_configurations_l845_84523


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l845_84590

theorem largest_n_for_equation : 
  (∃ (n : ℕ+), ∀ (m : ℕ+), 
    (∃ (x y z : ℕ+), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 8) → 
    m ≤ n) ∧ 
  (∃ (x y z : ℕ+), (10 : ℕ+)^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 8) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l845_84590


namespace NUMINAMATH_CALUDE_traffic_light_probability_theorem_l845_84592

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) : ℕ :=
  3 * 5 -- 5 seconds before each color change

/-- Calculates the probability of observing a color change -/
def probabilityOfColorChange (cycle : TrafficLightCycle) (observationInterval : ℕ) : ℚ :=
  (changeObservationWindow cycle : ℚ) / (cycleDuration cycle : ℚ)

theorem traffic_light_probability_theorem (cycle : TrafficLightCycle) 
    (h1 : cycle.green = 50)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 40)
    (h4 : observationInterval = 5) :
    probabilityOfColorChange cycle observationInterval = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_probability_theorem_l845_84592


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l845_84514

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 
  1 / x + 1 / y = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l845_84514


namespace NUMINAMATH_CALUDE_computer_table_price_l845_84599

/-- The selling price of an item given its cost price and markup percentage -/
def selling_price (cost : ℚ) (markup : ℚ) : ℚ :=
  cost * (1 + markup / 100)

/-- Theorem: The selling price of a computer table with cost price 6925 and 24% markup is 8587 -/
theorem computer_table_price : selling_price 6925 24 = 8587 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l845_84599


namespace NUMINAMATH_CALUDE_num_paths_correct_l845_84509

/-- The number of paths from (0,0) to (m,n) on Z^2 using only steps of +(1,0) or +(0,1) -/
def num_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that num_paths gives the correct number of paths -/
theorem num_paths_correct (m n : ℕ) : 
  num_paths m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_num_paths_correct_l845_84509


namespace NUMINAMATH_CALUDE_finite_primes_imply_equal_bases_l845_84584

def divides_set (a b c d : ℕ+) : Set ℕ :=
  {p : ℕ | ∃ n : ℕ, n > 0 ∧ p.Prime ∧ p ∣ (a * b^n + c * d^n)}

theorem finite_primes_imply_equal_bases (a b c d : ℕ+) :
  (Set.Finite (divides_set a b c d)) → b = d := by
  sorry

end NUMINAMATH_CALUDE_finite_primes_imply_equal_bases_l845_84584
