import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_identity_l2315_231538

theorem trigonometric_identity (α : Real) :
  (Real.sin (6 * α) + Real.sin (7 * α) + Real.sin (8 * α) + Real.sin (9 * α)) /
  (Real.cos (6 * α) + Real.cos (7 * α) + Real.cos (8 * α) + Real.cos (9 * α)) =
  Real.tan ((15 * α) / 2) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2315_231538


namespace NUMINAMATH_CALUDE_prob_same_gender_specific_schools_l2315_231521

/-- Represents a school with a certain number of male and female teachers -/
structure School :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- The probability of selecting two teachers of the same gender from two schools -/
def prob_same_gender (school_a school_b : School) : ℚ :=
  let total_combinations := school_a.male_count * school_b.male_count + 
                            school_a.female_count * school_b.female_count
  let total_selections := (school_a.male_count + school_a.female_count) * 
                          (school_b.male_count + school_b.female_count)
  total_combinations / total_selections

theorem prob_same_gender_specific_schools :
  let school_a : School := ⟨2, 1⟩
  let school_b : School := ⟨1, 2⟩
  prob_same_gender school_a school_b = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_gender_specific_schools_l2315_231521


namespace NUMINAMATH_CALUDE_gcd_59_power_l2315_231526

theorem gcd_59_power : Nat.gcd (59^7 + 1) (59^7 + 59^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_59_power_l2315_231526


namespace NUMINAMATH_CALUDE_smallest_twin_egg_number_l2315_231542

/-- Definition of a "twin egg number" -/
def is_twin_egg (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧
  ((n / 100) % 10 = (n / 10) % 10)

/-- Function to swap digits as described -/
def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  b * 1000 + a * 100 + d * 10 + c

/-- The F function as defined in the problem -/
def F (m : ℕ) : ℤ := (m - swap_digits m) / 11

/-- Main theorem statement -/
theorem smallest_twin_egg_number :
  ∀ m : ℕ,
  is_twin_egg m →
  (m / 1000 ≠ (m / 100) % 10) →
  ∃ k : ℕ, F m / 27 = k * k →
  4114 ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_twin_egg_number_l2315_231542


namespace NUMINAMATH_CALUDE_matthew_baking_time_l2315_231533

/-- The time it takes Matthew to make caramel-apple coffee cakes when his oven malfunctions -/
def baking_time (assembly_time bake_time_normal decorate_time : ℝ) : ℝ :=
  assembly_time + 2 * bake_time_normal + decorate_time

/-- Theorem stating that Matthew's total baking time is 5 hours when his oven malfunctions -/
theorem matthew_baking_time :
  baking_time 1 1.5 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_matthew_baking_time_l2315_231533


namespace NUMINAMATH_CALUDE_no_special_polyhedron_l2315_231599

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  four_faces_share_edges : Bool

/-- Theorem stating that there does not exist a convex polyhedron with the specified properties. -/
theorem no_special_polyhedron :
  ¬ ∃ (p : ConvexPolyhedron), 
    p.vertices = 8 ∧ 
    p.edges = 12 ∧ 
    p.faces = 6 ∧ 
    p.four_faces_share_edges = true :=
by
  sorry

end NUMINAMATH_CALUDE_no_special_polyhedron_l2315_231599


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2315_231525

theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) : 
  (n / 2 : ℝ) * (2 * a + (n - 1 : ℝ) * 2 * d) = 24 →
  (n / 2 : ℝ) * (2 * (a + d) + (n - 1 : ℝ) * 2 * d) = 30 →
  a + ((2 * n - 1 : ℝ) * d) - a = 10.5 →
  2 * n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2315_231525


namespace NUMINAMATH_CALUDE_cricket_team_size_l2315_231556

/-- The number of players on a cricket team satisfying certain conditions -/
theorem cricket_team_size :
  ∀ (total_players throwers right_handed : ℕ),
    throwers = 37 →
    right_handed = 57 →
    3 * (right_handed - throwers) = 2 * (total_players - throwers) →
    total_players = 67 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2315_231556


namespace NUMINAMATH_CALUDE_intersection_M_N_l2315_231548

def M : Set ℝ := {x | 2*x - x^2 ≥ 0}
def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 \ {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2315_231548


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l2315_231586

theorem min_value_trigonometric_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  1 / (Real.sin θ)^2 + 9 / (Real.cos θ)^2 ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l2315_231586


namespace NUMINAMATH_CALUDE_larger_triangle_equilateral_iff_l2315_231570

/-- Two identical right-angled triangles with angles α and β form a larger triangle when placed together with identical legs adjacent. -/
structure TrianglePair where
  α : Real
  β : Real
  right_angled : α + β = 90
  non_negative : 0 ≤ α ∧ 0 ≤ β

/-- The larger triangle formed by combining two identical right-angled triangles. -/
structure LargerTriangle where
  pair : TrianglePair
  side_a : Real
  side_b : Real
  side_c : Real
  angle_A : Real
  angle_B : Real
  angle_C : Real

/-- The larger triangle is equilateral if and only if the original right-angled triangles have α = 60° and β = 30°. -/
theorem larger_triangle_equilateral_iff (t : LargerTriangle) :
  (t.side_a = t.side_b ∧ t.side_b = t.side_c) ↔ (t.pair.α = 60 ∧ t.pair.β = 30) :=
sorry

end NUMINAMATH_CALUDE_larger_triangle_equilateral_iff_l2315_231570


namespace NUMINAMATH_CALUDE_gina_coin_value_l2315_231535

/-- Calculates the total value of a pile of coins given the total number of coins and the number of dimes. -/
def total_coin_value (total_coins : ℕ) (num_dimes : ℕ) : ℚ :=
  let num_nickels : ℕ := total_coins - num_dimes
  let dime_value : ℚ := 10 / 100
  let nickel_value : ℚ := 5 / 100
  (num_dimes : ℚ) * dime_value + (num_nickels : ℚ) * nickel_value

/-- Proves that given 50 total coins with 14 dimes, the total value is $3.20. -/
theorem gina_coin_value : total_coin_value 50 14 = 32 / 10 := by
  sorry

end NUMINAMATH_CALUDE_gina_coin_value_l2315_231535


namespace NUMINAMATH_CALUDE_problem_statement_l2315_231592

theorem problem_statement : 
  (∀ x y : ℝ, (Real.sqrt x + Real.sqrt y = 0) → (x = 0 ∧ y = 0)) ∨
  (∀ x : ℝ, (x^2 + 4*x - 5 = 0) → (x = -5)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2315_231592


namespace NUMINAMATH_CALUDE_marnie_chips_per_day_l2315_231587

/-- Calculates the number of chips Marnie eats each day starting from the second day -/
def chips_per_day (total_chips : ℕ) (first_day_chips : ℕ) (total_days : ℕ) : ℕ :=
  (total_chips - first_day_chips) / (total_days - 1)

/-- Theorem stating that Marnie eats 10 chips per day starting from the second day -/
theorem marnie_chips_per_day :
  chips_per_day 100 10 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_marnie_chips_per_day_l2315_231587


namespace NUMINAMATH_CALUDE_remainder_prime_divisible_by_210_l2315_231531

theorem remainder_prime_divisible_by_210 (p r : ℕ) : 
  Prime p → 
  r = p % 210 → 
  0 < r → 
  r < 210 → 
  ¬ Prime r → 
  (∃ (a b : ℕ), r = a^2 + b^2) → 
  r = 169 := by sorry

end NUMINAMATH_CALUDE_remainder_prime_divisible_by_210_l2315_231531


namespace NUMINAMATH_CALUDE_rotation_surface_area_theorem_l2315_231588

/-- Represents a plane curve -/
structure PlaneCurve where
  -- Add necessary fields for a plane curve

/-- Calculates the length of a plane curve -/
def curveLength (c : PlaneCurve) : ℝ :=
  sorry

/-- Calculates the distance of the center of gravity from the axis of rotation -/
def centerOfGravityDistance (c : PlaneCurve) : ℝ :=
  sorry

/-- Calculates the surface area generated by rotating a plane curve around an axis -/
def rotationSurfaceArea (c : PlaneCurve) : ℝ :=
  sorry

/-- Theorem: The surface area generated by rotating an arbitrary plane curve around an axis
    is equal to 2π times the distance of the center of gravity from the axis
    times the length of the curve -/
theorem rotation_surface_area_theorem (c : PlaneCurve) :
  rotationSurfaceArea c = 2 * Real.pi * centerOfGravityDistance c * curveLength c :=
sorry

end NUMINAMATH_CALUDE_rotation_surface_area_theorem_l2315_231588


namespace NUMINAMATH_CALUDE_abs_value_problem_l2315_231522

theorem abs_value_problem (x p : ℝ) : 
  |x - 3| = p ∧ x > 3 → x - p = 3 := by
sorry

end NUMINAMATH_CALUDE_abs_value_problem_l2315_231522


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2315_231549

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 + 8 * i) / (5 - 4 * i) = (-2 : ℚ) / 41 + (64 : ℚ) / 41 * i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2315_231549


namespace NUMINAMATH_CALUDE_alloy_density_proof_l2315_231540

/-- The specific gravity of gold relative to water -/
def gold_specific_gravity : ℝ := 19

/-- The specific gravity of copper relative to water -/
def copper_specific_gravity : ℝ := 9

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 4

/-- The specific gravity of the resulting alloy -/
def alloy_specific_gravity : ℝ := 17

/-- Theorem stating that mixing gold and copper in the given ratio results in the specified alloy density -/
theorem alloy_density_proof :
  (gold_copper_ratio * gold_specific_gravity + copper_specific_gravity) / (gold_copper_ratio + 1) = alloy_specific_gravity :=
by sorry

end NUMINAMATH_CALUDE_alloy_density_proof_l2315_231540


namespace NUMINAMATH_CALUDE_g_range_l2315_231508

def g (x : ℝ) := x^2 - 2*x

theorem g_range :
  ∀ x ∈ Set.Icc 0 3, -1 ≤ g x ∧ g x ≤ 3 ∧
  (∃ x₁ ∈ Set.Icc 0 3, g x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc 0 3, g x₂ = 3) :=
sorry

end NUMINAMATH_CALUDE_g_range_l2315_231508


namespace NUMINAMATH_CALUDE_problem_solution_l2315_231581

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3*x - 8
def h (r : ℝ) (x : ℝ) : ℝ := 3*x - r

theorem problem_solution :
  (f 2 = 4 ∧ g (f 2) = 4) ∧
  (∀ x : ℝ, f (g x) = g (f x) ↔ x = 2 ∨ x = 6) ∧
  (∀ r : ℝ, f (h r 2) = h r (f 2) ↔ r = 3 ∨ r = 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2315_231581


namespace NUMINAMATH_CALUDE_units_digit_63_plus_74_base9_l2315_231558

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (a b : ℕ) : ℕ := a * 9 + b

/-- Calculates the units digit of a number in base 9 -/
def unitsDigitBase9 (n : ℕ) : ℕ := n % 9

theorem units_digit_63_plus_74_base9 :
  unitsDigitBase9 (base9ToBase10 6 3 + base9ToBase10 7 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_63_plus_74_base9_l2315_231558


namespace NUMINAMATH_CALUDE_jack_school_time_l2315_231516

/-- Given information about Dave and Jack's walking speeds and Dave's time to school,
    prove that Jack takes 18 minutes to reach the same school. -/
theorem jack_school_time (dave_steps_per_min : ℕ) (dave_step_length : ℕ) (dave_time : ℕ)
                         (jack_steps_per_min : ℕ) (jack_step_length : ℕ) :
  dave_steps_per_min = 90 →
  dave_step_length = 75 →
  dave_time = 16 →
  jack_steps_per_min = 100 →
  jack_step_length = 60 →
  (dave_steps_per_min * dave_step_length * dave_time) / (jack_steps_per_min * jack_step_length) = 18 :=
by sorry

end NUMINAMATH_CALUDE_jack_school_time_l2315_231516


namespace NUMINAMATH_CALUDE_line_properties_l2315_231568

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def y_coord (l : Line3D) (x : ℝ) : ℝ := sorry

/-- The intersection point of the line with the z=0 plane -/
def z_plane_intersection (l : Line3D) : ℝ × ℝ × ℝ := sorry

theorem line_properties (l : Line3D) 
  (h1 : l.point1 = (1, 3, 2)) 
  (h2 : l.point2 = (4, 3, -1)) : 
  y_coord l 7 = 3 ∧ z_plane_intersection l = (3, 3, 0) := by sorry

end NUMINAMATH_CALUDE_line_properties_l2315_231568


namespace NUMINAMATH_CALUDE_mindy_income_multiplier_l2315_231504

/-- Given tax rates and combined rate, prove Mindy's income multiplier --/
theorem mindy_income_multiplier 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (combined_rate : ℝ) 
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.25)
  (h3 : combined_rate = 0.29) :
  ∃ k : ℝ, k = 4 ∧ 
    (mork_rate + mindy_rate * k) / (1 + k) = combined_rate :=
by sorry

end NUMINAMATH_CALUDE_mindy_income_multiplier_l2315_231504


namespace NUMINAMATH_CALUDE_simplify_expression_l2315_231595

theorem simplify_expression (x : ℝ) : (3 * x + 25) + (200 * x - 50) = 203 * x - 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2315_231595


namespace NUMINAMATH_CALUDE_integer_difference_l2315_231519

theorem integer_difference (x y : ℤ) (h1 : x < y) (h2 : x + y = -9) (h3 : x = -5) (h4 : y = -4) : y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_difference_l2315_231519


namespace NUMINAMATH_CALUDE_interest_rate_proof_l2315_231579

/-- 
Given a principal sum and an annual interest rate,
if the simple interest for 4 years is one-fifth of the principal,
then the annual interest rate is 5%.
-/
theorem interest_rate_proof (P R : ℝ) (P_pos : P > 0) : 
  (P * R * 4) / 100 = P / 5 → R = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l2315_231579


namespace NUMINAMATH_CALUDE_function_property_l2315_231585

-- Define the function f
variable (f : ℝ → ℝ)
-- Define the point a
variable (a : ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x < y → x < a → y < a → f x < f y)
variable (h2 : ∀ x, f (x + a) = f (a - x))
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ < a ∧ a < x₂)
variable (h4 : |x₁ - a| < |x₂ - a|)

-- State the theorem
theorem function_property : f (2*a - x₁) > f (2*a - x₂) := by sorry

end NUMINAMATH_CALUDE_function_property_l2315_231585


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2315_231561

theorem sum_of_three_numbers (a b c : ℕ) : 
  a = 200 → 
  b = 2 * c → 
  c = 100 → 
  a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2315_231561


namespace NUMINAMATH_CALUDE_parking_lot_ratio_l2315_231544

/-- Proves that the ratio of full-sized car spaces to compact car spaces is 11:4 
    given the total number of spaces and the number of full-sized car spaces. -/
theorem parking_lot_ratio (total_spaces full_sized_spaces : ℕ) 
  (h1 : total_spaces = 450)
  (h2 : full_sized_spaces = 330) :
  (full_sized_spaces : ℚ) / (total_spaces - full_sized_spaces : ℚ) = 11 / 4 := by
  sorry

#check parking_lot_ratio

end NUMINAMATH_CALUDE_parking_lot_ratio_l2315_231544


namespace NUMINAMATH_CALUDE_problem_statement_l2315_231597

theorem problem_statement :
  (∃ x₀ : ℝ, Real.tan x₀ = 2) ∧ ¬(∀ x : ℝ, x^2 + 2*x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2315_231597


namespace NUMINAMATH_CALUDE_profit_percent_l2315_231517

theorem profit_percent (P : ℝ) (C : ℝ) (h : P > 0) (h2 : C > 0) :
  (2/3 * P = 0.84 * C) → (P - C) / C * 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_l2315_231517


namespace NUMINAMATH_CALUDE_K_bounds_l2315_231552

/-- The number of triples in a given system for a natural number n -/
noncomputable def K (n : ℕ) : ℝ := sorry

/-- Theorem stating the bounds for K(n) -/
theorem K_bounds (n : ℕ) : n / 6 - 1 < K n ∧ K n < 2 * n / 9 := by sorry

end NUMINAMATH_CALUDE_K_bounds_l2315_231552


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2315_231555

theorem trigonometric_identity (θ c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h : Real.sin θ ^ 6 / c + Real.cos θ ^ 6 / d = 1 / (c + d)) :
  Real.sin θ ^ 18 / c^5 + Real.cos θ ^ 18 / d^5 = (c^4 + d^4) / (c + d)^9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2315_231555


namespace NUMINAMATH_CALUDE_half_percent_of_160_l2315_231528

theorem half_percent_of_160 : (1 / 2 * 1 / 100) * 160 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_of_160_l2315_231528


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l2315_231584

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l2315_231584


namespace NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l2315_231559

theorem cartesian_to_polar_conversion (x y ρ θ : ℝ) :
  x = -1 ∧ y = Real.sqrt 3 →
  ρ = 2 ∧ θ = 2 * Real.pi / 3 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ^2 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l2315_231559


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2315_231546

def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -6)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ vector_a = k • vector_b x) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2315_231546


namespace NUMINAMATH_CALUDE_room_length_l2315_231524

/-- The length of a room satisfying given conditions -/
theorem room_length : ∃ (L : ℝ), 
  (L > 0) ∧ 
  (9 * (2 * 12 * (L + 15) - (6 * 3 + 3 * 4 * 3)) = 8154) → 
  L = 25 := by
  sorry

end NUMINAMATH_CALUDE_room_length_l2315_231524


namespace NUMINAMATH_CALUDE_fraction_product_equality_l2315_231530

theorem fraction_product_equality : (1 / 3 : ℚ)^4 * (1 / 8 : ℚ) = 1 / 648 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l2315_231530


namespace NUMINAMATH_CALUDE_probability_calculation_l2315_231565

/-- The number of volunteers -/
def num_volunteers : ℕ := 5

/-- The number of venues -/
def num_venues : ℕ := 3

/-- The total number of ways to assign volunteers to venues -/
def total_assignments : ℕ := num_venues ^ num_volunteers

/-- The number of favorable assignments (where each venue has at least one volunteer) -/
def favorable_assignments : ℕ := 150

/-- The probability that each venue has at least one volunteer -/
def probability_all_venues_covered : ℚ := favorable_assignments / total_assignments

theorem probability_calculation :
  probability_all_venues_covered = 50 / 81 :=
sorry

end NUMINAMATH_CALUDE_probability_calculation_l2315_231565


namespace NUMINAMATH_CALUDE_base4_division_l2315_231550

-- Define a function to convert from base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (4^i)) 0

-- Define the numbers in base 4
def num2013Base4 : List Nat := [3, 1, 0, 2]
def num13Base4 : List Nat := [3, 1]
def result13Base4 : List Nat := [3, 1]

-- State the theorem
theorem base4_division :
  (base4ToDecimal num2013Base4) / (base4ToDecimal num13Base4) = base4ToDecimal result13Base4 :=
sorry

end NUMINAMATH_CALUDE_base4_division_l2315_231550


namespace NUMINAMATH_CALUDE_vectors_not_collinear_l2315_231571

/-- Given vectors a and b in ℝ³, prove that c₁ and c₂ are not collinear -/
theorem vectors_not_collinear (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, -2, 3))
  (hb : b = (3, 0, -1)) : 
  ¬ (∃ (k : ℝ), (2 • a + 4 • b) = k • (3 • b - a)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_collinear_l2315_231571


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2315_231557

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 138) (h2 : b = 85) (h3 : c = 130) (h4 : d = 120) (h5 : e = 95) :
  720 - (a + b + c + d + e) = 152 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2315_231557


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l2315_231523

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x^3 - 24*x^2 + 151*x - 650 = (x - p)*(x - q)*(x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 251 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l2315_231523


namespace NUMINAMATH_CALUDE_present_worth_calculation_l2315_231563

/-- Calculates the present worth of a sum given the banker's gain, time period, and interest rate -/
def present_worth (bankers_gain : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let simple_interest := (bankers_gain * rate * (100 + rate * time)).sqrt
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that the present worth is 7755 given the specified conditions -/
theorem present_worth_calculation :
  present_worth 24 2 (10/100) = 7755 := by
  sorry

end NUMINAMATH_CALUDE_present_worth_calculation_l2315_231563


namespace NUMINAMATH_CALUDE_middle_truncated_cone_volume_middle_truncated_cone_volume_is_7V_div_27_l2315_231582

/-- Given a cone with volume V whose height is divided into three equal parts by planes parallel to the base, the volume of the middle truncated cone is 7V/27. -/
theorem middle_truncated_cone_volume (V : ℝ) (h : V > 0) : ℝ :=
  let cone_volume := V
  let height_parts := 3
  let middle_truncated_cone_volume := (7 : ℝ) / 27 * V
  middle_truncated_cone_volume

/-- The volume of the middle truncated cone is 7V/27 -/
theorem middle_truncated_cone_volume_is_7V_div_27 (V : ℝ) (h : V > 0) :
  middle_truncated_cone_volume V h = (7 : ℝ) / 27 * V := by
  sorry

end NUMINAMATH_CALUDE_middle_truncated_cone_volume_middle_truncated_cone_volume_is_7V_div_27_l2315_231582


namespace NUMINAMATH_CALUDE_triangle_area_on_rectangle_l2315_231590

/-- Given a rectangle of 6 units by 8 units and a triangle DEF with vertices
    D(0,2), E(6,0), and F(3,8) located on the boundary of the rectangle,
    prove that the area of triangle DEF is 21 square units. -/
theorem triangle_area_on_rectangle (D E F : ℝ × ℝ) : 
  D = (0, 2) →
  E = (6, 0) →
  F = (3, 8) →
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 8
  let triangle_area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
  triangle_area = 21 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_on_rectangle_l2315_231590


namespace NUMINAMATH_CALUDE_f_even_implies_a_zero_f_not_odd_l2315_231569

/-- Definition of the function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + |x - a| + 1

/-- Theorem 1: If f is even, then a = 0 -/
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by sorry

/-- Theorem 2: f is not odd for any real a -/
theorem f_not_odd (a : ℝ) :
  ¬(∀ x : ℝ, f a (-x) = -(f a x)) := by sorry

end NUMINAMATH_CALUDE_f_even_implies_a_zero_f_not_odd_l2315_231569


namespace NUMINAMATH_CALUDE_g_comp_three_roots_l2315_231598

/-- The function g(x) = x^2 + 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- The statement that g(g(x)) has exactly 3 distinct real roots -/
def has_exactly_three_roots (d : ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x : ℝ, g_comp d x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
                    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

theorem g_comp_three_roots :
  ∀ d : ℝ, has_exactly_three_roots d ↔ d = -20 + 4 * Real.sqrt 14 ∨ d = -20 - 4 * Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_g_comp_three_roots_l2315_231598


namespace NUMINAMATH_CALUDE_function_properties_l2315_231512

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
variable (h : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)

-- Theorem statement
theorem function_properties :
  (f 0 = 0) ∧ (f (-1) = 0) ∧ (∀ x : ℝ, f (-x) = -f x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2315_231512


namespace NUMINAMATH_CALUDE_union_eq_univ_complement_inter_eq_open_interval_range_of_a_l2315_231500

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def B : Set ℝ := {x | -2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statements
theorem union_eq_univ : A ∪ B = Set.univ := by sorry

theorem complement_inter_eq_open_interval :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

theorem range_of_a (h : ∀ a, C a ⊆ B) :
  {a | ∀ x, x ∈ C a → x ∈ B} = Set.Icc (-2) 8 := by sorry

end NUMINAMATH_CALUDE_union_eq_univ_complement_inter_eq_open_interval_range_of_a_l2315_231500


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l2315_231529

/-- The sum of the infinite geometric series 5/3 - 5/8 + 25/128 - 125/1024 + ... -/
def infiniteGeometricSeriesSum : ℚ := 8/3

/-- The first term of the geometric series -/
def firstTerm : ℚ := 5/3

/-- The common ratio of the geometric series -/
def commonRatio : ℚ := 3/8

/-- Theorem stating that the sum of the infinite geometric series is 8/3 -/
theorem infinite_geometric_series_sum :
  infiniteGeometricSeriesSum = firstTerm / (1 - commonRatio) :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l2315_231529


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2315_231534

theorem hyperbola_m_range (m : ℝ) :
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m ≠ 0 ∧ m + 1 ≠ 0) ∧
   ((2 + m > 0 ∧ m + 1 < 0) ∨ (2 + m < 0 ∧ m + 1 > 0))) →
  m < -2 ∨ m > -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2315_231534


namespace NUMINAMATH_CALUDE_intersection_with_complement_is_empty_l2315_231501

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3}
def B : Set Nat := {1, 3, 4}

theorem intersection_with_complement_is_empty :
  A ∩ (U \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_is_empty_l2315_231501


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2315_231518

def M : Set ℝ := {x | x^2 - x > 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_of_M_and_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2315_231518


namespace NUMINAMATH_CALUDE_two_integers_make_fraction_integer_l2315_231505

theorem two_integers_make_fraction_integer : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, (1750 : ℕ) ∣ (m^2 - 4)) ∧ 
    (∀ m : ℕ, m > 0 → (1750 : ℕ) ∣ (m^2 - 4) → m ∈ S) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_make_fraction_integer_l2315_231505


namespace NUMINAMATH_CALUDE_conspiracy_split_l2315_231577

theorem conspiracy_split (S : Finset (Finset Nat)) :
  S.card = 6 →
  (∀ s ∈ S, s.card = 3) →
  (∃ T : Finset Nat, T ⊆ Finset.range 6 ∧ T.card = 3 ∧
    ∀ s ∈ S, (s ⊆ T → False) ∧ (s ⊆ (Finset.range 6 \ T) → False)) :=
by sorry

end NUMINAMATH_CALUDE_conspiracy_split_l2315_231577


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l2315_231593

theorem modular_congruence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ 24938 ≡ n [ZMOD 25] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l2315_231593


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l2315_231583

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 9}

/-- The number of digits -/
def n : Nat := 6

/-- The condition for divisibility by 15 -/
def divisible_by_15 (num : Nat) : Prop := num % 15 = 0

/-- The set of all possible six-digit numbers formed by the given digits -/
def all_numbers : Finset Nat := sorry

/-- The set of all six-digit numbers formed by the given digits that are divisible by 15 -/
def divisible_numbers : Finset Nat := sorry

/-- The probability of a randomly selected six-digit number being divisible by 15 -/
theorem probability_divisible_by_15 : 
  (Finset.card divisible_numbers : ℚ) / (Finset.card all_numbers : ℚ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l2315_231583


namespace NUMINAMATH_CALUDE_sin_function_properties_l2315_231554

noncomputable def f (x φ A : ℝ) : ℝ := Real.sin (2 * x + φ) + A

theorem sin_function_properties (φ A : ℝ) :
  -- Amplitude is A
  (∃ (x : ℝ), f x φ A - A = 1) ∧
  (∀ (x : ℝ), f x φ A - A ≤ 1) ∧
  -- Period is π
  (∀ (x : ℝ), f (x + π) φ A = f x φ A) ∧
  -- Initial phase is φ
  (∀ (x : ℝ), f x φ A = Real.sin (2 * x + φ) + A) ∧
  -- Maximum value occurs when x = π/4 + kπ, k ∈ ℤ
  (∀ (x : ℝ), f x φ A = A + 1 ↔ ∃ (k : ℤ), x = π/4 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_sin_function_properties_l2315_231554


namespace NUMINAMATH_CALUDE_square_area_decrease_l2315_231510

theorem square_area_decrease (initial_area : ℝ) (side_decrease_percent : ℝ) 
  (h1 : initial_area = 50) 
  (h2 : side_decrease_percent = 20) : 
  let new_area := initial_area * (1 - side_decrease_percent / 100)^2
  (initial_area - new_area) / initial_area * 100 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_area_decrease_l2315_231510


namespace NUMINAMATH_CALUDE_intersection_M_N_l2315_231591

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x : ℕ | x - 1 ≥ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2315_231591


namespace NUMINAMATH_CALUDE_favorite_numbers_exist_l2315_231514

theorem favorite_numbers_exist : ∃ x y : ℕ, x > y ∧ x ≠ y ∧ (x + y) + (x - y) + x * y + (x / y) = 98 := by
  sorry

end NUMINAMATH_CALUDE_favorite_numbers_exist_l2315_231514


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l2315_231536

theorem pasta_preference_ratio (total_students : ℕ) 
  (spaghetti ravioli fettuccine penne : ℕ) : 
  total_students = 800 →
  spaghetti = 300 →
  ravioli = 200 →
  fettuccine = 150 →
  penne = 150 →
  (fettuccine : ℚ) / penne = 1 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l2315_231536


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2315_231566

def A : Set ℝ := {0, 2, 4, 6}
def B : Set ℝ := {x | 3 < x ∧ x < 7}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2315_231566


namespace NUMINAMATH_CALUDE_problem_statement_l2315_231513

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  (a < 1/2 ∧ 1/2 < b) ∧ (a < a^2 + b^2 ∧ a^2 + b^2 < b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2315_231513


namespace NUMINAMATH_CALUDE_car_distance_l2315_231537

/-- Proves that a car traveling 2/3 as fast as a train going 90 miles per hour will cover 40 miles in 40 minutes -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time_minutes : ℝ) : 
  train_speed = 90 →
  car_speed_ratio = 2/3 →
  travel_time_minutes = 40 →
  (car_speed_ratio * train_speed) * (travel_time_minutes / 60) = 40 := by
  sorry

#check car_distance

end NUMINAMATH_CALUDE_car_distance_l2315_231537


namespace NUMINAMATH_CALUDE_shoe_pairs_in_box_l2315_231560

theorem shoe_pairs_in_box (total_shoes : ℕ) (prob_matching : ℚ) : 
  total_shoes = 18 → prob_matching = 1 / 17 → ∃ n : ℕ, n * 2 = total_shoes ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_shoe_pairs_in_box_l2315_231560


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2315_231545

theorem roots_of_polynomial (x : ℝ) : 
  x^2 * (x - 5)^2 * (x + 3) = 0 ↔ x = 0 ∨ x = 5 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2315_231545


namespace NUMINAMATH_CALUDE_knowledge_competition_probabilities_l2315_231567

/-- Represents the outcome of answering a question -/
inductive Answer
| Correct
| Incorrect

/-- Represents the state of a contestant in the competition -/
structure ContestantState where
  score : ℕ
  questions_answered : ℕ

/-- Represents the probabilities of correctly answering each question -/
structure QuestionProbabilities where
  pA : ℚ
  pB : ℚ
  pC : ℚ
  pD : ℚ

/-- Updates the contestant's state based on their answer -/
def updateState (state : ContestantState) (answer : Answer) (questionNumber : ℕ) : ContestantState :=
  match answer with
  | Answer.Correct =>
    let points := match questionNumber with
      | 1 => 1
      | 2 => 2
      | 3 => 3
      | 4 => 6
      | _ => 0
    { score := state.score + points, questions_answered := state.questions_answered + 1 }
  | Answer.Incorrect =>
    { score := state.score - 2, questions_answered := state.questions_answered + 1 }

/-- Checks if a contestant is eliminated based on their current state -/
def isEliminated (state : ContestantState) : Bool :=
  state.score < 8 || (state.questions_answered = 4 && state.score < 14)

/-- Checks if a contestant has advanced to the next round -/
def hasAdvanced (state : ContestantState) : Bool :=
  state.score ≥ 14

/-- Main theorem statement -/
theorem knowledge_competition_probabilities 
  (probs : QuestionProbabilities)
  (h1 : probs.pA = 3/4)
  (h2 : probs.pB = 1/2)
  (h3 : probs.pC = 1/3)
  (h4 : probs.pD = 1/4) :
  ∃ (advanceProb : ℚ) (ξDist : ℕ → ℚ) (ξExpected : ℚ),
    (advanceProb = 1/2) ∧ 
    (ξDist 2 = 1/8) ∧ (ξDist 3 = 1/2) ∧ (ξDist 4 = 3/8) ∧
    (ξExpected = 7/4) := by
  sorry

end NUMINAMATH_CALUDE_knowledge_competition_probabilities_l2315_231567


namespace NUMINAMATH_CALUDE_min_odd_integers_l2315_231511

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 34)
  (sum2 : a + b + c + d = 51)
  (sum3 : a + b + c + d + e + f = 72) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ odds, Odd x) ∧ 
    odds.card = 2 ∧
    (∀ (odds' : Finset ℤ), odds' ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ odds', Odd x) → odds'.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l2315_231511


namespace NUMINAMATH_CALUDE_product_ends_in_zero_theorem_l2315_231562

def is_valid_assignment (assignment : Char → ℕ) : Prop :=
  (∀ c₁ c₂, c₁ ≠ c₂ → assignment c₁ ≠ assignment c₂) ∧
  (∀ c, assignment c < 10)

def satisfies_equation (assignment : Char → ℕ) : Prop :=
  10 * (assignment 'Ж') + (assignment 'Ж') + (assignment 'Ж') =
  100 * (assignment 'М') + 10 * (assignment 'Ё') + (assignment 'Д')

def product_ends_in_zero (assignment : Char → ℕ) : Prop :=
  (assignment 'В' * assignment 'И' * assignment 'H' * assignment 'H' *
   assignment 'U' * assignment 'П' * assignment 'У' * assignment 'X') % 10 = 0

theorem product_ends_in_zero_theorem (assignment : Char → ℕ) :
  is_valid_assignment assignment → satisfies_equation assignment →
  product_ends_in_zero assignment :=
by
  sorry

#check product_ends_in_zero_theorem

end NUMINAMATH_CALUDE_product_ends_in_zero_theorem_l2315_231562


namespace NUMINAMATH_CALUDE_seed_germination_problem_l2315_231594

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 1/5 →
  total_germination_rate = 13/50 →
  (seeds_plot1 * germination_rate_plot1 + seeds_plot2 * (germination_rate_plot2 : ℚ)) / (seeds_plot1 + seeds_plot2) = total_germination_rate →
  (germination_rate_plot2 : ℚ) = 7/20 :=
by
  sorry

#check seed_germination_problem

end NUMINAMATH_CALUDE_seed_germination_problem_l2315_231594


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2315_231572

theorem sqrt_inequality : Real.sqrt 5 - Real.sqrt 6 < Real.sqrt 6 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2315_231572


namespace NUMINAMATH_CALUDE_curve_C_properties_l2315_231576

-- Define the curve C
def C (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1^2 + n * p.2^2 = 1}

-- Define what it means for a curve to be an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Define what it means for a curve to be a hyperbola with given asymptotes
def is_hyperbola_with_asymptotes (S : Set (ℝ × ℝ)) (f : ℝ → ℝ) : Prop :=
  sorry

-- Define what it means for a curve to consist of two straight lines
def is_two_straight_lines (S : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem curve_C_properties (m n : ℝ) :
  (m > n ∧ n > 0 → is_ellipse_with_foci_on_y_axis (C m n)) ∧
  (m * n < 0 → is_hyperbola_with_asymptotes (C m n) (λ x => Real.sqrt (-m/n) * x)) ∧
  (m = 0 ∧ n > 0 → is_two_straight_lines (C m n)) :=
  sorry

end NUMINAMATH_CALUDE_curve_C_properties_l2315_231576


namespace NUMINAMATH_CALUDE_four_line_theorem_l2315_231506

/-- A line in a plane -/
structure Line where
  -- Add necessary fields here
  
/-- A point in a plane -/
structure Point where
  -- Add necessary fields here

/-- A circle in a plane -/
structure Circle where
  -- Add necessary fields here

/-- The set of four lines in the plane -/
def FourLines : Type := Fin 4 → Line

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- Predicate to check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop := sorry

/-- Get the intersection point of two lines -/
def intersection (l1 l2 : Line) : Point := sorry

/-- Get the circumcircle of three points -/
def circumcircle (p1 p2 p3 : Point) : Circle := sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem four_line_theorem (lines : FourLines) :
  (∀ i j, i ≠ j → ¬are_parallel (lines i) (lines j)) →
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬are_concurrent (lines i) (lines j) (lines k)) →
  ∃ p : Point, ∀ i j k l,
    i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    point_on_circle p (circumcircle 
      (intersection (lines i) (lines j))
      (intersection (lines j) (lines k))
      (intersection (lines k) (lines i))) :=
sorry

end NUMINAMATH_CALUDE_four_line_theorem_l2315_231506


namespace NUMINAMATH_CALUDE_jesse_blocks_theorem_l2315_231564

/-- The number of blocks Jesse used to build the building -/
def building_blocks : ℕ := 80

/-- The number of blocks Jesse used to build the farmhouse -/
def farmhouse_blocks : ℕ := 123

/-- The number of blocks Jesse used to build the fenced-in area -/
def fenced_area_blocks : ℕ := 57

/-- The number of blocks Jesse has left -/
def remaining_blocks : ℕ := 84

/-- The total number of blocks Jesse started with -/
def total_blocks : ℕ := building_blocks + farmhouse_blocks + fenced_area_blocks + remaining_blocks

theorem jesse_blocks_theorem : total_blocks = 344 := by
  sorry

end NUMINAMATH_CALUDE_jesse_blocks_theorem_l2315_231564


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l2315_231543

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - a * x) / Real.exp x

theorem extreme_value_and_range :
  (∃ x : ℝ, ∀ y : ℝ, f 1 y ≥ f 1 x ∧ f 1 x = -1 / Real.exp 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x ≥ 1 - 2 * x) ↔ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l2315_231543


namespace NUMINAMATH_CALUDE_unique_root_is_half_l2315_231515

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ b ≥ c ≥ 0,
    and the quadratic equation ax^2 - bx + c = 0 having exactly one root,
    prove that this root is 1/2. -/
theorem unique_root_is_half (a b c : ℝ) 
    (arith_seq : ∃ (d : ℝ), b = a - d ∧ c = a - 2*d)
    (ordered : a ≥ b ∧ b ≥ c ∧ c ≥ 0)
    (one_root : ∃! x, a*x^2 - b*x + c = 0) :
    ∃ x, a*x^2 - b*x + c = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_is_half_l2315_231515


namespace NUMINAMATH_CALUDE_number_of_red_balls_l2315_231551

/-- Given a bag with white and red balls, prove the number of red balls. -/
theorem number_of_red_balls
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (red_frequency : ℚ) 
  (h1 : white_balls = 60)
  (h2 : red_frequency = 1/4)
  (h3 : total_balls = white_balls / (1 - red_frequency)) :
  total_balls - white_balls = 20 :=
by sorry

end NUMINAMATH_CALUDE_number_of_red_balls_l2315_231551


namespace NUMINAMATH_CALUDE_sandy_carrots_l2315_231573

def carrots_problem (initial_carrots : ℕ) (sam_took : ℕ) (sandy_left : ℕ) : Prop :=
  initial_carrots = sam_took + sandy_left

theorem sandy_carrots : ∃ initial_carrots : ℕ, carrots_problem initial_carrots 3 3 :=
  sorry

end NUMINAMATH_CALUDE_sandy_carrots_l2315_231573


namespace NUMINAMATH_CALUDE_smallest_b_value_l2315_231520

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 8) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, a'.val - k.val = 8 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val * k.val) = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2315_231520


namespace NUMINAMATH_CALUDE_farm_distance_problem_l2315_231580

/-- Represents the distances between three farms -/
structure FarmDistances where
  x : ℝ  -- Distance between first and second farms
  y : ℝ  -- Distance between second and third farms
  z : ℝ  -- Distance between first and third farms

/-- Theorem stating the conditions and results for the farm distance problem -/
theorem farm_distance_problem (a : ℝ) : 
  ∃ (d : FarmDistances), 
    d.x + d.y = 4 * d.z ∧                   -- Condition 1
    d.z + d.y = d.x + a ∧                   -- Condition 2
    d.x + d.z = 85 ∧                        -- Condition 3
    0 < a ∧ a < 85 ∧                        -- Interval for a
    d.x = (340 - a) / 6 ∧                   -- Distance x
    d.y = (2 * a + 85) / 3 ∧                -- Distance y
    d.z = (170 + a) / 6 ∧                   -- Distance z
    d.x + d.y > d.z ∧ d.y + d.z > d.x ∧ d.z + d.x > d.y -- Triangle inequality
    := by sorry

end NUMINAMATH_CALUDE_farm_distance_problem_l2315_231580


namespace NUMINAMATH_CALUDE_other_side_formula_l2315_231541

/-- Represents a rectangle with perimeter 30 and one side x -/
structure Rectangle30 where
  x : ℝ
  other : ℝ
  perimeter_eq : x + other = 15

theorem other_side_formula (rect : Rectangle30) : rect.other = 15 - rect.x := by
  sorry

end NUMINAMATH_CALUDE_other_side_formula_l2315_231541


namespace NUMINAMATH_CALUDE_expression_change_l2315_231532

theorem expression_change (x b : ℝ) (hb : b > 0) : 
  let f := fun t => t^3 - 2*t + 1
  (f (x + b) - f x = 3*b*x^2 + 3*b^2*x + b^3 - 2*b) ∧ 
  (f (x - b) - f x = -3*b*x^2 + 3*b^2*x - b^3 + 2*b) := by
  sorry

end NUMINAMATH_CALUDE_expression_change_l2315_231532


namespace NUMINAMATH_CALUDE_oranges_minus_apples_difference_l2315_231596

/-- The number of apples Leif has -/
def num_apples : ℕ := 14

/-- The number of dozens of oranges Leif has -/
def dozens_oranges : ℕ := 2

/-- The number of fruits in a dozen -/
def fruits_per_dozen : ℕ := 12

/-- Calculates the total number of oranges -/
def total_oranges : ℕ := dozens_oranges * fruits_per_dozen

/-- Theorem stating the difference between oranges and apples -/
theorem oranges_minus_apples_difference : 
  total_oranges - num_apples = 10 := by sorry

end NUMINAMATH_CALUDE_oranges_minus_apples_difference_l2315_231596


namespace NUMINAMATH_CALUDE_natural_number_equation_solutions_l2315_231574

theorem natural_number_equation_solutions :
  ∀ a b : ℕ,
  a^b + b^a = 10 * b^(a-2) + 100 ↔ (a = 109 ∧ b = 1) ∨ (a = 7 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_equation_solutions_l2315_231574


namespace NUMINAMATH_CALUDE_travel_options_l2315_231547

/-- The number of train departures from City A to City B -/
def train_departures : ℕ := 10

/-- The number of flights from City A to City B -/
def flights : ℕ := 2

/-- The number of long-distance bus services from City A to City B -/
def bus_services : ℕ := 12

/-- The total number of ways Xiao Zhang can travel from City A to City B -/
def total_ways : ℕ := train_departures + flights + bus_services

theorem travel_options : total_ways = 24 := by sorry

end NUMINAMATH_CALUDE_travel_options_l2315_231547


namespace NUMINAMATH_CALUDE_reciprocal_of_complex_l2315_231527

/-- The reciprocal of the complex number -3 + 4i is -0.12 - 0.16i -/
theorem reciprocal_of_complex (G : ℂ) : 
  G = -3 + 4*I → 1 / G = -0.12 - 0.16*I := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_complex_l2315_231527


namespace NUMINAMATH_CALUDE_B_is_smallest_l2315_231553

def A : ℤ := 32 + 7
def B : ℤ := 3 * 10 + 3
def C : ℤ := 50 - 9

theorem B_is_smallest : B ≤ A ∧ B ≤ C := by
  sorry

end NUMINAMATH_CALUDE_B_is_smallest_l2315_231553


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2315_231578

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(528 * m ≡ 1068 * m [MOD 30])) ∧ 
  (528 * n ≡ 1068 * n [MOD 30]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2315_231578


namespace NUMINAMATH_CALUDE_total_cars_theorem_l2315_231507

/-- Calculates the total number of cars at the end of the play -/
def total_cars_at_end (front_cars : ℕ) (back_multiplier : ℕ) (additional_cars : ℕ) : ℕ :=
  front_cars + (back_multiplier * front_cars) + additional_cars

/-- Theorem: Given the initial conditions, the total number of cars at the end of the play is 600 -/
theorem total_cars_theorem : total_cars_at_end 100 2 300 = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_theorem_l2315_231507


namespace NUMINAMATH_CALUDE_production_growth_equation_l2315_231589

/-- Represents the production growth scenario of Dream Enterprise --/
def production_growth_scenario (initial_value : ℝ) (growth_rate : ℝ) : Prop :=
  let feb_value := initial_value * (1 + growth_rate)
  let mar_value := initial_value * (1 + growth_rate)^2
  mar_value - feb_value = 220000

/-- Theorem stating the correct equation for the production growth scenario --/
theorem production_growth_equation :
  production_growth_scenario 2000000 x ↔ 2000000 * (1 + x)^2 - 2000000 * (1 + x) = 220000 :=
sorry

end NUMINAMATH_CALUDE_production_growth_equation_l2315_231589


namespace NUMINAMATH_CALUDE_age_difference_is_four_l2315_231502

/-- The age difference between Angelina and Justin -/
def ageDifference (angelinaFutureAge : ℕ) (justinCurrentAge : ℕ) : ℕ :=
  (angelinaFutureAge - 5) - justinCurrentAge

/-- Theorem stating that the age difference between Angelina and Justin is 4 years -/
theorem age_difference_is_four :
  ageDifference 40 31 = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_four_l2315_231502


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2315_231575

theorem triangle_angle_sum (a b c : ℝ) (h1 : a + b + c = 180) 
                           (h2 : a = 85) (h3 : b = 35) : c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2315_231575


namespace NUMINAMATH_CALUDE_negative_forty_divided_by_five_l2315_231539

theorem negative_forty_divided_by_five : (-40 : ℤ) / 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_forty_divided_by_five_l2315_231539


namespace NUMINAMATH_CALUDE_function_inequality_condition_l2315_231503

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^2 + x + 1) →
  a > 0 →
  b > 0 →
  (∀ x, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2315_231503


namespace NUMINAMATH_CALUDE_game_ends_in_49_rounds_l2315_231509

/-- Represents a player in the token game -/
inductive Player : Type
  | A | B | C | D

/-- The state of the game at any point -/
structure GameState :=
  (tokens : Player → Nat)

/-- The initial state of the game -/
def initialState : GameState :=
  { tokens := fun p => match p with
    | Player.A => 16
    | Player.B => 15
    | Player.C => 14
    | Player.D => 13 }

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (state : GameState) (count : Nat := 0) : Nat :=
  sorry

theorem game_ends_in_49_rounds :
  countRounds initialState = 49 :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_49_rounds_l2315_231509
