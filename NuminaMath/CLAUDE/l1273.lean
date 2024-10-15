import Mathlib

namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1273_127366

theorem inequality_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x^2 - x + 1) * (a*x + 3*a - 1) < 1) →
  a < 2/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1273_127366


namespace NUMINAMATH_CALUDE_derivative_at_two_l1273_127377

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2*x + 1

theorem derivative_at_two :
  (deriv f) 2 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_two_l1273_127377


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1273_127331

theorem polynomial_factorization (x y : ℝ) : x * y^2 - 36 * x = x * (y + 6) * (y - 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1273_127331


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l1273_127330

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  (Nat.gcd a b) * (Nat.lcm a b) = 12000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l1273_127330


namespace NUMINAMATH_CALUDE_woojin_harvest_weight_l1273_127365

-- Define the weights harvested by each family member
def younger_brother_weight : Real := 3.8
def older_sister_extra : Real := 8.4
def woojin_extra_grams : Real := 3720

-- Define the conversion factor from kg to g
def kg_to_g : Real := 1000

-- Theorem statement
theorem woojin_harvest_weight :
  let older_sister_weight := younger_brother_weight + older_sister_extra
  let woojin_weight_g := (older_sister_weight / 10) * kg_to_g + woojin_extra_grams
  woojin_weight_g / kg_to_g = 4.94 := by
sorry


end NUMINAMATH_CALUDE_woojin_harvest_weight_l1273_127365


namespace NUMINAMATH_CALUDE_factor_polynomial_l1273_127396

theorem factor_polynomial (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1273_127396


namespace NUMINAMATH_CALUDE_visible_yellow_bus_length_l1273_127342

/-- Proves that the visible length of the yellow bus is 18 feet --/
theorem visible_yellow_bus_length (red_bus_length green_truck_length yellow_bus_length orange_car_length : ℝ) :
  red_bus_length = 48 →
  red_bus_length = 4 * orange_car_length →
  yellow_bus_length = 3.5 * orange_car_length →
  green_truck_length = 2 * orange_car_length →
  yellow_bus_length - green_truck_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_visible_yellow_bus_length_l1273_127342


namespace NUMINAMATH_CALUDE_no_solution_to_exponential_equation_l1273_127383

theorem no_solution_to_exponential_equation :
  ¬∃ (x y : ℝ), (9 : ℝ) ^ (x^3 + y) + (9 : ℝ) ^ (x + y^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_exponential_equation_l1273_127383


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1273_127311

/-- The volume of a cube given its surface area -/
theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1273_127311


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l1273_127300

/-- Regular hexagon with vertices J and L -/
structure RegularHexagon where
  J : ℝ × ℝ
  L : ℝ × ℝ

/-- The area of a regular hexagon -/
def hexagon_area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating the area of the specific regular hexagon -/
theorem specific_hexagon_area :
  let h : RegularHexagon := { J := (0, 0), L := (10, 2) }
  hexagon_area h = 156 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l1273_127300


namespace NUMINAMATH_CALUDE_normal_binomial_properties_l1273_127309

/-- A random variable with normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ

/-- A random variable with binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ

/-- The probability that X is less than or equal to x -/
noncomputable def P (X : NormalRV) (x : ℝ) : ℝ := sorry

/-- The expected value of a normal random variable -/
noncomputable def E_normal (X : NormalRV) : ℝ := X.μ

/-- The expected value of a binomial random variable -/
noncomputable def E_binomial (Y : BinomialRV) : ℝ := Y.n * Y.p

/-- The variance of a binomial random variable -/
noncomputable def D_binomial (Y : BinomialRV) : ℝ := Y.n * Y.p * (1 - Y.p)

/-- The main theorem -/
theorem normal_binomial_properties (X : NormalRV) (Y : BinomialRV) 
    (h1 : P X 2 = 0.5)
    (h2 : E_binomial Y = E_normal X)
    (h3 : Y.n = 3) :
  X.μ = 2 ∧ Y.p = 2/3 ∧ 9 * D_binomial Y = 6 := by
  sorry

end NUMINAMATH_CALUDE_normal_binomial_properties_l1273_127309


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_4x_l1273_127394

theorem factorization_x_squared_minus_4x (x : ℝ) : x^2 - 4*x = x*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_4x_l1273_127394


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1273_127387

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1273_127387


namespace NUMINAMATH_CALUDE_polynomial_coefficient_l1273_127353

theorem polynomial_coefficient (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                         a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                         a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_l1273_127353


namespace NUMINAMATH_CALUDE_dihedral_angle_BAC_ACD_is_120_degrees_l1273_127312

-- Define a unit cube
def UnitCube := Set (ℝ × ℝ × ℝ)

-- Define a function to calculate the dihedral angle between two faces of a cube
def dihedralAngle (cube : UnitCube) (face1 face2 : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the specific faces for the B-A₁C-D dihedral angle
def faceBAC (cube : UnitCube) : Set (ℝ × ℝ × ℝ) := sorry
def faceACD (cube : UnitCube) : Set (ℝ × ℝ × ℝ) := sorry

-- State the theorem
theorem dihedral_angle_BAC_ACD_is_120_degrees (cube : UnitCube) : 
  dihedralAngle cube (faceBAC cube) (faceACD cube) = 120 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_BAC_ACD_is_120_degrees_l1273_127312


namespace NUMINAMATH_CALUDE_square_not_always_positive_l1273_127362

theorem square_not_always_positive : ¬ ∀ a : ℝ, a^2 > 0 := by sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l1273_127362


namespace NUMINAMATH_CALUDE_pizza_cost_l1273_127341

theorem pizza_cost (initial_amount : ℕ) (return_amount : ℕ) (juice_cost : ℕ) (juice_quantity : ℕ) (pizza_quantity : ℕ) :
  initial_amount = 50 ∧
  return_amount = 22 ∧
  juice_cost = 2 ∧
  juice_quantity = 2 ∧
  pizza_quantity = 2 →
  (initial_amount - return_amount - juice_cost * juice_quantity) / pizza_quantity = 12 :=
by sorry

end NUMINAMATH_CALUDE_pizza_cost_l1273_127341


namespace NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l1273_127337

theorem max_pairs_with_distinct_sums (n : ℕ) (hn : n = 2009) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 803 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs → p2 ∈ pairs → p1 ≠ p2 →
      p1.1 ≠ p2.1 ∧ p1.1 ≠ p2.2 ∧ p1.2 ≠ p2.1 ∧ p1.2 ≠ p2.2) ∧
    (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs → p2 ∈ pairs → p1 ≠ p2 →
      p1.1 + p1.2 ≠ p2.1 + p2.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (pairs' : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs' → p2 ∈ pairs' → p1 ≠ p2 →
        p1.1 ≠ p2.1 ∧ p1.1 ≠ p2.2 ∧ p1.2 ≠ p2.1 ∧ p1.2 ≠ p2.2) →
      (∀ (p1 p2 : ℕ × ℕ), p1 ∈ pairs' → p2 ∈ pairs' → p1 ≠ p2 →
        p1.1 + p1.2 ≠ p2.1 + p2.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ n) →
      pairs'.card ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_with_distinct_sums_l1273_127337


namespace NUMINAMATH_CALUDE_last_digit_of_3_power_2023_l1273_127373

/-- The last digit of 3^n for n ≥ 1 -/
def lastDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | 0 => 1
  | _ => 0  -- This case should never occur

theorem last_digit_of_3_power_2023 :
  lastDigitOf3Power 2023 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_3_power_2023_l1273_127373


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1273_127321

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1273_127321


namespace NUMINAMATH_CALUDE_caroling_boys_count_l1273_127313

/-- The number of boys who received 1 orange each -/
def boys_with_one_orange : ℕ := 2

/-- The number of boys who received 2 oranges each -/
def boys_with_two_oranges : ℕ := 4

/-- The number of boys who received 4 oranges -/
def boys_with_four_oranges : ℕ := 1

/-- The number of oranges received by boys with known names -/
def oranges_known_boys : ℕ := boys_with_one_orange + 2 * boys_with_two_oranges + 4 * boys_with_four_oranges

/-- The total number of oranges received by all boys -/
def total_oranges : ℕ := 23

/-- The number of oranges each of the other boys received -/
def oranges_per_other_boy : ℕ := 3

theorem caroling_boys_count : ∃ (n : ℕ), 
  n = boys_with_one_orange + boys_with_two_oranges + boys_with_four_oranges + 
      (total_oranges - oranges_known_boys) / oranges_per_other_boy ∧ 
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_caroling_boys_count_l1273_127313


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1273_127344

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.sin t.A * Real.cos t.B = Real.sin t.C) : 
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1273_127344


namespace NUMINAMATH_CALUDE_speed_of_current_l1273_127316

/-- 
Given a man's speed with and against a current, this theorem proves 
the speed of the current.
-/
theorem speed_of_current 
  (speed_with_current : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : speed_against_current = 18) : 
  ∃ (current_speed : ℝ), current_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_current_l1273_127316


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l1273_127332

noncomputable section

/-- Given two non-zero vectors a and b in ℝ², prove that under certain conditions,
    the projection of a onto b is (1/4) * b. -/
theorem projection_of_a_onto_b (a b : ℝ × ℝ) : 
  a ≠ (0, 0) → 
  b = (Real.sqrt 3, 1) → 
  a.1 * b.1 + a.2 * b.2 = π / 3 → 
  (a.1 - b.1) * a.1 + (a.2 - b.2) * a.2 = 0 → 
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b = (1/4) • b := by
  sorry

end

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l1273_127332


namespace NUMINAMATH_CALUDE_function_and_tangent_line_l1273_127369

/-- Given a function f(x) = (ax-6) / (x^2 + b) and its tangent line at (-1, f(-1)) 
    with equation x + 2y + 5 = 0, prove that f(x) = (2x-6) / (x^2 + 3) -/
theorem function_and_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => (a * x - 6) / (x^2 + b)
  let tangent_line : ℝ → ℝ := λ x => -(1/2) * x - 5/2
  (f (-1) = tangent_line (-1)) ∧ 
  (deriv f (-1) = deriv tangent_line (-1)) →
  f = λ x => (2 * x - 6) / (x^2 + 3) := by
sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_l1273_127369


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1273_127384

theorem inequality_system_solution (x : ℝ) :
  (1 - x > 3) ∧ (2 * x + 5 ≥ 0) → -2.5 ≤ x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1273_127384


namespace NUMINAMATH_CALUDE_inequality_proof_l1273_127339

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1273_127339


namespace NUMINAMATH_CALUDE_simplify_expression_l1273_127374

theorem simplify_expression (x y : ℝ) :
  (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1273_127374


namespace NUMINAMATH_CALUDE_room_perimeter_is_16_l1273_127327

/-- A rectangular room with specific properties -/
structure Room where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_eq : length = 3 * breadth
  area_eq : area = length * breadth

/-- The perimeter of a rectangular room -/
def perimeter (r : Room) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: The perimeter of a room with given properties is 16 meters -/
theorem room_perimeter_is_16 (r : Room) (h : r.area = 12) : perimeter r = 16 := by
  sorry

end NUMINAMATH_CALUDE_room_perimeter_is_16_l1273_127327


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1273_127304

theorem unique_integer_solution (a b c : ℤ) :
  a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c ↔ a = 1 ∧ b = 2 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1273_127304


namespace NUMINAMATH_CALUDE_negation_exists_product_zero_l1273_127395

open Real

theorem negation_exists_product_zero (f g : ℝ → ℝ) :
  (¬ ∃ x₀ : ℝ, f x₀ * g x₀ = 0) ↔ (∀ x : ℝ, f x ≠ 0 ∧ g x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_exists_product_zero_l1273_127395


namespace NUMINAMATH_CALUDE_max_profit_week_is_5_l1273_127363

/-- Price function based on week number -/
def price (x : ℕ) : ℚ :=
  if x ≤ 4 then 10 + 2 * x
  else if x ≤ 10 then 20
  else 20 - 2 * (x - 10)

/-- Cost function based on week number -/
def cost (x : ℕ) : ℚ :=
  -0.125 * (x - 8)^2 + 12

/-- Profit function based on week number -/
def profit (x : ℕ) : ℚ :=
  price x - cost x

/-- The week with maximum profit is the 5th week -/
theorem max_profit_week_is_5 :
  ∀ x : ℕ, x ≤ 16 → profit 5 ≥ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_week_is_5_l1273_127363


namespace NUMINAMATH_CALUDE_prob_at_most_one_even_is_three_fourths_l1273_127358

/-- A die is fair if each number has an equal probability of 1/6 -/
def FairDie (d : Fin 6 → ℝ) : Prop :=
  ∀ n : Fin 6, d n = 1 / 6

/-- The probability of getting an even number on a fair die -/
def ProbEven (d : Fin 6 → ℝ) : ℝ :=
  d 1 + d 3 + d 5

/-- The probability of getting an odd number on a fair die -/
def ProbOdd (d : Fin 6 → ℝ) : ℝ :=
  d 0 + d 2 + d 4

/-- The probability of at most one die showing an even number when throwing two fair dice -/
def ProbAtMostOneEven (d1 d2 : Fin 6 → ℝ) : ℝ :=
  ProbOdd d1 * ProbOdd d2 + ProbOdd d1 * ProbEven d2 + ProbEven d1 * ProbOdd d2

theorem prob_at_most_one_even_is_three_fourths 
  (red blue : Fin 6 → ℝ) 
  (hred : FairDie red) 
  (hblue : FairDie blue) : 
  ProbAtMostOneEven red blue = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_even_is_three_fourths_l1273_127358


namespace NUMINAMATH_CALUDE_lucas_50th_mod5_lucas_50th_remainder_l1273_127379

/-- Lucas sequence -/
def lucas : ℕ → ℤ
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

/-- Lucas sequence modulo 5 -/
def lucas_mod5 (n : ℕ) : ℤ := lucas n % 5

/-- The Lucas sequence modulo 5 has a period of 4 -/
axiom lucas_mod5_period : ∀ n, lucas_mod5 (n + 4) = lucas_mod5 n

/-- The 50th term of the Lucas sequence modulo 5 equals the 2nd term modulo 5 -/
theorem lucas_50th_mod5 : lucas_mod5 50 = lucas_mod5 2 := by sorry

/-- The remainder when the 50th term of the Lucas sequence is divided by 5 is 1 -/
theorem lucas_50th_remainder : lucas 50 % 5 = 1 := by sorry

end NUMINAMATH_CALUDE_lucas_50th_mod5_lucas_50th_remainder_l1273_127379


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1273_127389

theorem solve_exponential_equation :
  ∃ y : ℝ, (9 : ℝ) ^ y = (3 : ℝ) ^ 12 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1273_127389


namespace NUMINAMATH_CALUDE_water_breadth_in_cistern_l1273_127310

/-- Represents a rectangular cistern with water --/
structure WaterCistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ
  breadth : ℝ

/-- Theorem stating the correct breadth of water in the cistern --/
theorem water_breadth_in_cistern (c : WaterCistern)
  (h_length : c.length = 7)
  (h_width : c.width = 5)
  (h_wetArea : c.wetSurfaceArea = 68.6)
  (h_breadth_calc : c.breadth = (c.wetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))) :
  c.breadth = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_water_breadth_in_cistern_l1273_127310


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1273_127338

/-- A linear function y = mx - 1 passing through the second, third, and fourth quadrants implies m < 0 -/
theorem linear_function_quadrants (m : ℝ) : 
  (∀ x y : ℝ, y = m * x - 1 →
    ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) →
  m < 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1273_127338


namespace NUMINAMATH_CALUDE_equation_solutions_l1273_127336

theorem equation_solutions :
  (∀ x : ℝ, (x + 1)^2 - 9 = 0 ↔ x = -4 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - 12*x - 4 = 0 ↔ x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10) ∧
  (∀ x : ℝ, 3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1273_127336


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l1273_127382

theorem arithmetic_mean_of_specific_numbers :
  let numbers := [17, 29, 45, 64]
  (numbers.sum / numbers.length : ℚ) = 38.75 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l1273_127382


namespace NUMINAMATH_CALUDE_increase_in_position_for_given_slope_l1273_127352

/-- The increase in position for a person moving along a slope --/
def increase_in_position (slope_ratio : ℚ) (total_distance : ℝ) : ℝ :=
  sorry

/-- The theorem stating the increase in position for the given problem --/
theorem increase_in_position_for_given_slope : 
  increase_in_position (1/2) (100 * Real.sqrt 5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_position_for_given_slope_l1273_127352


namespace NUMINAMATH_CALUDE_shyam_weight_increase_l1273_127392

-- Define the ratio of Ram's weight to Shyam's weight
def weight_ratio : ℚ := 2 / 5

-- Define Ram's weight increase percentage
def ram_increase : ℚ := 10 / 100

-- Define the total new weight
def total_new_weight : ℚ := 828 / 10

-- Define the total weight increase percentage
def total_increase : ℚ := 15 / 100

-- Function to calculate Shyam's weight increase percentage
def shyam_increase_percentage : ℚ := sorry

-- Theorem statement
theorem shyam_weight_increase :
  abs (shyam_increase_percentage - 1709 / 10000) < 1 / 1000 := by sorry

end NUMINAMATH_CALUDE_shyam_weight_increase_l1273_127392


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1273_127393

/-- Represents an ellipse with semi-major axis a and eccentricity e -/
structure Ellipse where
  a : ℝ
  e : ℝ

/-- The equation of the ellipse in terms of m -/
def ellipse_equation (m : ℝ) : Prop :=
  m > 1 ∧ ∃ x y : ℝ, x^2 / m^2 + y^2 / (m^2 - 1) = 1

/-- The distances from a point on the ellipse to its foci -/
def focus_distances (left right : ℝ) : Prop :=
  left = 3 ∧ right = 1

/-- The theorem stating the eccentricity of the ellipse -/
theorem ellipse_eccentricity (m : ℝ) :
  ellipse_equation m →
  (∃ left right : ℝ, focus_distances left right) →
  ∃ e : Ellipse, e.e = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1273_127393


namespace NUMINAMATH_CALUDE_inequality_implication_l1273_127350

theorem inequality_implication (a b : ℝ) (h : a > b) : -5 * a < -5 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1273_127350


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l1273_127329

theorem xiaolis_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1.1 * x) / (0.9 * y) > x / y := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l1273_127329


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_condition_l1273_127354

theorem triangle_inequality_and_equality_condition (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_condition_l1273_127354


namespace NUMINAMATH_CALUDE_deposit_calculation_l1273_127375

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) :
  remaining_amount = 950 ∧ deposit_percentage = 0.05 →
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 50 :=
by sorry

end NUMINAMATH_CALUDE_deposit_calculation_l1273_127375


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l1273_127367

theorem largest_constant_inequality :
  ∃ (C : ℝ), (C = 2 / Real.sqrt 3) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + 2*z^2 + 1 ≥ C*(x + y + z)) ∧
  (∀ (C' : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + 2*z^2 + 1 ≥ C'*(x + y + z)) → C' ≤ C) :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l1273_127367


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l1273_127371

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (8, 6)

theorem vectors_orthogonal : v1.1 * v2.1 + v1.2 * v2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l1273_127371


namespace NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l1273_127388

/-- Proves that the initial percentage of concentrated kola in a solution is 9% -/
theorem initial_concentrated_kola_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_concentrated_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 64)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 8)
  (h5 : added_concentrated_kola = 6.8)
  (h6 : final_sugar_percentage = 26.536312849162012)
  (h7 : (((100 - initial_water_percentage - 9) * initial_volume / 100 + added_sugar) /
         (initial_volume + added_sugar + added_water + added_concentrated_kola)) * 100 = final_sugar_percentage) :
  9 = 100 - initial_water_percentage - ((initial_volume * initial_water_percentage / 100 + added_water) /
    (initial_volume + added_sugar + added_water + added_concentrated_kola) * 100) :=
by sorry

end NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l1273_127388


namespace NUMINAMATH_CALUDE_one_true_statement_l1273_127359

theorem one_true_statement (a b c : ℝ) : 
  (∃! n : Nat, n = 1 ∧ 
    (((a ≤ b → a * c^2 ≤ b * c^2) ∨ 
      (a > b → a * c^2 > b * c^2) ∨ 
      (a * c^2 ≤ b * c^2 → a ≤ b)))) := by sorry

end NUMINAMATH_CALUDE_one_true_statement_l1273_127359


namespace NUMINAMATH_CALUDE_sqrt_product_exists_l1273_127317

theorem sqrt_product_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ∃ x : ℝ, x^2 = a * b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_exists_l1273_127317


namespace NUMINAMATH_CALUDE_five_eighths_decimal_l1273_127335

theorem five_eighths_decimal : (5 : ℚ) / 8 = 0.625 := by sorry

end NUMINAMATH_CALUDE_five_eighths_decimal_l1273_127335


namespace NUMINAMATH_CALUDE_gcd_problem_l1273_127308

/-- The GCD operation -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- The problem statement -/
theorem gcd_problem (n m k j : ℕ+) :
  gcd_op (gcd_op (16 * n) (20 * m)) (gcd_op (18 * k) (24 * j)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1273_127308


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1273_127333

theorem trig_identity_proof : 
  Real.cos (70 * π / 180) * Real.sin (80 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1273_127333


namespace NUMINAMATH_CALUDE_petyas_coins_l1273_127302

/-- Represents the denominations of coins --/
inductive Coin
  | OneRuble
  | TwoRubles
  | Other

/-- Represents Petya's pocket of coins --/
structure Pocket where
  coins : List Coin

/-- Checks if a list of coins contains at least one 1 ruble coin --/
def hasOneRuble (coins : List Coin) : Prop :=
  Coin.OneRuble ∈ coins

/-- Checks if a list of coins contains at least one 2 rubles coin --/
def hasTwoRubles (coins : List Coin) : Prop :=
  Coin.TwoRubles ∈ coins

/-- The main theorem to prove --/
theorem petyas_coins (p : Pocket) :
  (∀ (subset : List Coin), subset ⊆ p.coins → subset.length = 3 → hasOneRuble subset) →
  (∀ (subset : List Coin), subset ⊆ p.coins → subset.length = 4 → hasTwoRubles subset) →
  p.coins.length = 5 →
  p.coins = [Coin.OneRuble, Coin.OneRuble, Coin.OneRuble, Coin.TwoRubles, Coin.TwoRubles] :=
by sorry

end NUMINAMATH_CALUDE_petyas_coins_l1273_127302


namespace NUMINAMATH_CALUDE_billy_ticket_difference_l1273_127306

/-- The difference between initial tickets and remaining tickets after purchases -/
def ticket_difference (initial_tickets yoyo_cost keychain_cost plush_toy_cost : ℝ) : ℝ :=
  initial_tickets - (initial_tickets - (yoyo_cost + keychain_cost + plush_toy_cost))

/-- Theorem stating the ticket difference for Billy's specific case -/
theorem billy_ticket_difference :
  ticket_difference 48.5 11.7 6.3 16.2 = 14.3 := by
  sorry

end NUMINAMATH_CALUDE_billy_ticket_difference_l1273_127306


namespace NUMINAMATH_CALUDE_fraction_of_fifteen_l1273_127368

theorem fraction_of_fifteen (x : ℚ) : 
  (x * 15 = 0.8 * 40 - 20) → x = 4/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_fifteen_l1273_127368


namespace NUMINAMATH_CALUDE_sum_of_possible_k_is_95_l1273_127381

/-- Given a quadratic equation x^2 + 10x + k = 0 with two distinct negative integer solutions,
    this function returns the sum of all possible values of k. -/
def sumOfPossibleK : ℤ := by
  sorry

/-- The theorem states that the sum of all possible values of k is 95. -/
theorem sum_of_possible_k_is_95 : sumOfPossibleK = 95 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_k_is_95_l1273_127381


namespace NUMINAMATH_CALUDE_fifty_three_is_perfect_sum_x_y_is_one_k_equals_36_max_x_minus_2y_is_two_l1273_127320

-- Definition of a perfect number
def isPerfectNumber (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Statement 1
theorem fifty_three_is_perfect : isPerfectNumber 53 := by sorry

-- Statement 2
theorem sum_x_y_is_one (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 5 = 0) : 
  x + y = 1 := by sorry

-- Statement 3
theorem k_equals_36 (k : ℤ) : 
  (∀ x y : ℤ, isPerfectNumber (2*x^2 + y^2 + 2*x*y + 12*x + k)) → k = 36 := by sorry

-- Statement 4
theorem max_x_minus_2y_is_two (x y : ℝ) (h : -x^2 + (7/2)*x + y - 3 = 0) :
  x - 2*y ≤ 2 := by sorry

end NUMINAMATH_CALUDE_fifty_three_is_perfect_sum_x_y_is_one_k_equals_36_max_x_minus_2y_is_two_l1273_127320


namespace NUMINAMATH_CALUDE_remainder_of_1742_base12_div_9_l1273_127349

/-- Converts a base-12 digit to base-10 --/
def base12ToBase10(digit : Nat) : Nat :=
  if digit < 12 then digit else 0

/-- Converts a base-12 number to base-10 --/
def convertBase12ToBase10(n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + base12ToBase10 d * 12^i) 0

/-- The base-12 representation of 1742₁₂ --/
def base12Num : List Nat := [2, 4, 7, 1]

theorem remainder_of_1742_base12_div_9 :
  (convertBase12ToBase10 base12Num) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1742_base12_div_9_l1273_127349


namespace NUMINAMATH_CALUDE_digit_sum_in_base_d_l1273_127385

/-- A function to represent a two-digit number in base d -/
def two_digit_number (d a b : ℕ) : ℕ := a * d + b

/-- The problem statement -/
theorem digit_sum_in_base_d (d A B : ℕ) : 
  d > 8 →
  A < d →
  B < d →
  two_digit_number d A B + two_digit_number d A A - two_digit_number d B A = 180 →
  A + B = 10 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_in_base_d_l1273_127385


namespace NUMINAMATH_CALUDE_acid_concentration_percentage_l1273_127334

/-- 
Given a solution with a certain volume of pure acid and a total volume,
calculate the percentage concentration of pure acid in the solution.
-/
theorem acid_concentration_percentage 
  (pure_acid_volume : ℝ) 
  (total_solution_volume : ℝ) 
  (h1 : pure_acid_volume = 4.800000000000001)
  (h2 : total_solution_volume = 12) :
  (pure_acid_volume / total_solution_volume) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_acid_concentration_percentage_l1273_127334


namespace NUMINAMATH_CALUDE_negation_of_universal_is_existential_l1273_127318

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Boy : U → Prop)
variable (LovesFootball : U → Prop)

-- State the theorem
theorem negation_of_universal_is_existential :
  (¬ ∀ x, Boy x → LovesFootball x) ↔ (∃ x, Boy x ∧ ¬ LovesFootball x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_is_existential_l1273_127318


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l1273_127361

theorem min_value_fraction (x : ℝ) (h : x > 0) :
  (x^2 + x + 3) / (x + 1) ≥ 2 * Real.sqrt 3 - 1 :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, (x^2 + x + 3) / (x + 1) = 2 * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l1273_127361


namespace NUMINAMATH_CALUDE_square_root_divided_by_19_equals_4_l1273_127378

theorem square_root_divided_by_19_equals_4 : 
  ∃ (x : ℝ), x > 0 ∧ (Real.sqrt x) / 19 = 4 ∧ x = 5776 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_19_equals_4_l1273_127378


namespace NUMINAMATH_CALUDE_work_to_pump_oil_horizontal_cylinder_l1273_127370

/-- Work required to pump oil from a horizontal cylindrical tank -/
theorem work_to_pump_oil_horizontal_cylinder 
  (δ : ℝ) -- specific weight of oil
  (H : ℝ) -- length of the cylinder
  (R : ℝ) -- radius of the cylinder
  (h : R > 0) -- assumption that radius is positive
  (h' : H > 0) -- assumption that length is positive
  (h'' : δ > 0) -- assumption that specific weight is positive
  : ∃ (Q : ℝ), Q = π * δ * H * R^3 :=
sorry

end NUMINAMATH_CALUDE_work_to_pump_oil_horizontal_cylinder_l1273_127370


namespace NUMINAMATH_CALUDE_problem_solution_l1273_127351

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1273_127351


namespace NUMINAMATH_CALUDE_student_616_selected_l1273_127356

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  first_selected : ℕ

/-- Checks if a student number is selected in the systematic sampling -/
def is_selected (s : SystematicSampling) (student : ℕ) : Prop :=
  ∃ k : ℕ, student = s.first_selected + k * (s.population / s.sample_size)

theorem student_616_selected (s : SystematicSampling)
  (h_pop : s.population = 1000)
  (h_sample : s.sample_size = 100)
  (h_46_selected : is_selected s 46) :
  is_selected s 616 := by
sorry

end NUMINAMATH_CALUDE_student_616_selected_l1273_127356


namespace NUMINAMATH_CALUDE_mrs_lim_revenue_l1273_127376

/-- Calculates the revenue from milk sales given the milk production and sales data --/
def milk_revenue (yesterday_morning : ℕ) (yesterday_evening : ℕ) (morning_decrease : ℕ) (remaining : ℕ) (price_per_gallon : ℚ) : ℚ :=
  let total_yesterday := yesterday_morning + yesterday_evening
  let this_morning := yesterday_morning - morning_decrease
  let total_milk := total_yesterday + this_morning
  let sold_milk := total_milk - remaining
  sold_milk * price_per_gallon

/-- Theorem stating that Mrs. Lim's revenue is $616 given the specified conditions --/
theorem mrs_lim_revenue :
  milk_revenue 68 82 18 24 (350/100) = 616 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lim_revenue_l1273_127376


namespace NUMINAMATH_CALUDE_perimeter_of_triangle_l1273_127305

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 9

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define points P and Q on the left branch
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P and Q are on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom Q_on_hyperbola : hyperbola Q.1 Q.2

-- State that PQ passes through the left focus
axiom PQ_through_left_focus : sorry

-- Define the length of PQ
def PQ_length : ℝ := 7

-- Define the property of hyperbola for P and Q
axiom hyperbola_property_P : dist P right_focus - dist P left_focus = 6
axiom hyperbola_property_Q : dist Q right_focus - dist Q left_focus = 6

-- Theorem to prove
theorem perimeter_of_triangle : 
  dist P right_focus + dist Q right_focus + PQ_length = 26 := sorry

end NUMINAMATH_CALUDE_perimeter_of_triangle_l1273_127305


namespace NUMINAMATH_CALUDE_Q_equals_sum_l1273_127325

/-- Binomial coefficient -/
def binomial (a b : ℕ) : ℕ :=
  if a ≥ b then
    Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))
  else
    0

/-- Coefficient of x^k in (1+x+x^2+x^3)^n -/
def Q (n k : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (fun j => binomial n j * binomial n (k - 2 * j))

/-- The main theorem -/
theorem Q_equals_sum (n k : ℕ) :
    Q n k = (Finset.range (n + 1)).sum (fun j => binomial n j * binomial n (k - 2 * j)) := by
  sorry

end NUMINAMATH_CALUDE_Q_equals_sum_l1273_127325


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l1273_127360

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The first three terms of the arithmetic progression -/
def first_three_terms (x : ℝ) : ℕ → ℝ
| 0 => x - 2
| 1 => x + 2
| 2 => 3*x + 4
| _ => 0  -- This is just a placeholder for other terms

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (first_three_terms x) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l1273_127360


namespace NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l1273_127355

theorem arithmetic_expressions_evaluation :
  ((-12) - 5 + (-14) - (-39) = 8) ∧
  (-2^2 * 5 - (-12) / 4 - 4 = -21) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_evaluation_l1273_127355


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1273_127397

theorem cubic_equation_solution (b : ℝ) : 
  let x := b
  let c := 0
  x^3 + c^2 = (b - x)^2 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1273_127397


namespace NUMINAMATH_CALUDE_circle_mapping_l1273_127301

-- Define the complex plane
variable (z : ℂ)

-- Define the transformation function
def w (z : ℂ) : ℂ := 3 * z + 2

-- Define the original circle
def original_circle (z : ℂ) : Prop := z.re^2 + z.im^2 = 4

-- Define the mapped circle
def mapped_circle (w : ℂ) : Prop := (w.re - 2)^2 + w.im^2 = 36

-- Theorem statement
theorem circle_mapping :
  ∀ z, original_circle z → mapped_circle (w z) :=
sorry

end NUMINAMATH_CALUDE_circle_mapping_l1273_127301


namespace NUMINAMATH_CALUDE_jayas_rank_from_bottom_l1273_127319

/-- Given a class of students, calculate the rank from the bottom based on the rank from the top. -/
def rankFromBottom (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Jaya's rank from the bottom in a class of 53 students where she ranks 5th from the top is 50th. -/
theorem jayas_rank_from_bottom :
  rankFromBottom 53 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jayas_rank_from_bottom_l1273_127319


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1273_127398

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  (180 * (n - 2) : ℚ) / n = 150 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l1273_127398


namespace NUMINAMATH_CALUDE_roller_coaster_cost_proof_l1273_127315

/-- The cost of the Ferris wheel in tickets -/
def ferris_wheel_cost : ℕ := 6

/-- The cost of the log ride in tickets -/
def log_ride_cost : ℕ := 7

/-- The number of tickets Antonieta initially has -/
def initial_tickets : ℕ := 2

/-- The number of additional tickets Antonieta needs to buy -/
def additional_tickets : ℕ := 16

/-- The cost of the roller coaster in tickets -/
def roller_coaster_cost : ℕ := 5

theorem roller_coaster_cost_proof :
  roller_coaster_cost = 
    (initial_tickets + additional_tickets) - (ferris_wheel_cost + log_ride_cost) :=
by sorry

end NUMINAMATH_CALUDE_roller_coaster_cost_proof_l1273_127315


namespace NUMINAMATH_CALUDE_wire_length_ratio_l1273_127357

-- Define the given conditions
def bonnie_wire_length : ℝ := 12 * 8
def roark_cube_volume : ℝ := 2
def roark_edge_length : ℝ := 1.5
def bonnie_cube_volume : ℝ := 8^3

-- Define the theorem
theorem wire_length_ratio :
  let roark_cubes_count : ℝ := bonnie_cube_volume / roark_cube_volume
  let roark_total_wire_length : ℝ := roark_cubes_count * (12 * roark_edge_length)
  bonnie_wire_length / roark_total_wire_length = 1 / 48 := by
sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l1273_127357


namespace NUMINAMATH_CALUDE_book_original_price_l1273_127343

/-- Proves that given a book sold for Rs 60 with a 20% profit rate, the original price of the book was Rs 50. -/
theorem book_original_price (selling_price : ℝ) (profit_rate : ℝ) : 
  selling_price = 60 → profit_rate = 0.20 → 
  ∃ (original_price : ℝ), original_price = 50 ∧ selling_price = original_price * (1 + profit_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_book_original_price_l1273_127343


namespace NUMINAMATH_CALUDE_quadratic_root_expression_l1273_127380

theorem quadratic_root_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ = 0 → 
  x₂^2 - 2*x₂ = 0 → 
  (x₁ * x₂) / (x₁^2 + x₂^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_expression_l1273_127380


namespace NUMINAMATH_CALUDE_x_value_l1273_127323

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1273_127323


namespace NUMINAMATH_CALUDE_tom_car_washing_earnings_l1273_127386

/-- The amount of money Tom had last week -/
def initial_amount : ℕ := 74

/-- The amount of money Tom has now -/
def current_amount : ℕ := 86

/-- The amount of money Tom made washing cars -/
def money_made : ℕ := current_amount - initial_amount

theorem tom_car_washing_earnings : 
  money_made = current_amount - initial_amount :=
by sorry

end NUMINAMATH_CALUDE_tom_car_washing_earnings_l1273_127386


namespace NUMINAMATH_CALUDE_trapezium_side_length_l1273_127314

/-- Theorem: In a trapezium with one parallel side of length 12 cm, a distance between parallel sides
    of 14 cm, and an area of 196 square centimeters, the length of the other parallel side is 16 cm. -/
theorem trapezium_side_length 
  (side1 : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 12) 
  (h2 : height = 14) 
  (h3 : area = 196) 
  (h4 : area = (side1 + side2) * height / 2) : 
  side2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l1273_127314


namespace NUMINAMATH_CALUDE_inequality_for_all_reals_l1273_127324

theorem inequality_for_all_reals (a : ℝ) : a + a^3 - a^4 - a^6 < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_all_reals_l1273_127324


namespace NUMINAMATH_CALUDE_alma_carrot_distribution_l1273_127372

/-- Given a number of carrots and goats, calculate the number of carrots left over
    when distributing carrots equally among goats. -/
def carrots_left_over (total_carrots : ℕ) (num_goats : ℕ) : ℕ :=
  total_carrots % num_goats

theorem alma_carrot_distribution :
  carrots_left_over 47 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_alma_carrot_distribution_l1273_127372


namespace NUMINAMATH_CALUDE_fold_reflection_l1273_127328

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given three points A, B, and C on a coordinate grid, if A coincides with B after folding,
    then C will coincide with the reflection of C across the perpendicular bisector of AB. -/
theorem fold_reflection (A B C : Point) (h : A.x = 10 ∧ A.y = 0 ∧ B.x = -6 ∧ B.y = 8 ∧ C.x = -4 ∧ C.y = 2) :
  ∃ (P : Point), P.x = 4 ∧ P.y = -2 ∧ 
  (2 * (C.x + P.x) = 2 * ((A.x + B.x) / 2)) ∧
  (C.y + P.y = 2 * ((A.y + B.y) / 2)) := by
  sorry


end NUMINAMATH_CALUDE_fold_reflection_l1273_127328


namespace NUMINAMATH_CALUDE_prob_two_green_marbles_l1273_127303

/-- The probability of drawing two green marbles without replacement from a bag containing 5 blue marbles and 7 green marbles is 7/22. -/
theorem prob_two_green_marbles (blue_marbles green_marbles : ℕ) 
  (h_blue : blue_marbles = 5) (h_green : green_marbles = 7) :
  let total_marbles := blue_marbles + green_marbles
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_green_marbles_l1273_127303


namespace NUMINAMATH_CALUDE_min_year_exceed_300k_l1273_127340

/-- Represents the linear regression equation for online shoppers --/
def online_shoppers (x : ℤ) : ℝ := 42 * x - 26

/-- Theorem: The minimum integer value of x for which the number of online shoppers exceeds 300 thousand is 8 --/
theorem min_year_exceed_300k :
  ∀ x : ℤ, (x ≥ 8 ↔ online_shoppers x > 300) ∧
  ∀ y : ℤ, y < 8 → online_shoppers y ≤ 300 :=
sorry


end NUMINAMATH_CALUDE_min_year_exceed_300k_l1273_127340


namespace NUMINAMATH_CALUDE_one_true_proposition_l1273_127347

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Define the converse
def converse (x : ℝ) : Prop := x^2 > 0 → x > 0

-- Define the negation
def negation (x : ℝ) : Prop := x > 0 → x^2 ≤ 0

-- Define the inverse negation
def inverse_negation (x : ℝ) : Prop := x^2 ≤ 0 → x ≤ 0

-- Theorem stating that exactly one of these is true
theorem one_true_proposition :
  ∃! p : (ℝ → Prop), p = converse ∨ p = negation ∨ p = inverse_negation ∧ ∀ x, p x :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l1273_127347


namespace NUMINAMATH_CALUDE_min_value_problem_l1273_127364

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = x * y) :
  3 * x + 4 * y ≥ 28 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 28 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1273_127364


namespace NUMINAMATH_CALUDE_bob_has_31_pennies_l1273_127322

/-- The number of pennies Alex currently has -/
def alexPennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bobPennies : ℕ := sorry

/-- If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bobPennies + 1 = 4 * (alexPennies - 1)

/-- If Bob gives Alex a penny, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bobPennies - 1 = 3 * (alexPennies + 1)

/-- Bob currently has 31 pennies -/
theorem bob_has_31_pennies : bobPennies = 31 := by sorry

end NUMINAMATH_CALUDE_bob_has_31_pennies_l1273_127322


namespace NUMINAMATH_CALUDE_apple_juice_fraction_l1273_127326

theorem apple_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_fullness pitcher2_fullness : ℚ) :
  pitcher1_capacity = 800 →
  pitcher2_capacity = 500 →
  pitcher1_fullness = 1/4 →
  pitcher2_fullness = 3/8 →
  (pitcher1_capacity * pitcher1_fullness + pitcher2_capacity * pitcher2_fullness) /
  (pitcher1_capacity + pitcher2_capacity) = 31/104 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_fraction_l1273_127326


namespace NUMINAMATH_CALUDE_strawberry_area_l1273_127307

/-- The area of strawberries in a circular garden -/
theorem strawberry_area (d : ℝ) (h1 : d = 16) : ∃ (A : ℝ), A = 8 * Real.pi ∧ A = (1/8) * Real.pi * d^2 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_area_l1273_127307


namespace NUMINAMATH_CALUDE_palindrome_count_is_60_l1273_127390

/-- Represents a time on a 24-hour digital clock --/
structure DigitalTime where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Checks if a given DigitalTime is a palindrome --/
def is_palindrome (t : DigitalTime) : Bool :=
  sorry

/-- Counts the number of palindromes on a 24-hour digital clock --/
def count_palindromes : Nat :=
  sorry

/-- Theorem stating that the number of palindromes on a 24-hour digital clock is 60 --/
theorem palindrome_count_is_60 : count_palindromes = 60 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_count_is_60_l1273_127390


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1273_127399

theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) →
  (∀ x, x^2 - b*x - a < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1273_127399


namespace NUMINAMATH_CALUDE_compound_interest_equation_l1273_127348

/-- The initial sum of money lent out -/
def P : ℝ := sorry

/-- The final amount after 2 years -/
def final_amount : ℝ := 341

/-- The semi-annual interest rate for the first year -/
def r1 : ℝ := 0.025

/-- The semi-annual interest rate for the second year -/
def r2 : ℝ := 0.03

/-- The number of compounding periods per year -/
def n : ℕ := 2

/-- The total number of compounding periods -/
def total_periods : ℕ := 4

theorem compound_interest_equation :
  P * (1 + r1)^n * (1 + r2)^n = final_amount := by sorry

end NUMINAMATH_CALUDE_compound_interest_equation_l1273_127348


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_89_l1273_127391

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

def is_divisible_by_89 (x : ℕ) : Prop := x % 89 = 0

theorem largest_number_divisible_by_89 :
  ∃ (n : ℕ), 
    (n % 2 = 1) ∧ 
    (3 ≤ n) ∧ 
    (n ≤ 7) ∧ 
    (is_divisible_by_89 (largest_n_digit_number n)) ∧
    (∀ (m : ℕ), 
      (m % 2 = 1) → 
      (3 ≤ m) → 
      (m ≤ 7) → 
      (is_divisible_by_89 (largest_n_digit_number m)) → 
      (largest_n_digit_number m ≤ largest_n_digit_number n)) ∧
    (largest_n_digit_number n = 9999951) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_89_l1273_127391


namespace NUMINAMATH_CALUDE_inequality_proof_l1273_127345

theorem inequality_proof (x : ℝ) :
  x > 0 →
  (x * Real.sqrt (12 - x) + Real.sqrt (12 * x - x^3) ≥ 12) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1273_127345


namespace NUMINAMATH_CALUDE_isosceles_if_neg_one_root_side_c_value_l1273_127346

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation b(x²-1) + 2ax + c(x²+1) = 0 -/
def equation (t : Triangle) (x : ℝ) : Prop :=
  t.b * (x^2 - 1) + 2 * t.a * x + t.c * (x^2 + 1) = 0

theorem isosceles_if_neg_one_root (t : Triangle) :
  equation t (-1) → t.a = t.c :=
sorry

theorem side_c_value (t : Triangle) :
  (∃ x : ℝ, ∀ y : ℝ, equation t x ↔ y = x) →
  t.a = 5 →
  t.b = 12 →
  t.c = 13 :=
sorry

end NUMINAMATH_CALUDE_isosceles_if_neg_one_root_side_c_value_l1273_127346
