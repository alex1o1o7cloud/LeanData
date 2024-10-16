import Mathlib

namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l532_53219

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem distinct_prime_factors_of_30_factorial :
  (Finset.filter (Nat.Prime) (Finset.range 31)).card = 10 ∧
  ∀ p : ℕ, Nat.Prime p → p ∣ factorial 30 ↔ p ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l532_53219


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l532_53204

/-- The equation is quadratic with respect to x if and only if m^2 - 2 = 2 -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2 = 2

/-- The equation is not degenerate if and only if m - 2 ≠ 0 -/
def is_not_degenerate (m : ℝ) : Prop := m - 2 ≠ 0

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m ∧ is_not_degenerate m → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l532_53204


namespace NUMINAMATH_CALUDE_solution_existence_l532_53212

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- State the theorem
theorem solution_existence (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) ↔ m ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_solution_existence_l532_53212


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l532_53290

-- Define an isosceles triangle with sides of lengths 3 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3)

-- Triangle inequality theorem
axiom triangle_inequality (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  IsoscelesTriangle a b c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 7 ∨ b = 7 ∨ c = 7) →
  a + b + c = 17 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l532_53290


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l532_53258

noncomputable def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y

theorem functional_equation_solutions (f : ℝ → ℝ) :
  FunctionalEquation f →
  (∀ x : ℝ, f x = 0) ∨
  (∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → f x = 1) ∧ f 0 = a) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l532_53258


namespace NUMINAMATH_CALUDE_fathers_age_l532_53271

theorem fathers_age (man_age father_age : ℝ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 5 = (1 / 2) * (father_age + 5) → 
  father_age = 25 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l532_53271


namespace NUMINAMATH_CALUDE_chord_bisected_at_P_l532_53249

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 5 = 1

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop := 5*x - 3*y - 13 = 0

-- Theorem statement
theorem chord_bisected_at_P :
  ∀ (A B : ℝ × ℝ),
  is_on_ellipse A.1 A.2 →
  is_on_ellipse B.1 B.2 →
  chord_equation A.1 A.2 →
  chord_equation B.1 B.2 →
  chord_equation P.1 P.2 →
  (A.1 + B.1) / 2 = P.1 ∧
  (A.2 + B.2) / 2 = P.2 :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_at_P_l532_53249


namespace NUMINAMATH_CALUDE_hotel_light_bulbs_l532_53255

theorem hotel_light_bulbs 
  (I F : ℕ) -- I: number of incandescent bulbs, F: number of fluorescent bulbs
  (h_positive : I > 0 ∧ F > 0) -- ensure positive numbers of bulbs
  (h_incandescent_on : (3 : ℝ) / 10 * I = (1 : ℝ) / 7 * (7 : ℝ) / 10 * (I + F)) -- 30% of incandescent on, which is 1/7 of all on bulbs
  (h_total_on : (7 : ℝ) / 10 * (I + F) = (3 : ℝ) / 10 * I + x * F) -- 70% of all bulbs are on
  (x : ℝ) -- x is the fraction of fluorescent bulbs that are on
  : x = (9 : ℝ) / 10 := by
sorry

end NUMINAMATH_CALUDE_hotel_light_bulbs_l532_53255


namespace NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_hyperbola_l532_53276

theorem largest_y_coordinate_degenerate_hyperbola : 
  ∀ (x y : ℝ), x^2 / 49 - (y - 3)^2 / 25 = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_hyperbola_l532_53276


namespace NUMINAMATH_CALUDE_sqrt_19_between_4_and_5_l532_53207

theorem sqrt_19_between_4_and_5 : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_19_between_4_and_5_l532_53207


namespace NUMINAMATH_CALUDE_line_projections_parallel_implies_parallel_or_skew_l532_53270

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Projection of a line onto a plane -/
def project_line (l : Line3D) (p : Plane3D) : Line3D :=
  sorry

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for skew lines -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_projections_parallel_implies_parallel_or_skew 
  (a b : Line3D) (α : Plane3D) :
  parallel (project_line a α) (project_line b α) →
  parallel a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_line_projections_parallel_implies_parallel_or_skew_l532_53270


namespace NUMINAMATH_CALUDE_fourth_power_plus_64_solutions_l532_53293

theorem fourth_power_plus_64_solutions :
  let solutions : Set ℂ := {2 + 2*I, -2 - 2*I, -2 + 2*I, 2 - 2*I}
  ∀ z : ℂ, z^4 + 64 = 0 ↔ z ∈ solutions :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_power_plus_64_solutions_l532_53293


namespace NUMINAMATH_CALUDE_unique_prime_product_l532_53203

theorem unique_prime_product (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q →
  p * q * r = 2014 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_product_l532_53203


namespace NUMINAMATH_CALUDE_simplify_fraction_l532_53229

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) :
  (m - 1) / m / ((m - 1) / (m^2)) = m := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l532_53229


namespace NUMINAMATH_CALUDE_triangle_area_l532_53247

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3/2 when c = √2, b = √6, and B = 120°. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l532_53247


namespace NUMINAMATH_CALUDE_intersection_points_polar_equations_l532_53262

/-- The number of intersection points between r = 3 cos θ and r = 6 sin θ -/
theorem intersection_points_polar_equations : ∃ (n : ℕ), n = 2 ∧
  ∀ (x y : ℝ),
    ((x - 3/2)^2 + y^2 = 9/4 ∨ x^2 + (y - 3)^2 = 9) →
    (∃ (θ : ℝ), 
      (x = 3 * Real.cos θ * Real.cos θ ∧ y = 3 * Real.sin θ * Real.cos θ) ∨
      (x = 6 * Real.sin θ * Real.cos θ ∧ y = 6 * Real.sin θ * Real.sin θ)) :=
by sorry


end NUMINAMATH_CALUDE_intersection_points_polar_equations_l532_53262


namespace NUMINAMATH_CALUDE_min_value_of_expression_l532_53277

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  let A := (a^3 + b^3) / (8*a*b + 9 - c^2) + 
           (b^3 + c^3) / (8*b*c + 9 - a^2) + 
           (c^3 + a^3) / (8*c*a + 9 - b^2)
  ∀ x, A ≥ x → x ≤ 3/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l532_53277


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l532_53243

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℤ, x^3 < 1)) ↔ (∃ x : ℤ, x^3 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l532_53243


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l532_53213

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took from the drawer -/
def pencils_taken : ℕ := 22

/-- The number of pencils remaining in the drawer -/
def remaining_pencils : ℕ := initial_pencils - pencils_taken

theorem pencils_in_drawer : remaining_pencils = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l532_53213


namespace NUMINAMATH_CALUDE_point_A_on_curve_l532_53297

/-- The equation of curve C is x^2 + x + y - 1 = 0 -/
def curve_equation (x y : ℝ) : Prop := x^2 + x + y - 1 = 0

/-- Point A has coordinates (0, 1) -/
def point_A : ℝ × ℝ := (0, 1)

/-- Theorem: Point A lies on curve C -/
theorem point_A_on_curve : curve_equation point_A.1 point_A.2 := by sorry

end NUMINAMATH_CALUDE_point_A_on_curve_l532_53297


namespace NUMINAMATH_CALUDE_prime_characterization_l532_53284

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, 2 ≤ k → k < n → ¬(k ∣ n)

theorem prime_characterization (n : ℕ) :
  Nat.Prime n ↔ is_prime n := by
  sorry

end NUMINAMATH_CALUDE_prime_characterization_l532_53284


namespace NUMINAMATH_CALUDE_merchant_profit_l532_53200

theorem merchant_profit (C S : ℝ) (h : 18 * C = 16 * S) : 
  (S - C) / C * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l532_53200


namespace NUMINAMATH_CALUDE_existence_of_negative_value_l532_53210

theorem existence_of_negative_value (a b c d : ℝ) 
  (h_not_all_zero : b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) :
  let f : ℝ → ℝ := λ x => a + b * Real.cos (2 * x) + c * Real.sin (5 * x) + d * Real.cos (8 * x)
  ∃ t : ℝ, f t = 4 * a → ∃ s : ℝ, f s < 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_negative_value_l532_53210


namespace NUMINAMATH_CALUDE_ternary_to_decimal_l532_53296

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

def ternary_number : List Nat := [1, 0, 2, 0, 1, 2]

theorem ternary_to_decimal :
  to_decimal ternary_number 3 = 320 := by
  sorry

end NUMINAMATH_CALUDE_ternary_to_decimal_l532_53296


namespace NUMINAMATH_CALUDE_problem_solution_l532_53282

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - 3*k) * (x + 3*k) = x^3 + 3*k*(x^2 - x - 7)) →
  k = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l532_53282


namespace NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l532_53254

def f (n : ℕ+) : ℤ := (n : ℤ)^3 - 9*(n : ℤ)^2 + 23*(n : ℤ) - 17

theorem not_prime_for_all_positive_n : ∀ n : ℕ+, ¬(Nat.Prime (Int.natAbs (f n))) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_for_all_positive_n_l532_53254


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l532_53253

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y = 1/x + 4/y + 8) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1/a + 4/b + 8 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l532_53253


namespace NUMINAMATH_CALUDE_ratio_problem_l532_53259

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 1 / 4) :
  e / f = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l532_53259


namespace NUMINAMATH_CALUDE_vote_ratio_l532_53227

/-- Given a total of 60 votes and Ben receiving 24 votes, 
    prove that the ratio of votes received by Ben to votes received by Matt is 2:3 -/
theorem vote_ratio (total_votes : Nat) (ben_votes : Nat) 
    (h1 : total_votes = 60) 
    (h2 : ben_votes = 24) : 
  ∃ (matt_votes : Nat), 
    matt_votes = total_votes - ben_votes ∧ 
    (ben_votes : ℚ) / (matt_votes : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_vote_ratio_l532_53227


namespace NUMINAMATH_CALUDE_bottle_caps_difference_l532_53224

/-- Represents the number of bottle caps in various states of Danny's collection --/
structure BottleCaps where
  thrown_away : ℕ
  found : ℕ
  final_count : ℕ

/-- Theorem stating the difference between found and thrown away bottle caps --/
theorem bottle_caps_difference (caps : BottleCaps)
  (h1 : caps.thrown_away = 6)
  (h2 : caps.found = 50)
  (h3 : caps.final_count = 60)
  : caps.found - caps.thrown_away = 44 := by
  sorry

#check bottle_caps_difference

end NUMINAMATH_CALUDE_bottle_caps_difference_l532_53224


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l532_53268

/-- The speed of a man rowing a boat in still water, given downstream conditions. -/
theorem mans_speed_in_still_water (current_speed : ℝ) (distance : ℝ) (time : ℝ) :
  current_speed = 8 →
  distance = 40 →
  time = 4.499640028797696 →
  ∃ (speed_still_water : ℝ), 
    abs (speed_still_water - ((distance / time) - (current_speed * 1000 / 3600))) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l532_53268


namespace NUMINAMATH_CALUDE_line_equation_proof_l532_53289

-- Define the point A
def A : ℝ × ℝ := (-1, 4)

-- Define the x-intercept
def x_intercept : ℝ := 3

-- Theorem statement
theorem line_equation_proof :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x + y - 3 = 0)) ∧ 
    (A.2 = m * A.1 + b) ∧
    (0 = m * x_intercept + b) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l532_53289


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l532_53273

theorem units_digit_of_fraction : 
  (30 * 31 * 32 * 33 * 34 * 35) / 1500 ≡ 2 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l532_53273


namespace NUMINAMATH_CALUDE_lcm_minus_gcd_equals_34_l532_53286

theorem lcm_minus_gcd_equals_34 : Nat.lcm 40 8 - Nat.gcd 24 54 = 34 := by
  sorry

end NUMINAMATH_CALUDE_lcm_minus_gcd_equals_34_l532_53286


namespace NUMINAMATH_CALUDE_parabola_roots_l532_53272

/-- Given a parabola y = ax² + bx + c with a ≠ 0, axis of symmetry x = -2, 
    and passing through (1, 0), prove its roots are -5 and 1 -/
theorem parabola_roots (a b c : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -5 ∨ x = 1) ↔ 
  (a * (-2)^2 + b * (-2) + c = a * (-2 + 3)^2 + b * (-2 + 3) + c) ∧
  (a * 1^2 + b * 1 + c = 0) := by
sorry

end NUMINAMATH_CALUDE_parabola_roots_l532_53272


namespace NUMINAMATH_CALUDE_pascal_triangle_complete_residue_l532_53242

theorem pascal_triangle_complete_residue (p : ℕ) (hp : Prime p) :
  ∃ n : ℕ, n ≤ p^2 ∧
    ∀ k : ℕ, k < p → ∃ j : ℕ, j ≤ n ∧ (Nat.choose n j) % p = k := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_complete_residue_l532_53242


namespace NUMINAMATH_CALUDE_first_discount_percentage_l532_53225

/-- Proves that the first discount is 15% given the initial price, final price, and second discount rate -/
theorem first_discount_percentage (initial_price final_price : ℝ) (second_discount : ℝ) :
  initial_price = 400 →
  final_price = 323 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    first_discount = 0.15 ∧
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l532_53225


namespace NUMINAMATH_CALUDE_erin_savings_days_l532_53241

/-- The daily amount Erin receives in dollars -/
def daily_amount : ℕ := 3

/-- The total amount Erin needs to receive in dollars -/
def total_amount : ℕ := 30

/-- The number of days it takes Erin to receive the total amount -/
def days_to_total : ℕ := total_amount / daily_amount

theorem erin_savings_days : days_to_total = 10 := by
  sorry

end NUMINAMATH_CALUDE_erin_savings_days_l532_53241


namespace NUMINAMATH_CALUDE_x_x_minus_3_is_quadratic_l532_53215

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x-3) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 3)

/-- Theorem: x(x-3) = 0 is a quadratic equation in one variable -/
theorem x_x_minus_3_is_quadratic : is_quadratic_equation_in_one_variable f := by
  sorry


end NUMINAMATH_CALUDE_x_x_minus_3_is_quadratic_l532_53215


namespace NUMINAMATH_CALUDE_log_equation_solution_l532_53251

theorem log_equation_solution :
  ∃! x : ℝ, Real.log (3 * x + 4) = 1 :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l532_53251


namespace NUMINAMATH_CALUDE_P_root_nature_l532_53232

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^5 - 4*x^4 - 6*x^3 - x + 8

-- Theorem stating that P(x) has no negative roots and at least one positive root
theorem P_root_nature :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) := by
  sorry


end NUMINAMATH_CALUDE_P_root_nature_l532_53232


namespace NUMINAMATH_CALUDE_smallest_divisor_of_7614_l532_53265

def n : ℕ := 7614

theorem smallest_divisor_of_7614 :
  ∃ (d : ℕ), d > 1 ∧ d ∣ n ∧ ∀ (k : ℕ), 1 < k ∧ k ∣ n → d ≤ k :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_7614_l532_53265


namespace NUMINAMATH_CALUDE_jane_max_tickets_l532_53238

/-- The maximum number of tickets that can be bought with a given budget, 
    given a regular price, discounted price, and discount threshold. -/
def maxTickets (budget : ℕ) (regularPrice discountPrice : ℕ) (discountThreshold : ℕ) : ℕ :=
  let regularTickets := budget / regularPrice
  let discountedTotal := 
    discountThreshold * regularPrice + 
    (budget - discountThreshold * regularPrice) / discountPrice
  max regularTickets discountedTotal

/-- Theorem: Given the specific conditions of the problem, 
    the maximum number of tickets Jane can buy is 11. -/
theorem jane_max_tickets : 
  maxTickets 150 15 12 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l532_53238


namespace NUMINAMATH_CALUDE_floor_of_4_7_l532_53275

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l532_53275


namespace NUMINAMATH_CALUDE_cuboid_faces_at_vertex_l532_53260

/-- A cuboid is a three-dimensional shape with six rectangular faces. -/
structure Cuboid where
  -- We don't need to define the specific properties of a cuboid for this problem

/-- The number of faces meeting at one vertex of a cuboid -/
def faces_at_vertex (c : Cuboid) : ℕ := 3

/-- Theorem: The number of faces meeting at one vertex of a cuboid is 3 -/
theorem cuboid_faces_at_vertex (c : Cuboid) : faces_at_vertex c = 3 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_faces_at_vertex_l532_53260


namespace NUMINAMATH_CALUDE_triple_application_equals_six_l532_53246

/-- The function f defined as f(p) = 2p - 20 --/
def f (p : ℝ) : ℝ := 2 * p - 20

/-- Theorem stating that there exists a unique real number p such that f(f(f(p))) = 6 --/
theorem triple_application_equals_six :
  ∃! p : ℝ, f (f (f p)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_application_equals_six_l532_53246


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l532_53218

-- Define a triangle with angles in 2:3:4 ratio
def triangle_with_ratio (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b = (3/2) * a ∧ c = 2 * a ∧
  a + b + c = 180

-- Theorem statement
theorem smallest_angle_measure (a b c : ℝ) 
  (h : triangle_with_ratio a b c) : a = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l532_53218


namespace NUMINAMATH_CALUDE_cannot_compare_full_mark_students_l532_53206

/-- Represents a school with a total number of students and full-mark scorers -/
structure School where
  total_students : ℕ
  full_mark_students : ℕ
  h_full_mark_valid : full_mark_students ≤ total_students

/-- The percentage of full-mark scorers in a school -/
def full_mark_percentage (s : School) : ℚ :=
  (s.full_mark_students : ℚ) / (s.total_students : ℚ) * 100

theorem cannot_compare_full_mark_students
  (school_A school_B : School)
  (h_A : full_mark_percentage school_A = 1)
  (h_B : full_mark_percentage school_B = 2) :
  ¬ (∀ (s₁ s₂ : School),
    full_mark_percentage s₁ = 1 →
    full_mark_percentage s₂ = 2 →
    (s₁.full_mark_students < s₂.full_mark_students ∨
     s₁.full_mark_students > s₂.full_mark_students ∨
     s₁.full_mark_students = s₂.full_mark_students)) :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_compare_full_mark_students_l532_53206


namespace NUMINAMATH_CALUDE_equation_system_solution_l532_53217

def equation_system (x y z : ℝ) : Prop :=
  x^2 + y + z = 1 ∧ x + y^2 + z = 1 ∧ x + y + z^2 = 1

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 0, 0), (0, 1, 0), (0, 0, 1), 
   (-1 - Real.sqrt 2, -1 - Real.sqrt 2, -1 - Real.sqrt 2),
   (-1 + Real.sqrt 2, -1 + Real.sqrt 2, -1 + Real.sqrt 2)}

theorem equation_system_solution :
  ∀ x y z : ℝ, equation_system x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_l532_53217


namespace NUMINAMATH_CALUDE_valid_parameterizations_l532_53236

/-- The slope of the line -/
def m : ℚ := 5 / 3

/-- The y-intercept of the line -/
def b : ℚ := 1

/-- The line equation: y = mx + b -/
def line_equation (x y : ℚ) : Prop := y = m * x + b

/-- A parameterization of a line -/
structure Parameterization where
  initial_point : ℚ × ℚ
  direction_vector : ℚ × ℚ

/-- Check if a parameterization is valid for the given line -/
def is_valid_parameterization (p : Parameterization) : Prop :=
  let (x₀, y₀) := p.initial_point
  let (dx, dy) := p.direction_vector
  line_equation x₀ y₀ ∧ dy / dx = m

/-- The five given parameterizations -/
def param_A : Parameterization := ⟨(3, 6), (3, 5)⟩
def param_B : Parameterization := ⟨(0, 1), (5, 3)⟩
def param_C : Parameterization := ⟨(1, 8/3), (5, 3)⟩
def param_D : Parameterization := ⟨(-1, -2/3), (3, 5)⟩
def param_E : Parameterization := ⟨(1, 1), (5, 8)⟩

theorem valid_parameterizations :
  is_valid_parameterization param_A ∧
  ¬is_valid_parameterization param_B ∧
  ¬is_valid_parameterization param_C ∧
  is_valid_parameterization param_D ∧
  ¬is_valid_parameterization param_E :=
sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l532_53236


namespace NUMINAMATH_CALUDE_line_intersects_segment_slope_range_l532_53240

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (-2, -1)

-- Define the line l
def l (k : ℝ) (x : ℝ) : ℝ := k * (x - 2) + 1

-- Define the segment AB
def segmentAB (t : ℝ) : ℝ × ℝ := (
  (1 - t) * A.1 + t * B.1,
  (1 - t) * A.2 + t * B.2
)

-- Theorem statement
theorem line_intersects_segment_slope_range :
  ∀ k : ℝ, (∃ t ∈ (Set.Icc 0 1), l k (segmentAB t).1 = (segmentAB t).2) →
  -2 ≤ k ∧ k ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_segment_slope_range_l532_53240


namespace NUMINAMATH_CALUDE_discount_difference_is_187_point_5_l532_53252

def initial_amount : ℝ := 15000

def single_discount_rate : ℝ := 0.3
def first_successive_discount_rate : ℝ := 0.25
def second_successive_discount_rate : ℝ := 0.05

def single_discount_amount : ℝ := initial_amount * (1 - single_discount_rate)

def successive_discount_amount : ℝ :=
  initial_amount * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)

theorem discount_difference_is_187_point_5 :
  successive_discount_amount - single_discount_amount = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_is_187_point_5_l532_53252


namespace NUMINAMATH_CALUDE_greatest_x_under_conditions_l532_53266

theorem greatest_x_under_conditions (x : ℕ) 
  (h1 : x > 0) 
  (h2 : ∃ k : ℕ, x = 5 * k) 
  (h3 : x^3 < 1331) : 
  ∀ y : ℕ, (y > 0 ∧ (∃ m : ℕ, y = 5 * m) ∧ y^3 < 1331) → y ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_under_conditions_l532_53266


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l532_53280

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3*x} = {x : ℝ | x ≥ 2}) → a = 6 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l532_53280


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l532_53245

theorem square_area_from_diagonal (d : ℝ) (h : d = 3.8) :
  (d^2 / 2) = 7.22 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l532_53245


namespace NUMINAMATH_CALUDE_triangle_problem_l532_53205

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * (b^2 + c^2 - a^2) →
  (1/2) * b * c * Real.sin A = 3/2 →
  (A = π/3) ∧
  ((b*c - 4*Real.sqrt 3) * Real.cos A + a*c * Real.cos B) / (a^2 - b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l532_53205


namespace NUMINAMATH_CALUDE_blue_card_value_is_five_l532_53263

/-- The value of a blue card in credits -/
def blue_card_value (total_credits : ℕ) (total_cards : ℕ) (red_card_value : ℕ) (red_cards : ℕ) : ℕ :=
  (total_credits - red_card_value * red_cards) / (total_cards - red_cards)

/-- Theorem stating that the value of a blue card is 5 credits -/
theorem blue_card_value_is_five :
  blue_card_value 84 20 3 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_card_value_is_five_l532_53263


namespace NUMINAMATH_CALUDE_BF_length_is_four_l532_53216

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D E F : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_at_A_and_C (q : Quadrilateral) : Prop := sorry
def E_and_F_on_AC (q : Quadrilateral) : Prop := sorry
def DE_perpendicular_to_AC (q : Quadrilateral) : Prop := sorry
def BF_perpendicular_to_AC (q : Quadrilateral) : Prop := sorry

-- Define the given lengths
def AE_length (q : Quadrilateral) : ℝ := 4
def DE_length (q : Quadrilateral) : ℝ := 6
def CE_length (q : Quadrilateral) : ℝ := 6

-- Define the length of BF
def BF_length (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem BF_length_is_four (q : Quadrilateral) 
  (h1 : is_right_angled_at_A_and_C q)
  (h2 : E_and_F_on_AC q)
  (h3 : DE_perpendicular_to_AC q)
  (h4 : BF_perpendicular_to_AC q) :
  BF_length q = 4 := by sorry

end NUMINAMATH_CALUDE_BF_length_is_four_l532_53216


namespace NUMINAMATH_CALUDE_susan_remaining_distance_l532_53279

/-- The total number of spaces on the board game --/
def total_spaces : ℕ := 72

/-- Susan's movements over 5 turns --/
def susan_movements : List ℤ := [12, -3, 0, 4, -3]

/-- The theorem stating the remaining distance Susan needs to move --/
theorem susan_remaining_distance :
  total_spaces - (susan_movements.sum) = 62 := by sorry

end NUMINAMATH_CALUDE_susan_remaining_distance_l532_53279


namespace NUMINAMATH_CALUDE_victors_specific_earnings_l532_53294

/-- Victor's earnings over two days given his hourly wage and hours worked each day -/
def victors_earnings (hourly_wage : ℕ) (hours_monday : ℕ) (hours_tuesday : ℕ) : ℕ :=
  hourly_wage * (hours_monday + hours_tuesday)

/-- Theorem: Victor's earnings over two days given specific conditions -/
theorem victors_specific_earnings :
  victors_earnings 6 5 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_victors_specific_earnings_l532_53294


namespace NUMINAMATH_CALUDE_sequence_general_term_l532_53274

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n + 3) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n + 1) - 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l532_53274


namespace NUMINAMATH_CALUDE_parallel_condition_l532_53256

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ * c₂ ≠ m₂ * c₁

/-- The theorem stating that a=1 is a necessary and sufficient condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  parallel a 2 (-1) 1 2 4 ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_parallel_condition_l532_53256


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l532_53222

theorem one_and_two_thirds_of_number_is_45 : ∃ x : ℚ, (5 / 3) * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_number_is_45_l532_53222


namespace NUMINAMATH_CALUDE_range_of_expressions_l532_53231

theorem range_of_expressions (a b : ℝ) 
  (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) : 
  (8 < 2*a + 3*b ∧ 2*a + 3*b < 32) ∧ 
  (-7 < a - b ∧ a - b < 2) := by
sorry

end NUMINAMATH_CALUDE_range_of_expressions_l532_53231


namespace NUMINAMATH_CALUDE_happy_point_range_l532_53298

theorem happy_point_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) (-3/2), a * x^2 - 2*x - 2*a - 3/2 = -x) →
  a ∈ Set.Icc (-1/4 : ℝ) 0 := by
sorry

end NUMINAMATH_CALUDE_happy_point_range_l532_53298


namespace NUMINAMATH_CALUDE_sphere_cube_surface_area_comparison_l532_53244

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

noncomputable def cube_volume (a : ℝ) : ℝ := a^3
noncomputable def cube_surface_area (a : ℝ) : ℝ := 6 * a^2

theorem sphere_cube_surface_area_comparison 
  (r a : ℝ) 
  (h_positive : r > 0 ∧ a > 0) 
  (h_equal_volume : sphere_volume r = cube_volume a) : 
  cube_surface_area a > sphere_surface_area r :=
by
  sorry

#check sphere_cube_surface_area_comparison

end NUMINAMATH_CALUDE_sphere_cube_surface_area_comparison_l532_53244


namespace NUMINAMATH_CALUDE_number_wall_solve_l532_53281

/-- Represents a row in the Number Wall -/
structure NumberWallRow :=
  (left : ℤ) (middle_left : ℤ) (middle_right : ℤ) (right : ℤ)

/-- Defines the Number Wall structure and rules -/
def NumberWall (bottom : NumberWallRow) : Prop :=
  ∃ (second : NumberWallRow) (third : NumberWallRow) (top : ℤ),
    second.left = bottom.left + bottom.middle_left
    ∧ second.middle_left = bottom.middle_left + bottom.middle_right
    ∧ second.middle_right = bottom.middle_right + bottom.right
    ∧ third.left = second.left + second.middle_left
    ∧ third.right = second.middle_right + second.right
    ∧ top = third.left + third.right
    ∧ top = 36

/-- The main theorem to prove -/
theorem number_wall_solve :
  ∀ m : ℤ, NumberWall ⟨m, 6, 12, 10⟩ → m = -28 :=
by sorry

end NUMINAMATH_CALUDE_number_wall_solve_l532_53281


namespace NUMINAMATH_CALUDE_sons_age_l532_53230

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 5 = 2 * (son_age + 5) →
  son_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l532_53230


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l532_53257

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (7^14 + 11^15) ∧ ∀ (q : ℕ), q.Prime → q ∣ (7^14 + 11^15) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l532_53257


namespace NUMINAMATH_CALUDE_divisibility_for_odd_n_l532_53285

theorem divisibility_for_odd_n (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, (82 : ℤ)^n + 454 * (69 : ℤ)^n = 1963 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_for_odd_n_l532_53285


namespace NUMINAMATH_CALUDE_smallest_disk_not_always_circumcircle_l532_53209

/-- Three noncollinear points in the plane -/
structure ThreePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  noncollinear : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The radius of the smallest disk containing three points -/
def smallest_disk_radius (p : ThreePoints) : ℝ :=
  sorry

/-- The radius of the circumcircle of three points -/
def circumcircle_radius (p : ThreePoints) : ℝ :=
  sorry

/-- Theorem stating that the smallest disk is not always the circumcircle -/
theorem smallest_disk_not_always_circumcircle :
  ∃ p : ThreePoints, smallest_disk_radius p < circumcircle_radius p :=
sorry

end NUMINAMATH_CALUDE_smallest_disk_not_always_circumcircle_l532_53209


namespace NUMINAMATH_CALUDE_right_triangle_legs_l532_53202

theorem right_triangle_legs (a b : ℝ) : 
  a > 0 → b > 0 →
  (a^2 + b^2 = 100) →  -- Pythagorean theorem (hypotenuse = 10)
  (a + b = 14) →       -- Derived from inradius and semiperimeter
  (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l532_53202


namespace NUMINAMATH_CALUDE_ab_sufficient_not_necessary_for_a_plus_b_l532_53228

theorem ab_sufficient_not_necessary_for_a_plus_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ a b, a > 0 → b > 0 → a * b > 1 → a + b > 2) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b > 2 ∧ a * b ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_ab_sufficient_not_necessary_for_a_plus_b_l532_53228


namespace NUMINAMATH_CALUDE_alex_money_left_l532_53220

/-- Calculates the amount of money Alex has left after deductions --/
theorem alex_money_left (weekly_income : ℕ) (tax_rate : ℚ) (water_bill : ℕ) (tithe_rate : ℚ) : 
  weekly_income = 500 →
  tax_rate = 1/10 →
  water_bill = 55 →
  tithe_rate = 1/10 →
  ↑weekly_income - (↑weekly_income * tax_rate + ↑water_bill + ↑weekly_income * tithe_rate) = 345 := by
sorry

end NUMINAMATH_CALUDE_alex_money_left_l532_53220


namespace NUMINAMATH_CALUDE_beths_crayon_packs_l532_53211

/-- The number of crayon packs Beth has after distribution and finding more -/
def beths_total_packs (initial_packs : ℚ) (total_friends : ℕ) (new_packs : ℚ) : ℚ :=
  (initial_packs / total_friends) + new_packs

/-- Theorem stating Beth's total packs under the given conditions -/
theorem beths_crayon_packs : 
  beths_total_packs 4 10 6 = 6.4 := by sorry

end NUMINAMATH_CALUDE_beths_crayon_packs_l532_53211


namespace NUMINAMATH_CALUDE_first_car_speed_l532_53233

/-- Represents the scenario of two cars traveling between points A and B -/
structure CarScenario where
  distance_AB : ℝ
  delay : ℝ
  speed_second_car : ℝ
  speed_first_car : ℝ

/-- Checks if the given scenario satisfies all conditions -/
def satisfies_conditions (s : CarScenario) : Prop :=
  s.distance_AB = 40 ∧
  s.delay = 1/3 ∧
  s.speed_second_car = 45 ∧
  ∃ (meeting_point : ℝ),
    0 < meeting_point ∧ meeting_point < s.distance_AB ∧
    (meeting_point / s.speed_second_car + s.delay = meeting_point / s.speed_first_car) ∧
    (s.distance_AB / s.speed_first_car = 
      meeting_point / s.speed_second_car + s.delay + meeting_point / s.speed_second_car + meeting_point / (2 * s.speed_second_car))

/-- The main theorem stating that if a scenario satisfies all conditions, 
    then the speed of the first car must be 30 km/h -/
theorem first_car_speed (s : CarScenario) :
  satisfies_conditions s → s.speed_first_car = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l532_53233


namespace NUMINAMATH_CALUDE_similar_triangle_sum_l532_53223

/-- Given a triangle with sides in ratio 3:5:7 and a similar triangle with longest side 21,
    the sum of the other two sides of the similar triangle is 24. -/
theorem similar_triangle_sum (a b c : ℝ) (x y z : ℝ) :
  a / b = 3 / 5 →
  b / c = 5 / 7 →
  a / c = 3 / 7 →
  x / y = a / b →
  y / z = b / c →
  x / z = a / c →
  z = 21 →
  x + y = 24 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_sum_l532_53223


namespace NUMINAMATH_CALUDE_largest_b_no_real_roots_l532_53221

theorem largest_b_no_real_roots : 
  ∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ 0) → b ≤ 7 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_b_no_real_roots_l532_53221


namespace NUMINAMATH_CALUDE_vertical_shift_equation_line_shift_theorem_l532_53226

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Applies a vertical shift to a linear function -/
def verticalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + shift }

theorem vertical_shift_equation (m : ℝ) (shift : ℝ) :
  let original := LinearFunction.mk m 0
  let shifted := verticalShift original shift
  shifted = LinearFunction.mk m shift := by sorry

/-- The main theorem proving that shifting y = -5x upwards by 2 units results in y = -5x + 2 -/
theorem line_shift_theorem :
  let original := LinearFunction.mk (-5) 0
  let shifted := verticalShift original 2
  shifted = LinearFunction.mk (-5) 2 := by sorry

end NUMINAMATH_CALUDE_vertical_shift_equation_line_shift_theorem_l532_53226


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l532_53267

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-coordinate of the intersection point of a line with the x-axis -/
def x_axis_intersection (l : Line) : ℝ :=
  sorry

theorem line_intersection_x_axis (l : Line) : 
  l.x₁ = 6 ∧ l.y₁ = 22 ∧ l.x₂ = -3 ∧ l.y₂ = 1 → x_axis_intersection l = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l532_53267


namespace NUMINAMATH_CALUDE_largest_three_digit_square_base_9_l532_53261

/-- The largest integer whose square has exactly 3 digits when written in base 9 -/
def N : ℕ := 26

/-- Condition for a number to have exactly 3 digits in base 9 -/
def has_three_digits_base_9 (n : ℕ) : Prop :=
  9^2 ≤ n^2 ∧ n^2 < 9^3

/-- Convert a natural number to its base 9 representation -/
def to_base_9 (n : ℕ) : ℕ :=
  (n / 9) * 10 + (n % 9)

theorem largest_three_digit_square_base_9 :
  (N = 26) ∧
  (has_three_digits_base_9 N) ∧
  (∀ m : ℕ, m > N → ¬(has_three_digits_base_9 m)) ∧
  (to_base_9 N = 28) :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_base_9_l532_53261


namespace NUMINAMATH_CALUDE_greatest_integer_square_thrice_plus_81_l532_53288

theorem greatest_integer_square_thrice_plus_81 :
  ∀ x : ℤ, x^2 = 3*x + 81 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_square_thrice_plus_81_l532_53288


namespace NUMINAMATH_CALUDE_min_cups_in_boxes_min_cups_for_100_boxes_l532_53269

theorem min_cups_in_boxes : ℕ → ℕ
  | n => (n * (n + 1)) / 2

theorem min_cups_for_100_boxes :
  min_cups_in_boxes 100 = 5050 := by sorry

end NUMINAMATH_CALUDE_min_cups_in_boxes_min_cups_for_100_boxes_l532_53269


namespace NUMINAMATH_CALUDE_decimal_is_fraction_l532_53234

def is_fraction (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem decimal_is_fraction :
  let x : ℝ := 0.666
  is_fraction x :=
sorry

end NUMINAMATH_CALUDE_decimal_is_fraction_l532_53234


namespace NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l532_53250

theorem complex_exp_210_deg_60th_power : 
  (Complex.exp (210 * π / 180 * I)) ^ 60 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_exp_210_deg_60th_power_l532_53250


namespace NUMINAMATH_CALUDE_range_of_a_l532_53248

-- Define a decreasing function on (-1, 1)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f y < f x

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) :
  (∀ a, f (2 * a - 1) < f (1 - a)) → 
  (∀ a, (2/3 : ℝ) < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l532_53248


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l532_53287

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l532_53287


namespace NUMINAMATH_CALUDE_slant_base_angle_is_36_degrees_l532_53237

/-- A regular pentagonal pyramid where the slant height is equal to the base edge -/
structure RegularPentagonalPyramid where
  /-- The base of the pyramid is a regular pentagon -/
  base : RegularPentagon
  /-- The slant height of the pyramid -/
  slant_height : ℝ
  /-- The base edge of the pyramid -/
  base_edge : ℝ
  /-- The slant height is equal to the base edge -/
  slant_height_eq_base_edge : slant_height = base_edge

/-- The angle between a slant height and a non-intersecting, non-perpendicular base edge -/
def slant_base_angle (p : RegularPentagonalPyramid) : Angle := sorry

/-- Theorem: The angle between a slant height and a non-intersecting, non-perpendicular base edge is 36° -/
theorem slant_base_angle_is_36_degrees (p : RegularPentagonalPyramid) :
  slant_base_angle p = 36 * π / 180 := by sorry

end NUMINAMATH_CALUDE_slant_base_angle_is_36_degrees_l532_53237


namespace NUMINAMATH_CALUDE_intersection_M_N_l532_53208

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 5}
def N : Set ℝ := {x | x * (x - 4) > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l532_53208


namespace NUMINAMATH_CALUDE_penny_remaining_money_l532_53264

/-- Calculates the remaining money after Penny's shopping trip --/
def remaining_money (initial_amount : ℚ) (sock_price : ℚ) (sock_quantity : ℕ)
  (hat_price : ℚ) (hat_quantity : ℕ) (scarf_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := sock_price * sock_quantity + hat_price * hat_quantity + scarf_price
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that Penny has $14 left after her purchases --/
theorem penny_remaining_money :
  remaining_money 50 4 3 10 2 8 (1/10) = 14 := by
  sorry

end NUMINAMATH_CALUDE_penny_remaining_money_l532_53264


namespace NUMINAMATH_CALUDE_gcd_pow_minus_one_l532_53283

theorem gcd_pow_minus_one (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_pow_minus_one_l532_53283


namespace NUMINAMATH_CALUDE_rooks_diagonal_move_l532_53239

/-- Represents a position on an 8x8 chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a configuration of 8 rooks on an 8x8 chessboard -/
structure RookConfiguration :=
  (positions : Fin 8 → Position)
  (no_attacks : ∀ i j, i ≠ j → 
    (positions i).row ≠ (positions j).row ∧ 
    (positions i).col ≠ (positions j).col)

/-- Checks if a position is adjacent diagonally to another position -/
def is_adjacent_diagonal (p1 p2 : Position) : Prop :=
  (p1.row.val + 1 = p2.row.val ∧ p1.col.val + 1 = p2.col.val) ∨
  (p1.row.val + 1 = p2.row.val ∧ p1.col.val = p2.col.val + 1) ∨
  (p1.row.val = p2.row.val + 1 ∧ p1.col.val + 1 = p2.col.val) ∨
  (p1.row.val = p2.row.val + 1 ∧ p1.col.val = p2.col.val + 1)

/-- The main theorem to be proved -/
theorem rooks_diagonal_move (initial : RookConfiguration) :
  ∃ (final : RookConfiguration),
    ∀ i, is_adjacent_diagonal (initial.positions i) (final.positions i) :=
sorry

end NUMINAMATH_CALUDE_rooks_diagonal_move_l532_53239


namespace NUMINAMATH_CALUDE_min_value_expression_l532_53214

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a + b) / c + (a + c) / b + (b + c) / a ≥ 8 ∧
  (2 * (a + b) / c + (a + c) / b + (b + c) / a = 8 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l532_53214


namespace NUMINAMATH_CALUDE_total_money_l532_53292

theorem total_money (brad_money : ℚ) (josh_money : ℚ) (doug_money : ℚ) : 
  josh_money = 2 * brad_money →
  josh_money = (3 / 4) * doug_money →
  doug_money = 32 →
  brad_money + josh_money + doug_money = 68 := by
sorry

end NUMINAMATH_CALUDE_total_money_l532_53292


namespace NUMINAMATH_CALUDE_complex_product_real_l532_53235

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of a complex number being real -/
def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_product_real (m : ℝ) :
  is_real ((2 + i) * (m - 2*i)) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l532_53235


namespace NUMINAMATH_CALUDE_expression_equals_one_l532_53278

theorem expression_equals_one :
  (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l532_53278


namespace NUMINAMATH_CALUDE_polygon_contains_half_unit_segment_l532_53291

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  area : ℝ
  isConvex : Bool

/-- A square with side length 1 -/
structure UnitSquare where
  -- Add necessary fields for a unit square

/-- Represents the placement of a polygon inside a square -/
structure PolygonInSquare where
  polygon : ConvexPolygon
  square : UnitSquare
  isInside : Bool

/-- A line segment -/
structure LineSegment where
  length : ℝ
  isParallelToSquareSide : Bool
  isInsidePolygon : Bool

/-- The main theorem -/
theorem polygon_contains_half_unit_segment 
  (p : PolygonInSquare) 
  (h1 : p.polygon.area > 0.5) 
  (h2 : p.polygon.isConvex) 
  (h3 : p.isInside) :
  ∃ (s : LineSegment), s.length = 0.5 ∧ s.isParallelToSquareSide ∧ s.isInsidePolygon :=
by sorry

end NUMINAMATH_CALUDE_polygon_contains_half_unit_segment_l532_53291


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l532_53295

theorem point_movement_on_number_line (A : ℝ) : 
  A + 7 - 4 = 0 → A = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l532_53295


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_l532_53299

theorem perfect_square_power_of_two (n : ℕ) : 
  (∃ m : ℕ, 2^5 + 2^11 + 2^n = m^2) ↔ n = 12 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_l532_53299


namespace NUMINAMATH_CALUDE_factorial_properties_l532_53201

-- Define ord_p
def ord_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Define S_p
def S_p (p : ℕ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem factorial_properties (n : ℕ) (p : ℕ) (m : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_div : p ^ m ∣ n ∧ ¬(p ^ (m + 1) ∣ n)) 
  (h_ord : ord_p p n = m) : 
  (ord_p p (n.factorial) = (n - S_p p n) / (p - 1)) ∧ 
  (∃ k : ℕ, (2 * n).factorial = k * n.factorial * (n + 1).factorial) ∧
  (Nat.Coprime m (n + 1) → 
    ∃ k : ℕ, (m * n + n).factorial = k * (m * n).factorial * (n + 1).factorial) := by
  sorry

end NUMINAMATH_CALUDE_factorial_properties_l532_53201
