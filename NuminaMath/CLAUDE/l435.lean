import Mathlib

namespace NUMINAMATH_CALUDE_fraction_pair_sum_equality_l435_43532

theorem fraction_pair_sum_equality (n : ℕ) (h : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_pair_sum_equality_l435_43532


namespace NUMINAMATH_CALUDE_unique_solution_is_seven_l435_43500

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem unique_solution_is_seven :
  ∃! n : ℕ, n > 0 ∧ n^2 * factorial n + factorial n = 5040 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_seven_l435_43500


namespace NUMINAMATH_CALUDE_constant_term_position_l435_43559

/-- The position of the constant term in the expansion of (√a - 2/∛a)^30 -/
theorem constant_term_position (a : ℝ) (h : a > 0) : 
  ∃ (r : ℕ), r = 18 ∧ 
  (∀ (k : ℕ), k ≠ r → (90 - 5 * k : ℚ) / 6 ≠ 0) ∧
  (90 - 5 * r : ℚ) / 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_position_l435_43559


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l435_43518

-- Define the room dimensions and paving rate
def room_length : Real := 5.5
def room_width : Real := 3.75
def paving_rate : Real := 400

-- Define the theorem
theorem paving_cost_calculation :
  let area : Real := room_length * room_width
  let cost : Real := area * paving_rate
  cost = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l435_43518


namespace NUMINAMATH_CALUDE_third_side_length_l435_43589

/-- Given a triangle with two sides of lengths 6 and 8, forming a 45-degree angle between them,
    the length of the third side is √(100 - 48√2). -/
theorem third_side_length (a b c θ : ℝ) (ha : a = 6) (hb : b = 8) (hθ : θ = π/4)
  (hc : c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ)) :
  c = Real.sqrt (100 - 48 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_third_side_length_l435_43589


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l435_43553

/-- The diagonal of a rectangle with side lengths 40√3 cm and 30√3 cm is 50√3 cm. -/
theorem rectangle_diagonal (a b d : ℝ) (ha : a = 40 * Real.sqrt 3) (hb : b = 30 * Real.sqrt 3) 
  (hd : d ^ 2 = a ^ 2 + b ^ 2) : d = 50 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_l435_43553


namespace NUMINAMATH_CALUDE_system_solution_l435_43533

theorem system_solution :
  ∃ (x y : ℝ), 3 * x + 2 * y = 19 ∧ 2 * x - y = 1 ∧ x = 3 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l435_43533


namespace NUMINAMATH_CALUDE_journey_fraction_by_foot_l435_43585

/-- Given a journey with a total distance of 24 km, where 1/4 of the distance
    is traveled by bus and 6 km is traveled by car, prove that the fraction
    of the distance traveled by foot is 1/2. -/
theorem journey_fraction_by_foot :
  ∀ (total_distance bus_fraction car_distance foot_distance : ℝ),
    total_distance = 24 →
    bus_fraction = 1/4 →
    car_distance = 6 →
    foot_distance = total_distance - (bus_fraction * total_distance + car_distance) →
    foot_distance / total_distance = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_journey_fraction_by_foot_l435_43585


namespace NUMINAMATH_CALUDE_unique_solution_system_l435_43542

/-- The system of equations has a unique solution when a = 5/3 and no solutions otherwise -/
theorem unique_solution_system (a x y : ℝ) : 
  (3 * (x - a)^2 + y = 2 - a) ∧ 
  (y^2 + ((x - 2) / (|x| - 2))^2 = 1) ∧ 
  (x ≥ 0) ∧ 
  (x ≠ 2) ↔ 
  (a = 5/3 ∧ x = 4/3 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l435_43542


namespace NUMINAMATH_CALUDE_quadratic_bound_l435_43504

theorem quadratic_bound (a b c : ℝ) (h1 : c > 0) (h2 : |a - b + c| ≤ 1) (h3 : |a + b + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ c + 1 / (4 * c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_bound_l435_43504


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_120_l435_43537

theorem product_of_consecutive_integers_120 : 
  ∃ (a b c d e : ℤ), 
    (b = a + 1) ∧ 
    (a * b = 120) ∧ 
    (d = c + 1) ∧ 
    (e = d + 1) ∧ 
    (c * d * e = 120) ∧ 
    (a + b + c + d + e = 36) :=
by sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_120_l435_43537


namespace NUMINAMATH_CALUDE_jerrys_age_l435_43570

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 20 → 
  mickey_age = 2 * jerry_age - 8 → 
  jerry_age = 14 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l435_43570


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l435_43535

theorem quadratic_one_solution (q : ℝ) : 
  q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 18 * x + 8 = 0) ↔ q = 81/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l435_43535


namespace NUMINAMATH_CALUDE_star_value_l435_43558

-- Define the * operation
def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

-- State the theorem
theorem star_value (a b : ℝ) :
  (star a b 3 5 = 2011) → (star a b 4 9 = 2009) → (star a b 1 2 = 2010) := by
  sorry

end NUMINAMATH_CALUDE_star_value_l435_43558


namespace NUMINAMATH_CALUDE_abs_value_inequality_l435_43519

theorem abs_value_inequality (x : ℝ) : 
  (|x - 2| + |x - 4| ≤ 3) ↔ (3/2 ≤ x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_abs_value_inequality_l435_43519


namespace NUMINAMATH_CALUDE_baker_pastries_l435_43515

/-- Given that Baker made 43 cakes, sold 154 pastries and 78 cakes,
    and sold 76 more pastries than cakes, prove that Baker made 154 pastries. -/
theorem baker_pastries :
  let cakes_made : ℕ := 43
  let pastries_sold : ℕ := 154
  let cakes_sold : ℕ := 78
  let difference : ℕ := 76
  pastries_sold = cakes_sold + difference →
  pastries_sold = 154
:= by sorry

end NUMINAMATH_CALUDE_baker_pastries_l435_43515


namespace NUMINAMATH_CALUDE_final_value_one_fourth_l435_43506

theorem final_value_one_fourth (x : ℝ) : 
  (1 / 4) * ((5 * x + 3) - 1) = (5 * x) / 4 + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_final_value_one_fourth_l435_43506


namespace NUMINAMATH_CALUDE_probability_of_cooking_l435_43587

/-- The set of courses Xiao Ming is interested in -/
inductive Course
| Planting
| Cooking
| Pottery
| Woodworking

/-- The probability of selecting a specific course from the set of courses -/
def probability_of_course (c : Course) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of selecting "Cooking" is 1/4 -/
theorem probability_of_cooking :
  probability_of_course Course.Cooking = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_cooking_l435_43587


namespace NUMINAMATH_CALUDE_linear_function_k_value_l435_43536

/-- Proves that for the linear function y = kx + 3 passing through the point (2, 5), the value of k is 1. -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3) → -- Condition 1: The function is y = kx + 3
  (5 : ℝ) = k * 2 + 3 →        -- Condition 2: The function passes through the point (2, 5)
  k = 1 :=                     -- Conclusion: The value of k is 1
by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l435_43536


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l435_43555

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) (ha : a ≠ 0) :
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a := by
  sorry

-- Problem 2
theorem simplify_expression_2 :
  (25^(1/3) - 125^(1/2)) / 25^(1/4) = 5^(1/6) - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l435_43555


namespace NUMINAMATH_CALUDE_equation_solution_l435_43543

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l435_43543


namespace NUMINAMATH_CALUDE_avg_cost_rounded_to_13_l435_43526

/- Define the number of pencils -/
def num_pencils : ℕ := 200

/- Define the cost of pencils in cents -/
def pencil_cost : ℕ := 1990

/- Define the shipping cost in cents -/
def shipping_cost : ℕ := 695

/- Define the function to calculate the average cost per pencil in cents -/
def avg_cost_per_pencil : ℚ :=
  (pencil_cost + shipping_cost : ℚ) / num_pencils

/- Define the function to round to the nearest whole number -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/- Theorem statement -/
theorem avg_cost_rounded_to_13 :
  round_to_nearest avg_cost_per_pencil = 13 := by
  sorry


end NUMINAMATH_CALUDE_avg_cost_rounded_to_13_l435_43526


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l435_43594

theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : ∀ n, S n = n / 2 * (a 1 + a n))
  (h2 : a 7 / a 4 = 2) :
  S 13 / S 7 = 26 / 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l435_43594


namespace NUMINAMATH_CALUDE_key_sequence_produces_desired_output_l435_43541

/-- Represents the mapping of keys to displayed letters on the magical keyboard. -/
def keyboard_mapping : Char → Char
| 'Q' => 'A'
| 'S' => 'D'
| 'D' => 'S'
| 'J' => 'H'
| 'K' => 'O'
| 'L' => 'P'
| 'R' => 'E'
| 'N' => 'M'
| 'Y' => 'T'
| c => c  -- For all other characters, map to themselves

/-- The sequence of key presses -/
def key_sequence : List Char := ['J', 'K', 'L', 'R', 'N', 'Q', 'Y', 'J']

/-- The desired display output -/
def desired_output : List Char := ['H', 'O', 'P', 'E', 'M', 'A', 'T', 'H']

/-- Theorem stating that the key sequence produces the desired output -/
theorem key_sequence_produces_desired_output :
  key_sequence.map keyboard_mapping = desired_output := by
  sorry

#eval key_sequence.map keyboard_mapping

end NUMINAMATH_CALUDE_key_sequence_produces_desired_output_l435_43541


namespace NUMINAMATH_CALUDE_tan_difference_of_angles_l435_43516

theorem tan_difference_of_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α - Real.sin β = -1/2) (h4 : Real.cos α - Real.cos β = 1/2) :
  Real.tan (α - β) = -Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_of_angles_l435_43516


namespace NUMINAMATH_CALUDE_triangle_midpoint_x_sum_l435_43512

theorem triangle_midpoint_x_sum (a b c : ℝ) (S : ℝ) : 
  a + b + c = S → 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = S :=
by sorry

end NUMINAMATH_CALUDE_triangle_midpoint_x_sum_l435_43512


namespace NUMINAMATH_CALUDE_range_of_m_range_of_t_l435_43549

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

def q (m : ℝ) : Prop := m^2 > 2*m + 8 ∧ 2*m + 8 > 0

def s (m t : ℝ) : Prop := t < m ∧ m < t + 1

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (p m ∧ q m) ↔ ((-4 < m ∧ m < -2) ∨ m > 4) :=
sorry

-- Theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∀ m : ℝ, s m t → q m) ∧ (∃ m : ℝ, q m ∧ ¬s m t) ↔
  ((-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_t_l435_43549


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l435_43524

-- Define the points
def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (-2, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AD : ℝ × ℝ := (D.1 - A.1, D.2 - A.2)
def DC : ℝ × ℝ := (C.1 - D.1, C.2 - D.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicular
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Define parallel
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Define trapezoid
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  parallel (B.1 - A.1, B.2 - A.2) (D.1 - C.1, D.2 - C.2) ∧
  ¬parallel (A.1 - D.1, A.2 - D.2) (B.1 - C.1, B.2 - C.2)

theorem quadrilateral_properties :
  perpendicular AB AD ∧ parallel AB DC ∧ is_trapezoid A B C D := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l435_43524


namespace NUMINAMATH_CALUDE_office_age_problem_l435_43571

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) 
  (group1_size : ℕ) (group2_size : ℕ) (avg_age_group2 : ℝ) 
  (age_person15 : ℕ) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_size = 5 →
  group2_size = 9 →
  avg_age_group2 = 16 →
  age_person15 = 56 →
  (total_persons * avg_age_all - group2_size * avg_age_group2 - age_person15) / group1_size = 14 := by
sorry

end NUMINAMATH_CALUDE_office_age_problem_l435_43571


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l435_43501

/-- The focal distance of the hyperbola 2x^2 - y^2 = 6 is 6 -/
theorem hyperbola_focal_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | 2 * x^2 - y^2 = 6}
  ∃ f : ℝ, f = 6 ∧ ∀ (x y : ℝ), (x, y) ∈ hyperbola →
    ∃ (F₁ F₂ : ℝ × ℝ), abs (x - F₁.1) + abs (x - F₂.1) = 2 * f :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l435_43501


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l435_43547

theorem polygon_interior_angles (n : ℕ) : 
  (n ≥ 3) → 
  (2005 + 180 = (n - 2) * 180) → 
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l435_43547


namespace NUMINAMATH_CALUDE_target_probability_l435_43595

theorem target_probability (p : ℝ) : 
  (1 - (1 - p)^3 = 0.875) → p = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_target_probability_l435_43595


namespace NUMINAMATH_CALUDE_kelly_cheese_days_l435_43544

/-- The number of weeks Kelly needs to cover -/
def weeks : ℕ := 4

/-- The number of packages of string cheese Kelly buys -/
def packages : ℕ := 2

/-- The number of string cheeses in each package -/
def cheeses_per_package : ℕ := 30

/-- The number of string cheeses the oldest child needs per day -/
def oldest_child_cheeses : ℕ := 2

/-- The number of string cheeses the youngest child needs per day -/
def youngest_child_cheeses : ℕ := 1

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: Kelly puts string cheeses in her kids' lunches 5 days per week -/
theorem kelly_cheese_days : 
  (packages * cheeses_per_package) / (oldest_child_cheeses + youngest_child_cheeses) / weeks = 5 := by
  sorry

end NUMINAMATH_CALUDE_kelly_cheese_days_l435_43544


namespace NUMINAMATH_CALUDE_cubic_roots_product_l435_43580

theorem cubic_roots_product (a b c : ℝ) : 
  (x^3 - 26*x^2 + 32*x - 15 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 74 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l435_43580


namespace NUMINAMATH_CALUDE_isosceles_trajectory_equation_l435_43531

/-- An isosceles triangle ABC with vertices A(3,20) and B(3,5) -/
structure IsoscelesTriangle where
  C : ℝ × ℝ
  isIsosceles : (C.1 - 3)^2 + (C.2 - 20)^2 = (3 - 3)^2 + (5 - 20)^2
  notCollinear : C.1 ≠ 3

/-- The trajectory equation of point C in an isosceles triangle ABC -/
def trajectoryEquation (t : IsoscelesTriangle) : Prop :=
  (t.C.1 - 3)^2 + (t.C.2 - 20)^2 = 225

/-- Theorem: The trajectory equation holds for any isosceles triangle satisfying the given conditions -/
theorem isosceles_trajectory_equation (t : IsoscelesTriangle) : trajectoryEquation t := by
  sorry


end NUMINAMATH_CALUDE_isosceles_trajectory_equation_l435_43531


namespace NUMINAMATH_CALUDE_distance_MN_equals_5_l435_43525

def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 5)

theorem distance_MN_equals_5 : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_MN_equals_5_l435_43525


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l435_43502

theorem polynomial_product_expansion (x : ℝ) :
  (x^3 - 3*x^2 + 3*x - 1) * (x^2 + 3*x + 3) = x^5 - 3*x^2 + 6*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l435_43502


namespace NUMINAMATH_CALUDE_plane_equation_proof_l435_43505

def plane_equation (w : ℝ × ℝ × ℝ) (s t : ℝ) : Prop :=
  w = (2 + 2*s - 3*t, 4 - 2*s, 1 - s + 3*t)

theorem plane_equation_proof :
  ∃ (A B C D : ℤ),
    (∀ x y z : ℝ, (∃ s t : ℝ, plane_equation (x, y, z) s t) ↔ A * x + B * y + C * z + D = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 2 ∧ B = -1 ∧ C = 2 ∧ D = -2 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l435_43505


namespace NUMINAMATH_CALUDE_solve_equation_l435_43592

theorem solve_equation (b : ℚ) (h : b + b/4 = 5/2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l435_43592


namespace NUMINAMATH_CALUDE_soy_sauce_bottles_l435_43572

/-- Represents the amount of soy sauce in ounces -/
def OuncesPerBottle : ℕ := 16

/-- Represents the number of ounces in a cup -/
def OuncesPerCup : ℕ := 8

/-- Represents the amount of soy sauce needed for each recipe in cups -/
def RecipeCups : List ℕ := [2, 1, 3]

/-- Calculates the total number of cups needed for all recipes -/
def TotalCups : ℕ := RecipeCups.sum

/-- Calculates the total number of ounces needed for all recipes -/
def TotalOunces : ℕ := TotalCups * OuncesPerCup

/-- Calculates the number of bottles needed, rounding up to the nearest whole number -/
def BottlesNeeded : ℕ := (TotalOunces + OuncesPerBottle - 1) / OuncesPerBottle

theorem soy_sauce_bottles : BottlesNeeded = 3 := by sorry

end NUMINAMATH_CALUDE_soy_sauce_bottles_l435_43572


namespace NUMINAMATH_CALUDE_inequality_solution_implies_k_range_l435_43554

theorem inequality_solution_implies_k_range :
  ∀ k : ℝ,
  (∀ x : ℝ, x > 1/2 ↔ (k^2 - 2*k + 3/2)^x < (k^2 - 2*k + 3/2)^(1-x)) →
  (1 - Real.sqrt 2 / 2 < k ∧ k < 1 + Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_k_range_l435_43554


namespace NUMINAMATH_CALUDE_min_value_expression_l435_43577

theorem min_value_expression (x : ℝ) : 
  (15 - x) * (14 - x) * (15 + x) * (14 + x) ≥ -142.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l435_43577


namespace NUMINAMATH_CALUDE_base3_to_base9_first_digit_l435_43513

def base3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

def first_digit_base9 (n : Nat) : Nat :=
  Nat.log 9 n + 1

theorem base3_to_base9_first_digit :
  let x : Nat := base3_to_decimal [1,2,1,1,2,2,1,1,1,2,2,2,1,1,1,1,2,2,2,2]
  first_digit_base9 x = 5 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base9_first_digit_l435_43513


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l435_43556

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equality : (3 : ℂ) / (1 - i)^2 = (3/2 : ℂ) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l435_43556


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l435_43528

def is_valid_representation (n : ℕ) (a b : ℕ) : Prop :=
  a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1

theorem smallest_dual_base_representation :
  ∃ (n : ℕ), is_valid_representation n 7 3 ∧
  ∀ (m : ℕ) (a b : ℕ), is_valid_representation m a b → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l435_43528


namespace NUMINAMATH_CALUDE_chord_equation_of_ellipse_l435_43521

/-- Given an ellipse and a point that bisects a chord of the ellipse, 
    prove the equation of the line containing the chord. -/
theorem chord_equation_of_ellipse (x y : ℝ → ℝ) :
  (∀ t, (x t)^2 / 36 + (y t)^2 / 9 = 1) →  -- Ellipse equation
  (∃ t₁ t₂, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = 4 ∧ 
    (y t₁ + y t₂) / 2 = 2) →  -- Midpoint condition
  (∃ A B : ℝ, ∀ t, A * (x t) + B * (y t) = 8) →  -- Line equation
  A = 1 ∧ B = 2 := by
sorry

end NUMINAMATH_CALUDE_chord_equation_of_ellipse_l435_43521


namespace NUMINAMATH_CALUDE_polynomial_equality_l435_43508

theorem polynomial_equality (x : ℝ) : ∃ (t a b : ℝ),
  (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60 ∧
  t = -9 ∧ a = -53 ∧ b = -156 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l435_43508


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l435_43593

theorem triangle_abc_properties (a b : ℝ) (B : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = Real.sqrt 3) (h3 : B = 60 * π / 180) :
  let A := Real.arcsin (a * Real.sin B / b)
  let C := π - A - B
  let c := a * Real.sin C / Real.sin A
  (A = 45 * π / 180) ∧ (C = 75 * π / 180) ∧ (c = (Real.sqrt 2 + Real.sqrt 6) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l435_43593


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l435_43530

/-- A rectangular prism with given conditions -/
structure RectangularPrism where
  length : ℝ
  breadth : ℝ
  height : ℝ
  length_breadth_diff : length - breadth = 23
  perimeter : 2 * length + 2 * breadth = 166

/-- The volume of a rectangular prism is 1590h cubic meters -/
theorem rectangular_prism_volume (prism : RectangularPrism) : 
  prism.length * prism.breadth * prism.height = 1590 * prism.height := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l435_43530


namespace NUMINAMATH_CALUDE_range_of_a_l435_43552

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l435_43552


namespace NUMINAMATH_CALUDE_inequality_problem_l435_43548

theorem inequality_problem (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (h : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) :
  (a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b) ∧
  (a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a)) ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 + y^2 + z^2 ≤ 2*(x*y + y*z + z*x) ∧
    ¬(x^4 + y^4 + z^4 ≤ 2*(x^2*y^2 + y^2*z^2 + z^2*x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l435_43548


namespace NUMINAMATH_CALUDE_total_items_is_110_l435_43569

/-- The number of croissants each person eats per day -/
def croissants_per_person : ℕ := 7

/-- The number of cakes each person eats per day -/
def cakes_per_person : ℕ := 18

/-- The number of pizzas each person eats per day -/
def pizzas_per_person : ℕ := 30

/-- The number of people eating -/
def number_of_people : ℕ := 2

/-- The total number of items consumed by both people in a day -/
def total_items : ℕ := 
  (croissants_per_person + cakes_per_person + pizzas_per_person) * number_of_people

theorem total_items_is_110 : total_items = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_items_is_110_l435_43569


namespace NUMINAMATH_CALUDE_fireworks_display_count_l435_43576

/-- The number of fireworks needed to display a single number. -/
def fireworks_per_number : ℕ := 6

/-- The number of fireworks needed to display a single letter. -/
def fireworks_per_letter : ℕ := 5

/-- The number of digits in the year display. -/
def year_digits : ℕ := 4

/-- The number of letters in "HAPPY NEW YEAR". -/
def phrase_letters : ℕ := 12

/-- The number of additional boxes of fireworks. -/
def additional_boxes : ℕ := 50

/-- The number of fireworks in each additional box. -/
def fireworks_per_box : ℕ := 8

/-- The total number of fireworks lit during the display. -/
def total_fireworks : ℕ := 
  year_digits * fireworks_per_number + 
  phrase_letters * fireworks_per_letter + 
  additional_boxes * fireworks_per_box

theorem fireworks_display_count : total_fireworks = 484 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_display_count_l435_43576


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l435_43590

/-- Given two points C and D, where C is at the origin and D is on the line y = 6,
    if the slope of CD is 3/4, then the sum of D's coordinates is 14. -/
theorem point_coordinates_sum (x : ℝ) : 
  let C : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (x, 6)
  (6 - 0) / (x - 0) = 3 / 4 →
  x + 6 = 14 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l435_43590


namespace NUMINAMATH_CALUDE_problem_statement_l435_43566

theorem problem_statement (a b : ℝ) :
  (a + 1)^2 + Real.sqrt (b - 2) = 0 → a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l435_43566


namespace NUMINAMATH_CALUDE_xyz_value_l435_43546

theorem xyz_value (a b c x y z : ℂ)
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + x * z = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l435_43546


namespace NUMINAMATH_CALUDE_multiplication_problem_l435_43582

theorem multiplication_problem (x : ℝ) (n : ℝ) (h1 : x = 13) (h2 : x * n = (36 - x) + 16) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l435_43582


namespace NUMINAMATH_CALUDE_inequality_proof_l435_43565

theorem inequality_proof (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h_prod : a * b * c * d * e = 1) : 
  (d * e) / (a * (b + 1)) + (e * a) / (b * (c + 1)) + 
  (a * b) / (c * (d + 1)) + (b * c) / (d * (e + 1)) + 
  (c * d) / (e * (a + 1)) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l435_43565


namespace NUMINAMATH_CALUDE_number_of_girls_l435_43562

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of possible arrangements -/
def total_arrangements : ℕ := 2880

/-- A function that calculates the number of possible arrangements given the number of boys and girls -/
def calculate_arrangements (boys girls : ℕ) : ℕ :=
  Nat.factorial boys * Nat.factorial girls

/-- Theorem stating that there are 5 girls -/
theorem number_of_girls : ∃ (girls : ℕ), girls = 5 ∧ 
  calculate_arrangements num_boys girls = total_arrangements :=
sorry

end NUMINAMATH_CALUDE_number_of_girls_l435_43562


namespace NUMINAMATH_CALUDE_additional_grazing_area_l435_43588

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 9^2 = 448 * π := by
  sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l435_43588


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l435_43539

theorem multiplication_puzzle (c d : ℕ) : 
  c ≤ 9 → d ≤ 9 → (30 + c) * (10 * d + 4) = 132 → c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l435_43539


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_15_l435_43563

theorem smallest_four_digit_multiple_of_15 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 15 = 0 → n ≥ 1005 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_15_l435_43563


namespace NUMINAMATH_CALUDE_trig_equality_proof_l435_43579

theorem trig_equality_proof (x : ℝ) : 
  (Real.sin x * Real.cos (2 * x) + Real.cos x * Real.cos (4 * x) = 
   Real.sin (π / 4 + 2 * x) * Real.sin (π / 4 - 3 * x)) ↔ 
  (∃ n : ℤ, x = π / 12 * (4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trig_equality_proof_l435_43579


namespace NUMINAMATH_CALUDE_sanctuary_feeding_sequences_l435_43510

/-- Represents the number of pairs of animals in the sanctuary -/
def num_pairs : ℕ := 5

/-- Calculates the number of distinct feeding sequences for animals in a sanctuary -/
def feeding_sequences (n : ℕ) : ℕ :=
  let male_choices := List.range n
  let female_choices := List.range n
  (female_choices.foldl (· * ·) 1) * (male_choices.tail.foldl (· * ·) 1)

/-- Theorem stating the number of distinct feeding sequences for the given conditions -/
theorem sanctuary_feeding_sequences :
  feeding_sequences num_pairs = 5760 :=
sorry

end NUMINAMATH_CALUDE_sanctuary_feeding_sequences_l435_43510


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l435_43517

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We'll use degrees for simplicity
  base_angle : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle * 2 + vertex_angle = 180

-- Define our specific isosceles triangle
def our_triangle (t : IsoscelesTriangle) : Prop :=
  t.base_angle = 50 ∧ t.vertex_angle = 80 ∨
  t.base_angle = 80 ∧ t.vertex_angle = 20

-- Theorem statement
theorem isosceles_triangle_base_angle :
  ∀ t : IsoscelesTriangle, (t.base_angle = 50 ∨ t.base_angle = 80) ↔ 
  (t.base_angle = 50 ∧ t.vertex_angle = 80 ∨ t.base_angle = 80 ∧ t.vertex_angle = 20) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l435_43517


namespace NUMINAMATH_CALUDE_exam_max_marks_l435_43586

theorem exam_max_marks :
  let pass_percentage : ℚ := 60 / 100
  let failing_score : ℕ := 210
  let failing_margin : ℕ := 90
  let max_marks : ℕ := 500
  (pass_percentage * max_marks : ℚ) = failing_score + failing_margin ∧
  max_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_exam_max_marks_l435_43586


namespace NUMINAMATH_CALUDE_combined_ppf_theorem_combined_ppf_range_l435_43514

/-- Production Possibility Frontier (PPF) for a single female -/
def single_ppf (K : ℝ) : ℝ := 40 - 2 * K

/-- Combined Production Possibility Frontier (PPF) for two females -/
def combined_ppf (K : ℝ) : ℝ := 80 - 2 * K

/-- Theorem stating that the combined PPF of two identical linear PPFs is the sum of their individual PPFs -/
theorem combined_ppf_theorem (K : ℝ) (h : K ≤ 40) :
  combined_ppf K = single_ppf (K / 2) + single_ppf (K / 2) :=
by sorry

/-- Corollary stating the range of K for the combined PPF -/
theorem combined_ppf_range (K : ℝ) :
  K ≤ 40 ↔ ∃ (K1 K2 : ℝ), K1 ≤ 20 ∧ K2 ≤ 20 ∧ K = K1 + K2 :=
by sorry

end NUMINAMATH_CALUDE_combined_ppf_theorem_combined_ppf_range_l435_43514


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l435_43584

/-- A line passes through points A(m,2) and B(-m,2m-1) with an inclination angle of 45° -/
theorem line_through_points_with_45_degree_angle (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (m, 2) ∈ line ∧ 
    (-m, 2*m - 1) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - 2) / (x - m) = 1)) → 
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l435_43584


namespace NUMINAMATH_CALUDE_problem_statement_l435_43545

theorem problem_statement (A B : ℝ) : 
  A^2 = 0.012345678987654321 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1) →
  B^2 = 0.012345679 →
  9 * 10^9 * (1 - |A|) * B = 1 ∨ 9 * 10^9 * (1 - |A|) * B = -1 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l435_43545


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l435_43522

/-- The eccentricity of a hyperbola with given equation and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (heq : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 ↔ (y = x / 2 ∨ y = -x / 2)) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l435_43522


namespace NUMINAMATH_CALUDE_incorrect_calculation_l435_43578

theorem incorrect_calculation : 3 * Real.sqrt 3 - Real.sqrt 3 ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l435_43578


namespace NUMINAMATH_CALUDE_savings_calculation_l435_43574

/-- Given a person's income and expenditure ratio, and their income, calculate their savings -/
def calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that given the specific income-expenditure ratio and income, the savings are 3000 -/
theorem savings_calculation :
  let income_ratio : ℕ := 10
  let expenditure_ratio : ℕ := 7
  let income : ℕ := 10000
  calculate_savings income_ratio expenditure_ratio income = 3000 := by
  sorry

#eval calculate_savings 10 7 10000

end NUMINAMATH_CALUDE_savings_calculation_l435_43574


namespace NUMINAMATH_CALUDE_min_vertical_distance_l435_43583

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

theorem min_vertical_distance : 
  ∃ (x₀ : ℝ), ∀ (x : ℝ), |f x - g x| ≥ |f x₀ - g x₀| ∧ |f x₀ - g x₀| = 10 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l435_43583


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l435_43597

theorem opposite_of_negative_five : -((-5) : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l435_43597


namespace NUMINAMATH_CALUDE_isabella_babysitting_weeks_l435_43520

/-- Calculates the number of weeks Isabella has been babysitting -/
def weeks_babysitting (hourly_rate : ℚ) (hours_per_day : ℚ) (days_per_week : ℚ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (hourly_rate * hours_per_day * days_per_week)

/-- Proves that Isabella has been babysitting for 7 weeks -/
theorem isabella_babysitting_weeks :
  weeks_babysitting 5 5 6 1050 = 7 := by
  sorry

end NUMINAMATH_CALUDE_isabella_babysitting_weeks_l435_43520


namespace NUMINAMATH_CALUDE_h_in_terms_of_f_l435_43561

-- Define the domain of f
def I : Set ℝ := Set.Icc (-3 : ℝ) 3

-- Define f as a function on the interval I
variable (f : I → ℝ)

-- Define h as a function derived from f
def h (x : ℝ) : ℝ := -(f ⟨x + 6, sorry⟩)

-- Theorem statement
theorem h_in_terms_of_f (x : ℝ) : h f x = -f ⟨x + 6, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_h_in_terms_of_f_l435_43561


namespace NUMINAMATH_CALUDE_three_students_two_groups_l435_43596

/-- The number of ways for students to sign up for activity groups. -/
def signUpWays (numStudents : ℕ) (numGroups : ℕ) : ℕ :=
  numGroups ^ numStudents

/-- Theorem: Three students signing up for two groups results in 8 ways. -/
theorem three_students_two_groups :
  signUpWays 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_students_two_groups_l435_43596


namespace NUMINAMATH_CALUDE_captainSelection_l435_43529

/-- The number of ways to select a captain and a vice-captain from a team of 11 people -/
def selectCaptains : ℕ :=
  11 * 10

/-- Theorem stating that the number of ways to select a captain and a vice-captain
    from a team of 11 people is equal to 110 -/
theorem captainSelection : selectCaptains = 110 := by
  sorry

end NUMINAMATH_CALUDE_captainSelection_l435_43529


namespace NUMINAMATH_CALUDE_orange_apple_cost_l435_43509

theorem orange_apple_cost : ∃ (x y : ℚ),
  (7 * x + 5 * y = 13) ∧
  (3 * x + 4 * y = 8) →
  (37 * x + 45 * y = 93) := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_cost_l435_43509


namespace NUMINAMATH_CALUDE_equation_equivalence_implies_mnp_18_l435_43503

theorem equation_equivalence_implies_mnp_18 
  (a x z c : ℝ) (m n p : ℤ) 
  (h : a^8*x*z - a^7*z - a^6*x = a^5*(c^5 - 1)) 
  (h_equiv : (a^m*x - a^n)*(a^p*z - a^3) = a^5*c^5) : 
  m * n * p = 18 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_implies_mnp_18_l435_43503


namespace NUMINAMATH_CALUDE_infinite_points_on_line_l435_43575

/-- A point on the line x + y = 4 with positive rational coordinates -/
structure PointOnLine where
  x : ℚ
  y : ℚ
  x_pos : 0 < x
  y_pos : 0 < y
  on_line : x + y = 4

/-- The set of all points on the line x + y = 4 with positive rational coordinates -/
def PointsOnLine : Set PointOnLine :=
  {p : PointOnLine | True}

/-- Theorem: There are infinitely many points on the line x + y = 4 with positive rational coordinates -/
theorem infinite_points_on_line : Set.Infinite PointsOnLine := by
  sorry

end NUMINAMATH_CALUDE_infinite_points_on_line_l435_43575


namespace NUMINAMATH_CALUDE_james_matches_count_l435_43598

/-- The number of boxes in a dozen -/
def boxesPerDozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def dozensOfBoxes : ℕ := 5

/-- The number of matches in each box -/
def matchesPerBox : ℕ := 20

/-- Theorem: Given the conditions, James has 1200 matches -/
theorem james_matches_count :
  dozensOfBoxes * boxesPerDozen * matchesPerBox = 1200 := by
  sorry

end NUMINAMATH_CALUDE_james_matches_count_l435_43598


namespace NUMINAMATH_CALUDE_savings_account_balance_l435_43560

theorem savings_account_balance 
  (total : ℕ) 
  (checking : ℕ) 
  (h1 : total = 9844)
  (h2 : checking = 6359) :
  total - checking = 3485 :=
by sorry

end NUMINAMATH_CALUDE_savings_account_balance_l435_43560


namespace NUMINAMATH_CALUDE_a_gt_b_relation_l435_43523

theorem a_gt_b_relation (a b : ℝ) :
  (∀ a b, a - 1 > b + 1 → a > b) ∧
  (∃ a b, a > b ∧ a - 1 ≤ b + 1) :=
sorry

end NUMINAMATH_CALUDE_a_gt_b_relation_l435_43523


namespace NUMINAMATH_CALUDE_f_properties_l435_43527

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

-- State the theorem
theorem f_properties (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a b c a = 0) (h5 : f a b c b = 0) (h6 : f a b c c = 0) :
  (f a b c 0) * (f a b c 1) < 0 ∧ (f a b c 0) * (f a b c 3) > 0 := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l435_43527


namespace NUMINAMATH_CALUDE_carrot_distribution_l435_43557

theorem carrot_distribution (total : ℕ) (leftover : ℕ) (people : ℕ) : 
  total = 74 → 
  leftover = 2 → 
  people > 1 → 
  people < 72 → 
  (total - leftover) % people = 0 → 
  72 % people = 0 := by
sorry

end NUMINAMATH_CALUDE_carrot_distribution_l435_43557


namespace NUMINAMATH_CALUDE_cubic_factorization_l435_43550

theorem cubic_factorization (x y z : ℝ) :
  x^3 + y^3 + z^3 - 3*x*y*z = (x + y + z) * (x^2 + y^2 + z^2 - x*y - y*z - z*x) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l435_43550


namespace NUMINAMATH_CALUDE_cobalt_61_neutron_count_l435_43538

/-- Represents an atom with its mass number and number of protons -/
structure Atom where
  mass_number : ℕ
  proton_count : ℕ

/-- Calculates the number of neutrons in an atom -/
def neutron_count (a : Atom) : ℕ := a.mass_number - a.proton_count

/-- Theorem: The number of neutrons in a ⁶¹₂₇Co atom is 34 -/
theorem cobalt_61_neutron_count :
  let co_61 : Atom := { mass_number := 61, proton_count := 27 }
  neutron_count co_61 = 34 := by
  sorry

end NUMINAMATH_CALUDE_cobalt_61_neutron_count_l435_43538


namespace NUMINAMATH_CALUDE_alex_corn_purchase_l435_43599

/-- The price of corn per pound -/
def corn_price : ℝ := 1.20

/-- The price of beans per pound -/
def bean_price : ℝ := 0.60

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- The total cost of the purchase -/
def total_cost : ℝ := 27.00

/-- The amount of corn bought in pounds -/
def corn_amount : ℝ := 15.0

theorem alex_corn_purchase :
  ∃ (bean_amount : ℝ),
    corn_amount + bean_amount = total_pounds ∧
    corn_price * corn_amount + bean_price * bean_amount = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_alex_corn_purchase_l435_43599


namespace NUMINAMATH_CALUDE_jerry_reaches_first_l435_43568

-- Define the points
variable (A B C D : Point)

-- Define the distances
variable (AB BD AC CD : ℝ)

-- Define the speeds
variable (speed_tom speed_jerry : ℝ)

-- Define the delay
variable (delay : ℝ)

-- Theorem statement
theorem jerry_reaches_first (h1 : AB = 32) (h2 : BD = 12) (h3 : AC = 13) (h4 : CD = 27)
  (h5 : speed_tom = 5) (h6 : speed_jerry = 4) (h7 : delay = 5) :
  (AB + BD) / speed_jerry < delay + (AC + CD) / speed_tom := by
  sorry

end NUMINAMATH_CALUDE_jerry_reaches_first_l435_43568


namespace NUMINAMATH_CALUDE_factorization_equality_l435_43507

theorem factorization_equality (m n : ℝ) : 2 * m^2 * n - 8 * m * n + 8 * n = 2 * n * (m - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l435_43507


namespace NUMINAMATH_CALUDE_max_value_theorem_l435_43534

theorem max_value_theorem (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ ≥ 0 ∧ b₀ ≥ 0 ∧ c₀ ≥ 0 ∧ 
    a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ * Real.sqrt 2 + 2 * a₀ * c₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l435_43534


namespace NUMINAMATH_CALUDE_jennifer_total_distance_l435_43564

/-- Represents the distances and changes for Jennifer's museum visits -/
structure MuseumDistances where
  first_museum : ℕ
  second_museum : ℕ
  cultural_center : ℕ
  traffic_increase : ℕ
  bus_decrease : ℕ
  bicycle_decrease : ℕ

/-- Calculates the total distance for Jennifer's museum visits -/
def total_distance (d : MuseumDistances) : ℕ :=
  (d.second_museum + d.traffic_increase) + 
  (d.cultural_center - d.bus_decrease) + 
  (d.first_museum - d.bicycle_decrease)

/-- Theorem stating that Jennifer's total distance is 32 miles -/
theorem jennifer_total_distance :
  ∀ d : MuseumDistances,
  d.first_museum = 5 ∧
  d.second_museum = 15 ∧
  d.cultural_center = 10 ∧
  d.traffic_increase = 5 ∧
  d.bus_decrease = 2 ∧
  d.bicycle_decrease = 1 →
  total_distance d = 32 :=
by sorry

end NUMINAMATH_CALUDE_jennifer_total_distance_l435_43564


namespace NUMINAMATH_CALUDE_volleyball_not_basketball_l435_43591

theorem volleyball_not_basketball (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (neither : ℕ) :
  total = 40 →
  basketball = 15 →
  volleyball = 20 →
  neither = 10 →
  volleyball - (basketball + volleyball - (total - neither)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_not_basketball_l435_43591


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l435_43573

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def num1 : List Bool := [true, false, true, true]  -- 1101₂
def num2 : List Bool := [true, false, true]        -- 101₂
def num3 : List Bool := [false, true, true, true]  -- 1110₂
def num4 : List Bool := [true, true, true]         -- 111₂
def num5 : List Bool := [false, true, false, true] -- 1010₂
def result : List Bool := [true, false, true, false, true] -- 10101₂

theorem binary_sum_theorem :
  binary_to_nat num1 + binary_to_nat num2 + binary_to_nat num3 +
  binary_to_nat num4 + binary_to_nat num5 = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l435_43573


namespace NUMINAMATH_CALUDE_pencils_per_row_l435_43551

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 25 → num_rows = 5 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l435_43551


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l435_43567

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original : ℕ) (current : ℕ) : ℕ :=
  current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l435_43567


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l435_43540

theorem max_value_trig_sum (a b φ : ℝ) :
  ∃ (max : ℝ), ∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ max ∧
  ∃ θ₀ : ℝ, a * Real.cos (θ₀ + φ) + b * Real.sin (θ₀ + φ) = max ∧
  max = Real.sqrt (a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l435_43540


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_solution_l435_43511

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_money_after_transfer (m : Merchants) 
  (h : problem_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

/-- The main theorem stating the solution -/
theorem solution (m : Merchants) 
  (h : problem_conditions m) : 
  ∃ x : ℕ, m.foma - x = m.ierema + x ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_after_transfer_solution_l435_43511


namespace NUMINAMATH_CALUDE_max_choir_members_choir_of_120_exists_l435_43581

/-- Represents a choir formation --/
structure ChoirFormation where
  rows : ℕ
  members_per_row : ℕ

/-- Represents the choir and its formations --/
structure Choir where
  total_members : ℕ
  original_formation : ChoirFormation
  new_formation : ChoirFormation

/-- The conditions of the choir problem --/
def choir_conditions (c : Choir) : Prop :=
  c.total_members < 120 ∧
  c.total_members = c.original_formation.rows * c.original_formation.members_per_row + 3 ∧
  c.total_members = (c.original_formation.rows - 1) * (c.original_formation.members_per_row + 2)

/-- The theorem stating the maximum number of choir members --/
theorem max_choir_members :
  ∀ c : Choir, choir_conditions c → c.total_members ≤ 120 :=
by sorry

/-- The theorem stating that 120 is achievable --/
theorem choir_of_120_exists :
  ∃ c : Choir, choir_conditions c ∧ c.total_members = 120 :=
by sorry

end NUMINAMATH_CALUDE_max_choir_members_choir_of_120_exists_l435_43581
