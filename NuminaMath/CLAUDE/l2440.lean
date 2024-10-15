import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2440_244013

theorem right_triangle_segment_ratio 
  (a b c r s : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (height_division : r + s = c) 
  (similarity_relations : a^2 = r * c ∧ b^2 = s * c) 
  (leg_ratio : a / b = 1 / 3) :
  r / s = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2440_244013


namespace NUMINAMATH_CALUDE_three_collinear_sufficient_not_necessary_for_coplanar_l2440_244018

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Main theorem: Three points on a line is sufficient but not necessary for four points to be coplanar -/
theorem three_collinear_sufficient_not_necessary_for_coplanar :
  (∀ p q r s : Point3D, (collinear p q r) → (coplanar p q r s)) ∧
  (∃ p q r s : Point3D, (coplanar p q r s) ∧ ¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s)) :=
sorry

end NUMINAMATH_CALUDE_three_collinear_sufficient_not_necessary_for_coplanar_l2440_244018


namespace NUMINAMATH_CALUDE_hockey_league_games_l2440_244057

/-- Calculate the number of games in a hockey league season -/
theorem hockey_league_games (num_teams : ℕ) (face_times : ℕ) : 
  num_teams = 18 → face_times = 10 → 
  (num_teams * (num_teams - 1) / 2) * face_times = 1530 :=
by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2440_244057


namespace NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l2440_244085

theorem least_integer_square_72_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 72 ∧ ∀ y : ℤ, y^2 = 2*y + 72 → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l2440_244085


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2440_244039

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 1991 is the only natural number n for which n + s(n) = 2011 -/
theorem unique_solution_for_equation : 
  ∃! n : ℕ, n + sum_of_digits n = 2011 ∧ n = 1991 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2440_244039


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2440_244087

/-- Represents a quadratic equation ax² - 6x + c = 0 with exactly one solution -/
structure UniqueQuadratic where
  a : ℝ
  c : ℝ
  has_unique_solution : ∃! x, a * x^2 - 6 * x + c = 0

theorem unique_quadratic_solution (q : UniqueQuadratic)
  (sum_eq_12 : q.a + q.c = 12)
  (a_lt_c : q.a < q.c) :
  q.a = 6 - 3 * Real.sqrt 3 ∧ q.c = 6 + 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2440_244087


namespace NUMINAMATH_CALUDE_cubic_minimum_at_negative_one_l2440_244035

/-- A cubic function with parameters p, q, and r -/
def cubic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  x^3 + p*x^2 + q*x + r

theorem cubic_minimum_at_negative_one (p q r : ℝ) :
  (∀ x, cubic_function p q r x ≥ 0) ∧
  (cubic_function p q r (-1) = 0) →
  r = p - 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_minimum_at_negative_one_l2440_244035


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l2440_244071

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- Theorem statement
theorem f_composition_negative_two (f : ℝ → ℝ) :
  (∀ x, x ≥ 0 → f x = 1 - Real.sqrt x) →
  (∀ x, x < 0 → f x = 2^x) →
  f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l2440_244071


namespace NUMINAMATH_CALUDE_midpoint_figure_area_l2440_244095

/-- The area of a figure in a 6x6 grid formed by connecting midpoints to the center -/
theorem midpoint_figure_area : 
  ∀ (grid_size : ℕ) (center_square_area corner_triangle_area : ℝ),
  grid_size = 6 →
  center_square_area = 4.5 →
  corner_triangle_area = 4.5 →
  center_square_area + 4 * corner_triangle_area = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_figure_area_l2440_244095


namespace NUMINAMATH_CALUDE_candies_distribution_l2440_244045

def candies_a : ℕ := 17
def candies_b : ℕ := 19
def num_people : ℕ := 9

theorem candies_distribution :
  (candies_a + candies_b) / num_people = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_distribution_l2440_244045


namespace NUMINAMATH_CALUDE_f_symmetry_l2440_244069

/-- A function f(x) = x^5 + ax^3 + bx - 8 for some real a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

/-- Theorem: If f(-2) = 10, then f(2) = -26 -/
theorem f_symmetry (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2440_244069


namespace NUMINAMATH_CALUDE_second_number_is_three_l2440_244076

theorem second_number_is_three (x y : ℝ) 
  (sum_is_ten : x + y = 10) 
  (relation : 2 * x = 3 * y + 5) : 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_three_l2440_244076


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l2440_244029

-- Define the pentagon and its extended points
variable (A B C D E A' B' C' D' E' : ℝ × ℝ)

-- Define the conditions of the pentagon
axiom extend_A : A' - B = B - A
axiom extend_B : B' - C = C - B
axiom extend_C : C' - D = D - C
axiom extend_D : D' - E = E - D
axiom extend_E : E' - A = A - E

-- Define the theorem
theorem pentagon_reconstruction :
  E = (1/31 : ℝ) • A' + (1/31 : ℝ) • B' + (2/31 : ℝ) • C' + (4/31 : ℝ) • D' + (8/31 : ℝ) • E' :=
sorry

end NUMINAMATH_CALUDE_pentagon_reconstruction_l2440_244029


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l2440_244072

theorem no_solution_to_inequality :
  ¬∃ x : ℝ, (4 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 5) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l2440_244072


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2440_244016

theorem trigonometric_equation_solution (t : ℝ) : 
  5.43 * Real.cos (22 * π / 180 - t) * Real.cos (82 * π / 180 - t) + 
  Real.cos (112 * π / 180 - t) * Real.cos (172 * π / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) ↔ 
  (∃ k : ℤ, t = 2 * π * k ∨ t = π / 2 * (4 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2440_244016


namespace NUMINAMATH_CALUDE_four_valid_configurations_l2440_244064

/-- Represents a square piece -/
inductive Square
| A | B | C | D | E | F | G | H

/-- Represents the F-shaped figure -/
structure FShape :=
  (squares : Fin 6 → Unit)

/-- Represents a topless rectangular box -/
structure ToplessBox :=
  (base : Unit)
  (sides : Fin 4 → Unit)

/-- Function to check if a square can be combined with the F-shape to form a topless box -/
def canFormBox (s : Square) (f : FShape) : Prop :=
  ∃ (box : ToplessBox), true  -- Placeholder, actual implementation would be more complex

/-- The main theorem stating that exactly 4 squares can form a topless box with the F-shape -/
theorem four_valid_configurations (squares : Fin 8 → Square) (f : FShape) :
  (∃! (validSquares : Finset Square), 
    validSquares.card = 4 ∧ 
    ∀ s, s ∈ validSquares ↔ canFormBox s f) :=
sorry

end NUMINAMATH_CALUDE_four_valid_configurations_l2440_244064


namespace NUMINAMATH_CALUDE_find_y_l2440_244050

theorem find_y (x z : ℤ) (y : ℚ) 
  (h1 : x = -2) 
  (h2 : z = 4) 
  (h3 : x^2 * y * z - x * y * z^2 = 48) : 
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2440_244050


namespace NUMINAMATH_CALUDE_problem_solution_l2440_244006

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
  (x + y) / (x - y) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2440_244006


namespace NUMINAMATH_CALUDE_brownies_remaining_l2440_244080

/-- Calculates the number of brownies left after consumption --/
def brownies_left (total : Nat) (tina_daily : Nat) (husband_daily : Nat) (days : Nat) (shared : Nat) : Nat :=
  total - (tina_daily * days) - (husband_daily * days) - shared

/-- Proves that 5 brownies are left under given conditions --/
theorem brownies_remaining : brownies_left 24 2 1 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_brownies_remaining_l2440_244080


namespace NUMINAMATH_CALUDE_wood_not_heavier_than_brick_l2440_244097

-- Define the mass of the block of wood in kg
def wood_mass_kg : ℝ := 8

-- Define the mass of the brick in g
def brick_mass_g : ℝ := 8000

-- Define the conversion factor from kg to g
def kg_to_g : ℝ := 1000

-- Theorem statement
theorem wood_not_heavier_than_brick : ¬(wood_mass_kg * kg_to_g > brick_mass_g) := by
  sorry

end NUMINAMATH_CALUDE_wood_not_heavier_than_brick_l2440_244097


namespace NUMINAMATH_CALUDE_no_solution_inequality_system_l2440_244017

theorem no_solution_inequality_system :
  ¬ ∃ (x y : ℝ), (4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2) ∧ (x - y ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_inequality_system_l2440_244017


namespace NUMINAMATH_CALUDE_jessica_mark_earnings_l2440_244008

/-- Given the working hours and earnings of Jessica and Mark, prove that t = 5 --/
theorem jessica_mark_earnings (t : ℝ) : 
  (t + 2) * (4 * t + 1) = (4 * t - 7) * (t + 3) + 4 → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_mark_earnings_l2440_244008


namespace NUMINAMATH_CALUDE_problem_solution_l2440_244037

theorem problem_solution : 
  (3 * Real.sqrt 18 / Real.sqrt 2 + Real.sqrt 12 * Real.sqrt 3 = 15) ∧
  ((2 + Real.sqrt 6)^2 - (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 8 + 4 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2440_244037


namespace NUMINAMATH_CALUDE_function_range_condition_l2440_244044

open Real

/-- Given a function f(x) = ax - ln x - 1, prove that there exists x₀ ∈ (0,e] 
    such that f(x₀) < 0 if and only if a ∈ (-∞, 1). -/
theorem function_range_condition (a : ℝ) : 
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ ℯ ∧ a * x₀ - log x₀ - 1 < 0) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_function_range_condition_l2440_244044


namespace NUMINAMATH_CALUDE_smallest_valid_integers_difference_l2440_244042

def is_valid_integer (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 1

theorem smallest_valid_integers_difference : 
  ∃ n₁ n₂ : ℕ, 
    is_valid_integer n₁ ∧
    is_valid_integer n₂ ∧
    n₁ < n₂ ∧
    (∀ m : ℕ, is_valid_integer m → m ≥ n₁) ∧
    (∀ m : ℕ, is_valid_integer m ∧ m ≠ n₁ → m ≥ n₂) ∧
    n₂ - n₁ = 27720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_integers_difference_l2440_244042


namespace NUMINAMATH_CALUDE_domain_implies_k_range_inequality_solution_set_l2440_244079

-- Problem I
theorem domain_implies_k_range (f : ℝ → ℝ) (h : ∀ x, ∃ y, f x = y) :
  (∀ x, f x = Real.sqrt (x^2 - x * k - k)) → k ∈ Set.Icc (-4) 0 := by sorry

-- Problem II
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | (x - a) * (x + a - 1) > 0} =
    if a = 1/2 then
      {x : ℝ | x ≠ 1/2}
    else if a < 1/2 then
      {x : ℝ | x > 1 - a ∨ x < a}
    else
      {x : ℝ | x > a ∨ x < 1 - a} := by sorry

end NUMINAMATH_CALUDE_domain_implies_k_range_inequality_solution_set_l2440_244079


namespace NUMINAMATH_CALUDE_molecular_weight_X_l2440_244082

/-- Given a compound Ba(X)₂ with total molecular weight 171 and Ba having
    molecular weight 137, prove that the molecular weight of X is 17. -/
theorem molecular_weight_X (total_weight : ℝ) (ba_weight : ℝ) (x_weight : ℝ) :
  total_weight = 171 →
  ba_weight = 137 →
  total_weight = ba_weight + 2 * x_weight →
  x_weight = 17 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_X_l2440_244082


namespace NUMINAMATH_CALUDE_units_digit_problem_l2440_244030

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ := 
  a * (r^(n+1) - 1) / (r - 1)

theorem units_digit_problem : 
  (2 * geometric_sum 1 3 9) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2440_244030


namespace NUMINAMATH_CALUDE_area_perimeter_ratio_inequality_l2440_244086

/-- A convex polygon -/
structure ConvexPolygon where
  area : ℝ
  perimeter : ℝ

/-- X is contained within Y -/
def isContainedIn (X Y : ConvexPolygon) : Prop := sorry

theorem area_perimeter_ratio_inequality {X Y : ConvexPolygon} 
  (h : isContainedIn X Y) :
  X.area / X.perimeter < 2 * Y.area / Y.perimeter := by
  sorry

end NUMINAMATH_CALUDE_area_perimeter_ratio_inequality_l2440_244086


namespace NUMINAMATH_CALUDE_sum_after_removal_l2440_244043

theorem sum_after_removal (a b c d e f : ℚ) : 
  a = 1/3 → b = 1/6 → c = 1/9 → d = 1/12 → e = 1/15 → f = 1/18 →
  a + b + c + f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_removal_l2440_244043


namespace NUMINAMATH_CALUDE_triangle_properties_l2440_244065

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x - Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_properties :
  ∀ (A B C : ℝ) (a b c : ℝ),
  f A = -1 →
  a = Real.sqrt 7 →
  ∃ (m n : ℝ × ℝ), m = (3, Real.sin B) ∧ n = (2, Real.sin C) ∧ ∃ (k : ℝ), m = k • n →
  A = π / 3 ∧
  b = 3 ∧
  c = 2 ∧
  (1 / 2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2440_244065


namespace NUMINAMATH_CALUDE_square_adjacent_to_multiple_of_five_l2440_244051

theorem square_adjacent_to_multiple_of_five (n : ℤ) (h : ¬ 5 ∣ n) :
  ∃ k : ℤ, n^2 = 5*k + 1 ∨ n^2 = 5*k - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_adjacent_to_multiple_of_five_l2440_244051


namespace NUMINAMATH_CALUDE_ayen_jog_time_l2440_244078

/-- The number of minutes Ayen usually jogs every day during weekdays -/
def usual_jog_time : ℕ := sorry

/-- The total time Ayen jogged this week in minutes -/
def total_jog_time : ℕ := 180

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The extra minutes Ayen jogged on Tuesday -/
def tuesday_extra : ℕ := 5

/-- The extra minutes Ayen jogged on Friday -/
def friday_extra : ℕ := 25

theorem ayen_jog_time : 
  usual_jog_time * weekdays + tuesday_extra + friday_extra = total_jog_time ∧
  usual_jog_time = 30 := by sorry

end NUMINAMATH_CALUDE_ayen_jog_time_l2440_244078


namespace NUMINAMATH_CALUDE_tax_rate_on_other_items_l2440_244027

/-- Represents the percentage of total spending on each category -/
structure SpendingPercentages where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Represents the tax rates for each category -/
structure TaxRates where
  clothing : ℝ
  food : ℝ
  other : ℝ

/-- Theorem: Given the spending percentages and known tax rates, 
    prove that the tax rate on other items is 8% -/
theorem tax_rate_on_other_items 
  (sp : SpendingPercentages)
  (tr : TaxRates)
  (h1 : sp.clothing = 0.4)
  (h2 : sp.food = 0.3)
  (h3 : sp.other = 0.3)
  (h4 : sp.clothing + sp.food + sp.other = 1)
  (h5 : tr.clothing = 0.04)
  (h6 : tr.food = 0)
  (h7 : sp.clothing * tr.clothing + sp.food * tr.food + sp.other * tr.other = 0.04) :
  tr.other = 0.08 := by
  sorry


end NUMINAMATH_CALUDE_tax_rate_on_other_items_l2440_244027


namespace NUMINAMATH_CALUDE_simplify_expression_l2440_244033

theorem simplify_expression (a : ℝ) : (2*a - 3)^2 - (a + 5)*(a - 5) = 3*a^2 - 12*a + 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2440_244033


namespace NUMINAMATH_CALUDE_melanie_dimes_proof_l2440_244068

def final_dimes (initial : ℕ) (received : ℕ) (given_away : ℕ) : ℕ :=
  initial + received - given_away

theorem melanie_dimes_proof :
  final_dimes 7 8 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_proof_l2440_244068


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2440_244038

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℚ) (new_person_weight : ℚ) :
  initial_people = 6 →
  initial_avg_weight = 160 →
  new_person_weight = 97 →
  let total_weight : ℚ := initial_people * initial_avg_weight + new_person_weight
  let new_people : ℕ := initial_people + 1
  let new_avg_weight : ℚ := total_weight / new_people
  new_avg_weight = 151 := by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2440_244038


namespace NUMINAMATH_CALUDE_yellow_second_probability_l2440_244055

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  red : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a yellow marble second given the bag contents and rules -/
def probability_yellow_second (bag_a bag_b bag_c : BagContents) : ℚ :=
  let total_a := bag_a.red + bag_a.black
  let total_b := bag_b.yellow + bag_b.blue
  let total_c := bag_c.yellow + bag_c.blue
  let prob_red_a := bag_a.red / total_a
  let prob_black_a := bag_a.black / total_a
  let prob_yellow_b := bag_b.yellow / total_b
  let prob_yellow_c := bag_c.yellow / total_c
  prob_red_a * prob_yellow_b + prob_black_a * prob_yellow_c

/-- Theorem stating that the probability of drawing a yellow marble second is 1/3 -/
theorem yellow_second_probability :
  let bag_a : BagContents := { red := 3, black := 6, yellow := 0, blue := 0 }
  let bag_b : BagContents := { red := 0, black := 0, yellow := 6, blue := 4 }
  let bag_c : BagContents := { red := 0, black := 0, yellow := 2, blue := 8 }
  probability_yellow_second bag_a bag_b bag_c = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_yellow_second_probability_l2440_244055


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2440_244004

def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

theorem sum_of_coefficients : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2440_244004


namespace NUMINAMATH_CALUDE_inequality_implication_l2440_244074

theorem inequality_implication (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2) → 
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2440_244074


namespace NUMINAMATH_CALUDE_not_p_and_not_not_p_l2440_244047

theorem not_p_and_not_not_p (p : Prop) : ¬(p ∧ ¬p) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_not_p_l2440_244047


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2440_244003

theorem no_real_roots_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2440_244003


namespace NUMINAMATH_CALUDE_estimated_probability_is_0_30_l2440_244025

/-- Represents a single shot result -/
inductive ShotResult
| Hit
| Miss

/-- Represents the result of three shots -/
structure ThreeShotResult :=
  (shot1 shot2 shot3 : ShotResult)

/-- Checks if a ThreeShotResult has exactly two hits -/
def hasTwoHits (result : ThreeShotResult) : Bool :=
  match result with
  | ⟨ShotResult.Hit, ShotResult.Hit, ShotResult.Miss⟩ => true
  | ⟨ShotResult.Hit, ShotResult.Miss, ShotResult.Hit⟩ => true
  | ⟨ShotResult.Miss, ShotResult.Hit, ShotResult.Hit⟩ => true
  | _ => false

/-- Converts a digit to a ShotResult -/
def digitToShotResult (d : Nat) : ShotResult :=
  if d ≤ 3 then ShotResult.Hit else ShotResult.Miss

/-- Converts a three-digit number to a ThreeShotResult -/
def numberToThreeShotResult (n : Nat) : ThreeShotResult :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ⟨digitToShotResult d1, digitToShotResult d2, digitToShotResult d3⟩

/-- The list of simulation results -/
def simulationResults : List Nat :=
  [321, 421, 191, 925, 271, 932, 800, 478, 589, 663,
   531, 297, 396, 021, 546, 388, 230, 113, 507, 965]

/-- Counts the number of ThreeShotResults with exactly two hits -/
def countTwoHits (results : List Nat) : Nat :=
  results.filter (fun n => hasTwoHits (numberToThreeShotResult n)) |>.length

/-- Theorem: The estimated probability of hitting the bullseye exactly twice in three shots is 0.30 -/
theorem estimated_probability_is_0_30 :
  (countTwoHits simulationResults : Rat) / simulationResults.length = 0.30 := by
  sorry


end NUMINAMATH_CALUDE_estimated_probability_is_0_30_l2440_244025


namespace NUMINAMATH_CALUDE_power_two_mod_seven_l2440_244067

theorem power_two_mod_seven : (2^200 - 3) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_mod_seven_l2440_244067


namespace NUMINAMATH_CALUDE_flagpole_length_correct_flagpole_length_is_60_l2440_244094

/-- The length of the flagpole in feet. -/
def flagpole_length : ℝ := 60

/-- The total distance the flag moves up and down the pole in feet. -/
def total_flag_movement : ℝ := 180

/-- Theorem stating that the flagpole length is correct given the total flag movement. -/
theorem flagpole_length_correct :
  flagpole_length * 3 = total_flag_movement :=
by sorry

/-- Theorem proving that the flagpole length is 60 feet. -/
theorem flagpole_length_is_60 :
  flagpole_length = 60 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_length_correct_flagpole_length_is_60_l2440_244094


namespace NUMINAMATH_CALUDE_water_displacement_cube_in_cylinder_l2440_244056

theorem water_displacement_cube_in_cylinder (cube_side : ℝ) (cylinder_radius : ℝ) 
  (h_cube : cube_side = 12) (h_cylinder : cylinder_radius = 6) : ∃ v : ℝ, v^2 = 4374 :=
by
  sorry

end NUMINAMATH_CALUDE_water_displacement_cube_in_cylinder_l2440_244056


namespace NUMINAMATH_CALUDE_painter_completion_time_l2440_244024

-- Define the start time
def start_time : Nat := 9

-- Define the quarter completion time
def quarter_time : Nat := 12

-- Define the time taken for quarter completion
def quarter_duration : Nat := quarter_time - start_time

-- Define the total duration
def total_duration : Nat := 4 * quarter_duration

-- Define the completion time
def completion_time : Nat := start_time + total_duration

-- Theorem statement
theorem painter_completion_time :
  start_time = 9 →
  quarter_time = 12 →
  completion_time = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_painter_completion_time_l2440_244024


namespace NUMINAMATH_CALUDE_altitude_intersection_property_l2440_244059

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Finds the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Theorem: In an acute triangle ABC with altitudes AP and BQ intersecting at H,
    if HP = 7 and HQ = 3, then (BP)(PC) - (AQ)(QC) = 40 -/
theorem altitude_intersection_property (t : Triangle) (P Q H : Point) :
  isAcute t →
  isPerpendicular t.A P t.B t.C →
  isPerpendicular t.B Q t.A t.C →
  H = lineIntersection t.A P t.B Q →
  distance H P = 7 →
  distance H Q = 3 →
  distance t.B P * distance P t.C - distance t.A Q * distance Q t.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_property_l2440_244059


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2440_244020

theorem cubic_expression_value (p q : ℝ) : 
  (8 * p + 2 * q = 2022) → 
  ((-8) * p + (-2) * q + 1 = -2021) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2440_244020


namespace NUMINAMATH_CALUDE_expand_binomial_product_l2440_244036

theorem expand_binomial_product (x : ℝ) : (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomial_product_l2440_244036


namespace NUMINAMATH_CALUDE_equal_charge_at_20_minutes_l2440_244060

/-- United Telephone's base rate -/
def united_base : ℝ := 11

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℝ := 20

theorem equal_charge_at_20_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
sorry

end NUMINAMATH_CALUDE_equal_charge_at_20_minutes_l2440_244060


namespace NUMINAMATH_CALUDE_students_playing_football_l2440_244022

theorem students_playing_football 
  (total : ℕ) 
  (cricket : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (h1 : total = 470) 
  (h2 : cricket = 175) 
  (h3 : neither = 50) 
  (h4 : both = 80) : 
  total - neither - cricket + both = 325 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_football_l2440_244022


namespace NUMINAMATH_CALUDE_cookies_with_three_cups_l2440_244000

/- Define the rate of cookies per cup of flour -/
def cookies_per_cup (total_cookies : ℕ) (total_cups : ℕ) : ℚ :=
  total_cookies / total_cups

/- Define the function to calculate cookies from cups of flour -/
def cookies_from_cups (rate : ℚ) (cups : ℕ) : ℚ :=
  rate * cups

/- Theorem statement -/
theorem cookies_with_three_cups 
  (h1 : cookies_per_cup 24 4 = 6) 
  (h2 : cookies_from_cups (cookies_per_cup 24 4) 3 = 18) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_cookies_with_three_cups_l2440_244000


namespace NUMINAMATH_CALUDE_average_of_first_two_l2440_244062

theorem average_of_first_two (total_avg : ℝ) (second_set_avg : ℝ) (third_set_avg : ℝ)
  (h1 : total_avg = 2.5)
  (h2 : second_set_avg = 1.4)
  (h3 : third_set_avg = 5) :
  let total_sum := 6 * total_avg
  let second_set_sum := 2 * second_set_avg
  let third_set_sum := 2 * third_set_avg
  let first_set_sum := total_sum - second_set_sum - third_set_sum
  first_set_sum / 2 = 1.1 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_two_l2440_244062


namespace NUMINAMATH_CALUDE_sector_chord_length_l2440_244015

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, 
    its chord length is 2sin(1) cm. -/
theorem sector_chord_length 
  (r : ℝ) 
  (α : ℝ) 
  (h_area : (1/2) * α * r^2 = 1) 
  (h_perim : 2*r + α*r = 4) : 
  2 * r * Real.sin (α/2) = 2 * Real.sin 1 := by
sorry

end NUMINAMATH_CALUDE_sector_chord_length_l2440_244015


namespace NUMINAMATH_CALUDE_curler_count_l2440_244084

theorem curler_count (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ) : 
  total = 16 →
  pink = total / 4 →
  blue = 2 * pink →
  green = total - (pink + blue) →
  green = 4 := by
  sorry

end NUMINAMATH_CALUDE_curler_count_l2440_244084


namespace NUMINAMATH_CALUDE_parabola_focus_l2440_244088

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -1/8 * x^2

-- Define symmetry about y-axis
def symmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the focus of a parabola
def focus (f h : ℝ) : Prop :=
  ∀ x y, parabola x y → (x - 0)^2 + (y - h)^2 = (y + 2)^2

-- Theorem statement
theorem parabola_focus :
  symmetricAboutYAxis (λ x => -1/8 * x^2) →
  focus 0 (-2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2440_244088


namespace NUMINAMATH_CALUDE_history_score_calculation_l2440_244032

theorem history_score_calculation (math_score : ℝ) (third_subject_score : ℝ) (desired_average : ℝ) :
  math_score = 74 →
  third_subject_score = 67 →
  desired_average = 75 →
  (math_score + third_subject_score + (3 * desired_average - math_score - third_subject_score)) / 3 = desired_average :=
by
  sorry

#check history_score_calculation

end NUMINAMATH_CALUDE_history_score_calculation_l2440_244032


namespace NUMINAMATH_CALUDE_income_distribution_l2440_244031

theorem income_distribution (income : ℝ) (h1 : income = 800000) : 
  let children_share := 0.2 * income * 3
  let wife_share := 0.3 * income
  let family_distribution := children_share + wife_share
  let remaining_after_family := income - family_distribution
  let orphan_donation := 0.05 * remaining_after_family
  let final_amount := remaining_after_family - orphan_donation
  final_amount = 76000 :=
by sorry

end NUMINAMATH_CALUDE_income_distribution_l2440_244031


namespace NUMINAMATH_CALUDE_highway_length_l2440_244028

theorem highway_length (speed1 speed2 time : ℝ) (h1 : speed1 = 13) (h2 : speed2 = 17) (h3 : time = 2) :
  (speed1 + speed2) * time = 60 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l2440_244028


namespace NUMINAMATH_CALUDE_candy_seller_problem_l2440_244083

theorem candy_seller_problem (num_clowns num_children initial_candies candies_per_person : ℕ) 
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : initial_candies = 700)
  (h4 : candies_per_person = 20) :
  initial_candies - (num_clowns + num_children) * candies_per_person = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_seller_problem_l2440_244083


namespace NUMINAMATH_CALUDE_sum_of_T_l2440_244054

/-- The sum of the geometric series for -1 < r < 1 -/
noncomputable def T (r : ℝ) : ℝ := 18 / (1 - r)

/-- Theorem: Sum of T(b) and T(-b) equals 337.5 -/
theorem sum_of_T (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 3024) :
  T b + T (-b) = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_T_l2440_244054


namespace NUMINAMATH_CALUDE_garden_border_material_l2440_244019

/-- The amount of material needed for a decorative border around a circular garden -/
theorem garden_border_material (garden_area : Real) (pi_estimate : Real) (extra_material : Real) : 
  garden_area = 50.24 → pi_estimate = 3.14 → extra_material = 4 →
  2 * pi_estimate * (garden_area / pi_estimate).sqrt + extra_material = 29.12 := by
sorry

end NUMINAMATH_CALUDE_garden_border_material_l2440_244019


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2440_244010

theorem identity_function_satisfies_equation (f : ℕ → ℕ) :
  (∀ x y : ℕ, f (x + f y) = f x + y) → (∀ x : ℕ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l2440_244010


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2440_244026

/-- The maximum value of x + 2y for points on the ellipse 2x^2 + 3y^2 = 12 is √22 -/
theorem max_value_on_ellipse :
  ∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 12 →
  ∀ z : ℝ, z = x + 2 * y →
  z ≤ Real.sqrt 22 ∧ ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 3 * y₀^2 = 12 ∧ x₀ + 2 * y₀ = Real.sqrt 22 :=
by sorry


end NUMINAMATH_CALUDE_max_value_on_ellipse_l2440_244026


namespace NUMINAMATH_CALUDE_batsman_average_after_11th_inning_l2440_244073

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalScore : ℕ
  averageScore : ℚ

/-- Calculates the new average score after an inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalScore + newInningScore : ℚ) / (stats.inningsPlayed + 1 : ℚ)

/-- Theorem: Given the conditions, the batsman's average after the 11th inning is 45 -/
theorem batsman_average_after_11th_inning
  (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 10)
  (h2 : newAverage stats 95 = stats.averageScore + 5) :
  newAverage stats 95 = 45 := by
  sorry

#check batsman_average_after_11th_inning

end NUMINAMATH_CALUDE_batsman_average_after_11th_inning_l2440_244073


namespace NUMINAMATH_CALUDE_number_problem_l2440_244040

theorem number_problem : ∃ x : ℝ, 4 * x = 28 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2440_244040


namespace NUMINAMATH_CALUDE_imaginary_part_zi_l2440_244098

def complex_coords (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem imaginary_part_zi (z : ℂ) (h : complex_coords z = (-2, 1)) : 
  (z * Complex.I).im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_zi_l2440_244098


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l2440_244090

/-- A function from positive reals to positive reals -/
def PositiveRealFunction := {f : ℝ → ℝ // ∀ x > 0, f x > 0}

/-- The property f(x f(y)) = y f(x) for all positive real x, y -/
def HasFunctionalProperty (f : PositiveRealFunction) :=
  ∀ x y, x > 0 → y > 0 → f.val (x * f.val y) = y * f.val x

/-- The property that f(x) → 0 as x → +∞ -/
def TendsToZeroAtInfinity (f : PositiveRealFunction) :=
  ∀ ε > 0, ∃ M, ∀ x > M, f.val x < ε

/-- The main theorem -/
theorem unique_function_satisfying_conditions
  (f : PositiveRealFunction)
  (h1 : HasFunctionalProperty f)
  (h2 : TendsToZeroAtInfinity f) :
  ∀ x > 0, f.val x = 1 / x :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l2440_244090


namespace NUMINAMATH_CALUDE_sum_of_digits_problem_l2440_244012

def S (n : ℕ) : ℕ := sorry  -- Sum of digits function

theorem sum_of_digits_problem (N : ℕ) 
  (h1 : S N + S (N + 1) = 200)
  (h2 : S (N + 2) + S (N + 3) = 105) :
  S (N + 1) + S (N + 2) = 103 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_problem_l2440_244012


namespace NUMINAMATH_CALUDE_largest_and_smallest_subsequence_l2440_244014

def original_number : ℕ := 798056132

-- Define a function to check if a number is a valid 5-digit subsequence of the original number
def is_valid_subsequence (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  ∃ (a b c d e : ℕ), 
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    ∃ (i j k l m : ℕ), 
      i < j ∧ j < k ∧ k < l ∧ l < m ∧
      (original_number / 10 ^ (8 - i) % 10 = a) ∧
      (original_number / 10 ^ (8 - j) % 10 = b) ∧
      (original_number / 10 ^ (8 - k) % 10 = c) ∧
      (original_number / 10 ^ (8 - l) % 10 = d) ∧
      (original_number / 10 ^ (8 - m) % 10 = e)

theorem largest_and_smallest_subsequence :
  (∀ n : ℕ, is_valid_subsequence n → n ≤ 98632) ∧
  (∀ n : ℕ, is_valid_subsequence n → n ≥ 56132) ∧
  is_valid_subsequence 98632 ∧
  is_valid_subsequence 56132 := by sorry

end NUMINAMATH_CALUDE_largest_and_smallest_subsequence_l2440_244014


namespace NUMINAMATH_CALUDE_arithmetic_mean_special_set_l2440_244093

/-- Given a set of n numbers where n > 1, one number is 1 - 2/n and all others are 1,
    the arithmetic mean of these numbers is 1 - 2/n² -/
theorem arithmetic_mean_special_set (n : ℕ) (h : n > 1) :
  let s : Finset ℕ := Finset.range n
  let f : ℕ → ℝ := fun i => if i = 0 then 1 - 2 / n else 1
  (s.sum f) / n = 1 - 2 / n^2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_special_set_l2440_244093


namespace NUMINAMATH_CALUDE_jerrys_average_score_l2440_244034

theorem jerrys_average_score (current_average : ℝ) : 
  (3 * current_average + 97) / 4 = current_average + 3 →
  current_average = 85 := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_score_l2440_244034


namespace NUMINAMATH_CALUDE_a_5_value_l2440_244091

def geometric_sequence_with_ratio_difference (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n, a (n + 2) / a (n + 1) - a (n + 1) / a n = k

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence_with_ratio_difference a 2 →
  a 1 = 1 →
  a 2 = 2 →
  a 5 = 384 := by
sorry

end NUMINAMATH_CALUDE_a_5_value_l2440_244091


namespace NUMINAMATH_CALUDE_money_distribution_l2440_244011

/-- Represents the distribution of money among x, y, and z -/
structure Distribution where
  x : ℚ  -- Amount x gets in rupees
  y : ℚ  -- Amount y gets in rupees
  z : ℚ  -- Amount z gets in rupees

/-- The problem statement and conditions -/
theorem money_distribution (d : Distribution) : 
  -- For each rupee x gets, z gets 30 paisa
  d.z = 0.3 * d.x →
  -- The share of y is Rs. 27
  d.y = 27 →
  -- The total amount is Rs. 105
  d.x + d.y + d.z = 105 →
  -- Prove that y gets 45 paisa for each rupee x gets
  d.y / d.x = 0.45 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2440_244011


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l2440_244070

/-- Systematic sampling problem -/
theorem systematic_sampling_first_number
  (population : ℕ)
  (sample_size : ℕ)
  (eighteenth_sample : ℕ)
  (h1 : population = 1000)
  (h2 : sample_size = 40)
  (h3 : eighteenth_sample = 443)
  (h4 : sample_size > 0)
  (h5 : population ≥ sample_size) :
  ∃ (first_sample : ℕ),
    first_sample + 17 * (population / sample_size) = eighteenth_sample ∧
    first_sample = 18 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l2440_244070


namespace NUMINAMATH_CALUDE_fraction_addition_l2440_244061

theorem fraction_addition (x P Q : ℚ) : 
  (8 * x^2 - 9 * x + 20) / (4 * x^3 - 5 * x^2 - 26 * x + 24) = 
  P / (2 * x^2 - 5 * x + 3) + Q / (2 * x - 3) →
  P = 4/9 ∧ Q = 68/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_addition_l2440_244061


namespace NUMINAMATH_CALUDE_interest_discount_sum_l2440_244053

/-- Given a sum, rate, and time, if the simple interest is 85 and the true discount is 80, then the sum is 1360 -/
theorem interest_discount_sum (P r t : ℝ) : 
  (P * r * t / 100 = 85) → 
  (P * r * t / (100 + r * t) = 80) → 
  P = 1360 := by
  sorry

end NUMINAMATH_CALUDE_interest_discount_sum_l2440_244053


namespace NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l2440_244007

/-- Given a quadratic equation with complex coefficients, prove the maximum and minimum absolute values of a specific coefficient. -/
theorem quadratic_coefficient_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  z₁^2 - 4*z₂ = 16 + 20*I →
  x^2 + z₁*x + z₂ + m = 0 →
  α^2 + z₁*α + z₂ + m = 0 →
  β^2 + z₁*β + z₂ + m = 0 →
  Complex.abs (α - β) = 2 * Real.sqrt 7 →
  (Complex.abs m = Real.sqrt 41 + 7 ∨ Complex.abs m = Real.sqrt 41 - 7) ∧
  ∀ m' : ℂ, Complex.abs m' ≤ Real.sqrt 41 + 7 ∧ Complex.abs m' ≥ Real.sqrt 41 - 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_bounds_l2440_244007


namespace NUMINAMATH_CALUDE_fraction_denominator_l2440_244063

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / 5 + (3 * y) / x = 0.7 * y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l2440_244063


namespace NUMINAMATH_CALUDE_second_carpenter_proof_l2440_244075

/-- The time taken by the second carpenter to complete the job alone -/
def second_carpenter_time : ℚ :=
  10 / 3

theorem second_carpenter_proof (first_carpenter_time : ℚ) 
  (first_carpenter_initial_work : ℚ) (combined_work_time : ℚ) :
  first_carpenter_time = 5 →
  first_carpenter_initial_work = 1 →
  combined_work_time = 2 →
  second_carpenter_time = 10 / 3 :=
by
  sorry

#eval second_carpenter_time

end NUMINAMATH_CALUDE_second_carpenter_proof_l2440_244075


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2440_244092

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  passesA : f 1 = 0
  passesB : f (-3) = 0
  passesC : f 0 = -3

/-- The theorem stating properties of the quadratic function -/
theorem quadratic_function_properties (qf : QuadraticFunction) :
  (∃ a b c : ℝ, ∀ x, qf.f x = a * x^2 + b * x + c) →
  (∀ x, qf.f x = x^2 + 2*x - 3) ∧
  (qf.f (-1) = -4 ∧ ∀ x, qf.f x ≥ qf.f (-1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2440_244092


namespace NUMINAMATH_CALUDE_simplify_fraction_l2440_244077

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  8 * x * y^2 / (6 * x^2 * y) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2440_244077


namespace NUMINAMATH_CALUDE_one_weighing_sufficient_l2440_244096

/-- Represents the types of balls -/
inductive BallType
| Aluminum
| Duralumin

/-- Represents a collection of balls -/
structure BallCollection where
  aluminum : ℕ
  duralumin : ℕ

/-- The mass of a ball collection -/
def mass (bc : BallCollection) : ℚ :=
  10 * bc.aluminum + 99/10 * bc.duralumin

theorem one_weighing_sufficient :
  ∃ (group1 group2 : BallCollection),
    group1.aluminum + group1.duralumin = 1000 ∧
    group2.aluminum + group2.duralumin = 1000 ∧
    group1.aluminum + group2.aluminum = 1000 ∧
    group1.duralumin + group2.duralumin = 1000 ∧
    mass group1 ≠ mass group2 :=
sorry

end NUMINAMATH_CALUDE_one_weighing_sufficient_l2440_244096


namespace NUMINAMATH_CALUDE_students_like_both_correct_l2440_244002

/-- The number of students who like both apple pie and chocolate cake -/
def students_like_both (total : ℕ) (apple : ℕ) (chocolate : ℕ) (pumpkin : ℕ) (none : ℕ) : ℕ := 
  apple + chocolate - (total - none)

theorem students_like_both_correct (total : ℕ) (apple : ℕ) (chocolate : ℕ) (pumpkin : ℕ) (none : ℕ) 
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 20)
  (h4 : pumpkin = 17)
  (h5 : none = 15) :
  students_like_both total apple chocolate pumpkin none = 7 := by
  sorry

#eval students_like_both 50 22 20 17 15

end NUMINAMATH_CALUDE_students_like_both_correct_l2440_244002


namespace NUMINAMATH_CALUDE_fourteen_percent_of_seven_hundred_is_ninety_eight_l2440_244046

theorem fourteen_percent_of_seven_hundred_is_ninety_eight :
  ∀ x : ℝ, (14 / 100) * x = 98 → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_percent_of_seven_hundred_is_ninety_eight_l2440_244046


namespace NUMINAMATH_CALUDE_christine_and_siri_money_l2440_244049

theorem christine_and_siri_money (christine_money siri_money : ℚ) : 
  christine_money = 20.5 → 
  christine_money = siri_money + 20 → 
  christine_money + siri_money = 21 := by
sorry

end NUMINAMATH_CALUDE_christine_and_siri_money_l2440_244049


namespace NUMINAMATH_CALUDE_total_peanuts_l2440_244023

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def marcos_peanuts : ℕ := kenya_peanuts + 37

theorem total_peanuts : jose_peanuts + kenya_peanuts + marcos_peanuts = 388 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l2440_244023


namespace NUMINAMATH_CALUDE_system_solution_l2440_244041

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2440_244041


namespace NUMINAMATH_CALUDE_compound_composition_l2440_244048

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- The number of fluorine atoms in the compound -/
def num_F : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 84

theorem compound_composition :
  (num_Al : ℝ) * atomic_weight_Al + (num_F : ℝ) * atomic_weight_F = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l2440_244048


namespace NUMINAMATH_CALUDE_alphabet_letters_with_dot_only_l2440_244099

theorem alphabet_letters_with_dot_only (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h_total : total = 40)
  (h_both : both = 10)
  (h_line_only : line_only = 24)
  (h_all_types : total = both + line_only + (total - both - line_only)) :
  total - both - line_only = 6 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_letters_with_dot_only_l2440_244099


namespace NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l2440_244081

theorem four_numbers_with_equal_sums (S : Finset ℕ) :
  S ⊆ Finset.range 38 →
  S.card = 10 →
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_with_equal_sums_l2440_244081


namespace NUMINAMATH_CALUDE_triangle_trigonometric_expression_l2440_244089

theorem triangle_trigonometric_expression (X Y Z : ℝ) : 
  (13 : ℝ) ^ 2 = X ^ 2 + Y ^ 2 - 2 * X * Y * Real.cos Z →
  (14 : ℝ) ^ 2 = X ^ 2 + Z ^ 2 - 2 * X * Z * Real.cos Y →
  (15 : ℝ) ^ 2 = Y ^ 2 + Z ^ 2 - 2 * Y * Z * Real.cos X →
  (Real.cos ((X - Y) / 2) / Real.sin (Z / 2)) - (Real.sin ((X - Y) / 2) / Real.cos (Z / 2)) = 28 / 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_trigonometric_expression_l2440_244089


namespace NUMINAMATH_CALUDE_simplify_expression_l2440_244058

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2440_244058


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l2440_244009

/-- Represents the type of stripe on a cube face -/
inductive StripeType
| Solid
| Dashed

/-- Represents the orientation of a stripe on a cube face -/
inductive StripeOrientation
| Horizontal
| Vertical

/-- Represents a single face configuration -/
structure FaceConfig where
  stripeType : StripeType
  orientation : StripeOrientation

/-- Represents a complete cube configuration -/
structure CubeConfig where
  faces : Fin 6 → FaceConfig

/-- Determines if a given cube configuration has a continuous stripe -/
def hasContinuousStripe (config : CubeConfig) : Bool := sorry

/-- The total number of possible cube configurations -/
def totalConfigurations : Nat := 4^6

/-- The number of configurations with a continuous stripe -/
def continuousStripeConfigurations : Nat := 3 * 16

/-- The probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  (continuousStripeConfigurations : ℚ) / totalConfigurations = 3 / 256 := by sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l2440_244009


namespace NUMINAMATH_CALUDE_robies_boxes_given_away_l2440_244021

/-- Given information about Robie's hockey cards and boxes, prove the number of boxes he gave away. -/
theorem robies_boxes_given_away
  (total_cards : ℕ)
  (cards_per_box : ℕ)
  (cards_not_in_box : ℕ)
  (boxes_with_robie : ℕ)
  (h1 : total_cards = 75)
  (h2 : cards_per_box = 10)
  (h3 : cards_not_in_box = 5)
  (h4 : boxes_with_robie = 5) :
  total_cards / cards_per_box - boxes_with_robie = 2 :=
by sorry

end NUMINAMATH_CALUDE_robies_boxes_given_away_l2440_244021


namespace NUMINAMATH_CALUDE_amazon_profit_per_package_l2440_244005

/-- Profit per package for Amazon distribution centers -/
theorem amazon_profit_per_package :
  ∀ (centers : ℕ)
    (first_center_daily_packages : ℕ)
    (second_center_multiplier : ℕ)
    (combined_weekly_profit : ℚ)
    (days_per_week : ℕ),
  centers = 2 →
  first_center_daily_packages = 10000 →
  second_center_multiplier = 3 →
  combined_weekly_profit = 14000 →
  days_per_week = 7 →
  let total_weekly_packages := first_center_daily_packages * days_per_week * (1 + second_center_multiplier)
  (combined_weekly_profit / total_weekly_packages : ℚ) = 1/20 := by
sorry

end NUMINAMATH_CALUDE_amazon_profit_per_package_l2440_244005


namespace NUMINAMATH_CALUDE_gravel_pile_volume_l2440_244052

/-- The volume of a conical pile of gravel -/
theorem gravel_pile_volume (diameter : Real) (height_ratio : Real) : 
  diameter = 10 →
  height_ratio = 0.6 →
  let height := height_ratio * diameter
  let radius := diameter / 2
  let volume := (1 / 3) * Real.pi * radius^2 * height
  volume = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_gravel_pile_volume_l2440_244052


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2440_244001

theorem binomial_expansion_coefficient (a : ℝ) (h : a ≠ 0) :
  let expansion := fun (x : ℝ) ↦ (x - a / x)^6
  let B := expansion 1  -- Constant term when x = 1
  B = 44 → a = -22/5 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2440_244001


namespace NUMINAMATH_CALUDE_complement_of_P_l2440_244066

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_of_P : 
  (Set.univ \ P) = {x | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l2440_244066
