import Mathlib

namespace NUMINAMATH_CALUDE_pascal_interior_sum_l1344_134477

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- Pascal's Triangle interior numbers start from the third row -/
def interior_start : ℕ := 3

theorem pascal_interior_sum :
  interior_sum 4 = 6 ∧
  interior_sum 5 = 14 →
  interior_sum 9 = 254 := by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l1344_134477


namespace NUMINAMATH_CALUDE_expression_equality_l1344_134440

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2/y) :
  (x^2 - 1/x^2) * (y^2 + 1/y^2) = (x^4/4) - (4/x^4) + 3.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1344_134440


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1344_134489

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1344_134489


namespace NUMINAMATH_CALUDE_triangle_angle_sum_special_case_l1344_134475

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (angle_sum : A + B + C = π)
  (law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C))

-- State the theorem
theorem triangle_angle_sum_special_case (t : Triangle) 
  (h : (t.a + t.c - t.b) * (t.a + t.c + t.b) = 3 * t.a * t.c) : 
  t.A + t.C = 2 * π / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_special_case_l1344_134475


namespace NUMINAMATH_CALUDE_sum_of_squares_l1344_134408

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 4*y = 8)
  (eq2 : y^2 + 6*z = 0)
  (eq3 : z^2 + 8*x = -16) :
  x^2 + y^2 + z^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1344_134408


namespace NUMINAMATH_CALUDE_battery_current_l1344_134449

theorem battery_current (voltage : ℝ) (resistance : ℝ) (current : ℝ → ℝ) :
  voltage = 48 →
  (∀ R, current R = voltage / R) →
  resistance = 12 →
  current resistance = 4 :=
by sorry

end NUMINAMATH_CALUDE_battery_current_l1344_134449


namespace NUMINAMATH_CALUDE_square_root_problem_l1344_134485

theorem square_root_problem (x : ℝ) (a : ℝ) 
  (h1 : x > 0)
  (h2 : Real.sqrt x = 3 * a - 4)
  (h3 : Real.sqrt x = 1 - 6 * a) :
  a = -1 ∧ x = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1344_134485


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1344_134448

def A : Set ℝ := {x | x^2 - 1 ≤ 0}
def B : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1344_134448


namespace NUMINAMATH_CALUDE_inequality_transformations_l1344_134493

theorem inequality_transformations :
  (∀ x : ℝ, x - 1 > 2 → x > 3) ∧
  (∀ x : ℝ, -4 * x > 8 → x < -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_transformations_l1344_134493


namespace NUMINAMATH_CALUDE_loan_amount_proof_l1344_134432

/-- Represents a loan with simple interest -/
structure Loan where
  principal : ℕ
  rate : ℕ
  time : ℕ
  interest : ℕ

/-- Calculates the simple interest for a loan -/
def simpleInterest (l : Loan) : ℕ :=
  l.principal * l.rate * l.time / 100

theorem loan_amount_proof (l : Loan) :
  l.rate = 8 ∧ l.time = l.rate ∧ l.interest = 704 →
  simpleInterest l = l.interest →
  l.principal = 1100 := by
sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l1344_134432


namespace NUMINAMATH_CALUDE_turtle_conservation_l1344_134480

theorem turtle_conservation (G H L : ℕ) : 
  G = 800 → H = 2 * G → L = 3 * G → G + H + L = 4800 := by
  sorry

end NUMINAMATH_CALUDE_turtle_conservation_l1344_134480


namespace NUMINAMATH_CALUDE_triple_tangent_identity_l1344_134452

theorem triple_tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) =
  ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) := by
  sorry

end NUMINAMATH_CALUDE_triple_tangent_identity_l1344_134452


namespace NUMINAMATH_CALUDE_set_operations_l1344_134441

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

def B : Set ℝ := {x | -1 < x ∧ x < 5}

theorem set_operations :
  (A ∪ B = {x | -3 ≤ x ∧ x < 5}) ∧
  (A ∩ B = {x | -1 < x ∧ x ≤ 4}) ∧
  ((U \ A) ∩ B = {x | 4 < x ∧ x < 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1344_134441


namespace NUMINAMATH_CALUDE_angle_supplement_l1344_134454

theorem angle_supplement (x : ℝ) : 
  (90 - x = 150) → (180 - x = 60) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l1344_134454


namespace NUMINAMATH_CALUDE_certain_number_proof_l1344_134433

theorem certain_number_proof :
  ∃ x : ℝ, x * (-4.5) = 2 * (-4.5) - 36 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1344_134433


namespace NUMINAMATH_CALUDE_gina_money_theorem_l1344_134451

theorem gina_money_theorem (initial_amount : ℚ) : 
  initial_amount = 400 → 
  initial_amount - (initial_amount * (1/4 + 1/8 + 1/5)) = 170 := by
  sorry

end NUMINAMATH_CALUDE_gina_money_theorem_l1344_134451


namespace NUMINAMATH_CALUDE_smallest_equal_purchase_l1344_134417

theorem smallest_equal_purchase (nuts : Nat) (bolts : Nat) (washers : Nat)
  (h_nuts : nuts = 13)
  (h_bolts : bolts = 8)
  (h_washers : washers = 17) :
  Nat.lcm (Nat.lcm nuts bolts) washers = 1768 := by
  sorry

end NUMINAMATH_CALUDE_smallest_equal_purchase_l1344_134417


namespace NUMINAMATH_CALUDE_fifth_from_end_l1344_134499

-- Define the sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 11

-- Define the final term
def final_term (a : ℕ → ℕ) (final : ℕ) : Prop :=
  ∃ k : ℕ, a k = final ∧ ∀ n > k, a n > final

-- Theorem statement
theorem fifth_from_end (a : ℕ → ℕ) :
  arithmetic_sequence a →
  final_term a 89 →
  ∃ k : ℕ, a k = 45 ∧ a (k + 4) = 89 :=
by sorry

end NUMINAMATH_CALUDE_fifth_from_end_l1344_134499


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_sides_l1344_134481

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℚ := n * (n - 3) / 2

/-- Theorem: A regular polygon whose number of diagonals is three times its number of sides has 9 sides -/
theorem regular_polygon_diagonals_sides : ∃ n : ℕ, n > 2 ∧ num_diagonals n = 3 * n ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_sides_l1344_134481


namespace NUMINAMATH_CALUDE_existence_of_relatively_prime_divisible_combination_l1344_134421

theorem existence_of_relatively_prime_divisible_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_relatively_prime_divisible_combination_l1344_134421


namespace NUMINAMATH_CALUDE_unshaded_parts_sum_l1344_134426

theorem unshaded_parts_sum (square_area shaded_area : ℝ) 
  (h1 : square_area = 36) 
  (h2 : shaded_area = 27) 
  (p q r s : ℝ) :
  p + q + r + s = 9 := by sorry

end NUMINAMATH_CALUDE_unshaded_parts_sum_l1344_134426


namespace NUMINAMATH_CALUDE_sharadek_word_guessing_l1344_134401

theorem sharadek_word_guessing (n : ℕ) (h : n ≤ 1000000) :
  ∃ (q : ℕ), q ≤ 20 ∧ 2^q ≥ n := by
  sorry

end NUMINAMATH_CALUDE_sharadek_word_guessing_l1344_134401


namespace NUMINAMATH_CALUDE_hotel_towels_l1344_134479

theorem hotel_towels (rooms : ℕ) (people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : rooms = 20)
  (h2 : people_per_room = 5)
  (h3 : towels_per_person = 3) :
  rooms * people_per_room * towels_per_person = 300 :=
by sorry

end NUMINAMATH_CALUDE_hotel_towels_l1344_134479


namespace NUMINAMATH_CALUDE_abs_neg_three_l1344_134450

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_l1344_134450


namespace NUMINAMATH_CALUDE_circumcenter_from_equal_distances_l1344_134490

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is outside a plane -/
def isOutsidePlane (P : Point3D) (t : Triangle3D) : Prop := sorry

/-- Check if a line is perpendicular to a plane -/
def isPerpendicularToPlane (P O : Point3D) (t : Triangle3D) : Prop := sorry

/-- Check if a point is the foot of a perpendicular -/
def isFootOfPerpendicular (O : Point3D) (P : Point3D) (t : Triangle3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (P Q : Point3D) : ℝ := sorry

/-- Check if a point is the circumcenter of a triangle -/
def isCircumcenter (O : Point3D) (t : Triangle3D) : Prop := sorry

/-- Main theorem -/
theorem circumcenter_from_equal_distances (P O : Point3D) (t : Triangle3D) :
  isOutsidePlane P t →
  isPerpendicularToPlane P O t →
  isFootOfPerpendicular O P t →
  distance P t.A = distance P t.B ∧ distance P t.B = distance P t.C →
  isCircumcenter O t := by sorry

end NUMINAMATH_CALUDE_circumcenter_from_equal_distances_l1344_134490


namespace NUMINAMATH_CALUDE_profit_equation_correct_l1344_134465

/-- Represents the profit scenario for a product with varying price and sales volume. -/
def profit_equation (x : ℝ) : Prop :=
  let initial_purchase_price : ℝ := 35
  let initial_selling_price : ℝ := 40
  let initial_sales_volume : ℝ := 200
  let price_increase : ℝ := x
  let sales_volume_decrease : ℝ := 5 * x
  let new_profit_per_unit : ℝ := (initial_selling_price + price_increase) - initial_purchase_price
  let new_sales_volume : ℝ := initial_sales_volume - sales_volume_decrease
  let total_profit : ℝ := 1870
  (new_profit_per_unit * new_sales_volume) = total_profit

/-- Theorem stating that the given equation correctly represents the profit scenario. -/
theorem profit_equation_correct :
  ∀ x : ℝ, profit_equation x ↔ (x + 5) * (200 - 5 * x) = 1870 :=
sorry

end NUMINAMATH_CALUDE_profit_equation_correct_l1344_134465


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l1344_134419

theorem last_two_digits_sum (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l1344_134419


namespace NUMINAMATH_CALUDE_scientific_notation_of_value_l1344_134436

-- Define the nanometer to meter conversion
def nm_to_m : ℝ := 1e-9

-- Define the value in meters
def value_in_meters : ℝ := 7 * nm_to_m

-- Theorem statement
theorem scientific_notation_of_value :
  ∃ (a : ℝ) (n : ℤ), value_in_meters = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_value_l1344_134436


namespace NUMINAMATH_CALUDE_alcohol_solution_concentration_l1344_134476

/-- Proves that adding 1.8 liters of pure alcohol to a 6-liter solution
    that is 35% alcohol results in a 50% alcohol solution. -/
theorem alcohol_solution_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_alcohol : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.35)
  (h3 : added_alcohol = 1.8)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_solution_concentration_l1344_134476


namespace NUMINAMATH_CALUDE_min_value_T_l1344_134425

/-- Given a quadratic inequality (1/a)x² + bx + c ≤ 0 with solution set ℝ and b > 0,
    the minimum value of T = (5 + 2ab + 4ac) / (ab + 1) is 4 -/
theorem min_value_T (a b c : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, (1/a) * x^2 + b * x + c ≤ 0) : 
  (∀ t, t = (5 + 2*a*b + 4*a*c) / (a*b + 1) → t ≥ 4) ∧ 
  (∃ t, t = (5 + 2*a*b + 4*a*c) / (a*b + 1) ∧ t = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_T_l1344_134425


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_million_l1344_134418

theorem smallest_n_exceeding_million (n : ℕ) : ∃ (n : ℕ), n > 0 ∧ 
  (∀ k < n, (12 : ℝ) ^ ((k * (k + 1) : ℝ) / (2 * 13)) ≤ 1000000) ∧
  (12 : ℝ) ^ ((n * (n + 1) : ℝ) / (2 * 13)) > 1000000 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_million_l1344_134418


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1344_134413

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = q * a n

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a) 
  (h_cond : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  (∀ q, q > 0 → 2 * a 5 + a 4 ≥ 12 * Real.sqrt 3) ∧
  (∃ q, q > 0 ∧ 2 * a 5 + a 4 = 12 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1344_134413


namespace NUMINAMATH_CALUDE_daniels_age_l1344_134498

/-- Given the ages of Uncle Ben, Edward, and Daniel, prove Daniel's age --/
theorem daniels_age (uncle_ben_age : ℚ) (edward_age : ℚ) (daniel_age : ℚ) : 
  uncle_ben_age = 50 →
  edward_age = 2/3 * uncle_ben_age →
  daniel_age = edward_age - 7 →
  daniel_age = 79/3 := by sorry

end NUMINAMATH_CALUDE_daniels_age_l1344_134498


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1344_134443

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1344_134443


namespace NUMINAMATH_CALUDE_specific_frustum_smaller_cone_altitude_l1344_134469

/-- Represents a frustum of a right circular cone. -/
structure Frustum where
  altitude : ℝ
  largerBaseArea : ℝ
  smallerBaseArea : ℝ

/-- Calculates the altitude of the smaller cone removed from a frustum. -/
def smallerConeAltitude (f : Frustum) : ℝ :=
  sorry

/-- Theorem stating that for a specific frustum, the altitude of the smaller cone is 15. -/
theorem specific_frustum_smaller_cone_altitude :
  let f : Frustum := { altitude := 15, largerBaseArea := 64 * Real.pi, smallerBaseArea := 16 * Real.pi }
  smallerConeAltitude f = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_frustum_smaller_cone_altitude_l1344_134469


namespace NUMINAMATH_CALUDE_square_root_and_cube_l1344_134472

theorem square_root_and_cube (x : ℝ) (x_nonzero : x ≠ 0) :
  (Real.sqrt 144 + 3^3) / x = 39 / x :=
sorry

end NUMINAMATH_CALUDE_square_root_and_cube_l1344_134472


namespace NUMINAMATH_CALUDE_train_crossing_time_l1344_134423

/-- Proves that the time taken for the first train to cross a telegraph post is 10 seconds,
    given the conditions of the problem. -/
theorem train_crossing_time
  (train_length : ℝ)
  (second_train_time : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : second_train_time = 15)
  (h3 : crossing_time = 12) :
  let second_train_speed := train_length / second_train_time
  let relative_speed := 2 * train_length / crossing_time
  let first_train_speed := relative_speed - second_train_speed
  train_length / first_train_speed = 10 :=
by sorry


end NUMINAMATH_CALUDE_train_crossing_time_l1344_134423


namespace NUMINAMATH_CALUDE_sqrt_x_minus_y_equals_plus_minus_two_l1344_134463

theorem sqrt_x_minus_y_equals_plus_minus_two
  (x y : ℝ) 
  (h : Real.sqrt (x - 3) + 2 * abs (y + 1) = 0) :
  Real.sqrt (x - y) = 2 ∨ Real.sqrt (x - y) = -2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_y_equals_plus_minus_two_l1344_134463


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l1344_134428

/-- A piecewise function f: ℝ → ℝ defined by two parts -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then (2*a - 1)*x + 7*a - 2 else a^x

/-- Theorem stating the condition for f to be monotonically decreasing -/
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) ↔ 3/8 ≤ a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l1344_134428


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1344_134411

/-- A function f: ℝ → ℝ has an extreme value at x₀ -/
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

/-- The statement that f'(x₀) = 0 is a necessary but not sufficient condition
    for f to have an extreme value at x₀ -/
theorem extreme_value_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, has_extreme_value f x₀ → (deriv f) x₀ = 0) ∧
  ¬(∀ x₀ : ℝ, (deriv f) x₀ = 0 → has_extreme_value f x₀) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1344_134411


namespace NUMINAMATH_CALUDE_simplify_expression_l1344_134446

theorem simplify_expression :
  3 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 3 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1344_134446


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1344_134473

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ r) (h2 : k ≠ 0) :
  k * p^2 - k * r^2 = 4 * (k * p - k * r) → p + r = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1344_134473


namespace NUMINAMATH_CALUDE_mothers_carrots_count_l1344_134410

/-- The number of carrots Olivia picked -/
def olivias_carrots : ℕ := 20

/-- The total number of good carrots -/
def good_carrots : ℕ := 19

/-- The total number of bad carrots -/
def bad_carrots : ℕ := 15

/-- The number of carrots Olivia's mother picked -/
def mothers_carrots : ℕ := (good_carrots + bad_carrots) - olivias_carrots

theorem mothers_carrots_count : mothers_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_mothers_carrots_count_l1344_134410


namespace NUMINAMATH_CALUDE_g_of_two_equals_six_l1344_134464

/-- Given a function g where g(x) = 5x - 4 for all x, prove that g(2) = 6 -/
theorem g_of_two_equals_six (g : ℝ → ℝ) (h : ∀ x, g x = 5 * x - 4) : g 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_equals_six_l1344_134464


namespace NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l1344_134402

/-- The sequence x_n = sin(n^2) does not converge to zero. -/
theorem sin_n_squared_not_converge_to_zero :
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |Real.sin (n^2)| < ε) := by
  sorry

end NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l1344_134402


namespace NUMINAMATH_CALUDE_multiply_24_99_l1344_134416

theorem multiply_24_99 : 24 * 99 = 2376 := by
  sorry

end NUMINAMATH_CALUDE_multiply_24_99_l1344_134416


namespace NUMINAMATH_CALUDE_solution_sets_l1344_134478

def f (a x : ℝ) : ℝ := a * x^2 + x - a

theorem solution_sets (a : ℝ) :
  (a = 1 → {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1 ∨ x < -2}) ∧
  (a < 0 →
    (a < -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | -(a + 1)/a < x ∧ x < 1}) ∧
    (a = -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | x ≠ 1}) ∧
    (0 > a ∧ a > -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | 1 < x ∧ x < -(a + 1)/a})) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l1344_134478


namespace NUMINAMATH_CALUDE_unique_amount_theorem_l1344_134424

def is_multiple_of_50 (n : ℕ) : Prop := ∃ k : ℕ, n = 50 * k

def min_banknotes (amount : ℕ) (max_denom : ℕ) : ℕ :=
  sorry

theorem unique_amount_theorem (amount : ℕ) : 
  is_multiple_of_50 amount →
  min_banknotes amount 5000 ≥ 15 →
  min_banknotes amount 1000 ≥ 35 →
  amount = 29950 :=
sorry

end NUMINAMATH_CALUDE_unique_amount_theorem_l1344_134424


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1344_134494

theorem absolute_value_equation_product (x : ℝ) :
  (∀ x, |x - 5| - 4 = 3 → x = 12 ∨ x = -2) ∧
  (∃ x₁ x₂, |x₁ - 5| - 4 = 3 ∧ |x₂ - 5| - 4 = 3 ∧ x₁ * x₂ = -24) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1344_134494


namespace NUMINAMATH_CALUDE_throne_occupant_identity_l1344_134430

-- Define the possible species
inductive Species
| Human
| Monkey

-- Define the possible truth-telling nature
inductive Nature
| Knight
| Liar

-- Define the statement made by A
def statement (s : Species) (n : Nature) : Prop :=
  ¬(s = Species.Monkey ∧ n = Nature.Knight)

-- Theorem to prove
theorem throne_occupant_identity :
  ∃ (s : Species) (n : Nature),
    statement s n ∧
    (n = Nature.Knight → statement s n = True) ∧
    (n = Nature.Liar → statement s n = False) ∧
    s = Species.Human ∧
    n = Nature.Knight := by
  sorry

end NUMINAMATH_CALUDE_throne_occupant_identity_l1344_134430


namespace NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l1344_134486

theorem reciprocal_of_sqrt_two : Real.sqrt 2 * (Real.sqrt 2 / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l1344_134486


namespace NUMINAMATH_CALUDE_max_rods_in_box_l1344_134456

/-- A rod with dimensions 1×1×4 -/
structure Rod :=
  (length : ℕ := 4)
  (width : ℕ := 1)
  (height : ℕ := 1)

/-- A cube-shaped box with dimensions 6×6×6 -/
structure Box :=
  (length : ℕ := 6)
  (width : ℕ := 6)
  (height : ℕ := 6)

/-- Predicate to check if a rod can be placed parallel to the box faces -/
def isParallel (r : Rod) (b : Box) : Prop :=
  (r.length ≤ b.length ∧ r.width ≤ b.width ∧ r.height ≤ b.height) ∨
  (r.length ≤ b.width ∧ r.width ≤ b.height ∧ r.height ≤ b.length) ∨
  (r.length ≤ b.height ∧ r.width ≤ b.length ∧ r.height ≤ b.width)

/-- The maximum number of rods that can fit in the box -/
def maxRods (r : Rod) (b : Box) : ℕ := 52

/-- Theorem stating that 52 is the maximum number of 1×1×4 rods that can fit in a 6×6×6 box -/
theorem max_rods_in_box (r : Rod) (b : Box) :
  isParallel r b → maxRods r b = 52 ∧ ¬∃ n : ℕ, n > 52 ∧ n * r.length * r.width * r.height ≤ b.length * b.width * b.height :=
sorry


end NUMINAMATH_CALUDE_max_rods_in_box_l1344_134456


namespace NUMINAMATH_CALUDE_no_real_solution_l1344_134453

theorem no_real_solution : ¬ ∃ (x : ℝ), (3 / (x^2 - x - 6) = 2 / (x^2 - 3*x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l1344_134453


namespace NUMINAMATH_CALUDE_josh_remaining_money_l1344_134437

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℝ) (hat_cost : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (num_cookies : ℕ) : ℝ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * num_cookies)

/-- Proves that Josh has $3 left after his purchases -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l1344_134437


namespace NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l1344_134496

theorem quadratic_equation_real_solutions (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l1344_134496


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1344_134431

theorem sqrt_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 4 * Real.sqrt (2 + x) + 4 * Real.sqrt (2 - x) = 6 * Real.sqrt 3 ∧ x = (3 * Real.sqrt 15) / 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1344_134431


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1344_134492

theorem trigonometric_expression_equality (x : Real) :
  (x > π / 2 ∧ x < π) →  -- x is in the second quadrant
  (Real.tan x)^2 + 3 * Real.tan x - 4 = 0 →
  (Real.sin x + Real.cos x) / (2 * Real.sin x - Real.cos x) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1344_134492


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l1344_134497

theorem number_satisfying_equation : ∃ x : ℝ, (0.08 * x) + (0.10 * 40) = 5.92 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l1344_134497


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l1344_134495

/-- Given a train that travels 40 km in one hour (including stoppages) and stops for 20 minutes each hour, 
    its speed excluding stoppages is 60 kmph. -/
theorem train_speed_excluding_stoppages : 
  ∀ (speed_with_stops : ℝ) (stop_time : ℝ) (total_time : ℝ),
  speed_with_stops = 40 →
  stop_time = 20 →
  total_time = 60 →
  (total_time - stop_time) / total_time * speed_with_stops = 60 := by
sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l1344_134495


namespace NUMINAMATH_CALUDE_profit_percentage_example_l1344_134470

/-- Calculates the percentage of profit given the cost price and selling price -/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that for a cost price of $600 and a selling price of $648, the percentage profit is 8% -/
theorem profit_percentage_example :
  percentage_profit 600 648 = 8 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_example_l1344_134470


namespace NUMINAMATH_CALUDE_min_area_with_prime_dimension_l1344_134484

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if a number is prime. -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Checks if at least one dimension of a rectangle is prime. -/
def hasOnePrimeDimension (r : Rectangle) : Prop := isPrime r.length ∨ isPrime r.width

/-- The main theorem stating the minimum area of a rectangle with given conditions. -/
theorem min_area_with_prime_dimension :
  ∀ r : Rectangle,
    r.length > 0 ∧ r.width > 0 →
    perimeter r = 120 →
    hasOnePrimeDimension r →
    (∀ r' : Rectangle, 
      r'.length > 0 ∧ r'.width > 0 →
      perimeter r' = 120 →
      hasOnePrimeDimension r' →
      area r ≤ area r') →
    area r = 116 :=
sorry

end NUMINAMATH_CALUDE_min_area_with_prime_dimension_l1344_134484


namespace NUMINAMATH_CALUDE_john_hotel_cost_l1344_134462

/-- Calculates the total cost of a hotel stay with a discount -/
def hotel_cost (nights : ℕ) (price_per_night : ℕ) (discount : ℕ) : ℕ :=
  nights * price_per_night - discount

/-- Proves that a 3-night stay at $250 per night with a $100 discount costs $650 -/
theorem john_hotel_cost : hotel_cost 3 250 100 = 650 := by
  sorry

end NUMINAMATH_CALUDE_john_hotel_cost_l1344_134462


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1344_134420

/-- Calculates the number of students in both band and chorus -/
def students_in_both (total : ℕ) (band : ℕ) (chorus : ℕ) (band_or_chorus : ℕ) : ℕ :=
  band + chorus - band_or_chorus

/-- Proves that the number of students in both band and chorus is 30 -/
theorem students_in_both_band_and_chorus :
  students_in_both 300 110 140 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1344_134420


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1344_134458

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := -2, y := 4, z := 1 }
  let a : Plane := { a := 3, b := 1, c := 2, d := 2 }
  let k : ℝ := 3
  let a' := transformPlane a k
  ¬ pointOnPlane A a' := by sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1344_134458


namespace NUMINAMATH_CALUDE_inequality_proof_l1344_134467

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  b^2 / a < a^2 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1344_134467


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1344_134429

theorem pure_imaginary_product (m : ℝ) : 
  (∃ (z : ℂ), z * z = -1 ∧ (Complex.mk 2 (-m) * Complex.mk 1 (-1)).re = 0 ∧ (Complex.mk 2 (-m) * Complex.mk 1 (-1)).im ≠ 0) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1344_134429


namespace NUMINAMATH_CALUDE_distance_AB_is_five_halves_l1344_134455

/-- Line represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Line represented by a linear equation ax + by = c -/
structure LinearLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def l₁ : ParametricLine :=
  { x := λ t => 1 + 3 * t
    y := λ t => 2 - 4 * t }

def l₂ : LinearLine :=
  { a := 2
    b := -4
    c := 5 }

def A : Point :=
  { x := 1
    y := 2 }

/-- Function to find the intersection point of a parametric line and a linear line -/
def intersection (pl : ParametricLine) (ll : LinearLine) : Point :=
  sorry

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem distance_AB_is_five_halves :
  distance A (intersection l₁ l₂) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_five_halves_l1344_134455


namespace NUMINAMATH_CALUDE_hash_eight_three_l1344_134447

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the theorem
theorem hash_eight_three : hash 8 3 = 127 :=
  by
    sorry

-- Define the conditions
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + 2 * s + 1

end NUMINAMATH_CALUDE_hash_eight_three_l1344_134447


namespace NUMINAMATH_CALUDE_characterize_S_l1344_134405

-- Define the set of possible values for 1/x + 1/y
def S : Set ℝ := { z | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ z = 1/x + 1/y }

-- Theorem statement
theorem characterize_S : S = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_characterize_S_l1344_134405


namespace NUMINAMATH_CALUDE_smallest_wonder_number_l1344_134459

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Predicate for a "wonder number" -/
def is_wonder_number (n : ℕ) : Prop :=
  (digit_sum n = digit_sum (3 * n)) ∧ 
  (digit_sum n ≠ digit_sum (2 * n))

/-- Theorem stating that 144 is the smallest wonder number -/
theorem smallest_wonder_number : 
  is_wonder_number 144 ∧ ∀ n < 144, ¬is_wonder_number n := by sorry

end NUMINAMATH_CALUDE_smallest_wonder_number_l1344_134459


namespace NUMINAMATH_CALUDE_hemisphere_to_sphere_surface_area_l1344_134438

/-- Given a hemisphere with base area 81π, prove that the total surface area
    of the sphere obtained by adding a top circular lid is 324π. -/
theorem hemisphere_to_sphere_surface_area :
  ∀ r : ℝ,
  r > 0 →
  π * r^2 = 81 * π →
  4 * π * r^2 = 324 * π := by
sorry

end NUMINAMATH_CALUDE_hemisphere_to_sphere_surface_area_l1344_134438


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1344_134482

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem to prove
theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1344_134482


namespace NUMINAMATH_CALUDE_mutually_exclusive_complementary_l1344_134407

-- Define the sample space
def Ω : Type := Unit

-- Define the events
def A : Set Ω := sorry
def B : Set Ω := sorry
def C : Set Ω := sorry
def D : Set Ω := sorry

-- Theorem for mutually exclusive events
theorem mutually_exclusive :
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (B ∩ D = ∅) :=
sorry

-- Theorem for complementary events
theorem complementary :
  B ∪ D = Set.univ ∧ B ∩ D = ∅ :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_complementary_l1344_134407


namespace NUMINAMATH_CALUDE_color_drawing_percentage_increase_l1344_134442

def black_and_white_cost : ℝ := 160
def color_cost : ℝ := 240

theorem color_drawing_percentage_increase : 
  (color_cost - black_and_white_cost) / black_and_white_cost * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_color_drawing_percentage_increase_l1344_134442


namespace NUMINAMATH_CALUDE_martha_coffee_savings_l1344_134461

/-- Calculates the annual savings from reducing coffee spending --/
def coffee_savings (latte_price : ℚ) (latte_days : ℕ) (ice_coffee_price : ℚ) (ice_coffee_days : ℕ) (reduction_percent : ℚ) : ℚ :=
  let weekly_latte_cost := latte_price * latte_days
  let weekly_ice_coffee_cost := ice_coffee_price * ice_coffee_days
  let weekly_total_cost := weekly_latte_cost + weekly_ice_coffee_cost
  let annual_cost := weekly_total_cost * 52
  annual_cost * reduction_percent

theorem martha_coffee_savings :
  coffee_savings 4 5 2 3 (1/4) = 338 :=
sorry

end NUMINAMATH_CALUDE_martha_coffee_savings_l1344_134461


namespace NUMINAMATH_CALUDE_jill_study_time_l1344_134406

/-- Calculates the total minutes Jill studies over 3 days given her study pattern -/
def total_study_minutes (day1_hours : ℕ) : ℕ :=
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60

/-- Proves that Jill's study pattern results in 540 minutes of total study time -/
theorem jill_study_time : total_study_minutes 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_jill_study_time_l1344_134406


namespace NUMINAMATH_CALUDE_square_root_problem_l1344_134414

theorem square_root_problem (x : ℝ) : 
  Real.sqrt x = 3.87 → x = 14.9769 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1344_134414


namespace NUMINAMATH_CALUDE_yard_area_l1344_134445

/-- The area of a rectangular yard with a square cut out -/
theorem yard_area (length width cut_size : ℕ) (h1 : length = 20) (h2 : width = 18) (h3 : cut_size = 4) :
  length * width - cut_size * cut_size = 344 := by
sorry

end NUMINAMATH_CALUDE_yard_area_l1344_134445


namespace NUMINAMATH_CALUDE_pies_sold_in_week_l1344_134412

/-- The number of pies sold daily by the restaurant -/
def daily_sales : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def weekly_sales : ℕ := daily_sales * days_in_week

theorem pies_sold_in_week : weekly_sales = 56 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_in_week_l1344_134412


namespace NUMINAMATH_CALUDE_correct_probability_l1344_134400

def total_rolls : ℕ := 12
def rolls_per_type : ℕ := 3
def num_types : ℕ := 4
def rolls_per_guest : ℕ := 4

def probability_correct_selection : ℚ :=
  (rolls_per_type * (rolls_per_type - 1) * (rolls_per_type - 2)) /
  (total_rolls * (total_rolls - 1) * (total_rolls - 2) * (total_rolls - 3))

theorem correct_probability :
  probability_correct_selection = 9 / 55 := by sorry

end NUMINAMATH_CALUDE_correct_probability_l1344_134400


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l1344_134439

theorem quadratic_solution_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 - x - (m + 1) = 0) →
  m ∈ Set.Icc (-5/4) 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l1344_134439


namespace NUMINAMATH_CALUDE_stadium_attendance_l1344_134403

theorem stadium_attendance (total : ℕ) (girls : ℕ) 
  (h1 : total = 600) 
  (h2 : girls = 240) : 
  total - ((total - girls) / 4 + girls / 8) = 480 := by
  sorry

end NUMINAMATH_CALUDE_stadium_attendance_l1344_134403


namespace NUMINAMATH_CALUDE_quadratic_properties_l1344_134466

/-- A quadratic function passing through (3, -1) -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 4

theorem quadratic_properties (a b : ℝ) (h_a : a ≠ 0) :
  let f := quadratic_function a b
  -- The function passes through (3, -1)
  (f 3 = -1) →
  -- (2, 2-2a) does not lie on the graph
  (f 2 ≠ 2 - 2*a) ∧
  -- When the graph intersects the x-axis at only one point
  (∃ x : ℝ, (f x = 0 ∧ ∀ y : ℝ, f y = 0 → y = x)) →
  -- The function is either y = -x^2 + 4x - 4 or y = -1/9x^2 + 4/3x - 4
  ((a = -1 ∧ b = 4) ∨ (a = -1/9 ∧ b = 4/3)) ∧
  -- When the graph passes through points (x₁, y₁) and (x₂, y₂) with x₁ < x₂ ≤ 2/3 and y₁ > y₂
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → x₂ ≤ 2/3 → f x₁ = y₁ → f x₂ = y₂ → y₁ > y₂ → a ≥ 3/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1344_134466


namespace NUMINAMATH_CALUDE_min_value_when_m_eq_one_m_range_when_f_geq_2x_l1344_134409

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| + |m * x - 1|

-- Part 1
theorem min_value_when_m_eq_one :
  (∃ (min : ℝ), ∀ x, f 1 x ≥ min ∧ ∃ x₀ ∈ Set.Icc (-1) 1, f 1 x₀ = min) ∧
  (∀ x, f 1 x = 2 ↔ x ∈ Set.Icc (-1) 1) := by sorry

-- Part 2
theorem m_range_when_f_geq_2x :
  (∀ x, f m x ≥ 2 * x) ↔ m ∈ Set.Iic (-1) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_min_value_when_m_eq_one_m_range_when_f_geq_2x_l1344_134409


namespace NUMINAMATH_CALUDE_second_hour_distance_l1344_134471

/-- Represents a 3-hour bike ride with specific distance relationships --/
structure BikeRide where
  first_hour : ℝ
  second_hour : ℝ
  third_hour : ℝ
  second_hour_condition : second_hour = 1.2 * first_hour
  third_hour_condition : third_hour = 1.25 * second_hour
  total_distance : first_hour + second_hour + third_hour = 37

/-- Theorem stating that the distance traveled in the second hour is 12 miles --/
theorem second_hour_distance (ride : BikeRide) : ride.second_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_hour_distance_l1344_134471


namespace NUMINAMATH_CALUDE_constant_speed_travel_time_l1344_134457

/-- 
Given:
- A person drives 120 miles in 3 hours
- The person maintains the same speed for another trip of 200 miles
Prove: The second trip will take 5 hours
-/
theorem constant_speed_travel_time 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) 
  (h1 : distance1 = 120) 
  (h2 : time1 = 3) 
  (h3 : distance2 = 200) : 
  (distance2 / (distance1 / time1)) = 5 := by
  sorry

#check constant_speed_travel_time

end NUMINAMATH_CALUDE_constant_speed_travel_time_l1344_134457


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l1344_134434

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (l : Line) (α β : Plane) 
  (h1 : perp l β) (h2 : para α β) : 
  perp l α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l1344_134434


namespace NUMINAMATH_CALUDE_same_color_probability_l1344_134488

/-- The probability of drawing two balls of the same color from a bag containing 4 green balls and 8 white balls. -/
theorem same_color_probability (green : ℕ) (white : ℕ) (h1 : green = 4) (h2 : white = 8) :
  let total := green + white
  let p_green := green / total
  let p_white := white / total
  let p_same_color := (p_green * (green - 1) / (total - 1)) + (p_white * (white - 1) / (total - 1))
  p_same_color = 17 / 33 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1344_134488


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_product_l1344_134468

theorem divisibility_implies_sum_product (p q r s : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s = 
    (x^4 + 4*x^3 + 6*x^2 + 4*x + 1) * k) →
  (p + q + r) * s = -2.2 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_product_l1344_134468


namespace NUMINAMATH_CALUDE_triangle_area_l1344_134422

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  A = 30 * π / 180 →
  B = 60 * π / 180 →
  C = π - A - B →
  b = a * Real.sin B / Real.sin A →
  (1/2) * a * b * Real.sin C = (9 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1344_134422


namespace NUMINAMATH_CALUDE_circle_min_area_l1344_134427

/-- Given positive real numbers x and y satisfying the equation 3/(2+x) + 3/(2+y) = 1,
    this theorem states that (x-4)^2 + (y-4)^2 = 256 is the equation of the circle
    with center (x,y) and radius xy when its area is minimized. -/
theorem circle_min_area (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : 3 / (2 + x) + 3 / (2 + y) = 1) :
    ∃ (center_x center_y : ℝ),
      (x - center_x)^2 + (y - center_y)^2 = (x * y)^2 ∧
      center_x = 4 ∧ center_y = 4 ∧ x * y = 16 ∧
      ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / (2 + x') + 3 / (2 + y') = 1 →
        x' * y' ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_min_area_l1344_134427


namespace NUMINAMATH_CALUDE_whisker_ratio_proof_l1344_134404

/-- The number of whiskers Princess Puff has -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers : ℕ := 22

/-- The number of whiskers Catman Do is missing compared to the ratio -/
def missing_whiskers : ℕ := 6

/-- The ratio of Catman Do's whiskers to Princess Puff's whiskers -/
def whisker_ratio : ℚ := 2 / 1

theorem whisker_ratio_proof :
  (catman_do_whiskers + missing_whiskers : ℚ) / princess_puff_whiskers = whisker_ratio :=
sorry

end NUMINAMATH_CALUDE_whisker_ratio_proof_l1344_134404


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_l1344_134460

theorem mashed_potatoes_count (tomatoes bacon : ℕ) 
  (h1 : tomatoes = 79)
  (h2 : bacon = 467) : 
  ∃ mashed_potatoes : ℕ, mashed_potatoes = tomatoes + 65 ∧ mashed_potatoes = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_l1344_134460


namespace NUMINAMATH_CALUDE_no_right_triangle_with_sides_x_2x_3x_l1344_134474

theorem no_right_triangle_with_sides_x_2x_3x :
  ¬ ∃ (x : ℝ), x > 0 ∧ x^2 + (2*x)^2 = (3*x)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_sides_x_2x_3x_l1344_134474


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1344_134487

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P * (1 + r/100)^2 = 3650 ∧ 
  P * (1 + r/100)^3 = 4015 → 
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1344_134487


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_nine_l1344_134491

theorem unique_three_digit_divisible_by_nine : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 4 ∧          -- units digit is 4
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 9 = 0 ∧           -- divisible by 9
    n = 684               -- the number is 684
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_nine_l1344_134491


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1344_134483

/-- The range of k for which the quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ 
    (k - 1) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1344_134483


namespace NUMINAMATH_CALUDE_probability_shaded_is_two_fifths_l1344_134444

/-- A structure representing the triangle selection scenario -/
structure TriangleSelection where
  total_triangles : ℕ
  shaded_triangles : ℕ
  shaded_triangles_le_total : shaded_triangles ≤ total_triangles

/-- The probability of selecting a triangle with a shaded part -/
def probability_shaded (ts : TriangleSelection) : ℚ :=
  ts.shaded_triangles / ts.total_triangles

/-- Theorem stating that the probability of selecting a shaded triangle is 2/5 -/
theorem probability_shaded_is_two_fifths (ts : TriangleSelection) 
    (h1 : ts.total_triangles = 5)
    (h2 : ts.shaded_triangles = 2) : 
  probability_shaded ts = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_shaded_is_two_fifths_l1344_134444


namespace NUMINAMATH_CALUDE_equation_solutions_l1344_134435

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ x = (3 + Real.sqrt 17) / 4 ∨ x = (3 - Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1344_134435


namespace NUMINAMATH_CALUDE_mall_entrance_exit_ways_l1344_134415

theorem mall_entrance_exit_ways (n : Nat) (h : n = 4) : 
  (n * (n - 1) : Nat) = 12 := by
  sorry

end NUMINAMATH_CALUDE_mall_entrance_exit_ways_l1344_134415
