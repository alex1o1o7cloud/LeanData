import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_inequality_theorem_l3967_396783

theorem quadratic_function_inequality_theorem :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 → x = -1) ∧
    (∀ x : ℝ, x ≤ a * x^2 + b * x + c) ∧
    (∀ x : ℝ, a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧
    a = 1/4 ∧ b = 1/2 ∧ c = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_theorem_l3967_396783


namespace NUMINAMATH_CALUDE_equation_solutions_l3967_396746

def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ -x^2 = (5*x - 2)/(x - 2) - (x + 4)/(x + 2)

theorem equation_solutions :
  {x : ℝ | equation x} = {3, -1, -1 + Real.sqrt 5, -1 - Real.sqrt 5} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3967_396746


namespace NUMINAMATH_CALUDE_cylinder_height_problem_l3967_396773

/-- The height of cylinder B given the conditions of the problem -/
def height_cylinder_B : ℝ := 75

/-- The base radius of cylinder A in cm -/
def radius_A : ℝ := 10

/-- The height of cylinder A in cm -/
def height_A : ℝ := 8

/-- The base radius of cylinder B in cm -/
def radius_B : ℝ := 4

/-- The volume ratio of cylinder B to cylinder A -/
def volume_ratio : ℝ := 1.5

theorem cylinder_height_problem :
  volume_ratio * (Real.pi * radius_A^2 * height_A) = Real.pi * radius_B^2 * height_cylinder_B :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_problem_l3967_396773


namespace NUMINAMATH_CALUDE_cubic_root_theorem_l3967_396714

theorem cubic_root_theorem :
  ∃ (a b c : ℕ+) (x : ℝ),
    a = 1 ∧ b = 9 ∧ c = 1 ∧
    x = (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 1) / c ∧
    27 * x^3 - 9 * x^2 - 9 * x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_theorem_l3967_396714


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l3967_396711

/-- 
Given a forest with an initial number of trees, and a forester who plants trees over two days,
this theorem proves that the ratio of trees planted on the second day to the first day is 1/3,
given specific conditions about the planting process.
-/
theorem tree_planting_ratio 
  (initial_trees : ℕ) 
  (trees_after_monday : ℕ) 
  (total_planted : ℕ) 
  (h1 : initial_trees = 30)
  (h2 : trees_after_monday = initial_trees * 3)
  (h3 : total_planted = 80) :
  (total_planted - (trees_after_monday - initial_trees)) / (trees_after_monday - initial_trees) = 1 / 3 := by
  sorry

#check tree_planting_ratio

end NUMINAMATH_CALUDE_tree_planting_ratio_l3967_396711


namespace NUMINAMATH_CALUDE_chord_length_on_unit_circle_l3967_396726

/-- The length of the chord intercepted by the line x-y=0 on the circle x^2 + y^2 = 1 is equal to 2 -/
theorem chord_length_on_unit_circle : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x = y}
  let chord := circle ∩ line
  ∃ (a b : ℝ × ℝ), a ∈ chord ∧ b ∈ chord ∧ a ≠ b ∧ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_on_unit_circle_l3967_396726


namespace NUMINAMATH_CALUDE_log_expression_equality_l3967_396721

theorem log_expression_equality : Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 - (Real.sqrt 3 + 1) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l3967_396721


namespace NUMINAMATH_CALUDE_regression_unit_increase_l3967_396752

/-- Represents a simple linear regression model -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The predicted value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

/-- Theorem: In the given linear regression model, when x increases by 1, y increases by 3 -/
theorem regression_unit_increase (model : LinearRegression) (x : ℝ) 
    (h : model = { intercept := 2, slope := 3 }) :
    predict model (x + 1) - predict model x = 3 := by
  sorry

end NUMINAMATH_CALUDE_regression_unit_increase_l3967_396752


namespace NUMINAMATH_CALUDE_gcd_18_30_is_6_and_even_l3967_396792

theorem gcd_18_30_is_6_and_even : 
  Nat.gcd 18 30 = 6 ∧ Even 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_is_6_and_even_l3967_396792


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3967_396790

def y (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties :
  ∀ (a b : ℝ),
  (∀ x : ℝ, y a b x < 0 ↔ 1 < x ∧ x < 3) →
  a > 0 →
  b = -2 * a →
  (a = 1 ∧ b = -2) ∧
  (∀ x : ℝ,
    y a b x ≤ -1 ↔
      ((0 < a ∧ a < 1 → 2 ≤ x ∧ x ≤ 2/a) ∧
       (a = 1 → x = 2) ∧
       (a > 1 → 2/a ≤ x ∧ x ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3967_396790


namespace NUMINAMATH_CALUDE_actions_probability_is_one_four_hundredth_l3967_396700

/-- The probability of selecting specific letters from given words -/
def select_probability (total : ℕ) (choose : ℕ) (specific : ℕ) : ℚ :=
  (specific : ℚ) / (Nat.choose total choose : ℚ)

/-- The probability of selecting all letters from ACTIONS -/
def actions_probability : ℚ :=
  (select_probability 5 3 1) * (select_probability 5 2 1) * (select_probability 4 1 1)

/-- Theorem stating the probability of selecting all letters from ACTIONS -/
theorem actions_probability_is_one_four_hundredth :
  actions_probability = 1 / 400 := by sorry

end NUMINAMATH_CALUDE_actions_probability_is_one_four_hundredth_l3967_396700


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l3967_396733

/-- Represents the three bags of balls -/
inductive Bag
  | A
  | B
  | C

/-- The number of balls in each bag -/
def ballCount (bag : Bag) : Nat :=
  match bag with
  | Bag.A => 1
  | Bag.B => 2
  | Bag.C => 3

/-- The color of balls in each bag -/
def ballColor (bag : Bag) : String :=
  match bag with
  | Bag.A => "red"
  | Bag.B => "white"
  | Bag.C => "yellow"

/-- The number of ways to draw two balls of different colors -/
def differentColorDraws : Nat := sorry

/-- The number of ways to draw two balls of the same color -/
def sameColorDraws : Nat := sorry

theorem ball_drawing_theorem :
  differentColorDraws = 11 ∧ sameColorDraws = 4 := by sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l3967_396733


namespace NUMINAMATH_CALUDE_monotone_increasing_range_l3967_396781

/-- The function f(x) = lg(x^2 + ax - a - 1) is monotonically increasing in [2, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → 2 ≤ y → x ≤ y →
    Real.log (x^2 + a*x - a - 1) ≤ Real.log (y^2 + a*y - a - 1)

/-- The theorem stating the range of a for which f(x) is monotonically increasing -/
theorem monotone_increasing_range :
  {a : ℝ | is_monotone_increasing a} = Set.Ioi (-3) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_range_l3967_396781


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3967_396701

def a : ℝ × ℝ := (1, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def c : ℝ × ℝ := (1, 2)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 + (b x).1) * c.1 + (a.2 + (b x).2) * c.2 = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3967_396701


namespace NUMINAMATH_CALUDE_bird_migration_distance_l3967_396709

/-- Calculates the total distance traveled by migrating birds over two seasons -/
theorem bird_migration_distance (num_birds : ℕ) (dist_jim_disney : ℝ) (dist_disney_london : ℝ) :
  num_birds = 20 →
  dist_jim_disney = 50 →
  dist_disney_london = 60 →
  num_birds * (dist_jim_disney + dist_disney_london) = 2200 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_distance_l3967_396709


namespace NUMINAMATH_CALUDE_chimps_moved_correct_l3967_396742

/-- The number of chimpanzees being moved to a new cage -/
def chimps_moved (total : ℕ) (staying : ℕ) : ℕ := total - staying

/-- Theorem stating that the number of chimpanzees moved is correct -/
theorem chimps_moved_correct (total : ℕ) (staying : ℕ) 
  (h1 : total = 45) (h2 : staying = 27) : 
  chimps_moved total staying = 18 := by
  sorry

end NUMINAMATH_CALUDE_chimps_moved_correct_l3967_396742


namespace NUMINAMATH_CALUDE_books_sold_l3967_396734

theorem books_sold (initial : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 41 → added = 2 → final = 10 → initial - (initial - final + added) = 33 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_l3967_396734


namespace NUMINAMATH_CALUDE_decimal_13_equals_binary_1101_l3967_396764

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def binary_to_decimal (l : List Bool) : ℕ :=
  l.enum.foldl (λ sum (i, b) => sum + if b then 2^i else 0) 0

def decimal_13 : ℕ := 13

def binary_1101 : List Bool := [true, false, true, true]

theorem decimal_13_equals_binary_1101 : 
  binary_to_decimal binary_1101 = decimal_13 :=
sorry

end NUMINAMATH_CALUDE_decimal_13_equals_binary_1101_l3967_396764


namespace NUMINAMATH_CALUDE_min_value_fraction_l3967_396703

theorem min_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ (m : ℝ), m = (x + y) / x ∧ ∀ (z w : ℝ), (-6 ≤ z ∧ z ≤ -3) → (3 ≤ w ∧ w ≤ 6) → m ≤ (z + w) / z :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3967_396703


namespace NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisibility_l3967_396794

theorem consecutive_odd_power_sum_divisibility (p q m n : ℕ) : 
  Odd p → Odd q → p = q + 2 → Odd m → Odd n → m > 0 → n > 0 → 
  ∃ k : ℤ, p^m + q^n = k * (p + q) :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_power_sum_divisibility_l3967_396794


namespace NUMINAMATH_CALUDE_tammy_mountain_climb_l3967_396750

/-- Tammy's mountain climbing problem -/
theorem tammy_mountain_climb 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_diff : ℝ) 
  (time_diff : ℝ) 
  (h_total_time : total_time = 14) 
  (h_total_distance : total_distance = 52) 
  (h_speed_diff : speed_diff = 0.5) 
  (h_time_diff : time_diff = 2) :
  ∃ (v : ℝ), 
    v > 0 ∧ 
    v + speed_diff > 0 ∧ 
    (∃ (t : ℝ), 
      t > 0 ∧ 
      t - time_diff > 0 ∧ 
      t + (t - time_diff) = total_time ∧ 
      v * t + (v + speed_diff) * (t - time_diff) = total_distance) ∧ 
    v + speed_diff = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_mountain_climb_l3967_396750


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3967_396766

/-- Given a school with the following student composition:
    - Total number of boys: 650
    - 44% are Muslims
    - 28% are Hindus
    - 117 boys are from other communities
    This theorem proves that 10% of the boys are Sikhs. -/
theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 650 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 117 →
  (total - (muslim_percent * total + hindu_percent * total + other)) / total = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3967_396766


namespace NUMINAMATH_CALUDE_exists_negative_value_implies_a_greater_than_nine_halves_l3967_396748

theorem exists_negative_value_implies_a_greater_than_nine_halves
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = x^3 - a*x^2 + 10)
  (a : ℝ)
  (h_exists : ∃ x ∈ Set.Icc 1 2, f x < 0) :
  a > 9/2 := by
sorry

end NUMINAMATH_CALUDE_exists_negative_value_implies_a_greater_than_nine_halves_l3967_396748


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3967_396756

theorem opposite_of_negative_three : -((-3 : ℤ)) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3967_396756


namespace NUMINAMATH_CALUDE_half_square_area_l3967_396722

theorem half_square_area (square_area : Real) (h1 : square_area = 100) :
  square_area / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_half_square_area_l3967_396722


namespace NUMINAMATH_CALUDE_missing_number_is_eight_l3967_396710

/-- Given the equation |9 - x(3 - 12)| - |5 - 11| = 75, prove that x = 8 is the solution. -/
theorem missing_number_is_eight : ∃ x : ℝ, 
  (|9 - x * (3 - 12)| - |5 - 11| = 75) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_eight_l3967_396710


namespace NUMINAMATH_CALUDE_circle_not_proportional_line_directly_proportional_hyperbola_inversely_proportional_line_through_origin_directly_proportional_another_line_through_origin_directly_proportional_l3967_396717

/-- A relation between x and y is directly proportional if it can be expressed as y = kx for some constant k ≠ 0 -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- A relation between x and y is inversely proportional if it can be expressed as xy = k for some constant k ≠ 0 -/
def InverselyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, x ≠ 0 → f x * x = k

/-- The main theorem stating that x^2 + y^2 = 16 is neither directly nor inversely proportional -/
theorem circle_not_proportional :
  ¬ (DirectlyProportional (fun x => Real.sqrt (16 - x^2)) ∨
     InverselyProportional (fun x => Real.sqrt (16 - x^2))) :=
sorry

/-- 2x + 3y = 6 describes y as directly proportional to x -/
theorem line_directly_proportional :
  DirectlyProportional (fun x => (6 - 2*x) / 3) ∨
  InverselyProportional (fun x => (6 - 2*x) / 3) :=
sorry

/-- xy = 5 describes y as inversely proportional to x -/
theorem hyperbola_inversely_proportional :
  DirectlyProportional (fun x => 5 / x) ∨
  InverselyProportional (fun x => 5 / x) :=
sorry

/-- x = 7y describes y as directly proportional to x -/
theorem line_through_origin_directly_proportional :
  DirectlyProportional (fun x => x / 7) ∨
  InverselyProportional (fun x => x / 7) :=
sorry

/-- x/y = 2 describes y as directly proportional to x -/
theorem another_line_through_origin_directly_proportional :
  DirectlyProportional (fun x => x / 2) ∨
  InverselyProportional (fun x => x / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_not_proportional_line_directly_proportional_hyperbola_inversely_proportional_line_through_origin_directly_proportional_another_line_through_origin_directly_proportional_l3967_396717


namespace NUMINAMATH_CALUDE_sphere_sum_l3967_396723

theorem sphere_sum (x y z : ℝ) : 
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_sum_l3967_396723


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_quadratic_inequality_l3967_396762

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ ∃ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∀ x > 0, x^2 + 3*x - 2 > 0) ↔ (∃ x > 0, x^2 + 3*x - 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_quadratic_inequality_l3967_396762


namespace NUMINAMATH_CALUDE_unfoldable_cone_ratio_l3967_396777

/-- A cone with lateral surface that forms a semicircle when unfolded -/
structure UnfoldableCone where
  /-- Radius of the base of the cone -/
  base_radius : ℝ
  /-- Length of the generatrix of the cone -/
  generatrix_length : ℝ
  /-- The lateral surface forms a semicircle when unfolded -/
  unfolded_is_semicircle : π * generatrix_length = 2 * π * base_radius

/-- 
If the lateral surface of a cone forms a semicircle when unfolded, 
then the ratio of the length of the cone's generatrix to the radius of its base is 2:1
-/
theorem unfoldable_cone_ratio (cone : UnfoldableCone) : 
  cone.generatrix_length / cone.base_radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_unfoldable_cone_ratio_l3967_396777


namespace NUMINAMATH_CALUDE_f_properties_l3967_396774

/-- The function f(x) defined as (2^x - a) / (2^x + a) where a > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x - a) / (2^x + a)

/-- Theorem stating properties of the function f -/
theorem f_properties (a : ℝ) (h : a > 0) 
  (h_odd : ∀ x, f a (-x) = -(f a x)) :
  (a = 1) ∧ 
  (∀ x y, x < y → f a x < f a y) ∧
  (∀ x, x ≤ 1 → f a x ≤ 1/3) ∧
  (f a 1 = 1/3) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l3967_396774


namespace NUMINAMATH_CALUDE_largest_even_integer_l3967_396743

theorem largest_even_integer (n : ℕ) : 
  (2 * (List.range 20).sum) = (4 * n - 12) → n = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_even_integer_l3967_396743


namespace NUMINAMATH_CALUDE_min_binomial_ratio_five_seven_l3967_396782

theorem min_binomial_ratio_five_seven (n : ℕ) : n > 0 → (
  (∃ r : ℕ, r < n ∧ (n.choose r : ℚ) / (n.choose (r + 1)) = 5 / 7) ↔ n ≥ 11
) := by sorry

end NUMINAMATH_CALUDE_min_binomial_ratio_five_seven_l3967_396782


namespace NUMINAMATH_CALUDE_distance_relation_l3967_396775

/-- Given four points on a directed line satisfying a certain condition, 
    prove a relationship between their distances. -/
theorem distance_relation (A B C D : ℝ) 
    (h : (C - A) / (B - C) + (D - A) / (B - D) = 0) : 
    1 / (C - A) + 1 / (D - A) = 2 / (B - A) := by
  sorry

end NUMINAMATH_CALUDE_distance_relation_l3967_396775


namespace NUMINAMATH_CALUDE_carrot_sticks_before_dinner_l3967_396780

theorem carrot_sticks_before_dinner 
  (before : ℕ) 
  (after : ℕ) 
  (total : ℕ) 
  (h1 : after = 15) 
  (h2 : total = 37) 
  (h3 : before + after = total) : 
  before = 22 := by
sorry

end NUMINAMATH_CALUDE_carrot_sticks_before_dinner_l3967_396780


namespace NUMINAMATH_CALUDE_cubic_root_function_l3967_396708

/-- Given a function y = kx^(1/3) where y = 3√2 when x = 64, prove that y = 3 when x = 8 -/
theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 3 * Real.sqrt 2) →
  k * 8^(1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_function_l3967_396708


namespace NUMINAMATH_CALUDE_ellipse_ratio_l3967_396720

theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a / b = b / c) : a^2 / b^2 = 2 / (-1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_l3967_396720


namespace NUMINAMATH_CALUDE_rect_to_spherical_l3967_396730

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical (x y z : ℝ) :
  x = 1 ∧ y = Real.sqrt 3 ∧ z = 2 →
  ∃ (ρ θ φ : ℝ),
    ρ > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    ρ = 3 ∧
    θ = Real.pi / 3 ∧
    φ = Real.arccos (2/3) ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_spherical_l3967_396730


namespace NUMINAMATH_CALUDE_min_value_z_l3967_396758

theorem min_value_z (x y : ℝ) : 
  x^2 + 2*y^2 + y^3 + 6*x - 4*y + 30 ≥ 20 ∧ 
  ∃ x y : ℝ, x^2 + 2*y^2 + y^3 + 6*x - 4*y + 30 = 20 :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l3967_396758


namespace NUMINAMATH_CALUDE_unique_solution_E_l3967_396784

/-- Definition of the function E --/
def E (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that -1/16 is the unique solution to E(a, 3, 2) = E(a, 5, 3) --/
theorem unique_solution_E :
  ∃! a : ℝ, E a 3 2 = E a 5 3 ∧ a = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_E_l3967_396784


namespace NUMINAMATH_CALUDE_series_sum_l3967_396713

theorem series_sum : 1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + 10 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3967_396713


namespace NUMINAMATH_CALUDE_divisibility_problem_l3967_396731

theorem divisibility_problem (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  15 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3967_396731


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3967_396771

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3967_396771


namespace NUMINAMATH_CALUDE_woodworker_job_days_l3967_396712

/-- Represents the woodworker's job details -/
structure WoodworkerJob where
  normal_days : ℕ            -- Normal number of days to complete the job
  normal_parts : ℕ           -- Normal number of parts produced
  productivity_increase : ℕ  -- Increase in parts produced per day
  extra_parts : ℕ            -- Extra parts produced with increased productivity

/-- Calculates the number of days required to finish the job with increased productivity -/
def days_with_increased_productivity (job : WoodworkerJob) : ℕ :=
  let normal_rate := job.normal_parts / job.normal_days
  let new_rate := normal_rate + job.productivity_increase
  let total_parts := job.normal_parts + job.extra_parts
  total_parts / new_rate

/-- Theorem stating that for the given conditions, the job takes 22 days with increased productivity -/
theorem woodworker_job_days (job : WoodworkerJob)
  (h1 : job.normal_days = 24)
  (h2 : job.normal_parts = 360)
  (h3 : job.productivity_increase = 5)
  (h4 : job.extra_parts = 80) :
  days_with_increased_productivity job = 22 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_job_days_l3967_396712


namespace NUMINAMATH_CALUDE_yard_length_24_trees_18m_spacing_l3967_396786

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_spacing : ℝ) : ℝ :=
  (num_trees - 1) * tree_spacing

/-- Theorem: The length of a yard with 24 trees planted at equal distances,
    with one tree at each end and 18 meters between consecutive trees, is 414 meters. -/
theorem yard_length_24_trees_18m_spacing :
  yard_length 24 18 = 414 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_24_trees_18m_spacing_l3967_396786


namespace NUMINAMATH_CALUDE_triangle_problem_l3967_396799

theorem triangle_problem (DC CB : ℝ) (h1 : DC = 12) (h2 : CB = 9)
  (AD : ℝ) (h3 : AD > 0)
  (AB : ℝ) (h4 : AB = (1/3) * AD)
  (ED : ℝ) (h5 : ED = (3/4) * AD) :
  ∃ FC : ℝ, FC = 14.625 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3967_396799


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l3967_396716

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if the points (5, 10), (-3, k), and (-11, 6) are collinear, then k = 8 -/
theorem collinear_points_k_value :
  collinear 5 10 (-3) k (-11) 6 → k = 8 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l3967_396716


namespace NUMINAMATH_CALUDE_divisibility_and_finiteness_l3967_396747

theorem divisibility_and_finiteness :
  (∀ x : ℕ+, ∃ y : ℕ+, (x + y + 1) ∣ (x^3 + y^3 + 1)) ∧
  (∀ x : ℕ+, Set.Finite {y : ℕ+ | (x + y + 1) ∣ (x^3 + y^3 + 1)}) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_and_finiteness_l3967_396747


namespace NUMINAMATH_CALUDE_candy_distribution_l3967_396729

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) : 
  total_candy = 30 → num_friends = 4 → 
  ∃ (removed : ℕ) (equal_share : ℕ), 
    removed ≤ 2 ∧ 
    (total_candy - removed) % num_friends = 0 ∧ 
    (total_candy - removed) / num_friends = equal_share ∧
    ∀ (r : ℕ), r < removed → (total_candy - r) % num_friends ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3967_396729


namespace NUMINAMATH_CALUDE_books_per_shelf_l3967_396719

theorem books_per_shelf 
  (total_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : total_shelves = 150) 
  (h2 : total_books = 2250) : 
  total_books / total_shelves = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l3967_396719


namespace NUMINAMATH_CALUDE_parallel_necessary_not_sufficient_l3967_396755

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem parallel_necessary_not_sufficient :
  (∀ a b : V, a = b → parallel a b) ∧
  (∃ a b : V, parallel a b ∧ a ≠ b) := by sorry

end NUMINAMATH_CALUDE_parallel_necessary_not_sufficient_l3967_396755


namespace NUMINAMATH_CALUDE_dried_grapes_weight_l3967_396728

/-- Calculates the weight of dried grapes from fresh grapes -/
theorem dried_grapes_weight
  (fresh_weight : ℝ)
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (h1 : fresh_weight = 40)
  (h2 : fresh_water_content = 0.9)
  (h3 : dried_water_content = 0.2) :
  (fresh_weight * (1 - fresh_water_content)) / (1 - dried_water_content) = 5 := by
  sorry

end NUMINAMATH_CALUDE_dried_grapes_weight_l3967_396728


namespace NUMINAMATH_CALUDE_expected_socks_removed_theorem_l3967_396779

/-- The expected number of socks removed to get both favorite socks -/
def expected_socks_removed (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

/-- Theorem stating the expected number of socks removed to get both favorite socks -/
theorem expected_socks_removed_theorem (n : ℕ) (h : n ≥ 2) :
  expected_socks_removed n = 2 * (n + 1) / 3 := by
  sorry

#check expected_socks_removed_theorem

end NUMINAMATH_CALUDE_expected_socks_removed_theorem_l3967_396779


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3967_396793

theorem min_distance_to_line (x y : ℝ) (h : 2 * x - y - 5 = 0) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 5 ∧
  ∀ (x' y' : ℝ), 2 * x' - y' - 5 = 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3967_396793


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3967_396749

theorem complex_modulus_problem (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3967_396749


namespace NUMINAMATH_CALUDE_min_subset_size_l3967_396744

def is_valid_subset (s : Finset ℕ) : Prop :=
  s ⊆ Finset.range 11 ∧
  ∀ n : ℕ, n ∈ Finset.range 21 →
    (n ∈ s ∨ ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = n)

theorem min_subset_size :
  ∃ (s : Finset ℕ), is_valid_subset s ∧ s.card = 6 ∧
  ∀ (t : Finset ℕ), is_valid_subset t → t.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_subset_size_l3967_396744


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3967_396705

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person D gets the red card"
def event_D_red (d : Distribution) : Prop := d Person.D = Card.Red

-- Statement: The events are mutually exclusive but not complementary
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_red d ∧ event_D_red d)) ∧
  (∃ d : Distribution, ¬event_A_red d ∧ ¬event_D_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3967_396705


namespace NUMINAMATH_CALUDE_work_left_after_collaboration_l3967_396795

/-- Represents the fraction of work completed in one day -/
def work_rate (days : ℕ) : ℚ := 1 / days

/-- Represents the total work completed by two people in a given number of days -/
def total_work (rate_a rate_b : ℚ) (days : ℕ) : ℚ := (rate_a + rate_b) * days

theorem work_left_after_collaboration (days_a days_b collab_days : ℕ) 
  (h1 : days_a = 15) (h2 : days_b = 20) (h3 : collab_days = 4) : 
  1 - total_work (work_rate days_a) (work_rate days_b) collab_days = 8 / 15 := by
  sorry

#check work_left_after_collaboration

end NUMINAMATH_CALUDE_work_left_after_collaboration_l3967_396795


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3967_396763

theorem student_average_greater_than_true_average 
  (x y w z : ℝ) (h : x < y ∧ y < w ∧ w < z) : 
  (x + y + 2*w + 2*z) / 6 > (x + y + w + z) / 4 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3967_396763


namespace NUMINAMATH_CALUDE_base7_addition_multiplication_l3967_396796

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def decimalToBase7 (n : Nat) : List Nat :=
  if n < 7 then [n]
  else (n % 7) :: decimalToBase7 (n / 7)

/-- Adds two base 7 numbers -/
def addBase7 (a b : List Nat) : List Nat :=
  decimalToBase7 (base7ToDecimal a + base7ToDecimal b)

/-- Multiplies a base 7 number by another base 7 number -/
def mulBase7 (a b : List Nat) : List Nat :=
  decimalToBase7 (base7ToDecimal a * base7ToDecimal b)

theorem base7_addition_multiplication :
  mulBase7 (addBase7 [5, 2] [4, 3, 3]) [2] = [4, 6, 6] := by sorry

end NUMINAMATH_CALUDE_base7_addition_multiplication_l3967_396796


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3967_396769

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem max_rectangle_area (length width : ℕ) :
  length + width = 25 →
  isPrime length →
  isComposite width →
  ∀ l w : ℕ, l + w = 25 → isPrime l → isComposite w → 
    rectangle_area length width ≥ rectangle_area l w →
  rectangle_area length width = 156 :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3967_396769


namespace NUMINAMATH_CALUDE_diamond_calculation_l3967_396751

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29 / 132 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3967_396751


namespace NUMINAMATH_CALUDE_volleyball_ticket_sales_l3967_396776

theorem volleyball_ticket_sales (total_tickets : ℕ) (tickets_left : ℕ) : 
  total_tickets = 100 →
  tickets_left = 40 →
  ∃ (jude_tickets : ℕ),
    (jude_tickets : ℚ) + 2 * (jude_tickets : ℚ) + ((1/2 : ℚ) * (jude_tickets : ℚ) + 4) = (total_tickets - tickets_left : ℚ) ∧
    jude_tickets = 16 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_ticket_sales_l3967_396776


namespace NUMINAMATH_CALUDE_town_distance_approx_l3967_396757

/-- Represents the map scale as a fraction of inches per mile -/
def map_scale : ℚ := 7 / (15 * 19)

/-- Represents the distance between two points on the map in inches -/
def map_distance : ℚ := 37 / 8

/-- Calculates the actual distance in miles given the map scale and map distance -/
def actual_distance (scale : ℚ) (distance : ℚ) : ℚ := distance / scale

/-- Theorem stating that the actual distance between the towns is approximately 41.0083 miles -/
theorem town_distance_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10000 ∧ 
  |actual_distance map_scale map_distance - 41.0083| < ε :=
sorry

end NUMINAMATH_CALUDE_town_distance_approx_l3967_396757


namespace NUMINAMATH_CALUDE_ship_supplies_problem_l3967_396736

theorem ship_supplies_problem (initial_supply : ℝ) 
  (remaining_supply : ℝ) (h1 : initial_supply = 400) 
  (h2 : remaining_supply = 96) : 
  ∃ x : ℝ, x = 2/5 ∧ 
    remaining_supply = (2/5) * (1 - x) * initial_supply :=
by sorry

end NUMINAMATH_CALUDE_ship_supplies_problem_l3967_396736


namespace NUMINAMATH_CALUDE_product_in_A_l3967_396759

-- Define the set A
def A : Set ℤ := {x | ∃ a b : ℤ, x = a^2 + b^2}

-- State the theorem
theorem product_in_A (x₁ x₂ : ℤ) (h₁ : x₁ ∈ A) (h₂ : x₂ ∈ A) : 
  x₁ * x₂ ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_in_A_l3967_396759


namespace NUMINAMATH_CALUDE_not_monotone_condition_l3967_396715

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of being not monotone on an interval
def not_monotone_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ (f x < f y ∧ ∃ z, x < z ∧ z < y ∧ f z < f x) ∨
                                 (f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)

-- State the theorem
theorem not_monotone_condition (k : ℝ) :
  not_monotone_on f k (k + 2) ↔ (-4 < k ∧ k < -2) ∨ (0 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_not_monotone_condition_l3967_396715


namespace NUMINAMATH_CALUDE_factor_expression_l3967_396791

theorem factor_expression (x : ℝ) : 3*x*(x+3) + 2*(x+3) = (x+3)*(3*x+2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3967_396791


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l3967_396745

theorem ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : num_seats = 14) (h2 : people_per_seat = 6) : 
  num_seats * people_per_seat = 84 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l3967_396745


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3967_396702

/-- The probability of A hitting the target -/
def prob_A_hit : ℚ := 2/3

/-- The probability of B hitting the target -/
def prob_B_hit : ℚ := 3/4

/-- The probability that A shoots 3 times and misses at least once -/
def prob_A_miss_at_least_once : ℚ := 19/27

/-- The probability that A shoots twice and hits both times, while B shoots twice and hits exactly once -/
def prob_A_hit_twice_B_hit_once : ℚ := 1/6

theorem shooting_probabilities 
  (hA : prob_A_hit = 2/3)
  (hB : prob_B_hit = 3/4)
  (indep : ∀ (n : ℕ) (m : ℕ), (prob_A_hit ^ n) * ((1 - prob_A_hit) ^ (m - n)) = 
    (2/3 ^ n) * ((1/3) ^ (m - n))) :
  prob_A_miss_at_least_once = 19/27 ∧ 
  prob_A_hit_twice_B_hit_once = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l3967_396702


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3967_396735

theorem ceiling_sum_sqrt : ⌈Real.sqrt 8⌉ + ⌈Real.sqrt 48⌉ + ⌈Real.sqrt 288⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3967_396735


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3967_396787

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos α = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3967_396787


namespace NUMINAMATH_CALUDE_horner_rule_v4_l3967_396770

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_rule_v4 :
  horner_v4 2 = 80 :=
by sorry

#eval horner_v4 2
#eval horner_polynomial 2

end NUMINAMATH_CALUDE_horner_rule_v4_l3967_396770


namespace NUMINAMATH_CALUDE_invisible_square_exists_l3967_396724

/-- A lattice point is invisible if the segment from the origin to that point contains another lattice point. -/
def invisible (x y : ℤ) : Prop :=
  ∃ k : ℤ, 1 < k ∧ k < max x.natAbs y.natAbs ∧ (k ∣ x) ∧ (k ∣ y)

/-- For any positive integer L, there exists a square with side length L where all lattice points are invisible. -/
theorem invisible_square_exists (L : ℕ) (hL : 0 < L) :
  ∃ x y : ℤ, ∀ i j : ℕ, i ≤ L → j ≤ L → invisible (x + i) (y + j) := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l3967_396724


namespace NUMINAMATH_CALUDE_hyperbola_range_l3967_396725

/-- The equation represents a hyperbola with parameter m -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m - 2) = 1

/-- The range of m for which the equation represents a hyperbola -/
theorem hyperbola_range :
  ∀ m : ℝ, is_hyperbola m ↔ m > -2 ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_range_l3967_396725


namespace NUMINAMATH_CALUDE_xiaotong_pe_score_l3967_396739

/-- Calculates the physical education score based on extracurricular activities and final exam scores -/
def physical_education_score (extracurricular_score : ℝ) (final_exam_score : ℝ) : ℝ :=
  0.3 * extracurricular_score + 0.7 * final_exam_score

/-- Xiaotong's physical education score theorem -/
theorem xiaotong_pe_score :
  let max_score : ℝ := 100
  let extracurricular_weight : ℝ := 0.3
  let final_exam_weight : ℝ := 0.7
  let xiaotong_extracurricular_score : ℝ := 90
  let xiaotong_final_exam_score : ℝ := 80
  physical_education_score xiaotong_extracurricular_score xiaotong_final_exam_score = 83 :=
by
  sorry

#eval physical_education_score 90 80

end NUMINAMATH_CALUDE_xiaotong_pe_score_l3967_396739


namespace NUMINAMATH_CALUDE_expansion_property_p_value_l3967_396704

/-- The value of p in the expansion of (x+y)^10 -/
def p : ℚ :=
  8/11

/-- The value of q in the expansion of (x+y)^10 -/
def q : ℚ :=
  3/11

/-- The third term in the expansion of (x+y)^10 -/
def third_term (x y : ℚ) : ℚ :=
  45 * x^8 * y^2

/-- The fourth term in the expansion of (x+y)^10 -/
def fourth_term (x y : ℚ) : ℚ :=
  120 * x^7 * y^3

theorem expansion_property : 
  p + q = 1 ∧ third_term p q = fourth_term p q :=
sorry

theorem p_value : p = 8/11 :=
sorry

end NUMINAMATH_CALUDE_expansion_property_p_value_l3967_396704


namespace NUMINAMATH_CALUDE_min_abs_z_l3967_396718

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*I) + Complex.abs (z - 2) = 10) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 10 / Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l3967_396718


namespace NUMINAMATH_CALUDE_f_unique_zero_l3967_396788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a + 1) * x + a * Real.log x

theorem f_unique_zero (a : ℝ) (h : a > 0) : 
  ∃! x : ℝ, x > 0 ∧ f a x = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_unique_zero_l3967_396788


namespace NUMINAMATH_CALUDE_frame_price_increase_l3967_396789

theorem frame_price_increase (budget : ℝ) (remaining : ℝ) (ratio : ℝ) : 
  budget = 60 → 
  remaining = 6 → 
  ratio = 3/4 → 
  let smaller_frame_price := budget - remaining
  let initial_frame_price := smaller_frame_price / ratio
  (initial_frame_price - budget) / budget * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_frame_price_increase_l3967_396789


namespace NUMINAMATH_CALUDE_odd_polyhedron_sum_not_nine_l3967_396732

/-- Represents a convex polyhedron with odd-sided faces and odd-valence vertices -/
structure OddPolyhedron where
  -- Number of edges
  e : ℕ
  -- Number of faces with i sides (i is odd)
  ℓ : ℕ → ℕ
  -- Number of vertices where i edges meet (i is odd)
  c : ℕ → ℕ
  -- Each face has an odd number of sides
  face_odd : ∀ i, ℓ i > 0 → Odd i
  -- Each vertex has an odd number of edges meeting at it
  vertex_odd : ∀ i, c i > 0 → Odd i
  -- Edge-face relation
  edge_face : 2 * e = ∑' i, i * ℓ i
  -- Edge-vertex relation
  edge_vertex : 2 * e = ∑' i, i * c i
  -- Euler's formula
  euler : e + 2 = (∑' i, ℓ i) + (∑' i, c i)

/-- The sum of triangular faces and vertices where three edges meet cannot be 9 -/
theorem odd_polyhedron_sum_not_nine (P : OddPolyhedron) : ¬(P.ℓ 3 + P.c 3 = 9) := by
  sorry

end NUMINAMATH_CALUDE_odd_polyhedron_sum_not_nine_l3967_396732


namespace NUMINAMATH_CALUDE_integral_over_pyramidal_region_l3967_396785

/-- The pyramidal region V -/
def V : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i ≥ 0 ∧ v 0 + v 1 + v 2 ≤ 1}

/-- The integrand function -/
def f (v : Fin 3 → ℝ) : ℝ :=
  v 0 * v 1^9 * v 2^8 * (1 - v 0 - v 1 - v 2)^4

/-- The theorem statement -/
theorem integral_over_pyramidal_region :
  ∫ v in V, f v = (Nat.factorial 9 * Nat.factorial 8 * Nat.factorial 4) / Nat.factorial 25 := by
  sorry

end NUMINAMATH_CALUDE_integral_over_pyramidal_region_l3967_396785


namespace NUMINAMATH_CALUDE_exists_zero_sequence_l3967_396738

-- Define the operations
def add_one (x : ℚ) : ℚ := x + 1
def neg_reciprocal (x : ℚ) : ℚ := -1 / x

-- Define a sequence of operations
inductive Operation
| AddOne
| NegReciprocal

def apply_operation (op : Operation) (x : ℚ) : ℚ :=
  match op with
  | Operation.AddOne => add_one x
  | Operation.NegReciprocal => neg_reciprocal x

-- Theorem statement
theorem exists_zero_sequence : ∃ (seq : List Operation), 
  let final_value := seq.foldl (λ acc op => apply_operation op acc) 0
  final_value = 0 ∧ seq.length > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_zero_sequence_l3967_396738


namespace NUMINAMATH_CALUDE_max_square_plots_l3967_396740

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFencing : ℕ := 1994

/-- Represents the field dimensions -/
def field : FieldDimensions := { width := 24, length := 52 }

/-- Calculates the number of square plots given the number of plots in a column -/
def numPlots (n : ℕ) : ℕ :=
  (13 * n * n) / 6

/-- Calculates the length of internal fencing needed for n plots in a column -/
def fencingNeeded (n : ℕ) : ℕ :=
  104 * n - 76

/-- Theorem stating the maximum number of square test plots -/
theorem max_square_plots :
  ∃ (n : ℕ), n ≤ 18 ∧ 6 ∣ n ∧
  fencingNeeded n ≤ availableFencing ∧
  (∀ (m : ℕ), m > n → fencingNeeded m > availableFencing ∨ ¬(6 ∣ m)) ∧
  numPlots n = 702 :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l3967_396740


namespace NUMINAMATH_CALUDE_translation_proof_l3967_396741

def original_function (x : ℝ) : ℝ := 4 * x + 3

def translated_function (x : ℝ) : ℝ := 4 * x + 16

def translation_vector : ℝ × ℝ := (-3, 1)

theorem translation_proof :
  ∀ x y : ℝ, 
    y = original_function x → 
    y + (translation_vector.2) = translated_function (x + translation_vector.1) :=
by sorry

end NUMINAMATH_CALUDE_translation_proof_l3967_396741


namespace NUMINAMATH_CALUDE_valid_assignment_y_equals_x_plus_1_l3967_396727

/-- Represents a variable name in a programming language --/
def Variable : Type := String

/-- Represents an expression in a programming language --/
inductive Expression
| Var : Variable → Expression
| Num : Int → Expression
| Add : Expression → Expression → Expression

/-- Represents an assignment statement in a programming language --/
structure Assignment :=
  (lhs : Variable)
  (rhs : Expression)

/-- Checks if an assignment statement is valid --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∃ (x : Variable), a.rhs = Expression.Add (Expression.Var x) (Expression.Num 1)

/-- The statement "y = x + 1" is a valid assignment --/
theorem valid_assignment_y_equals_x_plus_1 :
  is_valid_assignment { lhs := "y", rhs := Expression.Add (Expression.Var "x") (Expression.Num 1) } :=
by sorry

end NUMINAMATH_CALUDE_valid_assignment_y_equals_x_plus_1_l3967_396727


namespace NUMINAMATH_CALUDE_min_value_of_a_l3967_396765

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem min_value_of_a (a : ℝ) (h : A ∩ B a = ∅) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3967_396765


namespace NUMINAMATH_CALUDE_largest_divisible_by_9_after_erasure_l3967_396754

def original_number : ℕ := 321321321321

def erase_digits (n : ℕ) (positions : List ℕ) : ℕ :=
  sorry

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem largest_divisible_by_9_after_erasure :
  ∃ (positions : List ℕ),
    let result := erase_digits original_number positions
    is_divisible_by_9 result ∧
    ∀ (other_positions : List ℕ),
      let other_result := erase_digits original_number other_positions
      is_divisible_by_9 other_result →
      other_result ≤ result ∧
      result = 32132132121 :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_9_after_erasure_l3967_396754


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3967_396753

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 35*x^3) 
  (h3 : a^2 - b^2 = 4*x^2) : 
  a = 2*x ∨ a = -2*x := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3967_396753


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3967_396767

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^a * (3 : ℝ)^b = 3 → 
  ∀ x y : ℝ, x > 0 → y > 0 → (3 : ℝ)^x * (3 : ℝ)^y = 3 → 
  1/a + 1/b ≤ 1/x + 1/y := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3967_396767


namespace NUMINAMATH_CALUDE_triangle_height_l3967_396737

theorem triangle_height (base area height : ℝ) : 
  base = 4 ∧ area = 16 ∧ area = (base * height) / 2 → height = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l3967_396737


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3967_396706

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 1  -- diameter of smaller circle
  let d₂ : ℝ := 3  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let area_small : ℝ := π * r₁^2  -- area of smaller circle
  let area_large : ℝ := π * r₂^2  -- area of larger circle
  let area_between : ℝ := area_large - area_small  -- area between circles
  (area_between / area_small) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l3967_396706


namespace NUMINAMATH_CALUDE_composite_expression_l3967_396772

/-- A positive integer is composite if it can be expressed as a product of two integers
    greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- Every composite positive integer can be expressed as xy+xz+yz+1,
    where x, y, and z are positive integers. -/
theorem composite_expression (n : ℕ) (h : IsComposite n) :
    ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l3967_396772


namespace NUMINAMATH_CALUDE_no_solution_exists_l3967_396707

theorem no_solution_exists : ¬∃ (a b c : ℕ+), 
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3967_396707


namespace NUMINAMATH_CALUDE_lakers_win_in_seven_games_l3967_396797

/-- The probability of the Knicks winning a single game -/
def p_knicks_win : ℚ := 3/4

/-- The probability of the Lakers winning a single game -/
def p_lakers_win : ℚ := 1 - p_knicks_win

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 7

/-- The number of ways to choose 3 wins from 6 games -/
def ways_to_choose_3_from_6 : ℕ := 20

theorem lakers_win_in_seven_games :
  let p_lakers_win_series := (ways_to_choose_3_from_6 : ℚ) * p_lakers_win^3 * p_knicks_win^3 * p_lakers_win
  p_lakers_win_series = 540/16384 := by sorry

end NUMINAMATH_CALUDE_lakers_win_in_seven_games_l3967_396797


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l3967_396760

/-- Represents a participant's score in a day of the competition -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores over the three-day competition -/
structure CompetitionScore where
  day1 : DayScore
  day2 : DayScore
  day3 : DayScore

/-- Alpha's competition score -/
def alpha_score : CompetitionScore := {
  day1 := { scored := 200, attempted := 300 }
  day2 := { scored := 150, attempted := 200 }
  day3 := { scored := 100, attempted := 100 }
}

/-- Calculates the success ratio for a DayScore -/
def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

/-- Calculates the total success ratio for a CompetitionScore -/
def total_success_ratio (score : CompetitionScore) : ℚ :=
  (score.day1.scored + score.day2.scored + score.day3.scored) /
  (score.day1.attempted + score.day2.attempted + score.day3.attempted)

theorem beta_max_success_ratio :
  ∀ beta_score : CompetitionScore,
    (beta_score.day1.attempted + beta_score.day2.attempted + beta_score.day3.attempted = 600) →
    (beta_score.day1.attempted ≠ 300) →
    (beta_score.day2.attempted ≠ 200) →
    (beta_score.day1.scored > 0) →
    (beta_score.day2.scored > 0) →
    (beta_score.day3.scored > 0) →
    (success_ratio beta_score.day1 < success_ratio alpha_score.day1) →
    (success_ratio beta_score.day2 < success_ratio alpha_score.day2) →
    (success_ratio beta_score.day3 < success_ratio alpha_score.day3) →
    total_success_ratio beta_score ≤ 358 / 600 :=
by sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l3967_396760


namespace NUMINAMATH_CALUDE_minimum_width_for_garden_l3967_396768

theorem minimum_width_for_garden (w : ℝ) : w > 0 → w * (w + 10) ≥ 150 → 
  ∀ x > 0, x * (x + 10) ≥ 150 → 2 * (w + w + 10) ≤ 2 * (x + x + 10) → w = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_width_for_garden_l3967_396768


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3967_396798

/-- The line y = x + 1 intersects the circle x² + y² = 1 at two distinct points, 
    and neither of these points is the center of the circle (0, 0). -/
theorem line_circle_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (y₁ = x₁ + 1) ∧ (x₁^2 + y₁^2 = 1) ∧
    (y₂ = x₂ + 1) ∧ (x₂^2 + y₂^2 = 1) ∧
    (x₁ ≠ x₂) ∧ (y₁ ≠ y₂) ∧
    (x₁ ≠ 0 ∨ y₁ ≠ 0) ∧ (x₂ ≠ 0 ∨ y₂ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3967_396798


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3967_396778

theorem quadratic_transformation (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 9) →
  ∃ m k, ∀ x, 2 * (a * x^2 + b * x + c) = m * (x - 5)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3967_396778


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3967_396761

theorem arithmetic_sequence_common_difference 
  (a : Fin 4 → ℚ) 
  (h_arithmetic : ∀ i j k, i < j → j < k → a j - a i = a k - a j) 
  (h_first : a 0 = 1) 
  (h_last : a 3 = 2) : 
  ∀ i j, i < j → a j - a i = 1/3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3967_396761
