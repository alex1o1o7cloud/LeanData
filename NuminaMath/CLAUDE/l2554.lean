import Mathlib

namespace NUMINAMATH_CALUDE_three_numbers_product_sum_l2554_255454

theorem three_numbers_product_sum (x y z : ℝ) : 
  (x * y + x + y = 8) ∧ 
  (y * z + y + z = 15) ∧ 
  (x * z + x + z = 24) → 
  (x = 8 ∧ y = 3 ∧ z = 4) := by
sorry

end NUMINAMATH_CALUDE_three_numbers_product_sum_l2554_255454


namespace NUMINAMATH_CALUDE_earth_rotation_certain_l2554_255466

-- Define the type for events
inductive Event : Type
  | EarthRotation : Event
  | RainTomorrow : Event
  | TimeBackwards : Event
  | SnowfallWinter : Event

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.EarthRotation => True
  | _ => False

-- Define the conditions given in the problem
axiom earth_rotation_continuous : ∀ (t : ℝ), ∃ (angle : ℝ), angle ≥ 0 ∧ angle < 360
axiom weather_probabilistic : ∃ (p : ℝ), 0 < p ∧ p < 1
axiom time_forwards : ∀ (t1 t2 : ℝ), t1 < t2 → t1 ≠ t2
axiom snowfall_not_guaranteed : ∃ (winter : Set ℝ), ∃ (day : ℝ), day ∈ winter ∧ ¬∃ (snow : ℝ), snow > 0

-- The theorem to prove
theorem earth_rotation_certain : is_certain Event.EarthRotation :=
  sorry

end NUMINAMATH_CALUDE_earth_rotation_certain_l2554_255466


namespace NUMINAMATH_CALUDE_min_value_ab_l2554_255479

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 5/a + 20/b = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → 5/x + 20/y = 4 → a * b ≤ x * y ∧ a * b = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l2554_255479


namespace NUMINAMATH_CALUDE_parabola_c_value_l2554_255401

/-- A parabola in the xy-plane with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-3) = 2 →   -- Vertex condition
  p.x_coord (-5) = 0 →   -- Point condition
  p.c = -5/2 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2554_255401


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2554_255425

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2554_255425


namespace NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five_l2554_255444

theorem sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five : 
  Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_seven_less_than_two_sqrt_five_l2554_255444


namespace NUMINAMATH_CALUDE_sin_sum_identity_l2554_255416

theorem sin_sum_identity (x : ℝ) (h : Real.sin (2 * x + π / 5) = Real.sqrt 3 / 3) :
  Real.sin (4 * π / 5 - 2 * x) + Real.sin (3 * π / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l2554_255416


namespace NUMINAMATH_CALUDE_cricket_team_selection_l2554_255499

/-- The total number of players in the cricket team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of players to be chosen for the training squad -/
def squad_size : ℕ := 5

/-- The number of ways to choose the training squad under the given restrictions -/
def valid_selections : ℕ := 4356

theorem cricket_team_selection :
  (Nat.choose total_players squad_size) - (Nat.choose (total_players - num_quadruplets) 1) = valid_selections :=
sorry

end NUMINAMATH_CALUDE_cricket_team_selection_l2554_255499


namespace NUMINAMATH_CALUDE_credits_to_graduate_l2554_255480

/-- The number of semesters in college -/
def semesters : ℕ := 8

/-- The number of classes taken per semester -/
def classes_per_semester : ℕ := 5

/-- The number of credits per class -/
def credits_per_class : ℕ := 3

/-- The total number of credits needed to graduate -/
def total_credits : ℕ := semesters * classes_per_semester * credits_per_class

theorem credits_to_graduate : total_credits = 120 := by
  sorry

end NUMINAMATH_CALUDE_credits_to_graduate_l2554_255480


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2554_255462

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 3 - 6 * Complex.I) :
  z.im = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2554_255462


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2554_255473

theorem quadratic_root_sum (p q : ℚ) : 
  (1 - Real.sqrt 3) / 2 = -p / 2 - Real.sqrt ((p / 2) ^ 2 - q) →
  |p| + 2 * |q| = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2554_255473


namespace NUMINAMATH_CALUDE_function_identity_l2554_255419

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (2*x + 1) = 4*x^2 + 4*x) :
  ∀ x, f x = x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2554_255419


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_and_parabola_l2554_255463

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 2

/-- Parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l -/
def l (x y : ℝ) : Prop := y = Real.sqrt 2

theorem line_tangent_to_circle_and_parabola :
  ∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ l p.1 p.2 ∧
  ∃! q : ℝ × ℝ, C₂ q.1 q.2 ∧ l q.1 q.2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_and_parabola_l2554_255463


namespace NUMINAMATH_CALUDE_pencil_count_difference_l2554_255435

theorem pencil_count_difference (D J M E : ℕ) : 
  D = J + 15 → 
  J = 2 * M → 
  E = (J - M) / 2 → 
  J = 20 → 
  D - (M + E) = 20 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_difference_l2554_255435


namespace NUMINAMATH_CALUDE_spider_web_paths_l2554_255405

theorem spider_web_paths : Nat.choose 11 5 = 462 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_l2554_255405


namespace NUMINAMATH_CALUDE_max_b_value_l2554_255440

theorem max_b_value (a b c : ℕ) : 
  a * b * c = 360 →
  1 < c →
  c ≤ b →
  b < a →
  b ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l2554_255440


namespace NUMINAMATH_CALUDE_dynamic_load_calculation_l2554_255412

/-- Given an architectural formula for dynamic load on cylindrical columns -/
theorem dynamic_load_calculation (T H : ℝ) (hT : T = 3) (hH : H = 6) :
  (50 * T^3) / H^3 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_dynamic_load_calculation_l2554_255412


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l2554_255470

theorem polynomial_coefficients (a b c : ℚ) :
  let f : ℚ → ℚ := λ x => c * x^4 + a * x^3 - 3 * x^2 + b * x - 8
  (f 2 = -8) ∧ (f (-3) = -68) → (a = 5 ∧ b = 7 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l2554_255470


namespace NUMINAMATH_CALUDE_rahims_average_book_price_l2554_255431

/-- Calculates the average price per book given two purchases -/
def average_price_per_book (books1 : ℕ) (price1 : ℕ) (books2 : ℕ) (price2 : ℕ) : ℚ :=
  (price1 + price2 : ℚ) / (books1 + books2 : ℚ)

/-- Proves that Rahim's average price per book is 20 rupees -/
theorem rahims_average_book_price :
  average_price_per_book 50 1000 40 800 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rahims_average_book_price_l2554_255431


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2554_255432

def arithmetic_sequence (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_geometric_sequence (d : ℚ) (h : d ≠ 0) :
  let a := arithmetic_sequence 1 d
  (a 2) * (a 9) = (a 4)^2 →
  (∃ q : ℚ, q = 5/2 ∧
    (∀ n : ℕ, arithmetic_sequence 1 d n = 3*n - 2) ∧
    (∀ n : ℕ, sum_arithmetic_sequence 1 d n = (3*n^2 - n) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2554_255432


namespace NUMINAMATH_CALUDE_salary_B_is_5000_l2554_255417

/-- Calculates the salary of person B given the salaries of other people and the average salary -/
def calculate_salary_B (salary_A salary_C salary_D salary_E average_salary : ℕ) : ℕ :=
  5 * average_salary - (salary_A + salary_C + salary_D + salary_E)

/-- Proves that B's salary is 5000 given the conditions in the problem -/
theorem salary_B_is_5000 :
  let salary_A : ℕ := 8000
  let salary_C : ℕ := 11000
  let salary_D : ℕ := 7000
  let salary_E : ℕ := 9000
  let average_salary : ℕ := 8000
  calculate_salary_B salary_A salary_C salary_D salary_E average_salary = 5000 := by
  sorry

#eval calculate_salary_B 8000 11000 7000 9000 8000

end NUMINAMATH_CALUDE_salary_B_is_5000_l2554_255417


namespace NUMINAMATH_CALUDE_lid_circumference_l2554_255446

theorem lid_circumference (diameter : ℝ) (h : diameter = 2) :
  Real.pi * diameter = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_lid_circumference_l2554_255446


namespace NUMINAMATH_CALUDE_inequality_solution_l2554_255472

theorem inequality_solution (x : ℝ) : (x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2554_255472


namespace NUMINAMATH_CALUDE_fraction_reduction_l2554_255433

theorem fraction_reduction (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^2 + b^2 - c^2 + 2*a*b) / (a^2 + c^2 - b^2 + 2*a*c) = (a + b - c) / (a - b + c) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l2554_255433


namespace NUMINAMATH_CALUDE_smallest_positive_a_for_two_roots_in_unit_interval_l2554_255482

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate to check if a quadratic function has two distinct roots in (0,1) -/
def has_two_distinct_roots_in_unit_interval (f : QuadraticFunction) : Prop :=
  ∃ (r s : ℝ), 0 < r ∧ r < 1 ∧ 0 < s ∧ s < 1 ∧ r ≠ s ∧
    f.a * r^2 + f.b * r + f.c = 0 ∧
    f.a * s^2 + f.b * s + f.c = 0

/-- The main theorem stating the smallest positive integer a -/
theorem smallest_positive_a_for_two_roots_in_unit_interval :
  ∃ (a : ℤ), a > 0 ∧
    (∀ (f : QuadraticFunction), f.a = a → has_two_distinct_roots_in_unit_interval f) ∧
    (∀ (a' : ℤ), 0 < a' → a' < a →
      ∃ (f : QuadraticFunction), f.a = a' ∧ ¬has_two_distinct_roots_in_unit_interval f) ∧
    a = 5 :=
  sorry

end NUMINAMATH_CALUDE_smallest_positive_a_for_two_roots_in_unit_interval_l2554_255482


namespace NUMINAMATH_CALUDE_rain_on_tuesday_l2554_255404

theorem rain_on_tuesday (rain_monday : ℝ) (rain_both : ℝ) (no_rain : ℝ) 
  (h1 : rain_monday = 0.6)
  (h2 : rain_both = 0.4)
  (h3 : no_rain = 0.25) :
  ∃ rain_tuesday : ℝ, rain_tuesday = 0.55 ∧ 
  rain_monday + rain_tuesday - rain_both + no_rain = 1 :=
by sorry

end NUMINAMATH_CALUDE_rain_on_tuesday_l2554_255404


namespace NUMINAMATH_CALUDE_symmetric_line_l2554_255460

/-- Given a line L with equation x + 2y - 1 = 0 and a point P(1, -1),
    the line symmetric to L with respect to P has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (x + 2*y - 1 = 0) → -- original line equation
  (∃ (x' y' : ℝ), (x' = 2 - x ∧ y' = -2 - y) ∧ (x' + 2*y' - 1 = 0)) → -- symmetry condition
  (x + 2*y - 3 = 0) -- symmetric line equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2554_255460


namespace NUMINAMATH_CALUDE_tangent_line_difference_l2554_255475

/-- A curve defined by y = x^3 + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A line defined by y = kx + 1 -/
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 1

theorem tangent_line_difference (a b k : ℝ) :
  (curve a b 1 = 2) →  -- The curve passes through (1, 2)
  (line k 1 = 2) →     -- The line passes through (1, 2)
  (curve_derivative a 1 = k) →  -- The derivative of the curve at x=1 equals the slope of the line
  b - a = 5 := by
    sorry


end NUMINAMATH_CALUDE_tangent_line_difference_l2554_255475


namespace NUMINAMATH_CALUDE_taxi_seating_arrangements_l2554_255467

theorem taxi_seating_arrangements :
  let n : ℕ := 6  -- total number of people
  let m : ℕ := 4  -- maximum capacity of each taxi
  let k : ℕ := 2  -- number of taxis
  Nat.choose n m * 2 + (Nat.choose n (n / k)) = 50 :=
by sorry

end NUMINAMATH_CALUDE_taxi_seating_arrangements_l2554_255467


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2554_255437

-- Define the number of individuals and cities
def total_individuals : ℕ := 6
def total_cities : ℕ := 4

-- Define the number of restricted individuals (A and B)
def restricted_individuals : ℕ := 2

-- Function to calculate permutations
def permutations (n k : ℕ) : ℕ := (n.factorial) / (n - k).factorial

-- Theorem statement
theorem selection_schemes_count :
  (permutations total_individuals total_cities) -
  (restricted_individuals * permutations (total_individuals - 1) (total_cities - 1)) = 240 :=
sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2554_255437


namespace NUMINAMATH_CALUDE_power_of_two_equation_solution_l2554_255498

theorem power_of_two_equation_solution :
  ∀ (a n : ℕ), a ≥ n → n ≥ 2 → (∃ x : ℕ, (a + 1)^n + a - 1 = 2^x) →
  a = 4 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_solution_l2554_255498


namespace NUMINAMATH_CALUDE_jason_balloons_l2554_255439

/-- Given an initial number of violet balloons and a number of lost violet balloons,
    calculate the remaining number of violet balloons. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason's remaining violet balloons is 4,
    given he started with 7 and lost 3. -/
theorem jason_balloons : remaining_balloons 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_balloons_l2554_255439


namespace NUMINAMATH_CALUDE_two_correct_statements_l2554_255490

/-- A statement about triangles -/
inductive TriangleStatement
  | altitudes_intersect : TriangleStatement
  | medians_intersect_inside : TriangleStatement
  | right_triangle_one_altitude : TriangleStatement
  | angle_bisectors_intersect : TriangleStatement

/-- Predicate to check if a statement is correct -/
def is_correct (s : TriangleStatement) : Prop :=
  match s with
  | TriangleStatement.altitudes_intersect => true
  | TriangleStatement.medians_intersect_inside => true
  | TriangleStatement.right_triangle_one_altitude => false
  | TriangleStatement.angle_bisectors_intersect => true

/-- The main theorem to prove -/
theorem two_correct_statements :
  ∃ (s1 s2 : TriangleStatement),
    s1 ≠ s2 ∧
    is_correct s1 ∧
    is_correct s2 ∧
    ∀ (s : TriangleStatement),
      s ≠ s1 ∧ s ≠ s2 → ¬(is_correct s) :=
sorry

end NUMINAMATH_CALUDE_two_correct_statements_l2554_255490


namespace NUMINAMATH_CALUDE_number_in_different_bases_l2554_255483

theorem number_in_different_bases : ∃ (n : ℕ), 
  (∃ (a : ℕ), a < 7 ∧ n = a * 7 + 0) ∧ 
  (∃ (a b : ℕ), a < 9 ∧ b < 9 ∧ a ≠ b ∧ n = a * 9 + b) ∧ 
  (n = 3 * 8 + 5) := by
  sorry

end NUMINAMATH_CALUDE_number_in_different_bases_l2554_255483


namespace NUMINAMATH_CALUDE_cannot_obtain_703_from_604_l2554_255465

/-- Represents the computer operations -/
inductive Operation
  | square : Operation
  | split : Operation

/-- Applies the given operation to a natural number -/
def apply_operation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.square => n * n
  | Operation.split => 
      if n < 1000 then n
      else (n % 1000) + (n / 1000)

/-- Checks if it's possible to transform start into target using the given operations -/
def can_transform (start target : ℕ) : Prop :=
  ∃ (seq : List Operation), 
    (seq.foldl (λ acc op => apply_operation op acc) start) = target

/-- The main theorem stating that 703 cannot be obtained from 604 using the given operations -/
theorem cannot_obtain_703_from_604 : ¬ can_transform 604 703 := by
  sorry


end NUMINAMATH_CALUDE_cannot_obtain_703_from_604_l2554_255465


namespace NUMINAMATH_CALUDE_count_b_k_divisible_by_11_l2554_255456

-- Define b_k as a function that takes k and returns the concatenated number
def b (k : ℕ) : ℕ := sorry

-- Define a function to count how many b_k are divisible by 11 for 1 ≤ k ≤ 50
def count_divisible_by_11 : ℕ := sorry

-- Theorem stating that the count of b_k divisible by 11 for 1 ≤ k ≤ 50 is equal to X
theorem count_b_k_divisible_by_11 : count_divisible_by_11 = X := by sorry

end NUMINAMATH_CALUDE_count_b_k_divisible_by_11_l2554_255456


namespace NUMINAMATH_CALUDE_sin_15_deg_squared_value_l2554_255429

theorem sin_15_deg_squared_value : 
  7/16 - 7/8 * (Real.sin (15 * π / 180))^2 = 7 * Real.sqrt 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_deg_squared_value_l2554_255429


namespace NUMINAMATH_CALUDE_remaining_children_meals_l2554_255452

theorem remaining_children_meals (total_children_meals : ℕ) 
  (adults_consumed : ℕ) (child_adult_ratio : ℚ) :
  total_children_meals = 90 →
  adults_consumed = 42 →
  child_adult_ratio = 90 / 70 →
  total_children_meals - (↑adults_consumed * child_adult_ratio).floor = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_children_meals_l2554_255452


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_l2554_255406

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- The midpoint of two points -/
def is_midpoint (x₁ y₁ x₂ y₂ x_mid y_mid : ℝ) : Prop :=
  x_mid = (x₁ + x₂) / 2 ∧ y_mid = (y₁ + y₂) / 2

/-- A point is on a line -/
def is_on_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

/-- The main theorem -/
theorem chord_bisected_by_point (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ →
  is_on_ellipse x₂ y₂ →
  is_midpoint x₁ y₁ x₂ y₂ 4 2 →
  is_on_line x₁ y₁ ∧ is_on_line x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_l2554_255406


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2554_255413

theorem solution_set_inequality (x : ℝ) : 
  x * |x + 2| < 0 ↔ x < -2 ∨ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2554_255413


namespace NUMINAMATH_CALUDE_not_always_parallel_to_plane_l2554_255427

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_to_plane
  (a b : Line) (α : Plane)
  (diff : a ≠ b)
  (h1 : subset b α)
  (h2 : parallel_lines a b) :
  ¬(∀ a b α, subset b α → parallel_lines a b → parallel_line_plane a α) :=
sorry

end NUMINAMATH_CALUDE_not_always_parallel_to_plane_l2554_255427


namespace NUMINAMATH_CALUDE_circle_diameter_from_viewing_angles_l2554_255485

theorem circle_diameter_from_viewing_angles 
  (r : ℝ) (d α β : ℝ) 
  (h_positive : r > 0 ∧ d > 0)
  (h_angles : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2) :
  2 * r = (d * Real.sin α * Real.sin β) / (Real.sin ((α + β)/2) * Real.cos ((α - β)/2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_viewing_angles_l2554_255485


namespace NUMINAMATH_CALUDE_twelve_boys_handshakes_l2554_255428

/-- The number of handshakes when n boys each shake hands exactly once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 12 boys, where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 66 -/
theorem twelve_boys_handshakes : handshakes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_boys_handshakes_l2554_255428


namespace NUMINAMATH_CALUDE_unattainable_y_l2554_255469

theorem unattainable_y (x : ℝ) (y : ℝ) (h1 : x ≠ -3/2) (h2 : y = (1-x)/(2*x+3)) :
  y ≠ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_y_l2554_255469


namespace NUMINAMATH_CALUDE_prop_logic_evaluation_l2554_255447

theorem prop_logic_evaluation (p q : Prop) (hp : p ↔ (2 < 3)) (hq : q ↔ (2 > 3)) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_prop_logic_evaluation_l2554_255447


namespace NUMINAMATH_CALUDE_equal_chord_length_l2554_255411

/-- Given a circle C and two lines l1 and l2, prove that they intercept chords of equal length on C -/
theorem equal_chord_length (r d : ℝ) (h : r > 0) :
  let C := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let l1 := {p : ℝ × ℝ | 2 * p.1 + 3 * p.2 + 1 = 0}
  let l2 := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 - 1 = 0}
  let chord_length (l : Set (ℝ × ℝ)) := 
    Real.sqrt (4 * r^2 - 4 * (1 / (2^2 + 3^2)))
  chord_length l1 = d → chord_length l2 = d :=
by sorry

end NUMINAMATH_CALUDE_equal_chord_length_l2554_255411


namespace NUMINAMATH_CALUDE_money_distribution_l2554_255494

/-- Given that p, q, and r have $9000 among themselves, and r has two-thirds of the total amount with p and q, prove that r has $3600. -/
theorem money_distribution (p q r : ℝ) 
  (total : p + q + r = 9000)
  (r_proportion : r = (2/3) * (p + q)) :
  r = 3600 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l2554_255494


namespace NUMINAMATH_CALUDE_solution_system_equations_l2554_255403

theorem solution_system_equations (a : ℝ) (ha : a ≠ 0) :
  let x₁ := a / 2 + (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let y₁ := a / 2 - (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let x₂ := a / 2 - (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let y₂ := a / 2 + (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  (x₁ + y₁ = a ∧ x₁^5 + y₁^5 = 2 * a^5) ∧
  (x₂ + y₂ = a ∧ x₂^5 + y₂^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x + y = a ∧ x^5 + y^5 = 2 * a^5 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2554_255403


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_of_72_l2554_255488

theorem smallest_sum_of_factors_of_72 :
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c = 72 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (x y z : ℕ), 
      x ≠ y ∧ y ≠ z ∧ x ≠ z →
      x * y * z = 72 →
      x > 0 ∧ y > 0 ∧ z > 0 →
      a + b + c ≤ x + y + z ∧
      a + b + c = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_of_72_l2554_255488


namespace NUMINAMATH_CALUDE_paths_in_7x8_grid_l2554_255418

/-- The number of paths in a grid with only upward and rightward movements -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of paths in a 7x8 grid is 6435 -/
theorem paths_in_7x8_grid :
  gridPaths 7 8 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x8_grid_l2554_255418


namespace NUMINAMATH_CALUDE_max_value_of_f_l2554_255484

-- Define the function f
def f (x : ℝ) : ℝ := x * (4 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ x, x ∈ Set.Ioo 0 4 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2554_255484


namespace NUMINAMATH_CALUDE_brian_white_stones_l2554_255491

/-- Represents Brian's stone collection -/
structure StoneCollection where
  white : ℕ
  black : ℕ
  grey : ℕ
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Conditions of Brian's stone collection -/
def BrianCollection : StoneCollection → Prop := fun c =>
  c.white + c.black = 100 ∧
  c.grey + c.green = 100 ∧
  c.red + c.blue = 130 ∧
  c.white + c.black + c.grey + c.green + c.red + c.blue = 330 ∧
  c.white > c.black ∧
  c.white = c.grey ∧
  c.black = c.green ∧
  3 * c.blue = 2 * c.red ∧
  2 * (c.white + c.grey) = c.red

theorem brian_white_stones (c : StoneCollection) 
  (h : BrianCollection c) : c.white = 78 := by
  sorry

end NUMINAMATH_CALUDE_brian_white_stones_l2554_255491


namespace NUMINAMATH_CALUDE_second_class_average_mark_l2554_255459

theorem second_class_average_mark (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg_total : ℝ) 
  (h1 : students1 = 22)
  (h2 : students2 = 28)
  (h3 : avg1 = 40)
  (h4 : avg_total = 51.2) :
  (avg_total * (students1 + students2) - avg1 * students1) / students2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_class_average_mark_l2554_255459


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_half_l2554_255455

/-- If the terminal side of angle α passes through point P(-1, 2), then tan(α + π/2) = 1/2 -/
theorem tan_alpha_plus_pi_half (α : Real) : 
  (Real.tan α = -2) → Real.tan (α + π/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_half_l2554_255455


namespace NUMINAMATH_CALUDE_min_value_expression_l2554_255407

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  (a * c / b) + (c / (a * b)) - (c / 2) + (Real.sqrt 5 / (c - 2)) ≥ Real.sqrt 10 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2554_255407


namespace NUMINAMATH_CALUDE_range_of_m_l2554_255453

/-- A function f is monotonically increasing on ℝ -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The theorem statement -/
theorem range_of_m (f : ℝ → ℝ) (h1 : MonoIncreasing f) :
  ∀ m : ℝ, f (m^2) > f (-m) → m < -1 ∨ m > 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2554_255453


namespace NUMINAMATH_CALUDE_ratio_of_xy_l2554_255486

theorem ratio_of_xy (x y : ℝ) 
  (h1 : Real.sqrt (3 * x) * (1 + 1 / (x + y)) = 2)
  (h2 : Real.sqrt (7 * y) * (1 - 1 / (x + y)) = 4 * Real.sqrt 2)
  (h3 : x ≠ 0) : y / x = 6 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_xy_l2554_255486


namespace NUMINAMATH_CALUDE_sin_shift_l2554_255457

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l2554_255457


namespace NUMINAMATH_CALUDE_rectangle_width_l2554_255443

theorem rectangle_width (perimeter length width : ℝ) : 
  perimeter = 16 ∧ 
  width = length + 2 ∧ 
  perimeter = 2 * (length + width) →
  width = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l2554_255443


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2554_255496

theorem min_value_of_sum (x y z : ℝ) (h : x^2 + 2*y^2 + 5*z^2 = 22) :
  ∃ (m : ℝ), m = xy - yz - zx ∧ m ≥ (-55 - 11*Real.sqrt 5) / 10 ∧
  (∃ (x' y' z' : ℝ), x'^2 + 2*y'^2 + 5*z'^2 = 22 ∧
    x'*y' - y'*z' - z'*x' = (-55 - 11*Real.sqrt 5) / 10) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2554_255496


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2554_255471

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2554_255471


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l2554_255468

theorem smallest_multiples_sum : ∃ (a b : ℕ),
  (a ≥ 10 ∧ a < 100 ∧ a % 5 = 0 ∧ ∀ x : ℕ, (x ≥ 10 ∧ x < 100 ∧ x % 5 = 0) → a ≤ x) ∧
  (b ≥ 100 ∧ b < 1000 ∧ b % 6 = 0 ∧ ∀ y : ℕ, (y ≥ 100 ∧ y < 1000 ∧ y % 6 = 0) → b ≤ y) ∧
  a + b = 112 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l2554_255468


namespace NUMINAMATH_CALUDE_quiz_competition_arrangements_l2554_255436

/-- The number of permutations of k items chosen from n distinct items -/
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- Theorem: There are 24 ways to arrange 3 out of 4 distinct items in order -/
theorem quiz_competition_arrangements : permutations 4 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_arrangements_l2554_255436


namespace NUMINAMATH_CALUDE_pamelas_remaining_skittles_l2554_255421

def initial_skittles : ℕ := 50
def skittles_given : ℕ := 7

theorem pamelas_remaining_skittles :
  initial_skittles - skittles_given = 43 := by
  sorry

end NUMINAMATH_CALUDE_pamelas_remaining_skittles_l2554_255421


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2554_255445

theorem polynomial_expansion_equality (x y : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * (x + y)^2 =
  5*x^4 + 35*x^3 + 960*x^2 + 1649*x + 4000 - 8*x*y - 4*y^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2554_255445


namespace NUMINAMATH_CALUDE_football_game_cost_l2554_255461

/-- The cost of a football game, given the total spent and the costs of two other games. -/
theorem football_game_cost (total_spent strategy_cost batman_cost : ℚ) :
  total_spent = 35.52 ∧ strategy_cost = 9.46 ∧ batman_cost = 12.04 →
  total_spent - (strategy_cost + batman_cost) = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_football_game_cost_l2554_255461


namespace NUMINAMATH_CALUDE_school_sampling_is_systematic_l2554_255449

/-- Represents a student with a unique student number -/
structure Student where
  number : ℕ

/-- Represents the sampling method used -/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Function to check if a student number ends in 4 -/
def endsInFour (n : ℕ) : Bool :=
  n % 10 = 4

/-- The sampling method used in the school -/
def schoolSamplingMethod (students : List Student) : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the school's sampling method is systematic sampling -/
theorem school_sampling_is_systematic (students : List Student) :
  schoolSamplingMethod students = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_school_sampling_is_systematic_l2554_255449


namespace NUMINAMATH_CALUDE_parallel_properties_l2554_255423

-- Define a type for lines
variable {Line : Type}

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define a relation for two lines being parallel to the same line
def parallel_to_same (l1 l2 : Line) : Prop :=
  ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3

theorem parallel_properties :
  (∀ l1 l2 : Line, parallel_to_same parallel l1 l2 → parallel l1 l2) ∧
  (∀ l1 l2 : Line, parallel l1 l2 → parallel_to_same parallel l1 l2) ∧
  (∀ l1 l2 : Line, ¬parallel_to_same parallel l1 l2 → ¬parallel l1 l2) ∧
  (∀ l1 l2 : Line, ¬parallel l1 l2 → ¬parallel_to_same parallel l1 l2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_properties_l2554_255423


namespace NUMINAMATH_CALUDE_a_minus_b_value_l2554_255476

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l2554_255476


namespace NUMINAMATH_CALUDE_cube_inequality_l2554_255451

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2554_255451


namespace NUMINAMATH_CALUDE_problem_statement_l2554_255410

theorem problem_statement (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π/4)) 
  (h2 : Real.sin θ - Real.cos θ = -Real.sqrt 14 / 4) : 
  (2 * (Real.cos θ)^2 - 1) / Real.cos (π/4 + θ) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2554_255410


namespace NUMINAMATH_CALUDE_expression_divisibility_l2554_255477

theorem expression_divisibility (n : ℕ) (x : ℝ) (hx : x ≠ 1) :
  ∃ g : ℝ → ℝ, n * x^(n+1) * (1 - 1/x) - x^n * (1 - 1/x^n) = (x - 1)^2 * g x :=
by sorry

end NUMINAMATH_CALUDE_expression_divisibility_l2554_255477


namespace NUMINAMATH_CALUDE_rook_placement_theorem_l2554_255415

theorem rook_placement_theorem (n : ℕ) (h1 : n > 2) (h2 : Even n) :
  ∃ (coloring : Fin n → Fin n → Fin (n^2/2))
    (rook_positions : Fin n → Fin n × Fin n),
    (∀ i j : Fin n, (∃! k : Fin n, coloring i k = coloring j k) ∨ i = j) ∧
    (∀ i j : Fin n, i ≠ j →
      (rook_positions i).1 ≠ (rook_positions j).1 ∧
      (rook_positions i).2 ≠ (rook_positions j).2) ∧
    (∀ i j : Fin n, i ≠ j →
      coloring (rook_positions i).1 (rook_positions i).2 ≠
      coloring (rook_positions j).1 (rook_positions j).2) :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_theorem_l2554_255415


namespace NUMINAMATH_CALUDE_sequence_properties_l2554_255497

/-- Proof of properties of sequences A, G, and H -/
theorem sequence_properties
  (x y k : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hk : k > 0)
  (hxy : x ≠ y)
  (hk1 : k ≠ 1)
  (A : ℕ → ℝ)
  (G : ℕ → ℝ)
  (H : ℕ → ℝ)
  (hA1 : A 1 = (k * x + y) / (k + 1))
  (hG1 : G 1 = (x^k * y)^(1 / (k + 1)))
  (hH1 : H 1 = ((k + 1) * x * y) / (k * x + y))
  (hAn : ∀ n ≥ 2, A n = (A (n-1) + H (n-1)) / 2)
  (hGn : ∀ n ≥ 2, G n = (A (n-1) * H (n-1))^(1/2))
  (hHn : ∀ n ≥ 2, H n = 2 / (1 / A (n-1) + 1 / H (n-1))) :
  (∀ n ≥ 1, A (n+1) < A n) ∧
  (∀ n ≥ 1, G (n+1) = G n) ∧
  (∀ n ≥ 1, H n < H (n+1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2554_255497


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l2554_255493

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 7

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select two shoes from the box -/
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2

/-- The number of ways to select a matching pair of shoes -/
def matching_pairs : ℕ := num_pairs

/-- The probability of selecting a matching pair of shoes -/
def probability : ℚ := matching_pairs / total_combinations

theorem matching_shoes_probability :
  probability = 1 / 13 := by sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l2554_255493


namespace NUMINAMATH_CALUDE_determinant_equality_l2554_255474

theorem determinant_equality (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 7 →
  Matrix.det ![![x + 2*z, y + 2*w], ![z, w]] = 7 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l2554_255474


namespace NUMINAMATH_CALUDE_convex_polygon_27_diagonals_has_9_sides_l2554_255408

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 27 diagonals has 9 sides --/
theorem convex_polygon_27_diagonals_has_9_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 27 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_27_diagonals_has_9_sides_l2554_255408


namespace NUMINAMATH_CALUDE_triangle_existence_from_bisector_and_segments_l2554_255422

/-- Given an angle bisector and the segments it divides a side into,
    prove the existence of a triangle satisfying these conditions. -/
theorem triangle_existence_from_bisector_and_segments
  (l_c a' b' : ℝ) (h_positive : l_c > 0 ∧ a' > 0 ∧ b' > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    l_c ^ 2 = a * b - a' * b' ∧
    a' / b' = a / b :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_from_bisector_and_segments_l2554_255422


namespace NUMINAMATH_CALUDE_no_real_solutions_for_complex_product_l2554_255402

theorem no_real_solutions_for_complex_product : 
  ¬∃ (x : ℝ), (Complex.I : ℂ).im * ((x + 2 + Complex.I) * ((x + 3) + 2 * Complex.I) * ((x + 4) + Complex.I)).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_complex_product_l2554_255402


namespace NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_is_zero_l2554_255414

theorem nearest_integer_to_x_minus_y_is_zero
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : 2 * abs x + y = 5)
  (h2 : abs x * y + x^2 = 0) :
  round (x - y) = 0 :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_is_zero_l2554_255414


namespace NUMINAMATH_CALUDE_mistaken_calculation_correction_l2554_255442

theorem mistaken_calculation_correction (x : ℤ) : 
  x - 15 + 27 = 41 → x - 27 + 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_correction_l2554_255442


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2554_255450

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 284000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { significand := 2.84
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.significand * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2554_255450


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2554_255481

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2554_255481


namespace NUMINAMATH_CALUDE_sallys_garden_area_l2554_255489

/-- Represents a rectangular garden with fence posts. -/
structure GardenFence where
  total_posts : ℕ
  post_spacing : ℕ
  long_side_post_ratio : ℕ

/-- Calculates the area of the garden given its fence configuration. -/
def garden_area (fence : GardenFence) : ℕ :=
  let short_side_posts := (fence.total_posts / 2) / (fence.long_side_post_ratio + 1)
  let long_side_posts := short_side_posts * fence.long_side_post_ratio
  let short_side_length := (short_side_posts - 1) * fence.post_spacing
  let long_side_length := (long_side_posts - 1) * fence.post_spacing
  short_side_length * long_side_length

/-- Theorem stating that Sally's garden has an area of 297 square yards. -/
theorem sallys_garden_area :
  let sally_fence := GardenFence.mk 24 3 3
  garden_area sally_fence = 297 := by
  sorry

end NUMINAMATH_CALUDE_sallys_garden_area_l2554_255489


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_l2554_255424

theorem product_of_sums_equals_difference (n : ℕ) :
  (5 + 1) * (5^2 + 1^2) * (5^4 + 1^4) * (5^8 + 1^8) * (5^16 + 1^16) * (5^32 + 1^32) * (5^64 + 1^64) = 5^128 - 1^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_l2554_255424


namespace NUMINAMATH_CALUDE_water_displaced_squared_l2554_255478

/-- The volume of water displaced by a cube submerged in a cylindrical barrel -/
def water_displaced (cube_side : ℝ) (barrel_radius : ℝ) (barrel_height : ℝ) : ℝ :=
  cube_side ^ 3

/-- Theorem: The square of the volume of water displaced by a 10-foot cube
    in a cylindrical barrel is 1,000,000 cubic feet -/
theorem water_displaced_squared :
  let cube_side : ℝ := 10
  let barrel_radius : ℝ := 5
  let barrel_height : ℝ := 15
  (water_displaced cube_side barrel_radius barrel_height) ^ 2 = 1000000 := by
sorry

end NUMINAMATH_CALUDE_water_displaced_squared_l2554_255478


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l2554_255426

theorem least_k_for_inequality (n : ℕ) : 
  (0.0010101 : ℝ) * (10 : ℝ) ^ ((1586 : ℝ) / 500) > (n^2 - 3*n + 5 : ℝ) / (n^3 + 1) ∧ 
  ∀ k : ℚ, k < 1586/500 → (0.0010101 : ℝ) * (10 : ℝ) ^ (k : ℝ) ≤ (n^2 - 3*n + 5 : ℝ) / (n^3 + 1) :=
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l2554_255426


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l2554_255434

theorem smallest_integer_bound (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d) / 4 = 74 →
  d = 90 →
  max a (max b c) ≤ d →
  min a (min b c) ≥ 29 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l2554_255434


namespace NUMINAMATH_CALUDE_common_tangent_theorem_l2554_255400

/-- The value of 'a' for which the graphs of f(x) = ln(x) and g(x) = x^2 + ax 
    have a common tangent line parallel to y = x -/
def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ 
    (1 / x₁ = 1) ∧ 
    (2 * x₂ + a = 1) ∧ 
    (x₂^2 + a * x₂ = x₂ - 1)

theorem common_tangent_theorem :
  ∀ a : ℝ, tangent_condition a → (a = 3 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_theorem_l2554_255400


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_new_mixture_l2554_255430

def original_volume : ℝ := 24
def water_volume : ℝ := 16

def alcohol_A_fraction : ℝ := 0.3
def alcohol_B_fraction : ℝ := 0.4
def alcohol_C_fraction : ℝ := 0.3

def alcohol_A_purity : ℝ := 0.8
def alcohol_B_purity : ℝ := 0.9
def alcohol_C_purity : ℝ := 0.95

def new_mixture_volume : ℝ := original_volume + water_volume

def total_pure_alcohol : ℝ :=
  original_volume * (
    alcohol_A_fraction * alcohol_A_purity +
    alcohol_B_fraction * alcohol_B_purity +
    alcohol_C_fraction * alcohol_C_purity
  )

theorem alcohol_percentage_in_new_mixture :
  (total_pure_alcohol / new_mixture_volume) * 100 = 53.1 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_new_mixture_l2554_255430


namespace NUMINAMATH_CALUDE_larger_number_proof_l2554_255492

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 23)
  (lcm_eq : Nat.lcm a b = 23 * 16 * 17) :
  max a b = 391 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2554_255492


namespace NUMINAMATH_CALUDE_equation_one_solution_l2554_255464

theorem equation_one_solution : 
  {x : ℝ | (x + 3)^2 - 9 = 0} = {0, -6} := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2554_255464


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2554_255420

theorem complex_arithmetic_equality : (5 - Complex.I) - (3 - Complex.I) - 5 * Complex.I = 2 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2554_255420


namespace NUMINAMATH_CALUDE_max_quotient_value_l2554_255495

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 1200 ≤ b ∧ b ≤ 2400) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 1200 ≤ y ∧ y ≤ 2400 → y / x ≤ b / a) →
  b / a = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2554_255495


namespace NUMINAMATH_CALUDE_range_of_a_l2554_255441

theorem range_of_a (a : ℝ) : 
  Real.sqrt ((2*a - 1)^2) = 1 - 2*a → a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2554_255441


namespace NUMINAMATH_CALUDE_expression_value_l2554_255458

theorem expression_value (x y : ℝ) (h : x - y = 1) : 3*x - 3*y + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2554_255458


namespace NUMINAMATH_CALUDE_two_red_one_blue_probability_l2554_255487

/-- The probability of selecting 2 red and 1 blue marble from a bag of 12 red and 8 blue marbles -/
theorem two_red_one_blue_probability (total : ℕ) (red : ℕ) (blue : ℕ) (selected : ℕ) :
  total = red + blue →
  total = 20 →
  red = 12 →
  blue = 8 →
  selected = 3 →
  (Nat.choose red 2 * Nat.choose blue 1 : ℚ) / Nat.choose total selected = 44 / 95 := by
  sorry

#check two_red_one_blue_probability

end NUMINAMATH_CALUDE_two_red_one_blue_probability_l2554_255487


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l2554_255448

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 169 ∧ x2^2 + y^2 = 169 ∧ x1 + x2 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l2554_255448


namespace NUMINAMATH_CALUDE_sum_reciprocal_and_sum_squares_l2554_255438

theorem sum_reciprocal_and_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (1 / a + 1 / b ≥ 4) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → a^2 + b^2 ≤ x^2 + y^2) := by
  sorry

#check sum_reciprocal_and_sum_squares

end NUMINAMATH_CALUDE_sum_reciprocal_and_sum_squares_l2554_255438


namespace NUMINAMATH_CALUDE_billy_score_problem_l2554_255409

/-- Billy's video game score problem -/
theorem billy_score_problem (old_score : ℕ) (rounds : ℕ) : 
  old_score = 725 → rounds = 363 → (old_score + 1) / rounds = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_score_problem_l2554_255409
