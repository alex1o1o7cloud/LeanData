import Mathlib

namespace NUMINAMATH_CALUDE_product_inequality_l2480_248053

theorem product_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + 
  (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2480_248053


namespace NUMINAMATH_CALUDE_unique_solution_fourth_power_equation_l2480_248040

theorem unique_solution_fourth_power_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (4 * x)^5 = (8 * x)^4 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fourth_power_equation_l2480_248040


namespace NUMINAMATH_CALUDE_triangle_inequality_l2480_248007

theorem triangle_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hab : a ≥ b) (hbc : b ≥ c) :
  Real.sqrt (a * (a + b - Real.sqrt (a * b))) +
  Real.sqrt (b * (a + c - Real.sqrt (a * c))) +
  Real.sqrt (c * (b + c - Real.sqrt (b * c))) ≥
  a + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2480_248007


namespace NUMINAMATH_CALUDE_largest_integer_square_4_digits_base8_l2480_248073

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def N : ℕ := 63

/-- Conversion of a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem largest_integer_square_4_digits_base8 :
  (N^2 ≥ 8^3) ∧ (N^2 < 8^4) ∧ (∀ m : ℕ, m > N → m^2 ≥ 8^4) ∧ (toBase8 N = [7, 7]) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_square_4_digits_base8_l2480_248073


namespace NUMINAMATH_CALUDE_total_hamburgers_bought_l2480_248022

/-- Calculates the total number of hamburgers bought given the conditions --/
theorem total_hamburgers_bought
  (total_spent : ℚ)
  (single_burger_cost : ℚ)
  (double_burger_cost : ℚ)
  (double_burgers_count : ℕ)
  (h1 : total_spent = 68.5)
  (h2 : single_burger_cost = 1)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers_count = 37) :
  ∃ (single_burgers_count : ℕ),
    single_burgers_count + double_burgers_count = 50 ∧
    total_spent = single_burger_cost * single_burgers_count + double_burger_cost * double_burgers_count :=
by
  sorry


end NUMINAMATH_CALUDE_total_hamburgers_bought_l2480_248022


namespace NUMINAMATH_CALUDE_fraction_multiplication_and_addition_l2480_248051

theorem fraction_multiplication_and_addition : (2 : ℚ) / 9 * 5 / 11 + 1 / 3 = 43 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_and_addition_l2480_248051


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_l2480_248082

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem smallest_n_with_conditions : 
  ∀ n : ℕ, n > 0 → n % 3 = 0 → digit_product n = 882 → n ≥ 13677 := by
  sorry

#check smallest_n_with_conditions

end NUMINAMATH_CALUDE_smallest_n_with_conditions_l2480_248082


namespace NUMINAMATH_CALUDE_final_selling_price_l2480_248009

/-- The final selling price of a commodity after markup and reduction -/
theorem final_selling_price (a : ℝ) : 
  let initial_markup : ℝ := 1.25
  let price_reduction : ℝ := 0.9
  a * initial_markup * price_reduction = 1.125 * a := by sorry

end NUMINAMATH_CALUDE_final_selling_price_l2480_248009


namespace NUMINAMATH_CALUDE_function_monotonicity_l2480_248000

theorem function_monotonicity (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (f : Real → Real) 
  (hf : ∀ x, f x = Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ))
  (h3 : f (π / 2) = 0) :
  StrictMonoOn f (Set.Ioo (π / 4) (3 * π / 4)) := by
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l2480_248000


namespace NUMINAMATH_CALUDE_triangle_sin_A_l2480_248098

theorem triangle_sin_A (a b : ℝ) (sinB : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sinB = 2/3) :
  let sinA := a * sinB / b
  sinA = 1/2 := by sorry

end NUMINAMATH_CALUDE_triangle_sin_A_l2480_248098


namespace NUMINAMATH_CALUDE_triangle_square_count_l2480_248057

/-- Represents a geometric figure with three layers -/
structure ThreeLayerFigure where
  first_layer_triangles : Nat
  second_layer_squares : Nat
  third_layer_triangle : Nat

/-- Counts the total number of triangles in the figure -/
def count_triangles (figure : ThreeLayerFigure) : Nat :=
  figure.first_layer_triangles + figure.third_layer_triangle

/-- Counts the total number of squares in the figure -/
def count_squares (figure : ThreeLayerFigure) : Nat :=
  figure.second_layer_squares

/-- The specific figure described in the problem -/
def problem_figure : ThreeLayerFigure :=
  { first_layer_triangles := 3
  , second_layer_squares := 2
  , third_layer_triangle := 1 }

theorem triangle_square_count :
  count_triangles problem_figure = 4 ∧ count_squares problem_figure = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_count_l2480_248057


namespace NUMINAMATH_CALUDE_platform_length_l2480_248083

/-- Given a train and two structures it passes through, calculate the length of the second structure. -/
theorem platform_length
  (train_length : ℝ)
  (tunnel_length : ℝ)
  (tunnel_time : ℝ)
  (platform_time : ℝ)
  (h1 : train_length = 330)
  (h2 : tunnel_length = 1200)
  (h3 : tunnel_time = 45)
  (h4 : platform_time = 15) :
  (tunnel_length + train_length) / tunnel_time * platform_time - train_length = 180 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2480_248083


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l2480_248013

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to both foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  sum_distances : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParameters where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def compute_ellipse_parameters (e : Ellipse) : EllipseParameters :=
  sorry

/-- The main theorem: sum of center coordinates and axes lengths for the given ellipse -/
theorem ellipse_parameter_sum (e : Ellipse) 
    (h : e.F₁ = (0, 2) ∧ e.F₂ = (6, 2) ∧ e.sum_distances = 10) : 
    let p := compute_ellipse_parameters e
    p.h + p.k + p.a + p.b = 14 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l2480_248013


namespace NUMINAMATH_CALUDE_stream_speed_l2480_248002

/-- Proves that the speed of a stream is 3 kmph given certain conditions about a boat's travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_time = 1)
  (h3 : upstream_time = 1.5) :
  let stream_speed := (boat_speed * (upstream_time - downstream_time)) / (upstream_time + downstream_time)
  stream_speed = 3 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l2480_248002


namespace NUMINAMATH_CALUDE_initial_milk_percentage_l2480_248049

/-- Given a mixture of milk and water, prove that the initial milk percentage is 84% -/
theorem initial_milk_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_milk_percentage : ℝ) :
  initial_volume = 60 →
  added_water = 14.117647058823536 →
  final_milk_percentage = 68 →
  (initial_volume * (84 / 100)) / (initial_volume + added_water) = final_milk_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_milk_percentage_l2480_248049


namespace NUMINAMATH_CALUDE_base4_10201_equals_289_l2480_248088

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10201_equals_289 :
  base4_to_decimal [1, 0, 2, 0, 1] = 289 := by
  sorry

end NUMINAMATH_CALUDE_base4_10201_equals_289_l2480_248088


namespace NUMINAMATH_CALUDE_jim_skips_proof_l2480_248077

/-- The number of times Bob can skip a rock. -/
def bob_skips : ℕ := 12

/-- The number of rocks Bob and Jim each skipped. -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for both Bob and Jim. -/
def total_skips : ℕ := 270

/-- The number of times Jim can skip a rock. -/
def jim_skips : ℕ := 15

theorem jim_skips_proof : 
  bob_skips * rocks_skipped + jim_skips * rocks_skipped = total_skips :=
by sorry

end NUMINAMATH_CALUDE_jim_skips_proof_l2480_248077


namespace NUMINAMATH_CALUDE_log_x_125_l2480_248045

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_x_125 (x : ℝ) (h : log 8 (5 * x) = 3) : 
  log x 125 = 3 / log 8 5 := by sorry

end NUMINAMATH_CALUDE_log_x_125_l2480_248045


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2480_248099

/-- Given an arithmetic sequence {a_n} with sum S_n, prove that if S_6 = 36, S_n = 324, S_(n-6) = 144, and n > 0, then n = 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = (k / 2) * (2 * a 1 + (k - 1) * (a 2 - a 1))) →  -- Definition of S_n
  (n > 0) →                                                   -- Condition: n > 0
  (S 6 = 36) →                                                -- Condition: S_6 = 36
  (S n = 324) →                                               -- Condition: S_n = 324
  (S (n - 6) = 144) →                                         -- Condition: S_(n-6) = 144
  (n = 18) :=                                                 -- Conclusion: n = 18
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2480_248099


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2480_248054

theorem fraction_equation_solution (x : ℝ) : 
  (3 - x) / (2 - x) - 1 / (x - 2) = 3 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2480_248054


namespace NUMINAMATH_CALUDE_cylinder_rotations_l2480_248094

theorem cylinder_rotations (board_length : ℝ) (cyl1_circ cyl2_circ increase : ℝ) : 
  board_length = 6 →
  cyl1_circ = 3 →
  cyl2_circ = 2 →
  increase = 2 →
  (board_length * 10) / (cyl1_circ + increase) = 
    (board_length * 10) / (cyl2_circ + increase) + 3 := by
  sorry

#check cylinder_rotations

end NUMINAMATH_CALUDE_cylinder_rotations_l2480_248094


namespace NUMINAMATH_CALUDE_petes_number_l2480_248080

theorem petes_number (x : ℝ) : 5 * (3 * x - 6) = 195 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2480_248080


namespace NUMINAMATH_CALUDE_skip_speed_relation_l2480_248059

theorem skip_speed_relation (bruce_speed : ℝ) : 
  let tony_speed := 2 * bruce_speed
  let brandon_speed := (1/3) * tony_speed
  let colin_speed := 6 * brandon_speed
  colin_speed = 4 → bruce_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_skip_speed_relation_l2480_248059


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l2480_248064

open Real

/-- Given a function f(x) = -3ln(x) + ax² + bx, where a > 0 and b is real,
    if for any x > 0, f(x) ≥ f(3), then ln(a) < -b - 1 -/
theorem function_minimum_implies_inequality (a b : ℝ) (ha : a > 0) :
  (∀ x > 0, -3 * log x + a * x^2 + b * x ≥ -3 * log 3 + 9 * a + 3 * b) →
  log a < -b - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l2480_248064


namespace NUMINAMATH_CALUDE_quadratic_integer_solutions_l2480_248092

theorem quadratic_integer_solutions (p q x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  |x₁ - x₂| = 1 →
  |p - q| = 1 →
  (∃ (p' q' x₁' x₂' : ℤ), p = p' ∧ q = q' ∧ x₁ = x₁' ∧ x₂ = x₂') := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solutions_l2480_248092


namespace NUMINAMATH_CALUDE_descent_problem_l2480_248050

/-- The number of floors Austin and Jake descended. -/
def floors : ℕ := sorry

/-- The number of steps Jake descends per second. -/
def steps_per_second : ℕ := 3

/-- The number of steps per floor. -/
def steps_per_floor : ℕ := 30

/-- The time (in seconds) it takes Austin to reach the ground floor using the elevator. -/
def austin_time : ℕ := 60

/-- The time (in seconds) it takes Jake to reach the ground floor using the stairs. -/
def jake_time : ℕ := 90

theorem descent_problem :
  floors = (jake_time * steps_per_second) / steps_per_floor := by
  sorry

end NUMINAMATH_CALUDE_descent_problem_l2480_248050


namespace NUMINAMATH_CALUDE_divisor_condition_l2480_248020

def satisfies_condition (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k l : ℕ, k ∣ n → l ∣ n → k < n → l < n →
    (2*k - l) ∣ n ∨ (2*l - k) ∣ n

theorem divisor_condition (n : ℕ) :
  satisfies_condition n ↔ Nat.Prime n ∨ n ∈ ({6, 9, 15} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_divisor_condition_l2480_248020


namespace NUMINAMATH_CALUDE_smallest_value_S_l2480_248012

theorem smallest_value_S (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ d₁ d₂ d₃ : ℕ) : 
  ({a₁, a₂, a₃, b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃} = Finset.range 12) →
  (a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ + d₁ * d₂ * d₃ ≥ 613) ∧
  (∃ (a₁' a₂' a₃' b₁' b₂' b₃' c₁' c₂' c₃' d₁' d₂' d₃' : ℕ),
    {a₁', a₂', a₃', b₁', b₂', b₃', c₁', c₂', c₃', d₁', d₂', d₃'} = Finset.range 12 ∧
    a₁' * a₂' * a₃' + b₁' * b₂' * b₃' + c₁' * c₂' * c₃' + d₁' * d₂' * d₃' = 613) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_S_l2480_248012


namespace NUMINAMATH_CALUDE_min_both_composers_l2480_248019

theorem min_both_composers (total : ℕ) (beethoven : ℕ) (chopin : ℕ) 
  (h1 : total = 130) 
  (h2 : beethoven = 110) 
  (h3 : chopin = 90) 
  (h4 : beethoven ≤ total) 
  (h5 : chopin ≤ total) : 
  (beethoven + chopin - total : ℤ) ≥ 70 := by
  sorry

end NUMINAMATH_CALUDE_min_both_composers_l2480_248019


namespace NUMINAMATH_CALUDE_alice_savings_l2480_248075

/-- Alice's savings calculation --/
theorem alice_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) :
  sales = 2500 →
  basic_salary = 240 →
  commission_rate = 0.02 →
  savings_rate = 0.1 →
  (basic_salary + sales * commission_rate) * savings_rate = 29 := by
  sorry

end NUMINAMATH_CALUDE_alice_savings_l2480_248075


namespace NUMINAMATH_CALUDE_cd_case_side_length_l2480_248014

/-- Given a square CD case with a circumference of 60 centimeters,
    prove that the length of one side is 15 centimeters. -/
theorem cd_case_side_length (circumference : ℝ) (side_length : ℝ) 
  (h1 : circumference = 60) 
  (h2 : circumference = 4 * side_length) : 
  side_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_cd_case_side_length_l2480_248014


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l2480_248061

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, and a point M(x_0, 2√2) on the parabola,
    if a circle with center M is tangent to the y-axis and intersects MF at A such that |MA| / |AF| = 2,
    then p = 2 -/
theorem parabola_circle_intersection (C : Parabola) (M : PointOnParabola C)
  (circ : Circle) (A : ℝ × ℝ) :
  M.y = 2 * Real.sqrt 2 →
  circ.center = (M.x, M.y) →
  circ.radius = M.x →
  A.1 = M.x - C.p →
  (M.x - A.1) / A.1 = 2 →
  C.p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l2480_248061


namespace NUMINAMATH_CALUDE_secant_minimum_value_l2480_248041

/-- The secant function -/
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

/-- The function y = a * sec(bx + c) -/
noncomputable def f (a b c x : ℝ) : ℝ := a * sec (b * x + c)

theorem secant_minimum_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x : ℝ, f a b c x > 0 → f a b c x ≥ 3) →
  (∃ x : ℝ, f a b c x = 3) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_secant_minimum_value_l2480_248041


namespace NUMINAMATH_CALUDE_dividend_calculation_l2480_248085

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 20)
  (h2 : quotient = 8)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 166 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2480_248085


namespace NUMINAMATH_CALUDE_max_pieces_cut_l2480_248063

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The plywood sheet -/
def plywood : Rectangle := { length := 22, width := 15 }

/-- The piece to be cut -/
def piece : Rectangle := { length := 3, width := 5 }

/-- Theorem stating the maximum number of pieces that can be cut -/
theorem max_pieces_cut : 
  (area plywood) / (area piece) = 22 := by sorry

end NUMINAMATH_CALUDE_max_pieces_cut_l2480_248063


namespace NUMINAMATH_CALUDE_exponent_addition_l2480_248016

theorem exponent_addition (a : ℝ) (m n : ℕ) : a ^ m * a ^ n = a ^ (m + n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l2480_248016


namespace NUMINAMATH_CALUDE_walking_speed_equation_l2480_248035

theorem walking_speed_equation (x : ℝ) 
  (h1 : x > 0) -- Xiao Wang's speed is positive
  (h2 : x + 1 > 0) -- Xiao Zhang's speed is positive
  : 15 / x - 15 / (x + 1) = 1 / 2 ↔ 
    (15 / x = 15 / (x + 1) + 1 / 2 ∧ 
     15 / (x + 1) < 15 / x) :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_equation_l2480_248035


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l2480_248021

theorem king_middle_school_teachers :
  let total_students : ℕ := 1500
  let classes_per_student : ℕ := 5
  let regular_class_size : ℕ := 30
  let specialized_classes : ℕ := 10
  let specialized_class_size : ℕ := 15
  let classes_per_teacher : ℕ := 3

  let total_class_instances : ℕ := total_students * classes_per_student
  let specialized_class_instances : ℕ := specialized_classes * specialized_class_size
  let regular_class_instances : ℕ := total_class_instances - specialized_class_instances
  let regular_classes : ℕ := regular_class_instances / regular_class_size
  let total_classes : ℕ := regular_classes + specialized_classes
  let number_of_teachers : ℕ := total_classes / classes_per_teacher

  number_of_teachers = 85 := by sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l2480_248021


namespace NUMINAMATH_CALUDE_quadratic_functions_theorem_l2480_248006

/-- A quadratic function -/
def QuadraticFunction := ℝ → ℝ

/-- Condition that a function is quadratic -/
def IsQuadratic (f : QuadraticFunction) : Prop := 
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def VertexX (f : QuadraticFunction) : ℝ := sorry

/-- The x-intercepts of a function -/
def XIntercepts (f : QuadraticFunction) : Set ℝ := sorry

theorem quadratic_functions_theorem 
  (f g : QuadraticFunction)
  (hf : IsQuadratic f)
  (hg : IsQuadratic g)
  (h_relation : ∀ x, g x = -f (75 - x))
  (h_vertex : VertexX f ∈ XIntercepts g)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h_intercepts : {x₁, x₂, x₃, x₄} ⊆ XIntercepts f ∪ XIntercepts g)
  (h_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h_diff : x₃ - x₂ = 120) :
  x₄ - x₁ = 360 + 240 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_functions_theorem_l2480_248006


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2480_248003

/-- Simple interest calculation -/
theorem simple_interest_principal 
  (interest : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 9)
  (h3 : time = 5) : 
  ∃ (principal : ℝ), principal = 8925 ∧ interest = principal * rate * time / 100 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2480_248003


namespace NUMINAMATH_CALUDE_product_inequality_l2480_248084

theorem product_inequality (a b c d : ℝ) 
  (ha : 1 ≤ a ∧ a ≤ 2) 
  (hb : 1 ≤ b ∧ b ≤ 2) 
  (hc : 1 ≤ c ∧ c ≤ 2) 
  (hd : 1 ≤ d ∧ d ≤ 2) : 
  |((a - b) * (b - c) * (c - d) * (d - a))| ≤ (a * b * c * d) / 4 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l2480_248084


namespace NUMINAMATH_CALUDE_max_knight_moves_5x6_l2480_248072

/-- Represents a chess board --/
structure ChessBoard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a knight's move type --/
inductive MoveType
  | Normal
  | Short

/-- Represents a sequence of knight moves --/
def MoveSequence := List MoveType

/-- Checks if a move sequence is valid (alternating between Normal and Short, starting with Normal) --/
def isValidMoveSequence : MoveSequence → Bool
  | [] => true
  | [MoveType.Normal] => true
  | (MoveType.Normal :: MoveType.Short :: rest) => isValidMoveSequence rest
  | _ => false

/-- The maximum number of moves a knight can make on the given board --/
def maxKnightMoves (board : ChessBoard) (seq : MoveSequence) : Nat :=
  seq.length

/-- The main theorem to prove --/
theorem max_knight_moves_5x6 :
  ∀ (seq : MoveSequence),
    isValidMoveSequence seq →
    maxKnightMoves ⟨5, 6⟩ seq ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_knight_moves_5x6_l2480_248072


namespace NUMINAMATH_CALUDE_consecutive_numbers_theorem_l2480_248033

theorem consecutive_numbers_theorem 
  (a b c d e f g : ℤ) 
  (consecutive : b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6)
  (average_9 : (a + b + c + d + e + f + g) / 7 = 9)
  (a_half_of_g : 2 * a = g) : 
  ∃ (n : ℕ), n = 7 ∧ g - a + 1 = n :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_theorem_l2480_248033


namespace NUMINAMATH_CALUDE_division_problem_l2480_248028

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2480_248028


namespace NUMINAMATH_CALUDE_simplify_expression_l2480_248042

theorem simplify_expression :
  ∀ x : ℝ, x > 0 →
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) =
  Real.sqrt 3 + Real.sqrt 4 - Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2480_248042


namespace NUMINAMATH_CALUDE_reward_system_l2480_248066

/-- The number of bowls a customer needs to buy to get rewarded with two bowls -/
def bowls_for_reward : ℕ := sorry

theorem reward_system (total_bowls : ℕ) (customers : ℕ) (buying_customers : ℕ) 
  (bowls_per_customer : ℕ) (remaining_bowls : ℕ) :
  total_bowls = 70 →
  customers = 20 →
  buying_customers = customers / 2 →
  bowls_per_customer = 20 →
  remaining_bowls = 30 →
  bowls_for_reward = 10 := by sorry

end NUMINAMATH_CALUDE_reward_system_l2480_248066


namespace NUMINAMATH_CALUDE_red_ball_probability_l2480_248043

theorem red_ball_probability (x : ℕ) : 
  (8 : ℝ) / (x + 8 : ℝ) = 0.2 → x = 32 := by
sorry

end NUMINAMATH_CALUDE_red_ball_probability_l2480_248043


namespace NUMINAMATH_CALUDE_max_cookies_eaten_l2480_248093

/-- Given 36 cookies shared among three siblings, where one sibling eats twice as many as another,
    and the third eats the same as the second, the maximum number of cookies the second sibling
    could have eaten is 9. -/
theorem max_cookies_eaten (total_cookies : ℕ) (andy bella charlie : ℕ) : 
  total_cookies = 36 →
  bella = 2 * andy →
  charlie = andy →
  total_cookies = andy + bella + charlie →
  andy ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_eaten_l2480_248093


namespace NUMINAMATH_CALUDE_box_volume_l2480_248091

/-- Given a rectangular box with dimensions L, W, and H satisfying certain conditions,
    prove that its volume is 5184. -/
theorem box_volume (L W H : ℝ) (h1 : H * W = 288) (h2 : L * W = 1.5 * 288) 
    (h3 : L * H = 0.5 * (L * W)) : L * W * H = 5184 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2480_248091


namespace NUMINAMATH_CALUDE_max_elevation_is_650_l2480_248069

/-- The elevation function of a ball thrown vertically upward -/
def s (t : ℝ) : ℝ := 100 * t - 4 * t^2 + 25

/-- Theorem: The maximum elevation reached by the ball is 650 feet -/
theorem max_elevation_is_650 : 
  ∃ t_max : ℝ, ∀ t : ℝ, s t ≤ s t_max ∧ s t_max = 650 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_is_650_l2480_248069


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l2480_248062

/-- Represents the possible outcomes of rolling an 8-sided die -/
inductive DieOutcome
  | Prime
  | Composite
  | OddNonPrime
  | Reroll

/-- The probability of each outcome when rolling a fair 8-sided die -/
def outcomeProb (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime => 1/2
  | DieOutcome.Composite => 1/4
  | DieOutcome.OddNonPrime => 1/8
  | DieOutcome.Reroll => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expectedRollsPerDay : ℝ :=
  1

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- Theorem: The expected number of die rolls in a non-leap year
    is equal to the number of days in the year -/
theorem expected_rolls_in_year :
  (expectedRollsPerDay * daysInNonLeapYear : ℝ) = daysInNonLeapYear := by
  sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l2480_248062


namespace NUMINAMATH_CALUDE_max_product_at_endpoints_l2480_248070

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  m : ℤ
  n : ℤ
  f : ℝ → ℝ := λ x ↦ 10 * x^2 + m * x + n

/-- The property that a function has two distinct real roots in (1, 3) -/
def has_two_distinct_roots_in_open_interval (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, 1 < x ∧ x < y ∧ y < 3 ∧ f x = 0 ∧ f y = 0

/-- The theorem statement -/
theorem max_product_at_endpoints (qf : QuadraticFunction) 
  (h : has_two_distinct_roots_in_open_interval qf.f) :
  (qf.f 1) * (qf.f 3) ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_max_product_at_endpoints_l2480_248070


namespace NUMINAMATH_CALUDE_jason_final_pears_l2480_248060

def initial_pears : ℕ := 46
def pears_given_to_keith : ℕ := 47
def pears_received_from_mike : ℕ := 12

theorem jason_final_pears :
  (if initial_pears ≥ pears_given_to_keith
   then initial_pears - pears_given_to_keith
   else 0) + pears_received_from_mike = 12 := by
  sorry

end NUMINAMATH_CALUDE_jason_final_pears_l2480_248060


namespace NUMINAMATH_CALUDE_probability_five_heads_in_six_tosses_l2480_248029

def n : ℕ := 6  -- number of coin tosses
def k : ℕ := 5  -- number of heads we want to get
def p : ℚ := 1/2  -- probability of getting heads on a single toss (fair coin)

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of getting exactly k successes in n trials
def probability_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

-- The theorem to prove
theorem probability_five_heads_in_six_tosses :
  probability_k_successes n k p = 0.09375 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_heads_in_six_tosses_l2480_248029


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l2480_248071

theorem pigeonhole_on_permutation_sums (n : ℕ) (h_even : Even n) (h_pos : 0 < n)
  (A B : Fin n → Fin n) (h_A : Function.Bijective A) (h_B : Function.Bijective B) :
  ∃ (i j : Fin n), i ≠ j ∧ (A i + B i) % n = (A j + B j) % n := by
sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l2480_248071


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2480_248001

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (2, 1) →
  b = (1, m) →
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2480_248001


namespace NUMINAMATH_CALUDE_meters_equivalence_l2480_248030

-- Define the conversion rates
def meters_to_decimeters : ℝ := 10
def meters_to_centimeters : ℝ := 100

-- Define the theorem
theorem meters_equivalence : 
  7.34 = 7 + (3 / meters_to_decimeters) + (4 / meters_to_centimeters) := by
  sorry

end NUMINAMATH_CALUDE_meters_equivalence_l2480_248030


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l2480_248036

/-- The number of ways to choose k items from n distinguishable items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players on the volleyball team -/
def total_players : ℕ := 16

/-- The number of starters to be chosen -/
def num_starters : ℕ := 6

/-- Theorem: The number of ways to choose 6 starters from 16 players is 8008 -/
theorem volleyball_team_starters : choose total_players num_starters = 8008 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l2480_248036


namespace NUMINAMATH_CALUDE_white_shirts_count_l2480_248039

/-- The number of white t-shirts in each pack -/
def white_shirts_per_pack : ℕ := sorry

/-- The number of packs of white t-shirts bought -/
def white_packs : ℕ := 5

/-- The number of packs of blue t-shirts bought -/
def blue_packs : ℕ := 3

/-- The number of blue t-shirts in each pack -/
def blue_shirts_per_pack : ℕ := 9

/-- The total number of t-shirts bought -/
def total_shirts : ℕ := 57

theorem white_shirts_count : white_shirts_per_pack = 6 :=
  by sorry

end NUMINAMATH_CALUDE_white_shirts_count_l2480_248039


namespace NUMINAMATH_CALUDE_non_integer_mean_arrangement_l2480_248079

theorem non_integer_mean_arrangement (N : ℕ) (h : Even N) :
  ∃ (arr : List ℕ),
    (arr.length = N) ∧
    (∀ x, x ∈ arr ↔ 1 ≤ x ∧ x ≤ N) ∧
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ N →
      ¬(∃ (k : ℕ), (arr.take j).sum - (arr.take (i-1)).sum = k * (j - i + 1))) :=
by sorry

end NUMINAMATH_CALUDE_non_integer_mean_arrangement_l2480_248079


namespace NUMINAMATH_CALUDE_total_gifts_received_l2480_248046

def gifts_from_emilio : ℕ := 11
def gifts_from_jorge : ℕ := 6
def gifts_from_pedro : ℕ := 4

theorem total_gifts_received : 
  gifts_from_emilio + gifts_from_jorge + gifts_from_pedro = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_gifts_received_l2480_248046


namespace NUMINAMATH_CALUDE_xy_sum_product_l2480_248068

theorem xy_sum_product (x y : ℝ) (h1 : x + y = 2 * Real.sqrt 3) (h2 : x * y = Real.sqrt 6) :
  x^2 * y + x * y^2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_product_l2480_248068


namespace NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_l2480_248074

/-- An isosceles triangle with a semicircle inscribed -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The diameter of the semicircle is contained in the base of the triangle -/
  diameter_in_base : radius * 2 ≤ base

/-- The theorem stating the radius of the semicircle in the given isosceles triangle -/
theorem semicircle_radius_in_isosceles_triangle 
  (triangle : IsoscelesTriangleWithSemicircle) 
  (h1 : triangle.base = 20) 
  (h2 : triangle.height = 12) : 
  triangle.radius = 60 / (5 + Real.sqrt 61) :=
sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_isosceles_triangle_l2480_248074


namespace NUMINAMATH_CALUDE_partner_b_profit_share_l2480_248004

/-- Calculates the share of profit for partner B given the investment ratios and total profit -/
theorem partner_b_profit_share 
  (invest_a invest_b invest_c : ℚ) 
  (total_profit : ℚ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_b = (2/3) * invest_c)
  (h3 : total_profit = 5500) :
  (invest_b / (invest_a + invest_b + invest_c)) * total_profit = 1000 := by
  sorry

end NUMINAMATH_CALUDE_partner_b_profit_share_l2480_248004


namespace NUMINAMATH_CALUDE_shaded_area_circle_with_inscribed_square_l2480_248025

/-- The area of the shaded region in a circle with radius 2, where the unshaded region forms an inscribed square -/
theorem shaded_area_circle_with_inscribed_square :
  let circle_radius : ℝ := 2
  let circle_area := π * circle_radius^2
  let inscribed_square_side := 2 * circle_radius
  let inscribed_square_area := inscribed_square_side^2
  let unshaded_area := inscribed_square_area / 2
  let shaded_area := circle_area - unshaded_area
  shaded_area = 4 * π - 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_circle_with_inscribed_square_l2480_248025


namespace NUMINAMATH_CALUDE_min_value_expression_l2480_248058

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : x^2 + y^2 = z) :
  ∃ (min : ℝ), min = -2040200 ∧
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = z),
    (a + 1/b) * (a + 1/b - 2020) + (b + 1/a) * (b + 1/a - 2020) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2480_248058


namespace NUMINAMATH_CALUDE_shift_arrangements_count_l2480_248044

def total_volunteers : ℕ := 14
def shifts_per_day : ℕ := 3
def people_per_shift : ℕ := 4

def shift_arrangements : ℕ := (total_volunteers.choose people_per_shift) * 
                               ((total_volunteers - people_per_shift).choose people_per_shift) * 
                               ((total_volunteers - 2 * people_per_shift).choose people_per_shift)

theorem shift_arrangements_count : shift_arrangements = 3153150 := by
  sorry

end NUMINAMATH_CALUDE_shift_arrangements_count_l2480_248044


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2480_248034

/-- The x-intercept of the line 4x + 7y = 28 is the point (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  (4 * x + 7 * y = 28) → (x = 7 ∧ y = 0 → 4 * x + 7 * y = 28) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2480_248034


namespace NUMINAMATH_CALUDE_count_special_numbers_l2480_248024

def is_odd (n : Nat) : Bool := n % 2 = 1

def is_even (n : Nat) : Bool := n % 2 = 0

def digits : List Nat := [1, 2, 3, 4, 5]

def is_valid_number (n : List Nat) : Bool :=
  n.length = 5 ∧ 
  n.toFinset.card = 5 ∧
  n.all (λ d => d ∈ digits) ∧
  (∃ i, i ∈ [1, 2, 3] ∧ 
    is_odd (n.get! i) ∧ 
    is_even (n.get! (i-1)) ∧ 
    is_even (n.get! (i+1)))

theorem count_special_numbers :
  (List.filter is_valid_number (List.permutations digits)).length = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l2480_248024


namespace NUMINAMATH_CALUDE_number_puzzle_l2480_248097

theorem number_puzzle (x : ℝ) : 3 * (2 * x + 9) = 51 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2480_248097


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2480_248096

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) :
  3 * a 9 - a 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2480_248096


namespace NUMINAMATH_CALUDE_count_distinct_lines_l2480_248005

def S : Set ℕ := {0, 1, 2, 3}

def is_valid_line (a b : ℕ) : Prop := a ∈ S ∧ b ∈ S

def distinct_lines : ℕ := sorry

theorem count_distinct_lines :
  distinct_lines = 9 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_lines_l2480_248005


namespace NUMINAMATH_CALUDE_other_number_proof_l2480_248038

theorem other_number_proof (A B : ℕ) (hA : A = 24) (hHCF : Nat.gcd A B = 17) (hLCM : Nat.lcm A B = 312) : B = 221 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2480_248038


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2480_248047

theorem binomial_expansion_coefficient (a : ℝ) : 
  (6 : ℕ) * a^5 * (Real.sqrt 3 / 6) = -Real.sqrt 3 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2480_248047


namespace NUMINAMATH_CALUDE_perfect_square_property_l2480_248078

theorem perfect_square_property (x y z : ℤ) (h : x * y + y * z + z * x = 1) :
  (1 + x^2) * (1 + y^2) * (1 + z^2) = ((x + y) * (y + z) * (x + z))^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l2480_248078


namespace NUMINAMATH_CALUDE_eliana_fuel_cost_l2480_248026

/-- The amount Eliana spent on fuel in a week -/
def fuel_cost (refill_cost : ℕ) (refill_count : ℕ) : ℕ :=
  refill_cost * refill_count

/-- Proof that Eliana spent $63 on fuel this week -/
theorem eliana_fuel_cost :
  fuel_cost 21 3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_eliana_fuel_cost_l2480_248026


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l2480_248017

theorem quadratic_roots_max_value (a b u v : ℝ) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = u ∨ x = v) →
  (u + v = u^2 + v^2) →
  (u + v = u^4 + v^4) →
  (u + v = u^18 + v^18) →
  (∃ (M : ℝ), ∀ (a' b' u' v' : ℝ), 
    (∀ x, x^2 - a'*x + b' = 0 ↔ x = u' ∨ x = v') →
    (u' + v' = u'^2 + v'^2) →
    (u' + v' = u'^4 + v'^4) →
    (u' + v' = u'^18 + v'^18) →
    1/u'^20 + 1/v'^20 ≤ M) →
  1/u^20 + 1/v^20 = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l2480_248017


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l2480_248011

theorem average_of_eleven_numbers
  (first_six_avg : Real)
  (last_six_avg : Real)
  (sixth_number : Real)
  (h1 : first_six_avg = 58)
  (h2 : last_six_avg = 65)
  (h3 : sixth_number = 78) :
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l2480_248011


namespace NUMINAMATH_CALUDE_max_value_expression_l2480_248023

theorem max_value_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) (h5 : x = y) : 
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2480_248023


namespace NUMINAMATH_CALUDE_magician_marbles_left_l2480_248027

/-- Calculates the total number of marbles left after removing some from each color --/
def marblesLeft (initialRed initialBlue initialGreen redTaken : ℕ) : ℕ :=
  let blueTaken := 5 * redTaken
  let greenTaken := blueTaken / 2
  let redLeft := initialRed - redTaken
  let blueLeft := initialBlue - blueTaken
  let greenLeft := initialGreen - greenTaken
  redLeft + blueLeft + greenLeft

/-- Theorem stating that given the initial numbers of marbles and the rules for taking away marbles,
    the total number of marbles left is 93 --/
theorem magician_marbles_left :
  marblesLeft 40 60 35 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_magician_marbles_left_l2480_248027


namespace NUMINAMATH_CALUDE_safari_lions_l2480_248090

theorem safari_lions (safari_lions safari_snakes safari_giraffes savanna_lions savanna_snakes savanna_giraffes : ℕ) :
  safari_snakes = safari_lions / 2 →
  safari_giraffes = safari_snakes - 10 →
  savanna_lions = 2 * safari_lions →
  savanna_snakes = 3 * safari_snakes →
  savanna_giraffes = safari_giraffes + 20 →
  savanna_lions + savanna_snakes + savanna_giraffes = 410 →
  safari_lions = 72 := by
  sorry

end NUMINAMATH_CALUDE_safari_lions_l2480_248090


namespace NUMINAMATH_CALUDE_max_value_theorem_l2480_248067

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2/2 = 1) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 + y'^2/2 = 1 →
    x' * Real.sqrt (1 + y'^2) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2480_248067


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2480_248055

-- Problem 1
theorem problem_1 : (-2023)^0 + Real.sqrt 12 + 2 * (-1/2) = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) : (2*m + 1) * (2*m - 1) - 4*m*(m - 1) = 4*m - 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2480_248055


namespace NUMINAMATH_CALUDE_infinite_logarithm_equation_l2480_248052

theorem infinite_logarithm_equation : ∃! x : ℝ, x > 0 ∧ 2^x = x + 64 := by
  sorry

end NUMINAMATH_CALUDE_infinite_logarithm_equation_l2480_248052


namespace NUMINAMATH_CALUDE_median_equation_equal_intercepts_equation_l2480_248081

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (4, -6)
def C : ℝ × ℝ := (5, 1)

-- Define the equation of a line
def is_line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Theorem for the median equation
theorem median_equation :
  is_line_equation 1 (-2) (-3) (A.1 + (B.1 - A.1)/2) (A.2 + (B.2 - A.2)/2) ∧
  is_line_equation 1 (-2) (-3) C.1 C.2 :=
sorry

-- Theorem for the line with equal intercepts
theorem equal_intercepts_equation :
  is_line_equation 1 1 (-2) A.1 A.2 ∧
  ∃ (t : ℝ), is_line_equation 1 1 (-2) t 0 ∧ is_line_equation 1 1 (-2) 0 t :=
sorry

end NUMINAMATH_CALUDE_median_equation_equal_intercepts_equation_l2480_248081


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l2480_248032

/-- Two arithmetic sequences with integer terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_1764 
  (a b : ℕ → ℤ) 
  (ha : ArithmeticSequence a) 
  (hb : ArithmeticSequence b) 
  (h_a1 : a 1 = 1) 
  (h_b1 : b 1 = 1) 
  (h_a2_le_b2 : a 2 ≤ b 2) 
  (h_product : ∃ n : ℕ, a n * b n = 1764) :
  (∀ m : ℕ, (∃ k : ℕ, a k * b k = 1764) → m ≤ 44) ∧ 
  (∃ n : ℕ, a n * b n = 1764 ∧ n = 44) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l2480_248032


namespace NUMINAMATH_CALUDE_profit_maximum_at_five_l2480_248076

/-- Profit function parameters -/
def a : ℝ := -10
def b : ℝ := 100
def c : ℝ := 2000

/-- Profit function -/
def profit_function (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The point where the maximum profit occurs -/
def max_profit_point : ℝ := 5

theorem profit_maximum_at_five :
  ∀ x : ℝ, profit_function x ≤ profit_function max_profit_point :=
by sorry


end NUMINAMATH_CALUDE_profit_maximum_at_five_l2480_248076


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l2480_248089

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * tree_distance

/-- Theorem: A yard with 26 equally spaced trees, 13 meters apart, is 325 meters long -/
theorem yard_length_26_trees : 
  yard_length 26 13 = 325 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l2480_248089


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l2480_248008

theorem min_product_of_reciprocal_sum (a b : ℕ+) 
  (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (6 : ℚ)⁻¹) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = (6 : ℚ)⁻¹ → a * b ≤ c * d) ∧ a * b = 48 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l2480_248008


namespace NUMINAMATH_CALUDE_waiter_tip_calculation_l2480_248087

theorem waiter_tip_calculation (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℕ) :
  total_customers = 10 →
  non_tipping_customers = 5 →
  total_tips = 15 →
  (total_tips : ℚ) / (total_customers - non_tipping_customers : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_calculation_l2480_248087


namespace NUMINAMATH_CALUDE_pear_percentage_difference_l2480_248018

/-- Proves that the percentage difference between canned and poached pears is 20% -/
theorem pear_percentage_difference (total pears_sold pears_canned pears_poached : ℕ) :
  total = 42 →
  pears_sold = 20 →
  pears_poached = pears_sold / 2 →
  total = pears_sold + pears_canned + pears_poached →
  (pears_canned - pears_poached : ℚ) / pears_poached * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_pear_percentage_difference_l2480_248018


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2480_248031

/-- A hyperbola with eccentricity √6/2 has the equation x²/4 - y²/2 = 1 -/
theorem hyperbola_equation (e : ℝ) (h : e = Real.sqrt 6 / 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x^2 / (a^2) - y^2 / (b^2) = 1 ↔ 
    x^2 / 4 - y^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2480_248031


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2480_248086

def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ∀ n : ℕ, isTwoDigit n → sumOfDigits n = 9 → ¬ Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l2480_248086


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2480_248048

-- Define the complex number z
def z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)

-- Theorem stating that the imaginary part of z is -1
theorem imaginary_part_of_z : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2480_248048


namespace NUMINAMATH_CALUDE_intersection_distance_l2480_248065

/-- The distance between the intersection points of y = x - 3 and x² + 2y² = 8 is 4√3/3 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = A.1 - 3 ∧ A.1^2 + 2*A.2^2 = 8) ∧ 
    (B.2 = B.1 - 3 ∧ B.1^2 + 2*B.2^2 = 8) ∧ 
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (4 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2480_248065


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l2480_248037

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Properties of a rectangular prism -/
axiom rectangular_prism_properties (rp : RectangularPrism) : 
  rp.faces = 6 ∧ rp.edges = 12 ∧ rp.vertices = 8

/-- Theorem: The sum of faces, edges, and vertices of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l2480_248037


namespace NUMINAMATH_CALUDE_jenna_bob_difference_l2480_248056

/-- Prove that Jenna has $20 less than Bob in her account given the conditions. -/
theorem jenna_bob_difference (bob phil jenna : ℕ) : 
  bob = 60 → 
  phil = bob / 3 → 
  jenna = 2 * phil → 
  bob - jenna = 20 := by
sorry

end NUMINAMATH_CALUDE_jenna_bob_difference_l2480_248056


namespace NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l2480_248010

/-- A function that checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

/-- A function that reverses the digits of a number -/
def reverseDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that 10989 is the only five-digit number
    that when multiplied by 9, results in its reverse -/
theorem unique_reverse_multiple_of_nine :
  ∀ n : ℕ, isFiveDigit n → (9 * n = reverseDigits n) → n = 10989 :=
sorry

end NUMINAMATH_CALUDE_unique_reverse_multiple_of_nine_l2480_248010


namespace NUMINAMATH_CALUDE_ball_count_theorem_l2480_248095

theorem ball_count_theorem (total : ℕ) (red_freq black_freq : ℚ) : 
  total = 120 ∧ 
  red_freq = 15/100 ∧ 
  black_freq = 45/100 → 
  total - (red_freq * total).floor - (black_freq * total).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l2480_248095


namespace NUMINAMATH_CALUDE_problem_statement_l2480_248015

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- absolute value of m is 2
  : (a + b) / m - m^2 + 2 * c * d = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2480_248015
