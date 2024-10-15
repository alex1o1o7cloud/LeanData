import Mathlib

namespace NUMINAMATH_CALUDE_range_of_f_l1189_118955

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = 4 ∧
  (∀ y, (∃ x, x ∈ [0, 3] ∧ f x = y) ↔ y ∈ [a, b]) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1189_118955


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1189_118973

def number : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor number) = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l1189_118973


namespace NUMINAMATH_CALUDE_pizza_combinations_six_toppings_l1189_118922

/-- The number of different one- and two-topping pizzas that can be ordered from a pizza parlor with a given number of toppings. -/
def pizza_combinations (n : ℕ) : ℕ :=
  n + n * (n - 1) / 2

/-- Theorem: The number of different one- and two-topping pizzas that can be ordered from a pizza parlor with 6 toppings is equal to 21. -/
theorem pizza_combinations_six_toppings :
  pizza_combinations 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_six_toppings_l1189_118922


namespace NUMINAMATH_CALUDE_function_value_at_2010_l1189_118902

def positive_reals : Set ℝ := {x : ℝ | x > 0}

def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 3)

theorem function_value_at_2010 (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ positive_reals, f x > 0)
  (h2 : function_property f) :
  f 2010 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2010_l1189_118902


namespace NUMINAMATH_CALUDE_chess_tournament_l1189_118942

theorem chess_tournament (n : ℕ) (k : ℚ) : n > 2 →
  (8 : ℚ) + n * k = (n + 2) * (n + 1) / 2 →
  (∀ m : ℕ, m > 2 → (8 : ℚ) + m * k ≠ (m + 2) * (m + 1) / 2 → m ≠ n) →
  n = 7 ∨ n = 14 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_l1189_118942


namespace NUMINAMATH_CALUDE_range_of_x_l1189_118958

-- Define the sets
def S1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def S2 : Set ℝ := {x | x < 1 ∨ x > 4}

-- Define the condition
def condition (x : ℝ) : Prop := ¬(x ∈ S1 ∨ x ∈ S2)

-- State the theorem
theorem range_of_x : 
  ∀ x : ℝ, condition x → x ∈ {y : ℝ | 1 ≤ y ∧ y < 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1189_118958


namespace NUMINAMATH_CALUDE_function_properties_l1189_118966

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 2*b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b (-1)) ∧
    (f a b (-1) = 2) ∧
    (a = 2 ∧ b = 1) ∧
    (∀ x ∈ Set.Icc (-1) 1, f a b x ≤ 6) ∧
    (∀ x ∈ Set.Icc (-1) 1, f a b x ≥ 50/27) ∧
    (∃ x ∈ Set.Icc (-1) 1, f a b x = 6) ∧
    (∃ x ∈ Set.Icc (-1) 1, f a b x = 50/27) :=
by
  sorry


end NUMINAMATH_CALUDE_function_properties_l1189_118966


namespace NUMINAMATH_CALUDE_length_of_BC_l1189_118914

-- Define the triangles and their properties
def triangle_ABC (AB AC BC : ℝ) : Prop :=
  AB^2 + AC^2 = BC^2 ∧ AB > 0 ∧ AC > 0 ∧ BC > 0

def triangle_ABD (AB AD BD : ℝ) : Prop :=
  AB^2 + AD^2 = BD^2 ∧ AB > 0 ∧ AD > 0 ∧ BD > 0

-- State the theorem
theorem length_of_BC :
  ∀ AB AC BC AD BD,
  triangle_ABC AB AC BC →
  triangle_ABD AB AD BD →
  AB = 12 →
  AC = 16 →
  AD = 30 →
  BC = 20 :=
sorry

end NUMINAMATH_CALUDE_length_of_BC_l1189_118914


namespace NUMINAMATH_CALUDE_art_club_participation_l1189_118983

theorem art_club_participation (total : ℕ) (painting : ℕ) (sculpting : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : painting = 80)
  (h3 : sculpting = 60)
  (h4 : both = 20) :
  total - (painting + sculpting - both) = 30 := by
  sorry

end NUMINAMATH_CALUDE_art_club_participation_l1189_118983


namespace NUMINAMATH_CALUDE_haley_tv_watching_l1189_118930

/-- Haley's TV watching problem -/
theorem haley_tv_watching (total_hours sunday_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : sunday_hours = 3) :
  total_hours - sunday_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_haley_tv_watching_l1189_118930


namespace NUMINAMATH_CALUDE_distance_rides_to_car_l1189_118969

/-- The distance Heather walked from the car to the entrance -/
def distance_car_to_entrance : ℝ := 0.3333333333333333

/-- The distance Heather walked from the entrance to the carnival rides -/
def distance_entrance_to_rides : ℝ := 0.3333333333333333

/-- The total distance Heather walked -/
def total_distance : ℝ := 0.75

/-- The theorem states that given the above distances, 
    the distance Heather walked from the carnival rides back to the car 
    is 0.08333333333333337 miles -/
theorem distance_rides_to_car : 
  total_distance - (distance_car_to_entrance + distance_entrance_to_rides) = 0.08333333333333337 := by
  sorry

end NUMINAMATH_CALUDE_distance_rides_to_car_l1189_118969


namespace NUMINAMATH_CALUDE_vector_triangle_inequality_l1189_118940

variable {V : Type*} [NormedAddCommGroup V]

theorem vector_triangle_inequality (a b : V) : ‖a + b‖ ≤ ‖a‖ + ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_inequality_l1189_118940


namespace NUMINAMATH_CALUDE_expression_evaluation_l1189_118953

theorem expression_evaluation : (20 - 16) * (12 + 8) / 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1189_118953


namespace NUMINAMATH_CALUDE_multiplication_puzzle_solution_l1189_118998

theorem multiplication_puzzle_solution : 
  (78346 * 346 = 235038) ∧ (9374 * 82 = 768668) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_solution_l1189_118998


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1189_118993

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + b + 5*I = 9 + a*I → b = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1189_118993


namespace NUMINAMATH_CALUDE_hidden_dots_count_l1189_118984

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of dots on all faces of a standard die -/
def sumOfDots : ℕ := 21

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 3

/-- The visible faces on the stack of dice -/
def visibleFaces : List ℕ := [1, 3, 4, 5, 6]

/-- The total number of faces on the stack of dice -/
def totalFaces : ℕ := 18

/-- The number of hidden faces on the stack of dice -/
def hiddenFaces : ℕ := 13

/-- Theorem stating that the total number of hidden dots is 44 -/
theorem hidden_dots_count :
  (numberOfDice * sumOfDots) - (visibleFaces.sum) = 44 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l1189_118984


namespace NUMINAMATH_CALUDE_stratified_sampling_total_l1189_118964

theorem stratified_sampling_total (sample_size : ℕ) (model_a_count : ℕ) (total_model_b : ℕ) :
  sample_size = 80 →
  model_a_count = 50 →
  total_model_b = 1800 →
  (sample_size - model_a_count) * 60 = total_model_b →
  sample_size * 60 = 4800 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_total_l1189_118964


namespace NUMINAMATH_CALUDE_expression_minimizes_q_l1189_118971

/-- The function q in terms of x and the expression to be determined -/
def q (x : ℝ) (expression : ℝ → ℝ) : ℝ :=
  (expression x)^2 + (x + 1)^2 - 6

/-- The condition that y is least when x = 2 -/
axiom y_min_at_2 : ∀ (y : ℝ → ℝ), ∀ (x : ℝ), y 2 ≤ y x

/-- The relationship between q and y -/
axiom q_related_to_y : ∃ (y : ℝ → ℝ), ∀ (x : ℝ), q x (λ t => t - 2) = y x

/-- The theorem stating that (x - 2) minimizes q when x = 2 -/
theorem expression_minimizes_q :
  ∀ (x : ℝ), q 2 (λ t => t - 2) ≤ q x (λ t => t - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_minimizes_q_l1189_118971


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1189_118963

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: If 3n minus the number of diagonals equals 6, then n equals 6 -/
theorem polygon_sides_count (n : ℕ) (h : 3 * n - num_diagonals n = 6) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1189_118963


namespace NUMINAMATH_CALUDE_card_shop_problem_l1189_118978

theorem card_shop_problem :
  ∃ (x y : ℕ), 1.25 * (x : ℝ) + 1.75 * (y : ℝ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_card_shop_problem_l1189_118978


namespace NUMINAMATH_CALUDE_number_equation_solution_l1189_118951

theorem number_equation_solution :
  ∃ N : ℝ, N - (1002 / 20.04) = 2450 ∧ N = 2500 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1189_118951


namespace NUMINAMATH_CALUDE_mitch_weekly_earnings_is_118_80_l1189_118945

/-- Mitch's weekly earnings after expenses and taxes -/
def mitchWeeklyEarnings : ℝ :=
  let monToWedEarnings := 3 * 5 * 3
  let thuFriEarnings := 2 * 6 * 4
  let satEarnings := 4 * 6
  let sunEarnings := 5 * 8
  let totalEarnings := monToWedEarnings + thuFriEarnings + satEarnings + sunEarnings
  let afterExpenses := totalEarnings - 25
  let taxAmount := afterExpenses * 0.1
  afterExpenses - taxAmount

/-- Theorem stating that Mitch's weekly earnings after expenses and taxes is $118.80 -/
theorem mitch_weekly_earnings_is_118_80 :
  mitchWeeklyEarnings = 118.80 := by sorry

end NUMINAMATH_CALUDE_mitch_weekly_earnings_is_118_80_l1189_118945


namespace NUMINAMATH_CALUDE_quadratic_function_max_min_difference_l1189_118977

-- Define the function f(x) = x^2 + bx + c
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the theorem
theorem quadratic_function_max_min_difference (b c : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → f b c x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ f b c x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → min ≤ f b c x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ f b c x = min) ∧
    max - min = 25) →
  b = -4 ∨ b = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_min_difference_l1189_118977


namespace NUMINAMATH_CALUDE_triangle_third_side_l1189_118970

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 8 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c ≠ 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l1189_118970


namespace NUMINAMATH_CALUDE_closest_point_l1189_118916

/-- The curve y = 3 - x^2 for x > 0 -/
def curve (x : ℝ) : ℝ := 3 - x^2

/-- The fixed point P(0, 2) -/
def P : ℝ × ℝ := (0, 2)

/-- A point Q on the curve -/
def Q (x : ℝ) : ℝ × ℝ := (x, curve x)

/-- The squared distance between P and Q -/
def distance_squared (x : ℝ) : ℝ := (x - P.1)^2 + (curve x - P.2)^2

/-- The theorem stating that (√2/2, 5/2) is the closest point to P on the curve -/
theorem closest_point :
  ∃ (x : ℝ), x > 0 ∧ 
  ∀ (y : ℝ), y > 0 → distance_squared x ≤ distance_squared y ∧
  Q x = (Real.sqrt 2 / 2, 5 / 2) :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1189_118916


namespace NUMINAMATH_CALUDE_employee_pay_theorem_l1189_118903

def employee_pay (total : ℚ) (x_ratio : ℚ) (z_ratio : ℚ) :
  (ℚ × ℚ × ℚ) :=
  let y := total / (1 + x_ratio + z_ratio)
  let x := x_ratio * y
  let z := z_ratio * y
  (x, y, z)

theorem employee_pay_theorem (total : ℚ) (x_ratio : ℚ) (z_ratio : ℚ) :
  let (x, y, z) := employee_pay total x_ratio z_ratio
  (x + y + z = total) ∧ (x = x_ratio * y) ∧ (z = z_ratio * y) :=
by sorry

#eval employee_pay 934 1.2 0.8

end NUMINAMATH_CALUDE_employee_pay_theorem_l1189_118903


namespace NUMINAMATH_CALUDE_total_bills_count_l1189_118901

/-- Represents the number of bills and their total value -/
structure WalletContents where
  num_five_dollar_bills : ℕ
  num_ten_dollar_bills : ℕ
  total_value : ℕ

/-- Theorem stating that given the conditions, the total number of bills is 12 -/
theorem total_bills_count (w : WalletContents) 
  (h1 : w.num_five_dollar_bills = 4)
  (h2 : w.total_value = 100)
  (h3 : w.total_value = 5 * w.num_five_dollar_bills + 10 * w.num_ten_dollar_bills) :
  w.num_five_dollar_bills + w.num_ten_dollar_bills = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_bills_count_l1189_118901


namespace NUMINAMATH_CALUDE_guessing_game_solution_l1189_118939

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem guessing_game_solution :
  ∃! n : ℕ,
    1 ≤ n ∧ n ≤ 99 ∧
    is_perfect_square n ∧
    ¬(n < 5) ∧
    (n < 7 ∨ n < 10 ∨ n ≥ 100) ∧
    n = 9 :=
by sorry

end NUMINAMATH_CALUDE_guessing_game_solution_l1189_118939


namespace NUMINAMATH_CALUDE_a_bounds_l1189_118934

theorem a_bounds (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3) 
  (sum_squares_eq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_bounds_l1189_118934


namespace NUMINAMATH_CALUDE_tangent_segment_length_l1189_118941

-- Define the circle and points
variable (circle : Type) (A B C P Q R : ℝ × ℝ)

-- Define the properties of tangents and points
def is_tangent (point : ℝ × ℝ) (touch_point : ℝ × ℝ) : Prop := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem tangent_segment_length 
  (h1 : is_tangent A B)
  (h2 : is_tangent A C)
  (h3 : is_tangent P Q)
  (h4 : is_tangent R Q)
  (h5 : distance P B = distance P R)
  (h6 : distance A B = 24) :
  distance P Q = 12 := by sorry

end NUMINAMATH_CALUDE_tangent_segment_length_l1189_118941


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l1189_118927

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l1189_118927


namespace NUMINAMATH_CALUDE_box_comparison_l1189_118918

-- Define a structure for a box with three dimensions
structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the comparison operation for boxes
def Box.lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∧
  (a.x < b.x ∨ a.y < b.y ∨ a.z < b.z)

-- Define boxes A, B, and C
def A : Box := ⟨6, 5, 3⟩
def B : Box := ⟨5, 4, 1⟩
def C : Box := ⟨3, 2, 2⟩

-- Theorem to prove A > B and C < A
theorem box_comparison :
  (Box.lt B A) ∧ (Box.lt C A) := by
  sorry

end NUMINAMATH_CALUDE_box_comparison_l1189_118918


namespace NUMINAMATH_CALUDE_total_unique_polygons_l1189_118909

/-- Represents a regular polyhedron --/
inductive RegularPolyhedron
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Returns the number of unique non-planar polygons for a given regular polyhedron --/
def num_unique_polygons (p : RegularPolyhedron) : Nat :=
  match p with
  | .Tetrahedron => 1
  | .Cube => 1
  | .Octahedron => 3
  | .Dodecahedron => 2
  | .Icosahedron => 3

/-- The list of all regular polyhedra --/
def all_polyhedra : List RegularPolyhedron :=
  [RegularPolyhedron.Tetrahedron, RegularPolyhedron.Cube, RegularPolyhedron.Octahedron,
   RegularPolyhedron.Dodecahedron, RegularPolyhedron.Icosahedron]

/-- Theorem stating that the total number of unique non-planar polygons for all regular polyhedra is 10 --/
theorem total_unique_polygons :
  (all_polyhedra.map num_unique_polygons).sum = 10 := by
  sorry

#eval (all_polyhedra.map num_unique_polygons).sum

end NUMINAMATH_CALUDE_total_unique_polygons_l1189_118909


namespace NUMINAMATH_CALUDE_max_value_of_objective_function_l1189_118967

def objective_function (x₁ x₂ : ℝ) : ℝ := 4 * x₁ + 6 * x₂

def feasible_region (x₁ x₂ : ℝ) : Prop :=
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ + x₂ ≤ 18 ∧ 0.5 * x₁ + x₂ ≤ 12 ∧ 2 * x₁ ≤ 24 ∧ 2 * x₂ ≤ 18

theorem max_value_of_objective_function :
  ∃ (x₁ x₂ : ℝ), feasible_region x₁ x₂ ∧
    ∀ (y₁ y₂ : ℝ), feasible_region y₁ y₂ →
      objective_function x₁ x₂ ≥ objective_function y₁ y₂ ∧
      objective_function x₁ x₂ = 84 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_objective_function_l1189_118967


namespace NUMINAMATH_CALUDE_volume_63_ounces_l1189_118972

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assertion that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_63_ounces (s : Substance) 
  (h : volume s 112 = 48) : volume s 63 = 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_63_ounces_l1189_118972


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l1189_118974

theorem binomial_coefficient_20_10 (h1 : Nat.choose 17 7 = 19448)
                                   (h2 : Nat.choose 17 8 = 24310)
                                   (h3 : Nat.choose 17 9 = 24310) :
  Nat.choose 20 10 = 111826 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l1189_118974


namespace NUMINAMATH_CALUDE_volunteers_assignment_l1189_118957

/-- The number of ways to assign volunteers to service points -/
def assign_volunteers (n_volunteers : ℕ) (n_points : ℕ) : ℕ :=
  sorry

/-- Theorem stating that assigning 4 volunteers to 3 service points results in 36 ways -/
theorem volunteers_assignment :
  assign_volunteers 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_volunteers_assignment_l1189_118957


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_difference_squares_l1189_118962

theorem polynomial_coefficient_sum_difference_squares (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + Real.sqrt 3) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_difference_squares_l1189_118962


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l1189_118936

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_english = 70)
  (h2 : failed_both = 10)
  (h3 : passed_both = 20) :
  ∃ failed_hindi : ℝ, failed_hindi = 20 ∧ 
    passed_both + (failed_hindi + failed_english - failed_both) = 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l1189_118936


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1189_118996

theorem greatest_divisor_with_remainders : Nat.gcd (1557 - 7) (2037 - 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1189_118996


namespace NUMINAMATH_CALUDE_museum_time_per_student_l1189_118999

theorem museum_time_per_student 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (time_per_group : ℕ) 
  (h1 : total_students = 18)
  (h2 : num_groups = 3)
  (h3 : time_per_group = 24)
  (h4 : total_students % num_groups = 0) -- Ensures equal division
  : (time_per_group * num_groups) / total_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_museum_time_per_student_l1189_118999


namespace NUMINAMATH_CALUDE_inequality_proof_l1189_118992

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^4 + y^2*z^2)/(x^(5/2)*(y+z)) + 
  (y^4 + z^2*x^2)/(y^(5/2)*(z+x)) + 
  (z^4 + y^2*x^2)/(z^(5/2)*(y+x)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1189_118992


namespace NUMINAMATH_CALUDE_range_of_a_l1189_118968

theorem range_of_a (p q : Prop) (h_p : p ↔ ∀ x ∈ Set.Icc (1/2) 1, 1/x - a ≥ 0)
  (h_q : q ↔ ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) (h_pq : p ∧ q) :
  a ∈ Set.Iic (-2) ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1189_118968


namespace NUMINAMATH_CALUDE_base_7_even_digits_403_l1189_118995

/-- Counts the number of even digits in a base-7 number -/
def countEvenDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 -/
def toBase7 (n : ℕ) : List ℕ := sorry

theorem base_7_even_digits_403 :
  let base7Repr := toBase7 403
  countEvenDigitsBase7 403 = 1 := by sorry

end NUMINAMATH_CALUDE_base_7_even_digits_403_l1189_118995


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1189_118906

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 3 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ (x y : ℝ),
  hyperbola x y →
  (∃ (a b c : ℝ),
    a = 2 * Real.sqrt 3 ∧
    c = 3 ∧
    b^2 = a^2 - c^2 ∧
    ellipse x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l1189_118906


namespace NUMINAMATH_CALUDE_vector_on_line_l1189_118961

/-- Given distinct vectors a and b in a real vector space, 
    prove that the vector (1/4)*a + (3/4)*b lies on the line passing through a and b. -/
theorem vector_on_line (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/4 : ℝ) • a + (3/4 : ℝ) • b = a + t • (b - a) := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_l1189_118961


namespace NUMINAMATH_CALUDE_carrie_profit_calculation_l1189_118910

/-- Calculates Carrie's profit from making a wedding cake --/
theorem carrie_profit_calculation :
  let weekday_hours : ℕ := 5 * 4
  let weekend_hours : ℕ := 3 * 4
  let weekday_rate : ℚ := 35
  let weekend_rate : ℚ := 45
  let supply_cost : ℚ := 180
  let supply_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.07

  let total_earnings : ℚ := weekday_hours * weekday_rate + weekend_hours * weekend_rate
  let discounted_supply_cost : ℚ := supply_cost * (1 - supply_discount)
  let sales_tax : ℚ := total_earnings * sales_tax_rate
  let profit : ℚ := total_earnings - discounted_supply_cost - sales_tax

  profit = 991.20 := by sorry

end NUMINAMATH_CALUDE_carrie_profit_calculation_l1189_118910


namespace NUMINAMATH_CALUDE_three_parallel_lines_theorem_l1189_118959

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Checks if three lines are coplanar -/
def are_coplanar (l1 l2 l3 : Line3D) : Prop := sorry

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- The number of planes determined by three lines -/
def planes_from_lines (l1 l2 l3 : Line3D) : ℕ := sorry

/-- The number of parts the space is divided into by these planes -/
def space_divisions (planes : ℕ) : ℕ := sorry

theorem three_parallel_lines_theorem (a b c : Line3D) 
  (h_parallel_ab : are_parallel a b)
  (h_parallel_bc : are_parallel b c)
  (h_parallel_ac : are_parallel a c)
  (h_not_coplanar : ¬ are_coplanar a b c) :
  planes_from_lines a b c = 3 ∧ space_divisions (planes_from_lines a b c) = 7 := by
  sorry

end NUMINAMATH_CALUDE_three_parallel_lines_theorem_l1189_118959


namespace NUMINAMATH_CALUDE_sequence_matches_l1189_118905

/-- The sequence defined by a_n = 2^n - 1 -/
def a (n : ℕ) : ℕ := 2^n - 1

/-- The first four terms of the sequence match 1, 3, 7, 15 -/
theorem sequence_matches : 
  (a 1 = 1) ∧ (a 2 = 3) ∧ (a 3 = 7) ∧ (a 4 = 15) := by
  sorry

#eval a 1  -- Expected: 1
#eval a 2  -- Expected: 3
#eval a 3  -- Expected: 7
#eval a 4  -- Expected: 15

end NUMINAMATH_CALUDE_sequence_matches_l1189_118905


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1189_118982

/-- A linear function passing through (-4, 3) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 5

/-- The point (-4, 3) lies on the graph of f -/
def point_condition (a : ℝ) : Prop := f a (-4) = 3

/-- The equation ax - 5 = 3 -/
def equation (a x : ℝ) : Prop := a * x - 5 = 3

theorem linear_equation_solution (a : ℝ) (h : point_condition a) :
  ∃ x, equation a x ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1189_118982


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1189_118919

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- State the theorem
theorem set_intersection_theorem :
  (A ∩ B = {x | 0 < x ∧ x ≤ 2}) ∧
  (A ∩ (U \ B) = {x | x > 2}) := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1189_118919


namespace NUMINAMATH_CALUDE_snack_pack_distribution_l1189_118911

theorem snack_pack_distribution (pretzels : ℕ) (suckers : ℕ) (kids : ℕ) :
  pretzels = 64 →
  suckers = 32 →
  kids = 16 →
  (pretzels + 4 * pretzels + suckers) / kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_snack_pack_distribution_l1189_118911


namespace NUMINAMATH_CALUDE_bicycle_car_speed_problem_l1189_118947

theorem bicycle_car_speed_problem (distance : ℝ) (delay : ℝ) 
  (h_distance : distance = 10) 
  (h_delay : delay = 1/3) : 
  ∃ (x : ℝ), x > 0 ∧ distance / x = distance / (2 * x) + delay → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_car_speed_problem_l1189_118947


namespace NUMINAMATH_CALUDE_regular_square_pyramid_side_edge_l1189_118988

theorem regular_square_pyramid_side_edge 
  (base_edge : ℝ) 
  (volume : ℝ) 
  (h : base_edge = 4 * Real.sqrt 2) 
  (h' : volume = 32) : 
  ∃ (side_edge : ℝ), side_edge = 5 := by
sorry

end NUMINAMATH_CALUDE_regular_square_pyramid_side_edge_l1189_118988


namespace NUMINAMATH_CALUDE_emma_share_l1189_118937

theorem emma_share (total : ℕ) (ratio_daniel ratio_emma ratio_fiona : ℕ) (h1 : total = 153) (h2 : ratio_daniel = 3) (h3 : ratio_emma = 5) (h4 : ratio_fiona = 9) : 
  (ratio_emma * total) / (ratio_daniel + ratio_emma + ratio_fiona) = 45 := by
sorry

end NUMINAMATH_CALUDE_emma_share_l1189_118937


namespace NUMINAMATH_CALUDE_percentage_problem_l1189_118997

theorem percentage_problem (P : ℝ) : 
  (20 / 100) * 680 = (P / 100) * 140 + 80 → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1189_118997


namespace NUMINAMATH_CALUDE_town_shoppers_count_l1189_118926

/-- Represents the shopping scenario in the town. -/
structure ShoppingScenario where
  stores : Nat
  total_visits : Nat
  double_visitors : Nat
  max_visits_per_person : Nat

/-- The specific shopping scenario described in the problem. -/
def town_scenario : ShoppingScenario :=
  { stores := 8
  , total_visits := 22
  , double_visitors := 8
  , max_visits_per_person := 3 }

/-- The number of people who went shopping given a shopping scenario. -/
def shoppers (s : ShoppingScenario) : Nat :=
  s.double_visitors + (s.total_visits - 2 * s.double_visitors) / s.max_visits_per_person

/-- Theorem stating that the number of shoppers in the town scenario is 10. -/
theorem town_shoppers_count :
  shoppers town_scenario = 10 := by
  sorry

end NUMINAMATH_CALUDE_town_shoppers_count_l1189_118926


namespace NUMINAMATH_CALUDE_rent_increase_is_thirty_percent_l1189_118985

/-- Calculates the percentage increase in rent given last year's expenses and this year's total increase --/
def rent_increase_percentage (last_year_rent : ℕ) (last_year_food : ℕ) (last_year_insurance : ℕ) (food_increase_percent : ℕ) (insurance_multiplier : ℕ) (total_yearly_increase : ℕ) : ℕ :=
  let last_year_monthly_total := last_year_rent + last_year_food + last_year_insurance
  let this_year_food := last_year_food + (last_year_food * food_increase_percent) / 100
  let this_year_insurance := last_year_insurance * insurance_multiplier
  let monthly_increase_without_rent := (this_year_food + this_year_insurance) - (last_year_food + last_year_insurance)
  let yearly_increase_without_rent := monthly_increase_without_rent * 12
  let rent_increase := total_yearly_increase - yearly_increase_without_rent
  (rent_increase * 100) / (last_year_rent * 12)

theorem rent_increase_is_thirty_percent :
  rent_increase_percentage 1000 200 100 50 3 7200 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_is_thirty_percent_l1189_118985


namespace NUMINAMATH_CALUDE_negation_of_universal_non_negative_square_l1189_118956

theorem negation_of_universal_non_negative_square (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_non_negative_square_l1189_118956


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1189_118949

theorem quadratic_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 1 = 0 ∧ x₂^2 + m*x₂ - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1189_118949


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1189_118912

/-- A function f: ℝ → ℝ is quadratic if it can be expressed as f(x) = ax² + bx + c for some real constants a, b, and c, where a ≠ 0. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = (x-1)(x-2) -/
def f (x : ℝ) : ℝ := (x - 1) * (x - 2)

/-- Theorem: The function f(x) = (x-1)(x-2) is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f :=
sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1189_118912


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1189_118923

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 →
  b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 →
  c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 →
  d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 →
  (1/a + 1/b + 1/c + 1/d)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1189_118923


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l1189_118981

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2*n + 1) = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l1189_118981


namespace NUMINAMATH_CALUDE_seventy_eighth_ball_is_green_l1189_118931

def ball_color (n : ℕ) : String :=
  match n % 5 with
  | 0 => "violet"
  | 1 => "red"
  | 2 => "yellow"
  | 3 => "green"
  | 4 => "blue"
  | _ => "invalid"  -- This case should never occur

theorem seventy_eighth_ball_is_green : ball_color 78 = "green" := by
  sorry

end NUMINAMATH_CALUDE_seventy_eighth_ball_is_green_l1189_118931


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1189_118900

theorem fraction_equivalence : 
  let x : ℚ := 13/2
  (4 + x) / (7 + x) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1189_118900


namespace NUMINAMATH_CALUDE_multiplier_is_three_l1189_118965

theorem multiplier_is_three (n : ℝ) (h1 : 3 * n = (20 - n) + 20) (h2 : n = 10) : 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_three_l1189_118965


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l1189_118950

/-- The cost of soap for a year given the duration of a bar, its cost, and months in a year -/
def soap_cost_for_year (months_per_bar : ℚ) (cost_per_bar : ℚ) (months_in_year : ℕ) : ℚ :=
  (months_in_year / months_per_bar) * cost_per_bar

/-- Theorem: The cost of soap for a year is $48.00 given the specified conditions -/
theorem soap_cost_theorem :
  soap_cost_for_year 2 8 12 = 48 :=
by sorry

end NUMINAMATH_CALUDE_soap_cost_theorem_l1189_118950


namespace NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l1189_118917

theorem angle_measure_with_special_supplement_complement : 
  ∀ x : ℝ, 
    (0 < x) ∧ (x < 90) →
    (180 - x = 7 * (90 - x)) → 
    x = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l1189_118917


namespace NUMINAMATH_CALUDE_min_value_theorem_l1189_118954

/-- The minimum value of a function given specific conditions -/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hmn : m * n > 0) :
  let f := fun x => a^(x - 1) + 1
  let line := fun x y => 2 * m * x + n * y - 4 = 0
  ∃ (x y : ℝ), f x = y ∧ line x y →
  (4 / m + 2 / n : ℝ) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1189_118954


namespace NUMINAMATH_CALUDE_array_sum_theorem_l1189_118987

-- Define the array structure
def array_sum (p : ℕ) : ℚ := 3 * p^2 / ((3*p - 1) * (p - 1))

-- Define the result of (m+n) mod 2009
def result_mod_2009 (p : ℕ) : ℕ :=
  let m : ℕ := 3 * p^2
  let n : ℕ := (3*p - 1) * (p - 1)
  (m + n) % 2009

-- The main theorem
theorem array_sum_theorem :
  array_sum 2008 = 3 * 2008^2 / ((3*2008 - 1) * (2008 - 1)) ∧
  result_mod_2009 2008 = 1 := by sorry

end NUMINAMATH_CALUDE_array_sum_theorem_l1189_118987


namespace NUMINAMATH_CALUDE_difference_of_squares_l1189_118943

theorem difference_of_squares (a : ℝ) : a^2 - 100 = (a + 10) * (a - 10) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1189_118943


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l1189_118913

theorem power_mod_thirteen : 6^4032 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l1189_118913


namespace NUMINAMATH_CALUDE_alex_exam_result_l1189_118976

/-- Represents the scoring system and result of a multiple-choice exam -/
structure ExamResult where
  total_questions : ℕ
  correct_points : ℕ
  blank_points : ℕ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that for the given exam conditions, the maximum number of correct answers is 38 -/
theorem alex_exam_result :
  let exam : ExamResult :=
    { total_questions := 60
      correct_points := 5
      blank_points := 0
      incorrect_points := -2
      total_score := 150 }
  max_correct_answers exam = 38 := by
  sorry

end NUMINAMATH_CALUDE_alex_exam_result_l1189_118976


namespace NUMINAMATH_CALUDE_expression_values_l1189_118952

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a^2 - b*c = b^2 - a*c) (h2 : b^2 - a*c = c^2 - a*b) :
  (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b) = 7/2) ∨
  (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b) = -7) := by
  sorry

#check expression_values

end NUMINAMATH_CALUDE_expression_values_l1189_118952


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_parallel_l1189_118928

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the "contained in" relation
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_parallel
  (m n : Line) (α : Plane)
  (h1 : perpLinePlane m α)
  (h2 : perpLine n m)
  (h3 : ¬ containedIn n α) :
  parallel n α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_parallel_l1189_118928


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1189_118948

/-- An isosceles triangle with a median dividing its perimeter -/
structure IsoscelesTriangleWithMedian where
  /-- Length of each leg of the isosceles triangle -/
  leg : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : leg > 0
  /-- The median on one leg divides the perimeter into parts of 6cm and 12cm -/
  medianDivision : leg + leg / 2 = 12 ∧ leg / 2 + base = 6
  /-- Triangle inequality -/
  triangleInequality : 2 * leg > base ∧ base > 0

/-- Theorem: The base of the isosceles triangle is 2cm -/
theorem isosceles_triangle_base_length
  (t : IsoscelesTriangleWithMedian) : t.base = 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1189_118948


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1189_118933

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 12)
  (h2 : technicians = 6)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 9000 :=
by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l1189_118933


namespace NUMINAMATH_CALUDE_intersection_P_Q_l1189_118980

def P : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x > 4 ∨ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l1189_118980


namespace NUMINAMATH_CALUDE_election_result_proof_l1189_118925

theorem election_result_proof (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 5500 →
  candidate_percentage = 35 / 100 →
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -1650 := by
  sorry

end NUMINAMATH_CALUDE_election_result_proof_l1189_118925


namespace NUMINAMATH_CALUDE_angle_with_special_supplementary_complementary_relation_l1189_118938

theorem angle_with_special_supplementary_complementary_relation :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 90 →
  (180 - x = 3 * (90 - x)) →
  x = 45 :=
by sorry

end NUMINAMATH_CALUDE_angle_with_special_supplementary_complementary_relation_l1189_118938


namespace NUMINAMATH_CALUDE_sqrt_of_four_l1189_118960

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem stating that the square root of 4 is ±2
theorem sqrt_of_four : sqrt 4 = {2, -2} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l1189_118960


namespace NUMINAMATH_CALUDE_import_value_calculation_l1189_118907

theorem import_value_calculation (tax_free_limit : ℝ) (tax_rate : ℝ) (tax_paid : ℝ) : 
  tax_free_limit = 500 →
  tax_rate = 0.08 →
  tax_paid = 18.40 →
  ∃ total_value : ℝ, total_value = 730 ∧ tax_paid = tax_rate * (total_value - tax_free_limit) :=
by sorry

end NUMINAMATH_CALUDE_import_value_calculation_l1189_118907


namespace NUMINAMATH_CALUDE_triangle_properties_l1189_118920

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  2 * c * Real.cos B = 2 * a - b →
  -- Prove C = π/3
  C = π / 3 ∧
  -- When c = 3, prove a + b is in the range (3, 6]
  (c = 3 → 3 < a + b ∧ a + b ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1189_118920


namespace NUMINAMATH_CALUDE_f_greater_than_one_range_l1189_118908

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_range :
  {x : ℝ | f x > 1} = {x : ℝ | x > 1 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_range_l1189_118908


namespace NUMINAMATH_CALUDE_inequality_proof_l1189_118904

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1189_118904


namespace NUMINAMATH_CALUDE_rogers_final_money_rogers_final_money_proof_l1189_118994

/-- Calculates Roger's final amount of money after various transactions -/
theorem rogers_final_money (initial_amount : ℝ) (birthday_money : ℝ) (found_money : ℝ) 
  (game_cost : ℝ) (gift_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_amount + birthday_money + found_money
  let after_game_purchase := total_before_spending - game_cost
  let gift_cost := gift_percentage * after_game_purchase
  let final_amount := after_game_purchase - gift_cost
  final_amount

/-- Proves that Roger's final amount of money is $106.25 -/
theorem rogers_final_money_proof :
  rogers_final_money 84 56 20 35 0.15 = 106.25 := by
  sorry

end NUMINAMATH_CALUDE_rogers_final_money_rogers_final_money_proof_l1189_118994


namespace NUMINAMATH_CALUDE_total_players_is_77_l1189_118990

/-- The number of cricket players -/
def cricket_players : ℕ := 22

/-- The number of hockey players -/
def hockey_players : ℕ := 15

/-- The number of football players -/
def football_players : ℕ := 21

/-- The number of softball players -/
def softball_players : ℕ := 19

/-- Theorem stating that the total number of players is 77 -/
theorem total_players_is_77 : 
  cricket_players + hockey_players + football_players + softball_players = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_players_is_77_l1189_118990


namespace NUMINAMATH_CALUDE_barbara_candies_l1189_118986

/-- Calculates the remaining number of candies Barbara has -/
def remaining_candies (initial : ℝ) (used : ℝ) (received : ℝ) (eaten : ℝ) : ℝ :=
  initial - used + received - eaten

/-- Proves that Barbara has 18.4 candies left -/
theorem barbara_candies : remaining_candies 18.5 4.2 6.8 2.7 = 18.4 := by
  sorry

end NUMINAMATH_CALUDE_barbara_candies_l1189_118986


namespace NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_not_q_l1189_118946

theorem not_p_neither_sufficient_nor_necessary_for_not_q : ∃ (a : ℝ),
  (¬(a > 0) ∧ ¬(a^2 > a)) ∨ (¬(a > 0) ∧ (a^2 > a)) ∨ ((a > 0) ∧ ¬(a^2 > a)) :=
by sorry

end NUMINAMATH_CALUDE_not_p_neither_sufficient_nor_necessary_for_not_q_l1189_118946


namespace NUMINAMATH_CALUDE_fraction_value_l1189_118989

theorem fraction_value (x y : ℝ) (h : 2 * x = -y) :
  x * y / (x^2 - y^2) = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_value_l1189_118989


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l1189_118991

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for (∁ₐA) ∩ B
theorem intersection_complement_A_B : (Aᶜ) ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l1189_118991


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1189_118944

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1189_118944


namespace NUMINAMATH_CALUDE_ranch_problem_l1189_118921

theorem ranch_problem (ponies horses : ℕ) (horseshoe_fraction : ℚ) :
  ponies + horses = 163 →
  horses = ponies + 3 →
  ∃ (iceland_ponies : ℕ), iceland_ponies = (5 : ℚ) / 8 * horseshoe_fraction * ponies →
  horseshoe_fraction = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_ranch_problem_l1189_118921


namespace NUMINAMATH_CALUDE_max_value_A_l1189_118975

theorem max_value_A (x y z : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  (Real.sqrt (8 * x^4 + y) + Real.sqrt (8 * y^4 + z) + Real.sqrt (8 * z^4 + x) - 3) / (x + y + z) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_A_l1189_118975


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1189_118932

theorem decimal_to_fraction : 
  ∃ (n d : ℕ), n / d = (38 : ℚ) / 100 ∧ gcd n d = 1 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1189_118932


namespace NUMINAMATH_CALUDE_rug_design_inner_length_l1189_118924

theorem rug_design_inner_length : 
  ∀ (y : ℝ), 
  let inner_area := 2 * y
  let middle_area := 6 * y + 24
  let outer_area := 10 * y + 80
  (middle_area - inner_area = outer_area - middle_area) →
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_rug_design_inner_length_l1189_118924


namespace NUMINAMATH_CALUDE_farmer_plot_allocation_l1189_118915

theorem farmer_plot_allocation (x y : ℕ) (h : x ≠ y) :
  ∃ (a b : ℕ), a^2 + b^2 = 2 * (x^2 + y^2) :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_plot_allocation_l1189_118915


namespace NUMINAMATH_CALUDE_distance_is_134_l1189_118935

/-- The distance between two girls walking in opposite directions after 12 hours -/
def distance_between_girls : ℝ :=
  let girl1_speed1 : ℝ := 7
  let girl1_time1 : ℝ := 6
  let girl1_speed2 : ℝ := 10
  let girl1_time2 : ℝ := 6
  let girl2_speed1 : ℝ := 3
  let girl2_time1 : ℝ := 8
  let girl2_speed2 : ℝ := 2
  let girl2_time2 : ℝ := 4
  let girl1_distance : ℝ := girl1_speed1 * girl1_time1 + girl1_speed2 * girl1_time2
  let girl2_distance : ℝ := girl2_speed1 * girl2_time1 + girl2_speed2 * girl2_time2
  girl1_distance + girl2_distance

/-- Theorem stating that the distance between the girls after 12 hours is 134 km -/
theorem distance_is_134 : distance_between_girls = 134 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_134_l1189_118935


namespace NUMINAMATH_CALUDE_sin_cos_relation_in_triangle_l1189_118979

theorem sin_cos_relation_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A < Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_relation_in_triangle_l1189_118979


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_squares_and_divisibility_l1189_118929

theorem unique_prime_sum_of_squares_and_divisibility (p m n : ℤ) : 
  Prime p → 
  p = m^2 + n^2 → 
  p ∣ m^3 + n^3 - 4 → 
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_squares_and_divisibility_l1189_118929
