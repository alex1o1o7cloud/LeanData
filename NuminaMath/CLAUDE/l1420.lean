import Mathlib

namespace NUMINAMATH_CALUDE_carmen_initial_cats_l1420_142009

/-- Represents the number of cats Carmen initially had -/
def initial_cats : ℕ := sorry

/-- Represents the number of dogs Carmen has -/
def dogs : ℕ := 18

/-- Represents the number of cats Carmen gave up for adoption -/
def cats_given_up : ℕ := 3

/-- Represents the difference between cats and dogs after giving up cats -/
def cat_dog_difference : ℕ := 7

theorem carmen_initial_cats :
  initial_cats = 28 ∧
  initial_cats - cats_given_up = dogs + cat_dog_difference :=
sorry

end NUMINAMATH_CALUDE_carmen_initial_cats_l1420_142009


namespace NUMINAMATH_CALUDE_min_value_of_function_l1420_142074

theorem min_value_of_function (x : ℝ) (h : x < 0) :
  -x - 2/x ≥ 2 * Real.sqrt 2 ∧
  (-(-Real.sqrt 2) - 2/(-Real.sqrt 2) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1420_142074


namespace NUMINAMATH_CALUDE_right_triangle_square_distance_l1420_142052

/-- Given a right triangle with hypotenuse forming the side of a square outside the triangle,
    and the sum of the legs being d, the distance from the right angle vertex to the center
    of the square is (d * √2) / 2. -/
theorem right_triangle_square_distance (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b = d ∧
    a^2 + b^2 = c^2 ∧
    (d * Real.sqrt 2) / 2 = c * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_square_distance_l1420_142052


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1420_142078

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 6 + 2 * a 4 * a 5 + a 5 ^ 2 = 25 →
  a 4 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1420_142078


namespace NUMINAMATH_CALUDE_ten_person_tournament_matches_l1420_142040

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of matches in a 10-person round-robin tournament is 45 -/
theorem ten_person_tournament_matches :
  num_matches 10 = 45 := by
  sorry

/-- Lemma: The number of matches formula is valid for any number of players -/
lemma num_matches_formula_valid (n : ℕ) (n_ge_2 : n ≥ 2) :
  num_matches n = (n * (n - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ten_person_tournament_matches_l1420_142040


namespace NUMINAMATH_CALUDE_warehouse_notebooks_l1420_142000

/-- The number of notebooks in a warehouse --/
def total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) : ℕ :=
  num_boxes * parts_per_box * notebooks_per_part

/-- Theorem: The total number of notebooks in the warehouse is 660 --/
theorem warehouse_notebooks : 
  total_notebooks 22 6 5 = 660 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_notebooks_l1420_142000


namespace NUMINAMATH_CALUDE_max_value_constrained_expression_l1420_142035

theorem max_value_constrained_expression :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 →
  x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constrained_expression_l1420_142035


namespace NUMINAMATH_CALUDE_parabola_parameter_range_l1420_142043

theorem parabola_parameter_range (a m n : ℝ) : 
  a ≠ 0 → 
  n = a * m^2 - 4 * a^2 * m - 3 →
  0 ≤ m → m ≤ 4 → 
  n ≤ -3 →
  (a ≥ 1 ∨ a < 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_parameter_range_l1420_142043


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1420_142029

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 3 * x^2 + 8 * x - 5) - (x^3 + 6 * x^2 + 2 * x - 15) - (2 * x^3 + x^2 + 4 * x - 8) = 
  -4 * x^2 + 2 * x + 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1420_142029


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1420_142059

theorem intersection_point_sum (a' b' : ℚ) : 
  (2 = (1/3) * 4 + a') ∧ (4 = (1/3) * 2 + b') → a' + b' = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1420_142059


namespace NUMINAMATH_CALUDE_production_reduction_for_breakeven_l1420_142013

/-- The problem setup and proof statement --/
theorem production_reduction_for_breakeven (initial_production : ℕ) 
  (price_per_item : ℚ) (profit : ℚ) (variable_cost_per_item : ℚ) 
  (h1 : initial_production = 4000)
  (h2 : price_per_item = 6250)
  (h3 : profit = 2000000)
  (h4 : variable_cost_per_item = 3750) : 
  let constant_costs := initial_production * price_per_item - profit
  let breakeven_production := constant_costs / (price_per_item - variable_cost_per_item)
  (initial_production - breakeven_production) / initial_production = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_production_reduction_for_breakeven_l1420_142013


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1420_142030

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) : 
  (3 * x - 6 * y = 12) → 
  (∃ (m b : ℝ), ∀ (x' y' : ℝ), y' = m * x' + b ∧ (3 * x' - 6 * y' = 12)) → 
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1420_142030


namespace NUMINAMATH_CALUDE_f_max_property_l1420_142032

def f_properties (f : ℚ → ℚ) : Prop :=
  (f 0 = 0) ∧
  (∀ α : ℚ, α ≠ 0 → f α > 0) ∧
  (∀ α β : ℚ, f (α * β) = f α * f β) ∧
  (∀ α β : ℚ, f (α + β) ≤ f α + f β) ∧
  (∀ m : ℤ, f m ≤ 1989)

theorem f_max_property (f : ℚ → ℚ) (h : f_properties f) :
  ∀ α β : ℚ, f α ≠ f β → f (α + β) = max (f α) (f β) := by
  sorry

end NUMINAMATH_CALUDE_f_max_property_l1420_142032


namespace NUMINAMATH_CALUDE_clothing_distribution_l1420_142051

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l1420_142051


namespace NUMINAMATH_CALUDE_condition_sufficiency_for_increasing_f_l1420_142018

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3

theorem condition_sufficiency_for_increasing_f :
  (∀ x ≥ 2, Monotone (f 1)) ∧
  ¬(∀ a : ℝ, (∀ x ≥ 2, Monotone (f a)) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficiency_for_increasing_f_l1420_142018


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1420_142066

/-- A line passing through point A(1,2) with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point A(1,2) -/
  passes_through_A : m * 1 + b = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b = b / m

/-- The equation of an EqualInterceptLine is either 2x - y = 0 or x + y = 3 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 2 ∧ l.b = 0) ∨ (l.m = 1 ∧ l.b = 2) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1420_142066


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l1420_142006

def complex_equation (z : ℂ) : Prop :=
  (1 - Complex.I) * z = (1 + Complex.I)^2

theorem complex_magnitude_theorem (z : ℂ) :
  complex_equation z → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l1420_142006


namespace NUMINAMATH_CALUDE_trapezoid_balance_l1420_142063

-- Define the shapes and their weights
variable (C P T : ℝ)

-- Define the balance conditions
axiom balance1 : C = 2 * P
axiom balance2 : T = C + P

-- Theorem to prove
theorem trapezoid_balance : T = 3 * P := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_balance_l1420_142063


namespace NUMINAMATH_CALUDE_ounces_in_pound_l1420_142070

/-- Represents the number of ounces in one pound -/
def ounces_per_pound : ℕ := sorry

theorem ounces_in_pound : 
  (2100 : ℕ) * 13 = 1680 * (16 + 4 / ounces_per_pound) → ounces_per_pound = 16 := by
  sorry

end NUMINAMATH_CALUDE_ounces_in_pound_l1420_142070


namespace NUMINAMATH_CALUDE_fermat_numbers_coprime_l1420_142004

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_fermat_numbers_coprime_l1420_142004


namespace NUMINAMATH_CALUDE_correct_calculation_l1420_142036

theorem correct_calculation (a : ℝ) : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1420_142036


namespace NUMINAMATH_CALUDE_trail_mix_weight_l1420_142055

theorem trail_mix_weight (peanuts chocolate_chips raisins : ℝ) 
  (h1 : peanuts = 0.17)
  (h2 : chocolate_chips = 0.17)
  (h3 : raisins = 0.08) :
  peanuts + chocolate_chips + raisins = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l1420_142055


namespace NUMINAMATH_CALUDE_transformation_constructible_l1420_142076

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define the points
variable (A B C A' B' C' : V)

-- Define the transformation
variable (T : V → V)

-- Define the non-collinearity condition
def NonCollinear (A B C : V) : Prop :=
  ¬ ∃ (t : ℝ), C - A = t • (B - A)

-- Define the concept of constructible with straightedge and compass
def Constructible (P P' : V) (T : V → V) : Prop :=
  ∃ (construction : V → V),
    (∀ X Y : V, ∃ line : ℝ → V, line 0 = X ∧ line 1 = Y) ∧
    (∀ X Y Z : V, ∃ W : V, ‖X - W‖ = ‖Y - Z‖) ∧
    construction P = P'

-- State the theorem
theorem transformation_constructible
  (h_non_collinear : NonCollinear A B C)
  (h_transform : T A = A' ∧ T B = B' ∧ T C = C') :
  ∀ P : V, Constructible P (T P) T :=
by sorry

end NUMINAMATH_CALUDE_transformation_constructible_l1420_142076


namespace NUMINAMATH_CALUDE_intersection_of_lines_AB_CD_l1420_142069

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := A
  let b := B
  let c := C
  let d := D
  (20, -18, 11)

/-- Theorem stating that the intersection point of lines AB and CD is (20, -18, 11) --/
theorem intersection_of_lines_AB_CD :
  let A : ℝ × ℝ × ℝ := (8, -6, 5)
  let B : ℝ × ℝ × ℝ := (18, -16, 10)
  let C : ℝ × ℝ × ℝ := (-4, 6, -12)
  let D : ℝ × ℝ × ℝ := (4, -4, 8)
  intersection_point A B C D = (20, -18, 11) := by
  sorry

#check intersection_of_lines_AB_CD

end NUMINAMATH_CALUDE_intersection_of_lines_AB_CD_l1420_142069


namespace NUMINAMATH_CALUDE_car_initial_payment_l1420_142044

/-- Calculates the initial payment for a car purchase given the total cost,
    monthly payment, and number of months. -/
def initial_payment (total_cost monthly_payment num_months : ℕ) : ℕ :=
  total_cost - monthly_payment * num_months

theorem car_initial_payment :
  initial_payment 13380 420 19 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_car_initial_payment_l1420_142044


namespace NUMINAMATH_CALUDE_division_problem_l1420_142008

/-- Given a division problem with quotient, divisor, and remainder, calculate the dividend -/
theorem division_problem (quotient divisor remainder : ℕ) (h1 : quotient = 256) (h2 : divisor = 3892) (h3 : remainder = 354) :
  divisor * quotient + remainder = 996706 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1420_142008


namespace NUMINAMATH_CALUDE_property_implies_linear_l1420_142042

/-- A function f: ℚ → ℚ satisfies the given property if
    f(x) + f(t) = f(y) + f(z) for all rational x < y < z < t
    that form an arithmetic progression -/
def SatisfiesProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ) (d : ℚ), 0 < d → x < y ∧ y < z ∧ z < t →
  y = x + d ∧ z = y + d ∧ t = z + d →
  f x + f t = f y + f z

/-- A function f: ℚ → ℚ is linear if there exist rational m and b
    such that f(x) = mx + b for all rational x -/
def IsLinear (f : ℚ → ℚ) : Prop :=
  ∃ (m b : ℚ), ∀ (x : ℚ), f x = m * x + b

theorem property_implies_linear (f : ℚ → ℚ) :
  SatisfiesProperty f → IsLinear f := by
  sorry

end NUMINAMATH_CALUDE_property_implies_linear_l1420_142042


namespace NUMINAMATH_CALUDE_sin_shift_l1420_142001

theorem sin_shift (x : ℝ) : Real.sin (3 * x - π / 3) = Real.sin (3 * (x - π / 9)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l1420_142001


namespace NUMINAMATH_CALUDE_men_left_bus_l1420_142085

/-- Represents the state of passengers on the bus --/
structure BusState where
  men : ℕ
  women : ℕ

/-- The initial state of the bus --/
def initialState : BusState :=
  { men := 48, women := 24 }

/-- The final state of the bus after some men leave and 8 women enter --/
def finalState : BusState :=
  { men := 32, women := 32 }

/-- The number of women who entered the bus in city Y --/
def womenEntered : ℕ := 8

theorem men_left_bus (initial : BusState) (final : BusState) :
  initial.men + initial.women = 72 →
  initial.women = initial.men / 2 →
  final.men = final.women →
  final.women = initial.women + womenEntered →
  initial.men - final.men = 16 := by
  sorry

#check men_left_bus initialState finalState

end NUMINAMATH_CALUDE_men_left_bus_l1420_142085


namespace NUMINAMATH_CALUDE_perpendicular_planes_l1420_142053

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b c : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h3 : perpendicular a α)
  (h4 : subset b β)
  (h5 : parallel a b) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l1420_142053


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1420_142067

/-- Given a train that crosses a platform and passes a stationary man, calculate its speed -/
theorem train_speed_calculation (platform_length : ℝ) (platform_crossing_time : ℝ) (man_passing_time : ℝ) :
  platform_length = 220 →
  platform_crossing_time = 30 →
  man_passing_time = 19 →
  ∃ (train_speed : ℝ), train_speed = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1420_142067


namespace NUMINAMATH_CALUDE_shortest_rope_length_l1420_142095

theorem shortest_rope_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (ratio : b = (5/4) * a ∧ c = (6/4) * a) (sum_condition : a + c = b + 100) : 
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_shortest_rope_length_l1420_142095


namespace NUMINAMATH_CALUDE_complement_cardinality_l1420_142020

def U : Finset Nat := {1,2,3,4,5,6}
def M : Finset Nat := {2,3,5}
def N : Finset Nat := {4,5}

theorem complement_cardinality : Finset.card (U \ (M ∪ N)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_cardinality_l1420_142020


namespace NUMINAMATH_CALUDE_total_votes_polled_l1420_142022

/-- Represents the total number of votes polled in an election --/
def total_votes : ℕ := sorry

/-- Represents the number of valid votes received by candidate B --/
def votes_B : ℕ := 2509

/-- Theorem stating the total number of votes polled in the election --/
theorem total_votes_polled :
  (total_votes : ℚ) * (80 : ℚ) / 100 = 
    (votes_B : ℚ) + (votes_B : ℚ) + (total_votes : ℚ) * (15 : ℚ) / 100 ∧
  total_votes = 7720 :=
sorry

end NUMINAMATH_CALUDE_total_votes_polled_l1420_142022


namespace NUMINAMATH_CALUDE_fraction_equality_l1420_142028

theorem fraction_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a + c) / (b + d) = a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1420_142028


namespace NUMINAMATH_CALUDE_prime_sum_less_than_ten_l1420_142047

theorem prime_sum_less_than_ten (d e f : ℕ) : 
  Prime d → Prime e → Prime f →
  d < 10 → e < 10 → f < 10 →
  d + e = f →
  d < e →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_less_than_ten_l1420_142047


namespace NUMINAMATH_CALUDE_speed_of_train2_l1420_142007

-- Define the problem parameters
def distance_between_stations : ℝ := 200
def speed_of_train1 : ℝ := 20
def start_time_train1 : ℝ := 7
def start_time_train2 : ℝ := 8
def meeting_time : ℝ := 12

-- Define the theorem
theorem speed_of_train2 (speed_train2 : ℝ) : speed_train2 = 25 := by
  -- Assuming the conditions of the problem
  have h1 : distance_between_stations = 200 := by rfl
  have h2 : speed_of_train1 = 20 := by rfl
  have h3 : start_time_train1 = 7 := by rfl
  have h4 : start_time_train2 = 8 := by rfl
  have h5 : meeting_time = 12 := by rfl

  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_speed_of_train2_l1420_142007


namespace NUMINAMATH_CALUDE_range_of_a_l1420_142056

theorem range_of_a (x a : ℝ) :
  (∀ x, (x - 2) * (x - 3) < 0 → -4 < x - a ∧ x - a < 4) →
  -1 ≤ a ∧ a ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1420_142056


namespace NUMINAMATH_CALUDE_f_derivative_l1420_142016

-- Define the function f
def f (x : ℝ) : ℝ := (2*x - 1) * (x^2 + 3)

-- State the theorem
theorem f_derivative :
  deriv f = fun x => 6*x^2 - 2*x + 6 :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l1420_142016


namespace NUMINAMATH_CALUDE_janes_bagels_l1420_142014

theorem janes_bagels (b m : ℕ) : 
  b + m = 5 →
  (75 * b + 50 * m) % 100 = 0 →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_janes_bagels_l1420_142014


namespace NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l1420_142090

theorem sum_of_distance_and_reciprocal (a b : ℝ) : 
  (|a| = 5 ∧ b = (-1/3)⁻¹) → (a + b = 2 ∨ a + b = -8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_distance_and_reciprocal_l1420_142090


namespace NUMINAMATH_CALUDE_remaining_content_is_two_fifteenths_l1420_142058

/-- The fraction of content remaining after four days of evaporation -/
def remaining_content : ℚ :=
  let day1_remaining := 1 - 2/3
  let day2_remaining := day1_remaining * (1 - 1/4)
  let day3_remaining := day2_remaining * (1 - 1/5)
  let day4_remaining := day3_remaining * (1 - 1/3)
  day4_remaining

/-- Theorem stating that the remaining content after four days is 2/15 -/
theorem remaining_content_is_two_fifteenths :
  remaining_content = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_remaining_content_is_two_fifteenths_l1420_142058


namespace NUMINAMATH_CALUDE_parabolic_arch_bridge_width_l1420_142049

/-- Parabolic arch bridge problem -/
theorem parabolic_arch_bridge_width 
  (a : ℝ) 
  (h1 : a = -8) 
  (h2 : 4^2 = a * (-2)) 
  : let new_y := -3/2
    let new_x := Real.sqrt (a * new_y)
    2 * new_x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabolic_arch_bridge_width_l1420_142049


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1420_142099

theorem arithmetic_equality : 19 * 17 + 29 * 17 + 48 * 25 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1420_142099


namespace NUMINAMATH_CALUDE_paperware_cost_relationship_l1420_142064

/-- Represents the cost of paper plates and cups -/
structure PaperwareCost where
  plate : ℝ
  cup : ℝ

/-- The total cost of a given number of plates and cups -/
def total_cost (c : PaperwareCost) (plates : ℝ) (cups : ℝ) : ℝ :=
  c.plate * plates + c.cup * cups

/-- Theorem stating the relationship between the costs of different quantities of plates and cups -/
theorem paperware_cost_relationship (c : PaperwareCost) :
  total_cost c 20 40 = 1.20 → total_cost c 100 200 = 6.00 := by
  sorry

end NUMINAMATH_CALUDE_paperware_cost_relationship_l1420_142064


namespace NUMINAMATH_CALUDE_larger_number_problem_l1420_142037

theorem larger_number_problem (x y : ℕ) : 
  x + y = 70 ∧ 
  y = 15 ∧ 
  x = 3 * y + 10 → 
  x = 55 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1420_142037


namespace NUMINAMATH_CALUDE_cake_price_l1420_142050

theorem cake_price (num_cakes : ℕ) (num_pies : ℕ) (pie_price : ℚ) (total_revenue : ℚ) : 
  num_cakes = 453 → 
  num_pies = 126 → 
  pie_price = 7 → 
  total_revenue = 6318 → 
  ∃ (cake_price : ℚ), cake_price = 12 ∧ num_cakes * cake_price + num_pies * pie_price = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_cake_price_l1420_142050


namespace NUMINAMATH_CALUDE_smoothie_combinations_l1420_142061

theorem smoothie_combinations : 
  let num_flavors : ℕ := 5
  let num_toppings : ℕ := 8
  let topping_choices : ℕ := 3
  num_flavors * (Nat.choose num_toppings topping_choices) = 280 :=
by sorry

end NUMINAMATH_CALUDE_smoothie_combinations_l1420_142061


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l1420_142081

def is_valid_triangle (t : Finset Nat) : Prop :=
  t.card = 3 ∧ ∀ x ∈ t, 1 ≤ x ∧ x ≤ 9

def sum_of_triangle (t : Finset Nat) : Nat :=
  t.sum id

def valid_sum (s : Nat) : Prop :=
  12 ≤ s ∧ s ≤ 27 ∧ s ≠ 14 ∧ s ≠ 25

theorem triangle_sum_theorem :
  {s : Nat | ∃ t1 t2 : Finset Nat,
    is_valid_triangle t1 ∧
    is_valid_triangle t2 ∧
    t1 ∩ t2 = ∅ ∧
    sum_of_triangle t1 = s ∧
    sum_of_triangle t2 = s ∧
    valid_sum s} =
  {12, 13, 15, 16, 17, 18, 19} :=
sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l1420_142081


namespace NUMINAMATH_CALUDE_lucy_money_problem_l1420_142060

theorem lucy_money_problem (initial_amount : ℝ) : 
  let doubled := 2 * initial_amount
  let after_loss := doubled * (2/3)
  let after_spending := after_loss * (3/4)
  after_spending = 15 → initial_amount = 15 := by
sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l1420_142060


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1420_142048

theorem cyclic_sum_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1420_142048


namespace NUMINAMATH_CALUDE_power_sum_difference_l1420_142098

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) = 58929 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1420_142098


namespace NUMINAMATH_CALUDE_points_symmetric_wrt_origin_l1420_142097

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Point A has coordinates (3, 4) -/
def A : ℝ × ℝ := (3, 4)

/-- Point B has coordinates (-3, -4) -/
def B : ℝ × ℝ := (-3, -4)

theorem points_symmetric_wrt_origin : symmetric_wrt_origin A B := by
  sorry

end NUMINAMATH_CALUDE_points_symmetric_wrt_origin_l1420_142097


namespace NUMINAMATH_CALUDE_opposite_sides_inequality_l1420_142082

/-- Given two points P and A on opposite sides of a line, prove that P satisfies a specific inequality --/
theorem opposite_sides_inequality (x y : ℝ) :
  (3*x + 2*y - 8) * (3*1 + 2*2 - 8) < 0 →
  3*x + 2*y > 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_inequality_l1420_142082


namespace NUMINAMATH_CALUDE_triangle_max_area_l1420_142093

/-- Given a triangle ABC with circumradius 1 and tan(A) / tan(B) = (2c - b) / b, 
    the maximum area of the triangle is 3√3 / 4 -/
theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  Real.tan A / Real.tan B = (2 * c - b) / b →
  (∃ (S : ℝ), S = 1/2 * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = 1/2 * b * c * Real.sin A → S' ≤ 3 * Real.sqrt 3 / 4) :=
by sorry


end NUMINAMATH_CALUDE_triangle_max_area_l1420_142093


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l1420_142071

theorem pulley_centers_distance (r1 r2 contact_distance : ℝ) 
  (h1 : r1 = 12)
  (h2 : r2 = 6)
  (h3 : contact_distance = 30) :
  ∃ (center_distance : ℝ), center_distance = 2 * Real.sqrt 234 := by
sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l1420_142071


namespace NUMINAMATH_CALUDE_f_form_when_a_equals_b_f_max_value_with_three_zeros_l1420_142088

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := (x - a) * (x^2 - (b - 1) * x - b)

-- Theorem 1: When a = b = 1, f(x) = (x-1)^2(x+1)
theorem f_form_when_a_equals_b (x : ℝ) :
  f 1 1 x = (x - 1)^2 * (x + 1) := by sorry

-- Theorem 2: When f(x) = x(x-1)(x+1), the maximum value is 2√3/9
theorem f_max_value_with_three_zeros :
  let g (x : ℝ) := x * (x - 1) * (x + 1)
  ∃ (x_max : ℝ), g x_max = 2 * Real.sqrt 3 / 9 ∧ ∀ (x : ℝ), g x ≤ g x_max := by sorry

end NUMINAMATH_CALUDE_f_form_when_a_equals_b_f_max_value_with_three_zeros_l1420_142088


namespace NUMINAMATH_CALUDE_mark_weekly_reading_pages_l1420_142083

-- Define the initial reading time in hours
def initial_reading_time : ℝ := 2

-- Define the percentage increase in reading time
def reading_time_increase : ℝ := 150

-- Define the initial pages read per day
def initial_pages_per_day : ℝ := 100

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem mark_weekly_reading_pages :
  let new_reading_time := initial_reading_time * (1 + reading_time_increase / 100)
  let new_pages_per_day := initial_pages_per_day * (new_reading_time / initial_reading_time)
  let weekly_pages := new_pages_per_day * days_in_week
  weekly_pages = 1750 := by sorry

end NUMINAMATH_CALUDE_mark_weekly_reading_pages_l1420_142083


namespace NUMINAMATH_CALUDE_largest_two_digit_power_ending_l1420_142092

/-- A number is a two-digit number if it's between 10 and 99, inclusive. -/
def IsTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number satisfies the power condition if all its positive integer powers end with itself modulo 100. -/
def SatisfiesPowerCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → n^k % 100 = n % 100

/-- 76 is the largest two-digit number divisible by 4 that satisfies the power condition. -/
theorem largest_two_digit_power_ending : 
  IsTwoDigit 76 ∧ 
  76 % 4 = 0 ∧ 
  SatisfiesPowerCondition 76 ∧ 
  ∀ n : ℕ, IsTwoDigit n → n % 4 = 0 → SatisfiesPowerCondition n → n ≤ 76 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_power_ending_l1420_142092


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1420_142080

theorem inequality_solution_set (x : ℝ) :
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ∧ x^2 - 6*x + 8 ≥ 0) ↔ 
  (x ∈ Set.Icc (-5) 1 ∪ Set.Icc 5 11) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1420_142080


namespace NUMINAMATH_CALUDE_min_sum_with_exponential_constraint_l1420_142002

theorem min_sum_with_exponential_constraint (a b : ℝ) :
  a > 0 → b > 0 → (2 : ℝ)^a * 4^b = (2^a)^b →
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ)^x * 4^y = (2^x)^y → a + b ≤ x + y) →
  a + b = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_exponential_constraint_l1420_142002


namespace NUMINAMATH_CALUDE_angle_line_plane_l1420_142017

-- Define the line and plane
def line_eq1 (x z : ℝ) : Prop := x - 2*z + 3 = 0
def line_eq2 (y z : ℝ) : Prop := y + 3*z - 1 = 0
def plane_eq (x y z : ℝ) : Prop := 2*x - y + z + 3 = 0

-- Define the angle between the line and plane
def angle_between_line_and_plane : ℝ := sorry

-- State the theorem
theorem angle_line_plane :
  Real.sin angle_between_line_and_plane = 4 * Real.sqrt 21 / 21 :=
sorry

end NUMINAMATH_CALUDE_angle_line_plane_l1420_142017


namespace NUMINAMATH_CALUDE_infinite_solutions_c_equals_six_l1420_142005

theorem infinite_solutions_c_equals_six :
  ∃! c : ℝ, ∀ y : ℝ, 2 * (4 + c * y) = 12 * y + 8 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_c_equals_six_l1420_142005


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l1420_142084

theorem basketball_shot_probability :
  let p1 : ℚ := 2/3  -- Probability of making the first shot
  let p2_success : ℚ := 2/3  -- Probability of making the second shot if the first shot was successful
  let p2_fail : ℚ := 1/3  -- Probability of making the second shot if the first shot failed
  let p3_success : ℚ := 2/3  -- Probability of making the third shot after making the second
  let p3_fail : ℚ := 1/3  -- Probability of making the third shot after missing the second
  
  (p1 * p2_success * p3_success) +  -- Case 1: Make all three shots
  (p1 * (1 - p2_success) * p3_fail) +  -- Case 2: Make first, miss second, make third
  ((1 - p1) * p2_fail * p3_success) +  -- Case 3: Miss first, make second and third
  ((1 - p1) * (1 - p2_fail) * p3_fail) = 14/27  -- Case 4: Miss first and second, make third
  := by sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l1420_142084


namespace NUMINAMATH_CALUDE_water_after_40_days_l1420_142087

/-- Calculates the remaining water in a trough after a given number of days -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem stating that given the initial conditions, the remaining water after 40 days is 270 gallons -/
theorem water_after_40_days :
  let initial_amount : ℝ := 300
  let evaporation_rate : ℝ := 0.75
  let days : ℝ := 40
  remaining_water initial_amount evaporation_rate days = 270 := by
sorry

#eval remaining_water 300 0.75 40

end NUMINAMATH_CALUDE_water_after_40_days_l1420_142087


namespace NUMINAMATH_CALUDE_abs_four_minus_xy_gt_two_abs_x_minus_y_l1420_142079

theorem abs_four_minus_xy_gt_two_abs_x_minus_y 
  (x y : ℝ) (hx : |x| < 2) (hy : |y| < 2) : 
  |4 - x * y| > 2 * |x - y| := by
  sorry

end NUMINAMATH_CALUDE_abs_four_minus_xy_gt_two_abs_x_minus_y_l1420_142079


namespace NUMINAMATH_CALUDE_no_eight_face_polyhedron_from_cube_cut_l1420_142024

/-- Represents a polyhedron --/
structure Polyhedron where
  faces : ℕ

/-- Represents a cube --/
structure Cube where
  faces : ℕ
  faces_eq_six : faces = 6

/-- Represents the result of cutting a cube with a single plane --/
structure CubeCut where
  original : Cube
  piece1 : Polyhedron
  piece2 : Polyhedron
  single_cut : piece1.faces + piece2.faces = original.faces + 2

/-- Theorem stating that a polyhedron with 8 faces cannot be obtained from cutting a cube with a single plane --/
theorem no_eight_face_polyhedron_from_cube_cut (cut : CubeCut) :
  cut.piece1.faces ≠ 8 ∧ cut.piece2.faces ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_eight_face_polyhedron_from_cube_cut_l1420_142024


namespace NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l1420_142041

/-- Two vectors are collinear if their cross product is zero -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a is defined as (1, m-1) -/
def a (m : ℝ) : ℝ × ℝ := (1, m - 1)

/-- Vector b is defined as (m, 2) -/
def b (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Theorem stating that m = 2 is a sufficient but not necessary condition for collinearity -/
theorem m_eq_2_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → are_collinear (a m) (b m)) ∧
  ¬(∀ m : ℝ, are_collinear (a m) (b m) → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l1420_142041


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1420_142023

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ = 2 and a₂ + a₃ = 13,
    prove that a₄ + a₅ + a₆ = 42 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : IsArithmeticSequence a)
    (h_a1 : a 1 = 2)
    (h_a2_a3 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1420_142023


namespace NUMINAMATH_CALUDE_product_range_difference_l1420_142091

theorem product_range_difference (f g : ℝ → ℝ) :
  (∀ x, -3 ≤ f x ∧ f x ≤ 9) →
  (∀ x, -1 ≤ g x ∧ g x ≤ 6) →
  (∃ a b, ∀ x, f x * g x ≤ a ∧ b ≤ f x * g x ∧ a - b = 72) :=
by sorry

end NUMINAMATH_CALUDE_product_range_difference_l1420_142091


namespace NUMINAMATH_CALUDE_total_food_eaten_l1420_142054

/-- The amount of food Ella's dog eats relative to Ella -/
def dog_food_ratio : ℕ := 4

/-- The number of days -/
def days : ℕ := 10

/-- The amount of food Ella eats per day (in pounds) -/
def ella_food_per_day : ℕ := 20

/-- The total amount of food eaten by Ella and her dog (in pounds) -/
def total_food : ℕ := days * ella_food_per_day * (1 + dog_food_ratio)

theorem total_food_eaten :
  total_food = 1000 := by sorry

end NUMINAMATH_CALUDE_total_food_eaten_l1420_142054


namespace NUMINAMATH_CALUDE_a_4_plus_a_5_l1420_142027

def a : ℕ → ℕ
  | n => if n % 2 = 1 then 2 * n + 1 else 2^n

theorem a_4_plus_a_5 : a 4 + a 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_a_4_plus_a_5_l1420_142027


namespace NUMINAMATH_CALUDE_g_f_neg_four_equals_nine_l1420_142011

/-- Given a function f and a function g, prove that g(f(-4)) = 9 
    under certain conditions. -/
theorem g_f_neg_four_equals_nine 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h1 : ∀ x, f x = 3 * x^2 - 7) 
  (h2 : g (f 4) = 9) : 
  g (f (-4)) = 9 := by
sorry

end NUMINAMATH_CALUDE_g_f_neg_four_equals_nine_l1420_142011


namespace NUMINAMATH_CALUDE_floor_plus_self_equals_ten_point_three_l1420_142068

theorem floor_plus_self_equals_ten_point_three (r : ℝ) :
  (⌊r⌋ : ℝ) + r = 10.3 → r = 5.3 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equals_ten_point_three_l1420_142068


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1420_142096

/-- Given three nonconstant geometric sequences with different common ratios,
    if a certain condition holds, then the sum of their common ratios is 1 + 2√2 -/
theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ c₂ c₃ m n o : ℝ) 
  (hm : m ≠ 1) (hn : n ≠ 1) (ho : o ≠ 1)  -- nonconstant sequences
  (hm_ne_n : m ≠ n) (hm_ne_o : m ≠ o) (hn_ne_o : n ≠ o)  -- different ratios
  (ha₂ : a₂ = k * m) (ha₃ : a₃ = k * m^2)  -- first sequence
  (hb₂ : b₂ = k * n) (hb₃ : b₃ = k * n^2)  -- second sequence
  (hc₂ : c₂ = k * o) (hc₃ : c₃ = k * o^2)  -- third sequence
  (heq : a₃ - b₃ + c₃ = 2 * (a₂ - b₂ + c₂))  -- given condition
  : m + n + o = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1420_142096


namespace NUMINAMATH_CALUDE_red_subsequence_2019th_element_l1420_142034

/-- Represents the number of elements in the nth group of the red-colored subsequence -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the last element of the nth group in the red-colored subsequence -/
def last_element_of_group (n : ℕ) : ℕ := n * (2 * n - 1)

/-- Represents the sum of elements in the first n groups of the red-colored subsequence -/
def sum_of_elements (n : ℕ) : ℕ := (1 + group_size n) * n / 2

/-- The group number containing the 2019th element -/
def target_group : ℕ := 45

/-- The position of the 2019th element within its group -/
def position_in_group : ℕ := 83

/-- The theorem stating that the 2019th number in the red-colored subsequence is 3993 -/
theorem red_subsequence_2019th_element : 
  last_element_of_group (target_group - 1) + 1 + (position_in_group - 1) * 2 = 3993 :=
sorry

end NUMINAMATH_CALUDE_red_subsequence_2019th_element_l1420_142034


namespace NUMINAMATH_CALUDE_snack_eaters_count_l1420_142033

def snack_eaters_final (initial_attendees : ℕ) 
  (first_hour_eat_percent : ℚ) 
  (first_hour_not_eat_percent : ℚ)
  (second_hour_undecided_join_percent : ℚ)
  (second_hour_not_eat_join_percent : ℚ)
  (second_hour_newcomers : ℕ)
  (second_hour_newcomers_eat : ℕ)
  (second_hour_leave_percent : ℚ)
  (third_hour_increase_percent : ℚ)
  (third_hour_leave_percent : ℚ)
  (fourth_hour_latecomers : ℕ)
  (fourth_hour_latecomers_eat_percent : ℚ)
  (fourth_hour_workshop_leave : ℕ) : ℕ :=
  sorry

theorem snack_eaters_count : 
  snack_eaters_final 7500 (55/100) (35/100) (20/100) (15/100) 75 50 (40/100) (10/100) (1/2) 150 (60/100) 300 = 1347 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_count_l1420_142033


namespace NUMINAMATH_CALUDE_ratio_problem_l1420_142003

theorem ratio_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : 2 * (a + b) = 3 * (a - b)) :
  a = 5 * b := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1420_142003


namespace NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l1420_142031

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

theorem twelve_factorial_mod_thirteen : factorial 12 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l1420_142031


namespace NUMINAMATH_CALUDE_function_equality_implies_m_value_l1420_142065

-- Define the functions f and g
def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m
def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

-- State the theorem
theorem function_equality_implies_m_value :
  ∀ m : ℚ, 3 * (f m 5) = 2 * (g m 5) → m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_value_l1420_142065


namespace NUMINAMATH_CALUDE_white_balls_count_l1420_142089

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 10 →
  red = 37 →
  purple = 3 →
  prob_not_red_purple = 3/5 →
  ∃ white : ℕ, white = 20 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l1420_142089


namespace NUMINAMATH_CALUDE_pen_cost_l1420_142019

theorem pen_cost (notebook pen case : ℝ) 
  (total_cost : notebook + pen + case = 3.50)
  (pen_triple : pen = 3 * notebook)
  (case_more : case = notebook + 0.50) :
  pen = 1.80 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l1420_142019


namespace NUMINAMATH_CALUDE_mosaic_perimeter_l1420_142046

/-- A mosaic constructed with a regular hexagon, squares, and equilateral triangles. -/
structure Mosaic where
  hexagon_side_length : ℝ
  num_squares : ℕ
  num_triangles : ℕ

/-- The outside perimeter of the mosaic. -/
def outside_perimeter (m : Mosaic) : ℝ :=
  (m.num_squares + m.num_triangles) * m.hexagon_side_length

/-- Theorem stating that the outside perimeter of the specific mosaic is 240 cm. -/
theorem mosaic_perimeter :
  ∀ (m : Mosaic),
    m.hexagon_side_length = 20 →
    m.num_squares = 6 →
    m.num_triangles = 6 →
    outside_perimeter m = 240 := by
  sorry

end NUMINAMATH_CALUDE_mosaic_perimeter_l1420_142046


namespace NUMINAMATH_CALUDE_soap_boxes_in_carton_l1420_142077

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the theoretical maximum number of smaller boxes that can fit in a larger box -/
def maxBoxes (large : BoxDimensions) (small : BoxDimensions) : ℕ :=
  (boxVolume large) / (boxVolume small)

/-- Theorem: The maximum number of soap boxes that can theoretically fit in the carton is 150 -/
theorem soap_boxes_in_carton :
  let carton := BoxDimensions.mk 25 42 60
  let soapBox := BoxDimensions.mk 7 12 5
  maxBoxes carton soapBox = 150 := by
  sorry

end NUMINAMATH_CALUDE_soap_boxes_in_carton_l1420_142077


namespace NUMINAMATH_CALUDE_random_events_count_l1420_142086

/-- Represents an event --/
inductive Event
| ClassPresident
| StrongerTeamWins
| BirthdayProblem
| SetInclusion
| PainterDeath
| JulySnow
| EvenSum
| RedLights

/-- Determines if an event is random --/
def isRandomEvent : Event → Bool
| Event.ClassPresident => true
| Event.StrongerTeamWins => true
| Event.BirthdayProblem => true
| Event.SetInclusion => false
| Event.PainterDeath => false
| Event.JulySnow => true
| Event.EvenSum => false
| Event.RedLights => true

/-- List of all events --/
def allEvents : List Event := [
  Event.ClassPresident,
  Event.StrongerTeamWins,
  Event.BirthdayProblem,
  Event.SetInclusion,
  Event.PainterDeath,
  Event.JulySnow,
  Event.EvenSum,
  Event.RedLights
]

/-- Theorem: The number of random events in the list is 5 --/
theorem random_events_count :
  (allEvents.filter isRandomEvent).length = 5 := by sorry

end NUMINAMATH_CALUDE_random_events_count_l1420_142086


namespace NUMINAMATH_CALUDE_proposition_equivalences_l1420_142057

-- Define opposite numbers
def opposite (x y : ℝ) : Prop := x = -y

-- Define having real roots for a quadratic equation
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem proposition_equivalences :
  -- Converse of "If x+y=0, then x and y are opposite numbers"
  (∀ x y : ℝ, opposite x y → x + y = 0) ∧
  -- Contrapositive of "If q ≤ 1, then x^2+2x+q=0 has real roots"
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) ∧
  -- Existence of α and β satisfying the trigonometric equation
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalences_l1420_142057


namespace NUMINAMATH_CALUDE_eventually_single_digit_or_zero_l1420_142075

/-- Function to calculate the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let digits := Nat.digits 10 n
  digits.foldl (·*·) 1

/-- Predicate to check if a number is single-digit or zero -/
def isSingleDigitOrZero (n : ℕ) : Prop :=
  n < 10

/-- Theorem stating that repeatedly applying digitProduct will eventually
    result in a single-digit number or zero -/
theorem eventually_single_digit_or_zero (n : ℕ) :
  ∃ k : ℕ, isSingleDigitOrZero ((digitProduct^[k]) n) :=
sorry


end NUMINAMATH_CALUDE_eventually_single_digit_or_zero_l1420_142075


namespace NUMINAMATH_CALUDE_system_solution_l1420_142012

theorem system_solution : 
  ∀ x y : ℝ, (x^3 + 3*x*y^2 = 49 ∧ x^2 + 8*x*y + y^2 = 8*y + 17*x) → 
  ((x = 1 ∧ y = 4) ∨ (x = 1 ∧ y = -4)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1420_142012


namespace NUMINAMATH_CALUDE_parallelogram_cyclic_equidistant_implies_bisector_l1420_142072

-- Define the necessary structures and functions
structure Point := (x y : ℝ)

def Line := Point → Point → Prop

def parallelogram (A B C D : Point) : Prop := sorry

def cyclic_quadrilateral (B C E D : Point) : Prop := sorry

def intersects_interior (l : Line) (A B : Point) (F : Point) : Prop := sorry

def intersects (l : Line) (A B : Point) (G : Point) : Prop := sorry

def distance (P Q : Point) : ℝ := sorry

def angle_bisector (l : Line) (A B C : Point) : Prop := sorry

-- State the theorem
theorem parallelogram_cyclic_equidistant_implies_bisector
  (A B C D E F G : Point) (ℓ : Line) :
  parallelogram A B C D →
  cyclic_quadrilateral B C E D →
  ℓ A F →
  ℓ A G →
  intersects_interior ℓ D C F →
  intersects ℓ B C G →
  distance E F = distance E G →
  distance E F = distance E C →
  angle_bisector ℓ D A B :=
sorry

end NUMINAMATH_CALUDE_parallelogram_cyclic_equidistant_implies_bisector_l1420_142072


namespace NUMINAMATH_CALUDE_wall_width_calculation_l1420_142094

/-- Given a wall with specific proportions and volume, calculate its width -/
theorem wall_width_calculation (w h l : ℝ) (V : ℝ) (h_def : h = 6 * w) (l_def : l = 7 * h^2) (V_def : V = w * h * l) (V_val : V = 86436) :
  w = (86436 / 1512) ^ (1/4) := by
sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l1420_142094


namespace NUMINAMATH_CALUDE_time_per_question_l1420_142073

/-- Proves that given a test with 100 questions, where 40 questions are left unanswered
    and 2 hours are spent answering, the time taken for each answered question is 2 minutes. -/
theorem time_per_question (total_questions : Nat) (unanswered_questions : Nat) (time_spent : Nat) :
  total_questions = 100 →
  unanswered_questions = 40 →
  time_spent = 120 →
  (time_spent : ℚ) / ((total_questions - unanswered_questions) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_per_question_l1420_142073


namespace NUMINAMATH_CALUDE_second_train_length_l1420_142025

/-- Calculates the length of the second train given the parameters of two trains passing each other. -/
theorem second_train_length
  (first_train_length : ℝ)
  (first_train_speed : ℝ)
  (second_train_speed : ℝ)
  (initial_distance : ℝ)
  (crossing_time : ℝ)
  (h1 : first_train_length = 100)
  (h2 : first_train_speed = 10)
  (h3 : second_train_speed = 15)
  (h4 : initial_distance = 50)
  (h5 : crossing_time = 60)
  : ∃ (second_train_length : ℝ),
    second_train_length = 150 ∧
    second_train_length + first_train_length + initial_distance =
      (second_train_speed - first_train_speed) * crossing_time :=
by
  sorry


end NUMINAMATH_CALUDE_second_train_length_l1420_142025


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l1420_142062

/-- Given a square divided into 4 identical rectangles, each with a perimeter of 20,
    the area of the square is 1600/9. -/
theorem square_area_from_rectangle_perimeter :
  ∀ (s : ℝ), s > 0 →
  (∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = 20 ∧ 2 * l = s ∧ 2 * w = s) →
  s^2 = 1600 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l1420_142062


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1420_142045

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := by
  sorry

#check quadratic_no_real_roots

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1420_142045


namespace NUMINAMATH_CALUDE_base_number_proof_l1420_142015

theorem base_number_proof (e : ℕ) (x : ℕ) : 
  e = x^19 ∧ e % 10 = 7 → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_base_number_proof_l1420_142015


namespace NUMINAMATH_CALUDE_correct_matching_probability_l1420_142010

/-- The number of celebrities and baby pictures --/
def n : ℕ := 3

/-- The total number of possible arrangements --/
def total_arrangements : ℕ := n.factorial

/-- The number of correct arrangements --/
def correct_arrangements : ℕ := 1

/-- The probability of correctly matching all celebrities to their baby pictures --/
def probability : ℚ := correct_arrangements / total_arrangements

theorem correct_matching_probability :
  probability = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l1420_142010


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1420_142038

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2*x - 1) + 2
  f (1/2) = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1420_142038


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1420_142026

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1420_142026


namespace NUMINAMATH_CALUDE_ticket_cost_calculation_l1420_142021

/-- The total cost of tickets for various events -/
def total_cost (movie_price : ℚ) (football_price : ℚ) (concert_price : ℚ) (theater_price : ℚ) : ℚ :=
  8 * movie_price + 5 * football_price + 3 * concert_price + 4 * theater_price

/-- The theorem stating the total cost of tickets -/
theorem ticket_cost_calculation : ∃ (movie_price football_price concert_price theater_price : ℚ),
  (8 * movie_price = 2 * football_price) ∧
  (movie_price = 30) ∧
  (concert_price = football_price - 10) ∧
  (theater_price = 40 * (1 - 0.1)) ∧
  (total_cost movie_price football_price concert_price theater_price = 1314) :=
by
  sorry


end NUMINAMATH_CALUDE_ticket_cost_calculation_l1420_142021


namespace NUMINAMATH_CALUDE_log_domain_condition_l1420_142039

theorem log_domain_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x^2 + 2*x + a) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_log_domain_condition_l1420_142039
