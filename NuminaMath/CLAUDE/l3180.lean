import Mathlib

namespace NUMINAMATH_CALUDE_min_books_borrowed_l3180_318049

theorem min_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat)
  (avg_books : Rat) (max_books : Nat) :
  total_students = 32 →
  zero_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books = 2 →
  max_books = 11 →
  ∃ (min_books : Nat),
    min_books = 4 ∧
    min_books ≤ max_books ∧
    (total_students - zero_books - one_book - two_books) * min_books +
    one_book * 1 + two_books * 2 =
    (total_students : Rat) * avg_books := by
  sorry

end NUMINAMATH_CALUDE_min_books_borrowed_l3180_318049


namespace NUMINAMATH_CALUDE_jason_nickels_l3180_318075

theorem jason_nickels : 
  ∀ (n q : ℕ), 
    n = q + 10 → 
    5 * n + 10 * q = 680 → 
    n = 52 := by
  sorry

end NUMINAMATH_CALUDE_jason_nickels_l3180_318075


namespace NUMINAMATH_CALUDE_mixture_division_l3180_318001

/-- Converts pounds to ounces -/
def pounds_to_ounces (pounds : ℚ) : ℚ := pounds * 16

/-- Calculates the amount of mixture in each container -/
def mixture_per_container (total_weight : ℚ) (num_containers : ℕ) : ℚ :=
  (pounds_to_ounces total_weight) / num_containers

theorem mixture_division (total_weight : ℚ) (num_containers : ℕ) 
  (h1 : total_weight = 57 + 3/8) 
  (h2 : num_containers = 7) :
  ∃ (ε : ℚ), abs (mixture_per_container total_weight num_containers - 131.14) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_mixture_division_l3180_318001


namespace NUMINAMATH_CALUDE_total_distance_driven_l3180_318000

/-- Proves that driving at 55 mph for 2 hours and then 3 hours results in a total distance of 275 miles -/
theorem total_distance_driven (speed : ℝ) (time_before_lunch : ℝ) (time_after_lunch : ℝ) 
  (h1 : speed = 55)
  (h2 : time_before_lunch = 2)
  (h3 : time_after_lunch = 3) :
  speed * time_before_lunch + speed * time_after_lunch = 275 := by
  sorry

#check total_distance_driven

end NUMINAMATH_CALUDE_total_distance_driven_l3180_318000


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l3180_318021

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 1 ∧ y = -Real.sqrt 3 →
  ρ = 2 ∧ θ = 5 * Real.pi / 3 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l3180_318021


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l3180_318089

theorem minimum_value_of_function (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^4 / (a^3 + b^2 + c^2)) + (b^4 / (b^3 + a^2 + c^2)) + (c^4 / (c^3 + b^2 + a^2)) ≥ 1/7 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l3180_318089


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3180_318098

theorem expression_value_at_three :
  let x : ℝ := 3
  x^5 - (5*x)^2 = 18 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3180_318098


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_11_l3180_318011

theorem three_digit_divisible_by_11 (x y z : ℕ) (A : ℕ) : 
  (100 ≤ A) ∧ (A < 1000) ∧ 
  (A = 100 * x + 10 * y + z) ∧ 
  (x + z = y) → 
  ∃ k : ℕ, A = 11 * k := by
sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_11_l3180_318011


namespace NUMINAMATH_CALUDE_equation_solutions_l3180_318013

theorem equation_solutions : 
  ∀ x : ℝ, 2*x - 6 = 3*x*(x - 3) ↔ x = 3 ∨ x = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3180_318013


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l3180_318082

theorem square_perimeter_ratio (area1 area2 perimeter1 perimeter2 : ℝ) :
  area1 > 0 ∧ area2 > 0 →
  area1 / area2 = 49 / 64 →
  perimeter1 / perimeter2 = 7 / 8 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l3180_318082


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l3180_318033

theorem unique_solution_to_equation : 
  ∃! (x y : ℕ+), (x.val : ℝ)^6 * (y.val : ℝ)^6 - 19 * (x.val : ℝ)^3 * (y.val : ℝ)^3 + 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l3180_318033


namespace NUMINAMATH_CALUDE_perfect_squares_as_sum_of_powers_of_two_l3180_318088

theorem perfect_squares_as_sum_of_powers_of_two (n a b : ℕ) (h1 : a ≥ b) (h2 : n^2 = 2^a + 2^b) :
  ∃ k : ℕ, n^2 = 4^(k+1) ∨ n^2 = 9 * 4^k :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_as_sum_of_powers_of_two_l3180_318088


namespace NUMINAMATH_CALUDE_translated_sine_function_l3180_318023

/-- Given a function f and its right-translated version g, prove that g has the expected form. -/
theorem translated_sine_function (f g : ℝ → ℝ) (h : ℝ → ℝ → Prop) : 
  (∀ x, f x = 2 * Real.sin (2 * x + 2 * Real.pi / 3)) →
  (∀ x, h x (g x) ↔ h (x - Real.pi / 6) (f x)) →
  (∀ x, g x = 2 * Real.sin (2 * x + Real.pi / 3)) := by
  sorry


end NUMINAMATH_CALUDE_translated_sine_function_l3180_318023


namespace NUMINAMATH_CALUDE_tangent_parallel_to_xy_l3180_318045

-- Define the function f(x) = x^2 - x
def f (x : ℝ) : ℝ := x^2 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem tangent_parallel_to_xy (P : ℝ × ℝ) :
  (P.1 = 1 ∧ P.2 = 0) ↔
  (f' P.1 = 1 ∧ P.2 = f P.1) := by
  sorry

#check tangent_parallel_to_xy

end NUMINAMATH_CALUDE_tangent_parallel_to_xy_l3180_318045


namespace NUMINAMATH_CALUDE_equation_solution_l3180_318026

theorem equation_solution (x : ℝ) : x * (3 * x + 6) = 7 * (3 * x + 6) ↔ x = 7 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3180_318026


namespace NUMINAMATH_CALUDE_max_product_of_distances_l3180_318073

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_product_of_distances (P : ℝ × ℝ) (h : is_on_ellipse P.1 P.2) :
  ∃ (max : ℝ), max = 8 ∧ ∀ Q : ℝ × ℝ, is_on_ellipse Q.1 Q.2 →
    distance Q F1 * distance Q F2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_product_of_distances_l3180_318073


namespace NUMINAMATH_CALUDE_carries_payment_l3180_318072

def clothes_shopping (shirt_quantity : ℕ) (pants_quantity : ℕ) (jacket_quantity : ℕ)
                     (shirt_price : ℕ) (pants_price : ℕ) (jacket_price : ℕ) : ℕ :=
  let total_cost := shirt_quantity * shirt_price + pants_quantity * pants_price + jacket_quantity * jacket_price
  total_cost / 2

theorem carries_payment :
  clothes_shopping 4 2 2 8 18 60 = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_carries_payment_l3180_318072


namespace NUMINAMATH_CALUDE_book_pages_count_l3180_318097

theorem book_pages_count : 
  let days : ℕ := 10
  let first_four_days_avg : ℕ := 20
  let first_four_days_count : ℕ := 4
  let break_day_count : ℕ := 1
  let next_four_days_avg : ℕ := 30
  let next_four_days_count : ℕ := 4
  let last_day_pages : ℕ := 15
  (first_four_days_avg * first_four_days_count) + 
  (next_four_days_avg * next_four_days_count) + 
  last_day_pages = 215 := by
sorry

end NUMINAMATH_CALUDE_book_pages_count_l3180_318097


namespace NUMINAMATH_CALUDE_function_symmetry_l3180_318090

def symmetric_about (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (2 * p.1 - x) = 2 * p.2 - y

theorem function_symmetry (f : ℝ → ℝ) 
    (h1 : symmetric_about f (-1, 0))
    (h2 : ∀ x > 0, f x = 1 / x) :
    ∀ x < -2, f x = 1 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3180_318090


namespace NUMINAMATH_CALUDE_apple_price_calculation_l3180_318005

/-- Calculates the price of each apple given the produce inventory and total worth -/
theorem apple_price_calculation (asparagus_bundles : ℕ) (asparagus_price : ℚ)
  (grape_boxes : ℕ) (grape_price : ℚ) (apple_count : ℕ) (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  grape_price = 5/2 →
  apple_count = 700 →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + grape_boxes * grape_price)) / apple_count = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_calculation_l3180_318005


namespace NUMINAMATH_CALUDE_always_true_inequality_l3180_318092

theorem always_true_inequality (a b x : ℝ) (h : a > b) : a * (2 : ℝ)^x > b * (2 : ℝ)^x := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequality_l3180_318092


namespace NUMINAMATH_CALUDE_function_difference_l3180_318036

theorem function_difference (m : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 - m * x - 8
  (f 5 - g 5 = 20) → m = -53/5 := by
sorry

end NUMINAMATH_CALUDE_function_difference_l3180_318036


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l3180_318046

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation (k²+1)x²-2x+1=0 -/
def equation_D (k : ℝ) (x : ℝ) : Prop :=
  (k^2 + 1) * x^2 - 2*x + 1 = 0

theorem equation_D_is_quadratic (k : ℝ) :
  is_quadratic_in_x (k^2 + 1) (-2) 1 := by sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l3180_318046


namespace NUMINAMATH_CALUDE_y_derivative_l3180_318080

noncomputable def y (x : ℝ) : ℝ :=
  2 * x - Real.log (1 + Real.sqrt (1 - Real.exp (4 * x))) - Real.exp (-2 * x) * Real.arcsin (Real.exp (2 * x))

theorem y_derivative (x : ℝ) :
  deriv y x = 2 * Real.exp (-2 * x) * Real.arcsin (Real.exp (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3180_318080


namespace NUMINAMATH_CALUDE_quadricycles_count_l3180_318069

/-- Given a total number of children and wheels, calculate the number of quadricycles -/
def count_quadricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  let scooter_wheels := 2
  let quadricycle_wheels := 4
  let quadricycles := (total_wheels - scooter_wheels * total_children) / (quadricycle_wheels - scooter_wheels)
  quadricycles

/-- Theorem stating that given 9 children and 30 wheels, there are 6 quadricycles -/
theorem quadricycles_count : count_quadricycles 9 30 = 6 := by
  sorry

#eval count_quadricycles 9 30

end NUMINAMATH_CALUDE_quadricycles_count_l3180_318069


namespace NUMINAMATH_CALUDE_shopping_expenditure_l3180_318010

theorem shopping_expenditure (initial_amount : ℝ) : 
  initial_amount * (1 - 0.2) * (1 - 0.15) * (1 - 0.25) = 217 →
  initial_amount = 425.49 := by
sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l3180_318010


namespace NUMINAMATH_CALUDE_coin_division_problem_l3180_318041

theorem coin_division_problem : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 7 = 5 → n ≤ m) ∧ 
  n % 8 = 6 ∧ 
  n % 7 = 5 ∧ 
  n % 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l3180_318041


namespace NUMINAMATH_CALUDE_principal_cup_problem_l3180_318039

/-- The probability of team A answering correctly -/
def P_A : ℚ := 3/4

/-- The probability of both teams A and C answering incorrectly -/
def P_AC_incorrect : ℚ := 1/12

/-- The probability of both teams B and C answering correctly -/
def P_BC_correct : ℚ := 1/4

/-- The probability of team B answering correctly -/
def P_B : ℚ := 3/8

/-- The probability of team C answering correctly -/
def P_C : ℚ := 2/3

/-- The probability of exactly two teams answering correctly -/
def P_two_correct : ℚ := 15/32

theorem principal_cup_problem (P_A P_AC_incorrect P_BC_correct P_B P_C P_two_correct : ℚ) :
  P_A = 3/4 →
  P_AC_incorrect = 1/12 →
  P_BC_correct = 1/4 →
  P_B = 3/8 ∧
  P_C = 2/3 ∧
  P_two_correct = 15/32 :=
by
  sorry

end NUMINAMATH_CALUDE_principal_cup_problem_l3180_318039


namespace NUMINAMATH_CALUDE_base7_product_l3180_318068

/-- Converts a base 7 number represented as a list of digits to its decimal (base 10) equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal (base 10) number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The problem statement -/
theorem base7_product : 
  toBase7 (toDecimal [1,3,2,4] * toDecimal [2,3]) = [3,1,4,1,5] := by
  sorry

end NUMINAMATH_CALUDE_base7_product_l3180_318068


namespace NUMINAMATH_CALUDE_three_roots_implies_m_equals_two_l3180_318031

/-- The function f(x) = x^2 - 2|x| + 2 - m -/
def f (x m : ℝ) : ℝ := x^2 - 2 * abs x + 2 - m

/-- The number of roots of f(x) for a given m -/
def num_roots (m : ℝ) : ℕ := sorry

theorem three_roots_implies_m_equals_two :
  ∀ m : ℝ, num_roots m = 3 → m = 2 := by sorry

end NUMINAMATH_CALUDE_three_roots_implies_m_equals_two_l3180_318031


namespace NUMINAMATH_CALUDE_kenny_basketball_hours_l3180_318054

/-- Represents the number of hours Kenny spent on different activities -/
structure KennyActivities where
  basketball : ℕ
  running : ℕ
  trumpet : ℕ

/-- Defines the relationships between Kenny's activities -/
def valid_activities (k : KennyActivities) : Prop :=
  k.running = 2 * k.basketball ∧
  k.trumpet = 2 * k.running ∧
  k.trumpet = 40

/-- Theorem: Given the conditions, Kenny played basketball for 10 hours -/
theorem kenny_basketball_hours (k : KennyActivities) 
  (h : valid_activities k) : k.basketball = 10 := by
  sorry


end NUMINAMATH_CALUDE_kenny_basketball_hours_l3180_318054


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3180_318048

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 2 → b = 4 → c = 4 →  -- Two sides are equal (isosceles) and one side is 2
  a + b + c = 10 :=         -- The perimeter is 10
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3180_318048


namespace NUMINAMATH_CALUDE_ellipse_properties_l3180_318087

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the condition for line l
def line_l (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 = 0

-- Define the equality of distances condition
def equal_distances (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1)^2 + y1^2 = (x2 - 1)^2 + y2^2

-- Main theorem
theorem ellipse_properties :
  -- Given conditions
  (∃ (x y : ℝ), x = 1/2 ∧ y = Real.sqrt 3 ∧ ellipse_C x y) →
  -- Conclusions
  (∀ (k m x1 y1 x2 y2 : ℝ),
    -- Line l intersects ellipse C at M(x1, y1) and N(x2, y2)
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line_l k m x1 y1 ∧ line_l k m x2 y2 ∧
    -- AM ⊥ AN and |AM| = |AN|
    perpendicular x1 y1 x2 y2 ∧ equal_distances x1 y1 x2 y2 →
    -- Then line l has one of these equations
    (k = Real.sqrt 5 ∧ m = -3/5 * Real.sqrt 5) ∨
    (k = -Real.sqrt 5 ∧ m = -3/5 * Real.sqrt 5) ∨
    (k = 0 ∧ m = -3/5)) ∧
  -- The locus of H
  (∀ (x y : ℝ), x ≠ 1 →
    (∃ (k m x1 y1 x2 y2 : ℝ),
      ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
      line_l k m x1 y1 ∧ line_l k m x2 y2 ∧
      perpendicular x1 y1 x2 y2 ∧
      -- H is on the perpendicular from A to MN
      (y - 0) / (x - 1) = -1 / k) ↔
    (x - 1/5)^2 + y^2 = 16/25) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3180_318087


namespace NUMINAMATH_CALUDE_delta_value_l3180_318058

theorem delta_value (Δ : ℤ) : 4 * (-3) = Δ + 3 → Δ = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3180_318058


namespace NUMINAMATH_CALUDE_circle_placement_existence_l3180_318055

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Represents a circle --/
structure Circle where
  diameter : ℝ

/-- Checks if a circle intersects a square --/
def circleIntersectsSquare (c : Circle) (s : Square) : Prop :=
  sorry

/-- Theorem: In a 20 by 25 rectangle with 120 unit squares, 
    there exists a point for a circle with diameter 1 that doesn't intersect any square --/
theorem circle_placement_existence 
  (r : Rectangle) 
  (squares : Finset Square) 
  (c : Circle) : 
  r.width = 20 ∧ 
  r.height = 25 ∧ 
  squares.card = 120 ∧ 
  (∀ s ∈ squares, s.side = 1) ∧ 
  c.diameter = 1 →
  ∃ (x y : ℝ), ∀ s ∈ squares, ¬circleIntersectsSquare { diameter := 1 } s :=
sorry

end NUMINAMATH_CALUDE_circle_placement_existence_l3180_318055


namespace NUMINAMATH_CALUDE_average_equation_solution_l3180_318007

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((3*x + 4) + (7*x - 5) + (4*x + 9)) = 5*x - 3 → x = 17 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3180_318007


namespace NUMINAMATH_CALUDE_base_n_problem_l3180_318029

theorem base_n_problem (n d : ℕ) : 
  n > 0 → 
  d < 10 → 
  3 * n^2 + 2 * n + d = 263 → 
  3 * n^2 + 2 * n + 4 = 396 + 7 * d → 
  n + d = 11 := by sorry

end NUMINAMATH_CALUDE_base_n_problem_l3180_318029


namespace NUMINAMATH_CALUDE_finite_perfect_squares_l3180_318043

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (S : Finset ℤ), ∀ (n : ℤ),
    (∃ (x : ℤ), a * n^2 + b = x^2) ∧ (∃ (y : ℤ), a * (n + 1)^2 + b = y^2) →
    n ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_perfect_squares_l3180_318043


namespace NUMINAMATH_CALUDE_simplify_expression_l3180_318042

theorem simplify_expression : (5^7 + 2^8) * (1^5 - (-1)^5)^10 = 80263680 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3180_318042


namespace NUMINAMATH_CALUDE_pepperoni_coverage_is_four_ninths_l3180_318077

/-- Represents a circular pizza with pepperoni toppings -/
structure PizzaWithPepperoni where
  pizzaDiameter : ℝ
  pepperoniAcrossDiameter : ℕ
  totalPepperoni : ℕ

/-- Calculates the fraction of the pizza covered by pepperoni -/
def pepperoniCoverage (pizza : PizzaWithPepperoni) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the pizza covered by pepperoni is 4/9 -/
theorem pepperoni_coverage_is_four_ninths (pizza : PizzaWithPepperoni) 
  (h1 : pizza.pizzaDiameter = 18)
  (h2 : pizza.pepperoniAcrossDiameter = 9)
  (h3 : pizza.totalPepperoni = 36) : 
  pepperoniCoverage pizza = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_is_four_ninths_l3180_318077


namespace NUMINAMATH_CALUDE_committee_probability_l3180_318016

def total_members : ℕ := 20
def boys : ℕ := 12
def girls : ℕ := 8
def committee_size : ℕ := 4

def probability_at_least_one_boy_and_girl : ℚ :=
  1 - (Nat.choose boys committee_size + Nat.choose girls committee_size : ℚ) / Nat.choose total_members committee_size

theorem committee_probability :
  probability_at_least_one_boy_and_girl = 4280 / 4845 :=
sorry

end NUMINAMATH_CALUDE_committee_probability_l3180_318016


namespace NUMINAMATH_CALUDE_four_card_selection_with_face_l3180_318084

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Number of face cards per suit -/
def FaceCardsPerSuit : ℕ := 3

/-- Number of cards per suit -/
def CardsPerSuit : ℕ := 13

/-- Theorem: Number of ways to choose 4 cards from a standard deck
    such that all four cards are of different suits and at least one is a face card -/
theorem four_card_selection_with_face (deck : ℕ) (suits : ℕ) (face_per_suit : ℕ) (cards_per_suit : ℕ)
    (h1 : deck = StandardDeck)
    (h2 : suits = NumSuits)
    (h3 : face_per_suit = FaceCardsPerSuit)
    (h4 : cards_per_suit = CardsPerSuit) :
  suits * face_per_suit * (cards_per_suit ^ (suits - 1)) = 26364 :=
sorry

end NUMINAMATH_CALUDE_four_card_selection_with_face_l3180_318084


namespace NUMINAMATH_CALUDE_lower_selling_price_l3180_318081

/-- Proves that the lower selling price is 340 given the conditions of the problem -/
theorem lower_selling_price (cost_price selling_price : ℕ) :
  cost_price = 250 →
  selling_price = 350 →
  (selling_price - cost_price : ℚ) / cost_price = 
    ((340 - cost_price : ℚ) / cost_price) + 4 / 100 →
  340 = (selling_price - cost_price) * 100 / 104 + cost_price :=
by sorry

end NUMINAMATH_CALUDE_lower_selling_price_l3180_318081


namespace NUMINAMATH_CALUDE_race_time_calculation_l3180_318018

theorem race_time_calculation (race_length : ℝ) (distance_difference : ℝ) (time_difference : ℝ) :
  race_length = 1000 →
  distance_difference = 40 →
  time_difference = 8 →
  ∃ (time_A : ℝ),
    time_A > 0 ∧
    race_length / time_A = (race_length - distance_difference) / (time_A + time_difference) ∧
    time_A = 200 := by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l3180_318018


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3180_318078

theorem product_sum_theorem (a b c : ℤ) : 
  a * b * c = -13 → (a + b + c = -11 ∨ a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3180_318078


namespace NUMINAMATH_CALUDE_haley_gives_away_48_papers_l3180_318032

/-- The number of origami papers Haley gives away -/
def total_papers (num_cousins : ℕ) (papers_per_cousin : ℕ) : ℕ :=
  num_cousins * papers_per_cousin

/-- Theorem stating that Haley gives away 48 origami papers -/
theorem haley_gives_away_48_papers : total_papers 6 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_haley_gives_away_48_papers_l3180_318032


namespace NUMINAMATH_CALUDE_rectangle_perimeter_minus_4_l3180_318003

/-- The perimeter of a rectangle minus 4, given its width and length. -/
def perimeterMinus4 (width length : ℝ) : ℝ :=
  2 * width + 2 * length - 4

/-- Theorem: For a rectangle with width 4 cm and length 8 cm, 
    the perimeter minus 4 equals 20 cm. -/
theorem rectangle_perimeter_minus_4 :
  perimeterMinus4 4 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_minus_4_l3180_318003


namespace NUMINAMATH_CALUDE_river_flow_volume_l3180_318047

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 3) 
  (h_width : width = 32) 
  (h_flow_rate : flow_rate_kmph = 2) : 
  depth * width * (flow_rate_kmph * 1000 / 60) = 3200 := by
  sorry

#check river_flow_volume

end NUMINAMATH_CALUDE_river_flow_volume_l3180_318047


namespace NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l3180_318025

theorem fraction_of_fraction (a b c d : ℚ) (h : b ≠ 0) (k : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem two_ninths_of_three_fourths :
  (2 / 9) / (3 / 4) = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l3180_318025


namespace NUMINAMATH_CALUDE_cone_volume_approximation_l3180_318070

theorem cone_volume_approximation (L h : ℝ) (h1 : L > 0) (h2 : h > 0) :
  (7 / 264 : ℝ) * L^2 * h = (1 / 3 : ℝ) * ((22 / 7 : ℝ) / 4) * L^2 * h := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_approximation_l3180_318070


namespace NUMINAMATH_CALUDE_root_transformation_l3180_318004

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3*r₁^2 + 8 = 0) ∧ 
  (r₂^3 - 3*r₂^2 + 8 = 0) ∧ 
  (r₃^3 - 3*r₃^2 + 8 = 0) →
  ((3*r₁)^3 - 9*(3*r₁)^2 + 216 = 0) ∧
  ((3*r₂)^3 - 9*(3*r₂)^2 + 216 = 0) ∧
  ((3*r₃)^3 - 9*(3*r₃)^2 + 216 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l3180_318004


namespace NUMINAMATH_CALUDE_units_digit_of_power_difference_l3180_318060

theorem units_digit_of_power_difference : ∃ n : ℕ, (25^2010 - 3^2012) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_power_difference_l3180_318060


namespace NUMINAMATH_CALUDE_factorial_ratio_l3180_318028

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3180_318028


namespace NUMINAMATH_CALUDE_square_difference_value_l3180_318057

theorem square_difference_value (m n : ℝ) (h : m^2 + n^2 = 6*m - 4*n - 13) : 
  m^2 - n^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l3180_318057


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3180_318064

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ a ≤ -1 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3180_318064


namespace NUMINAMATH_CALUDE_betty_order_total_cost_l3180_318085

/-- Calculate the total cost of Betty's order -/
theorem betty_order_total_cost :
  let slipper_quantity : ℕ := 6
  let slipper_price : ℚ := 5/2
  let lipstick_quantity : ℕ := 4
  let lipstick_price : ℚ := 5/4
  let hair_color_quantity : ℕ := 8
  let hair_color_price : ℚ := 3
  let total_items : ℕ := slipper_quantity + lipstick_quantity + hair_color_quantity
  let total_cost : ℚ := slipper_quantity * slipper_price + 
                        lipstick_quantity * lipstick_price + 
                        hair_color_quantity * hair_color_price
  total_items = 18 ∧ total_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_total_cost_l3180_318085


namespace NUMINAMATH_CALUDE_ticket_cost_difference_l3180_318015

def adult_count : ℕ := 9
def child_count : ℕ := 7
def adult_ticket_price : ℚ := 11
def child_ticket_price : ℚ := 7
def discount_rate : ℚ := 0.15
def discount_threshold : ℕ := 10

def total_tickets : ℕ := adult_count + child_count

def adult_total : ℚ := adult_count * adult_ticket_price
def child_total : ℚ := child_count * child_ticket_price
def total_cost : ℚ := adult_total + child_total

def discount_applies : Prop := total_tickets > discount_threshold

def discounted_cost : ℚ := total_cost * (1 - discount_rate)

def adult_proportion : ℚ := adult_total / total_cost
def child_proportion : ℚ := child_total / total_cost

def adult_discounted : ℚ := adult_total - (discount_rate * total_cost * adult_proportion)
def child_discounted : ℚ := child_total - (discount_rate * total_cost * child_proportion)

theorem ticket_cost_difference : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |adult_discounted - child_discounted - 42.52| < ε :=
sorry

end NUMINAMATH_CALUDE_ticket_cost_difference_l3180_318015


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l3180_318034

/-- For any triangle with circumradius R, inradius r, and semiperimeter p,
    the inequality 27Rr ≤ 2p² ≤ 27R²/2 holds. -/
theorem triangle_radius_inequality (R r p : ℝ) (h_positive : R > 0 ∧ r > 0 ∧ p > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    p = (a + b + c) / 2 ∧ 
    R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)) ∧
    r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_radius_inequality_l3180_318034


namespace NUMINAMATH_CALUDE_lemonade_cups_calculation_l3180_318014

theorem lemonade_cups_calculation (sugar_cups : ℕ) (ratio : ℚ) : 
  sugar_cups = 28 → ratio = 1/2 → sugar_cups + (sugar_cups / ratio) = 84 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_cups_calculation_l3180_318014


namespace NUMINAMATH_CALUDE_symmetric_function_property_l3180_318099

def symmetricAround (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f y = x

def symmetricAfterShift (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 1) = y ↔ f y = x

theorem symmetric_function_property (f : ℝ → ℝ)
  (h1 : symmetricAround f)
  (h2 : symmetricAfterShift f)
  (h3 : f 1 = 0) :
  f 2011 = -2010 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_property_l3180_318099


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3180_318067

theorem cube_volume_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := s₂ * Real.sqrt 3
  (s₁^3) / (s₂^3) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3180_318067


namespace NUMINAMATH_CALUDE_madeline_work_hours_l3180_318063

/-- Calculates the minimum number of work hours needed to cover expenses and savings --/
def min_work_hours (rent : ℕ) (groceries : ℕ) (medical : ℕ) (utilities : ℕ) (savings : ℕ) (hourly_wage : ℕ) : ℕ :=
  let total_expenses := rent + groceries + medical + utilities + savings
  (total_expenses + hourly_wage - 1) / hourly_wage

theorem madeline_work_hours :
  min_work_hours 1200 400 200 60 200 15 = 138 := by
  sorry

end NUMINAMATH_CALUDE_madeline_work_hours_l3180_318063


namespace NUMINAMATH_CALUDE_balls_after_1729_steps_l3180_318040

/-- Represents the state of boxes in Lisa's ball-placing game -/
def BoxState := List Nat

/-- Converts a natural number to its septenary (base-7) representation -/
def toSeptenary (n : Nat) : List Nat :=
  sorry

/-- Calculates the sum of a list of natural numbers -/
def sum (l : List Nat) : Nat :=
  sorry

/-- Simulates Lisa's ball-placing process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

/-- Counts the total number of balls in a given box state -/
def countBalls (state : BoxState) : Nat :=
  sorry

/-- Theorem stating that the number of balls after 1729 steps
    is equal to the sum of digits in the septenary representation of 1729 -/
theorem balls_after_1729_steps :
  countBalls (simulateSteps 1729) = sum (toSeptenary 1729) :=
sorry

end NUMINAMATH_CALUDE_balls_after_1729_steps_l3180_318040


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3180_318095

theorem fraction_evaluation : (1 - 2/5) / (1 - 1/4) = 4/5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3180_318095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3180_318079

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_property : ∀ n, S n = n * (a 0 + a (n-1)) / 2

/-- Theorem: For an arithmetic sequence with S_3 = 9 and S_6 = 36, S_9 = 81 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 9) (h6 : seq.S 6 = 36) : seq.S 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3180_318079


namespace NUMINAMATH_CALUDE_find_n_l3180_318094

theorem find_n (n : ℕ) 
  (h1 : Nat.gcd n 180 = 12) 
  (h2 : Nat.lcm n 180 = 720) : 
  n = 48 := by
sorry

end NUMINAMATH_CALUDE_find_n_l3180_318094


namespace NUMINAMATH_CALUDE_sun_moon_volume_ratio_l3180_318053

/-- The ratio of the Sun-Earth distance to the Moon-Earth distance -/
def distance_ratio : ℝ := 387

/-- The ratio of the Sun's volume to the Moon's volume -/
def volume_ratio : ℝ := distance_ratio ^ 3

theorem sun_moon_volume_ratio : 
  volume_ratio = distance_ratio ^ 3 := by sorry

end NUMINAMATH_CALUDE_sun_moon_volume_ratio_l3180_318053


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l3180_318093

theorem fractional_equation_positive_root (x m : ℝ) : 
  (∃ x > 0, (3 / (x - 4) = 1 - (x + m) / (4 - x))) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l3180_318093


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3180_318051

theorem sufficient_condition_range (m : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 3) → x ≤ m) → m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3180_318051


namespace NUMINAMATH_CALUDE_arcsin_plus_arcsin_2x_eq_arccos_l3180_318071

theorem arcsin_plus_arcsin_2x_eq_arccos (x : ℝ) : 
  (Real.arcsin x + Real.arcsin (2*x) = Real.arccos x) ↔ 
  (x = 0 ∨ x = 2/Real.sqrt 5 ∨ x = -2/Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_arcsin_plus_arcsin_2x_eq_arccos_l3180_318071


namespace NUMINAMATH_CALUDE_board_problem_l3180_318035

def board_operation (a b c : ℤ) : ℤ × ℤ × ℤ :=
  (a, b, a + b - c)

def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

def can_reach_sequence (start_a start_b start_c target_a target_b target_c : ℤ) : Prop :=
  ∃ (n : ℕ), ∃ (seq : ℕ → ℤ × ℤ × ℤ),
    seq 0 = (start_a, start_b, start_c) ∧
    (∀ i, i < n → 
      let (a, b, c) := seq i
      seq (i + 1) = board_operation a b c ∨ 
      seq (i + 1) = board_operation a c b ∨ 
      seq (i + 1) = board_operation b c a) ∧
    seq n = (target_a, target_b, target_c)

theorem board_problem :
  can_reach_sequence 3 9 15 2013 2019 2025 ∧
  is_arithmetic_sequence 2013 2019 2025 :=
sorry

end NUMINAMATH_CALUDE_board_problem_l3180_318035


namespace NUMINAMATH_CALUDE_library_books_end_of_month_l3180_318059

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) :
  initial_books = 75 →
  loaned_books = 60 →
  return_rate = 65 / 100 →
  initial_books - loaned_books + (return_rate * loaned_books).floor = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_end_of_month_l3180_318059


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3180_318083

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 400 →
  first_discount = 12 →
  final_price = 334.4 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3180_318083


namespace NUMINAMATH_CALUDE_distance_to_right_focus_is_18_l3180_318086

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Axiom: P is on the left branch of the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the distance from P to the left focus
def distance_to_left_focus : ℝ := 10

-- Define the distance from P to the right focus
def distance_to_right_focus : ℝ := sorry

-- Theorem to prove
theorem distance_to_right_focus_is_18 :
  distance_to_right_focus = 18 :=
sorry

end NUMINAMATH_CALUDE_distance_to_right_focus_is_18_l3180_318086


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l3180_318030

theorem ellipse_hyperbola_same_foci (k : ℝ) : k > 0 →
  (∀ x y : ℝ, x^2/9 + y^2/k^2 = 1 ↔ x^2/k - y^2/3 = 1) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l3180_318030


namespace NUMINAMATH_CALUDE_william_hot_dogs_l3180_318065

/-- The number of hot dogs William sold during the first three innings -/
def first_innings_sales : ℕ := 19

/-- The number of hot dogs William sold during the next three innings -/
def next_innings_sales : ℕ := 27

/-- The number of hot dogs William had left to sell -/
def remaining_hot_dogs : ℕ := 45

/-- The total number of hot dogs William had at first -/
def total_hot_dogs : ℕ := first_innings_sales + next_innings_sales + remaining_hot_dogs

theorem william_hot_dogs : total_hot_dogs = 91 := by sorry

end NUMINAMATH_CALUDE_william_hot_dogs_l3180_318065


namespace NUMINAMATH_CALUDE_end_with_same_digits_l3180_318027

/-- A function that returns the last four digits of a number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- A function that returns the first three digits of a four-digit number -/
def firstThreeDigits (n : ℕ) : ℕ := n / 10

theorem end_with_same_digits (N : ℕ) (h1 : N > 0) 
  (h2 : lastFourDigits N = lastFourDigits (N^2)) 
  (h3 : lastFourDigits N ≥ 1000) : firstThreeDigits (lastFourDigits N) = 937 := by
  sorry

end NUMINAMATH_CALUDE_end_with_same_digits_l3180_318027


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3180_318037

/-- A geometric sequence with the given property -/
structure GeometricSequence where
  a : ℕ → ℝ
  has_identical_roots : ∃ x : ℝ, a 1 * x^2 - a 3 * x + a 2 = 0 ∧ 
    ∀ y : ℝ, a 1 * y^2 - a 3 * y + a 2 = 0 → y = x

/-- Sum of the first n terms of a geometric sequence -/
def sum (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  sum seq 9 / sum seq 3 = 21 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3180_318037


namespace NUMINAMATH_CALUDE_roots_independent_of_k_l3180_318006

/-- The polynomial function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^4 - (k+3)*x^3 - (k-11)*x^2 + (k+3)*x + (k-12)

/-- Theorem stating that 1 and -1 are roots of the polynomial for all real k -/
theorem roots_independent_of_k :
  ∀ k : ℝ, f k 1 = 0 ∧ f k (-1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_independent_of_k_l3180_318006


namespace NUMINAMATH_CALUDE_store_charge_with_interest_l3180_318091

/-- Proves that a principal amount of $35 with 7% simple annual interest results in a total debt of $37.45 after one year -/
theorem store_charge_with_interest (P : ℝ) (interest_rate : ℝ) (total_debt : ℝ) : 
  interest_rate = 0.07 →
  total_debt = 37.45 →
  P * (1 + interest_rate) = total_debt →
  P = 35 := by
sorry

end NUMINAMATH_CALUDE_store_charge_with_interest_l3180_318091


namespace NUMINAMATH_CALUDE_no_perfect_squares_sum_l3180_318061

theorem no_perfect_squares_sum (x y : ℕ) : 
  ¬(∃ (a b : ℕ), x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_sum_l3180_318061


namespace NUMINAMATH_CALUDE_total_subscription_is_50000_l3180_318096

/-- Represents the subscription amounts and profit distribution for a business venture -/
structure BusinessSubscription where
  /-- Subscription amount for C -/
  c : ℕ
  /-- Total profit -/
  totalProfit : ℕ
  /-- A's share of the profit -/
  aProfit : ℕ

/-- Calculates the total subscription amount based on the given conditions -/
def totalSubscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c + 14000

/-- Theorem stating that the total subscription amount is 50000 given the problem conditions -/
theorem total_subscription_is_50000 (bs : BusinessSubscription) 
  (h1 : bs.totalProfit = 70000)
  (h2 : bs.aProfit = 29400)
  (h3 : bs.aProfit * (3 * bs.c + 14000) = bs.totalProfit * (bs.c + 9000)) :
  totalSubscription bs = 50000 := by
  sorry

end NUMINAMATH_CALUDE_total_subscription_is_50000_l3180_318096


namespace NUMINAMATH_CALUDE_boat_journey_time_l3180_318012

/-- The boat's journey time given specific conditions -/
theorem boat_journey_time 
  (stream_velocity : ℝ) 
  (boat_speed_still : ℝ) 
  (distance_AB : ℝ) 
  (h1 : stream_velocity = 4)
  (h2 : boat_speed_still = 14)
  (h3 : distance_AB = 180) :
  let downstream_speed := boat_speed_still + stream_velocity
  let upstream_speed := boat_speed_still - stream_velocity
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := (distance_AB / 2) / upstream_speed
  time_downstream + time_upstream = 19 := by
sorry

end NUMINAMATH_CALUDE_boat_journey_time_l3180_318012


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3180_318002

theorem perfect_square_condition (n : ℤ) : 
  ∃ (k : ℤ), n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = k^2 ↔ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3180_318002


namespace NUMINAMATH_CALUDE_inequality_solution_l3180_318022

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 3) ↔ (-2 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3180_318022


namespace NUMINAMATH_CALUDE_umbrella_cost_l3180_318020

theorem umbrella_cost (house_umbrellas car_umbrellas total_cost : ℕ) 
  (h1 : house_umbrellas = 2)
  (h2 : car_umbrellas = 1)
  (h3 : total_cost = 24) :
  total_cost / (house_umbrellas + car_umbrellas) = 8 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_cost_l3180_318020


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3180_318062

-- Define the bowties operation
noncomputable def bowtie (c d : ℝ) : ℝ := c - Real.sqrt (d - Real.sqrt (d - Real.sqrt d))

-- Theorem statement
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 7 x = 3 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3180_318062


namespace NUMINAMATH_CALUDE_solve_for_y_l3180_318009

theorem solve_for_y (x y : ℝ) : 3 * x - 2 * y = 6 → y = (3 * x / 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3180_318009


namespace NUMINAMATH_CALUDE_waiter_earnings_theorem_l3180_318050

/-- Calculates the total earnings for the first four nights of a five-day work week,
    given the target average per night and the required earnings for the last night. -/
def earnings_first_four_nights (days_per_week : ℕ) (target_average : ℚ) (last_night_earnings : ℚ) : ℚ :=
  days_per_week * target_average - last_night_earnings

theorem waiter_earnings_theorem :
  earnings_first_four_nights 5 50 115 = 135 := by
  sorry

end NUMINAMATH_CALUDE_waiter_earnings_theorem_l3180_318050


namespace NUMINAMATH_CALUDE_least_homeowners_l3180_318056

theorem least_homeowners (total_members : ℕ) (men_percentage : ℚ) (women_percentage : ℚ)
  (h_total : total_members = 150)
  (h_men_percentage : men_percentage = 1/10)
  (h_women_percentage : women_percentage = 1/5) :
  ∃ (men women : ℕ),
    men + women = total_members ∧
    ∃ (men_homeowners women_homeowners : ℕ),
      men_homeowners = ⌈men_percentage * men⌉ ∧
      women_homeowners = ⌈women_percentage * women⌉ ∧
      men_homeowners + women_homeowners = 16 ∧
      ∀ (other_men other_women : ℕ),
        other_men + other_women = total_members →
        ∃ (other_men_homeowners other_women_homeowners : ℕ),
          other_men_homeowners = ⌈men_percentage * other_men⌉ ∧
          other_women_homeowners = ⌈women_percentage * other_women⌉ →
          other_men_homeowners + other_women_homeowners ≥ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_least_homeowners_l3180_318056


namespace NUMINAMATH_CALUDE_fish_rice_equivalence_l3180_318074

/-- Represents the value of one fish in terms of bags of rice -/
def fish_to_rice_ratio : ℚ := 21 / 20

theorem fish_rice_equivalence (fish bread rice : ℚ) 
  (h1 : 4 * fish = 3 * bread) 
  (h2 : 5 * bread = 7 * rice) : 
  fish = fish_to_rice_ratio * rice := by
  sorry

#check fish_rice_equivalence

end NUMINAMATH_CALUDE_fish_rice_equivalence_l3180_318074


namespace NUMINAMATH_CALUDE_cats_given_by_mr_sheridan_l3180_318052

/-- The number of cats Mrs. Sheridan initially had -/
def initial_cats : ℕ := 17

/-- The total number of cats Mrs. Sheridan has now -/
def total_cats : ℕ := 31

/-- The number of cats Mr. Sheridan gave to Mrs. Sheridan -/
def given_cats : ℕ := total_cats - initial_cats

theorem cats_given_by_mr_sheridan : given_cats = 14 := by sorry

end NUMINAMATH_CALUDE_cats_given_by_mr_sheridan_l3180_318052


namespace NUMINAMATH_CALUDE_ratio_closest_to_five_l3180_318044

theorem ratio_closest_to_five : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |(10^2000 + 10^2002) / (10^2001 + 10^2001) - 5| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_closest_to_five_l3180_318044


namespace NUMINAMATH_CALUDE_sexagenary_cycle_3023_l3180_318038

/-- Represents a year in the sexagenary cycle -/
structure SexagenaryYear where
  heavenlyStem : Fin 10
  earthlyBranch : Fin 12

/-- The sexagenary cycle -/
def sexagenaryCycle : ℕ → SexagenaryYear := sorry

/-- Maps a natural number to its representation in the sexagenary cycle -/
def toSexagenaryYear (year : ℕ) : SexagenaryYear :=
  sexagenaryCycle (year % 60)

/-- Checks if a given SexagenaryYear corresponds to "Gui Mao" -/
def isGuiMao (year : SexagenaryYear) : Prop :=
  year.heavenlyStem = 9 ∧ year.earthlyBranch = 3

/-- Checks if a given SexagenaryYear corresponds to "Gui Wei" -/
def isGuiWei (year : SexagenaryYear) : Prop :=
  year.heavenlyStem = 9 ∧ year.earthlyBranch = 7

theorem sexagenary_cycle_3023 :
  isGuiMao (toSexagenaryYear 2023) →
  isGuiWei (toSexagenaryYear 3023) := by
  sorry

end NUMINAMATH_CALUDE_sexagenary_cycle_3023_l3180_318038


namespace NUMINAMATH_CALUDE_prob_all_even_is_one_tenth_and_half_l3180_318076

/-- Represents a die with a given number of sides -/
structure Die :=
  (sides : ℕ)
  (sides_pos : sides > 0)

/-- The number of even outcomes on a die -/
def evenOutcomes (d : Die) : ℕ :=
  d.sides / 2

/-- The probability of rolling an even number on a die -/
def probEven (d : Die) : ℚ :=
  evenOutcomes d / d.sides

/-- The three dice in the problem -/
def die1 : Die := ⟨6, by norm_num⟩
def die2 : Die := ⟨7, by norm_num⟩
def die3 : Die := ⟨9, by norm_num⟩

/-- The theorem to be proved -/
theorem prob_all_even_is_one_tenth_and_half :
  probEven die1 * probEven die2 * probEven die3 = 1 / (10 : ℚ) + 1 / (20 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_all_even_is_one_tenth_and_half_l3180_318076


namespace NUMINAMATH_CALUDE_high_season_packs_correct_l3180_318008

/-- Represents the number of tuna packs sold per hour during the high season -/
def high_season_packs : ℕ := 6

/-- Represents the number of tuna packs sold per hour during the low season -/
def low_season_packs : ℕ := 4

/-- Represents the price of each tuna pack in dollars -/
def price_per_pack : ℕ := 60

/-- Represents the number of hours fish are sold per day -/
def hours_per_day : ℕ := 15

/-- Represents the additional revenue in dollars during the high season compared to the low season -/
def additional_revenue : ℕ := 1800

theorem high_season_packs_correct :
  high_season_packs * hours_per_day * price_per_pack =
  low_season_packs * hours_per_day * price_per_pack + additional_revenue :=
by sorry

end NUMINAMATH_CALUDE_high_season_packs_correct_l3180_318008


namespace NUMINAMATH_CALUDE_balloon_arrangements_l3180_318019

def balloon_permutations : ℕ := 1260

theorem balloon_arrangements :
  (7 * 6 * 5 * 4 * 3) / 2 = balloon_permutations := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l3180_318019


namespace NUMINAMATH_CALUDE_marks_radiator_cost_l3180_318024

/-- The total cost for Mark's car radiator replacement -/
def total_cost (labor_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  labor_hours * hourly_rate + part_cost

/-- Proof that Mark's total cost for car radiator replacement is $300 -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marks_radiator_cost_l3180_318024


namespace NUMINAMATH_CALUDE_existence_of_m_n_l3180_318017

theorem existence_of_m_n (d : ℤ) : ∃ m n : ℤ, d * (m^2 - n) = n - 2*m + 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l3180_318017


namespace NUMINAMATH_CALUDE_donut_selections_l3180_318066

theorem donut_selections (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l3180_318066
