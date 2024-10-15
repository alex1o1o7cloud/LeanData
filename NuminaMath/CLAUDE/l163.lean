import Mathlib

namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l163_16320

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 11*x - 42 = 0 → 
  (∃ y : ℝ, y ≠ x ∧ y^2 - 11*y - 42 = 0) → 
  (x ≤ 14 ∧ (∀ z : ℝ, z^2 - 11*z - 42 = 0 → z ≤ 14)) :=
sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l163_16320


namespace NUMINAMATH_CALUDE_max_area_triangle_l163_16311

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is √2 + 1 when a*sin(C) = c*cos(A) and a = 2 -/
theorem max_area_triangle (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a * Real.sin C = c * Real.cos A ∧  -- Given condition
  a = 2 →  -- Given condition
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧  -- Area formula
              ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧  -- S is maximum
  ((1/2) * b * c * Real.sin A ≤ Real.sqrt 2 + 1) ∧  -- Upper bound
  (∃ b' c', (1/2) * b' * c' * Real.sin A = Real.sqrt 2 + 1)  -- Maximum is achievable
  := by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l163_16311


namespace NUMINAMATH_CALUDE_garden_flower_distribution_l163_16383

theorem garden_flower_distribution :
  ∀ (total_flowers white_flowers red_flowers white_roses white_tulips red_roses red_tulips : ℕ),
  total_flowers = 100 →
  white_flowers = 60 →
  red_flowers = total_flowers - white_flowers →
  white_roses = (3 * white_flowers) / 5 →
  white_tulips = white_flowers - white_roses →
  red_tulips = red_flowers / 2 →
  red_roses = red_flowers - red_tulips →
  (white_tulips + red_tulips) * 100 / total_flowers = 44 ∧
  (white_roses + red_roses) * 100 / total_flowers = 56 :=
by sorry

end NUMINAMATH_CALUDE_garden_flower_distribution_l163_16383


namespace NUMINAMATH_CALUDE_nonnegative_fraction_implies_nonnegative_x_l163_16361

theorem nonnegative_fraction_implies_nonnegative_x (x : ℝ) :
  (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0 → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_fraction_implies_nonnegative_x_l163_16361


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l163_16377

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : Point)

-- Define properties of a quadrilateral
def has_one_pair_parallel_sides (q : Quadrilateral) : Prop := sorry
def has_one_pair_equal_sides (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- The main theorem
theorem quadrilateral_properties :
  (∃ q : Quadrilateral, has_one_pair_parallel_sides q ∧ has_one_pair_equal_sides q ∧ ¬is_parallelogram q) ∧
  (∀ q : Quadrilateral, is_parallelogram q → has_one_pair_parallel_sides q ∧ has_one_pair_equal_sides q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l163_16377


namespace NUMINAMATH_CALUDE_fourth_hour_highest_speed_l163_16358

def distance_traveled : Fin 7 → ℝ
| 0 => 70
| 1 => 95
| 2 => 85
| 3 => 100
| 4 => 90
| 5 => 85
| 6 => 75

def average_speed (hour : Fin 7) : ℝ := distance_traveled hour

theorem fourth_hour_highest_speed :
  ∀ (hour : Fin 7), average_speed 3 ≥ average_speed hour :=
by sorry

end NUMINAMATH_CALUDE_fourth_hour_highest_speed_l163_16358


namespace NUMINAMATH_CALUDE_system_solution_l163_16318

theorem system_solution :
  ∃ (x y z t : ℂ),
    x + y = 10 ∧
    z + t = 5 ∧
    x * y = z * t ∧
    x^3 + y^3 + z^3 + t^3 = 1080 ∧
    x = 5 + Real.sqrt 17 ∧
    y = 5 - Real.sqrt 17 ∧
    z = (5 + Complex.I * Real.sqrt 7) / 2 ∧
    t = (5 - Complex.I * Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l163_16318


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l163_16347

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Theorem: The width of the foil-covered prism is 10 inches -/
theorem foil_covered_prism_width : 
  ∀ (inner : PrismDimensions),
    volume inner = 128 →
    inner.width = 2 * inner.length →
    inner.width = 2 * inner.height →
    ∃ (outer : PrismDimensions),
      outer.length = inner.length + 2 ∧
      outer.width = inner.width + 2 ∧
      outer.height = inner.height + 2 ∧
      outer.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l163_16347


namespace NUMINAMATH_CALUDE_inverse_evaluation_l163_16301

def problem (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧
  f 4 = 7 ∧
  f 6 = 3 ∧
  f 3 = 6 ∧
  f_inv ∘ f = id ∧
  f ∘ f_inv = id

theorem inverse_evaluation (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h : problem f f_inv) : 
  f_inv (f_inv 6 + f_inv 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_evaluation_l163_16301


namespace NUMINAMATH_CALUDE_f_at_three_l163_16384

def f (x : ℝ) : ℝ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem f_at_three : f 3 = 181 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_l163_16384


namespace NUMINAMATH_CALUDE_apple_cost_is_40_l163_16356

/-- The cost of apples and pears at Clark's Food Store -/
structure FruitCosts where
  pear_cost : ℕ
  apple_cost : ℕ
  apple_quantity : ℕ
  pear_quantity : ℕ
  total_spent : ℕ

/-- Theorem: The cost of a dozen apples is 40 dollars -/
theorem apple_cost_is_40 (fc : FruitCosts) 
  (h1 : fc.pear_cost = 50)
  (h2 : fc.apple_quantity = 14 ∧ fc.pear_quantity = 14)
  (h3 : fc.total_spent = 1260)
  : fc.apple_cost = 40 := by
  sorry

#check apple_cost_is_40

end NUMINAMATH_CALUDE_apple_cost_is_40_l163_16356


namespace NUMINAMATH_CALUDE_morning_bikes_count_l163_16331

/-- The number of bikes sold in the morning -/
def morning_bikes : ℕ := 19

/-- The number of bikes sold in the afternoon -/
def afternoon_bikes : ℕ := 27

/-- The number of bike clamps given with each bike -/
def clamps_per_bike : ℕ := 2

/-- The total number of bike clamps given away -/
def total_clamps : ℕ := 92

/-- Theorem stating that the number of bikes sold in the morning is 19 -/
theorem morning_bikes_count : 
  morning_bikes = 19 ∧ 
  clamps_per_bike * (morning_bikes + afternoon_bikes) = total_clamps := by
  sorry

end NUMINAMATH_CALUDE_morning_bikes_count_l163_16331


namespace NUMINAMATH_CALUDE_sum_of_ages_l163_16366

/-- The sum of Jed and Matt's present ages given their age relationship and Jed's future age -/
theorem sum_of_ages (jed_age matt_age : ℕ) : 
  jed_age = matt_age + 10 →  -- Jed is 10 years older than Matt
  jed_age + 10 = 25 →        -- In 10 years, Jed will be 25 years old
  jed_age + matt_age = 20 :=  -- The sum of their present ages is 20
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l163_16366


namespace NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l163_16376

theorem coefficient_x6_in_expansion : 
  let expansion := (1 + X : Polynomial ℤ)^6 * (1 - X : Polynomial ℤ)^6
  (expansion.coeff 6) = -20 := by sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l163_16376


namespace NUMINAMATH_CALUDE_igor_process_terminates_l163_16378

/-- Appends a digit to make a number divisible by 11 -/
def appendDigit (n : Nat) : Nat :=
  let m := n * 10
  (m + (11 - m % 11) % 11)

/-- Performs one step of Igor's process -/
def igorStep (n : Nat) : Nat :=
  (appendDigit n) / 11

/-- Checks if Igor can continue the process -/
def canContinue (n : Nat) : Bool :=
  ∃ (d : Nat), d < 10 ∧ (n * 10 + d) % 11 = 0

/-- The sequence of numbers generated by Igor's process -/
def igorSequence : Nat → Nat
  | 0 => 2018
  | n + 1 => igorStep (igorSequence n)

theorem igor_process_terminates :
  ∃ (N : Nat), ¬(canContinue (igorSequence N)) :=
sorry

end NUMINAMATH_CALUDE_igor_process_terminates_l163_16378


namespace NUMINAMATH_CALUDE_chessboard_diagonal_squares_l163_16370

/-- The number of squares a diagonal passes through on a chessboard -/
def diagonalSquares (width : Nat) (height : Nat) : Nat :=
  width + height + Nat.gcd width height - 2

/-- Theorem: The diagonal of a 1983 × 999 chessboard passes through 2979 squares -/
theorem chessboard_diagonal_squares :
  diagonalSquares 1983 999 = 2979 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_diagonal_squares_l163_16370


namespace NUMINAMATH_CALUDE_donut_shop_problem_l163_16306

def donut_combinations (total_donuts : ℕ) (types : ℕ) : ℕ :=
  let remaining := total_donuts - types
  (types.choose 1) * (types.choose 1) * (types.choose 1) + 
  (types.choose 2) * (remaining.choose 2) +
  (types.choose 3) * (remaining.choose 1)

theorem donut_shop_problem :
  donut_combinations 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_donut_shop_problem_l163_16306


namespace NUMINAMATH_CALUDE_min_product_tangents_acute_triangle_l163_16335

theorem min_product_tangents_acute_triangle (α β γ : Real) 
  (h_acute : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α < π/2 ∧ β < π/2 ∧ γ < π/2) 
  (h_sum : α + β + γ = π) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ Real.sqrt 27 ∧ 
  (Real.tan α * Real.tan β * Real.tan γ = Real.sqrt 27 ↔ α = π/3 ∧ β = π/3 ∧ γ = π/3) :=
sorry

end NUMINAMATH_CALUDE_min_product_tangents_acute_triangle_l163_16335


namespace NUMINAMATH_CALUDE_smallest_element_100th_set_l163_16338

/-- Defines the smallest element of the nth set in the sequence -/
def smallest_element (n : ℕ) : ℕ := 
  (n - 1) * (n + 2) / 2 + 1

/-- The sequence of sets where the nth set contains n+1 consecutive integers -/
def set_sequence (n : ℕ) : Set ℕ :=
  {k : ℕ | smallest_element n ≤ k ∧ k < smallest_element (n + 1)}

/-- Theorem stating that the smallest element of the 100th set is 5050 -/
theorem smallest_element_100th_set : 
  smallest_element 100 = 5050 := by sorry

end NUMINAMATH_CALUDE_smallest_element_100th_set_l163_16338


namespace NUMINAMATH_CALUDE_only_propositions_3_and_4_are_correct_l163_16390

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations between planes and lines
def perpendicular (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p q : Plane) : Plane := sorry

-- Define the planes and lines
def α : Plane := sorry
def β : Plane := sorry
def γ : Plane := sorry
def l : Line := sorry
def m : Line := sorry
def n : Line := sorry

-- Define the propositions
def proposition_1 : Prop :=
  (perpendicular α γ ∧ perpendicular β γ) → parallel α β

def proposition_2 : Prop :=
  (parallel_line_plane m β ∧ parallel_line_plane n β) → parallel α β

def proposition_3 : Prop :=
  (line_in_plane l α ∧ parallel α β) → parallel_line_plane l β

def proposition_4 : Prop :=
  (intersection α β = γ ∧ intersection β γ = m ∧ intersection γ α = n ∧ parallel_line_plane l m) →
  parallel_line_plane m n

-- Theorem to prove
theorem only_propositions_3_and_4_are_correct :
  ¬proposition_1 ∧ ¬proposition_2 ∧ proposition_3 ∧ proposition_4 :=
sorry

end NUMINAMATH_CALUDE_only_propositions_3_and_4_are_correct_l163_16390


namespace NUMINAMATH_CALUDE_no_intersection_l163_16310

-- Define the two functions
def f (x : ℝ) : ℝ := |3*x + 6|
def g (x : ℝ) : ℝ := -|4*x - 3|

-- Theorem statement
theorem no_intersection :
  ¬ ∃ (x y : ℝ), f x = y ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l163_16310


namespace NUMINAMATH_CALUDE_chocolate_division_l163_16381

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  (total_chocolate / num_piles) * piles_given = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l163_16381


namespace NUMINAMATH_CALUDE_common_roots_product_l163_16300

-- Define the cubic equations
def cubic1 (C : ℝ) (x : ℝ) : ℝ := x^3 + C*x + 20
def cubic2 (D : ℝ) (x : ℝ) : ℝ := x^3 + D*x^2 + 80

-- Define the theorem
theorem common_roots_product (C D : ℝ) (u v : ℝ) :
  (∃ w, cubic1 C u = 0 ∧ cubic1 C v = 0 ∧ cubic1 C w = 0) →
  (∃ t, cubic2 D u = 0 ∧ cubic2 D v = 0 ∧ cubic2 D t = 0) →
  u * v = 10 * Real.rpow 4 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l163_16300


namespace NUMINAMATH_CALUDE_right_triangle_with_incircle_legs_l163_16346

/-- A right-angled triangle with an incircle touching the hypotenuse -/
structure RightTriangleWithIncircle where
  -- The lengths of the sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the segments of the hypotenuse
  ap : ℝ
  bp : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  hypotenuse : c = ap + bp
  incircle_property : ap = (a + b + c) / 2 - a ∧ bp = (a + b + c) / 2 - b
  -- Given values
  ap_value : ap = 12
  bp_value : bp = 5

/-- The main theorem -/
theorem right_triangle_with_incircle_legs 
  (triangle : RightTriangleWithIncircle) : 
  triangle.a = 8 ∧ triangle.b = 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_incircle_legs_l163_16346


namespace NUMINAMATH_CALUDE_terry_lunch_options_l163_16394

theorem terry_lunch_options :
  ∀ (lettuce_types tomato_types olive_types soup_types : ℕ),
    lettuce_types = 2 →
    tomato_types = 3 →
    olive_types = 4 →
    soup_types = 2 →
    (lettuce_types * tomato_types * olive_types * soup_types) = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_terry_lunch_options_l163_16394


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l163_16302

/-- The line equation passing through a fixed point for all real m -/
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x + (m - 1) * y - 3 = 0

/-- The theorem stating that the line passes through (1, -1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 1 (-1) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l163_16302


namespace NUMINAMATH_CALUDE_opposite_expressions_imply_y_value_l163_16326

theorem opposite_expressions_imply_y_value :
  ∀ y : ℚ, (4 * y + 8) = -(8 * y - 7) → y = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_imply_y_value_l163_16326


namespace NUMINAMATH_CALUDE_expression_evaluation_l163_16339

theorem expression_evaluation :
  let x : ℝ := 3 + Real.sqrt 2
  (1 - 1 / (x + 3)) / ((x + 2) / (x^2 - 9)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l163_16339


namespace NUMINAMATH_CALUDE_inequality_equivalence_l163_16313

theorem inequality_equivalence (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ 7 / 3 < x ∧ x < 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l163_16313


namespace NUMINAMATH_CALUDE_triangle_sine_product_inequality_l163_16359

theorem triangle_sine_product_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_product_inequality_l163_16359


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_seven_l163_16332

theorem gcd_of_powers_of_seven : Nat.gcd (7^11 + 1) (7^11 + 7^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_seven_l163_16332


namespace NUMINAMATH_CALUDE_min_x_minus_y_l163_16345

theorem min_x_minus_y (x y : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) →
  y ∈ Set.Icc 0 (2 * Real.pi) →
  2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1/2 →
  ∃ (z : Real), z = x - y ∧ ∀ (w : Real), w = x - y → z ≤ w ∧ z = -Real.pi/2 :=
by sorry

end NUMINAMATH_CALUDE_min_x_minus_y_l163_16345


namespace NUMINAMATH_CALUDE_simplify_radicals_l163_16350

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l163_16350


namespace NUMINAMATH_CALUDE_problem_proof_l163_16357

theorem problem_proof (a b : ℝ) (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) : 
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) := by sorry

end NUMINAMATH_CALUDE_problem_proof_l163_16357


namespace NUMINAMATH_CALUDE_solution_pairs_l163_16368

theorem solution_pairs : 
  ∀ x y : ℝ, 
    (x^2 + y^2 + x + y = x*y*(x + y) - 10/27 ∧ 
     |x*y| ≤ 25/9) ↔ 
    ((x = -1/3 ∧ y = -1/3) ∨ (x = 5/3 ∧ y = 5/3)) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l163_16368


namespace NUMINAMATH_CALUDE_product_of_hash_operations_l163_16308

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + a / b

-- Theorem statement
theorem product_of_hash_operations :
  let x := hash 8 3
  let y := hash 5 4
  x * y = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_product_of_hash_operations_l163_16308


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l163_16386

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l163_16386


namespace NUMINAMATH_CALUDE_min_value_of_sum_l163_16315

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (2 * y + 3) = 1 / 4) : 
  x + 3 * y ≥ 2 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l163_16315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l163_16369

theorem arithmetic_sequence_length :
  ∀ (a₁ d : ℤ) (n : ℕ),
    a₁ = -6 →
    d = 5 →
    a₁ + (n - 1) * d = 59 →
    n = 14 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l163_16369


namespace NUMINAMATH_CALUDE_min_cost_disinfectants_l163_16391

/-- Represents the price and quantity of disinfectants A and B -/
structure Disinfectants where
  price_A : ℕ
  price_B : ℕ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Calculates the total cost of purchasing disinfectants -/
def total_cost (d : Disinfectants) : ℕ :=
  d.price_A * d.quantity_A + d.price_B * d.quantity_B

/-- Represents the constraints on quantities of disinfectants -/
def valid_quantities (d : Disinfectants) : Prop :=
  d.quantity_A + d.quantity_B = 30 ∧
  d.quantity_A ≥ d.quantity_B + 5 ∧
  d.quantity_A ≤ 2 * d.quantity_B

theorem min_cost_disinfectants :
  ∃ (d : Disinfectants),
    d.price_A = 45 ∧
    d.price_B = 35 ∧
    9 * d.price_A + 6 * d.price_B = 615 ∧
    8 * d.price_A + 12 * d.price_B = 780 ∧
    valid_quantities d ∧
    (∀ (d' : Disinfectants), valid_quantities d' → total_cost d ≤ total_cost d') ∧
    total_cost d = 1230 :=
by
  sorry

end NUMINAMATH_CALUDE_min_cost_disinfectants_l163_16391


namespace NUMINAMATH_CALUDE_function_equality_l163_16389

theorem function_equality (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))) → 
  ∃ c : ℝ, ∀ x : ℝ, f x = c - x :=
sorry

end NUMINAMATH_CALUDE_function_equality_l163_16389


namespace NUMINAMATH_CALUDE_watermelon_puzzle_l163_16352

theorem watermelon_puzzle (A B C : ℕ) 
  (h1 : C - (A + B) = 6)
  (h2 : (B + C) - A = 16)
  (h3 : (C + A) - B = 8) :
  A + B + C = 18 := by
sorry

end NUMINAMATH_CALUDE_watermelon_puzzle_l163_16352


namespace NUMINAMATH_CALUDE_james_sales_problem_l163_16398

theorem james_sales_problem :
  let houses_day1 : ℕ := 20
  let houses_day2 : ℕ := 40
  let sale_rate_day2 : ℚ := 4/5
  let total_items : ℕ := 104
  let items_per_house : ℕ := 2
  
  (houses_day1 * items_per_house + 
   (houses_day2 : ℚ) * sale_rate_day2 * (items_per_house : ℚ) = (total_items : ℚ)) ∧
  (houses_day2 = 2 * houses_day1) :=
by
  sorry

end NUMINAMATH_CALUDE_james_sales_problem_l163_16398


namespace NUMINAMATH_CALUDE_inequality_proof_l163_16379

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  1 / Real.sqrt (1 + x^2) + 1 / Real.sqrt (1 + y^2) ≤ 2 / Real.sqrt (1 + x*y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l163_16379


namespace NUMINAMATH_CALUDE_x_wins_more_probability_l163_16375

/-- Represents a soccer tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (win_probability : ℚ)

/-- Represents the result of the tournament for two specific teams --/
inductive TournamentResult
  | XWinsMore
  | YWinsMore
  | Tie

/-- The probability of team X finishing with more points than team Y --/
def prob_X_wins_more (t : Tournament) : ℚ :=
  sorry

/-- The main theorem stating the probability of team X finishing with more points than team Y --/
theorem x_wins_more_probability (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.win_probability = 1/2) : 
  prob_X_wins_more t = 1/2 :=
sorry

end NUMINAMATH_CALUDE_x_wins_more_probability_l163_16375


namespace NUMINAMATH_CALUDE_angle_four_value_l163_16373

theorem angle_four_value (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle1 + 70 + 40 = 180)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 35 := by
sorry

end NUMINAMATH_CALUDE_angle_four_value_l163_16373


namespace NUMINAMATH_CALUDE_translation_preserves_segment_find_translated_point_l163_16397

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def apply_translation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_preserves_segment (A B A' : Point) (t : Translation) :
  apply_translation t A = A' →
  apply_translation t B = 
    { x := B.x + (A'.x - A.x), 
      y := B.y + (A'.y - A.y) } := by sorry

/-- The main theorem -/
theorem find_translated_point :
  let A : Point := { x := -1, y := 2 }
  let A' : Point := { x := 3, y := -4 }
  let B : Point := { x := 2, y := 4 }
  let t : Translation := { dx := A'.x - A.x, dy := A'.y - A.y }
  apply_translation t B = { x := 6, y := -2 } := by sorry

end NUMINAMATH_CALUDE_translation_preserves_segment_find_translated_point_l163_16397


namespace NUMINAMATH_CALUDE_absolute_value_plus_power_minus_sqrt_l163_16351

theorem absolute_value_plus_power_minus_sqrt : |-2| + 2023^0 - Real.sqrt 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_power_minus_sqrt_l163_16351


namespace NUMINAMATH_CALUDE_distinct_cuttings_count_l163_16334

/-- Represents a square grid --/
def Square (n : ℕ) := Fin n → Fin n → Bool

/-- Represents an L-shaped piece (corner) --/
structure LPiece where
  size : ℕ
  position : Fin 4 × Fin 4

/-- Represents a cutting of a 4x4 square --/
structure Cutting where
  lpieces : Fin 3 → LPiece
  small_square : Fin 4 × Fin 4

/-- Checks if two cuttings are distinct (considering rotations and reflections) --/
def is_distinct (c1 c2 : Cutting) : Bool := sorry

/-- Counts the number of distinct ways to cut a 4x4 square --/
def count_distinct_cuttings : ℕ := sorry

/-- The main theorem stating that there are 64 distinct ways to cut the 4x4 square --/
theorem distinct_cuttings_count : count_distinct_cuttings = 64 := by sorry

end NUMINAMATH_CALUDE_distinct_cuttings_count_l163_16334


namespace NUMINAMATH_CALUDE_vince_ride_length_l163_16354

-- Define the length of Zachary's bus ride
def zachary_ride : ℝ := 0.5

-- Define how much longer Vince's ride is compared to Zachary's
def difference : ℝ := 0.13

-- Define Vince's bus ride length
def vince_ride : ℝ := zachary_ride + difference

-- Theorem statement
theorem vince_ride_length : vince_ride = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_vince_ride_length_l163_16354


namespace NUMINAMATH_CALUDE_sin_even_function_phi_l163_16343

theorem sin_even_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, f (x - φ) = f (φ - x)) →
  φ = π / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_even_function_phi_l163_16343


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l163_16319

theorem room_breadth_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 18 →
  carpet_width = 0.75 →
  carpet_cost_per_meter = 4.50 →
  total_cost = 810 →
  (total_cost / carpet_cost_per_meter) * carpet_width / room_length = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l163_16319


namespace NUMINAMATH_CALUDE_fraction_equality_l163_16337

/-- Given two integers A and B satisfying the equation for all real x except 0, 3, and roots of x^2 + 2x + 1 = 0, prove that B/A = 0 -/
theorem fraction_equality (A B : ℤ) 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 3 → x^2 + 2*x + 1 ≠ 0 → 
    (A / (x - 3) : ℝ) + (B / (x^2 + 2*x + 1) : ℝ) = (x^3 - x^2 + 3*x + 1) / (x^3 - x - 3)) : 
  (B : ℚ) / A = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l163_16337


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l163_16380

theorem geometric_sequence_first_term 
  (a₅ a₆ : ℚ)
  (h₁ : a₅ = 48)
  (h₂ : a₆ = 64)
  : ∃ (a : ℚ), a₅ = a * (a₆ / a₅)^4 ∧ a = 243 / 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l163_16380


namespace NUMINAMATH_CALUDE_point_moved_right_l163_16396

def move_right (x y d : ℝ) : ℝ × ℝ := (x + d, y)

theorem point_moved_right :
  let A : ℝ × ℝ := (2, -1)
  let d : ℝ := 3
  move_right A.1 A.2 d = (5, -1) := by sorry

end NUMINAMATH_CALUDE_point_moved_right_l163_16396


namespace NUMINAMATH_CALUDE_punch_mixture_l163_16321

theorem punch_mixture (total_volume : ℕ) (lemonade_parts : ℕ) (extra_cranberry_parts : ℕ) :
  total_volume = 72 →
  lemonade_parts = 3 →
  extra_cranberry_parts = 18 →
  lemonade_parts + (lemonade_parts + extra_cranberry_parts) = total_volume →
  lemonade_parts + extra_cranberry_parts = 21 := by
  sorry

#check punch_mixture

end NUMINAMATH_CALUDE_punch_mixture_l163_16321


namespace NUMINAMATH_CALUDE_max_true_statements_l163_16372

theorem max_true_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (s1 s2 s3 s4 s5 : Bool),
    s1 = (1 / a > 1 / b) ∧
    s2 = (a^2 > b^2) ∧
    s3 = (a > b) ∧
    s4 = (|a| > 1) ∧
    s5 = (b < 1) ∧
    s1 + s2 + s3 + s4 + s5 ≤ 4) ∧
  (∃ (s1 s2 s3 s4 s5 : Bool),
    s1 = (1 / a > 1 / b) ∧
    s2 = (a^2 > b^2) ∧
    s3 = (a > b) ∧
    s4 = (|a| > 1) ∧
    s5 = (b < 1) ∧
    s1 + s2 + s3 + s4 + s5 = 4) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l163_16372


namespace NUMINAMATH_CALUDE_circle_inequality_l163_16349

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_l163_16349


namespace NUMINAMATH_CALUDE_championship_completion_impossible_l163_16328

/-- Represents a chess game between two players -/
structure Game where
  player1 : Nat
  player2 : Nat
  deriving Repr

/-- Represents the state of the chess championship -/
structure ChampionshipState where
  numPlayers : Nat
  gamesPlayed : List Game
  deriving Repr

/-- Checks if the championship rules are followed -/
def rulesFollowed (state : ChampionshipState) : Prop :=
  ∀ p1 p2, p1 < state.numPlayers → p2 < state.numPlayers → p1 ≠ p2 →
    let gamesPlayedByP1 := (state.gamesPlayed.filter (λ g => g.player1 = p1 ∨ g.player2 = p1)).length
    let gamesPlayedByP2 := (state.gamesPlayed.filter (λ g => g.player1 = p2 ∨ g.player2 = p2)).length
    (gamesPlayedByP1 : Int) - gamesPlayedByP2 ≤ 1 ∧ gamesPlayedByP2 - gamesPlayedByP1 ≤ 1

/-- Checks if the championship is complete -/
def isComplete (state : ChampionshipState) : Prop :=
  state.gamesPlayed.length = state.numPlayers * (state.numPlayers - 1) / 2

/-- Theorem: There exists a championship state that follows the rules but cannot be completed -/
theorem championship_completion_impossible : ∃ (state : ChampionshipState), 
  rulesFollowed state ∧ ¬∃ (finalState : ChampionshipState), 
    finalState.numPlayers = state.numPlayers ∧ 
    state.gamesPlayed ⊆ finalState.gamesPlayed ∧ 
    rulesFollowed finalState ∧ 
    isComplete finalState :=
sorry

end NUMINAMATH_CALUDE_championship_completion_impossible_l163_16328


namespace NUMINAMATH_CALUDE_share_calculation_l163_16329

theorem share_calculation (total : ℕ) (a b c : ℕ) : 
  total = 770 →
  a = b + 40 →
  c = a + 30 →
  total = a + b + c →
  b = 220 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l163_16329


namespace NUMINAMATH_CALUDE_only_C_has_inverse_l163_16360

-- Define the set of graph labels
inductive GraphLabel
  | A | B | C | D | E

-- Define a predicate for functions that have inverses
def has_inverse (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.C => True
  | _ => False

-- Theorem statement
theorem only_C_has_inverse :
  ∀ g : GraphLabel, has_inverse g ↔ g = GraphLabel.C :=
by sorry

end NUMINAMATH_CALUDE_only_C_has_inverse_l163_16360


namespace NUMINAMATH_CALUDE_security_deposit_percentage_l163_16395

/-- Security deposit calculation for a mountain cabin rental --/
theorem security_deposit_percentage
  (daily_rate : ℚ)
  (duration_days : ℕ)
  (pet_fee : ℚ)
  (service_fee_rate : ℚ)
  (security_deposit : ℚ)
  (h1 : daily_rate = 125)
  (h2 : duration_days = 14)
  (h3 : pet_fee = 100)
  (h4 : service_fee_rate = 1/5)
  (h5 : security_deposit = 1110) :
  security_deposit / (daily_rate * duration_days + pet_fee + service_fee_rate * (daily_rate * duration_days + pet_fee)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_security_deposit_percentage_l163_16395


namespace NUMINAMATH_CALUDE_product_of_integers_l163_16363

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 2 * a * b + 12 * a = 3 * b + 240 → a * b = 255 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l163_16363


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l163_16344

/-- Given a quadratic function f(x) = ax^2 + 3x - 2, 
    if the slope of its tangent line at x = 2 is 7, then a = 1 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 + 3 * x - 2
  let f' : ℝ → ℝ := λ x => 2 * a * x + 3
  f' 2 = 7 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l163_16344


namespace NUMINAMATH_CALUDE_white_balls_count_l163_16309

theorem white_balls_count (total : ℕ) (red : ℕ) (prob : ℚ) : 
  red = 8 → 
  prob = 2/5 → 
  prob = red / total → 
  total - red = 12 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l163_16309


namespace NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_log_squared_difference_plus_log_l163_16305

-- Part 1
theorem sqrt_3_times_612_times_3_and_half (x : ℝ) :
  x = Real.sqrt 3 * 612 * (3 + 3/2) → x = 3 := by sorry

-- Part 2
theorem log_squared_difference_plus_log (x : ℝ) :
  x = (Real.log 5 / Real.log 10)^2 - (Real.log 2 / Real.log 10)^2 + (Real.log 4 / Real.log 10) → x = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_log_squared_difference_plus_log_l163_16305


namespace NUMINAMATH_CALUDE_valid_pairings_count_l163_16307

def num_bowls : ℕ := 6
def num_glasses : ℕ := 6

theorem valid_pairings_count :
  (num_bowls * num_glasses : ℕ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairings_count_l163_16307


namespace NUMINAMATH_CALUDE_number_equality_l163_16374

theorem number_equality (T : ℝ) : (1/3 : ℝ) * (1/8 : ℝ) * T = (1/4 : ℝ) * (1/6 : ℝ) * 150 → T = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l163_16374


namespace NUMINAMATH_CALUDE_employee_price_calculation_l163_16348

/-- Calculates the employee's price for a video recorder given the wholesale cost, markup percentage, and employee discount percentage. -/
theorem employee_price_calculation 
  (wholesale_cost : ℝ) 
  (markup_percentage : ℝ) 
  (employee_discount_percentage : ℝ) : 
  wholesale_cost = 200 ∧ 
  markup_percentage = 20 ∧ 
  employee_discount_percentage = 30 → 
  wholesale_cost * (1 + markup_percentage / 100) * (1 - employee_discount_percentage / 100) = 168 := by
  sorry

#check employee_price_calculation

end NUMINAMATH_CALUDE_employee_price_calculation_l163_16348


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l163_16387

theorem radical_conjugate_sum_product (a b : ℝ) 
  (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 4)
  (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 9) : 
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l163_16387


namespace NUMINAMATH_CALUDE_stock_price_calculation_l163_16336

theorem stock_price_calculation (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) :
  initial_price = 120 →
  first_year_increase = 0.8 →
  second_year_decrease = 0.3 →
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 151.2 :=
by sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l163_16336


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l163_16392

/-- The nonnegative difference between the roots of x^2 + 30x + 180 = -36 is 6 -/
theorem root_difference_quadratic : 
  let f : ℝ → ℝ := λ x => x^2 + 30*x + 216
  ∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| = 6 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l163_16392


namespace NUMINAMATH_CALUDE_wood_length_proof_l163_16303

/-- The initial length of the wood Tom cut. -/
def initial_length : ℝ := 143

/-- The length cut off from the initial piece of wood. -/
def cut_length : ℝ := 25

/-- The original length of other boards before cutting. -/
def other_boards_original : ℝ := 125

/-- The length cut off from other boards. -/
def other_boards_cut : ℝ := 7

theorem wood_length_proof :
  initial_length - cut_length > other_boards_original - other_boards_cut ∧
  initial_length = 143 := by
  sorry

#check wood_length_proof

end NUMINAMATH_CALUDE_wood_length_proof_l163_16303


namespace NUMINAMATH_CALUDE_min_value_expression_l163_16355

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 3) : 
  ∃ (min : ℝ), min = 16/9 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
    (x + y) / (x * y * z) ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l163_16355


namespace NUMINAMATH_CALUDE_distance_AB_l163_16393

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, -Real.sqrt 3 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * Real.sin (θ - Real.pi / 6)

def intersection_OA (t : ℝ) : Prop :=
  ∃ θ : ℝ, line_l t = curve_C1 θ

def intersection_OB (t : ℝ) : Prop :=
  ∃ θ : ℝ, line_l t = (curve_C2 θ * Real.cos θ, curve_C2 θ * Real.sin θ)

theorem distance_AB :
  ∀ t₁ t₂ : ℝ, intersection_OA t₁ → intersection_OB t₂ →
    Real.sqrt ((t₂ - t₁)^2 + (-Real.sqrt 3 * t₂ + Real.sqrt 3 * t₁)^2) = 4 - Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_l163_16393


namespace NUMINAMATH_CALUDE_expression_value_l163_16365

theorem expression_value (a b : ℝ) (h : (a - 3)^2 + |b + 2| = 0) :
  (-a^2 + 3*a*b - 3*b^2) - 2*(-1/2*a^2 + 4*a*b - 3/2*b^2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l163_16365


namespace NUMINAMATH_CALUDE_dog_toy_cost_l163_16371

/-- The cost of dog toys with a "buy one get one half off" deal -/
theorem dog_toy_cost (regular_price : ℝ) (num_toys : ℕ) : regular_price = 12 → num_toys = 4 →
  let discounted_price := regular_price / 2
  let pair_price := regular_price + discounted_price
  let total_cost := (num_toys / 2 : ℝ) * pair_price
  total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_dog_toy_cost_l163_16371


namespace NUMINAMATH_CALUDE_arrangement_exists_for_23_l163_16353

/-- Definition of the Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of an arrangement for P = 23 -/
theorem arrangement_exists_for_23 : ∃ (F : ℕ → ℤ), F 12 % 23 = 0 ∧
  (∀ n, F (n + 2) = 3 * F (n + 1) - F n) ∧ F 0 = 0 ∧ F 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_arrangement_exists_for_23_l163_16353


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l163_16340

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_complement_theorem : 
  B ∩ (Set.compl A) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l163_16340


namespace NUMINAMATH_CALUDE_angle_BCD_measure_l163_16341

-- Define a pentagon
structure Pentagon :=
  (A B C D E : ℝ)

-- Define the conditions
def pentagon_conditions (p : Pentagon) : Prop :=
  p.A = 100 ∧ p.D = 120 ∧ p.E = 80 ∧ p.A + p.B + p.C + p.D + p.E = 540

-- Theorem statement
theorem angle_BCD_measure (p : Pentagon) :
  pentagon_conditions p → p.B = 140 → p.C = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_BCD_measure_l163_16341


namespace NUMINAMATH_CALUDE_painted_cubes_l163_16314

theorem painted_cubes (n : ℕ) (h : n = 5) : 
  n^3 - (n - 2)^3 = 98 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_l163_16314


namespace NUMINAMATH_CALUDE_infinite_prime_divisors_l163_16388

/-- A sequence of positive integers where no term divides another -/
def NonDivisibleSequence (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≠ j → ¬(a i ∣ a j)

/-- The set of primes dividing at least one term of the sequence -/
def PrimeDivisorsSet (a : ℕ → ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ i, p ∣ a i}

theorem infinite_prime_divisors (a : ℕ → ℕ) 
    (h : NonDivisibleSequence a) : Set.Infinite (PrimeDivisorsSet a) := by
  sorry

end NUMINAMATH_CALUDE_infinite_prime_divisors_l163_16388


namespace NUMINAMATH_CALUDE_standard_pairs_parity_l163_16323

/-- Represents the color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard -/
def Chessboard (m n : ℕ) := Fin m → Fin n → Color

/-- Counts the number of standard pairs on the chessboard -/
def count_standard_pairs (board : Chessboard m n) : ℕ := sorry

/-- Counts the number of blue squares on the edges (excluding corners) -/
def count_blue_edges (board : Chessboard m n) : ℕ := sorry

/-- The main theorem: The parity of standard pairs is equivalent to the parity of blue edge squares -/
theorem standard_pairs_parity (m n : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 3) (board : Chessboard m n) :
  Even (count_standard_pairs board) ↔ Even (count_blue_edges board) := by sorry

end NUMINAMATH_CALUDE_standard_pairs_parity_l163_16323


namespace NUMINAMATH_CALUDE_work_done_circular_path_l163_16385

/-- The work done by a force field on a mass point moving along a circular path -/
theorem work_done_circular_path (m a : ℝ) (h : a > 0) : 
  let force (x y : ℝ) := (x + y, -x)
  let path (t : ℝ) := (a * Real.cos t, -a * Real.sin t)
  let work := ∫ t in (0)..(2 * Real.pi), 
    m * (force (path t).1 (path t).2).1 * (-a * Real.sin t) + 
    m * (force (path t).1 (path t).2).2 * (-a * Real.cos t)
  work = 0 :=
sorry

end NUMINAMATH_CALUDE_work_done_circular_path_l163_16385


namespace NUMINAMATH_CALUDE_problem_solution_l163_16364

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 12) : 
  q = 6 + 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l163_16364


namespace NUMINAMATH_CALUDE_pencil_cost_l163_16333

theorem pencil_cost (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 340)
  (eq2 : 3 * x + 6 * y = 264) : 
  y = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l163_16333


namespace NUMINAMATH_CALUDE_dress_price_proof_l163_16325

theorem dress_price_proof (P : ℝ) (Pd : ℝ) (Pf : ℝ) 
  (h1 : Pd = 0.85 * P) 
  (h2 : Pf = 1.25 * Pd) 
  (h3 : P - Pf = 5.25) : 
  Pd = 71.40 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_proof_l163_16325


namespace NUMINAMATH_CALUDE_product_of_integers_l163_16322

theorem product_of_integers (A B C D : ℕ+) : 
  (A : ℝ) + (B : ℝ) + (C : ℝ) + (D : ℝ) = 40 →
  (A : ℝ) + 3 = (B : ℝ) - 3 ∧ 
  (A : ℝ) + 3 = (C : ℝ) * 3 ∧ 
  (A : ℝ) + 3 = (D : ℝ) / 3 →
  (A : ℝ) * (B : ℝ) * (C : ℝ) * (D : ℝ) = 2666.25 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l163_16322


namespace NUMINAMATH_CALUDE_equation_solution_l163_16327

theorem equation_solution :
  ∃ x : ℝ, 3 * (16 : ℝ)^x + 37 * (36 : ℝ)^x = 26 * (81 : ℝ)^x ∧ x = (1/2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l163_16327


namespace NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_alpha_half_l163_16362

theorem cos_squared_pi_sixth_plus_alpha_half (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_sixth_plus_alpha_half_l163_16362


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l163_16312

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l163_16312


namespace NUMINAMATH_CALUDE_problem_solution_l163_16382

theorem problem_solution : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * Nat.factorial 7 = 
  (5^128 - 4^128) * 5040 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l163_16382


namespace NUMINAMATH_CALUDE_student_count_l163_16367

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 21)
  (h2 : rank_from_left = 11) :
  rank_from_right + rank_from_left - 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l163_16367


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_fourteen_l163_16304

/-- A cubic function f(x) with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 4

/-- Theorem stating that f(-2) = -14 given the conditions -/
theorem f_neg_two_eq_neg_fourteen (a b : ℝ) :
  (f a b 2 = 6) → (f a b (-2) = -14) := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_fourteen_l163_16304


namespace NUMINAMATH_CALUDE_candy_bar_total_cost_l163_16316

/-- The cost of a candy bar in dollars -/
def candy_bar_cost : ℕ := 3

/-- The number of candy bars bought -/
def number_of_candy_bars : ℕ := 2

/-- The total cost of candy bars -/
def total_cost : ℕ := candy_bar_cost * number_of_candy_bars

theorem candy_bar_total_cost : total_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_total_cost_l163_16316


namespace NUMINAMATH_CALUDE_curves_intersect_once_l163_16324

/-- Two curves intersect at exactly one point -/
def intersect_once (f g : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = g x

/-- The first curve -/
def curve1 (b : ℝ) (x : ℝ) : ℝ := b * x^2 - 2 * x + 5

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 3 * x + 4

/-- The theorem stating the condition for the curves to intersect at exactly one point -/
theorem curves_intersect_once :
  ∀ b : ℝ, intersect_once (curve1 b) curve2 ↔ b = 25/4 := by sorry

end NUMINAMATH_CALUDE_curves_intersect_once_l163_16324


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l163_16399

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 8) (hprod : x * y = 12) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 8 → a * b = 12 → 1/x + 1/y ≤ 1/a + 1/b) ∧ 
  1/x + 1/y = 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l163_16399


namespace NUMINAMATH_CALUDE_max_bookshelves_l163_16317

def room_space : ℕ := 400
def shelf_space : ℕ := 80
def reserved_space : ℕ := 160

theorem max_bookshelves : 
  (room_space - reserved_space) / shelf_space = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_bookshelves_l163_16317


namespace NUMINAMATH_CALUDE_range_of_f_l163_16342

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, x ≠ -2 ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l163_16342


namespace NUMINAMATH_CALUDE_tom_marble_pairs_l163_16330

/-- The number of distinct pairs of marbles Tom can choose -/
def distinct_pairs : ℕ := 8

/-- The number of red marbles Tom has -/
def red_marbles : ℕ := 1

/-- The number of blue marbles Tom has -/
def blue_marbles : ℕ := 1

/-- The number of yellow marbles Tom has -/
def yellow_marbles : ℕ := 4

/-- The number of green marbles Tom has -/
def green_marbles : ℕ := 2

/-- Theorem stating that the number of distinct pairs of marbles Tom can choose is 8 -/
theorem tom_marble_pairs :
  distinct_pairs = 8 :=
by sorry

end NUMINAMATH_CALUDE_tom_marble_pairs_l163_16330
