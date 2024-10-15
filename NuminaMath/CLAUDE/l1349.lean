import Mathlib

namespace NUMINAMATH_CALUDE_vasya_toy_choices_l1349_134989

/-- The number of different remote-controlled cars available -/
def num_cars : ℕ := 7

/-- The number of different construction sets available -/
def num_sets : ℕ := 5

/-- The total number of toys available -/
def total_toys : ℕ := num_cars + num_sets

/-- The number of toys Vasya can choose -/
def toys_to_choose : ℕ := 2

theorem vasya_toy_choices :
  Nat.choose total_toys toys_to_choose = 66 :=
sorry

end NUMINAMATH_CALUDE_vasya_toy_choices_l1349_134989


namespace NUMINAMATH_CALUDE_initial_money_calculation_l1349_134984

/-- Calculates the initial amount of money given the cost of bread, peanut butter, and the amount left over --/
theorem initial_money_calculation (bread_cost : ℝ) (bread_quantity : ℕ) (peanut_butter_cost : ℝ) (money_left : ℝ) : 
  bread_cost = 2.25 →
  bread_quantity = 3 →
  peanut_butter_cost = 2 →
  money_left = 5.25 →
  bread_cost * (bread_quantity : ℝ) + peanut_butter_cost + money_left = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l1349_134984


namespace NUMINAMATH_CALUDE_completing_square_form_l1349_134970

theorem completing_square_form (x : ℝ) :
  x^2 - 2*x - 1 = 0 ↔ (x - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_form_l1349_134970


namespace NUMINAMATH_CALUDE_max_sin_angle_ellipse_l1349_134966

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def is_focus (F : ℝ × ℝ) (a b : ℝ) : Prop :=
  F.1^2 + F.2^2 = a^2 - b^2 ∧ a > b ∧ a > 0 ∧ b > 0

def angle_sin (A B C : ℝ × ℝ) : ℝ := sorry

theorem max_sin_angle_ellipse :
  ∃ (a b : ℝ) (F₁ F₂ : ℝ × ℝ),
    a = 3 ∧ b = Real.sqrt 5 ∧
    is_focus F₁ a b ∧ is_focus F₂ a b ∧
    (∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
      angle_sin F₁ P F₂ ≤ 4 * Real.sqrt 5 / 9) ∧
    (∃ (P : ℝ × ℝ), ellipse P.1 P.2 ∧
      angle_sin F₁ P F₂ = 4 * Real.sqrt 5 / 9) :=
sorry

end NUMINAMATH_CALUDE_max_sin_angle_ellipse_l1349_134966


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l1349_134908

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2} ∪ {x : ℝ | x ≥ 3/2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x > a^2 - x^2 + 2*x} = {a : ℝ | -Real.sqrt 5 < a ∧ a < Real.sqrt 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_a_l1349_134908


namespace NUMINAMATH_CALUDE_train_speed_l1349_134943

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : Real) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235.03)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45.0036 := by
  sorry

#eval (140 + 235.03) / 30 * 3.6

end NUMINAMATH_CALUDE_train_speed_l1349_134943


namespace NUMINAMATH_CALUDE_larger_number_proof_l1349_134931

theorem larger_number_proof (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 23 ∧
  ∃ (x y : ℕ), x * y = Nat.lcm a b ∧ x = 13 ∧ y = 14 →
  max a b = 322 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1349_134931


namespace NUMINAMATH_CALUDE_triangle_preserves_triangle_parallelogram_preserves_parallelogram_square_not_always_square_rhombus_not_always_rhombus_l1349_134987

-- Define the oblique projection method
structure ObliqueProjection where
  -- Add necessary fields for oblique projection

-- Define geometric shapes
structure Triangle where
  -- Add necessary fields for triangle

structure Parallelogram where
  -- Add necessary fields for parallelogram

structure Square where
  -- Add necessary fields for square

structure Rhombus where
  -- Add necessary fields for rhombus

-- Define the intuitive diagram function
def intuitiveDiagram (op : ObliqueProjection) (shape : Type) : Type :=
  sorry

-- Theorem statements
theorem triangle_preserves_triangle (op : ObliqueProjection) (t : Triangle) :
  intuitiveDiagram op Triangle = Triangle :=
sorry

theorem parallelogram_preserves_parallelogram (op : ObliqueProjection) (p : Parallelogram) :
  intuitiveDiagram op Parallelogram = Parallelogram :=
sorry

theorem square_not_always_square (op : ObliqueProjection) :
  ¬(∀ (s : Square), intuitiveDiagram op Square = Square) :=
sorry

theorem rhombus_not_always_rhombus (op : ObliqueProjection) :
  ¬(∀ (r : Rhombus), intuitiveDiagram op Rhombus = Rhombus) :=
sorry

end NUMINAMATH_CALUDE_triangle_preserves_triangle_parallelogram_preserves_parallelogram_square_not_always_square_rhombus_not_always_rhombus_l1349_134987


namespace NUMINAMATH_CALUDE_trig_inequality_l1349_134990

theorem trig_inequality (a b c x : ℝ) :
  -(Real.sin (π/4 - (b-c)/2))^2 ≤ Real.sin (a*x + b) * Real.cos (a*x + c) ∧
  Real.sin (a*x + b) * Real.cos (a*x + c) ≤ (Real.cos (π/4 - (b-c)/2))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1349_134990


namespace NUMINAMATH_CALUDE_charitable_woman_purse_l1349_134923

/-- The charitable woman's purse problem -/
theorem charitable_woman_purse (P : ℚ) : 
  (P > 0) →
  (P - ((1/2) * P + 1) - ((1/2) * ((1/2) * P - 1) + 2) - ((1/2) * ((1/4) * P - 2.5) + 3) = 1) →
  P = 42 := by
  sorry

end NUMINAMATH_CALUDE_charitable_woman_purse_l1349_134923


namespace NUMINAMATH_CALUDE_elena_pen_purchase_l1349_134973

/-- The number of brand X pens Elena purchased -/
def brand_x_pens : ℕ := 9

/-- The number of brand Y pens Elena purchased -/
def brand_y_pens : ℕ := 12 - brand_x_pens

/-- The cost of a single brand X pen -/
def cost_x : ℚ := 4

/-- The cost of a single brand Y pen -/
def cost_y : ℚ := 2.2

/-- The total cost of all pens -/
def total_cost : ℚ := 42

theorem elena_pen_purchase :
  (brand_x_pens : ℚ) * cost_x + (brand_y_pens : ℚ) * cost_y = total_cost ∧
  brand_x_pens + brand_y_pens = 12 :=
by sorry

end NUMINAMATH_CALUDE_elena_pen_purchase_l1349_134973


namespace NUMINAMATH_CALUDE_root_product_equals_27_l1349_134996

theorem root_product_equals_27 : 
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_27_l1349_134996


namespace NUMINAMATH_CALUDE_elect_representatives_l1349_134920

theorem elect_representatives (total_students : ℕ) (girls : ℕ) (representatives : ℕ) 
  (h1 : total_students = 10) 
  (h2 : girls = 3) 
  (h3 : representatives = 2) : 
  (Nat.choose total_students representatives - Nat.choose (total_students - girls) representatives) = 48 :=
sorry

end NUMINAMATH_CALUDE_elect_representatives_l1349_134920


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l1349_134991

/-- The cost of a single soda, given the total cost of sandwiches and sodas, and the cost of a single sandwich. -/
def soda_cost (total_cost sandwich_cost : ℚ) : ℚ :=
  (total_cost - 2 * sandwich_cost) / 4

theorem soda_cost_calculation (total_cost sandwich_cost : ℚ) 
  (h1 : total_cost = (8.36 : ℚ))
  (h2 : sandwich_cost = (2.44 : ℚ)) :
  soda_cost total_cost sandwich_cost = (0.87 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l1349_134991


namespace NUMINAMATH_CALUDE_solve_for_y_l1349_134900

theorem solve_for_y (x y : ℝ) : 4 * x + y = 9 → y = 9 - 4 * x := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1349_134900


namespace NUMINAMATH_CALUDE_ellipse_equation_l1349_134965

/-- An ellipse with the given properties has the equation x²/2 + 3y²/2 = 1 or 3x²/2 + y²/2 = 1 -/
theorem ellipse_equation (E : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ E ↔ ∃ (m n : ℝ), m * x^2 + n * y^2 = 1) →  -- E is an ellipse
  (0, 0) ∈ E →  -- center at origin
  (∃ (a : ℝ), (a, 0) ∈ E ∧ (-a, 0) ∈ E) ∨ (∃ (b : ℝ), (0, b) ∈ E ∧ (0, -b) ∈ E) →  -- foci on coordinate axis
  (∃ (x : ℝ), P = (x, x + 1) ∧ Q = (x, x + 1) ∧ P ∈ E ∧ Q ∈ E) →  -- P and Q on y = x + 1 and on E
  P.1 * Q.1 + P.2 * Q.2 = 0 →  -- OP · OQ = 0
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 5/2 →  -- |PQ|² = (√10/2)² = 5/2
  (∀ (x y : ℝ), (x, y) ∈ E ↔ x^2/2 + 3*y^2/2 = 1) ∨
  (∀ (x y : ℝ), (x, y) ∈ E ↔ 3*x^2/2 + y^2/2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1349_134965


namespace NUMINAMATH_CALUDE_mall_product_properties_l1349_134981

/-- Represents the shopping mall's product pricing and sales model -/
structure ProductModel where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_volume : ℝ → ℝ
  profit : ℝ → ℝ

/-- The specific product model for the shopping mall -/
def mall_product : ProductModel :=
  { purchase_price := 30
    min_selling_price := 30
    max_selling_price := 55
    sales_volume := λ x => -2 * x + 140
    profit := λ x => (x - 30) * (-2 * x + 140) }

theorem mall_product_properties (x : ℝ) :
  let m := mall_product
  (x ≥ m.min_selling_price ∧ x ≤ m.max_selling_price) →
  (m.profit 35 = 350 ∧
   m.profit 40 = 600 ∧
   ∀ y, m.min_selling_price ≤ y ∧ y ≤ m.max_selling_price → m.profit y ≠ 900) :=
by sorry


end NUMINAMATH_CALUDE_mall_product_properties_l1349_134981


namespace NUMINAMATH_CALUDE_remainder_of_1725_base14_div_9_l1349_134932

/-- Converts a base-14 number to decimal --/
def base14ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 14 + d) 0

/-- The base-14 representation of 1725₁₄ --/
def number : List Nat := [1, 7, 2, 5]

theorem remainder_of_1725_base14_div_9 :
  (base14ToDecimal number) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1725_base14_div_9_l1349_134932


namespace NUMINAMATH_CALUDE_correct_calculation_l1349_134998

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1349_134998


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1349_134976

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x^2 ∧ 6*s^2 = 4*x) → x = 1/216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1349_134976


namespace NUMINAMATH_CALUDE_prob_at_least_one_value_l1349_134955

/-- The probability of picking a road from A to B that is at least 5 miles long -/
def prob_AB : ℚ := 2/3

/-- The probability of picking a road from B to C that is at least 5 miles long -/
def prob_BC : ℚ := 3/4

/-- The probability that at least one of the randomly picked roads (one from A to B, one from B to C) is at least 5 miles long -/
def prob_at_least_one : ℚ := 1 - (1 - prob_AB) * (1 - prob_BC)

theorem prob_at_least_one_value : prob_at_least_one = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_value_l1349_134955


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1349_134917

theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (number of years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N > 0) →  -- Sum of children's ages N years ago was positive
  (T - N = 3 * (T - 4*N)) →  -- Condition about Tom's age N years ago
  T / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1349_134917


namespace NUMINAMATH_CALUDE_quartic_factorization_and_solutions_l1349_134992

theorem quartic_factorization_and_solutions :
  ∃ (x₁ x₂ x₃ x₄ : ℂ),
    (∀ x : ℂ, x^4 + 1 = (x^2 + Real.sqrt 2 * x + 1) * (x^2 - Real.sqrt 2 * x + 1)) ∧
    x₁ = -Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2 ∧
    x₂ = -Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∧
    x₃ = Real.sqrt 2 / 2 + Complex.I * Real.sqrt 2 / 2 ∧
    x₄ = Real.sqrt 2 / 2 - Complex.I * Real.sqrt 2 / 2 ∧
    {x | x^4 + 1 = 0} = {x₁, x₂, x₃, x₄} := by
  sorry

end NUMINAMATH_CALUDE_quartic_factorization_and_solutions_l1349_134992


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1349_134954

-- Define a pentagon as a polygon with 5 sides
def Pentagon : Nat := 5

-- Theorem stating that the sum of interior angles of a pentagon is 540 degrees
theorem sum_interior_angles_pentagon :
  (Pentagon - 2) * 180 = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1349_134954


namespace NUMINAMATH_CALUDE_parabola_transformation_l1349_134983

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b
    c := p.c + v }

/-- The original parabola y = x^2 + 2 -/
def original_parabola : Parabola :=
  { a := 1
    b := 0
    c := 2 }

theorem parabola_transformation :
  let p1 := shift_horizontal original_parabola (-1)
  let p2 := shift_vertical p1 (-1)
  p2 = { a := 1, b := 2, c := 1 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1349_134983


namespace NUMINAMATH_CALUDE_problem_statement_l1349_134950

theorem problem_statement (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) 
  (h3 : ¬p) : 
  ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1349_134950


namespace NUMINAMATH_CALUDE_sum_of_angles_l1349_134913

-- Define a rectangle
structure Rectangle where
  angles : ℕ
  is_rectangle : angles = 4

-- Define a square
structure Square where
  angles : ℕ
  is_square : angles = 4

-- Theorem statement
theorem sum_of_angles (rect : Rectangle) (sq : Square) : rect.angles + sq.angles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l1349_134913


namespace NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l1349_134919

/-- The number of balls --/
def n : ℕ := 25

/-- The number of bins --/
def m : ℕ := 6

/-- The number of ways to distribute n balls into m bins --/
def total_distributions : ℕ := Nat.choose (n + m - 1) n

/-- The number of ways to distribute balls according to the 5-5-3-3-2-2 pattern --/
def ways_p : ℕ := Nat.choose n 5 * Nat.choose 20 5 * Nat.choose 15 3 * Nat.choose 12 3 * Nat.choose 9 2 * Nat.choose 7 2

/-- The number of ways to distribute balls equally (4-4-4-4-4-5 pattern) --/
def ways_q : ℕ := Nat.choose n 4 * Nat.choose 21 4 * Nat.choose 17 4 * Nat.choose 13 4 * Nat.choose 9 4 * Nat.choose 5 5

/-- The probability of the 5-5-3-3-2-2 distribution --/
def p : ℚ := ways_p / total_distributions

/-- The probability of the equal distribution --/
def q : ℚ := ways_q / total_distributions

theorem probability_ratio_equals_ways_ratio : p / q = ways_p / ways_q := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l1349_134919


namespace NUMINAMATH_CALUDE_katie_mp3_songs_l1349_134964

theorem katie_mp3_songs (initial : ℕ) (deleted : ℕ) (added : ℕ) (final : ℕ) : 
  initial = 11 → deleted = 7 → added = 24 → final = initial - deleted + added → final = 28 := by
  sorry

end NUMINAMATH_CALUDE_katie_mp3_songs_l1349_134964


namespace NUMINAMATH_CALUDE_middle_circle_radius_l1349_134994

/-- Given three circles in a geometric sequence with radii r₁, r₂, and r₃,
    where r₁ = 5 cm and r₃ = 20 cm, prove that r₂ = 10 cm. -/
theorem middle_circle_radius (r₁ r₂ r₃ : ℝ) 
    (h_geom_seq : r₂^2 = r₁ * r₃)
    (h_r₁ : r₁ = 5)
    (h_r₃ : r₃ = 20) : 
  r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l1349_134994


namespace NUMINAMATH_CALUDE_pet_store_cats_l1349_134930

theorem pet_store_cats (initial_siamese : ℕ) (sold : ℕ) (remaining : ℕ) (initial_house : ℕ) : 
  initial_siamese = 13 → 
  sold = 10 → 
  remaining = 8 → 
  initial_siamese + initial_house - sold = remaining → 
  initial_house = 5 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cats_l1349_134930


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1349_134993

theorem largest_divisor_of_difference_of_squares (m n : ℤ) :
  (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) →  -- m and n are even
  n < m →                              -- n is less than m
  (∀ x : ℤ, x ∣ (m^2 - n^2) → x ≤ 8) ∧ -- 8 is an upper bound for divisors
  8 ∣ (m^2 - n^2)                      -- 8 divides m^2 - n^2
  := by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1349_134993


namespace NUMINAMATH_CALUDE_sqrt_not_defined_for_negative_one_l1349_134914

theorem sqrt_not_defined_for_negative_one :
  ¬ (∃ (y : ℝ), y^2 = -1) :=
sorry

end NUMINAMATH_CALUDE_sqrt_not_defined_for_negative_one_l1349_134914


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l1349_134901

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 2) : ⌈x⌉ - x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l1349_134901


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1349_134945

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1349_134945


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1349_134978

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  (GeometricSequence a q) →
  (¬(q > 1 → IncreasingSequence a) ∧ ¬(IncreasingSequence a → q > 1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l1349_134978


namespace NUMINAMATH_CALUDE_complex_fraction_real_condition_l1349_134957

theorem complex_fraction_real_condition (a : ℝ) : 
  (((1 : ℂ) + 2 * I) / (a + I)).im = 0 ↔ a = (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_real_condition_l1349_134957


namespace NUMINAMATH_CALUDE_sum_of_sequences_l1349_134927

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_sequences : 
  (arithmetic_sum 2 10 5) + (arithmetic_sum 10 10 5) = 260 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l1349_134927


namespace NUMINAMATH_CALUDE_possible_m_values_l1349_134982

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l1349_134982


namespace NUMINAMATH_CALUDE_grandpa_water_distribution_l1349_134995

/-- The number of water bottles Grandpa has -/
def num_bottles : ℕ := 12

/-- The volume of each water bottle in liters -/
def bottle_volume : ℚ := 3

/-- The volume of water to be distributed to each student in liters -/
def student_share : ℚ := 3/4

/-- The number of students Grandpa can share water with -/
def num_students : ℕ := 48

theorem grandpa_water_distribution :
  (↑num_bottles * bottle_volume) / student_share = num_students := by
  sorry

end NUMINAMATH_CALUDE_grandpa_water_distribution_l1349_134995


namespace NUMINAMATH_CALUDE_M_intersect_N_l1349_134907

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem M_intersect_N : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1349_134907


namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1349_134969

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_6_8_10 :
  is_pythagorean_triple 6 8 10 ∧
  ¬ is_pythagorean_triple 1 1 2 ∧
  ¬ is_pythagorean_triple 1 2 2 ∧
  ¬ is_pythagorean_triple 5 12 15 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l1349_134969


namespace NUMINAMATH_CALUDE_h_composition_equals_902_l1349_134903

/-- The function h as defined in the problem -/
def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

/-- Theorem stating that h(h(2)) = 902 -/
theorem h_composition_equals_902 : h (h 2) = 902 := by
  sorry

end NUMINAMATH_CALUDE_h_composition_equals_902_l1349_134903


namespace NUMINAMATH_CALUDE_inheritance_calculation_l1349_134921

theorem inheritance_calculation (x : ℝ) 
  (h1 : x > 0)
  (h2 : 0.3 * x + 0.12 * (0.7 * x) + 0.05 * (0.7 * x - 0.12 * (0.7 * x)) = 16800) :
  x = 40500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l1349_134921


namespace NUMINAMATH_CALUDE_unique_intersection_values_l1349_134928

-- Define the complex plane
variable (z : ℂ)

-- Define the condition from the original problem
def intersection_condition (k : ℝ) : Prop :=
  ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k

-- State the theorem
theorem unique_intersection_values :
  ∀ k : ℝ, intersection_condition k ↔ (k = 13 - Real.sqrt 153 ∨ k = 13 + Real.sqrt 153) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_values_l1349_134928


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l1349_134935

theorem max_value_x_sqrt_3_minus_x_squared :
  ∀ x : ℝ, 0 < x → x < Real.sqrt 3 →
    x * Real.sqrt (3 - x^2) ≤ 9/4 ∧
    ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < Real.sqrt 3 ∧ x₀ * Real.sqrt (3 - x₀^2) = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l1349_134935


namespace NUMINAMATH_CALUDE_processing_400_parts_l1349_134963

/-- Linear regression function for processing time -/
def processingTime (x : ℝ) : ℝ := 0.2 * x + 3

/-- Theorem: Processing 400 parts takes 83 hours -/
theorem processing_400_parts : processingTime 400 = 83 := by
  sorry

end NUMINAMATH_CALUDE_processing_400_parts_l1349_134963


namespace NUMINAMATH_CALUDE_special_number_value_l1349_134974

/-- A number with specified digits in certain decimal places -/
def SpecialNumber : ℝ :=
  60 + 0.06

/-- Proof that the SpecialNumber is equal to 60.06 -/
theorem special_number_value : SpecialNumber = 60.06 := by
  sorry

#check special_number_value

end NUMINAMATH_CALUDE_special_number_value_l1349_134974


namespace NUMINAMATH_CALUDE_base8_subtraction_l1349_134951

-- Define a function to convert base 8 to decimal
def base8ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert decimal to base 8
def decimalToBase8 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction : base8ToDecimal 52 - base8ToDecimal 27 = base8ToDecimal 23 := by
  sorry

end NUMINAMATH_CALUDE_base8_subtraction_l1349_134951


namespace NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l1349_134949

theorem probability_of_drawing_red_ball (white_balls red_balls : ℕ) 
  (h1 : white_balls = 5) (h2 : red_balls = 2) : 
  (red_balls : ℚ) / (white_balls + red_balls) = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_red_ball_l1349_134949


namespace NUMINAMATH_CALUDE_orange_buckets_problem_l1349_134971

/-- The problem of calculating the number of oranges and their total weight -/
theorem orange_buckets_problem :
  let bucket1 : ℝ := 22.5
  let bucket2 : ℝ := 2 * bucket1 + 3
  let bucket3 : ℝ := bucket2 - 11.5
  let bucket4 : ℝ := 1.5 * (bucket1 + bucket3)
  let weight1 : ℝ := 0.3
  let weight3 : ℝ := 0.4
  let weight4 : ℝ := 0.35
  let total_oranges : ℝ := bucket1 + bucket2 + bucket3 + bucket4
  let total_weight : ℝ := weight1 * bucket1 + weight3 * bucket3 + weight4 * bucket4
  total_oranges = 195.5 ∧ total_weight = 52.325 := by
  sorry


end NUMINAMATH_CALUDE_orange_buckets_problem_l1349_134971


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1349_134959

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 8 x = Nat.choose 8 (2*x - 1)) → (x = 1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1349_134959


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1365_l1349_134968

theorem sum_largest_smallest_prime_factors_1365 : ∃ (p q : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  p ∣ 1365 ∧ 
  q ∣ 1365 ∧ 
  (∀ r : ℕ, Nat.Prime r → r ∣ 1365 → p ≤ r ∧ r ≤ q) ∧ 
  p + q = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_1365_l1349_134968


namespace NUMINAMATH_CALUDE_waiting_room_ratio_l1349_134938

theorem waiting_room_ratio : 
  ∀ (initial_waiting : ℕ) (arrivals : ℕ) (interview : ℕ),
    initial_waiting = 22 →
    arrivals = 3 →
    (initial_waiting + arrivals) % interview = 0 →
    interview ≠ 1 →
    interview < initial_waiting + arrivals →
    (initial_waiting + arrivals) / interview = 5 :=
by sorry

end NUMINAMATH_CALUDE_waiting_room_ratio_l1349_134938


namespace NUMINAMATH_CALUDE_sum_reciprocal_product_bound_sum_product_bound_l1349_134988

-- Part (a)
theorem sum_reciprocal_product_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by sorry

-- Part (b)
theorem sum_product_bound (u v : ℝ) (hu : 0 < u ∧ u < 1) (hv : 0 < v ∧ v < 1) :
  0 < u + v - u*v ∧ u + v - u*v < 1 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_product_bound_sum_product_bound_l1349_134988


namespace NUMINAMATH_CALUDE_f_monotonicity_l1349_134925

noncomputable def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem f_monotonicity :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_l1349_134925


namespace NUMINAMATH_CALUDE_negative_quarter_power_times_four_power_l1349_134911

theorem negative_quarter_power_times_four_power (n : ℕ) :
  ((-0.25 : ℝ) ^ n) * (4 ^ (n + 1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_quarter_power_times_four_power_l1349_134911


namespace NUMINAMATH_CALUDE_ellipse_focus_coordinates_l1349_134979

/-- Given an ellipse with specified major and minor axis endpoints, 
    prove that the focus with the smaller y-coordinate has coordinates (5 - √5, 2) -/
theorem ellipse_focus_coordinates 
  (major_endpoint1 : ℝ × ℝ)
  (major_endpoint2 : ℝ × ℝ)
  (minor_endpoint1 : ℝ × ℝ)
  (minor_endpoint2 : ℝ × ℝ)
  (h1 : major_endpoint1 = (2, 2))
  (h2 : major_endpoint2 = (8, 2))
  (h3 : minor_endpoint1 = (5, 4))
  (h4 : minor_endpoint2 = (5, 0)) :
  ∃ (focus : ℝ × ℝ), focus = (5 - Real.sqrt 5, 2) ∧ 
  (∀ (other_focus : ℝ × ℝ), other_focus.2 ≤ focus.2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_coordinates_l1349_134979


namespace NUMINAMATH_CALUDE_license_plate_count_l1349_134997

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in a license plate -/
def num_plate_letters : ℕ := 2

/-- The number of possible positions for the letter block -/
def num_letter_block_positions : ℕ := num_plate_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := num_digits ^ num_plate_digits * (num_letters ^ num_plate_letters) * num_letter_block_positions

theorem license_plate_count :
  total_license_plates = 40560000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1349_134997


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1349_134962

/-- A rhombus with given diagonal lengths has the specified perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 52 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1349_134962


namespace NUMINAMATH_CALUDE_ellipse_string_length_l1349_134948

/-- Represents an ellipse with semi-major axis 'a' and semi-minor axis 'b' --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- The length of the string used in the pin-and-string method for drawing an ellipse --/
def string_length (e : Ellipse) : ℝ := 2 * e.a

/-- Theorem stating that for an ellipse with semi-major axis 6 cm and semi-minor axis 4 cm,
    the length of the string used in the pin-and-string method is 12 cm --/
theorem ellipse_string_length :
  let e : Ellipse := ⟨6, 4, by norm_num, by norm_num, by norm_num⟩
  string_length e = 12 := by sorry

end NUMINAMATH_CALUDE_ellipse_string_length_l1349_134948


namespace NUMINAMATH_CALUDE_power_function_properties_l1349_134904

noncomputable def f (x : ℝ) : ℝ := x^(2/3)

theorem power_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, f x = x^a) ∧ 
  f 8 = 4 ∧
  f 0 = 0 ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y < 0 → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_power_function_properties_l1349_134904


namespace NUMINAMATH_CALUDE_total_hours_is_fifty_l1349_134999

/-- Represents the worker's pay structure and work week --/
structure WorkWeek where
  ordinary_rate : ℚ  -- Rate for ordinary time in dollars per hour
  overtime_rate : ℚ  -- Rate for overtime in dollars per hour
  total_pay : ℚ      -- Total pay for the week in dollars
  overtime_hours : ℕ  -- Number of overtime hours worked

/-- Calculates the total hours worked given a WorkWeek --/
def total_hours (w : WorkWeek) : ℚ :=
  let ordinary_hours := (w.total_pay - w.overtime_rate * w.overtime_hours) / w.ordinary_rate
  ordinary_hours + w.overtime_hours

/-- Theorem stating that given the specific conditions, the total hours worked is 50 --/
theorem total_hours_is_fifty : 
  ∀ (w : WorkWeek), 
    w.ordinary_rate = 0.60 ∧ 
    w.overtime_rate = 0.90 ∧ 
    w.total_pay = 32.40 ∧ 
    w.overtime_hours = 8 → 
    total_hours w = 50 :=
by
  sorry


end NUMINAMATH_CALUDE_total_hours_is_fifty_l1349_134999


namespace NUMINAMATH_CALUDE_complex_subtraction_l1349_134980

theorem complex_subtraction : (7 - 3*Complex.I) - (2 + 5*Complex.I) = 5 - 8*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1349_134980


namespace NUMINAMATH_CALUDE_max_n_for_positive_an_l1349_134934

theorem max_n_for_positive_an (n : ℕ) : 
  (∀ k : ℕ, k > n → (19 : ℤ) - 2 * k ≤ 0) ∧ 
  ((19 : ℤ) - 2 * n > 0) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_max_n_for_positive_an_l1349_134934


namespace NUMINAMATH_CALUDE_total_raised_is_100_l1349_134926

/-- The amount of money raised by a local business for charity -/
def total_raised (num_tickets : ℕ) (ticket_price : ℚ) (donation_15 : ℕ) (donation_20 : ℕ) : ℚ :=
  num_tickets * ticket_price + donation_15 * 15 + donation_20 * 20

/-- Proof that the total amount raised is $100.00 -/
theorem total_raised_is_100 : total_raised 25 2 2 1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_raised_is_100_l1349_134926


namespace NUMINAMATH_CALUDE_dress_price_difference_l1349_134986

/-- Proves that the final price of a dress is $2.3531875 more than the original price
    given specific discounts, increases, and taxes. -/
theorem dress_price_difference (original_price : ℝ) : 
  original_price * 0.85 = 78.2 →
  let price_after_sale := 78.2
  let price_after_increase := price_after_sale * 1.25
  let price_after_coupon := price_after_increase * 0.9
  let final_price := price_after_coupon * 1.0725
  final_price - original_price = 2.3531875 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_difference_l1349_134986


namespace NUMINAMATH_CALUDE_gasoline_needed_for_distance_l1349_134933

/-- Given a car with fuel efficiency and a known fuel consumption for a specific distance,
    calculate the amount of gasoline needed for any distance. -/
theorem gasoline_needed_for_distance (fuel_efficiency : ℝ) (known_distance : ℝ) (known_gasoline : ℝ) (distance : ℝ) :
  fuel_efficiency = 20 →
  known_distance = 130 →
  known_gasoline = 6.5 →
  known_distance / known_gasoline = fuel_efficiency →
  distance / fuel_efficiency = distance / 20 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_needed_for_distance_l1349_134933


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l1349_134909

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  boys_soccer_percentage = 82 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 63 := by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l1349_134909


namespace NUMINAMATH_CALUDE_log_equation_holds_l1349_134977

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_holds_l1349_134977


namespace NUMINAMATH_CALUDE_fraction_equation_l1349_134939

theorem fraction_equation : ∃ (A B C : ℤ), 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by
  -- We claim that A = 500, B = -1, C = -500 satisfy the equation
  use 500, -1, -500
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_fraction_equation_l1349_134939


namespace NUMINAMATH_CALUDE_amp_composition_l1349_134940

-- Define the & operations
def postfix_amp (x : ℤ) : ℤ := 8 - x
def prefix_amp (x : ℤ) : ℤ := x - 8

-- The theorem to prove
theorem amp_composition : prefix_amp (postfix_amp 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l1349_134940


namespace NUMINAMATH_CALUDE_product_of_even_or_odd_is_even_l1349_134929

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the product of two functions
def FunctionProduct (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x * g x

-- State the theorem
theorem product_of_even_or_odd_is_even 
  (f φ : ℝ → ℝ) 
  (h : (IsEven f ∧ IsEven φ) ∨ (IsOdd f ∧ IsOdd φ)) : 
  IsEven (FunctionProduct f φ) := by
  sorry

end NUMINAMATH_CALUDE_product_of_even_or_odd_is_even_l1349_134929


namespace NUMINAMATH_CALUDE_projection_sphere_existence_l1349_134936

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for a set of lines
def LineSet := List Line3D

-- Function to check if lines are pairwise non-parallel
def pairwiseNonParallel (lines : LineSet) : Prop := sorry

-- Function to perform orthogonal projection
def orthogonalProject (p : Point3D) (l : Line3D) : Point3D := sorry

-- Function to generate all points from repeated projections
def allProjectionPoints (o : Point3D) (lines : LineSet) : Set Point3D := sorry

-- Theorem statement
theorem projection_sphere_existence 
  (o : Point3D) 
  (lines : LineSet) 
  (h : pairwiseNonParallel lines) :
  ∃ (r : ℝ), ∀ p ∈ allProjectionPoints o lines, 
    (p.x - o.x)^2 + (p.y - o.y)^2 + (p.z - o.z)^2 ≤ r^2 := by
  sorry

end NUMINAMATH_CALUDE_projection_sphere_existence_l1349_134936


namespace NUMINAMATH_CALUDE_greatest_rope_piece_length_l1349_134958

theorem greatest_rope_piece_length : Nat.gcd 48 (Nat.gcd 60 72) = 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_rope_piece_length_l1349_134958


namespace NUMINAMATH_CALUDE_complex_square_roots_l1349_134944

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -115 + 66 * I ↔ z = 3 + 11 * I ∨ z = -3 - 11 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l1349_134944


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l1349_134941

-- Define the triangle RWU
structure Triangle (R W U : Type) where
  angle_SWR : ℝ  -- Exterior angle
  angle_WRU : ℝ  -- Interior angle
  angle_WUR : ℝ  -- Interior angle (to be proved)
  straight_line : Prop  -- RTQU forms a straight line

-- State the theorem
theorem exterior_angle_theorem 
  (t : Triangle R W U) 
  (h1 : t.angle_SWR = 50)
  (h2 : t.angle_WRU = 30)
  (h3 : t.straight_line) : 
  t.angle_WUR = 20 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l1349_134941


namespace NUMINAMATH_CALUDE_new_device_improvement_l1349_134960

/-- Represents the sample mean and variance of a device's measurements -/
structure DeviceStats where
  mean : ℝ
  variance : ℝ

/-- Determines if there's significant improvement based on the given criterion -/
def significantImprovement (old new : DeviceStats) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

/-- The old device's statistics -/
def oldDevice : DeviceStats :=
  { mean := 10.3, variance := 0.04 }

/-- The new device's statistics -/
def newDevice : DeviceStats :=
  { mean := 10, variance := 0.036 }

/-- Theorem stating that the new device shows significant improvement -/
theorem new_device_improvement :
  significantImprovement oldDevice newDevice := by
  sorry

#check new_device_improvement

end NUMINAMATH_CALUDE_new_device_improvement_l1349_134960


namespace NUMINAMATH_CALUDE_same_tangent_line_implies_b_value_l1349_134967

def f (x : ℝ) : ℝ := 2 * x^3 + 1
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - b

theorem same_tangent_line_implies_b_value :
  ∀ b : ℝ, (∃ x₀ : ℝ, (deriv f x₀ = deriv (g b) x₀) ∧ 
    (f x₀ = g b x₀)) → (b = 0 ∨ b = -1) :=
by sorry

end NUMINAMATH_CALUDE_same_tangent_line_implies_b_value_l1349_134967


namespace NUMINAMATH_CALUDE_sixth_power_sum_l1349_134952

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l1349_134952


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1349_134947

theorem polynomial_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  let p := (10 * d + 17 + 12 * d^2) + (6 * d + 3)
  ∃ a b c : ℤ, p = a * d + b + c * d^2 ∧ a + b + c = 48 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1349_134947


namespace NUMINAMATH_CALUDE_sampling_methods_appropriateness_l1349_134953

/-- Represents a sampling scenario with a population size and sample size -/
structure SamplingScenario where
  populationSize : Nat
  sampleSize : Nat

/-- Determines if simple random sampling is appropriate for a given scenario -/
def isSimpleRandomSamplingAppropriate (scenario : SamplingScenario) : Prop :=
  scenario.sampleSize ≤ scenario.populationSize ∧ scenario.sampleSize ≤ 10

/-- Determines if systematic sampling is appropriate for a given scenario -/
def isSystematicSamplingAppropriate (scenario : SamplingScenario) : Prop :=
  scenario.sampleSize > 10 ∧ scenario.populationSize ≥ 100

theorem sampling_methods_appropriateness :
  let scenario1 : SamplingScenario := ⟨10, 2⟩
  let scenario2 : SamplingScenario := ⟨1000, 50⟩
  isSimpleRandomSamplingAppropriate scenario1 ∧
  isSystematicSamplingAppropriate scenario2 :=
by sorry

end NUMINAMATH_CALUDE_sampling_methods_appropriateness_l1349_134953


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l1349_134956

/-- The difference in miles traveled between two cyclists after 3 hours -/
theorem cyclist_distance_difference (carlos_start : ℝ) (carlos_total : ℝ) (dana_total : ℝ) : 
  carlos_start = 5 → 
  carlos_total = 50 → 
  dana_total = 40 → 
  carlos_total - dana_total = 10 := by
  sorry

#check cyclist_distance_difference

end NUMINAMATH_CALUDE_cyclist_distance_difference_l1349_134956


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1349_134915

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | 1 < x ∧ x < 8}
  let B : Set ℝ := {1, 3, 5, 6, 7}
  A ∩ B = {3, 5, 6, 7} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1349_134915


namespace NUMINAMATH_CALUDE_magnitude_z_l1349_134912

theorem magnitude_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (25 * Real.sqrt 34) / 34 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_z_l1349_134912


namespace NUMINAMATH_CALUDE_fish_price_proof_l1349_134902

theorem fish_price_proof (discount_rate : ℝ) (discounted_price : ℝ) (package_weight : ℝ) :
  discount_rate = 0.4 →
  discounted_price = 2 →
  package_weight = 1/4 →
  (1 - discount_rate) * (1 / package_weight) * discounted_price = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_fish_price_proof_l1349_134902


namespace NUMINAMATH_CALUDE_jean_spots_ratio_l1349_134922

/-- Jean the jaguar's spot distribution --/
def jean_spots (total_spots upper_torso_spots side_spots : ℕ) : Prop :=
  total_spots = upper_torso_spots + side_spots ∧
  upper_torso_spots = 30 ∧
  side_spots = 10 ∧
  2 * upper_torso_spots = total_spots

theorem jean_spots_ratio :
  ∀ total_spots upper_torso_spots side_spots,
  jean_spots total_spots upper_torso_spots side_spots →
  (total_spots / 2 : ℚ) / total_spots = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jean_spots_ratio_l1349_134922


namespace NUMINAMATH_CALUDE_chord_equation_l1349_134937

/-- Given a circle and a chord, prove the equation of the chord. -/
theorem chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4*x - 5 = 0) →  -- Circle equation
  (∃ (a b : ℝ), (a + 3) / 2 = 3 ∧ (b + 1) / 2 = 1) →  -- Midpoint condition
  (x + y - 4 = 0) :=  -- Equation of line AB
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1349_134937


namespace NUMINAMATH_CALUDE_mobile_phone_cost_l1349_134975

def refrigerator_cost : ℝ := 15000
def refrigerator_loss_percent : ℝ := 0.04
def mobile_profit_percent : ℝ := 0.10
def overall_profit : ℝ := 200

theorem mobile_phone_cost (mobile_cost : ℝ) : 
  (refrigerator_cost * (1 - refrigerator_loss_percent) + mobile_cost * (1 + mobile_profit_percent)) - 
  (refrigerator_cost + mobile_cost) = overall_profit →
  mobile_cost = 6000 := by
sorry

end NUMINAMATH_CALUDE_mobile_phone_cost_l1349_134975


namespace NUMINAMATH_CALUDE_honey_lasts_16_days_l1349_134972

/-- Represents the number of days Tabitha can enjoy honey in her tea -/
def honey_days : ℕ :=
  let evening_servings : ℕ := 2 * 2  -- 2 cups with 2 servings each
  let morning_servings : ℕ := 1 * 1  -- 1 cup with 1 serving
  let afternoon_servings : ℕ := 1 * 1  -- 1 cup with 1 serving
  let daily_servings : ℕ := evening_servings + morning_servings + afternoon_servings
  let container_ounces : ℕ := 16
  let servings_per_ounce : ℕ := 6
  let total_servings : ℕ := container_ounces * servings_per_ounce
  total_servings / daily_servings

theorem honey_lasts_16_days : honey_days = 16 := by
  sorry

end NUMINAMATH_CALUDE_honey_lasts_16_days_l1349_134972


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1349_134905

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1349_134905


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1349_134985

theorem quadratic_factorization (x : ℝ) : 
  (x^2 - 4*x + 3 = (x-1)*(x-3)) ∧ 
  (4*x^2 + 12*x - 7 = (2*x+7)*(2*x-1)) := by
  sorry

#check quadratic_factorization

end NUMINAMATH_CALUDE_quadratic_factorization_l1349_134985


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1349_134942

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1349_134942


namespace NUMINAMATH_CALUDE_proportion_problem_l1349_134916

theorem proportion_problem :
  ∃ (a b c d : ℝ),
    (a / b = c / d) ∧
    (a + d = 14) ∧
    (b + c = 11) ∧
    (a^2 + b^2 + c^2 + d^2 = 221) ∧
    (a = 12 ∧ b = 8 ∧ c = 3 ∧ d = 2) := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1349_134916


namespace NUMINAMATH_CALUDE_max_ac_value_l1349_134961

theorem max_ac_value (a b c d : ℤ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : c > d) 
  (h4 : d ≥ -2021) 
  (h5 : (a + b) * (d + a) = (b + c) * (c + d)) 
  (h6 : b + c ≠ 0) 
  (h7 : d + a ≠ 0) : 
  a * c ≤ 510050 := by
  sorry

end NUMINAMATH_CALUDE_max_ac_value_l1349_134961


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1349_134946

theorem binomial_expansion_example : 104^3 + 3*(104^2)*2 + 3*104*(2^2) + 2^3 = 106^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1349_134946


namespace NUMINAMATH_CALUDE_trig_simplification_trig_value_given_tan_l1349_134906

/-- Proves that the given trigonometric expression simplifies to -1 --/
theorem trig_simplification : 
  (Real.sqrt (1 - 2 * Real.sin (10 * π / 180) * Real.cos (10 * π / 180))) / 
  (Real.sin (170 * π / 180) - Real.sqrt (1 - Real.sin (170 * π / 180) ^ 2)) = -1 := by
  sorry

/-- Proves that given tan θ = 2, the expression 2 + sin θ * cos θ - cos² θ equals 11/5 --/
theorem trig_value_given_tan (θ : Real) (h : Real.tan θ = 2) : 
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_trig_value_given_tan_l1349_134906


namespace NUMINAMATH_CALUDE_expression_simplification_l1349_134924

theorem expression_simplification (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1349_134924


namespace NUMINAMATH_CALUDE_faster_train_speed_l1349_134918

/-- The speed of the faster train given the conditions of the problem -/
def speed_of_faster_train (speed_difference : ℝ) (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  speed_difference * 2

/-- Theorem stating that the speed of the faster train is 72 kmph -/
theorem faster_train_speed :
  speed_of_faster_train 36 15 150 = 72 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l1349_134918


namespace NUMINAMATH_CALUDE_divisor_product_theorem_l1349_134910

/-- d(n) is the number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- s(n) is the sum of positive divisors of n -/
def s (n : ℕ) : ℕ := (Nat.divisors n).sum id

/-- The main theorem: s(x) * d(x) = 96 if and only if x is 14, 15, or 47 -/
theorem divisor_product_theorem (x : ℕ) : s x * d x = 96 ↔ x = 14 ∨ x = 15 ∨ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_divisor_product_theorem_l1349_134910
