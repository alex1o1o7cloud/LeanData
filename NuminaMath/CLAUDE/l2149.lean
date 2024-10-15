import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l2149_214931

theorem smallest_n_for_divisible_by_20 :
  ∃ (n : ℕ), n = 7 ∧ n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 7 → m ≥ 4 →
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b c d : ℤ), a ∈ T → b ∈ T → c ∈ T → d ∈ T →
      a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
      ¬(20 ∣ (a + b - c - d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_by_20_l2149_214931


namespace NUMINAMATH_CALUDE_parallelograms_from_congruent_triangles_l2149_214965

/-- Represents a triangle -/
structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- Forms quadrilaterals from two triangles -/
def form_quadrilaterals (t1 t2 : Triangle) : Set Quadrilateral := sorry

/-- Counts the number of parallelograms in a set of quadrilaterals -/
def count_parallelograms (qs : Set Quadrilateral) : ℕ := sorry

theorem parallelograms_from_congruent_triangles 
  (t1 t2 : Triangle) 
  (h : are_congruent t1 t2) : 
  count_parallelograms (form_quadrilaterals t1 t2) = 3 := sorry

end NUMINAMATH_CALUDE_parallelograms_from_congruent_triangles_l2149_214965


namespace NUMINAMATH_CALUDE_water_volume_calculation_l2149_214942

/-- Represents a cylindrical tank with an internal obstruction --/
structure Tank where
  radius : ℝ
  height : ℝ
  obstruction_radius : ℝ

/-- Calculates the volume of water in the tank --/
def water_volume (tank : Tank) (depth : ℝ) : ℝ :=
  sorry

/-- The specific tank in the problem --/
def problem_tank : Tank :=
  { radius := 5
  , height := 12
  , obstruction_radius := 2 }

theorem water_volume_calculation :
  water_volume problem_tank 3 = 110 * Real.pi - 96 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_calculation_l2149_214942


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l2149_214911

/-- Given points A and B, and a point C on the line y=x that intersects AB,
    prove that if AC = 2CB, then the y-coordinate of B is 4. -/
theorem intersection_point_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, a)
  let C : ℝ × ℝ := (x, x)
  ∃ x : ℝ, (C.1 - A.1, C.2 - A.2) = 2 • (B.1 - C.1, B.2 - C.2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l2149_214911


namespace NUMINAMATH_CALUDE_least_positive_angle_l2149_214995

theorem least_positive_angle (x a b : ℝ) (h1 : Real.tan x = 2 * a / (3 * b)) 
  (h2 : Real.tan (3 * x) = 3 * b / (2 * a + 3 * b)) :
  x = Real.arctan (2 / 3) ∧ x > 0 ∧ ∀ y, y > 0 → y = Real.arctan (2 / 3) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_l2149_214995


namespace NUMINAMATH_CALUDE_star_calculation_star_equation_solutions_l2149_214924

-- Define the ☆ operation
noncomputable def star (x y : ℤ) : ℤ :=
  if x = 0 then |y|
  else if y = 0 then |x|
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then |x| + |y|
  else -(|x| + |y|)

-- Theorem for the first part of the problem
theorem star_calculation : star 11 (star 0 (-12)) = 23 := by sorry

-- Theorem for the second part of the problem
theorem star_equation_solutions :
  {a : ℤ | 2 * (star 2 a) - 1 = 3 * a} = {3, -5} := by sorry

end NUMINAMATH_CALUDE_star_calculation_star_equation_solutions_l2149_214924


namespace NUMINAMATH_CALUDE_train_speed_equation_l2149_214947

theorem train_speed_equation (x : ℝ) (h1 : x > 0) (h2 : x + 20 > 0) : 
  (400 / x) - (400 / (x + 20)) = 0.5 ↔ 
  (400 / x) - (400 / (x + 20)) = (30 : ℝ) / 60 ∧
  (400 / x) > (400 / (x + 20)) ∧
  (400 / x) - (400 / (x + 20)) = (400 / x - 400 / (x + 20)) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_equation_l2149_214947


namespace NUMINAMATH_CALUDE_machine_production_time_l2149_214978

theorem machine_production_time (x : ℝ) (T : ℝ) : T = 10 :=
  let machine_B_rate := 2 * x / 5
  let combined_rate := x / 2
  have h1 : x / T + machine_B_rate = combined_rate := by sorry
  sorry

end NUMINAMATH_CALUDE_machine_production_time_l2149_214978


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l2149_214917

theorem average_marks_combined_classes (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) :
  students1 = 25 →
  students2 = 40 →
  avg1 = 50 →
  avg2 = 65 →
  let total_students := students1 + students2
  let total_marks := students1 * avg1 + students2 * avg2
  abs ((total_marks / total_students) - 59.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l2149_214917


namespace NUMINAMATH_CALUDE_cupcake_net_profit_l2149_214964

/-- Calculates the net profit from selling cupcakes given the specified conditions. -/
theorem cupcake_net_profit : 
  let cost_per_cupcake : ℚ := 0.75
  let selling_price : ℚ := 2.00
  let burnt_cupcakes : ℕ := 24
  let eaten_cupcakes : ℕ := 9
  let total_cupcakes : ℕ := 72
  let sellable_cupcakes : ℕ := total_cupcakes - (burnt_cupcakes + eaten_cupcakes)
  let total_cost : ℚ := cost_per_cupcake * total_cupcakes
  let total_revenue : ℚ := selling_price * sellable_cupcakes
  total_revenue - total_cost = 24.00 := by
sorry


end NUMINAMATH_CALUDE_cupcake_net_profit_l2149_214964


namespace NUMINAMATH_CALUDE_average_temperature_l2149_214960

def temperature_problem (new_york miami san_diego phoenix denver : ℝ) : Prop :=
  miami = new_york + 10 ∧
  san_diego = miami + 25 ∧
  phoenix = san_diego * 1.15 ∧
  denver = (new_york + miami + san_diego) / 3 - 5 ∧
  new_york = 80

theorem average_temperature 
  (new_york miami san_diego phoenix denver : ℝ) 
  (h : temperature_problem new_york miami san_diego phoenix denver) : 
  (new_york + miami + san_diego + phoenix + denver) / 5 = 101.45 := by
  sorry

#check average_temperature

end NUMINAMATH_CALUDE_average_temperature_l2149_214960


namespace NUMINAMATH_CALUDE_bag_balls_problem_l2149_214993

theorem bag_balls_problem (b g : ℕ) (p : ℚ) : 
  b = 8 →
  p = 1/3 →
  p = b / (b + g) →
  g = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_bag_balls_problem_l2149_214993


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2149_214934

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 8) :
  a 5 = 16 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2149_214934


namespace NUMINAMATH_CALUDE_group_size_calculation_l2149_214997

/-- Proves that the number of people in a group is 5, given the average weight increase and weight difference of replaced individuals. -/
theorem group_size_calculation (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 1.5 → weight_difference = 7.5 → 
  (weight_difference / average_increase : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2149_214997


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2149_214914

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = -3 + 2 * Complex.I) :
  z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2149_214914


namespace NUMINAMATH_CALUDE_range_of_k_l2149_214989

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := x^2 - x - 2 > 0

-- Define the property that p is sufficient but not necessary for q
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ (∃ x, q x ∧ ¬(p x k))

-- Theorem statement
theorem range_of_k :
  ∀ k, sufficient_not_necessary k ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l2149_214989


namespace NUMINAMATH_CALUDE_binomial_inequality_l2149_214979

theorem binomial_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : x ≠ 0) (h3 : n ≥ 2) :
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l2149_214979


namespace NUMINAMATH_CALUDE_set_equality_l2149_214970

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def A : Finset ℕ := {3,4,5}
def B : Finset ℕ := {1,3,6}
def C : Finset ℕ := {2,7,8}

theorem set_equality : C = (C ∪ A) ∩ (C ∪ B) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2149_214970


namespace NUMINAMATH_CALUDE_opposite_and_abs_of_2_minus_sqrt_3_l2149_214948

theorem opposite_and_abs_of_2_minus_sqrt_3 :
  let x : ℝ := 2 - Real.sqrt 3
  (- x = Real.sqrt 3 - 2) ∧ (abs x = 2 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_opposite_and_abs_of_2_minus_sqrt_3_l2149_214948


namespace NUMINAMATH_CALUDE_total_dress_designs_l2149_214930

/-- The number of fabric colors available. -/
def num_colors : ℕ := 4

/-- The number of patterns available. -/
def num_patterns : ℕ := 5

/-- Each dress design requires exactly one color and one pattern. -/
axiom dress_design_requirement : True

/-- The total number of different dress designs. -/
def total_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the total number of different dress designs is 20. -/
theorem total_dress_designs : total_designs = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l2149_214930


namespace NUMINAMATH_CALUDE_remainder_equality_l2149_214974

theorem remainder_equality (a b c : ℕ) :
  (2 * a + b) % 10 = (2 * b + c) % 10 ∧
  (2 * b + c) % 10 = (2 * c + a) % 10 →
  a % 10 = b % 10 ∧ b % 10 = c % 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2149_214974


namespace NUMINAMATH_CALUDE_estimate_wheat_amount_l2149_214994

/-- Estimates the amount of wheat in a mixed batch of grain -/
theorem estimate_wheat_amount (total_mixed : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) : 
  total_mixed = 1524 →
  sample_size = 254 →
  wheat_in_sample = 28 →
  (total_mixed * wheat_in_sample) / sample_size = 168 :=
by sorry

end NUMINAMATH_CALUDE_estimate_wheat_amount_l2149_214994


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2149_214910

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 2*m - 8 = 0 → n^2 + 2*n - 8 = 0 → m^2 + 3*m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2149_214910


namespace NUMINAMATH_CALUDE_intersection_A_B_l2149_214909

def set_A : Set ℝ := {x | ∃ t : ℝ, x = t^2 + 1}
def set_B : Set ℝ := {x | x * (x - 1) = 0}

theorem intersection_A_B :
  set_A ∩ set_B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2149_214909


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l2149_214938

theorem min_value_of_quadratic (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x^2 + 4*y^2 ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = x^2 + 4*y^2 → w ≥ z := by
sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l2149_214938


namespace NUMINAMATH_CALUDE_agricultural_profit_optimization_l2149_214943

/-- Represents the profit optimization problem for an agricultural product company -/
theorem agricultural_profit_optimization
  (retail_profit : ℝ) -- Profit from retailing one box
  (wholesale_profit : ℝ) -- Profit from wholesaling one box
  (total_boxes : ℕ) -- Total number of boxes to be sold
  (retail_limit : ℝ) -- Maximum percentage of boxes that can be sold through retail
  (h1 : retail_profit = 70)
  (h2 : wholesale_profit = 40)
  (h3 : total_boxes = 1000)
  (h4 : retail_limit = 0.3) :
  ∃ (retail_boxes wholesale_boxes : ℕ) (max_profit : ℝ),
    retail_boxes + wholesale_boxes = total_boxes ∧
    retail_boxes ≤ (retail_limit * total_boxes) ∧
    max_profit = retail_profit * retail_boxes + wholesale_profit * wholesale_boxes ∧
    retail_boxes = 300 ∧
    wholesale_boxes = 700 ∧
    max_profit = 49000 ∧
    ∀ (r w : ℕ),
      r + w = total_boxes →
      r ≤ (retail_limit * total_boxes) →
      retail_profit * r + wholesale_profit * w ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_agricultural_profit_optimization_l2149_214943


namespace NUMINAMATH_CALUDE_pet_store_profit_l2149_214905

-- Define Brandon's selling price
def brandon_price : ℕ := 100

-- Define the pet store's pricing strategy
def pet_store_price (brandon_price : ℕ) : ℕ := 3 * brandon_price + 5

-- Define the profit calculation
def profit (selling_price cost_price : ℕ) : ℕ := selling_price - cost_price

-- Theorem to prove
theorem pet_store_profit :
  profit (pet_store_price brandon_price) brandon_price = 205 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_profit_l2149_214905


namespace NUMINAMATH_CALUDE_area_triangle_ABP_l2149_214990

/-- Given points A and B in ℝ², and a point P on the x-axis forming a right angle with AB, 
    prove that the area of triangle ABP is 5/2. -/
theorem area_triangle_ABP (A B P : ℝ × ℝ) : 
  A = (1, 1) →
  B = (2, -1) →
  P.2 = 0 →  -- P is on the x-axis
  (P.1 - B.1) * (B.1 - A.1) + (P.2 - B.2) * (B.2 - A.2) = 0 →  -- ∠ABP = 90°
  abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2)) / 2 = 5/2 := by
sorry


end NUMINAMATH_CALUDE_area_triangle_ABP_l2149_214990


namespace NUMINAMATH_CALUDE_flour_in_mixing_bowl_l2149_214963

theorem flour_in_mixing_bowl (total_sugar : ℚ) (total_flour : ℚ) 
  (h1 : total_sugar = 5)
  (h2 : total_flour = 18)
  (h3 : total_flour - total_sugar = 5) :
  total_flour - (total_sugar + 5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_flour_in_mixing_bowl_l2149_214963


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2149_214926

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (Real.sqrt ((t.leg : ℝ)^2 - ((t.base : ℝ)/2)^2)) / 2

/-- Theorem: The minimum perimeter of two noncongruent integer-sided isosceles triangles
    with the same perimeter, same area, and bases in the ratio 8:7 is 676 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t₁ t₂ : IsoscelesTriangle),
    t₁ ≠ t₂ ∧
    perimeter t₁ = perimeter t₂ ∧
    area t₁ = area t₂ ∧
    8 * t₁.base = 7 * t₂.base ∧
    (∀ (s₁ s₂ : IsoscelesTriangle),
      s₁ ≠ s₂ →
      perimeter s₁ = perimeter s₂ →
      area s₁ = area s₂ →
      8 * s₁.base = 7 * s₂.base →
      perimeter t₁ ≤ perimeter s₁) ∧
    perimeter t₁ = 676 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2149_214926


namespace NUMINAMATH_CALUDE_function_properties_l2149_214961

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - 1

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a ≠ 0) :
  -- f(x) has an extremum at x = -1
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -1 ∧ |x + 1| < ε → f a x ≤ f a (-1)) →
  -- The line y = m intersects the graph of y = f(x) at three distinct points
  (∃ (m : ℝ), ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) →
  -- 1. When a < 0, f(x) is increasing on (-∞, +∞)
  (a < 0 → ∀ (x y : ℝ), x < y → f a x < f a y) ∧
  -- 2. When a > 0, f(x) is increasing on (-∞, -√a) ∪ (√a, +∞) and decreasing on (-√a, √a)
  (a > 0 → (∀ (x y : ℝ), (x < y ∧ y < -Real.sqrt a) ∨ (x > Real.sqrt a ∧ y > x) → f a x < f a y) ∧
           (∀ (x y : ℝ), -Real.sqrt a < x ∧ x < y ∧ y < Real.sqrt a → f a x > f a y)) ∧
  -- 3. The range of values for m is (-3, 1)
  (∃ (m : ℝ), -3 < m ∧ m < 1 ∧
    ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) ∧
  (∀ (m : ℝ), (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) → -3 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2149_214961


namespace NUMINAMATH_CALUDE_only_M_eq_neg_M_is_valid_assignment_l2149_214939

/-- Represents a simple programming language expression -/
inductive Expr
  | num (n : Int)
  | var (name : String)
  | assign (lhs : String) (rhs : Expr)
  | add (e1 e2 : Expr)

/-- Checks if an expression is a valid assignment statement -/
def isValidAssignment : Expr → Bool
  | Expr.assign _ _ => true
  | _ => false

/-- The given statements from the problem -/
def statements : List Expr := [
  Expr.assign "A" (Expr.num 3),
  Expr.assign "M" (Expr.var "M"),
  Expr.assign "B" (Expr.assign "A" (Expr.num 2)),
  Expr.add (Expr.var "x") (Expr.var "y")
]

theorem only_M_eq_neg_M_is_valid_assignment :
  statements.filter isValidAssignment = [Expr.assign "M" (Expr.var "M")] := by sorry

end NUMINAMATH_CALUDE_only_M_eq_neg_M_is_valid_assignment_l2149_214939


namespace NUMINAMATH_CALUDE_profit_percent_l2149_214967

theorem profit_percent (P : ℝ) (h : P > 0) : 
  (2 / 3 * P) * (1 + (-0.2)) = 0.8 * ((5 / 6) * P) → 
  (P - (5 / 6 * P)) / (5 / 6 * P) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_l2149_214967


namespace NUMINAMATH_CALUDE_subtraction_result_l2149_214966

theorem subtraction_result : 500000000000 - 3 * 111111111111 = 166666666667 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l2149_214966


namespace NUMINAMATH_CALUDE_g_8_equals_1036_l2149_214984

def g (x : ℝ) : ℝ := 3*x^4 - 22*x^3 + 37*x^2 - 28*x - 84

theorem g_8_equals_1036 : g 8 = 1036 := by
  sorry

end NUMINAMATH_CALUDE_g_8_equals_1036_l2149_214984


namespace NUMINAMATH_CALUDE_max_cans_consumed_correct_verify_100_cans_l2149_214953

def exchange_rate : ℕ := 3

def max_cans_consumed (n : ℕ) : ℕ :=
  n + (n - 1) / 2

theorem max_cans_consumed_correct (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), k * exchange_rate ≤ max_cans_consumed n ∧
             max_cans_consumed n < (k + 1) * exchange_rate :=
by sorry

-- Verify the specific case for 100 cans
theorem verify_100_cans :
  max_cans_consumed 67 ≥ 100 ∧ max_cans_consumed 66 < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_cans_consumed_correct_verify_100_cans_l2149_214953


namespace NUMINAMATH_CALUDE_power_multiplication_l2149_214962

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l2149_214962


namespace NUMINAMATH_CALUDE_max_odd_integers_with_even_product_l2149_214925

theorem max_odd_integers_with_even_product (integers : Finset ℕ) 
  (h1 : integers.card = 7)
  (h2 : ∀ n ∈ integers, n > 0)
  (h3 : Even (integers.prod id)) :
  { odd_count : ℕ // odd_count ≤ 6 ∧ 
    ∃ (odd_subset : Finset ℕ), 
      odd_subset ⊆ integers ∧ 
      odd_subset.card = odd_count ∧ 
      ∀ n ∈ odd_subset, Odd n } :=
by sorry

end NUMINAMATH_CALUDE_max_odd_integers_with_even_product_l2149_214925


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_l2149_214915

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- Define the line
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | p ∈ line ∧ ∃ t, curve t = p}

-- Theorem statement
theorem distance_between_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_l2149_214915


namespace NUMINAMATH_CALUDE_panthers_scored_17_points_l2149_214958

-- Define the points scored by the Wildcats
def wildcats_points : ℕ := 36

-- Define the difference in points between Wildcats and Panthers
def point_difference : ℕ := 19

-- Define the points scored by the Panthers
def panthers_points : ℕ := wildcats_points - point_difference

-- Theorem to prove
theorem panthers_scored_17_points : panthers_points = 17 := by
  sorry

end NUMINAMATH_CALUDE_panthers_scored_17_points_l2149_214958


namespace NUMINAMATH_CALUDE_reservoir_capacity_problem_l2149_214988

/-- Theorem about a reservoir's capacity and water levels -/
theorem reservoir_capacity_problem (current_amount : ℝ) 
  (h1 : current_amount = 6)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.6 * total_capacity) :
  total_capacity - normal_level = 7 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_problem_l2149_214988


namespace NUMINAMATH_CALUDE_exists_unique_max_N_l2149_214976

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ+ := sorry

/-- The function f(n) = d(n) / (n^(1/3)) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val ^ (1/3 : ℝ)

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- The theorem stating the existence of a unique N maximizing f(n) -/
theorem exists_unique_max_N : ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N = 6 := by sorry

end NUMINAMATH_CALUDE_exists_unique_max_N_l2149_214976


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2149_214992

theorem complex_fraction_equality (z : ℂ) :
  z = 2 + I →
  (2 * I) / (z - 1) = 1 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2149_214992


namespace NUMINAMATH_CALUDE_function_derivative_implies_coefficients_l2149_214928

/-- Given a function f(x) = x^m + ax with derivative f'(x) = 2x + 1, prove that m = 3 and a = 1 -/
theorem function_derivative_implies_coefficients 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^m + a*x) 
  (h2 : ∀ x, deriv f x = 2*x + 1) : 
  m = 3 ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_implies_coefficients_l2149_214928


namespace NUMINAMATH_CALUDE_vector_BC_l2149_214991

def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (4, 5)

theorem vector_BC : (C.1 - B.1, C.2 - B.2) = (3, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l2149_214991


namespace NUMINAMATH_CALUDE_compare_fractions_l2149_214949

theorem compare_fractions : -3/8 > -4/9 := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l2149_214949


namespace NUMINAMATH_CALUDE_swim_time_proof_l2149_214935

-- Define the given constants
def downstream_distance : ℝ := 16
def upstream_distance : ℝ := 10
def still_water_speed : ℝ := 6.5

-- Define the theorem
theorem swim_time_proof :
  ∃ (t c : ℝ),
    t > 0 ∧
    c ≥ 0 ∧
    c < still_water_speed ∧
    downstream_distance / (still_water_speed + c) = t ∧
    upstream_distance / (still_water_speed - c) = t ∧
    t = 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_proof_l2149_214935


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2149_214999

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * Real.sqrt 2 * x

-- Define the line l
def line (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Define the isosceles triangle condition
def isosceles_triangle (A M N : ℝ × ℝ) : Prop :=
  (A.1 - M.1)^2 + (A.2 - M.2)^2 = (A.1 - N.1)^2 + (A.2 - N.2)^2

theorem ellipse_and_line_intersection
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (x y : ℝ), ellipse a b x y ∧ parabola x y)
  (h4 : ∃ (x1 x2 y1 y2 : ℝ), 
    x1^2 + x2^2 + y1^2 + y2^2 = 2 * a * b * Real.sqrt 3)
  (k m : ℝ) (h5 : k ≠ 0)
  (h6 : ∃ (M N : ℝ × ℝ), 
    ellipse a b M.1 M.2 ∧ 
    ellipse a b N.1 N.2 ∧ 
    line k m M.1 M.2 ∧ 
    line k m N.1 N.2 ∧ 
    M ≠ N)
  (h7 : ∃ (A : ℝ × ℝ), ellipse a b A.1 A.2 ∧ A.2 < 0)
  (h8 : ∀ (A M N : ℝ × ℝ), 
    ellipse a b A.1 A.2 ∧ A.2 < 0 ∧
    ellipse a b M.1 M.2 ∧ 
    ellipse a b N.1 N.2 ∧ 
    line k m M.1 M.2 ∧ 
    line k m N.1 N.2 →
    isosceles_triangle A M N) :
  a = Real.sqrt 3 ∧ b = 1 ∧ 1/2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2149_214999


namespace NUMINAMATH_CALUDE_meal_cost_is_27_l2149_214929

/-- Represents the cost of a meal with tax and tip. -/
structure MealCost where
  pretax : ℝ
  tax_rate : ℝ
  tip_rate : ℝ
  total : ℝ

/-- Calculates the total cost of a meal including tax and tip. -/
def total_cost (m : MealCost) : ℝ :=
  m.pretax * (1 + m.tax_rate + m.tip_rate)

/-- Theorem stating that given the conditions, the pre-tax meal cost is $27. -/
theorem meal_cost_is_27 :
  ∃ (m : MealCost),
    m.tax_rate = 0.08 ∧
    m.tip_rate = 0.18 ∧
    m.total = 33.60 ∧
    total_cost m = m.total ∧
    m.pretax = 27 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_is_27_l2149_214929


namespace NUMINAMATH_CALUDE_granola_cost_per_bag_l2149_214986

theorem granola_cost_per_bag 
  (total_bags : ℕ) 
  (full_price_bags : ℕ) 
  (full_price : ℚ) 
  (discounted_bags : ℕ) 
  (discounted_price : ℚ) 
  (net_profit : ℚ) 
  (h1 : total_bags = 20)
  (h2 : full_price_bags = 15)
  (h3 : full_price = 6)
  (h4 : discounted_bags = 5)
  (h5 : discounted_price = 4)
  (h6 : net_profit = 50)
  (h7 : total_bags = full_price_bags + discounted_bags) :
  (full_price_bags * full_price + discounted_bags * discounted_price - net_profit) / total_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_granola_cost_per_bag_l2149_214986


namespace NUMINAMATH_CALUDE_cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2_l2149_214940

theorem cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2 :
  (Real.cos (10 * π / 180) - Real.sqrt 3 * Real.cos (-100 * π / 180)) /
  Real.sqrt (1 - Real.sin (10 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2_l2149_214940


namespace NUMINAMATH_CALUDE_solution_implies_m_equals_three_l2149_214902

theorem solution_implies_m_equals_three (x y m : ℝ) : 
  x = -2 → y = 1 → m * x + 5 * y = -1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_equals_three_l2149_214902


namespace NUMINAMATH_CALUDE_height_weight_correlation_l2149_214955

-- Define the relationship types
inductive Relationship
| Functional
| Correlated
| Unrelated

-- Define the variables
structure Square where
  side : ℝ

structure Vehicle where
  speed : ℝ

structure Person where
  height : ℝ
  weight : ℝ
  eyesight : ℝ

-- Define the relationships between variables
def square_area_perimeter_relation (s : Square) : Relationship :=
  Relationship.Functional

def vehicle_distance_time_relation (v : Vehicle) : Relationship :=
  Relationship.Functional

def person_height_weight_relation (p : Person) : Relationship :=
  Relationship.Correlated

def person_height_eyesight_relation (p : Person) : Relationship :=
  Relationship.Unrelated

-- Theorem statement
theorem height_weight_correlation :
  ∃ (p : Person), person_height_weight_relation p = Relationship.Correlated ∧
    (∀ (s : Square), square_area_perimeter_relation s ≠ Relationship.Correlated) ∧
    (∀ (v : Vehicle), vehicle_distance_time_relation v ≠ Relationship.Correlated) ∧
    (person_height_eyesight_relation p ≠ Relationship.Correlated) :=
  sorry

end NUMINAMATH_CALUDE_height_weight_correlation_l2149_214955


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l2149_214907

/-- Define the first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- Define the last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Define the sum of elements in the nth set -/
def sum_of_set (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : sum_of_set 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l2149_214907


namespace NUMINAMATH_CALUDE_power_function_positive_l2149_214987

theorem power_function_positive (α : ℝ) (x : ℝ) (h : x > 0) : x^α > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_positive_l2149_214987


namespace NUMINAMATH_CALUDE_deer_families_moved_out_l2149_214998

theorem deer_families_moved_out (total : ℕ) (stayed : ℕ) (moved_out : ℕ) : 
  total = 79 → stayed = 45 → moved_out = total - stayed → moved_out = 34 := by
  sorry

end NUMINAMATH_CALUDE_deer_families_moved_out_l2149_214998


namespace NUMINAMATH_CALUDE_fraction_division_result_l2149_214901

theorem fraction_division_result : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_result_l2149_214901


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l2149_214941

theorem cylinder_volume_equality (y : ℝ) : y > 0 →
  (π * (7 + 4)^2 * 5 = π * 7^2 * (5 + y)) → y = 360 / 49 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l2149_214941


namespace NUMINAMATH_CALUDE_second_investment_interest_rate_l2149_214950

theorem second_investment_interest_rate
  (total_income : ℝ)
  (investment1_principal : ℝ)
  (investment1_rate : ℝ)
  (investment2_principal : ℝ)
  (total_investment : ℝ)
  (h1 : total_income = 575)
  (h2 : investment1_principal = 3000)
  (h3 : investment1_rate = 0.085)
  (h4 : investment2_principal = 5000)
  (h5 : total_investment = 8000)
  (h6 : total_investment = investment1_principal + investment2_principal) :
  let investment1_income := investment1_principal * investment1_rate
  let investment2_income := total_income - investment1_income
  let investment2_rate := investment2_income / investment2_principal
  investment2_rate = 0.064 := by
  sorry

end NUMINAMATH_CALUDE_second_investment_interest_rate_l2149_214950


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2149_214904

/-- Given a point A with coordinates (m+1, 2m-4) that is moved up by 2 units
    and lands on the x-axis, prove that m = 1. -/
theorem point_on_x_axis (m : ℝ) : 
  let initial_y := 2*m - 4
  let moved_y := initial_y + 2
  moved_y = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2149_214904


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2149_214923

theorem quadratic_equation_root (a b : ℝ) (h : a ≠ 0) :
  (a * 2019^2 + b * 2019 + 2 = 0) →
  (a * (2019 - 1)^2 + b * (2019 - 1) = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2149_214923


namespace NUMINAMATH_CALUDE_max_value_is_72_l2149_214982

/-- Represents a type of stone with its weight and value -/
structure Stone where
  weight : ℕ
  value : ℕ

/-- The problem setup -/
def stones : List Stone := [
  { weight := 3, value := 9 },
  { weight := 6, value := 15 },
  { weight := 1, value := 1 }
]

/-- The maximum weight Tanya can carry -/
def maxWeight : ℕ := 24

/-- The minimum number of each type of stone available -/
def minStoneCount : ℕ := 10

/-- Calculates the maximum value of stones that can be carried given the constraints -/
def maxValue : ℕ :=
  sorry -- Proof goes here

/-- Theorem stating that the maximum value is 72 -/
theorem max_value_is_72 : maxValue = 72 := by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_max_value_is_72_l2149_214982


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2149_214956

theorem unique_solution_quadratic_inequality (m : ℝ) : 
  (∃! x : ℝ, x^2 - m*x + 1 ≤ 0) → (m = 2 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l2149_214956


namespace NUMINAMATH_CALUDE_trail_mix_weight_l2149_214916

def peanuts_weight : Float := 0.16666666666666666
def chocolate_chips_weight : Float := 0.16666666666666666
def raisins_weight : Float := 0.08333333333333333

theorem trail_mix_weight :
  peanuts_weight + chocolate_chips_weight + raisins_weight = 0.41666666666666663 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l2149_214916


namespace NUMINAMATH_CALUDE_matching_arrangements_count_l2149_214972

def number_of_people : Nat := 5

/-- The number of arrangements where exactly two people sit in seats matching their numbers -/
def matching_arrangements : Nat :=
  (number_of_people.choose 2) * 2 * 1 * 1

theorem matching_arrangements_count : matching_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_matching_arrangements_count_l2149_214972


namespace NUMINAMATH_CALUDE_friend_walking_rates_l2149_214944

theorem friend_walking_rates (trail_length : ℝ) (p_distance : ℝ) 
  (h1 : trail_length = 36)
  (h2 : p_distance = 20)
  (h3 : p_distance < trail_length) :
  let q_distance := trail_length - p_distance
  let rate_ratio := p_distance / q_distance
  (rate_ratio - 1) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_friend_walking_rates_l2149_214944


namespace NUMINAMATH_CALUDE_store_purchase_combinations_l2149_214945

/-- The number of ways to buy three items (headphones, a keyboard, and a mouse) in a store with given inventory. -/
theorem store_purchase_combinations (headphones : ℕ) (mice : ℕ) (keyboards : ℕ) (keyboard_mouse_sets : ℕ) (headphone_mouse_sets : ℕ) : 
  headphones = 9 → 
  mice = 13 → 
  keyboards = 5 → 
  keyboard_mouse_sets = 4 → 
  headphone_mouse_sets = 5 → 
  headphones * keyboard_mouse_sets + 
  keyboards * headphone_mouse_sets + 
  headphones * mice * keyboards = 646 := by
sorry

end NUMINAMATH_CALUDE_store_purchase_combinations_l2149_214945


namespace NUMINAMATH_CALUDE_symmetric_points_l2149_214952

/-- Given that point A(2, 4) is symmetric to point B(b-1, 2a) with respect to the origin, prove that a - b = -1 -/
theorem symmetric_points (a b : ℝ) : 
  (2 = -(b - 1) ∧ 4 = -2*a) → a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l2149_214952


namespace NUMINAMATH_CALUDE_number_of_paths_equals_combination_l2149_214959

def grid_width : ℕ := 7
def grid_height : ℕ := 4

def total_steps : ℕ := grid_width + grid_height - 2
def up_steps : ℕ := grid_height - 1

theorem number_of_paths_equals_combination :
  (Nat.choose total_steps up_steps) = 84 := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_equals_combination_l2149_214959


namespace NUMINAMATH_CALUDE_slower_train_speed_l2149_214980

/-- Prove that the speed of the slower train is 36 km/hr -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 55) 
  (h2 : faster_speed = 47) 
  (h3 : passing_time = 36) : 
  ∃ (slower_speed : ℝ), 
    slower_speed = 36 ∧ 
    (2 * train_length) = (faster_speed - slower_speed) * (5/18) * passing_time :=
sorry

end NUMINAMATH_CALUDE_slower_train_speed_l2149_214980


namespace NUMINAMATH_CALUDE_price_change_l2149_214985

theorem price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * (1 - 0.2)
  let price_after_increase := price_after_decrease * (1 + 0.5)
  price_after_increase = P * 1.2 := by
sorry

end NUMINAMATH_CALUDE_price_change_l2149_214985


namespace NUMINAMATH_CALUDE_liam_activity_balance_l2149_214921

/-- Utility function for Liam's activities -/
def utility (reading : ℝ) (basketball : ℝ) : ℝ := reading * basketball

/-- Wednesday's utility calculation -/
def wednesday_utility (t : ℝ) : ℝ := utility (10 - t) t

/-- Thursday's utility calculation -/
def thursday_utility (t : ℝ) : ℝ := utility (t + 4) (3 - t)

/-- The theorem stating that t = 3 is the only valid solution -/
theorem liam_activity_balance :
  ∃! t : ℝ, t > 0 ∧ t < 10 ∧ wednesday_utility t = thursday_utility t ∧ t = 3 := by sorry

end NUMINAMATH_CALUDE_liam_activity_balance_l2149_214921


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2149_214922

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 8*x + 1 = 0) ∧
  (∃ x : ℝ, x*(x-2) - x + 2 = 0) ∧
  (∀ x : ℝ, x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) ∧
  (∀ x : ℝ, x*(x-2) - x + 2 = 0 ↔ x = 2 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2149_214922


namespace NUMINAMATH_CALUDE_total_pay_for_given_scenario_l2149_214903

/-- The total amount paid to two employees, where one is paid 120% of the other's pay -/
def total_pay (y_pay : ℝ) : ℝ :=
  y_pay + 1.2 * y_pay

theorem total_pay_for_given_scenario :
  total_pay 260 = 572 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_for_given_scenario_l2149_214903


namespace NUMINAMATH_CALUDE_value_of_3x2_minus_3y2_l2149_214927

theorem value_of_3x2_minus_3y2 (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 4) : 
  3 * (x^2 - y^2) = 240 := by
sorry

end NUMINAMATH_CALUDE_value_of_3x2_minus_3y2_l2149_214927


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2149_214919

def polynomial (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - 5*x + 15

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 108 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2149_214919


namespace NUMINAMATH_CALUDE_product_coefficients_sum_l2149_214951

theorem product_coefficients_sum (m n : ℚ) : 
  (∀ k : ℚ, (5*k^2 - 4*k + m) * (2*k^2 + n*k - 5) = 10*k^4 - 28*k^3 + 23*k^2 - 18*k + 15) →
  m + n = 35/3 := by
sorry

end NUMINAMATH_CALUDE_product_coefficients_sum_l2149_214951


namespace NUMINAMATH_CALUDE_final_order_exact_points_total_games_l2149_214977

-- Define the structure for a team's game outcomes
structure TeamOutcome where
  name : String
  wins : Nat
  losses : Nat
  draws : Nat
  bonusWins : Nat
  extraBonus : Nat

-- Define the point system
def regularWinPoints : Nat := 3
def regularLossPoints : Nat := 0
def regularDrawPoints : Nat := 1
def bonusWinPoints : Nat := 2
def extraBonusPoints : Nat := 1

-- Calculate total points for a team
def calculatePoints (team : TeamOutcome) : Nat :=
  team.wins * regularWinPoints +
  team.losses * regularLossPoints +
  team.draws * regularDrawPoints +
  team.bonusWins * bonusWinPoints +
  team.extraBonus * extraBonusPoints

-- Define the teams
def soccerStars : TeamOutcome := ⟨"Team Soccer Stars", 18, 5, 7, 6, 4⟩
def lightningStrikers : TeamOutcome := ⟨"Lightning Strikers", 15, 8, 7, 5, 3⟩
def goalGrabbers : TeamOutcome := ⟨"Goal Grabbers", 21, 5, 4, 4, 9⟩
def cleverKickers : TeamOutcome := ⟨"Clever Kickers", 11, 10, 9, 2, 1⟩

-- Theorem to prove the final order of teams
theorem final_order :
  calculatePoints goalGrabbers > calculatePoints soccerStars ∧
  calculatePoints soccerStars > calculatePoints lightningStrikers ∧
  calculatePoints lightningStrikers > calculatePoints cleverKickers :=
by sorry

-- Theorem to prove the exact points for each team
theorem exact_points :
  calculatePoints goalGrabbers = 84 ∧
  calculatePoints soccerStars = 77 ∧
  calculatePoints lightningStrikers = 65 ∧
  calculatePoints cleverKickers = 47 :=
by sorry

-- Theorem to prove that each team played exactly 30 games
theorem total_games (team : TeamOutcome) :
  team.wins + team.losses + team.draws = 30 :=
by sorry

end NUMINAMATH_CALUDE_final_order_exact_points_total_games_l2149_214977


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2149_214920

theorem complex_equation_solution (z : ℂ) :
  z / (1 - Complex.I) = Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2149_214920


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l2149_214969

/-- The number of matches required in a single-elimination tournament -/
def matches_required (num_players : ℕ) : ℕ :=
  num_players - 1

/-- Theorem: A single-elimination tournament with 32 players requires 31 matches -/
theorem chess_tournament_matches :
  matches_required 32 = 31 :=
by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_matches_l2149_214969


namespace NUMINAMATH_CALUDE_pizza_stand_total_slices_l2149_214996

/-- Given the conditions of the pizza stand problem, prove that the total number of slices sold is 5000. -/
theorem pizza_stand_total_slices : 
  let small_price : ℕ := 150
  let large_price : ℕ := 250
  let total_revenue : ℕ := 1050000
  let small_slices_sold : ℕ := 2000
  let large_slices_sold : ℕ := (total_revenue - small_price * small_slices_sold) / large_price
  small_slices_sold + large_slices_sold = 5000 := by
sorry


end NUMINAMATH_CALUDE_pizza_stand_total_slices_l2149_214996


namespace NUMINAMATH_CALUDE_min_seats_occupied_l2149_214973

theorem min_seats_occupied (total_seats : ℕ) (initial_occupied : ℕ) : 
  total_seats = 150 → initial_occupied = 2 → 
  (∃ (additional_seats : ℕ), 
    additional_seats = 49 ∧ 
    ∀ (x : ℕ), x < additional_seats → 
      ∃ (y : ℕ), y ≤ total_seats - initial_occupied - x ∧ 
      y ≥ 2 ∧ 
      ∀ (z : ℕ), z < y → (z = 1 ∨ z = y)) :=
by sorry

end NUMINAMATH_CALUDE_min_seats_occupied_l2149_214973


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2149_214933

theorem count_negative_numbers : let numbers := [-3^2, (-1)^2006, 0, |(-2)|, -(-2), -3 * 2^2]
  (numbers.filter (· < 0)).length = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2149_214933


namespace NUMINAMATH_CALUDE_inequality_proof_l2149_214900

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 ∧
  ∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + 
    Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + 
    Real.sqrt (d / (a + b + c + e)) + 
    Real.sqrt (e / (a + b + c + d)) > m) → 
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2149_214900


namespace NUMINAMATH_CALUDE_empty_set_proof_l2149_214936

theorem empty_set_proof : {x : ℝ | x > 9 ∧ x < 3} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l2149_214936


namespace NUMINAMATH_CALUDE_circle_line_relationship_l2149_214918

-- Define the circle C: x^2 + y^2 = 4
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l: √3x + y - 8 = 0
def l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 8 = 0

-- Theorem statement
theorem circle_line_relationship :
  -- C and l are separate
  (∀ x y : ℝ, C x y → ¬ l x y) ∧
  -- The shortest distance from any point on C to l is 2
  (∀ x y : ℝ, C x y → ∃ d : ℝ, d = 2 ∧ 
    ∀ x' y' : ℝ, l x' y' → Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d) :=
sorry

end NUMINAMATH_CALUDE_circle_line_relationship_l2149_214918


namespace NUMINAMATH_CALUDE_article_cost_price_l2149_214912

theorem article_cost_price : ∃ (C : ℝ), 
  (C = 600) ∧ 
  (∃ (SP : ℝ), SP = 1.05 * C) ∧ 
  (∃ (SP_new C_new : ℝ), 
    C_new = 0.95 * C ∧ 
    SP_new = 1.05 * C - 3 ∧ 
    SP_new = 1.045 * C_new) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_price_l2149_214912


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2149_214983

def population_size : ℕ := 50
def sample_size : ℕ := 5
def starting_point : ℕ := 5
def step_size : ℕ := population_size / sample_size

def systematic_sample (start : ℕ) (step : ℕ) (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => start + i * step)

theorem systematic_sampling_result :
  systematic_sample starting_point step_size sample_size = [5, 15, 25, 35, 45] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2149_214983


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2149_214937

def i : ℂ := Complex.I

theorem complex_absolute_value : 
  Complex.abs ((1 : ℂ) / (1 - i) - i) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2149_214937


namespace NUMINAMATH_CALUDE_parallel_plane_through_point_l2149_214908

def plane_equation (x y z : ℝ) : ℝ := 3*x + 2*y - 4*z - 16

theorem parallel_plane_through_point :
  let given_plane (x y z : ℝ) := 3*x + 2*y - 4*z - 5
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ given_plane x y z = k) ∧
  plane_equation 2 3 (-1) = 0 ∧
  (∃ (A B C D : ℤ), 
    (∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) :=
by sorry

end NUMINAMATH_CALUDE_parallel_plane_through_point_l2149_214908


namespace NUMINAMATH_CALUDE_twenty_five_percent_more_than_eighty_twenty_five_percent_more_than_eighty_proof_l2149_214968

theorem twenty_five_percent_more_than_eighty : ℝ → Prop :=
  fun x => (3/4 * x = 100) → (x = 400/3)

-- The proof is omitted
theorem twenty_five_percent_more_than_eighty_proof : 
  ∃ x : ℝ, twenty_five_percent_more_than_eighty x :=
sorry

end NUMINAMATH_CALUDE_twenty_five_percent_more_than_eighty_twenty_five_percent_more_than_eighty_proof_l2149_214968


namespace NUMINAMATH_CALUDE_combined_shape_area_l2149_214957

/-- The area of a shape formed by attaching a square to a rectangle -/
theorem combined_shape_area (rectangle_length rectangle_width square_side : Real) :
  rectangle_length = 0.45 →
  rectangle_width = 0.25 →
  square_side = 0.15 →
  rectangle_length * rectangle_width + square_side * square_side = 0.135 := by
  sorry

end NUMINAMATH_CALUDE_combined_shape_area_l2149_214957


namespace NUMINAMATH_CALUDE_unique_solution_exists_l2149_214954

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

-- Theorem statement
theorem unique_solution_exists :
  ∃! y : ℝ, star 2 y = 5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l2149_214954


namespace NUMINAMATH_CALUDE_total_symbol_count_is_62_l2149_214906

/-- The number of distinct symbols that can be represented by a sequence of dots and dashes of a given length. -/
def symbolCount (length : Nat) : Nat :=
  2^length

/-- The total number of distinct symbols that can be represented using sequences of 1 to 5 dots and/or dashes. -/
def totalSymbolCount : Nat :=
  (symbolCount 1) + (symbolCount 2) + (symbolCount 3) + (symbolCount 4) + (symbolCount 5)

/-- Theorem stating that the total number of distinct symbols is 62. -/
theorem total_symbol_count_is_62 : totalSymbolCount = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_symbol_count_is_62_l2149_214906


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l2149_214975

/-- Number of players in a soccer team -/
def total_players : ℕ := 24

/-- Number of starting players -/
def starting_players : ℕ := 12

/-- Number of substitute players -/
def substitute_players : ℕ := 12

/-- Maximum number of substitutions allowed -/
def max_substitutions : ℕ := 4

/-- 
Calculate the number of ways to make substitutions in a soccer game
n: current number of substitutions made
-/
def substitution_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 12 * (13 - n) * substitution_ways (n - 1)

/-- 
The total number of ways to make substitutions is the sum of ways
to make 0, 1, 2, 3, and 4 substitutions
-/
def total_ways : ℕ := 
  (List.range 5).map substitution_ways |>.sum

/-- Main theorem to prove -/
theorem soccer_substitutions_remainder :
  total_ways % 1000 = 573 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l2149_214975


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l2149_214981

/-- Given three lines that intersect at the same point, prove that k = -5 --/
theorem intersecting_lines_k_value (x y k : ℝ) :
  (y = 5 * x + 3) ∧
  (y = -2 * x - 25) ∧
  (y = 3 * x + k) →
  k = -5 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l2149_214981


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l2149_214971

/-- Represents the duration in days that a supply of pills will last -/
def duration_in_days (num_pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) : ℚ :=
  (num_pills : ℚ) * (days_between_doses : ℚ) / pill_fraction

/-- Converts days to months, assuming 30 days per month -/
def days_to_months (days : ℚ) : ℚ :=
  days / 30

theorem medicine_supply_duration :
  let num_pills : ℕ := 60
  let pill_fraction : ℚ := 3/4
  let days_between_doses : ℕ := 3
  let duration_days := duration_in_days num_pills pill_fraction days_between_doses
  let duration_months := days_to_months duration_days
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/3 ∧ |duration_months - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l2149_214971


namespace NUMINAMATH_CALUDE_dans_initial_money_l2149_214913

/-- Calculates the initial amount of money given the number of items bought,
    the cost per item, and the amount left after purchase. -/
def initialMoney (itemsBought : ℕ) (costPerItem : ℕ) (amountLeft : ℕ) : ℕ :=
  itemsBought * costPerItem + amountLeft

/-- Theorem stating that given the specific conditions of Dan's purchase,
    his initial amount of money was $298. -/
theorem dans_initial_money :
  initialMoney 99 3 1 = 298 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l2149_214913


namespace NUMINAMATH_CALUDE_selection_methods_count_l2149_214946

def num_students : ℕ := 5
def num_selected : ℕ := 4
def num_days : ℕ := 3
def num_friday : ℕ := 2
def num_saturday : ℕ := 1
def num_sunday : ℕ := 1

theorem selection_methods_count :
  (num_students.choose num_friday) *
  ((num_students - num_friday).choose num_saturday) *
  ((num_students - num_friday - num_saturday).choose num_sunday) = 60 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l2149_214946


namespace NUMINAMATH_CALUDE_function_domain_l2149_214932

/-- The function f(x) = √(2-2^x) + 1/ln(x) is defined if and only if x ∈ (0,1) -/
theorem function_domain (x : ℝ) : 
  (∃ (y : ℝ), y = Real.sqrt (2 - 2^x) + 1 / Real.log x) ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_l2149_214932
