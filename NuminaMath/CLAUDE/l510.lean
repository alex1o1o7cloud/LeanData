import Mathlib

namespace NUMINAMATH_CALUDE_front_view_of_given_stack_map_l510_51055

/-- Represents a stack map as a list of lists of natural numbers -/
def StackMap := List (List Nat)

/-- Calculates the front view of a stack map -/
def frontView (sm : StackMap) : List Nat :=
  let columns := sm.map List.length
  List.map (fun col => List.foldl Nat.max 0 (List.map (fun row => row.getD col 0) sm)) (List.range (columns.foldl Nat.max 0))

/-- The given stack map -/
def givenStackMap : StackMap := [[4, 1], [1, 2, 4], [3, 1]]

theorem front_view_of_given_stack_map :
  frontView givenStackMap = [4, 2, 4] := by sorry

end NUMINAMATH_CALUDE_front_view_of_given_stack_map_l510_51055


namespace NUMINAMATH_CALUDE_pencil_profit_proof_l510_51049

/-- Proves that selling 1500 pencils results in a profit of exactly $150.00 -/
theorem pencil_profit_proof (total_pencils : ℕ) (buy_price sell_price : ℚ) (profit_target : ℚ) 
  (h1 : total_pencils = 2000)
  (h2 : buy_price = 15/100)
  (h3 : sell_price = 30/100)
  (h4 : profit_target = 150) :
  (1500 : ℚ) * sell_price - (total_pencils : ℚ) * buy_price = profit_target := by
  sorry

end NUMINAMATH_CALUDE_pencil_profit_proof_l510_51049


namespace NUMINAMATH_CALUDE_f_positive_iff_x_range_l510_51053

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_positive_iff_x_range :
  (∀ x : ℝ, (∀ a ∈ Set.Icc (-1 : ℝ) 1, f x a > 0)) ↔
  (∀ x : ℝ, x < 1 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_x_range_l510_51053


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l510_51004

/-- The number of ways to select a president and a vice president from a group of 4 people -/
def select_president_and_vice_president (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select a president and a vice president from a group of 4 people is 12 -/
theorem president_vice_president_selection :
  select_president_and_vice_president 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l510_51004


namespace NUMINAMATH_CALUDE_golf_strokes_over_par_l510_51093

/-- Given a golfer who plays 9 rounds with an average of 4 strokes per hole,
    and a par value of 3 per hole, prove that the golfer will be 9 strokes over par. -/
theorem golf_strokes_over_par (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_value_per_hole : ℕ)
  (h1 : rounds = 9)
  (h2 : avg_strokes_per_hole = 4)
  (h3 : par_value_per_hole = 3) :
  rounds * avg_strokes_per_hole - rounds * par_value_per_hole = 9 :=
by sorry

end NUMINAMATH_CALUDE_golf_strokes_over_par_l510_51093


namespace NUMINAMATH_CALUDE_max_product_constrained_max_value_is_three_l510_51073

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/3 + b/4 = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x/3 + y/4 = 1 → x*y ≤ a*b := by
  sorry

theorem max_value_is_three (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/3 + b/4 = 1) :
  a*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_value_is_three_l510_51073


namespace NUMINAMATH_CALUDE_correct_mark_is_63_l510_51079

/-- Proves that the correct mark is 63 given the conditions of the problem -/
theorem correct_mark_is_63 (n : ℕ) (wrong_mark : ℕ) (avg_increase : ℚ) : 
  n = 40 → 
  wrong_mark = 83 → 
  avg_increase = 1/2 → 
  (wrong_mark - (n * avg_increase : ℚ).floor : ℤ) = 63 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_is_63_l510_51079


namespace NUMINAMATH_CALUDE_stone_pile_total_l510_51074

/-- Represents the number of stones in each pile -/
structure StonePiles where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- The conditions of the stone pile problem -/
def stone_pile_conditions (piles : StonePiles) : Prop :=
  piles.fifth = 6 * piles.third ∧
  piles.second = 2 * (piles.third + piles.fifth) ∧
  piles.first * 3 = piles.fifth ∧
  piles.first + 10 = piles.fourth ∧
  2 * piles.fourth = piles.second

/-- The theorem stating that under the given conditions, the total number of stones is 60 -/
theorem stone_pile_total (piles : StonePiles) 
  (h : stone_pile_conditions piles) : 
  piles.first + piles.second + piles.third + piles.fourth + piles.fifth = 60 := by
  sorry


end NUMINAMATH_CALUDE_stone_pile_total_l510_51074


namespace NUMINAMATH_CALUDE_dvd_sales_l510_51001

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end NUMINAMATH_CALUDE_dvd_sales_l510_51001


namespace NUMINAMATH_CALUDE_wall_bricks_count_l510_51057

/-- The number of bricks in the wall after adjustments -/
def total_bricks : ℕ :=
  let initial_courses := 5
  let additional_courses := 7
  let bricks_per_course := 450
  let initial_bricks := initial_courses * bricks_per_course
  let added_bricks := additional_courses * bricks_per_course
  let removed_bricks := [
    bricks_per_course / 3,
    bricks_per_course / 4,
    bricks_per_course / 5,
    bricks_per_course / 6,
    bricks_per_course / 7,
    bricks_per_course / 9,
    10
  ]
  initial_bricks + added_bricks - removed_bricks.sum

/-- Theorem stating that the total number of bricks in the wall is 4848 -/
theorem wall_bricks_count : total_bricks = 4848 := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l510_51057


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l510_51071

/-- Given an isosceles right triangle with perimeter 2p, its area is (3-2√2)p² -/
theorem isosceles_right_triangle_area (p : ℝ) (h : p > 0) : 
  ∃ (x : ℝ), 
    x > 0 ∧ 
    (2 * x + x * Real.sqrt 2 = 2 * p) ∧ 
    ((1 / 2) * x * x = (3 - 2 * Real.sqrt 2) * p^2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l510_51071


namespace NUMINAMATH_CALUDE_number_less_than_l510_51052

theorem number_less_than : (0.86 : ℝ) - 0.82 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_l510_51052


namespace NUMINAMATH_CALUDE_salary_increase_l510_51036

/-- Regression line for worker's salary with respect to labor productivity -/
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

/-- Theorem: When labor productivity increases by 1000 Yuan (1 unit in x), 
    the salary increases by 90 Yuan -/
theorem salary_increase (x : ℝ) : 
  regression_line (x + 1) - regression_line x = 90 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l510_51036


namespace NUMINAMATH_CALUDE_zoo_visitors_l510_51013

theorem zoo_visitors (num_cars : ℝ) (people_per_car : ℝ) :
  num_cars = 3.0 → people_per_car = 63.0 → num_cars * people_per_car = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l510_51013


namespace NUMINAMATH_CALUDE_max_profit_is_45_6_l510_51088

/-- Profit function for location A -/
def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def profit_B (x : ℝ) : ℝ := 2 * x

/-- Total number of cars sold -/
def total_cars : ℕ := 15

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (total_cars - x)

theorem max_profit_is_45_6 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_cars ∧
  ∀ y : ℝ, y ≥ 0 → y ≤ total_cars → total_profit y ≤ total_profit x ∧
  total_profit x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_45_6_l510_51088


namespace NUMINAMATH_CALUDE_box_filled_by_small_cubes_l510_51087

/-- Proves that a 1m³ box can be filled by 15625 cubes of 4cm edge length -/
theorem box_filled_by_small_cubes :
  let box_edge : ℝ := 1  -- 1 meter
  let small_cube_edge : ℝ := 0.04  -- 4 cm in meters
  let num_small_cubes : ℕ := 15625
  (box_edge ^ 3) = (small_cube_edge ^ 3) * num_small_cubes := by
  sorry

#check box_filled_by_small_cubes

end NUMINAMATH_CALUDE_box_filled_by_small_cubes_l510_51087


namespace NUMINAMATH_CALUDE_blue_garden_yield_l510_51021

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

theorem blue_garden_yield :
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_garden_yield_l510_51021


namespace NUMINAMATH_CALUDE_sphere_intersection_circles_area_sum_l510_51051

/-- Given a sphere of radius R and a point inside it at distance d from the center,
    the sum of the areas of three circles formed by the intersection of three
    mutually perpendicular planes passing through the point is equal to π(3R² - d²). -/
theorem sphere_intersection_circles_area_sum
  (R d : ℝ) (h_R : R > 0) (h_d : 0 ≤ d ∧ d < R) :
  ∃ (A : ℝ), A = π * (3 * R^2 - d^2) ∧
  ∀ (x y z : ℝ),
    x^2 + y^2 + z^2 = d^2 →
    A = π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_intersection_circles_area_sum_l510_51051


namespace NUMINAMATH_CALUDE_guitar_purchase_savings_l510_51062

/-- Proves that the difference in cost between Guitar Center and Sweetwater is $50 --/
theorem guitar_purchase_savings (retail_price : ℝ) 
  (gc_discount_rate : ℝ) (gc_shipping_fee : ℝ) 
  (sw_discount_rate : ℝ) :
  retail_price = 1000 →
  gc_discount_rate = 0.15 →
  gc_shipping_fee = 100 →
  sw_discount_rate = 0.10 →
  (retail_price * (1 - gc_discount_rate) + gc_shipping_fee) -
  (retail_price * (1 - sw_discount_rate)) = 50 := by
sorry

end NUMINAMATH_CALUDE_guitar_purchase_savings_l510_51062


namespace NUMINAMATH_CALUDE_melanie_catch_melanie_catch_is_ten_l510_51003

def sara_catch : ℕ := 5
def melanie_multiplier : ℕ := 2

theorem melanie_catch : ℕ := sara_catch * melanie_multiplier

theorem melanie_catch_is_ten : melanie_catch = 10 := by
  sorry

end NUMINAMATH_CALUDE_melanie_catch_melanie_catch_is_ten_l510_51003


namespace NUMINAMATH_CALUDE_composite_quotient_l510_51040

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product_list (l : List ℕ) : ℕ := l.foldl (·*·) 1

theorem composite_quotient :
  (product_list first_eight_composites) / (product_list next_eight_composites) = 1 / 1430 := by
  sorry

end NUMINAMATH_CALUDE_composite_quotient_l510_51040


namespace NUMINAMATH_CALUDE_pauline_car_count_l510_51078

/-- Represents the total number of matchbox cars Pauline has. -/
def total_cars : ℕ := 125

/-- Represents the number of convertible cars Pauline has. -/
def convertibles : ℕ := 35

/-- Represents the percentage of regular cars as a rational number. -/
def regular_cars_percent : ℚ := 64 / 100

/-- Represents the percentage of trucks as a rational number. -/
def trucks_percent : ℚ := 8 / 100

/-- Theorem stating that given the conditions, Pauline has 125 matchbox cars in total. -/
theorem pauline_car_count : 
  (regular_cars_percent + trucks_percent) * total_cars + convertibles = total_cars :=
sorry

end NUMINAMATH_CALUDE_pauline_car_count_l510_51078


namespace NUMINAMATH_CALUDE_inequality_solution_set_l510_51026

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 + 2*x < 3

-- Define the solution set
def solution_set : Set ℝ := {x | -3 < x ∧ x < 1}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l510_51026


namespace NUMINAMATH_CALUDE_zaras_estimate_bound_l510_51016

theorem zaras_estimate_bound (x y ε : ℝ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x - y < ε) 
  (h4 : ε > 0) : 
  (x + 2*ε) - (y - ε) < 2*ε := by
sorry

end NUMINAMATH_CALUDE_zaras_estimate_bound_l510_51016


namespace NUMINAMATH_CALUDE_number_calculation_l510_51015

theorem number_calculation (n : ℝ) : 
  (0.20 * 0.45 * 0.60 * 0.75 * n = 283.5) → n = 7000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l510_51015


namespace NUMINAMATH_CALUDE_quadratic_equation_and_inequality_l510_51039

theorem quadratic_equation_and_inequality 
  (a b : ℝ) 
  (h1 : (a:ℝ) * (-1/2)^2 + b * (-1/2) + 2 = 0)
  (h2 : (a:ℝ) * 2^2 + b * 2 + 2 = 0) :
  (a = -2 ∧ b = 3) ∧ 
  (∀ x : ℝ, a * x^2 + b * x - 1 > 0 ↔ 1/2 < x ∧ x < 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_and_inequality_l510_51039


namespace NUMINAMATH_CALUDE_andy_math_problem_l510_51045

theorem andy_math_problem (start_num end_num count : ℕ) : 
  end_num = 125 → count = 46 → end_num - start_num + 1 = count → start_num = 80 := by
  sorry

end NUMINAMATH_CALUDE_andy_math_problem_l510_51045


namespace NUMINAMATH_CALUDE_determinant_of_cubic_roots_l510_51084

theorem determinant_of_cubic_roots (p q r : ℝ) (a b c : ℝ) : 
  (∀ x : ℝ, x^3 + 3*p*x^2 + q*x + r = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  let matrix := !![a, b, c; b, c, a; c, a, b]
  Matrix.det matrix = 3*p*q := by
sorry

end NUMINAMATH_CALUDE_determinant_of_cubic_roots_l510_51084


namespace NUMINAMATH_CALUDE_sally_grew_six_carrots_l510_51012

/-- The number of carrots Fred grew -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots Sally grew -/
def sally_carrots : ℕ := total_carrots - fred_carrots

theorem sally_grew_six_carrots : sally_carrots = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_six_carrots_l510_51012


namespace NUMINAMATH_CALUDE_cos_B_value_triangle_area_l510_51043

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition_1 (t : Triangle) : Prop :=
  (Real.sin t.B) ^ 2 = (Real.sin t.A) * (Real.sin t.C) ∧ t.a = Real.sqrt 2 * t.b

def satisfies_condition_2 (t : Triangle) : Prop :=
  Real.cos t.B = 3 / 4 ∧ t.a = 2

-- Define the theorems to prove
theorem cos_B_value (t : Triangle) (h : satisfies_condition_1 t) :
  Real.cos t.B = 3 / 4 := by sorry

theorem triangle_area (t : Triangle) (h : satisfies_condition_2 t) :
  let area := 1 / 2 * t.a * t.c * Real.sin t.B
  area = Real.sqrt 7 / 4 ∨ area = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_cos_B_value_triangle_area_l510_51043


namespace NUMINAMATH_CALUDE_max_sum_at_vertex_l510_51041

/-- Represents a face of the cube -/
structure Face :=
  (number : ℕ)

/-- Represents a cube with six numbered faces -/
structure Cube :=
  (faces : Fin 6 → Face)
  (opposite_sum : ∀ i : Fin 3, (faces i).number + (faces (i + 3)).number = 10)

/-- Represents a vertex of the cube -/
structure Vertex :=
  (face1 : Face)
  (face2 : Face)
  (face3 : Face)

/-- The theorem stating the maximum sum at a vertex -/
theorem max_sum_at_vertex (c : Cube) : 
  (∃ v : Vertex, v.face1 ∈ Set.range c.faces ∧ 
                 v.face2 ∈ Set.range c.faces ∧ 
                 v.face3 ∈ Set.range c.faces ∧ 
                 v.face1 ≠ v.face2 ∧ v.face2 ≠ v.face3 ∧ v.face1 ≠ v.face3) →
  (∀ v : Vertex, v.face1 ∈ Set.range c.faces ∧ 
                v.face2 ∈ Set.range c.faces ∧ 
                v.face3 ∈ Set.range c.faces ∧ 
                v.face1 ≠ v.face2 ∧ v.face2 ≠ v.face3 ∧ v.face1 ≠ v.face3 →
                v.face1.number + v.face2.number + v.face3.number ≤ 22) :=
sorry

end NUMINAMATH_CALUDE_max_sum_at_vertex_l510_51041


namespace NUMINAMATH_CALUDE_marble_173_is_gray_l510_51060

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Defines the pattern of marbles -/
def marblePattern : List MarbleColor :=
  List.replicate 6 MarbleColor.Gray ++
  List.replicate 3 MarbleColor.White ++
  List.replicate 5 MarbleColor.Black

/-- Determines the color of the nth marble in the sequence -/
def nthMarbleColor (n : Nat) : MarbleColor :=
  let patternLength := marblePattern.length
  let indexInPattern := (n - 1) % patternLength
  marblePattern[indexInPattern]'
    (by
      have h : indexInPattern < patternLength := Nat.mod_lt _ (Nat.zero_lt_of_ne_zero (by decide))
      exact h
    )

/-- Theorem: The 173rd marble is gray -/
theorem marble_173_is_gray : nthMarbleColor 173 = MarbleColor.Gray := by
  sorry

end NUMINAMATH_CALUDE_marble_173_is_gray_l510_51060


namespace NUMINAMATH_CALUDE_problem_l510_51083

/-- Given m > 0, prove the following statements -/
theorem problem (m : ℝ) (hm : m > 0) :
  /- If (x+2)(x-6) ≤ 0 implies 2-m ≤ x ≤ 2+m for all x, then m ≥ 4 -/
  ((∀ x, (x + 2) * (x - 6) ≤ 0 → 2 - m ≤ x ∧ x ≤ 2 + m) → m ≥ 4) ∧
  /- If m = 5, and for all x, ((x+2)(x-6) ≤ 0) ∨ (-3 ≤ x ≤ 7) is true, 
     and ((x+2)(x-6) ≤ 0) ∧ (-3 ≤ x ≤ 7) is false, 
     then x ∈ [-3,-2) ∪ (6,7] -/
  (m = 5 → 
    (∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) ∧
           ¬((x + 2) * (x - 6) ≤ 0 ∧ -3 ≤ x ∧ x ≤ 7)) →
    (∀ x, x ∈ Set.Ioo (-3) (-2) ∪ Set.Ioc 6 7)) :=
by sorry

end NUMINAMATH_CALUDE_problem_l510_51083


namespace NUMINAMATH_CALUDE_constant_term_expansion_l510_51082

/-- The constant term in the expansion of (x - 3/x^2)^6 -/
def constant_term : ℕ := 135

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℕ := sorry

/-- Theorem: The constant term in the expansion of (x - 3/x^2)^6 is 135 -/
theorem constant_term_expansion :
  constant_term = binomial_coeff 6 2 * 3^2 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l510_51082


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l510_51058

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def OnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The focus of the parabola y^2 = 4x -/
def FocusOfParabola : Point :=
  ⟨1, 0⟩

/-- The orthocenter of a triangle -/
def Orthocenter (a b c : Point) : Point :=
  sorry  -- Definition of orthocenter

/-- The area of a triangle -/
def TriangleArea (a b c : Point) : ℝ :=
  sorry  -- Definition of triangle area

/-- The main theorem -/
theorem parabola_triangle_area :
  ∀ (A B : Point),
    OnParabola A →
    OnParabola B →
    Orthocenter ⟨0, 0⟩ A B = FocusOfParabola →
    TriangleArea ⟨0, 0⟩ A B = 10 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l510_51058


namespace NUMINAMATH_CALUDE_ellipse_t_squared_range_l510_51064

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the condition for points A, B, and P
def condition (A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  (A.1, A.2) + (B.1, B.2) = t • (P.1, P.2)

-- Define the inequality condition
def inequality (A B P : ℝ × ℝ) : Prop :=
  ‖(P.1 - A.1, P.2 - A.2) - (P.1 - B.1, P.2 - B.2)‖ < Real.sqrt 3

-- Theorem statement
theorem ellipse_t_squared_range :
  ∀ (A B P : ℝ × ℝ) (t : ℝ),
    ellipse A.1 A.2 → ellipse B.1 B.2 → ellipse P.1 P.2 →
    condition A B P t → inequality A B P →
    20 - Real.sqrt 283 < t^2 ∧ t^2 < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_t_squared_range_l510_51064


namespace NUMINAMATH_CALUDE_equation_solution_l510_51063

theorem equation_solution : ∃! x : ℝ, (81 : ℝ)^(x - 2) / (9 : ℝ)^(x - 2) = (27 : ℝ)^(3*x + 2) ∧ x = -10/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l510_51063


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l510_51096

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem stating the property of the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.a 2 + seq.S 3 = 4)
  (h2 : seq.a 3 + seq.S 5 = 12) :
  seq.a 4 + seq.S 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l510_51096


namespace NUMINAMATH_CALUDE_eliana_steps_proof_l510_51056

/-- The number of steps Eliana walked on the first day before adding 300 steps -/
def first_day_steps : ℕ := 200

/-- The total number of steps for all three days -/
def total_steps : ℕ := 1600

theorem eliana_steps_proof :
  first_day_steps + 300 + 2 * (first_day_steps + 300) + 100 = total_steps :=
by sorry

end NUMINAMATH_CALUDE_eliana_steps_proof_l510_51056


namespace NUMINAMATH_CALUDE_square_plus_double_perfect_square_l510_51006

theorem square_plus_double_perfect_square (a : ℕ) : 
  ∃ (k : ℕ), a^2 + 2*a = k^2 ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_square_plus_double_perfect_square_l510_51006


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l510_51010

/-- Proves that a faster speed allowing 20 km more distance in the same time as 50 km at 10 km/hr is 14 km/hr -/
theorem faster_speed_calculation (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 50 →
  actual_speed = 10 →
  additional_distance = 20 →
  ∃ (faster_speed : ℝ),
    (actual_distance / actual_speed = (actual_distance + additional_distance) / faster_speed) ∧
    faster_speed = 14 :=
by sorry

end NUMINAMATH_CALUDE_faster_speed_calculation_l510_51010


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l510_51059

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents income levels --/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with different income levels --/
structure Community where
  highIncome : Nat
  middleIncome : Nat
  lowIncome : Nat

/-- Represents a school class --/
structure SchoolClass where
  totalStudents : Nat
  specialtyType : String

/-- Determines the most appropriate sampling method for a community survey --/
def communitySamplingMethod (community : Community) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Determines the most appropriate sampling method for a school class survey --/
def schoolClassSamplingMethod (schoolClass : SchoolClass) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Theorem stating the appropriate sampling methods for the given surveys --/
theorem appropriate_sampling_methods
  (community : Community)
  (schoolClass : SchoolClass) :
  communitySamplingMethod {highIncome := 125, middleIncome := 280, lowIncome := 95} 100 = SamplingMethod.Stratified ∧
  schoolClassSamplingMethod {totalStudents := 15, specialtyType := "art"} 3 = SamplingMethod.SimpleRandom :=
  sorry

end NUMINAMATH_CALUDE_appropriate_sampling_methods_l510_51059


namespace NUMINAMATH_CALUDE_union_equality_implies_a_value_l510_51000

def A (a : ℝ) : Set ℝ := {2*a, 3}
def B : Set ℝ := {2, 3}

theorem union_equality_implies_a_value (a : ℝ) :
  A a ∪ B = {2, 3, 4} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_value_l510_51000


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l510_51097

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) (D : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) - Real.cos x ^ 2 + 1
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- D is on angle bisector of A
  2 * Real.cos (A / 2) * Real.sin (B / 2) = Real.sin (C / 2) →
  -- f(A) = 3/2
  f A = 3 / 2 →
  -- AD = √2 BD = 2
  2 * Real.sin (B / 2) = Real.sqrt 2 * Real.sin (C / 2) ∧
  2 * Real.sin (B / 2) = 2 * Real.sin ((B + C) / 2) →
  -- Conclusion
  Real.cos C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l510_51097


namespace NUMINAMATH_CALUDE_prime_even_intersection_l510_51094

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by
  sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l510_51094


namespace NUMINAMATH_CALUDE_gcd_98_63_l510_51038

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l510_51038


namespace NUMINAMATH_CALUDE_triangle_properties_l510_51031

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π/2) →
  (a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) →
  (c = Real.sqrt 11) →
  (Real.sin C = 2 * Real.sqrt 2 / 3) →
  (Real.sin A / Real.sin B = Real.sqrt 7) ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 14) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l510_51031


namespace NUMINAMATH_CALUDE_diagonal_sum_inequality_l510_51077

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define the sum of diagonal lengths for a quadrilateral
def sum_of_diagonals (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the "inside" relation for quadrilaterals
def inside (inner outer : ConvexQuadrilateral) : Prop := sorry

-- Theorem statement
theorem diagonal_sum_inequality {P P' : ConvexQuadrilateral} 
  (h_inside : inside P' P) : 
  sum_of_diagonals P' < 2 * sum_of_diagonals P := by
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_inequality_l510_51077


namespace NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l510_51080

/-- Two angles have the same terminal side if their difference is a multiple of 360 degrees -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β - α = k * 360

/-- The main theorem: -324° has the same terminal side as 36° -/
theorem negative_324_same_terminal_side_as_36 :
  same_terminal_side 36 (-324) := by
  sorry

end NUMINAMATH_CALUDE_negative_324_same_terminal_side_as_36_l510_51080


namespace NUMINAMATH_CALUDE_T_is_three_rays_l510_51024

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (5 = p.1 + 3 ∧ 5 ≥ p.2 - 6) ∨
    (5 = p.2 - 6 ∧ 5 ≥ p.1 + 3) ∨
    (p.1 + 3 = p.2 - 6 ∧ 5 ≥ p.1 + 3)}

-- Define the three rays
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 11}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 2 ∧ p.2 = 11}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 ≤ 2 ∧ p.2 ≤ 11}

-- Theorem statement
theorem T_is_three_rays : T = ray1 ∪ ray2 ∪ ray3 := by
  sorry

end NUMINAMATH_CALUDE_T_is_three_rays_l510_51024


namespace NUMINAMATH_CALUDE_larger_number_proof_l510_51068

/-- Given two positive integers with HCF 23 and LCM factors 13 and 16, prove the larger number is 368 -/
theorem larger_number_proof (a b : ℕ) : 
  a > 0 → b > 0 → Nat.gcd a b = 23 → Nat.lcm a b = 23 * 13 * 16 → max a b = 368 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l510_51068


namespace NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l510_51022

/-- Calculates the percentage of female officers on duty -/
def percentage_female_officers_on_duty (total_on_duty : ℕ) (female_ratio_on_duty : ℚ) (total_female_officers : ℕ) : ℚ :=
  (female_ratio_on_duty * total_on_duty : ℚ) / total_female_officers * 100

/-- Theorem stating that the percentage of female officers on duty is 20% -/
theorem female_officers_on_duty_percentage 
  (total_on_duty : ℕ) 
  (female_ratio_on_duty : ℚ) 
  (total_female_officers : ℕ) 
  (h1 : total_on_duty = 100)
  (h2 : female_ratio_on_duty = 1/2)
  (h3 : total_female_officers = 250) :
  percentage_female_officers_on_duty total_on_duty female_ratio_on_duty total_female_officers = 20 :=
sorry

end NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l510_51022


namespace NUMINAMATH_CALUDE_inscribed_square_area_l510_51066

/-- The area of a square inscribed in the ellipse x^2/4 + y^2/8 = 1, 
    with sides parallel to the coordinate axes. -/
theorem inscribed_square_area : 
  ∃ (s : ℝ), s > 0 ∧ 
  (s^2 / 4 + s^2 / 8 = 1) ∧ 
  (4 * s^2 = 32 / 3) := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l510_51066


namespace NUMINAMATH_CALUDE_tadpoles_let_go_75_percent_l510_51011

/-- The percentage of tadpoles let go, given the total caught and number kept -/
def tadpoles_let_go_percentage (total : ℕ) (kept : ℕ) : ℚ :=
  (total - kept : ℚ) / total * 100

/-- Theorem stating that the percentage of tadpoles let go is 75% -/
theorem tadpoles_let_go_75_percent (total : ℕ) (kept : ℕ) 
  (h1 : total = 180) (h2 : kept = 45) : 
  tadpoles_let_go_percentage total kept = 75 := by
  sorry

#eval tadpoles_let_go_percentage 180 45

end NUMINAMATH_CALUDE_tadpoles_let_go_75_percent_l510_51011


namespace NUMINAMATH_CALUDE_max_drumming_bunnies_l510_51086

/-- Represents a drum with a specific size -/
structure Drum where
  size : ℕ

/-- Represents a pair of drumsticks with a specific length -/
structure Drumsticks where
  length : ℕ

/-- Represents a bunny with its assigned drum and drumsticks -/
structure Bunny where
  drum : Drum
  sticks : Drumsticks

/-- Determines if a bunny can drum based on its drum and sticks compared to another bunny -/
def canDrum (b1 b2 : Bunny) : Prop :=
  b1.drum.size > b2.drum.size ∧ b1.sticks.length > b2.sticks.length

theorem max_drumming_bunnies 
  (bunnies : Fin 7 → Bunny)
  (h_diff_drums : ∀ i j, i ≠ j → (bunnies i).drum.size ≠ (bunnies j).drum.size)
  (h_diff_sticks : ∀ i j, i ≠ j → (bunnies i).sticks.length ≠ (bunnies j).sticks.length) :
  ∃ (drummers : Finset (Fin 7)),
    drummers.card = 6 ∧
    ∀ i ∈ drummers, ∃ j, canDrum (bunnies i) (bunnies j) :=
by
  sorry

end NUMINAMATH_CALUDE_max_drumming_bunnies_l510_51086


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l510_51050

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + 7 * y = 14
def line2 (k x y : ℝ) : Prop := k * x - y = k + 1

-- Define the intersection point
def intersection (k : ℝ) : Prop :=
  ∃ x y : ℝ, line1 x y ∧ line2 k x y

-- Define the first quadrant condition
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem intersection_in_first_quadrant :
  ∀ k : ℝ, (∃ x y : ℝ, intersection k ∧ first_quadrant x y) → k > 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l510_51050


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_3_and_4_l510_51019

theorem smallest_five_digit_multiple_of_3_and_4 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  3 ∣ n ∧ 
  4 ∣ n ∧ 
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) → 3 ∣ m → 4 ∣ m → m ≥ n) ∧
  n = 10008 :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_3_and_4_l510_51019


namespace NUMINAMATH_CALUDE_smallest_x_value_l510_51047

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (240 + x)) :
  x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l510_51047


namespace NUMINAMATH_CALUDE_probability_of_ravi_selection_l510_51091

theorem probability_of_ravi_selection 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 5/7) 
  (h2 : p_both = 0.14285714285714288) : 
  p_both / p_ram = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_ravi_selection_l510_51091


namespace NUMINAMATH_CALUDE_contrapositive_quadratic_roots_l510_51069

theorem contrapositive_quadratic_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 - b * x + c = 0 → x > 0) → a * c > 0
  ↔
  a * c ≤ 0 → ∃ x : ℝ, a * x^2 - b * x + c = 0 ∧ x ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_quadratic_roots_l510_51069


namespace NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_necessary_not_sufficient_l510_51072

theorem abs_sum_eq_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) := by sorry

end NUMINAMATH_CALUDE_abs_sum_eq_sum_abs_necessary_not_sufficient_l510_51072


namespace NUMINAMATH_CALUDE_fraction_sum_reciprocal_l510_51075

theorem fraction_sum_reciprocal (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = (x * y) / (y + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_reciprocal_l510_51075


namespace NUMINAMATH_CALUDE_john_notebooks_l510_51070

/-- Calculates the maximum number of notebooks that can be purchased with a given amount of money, considering a bulk discount. -/
def max_notebooks (total_cents : ℕ) (notebook_price : ℕ) (discount : ℕ) (bulk_size : ℕ) : ℕ :=
  let discounted_price := notebook_price - discount
  let bulk_set_price := discounted_price * bulk_size
  let bulk_sets := total_cents / bulk_set_price
  let remaining_cents := total_cents % bulk_set_price
  let additional_notebooks := remaining_cents / notebook_price
  bulk_sets * bulk_size + additional_notebooks

/-- Proves that given 2545 cents, with notebooks costing 235 cents each and a 15 cent discount
    per notebook when bought in sets of 5, the maximum number of notebooks that can be purchased is 11. -/
theorem john_notebooks : max_notebooks 2545 235 15 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_notebooks_l510_51070


namespace NUMINAMATH_CALUDE_sticker_count_l510_51009

def ryan_stickers : ℕ := 30

def steven_stickers (ryan : ℕ) : ℕ := 3 * ryan

def terry_stickers (steven : ℕ) : ℕ := steven + 20

def total_stickers (ryan steven terry : ℕ) : ℕ := ryan + steven + terry

theorem sticker_count :
  total_stickers ryan_stickers (steven_stickers ryan_stickers) (terry_stickers (steven_stickers ryan_stickers)) = 230 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l510_51009


namespace NUMINAMATH_CALUDE_chess_tournament_wins_l510_51095

theorem chess_tournament_wins (total_games : ℕ) (total_points : ℚ)
  (h1 : total_games = 20)
  (h2 : total_points = 12.5) :
  ∃ (wins losses draws : ℕ),
    wins + losses + draws = total_games ∧
    wins - losses = 5 ∧
    wins + draws / 2 = total_points := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_wins_l510_51095


namespace NUMINAMATH_CALUDE_daisy_percentage_in_bouquet_l510_51067

theorem daisy_percentage_in_bouquet : 
  ∀ (total_flowers : ℕ) (white_flowers yellow_flowers white_tulips yellow_tulips white_daisies yellow_daisies : ℕ),
  total_flowers > 0 →
  white_flowers + yellow_flowers = total_flowers →
  white_tulips + white_daisies = white_flowers →
  yellow_tulips + yellow_daisies = yellow_flowers →
  white_tulips = white_flowers / 2 →
  yellow_daisies = (2 * yellow_flowers) / 3 →
  white_flowers = (7 * total_flowers) / 10 →
  (white_daisies + yellow_daisies) * 100 = 55 * total_flowers :=
by sorry

end NUMINAMATH_CALUDE_daisy_percentage_in_bouquet_l510_51067


namespace NUMINAMATH_CALUDE_expression_simplification_l510_51005

theorem expression_simplification :
  (((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l510_51005


namespace NUMINAMATH_CALUDE_profit_achieved_l510_51025

/-- The number of pencils purchased -/
def num_purchased : ℕ := 1800

/-- The cost of each pencil when purchased -/
def cost_per_pencil : ℚ := 15 / 100

/-- The selling price of each pencil -/
def selling_price : ℚ := 30 / 100

/-- The desired profit -/
def desired_profit : ℚ := 150

/-- The number of pencils that must be sold to make the desired profit -/
def num_sold : ℕ := 1400

theorem profit_achieved : 
  (num_sold : ℚ) * selling_price - (num_purchased : ℚ) * cost_per_pencil = desired_profit := by
  sorry

end NUMINAMATH_CALUDE_profit_achieved_l510_51025


namespace NUMINAMATH_CALUDE_largest_perfect_square_product_l510_51007

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A function that checks if a number is a one-digit positive integer -/
def is_one_digit_positive (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

/-- The main theorem stating that 144 is the largest perfect square
    that can be written as the product of three different one-digit positive integers -/
theorem largest_perfect_square_product : 
  (∀ a b c : ℕ, 
    is_one_digit_positive a ∧ 
    is_one_digit_positive b ∧ 
    is_one_digit_positive c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square (a * b * c) →
    a * b * c ≤ 144) ∧
  (∃ a b c : ℕ,
    is_one_digit_positive a ∧
    is_one_digit_positive b ∧
    is_one_digit_positive c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c = 144 ∧
    is_perfect_square 144) :=
by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_product_l510_51007


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l510_51023

/-- 
Given positive integers a, b, and c with no two having a common divisor greater than 1,
2abc-ab-bc-ca is the largest integer that cannot be expressed as xbc+yca+zab 
for non-negative integers x, y, z
-/
theorem largest_non_representable_integer (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : Nat.gcd a b = 1) (hbc : Nat.gcd b c = 1) (hca : Nat.gcd c a = 1) :
  (∀ x y z : ℕ, 2*a*b*c - a*b - b*c - c*a ≠ x*b*c + y*c*a + z*a*b) ∧
  (∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
    ∃ x y z : ℕ, n = x*b*c + y*c*a + z*a*b) := by
  sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l510_51023


namespace NUMINAMATH_CALUDE_baker_bread_rolls_l510_51008

theorem baker_bread_rolls (regular_rolls : ℕ) (regular_flour : ℚ) 
  (new_rolls : ℕ) (new_flour : ℚ) :
  regular_rolls = 40 →
  regular_flour = 1 / 8 →
  new_rolls = 25 →
  regular_rolls * regular_flour = new_rolls * new_flour →
  new_flour = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_baker_bread_rolls_l510_51008


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l510_51017

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l510_51017


namespace NUMINAMATH_CALUDE_fourth_berry_count_l510_51035

/-- A sequence of berry counts where the difference between consecutive terms increases by 2 -/
def BerrySequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = (a (n + 1) - a n) + 2

theorem fourth_berry_count
  (a : ℕ → ℕ)
  (seq : BerrySequence a)
  (first : a 0 = 3)
  (second : a 1 = 4)
  (third : a 2 = 7)
  (fifth : a 4 = 19) :
  a 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_berry_count_l510_51035


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l510_51054

/-- Given two solutions with different alcohol concentrations, prove that mixing them in specific quantities results in a desired alcohol concentration. -/
theorem alcohol_mixture_proof (x_volume : ℝ) (y_volume : ℝ) (x_concentration : ℝ) (y_concentration : ℝ) (target_concentration : ℝ) :
  x_volume = 300 →
  y_volume = 200 →
  x_concentration = 0.1 →
  y_concentration = 0.3 →
  target_concentration = 0.18 →
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = target_concentration :=
by sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l510_51054


namespace NUMINAMATH_CALUDE_count_hexagons_l510_51061

/-- The number of regular hexagons in a larger hexagon -/
def num_hexagons (n : ℕ+) : ℚ :=
  (n^2 + n : ℚ)^2 / 4

/-- Theorem: The number of regular hexagons with vertices among the vertices of equilateral triangles
    in a regular hexagon of side length n is (n² + n)² / 4 -/
theorem count_hexagons (n : ℕ+) :
  num_hexagons n = (n^2 + n : ℚ)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_count_hexagons_l510_51061


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l510_51014

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l510_51014


namespace NUMINAMATH_CALUDE_difference_solution_eq_one_difference_solution_eq_two_difference_solution_eq_three_l510_51076

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  b / a = b - a

/-- Theorem for 4x = m -/
theorem difference_solution_eq_one (m : ℝ) :
  is_difference_solution_equation 4 m ↔ m = 16 / 3 := by sorry

/-- Theorem for 4x = ab + a -/
theorem difference_solution_eq_two (a b : ℝ) :
  is_difference_solution_equation 4 (a * b + a) → 3 * (a * b + a) = 16 := by sorry

/-- Theorem for 4x = mn + m and -2x = mn + n -/
theorem difference_solution_eq_three (m n : ℝ) :
  is_difference_solution_equation 4 (m * n + m) →
  is_difference_solution_equation (-2) (m * n + n) →
  3 * (m * n + m) - 9 * (m * n + n)^2 = 0 := by sorry

end NUMINAMATH_CALUDE_difference_solution_eq_one_difference_solution_eq_two_difference_solution_eq_three_l510_51076


namespace NUMINAMATH_CALUDE_remaining_balance_proof_l510_51089

def gift_card_balance (initial_balance : ℚ) (latte_price : ℚ) (croissant_price : ℚ) 
  (days : ℕ) (cookie_price : ℚ) (num_cookies : ℕ) : ℚ :=
  initial_balance - (latte_price + croissant_price) * days - cookie_price * num_cookies

theorem remaining_balance_proof :
  gift_card_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_proof_l510_51089


namespace NUMINAMATH_CALUDE_logistics_center_equidistant_l510_51099

def rectilinear_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

def town_A : ℝ × ℝ := (2, 3)
def town_B : ℝ × ℝ := (-6, 9)
def town_C : ℝ × ℝ := (-3, -8)
def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_equidistant :
  let (x, y) := logistics_center
  rectilinear_distance x y town_A.1 town_A.2 =
  rectilinear_distance x y town_B.1 town_B.2 ∧
  rectilinear_distance x y town_B.1 town_B.2 =
  rectilinear_distance x y town_C.1 town_C.2 :=
by sorry

end NUMINAMATH_CALUDE_logistics_center_equidistant_l510_51099


namespace NUMINAMATH_CALUDE_order_of_abc_l510_51018

theorem order_of_abc : 
  let a : ℝ := 2017^0
  let b : ℝ := 2015 * 2017 - 2016^2
  let c : ℝ := (-2/3)^2016 * (3/2)^2017
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l510_51018


namespace NUMINAMATH_CALUDE_circle_area_triple_radius_l510_51020

theorem circle_area_triple_radius (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A := by sorry

end NUMINAMATH_CALUDE_circle_area_triple_radius_l510_51020


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l510_51037

/-- A geometric sequence with positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_eq : a 1 * a 9 = 2 * a 52)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l510_51037


namespace NUMINAMATH_CALUDE_marcel_potatoes_l510_51033

theorem marcel_potatoes (marcel_corn : ℕ) (dale_potatoes : ℕ) (total_vegetables : ℕ)
  (h1 : marcel_corn = 10)
  (h2 : total_vegetables = 27)
  (h3 : dale_potatoes = 8) :
  total_vegetables - (marcel_corn + marcel_corn / 2 + dale_potatoes) = 4 := by
sorry

end NUMINAMATH_CALUDE_marcel_potatoes_l510_51033


namespace NUMINAMATH_CALUDE_three_correct_deliveries_l510_51030

def num_houses : ℕ := 5
def num_packages : ℕ := 5

def probability_three_correct : ℚ := 1 / 6

theorem three_correct_deliveries :
  let total_arrangements := num_houses.factorial
  let correct_three_ways := num_houses.choose 3
  let incorrect_two_ways := 1  -- derangement of 2
  let prob_three_correct := correct_three_ways * incorrect_two_ways / total_arrangements
  prob_three_correct = probability_three_correct := by sorry

end NUMINAMATH_CALUDE_three_correct_deliveries_l510_51030


namespace NUMINAMATH_CALUDE_hyperbola_foci_product_l510_51090

/-- Given a hyperbola C with equation x²/9 - y²/m = 1, foci F₁ and F₂, and a point P on C
    such that PF₁ · PF₂ = 0, if the directrix of the parabola y² = 16x passes through a focus of C,
    then |PF₁| * |PF₂| = 14 -/
theorem hyperbola_foci_product (m : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  (∃ x y, x^2 / 9 - y^2 / m = 1 ∧ (x, y) = P) →  -- P is on the hyperbola
  (F₁.1 = -4 ∧ F₁.2 = 0) →  -- F₁ is (-4, 0)
  (F₂.1 = 4 ∧ F₂.2 = 0) →   -- F₂ is (4, 0)
  ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) →  -- PF₁ · PF₂ = 0
  (‖P - F₁‖ * ‖P - F₂‖ = 14) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_foci_product_l510_51090


namespace NUMINAMATH_CALUDE_f_properties_l510_51034

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 11

-- State the theorem
theorem f_properties :
  -- Part 1: Tangent line at x = 1 is y = 5
  (∀ y, (y - f 1 = 0 * (x - 1)) ↔ y = 5) ∧
  -- Part 2: Monotonicity intervals
  (∀ x, x < -1 → (deriv f) x > 0) ∧
  (∀ x, x > 1 → (deriv f) x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → (deriv f) x < 0) ∧
  -- Part 3: Maximum value on [-1, 1] is 17
  (∀ x, -1 ≤ x ∧ x ≤ 1 → f x ≤ 17) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ f x = 17) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l510_51034


namespace NUMINAMATH_CALUDE_union_complement_equals_reals_l510_51002

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 > 2*x + 3}

noncomputable def B : Set ℝ := {x | Real.log x / Real.log 3 > 1}

theorem union_complement_equals_reals : A ∪ (U \ B) = U := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_reals_l510_51002


namespace NUMINAMATH_CALUDE_husband_additional_payment_l510_51092

def medical_procedure_1 : ℚ := 128
def medical_procedure_2 : ℚ := 256
def medical_procedure_3 : ℚ := 64
def house_help_salary : ℚ := 160
def tax_rate : ℚ := 0.05

def total_medical_expenses : ℚ := medical_procedure_1 + medical_procedure_2 + medical_procedure_3
def couple_medical_contribution : ℚ := total_medical_expenses / 2
def house_help_medical_contribution : ℚ := min (total_medical_expenses / 2) house_help_salary
def tax_deduction : ℚ := house_help_salary * tax_rate
def total_couple_expense : ℚ := couple_medical_contribution + (total_medical_expenses / 2 - house_help_medical_contribution) + tax_deduction
def husband_paid : ℚ := couple_medical_contribution

theorem husband_additional_payment (
  split_equally : total_couple_expense / 2 < husband_paid
) : husband_paid - total_couple_expense / 2 = 76 := by sorry

end NUMINAMATH_CALUDE_husband_additional_payment_l510_51092


namespace NUMINAMATH_CALUDE_problem_solution_l510_51098

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l510_51098


namespace NUMINAMATH_CALUDE_equation_solutions_l510_51027

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (1/2 * (x₁ - 3)^2 = 18 ∧ x₁ = 9) ∧
                (1/2 * (x₂ - 3)^2 = 18 ∧ x₂ = -3)) ∧
  (∃ y₁ y₂ : ℝ, (y₁^2 + 6*y₁ = 5 ∧ y₁ = -3 + Real.sqrt 14) ∧
                (y₂^2 + 6*y₂ = 5 ∧ y₂ = -3 - Real.sqrt 14)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l510_51027


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l510_51042

/-- Represents the angles of a quadrilateral in arithmetic sequence -/
structure QuadrilateralAngles where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- Conditions for the quadrilateral angles -/
def quadrilateral_conditions (q : QuadrilateralAngles) : Prop :=
  q.a > 0 ∧
  q.d > 0 ∧
  q.a + (q.a + q.d) + (q.a + 2 * q.d) + (q.a + 3 * q.d) = 360 ∧
  q.a + (q.a + 2 * q.d) = 160

theorem smallest_angle_measure (q : QuadrilateralAngles) 
  (h : quadrilateral_conditions q) : q.a = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l510_51042


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_sum_of_ratios_equals_zero_l510_51032

-- Question 1
theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ p ≠ q →
    (a * Real.log (p + 2) - (p + 1)^2 - (a * Real.log (q + 2) - (q + 1)^2)) / (p - q) > 1) →
  a ≥ 28 :=
sorry

-- Question 2
theorem sum_of_ratios_equals_zero (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let f := fun x => (x - a) * (x - b) * (x - c)
  let f' := fun x => 3 * x^2 - 2 * (a + b + c) * x + (a * b + b * c + c * a)
  a / (f' a) + b / (f' b) + c / (f' c) = 0 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_sum_of_ratios_equals_zero_l510_51032


namespace NUMINAMATH_CALUDE_seafood_noodles_plates_l510_51044

/-- Given a chef's banquet with a total of 55 plates, 25 plates of lobster rolls,
    and 14 plates of spicy hot noodles, prove that the number of seafood noodle plates is 16. -/
theorem seafood_noodles_plates (total : ℕ) (lobster : ℕ) (spicy : ℕ) (seafood : ℕ)
  (h1 : total = 55)
  (h2 : lobster = 25)
  (h3 : spicy = 14)
  (h4 : total = lobster + spicy + seafood) :
  seafood = 16 := by
  sorry

end NUMINAMATH_CALUDE_seafood_noodles_plates_l510_51044


namespace NUMINAMATH_CALUDE_clay_target_permutations_l510_51046

theorem clay_target_permutations : 
  (Nat.factorial 9) / ((Nat.factorial 3) * (Nat.factorial 3) * (Nat.factorial 3)) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_clay_target_permutations_l510_51046


namespace NUMINAMATH_CALUDE_point_on_double_angle_l510_51081

/-- Given a point P(-1, 2) on the terminal side of angle α, 
    prove that the point (-3, -4) lies on the terminal side of angle 2α. -/
theorem point_on_double_angle (α : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  let cos_α : ℝ := P.1 / r
  let sin_α : ℝ := P.2 / r
  let cos_2α : ℝ := cos_α^2 - sin_α^2
  let sin_2α : ℝ := 2 * sin_α * cos_α
  let Q : ℝ × ℝ := (-3, -4)
  (∃ k : ℝ, k > 0 ∧ Q.1 = k * cos_2α ∧ Q.2 = k * sin_2α) :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_double_angle_l510_51081


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_twelve_l510_51048

theorem factorial_ratio_equals_twelve : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_twelve_l510_51048


namespace NUMINAMATH_CALUDE_factorial_vs_power_l510_51028

theorem factorial_vs_power : 100^200 > Nat.factorial 200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_vs_power_l510_51028


namespace NUMINAMATH_CALUDE_turtleneck_discount_l510_51029

theorem turtleneck_discount (C : ℝ) (C_pos : C > 0) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_profit := 0.41
  let initial_price := C * (1 + initial_markup)
  let new_year_price := initial_price * (1 + new_year_markup)
  let february_price := C * (1 + february_profit)
  let discount := 1 - (february_price / new_year_price)
  discount = 0.06 := by
sorry

end NUMINAMATH_CALUDE_turtleneck_discount_l510_51029


namespace NUMINAMATH_CALUDE_cuboid_volume_l510_51085

theorem cuboid_volume (a b c : ℝ) (h1 : a * b = 2) (h2 : b * c = 6) (h3 : a * c = 9) :
  a * b * c = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l510_51085


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l510_51065

theorem quadratic_equation_range (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, 2 * x^2 - 2 * x + 3 * m - 1 = 0) →
  (x₁ * x₂ > x₁ + x₂ - 4) →
  (-5/3 < m ∧ m ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l510_51065
