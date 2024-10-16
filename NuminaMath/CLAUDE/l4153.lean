import Mathlib

namespace NUMINAMATH_CALUDE_brad_reads_more_than_greg_l4153_415349

/-- Greg's daily reading pages -/
def greg_pages : ℕ := 18

/-- Brad's daily reading pages -/
def brad_pages : ℕ := 26

/-- The difference in pages read between Brad and Greg -/
def page_difference : ℕ := brad_pages - greg_pages

theorem brad_reads_more_than_greg : page_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_brad_reads_more_than_greg_l4153_415349


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l4153_415368

/-- A right triangle with an inscribed rectangle -/
structure InscribedRectangle where
  /-- Length of AB in the right triangle AGD -/
  ab : ℝ
  /-- Length of CD in the right triangle AGD -/
  cd : ℝ
  /-- Length of BC in the inscribed rectangle BCFE -/
  bc : ℝ
  /-- Length of FE in the inscribed rectangle BCFE -/
  fe : ℝ
  /-- BC is parallel to AD -/
  bc_parallel_ad : True
  /-- FE is parallel to AD -/
  fe_parallel_ad : True
  /-- Length of BC is one-third of FE -/
  bc_one_third_fe : bc = fe / 3
  /-- AB = 40 units -/
  ab_eq_40 : ab = 40
  /-- CD = 70 units -/
  cd_eq_70 : cd = 70

/-- The area of the inscribed rectangle BCFE is 2800 square units -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) : 
  rect.bc * rect.fe = 2800 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l4153_415368


namespace NUMINAMATH_CALUDE_twins_age_problem_l4153_415357

theorem twins_age_problem :
  ∀ (x y : ℕ),
  x * x = 8 →
  (x + y) * (x + y) = x * x + 17 →
  y = 3 :=
by sorry

end NUMINAMATH_CALUDE_twins_age_problem_l4153_415357


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l4153_415337

def ellipse_equation (x y : ℝ) : Prop := x^2 / 36 + y^2 / 20 = 1

theorem ellipse_standard_equation 
  (major_axis : ℝ) (eccentricity : ℝ) (foci_on_x_axis : Prop) :
  major_axis = 12 → eccentricity = 2/3 → foci_on_x_axis →
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    x^2 / ((major_axis/2)^2) + y^2 / ((major_axis/2)^2 * (1 - eccentricity^2)) = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l4153_415337


namespace NUMINAMATH_CALUDE_max_sum_abc_l4153_415311

-- Define the set of digits
def Digit := Fin 10

-- Define the property that A, B, C, D are different digits
def are_different (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Define the property that (A+B+C)/D is an integer
def is_integer_fraction (A B C D : Digit) : Prop :=
  ∃ k : ℕ, k * D.val = A.val + B.val + C.val

-- Theorem statement
theorem max_sum_abc (A B C D : Digit) 
  (h1 : are_different A B C D)
  (h2 : is_integer_fraction A B C D) :
  A.val + B.val + C.val ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_sum_abc_l4153_415311


namespace NUMINAMATH_CALUDE_nuts_in_third_box_l4153_415378

theorem nuts_in_third_box (A B C : ℝ) 
  (h1 : A = B + C - 6) 
  (h2 : B = A + C - 10) : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_nuts_in_third_box_l4153_415378


namespace NUMINAMATH_CALUDE_painting_supplies_theorem_l4153_415340

/-- Represents the cost and quantity of painting supplies -/
structure PaintingSupplies where
  brush_cost : ℝ
  board_cost : ℝ
  total_items : ℕ
  max_cost : ℝ

/-- Theorem stating the properties of the painting supplies purchase -/
theorem painting_supplies_theorem (ps : PaintingSupplies) 
  (h1 : 340 / ps.brush_cost = 300 / ps.board_cost)
  (h2 : ps.brush_cost = ps.board_cost + 2)
  (h3 : ps.total_items = 30)
  (h4 : ∀ a : ℕ, a ≤ ps.total_items → 
    ps.brush_cost * (ps.total_items - a) + ps.board_cost * a ≤ ps.max_cost) :
  ps.brush_cost = 17 ∧ ps.board_cost = 15 ∧ 
  (∃ min_boards : ℕ, min_boards = 18 ∧ 
    ∀ a : ℕ, a < min_boards → 
      ps.brush_cost * (ps.total_items - a) + ps.board_cost * a > ps.max_cost) := by
  sorry

#check painting_supplies_theorem

end NUMINAMATH_CALUDE_painting_supplies_theorem_l4153_415340


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4153_415304

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem quadratic_function_properties :
  (f (-1) = 0 ∧ f 3 = 0 ∧ f 0 = -3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 4 → f x ≤ 2*m) ↔ 5/2 ≤ m) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l4153_415304


namespace NUMINAMATH_CALUDE_problem_solution_l4153_415390

theorem problem_solution : 
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18 = 3 * Real.sqrt 2) ∧ 
  ((7 - 4 * Real.sqrt 3) * (7 + 4 * Real.sqrt 3) - (Real.sqrt 3 - 1)^2 + (1/3)⁻¹ = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4153_415390


namespace NUMINAMATH_CALUDE_household_cats_count_l4153_415392

/-- Represents a cat in the household -/
structure Cat where
  kittens : ℕ
  male_kittens : ℕ
  female_kittens : ℕ
  grandkittens : ℕ

/-- The total number of cats in the household -/
def total_cats (cats : List Cat) : ℕ :=
  cats.length + (cats.map (λ c => c.kittens + c.grandkittens)).sum

/-- Theorem stating the total number of cats in the household -/
theorem household_cats_count :
  let cat_a : Cat := { kittens := 4, male_kittens := 2, female_kittens := 2, grandkittens := 2 }
  let cat_b : Cat := { kittens := 3, male_kittens := 1, female_kittens := 2, grandkittens := 0 }
  let cat_c : Cat := { kittens := 5, male_kittens := 3, female_kittens := 2, grandkittens := 0 }
  let household : List Cat := [cat_a, cat_b, cat_c]
  total_cats household = 17 := by
  sorry


end NUMINAMATH_CALUDE_household_cats_count_l4153_415392


namespace NUMINAMATH_CALUDE_triangle_properties_l4153_415396

/-- An acute triangle with side lengths a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- The main theorem about the specific triangle -/
theorem triangle_properties (t : AcuteTriangle)
    (h1 : t.a = 2)
    (h2 : 2 * Real.sin t.A = Real.sin t.C)
    (h3 : Real.cos t.C = 1/4) :
    t.c = 4 ∧ (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4153_415396


namespace NUMINAMATH_CALUDE_min_formula_l4153_415376

theorem min_formula (a b : ℝ) : min a b = (a + b - Real.sqrt ((a - b)^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_formula_l4153_415376


namespace NUMINAMATH_CALUDE_exactly_one_false_l4153_415313

theorem exactly_one_false :
  (∀ a b : ℝ, a ≥ b ∧ b > -1 → a / (1 + a) ≥ b / (1 + b)) ∧
  (∀ m n : ℕ+, m ≤ n → Real.sqrt (m * (n - m)) ≤ n / 2) ∧
  ¬(∀ a b x₁ y₁ : ℝ, x₁^2 + y₁^2 = 9 ∧ (a - x₁)^2 + (b - y₁)^2 = 1 →
    ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 9 ∧ (p.1 - a)^2 + (p.2 - b)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_false_l4153_415313


namespace NUMINAMATH_CALUDE_subset_sum_modulo_l4153_415362

theorem subset_sum_modulo (N : ℕ) (A : Finset ℕ) :
  A.card = N →
  A ⊆ Finset.range (N^2) →
  ∃ (B : Finset ℕ), 
    B.card = N ∧ 
    B ⊆ Finset.range (N^2) ∧ 
    ((A.product B).image (λ (p : ℕ × ℕ) => (p.1 + p.2) % (N^2))).card ≥ N^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_modulo_l4153_415362


namespace NUMINAMATH_CALUDE_investment_rate_proof_l4153_415398

def total_investment : ℝ := 17000
def investment_at_4_percent : ℝ := 12000
def total_interest : ℝ := 1380
def known_rate : ℝ := 0.04

theorem investment_rate_proof :
  let remaining_investment := total_investment - investment_at_4_percent
  let interest_at_4_percent := investment_at_4_percent * known_rate
  let remaining_interest := total_interest - interest_at_4_percent
  let unknown_rate := remaining_interest / remaining_investment
  unknown_rate = 0.18 := by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l4153_415398


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l4153_415336

def total_players : ℕ := 15
def preselected_players : ℕ := 3
def lineup_size : ℕ := 5

theorem starting_lineup_combinations :
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = 66 :=
by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l4153_415336


namespace NUMINAMATH_CALUDE_star_to_maltese_cross_l4153_415308

/-- Represents a four-pointed star shape -/
structure FourPointedStar :=
  (center : Point)
  (vertices : Fin 4 → Point)

/-- Represents a Maltese cross shape -/
structure MalteseCross :=
  (center : Point)
  (arms : Fin 4 → Point)

/-- Represents a part of the cut star -/
structure StarPart :=
  (vertex : Point)
  (center : Point)

/-- Function to cut a star into four parts -/
def cutStar (star : FourPointedStar) : Fin 4 → StarPart :=
  sorry

/-- Function to arrange star parts into a Maltese cross -/
def arrangeParts (parts : Fin 4 → StarPart) : MalteseCross :=
  sorry

/-- Theorem stating that a four-pointed star can be cut and rearranged into a Maltese cross -/
theorem star_to_maltese_cross (star : FourPointedStar) :
  ∃ (cross : MalteseCross), arrangeParts (cutStar star) = cross :=
sorry

end NUMINAMATH_CALUDE_star_to_maltese_cross_l4153_415308


namespace NUMINAMATH_CALUDE_honey_container_size_l4153_415342

/-- The number of ounces in Tabitha's honey container -/
def honey_container_ounces : ℕ :=
  let servings_per_cup : ℕ := 1
  let cups_per_night : ℕ := 2
  let servings_per_ounce : ℕ := 6
  let nights_honey_lasts : ℕ := 48
  (servings_per_cup * cups_per_night * nights_honey_lasts) / servings_per_ounce

theorem honey_container_size :
  honey_container_ounces = 16 := by
  sorry

end NUMINAMATH_CALUDE_honey_container_size_l4153_415342


namespace NUMINAMATH_CALUDE_polynomial_equality_l4153_415348

theorem polynomial_equality (a b c d e : ℝ) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = (2*x - 1)^4) →
  a + c = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4153_415348


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_length_l4153_415334

/-- Given a right triangle ABC with legs AB and AC, and points X on AB and Y on AC,
    prove that the hypotenuse BC has length 6√42 under specific conditions. -/
theorem right_triangle_hypotenuse_length 
  (A B C X Y : ℝ × ℝ) -- Points in 2D plane
  (h_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) -- Right angle at A
  (h_X_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (t * B.1 + (1 - t) * A.1, t * B.2 + (1 - t) * A.2))
  (h_Y_on_AC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = (s * C.1 + (1 - s) * A.1, s * C.2 + (1 - s) * A.2))
  (h_AX_XB : dist A X = (1/4) * dist A B)
  (h_AY_YC : dist A Y = (2/3) * dist A C)
  (h_BY : dist B Y = 24)
  (h_CX : dist C X = 18) :
  dist B C = 6 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_length_l4153_415334


namespace NUMINAMATH_CALUDE_angle_420_equivalent_to_60_l4153_415380

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem angle_420_equivalent_to_60 :
  same_terminal_side 420 60 :=
sorry

end NUMINAMATH_CALUDE_angle_420_equivalent_to_60_l4153_415380


namespace NUMINAMATH_CALUDE_remainder_13754_div_11_l4153_415315

theorem remainder_13754_div_11 : 13754 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13754_div_11_l4153_415315


namespace NUMINAMATH_CALUDE_cubes_fill_box_l4153_415352

/-- Proves that 2-inch cubes fill 100% of a 8×6×12 inch box -/
theorem cubes_fill_box (box_length box_width box_height cube_side: ℕ) 
  (h1: box_length = 8)
  (h2: box_width = 6)
  (h3: box_height = 12)
  (h4: cube_side = 2)
  (h5: box_length % cube_side = 0)
  (h6: box_width % cube_side = 0)
  (h7: box_height % cube_side = 0) :
  (((box_length / cube_side) * (box_width / cube_side) * (box_height / cube_side)) * cube_side^3) / (box_length * box_width * box_height) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cubes_fill_box_l4153_415352


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4153_415389

/-- Given a quadratic equation x^2 + (a+1)x + 4 = 0 with roots x₁ and x₂, where x₁ = 1 + √3i and a ∈ ℝ,
    prove that a = -3 and the distance between the points corresponding to x₁ and x₂ in the complex plane is 2√3. -/
theorem quadratic_equation_roots (a : ℝ) (x₁ x₂ : ℂ) : 
  x₁^2 + (a+1)*x₁ + 4 = 0 ∧ 
  x₂^2 + (a+1)*x₂ + 4 = 0 ∧
  x₁ = 1 + Complex.I * Real.sqrt 3 →
  a = -3 ∧ 
  Complex.abs (x₁ - x₂) = Real.sqrt 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4153_415389


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l4153_415373

-- Define a structure for the circle arrangement
structure CircleArrangement where
  numbers : List ℕ
  connections : List (ℕ × ℕ)

-- Define the property of valid ratios for connected circles
def validConnectedRatio (a b : ℕ) : Prop :=
  a / b = 3 ∨ a / b = 9 ∨ b / a = 3 ∨ b / a = 9

-- Define the property of invalid ratios for unconnected circles
def invalidUnconnectedRatio (a b : ℕ) : Prop :=
  a / b ≠ 3 ∧ a / b ≠ 9 ∧ b / a ≠ 3 ∧ b / a ≠ 9

-- Define the property of a valid circle arrangement
def validArrangement (arr : CircleArrangement) : Prop :=
  (∀ (a b : ℕ), (a, b) ∈ arr.connections → validConnectedRatio a b) ∧
  (∀ (a b : ℕ), a ∈ arr.numbers ∧ b ∈ arr.numbers ∧ (a, b) ∉ arr.connections → invalidUnconnectedRatio a b)

-- Theorem stating the existence of a valid arrangement
theorem exists_valid_arrangement : ∃ (arr : CircleArrangement), validArrangement arr :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l4153_415373


namespace NUMINAMATH_CALUDE_min_value_expression_l4153_415358

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 4) :
  (2 * x / y) + (3 * y / z) + (4 * z / x) ≥ 6 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 4 ∧
    (2 * a / b) + (3 * b / c) + (4 * c / a) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l4153_415358


namespace NUMINAMATH_CALUDE_reasonable_statements_correct_l4153_415329

-- Define the set of all statements
inductive Statement : Type
  | A : Statement
  | B : Statement
  | C : Statement
  | D : Statement

-- Define the properties of each statement
def is_sampling_method (s : Statement) : Prop :=
  match s with
  | Statement.A => true
  | _ => false

def is_key_to_sampling (s : Statement) : Prop :=
  match s with
  | Statement.B => true
  | _ => false

def is_for_mobile_animals (s : Statement) : Prop :=
  match s with
  | Statement.C => true
  | _ => false

def reflects_population_trend (s : Statement) : Prop :=
  match s with
  | Statement.D => true
  | _ => false

-- Define the set of reasonable statements
def reasonable_statements : Set Statement :=
  {s | is_sampling_method s ∨ is_key_to_sampling s ∨ is_for_mobile_animals s}

-- Theorem: The set of reasonable statements is equal to {A, B, C}
theorem reasonable_statements_correct :
  reasonable_statements = {Statement.A, Statement.B, Statement.C} := by
  sorry

end NUMINAMATH_CALUDE_reasonable_statements_correct_l4153_415329


namespace NUMINAMATH_CALUDE_max_a_value_l4153_415347

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -24) → 
  (a > 0) → 
  a ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l4153_415347


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l4153_415355

/-- A point on the grid --/
structure Point where
  x : Int
  y : Int

/-- A triangle defined by three points --/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Check if a triangle is isosceles --/
def isIsosceles (t : Triangle) : Bool :=
  let d12 := (t.p1.x - t.p2.x)^2 + (t.p1.y - t.p2.y)^2
  let d23 := (t.p2.x - t.p3.x)^2 + (t.p2.y - t.p3.y)^2
  let d31 := (t.p3.x - t.p1.x)^2 + (t.p3.y - t.p1.y)^2
  d12 = d23 || d23 = d31 || d31 = d12

/-- The list of triangles on the grid --/
def triangles : List Triangle := [
  { p1 := { x := 1, y := 6 }, p2 := { x := 3, y := 6 }, p3 := { x := 2, y := 3 } },
  { p1 := { x := 4, y := 2 }, p2 := { x := 4, y := 4 }, p3 := { x := 6, y := 2 } },
  { p1 := { x := 0, y := 0 }, p2 := { x := 3, y := 1 }, p3 := { x := 6, y := 0 } },
  { p1 := { x := 7, y := 3 }, p2 := { x := 6, y := 5 }, p3 := { x := 9, y := 3 } },
  { p1 := { x := 8, y := 0 }, p2 := { x := 9, y := 2 }, p3 := { x := 10, y := 0 } }
]

theorem four_isosceles_triangles :
  (triangles.filter isIsosceles).length = 4 := by sorry


end NUMINAMATH_CALUDE_four_isosceles_triangles_l4153_415355


namespace NUMINAMATH_CALUDE_point_upper_left_of_line_l4153_415383

/-- A point in the plane is represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane is represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a point is on the upper left side of a line -/
def isUpperLeftSide (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

/-- The main theorem -/
theorem point_upper_left_of_line (t : ℝ) :
  let p : Point := ⟨-2, t⟩
  let l : Line := ⟨1, -1, 4⟩
  isUpperLeftSide p l → t > 2 := by
  sorry


end NUMINAMATH_CALUDE_point_upper_left_of_line_l4153_415383


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l4153_415391

theorem cube_root_equation_solution :
  ∃ y : ℝ, (30 * y + (30 * y + 24) ^ (1/3)) ^ (1/3) = 24 ∧ y = 460 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l4153_415391


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4153_415328

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 - b * x + a > 0} = Set.Ioo (-1/2) (-1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4153_415328


namespace NUMINAMATH_CALUDE_equation_solution_l4153_415379

theorem equation_solution :
  let f (x : ℝ) := (6*x + 3) / (3*x^2 + 6*x - 9) - 3*x / (3*x - 3)
  ∀ x : ℝ, x ≠ 1 → (f x = 0 ↔ x = (3 + Real.sqrt 21) / 2 ∨ x = (3 - Real.sqrt 21) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4153_415379


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_existence_l4153_415394

/-- Given a circle of radius r tangent to the legs of a right angle,
    there exists a point M on the circumference forming a rectangle MPOQ
    with perimeter 2p if and only if r(2 - √2) ≤ p ≤ r(2 + √2) -/
theorem tangent_circle_rectangle_existence (r p : ℝ) (hr : r > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = p ∧ x^2 + y^2 = 2*r*p) ↔
  r*(2 - Real.sqrt 2) ≤ p ∧ p ≤ r*(2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_rectangle_existence_l4153_415394


namespace NUMINAMATH_CALUDE_ravi_work_time_l4153_415324

-- Define the work completion times
def prakash_time : ℝ := 75
def combined_time : ℝ := 30

-- Define Ravi's time as a variable
def ravi_time : ℝ := 50

-- Theorem statement
theorem ravi_work_time :
  (1 / ravi_time + 1 / prakash_time = 1 / combined_time) →
  ravi_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_ravi_work_time_l4153_415324


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l4153_415317

/-- Given two vectors a and b in R², where a = (1, 0) and b = (-1, m),
    if a is perpendicular to (m * a - b), then m = -1. -/
theorem perpendicular_vectors_imply_m_value :
  let a : Fin 2 → ℝ := ![1, 0]
  let b : Fin 2 → ℝ := ![-1, m]
  (∀ i : Fin 2, (a i) * ((m * a i) - (b i)) = 0) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l4153_415317


namespace NUMINAMATH_CALUDE_value_of_Y_l4153_415399

theorem value_of_Y : ∀ P Q Y : ℚ,
  P = 6036 / 2 →
  Q = P / 4 →
  Y = P - 3 * Q →
  Y = 754.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_Y_l4153_415399


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4153_415338

theorem quadratic_inequality_solution_set (x : ℝ) :
  3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4153_415338


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4153_415312

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_condition : 2*a + 2*b + 2*c = 3) :
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 2 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ 
    2*a₀ + 2*b₀ + 2*c₀ = 3 ∧
    1 / (2*a₀ + b₀) + 1 / (2*b₀ + c₀) + 1 / (2*c₀ + a₀) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l4153_415312


namespace NUMINAMATH_CALUDE_total_books_l4153_415339

theorem total_books (tim_books mike_books : ℕ) 
  (h1 : tim_books = 22) 
  (h2 : mike_books = 20) : 
  tim_books + mike_books = 42 := by
sorry

end NUMINAMATH_CALUDE_total_books_l4153_415339


namespace NUMINAMATH_CALUDE_certain_number_exists_l4153_415333

theorem certain_number_exists : ∃ (n : ℕ), n > 0 ∧ 49 % n = 4 ∧ 66 % n = 6 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l4153_415333


namespace NUMINAMATH_CALUDE_haley_extra_tickets_l4153_415341

/-- The number of extra concert tickets Haley bought -/
def extra_tickets (ticket_price : ℕ) (tickets_for_friends : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / ticket_price) - tickets_for_friends

/-- Theorem: Haley bought 5 extra tickets -/
theorem haley_extra_tickets :
  extra_tickets 4 3 32 = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_extra_tickets_l4153_415341


namespace NUMINAMATH_CALUDE_hyperbola_condition_l4153_415319

theorem hyperbola_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (1 + k) - y^2 / (1 - k) = 1 ∧ 
   ((1 + k > 0 ∧ 1 - k > 0) ∨ (1 + k < 0 ∧ 1 - k < 0))) ↔ 
  -1 < k ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l4153_415319


namespace NUMINAMATH_CALUDE_no_rectangle_solution_l4153_415332

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_rectangle_solution : ¬∃ (x y : ℕ), 
  is_prime x ∧ is_prime y ∧ 
  x < y ∧ y < 6 ∧ 
  2 * (x + y) = 21 ∧ 
  x * y = 45 :=
sorry

end NUMINAMATH_CALUDE_no_rectangle_solution_l4153_415332


namespace NUMINAMATH_CALUDE_horizontal_shift_l4153_415395

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the shift amount
variable (a : ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Theorem statement
theorem horizontal_shift (h : y = f x) :
  y = f (x - a) ↔ y = f ((x + a) - a) :=
sorry

end NUMINAMATH_CALUDE_horizontal_shift_l4153_415395


namespace NUMINAMATH_CALUDE_least_sum_of_primes_l4153_415331

theorem least_sum_of_primes (p q : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  p > 1 → 
  q > 1 → 
  15 * (p^2 + 1) = 29 * (q^2 + 1) → 
  ∃ (p' q' : ℕ), Nat.Prime p' ∧ Nat.Prime q' ∧ p' > 1 ∧ q' > 1 ∧
    15 * (p'^2 + 1) = 29 * (q'^2 + 1) ∧
    p' + q' = 14 ∧
    ∀ (p'' q'' : ℕ), Nat.Prime p'' → Nat.Prime q'' → p'' > 1 → q'' > 1 →
      15 * (p''^2 + 1) = 29 * (q''^2 + 1) → p'' + q'' ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_primes_l4153_415331


namespace NUMINAMATH_CALUDE_square_fence_poles_l4153_415365

theorem square_fence_poles (poles_per_side : ℕ) (h : poles_per_side = 27) :
  poles_per_side * 4 - 4 = 104 :=
by sorry

end NUMINAMATH_CALUDE_square_fence_poles_l4153_415365


namespace NUMINAMATH_CALUDE_complement_union_theorem_l4153_415385

def U : Set Int := {-1, 0, 1, 2, 3}
def P : Set Int := {0, 1, 2}
def Q : Set Int := {-1, 0}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {-1, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l4153_415385


namespace NUMINAMATH_CALUDE_fair_payment_division_l4153_415354

/-- Represents the payment for digging the trench -/
def total_payment : ℚ := 2

/-- Represents Abraham's digging rate relative to Benjamin's soil throwing rate -/
def abraham_dig_rate : ℚ := 1

/-- Represents Benjamin's digging rate relative to Abraham's soil throwing rate -/
def benjamin_dig_rate : ℚ := 4

/-- Represents the ratio of Abraham's payment to the total payment -/
def abraham_payment_ratio : ℚ := 1/3

/-- Represents the ratio of Benjamin's payment to the total payment -/
def benjamin_payment_ratio : ℚ := 2/3

/-- Theorem stating the fair division of payment between Abraham and Benjamin -/
theorem fair_payment_division :
  abraham_payment_ratio * total_payment = 2/3 ∧
  benjamin_payment_ratio * total_payment = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_fair_payment_division_l4153_415354


namespace NUMINAMATH_CALUDE_quadratic_transformation_has_integer_roots_l4153_415369

/-- Represents a quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Checks if a quadratic polynomial has integer roots -/
def has_integer_roots (p : QuadraticPolynomial) : Prop :=
  ∃ (x : ℤ), p.a * x^2 + p.b * x + p.c = 0

/-- Represents a single step in the transformation process -/
inductive TransformationStep
  | IncreaseX
  | DecreaseX
  | IncreaseConstant
  | DecreaseConstant

/-- Applies a transformation step to a polynomial -/
def apply_step (p : QuadraticPolynomial) (step : TransformationStep) : QuadraticPolynomial :=
  match step with
  | TransformationStep.IncreaseX => { a := p.a, b := p.b + 1, c := p.c }
  | TransformationStep.DecreaseX => { a := p.a, b := p.b - 1, c := p.c }
  | TransformationStep.IncreaseConstant => { a := p.a, b := p.b, c := p.c + 1 }
  | TransformationStep.DecreaseConstant => { a := p.a, b := p.b, c := p.c - 1 }

theorem quadratic_transformation_has_integer_roots 
  (initial : QuadraticPolynomial)
  (final : QuadraticPolynomial)
  (h_initial : initial = { a := 1, b := 10, c := 20 })
  (h_final : final = { a := 1, b := 20, c := 10 })
  (h_transform : ∃ (steps : List TransformationStep), 
    final = steps.foldl apply_step initial) :
  ∃ (intermediate : QuadraticPolynomial),
    (∃ (steps : List TransformationStep), intermediate = steps.foldl apply_step initial) ∧
    has_integer_roots intermediate :=
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_has_integer_roots_l4153_415369


namespace NUMINAMATH_CALUDE_last_second_occurrence_is_two_l4153_415384

-- Define the Fibonacci sequence modulo 10
def fib_mod_10 : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => (fib_mod_10 n + fib_mod_10 (n + 1)) % 10

-- Define a function to check if a digit has appeared at least twice up to a given index
def appears_twice (d : ℕ) (n : ℕ) : Prop :=
  ∃ i j, i < j ∧ j ≤ n ∧ fib_mod_10 i = d ∧ fib_mod_10 j = d

-- State the theorem
theorem last_second_occurrence_is_two :
  ∀ d, d ≠ 2 → ∃ n, appears_twice d n ∧ ¬appears_twice 2 n :=
sorry

end NUMINAMATH_CALUDE_last_second_occurrence_is_two_l4153_415384


namespace NUMINAMATH_CALUDE_base5_multiplication_l4153_415310

/-- Converts a base 5 number to its decimal equivalent -/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base 5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base5_multiplication (a b : List Nat) :
  decimalToBase5 (base5ToDecimal a * base5ToDecimal b) = [2, 1, 3, 4] :=
  by sorry

end NUMINAMATH_CALUDE_base5_multiplication_l4153_415310


namespace NUMINAMATH_CALUDE_parallel_line_slope_l4153_415377

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ (m : ℝ), m = (1 : ℝ) / 2 ∧ ∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x + c := by
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l4153_415377


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l4153_415305

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l4153_415305


namespace NUMINAMATH_CALUDE_girls_attending_event_l4153_415322

theorem girls_attending_event (total_students : ℕ) (total_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1500) 
  (h2 : total_attendees = 900) (h3 : girls + boys = total_students) 
  (h4 : (3 * girls) / 5 + (2 * boys) / 3 = total_attendees) : 
  (3 * girls) / 5 = 900 := by
  sorry

end NUMINAMATH_CALUDE_girls_attending_event_l4153_415322


namespace NUMINAMATH_CALUDE_car_speed_conversion_l4153_415359

/-- Converts speed from m/s to km/h -/
def speed_ms_to_kmh (speed_ms : ℝ) : ℝ := speed_ms * 3.6

/-- Given a car's speed of 10 m/s, its speed in km/h is 36 km/h -/
theorem car_speed_conversion :
  let speed_ms : ℝ := 10
  speed_ms_to_kmh speed_ms = 36 := by sorry

end NUMINAMATH_CALUDE_car_speed_conversion_l4153_415359


namespace NUMINAMATH_CALUDE_s_99_digits_l4153_415366

/-- s(n) is an n-digit number formed by attaching the first n perfect squares, in order, into one integer. -/
def s (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The theorem states that s(99) has 189 digits -/
theorem s_99_digits : num_digits (s 99) = 189 := by sorry

end NUMINAMATH_CALUDE_s_99_digits_l4153_415366


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l4153_415360

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions for a valid triangle
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the quadratic expression
def quadratic_expr (t : Triangle) (x : ℝ) : ℝ :=
  t.b^2 * x^2 + (t.b^2 + t.c^2 - t.a^2) * x + t.c^2

-- Theorem statement
theorem quadratic_always_positive (t : Triangle) :
  ∀ x : ℝ, quadratic_expr t x > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l4153_415360


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l4153_415303

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (15 * m - 1) % 11 = 0 → n ≤ m) ∧ 
  (15 * n - 1) % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l4153_415303


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l4153_415307

/-- A parabola defined by y = 3x² - 6x + c -/
def parabola (x y c : ℝ) : Prop := y = 3 * x^2 - 6 * x + c

/-- Three points on the parabola -/
def point_A (y₁ c : ℝ) : Prop := parabola (-3) y₁ c
def point_B (y₂ c : ℝ) : Prop := parabola (-1) y₂ c
def point_C (y₃ c : ℝ) : Prop := parabola 5 y₃ c

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_point_relationship (y₁ y₂ y₃ c : ℝ) 
  (hA : point_A y₁ c) (hB : point_B y₂ c) (hC : point_C y₃ c) :
  y₁ = y₃ ∧ y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l4153_415307


namespace NUMINAMATH_CALUDE_max_b_value_l4153_415397

theorem max_b_value (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 →
  b ≤ 9 ∧ ∃ (a₀ b₀ : ℤ), (a₀ + b₀)^2 + a₀*(a₀ + b₀) + b₀ = 0 ∧ b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l4153_415397


namespace NUMINAMATH_CALUDE_decimal_power_division_l4153_415300

theorem decimal_power_division : (0.4 : ℝ)^4 / (0.04 : ℝ)^3 = 400 := by
  sorry

end NUMINAMATH_CALUDE_decimal_power_division_l4153_415300


namespace NUMINAMATH_CALUDE_marias_water_bottles_l4153_415320

/-- Calculates the final number of water bottles Maria has -/
def final_bottle_count (initial : ℕ) (drunk : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Proves that Maria's final bottle count is correct -/
theorem marias_water_bottles (initial : ℕ) (drunk : ℕ) (bought : ℕ) 
  (h1 : initial ≥ drunk) : 
  final_bottle_count initial drunk bought = initial - drunk + bought :=
by
  sorry

#eval final_bottle_count 14 8 45

end NUMINAMATH_CALUDE_marias_water_bottles_l4153_415320


namespace NUMINAMATH_CALUDE_scientific_notation_700_3_l4153_415372

/-- Definition of scientific notation -/
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

/-- Theorem: 700.3 in scientific notation -/
theorem scientific_notation_700_3 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation 700.3 a n ∧ a = 7.003 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_700_3_l4153_415372


namespace NUMINAMATH_CALUDE_parabola_locus_l4153_415375

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of parabola C -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P lies on parabola C -/
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola_C P.1 P.2

/-- Vector relation between P, Q, and F -/
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1, P.2 - Q.2) = (2*(focus.1 - Q.1), 2*(focus.2 - Q.2))

/-- Curve E: 9y² = 12x - 8 -/
def curve_E (x y : ℝ) : Prop := 9*y^2 = 12*x - 8

theorem parabola_locus :
  ∀ Q : ℝ × ℝ,
  (∃ P : ℝ × ℝ, point_on_parabola P ∧ vector_relation P Q) →
  curve_E Q.1 Q.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_locus_l4153_415375


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l4153_415382

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (m + 3) / (x - 1) = 1) → 
  (m > -4 ∧ m ≠ -3) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l4153_415382


namespace NUMINAMATH_CALUDE_origami_distribution_l4153_415318

theorem origami_distribution (total_papers : ℕ) (num_cousins : ℕ) (papers_per_cousin : ℕ) 
    (h1 : total_papers = 48)
    (h2 : num_cousins = 6)
    (h3 : papers_per_cousin * num_cousins = total_papers) :
  papers_per_cousin = 8 := by
  sorry

end NUMINAMATH_CALUDE_origami_distribution_l4153_415318


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l4153_415387

/-- The complex number i(1+i) corresponds to a point in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l4153_415387


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l4153_415351

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l4153_415351


namespace NUMINAMATH_CALUDE_final_height_in_feet_l4153_415393

def initial_height : ℕ := 66
def growth_rate : ℕ := 2
def growth_duration : ℕ := 3
def inches_per_foot : ℕ := 12

theorem final_height_in_feet :
  (initial_height + growth_rate * growth_duration) / inches_per_foot = 6 :=
by sorry

end NUMINAMATH_CALUDE_final_height_in_feet_l4153_415393


namespace NUMINAMATH_CALUDE_divisibility_by_seventeen_l4153_415321

theorem divisibility_by_seventeen (x : ℤ) (y z w : ℕ) 
  (hy : Odd y) (hz : Odd z) (hw : Odd w) : 
  ∃ k : ℤ, x^(y^(z^w)) - x^(y^z) = 17 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_seventeen_l4153_415321


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l4153_415344

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (4 * (x - 1) > 3 * x - 2) ∧ (2 * x - 3 ≤ 5)}
  S = {3, 4} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l4153_415344


namespace NUMINAMATH_CALUDE_equal_pay_implies_harry_worked_33_hours_l4153_415350

/-- Payment structure for an employee -/
structure PaymentStructure where
  base_rate : ℝ
  base_hours : ℕ
  overtime_multiplier : ℝ

/-- Calculate the total pay for an employee given their payment structure and hours worked -/
def calculate_pay (ps : PaymentStructure) (hours_worked : ℕ) : ℝ :=
  let base_pay := ps.base_rate * (min ps.base_hours hours_worked)
  let overtime_hours := max 0 (hours_worked - ps.base_hours)
  let overtime_pay := ps.base_rate * ps.overtime_multiplier * overtime_hours
  base_pay + overtime_pay

theorem equal_pay_implies_harry_worked_33_hours 
  (x : ℝ) 
  (harry_structure : PaymentStructure)
  (james_structure : PaymentStructure)
  (h_harry : harry_structure = { base_rate := x, base_hours := 15, overtime_multiplier := 1.5 })
  (h_james : james_structure = { base_rate := x, base_hours := 40, overtime_multiplier := 2 })
  (james_hours : ℕ)
  (h_james_hours : james_hours = 41)
  (harry_hours : ℕ)
  (h_equal_pay : calculate_pay harry_structure harry_hours = calculate_pay james_structure james_hours) :
  harry_hours = 33 := by
  sorry

end NUMINAMATH_CALUDE_equal_pay_implies_harry_worked_33_hours_l4153_415350


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4153_415323

theorem sqrt_equation_solution :
  ∃ x : ℝ, 3 * Real.sqrt (x + 15) = 36 ∧ x = 129 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4153_415323


namespace NUMINAMATH_CALUDE_line_l_equation_tangent_circle_a_l4153_415364

-- Define the lines and circle
def l1 (x y : ℝ) : Prop := 2 * x - y = 1
def l2 (x y : ℝ) : Prop := x + 2 * y = 3
def l3 (x y : ℝ) : Prop := x - y + 1 = 0
def C (x y a : ℝ) : Prop := (x - a)^2 + y^2 = 8 ∧ a > 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statements
theorem line_l_equation :
  (∀ x y : ℝ, l1 x y ∧ l2 x y → (x, y) = P) →
  (∀ x y : ℝ, l x y → l3 ((x + 2) / 2) ((2 - x) / 2)) →
  ∀ x y : ℝ, l x y ↔ x + y - 2 = 0 := by sorry

theorem tangent_circle_a :
  (∀ x y : ℝ, l x y → C x y 6) →
  (∀ x y a : ℝ, l x y → C x y a → a = 6) := by sorry

end NUMINAMATH_CALUDE_line_l_equation_tangent_circle_a_l4153_415364


namespace NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l4153_415335

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((5 : ℚ) / 9) = 27 / 35 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l4153_415335


namespace NUMINAMATH_CALUDE_third_number_in_first_set_l4153_415327

theorem third_number_in_first_set (x : ℝ) : 
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_first_set_l4153_415327


namespace NUMINAMATH_CALUDE_gigi_jellybean_count_gigi_has_15_jellybeans_l4153_415309

theorem gigi_jellybean_count : ℕ → ℕ → ℕ → Prop :=
  fun gigi rory lorelai =>
    (rory = gigi + 30) →
    (lorelai = 3 * (gigi + rory)) →
    (lorelai = 180) →
    (gigi = 15)

theorem gigi_has_15_jellybeans :
  ∀ (gigi rory lorelai : ℕ),
    gigi_jellybean_count gigi rory lorelai :=
by
  sorry

end NUMINAMATH_CALUDE_gigi_jellybean_count_gigi_has_15_jellybeans_l4153_415309


namespace NUMINAMATH_CALUDE_option_A_correct_option_C_correct_l4153_415301

-- Define the set M
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

-- Define the set B
def B : Set ℤ := {b | ∃ n : ℕ, b = 2*n + 1}

-- Theorem for option A
theorem option_A_correct : ∀ a₁ a₂ : ℤ, a₁ ∈ M → a₂ ∈ M → (a₁ * a₂) ∈ M := by
  sorry

-- Theorem for option C
theorem option_C_correct : B ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_option_A_correct_option_C_correct_l4153_415301


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l4153_415371

theorem pen_cost_calculation (pack_size : ℕ) (pack_cost : ℚ) (desired_pens : ℕ) : 
  pack_size = 150 → pack_cost = 45 → desired_pens = 3600 →
  (desired_pens : ℚ) * (pack_cost / pack_size) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_calculation_l4153_415371


namespace NUMINAMATH_CALUDE_benjamins_house_paintable_area_l4153_415325

/-- Represents the dimensions of a room --/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area in Benjamin's house --/
def total_paintable_area (
  num_bedrooms : ℕ
  ) (room_dims : RoomDimensions)
  (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (room_dims.length * room_dims.height + room_dims.width * room_dims.height)
  let paintable_area_per_room := wall_area - unpaintable_area
  num_bedrooms * paintable_area_per_room

/-- Theorem stating the total paintable area in Benjamin's house --/
theorem benjamins_house_paintable_area :
  total_paintable_area 4 ⟨14, 12, 9⟩ 70 = 1592 := by
  sorry

end NUMINAMATH_CALUDE_benjamins_house_paintable_area_l4153_415325


namespace NUMINAMATH_CALUDE_mode_of_data_set_l4153_415346

def data_set : List Int := [-1, 0, 2, -1, 3]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.argmax (fun x => l.count x)

theorem mode_of_data_set :
  mode data_set = some (-1) := by
  sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l4153_415346


namespace NUMINAMATH_CALUDE_aye_aye_friendship_l4153_415330

theorem aye_aye_friendship (n : ℕ) (h : n = 23) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ 
  (∃ (f : Fin n → Finset (Fin n)), 
    (∀ x, x ∉ f x) ∧
    (∀ x y, y ∈ f x ↔ x ∈ f y) ∧
    (f i).card = k ∧ (f j).card = k) :=
by sorry


end NUMINAMATH_CALUDE_aye_aye_friendship_l4153_415330


namespace NUMINAMATH_CALUDE_solution_interval_l4153_415388

theorem solution_interval (c : ℝ) : (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ c ∈ Set.Ici (-4) ∩ Set.Iio (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l4153_415388


namespace NUMINAMATH_CALUDE_ratio_of_w_to_y_l4153_415381

theorem ratio_of_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 3 / 2)
  (hz_x : z / x = 1 / 3) :
  w / y = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_w_to_y_l4153_415381


namespace NUMINAMATH_CALUDE_inequality_proof_l4153_415343

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4153_415343


namespace NUMINAMATH_CALUDE_part1_part2_l4153_415367

-- Define complex numbers
def z1 (m : ℝ) : ℂ := m - 2*Complex.I
def z2 (n : ℝ) : ℂ := 1 - n*Complex.I

-- Part 1
theorem part1 : Complex.abs (z1 1 + z2 (-1)) = Real.sqrt 5 := by sorry

-- Part 2
theorem part2 : z1 0 = (z2 1)^2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l4153_415367


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4153_415316

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n % 11 = 10 ∧
  ∀ m : ℕ, m > 0 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 ∧
    m % 11 = 10 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l4153_415316


namespace NUMINAMATH_CALUDE_largest_k_phi_sigma_power_two_l4153_415370

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem largest_k_phi_sigma_power_two :
  (∀ k : ℕ, k > 31 → phi (sigma (2^k)) ≠ 2^k) ∧
  phi (sigma (2^31)) = 2^31 := by sorry

end NUMINAMATH_CALUDE_largest_k_phi_sigma_power_two_l4153_415370


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l4153_415374

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ ∀ M : ℝ, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1/a' + 1/b' > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l4153_415374


namespace NUMINAMATH_CALUDE_even_implies_symmetric_at_most_one_intersection_l4153_415302

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define symmetry about a point
def symmetric_about (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Theorem 1: If f(x+1) is even, then f(x) is symmetric about x = 1
theorem even_implies_symmetric :
  is_even (fun x ↦ f (x + 1)) → symmetric_about f 1 := by sorry

-- Theorem 2: Any function has at most one intersection with a vertical line
theorem at_most_one_intersection (a : ℝ) :
  ∃! y, f y = a := by sorry

end NUMINAMATH_CALUDE_even_implies_symmetric_at_most_one_intersection_l4153_415302


namespace NUMINAMATH_CALUDE_lemonade_proportion_l4153_415314

/-- Given that 24 lemons make 32 gallons of lemonade, proves that 3 lemons make 4 gallons -/
theorem lemonade_proportion :
  (24 : ℚ) / 32 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_lemonade_proportion_l4153_415314


namespace NUMINAMATH_CALUDE_triangle_side_length_l4153_415363

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4153_415363


namespace NUMINAMATH_CALUDE_function_inequality_solution_l4153_415353

-- Define the function f
noncomputable def f (x : ℝ) (p q a b : ℝ) (h : ℝ → ℝ) : ℝ :=
  if q > 0 then
    (a * x) / (1 - q) + h x * q^x + b / (1 - q) - a * p / ((1 - q)^2)
  else
    (a * x) / (1 - q) + h x * (-q)^x + b / (1 - q) - a * p / ((1 - q)^2)

-- State the theorem
theorem function_inequality_solution (p q a b : ℝ) (h : ℝ → ℝ) :
  q ≠ 1 →
  (∀ x, q > 0 → h (x + p) ≥ h x) →
  (∀ x, q < 0 → h (x + p) ≥ -h x) →
  (∀ x, f (x + p) p q a b h - q * f x p q a b h ≥ a * x + b) ↔
  (∀ x, f x p q a b h = if q > 0 then
    (a * x) / (1 - q) + h x * q^x + b / (1 - q) - a * p / ((1 - q)^2)
  else
    (a * x) / (1 - q) + h x * (-q)^x + b / (1 - q) - a * p / ((1 - q)^2)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l4153_415353


namespace NUMINAMATH_CALUDE_ellipse_midpoint_theorem_l4153_415386

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Defines a line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) := {p : ℝ × ℝ | p.2 = m * (p.1 - x₀) + y₀}

theorem ellipse_midpoint_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := Ellipse a b
  let L := Line (4/5) 3 0
  (0, 4) ∈ C ∧ 
  (a^2 - b^2) / a^2 = 9/25 →
  ∃ p q : ℝ × ℝ, p ∈ C ∧ p ∈ L ∧ q ∈ C ∧ q ∈ L ∧ 
  (p.1 + q.1) / 2 = 3/2 ∧ (p.2 + q.2) / 2 = -6/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_midpoint_theorem_l4153_415386


namespace NUMINAMATH_CALUDE_a_value_proof_l4153_415361

theorem a_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l4153_415361


namespace NUMINAMATH_CALUDE_log_difference_decreases_l4153_415356

theorem log_difference_decreases (m n : ℕ) (h : m > n) :
  Real.log (1 + 1 / (m : ℝ)) < Real.log (1 + 1 / (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_log_difference_decreases_l4153_415356


namespace NUMINAMATH_CALUDE_floor_of_3_point_9_l4153_415345

theorem floor_of_3_point_9 : ⌊(3.9 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_3_point_9_l4153_415345


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2013_l4153_415306

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_eq_1 : a 1 = 1
  d : ℝ
  d_neq_0 : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 2)^2 = a 1 * a 5

/-- The 2013th term of the arithmetic sequence is 4025 -/
theorem arithmetic_sequence_2013 (seq : ArithmeticSequence) : seq.a 2013 = 4025 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2013_l4153_415306


namespace NUMINAMATH_CALUDE_function_minimum_l4153_415326

theorem function_minimum (f : ℝ → ℝ) (a : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x - a) * (deriv f x) ≥ 0) :
  ∀ x, f x ≥ f a :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_l4153_415326
