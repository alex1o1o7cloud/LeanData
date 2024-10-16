import Mathlib

namespace NUMINAMATH_CALUDE_product_sum_theorem_l2372_237203

theorem product_sum_theorem (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120 →
  p + q + r + s + t = 32 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2372_237203


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2372_237292

theorem fourth_grade_students (initial_students leaving_students new_students : ℕ) :
  initial_students = 35 →
  leaving_students = 10 →
  new_students = 10 →
  initial_students - leaving_students + new_students = 35 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2372_237292


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l2372_237205

theorem quadratic_function_max_value (m n : ℝ) : 
  m^2 - 4*n ≥ 0 →
  (m - 1)^2 + (n - 1)^2 + (m - n)^2 ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l2372_237205


namespace NUMINAMATH_CALUDE_aurelia_percentage_l2372_237213

/-- Given Lauryn's earnings and the total earnings of Lauryn and Aurelia,
    calculate the percentage of Lauryn's earnings that Aurelia made. -/
theorem aurelia_percentage (lauryn_earnings total_earnings : ℝ) : 
  lauryn_earnings = 2000 →
  total_earnings = 3400 →
  (100 * (total_earnings - lauryn_earnings)) / lauryn_earnings = 70 := by
sorry

end NUMINAMATH_CALUDE_aurelia_percentage_l2372_237213


namespace NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l2372_237225

/-- Represents a cube made of unit cubes --/
structure Cube where
  size : ℕ

/-- Calculates the number of visible unit cubes from a corner of the cube --/
def visibleUnitCubes (c : Cube) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem: The maximum number of visible unit cubes from a single point in a 9x9x9 cube is 220 --/
theorem max_visible_cubes_9x9x9 :
  ∀ (c : Cube), c.size = 9 → visibleUnitCubes c = 220 := by
  sorry

#eval visibleUnitCubes { size := 9 }

end NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l2372_237225


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2372_237295

-- Define the polynomials
def f (x : ℝ) : ℝ := 3*x^5 + 7*x^4 - 15*x^3 - 35*x^2 + 22*x + 24
def g (x : ℝ) : ℝ := x^3 + 5*x^2 - 4*x + 2
def r (x : ℝ) : ℝ := -258*x^2 + 186*x - 50

-- State the theorem
theorem polynomial_division_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x ∧ (∀ x, r x = -258*x^2 + 186*x - 50) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2372_237295


namespace NUMINAMATH_CALUDE_inequality_proof_l2372_237207

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  1 / (8 * a^2 - 18 * a + 11) + 1 / (8 * b^2 - 18 * b + 11) + 1 / (8 * c^2 - 18 * c + 11) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2372_237207


namespace NUMINAMATH_CALUDE_number_ratio_l2372_237229

theorem number_ratio : 
  ∀ (s l : ℕ), 
  s > 0 → 
  l > s → 
  l - s = 16 → 
  s = 28 → 
  (l : ℚ) / s = 11 / 7 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l2372_237229


namespace NUMINAMATH_CALUDE_function_properties_l2372_237293

def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - m

theorem function_properties (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f m y ≤ f m x ∧ f m x = 0) →
  (m = 0 ∨ m = 4) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, ∀ y ∈ Set.Icc (-1 : ℝ) 0, x ≤ y → f m x ≥ f m y) →
  (m ≤ -2) ∧
  (Set.range (f m) = Set.Icc 2 3) ↔ m = 6 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2372_237293


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2372_237285

/-- Represents the number of math books -/
def num_math_books : Nat := 4

/-- Represents the number of history books -/
def num_history_books : Nat := 4

/-- Represents the condition that a math book must be at each end -/
def math_books_at_ends : Nat := 2

/-- Represents the remaining math books to be placed -/
def remaining_math_books : Nat := num_math_books - math_books_at_ends

/-- Represents the arrangement of books satisfying all conditions -/
def valid_arrangement (n m : Nat) : Nat :=
  (n * (n - 1)) * (m.factorial) * (remaining_math_books.factorial)

/-- Theorem stating the number of valid arrangements -/
theorem book_arrangement_count :
  valid_arrangement num_math_books num_history_books = 576 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2372_237285


namespace NUMINAMATH_CALUDE_first_year_exceeding_target_l2372_237297

def initial_investment : ℝ := 1.3
def annual_increase_rate : ℝ := 0.12
def target_investment : ℝ := 2.0
def start_year : ℕ := 2015

def investment (year : ℕ) : ℝ :=
  initial_investment * (1 + annual_increase_rate) ^ (year - start_year)

theorem first_year_exceeding_target :
  (∀ y < 2019, investment y ≤ target_investment) ∧
  investment 2019 > target_investment :=
sorry

end NUMINAMATH_CALUDE_first_year_exceeding_target_l2372_237297


namespace NUMINAMATH_CALUDE_trig_equation_solution_l2372_237244

theorem trig_equation_solution (t : ℝ) :
  4 * (Real.sin t * Real.cos t ^ 5 + Real.cos t * Real.sin t ^ 5) + Real.sin (2 * t) ^ 3 = 1 ↔
  ∃ k : ℤ, t = (-1) ^ k * (Real.pi / 12) + k * (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l2372_237244


namespace NUMINAMATH_CALUDE_invisible_dots_count_l2372_237218

/-- The number of dots on a standard six-sided die -/
def standardDieDots : ℕ := 21

/-- The total number of dots on four standard six-sided dice -/
def totalDots : ℕ := 4 * standardDieDots

/-- The list of visible face values on the stacked dice -/
def visibleFaces : List ℕ := [1, 1, 2, 3, 4, 4, 5, 6]

/-- The sum of the visible face values -/
def visibleDotsSum : ℕ := visibleFaces.sum

/-- Theorem: The number of dots not visible on four stacked standard six-sided dice -/
theorem invisible_dots_count : totalDots - visibleDotsSum = 58 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l2372_237218


namespace NUMINAMATH_CALUDE_equation_solution_l2372_237276

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = 3/5 ∧ 
  ∀ (x : ℝ), (x - 3)^2 + 4*x*(x - 3) = 0 ↔ (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2372_237276


namespace NUMINAMATH_CALUDE_cleo_final_marbles_l2372_237217

def initial_marbles : ℕ := 240

def day2_fraction : ℚ := 2/3
def day2_people : ℕ := 3

def day3_fraction : ℚ := 3/5
def day3_people : ℕ := 2

def day4_cleo_fraction : ℚ := 7/8
def day4_estela_fraction : ℚ := 1/4

theorem cleo_final_marbles :
  let day2_marbles := (initial_marbles : ℚ) * day2_fraction
  let day2_per_person := ⌊day2_marbles / day2_people⌋
  let day3_remaining := initial_marbles - (day2_per_person * day2_people)
  let day3_marbles := (day3_remaining : ℚ) * day3_fraction
  let day3_cleo := ⌊day3_marbles / day3_people⌋
  let day4_cleo := ⌊(day3_cleo : ℚ) * day4_cleo_fraction⌋
  let day4_estela := ⌊(day4_cleo : ℚ) * day4_estela_fraction⌋
  day4_cleo - day4_estela = 16 := by sorry

end NUMINAMATH_CALUDE_cleo_final_marbles_l2372_237217


namespace NUMINAMATH_CALUDE_container_capacity_l2372_237268

theorem container_capacity : 
  ∀ (C : ℝ), 
    C > 0 → 
    (0.40 * C + 28 = 0.75 * C) → 
    C = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2372_237268


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3780_l2372_237263

/-- The largest perfect square factor of a natural number -/
def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The largest perfect square factor of 3780 is 36 -/
theorem largest_perfect_square_factor_of_3780 :
  largest_perfect_square_factor 3780 = 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3780_l2372_237263


namespace NUMINAMATH_CALUDE_yoongi_age_l2372_237269

theorem yoongi_age (yoongi_age hoseok_age : ℕ) 
  (sum_of_ages : yoongi_age + hoseok_age = 16)
  (age_difference : yoongi_age = hoseok_age + 2) : 
  yoongi_age = 9 := by
sorry

end NUMINAMATH_CALUDE_yoongi_age_l2372_237269


namespace NUMINAMATH_CALUDE_acoustics_class_male_count_l2372_237236

/-- The number of male students in the acoustics class -/
def male_students : ℕ := 120

/-- The number of female students in the acoustics class -/
def female_students : ℕ := 100

/-- The percentage of male students who are engineering students -/
def male_eng_percent : ℚ := 25 / 100

/-- The percentage of female students who are engineering students -/
def female_eng_percent : ℚ := 20 / 100

/-- The percentage of male engineering students who passed the final exam -/
def male_pass_percent : ℚ := 20 / 100

/-- The percentage of female engineering students who passed the final exam -/
def female_pass_percent : ℚ := 25 / 100

/-- The percentage of all engineering students who passed the exam -/
def total_pass_percent : ℚ := 22 / 100

theorem acoustics_class_male_count :
  male_students = 120 ∧
  (male_eng_percent * male_students * male_pass_percent +
   female_eng_percent * female_students * female_pass_percent) =
  total_pass_percent * (male_eng_percent * male_students + female_eng_percent * female_students) :=
by sorry

end NUMINAMATH_CALUDE_acoustics_class_male_count_l2372_237236


namespace NUMINAMATH_CALUDE_cos_two_alpha_l2372_237215

theorem cos_two_alpha (α : Real) (h : Real.sin α + Real.cos α = 2/3) :
  Real.cos (2 * α) = 2 * Real.sqrt 14 / 9 ∨ Real.cos (2 * α) = -2 * Real.sqrt 14 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_two_alpha_l2372_237215


namespace NUMINAMATH_CALUDE_arccos_cos_nine_l2372_237282

/-- The arccosine of the cosine of 9 is equal to 9 modulo 2π. -/
theorem arccos_cos_nine : Real.arccos (Real.cos 9) = 9 % (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_nine_l2372_237282


namespace NUMINAMATH_CALUDE_min_height_of_box_l2372_237255

/-- Represents a rectangular box with square bases -/
structure Box where
  base_side : ℝ
  height : ℝ
  height_twice_base : height = 2 * base_side

/-- Calculates the surface area of a box -/
def surface_area (b : Box) : ℝ :=
  2 * b.base_side^2 + 4 * b.base_side * b.height

/-- Theorem stating the minimum height of the box given the constraints -/
theorem min_height_of_box (b : Box) 
  (area_constraint : surface_area b ≥ 150) :
  b.height ≥ 2 * Real.sqrt 15 := by
  sorry

#check min_height_of_box

end NUMINAMATH_CALUDE_min_height_of_box_l2372_237255


namespace NUMINAMATH_CALUDE_john_newspaper_profit_l2372_237200

/-- Calculates the profit made by John selling newspapers --/
theorem john_newspaper_profit :
  let total_newspapers : ℕ := 500
  let selling_price : ℚ := 2
  let sold_percentage : ℚ := 80 / 100
  let discount_percentage : ℚ := 75 / 100
  let profit : ℚ := (total_newspapers : ℚ) * sold_percentage * selling_price - 
                    total_newspapers * (selling_price * (1 - discount_percentage))
  profit = 550
  := by sorry

end NUMINAMATH_CALUDE_john_newspaper_profit_l2372_237200


namespace NUMINAMATH_CALUDE_star_arrangements_l2372_237234

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    where rotations are considered equivalent but reflections are not. -/
theorem star_arrangements : (factorial 12) / 6 = 79833600 := by
  sorry

end NUMINAMATH_CALUDE_star_arrangements_l2372_237234


namespace NUMINAMATH_CALUDE_greatest_number_in_set_l2372_237219

/-- A set of consecutive multiples of 2 -/
def ConsecutiveMultiplesOf2 (s : Set ℕ) : Prop :=
  ∃ start : ℕ, ∀ n ∈ s, ∃ k : ℕ, n = start + 2 * k

theorem greatest_number_in_set (s : Set ℕ) 
  (h1 : ConsecutiveMultiplesOf2 s)
  (h2 : Fintype s)
  (h3 : Fintype.card s = 50)
  (h4 : 56 ∈ s)
  (h5 : ∀ n ∈ s, n ≥ 56) :
  ∃ m ∈ s, m = 154 ∧ ∀ n ∈ s, n ≤ m :=
sorry

end NUMINAMATH_CALUDE_greatest_number_in_set_l2372_237219


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2372_237289

/-- Proves that given a 200-gallon tank filled with two types of fuel,
    where one contains 12% ethanol and the other 16% ethanol,
    if the full tank contains 30 gallons of ethanol,
    then the volume of the first fuel added is 50 gallons. -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) 
    (total_ethanol : ℝ) (fuel_a : ℝ) :
  tank_capacity = 200 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  fuel_a * ethanol_a + (tank_capacity - fuel_a) * ethanol_b = total_ethanol →
  fuel_a = 50 := by
sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2372_237289


namespace NUMINAMATH_CALUDE_max_sphere_surface_area_from_cube_l2372_237290

/-- Given a cube with side length 2, the maximum surface area of a sphere carved from this cube is 4π. -/
theorem max_sphere_surface_area_from_cube (cube_side_length : ℝ) (sphere_surface_area : ℝ → ℝ) :
  cube_side_length = 2 →
  (∀ r : ℝ, r ≤ 1 → sphere_surface_area r ≤ sphere_surface_area 1) →
  sphere_surface_area 1 = 4 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_max_sphere_surface_area_from_cube_l2372_237290


namespace NUMINAMATH_CALUDE_oil_leak_total_l2372_237265

theorem oil_leak_total (before_repairs : ℕ) (during_repairs : ℕ) 
  (h1 : before_repairs = 6522) 
  (h2 : during_repairs = 5165) : 
  before_repairs + during_repairs = 11687 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_total_l2372_237265


namespace NUMINAMATH_CALUDE_cos_90_degrees_zero_l2372_237291

theorem cos_90_degrees_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_zero_l2372_237291


namespace NUMINAMATH_CALUDE_square_area_and_diagonal_l2372_237245

/-- Given a square with perimeter 40 feet, prove its area and diagonal length -/
theorem square_area_and_diagonal (perimeter : ℝ) (h : perimeter = 40) :
  let side := perimeter / 4
  (side ^ 2 = 100) ∧ (side * Real.sqrt 2 = 10 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_area_and_diagonal_l2372_237245


namespace NUMINAMATH_CALUDE_vector_simplification_l2372_237231

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define the vectors
variable (O P Q M : V)

-- State the theorem
theorem vector_simplification :
  (P - O) + (Q - P) - (Q - M) = M - O :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l2372_237231


namespace NUMINAMATH_CALUDE_additional_license_plates_l2372_237298

theorem additional_license_plates 
  (initial_first : Nat) 
  (initial_second : Nat) 
  (initial_third : Nat) 
  (added_letters : Nat) 
  (h1 : initial_first = 5) 
  (h2 : initial_second = 3) 
  (h3 : initial_third = 4) 
  (h4 : added_letters = 1) : 
  (initial_first + added_letters) * (initial_second + added_letters) * (initial_third + added_letters) - 
  (initial_first * initial_second * initial_third) = 60 := by
sorry

end NUMINAMATH_CALUDE_additional_license_plates_l2372_237298


namespace NUMINAMATH_CALUDE_corner_cut_cube_edges_l2372_237286

/-- Represents a solid formed by removing smaller cubes from the corners of a larger cube -/
structure CornerCutCube where
  original_side_length : ℝ
  removed_side_length : ℝ

/-- Calculates the number of edges in the resulting solid -/
def edge_count (c : CornerCutCube) : ℕ :=
  sorry

/-- Theorem stating that a cube of side length 5 with corners of side length 2 removed has 48 edges -/
theorem corner_cut_cube_edges :
  let c : CornerCutCube := { original_side_length := 5, removed_side_length := 2 }
  edge_count c = 48 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_cube_edges_l2372_237286


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2372_237253

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 200) :
  4 * a 5 - 2 * a 3 = 80 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2372_237253


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2372_237254

theorem min_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x^2 + y^2 = x*y*(x^2*y^2 + 2)) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a*b*(a^2*b^2 + 2) → 
  1/x + 1/y ≤ 1/a + 1/b :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2372_237254


namespace NUMINAMATH_CALUDE_no_solution_equation_l2372_237294

theorem no_solution_equation : ¬∃ (x : ℝ), x - 9 / (x - 5) = 5 - 9 / (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2372_237294


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2372_237270

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the line l passing through (-4,0) and tangent to circle M
def line_l (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x + 4) ∧ k^2 / (1 + k^2) = 1/9

-- Theorem statement
theorem trajectory_and_intersection :
  -- The trajectory of the center of circle P forms curve C
  (∀ x y : ℝ, (∃ r : ℝ, 0 < r ∧ r < 3 ∧
    (∀ x' y' : ℝ, (x' - x)^2 + (y' - y)^2 = r^2 →
      (circle_M x' y' → (x' - x)^2 + (y' - y)^2 = (1 + r)^2) ∧
      (circle_N x' y' → (x' - x)^2 + (y' - y)^2 = (3 - r)^2))
  ) → curve_C x y) ∧
  -- The line l intersects curve C at two points with distance 18/7
  (∀ x₁ y₁ x₂ y₂ : ℝ, curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) →
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (18/7)^2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2372_237270


namespace NUMINAMATH_CALUDE_power_function_property_l2372_237249

/-- A power function with a specific property -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x > 0, f x = x ^ α

theorem power_function_property (f : ℝ → ℝ) (h1 : PowerFunction f) (h2 : f 4 / f 2 = 3) :
  f (1/2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l2372_237249


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l2372_237272

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem parallel_lines_theorem 
  (m n : Line) (α β : Plane) 
  (hm_neq_n : m ≠ n)
  (hα_neq_β : α ≠ β)
  (hm_parallel_β : parallel_line_plane m β)
  (hm_in_α : contained_in m α)
  (hα_intersect_β : intersect α β n) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l2372_237272


namespace NUMINAMATH_CALUDE_office_age_problem_l2372_237238

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) (group1_persons : ℕ) (avg_age_group1 : ℝ) (group2_persons : ℕ) (person15_age : ℝ) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_persons = 5 →
  avg_age_group1 = 14 →
  group2_persons = 9 →
  person15_age = 56 →
  (total_persons * avg_age_all - group1_persons * avg_age_group1 - person15_age) / group2_persons = 16 := by
  sorry

end NUMINAMATH_CALUDE_office_age_problem_l2372_237238


namespace NUMINAMATH_CALUDE_garden_border_perimeter_l2372_237243

/-- The total perimeter of Mrs. Hilt's garden border -/
theorem garden_border_perimeter :
  let num_rocks_a : ℝ := 125.0
  let circumference_a : ℝ := 0.5
  let num_rocks_b : ℝ := 64.0
  let circumference_b : ℝ := 0.7
  let total_perimeter : ℝ := num_rocks_a * circumference_a + num_rocks_b * circumference_b
  total_perimeter = 107.3 := by
sorry

end NUMINAMATH_CALUDE_garden_border_perimeter_l2372_237243


namespace NUMINAMATH_CALUDE_min_value_expression_l2372_237266

theorem min_value_expression (r s t : ℝ) 
  (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 12 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2372_237266


namespace NUMINAMATH_CALUDE_square_inequality_l2372_237240

theorem square_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l2372_237240


namespace NUMINAMATH_CALUDE_three_X_seven_equals_eight_l2372_237227

/-- The operation X defined for two real numbers -/
def X (a b : ℝ) : ℝ := b + 15 * a - a^2 - 5 * b

/-- Theorem stating that 3X7 equals 8 -/
theorem three_X_seven_equals_eight : X 3 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_X_seven_equals_eight_l2372_237227


namespace NUMINAMATH_CALUDE_sphere_dihedral_angle_segment_fraction_l2372_237212

/-- The fraction of the segment AB that lies outside two equal touching spheres inscribed in a dihedral angle -/
theorem sphere_dihedral_angle_segment_fraction (α : Real) : 
  α > 0 → α < π → 
  let f := (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)
  0 ≤ f ∧ f ≤ 1 := by sorry

end NUMINAMATH_CALUDE_sphere_dihedral_angle_segment_fraction_l2372_237212


namespace NUMINAMATH_CALUDE_optimal_planting_solution_l2372_237267

/-- Represents the planting problem with two types of flowers -/
structure PlantingProblem where
  costA3B4 : ℕ  -- Cost of 3 pots of A and 4 pots of B
  costA4B3 : ℕ  -- Cost of 4 pots of A and 3 pots of B
  totalPots : ℕ  -- Total number of pots to be planted
  survivalRateA : ℚ  -- Survival rate of type A flowers
  survivalRateB : ℚ  -- Survival rate of type B flowers
  maxReplacement : ℕ  -- Maximum number of pots to be replaced next year

/-- Represents the solution to the planting problem -/
structure PlantingSolution where
  costA : ℕ  -- Cost of each pot of type A flowers
  costB : ℕ  -- Cost of each pot of type B flowers
  potsA : ℕ  -- Number of pots of type A flowers to plant
  potsB : ℕ  -- Number of pots of type B flowers to plant
  totalCost : ℕ  -- Total cost of planting

/-- Theorem stating the optimal solution for the planting problem -/
theorem optimal_planting_solution (problem : PlantingProblem) 
  (h1 : problem.costA3B4 = 360)
  (h2 : problem.costA4B3 = 340)
  (h3 : problem.totalPots = 600)
  (h4 : problem.survivalRateA = 7/10)
  (h5 : problem.survivalRateB = 9/10)
  (h6 : problem.maxReplacement = 100) :
  ∃ (solution : PlantingSolution),
    solution.costA = 40 ∧
    solution.costB = 60 ∧
    solution.potsA = 200 ∧
    solution.potsB = 400 ∧
    solution.totalCost = 32000 ∧
    solution.potsA + solution.potsB = problem.totalPots ∧
    (1 - problem.survivalRateA) * solution.potsA + (1 - problem.survivalRateB) * solution.potsB ≤ problem.maxReplacement ∧
    ∀ (altSolution : PlantingSolution),
      altSolution.potsA + altSolution.potsB = problem.totalPots →
      (1 - problem.survivalRateA) * altSolution.potsA + (1 - problem.survivalRateB) * altSolution.potsB ≤ problem.maxReplacement →
      solution.totalCost ≤ altSolution.totalCost :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_planting_solution_l2372_237267


namespace NUMINAMATH_CALUDE_max_distinct_factors_max_additional_factors_l2372_237222

theorem max_distinct_factors (x : Finset ℕ) :
  (∀ y ∈ x, y > 0) →
  (Nat.lcm 1024 2016 = Finset.lcm x (Nat.lcm 1024 2016)) →
  x.card ≤ 66 :=
by sorry

theorem max_additional_factors :
  ∃ (x : Finset ℕ), x.card = 64 ∧
  (∀ y ∈ x, y > 0) ∧
  (Nat.lcm 1024 2016 = Finset.lcm x (Nat.lcm 1024 2016)) :=
by sorry

end NUMINAMATH_CALUDE_max_distinct_factors_max_additional_factors_l2372_237222


namespace NUMINAMATH_CALUDE_smallest_number_with_18_factors_l2372_237221

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_18_factors : 
  ∃ m : ℕ, m > 1 ∧ 
           num_factors m = 18 ∧ 
           num_factors m - 2 ≥ 16 ∧
           ∀ k : ℕ, k > 1 → num_factors k = 18 → num_factors k - 2 ≥ 16 → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_18_factors_l2372_237221


namespace NUMINAMATH_CALUDE_prob_not_beside_partner_is_four_fifths_l2372_237252

/-- The number of people to be seated -/
def total_people : ℕ := 5

/-- The number of couples -/
def num_couples : ℕ := 2

/-- The number of single people -/
def num_singles : ℕ := total_people - 2 * num_couples

/-- The total number of seating arrangements -/
def total_arrangements : ℕ := Nat.factorial total_people

/-- The number of arrangements where all couples are seated together -/
def couples_together_arrangements : ℕ := 
  (Nat.factorial (num_couples + num_singles)) * (2 ^ num_couples)

/-- The probability that at least one person is not beside their partner -/
def prob_not_beside_partner : ℚ := 
  1 - (couples_together_arrangements : ℚ) / (total_arrangements : ℚ)

theorem prob_not_beside_partner_is_four_fifths : 
  prob_not_beside_partner = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_not_beside_partner_is_four_fifths_l2372_237252


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2372_237259

theorem arithmetic_sequence_length :
  ∀ (a₁ : ℤ) (aₙ : ℤ) (d : ℤ),
    a₁ = -48 →
    aₙ = 72 →
    d = 6 →
    ∃ (n : ℕ), n = 21 ∧ aₙ = a₁ + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2372_237259


namespace NUMINAMATH_CALUDE_floor_equation_difference_l2372_237211

theorem floor_equation_difference : ∃ (x y : ℤ), 
  (∀ z : ℤ, ⌊(z : ℚ) / 3⌋ = 102 → z ≤ x) ∧ 
  (⌊(x : ℚ) / 3⌋ = 102) ∧
  (∀ z : ℤ, ⌊(z : ℚ) / 3⌋ = -102 → y ≤ z) ∧ 
  (⌊(y : ℚ) / 3⌋ = -102) ∧
  (x - y = 614) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_difference_l2372_237211


namespace NUMINAMATH_CALUDE_no_obtuse_right_triangle_l2372_237275

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isRight (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: An obtuse right triangle cannot exist
theorem no_obtuse_right_triangle :
  ∀ t : Triangle,
  (t.angle1 + t.angle2 + t.angle3 = 180) →
  ¬(t.isRight ∧ t.isObtuse) :=
by
  sorry


end NUMINAMATH_CALUDE_no_obtuse_right_triangle_l2372_237275


namespace NUMINAMATH_CALUDE_bisection_interval_valid_l2372_237284

/-- The function f(x) = x^3 + 5 -/
def f (x : ℝ) : ℝ := x^3 + 5

/-- Theorem stating that [-2, 1] is a valid initial interval for the bisection method -/
theorem bisection_interval_valid :
  f (-2) * f 1 < 0 := by sorry

end NUMINAMATH_CALUDE_bisection_interval_valid_l2372_237284


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2372_237299

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x + 1)

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2372_237299


namespace NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l2372_237224

theorem square_triangle_equal_perimeter (x : ℝ) : 
  4 * (x + 2) = 3 * (2 * x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l2372_237224


namespace NUMINAMATH_CALUDE_relay_race_ratio_l2372_237288

/-- Relay race problem -/
theorem relay_race_ratio (mary susan jen tiffany : ℕ) : 
  susan = jen + 10 →
  jen = 30 →
  tiffany = mary - 7 →
  mary + susan + jen + tiffany = 223 →
  mary / susan = 2 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_ratio_l2372_237288


namespace NUMINAMATH_CALUDE_divisibility_problem_l2372_237201

theorem divisibility_problem (n m k : ℕ) (h1 : n = 172835) (h2 : m = 136) (h3 : k = 21) :
  (n + k) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2372_237201


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l2372_237256

/-- The amount of money Jerry needs to complete his action figure collection -/
def jerry_needs_money (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $72 to complete his collection -/
theorem jerry_collection_cost : jerry_needs_money 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l2372_237256


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2372_237271

theorem sum_of_fractions : (1 : ℚ) / 3 + 5 / 9 = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2372_237271


namespace NUMINAMATH_CALUDE_common_divisors_2n_plus_3_and_3n_plus_2_l2372_237204

theorem common_divisors_2n_plus_3_and_3n_plus_2 (n : ℕ) :
  {d : ℕ | d ∣ (2*n + 3) ∧ d ∣ (3*n + 2)} = {d : ℕ | d = 1 ∨ (d = 5 ∧ n % 5 = 1)} := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_2n_plus_3_and_3n_plus_2_l2372_237204


namespace NUMINAMATH_CALUDE_centroid_circumcenter_distance_squared_l2372_237206

/-- Given a triangle with medians m_a, m_b, m_c and circumradius R,
    the squared distance between the centroid and circumcenter (SM^2)
    is equal to R^2 - (4/27)(m_a^2 + m_b^2 + m_c^2) -/
theorem centroid_circumcenter_distance_squared
  (m_a m_b m_c R : ℝ) :
  ∃ (SM : ℝ),
    SM^2 = R^2 - (4/27) * (m_a^2 + m_b^2 + m_c^2) :=
by sorry

end NUMINAMATH_CALUDE_centroid_circumcenter_distance_squared_l2372_237206


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l2372_237223

theorem spelling_bee_contestants (total : ℕ) : 
  (total / 2 : ℚ) / 4 = 30 → total = 240 := by sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l2372_237223


namespace NUMINAMATH_CALUDE_loan_to_c_l2372_237260

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem loan_to_c (loan_to_b : ℝ) (time_b : ℝ) (time_c : ℝ) (total_interest : ℝ) (rate : ℝ) :
  loan_to_b = 5000 →
  time_b = 2 →
  time_c = 4 →
  total_interest = 2640 →
  rate = 0.12 →
  ∃ (loan_to_c : ℝ),
    loan_to_c = 3000 ∧
    total_interest = simple_interest loan_to_b rate time_b + simple_interest loan_to_c rate time_c :=
by sorry

end NUMINAMATH_CALUDE_loan_to_c_l2372_237260


namespace NUMINAMATH_CALUDE_services_total_cost_l2372_237287

def hair_cost : ℝ := 50
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.20

def total_cost : ℝ := hair_cost + manicure_cost + (hair_cost * tip_percentage) + (manicure_cost * tip_percentage)

theorem services_total_cost : total_cost = 96 := by
  sorry

end NUMINAMATH_CALUDE_services_total_cost_l2372_237287


namespace NUMINAMATH_CALUDE_bicyclist_speed_increase_l2372_237251

theorem bicyclist_speed_increase (x : ℝ) : 
  (1 + x) * 1.1 = 1.43 → x = 0.3 := by sorry

end NUMINAMATH_CALUDE_bicyclist_speed_increase_l2372_237251


namespace NUMINAMATH_CALUDE_circle_center_sum_l2372_237280

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, prove that its center (h, k) satisfies h + k = -1 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 4*h + 6*k - 9)) → 
  h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2372_237280


namespace NUMINAMATH_CALUDE_min_balls_same_color_l2372_237233

/-- Given a bag with 6 balls each of 4 different colors, the minimum number of balls
    that must be drawn to ensure two balls of the same color are drawn is 5. -/
theorem min_balls_same_color (num_colors : ℕ) (balls_per_color : ℕ) :
  num_colors = 4 →
  balls_per_color = 6 →
  5 = Nat.succ num_colors :=
by sorry

end NUMINAMATH_CALUDE_min_balls_same_color_l2372_237233


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2372_237235

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 10*a^2 + 16*a - 2 = 0 →
  b^3 - 10*b^2 + 16*b - 2 = 0 →
  c^3 - 10*c^2 + 16*c - 2 = 0 →
  (a / (b*c + 2)) + (b / (a*c + 2)) + (c / (a*b + 2)) = 4 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2372_237235


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2372_237208

/-- Given a complex number z satisfying (1+2i)z=4+3i, prove that z is located in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant (z : ℂ) (h : (1 + 2*Complex.I)*z = 4 + 3*Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2372_237208


namespace NUMINAMATH_CALUDE_max_knights_between_knights_max_knights_between_knights_proof_l2372_237216

theorem max_knights_between_knights (total_knights : ℕ) (total_samurais : ℕ) 
  (knights_with_samurai_right : ℕ) (max_knights_between_knights : ℕ) : Prop :=
  total_knights = 40 →
  total_samurais = 10 →
  knights_with_samurai_right = 7 →
  max_knights_between_knights = 32 →
  max_knights_between_knights = total_knights - (knights_with_samurai_right + 1)

-- The proof would go here, but we're skipping it as per instructions
theorem max_knights_between_knights_proof : 
  max_knights_between_knights 40 10 7 32 := by sorry

end NUMINAMATH_CALUDE_max_knights_between_knights_max_knights_between_knights_proof_l2372_237216


namespace NUMINAMATH_CALUDE_equation_roots_range_l2372_237283

theorem equation_roots_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l2372_237283


namespace NUMINAMATH_CALUDE_missing_number_proof_l2372_237232

theorem missing_number_proof (x : ℤ) : (4 + 3) + (8 - x - 1) = 11 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2372_237232


namespace NUMINAMATH_CALUDE_ab_value_l2372_237246

theorem ab_value (a b : ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) : 
  a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l2372_237246


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l2372_237237

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (1000 ≤ n ∧ n < 10000) ∧ 
    (75 * n) % 450 = 225 ∧
    (∀ m, (1000 ≤ m ∧ m < 10000) → (75 * m) % 450 = 225 → n ≤ m) ∧
    n = 1005 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l2372_237237


namespace NUMINAMATH_CALUDE_twenty_five_percent_more_than_eighty_twenty_five_percent_more_than_eighty_proof_l2372_237228

theorem twenty_five_percent_more_than_eighty : ℝ → Prop :=
  fun x => (3/4 * x = 100) → (x = 400/3)

-- The proof is omitted
theorem twenty_five_percent_more_than_eighty_proof : 
  ∃ x : ℝ, twenty_five_percent_more_than_eighty x :=
sorry

end NUMINAMATH_CALUDE_twenty_five_percent_more_than_eighty_twenty_five_percent_more_than_eighty_proof_l2372_237228


namespace NUMINAMATH_CALUDE_cone_angle_calculation_l2372_237274

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cone -/
structure Cone where
  vertex : ℝ × ℝ × ℝ

/-- Configuration of spheres and cone -/
structure SphereConeConfiguration where
  sphere1 : Sphere
  sphere2 : Sphere
  sphere3 : Sphere
  cone : Cone
  spheresTouch : Bool
  coneTouchesSpheres : Bool
  vertexBetweenContacts : Bool

/-- The angle at the vertex of the cone -/
def coneAngle (config : SphereConeConfiguration) : ℝ :=
  sorry

theorem cone_angle_calculation (config : SphereConeConfiguration) 
  (h1 : config.sphere1.radius = 2)
  (h2 : config.sphere2.radius = 2)
  (h3 : config.sphere3.radius = 1)
  (h4 : config.spheresTouch = true)
  (h5 : config.coneTouchesSpheres = true)
  (h6 : config.vertexBetweenContacts = true) :
  coneAngle config = 2 * Real.arctan (1 / 8) :=
sorry

end NUMINAMATH_CALUDE_cone_angle_calculation_l2372_237274


namespace NUMINAMATH_CALUDE_ababab_no_large_prime_factors_l2372_237273

theorem ababab_no_large_prime_factors (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  ∀ p : ℕ, p.Prime → p ∣ (101010 * a + 10101 * b) → p ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_ababab_no_large_prime_factors_l2372_237273


namespace NUMINAMATH_CALUDE_thousandth_digit_is_one_l2372_237209

/-- The number of digits in n -/
def num_digits : ℕ := 1998

/-- The number n as a natural number -/
def n : ℕ := (10^num_digits - 1) / 9

/-- The 1000th digit after the decimal point of √n -/
def thousandth_digit_after_decimal (n : ℕ) : ℕ :=
  -- Definition placeholder, actual implementation would be complex
  sorry

/-- Theorem stating that the 1000th digit after the decimal point of √n is 1 -/
theorem thousandth_digit_is_one :
  thousandth_digit_after_decimal n = 1 := by sorry

end NUMINAMATH_CALUDE_thousandth_digit_is_one_l2372_237209


namespace NUMINAMATH_CALUDE_compare_polynomial_expressions_l2372_237214

theorem compare_polynomial_expressions {a b c : ℝ} (h1 : a > b) (h2 : b > c) :
  a^2*b + b^2*c + c^2*a > a*b^2 + b*c^2 + c*a^2 := by
  sorry

end NUMINAMATH_CALUDE_compare_polynomial_expressions_l2372_237214


namespace NUMINAMATH_CALUDE_f_properties_l2372_237220

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4

-- State the theorem
theorem f_properties :
  (∀ x y : ℝ, f (x * y) + f (y - x) ≥ f (y + x)) ∧
  (∀ x : ℝ, f x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l2372_237220


namespace NUMINAMATH_CALUDE_min_area_at_one_eighth_l2372_237264

-- Define the lines l₁ and l₂
def l₁ (k x y : ℝ) : Prop := k * x - 2 * y - 2 * k + 8 = 0
def l₂ (k x y : ℝ) : Prop := 2 * x + k^2 * y - 4 * k^2 - 4 = 0

-- Define the area of the quadrilateral as a function of k
noncomputable def quadrilateral_area (k : ℝ) : ℝ := 
  let x₁ := (2 * k - 8) / k
  let y₁ := 4 - k
  let x₂ := 2 * k^2 + 2
  let y₂ := 4 + 4 / k^2
  (x₁ * y₁) / 2 + (x₂ * y₂) / 2

-- State the theorem
theorem min_area_at_one_eighth (k : ℝ) (h : 0 < k ∧ k < 4) :
  ∃ (min_k : ℝ), min_k = 1/8 ∧ 
  ∀ k', 0 < k' ∧ k' < 4 → quadrilateral_area min_k ≤ quadrilateral_area k' :=
sorry

end NUMINAMATH_CALUDE_min_area_at_one_eighth_l2372_237264


namespace NUMINAMATH_CALUDE_problem_solution_l2372_237262

theorem problem_solution :
  let x : ℤ := 5
  let y : ℤ := x + 3
  let z : ℤ := 3 * y + 1
  z = 25 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2372_237262


namespace NUMINAMATH_CALUDE_ratio_to_nine_l2372_237241

theorem ratio_to_nine (x : ℝ) : (x / 9 = 5 / 1) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_nine_l2372_237241


namespace NUMINAMATH_CALUDE_lamplighter_monkey_speed_l2372_237261

/-- A Lamplighter monkey's movement characteristics -/
structure LamplighterMonkey where
  swingingSpeed : ℝ
  runningSpeed : ℝ
  runningTime : ℝ
  swingingTime : ℝ
  totalDistance : ℝ

/-- Theorem: Given the characteristics of a Lamplighter monkey's movement,
    prove that its running speed is 15 feet per second -/
theorem lamplighter_monkey_speed (monkey : LamplighterMonkey)
  (h1 : monkey.swingingSpeed = 10)
  (h2 : monkey.runningTime = 5)
  (h3 : monkey.swingingTime = 10)
  (h4 : monkey.totalDistance = 175) :
  monkey.runningSpeed = 15 := by
  sorry

#check lamplighter_monkey_speed

end NUMINAMATH_CALUDE_lamplighter_monkey_speed_l2372_237261


namespace NUMINAMATH_CALUDE_value_of_c_l2372_237202

theorem value_of_c (a b c : ℝ) : 
  12 = 0.06 * a → 
  6 = 0.12 * b → 
  c = b / a → 
  c = 0.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l2372_237202


namespace NUMINAMATH_CALUDE_cross_section_area_theorem_l2372_237296

/-- A rectangular parallelepiped inscribed in a sphere -/
structure InscribedParallelepiped where
  R : ℝ  -- radius of the sphere
  diagonal_inclination : ℝ  -- angle between diagonals and base plane
  diagonal_inclination_is_45 : diagonal_inclination = Real.pi / 4

/-- The cross-section plane of the parallelepiped -/
structure CrossSectionPlane (p : InscribedParallelepiped) where
  angle_with_diagonal : ℝ  -- angle between the plane and diagonal BD₁
  angle_is_arcsin_sqrt2_4 : angle_with_diagonal = Real.arcsin (Real.sqrt 2 / 4)

/-- The area of the cross-section -/
noncomputable def cross_section_area (p : InscribedParallelepiped) (plane : CrossSectionPlane p) : ℝ :=
  2 * p.R^2 * Real.sqrt 3 / 3

/-- Theorem stating that the area of the cross-section is (2R²√3)/3 -/
theorem cross_section_area_theorem (p : InscribedParallelepiped) (plane : CrossSectionPlane p) :
    cross_section_area p plane = 2 * p.R^2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_theorem_l2372_237296


namespace NUMINAMATH_CALUDE_anna_quiz_goal_impossible_l2372_237257

theorem anna_quiz_goal_impossible (total_quizzes : Nat) (goal_percentage : Rat) 
  (completed_quizzes : Nat) (completed_as : Nat) : 
  total_quizzes = 60 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 40 →
  completed_as = 30 →
  ¬∃ (remaining_as : Nat), 
    (completed_as + remaining_as : Rat) / total_quizzes ≥ goal_percentage ∧ 
    remaining_as ≤ total_quizzes - completed_quizzes :=
by sorry

end NUMINAMATH_CALUDE_anna_quiz_goal_impossible_l2372_237257


namespace NUMINAMATH_CALUDE_rod_pieces_count_l2372_237242

/-- The length of the rod in meters -/
def rod_length_m : ℝ := 34

/-- The length of each piece in centimeters -/
def piece_length_cm : ℝ := 85

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem rod_pieces_count : 
  ⌊(rod_length_m * m_to_cm) / piece_length_cm⌋ = 40 := by sorry

end NUMINAMATH_CALUDE_rod_pieces_count_l2372_237242


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2372_237226

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2372_237226


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2372_237279

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = 2*c ∨ x = 3*d) : 
  c = 1/6 ∧ d = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2372_237279


namespace NUMINAMATH_CALUDE_sum_of_common_terms_l2372_237277

/-- The sequence formed by common terms of {2n-1} and {3n-2} in ascending order -/
def a : ℕ → ℕ := sorry

/-- The sum of the first n terms of sequence a -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the first n terms of sequence a is 3n^2 - 2n -/
theorem sum_of_common_terms (n : ℕ) : S n = 3 * n^2 - 2 * n := by sorry

end NUMINAMATH_CALUDE_sum_of_common_terms_l2372_237277


namespace NUMINAMATH_CALUDE_product_equals_fraction_l2372_237210

/-- The decimal representation of the repeating decimal 0.456̅ -/
def repeating_decimal : ℚ := 152 / 333

/-- The product of the repeating decimal 0.456̅ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̅ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l2372_237210


namespace NUMINAMATH_CALUDE_contaminated_constant_l2372_237248

theorem contaminated_constant (x : ℝ) (h : 2 * (x - 3) - 2 = x + 1) (h_sol : x = 9) : 2 * (9 - 3) - (9 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_contaminated_constant_l2372_237248


namespace NUMINAMATH_CALUDE_ryan_chinese_learning_hours_l2372_237278

/-- Given Ryan's daily Chinese learning hours and number of learning days, 
    calculate the total hours spent learning Chinese -/
def total_chinese_hours (daily_hours : ℕ) (days : ℕ) : ℕ :=
  daily_hours * days

/-- Theorem stating that Ryan spends 24 hours learning Chinese in 6 days -/
theorem ryan_chinese_learning_hours :
  total_chinese_hours 4 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_learning_hours_l2372_237278


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2372_237250

theorem cookie_jar_problem (initial_cookies : ℕ) 
  (cookies_removed : ℕ) (cookies_added : ℕ) : 
  initial_cookies = 7 → 
  cookies_removed = 1 → 
  cookies_added = 5 → 
  initial_cookies - cookies_removed = (initial_cookies + cookies_added) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2372_237250


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l2372_237258

theorem x_range_for_inequality (x t : ℝ) :
  (t ∈ Set.Icc 1 3) →
  (((1/8) * (2*x - x^2) ≤ t^2 - 3*t + 2) ∧ (t^2 - 3*t + 2 ≤ 3 - x^2)) →
  (x ∈ Set.Icc (-1) (1 - Real.sqrt 3)) := by
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l2372_237258


namespace NUMINAMATH_CALUDE_equality_implies_product_equality_l2372_237281

theorem equality_implies_product_equality (a b c : ℝ) : a = b → a * c = b * c := by sorry

end NUMINAMATH_CALUDE_equality_implies_product_equality_l2372_237281


namespace NUMINAMATH_CALUDE_evaluate_expression_l2372_237230

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 - (-9) = 8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2372_237230


namespace NUMINAMATH_CALUDE_remainders_of_1493827_l2372_237247

theorem remainders_of_1493827 : 
  (1493827 % 4 = 3) ∧ (1493827 % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_remainders_of_1493827_l2372_237247


namespace NUMINAMATH_CALUDE_emily_vacation_days_l2372_237239

/-- The number of days food lasts for dogs -/
def vacation_days (num_dogs : ℕ) (food_per_dog : ℕ) (total_food : ℕ) : ℕ :=
  total_food * 1000 / (num_dogs * food_per_dog)

/-- Theorem: Emily's vacation lasts 14 days -/
theorem emily_vacation_days :
  vacation_days 4 250 14 = 14 := by
  sorry

end NUMINAMATH_CALUDE_emily_vacation_days_l2372_237239
