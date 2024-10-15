import Mathlib

namespace NUMINAMATH_CALUDE_number_division_puzzle_l366_36673

theorem number_division_puzzle (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / b = (a + b) / (2 * a) ∧ a / b ≠ 1 → a / b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_number_division_puzzle_l366_36673


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l366_36679

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^4 * 3^5) = 675 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l366_36679


namespace NUMINAMATH_CALUDE_system_solution_unique_l366_36650

theorem system_solution_unique : 
  ∃! (x y z : ℝ), 3*x + 2*y - z = 4 ∧ 2*x - y + 3*z = 9 ∧ x - 2*y + 2*z = 3 ∧ x = 1 ∧ y = 2 ∧ z = 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l366_36650


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l366_36608

/-- The area of the triangle formed by the tangent line to y = x^3 at (3, 27) and the axes is 54 -/
theorem tangent_line_triangle_area : 
  let f : ℝ → ℝ := fun x ↦ x^3
  let point : ℝ × ℝ := (3, 27)
  let tangent_line : ℝ → ℝ := fun x ↦ 27 * x - 54
  let triangle_area := 
    let x_intercept := (tangent_line 0) / (-27)
    let y_intercept := tangent_line 0
    (1/2) * x_intercept * (-y_intercept)
  triangle_area = 54 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l366_36608


namespace NUMINAMATH_CALUDE_inequality_solution_set_l366_36610

theorem inequality_solution_set (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) 3) = {x | (x - 3) * (x + 2) < 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l366_36610


namespace NUMINAMATH_CALUDE_ben_has_fifteen_shirts_l366_36689

/-- The number of new shirts Alex has -/
def alex_shirts : ℕ := 4

/-- The number of additional shirts Joe has compared to Alex -/
def joe_extra_shirts : ℕ := 3

/-- The number of additional shirts Ben has compared to Joe -/
def ben_extra_shirts : ℕ := 8

/-- The number of new shirts Joe has -/
def joe_shirts : ℕ := alex_shirts + joe_extra_shirts

/-- The number of new shirts Ben has -/
def ben_shirts : ℕ := joe_shirts + ben_extra_shirts

theorem ben_has_fifteen_shirts : ben_shirts = 15 := by
  sorry

end NUMINAMATH_CALUDE_ben_has_fifteen_shirts_l366_36689


namespace NUMINAMATH_CALUDE_building_height_proof_l366_36644

/-- Proves the height of the first 10 stories in a 20-story building -/
theorem building_height_proof (total_stories : Nat) (first_section : Nat) (height_difference : Nat) (total_height : Nat) :
  total_stories = 20 →
  first_section = 10 →
  height_difference = 3 →
  total_height = 270 →
  ∃ (first_story_height : Nat),
    first_story_height * first_section + (first_story_height + height_difference) * (total_stories - first_section) = total_height ∧
    first_story_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_building_height_proof_l366_36644


namespace NUMINAMATH_CALUDE_tan_squared_sum_l366_36633

theorem tan_squared_sum (x y : ℝ) 
  (h : 2 * Real.sin x * Real.sin y + 3 * Real.cos y + 6 * Real.cos x * Real.sin y = 7) : 
  Real.tan x ^ 2 + 2 * Real.tan y ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_sum_l366_36633


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l366_36638

theorem circular_arrangement_students (n : ℕ) 
  (h1 : n > 0) 
  (h2 : 10 ≤ n ∧ 40 ≤ n) 
  (h3 : (40 - 10) * 2 = n) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l366_36638


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l366_36684

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l366_36684


namespace NUMINAMATH_CALUDE_overlapping_circles_area_l366_36665

/-- The area of the common part of two equal circles with radius R,
    where the circumference of each circle passes through the center of the other. -/
theorem overlapping_circles_area (R : ℝ) (R_pos : R > 0) :
  ∃ (A : ℝ), A = R^2 * (4 * Real.pi - 3 * Real.sqrt 3) / 6 ∧
  A = 2 * (1/3 * Real.pi * R^2 - R^2 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_overlapping_circles_area_l366_36665


namespace NUMINAMATH_CALUDE_processing_time_theorem_l366_36660

/-- Calculates the total processing time in hours for a set of pictures --/
def total_processing_time (tree_count : ℕ) (flower_count : ℕ) (grass_count : ℕ) 
  (tree_time : ℚ) (flower_time : ℚ) (grass_time : ℚ) : ℚ :=
  ((tree_count : ℚ) * tree_time + (flower_count : ℚ) * flower_time + (grass_count : ℚ) * grass_time) / 60

/-- Theorem stating the total processing time for the given set of pictures --/
theorem processing_time_theorem : 
  total_processing_time 320 400 240 (3/2) (5/2) 1 = 860/30 := by
  sorry

end NUMINAMATH_CALUDE_processing_time_theorem_l366_36660


namespace NUMINAMATH_CALUDE_complex_cube_root_l366_36655

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_root_l366_36655


namespace NUMINAMATH_CALUDE_olivias_initial_money_l366_36674

theorem olivias_initial_money (initial_money : ℕ) : 
  (initial_money + 91 - (91 + 39) = 14) → initial_money = 53 := by
  sorry

end NUMINAMATH_CALUDE_olivias_initial_money_l366_36674


namespace NUMINAMATH_CALUDE_emmy_has_200_l366_36645

/-- The amount of money Emmy has -/
def emmys_money : ℕ := sorry

/-- The amount of money Gerry has -/
def gerrys_money : ℕ := 100

/-- The cost of one apple -/
def apple_cost : ℕ := 2

/-- The total number of apples Emmy and Gerry can buy -/
def total_apples : ℕ := 150

/-- Theorem: Emmy has $200 -/
theorem emmy_has_200 : emmys_money = 200 := by
  have total_cost : ℕ := apple_cost * total_apples
  have sum_of_money : emmys_money + gerrys_money = total_cost := sorry
  sorry

end NUMINAMATH_CALUDE_emmy_has_200_l366_36645


namespace NUMINAMATH_CALUDE_tangent_lines_imply_a_greater_than_three_l366_36690

/-- The function f(x) = -x^3 + ax^2 - 2x --/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 2*x

/-- The derivative of f(x) --/
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x - 2

/-- The condition for a line to be tangent to f(x) at point t --/
def is_tangent (a : ℝ) (t : ℝ) : Prop :=
  -1 + t^3 - a*t^2 + 2*t = (-3*t^2 + 2*a*t - 2)*(-t)

/-- The theorem statement --/
theorem tangent_lines_imply_a_greater_than_three (a : ℝ) :
  (∃ t₁ t₂ t₃ : ℝ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧
    is_tangent a t₁ ∧ is_tangent a t₂ ∧ is_tangent a t₃) →
  a > 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_imply_a_greater_than_three_l366_36690


namespace NUMINAMATH_CALUDE_f_equals_cos_2x_l366_36603

theorem f_equals_cos_2x (x : ℝ) : 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) = 
  Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_f_equals_cos_2x_l366_36603


namespace NUMINAMATH_CALUDE_no_real_solution_complex_roots_l366_36691

theorem no_real_solution_complex_roots :
  ∀ x : ℂ, (2 * x - 36) / 3 = (3 * x^2 + 6 * x + 1) / 4 →
  (∃ b : ℝ, x = -5/9 + b * I ∨ x = -5/9 - b * I) ∧
  (∀ y : ℝ, (2 * y - 36) / 3 ≠ (3 * y^2 + 6 * y + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_complex_roots_l366_36691


namespace NUMINAMATH_CALUDE_relationship_abc_l366_36677

theorem relationship_abc : 
  let a : ℝ := (0.3 : ℝ)^3
  let b : ℝ := 3^(0.3 : ℝ)
  let c : ℝ := (0.2 : ℝ)^3
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l366_36677


namespace NUMINAMATH_CALUDE_not_all_naturals_equal_l366_36612

-- Define the statement we want to disprove
def all_naturals_equal (n : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (∀ i j, i < n → j < n → a i = a j)

-- Theorem stating that the above statement is false
theorem not_all_naturals_equal : ¬ (∀ n : ℕ, all_naturals_equal n) := by
  sorry

-- Note: The proof is omitted (replaced with 'sorry') as per the instructions

end NUMINAMATH_CALUDE_not_all_naturals_equal_l366_36612


namespace NUMINAMATH_CALUDE_firm_partners_count_l366_36600

theorem firm_partners_count (partners associates : ℕ) : 
  partners / associates = 2 / 63 →
  partners / (associates + 35) = 1 / 34 →
  partners = 14 :=
by sorry

end NUMINAMATH_CALUDE_firm_partners_count_l366_36600


namespace NUMINAMATH_CALUDE_product_of_real_parts_l366_36605

theorem product_of_real_parts (x₁ x₂ : ℂ) : 
  x₁^2 - 4*x₁ = -1 - 3*I → 
  x₂^2 - 4*x₂ = -1 - 3*I → 
  x₁ ≠ x₂ → 
  (x₁.re * x₂.re : ℝ) = (8 - Real.sqrt 6 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l366_36605


namespace NUMINAMATH_CALUDE_function_symmetry_l366_36680

theorem function_symmetry (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x + 2 * f (1 / x) = 3 * x) :
  ∀ x ≠ 0, f x = f (-x) ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l366_36680


namespace NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l366_36630

/-- A geometric sequence with a special property -/
def SpecialGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) ∧ a 1 + a 5 = a 1 * a 5

/-- The ratio of the 13th to the 9th term is 9 -/
theorem special_geometric_sequence_ratio
  (a : ℕ → ℝ) (h : SpecialGeometricSequence a) :
  a 13 / a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_geometric_sequence_ratio_l366_36630


namespace NUMINAMATH_CALUDE_square_dissection_theorem_l366_36685

/-- A dissection of a square is a list of polygons that can be rearranged to form the original square. -/
def Dissection (n : ℕ) := List (List (ℕ × ℕ))

/-- A function that checks if a list of polygons can be arranged to form a square of side length n. -/
def CanFormSquare (pieces : List (List (ℕ × ℕ))) (n : ℕ) : Prop := sorry

/-- A function that checks if two lists of polygons are equivalent up to translation and rotation. -/
def AreEquivalent (pieces1 pieces2 : List (List (ℕ × ℕ))) : Prop := sorry

theorem square_dissection_theorem :
  ∃ (d : Dissection 7),
    d.length ≤ 5 ∧
    ∃ (s1 s2 s3 : List (List (ℕ × ℕ))),
      CanFormSquare s1 6 ∧
      CanFormSquare s2 3 ∧
      CanFormSquare s3 2 ∧
      AreEquivalent (s1 ++ s2 ++ s3) d :=
sorry

end NUMINAMATH_CALUDE_square_dissection_theorem_l366_36685


namespace NUMINAMATH_CALUDE_range_of_x_l366_36629

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) : f (x^2 - 4) < 2 → x ∈ Set.Ioo (-Real.sqrt 5) (-2) ∪ Set.Ioo 2 (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l366_36629


namespace NUMINAMATH_CALUDE_correct_algebraic_operation_l366_36668

theorem correct_algebraic_operation (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_operation_l366_36668


namespace NUMINAMATH_CALUDE_equation_solution_l366_36698

theorem equation_solution (x : ℝ) : 
  Real.sqrt (1 + Real.sqrt (4 + Real.sqrt (2 * x + 3))) = (1 + Real.sqrt (2 * x + 3)) ^ (1/4) → 
  x = -23/32 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l366_36698


namespace NUMINAMATH_CALUDE_a_less_than_11_necessary_not_sufficient_l366_36670

theorem a_less_than_11_necessary_not_sufficient :
  (∀ a : ℝ, (∃ x : ℝ, x^2 - 2*x + a < 0) → a < 11) ∧
  (∃ a : ℝ, a < 11 ∧ ¬(∃ x : ℝ, x^2 - 2*x + a < 0)) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_11_necessary_not_sufficient_l366_36670


namespace NUMINAMATH_CALUDE_board_tileable_iff_divisibility_l366_36669

/-- A board is tileable if it can be covered completely with 3×1 tiles -/
def is_tileable (m n : ℕ) : Prop :=
  ∃ (tiling : Set (ℕ × ℕ × Bool)), 
    (∀ (tile : ℕ × ℕ × Bool), tile ∈ tiling → 
      (let (x, y, horizontal) := tile
       (x ≥ m ∨ y ≥ m) ∧ x < n ∧ y < n ∧
       (if horizontal then x + 2 < n else y + 2 < n))) ∧
    (∀ (i j : ℕ), m ≤ i ∧ i < n ∧ m ≤ j ∧ j < n → 
      ∃! (tile : ℕ × ℕ × Bool), tile ∈ tiling ∧
        (let (x, y, horizontal) := tile
         i ∈ Set.range (fun k => x + k) ∧ 
         j ∈ Set.range (fun k => y + k) ∧
         (if horizontal then x + 2 = i else y + 2 = j)))

/-- The main theorem -/
theorem board_tileable_iff_divisibility {m n : ℕ} (h_pos : 0 < m) (h_lt : m < n) :
  is_tileable m n ↔ (n - m) * (n + m) % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_board_tileable_iff_divisibility_l366_36669


namespace NUMINAMATH_CALUDE_constant_ratio_solution_l366_36649

/-- The constant ratio of (3x - 4) to (y + 15) -/
def k (x y : ℚ) : ℚ := (3 * x - 4) / (y + 15)

theorem constant_ratio_solution (x₀ y₀ x₁ y₁ : ℚ) 
  (h₀ : y₀ = 4)
  (h₁ : x₀ = 5)
  (h₂ : y₁ = 15)
  (h₃ : k x₀ y₀ = k x₁ y₁) :
  x₁ = 406 / 57 := by
  sorry

end NUMINAMATH_CALUDE_constant_ratio_solution_l366_36649


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l366_36693

def num_arrangements (num_books : ℕ) (num_pushkin : ℕ) (num_tarle : ℕ) (height_pushkin : ℕ) (height_tarle : ℕ) (height_center : ℕ) : ℕ :=
  3 * (Nat.factorial 2) * (Nat.factorial 4)

theorem book_arrangement_proof :
  num_arrangements 7 2 4 30 25 40 = 144 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l366_36693


namespace NUMINAMATH_CALUDE_chips_yield_more_ounces_l366_36666

def total_ounces (budget : ℚ) (price_per_bag : ℚ) (ounces_per_bag : ℚ) : ℚ :=
  (budget / price_per_bag).floor * ounces_per_bag

theorem chips_yield_more_ounces : 
  let budget : ℚ := 7
  let candy_price : ℚ := 1
  let candy_ounces : ℚ := 12
  let chips_price : ℚ := 1.4
  let chips_ounces : ℚ := 17
  total_ounces budget chips_price chips_ounces > total_ounces budget candy_price candy_ounces := by
  sorry

end NUMINAMATH_CALUDE_chips_yield_more_ounces_l366_36666


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l366_36606

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 1 > 0 ∧ x - 3 < 2}
  S = Set.Ioo (-1) 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l366_36606


namespace NUMINAMATH_CALUDE_vasya_distance_fraction_l366_36621

/-- Represents the fraction of the total distance driven by each person -/
structure DistanceFractions where
  anton : ℚ
  vasya : ℚ
  sasha : ℚ
  dima : ℚ

/-- Theorem stating that given the conditions, Vasya drove 2/5 of the total distance -/
theorem vasya_distance_fraction 
  (df : DistanceFractions)
  (h1 : df.anton = df.vasya / 2)
  (h2 : df.sasha = df.anton + df.dima)
  (h3 : df.dima = 1 / 10)
  (h4 : df.anton + df.vasya + df.sasha + df.dima = 1) :
  df.vasya = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_distance_fraction_l366_36621


namespace NUMINAMATH_CALUDE_city_connections_l366_36623

/-- The number of cities in the problem -/
def num_cities : ℕ := 6

/-- The function to calculate the number of unique pairwise connections -/
def unique_connections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 6 cities, the number of unique pairwise connections is 15 -/
theorem city_connections : unique_connections num_cities = 15 := by
  sorry

end NUMINAMATH_CALUDE_city_connections_l366_36623


namespace NUMINAMATH_CALUDE_number_of_divisors_of_30_l366_36662

theorem number_of_divisors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_30_l366_36662


namespace NUMINAMATH_CALUDE_apple_tree_width_proof_l366_36661

/-- The width of an apple tree in Quinton's backyard -/
def apple_tree_width : ℝ := 10

/-- The space between apple trees -/
def apple_tree_space : ℝ := 12

/-- The width of a peach tree -/
def peach_tree_width : ℝ := 12

/-- The space between peach trees -/
def peach_tree_space : ℝ := 15

/-- The total space taken by all trees -/
def total_space : ℝ := 71

theorem apple_tree_width_proof :
  2 * apple_tree_width + apple_tree_space + 2 * peach_tree_width + peach_tree_space = total_space :=
by sorry

end NUMINAMATH_CALUDE_apple_tree_width_proof_l366_36661


namespace NUMINAMATH_CALUDE_school_population_l366_36601

/-- Given the initial number of girls and boys in a school, and the number of additional girls who joined,
    calculate the total number of pupils after the new girls joined. -/
theorem school_population (initial_girls initial_boys additional_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  additional_girls = 418 →
  initial_girls + initial_boys + additional_girls = 1346 := by
sorry

end NUMINAMATH_CALUDE_school_population_l366_36601


namespace NUMINAMATH_CALUDE_at_most_two_greater_than_one_l366_36678

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬(2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_two_greater_than_one_l366_36678


namespace NUMINAMATH_CALUDE_unique_values_l366_36688

/-- The polynomial we're working with -/
def f (p q : ℤ) (x : ℝ) : ℝ := x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 8

/-- The condition that the polynomial is divisible by (x + 2)(x - 1) -/
def is_divisible (p q : ℤ) : Prop :=
  ∀ x : ℝ, (x + 2 = 0 ∨ x - 1 = 0) → f p q x = 0

/-- The theorem stating that p = -54 and q = -48 are the unique values satisfying the condition -/
theorem unique_values :
  ∃! (p q : ℤ), is_divisible p q ∧ p = -54 ∧ q = -48 := by sorry

end NUMINAMATH_CALUDE_unique_values_l366_36688


namespace NUMINAMATH_CALUDE_managers_wage_l366_36625

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The wages at Joe's Steakhouse satisfy the given conditions -/
def valid_wages (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.25 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.manager - 3.1875

theorem managers_wage (w : Wages) (h : valid_wages w) : w.manager = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_managers_wage_l366_36625


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l366_36640

def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_pos_geo : is_positive_geometric_sequence a)
  (h_eq : 1 / (a 2 * a 4) + 2 / (a 4 * a 4) + 1 / (a 4 * a 6) = 81) :
  1 / a 3 + 1 / a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l366_36640


namespace NUMINAMATH_CALUDE_percentage_difference_l366_36636

theorem percentage_difference : 
  (55 / 100 * 40) - (4 / 5 * 25) = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l366_36636


namespace NUMINAMATH_CALUDE_sqrt_six_range_l366_36620

theorem sqrt_six_range : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_range_l366_36620


namespace NUMINAMATH_CALUDE_homework_ratio_l366_36681

theorem homework_ratio (total : ℕ) (finished : ℕ) 
  (h1 : total = 65) (h2 : finished = 45) : 
  (finished : ℚ) / (total - finished) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_homework_ratio_l366_36681


namespace NUMINAMATH_CALUDE_fox_jeans_price_l366_36607

/-- The regular price of Fox jeans -/
def F : ℝ := 15

/-- The regular price of Pony jeans -/
def P : ℝ := 18

/-- The discount rate for Fox jeans -/
def discount_rate_fox : ℝ := 0.08

/-- The discount rate for Pony jeans -/
def discount_rate_pony : ℝ := 0.14

/-- The total savings on 5 pairs of jeans (3 Fox, 2 Pony) -/
def total_savings : ℝ := 8.64

theorem fox_jeans_price :
  F = 15 ∧
  P = 18 ∧
  discount_rate_fox + discount_rate_pony = 0.22 ∧
  3 * (F * discount_rate_fox) + 2 * (P * discount_rate_pony) = total_savings :=
by sorry

end NUMINAMATH_CALUDE_fox_jeans_price_l366_36607


namespace NUMINAMATH_CALUDE_square_area_increase_l366_36687

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.2 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.44
  := by sorry

end NUMINAMATH_CALUDE_square_area_increase_l366_36687


namespace NUMINAMATH_CALUDE_initial_cards_proof_l366_36653

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := 455

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := 301

/-- The number of cards Nell has left -/
def cards_left : ℕ := 154

/-- Theorem stating that the initial number of cards is equal to the sum of cards given away and cards left -/
theorem initial_cards_proof : initial_cards = cards_given + cards_left := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_proof_l366_36653


namespace NUMINAMATH_CALUDE_profit_achieved_min_lemons_optimal_l366_36646

/-- The number of lemons bought in one purchase -/
def lemons_bought : ℕ := 4

/-- The cost in cents for buying lemons_bought lemons -/
def buying_cost : ℕ := 25

/-- The number of lemons sold in one sale -/
def lemons_sold : ℕ := 7

/-- The revenue in cents from selling lemons_sold lemons -/
def selling_revenue : ℕ := 50

/-- The desired profit in cents -/
def desired_profit : ℕ := 150

/-- The minimum number of lemons needed to be sold to achieve the desired profit -/
def min_lemons_to_sell : ℕ := 169

theorem profit_achieved (n : ℕ) : n ≥ min_lemons_to_sell →
  (n * selling_revenue / lemons_sold - n * buying_cost / lemons_bought) ≥ desired_profit :=
by sorry

theorem min_lemons_optimal : 
  ∀ m : ℕ, m < min_lemons_to_sell →
  (m * selling_revenue / lemons_sold - m * buying_cost / lemons_bought) < desired_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_achieved_min_lemons_optimal_l366_36646


namespace NUMINAMATH_CALUDE_parabola_c_value_l366_36658

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 3 = 5 →  -- vertex condition
  p.x_coord 6 = 0 →  -- point condition
  p.c = 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l366_36658


namespace NUMINAMATH_CALUDE_cone_max_volume_surface_ratio_l366_36631

/-- For a cone with slant height 2, the ratio of its volume to its lateral surface area
    is maximized when the radius of its base is √2. -/
theorem cone_max_volume_surface_ratio (r : ℝ) (h : ℝ) : 
  let l : ℝ := 2
  let S := 2 * Real.pi * r
  let V := (1/3) * Real.pi * r^2 * Real.sqrt (l^2 - r^2)
  (∀ r' : ℝ, 0 < r' → V / S ≤ ((1/3) * Real.pi * r'^2 * Real.sqrt (l^2 - r'^2)) / (2 * Real.pi * r')) →
  r = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_cone_max_volume_surface_ratio_l366_36631


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l366_36654

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18)
  (h2 : a * b + c + d = 85)
  (h3 : a * d + b * c = 187)
  (h4 : c * d = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 18 ∧ 
    a' * b' + c' + d' = 85 ∧ 
    a' * d' + b' * c' = 187 ∧ 
    c' * d' = 110 ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 = 120 :=
by sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_max_sum_of_squares_l366_36654


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l366_36624

/-- Acme's setup fee -/
def acme_setup : ℕ := 75

/-- Acme's per-shirt cost -/
def acme_per_shirt : ℕ := 8

/-- Gamma's per-shirt cost -/
def gamma_per_shirt : ℕ := 16

/-- The minimum number of shirts for which Acme becomes cheaper than Gamma -/
def min_shirts : ℕ := 10

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts < gamma_per_shirt * min_shirts ∧
  ∀ n : ℕ, n < min_shirts →
    acme_setup + acme_per_shirt * n ≥ gamma_per_shirt * n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l366_36624


namespace NUMINAMATH_CALUDE_integer_product_condition_l366_36651

theorem integer_product_condition (a : ℚ) : 
  (∀ n : ℕ, ∃ k : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = k) ↔ 
  (∃ k : ℤ, a = k / 6) :=
sorry

end NUMINAMATH_CALUDE_integer_product_condition_l366_36651


namespace NUMINAMATH_CALUDE_ceiling_sum_of_roots_l366_36615

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_roots_l366_36615


namespace NUMINAMATH_CALUDE_intersection_A_B_l366_36632

def set_A : Set ℝ := {x | x^2 - 2*x < 0}
def set_B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_A_B : set_A ∩ set_B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l366_36632


namespace NUMINAMATH_CALUDE_equation_solution_l366_36675

theorem equation_solution : 
  ∃ x : ℚ, (x + 3*x = 500 - (4*x + 5*x)) ∧ (x = 500/13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l366_36675


namespace NUMINAMATH_CALUDE_probability_of_six_red_balls_l366_36604

def total_balls : ℕ := 100
def red_balls : ℕ := 80
def white_balls : ℕ := 20
def drawn_balls : ℕ := 10
def red_drawn : ℕ := 6

theorem probability_of_six_red_balls :
  (Nat.choose red_balls red_drawn * Nat.choose white_balls (drawn_balls - red_drawn)) / 
  Nat.choose total_balls drawn_balls = 
  (Nat.choose red_balls red_drawn * Nat.choose white_balls (drawn_balls - red_drawn)) / 
  Nat.choose total_balls drawn_balls := by sorry

end NUMINAMATH_CALUDE_probability_of_six_red_balls_l366_36604


namespace NUMINAMATH_CALUDE_books_ratio_l366_36699

/-- Given the number of books Elmo, Laura, and Stu have, prove the ratio of Laura's books to Stu's books -/
theorem books_ratio (elmo_books laura_books stu_books : ℕ) : 
  elmo_books = 24 →
  stu_books = 4 →
  elmo_books = 3 * laura_books →
  laura_books / stu_books = 2 := by
sorry

end NUMINAMATH_CALUDE_books_ratio_l366_36699


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l366_36648

theorem quadratic_inequality_solution (x : ℤ) :
  1 ≤ x ∧ x ≤ 10 → (x^2 < 3*x ↔ x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l366_36648


namespace NUMINAMATH_CALUDE_girls_in_class_l366_36626

/-- Proves the number of girls in a class with a given ratio and total students -/
theorem girls_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) :
  total = 63 ∧ girls_ratio = 4 ∧ boys_ratio = 3 →
  (girls_ratio * total) / (girls_ratio + boys_ratio) = 36 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l366_36626


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l366_36656

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -2 ∨ x > 5}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l366_36656


namespace NUMINAMATH_CALUDE_even_sum_probability_l366_36695

/-- Represents a 4x4 grid filled with numbers from 1 to 16 -/
def Grid := Fin 4 → Fin 4 → Fin 16

/-- Checks if a list of numbers has an even sum -/
def hasEvenSum (l : List (Fin 16)) : Prop :=
  (l.map (fun x => x.val + 1)).sum % 2 = 0

/-- Checks if all rows and columns in a grid have even sums -/
def allRowsAndColumnsEven (g : Grid) : Prop :=
  (∀ i : Fin 4, hasEvenSum [g i 0, g i 1, g i 2, g i 3]) ∧
  (∀ j : Fin 4, hasEvenSum [g 0 j, g 1 j, g 2 j, g 3 j])

/-- The total number of ways to arrange 16 numbers in a 4x4 grid -/
def totalArrangements : ℕ := 20922789888000

/-- The number of valid arrangements with even sums in all rows and columns -/
def validArrangements : ℕ := 36

theorem even_sum_probability :
  (validArrangements : ℚ) / totalArrangements =
  (36 : ℚ) / 20922789888000 :=
sorry

end NUMINAMATH_CALUDE_even_sum_probability_l366_36695


namespace NUMINAMATH_CALUDE_pauls_birthday_crayons_l366_36641

/-- The number of crayons Paul received for his birthday -/
def crayons_received (crayons_left : ℕ) (crayons_lost_or_given : ℕ) 
  (crayons_lost : ℕ) (crayons_given : ℕ) : ℕ :=
  crayons_left + crayons_lost_or_given

/-- Theorem stating the number of crayons Paul received for his birthday -/
theorem pauls_birthday_crayons :
  ∃ (crayons_lost crayons_given : ℕ),
    crayons_lost = 2 * crayons_given ∧
    crayons_lost + crayons_given = 9750 ∧
    crayons_received 2560 9750 crayons_lost crayons_given = 12310 := by
  sorry


end NUMINAMATH_CALUDE_pauls_birthday_crayons_l366_36641


namespace NUMINAMATH_CALUDE_two_digit_numbers_count_l366_36628

def digits_a : Finset Nat := {1, 2, 3, 4, 5, 6}
def digits_b : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

def is_two_digit (n : Nat) : Prop := n ≥ 10 ∧ n ≤ 99

def count_two_digit_numbers (digits : Finset Nat) : Nat :=
  (digits.filter (λ d => d > 0)).card * digits.card

theorem two_digit_numbers_count :
  (count_two_digit_numbers digits_a = 36) ∧
  (count_two_digit_numbers digits_b = 42) := by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_count_l366_36628


namespace NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l366_36694

theorem min_value_of_quartic_plus_constant :
  ∃ (min : ℝ), min = 2023 ∧ ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l366_36694


namespace NUMINAMATH_CALUDE_prop_one_correct_prop_two_not_always_true_prop_three_not_always_true_l366_36643

-- Define the custom distance function
def customDist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

-- Proposition 1
theorem prop_one_correct (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) 
  (h₁ : x₀ ∈ Set.Icc x₁ x₂) (h₂ : y₀ ∈ Set.Icc y₁ y₂) :
  customDist x₁ y₁ x₀ y₀ + customDist x₀ y₀ x₂ y₂ = customDist x₁ y₁ x₂ y₂ := by sorry

-- Proposition 2
theorem prop_two_not_always_true :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    customDist x₁ y₁ x₂ y₂ + customDist x₂ y₂ x₃ y₃ ≤ customDist x₁ y₁ x₃ y₃ := by sorry

-- Proposition 3
theorem prop_three_not_always_true :
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∧ 
    (customDist x₁ y₁ x₂ y₂)^2 + (customDist x₁ y₁ x₃ y₃)^2 ≠ (customDist x₂ y₂ x₃ y₃)^2 := by sorry

end NUMINAMATH_CALUDE_prop_one_correct_prop_two_not_always_true_prop_three_not_always_true_l366_36643


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l366_36697

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l366_36697


namespace NUMINAMATH_CALUDE_triangle_construction_from_nagel_point_vertex_and_altitude_foot_l366_36614

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by its vertices -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- The Nagel point of a triangle -/
def nagelPoint (t : Triangle) : Point2D := sorry

/-- The foot of the altitude from a vertex -/
def altitudeFoot (t : Triangle) (v : Point2D) : Point2D := sorry

/-- Theorem: Given a Nagel point, a vertex, and the foot of the altitude from that vertex,
    a triangle can be constructed -/
theorem triangle_construction_from_nagel_point_vertex_and_altitude_foot
  (N : Point2D) (B : Point2D) (F : Point2D) :
  ∃ (t : Triangle), nagelPoint t = N ∧ t.B = B ∧ altitudeFoot t B = F :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_from_nagel_point_vertex_and_altitude_foot_l366_36614


namespace NUMINAMATH_CALUDE_tenth_square_area_l366_36618

/-- The area of the nth square in a sequence where each square is formed by connecting
    the midpoints of the previous square's sides, and the first square has a side length of 2. -/
def square_area (n : ℕ) : ℚ :=
  2 * (1 / 2) ^ (n - 1)

/-- Theorem stating that the area of the 10th square in the sequence is 1/256. -/
theorem tenth_square_area :
  square_area 10 = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_tenth_square_area_l366_36618


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l366_36692

/-- The surface area of a rectangular prism formed by three cubes -/
def surface_area_rectangular_prism (a : ℝ) : ℝ := 14 * a^2

/-- The surface area of a single cube -/
def surface_area_cube (a : ℝ) : ℝ := 6 * a^2

theorem rectangular_prism_surface_area (a : ℝ) (h : a > 0) :
  surface_area_rectangular_prism a = 3 * surface_area_cube a - 4 * a^2 :=
sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l366_36692


namespace NUMINAMATH_CALUDE_optimal_threshold_at_intersection_l366_36696

/-- Represents the height distribution of vehicles for a given class --/
def HeightDistribution := ℝ → ℝ

/-- The cost for class 1 vehicles --/
def class1Cost : ℝ := 200

/-- The cost for class 2 vehicles --/
def class2Cost : ℝ := 300

/-- The height distribution for class 1 vehicles --/
noncomputable def class1Distribution : HeightDistribution := sorry

/-- The height distribution for class 2 vehicles --/
noncomputable def class2Distribution : HeightDistribution := sorry

/-- The intersection point of the two height distributions --/
noncomputable def intersectionPoint : ℝ := sorry

/-- The error function for a given threshold --/
def errorFunction (h : ℝ) : ℝ := sorry

/-- Theorem: The optimal threshold that minimizes classification errors
    is at the intersection point of the two height distributions --/
theorem optimal_threshold_at_intersection :
  ∀ h : ℝ, h ≠ intersectionPoint → errorFunction h > errorFunction intersectionPoint :=
by sorry

end NUMINAMATH_CALUDE_optimal_threshold_at_intersection_l366_36696


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l366_36686

/-- Sum of tens and ones digits of (3+4)^11 -/
theorem sum_of_digits_of_power : ∃ (n : ℕ), 
  (3 + 4)^11 = n ∧ 
  (n / 10 % 10 + n % 10 = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l366_36686


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l366_36611

/-- The total surface area of a pyramid formed from a cube -/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) : 
  let cube_edge := a
  let base_side := a * Real.sqrt 2 / 2
  let slant_height := 3 * a * Real.sqrt 2 / 4
  let lateral_area := 4 * (1/2 * base_side * slant_height)
  let base_area := base_side ^ 2
  lateral_area + base_area = 2 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_surface_area_l366_36611


namespace NUMINAMATH_CALUDE_student_number_problem_l366_36627

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 106 → x = 122 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l366_36627


namespace NUMINAMATH_CALUDE_colin_average_mile_time_l366_36619

def average_mile_time (first_mile : ℕ) (second_mile : ℕ) (third_mile : ℕ) (fourth_mile : ℕ) : ℚ :=
  (first_mile + second_mile + third_mile + fourth_mile) / 4

theorem colin_average_mile_time :
  average_mile_time 6 5 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_colin_average_mile_time_l366_36619


namespace NUMINAMATH_CALUDE_brendas_age_l366_36634

/-- Proves that Brenda's age is 8/3 years given the conditions in the problem. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)   -- Addison's age is four times Brenda's age
  (h2 : J = B + 8)   -- Janet is eight years older than Brenda
  (h3 : A = J)       -- Addison and Janet are twins
  : B = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l366_36634


namespace NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l366_36622

/-- The amount Mary Anne spends on sparkling water in a year -/
def sparkling_water_cost (daily_consumption : ℚ) (bottle_cost : ℚ) : ℚ :=
  (365 : ℚ) * daily_consumption * bottle_cost

theorem mary_anne_sparkling_water_cost :
  sparkling_water_cost (1/5) 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l366_36622


namespace NUMINAMATH_CALUDE_sum_of_first_20_lucky_numbers_mod_1000_l366_36602

def isLucky (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

def luckyNumbers : List ℕ :=
  (List.range 20).map (λ i => 7 * (10^i - 1) / 9)

theorem sum_of_first_20_lucky_numbers_mod_1000 :
  (luckyNumbers.sum) % 1000 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_20_lucky_numbers_mod_1000_l366_36602


namespace NUMINAMATH_CALUDE_tens_digit_of_23_pow_2023_l366_36642

theorem tens_digit_of_23_pow_2023 : ∃ n : ℕ, 23^2023 ≡ 60 + n [ZMOD 100] ∧ 0 ≤ n ∧ n < 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_23_pow_2023_l366_36642


namespace NUMINAMATH_CALUDE_jerome_has_zero_left_l366_36659

/-- Represents Jerome's financial transactions --/
def jerome_transactions (initial_euros : ℝ) (exchange_rate : ℝ) (meg_amount : ℝ) : ℝ := by
  -- Convert initial amount to dollars
  let initial_dollars := initial_euros * exchange_rate
  -- Subtract Meg's amount
  let after_meg := initial_dollars - meg_amount
  -- Subtract Bianca's amount (thrice Meg's)
  let after_bianca := after_meg - (3 * meg_amount)
  -- Give all remaining money to Nathan
  exact 0

/-- Theorem stating that Jerome has $0 left after transactions --/
theorem jerome_has_zero_left : 
  ∀ (initial_euros : ℝ) (exchange_rate : ℝ) (meg_amount : ℝ),
  initial_euros > 0 ∧ exchange_rate > 0 ∧ meg_amount > 0 →
  jerome_transactions initial_euros exchange_rate meg_amount = 0 := by
  sorry

#check jerome_has_zero_left

end NUMINAMATH_CALUDE_jerome_has_zero_left_l366_36659


namespace NUMINAMATH_CALUDE_unique_digit_equation_l366_36664

/-- Represents a mapping from symbols to digits -/
def SymbolMap := Char → Fin 10

/-- Checks if a SymbolMap assigns unique digits to different symbols -/
def isValidMap (m : SymbolMap) : Prop :=
  ∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Represents the equation "华 ÷ (3 * 好) = 杯赛" -/
def equationHolds (m : SymbolMap) : Prop :=
  (m '华').val = (m '杯').val * 100 + (m '赛').val * 10 + (m '赛').val

theorem unique_digit_equation :
  ∀ m : SymbolMap,
    isValidMap m →
    equationHolds m →
    (m '好').val = 2 := by sorry

end NUMINAMATH_CALUDE_unique_digit_equation_l366_36664


namespace NUMINAMATH_CALUDE_count_standing_orders_l366_36682

/-- The number of different standing orders for 9 students -/
def standing_orders : ℕ := 20

/-- The number of students -/
def num_students : ℕ := 9

/-- The position of the tallest student (middle position) -/
def tallest_position : ℕ := 5

/-- The rank of the student who must stand next to the tallest -/
def adjacent_rank : ℕ := 4

/-- Theorem stating the number of different standing orders -/
theorem count_standing_orders :
  standing_orders = 20 ∧
  num_students = 9 ∧
  tallest_position = 5 ∧
  adjacent_rank = 4 := by
  sorry


end NUMINAMATH_CALUDE_count_standing_orders_l366_36682


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l366_36667

theorem unique_number_with_conditions : ∃! b : ℤ, 
  40 < b ∧ b < 120 ∧ 
  b % 4 = 3 ∧ 
  b % 5 = 3 ∧ 
  b % 6 = 3 ∧
  b = 63 := by sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l366_36667


namespace NUMINAMATH_CALUDE_seating_arrangement_exists_l366_36671

-- Define a type for people
def Person : Type := Fin 5

-- Define a relation for acquaintance
def Acquainted : Person → Person → Prop := sorry

-- Define the condition that among any 3 people, 2 know each other and 2 don't
axiom acquaintance_condition : 
  ∀ (a b c : Person), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((Acquainted a b ∧ Acquainted a c) ∨ 
     (Acquainted a b ∧ Acquainted b c) ∨ 
     (Acquainted a c ∧ Acquainted b c)) ∧
    ((¬Acquainted a b ∧ ¬Acquainted a c) ∨ 
     (¬Acquainted a b ∧ ¬Acquainted b c) ∨ 
     (¬Acquainted a c ∧ ¬Acquainted b c))

-- Define a circular arrangement
def CircularArrangement : Type := Fin 5 → Person

-- Define the property that each person is adjacent to two acquaintances
def ValidArrangement (arr : CircularArrangement) : Prop :=
  ∀ (i : Fin 5), 
    Acquainted (arr i) (arr ((i + 1) % 5)) ∧ 
    Acquainted (arr i) (arr ((i + 4) % 5))

-- The theorem to be proved
theorem seating_arrangement_exists : 
  ∃ (arr : CircularArrangement), ValidArrangement arr :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_exists_l366_36671


namespace NUMINAMATH_CALUDE_multiples_of_12_between_30_and_200_l366_36672

theorem multiples_of_12_between_30_and_200 : 
  (Finset.filter (fun n => n % 12 = 0 ∧ n ≥ 30 ∧ n ≤ 200) (Finset.range 201)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_between_30_and_200_l366_36672


namespace NUMINAMATH_CALUDE_negative_six_divided_by_three_l366_36609

theorem negative_six_divided_by_three : (-6) / 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_six_divided_by_three_l366_36609


namespace NUMINAMATH_CALUDE_fold_triangle_crease_length_l366_36613

theorem fold_triangle_crease_length 
  (A B C : ℝ × ℝ) 
  (h_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (h_side_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3)
  (h_side_BC : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 4)
  (h_side_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 5) :
  let D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let F : ℝ × ℝ := 
    let m := (B.2 - A.2) / (B.1 - A.1)
    let b := D.2 - m * D.1
    ((E.2 - b) / m, E.2)
  Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = 15/8 := by
sorry

end NUMINAMATH_CALUDE_fold_triangle_crease_length_l366_36613


namespace NUMINAMATH_CALUDE_kanul_spending_l366_36652

/-- The total amount Kanul had initially -/
def T : ℝ := 5714.29

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery -/
def machinery : ℝ := 1000

/-- The fraction of the total amount spent as cash -/
def cash_fraction : ℝ := 0.30

theorem kanul_spending :
  raw_materials + machinery + cash_fraction * T = T := by sorry

end NUMINAMATH_CALUDE_kanul_spending_l366_36652


namespace NUMINAMATH_CALUDE_forest_farms_theorem_l366_36683

-- Define a farm as a pair of natural numbers (total years, high-quality years)
def Farm := ℕ × ℕ

-- Function to calculate probability of selecting two high-quality years
def prob_two_high_quality (f : Farm) : ℚ :=
  let (total, high) := f
  (high.choose 2 : ℚ) / (total.choose 2 : ℚ)

-- Function to calculate probability of selecting a high-quality year
def prob_high_quality (f : Farm) : ℚ :=
  let (total, high) := f
  high / total

-- Distribution type for discrete random variable
def Distribution := List (ℕ × ℚ)

-- Function to calculate the distribution of high-quality projects
def distribution_high_quality (f1 f2 f3 : Farm) : Distribution :=
  sorry  -- Placeholder for the actual calculation

-- Main theorem
theorem forest_farms_theorem (farm_b farm_c : Farm) :
  -- Part 1
  prob_two_high_quality (7, 4) = 2/7 ∧
  -- Part 2
  distribution_high_quality (6, 3) (7, 4) (10, 5) = 
    [(0, 3/28), (1, 5/14), (2, 11/28), (3, 1/7)] ∧
  -- Part 3
  ∃ (avg_b avg_c : ℚ), 
    prob_high_quality farm_b = 4/7 ∧ 
    prob_high_quality farm_c = 1/2 ∧ 
    avg_b ≠ avg_c :=
by sorry

end NUMINAMATH_CALUDE_forest_farms_theorem_l366_36683


namespace NUMINAMATH_CALUDE_tenth_term_value_l366_36617

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_property : a 1 + a 3 + a 5 = 9
  product_property : a 3 * (a 4)^2 = 27

/-- The 10th term of the arithmetic sequence is either -39 or 30 -/
theorem tenth_term_value (seq : ArithmeticSequence) : seq.a 10 = -39 ∨ seq.a 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l366_36617


namespace NUMINAMATH_CALUDE_total_equipment_cost_l366_36663

def num_players : ℕ := 16
def jersey_cost : ℚ := 25
def shorts_cost : ℚ := 15.20
def socks_cost : ℚ := 6.80

theorem total_equipment_cost :
  (num_players : ℚ) * (jersey_cost + shorts_cost + socks_cost) = 752 := by
  sorry

end NUMINAMATH_CALUDE_total_equipment_cost_l366_36663


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l366_36616

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 1 = 0) → (x₂^2 - 2*x₂ - 1 = 0) → (x₁ + x₂ - x₁*x₂ = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l366_36616


namespace NUMINAMATH_CALUDE_S_is_infinite_l366_36637

/-- Number of distinct odd prime divisors of a natural number -/
def num_odd_prime_divisors (m : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which the number of distinct odd prime divisors of n(n+3) is divisible by 3 -/
def S : Set ℕ := {n : ℕ | 3 ∣ num_odd_prime_divisors (n * (n + 3))}

/-- The set S is infinite -/
theorem S_is_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_is_infinite_l366_36637


namespace NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l366_36635

/-- Represents the regression line equation for pig iron cost -/
def regression_line (x : ℝ) : ℝ := 256 + 2 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_relationship :
  ∀ x : ℝ, regression_line (x + 1) = regression_line x + 2 :=
by sorry

end NUMINAMATH_CALUDE_scrap_rate_cost_relationship_l366_36635


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l366_36676

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  h : ℕ
  c : ℕ
  o : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight (element : String) : ℕ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculate the molecular weight of a compound -/
def molecular_weight (compound : Compound) : ℕ :=
  compound.h * atomic_weight "H" +
  compound.c * atomic_weight "C" +
  compound.o * atomic_weight "O"

/-- Theorem: A compound with 2 H atoms, 1 C atom, and molecular weight 62 amu has 3 O atoms -/
theorem compound_oxygen_atoms (compound : Compound) :
  compound.h = 2 ∧ compound.c = 1 ∧ molecular_weight compound = 62 →
  compound.o = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l366_36676


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l366_36647

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l366_36647


namespace NUMINAMATH_CALUDE_bus_dispatch_interval_l366_36639

/-- The speed of the bus -/
def bus_speed : ℝ := sorry

/-- The speed of Xiao Wang -/
def person_speed : ℝ := sorry

/-- The interval between each bus dispatch in minutes -/
def dispatch_interval : ℝ := sorry

/-- The time between a bus passing Xiao Wang from behind in minutes -/
def overtake_time : ℝ := 6

/-- The time between a bus coming towards Xiao Wang in minutes -/
def approach_time : ℝ := 3

/-- Theorem stating that given the conditions, the dispatch interval is 4 minutes -/
theorem bus_dispatch_interval : 
  bus_speed > 0 ∧ 
  person_speed > 0 ∧ 
  person_speed < bus_speed ∧
  overtake_time * (bus_speed - person_speed) = dispatch_interval * bus_speed ∧
  approach_time * (bus_speed + person_speed) = dispatch_interval * bus_speed →
  dispatch_interval = 4 := by sorry

end NUMINAMATH_CALUDE_bus_dispatch_interval_l366_36639


namespace NUMINAMATH_CALUDE_pie_crust_flour_usage_l366_36657

/-- Given that 30 pie crusts each use 1/6 cup of flour, and 25 new pie crusts use
    the same total amount of flour, prove that each new pie crust uses 1/5 cup of flour. -/
theorem pie_crust_flour_usage
  (original_crusts : ℕ)
  (original_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (h1 : original_crusts = 30)
  (h2 : original_flour_per_crust = 1/6)
  (h3 : new_crusts = 25)
  (h4 : original_crusts * original_flour_per_crust = new_crusts * new_flour_per_crust) :
  new_flour_per_crust = 1/5 :=
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_usage_l366_36657
