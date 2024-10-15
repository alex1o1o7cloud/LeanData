import Mathlib

namespace NUMINAMATH_CALUDE_book_shelf_problem_l3831_383170

theorem book_shelf_problem (total_books : ℕ) (books_moved : ℕ) 
  (h1 : total_books = 180)
  (h2 : books_moved = 15)
  (h3 : ∃ (upper lower : ℕ), 
    upper + lower = total_books ∧ 
    (lower + books_moved) = 2 * (upper - books_moved)) :
  ∃ (original_upper original_lower : ℕ),
    original_upper = 75 ∧ 
    original_lower = 105 ∧
    original_upper + original_lower = total_books := by
  sorry

end NUMINAMATH_CALUDE_book_shelf_problem_l3831_383170


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3831_383189

theorem trigonometric_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3831_383189


namespace NUMINAMATH_CALUDE_ball_cost_l3831_383178

theorem ball_cost (C : ℝ) : 
  (C / 2 + C / 6 + C / 12 + 5 = C) → C = 20 := by sorry

end NUMINAMATH_CALUDE_ball_cost_l3831_383178


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3831_383156

/-- A line in the form (m-2)x-y+3m+2=0 passes through the point (-3, 8) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m - 2) * (-3) - 8 + 3 * m + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3831_383156


namespace NUMINAMATH_CALUDE_fraction_equality_l3831_383110

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 4) : 
  (a - 2*a*b - b) / (2*a + 7*a*b - 2*b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3831_383110


namespace NUMINAMATH_CALUDE_product_of_roots_l3831_383171

theorem product_of_roots (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2017 ∧ y₁^3 - 3*x₁^2*y₁ = 2016)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2017 ∧ y₂^3 - 3*x₂^2*y₂ = 2016)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2017 ∧ y₃^3 - 3*x₃^2*y₃ = 2016)
  (h₄ : x₄^3 - 3*x₄*y₄^2 = 2017 ∧ y₄^3 - 3*x₄^2*y₄ = 2016)
  (h₅ : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0 ∧ y₄ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) * (1 - x₄/y₄) = -1/1008 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3831_383171


namespace NUMINAMATH_CALUDE_school_boys_count_l3831_383147

theorem school_boys_count :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 64 →
  boys = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3831_383147


namespace NUMINAMATH_CALUDE_difference_of_squares_l3831_383131

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3831_383131


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3831_383140

/-- Represents the four types of crops -/
inductive Crop
| Corn
| Wheat
| Soybeans
| Potatoes

/-- Represents a position in the 3x3 grid -/
structure Position :=
  (row : Fin 3)
  (col : Fin 3)

/-- Checks if two positions are adjacent -/
def are_adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col.val + 1 = p2.col.val ∨ p2.col.val + 1 = p1.col.val)) ∨
  (p1.col = p2.col ∧ (p1.row.val + 1 = p2.row.val ∨ p2.row.val + 1 = p1.row.val))

/-- Represents a planting arrangement -/
def Arrangement := Position → Crop

/-- Checks if an arrangement is valid according to the rules -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  ∀ p1 p2 : Position,
    are_adjacent p1 p2 →
      (arr p1 ≠ arr p2) ∧
      ¬(arr p1 = Crop.Corn ∧ arr p2 = Crop.Wheat) ∧
      ¬(arr p1 = Crop.Wheat ∧ arr p2 = Crop.Corn)

/-- The main theorem to be proved -/
theorem valid_arrangements_count :
  ∃ (arrangements : Finset Arrangement),
    (∀ arr ∈ arrangements, is_valid_arrangement arr) ∧
    (∀ arr, is_valid_arrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 16 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3831_383140


namespace NUMINAMATH_CALUDE_tria_currency_base_l3831_383172

/-- Converts a number from base r to base 10 -/
def toBase10 (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement -/
theorem tria_currency_base : ∃! r : Nat, r > 1 ∧
  toBase10 [5, 3, 2] r + toBase10 [2, 6, 0] r + toBase10 [2, 0, 8] r = toBase10 [1, 0, 0, 0] r :=
by
  sorry

end NUMINAMATH_CALUDE_tria_currency_base_l3831_383172


namespace NUMINAMATH_CALUDE_sand_in_last_bag_l3831_383195

theorem sand_in_last_bag (total_sand : Nat) (bag_capacity : Nat) (h1 : total_sand = 757) (h2 : bag_capacity = 65) :
  total_sand % bag_capacity = 42 := by
sorry

end NUMINAMATH_CALUDE_sand_in_last_bag_l3831_383195


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3831_383181

theorem baseball_card_value_decrease :
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.2)
  let value_after_second_year := value_after_first_year * (1 - 0.2)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 36 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3831_383181


namespace NUMINAMATH_CALUDE_solve_for_q_l3831_383130

theorem solve_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l3831_383130


namespace NUMINAMATH_CALUDE_unit_circle_tangent_l3831_383100

theorem unit_circle_tangent (x y θ : ℝ) : 
  x^2 + y^2 = 1 →  -- Point (x, y) is on the unit circle
  x > 0 →          -- Point is in the first quadrant
  y > 0 →          -- Point is in the first quadrant
  x = Real.cos θ → -- θ is the angle from positive x-axis
  y = Real.sin θ → -- to the ray through (x, y)
  Real.arccos ((4*x + 3*y) / 5) = θ → -- Given condition
  Real.tan θ = 1/3 := by sorry

end NUMINAMATH_CALUDE_unit_circle_tangent_l3831_383100


namespace NUMINAMATH_CALUDE_laptop_price_difference_l3831_383159

/-- The price difference between two stores for Laptop Y -/
theorem laptop_price_difference
  (list_price : ℝ)
  (gadget_gurus_discount_percent : ℝ)
  (tech_trends_discount_amount : ℝ)
  (h1 : list_price = 300)
  (h2 : gadget_gurus_discount_percent = 0.15)
  (h3 : tech_trends_discount_amount = 45) :
  list_price * (1 - gadget_gurus_discount_percent) = list_price - tech_trends_discount_amount :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_difference_l3831_383159


namespace NUMINAMATH_CALUDE_fraction_inequality_l3831_383173

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3831_383173


namespace NUMINAMATH_CALUDE_museum_pictures_l3831_383168

theorem museum_pictures (zoo_pics : ℕ) (deleted_pics : ℕ) (remaining_pics : ℕ) :
  zoo_pics = 41 →
  deleted_pics = 15 →
  remaining_pics = 55 →
  ∃ museum_pics : ℕ, zoo_pics + museum_pics = remaining_pics + deleted_pics ∧ museum_pics = 29 :=
by sorry

end NUMINAMATH_CALUDE_museum_pictures_l3831_383168


namespace NUMINAMATH_CALUDE_x_value_l3831_383198

theorem x_value (x : ℝ) : x = 88 * (1 + 0.3) → x = 114.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3831_383198


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l3831_383167

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l3831_383167


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l3831_383196

/-- Given a polynomial f(x,y,z) = x³ + 2y³ + 4z³ - 6xyz, prove that for all real numbers a, b, and c,
    f(a,b,c) = 0 if and only if a + b∛2 + c∛4 = 0 -/
theorem polynomial_equivalence (a b c : ℝ) :
  a^3 + 2*b^3 + 4*c^3 - 6*a*b*c = 0 ↔ a + b*(2^(1/3)) + c*(4^(1/3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l3831_383196


namespace NUMINAMATH_CALUDE_decimal_order_l3831_383183

theorem decimal_order : 0.6 < 0.67 ∧ 0.67 < 0.676 ∧ 0.676 < 0.677 := by
  sorry

end NUMINAMATH_CALUDE_decimal_order_l3831_383183


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3831_383184

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < 3 → a * x^2 + x + b > 0) ∧
  (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 3) → a * x^2 + x + b ≤ 0) →
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3831_383184


namespace NUMINAMATH_CALUDE_fraction_modification_l3831_383135

theorem fraction_modification (a : ℕ) : (29 - a : ℚ) / (43 + a) = 3/5 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_l3831_383135


namespace NUMINAMATH_CALUDE_smallest_x_value_l3831_383127

theorem smallest_x_value (x : ℚ) : 
  (6 * (9 * x^2 + 9 * x + 10) = x * (9 * x - 45)) → x ≥ -4/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3831_383127


namespace NUMINAMATH_CALUDE_karina_birth_year_l3831_383126

def current_year : ℕ := 2022
def brother_birth_year : ℕ := 1990

theorem karina_birth_year (karina_age brother_age : ℕ) 
  (h1 : karina_age = 2 * brother_age)
  (h2 : brother_age = current_year - brother_birth_year) :
  current_year - karina_age = 1958 := by
sorry

end NUMINAMATH_CALUDE_karina_birth_year_l3831_383126


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l3831_383150

theorem consecutive_integers_average (c d : ℝ) : 
  (c ≥ 1) →  -- Ensure c is positive
  (d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l3831_383150


namespace NUMINAMATH_CALUDE_group_age_problem_l3831_383138

theorem group_age_problem (n : ℕ) (h1 : n > 0) : 
  (15 * n + 35) / (n + 1) = 17 → n = 9 := by sorry

end NUMINAMATH_CALUDE_group_age_problem_l3831_383138


namespace NUMINAMATH_CALUDE_buns_eaten_proof_l3831_383158

/-- Represents the number of buns eaten by Zhenya -/
def zhenya_buns : ℕ := 40

/-- Represents the number of buns eaten by Sasha -/
def sasha_buns : ℕ := 30

/-- The total number of buns eaten -/
def total_buns : ℕ := 70

/-- The total eating time in minutes -/
def total_time : ℕ := 180

/-- Zhenya's eating rate in buns per minute -/
def zhenya_rate : ℚ := 1/2

/-- Sasha's eating rate in buns per minute -/
def sasha_rate : ℚ := 3/10

theorem buns_eaten_proof :
  zhenya_buns + sasha_buns = total_buns ∧
  zhenya_rate * total_time = zhenya_buns ∧
  sasha_rate * total_time = sasha_buns :=
by sorry

#check buns_eaten_proof

end NUMINAMATH_CALUDE_buns_eaten_proof_l3831_383158


namespace NUMINAMATH_CALUDE_y₁_greater_than_y₂_l3831_383153

-- Define the line
def line (x : ℝ) (b : ℝ) : ℝ := -2023 * x + b

-- Define the points A and B
def point_A (y₁ : ℝ) : ℝ × ℝ := (-2, y₁)
def point_B (y₂ : ℝ) : ℝ × ℝ := (-1, y₂)

-- Theorem statement
theorem y₁_greater_than_y₂ (b y₁ y₂ : ℝ) 
  (h₁ : line (-2) b = y₁)
  (h₂ : line (-1) b = y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_greater_than_y₂_l3831_383153


namespace NUMINAMATH_CALUDE_democrat_ratio_l3831_383176

theorem democrat_ratio (total : ℕ) (female_dem : ℕ) :
  total = 840 →
  female_dem = 140 →
  (∃ (female male : ℕ),
    female + male = total ∧
    2 * female_dem = female ∧
    4 * female_dem = male) →
  3 * (2 * female_dem) = total :=
by
  sorry

end NUMINAMATH_CALUDE_democrat_ratio_l3831_383176


namespace NUMINAMATH_CALUDE_complex_projective_transformation_properties_l3831_383119

-- Define a complex projective transformation
noncomputable def ComplexProjectiveTransformation := ℂ → ℂ

-- State the theorem
theorem complex_projective_transformation_properties
  (f : ComplexProjectiveTransformation) :
  (∃ (a b c d : ℂ), ∀ z, f z = (a * z + b) / (c * z + d)) ∧
  (∃! (p q : ℂ), f p = p ∧ f q = q) :=
sorry

end NUMINAMATH_CALUDE_complex_projective_transformation_properties_l3831_383119


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3831_383160

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3831_383160


namespace NUMINAMATH_CALUDE_square_diagonal_from_rectangle_area_l3831_383128

theorem square_diagonal_from_rectangle_area (length width : ℝ) (h1 : length = 90) (h2 : width = 80) :
  let rectangle_area := length * width
  let square_side := (rectangle_area : ℝ).sqrt
  let square_diagonal := (2 * square_side ^ 2).sqrt
  square_diagonal = 120 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_from_rectangle_area_l3831_383128


namespace NUMINAMATH_CALUDE_A_power_100_l3831_383164

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 1; -9, -2]

theorem A_power_100 : A ^ 100 = !![301, 100; -900, -299] := by sorry

end NUMINAMATH_CALUDE_A_power_100_l3831_383164


namespace NUMINAMATH_CALUDE_union_complement_equality_l3831_383114

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l3831_383114


namespace NUMINAMATH_CALUDE_fraction_product_cube_main_problem_l3831_383108

theorem fraction_product_cube (a b c d e f : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 * (e / f) ^ 3 = ((a * c * e) / (b * d * f)) ^ 3 :=
sorry

theorem main_problem : 
  (8 / 9) ^ 3 * (1 / 3) ^ 3 * (2 / 5) ^ 3 = 4096 / 2460375 :=
sorry

end NUMINAMATH_CALUDE_fraction_product_cube_main_problem_l3831_383108


namespace NUMINAMATH_CALUDE_student_weight_l3831_383107

theorem student_weight (student sister brother : ℝ) 
  (h1 : student - 5 = 2 * sister)
  (h2 : student + sister + brother = 150)
  (h3 : brother = sister - 10) :
  student = 82.5 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l3831_383107


namespace NUMINAMATH_CALUDE_rectangle_width_problem_l3831_383192

theorem rectangle_width_problem (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 75 →
  width = 5 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_problem_l3831_383192


namespace NUMINAMATH_CALUDE_subset_union_implies_complement_superset_l3831_383121

universe u

theorem subset_union_implies_complement_superset
  {U : Type u} [CompleteLattice U]
  (M N : Set U) (h : M ∪ N = N) :
  (M : Set U)ᶜ ⊇ (N : Set U)ᶜ :=
by sorry

end NUMINAMATH_CALUDE_subset_union_implies_complement_superset_l3831_383121


namespace NUMINAMATH_CALUDE_x0_value_l3831_383162

-- Define the function f
def f (x : ℝ) : ℝ := x^5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 5 * x^4

-- Theorem statement
theorem x0_value (x₀ : ℝ) (h : f' x₀ = 20) : x₀ = Real.sqrt 2 ∨ x₀ = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l3831_383162


namespace NUMINAMATH_CALUDE_apples_after_sharing_l3831_383177

/-- The total number of apples Craig and Judy have after sharing -/
def total_apples_after_sharing (craig_initial : ℕ) (judy_initial : ℕ) (craig_shared : ℕ) (judy_shared : ℕ) : ℕ :=
  (craig_initial - craig_shared) + (judy_initial - judy_shared)

/-- Theorem: Given the initial and shared apple counts, Craig and Judy have 19 apples together after sharing -/
theorem apples_after_sharing :
  total_apples_after_sharing 20 11 7 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_apples_after_sharing_l3831_383177


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3831_383142

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 288) + (Real.sqrt 245 / Real.sqrt 96) = (30 + 7 * Real.sqrt 30) / 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3831_383142


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l3831_383102

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l3831_383102


namespace NUMINAMATH_CALUDE_expression_value_l3831_383151

theorem expression_value : 
  2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3831_383151


namespace NUMINAMATH_CALUDE_initial_knives_count_l3831_383144

/-- Represents the initial number of knives --/
def initial_knives : ℕ := 24

/-- Represents the initial number of teaspoons --/
def initial_teaspoons : ℕ := 2 * initial_knives

/-- Represents the additional knives --/
def additional_knives : ℕ := initial_knives / 3

/-- Represents the additional teaspoons --/
def additional_teaspoons : ℕ := (2 * initial_teaspoons) / 3

/-- The total number of cutlery pieces after additions --/
def total_cutlery : ℕ := 112

theorem initial_knives_count : 
  initial_knives + initial_teaspoons + additional_knives + additional_teaspoons = total_cutlery :=
by sorry

end NUMINAMATH_CALUDE_initial_knives_count_l3831_383144


namespace NUMINAMATH_CALUDE_matthew_cakes_l3831_383139

theorem matthew_cakes (initial_crackers : ℕ) (friends : ℕ) (crackers_eaten : ℕ) :
  initial_crackers = 22 →
  friends = 11 →
  crackers_eaten = 2 →
  ∃ (initial_cakes : ℕ),
    initial_cakes = 22 ∧
    initial_crackers / friends = crackers_eaten ∧
    initial_cakes / friends = crackers_eaten :=
by sorry

end NUMINAMATH_CALUDE_matthew_cakes_l3831_383139


namespace NUMINAMATH_CALUDE_root_cubic_value_l3831_383190

theorem root_cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 - 7 = -6 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_value_l3831_383190


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l3831_383133

/-- Represents the frequencies of fruits in a survey --/
structure FruitSurvey where
  apples : ℕ
  bananas : ℕ
  cherries : ℕ
  oranges : ℕ
  grapes : ℕ

/-- Calculates the total number of responses in the survey --/
def totalResponses (survey : FruitSurvey) : ℕ :=
  survey.apples + survey.bananas + survey.cherries + survey.oranges + survey.grapes

/-- Calculates the percentage of respondents who preferred apples --/
def applePercentage (survey : FruitSurvey) : ℚ :=
  (survey.apples : ℚ) / (totalResponses survey : ℚ) * 100

/-- The given survey results --/
def givenSurvey : FruitSurvey :=
  { apples := 70
  , bananas := 50
  , cherries := 30
  , oranges := 50
  , grapes := 40 }

theorem apple_preference_percentage :
  applePercentage givenSurvey = 29 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l3831_383133


namespace NUMINAMATH_CALUDE_largest_common_term_l3831_383129

def is_in_sequence (start : ℤ) (diff : ℤ) (n : ℤ) : Prop :=
  ∃ k : ℤ, n = start + k * diff

theorem largest_common_term : ∃ n : ℤ,
  (1 ≤ n ∧ n ≤ 100) ∧
  (is_in_sequence 2 5 n) ∧
  (is_in_sequence 3 8 n) ∧
  (∀ m : ℤ, (1 ≤ m ∧ m ≤ 100) → 
    (is_in_sequence 2 5 m) → 
    (is_in_sequence 3 8 m) → 
    m ≤ n) ∧
  n = 67 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_term_l3831_383129


namespace NUMINAMATH_CALUDE_classroom_shirts_and_shorts_l3831_383113

theorem classroom_shirts_and_shorts (total_students : ℕ) 
  (h1 : total_students = 81)
  (h2 : ∃ striped_shirts : ℕ, striped_shirts = 2 * total_students / 3)
  (h3 : ∃ checkered_shirts : ℕ, checkered_shirts = total_students - striped_shirts)
  (h4 : ∃ shorts : ℕ, striped_shirts = shorts + 8) :
  ∃ difference : ℕ, shorts = checkered_shirts + difference ∧ difference = 19 := by
  sorry

end NUMINAMATH_CALUDE_classroom_shirts_and_shorts_l3831_383113


namespace NUMINAMATH_CALUDE_max_servings_emily_l3831_383163

/-- Represents the recipe requirements for 4 servings --/
structure Recipe where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ

/-- Represents Emily's available ingredients --/
structure Available where
  bananas : ℕ
  strawberries : ℕ
  yogurt : ℕ

/-- Calculates the maximum number of servings that can be made --/
def max_servings (recipe : Recipe) (available : Available) : ℕ :=
  min
    (available.bananas * 4 / recipe.bananas)
    (min
      (available.strawberries * 4 / recipe.strawberries)
      (available.yogurt * 4 / recipe.yogurt))

theorem max_servings_emily :
  let recipe := Recipe.mk 3 1 2
  let available := Available.mk 10 3 12
  max_servings recipe available = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l3831_383163


namespace NUMINAMATH_CALUDE_solution_ranges_l3831_383118

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2*m - 2

-- Define the conditions
def has_solution_in_closed_interval (m : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc 0 (3/2) ∧ quadratic m x = 0

def has_solution_in_open_interval (m : ℝ) : Prop :=
  ∃ x, x ∈ Set.Ioo 0 (3/2) ∧ quadratic m x = 0

def has_exactly_one_solution_in_open_interval (m : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo 0 (3/2) ∧ quadratic m x = 0

def has_two_solutions_in_closed_interval (m : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ x ∈ Set.Icc 0 (3/2) ∧ y ∈ Set.Icc 0 (3/2) ∧ quadratic m x = 0 ∧ quadratic m y = 0

-- Theorem statements
theorem solution_ranges :
  (∀ m, has_solution_in_closed_interval m ↔ m ∈ Set.Icc (-1/2) (4 - 2*Real.sqrt 2)) ∧
  (∀ m, has_solution_in_open_interval m ↔ m ∈ Set.Ico (-1/2) (4 - 2*Real.sqrt 2)) ∧
  (∀ m, has_exactly_one_solution_in_open_interval m ↔ m ∈ Set.Ioc (-1/2) 1) ∧
  (∀ m, has_two_solutions_in_closed_interval m ↔ m ∈ Set.Ioo 1 (4 - 2*Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_solution_ranges_l3831_383118


namespace NUMINAMATH_CALUDE_expand_product_l3831_383152

theorem expand_product (x : ℝ) : (x + 3) * (x + 4) = x^2 + 7*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3831_383152


namespace NUMINAMATH_CALUDE_power_of_power_l3831_383149

theorem power_of_power (a : ℝ) : (a^2)^10 = a^20 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l3831_383149


namespace NUMINAMATH_CALUDE_pool_earnings_theorem_l3831_383175

def calculate_weekly_earnings (kid_fee : ℚ) (adult_fee : ℚ) (weekend_surcharge : ℚ) 
  (weekday_kids : ℕ) (weekday_adults : ℕ) (weekend_kids : ℕ) (weekend_adults : ℕ) 
  (weekdays : ℕ) (weekend_days : ℕ) : ℚ :=
  let weekday_earnings := (kid_fee * weekday_kids + adult_fee * weekday_adults) * weekdays
  let weekend_kid_fee := kid_fee * (1 + weekend_surcharge)
  let weekend_adult_fee := adult_fee * (1 + weekend_surcharge)
  let weekend_earnings := (weekend_kid_fee * weekend_kids + weekend_adult_fee * weekend_adults) * weekend_days
  weekday_earnings + weekend_earnings

theorem pool_earnings_theorem : 
  calculate_weekly_earnings 3 6 (1/2) 8 10 12 15 5 2 = 798 := by
  sorry

end NUMINAMATH_CALUDE_pool_earnings_theorem_l3831_383175


namespace NUMINAMATH_CALUDE_sum_59_28_rounded_equals_90_l3831_383193

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem sum_59_28_rounded_equals_90 : 
  round_to_nearest_ten (59 + 28) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_59_28_rounded_equals_90_l3831_383193


namespace NUMINAMATH_CALUDE_find_b_value_l3831_383141

theorem find_b_value (a b : ℝ) (eq1 : 3 * a + 2 = 2) (eq2 : b - a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l3831_383141


namespace NUMINAMATH_CALUDE_ten_steps_climb_ways_l3831_383174

/-- Number of ways to climb n steps when one can move to the next step or skip one step -/
def climbWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => climbWays (n + 1) + climbWays n

/-- The number of ways to climb 10 steps is 89 -/
theorem ten_steps_climb_ways : climbWays 10 = 89 := by sorry

end NUMINAMATH_CALUDE_ten_steps_climb_ways_l3831_383174


namespace NUMINAMATH_CALUDE_min_containers_correct_l3831_383104

/-- Calculates the minimum number of containers needed to transport boxes with weight restrictions. -/
def min_containers (total_boxes : ℕ) (main_box_weight : ℕ) (light_boxes : ℕ) (light_box_weight : ℕ) (max_container_weight : ℕ) : ℕ :=
  let total_weight := (total_boxes - light_boxes) * main_box_weight + light_boxes * light_box_weight
  let boxes_per_container := max_container_weight * 1000 / main_box_weight
  (total_boxes + boxes_per_container - 1) / boxes_per_container

theorem min_containers_correct :
  min_containers 90000 3300 5000 200 100 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_min_containers_correct_l3831_383104


namespace NUMINAMATH_CALUDE_special_function_inequality_l3831_383187

/-- A non-negative differentiable function satisfying certain conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  non_negative : ∀ x ∈ domain, f x ≥ 0
  differentiable : DifferentiableOn ℝ f domain
  condition : ∀ x ∈ domain, x * (deriv f x) + f x ≤ 0

/-- Theorem statement -/
theorem special_function_inequality (φ : SpecialFunction) (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a < b) :
    b * φ.f b ≤ a * φ.f a := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l3831_383187


namespace NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l3831_383180

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the form y = kx -/
structure Line where
  k : ℝ

/-- Predicate to check if a point is on the x-axis -/
def onXAxis (p : Point) : Prop :=
  p.y = 0

/-- Predicate to check if two points are symmetric about a line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  (p1.y + p2.y) / 2 = l.k * ((p1.x + p2.x) / 2) ∧
  (p1.y - p2.y) / (p1.x - p2.x) * l.k = -1

theorem symmetric_point_on_x_axis (A : Point) (l : Line) :
  A.x = 3 ∧ A.y = 5 →
  ∃ (B : Point), areSymmetric A B l ∧ onXAxis B →
  l.k = (-3 + Real.sqrt 34) / 5 ∨ l.k = (-3 - Real.sqrt 34) / 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_on_x_axis_l3831_383180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_specific_l3831_383148

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (start : Int) (diff : Int) (endInclusive : Int) : Nat :=
  if start > endInclusive then 0
  else ((endInclusive - start) / diff).toNat + 1

theorem arithmetic_sequence_length_specific :
  arithmeticSequenceLength (-48) 7 119 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_specific_l3831_383148


namespace NUMINAMATH_CALUDE_range_of_a_l3831_383132

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) a, f x ∈ Set.Icc (-4 : ℝ) 5) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) a, f x = -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) a, f x = 5) →
  a ∈ Set.Icc 2 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3831_383132


namespace NUMINAMATH_CALUDE_absent_laborers_l3831_383145

/-- Proves that 6 laborers were absent given the problem conditions -/
theorem absent_laborers (total_laborers : ℕ) (planned_days : ℕ) (actual_days : ℕ)
  (h1 : total_laborers = 15)
  (h2 : planned_days = 9)
  (h3 : actual_days = 15)
  (h4 : total_laborers * planned_days = (total_laborers - absent) * actual_days) :
  absent = 6 := by
  sorry

end NUMINAMATH_CALUDE_absent_laborers_l3831_383145


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l3831_383134

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0.5 then 1 / (0.5 - x) else 0.5

theorem f_satisfies_equation :
  ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l3831_383134


namespace NUMINAMATH_CALUDE_M_greater_than_N_l3831_383161

theorem M_greater_than_N (a : ℝ) : 2*a*(a-2) + 7 > (a-2)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l3831_383161


namespace NUMINAMATH_CALUDE_sum_of_six_odd_squares_not_1986_l3831_383155

theorem sum_of_six_odd_squares_not_1986 : ¬ ∃ (a b c d e f : ℤ), 
  (Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f) ∧
  (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 1986) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_six_odd_squares_not_1986_l3831_383155


namespace NUMINAMATH_CALUDE_car_journey_average_speed_l3831_383103

-- Define the parameters of the journey
def distance_part1 : ℝ := 18  -- km
def time_part1 : ℝ := 24      -- minutes
def time_part2 : ℝ := 35      -- minutes
def speed_part2 : ℝ := 72     -- km/h

-- Define the theorem
theorem car_journey_average_speed :
  let total_distance := distance_part1 + speed_part2 * (time_part2 / 60)
  let total_time := time_part1 + time_part2
  let average_speed := total_distance / (total_time / 60)
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.005 ∧ |average_speed - 61.02| < ε :=
by sorry

end NUMINAMATH_CALUDE_car_journey_average_speed_l3831_383103


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3831_383116

/-- Given (1 + 2i)a + b = 2i, where a and b are real numbers, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3831_383116


namespace NUMINAMATH_CALUDE_acute_triangle_angle_b_l3831_383143

theorem acute_triangle_angle_b (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute triangle condition
  A + B + C = π →  -- Sum of angles in a triangle
  Real.sqrt 3 * a = 2 * b * Real.sin B * Real.cos C + 2 * b * Real.sin C * Real.cos B →
  B = π/3 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_b_l3831_383143


namespace NUMINAMATH_CALUDE_average_and_subtraction_l3831_383123

theorem average_and_subtraction (y : ℝ) : 
  (15 + 25 + y) / 3 = 22 → y - 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_and_subtraction_l3831_383123


namespace NUMINAMATH_CALUDE_rectangle_area_with_circles_l3831_383112

/-- The area of a rectangle with specific circle arrangement -/
theorem rectangle_area_with_circles (d : ℝ) (w l : ℝ) : 
  d = 6 →                    -- diameter of each circle
  w = 3 * d →                -- width equals total diameter of three circles
  l = 2 * w →                -- length is twice the width
  w * l = 648 := by           -- area of the rectangle
  sorry

#check rectangle_area_with_circles

end NUMINAMATH_CALUDE_rectangle_area_with_circles_l3831_383112


namespace NUMINAMATH_CALUDE_smallest_coin_arrangement_l3831_383157

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The number of proper factors of a positive integer -/
def num_proper_factors (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 180 is the smallest positive integer satisfying the given conditions -/
theorem smallest_coin_arrangement :
  ∀ n : ℕ+, (num_factors n = 9 ∧ num_proper_factors n = 7) → n ≥ 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_coin_arrangement_l3831_383157


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_l3831_383199

/-- The definition of a repeating decimal 0.3333... -/
def repeating_third : ℚ := 1/3

/-- Proof that 1 - 0.3333... = 2/3 -/
theorem one_minus_repeating_third :
  1 - repeating_third = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_l3831_383199


namespace NUMINAMATH_CALUDE_lower_class_students_l3831_383191

/-- Proves that in a school with 120 total students, where the lower class has 36 more students than the upper class, the number of students in the lower class is 78. -/
theorem lower_class_students (total : ℕ) (upper : ℕ) (lower : ℕ) 
  (h1 : total = 120)
  (h2 : upper + lower = total)
  (h3 : lower = upper + 36) :
  lower = 78 := by
  sorry

end NUMINAMATH_CALUDE_lower_class_students_l3831_383191


namespace NUMINAMATH_CALUDE_sophie_goal_theorem_l3831_383194

def sophie_marks : List ℚ := [73/100, 82/100, 85/100]

def total_tests : ℕ := 5

def goal_average : ℚ := 80/100

def pair_D : List ℚ := [73/100, 83/100]
def pair_A : List ℚ := [79/100, 82/100]
def pair_B : List ℚ := [70/100, 91/100]
def pair_C : List ℚ := [76/100, 86/100]

theorem sophie_goal_theorem :
  (sophie_marks.sum + pair_D.sum) / total_tests < goal_average ∧
  (sophie_marks.sum + pair_A.sum) / total_tests ≥ goal_average ∧
  (sophie_marks.sum + pair_B.sum) / total_tests ≥ goal_average ∧
  (sophie_marks.sum + pair_C.sum) / total_tests ≥ goal_average :=
by sorry

end NUMINAMATH_CALUDE_sophie_goal_theorem_l3831_383194


namespace NUMINAMATH_CALUDE_problem_1_l3831_383146

theorem problem_1 (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) :
  a^2 + b^2 = 10 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3831_383146


namespace NUMINAMATH_CALUDE_stating_arithmetic_sequence_length_is_twelve_l3831_383182

/-- 
The number of terms in an arithmetic sequence with 
first term 3, last term 69, and common difference 6
-/
def arithmetic_sequence_length : ℕ := 
  (69 - 3) / 6 + 1

/-- 
Theorem stating that the arithmetic sequence length is 12
-/
theorem arithmetic_sequence_length_is_twelve : 
  arithmetic_sequence_length = 12 := by sorry

end NUMINAMATH_CALUDE_stating_arithmetic_sequence_length_is_twelve_l3831_383182


namespace NUMINAMATH_CALUDE_krishans_money_l3831_383169

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan's amount is 3774. -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 637 →
  krishan = 3774 := by
sorry

end NUMINAMATH_CALUDE_krishans_money_l3831_383169


namespace NUMINAMATH_CALUDE_birds_in_tree_l3831_383101

/-- The total number of birds in a tree after two groups join -/
def total_birds (initial : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  initial + group1 + group2

/-- Theorem stating that the total number of birds is 76 -/
theorem birds_in_tree : total_birds 24 37 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3831_383101


namespace NUMINAMATH_CALUDE_vector_addition_l3831_383122

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![3, 0]

-- State the theorem
theorem vector_addition :
  (a + b) = ![4, 2] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l3831_383122


namespace NUMINAMATH_CALUDE_circle_radius_condition_l3831_383117

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 4*x + y^2 + 2*y + c = 0 ↔ (x - 2)^2 + (y + 1)^2 = 5^2) → 
  c = -20 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l3831_383117


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3831_383186

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x ≥ 0 → 2*x + 1/(2*x + 1) ≥ 1) ∧
  (∃ x, 2*x + 1/(2*x + 1) ≥ 1 ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3831_383186


namespace NUMINAMATH_CALUDE_cherry_tree_leaves_l3831_383120

theorem cherry_tree_leaves (original_plan : ℕ) (actual_multiplier : ℕ) (leaves_per_tree : ℕ) : 
  original_plan = 7 → 
  actual_multiplier = 2 → 
  leaves_per_tree = 100 → 
  (original_plan * actual_multiplier * leaves_per_tree) = 1400 := by
sorry

end NUMINAMATH_CALUDE_cherry_tree_leaves_l3831_383120


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_l3831_383179

def Digits : Finset ℕ := {4, 5, 6, 7, 8, 9, 10}

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 
  a ≥ 100 ∧ a < 1000 ∧ 
  b ≥ 100 ∧ b < 1000 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10 \ {0}))) = 7 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits ∧ (d ∈ (Finset.range 10 \ {0}) ∨ d = 10)) ((Finset.range 10 \ {0}) ∪ {10}))) = 6

theorem smallest_sum_of_digits : 
  ∀ a b : ℕ, is_valid_pair a b → a + b ≥ 1245 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_l3831_383179


namespace NUMINAMATH_CALUDE_bulb_over_4000_hours_probability_l3831_383125

-- Define the probabilities
def prob_x : ℝ := 0.60  -- Probability of a bulb coming from factory X
def prob_y : ℝ := 1 - prob_x  -- Probability of a bulb coming from factory Y
def prob_x_over_4000 : ℝ := 0.59  -- Probability of factory X's bulb lasting over 4000 hours
def prob_y_over_4000 : ℝ := 0.65  -- Probability of factory Y's bulb lasting over 4000 hours

-- Define the theorem
theorem bulb_over_4000_hours_probability :
  prob_x * prob_x_over_4000 + prob_y * prob_y_over_4000 = 0.614 :=
by sorry

end NUMINAMATH_CALUDE_bulb_over_4000_hours_probability_l3831_383125


namespace NUMINAMATH_CALUDE_rotate_A_180_origin_l3831_383106

def rotate_180_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

theorem rotate_A_180_origin :
  let A : ℝ × ℝ := (1, 2)
  rotate_180_origin A = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_rotate_A_180_origin_l3831_383106


namespace NUMINAMATH_CALUDE_village_population_l3831_383197

theorem village_population (P : ℝ) 
  (h1 : 0.9 * P * 0.8 = 4500) : P = 6250 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3831_383197


namespace NUMINAMATH_CALUDE_sqrt_sum_diff_equals_fifteen_halves_l3831_383111

theorem sqrt_sum_diff_equals_fifteen_halves :
  Real.sqrt 9 + Real.sqrt 25 - Real.sqrt (1/4) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_diff_equals_fifteen_halves_l3831_383111


namespace NUMINAMATH_CALUDE_puppy_food_bags_puppy_food_bags_proof_l3831_383105

/-- Calculates the number of bags of special dog food needed for a puppy's first year --/
theorem puppy_food_bags : ℕ :=
  let days_in_year : ℕ := 365
  let first_period : ℕ := 60
  let second_period : ℕ := days_in_year - first_period
  let first_period_consumption : ℕ := first_period * 2
  let second_period_consumption : ℕ := second_period * 4
  let total_consumption : ℕ := first_period_consumption + second_period_consumption
  let ounces_per_pound : ℕ := 16
  let pounds_per_bag : ℕ := 5
  let ounces_per_bag : ℕ := ounces_per_pound * pounds_per_bag
  let bags_needed : ℕ := (total_consumption + ounces_per_bag - 1) / ounces_per_bag
  17

/-- Proof that the number of bags needed is 17 --/
theorem puppy_food_bags_proof : puppy_food_bags = 17 := by
  sorry

end NUMINAMATH_CALUDE_puppy_food_bags_puppy_food_bags_proof_l3831_383105


namespace NUMINAMATH_CALUDE_min_abs_z_l3831_383124

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 17) : 
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 56 / 17 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_l3831_383124


namespace NUMINAMATH_CALUDE_extreme_values_l3831_383136

/-- A quadratic function passing through four points with specific properties. -/
structure QuadraticFunction where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  y₄ : ℝ
  h₁ : y₂ < y₃
  h₂ : y₃ = y₄

/-- Theorem stating that y₁ is the smallest and y₃ is the largest among y₁, y₂, and y₃ -/
theorem extreme_values (f : QuadraticFunction) : 
  f.y₁ ≤ f.y₂ ∧ f.y₁ ≤ f.y₃ ∧ f.y₂ < f.y₃ := by
  sorry

#check extreme_values

end NUMINAMATH_CALUDE_extreme_values_l3831_383136


namespace NUMINAMATH_CALUDE_land_division_l3831_383137

theorem land_division (total_land : ℝ) (num_siblings : ℕ) (jose_share : ℝ) : 
  total_land = 20000 ∧ num_siblings = 4 → 
  jose_share = total_land / (num_siblings + 1) ∧ 
  jose_share = 4000 := by
sorry

end NUMINAMATH_CALUDE_land_division_l3831_383137


namespace NUMINAMATH_CALUDE_millie_bracelets_l3831_383188

theorem millie_bracelets (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
  (h1 : lost = 2) 
  (h2 : remaining = 7) 
  (h3 : initial = remaining + lost) : initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_millie_bracelets_l3831_383188


namespace NUMINAMATH_CALUDE_sum_f_2015_is_zero_l3831_383165

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

def sum_f (f : ℝ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_f f n + f (n + 1)

theorem sum_f_2015_is_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_sym : is_symmetric_about_one f) 
  (h_f_neg_one : f (-1) = 1) : 
  sum_f f 2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_2015_is_zero_l3831_383165


namespace NUMINAMATH_CALUDE_max_sum_of_factors_of_60_l3831_383115

theorem max_sum_of_factors_of_60 : 
  ∃ (a b : ℕ), a * b = 60 ∧ 
  ∀ (x y : ℕ), x * y = 60 → x + y ≤ a + b ∧ a + b = 61 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_of_60_l3831_383115


namespace NUMINAMATH_CALUDE_parabola_tangent_property_l3831_383185

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Given a parabola Γ: y² = 2px (p > 0) with focus F, and a point Q outside Γ (not on the x-axis),
    let tangents QA and QB intersect Γ at A and B respectively, and the y-axis at C and D.
    If M is the circumcenter of triangle QAB, then FM is tangent to the circumcircle of triangle FCD. -/
theorem parabola_tangent_property (Γ : Parabola) (F Q A B C D M : Point) :
  Q.x ≠ 0 →  -- Q is not on y-axis
  Q.y ≠ 0 →  -- Q is not on x-axis
  (∃ t : ℝ, A = Point.mk (t^2 / (2 * Γ.p)) t) →  -- A is on the parabola
  (∃ t : ℝ, B = Point.mk (t^2 / (2 * Γ.p)) t) →  -- B is on the parabola
  F = Point.mk (Γ.p / 2) 0 →  -- F is the focus
  C = Point.mk 0 C.y →  -- C is on y-axis
  D = Point.mk 0 D.y →  -- D is on y-axis
  (∃ l : Line, l.a * Q.x + l.b * Q.y + l.c = 0 ∧ l.a * A.x + l.b * A.y + l.c = 0) →  -- QA is a line
  (∃ l : Line, l.a * Q.x + l.b * Q.y + l.c = 0 ∧ l.a * B.x + l.b * B.y + l.c = 0) →  -- QB is a line
  (∀ P : Point, (P.x - M.x)^2 + (P.y - M.y)^2 = (A.x - M.x)^2 + (A.y - M.y)^2 →
               (P.x - M.x)^2 + (P.y - M.y)^2 = (B.x - M.x)^2 + (B.y - M.y)^2 →
               (P.x - M.x)^2 + (P.y - M.y)^2 = (Q.x - M.x)^2 + (Q.y - M.y)^2) →  -- M is circumcenter of QAB
  (∃ T : Point, ∃ r : ℝ,
    (T.x - F.x)^2 + (T.y - F.y)^2 = (C.x - F.x)^2 + (C.y - F.y)^2 ∧
    (T.x - F.x)^2 + (T.y - F.y)^2 = (D.x - F.x)^2 + (D.y - F.y)^2 ∧
    (M.x - F.x) * (T.x - F.x) + (M.y - F.y) * (T.y - F.y) = r^2) →  -- FM is tangent to circumcircle of FCD
  True :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_property_l3831_383185


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3831_383109

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - 2*x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 510 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3831_383109


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3831_383166

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The unique positive solution is 5/3
  use 5/3
  constructor
  · constructor
    · -- Prove 5/3 > 0
      sorry
    · -- Prove 3 * (5/3)^2 + 7 * (5/3) - 20 = 0
      sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3831_383166


namespace NUMINAMATH_CALUDE_latestPossibleTime_is_latest_l3831_383154

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60
  seconds_valid : seconds < 60

/-- Checks if a given time matches the visible digits pattern -/
def matchesVisibleDigits (t : Time) : Bool :=
  let h1 := t.hours / 10
  let h2 := t.hours % 10
  let m1 := t.minutes / 10
  let m2 := t.minutes % 10
  let s1 := t.seconds / 10
  let s2 := t.seconds % 10
  (h1 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h2 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h1 = 2 ∧ m2 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h1 = 2 ∧ m1 = 0 ∧ s2 = 2 ∧ h2 = 2) ∨
  (h1 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ m2 = 2)

/-- The latest possible time satisfying the conditions -/
def latestPossibleTime : Time := {
  hours := 23
  minutes := 50
  seconds := 22
  hours_valid := by simp
  minutes_valid := by simp
  seconds_valid := by simp
}

/-- Theorem stating that the latestPossibleTime is indeed the latest time satisfying the conditions -/
theorem latestPossibleTime_is_latest :
  matchesVisibleDigits latestPossibleTime ∧
  ∀ t : Time, matchesVisibleDigits t → t.hours * 3600 + t.minutes * 60 + t.seconds ≤
    latestPossibleTime.hours * 3600 + latestPossibleTime.minutes * 60 + latestPossibleTime.seconds :=
by
  sorry


end NUMINAMATH_CALUDE_latestPossibleTime_is_latest_l3831_383154
