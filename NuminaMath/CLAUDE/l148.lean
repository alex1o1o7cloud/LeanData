import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_range_l148_14896

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = 3 * k - 1) →
  (x + 2 * y = -2) →
  (x - y ≤ 5) →
  (k ≤ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_range_l148_14896


namespace NUMINAMATH_CALUDE_multiple_of_six_between_twelve_and_thirty_l148_14828

theorem multiple_of_six_between_twelve_and_thirty (x : ℕ) :
  (∃ k : ℕ, x = 6 * k) →
  x^2 > 144 →
  x < 30 →
  x = 18 ∨ x = 24 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_six_between_twelve_and_thirty_l148_14828


namespace NUMINAMATH_CALUDE_sum_of_coordinates_is_50_l148_14859

def is_valid_point (x y : ℝ) : Prop :=
  (y = 15 + 3 ∨ y = 15 - 3) ∧ 
  ((x - 5)^2 + (y - 15)^2 = 10^2)

theorem sum_of_coordinates_is_50 :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    is_valid_point x₁ y₁ ∧
    is_valid_point x₂ y₂ ∧
    is_valid_point x₃ y₃ ∧
    is_valid_point x₄ y₄ ∧
    x₁ + y₁ + x₂ + y₂ + x₃ + y₃ + x₄ + y₄ = 50 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_is_50_l148_14859


namespace NUMINAMATH_CALUDE_parallel_slope_relation_l148_14812

-- Define a structure for a line with a slope
structure Line where
  slope : ℝ

-- Define parallel relation for lines
def parallel (l₁ l₂ : Line) : Prop := sorry

theorem parallel_slope_relation :
  ∀ (l₁ l₂ : Line),
    (parallel l₁ l₂ → l₁.slope = l₂.slope) ∧
    ∃ (l₃ l₄ : Line), l₃.slope = l₄.slope ∧ ¬parallel l₃ l₄ := by
  sorry

end NUMINAMATH_CALUDE_parallel_slope_relation_l148_14812


namespace NUMINAMATH_CALUDE_special_polygon_interior_sum_special_polygon_exists_l148_14891

/-- A polygon where each interior angle is 7.5 times its corresponding exterior angle -/
structure SpecialPolygon where
  n : ℕ  -- number of sides
  interior_angle : ℝ  -- measure of each interior angle
  h_interior_exterior : interior_angle = 7.5 * (360 / n)  -- relation between interior and exterior angles

/-- The sum of interior angles of a SpecialPolygon is 2700° -/
theorem special_polygon_interior_sum (P : SpecialPolygon) : 
  P.n * P.interior_angle = 2700 := by
  sorry

/-- A SpecialPolygon with 17 sides exists -/
theorem special_polygon_exists : 
  ∃ P : SpecialPolygon, P.n = 17 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_interior_sum_special_polygon_exists_l148_14891


namespace NUMINAMATH_CALUDE_dividend_calculation_l148_14806

/-- Calculates the total dividend paid to a shareholder --/
def total_dividend (expected_earnings : ℚ) (actual_earnings : ℚ) (base_dividend_ratio : ℚ) 
  (additional_dividend_rate : ℚ) (additional_earnings_threshold : ℚ) (num_shares : ℕ) : ℚ :=
  let base_dividend := expected_earnings * base_dividend_ratio
  let earnings_difference := actual_earnings - expected_earnings
  let additional_dividend := 
    if earnings_difference > 0 
    then (earnings_difference / additional_earnings_threshold).floor * additional_dividend_rate
    else 0
  let total_dividend_per_share := base_dividend + additional_dividend
  total_dividend_per_share * num_shares

/-- Theorem stating the total dividend paid to a shareholder with given conditions --/
theorem dividend_calculation : 
  total_dividend 0.80 1.10 (1/2) 0.04 0.10 100 = 52 := by
  sorry

#eval total_dividend 0.80 1.10 (1/2) 0.04 0.10 100

end NUMINAMATH_CALUDE_dividend_calculation_l148_14806


namespace NUMINAMATH_CALUDE_point_on_curve_l148_14856

theorem point_on_curve : (3^2 : ℝ) - 3 * 10 + 2 * 10 + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_point_on_curve_l148_14856


namespace NUMINAMATH_CALUDE_part_I_part_II_l148_14892

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - 2*a| + a^2 - 4*a

-- Part I
theorem part_I :
  let f_neg_one (x : ℝ) := x * |x + 2| + 5
  ∃ (min max : ℝ), min = 2 ∧ max = 5 ∧
    (∀ x ∈ Set.Icc (-3) 0, f_neg_one x ≥ min ∧ f_neg_one x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-3) 0, f_neg_one x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-3) 0, f_neg_one x₂ = max) :=
sorry

-- Part II
theorem part_II :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  (∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 →
    (1 + Real.sqrt 2) / 2 < 1 / x₁ + 1 / x₂ + 1 / x₃) :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_l148_14892


namespace NUMINAMATH_CALUDE_twenty_numbers_arrangement_exists_l148_14899

theorem twenty_numbers_arrangement_exists : ∃ (a b : ℝ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) := by
  sorry

end NUMINAMATH_CALUDE_twenty_numbers_arrangement_exists_l148_14899


namespace NUMINAMATH_CALUDE_find_number_l148_14838

theorem find_number : ∃! x : ℝ, ((35 - x) * 2 + 12) / 8 = 9 := by sorry

end NUMINAMATH_CALUDE_find_number_l148_14838


namespace NUMINAMATH_CALUDE_sine_function_parameters_l148_14862

theorem sine_function_parameters
  (y : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, y x = a * Real.sin (b * x + c))
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : y (π / 6) = 3)
  (h5 : ∀ x, y (x + π) = y x) :
  a = 3 ∧ b = 2 ∧ c = π / 6 := by
sorry

end NUMINAMATH_CALUDE_sine_function_parameters_l148_14862


namespace NUMINAMATH_CALUDE_complement_of_union_l148_14878

open Set

def U : Set Nat := {1,2,3,4,5,6}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_of_union : (U \ (S ∪ T)) = {2,4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l148_14878


namespace NUMINAMATH_CALUDE_quadrilateral_area_product_is_square_quadrilateral_area_product_not_end_1988_l148_14848

/-- Represents a convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- The area of the first triangle -/
  area1 : ℕ
  /-- The area of the second triangle -/
  area2 : ℕ
  /-- The area of the third triangle -/
  area3 : ℕ
  /-- The area of the fourth triangle -/
  area4 : ℕ

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem quadrilateral_area_product_is_square (q : ConvexQuadrilateral) :
  ∃ (n : ℕ), q.area1 * q.area2 * q.area3 * q.area4 = n * n := by
  sorry

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals cannot end in 1988 -/
theorem quadrilateral_area_product_not_end_1988 (q : ConvexQuadrilateral) :
  ¬(q.area1 * q.area2 * q.area3 * q.area4 % 10000 = 1988) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_product_is_square_quadrilateral_area_product_not_end_1988_l148_14848


namespace NUMINAMATH_CALUDE_incorrect_product_calculation_l148_14837

theorem incorrect_product_calculation (x : ℕ) : 
  (53 * x - 35 * x = 540) → (53 * x = 1590) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_product_calculation_l148_14837


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l148_14895

/-- The surface area of a sphere circumscribing a right circular cone -/
theorem circumscribed_sphere_surface_area 
  (base_radius : ℝ) 
  (slant_height : ℝ) 
  (h1 : base_radius = Real.sqrt 3)
  (h2 : slant_height = 2) :
  ∃ (sphere_radius : ℝ), 
    4 * Real.pi * sphere_radius^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l148_14895


namespace NUMINAMATH_CALUDE_rectangle_mn_value_l148_14860

-- Define the rectangle ABCD
def Rectangle (AB BC : ℝ) : Prop :=
  AB > 0 ∧ BC > 0

-- Define the perimeter of the rectangle
def Perimeter (AB BC : ℝ) : ℝ :=
  2 * (AB + BC)

-- Define the area of the rectangle
def Area (AB BC : ℝ) : ℝ :=
  AB * BC

-- Define the quadratic equation
def QuadraticRoots (m n : ℝ) (x y : ℝ) : Prop :=
  x^2 + m*x + n = 0 ∧ y^2 + m*y + n = 0

-- State the theorem
theorem rectangle_mn_value (AB BC m n : ℝ) :
  Rectangle AB BC →
  Perimeter AB BC = 12 →
  Area AB BC = 5 →
  QuadraticRoots m n AB BC →
  m * n = -30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_mn_value_l148_14860


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l148_14840

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := sorry

-- State the theorem
theorem arithmetic_geometric_ratio (d : ℝ) :
  d ≠ 0 →
  (∃ r : ℝ, arithmetic_sequence d 3 = arithmetic_sequence d 1 * r ∧ 
            arithmetic_sequence d 4 = arithmetic_sequence d 3 * r) →
  (arithmetic_sequence d 1 + arithmetic_sequence d 5 + arithmetic_sequence d 17) / 
  (arithmetic_sequence d 2 + arithmetic_sequence d 6 + arithmetic_sequence d 18) = 8 / 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l148_14840


namespace NUMINAMATH_CALUDE_accident_calculation_highway_accidents_l148_14831

/-- Given an accident rate and total number of vehicles, calculate the number of vehicles involved in accidents --/
theorem accident_calculation (accident_rate : ℕ) (vehicles_per_set : ℕ) (total_vehicles : ℕ) :
  accident_rate > 0 →
  vehicles_per_set > 0 →
  total_vehicles ≥ vehicles_per_set →
  (total_vehicles / vehicles_per_set) * accident_rate = 
    (total_vehicles * accident_rate) / vehicles_per_set :=
by
  sorry

/-- Calculate the number of vehicles involved in accidents on a highway --/
theorem highway_accidents :
  let accident_rate := 80  -- vehicles involved in accidents per set
  let vehicles_per_set := 100000000  -- vehicles per set (100 million)
  let total_vehicles := 4000000000  -- total vehicles (4 billion)
  (total_vehicles / vehicles_per_set) * accident_rate = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_accident_calculation_highway_accidents_l148_14831


namespace NUMINAMATH_CALUDE_inequality_proof_l148_14889

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ 2)
  (h5 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l148_14889


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l148_14858

/-- Represents the typing service rates and manuscript details --/
structure ManuscriptTyping where
  total_pages : Nat
  initial_cost : Nat
  first_revision_cost : Nat
  second_revision_cost : Nat
  subsequent_revision_cost : Nat
  pages_revised_once : Nat
  pages_revised_twice : Nat
  pages_revised_thrice : Nat
  pages_revised_four_times : Nat
  pages_revised_five_times : Nat

/-- Calculates the total cost of typing and revising a manuscript --/
def total_typing_cost (m : ManuscriptTyping) : Nat :=
  m.total_pages * m.initial_cost +
  m.pages_revised_once * m.first_revision_cost +
  m.pages_revised_twice * (m.first_revision_cost + m.second_revision_cost) +
  m.pages_revised_thrice * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost) +
  m.pages_revised_four_times * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost * 2) +
  m.pages_revised_five_times * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost * 3)

/-- Theorem stating that the total cost for the given manuscript is $5750 --/
theorem manuscript_typing_cost :
  let m := ManuscriptTyping.mk 400 10 8 6 4 60 40 20 10 5
  total_typing_cost m = 5750 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l148_14858


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l148_14853

-- Define a normally distributed random variable
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
def P (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (x : normal_dist 4 σ) 
  (h : P {y : ℝ | y > 2} = 0.6) :
  P {y : ℝ | y > 6} = 0.4 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l148_14853


namespace NUMINAMATH_CALUDE_abc_fraction_theorem_l148_14817

theorem abc_fraction_theorem (a b c : ℕ+) :
  ∃ (n : ℕ), n > 0 ∧ n = (a * b * c + a * b + a) / (a * b * c + b * c + c) → n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_theorem_l148_14817


namespace NUMINAMATH_CALUDE_sum_of_non_visible_faces_l148_14866

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of all faces on a standard die -/
def sumOfDieFaces : ℕ := (List.range 6).map (· + 1) |>.sum

/-- The total number of dice -/
def numberOfDice : ℕ := 4

/-- The list of visible face values -/
def visibleFaces : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 5, 6]

/-- The sum of visible face values -/
def sumOfVisibleFaces : ℕ := visibleFaces.sum

/-- Theorem: The sum of non-visible face values is 52 -/
theorem sum_of_non_visible_faces :
  numberOfDice * sumOfDieFaces - sumOfVisibleFaces = 52 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_non_visible_faces_l148_14866


namespace NUMINAMATH_CALUDE_parallel_planes_properties_l148_14871

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the given condition
variable (a : Line) (α β : Plane)
variable (h : contains α a)

-- Theorem statement
theorem parallel_planes_properties :
  (∀ β, parallel_planes α β → parallel_line_plane a β) ∧
  (∀ β, ¬parallel_line_plane a β → ¬parallel_planes α β) ∧
  ¬(∀ β, parallel_line_plane a β → parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_properties_l148_14871


namespace NUMINAMATH_CALUDE_pages_per_night_l148_14850

/-- Given a book with 1200 pages read over 10.0 days, prove that 120 pages are read each night. -/
theorem pages_per_night (total_pages : ℕ) (reading_days : ℝ) :
  total_pages = 1200 → reading_days = 10.0 → (total_pages : ℝ) / reading_days = 120 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_night_l148_14850


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_two_l148_14884

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

-- State the theorem
theorem tangent_line_at_negative_two (b : ℝ) :
  b = -6 →
  let x₀ := -2
  let y₀ := f b x₀
  let m := (3 * x₀^2 - 12)  -- Derivative at x₀
  ∀ x, y₀ + m * (x - x₀) = 10 := by
sorry

-- Note: The actual proof is omitted as per instructions

end NUMINAMATH_CALUDE_tangent_line_at_negative_two_l148_14884


namespace NUMINAMATH_CALUDE_loan_amount_proof_l148_14809

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- The loan satisfies the given conditions -/
def loan_conditions (loan : SimpleLoan) : Prop :=
  loan.rate = 0.06 ∧
  loan.time = loan.rate ∧
  loan.interest = 432 ∧
  loan.interest = loan.principal * loan.rate * loan.time

theorem loan_amount_proof (loan : SimpleLoan) 
  (h : loan_conditions loan) : loan.principal = 1200 := by
  sorry

#check loan_amount_proof

end NUMINAMATH_CALUDE_loan_amount_proof_l148_14809


namespace NUMINAMATH_CALUDE_complex_number_modulus_l148_14864

theorem complex_number_modulus (z : ℂ) : z = 1 / (Complex.I - 1) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l148_14864


namespace NUMINAMATH_CALUDE_max_students_distribution_l148_14800

theorem max_students_distribution (pens toys : ℕ) (h1 : pens = 451) (h2 : toys = 410) :
  Nat.gcd pens toys = 41 :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l148_14800


namespace NUMINAMATH_CALUDE_problem_solution_l148_14877

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^2 / y + y^2 / x + y = 95 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l148_14877


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l148_14849

/-- Given two points A and B symmetric about the y-axis, prove that m-n = -4 -/
theorem symmetric_points_difference (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (m - 2, 3) ∧ 
    B = (4, n + 1) ∧ 
    (A.1 = -B.1) ∧  -- x-coordinates are opposite
    (A.2 = B.2))    -- y-coordinates are equal
  → m - n = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l148_14849


namespace NUMINAMATH_CALUDE_intersection_sum_modulo13_l148_14898

theorem intersection_sum_modulo13 : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 13 ∧ 
  (∃ (y : ℤ), y ≡ 3*x + 4 [ZMOD 13] ∧ y ≡ 8*x + 9 [ZMOD 13]) ∧
  (∀ (x' : ℤ), 0 ≤ x' ∧ x' < 13 → 
    (∃ (y' : ℤ), y' ≡ 3*x' + 4 [ZMOD 13] ∧ y' ≡ 8*x' + 9 [ZMOD 13]) → 
    x' = x) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_modulo13_l148_14898


namespace NUMINAMATH_CALUDE_yoongi_age_proof_l148_14854

/-- Yoongi's age -/
def yoongi_age : ℕ := 8

/-- Hoseok's age -/
def hoseok_age : ℕ := yoongi_age + 2

/-- The sum of Yoongi's and Hoseok's ages -/
def total_age : ℕ := yoongi_age + hoseok_age

theorem yoongi_age_proof : yoongi_age = 8 :=
  by
    have h1 : hoseok_age = yoongi_age + 2 := rfl
    have h2 : total_age = 18 := rfl
    sorry

end NUMINAMATH_CALUDE_yoongi_age_proof_l148_14854


namespace NUMINAMATH_CALUDE_problem_solving_probability_l148_14883

theorem problem_solving_probability (prob_a prob_b : ℝ) 
  (h_a : prob_a = 1/2)
  (h_b : prob_b = 1/3) :
  (1 - prob_a) * (1 - prob_b) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l148_14883


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l148_14832

theorem trig_expression_equals_one :
  (Real.sqrt 3 * Real.sin (20 * π / 180) + Real.sin (70 * π / 180)) /
  Real.sqrt (2 - 2 * Real.cos (100 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l148_14832


namespace NUMINAMATH_CALUDE_max_bowls_proof_l148_14829

/-- Represents the number of clusters in a spoonful for the nth bowl -/
def clusters_per_spoon (n : ℕ) : ℕ := 3 + n

/-- Represents the number of spoonfuls in the nth bowl -/
def spoonfuls_per_bowl (n : ℕ) : ℕ := 27 - 2 * n

/-- Calculates the total clusters used up to and including the nth bowl -/
def total_clusters (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + clusters_per_spoon (i + 1) * spoonfuls_per_bowl (i + 1)) 0

/-- The maximum number of bowls that can be made from 500 clusters -/
def max_bowls : ℕ := 4

theorem max_bowls_proof : 
  total_clusters max_bowls ≤ 500 ∧ 
  total_clusters (max_bowls + 1) > 500 := by
  sorry

#eval max_bowls

end NUMINAMATH_CALUDE_max_bowls_proof_l148_14829


namespace NUMINAMATH_CALUDE_football_player_goals_l148_14841

/-- Proves that a football player scored 2 goals in their fifth match -/
theorem football_player_goals (total_matches : ℕ) (total_goals : ℕ) (average_increase : ℚ) : 
  total_matches = 5 → 
  total_goals = 4 → 
  average_increase = 3/10 → 
  (total_goals : ℚ) / total_matches = 
    ((total_goals : ℚ) - (total_goals - goals_in_fifth_match)) / (total_matches - 1) + average_increase →
  goals_in_fifth_match = 2 :=
by
  sorry

#check football_player_goals

end NUMINAMATH_CALUDE_football_player_goals_l148_14841


namespace NUMINAMATH_CALUDE_pizza_slices_l148_14835

-- Define the number of slices in each pizza
def slices_per_pizza : ℕ := sorry

-- Define the total number of pizzas
def total_pizzas : ℕ := 2

-- Define the fractions eaten by each person
def bob_fraction : ℚ := 1/2
def tom_fraction : ℚ := 1/3
def sally_fraction : ℚ := 1/6
def jerry_fraction : ℚ := 1/4

-- Define the number of slices left over
def slices_left : ℕ := 9

theorem pizza_slices : 
  slices_per_pizza = 12 ∧
  (bob_fraction + tom_fraction + sally_fraction + jerry_fraction) * slices_per_pizza * total_pizzas = 
    slices_per_pizza * total_pizzas - slices_left :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_l148_14835


namespace NUMINAMATH_CALUDE_final_collection_is_55_l148_14830

def museum_donations (initial_collection : ℕ) : ℕ :=
  let guggenheim_donation := 51
  let metropolitan_donation := 2 * guggenheim_donation
  let damaged_sets := 20
  let after_damage := initial_collection - guggenheim_donation - metropolitan_donation - damaged_sets
  let louvre_donation := after_damage / 2
  let after_louvre := after_damage - louvre_donation
  let british_donation := (2 * after_louvre) / 3
  after_louvre - british_donation

theorem final_collection_is_55 :
  museum_donations 500 = 55 := by
  sorry

end NUMINAMATH_CALUDE_final_collection_is_55_l148_14830


namespace NUMINAMATH_CALUDE_father_and_xiaolin_ages_l148_14885

theorem father_and_xiaolin_ages :
  ∀ (f x : ℕ),
  f = 11 * x →
  f + 7 = 4 * (x + 7) →
  f = 33 ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_father_and_xiaolin_ages_l148_14885


namespace NUMINAMATH_CALUDE_power_equality_l148_14815

theorem power_equality (x : ℝ) : (1/4 : ℝ) * (2^32) = 4^x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l148_14815


namespace NUMINAMATH_CALUDE_percentage_equation_l148_14803

theorem percentage_equation (x : ℝ) : (65 / 100 * x = 20 / 100 * 617.50) → x = 190 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l148_14803


namespace NUMINAMATH_CALUDE_sticks_per_pot_l148_14876

/-- Given:
  * There are 466 pots
  * Each pot has 53 flowers
  * There are 109044 flowers and sticks in total
  Prove that there are 181 sticks in each pot -/
theorem sticks_per_pot (num_pots : ℕ) (flowers_per_pot : ℕ) (total_items : ℕ) :
  num_pots = 466 →
  flowers_per_pot = 53 →
  total_items = 109044 →
  (total_items - num_pots * flowers_per_pot) / num_pots = 181 := by
  sorry

#eval (109044 - 466 * 53) / 466  -- Should output 181

end NUMINAMATH_CALUDE_sticks_per_pot_l148_14876


namespace NUMINAMATH_CALUDE_expand_and_complete_square_l148_14890

theorem expand_and_complete_square (x : ℝ) : 
  -2 * (x - 3) * (x + 1/2) = -2 * (x - 5/4)^2 + 49/8 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_complete_square_l148_14890


namespace NUMINAMATH_CALUDE_cone_base_radius_l148_14887

/-- Given a cone with surface area 15π cm² and lateral surface that unfolds into a semicircle,
    prove that the radius of its base is √5 cm. -/
theorem cone_base_radius (surface_area : ℝ) (r : ℝ) :
  surface_area = 15 * Real.pi ∧
  (∃ l : ℝ, π * l = 2 * π * r ∧ surface_area = π * r^2 + π * r * l) →
  r = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_cone_base_radius_l148_14887


namespace NUMINAMATH_CALUDE_min_toothpicks_theorem_l148_14801

/-- Represents a triangular grid made of toothpicks -/
structure ToothpickGrid where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (grid : ToothpickGrid) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_theorem (grid : ToothpickGrid) 
  (h1 : grid.total_toothpicks = 40)
  (h2 : grid.upward_triangles = 10)
  (h3 : grid.downward_triangles = 15) : 
  min_toothpicks_to_remove grid = 10 := by sorry

end NUMINAMATH_CALUDE_min_toothpicks_theorem_l148_14801


namespace NUMINAMATH_CALUDE_division_problem_l148_14846

theorem division_problem (dividend quotient divisor remainder multiple : ℕ) :
  remainder = 6 →
  dividend = 86 →
  divisor = 5 * quotient →
  divisor = multiple * remainder + 2 →
  dividend = divisor * quotient + remainder →
  multiple = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l148_14846


namespace NUMINAMATH_CALUDE_store_sales_problem_l148_14851

theorem store_sales_problem (d : ℕ) : 
  (86 + 50 * d) / (d + 1) = 53 → d = 11 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_problem_l148_14851


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l148_14836

theorem diophantine_equation_solutions :
  {(x, y) : ℕ × ℕ | 3 * x + 2 * y = 21 ∧ x > 0 ∧ y > 0} =
  {(5, 3), (3, 6), (1, 9)} := by
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l148_14836


namespace NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l148_14847

theorem smallest_integers_difference : ℕ → Prop :=
  fun n =>
    (∃ a b : ℕ,
      (a > 1 ∧ b > 1 ∧ a < b) ∧
      (∀ k : ℕ, 3 ≤ k → k ≤ 12 → a % k = 1 ∧ b % k = 1) ∧
      (∀ x : ℕ, x > 1 ∧ x < a → ∃ k : ℕ, 3 ≤ k ∧ k ≤ 12 ∧ x % k ≠ 1) ∧
      (b - a = n)) →
    n = 13860

theorem smallest_integers_difference_exists : ∃ n : ℕ, smallest_integers_difference n :=
  sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l148_14847


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_and_intersecting_l148_14833

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- A point lies on a parametric line if there exists a t satisfying both equations -/
def lies_on (p : ℝ × ℝ) (l : ParametricLine) : Prop :=
  ∃ t : ℝ, l.x t = p.1 ∧ l.y t = p.2

/-- The main theorem -/
theorem vector_to_line_parallel_and_intersecting :
  let l : ParametricLine := { x := λ t => 5 * t + 1, y := λ t => 2 * t + 1 }
  let v : ℝ × ℝ := (12.5, 5)
  let w : ℝ × ℝ := (5, 2)
  parallel v w ∧ lies_on v l := by sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_and_intersecting_l148_14833


namespace NUMINAMATH_CALUDE_paula_karl_age_sum_l148_14802

theorem paula_karl_age_sum : ∀ (P K : ℕ),
  (P - 5 = 3 * (K - 5)) →
  (P + 6 = 2 * (K + 6)) →
  P + K = 54 := by
  sorry

end NUMINAMATH_CALUDE_paula_karl_age_sum_l148_14802


namespace NUMINAMATH_CALUDE_quadratic_function_range_l148_14880

/-- Given a quadratic function f(x) = ax^2 - 2ax + c where a and c are real numbers,
    if f(2017) < f(-2016), then the set of real numbers m that satisfies f(m) ≤ f(0)
    is equal to the closed interval [0, 2]. -/
theorem quadratic_function_range (a c : ℝ) :
  let f := fun x : ℝ => a * x^2 - 2 * a * x + c
  (f 2017 < f (-2016)) →
  {m : ℝ | f m ≤ f 0} = Set.Icc 0 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l148_14880


namespace NUMINAMATH_CALUDE_intersection_distance_l148_14857

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the directrix of C
def directrix (x : ℝ) : Prop :=
  x = -2

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    directrix A.1 ∧
    directrix B.1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l148_14857


namespace NUMINAMATH_CALUDE_vector_magnitude_solution_l148_14882

/-- Given a vector a = (5, x) with magnitude 9, prove that x = 2√14 or x = -2√14 -/
theorem vector_magnitude_solution (x : ℝ) : 
  let a : ℝ × ℝ := (5, x)
  (‖a‖ = 9) → (x = 2 * Real.sqrt 14 ∨ x = -2 * Real.sqrt 14) := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_solution_l148_14882


namespace NUMINAMATH_CALUDE_fraction_equality_implies_k_l148_14839

theorem fraction_equality_implies_k (x y z k : ℝ) :
  (9 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 15 / (z - y)) →
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_k_l148_14839


namespace NUMINAMATH_CALUDE_multiplication_result_l148_14807

theorem multiplication_result : 2.68 * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l148_14807


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l148_14875

/-- The number of ways to arrange four people from two teachers and four students,
    where the teachers must be selected and adjacent. -/
def arrangement_count : ℕ := 72

/-- The number of teachers -/
def teacher_count : ℕ := 2

/-- The number of students -/
def student_count : ℕ := 4

/-- The total number of people to be selected -/
def selection_count : ℕ := 4

theorem photo_arrangement_count :
  arrangement_count = 
    (teacher_count.factorial) *              -- Ways to arrange teachers
    (student_count.choose (selection_count - teacher_count)) * -- Ways to choose students
    ((selection_count - 1).factorial) :=     -- Ways to arrange teachers bundle and students
  by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l148_14875


namespace NUMINAMATH_CALUDE_xy_difference_l148_14869

theorem xy_difference (x y : ℝ) (h : 10 * x^2 - 16 * x * y + 8 * y^2 + 6 * x - 4 * y + 1 = 0) :
  x - y = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_xy_difference_l148_14869


namespace NUMINAMATH_CALUDE_always_odd_l148_14808

theorem always_odd (n : ℤ) : ∃ k : ℤ, n^2 + n + 5 = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l148_14808


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_fourth_power_l148_14810

theorem nearest_integer_to_3_plus_sqrt5_fourth_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^4 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_fourth_power_l148_14810


namespace NUMINAMATH_CALUDE_correct_average_marks_l148_14897

/-- Calculates the correct average marks for a class given the reported average, 
    number of students, and corrections for three students' marks. -/
def correctAverageMarks (reportedAverage : ℚ) (numStudents : ℕ) 
    (wrongMark1 wrongMark2 wrongMark3 : ℚ) 
    (correctMark1 correctMark2 correctMark3 : ℚ) : ℚ :=
  let incorrectTotal := reportedAverage * numStudents
  let wronglyNotedMarks := wrongMark1 + wrongMark2 + wrongMark3
  let correctMarks := correctMark1 + correctMark2 + correctMark3
  let correctTotal := incorrectTotal - wronglyNotedMarks + correctMarks
  correctTotal / numStudents

/-- The correct average marks for the class are 63.125 -/
theorem correct_average_marks :
  correctAverageMarks 65 40 100 85 15 20 50 55 = 63.125 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l148_14897


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l148_14823

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  a / b = 2 / 5 →    -- Given ratio of a to b
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- r and s are segments of c
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l148_14823


namespace NUMINAMATH_CALUDE_container_volume_ratio_l148_14852

theorem container_volume_ratio :
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (3 / 4 : ℚ) * volume_container1 = (5 / 8 : ℚ) * volume_container2 →
  volume_container1 / volume_container2 = (5 / 6 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l148_14852


namespace NUMINAMATH_CALUDE_no_two_digit_number_exists_l148_14894

theorem no_two_digit_number_exists : ¬∃ (n : ℕ), 
  (10 ≤ n ∧ n < 100) ∧ 
  (∃ (d₁ d₂ : ℕ), 
    d₁ < 10 ∧ d₂ < 10 ∧
    n = 10 * d₁ + d₂ ∧
    n = 2 * (d₁^2 + d₂^2) + 6 ∧
    n = 4 * (d₁ * d₂) + 6) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_number_exists_l148_14894


namespace NUMINAMATH_CALUDE_inverse_sum_mod_23_l148_14844

theorem inverse_sum_mod_23 : 
  (((13⁻¹ : ZMod 23) + (17⁻¹ : ZMod 23) + (19⁻¹ : ZMod 23))⁻¹ : ZMod 23) = 8 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_23_l148_14844


namespace NUMINAMATH_CALUDE_problem_statement_l148_14834

theorem problem_statement :
  let p := ∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x
  let q := ∃ x : ℝ, x^2 = 2 - x
  (¬p ∧ q) → (∃ x : ℝ, x^2 = 2 - x ∧ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l148_14834


namespace NUMINAMATH_CALUDE_expression_evaluation_l148_14818

theorem expression_evaluation (y : ℝ) (h : y = -3) : 
  (5 + y * (4 + y) - 4^2) / (y - 2 + y^2) = -3.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l148_14818


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l148_14813

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_geometric_sequence (a b d : ℕ) :
  (∃ r : ℚ, r > 1 ∧ fib b = r * fib a ∧ fib d = r * fib b) →  -- Fₐ, Fᵦ, Fᵈ form an increasing geometric sequence
  a + b + d = 3000 →  -- Sum of indices is 3000
  b = a + 2 →  -- b - a = 2
  d = b + 2 →  -- d = b + 2
  a = 998 := by  -- Conclusion: a = 998
sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l148_14813


namespace NUMINAMATH_CALUDE_pet_shop_hamsters_l148_14872

theorem pet_shop_hamsters (total : ℕ) (kittens : ℕ) (birds : ℕ) 
  (h1 : total = 77)
  (h2 : kittens = 32)
  (h3 : birds = 30)
  : total - kittens - birds = 15 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_hamsters_l148_14872


namespace NUMINAMATH_CALUDE_fish_per_black_duck_is_ten_l148_14893

/-- Represents the number of fish per duck for each duck color -/
structure FishPerDuck where
  white : ℕ
  multicolor : ℕ

/-- Represents the number of ducks for each color -/
structure DuckCounts where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- Calculates the number of fish per black duck -/
def fishPerBlackDuck (fpd : FishPerDuck) (dc : DuckCounts) (totalFish : ℕ) : ℚ :=
  let fishForWhite := fpd.white * dc.white
  let fishForMulticolor := fpd.multicolor * dc.multicolor
  let fishForBlack := totalFish - fishForWhite - fishForMulticolor
  (fishForBlack : ℚ) / dc.black

theorem fish_per_black_duck_is_ten :
  let fpd : FishPerDuck := { white := 5, multicolor := 12 }
  let dc : DuckCounts := { white := 3, black := 7, multicolor := 6 }
  let totalFish : ℕ := 157
  fishPerBlackDuck fpd dc totalFish = 10 := by
  sorry

end NUMINAMATH_CALUDE_fish_per_black_duck_is_ten_l148_14893


namespace NUMINAMATH_CALUDE_max_factors_of_b_power_n_l148_14819

def is_prime (p : ℕ) : Prop := sorry

-- Function to count factors of a number
def count_factors (n : ℕ) : ℕ := sorry

-- Function to check if a number is the product of exactly two distinct primes less than 15
def is_product_of_two_primes_less_than_15 (b : ℕ) : Prop := sorry

theorem max_factors_of_b_power_n :
  ∃ (b n : ℕ),
    b ≤ 15 ∧
    n ≤ 15 ∧
    is_product_of_two_primes_less_than_15 b ∧
    count_factors (b^n) = 256 ∧
    ∀ (b' n' : ℕ),
      b' ≤ 15 →
      n' ≤ 15 →
      is_product_of_two_primes_less_than_15 b' →
      count_factors (b'^n') ≤ 256 := by
  sorry

end NUMINAMATH_CALUDE_max_factors_of_b_power_n_l148_14819


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l148_14824

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l148_14824


namespace NUMINAMATH_CALUDE_prob_king_or_ace_eq_two_thirteenth_l148_14870

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image (·.1)).card = 13)
  (suit_count : (cards.image (·.2)).card = 4)
  (unique_pairs : ∀ r s, (r, s) ∈ cards → r ∈ Finset.range 13 ∧ s ∈ Finset.range 4)

/-- The probability of drawing a King or an Ace from the top of a shuffled deck -/
def prob_king_or_ace (d : Deck) : ℚ :=
  (d.cards.filter (λ p => p.1 = 0 ∨ p.1 = 12)).card / d.cards.card

/-- Theorem: The probability of drawing a King or an Ace is 2/13 -/
theorem prob_king_or_ace_eq_two_thirteenth (d : Deck) : 
  prob_king_or_ace d = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_or_ace_eq_two_thirteenth_l148_14870


namespace NUMINAMATH_CALUDE_root_line_discriminant_intersection_l148_14865

/-- The discriminant curve in the pq-plane -/
def discriminant_curve (p q : ℝ) : Prop := 4 * p^3 + 27 * q^2 = 0

/-- The root line for a given value of a -/
def root_line (a p q : ℝ) : Prop := a * p + q + a^3 = 0

/-- The intersection points of the root line and the discriminant curve -/
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {(p, q) | discriminant_curve p q ∧ root_line a p q}

theorem root_line_discriminant_intersection (a : ℝ) :
  (a ≠ 0 → intersection_points a = {(-3 * a^2, 2 * a^3), (-3 * a^2 / 4, -a^3 / 4)}) ∧
  (a = 0 → intersection_points a = {(0, 0)}) := by
  sorry

end NUMINAMATH_CALUDE_root_line_discriminant_intersection_l148_14865


namespace NUMINAMATH_CALUDE_collinear_points_right_triangle_l148_14822

/-- Given that point O is the origin, this function defines vectors OA, OB, and OC -/
def vectors (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((3, -4), (6, -3), (5 - m, -3 - m))

/-- Theorem stating that if A, B, and C are collinear, then m = 1/2 -/
theorem collinear_points (m : ℝ) :
  let (oa, ob, oc) := vectors m
  (∃ (k : ℝ), (ob.1 - oa.1, ob.2 - oa.2) = k • (oc.1 - oa.1, oc.2 - oa.2)) →
  m = 1/2 := by sorry

/-- Theorem stating that if ABC is a right triangle with A as the right angle, then m = 7/4 -/
theorem right_triangle (m : ℝ) :
  let (oa, ob, oc) := vectors m
  let ab := (ob.1 - oa.1, ob.2 - oa.2)
  let ac := (oc.1 - oa.1, oc.2 - oa.2)
  (ab.1 * ac.1 + ab.2 * ac.2 = 0) →
  m = 7/4 := by sorry

end NUMINAMATH_CALUDE_collinear_points_right_triangle_l148_14822


namespace NUMINAMATH_CALUDE_integer_solution_equation_l148_14886

theorem integer_solution_equation (x y : ℤ) : 
  9 * x + 2 = y * (y + 1) ↔ ∃ k : ℤ, x = k * (k + 1) ∧ y = 3 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_equation_l148_14886


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l148_14821

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- State the theorem
theorem quadratic_function_properties :
  -- f is a quadratic function
  (∃ (a b c : ℝ), ∀ x, f x = a*x^2 + b*x + c) ∧
  -- f'(x) = 2x + 2
  (∀ x, deriv f x = 2*x + 2) ∧
  -- f(x) = 0 has two equal real roots
  (∃! x : ℝ, f x = 0) →
  -- 1. f(x) = x^2 + 2x + 1
  (∀ x, f x = x^2 + 2*x + 1) ∧
  -- 2. The area enclosed by f(x) and the coordinate axes is 1/3
  (∫ x in (-1)..0, f x = 1/3) ∧
  -- 3. The value of t that divides the enclosed area into two equal parts is 1 - 1/32
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧
    ∫ x in (-1)..(-t), f x = ∫ x in (-t)..0, f x ∧
    t = 1 - 1/(2^5)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l148_14821


namespace NUMINAMATH_CALUDE_factorization_x9_minus_512_l148_14881

theorem factorization_x9_minus_512 (x : ℝ) : 
  x^9 - 512 = (x - 2) * (x^2 + 2*x + 4) * (x^6 + 2*x^3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x9_minus_512_l148_14881


namespace NUMINAMATH_CALUDE_sqrt_300_simplification_l148_14843

theorem sqrt_300_simplification : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_simplification_l148_14843


namespace NUMINAMATH_CALUDE_comic_collection_equality_l148_14842

/-- Kymbrea's initial comic book collection --/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book addition rate --/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection --/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book addition rate --/
def lashawn_rate : ℕ := 7

/-- The number of months after which LaShawn's collection will be greater than or equal to Kymbrea's --/
def months_until_equal : ℕ := 8

theorem comic_collection_equality :
  ∀ m : ℕ, m < months_until_equal →
    (lashawn_initial + lashawn_rate * m < kymbrea_initial + kymbrea_rate * m) ∧
    (lashawn_initial + lashawn_rate * months_until_equal ≥ kymbrea_initial + kymbrea_rate * months_until_equal) :=
by sorry

end NUMINAMATH_CALUDE_comic_collection_equality_l148_14842


namespace NUMINAMATH_CALUDE_repeating_decimal_56_l148_14804

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_56 :
  RepeatingDecimal 5 6 = 56 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_56_l148_14804


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l148_14863

/-- A quadrilateral inscribed in a circle with perpendicular diagonals -/
structure InscribedQuadrilateral where
  R : ℝ  -- Radius of the circumscribed circle
  d1 : ℝ  -- Distance of first diagonal from circle center
  d2 : ℝ  -- Distance of second diagonal from circle center

/-- The sides of the quadrilateral -/
def quadrilateralSides (q : InscribedQuadrilateral) : Set ℝ :=
  {x | ∃ (n : ℤ), x = 4 * (2 * Real.sqrt 13 + n) ∨ x = 4 * (8 + 2 * n * Real.sqrt 13)}

/-- Theorem stating the sides of the quadrilateral given specific conditions -/
theorem inscribed_quadrilateral_sides 
  (q : InscribedQuadrilateral) 
  (h1 : q.R = 17) 
  (h2 : q.d1 = 8) 
  (h3 : q.d2 = 9) : 
  ∀ s, s ∈ quadrilateralSides q ↔ 
    (s = 4 * (2 * Real.sqrt 13 - 1) ∨ 
     s = 4 * (2 * Real.sqrt 13 + 1) ∨ 
     s = 4 * (8 - 2 * Real.sqrt 13) ∨ 
     s = 4 * (8 + 2 * Real.sqrt 13)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l148_14863


namespace NUMINAMATH_CALUDE_race_head_start_l148_14861

theorem race_head_start (va vb : ℝ) (h : va = 20/15 * vb) :
  let x : ℝ := 1/4
  ∀ L : ℝ, L > 0 → L / va = (L - x * L) / vb :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l148_14861


namespace NUMINAMATH_CALUDE_weights_division_l148_14805

theorem weights_division (n : ℕ) : 
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔ 
  (n > 3 ∧ (n % 3 = 0 ∨ n % 3 = 2)) :=
sorry

end NUMINAMATH_CALUDE_weights_division_l148_14805


namespace NUMINAMATH_CALUDE_circles_externally_tangent_m_value_l148_14827

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + m + 6 = 0

-- Define external tangency
def externally_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  ∀ (x' y' : ℝ), (C1 x' y' ∧ C2 x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_externally_tangent_m_value :
  externally_tangent circle_C1 (circle_C2 · · 26) :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_m_value_l148_14827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l148_14826

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 6 + a 8 + a 10 + a 12 = 90) →
  (a 10 - (1/3) * a 14 = 12) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l148_14826


namespace NUMINAMATH_CALUDE_nine_to_ten_div_eightyone_to_four_equals_eightyone_l148_14845

theorem nine_to_ten_div_eightyone_to_four_equals_eightyone :
  9^10 / 81^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_nine_to_ten_div_eightyone_to_four_equals_eightyone_l148_14845


namespace NUMINAMATH_CALUDE_flower_pots_total_cost_l148_14814

/-- The number of flower pots -/
def num_pots : ℕ := 6

/-- The price difference between consecutive pots -/
def price_diff : ℚ := 25 / 100

/-- The price of the largest pot -/
def largest_pot_price : ℚ := 1925 / 1000

/-- Calculate the total cost of all flower pots -/
def total_cost : ℚ :=
  let smallest_pot_price := largest_pot_price - (num_pots - 1 : ℕ) * price_diff
  (num_pots : ℚ) * smallest_pot_price + (num_pots - 1 : ℕ) * (num_pots : ℚ) * price_diff / 2

theorem flower_pots_total_cost :
  total_cost = 780 / 100 := by sorry

end NUMINAMATH_CALUDE_flower_pots_total_cost_l148_14814


namespace NUMINAMATH_CALUDE_max_profit_at_8_l148_14855

noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then (1/2) * x^2 + 4*x
  else if x ≥ 8 then 11*x + 49/x - 35
  else 0

noncomputable def P (x : ℝ) : ℝ :=
  10*x - C x - 5

theorem max_profit_at_8 :
  ∀ x > 0, P x ≤ P 8 ∧ P 8 = 127/8 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_8_l148_14855


namespace NUMINAMATH_CALUDE_grape_purchase_amount_l148_14874

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The number of kg of mangoes purchased -/
def mango_kg : ℕ := 9

/-- The total amount paid -/
def total_paid : ℕ := 1195

/-- The number of kg of grapes purchased -/
def grape_kg : ℕ := (total_paid - mango_price * mango_kg) / grape_price

theorem grape_purchase_amount : grape_kg = 10 := by
  sorry

end NUMINAMATH_CALUDE_grape_purchase_amount_l148_14874


namespace NUMINAMATH_CALUDE_remainder_7623_div_11_l148_14888

theorem remainder_7623_div_11 : 7623 % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_7623_div_11_l148_14888


namespace NUMINAMATH_CALUDE_total_distance_two_parts_l148_14867

/-- Calculates the total distance traveled by a car with varying speeds -/
theorem total_distance_two_parts (v1 v2 t1 t2 D1 D2 : ℝ) :
  D1 = v1 * t1 →
  D2 = v2 * t2 →
  let D := D1 + D2
  D = v1 * t1 + v2 * t2 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_two_parts_l148_14867


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l148_14873

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l148_14873


namespace NUMINAMATH_CALUDE_equation_solutions_l148_14816

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x + 3) = 5 * x ∧ x = 2) ∧
  (∃ x : ℝ, (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6 ∧ x = -9.2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l148_14816


namespace NUMINAMATH_CALUDE_tank_capacity_l148_14868

/-- Represents a water tank with a given capacity --/
structure WaterTank where
  capacity : ℚ
  initial_fill : ℚ
  final_fill : ℚ
  added_water : ℚ

/-- Theorem stating the capacity of the tank given the conditions --/
theorem tank_capacity (tank : WaterTank)
  (h1 : tank.initial_fill = 3 / 4)
  (h2 : tank.final_fill = 7 / 8)
  (h3 : tank.added_water = 5)
  (h4 : tank.initial_fill * tank.capacity + tank.added_water = tank.final_fill * tank.capacity) :
  tank.capacity = 40 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l148_14868


namespace NUMINAMATH_CALUDE_number_problem_l148_14820

theorem number_problem (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 
  (40/100) * N = 120 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l148_14820


namespace NUMINAMATH_CALUDE_prob_even_sum_is_five_ninths_l148_14879

/-- Represents a spinner with a list of numbers -/
def Spinner := List Nat

/-- The spinner X with numbers 1, 4, 5 -/
def X : Spinner := [1, 4, 5]

/-- The spinner Y with numbers 1, 2, 3 -/
def Y : Spinner := [1, 2, 3]

/-- The spinner Z with numbers 2, 4, 6 -/
def Z : Spinner := [2, 4, 6]

/-- Predicate to check if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- Function to calculate the probability of an even sum when spinning X, Y, and Z -/
def probEvenSum (x y z : Spinner) : Rat :=
  let totalOutcomes := (x.length * y.length * z.length : Nat)
  let evenSumOutcomes := x.countP (fun a => 
    y.countP (fun b => 
      z.countP (fun c => 
        isEven (a + b + c)) = z.length) = y.length) * x.length
  evenSumOutcomes / totalOutcomes

/-- Theorem stating that the probability of an even sum when spinning X, Y, and Z is 5/9 -/
theorem prob_even_sum_is_five_ninths : probEvenSum X Y Z = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_five_ninths_l148_14879


namespace NUMINAMATH_CALUDE_dans_age_l148_14825

theorem dans_age : 
  ∀ x : ℕ, (x + 18 = 5 * (x - 6)) → x = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_dans_age_l148_14825


namespace NUMINAMATH_CALUDE_raffle_probabilities_l148_14811

/-- Represents the raffle ticket distribution -/
structure RaffleTickets where
  total : ℕ
  first_prize : ℕ
  second_prize : ℕ
  third_prize : ℕ
  h_total : total = first_prize + second_prize + third_prize

/-- The probability of drawing exactly k tickets of a specific type from n tickets in m draws -/
def prob_draw (n k m : ℕ) : ℚ :=
  (n.choose k * (n - k).choose (m - k)) / n.choose m

theorem raffle_probabilities (r : RaffleTickets)
    (h1 : r.total = 10)
    (h2 : r.first_prize = 2)
    (h3 : r.second_prize = 3)
    (h4 : r.third_prize = 5) :
  /- (I) Probability of drawing 2 first prize tickets -/
  (prob_draw r.first_prize 2 2 = 1 / 45) ∧
  /- (II) Probability of drawing at most 1 first prize ticket in 3 draws -/
  (prob_draw r.first_prize 0 3 + prob_draw r.first_prize 1 3 = 14 / 15) ∧
  /- (III) Mathematical expectation of second prize tickets in 3 draws -/
  (0 * prob_draw r.second_prize 0 3 +
   1 * prob_draw r.second_prize 1 3 +
   2 * prob_draw r.second_prize 2 3 +
   3 * prob_draw r.second_prize 3 3 = 9 / 10) :=
by sorry

end NUMINAMATH_CALUDE_raffle_probabilities_l148_14811
