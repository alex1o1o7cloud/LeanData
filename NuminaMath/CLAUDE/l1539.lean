import Mathlib

namespace multiplication_results_l1539_153942

theorem multiplication_results (h : 25 * 4 = 100) : 
  (25 * 8 = 200) ∧ 
  (25 * 12 = 300) ∧ 
  (250 * 40 = 10000) ∧ 
  (25 * 24 = 600) := by
sorry

end multiplication_results_l1539_153942


namespace disjunction_true_l1539_153910

theorem disjunction_true : 
  (∀ x : ℝ, x < 0 → 2^x > x) ∨ (∃ x : ℝ, x^2 + x + 1 < 0) := by sorry

end disjunction_true_l1539_153910


namespace function_determination_l1539_153915

theorem function_determination (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) :
  ∀ x, f x = x^2 - 2*x := by
sorry

end function_determination_l1539_153915


namespace extended_segment_coordinates_l1539_153956

/-- Given two points A and B on a plane, and a point C such that BC = 1/2 * AB,
    this theorem proves that C has specific coordinates. -/
theorem extended_segment_coordinates (A B C : ℝ × ℝ) : 
  A = (2, -2) → 
  B = (14, 4) → 
  C.1 - B.1 = (B.1 - A.1) / 2 → 
  C.2 - B.2 = (B.2 - A.2) / 2 → 
  C = (20, 7) := by
sorry


end extended_segment_coordinates_l1539_153956


namespace complex_square_root_l1539_153978

theorem complex_square_root : ∃ (z : ℂ),
  let a : ℝ := Real.sqrt ((-81 + Real.sqrt 8865) / 2)
  let b : ℝ := -24 / a
  z = Complex.mk a b ∧ z^2 = Complex.mk (-81) (-48) := by
  sorry

end complex_square_root_l1539_153978


namespace solve_for_x_l1539_153966

theorem solve_for_x (x y : ℝ) (h1 : x - y = 7) (h2 : x + y = 11) : x = 9 := by
  sorry

end solve_for_x_l1539_153966


namespace absolute_value_equation_l1539_153982

theorem absolute_value_equation (x : ℝ) : |x - 3| = 2 → x = 5 ∨ x = 1 := by
  sorry

end absolute_value_equation_l1539_153982


namespace student_problem_attempt_l1539_153937

theorem student_problem_attempt :
  ∀ (correct incorrect : ℕ),
    correct + incorrect ≤ 20 ∧
    8 * correct - 5 * incorrect = 13 →
    correct + incorrect = 13 :=
by
  sorry

end student_problem_attempt_l1539_153937


namespace sandy_age_l1539_153980

theorem sandy_age (S M : ℕ) (h1 : S = M - 16) (h2 : S * 9 = M * 7) : S = 56 := by
  sorry

end sandy_age_l1539_153980


namespace binary_110_equals_6_l1539_153933

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Theorem statement
theorem binary_110_equals_6 :
  binary_to_decimal [false, true, true] = 6 := by
  sorry

end binary_110_equals_6_l1539_153933


namespace A_union_complement_B_eq_A_l1539_153913

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,3,5}
def B : Set Nat := {2,4}

theorem A_union_complement_B_eq_A : A ∪ (U \ B) = A := by
  sorry

end A_union_complement_B_eq_A_l1539_153913


namespace x_convergence_interval_l1539_153911

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 8) / (x n + 7)

theorem x_convergence_interval :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧ x m ≤ 4 + 1 / (2^18) ∧
  ∀ k : ℕ, 0 < k ∧ k < 81 → x k > 4 + 1 / (2^18) := by
  sorry

end x_convergence_interval_l1539_153911


namespace find_m_l1539_153972

theorem find_m : ∃ m : ℝ, (15 : ℝ)^(4*m) = (1/15 : ℝ)^(m-30) → m = 6 := by
  sorry

end find_m_l1539_153972


namespace square_sum_eq_double_product_iff_equal_l1539_153947

theorem square_sum_eq_double_product_iff_equal (a b : ℝ) :
  a^2 + b^2 = 2*a*b ↔ a = b := by
sorry

end square_sum_eq_double_product_iff_equal_l1539_153947


namespace pure_imaginary_product_l1539_153912

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (1 - Complex.I) * (a + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end pure_imaginary_product_l1539_153912


namespace rhombus_fourth_vertex_area_l1539_153949

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its four vertices -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- A rhombus defined by its four vertices -/
structure Rhombus where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Predicate to check if a square is a unit square -/
def isUnitSquare (sq : Square) : Prop := sorry

/-- Predicate to check if a point lies on a side of the square -/
def pointOnSide (p : Point) (sq : Square) : Prop := sorry

/-- Function to calculate the area of a set of points -/
def areaOfSet (s : Set Point) : ℝ := sorry

/-- The set of all possible locations for the fourth vertex of the rhombus -/
def fourthVertexSet (sq : Square) : Set Point := sorry

/-- Main theorem -/
theorem rhombus_fourth_vertex_area (sq : Square) :
  isUnitSquare sq →
  (∃ (r : Rhombus), 
    pointOnSide r.p sq ∧ 
    pointOnSide r.q sq ∧ 
    pointOnSide r.r sq) →
  areaOfSet (fourthVertexSet sq) = 7/3 := sorry

end rhombus_fourth_vertex_area_l1539_153949


namespace distributive_law_analogy_l1539_153987

theorem distributive_law_analogy (a b c : ℝ) (h : c ≠ 0) :
  (a + b) * c = a * c + b * c ↔ (a + b) / c = a / c + b / c :=
sorry

end distributive_law_analogy_l1539_153987


namespace democrat_count_l1539_153976

theorem democrat_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 810 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (1 / 3 : ℚ) * total →
  female / 2 = 135 :=
by
  sorry

end democrat_count_l1539_153976


namespace spade_calculation_l1539_153945

/-- Definition of the ♠ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 6 ♠ (7 ♠ 7) = 36 -/
theorem spade_calculation : spade 6 (spade 7 7) = 36 := by
  sorry

end spade_calculation_l1539_153945


namespace cosine_sum_theorem_l1539_153929

theorem cosine_sum_theorem (x m : Real) (h : Real.cos (x - Real.pi/6) = m) :
  Real.cos x + Real.cos (x - Real.pi/3) = Real.sqrt 3 * m := by
  sorry

end cosine_sum_theorem_l1539_153929


namespace negation_equivalence_l1539_153926

theorem negation_equivalence (P Q : Prop) :
  ¬(P → ¬Q) ↔ (P ∧ Q) := by
  sorry

end negation_equivalence_l1539_153926


namespace two_from_same_class_three_from_same_class_l1539_153924

/-- A function representing the distribution of students among classes -/
def Distribution (n : ℕ) := Fin 3 → ℕ

/-- The sum of students in all classes equals the total number of students -/
def valid_distribution (n : ℕ) (d : Distribution n) : Prop :=
  (d 0) + (d 1) + (d 2) = n

/-- There exists a class with at least k students -/
def exists_class_with_k_students (n k : ℕ) (d : Distribution n) : Prop :=
  ∃ i : Fin 3, d i ≥ k

theorem two_from_same_class (n : ℕ) (h : n ≥ 4) :
  ∀ d : Distribution n, valid_distribution n d → exists_class_with_k_students n 2 d :=
sorry

theorem three_from_same_class (n : ℕ) (h : n ≥ 7) :
  ∀ d : Distribution n, valid_distribution n d → exists_class_with_k_students n 3 d :=
sorry

end two_from_same_class_three_from_same_class_l1539_153924


namespace square_of_105_l1539_153917

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end square_of_105_l1539_153917


namespace ellipse_eccentricity_l1539_153963

/-- The eccentricity of an ellipse with equation x^2 + ky^2 = 3k (k > 0) is √3/2,
    given that one of its foci coincides with the focus of the parabola y^2 = 12x. -/
theorem ellipse_eccentricity (k : ℝ) (h_k : k > 0) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 3*k}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let parabola_focus : ℝ × ℝ := (3, 0)
  ∃ (ellipse_focus : ℝ × ℝ), 
    ellipse_focus ∈ ellipse ∧ 
    ellipse_focus = parabola_focus →
    let a := Real.sqrt (3*k)
    let b := Real.sqrt 3
    let c := 3
    let eccentricity := c / a
    eccentricity = Real.sqrt 3 / 2 :=
by sorry

end ellipse_eccentricity_l1539_153963


namespace davids_pushups_l1539_153946

theorem davids_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) :
  zachary_pushups = 47 →
  david_extra_pushups = 15 →
  zachary_pushups + david_extra_pushups = 62 :=
by sorry

end davids_pushups_l1539_153946


namespace rhombus_area_l1539_153927

/-- The area of a rhombus with vertices at (0, 3.5), (8, 0), (0, -3.5), and (-8, 0) is 56 square units. -/
theorem rhombus_area : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (8, 0), (0, -3.5), (-8, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |8 - (-8)|
  (diag1 * diag2) / 2 = 56 := by
  sorry

end rhombus_area_l1539_153927


namespace books_read_by_tony_dean_breanna_l1539_153983

/-- The number of different books read by Tony, Dean, and Breanna -/
def totalDifferentBooks (tonyBooks deanBooks breannaBooks sharedTonyDean sharedAll : ℕ) : ℕ :=
  tonyBooks + deanBooks + breannaBooks - sharedTonyDean - sharedAll

/-- Theorem stating the total number of different books read -/
theorem books_read_by_tony_dean_breanna : 
  totalDifferentBooks 23 12 17 3 1 = 48 := by
  sorry

#eval totalDifferentBooks 23 12 17 3 1

end books_read_by_tony_dean_breanna_l1539_153983


namespace estate_area_calculation_l1539_153903

-- Define the scale conversion factor
def scale : ℝ := 500

-- Define the map dimensions
def map_width : ℝ := 5
def map_height : ℝ := 3

-- Define the actual dimensions
def actual_width : ℝ := scale * map_width
def actual_height : ℝ := scale * map_height

-- Define the actual area
def actual_area : ℝ := actual_width * actual_height

-- Theorem to prove
theorem estate_area_calculation :
  actual_area = 3750000 := by
  sorry

end estate_area_calculation_l1539_153903


namespace system_solution_l1539_153986

theorem system_solution : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 := by
  sorry

end system_solution_l1539_153986


namespace gwen_final_amount_l1539_153904

def initial_amount : ℚ := 5.00
def candy_expense : ℚ := 3.25
def recycling_income : ℚ := 1.50
def card_expense : ℚ := 0.70

theorem gwen_final_amount :
  initial_amount - candy_expense + recycling_income - card_expense = 2.55 := by
  sorry

end gwen_final_amount_l1539_153904


namespace remainder_squared_plus_five_l1539_153996

theorem remainder_squared_plus_five (a : ℕ) (h : a % 7 = 4) :
  (a^2 + 5) % 7 = 0 := by
  sorry

end remainder_squared_plus_five_l1539_153996


namespace fence_perimeter_l1539_153965

/-- The number of posts -/
def num_posts : ℕ := 36

/-- The width of each post in feet -/
def post_width : ℚ := 1/2

/-- The distance between adjacent posts in feet -/
def post_spacing : ℕ := 6

/-- The number of posts per side of the square field -/
def posts_per_side : ℕ := 10

/-- The length of one side of the square field in feet -/
def side_length : ℚ := (posts_per_side - 1) * post_spacing + posts_per_side * post_width

/-- The outer perimeter of the fence in feet -/
def outer_perimeter : ℚ := 4 * side_length

theorem fence_perimeter : outer_perimeter = 236 := by sorry

end fence_perimeter_l1539_153965


namespace f_inequality_l1539_153964

open Real

-- Define a derivable function f on ℝ
variable (f : ℝ → ℝ)

-- Define the condition that f is twice differentiable
variable (hf : TwiceDifferentiable ℝ f)

-- Define the condition 3f(x) > f''(x) for all x ∈ ℝ
variable (h1 : ∀ x : ℝ, 3 * f x > (deriv^[2] f) x)

-- Define the condition f(1) = e^3
variable (h2 : f 1 = exp 3)

-- State the theorem
theorem f_inequality : f 2 < exp 6 := by
  sorry

end f_inequality_l1539_153964


namespace exists_quadrilateral_equal_tangents_l1539_153941

-- Define a quadrilateral as a structure with four angles
structure Quadrilateral where
  α : Real
  β : Real
  γ : Real
  δ : Real

-- Define the property of having equal tangents for all angles
def hasEqualTangents (q : Quadrilateral) : Prop :=
  Real.tan q.α = Real.tan q.β ∧
  Real.tan q.β = Real.tan q.γ ∧
  Real.tan q.γ = Real.tan q.δ

-- Define the property of angles summing to 360°
def anglesSum360 (q : Quadrilateral) : Prop :=
  q.α + q.β + q.γ + q.δ = 360

-- Theorem stating the existence of a quadrilateral with equal tangents
theorem exists_quadrilateral_equal_tangents :
  ∃ q : Quadrilateral, anglesSum360 q ∧ hasEqualTangents q :=
sorry

end exists_quadrilateral_equal_tangents_l1539_153941


namespace train_bridge_crossing_time_l1539_153923

theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (total_length : ℝ) 
  (h1 : train_length = 130)
  (h2 : train_speed_kmh = 45)
  (h3 : total_length = 245) :
  let bridge_length := total_length - train_length
  let total_distance := total_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 19.6 := by sorry

end train_bridge_crossing_time_l1539_153923


namespace modified_cube_surface_area_l1539_153907

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents the resulting structure after modifications -/
structure ModifiedCube where
  original : Cube 9
  small_cubes : ℕ := 27
  removed_corners : ℕ := 8

/-- Calculates the surface area of the modified cube structure -/
def surface_area (mc : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 1056 -/
theorem modified_cube_surface_area :
  ∀ (mc : ModifiedCube), surface_area mc = 1056 :=
sorry

end modified_cube_surface_area_l1539_153907


namespace quartic_ratio_l1539_153936

theorem quartic_ratio (a b c d e : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) →
  c / e = 35 / 24 := by
  sorry

end quartic_ratio_l1539_153936


namespace shaded_percentage_is_correct_l1539_153940

/-- Represents a 7x7 grid with a checkered shading pattern and unshaded fourth row and column -/
structure CheckeredGrid :=
  (size : Nat)
  (is_seven_by_seven : size = 7)
  (checkered_pattern : Bool)
  (unshaded_fourth_row : Bool)
  (unshaded_fourth_column : Bool)

/-- Calculates the number of shaded squares in the CheckeredGrid -/
def shaded_squares (grid : CheckeredGrid) : Nat :=
  grid.size * grid.size - (grid.size + grid.size - 1)

/-- Calculates the total number of squares in the CheckeredGrid -/
def total_squares (grid : CheckeredGrid) : Nat :=
  grid.size * grid.size

/-- Theorem stating that the percentage of shaded squares is 36/49 -/
theorem shaded_percentage_is_correct (grid : CheckeredGrid) :
  (shaded_squares grid : ℚ) / (total_squares grid : ℚ) = 36 / 49 := by
  sorry

#eval (36 : ℚ) / 49  -- To show the approximate decimal value

end shaded_percentage_is_correct_l1539_153940


namespace min_reciprocal_sum_l1539_153905

/-- Given two orthogonal vectors a = (x-1, y) and b = (1, 2), with x > 0 and y > 0,
    the minimum value of 1/x + 1/y is 3 + 2√2 -/
theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
    (h_orthogonal : (x - 1) * 1 + y * 2 = 0) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → (x' - 1) * 1 + y' * 2 = 0 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) ∧
  1 / x + 1 / y = 3 + 2 * Real.sqrt 2 := by
  sorry

end min_reciprocal_sum_l1539_153905


namespace equations_solution_l1539_153918

def satisfies_equations (a b c d : ℕ) : Prop :=
  a + b = c * d ∧ c + d = a * b

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2,2,2,2), (1,2,3,5), (2,1,3,5), (1,2,5,3), (2,1,5,3), (3,5,1,2), (5,3,1,2), (3,5,2,1), (5,3,2,1)}

theorem equations_solution :
  ∀ (a b c d : ℕ), satisfies_equations a b c d ↔ (a, b, c, d) ∈ solution_set :=
by sorry

end equations_solution_l1539_153918


namespace total_tax_percentage_l1539_153919

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.05)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.055 := by
sorry


end total_tax_percentage_l1539_153919


namespace pears_transport_l1539_153973

/-- Prove that given 8 tons of apples and the amount of pears being 7 times the amount of apples,
    the total amount of pears transported is 56 tons. -/
theorem pears_transport (apple_tons : ℕ) (pear_multiplier : ℕ) : 
  apple_tons = 8 → pear_multiplier = 7 → apple_tons * pear_multiplier = 56 := by
  sorry

end pears_transport_l1539_153973


namespace special_sequence_bijective_l1539_153962

/-- A sequence of integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, ∃ k > n, a k > 0) ∧  -- Infinite positive values
  (∀ n : ℕ, ∃ k > n, a k < 0) ∧  -- Infinite negative values
  (∀ n : ℕ+, ∀ i j, i ≠ j → i ≤ n → j ≤ n → a i % n ≠ a j % n)  -- Distinct modulo n

/-- The theorem stating that every integer appears exactly once in the sequence -/
theorem special_sequence_bijective (a : ℕ → ℤ) (h : SpecialSequence a) :
  Function.Bijective a :=
sorry

end special_sequence_bijective_l1539_153962


namespace log_16_2_l1539_153954

theorem log_16_2 : Real.log 2 / Real.log 16 = 1 / 4 := by sorry

end log_16_2_l1539_153954


namespace rose_ratio_l1539_153943

theorem rose_ratio (total : ℕ) (red : ℕ) (yellow : ℕ) (white : ℕ) : 
  total = 80 →
  yellow = (total - red) / 4 →
  red + white = 75 →
  total = red + yellow + white →
  (red : ℚ) / total = 3 / 4 := by
  sorry

end rose_ratio_l1539_153943


namespace jesse_bananas_l1539_153948

/-- The number of friends Jesse shares his bananas with -/
def num_friends : ℕ := 3

/-- The number of bananas each friend would get if Jesse shares equally -/
def bananas_per_friend : ℕ := 7

/-- The total number of bananas Jesse has -/
def total_bananas : ℕ := num_friends * bananas_per_friend

theorem jesse_bananas : total_bananas = 21 := by
  sorry

end jesse_bananas_l1539_153948


namespace matthews_crayons_count_l1539_153916

/-- The number of crayons Annie starts with -/
def initial_crayons : ℕ := 4

/-- The number of crayons Annie ends with -/
def final_crayons : ℕ := 40

/-- The number of crayons Matthew gave to Annie -/
def matthews_crayons : ℕ := final_crayons - initial_crayons

theorem matthews_crayons_count : matthews_crayons = 36 := by
  sorry

end matthews_crayons_count_l1539_153916


namespace polygon_sides_when_interior_thrice_exterior_l1539_153958

theorem polygon_sides_when_interior_thrice_exterior : ∀ n : ℕ,
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 :=
by
  sorry

end polygon_sides_when_interior_thrice_exterior_l1539_153958


namespace lower_right_is_four_l1539_153921

def Grid := Fin 4 → Fin 4 → Fin 4

def valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧
  (∀ i j k, i ≠ j → g k i ≠ g k j)

def initial_conditions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 3 = 2 ∧ g 1 2 = 3 ∧ g 2 0 = 3 ∧ g 3 1 = 0

theorem lower_right_is_four (g : Grid) 
  (h1 : valid_grid g) 
  (h2 : initial_conditions g) : 
  g 3 3 = 3 := by sorry

end lower_right_is_four_l1539_153921


namespace magnitude_of_b_l1539_153961

def a : ℝ × ℝ := (2, 1)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h1 : a.fst * b.fst + a.snd * b.snd = 10)
  (h2 : (a.fst + 2 * b.fst)^2 + (a.snd + 2 * b.snd)^2 = 50) :
  Real.sqrt (b.fst^2 + b.snd^2) = Real.sqrt 5 / 2 := by
  sorry

end magnitude_of_b_l1539_153961


namespace star_property_l1539_153999

-- Define the operation ※
def star (a b m n : ℕ) : ℕ := (a^b)^m + (b^a)^n

-- State the theorem
theorem star_property (m n : ℕ) :
  (star 1 4 m n = 10) → (star 2 2 m n = 15) → (4^(2*m + n - 1) = 81) := by
  sorry

end star_property_l1539_153999


namespace four_numbers_product_equality_l1539_153971

theorem four_numbers_product_equality (p : ℝ) (hp : p ≥ 1) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a : ℝ) > p ∧ (b : ℝ) > p ∧ (c : ℝ) > p ∧ (d : ℝ) > p ∧
    (a : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (b : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (c : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (d : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (a * b : ℕ) = c * d :=
by sorry

end four_numbers_product_equality_l1539_153971


namespace vacation_group_size_l1539_153989

def airbnb_cost : ℕ := 3200
def car_cost : ℕ := 800
def share_per_person : ℕ := 500

theorem vacation_group_size :
  (airbnb_cost + car_cost) / share_per_person = 8 :=
by sorry

end vacation_group_size_l1539_153989


namespace negation_equivalence_l1539_153934

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (a % 2 = 0 ∧ c % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c :=
sorry

end negation_equivalence_l1539_153934


namespace rhombus_side_length_l1539_153920

/-- A rhombus with area L and longer diagonal three times the shorter diagonal has side length √(5L/3) -/
theorem rhombus_side_length (L : ℝ) (h : L > 0) : 
  ∃ (short_diag long_diag side : ℝ),
    short_diag > 0 ∧
    long_diag = 3 * short_diag ∧
    L = (1/2) * short_diag * long_diag ∧
    side^2 = (short_diag/2)^2 + (long_diag/2)^2 ∧
    side = Real.sqrt ((5 * L) / 3) :=
by sorry

end rhombus_side_length_l1539_153920


namespace miniature_cars_per_package_l1539_153991

theorem miniature_cars_per_package (total_packages : ℕ) (fraction_given_away : ℚ) (cars_left : ℕ) : 
  total_packages = 10 → 
  fraction_given_away = 2/5 → 
  cars_left = 30 → 
  ∃ (cars_per_package : ℕ), 
    cars_per_package = 5 ∧ 
    (total_packages * cars_per_package) * (1 - fraction_given_away) = cars_left :=
by
  sorry

end miniature_cars_per_package_l1539_153991


namespace negation_equivalence_l1539_153951

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end negation_equivalence_l1539_153951


namespace sqrt_operations_l1539_153909

theorem sqrt_operations :
  (∀ x y : ℝ, x > 0 → y > 0 → (Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y)) ∧
  (Real.sqrt 12 / Real.sqrt 3 = 2) ∧
  (Real.sqrt 8 = 2 * Real.sqrt 2) ∧
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by sorry

end sqrt_operations_l1539_153909


namespace hexagon_perimeter_l1539_153906

theorem hexagon_perimeter (AB BC CD DE EF : ℝ) (AC AD AE AF : ℝ) : 
  AB = 1 →
  BC = 1 →
  CD = 1 →
  DE = 2 →
  EF = 1 →
  AC^2 = AB^2 + BC^2 →
  AD^2 = AC^2 + CD^2 →
  AE^2 = AD^2 + DE^2 →
  AF^2 = AE^2 + EF^2 →
  AB + BC + CD + DE + EF + AF = 6 + 2 * Real.sqrt 2 := by
  sorry

end hexagon_perimeter_l1539_153906


namespace sin_cos_ratio_l1539_153944

theorem sin_cos_ratio (θ : Real) (h : Real.sqrt 2 * Real.sin (θ + π/4) = 3 * Real.cos θ) :
  Real.sin θ / (Real.sin θ - Real.cos θ) = 2 := by
  sorry

end sin_cos_ratio_l1539_153944


namespace tree_growth_problem_l1539_153935

/-- A tree growing problem -/
theorem tree_growth_problem (initial_height : ℝ) (growth_rate : ℝ) :
  initial_height = 4 →
  growth_rate = 0.4 →
  ∃ (total_years : ℕ),
    total_years = 6 ∧
    (initial_height + total_years * growth_rate) = 
    (initial_height + 4 * growth_rate) * (1 + 1/7) :=
by sorry

end tree_growth_problem_l1539_153935


namespace pavan_journey_l1539_153939

theorem pavan_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 11 →
  speed1 = 30 →
  speed2 = 25 →
  ∃ (total_distance : ℝ),
    total_distance / (2 * speed1) + total_distance / (2 * speed2) = total_time ∧
    total_distance = 300 :=
by sorry

end pavan_journey_l1539_153939


namespace trig_identity_l1539_153979

theorem trig_identity (α : ℝ) 
  (h1 : Real.cos (7 * Real.pi / 2 + α) = 4 / 7)
  (h2 : Real.tan α < 0) :
  Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) * Real.tan α = (4 + Real.sqrt 33) / 7 := by
sorry

end trig_identity_l1539_153979


namespace steve_exceeds_goal_and_optimal_strategy_l1539_153998

/-- Represents the berry types --/
inductive Berry
  | Lingonberry
  | Cloudberry
  | Blueberry

/-- Represents Steve's berry-picking job --/
structure BerryJob where
  goal : ℕ
  payRates : Berry → ℕ
  basketCapacity : ℕ
  mondayPicking : Berry → ℕ
  tuesdayPicking : Berry → ℕ

def stevesJob : BerryJob :=
  { goal := 150
  , payRates := fun b => match b with
      | Berry.Lingonberry => 2
      | Berry.Cloudberry => 3
      | Berry.Blueberry => 5
  , basketCapacity := 30
  , mondayPicking := fun b => match b with
      | Berry.Lingonberry => 8
      | Berry.Cloudberry => 10
      | Berry.Blueberry => 12
  , tuesdayPicking := fun b => match b with
      | Berry.Lingonberry => 24
      | Berry.Cloudberry => 20
      | Berry.Blueberry => 5
  }

def earnings (job : BerryJob) (picking : Berry → ℕ) : ℕ :=
  (picking Berry.Lingonberry * job.payRates Berry.Lingonberry) +
  (picking Berry.Cloudberry * job.payRates Berry.Cloudberry) +
  (picking Berry.Blueberry * job.payRates Berry.Blueberry)

def totalEarnings (job : BerryJob) : ℕ :=
  earnings job job.mondayPicking + earnings job job.tuesdayPicking

theorem steve_exceeds_goal_and_optimal_strategy (job : BerryJob) :
  (totalEarnings job > job.goal) ∧
  (∀ picking : Berry → ℕ,
    (picking Berry.Lingonberry + picking Berry.Cloudberry + picking Berry.Blueberry ≤ job.basketCapacity) →
    (earnings job picking ≤ job.basketCapacity * job.payRates Berry.Blueberry)) :=
by sorry

end steve_exceeds_goal_and_optimal_strategy_l1539_153998


namespace function_composition_condition_l1539_153955

theorem function_composition_condition (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), ∀ x, f (g x) = x + a ∧ g (f x) = x + b) ↔ |a| = |b| :=
by sorry

end function_composition_condition_l1539_153955


namespace right_triangle_sides_l1539_153908

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = 8 ∧ b = 15 ∧ c = 17 ∧ a^2 + b^2 = c^2 := by
  sorry

end right_triangle_sides_l1539_153908


namespace hyperbola_transverse_axis_length_l1539_153950

/-- The length of the transverse axis of a hyperbola with equation x²/9 - y²/16 = 1 is 6 -/
theorem hyperbola_transverse_axis_length :
  ∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 1 →
  ∃ (a : ℝ), a > 0 ∧ a^2 = 9 ∧ 2 * a = 6 :=
by sorry

end hyperbola_transverse_axis_length_l1539_153950


namespace equilateral_triangle_on_parallel_lines_l1539_153990

-- Define the parallel lines
variable (D₁ D₂ D₃ : Set (ℝ × ℝ))

-- Define the property of being parallel
def Parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point on a line
def PointOnLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := p ∈ l

-- Define an equilateral triangle
def IsEquilateralTriangle (A₁ A₂ A₃ : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem equilateral_triangle_on_parallel_lines
  (h₁ : Parallel D₁ D₂)
  (h₂ : Parallel D₂ D₃)
  (h₃ : Parallel D₁ D₃) :
  ∃ (A₁ A₂ A₃ : ℝ × ℝ),
    IsEquilateralTriangle A₁ A₂ A₃ ∧
    PointOnLine A₁ D₁ ∧
    PointOnLine A₂ D₂ ∧
    PointOnLine A₃ D₃ := by
  sorry

end equilateral_triangle_on_parallel_lines_l1539_153990


namespace sixtieth_digit_is_five_l1539_153981

def repeating_decimal (whole : ℕ) (repeating : List ℕ) : ℚ := sorry

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem sixtieth_digit_is_five :
  let decimal := repeating_decimal 6 [4, 5, 3]
  nth_digit_after_decimal decimal 60 = 5 := by sorry

end sixtieth_digit_is_five_l1539_153981


namespace local_minimum_condition_l1539_153968

/-- The function f(x) = x(x-a)² has a local minimum at x=2 if and only if a = 2 -/
theorem local_minimum_condition (a : ℝ) :
  (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → x * (x - a)^2 ≥ 2 * (2 - a)^2) ↔ a = 2 := by
  sorry

end local_minimum_condition_l1539_153968


namespace min_value_x_plus_y_l1539_153984

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 := by
  sorry

end min_value_x_plus_y_l1539_153984


namespace solution_set_of_inequality_range_of_m_for_full_solution_set_solution_set_eq_nonnegative_reals_l1539_153970

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- Theorem for part I
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ 3 * x + 4 ↔ x ≥ 0 := by sorry

-- Theorem for part II
theorem range_of_m_for_full_solution_set :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ m) ↔ m ∈ Set.Iic 4 := by sorry

-- Define the solution set for part I
def solution_set : Set ℝ := {x : ℝ | f x ≤ 3 * x + 4}

-- Theorem stating that the solution set is equivalent to [0, +∞)
theorem solution_set_eq_nonnegative_reals :
  solution_set = Set.Ici 0 := by sorry

end solution_set_of_inequality_range_of_m_for_full_solution_set_solution_set_eq_nonnegative_reals_l1539_153970


namespace and_false_necessary_not_sufficient_for_or_false_l1539_153902

theorem and_false_necessary_not_sufficient_for_or_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
sorry

end and_false_necessary_not_sufficient_for_or_false_l1539_153902


namespace bowling_ball_weight_l1539_153930

/-- Given that 8 bowling balls weigh the same as 5 kayaks, and 4 kayaks weigh 120 pounds,
    prove that one bowling ball weighs 18.75 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowl_weight kayak_weight : ℝ),
    8 * bowl_weight = 5 * kayak_weight →
    4 * kayak_weight = 120 →
    bowl_weight = 18.75 := by
  sorry

end bowling_ball_weight_l1539_153930


namespace bowling_ball_weight_l1539_153993

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (7 * bowling_ball_weight = 4 * canoe_weight) →
    (3 * canoe_weight = 84) →
    bowling_ball_weight = 16 := by
  sorry

end bowling_ball_weight_l1539_153993


namespace quadratic_inequality_solution_l1539_153995

-- Define the quadratic function
def f (x : ℝ) := -x^2 + 3*x + 28

-- Define the solution set
def solution_set := {x : ℝ | x ≤ -4 ∨ x ≥ 7}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x ≤ 0} = solution_set :=
by sorry

end quadratic_inequality_solution_l1539_153995


namespace problem_solution_l1539_153901

def set_A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 1}

def set_B : Set ℝ := {x | (4 - x) / (x + 2) ≥ 0}

theorem problem_solution :
  (∀ a : ℝ, a = 2 → (set_A a)ᶜ ∩ set_B = {x | -2 < x ∧ x < 1}) ∧
  (∀ a : ℝ, set_A a ∪ set_B = set_B ↔ a < -2 ∨ (-1 < a ∧ a ≤ 3/2)) :=
sorry

end problem_solution_l1539_153901


namespace quadratic_roots_and_triangle_perimeter_l1539_153997

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (m + 3) * x + 4 * m - 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m - 5)^2

-- Define the isosceles triangle ABC
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : b = c
  one_side_is_five : a = 5
  roots_are_sides : ∃ m : ℝ, (quadratic_equation m b = 0) ∧ (quadratic_equation m c = 0)

-- Theorem statement
theorem quadratic_roots_and_triangle_perimeter :
  (∀ m : ℝ, discriminant m ≥ 0) ∧
  (∀ t : IsoscelesTriangle, t.a + t.b + t.c = 13 ∨ t.a + t.b + t.c = 14) :=
by sorry

end quadratic_roots_and_triangle_perimeter_l1539_153997


namespace skylar_donation_l1539_153977

/-- Represents the donation scenario for Skylar -/
structure DonationScenario where
  start_age : ℕ
  current_age : ℕ
  annual_donation : ℕ

/-- Calculates the total amount donated given a DonationScenario -/
def total_donated (scenario : DonationScenario) : ℕ :=
  (scenario.current_age - scenario.start_age) * scenario.annual_donation

/-- Theorem stating that Skylar's total donation is $432,000 -/
theorem skylar_donation :
  let scenario : DonationScenario := {
    start_age := 17,
    current_age := 71,
    annual_donation := 8000
  }
  total_donated scenario = 432000 := by
  sorry

end skylar_donation_l1539_153977


namespace diophantine_approximation_l1539_153974

theorem diophantine_approximation (α : ℝ) (C : ℝ) (h_α : α > 0) (h_C : C > 1) :
  ∃ (x : ℕ) (y : ℤ), (x : ℝ) < C ∧ |x * α - y| ≤ 1 / C := by
  sorry

end diophantine_approximation_l1539_153974


namespace third_trapezoid_largest_area_l1539_153925

-- Define the lengths of the segments
def a : ℝ := 2.12
def b : ℝ := 2.71
def c : ℝ := 3.53

-- Define the area calculation function for a trapezoid
def trapezoidArea (top bottom height : ℝ) : ℝ := (top + bottom) * height

-- Define the three possible trapezoids
def trapezoid1 : ℝ := trapezoidArea a c b
def trapezoid2 : ℝ := trapezoidArea b c a
def trapezoid3 : ℝ := trapezoidArea a b c

-- Theorem statement
theorem third_trapezoid_largest_area :
  trapezoid3 > trapezoid1 ∧ trapezoid3 > trapezoid2 :=
by sorry

end third_trapezoid_largest_area_l1539_153925


namespace eggs_leftover_l1539_153969

theorem eggs_leftover (abigail beatrice carson : ℕ) 
  (h_abigail : abigail = 60)
  (h_beatrice : beatrice = 75)
  (h_carson : carson = 27) :
  (abigail + beatrice + carson) % 18 = 0 := by
sorry

end eggs_leftover_l1539_153969


namespace john_burritos_days_l1539_153988

theorem john_burritos_days (boxes : ℕ) (burritos_per_box : ℕ) (fraction_given_away : ℚ)
  (burritos_eaten_per_day : ℕ) (burritos_left : ℕ) :
  boxes = 3 →
  burritos_per_box = 20 →
  fraction_given_away = 1 / 3 →
  burritos_eaten_per_day = 3 →
  burritos_left = 10 →
  (boxes * burritos_per_box * (1 - fraction_given_away) - burritos_left) / burritos_eaten_per_day = 10 := by
  sorry

#check john_burritos_days

end john_burritos_days_l1539_153988


namespace machine_production_time_l1539_153994

theorem machine_production_time (T : ℝ) : 
  T > 0 ∧ 
  (1 / T + 1 / 30 = 1 / 12) → 
  T = 45 := by
sorry

end machine_production_time_l1539_153994


namespace drum_capacity_ratio_l1539_153992

theorem drum_capacity_ratio (CX CY : ℝ) 
  (h1 : CX > 0) 
  (h2 : CY > 0) 
  (h3 : (1/2 * CX + 1/3 * CY) / CY = 7/12) : 
  CY / CX = 2 := by
sorry

end drum_capacity_ratio_l1539_153992


namespace count_solutions_l1539_153960

def is_solution (m n r : ℕ+) : Prop :=
  m * n + n * r + m * r = 2 * (m + n + r)

theorem count_solutions : 
  ∃! (solutions : Finset (ℕ+ × ℕ+ × ℕ+)), 
    (∀ (m n r : ℕ+), (m, n, r) ∈ solutions ↔ is_solution m n r) ∧ 
    solutions.card = 7 :=
sorry

end count_solutions_l1539_153960


namespace cylinder_volume_from_square_rotation_l1539_153938

/-- The volume of a cylinder formed by rotating a square about its vertical line of symmetry -/
theorem cylinder_volume_from_square_rotation (square_side : ℝ) (height : ℝ) : 
  square_side = 10 → height = 20 → 
  (π * (square_side / 2)^2 * height : ℝ) = 500 * π := by
  sorry

end cylinder_volume_from_square_rotation_l1539_153938


namespace binomial_expansion_special_case_l1539_153931

theorem binomial_expansion_special_case : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 8^4 := by
  sorry

end binomial_expansion_special_case_l1539_153931


namespace sphere_radius_l1539_153928

theorem sphere_radius (r_A : ℝ) : 
  let r_B : ℝ := 10
  (r_A^2 / r_B^2 = 16) → r_A = 40 := by
sorry

end sphere_radius_l1539_153928


namespace locus_of_midpoints_correct_l1539_153953

/-- Given a square ABCD with center at the origin, rotating around its center,
    and a fixed line l with equation y = a, this function represents the locus of
    the midpoints of segments PQ, where P is the foot of the perpendicular from D to l,
    and Q is the midpoint of AB. -/
def locusOfMidpoints (a : ℝ) (x y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 
    p.1 = t ∧ 
    p.2 = -t + a/2 ∧ 
    t = (x - y)/4 ∧ 
    x - y ∈ Set.Icc (-Real.sqrt (2 * (x^2 + y^2))) (Real.sqrt (2 * (x^2 + y^2)))}

/-- Theorem stating that the locus of midpoints is correct for any rotating square ABCD
    with center at the origin and any fixed line y = a. -/
theorem locus_of_midpoints_correct (a : ℝ) : 
  ∀ x y : ℝ, locusOfMidpoints a x y = 
    {p : ℝ × ℝ | ∃ t : ℝ, 
      p.1 = t ∧ 
      p.2 = -t + a/2 ∧ 
      t = (x - y)/4 ∧ 
      x - y ∈ Set.Icc (-Real.sqrt (2 * (x^2 + y^2))) (Real.sqrt (2 * (x^2 + y^2)))} :=
by sorry

end locus_of_midpoints_correct_l1539_153953


namespace rhombus_diagonal_length_main_theorem_l1539_153975

/-- Represents a rhombus with given properties -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem stating the relationship between the diagonals and perimeter of a specific rhombus -/
theorem rhombus_diagonal_length (r : Rhombus) 
    (h1 : r.diagonal1 = 10)
    (h2 : r.perimeter = 52) : 
    r.diagonal2 = 24 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ r : Rhombus, r.diagonal1 = 10 ∧ r.perimeter = 52 ∧ r.diagonal2 = 24 := by
  sorry

end rhombus_diagonal_length_main_theorem_l1539_153975


namespace min_value_z_l1539_153914

theorem min_value_z (x y : ℝ) (h1 : x - y + 1 ≥ 0) (h2 : x + y - 1 ≥ 0) (h3 : x ≤ 3) :
  ∃ (z : ℝ), z = 2*x - 3*y ∧ z ≥ -6 ∧ (∀ (x' y' : ℝ), x' - y' + 1 ≥ 0 → x' + y' - 1 ≥ 0 → x' ≤ 3 → 2*x' - 3*y' ≥ z) :=
by
  sorry

end min_value_z_l1539_153914


namespace power_of_two_with_three_identical_digits_l1539_153900

theorem power_of_two_with_three_identical_digits :
  ∃ (k : ℕ), k = 39 ∧ ∃ (d : ℕ), d < 10 ∧ (2^k) % 1000 = d * 111 :=
sorry

end power_of_two_with_three_identical_digits_l1539_153900


namespace simplify_expression_l1539_153959

theorem simplify_expression : (2^5 + 4^4) * (2^2 - (-2)^3)^8 = 123876479488 := by
  sorry

end simplify_expression_l1539_153959


namespace average_rstp_l1539_153932

theorem average_rstp (r s t u : ℝ) (h : (5 / 2) * (r + s + t + u) = 20) :
  (r + s + t + u) / 4 = 2 := by
sorry

end average_rstp_l1539_153932


namespace pyramid_volume_l1539_153952

/-- The volume of a pyramid with a square base of side length 2 and height 2 is 8/3 cubic units -/
theorem pyramid_volume (base_side_length height : ℝ) (h1 : base_side_length = 2) (h2 : height = 2) :
  (1 / 3 : ℝ) * base_side_length^2 * height = 8 / 3 := by
  sorry

end pyramid_volume_l1539_153952


namespace triangle_most_stable_triangular_structures_sturdy_l1539_153957

-- Define a structure
structure Shape :=
  (stability : ℝ)

-- Define a triangle
def Triangle : Shape :=
  { stability := 1 }

-- Define other shapes (for comparison)
def Square : Shape :=
  { stability := 0.8 }

def Pentagon : Shape :=
  { stability := 0.9 }

-- Theorem: Triangles have the highest stability
theorem triangle_most_stable :
  ∀ s : Shape, Triangle.stability ≥ s.stability :=
sorry

-- Theorem: Structures using triangles are sturdy
theorem triangular_structures_sturdy (structure_stability : Shape → ℝ) :
  structure_stability Triangle = 1 →
  ∀ s : Shape, structure_stability Triangle ≥ structure_stability s :=
sorry

end triangle_most_stable_triangular_structures_sturdy_l1539_153957


namespace union_and_intersection_range_of_a_l1539_153967

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x < 10}) :=
sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  C a ⊆ B → a ≤ 3 :=
sorry

end union_and_intersection_range_of_a_l1539_153967


namespace sum_of_numbers_l1539_153922

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 := by
  sorry

end sum_of_numbers_l1539_153922


namespace consecutive_multiples_of_four_sum_l1539_153985

theorem consecutive_multiples_of_four_sum (n : ℕ) : 
  (4*n + (4*n + 8) = 140) → (4*n + (4*n + 4) + (4*n + 8) = 210) :=
by
  sorry

end consecutive_multiples_of_four_sum_l1539_153985
