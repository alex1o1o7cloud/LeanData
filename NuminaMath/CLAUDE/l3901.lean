import Mathlib

namespace max_increasing_subsequences_l3901_390144

theorem max_increasing_subsequences (A : Fin 2001 → ℕ+) :
  (Finset.univ.filter (fun i : Fin 2001 => 
    ∃ j k, i < j ∧ j < k ∧ 
    A j = A i + 1 ∧ A k = A j + 1)).card ≤ 667^3 := by
  sorry

end max_increasing_subsequences_l3901_390144


namespace sum_of_roots_is_eight_l3901_390192

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- The number of real roots of an equation -/
noncomputable def numRealRoots (f : ℝ → ℝ) : ℕ :=
  sorry

/-- Theorem: For an even function f and an odd function g, 
    the sum of the number of real roots of f(f(x)) = 0, f(g(x)) = 0, 
    g(g(x)) = 0, and g(f(x)) = 0 is equal to 8 -/
theorem sum_of_roots_is_eight 
    (f : ℝ → ℝ) (g : ℝ → ℝ) 
    (hf : IsEven f) (hg : IsOdd g) : 
  numRealRoots (λ x => f (f x)) + 
  numRealRoots (λ x => f (g x)) + 
  numRealRoots (λ x => g (g x)) + 
  numRealRoots (λ x => g (f x)) = 8 := by
  sorry

end sum_of_roots_is_eight_l3901_390192


namespace a_2007_mod_100_l3901_390128

/-- Sequence a_n defined recursively -/
def a : ℕ → ℕ
  | 0 => 7
  | n + 1 => 7^(a n)

/-- Theorem stating that a_2007 ≡ 43 (mod 100) -/
theorem a_2007_mod_100 : a 2006 % 100 = 43 := by
  sorry

end a_2007_mod_100_l3901_390128


namespace subtraction_value_l3901_390140

theorem subtraction_value (N x : ℝ) : 
  ((N - x) / 7 = 7) ∧ ((N - 6) / 8 = 6) → x = 5 := by
  sorry

end subtraction_value_l3901_390140


namespace no_right_triangle_perimeter_area_equality_l3901_390180

theorem no_right_triangle_perimeter_area_equality :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (a + b + Real.sqrt (a^2 + b^2))^2 = 2 * a * b := by
  sorry

end no_right_triangle_perimeter_area_equality_l3901_390180


namespace roots_equal_opposite_signs_l3901_390168

theorem roots_equal_opposite_signs (a b c m : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = -x ∧
    (x^2 - b*x) / (a*x - c) = (m - 1) / (m + 1) ∧
    (y^2 - b*y) / (a*y - c) = (m - 1) / (m + 1)) →
  m = (a - b) / (a + b) := by
sorry

end roots_equal_opposite_signs_l3901_390168


namespace sqrt_49_times_sqrt_64_l3901_390164

theorem sqrt_49_times_sqrt_64 : Real.sqrt (49 * Real.sqrt 64) = 14 * Real.sqrt 2 := by
  sorry

end sqrt_49_times_sqrt_64_l3901_390164


namespace lcm_of_32_and_12_l3901_390171

theorem lcm_of_32_and_12 (n m : ℕ+) (h1 : n = 32) (h2 : m = 12) (h3 : Nat.gcd n m = 8) :
  Nat.lcm n m = 48 := by
  sorry

end lcm_of_32_and_12_l3901_390171


namespace sufficient_not_necessary_l3901_390117

theorem sufficient_not_necessary (a b : ℝ) :
  ((a - b) * a^2 < 0 → a < b) ∧
  ¬(a < b → (a - b) * a^2 < 0) :=
sorry

end sufficient_not_necessary_l3901_390117


namespace hulk_jump_distance_l3901_390160

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem hulk_jump_distance (n : ℕ) : 
  (∀ k < n, geometric_sequence 3 3 k ≤ 500) ∧ 
  geometric_sequence 3 3 n > 500 → n = 6 :=
sorry

end hulk_jump_distance_l3901_390160


namespace equation_describes_hyperbola_l3901_390125

/-- The equation (x-y)^2 = x^2 + y^2 - 2 describes a hyperbola -/
theorem equation_describes_hyperbola :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), (x - y)^2 = x^2 + y^2 - 2 ↔ (x * y = 1)) :=
sorry

end equation_describes_hyperbola_l3901_390125


namespace number_and_square_sum_l3901_390104

theorem number_and_square_sum (x : ℝ) : x + x^2 = 132 → x = 11 ∨ x = -12 := by
  sorry

end number_and_square_sum_l3901_390104


namespace complex_modulus_problem_l3901_390148

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 2 + 2 * Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end complex_modulus_problem_l3901_390148


namespace problem_statement_l3901_390184

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 3) :
  (a + 1) * (b - 1) = -3 := by
sorry

end problem_statement_l3901_390184


namespace equation_solution_l3901_390165

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 33 := by
  sorry

end equation_solution_l3901_390165


namespace quadratic_coefficient_sum_l3901_390176

/-- A quadratic function passing through specific points with a given vertex -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (QuadraticFunction a b c 1 = 4) →
  (QuadraticFunction a b c (-2) = -1) →
  (∀ x, QuadraticFunction a b c x ≥ QuadraticFunction a b c (-1)) →
  (QuadraticFunction a b c (-1) = -2) →
  a + b + c = 5 := by
  sorry

end quadratic_coefficient_sum_l3901_390176


namespace inequality_proof_l3901_390196

theorem inequality_proof (x : ℝ) (h : x ≥ (1/2 : ℝ)) :
  Real.sqrt (9*x + 7) < Real.sqrt x + Real.sqrt (x + 1) + Real.sqrt (x + 2) ∧
  Real.sqrt x + Real.sqrt (x + 1) + Real.sqrt (x + 2) < Real.sqrt (9*x + 9) :=
by sorry

end inequality_proof_l3901_390196


namespace diagonals_150_sided_polygon_l3901_390130

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in the polygon -/
def sides : ℕ := 150

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon :
  num_diagonals sides = 11025 := by
  sorry

end diagonals_150_sided_polygon_l3901_390130


namespace exists_valid_coloring_l3901_390129

/-- A coloring function for the Cartesian plane with integer coordinates. -/
def ColoringFunction := ℤ → ℤ → Fin 3

/-- Proposition that a color appears infinitely many times on infinitely many horizontal lines. -/
def InfiniteAppearance (f : ColoringFunction) (c : Fin 3) : Prop :=
  ∀ N : ℕ, ∃ y > N, ∀ M : ℕ, ∃ x > M, f x y = c

/-- Proposition that three points of different colors are not collinear. -/
def NotCollinear (f : ColoringFunction) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ : ℤ,
    f x₁ y₁ ≠ f x₂ y₂ ∧ f x₂ y₂ ≠ f x₃ y₃ ∧ f x₃ y₃ ≠ f x₁ y₁ →
    (x₁ - x₂) * (y₃ - y₂) ≠ (x₃ - x₂) * (y₁ - y₂)

/-- Theorem stating the existence of a valid coloring function. -/
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (∀ c : Fin 3, InfiniteAppearance f c) ∧ NotCollinear f := by
  sorry

end exists_valid_coloring_l3901_390129


namespace marble_selection_probability_l3901_390183

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 3

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles + green_marbles

/-- The number of marbles to be selected -/
def selected_marbles : ℕ := 4

/-- The probability of selecting exactly one marble of each color, with one color being chosen twice -/
theorem marble_selection_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 2 * Nat.choose green_marbles 1 +
   Nat.choose red_marbles 1 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 2) /
  Nat.choose total_marbles selected_marbles = 9 / 14 :=
sorry

end marble_selection_probability_l3901_390183


namespace unique_solution_l3901_390157

theorem unique_solution : 
  ∀ x y : ℤ, (2*x + 5*y + 1)*(2^(Int.natAbs x) + x^2 + x + y) = 105 ↔ x = 0 ∧ y = 4 := by
  sorry

end unique_solution_l3901_390157


namespace gcf_of_48_160_120_l3901_390105

theorem gcf_of_48_160_120 : Nat.gcd 48 (Nat.gcd 160 120) = 8 := by
  sorry

end gcf_of_48_160_120_l3901_390105


namespace triangle_shape_l3901_390190

theorem triangle_shape (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : (Real.cos A + 2 * Real.cos C) / (Real.cos A + 2 * Real.cos B) = Real.sin B / Real.sin C) :
  A = π / 2 ∨ B = C :=
by sorry

end triangle_shape_l3901_390190


namespace tree_age_difference_l3901_390175

/-- The number of rings in one group -/
def rings_per_group : ℕ := 6

/-- The number of ring groups in the first tree -/
def first_tree_groups : ℕ := 70

/-- The number of ring groups in the second tree -/
def second_tree_groups : ℕ := 40

/-- Each ring represents one year of growth -/
axiom ring_year_correspondence : ∀ (n : ℕ), n.succ.pred = n

theorem tree_age_difference : 
  (first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group) = 180 := by
  sorry

end tree_age_difference_l3901_390175


namespace systematic_sampling_calculation_l3901_390199

def population_size : ℕ := 2005
def sample_size : ℕ := 50

theorem systematic_sampling_calculation :
  let sampling_interval := population_size / sample_size
  let discarded := population_size % sample_size
  sampling_interval = 40 ∧ discarded = 5 := by
  sorry

end systematic_sampling_calculation_l3901_390199


namespace penguins_to_feed_l3901_390114

theorem penguins_to_feed (total_penguins : ℕ) (fed_penguins : ℕ) 
  (h1 : total_penguins = 36) 
  (h2 : fed_penguins = 19) : 
  total_penguins - fed_penguins = 17 := by
  sorry

end penguins_to_feed_l3901_390114


namespace difference_of_squares_problem_solution_l3901_390189

theorem difference_of_squares (k : ℝ) : 
  (5 + k) * (5 - k) = 5^2 - k^2 := by sorry

theorem problem_solution : 
  ∃ n : ℝ, (5 + 2) * (5 - 2) = 5^2 - n ∧ n = 2^2 := by sorry

end difference_of_squares_problem_solution_l3901_390189


namespace hope_students_approximation_l3901_390141

/-- Rounds a natural number to the nearest thousand -/
def roundToNearestThousand (n : ℕ) : ℕ :=
  1000 * ((n + 500) / 1000)

/-- The number of students in Hope Primary School -/
def hopeStudents : ℕ := 1996

theorem hope_students_approximation :
  roundToNearestThousand hopeStudents = 2000 := by
  sorry

end hope_students_approximation_l3901_390141


namespace smallest_surface_area_l3901_390172

/-- Represents the dimensions of a cigarette box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  thickness : ℝ
  length_gt_width : length > width
  width_gt_thickness : width > thickness

/-- Calculates the surface area of a rectangular package -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + l * h)

/-- Represents different packaging methods for 10 boxes -/
inductive PackagingMethod
  | method1
  | method2
  | method3
  | method4

/-- Calculates the surface area for a given packaging method -/
def packaging_surface_area (method : PackagingMethod) (box : BoxDimensions) : ℝ :=
  match method with
  | .method1 => surface_area (10 * box.length) box.width box.thickness
  | .method2 => surface_area box.length (10 * box.width) box.thickness
  | .method3 => surface_area box.length box.width (10 * box.thickness)
  | .method4 => surface_area box.length (2 * box.width) (5 * box.thickness)

theorem smallest_surface_area (box : BoxDimensions) 
  (h1 : box.length = 88)
  (h2 : box.width = 58)
  (h3 : box.thickness = 22) :
  (∀ m : PackagingMethod, packaging_surface_area .method4 box ≤ packaging_surface_area m box) ∧ 
  packaging_surface_area .method4 box = 65296 := by
  sorry

#eval surface_area 88 (2 * 58) (5 * 22)

end smallest_surface_area_l3901_390172


namespace solve_pretzel_problem_l3901_390170

def pretzel_problem (barry_pretzels : ℕ) (angie_ratio : ℕ) (shelly_ratio : ℚ) (dave_percentage : ℚ) : Prop :=
  let shelly_pretzels := barry_pretzels * shelly_ratio
  let angie_pretzels := shelly_pretzels * angie_ratio
  let dave_pretzels := (angie_pretzels + shelly_pretzels) * dave_percentage
  angie_pretzels = 18 ∧ dave_pretzels = 6

theorem solve_pretzel_problem :
  pretzel_problem 12 3 (1/2) (1/4) := by
  sorry

end solve_pretzel_problem_l3901_390170


namespace circles_are_intersecting_l3901_390135

/-- Two circles are intersecting if the distance between their centers is greater than the absolute 
    difference of their radii and less than the sum of their radii. -/
def are_intersecting (r₁ r₂ d : ℝ) : Prop :=
  d > |r₁ - r₂| ∧ d < r₁ + r₂

/-- Given two circles with radii 5 and 8, whose centers are 8 units apart, 
    prove that they are intersecting. -/
theorem circles_are_intersecting : are_intersecting 5 8 8 := by
  sorry

end circles_are_intersecting_l3901_390135


namespace initial_balls_count_l3901_390167

theorem initial_balls_count (initial : ℕ) (current : ℕ) (removed : ℕ) : 
  current = 6 → removed = 2 → initial = current + removed → initial = 8 := by
  sorry

end initial_balls_count_l3901_390167


namespace remainder_5n_mod_3_l3901_390155

theorem remainder_5n_mod_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end remainder_5n_mod_3_l3901_390155


namespace polygon_sides_count_l3901_390158

-- Define the properties of the polygon
def is_valid_polygon (n : ℕ) : Prop :=
  n > 2 ∧
  ∀ k : ℕ, k ≤ n → 100 + (k - 1) * 10 < 180

-- Theorem statement
theorem polygon_sides_count : ∃ (n : ℕ), is_valid_polygon n ∧ n = 8 :=
sorry

end polygon_sides_count_l3901_390158


namespace window_area_ratio_l3901_390194

theorem window_area_ratio (AB : ℝ) (h1 : AB = 40) : 
  let AD : ℝ := 3 / 2 * AB
  let rectangle_area : ℝ := AD * AB
  let semicircle_area : ℝ := π * (AB / 2) ^ 2
  rectangle_area / semicircle_area = 6 / π := by sorry

end window_area_ratio_l3901_390194


namespace edward_pipe_per_bolt_l3901_390109

/-- The number of feet of pipe per bolt in Edward's plumbing job -/
def feet_per_bolt (total_pipe_length : ℕ) (washers_used : ℕ) (washers_per_bolt : ℕ) : ℚ :=
  total_pipe_length / (washers_used / washers_per_bolt)

/-- Theorem stating that Edward uses 5 feet of pipe per bolt -/
theorem edward_pipe_per_bolt :
  feet_per_bolt 40 16 2 = 5 := by
  sorry

end edward_pipe_per_bolt_l3901_390109


namespace vertex_of_quadratic_l3901_390198

def f (x : ℝ) : ℝ := -3 * (x - 2)^2

theorem vertex_of_quadratic :
  ∃ (a : ℝ), a < 0 ∧ ∀ (x : ℝ), f x = a * (x - 2)^2 ∧ f 2 = 0 :=
sorry

end vertex_of_quadratic_l3901_390198


namespace calculate_expression_l3901_390191

theorem calculate_expression : (1 + Real.pi) ^ 0 + 2 - |(-3)| + 2 * Real.sin (π / 4) = Real.sqrt 2 := by
  sorry

end calculate_expression_l3901_390191


namespace square_area_remainder_l3901_390152

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Square in 2D space -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is on a side of a square -/
def is_on_side (p : Point) (s : Square) : Prop :=
  (abs (p.x - s.center.x) = s.side_length / 2 ∧ abs (p.y - s.center.y) ≤ s.side_length / 2) ∨
  (abs (p.y - s.center.y) = s.side_length / 2 ∧ abs (p.x - s.center.x) ≤ s.side_length / 2)

theorem square_area_remainder (A B C D : Point) (S : Square) :
  A.x = 0 ∧ A.y = 12 ∧
  B.x = 10 ∧ B.y = 9 ∧
  C.x = 8 ∧ C.y = 0 ∧
  D.x = -4 ∧ D.y = 7 ∧
  is_on_side A S ∧ is_on_side B S ∧ is_on_side C S ∧ is_on_side D S ∧
  (∀ S' : Square, is_on_side A S' ∧ is_on_side B S' ∧ is_on_side C S' ∧ is_on_side D S' → S' = S) →
  (10 * S.side_length ^ 2) % 1000 = 936 := by
  sorry

end square_area_remainder_l3901_390152


namespace approximate_and_scientific_notation_l3901_390178

/-- Determines the place value of the last non-zero digit in a number -/
def lastNonZeroDigitPlace (n : ℕ) : ℕ := sorry

/-- Converts a natural number to scientific notation -/
def toScientificNotation (n : ℕ) : ℝ × ℤ := sorry

theorem approximate_and_scientific_notation :
  (lastNonZeroDigitPlace 24000 = 100) ∧
  (toScientificNotation 46400000 = (4.64, 7)) := by sorry

end approximate_and_scientific_notation_l3901_390178


namespace geometric_sequence_ratio_sum_l3901_390101

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) :
  k ≠ 0 →
  p ≠ 1 →
  r ≠ 1 →
  p ≠ r →
  a₂ = k * p →
  a₃ = k * p^2 →
  b₂ = k * r →
  b₃ = k * r^2 →
  a₃ - b₃ = 4 * (a₂ - b₂) →
  p + r = 4 := by sorry

end geometric_sequence_ratio_sum_l3901_390101


namespace tens_digit_of_23_to_1987_l3901_390108

theorem tens_digit_of_23_to_1987 : ∃ k : ℕ, 23^1987 ≡ 40 + k [ZMOD 100] :=
sorry

end tens_digit_of_23_to_1987_l3901_390108


namespace perfect_square_trinomial_l3901_390138

/-- A quadratic trinomial x^2 + mx + 1 is a perfect square if and only if m = ±2 -/
theorem perfect_square_trinomial (m : ℝ) :
  (∀ x, ∃ a, x^2 + m*x + 1 = (x + a)^2) ↔ (m = 2 ∨ m = -2) :=
sorry

end perfect_square_trinomial_l3901_390138


namespace rohan_salary_l3901_390110

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percentage : ℝ
  rent_percentage : ℝ
  entertainment_percentage : ℝ
  conveyance_percentage : ℝ
  savings : ℝ

/-- Theorem stating Rohan's monthly salary given his expenses and savings -/
theorem rohan_salary (r : RohanFinances) 
  (h1 : r.food_percentage = 0.4)
  (h2 : r.rent_percentage = 0.2)
  (h3 : r.entertainment_percentage = 0.1)
  (h4 : r.conveyance_percentage = 0.1)
  (h5 : r.savings = 2000)
  (h6 : r.savings = r.salary * (1 - (r.food_percentage + r.rent_percentage + r.entertainment_percentage + r.conveyance_percentage))) :
  r.salary = 10000 := by
  sorry

#check rohan_salary

end rohan_salary_l3901_390110


namespace floor_product_equation_l3901_390146

theorem floor_product_equation (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 29 ↔ x ≥ 5.8 ∧ x < 6 :=
sorry

end floor_product_equation_l3901_390146


namespace lola_baked_eight_pies_l3901_390131

/-- Represents the number of pastries baked by Lola and Lulu -/
structure Pastries where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by both Lola and Lulu -/
def total_pastries (p : Pastries) : ℕ :=
  p.lola_cupcakes + p.lola_poptarts + p.lola_pies +
  p.lulu_cupcakes + p.lulu_poptarts + p.lulu_pies

/-- Theorem stating that Lola baked 8 blueberry pies -/
theorem lola_baked_eight_pies (p : Pastries)
  (h1 : p.lola_cupcakes = 13)
  (h2 : p.lola_poptarts = 10)
  (h3 : p.lulu_cupcakes = 16)
  (h4 : p.lulu_poptarts = 12)
  (h5 : p.lulu_pies = 14)
  (h6 : total_pastries p = 73) :
  p.lola_pies = 8 := by
  sorry

end lola_baked_eight_pies_l3901_390131


namespace range_of_expression_l3901_390127

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (z : ℝ), z = 4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ∧ 1 ≤ z ∧ z ≤ 22 + 4*Real.sqrt 5 := by
  sorry

end range_of_expression_l3901_390127


namespace geometric_sequence_special_case_l3901_390162

/-- A geometric sequence with first term 1 and nth term equal to the product of the first 5 terms has n = 11 -/
theorem geometric_sequence_special_case (a : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) / a k = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- first term is 1
  a n = a 1 * a 2 * a 3 * a 4 * a 5 →   -- nth term equals product of first 5 terms
  n = 11 := by
sorry

end geometric_sequence_special_case_l3901_390162


namespace coins_after_fifth_hour_l3901_390102

def coins_in_jar (hour1 : ℕ) (hour2_3 : ℕ) (hour4 : ℕ) (taken_out : ℕ) : ℕ :=
  hour1 + 2 * hour2_3 + hour4 - taken_out

theorem coins_after_fifth_hour :
  coins_in_jar 20 30 40 20 = 100 := by
  sorry

end coins_after_fifth_hour_l3901_390102


namespace sin_alpha_value_l3901_390145

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α + π/2) = 3/5) : Real.sin α = 4/5 := by
  sorry

end sin_alpha_value_l3901_390145


namespace total_dragons_is_eight_l3901_390147

/-- Represents the number of heads on a dragon -/
inductive DragonHeads
  | two
  | seven

/-- Counts the total number of dragons given the conditions of the problem -/
def count_dragons : Nat :=
  let two_headed := 6
  let seven_headed := 2
  two_headed + seven_headed

/-- The main theorem stating that the total number of dragons is 8 -/
theorem total_dragons_is_eight :
  count_dragons = 8 ∧
  ∃ (x y : Nat),
    x * 2 + y * 7 = 25 + 7 ∧  -- Total heads including the counting head
    x + y = count_dragons ∧
    x ≥ 0 ∧ y > 0 :=
by sorry

end total_dragons_is_eight_l3901_390147


namespace original_price_of_discounted_shoes_l3901_390112

/-- Given a pair of shoes sold at a 20% discount for $480, prove that its original price was $600. -/
theorem original_price_of_discounted_shoes (discount_rate : ℝ) (discounted_price : ℝ) : 
  discount_rate = 0.20 → discounted_price = 480 → (1 - discount_rate) * 600 = discounted_price := by
  sorry

end original_price_of_discounted_shoes_l3901_390112


namespace other_root_of_complex_polynomial_l3901_390106

theorem other_root_of_complex_polynomial (m n : ℝ) :
  (Complex.I + 3) ^ 2 + m * (Complex.I + 3) + n = 0 →
  (3 - Complex.I) ^ 2 + m * (3 - Complex.I) + n = 0 := by
  sorry

end other_root_of_complex_polynomial_l3901_390106


namespace ellipse_k_value_l3901_390116

/-- An ellipse with equation 5x^2 + ky^2 = 5 and one focus at (0, 2) has k = 1 -/
theorem ellipse_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 5 * x^2 + k * y^2 = 5) →  -- Equation of the ellipse
  (∃ (c : ℝ), c^2 = 5/k - 1) →            -- Property of ellipse: c^2 = a^2 - b^2
  (2 : ℝ)^2 = 5/k - 1 →                   -- Focus at (0, 2)
  k = 1 := by
sorry


end ellipse_k_value_l3901_390116


namespace point_movement_on_number_line_l3901_390187

theorem point_movement_on_number_line : 
  let start : ℤ := -2
  let move_right : ℤ := 7
  let move_left : ℤ := 4
  start + move_right - move_left = 1 :=
by sorry

end point_movement_on_number_line_l3901_390187


namespace hexahedron_octahedron_volume_ratio_l3901_390123

theorem hexahedron_octahedron_volume_ratio :
  ∀ (a b : ℝ),
    a > 0 → b > 0 →
    6 * a^2 = 2 * Real.sqrt 3 * b^2 →
    (a^3) / ((Real.sqrt 2 / 3) * b^3) = 3 / Real.sqrt (6 * Real.sqrt 3) :=
by sorry

end hexahedron_octahedron_volume_ratio_l3901_390123


namespace intersection_of_A_and_B_l3901_390113

def A : Set ℝ := {x | 2 * x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l3901_390113


namespace octal_127_equals_binary_1010111_l3901_390151

def octal_to_decimal (octal : ℕ) : ℕ := 
  (octal % 10) + 8 * ((octal / 10) % 10) + 64 * (octal / 100)

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem octal_127_equals_binary_1010111 : 
  decimal_to_binary (octal_to_decimal 127) = [1, 0, 1, 0, 1, 1, 1] := by
  sorry

end octal_127_equals_binary_1010111_l3901_390151


namespace vertical_shift_theorem_l3901_390153

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define a constant for vertical shift
variable (k : ℝ)

-- Define the shifted function
def shifted_f (x : ℝ) : ℝ := f x + k

-- Theorem: The graph of y = f(x) + k is a vertical shift of y = f(x) by k units
theorem vertical_shift_theorem :
  ∀ (x y : ℝ), y = shifted_f f k x ↔ y - k = f x :=
by sorry

end vertical_shift_theorem_l3901_390153


namespace quadratic_roots_real_and_equal_l3901_390186

theorem quadratic_roots_real_and_equal :
  let a : ℝ := 1
  let b : ℝ := -4 * Real.sqrt 2
  let c : ℝ := 8
  let discriminant := b^2 - 4*a*c
  discriminant = 0 ∧ ∃ x : ℝ, x^2 - 4*x*(Real.sqrt 2) + 8 = 0 := by
  sorry

end quadratic_roots_real_and_equal_l3901_390186


namespace quadratic_discriminant_l3901_390143

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of x^2 + x - 2 = 0 is 9 -/
theorem quadratic_discriminant :
  discriminant 1 1 (-2) = 9 := by
  sorry

end quadratic_discriminant_l3901_390143


namespace multiplier_proof_l3901_390149

theorem multiplier_proof (number : ℝ) (difference : ℝ) (subtractor : ℝ) :
  number = 15.0 →
  difference = 40 →
  subtractor = 5 →
  ∃ (multiplier : ℝ), multiplier * number - subtractor = difference ∧ multiplier = 3 := by
  sorry

end multiplier_proof_l3901_390149


namespace five_by_seven_double_covered_cells_l3901_390111

/-- Represents a rectangular grid with fold lines -/
structure FoldableGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (foldLines : List (ℕ × ℕ × ℕ × ℕ))  -- List of start and end points of fold lines

/-- Counts the number of cells covered exactly twice after folding -/
def countDoubleCoveredCells (grid : FoldableGrid) : ℕ :=
  sorry

/-- The main theorem stating that a 5x7 grid with specific fold lines has 9 double-covered cells -/
theorem five_by_seven_double_covered_cells :
  ∃ (foldLines : List (ℕ × ℕ × ℕ × ℕ)),
    let grid := FoldableGrid.mk 5 7 foldLines
    countDoubleCoveredCells grid = 9 :=
  sorry

end five_by_seven_double_covered_cells_l3901_390111


namespace operation_problem_l3901_390159

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (diamond circ : Operation) 
  (h : (applyOp diamond 15 3) / (applyOp circ 8 2) = 3) :
  (applyOp diamond 9 4) / (applyOp circ 14 7) = 13/7 := by
  sorry

end operation_problem_l3901_390159


namespace geometric_sequences_l3901_390188

def is_geometric (a : ℕ → ℝ) : Prop := ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequences (a : ℕ → ℝ) (h : is_geometric a) :
  (is_geometric (λ n => 1 / a n)) ∧ (is_geometric (λ n => a n * a (n + 1))) := by sorry

end geometric_sequences_l3901_390188


namespace job_candidate_probability_l3901_390173

theorem job_candidate_probability 
  (p_excel : ℝ) 
  (p_day_shift : ℝ) 
  (h_excel : p_excel = 0.2) 
  (h_day_shift : p_day_shift = 0.7) : 
  p_excel * (1 - p_day_shift) = 0.06 := by
sorry

end job_candidate_probability_l3901_390173


namespace alex_jamie_pairing_probability_l3901_390169

/-- The number of students participating in the event -/
def total_students : ℕ := 32

/-- The probability of Alex being paired with Jamie in a random pairing -/
def probability_alex_jamie : ℚ := 1 / 31

/-- Theorem stating that the probability of Alex being paired with Jamie
    in a random pairing of 32 students is 1/31 -/
theorem alex_jamie_pairing_probability :
  probability_alex_jamie = 1 / (total_students - 1) :=
sorry

end alex_jamie_pairing_probability_l3901_390169


namespace odd_number_characterization_l3901_390150

theorem odd_number_characterization (n : ℤ) : 
  Odd n ↔ ∃ k : ℤ, n = 2 * k + 1 :=
sorry

end odd_number_characterization_l3901_390150


namespace coin_flip_solution_l3901_390174

def coin_flip_problem (n : ℕ) : Prop :=
  let p_tails : ℚ := 1/2
  let p_sequence : ℚ := 0.0625
  (p_tails ^ 2 * (1 - p_tails) ^ 2 = p_sequence) ∧ (n = 4)

theorem coin_flip_solution :
  ∃ n : ℕ, coin_flip_problem n :=
sorry

end coin_flip_solution_l3901_390174


namespace tamara_height_l3901_390132

/-- Given the heights of Tamara, Kim, and Gavin, prove Tamara's height is 95 inches -/
theorem tamara_height (kim : ℝ) : 
  let tamara := 3 * kim - 4
  let gavin := 2 * kim + 6
  (3 * kim - 4) + kim + (2 * kim + 6) = 200 → tamara = 95 := by
sorry

end tamara_height_l3901_390132


namespace max_red_socks_l3901_390156

theorem max_red_socks (r b : ℕ) : 
  let t := r + b
  (t ≤ 2023) →
  (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 2 / 5 →
  r ≤ 990 ∧ ∃ (r' : ℕ), r' = 990 ∧ 
    ∃ (b' : ℕ), (r' + b' ≤ 2023) ∧ 
    (r' * (r' - 1) + b' * (b' - 1)) / ((r' + b') * (r' + b' - 1)) = 2 / 5 :=
by sorry

end max_red_socks_l3901_390156


namespace infinitely_many_primes_4k_minus_1_l3901_390139

theorem infinitely_many_primes_4k_minus_1 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k - 1} := by
  sorry

end infinitely_many_primes_4k_minus_1_l3901_390139


namespace max_daily_revenue_l3901_390119

def P (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 70
  else 0

def Q (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def dailyRevenue (t : ℕ) : ℝ := P t * Q t

theorem max_daily_revenue :
  (∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ dailyRevenue t = 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 → dailyRevenue t ≤ 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 ∧ dailyRevenue t = 1125 → t = 25) :=
by sorry

end max_daily_revenue_l3901_390119


namespace negation_of_universal_proposition_l3901_390107

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x) ↔ (∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x) := by
  sorry

end negation_of_universal_proposition_l3901_390107


namespace exists_positive_sum_G2_l3901_390126

-- Define the grid as a function from pairs of integers to real numbers
def Grid := ℤ × ℤ → ℝ

-- Define a shape as a finite set of integer pairs (representing cell positions)
def Shape := Finset (ℤ × ℤ)

-- Define the sum of numbers covered by a shape at a given position
def shapeSum (g : Grid) (s : Shape) (pos : ℤ × ℤ) : ℝ :=
  s.sum (λ (x, y) => g (x + pos.1, y + pos.2))

-- State the theorem
theorem exists_positive_sum_G2 (g : Grid) (G1 G2 : Shape) 
  (h : ∀ pos : ℤ × ℤ, shapeSum g G1 pos > 0) :
  ∃ pos : ℤ × ℤ, shapeSum g G2 pos > 0 := by
  sorry

end exists_positive_sum_G2_l3901_390126


namespace base8_subtraction_result_l3901_390142

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Represents subtraction in base-8 --/
def base8_subtraction (a b : ℕ) : ℤ := sorry

theorem base8_subtraction_result : 
  base8_subtraction (base8_to_base10 46) (base8_to_base10 63) = -13 := by sorry

end base8_subtraction_result_l3901_390142


namespace computer_price_comparison_l3901_390134

theorem computer_price_comparison (price1 : ℝ) (discount1 : ℝ) (discount2 : ℝ) (price_diff : ℝ) : 
  price1 = 950 ∧ 
  discount1 = 0.06 ∧ 
  discount2 = 0.05 ∧ 
  price_diff = 19 →
  ∃ (price2 : ℝ), 
    price2 * (1 - discount2) = price1 * (1 - discount1) + price_diff ∧ 
    price2 = 960 := by
  sorry

end computer_price_comparison_l3901_390134


namespace logarithm_expression_equality_l3901_390115

theorem logarithm_expression_equality : 
  (Real.log 243 / Real.log 3) / (Real.log 81 / Real.log 3) - 
  (Real.log 729 / Real.log 3) / (Real.log 27 / Real.log 3) = 2 := by
  sorry

end logarithm_expression_equality_l3901_390115


namespace thursday_coffee_consumption_l3901_390133

/-- Represents the relationship between coffee consumption, sleep, and preparation time -/
def coffee_relation (k : ℝ) (c h p : ℝ) : Prop :=
  c * (h + p) = k

theorem thursday_coffee_consumption 
  (k : ℝ)
  (c_wed h_wed p_wed : ℝ)
  (h_thu p_thu : ℝ)
  (hw : coffee_relation k c_wed h_wed p_wed)
  (wed_data : c_wed = 3 ∧ h_wed = 8 ∧ p_wed = 2)
  (thu_data : h_thu = 5 ∧ p_thu = 3) :
  ∃ c_thu : ℝ, coffee_relation k c_thu h_thu p_thu ∧ c_thu = 15/4 := by
  sorry

end thursday_coffee_consumption_l3901_390133


namespace cookie_boxes_problem_l3901_390103

theorem cookie_boxes_problem (n : ℕ) : 
  n - 7 ≥ 1 → 
  n - 2 ≥ 1 → 
  (n - 7) + (n - 2) < n → 
  n = 8 :=
by sorry

end cookie_boxes_problem_l3901_390103


namespace lanas_tulips_l3901_390195

/-- The number of tulips Lana picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Lana picked -/
def total_flowers : ℕ := sorry

/-- The number of flowers used for bouquets -/
def used_flowers : ℕ := 70

/-- The number of leftover flowers -/
def leftover_flowers : ℕ := 3

/-- The number of roses Lana picked -/
def roses : ℕ := 37

theorem lanas_tulips :
  (total_flowers = tulips + roses) ∧
  (total_flowers = used_flowers + leftover_flowers) →
  tulips = 36 := by sorry

end lanas_tulips_l3901_390195


namespace basketball_lineup_combinations_l3901_390181

theorem basketball_lineup_combinations (n : Nat) (k : Nat) (m : Nat) : 
  n = 20 → k = 13 → m = 1 →
  n * Nat.choose (n - 1) (k - m) = 1007760 := by
  sorry

end basketball_lineup_combinations_l3901_390181


namespace digit_difference_in_base_d_l3901_390197

/-- Given two digits A and B in base d > 7 such that AB̅_d + AA̅_d = 172_d, prove that A_d - B_d = 5 --/
theorem digit_difference_in_base_d (d : ℕ) (A B : ℕ) (h1 : d > 7) 
  (h2 : A < d ∧ B < d) 
  (h3 : A * d + B + A * d + A = 1 * d^2 + 7 * d + 2) : 
  A - B = 5 := by
  sorry

end digit_difference_in_base_d_l3901_390197


namespace least_tablets_extracted_l3901_390177

theorem least_tablets_extracted (tablets_a tablets_b : ℕ) 
  (ha : tablets_a = 10) (hb : tablets_b = 16) :
  ∃ (n : ℕ), n ≤ tablets_a + tablets_b ∧ 
  (∀ (k : ℕ), k < n → 
    (k < tablets_a + 2 → ∃ (x y : ℕ), x + y = k ∧ (x < 2 ∨ y < 2)) ∧
    (k ≥ tablets_a + 2 → ∃ (x y : ℕ), x + y = k ∧ x ≥ 2 ∧ y ≥ 2)) ∧
  n = 12 :=
sorry

end least_tablets_extracted_l3901_390177


namespace r_fourth_plus_inverse_r_fourth_l3901_390121

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inverse_r_fourth_l3901_390121


namespace tangent_line_equation_l3901_390163

-- Define the curve
def f (x : ℝ) : ℝ := 2*x - x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2 - 3*x^2

-- Define the point of tangency
def x₀ : ℝ := -1
def y₀ : ℝ := f x₀

-- Define the slope of the tangent line at x₀
def m : ℝ := f' x₀

-- Statement: The equation of the tangent line is x + y + 2 = 0
theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ x + y + 2 = 0 :=
by sorry

end tangent_line_equation_l3901_390163


namespace payment_bill_value_l3901_390136

/-- Proves the value of a single bill used for payment given the number of items,
    cost per item, number of change bills, and value of each change bill. -/
theorem payment_bill_value
  (num_games : ℕ)
  (cost_per_game : ℕ)
  (num_change_bills : ℕ)
  (change_bill_value : ℕ)
  (h1 : num_games = 6)
  (h2 : cost_per_game = 15)
  (h3 : num_change_bills = 2)
  (h4 : change_bill_value = 5) :
  num_games * cost_per_game + num_change_bills * change_bill_value = 100 := by
  sorry

#check payment_bill_value

end payment_bill_value_l3901_390136


namespace updated_mean_after_decrement_l3901_390193

def original_mean : ℝ := 250
def num_observations : ℕ := 100
def decrement : ℝ := 20

theorem updated_mean_after_decrement :
  (original_mean * num_observations - decrement * num_observations) / num_observations = 230 := by
  sorry

end updated_mean_after_decrement_l3901_390193


namespace parallel_lines_imply_a_equals_3_l3901_390137

-- Define the equations of the lines
def line1 (a x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def line2 (a x y : ℝ) : Prop := 2 * x + (a - 1) * y + 1 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_imply_a_equals_3 (a : ℝ) :
  parallel (line1 a) (line2 a) → a = 3 :=
by sorry

end parallel_lines_imply_a_equals_3_l3901_390137


namespace fraction_sum_equals_three_tenths_l3901_390118

theorem fraction_sum_equals_three_tenths : 
  (1 : ℚ) / 10 + (2 : ℚ) / 20 + (3 : ℚ) / 30 = (3 : ℚ) / 10 := by
  sorry

end fraction_sum_equals_three_tenths_l3901_390118


namespace tape_shortage_l3901_390179

/-- The amount of tape Joy has -/
def tape_amount : ℕ := 180

/-- The width of the field -/
def field_width : ℕ := 35

/-- The length of the field -/
def field_length : ℕ := 80

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem tape_shortage : 
  rectangle_perimeter field_length field_width = tape_amount + 50 := by
  sorry

end tape_shortage_l3901_390179


namespace system_inequalities_solution_set_l3901_390100

theorem system_inequalities_solution_set (m : ℝ) :
  (∀ x : ℝ, x > 4 ∧ x > m ↔ x > 4) ↔ m ≤ 4 := by
  sorry

end system_inequalities_solution_set_l3901_390100


namespace swimmers_passing_theorem_l3901_390154

/-- Represents the number of times two swimmers pass each other in a pool --/
def swimmers_passing_count (pool_length : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of times swimmers pass each other under given conditions --/
theorem swimmers_passing_theorem :
  let pool_length : ℝ := 100
  let speed1 : ℝ := 4
  let speed2 : ℝ := 3
  let total_time : ℝ := 15 * 60  -- 15 minutes in seconds
  swimmers_passing_count pool_length speed1 speed2 total_time = 27 := by
  sorry

end swimmers_passing_theorem_l3901_390154


namespace cylinder_volume_l3901_390182

/-- The volume of a cylinder with height 4 and circular faces with circumference 10π is 100π. -/
theorem cylinder_volume (h : ℝ) (c : ℝ) (v : ℝ) : 
  h = 4 → c = 10 * Real.pi → v = Real.pi * (c / (2 * Real.pi))^2 * h → v = 100 * Real.pi := by
  sorry

end cylinder_volume_l3901_390182


namespace inequality_range_l3901_390166

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ a ∈ Set.Ici 3 := by
sorry

end inequality_range_l3901_390166


namespace concatenated_numbers_remainder_l3901_390124

-- Define a function to concatenate numbers from 1 to n
def concatenateNumbers (n : ℕ) : ℕ := sorry

-- Define a function to calculate the remainder when a number is divided by 9
def remainderMod9 (n : ℕ) : ℕ := n % 9

-- Theorem statement
theorem concatenated_numbers_remainder (n : ℕ) (h : n = 2001) :
  remainderMod9 (concatenateNumbers n) = 6 := by sorry

end concatenated_numbers_remainder_l3901_390124


namespace club_leader_selection_l3901_390161

/-- Represents a club with members of two genders, some wearing glasses -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  boys_with_glasses : Nat
  girls_with_glasses : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def ways_to_choose_leaders (c : Club) : Nat :=
  (c.boys_with_glasses * (c.boys_with_glasses - 1)) +
  (c.girls_with_glasses * (c.girls_with_glasses - 1))

/-- The main theorem to prove -/
theorem club_leader_selection (c : Club) 
  (h1 : c.total_members = 24)
  (h2 : c.boys = 12)
  (h3 : c.girls = 12)
  (h4 : c.boys_with_glasses = 6)
  (h5 : c.girls_with_glasses = 6) :
  ways_to_choose_leaders c = 60 := by
  sorry

#eval ways_to_choose_leaders { total_members := 24, boys := 12, girls := 12, boys_with_glasses := 6, girls_with_glasses := 6 }

end club_leader_selection_l3901_390161


namespace correct_sums_l3901_390120

theorem correct_sums (total : ℕ) (h1 : total = 75) : ∃ (right : ℕ), right * 3 = total ∧ right = 25 := by
  sorry

end correct_sums_l3901_390120


namespace D_72_l3901_390185

/-- The number of ways to write a positive integer as a product of integers greater than 1, considering order. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- The prime factorization of 72 is 2^3 * 3^2 -/
axiom prime_factorization_72 : ∃ (a b : ℕ), 72 = 2^3 * 3^2

/-- Theorem: The number of ways to write 72 as a product of integers greater than 1, considering order, is 26 -/
theorem D_72 : D 72 = 26 := by
  sorry

end D_72_l3901_390185


namespace simplify_fraction_l3901_390122

theorem simplify_fraction (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  (15 * a^4 * b) / (75 * a^3 * b^2) = 2 / 15 := by
  sorry

end simplify_fraction_l3901_390122
