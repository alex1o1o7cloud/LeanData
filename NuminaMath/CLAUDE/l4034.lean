import Mathlib

namespace no_largest_non_expressible_l4034_403481

-- Define a function to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the property of being expressible as the sum of a multiple of 36 and a non-square
def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 36 * a + b ∧ b > 0 ∧ ¬(is_square b)

-- Theorem statement
theorem no_largest_non_expressible :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ¬(is_expressible m) :=
sorry

end no_largest_non_expressible_l4034_403481


namespace integer_fraction_sum_l4034_403428

theorem integer_fraction_sum (n : ℤ) (h : n ≥ 8) :
  (∃ k : ℤ, n + 1 / (n - 7) = k) ↔ n = 8 := by
  sorry

end integer_fraction_sum_l4034_403428


namespace tangent_slope_implies_a_value_l4034_403417

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_slope_implies_a_value :
  ∀ a b : ℝ, f_derivative a 1 = -1 → a = 2 := by
  sorry

end tangent_slope_implies_a_value_l4034_403417


namespace population_ratio_l4034_403444

/-- Given three cities X, Y, and Z, where the population of X is 5 times that of Y,
    and the population of Y is twice that of Z, prove that the ratio of the
    population of X to Z is 10:1 -/
theorem population_ratio (x y z : ℕ) (hxy : x = 5 * y) (hyz : y = 2 * z) :
  x / z = 10 := by
  sorry

end population_ratio_l4034_403444


namespace books_remaining_l4034_403475

theorem books_remaining (initial_books given_away : ℝ) 
  (h1 : initial_books = 54.0)
  (h2 : given_away = 23.0) : 
  initial_books - given_away = 31.0 := by
sorry

end books_remaining_l4034_403475


namespace fifteen_factorial_trailing_zeros_l4034_403420

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of trailing zeros in n when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Base 18 expressed as 2 · 3² -/
def base18 : ℕ := 2 * 3^2

/-- The main theorem -/
theorem fifteen_factorial_trailing_zeros :
  trailingZeros (factorial 15) base18 = 3 := by sorry

end fifteen_factorial_trailing_zeros_l4034_403420


namespace sams_basketball_score_l4034_403478

theorem sams_basketball_score (total : ℕ) (friend_score : ℕ) (sam_score : ℕ) :
  total = 87 →
  friend_score = 12 →
  total = sam_score + friend_score →
  sam_score = 75 :=
by
  sorry

end sams_basketball_score_l4034_403478


namespace shortest_paths_count_l4034_403486

/-- The number of shortest paths on a chess board from (0,0) to (m,n) -/
def numShortestPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of shortest paths from (0,0) to (m,n) on a chess board,
    where movement is restricted to coordinate axis directions and
    direction changes only at integer coordinates, is equal to (m+n choose m) -/
theorem shortest_paths_count (m n : ℕ) :
  numShortestPaths m n = Nat.choose (m + n) m := by
  sorry

end shortest_paths_count_l4034_403486


namespace hyperbola_ratio_l4034_403458

theorem hyperbola_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (3^2 / a^2 - (3 * Real.sqrt 2)^2 / b^2 = 1) →
  (Real.tan (45 * π / 360) = b / a) →
  (a / b = Real.sqrt 2 + 1) := by
sorry

end hyperbola_ratio_l4034_403458


namespace infinite_solutions_in_interval_l4034_403415

theorem infinite_solutions_in_interval (x : Real) (h : x ∈ Set.Icc 0 (2 * Real.pi)) :
  Real.cos ((Real.pi / 2) * Real.cos x + (Real.pi / 2) * Real.sin x) =
  Real.sin ((Real.pi / 2) * Real.cos x - (Real.pi / 2) * Real.sin x) := by
  sorry

end infinite_solutions_in_interval_l4034_403415


namespace hidden_faces_sum_l4034_403448

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 1, 2, 3, 4, 5, 6, 6, 5]

def total_faces : ℕ := 24

theorem hidden_faces_sum :
  let total_dots := 4 * standard_die_sum
  let visible_dots := visible_faces.sum
  let hidden_faces := total_faces - visible_faces.length
  hidden_faces = 15 ∧ total_dots - visible_dots = 51 := by sorry

end hidden_faces_sum_l4034_403448


namespace sphere_division_l4034_403425

/-- The maximum number of parts into which the surface of a sphere can be divided by n great circles -/
def max_sphere_parts (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating that max_sphere_parts gives the correct number of maximum parts -/
theorem sphere_division (n : ℕ) :
  max_sphere_parts n = n^2 - n + 2 := by sorry

end sphere_division_l4034_403425


namespace complement_intersection_theorem_l4034_403429

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ 0 ∨ x > 3}
def Q : Set ℝ := {0, 1, 2, 3}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl P) ∩ Q = {1, 2, 3} := by sorry

end complement_intersection_theorem_l4034_403429


namespace no_multiples_of_5005_l4034_403452

theorem no_multiples_of_5005 : ¬∃ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 49 ∧ 
  ∃ (k : ℕ+), 5005 * k = 10^j - 10^i := by
  sorry

end no_multiples_of_5005_l4034_403452


namespace ratio_a_to_c_l4034_403437

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 1 := by
sorry

end ratio_a_to_c_l4034_403437


namespace book_chunk_sheets_l4034_403451

/-- Checks if two numbers have the same digits (possibly in different order) -/
def sameDigits (a b : Nat) : Bool := sorry

/-- Finds the smallest even number greater than n composed of the same digits as n -/
def smallestEvenWithSameDigits (n : Nat) : Nat := sorry

theorem book_chunk_sheets (first_page last_page : Nat) : 
  first_page = 163 →
  last_page = smallestEvenWithSameDigits first_page →
  (last_page - first_page + 1) / 2 = 77 := by sorry

end book_chunk_sheets_l4034_403451


namespace math_competition_problem_l4034_403473

theorem math_competition_problem (n : ℕ) (S : Finset (Finset (Fin 6))) :
  (∀ (i j : Fin 6), i ≠ j → (S.filter (λ s => i ∈ s ∧ j ∈ s)).card > (2 * S.card) / 5) →
  (∀ s ∈ S, s.card ≤ 5) →
  (S.filter (λ s => s.card = 5)).card ≥ 2 :=
sorry

end math_competition_problem_l4034_403473


namespace y_coordinate_of_C_is_18_l4034_403413

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- Theorem: The y-coordinate of vertex C in the given pentagon is 18 -/
theorem y_coordinate_of_C_is_18 (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 6))
  (h3 : p.D = (6, 6))
  (h4 : p.E = (6, 0))
  (h5 : hasVerticalSymmetry p)
  (h6 : pentagonArea p = 72)
  : p.C.2 = 18 := by sorry

end y_coordinate_of_C_is_18_l4034_403413


namespace choose_three_from_seven_l4034_403411

/-- The number of ways to choose 3 distinct people from a group of 7 to fill 3 distinct positions -/
def ways_to_choose_officers (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2)

/-- Theorem stating that choosing 3 distinct people from a group of 7 to fill 3 distinct positions can be done in 210 ways -/
theorem choose_three_from_seven :
  ways_to_choose_officers 7 = 210 := by
  sorry

end choose_three_from_seven_l4034_403411


namespace polynomial_division_quotient_l4034_403424

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 9 * X^4 + 8 * X^3 - 12 * X^2 - 7 * X + 4
  let divisor : Polynomial ℚ := 3 * X^2 + 2 * X + 5
  let quotient : Polynomial ℚ := 3 * X^2 - 2 * X + 2
  (dividend.div divisor) = quotient := by sorry

end polynomial_division_quotient_l4034_403424


namespace polynomial_remainder_l4034_403438

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 8*x^3 + 15*x^2 + 12*x - 20
  let g : ℝ → ℝ := λ x => x - 2
  (f 2) = 16 := by sorry

end polynomial_remainder_l4034_403438


namespace cubes_fill_box_l4034_403403

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the size of a cube -/
def CubeSize : ℕ := 2

/-- Calculate the number of cubes that can fit along a given dimension -/
def cubesAlongDimension (dimension : ℕ) : ℕ :=
  dimension / CubeSize

/-- Calculate the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) : ℕ :=
  (cubesAlongDimension box.length) * (cubesAlongDimension box.width) * (cubesAlongDimension box.height)

/-- Calculate the volume of the box -/
def boxVolume (box : BoxDimensions) : ℕ :=
  box.length * box.width * box.height

/-- Calculate the volume occupied by the cubes -/
def cubesVolume (box : BoxDimensions) : ℕ :=
  totalCubes box * (CubeSize * CubeSize * CubeSize)

/-- The main theorem: The volume occupied by cubes is equal to the box volume -/
theorem cubes_fill_box (box : BoxDimensions) 
  (h1 : box.length = 8) (h2 : box.width = 6) (h3 : box.height = 12) : 
  cubesVolume box = boxVolume box := by
  sorry

#check cubes_fill_box

end cubes_fill_box_l4034_403403


namespace quadratic_value_at_5_l4034_403446

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value_at_5 
  (a b c : ℝ) 
  (max_at_2 : ∀ x, quadratic a b c x ≤ quadratic a b c 2)
  (max_value : quadratic a b c 2 = 6)
  (passes_origin : quadratic a b c 0 = -10) :
  quadratic a b c 5 = -30 := by
sorry

end quadratic_value_at_5_l4034_403446


namespace equation_equivalence_l4034_403464

theorem equation_equivalence : ∀ x : ℝ, x * (x + 2) = 5 ↔ x^2 + 2*x - 5 = 0 := by sorry

end equation_equivalence_l4034_403464


namespace fish_population_estimate_l4034_403414

/-- Represents the data from a single round of fish catching --/
structure RoundData where
  caught : Nat
  tagged : Nat

/-- Represents the data from the fish population study --/
structure FishStudy where
  round1 : RoundData
  round2 : RoundData
  round3 : RoundData

/-- The Lincoln-Petersen estimator function --/
def lincolnPetersen (c1 c2 r2 : Nat) : Nat :=
  (c1 * c2) / r2

/-- Theorem stating that the estimated fish population is 800 --/
theorem fish_population_estimate (study : FishStudy)
    (h1 : study.round1 = { caught := 30, tagged := 0 })
    (h2 : study.round2 = { caught := 80, tagged := 6 })
    (h3 : study.round3 = { caught := 100, tagged := 10 }) :
    lincolnPetersen study.round2.caught study.round3.caught study.round3.tagged = 800 := by
  sorry


end fish_population_estimate_l4034_403414


namespace bisector_quadrilateral_is_square_l4034_403433

/-- A rectangle that is not a square -/
structure NonSquareRectangle where
  length : ℝ
  width : ℝ
  length_positive : 0 < length
  width_positive : 0 < width
  not_square : length ≠ width

/-- The quadrilateral formed by the intersection of angle bisectors -/
structure BisectorQuadrilateral (r : NonSquareRectangle) where
  vertices : Fin 4 → ℝ × ℝ

/-- Theorem: The quadrilateral formed by the intersection of angle bisectors in a non-square rectangle is a square -/
theorem bisector_quadrilateral_is_square (r : NonSquareRectangle) (q : BisectorQuadrilateral r) :
  IsSquare q.vertices := by sorry

end bisector_quadrilateral_is_square_l4034_403433


namespace polyhedron_sum_l4034_403449

/-- A convex polyhedron with triangular and pentagonal faces -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  triangular_faces : ℕ
  pentagonal_faces : ℕ
  T : ℕ  -- number of triangular faces meeting at each vertex
  P : ℕ  -- number of pentagonal faces meeting at each vertex
  faces_sum : faces = triangular_faces + pentagonal_faces
  faces_32 : faces = 32
  vertex_relation : vertices * (T + P - 2) = 60
  face_relation : 5 * vertices * T + 3 * vertices * P = 480

/-- The sum of P, T, and V for the specific polyhedron is 34 -/
theorem polyhedron_sum (poly : ConvexPolyhedron) : poly.P + poly.T + poly.vertices = 34 := by
  sorry

end polyhedron_sum_l4034_403449


namespace exists_fourth_root_of_3_to_20_l4034_403418

theorem exists_fourth_root_of_3_to_20 : ∃ n : ℕ, n^4 = 3^20 ∧ n = 243 := by sorry

end exists_fourth_root_of_3_to_20_l4034_403418


namespace cube_with_holes_surface_area_l4034_403454

/-- Calculates the total surface area of a cube with holes --/
def cube_surface_area_with_holes (cube_edge_length : ℝ) (hole_side_length : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge_length^2
  let hole_area := 6 * hole_side_length^2
  let exposed_internal_area := 6 * 4 * hole_side_length^2
  original_surface_area - hole_area + exposed_internal_area

/-- Theorem stating the total surface area of the cube with holes --/
theorem cube_with_holes_surface_area :
  cube_surface_area_with_holes 5 2 = 222 := by
  sorry

end cube_with_holes_surface_area_l4034_403454


namespace any_nonzero_rational_to_zero_power_is_one_l4034_403477

theorem any_nonzero_rational_to_zero_power_is_one (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end any_nonzero_rational_to_zero_power_is_one_l4034_403477


namespace farmer_sheep_problem_l4034_403455

theorem farmer_sheep_problem (total : ℕ) 
  (h1 : total % 3 = 0)  -- First son's share is whole
  (h2 : total % 5 = 0)  -- Second son's share is whole
  (h3 : total % 6 = 0)  -- Third son's share is whole
  (h4 : total % 8 = 0)  -- Daughter's share is whole
  (h5 : total - (total / 3 + total / 5 + total / 6 + total / 8) = 12)  -- Charity's share
  : total = 68 := by
  sorry

end farmer_sheep_problem_l4034_403455


namespace z_in_fourth_quadrant_l4034_403401

-- Define the complex number z
def z : ℂ := (2 - Complex.I) * (1 - Complex.I)

-- Theorem statement
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l4034_403401


namespace complex_simplification_l4034_403467

theorem complex_simplification :
  let i : ℂ := Complex.I
  (1 + i)^2 / (2 - 3*i) = 6/5 - 4/5 * i :=
by sorry

end complex_simplification_l4034_403467


namespace integer_part_of_M_l4034_403439

theorem integer_part_of_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  4 < Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ∧
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) < 5 :=
by sorry

end integer_part_of_M_l4034_403439


namespace min_value_at_two_l4034_403489

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1

-- State the theorem
theorem min_value_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 := by
  sorry

end min_value_at_two_l4034_403489


namespace lose_sector_area_l4034_403422

/-- Given a circular spinner with radius 15 cm and a probability of winning of 1/3,
    the area of the LOSE sector is 150π sq cm. -/
theorem lose_sector_area (radius : ℝ) (win_prob : ℝ) (lose_area : ℝ) : 
  radius = 15 → 
  win_prob = 1/3 → 
  lose_area = 150 * Real.pi → 
  lose_area = (1 - win_prob) * Real.pi * radius^2 := by
sorry

end lose_sector_area_l4034_403422


namespace no_linear_term_condition_l4034_403484

theorem no_linear_term_condition (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (x + a) * (x - 1/2) = x^2 + b + c * x) → a = 1/2 := by
  sorry

end no_linear_term_condition_l4034_403484


namespace camel_zebra_ratio_l4034_403469

/-- Proves that the ratio of camels to zebras is 1:2 given the specified conditions -/
theorem camel_zebra_ratio :
  ∀ (zebras camels monkeys giraffes : ℕ),
    zebras = 12 →
    monkeys = 4 * camels →
    giraffes = 2 →
    monkeys = giraffes + 22 →
    (camels : ℚ) / zebras = 1 / 2 := by
  sorry

end camel_zebra_ratio_l4034_403469


namespace carls_ride_distance_l4034_403496

/-- The distance between Carl's house and Ralph's house -/
def distance : ℝ := 10

/-- The time Carl spent riding to Ralph's house in hours -/
def time : ℝ := 5

/-- Carl's speed in miles per hour -/
def speed : ℝ := 2

/-- Theorem: The distance between Carl's house and Ralph's house is 10 miles -/
theorem carls_ride_distance : distance = speed * time := by
  sorry

end carls_ride_distance_l4034_403496


namespace sum_of_three_numbers_l4034_403488

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 := by
  sorry

end sum_of_three_numbers_l4034_403488


namespace min_reciprocal_sum_l4034_403491

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 2 * x + 3 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  1 / a + 1 / b = 65 / 6 :=
sorry

end min_reciprocal_sum_l4034_403491


namespace triangle_altitude_l4034_403440

/-- Given a triangle with area 720 square feet and base 40 feet, its altitude is 36 feet -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 720 → base = 40 → area = (1/2) * base * altitude → altitude = 36 := by
  sorry

end triangle_altitude_l4034_403440


namespace dimes_percentage_l4034_403482

theorem dimes_percentage (num_nickels num_dimes : ℕ) 
  (nickel_value dime_value : ℕ) : 
  num_nickels = 40 → 
  num_dimes = 30 → 
  nickel_value = 5 → 
  dime_value = 10 → 
  (num_dimes * dime_value : ℚ) / 
  (num_nickels * nickel_value + num_dimes * dime_value) * 100 = 60 := by
sorry

end dimes_percentage_l4034_403482


namespace prop_a_necessary_not_sufficient_l4034_403483

theorem prop_a_necessary_not_sufficient :
  ¬(∀ x y : ℤ, (x ≠ 1000 ∨ y ≠ 1002) ↔ (x + y ≠ 2002)) ∧
  (∀ x y : ℤ, (x + y ≠ 2002) → (x ≠ 1000 ∨ y ≠ 1002)) :=
by sorry

end prop_a_necessary_not_sufficient_l4034_403483


namespace union_complement_equality_l4034_403494

def I : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_equality : A ∪ (I \ B) = {0, 1, 2} := by sorry

end union_complement_equality_l4034_403494


namespace max_students_distribution_l4034_403485

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1001) (h2 : pencils = 910) :
  (∃ (students pen_per_student pencil_per_student : ℕ),
    students * pen_per_student = pens ∧
    students * pencil_per_student = pencils ∧
    ∀ s : ℕ, s * pen_per_student = pens → s * pencil_per_student = pencils → s ≤ students) ↔
  students = Nat.gcd pens pencils :=
sorry

end max_students_distribution_l4034_403485


namespace number_not_divisible_by_8_and_digit_product_l4034_403466

def numbers : List Nat := [1616, 1728, 1834, 1944, 2056]

def is_divisible_by_8 (n : Nat) : Bool :=
  n % 8 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem number_not_divisible_by_8_and_digit_product :
  ∃ n ∈ numbers, ¬is_divisible_by_8 n ∧ units_digit n * tens_digit n = 12 := by
  sorry

end number_not_divisible_by_8_and_digit_product_l4034_403466


namespace group_size_proof_l4034_403495

theorem group_size_proof (n : ℕ) (D : ℝ) (h : D > 0) : 
  (n : ℝ) / 8 * D + (n : ℝ) / 10 * D = D → n = 9 := by
  sorry

end group_size_proof_l4034_403495


namespace average_snack_sales_theorem_l4034_403480

/-- Represents the sales data for snacks per 6 movie tickets sold -/
structure SnackSales where
  crackers_quantity : ℕ
  crackers_price : ℚ
  beverage_quantity : ℕ
  beverage_price : ℚ
  chocolate_quantity : ℕ
  chocolate_price : ℚ

/-- Calculates the average snack sales per movie ticket -/
def average_snack_sales_per_ticket (sales : SnackSales) : ℚ :=
  let total_sales := sales.crackers_quantity * sales.crackers_price +
                     sales.beverage_quantity * sales.beverage_price +
                     sales.chocolate_quantity * sales.chocolate_price
  total_sales / 6

/-- The main theorem stating the average snack sales per movie ticket -/
theorem average_snack_sales_theorem (sales : SnackSales) 
  (h1 : sales.crackers_quantity = 3)
  (h2 : sales.crackers_price = 9/4)
  (h3 : sales.beverage_quantity = 4)
  (h4 : sales.beverage_price = 3/2)
  (h5 : sales.chocolate_quantity = 4)
  (h6 : sales.chocolate_price = 1) :
  average_snack_sales_per_ticket sales = 279/100 := by
  sorry

end average_snack_sales_theorem_l4034_403480


namespace not_divides_power_minus_one_l4034_403441

theorem not_divides_power_minus_one (n : ℕ) (hn : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end not_divides_power_minus_one_l4034_403441


namespace probability_of_dime_l4034_403456

theorem probability_of_dime (quarter_value nickel_value penny_value dime_value : ℚ)
  (total_quarter_value total_nickel_value total_penny_value total_dime_value : ℚ)
  (h1 : quarter_value = 25/100)
  (h2 : nickel_value = 5/100)
  (h3 : penny_value = 1/100)
  (h4 : dime_value = 10/100)
  (h5 : total_quarter_value = 15)
  (h6 : total_nickel_value = 5)
  (h7 : total_penny_value = 2)
  (h8 : total_dime_value = 12) :
  (total_dime_value / dime_value) / 
  ((total_quarter_value / quarter_value) + 
   (total_nickel_value / nickel_value) + 
   (total_penny_value / penny_value) + 
   (total_dime_value / dime_value)) = 1/4 :=
by sorry

end probability_of_dime_l4034_403456


namespace barbara_typing_speed_reduction_l4034_403465

/-- Calculates the reduction in typing speed given the original speed, document length, and typing time. -/
def typing_speed_reduction (original_speed : ℕ) (document_length : ℕ) (typing_time : ℕ) : ℕ :=
  original_speed - (document_length / typing_time)

/-- Theorem stating that Barbara's typing speed has reduced by 40 words per minute. -/
theorem barbara_typing_speed_reduction :
  typing_speed_reduction 212 3440 20 = 40 := by
  sorry

#eval typing_speed_reduction 212 3440 20

end barbara_typing_speed_reduction_l4034_403465


namespace increasing_order_x_z_y_l4034_403400

theorem increasing_order_x_z_y (x : ℝ) (h : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by
  sorry

end increasing_order_x_z_y_l4034_403400


namespace axb_equals_bxa_l4034_403430

open Matrix

variable {n : ℕ}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem axb_equals_bxa (h : A * X * B + A + B = 0) : A * X * B = B * X * A := by
  sorry

end axb_equals_bxa_l4034_403430


namespace expansion_theorem_l4034_403416

theorem expansion_theorem (x : ℝ) (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n 2) / (Nat.choose n 4) = 3 / 14) →
  (n = 10 ∧ 
   ∃ m : ℕ, m = 8 ∧ 
   (Nat.choose n m) = 45 ∧ 
   20 - 2 * m - (1/2) * m = 0) :=
by sorry

end expansion_theorem_l4034_403416


namespace negation_of_existence_quadratic_inequality_negation_l4034_403457

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_of_existence_quadratic_inequality_negation_l4034_403457


namespace sin_cos_power_six_bounds_l4034_403498

theorem sin_cos_power_six_bounds :
  ∀ x : ℝ, (1 : ℝ) / 4 ≤ Real.sin x ^ 6 + Real.cos x ^ 6 ∧
            Real.sin x ^ 6 + Real.cos x ^ 6 ≤ 1 ∧
            (∃ y : ℝ, Real.sin y ^ 6 + Real.cos y ^ 6 = (1 : ℝ) / 4) ∧
            (∃ z : ℝ, Real.sin z ^ 6 + Real.cos z ^ 6 = 1) :=
by sorry

end sin_cos_power_six_bounds_l4034_403498


namespace current_for_given_resistance_l4034_403470

/-- Represents the relationship between voltage (V), current (I), and resistance (R) -/
def ohms_law (V I R : ℝ) : Prop := V = I * R

theorem current_for_given_resistance (V I R : ℝ) (h1 : V = 48) (h2 : R = 12) (h3 : ohms_law V I R) :
  I = 4 := by
  sorry

end current_for_given_resistance_l4034_403470


namespace freddy_travel_time_l4034_403412

/-- Represents the travel details of a person -/
structure TravelDetails where
  start : String
  destination : String
  distance : ℝ
  time : ℝ

/-- Given travel conditions for Eddy and Freddy -/
def travel_conditions : Prop :=
  ∃ (eddy freddy : TravelDetails),
    eddy.start = "A" ∧
    eddy.destination = "B" ∧
    freddy.start = "A" ∧
    freddy.destination = "C" ∧
    eddy.distance = 540 ∧
    freddy.distance = 300 ∧
    eddy.time = 3 ∧
    (eddy.distance / eddy.time) / (freddy.distance / freddy.time) = 2.4

/-- Theorem: Freddy's travel time is 4 hours -/
theorem freddy_travel_time : travel_conditions → ∃ (freddy : TravelDetails), freddy.time = 4 := by
  sorry


end freddy_travel_time_l4034_403412


namespace equidistant_point_on_leg_l4034_403459

/-- 
Given a right triangle with legs 240 and 320 rods, and hypotenuse 400 rods,
prove that the point on the longer leg equidistant from the other two vertices
is 95 rods from the right angle.
-/
theorem equidistant_point_on_leg (a b c x : ℝ) : 
  a = 240 → b = 320 → c = 400 → 
  a^2 + b^2 = c^2 →
  x^2 + a^2 = (b - x)^2 + b^2 →
  x = 95 := by
sorry

end equidistant_point_on_leg_l4034_403459


namespace line_through_point_parallel_to_y_axis_l4034_403406

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line parallel to y-axis passing through a given point
def LineParallelToYAxis (p : Point2D) := {q : Point2D | q.x = p.x}

theorem line_through_point_parallel_to_y_axis 
  (A : Point2D) 
  (h : A.x = -3 ∧ A.y = 1) 
  (P : Point2D) 
  (h_on_line : P ∈ LineParallelToYAxis A) : 
  P.x = -3 := by
sorry

end line_through_point_parallel_to_y_axis_l4034_403406


namespace cylinder_different_views_l4034_403462

/-- Represents a geometric body --/
inductive GeometricBody
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

/-- Represents the dimensions of a view --/
structure ViewDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Returns true if all three views have the same dimensions --/
def sameViewDimensions (top front left : ViewDimensions) : Prop :=
  top.length = front.length ∧
  front.height = left.height ∧
  left.width = top.width

/-- Returns the three orthogonal views of a geometric body --/
def getViews (body : GeometricBody) : (ViewDimensions × ViewDimensions × ViewDimensions) :=
  sorry

theorem cylinder_different_views :
  ∀ (body : GeometricBody),
    (∃ (top front left : ViewDimensions),
      getViews body = (top, front, left) ∧
      ¬(sameViewDimensions top front left)) ↔
    body = GeometricBody.Cylinder :=
  sorry

end cylinder_different_views_l4034_403462


namespace partnership_capital_share_l4034_403436

theorem partnership_capital_share 
  (total_capital : ℝ) 
  (total_profit : ℝ) 
  (a_profit_share : ℝ) 
  (b_capital_share : ℝ) 
  (c_capital_share : ℝ) 
  (h1 : b_capital_share = (1 / 4 : ℝ) * total_capital) 
  (h2 : c_capital_share = (1 / 5 : ℝ) * total_capital) 
  (h3 : a_profit_share = (800 : ℝ)) 
  (h4 : total_profit = (2400 : ℝ)) 
  (h5 : a_profit_share / total_profit = (1 / 3 : ℝ)) :
  ∃ (a_capital_share : ℝ), 
    a_capital_share = (1 / 3 : ℝ) * total_capital ∧ 
    a_capital_share + b_capital_share + c_capital_share ≤ total_capital := by
  sorry

end partnership_capital_share_l4034_403436


namespace pen_distribution_l4034_403497

theorem pen_distribution (total_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) :
  total_pencils = 928 →
  num_students = 16 →
  total_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = 16 * k :=
by sorry

end pen_distribution_l4034_403497


namespace base8_536_equals_base7_1054_l4034_403423

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10_to_base7 (n : ℕ) : ℕ := sorry

/-- Theorem stating that 536 in base 8 is equal to 1054 in base 7 -/
theorem base8_536_equals_base7_1054 : 
  base10_to_base7 (base8_to_base10 536) = 1054 := by sorry

end base8_536_equals_base7_1054_l4034_403423


namespace isosceles_triangle_altitude_midpoint_l4034_403460

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- Check if a point is on the altitude from A to BC -/
def isOnAltitude (t : Triangle) (d : Point) : Prop :=
  (t.B.y - t.C.y) * (d.x - t.A.x) = (t.C.x - t.B.x) * (d.y - t.A.y)

theorem isosceles_triangle_altitude_midpoint (t : Triangle) (d : Point) :
  t.A = Point.mk 5 7 →
  t.B = Point.mk (-1) 3 →
  d = Point.mk 1 5 →
  isIsosceles t →
  isOnAltitude t d →
  t.C = Point.mk 3 7 := by
  sorry

end isosceles_triangle_altitude_midpoint_l4034_403460


namespace dividend_proof_l4034_403487

theorem dividend_proof (y : ℝ) (x : ℝ) (h : y > 3) :
  (x = (3 * y + 5) * (2 * y - 1) + (5 * y - 13)) →
  (x = 6 * y^2 + 12 * y - 18) := by
  sorry

end dividend_proof_l4034_403487


namespace second_month_sale_l4034_403409

def sale_month1 : ℕ := 7435
def sale_month3 : ℕ := 7855
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7562
def sale_month6 : ℕ := 5991
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / num_months = average_sale ∧
    sale_month2 = 7927 :=
by sorry

end second_month_sale_l4034_403409


namespace trig_expression_equals_sqrt_three_l4034_403405

/-- Proves that the given trigonometric expression evaluates to √3 --/
theorem trig_expression_equals_sqrt_three :
  (Real.cos (350 * π / 180) - 2 * Real.sin (160 * π / 180)) / Real.sin (-190 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_expression_equals_sqrt_three_l4034_403405


namespace circle_area_difference_l4034_403426

theorem circle_area_difference (L W : ℝ) (h1 : L > 0) (h2 : W > 0) 
  (h3 : L * π = 704) (h4 : W * π = 396) : 
  (π * (L / 2)^2 - π * (W / 2)^2) = (L^2 - W^2) / (4 * π) := by
  sorry

end circle_area_difference_l4034_403426


namespace books_per_bookshelf_l4034_403450

theorem books_per_bookshelf (total_books : ℕ) (num_bookshelves : ℕ) 
  (h1 : total_books = 42) 
  (h2 : num_bookshelves = 21) :
  total_books / num_bookshelves = 2 := by
  sorry

end books_per_bookshelf_l4034_403450


namespace square_sum_of_roots_l4034_403421

theorem square_sum_of_roots (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := by
  sorry

end square_sum_of_roots_l4034_403421


namespace factor_difference_of_squares_l4034_403435

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_difference_of_squares_l4034_403435


namespace beach_towel_usage_per_person_per_day_l4034_403453

theorem beach_towel_usage_per_person_per_day :
  let num_families : ℕ := 3
  let people_per_family : ℕ := 4
  let total_days : ℕ := 7
  let towels_per_load : ℕ := 14
  let total_loads : ℕ := 6
  let total_people : ℕ := num_families * people_per_family
  let total_towels : ℕ := towels_per_load * total_loads
  let towels_per_day : ℕ := total_towels / total_days
  towels_per_day / total_people = 1 :=
by sorry

end beach_towel_usage_per_person_per_day_l4034_403453


namespace simplify_expression_l4034_403431

theorem simplify_expression (x : ℝ) (hx : x > 0) :
  2 / (3 * x) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x :=
by sorry

end simplify_expression_l4034_403431


namespace rect_to_cylindrical_l4034_403490

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 ∧ y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 2 →
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) := by
sorry

end rect_to_cylindrical_l4034_403490


namespace coefficient_x5_is_11_l4034_403404

/-- The coefficient of x^5 in the expansion of ((x^2 + x - 1)^5) -/
def coefficient_x5 : ℤ :=
  (Nat.choose 5 0) * (Nat.choose 5 5) -
  (Nat.choose 5 1) * (Nat.choose 4 3) +
  (Nat.choose 5 2) * (Nat.choose 3 1)

/-- Theorem stating that the coefficient of x^5 in ((x^2 + x - 1)^5) is 11 -/
theorem coefficient_x5_is_11 : coefficient_x5 = 11 := by
  sorry

end coefficient_x5_is_11_l4034_403404


namespace complement_intersection_eq_singleton_l4034_403407

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_eq_singleton :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_eq_singleton_l4034_403407


namespace judy_spending_l4034_403479

-- Define the prices and quantities
def carrot_price : ℚ := 1
def carrot_quantity : ℕ := 8
def milk_price : ℚ := 3
def milk_quantity : ℕ := 4
def pineapple_regular_price : ℚ := 4
def pineapple_quantity : ℕ := 3
def flour_price : ℚ := 5
def flour_quantity : ℕ := 3
def ice_cream_price : ℚ := 7
def ice_cream_quantity : ℕ := 2

-- Define the discount conditions
def discount_threshold : ℚ := 50
def discount_rate : ℚ := 0.1
def coupon_value : ℚ := 5
def coupon_threshold : ℚ := 30

-- Calculate the total before discounts
def total_before_discounts : ℚ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  (pineapple_regular_price / 2) * pineapple_quantity +
  flour_price * flour_quantity +
  ice_cream_price * ice_cream_quantity

-- Apply discounts
def final_total : ℚ :=
  let discounted_total := 
    if total_before_discounts > discount_threshold
    then total_before_discounts * (1 - discount_rate)
    else total_before_discounts
  if discounted_total ≥ coupon_threshold
  then discounted_total - coupon_value
  else discounted_total

-- Theorem statement
theorem judy_spending : final_total = 44.5 := by sorry

end judy_spending_l4034_403479


namespace factorial_ratio_l4034_403408

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by sorry

end factorial_ratio_l4034_403408


namespace polynomial_multiple_divisible_by_three_l4034_403463

theorem polynomial_multiple_divisible_by_three 
  {R : Type*} [CommRing R] [Nontrivial R] :
  ∀ (P : Polynomial R), P ≠ 0 → 
  ∃ (Q : Polynomial R), Q ≠ 0 ∧ 
  ∀ (i : ℕ), (P * Q).coeff i ≠ 0 → i % 3 = 0 := by
  sorry

end polynomial_multiple_divisible_by_three_l4034_403463


namespace square_sum_equals_one_l4034_403493

theorem square_sum_equals_one (x y : ℝ) :
  (x^2 + y^2 + 1)^2 - 4 = 0 → x^2 + y^2 = 1 := by
sorry

end square_sum_equals_one_l4034_403493


namespace trebled_result_of_doubled_plus_nine_l4034_403447

theorem trebled_result_of_doubled_plus_nine (x : ℕ) : x = 4 → 3 * (2 * x + 9) = 51 := by
  sorry

end trebled_result_of_doubled_plus_nine_l4034_403447


namespace smallest_n_boxes_two_boxes_satisfies_two_is_smallest_l4034_403499

theorem smallest_n_boxes (n : ℕ) : 
  (∃ k : ℕ, 15 * n - 2 = 7 * k) → n ≥ 2 :=
by
  sorry

theorem two_boxes_satisfies : 
  ∃ k : ℕ, 15 * 2 - 2 = 7 * k :=
by
  sorry

theorem two_is_smallest : 
  ∀ m : ℕ, m < 2 → ¬(∃ k : ℕ, 15 * m - 2 = 7 * k) :=
by
  sorry

end smallest_n_boxes_two_boxes_satisfies_two_is_smallest_l4034_403499


namespace cubic_equation_three_distinct_roots_l4034_403461

theorem cubic_equation_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 12*x + a = 0 ∧
    y^3 - 12*y + a = 0 ∧
    z^3 - 12*z + a = 0) ↔
  -16 < a ∧ a < 16 :=
by sorry

end cubic_equation_three_distinct_roots_l4034_403461


namespace complex_fraction_equality_l4034_403492

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (1 + Complex.I) / (3 - Complex.I) - Complex.I / (3 + Complex.I) = (1 + Complex.I) / 10 := by
  sorry

end complex_fraction_equality_l4034_403492


namespace thank_you_cards_percentage_l4034_403476

/-- The percentage of students who gave thank you cards to Ms. Jones -/
def percentage_thank_you_cards (
  total_students : ℕ)
  (gift_card_value : ℚ)
  (total_gift_card_amount : ℚ)
  (gift_card_fraction : ℚ) : ℚ :=
  (total_gift_card_amount / gift_card_value / gift_card_fraction) / total_students * 100

/-- Theorem stating that 30% of Ms. Jones' class gave her thank you cards -/
theorem thank_you_cards_percentage :
  percentage_thank_you_cards 50 10 50 (1/3) = 30 := by
  sorry

end thank_you_cards_percentage_l4034_403476


namespace sum_of_X_and_Y_is_12_l4034_403432

/-- Converts a single-digit number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := n

/-- Converts a two-digit number from base 6 to base 10 -/
def twoDigitBase6ToBase10 (tens : ℕ) (ones : ℕ) : ℕ := 
  6 * tens + ones

theorem sum_of_X_and_Y_is_12 (X Y : ℕ) : 
  (X < 6 ∧ Y < 6) →  -- Ensure X and Y are single digits in base 6
  twoDigitBase6ToBase10 1 3 + twoDigitBase6ToBase10 X Y = 
  twoDigitBase6ToBase10 2 0 + twoDigitBase6ToBase10 5 2 →
  X + Y = 12 := by
sorry

end sum_of_X_and_Y_is_12_l4034_403432


namespace initial_customers_count_l4034_403402

/-- The number of customers who left the restaurant -/
def customers_left : ℕ := 11

/-- The number of customers who remained in the restaurant -/
def customers_remained : ℕ := 3

/-- The initial number of customers in the restaurant -/
def initial_customers : ℕ := customers_left + customers_remained

theorem initial_customers_count : initial_customers = 14 := by
  sorry

end initial_customers_count_l4034_403402


namespace max_value_implies_a_l4034_403410

theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) :
  (∃ (M : ℝ), M = 32 ∧ ∀ (x : ℝ), a * x * (x - 2)^2 ≤ M) →
  a = 27 := by
  sorry

end max_value_implies_a_l4034_403410


namespace vector_parallel_proof_l4034_403419

def a (m : ℝ) : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (-2, 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_parallel_proof (m : ℝ) :
  parallel ((a m).1 - b.1, (a m).2 - b.2) b → m = -3 := by
  sorry

end vector_parallel_proof_l4034_403419


namespace equivalent_propositions_l4034_403427

theorem equivalent_propositions (x y : ℝ) :
  (x > 1 ∧ y < -3 → x - y > 4) ↔ (x - y ≤ 4 → x ≤ 1 ∨ y ≥ -3) := by
sorry

end equivalent_propositions_l4034_403427


namespace complex_distance_problem_l4034_403471

theorem complex_distance_problem (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs (z - 1) = Real.sqrt 2 := by
  sorry

end complex_distance_problem_l4034_403471


namespace corrected_mean_calculation_l4034_403445

def original_mean : ℝ := 45
def num_observations : ℕ := 100
def incorrect_observations : List ℝ := [32, 12, 25]
def correct_observations : List ℝ := [67, 52, 85]

theorem corrected_mean_calculation :
  let original_sum := original_mean * num_observations
  let incorrect_sum := incorrect_observations.sum
  let correct_sum := correct_observations.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_sum := original_sum + adjustment
  let corrected_mean := corrected_sum / num_observations
  corrected_mean = 46.35 := by sorry

end corrected_mean_calculation_l4034_403445


namespace zero_point_location_l4034_403472

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x + a

-- State the theorem
theorem zero_point_location (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (0 < a) → (a < 2) →
  (f a x₁ = 0) → (f a x₂ = 0) → (f a x₃ = 0) →
  (x₁ < x₂) → (x₂ < x₃) →
  (0 < x₂) ∧ (x₂ < 1) := by
  sorry

end zero_point_location_l4034_403472


namespace probability_of_letter_in_probability_l4034_403442

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in the word 'PROBABILITY' -/
def unique_letters : ℕ := 9

/-- The probability of randomly selecting a letter from the alphabet
    that appears in the word 'PROBABILITY' -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_letter_in_probability :
  probability = 9 / 26 := by
  sorry

end probability_of_letter_in_probability_l4034_403442


namespace inequality_proof_l4034_403443

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d) ∧
  ∀ k > 3/4, ∃ a' b' c' d' : ℝ, a' ≥ -1 ∧ b' ≥ -1 ∧ c' ≥ -1 ∧ d' ≥ -1 ∧
    a'^3 + b'^3 + c'^3 + d'^3 + 1 < k * (a' + b' + c' + d') :=
by sorry

end inequality_proof_l4034_403443


namespace min_distance_exp_ln_l4034_403468

/-- The minimum distance between points on y = e^x and y = ln(x) is √2 -/
theorem min_distance_exp_ln (P Q : ℝ × ℝ) :
  (∃ x : ℝ, P = (x, Real.exp x)) →
  (∃ y : ℝ, Q = (y, Real.log y)) →
  ∀ P' Q' : ℝ × ℝ,
  (∃ x' : ℝ, P' = (x', Real.exp x')) →
  (∃ y' : ℝ, Q' = (y', Real.log y')) →
  Real.sqrt 2 ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) :=
by sorry

end min_distance_exp_ln_l4034_403468


namespace sqrt_sum_bounds_l4034_403434

theorem sqrt_sum_bounds :
  let m := Real.sqrt 4 + Real.sqrt 3
  3 < m ∧ m < 4 := by sorry

end sqrt_sum_bounds_l4034_403434


namespace factor_expression_l4034_403474

theorem factor_expression (y : ℝ) : 6*y*(y+2) + 15*(y+2) + 12 = 3*(2*y+5)*(y+2) := by
  sorry

end factor_expression_l4034_403474
