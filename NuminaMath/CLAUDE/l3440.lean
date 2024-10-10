import Mathlib

namespace special_quadratic_a_range_l3440_344052

/-- A quadratic function with specific properties -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  symmetry : ∀ x, f (2 + x) = f (2 - x)
  inequality : ∃ a : ℝ, f a ≤ f 0 ∧ f 0 < f 1

/-- The range of 'a' for a special quadratic function -/
def range_of_a (q : SpecialQuadratic) : Set ℝ :=
  {x | x ≤ 0 ∨ x ≥ 4}

/-- Theorem stating the range of 'a' for a special quadratic function -/
theorem special_quadratic_a_range (q : SpecialQuadratic) :
  ∀ a : ℝ, q.f a ≤ q.f 0 → a ∈ range_of_a q := by
  sorry

end special_quadratic_a_range_l3440_344052


namespace passing_marks_l3440_344084

theorem passing_marks (T : ℝ) 
  (h1 : 0.3 * T + 60 = 0.4 * T) 
  (h2 : 0.5 * T = 0.4 * T + 40) : 
  0.4 * T = 240 := by
  sorry

end passing_marks_l3440_344084


namespace unique_intersection_k_values_l3440_344047

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the theorem
theorem unique_intersection_k_values :
  ∃! z, equation1 z ∧ equation2 z k → k = 0.631 ∨ k = 25.369 :=
sorry

end unique_intersection_k_values_l3440_344047


namespace intersection_with_complement_l3440_344029

def U : Finset ℕ := {0, 1, 2, 3}
def A : Finset ℕ := {0, 1, 2}
def B : Finset ℕ := {0, 2, 3}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1} := by sorry

end intersection_with_complement_l3440_344029


namespace intersection_point_existence_l3440_344037

theorem intersection_point_existence : ∃ x : ℝ, 1/2 < x ∧ x < 1 ∧ Real.exp x = -2*x + 3 := by
  sorry

end intersection_point_existence_l3440_344037


namespace log_equation_root_range_l3440_344046

theorem log_equation_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ 
   Real.log (x - 1) + Real.log (3 - x) = Real.log (a - x) ∧
   Real.log (y - 1) + Real.log (3 - y) = Real.log (a - y)) →
  (3 < a ∧ a < 13/4) :=
by sorry

end log_equation_root_range_l3440_344046


namespace cylindrical_eight_queens_impossible_l3440_344005

/-- Represents a position on the cylindrical chessboard -/
structure Position :=
  (x : Fin 8) -- column
  (y : Fin 8) -- row

/-- Checks if two positions are attacking each other on the cylindrical chessboard -/
def isAttacking (p1 p2 : Position) : Prop :=
  p1.x = p2.x ∨ 
  p1.y = p2.y ∨ 
  (p1.x.val - p2.x.val) % 8 = (p1.y.val - p2.y.val) % 8 ∨
  (p1.x.val - p2.x.val) % 8 = (p2.y.val - p1.y.val) % 8

/-- A configuration of 8 queens on the cylindrical chessboard -/
def QueenConfiguration := Fin 8 → Position

/-- Theorem: It's impossible to place 8 queens on a cylindrical chessboard without attacks -/
theorem cylindrical_eight_queens_impossible :
  ∀ (config : QueenConfiguration), 
    ∃ (i j : Fin 8), i ≠ j ∧ isAttacking (config i) (config j) := by
  sorry


end cylindrical_eight_queens_impossible_l3440_344005


namespace tanning_salon_pricing_l3440_344016

theorem tanning_salon_pricing (first_visit_charge : ℕ) (total_customers : ℕ) (second_visits : ℕ) (third_visits : ℕ) (total_revenue : ℕ) :
  first_visit_charge = 10 →
  total_customers = 100 →
  second_visits = 30 →
  third_visits = 10 →
  total_revenue = 1240 →
  ∃ (subsequent_visit_charge : ℕ),
    subsequent_visit_charge = 6 ∧
    total_revenue = first_visit_charge * total_customers + subsequent_visit_charge * (second_visits + third_visits) :=
by sorry

end tanning_salon_pricing_l3440_344016


namespace barkley_bones_theorem_l3440_344061

/-- Calculates the number of bones Barkley has available after a given number of months -/
def bones_available (bones_per_month : ℕ) (months : ℕ) (buried_bones : ℕ) : ℕ :=
  bones_per_month * months - buried_bones

/-- Theorem: Barkley has 8 bones available after 5 months -/
theorem barkley_bones_theorem :
  bones_available 10 5 42 = 8 := by
  sorry

end barkley_bones_theorem_l3440_344061


namespace grassy_plot_width_l3440_344050

/-- Proves that the width of a rectangular grassy plot is 60 meters given the specified conditions --/
theorem grassy_plot_width :
  ∀ (w : ℝ),
  let plot_length : ℝ := 100
  let path_width : ℝ := 2.5
  let gravel_cost_per_sqm : ℝ := 0.9  -- 90 paise = 0.9 rupees
  let total_gravel_cost : ℝ := 742.5
  let total_length : ℝ := plot_length + 2 * path_width
  let total_width : ℝ := w + 2 * path_width
  let path_area : ℝ := total_length * total_width - plot_length * w
  gravel_cost_per_sqm * path_area = total_gravel_cost →
  w = 60 := by
sorry

end grassy_plot_width_l3440_344050


namespace triangle_side_length_l3440_344014

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →  -- Convert 45° to radians
  C = 105 * π / 180 →  -- Convert 105° to radians
  b = Real.sqrt 2 →
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.sin B = b * Real.sin A →  -- Law of sines
  c * Real.sin B = b * Real.sin C →  -- Law of sines
  a = 1 := by
sorry

end triangle_side_length_l3440_344014


namespace remaining_area_in_square_configuration_l3440_344068

/-- The area of the remaining space in a square configuration -/
theorem remaining_area_in_square_configuration : 
  ∀ (s : ℝ) (small_square : ℝ) (rect1_length rect1_width : ℝ) (rect2_length rect2_width : ℝ),
  s = 4 →
  small_square = 1 →
  rect1_length = 2 ∧ rect1_width = 1 →
  rect2_length = 1 ∧ rect2_width = 2 →
  s^2 - (small_square^2 + rect1_length * rect1_width + rect2_length * rect2_width) = 11 :=
by sorry

end remaining_area_in_square_configuration_l3440_344068


namespace cubic_tangent_line_theorem_l3440_344077

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d, 
    if the equation of the tangent line to its graph at x=0 is 24x + y - 12 = 0, 
    then c + 2d = 0 -/
theorem cubic_tangent_line_theorem (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * b * x + c
  (∀ x y, y = f x → (24 * 0 + y - 12 = 0 ↔ 24 * x + y - 12 = 0)) →
  c + 2 * d = 0 := by
  sorry

end cubic_tangent_line_theorem_l3440_344077


namespace fast_food_cost_l3440_344099

/-- The total cost of buying fast food -/
def total_cost (a b : ℕ) : ℕ := 30 * a + 20 * b

/-- The price of one serving of type A fast food -/
def price_A : ℕ := 30

/-- The price of one serving of type B fast food -/
def price_B : ℕ := 20

/-- Theorem: The total cost of buying 'a' servings of type A fast food and 'b' servings of type B fast food is 30a + 20b yuan -/
theorem fast_food_cost (a b : ℕ) : total_cost a b = price_A * a + price_B * b := by
  sorry

end fast_food_cost_l3440_344099


namespace power_sum_equation_l3440_344002

theorem power_sum_equation (p : ℕ) (a : ℤ) (n : ℕ) :
  Nat.Prime p → (2^p : ℤ) + 3^p = a^n → n = 1 := by
  sorry

end power_sum_equation_l3440_344002


namespace equidistant_function_property_l3440_344000

/-- Given a function g(z) = (c+di)z where c and d are real numbers,
    if g(z) is equidistant from z and the origin for all complex z,
    and |c+di| = 5, then d^2 = 99/4 -/
theorem equidistant_function_property (c d : ℝ) :
  (∀ z : ℂ, ‖(c + d * Complex.I) * z - z‖ = ‖(c + d * Complex.I) * z‖) →
  Complex.abs (c + d * Complex.I) = 5 →
  d^2 = 99/4 := by
  sorry

end equidistant_function_property_l3440_344000


namespace equation_represents_two_intersecting_lines_l3440_344070

-- Define the equation
def equation (x y : ℝ) : Prop := (x - y)^2 = 3 * x^2 - y^2

-- Theorem statement
theorem equation_represents_two_intersecting_lines :
  ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
  (∀ (x y : ℝ), equation x y ↔ (y = m₁ * x ∨ y = m₂ * x)) :=
sorry

end equation_represents_two_intersecting_lines_l3440_344070


namespace dvd_sales_l3440_344076

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end dvd_sales_l3440_344076


namespace resulting_polynomial_degree_is_eight_l3440_344067

/-- The degree of the polynomial resulting from the given operations -/
def resultingPolynomialDegree : ℕ :=
  let expr1 := fun x : ℝ => x^4
  let expr2 := fun x : ℝ => x^2 - 1/x^2
  let expr3 := fun x : ℝ => 1 - 3/x + 3/x^2
  let squaredExpr2 := fun x : ℝ => (expr2 x)^2
  let result := fun x : ℝ => (expr1 x) * (squaredExpr2 x) * (expr3 x)
  8

/-- Theorem stating that the degree of the resulting polynomial is 8 -/
theorem resulting_polynomial_degree_is_eight :
  resultingPolynomialDegree = 8 := by sorry

end resulting_polynomial_degree_is_eight_l3440_344067


namespace sum_of_roots_is_twelve_l3440_344025

/-- A function satisfying the symmetry property g(3+x) = g(3-x) -/
def SymmetricAboutThree (g : ℝ → ℝ) : Prop :=
  ∀ x, g (3 + x) = g (3 - x)

/-- The property that a function has exactly four distinct real roots -/
def HasFourDistinctRealRoots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x, g x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The main theorem statement -/
theorem sum_of_roots_is_twelve (g : ℝ → ℝ) 
  (h1 : SymmetricAboutThree g) (h2 : HasFourDistinctRealRoots g) :
  ∃ (a b c d : ℝ), (HasFourDistinctRealRoots g → g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    a + b + c + d = 12 :=
sorry

end sum_of_roots_is_twelve_l3440_344025


namespace union_equality_implies_a_value_l3440_344075

def A (a : ℝ) : Set ℝ := {2*a, 3}
def B : Set ℝ := {2, 3}

theorem union_equality_implies_a_value (a : ℝ) :
  A a ∪ B = {2, 3, 4} → a = 2 := by
  sorry

end union_equality_implies_a_value_l3440_344075


namespace inequality_system_solutions_l3440_344071

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 5 > 0 ∧ x - m ≤ 1))) → 
  -3 ≤ m ∧ m < -2 :=
by sorry

end inequality_system_solutions_l3440_344071


namespace fraction_not_simplifiable_l3440_344051

theorem fraction_not_simplifiable (n : ℕ) : 
  (∃ k : ℕ, n = 6 * k + 1 ∨ n = 6 * k + 3) ↔ 
  ¬∃ (a b : ℕ), (n^2 + 2 : ℚ) / (n * (n + 1)) = (a : ℚ) / b ∧ 
                gcd a b = 1 ∧ 
                b < n * (n + 1) :=
sorry

end fraction_not_simplifiable_l3440_344051


namespace largest_five_digit_integer_l3440_344015

def digit_product (n : ℕ) : ℕ := 
  (n.digits 10).prod

def digit_sum (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem largest_five_digit_integer : 
  ∀ n : ℕ, 
    n ≤ 99999 ∧ 
    n ≥ 10000 ∧ 
    digit_product n = 40320 ∧ 
    digit_sum n < 35 → 
    n ≤ 98764 :=
by sorry

end largest_five_digit_integer_l3440_344015


namespace bowtie_equation_l3440_344045

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3))))

-- State the theorem
theorem bowtie_equation (g : ℝ) : bowtie 8 g = 11 → g = 3 := by
  sorry

end bowtie_equation_l3440_344045


namespace compare_polynomial_expressions_l3440_344033

theorem compare_polynomial_expressions {a b c : ℝ} (h1 : a > b) (h2 : b > c) :
  a^2*b + b^2*c + c^2*a > a*b^2 + b*c^2 + c*a^2 := by
  sorry

end compare_polynomial_expressions_l3440_344033


namespace initial_mean_calculation_l3440_344097

/-- Given 20 observations with an initial mean, prove that correcting one observation
    from 40 to 25 results in a new mean of 34.9 if and only if the initial mean was 35.65 -/
theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℝ) :
  n = 20 ∧
  corrected_mean = 34.9 ∧
  (n : ℝ) * initial_mean - 15 = (n : ℝ) * corrected_mean →
  initial_mean = 35.65 := by
  sorry

end initial_mean_calculation_l3440_344097


namespace gold_quarter_value_ratio_is_80_l3440_344006

/-- Represents the ratio of melted gold value to face value for gold quarters -/
def gold_quarter_value_ratio : ℚ :=
  let quarter_weight : ℚ := 1 / 5
  let melted_gold_value_per_ounce : ℚ := 100
  let quarter_face_value : ℚ := 1 / 4
  let quarters_per_ounce : ℚ := 1 / quarter_weight
  let melted_value_per_quarter : ℚ := melted_gold_value_per_ounce * quarter_weight
  melted_value_per_quarter / quarter_face_value

/-- Theorem stating that the gold quarter value ratio is 80 -/
theorem gold_quarter_value_ratio_is_80 : gold_quarter_value_ratio = 80 := by
  sorry

end gold_quarter_value_ratio_is_80_l3440_344006


namespace students_taking_no_subjects_l3440_344064

theorem students_taking_no_subjects (total : ℕ) (music art dance : ℕ) 
  (music_and_art music_and_dance art_and_dance : ℕ) (all_three : ℕ) :
  total = 800 ∧ 
  music = 140 ∧ 
  art = 90 ∧ 
  dance = 75 ∧
  music_and_art = 50 ∧
  music_and_dance = 30 ∧
  art_and_dance = 25 ∧
  all_three = 20 →
  total - (music + art + dance - music_and_art - music_and_dance - art_and_dance + all_three) = 580 := by
  sorry

end students_taking_no_subjects_l3440_344064


namespace intersection_points_range_l3440_344085

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x^3 + 1
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - b

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ = g b x₁ ∧ f x₂ = g b x₂ ∧ f x₃ = g b x₃

-- State the theorem
theorem intersection_points_range :
  ∀ b : ℝ, has_three_distinct_intersections b ↔ -1 < b ∧ b < 0 :=
sorry

end intersection_points_range_l3440_344085


namespace fort_blocks_theorem_l3440_344028

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and wall thickness -/
def blocksNeeded (dimensions : FortDimensions) (wallThickness : ℕ) : ℕ :=
  let totalVolume := dimensions.length * dimensions.width * dimensions.height
  let interiorLength := dimensions.length - 2 * wallThickness
  let interiorWidth := dimensions.width - 2 * wallThickness
  let interiorHeight := dimensions.height - wallThickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating that a fort with given dimensions and wall thickness requires 728 blocks -/
theorem fort_blocks_theorem :
  let dimensions : FortDimensions := ⟨15, 12, 6⟩
  let wallThickness : ℕ := 2
  blocksNeeded dimensions wallThickness = 728 := by
  sorry

end fort_blocks_theorem_l3440_344028


namespace base7_product_l3440_344044

/-- Converts a base 7 number to base 10 -/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- The problem statement -/
theorem base7_product :
  let a := [1, 2, 3]  -- 321 in base 7 (least significant digit first)
  let b := [3, 1]     -- 13 in base 7 (least significant digit first)
  toBase7 (toBase10 a * toBase10 b) = [3, 0, 5, 4] := by
  sorry

end base7_product_l3440_344044


namespace f_919_equals_6_l3440_344031

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_six (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) = f x

theorem f_919_equals_6 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 4) = f (x - 2))
  (h3 : ∀ x ∈ Set.Icc (-3) 0, f x = (6 : ℝ) ^ (-x)) :
  f 919 = 6 := by
  sorry

end f_919_equals_6_l3440_344031


namespace no_decagon_partition_l3440_344043

/-- A partition of a polygon into triangles -/
structure TrianglePartition (n : ℕ) where
  black_sides : ℕ
  white_sides : ℕ
  adjacent_diff_color : Prop
  decagon_sides_black : Prop

/-- The theorem stating that a decagon cannot be partitioned in the specified manner -/
theorem no_decagon_partition : ¬ ∃ (p : TrianglePartition 10),
  p.black_sides % 3 = 0 ∧ 
  p.white_sides % 3 = 0 ∧
  p.black_sides - p.white_sides = 10 :=
sorry

end no_decagon_partition_l3440_344043


namespace folder_cost_l3440_344030

/-- The cost of office supplies problem -/
theorem folder_cost (pencil_cost : ℚ) (pencil_count : ℕ) (folder_count : ℕ) (total_cost : ℚ) : 
  pencil_cost = 1/2 →
  pencil_count = 24 →
  folder_count = 20 →
  total_cost = 30 →
  (total_cost - pencil_cost * pencil_count) / folder_count = 9/10 := by
  sorry

end folder_cost_l3440_344030


namespace complex_sum_of_parts_l3440_344059

theorem complex_sum_of_parts (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  zi = Complex.mk 1 (-2) → a + b = -3 := by
  sorry

end complex_sum_of_parts_l3440_344059


namespace diagonal_length_of_quadrilateral_l3440_344021

/-- The length of a diagonal in a quadrilateral with given area and offsets -/
theorem diagonal_length_of_quadrilateral (area : ℝ) (offset1 offset2 : ℝ) 
  (h_area : area = 140)
  (h_offset1 : offset1 = 8)
  (h_offset2 : offset2 = 2)
  (h_quad_area : area = (1/2) * (offset1 + offset2) * diagonal_length) :
  diagonal_length = 28 := by
  sorry

end diagonal_length_of_quadrilateral_l3440_344021


namespace b_completion_time_l3440_344054

/- Define the work completion time for A -/
def a_completion_time : ℝ := 9

/- Define B's efficiency relative to A -/
def b_efficiency_factor : ℝ := 1.5

/- Theorem statement -/
theorem b_completion_time :
  let a_rate := 1 / a_completion_time
  let b_rate := b_efficiency_factor * a_rate
  (1 / b_rate) = 6 := by sorry

end b_completion_time_l3440_344054


namespace max_knights_between_knights_max_knights_between_knights_proof_l3440_344035

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

end max_knights_between_knights_max_knights_between_knights_proof_l3440_344035


namespace car_speed_comparison_l3440_344010

/-- Proves that the average speed of Car A is less than or equal to the average speed of Car B -/
theorem car_speed_comparison
  (u v : ℝ) -- speeds in miles per hour
  (hu : u > 0) (hv : v > 0) -- speeds are positive
  (x : ℝ) -- average speed of Car A
  (hx : x = 3 / (1 / u + 2 / v)) -- definition of x
  (y : ℝ) -- average speed of Car B
  (hy : y = (u + 2 * v) / 3) -- definition of y
  : x ≤ y :=
by sorry

end car_speed_comparison_l3440_344010


namespace robin_gum_count_l3440_344023

/-- Calculates the total number of gum pieces given the number of packages, pieces per package, and extra pieces. -/
def total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : ℕ :=
  packages * pieces_per_package + extra_pieces

/-- Proves that Robin has 997 pieces of gum given the specified conditions. -/
theorem robin_gum_count :
  total_gum_pieces 43 23 8 = 997 := by
  sorry

end robin_gum_count_l3440_344023


namespace cos_two_alpha_l3440_344034

theorem cos_two_alpha (α : Real) (h : Real.sin α + Real.cos α = 2/3) :
  Real.cos (2 * α) = 2 * Real.sqrt 14 / 9 ∨ Real.cos (2 * α) = -2 * Real.sqrt 14 / 9 := by
sorry

end cos_two_alpha_l3440_344034


namespace median_in_middle_interval_l3440_344079

/-- Represents the intervals of scores -/
inductive ScoreInterval
| I60to64
| I65to69
| I70to74
| I75to79
| I80to84

/-- The total number of students -/
def totalStudents : ℕ := 100

/-- The number of intervals -/
def numIntervals : ℕ := 5

/-- The number of students in each interval -/
def studentsPerInterval : ℕ := totalStudents / numIntervals

/-- The position of the median in the ordered list of scores -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Theorem stating that the median score falls in the middle interval -/
theorem median_in_middle_interval :
  medianPosition > 2 * studentsPerInterval ∧
  medianPosition ≤ 3 * studentsPerInterval :=
sorry

end median_in_middle_interval_l3440_344079


namespace nth_prime_greater_than_3n_l3440_344048

-- Define the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem nth_prime_greater_than_3n (n : ℕ) (h : n > 12) : nth_prime n > 3 * n := by
  sorry

end nth_prime_greater_than_3n_l3440_344048


namespace reef_age_conversion_l3440_344058

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

theorem reef_age_conversion :
  octal_to_decimal 367 = 247 := by
  sorry

end reef_age_conversion_l3440_344058


namespace trolley_problem_l3440_344036

/-- Trolley problem theorem -/
theorem trolley_problem (initial_passengers : ℕ) 
  (second_stop_off : ℕ) (second_stop_on_multiplier : ℕ)
  (third_stop_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 10 →
  second_stop_off = 3 →
  second_stop_on_multiplier = 2 →
  third_stop_off = 18 →
  final_passengers = 12 →
  (initial_passengers - second_stop_off + second_stop_on_multiplier * initial_passengers - third_stop_off) + 3 = final_passengers :=
by
  sorry

#check trolley_problem

end trolley_problem_l3440_344036


namespace f_zero_eq_three_l3440_344060

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 0  -- handle the case where x = 1 (2x-1 = 1)
  else (1 - ((x + 1) / 2)^2) / ((x + 1) / 2)^2

theorem f_zero_eq_three :
  f 0 = 3 :=
by sorry

end f_zero_eq_three_l3440_344060


namespace equidistant_locus_equation_l3440_344074

/-- The locus of points equidistant from the coordinate axes -/
def EquidistantLocus : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 = abs p.2}

/-- The equation |x| - |y| = 0 holds for points on the locus -/
theorem equidistant_locus_equation (p : ℝ × ℝ) :
  p ∈ EquidistantLocus ↔ abs p.1 - abs p.2 = 0 := by
  sorry

end equidistant_locus_equation_l3440_344074


namespace high_correlation_implies_r_near_one_l3440_344020

-- Define the correlation coefficient
def correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define what it means for a correlation to be "very high"
def is_very_high_correlation (r : ℝ) : Prop := sorry

-- Theorem statement
theorem high_correlation_implies_r_near_one 
  (x y : List ℝ) (r : ℝ) 
  (h1 : r = correlation_coefficient x y) 
  (h2 : is_very_high_correlation r) : 
  ∀ ε > 0, |r| > 1 - ε := by
  sorry

end high_correlation_implies_r_near_one_l3440_344020


namespace circle_area_decrease_l3440_344027

theorem circle_area_decrease (r : ℝ) (h : r > 0) : 
  let r' := 0.8 * r
  let A := π * r^2
  let A' := π * r'^2
  (A - A') / A = 0.36 := by
  sorry

end circle_area_decrease_l3440_344027


namespace distribute_five_items_three_bags_l3440_344081

/-- The number of ways to distribute n distinct items into k identical bags,
    allowing for empty bags and the possibility of leaving items out. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 106 ways to distribute 5 distinct items
    into 3 identical bags, allowing for empty bags and the possibility of
    leaving one item out. -/
theorem distribute_five_items_three_bags : distribute 5 3 = 106 := by sorry

end distribute_five_items_three_bags_l3440_344081


namespace fair_coin_probability_l3440_344069

-- Define a fair coin
def fair_coin := { p : ℝ | 0 ≤ p ∧ p ≤ 1 ∧ p = 1 - p }

-- Define the number of tosses
def num_tosses : ℕ := 20

-- Define the number of heads
def num_heads : ℕ := 8

-- Define the number of tails
def num_tails : ℕ := 12

-- Theorem statement
theorem fair_coin_probability : 
  ∀ (p : ℝ), p ∈ fair_coin → p = 1/2 :=
sorry

end fair_coin_probability_l3440_344069


namespace equation_solution_l3440_344080

theorem equation_solution (y : ℝ) (h : y ≠ 2) : 
  (y^2 - 10*y + 24) / (y - 2) + (4*y^2 + 8*y - 48) / (4*y - 8) = 0 ↔ y = 0 :=
by sorry

end equation_solution_l3440_344080


namespace kevin_food_spending_l3440_344094

def total_budget : ℕ := 20
def samuel_ticket : ℕ := 14
def samuel_total : ℕ := 20
def kevin_ticket : ℕ := 14
def kevin_drinks : ℕ := 2

theorem kevin_food_spending :
  ∃ (kevin_food : ℕ),
    kevin_food = total_budget - (kevin_ticket + kevin_drinks) ∧
    kevin_food = 4 :=
by sorry

end kevin_food_spending_l3440_344094


namespace f_nonpositive_implies_k_geq_one_l3440_344093

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log x - k * x + 1

theorem f_nonpositive_implies_k_geq_one (k : ℝ) :
  (∀ x > 0, f k x ≤ 0) → k ≥ 1 := by
  sorry

end f_nonpositive_implies_k_geq_one_l3440_344093


namespace quadratic_form_sum_l3440_344042

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ), 
  (6 * x^2 + 12 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = 9) := by
  sorry

end quadratic_form_sum_l3440_344042


namespace polynomial_division_remainder_l3440_344012

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^4 + 13 * X^3 + 5 * X^2 - 10 * X + 20 = 
  (X^2 + 5 * X + 1) * q + (-68 * X + 8) := by
  sorry

end polynomial_division_remainder_l3440_344012


namespace right_triangle_area_l3440_344082

/-- The area of a right triangle with base 12 and height 15 is 90 -/
theorem right_triangle_area : ∀ (base height area : ℝ),
  base = 12 →
  height = 15 →
  area = (1/2) * base * height →
  area = 90 := by
  sorry

end right_triangle_area_l3440_344082


namespace all_solutions_irrational_l3440_344017

/-- A number is rational if it can be expressed as a ratio of two integers -/
def IsRational (x : ℝ) : Prop := ∃ (m n : ℤ), n ≠ 0 ∧ x = m / n

/-- A number is irrational if it is not rational -/
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

/-- The equation in question -/
def SatisfiesEquation (x : ℝ) : Prop := 0.001 * x^3 + x^2 - 1 = 0

theorem all_solutions_irrational :
  ∀ x : ℝ, SatisfiesEquation x → IsIrrational x := by
  sorry

end all_solutions_irrational_l3440_344017


namespace no_real_solutions_l3440_344078

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 4) ∧ (x * y - 9 * z^2 = -5) := by
  sorry

end no_real_solutions_l3440_344078


namespace dog_eaten_cost_l3440_344003

-- Define the ingredients and their costs
def flour_cost : ℝ := 3.20
def sugar_cost : ℝ := 2.10
def butter_cost : ℝ := 5.50
def egg_cost : ℝ := 0.45
def baking_soda_cost : ℝ := 0.60
def baking_powder_cost : ℝ := 1.30
def salt_cost : ℝ := 0.35
def vanilla_extract_cost : ℝ := 1.75
def milk_cost : ℝ := 1.40
def vegetable_oil_cost : ℝ := 2.10

-- Define the quantities of ingredients
def flour_qty : ℝ := 2.5
def sugar_qty : ℝ := 1.5
def butter_qty : ℝ := 0.75
def egg_qty : ℝ := 4
def baking_soda_qty : ℝ := 1
def baking_powder_qty : ℝ := 1
def salt_qty : ℝ := 1
def vanilla_extract_qty : ℝ := 1
def milk_qty : ℝ := 1.25
def vegetable_oil_qty : ℝ := 0.75

-- Define other constants
def sales_tax_rate : ℝ := 0.07
def total_slices : ℕ := 12
def mother_eaten_slices : ℕ := 4

-- Theorem to prove
theorem dog_eaten_cost (total_cost : ℝ) (cost_with_tax : ℝ) (cost_per_slice : ℝ) :
  total_cost = flour_cost * flour_qty + sugar_cost * sugar_qty + butter_cost * butter_qty +
               egg_cost * egg_qty + baking_soda_cost * baking_soda_qty + 
               baking_powder_cost * baking_powder_qty + salt_cost * salt_qty +
               vanilla_extract_cost * vanilla_extract_qty + milk_cost * milk_qty +
               vegetable_oil_cost * vegetable_oil_qty →
  cost_with_tax = total_cost * (1 + sales_tax_rate) →
  cost_per_slice = cost_with_tax / total_slices →
  cost_per_slice * (total_slices - mother_eaten_slices) = 17.44 :=
by sorry

end dog_eaten_cost_l3440_344003


namespace fraction_difference_2023_2022_l3440_344049

theorem fraction_difference_2023_2022 : 
  (2023 : ℚ) / 2022 - 2022 / 2023 = 4045 / (2022 * 2023) := by
  sorry

end fraction_difference_2023_2022_l3440_344049


namespace hcf_problem_l3440_344056

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 17820) (h2 : Nat.lcm a b = 1485) :
  Nat.gcd a b = 12 := by
sorry

end hcf_problem_l3440_344056


namespace no_real_solutions_condition_l3440_344011

theorem no_real_solutions_condition (a : ℝ) : 
  (∀ x : ℝ, (a^2 + 2*a)*x^2 + 3*a*x + 1 ≠ 0) ↔ (0 < a ∧ a < 8/5) :=
by sorry

end no_real_solutions_condition_l3440_344011


namespace parabola_focus_coordinates_l3440_344066

/-- Given a hyperbola C₁ and a parabola C₂, prove that the focus of C₂ has coordinates (0, 3/2) -/
theorem parabola_focus_coordinates 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop) 
  (h_C₁ : ∀ x y, C₁ x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (h_C₂ : ∀ x y, C₂ x y ↔ x^2 = 2 * p * y)
  (h_eccentricity : a^2 + b^2 = 2 * a^2)  -- Eccentricity of C₁ is √2
  (P : ℝ × ℝ) 
  (h_P_on_C₂ : C₂ P.1 P.2)
  (h_tangent_parallel : ∃ (m : ℝ), m = 1 ∨ m = -1 ∧ 
    ∀ x y, C₂ x y → (y - P.2) = m * (x - P.1))
  (F : ℝ × ℝ)
  (h_F_focus : F.1 = 0 ∧ F.2 = p / 2)
  (h_PF_distance : (P.1 - F.1)^2 + (P.2 - F.2)^2 = 9)  -- |PF| = 3
  : F = (0, 3/2) := by sorry

end parabola_focus_coordinates_l3440_344066


namespace library_wall_leftover_space_l3440_344092

theorem library_wall_leftover_space
  (wall_length : ℝ)
  (desk_length : ℝ)
  (bookcase_length : ℝ)
  (h_wall : wall_length = 15)
  (h_desk : desk_length = 2)
  (h_bookcase : bookcase_length = 1.5)
  : ∃ (num_items : ℕ),
    let total_length := num_items * desk_length + num_items * bookcase_length
    wall_length - total_length = 1 ∧
    ∀ (n : ℕ), n * desk_length + n * bookcase_length ≤ wall_length → n ≤ num_items :=
by sorry

end library_wall_leftover_space_l3440_344092


namespace new_average_production_theorem_l3440_344022

/-- Calculates the new average daily production after adding a new day's production -/
def newAverageDailyProduction (n : ℕ) (pastAverage : ℚ) (todayProduction : ℚ) : ℚ :=
  ((n : ℚ) * pastAverage + todayProduction) / ((n : ℚ) + 1)

theorem new_average_production_theorem :
  let n : ℕ := 8
  let pastAverage : ℚ := 50
  let todayProduction : ℚ := 95
  newAverageDailyProduction n pastAverage todayProduction = 55 := by
  sorry

end new_average_production_theorem_l3440_344022


namespace permutations_with_fixed_front_five_people_one_fixed_front_l3440_344055

/-- The number of ways to arrange n people in a line. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with one specific person always at the front. -/
def permutationsWithFixed (n : ℕ) : ℕ := permutations (n - 1)

theorem permutations_with_fixed_front (n : ℕ) (h : n > 1) :
  permutationsWithFixed n = Nat.factorial (n - 1) := by
  sorry

/-- There are 5 people, and we want to arrange them with one specific person at the front. -/
theorem five_people_one_fixed_front :
  permutationsWithFixed 5 = 24 := by
  sorry

end permutations_with_fixed_front_five_people_one_fixed_front_l3440_344055


namespace parallelogram_vertex_D_l3440_344001

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if ABCD forms a parallelogram -/
def isParallelogram (A B C D : Point3D) : Prop :=
  (B.x - A.x) + (D.x - C.x) = 0 ∧
  (B.y - A.y) + (D.y - C.y) = 0 ∧
  (B.z - A.z) + (D.z - C.z) = 0

theorem parallelogram_vertex_D :
  let A : Point3D := ⟨2, 0, 3⟩
  let B : Point3D := ⟨0, 3, -5⟩
  let C : Point3D := ⟨0, 0, 3⟩
  let D : Point3D := ⟨2, -3, 11⟩
  isParallelogram A B C D := by sorry

end parallelogram_vertex_D_l3440_344001


namespace f_properties_l3440_344019

noncomputable section

variables (a : ℝ) (x m : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (a / (a - 1)) * (2^x - 2^(-x))

theorem f_properties (h1 : a > 0) (h2 : a ≠ 1) :
  -- f is an odd function
  (∀ x, f a (-x) = -f a x) ∧
  -- f is decreasing when 0 < a < 1
  ((0 < a ∧ a < 1) → (∀ x₁ x₂, x₁ < x₂ → f a x₁ > f a x₂)) ∧
  -- f is increasing when a > 1
  (a > 1 → (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂)) ∧
  -- For x ∈ (-1, 1), if f(m-1) + f(m) < 0, then:
  (∀ m, -1 < m ∧ m < 1 → f a (m-1) + f a m < 0 →
    ((0 < a ∧ a < 1 → 1/2 < m ∧ m < 1) ∧
     (a > 1 → 0 < m ∧ m < 1/2))) :=
by sorry

end f_properties_l3440_344019


namespace complement_intersection_theorem_l3440_344040

universe u

def U : Set (Fin 5) := {1, 2, 3, 4, 5}
def M : Set (Fin 5) := {1, 2}
def N : Set (Fin 5) := {3, 5}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3, 5} := by sorry

end complement_intersection_theorem_l3440_344040


namespace books_left_over_l3440_344073

/-- Calculates the number of books left over after filling a bookcase -/
theorem books_left_over
  (initial_books : ℕ)
  (shelves : ℕ)
  (books_per_shelf : ℕ)
  (new_books : ℕ)
  (h1 : initial_books = 56)
  (h2 : shelves = 4)
  (h3 : books_per_shelf = 20)
  (h4 : new_books = 26) :
  initial_books + new_books - (shelves * books_per_shelf) = 2 :=
by
  sorry

#check books_left_over

end books_left_over_l3440_344073


namespace klinker_double_age_l3440_344086

/-- Represents the current age of Mr. Klinker -/
def klinker_age : ℕ := 35

/-- Represents the current age of Mr. Klinker's daughter -/
def daughter_age : ℕ := 10

/-- Represents the number of years until Mr. Klinker is twice as old as his daughter -/
def years_until_double : ℕ := 15

/-- Proves that in 15 years, Mr. Klinker will be twice as old as his daughter -/
theorem klinker_double_age :
  klinker_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end klinker_double_age_l3440_344086


namespace gcd_111_1850_l3440_344090

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by sorry

end gcd_111_1850_l3440_344090


namespace angle_A_is_120_degrees_l3440_344024

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition given in the problem
def satisfiesCondition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

-- Theorem statement
theorem angle_A_is_120_degrees (t : Triangle) 
  (h : satisfiesCondition t) : t.A = 2 * π / 3 := by
  sorry


end angle_A_is_120_degrees_l3440_344024


namespace point_on_line_l3440_344007

/-- Given three points in the plane, this function checks if they are collinear -/
def are_collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that the point (14,7) lies on the line passing through (2,1) and (10,5) -/
theorem point_on_line : are_collinear 2 1 10 5 14 7 := by
  sorry

end point_on_line_l3440_344007


namespace cube_volume_problem_l3440_344065

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  s^3 - ((s + 2) * (s - 3) * s) = 8 →
  s^3 = 8 :=
by sorry

end cube_volume_problem_l3440_344065


namespace will_toy_purchase_l3440_344072

def max_toys_purchasable (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : ℕ :=
  ((initial_amount - game_cost) / toy_cost)

theorem will_toy_purchase : max_toys_purchasable 57 27 6 = 5 := by
  sorry

end will_toy_purchase_l3440_344072


namespace degenerate_iff_c_eq_52_l3440_344098

/-- A point in R^2 -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the graph -/
def equation (p : Point) (c : ℝ) : Prop :=
  3 * p.x^2 + p.y^2 + 6 * p.x - 14 * p.y + c = 0

/-- The graph is degenerate (represents a single point) -/
def is_degenerate (c : ℝ) : Prop :=
  ∃! p : Point, equation p c

theorem degenerate_iff_c_eq_52 :
  ∀ c : ℝ, is_degenerate c ↔ c = 52 := by sorry

end degenerate_iff_c_eq_52_l3440_344098


namespace saree_price_calculation_l3440_344057

/-- Calculates the final price after applying multiple discounts and a tax rate -/
def finalPrice (originalPrice : ℝ) (discounts : List ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := discounts.foldl (fun price discount => price * (1 - discount)) originalPrice
  discountedPrice * (1 + taxRate)

/-- Theorem: The final price of a 510 Rs item after specific discounts and tax is approximately 302.13 Rs -/
theorem saree_price_calculation :
  let originalPrice : ℝ := 510
  let discounts : List ℝ := [0.12, 0.15, 0.20, 0.10]
  let taxRate : ℝ := 0.10
  abs (finalPrice originalPrice discounts taxRate - 302.13) < 0.01 := by
  sorry

#eval finalPrice 510 [0.12, 0.15, 0.20, 0.10] 0.10

end saree_price_calculation_l3440_344057


namespace cycle_price_proof_l3440_344004

theorem cycle_price_proof (sale_price : ℝ) (gain_percentage : ℝ) 
  (h1 : sale_price = 1440)
  (h2 : gain_percentage = 60) : 
  ∃ original_price : ℝ, 
    original_price = 900 ∧ 
    sale_price = original_price + (gain_percentage / 100) * original_price :=
by
  sorry

end cycle_price_proof_l3440_344004


namespace dice_sum_product_l3440_344096

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 120 →
  a + b + c + d ≠ 14 :=
by sorry

end dice_sum_product_l3440_344096


namespace sin_45_degrees_l3440_344089

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end sin_45_degrees_l3440_344089


namespace sin_cos_sum_11_19_l3440_344088

theorem sin_cos_sum_11_19 : 
  Real.sin (11 * π / 180) * Real.cos (19 * π / 180) + 
  Real.cos (11 * π / 180) * Real.sin (19 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_11_19_l3440_344088


namespace fixed_point_of_exponential_function_l3440_344041

theorem fixed_point_of_exponential_function 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 7 + a^(x - 3)
  f 3 = 8 := by
sorry

end fixed_point_of_exponential_function_l3440_344041


namespace add_248_64_l3440_344026

theorem add_248_64 : 248 + 64 = 312 := by
  sorry

end add_248_64_l3440_344026


namespace trigonometric_inequality_l3440_344083

open Real

theorem trigonometric_inequality : 
  let a : ℝ := sin (46 * π / 180)
  let b : ℝ := cos (46 * π / 180)
  let c : ℝ := tan (46 * π / 180)
  c > a ∧ a > b :=
by sorry

end trigonometric_inequality_l3440_344083


namespace shoes_price_proof_l3440_344032

theorem shoes_price_proof (total_cost jersey_count shoe_count : ℕ) 
  (jersey_price_ratio : ℚ) (h1 : total_cost = 560) (h2 : jersey_count = 4) 
  (h3 : shoe_count = 6) (h4 : jersey_price_ratio = 1/4) : 
  shoe_count * (total_cost / (shoe_count + jersey_count * jersey_price_ratio)) = 480 := by
  sorry

end shoes_price_proof_l3440_344032


namespace base2_to_base4_conversion_l3440_344053

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

/-- The base 2 representation of the number -/
def base2Number : ℕ := 1011101100

/-- The expected base 4 representation of the number -/
def expectedBase4Number : ℕ := 23230

theorem base2_to_base4_conversion :
  base2ToBase4 base2Number = expectedBase4Number := by sorry

end base2_to_base4_conversion_l3440_344053


namespace parabola_vertex_coordinates_l3440_344087

/-- The vertex coordinates of the parabola y = 2 - (2x + 1)^2 are (-1/2, 2) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := 2 - (2*x + 1)^2
  ∃ (a b : ℝ), (∀ x, f x ≤ f a) ∧ f a = b ∧ a = -1/2 ∧ b = 2 := by
  sorry

end parabola_vertex_coordinates_l3440_344087


namespace simon_initial_stamps_l3440_344063

/-- The number of stamps Simon has after receiving stamps from friends -/
def total_stamps : ℕ := 61

/-- The number of stamps Simon received from friends -/
def received_stamps : ℕ := 27

/-- The number of stamps Simon initially had -/
def initial_stamps : ℕ := total_stamps - received_stamps

theorem simon_initial_stamps :
  initial_stamps = 34 := by sorry

end simon_initial_stamps_l3440_344063


namespace smallest_value_l3440_344039

def Q (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

theorem smallest_value (x₁ x₂ x₃ : ℝ) (hzeros : Q x₁ = 0 ∧ Q x₂ = 0 ∧ Q x₃ = 0) :
  min (min (Q (-1)) (1 + (-3) + (-9) + 2)) (min (x₁ * x₂ * x₃) (Q 1)) = x₁ * x₂ * x₃ :=
sorry

end smallest_value_l3440_344039


namespace point_on_axes_l3440_344038

theorem point_on_axes (a : ℝ) :
  let P : ℝ × ℝ := (2*a - 1, a + 2)
  (P.1 = 0 ∨ P.2 = 0) → (P = (-5, 0) ∨ P = (0, 2.5)) :=
by sorry

end point_on_axes_l3440_344038


namespace exactly_three_even_dice_probability_l3440_344095

def num_sides : ℕ := 12
def num_dice : ℕ := 4
def num_even_sides : ℕ := 6

def prob_even_on_one_die : ℚ := num_even_sides / num_sides

theorem exactly_three_even_dice_probability :
  (num_dice.choose 3) * (prob_even_on_one_die ^ 3) * ((1 - prob_even_on_one_die) ^ (num_dice - 3)) = 1/4 := by
  sorry

end exactly_three_even_dice_probability_l3440_344095


namespace carpet_width_l3440_344091

/-- Given a rectangular carpet with length 9 feet that covers 20% of a 180 square feet living room floor, prove that the width of the carpet is 4 feet. -/
theorem carpet_width (carpet_length : ℝ) (room_area : ℝ) (coverage_percent : ℝ) :
  carpet_length = 9 →
  room_area = 180 →
  coverage_percent = 20 →
  (coverage_percent / 100) * room_area / carpet_length = 4 := by
sorry

end carpet_width_l3440_344091


namespace triangle_max_area_l3440_344013

theorem triangle_max_area (A B C : Real) (a b c : Real) :
  -- Triangle ABC with circumradius 1
  (a / Real.sin A = 2) ∧ (b / Real.sin B = 2) ∧ (c / Real.sin C = 2) →
  -- Given condition
  (Real.tan A) / (Real.tan B) = (2 * c - b) / b →
  -- Theorem: Maximum area is √3/2
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
                S ≤ Real.sqrt 3 / 2 ∧
                (∃ (B' C' : Real), S = Real.sqrt 3 / 2)) :=
by sorry

end triangle_max_area_l3440_344013


namespace hide_and_seek_l3440_344009

-- Define the participants
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
variable (h1 : Andrew → (Boris ∧ ¬Vasya))
variable (h2 : Boris → (Gena ∨ Denis))
variable (h3 : ¬Vasya → (¬Boris ∧ ¬Denis))
variable (h4 : ¬Andrew → (Boris ∧ ¬Gena))

-- Theorem statement
theorem hide_and_seek :
  Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena := by sorry

end hide_and_seek_l3440_344009


namespace min_sum_squares_l3440_344008

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) 
  (pos₁ : y₁ > 0) (pos₂ : y₂ > 0) (pos₃ : y₃ > 0)
  (sum_constraint : y₁ + 3*y₂ + 4*y₃ = 72) :
  ∃ (min : ℝ), min = 2592/13 ∧ 
  ∀ (z₁ z₂ z₃ : ℝ), z₁ > 0 → z₂ > 0 → z₃ > 0 → 
  z₁ + 3*z₂ + 4*z₃ = 72 → 
  z₁^2 + z₂^2 + z₃^2 ≥ min :=
sorry

end min_sum_squares_l3440_344008


namespace function_critical_points_and_inequality_l3440_344018

open Real

noncomputable def f (a x : ℝ) : ℝ := (x - 2) * exp x - a * x^2 + 2 * a * x - 2 * a

theorem function_critical_points_and_inequality (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧
    (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, 0 < x → x < x₂ → f a x < -2 * a)) →
  a = exp 1 / 4 ∨ a = 2 * exp 1 / 3 :=
by sorry

end function_critical_points_and_inequality_l3440_344018


namespace calendar_sum_l3440_344062

/-- Given three consecutive numbers in a vertical column of a calendar where the top number is n,
    the sum of these three numbers is equal to 3n + 21. -/
theorem calendar_sum (n : ℕ) : n + (n + 7) + (n + 14) = 3 * n + 21 := by
  sorry

end calendar_sum_l3440_344062
