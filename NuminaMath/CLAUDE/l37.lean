import Mathlib

namespace NUMINAMATH_CALUDE_solve_maple_tree_price_l37_3746

/-- Represents the problem of calculating the price per maple tree --/
def maple_tree_price_problem (initial_cash : ℕ) (cypress_trees : ℕ) (pine_trees : ℕ) (maple_trees : ℕ)
  (cypress_price : ℕ) (pine_price : ℕ) (cabin_price : ℕ) (remaining_cash : ℕ) : Prop :=
  let total_after_sale := cabin_price + remaining_cash
  let total_from_trees := total_after_sale - initial_cash
  let cypress_revenue := cypress_trees * cypress_price
  let pine_revenue := pine_trees * pine_price
  let maple_revenue := total_from_trees - cypress_revenue - pine_revenue
  maple_revenue / maple_trees = 300

/-- The main theorem stating the solution to the problem --/
theorem solve_maple_tree_price :
  maple_tree_price_problem 150 20 600 24 100 200 129000 350 := by
  sorry

#check solve_maple_tree_price

end NUMINAMATH_CALUDE_solve_maple_tree_price_l37_3746


namespace NUMINAMATH_CALUDE_remainder_problem_l37_3753

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l37_3753


namespace NUMINAMATH_CALUDE_family_age_relations_l37_3757

structure Family where
  rachel_age : ℕ
  grandfather_age : ℕ
  mother_age : ℕ
  father_age : ℕ
  aunt_age : ℕ

def family_ages : Family where
  rachel_age := 12
  grandfather_age := 7 * 12
  mother_age := (7 * 12) / 2
  father_age := (7 * 12) / 2 + 5
  aunt_age := 7 * 12 - 8

theorem family_age_relations (f : Family) :
  f.rachel_age = 12 ∧
  f.grandfather_age = 7 * f.rachel_age ∧
  f.mother_age = f.grandfather_age / 2 ∧
  f.father_age = f.mother_age + 5 ∧
  f.aunt_age = f.grandfather_age - 8 →
  f = family_ages :=
by sorry

end NUMINAMATH_CALUDE_family_age_relations_l37_3757


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l37_3776

theorem inequality_system_solution_set :
  {x : ℝ | 2 * x - 1 > 0 ∧ x + 1 ≤ 3} = {x : ℝ | 1/2 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l37_3776


namespace NUMINAMATH_CALUDE_seaside_pet_care_arrangement_l37_3775

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs fishes : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial fishes

/-- Theorem stating the number of arrangements for the given problem -/
theorem seaside_pet_care_arrangement :
  arrange_animals 4 3 5 = 103680 := by
  sorry

end NUMINAMATH_CALUDE_seaside_pet_care_arrangement_l37_3775


namespace NUMINAMATH_CALUDE_min_dot_product_sum_l37_3760

/-- The ellipse on which point P moves --/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Circle E --/
def circle_E (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Circle F --/
def circle_F (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The dot product of vectors PA and PB plus the dot product of vectors PC and PD --/
def dot_product_sum (a b : ℝ) : ℝ := 2 * (a^2 + b^2)

theorem min_dot_product_sum :
  ∀ a b : ℝ, ellipse a b → 
  (∀ x y : ℝ, circle_E x y → ∀ u v : ℝ, circle_F u v → 
    dot_product_sum a b ≥ 6) ∧ 
  (∃ x y u v : ℝ, circle_E x y ∧ circle_F u v ∧ dot_product_sum a b = 6) :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_sum_l37_3760


namespace NUMINAMATH_CALUDE_arrange_balls_theorem_l37_3774

/-- The number of ways to arrange balls of different colors in a row -/
def arrangeMulticolorBalls (red : ℕ) (yellow : ℕ) (white : ℕ) : ℕ :=
  Nat.factorial (red + yellow + white) / (Nat.factorial red * Nat.factorial yellow * Nat.factorial white)

/-- Theorem: There are 1260 ways to arrange 2 red, 3 yellow, and 4 white indistinguishable balls in a row -/
theorem arrange_balls_theorem : arrangeMulticolorBalls 2 3 4 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_theorem_l37_3774


namespace NUMINAMATH_CALUDE_complex_simplification_l37_3728

theorem complex_simplification :
  let i : ℂ := Complex.I
  (1 + i)^2 / (2 - 3*i) = 6/5 - 4/5 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l37_3728


namespace NUMINAMATH_CALUDE_excess_amount_l37_3781

theorem excess_amount (x : ℝ) (a : ℝ) : 
  x = 0.16 * x + a → x = 50 → a = 42 := by
  sorry

end NUMINAMATH_CALUDE_excess_amount_l37_3781


namespace NUMINAMATH_CALUDE_custom_mult_equation_solution_l37_3770

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := a * b + a + b

/-- Theorem stating that if 3 * (3x - 1) = 27 under the custom multiplication, then x = 7/3 -/
theorem custom_mult_equation_solution :
  ∀ x : ℝ, custom_mult 3 (3 * x - 1) = 27 → x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_equation_solution_l37_3770


namespace NUMINAMATH_CALUDE_b_95_mod_49_l37_3783

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_95_mod_49 : b 95 ≡ 28 [ZMOD 49] := by sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l37_3783


namespace NUMINAMATH_CALUDE_delta_value_l37_3751

theorem delta_value (Δ : ℤ) (h : 5 * (-3) = Δ - 3) : Δ = -12 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l37_3751


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l37_3749

theorem arctan_sum_equation (y : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/y) = π/4 →
  y = -43/3 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l37_3749


namespace NUMINAMATH_CALUDE_perimeter_ratio_l37_3779

/-- Triangle PQR with sides of length 6, 8, and 10 units -/
def PQR : Fin 3 → ℝ := ![6, 8, 10]

/-- Triangle STU with sides of length 9, 12, and 15 units -/
def STU : Fin 3 → ℝ := ![9, 12, 15]

/-- Perimeter of a triangle given its side lengths -/
def perimeter (triangle : Fin 3 → ℝ) : ℝ :=
  triangle 0 + triangle 1 + triangle 2

/-- The ratio of the perimeter of triangle PQR to the perimeter of triangle STU is 2/3 -/
theorem perimeter_ratio :
  perimeter PQR / perimeter STU = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_l37_3779


namespace NUMINAMATH_CALUDE_circle_tangent_perpendicular_l37_3723

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Define a predicate to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

theorem circle_tangent_perpendicular (A B C : Point) :
  ¬collinear A B C →
  ∃ (α β γ : ℝ),
    (β + γ + angle B A C = π / 2 ∨ β + γ + angle B A C = -π / 2) ∧
    (γ + α + angle A B C = π / 2 ∨ γ + α + angle A B C = -π / 2) ∧
    (α + β + angle A C B = π / 2 ∨ α + β + angle A C B = -π / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_perpendicular_l37_3723


namespace NUMINAMATH_CALUDE_initial_sum_proof_l37_3799

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem initial_sum_proof (P : ℝ) (r : ℝ) : 
  simple_interest P r 5 = 1500 ∧ 
  simple_interest P (r + 0.05) 5 = 1750 → 
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_sum_proof_l37_3799


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l37_3731

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) :
  (360 : ℝ) / n = 60 → n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l37_3731


namespace NUMINAMATH_CALUDE_sqrt_18_minus_1_interval_l37_3793

theorem sqrt_18_minus_1_interval : 3 < Real.sqrt 18 - 1 ∧ Real.sqrt 18 - 1 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_1_interval_l37_3793


namespace NUMINAMATH_CALUDE_shortest_combined_track_length_l37_3720

def melanie_pieces : List Nat := [8, 12]
def martin_pieces : List Nat := [20, 30]
def area_width : Nat := 100
def area_length : Nat := 200

theorem shortest_combined_track_length :
  let melanie_gcd := melanie_pieces.foldl Nat.gcd 0
  let martin_gcd := martin_pieces.foldl Nat.gcd 0
  let common_segment := Nat.lcm melanie_gcd martin_gcd
  let length_segments := area_length / common_segment
  let width_segments := area_width / common_segment
  let total_segments := 2 * (length_segments + width_segments)
  let single_track_length := total_segments * common_segment
  single_track_length * 2 = 1200 := by
sorry


end NUMINAMATH_CALUDE_shortest_combined_track_length_l37_3720


namespace NUMINAMATH_CALUDE_percent_of_y_l37_3738

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 5 + (3 * y) / 10) / y = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l37_3738


namespace NUMINAMATH_CALUDE_concert_cost_for_two_l37_3785

def concert_cost (ticket_price : ℝ) (processing_fee_rate : ℝ) (parking_fee : ℝ) (entrance_fee : ℝ) (num_people : ℕ) : ℝ :=
  let total_ticket_price := ticket_price * num_people
  let processing_fee := total_ticket_price * processing_fee_rate
  let total_entrance_fee := entrance_fee * num_people
  total_ticket_price + processing_fee + parking_fee + total_entrance_fee

theorem concert_cost_for_two :
  concert_cost 50 0.15 10 5 2 = 135 := by
  sorry

end NUMINAMATH_CALUDE_concert_cost_for_two_l37_3785


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l37_3795

/-- Proves that for a sinusoidal function y = A sin(Bx) - C with given properties, A = 2 and C = 1 -/
theorem sinusoidal_function_properties (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (hMax : A - C = 3) (hMin : -A - C = -1) : A = 2 ∧ C = 1 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l37_3795


namespace NUMINAMATH_CALUDE_cubes_fill_box_l37_3726

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

end NUMINAMATH_CALUDE_cubes_fill_box_l37_3726


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l37_3763

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 6 →
  c = 3 →
  C = 60 * π / 180 →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a / Real.sin A = c / Real.sin C) →
  A + B + C = π →
  A = 75 * π / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l37_3763


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l37_3780

/-- Given a hyperbola and a parabola with specific properties, prove that the eccentricity of the hyperbola is √2 + 1 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 + b^2)) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 4 * c * x
  let A := (c, 2 * c)
  let B := (-c, -2 * c)
  (hyperbola A.1 A.2 ∧ parabola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ parabola B.1 B.2) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * c)^2 →
  let e := c / a
  e = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l37_3780


namespace NUMINAMATH_CALUDE_shirt_cost_l37_3798

theorem shirt_cost (total_cost coat_price shirt_price : ℚ) : 
  total_cost = 600 →
  shirt_price = (1 / 3) * coat_price →
  total_cost = shirt_price + coat_price →
  shirt_price = 150 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l37_3798


namespace NUMINAMATH_CALUDE_line_circle_intersection_l37_3768

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), (y = k * x + 1 ∧ x^2 + y^2 = 2) ∧ 
  ¬(∃ (x y : ℝ), y = k * x + 1 ∧ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l37_3768


namespace NUMINAMATH_CALUDE_book_chunk_sheets_l37_3732

/-- Checks if two numbers have the same digits (possibly in different order) -/
def sameDigits (a b : Nat) : Bool := sorry

/-- Finds the smallest even number greater than n composed of the same digits as n -/
def smallestEvenWithSameDigits (n : Nat) : Nat := sorry

theorem book_chunk_sheets (first_page last_page : Nat) : 
  first_page = 163 →
  last_page = smallestEvenWithSameDigits first_page →
  (last_page - first_page + 1) / 2 = 77 := by sorry

end NUMINAMATH_CALUDE_book_chunk_sheets_l37_3732


namespace NUMINAMATH_CALUDE_number_of_valid_paths_l37_3741

-- Define the grid dimensions
def rows : Nat := 4
def columns : Nat := 10

-- Define the total number of moves
def total_moves : Nat := rows + columns - 2

-- Define the number of unrestricted paths
def unrestricted_paths : Nat := Nat.choose total_moves (rows - 1)

-- Define the number of paths through the first forbidden segment
def forbidden_paths1 : Nat := 360

-- Define the number of paths through the second forbidden segment
def forbidden_paths2 : Nat := 420

-- Theorem statement
theorem number_of_valid_paths :
  unrestricted_paths - forbidden_paths1 - forbidden_paths2 = 221 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_paths_l37_3741


namespace NUMINAMATH_CALUDE_rocky_knockout_percentage_l37_3747

/-- Proves that the percentage of Rocky's knockouts that were in the first round is 20% -/
theorem rocky_knockout_percentage : 
  ∀ (total_fights : ℕ) 
    (knockout_percentage : ℚ) 
    (first_round_knockouts : ℕ),
  total_fights = 190 →
  knockout_percentage = 1/2 →
  first_round_knockouts = 19 →
  (first_round_knockouts : ℚ) / (knockout_percentage * total_fights) = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rocky_knockout_percentage_l37_3747


namespace NUMINAMATH_CALUDE_triangle_theorem_l37_3722

noncomputable section

variables {a b c : ℝ} {A B C : Real}

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b + t.c = 2 * t.a) 
  (h2 : 3 * t.c * Real.sin t.B = 4 * t.a * Real.sin t.C) : 
  Real.cos t.B = -1/4 ∧ Real.sin (2 * t.B + π/6) = -(3 * Real.sqrt 5 + 7)/16 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l37_3722


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l37_3744

theorem basketball_score_ratio : 
  ∀ (marks_two_pointers marks_three_pointers marks_free_throws : ℕ)
    (total_points : ℕ),
  marks_two_pointers = 25 →
  marks_three_pointers = 8 →
  marks_free_throws = 10 →
  total_points = 201 →
  ∃ (ratio : ℚ),
    ratio = 1/2 ∧
    (2 * marks_two_pointers * 2 + ratio * (marks_three_pointers * 3 + marks_free_throws)) +
    (marks_two_pointers * 2 + marks_three_pointers * 3 + marks_free_throws) = total_points :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_ratio_l37_3744


namespace NUMINAMATH_CALUDE_number_of_students_in_class_l37_3703

/-- Represents the problem of calculating the number of students in a class based on their payments for a science project. -/
theorem number_of_students_in_class
  (full_payment : ℕ)
  (half_payment : ℕ)
  (num_half_payers : ℕ)
  (total_collected : ℕ)
  (h1 : full_payment = 50)
  (h2 : half_payment = 25)
  (h3 : num_half_payers = 4)
  (h4 : total_collected = 1150) :
  ∃ (num_students : ℕ),
    num_students * full_payment - num_half_payers * (full_payment - half_payment) = total_collected ∧
    num_students = 25 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_in_class_l37_3703


namespace NUMINAMATH_CALUDE_hyperbola_equation_l37_3792

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its eccentricity is √3 and the directrix of the parabola y² = 12x
    passes through one of its foci, then the equation of the hyperbola is
    x²/3 - y²/6 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c / a = Real.sqrt 3 ∧ c = 3) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l37_3792


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l37_3765

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l37_3765


namespace NUMINAMATH_CALUDE_least_possible_b_in_right_triangle_l37_3714

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fib n + fib (n + 1)

-- Define a predicate to check if a number is in the Fibonacci sequence
def is_fibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

theorem least_possible_b_in_right_triangle :
  ∀ a b : ℕ,
  a + b = 90 →  -- Sum of acute angles in a right triangle is 90°
  a > b →  -- a is greater than b
  is_fibonacci a →  -- a is in the Fibonacci sequence
  is_fibonacci b →  -- b is in the Fibonacci sequence
  b ≥ 1 →  -- b is at least 1 (as it's an angle)
  ∀ c : ℕ, (c < b ∧ is_fibonacci c) → c < 1 :=
by sorry

#check least_possible_b_in_right_triangle

end NUMINAMATH_CALUDE_least_possible_b_in_right_triangle_l37_3714


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l37_3710

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x₀ : ℝ, a * x₀^2 + 2 * x₀ + 1 < 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l37_3710


namespace NUMINAMATH_CALUDE_nested_radical_fifteen_l37_3707

theorem nested_radical_fifteen (x : ℝ) : x = Real.sqrt (15 + x) → x = (1 + Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_fifteen_l37_3707


namespace NUMINAMATH_CALUDE_evaluate_expression_l37_3719

theorem evaluate_expression : (900^2 : ℚ) / (200^2 - 196^2) = 511 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l37_3719


namespace NUMINAMATH_CALUDE_no_multiples_of_5005_l37_3733

theorem no_multiples_of_5005 : ¬∃ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 49 ∧ 
  ∃ (k : ℕ+), 5005 * k = 10^j - 10^i := by
  sorry

end NUMINAMATH_CALUDE_no_multiples_of_5005_l37_3733


namespace NUMINAMATH_CALUDE_equation_solution_l37_3750

theorem equation_solution : 
  ∀ y : ℝ, (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l37_3750


namespace NUMINAMATH_CALUDE_sequence_convergence_to_one_l37_3777

theorem sequence_convergence_to_one (a : ℕ → ℕ) 
  (h : ∀ (m n : ℕ), (a (m + n)) ∣ (a m * a n - 1)) : 
  ∃ C : ℕ, ∀ k : ℕ, k > C → a k = 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_convergence_to_one_l37_3777


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l37_3766

theorem area_between_concentric_circles (r : ℝ) (h1 : r > 0) :
  let R := 3 * r
  R - r = 3 →
  π * R^2 - π * r^2 = 18 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l37_3766


namespace NUMINAMATH_CALUDE_hair_cut_total_l37_3784

theorem hair_cut_total (day1 : Float) (day2 : Float) (h1 : day1 = 0.38) (h2 : day2 = 0.5) :
  day1 + day2 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_total_l37_3784


namespace NUMINAMATH_CALUDE_subway_speed_problem_l37_3787

theorem subway_speed_problem :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ (t^2 + 2*t) - (3^2 + 2*3) = 20 ∧ t = 5 := by
sorry

end NUMINAMATH_CALUDE_subway_speed_problem_l37_3787


namespace NUMINAMATH_CALUDE_question_mark_value_l37_3789

theorem question_mark_value : ∃ (x : ℕ), x * 40 = 173 * 240 ∧ x = 1036 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l37_3789


namespace NUMINAMATH_CALUDE_binomial_18_12_l37_3762

theorem binomial_18_12 (h1 : Nat.choose 17 10 = 19448)
                        (h2 : Nat.choose 17 11 = 12376)
                        (h3 : Nat.choose 19 12 = 50388) :
  Nat.choose 18 12 = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_12_l37_3762


namespace NUMINAMATH_CALUDE_sugar_water_dilution_l37_3717

theorem sugar_water_dilution (initial_weight : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_weight = 300 →
  initial_concentration = 0.08 →
  final_concentration = 0.05 →
  initial_concentration * initial_weight = final_concentration * (initial_weight + water_added) →
  water_added = 180 := by
sorry

end NUMINAMATH_CALUDE_sugar_water_dilution_l37_3717


namespace NUMINAMATH_CALUDE_coin_combinations_eq_20_l37_3724

/-- The number of combinations of coins that sum to 50 cents -/
def coin_combinations : ℕ :=
  let penny_value : ℕ := 1
  let nickel_value : ℕ := 5
  let quarter_value : ℕ := 25
  let target_value : ℕ := 50
  (Finset.filter (fun (p, n, q) => 
    p * penny_value + n * nickel_value + q * quarter_value = target_value)
    (Finset.product (Finset.range (target_value + 1))
      (Finset.product (Finset.range (target_value / nickel_value + 1))
        (Finset.range (target_value / quarter_value + 1))))).card

/-- Theorem stating that the number of coin combinations is 20 -/
theorem coin_combinations_eq_20 : coin_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_coin_combinations_eq_20_l37_3724


namespace NUMINAMATH_CALUDE_course_ratio_l37_3754

theorem course_ratio (max_courses sid_courses : ℕ) (m : ℚ) : 
  max_courses = 40 →
  max_courses + sid_courses = 200 →
  sid_courses = m * max_courses →
  m = 4 ∧ sid_courses / max_courses = 4 := by
  sorry

end NUMINAMATH_CALUDE_course_ratio_l37_3754


namespace NUMINAMATH_CALUDE_factorization_proof_l37_3739

theorem factorization_proof (x : ℝ) : 
  3 * x^2 * (x - 2) + 4 * x * (x - 2) + 2 * (x - 2) = (x - 2) * (x + 2) * (3 * x + 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l37_3739


namespace NUMINAMATH_CALUDE_no_common_elements_l37_3706

theorem no_common_elements : ¬∃ (n m : ℕ), n^2 - 1 = m^2 + 1 := by sorry

end NUMINAMATH_CALUDE_no_common_elements_l37_3706


namespace NUMINAMATH_CALUDE_batsman_average_increase_l37_3761

def batsman_average (total_runs : ℕ) (innings : ℕ) : ℚ :=
  (total_runs : ℚ) / (innings : ℚ)

theorem batsman_average_increase
  (prev_total : ℕ)
  (new_score : ℕ)
  (innings : ℕ)
  (avg_increase : ℚ)
  (h1 : innings = 17)
  (h2 : new_score = 88)
  (h3 : avg_increase = 3)
  (h4 : batsman_average (prev_total + new_score) innings - batsman_average prev_total (innings - 1) = avg_increase) :
  batsman_average (prev_total + new_score) innings = 40 :=
by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l37_3761


namespace NUMINAMATH_CALUDE_proposal_i_percentage_l37_3786

def survey_results (P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all : ℝ) : Prop :=
  P_i + P_ii + P_iii - P_i_and_ii - P_i_and_iii - P_ii_and_iii + P_all = 78 ∧
  P_ii = 30 ∧
  P_iii = 20 ∧
  P_all = 5 ∧
  P_i_and_ii + P_i_and_iii + P_ii_and_iii = 32

theorem proposal_i_percentage :
  ∀ P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all : ℝ,
  survey_results P_i P_ii P_iii P_i_and_ii P_i_and_iii P_ii_and_iii P_all →
  P_i = 55 :=
by sorry

end NUMINAMATH_CALUDE_proposal_i_percentage_l37_3786


namespace NUMINAMATH_CALUDE_inequality_proof_l37_3715

theorem inequality_proof (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2) :
  (abs (a + b + c - a * b * c) ≤ 2) ∧
  (abs (a^3 + b^3 + c^3 - 3 * a * b * c) ≤ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l37_3715


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l37_3794

def A : Set ℝ := {0, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 2}

theorem intersection_implies_a_zero (a : ℝ) :
  A ∩ B a = {1} → a = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l37_3794


namespace NUMINAMATH_CALUDE_sqrt_360000_l37_3704

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_l37_3704


namespace NUMINAMATH_CALUDE_missing_number_in_proportion_l37_3745

theorem missing_number_in_proportion : 
  ∃ x : ℚ, (2 : ℚ) / x = (4 : ℚ) / 3 / (10 : ℚ) / 3 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_in_proportion_l37_3745


namespace NUMINAMATH_CALUDE_sum_of_cubes_product_l37_3797

theorem sum_of_cubes_product (x y : ℤ) : x^3 + y^3 = 189 → x * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_product_l37_3797


namespace NUMINAMATH_CALUDE_range_of_m_l37_3773

theorem range_of_m (m : ℝ) : 
  (|m + 3| = m + 3) →
  (|3*m + 9| ≥ 4*m - 3 ↔ -3 ≤ m ∧ m ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l37_3773


namespace NUMINAMATH_CALUDE_convex_polyhedron_symmetry_l37_3772

-- Define a structure for a polyhedron
structure Polyhedron where
  -- Add necessary fields (omitted for simplicity)

-- Define a property for convexity
def is_convex (p : Polyhedron) : Prop :=
  sorry

-- Define a property for central symmetry of faces
def has_centrally_symmetric_faces (p : Polyhedron) : Prop :=
  sorry

-- Define a property for subdivision into smaller polyhedra
def can_be_subdivided (p : Polyhedron) (subdivisions : List Polyhedron) : Prop :=
  sorry

-- Main theorem
theorem convex_polyhedron_symmetry 
  (p : Polyhedron) 
  (subdivisions : List Polyhedron) :
  is_convex p → 
  can_be_subdivided p subdivisions → 
  (∀ sub ∈ subdivisions, has_centrally_symmetric_faces sub) → 
  has_centrally_symmetric_faces p :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_symmetry_l37_3772


namespace NUMINAMATH_CALUDE_complex_square_l37_3718

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square : (1 + i)^2 = 2*i := by sorry

end NUMINAMATH_CALUDE_complex_square_l37_3718


namespace NUMINAMATH_CALUDE_road_length_calculation_l37_3730

/-- Given a map scale and a road length on the map, calculate the actual road length in kilometers. -/
def actual_road_length (map_scale : ℚ) (map_length : ℚ) : ℚ :=
  map_length * map_scale / 100000

/-- Theorem stating that for a map scale of 1:2500000 and a road length of 6 cm on the map,
    the actual length of the road is 150 km. -/
theorem road_length_calculation :
  let map_scale : ℚ := 2500000
  let map_length : ℚ := 6
  actual_road_length map_scale map_length = 150 := by
  sorry

#eval actual_road_length 2500000 6

end NUMINAMATH_CALUDE_road_length_calculation_l37_3730


namespace NUMINAMATH_CALUDE_five_colored_flags_count_l37_3764

/-- The number of different colors available -/
def num_colors : ℕ := 11

/-- The number of stripes in the flag -/
def num_stripes : ℕ := 5

/-- The number of ways to choose and arrange colors for the flag -/
def num_flags : ℕ := (num_colors.choose num_stripes) * num_stripes.factorial

/-- Theorem stating the number of different five-colored flags -/
theorem five_colored_flags_count : num_flags = 55440 := by
  sorry

end NUMINAMATH_CALUDE_five_colored_flags_count_l37_3764


namespace NUMINAMATH_CALUDE_wire_cutting_l37_3721

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 70 ∧ 
  ratio = 2 / 5 ∧ 
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l37_3721


namespace NUMINAMATH_CALUDE_middle_income_sample_size_l37_3708

/-- Calculates the number of middle-income households to be sampled in a stratified sampling method. -/
theorem middle_income_sample_size 
  (total_households : ℕ) 
  (middle_income_households : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_households = 600) 
  (h2 : middle_income_households = 360) 
  (h3 : sample_size = 80) :
  (middle_income_households : ℚ) / (total_households : ℚ) * (sample_size : ℚ) = 48 := by
  sorry

end NUMINAMATH_CALUDE_middle_income_sample_size_l37_3708


namespace NUMINAMATH_CALUDE_incorrect_page_number_l37_3705

theorem incorrect_page_number (n : ℕ) (x : ℕ) : 
  (n ≥ 1) →
  (x ≤ n) →
  (n * (n + 1) / 2 + x = 2076) →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_incorrect_page_number_l37_3705


namespace NUMINAMATH_CALUDE_jimmy_has_five_figures_l37_3702

/-- Represents the collection of action figures Jimmy has --/
structure ActionFigures where
  regular : ℕ  -- number of regular figures worth $15
  special : ℕ  -- number of special figures worth $20
  h_special : special = 1  -- there is exactly one special figure

/-- The total value of the collection before the price reduction --/
def total_value (af : ActionFigures) : ℕ :=
  15 * af.regular + 20 * af.special

/-- The total earnings after selling all figures with $5 discount --/
def total_earnings (af : ActionFigures) : ℕ :=
  10 * af.regular + 15 * af.special

/-- Theorem stating that Jimmy has 5 action figures in total --/
theorem jimmy_has_five_figures :
  ∃ (af : ActionFigures), total_earnings af = 55 ∧ af.regular + af.special = 5 :=
sorry

end NUMINAMATH_CALUDE_jimmy_has_five_figures_l37_3702


namespace NUMINAMATH_CALUDE_all_terms_are_integers_l37_3790

/-- An infinite increasing arithmetic progression where the product of any two distinct terms is also a term in the progression. -/
structure SpecialArithmeticProgression where
  -- The sequence of terms in the progression
  sequence : ℕ → ℚ
  -- The common difference of the progression
  common_difference : ℚ
  -- The progression is increasing
  increasing : ∀ n : ℕ, sequence n < sequence (n + 1)
  -- The progression follows the arithmetic sequence formula
  is_arithmetic : ∀ n : ℕ, sequence (n + 1) = sequence n + common_difference
  -- The product of any two distinct terms is also a term
  product_is_term : ∀ m n : ℕ, m ≠ n → ∃ k : ℕ, sequence m * sequence n = sequence k

/-- All terms in a SpecialArithmeticProgression are integers. -/
theorem all_terms_are_integers (ap : SpecialArithmeticProgression) : 
  ∀ n : ℕ, ∃ k : ℤ, ap.sequence n = k :=
sorry

end NUMINAMATH_CALUDE_all_terms_are_integers_l37_3790


namespace NUMINAMATH_CALUDE_trigonometric_identities_l37_3755

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.cos α)^2 = 1/5 ∧
  (Real.sin α / (Real.sin α + Real.cos α) = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l37_3755


namespace NUMINAMATH_CALUDE_reflection_about_x_axis_l37_3771

/-- The reflection of a line about the x-axis -/
def reflect_about_x_axis (line : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  λ x y => line x (-y)

/-- The original line -/
def original_line : ℝ → ℝ → Prop :=
  λ x y => x - y + 1 = 0

/-- The reflected line -/
def reflected_line : ℝ → ℝ → Prop :=
  λ x y => x + y + 1 = 0

theorem reflection_about_x_axis :
  reflect_about_x_axis original_line = reflected_line := by
  sorry

end NUMINAMATH_CALUDE_reflection_about_x_axis_l37_3771


namespace NUMINAMATH_CALUDE_smallest_denominator_for_repeating_2015_l37_3778

/-- Given positive integers a and b where a/b is a repeating decimal with the sequence 2015,
    the smallest possible value of b is 129. -/
theorem smallest_denominator_for_repeating_2015 (a b : ℕ+) :
  (∃ k : ℕ, (a : ℚ) / b = 2015 / (10000 ^ k - 1)) →
  (∀ c : ℕ+, c < b → ¬∃ d : ℕ+, (d : ℚ) / c = 2015 / 9999) →
  b = 129 := by
  sorry


end NUMINAMATH_CALUDE_smallest_denominator_for_repeating_2015_l37_3778


namespace NUMINAMATH_CALUDE_natashas_journey_l37_3767

/-- Natasha's hill climbing problem -/
theorem natashas_journey (time_up : ℝ) (time_down : ℝ) (speed_up : ℝ) :
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (speed_up * time_up * 2) / (time_up + time_down) = 3 :=
by sorry

end NUMINAMATH_CALUDE_natashas_journey_l37_3767


namespace NUMINAMATH_CALUDE_window_width_theorem_l37_3740

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the configuration of a window -/
structure Window where
  pane : Pane
  grid_width : Nat
  grid_height : Nat
  border_width : ℝ

/-- Calculates the total width of a window -/
def total_window_width (w : Window) : ℝ :=
  w.grid_width * w.pane.width + (w.grid_width + 1) * w.border_width

/-- Theorem stating the total width of the window -/
theorem window_width_theorem (w : Window) 
  (h1 : w.grid_width = 3)
  (h2 : w.grid_height = 3)
  (h3 : w.border_width = 3)
  : total_window_width w = 3 * w.pane.width + 12 := by
  sorry

#check window_width_theorem

end NUMINAMATH_CALUDE_window_width_theorem_l37_3740


namespace NUMINAMATH_CALUDE_consecutive_integer_product_l37_3756

theorem consecutive_integer_product (n : ℤ) : 
  (6 ∣ n * (n + 1)) ∨ (n * (n + 1) % 18 = 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_l37_3756


namespace NUMINAMATH_CALUDE_cos_equality_with_large_angle_l37_3759

theorem cos_equality_with_large_angle (n : ℕ) :
  0 ≤ n ∧ n ≤ 200 →
  n = 166 →
  Real.cos (n * π / 180) = Real.cos (1274 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_cos_equality_with_large_angle_l37_3759


namespace NUMINAMATH_CALUDE_max_consecutive_sum_under_1000_l37_3742

theorem max_consecutive_sum_under_1000 : 
  (∀ k : ℕ, k ≤ 44 → k * (k + 1) ≤ 2000) ∧ 
  45 * 46 > 2000 := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_under_1000_l37_3742


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l37_3729

theorem min_value_of_fraction (x : ℝ) (h : x ≥ 3/2) :
  (2*x^2 - 2*x + 1) / (x - 1) ≥ 2*Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l37_3729


namespace NUMINAMATH_CALUDE_count_random_events_l37_3735

-- Define the type for events
inductive Event
  | throwDice : Event
  | pearFall : Event
  | winLottery : Event
  | haveBoy : Event
  | waterBoil : Event

-- Define a function to determine if an event is random
def isRandom (e : Event) : Bool :=
  match e with
  | Event.throwDice => true
  | Event.pearFall => false
  | Event.winLottery => true
  | Event.haveBoy => true
  | Event.waterBoil => false

-- Define the list of all events
def allEvents : List Event :=
  [Event.throwDice, Event.pearFall, Event.winLottery, Event.haveBoy, Event.waterBoil]

-- State the theorem
theorem count_random_events :
  (allEvents.filter isRandom).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_random_events_l37_3735


namespace NUMINAMATH_CALUDE_roots_of_quadratic_expression_l37_3788

theorem roots_of_quadratic_expression (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - x₁ - 2023 = 0) 
  (h₂ : x₂^2 - x₂ - 2023 = 0) : 
  x₁^3 - 2023*x₁ + x₂^2 = 4047 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_expression_l37_3788


namespace NUMINAMATH_CALUDE_max_value_inequality_l37_3713

theorem max_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (4 * x * z + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l37_3713


namespace NUMINAMATH_CALUDE_charles_vowel_learning_time_l37_3782

/-- The number of days Charles takes to learn one alphabet. -/
def days_per_alphabet : ℕ := 7

/-- The number of vowels in the English alphabet. -/
def number_of_vowels : ℕ := 5

/-- The total number of days Charles needs to finish learning all vowels. -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem charles_vowel_learning_time : total_days = 35 := by
  sorry

end NUMINAMATH_CALUDE_charles_vowel_learning_time_l37_3782


namespace NUMINAMATH_CALUDE_ordering_from_log_half_inequalities_l37_3736

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem ordering_from_log_half_inequalities 
  (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : log_half b < log_half a) 
  (h5 : log_half a < log_half c) : 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ordering_from_log_half_inequalities_l37_3736


namespace NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l37_3752

theorem smallest_value_for_y_between_zero_and_one 
  (y : ℝ) (h1 : 0 < y) (h2 : y < 1) :
  y^3 ≤ min (2*y) (min (3*y) (min (y^(1/3)) (1/y))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l37_3752


namespace NUMINAMATH_CALUDE_apartment_counts_equation_l37_3700

/-- Represents the number of apartments of each type in a building -/
structure ApartmentCounts where
  studio : ℝ
  twoPerson : ℝ
  threePerson : ℝ
  fourPerson : ℝ
  fivePerson : ℝ

/-- The apartment complex configuration -/
structure ApartmentComplex where
  buildingCount : ℕ
  maxOccupancy : ℕ
  occupancyRate : ℝ
  studioCapacity : ℝ
  twoPersonCapacity : ℝ
  threePersonCapacity : ℝ
  fourPersonCapacity : ℝ
  fivePersonCapacity : ℝ

/-- Theorem stating the equation for apartment counts given the complex configuration -/
theorem apartment_counts_equation (complex : ApartmentComplex) 
    (counts : ApartmentCounts) : 
    complex.buildingCount = 8 ∧ 
    complex.maxOccupancy = 3000 ∧ 
    complex.occupancyRate = 0.9 ∧
    complex.studioCapacity = 0.95 ∧
    complex.twoPersonCapacity = 0.85 ∧
    complex.threePersonCapacity = 0.8 ∧
    complex.fourPersonCapacity = 0.75 ∧
    complex.fivePersonCapacity = 0.65 →
    0.11875 * counts.studio + 0.2125 * counts.twoPerson + 
    0.3 * counts.threePerson + 0.375 * counts.fourPerson + 
    0.40625 * counts.fivePerson = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_apartment_counts_equation_l37_3700


namespace NUMINAMATH_CALUDE_initial_customers_count_l37_3725

/-- The number of customers who left the restaurant -/
def customers_left : ℕ := 11

/-- The number of customers who remained in the restaurant -/
def customers_remained : ℕ := 3

/-- The initial number of customers in the restaurant -/
def initial_customers : ℕ := customers_left + customers_remained

theorem initial_customers_count : initial_customers = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_count_l37_3725


namespace NUMINAMATH_CALUDE_f_properties_l37_3748

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x, f x b c = -f (-x) b c → c = 0) ∧
  (∃! x, f x 0 c = 0) ∧
  (∀ x, f (-x) b c + f x b c = 2 * c) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l37_3748


namespace NUMINAMATH_CALUDE_number_not_divisible_by_8_and_digit_product_l37_3727

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

end NUMINAMATH_CALUDE_number_not_divisible_by_8_and_digit_product_l37_3727


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l37_3743

-- Define the curve
def curve (x y : ℝ) : Prop := x * y - x + 2 * y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define the tangent line at point A
def tangent_line (x y : ℝ) : Prop := x + 3 * y - 7 = 0

-- Theorem statement
theorem tangent_triangle_area : 
  curve point_A.1 point_A.2 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    tangent_line x₁ y₁ ∧ 
    tangent_line x₂ y₂ ∧ 
    x₁ = 0 ∧ 
    y₂ = 0 ∧ 
    (1/2 * x₂ * y₁ = 49/6)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l37_3743


namespace NUMINAMATH_CALUDE_polynomial_roots_l37_3709

def P (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem polynomial_roots :
  (∀ x < 0, P x > 0) ∧ (∃ x > 0, P x = 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l37_3709


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l37_3791

/-- Represents a company with employees -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  male_employees : ℕ
  h_total : total_employees = female_employees + male_employees

/-- Represents a stratified sample from a company -/
structure StratifiedSample where
  company : Company
  sample_size : ℕ
  female_sample : ℕ
  male_sample : ℕ
  h_sample : sample_size = female_sample + male_sample
  h_proportion : female_sample * company.total_employees = company.female_employees * sample_size

theorem stratified_sample_male_count (c : Company) (s : StratifiedSample) 
    (h_company : c.total_employees = 300 ∧ c.female_employees = 160)
    (h_sample : s.company = c ∧ s.sample_size = 15) :
    s.male_sample = 7 := by
  sorry

#check stratified_sample_male_count

end NUMINAMATH_CALUDE_stratified_sample_male_count_l37_3791


namespace NUMINAMATH_CALUDE_mod_power_minus_three_l37_3716

theorem mod_power_minus_three (m : ℕ) : 
  0 ≤ m ∧ m < 37 ∧ (4 * m) % 37 = 1 → 
  (((3 ^ m) ^ 4 - 3) : ℤ) % 37 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mod_power_minus_three_l37_3716


namespace NUMINAMATH_CALUDE_intersecting_planes_parallel_line_l37_3769

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection relation for planes
variable (intersect : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Theorem statement
theorem intersecting_planes_parallel_line 
  (α β : Plane) 
  (h_intersect : intersect α β) :
  ∃ l : Line, parallel l α ∧ parallel l β := by
  sorry

end NUMINAMATH_CALUDE_intersecting_planes_parallel_line_l37_3769


namespace NUMINAMATH_CALUDE_textbook_cost_ratio_l37_3711

/-- The ratio of the cost of bookstore textbooks to online ordered books -/
theorem textbook_cost_ratio : 
  ∀ (sale_price online_price bookstore_price total_price : ℕ),
  sale_price = 5 * 10 →
  online_price = 40 →
  total_price = 210 →
  bookstore_price = total_price - sale_price - online_price →
  ∃ (k : ℕ), bookstore_price = k * online_price →
  (bookstore_price : ℚ) / online_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_textbook_cost_ratio_l37_3711


namespace NUMINAMATH_CALUDE_function_property_implications_l37_3701

def FunctionProperty (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = f (x^2 + x + 1)

theorem function_property_implications
  (f : ℤ → ℤ) (h : FunctionProperty f) :
  ((∀ x : ℤ, f x = f (-x)) → (∃ c : ℤ, ∀ x : ℤ, f x = c)) ∧
  ((∀ x : ℤ, f (-x) = -f x) → (∀ x : ℤ, f x = 0)) := by
  sorry

end NUMINAMATH_CALUDE_function_property_implications_l37_3701


namespace NUMINAMATH_CALUDE_basil_plants_theorem_l37_3758

/-- Calculates the number of basil plants sold given the costs, selling price, and net profit -/
def basil_plants_sold (seed_cost potting_soil_cost selling_price net_profit : ℚ) : ℚ :=
  (net_profit + seed_cost + potting_soil_cost) / selling_price

/-- Theorem stating that the number of basil plants sold is 20 -/
theorem basil_plants_theorem :
  basil_plants_sold 2 8 5 90 = 20 := by
  sorry

end NUMINAMATH_CALUDE_basil_plants_theorem_l37_3758


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l37_3734

/-- Given a line passing through points (1,3) and (4,-2), prove that the sum of its slope and y-intercept is -1/3 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℚ), 
  (3 : ℚ) = m * 1 + b →  -- Point (1,3) satisfies the equation
  (-2 : ℚ) = m * 4 + b → -- Point (4,-2) satisfies the equation
  m + b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l37_3734


namespace NUMINAMATH_CALUDE_stating_teacher_duty_arrangements_l37_3796

/-- Represents the number of science teachers -/
def num_science_teachers : ℕ := 6

/-- Represents the number of liberal arts teachers -/
def num_liberal_arts_teachers : ℕ := 2

/-- Represents the number of days for duty arrangement -/
def num_days : ℕ := 3

/-- Represents the number of science teachers required per day -/
def science_teachers_per_day : ℕ := 2

/-- Represents the number of liberal arts teachers required per day -/
def liberal_arts_teachers_per_day : ℕ := 1

/-- Represents the minimum number of days a teacher should be on duty -/
def min_duty_days : ℕ := 1

/-- Represents the maximum number of days a teacher can be on duty -/
def max_duty_days : ℕ := 2

/-- 
Calculates the number of different arrangements for the teacher duty roster
given the specified conditions
-/
def num_arrangements : ℕ := 540

/-- 
Theorem stating that the number of different arrangements for the teacher duty roster
is equal to 540, given the specified conditions
-/
theorem teacher_duty_arrangements :
  num_arrangements = 540 :=
by sorry

end NUMINAMATH_CALUDE_stating_teacher_duty_arrangements_l37_3796


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l37_3737

-- Problem 1
theorem problem_1 : 2 * (Real.sqrt 5 - 1) - Real.sqrt 5 = Real.sqrt 5 - 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 3 * (Real.sqrt 3 + 4 / Real.sqrt 3) = 7 := by sorry

-- Problem 3
theorem problem_3 : |Real.sqrt 3 - 2| + 3 - 27 + Real.sqrt ((-5)^2) = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l37_3737


namespace NUMINAMATH_CALUDE_caffeine_content_proof_l37_3712

/-- The amount of caffeine in one energy drink -/
def caffeine_per_drink : ℕ := sorry

/-- The maximum safe amount of caffeine per day -/
def max_safe_caffeine : ℕ := 500

/-- The number of energy drinks Brandy consumes -/
def num_drinks : ℕ := 4

/-- The additional amount of caffeine Brandy can safely consume after drinking the energy drinks -/
def additional_safe_caffeine : ℕ := 20

theorem caffeine_content_proof :
  caffeine_per_drink * num_drinks + additional_safe_caffeine = max_safe_caffeine ∧
  caffeine_per_drink = 120 := by sorry

end NUMINAMATH_CALUDE_caffeine_content_proof_l37_3712
