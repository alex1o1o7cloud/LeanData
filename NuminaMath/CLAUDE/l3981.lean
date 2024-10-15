import Mathlib

namespace NUMINAMATH_CALUDE_common_face_sum_l3981_398123

/-- Represents a cube with numbers at its vertices -/
structure NumberedCube where
  vertices : Fin 8 → Nat
  additional : Fin 8 → Nat

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : Nat) : Nat := n * (n + 1) / 2

/-- The sum of numbers on a face of the cube -/
def face_sum (cube : NumberedCube) (face : Fin 6) : Nat :=
  sorry -- Definition of face sum

/-- The theorem stating the common sum on each face -/
theorem common_face_sum (cube : NumberedCube) : 
  (∀ (i j : Fin 6), face_sum cube i = face_sum cube j) → 
  (∀ (i : Fin 8), cube.vertices i ∈ Finset.range 9) →
  (∀ (i : Fin 6), face_sum cube i = 9) :=
sorry

end NUMINAMATH_CALUDE_common_face_sum_l3981_398123


namespace NUMINAMATH_CALUDE_intersection_locus_l3981_398135

/-- Given two fixed points A(a, 0) and B(b, 0) on the x-axis, and a moving point C(0, c) on the y-axis,
    prove that the locus of the intersection point of line BC and line l (which passes through the origin
    and is perpendicular to AC) satisfies the equation (x - b/2)²/(b²/4) + y²/(ab/4) = 1 -/
theorem intersection_locus (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ∃ (x y : ℝ → ℝ), ∀ (c : ℝ),
    let l := {p : ℝ × ℝ | p.2 = (a / c) * p.1}
    let bc := {p : ℝ × ℝ | p.1 / b + p.2 / c = 1}
    let intersection := Set.inter l bc
    (x c, y c) ∈ intersection ∧
    (x c - b/2)^2 / (b^2/4) + (y c)^2 / (a*b/4) = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_locus_l3981_398135


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l3981_398182

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l3981_398182


namespace NUMINAMATH_CALUDE_chemical_mixture_percentage_l3981_398163

theorem chemical_mixture_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 80 →
  initial_percentage = 0.3 →
  added_volume = 20 →
  let final_volume := initial_volume + added_volume
  let initial_x_volume := initial_volume * initial_percentage
  let final_x_volume := initial_x_volume + added_volume
  final_x_volume / final_volume = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_percentage_l3981_398163


namespace NUMINAMATH_CALUDE_M_inter_N_empty_l3981_398155

/-- Set M of complex numbers -/
def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = (t / (1 + t) : ℂ) + Complex.I * ((1 + t) / t : ℂ)}

/-- Set N of complex numbers -/
def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = (Real.sqrt 2 : ℂ) * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

/-- Theorem stating that the intersection of M and N is empty -/
theorem M_inter_N_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_empty_l3981_398155


namespace NUMINAMATH_CALUDE_power_of_five_preceded_by_coprimes_l3981_398184

theorem power_of_five_preceded_by_coprimes (x : ℕ) : 
  (5^x - 1 - (5^x / 5 - 1) = 7812500) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_preceded_by_coprimes_l3981_398184


namespace NUMINAMATH_CALUDE_two_digit_number_ratio_l3981_398176

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  tens_valid : tens ≥ 1 ∧ tens ≤ 9
  units_valid : units ≤ 9

def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.tens + n.units

def TwoDigitNumber.interchanged (n : TwoDigitNumber) : ℕ :=
  10 * n.units + n.tens

theorem two_digit_number_ratio (n : TwoDigitNumber) 
  (h1 : n.value - n.interchanged = 36)
  (h2 : (n.tens + n.units) - (n.tens - n.units) = 8) :
  n.tens = 2 * n.units :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_ratio_l3981_398176


namespace NUMINAMATH_CALUDE_volume_is_360_l3981_398110

/-- A rectangular parallelepiped with edge lengths 4, 6, and 15 -/
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 4
  width_eq : width = 6
  height_eq : height = 15

/-- The volume of a rectangular parallelepiped -/
def volume (rp : RectangularParallelepiped) : ℝ :=
  rp.length * rp.width * rp.height

/-- Theorem: The volume of the given rectangular parallelepiped is 360 cubic units -/
theorem volume_is_360 (rp : RectangularParallelepiped) : volume rp = 360 := by
  sorry

end NUMINAMATH_CALUDE_volume_is_360_l3981_398110


namespace NUMINAMATH_CALUDE_log_equation_implies_m_value_l3981_398141

theorem log_equation_implies_m_value 
  (m n : ℝ) (c : ℝ) 
  (h : Real.log (m^2) = c - 2 * Real.log n) :
  m = Real.sqrt (Real.exp c / n) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_implies_m_value_l3981_398141


namespace NUMINAMATH_CALUDE_f_properties_l3981_398146

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x ∈ Set.Icc (-3) (-2), f x ≤ -4) ∧
  (∀ x ∈ Set.Icc (-3) (-2), f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-3) (-2), f x = -4) ∧
  (∃ x ∈ Set.Icc (-3) (-2), f x = -7) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l3981_398146


namespace NUMINAMATH_CALUDE_jimin_has_most_candy_left_l3981_398113

def jimin_fraction : ℚ := 1/9
def taehyung_fraction : ℚ := 1/3
def hoseok_fraction : ℚ := 1/6

theorem jimin_has_most_candy_left : 
  jimin_fraction < taehyung_fraction ∧ 
  jimin_fraction < hoseok_fraction :=
by sorry

end NUMINAMATH_CALUDE_jimin_has_most_candy_left_l3981_398113


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3981_398129

theorem quadratic_roots_product (c d : ℝ) : 
  (3 * c ^ 2 + 9 * c - 21 = 0) → 
  (3 * d ^ 2 + 9 * d - 21 = 0) → 
  (3 * c - 4) * (6 * d - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3981_398129


namespace NUMINAMATH_CALUDE_donation_change_l3981_398178

def original_donations : List ℕ := [5, 3, 6, 5, 10]

def median (l : List ℕ) : ℕ := sorry
def mode (l : List ℕ) : List ℕ := sorry

def new_donations (a : ℕ) : List ℕ :=
  let index := original_donations.indexOf 3
  original_donations.set index (3 + a)

theorem donation_change (a : ℕ) :
  (median (new_donations a) = median original_donations ∧
   mode (new_donations a) = mode original_donations) ↔
  (a = 1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_donation_change_l3981_398178


namespace NUMINAMATH_CALUDE_rectangle_region_perimeter_l3981_398177

/-- Given a region formed by four congruent rectangles with a total area of 360 square centimeters
    and each rectangle having a width to length ratio of 3:4, the perimeter of the region is 14√7.5 cm. -/
theorem rectangle_region_perimeter (total_area : ℝ) (width : ℝ) (length : ℝ) : 
  total_area = 360 →
  width / length = 3 / 4 →
  width * length = total_area / 4 →
  2 * (2 * width + 2 * length) = 14 * Real.sqrt 7.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_region_perimeter_l3981_398177


namespace NUMINAMATH_CALUDE_eleven_times_digit_sum_l3981_398161

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem eleven_times_digit_sum :
  ∀ n : ℕ, n = 11 * sum_of_digits n ↔ n = 0 ∨ n = 198 := by sorry

end NUMINAMATH_CALUDE_eleven_times_digit_sum_l3981_398161


namespace NUMINAMATH_CALUDE_seller_total_loss_l3981_398158

/-- Represents the total loss of a seller in a transaction with a counterfeit banknote -/
def seller_loss (item_cost change_given fake_note_value real_note_value : ℕ) : ℕ :=
  item_cost + change_given + real_note_value

/-- Theorem stating the total loss of the seller in the given scenario -/
theorem seller_total_loss :
  let item_cost : ℕ := 20
  let customer_payment : ℕ := 100
  let change_given : ℕ := customer_payment - item_cost
  let fake_note_value : ℕ := 100
  let real_note_value : ℕ := 100
  seller_loss item_cost change_given fake_note_value real_note_value = 200 := by
  sorry

#eval seller_loss 20 80 100 100

end NUMINAMATH_CALUDE_seller_total_loss_l3981_398158


namespace NUMINAMATH_CALUDE_jimmy_payment_jimmy_paid_fifty_l3981_398134

/-- The amount Jimmy paid with, given his purchases and change received. -/
theorem jimmy_payment (pen_price notebook_price folder_price : ℕ)
  (pen_count notebook_count folder_count : ℕ)
  (change : ℕ) : ℕ :=
  let total_cost := pen_price * pen_count + notebook_price * notebook_count + folder_price * folder_count
  total_cost + change

/-- Proof that Jimmy paid $50 given his purchases and change received. -/
theorem jimmy_paid_fifty :
  jimmy_payment 1 3 5 3 4 2 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_payment_jimmy_paid_fifty_l3981_398134


namespace NUMINAMATH_CALUDE_function_property_l3981_398109

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def symmetric_about_origin (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f (-x - 1)

-- State the theorem
theorem function_property (h1 : is_even f) (h2 : symmetric_about_origin f) (h3 : f 0 = 1) :
  f (-1) + f 2 = -1 := by sorry

end NUMINAMATH_CALUDE_function_property_l3981_398109


namespace NUMINAMATH_CALUDE_binary_1011001_to_base6_l3981_398174

/-- Converts a binary (base-2) number to its decimal (base-10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base-10) number to its base-6 representation -/
def decimal_to_base6 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The binary representation of 1011001 -/
def binary_1011001 : List Bool := [true, false, false, true, true, false, true]

theorem binary_1011001_to_base6 :
  decimal_to_base6 (binary_to_decimal binary_1011001) = [2, 2, 5] :=
sorry

end NUMINAMATH_CALUDE_binary_1011001_to_base6_l3981_398174


namespace NUMINAMATH_CALUDE_total_cost_in_dollars_l3981_398199

/-- The cost of a single pencil in cents -/
def pencil_cost : ℚ := 2

/-- The cost of a single eraser in cents -/
def eraser_cost : ℚ := 5

/-- The number of pencils to be purchased -/
def num_pencils : ℕ := 500

/-- The number of erasers to be purchased -/
def num_erasers : ℕ := 250

/-- The conversion rate from cents to dollars -/
def cents_to_dollars : ℚ := 1 / 100

theorem total_cost_in_dollars : 
  (pencil_cost * num_pencils + eraser_cost * num_erasers) * cents_to_dollars = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_in_dollars_l3981_398199


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3981_398117

theorem complex_magnitude_one (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w^2 + 1/w^2 = s) : 
  Complex.abs w = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3981_398117


namespace NUMINAMATH_CALUDE_scientific_notation_of_18860000_l3981_398168

theorem scientific_notation_of_18860000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 18860000 = a * (10 : ℝ) ^ n ∧ a = 1.886 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_18860000_l3981_398168


namespace NUMINAMATH_CALUDE_math_city_intersections_l3981_398142

/-- The number of intersections for n non-parallel streets --/
def intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of streets in Math City --/
def num_streets : ℕ := 10

/-- The number of streets with tunnels --/
def streets_with_tunnels : ℕ := 2

/-- The number of intersections bypassed by each tunnel --/
def bypassed_per_tunnel : ℕ := 1

theorem math_city_intersections :
  intersections num_streets - streets_with_tunnels * bypassed_per_tunnel = 43 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l3981_398142


namespace NUMINAMATH_CALUDE_equation_root_implies_m_values_l3981_398188

theorem equation_root_implies_m_values (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*m*x + m^2 - 1 = 0) ∧ 
  (3^2 + 2*m*3 + m^2 - 1 = 0) →
  m = -2 ∨ m = -4 := by
sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_values_l3981_398188


namespace NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l3981_398128

/-- A quadratic function with axis of symmetry at x = 1 satisfies c > 2b -/
theorem quadratic_symmetry_inequality (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (2 - x)^2 + b * (2 - x) + c) →
  c > 2 * b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l3981_398128


namespace NUMINAMATH_CALUDE_vertical_multiplication_puzzle_l3981_398166

theorem vertical_multiplication_puzzle :
  ∀ a b : ℕ,
    10 < a ∧ a < 20 →
    10 < b ∧ b < 20 →
    100 ≤ a * b ∧ a * b < 1000 →
    (a * b) / 100 = 2 →
    a * b % 10 = 7 →
    (a = 13 ∧ b = 19) ∨ (a = 19 ∧ b = 13) :=
by sorry

end NUMINAMATH_CALUDE_vertical_multiplication_puzzle_l3981_398166


namespace NUMINAMATH_CALUDE_danicas_car_arrangement_l3981_398187

/-- The number of cars Danica currently has -/
def current_cars : ℕ := 29

/-- The number of cars required in each row -/
def cars_per_row : ℕ := 8

/-- The function to calculate the number of additional cars needed -/
def additional_cars_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - (current % per_row)) % per_row

theorem danicas_car_arrangement :
  additional_cars_needed current_cars cars_per_row = 3 := by
  sorry

end NUMINAMATH_CALUDE_danicas_car_arrangement_l3981_398187


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l3981_398165

theorem polynomial_functional_equation (p : ℝ → ℝ) 
  (h1 : p 3 = 10)
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) :
  ∀ x : ℝ, p x = x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l3981_398165


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l3981_398140

/-- A linear function passing through the first, third, and fourth quadrants -/
def passes_through_134_quadrants (k : ℝ) : Prop :=
  (k - 3 > 0) ∧ (-k + 2 < 0)

/-- Theorem stating that if a linear function y=(k-3)x-k+2 passes through
    the first, third, and fourth quadrants, then k > 3 -/
theorem linear_function_quadrants (k : ℝ) :
  passes_through_134_quadrants k → k > 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l3981_398140


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_evaluation_l3981_398153

theorem complex_arithmetic_expression_evaluation : 
  let a := (8/7 - 23/49) / (22/147)
  let b := (0.6 / (15/4)) * (5/2)
  let c := 3.75 / (3/2)
  ((a - b + c) / 2.2) = 3 := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_evaluation_l3981_398153


namespace NUMINAMATH_CALUDE_ratio_calculation_l3981_398102

theorem ratio_calculation (P M Q R N : ℚ) :
  R = (40 / 100) * M →
  M = (25 / 100) * Q →
  Q = (30 / 100) * P →
  N = (60 / 100) * P →
  R / N = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3981_398102


namespace NUMINAMATH_CALUDE_parabola_equation_l3981_398179

-- Define the parabola C
def Parabola : Type := ℝ → ℝ → Prop

-- Define the line x - y = 0
def Line (x y : ℝ) : Prop := x - y = 0

-- Define a point on the 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint of two points
def Midpoint (A B P : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

-- State the theorem
theorem parabola_equation (C : Parabola) (A B P : Point) :
  -- The vertex of parabola C is at the origin
  C 0 0 →
  -- The focus of parabola C is on the x-axis (we don't need to specify the exact location)
  ∃ f : ℝ, C f 0 →
  -- The line x - y = 0 intersects parabola C at points A and B
  Line A.x A.y ∧ C A.x A.y ∧ Line B.x B.y ∧ C B.x B.y →
  -- P(1,1) is the midpoint of segment AB
  P.x = 1 ∧ P.y = 1 ∧ Midpoint A B P →
  -- The equation of parabola C is x^2 = 2y
  ∀ x y : ℝ, C x y ↔ x^2 = 2*y :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3981_398179


namespace NUMINAMATH_CALUDE_equation_solution_l3981_398100

theorem equation_solution (x y : ℝ) : 
  (4 * x + y = 9) → (y = 9 - 4 * x) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3981_398100


namespace NUMINAMATH_CALUDE_min_tiles_for_square_l3981_398125

theorem min_tiles_for_square (tile_width : ℕ) (tile_height : ℕ) : 
  tile_width = 12 →
  tile_height = 15 →
  ∃ (square_side : ℕ) (num_tiles : ℕ),
    square_side % tile_width = 0 ∧
    square_side % tile_height = 0 ∧
    num_tiles = (square_side * square_side) / (tile_width * tile_height) ∧
    num_tiles = 20 ∧
    ∀ (smaller_side : ℕ) (smaller_num_tiles : ℕ),
      smaller_side < square_side →
      smaller_side % tile_width = 0 →
      smaller_side % tile_height = 0 →
      smaller_num_tiles = (smaller_side * smaller_side) / (tile_width * tile_height) →
      smaller_num_tiles < num_tiles :=
by sorry

end NUMINAMATH_CALUDE_min_tiles_for_square_l3981_398125


namespace NUMINAMATH_CALUDE_percentage_difference_l3981_398196

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 0.8181818181818181)) :
  x = y * 0.55 :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l3981_398196


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l3981_398111

/-- The length of the path traveled by point A in a rectangle ABCD after three 90° rotations -/
theorem rectangle_rotation_path_length (AB BC : ℝ) (h1 : AB = 3) (h2 : BC = 5) : 
  let diagonal := Real.sqrt (AB^2 + BC^2)
  let single_rotation_arc := π * diagonal / 2
  3 * single_rotation_arc = (3 * π * Real.sqrt 34) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l3981_398111


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3981_398156

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (1 - 2*i) / (2 + i) = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3981_398156


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3981_398195

/-- Given a geometric sequence {a_n} with common ratio q < 0,
    if a_2 = 1 - a_1 and a_4 = 4 - a_3, then a_4 + a_5 = 16 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h1 : q < 0)
  (h2 : ∀ n, a (n + 1) = q * a n)  -- Definition of geometric sequence
  (h3 : a 2 = 1 - a 1)
  (h4 : a 4 = 4 - a 3) :
  a 4 + a 5 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3981_398195


namespace NUMINAMATH_CALUDE_cube_skew_pairs_l3981_398106

/-- A cube with 8 vertices and 28 lines passing through any two vertices -/
structure Cube :=
  (vertices : Nat)
  (lines : Nat)
  (h_vertices : vertices = 8)
  (h_lines : lines = 28)

/-- The number of sets of 4 points not in the same plane in the cube -/
def sets_of_four_points (c : Cube) : Nat := 58

/-- The number of pairs of skew lines contributed by each set of 4 points -/
def skew_pairs_per_set : Nat := 3

/-- The total number of pairs of skew lines in the cube -/
def total_skew_pairs (c : Cube) : Nat :=
  (sets_of_four_points c) * skew_pairs_per_set

/-- Theorem: The number of pairs of skew lines in the cube is 174 -/
theorem cube_skew_pairs (c : Cube) : total_skew_pairs c = 174 := by
  sorry

end NUMINAMATH_CALUDE_cube_skew_pairs_l3981_398106


namespace NUMINAMATH_CALUDE_inverse_function_equality_f_equals_f_inverse_l3981_398191

def f (x : ℝ) : ℝ := 4 * x - 5

theorem inverse_function_equality (f : ℝ → ℝ) (h : Function.Bijective f) :
  ∃ x : ℝ, f x = Function.invFun f x :=
by
  sorry

theorem f_equals_f_inverse :
  ∃ x : ℝ, f x = Function.invFun f x ∧ x = 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_function_equality_f_equals_f_inverse_l3981_398191


namespace NUMINAMATH_CALUDE_g_75_solutions_l3981_398127

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_75_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 75 x = 0) ∧
                    (∀ x ∉ S, g 75 x ≠ 0) ∧
                    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_75_solutions_l3981_398127


namespace NUMINAMATH_CALUDE_piglet_straws_l3981_398172

theorem piglet_straws (total_straws : ℕ) (adult_pig_fraction : ℚ) (piglet_fraction : ℚ) (num_piglets : ℕ) :
  total_straws = 300 →
  adult_pig_fraction = 7 / 15 →
  piglet_fraction = 2 / 5 →
  num_piglets = 20 →
  (piglet_fraction * total_straws) / num_piglets = 6 := by
  sorry

end NUMINAMATH_CALUDE_piglet_straws_l3981_398172


namespace NUMINAMATH_CALUDE_negation_square_positive_l3981_398183

theorem negation_square_positive :
  (¬ ∀ n : ℕ, n^2 > 0) ↔ (∃ n : ℕ, n^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_square_positive_l3981_398183


namespace NUMINAMATH_CALUDE_map_distance_example_l3981_398137

/-- Given a map scale and an actual distance, calculates the distance on the map -/
def map_distance (scale : ℚ) (actual_distance : ℚ) : ℚ :=
  actual_distance * scale

/-- Theorem: For a map with scale 1:5000000 and actual distance 400km, the map distance is 8cm -/
theorem map_distance_example : 
  let scale : ℚ := 1 / 5000000
  let actual_distance : ℚ := 400 * 100000  -- 400km in cm
  map_distance scale actual_distance = 8 := by
  sorry

#eval map_distance (1 / 5000000) (400 * 100000)

end NUMINAMATH_CALUDE_map_distance_example_l3981_398137


namespace NUMINAMATH_CALUDE_similar_triangles_count_l3981_398107

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base apex foot : Point)

/-- Checks if a line is an altitude of a triangle -/
def isAltitude (alt : Altitude) (t : Triangle) : Prop := sorry

/-- Represents the intersection of two lines -/
def lineIntersection (p1 p2 q1 q2 : Point) : Point := sorry

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Main theorem -/
theorem similar_triangles_count 
  (ABC : Triangle) 
  (h_acute : isAcute ABC)
  (AL : Altitude)
  (h_AL : isAltitude AL ABC)
  (BM : Altitude)
  (h_BM : isAltitude BM ABC)
  (D : Point)
  (h_D : D = lineIntersection AL.foot BM.foot ABC.A ABC.B) :
  ∃ (pairs : List (Triangle × Triangle)), 
    (∀ (p : Triangle × Triangle), p ∈ pairs → areSimilar p.1 p.2) ∧ 
    pairs.length = 10 ∧
    (∀ (t1 t2 : Triangle), areSimilar t1 t2 → (t1, t2) ∈ pairs ∨ (t2, t1) ∈ pairs) :=
sorry

end NUMINAMATH_CALUDE_similar_triangles_count_l3981_398107


namespace NUMINAMATH_CALUDE_monomial_degree_equality_l3981_398151

-- Define the degree of a monomial
def degree (x y z : ℕ) (m : ℕ) : ℕ := x + y

-- Define the theorem
theorem monomial_degree_equality (m : ℕ) :
  degree 2 4 0 0 = degree 0 1 (m + 2) m →
  3 * m - 2 = 7 := by sorry

end NUMINAMATH_CALUDE_monomial_degree_equality_l3981_398151


namespace NUMINAMATH_CALUDE_star_product_scaling_l3981_398190

/-- Given that 2994 ã · 14.5 = 179, prove that 29.94 ã · 1.45 = 0.179 -/
theorem star_product_scaling (h : 2994 * 14.5 = 179) : 29.94 * 1.45 = 0.179 := by
  sorry

end NUMINAMATH_CALUDE_star_product_scaling_l3981_398190


namespace NUMINAMATH_CALUDE_max_c_value_l3981_398154

theorem max_c_value (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) :
  c ≤ 47 ∧ ∃ d₀, 5 * 47 + (d₀ - 12)^2 = 235 := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_l3981_398154


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3981_398160

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  -- Line equation
  (∀ x y, a * x - b * y - 2 = 0 → True) →
  -- Curve equation
  (∀ x, f x = x^3 + x) →
  -- Point P
  f 1 = 2 →
  -- Perpendicular tangents at P
  (a / b) * (f' 1) = -1 →
  -- Conclusion
  a / b = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3981_398160


namespace NUMINAMATH_CALUDE_final_green_probability_l3981_398131

/-- Represents the total number of amoeba in the dish -/
def total_amoeba : ℕ := 10

/-- Represents the initial number of green amoeba -/
def initial_green : ℕ := 7

/-- Represents the initial number of blue amoeba -/
def initial_blue : ℕ := 3

/-- Theorem stating the probability of the final amoeba being green -/
theorem final_green_probability :
  (initial_green : ℚ) / total_amoeba = 7 / 10 :=
sorry

end NUMINAMATH_CALUDE_final_green_probability_l3981_398131


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3981_398169

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m - 2) + y^2 / (m - 6) = 1 ∧ (m - 2) * (m - 6) < 0

/-- The theorem stating the condition for the equation to represent a hyperbola -/
theorem hyperbola_condition (m : ℝ) : is_hyperbola m ↔ 2 < m ∧ m < 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3981_398169


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l3981_398173

theorem complex_number_magnitude (z : ℂ) : z = 2 / (1 - Complex.I) + Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l3981_398173


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l3981_398121

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) := {x : ℝ | f a x > 0}

-- Define the given solution set
def given_set := {x : ℝ | 1/2 < x ∧ x < 2}

theorem quadratic_inequality_problem (a : ℝ) 
  (h : solution_set a = given_set) :
  (a = -2) ∧ 
  ({x : ℝ | a * x^2 - 5 * x + a^2 - 1 > 0} = {x : ℝ | -3 < x ∧ x < 1/2}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l3981_398121


namespace NUMINAMATH_CALUDE_shell_collection_ratio_l3981_398148

theorem shell_collection_ratio :
  ∀ (ben_shells laurie_shells alan_shells : ℕ),
    alan_shells = 4 * ben_shells →
    laurie_shells = 36 →
    alan_shells = 48 →
    ben_shells.gcd laurie_shells = ben_shells →
    (ben_shells : ℚ) / laurie_shells = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_ratio_l3981_398148


namespace NUMINAMATH_CALUDE_smallest_n_mod_20_sum_l3981_398162

theorem smallest_n_mod_20_sum (n : ℕ) : n ≥ 9 ↔ 
  ∀ (S : Finset ℤ), S.card = n → 
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      (a + b) % 20 = (c + d) % 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_mod_20_sum_l3981_398162


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3981_398103

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - Complex.I) * (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3981_398103


namespace NUMINAMATH_CALUDE_boat_downstream_speed_l3981_398124

/-- Given a boat's speed in still water and its upstream speed, calculate its downstream speed. -/
theorem boat_downstream_speed
  (still_water_speed : ℝ)
  (upstream_speed : ℝ)
  (h1 : still_water_speed = 7)
  (h2 : upstream_speed = 4) :
  still_water_speed + (still_water_speed - upstream_speed) = 10 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_speed_l3981_398124


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l3981_398132

def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check :
  (canFormTriangle 2 3 4) ∧
  ¬(canFormTriangle 3 4 7) ∧
  ¬(canFormTriangle 4 6 2) ∧
  ¬(canFormTriangle 7 10 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l3981_398132


namespace NUMINAMATH_CALUDE_line_intersects_and_passes_through_point_l3981_398180

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y + 2 * m - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 8 = 0

-- Theorem statement
theorem line_intersects_and_passes_through_point :
  ∀ m : ℝ,
  (∃ x y : ℝ, line_l m x y ∧ circle_C x y) ∧
  (line_l m (-2) 2) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_and_passes_through_point_l3981_398180


namespace NUMINAMATH_CALUDE_area_ratio_of_triangles_l3981_398157

/-- Given two triangles PQR and XYZ with known base and height measurements,
    prove that the area of PQR is 1/3 of the area of XYZ. -/
theorem area_ratio_of_triangles (base_PQR height_PQR base_XYZ height_XYZ : ℝ)
  (h1 : base_PQR = 3)
  (h2 : height_PQR = 2)
  (h3 : base_XYZ = 6)
  (h4 : height_XYZ = 3) :
  (1 / 2 * base_PQR * height_PQR) / (1 / 2 * base_XYZ * height_XYZ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_triangles_l3981_398157


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3981_398133

theorem absolute_value_equation (x : ℝ) : |4*x - 3| + 2 = 2 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3981_398133


namespace NUMINAMATH_CALUDE_unique_solution_for_class_representatives_l3981_398189

theorem unique_solution_for_class_representatives (m n : ℕ) : 
  10 ≥ m ∧ m > n ∧ n ≥ 4 →
  ((m - n)^2 = m + n) ↔ (m = 10 ∧ n = 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_class_representatives_l3981_398189


namespace NUMINAMATH_CALUDE_simplify_expression_l3981_398150

theorem simplify_expression (a b : ℝ) : (-a^2 * b^3)^3 = -a^6 * b^9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3981_398150


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_calculation_l3981_398126

theorem absolute_value_and_exponents_calculation : 
  |(-5 : ℝ)| + (1/3)⁻¹ - (π - 2)^0 = 7 := by sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_calculation_l3981_398126


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3981_398147

/-- The function f(x) = x^3 + x + 2 -/
def f (x : ℝ) : ℝ := x^3 + x + 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ f' x = 4) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3981_398147


namespace NUMINAMATH_CALUDE_system_solution_l3981_398143

theorem system_solution (a₁ a₂ b₁ b₂ : ℝ) :
  (∃ x y : ℝ, a₁ * x + b₁ * y = 21 ∧ a₂ * x + b₂ * y = 12 ∧ x = 3 ∧ y = 6) →
  (∃ m n : ℝ, a₁ * (2 * m + n) + b₁ * (m - n) = 21 ∧ 
              a₂ * (2 * m + n) + b₂ * (m - n) = 12 ∧
              m = 3 ∧ n = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3981_398143


namespace NUMINAMATH_CALUDE_cubic_expansion_equals_cube_problem_solution_l3981_398115

theorem cubic_expansion_equals_cube (n : ℕ) : n^3 + 3*(n^2) + 3*n + 1 = (n + 1)^3 := by sorry

theorem problem_solution : 98^3 + 3*(98^2) + 3*98 + 1 = 970299 := by sorry

end NUMINAMATH_CALUDE_cubic_expansion_equals_cube_problem_solution_l3981_398115


namespace NUMINAMATH_CALUDE_arithmetic_proof_l3981_398120

theorem arithmetic_proof : 4 * (9 - 6)^2 / 2 - 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l3981_398120


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3981_398108

theorem quadratic_root_relation (p : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + 3 = 0 ∧ y^2 + p*y + 3 = 0 ∧ y = 3*x) → 
  (p = 4 ∨ p = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3981_398108


namespace NUMINAMATH_CALUDE_apple_sales_theorem_l3981_398192

/-- Calculate the total money earned from selling apples from a rectangular plot of trees -/
def apple_sales_revenue (rows : ℕ) (cols : ℕ) (apples_per_tree : ℕ) (price_per_apple : ℚ) : ℚ :=
  (rows * cols * apples_per_tree : ℕ) * price_per_apple

/-- Theorem: The total money earned from selling apples from a 3x4 plot of trees,
    where each tree produces 5 apples and each apple is sold for $0.5, is equal to $30 -/
theorem apple_sales_theorem :
  apple_sales_revenue 3 4 5 (1/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_apple_sales_theorem_l3981_398192


namespace NUMINAMATH_CALUDE_floor_sqrt_150_l3981_398197

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_150_l3981_398197


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l3981_398104

theorem smallest_digit_divisible_by_11 : 
  ∃ (d : Nat), d < 10 ∧ 
    (∀ (x : Nat), x < d → ¬(489000 + x * 100 + 7).ModEq 0 11) ∧
    (489000 + d * 100 + 7).ModEq 0 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l3981_398104


namespace NUMINAMATH_CALUDE_regions_in_circle_l3981_398198

/-- The number of regions created by radii and concentric circles in a larger circle -/
def num_regions (num_radii : ℕ) (num_circles : ℕ) : ℕ :=
  (num_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (num_radii num_circles : ℕ) 
  (h1 : num_radii = 16) 
  (h2 : num_circles = 10) : 
  num_regions num_radii num_circles = 176 := by
  sorry

#eval num_regions 16 10

end NUMINAMATH_CALUDE_regions_in_circle_l3981_398198


namespace NUMINAMATH_CALUDE_special_cone_volume_l3981_398170

/-- A cone with base area π and lateral surface in the shape of a semicircle -/
structure SpecialCone where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The height of the cone -/
  h : ℝ
  /-- The slant height of the cone -/
  l : ℝ
  /-- The base area is π -/
  base_area : π * r^2 = π
  /-- The lateral surface is a semicircle -/
  lateral_surface : π * l = 2 * π * r

/-- The volume of the special cone is (√3/3)π -/
theorem special_cone_volume (c : SpecialCone) : 
  (1/3) * π * c.r^2 * c.h = (Real.sqrt 3 / 3) * π := by
  sorry


end NUMINAMATH_CALUDE_special_cone_volume_l3981_398170


namespace NUMINAMATH_CALUDE_max_x_on_3x3_grid_l3981_398193

/-- Represents a 3x3 grid where X's can be placed. -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Checks if three X's are aligned in any direction on the grid. -/
def hasThreeAligned (g : Grid) : Bool :=
  sorry

/-- Counts the number of X's placed on the grid. -/
def countX (g : Grid) : Nat :=
  sorry

/-- Theorem stating the maximum number of X's that can be placed on a 3x3 grid
    without three X's aligning vertically, horizontally, or diagonally is 4. -/
theorem max_x_on_3x3_grid :
  (∃ g : Grid, ¬hasThreeAligned g ∧ countX g = 4) ∧
  (∀ g : Grid, ¬hasThreeAligned g → countX g ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_x_on_3x3_grid_l3981_398193


namespace NUMINAMATH_CALUDE_least_value_f_1998_l3981_398145

/-- A function from positive integers to positive integers satisfying the given property -/
def FunctionF :=
  {f : ℕ+ → ℕ+ | ∀ m n : ℕ+, f (n^2 * f m) = m * (f n)^2}

/-- The theorem stating the least possible value of f(1998) -/
theorem least_value_f_1998 :
  (∃ f ∈ FunctionF, f 1998 = 120) ∧
  (∀ f ∈ FunctionF, f 1998 ≥ 120) :=
sorry

end NUMINAMATH_CALUDE_least_value_f_1998_l3981_398145


namespace NUMINAMATH_CALUDE_cross_product_result_l3981_398114

def u : ℝ × ℝ × ℝ := (-3, 4, 2)
def v : ℝ × ℝ × ℝ := (8, -5, 6)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

theorem cross_product_result : cross_product u v = (34, -34, -17) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l3981_398114


namespace NUMINAMATH_CALUDE_odd_digits_base4_345_l3981_398101

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 345₁₀ is 3 -/
theorem odd_digits_base4_345 : countOddDigits (toBase4 345) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_345_l3981_398101


namespace NUMINAMATH_CALUDE_amelia_remaining_money_l3981_398181

-- Define the given amounts and percentages
def initial_amount : ℝ := 60
def first_course_cost : ℝ := 15
def second_course_additional_cost : ℝ := 5
def dessert_percentage : ℝ := 0.25
def drink_percentage : ℝ := 0.20

-- Define the theorem
theorem amelia_remaining_money :
  let second_course_cost := first_course_cost + second_course_additional_cost
  let dessert_cost := dessert_percentage * second_course_cost
  let first_three_courses_cost := first_course_cost + second_course_cost + dessert_cost
  let drink_cost := drink_percentage * first_three_courses_cost
  let total_cost := first_three_courses_cost + drink_cost
  initial_amount - total_cost = 12 := by sorry

end NUMINAMATH_CALUDE_amelia_remaining_money_l3981_398181


namespace NUMINAMATH_CALUDE_min_value_theorem_l3981_398185

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * Real.sqrt x + 2 / x^2 ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x^2 = 5 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3981_398185


namespace NUMINAMATH_CALUDE_sugar_price_increase_l3981_398130

theorem sugar_price_increase (original_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) : 
  original_price = 6 →
  consumption_reduction = 19.999999999999996 →
  (1 - consumption_reduction / 100) * new_price = original_price →
  new_price = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l3981_398130


namespace NUMINAMATH_CALUDE_cos_two_alpha_zero_l3981_398149

theorem cos_two_alpha_zero (α : Real) (h : Real.sin (π/6 - α) = Real.cos (π/6 + α)) : 
  Real.cos (2 * α) = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_two_alpha_zero_l3981_398149


namespace NUMINAMATH_CALUDE_base8_157_equals_base10_111_l3981_398167

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Theorem: The base-8 number 157 is equal to the base-10 number 111 --/
theorem base8_157_equals_base10_111 : base8_to_base10 157 = 111 := by
  sorry

end NUMINAMATH_CALUDE_base8_157_equals_base10_111_l3981_398167


namespace NUMINAMATH_CALUDE_chiefs_gold_l3981_398159

/-- A graph representing druids and their willingness to shake hands. -/
structure DruidGraph where
  /-- The set of vertices (druids) in the graph. -/
  V : Type
  /-- The edge relation, representing willingness to shake hands. -/
  E : V → V → Prop
  /-- The graph has no cycles of length 4 or more. -/
  no_long_cycles : ∀ (a b c d : V), E a b → E b c → E c d → E d a → (a = c ∨ b = d)

/-- The number of vertices in a DruidGraph. -/
def num_vertices (G : DruidGraph) : ℕ := sorry

/-- The number of edges in a DruidGraph. -/
def num_edges (G : DruidGraph) : ℕ := sorry

/-- 
The chief's gold theorem: In a DruidGraph, the chief can keep at least 3 gold coins.
This is equivalent to showing that 3n - 2e ≥ 3, where n is the number of vertices and e is the number of edges.
-/
theorem chiefs_gold (G : DruidGraph) : 
  3 * (num_vertices G) - 2 * (num_edges G) ≥ 3 := by sorry

end NUMINAMATH_CALUDE_chiefs_gold_l3981_398159


namespace NUMINAMATH_CALUDE_license_plate_count_l3981_398152

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 20

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of possible license plates meeting the specified criteria -/
def num_license_plates : ℕ := num_consonants * 1 * num_consonants * num_even_digits

/-- Theorem stating that the number of license plates meeting the criteria is 2000 -/
theorem license_plate_count : num_license_plates = 2000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3981_398152


namespace NUMINAMATH_CALUDE_total_clam_shells_is_43_l3981_398138

/-- The number of clam shells found by Sam, Mary, and Lucy -/
def clam_shells (name : String) : ℕ :=
  match name with
  | "Sam" => 8
  | "Mary" => 20
  | "Lucy" => 15
  | _ => 0

/-- The total number of clam shells found by Sam, Mary, and Lucy -/
def total_clam_shells : ℕ :=
  clam_shells "Sam" + clam_shells "Mary" + clam_shells "Lucy"

/-- Theorem stating that the total number of clam shells found is 43 -/
theorem total_clam_shells_is_43 : total_clam_shells = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_clam_shells_is_43_l3981_398138


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3981_398139

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 2*x - 5 = 0) ↔ ((x + 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3981_398139


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3981_398118

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^3 > x^(1/3)) ↔ (∃ x : ℝ, x > 1 ∧ x^3 ≤ x^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3981_398118


namespace NUMINAMATH_CALUDE_rope_segment_relation_l3981_398122

theorem rope_segment_relation (x : ℝ) : x > 0 ∧ x ≤ 2 →
  (x^2 = 2*(2 - x) ↔ x^2 = (2 - x) * 2) := by
  sorry

end NUMINAMATH_CALUDE_rope_segment_relation_l3981_398122


namespace NUMINAMATH_CALUDE_remaining_water_l3981_398175

theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/4 → remaining = initial - used → remaining = 1/4 := by sorry

end NUMINAMATH_CALUDE_remaining_water_l3981_398175


namespace NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l3981_398171

theorem four_numbers_product_sum_prime :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.Prime (a * b + c * d) ∧
    Nat.Prime (a * c + b * d) ∧
    Nat.Prime (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_product_sum_prime_l3981_398171


namespace NUMINAMATH_CALUDE_sqrt_400_div_2_l3981_398144

theorem sqrt_400_div_2 : Real.sqrt 400 / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_400_div_2_l3981_398144


namespace NUMINAMATH_CALUDE_circle_center_distance_to_line_l3981_398136

theorem circle_center_distance_to_line : ∃ (center : ℝ × ℝ),
  (∀ (x y : ℝ), x^2 + 2*x + y^2 = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = 1) ∧
  |center.1 - 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_distance_to_line_l3981_398136


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3981_398119

theorem reciprocal_of_negative_two :
  (∃ x : ℚ, -2 * x = 1) ∧ (∀ x : ℚ, -2 * x = 1 → x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3981_398119


namespace NUMINAMATH_CALUDE_regression_lines_intersection_l3981_398194

/-- A regression line in 2D space -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def RegressionLine.point_on_line (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : (s, t) = l₁.point_on_line s)
  (h₂ : (s, t) = l₂.point_on_line s) :
  ∃ (x y : ℝ), l₁.point_on_line x = (x, y) ∧ l₂.point_on_line x = (x, y) ∧ x = s ∧ y = t :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersection_l3981_398194


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_15_l3981_398112

theorem sum_of_fifth_powers_divisible_by_15 
  (a b c d e : ℤ) 
  (h : a + b + c + d + e = 0) : 
  ∃ k : ℤ, a^5 + b^5 + c^5 + d^5 + e^5 = 15 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_15_l3981_398112


namespace NUMINAMATH_CALUDE_age_problem_solution_l3981_398164

/-- Represents the age relationship between a father and daughter -/
structure AgeProblem where
  daughter_age : ℕ
  father_age : ℕ
  years_ago : ℕ
  years_future : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : AgeProblem) : Prop :=
  p.father_age = 3 * p.daughter_age ∧
  (p.father_age - p.years_ago) = 5 * (p.daughter_age - p.years_ago)

/-- The future condition we want to prove -/
def future_condition (p : AgeProblem) : Prop :=
  (p.father_age + p.years_future) = 2 * (p.daughter_age + p.years_future)

/-- The theorem to prove -/
theorem age_problem_solution :
  ∀ p : AgeProblem,
    problem_conditions p →
    (p.years_future = 14 ↔ future_condition p) :=
by
  sorry


end NUMINAMATH_CALUDE_age_problem_solution_l3981_398164


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l3981_398116

/-- The number of students that need to be initially removed in a systematic sampling -/
def studentsToRemove (totalStudents sampleSize : ℕ) : ℕ :=
  totalStudents % sampleSize

theorem systematic_sampling_removal (totalStudents sampleSize : ℕ) 
  (h1 : totalStudents = 1387)
  (h2 : sampleSize = 9)
  (h3 : sampleSize > 0) :
  studentsToRemove totalStudents sampleSize = 1 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l3981_398116


namespace NUMINAMATH_CALUDE_boys_in_class_l3981_398186

theorem boys_in_class (total : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total = 160 → 
  girls_fraction = 1/4 → 
  boys = total - (girls_fraction * total).num → 
  boys = 120 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l3981_398186


namespace NUMINAMATH_CALUDE_simplify_expression_l3981_398105

theorem simplify_expression (x : ℝ) : (5 - 2*x^2) - (7 - 3*x^2) = -2 + x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3981_398105
