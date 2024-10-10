import Mathlib

namespace max_product_constraint_l306_30656

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ a₀ * b₀ = 1 / 24 := by
  sorry

end max_product_constraint_l306_30656


namespace inscribed_circle_probability_l306_30672

theorem inscribed_circle_probability (a b : ℝ) (h_right_triangle : a = 8 ∧ b = 15) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  1 - (π * r^2) / (a * b / 2) = 1 - 3 * π / 20 :=
by sorry

end inscribed_circle_probability_l306_30672


namespace simultaneous_equations_solution_l306_30609

theorem simultaneous_equations_solution :
  ∀ a b : ℚ,
  (a + b) * (a^2 - b^2) = 4 ∧
  (a - b) * (a^2 + b^2) = 5/2 →
  ((a = 3/2 ∧ b = 1/2) ∨ (a = -1/2 ∧ b = -3/2)) :=
by sorry

end simultaneous_equations_solution_l306_30609


namespace triangle_existence_condition_l306_30694

/-- A triangle with given inscribed circle radius, circumscribed circle radius, and one angle. -/
structure Triangle where
  r : ℝ  -- radius of inscribed circle
  R : ℝ  -- radius of circumscribed circle
  α : ℝ  -- one angle of the triangle (in radians)

/-- Theorem stating the conditions for the existence of a triangle with given parameters. -/
theorem triangle_existence_condition (t : Triangle) :
  (∃ (triangle : Triangle), triangle = t) ↔ 
  (0 < t.α ∧ t.α < Real.pi ∧ t.R ≥ 2 * t.r) :=
sorry

end triangle_existence_condition_l306_30694


namespace f_1991_equals_1988_l306_30685

/-- Represents the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Represents the cumulative sum of digits up to r-digit numbers -/
def g (r : ℕ) : ℕ := r * 10^r - (10^r - 1) / 9

/-- 
f(n) represents the number of digits in the number containing the 10^nth digit 
in the sequence of natural numbers written in order without spaces
-/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating that f(1991) = 1988 -/
theorem f_1991_equals_1988 : f 1991 = 1988 := by sorry

end f_1991_equals_1988_l306_30685


namespace triangle_inequality_l306_30680

theorem triangle_inequality (a b c : ℝ) (x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (h4 : 0 ≤ x ∧ x ≤ π) (h5 : 0 ≤ y ∧ y ≤ π) (h6 : 0 ≤ z ∧ z ≤ π)
  (h7 : x + y + z = π) : 
  b * c + c * a - a * b < b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ∧
  b * c * Real.cos x + c * a * Real.cos y + a * b * Real.cos z ≤ (1/2) * (a^2 + b^2 + c^2) :=
sorry

end triangle_inequality_l306_30680


namespace B_2_2_equals_12_l306_30644

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2_equals_12 : B 2 2 = 12 := by sorry

end B_2_2_equals_12_l306_30644


namespace exists_fib_div_1000_l306_30696

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: There exists a Fibonacci number divisible by 1000 -/
theorem exists_fib_div_1000 : ∃ n : ℕ, 1000 ∣ fib n := by
  sorry

end exists_fib_div_1000_l306_30696


namespace part_one_part_two_l306_30630

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_one :
  {x : ℝ | f 3 x ≥ 1} = Set.Icc 0 (4/3) := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x - |2*x - 5| ≤ 0) →
  a ∈ Set.Icc (-1) 4 := by sorry

end part_one_part_two_l306_30630


namespace basketball_score_proof_l306_30697

theorem basketball_score_proof (two_point_shots three_point_shots free_throws : ℕ) :
  (3 * three_point_shots = 2 * two_point_shots) →
  (free_throws = 2 * three_point_shots) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 80) →
  free_throws = 20 := by
  sorry

end basketball_score_proof_l306_30697


namespace polynomial_roots_sum_l306_30681

theorem polynomial_roots_sum (p q r : ℤ) (m : ℤ) : 
  (∀ x : ℤ, x^3 - 2024*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 104 := by
sorry

end polynomial_roots_sum_l306_30681


namespace tile_perimeter_theorem_l306_30620

/-- Represents the shape of the tile configuration -/
inductive TileShape
  | L

/-- Represents the possible perimeters after adding tiles -/
def PossiblePerimeters : Set ℕ := {12, 14, 16}

/-- The initial tile configuration -/
structure InitialConfig where
  shape : TileShape
  tileCount : ℕ
  tileSize : ℕ
  perimeter : ℕ

/-- The configuration after adding tiles -/
structure FinalConfig where
  initial : InitialConfig
  addedTiles : ℕ

/-- Predicate to check if a perimeter is possible after adding tiles -/
def IsValidPerimeter (config : FinalConfig) (p : ℕ) : Prop :=
  p ∈ PossiblePerimeters

/-- Main theorem statement -/
theorem tile_perimeter_theorem (config : FinalConfig)
  (h1 : config.initial.shape = TileShape.L)
  (h2 : config.initial.tileCount = 8)
  (h3 : config.initial.tileSize = 1)
  (h4 : config.initial.perimeter = 12)
  (h5 : config.addedTiles = 2) :
  ∃ (p : ℕ), IsValidPerimeter config p :=
sorry

end tile_perimeter_theorem_l306_30620


namespace roots_sum_squares_l306_30606

theorem roots_sum_squares (r s : ℝ) : 
  r^2 - 5*r + 6 = 0 → s^2 - 5*s + 6 = 0 → r^2 + s^2 = 13 := by
  sorry

end roots_sum_squares_l306_30606


namespace initial_volume_calculation_l306_30618

theorem initial_volume_calculation (initial_percentage : Real) 
  (final_percentage : Real) (pure_alcohol_added : Real) 
  (h1 : initial_percentage = 0.35)
  (h2 : final_percentage = 0.50)
  (h3 : pure_alcohol_added = 1.8) : 
  ∃ (initial_volume : Real), 
    initial_volume * initial_percentage + pure_alcohol_added = 
    (initial_volume + pure_alcohol_added) * final_percentage ∧ 
    initial_volume = 6 := by
  sorry

end initial_volume_calculation_l306_30618


namespace conference_seating_optimization_l306_30660

theorem conference_seating_optimization
  (initial_chairs : ℕ)
  (chairs_per_row : ℕ)
  (expected_participants : ℕ)
  (h1 : initial_chairs = 144)
  (h2 : chairs_per_row = 12)
  (h3 : expected_participants = 100)
  : ∃ (chairs_to_remove : ℕ),
    chairs_to_remove = 36 ∧
    (initial_chairs - chairs_to_remove) % chairs_per_row = 0 ∧
    initial_chairs - chairs_to_remove ≥ expected_participants ∧
    ∀ (x : ℕ), x < chairs_to_remove →
      (initial_chairs - x) % chairs_per_row ≠ 0 ∨
      initial_chairs - x > expected_participants + chairs_per_row - 1 :=
by
  sorry

end conference_seating_optimization_l306_30660


namespace cubic_equation_roots_l306_30657

theorem cubic_equation_roots (p : ℝ) : 
  (∃ x y z : ℤ, x > 0 ∧ y > 0 ∧ z > 0 ∧
   (∀ t : ℝ, 5*t^3 - 5*(p+1)*t^2 + (71*p - 1)*t + 1 = 66*p ↔ t = x ∨ t = y ∨ t = z))
  ↔ p = 76 := by
sorry

end cubic_equation_roots_l306_30657


namespace flowers_given_to_brother_correct_flowers_given_l306_30655

theorem flowers_given_to_brother (amanda_flowers : ℕ) (peter_flowers_left : ℕ) : ℕ :=
  let peter_initial_flowers := 3 * amanda_flowers
  peter_initial_flowers - peter_flowers_left

theorem correct_flowers_given (amanda_flowers : ℕ) (peter_flowers_left : ℕ)
    (h1 : amanda_flowers = 20)
    (h2 : peter_flowers_left = 45) :
    flowers_given_to_brother amanda_flowers peter_flowers_left = 15 := by
  sorry

end flowers_given_to_brother_correct_flowers_given_l306_30655


namespace numbers_sum_l306_30665

/-- Given the conditions about Mickey's, Jayden's, and Coraline's numbers, 
    prove that their sum is 180. -/
theorem numbers_sum (M J C : ℕ) : 
  M = J + 20 →  -- Mickey's number is greater than Jayden's by 20
  J = C - 40 →  -- Jayden's number is 40 less than Coraline's
  C = 80 →      -- Coraline's number is 80
  M + J + C = 180 := by
sorry

end numbers_sum_l306_30665


namespace perp_bisector_x_intercept_range_l306_30625

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the perpendicular bisector intersection with x-axis
def perp_bisector_x_intercept (A B : PointOnParabola) : ℝ :=
  sorry -- Definition of x₀ in terms of A and B

-- Theorem statement
theorem perp_bisector_x_intercept_range (A B : PointOnParabola) :
  A ≠ B → perp_bisector_x_intercept A B > 1 :=
sorry

end perp_bisector_x_intercept_range_l306_30625


namespace shaded_fraction_of_large_rectangle_l306_30622

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

theorem shaded_fraction_of_large_rectangle (large : Rectangle) (small : Rectangle) 
  (h1 : large.width = 15)
  (h2 : large.height = 20)
  (h3 : small.area = (1 / 5) * large.area)
  (h4 : small.area > 0) :
  (1 / 2) * small.area / large.area = 1 / 10 := by
  sorry

end shaded_fraction_of_large_rectangle_l306_30622


namespace overtake_time_l306_30624

/-- The time it takes for person B to overtake person A given their speeds and start times -/
theorem overtake_time (speed_A speed_B : ℝ) (start_delay : ℝ) : 
  speed_A = 5 →
  speed_B = 5.555555555555555 →
  start_delay = 0.5 →
  speed_B > speed_A →
  (start_delay * speed_A) / (speed_B - speed_A) = 4.5 := by
  sorry

#check overtake_time

end overtake_time_l306_30624


namespace infinite_series_equality_l306_30683

theorem infinite_series_equality (a b : ℝ) 
  (h : ∑' n, a / b^n = 6) : 
  ∑' n, a / (a + b)^n = 6/7 := by
sorry

end infinite_series_equality_l306_30683


namespace fraction_order_l306_30698

theorem fraction_order : 
  let f1 := 21 / 14
  let f2 := 25 / 18
  let f3 := 23 / 16
  let f4 := 27 / 19
  f2 < f4 ∧ f4 < f3 ∧ f3 < f1 := by
  sorry

end fraction_order_l306_30698


namespace intersection_M_N_l306_30629

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end intersection_M_N_l306_30629


namespace product_of_1011_2_and_102_3_l306_30648

def base_2_to_10 (n : ℕ) : ℕ := sorry

def base_3_to_10 (n : ℕ) : ℕ := sorry

theorem product_of_1011_2_and_102_3 : 
  (base_2_to_10 1011) * (base_3_to_10 102) = 121 := by sorry

end product_of_1011_2_and_102_3_l306_30648


namespace green_blue_difference_l306_30640

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : blue * 18 = total * 3 ∧ yellow * 18 = total * 7 ∧ green * 18 = total * 8

theorem green_blue_difference (bag : DiskBag) (h : bag.total = 72) : 
  bag.green - bag.blue = 20 := by
  sorry

end green_blue_difference_l306_30640


namespace arithmetic_mean_of_specific_numbers_l306_30651

theorem arithmetic_mean_of_specific_numbers :
  let numbers : List ℝ := [-5, 3.5, 12, 20]
  (numbers.sum / numbers.length : ℝ) = 7.625 := by
  sorry

end arithmetic_mean_of_specific_numbers_l306_30651


namespace function_sum_theorem_l306_30602

/-- Given a function f(x) = a^x + a^(-x) where a > 0 and a ≠ 1, 
    if f(1) = 3, then f(0) + f(1) + f(2) = 12 -/
theorem function_sum_theorem (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + a^(-x)
  f 1 = 3 → f 0 + f 1 + f 2 = 12 := by
sorry

end function_sum_theorem_l306_30602


namespace square_sum_from_product_and_sum_l306_30688

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 12) 
  (h2 : x + y = 10) : 
  x^2 + y^2 = 76 := by
sorry

end square_sum_from_product_and_sum_l306_30688


namespace rational_numbers_definition_l306_30638

-- Define the set of rational numbers
def RationalNumbers : Set ℚ := {q : ℚ | true}

-- Define the set of integers as a subset of rational numbers
def Integers : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions as a subset of rational numbers
def Fractions : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Theorem stating that rational numbers are the union of integers and fractions
theorem rational_numbers_definition : 
  RationalNumbers = Integers ∪ Fractions := by
  sorry

end rational_numbers_definition_l306_30638


namespace computer_printer_price_l306_30659

/-- The total price of a basic computer and printer, given specific conditions -/
theorem computer_printer_price (basic_price enhanced_price printer_price total_price : ℝ) : 
  basic_price = 2125 →
  enhanced_price = basic_price + 500 →
  printer_price = (1 / 8) * (enhanced_price + printer_price) →
  total_price = basic_price + printer_price →
  total_price = 2500 := by
  sorry

end computer_printer_price_l306_30659


namespace side_length_of_octagon_l306_30668

theorem side_length_of_octagon (perimeter : ℝ) (num_sides : ℕ) (h1 : perimeter = 23.6) (h2 : num_sides = 8) :
  perimeter / num_sides = 2.95 := by
  sorry

end side_length_of_octagon_l306_30668


namespace parabola_tangent_to_line_l306_30601

/-- A parabola y = ax² + bx + 7 is tangent to the line y = 2x + 3 if and only if b = 2 ± 4√a -/
theorem parabola_tangent_to_line (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x^2 + b * x + 7 ∧ y = 2 * x + 3 ∧
    ∀ x' : ℝ, a * x'^2 + b * x' + 7 ≥ 2 * x' + 3) ↔
  (b = 2 + 4 * Real.sqrt a ∨ b = 2 - 4 * Real.sqrt a) :=
by sorry

end parabola_tangent_to_line_l306_30601


namespace inverse_proportion_problem_l306_30642

/-- Given that p and r · q are inversely proportional, prove that p = 128/15 when q = 10 and r = 3,
    given that p = 16 when q = 8 and r = 2 -/
theorem inverse_proportion_problem (p q r : ℝ) (h1 : ∃ k, p * (r * q) = k) 
  (h2 : p = 16 ∧ q = 8 ∧ r = 2) : 
  (q = 10 ∧ r = 3) → p = 128 / 15 := by
  sorry

end inverse_proportion_problem_l306_30642


namespace lcm_of_72_108_2100_l306_30682

theorem lcm_of_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end lcm_of_72_108_2100_l306_30682


namespace amy_candy_distribution_l306_30637

/-- Proves that Amy puts 10 candies in each basket given the conditions of the problem -/
theorem amy_candy_distribution (chocolate_bars : ℕ) (num_baskets : ℕ) : 
  chocolate_bars = 5 →
  num_baskets = 25 →
  (chocolate_bars + 7 * chocolate_bars + 6 * (7 * chocolate_bars)) / num_baskets = 10 :=
by
  sorry

end amy_candy_distribution_l306_30637


namespace class_trip_theorem_l306_30647

/-- Represents the possible solutions for the class trip problem -/
inductive ClassTripSolution
  | five : ClassTripSolution
  | twentyFive : ClassTripSolution

/-- Checks if a given number of students and monthly contribution satisfy the problem conditions -/
def validSolution (numStudents : ℕ) (monthlyContribution : ℕ) : Prop :=
  numStudents * monthlyContribution * 9 = 22725

/-- The main theorem stating that only two solutions exist for the class trip problem -/
theorem class_trip_theorem : 
  ∀ (sol : ClassTripSolution), 
    (sol = ClassTripSolution.five ∧ validSolution 5 505) ∨
    (sol = ClassTripSolution.twentyFive ∧ validSolution 25 101) :=
by sorry

end class_trip_theorem_l306_30647


namespace quadrilateral_perimeter_l306_30613

/-- A quadrilateral with vertices at (1,2), (4,5), (5,4), and (4,1) has a perimeter of 4√2 + 2√10 -/
theorem quadrilateral_perimeter : 
  let vertices : List (ℝ × ℝ) := [(1, 2), (4, 5), (5, 4), (4, 1)]
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := (List.zip vertices (vertices.rotateLeft 1)).map (fun (p, q) => distance p q) |>.sum
  perimeter = 4 * Real.sqrt 2 + 2 * Real.sqrt 10 := by
  sorry

end quadrilateral_perimeter_l306_30613


namespace square_root_probability_l306_30608

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def count_valid_numbers : ℕ := 71

def total_two_digit_numbers : ℕ := 90

theorem square_root_probability : 
  (count_valid_numbers : ℚ) / total_two_digit_numbers = 71 / 90 := by sorry

end square_root_probability_l306_30608


namespace constant_dot_product_l306_30666

-- Define the curve E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point D
def D : ℝ × ℝ := (-2, 0)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from D to a point
def vector_DA (A : ℝ × ℝ) : ℝ × ℝ := (A.1 - D.1, A.2 - D.2)

theorem constant_dot_product :
  ∀ A B : ℝ × ℝ, A ∈ E → B ∈ E →
  dot_product (vector_DA A) (vector_DA B) = 3 := by sorry

end constant_dot_product_l306_30666


namespace expected_rolls_for_2010_l306_30612

/-- Represents the probability of getting a certain sum with a fair six-sided die -/
def probability (n : ℕ) : ℚ :=
  sorry

/-- Represents the expected number of rolls to reach a sum of n with a fair six-sided die -/
def expected_rolls (n : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the expected number of rolls to reach a sum of 2010 -/
theorem expected_rolls_for_2010 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000000 ∧ 
  abs (expected_rolls 2010 - 574523809/1000000) < ε :=
sorry

end expected_rolls_for_2010_l306_30612


namespace tan_beta_value_l306_30650

theorem tan_beta_value (α β : Real) 
  (h1 : (Real.sin α * Real.cos α) / (Real.cos (2 * α) + 1) = 1)
  (h2 : Real.tan (α - β) = 3) : 
  Real.tan β = -1/7 := by sorry

end tan_beta_value_l306_30650


namespace journey_time_l306_30677

/-- Represents a journey with constant speed -/
structure Journey where
  quarter_time : ℝ  -- Time to cover 1/4 of the journey
  third_time : ℝ    -- Time to cover 1/3 of the journey

/-- The total time for the journey -/
def total_time (j : Journey) : ℝ :=
  (j.third_time - j.quarter_time) * 12 - j.quarter_time

/-- Theorem stating that for the given journey, the total time is 280 minutes -/
theorem journey_time (j : Journey) 
  (h1 : j.quarter_time = 20) 
  (h2 : j.third_time = 45) : 
  total_time j = 280 := by
  sorry

end journey_time_l306_30677


namespace range_of_a_for_sufficient_not_necessary_l306_30633

theorem range_of_a_for_sufficient_not_necessary (a : ℝ) : 
  (∀ x : ℝ, x < 1 → x < a) ∧ 
  (∃ x : ℝ, x < a ∧ x ≥ 1) ↔ 
  a > 1 :=
by sorry

end range_of_a_for_sufficient_not_necessary_l306_30633


namespace johns_friends_l306_30662

theorem johns_friends (num_pizzas : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) :
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  slices_per_person = 4 →
  (num_pizzas * slices_per_pizza) / slices_per_person - 1 = 5 :=
by
  sorry

end johns_friends_l306_30662


namespace v2_equals_14_l306_30627

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qin_jiushao (a b c d e : ℝ) (x : ℝ) : ℝ × ℝ × ℝ := 
  let v₀ := x
  let v₁ := a * x + b
  let v₂ := v₁ * x + c
  (v₀, v₁, v₂)

/-- The theorem stating that v₂ = 14 for the given function and x = 2 -/
theorem v2_equals_14 : 
  let (v₀, v₁, v₂) := qin_jiushao 2 3 0 5 (-4) 2
  v₂ = 14 := by
sorry

end v2_equals_14_l306_30627


namespace paint_bottle_cost_l306_30636

theorem paint_bottle_cost (num_cars num_paintbrushes num_paint_bottles : ℕ)
                          (car_cost paintbrush_cost total_spent : ℚ)
                          (h1 : num_cars = 5)
                          (h2 : num_paintbrushes = 5)
                          (h3 : num_paint_bottles = 5)
                          (h4 : car_cost = 20)
                          (h5 : paintbrush_cost = 2)
                          (h6 : total_spent = 160)
                          : (total_spent - (num_cars * car_cost + num_paintbrushes * paintbrush_cost)) / num_paint_bottles = 10 := by
  sorry

end paint_bottle_cost_l306_30636


namespace girls_attending_event_l306_30690

theorem girls_attending_event (total_students : ℕ) (total_attending : ℕ) 
  (h_total : total_students = 1500)
  (h_attending : total_attending = 900)
  (h_girls_ratio : ∀ g : ℕ, g ≤ total_students → (3 * g) / 4 ≤ total_attending)
  (h_boys_ratio : ∀ b : ℕ, b ≤ total_students → (2 * b) / 5 ≤ total_attending)
  (h_all_students : ∀ g b : ℕ, g + b = total_students → (3 * g) / 4 + (2 * b) / 5 = total_attending) :
  ∃ g : ℕ, g ≤ total_students ∧ (3 * g) / 4 = 643 := by
sorry

end girls_attending_event_l306_30690


namespace symmetric_points_sum_l306_30607

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given that point A(1,a) and point B(b,-2) are symmetric with respect to the origin O, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b (-2)) : a + b = 1 := by
  sorry

end symmetric_points_sum_l306_30607


namespace granger_age_is_42_l306_30695

/-- Mr. Granger's current age -/
def granger_age : ℕ := sorry

/-- Mr. Granger's son's current age -/
def son_age : ℕ := sorry

/-- First condition: Mr. Granger's age is 10 years more than twice his son's age -/
axiom condition1 : granger_age = 2 * son_age + 10

/-- Second condition: Last year, Mr. Granger's age was 4 years less than 3 times his son's age -/
axiom condition2 : granger_age - 1 = 3 * (son_age - 1) - 4

/-- Theorem: Mr. Granger's age is 42 years -/
theorem granger_age_is_42 : granger_age = 42 := by sorry

end granger_age_is_42_l306_30695


namespace symmetry_of_sum_and_product_l306_30603

-- Define a property for function symmetry about a point
def SymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * b

-- Theorem statement
theorem symmetry_of_sum_and_product 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : SymmetricAbout f a b) (hg : SymmetricAbout g a b) :
  (SymmetricAbout (fun x ↦ f x + g x) a (2 * b)) ∧
  (∃ f g : ℝ → ℝ, SymmetricAbout f 0 0 ∧ SymmetricAbout g 0 0 ∧
    ¬∃ c d : ℝ, SymmetricAbout (fun x ↦ f x * g x) c d) :=
by sorry

end symmetry_of_sum_and_product_l306_30603


namespace ab_ratio_for_inscribed_triangle_l306_30676

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - p.y)^2 / e.a^2 + (p.x + p.y)^2 / e.b^2 = 1

/-- Checks if two points form a line parallel to y = x -/
def isParallelToYEqualX (p1 p2 : Point) : Prop :=
  p1.x - p1.y = p2.x - p2.y

/-- Theorem: AB/b ratio for an equilateral triangle inscribed in a specific ellipse -/
theorem ab_ratio_for_inscribed_triangle
  (e : Ellipse)
  (t : EquilateralTriangle)
  (h1 : t.A = ⟨0, e.b⟩)
  (h2 : isOnEllipse t.A e ∧ isOnEllipse t.B e ∧ isOnEllipse t.C e)
  (h3 : isParallelToYEqualX t.B t.C)
  (h4 : e.a = e.b * Real.sqrt 2)  -- Condition for focus at vertex C
  : Real.sqrt ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2) / e.b = 8/5 :=
sorry

end ab_ratio_for_inscribed_triangle_l306_30676


namespace point_coordinates_given_distance_to_x_axis_l306_30675

def distance_to_x_axis (y : ℝ) : ℝ := |y|

theorem point_coordinates_given_distance_to_x_axis (m : ℝ) :
  distance_to_x_axis m = 4 → m = 4 ∨ m = -4 := by
  sorry

end point_coordinates_given_distance_to_x_axis_l306_30675


namespace negation_of_universal_proposition_l306_30611

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 5*x + 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 5*x + 3 > 0) := by
  sorry

end negation_of_universal_proposition_l306_30611


namespace sqrt_a_squared_b_l306_30653

theorem sqrt_a_squared_b (a b : ℝ) (h : a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end sqrt_a_squared_b_l306_30653


namespace geometric_sum_10_terms_l306_30604

theorem geometric_sum_10_terms : 
  let a : ℚ := 3/4
  let r : ℚ := 3/4
  let n : ℕ := 10
  let S : ℚ := (a * (1 - r^n)) / (1 - r)
  S = 2971581/1048576 := by sorry

end geometric_sum_10_terms_l306_30604


namespace f_2012_eq_neg_2_l306_30686

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2012_eq_neg_2 (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1)))
  (h2 : is_even (λ x => f (x - 1)))
  (h3 : f 0 = 2) :
  f 2012 = -2 := by
  sorry

end f_2012_eq_neg_2_l306_30686


namespace f_not_differentiable_at_zero_l306_30600

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.sin (x * Real.sin (3 / x)) else 0

theorem f_not_differentiable_at_zero :
  ¬ DifferentiableAt ℝ f 0 := by sorry

end f_not_differentiable_at_zero_l306_30600


namespace square_to_rectangle_ratio_l306_30632

theorem square_to_rectangle_ratio : 
  ∀ (square_side : ℝ) (rectangle_base rectangle_height : ℝ),
  square_side = 4 →
  rectangle_base = 2 * Real.sqrt 5 →
  rectangle_height * rectangle_base = square_side^2 →
  rectangle_height / rectangle_base = 0.8 := by
  sorry

end square_to_rectangle_ratio_l306_30632


namespace correct_calculation_l306_30679

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end correct_calculation_l306_30679


namespace union_intersection_relation_l306_30667

theorem union_intersection_relation (M N : Set α) : 
  (∃ (x : α), x ∈ M ∩ N → x ∈ M ∪ N) ∧ 
  (∃ (M N : Set α), (∃ (x : α), x ∈ M ∪ N) ∧ M ∩ N = ∅) :=
by sorry

end union_intersection_relation_l306_30667


namespace function_proof_l306_30623

/-- Given a function f(x) = a^x + b, prove that if f(1) = 3 and f(0) = 2, then f(x) = 2^x + 1 -/
theorem function_proof (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a^x + b) 
  (h2 : f 1 = 3) (h3 : f 0 = 2) : ∀ x, f x = 2^x + 1 := by
  sorry

end function_proof_l306_30623


namespace freshmen_psych_liberal_arts_percentage_is_four_l306_30699

/-- The percentage of students who are freshmen psychology majors in the School of Liberal Arts -/
def freshmen_psych_liberal_arts_percentage (total_students : ℕ) : ℚ :=
  let freshmen_percentage : ℚ := 50 / 100
  let liberal_arts_percentage : ℚ := 40 / 100
  let psychology_percentage : ℚ := 20 / 100
  freshmen_percentage * liberal_arts_percentage * psychology_percentage * 100

theorem freshmen_psych_liberal_arts_percentage_is_four (total_students : ℕ) :
  freshmen_psych_liberal_arts_percentage total_students = 4 := by
  sorry

end freshmen_psych_liberal_arts_percentage_is_four_l306_30699


namespace darwin_money_problem_l306_30645

theorem darwin_money_problem (initial_money : ℝ) : 
  (3/4 * (2/3 * initial_money) = 300) → initial_money = 600 := by
  sorry

end darwin_money_problem_l306_30645


namespace composite_sequence_l306_30646

theorem composite_sequence (a n : ℕ) (ha : a ≥ 2) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (a^k + i).Prime = false :=
sorry

end composite_sequence_l306_30646


namespace square_to_eight_acute_triangles_l306_30619

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : a > 0 ∧ b > 0 ∧ c > 0
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define an acute-angled triangle
def IsAcuteAngled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2 ∧
  t.b^2 + t.c^2 > t.a^2 ∧
  t.c^2 + t.a^2 > t.b^2

-- Theorem: A square can be divided into 8 acute-angled triangles
theorem square_to_eight_acute_triangles (s : Square) :
  ∃ (t₁ t₂ t₃ t₄ t₅ t₆ t₇ t₈ : Triangle),
    IsAcuteAngled t₁ ∧
    IsAcuteAngled t₂ ∧
    IsAcuteAngled t₃ ∧
    IsAcuteAngled t₄ ∧
    IsAcuteAngled t₅ ∧
    IsAcuteAngled t₆ ∧
    IsAcuteAngled t₇ ∧
    IsAcuteAngled t₈ :=
  sorry

end square_to_eight_acute_triangles_l306_30619


namespace solution_fraction_l306_30617

theorem solution_fraction (initial_amount : ℝ) (first_day_fraction : ℝ) (second_day_addition : ℝ) : 
  initial_amount = 4 →
  first_day_fraction = 1/2 →
  second_day_addition = 1 →
  (initial_amount - first_day_fraction * initial_amount + second_day_addition) / initial_amount = 3/4 := by
  sorry

end solution_fraction_l306_30617


namespace lamp_arrangement_theorem_l306_30641

/-- The probability of a specific arrangement of lamps -/
def lamp_arrangement_probability (total_lamps green_lamps on_lamps : ℕ) : ℚ :=
  let favorable_arrangements := (Nat.choose 6 3) * (Nat.choose 7 3)
  let total_arrangements := (Nat.choose total_lamps green_lamps) * (Nat.choose total_lamps on_lamps)
  (favorable_arrangements : ℚ) / total_arrangements

/-- The specific lamp arrangement probability for 8 lamps, 4 green, 4 on -/
def specific_lamp_probability : ℚ := lamp_arrangement_probability 8 4 4

theorem lamp_arrangement_theorem : specific_lamp_probability = 10 / 49 := by
  sorry

end lamp_arrangement_theorem_l306_30641


namespace system_of_inequalities_solution_l306_30687

theorem system_of_inequalities_solution (x : ℝ) :
  (4 * x^2 - 27 * x + 18 > 0 ∧ x^2 + 4 * x + 4 > 0) ↔ 
  ((x < 3/4 ∨ x > 6) ∧ x ≠ -2) :=
by sorry

end system_of_inequalities_solution_l306_30687


namespace count_even_factors_l306_30610

def n : ℕ := 2^4 * 3^3 * 5^2

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 48 := by sorry

end count_even_factors_l306_30610


namespace exponent_multiplication_l306_30639

theorem exponent_multiplication (x : ℝ) (m n : ℕ) :
  x^m * x^n = x^(m + n) := by sorry

end exponent_multiplication_l306_30639


namespace boys_in_row_l306_30616

/-- Represents the number of boys in a row with given conditions -/
def number_of_boys (left_position right_position between : ℕ) : ℕ :=
  left_position + between + right_position

/-- Theorem stating that under the given conditions, the number of boys in the row is 24 -/
theorem boys_in_row :
  let rajan_position := 6
  let vinay_position := 10
  let boys_between := 8
  number_of_boys rajan_position vinay_position boys_between = 24 := by
  sorry

end boys_in_row_l306_30616


namespace special_circle_equation_midpoint_trajectory_l306_30628

/-- A circle passing through two points with its center on a line -/
structure SpecialCircle where
  -- The circle passes through these two points
  A : ℝ × ℝ := (1, 0)
  B : ℝ × ℝ := (-1, -2)
  -- The center C lies on this line
  center_line : ℝ × ℝ → Prop := fun (x, y) ↦ x - y + 1 = 0

/-- The endpoint B of line segment AB -/
def endpointB : ℝ × ℝ := (4, 3)

theorem special_circle_equation (c : SpecialCircle) :
  ∃ (center : ℝ × ℝ),
    c.center_line center ∧
    ∀ (x y : ℝ), (x + 1)^2 + y^2 = 4 ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (c.A.1 - center.1)^2 + (c.A.2 - center.2)^2 ∧
       (x - center.1)^2 + (y - center.2)^2 = (c.B.1 - center.1)^2 + (c.B.2 - center.2)^2) :=
sorry

theorem midpoint_trajectory (c : SpecialCircle) :
  ∀ (x y : ℝ), (x - 1.5)^2 + (y - 1.5)^2 = 1 ↔
    ∃ (a : ℝ × ℝ), 
      (a.1 + 1)^2 + a.2^2 = 4 ∧
      x = (a.1 + endpointB.1) / 2 ∧
      y = (a.2 + endpointB.2) / 2 :=
sorry

end special_circle_equation_midpoint_trajectory_l306_30628


namespace students_not_in_program_x_l306_30658

/-- Represents a grade level in the school -/
inductive GradeLevel
  | Elementary
  | Middle
  | High

/-- Represents the gender of students -/
inductive Gender
  | Girl
  | Boy

/-- The number of students in each grade level and gender -/
def studentCount (level : GradeLevel) (gender : Gender) : ℕ :=
  match level, gender with
  | GradeLevel.Elementary, Gender.Girl => 192
  | GradeLevel.Elementary, Gender.Boy => 135
  | GradeLevel.Middle, Gender.Girl => 233
  | GradeLevel.Middle, Gender.Boy => 163
  | GradeLevel.High, Gender.Girl => 117
  | GradeLevel.High, Gender.Boy => 89

/-- The number of students in Program X for each grade level and gender -/
def programXCount (level : GradeLevel) (gender : Gender) : ℕ :=
  match level, gender with
  | GradeLevel.Elementary, Gender.Girl => 48
  | GradeLevel.Elementary, Gender.Boy => 28
  | GradeLevel.Middle, Gender.Girl => 98
  | GradeLevel.Middle, Gender.Boy => 51
  | GradeLevel.High, Gender.Girl => 40
  | GradeLevel.High, Gender.Boy => 25

/-- The total number of students not participating in Program X -/
def studentsNotInProgramX : ℕ :=
  (studentCount GradeLevel.Elementary Gender.Girl - programXCount GradeLevel.Elementary Gender.Girl) +
  (studentCount GradeLevel.Elementary Gender.Boy - programXCount GradeLevel.Elementary Gender.Boy) +
  (studentCount GradeLevel.Middle Gender.Girl - programXCount GradeLevel.Middle Gender.Girl) +
  (studentCount GradeLevel.Middle Gender.Boy - programXCount GradeLevel.Middle Gender.Boy) +
  (studentCount GradeLevel.High Gender.Girl - programXCount GradeLevel.High Gender.Girl) +
  (studentCount GradeLevel.High Gender.Boy - programXCount GradeLevel.High Gender.Boy)

theorem students_not_in_program_x :
  studentsNotInProgramX = 639 := by
  sorry

end students_not_in_program_x_l306_30658


namespace circular_arrangement_students_l306_30674

/-- Given a circular arrangement of students, if the 10th and 45th positions
    are opposite each other, then the total number of students is 70. -/
theorem circular_arrangement_students (n : ℕ) : 
  (10 + n / 2 ≡ 45 [MOD n]) → n = 70 := by sorry

end circular_arrangement_students_l306_30674


namespace min_value_expression_l306_30634

theorem min_value_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (|a - 3*b - 2| + |3*a - b|) / Real.sqrt (a^2 + (b + 1)^2) ≥ 2 ∧
  (a = 0 ∧ b = 0 → (|a - 3*b - 2| + |3*a - b|) / Real.sqrt (a^2 + (b + 1)^2) = 2) :=
by sorry

#check min_value_expression

end min_value_expression_l306_30634


namespace inequality_implies_a_squared_gt_3b_l306_30621

theorem inequality_implies_a_squared_gt_3b (a b c : ℝ) 
  (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : 
  a^2 > 3 * b := by
  sorry

end inequality_implies_a_squared_gt_3b_l306_30621


namespace inequality_preservation_l306_30671

theorem inequality_preservation (a b c : ℝ) (h : a < b) : a - c < b - c := by
  sorry

end inequality_preservation_l306_30671


namespace ratio_problem_l306_30684

theorem ratio_problem (a b c : ℚ) 
  (h1 : a / b = (-5/4) / (3/2))
  (h2 : b / c = (2/3) / (-5)) :
  a / c = 1 / 6 := by sorry

end ratio_problem_l306_30684


namespace dog_arrangement_theorem_l306_30654

theorem dog_arrangement_theorem (n : ℕ) (h : n = 5) :
  (n! / 2) = 60 :=
sorry

end dog_arrangement_theorem_l306_30654


namespace field_length_proof_l306_30691

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 5 →
  pond_side^2 = (1/8) * (length * width) →
  length = 20 := by
  sorry

end field_length_proof_l306_30691


namespace death_rate_calculation_l306_30649

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate (people per two seconds) -/
def birth_rate : ℕ := 7

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 259200

/-- Represents the death rate (people per two seconds) -/
def death_rate : ℕ := 1

theorem death_rate_calculation :
  (birth_rate - death_rate) * seconds_per_day / 2 = net_increase_per_day :=
sorry

end death_rate_calculation_l306_30649


namespace expression_evaluation_l306_30643

theorem expression_evaluation (b : ℝ) (h : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 := by
  sorry

end expression_evaluation_l306_30643


namespace events_A_B_mutually_exclusive_events_A_C_not_independent_l306_30615

/-- Represents the possible outcomes when drawing a ball from Box A -/
inductive BoxA
| one
| two
| three
| four

/-- Represents the possible outcomes when drawing a ball from Box B -/
inductive BoxB
| five
| six
| seven
| eight

/-- The type of all possible outcomes when drawing one ball from each box -/
def Outcome := BoxA × BoxB

/-- The sum of the numbers on the balls drawn -/
def sum (o : Outcome) : ℕ :=
  match o with
  | (BoxA.one, b) => 1 + boxBToNat b
  | (BoxA.two, b) => 2 + boxBToNat b
  | (BoxA.three, b) => 3 + boxBToNat b
  | (BoxA.four, b) => 4 + boxBToNat b
where
  boxBToNat : BoxB → ℕ
  | BoxB.five => 5
  | BoxB.six => 6
  | BoxB.seven => 7
  | BoxB.eight => 8

/-- Event A: the sum of the numbers drawn is even -/
def eventA (o : Outcome) : Prop := Even (sum o)

/-- Event B: the sum of the numbers drawn is 9 -/
def eventB (o : Outcome) : Prop := sum o = 9

/-- Event C: the sum of the numbers drawn is greater than 9 -/
def eventC (o : Outcome) : Prop := sum o > 9

/-- The probability measure on the sample space -/
def P : Set Outcome → ℝ := sorry

theorem events_A_B_mutually_exclusive :
  ∀ o : Outcome, ¬(eventA o ∧ eventB o) := by sorry

theorem events_A_C_not_independent :
  P {o | eventA o ∧ eventC o} ≠ P {o | eventA o} * P {o | eventC o} := by sorry

end events_A_B_mutually_exclusive_events_A_C_not_independent_l306_30615


namespace triangle_problem_l306_30664

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.c = 3)
  (h3 : Real.cos t.B = 1/4) :
  t.b = Real.sqrt 10 ∧ Real.sin (2 * t.C) = (3 * Real.sqrt 15) / 16 := by
  sorry


end triangle_problem_l306_30664


namespace balloon_difference_l306_30626

theorem balloon_difference (allan_balloons jake_balloons : ℕ) : 
  allan_balloons = 5 →
  jake_balloons = 11 →
  jake_balloons > allan_balloons →
  jake_balloons - allan_balloons = 6 := by
  sorry

end balloon_difference_l306_30626


namespace arithmetic_sequence_count_l306_30661

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 4.5
  let aₙ : ℝ := 56.5
  let d : ℝ := 4
  let n := (aₙ - a₁) / d + 1
  n = 14 := by sorry

end arithmetic_sequence_count_l306_30661


namespace union_complement_problem_l306_30678

theorem union_complement_problem (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
    (hA : A = {3, 4}) (hB : B = {1, 4, 5}) :
  A ∪ (U \ B) = {2, 3, 4} := by
sorry

end union_complement_problem_l306_30678


namespace books_to_tables_ratio_l306_30673

theorem books_to_tables_ratio 
  (num_tables : ℕ) 
  (total_books : ℕ) 
  (h1 : num_tables = 500) 
  (h2 : total_books = 100000) : 
  (total_books / num_tables : ℚ) = 200 := by
sorry

end books_to_tables_ratio_l306_30673


namespace extremum_at_negative_one_l306_30692

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extremum_at_negative_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a (-1) ≤ f a x ∨ f a (-1) ≥ f a x) → 
  a = 3 := by
sorry

end extremum_at_negative_one_l306_30692


namespace balloon_height_calculation_l306_30693

theorem balloon_height_calculation (initial_budget : ℚ) (sheet_cost : ℚ) (rope_cost : ℚ) (propane_cost : ℚ) (helium_price_per_oz : ℚ) (height_per_oz : ℚ) : 
  initial_budget = 200 →
  sheet_cost = 42 →
  rope_cost = 18 →
  propane_cost = 14 →
  helium_price_per_oz = 3/2 →
  height_per_oz = 113 →
  ((initial_budget - sheet_cost - rope_cost - propane_cost) / helium_price_per_oz) * height_per_oz = 9492 :=
by sorry

end balloon_height_calculation_l306_30693


namespace quadratic_function_theorem_l306_30631

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The derivative of a function -/
def HasDerivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

/-- A quadratic equation has two equal real roots -/
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - r| < δ → |f x| < ε)

/-- The main theorem -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : HasEqualRoots f)
  (h3 : HasDerivative f (λ x ↦ 2 * x + 2)) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end quadratic_function_theorem_l306_30631


namespace polynomial_simplification_l306_30689

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 6 * x - 8) + (-7 * x^4 - 4 * x^3 + 2 * x^2 - 6 * x + 15) =
  -5 * x^4 - x^3 - 3 * x^2 + 7 :=
by sorry

end polynomial_simplification_l306_30689


namespace radical_conjugate_sum_product_l306_30614

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 1 → 
  a + b = 5 := by
sorry

end radical_conjugate_sum_product_l306_30614


namespace solution_check_l306_30663

theorem solution_check (x : ℝ) : x = 2 →
  (2 * x - 4 = 0) ∧ 
  (3 * x + 6 ≠ 0) ∧ 
  (2 * x + 4 ≠ 0) ∧ 
  (1/2 * x ≠ -4) := by
sorry

end solution_check_l306_30663


namespace statement_equivalence_l306_30652

theorem statement_equivalence (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  sorry

end statement_equivalence_l306_30652


namespace second_year_interest_rate_l306_30635

/-- Calculates the interest rate for the second year given the initial principal,
    first year interest rate, and final amount after two years. -/
theorem second_year_interest_rate
  (initial_principal : ℝ)
  (first_year_rate : ℝ)
  (final_amount : ℝ)
  (h1 : initial_principal = 4000)
  (h2 : first_year_rate = 0.04)
  (h3 : final_amount = 4368) :
  let first_year_amount := initial_principal * (1 + first_year_rate)
  let second_year_rate := (final_amount / first_year_amount) - 1
  second_year_rate = 0.05 := by
sorry

end second_year_interest_rate_l306_30635


namespace sqrt_equation_solution_l306_30605

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 3 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 1 →
  x ≥ 8 + Real.sqrt 17 := by
  sorry

end sqrt_equation_solution_l306_30605


namespace simplify_and_rationalize_l306_30670

theorem simplify_and_rationalize :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (Real.sqrt 8 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 12) / (Real.sqrt 9 / Real.sqrt 14) = 
  Real.sqrt a / b ∧
  a = 28 ∧ b = 3 := by
  sorry

end simplify_and_rationalize_l306_30670


namespace cube_difference_l306_30669

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := by
  sorry

end cube_difference_l306_30669
