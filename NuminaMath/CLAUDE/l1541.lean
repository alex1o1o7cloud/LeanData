import Mathlib

namespace amc8_paths_count_l1541_154159

/-- Represents a position on the grid --/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a letter on the grid --/
inductive Letter
  | A | M | C | Eight

/-- Defines the grid layout --/
def grid : Position → Letter := sorry

/-- Checks if two positions are adjacent --/
def isAdjacent (p1 p2 : Position) : Bool := sorry

/-- Defines a valid path on the grid --/
def ValidPath : List Position → Prop := sorry

/-- Counts the number of valid paths spelling AMC8 --/
def countAMC8Paths : Nat := sorry

/-- Theorem stating that the number of valid AMC8 paths is 24 --/
theorem amc8_paths_count : countAMC8Paths = 24 := by sorry

end amc8_paths_count_l1541_154159


namespace quadratic_sum_l1541_154106

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * x^2 + b * x + c) →
  (QuadraticFunction a b c 1 = 64) →
  (QuadraticFunction a b c (-2) = 0) →
  (QuadraticFunction a b c 4 = 0) →
  a + b + c = 64 := by
  sorry

end quadratic_sum_l1541_154106


namespace inverse_proportion_ratio_l1541_154124

/-- Given that x is inversely proportional to y, this function represents their relationship -/
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_ratio
  (x₁ x₂ y₁ y₂ : ℝ)
  (hx₁ : x₁ ≠ 0)
  (hx₂ : x₂ ≠ 0)
  (hy₁ : y₁ ≠ 0)
  (hy₂ : y₂ ≠ 0)
  (hxy₁ : inverse_proportion x₁ y₁)
  (hxy₂ : inverse_proportion x₂ y₂)
  (hx_ratio : x₁ / x₂ = 3 / 4) :
  y₁ / y₂ = 4 / 3 := by
sorry

end inverse_proportion_ratio_l1541_154124


namespace fraction_puzzle_l1541_154157

theorem fraction_puzzle : ∃ (x y : ℕ), 
  x + 35 = y ∧ 
  x ≠ 0 ∧ 
  y ≠ 0 ∧
  (x : ℚ) / y + (x.gcd y : ℚ) * x / ((y.gcd x) * y) = 16 / 13 ∧
  x = 56 ∧
  y = 91 := by
sorry

end fraction_puzzle_l1541_154157


namespace square_difference_equality_l1541_154103

theorem square_difference_equality : (19 + 12)^2 - (12^2 + 19^2) = 456 := by
  sorry

end square_difference_equality_l1541_154103


namespace no_solution_l1541_154195

/-- The function f(t) = t^3 + t -/
def f (t : ℚ) : ℚ := t^3 + t

/-- Iterative application of f, n times -/
def f_iter (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

/-- There do not exist rational numbers x and y and positive integers m and n
    such that xy = 3 and f^m(x) = f^n(y) -/
theorem no_solution :
  ¬ ∃ (x y : ℚ) (m n : ℕ+), x * y = 3 ∧ f_iter m x = f_iter n y := by
  sorry

end no_solution_l1541_154195


namespace function_is_linear_l1541_154153

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the equation is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by sorry

end function_is_linear_l1541_154153


namespace imaginary_part_of_complex_fraction_l1541_154111

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 3*Complex.I) / (3 - Complex.I) → z.im = 2/5 := by
  sorry

end imaginary_part_of_complex_fraction_l1541_154111


namespace school_pencil_order_l1541_154114

/-- The number of pencils each student receives -/
def pencils_per_student : ℕ := 3

/-- The number of students in the school -/
def number_of_students : ℕ := 65

/-- The total number of pencils ordered by the school -/
def total_pencils : ℕ := pencils_per_student * number_of_students

theorem school_pencil_order : total_pencils = 195 := by
  sorry

end school_pencil_order_l1541_154114


namespace budget_reduction_proof_l1541_154165

def magazine_cost : ℝ := 840.00
def online_cost_pounds : ℝ := 960.00
def exchange_rate : ℝ := 1.40
def magazine_cut_rate : ℝ := 0.30
def online_cut_rate : ℝ := 0.20

def total_reduction : ℝ :=
  (magazine_cost * magazine_cut_rate) +
  (online_cost_pounds * online_cut_rate * exchange_rate)

theorem budget_reduction_proof :
  total_reduction = 520.80 := by
sorry

end budget_reduction_proof_l1541_154165


namespace right_triangle_perimeter_equal_area_l1541_154179

theorem right_triangle_perimeter_equal_area (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive integers
  a^2 + b^2 = c^2 →        -- Right-angled triangle (Pythagorean theorem)
  a + b + c = (a * b) / 2  -- Perimeter equals area
  → (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
    (a = 8 ∧ b = 6 ∧ c = 10) ∨ (a = 12 ∧ b = 5 ∧ c = 13) :=
by sorry

#check right_triangle_perimeter_equal_area

end right_triangle_perimeter_equal_area_l1541_154179


namespace twins_shirts_l1541_154137

/-- The number of shirts Hazel and Razel have in total -/
def total_shirts (hazel_shirts : ℕ) (razel_shirts : ℕ) : ℕ :=
  hazel_shirts + razel_shirts

/-- Theorem: If Hazel received 6 shirts and Razel received twice the number of shirts as Hazel,
    then the total number of shirts they have is 18. -/
theorem twins_shirts :
  let hazel_shirts : ℕ := 6
  let razel_shirts : ℕ := 2 * hazel_shirts
  total_shirts hazel_shirts razel_shirts = 18 := by
sorry

end twins_shirts_l1541_154137


namespace average_daily_low_temp_l1541_154120

def daily_low_temperatures : List ℝ := [40, 47, 45, 41, 39]

theorem average_daily_low_temp : 
  (daily_low_temperatures.sum / daily_low_temperatures.length : ℝ) = 42.4 := by
  sorry

end average_daily_low_temp_l1541_154120


namespace fraction_to_decimal_decimal_expansion_main_proof_l1541_154177

theorem fraction_to_decimal : (7 : ℚ) / 200 = (35 : ℚ) / 1000 := by sorry

theorem decimal_expansion : (35 : ℚ) / 1000 = 0.035 := by sorry

theorem main_proof : (7 : ℚ) / 200 = 0.035 := by
  rw [fraction_to_decimal]
  exact decimal_expansion

end fraction_to_decimal_decimal_expansion_main_proof_l1541_154177


namespace train_passing_time_l1541_154140

/-- Proves that a train with given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 285 →
  train_speed_kmh = 54 →
  passing_time = 19 →
  train_length / (train_speed_kmh * 1000 / 3600) = passing_time :=
by sorry

end train_passing_time_l1541_154140


namespace lost_revenue_calculation_l1541_154158

/-- Represents the movie theater scenario --/
structure MovieTheater where
  capacity : ℕ
  ticketPrice : ℚ
  ticketsSold : ℕ

/-- Calculates the lost revenue for a movie theater --/
def lostRevenue (theater : MovieTheater) : ℚ :=
  (theater.capacity : ℚ) * theater.ticketPrice - (theater.ticketsSold : ℚ) * theater.ticketPrice

/-- Theorem stating the lost revenue for the given scenario --/
theorem lost_revenue_calculation (theater : MovieTheater) 
  (h1 : theater.capacity = 50)
  (h2 : theater.ticketPrice = 8)
  (h3 : theater.ticketsSold = 24) : 
  lostRevenue theater = 208 := by
  sorry

#eval lostRevenue { capacity := 50, ticketPrice := 8, ticketsSold := 24 }

end lost_revenue_calculation_l1541_154158


namespace mancino_garden_width_is_5_l1541_154136

/-- The width of Mancino's gardens -/
def mancino_garden_width : ℝ := 5

/-- The number of Mancino's gardens -/
def mancino_garden_count : ℕ := 3

/-- The length of Mancino's gardens -/
def mancino_garden_length : ℝ := 16

/-- The number of Marquita's gardens -/
def marquita_garden_count : ℕ := 2

/-- The length of Marquita's gardens -/
def marquita_garden_length : ℝ := 8

/-- The width of Marquita's gardens -/
def marquita_garden_width : ℝ := 4

/-- The total area of all gardens -/
def total_garden_area : ℝ := 304

theorem mancino_garden_width_is_5 :
  mancino_garden_width = 5 ∧
  mancino_garden_count * mancino_garden_length * mancino_garden_width +
  marquita_garden_count * marquita_garden_length * marquita_garden_width =
  total_garden_area :=
by sorry

end mancino_garden_width_is_5_l1541_154136


namespace equations_not_equivalent_l1541_154135

theorem equations_not_equivalent : 
  ¬(∀ x : ℝ, (2 * (x - 10)) / (x^2 - 13*x + 30) = 1 ↔ x^2 - 15*x + 50 = 0) :=
by sorry

end equations_not_equivalent_l1541_154135


namespace units_digit_sum_factorials_2010_l1541_154197

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2010 :
  sum_factorials 2010 % 10 = 3 := by
  sorry

end units_digit_sum_factorials_2010_l1541_154197


namespace cd_total_length_l1541_154110

theorem cd_total_length : 
  let cd1 : ℝ := 1.5
  let cd2 : ℝ := 1.5
  let cd3 : ℝ := 2 * cd1
  let cd4 : ℝ := 0.5 * cd2
  let cd5 : ℝ := cd1 + cd2
  cd1 + cd2 + cd3 + cd4 + cd5 = 9.75 := by
sorry

end cd_total_length_l1541_154110


namespace triangle_side_comparison_l1541_154147

theorem triangle_side_comparison (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (Real.sin A > Real.sin B) →
  (a > b) := by
sorry

end triangle_side_comparison_l1541_154147


namespace senate_arrangement_l1541_154161

/-- The number of ways to arrange senators around a circular table. -/
def arrange_senators (num_democrats num_republicans : ℕ) : ℕ :=
  (num_republicans - 1).factorial * (num_republicans.choose num_democrats) * num_democrats.factorial

/-- Theorem: The number of ways to arrange 4 Democrats and 6 Republicans around a circular table
    such that no two Democrats sit next to each other is 43,200. -/
theorem senate_arrangement :
  arrange_senators 4 6 = 43200 :=
sorry

end senate_arrangement_l1541_154161


namespace marbles_in_jar_l1541_154150

/-- The number of marbles in a jar when two boys combine their collections -/
theorem marbles_in_jar (ben_marbles : ℕ) (leo_extra_marbles : ℕ) : 
  ben_marbles = 56 → leo_extra_marbles = 20 → 
  ben_marbles + (ben_marbles + leo_extra_marbles) = 132 := by
  sorry

#check marbles_in_jar

end marbles_in_jar_l1541_154150


namespace angle_sum_at_point_l1541_154113

/-- 
Given three angles that meet at a point in a plane, 
if two of the angles are 145° and 95°, 
then the third angle is 120°.
-/
theorem angle_sum_at_point (a b c : ℝ) : 
  a + b + c = 360 → a = 145 → b = 95 → c = 120 := by sorry

end angle_sum_at_point_l1541_154113


namespace addition_point_value_l1541_154167

/-- The 0.618 method for finding the optimal addition amount --/
def addition_point (lower upper good : ℝ) : ℝ :=
  upper + lower - good

/-- Theorem: The addition point value using the 0.618 method --/
theorem addition_point_value (lower upper good : ℝ)
  (h_range : lower = 628 ∧ upper = 774)
  (h_good : good = lower + 0.618 * (upper - lower))
  (h_good_value : good = 718) :
  addition_point lower upper good = 684 := by
  sorry

#eval addition_point 628 774 718

end addition_point_value_l1541_154167


namespace f_two_zeros_l1541_154131

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

theorem f_two_zeros (a : ℝ) :
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ f a z1 = 0 ∧ f a z2 = 0 ∧ ∀ z, f a z = 0 → z = z1 ∨ z = z2) ↔
  (1/2 ≤ a ∧ a < 1) ∨ (2 ≤ a) :=
sorry

end f_two_zeros_l1541_154131


namespace horner_method_value_l1541_154160

def horner_polynomial (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℤ) : ℤ :=
  horner_polynomial [3, 5, 6, 79, -8, 35, 12] x

theorem horner_method_value :
  f (-4) = 220 := by
  sorry

end horner_method_value_l1541_154160


namespace rational_solution_system_l1541_154162

theorem rational_solution_system (x y z t w : ℚ) :
  t^2 - w^2 + z^2 = 2*x*y ∧
  t^2 - y^2 + w^2 = 2*x*z ∧
  t^2 - w^2 + x^2 = 2*y*z →
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

#check rational_solution_system

end rational_solution_system_l1541_154162


namespace perpendicular_line_through_center_l1541_154112

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 - 3 = 0

/-- The given line equation -/
def given_line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

/-- The perpendicular line equation -/
def perpendicular_line_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the line passing through the center of the circle and perpendicular to the given line has the equation x - y + 1 = 0 -/
theorem perpendicular_line_through_center :
  ∀ (x y : ℝ), 
    (x, y) = circle_center → 
    (∀ (x' y' : ℝ), given_line_equation x' y' → (x - x') * (y - y') = -1) → 
    perpendicular_line_equation x y :=
sorry

end perpendicular_line_through_center_l1541_154112


namespace probability_estimate_l1541_154193

def is_hit (d : Nat) : Bool := d ≥ 2 ∧ d ≤ 9

def group_has_three_hits (g : List Nat) : Bool :=
  (g.filter is_hit).length ≥ 3

def count_successful_groups (groups : List (List Nat)) : Nat :=
  (groups.filter group_has_three_hits).length

theorem probability_estimate (groups : List (List Nat))
  (h1 : groups.length = 20)
  (h2 : ∀ g ∈ groups, g.length = 4)
  (h3 : ∀ g ∈ groups, ∀ d ∈ g, d ≤ 9)
  (h4 : count_successful_groups groups = 15) :
  (count_successful_groups groups : ℚ) / groups.length = 3/4 := by
  sorry

#check probability_estimate

end probability_estimate_l1541_154193


namespace mary_shirts_left_l1541_154169

/-- The number of shirts Mary has left after giving away some of her blue and brown shirts -/
def shirts_left (blue : ℕ) (brown : ℕ) : ℕ :=
  (blue - blue / 2) + (brown - brown / 3)

/-- Theorem stating that Mary has 37 shirts left -/
theorem mary_shirts_left : shirts_left 26 36 = 37 := by
  sorry

end mary_shirts_left_l1541_154169


namespace triangle_expression_l1541_154139

theorem triangle_expression (A B C : ℝ) : 
  A = 15 * π / 180 →
  A + B + C = π →
  Real.sqrt 3 * Real.sin A - Real.cos (B + C) = Real.sqrt 2 := by
  sorry

end triangle_expression_l1541_154139


namespace complex_magnitude_problem_l1541_154187

theorem complex_magnitude_problem (z : ℂ) : 
  z + Complex.I = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_magnitude_problem_l1541_154187


namespace sqrt_2x_minus_6_real_l1541_154148

theorem sqrt_2x_minus_6_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 6) ↔ x ≥ 3 := by
  sorry

end sqrt_2x_minus_6_real_l1541_154148


namespace AMC10_paths_count_l1541_154198

/-- Represents the number of paths to spell "AMC10" given specific adjacency conditions -/
def number_of_AMC10_paths (
  adjacent_Ms : Nat
  ) (adjacent_Cs : Nat)
  (adjacent_1s : Nat)
  (adjacent_0s : Nat) : Nat :=
  adjacent_Ms * adjacent_Cs * adjacent_1s * adjacent_0s

/-- Theorem stating that the number of paths to spell "AMC10" is 48 -/
theorem AMC10_paths_count :
  number_of_AMC10_paths 4 3 2 2 = 48 := by
  sorry

end AMC10_paths_count_l1541_154198


namespace f_negative_a_value_l1541_154196

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem f_negative_a_value (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end f_negative_a_value_l1541_154196


namespace parabola_equation_l1541_154133

/-- A parabola in the Cartesian coordinate system with directrix y = 4 -/
structure Parabola where
  /-- The equation of the parabola -/
  equation : ℝ → ℝ → Prop

/-- The standard form of a parabola equation -/
def StandardForm (p : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 = -4*p*y

/-- Theorem: The standard equation of a parabola with directrix y = 4 is x^2 = -16y -/
theorem parabola_equation (P : Parabola) : 
  P.equation = StandardForm 4 := by sorry

end parabola_equation_l1541_154133


namespace total_time_equals_sum_of_activities_l1541_154168

/-- The total time Joan initially had for her music practice -/
def total_time : ℕ := 120

/-- Time Joan spent on the piano -/
def piano_time : ℕ := 30

/-- Time Joan spent writing music -/
def writing_time : ℕ := 25

/-- Time Joan spent reading about piano history -/
def reading_time : ℕ := 38

/-- Time Joan has left for finger exerciser -/
def exerciser_time : ℕ := 27

/-- Theorem stating that the total time is equal to the sum of individual activity times -/
theorem total_time_equals_sum_of_activities : 
  total_time = piano_time + writing_time + reading_time + exerciser_time := by
  sorry

end total_time_equals_sum_of_activities_l1541_154168


namespace special_shape_is_regular_tetrahedron_l1541_154173

/-- A 3D shape with the property that the angle between diagonals of adjacent sides is 60 degrees -/
structure SpecialShape :=
  (is_3d : Bool)
  (diagonal_angle : ℝ)
  (angle_property : diagonal_angle = 60)

/-- Definition of a regular tetrahedron -/
structure RegularTetrahedron :=
  (is_3d : Bool)
  (num_faces : Nat)
  (face_type : String)
  (num_faces_property : num_faces = 4)
  (face_type_property : face_type = "equilateral triangle")

/-- Theorem stating that a SpecialShape is equivalent to a RegularTetrahedron -/
theorem special_shape_is_regular_tetrahedron (s : SpecialShape) : 
  ∃ (t : RegularTetrahedron), true :=
sorry

end special_shape_is_regular_tetrahedron_l1541_154173


namespace range_of_a_l1541_154156

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  -Real.sqrt 6 / 3 ≤ a ∧ a ≤ Real.sqrt 6 / 3 := by sorry

end range_of_a_l1541_154156


namespace crescent_lake_loop_length_l1541_154192

/-- Represents the distance walked on each day of the trip -/
structure DailyDistances where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (d : DailyDistances) : Prop :=
  d.day1 + d.day2 + d.day3 = 32 ∧
  (d.day2 + d.day3) / 2 = 12 ∧
  d.day3 + d.day4 + d.day5 = 45 ∧
  d.day1 + d.day4 = 30

/-- The theorem stating that if the conditions are satisfied, the total distance is 69 miles -/
theorem crescent_lake_loop_length 
  (d : DailyDistances) 
  (h : satisfies_conditions d) : 
  d.day1 + d.day2 + d.day3 + d.day4 + d.day5 = 69 := by
  sorry

end crescent_lake_loop_length_l1541_154192


namespace license_plate_combinations_count_l1541_154174

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of choices for the last character (letters + digits) -/
def last_char_choices : ℕ := num_letters + num_digits

/-- A function to calculate the number of valid license plate combinations -/
def license_plate_combinations : ℕ :=
  num_letters * last_char_choices * 2

/-- Theorem stating that the number of valid license plate combinations is 1872 -/
theorem license_plate_combinations_count :
  license_plate_combinations = 1872 := by
  sorry

end license_plate_combinations_count_l1541_154174


namespace like_terms_exponent_product_l1541_154125

theorem like_terms_exponent_product (x y : ℝ) (m n : ℕ) : 
  (∀ (a b : ℝ), a * x^3 * y^n = b * x^m * y^2 → a ≠ 0 → b ≠ 0 → m = 3 ∧ n = 2) →
  m * n = 6 := by
  sorry

end like_terms_exponent_product_l1541_154125


namespace existence_of_non_triangle_forming_numbers_l1541_154143

theorem existence_of_non_triangle_forming_numbers : 
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧ 
  (∀ (c : ℕ), ∃ (k : ℕ), c = k^2 → 
    ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by sorry

end existence_of_non_triangle_forming_numbers_l1541_154143


namespace circle_x_axis_intersection_l1541_154138

/-- Given a circle with diameter endpoints (0,0) and (10,10), 
    the x-coordinate of the second intersection point with the x-axis is 10 -/
theorem circle_x_axis_intersection :
  ∀ (C : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - 5)^2 + (y - 5)^2 = 50) →
    (0, 0) ∈ C →
    (10, 10) ∈ C →
    ∃ (x : ℝ), x ≠ 0 ∧ (x, 0) ∈ C ∧ x = 10 :=
by sorry

end circle_x_axis_intersection_l1541_154138


namespace difference_of_squares_l1541_154149

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end difference_of_squares_l1541_154149


namespace parabola_sum_l1541_154178

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_sum (p : Parabola) : 
  p.y_coord 4 = 2 ∧  -- vertex (4,2)
  p.y_coord 1 = -4 ∧  -- point (1,-4)
  p.y_coord 7 = 0 ∧  -- point (7,0)
  (∀ x : ℝ, p.y_coord (8 - x) = p.y_coord x) →  -- vertical axis of symmetry at x = 4
  p.a + p.b + p.c = -4 := by sorry

end parabola_sum_l1541_154178


namespace parabola_shift_theorem_l1541_154102

theorem parabola_shift_theorem (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁ * x₂ = x₁ + x₂ + 49 →
  x₁ * x₂ = -6 * m →
  x₁ + x₂ = 2 * m - 1 →
  min (abs x₁) (abs x₂) = 4 :=
sorry

end parabola_shift_theorem_l1541_154102


namespace claudia_coins_l1541_154123

/-- Represents the number of different coin combinations possible with n coins -/
def combinations (n : ℕ) : ℕ := sorry

/-- Represents the number of different values that can be formed with n coins -/
def values (n : ℕ) : ℕ := sorry

theorem claudia_coins :
  ∀ x y : ℕ,
  x + y = 15 →                           -- Total number of coins is 15
  combinations (x + y) = 23 →            -- 23 different combinations possible
  (∀ n : ℕ, n ≤ 10 → values n ≥ 15) →    -- At least 15 values with no more than 10 coins
  y = 9                                  -- Claudia has 9 10-cent coins
  := by sorry

end claudia_coins_l1541_154123


namespace point_condition_y_intercept_condition_l1541_154176

/-- The equation of the line -/
def line_equation (x y t : ℝ) : Prop :=
  2 * x + (t - 2) * y + 3 - 2 * t = 0

/-- Theorem: If the line passes through (1, 1), then t = 5 -/
theorem point_condition (t : ℝ) : line_equation 1 1 t → t = 5 := by
  sorry

/-- Theorem: If the y-intercept of the line is -3, then t = 9/5 -/
theorem y_intercept_condition (t : ℝ) : line_equation 0 (-3) t → t = 9/5 := by
  sorry

end point_condition_y_intercept_condition_l1541_154176


namespace lisa_equal_earnings_l1541_154146

/-- Given Greta's work hours, Greta's hourly rate, and Lisa's hourly rate,
    calculates the number of hours Lisa needs to work to equal Greta's earnings. -/
def lisa_work_hours (greta_hours : ℕ) (greta_rate : ℚ) (lisa_rate : ℚ) : ℚ :=
  (greta_hours : ℚ) * greta_rate / lisa_rate

/-- Proves that Lisa needs to work 32 hours to equal Greta's earnings,
    given the specified conditions. -/
theorem lisa_equal_earnings : lisa_work_hours 40 12 15 = 32 := by
  sorry

end lisa_equal_earnings_l1541_154146


namespace badminton_probability_l1541_154155

theorem badminton_probability (p : ℝ) (n : ℕ) : 
  p = 3/4 → n = 3 → 
  (1 - p)^n = 1/64 → 
  n.choose 1 * p * (1 - p)^(n-1) = 9/64 := by
  sorry

end badminton_probability_l1541_154155


namespace new_person_weight_l1541_154127

/-- Given a group of 10 persons, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.2 kg,
    then the weight of the new person is 97 kg. -/
theorem new_person_weight
  (n : ℕ) (old_weight average_increase : ℝ)
  (h1 : n = 10)
  (h2 : old_weight = 65)
  (h3 : average_increase = 3.2) :
  let new_weight := old_weight + n * average_increase
  new_weight = 97 := by
  sorry

end new_person_weight_l1541_154127


namespace sin_cos_difference_zero_l1541_154180

theorem sin_cos_difference_zero : Real.sin (36 * π / 180) * Real.cos (36 * π / 180) - Real.cos (36 * π / 180) * Real.sin (36 * π / 180) = 0 := by
  sorry

end sin_cos_difference_zero_l1541_154180


namespace coefficient_d_nonzero_l1541_154141

/-- A polynomial of degree 5 with five distinct roots including 0 and 1 -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- The theorem stating that the coefficient d must be nonzero -/
theorem coefficient_d_nonzero (a b c d f : ℝ) :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ p ≠ 1 ∧ q ≠ 0 ∧ q ≠ 1 ∧ r ≠ 0 ∧ r ≠ 1 ∧
    ∀ x : ℝ, Q a b c d f x = 0 ↔ x = 0 ∨ x = 1 ∨ x = p ∨ x = q ∨ x = r) →
  d ≠ 0 :=
sorry

end coefficient_d_nonzero_l1541_154141


namespace asymptote_sum_l1541_154101

/-- Given a function g(x) = (x+5) / (x^2 + cx + d) with vertical asymptotes at x = 2 and x = -3,
    prove that the sum of c and d is -5. -/
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -3 → 
    (x + 5) / (x^2 + c*x + d) = (x + 5) / ((x - 2) * (x + 3))) →
  c + d = -5 := by
sorry

end asymptote_sum_l1541_154101


namespace billion_to_scientific_notation_l1541_154126

/-- Represents the number 56.9 billion -/
def billion_value : ℝ := 56.9 * 1000000000

/-- Represents the scientific notation of 56.9 billion -/
def scientific_notation : ℝ := 5.69 * 10^9

/-- Theorem stating that 56.9 billion is equal to 5.69 × 10^9 in scientific notation -/
theorem billion_to_scientific_notation : billion_value = scientific_notation := by
  sorry

end billion_to_scientific_notation_l1541_154126


namespace chord_angle_cosine_l1541_154171

theorem chord_angle_cosine (r : ℝ) (α β : ℝ) : 
  r > 0 ∧ 
  2 * r * Real.sin (α / 2) = 2 ∧
  2 * r * Real.sin (β / 2) = 3 ∧
  2 * r * Real.sin ((α + β) / 2) = 4 ∧
  α + β < π →
  Real.cos α = 17 / 32 := by
sorry

end chord_angle_cosine_l1541_154171


namespace people_in_room_l1541_154108

theorem people_in_room (total_chairs : ℕ) (people : ℕ) : 
  (3 * people : ℚ) / 5 = (5 * total_chairs : ℚ) / 6 →  -- Three-fifths of people are seated in five-sixths of chairs
  total_chairs - (5 * total_chairs) / 6 = 10 →         -- 10 chairs are empty
  people = 83 := by
sorry

end people_in_room_l1541_154108


namespace rectangle_area_rectangle_area_is_44_l1541_154164

/-- The area of a rectangle containing two smaller squares and one larger square -/
theorem rectangle_area (small_square_area : ℝ) (h1 : small_square_area = 4) : ℝ :=
  let small_side := Real.sqrt small_square_area
  let large_side := 3 * small_side
  2 * small_square_area + large_side ^ 2

/-- Proof that the area of the rectangle is 44 square inches -/
theorem rectangle_area_is_44 : rectangle_area 4 rfl = 44 := by
  sorry

end rectangle_area_rectangle_area_is_44_l1541_154164


namespace negation_of_existence_l1541_154130

theorem negation_of_existence (p : Prop) :
  (¬ ∃ (x y : ℤ), x^2 + y^2 = 2015) ↔ (∀ (x y : ℤ), x^2 + y^2 ≠ 2015) := by
  sorry

end negation_of_existence_l1541_154130


namespace power_fraction_simplification_l1541_154191

theorem power_fraction_simplification :
  (2^2020 - 2^2018) / (2^2020 + 2^2018) = 3/5 := by
  sorry

end power_fraction_simplification_l1541_154191


namespace line_segment_proportion_l1541_154105

theorem line_segment_proportion (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end line_segment_proportion_l1541_154105


namespace mumblian_language_word_count_l1541_154172

/-- The number of letters in the Mumblian alphabet -/
def alphabet_size : ℕ := 5

/-- The maximum word length in the Mumblian language -/
def max_word_length : ℕ := 3

/-- The number of words of a given length in the Mumblian language -/
def words_of_length (n : ℕ) : ℕ := 
  if n > 0 ∧ n ≤ max_word_length then alphabet_size ^ n else 0

/-- The total number of words in the Mumblian language -/
def total_words : ℕ := 
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3)

theorem mumblian_language_word_count : total_words = 155 := by
  sorry

end mumblian_language_word_count_l1541_154172


namespace hexagon_square_side_ratio_l1541_154109

theorem hexagon_square_side_ratio (s_h s_s : ℝ) 
  (h_positive : s_h > 0 ∧ s_s > 0)
  (h_perimeter : 6 * s_h = 4 * s_s) : 
  s_s / s_h = 3 / 2 := by
  sorry

end hexagon_square_side_ratio_l1541_154109


namespace exam_mean_score_l1541_154121

/-- Given an exam score distribution where 58 is 2 standard deviations below the mean
    and 98 is 3 standard deviations above the mean, the mean score is 74. -/
theorem exam_mean_score (μ σ : ℝ) 
  (h1 : 58 = μ - 2 * σ) 
  (h2 : 98 = μ + 3 * σ) : 
  μ = 74 := by
  sorry

end exam_mean_score_l1541_154121


namespace polynomial_evaluation_l1541_154122

/-- A polynomial with integer coefficients where each coefficient is between 0 and 4 inclusive -/
def IntPolynomial (m : ℕ) := { b : Fin (m + 1) → ℤ // ∀ i, 0 ≤ b i ∧ b i < 5 }

/-- Evaluation of an IntPolynomial at a given value -/
def evalPoly {m : ℕ} (P : IntPolynomial m) (x : ℝ) : ℝ :=
  (Finset.range (m + 1)).sum (fun i => (P.val i : ℝ) * x ^ i)

theorem polynomial_evaluation (m : ℕ) (P : IntPolynomial m) :
  evalPoly P (Real.sqrt 5) = 23 + 19 * Real.sqrt 5 →
  evalPoly P 3 = 132 := by
  sorry

end polynomial_evaluation_l1541_154122


namespace sin_balanceable_same_balancing_pair_for_square_and_exp_cos_squared_balancing_pair_range_l1541_154185

/-- A function f is balanceable if there exist real numbers m and k (m ≠ 0) such that
    m * f x = f (x + k) + f (x - k) for all x in the domain of f. -/
def Balanceable (f : ℝ → ℝ) : Prop :=
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

/-- A balancing pair for a function f is a pair (m, k) that satisfies the balanceable condition. -/
def BalancingPair (f : ℝ → ℝ) (m k : ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x, m * f x = f (x + k) + f (x - k)

theorem sin_balanceable :
  ∃ n : ℤ, BalancingPair Real.sin 1 (2 * π * n + π / 3) ∨ BalancingPair Real.sin 1 (2 * π * n - π / 3) :=
sorry

theorem same_balancing_pair_for_square_and_exp :
  ∀ a : ℝ, a ≠ 0 →
  (BalancingPair (fun x ↦ x^2) 2 0 ∧ BalancingPair (fun x ↦ a + 2^x) 2 0) :=
sorry

theorem cos_squared_balancing_pair_range :
  ∃ m₁ m₂ : ℝ,
  BalancingPair (fun x ↦ Real.cos x ^ 2) m₁ (π / 2) ∧
  BalancingPair (fun x ↦ Real.cos x ^ 2) m₂ (π / 4) ∧
  ∀ x, 0 ≤ x ∧ x ≤ π / 4 → 1 ≤ m₁^2 + m₂^2 ∧ m₁^2 + m₂^2 ≤ 8 :=
sorry

end sin_balanceable_same_balancing_pair_for_square_and_exp_cos_squared_balancing_pair_range_l1541_154185


namespace existence_of_equal_sums_l1541_154175

theorem existence_of_equal_sums (m n : ℕ) (a : Fin m → ℕ) (b : Fin n → ℕ) 
  (ha : ∀ i j : Fin m, i ≤ j → a i ≤ a j) 
  (hb : ∀ i j : Fin n, i ≤ j → b i ≤ b j)
  (ha_bound : ∀ i : Fin m, a i ≤ n)
  (hb_bound : ∀ i : Fin n, b i ≤ m) :
  ∃ (i : Fin m) (j : Fin n), a i + i.val + 1 = b j + j.val + 1 := by
sorry

end existence_of_equal_sums_l1541_154175


namespace latus_rectum_equation_l1541_154119

/-- The equation of the latus rectum of the parabola y = -1/4 * x^2 -/
theorem latus_rectum_equation (x y : ℝ) :
  y = -1/4 * x^2 → (∃ (p : ℝ), p = -1/2 ∧ y = p) :=
by sorry

end latus_rectum_equation_l1541_154119


namespace division_multiplication_problem_l1541_154154

theorem division_multiplication_problem : ((-128) / (-16)) * 5 = 40 := by
  sorry

end division_multiplication_problem_l1541_154154


namespace largest_y_value_l1541_154142

theorem largest_y_value : 
  (∃ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y) → 
  (∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y → y ≤ 3/25) ∧
  (∃ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y ∧ y = 3/25) := by
  sorry

end largest_y_value_l1541_154142


namespace diophantine_equation_solutions_l1541_154107

theorem diophantine_equation_solutions (p : ℕ) (h_prime : Nat.Prime p) :
  ∀ x y n : ℕ, x > 0 ∧ y > 0 ∧ n > 0 →
  p^n = x^3 + y^3 ↔
  (p = 2 ∧ ∃ k : ℕ, x = 2^k ∧ y = 2^k ∧ n = 3*k + 1) ∨
  (p = 3 ∧ ∃ k : ℕ, (x = 3^k ∧ y = 2 * 3^k ∧ n = 3*k + 2) ∨
                    (x = 2 * 3^k ∧ y = 3^k ∧ n = 3*k + 2)) :=
by sorry


end diophantine_equation_solutions_l1541_154107


namespace even_operations_l1541_154194

-- Define an even integer
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define a perfect square
def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k * k

theorem even_operations (n : ℤ) (h : is_even n) :
  (is_even (n^2)) ∧ 
  (∀ m : ℤ, is_even m → is_perfect_square m → is_even (Int.sqrt m)) ∧
  (∀ k : ℤ, ¬(is_even k) → is_even (n * k)) ∧
  (is_even (n^3)) :=
sorry

end even_operations_l1541_154194


namespace framing_needed_l1541_154118

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photograph. -/
theorem framing_needed (orig_width orig_height border_width : ℕ) : 
  orig_width = 5 →
  orig_height = 7 →
  border_width = 3 →
  let enlarged_width := 2 * orig_width
  let enlarged_height := 2 * orig_height
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (framed_width + framed_height)
  let feet := (perimeter + 11) / 12  -- Ceiling division to get the next whole foot
  feet = 6 := by
  sorry

#check framing_needed

end framing_needed_l1541_154118


namespace inscribed_sphere_cone_relation_l1541_154181

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphereCone where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  b : ℝ
  d : ℝ
  sphere_radius_eq : sphere_radius = b * (Real.sqrt d - 1)

/-- The theorem stating the relationship between b and d for the given cone and sphere -/
theorem inscribed_sphere_cone_relation (cone : InscribedSphereCone) 
  (h1 : cone.base_radius = 15)
  (h2 : cone.height = 30) :
  cone.b + cone.d = 12.5 := by
  sorry

end inscribed_sphere_cone_relation_l1541_154181


namespace product_count_in_range_l1541_154189

theorem product_count_in_range (total_sample : ℕ) 
  (freq_96_100 : ℚ) (freq_98_104 : ℚ) (count_less_100 : ℕ) :
  freq_96_100 = 3/10 →
  freq_98_104 = 3/8 →
  count_less_100 = 36 →
  total_sample = count_less_100 / freq_96_100 →
  (freq_98_104 * total_sample : ℚ) = 60 :=
by sorry

end product_count_in_range_l1541_154189


namespace mia_bought_three_more_notebooks_l1541_154183

/-- Represents the price of a single notebook in cents -/
def notebook_price : ℕ := 50

/-- Represents the number of notebooks Colin bought -/
def colin_notebooks : ℕ := 5

/-- Represents the number of notebooks Mia bought -/
def mia_notebooks : ℕ := 8

/-- Represents Colin's total payment in cents -/
def colin_payment : ℕ := 250

/-- Represents Mia's total payment in cents -/
def mia_payment : ℕ := 400

theorem mia_bought_three_more_notebooks :
  mia_notebooks = colin_notebooks + 3 ∧
  notebook_price > 1 ∧
  notebook_price * colin_notebooks = colin_payment ∧
  notebook_price * mia_notebooks = mia_payment :=
by sorry

end mia_bought_three_more_notebooks_l1541_154183


namespace geometric_sequence_sum_ratio_l1541_154186

/-- Given a geometric sequence with common ratio 2, prove that S_4 / a_1 = 15 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n) :
  (a 0 * (1 - 2^4)) / (a 0 * (1 - 2)) = 15 := by
  sorry

end geometric_sequence_sum_ratio_l1541_154186


namespace senior_count_l1541_154104

/-- Represents the count of students in each grade level -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Given the conditions of the student sample, proves the number of seniors -/
theorem senior_count (total : ℕ) (counts : StudentCounts) : 
  total = 800 ∧ 
  counts.juniors = (23 * total) / 100 ∧ 
  counts.sophomores = (25 * total) / 100 ∧ 
  counts.freshmen = counts.sophomores + 56 ∧ 
  total = counts.freshmen + counts.sophomores + counts.juniors + counts.seniors → 
  counts.seniors = 160 := by
sorry


end senior_count_l1541_154104


namespace system_solution_l1541_154184

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * y = |2 * x + 3| - |2 * x - 3|
def equation2 (x y : ℝ) : Prop := 4 * x = |y + 2| - |y - 2|

-- Define the solution set
def solutionSet (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y ↔ solutionSet x y :=
sorry

end system_solution_l1541_154184


namespace gamma_value_l1541_154188

theorem gamma_value (γ δ : ℂ) : 
  (γ + δ).re > 0 →
  (Complex.I * (γ - 3 * δ)).re > 0 →
  δ = 4 + 3 * Complex.I →
  γ = 16 - 3 * Complex.I := by
sorry

end gamma_value_l1541_154188


namespace max_page_number_with_fifteen_fives_l1541_154128

/-- Represents the count of a specific digit in a number -/
def digitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Represents the total count of a specific digit used in numbering pages from 1 to n -/
def totalDigitCount (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The maximum page number that can be reached with a given number of a specific digit -/
def maxPageNumber (availableDigits : ℕ) (digit : ℕ) : ℕ := sorry

theorem max_page_number_with_fifteen_fives :
  maxPageNumber 15 5 = 59 := by sorry

end max_page_number_with_fifteen_fives_l1541_154128


namespace tangent_line_to_parabola_l1541_154199

/-- The value of j that makes the line 4x + 7y + j = 0 tangent to the parabola y^2 = 32x -/
def tangent_j : ℝ := 98

/-- The line equation -/
def line (x y j : ℝ) : Prop := 4 * x + 7 * y + j = 0

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 32 * x

/-- Theorem stating that tangent_j is the unique value making the line tangent to the parabola -/
theorem tangent_line_to_parabola :
  ∃! j : ℝ, ∀ x y : ℝ, line x y j ∧ parabola x y → 
    (∃! p : ℝ × ℝ, line p.1 p.2 j ∧ parabola p.1 p.2) ∧ j = tangent_j :=
by sorry

end tangent_line_to_parabola_l1541_154199


namespace objects_per_hour_l1541_154166

/-- The number of objects one person can make in an hour -/
def n : ℕ := 12

/-- The time Ann works in hours -/
def ann_time : ℚ := 1

/-- The time Bob works in hours -/
def bob_time : ℚ := 2/3

/-- The time Cody works in hours -/
def cody_time : ℚ := 1/3

/-- The time Deb works in hours -/
def deb_time : ℚ := 1/3

/-- The total number of objects made -/
def total_objects : ℕ := 28

theorem objects_per_hour :
  n * (ann_time + bob_time + cody_time + deb_time) = total_objects := by
  sorry

end objects_per_hour_l1541_154166


namespace circular_matrix_determinant_properties_l1541_154151

def circularMatrix (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, b, c],
    ![c, a, b],
    ![b, c, a]]

theorem circular_matrix_determinant_properties :
  (∃ (S : Set (ℚ × ℚ × ℚ)), Set.Infinite S ∧
    ∀ (abc : ℚ × ℚ × ℚ), abc ∈ S →
      Matrix.det (circularMatrix abc.1 abc.2.1 abc.2.2) = 1) ∧
  (∃ (T : Set (ℤ × ℤ × ℤ)), Set.Finite T ∧
    ∀ (abc : ℤ × ℤ × ℤ), Matrix.det (circularMatrix ↑abc.1 ↑abc.2.1 ↑abc.2.2) = 1 →
      abc ∈ T) := by
  sorry

end circular_matrix_determinant_properties_l1541_154151


namespace product_zero_implies_factor_zero_l1541_154170

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  contrapose!
  intro h
  sorry

end product_zero_implies_factor_zero_l1541_154170


namespace min_additional_wins_correct_l1541_154145

/-- The minimum number of additional wins required to achieve a 90% winning percentage -/
def min_additional_wins : ℕ := 26

/-- The initial number of games played -/
def initial_games : ℕ := 4

/-- The initial number of games won -/
def initial_wins : ℕ := 1

/-- The target winning percentage -/
def target_percentage : ℚ := 9/10

theorem min_additional_wins_correct :
  ∀ n : ℕ, 
    (n ≥ min_additional_wins) ↔ 
    ((initial_wins + n : ℚ) / (initial_games + n)) ≥ target_percentage :=
sorry

end min_additional_wins_correct_l1541_154145


namespace reciprocal_sum_theorem_l1541_154134

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end reciprocal_sum_theorem_l1541_154134


namespace linear_function_shift_l1541_154100

/-- 
A linear function y = -2x + b is shifted 3 units upwards.
This theorem proves that if the shifted function passes through the point (2, 0),
then b = 1.
-/
theorem linear_function_shift (b : ℝ) : 
  (∀ x y : ℝ, y = -2 * x + b + 3 → (x = 2 ∧ y = 0) → b = 1) := by
sorry

end linear_function_shift_l1541_154100


namespace meeting_size_l1541_154182

/-- Represents the number of people attending the meeting -/
def n : ℕ → ℕ := λ k => 12 * k

/-- Represents the number of handshakes each person makes -/
def handshakes : ℕ → ℕ := λ k => 3 * k + 6

/-- Represents the number of mutual handshakes between any two people -/
def mutual_handshakes : ℕ → ℚ := λ k => 
  ((3 * k + 6) * (3 * k + 5)) / (12 * k - 1)

theorem meeting_size : 
  ∃ k : ℕ, k > 0 ∧ 
    (∀ i j : Fin (n k), i ≠ j → 
      (mutual_handshakes k).num % (mutual_handshakes k).den = 0) ∧
    n k = 36 := by
  sorry

end meeting_size_l1541_154182


namespace complement_union_problem_l1541_154163

def U : Finset Nat := {1,2,3,4,5,6,7,8}

theorem complement_union_problem (A B : Finset Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {3})
  (h4 : (U \ B) ∩ A = {1,2})
  (h5 : (U \ A) ∩ B = {4,5}) :
  U \ (A ∪ B) = {6,7,8} := by
  sorry

end complement_union_problem_l1541_154163


namespace square_of_98_l1541_154132

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end square_of_98_l1541_154132


namespace triangle_ratio_l1541_154190

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

#check triangle_ratio

end triangle_ratio_l1541_154190


namespace at_least_one_not_in_area_l1541_154129

theorem at_least_one_not_in_area (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (∃ x, x = p ∨ x = q) ∧ (x → False) :=
by sorry

end at_least_one_not_in_area_l1541_154129


namespace spade_calculation_l1541_154116

def spade (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem spade_calculation : spade 2 (spade 3 (spade 1 2)) = 4 := by
  sorry

end spade_calculation_l1541_154116


namespace lana_extra_flowers_l1541_154144

/-- The number of extra flowers Lana picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem: Lana picked 3 extra flowers -/
theorem lana_extra_flowers :
  extra_flowers 36 37 70 = 3 := by
  sorry

end lana_extra_flowers_l1541_154144


namespace alphabet_letters_l1541_154152

theorem alphabet_letters (total : ℕ) (both : ℕ) (line_only : ℕ) (h1 : total = 60) (h2 : both = 20) (h3 : line_only = 36) :
  total = both + line_only + (total - (both + line_only)) →
  total - (both + line_only) = 24 :=
by sorry

end alphabet_letters_l1541_154152


namespace collinear_points_sum_l1541_154115

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

-- State the theorem
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, b, a) (b, 2, a) (b, a, 3) → a + b = 4 := by
  sorry

end collinear_points_sum_l1541_154115


namespace equation_solution_l1541_154117

theorem equation_solution : 
  ∀ x y z : ℕ, 2^x + 3^y + 7 = z! ↔ (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

#check equation_solution

end equation_solution_l1541_154117
