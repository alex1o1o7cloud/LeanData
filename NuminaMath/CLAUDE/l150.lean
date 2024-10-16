import Mathlib

namespace NUMINAMATH_CALUDE_coal_transport_trucks_l150_15049

/-- The number of trucks needed to transport a given amount of coal -/
def trucks_needed (total_coal : ℕ) (truck_capacity : ℕ) : ℕ :=
  (total_coal + truck_capacity - 1) / truck_capacity

/-- Proof that 19 trucks are needed to transport 47,500 kg of coal when each truck can carry 2,500 kg -/
theorem coal_transport_trucks : trucks_needed 47500 2500 = 19 := by
  sorry

end NUMINAMATH_CALUDE_coal_transport_trucks_l150_15049


namespace NUMINAMATH_CALUDE_circle_max_min_linear_function_l150_15074

theorem circle_max_min_linear_function :
  ∀ x y : ℝ, x^2 + y^2 = 16*x + 8*y + 20 →
  (∀ x' y' : ℝ, x'^2 + y'^2 = 16*x' + 8*y' + 20 → 4*x' + 3*y' ≤ 116) ∧
  (∀ x' y' : ℝ, x'^2 + y'^2 = 16*x' + 8*y' + 20 → 4*x' + 3*y' ≥ -64) ∧
  (∃ x₁ y₁ : ℝ, x₁^2 + y₁^2 = 16*x₁ + 8*y₁ + 20 ∧ 4*x₁ + 3*y₁ = 116) ∧
  (∃ x₂ y₂ : ℝ, x₂^2 + y₂^2 = 16*x₂ + 8*y₂ + 20 ∧ 4*x₂ + 3*y₂ = -64) :=
by sorry


end NUMINAMATH_CALUDE_circle_max_min_linear_function_l150_15074


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_four_l150_15052

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of 4 is -4 -/
theorem opposite_of_four : opposite 4 = -4 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_four_l150_15052


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l150_15090

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l150_15090


namespace NUMINAMATH_CALUDE_geometric_series_product_l150_15075

theorem geometric_series_product (y : ℝ) : 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_product_l150_15075


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_one_l150_15094

-- Define the complex number z
def z : ℂ := (3 - Complex.I) * (2 + Complex.I)

-- Theorem statement
theorem imaginary_part_of_z_is_one : Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_one_l150_15094


namespace NUMINAMATH_CALUDE_square_division_impossibility_l150_15050

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane --/
structure Square where
  side : ℝ
  center : Point

/-- Represents a division of a square --/
structure SquareDivision where
  square : Square
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a square --/
def is_inside (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) < s.side / 2 ∧ abs (p.y - s.center.y) < s.side / 2

/-- The theorem stating the impossibility of the division --/
theorem square_division_impossibility (s : Square) : 
  ¬ ∃ (d : SquareDivision), 
    (is_inside d.point1 s) ∧ 
    (is_inside d.point2 s) ∧ 
    (∃ (areas : List ℝ), areas.length = 9 ∧ (∀ a ∈ areas, a > 0) ∧ areas.sum = s.side ^ 2) :=
sorry

end NUMINAMATH_CALUDE_square_division_impossibility_l150_15050


namespace NUMINAMATH_CALUDE_power_function_through_point_l150_15064

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x > 0, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 → ∀ x > 0, f x = x ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l150_15064


namespace NUMINAMATH_CALUDE_complex_expressions_calculation_l150_15004

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define square root of 3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

theorem complex_expressions_calculation :
  -- Expression 1
  ((1 + 2*i)^2 + 3*(1 - i)) / (2 + i) = 1/5 + 2/5*i ∧
  -- Expression 2
  (1 - i) / (1 + i)^2 + (1 + i) / (1 - i)^2 = -1 ∧
  -- Expression 3
  (1 - sqrt3*i) / (sqrt3 + i)^2 = -1/4 - (sqrt3/4)*i :=
by sorry

end NUMINAMATH_CALUDE_complex_expressions_calculation_l150_15004


namespace NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l150_15005

theorem negation_of_exists_square_nonpositive :
  (¬ ∃ a : ℝ, a^2 ≤ 0) ↔ (∀ a : ℝ, a^2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l150_15005


namespace NUMINAMATH_CALUDE_furniture_dealer_tables_l150_15054

/-- The number of four-legged tables -/
def F : ℕ := 16

/-- The number of three-legged tables -/
def T : ℕ := (124 - 4 * F) / 3

/-- The total number of tables -/
def total_tables : ℕ := F + T

/-- Theorem stating that the total number of tables is 36 -/
theorem furniture_dealer_tables : total_tables = 36 := by
  sorry

end NUMINAMATH_CALUDE_furniture_dealer_tables_l150_15054


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l150_15085

theorem fraction_sum_equality : 
  (3 : ℚ) / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l150_15085


namespace NUMINAMATH_CALUDE_point_distance_theorem_l150_15057

theorem point_distance_theorem (x y : ℝ) (h1 : x > 1) :
  y = 12 ∧ (x - 1)^2 + (y - 6)^2 = 10^2 →
  x^2 + y^2 = 15^2 := by
sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l150_15057


namespace NUMINAMATH_CALUDE_five_cuts_sixteen_pieces_l150_15055

/-- The number of pieces obtained by cutting a cake n times -/
def cakePieces (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem: The number of pieces obtained by cutting a cake 5 times is 16 -/
theorem five_cuts_sixteen_pieces : cakePieces 5 = 16 := by
  sorry

#eval cakePieces 5  -- This will evaluate to 16

end NUMINAMATH_CALUDE_five_cuts_sixteen_pieces_l150_15055


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_congruences_l150_15065

theorem smallest_n_satisfying_congruences : 
  ∃ n : ℕ, n > 20 ∧ n % 6 = 4 ∧ n % 7 = 5 ∧ 
  ∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 5 → n ≤ m :=
by
  use 40
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_congruences_l150_15065


namespace NUMINAMATH_CALUDE_max_m_is_6_min_value_is_9_min_value_achievable_l150_15019

-- Define the condition that |x+2|+|x-4|-m≥0 for all real x
def condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Theorem 1: The maximum value of m is 6
theorem max_m_is_6 (h : condition m) : m ≤ 6 :=
sorry

-- Define the constraint equation for a and b
def constraint (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 4 / (a + 5*b) + 1 / (3*a + 2*b) = 1

-- Theorem 2: The minimum value of 4a+7b is 9
theorem min_value_is_9 (a b : ℝ) (h : constraint a b) : 
  4*a + 7*b ≥ 9 :=
sorry

-- Theorem 3: The minimum value 9 is achievable
theorem min_value_achievable : 
  ∃ a b : ℝ, constraint a b ∧ 4*a + 7*b = 9 :=
sorry

end NUMINAMATH_CALUDE_max_m_is_6_min_value_is_9_min_value_achievable_l150_15019


namespace NUMINAMATH_CALUDE_unique_twin_prime_sum_prime_power_l150_15046

-- Define twin primes
def is_twin_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (p + 2)

-- Define prime power
def is_prime_power (n : ℕ) : Prop :=
  ∃ (q k : ℕ), Nat.Prime q ∧ k > 0 ∧ n = q^k

-- Theorem statement
theorem unique_twin_prime_sum_prime_power :
  ∃! (p : ℕ), is_twin_prime p ∧ is_prime_power (p + (p + 2)) :=
sorry

end NUMINAMATH_CALUDE_unique_twin_prime_sum_prime_power_l150_15046


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l150_15099

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 40)
  (area2 : w * h = 10)
  (area3 : l * h = 8) :
  l * w * h = 40 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l150_15099


namespace NUMINAMATH_CALUDE_composition_equality_l150_15059

/-- Given two functions f and g, and a real number a, prove that f(g(a)) = 4 implies a = 1 -/
theorem composition_equality (f g : ℝ → ℝ) (a : ℝ) 
  (hf : f = fun x ↦ (2 * x / 3) + 2)
  (hg : g = fun x ↦ 5 - 2 * x)
  (h : f (g a) = 4) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_composition_equality_l150_15059


namespace NUMINAMATH_CALUDE_equation_equality_l150_15006

theorem equation_equality (a : ℝ) (h : a ≠ 0) :
  ((1 / a) / ((1 / a) * (1 / a)) - 1 / a) / (1 / a) = (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l150_15006


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l150_15038

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → (∀ a b : ℕ, a^2 - b^2 = 221 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 24421 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l150_15038


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l150_15096

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem parallel_line_plane_not_imply_parallel_lines 
  (l m : Line) (α : Plane) : 
  ¬(parallel_line_plane l α ∧ contains α m → parallel_lines l m) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_not_imply_parallel_lines_l150_15096


namespace NUMINAMATH_CALUDE_minimum_value_and_max_when_half_l150_15082

noncomputable def f (a x : ℝ) : ℝ := 1 - 2*a - 2*a*Real.cos x - 2*(Real.sin x)^2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a ≤ 2 then -a^2/2 - 2*a - 1
  else 1 - 4*a

theorem minimum_value_and_max_when_half (a : ℝ) :
  (∀ x, f a x ≥ g a) ∧
  (g a = 1/2 → a = -1 ∧ ∃ x, f (-1) x = 5 ∧ ∀ y, f (-1) y ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_and_max_when_half_l150_15082


namespace NUMINAMATH_CALUDE_alcohol_concentration_problem_l150_15062

/-- Proves that the initial concentration of alcohol in the second vessel is 55% --/
theorem alcohol_concentration_problem (vessel1_capacity : ℝ) (vessel1_concentration : ℝ)
  (vessel2_capacity : ℝ) (total_liquid : ℝ) (final_vessel_capacity : ℝ)
  (final_concentration : ℝ) :
  vessel1_capacity = 2 →
  vessel1_concentration = 20 →
  vessel2_capacity = 6 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  final_concentration = 37 →
  ∃ vessel2_concentration : ℝ,
    vessel2_concentration = 55 ∧
    vessel1_capacity * (vessel1_concentration / 100) +
    vessel2_capacity * (vessel2_concentration / 100) =
    final_vessel_capacity * (final_concentration / 100) :=
by sorry


end NUMINAMATH_CALUDE_alcohol_concentration_problem_l150_15062


namespace NUMINAMATH_CALUDE_truck_toll_theorem_l150_15013

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  3.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem :
  let totalWheels := 18
  let frontAxleWheels := 2
  let otherAxleWheels := 4
  let axles := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  toll axles = 5.00 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_theorem_l150_15013


namespace NUMINAMATH_CALUDE_cake_area_increase_percentage_cake_area_increase_percentage_approx_l150_15032

/-- The percent increase in area of a circular cake when its diameter increases from 8 inches to 10 inches -/
theorem cake_area_increase_percentage : ℝ := by
  -- Define the initial and final diameters
  let initial_diameter : ℝ := 8
  let final_diameter : ℝ := 10
  
  -- Define the function to calculate the area of a circular cake given its diameter
  let cake_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2
  
  -- Calculate the initial and final areas
  let initial_area := cake_area initial_diameter
  let final_area := cake_area final_diameter
  
  -- Calculate the percent increase
  let percent_increase := (final_area - initial_area) / initial_area * 100
  
  -- Prove that the percent increase is 56.25%
  sorry

/-- The result of cake_area_increase_percentage is approximately 56.25 -/
theorem cake_area_increase_percentage_approx :
  |cake_area_increase_percentage - 56.25| < 0.01 := by sorry

end NUMINAMATH_CALUDE_cake_area_increase_percentage_cake_area_increase_percentage_approx_l150_15032


namespace NUMINAMATH_CALUDE_no_solution_exists_l150_15027

theorem no_solution_exists : ¬∃ n : ℤ,
  50 ≤ n ∧ n ≤ 150 ∧
  8 ∣ n ∧
  n % 10 = 6 ∧
  n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l150_15027


namespace NUMINAMATH_CALUDE_chicken_theorem_l150_15067

/-- The number of chickens Colten has -/
def colten_chickens : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar_chickens : ℕ := 3 * colten_chickens - 4

/-- The number of chickens Quentin has -/
def quentin_chickens : ℕ := 2 * skylar_chickens + 25

theorem chicken_theorem : 
  colten_chickens + skylar_chickens + quentin_chickens = 383 :=
by sorry

end NUMINAMATH_CALUDE_chicken_theorem_l150_15067


namespace NUMINAMATH_CALUDE_largest_number_in_set_l150_15009

def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

theorem largest_number_in_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : (a + b + c) / 3 = 6)
  (h_median : b = 6)
  (h_smallest : a = 2) : 
  c = 10 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l150_15009


namespace NUMINAMATH_CALUDE_steven_jill_peach_difference_l150_15078

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 6

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 18

/-- Theorem: Steven has 13 more peaches than Jill -/
theorem steven_jill_peach_difference : steven_peaches - jill_peaches = 13 := by
  sorry

end NUMINAMATH_CALUDE_steven_jill_peach_difference_l150_15078


namespace NUMINAMATH_CALUDE_dot_product_parallel_relationship_l150_15095

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2) ∨ b = (k * a.1, k * a.2)

/-- Dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_parallel_relationship (a b : ℝ × ℝ) :
  ¬(∀ a b : ℝ × ℝ, dot_product a b > 0 → parallel a b) ∧
  ¬(∀ a b : ℝ × ℝ, parallel a b → dot_product a b > 0) :=
by sorry

end NUMINAMATH_CALUDE_dot_product_parallel_relationship_l150_15095


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l150_15097

theorem cubic_sum_theorem (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (product_sum_condition : a * b + a * c + b * c = -4)
  (product_condition : a * b * c = -4) : 
  a^3 + b^3 + c^3 = 1 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l150_15097


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l150_15063

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 50*x + 575 ≤ 25) ↔ (25 - 5*Real.sqrt 3 ≤ x ∧ x ≤ 25 + 5*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l150_15063


namespace NUMINAMATH_CALUDE_x_range_when_ln_x_negative_l150_15012

theorem x_range_when_ln_x_negative (x : ℝ) (h : Real.log x < 0) : 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_when_ln_x_negative_l150_15012


namespace NUMINAMATH_CALUDE_banana_tree_problem_l150_15016

theorem banana_tree_problem (bananas_left : ℕ) (bananas_eaten : ℕ) : 
  bananas_left = 100 →
  bananas_eaten = 70 →
  (∃ (initial_bananas : ℕ), initial_bananas = bananas_left + bananas_eaten + 2 * bananas_eaten ∧ initial_bananas = 310) :=
by sorry

end NUMINAMATH_CALUDE_banana_tree_problem_l150_15016


namespace NUMINAMATH_CALUDE_total_grains_in_gray_parts_l150_15080

/-- Represents a circle with grains -/
structure GrainCircle where
  total : ℕ
  intersection : ℕ

/-- Calculates the number of grains in the non-intersecting part of a circle -/
def nonIntersectingGrains (circle : GrainCircle) : ℕ :=
  circle.total - circle.intersection

/-- The main theorem -/
theorem total_grains_in_gray_parts 
  (circle1 circle2 : GrainCircle)
  (h1 : circle1.total = 87)
  (h2 : circle2.total = 110)
  (h3 : circle1.intersection = 68)
  (h4 : circle2.intersection = 68) :
  nonIntersectingGrains circle1 + nonIntersectingGrains circle2 = 61 := by
  sorry

#eval nonIntersectingGrains { total := 87, intersection := 68 } +
      nonIntersectingGrains { total := 110, intersection := 68 }

end NUMINAMATH_CALUDE_total_grains_in_gray_parts_l150_15080


namespace NUMINAMATH_CALUDE_shells_equation_initial_shells_value_l150_15056

/-- The number of shells Lucy initially put in her bucket -/
def initial_shells : ℕ := sorry

/-- The number of additional shells Lucy found -/
def additional_shells : ℕ := 21

/-- The total number of shells Lucy has now -/
def total_shells : ℕ := 89

/-- Theorem stating that the initial number of shells plus the additional shells equals the total shells -/
theorem shells_equation : initial_shells + additional_shells = total_shells := by sorry

/-- Theorem proving that the initial number of shells is 68 -/
theorem initial_shells_value : initial_shells = 68 := by sorry

end NUMINAMATH_CALUDE_shells_equation_initial_shells_value_l150_15056


namespace NUMINAMATH_CALUDE_three_person_arrangement_l150_15086

def number_of_arrangements (n : ℕ) : ℕ := Nat.factorial n

theorem three_person_arrangement :
  number_of_arrangements 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_person_arrangement_l150_15086


namespace NUMINAMATH_CALUDE_two_digit_congruent_to_three_mod_four_count_l150_15051

theorem two_digit_congruent_to_three_mod_four_count : 
  (Finset.filter (fun n => n ≥ 10 ∧ n ≤ 99 ∧ n % 4 = 3) (Finset.range 100)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_congruent_to_three_mod_four_count_l150_15051


namespace NUMINAMATH_CALUDE_candy_tins_count_l150_15014

/-- The number of candy tins given the total number of strawberry-flavored candies
    and the number of strawberry-flavored candies per tin. -/
def number_of_candy_tins (total_strawberry_candies : ℕ) (strawberry_candies_per_tin : ℕ) : ℕ :=
  total_strawberry_candies / strawberry_candies_per_tin

/-- Theorem stating that the number of candy tins is 9 given the problem conditions. -/
theorem candy_tins_count : number_of_candy_tins 27 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_tins_count_l150_15014


namespace NUMINAMATH_CALUDE_lizzy_candy_spending_l150_15039

/-- The amount of money Lizzy spent on candy --/
def candy_spent : ℕ := sorry

/-- The amount of money Lizzy received from her mother --/
def mother_gave : ℕ := 80

/-- The amount of money Lizzy received from her father --/
def father_gave : ℕ := 40

/-- The amount of money Lizzy received from her uncle --/
def uncle_gave : ℕ := 70

/-- The total amount of money Lizzy has now --/
def current_total : ℕ := 140

theorem lizzy_candy_spending :
  candy_spent = 50 ∧
  current_total = mother_gave + father_gave - candy_spent + uncle_gave :=
sorry

end NUMINAMATH_CALUDE_lizzy_candy_spending_l150_15039


namespace NUMINAMATH_CALUDE_same_number_on_cards_l150_15068

theorem same_number_on_cards (n : ℕ) (cards : Fin n → ℕ) : 
  (∀ i, cards i ∈ Finset.range n) →
  (∀ s : Finset (Fin n), (s.sum cards) % (n + 1) ≠ 0) →
  ∀ i j, cards i = cards j :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_cards_l150_15068


namespace NUMINAMATH_CALUDE_monotone_increasing_range_a_inequality_for_m_n_l150_15011

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem monotone_increasing_range_a :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → Monotone (f a)) → a ≤ 2 := by sorry

theorem inequality_for_m_n :
  ∀ m n : ℝ, m ≠ n → (m - n) / (Real.log m - Real.log n) < (m + n) / 2 := by sorry

end NUMINAMATH_CALUDE_monotone_increasing_range_a_inequality_for_m_n_l150_15011


namespace NUMINAMATH_CALUDE_car_travel_problem_l150_15069

/-- Represents a car's travel information -/
structure CarTravel where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem statement -/
theorem car_travel_problem 
  (p : CarTravel) 
  (q : CarTravel) 
  (h1 : p.time = 3) 
  (h2 : p.speed = 60) 
  (h3 : q.speed = 3 * p.speed) 
  (h4 : q.distance = p.distance / 2) 
  (h5 : p.distance = p.speed * p.time) 
  (h6 : q.distance = q.speed * q.time) : 
  q.time = 0.5 := by
sorry

end NUMINAMATH_CALUDE_car_travel_problem_l150_15069


namespace NUMINAMATH_CALUDE_scientific_notation_of_216000_l150_15092

theorem scientific_notation_of_216000 :
  (216000 : ℝ) = 2.16 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_216000_l150_15092


namespace NUMINAMATH_CALUDE_fencing_cost_distribution_impossible_equal_distribution_impossible_l150_15026

/-- Represents the dimensions of the cottage settlement. -/
structure Settlement where
  n : ℕ
  m : ℕ

/-- Calculates the total cost of fencing for the entire settlement. -/
def totalFencingCost (s : Settlement) : ℕ :=
  10000 * (2 * s.n * s.m + s.n + s.m - 4)

/-- Calculates the sum of costs if equal numbers of residents spent 0, 10000, 30000, 40000,
    and the rest spent 20000 rubles. -/
def proposedCostSum (s : Settlement) : ℕ :=
  100000 + 20000 * (s.n * s.m - 4)

/-- Theorem stating that the proposed cost distribution is impossible. -/
theorem fencing_cost_distribution_impossible (s : Settlement) :
  totalFencingCost s ≠ proposedCostSum s :=
sorry

/-- Theorem stating that it's impossible to have equal numbers of residents spending
    0, 10000, 30000, 40000 rubles with the rest spending 20000 rubles. -/
theorem equal_distribution_impossible (s : Settlement) :
  ¬ ∃ (k : ℕ), k > 0 ∧ 
    s.n * s.m = 4 * k + (s.n * s.m - 4 * k) ∧
    totalFencingCost s = k * (0 + 10000 + 30000 + 40000) + (s.n * s.m - 4 * k) * 20000 :=
sorry

end NUMINAMATH_CALUDE_fencing_cost_distribution_impossible_equal_distribution_impossible_l150_15026


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l150_15058

theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ),
    (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
    a 1 = 4 →                            -- First term
    a 10 = 93312 →                       -- Last term
    a 7 = 186624 :=                      -- Seventh term
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l150_15058


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l150_15020

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b ≥ 2) ∧ (∀ ε > 0, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1 / a' + 1 / b' < 2 + ε) :=
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l150_15020


namespace NUMINAMATH_CALUDE_solution_set_equivalence_k_range_l150_15022

noncomputable section

-- Define the function f
def f (k x : ℝ) : ℝ := (k * x) / (x^2 + 3 * k)

-- Define the conditions
variable (k : ℝ)
variable (h_k_pos : k > 0)

-- Part 1
theorem solution_set_equivalence :
  (∃ m : ℝ, ∀ x : ℝ, f k x > m ↔ x < -3 ∨ x > -2) →
  ∃ m : ℝ, ∀ x : ℝ, 5 * m * x^2 + (k / 2) * x + 3 > 0 ↔ -1 < x ∧ x < 3/2 :=
sorry

-- Part 2
theorem k_range :
  (∃ x : ℝ, x > 3 ∧ f k x > 1) →
  k > 12 :=
sorry

end

end NUMINAMATH_CALUDE_solution_set_equivalence_k_range_l150_15022


namespace NUMINAMATH_CALUDE_netball_points_calculation_l150_15070

theorem netball_points_calculation 
  (w d : ℕ) 
  (h1 : w > d) 
  (h2 : 7 * w + 3 * d = 44) : 
  5 * w + 2 * d = 31 := by
sorry

end NUMINAMATH_CALUDE_netball_points_calculation_l150_15070


namespace NUMINAMATH_CALUDE_book_selection_ways_l150_15028

theorem book_selection_ways (math_books physics_books chemistry_books : ℕ) 
  (h_math : math_books = 5)
  (h_physics : physics_books = 4)
  (h_chemistry : chemistry_books = 5) :
  math_books + physics_books + chemistry_books = 14 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_ways_l150_15028


namespace NUMINAMATH_CALUDE_cube_inequality_negation_l150_15034

theorem cube_inequality_negation (x y : ℝ) (h : x > y) : 
  ¬(x^3 > y^3) ↔ x^3 ≤ y^3 := by
sorry

end NUMINAMATH_CALUDE_cube_inequality_negation_l150_15034


namespace NUMINAMATH_CALUDE_range_of_a_l150_15035

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x, x^2 - 2*x + a > 0

def q (a : ℝ) : Prop := ∀ x y, x < y → (a - 1)^x < (a - 1)^y

-- Define the theorem
theorem range_of_a :
  (∃ a, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a, 1 < a ∧ a ≤ 2) ∧ (∀ a, (1 < a ∧ a ≤ 2) → (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l150_15035


namespace NUMINAMATH_CALUDE_bacteria_growth_calculation_l150_15030

/-- Given an original bacteria count and a current bacteria count, 
    calculate the increase in bacteria. -/
def bacteria_increase (original current : ℕ) : ℕ :=
  current - original

/-- Theorem stating that the increase in bacteria from 600 to 8917 is 8317. -/
theorem bacteria_growth_calculation :
  bacteria_increase 600 8917 = 8317 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_calculation_l150_15030


namespace NUMINAMATH_CALUDE_age_difference_l150_15003

theorem age_difference (patrick michael monica : ℕ) : 
  patrick * 5 = michael * 3 →
  michael * 5 = monica * 3 →
  patrick + michael + monica = 147 →
  monica - patrick = 48 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l150_15003


namespace NUMINAMATH_CALUDE_range_where_g_geq_f_max_value_g_minus_f_l150_15018

-- Define the functions f and g
def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := -x^2 + 6*x - 5

-- Theorem for the range of x where g(x) ≥ f(x)
theorem range_where_g_geq_f :
  {x : ℝ | g x ≥ f x} = Set.Ici 1 ∩ Set.Iic 4 :=
sorry

-- Theorem for the maximum value of g(x) - f(x)
theorem max_value_g_minus_f :
  ∃ (x : ℝ), ∀ (y : ℝ), g y - f y ≤ g x - f x ∧ g x - f x = 9/4 :=
sorry

end NUMINAMATH_CALUDE_range_where_g_geq_f_max_value_g_minus_f_l150_15018


namespace NUMINAMATH_CALUDE_sequence_problem_l150_15060

/-- An arithmetic sequence where no term is 0 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m ∧ ∀ k, a k ≠ 0

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, b (n + 1) / b n = b (m + 1) / b m

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : a 5 - a 7 ^ 2 + a 9 = 0)
    (h_b7 : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l150_15060


namespace NUMINAMATH_CALUDE_units_digit_17_35_l150_15089

theorem units_digit_17_35 : 17^35 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_35_l150_15089


namespace NUMINAMATH_CALUDE_smallest_with_9_odd_18_even_factors_l150_15007

/-- The number of odd factors of an integer -/
def num_odd_factors (n : ℕ) : ℕ := sorry

/-- The number of even factors of an integer -/
def num_even_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest integer with exactly 9 odd factors and 18 even factors is 900 -/
theorem smallest_with_9_odd_18_even_factors :
  ∀ n : ℕ, num_odd_factors n = 9 ∧ num_even_factors n = 18 → n ≥ 900 ∧
  ∃ m : ℕ, m = 900 ∧ num_odd_factors m = 9 ∧ num_even_factors m = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_9_odd_18_even_factors_l150_15007


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l150_15091

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 7 * a 11 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 20 / a 10 = 2/3 ∨ a 20 / a 10 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l150_15091


namespace NUMINAMATH_CALUDE_quadratic_minimum_l150_15037

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 5 ≥ -4 ∧ ∃ y : ℝ, y^2 + 6*y + 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l150_15037


namespace NUMINAMATH_CALUDE_harvest_duration_proof_l150_15081

/-- Calculates the number of weeks the harvest lasted. -/
def harvest_duration (weekly_earnings : ℕ) (total_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Proves that the harvest lasted 89 weeks given the conditions. -/
theorem harvest_duration_proof (weekly_earnings total_earnings : ℕ) 
  (h1 : weekly_earnings = 2)
  (h2 : total_earnings = 178) :
  harvest_duration weekly_earnings total_earnings = 89 := by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_proof_l150_15081


namespace NUMINAMATH_CALUDE_parallelogram_area_l150_15043

/-- The area of a parallelogram with base 12 and height 6 is 72 -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 12 → 
  height = 6 → 
  area = base * height → 
  area = 72 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l150_15043


namespace NUMINAMATH_CALUDE_mileage_difference_l150_15079

/-- The difference between advertised and actual mileage -/
theorem mileage_difference (advertised_mpg : ℝ) (tank_capacity : ℝ) (miles_driven : ℝ) :
  advertised_mpg = 35 →
  tank_capacity = 12 →
  miles_driven = 372 →
  advertised_mpg - (miles_driven / tank_capacity) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mileage_difference_l150_15079


namespace NUMINAMATH_CALUDE_oxygen_weight_in_N2O_l150_15071

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of nitrogen atoms in N2O -/
def num_N : ℕ := 2

/-- The number of oxygen atoms in N2O -/
def num_O : ℕ := 1

/-- The molecular weight of the oxygen part in N2O -/
def molecular_weight_O_part : ℝ := num_O * atomic_weight_O

theorem oxygen_weight_in_N2O : 
  molecular_weight_O_part = 16.00 := by sorry

end NUMINAMATH_CALUDE_oxygen_weight_in_N2O_l150_15071


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l150_15040

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

-- State the theorem
theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) →
  (a < -1 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l150_15040


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l150_15033

theorem min_value_x_plus_four_over_x (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l150_15033


namespace NUMINAMATH_CALUDE_larger_triangle_perimeter_l150_15029

/-- Given an isosceles triangle with side lengths 7, 7, and 12, and a similar triangle
    with longest side 36, the perimeter of the larger triangle is 78. -/
theorem larger_triangle_perimeter (a b c : ℝ) (d : ℝ) : 
  a = 7 ∧ b = 7 ∧ c = 12 ∧ d = 36 ∧ 
  (a = b) ∧ (c ≥ a) ∧ (c ≥ b) ∧
  (d / c = 36 / 12) →
  d + (d * a / c) + (d * b / c) = 78 := by
  sorry


end NUMINAMATH_CALUDE_larger_triangle_perimeter_l150_15029


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l150_15098

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l150_15098


namespace NUMINAMATH_CALUDE_pi_half_irrational_l150_15061

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l150_15061


namespace NUMINAMATH_CALUDE_prob_different_grades_is_four_fifths_l150_15021

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Calculates the probability of selecting two students from different grades --/
def probabilityDifferentGrades (dist : GradeDistribution) : ℚ :=
  4/5

/-- Theorem stating that the probability of selecting two students from different grades is 4/5 --/
theorem prob_different_grades_is_four_fifths (dist : GradeDistribution) 
  (h1 : dist.grade10 = 180)
  (h2 : dist.grade11 = 180)
  (h3 : dist.grade12 = 90) :
  probabilityDifferentGrades dist = 4/5 := by
  sorry

#check prob_different_grades_is_four_fifths

end NUMINAMATH_CALUDE_prob_different_grades_is_four_fifths_l150_15021


namespace NUMINAMATH_CALUDE_remainder_theorem_l150_15041

/-- The polynomial f(x) = x^4 - 6x^3 + 12x^2 + 20x - 8 -/
def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 12*x^2 + 20*x - 8

/-- The theorem stating that the remainder when f(x) is divided by (x-4) is 136 -/
theorem remainder_theorem : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = (x - 4) * q x + 136 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l150_15041


namespace NUMINAMATH_CALUDE_integer_product_100000_l150_15093

theorem integer_product_100000 : ∃ (a b : ℤ), 
  a * b = 100000 ∧ 
  a % 10 ≠ 0 ∧ 
  b % 10 ≠ 0 ∧ 
  (a = 32 ∨ b = 32) :=
by sorry

end NUMINAMATH_CALUDE_integer_product_100000_l150_15093


namespace NUMINAMATH_CALUDE_exam_average_l150_15053

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 115)
  (h3 : passed_avg = 39)
  (h4 : failed_avg = 15) :
  (passed_boys * passed_avg + (total_boys - passed_boys) * failed_avg) / total_boys = 38 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l150_15053


namespace NUMINAMATH_CALUDE_work_completion_time_l150_15087

/-- The time taken to complete a work when three workers with given efficiencies work together -/
theorem work_completion_time 
  (total_work : ℝ) 
  (efficiency_x efficiency_y efficiency_z : ℝ) 
  (hx : efficiency_x = 1 / 20)
  (hy : efficiency_y = 3 / 80)
  (hz : efficiency_z = 3 / 40)
  (h_total : total_work = 1) :
  (total_work / (efficiency_x + efficiency_y + efficiency_z)) = 80 / 13 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l150_15087


namespace NUMINAMATH_CALUDE_correct_bottles_per_pack_l150_15008

/-- The number of bottles in each pack of soda -/
def bottles_per_pack : ℕ := 6

/-- The number of packs Rebecca bought -/
def packs_bought : ℕ := 3

/-- The number of bottles Rebecca drinks per day -/
def bottles_per_day : ℚ := 1/2

/-- The number of days in the given period -/
def days : ℕ := 28

/-- The number of bottles remaining after the given period -/
def bottles_remaining : ℕ := 4

/-- Theorem stating that the number of bottles in each pack is correct -/
theorem correct_bottles_per_pack :
  bottles_per_pack * packs_bought - (bottles_per_day * days).floor = bottles_remaining :=
sorry

end NUMINAMATH_CALUDE_correct_bottles_per_pack_l150_15008


namespace NUMINAMATH_CALUDE_buffer_solution_calculation_l150_15024

theorem buffer_solution_calculation (initial_volume_A initial_volume_B total_volume_needed : ℚ) :
  initial_volume_A = 0.05 →
  initial_volume_B = 0.03 →
  initial_volume_A + initial_volume_B = 0.08 →
  total_volume_needed = 0.64 →
  (total_volume_needed * (initial_volume_B / (initial_volume_A + initial_volume_B))) = 0.24 := by
sorry

end NUMINAMATH_CALUDE_buffer_solution_calculation_l150_15024


namespace NUMINAMATH_CALUDE_percentage_decrease_after_increase_l150_15025

theorem percentage_decrease_after_increase (x : ℝ) (hx : x > 0) :
  let y := x * 1.6
  y * (1 - 0.375) = x :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_after_increase_l150_15025


namespace NUMINAMATH_CALUDE_power_mod_six_l150_15084

theorem power_mod_six : 5^2013 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_six_l150_15084


namespace NUMINAMATH_CALUDE_lemonade_amount_l150_15001

/-- Represents the recipe for a cold drink -/
structure DrinkRecipe where
  tea : Rat
  lemonade : Rat

/-- Represents the total amount of drink in the pitcher -/
def totalAmount : Rat := 18

/-- The recipe for one serving of the drink -/
def recipe : DrinkRecipe := {
  tea := 1/4,
  lemonade := 5/4
}

/-- Calculates the amount of lemonade in the pitcher -/
def lemonadeInPitcher (r : DrinkRecipe) (total : Rat) : Rat :=
  (r.lemonade / (r.tea + r.lemonade)) * total

theorem lemonade_amount :
  lemonadeInPitcher recipe totalAmount = 15 := by sorry

end NUMINAMATH_CALUDE_lemonade_amount_l150_15001


namespace NUMINAMATH_CALUDE_sin_inequality_solution_set_l150_15088

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2*n - 1)*Real.pi - θ < x ∧ x < 2*n*Real.pi + θ} = {x : ℝ | Real.sin x < a} := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_solution_set_l150_15088


namespace NUMINAMATH_CALUDE_smallest_N_with_301_l150_15031

/-- The function that generates the concatenated string for a given N -/
def generateString (N : ℕ) : String := sorry

/-- The predicate that checks if "301" appears in a string -/
def contains301 (s : String) : Prop := sorry

/-- The theorem stating that 38 is the smallest N that satisfies the condition -/
theorem smallest_N_with_301 : 
  (∀ n < 38, ¬ contains301 (generateString n)) ∧ 
  contains301 (generateString 38) := by sorry

end NUMINAMATH_CALUDE_smallest_N_with_301_l150_15031


namespace NUMINAMATH_CALUDE_max_odd_numbers_in_pyramid_l150_15036

/-- Represents a number pyramid where each number above the bottom row
    is the sum of the two numbers immediately below it. -/
structure NumberPyramid where
  rows : Nat
  cells : Nat

/-- Represents the maximum number of odd numbers that can be placed in a number pyramid. -/
def maxOddNumbers (pyramid : NumberPyramid) : Nat :=
  14

/-- Theorem stating that the maximum number of odd numbers in a number pyramid is 14. -/
theorem max_odd_numbers_in_pyramid (pyramid : NumberPyramid) :
  maxOddNumbers pyramid = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_odd_numbers_in_pyramid_l150_15036


namespace NUMINAMATH_CALUDE_inequality_proof_l150_15066

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l150_15066


namespace NUMINAMATH_CALUDE_inverse_mod_89_l150_15002

theorem inverse_mod_89 (h : (16⁻¹ : ZMod 89) = 28) : (256⁻¹ : ZMod 89) = 56 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_89_l150_15002


namespace NUMINAMATH_CALUDE_pasture_feeding_duration_l150_15042

/-- Represents the daily grass consumption of a single cow -/
def daily_consumption_per_cow : ℝ := sorry

/-- Represents the initial amount of grass in the pasture -/
def initial_grass : ℝ := sorry

/-- Represents the daily growth rate of grass -/
def daily_growth_rate : ℝ := sorry

/-- The grass consumed by a number of cows over a period of days
    equals the initial grass plus the grass grown during that period -/
def grass_consumption (cows : ℝ) (days : ℝ) : Prop :=
  cows * daily_consumption_per_cow * days = initial_grass + daily_growth_rate * days

theorem pasture_feeding_duration :
  grass_consumption 20 40 ∧ grass_consumption 35 10 →
  grass_consumption 25 20 := by sorry

end NUMINAMATH_CALUDE_pasture_feeding_duration_l150_15042


namespace NUMINAMATH_CALUDE_lcm_problem_l150_15076

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l150_15076


namespace NUMINAMATH_CALUDE_baseball_cards_distribution_l150_15015

theorem baseball_cards_distribution (total_cards : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) :
  total_cards = 24 →
  num_friends = 4 →
  total_cards = num_friends * cards_per_friend →
  cards_per_friend = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_distribution_l150_15015


namespace NUMINAMATH_CALUDE_binomial_square_proof_l150_15000

theorem binomial_square_proof :
  ∃ (r s : ℚ), (r * x + s)^2 = (81/16 : ℚ) * x^2 + 18 * x + 16 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_square_proof_l150_15000


namespace NUMINAMATH_CALUDE_slower_ball_speed_l150_15083

/-- Two balls moving on a circular path with the following properties:
    - When moving in the same direction, they meet every 20 seconds
    - When moving in opposite directions, they meet every 4 seconds
    - When moving towards each other, the distance between them decreases by 75 cm every 3 seconds
    Prove that the speed of the slower ball is 10 cm/s -/
theorem slower_ball_speed (v u : ℝ) (C : ℝ) : 
  (20 * (v - u) = C) →  -- Same direction meeting condition
  (4 * (v + u) = C) →   -- Opposite direction meeting condition
  ((v + u) * 3 = 75) →  -- Approaching speed condition
  (u = 10) :=           -- Speed of slower ball
by sorry

end NUMINAMATH_CALUDE_slower_ball_speed_l150_15083


namespace NUMINAMATH_CALUDE_simplify_expression_l150_15073

theorem simplify_expression (b : ℝ) (h : b ≠ -2/3) :
  3 - 2 / (2 + b / (1 + b)) = 3 - (1 + b) / (2 + 3*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l150_15073


namespace NUMINAMATH_CALUDE_symbol_values_l150_15047

theorem symbol_values (star circle ring : ℤ) 
  (h1 : star + ring = 46)
  (h2 : star + circle = 91)
  (h3 : circle + ring = 63) :
  star = 37 ∧ circle = 54 ∧ ring = 9 := by
  sorry

end NUMINAMATH_CALUDE_symbol_values_l150_15047


namespace NUMINAMATH_CALUDE_integer_quotient_characterization_l150_15072

def solution_set : Set (ℤ × ℤ) :=
  {(1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2)}

theorem integer_quotient_characterization (m n : ℤ) :
  (∃ k : ℤ, (n^3 + 1) = k * (m * n - 1)) ↔ (m, n) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_integer_quotient_characterization_l150_15072


namespace NUMINAMATH_CALUDE_noemi_initial_money_l150_15077

def roulette_loss : Int := 600
def blackjack_win : Int := 400
def poker_loss : Int := 400
def baccarat_win : Int := 500
def meal_cost : Int := 200
def final_amount : Int := 1800

theorem noemi_initial_money :
  ∃ (initial_money : Int),
    initial_money = 
      roulette_loss + blackjack_win + poker_loss + baccarat_win + meal_cost + final_amount :=
by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_money_l150_15077


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l150_15010

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 2 ∧ b > 2 ∧
    n = 2 * a + 1 ∧
    n = 1 * b + 2 ∧
    (∀ (m : ℕ) (c d : ℕ), 
      c > 2 ∧ d > 2 ∧
      m = 2 * c + 1 ∧
      m = 1 * d + 2 →
      n ≤ m) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l150_15010


namespace NUMINAMATH_CALUDE_video_game_lives_l150_15023

theorem video_game_lives (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) :
  initial_players = 10 →
  players_quit = 7 →
  total_lives = 24 →
  (total_lives / (initial_players - players_quit) : ℚ) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l150_15023


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_a_ge_two_l150_15044

/-- A line that does not pass through the second quadrant -/
structure LineNotInSecondQuadrant where
  a : ℝ
  not_in_second_quadrant : ∀ (x y : ℝ), (a - 2) * y = (3 * a - 1) * x - 4 → ¬(x < 0 ∧ y > 0)

/-- The range of values for a when the line does not pass through the second quadrant -/
theorem line_not_in_second_quadrant_iff_a_ge_two (l : LineNotInSecondQuadrant) :
  l.a ∈ Set.Ici 2 ↔ ∀ (x y : ℝ), (l.a - 2) * y = (3 * l.a - 1) * x - 4 → ¬(x < 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_a_ge_two_l150_15044


namespace NUMINAMATH_CALUDE_num_terms_xyz_4_is_15_l150_15048

/-- The number of terms in the expansion of (x+y+z)^4 -/
def num_terms_xyz_4 : ℕ := sorry

/-- Theorem stating that the number of terms in (x+y+z)^4 is 15 -/
theorem num_terms_xyz_4_is_15 : num_terms_xyz_4 = 15 := by sorry

end NUMINAMATH_CALUDE_num_terms_xyz_4_is_15_l150_15048


namespace NUMINAMATH_CALUDE_horner_third_step_equals_12_l150_15045

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_step (a : ℝ) (x : ℝ) (prev : ℝ) : ℝ := prev * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (horner_step · x) 0

theorem horner_third_step_equals_12 :
  let coeffs := [2, 0, -3, 2, 1, -3]
  let x := 2
  let v3 := (horner_method (coeffs.take 4) x)
  v3 = 12 := by sorry

end NUMINAMATH_CALUDE_horner_third_step_equals_12_l150_15045


namespace NUMINAMATH_CALUDE_point_P_coordinates_l150_15017

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, (P.1 - A.1, P.2 - A.2) = 3 • (B.1 - A.1, B.2 - A.2) ∧ P = (8, -15) := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l150_15017
