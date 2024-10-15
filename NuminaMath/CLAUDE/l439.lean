import Mathlib

namespace NUMINAMATH_CALUDE_total_students_l439_43924

/-- The number of students in the three classrooms -/
structure ClassroomCounts where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (c : ClassroomCounts) : Prop :=
  c.tina = c.maura ∧
  c.zack = (c.tina + c.maura) / 2 ∧
  c.zack - 1 = 22

/-- The theorem stating the total number of students -/
theorem total_students (c : ClassroomCounts) 
  (h : satisfies_conditions c) : c.tina + c.maura + c.zack = 69 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l439_43924


namespace NUMINAMATH_CALUDE_exponent_multiplication_l439_43947

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l439_43947


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l439_43923

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N :
  M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l439_43923


namespace NUMINAMATH_CALUDE_pie_fraction_to_percentage_l439_43919

theorem pie_fraction_to_percentage : 
  let apple_fraction : ℚ := 1/5
  let cherry_fraction : ℚ := 3/4
  let total_fraction : ℚ := apple_fraction + cherry_fraction
  (total_fraction * 100 : ℚ) = 95 := by sorry

end NUMINAMATH_CALUDE_pie_fraction_to_percentage_l439_43919


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l439_43926

theorem quadratic_equation_two_distinct_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*k*x₁ - 2 = 0 ∧ x₂^2 - 3*k*x₂ - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l439_43926


namespace NUMINAMATH_CALUDE_angle_ABF_is_right_l439_43952

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = (Real.sqrt 5 - 1) / 2

/-- The angle ABF in an ellipse, where A is the left vertex, 
    F is the right focus, and B is one endpoint of the minor axis -/
def angle_ABF (E : Ellipse) : ℝ := sorry

/-- Theorem: In an ellipse with the given properties, the angle ABF is 90° -/
theorem angle_ABF_is_right (E : Ellipse) : angle_ABF E = 90 := by sorry

end NUMINAMATH_CALUDE_angle_ABF_is_right_l439_43952


namespace NUMINAMATH_CALUDE_memory_efficiency_improvement_l439_43988

theorem memory_efficiency_improvement (x : ℝ) (h : x > 0) :
  (100 / x) - (100 / (1.2 * x)) = 5 / 12 ↔
  (100 / x) - (100 / ((1 + 0.2) * x)) = 5 / 12 :=
by sorry

end NUMINAMATH_CALUDE_memory_efficiency_improvement_l439_43988


namespace NUMINAMATH_CALUDE_candy_bar_fundraiser_l439_43993

theorem candy_bar_fundraiser (cost_per_bar : ℝ) (avg_sold_per_member : ℝ) (total_earnings : ℝ)
  (h1 : cost_per_bar = 0.5)
  (h2 : avg_sold_per_member = 8)
  (h3 : total_earnings = 80) :
  (total_earnings / cost_per_bar) / avg_sold_per_member = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_fundraiser_l439_43993


namespace NUMINAMATH_CALUDE_max_volume_box_l439_43991

/-- The volume function of the box -/
def V (x : ℝ) : ℝ := (48 - 2*x)^2 * x

/-- The domain of x -/
def valid_x (x : ℝ) : Prop := 0 < x ∧ x < 24

theorem max_volume_box :
  ∃ (x_max : ℝ), valid_x x_max ∧
  (∀ x, valid_x x → V x ≤ V x_max) ∧
  x_max = 8 ∧ V x_max = 8192 := by
sorry

end NUMINAMATH_CALUDE_max_volume_box_l439_43991


namespace NUMINAMATH_CALUDE_perimeter_of_special_isosceles_triangle_l439_43920

-- Define the real numbers m and n
variable (m n : ℝ)

-- Define the condition |m-2| + √(n-4) = 0
def condition (m n : ℝ) : Prop := abs (m - 2) + Real.sqrt (n - 4) = 0

-- Define an isosceles triangle with sides m and n
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (base : ℝ)
  (is_isosceles : side1 = side2)

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

-- State the theorem
theorem perimeter_of_special_isosceles_triangle :
  ∀ m n : ℝ, condition m n →
  ∃ t : IsoscelesTriangle, (t.side1 = m ∨ t.side1 = n) ∧ (t.base = m ∨ t.base = n) →
  perimeter t = 10 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_special_isosceles_triangle_l439_43920


namespace NUMINAMATH_CALUDE_physics_majors_consecutive_probability_l439_43975

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 2

/-- The number of biology majors -/
def biology_majors : ℕ := 1

/-- The probability of all physics majors sitting in consecutive seats -/
def consecutive_physics_probability : ℚ := 1 / 24

theorem physics_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let valid_arrangements := 3 * Nat.factorial (total_people - physics_majors)
  consecutive_physics_probability = (valid_arrangements : ℚ) / total_arrangements :=
sorry

end NUMINAMATH_CALUDE_physics_majors_consecutive_probability_l439_43975


namespace NUMINAMATH_CALUDE_sum_of_digits_in_repeating_decimal_l439_43984

/-- The repeating decimal representation of 3/11 -/
def repeating_decimal : ℚ := 3 / 11

/-- The first digit in the repeating part of the decimal -/
def a : ℕ := 2

/-- The second digit in the repeating part of the decimal -/
def b : ℕ := 7

/-- Theorem stating that the sum of a and b is 9 -/
theorem sum_of_digits_in_repeating_decimal : a + b = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_repeating_decimal_l439_43984


namespace NUMINAMATH_CALUDE_possible_values_of_a_l439_43925

def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (a = -1/2 ∨ a = 0 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l439_43925


namespace NUMINAMATH_CALUDE_fraction_simplification_l439_43911

theorem fraction_simplification (x m n : ℝ) (hx : x ≠ 0) (hmn : m + n ≠ 0) :
  x / (x * (m + n)) = 1 / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l439_43911


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l439_43999

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l439_43999


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l439_43940

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l439_43940


namespace NUMINAMATH_CALUDE_rabbit_travel_time_l439_43987

def rabbit_speed : ℝ := 10  -- miles per hour
def distance : ℝ := 3  -- miles

theorem rabbit_travel_time : 
  (distance / rabbit_speed) * 60 = 18 := by sorry

end NUMINAMATH_CALUDE_rabbit_travel_time_l439_43987


namespace NUMINAMATH_CALUDE_verandah_width_is_two_l439_43901

/-- Represents the dimensions of a rectangular room with a surrounding verandah. -/
structure RoomWithVerandah where
  roomLength : ℝ
  roomWidth : ℝ
  verandahWidth : ℝ

/-- Calculates the area of the verandah given the room dimensions. -/
def verandahArea (r : RoomWithVerandah) : ℝ :=
  (r.roomLength + 2 * r.verandahWidth) * (r.roomWidth + 2 * r.verandahWidth) - r.roomLength * r.roomWidth

/-- Theorem stating that for a room of 15m x 12m with a verandah of area 124 sq m, the verandah width is 2m. -/
theorem verandah_width_is_two :
  ∃ (r : RoomWithVerandah), r.roomLength = 15 ∧ r.roomWidth = 12 ∧ verandahArea r = 124 ∧ r.verandahWidth = 2 :=
by sorry

end NUMINAMATH_CALUDE_verandah_width_is_two_l439_43901


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l439_43960

/-- The surface area of a cuboid given its dimensions -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a cuboid with length 4, width 5, and height 6 is 148 -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 5 6 = 148 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l439_43960


namespace NUMINAMATH_CALUDE_intersection_M_N_l439_43963

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l439_43963


namespace NUMINAMATH_CALUDE_numerical_puzzle_solution_l439_43972

theorem numerical_puzzle_solution :
  ∃! (A B C D E F : ℕ),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ F ≤ 9 ∧
    (10 * A + A) * (10 * A + B) = 1000 * C + 100 * D + 10 * E + F ∧
    (10 * C + C) * (100 * C + 10 * E + F) = 1000 * C + 100 * D + 10 * E + F ∧
    A = 4 ∧ B = 5 ∧ C = 1 ∧ D = 9 ∧ E = 8 ∧ F = 0 :=
by sorry

end NUMINAMATH_CALUDE_numerical_puzzle_solution_l439_43972


namespace NUMINAMATH_CALUDE_journey_results_correct_l439_43933

/-- Truck's journey between Town A and Village B -/
structure TruckJourney where
  uphill_distance : ℝ
  downhill_distance : ℝ
  flat_distance : ℝ
  round_trip_time_diff : ℝ
  uphill_speed_ratio : ℝ
  downhill_speed_ratio : ℝ
  flat_speed_ratio : ℝ

/-- Calculated speeds and times for the journey -/
structure JourneyResults where
  uphill_speed : ℝ
  downhill_speed : ℝ
  flat_speed : ℝ
  time_a_to_b : ℝ
  time_b_to_a : ℝ

/-- Theorem stating the correctness of the calculated results -/
theorem journey_results_correct (j : TruckJourney)
  (res : JourneyResults)
  (h1 : j.uphill_distance = 20)
  (h2 : j.downhill_distance = 14)
  (h3 : j.flat_distance = 5)
  (h4 : j.round_trip_time_diff = 1/6)
  (h5 : j.uphill_speed_ratio = 3)
  (h6 : j.downhill_speed_ratio = 6)
  (h7 : j.flat_speed_ratio = 5)
  (h8 : res.uphill_speed = 18)
  (h9 : res.downhill_speed = 36)
  (h10 : res.flat_speed = 30)
  (h11 : res.time_a_to_b = 5/3)
  (h12 : res.time_b_to_a = 3/2) :
  (j.uphill_distance / res.uphill_speed +
   j.downhill_distance / res.downhill_speed +
   j.flat_distance / res.flat_speed) -
  (j.uphill_distance / res.downhill_speed +
   j.downhill_distance / res.uphill_speed +
   j.flat_distance / res.flat_speed) = j.round_trip_time_diff ∧
  res.time_a_to_b =
    j.uphill_distance / res.uphill_speed +
    j.downhill_distance / res.downhill_speed +
    j.flat_distance / res.flat_speed ∧
  res.time_b_to_a =
    j.uphill_distance / res.downhill_speed +
    j.downhill_distance / res.uphill_speed +
    j.flat_distance / res.flat_speed ∧
  res.uphill_speed / res.downhill_speed = j.uphill_speed_ratio / j.downhill_speed_ratio ∧
  res.downhill_speed / res.flat_speed = j.downhill_speed_ratio / j.flat_speed_ratio :=
by sorry


end NUMINAMATH_CALUDE_journey_results_correct_l439_43933


namespace NUMINAMATH_CALUDE_hexagon_area_l439_43906

/-- Regular hexagon with vertices A at (0,0) and C at (10,2) -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ
  is_regular : Bool
  A_is_origin : A = (0, 0)
  C_coordinates : C = (10, 2)

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that the area of the specified regular hexagon is 52√3 -/
theorem hexagon_area (h : RegularHexagon) : area h = 52 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l439_43906


namespace NUMINAMATH_CALUDE_original_number_problem_l439_43965

theorem original_number_problem (x : ℝ) :
  1 - 1/x = 5/2 → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l439_43965


namespace NUMINAMATH_CALUDE_complex_cube_root_l439_43943

theorem complex_cube_root : ∃ (z : ℂ), z^2 + 2 = 0 ∧ (z^3 = 2 * Real.sqrt 2 * I ∨ z^3 = -2 * Real.sqrt 2 * I) := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l439_43943


namespace NUMINAMATH_CALUDE_committee_formation_l439_43955

theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 4) : 
  (Nat.choose (n - 1) (k - 1)) * (Nat.choose (n - 1) k) = 1225 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l439_43955


namespace NUMINAMATH_CALUDE_range_of_a_l439_43961

/-- Proposition p: For all real x, x²-2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: There exists a real x₀ such that x₀²+2ax₀+2-a=0 -/
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The range of a given the conditions on p and q -/
theorem range_of_a : ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l439_43961


namespace NUMINAMATH_CALUDE_solve_equation_l439_43912

theorem solve_equation (x : ℝ) (h : x - 3*x + 4*x = 140) : x = 70 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l439_43912


namespace NUMINAMATH_CALUDE_count_integers_with_fourth_power_between_negative_hundred_and_hundred_l439_43935

theorem count_integers_with_fourth_power_between_negative_hundred_and_hundred :
  (∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ -100 < x^4 ∧ x^4 < 100) ∧ Finset.card S = 7) := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_fourth_power_between_negative_hundred_and_hundred_l439_43935


namespace NUMINAMATH_CALUDE_m_equals_2_sufficient_not_necessary_l439_43931

def M (m : ℝ) : Set ℝ := {-1, m^2}
def N : Set ℝ := {2, 4}

theorem m_equals_2_sufficient_not_necessary :
  ∃ m : ℝ, (M m ∩ N = {4} ∧ m ≠ 2) ∧
  ∀ m : ℝ, m = 2 → M m ∩ N = {4} :=
sorry

end NUMINAMATH_CALUDE_m_equals_2_sufficient_not_necessary_l439_43931


namespace NUMINAMATH_CALUDE_function_value_at_2004_l439_43951

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) + 4,
    where α, β, a, and b are non-zero real numbers, and f(2003) = 6,
    prove that f(2004) = 2. -/
theorem function_value_at_2004 
  (α β a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4) 
  (h2003 : f 2003 = 6) : 
  f 2004 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2004_l439_43951


namespace NUMINAMATH_CALUDE_new_average_weight_l439_43902

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 19 → 
  initial_average = 15 → 
  new_student_weight = 3 → 
  (initial_students * initial_average + new_student_weight) / (initial_students + 1) = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l439_43902


namespace NUMINAMATH_CALUDE_pencils_per_child_l439_43942

theorem pencils_per_child (total_children : ℕ) (total_pencils : ℕ) 
  (h1 : total_children = 9) 
  (h2 : total_pencils = 18) : 
  total_pencils / total_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_child_l439_43942


namespace NUMINAMATH_CALUDE_pizza_combinations_l439_43962

/-- The number of pizza toppings available. -/
def num_toppings : ℕ := 8

/-- The number of incompatible topping pairs. -/
def num_incompatible_pairs : ℕ := 1

/-- Calculates the number of combinations of n items taken k at a time. -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of possible one-topping and two-topping pizzas, given the number of toppings
    and the number of incompatible pairs. -/
def total_pizzas (n incompatible : ℕ) : ℕ :=
  n + combinations n 2 - incompatible

/-- Theorem stating that the total number of possible one-topping and two-topping pizzas
    is 35, given 8 toppings and 1 incompatible pair. -/
theorem pizza_combinations :
  total_pizzas num_toppings num_incompatible_pairs = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l439_43962


namespace NUMINAMATH_CALUDE_calculator_addition_correct_l439_43945

/-- Represents a calculator button --/
inductive CalculatorButton
  | Digit (n : Nat)
  | Plus
  | Equals

/-- Represents a sequence of button presses on a calculator --/
def ButtonSequence := List CalculatorButton

/-- Evaluates a sequence of button presses and returns the result --/
def evaluate (seq : ButtonSequence) : Nat :=
  sorry

/-- The correct sequence of button presses to calculate 569 + 728 --/
def correctSequence : ButtonSequence :=
  [CalculatorButton.Digit 569, CalculatorButton.Plus, CalculatorButton.Digit 728, CalculatorButton.Equals]

theorem calculator_addition_correct :
  evaluate correctSequence = 569 + 728 :=
sorry

end NUMINAMATH_CALUDE_calculator_addition_correct_l439_43945


namespace NUMINAMATH_CALUDE_quadratic_solution_l439_43932

theorem quadratic_solution (x : ℝ) : x^2 - 4*x + 3 = 0 ∧ x ≥ 0 → x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l439_43932


namespace NUMINAMATH_CALUDE_solve_for_a_l439_43915

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem solve_for_a (a : ℝ) (h1 : a > 1) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 ↔ |f a (2*x + a) - 2*f a x| ≤ 2) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l439_43915


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l439_43904

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 18 = 0 ∧ x^3 - 3*x^2 - 9*x + 5 = 59 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l439_43904


namespace NUMINAMATH_CALUDE_triangle_base_length_l439_43957

theorem triangle_base_length 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : height = 8) 
  (h2 : area = 24) 
  (h3 : area = (1/2) * height * base) : 
  base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l439_43957


namespace NUMINAMATH_CALUDE_camping_products_costs_l439_43930

/-- The wholesale cost of a sleeping bag -/
def sleeping_bag_cost : ℚ := 560 / 23

/-- The wholesale cost of a tent -/
def tent_cost : ℚ := 200 / 3

/-- The selling price of a sleeping bag -/
def sleeping_bag_price : ℚ := 28

/-- The selling price of a tent -/
def tent_price : ℚ := 80

/-- The gross profit percentage for sleeping bags -/
def sleeping_bag_profit_percent : ℚ := 15 / 100

/-- The gross profit percentage for tents -/
def tent_profit_percent : ℚ := 20 / 100

theorem camping_products_costs :
  (sleeping_bag_cost * (1 + sleeping_bag_profit_percent) = sleeping_bag_price) ∧
  (tent_cost * (1 + tent_profit_percent) = tent_price) := by
  sorry

end NUMINAMATH_CALUDE_camping_products_costs_l439_43930


namespace NUMINAMATH_CALUDE_smallest_prime_10_less_than_perfect_square_l439_43927

/-- A number is a perfect square if it's the square of an integer -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- The smallest prime that is 10 less than a perfect square -/
theorem smallest_prime_10_less_than_perfect_square :
  (∃ a : ℕ, Nat.Prime a ∧
    (∃ b : ℕ, is_perfect_square b ∧ a = b - 10) ∧
    (∀ a' : ℕ, a' < a →
      ¬(Nat.Prime a' ∧ ∃ b' : ℕ, is_perfect_square b' ∧ a' = b' - 10))) →
  (∃ a : ℕ, a = 71 ∧ Nat.Prime a ∧
    (∃ b : ℕ, is_perfect_square b ∧ a = b - 10) ∧
    (∀ a' : ℕ, a' < a →
      ¬(Nat.Prime a' ∧ ∃ b' : ℕ, is_perfect_square b' ∧ a' = b' - 10))) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_10_less_than_perfect_square_l439_43927


namespace NUMINAMATH_CALUDE_solve_equation_l439_43909

theorem solve_equation : ∃ y : ℝ, (60 / 100 = Real.sqrt ((y + 20) / 100)) ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l439_43909


namespace NUMINAMATH_CALUDE_vertical_translation_of_linear_function_l439_43959

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Translates a linear function vertically by a given amount -/
def translate_vertical (f : LinearFunction) (k : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + k }

/-- The original function y = -3x -/
def original_function : LinearFunction :=
  { m := -3, b := 0 }

/-- The amount of vertical translation -/
def translation_amount : ℝ := 2

theorem vertical_translation_of_linear_function :
  translate_vertical original_function translation_amount =
  { m := -3, b := 2 } :=
sorry

end NUMINAMATH_CALUDE_vertical_translation_of_linear_function_l439_43959


namespace NUMINAMATH_CALUDE_complex_expression_equality_l439_43914

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- State the theorem
theorem complex_expression_equality : (2 / z) + z^2 = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l439_43914


namespace NUMINAMATH_CALUDE_basketball_spectators_l439_43958

theorem basketball_spectators (total : ℕ) (children : ℕ) 
  (h1 : total = 10000)
  (h2 : children = 2500)
  (h3 : children = 5 * (total - children - (total - children - children) / 5)) :
  total - children - (total - children - children) / 5 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_basketball_spectators_l439_43958


namespace NUMINAMATH_CALUDE_vegetable_cost_l439_43971

def initial_amount : ℤ := 100
def roast_cost : ℤ := 17
def remaining_amount : ℤ := 72

theorem vegetable_cost :
  initial_amount - roast_cost - remaining_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_cost_l439_43971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l439_43938

theorem arithmetic_sequence_sum_divisibility :
  ∀ (a d : ℕ+), ∃ (k : ℕ+), (12 * a + 66 * d : ℕ) = 6 * k ∧
  ∀ (m : ℕ+), m < 6 → ∃ (a' d' : ℕ+), ¬(∃ (k' : ℕ+), (12 * a' + 66 * d' : ℕ) = m * k') :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l439_43938


namespace NUMINAMATH_CALUDE_factorization_cubic_quadratic_l439_43948

theorem factorization_cubic_quadratic (a : ℝ) : a^3 - 2*a^2 = a^2*(a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_quadratic_l439_43948


namespace NUMINAMATH_CALUDE_max_n_with_divisor_condition_l439_43953

theorem max_n_with_divisor_condition (N : ℕ) : 
  (∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ N ∧ d₂ ∣ N ∧ d₃ ∣ N ∧
    d₁ < d₂ ∧ 
    (∀ d : ℕ, d ∣ N → d ≤ d₁ ∨ d ≥ d₂) ∧
    (∀ d : ℕ, d ∣ N → d ≤ d₃ ∨ d > N / d₃) ∧
    d₃ = 21 * d₂) →
  N ≤ 441 := by
sorry

end NUMINAMATH_CALUDE_max_n_with_divisor_condition_l439_43953


namespace NUMINAMATH_CALUDE_expression_value_at_nine_l439_43979

theorem expression_value_at_nine :
  let x : ℝ := 9
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_nine_l439_43979


namespace NUMINAMATH_CALUDE_inequality_solution_l439_43997

theorem inequality_solution (x : ℝ) : 
  (∃ a : ℝ, a ∈ Set.Icc (-1) 2 ∧ (2 - a) * x^3 + (1 - 2*a) * x^2 - 6*x + 5 + 4*a - a^2 < 0) ↔ 
  (x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l439_43997


namespace NUMINAMATH_CALUDE_garden_circle_diameter_l439_43905

/-- Given a circular ground with a 2-metre broad garden around it,
    if the area of the garden is 226.19467105846502 square metres,
    then the diameter of the circular ground is 34 metres. -/
theorem garden_circle_diameter (r : ℝ) : 
  (π * ((r + 2)^2 - r^2) = 226.19467105846502) → 
  (2 * r = 34) := by
  sorry

end NUMINAMATH_CALUDE_garden_circle_diameter_l439_43905


namespace NUMINAMATH_CALUDE_quadratic_complex_roots_l439_43950

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ),
  z₁ = Complex.mk (Real.sqrt 7 - 1) ((Real.sqrt 7) / 2) ∧
  z₂ = Complex.mk (-(Real.sqrt 7) - 1) (-(Real.sqrt 7) / 2) ∧
  z₁^2 + 2*z₁ = Complex.mk 3 7 ∧
  z₂^2 + 2*z₂ = Complex.mk 3 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complex_roots_l439_43950


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l439_43903

/-- Given a cube with volume 1000 cm³, if 8 small cubes of equal size are cut off from its corners
    such that the remaining volume is 488 cm³, then the edge length of each small cube is 4 cm. -/
theorem small_cube_edge_length (x : ℝ) : 
  (1000 : ℝ) - 8 * x^3 = 488 → x = 4 := by sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l439_43903


namespace NUMINAMATH_CALUDE_solution_exists_l439_43981

-- Define the function f
def f (x : ℝ) : ℝ := (40 * x + (40 * x + 24) ^ (1/4)) ^ (1/4)

-- State the theorem
theorem solution_exists : ∃ x : ℝ, f x = 24 := by
  use 8293.8
  sorry

end NUMINAMATH_CALUDE_solution_exists_l439_43981


namespace NUMINAMATH_CALUDE_odd_function_implies_m_equals_one_inequality_implies_a_range_l439_43985

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp x - m / exp x

theorem odd_function_implies_m_equals_one (m : ℝ) :
  (∀ x, f m x = -f m (-x)) → m = 1 := by sorry

theorem inequality_implies_a_range (m : ℝ) :
  m = 1 →
  (∀ a : ℝ, f m (a - 1) + f m (2 * a^2) ≤ 0 → -1 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_m_equals_one_inequality_implies_a_range_l439_43985


namespace NUMINAMATH_CALUDE_sin_120_degrees_l439_43996

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l439_43996


namespace NUMINAMATH_CALUDE_robotics_club_proof_l439_43944

theorem robotics_club_proof (total : ℕ) (programming : ℕ) (electronics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : programming = 80)
  (h3 : electronics = 50)
  (h4 : both = 15) :
  total - (programming + electronics - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_proof_l439_43944


namespace NUMINAMATH_CALUDE_census_suitable_for_class_spirit_awareness_only_class_spirit_awareness_census_suitable_l439_43929

/-- Represents a survey scenario --/
inductive SurveyScenario
  | ShellLethalRadius
  | TVViewershipRating
  | YellowRiverFishSpecies
  | ClassSpiritAwareness

/-- Determines if a census method is suitable for a given survey scenario --/
def isCensusSuitable (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.ClassSpiritAwareness => True
  | _ => False

/-- Theorem: The survey to ascertain the awareness rate of the "Shanxi Spirit" 
    among the students of a certain class is suitable for using a census method --/
theorem census_suitable_for_class_spirit_awareness :
  isCensusSuitable SurveyScenario.ClassSpiritAwareness :=
by sorry

/-- Theorem: The survey to ascertain the awareness rate of the "Shanxi Spirit" 
    among the students of a certain class is the only one suitable for using a census method --/
theorem only_class_spirit_awareness_census_suitable :
  ∀ (scenario : SurveyScenario), 
    isCensusSuitable scenario ↔ scenario = SurveyScenario.ClassSpiritAwareness :=
by sorry

end NUMINAMATH_CALUDE_census_suitable_for_class_spirit_awareness_only_class_spirit_awareness_census_suitable_l439_43929


namespace NUMINAMATH_CALUDE_third_podcast_length_l439_43998

/-- Given a 6-hour drive and four podcasts, prove that the third podcast must be 105 minutes long to fill the entire drive time. -/
theorem third_podcast_length :
  let total_drive_time : ℕ := 6 * 60
  let first_podcast : ℕ := 45
  let second_podcast : ℕ := first_podcast * 2
  let fourth_podcast : ℕ := 60
  let next_podcast : ℕ := 60
  ∃ (third_podcast : ℕ),
    third_podcast = 105 ∧
    total_drive_time = first_podcast + second_podcast + third_podcast + fourth_podcast + next_podcast :=
by sorry

end NUMINAMATH_CALUDE_third_podcast_length_l439_43998


namespace NUMINAMATH_CALUDE_max_value_ln_x_over_x_l439_43974

/-- The function f(x) = ln(x) / x attains its maximum value of 1/e for x > 0 -/
theorem max_value_ln_x_over_x :
  ∃ (c : ℝ), c > 0 ∧ 
    (∀ x > 0, (Real.log x) / x ≤ (Real.log c) / c) ∧
    (Real.log c) / c = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ln_x_over_x_l439_43974


namespace NUMINAMATH_CALUDE_work_completion_time_l439_43921

/-- The time taken to complete a work given the rates of two workers and their working schedule -/
theorem work_completion_time
  (p_completion_time q_completion_time : ℝ)
  (p_solo_time : ℝ)
  (hp : p_completion_time = 20)
  (hq : q_completion_time = 12)
  (hp_solo : p_solo_time = 4)
  : ∃ (total_time : ℝ), total_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l439_43921


namespace NUMINAMATH_CALUDE_barn_paint_area_l439_43956

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a barn with given dimensions -/
def totalPaintArea (d : BarnDimensions) : ℝ :=
  let wallArea1 := 2 * d.width * d.height
  let wallArea2 := 2 * d.length * d.height
  let ceilingArea := d.width * d.length
  let roofArea := d.width * d.length
  2 * wallArea1 + 2 * wallArea2 + ceilingArea + roofArea

/-- The theorem stating that the total paint area for the given barn dimensions is 1116 sq yd -/
theorem barn_paint_area :
  let barn := BarnDimensions.mk 12 15 7
  totalPaintArea barn = 1116 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l439_43956


namespace NUMINAMATH_CALUDE_percent_increase_proof_l439_43982

def initial_cost : ℝ := 120000
def final_cost : ℝ := 192000

theorem percent_increase_proof :
  (final_cost - initial_cost) / initial_cost * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percent_increase_proof_l439_43982


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l439_43970

/-- An ellipse with given properties -/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse with the given properties -/
def ellipse_area (e : Ellipse) : ℝ :=
  sorry

/-- The theorem stating that the area of the specific ellipse is 42π -/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-5, -1),
    major_axis_endpoint2 := (15, -1),
    point_on_ellipse := (12, 2)
  }
  ellipse_area e = 42 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l439_43970


namespace NUMINAMATH_CALUDE_find_y_value_l439_43969

theorem find_y_value (y : ℝ) (h : (15^2 * 8^3) / y = 450) : y = 256 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l439_43969


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l439_43989

theorem exponential_equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (16 : ℝ)^5 ∧ x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l439_43989


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l439_43986

-- Define the edge lengths
def edge_length_small : ℚ := 4
def edge_length_large : ℚ := 24  -- 2 feet = 24 inches

-- Define the volume ratio
def volume_ratio : ℚ := (edge_length_small / edge_length_large) ^ 3

-- Theorem statement
theorem cube_volume_ratio :
  volume_ratio = 1 / 216 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l439_43986


namespace NUMINAMATH_CALUDE_amare_dresses_l439_43990

/-- The number of dresses Amare needs to make -/
def number_of_dresses : ℕ := 4

/-- The amount of fabric required for one dress in yards -/
def fabric_per_dress : ℚ := 5.5

/-- The amount of fabric Amare has in feet -/
def fabric_amare_has : ℕ := 7

/-- The amount of fabric Amare still needs in feet -/
def fabric_amare_needs : ℕ := 59

/-- The number of feet in a yard -/
def feet_per_yard : ℕ := 3

theorem amare_dresses :
  number_of_dresses = 
    (((fabric_amare_has + fabric_amare_needs : ℚ) / feet_per_yard) / fabric_per_dress).floor :=
by sorry

end NUMINAMATH_CALUDE_amare_dresses_l439_43990


namespace NUMINAMATH_CALUDE_class_size_proof_l439_43922

theorem class_size_proof (total_average : ℝ) (group1_size : ℕ) (group1_average : ℝ)
                         (group2_size : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_size = 5 →
  group1_average = 14 →
  group2_size = 9 →
  group2_average = 16 →
  last_student_age = 11 →
  ∃ (total_students : ℕ), total_students = 15 ∧
    (total_students : ℝ) * total_average =
      (group1_size : ℝ) * group1_average +
      (group2_size : ℝ) * group2_average +
      last_student_age :=
by
  sorry

#check class_size_proof

end NUMINAMATH_CALUDE_class_size_proof_l439_43922


namespace NUMINAMATH_CALUDE_hydrogen_weight_in_H2CrO4_l439_43928

def atomic_weight_H : ℝ := 1.008
def molecular_weight_H2CrO4 : ℝ := 118

theorem hydrogen_weight_in_H2CrO4 :
  let hydrogen_count : ℕ := 2
  let hydrogen_weight : ℝ := atomic_weight_H * hydrogen_count
  hydrogen_weight = 2.016 := by sorry

end NUMINAMATH_CALUDE_hydrogen_weight_in_H2CrO4_l439_43928


namespace NUMINAMATH_CALUDE_rectangle_to_square_l439_43992

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Theorem: A 12 × 3 rectangle can be cut into three equal parts that form a 6 × 6 square -/
theorem rectangle_to_square (rect : Rectangle) (sq : Square) : 
  rect.width = 12 ∧ rect.height = 3 ∧ sq.side = 6 →
  ∃ (part_width part_height : ℝ),
    part_width * part_height = rect.width * rect.height / 3 ∧
    3 * part_width = sq.side ∧
    part_height = sq.side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l439_43992


namespace NUMINAMATH_CALUDE_correct_matching_probability_l439_43939

/-- The number of celebrities, recent photos, and baby photos -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities to their recent photos -/
def prob_recent : ℚ := 1 / (n.factorial : ℚ)

/-- The probability of correctly matching all recent photos to baby photos -/
def prob_baby : ℚ := 1 / (n.factorial : ℚ)

/-- The overall probability of correctly matching all celebrities to their baby photos through recent photos -/
def prob_total : ℚ := prob_recent * prob_baby

theorem correct_matching_probability :
  prob_total = 1 / 576 := by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l439_43939


namespace NUMINAMATH_CALUDE_georgia_muffins_per_batch_l439_43908

/-- Calculates the number of muffins per batch given the total number of students,
    total batches made, and the number of months. -/
def muffins_per_batch (students : ℕ) (total_batches : ℕ) (months : ℕ) : ℕ :=
  students * months / total_batches

/-- Proves that given 24 students and 36 batches of muffins made in 9 months,
    the number of muffins per batch is 6. -/
theorem georgia_muffins_per_batch :
  muffins_per_batch 24 36 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_georgia_muffins_per_batch_l439_43908


namespace NUMINAMATH_CALUDE_valid_subcommittee_count_l439_43936

def total_members : ℕ := 12
def teacher_count : ℕ := 6
def subcommittee_size : ℕ := 5
def min_teachers : ℕ := 2

def subcommittee_count : ℕ := 696

theorem valid_subcommittee_count :
  (total_members.choose subcommittee_size) -
  ((teacher_count.choose 0) * ((total_members - teacher_count).choose subcommittee_size) +
   (teacher_count.choose 1) * ((total_members - teacher_count).choose (subcommittee_size - 1)))
  = subcommittee_count :=
by sorry

end NUMINAMATH_CALUDE_valid_subcommittee_count_l439_43936


namespace NUMINAMATH_CALUDE_sophomore_count_l439_43934

theorem sophomore_count (total : ℕ) (sophomore_percent : ℚ) (senior_percent : ℚ)
  (h_total : total = 50)
  (h_sophomore_percent : sophomore_percent = 1/5)
  (h_senior_percent : senior_percent = 1/4)
  (h_team_equal : ∃ (team_size : ℕ), 
    sophomore_percent * (total - seniors) = ↑team_size ∧
    senior_percent * seniors = ↑team_size)
  (seniors : ℕ) :
  total - seniors = 22 :=
sorry

end NUMINAMATH_CALUDE_sophomore_count_l439_43934


namespace NUMINAMATH_CALUDE_range_of_a_l439_43910

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x < -1}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 3}

-- Define the complement of A
def A_complement : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem statement
theorem range_of_a (a : ℝ) : B a ⊆ A_complement ↔ a ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l439_43910


namespace NUMINAMATH_CALUDE_initial_speed_is_850_l439_43978

/-- Represents the airplane's journey with given conditions -/
structure AirplaneJourney where
  totalDistance : ℝ
  distanceBeforeLanding : ℝ
  landingDuration : ℝ
  speedReduction : ℝ
  totalTime : ℝ

/-- Calculates the initial speed of the airplane given the journey parameters -/
def initialSpeed (journey : AirplaneJourney) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the initial speed is 850 km/h for the given conditions -/
theorem initial_speed_is_850 :
  let journey : AirplaneJourney := {
    totalDistance := 2900
    distanceBeforeLanding := 1700
    landingDuration := 1.5
    speedReduction := 50
    totalTime := 5
  }
  initialSpeed journey = 850 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_850_l439_43978


namespace NUMINAMATH_CALUDE_parallel_line_slope_l439_43964

/-- Given a line with equation 5x - 3y = 9, prove that the slope of any parallel line is 5/3 -/
theorem parallel_line_slope (x y : ℝ) (h : 5 * x - 3 * y = 9) :
  ∃ (m : ℝ), m = 5 / 3 ∧ ∀ (x₁ y₁ : ℝ), (5 * x₁ - 3 * y₁ = 9) → (y₁ - y) = m * (x₁ - x) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l439_43964


namespace NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l439_43977

theorem condition_sufficiency_not_necessity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 < 1 → a * b + 1 > a + b) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b + 1 > a + b ∧ a^2 + b^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_condition_sufficiency_not_necessity_l439_43977


namespace NUMINAMATH_CALUDE_area_between_semicircles_l439_43994

/-- Given a semicircle with diameter D, which is divided into two parts,
    and semicircles constructed on each part inside the given semicircle,
    the area enclosed between the three semicircles is equal to πCD²/4,
    where CD is the length of the perpendicular from the division point to the semicircle. -/
theorem area_between_semicircles (D r : ℝ) (h : 0 < r ∧ r < D) : 
  let R := D / 2
  let area := π * r * (R - r)
  let CD := Real.sqrt (2 * r * (D - r))
  area = π * CD^2 / 4 := by sorry

end NUMINAMATH_CALUDE_area_between_semicircles_l439_43994


namespace NUMINAMATH_CALUDE_modulus_of_Z_l439_43949

/-- The modulus of the complex number Z = 1/(1+i) + i^3 is equal to √10/2 -/
theorem modulus_of_Z : Complex.abs (1 / (1 + Complex.I) + Complex.I^3) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l439_43949


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l439_43918

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l439_43918


namespace NUMINAMATH_CALUDE_A_intersect_B_l439_43916

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < 2 - x ∧ 2 - x < 3}

theorem A_intersect_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l439_43916


namespace NUMINAMATH_CALUDE_smallest_lcm_three_digit_gcd_five_l439_43937

theorem smallest_lcm_three_digit_gcd_five :
  ∃ (m n : ℕ), 
    100 ≤ m ∧ m < 1000 ∧
    100 ≤ n ∧ n < 1000 ∧
    Nat.gcd m n = 5 ∧
    Nat.lcm m n = 2100 ∧
    ∀ (p q : ℕ), 
      100 ≤ p ∧ p < 1000 ∧
      100 ≤ q ∧ q < 1000 ∧
      Nat.gcd p q = 5 →
      Nat.lcm p q ≥ 2100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_three_digit_gcd_five_l439_43937


namespace NUMINAMATH_CALUDE_min_value_quadratic_l439_43968

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), ∀ (z : ℝ), z = 4 * x^2 + 8 * x + 16 → z ≥ min_z ∧ ∃ (x₀ : ℝ), 4 * x₀^2 + 8 * x₀ + 16 = min_z :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l439_43968


namespace NUMINAMATH_CALUDE_integral_value_l439_43941

theorem integral_value : ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - (x - 1)^2) - x^2) = π/4 - 1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_value_l439_43941


namespace NUMINAMATH_CALUDE_scaling_transformation_cosine_curve_l439_43954

/-- The scaling transformation applied to the curve y = cos 6x results in y' = 2cos 2x' -/
theorem scaling_transformation_cosine_curve :
  ∀ (x y x' y' : ℝ),
  y = Real.cos (6 * x) →
  x' = 3 * x →
  y' = 2 * y →
  y' = 2 * Real.cos (2 * x') := by
sorry

end NUMINAMATH_CALUDE_scaling_transformation_cosine_curve_l439_43954


namespace NUMINAMATH_CALUDE_charlie_won_two_games_l439_43973

/-- Represents a player in the tournament -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player

/-- The number of games won by a player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Alice => 2
  | Player.Bob => 1
  | Player.Charlie => sorry  -- To be proven

/-- The number of games lost by a player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Alice => 1
  | Player.Bob => 2
  | Player.Charlie => 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := 3

theorem charlie_won_two_games :
  games_won Player.Charlie = 2 := by sorry

end NUMINAMATH_CALUDE_charlie_won_two_games_l439_43973


namespace NUMINAMATH_CALUDE_solve_for_y_l439_43946

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 64) (h2 : x = 8) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l439_43946


namespace NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l439_43917

/-- Calculates the total amount of milk in ounces bought by a shopkeeper -/
theorem shopkeeper_milk_ounces 
  (packets : ℕ) 
  (ml_per_packet : ℕ) 
  (ml_per_ounce : ℕ) 
  (h1 : packets = 150)
  (h2 : ml_per_packet = 250)
  (h3 : ml_per_ounce = 30) : 
  (packets * ml_per_packet) / ml_per_ounce = 1250 := by
  sorry

#check shopkeeper_milk_ounces

end NUMINAMATH_CALUDE_shopkeeper_milk_ounces_l439_43917


namespace NUMINAMATH_CALUDE_three_lines_determine_plane_l439_43966

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane where
  -- Define properties of a plane

/-- Represents the intersection of two lines -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Represents that three lines have no common point -/
def no_common_point (l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- Represents that a plane contains a line -/
def plane_contains_line (p : Plane) (l : Line3D) : Prop :=
  sorry

/-- Three lines intersecting in pairs without a common point determine a unique plane -/
theorem three_lines_determine_plane (l1 l2 l3 : Line3D) :
  intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1 ∧ no_common_point l1 l2 l3 →
  ∃! p : Plane, plane_contains_line p l1 ∧ plane_contains_line p l2 ∧ plane_contains_line p l3 :=
sorry

end NUMINAMATH_CALUDE_three_lines_determine_plane_l439_43966


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l439_43983

theorem hexagon_angle_measure (a b c d e : ℝ) (h1 : a = 134) (h2 : b = 98) (h3 : c = 120) (h4 : d = 110) (h5 : e = 96) :
  720 - (a + b + c + d + e) = 162 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l439_43983


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l439_43976

def original_number : ℕ := 8679921
def divisor : ℕ := 330

theorem sum_of_prime_factors : 
  ∃ (n : ℕ), 
    n ≥ original_number ∧ 
    n % divisor = 0 ∧
    (∀ m : ℕ, m ≥ original_number ∧ m % divisor = 0 → m ≥ n) ∧
    (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id = 284) :=
sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l439_43976


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l439_43980

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 7

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 500

/-- Al's rest days in his cycle -/
def al_rest_days : Finset ℕ := {6, 7}

/-- Barb's rest day in her cycle -/
def barb_rest_day : ℕ := 5

/-- The number of days both Al and Barb rest in the same 35-day period -/
def coinciding_rest_days_per_cycle : ℕ := 1

theorem coinciding_rest_days_count : 
  (total_days / (al_cycle * barb_cycle)) * coinciding_rest_days_per_cycle = 14 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l439_43980


namespace NUMINAMATH_CALUDE_min_value_fraction_l439_43967

theorem min_value_fraction (x : ℝ) (h : x > 9) :
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l439_43967


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l439_43900

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l439_43900


namespace NUMINAMATH_CALUDE_minimal_kamber_group_common_meal_l439_43995

/-- The number of citizens in the city -/
def num_citizens : ℕ := 2017

/-- The number of meal types available -/
def num_meals : ℕ := 25

/-- A citizen is represented by a natural number -/
def Citizen := Fin num_citizens

/-- A meal is represented by a natural number -/
def Meal := Fin num_meals

/-- Predicate indicating whether a citizen likes a meal -/
def likes (c : Citizen) (m : Meal) : Prop := sorry

/-- A set of citizens is a suitable list if each meal is liked by at least one person in the set -/
def is_suitable_list (s : Set Citizen) : Prop :=
  ∀ m : Meal, ∃ c ∈ s, likes c m

/-- A set of citizens is a kamber group if it contains at least one person from each suitable list -/
def is_kamber_group (k : Set Citizen) : Prop :=
  ∀ s : Set Citizen, is_suitable_list s → (∃ c ∈ k, c ∈ s)

/-- A kamber group is minimal if no proper subset is also a kamber group -/
def is_minimal_kamber_group (k : Set Citizen) : Prop :=
  is_kamber_group k ∧ ∀ k' ⊂ k, ¬is_kamber_group k'

theorem minimal_kamber_group_common_meal (k : Set Citizen) 
  (h : is_minimal_kamber_group k) : 
  ∃ m : Meal, ∀ c ∈ k, likes c m := by sorry


end NUMINAMATH_CALUDE_minimal_kamber_group_common_meal_l439_43995


namespace NUMINAMATH_CALUDE_mixture_qualified_probability_l439_43907

theorem mixture_qualified_probability 
  (batch1_defective_rate : ℝ)
  (batch2_defective_rate : ℝ)
  (mix_ratio1 : ℝ)
  (mix_ratio2 : ℝ)
  (h1 : batch1_defective_rate = 0.05)
  (h2 : batch2_defective_rate = 0.15)
  (h3 : mix_ratio1 = 3)
  (h4 : mix_ratio2 = 2) :
  let total_ratio := mix_ratio1 + mix_ratio2
  let batch1_qualified_rate := 1 - batch1_defective_rate
  let batch2_qualified_rate := 1 - batch2_defective_rate
  let mixture_qualified_rate := 
    (batch1_qualified_rate * mix_ratio1 + batch2_qualified_rate * mix_ratio2) / total_ratio
  mixture_qualified_rate = 0.91 := by
sorry

end NUMINAMATH_CALUDE_mixture_qualified_probability_l439_43907


namespace NUMINAMATH_CALUDE_solve_equation_l439_43913

theorem solve_equation (x : ℤ) (h : 9873 + x = 13200) : x = 3327 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l439_43913
