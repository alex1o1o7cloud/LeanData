import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_l3400_340092

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = Real.sqrt 19) :
  let S := (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)
  S = (3 * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3400_340092


namespace NUMINAMATH_CALUDE_stratified_sampling_l3400_340017

theorem stratified_sampling (total_A total_B sample_A : ℕ) 
  (h1 : total_A = 800)
  (h2 : total_B = 500)
  (h3 : sample_A = 48) :
  (total_B : ℚ) / (total_A + total_B) * sample_A = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3400_340017


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3400_340025

theorem min_value_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, x > 0 → Real.log x - a * Real.exp x - b + 1 ≤ 0) →
  ∃ m : ℝ, m = 0 ∧ (∀ a' b' : ℝ, (∀ x : ℝ, x > 0 → Real.log x - a' * Real.exp x - b' + 1 ≤ 0) → a' + b' ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3400_340025


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l3400_340048

/-- Given Isabella's initial hair length and growth, calculate her final hair length -/
theorem isabellas_hair_length 
  (initial_length : ℕ) 
  (growth : ℕ) 
  (h1 : initial_length = 18) 
  (h2 : growth = 6) : 
  initial_length + growth = 24 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l3400_340048


namespace NUMINAMATH_CALUDE_negative_plus_square_not_always_positive_l3400_340039

theorem negative_plus_square_not_always_positive : 
  ∃ x : ℝ, x < 0 ∧ x + x^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_plus_square_not_always_positive_l3400_340039


namespace NUMINAMATH_CALUDE_problem_solution_l3400_340001

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Define the theorem
theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -- Part 1: Solution set of f(x) + f(2x+1) ≥ 6
  {x : ℝ | f x + f (2*x + 1) ≥ 6} = Set.Iic (-1) ∪ Set.Ici 3 ∧
  -- Part 2: Range of m given the condition
  ∀ m : ℝ, (∀ x : ℝ, f (x - m) - f (-x) ≤ 4/a + 1/b) → -13 ≤ m ∧ m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3400_340001


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3400_340052

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (2 * a^2 - 7 * a + 3 = 0) →
  (2 * b^2 - 7 * b + 3 = 0) →
  (a - b)^2 = 25 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3400_340052


namespace NUMINAMATH_CALUDE_bigger_part_of_54_l3400_340007

theorem bigger_part_of_54 (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) (h3 : x > 0) (h4 : y > 0) :
  max x y = 34 := by
sorry

end NUMINAMATH_CALUDE_bigger_part_of_54_l3400_340007


namespace NUMINAMATH_CALUDE_cookout_ratio_l3400_340006

def cookout_2004 : ℕ := 60
def cookout_2005 : ℕ := cookout_2004 / 2
def cookout_2006 : ℕ := 20

theorem cookout_ratio : 
  (cookout_2006 : ℚ) / cookout_2005 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cookout_ratio_l3400_340006


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_pow6_l3400_340047

theorem nearest_integer_to_3_plus_sqrt5_pow6 :
  ∃ n : ℤ, n = 22608 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_pow6_l3400_340047


namespace NUMINAMATH_CALUDE_line_intersections_l3400_340058

/-- The line equation y = -2x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -2 * x + 4

/-- The point (x, y) lies on the x-axis -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The point (x, y) lies on the y-axis -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The line y = -2x + 4 intersects the x-axis at (2, 0) and the y-axis at (0, 4) -/
theorem line_intersections :
  (∃ (x y : ℝ), line_equation x y ∧ on_x_axis x y ∧ x = 2 ∧ y = 0) ∧
  (∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = 4) :=
sorry

end NUMINAMATH_CALUDE_line_intersections_l3400_340058


namespace NUMINAMATH_CALUDE_asymptote_sum_l3400_340020

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 2 ∧ x ≠ 3 → 
    (x^3 + A*x^2 + B*x + C ≠ 0)) →
  ((x + 1) * (x - 2) * (x - 3) = x^3 + A*x^2 + B*x + C) →
  A + B + C = -5 := by
sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3400_340020


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l3400_340044

/-- Given three points A, B, and C in the plane satisfying certain conditions,
    prove that the sum of the coordinates of A is 16. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  (B.1 - C.1) / (B.1 - A.1) = 2/3 →
  (B.2 - C.2) / (B.2 - A.2) = 2/3 →
  B = (2, 5) →
  C = (5, 8) →
  A.1 + A.2 = 16 := by
sorry


end NUMINAMATH_CALUDE_sum_of_coordinates_A_l3400_340044


namespace NUMINAMATH_CALUDE_nonagon_arithmetic_mean_property_l3400_340015

/-- Represents a vertex of the nonagon with its assigned number -/
structure Vertex where
  index : Fin 9
  value : Nat

/-- Checks if three vertices form an equilateral triangle in a regular nonagon -/
def isEquilateralTriangle (v1 v2 v3 : Vertex) : Prop :=
  (v2.index - v1.index) % 3 = 0 ∧ (v3.index - v2.index) % 3 = 0 ∧ (v1.index - v3.index) % 3 = 0

/-- Checks if one number is the arithmetic mean of the other two -/
def isArithmeticMean (a b c : Nat) : Prop :=
  2 * b = a + c

/-- The arrangement of numbers on the nonagon -/
def arrangement : List Vertex :=
  List.map (fun i => ⟨i, 2016 + i⟩) (List.range 9)

/-- The main theorem to prove -/
theorem nonagon_arithmetic_mean_property :
  ∀ v1 v2 v3 : Vertex,
    v1 ∈ arrangement →
    v2 ∈ arrangement →
    v3 ∈ arrangement →
    isEquilateralTriangle v1 v2 v3 →
    isArithmeticMean v1.value v2.value v3.value ∨
    isArithmeticMean v2.value v3.value v1.value ∨
    isArithmeticMean v3.value v1.value v2.value :=
  sorry

end NUMINAMATH_CALUDE_nonagon_arithmetic_mean_property_l3400_340015


namespace NUMINAMATH_CALUDE_new_person_weight_is_68_l3400_340090

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 68 kg -/
theorem new_person_weight_is_68 :
  new_person_weight 6 3.5 47 = 68 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_68_l3400_340090


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3400_340064

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → n + (n + 1) ≥ m + (m + 1)) → 
  n + (n + 1) = 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3400_340064


namespace NUMINAMATH_CALUDE_min_sum_xy_l3400_340097

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y - x*y = 0) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_xy_l3400_340097


namespace NUMINAMATH_CALUDE_intersection_M_N_l3400_340085

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x | x ≥ 3 ∨ x ≤ -2}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3400_340085


namespace NUMINAMATH_CALUDE_intersection_implies_a_greater_than_one_l3400_340036

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a}

def B (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = b^p.1 + 1}

-- State the theorem
theorem intersection_implies_a_greater_than_one 
  (a b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (A a ∩ B b).Nonempty → a > 1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_a_greater_than_one_l3400_340036


namespace NUMINAMATH_CALUDE_power_function_properties_l3400_340008

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^m

theorem power_function_properties :
  ∃ (m : ℝ), ∀ (x : ℝ), f m x = x^2 ∧
  ∀ (k : ℝ),
    (∀ (x : ℝ), x < 2 ∨ x > k → f m x > (k + 2) * x - 2 * k) ∧
    (k = 2 → ∀ (x : ℝ), x ≠ 2 → f m x > (k + 2) * x - 2 * k) ∧
    (k < 2 → ∀ (x : ℝ), x < k ∨ x > 2 → f m x > (k + 2) * x - 2 * k) :=
by sorry

end NUMINAMATH_CALUDE_power_function_properties_l3400_340008


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3400_340086

theorem trigonometric_identities :
  (∃ (x y : ℝ), 
    x = Real.sin (-14 * Real.pi / 3) + Real.cos (20 * Real.pi / 3) + Real.tan (-53 * Real.pi / 6) ∧
    x = (-3 - Real.sqrt 3) / 6 ∧
    y = Real.tan (675 * Real.pi / 180) - Real.sin (-330 * Real.pi / 180) - Real.cos (960 * Real.pi / 180) ∧
    y = -2) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3400_340086


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3400_340038

theorem arithmetic_sequence_terms (a₁ l d : ℤ) (h₁ : a₁ = 165) (h₂ : l = 30) (h₃ : d = -5) :
  ∃ n : ℕ, n = 28 ∧ l = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3400_340038


namespace NUMINAMATH_CALUDE_sqrt_sum_equal_l3400_340050

theorem sqrt_sum_equal : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equal_l3400_340050


namespace NUMINAMATH_CALUDE_converse_of_square_inequality_l3400_340054

theorem converse_of_square_inequality :
  (∀ a b : ℝ, a > b → a^2 > b^2) →
  (∀ a b : ℝ, a^2 > b^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_converse_of_square_inequality_l3400_340054


namespace NUMINAMATH_CALUDE_condition_relationship_l3400_340076

theorem condition_relationship (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l3400_340076


namespace NUMINAMATH_CALUDE_probability_of_blue_is_four_thirteenths_l3400_340022

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue

/-- Calculates the probability of selecting a blue jelly bean -/
def probabilityOfBlue (bag : JellyBeanBag) : ℚ :=
  bag.blue / (totalJellyBeans bag)

/-- Theorem: The probability of selecting a blue jelly bean from the given bag is 4/13 -/
theorem probability_of_blue_is_four_thirteenths :
  let bag : JellyBeanBag := { red := 5, green := 6, yellow := 7, blue := 8 }
  probabilityOfBlue bag = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blue_is_four_thirteenths_l3400_340022


namespace NUMINAMATH_CALUDE_chocolate_candies_cost_l3400_340082

/-- The cost of buying a specific number of chocolate candies -/
theorem chocolate_candies_cost
  (candies_per_box : ℕ)
  (cost_per_box : ℚ)
  (total_candies : ℕ)
  (h1 : candies_per_box = 30)
  (h2 : cost_per_box = 7.5)
  (h3 : total_candies = 450) :
  (total_candies / candies_per_box : ℚ) * cost_per_box = 112.5 :=
sorry

end NUMINAMATH_CALUDE_chocolate_candies_cost_l3400_340082


namespace NUMINAMATH_CALUDE_closest_result_is_180_l3400_340009

theorem closest_result_is_180 (options : List ℝ := [160, 180, 190, 200, 240]) : 
  let result := (0.000345 * 7650000) / 15
  options.argmin (λ x => |x - result|) = some 180 := by
  sorry

end NUMINAMATH_CALUDE_closest_result_is_180_l3400_340009


namespace NUMINAMATH_CALUDE_reunion_handshakes_l3400_340096

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a reunion of 11 boys where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 55 -/
theorem reunion_handshakes : handshakes 11 = 55 := by
  sorry

end NUMINAMATH_CALUDE_reunion_handshakes_l3400_340096


namespace NUMINAMATH_CALUDE_probability_at_least_one_first_class_l3400_340028

theorem probability_at_least_one_first_class (total : Nat) (first_class : Nat) (second_class : Nat) (selected : Nat) :
  total = 6 →
  first_class = 4 →
  second_class = 2 →
  selected = 2 →
  (1 : ℚ) - (Nat.choose second_class selected : ℚ) / (Nat.choose total selected : ℚ) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_first_class_l3400_340028


namespace NUMINAMATH_CALUDE_initial_boys_count_l3400_340070

/-- Given an initial group of boys with an average weight of 102 kg,
    adding a new person weighing 40 kg reduces the average by 2 kg.
    This function calculates the initial number of boys. -/
def initial_number_of_boys : ℕ :=
  let initial_avg : ℚ := 102
  let new_person_weight : ℚ := 40
  let avg_decrease : ℚ := 2
  let n : ℕ := 30  -- The number we want to prove
  n

/-- Theorem stating that the initial number of boys is 30 -/
theorem initial_boys_count :
  let n := initial_number_of_boys
  let initial_avg : ℚ := 102
  let new_person_weight : ℚ := 40
  let avg_decrease : ℚ := 2
  (n : ℚ) * initial_avg + new_person_weight = (n + 1) * (initial_avg - avg_decrease) :=
by sorry

end NUMINAMATH_CALUDE_initial_boys_count_l3400_340070


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3400_340095

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (2, 1) →
  a • b = 10 →
  ‖a + b‖ = 5 * Real.sqrt 2 →
  ‖b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3400_340095


namespace NUMINAMATH_CALUDE_town_population_l3400_340024

theorem town_population (pet_owners_percentage : Real) 
  (dog_owners_fraction : Real) (cat_owners : ℕ) :
  pet_owners_percentage = 0.6 →
  dog_owners_fraction = 0.5 →
  cat_owners = 30 →
  (cat_owners : Real) / (1 - dog_owners_fraction) / pet_owners_percentage = 100 :=
by sorry

end NUMINAMATH_CALUDE_town_population_l3400_340024


namespace NUMINAMATH_CALUDE_certain_yellow_ball_pick_l3400_340016

theorem certain_yellow_ball_pick (total_balls : ℕ) (red_balls : ℕ) (yellow_balls : ℕ) (m : ℕ) : 
  total_balls = 8 →
  red_balls = 3 →
  yellow_balls = 5 →
  total_balls = red_balls + yellow_balls →
  m ≤ red_balls →
  yellow_balls = total_balls - m →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_yellow_ball_pick_l3400_340016


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3400_340099

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3400_340099


namespace NUMINAMATH_CALUDE_system_two_solutions_l3400_340032

open Real

-- Define the system of equations
def equation1 (a x y : ℝ) : Prop :=
  arcsin ((a + y) / 2) = arcsin ((x + 3) / 3)

def equation2 (b x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x + 6*y = b

-- Define the condition for exactly two solutions
def hasTwoSolutions (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    equation1 a x₁ y₁ ∧ equation1 a x₂ y₂ ∧
    equation2 b x₁ y₁ ∧ equation2 b x₂ y₂ ∧
    ∀ (x y : ℝ), equation1 a x y ∧ equation2 b x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)

-- Theorem statement
theorem system_two_solutions (a : ℝ) :
  (∃ b, hasTwoSolutions a b) ↔ -7/2 < a ∧ a < 19/2 :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3400_340032


namespace NUMINAMATH_CALUDE_john_steve_race_l3400_340041

theorem john_steve_race (john_speed steve_speed : ℝ) (final_push_time : ℝ) (finish_ahead : ℝ) :
  john_speed = 4.2 →
  steve_speed = 3.8 →
  final_push_time = 42.5 →
  finish_ahead = 2 →
  john_speed * final_push_time - steve_speed * final_push_time - finish_ahead = 15 := by
  sorry

end NUMINAMATH_CALUDE_john_steve_race_l3400_340041


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3400_340068

/-- The ratio of wire lengths for equivalent volume cubes -/
theorem wire_length_ratio (large_cube_edge : ℝ) (small_cube_edge : ℝ) : 
  large_cube_edge = 8 →
  small_cube_edge = 2 →
  (12 * large_cube_edge) / (12 * small_cube_edge * (large_cube_edge / small_cube_edge)^3) = 1/16 := by
  sorry

#check wire_length_ratio

end NUMINAMATH_CALUDE_wire_length_ratio_l3400_340068


namespace NUMINAMATH_CALUDE_carrot_sticks_leftover_l3400_340000

theorem carrot_sticks_leftover (total_carrots : ℕ) (num_people : ℕ) (h1 : total_carrots = 74) (h2 : num_people = 12) :
  total_carrots % num_people = 2 := by
  sorry

end NUMINAMATH_CALUDE_carrot_sticks_leftover_l3400_340000


namespace NUMINAMATH_CALUDE_f_properties_l3400_340056

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

/-- Theorem stating the properties of f(x) and its extreme values -/
theorem f_properties :
  ∀ a b : ℝ,
  (∀ x : ℝ, f' a b (x + 1/2) = f' a b (-x + 1/2)) →  -- f'(x) is symmetric about x = -1/2
  f' a b 1 = 0 →                                     -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                                  -- Values of a and b
  f a b (-2) = 21 ∧                                  -- Local maximum
  f a b 1 = -6 ∧                                     -- Local minimum
  (∀ x : ℝ, x < -2 → f' a b x > 0) ∧                 -- f(x) increasing on (-∞, -2)
  (∀ x : ℝ, -2 < x ∧ x < 1 → f' a b x < 0) ∧         -- f(x) decreasing on (-2, 1)
  (∀ x : ℝ, x > 1 → f' a b x > 0)                    -- f(x) increasing on (1, ∞)
  := by sorry


end NUMINAMATH_CALUDE_f_properties_l3400_340056


namespace NUMINAMATH_CALUDE_george_first_half_correct_l3400_340078

def trivia_game (first_half_correct : ℕ) (second_half_correct : ℕ) (points_per_question : ℕ) (total_score : ℕ) : Prop :=
  first_half_correct * points_per_question + second_half_correct * points_per_question = total_score

theorem george_first_half_correct :
  ∃ (x : ℕ), trivia_game x 4 3 30 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_george_first_half_correct_l3400_340078


namespace NUMINAMATH_CALUDE_fair_rides_l3400_340089

theorem fair_rides (total_tickets : ℕ) (spent_tickets : ℕ) (ride_cost : ℕ) : 
  total_tickets = 79 → spent_tickets = 23 → ride_cost = 7 → 
  (total_tickets - spent_tickets) / ride_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_fair_rides_l3400_340089


namespace NUMINAMATH_CALUDE_original_workers_count_l3400_340080

/-- Represents the number of days required to complete the work. -/
def original_days : ℕ := 12

/-- Represents the number of days saved after additional workers joined. -/
def days_saved : ℕ := 4

/-- Represents the number of additional workers who joined. -/
def additional_workers : ℕ := 5

/-- Represents the number of additional workers working at twice the original rate. -/
def double_rate_workers : ℕ := 3

/-- Represents the number of additional workers working at the original rate. -/
def normal_rate_workers : ℕ := 2

/-- Theorem stating that the original number of workers is 16. -/
theorem original_workers_count : ℕ := by
  sorry

#check original_workers_count

end NUMINAMATH_CALUDE_original_workers_count_l3400_340080


namespace NUMINAMATH_CALUDE_miss_walter_stickers_l3400_340091

theorem miss_walter_stickers (gold : ℕ) (silver : ℕ) (bronze : ℕ) (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : silver = 2 * gold)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver - bronze = 20 := by
  sorry

end NUMINAMATH_CALUDE_miss_walter_stickers_l3400_340091


namespace NUMINAMATH_CALUDE_sandy_siblings_l3400_340002

def total_tokens : ℕ := 1000000
def sandy_share : ℕ := total_tokens / 2
def extra_tokens : ℕ := 375000

theorem sandy_siblings :
  ∃ (num_siblings : ℕ),
    num_siblings > 0 ∧
    sandy_share = (total_tokens - sandy_share) / num_siblings + extra_tokens ∧
    num_siblings = 4 :=
by sorry

end NUMINAMATH_CALUDE_sandy_siblings_l3400_340002


namespace NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l3400_340067

/-- A regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_144_degrees_has_10_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 2 : ℝ) * 180 / n = 144 →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l3400_340067


namespace NUMINAMATH_CALUDE_min_abs_sum_l3400_340061

theorem min_abs_sum (x : ℝ) : 
  ∀ a : ℝ, (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) → a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_l3400_340061


namespace NUMINAMATH_CALUDE_no_positive_lower_bound_l3400_340088

/-- The number of positive integers not containing the digit 9 that are less than or equal to n -/
def f (n : ℕ+) : ℕ := sorry

/-- For any positive real number p, there exists a positive integer n such that f(n)/n < p -/
theorem no_positive_lower_bound :
  ∀ p : ℝ, p > 0 → ∃ n : ℕ+, (f n : ℝ) / n < p :=
sorry

end NUMINAMATH_CALUDE_no_positive_lower_bound_l3400_340088


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3400_340060

theorem max_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b * c + a + c = b) :
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1) ≤ 26 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3400_340060


namespace NUMINAMATH_CALUDE_complex_equation_imag_part_l3400_340010

theorem complex_equation_imag_part : 
  ∃ (z : ℂ), (3 - 4*I) * z = Complex.abs (4 + 3*I) ∧ z.im = 4/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_imag_part_l3400_340010


namespace NUMINAMATH_CALUDE_circle_radius_problem_l3400_340018

theorem circle_radius_problem (r : ℝ) : 
  3 * (2 * Real.pi * r) + 6 = 2 * (Real.pi * r^2) → 
  r = (3 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l3400_340018


namespace NUMINAMATH_CALUDE_sum_of_digits_special_product_l3400_340013

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_special_product (m n : ℕ) (d : ℕ) :
  m > 0 → n > 0 → d > 0 → d ≤ n → d = (Nat.digits 10 m).length →
  sum_of_digits ((10^n - 1) * m) = 9 * n := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_special_product_l3400_340013


namespace NUMINAMATH_CALUDE_bakers_remaining_cakes_l3400_340084

theorem bakers_remaining_cakes 
  (initial_cakes : ℝ) 
  (bought_cakes : ℝ) 
  (h1 : initial_cakes = 397.5) 
  (h2 : bought_cakes = 289) : 
  initial_cakes - bought_cakes = 108.5 := by
sorry

end NUMINAMATH_CALUDE_bakers_remaining_cakes_l3400_340084


namespace NUMINAMATH_CALUDE_smallest_other_integer_l3400_340062

theorem smallest_other_integer (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  Nat.gcd m n = x + 6 →
  Nat.lcm m n = x * (x + 6) →
  m = 60 →
  (∀ k : ℕ, k > 0 ∧ k < n → 
    (Nat.gcd 60 k ≠ x + 6 ∨ Nat.lcm 60 k ≠ x * (x + 6))) →
  n = 93 := by
sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l3400_340062


namespace NUMINAMATH_CALUDE_arun_weight_average_l3400_340073

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72)
  (h2 : 60 < w ∧ w < 70)
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := by sorry

end NUMINAMATH_CALUDE_arun_weight_average_l3400_340073


namespace NUMINAMATH_CALUDE_misread_weight_l3400_340042

theorem misread_weight (class_size : ℕ) (incorrect_avg : ℚ) (correct_avg : ℚ) (correct_weight : ℚ) (x : ℚ) :
  class_size = 20 →
  incorrect_avg = 58.4 →
  correct_avg = 58.85 →
  correct_weight = 65 →
  class_size * correct_avg = class_size * incorrect_avg - x + correct_weight →
  x = 56 := by
sorry

end NUMINAMATH_CALUDE_misread_weight_l3400_340042


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3400_340019

/-- Proof that when m(m-1) + mi is purely imaginary, m = 1 -/
theorem purely_imaginary_condition (m : ℝ) : 
  (m * (m - 1) : ℂ) + m * Complex.I = Complex.I * (r : ℝ) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3400_340019


namespace NUMINAMATH_CALUDE_specific_function_value_l3400_340011

def is_odd_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 2) = f x)

def agrees_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → f x = x

theorem specific_function_value
  (f : ℝ → ℝ)
  (h_odd_periodic : is_odd_periodic f)
  (h_unit : agrees_on_unit_interval f) :
  f 2011.5 = -0.5 := by
sorry

end NUMINAMATH_CALUDE_specific_function_value_l3400_340011


namespace NUMINAMATH_CALUDE_sum4_is_27857_l3400_340043

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum3 : a + a*r + a*r^2 = 13
  sum5 : a + a*r + a*r^2 + a*r^3 + a*r^4 = 121

/-- The sum of the first 4 terms of the geometric sequence -/
def sum4 (seq : GeometricSequence) : ℝ :=
  seq.a + seq.a * seq.r + seq.a * seq.r^2 + seq.a * seq.r^3

/-- Theorem stating that the sum of the first 4 terms is 27.857 -/
theorem sum4_is_27857 (seq : GeometricSequence) : sum4 seq = 27.857 := by
  sorry

end NUMINAMATH_CALUDE_sum4_is_27857_l3400_340043


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l3400_340079

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂ 
    such that |⃗P₁P| = 2|⃗PP₂|, prove that P has coordinates (-2, 11). -/
theorem extension_point_coordinates (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) → 
  P₂ = (0, 5) → 
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) → 
  ‖P - P₁‖ = 2 * ‖P₂ - P‖ → 
  P = (-2, 11) := by
  sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l3400_340079


namespace NUMINAMATH_CALUDE_equation_solutions_l3400_340051

theorem equation_solutions :
  (∀ x : ℝ, 5 * x^2 - 10 = 0 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2) ∧
  (∀ x : ℝ, 3 * (x - 4)^2 = 375 ↔ x = 4 + 5 * Real.sqrt 5 ∨ x = 4 - 5 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3400_340051


namespace NUMINAMATH_CALUDE_least_common_period_is_24_l3400_340005

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) + f (x - 4) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriod f p ∧ ∀ q, IsPeriod f q → p ≤ q

theorem least_common_period_is_24 :
  ∃ p : ℝ, p = 24 ∧
    (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f p) ∧
    (∀ q : ℝ, (∀ f : ℝ → ℝ, FunctionalEquation f → IsLeastPeriod f q) → p ≤ q) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_24_l3400_340005


namespace NUMINAMATH_CALUDE_ad_sequence_count_l3400_340074

/-- Represents the number of Olympic ads -/
def num_olympic_ads : ℕ := 3

/-- Represents the number of commercial ads -/
def num_commercial_ads : ℕ := 2

/-- Represents the total number of ads -/
def total_ads : ℕ := num_olympic_ads + num_commercial_ads

/-- Represents the constraint that the last ad must be an Olympic ad -/
def last_ad_is_olympic : Prop := true

/-- Represents the constraint that commercial ads cannot be played consecutively -/
def no_consecutive_commercial_ads : Prop := true

/-- The number of different playback sequences -/
def num_sequences : ℕ := 36

theorem ad_sequence_count :
  num_olympic_ads = 3 →
  num_commercial_ads = 2 →
  total_ads = 5 →
  last_ad_is_olympic →
  no_consecutive_commercial_ads →
  num_sequences = 36 :=
by sorry

end NUMINAMATH_CALUDE_ad_sequence_count_l3400_340074


namespace NUMINAMATH_CALUDE_prob_less_than_130_l3400_340049

-- Define the normal distribution
def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define the cumulative distribution function (CDF) for the normal distribution
def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry

-- Define the probability of a score being within μ ± kσ
def prob_within_k_sigma (k : ℝ) : ℝ := sorry

-- Theorem to prove
theorem prob_less_than_130 :
  let μ : ℝ := 110
  let σ : ℝ := 20
  ∃ ε > 0, |normal_cdf μ σ 130 - 0.97725| < ε :=
sorry

end NUMINAMATH_CALUDE_prob_less_than_130_l3400_340049


namespace NUMINAMATH_CALUDE_constant_b_value_l3400_340004

theorem constant_b_value (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5/2) * (a * x^2 + b * x + c) = 
    6 * x^4 - 17 * x^3 + 11 * x^2 - 7/2 * x + 5/3) →
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_constant_b_value_l3400_340004


namespace NUMINAMATH_CALUDE_unique_number_theorem_l3400_340075

/-- A function that checks if a number n can be expressed as 2a + xb 
    where a and b are positive integers --/
def isExpressible (n x : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = 2 * a + x * b

/-- The main theorem stating that 5 is the unique number satisfying the condition --/
theorem unique_number_theorem :
  ∃! (x : ℕ), x > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = 8 ∧ 
    (∀ n ∈ S, n < 15 ∧ isExpressible n x) ∧
    (∀ n < 15, isExpressible n x → n ∈ S)) ∧
  x = 5 := by
  sorry


end NUMINAMATH_CALUDE_unique_number_theorem_l3400_340075


namespace NUMINAMATH_CALUDE_perpendicular_angle_values_l3400_340021

theorem perpendicular_angle_values (α : Real) : 
  (4 * Real.pi < α ∧ α < 6 * Real.pi) →
  (∃ k : ℤ, α = -Real.pi/6 + k * Real.pi) →
  (α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_angle_values_l3400_340021


namespace NUMINAMATH_CALUDE_intersection_with_y_axis_l3400_340053

/-- The intersection point of the line y = 5x + 1 with the y-axis is (0, 1) -/
theorem intersection_with_y_axis :
  let f : ℝ → ℝ := λ x ↦ 5 * x + 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_with_y_axis_l3400_340053


namespace NUMINAMATH_CALUDE_unbounded_digit_sum_ratio_l3400_340045

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- State the theorem
theorem unbounded_digit_sum_ratio :
  ∀ c : ℝ, c > 0 → ∃ n : ℕ, (sum_of_digits n : ℝ) / (sum_of_digits (n^2)) > c :=
sorry

end NUMINAMATH_CALUDE_unbounded_digit_sum_ratio_l3400_340045


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3400_340026

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ 
  (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3400_340026


namespace NUMINAMATH_CALUDE_log_inequality_relationship_l3400_340055

theorem log_inequality_relationship (a b : ℝ) :
  (∀ a b, Real.log a > Real.log b → a > b) ∧
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_relationship_l3400_340055


namespace NUMINAMATH_CALUDE_dealership_anticipation_l3400_340066

/-- Given a ratio of SUVs to trucks and an expected number of SUVs,
    calculate the anticipated number of trucks -/
def anticipatedTrucks (suvRatio truckRatio expectedSUVs : ℕ) : ℕ :=
  (expectedSUVs * truckRatio) / suvRatio

/-- Theorem: Given the ratio of SUVs to trucks is 3:5,
    if 45 SUVs are expected to be sold,
    then 75 trucks are anticipated to be sold -/
theorem dealership_anticipation :
  anticipatedTrucks 3 5 45 = 75 := by
  sorry

end NUMINAMATH_CALUDE_dealership_anticipation_l3400_340066


namespace NUMINAMATH_CALUDE_multiple_of_17_binary_properties_l3400_340027

/-- A function that returns the number of 1's in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- A function that returns the number of 0's in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem multiple_of_17_binary_properties (n : ℕ) 
  (h1 : n % 17 = 0) 
  (h2 : count_ones n = 3) : 
  (count_zeros n ≥ 6) ∧ 
  (count_zeros n = 7 → Even n) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_17_binary_properties_l3400_340027


namespace NUMINAMATH_CALUDE_smallest_T_value_l3400_340072

theorem smallest_T_value : ∃ (m : ℕ), 
  (∀ k : ℕ, k < m → 8 * k < 2400) ∧ 
  8 * m ≥ 2400 ∧
  9 * m - 2400 = 300 := by
  sorry

end NUMINAMATH_CALUDE_smallest_T_value_l3400_340072


namespace NUMINAMATH_CALUDE_south_american_stamps_cost_l3400_340033

/-- Represents a country in Maria's stamp collection. -/
inductive Country
| Brazil
| Peru
| France
| Spain

/-- Represents a decade in which stamps were issued. -/
inductive Decade
| Fifties
| Sixties
| Nineties

/-- The cost of a stamp in cents for a given country. -/
def stampCost (c : Country) : ℕ :=
  match c with
  | Country.Brazil => 7
  | Country.Peru => 5
  | Country.France => 7
  | Country.Spain => 6

/-- Whether a country is in South America. -/
def isSouthAmerican (c : Country) : Bool :=
  match c with
  | Country.Brazil => true
  | Country.Peru => true
  | _ => false

/-- The number of stamps Maria has for a given country and decade. -/
def stampCount (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Brazil, Decade.Fifties => 6
  | Country.Brazil, Decade.Sixties => 9
  | Country.Peru, Decade.Fifties => 8
  | Country.Peru, Decade.Sixties => 6
  | _, _ => 0

/-- The total cost of stamps for a given country and decade, in cents. -/
def decadeCost (c : Country) (d : Decade) : ℕ :=
  stampCost c * stampCount c d

/-- The theorem stating the total cost of South American stamps issued before the 90s. -/
theorem south_american_stamps_cost :
  (decadeCost Country.Brazil Decade.Fifties +
   decadeCost Country.Brazil Decade.Sixties +
   decadeCost Country.Peru Decade.Fifties +
   decadeCost Country.Peru Decade.Sixties) = 175 := by
  sorry


end NUMINAMATH_CALUDE_south_american_stamps_cost_l3400_340033


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3400_340003

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side lengths and angles
def sideLength (p q : ℝ × ℝ) : ℝ := sorry
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_side_lengths (t : Triangle) :
  sideLength t.A t.C = 5 →
  sideLength t.B t.C - sideLength t.A t.B = 2 →
  angle t.C t.A t.B = 2 * angle t.A t.C t.B →
  sideLength t.A t.B = 4 ∧ sideLength t.B t.C = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3400_340003


namespace NUMINAMATH_CALUDE_negation_of_implication_l3400_340065

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2015 → x > 0) ↔ (x ≤ 2015 → x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3400_340065


namespace NUMINAMATH_CALUDE_sun_division_problem_l3400_340069

/-- Proves that the total amount is 156 rupees given the conditions of the problem -/
theorem sun_division_problem (x y z : ℝ) : 
  (∀ (r : ℝ), r > 0 → y = 0.45 * r ∧ z = 0.5 * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 50 paisa
  y = 36 →  -- y's share is Rs. 36
  x + y + z = 156 := by  -- The total amount is Rs. 156
sorry

end NUMINAMATH_CALUDE_sun_division_problem_l3400_340069


namespace NUMINAMATH_CALUDE_landscape_ratio_l3400_340037

theorem landscape_ratio (length : ℝ) (playground_area : ℝ) (playground_ratio : ℝ) :
  length = 240 →
  playground_area = 1200 →
  playground_ratio = 1 / 6 →
  ∃ breadth : ℝ, breadth > 0 ∧ length / breadth = 8 := by
  sorry

end NUMINAMATH_CALUDE_landscape_ratio_l3400_340037


namespace NUMINAMATH_CALUDE_cylinder_dihedral_angle_l3400_340030

-- Define the cylinder and its properties
structure Cylinder where
  A : Point
  A₁ : Point
  B : Point
  B₁ : Point
  C : Point
  α : Real  -- dihedral angle
  β : Real  -- ∠CAB
  γ : Real  -- ∠CA₁B

-- Define the theorem
theorem cylinder_dihedral_angle (cyl : Cylinder) :
  cyl.α = Real.arcsin (Real.cos cyl.β / Real.cos cyl.γ) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_dihedral_angle_l3400_340030


namespace NUMINAMATH_CALUDE_betsy_games_won_l3400_340040

theorem betsy_games_won (betsy helen susan : ℕ) 
  (helen_games : helen = 2 * betsy)
  (susan_games : susan = 3 * betsy)
  (total_games : betsy + helen + susan = 30) :
  betsy = 5 := by
sorry

end NUMINAMATH_CALUDE_betsy_games_won_l3400_340040


namespace NUMINAMATH_CALUDE_successive_discounts_l3400_340012

theorem successive_discounts (P d1 d2 : ℝ) (h1 : 0 ≤ d1 ∧ d1 < 1) (h2 : 0 ≤ d2 ∧ d2 < 1) :
  let final_price := P * (1 - d1) * (1 - d2)
  let percentage := (final_price / P) * 100
  percentage = (1 - d1) * (1 - d2) * 100 :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_l3400_340012


namespace NUMINAMATH_CALUDE_no_line_bisected_by_P_intersects_hyperbola_l3400_340059

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The hyperbola equation -/
def isOnHyperbola (p : Point) : Prop :=
  p.x^2 / 9 - p.y^2 / 4 = 1

/-- Check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- Check if a point bisects a line segment -/
def isMidpoint (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  p.x = (p1.x + p2.x) / 2 ∧ p.y = (p1.y + p2.y) / 2

/-- The main theorem -/
theorem no_line_bisected_by_P_intersects_hyperbola :
  ¬ ∃ (l : Line) (p1 p2 : Point),
    p1 ≠ p2 ∧
    isOnHyperbola p1 ∧
    isOnHyperbola p2 ∧
    isOnLine p1 l ∧
    isOnLine p2 l ∧
    isMidpoint ⟨2, 1⟩ p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_no_line_bisected_by_P_intersects_hyperbola_l3400_340059


namespace NUMINAMATH_CALUDE_point_A_not_on_transformed_plane_l3400_340031

/-- The similarity transformation coefficient -/
def k : ℚ := 2/3

/-- The original plane equation -/
def plane_a (x y z : ℚ) : Prop := 5*x + y - z + 6 = 0

/-- The transformed plane equation -/
def plane_a' (x y z : ℚ) : Prop := 5*x + y - z + 4 = 0

/-- The point A -/
def point_A : ℚ × ℚ × ℚ := (1, -2, 1)

/-- Theorem stating that point A is not on the transformed plane -/
theorem point_A_not_on_transformed_plane : 
  ¬ plane_a' point_A.1 point_A.2.1 point_A.2.2 :=
sorry

end NUMINAMATH_CALUDE_point_A_not_on_transformed_plane_l3400_340031


namespace NUMINAMATH_CALUDE_sixth_root_of_594823321_l3400_340014

theorem sixth_root_of_594823321 : (594823321 : ℝ) ^ (1/6 : ℝ) = 51 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_594823321_l3400_340014


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3400_340023

def f (x : ℝ) : ℝ := 5 * x + 3

theorem inequality_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 2| < b → |f x + 7| < a) ↔ b ≤ a / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3400_340023


namespace NUMINAMATH_CALUDE_xavier_probability_of_success_l3400_340057

theorem xavier_probability_of_success 
  (p_yvonne : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_and_yvonne_not_zelda : ℝ) 
  (h1 : p_yvonne = 2/3) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_and_yvonne_not_zelda = 0.0625) :
  ∃ p_xavier : ℝ, 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_and_yvonne_not_zelda ∧ 
    p_xavier = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_xavier_probability_of_success_l3400_340057


namespace NUMINAMATH_CALUDE_esteban_exercise_time_l3400_340063

/-- Proves that Esteban exercised for 10 minutes each day given the conditions. -/
theorem esteban_exercise_time :
  -- Natasha's exercise time per day in minutes
  let natasha_daily := 30
  -- Number of days Natasha exercised
  let natasha_days := 7
  -- Number of days Esteban exercised
  let esteban_days := 9
  -- Total exercise time for both in hours
  let total_hours := 5
  -- Calculate Esteban's daily exercise time in minutes
  let esteban_daily := 
    (total_hours * 60 - natasha_daily * natasha_days) / esteban_days
  -- Prove that Esteban's daily exercise time is 10 minutes
  esteban_daily = 10 := by
  sorry

end NUMINAMATH_CALUDE_esteban_exercise_time_l3400_340063


namespace NUMINAMATH_CALUDE_monkey_climb_time_25m_l3400_340029

/-- Represents the time taken for a monkey to climb a greased pole -/
def monkey_climb_time (pole_height : ℕ) (ascend_rate : ℕ) (slip_rate : ℕ) : ℕ :=
  let full_cycles := (pole_height - ascend_rate) / (ascend_rate - slip_rate)
  full_cycles * 2 + 1

/-- Theorem stating that it takes 45 minutes for the monkey to climb the pole -/
theorem monkey_climb_time_25m :
  monkey_climb_time 25 3 2 = 45 := by sorry

end NUMINAMATH_CALUDE_monkey_climb_time_25m_l3400_340029


namespace NUMINAMATH_CALUDE_multiplication_equation_solution_l3400_340071

theorem multiplication_equation_solution : 
  ∃ x : ℕ, 18396 * x = 183868020 ∧ x = 9990 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equation_solution_l3400_340071


namespace NUMINAMATH_CALUDE_interval_condition_l3400_340087

theorem interval_condition (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 5 ∧ 2 < 5*x ∧ 5*x < 5) ↔ (1/2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_interval_condition_l3400_340087


namespace NUMINAMATH_CALUDE_pie_remainder_l3400_340083

theorem pie_remainder (carlos_portion jessica_portion : Real) : 
  carlos_portion = 0.6 →
  jessica_portion = 0.25 * (1 - carlos_portion) →
  1 - carlos_portion - jessica_portion = 0.3 := by
sorry

end NUMINAMATH_CALUDE_pie_remainder_l3400_340083


namespace NUMINAMATH_CALUDE_positive_square_iff_greater_l3400_340093

theorem positive_square_iff_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a^2 > b^2 := by sorry

end NUMINAMATH_CALUDE_positive_square_iff_greater_l3400_340093


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l3400_340046

/-- Represents the time difference between two dates in hours -/
def timeDifference : ℚ := 189

/-- Represents the daily loss rate of the watch in minutes per day -/
def dailyLossRate : ℚ := 13 / 4

/-- Calculates the positive correction in minutes to be added to the watch -/
def watchCorrection (timeDiff : ℚ) (lossRate : ℚ) : ℚ :=
  timeDiff * (lossRate / 24)

/-- Theorem stating that the watch correction is 2457/96 minutes -/
theorem watch_correction_theorem :
  watchCorrection timeDifference dailyLossRate = 2457 / 96 := by
  sorry

#eval watchCorrection timeDifference dailyLossRate

end NUMINAMATH_CALUDE_watch_correction_theorem_l3400_340046


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3400_340081

theorem smallest_integer_solution : 
  ∃ x : ℤ, (x ≥ 0) ∧ 
    (⌊x / 8⌋ - ⌊x / 40⌋ + ⌊x / 240⌋ = 210) ∧ 
    (∀ y : ℤ, y ≥ 0 → ⌊y / 8⌋ - ⌊y / 40⌋ + ⌊y / 240⌋ = 210 → y ≥ x) ∧
    x = 2016 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3400_340081


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l3400_340094

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Conditions for jelly sales -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 2 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.raspberry = s.grape / 3 ∧
  s.plum = 6

theorem strawberry_jelly_sales (s : JellySales) :
  valid_jelly_sales s → s.strawberry = 18 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jelly_sales_l3400_340094


namespace NUMINAMATH_CALUDE_gcd_459_357_f_neg_four_l3400_340034

-- Part 1: GCD of 459 and 357
theorem gcd_459_357 : Int.gcd 459 357 = 51 := by sorry

-- Part 2: Polynomial evaluation
def f (x : ℤ) : ℤ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem f_neg_four : f (-4) = 3392 := by sorry

end NUMINAMATH_CALUDE_gcd_459_357_f_neg_four_l3400_340034


namespace NUMINAMATH_CALUDE_steve_sleeping_time_l3400_340035

theorem steve_sleeping_time (total_hours : ℝ) (school_fraction : ℝ) (assignment_fraction : ℝ) (family_hours : ℝ)
  (h1 : total_hours = 24)
  (h2 : school_fraction = 1 / 6)
  (h3 : assignment_fraction = 1 / 12)
  (h4 : family_hours = 10) :
  (total_hours - (school_fraction * total_hours + assignment_fraction * total_hours + family_hours)) / total_hours = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_steve_sleeping_time_l3400_340035


namespace NUMINAMATH_CALUDE_factorization_proof_l3400_340077

theorem factorization_proof : 989 * 1001 * 1007 + 320 = 991 * 997 * 1009 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3400_340077


namespace NUMINAMATH_CALUDE_cost_price_is_36_l3400_340098

/-- Given the total cloth length, total selling price, and loss per metre, 
    calculate the cost price for one metre of cloth. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Theorem: The cost price for one metre of cloth is Rs. 36 given the problem conditions. -/
theorem cost_price_is_36 :
  cost_price_per_metre 300 9000 6 = 36 := by
  sorry

#eval cost_price_per_metre 300 9000 6

end NUMINAMATH_CALUDE_cost_price_is_36_l3400_340098
