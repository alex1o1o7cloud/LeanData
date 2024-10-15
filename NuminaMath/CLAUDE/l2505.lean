import Mathlib

namespace NUMINAMATH_CALUDE_total_seashells_l2505_250576

def mary_seashells : ℕ := 18
def jessica_seashells : ℕ := 41

theorem total_seashells : mary_seashells + jessica_seashells = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l2505_250576


namespace NUMINAMATH_CALUDE_lateral_surface_is_parallelogram_l2505_250559

-- Define the types for our geometric objects
inductive PrismType
| Right
| Oblique

-- Define the shapes we're considering
inductive Shape
| Rectangle
| Parallelogram

-- Define a function that returns the possible shapes of a prism's lateral surface
def lateralSurfaceShape (p : PrismType) : Set Shape :=
  match p with
  | PrismType.Right => {Shape.Rectangle}
  | PrismType.Oblique => {Shape.Rectangle, Shape.Parallelogram}

-- Theorem statement
theorem lateral_surface_is_parallelogram :
  ∀ (p : PrismType), ∃ (s : Shape), s ∈ lateralSurfaceShape p → s = Shape.Parallelogram := by
  sorry

#check lateral_surface_is_parallelogram

end NUMINAMATH_CALUDE_lateral_surface_is_parallelogram_l2505_250559


namespace NUMINAMATH_CALUDE_xiao_ming_math_grade_l2505_250579

/-- Calculates a student's semester math grade based on component scores and weights -/
def semesterMathGrade (routineStudyScore midTermScore finalExamScore : ℝ) : ℝ :=
  0.3 * routineStudyScore + 0.3 * midTermScore + 0.4 * finalExamScore

/-- Xiao Ming's semester math grade is 92.4 points -/
theorem xiao_ming_math_grade :
  semesterMathGrade 90 90 96 = 92.4 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_math_grade_l2505_250579


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2505_250598

/-- A function g(x) with specific properties -/
noncomputable def g (A B C : ℤ) : ℝ → ℝ := λ x => x^2 / (A * x^2 + B * x + C)

/-- Theorem stating the sum of coefficients A, B, and C is zero -/
theorem sum_of_coefficients_zero
  (A B C : ℤ)
  (h1 : ∀ x > 2, g A B C x > 0.3)
  (h2 : (A * 1^2 + B * 1 + C = 0) ∧ (A * (-3)^2 + B * (-3) + C = 0)) :
  A + B + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2505_250598


namespace NUMINAMATH_CALUDE_solution_range_l2505_250507

-- Define the system of linear equations
def system (x y a : ℝ) : Prop :=
  (3 * x + y = 1 + a) ∧ (x + 3 * y = 3)

-- Define the theorem
theorem solution_range (x y a : ℝ) :
  system x y a → x + y < 2 → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2505_250507


namespace NUMINAMATH_CALUDE_second_projectile_speed_l2505_250515

/-- Given two projectiles launched simultaneously from a distance apart, 
    with one traveling at a known speed, and both meeting after a certain time, 
    this theorem proves the speed of the second projectile. -/
theorem second_projectile_speed 
  (initial_distance : ℝ) 
  (speed_first : ℝ) 
  (time_to_meet : ℝ) 
  (h1 : initial_distance = 1998) 
  (h2 : speed_first = 444) 
  (h3 : time_to_meet = 2) : 
  ∃ (speed_second : ℝ), speed_second = 555 :=
by
  sorry

end NUMINAMATH_CALUDE_second_projectile_speed_l2505_250515


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_property_l2505_250514

theorem smallest_integer_with_divisibility_property : ∃ (n : ℕ), 
  (∀ i ∈ Finset.range 28, i.succ ∣ n) ∧ 
  ¬(29 ∣ n) ∧ 
  ¬(30 ∣ n) ∧
  (∀ m : ℕ, m < n → ¬(∀ i ∈ Finset.range 28, i.succ ∣ m) ∨ (29 ∣ m) ∨ (30 ∣ m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_property_l2505_250514


namespace NUMINAMATH_CALUDE_room_length_calculation_l2505_250545

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 ∧ total_cost = 16500 ∧ rate_per_sqm = 800 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2505_250545


namespace NUMINAMATH_CALUDE_ratio_of_sum_equals_three_times_difference_l2505_250527

theorem ratio_of_sum_equals_three_times_difference
  (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) (h4 : x + y = 3 * (x - y)) :
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_equals_three_times_difference_l2505_250527


namespace NUMINAMATH_CALUDE_power_three_fifteen_mod_five_l2505_250582

theorem power_three_fifteen_mod_five : 3^15 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_three_fifteen_mod_five_l2505_250582


namespace NUMINAMATH_CALUDE_remainder_problem_l2505_250538

theorem remainder_problem (N : ℕ) : 
  (N / 5 = 5) ∧ (N % 5 = 0) → N % 11 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2505_250538


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2505_250561

theorem unique_solution_to_equation (x : ℝ) : (x^2 + 4*x - 5)^0 = x^2 - 5*x + 5 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2505_250561


namespace NUMINAMATH_CALUDE_mappings_count_l2505_250512

theorem mappings_count (A B : Finset Char) :
  A = {('a' : Char), 'b'} →
  B = {('c' : Char), 'd'} →
  Fintype.card (A → B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mappings_count_l2505_250512


namespace NUMINAMATH_CALUDE_probability_two_absent_one_present_l2505_250539

/-- The probability of a student being absent on a given day -/
def p_absent : ℚ := 2 / 25

/-- The probability of a student being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of students chosen -/
def n_students : ℕ := 3

/-- The number of students that should be absent -/
def n_absent : ℕ := 2

-- Theorem statement
theorem probability_two_absent_one_present :
  (n_students.choose n_absent : ℚ) * p_absent ^ n_absent * p_present ^ (n_students - n_absent) = 276 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_absent_one_present_l2505_250539


namespace NUMINAMATH_CALUDE_min_sum_with_linear_constraint_l2505_250583

theorem min_sum_with_linear_constraint (a b : ℕ) (h : 23 * a - 13 * b = 1) :
  ∃ (a' b' : ℕ), 23 * a' - 13 * b' = 1 ∧ a' + b' ≤ a + b ∧ a' + b' = 11 :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_linear_constraint_l2505_250583


namespace NUMINAMATH_CALUDE_constant_c_value_l2505_250516

theorem constant_c_value : ∃ (d e c : ℝ), 
  (∀ x : ℝ, (6*x^2 - 2*x + 5/2)*(d*x^2 + e*x + c) = 18*x^4 - 9*x^3 + 13*x^2 - 7/2*x + 15/4) →
  c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l2505_250516


namespace NUMINAMATH_CALUDE_inequality_proof_l2505_250540

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) (hn : n > 0) : 
  (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) ≥ 
  3^n / (3^(n+2) - 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2505_250540


namespace NUMINAMATH_CALUDE_train_speed_is_88_l2505_250521

/-- Represents the transportation problem with train and ship --/
structure TransportProblem where
  rail_distance : ℝ
  river_distance : ℝ
  train_delay : ℝ
  train_arrival_diff : ℝ
  speed_difference : ℝ

/-- Calculates the train speed given the problem parameters --/
def calculate_train_speed (p : TransportProblem) : ℝ :=
  let train_time := p.rail_distance / x
  let ship_time := p.river_distance / (x - p.speed_difference)
  let time_diff := ship_time - train_time
  x
where
  x := 88 -- The solution we want to prove

/-- Theorem stating that the calculated train speed is correct --/
theorem train_speed_is_88 (p : TransportProblem) 
  (h1 : p.rail_distance = 88)
  (h2 : p.river_distance = 108)
  (h3 : p.train_delay = 1)
  (h4 : p.train_arrival_diff = 1/4)
  (h5 : p.speed_difference = 40) :
  calculate_train_speed p = 88 := by
  sorry

#eval calculate_train_speed { 
  rail_distance := 88, 
  river_distance := 108, 
  train_delay := 1, 
  train_arrival_diff := 1/4, 
  speed_difference := 40 
}

end NUMINAMATH_CALUDE_train_speed_is_88_l2505_250521


namespace NUMINAMATH_CALUDE_triangle_inequality_l2505_250569

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 / (b + c - a)) + (b^2 / (c + a - b)) + (c^2 / (a + b - c)) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2505_250569


namespace NUMINAMATH_CALUDE_projectile_max_height_l2505_250558

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 30

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 155

/-- Theorem stating that the maximum value of h(t) is equal to max_height -/
theorem projectile_max_height : 
  ∃ t, h t = max_height ∧ ∀ s, h s ≤ h t :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2505_250558


namespace NUMINAMATH_CALUDE_color_tv_price_l2505_250532

/-- The original price of a color TV -/
def original_price : ℝ := 1200

/-- The price after 40% increase -/
def increased_price (x : ℝ) : ℝ := x * (1 + 0.4)

/-- The final price after 20% discount -/
def final_price (x : ℝ) : ℝ := increased_price x * 0.8

theorem color_tv_price :
  final_price original_price - original_price = 144 := by sorry

end NUMINAMATH_CALUDE_color_tv_price_l2505_250532


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2505_250513

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2505_250513


namespace NUMINAMATH_CALUDE_digit_sum_of_power_product_l2505_250544

def power_product (a b c d e : ℕ) : ℕ := a^b * c^d * e

theorem digit_sum_of_power_product :
  ∃ (f : ℕ → ℕ), f (power_product 2 2010 5 2012 7) = 13 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_of_power_product_l2505_250544


namespace NUMINAMATH_CALUDE_existence_of_nth_root_l2505_250580

theorem existence_of_nth_root (n b : ℕ) (hn : n > 1) (hb : b > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a : ℤ, (k : ℤ) ∣ b - a^n) :
  ∃ A : ℤ, b = A^n := by
sorry

end NUMINAMATH_CALUDE_existence_of_nth_root_l2505_250580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2505_250565

/-- An arithmetic sequence {aₙ} with a₇ = 4 and a₁₉ = 2a₉ has the general term formula aₙ = (n+1)/2 -/
theorem arithmetic_sequence_formula (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_a7 : a 7 = 4)
  (h_a19 : a 19 = 2 * a 9) :
  ∀ n : ℕ, a n = (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2505_250565


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2505_250572

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2505_250572


namespace NUMINAMATH_CALUDE_fair_wall_painting_l2505_250557

theorem fair_wall_painting (people : ℕ) (rooms_type1 rooms_type2 : ℕ) 
  (walls_per_room_type1 walls_per_room_type2 : ℕ) :
  people = 5 →
  rooms_type1 = 5 →
  rooms_type2 = 4 →
  walls_per_room_type1 = 4 →
  walls_per_room_type2 = 5 →
  (rooms_type1 * walls_per_room_type1 + rooms_type2 * walls_per_room_type2) / people = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_fair_wall_painting_l2505_250557


namespace NUMINAMATH_CALUDE_circle_equation_l2505_250533

/-- Given two circles C1 and C2 where:
    1. C1 has equation (x-1)^2 + (y-1)^2 = 1
    2. The coordinate axes are common tangents of C1 and C2
    3. The distance between the centers of C1 and C2 is 3√2
    Then the equation of C2 must be one of:
    (x-4)^2 + (y-4)^2 = 16
    (x+2)^2 + (y+2)^2 = 4
    (x-2√2)^2 + (y+2√2)^2 = 8
    (x+2√2)^2 + (y-2√2)^2 = 8 -/
theorem circle_equation (C1 C2 : Set (ℝ × ℝ)) : 
  (∀ x y, (x-1)^2 + (y-1)^2 = 1 ↔ (x, y) ∈ C1) →
  (∀ x, (x, 0) ∈ C1 → (x, 0) ∈ C2) →
  (∀ y, (0, y) ∈ C1 → (0, y) ∈ C2) →
  (∃ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ C1 ∧ (x₂, y₂) ∈ C2 ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 18) →
  (∀ x y, (x, y) ∈ C2 ↔ 
    ((x-4)^2 + (y-4)^2 = 16) ∨
    ((x+2)^2 + (y+2)^2 = 4) ∨
    ((x-2*Real.sqrt 2)^2 + (y+2*Real.sqrt 2)^2 = 8) ∨
    ((x+2*Real.sqrt 2)^2 + (y-2*Real.sqrt 2)^2 = 8)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2505_250533


namespace NUMINAMATH_CALUDE_polynomial_transformation_l2505_250543

theorem polynomial_transformation (x : ℝ) (hx : x ≠ 0) :
  let z := x - 1 / x
  x^4 - 3*x^3 - 2*x^2 + 3*x + 1 = x^2 * (z^2 - 3*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l2505_250543


namespace NUMINAMATH_CALUDE_unique_sum_of_eight_only_36_37_l2505_250552

/-- A function that returns true if there exists exactly one set of 8 different positive integers that sum to n -/
def unique_sum_of_eight (n : ℕ) : Prop :=
  ∃! (s : Finset ℕ), s.card = 8 ∧ (∀ x ∈ s, x > 0) ∧ s.sum id = n

/-- Theorem stating that 36 and 37 are the only natural numbers with a unique sum of eight different positive integers -/
theorem unique_sum_of_eight_only_36_37 :
  ∀ n : ℕ, unique_sum_of_eight n ↔ n = 36 ∨ n = 37 := by
  sorry

#check unique_sum_of_eight_only_36_37

end NUMINAMATH_CALUDE_unique_sum_of_eight_only_36_37_l2505_250552


namespace NUMINAMATH_CALUDE_max_loquat_wholesale_l2505_250571

-- Define the fruit types
inductive Fruit
| Loquat
| Cherries
| Apples

-- Define the wholesale and retail prices
def wholesale_price (f : Fruit) : ℝ :=
  match f with
  | Fruit.Loquat => 8
  | Fruit.Cherries => 36
  | Fruit.Apples => 12

def retail_price (f : Fruit) : ℝ :=
  match f with
  | Fruit.Loquat => 10
  | Fruit.Cherries => 42
  | Fruit.Apples => 16

-- Define the theorem
theorem max_loquat_wholesale (x : ℝ) :
  -- Conditions
  (wholesale_price Fruit.Cherries = wholesale_price Fruit.Loquat + 28) →
  (80 * wholesale_price Fruit.Loquat + 120 * wholesale_price Fruit.Cherries = 4960) →
  (∃ y : ℝ, x * wholesale_price Fruit.Loquat + 
            (160 - x) * wholesale_price Fruit.Apples + 
            y * wholesale_price Fruit.Cherries = 5280) →
  (x * (retail_price Fruit.Loquat - wholesale_price Fruit.Loquat) +
   (160 - x) * (retail_price Fruit.Apples - wholesale_price Fruit.Apples) +
   ((5280 - x * wholesale_price Fruit.Loquat - (160 - x) * wholesale_price Fruit.Apples) / 
    wholesale_price Fruit.Cherries) * 
   (retail_price Fruit.Cherries - wholesale_price Fruit.Cherries) ≥ 1120) →
  -- Conclusion
  x ≤ 60 :=
by sorry


end NUMINAMATH_CALUDE_max_loquat_wholesale_l2505_250571


namespace NUMINAMATH_CALUDE_valid_cube_assignment_exists_l2505_250592

/-- Represents a vertex of a cube -/
inductive Vertex
| A | B | C | D | E | F | G | H

/-- Checks if two vertices are connected by an edge -/
def isConnected (v1 v2 : Vertex) : Prop := sorry

/-- Represents an assignment of natural numbers to the vertices of a cube -/
def CubeAssignment := Vertex → Nat

/-- Checks if the assignment satisfies the divisibility condition for connected vertices -/
def satisfiesConnectedDivisibility (assignment : CubeAssignment) : Prop :=
  ∀ v1 v2, isConnected v1 v2 → 
    (assignment v1 ∣ assignment v2) ∨ (assignment v2 ∣ assignment v1)

/-- Checks if the assignment satisfies the non-divisibility condition for non-connected vertices -/
def satisfiesNonConnectedNonDivisibility (assignment : CubeAssignment) : Prop :=
  ∀ v1 v2, ¬isConnected v1 v2 → 
    ¬(assignment v1 ∣ assignment v2) ∧ ¬(assignment v2 ∣ assignment v1)

/-- The main theorem stating that a valid assignment exists -/
theorem valid_cube_assignment_exists : 
  ∃ (assignment : CubeAssignment), 
    satisfiesConnectedDivisibility assignment ∧ 
    satisfiesNonConnectedNonDivisibility assignment := by
  sorry

end NUMINAMATH_CALUDE_valid_cube_assignment_exists_l2505_250592


namespace NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l2505_250584

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l2505_250584


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l2505_250541

theorem smallest_positive_largest_negative_smallest_abs_rational 
  (a b : ℤ) (c : ℚ) 
  (ha : a = 1) 
  (hb : b = -1) 
  (hc : c = 0) : a - b - c = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_smallest_abs_rational_l2505_250541


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2505_250504

theorem tan_45_degrees (Q : ℝ × ℝ) : 
  (Q.1 = 1 / Real.sqrt 2) → 
  (Q.2 = 1 / Real.sqrt 2) → 
  (Q.1^2 + Q.2^2 = 1) →
  Real.tan (π/4) = 1 := by
  sorry


end NUMINAMATH_CALUDE_tan_45_degrees_l2505_250504


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2505_250509

def is_symmetrical_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_of_point : 
  is_symmetrical_wrt_origin (-1, 1) (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2505_250509


namespace NUMINAMATH_CALUDE_christmas_cards_count_l2505_250566

/-- The number of Christmas cards John sent -/
def christmas_cards : ℕ := 20

/-- The number of birthday cards John sent -/
def birthday_cards : ℕ := 15

/-- The cost of each card in dollars -/
def cost_per_card : ℕ := 2

/-- The total amount John spent on cards in dollars -/
def total_spent : ℕ := 70

/-- Theorem stating that the number of Christmas cards is 20 -/
theorem christmas_cards_count :
  christmas_cards = 20 ∧
  birthday_cards = 15 ∧
  cost_per_card = 2 ∧
  total_spent = 70 →
  christmas_cards * cost_per_card + birthday_cards * cost_per_card = total_spent :=
by sorry

end NUMINAMATH_CALUDE_christmas_cards_count_l2505_250566


namespace NUMINAMATH_CALUDE_apple_banana_cost_l2505_250574

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 3 * a + 2 * b

/-- Theorem stating that the total cost of buying 3 kg of apples at 'a' yuan/kg
    and 2 kg of bananas at 'b' yuan/kg is (3a + 2b) yuan -/
theorem apple_banana_cost (a b : ℝ) :
  total_cost a b = 3 * a + 2 * b := by sorry

end NUMINAMATH_CALUDE_apple_banana_cost_l2505_250574


namespace NUMINAMATH_CALUDE_dvd_book_capacity_l2505_250529

/-- Represents the capacity of a DVD book -/
structure DVDBook where
  current : ℕ  -- Number of DVDs currently in the book
  remaining : ℕ  -- Number of additional DVDs that can be added

/-- Calculates the total capacity of a DVD book -/
def totalCapacity (book : DVDBook) : ℕ :=
  book.current + book.remaining

/-- Theorem: The total capacity of the given DVD book is 126 -/
theorem dvd_book_capacity : 
  ∀ (book : DVDBook), book.current = 81 → book.remaining = 45 → totalCapacity book = 126 :=
by
  sorry


end NUMINAMATH_CALUDE_dvd_book_capacity_l2505_250529


namespace NUMINAMATH_CALUDE_largest_common_term_l2505_250528

def sequence1 (n : ℕ) : ℤ := 2 + 4 * (n - 1)
def sequence2 (n : ℕ) : ℤ := 5 + 6 * (n - 1)

def is_common_term (x : ℤ) : Prop :=
  ∃ (n m : ℕ), sequence1 n = x ∧ sequence2 m = x

def is_in_range (x : ℤ) : Prop := 1 ≤ x ∧ x ≤ 200

theorem largest_common_term :
  ∃ (x : ℤ), is_common_term x ∧ is_in_range x ∧
  ∀ (y : ℤ), is_common_term y ∧ is_in_range y → y ≤ x ∧
  x = 190 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l2505_250528


namespace NUMINAMATH_CALUDE_cube_lateral_surface_area_l2505_250525

/-- The lateral surface area of a cube with side length 12 meters is 576 square meters. -/
theorem cube_lateral_surface_area : 
  let side_length : ℝ := 12
  let lateral_surface_area := 4 * side_length * side_length
  lateral_surface_area = 576 := by
sorry

end NUMINAMATH_CALUDE_cube_lateral_surface_area_l2505_250525


namespace NUMINAMATH_CALUDE_expression_simplification_l2505_250548

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2505_250548


namespace NUMINAMATH_CALUDE_election_result_l2505_250588

/-- Represents an election with five candidates -/
structure Election :=
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (votes_D : ℕ)
  (votes_E : ℕ)

/-- Conditions for the election -/
def ElectionConditions (e : Election) : Prop :=
  e.votes_A = (30 * e.total_votes) / 100 ∧
  e.votes_B = (25 * e.total_votes) / 100 ∧
  e.votes_C = (20 * e.total_votes) / 100 ∧
  e.votes_D = (15 * e.total_votes) / 100 ∧
  e.votes_E = e.total_votes - (e.votes_A + e.votes_B + e.votes_C + e.votes_D) ∧
  e.votes_A = e.votes_B + 1200

theorem election_result (e : Election) (h : ElectionConditions e) :
  e.total_votes = 24000 ∧ e.votes_E = 2400 := by
  sorry

end NUMINAMATH_CALUDE_election_result_l2505_250588


namespace NUMINAMATH_CALUDE_complex_multiplication_l2505_250522

theorem complex_multiplication (z₁ z₂ z : ℂ) : 
  z₁ = 1 - 3*I ∧ z₂ = 6 - 8*I ∧ z = z₁ * z₂ → z = -18 - 26*I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2505_250522


namespace NUMINAMATH_CALUDE_isosceles_triangle_triangle_area_l2505_250562

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem isosceles_triangle (t : Triangle) (h : t.a * Real.sin t.A = t.b * Real.sin t.B) :
  t.a = t.b := by
  sorry

-- Part 2
theorem triangle_area (t : Triangle) 
  (h1 : t.a + t.b = t.a * t.b)
  (h2 : t.c = 2)
  (h3 : t.C = π / 3) :
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_triangle_area_l2505_250562


namespace NUMINAMATH_CALUDE_ellipse_equation_l2505_250518

/-- Given an ellipse C with equation x²/a² + y²/b² = 1, where a > b > 0,
    focal length = 4, and passing through point P(√2, √3),
    prove that the equation of the ellipse is x²/8 + y²/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c = 2 ∧ a^2 - b^2 = c^2) →
  (2 / a^2 + 3 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2505_250518


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_a_range_l2505_250500

/-- A parabola with equation y = ax^2 - 1 where a ≠ 0 -/
structure Parabola where
  a : ℝ
  a_nonzero : a ≠ 0

/-- A point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2 - 1

/-- Two points are symmetric about the line y + x = 0 -/
def symmetric_about_line (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 + p1.2 + p2.1 + p2.2 = 0

/-- The main theorem -/
theorem parabola_symmetric_points_a_range (p : Parabola) 
  (p1 p2 : ParabolaPoint p) (h_distinct : p1 ≠ p2) 
  (h_symmetric : symmetric_about_line (p1.x, p1.y) (p2.x, p2.y)) : 
  p.a > 3/4 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_a_range_l2505_250500


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2505_250520

/-- Circle C in the Cartesian coordinate plane -/
def CircleC (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

/-- Line in the Cartesian coordinate plane -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

/-- New circle with radius 2 centered at a point (a, b) -/
def NewCircle (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 4

/-- Theorem stating the range of k values -/
theorem circle_line_intersection_range :
  ∀ k : ℝ, (∃ a b : ℝ, Line k a b ∧
    (∃ x y : ℝ, CircleC x y ∧ NewCircle a b x y)) ↔
  0 ≤ k ∧ k ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2505_250520


namespace NUMINAMATH_CALUDE_store_owner_order_theorem_l2505_250575

/-- The number of bottles of soda ordered by a store owner in April and May -/
def total_bottles_ordered (april_cases : ℕ) (may_cases : ℕ) (bottles_per_case : ℕ) : ℕ :=
  (april_cases + may_cases) * bottles_per_case

/-- Theorem stating that the store owner ordered 1000 bottles in April and May -/
theorem store_owner_order_theorem :
  total_bottles_ordered 20 30 20 = 1000 := by
  sorry

#eval total_bottles_ordered 20 30 20

end NUMINAMATH_CALUDE_store_owner_order_theorem_l2505_250575


namespace NUMINAMATH_CALUDE_half_difference_donations_l2505_250560

theorem half_difference_donations (margo_donation julie_donation : ℕ) 
  (h1 : margo_donation = 4300) 
  (h2 : julie_donation = 4700) : 
  (julie_donation - margo_donation) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_half_difference_donations_l2505_250560


namespace NUMINAMATH_CALUDE_range_of_sqrt_function_l2505_250551

theorem range_of_sqrt_function :
  ∀ x : ℝ, (∃ y : ℝ, y = Real.sqrt (x + 2)) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_range_of_sqrt_function_l2505_250551


namespace NUMINAMATH_CALUDE_dm_length_l2505_250502

/-- A square with side length 3 and two points that divide it into three equal areas -/
structure EqualAreaSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The point M on side AD -/
  m : ℝ
  /-- The point N on side AB -/
  n : ℝ
  /-- The side length is 3 -/
  side_eq : side = 3
  /-- The point M is between 0 and the side length -/
  m_range : 0 ≤ m ∧ m ≤ side
  /-- The point N is between 0 and the side length -/
  n_range : 0 ≤ n ∧ n ≤ side
  /-- CM and CN divide the square into three equal areas -/
  equal_areas : (1/2 * m * side) = (1/2 * n * side) ∧ (1/2 * m * side) = (1/3 * side^2)

/-- The length of DM in an EqualAreaSquare is 2 -/
theorem dm_length (s : EqualAreaSquare) : s.m = 2 := by
  sorry

end NUMINAMATH_CALUDE_dm_length_l2505_250502


namespace NUMINAMATH_CALUDE_odd_square_plus_multiple_l2505_250599

theorem odd_square_plus_multiple (o n : ℤ) 
  (ho : ∃ k, o = 2 * k + 1) : 
  Odd (o^2 + n * o) ↔ Even n := by
sorry

end NUMINAMATH_CALUDE_odd_square_plus_multiple_l2505_250599


namespace NUMINAMATH_CALUDE_dot_product_on_curve_l2505_250549

/-- Given a point M on the graph of f(x) = (x^2 + 4) / x, prove that the dot product of
    vectors MA and MB is -2, where A is the foot of the perpendicular from M to y = x,
    and B is the foot of the perpendicular from M to the y-axis. -/
theorem dot_product_on_curve (t : ℝ) (ht : t > 0) :
  let M : ℝ × ℝ := (t, (t^2 + 4) / t)
  let A : ℝ × ℝ := ((M.1 + M.2) / 2, (M.1 + M.2) / 2)  -- Foot on y = x
  let B : ℝ × ℝ := (0, M.2)  -- Foot on y-axis
  let MA : ℝ × ℝ := (A.1 - M.1, A.2 - M.2)
  let MB : ℝ × ℝ := (B.1 - M.1, B.2 - M.2)
  MA.1 * MB.1 + MA.2 * MB.2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_on_curve_l2505_250549


namespace NUMINAMATH_CALUDE_record_4800_steps_l2505_250573

/-- The standard number of steps per day -/
def standard : ℕ := 5000

/-- Function to calculate the recorded steps -/
def recordedSteps (actualSteps : ℕ) : ℤ :=
  (actualSteps : ℤ) - standard

/-- Theorem stating that 4800 steps should be recorded as -200 -/
theorem record_4800_steps :
  recordedSteps 4800 = -200 := by sorry

end NUMINAMATH_CALUDE_record_4800_steps_l2505_250573


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l2505_250586

theorem sum_greater_than_four (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y > x + y) : x + y > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l2505_250586


namespace NUMINAMATH_CALUDE_thirty_percent_more_than_75_l2505_250550

theorem thirty_percent_more_than_75 (x : ℝ) : x / 2 = 75 * 1.3 → x = 195 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_more_than_75_l2505_250550


namespace NUMINAMATH_CALUDE_three_true_propositions_l2505_250511

-- Definition of reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Definition of congruent triangles
def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of equal area triangles
def equal_area_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of real solutions for quadratic equation
def has_real_solutions (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_true_propositions :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∀ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) → ¬(equal_area_triangles t1 t2)) ∧
  (∀ m : ℝ, ¬(has_real_solutions m) → m > 1) :=
sorry

end NUMINAMATH_CALUDE_three_true_propositions_l2505_250511


namespace NUMINAMATH_CALUDE_ibrahim_lacking_money_l2505_250591

/-- The amount of money Ibrahim lacks to buy all items -/
def money_lacking (mp3_cost cd_cost headphones_cost case_cost savings father_contribution : ℕ) : ℕ :=
  (mp3_cost + cd_cost + headphones_cost + case_cost) - (savings + father_contribution)

/-- Theorem stating that Ibrahim lacks 165 euros -/
theorem ibrahim_lacking_money : 
  money_lacking 135 25 50 30 55 20 = 165 := by
  sorry

end NUMINAMATH_CALUDE_ibrahim_lacking_money_l2505_250591


namespace NUMINAMATH_CALUDE_geometric_body_volume_l2505_250519

/-- The volume of a geometric body composed of two tetrahedra --/
theorem geometric_body_volume :
  let side_length : ℝ := 1
  let height : ℝ := Real.sqrt 3 / 2
  let tetrahedron_volume : ℝ := (1 / 3) * ((Real.sqrt 3 / 4) * side_length ^ 2) * height
  let total_volume : ℝ := 2 * tetrahedron_volume
  total_volume = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_body_volume_l2505_250519


namespace NUMINAMATH_CALUDE_antonias_supplements_l2505_250564

theorem antonias_supplements :
  let total_pills : ℕ := 3 * 120 + 2 * 30
  let days : ℕ := 14
  let remaining_pills : ℕ := 350
  let supplements : ℕ := (total_pills - remaining_pills) / days
  supplements = 5 :=
by sorry

end NUMINAMATH_CALUDE_antonias_supplements_l2505_250564


namespace NUMINAMATH_CALUDE_wall_length_calculation_l2505_250501

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the wall's length is approximately 86 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 54 →
  wall_width = 68 →
  (mirror_side * mirror_side) * 2 = wall_width * (round ((mirror_side * mirror_side) * 2 / wall_width)) :=
by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l2505_250501


namespace NUMINAMATH_CALUDE_graces_mother_age_l2505_250524

theorem graces_mother_age :
  ∀ (grace_age grandmother_age mother_age : ℕ),
    grace_age = 60 →
    grace_age = (3 * grandmother_age) / 8 →
    grandmother_age = 2 * mother_age →
    mother_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_graces_mother_age_l2505_250524


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odd_integers_l2505_250526

theorem largest_of_three_consecutive_odd_integers (a b c : ℤ) : 
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) →  -- a, b, c are odd
  (b = a + 2 ∧ c = b + 2) →              -- a, b, c are consecutive
  (a + b + c = -147) →                   -- sum is -147
  (max a (max b c) = -47) :=             -- largest is -47
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odd_integers_l2505_250526


namespace NUMINAMATH_CALUDE_total_trolls_l2505_250593

/-- The number of trolls in different locations --/
structure TrollCounts where
  forest : ℕ
  bridge : ℕ
  plains : ℕ
  mountain : ℕ

/-- The conditions of the troll counting problem --/
def troll_conditions (t : TrollCounts) : Prop :=
  t.forest = 8 ∧
  t.forest = 2 * t.bridge - 4 ∧
  t.plains = t.bridge / 2 ∧
  t.mountain = t.plains + 3 ∧
  t.forest - t.mountain = 2 * t.bridge

/-- The theorem stating that given the conditions, the total number of trolls is 23 --/
theorem total_trolls (t : TrollCounts) (h : troll_conditions t) : 
  t.forest + t.bridge + t.plains + t.mountain = 23 := by
  sorry


end NUMINAMATH_CALUDE_total_trolls_l2505_250593


namespace NUMINAMATH_CALUDE_time_difference_problem_l2505_250563

theorem time_difference_problem (speed_ratio : ℚ) (time_A : ℚ) :
  speed_ratio = 3 / 4 →
  time_A = 2 →
  ∃ (time_B : ℚ), time_A - time_B = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_time_difference_problem_l2505_250563


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2505_250590

theorem imaginary_part_of_z (z : ℂ) : z = (3 + 4*Complex.I)*Complex.I → z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2505_250590


namespace NUMINAMATH_CALUDE_coin_stack_problem_l2505_250508

/-- Thickness of a nickel in millimeters -/
def nickel_thickness : ℚ := 39/20

/-- Thickness of a quarter in millimeters -/
def quarter_thickness : ℚ := 35/20

/-- Total height of the stack in millimeters -/
def stack_height : ℚ := 20

/-- The number of coins in the stack -/
def num_coins : ℕ := 10

theorem coin_stack_problem :
  ∃ (n q : ℕ), n * nickel_thickness + q * quarter_thickness = stack_height ∧ n + q = num_coins :=
sorry

end NUMINAMATH_CALUDE_coin_stack_problem_l2505_250508


namespace NUMINAMATH_CALUDE_ball_weight_order_l2505_250596

theorem ball_weight_order (a b c d : ℝ) 
  (eq1 : a + b = c + d)
  (ineq1 : a + d > b + c)
  (ineq2 : a + c < b) :
  d > b ∧ b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ball_weight_order_l2505_250596


namespace NUMINAMATH_CALUDE_decimal_rep_17_70_digit_150_of_17_70_l2505_250577

/-- The decimal representation of 17/70 has a repeating cycle of 6 digits -/
def decimal_cycle (n : ℕ) : ℕ := n % 6

/-- The digits in the repeating cycle of 17/70 -/
def cycle_digits : Fin 6 → ℕ
| 0 => 2
| 1 => 4
| 2 => 2
| 3 => 8
| 4 => 5
| 5 => 7

theorem decimal_rep_17_70 (n : ℕ) : 
  n > 0 → cycle_digits (decimal_cycle n) = 7 → n % 6 = 0 := by sorry

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 7 -/
theorem digit_150_of_17_70 : cycle_digits (decimal_cycle 150) = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_rep_17_70_digit_150_of_17_70_l2505_250577


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2505_250581

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2505_250581


namespace NUMINAMATH_CALUDE_right_triangle_area_l2505_250553

theorem right_triangle_area (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 → 
  a + b = 24 → 
  c = 24 → 
  d = 24 → 
  a^2 + b^2 = c^2 → 
  (1/2) * a * d = 216 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2505_250553


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_x_l2505_250537

theorem negation_of_universal_positive_square_plus_x (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_x_l2505_250537


namespace NUMINAMATH_CALUDE_cat_direction_at_noon_l2505_250547

/-- Represents the activities of the Cat -/
inductive CatActivity
  | TellingTale
  | SingingSong

/-- Represents the direction the Cat is going -/
inductive CatDirection
  | Left
  | Right

/-- The Cat's state at a given time -/
structure CatState where
  activity : CatActivity
  timeSpentOnCurrentActivity : ℕ

def minutes_per_tale : ℕ := 5
def minutes_per_song : ℕ := 4
def start_time : ℕ := 0
def end_time : ℕ := 120  -- 2 hours = 120 minutes

def initial_state : CatState :=
  { activity := CatActivity.TellingTale, timeSpentOnCurrentActivity := 0 }

def next_activity (current : CatActivity) : CatActivity :=
  match current with
  | CatActivity.TellingTale => CatActivity.SingingSong
  | CatActivity.SingingSong => CatActivity.TellingTale

def activity_duration (activity : CatActivity) : ℕ :=
  match activity with
  | CatActivity.TellingTale => minutes_per_tale
  | CatActivity.SingingSong => minutes_per_song

def update_state (state : CatState) (elapsed_time : ℕ) : CatState :=
  let total_time := state.timeSpentOnCurrentActivity + elapsed_time
  let current_activity_duration := activity_duration state.activity
  if total_time < current_activity_duration then
    { activity := state.activity, timeSpentOnCurrentActivity := total_time }
  else
    { activity := next_activity state.activity, timeSpentOnCurrentActivity := total_time % current_activity_duration }

def final_state : CatState :=
  update_state initial_state (end_time - start_time)

def activity_to_direction (activity : CatActivity) : CatDirection :=
  match activity with
  | CatActivity.TellingTale => CatDirection.Left
  | CatActivity.SingingSong => CatDirection.Right

theorem cat_direction_at_noon :
  activity_to_direction final_state.activity = CatDirection.Left :=
sorry

end NUMINAMATH_CALUDE_cat_direction_at_noon_l2505_250547


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_is_144_l2505_250556

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let sum_of_angles : ℝ := (n - 2) * 180  -- sum of interior angles formula
  let angle_measure : ℝ := sum_of_angles / n  -- measure of each angle (sum divided by number of sides)
  angle_measure

/-- Proof that the measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle_is_144 : 
  regular_decagon_interior_angle = 144 := by
  sorry


end NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_is_144_l2505_250556


namespace NUMINAMATH_CALUDE_equal_roots_condition_no_three_equal_values_same_solutions_condition_l2505_250567

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement ①
theorem equal_roots_condition (a b c : ℝ) (h : a ≠ 0) :
  b^2 - 4*a*c = 0 → ∃! x : ℝ, quadratic a b c x = 0 :=
sorry

-- Statement ②
theorem no_three_equal_values (a b c : ℝ) (h : a ≠ 0) :
  ¬∃ (m n s : ℝ), m ≠ n ∧ n ≠ s ∧ m ≠ s ∧
    quadratic a b c m = quadratic a b c n ∧
    quadratic a b c n = quadratic a b c s :=
sorry

-- Statement ③
theorem same_solutions_condition (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, quadratic a b c x + 2 = 0 ↔ (x + 2) * (x - 3) = 0) →
  4*a - 2*b + c = -2 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_no_three_equal_values_same_solutions_condition_l2505_250567


namespace NUMINAMATH_CALUDE_range_of_x_l2505_250517

theorem range_of_x (x : ℝ) 
  (hP : x^2 - 2*x - 3 ≥ 0)
  (hQ : |1 - x/2| ≥ 1) :
  x ≥ 4 ∨ x ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l2505_250517


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l2505_250589

theorem correct_mark_calculation (n : ℕ) (initial_avg : ℚ) (wrong_mark : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ initial_avg = 100 ∧ wrong_mark = 60 ∧ correct_avg = 95 →
  ∃ x : ℚ, n * initial_avg - wrong_mark + x = n * correct_avg ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l2505_250589


namespace NUMINAMATH_CALUDE_six_ronna_scientific_notation_l2505_250510

/-- Represents the number of zeros after a number for the "ronna" prefix -/
def ronna_zeros : ℕ := 27

/-- Converts a number with the "ronna" prefix to its scientific notation -/
def ronna_to_scientific (n : ℕ) : ℝ := n * (10 : ℝ) ^ ronna_zeros

/-- Theorem stating that 6 ronna is equal to 6 × 10^27 -/
theorem six_ronna_scientific_notation : ronna_to_scientific 6 = 6 * (10 : ℝ) ^ 27 := by
  sorry

end NUMINAMATH_CALUDE_six_ronna_scientific_notation_l2505_250510


namespace NUMINAMATH_CALUDE_special_numbers_l2505_250597

def is_special_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (a + b * 100 + c * 10 + d) * 10 = a * 100 + b * 10 + c + d

theorem special_numbers :
  ∀ n : ℕ, is_special_number n ↔ 
    n = 2019 ∨ n = 3028 ∨ n = 4037 ∨ n = 5046 ∨ 
    n = 6055 ∨ n = 7064 ∨ n = 8073 ∨ n = 9082 :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_l2505_250597


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l2505_250554

-- Define the problem parameters
def total_distance : Real := 8
def running_distance : Real := 4
def swimming_distance : Real := 4
def running_speed : Real := 10
def swimming_speed : Real := 6

-- Define the theorem
theorem triathlete_average_speed :
  let running_time := running_distance / running_speed
  let swimming_time := swimming_distance / swimming_speed
  let total_time := running_time + swimming_time
  let average_speed_mph := total_distance / total_time
  let average_speed_mpm := average_speed_mph / 60
  average_speed_mpm = 0.125 := by sorry

end NUMINAMATH_CALUDE_triathlete_average_speed_l2505_250554


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_area_l2505_250546

/-- The area of the largest circle that can be inscribed in a square with side length 2 decimeters is π square decimeters. -/
theorem largest_inscribed_circle_area (square_side : ℝ) (h : square_side = 2) :
  let circle_area := π * (square_side / 2)^2
  circle_area = π := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_area_l2505_250546


namespace NUMINAMATH_CALUDE_binomial_18_10_l2505_250506

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l2505_250506


namespace NUMINAMATH_CALUDE_complex_modulus_l2505_250530

theorem complex_modulus (a b : ℝ) :
  (1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2505_250530


namespace NUMINAMATH_CALUDE_harolds_rent_l2505_250535

/-- Harold's monthly finances --/
def harolds_finances (rent : ℝ) : Prop :=
  let income : ℝ := 2500
  let car_payment : ℝ := 300
  let utilities : ℝ := car_payment / 2
  let groceries : ℝ := 50
  let remaining : ℝ := income - rent - car_payment - utilities - groceries
  let retirement_savings : ℝ := remaining / 2
  let final_balance : ℝ := remaining - retirement_savings
  final_balance = 650

/-- Theorem: Harold's rent is $700.00 --/
theorem harolds_rent : ∃ (rent : ℝ), harolds_finances rent ∧ rent = 700 := by
  sorry

end NUMINAMATH_CALUDE_harolds_rent_l2505_250535


namespace NUMINAMATH_CALUDE_second_meeting_time_is_12_minutes_l2505_250503

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  trackLength : ℝ
  firstMeetingDistance : ℝ
  firstMeetingTime : ℝ
  marie : Runner
  john : Runner

/-- Calculates the time of the second meeting given a race scenario -/
def secondMeetingTime (scenario : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that the second meeting occurs 12 minutes after the start -/
theorem second_meeting_time_is_12_minutes (scenario : RaceScenario) 
  (h1 : scenario.trackLength = 500)
  (h2 : scenario.firstMeetingDistance = 100)
  (h3 : scenario.firstMeetingTime = 2)
  (h4 : scenario.marie.initialPosition = 0)
  (h5 : scenario.john.initialPosition = 500)
  (h6 : scenario.marie.speed = scenario.firstMeetingDistance / scenario.firstMeetingTime)
  (h7 : scenario.john.speed = (scenario.trackLength - scenario.firstMeetingDistance) / scenario.firstMeetingTime) :
  secondMeetingTime scenario = 12 :=
sorry

end NUMINAMATH_CALUDE_second_meeting_time_is_12_minutes_l2505_250503


namespace NUMINAMATH_CALUDE_debate_club_election_l2505_250542

def election_ways (n m k : ℕ) : ℕ :=
  (n - k).factorial / ((n - k - m).factorial * m.factorial) +
  k.factorial * (n - k).choose (m - k)

theorem debate_club_election :
  election_ways 30 5 4 = 6378720 :=
sorry

end NUMINAMATH_CALUDE_debate_club_election_l2505_250542


namespace NUMINAMATH_CALUDE_celia_running_time_l2505_250578

/-- Given that Celia runs twice as fast as Lexie, and Lexie takes 20 minutes to run a mile,
    prove that Celia will take 300 minutes to run 30 miles. -/
theorem celia_running_time :
  ∀ (lexie_speed celia_speed : ℝ),
  celia_speed = 2 * lexie_speed →
  lexie_speed * 20 = 1 →
  celia_speed * 300 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_celia_running_time_l2505_250578


namespace NUMINAMATH_CALUDE_find_first_number_l2505_250587

theorem find_first_number (x : ℝ) : 
  let set1 := [10, 70, 19]
  let set2 := [x, 40, 60]
  (List.sum set2 / 3 : ℝ) = (List.sum set1 / 3 : ℝ) + 7 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_first_number_l2505_250587


namespace NUMINAMATH_CALUDE_kylies_towels_l2505_250594

theorem kylies_towels (daughters_towels husband_towels machine_capacity loads : ℕ) 
  (h1 : daughters_towels = 6)
  (h2 : husband_towels = 3)
  (h3 : machine_capacity = 4)
  (h4 : loads = 3) : 
  ∃ k : ℕ, k = loads * machine_capacity - daughters_towels - husband_towels ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_kylies_towels_l2505_250594


namespace NUMINAMATH_CALUDE_incorrect_representation_of_roots_l2505_250534

theorem incorrect_representation_of_roots : ∃ x : ℝ, x^2 - 3*x = 0 ∧ ¬(x = x ∧ x = 2*x) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_representation_of_roots_l2505_250534


namespace NUMINAMATH_CALUDE_range_of_g_l2505_250523

def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_g :
  {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ g x = y} = {y : ℝ | y < -27 ∨ y > -27} :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l2505_250523


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2505_250536

theorem collinear_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, Real.sqrt (1 + Real.sin (40 * π / 180))]
  let b : Fin 2 → ℝ := ![1 / Real.sin (65 * π / 180), x]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, a i = k * b i)) → x = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2505_250536


namespace NUMINAMATH_CALUDE_books_read_per_year_l2505_250595

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  36 * c * s

/-- Theorem stating the total number of books read by the entire student body in one year -/
theorem books_read_per_year (c s : ℕ) : 
  total_books_read c s = 3 * 12 * c * s := by
  sorry

#check books_read_per_year

end NUMINAMATH_CALUDE_books_read_per_year_l2505_250595


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l2505_250505

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define point H
def H : ℝ × ℝ := (1, -1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 3/2)^2 = 25/4

-- Theorem statement
theorem tangent_circle_equation :
  ∃ (A B : ℝ × ℝ),
    (parabola A.1 A.2) ∧
    (parabola B.1 B.2) ∧
    (∃ (m₁ m₂ : ℝ),
      (A.2 - H.2 = m₁ * (A.1 - H.1)) ∧
      (B.2 - H.2 = m₂ * (B.1 - H.1)) ∧
      (∀ (x y : ℝ), parabola x y → m₁ * (x - A.1) + A.2 ≥ y) ∧
      (∀ (x y : ℝ), parabola x y → m₂ * (x - B.1) + B.2 ≥ y)) →
    ∀ (x y : ℝ), circle_equation x y ↔ 
      ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ x = t * A.1 + (1 - t) * B.1 ∧ y = t * A.2 + (1 - t) * B.2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l2505_250505


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l2505_250570

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 8)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 15) :
  a * b + c * d = 84 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l2505_250570


namespace NUMINAMATH_CALUDE_inequality_solution_set_f_inequality_l2505_250531

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Theorem for part (I)
theorem inequality_solution_set (x : ℝ) :
  f (x + 8) ≥ 10 - f x ↔ x ≤ -10 ∨ x ≥ 0 := by sorry

-- Theorem for part (II)
theorem f_inequality (x y : ℝ) (hx : |x| > 1) (hy : |y| < 1) :
  f y < |x| * f (y / x^2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_f_inequality_l2505_250531


namespace NUMINAMATH_CALUDE_accurate_reading_is_10_30_l2505_250585

/-- Represents a scale reading with a lower bound, upper bound, and increment -/
structure ScaleReading where
  lowerBound : ℝ
  upperBound : ℝ
  increment : ℝ

/-- Represents the position of an arrow on the scale -/
structure ArrowPosition where
  value : ℝ
  beforeMidpoint : Bool

/-- Given a scale reading and an arrow position, determines the most accurate reading -/
def mostAccurateReading (scale : ScaleReading) (arrow : ArrowPosition) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the most accurate reading is 10.30 -/
theorem accurate_reading_is_10_30 :
  let scale := ScaleReading.mk 10.2 10.4 0.05
  let arrow := ArrowPosition.mk 10.33 true
  mostAccurateReading scale arrow = 10.30 := by
  sorry

end NUMINAMATH_CALUDE_accurate_reading_is_10_30_l2505_250585


namespace NUMINAMATH_CALUDE_fifty_paise_coins_count_l2505_250555

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- Represents the bag of coins -/
structure CoinBag where
  numCoins : CoinType → ℕ
  totalValue : ℚ
  equalCoins : ∀ (c1 c2 : CoinType), numCoins c1 = numCoins c2

def coinValue : CoinType → ℚ
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

theorem fifty_paise_coins_count (bag : CoinBag) 
  (h1 : bag.totalValue = 105)
  (h2 : bag.numCoins CoinType.OneRupee = 60) :
  bag.numCoins CoinType.FiftyPaise = 60 := by
  sorry

end NUMINAMATH_CALUDE_fifty_paise_coins_count_l2505_250555


namespace NUMINAMATH_CALUDE_det_A_l2505_250568

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, 2; 8, 5, -1; 3, 3, 7]

theorem det_A : A.det = 132 := by sorry

end NUMINAMATH_CALUDE_det_A_l2505_250568
