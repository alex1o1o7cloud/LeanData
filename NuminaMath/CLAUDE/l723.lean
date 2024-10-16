import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l723_72338

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l723_72338


namespace NUMINAMATH_CALUDE_pretzel_problem_l723_72334

theorem pretzel_problem (barry_pretzels shelly_pretzels angie_pretzels : ℕ) : 
  barry_pretzels = 12 →
  shelly_pretzels = barry_pretzels / 2 →
  angie_pretzels = 3 * shelly_pretzels →
  angie_pretzels = 18 := by
  sorry

end NUMINAMATH_CALUDE_pretzel_problem_l723_72334


namespace NUMINAMATH_CALUDE_inequality_solution_set_sqrt_sum_inequality_l723_72323

-- Part I
theorem inequality_solution_set (x : ℝ) :
  (|x - 5| - |2*x + 3| ≥ 1) ↔ (-7 ≤ x ∧ x ≤ 1/3) := by sorry

-- Part II
theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1/2) :
  Real.sqrt a + Real.sqrt b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_sqrt_sum_inequality_l723_72323


namespace NUMINAMATH_CALUDE_james_total_earnings_l723_72359

def january_earnings : ℕ := 4000

def february_earnings : ℕ := 2 * january_earnings

def march_earnings : ℕ := february_earnings - 2000

def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings : total_earnings = 18000 := by
  sorry

end NUMINAMATH_CALUDE_james_total_earnings_l723_72359


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l723_72324

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Performs one operation on the urn state -/
def performOperation (state : UrnState) : UrnState :=
  sorry

/-- Calculates the probability of drawing a specific color -/
def drawProbability (state : UrnState) (color : Color) : ℚ :=
  sorry

/-- Calculates the probability of a specific sequence of draws -/
def sequenceProbability (sequence : List Color) : ℚ :=
  sorry

/-- Counts the number of valid sequences resulting in 3 red and 3 blue balls -/
def countValidSequences : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initialState : UrnState := ⟨1, 1⟩
  let finalState : UrnState := ⟨3, 3⟩
  let numOperations : ℕ := 5
  (countValidSequences * sequenceProbability [Color.Red, Color.Red, Color.Red, Color.Blue, Color.Blue]) = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l723_72324


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l723_72321

theorem degree_to_radian_conversion (π : Real) (h : π = Real.pi) :
  120 * (π / 180) = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l723_72321


namespace NUMINAMATH_CALUDE_quarter_percent_of_200_l723_72322

theorem quarter_percent_of_200 : (1 / 4 : ℚ) / 100 * 200 = (1 / 2 : ℚ) := by sorry

#eval (1 / 4 : ℚ) / 100 * 200

end NUMINAMATH_CALUDE_quarter_percent_of_200_l723_72322


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l723_72304

theorem fuel_cost_savings 
  (old_efficiency : ℝ) 
  (old_fuel_cost : ℝ) 
  (trip_distance : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_cost_increase : ℝ) 
  (h1 : old_efficiency > 0)
  (h2 : old_fuel_cost > 0)
  (h3 : trip_distance = 1000)
  (h4 : efficiency_improvement = 0.6)
  (h5 : fuel_cost_increase = 0.25) :
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost := (trip_distance / old_efficiency) * old_fuel_cost
  let new_trip_cost := (trip_distance / new_efficiency) * new_fuel_cost
  let savings_percentage := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percentage = 21.875 := by
sorry

end NUMINAMATH_CALUDE_fuel_cost_savings_l723_72304


namespace NUMINAMATH_CALUDE_range_of_x_l723_72327

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_x (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, |a + b| + |a - b| ≥ |a| * f x) →
  ∃ x, x ∈ Set.Icc 0 4 ∧ ∀ y, (∀ z, |a + b| + |a - b| ≥ |a| * f z) → y ∈ Set.Icc 0 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l723_72327


namespace NUMINAMATH_CALUDE_geometry_propositions_l723_72317

-- Define the concept of vertical angles
def are_vertical_angles (α β : Real) : Prop := sorry

-- Define the concept of complementary angles
def are_complementary (α β : Real) : Prop := α + β = 90

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem geometry_propositions :
  -- Proposition 1: Vertical angles are equal
  ∀ (α β : Real), are_vertical_angles α β → α = β
  
  -- Proposition 2: Complementary angles of equal angles are equal
  ∧ ∀ (α β γ δ : Real), α = β ∧ are_complementary α γ ∧ are_complementary β δ → γ = δ
  
  -- Proposition 3: If b is parallel to a and c is parallel to a, then b is parallel to c
  ∧ ∀ (a b c : Line), parallel b a ∧ parallel c a → parallel b c :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l723_72317


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l723_72302

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line y = x -/
def bisector_line : Line := { a := 1, b := -1, c := 0 }

/-- Checks if a line is the angle bisector of two other lines -/
def is_angle_bisector (bisector : Line) (l1 : Line) (l2 : Line) : Prop := sorry

/-- Theorem: If the bisector of the angle between lines l₁ and l₂ is y = x,
    and the equation of l₁ is ax + by + c = 0 (ab > 0),
    then the equation of l₂ is bx + ay + c = 0 -/
theorem symmetric_line_equation (l1 : Line) (l2 : Line) 
    (h1 : is_angle_bisector bisector_line l1 l2)
    (h2 : l1.a * l1.b > 0) : 
  l2.a = l1.b ∧ l2.b = l1.a ∧ l2.c = l1.c := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l723_72302


namespace NUMINAMATH_CALUDE_book_cost_prices_correct_l723_72326

/-- Represents the cost and quantity information for a type of book -/
structure BookType where
  cost_per_book : ℝ
  total_cost : ℝ
  quantity : ℝ

/-- Proves that given the conditions, the cost prices for book types A and B are correct -/
theorem book_cost_prices_correct (book_a book_b : BookType)
  (h1 : book_a.cost_per_book = book_b.cost_per_book + 15)
  (h2 : book_a.total_cost = 675)
  (h3 : book_b.total_cost = 450)
  (h4 : book_a.quantity = book_b.quantity)
  (h5 : book_a.quantity = book_a.total_cost / book_a.cost_per_book)
  (h6 : book_b.quantity = book_b.total_cost / book_b.cost_per_book) :
  book_a.cost_per_book = 45 ∧ book_b.cost_per_book = 30 := by
  sorry

#check book_cost_prices_correct

end NUMINAMATH_CALUDE_book_cost_prices_correct_l723_72326


namespace NUMINAMATH_CALUDE_quarter_circle_perimeter_l723_72301

/-- The perimeter of a region defined by quarter circles on the corners of a rectangle --/
theorem quarter_circle_perimeter (π : ℝ) (h : π > 0) : 
  let shorter_side : ℝ := 2 / π
  let longer_side : ℝ := 4 / π
  let quarter_circle_perimeter : ℝ := π * shorter_side / 2
  4 * quarter_circle_perimeter = 4 := by
  sorry

end NUMINAMATH_CALUDE_quarter_circle_perimeter_l723_72301


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l723_72350

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l723_72350


namespace NUMINAMATH_CALUDE_no_integer_solution_l723_72371

theorem no_integer_solution :
  ¬ ∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l723_72371


namespace NUMINAMATH_CALUDE_consecutive_points_length_l723_72388

/-- Given 5 consecutive points on a straight line, prove that ae = 18 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (∃ x y z w : ℝ, 
    x = b - a ∧ 
    y = c - b ∧ 
    z = d - c ∧ 
    w = e - d ∧
    y = 2 * z ∧ 
    w = 4 ∧ 
    x = 5 ∧ 
    x + y = 11) →
  e - a = 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l723_72388


namespace NUMINAMATH_CALUDE_division_problem_l723_72366

theorem division_problem (a b c : ℚ) : 
  a = (2 : ℚ) / 3 * (b + c) →
  b = (6 : ℚ) / 9 * (a + c) →
  a = 200 →
  a + b + c = 500 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l723_72366


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l723_72340

theorem dining_bill_calculation (number_of_people : ℕ) (individual_payment : ℚ) (tip_percentage : ℚ) 
  (h1 : number_of_people = 6)
  (h2 : individual_payment = 25.48)
  (h3 : tip_percentage = 0.10) :
  (number_of_people : ℚ) * individual_payment / (1 + tip_percentage) = 139.89 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l723_72340


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l723_72375

theorem circle_center_and_radius :
  ∀ (x y : ℝ), 4*x^2 - 8*x + 4*y^2 + 24*y + 28 = 0 ↔ 
  (x - 1)^2 + (y + 3)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l723_72375


namespace NUMINAMATH_CALUDE_eight_x_plus_y_value_l723_72365

theorem eight_x_plus_y_value (x y z : ℝ) 
  (eq1 : x + 2*y - 3*z = 7) 
  (eq2 : 2*x - y + 2*z = 6) : 
  8*x + y = 32 := by sorry

end NUMINAMATH_CALUDE_eight_x_plus_y_value_l723_72365


namespace NUMINAMATH_CALUDE_sqrt_two_div_sqrt_eighteen_equals_one_third_l723_72332

theorem sqrt_two_div_sqrt_eighteen_equals_one_third :
  Real.sqrt 2 / Real.sqrt 18 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_div_sqrt_eighteen_equals_one_third_l723_72332


namespace NUMINAMATH_CALUDE_new_average_after_increase_l723_72378

theorem new_average_after_increase (numbers : List ℝ) (h1 : numbers.length = 8) 
  (h2 : numbers.sum / numbers.length = 8) : 
  let new_numbers := numbers.map (λ x => if numbers.indexOf x < 5 then x + 4 else x)
  new_numbers.sum / new_numbers.length = 10.5 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_increase_l723_72378


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l723_72373

/-- For a geometric sequence with common ratio 2 and sum of first 3 terms 34685, the second term is 9910 -/
theorem geometric_sequence_second_term : ∀ (a : ℕ → ℚ), 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  (a 1 + a 2 + a 3 = 34685) →   -- sum of first 3 terms is 34685
  a 2 = 9910 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l723_72373


namespace NUMINAMATH_CALUDE_vector_identity_l723_72369

variable (α β γ : ℝ)
variable (v : ℝ → ℝ → ℝ → ℝ)
variable (i j k : ℝ → ℝ → ℝ → ℝ)

def cross_product (a b : ℝ → ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ → ℝ := sorry

theorem vector_identity (h : ∃ c : ℝ, 
  ∀ v, cross_product (α • i) (cross_product v i) + 
       cross_product (β • j) (cross_product v j) + 
       cross_product (γ • k) (cross_product v k) = c • v) : 
  ∃ c : ℝ, c = α + β + γ - 1 := by sorry

end NUMINAMATH_CALUDE_vector_identity_l723_72369


namespace NUMINAMATH_CALUDE_system_solution_l723_72362

theorem system_solution (x y : ℝ) : 
  x^3 + y^3 = 7 ∧ x*y*(x + y) = -2 ↔ (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l723_72362


namespace NUMINAMATH_CALUDE_yoongi_multiplication_l723_72336

theorem yoongi_multiplication (x : ℝ) : 8 * x = 64 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_multiplication_l723_72336


namespace NUMINAMATH_CALUDE_special_function_property_l723_72306

/-- A function f: ℝ → ℝ satisfying specific properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1) = -f (-x - 1)) ∧  -- f(x-1) is odd
  (∀ x, f (x + 1) = f (-x + 1)) ∧  -- f(x+1) is even
  (∀ x, x > -1 ∧ x < 1 → f x = -Real.exp x)  -- f(x) = -e^x for x ∈ (-1,1)

/-- Theorem stating the property of the special function -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x, f (2 * x) = f (2 * x + 8) :=
by sorry

end NUMINAMATH_CALUDE_special_function_property_l723_72306


namespace NUMINAMATH_CALUDE_power_23_mod_5_l723_72339

theorem power_23_mod_5 : 2^23 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_23_mod_5_l723_72339


namespace NUMINAMATH_CALUDE_norbs_age_l723_72319

def guesses : List Nat := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : Nat) : Prop := Nat.Prime n

def at_least_half_too_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length ≥ guesses.length / 2

def two_off_by_one (age : Nat) : Prop :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length = 2

theorem norbs_age :
  ∃! age : Nat,
    age ∈ guesses ∧
    is_prime age ∧
    at_least_half_too_low age ∧
    two_off_by_one age ∧
    age = 37 :=
sorry

end NUMINAMATH_CALUDE_norbs_age_l723_72319


namespace NUMINAMATH_CALUDE_speaking_orders_count_l723_72380

def total_students : ℕ := 6
def speakers_to_select : ℕ := 4
def specific_students : ℕ := 2

theorem speaking_orders_count : 
  (total_students.choose speakers_to_select * speakers_to_select.factorial) -
  ((total_students - specific_students).choose speakers_to_select * speakers_to_select.factorial) = 336 :=
by sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l723_72380


namespace NUMINAMATH_CALUDE_power_value_theorem_l723_72320

theorem power_value_theorem (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : 
  a^(2*m - n) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_power_value_theorem_l723_72320


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l723_72381

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) (remaining_avg_age : ℕ) : 
  team_size = 11 → 
  captain_age = 24 → 
  team_avg_age = 23 → 
  remaining_avg_age = team_avg_age - 1 → 
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 7 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l723_72381


namespace NUMINAMATH_CALUDE_divisibility_count_l723_72364

theorem divisibility_count : ∃! n : ℕ, n > 0 ∧ n < 500 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 7 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_count_l723_72364


namespace NUMINAMATH_CALUDE_least_prime_factor_11_5_minus_11_4_l723_72330

theorem least_prime_factor_11_5_minus_11_4 :
  Nat.minFac (11^5 - 11^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_11_5_minus_11_4_l723_72330


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l723_72360

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 8 + c + d) / 5 = 18 → (c + d) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l723_72360


namespace NUMINAMATH_CALUDE_mrs_franklin_students_l723_72353

/-- The number of Valentines Mrs. Franklin has -/
def valentines_owned : ℝ := 58.0

/-- The number of additional Valentines Mrs. Franklin needs -/
def valentines_needed : ℝ := 16.0

/-- The total number of students Mrs. Franklin has -/
def total_students : ℝ := valentines_owned + valentines_needed

theorem mrs_franklin_students : total_students = 74.0 := by
  sorry

end NUMINAMATH_CALUDE_mrs_franklin_students_l723_72353


namespace NUMINAMATH_CALUDE_right_quadrilateral_area_l723_72389

/-- A quadrilateral with right angles at B and D, diagonal AC = 5,
    two sides with distinct integer lengths, and one side of length 3 -/
structure RightQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  right_angle_B : AB * BC = 0
  right_angle_D : CD * DA = 0
  diagonal_length : AB^2 + BC^2 = 25
  distinct_integer_sides : ∃ (x y : ℕ), (AB = x ∨ BC = x ∨ CD = x ∨ DA = x) ∧
                                        (AB = y ∨ BC = y ∨ CD = y ∨ DA = y) ∧
                                        x ≠ y
  one_side_three : AB = 3 ∨ BC = 3 ∨ CD = 3 ∨ DA = 3

/-- The area of the RightQuadrilateral is 12 -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : 
  (q.AB * q.BC + q.CD * q.DA) / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_right_quadrilateral_area_l723_72389


namespace NUMINAMATH_CALUDE_bake_four_pans_l723_72335

/-- The number of pans of cookies that can be baked in a given time -/
def pans_of_cookies (total_time minutes_per_pan : ℕ) : ℕ :=
  total_time / minutes_per_pan

/-- Proof that 4 pans of cookies can be baked in 28 minutes when each pan takes 7 minutes -/
theorem bake_four_pans : pans_of_cookies 28 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bake_four_pans_l723_72335


namespace NUMINAMATH_CALUDE_nickel_chocolates_l723_72314

theorem nickel_chocolates (robert : ℕ) (difference : ℕ) (h1 : robert = 12) (h2 : difference = 9) :
  robert - difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l723_72314


namespace NUMINAMATH_CALUDE_solution_difference_l723_72399

theorem solution_difference (a b : ℝ) : 
  (∀ x, (x - 5) * (x + 5) = 26 * x - 130 ↔ x = a ∨ x = b) →
  a ≠ b →
  a > b →
  a - b = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l723_72399


namespace NUMINAMATH_CALUDE_shobhas_current_age_l723_72344

/-- Given the ratio of Shekhar's age to Shobha's age and Shekhar's future age, 
    prove Shobha's current age -/
theorem shobhas_current_age 
  (shekhar_age shobha_age : ℕ) 
  (ratio : shekhar_age / shobha_age = 4 / 3)
  (future_age : shekhar_age + 6 = 26) : 
  shobha_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_shobhas_current_age_l723_72344


namespace NUMINAMATH_CALUDE_computer_operations_l723_72367

/-- Calculates the total number of operations a computer can perform given its operation rate and runtime. -/
theorem computer_operations (rate : ℝ) (time : ℝ) (h1 : rate = 4 * 10^8) (h2 : time = 6 * 10^5) :
  rate * time = 2.4 * 10^14 := by
  sorry

#check computer_operations

end NUMINAMATH_CALUDE_computer_operations_l723_72367


namespace NUMINAMATH_CALUDE_second_hose_spray_rate_l723_72341

/-- Calculates the spray rate of the second hose needed to fill a pool --/
theorem second_hose_spray_rate 
  (pool_capacity : ℝ) 
  (first_hose_rate : ℝ) 
  (total_time : ℝ) 
  (second_hose_time : ℝ) 
  (h1 : pool_capacity = 390)
  (h2 : first_hose_rate = 50)
  (h3 : total_time = 5)
  (h4 : second_hose_time = 2)
  : ∃ (second_hose_rate : ℝ), 
    second_hose_rate * second_hose_time + first_hose_rate * total_time = pool_capacity ∧ 
    second_hose_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_hose_spray_rate_l723_72341


namespace NUMINAMATH_CALUDE_nested_root_equality_l723_72363

theorem nested_root_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 7) ^ (1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_nested_root_equality_l723_72363


namespace NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l723_72333

-- Define a right angle as 90 degrees
def right_angle : ℝ := 90

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180

-- Define a right triangle
structure RightTriangle extends Triangle where
  has_right_angle : angle1 = right_angle ∨ angle2 = right_angle ∨ angle3 = right_angle

-- Theorem: A right triangle has exactly one right angle
theorem right_triangle_has_one_right_angle (rt : RightTriangle) : 
  (rt.angle1 = right_angle ∧ rt.angle2 ≠ right_angle ∧ rt.angle3 ≠ right_angle) ∨
  (rt.angle1 ≠ right_angle ∧ rt.angle2 = right_angle ∧ rt.angle3 ≠ right_angle) ∨
  (rt.angle1 ≠ right_angle ∧ rt.angle2 ≠ right_angle ∧ rt.angle3 = right_angle) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_has_one_right_angle_l723_72333


namespace NUMINAMATH_CALUDE_round_trip_time_l723_72382

/-- Calculates the total time for a round trip on a river given the rower's speed, river speed, and distance. -/
theorem round_trip_time (rower_speed river_speed distance : ℝ) 
  (h1 : rower_speed = 6)
  (h2 : river_speed = 2)
  (h3 : distance = 2.67) : 
  (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1.00125 := by
  sorry

#eval (2.67 / (6 - 2)) + (2.67 / (6 + 2))

end NUMINAMATH_CALUDE_round_trip_time_l723_72382


namespace NUMINAMATH_CALUDE_number_problem_l723_72398

theorem number_problem (x : ℝ) : 35 - 3 * x = 8 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l723_72398


namespace NUMINAMATH_CALUDE_subcommittee_count_l723_72377

/-- The number of people in the main committee -/
def committee_size : ℕ := 7

/-- The size of each sub-committee -/
def subcommittee_size : ℕ := 2

/-- The number of people that can be chosen for the second position in the sub-committee -/
def remaining_choices : ℕ := committee_size - 1

theorem subcommittee_count :
  (committee_size.choose subcommittee_size) / committee_size = remaining_choices :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l723_72377


namespace NUMINAMATH_CALUDE_number_of_students_l723_72346

theorem number_of_students (initial_avg : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (correct_avg : ℝ) :
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_mark = 10 →
  correct_avg = 95 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * correct_avg = n * initial_avg - (wrong_mark - correct_mark) ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l723_72346


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l723_72357

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 16*x + 15 = 0 → ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 226 ∧ (x = s₁ ∨ x = s₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l723_72357


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l723_72391

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 + x - 12 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l723_72391


namespace NUMINAMATH_CALUDE_prob_two_sixes_one_four_l723_72395

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling a specific number on a single die -/
def single_prob : ℚ := 1 / num_sides

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The number of ways to arrange two 6's and one 4 in three dice rolls -/
def num_arrangements : ℕ := 3

/-- The probability of rolling exactly two 6's and one 4 when rolling three six-sided dice simultaneously -/
theorem prob_two_sixes_one_four : 
  (single_prob ^ num_dice * num_arrangements : ℚ) = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_prob_two_sixes_one_four_l723_72395


namespace NUMINAMATH_CALUDE_rachel_homework_l723_72383

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 7 →
  math_pages = reading_pages + 4 →
  reading_pages = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l723_72383


namespace NUMINAMATH_CALUDE_area_PNR_l723_72347

-- Define the points
variable (P Q R M N : ℝ × ℝ)

-- Define the conditions
axiom right_angle_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
axiom PR_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 12
axiom QR_length : Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 16
axiom M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
axiom N_on_QR : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N = (Q.1 + t * (R.1 - Q.1), Q.2 + t * (R.2 - Q.2))
axiom MN_perpendicular_PQ : (M.1 - N.1) * (P.1 - Q.1) + (M.2 - N.2) * (P.2 - Q.2) = 0

-- Theorem to prove
theorem area_PNR : 
  (1/2) * Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) * Real.sqrt ((N.1 - R.1)^2 + (N.2 - R.2)^2) = 21 :=
sorry

end NUMINAMATH_CALUDE_area_PNR_l723_72347


namespace NUMINAMATH_CALUDE_bookstore_shipment_size_l723_72342

theorem bookstore_shipment_size (displayed_percentage : ℚ) (stored_amount : ℕ) : 
  displayed_percentage = 1/4 →
  stored_amount = 225 →
  ∃ total : ℕ, total = 300 ∧ (1 - displayed_percentage) * total = stored_amount :=
by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_size_l723_72342


namespace NUMINAMATH_CALUDE_jamies_father_age_ratio_l723_72385

/-- The year of Jamie's 10th birthday -/
def birth_year : ℕ := 2010

/-- Jamie's age on his 10th birthday -/
def jamie_initial_age : ℕ := 10

/-- The ratio of Jamie's father's age to Jamie's age on Jamie's 10th birthday -/
def initial_age_ratio : ℕ := 5

/-- The year when Jamie's father's age is twice Jamie's age -/
def target_year : ℕ := 2040

/-- The ratio of Jamie's father's age to Jamie's age in the target year -/
def target_age_ratio : ℕ := 2

theorem jamies_father_age_ratio :
  target_year = birth_year + (initial_age_ratio - target_age_ratio) * jamie_initial_age := by
  sorry

#check jamies_father_age_ratio

end NUMINAMATH_CALUDE_jamies_father_age_ratio_l723_72385


namespace NUMINAMATH_CALUDE_parabola_directrix_l723_72313

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_equation (y : ℝ) : Prop :=
  y = -17/4

/-- Theorem: The directrix of the given parabola is y = -17/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_directrix : ℝ, directrix_equation y_directrix :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l723_72313


namespace NUMINAMATH_CALUDE_polarEqIsCircle_l723_72352

-- Define a polar coordinate
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

-- Define a circle in Cartesian coordinates
def isCircle (f : ℝ × ℝ → Prop) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ (x y : ℝ),
    f (x, y) ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the polar equation ρ = 1
def polarEq (p : PolarCoord) : Prop :=
  p.ρ = 1

-- Theorem: The polar equation ρ = 1 represents a circle
theorem polarEqIsCircle : isCircle (fun (x, y) ↦ ∃ p : PolarCoord, polarEq p ∧ x = p.ρ * Real.cos p.θ ∧ y = p.ρ * Real.sin p.θ) := by
  sorry


end NUMINAMATH_CALUDE_polarEqIsCircle_l723_72352


namespace NUMINAMATH_CALUDE_exists_tangent_circle_l723_72394

/-- Two parallel lines in a plane -/
structure ParallelLines where
  distance : ℝ
  distance_pos : distance > 0

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The configuration of our geometry problem -/
structure Configuration where
  lines : ParallelLines
  given_circle : Circle
  circle_between_lines : given_circle.center.2 > 0 ∧ given_circle.center.2 < lines.distance

/-- The theorem stating the existence of the sought circle -/
theorem exists_tangent_circle (config : Configuration) :
  ∃ (tangent_circle : Circle),
    tangent_circle.radius = config.lines.distance / 2 ∧
    (tangent_circle.center.2 = config.lines.distance / 2 ∨
     tangent_circle.center.2 = config.lines.distance / 2) ∧
    ((tangent_circle.center.1 - config.given_circle.center.1) ^ 2 +
     (tangent_circle.center.2 - config.given_circle.center.2) ^ 2 =
     (tangent_circle.radius + config.given_circle.radius) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_exists_tangent_circle_l723_72394


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l723_72390

theorem lcm_gcf_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l723_72390


namespace NUMINAMATH_CALUDE_no_prime_pair_divisibility_l723_72379

theorem no_prime_pair_divisibility : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pair_divisibility_l723_72379


namespace NUMINAMATH_CALUDE_f_positive_at_one_f_solution_set_l723_72331

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

-- Theorem 1
theorem f_positive_at_one (a : ℝ) :
  f a 1 > 0 ↔ a ∈ Set.Ioo (3 - 2 * Real.sqrt 3) (3 + 2 * Real.sqrt 3) :=
sorry

-- Theorem 2
theorem f_solution_set (a b : ℝ) :
  (∀ x, f a x > b ↔ x ∈ Set.Ioo (-1) 3) ↔
  ((a = 3 - Real.sqrt 3 ∨ a = 3 + Real.sqrt 3) ∧ b = -3) :=
sorry

end NUMINAMATH_CALUDE_f_positive_at_one_f_solution_set_l723_72331


namespace NUMINAMATH_CALUDE_area_of_larger_rectangle_l723_72325

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The area of a larger rectangle formed by six identical smaller rectangles -/
def largerRectangleArea (smallRect : Rectangle) : ℝ :=
  (3 * smallRect.width) * (2 * smallRect.length)

theorem area_of_larger_rectangle :
  ∀ (smallRect : Rectangle),
    smallRect.length = 2 * smallRect.width →
    smallRect.length + smallRect.width = 21 →
    largerRectangleArea smallRect = 588 := by
  sorry

end NUMINAMATH_CALUDE_area_of_larger_rectangle_l723_72325


namespace NUMINAMATH_CALUDE_smallest_m_with_divisible_digit_sum_l723_72308

/-- Represents the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Checks if a number's digit sum is divisible by 6 -/
def hasSumDivisibleBy6 (n : ℕ) : Prop :=
  (digitSum n) % 6 = 0

/-- Main theorem: 9 is the smallest m satisfying the condition -/
theorem smallest_m_with_divisible_digit_sum : 
  ∀ (start : ℕ), ∃ (i : ℕ), i < 9 ∧ hasSumDivisibleBy6 (start + i) ∧
  ∀ (m : ℕ), m < 9 → ∃ (start' : ℕ), ∀ (j : ℕ), j < m → ¬hasSumDivisibleBy6 (start' + j) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_with_divisible_digit_sum_l723_72308


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l723_72311

theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_milk : ℝ) 
  (new_ratio : ℝ) :
  total_volume = 20 →
  added_milk = 5 →
  new_ratio = 4 →
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_volume ∧
    (initial_milk + added_milk) / initial_water = new_ratio ∧
    initial_milk / initial_water = 3 := by
sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l723_72311


namespace NUMINAMATH_CALUDE_binomial_coefficient_fifth_power_fourth_term_l723_72356

theorem binomial_coefficient_fifth_power_fourth_term : 
  Nat.choose 5 3 = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_fifth_power_fourth_term_l723_72356


namespace NUMINAMATH_CALUDE_trigonometric_product_transformation_l723_72374

theorem trigonometric_product_transformation (α : ℝ) :
  4.66 * Real.sin (5 * π / 2 + 4 * α) - Real.sin (5 * π / 2 + 2 * α) ^ 6 + Real.cos (7 * π / 2 - 2 * α) ^ 6 = 
  (1 / 8) * Real.sin (4 * α) * Real.sin (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_transformation_l723_72374


namespace NUMINAMATH_CALUDE_root_permutation_l723_72386

theorem root_permutation (r s t : ℝ) : 
  (r^3 - 21*r + 35 = 0) → 
  (s^3 - 21*s + 35 = 0) → 
  (t^3 - 21*t + 35 = 0) → 
  (r ≠ s) → (s ≠ t) → (t ≠ r) →
  (r^2 + 2*r - 14 = s) ∧ 
  (s^2 + 2*s - 14 = t) ∧ 
  (t^2 + 2*t - 14 = r) := by
sorry

end NUMINAMATH_CALUDE_root_permutation_l723_72386


namespace NUMINAMATH_CALUDE_courtyard_paving_l723_72312

/-- Calculates the number of bricks required to pave a courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℝ) :
  courtyard_length = 35 ∧ 
  courtyard_width = 24 ∧ 
  brick_length = 0.15 ∧ 
  brick_width = 0.08 →
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 70000 := by
  sorry

#check courtyard_paving

end NUMINAMATH_CALUDE_courtyard_paving_l723_72312


namespace NUMINAMATH_CALUDE_nancy_carrot_count_l723_72348

/-- Calculates the number of good-quality carrots Nancy has at the end -/
def nancys_carrots (initial_carrots : ℕ) (kept_carrots : ℕ) (planted_seeds : ℕ) (growth_factor : ℕ) (poor_quality_ratio : ℕ) : ℕ :=
  let new_carrots := planted_seeds * growth_factor
  let total_carrots := (initial_carrots - kept_carrots - planted_seeds) + new_carrots + kept_carrots
  let poor_quality_carrots := total_carrots / poor_quality_ratio
  total_carrots - poor_quality_carrots

theorem nancy_carrot_count :
  nancys_carrots 12 2 5 3 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrot_count_l723_72348


namespace NUMINAMATH_CALUDE_albert_pizza_consumption_l723_72329

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The number of large pizzas Albert buys -/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas Albert buys -/
def num_small_pizzas : ℕ := 2

/-- The total number of slices Albert eats -/
def total_slices : ℕ := num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices

theorem albert_pizza_consumption :
  total_slices = 48 := by
  sorry

end NUMINAMATH_CALUDE_albert_pizza_consumption_l723_72329


namespace NUMINAMATH_CALUDE_max_y_proof_unique_x_exists_no_greater_y_l723_72343

/-- The maximum value of y such that there exists a unique x satisfying the given inequality -/
def max_y : ℕ := 112

theorem max_y_proof :
  ∀ y : ℕ, y > max_y →
    ¬(∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + y) ∧ (x:ℚ)/(x + y) < 8/15) :=
by sorry

theorem unique_x_exists :
  ∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + max_y) ∧ (x:ℚ)/(x + max_y) < 8/15 :=
by sorry

theorem no_greater_y :
  ∀ y : ℕ, y > max_y →
    ¬(∃! x : ℕ, (9:ℚ)/17 < (x:ℚ)/(x + y) ∧ (x:ℚ)/(x + y) < 8/15) :=
by sorry

end NUMINAMATH_CALUDE_max_y_proof_unique_x_exists_no_greater_y_l723_72343


namespace NUMINAMATH_CALUDE_unique_volume_constraint_l723_72300

def box_volume (x : ℕ) : ℕ := (x + 5) * (x - 5) * (x^2 + 5*x)

theorem unique_volume_constraint : ∃! x : ℕ, x > 5 ∧ box_volume x < 1000 := by
  sorry

end NUMINAMATH_CALUDE_unique_volume_constraint_l723_72300


namespace NUMINAMATH_CALUDE_ab_bc_ratio_l723_72345

/-- A rectangle divided into five congruent rectangles -/
structure DividedRectangle where
  -- The width of each congruent rectangle
  x : ℝ
  -- Assumption that x is positive
  x_pos : x > 0

/-- The length of side AB in the divided rectangle -/
def length_AB (r : DividedRectangle) : ℝ := 5 * r.x

/-- The length of side BC in the divided rectangle -/
def length_BC (r : DividedRectangle) : ℝ := 3 * r.x

/-- Theorem stating that the ratio of AB to BC is 5:3 -/
theorem ab_bc_ratio (r : DividedRectangle) :
  length_AB r / length_BC r = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_bc_ratio_l723_72345


namespace NUMINAMATH_CALUDE_no_perfect_square_300_ones_l723_72361

/-- Represents the count of digits '1' in the decimal representation of a number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- Checks if a number's decimal representation contains only '0' and '1' -/
def only_zero_and_one (n : ℕ) : Prop := sorry

/-- Theorem: There does not exist a perfect square integer with exactly 300 digits of '1' 
    and no other digits except '0' in its decimal representation -/
theorem no_perfect_square_300_ones : 
  ¬ ∃ (n : ℕ), count_ones n = 300 ∧ only_zero_and_one n ∧ ∃ (k : ℕ), n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_300_ones_l723_72361


namespace NUMINAMATH_CALUDE_f_decreasing_when_a_negative_l723_72328

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- Theorem statement
theorem f_decreasing_when_a_negative (a : ℝ) (h1 : a ≠ 0) (h2 : a < 0) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_when_a_negative_l723_72328


namespace NUMINAMATH_CALUDE_simplify_expressions_l723_72309

variable (x y a : ℝ)

theorem simplify_expressions :
  (5 * x - 3 * (2 * x - 3 * y) + x = 9 * y) ∧
  (3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3) := by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l723_72309


namespace NUMINAMATH_CALUDE_race_length_for_simultaneous_finish_l723_72318

theorem race_length_for_simultaneous_finish 
  (speed_ratio : ℝ) 
  (head_start : ℝ) 
  (race_length : ℝ) : 
  speed_ratio = 4 →
  head_start = 63 →
  race_length / speed_ratio = (race_length - head_start) / 1 →
  race_length = 84 := by
sorry

end NUMINAMATH_CALUDE_race_length_for_simultaneous_finish_l723_72318


namespace NUMINAMATH_CALUDE_fifi_green_hangers_l723_72337

/-- The number of green hangers in Fifi's closet -/
def green_hangers : ℕ := 4

/-- The number of pink hangers in Fifi's closet -/
def pink_hangers : ℕ := 7

/-- The number of blue hangers in Fifi's closet -/
def blue_hangers : ℕ := green_hangers - 1

/-- The number of yellow hangers in Fifi's closet -/
def yellow_hangers : ℕ := blue_hangers - 1

/-- The total number of hangers in Fifi's closet -/
def total_hangers : ℕ := 16

theorem fifi_green_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = total_hangers :=
by sorry

end NUMINAMATH_CALUDE_fifi_green_hangers_l723_72337


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l723_72393

/-- The maximum number of cubes that can fit in a rectangular box -/
def max_cubes (box_length box_width box_height cube_volume : ℕ) : ℕ :=
  (box_length * box_width * box_height) / cube_volume

/-- Theorem stating the maximum number of 43 cm³ cubes in a 13x17x22 cm box -/
theorem max_cubes_in_box : max_cubes 13 17 22 43 = 114 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_in_box_l723_72393


namespace NUMINAMATH_CALUDE_sum_of_cubes_l723_72354

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l723_72354


namespace NUMINAMATH_CALUDE_assignments_for_twenty_points_l723_72358

/-- Calculates the number of assignments required for a given number of points -/
def assignments_required (points : ℕ) : ℕ :=
  let segments := (points + 3) / 4
  (segments * (segments + 1) * 2) 

/-- The theorem stating that 60 assignments are required for 20 points -/
theorem assignments_for_twenty_points :
  assignments_required 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_assignments_for_twenty_points_l723_72358


namespace NUMINAMATH_CALUDE_tangent_line_implies_sum_l723_72355

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := (a * x - 1) * Real.log x + b

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := a * Real.log x + (a * x - 1) / x

theorem tangent_line_implies_sum (a b : ℝ) : 
  (∀ x, f_derivative a x = f a b x) →  -- f_derivative is the derivative of f
  f_derivative a 1 = -a →              -- Slope condition at x = 1
  f a b 1 = -a + 1 →                   -- Point condition at x = 1
  a + b = 1 := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_implies_sum_l723_72355


namespace NUMINAMATH_CALUDE_original_car_price_l723_72349

theorem original_car_price (used_price : ℝ) (percentage : ℝ) (original_price : ℝ) : 
  used_price = 15000 →
  percentage = 0.40 →
  used_price = percentage * original_price →
  original_price = 37500 := by
sorry

end NUMINAMATH_CALUDE_original_car_price_l723_72349


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l723_72351

def total_members : ℕ := 18
def officer_positions : ℕ := 6
def past_officers : ℕ := 8

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem officer_selection_theorem : 
  choose total_members officer_positions - 
  (choose (total_members - past_officers) officer_positions + 
   past_officers * choose (total_members - past_officers) (officer_positions - 1)) = 16338 :=
sorry

end NUMINAMATH_CALUDE_officer_selection_theorem_l723_72351


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l723_72368

/-- The distance between two vehicles moving in the same direction after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between a truck moving at 65 km/h and a car moving at 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_hours : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_three_minutes_l723_72368


namespace NUMINAMATH_CALUDE_no_house_spirits_l723_72384

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (HouseSpirit : U → Prop)
variable (LovesMischief : U → Prop)
variable (LovesCleanlinessAndOrder : U → Prop)

-- State the theorem
theorem no_house_spirits
  (h1 : ∀ x, HouseSpirit x → LovesMischief x)
  (h2 : ∀ x, HouseSpirit x → LovesCleanlinessAndOrder x)
  (h3 : ∀ x, LovesCleanlinessAndOrder x → ¬LovesMischief x) :
  ¬∃ x, HouseSpirit x :=
by sorry

end NUMINAMATH_CALUDE_no_house_spirits_l723_72384


namespace NUMINAMATH_CALUDE_a_not_square_l723_72307

/-- Sequence definition -/
def a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => a n + 2 / (a n)

/-- Theorem statement -/
theorem a_not_square : ∀ n : ℕ, ¬ ∃ q : ℚ, a n = q ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_not_square_l723_72307


namespace NUMINAMATH_CALUDE_quarters_found_l723_72303

def dime_value : ℚ := 0.1
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def quarter_value : ℚ := 0.25

def num_dimes : ℕ := 3
def num_nickels : ℕ := 4
def num_pennies : ℕ := 200
def total_amount : ℚ := 5

theorem quarters_found :
  ∃ (num_quarters : ℕ),
    (num_quarters : ℚ) * quarter_value +
    (num_dimes : ℚ) * dime_value +
    (num_nickels : ℚ) * nickel_value +
    (num_pennies : ℚ) * penny_value = total_amount ∧
    num_quarters = 10 :=
by sorry

end NUMINAMATH_CALUDE_quarters_found_l723_72303


namespace NUMINAMATH_CALUDE_marbles_in_first_jar_l723_72396

theorem marbles_in_first_jar (jar1 jar2 jar3 : ℕ) : 
  jar2 = 2 * jar1 →
  jar3 = jar1 / 4 →
  jar1 + jar2 + jar3 = 260 →
  jar1 = 80 := by
sorry

end NUMINAMATH_CALUDE_marbles_in_first_jar_l723_72396


namespace NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l723_72387

/-- The volume of a cube with total edge length of 48 cm is 64 cubic centimeters. -/
theorem cube_volume_from_total_edge_length :
  ∀ (edge_length : ℝ),
  12 * edge_length = 48 →
  edge_length ^ 3 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l723_72387


namespace NUMINAMATH_CALUDE_weighted_sum_inequality_l723_72316

theorem weighted_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (sum_geq_one : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_weighted_sum_inequality_l723_72316


namespace NUMINAMATH_CALUDE_gambler_winning_percentage_l723_72397

/-- Calculates the final winning percentage of a gambler --/
theorem gambler_winning_percentage
  (initial_games : ℕ)
  (initial_win_rate : ℚ)
  (additional_games : ℕ)
  (new_win_rate : ℚ)
  (h1 : initial_games = 30)
  (h2 : initial_win_rate = 2/5)
  (h3 : additional_games = 30)
  (h4 : new_win_rate = 4/5) :
  let total_games := initial_games + additional_games
  let total_wins := initial_games * initial_win_rate + additional_games * new_win_rate
  total_wins / total_games = 3/5 := by
sorry

#eval (2/5 : ℚ)  -- To verify that 2/5 is indeed 0.4
#eval (4/5 : ℚ)  -- To verify that 4/5 is indeed 0.8
#eval (3/5 : ℚ)  -- To verify that 3/5 is indeed 0.6

end NUMINAMATH_CALUDE_gambler_winning_percentage_l723_72397


namespace NUMINAMATH_CALUDE_interior_triangle_perimeter_is_715_l723_72310

/-- Triangle ABC with parallel lines forming interior triangle XYZ -/
structure ParallelLineTriangle where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of BC -/
  bc : ℝ
  /-- Side length of AC -/
  ac : ℝ
  /-- Length of intersection of ℓA with interior of triangle ABC -/
  ℓa_intersection : ℝ
  /-- Length of intersection of ℓB with interior of triangle ABC -/
  ℓb_intersection : ℝ
  /-- Length of intersection of ℓC with interior of triangle ABC -/
  ℓc_intersection : ℝ

/-- Perimeter of the interior triangle XYZ formed by lines ℓA, ℓB, and ℓC -/
def interior_triangle_perimeter (t : ParallelLineTriangle) : ℝ := sorry

/-- Theorem stating that the perimeter of the interior triangle is 715 for the given conditions -/
theorem interior_triangle_perimeter_is_715 (t : ParallelLineTriangle) 
  (h1 : t.ab = 120)
  (h2 : t.bc = 220)
  (h3 : t.ac = 180)
  (h4 : t.ℓa_intersection = 55)
  (h5 : t.ℓb_intersection = 45)
  (h6 : t.ℓc_intersection = 15) :
  interior_triangle_perimeter t = 715 := by sorry

end NUMINAMATH_CALUDE_interior_triangle_perimeter_is_715_l723_72310


namespace NUMINAMATH_CALUDE_system_infinite_solutions_l723_72370

/-- A system of two linear equations in two variables -/
structure LinearSystem where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The system has infinite solutions -/
def HasInfiniteSolutions (sys : LinearSystem) : Prop :=
  ∃ (t : ℝ), sys.a * sys.d = sys.b * sys.c ∧ sys.a * sys.f = sys.c * sys.e

/-- The given system of equations -/
def givenSystem (k : ℝ) : LinearSystem where
  a := 2
  b := -3
  c := 4
  d := -6
  e := 5
  f := k

/-- Theorem: The given system has infinite solutions if and only if k = 10 -/
theorem system_infinite_solutions :
  ∀ k, HasInfiniteSolutions (givenSystem k) ↔ k = 10 := by sorry

end NUMINAMATH_CALUDE_system_infinite_solutions_l723_72370


namespace NUMINAMATH_CALUDE_odd_prime_divisibility_l723_72305

theorem odd_prime_divisibility (p a b c : ℤ) : 
  Prime p → 
  Odd p → 
  (p ∣ a^2023 + b^2023) → 
  (p ∣ b^2024 + c^2024) → 
  (p ∣ a^2025 + c^2025) → 
  (p ∣ a) ∧ (p ∣ b) ∧ (p ∣ c) := by
sorry

end NUMINAMATH_CALUDE_odd_prime_divisibility_l723_72305


namespace NUMINAMATH_CALUDE_universal_transportation_method_l723_72372

-- Define a type for cities
variable {City : Type}

-- Define a relation for connectivity between cities
variable (connected : City → City → Prop)

-- Define air and water connectivity
variable (air_connected : City → City → Prop)
variable (water_connected : City → City → Prop)

-- Axiom: Any two cities are connected by either air or water
axiom connectivity : ∀ (c1 c2 : City), c1 ≠ c2 → air_connected c1 c2 ∨ water_connected c1 c2

-- Define the theorem
theorem universal_transportation_method 
  (h : ∀ (c1 c2 : City), connected c1 c2 ↔ (air_connected c1 c2 ∨ water_connected c1 c2)) :
  (∀ (c1 c2 : City), air_connected c1 c2) ∨ (∀ (c1 c2 : City), water_connected c1 c2) :=
sorry

end NUMINAMATH_CALUDE_universal_transportation_method_l723_72372


namespace NUMINAMATH_CALUDE_gcd_35_and_number_between_80_90_l723_72376

theorem gcd_35_and_number_between_80_90 :
  ∃! n : ℕ, 80 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 35 n = 7 :=
by sorry

end NUMINAMATH_CALUDE_gcd_35_and_number_between_80_90_l723_72376


namespace NUMINAMATH_CALUDE_sphere_to_cone_height_l723_72392

theorem sphere_to_cone_height (R : ℝ) (h : ℝ) (r : ℝ) (l : ℝ) : 
  R > 0 → r > 0 → h > 0 → l > 0 →
  (4 / 3) * Real.pi * R^3 = (1 / 3) * Real.pi * r^2 * h →  -- Volume conservation
  Real.pi * r * l = 3 * Real.pi * r^2 →  -- Lateral surface area condition
  l^2 = r^2 + h^2 →  -- Pythagorean theorem
  h = 4 * R * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_to_cone_height_l723_72392


namespace NUMINAMATH_CALUDE_lilly_buys_seven_flowers_l723_72315

/-- Calculates the number of flowers Lilly can buy for Maria's birthday --/
def flowers_for_maria (days : ℕ) (daily_savings : ℕ) (wrapping_cost : ℕ) (other_costs : ℕ) (flower_cost : ℕ) : ℕ :=
  let total_savings := days * daily_savings
  let total_expenses := wrapping_cost + other_costs
  let money_for_flowers := total_savings - total_expenses
  money_for_flowers / flower_cost

/-- Theorem stating that Lilly can buy 7 flowers for Maria's birthday --/
theorem lilly_buys_seven_flowers :
  flowers_for_maria 22 2 6 10 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lilly_buys_seven_flowers_l723_72315
