import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l3434_343491

theorem inequality_solution_set (m : ℝ) : 
  (∃ (a b c : ℤ), (∀ x : ℝ, (x^2 - 2*x + m ≤ 0) ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
   (∀ y : ℤ, (y^2 - 2*y + m ≤ 0) → (y = a ∨ y = b ∨ y = c))) ↔ 
  (m = -2 ∨ m = 0) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3434_343491


namespace NUMINAMATH_CALUDE_lindas_substitution_l3434_343404

theorem lindas_substitution (a b c d : ℕ) (e : ℝ) : 
  a = 120 → b = 5 → c = 4 → d = 10 →
  (a / b * c + d - e : ℝ) = (a / (b * (c + (d - e)))) →
  e = 16 := by
sorry

end NUMINAMATH_CALUDE_lindas_substitution_l3434_343404


namespace NUMINAMATH_CALUDE_tax_amount_l3434_343422

def gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def net_pay : ℝ := 740

theorem tax_amount : 
  gross_pay * (1 - retirement_rate) - net_pay = 100 := by
  sorry

end NUMINAMATH_CALUDE_tax_amount_l3434_343422


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l3434_343498

theorem profit_percentage_previous_year
  (revenue_prev : ℝ)
  (profit_prev : ℝ)
  (revenue_decrease : ℝ)
  (profit_percentage_2009 : ℝ)
  (profit_increase : ℝ)
  (h1 : revenue_decrease = 0.2)
  (h2 : profit_percentage_2009 = 0.15)
  (h3 : profit_increase = 1.5)
  (h4 : profit_prev > 0)
  (h5 : revenue_prev > 0) :
  profit_prev / revenue_prev = 0.08 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l3434_343498


namespace NUMINAMATH_CALUDE_monster_feast_l3434_343403

theorem monster_feast (sequence : Fin 3 → ℕ) 
  (double_next : ∀ i : Fin 2, sequence (Fin.succ i) = 2 * sequence i)
  (total_consumed : sequence 0 + sequence 1 + sequence 2 = 847) :
  sequence 0 = 121 := by
sorry

end NUMINAMATH_CALUDE_monster_feast_l3434_343403


namespace NUMINAMATH_CALUDE_expected_value_of_coin_flips_l3434_343499

def penny : ℚ := 1
def nickel : ℚ := 5
def dime : ℚ := 10
def quarter : ℚ := 25
def half_dollar : ℚ := 50
def dollar : ℚ := 100

def coin_flip_probability : ℚ := 1/2

theorem expected_value_of_coin_flips :
  coin_flip_probability * (penny + nickel + dime + quarter + half_dollar + dollar) = 95.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_coin_flips_l3434_343499


namespace NUMINAMATH_CALUDE_applesauce_ratio_l3434_343439

/-- Given a box of apples and information about pie-making, calculate the ratio of apples used for applesauce to the total weight of apples. -/
theorem applesauce_ratio (total_weight : ℝ) (weight_per_pie : ℝ) (num_pies : ℕ) 
  (h1 : total_weight = 120)
  (h2 : weight_per_pie = 4)
  (h3 : num_pies = 15) :
  (total_weight - weight_per_pie * num_pies) / total_weight = 1 / 2 := by
  sorry

#check applesauce_ratio

end NUMINAMATH_CALUDE_applesauce_ratio_l3434_343439


namespace NUMINAMATH_CALUDE_chair_table_price_percentage_l3434_343430

/-- The price of a chair in dollars -/
def chair_price : ℚ := (96 - 84)

/-- The price of a table in dollars -/
def table_price : ℚ := 84

/-- The price of 2 chairs and 1 table -/
def price_2c1t : ℚ := 2 * chair_price + table_price

/-- The price of 1 chair and 2 tables -/
def price_1c2t : ℚ := chair_price + 2 * table_price

/-- The percentage of price_2c1t to price_1c2t -/
def percentage : ℚ := price_2c1t / price_1c2t * 100

theorem chair_table_price_percentage :
  percentage = 60 := by sorry

end NUMINAMATH_CALUDE_chair_table_price_percentage_l3434_343430


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l3434_343434

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l3434_343434


namespace NUMINAMATH_CALUDE_salary_and_new_savings_l3434_343473

/-- Represents expenses as percentages of salary -/
structure Expenses where
  food : ℚ
  rent : ℚ
  entertainment : ℚ
  conveyance : ℚ
  utilities : ℚ
  miscellaneous : ℚ

/-- Calculates the total expenses as a percentage -/
def totalExpenses (e : Expenses) : ℚ :=
  e.food + e.rent + e.entertainment + e.conveyance + e.utilities + e.miscellaneous

/-- Calculates the savings percentage -/
def savingsPercentage (e : Expenses) : ℚ :=
  1 - totalExpenses e

/-- Theorem: Given the initial expenses and savings, prove the monthly salary and new savings percentage -/
theorem salary_and_new_savings 
  (initial_expenses : Expenses)
  (initial_savings : ℚ)
  (salary : ℚ)
  (new_entertainment : ℚ)
  (new_conveyance : ℚ)
  (h1 : initial_expenses.food = 0.30)
  (h2 : initial_expenses.rent = 0.25)
  (h3 : initial_expenses.entertainment = 0.15)
  (h4 : initial_expenses.conveyance = 0.10)
  (h5 : initial_expenses.utilities = 0.05)
  (h6 : initial_expenses.miscellaneous = 0.05)
  (h7 : initial_savings = 1500)
  (h8 : savingsPercentage initial_expenses * salary = initial_savings)
  (h9 : new_entertainment = initial_expenses.entertainment + 0.05)
  (h10 : new_conveyance = initial_expenses.conveyance - 0.03)
  : salary = 15000 ∧ 
    savingsPercentage { initial_expenses with 
      entertainment := new_entertainment,
      conveyance := new_conveyance 
    } = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_salary_and_new_savings_l3434_343473


namespace NUMINAMATH_CALUDE_transaction_gain_per_year_l3434_343436

/-- Calculate simple interest -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  (principal * rate * time) / 100

theorem transaction_gain_per_year 
  (principal : ℚ) 
  (borrowRate lendRate : ℚ) 
  (time : ℚ) 
  (h1 : principal = 5000)
  (h2 : borrowRate = 4)
  (h3 : lendRate = 6)
  (h4 : time = 2) :
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time = 200 := by
  sorry

end NUMINAMATH_CALUDE_transaction_gain_per_year_l3434_343436


namespace NUMINAMATH_CALUDE_age_difference_l3434_343492

theorem age_difference (P M Mo : ℕ) 
  (h1 : P * 5 = M * 3) 
  (h2 : M * 5 = Mo * 3) 
  (h3 : P + M + Mo = 196) : 
  Mo - P = 64 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3434_343492


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3434_343468

theorem election_votes_theorem (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 360) →
  total_votes = 900 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3434_343468


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l3434_343440

/-- Given a point M with rectangular coordinates (√3, 1, -2),
    prove that its cylindrical coordinates are (2, π/6, -2) -/
theorem rectangular_to_cylindrical :
  let x : ℝ := Real.sqrt 3
  let y : ℝ := 1
  let z : ℝ := -2
  let ρ : ℝ := 2
  let θ : ℝ := π / 6
  x = ρ * Real.cos θ ∧
  y = ρ * Real.sin θ ∧
  z = -2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l3434_343440


namespace NUMINAMATH_CALUDE_largest_number_proof_l3434_343416

theorem largest_number_proof (a b c d : ℕ) 
  (sum1 : a + b + c = 180)
  (sum2 : a + b + d = 197)
  (sum3 : a + c + d = 208)
  (sum4 : b + c + d = 222) :
  max a (max b (max c d)) = 89 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l3434_343416


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3434_343467

/-- The minimum distance between the curves y = e^(3x + 11) and y = (ln x - 11) / 3 -/
theorem min_distance_between_curves : ∃ d : ℝ, d > 0 ∧
  (∀ x y z : ℝ, y = Real.exp (3 * x + 11) ∧ z = (Real.log y - 11) / 3 →
    d ≤ Real.sqrt ((x - y)^2 + (y - z)^2)) ∧
  d = Real.sqrt 2 * (Real.log 3 + 12) / 3 :=
sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3434_343467


namespace NUMINAMATH_CALUDE_surface_area_difference_l3434_343431

def rectangular_solid_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

def cube_surface_area (s : ℝ) : ℝ :=
  6 * s^2

def new_exposed_area (s : ℝ) : ℝ :=
  3 * s^2

theorem surface_area_difference :
  let original_area := rectangular_solid_surface_area 4 5 6
  let removed_area := cube_surface_area 2
  let exposed_area := new_exposed_area 2
  original_area - removed_area + exposed_area = original_area - 12
  := by sorry

end NUMINAMATH_CALUDE_surface_area_difference_l3434_343431


namespace NUMINAMATH_CALUDE_equal_fractions_l3434_343417

theorem equal_fractions (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  let f1 := (x + y) / (x^2 + x*y + y^2)
  let f2 := (y + z) / (y^2 + y*z + z^2)
  let f3 := (z + x) / (z^2 + z*x + x^2)
  (f1 = f2 ∨ f2 = f3 ∨ f3 = f1) → (f1 = f2 ∧ f2 = f3) :=
by sorry

end NUMINAMATH_CALUDE_equal_fractions_l3434_343417


namespace NUMINAMATH_CALUDE_three_sum_exists_l3434_343461

theorem three_sum_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_strict_increasing : ∀ i j : Fin (n + 1), i < j → a i < a j)
  (h_upper_bound : ∀ i : Fin (n + 1), a i < 2 * n) :
  ∃ i j k : Fin (n + 1), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j = a k :=
by sorry

end NUMINAMATH_CALUDE_three_sum_exists_l3434_343461


namespace NUMINAMATH_CALUDE_dad_steps_l3434_343443

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l3434_343443


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3434_343426

/-- The decimal representation of a repeating decimal ending in 6 -/
def S : ℚ := 0.666666

/-- Theorem stating that the decimal 0.666... is equal to 2/3 -/
theorem decimal_to_fraction : S = 2/3 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3434_343426


namespace NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l3434_343463

/-- Given a line segment with one endpoint (6, 1) and midpoint (5, 7),
    the sum of the coordinates of the other endpoint is 17. -/
theorem endpoint_sum : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun x1 y1 mx my x2 y2 =>
    x1 = 6 ∧ y1 = 1 ∧ mx = 5 ∧ my = 7 ∧
    (x1 + x2) / 2 = mx ∧ (y1 + y2) / 2 = my →
    x2 + y2 = 17

theorem endpoint_sum_proof : endpoint_sum 6 1 5 7 4 13 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l3434_343463


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3434_343423

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Theorem: If a_1, a_3, and a_7 of an arithmetic sequence form a geometric sequence,
    then the common ratio of this geometric sequence is 2 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 3) ^ 2 = (seq.a 1) * (seq.a 7)) :
  (seq.a 3) / (seq.a 1) = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3434_343423


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3434_343413

-- Define the sets M and N
def M : Set ℝ := {x | 3 * x - x^2 > 0}
def N : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3434_343413


namespace NUMINAMATH_CALUDE_mean_calculation_l3434_343438

theorem mean_calculation (x : ℝ) :
  (28 + x + 50 + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
  sorry

end NUMINAMATH_CALUDE_mean_calculation_l3434_343438


namespace NUMINAMATH_CALUDE_max_cables_sixty_cables_achievable_l3434_343406

/-- Represents the network of computers in the organization -/
structure ComputerNetwork where
  total_employees : ℕ
  brand_a_computers : ℕ
  brand_b_computers : ℕ
  cables : ℕ

/-- Predicate to check if the network satisfies the given conditions -/
def valid_network (n : ComputerNetwork) : Prop :=
  n.total_employees = 50 ∧
  n.brand_a_computers = 30 ∧
  n.brand_b_computers = 20 ∧
  n.cables ≤ n.brand_a_computers * n.brand_b_computers ∧
  n.cables ≥ 2 * n.brand_a_computers

/-- Predicate to check if all employees can communicate -/
def all_can_communicate (n : ComputerNetwork) : Prop :=
  n.cables ≥ n.total_employees - 1

/-- Theorem stating the maximum number of cables -/
theorem max_cables (n : ComputerNetwork) :
  valid_network n → all_can_communicate n → n.cables ≤ 60 :=
by
  sorry

/-- Theorem stating that 60 cables is achievable -/
theorem sixty_cables_achievable :
  ∃ n : ComputerNetwork, valid_network n ∧ all_can_communicate n ∧ n.cables = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cables_sixty_cables_achievable_l3434_343406


namespace NUMINAMATH_CALUDE_subtract_multiply_real_l3434_343411

theorem subtract_multiply_real : 3.56 - 2.1 * 1.5 = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_subtract_multiply_real_l3434_343411


namespace NUMINAMATH_CALUDE_f_of_one_eq_six_l3434_343455

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem f_of_one_eq_six : f 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_six_l3434_343455


namespace NUMINAMATH_CALUDE_rocket_fuel_ratio_l3434_343485

theorem rocket_fuel_ratio (m M : ℝ) (h : m > 0) :
  2000 * Real.log (1 + M / m) = 12000 → M / m = Real.exp 6 - 1 := by
  sorry

end NUMINAMATH_CALUDE_rocket_fuel_ratio_l3434_343485


namespace NUMINAMATH_CALUDE_pencil_count_l3434_343435

/-- Proves that given the specified costs and quantities, the number of pencils needed is 24 --/
theorem pencil_count (pencil_cost folder_cost total_cost : ℚ) (folder_count : ℕ) : 
  pencil_cost = 1/2 →
  folder_cost = 9/10 →
  folder_count = 20 →
  total_cost = 30 →
  (total_cost - folder_cost * folder_count) / pencil_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l3434_343435


namespace NUMINAMATH_CALUDE_problem_statement_l3434_343410

theorem problem_statement (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a*x - 2) ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3434_343410


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3434_343456

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3434_343456


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3434_343437

theorem complex_equation_solution (z : ℂ) : 
  (Complex.I * z = Complex.I + z) → z = (1 - Complex.I) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3434_343437


namespace NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l3434_343419

/-- The line equation ax + by = c forming a triangle with coordinate axes -/
structure TriangleLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the sum of altitudes of the triangle formed by the given line and coordinate axes -/
def sumOfAltitudes (line : TriangleLine) : ℝ :=
  sorry

/-- The specific line 8x + 3y = 48 -/
def specificLine : TriangleLine :=
  { a := 8, b := 3, c := 48 }

theorem sum_of_altitudes_for_specific_line :
  sumOfAltitudes specificLine = 370 / Real.sqrt 292 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l3434_343419


namespace NUMINAMATH_CALUDE_remainder_proof_l3434_343442

theorem remainder_proof (x y : ℤ) 
  (hx : x % 52 = 19) 
  (hy : (3 * y) % 7 = 5) : 
  ((x + 2*y)^2) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3434_343442


namespace NUMINAMATH_CALUDE_quadratic_incenter_on_diagonal_l3434_343428

/-- A quadratic function f(x) = x^2 + ax + b -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The incenter of a triangle -/
def Incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The line y = x -/
def LineYEqX : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

theorem quadratic_incenter_on_diagonal (a b : ℝ) :
  let f := QuadraticFunction a b
  let A := (0, f 0)
  let B := ((- a + Real.sqrt (a^2 - 4*b)) / 2, 0)
  let C := ((- a - Real.sqrt (a^2 - 4*b)) / 2, 0)
  Incenter A B C ∈ LineYEqX →
  a + b + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_incenter_on_diagonal_l3434_343428


namespace NUMINAMATH_CALUDE_books_on_shelf_correct_book_count_l3434_343469

theorem books_on_shelf (initial_figures : ℕ) (added_figures : ℕ) (extra_books : ℕ) : ℕ :=
  let total_figures := initial_figures + added_figures
  let total_books := total_figures + extra_books
  total_books

theorem correct_book_count : books_on_shelf 2 4 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_correct_book_count_l3434_343469


namespace NUMINAMATH_CALUDE_z_less_than_y_l3434_343489

/-- 
Given:
- w is 40% less than u, so w = 0.6u
- u is 40% less than y, so u = 0.6y
- z is greater than w by 50% of w, so z = 1.5w

Prove that z is 46% less than y, which means z = 0.54y
-/
theorem z_less_than_y (y u w z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 1.5 * w) : 
  z = 0.54 * y := by
  sorry

end NUMINAMATH_CALUDE_z_less_than_y_l3434_343489


namespace NUMINAMATH_CALUDE_triangle_side_length_l3434_343425

/-- Represents a triangle with sides a, b, c and median ma from vertex A to midpoint of side BC. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ma : ℝ

/-- The theorem states that for a triangle with sides 6 and 9, and a median of 5,
    the third side has length √134. -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 6)
  (h2 : t.b = 9) 
  (h3 : t.ma = 5) : 
  t.c = Real.sqrt 134 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3434_343425


namespace NUMINAMATH_CALUDE_cracker_sales_percentage_increase_l3434_343487

theorem cracker_sales_percentage_increase
  (total_boxes : ℕ)
  (saturday_boxes : ℕ)
  (h1 : total_boxes = 150)
  (h2 : saturday_boxes = 60) :
  let sunday_boxes := total_boxes - saturday_boxes
  ((sunday_boxes - saturday_boxes) : ℚ) / saturday_boxes * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cracker_sales_percentage_increase_l3434_343487


namespace NUMINAMATH_CALUDE_sphere_radius_is_four_l3434_343479

/-- Represents a truncated cone with given dimensions and a tangent sphere -/
structure TruncatedConeWithSphere where
  baseRadius : ℝ
  topRadius : ℝ
  height : ℝ
  sphereRadius : ℝ

/-- Checks if the given dimensions satisfy the conditions for a truncated cone with a tangent sphere -/
def isValidConfiguration (cone : TruncatedConeWithSphere) : Prop :=
  cone.baseRadius > cone.topRadius ∧
  cone.height > 0 ∧
  cone.sphereRadius > 0 ∧
  -- The sphere is tangent to the top, bottom, and lateral surface
  cone.sphereRadius = cone.height - Real.sqrt ((cone.baseRadius - cone.topRadius)^2 + cone.height^2)

/-- Theorem stating that for a truncated cone with given dimensions and a tangent sphere, the radius of the sphere is 4 -/
theorem sphere_radius_is_four :
  ∀ (cone : TruncatedConeWithSphere),
    cone.baseRadius = 24 ∧
    cone.topRadius = 6 ∧
    cone.height = 20 ∧
    isValidConfiguration cone →
    cone.sphereRadius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_four_l3434_343479


namespace NUMINAMATH_CALUDE_equation_solution_l3434_343451

theorem equation_solution : ∃ x : ℝ, (Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) ∧ (x = 39) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3434_343451


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3434_343449

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3434_343449


namespace NUMINAMATH_CALUDE_insufficient_payment_l3434_343414

def egg_price : ℝ := 3
def pancake_price : ℝ := 2
def cocoa_price : ℝ := 2
def croissant_price : ℝ := 1
def tax_rate : ℝ := 0.07

def initial_order_cost : ℝ := 4 * egg_price + 3 * pancake_price + 5 * cocoa_price + 2 * croissant_price

def additional_order_cost : ℝ := 2 * 3 * pancake_price + 3 * cocoa_price

def total_cost_before_tax : ℝ := initial_order_cost + additional_order_cost

def total_cost_with_tax : ℝ := total_cost_before_tax * (1 + tax_rate)

def payment : ℝ := 50

theorem insufficient_payment : total_cost_with_tax > payment ∧ 
  total_cost_with_tax - payment = 1.36 := by sorry

end NUMINAMATH_CALUDE_insufficient_payment_l3434_343414


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3434_343486

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 1 + a 3 = 2 →
  a 3 + a 5 = 4 →
  a 7 + a 9 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3434_343486


namespace NUMINAMATH_CALUDE_triangle_side_length_l3434_343495

/-- Given a triangle ABC with angle A = π/6, side a = 1, and side b = √3, 
    the length of side c is either 2 or 1. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → a = 1 → b = Real.sqrt 3 → 
  (c = 2 ∨ c = 1) := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3434_343495


namespace NUMINAMATH_CALUDE_tan_negative_23pi_over_3_l3434_343471

theorem tan_negative_23pi_over_3 : Real.tan (-23 * Real.pi / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_23pi_over_3_l3434_343471


namespace NUMINAMATH_CALUDE_acute_triangle_with_largest_five_times_smallest_l3434_343402

theorem acute_triangle_with_largest_five_times_smallest (α β γ : ℕ) : 
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- All angles are positive
  α + β + γ = 180 →  -- Sum of angles in a triangle
  α ≤ 89 ∧ β ≤ 89 ∧ γ ≤ 89 →  -- Acute triangle condition
  α ≥ β ∧ β ≥ γ →  -- Ordering of angles
  α = 5 * γ →  -- Largest angle is five times the smallest
  (α = 85 ∧ β = 78 ∧ γ = 17) := by
  sorry

#check acute_triangle_with_largest_five_times_smallest

end NUMINAMATH_CALUDE_acute_triangle_with_largest_five_times_smallest_l3434_343402


namespace NUMINAMATH_CALUDE_calculation_proof_l3434_343400

theorem calculation_proof : (-1/2)⁻¹ - 4 * Real.cos (30 * π / 180) - (π + 2013)^0 + Real.sqrt 12 = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3434_343400


namespace NUMINAMATH_CALUDE_regular_octagon_area_equals_diagonal_product_l3434_343483

/-- A regular octagon -/
structure RegularOctagon where
  -- We don't need to specify all properties of a regular octagon,
  -- just the existence of such a shape
  dummy : Unit

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The length of the longest diagonal of a regular octagon -/
def longest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The length of the shortest diagonal of a regular octagon -/
def shortest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of a regular octagon is equal to the product of 
    the lengths of its longest and shortest diagonals -/
theorem regular_octagon_area_equals_diagonal_product (o : RegularOctagon) :
  area o = longest_diagonal o * shortest_diagonal o :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_area_equals_diagonal_product_l3434_343483


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3434_343408

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 :=
sorry

theorem equality_condition (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 3) :
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3434_343408


namespace NUMINAMATH_CALUDE_max_p_value_l3434_343401

-- Define the equation
def equation (x p : ℝ) : Prop :=
  2 * Real.cos (2 * Real.pi - Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2)) - 3 =
  p - 2 * Real.sin (-Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2))

-- Define the theorem
theorem max_p_value :
  ∃ (p_max : ℝ), p_max = -2 ∧
  (∀ p : ℝ, (∃ x : ℝ, equation x p) → p ≤ p_max) ∧
  (∃ x : ℝ, equation x p_max) :=
sorry

end NUMINAMATH_CALUDE_max_p_value_l3434_343401


namespace NUMINAMATH_CALUDE_samuels_birds_berries_l3434_343424

/-- The number of berries a single bird eats per day -/
def berries_per_day : ℕ := 7

/-- The number of birds Samuel has -/
def samuels_birds : ℕ := 5

/-- The number of days we're considering -/
def days : ℕ := 4

/-- Theorem: Samuel's birds eat 140 berries in 4 days -/
theorem samuels_birds_berries : 
  berries_per_day * samuels_birds * days = 140 := by
  sorry

end NUMINAMATH_CALUDE_samuels_birds_berries_l3434_343424


namespace NUMINAMATH_CALUDE_pencil_cost_2500_l3434_343432

/-- The cost of buying a certain number of pencils with a discount applied after a threshold -/
def pencil_cost (box_size : ℕ) (box_cost : ℚ) (total_pencils : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let unit_cost := box_cost / box_size
  let regular_cost := min total_pencils discount_threshold * unit_cost
  let discounted_pencils := max (total_pencils - discount_threshold) 0
  let discounted_cost := discounted_pencils * (unit_cost * (1 - discount_rate))
  regular_cost + discounted_cost

theorem pencil_cost_2500 :
  pencil_cost 200 50 2500 1000 (1/10) = 587.5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_2500_l3434_343432


namespace NUMINAMATH_CALUDE_mike_owes_laura_l3434_343459

theorem mike_owes_laura (rate : ℚ) (rooms : ℚ) (total : ℚ) : 
  rate = 13 / 3 → rooms = 8 / 5 → total = rate * rooms → total = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_owes_laura_l3434_343459


namespace NUMINAMATH_CALUDE_streetlights_per_square_l3434_343490

theorem streetlights_per_square 
  (total_streetlights : ℕ) 
  (num_squares : ℕ) 
  (unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : num_squares = 15) 
  (h3 : unused_streetlights = 20) : 
  (total_streetlights - unused_streetlights) / num_squares = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_streetlights_per_square_l3434_343490


namespace NUMINAMATH_CALUDE_certain_number_proof_l3434_343458

theorem certain_number_proof : ∃ x : ℕ, (2994 : ℚ) / x = 177 ∧ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3434_343458


namespace NUMINAMATH_CALUDE_percentage_problem_l3434_343407

theorem percentage_problem (x : ℝ) (P : ℝ) 
  (h1 : x = 180)
  (h2 : P * x = 0.10 * 500 - 5) :
  P = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3434_343407


namespace NUMINAMATH_CALUDE_jill_walking_time_l3434_343433

/-- The time it takes Jill to walk to school given Dave's and Jill's walking parameters -/
theorem jill_walking_time (dave_steps_per_min : ℕ) (dave_step_length : ℕ) (dave_time : ℕ)
  (jill_steps_per_min : ℕ) (jill_step_length : ℕ) 
  (h1 : dave_steps_per_min = 80) (h2 : dave_step_length = 65) (h3 : dave_time = 20)
  (h4 : jill_steps_per_min = 120) (h5 : jill_step_length = 50) :
  (dave_steps_per_min * dave_step_length * dave_time : ℚ) / (jill_steps_per_min * jill_step_length) = 52/3 :=
by sorry

end NUMINAMATH_CALUDE_jill_walking_time_l3434_343433


namespace NUMINAMATH_CALUDE_fourth_root_of_cubic_l3434_343405

theorem fourth_root_of_cubic (c d : ℚ) :
  (∀ x : ℚ, c * x^3 + (c + 3*d) * x^2 + (d - 4*c) * x + (10 - c) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 2 ∨ x = -9/2) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_of_cubic_l3434_343405


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3434_343464

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3434_343464


namespace NUMINAMATH_CALUDE_equal_diff_squares_properties_l3434_343454

-- Definition of "equal difference of squares sequence"
def is_equal_diff_squares (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = p

theorem equal_diff_squares_properties :
  -- Statement 1
  is_equal_diff_squares (fun n => (-1) ^ n) ∧
  -- Statement 2
  (∀ a : ℕ → ℝ, is_equal_diff_squares a →
    ∃ d : ℝ, ∀ n ≥ 2, a n ^ 2 - a (n - 1) ^ 2 = d) ∧
  -- Statement 3
  (∀ a : ℕ → ℝ, is_equal_diff_squares a →
    (∃ d : ℝ, ∀ n ≥ 2, a n - a (n - 1) = d) →
    ∃ c : ℝ, ∀ n, a n = c) ∧
  -- Statement 4
  ∃ a : ℕ → ℝ, is_equal_diff_squares a ∧
    ∀ k : ℕ+, is_equal_diff_squares (fun n => a (k * n)) :=
by sorry

end NUMINAMATH_CALUDE_equal_diff_squares_properties_l3434_343454


namespace NUMINAMATH_CALUDE_strictly_increasing_function_bounds_l3434_343447

theorem strictly_increasing_function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_property : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1) / 2 * n :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_bounds_l3434_343447


namespace NUMINAMATH_CALUDE_point_A_coordinates_l3434_343470

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the left -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- Translate a point upwards -/
def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

/-- The theorem stating the coordinates of point A -/
theorem point_A_coordinates (A : Point) 
  (hB : ∃ d : ℝ, translateLeft A d = Point.mk 1 2)
  (hC : ∃ d : ℝ, translateUp A d = Point.mk 3 4) : 
  A = Point.mk 3 2 := by sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l3434_343470


namespace NUMINAMATH_CALUDE_no_real_solutions_l3434_343444

theorem no_real_solutions (n : ℝ) : 
  (∀ x : ℝ, (x + 6) * (x - 3) ≠ n + 4 * x) ↔ n < -73/4 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3434_343444


namespace NUMINAMATH_CALUDE_largest_number_is_482_l3434_343412

/-- Given a systematic sample from a set of products, this function calculates the largest number in the sample. -/
def largest_sample_number (total_products : ℕ) (smallest_number : ℕ) (second_smallest : ℕ) : ℕ :=
  let sampling_interval := second_smallest - smallest_number
  let sample_size := total_products / sampling_interval
  smallest_number + sampling_interval * (sample_size - 1)

/-- Theorem stating that for the given conditions, the largest number in the sample is 482. -/
theorem largest_number_is_482 :
  largest_sample_number 500 7 32 = 482 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_482_l3434_343412


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l3434_343493

/-- A function that returns the number of ways to write n as the sum of two primes -/
def count_prime_sum_ways (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range n)).card

/-- Theorem stating that 10003 cannot be written as the sum of two primes -/
theorem no_prime_sum_10003 : count_prime_sum_ways 10003 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l3434_343493


namespace NUMINAMATH_CALUDE_point_coordinates_l3434_343474

/-- Given a point A(-m, √m) in the Cartesian coordinate system,
    prove that its coordinates are (-16, 4) if its distance to the x-axis is 4. -/
theorem point_coordinates (m : ℝ) :
  (∃ A : ℝ × ℝ, A = (-m, Real.sqrt m) ∧ |A.2| = 4) →
  (∃ A : ℝ × ℝ, A = (-16, 4)) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3434_343474


namespace NUMINAMATH_CALUDE_max_parts_correct_max_parts_2004_l3434_343460

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating that max_parts gives the correct maximum number of parts -/
theorem max_parts_correct (n : ℕ) : 
  max_parts n = 1 + n * (n + 1) / 2 := by sorry

/-- The specific case for 2004 lines -/
theorem max_parts_2004 : max_parts 2004 = 2009011 := by sorry

end NUMINAMATH_CALUDE_max_parts_correct_max_parts_2004_l3434_343460


namespace NUMINAMATH_CALUDE_ellipse_parabola_shared_focus_eccentricity_l3434_343462

/-- The eccentricity of an ellipse sharing a focus with a parabola -/
theorem ellipse_parabola_shared_focus_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (hab : a > b) 
  (hb : b > 0) : 
  ∃ (x y : ℝ), 
    x^2 = 2*p*y ∧ 
    y^2/a^2 + x^2/b^2 = 1 ∧ 
    (∃ (t : ℝ), x = 2*p*t ∧ y = p*t^2) → 
    Real.sqrt 2 - 1 = Real.sqrt (1 - b^2/a^2) := by
  sorry

#check ellipse_parabola_shared_focus_eccentricity

end NUMINAMATH_CALUDE_ellipse_parabola_shared_focus_eccentricity_l3434_343462


namespace NUMINAMATH_CALUDE_pavan_travel_distance_l3434_343497

theorem pavan_travel_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h_total_time : total_time = 11)
  (h_speed1 : speed1 = 30)
  (h_speed2 : speed2 = 25)
  (h_half_distance : ∀ d : ℝ, d / 2 / speed1 + d / 2 / speed2 = total_time) :
  ∃ d : ℝ, d = 150 ∧ d / 2 / speed1 + d / 2 / speed2 = total_time :=
by sorry

end NUMINAMATH_CALUDE_pavan_travel_distance_l3434_343497


namespace NUMINAMATH_CALUDE_sine_sum_equality_l3434_343465

theorem sine_sum_equality : 
  Real.sin (7 * π / 30) + Real.sin (11 * π / 30) - Real.sin (π / 30) - Real.sin (13 * π / 30) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_equality_l3434_343465


namespace NUMINAMATH_CALUDE_cost_calculation_l3434_343450

/-- The cost of 1 kg of flour in dollars -/
def flour_cost : ℝ := 20.50

/-- The cost relationship between mangos and rice -/
def mango_rice_relation (mango_cost rice_cost : ℝ) : Prop :=
  10 * mango_cost = rice_cost

/-- The cost relationship between flour and rice -/
def flour_rice_relation (rice_cost : ℝ) : Prop :=
  6 * flour_cost = 2 * rice_cost

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost (mango_cost rice_cost : ℝ) : ℝ :=
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost

theorem cost_calculation :
  ∀ mango_cost rice_cost : ℝ,
  mango_rice_relation mango_cost rice_cost →
  flour_rice_relation rice_cost →
  total_cost mango_cost rice_cost = 311.60 := by
sorry

end NUMINAMATH_CALUDE_cost_calculation_l3434_343450


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l3434_343452

theorem geometric_sequence_proof :
  ∃ (q : ℚ) (n : ℕ),
    let a₁ : ℚ := 6
    let S : ℚ := (a₁ * (1 - q^n)) / (1 - q)
    let R : ℚ := ((1 / a₁) * (1 - (1/q)^n)) / (1 - (1/q))
    S = 45/4 ∧
    R = 5/2 ∧
    n = 4 ∧
    q = 1/2 ∧
    [a₁, a₁ * q, a₁ * q^2, a₁ * q^3] = [6, 3, 3/2, 3/4] :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l3434_343452


namespace NUMINAMATH_CALUDE_equation_solution_l3434_343494

theorem equation_solution :
  let f (x : ℝ) := x^2 * (x - 2) - (4 * x^2 + 4)
  ∀ x : ℝ, x ≠ 2 → (f x = 0 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3434_343494


namespace NUMINAMATH_CALUDE_zero_of_f_l3434_343484

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- The zero of f(x) is -1 -/
theorem zero_of_f : f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_zero_of_f_l3434_343484


namespace NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l3434_343496

theorem alternating_sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  |a₀| - |a₁| + |a₂| - |a₃| + |a₄| - |a₅| = 1 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l3434_343496


namespace NUMINAMATH_CALUDE_sum_congruence_modulo_9_l3434_343472

theorem sum_congruence_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_modulo_9_l3434_343472


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_foci_condition_l3434_343477

-- Define the curve C
def C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - k) + p.2^2 / (k - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  (4 - k) * (k - 1) < 0

-- Define what it means for C to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (k : ℝ) : Prop :=
  k - 1 > 0 ∧ 4 - k > 0 ∧ 4 - k > k - 1

-- Theorem 1: If C is a hyperbola, then k < 1 or k > 4
theorem hyperbola_condition (k : ℝ) :
  is_hyperbola k → k < 1 ∨ k > 4 :=
by sorry

-- Theorem 2: If C is an ellipse with foci on the x-axis, then 1 < k < 2.5
theorem ellipse_x_foci_condition (k : ℝ) :
  is_ellipse_x_foci k → 1 < k ∧ k < 2.5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_foci_condition_l3434_343477


namespace NUMINAMATH_CALUDE_probability_shaded_is_half_l3434_343475

/-- Represents a triangle in the diagram -/
structure Triangle where
  is_shaded : Bool

/-- The diagram containing the triangles -/
structure Diagram where
  triangles : Finset Triangle

/-- Calculates the probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  (d.triangles.filter (·.is_shaded)).card / d.triangles.card

/-- The theorem statement -/
theorem probability_shaded_is_half (d : Diagram) :
    d.triangles.card = 4 ∧ 
    (d.triangles.filter (·.is_shaded)).card > 0 →
    probability_shaded d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_shaded_is_half_l3434_343475


namespace NUMINAMATH_CALUDE_paul_bought_45_cookies_l3434_343476

/-- The number of cookies Paul bought -/
def paul_cookies : ℕ := 45

/-- The number of cookies Paula bought -/
def paula_cookies : ℕ := paul_cookies - 3

/-- The total number of cookies bought by Paul and Paula -/
def total_cookies : ℕ := 87

/-- Theorem stating that Paul bought 45 cookies given the conditions -/
theorem paul_bought_45_cookies : 
  paul_cookies = 45 ∧ 
  paula_cookies = paul_cookies - 3 ∧ 
  paul_cookies + paula_cookies = total_cookies :=
sorry

end NUMINAMATH_CALUDE_paul_bought_45_cookies_l3434_343476


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l3434_343420

theorem arthur_walk_distance :
  let blocks_east : ℕ := 8
  let blocks_north : ℕ := 15
  let miles_per_block : ℝ := 0.25
  let miles_east : ℝ := blocks_east * miles_per_block
  let miles_north : ℝ := blocks_north * miles_per_block
  let diagonal_miles : ℝ := Real.sqrt (miles_east^2 + miles_north^2)
  let total_miles : ℝ := miles_east + miles_north + diagonal_miles
  total_miles = 10 := by
sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l3434_343420


namespace NUMINAMATH_CALUDE_final_temp_is_50_l3434_343427

/-- Represents the thermal equilibrium problem with two metal bars and water. -/
structure ThermalEquilibrium where
  initialWaterTemp : ℝ
  initialBarTemp : ℝ
  firstEquilibriumTemp : ℝ

/-- Calculates the final equilibrium temperature after adding the second metal bar. -/
def finalEquilibriumTemp (te : ThermalEquilibrium) : ℝ :=
  sorry

/-- Theorem stating that the final equilibrium temperature is 50°C. -/
theorem final_temp_is_50 (te : ThermalEquilibrium)
    (h1 : te.initialWaterTemp = 80)
    (h2 : te.initialBarTemp = 20)
    (h3 : te.firstEquilibriumTemp = 60) :
  finalEquilibriumTemp te = 50 :=
by sorry

end NUMINAMATH_CALUDE_final_temp_is_50_l3434_343427


namespace NUMINAMATH_CALUDE_box_volume_increase_l3434_343482

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4000)
  (surface_area : 2 * l * w + 2 * w * h + 2 * h * l = 1680)
  (edge_sum : 4 * l + 4 * w + 4 * h = 200) :
  (l + 2) * (w + 3) * (h + 1) = 5736 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3434_343482


namespace NUMINAMATH_CALUDE_angle_sum_sine_l3434_343466

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (t, 2t) where t < 0,
    prove that sin(θ + π/3) = -(2√5 + √15)/10 -/
theorem angle_sum_sine (t : ℝ) (θ : ℝ) (h1 : t < 0) 
    (h2 : Real.cos θ = -1 / Real.sqrt 5) 
    (h3 : Real.sin θ = -2 / Real.sqrt 5) : 
  Real.sin (θ + π/3) = -(2 * Real.sqrt 5 + Real.sqrt 15) / 10 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_sine_l3434_343466


namespace NUMINAMATH_CALUDE_lunch_cost_with_tip_l3434_343409

theorem lunch_cost_with_tip (total_cost : ℝ) (tip_percentage : ℝ) (original_cost : ℝ) :
  total_cost = 58.075 ∧
  tip_percentage = 0.15 ∧
  total_cost = original_cost * (1 + tip_percentage) →
  original_cost = 50.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_with_tip_l3434_343409


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3434_343421

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * Complex.I) * z = 5) : 
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3434_343421


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3434_343429

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∀ k : ℤ, k > 180 → ¬(k ∣ (15*x + 3) * (15*x + 9) * (10*x + 5))) ∧
  (180 ∣ (15*x + 3) * (15*x + 9) * (10*x + 5)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3434_343429


namespace NUMINAMATH_CALUDE_circle_tangent_axes_l3434_343446

/-- Given a point M(x, y) in the first quadrant and a circle passing through M
    that is tangent to both coordinate axes, the product of the radii to the
    points of tangency equals x² + y². -/
theorem circle_tangent_axes (x y r₁ r₂ : ℝ) : 
  x > 0 → y > 0 → 
  ∃ (r : ℝ), (x - r)^2 + (y - r)^2 = r^2 ∧ 
             r₁ + r₂ = 2*r ∧ 
             r₁ * r₂ = r^2 → 
  r₁ * r₂ = x^2 + y^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_axes_l3434_343446


namespace NUMINAMATH_CALUDE_min_packs_for_100_cans_l3434_343488

/-- Represents the number of cans in each pack size -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size -/
def cans_in_pack (p : PackSize) : Nat :=
  match p with
  | PackSize.small => 8
  | PackSize.medium => 14
  | PackSize.large => 28

/-- Represents a combination of packs -/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination -/
def total_cans (c : PackCombination) : Nat :=
  c.small * cans_in_pack PackSize.small +
  c.medium * cans_in_pack PackSize.medium +
  c.large * cans_in_pack PackSize.large

/-- Calculates the total number of packs in a combination -/
def total_packs (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- Predicate to check if a combination is valid (exactly 100 cans) -/
def is_valid_combination (c : PackCombination) : Prop :=
  total_cans c = 100

/-- Theorem: The minimum number of packs to buy exactly 100 cans is 5 -/
theorem min_packs_for_100_cans :
  ∃ (c : PackCombination),
    is_valid_combination c ∧
    total_packs c = 5 ∧
    (∀ (c' : PackCombination), is_valid_combination c' → total_packs c' ≥ 5) :=
  sorry

end NUMINAMATH_CALUDE_min_packs_for_100_cans_l3434_343488


namespace NUMINAMATH_CALUDE_run_distance_proof_l3434_343445

/-- Calculates the total distance run given a running speed and movie lengths -/
def total_distance_run (speed : ℚ) (movie_lengths : List ℚ) : ℚ :=
  (movie_lengths.sum) / speed

/-- Theorem stating that given the specified running speed and movie lengths, 
    the total distance run is 41 miles -/
theorem run_distance_proof : 
  let speed : ℚ := 12  -- 12 minutes per mile
  let movie_lengths : List ℚ := [96, 138, 108, 150]
  total_distance_run speed movie_lengths = 41 := by
  sorry

#eval total_distance_run 12 [96, 138, 108, 150]

end NUMINAMATH_CALUDE_run_distance_proof_l3434_343445


namespace NUMINAMATH_CALUDE_passage_uses_deductive_reasoning_l3434_343418

/-- Represents a statement in the chain of reasoning --/
inductive Statement
| NamesNotCorrect
| LanguageNotCorrect
| ThingsNotDoneSuccessfully
| RitualsAndMusicNotFlourish
| PunishmentsNotProper
| PeopleConfused

/-- Represents the chain of reasoning in the passage --/
def reasoning_chain : List (Statement × Statement) :=
  [(Statement.NamesNotCorrect, Statement.LanguageNotCorrect),
   (Statement.LanguageNotCorrect, Statement.ThingsNotDoneSuccessfully),
   (Statement.ThingsNotDoneSuccessfully, Statement.RitualsAndMusicNotFlourish),
   (Statement.RitualsAndMusicNotFlourish, Statement.PunishmentsNotProper),
   (Statement.PunishmentsNotProper, Statement.PeopleConfused)]

/-- Definition of deductive reasoning --/
def is_deductive_reasoning (chain : List (Statement × Statement)) : Prop :=
  ∀ (premise conclusion : Statement), 
    (premise, conclusion) ∈ chain → 
    (∃ (general_premise : Statement), 
      (general_premise, premise) ∈ chain ∧ (general_premise, conclusion) ∈ chain)

/-- Theorem stating that the reasoning in the passage is deductive --/
theorem passage_uses_deductive_reasoning : 
  is_deductive_reasoning reasoning_chain :=
sorry

end NUMINAMATH_CALUDE_passage_uses_deductive_reasoning_l3434_343418


namespace NUMINAMATH_CALUDE_nine_multiple_plus_k_equals_ones_l3434_343457

/-- Given a natural number N and a positive integer k, there exists a number M
    consisting of k ones such that N · 9 + k = M. -/
theorem nine_multiple_plus_k_equals_ones (N : ℕ) (k : ℕ+) :
  ∃ M : ℕ, (∀ d : ℕ, d < k → (M / 10^d) % 10 = 1) ∧ N * 9 + k = M :=
sorry

end NUMINAMATH_CALUDE_nine_multiple_plus_k_equals_ones_l3434_343457


namespace NUMINAMATH_CALUDE_average_payment_is_460_l3434_343480

/-- The total number of installments -/
def total_installments : ℕ := 52

/-- The number of initial payments -/
def initial_payments : ℕ := 12

/-- The amount of each initial payment -/
def initial_payment_amount : ℚ := 410

/-- The additional amount for each remaining payment -/
def additional_amount : ℚ := 65

/-- The amount of each remaining payment -/
def remaining_payment_amount : ℚ := initial_payment_amount + additional_amount

/-- The number of remaining payments -/
def remaining_payments : ℕ := total_installments - initial_payments

theorem average_payment_is_460 :
  (initial_payments * initial_payment_amount + remaining_payments * remaining_payment_amount) / total_installments = 460 := by
  sorry

end NUMINAMATH_CALUDE_average_payment_is_460_l3434_343480


namespace NUMINAMATH_CALUDE_equation_to_lines_l3434_343478

/-- The set of points satisfying 2x^2 + y^2 + 3xy + 3x + y = 2 is equivalent to the set of points on the lines y = -x - 2 and y = -2x + 1 -/
theorem equation_to_lines :
  ∀ x y : ℝ, 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_to_lines_l3434_343478


namespace NUMINAMATH_CALUDE_transformation_result_l3434_343453

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a point 90° about the y-axis -/
def rotateY90 (p : Point3D) : Point3D :=
  ⟨p.z, p.y, -p.x⟩

/-- Reflects a point through the yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

/-- Reflects a point through the xz-plane -/
def reflectXZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

/-- Reflects a point through the xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

/-- Applies the sequence of transformations to a point -/
def applyTransformations (p : Point3D) : Point3D :=
  p |> rotateY90
    |> reflectYZ
    |> reflectXZ
    |> rotateY90
    |> reflectXZ
    |> reflectXY

theorem transformation_result :
  let initialPoint : Point3D := ⟨2, 2, 2⟩
  applyTransformations initialPoint = ⟨-2, 2, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l3434_343453


namespace NUMINAMATH_CALUDE_max_value_inequality_l3434_343415

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≤ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3434_343415


namespace NUMINAMATH_CALUDE_train_length_l3434_343448

theorem train_length (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  train_speed = 45 * (1000 / 3600) ∧
  bridge_length = 220 ∧
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 155 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3434_343448


namespace NUMINAMATH_CALUDE_distance_between_points_l3434_343481

/-- The distance between two points given round trip time and speed -/
theorem distance_between_points (speed : ℝ) (time : ℝ) (h1 : speed > 0) (h2 : time > 0) :
  let total_distance := speed * time
  let distance_between := total_distance / 2
  distance_between = 120 :=
by
  sorry

#check distance_between_points 60 4

end NUMINAMATH_CALUDE_distance_between_points_l3434_343481


namespace NUMINAMATH_CALUDE_parabola_focal_line_theorem_l3434_343441

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point2D | p.y^2 = 4 * p.x}

/-- The focal length of the parabola y^2 = 4x -/
def focal_length : ℝ := 2

/-- A line passing through the focus of the parabola -/
structure FocalLine where
  intersects_parabola : Point2D → Point2D → Prop

/-- The length of a line segment between two points -/
def line_segment_length (A B : Point2D) : ℝ := sorry

theorem parabola_focal_line_theorem (l : FocalLine) (A B : Point2D) 
  (h1 : A ∈ Parabola) (h2 : B ∈ Parabola) 
  (h3 : l.intersects_parabola A B) 
  (h4 : (A.x + B.x) / 2 = 3) :
  line_segment_length A B = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_focal_line_theorem_l3434_343441
