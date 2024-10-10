import Mathlib

namespace frank_remaining_money_l1348_134897

/-- Calculates the remaining money after buying the most expensive lamp -/
def remaining_money (cheapest_lamp_cost most_expensive_factor current_money : ℕ) : ℕ :=
  current_money - (cheapest_lamp_cost * most_expensive_factor)

/-- Proves that Frank will have $30 remaining after buying the most expensive lamp -/
theorem frank_remaining_money :
  remaining_money 20 3 90 = 30 := by sorry

end frank_remaining_money_l1348_134897


namespace f_value_at_3_l1348_134860

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = 4) : f a b 3 = -12 := by
  sorry

end f_value_at_3_l1348_134860


namespace max_pages_for_budget_l1348_134843

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the budget in dollars
def budget : ℕ := 25

-- Define the function to calculate the maximum number of pages
def max_pages (cost : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost

-- Theorem statement
theorem max_pages_for_budget :
  max_pages cost_per_page budget = 833 := by
  sorry

end max_pages_for_budget_l1348_134843


namespace f_derivative_at_negative_one_l1348_134813

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- State the theorem
theorem f_derivative_at_negative_one (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by sorry

end f_derivative_at_negative_one_l1348_134813


namespace power_equation_solution_l1348_134854

theorem power_equation_solution (a : ℝ) (k : ℝ) (h1 : a ≠ 0) : 
  (a^10 / (a^k)^4 = a^2) → k = 2 := by
sorry

end power_equation_solution_l1348_134854


namespace perfect_square_expression_l1348_134842

theorem perfect_square_expression (x : ℤ) :
  ∃ d : ℤ, (4 * x + 1 - Real.sqrt (8 * x + 1 : ℝ)) / 2 = d →
  ∃ k : ℤ, d = k^2 := by
sorry

end perfect_square_expression_l1348_134842


namespace cube_surface_area_l1348_134873

/-- Given three vertices of a cube, prove that its surface area is 150 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (5, 9, 6) → B = (5, 14, 6) → C = (5, 14, 11) → 
  (let surface_area := 6 * (B.2 - A.2)^2
   surface_area = 150) :=
by sorry

end cube_surface_area_l1348_134873


namespace positive_addition_positive_multiplication_positive_division_positive_exponentiation_positive_root_extraction_l1348_134859

-- Define positive real numbers
def PositiveReal := {x : ℝ | x > 0}

-- Theorem for addition
theorem positive_addition (a b : PositiveReal) : (↑a + ↑b : ℝ) > 0 := by sorry

-- Theorem for multiplication
theorem positive_multiplication (a b : PositiveReal) : (↑a * ↑b : ℝ) > 0 := by sorry

-- Theorem for division
theorem positive_division (a b : PositiveReal) : (↑a / ↑b : ℝ) > 0 := by sorry

-- Theorem for exponentiation
theorem positive_exponentiation (a : PositiveReal) (n : ℝ) : (↑a ^ n : ℝ) > 0 := by sorry

-- Theorem for root extraction
theorem positive_root_extraction (a : PositiveReal) (n : PositiveReal) : 
  ∃ (x : ℝ), x > 0 ∧ x ^ (↑n : ℝ) = ↑a := by sorry

end positive_addition_positive_multiplication_positive_division_positive_exponentiation_positive_root_extraction_l1348_134859


namespace ladder_wood_length_50ft_l1348_134826

/-- Calculates the total length of wood needed for ladder rungs -/
def ladder_wood_length (rung_length inches_between_rungs total_height_feet : ℚ) : ℚ :=
  let inches_per_foot : ℚ := 12
  let total_height_inches : ℚ := total_height_feet * inches_per_foot
  let space_per_rung : ℚ := rung_length + inches_between_rungs
  let num_rungs : ℚ := total_height_inches / space_per_rung
  (num_rungs * rung_length) / inches_per_foot

/-- The total length of wood needed for rungs to climb 50 feet is 37.5 feet -/
theorem ladder_wood_length_50ft :
  ladder_wood_length 18 6 50 = 37.5 := by
  sorry

end ladder_wood_length_50ft_l1348_134826


namespace range_of_c_minus_b_l1348_134872

/-- Represents a triangle with side lengths a, b, c and opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The range of c - b in a triangle where a = 1 and C - B = π/2 -/
theorem range_of_c_minus_b (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.C - t.B = π/2) : 
  ∃ (l u : ℝ), l = Real.sqrt 2 / 2 ∧ u = 1 ∧ 
  ∀ x, (t.c - t.b = x) → l < x ∧ x < u :=
sorry

end range_of_c_minus_b_l1348_134872


namespace water_left_is_84_ounces_l1348_134885

/-- Represents the water cooler problem --/
def water_cooler_problem (initial_gallons : ℕ) (ounces_per_cup : ℕ) (rows : ℕ) (chairs_per_row : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let initial_ounces := initial_gallons * ounces_per_gallon
  let total_cups := rows * chairs_per_row
  let ounces_poured := total_cups * ounces_per_cup
  initial_ounces - ounces_poured

/-- Theorem stating that the water left in the cooler is 84 ounces --/
theorem water_left_is_84_ounces :
  water_cooler_problem 3 6 5 10 128 = 84 := by
  sorry

end water_left_is_84_ounces_l1348_134885


namespace coefficient_of_x_power_5_l1348_134840

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient function for the expansion of (x - 1/√x)^8
def coefficient (r : ℕ) : ℤ :=
  (-1)^r * (binomial 8 r)

-- Theorem statement
theorem coefficient_of_x_power_5 : coefficient 2 = 28 := by sorry

end coefficient_of_x_power_5_l1348_134840


namespace sum_of_edges_l1348_134834

/-- A rectangular solid with given properties -/
structure RectangularSolid where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  volume_eq : a * b * c = 8
  surface_area_eq : 2 * (a * b + b * c + c * a) = 32
  width_sq_eq : b ^ 2 = a * c

/-- The sum of all edges of the rectangular solid is 32 -/
theorem sum_of_edges (solid : RectangularSolid) :
  4 * (solid.a + solid.b + solid.c) = 32 := by
  sorry

end sum_of_edges_l1348_134834


namespace annies_plants_leaves_l1348_134850

/-- Calculates the total number of leaves for Annie's plants -/
def total_leaves (basil_pots rosemary_pots thyme_pots : ℕ) 
                 (basil_leaves rosemary_leaves thyme_leaves : ℕ) : ℕ :=
  basil_pots * basil_leaves + rosemary_pots * rosemary_leaves + thyme_pots * thyme_leaves

/-- Proves that Annie's plants have a total of 354 leaves -/
theorem annies_plants_leaves : 
  total_leaves 3 9 6 4 18 30 = 354 := by
  sorry

#eval total_leaves 3 9 6 4 18 30

end annies_plants_leaves_l1348_134850


namespace loops_per_day_l1348_134861

def weekly_goal : ℕ := 3500
def track_length : ℕ := 50
def days_in_week : ℕ := 7

theorem loops_per_day : 
  ∀ (goal : ℕ) (track : ℕ) (days : ℕ),
  goal = weekly_goal → 
  track = track_length → 
  days = days_in_week →
  (goal / track) / days = 10 := by sorry

end loops_per_day_l1348_134861


namespace cos_240_degrees_l1348_134837

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l1348_134837


namespace test_series_count_l1348_134867

/-- The number of tests in Professor Tester's series -/
def n : ℕ := 8

/-- John's average score if he scored 97 on the last test -/
def avg_with_97 : ℚ := 90

/-- John's average score if he scored 73 on the last test -/
def avg_with_73 : ℚ := 87

/-- The score difference between the two scenarios -/
def score_diff : ℚ := 97 - 73

/-- The average difference between the two scenarios -/
def avg_diff : ℚ := avg_with_97 - avg_with_73

theorem test_series_count :
  score_diff / (n + 1 : ℚ) = avg_diff :=
sorry

end test_series_count_l1348_134867


namespace sum_of_fractions_bounds_l1348_134890

theorem sum_of_fractions_bounds (v w x y z : ℝ) (hv : v > 0) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < (v / (v + w)) + (w / (w + x)) + (x / (x + y)) + (y / (y + z)) + (z / (z + v)) ∧
  (v / (v + w)) + (w / (w + x)) + (x / (x + y)) + (y / (y + z)) + (z / (z + v)) < 4 :=
by sorry

end sum_of_fractions_bounds_l1348_134890


namespace late_arrivals_count_l1348_134841

/-- Represents the number of people per lollipop -/
def people_per_lollipop : ℕ := 5

/-- Represents the initial number of people -/
def initial_people : ℕ := 45

/-- Represents the total number of lollipops given away -/
def total_lollipops : ℕ := 12

/-- Calculates the number of people who came in later -/
def late_arrivals : ℕ := total_lollipops * people_per_lollipop - initial_people

theorem late_arrivals_count : late_arrivals = 15 := by
  sorry

end late_arrivals_count_l1348_134841


namespace four_weeks_filming_time_l1348_134822

/-- Calculates the total filming time in hours for a given number of weeks -/
def total_filming_time (episode_length : ℕ) (filming_factor : ℚ) (episodes_per_week : ℕ) (weeks : ℕ) : ℚ :=
  let filming_time := episode_length * (1 + filming_factor)
  let total_episodes := episodes_per_week * weeks
  (filming_time * total_episodes) / 60

theorem four_weeks_filming_time :
  total_filming_time 20 (1/2) 5 4 = 10 := by
  sorry

#eval total_filming_time 20 (1/2) 5 4

end four_weeks_filming_time_l1348_134822


namespace weekend_earnings_l1348_134852

def newspaper_earnings : ℕ := 16
def car_washing_earnings : ℕ := 74

theorem weekend_earnings :
  newspaper_earnings + car_washing_earnings = 90 := by sorry

end weekend_earnings_l1348_134852


namespace total_ants_employed_l1348_134899

/-- The total number of ants employed for all construction tasks -/
def total_ants (carrying_red carrying_black digging_red digging_black assembling_red assembling_black : ℕ) : ℕ :=
  carrying_red + carrying_black + digging_red + digging_black + assembling_red + assembling_black

/-- Theorem stating that the total number of ants employed is 2464 -/
theorem total_ants_employed :
  total_ants 413 487 356 518 298 392 = 2464 := by
  sorry

#eval total_ants 413 487 356 518 298 392

end total_ants_employed_l1348_134899


namespace right_triangle_perimeter_l1348_134803

theorem right_triangle_perimeter (base height : ℝ) (h_base : base = 4) (h_height : height = 3) :
  let hypotenuse := Real.sqrt (base^2 + height^2)
  base + height + hypotenuse = 12 := by
sorry

end right_triangle_perimeter_l1348_134803


namespace marcus_car_mpg_l1348_134817

/-- Represents a car with its mileage and fuel efficiency characteristics -/
structure Car where
  initial_mileage : ℕ
  final_mileage : ℕ
  tank_capacity : ℕ
  num_fills : ℕ

/-- Calculates the miles per gallon for a given car -/
def miles_per_gallon (c : Car) : ℚ :=
  (c.final_mileage - c.initial_mileage : ℚ) / (c.tank_capacity * c.num_fills : ℚ)

/-- Theorem stating that Marcus's car gets 30 miles per gallon -/
theorem marcus_car_mpg :
  let marcus_car : Car := {
    initial_mileage := 1728,
    final_mileage := 2928,
    tank_capacity := 20,
    num_fills := 2
  }
  miles_per_gallon marcus_car = 30 := by
  sorry

end marcus_car_mpg_l1348_134817


namespace triangle_area_from_perimeter_and_inradius_l1348_134895

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) 
  (inradius : ℝ) 
  (h_perimeter : perimeter = 40) 
  (h_inradius : inradius = 2.5) : 
  inradius * (perimeter / 2) = 50 := by
  sorry

end triangle_area_from_perimeter_and_inradius_l1348_134895


namespace circle_area_from_sector_l1348_134866

theorem circle_area_from_sector (r : ℝ) (P : ℝ) (Q : ℝ) : 
  P = 2 → -- The area of sector COD is 2
  P = (1/6) * π * r^2 → -- Area of sector COD is 1/6 of circle area
  Q = π * r^2 → -- Q is the area of the entire circle
  Q = 12 := by
  sorry

end circle_area_from_sector_l1348_134866


namespace number_for_B_l1348_134880

/-- Given that the number for A is a, and the number for B is 1 less than twice the number for A,
    prove that the number for B can be expressed as 2a - 1. -/
theorem number_for_B (a : ℝ) : 2 * a - 1 = 2 * a - 1 := by sorry

end number_for_B_l1348_134880


namespace exists_point_with_no_interior_lattice_points_l1348_134853

-- Define a point with integer coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a function to check if a point is on a line
def onLine (p : IntPoint) (a b c : Int) : Prop :=
  a * p.x + b * p.y = c

-- Define a function to check if a point is in the interior of a segment
def inInterior (p q r : IntPoint) : Prop :=
  ∃ t : Rat, 0 < t ∧ t < 1 ∧
  p.x = q.x + t * (r.x - q.x) ∧
  p.y = q.y + t * (r.y - q.y)

-- Main theorem
theorem exists_point_with_no_interior_lattice_points
  (A B C : IntPoint) (hABC : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  ∃ P : IntPoint,
    P ≠ A ∧ P ≠ B ∧ P ≠ C ∧
    (∀ Q : IntPoint, ¬(inInterior Q P A ∨ inInterior Q P B ∨ inInterior Q P C)) :=
  sorry

end exists_point_with_no_interior_lattice_points_l1348_134853


namespace exponential_inequality_l1348_134824

theorem exponential_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (1 / 3 : ℝ) ^ x < (1 / 3 : ℝ) ^ y := by
  sorry

end exponential_inequality_l1348_134824


namespace daily_expense_reduction_l1348_134898

theorem daily_expense_reduction (total_expense : ℕ) (original_days : ℕ) (extended_days : ℕ) :
  total_expense = 360 →
  original_days = 20 →
  extended_days = 24 →
  (total_expense / original_days) - (total_expense / extended_days) = 3 := by
  sorry

end daily_expense_reduction_l1348_134898


namespace multiply_preserves_inequality_l1348_134835

theorem multiply_preserves_inequality (a b c : ℝ) : a > b → c > 0 → a * c > b * c := by
  sorry

end multiply_preserves_inequality_l1348_134835


namespace andrew_payment_l1348_134855

/-- Calculate the total amount Andrew paid to the shopkeeper for grapes and mangoes. -/
theorem andrew_payment (grape_quantity : ℕ) (grape_price : ℕ) (mango_quantity : ℕ) (mango_price : ℕ) :
  grape_quantity = 11 →
  grape_price = 98 →
  mango_quantity = 7 →
  mango_price = 50 →
  grape_quantity * grape_price + mango_quantity * mango_price = 1428 :=
by
  sorry

#check andrew_payment

end andrew_payment_l1348_134855


namespace delta_y_value_l1348_134864

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 + 1

-- State the theorem
theorem delta_y_value (x Δx : ℝ) (hx : x = 1) (hΔx : Δx = 0.1) :
  f (x + Δx) - f x = 0.63 := by
  sorry

end delta_y_value_l1348_134864


namespace dodecahedron_triangle_probability_l1348_134848

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Nat
  connections_per_vertex : Nat

/-- The probability of forming a triangle with three randomly chosen vertices -/
def triangle_probability (d : RegularDodecahedron) : ℚ :=
  sorry

/-- Theorem stating the probability of forming a triangle in a regular dodecahedron -/
theorem dodecahedron_triangle_probability :
  let d : RegularDodecahedron := ⟨20, 3⟩
  triangle_probability d = 1 / 57 := by
  sorry

end dodecahedron_triangle_probability_l1348_134848


namespace last_two_digits_sum_of_squares_l1348_134883

theorem last_two_digits_sum_of_squares :
  ∀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ),
  a₁ % 100 = 11 →
  a₂ % 100 = 12 →
  a₃ % 100 = 13 →
  a₄ % 100 = 14 →
  a₅ % 100 = 15 →
  a₆ % 100 = 16 →
  a₇ % 100 = 17 →
  a₈ % 100 = 18 →
  a₉ % 100 = 19 →
  (a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 + a₆^2 + a₇^2 + a₈^2 + a₉^2) % 100 = 85 :=
by sorry

end last_two_digits_sum_of_squares_l1348_134883


namespace range_of_a_l1348_134829

-- Define the statements p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x - 1 < 0

def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∨ q a)) → (a ≤ -4 ∨ a ≥ 1) :=
sorry

end range_of_a_l1348_134829


namespace equality_implies_equal_expressions_l1348_134847

theorem equality_implies_equal_expressions (a b : ℝ) : a = b → 2 * (a - 1) = 2 * (b - 1) := by
  sorry

end equality_implies_equal_expressions_l1348_134847


namespace parabola_properties_l1348_134839

/-- Parabola represented by its parameter p -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and a point on it, prove the standard form of the parabola
    and the ratio of distances from two points to the focus -/
theorem parabola_properties (E : Parabola) (A : Point)
    (h_on_parabola : A.y^2 = 2 * E.p * A.x)
    (h_y_pos : A.y > 0)
    (h_A_coords : A.x = 9 ∧ A.y = 6)
    (h_AF_length : 5 = |A.x - E.p| + |A.y|) : 
  (∀ (x y : ℝ), y^2 = 4*x ↔ y^2 = 2*E.p*x) ∧ 
  ∃ (B : Point), B ≠ A ∧ 
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
      B.x = t * A.x + (1 - t) * E.p ∧
      B.y = t * A.y) ∧
    5 / (|B.x - E.p| + |B.y|) = 4 :=
by sorry

end parabola_properties_l1348_134839


namespace x_over_y_equals_one_l1348_134832

theorem x_over_y_equals_one (x y : ℝ) 
  (h1 : 1 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 3) 
  (h3 : ∃ n : ℤ, x / y = n) : 
  x / y = 1 := by
sorry

end x_over_y_equals_one_l1348_134832


namespace find_number_l1348_134825

theorem find_number : ∃ x : ℝ, ((55 + x) / 7 + 40) * 5 = 555 ∧ x = 442 := by
  sorry

end find_number_l1348_134825


namespace students_playing_neither_l1348_134821

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 36)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 7 := by
  sorry

end students_playing_neither_l1348_134821


namespace parallel_transitive_l1348_134893

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end parallel_transitive_l1348_134893


namespace xyz_value_l1348_134823

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 8 := by
  sorry

end xyz_value_l1348_134823


namespace quadratic_equation_proof_l1348_134833

theorem quadratic_equation_proof (c : ℝ) : 
  (∃ c_modified : ℝ, c = c_modified + 2 ∧ (-1)^2 + 4*(-1) + c_modified = 0) →
  c = 5 ∧ ∀ x : ℝ, x^2 + 4*x + c ≠ 0 := by
sorry

end quadratic_equation_proof_l1348_134833


namespace calculate_value_probability_l1348_134809

def calculate_letters : Finset Char := {'C', 'A', 'L', 'C', 'U', 'L', 'A', 'T', 'E'}
def value_letters : Finset Char := {'V', 'A', 'L', 'U', 'E'}

theorem calculate_value_probability :
  (calculate_letters.filter (λ c => c ∈ value_letters)).card / calculate_letters.card = 2 / 3 := by
sorry

end calculate_value_probability_l1348_134809


namespace nail_salon_revenue_l1348_134868

/-- Calculates the total money made from manicures in a nail salon --/
def total_manicure_money (manicure_cost : ℝ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ) : ℝ :=
  let total_people : ℕ := total_fingers / fingers_per_person
  let clients : ℕ := total_people - non_clients
  (clients : ℝ) * manicure_cost

/-- Theorem stating the total money made from manicures in the given scenario --/
theorem nail_salon_revenue :
  total_manicure_money 20 210 10 11 = 200 := by
  sorry

end nail_salon_revenue_l1348_134868


namespace chris_has_12_marbles_l1348_134836

-- Define the number of marbles Chris and Ryan have
def chris_marbles : ℕ := sorry
def ryan_marbles : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := chris_marbles + ryan_marbles

-- Define the number of marbles remaining after they take their share
def remaining_marbles : ℕ := 20

-- Theorem stating that Chris has 12 marbles
theorem chris_has_12_marbles :
  chris_marbles = 12 :=
by
  sorry


end chris_has_12_marbles_l1348_134836


namespace constant_term_is_165_l1348_134810

-- Define the derivative function
def derivative (q : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define the equation q' = 3q + c
def equation (c : ℝ) (q : ℝ → ℝ) : Prop :=
  ∀ x, derivative q x = 3 * q x + c

-- State the theorem
theorem constant_term_is_165 :
  ∃ (q : ℝ → ℝ) (c : ℝ),
    equation c q ∧
    derivative (derivative q) 6 = 210 ∧
    c = 165 :=
sorry

end constant_term_is_165_l1348_134810


namespace function_from_derivative_and_point_l1348_134881

/-- Given a function f: ℝ → ℝ, if its derivative is 4x³ for all x
and f(1) = -1, then f(x) = x⁴ - 2 for all x. -/
theorem function_from_derivative_and_point (f : ℝ → ℝ) 
    (h1 : ∀ x, deriv f x = 4 * x^3)
    (h2 : f 1 = -1) :
    ∀ x, f x = x^4 - 2 := by
  sorry

end function_from_derivative_and_point_l1348_134881


namespace sum_of_squares_of_roots_l1348_134857

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 + 2 * a^2 - 5 * a - 15 = 0) ∧
  (3 * b^3 + 2 * b^2 - 5 * b - 15 = 0) ∧
  (3 * c^3 + 2 * c^2 - 5 * c - 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by
sorry

end sum_of_squares_of_roots_l1348_134857


namespace intersection_equals_A_l1348_134831

def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

theorem intersection_equals_A : A ∩ B = A := by sorry

end intersection_equals_A_l1348_134831


namespace isosceles_triangle_perimeter_l1348_134876

/-- An isosceles triangle with side lengths 9 and 5 has a perimeter of either 19 or 23 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 9 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 9 ∧ c = 9) →  -- isosceles with sides 9 and 5
  a + b + c = 19 ∨ a + b + c = 23 := by
sorry

end isosceles_triangle_perimeter_l1348_134876


namespace proposition_relationship_l1348_134894

theorem proposition_relationship :
  (∀ a : ℝ, 0 < a ∧ a < 1 → ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧ ¬(0 < a ∧ a < 1)) := by
  sorry

end proposition_relationship_l1348_134894


namespace negative_root_iff_negative_a_l1348_134838

theorem negative_root_iff_negative_a (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 1 = 0) ↔ a < 0 := by
  sorry

end negative_root_iff_negative_a_l1348_134838


namespace mitchell_chews_145_pieces_l1348_134828

/-- The number of pieces of gum Mitchell chews -/
def chewed_pieces (packets : ℕ) (pieces_per_packet : ℕ) (unchewed : ℕ) : ℕ :=
  packets * pieces_per_packet - unchewed

/-- Proof that Mitchell chews 145 pieces of gum -/
theorem mitchell_chews_145_pieces :
  chewed_pieces 15 10 5 = 145 := by
  sorry

end mitchell_chews_145_pieces_l1348_134828


namespace roots_of_cubic_polynomial_l1348_134884

theorem roots_of_cubic_polynomial :
  let p : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) := by
  sorry

end roots_of_cubic_polynomial_l1348_134884


namespace distinguishable_cube_colorings_count_l1348_134871

/-- The number of distinguishable ways to color a cube with six different colors -/
def distinguishable_cube_colorings : ℕ := 30

/-- A cube has six faces -/
def cube_faces : ℕ := 6

/-- The number of rotational symmetries of a cube -/
def cube_rotational_symmetries : ℕ := 24

/-- The total number of ways to arrange 6 colors on 6 faces -/
def total_arrangements : ℕ := 720  -- 6!

theorem distinguishable_cube_colorings_count :
  distinguishable_cube_colorings = total_arrangements / cube_rotational_symmetries :=
by sorry

end distinguishable_cube_colorings_count_l1348_134871


namespace quadratic_distinct_roots_l1348_134875

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < 1/4 :=
by sorry

end quadratic_distinct_roots_l1348_134875


namespace sum_of_squares_l1348_134812

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 21 → 
  a * b + b * c + a * c = 100 → 
  a^2 + b^2 + c^2 = 241 := by
  sorry

end sum_of_squares_l1348_134812


namespace zero_decomposition_l1348_134804

/-- Represents a base-10 arithmetic system -/
structure Base10Arithmetic where
  /-- Multiplication operation in base-10 arithmetic -/
  mul : ℤ → ℤ → ℤ
  /-- Axiom: Multiplication by zero always results in zero -/
  mul_zero : ∀ a : ℤ, mul 0 a = 0

/-- 
Theorem: In base-10 arithmetic, the only way to decompose 0 into a product 
of two integers is 0 * a = 0, where a is any integer.
-/
theorem zero_decomposition (B : Base10Arithmetic) : 
  ∀ x y : ℤ, B.mul x y = 0 → x = 0 ∨ y = 0 := by
  sorry

end zero_decomposition_l1348_134804


namespace chef_cherries_remaining_l1348_134862

theorem chef_cherries_remaining (initial_cherries used_cherries : ℕ) 
  (h1 : initial_cherries = 77)
  (h2 : used_cherries = 60) :
  initial_cherries - used_cherries = 17 := by
  sorry

end chef_cherries_remaining_l1348_134862


namespace sum_interior_angles_hexagon_l1348_134845

/-- The sum of interior angles of a hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  ∀ (n : ℕ) (sum_interior_angles : ℕ → ℝ),
  n = 6 →
  (∀ k : ℕ, sum_interior_angles k = (k - 2) * 180) →
  sum_interior_angles n = 720 := by
sorry

end sum_interior_angles_hexagon_l1348_134845


namespace binomial_expansion_theorem_l1348_134886

theorem binomial_expansion_theorem (n : ℕ) (a b k : ℝ) : 
  n ≥ 2 → 
  a * b ≠ 0 → 
  a = b + k → 
  k > 0 → 
  (2 : ℝ) * (n.choose 1) * (2 * b) ^ (n - 1) * k + 
  (8 : ℝ) * (n.choose 3) * (2 * b) ^ (n - 3) * k ^ 3 = 0 → 
  n = 3 := by
sorry

end binomial_expansion_theorem_l1348_134886


namespace bombardment_death_percentage_l1348_134870

/-- The percentage of people who died by bombardment in a Sri Lankan village --/
def bombardment_percentage (initial_population final_population : ℕ) (departure_rate : ℚ) : ℚ :=
  let x := (initial_population - final_population / (1 - departure_rate)) / initial_population
  x * 100

/-- Theorem stating the percentage of people who died by bombardment --/
theorem bombardment_death_percentage :
  let initial_population : ℕ := 4399
  let final_population : ℕ := 3168
  let departure_rate : ℚ := 1/5
  abs (bombardment_percentage initial_population final_population departure_rate - 9.98) < 0.01 := by
  sorry

end bombardment_death_percentage_l1348_134870


namespace sum_of_periodic_functions_periodicity_l1348_134891

/-- A periodic function with period T -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- A function with smallest positive period T -/
def HasSmallestPeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  IsPeriodic f T ∧ ∀ S, 0 < S → S < T → ¬IsPeriodic f S

/-- The main theorem about the sum of two periodic functions -/
theorem sum_of_periodic_functions_periodicity
  (f₁ f₂ : ℝ → ℝ) (T : ℝ) (hT : T > 0)
  (h₁ : HasSmallestPeriod f₁ T) (h₂ : HasSmallestPeriod f₂ T) :
  ∃ y : ℝ → ℝ, (y = f₁ + f₂) ∧ 
  IsPeriodic y T ∧ 
  ¬(∃ S : ℝ, HasSmallestPeriod y S) :=
sorry

end sum_of_periodic_functions_periodicity_l1348_134891


namespace problem_1_l1348_134815

theorem problem_1 : |(-6)| - 7 + (-3) = -4 := by sorry

end problem_1_l1348_134815


namespace min_both_beethoven_chopin_l1348_134889

theorem min_both_beethoven_chopin 
  (total : ℕ) 
  (beethoven_fans : ℕ) 
  (chopin_fans : ℕ) 
  (h1 : total = 150) 
  (h2 : beethoven_fans = 120) 
  (h3 : chopin_fans = 95) :
  (beethoven_fans + chopin_fans - total : ℤ).natAbs ≥ 65 :=
by sorry

end min_both_beethoven_chopin_l1348_134889


namespace cubic_roots_sum_l1348_134863

theorem cubic_roots_sum (a b c : ℝ) : 
  (3 * a^3 - 6 * a^2 + 99 * a - 2 = 0) →
  (3 * b^3 - 6 * b^2 + 99 * b - 2 = 0) →
  (3 * c^3 - 6 * c^2 + 99 * c - 2 = 0) →
  (a + b - 2)^3 + (b + c - 2)^3 + (c + a - 2)^3 = -196 := by
sorry

end cubic_roots_sum_l1348_134863


namespace radical_simplification_l1348_134808

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (21 * q) = 21 * q * Real.sqrt (21 * q) :=
by sorry

end radical_simplification_l1348_134808


namespace quadratic_inequality_solution_range_l1348_134888

theorem quadratic_inequality_solution_range (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end quadratic_inequality_solution_range_l1348_134888


namespace possible_x_values_l1348_134856

def M (x : ℝ) : Set ℝ := {-2, 3*x^2 + 3*x - 4, x^2 + x - 4}

theorem possible_x_values (x : ℝ) : 2 ∈ M x → x = 2 ∨ x = -3 := by
  sorry

end possible_x_values_l1348_134856


namespace gcd_lcm_sum_l1348_134858

theorem gcd_lcm_sum : Nat.gcd 42 63 + Nat.lcm 48 18 = 165 := by
  sorry

end gcd_lcm_sum_l1348_134858


namespace impossibility_of_all_powers_of_two_l1348_134877

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card
  oddCount : ℕ

/-- The procedure of creating a new card from existing cards -/
def createNewCard (state : TableState) : Card :=
  sorry

/-- The evolution of the table state over time -/
def evolveTable (initialState : TableState) : ℕ → TableState
  | 0 => initialState
  | n + 1 => let prevState := evolveTable initialState n
              let newCard := createNewCard prevState
              { cards := newCard :: prevState.cards,
                oddCount := if newCard.value % 2 = 1 then prevState.oddCount + 1 else prevState.oddCount }

/-- Checks if a number is divisible by 2^d -/
def isDivisibleByPowerOfTwo (n d : ℕ) : Bool :=
  n % (2^d) = 0

theorem impossibility_of_all_powers_of_two :
  ∀ (initialCards : List Card),
    initialCards.length = 100 →
    (initialCards.filter (λ c => c.value % 2 = 1)).length = 28 →
    ∃ (d : ℕ), ∀ (t : ℕ),
      ¬∃ (card : Card),
        card ∈ (evolveTable { cards := initialCards, oddCount := 28 } t).cards ∧
        isDivisibleByPowerOfTwo card.value d :=
  sorry

end impossibility_of_all_powers_of_two_l1348_134877


namespace cubic_function_extrema_condition_l1348_134818

/-- Given a cubic function f(x) = x³ - 3x² + ax - b that has both a maximum and a minimum value,
    prove that the parameter a must be less than 3. -/
theorem cubic_function_extrema_condition (a b : ℝ) : 
  (∃ (x_min x_max : ℝ), ∀ x : ℝ, 
    x^3 - 3*x^2 + a*x - b ≤ x_max^3 - 3*x_max^2 + a*x_max - b ∧ 
    x^3 - 3*x^2 + a*x - b ≥ x_min^3 - 3*x_min^2 + a*x_min - b) →
  a < 3 :=
by sorry

end cubic_function_extrema_condition_l1348_134818


namespace chocolate_bars_per_small_box_l1348_134882

theorem chocolate_bars_per_small_box 
  (total_bars : ℕ) 
  (small_boxes : ℕ) 
  (h1 : total_bars = 525) 
  (h2 : small_boxes = 21) : 
  total_bars / small_boxes = 25 := by
  sorry

end chocolate_bars_per_small_box_l1348_134882


namespace cafeteria_line_swaps_l1348_134820

/-- Represents a student in the line -/
inductive Student
| Boy : Student
| Girl : Student

/-- The initial line of students -/
def initial_line : List Student :=
  (List.range 8).bind (fun _ => [Student.Boy, Student.Girl])

/-- The final line of students -/
def final_line : List Student :=
  (List.replicate 8 Student.Girl) ++ (List.replicate 8 Student.Boy)

/-- The number of swaps required -/
def num_swaps : Nat := (List.range 8).sum

theorem cafeteria_line_swaps :
  num_swaps = 36 ∧
  num_swaps = (initial_line.length / 2) * ((initial_line.length / 2) + 1) / 2 :=
sorry

end cafeteria_line_swaps_l1348_134820


namespace jack_walked_4_miles_l1348_134846

/-- The distance Jack walked given his walking time and rate -/
def jack_distance (time_hours : ℝ) (rate : ℝ) : ℝ :=
  time_hours * rate

theorem jack_walked_4_miles :
  let time_hours : ℝ := 1.25  -- 1 hour and 15 minutes in decimal hours
  let rate : ℝ := 3.2         -- miles per hour
  jack_distance time_hours rate = 4 := by
sorry

end jack_walked_4_miles_l1348_134846


namespace inverse_variation_problem_l1348_134892

/-- Given that x² varies inversely with √w, prove that w = 1 when x = 6,
    given that x = 3 when w = 16. -/
theorem inverse_variation_problem (x w : ℝ) (k : ℝ) (h1 : x^2 * Real.sqrt w = k)
    (h2 : 3^2 * Real.sqrt 16 = k) (h3 : x = 6) : w = 1 := by
  sorry

end inverse_variation_problem_l1348_134892


namespace segment_length_product_l1348_134819

theorem segment_length_product (b₁ b₂ : ℝ) : 
  (((3 * b₁ - 5)^2 + (b₁ + 3)^2 = 45) ∧ 
   ((3 * b₂ - 5)^2 + (b₂ + 3)^2 = 45) ∧ 
   b₁ ≠ b₂) → 
  b₁ * b₂ = -11/10 := by
sorry

end segment_length_product_l1348_134819


namespace reciprocal_of_repeating_decimal_l1348_134879

-- Define the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Theorem statement
theorem reciprocal_of_repeating_decimal :
  (repeating_decimal⁻¹ : ℚ) = 99 / 34 := by sorry

end reciprocal_of_repeating_decimal_l1348_134879


namespace plane_trip_distance_l1348_134865

/-- Proves that if a person takes a trip a certain number of times and travels a total distance,
    then the distance of each trip is the total distance divided by the number of trips. -/
theorem plane_trip_distance (num_trips : ℝ) (total_distance : ℝ) 
    (h1 : num_trips = 32) 
    (h2 : total_distance = 8192) : 
  total_distance / num_trips = 256 := by
  sorry

#check plane_trip_distance

end plane_trip_distance_l1348_134865


namespace intersection_M_N_l1348_134805

def M : Set ℝ := {x | 1 - 2/x < 0}
def N : Set ℝ := {x | -1 ≤ x}

theorem intersection_M_N : ∀ x : ℝ, x ∈ M ∩ N ↔ 0 < x ∧ x < 2 := by sorry

end intersection_M_N_l1348_134805


namespace addition_subtraction_reduces_system_l1348_134807

/-- A method for solving systems of linear equations with two variables -/
inductive SolvingMethod
| Substitution
| AdditionSubtraction

/-- Represents a system of linear equations with two variables -/
structure LinearSystem :=
  (equations : List (LinearEquation))

/-- Represents a linear equation -/
structure LinearEquation :=
  (coefficients : List ℝ)
  (constant : ℝ)

/-- A function that determines if a method reduces a system to a single variable -/
def reduces_to_single_variable (method : SolvingMethod) (system : LinearSystem) : Prop :=
  sorry

/-- The theorem stating that the addition-subtraction method reduces a system to a single variable -/
theorem addition_subtraction_reduces_system :
  ∀ (system : LinearSystem),
    reduces_to_single_variable SolvingMethod.AdditionSubtraction system :=
  sorry

end addition_subtraction_reduces_system_l1348_134807


namespace selling_multiple_satisfies_profit_equation_l1348_134896

/-- The multiple of the value of components that John sells computers for -/
def selling_multiple : ℝ := 1.4

/-- Cost of parts for one computer -/
def parts_cost : ℝ := 800

/-- Number of computers built per month -/
def computers_per_month : ℕ := 60

/-- Monthly rent -/
def monthly_rent : ℝ := 5000

/-- Monthly non-rent extra expenses -/
def extra_expenses : ℝ := 3000

/-- Monthly profit -/
def monthly_profit : ℝ := 11200

/-- Theorem stating that the selling multiple satisfies the profit equation -/
theorem selling_multiple_satisfies_profit_equation :
  computers_per_month * parts_cost * selling_multiple -
  (computers_per_month * parts_cost + monthly_rent + extra_expenses) = monthly_profit := by
  sorry

end selling_multiple_satisfies_profit_equation_l1348_134896


namespace inverse_proportion_k_value_l1348_134887

theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x, x ≠ 0 → (k / x) = -1 ↔ x = 2) → k = -2 := by
  sorry

end inverse_proportion_k_value_l1348_134887


namespace inequality_range_l1348_134830

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 := by
sorry

end inequality_range_l1348_134830


namespace least_cans_required_l1348_134800

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_cans_required :
  let gcd := Nat.gcd (Nat.gcd maaza_volume pepsi_volume) sprite_volume
  maaza_volume / gcd + pepsi_volume / gcd + sprite_volume / gcd = 133 := by
  sorry

end least_cans_required_l1348_134800


namespace rectangular_box_volume_l1348_134878

theorem rectangular_box_volume 
  (face_area1 face_area2 face_area3 : ℝ) 
  (h1 : face_area1 = 18)
  (h2 : face_area2 = 50)
  (h3 : face_area3 = 45) :
  ∃ (l w h : ℝ), 
    l * w = face_area1 ∧ 
    w * h = face_area2 ∧ 
    l * h = face_area3 ∧ 
    l * w * h = 30 * Real.sqrt 5 :=
by sorry

end rectangular_box_volume_l1348_134878


namespace cos_180_degrees_l1348_134851

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l1348_134851


namespace temperature_conversion_l1348_134816

theorem temperature_conversion (C F : ℝ) : 
  C = 4/7 * (F - 40) → C = 25 → F = 83.75 := by
  sorry

end temperature_conversion_l1348_134816


namespace hyperbola_asymptote_angle_cos_double_l1348_134814

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the acute angle between asymptotes
def asymptote_angle (α : ℝ) : Prop := 
  ∃ (x y : ℝ), hyperbola x y ∧ 
  (∀ (x' y' : ℝ), hyperbola x' y' → 
    α = Real.arctan (abs (y / x)) ∧ α > 0 ∧ α < Real.pi / 2)

-- Theorem statement
theorem hyperbola_asymptote_angle_cos_double :
  ∀ α : ℝ, asymptote_angle α → Real.cos (2 * α) = -7/25 :=
by sorry

end hyperbola_asymptote_angle_cos_double_l1348_134814


namespace system_of_equations_solution_l1348_134801

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (3 * x - 4 * y = -7) ∧ 
    (6 * x - 5 * y = 9) ∧ 
    (x = 71 / 9) ∧ 
    (y = 23 / 3) := by
  sorry

end system_of_equations_solution_l1348_134801


namespace interest_difference_l1348_134802

theorem interest_difference (principal rate time : ℝ) : 
  principal = 300 → 
  rate = 4 → 
  time = 8 → 
  principal - (principal * rate * time / 100) = 204 :=
by
  sorry

end interest_difference_l1348_134802


namespace teacher_student_grouping_probability_l1348_134811

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of groups -/
def num_groups : ℕ := 2

/-- The number of teachers per group -/
def teachers_per_group : ℕ := 1

/-- The number of students per group -/
def students_per_group : ℕ := 2

/-- The probability that teacher A and student B are in the same group -/
def prob_same_group : ℚ := 1/2

theorem teacher_student_grouping_probability :
  (num_teachers = 2) →
  (num_students = 4) →
  (num_groups = 2) →
  (teachers_per_group = 1) →
  (students_per_group = 2) →
  prob_same_group = 1/2 := by
  sorry

end teacher_student_grouping_probability_l1348_134811


namespace brenda_mice_fraction_l1348_134849

/-- The fraction of baby mice Brenda gave to Robbie -/
def f : ℚ := sorry

/-- The total number of baby mice -/
def total_mice : ℕ := 3 * 8

theorem brenda_mice_fraction :
  (f * total_mice : ℚ) +                        -- Mice given to Robbie
  (3 * f * total_mice : ℚ) +                    -- Mice sold to pet store
  ((1 - 4 * f) * total_mice / 2 : ℚ) +          -- Mice sold to snake owners
  4 = total_mice ∧                              -- Remaining mice
  f = 1 / 6 := by sorry

end brenda_mice_fraction_l1348_134849


namespace cafeteria_pies_l1348_134874

def number_of_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  ((initial_apples - handed_out) / apples_per_pie : ℕ)

theorem cafeteria_pies :
  number_of_pies 150 24 15 = 8 := by
  sorry

end cafeteria_pies_l1348_134874


namespace perpendicular_length_is_five_l1348_134827

/-- Properties of a right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  is_right : DE = 5 ∧ EF = 12

/-- The length of the perpendicular from the hypotenuse to the midpoint of the angle bisector -/
def perpendicular_length (t : RightTriangle) : ℝ :=
  sorry

/-- Theorem: The perpendicular length is 5 -/
theorem perpendicular_length_is_five (t : RightTriangle) :
  perpendicular_length t = 5 := by
  sorry

end perpendicular_length_is_five_l1348_134827


namespace max_single_player_salary_is_454000_l1348_134806

/-- Represents a basketball team in the semi-professional league --/
structure BasketballTeam where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player on the team --/
def maxSinglePlayerSalary (team : BasketballTeam) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player --/
theorem max_single_player_salary_is_454000 :
  let team := BasketballTeam.mk 23 18000 850000
  maxSinglePlayerSalary team = 454000 := by
  sorry

#eval maxSinglePlayerSalary (BasketballTeam.mk 23 18000 850000)

end max_single_player_salary_is_454000_l1348_134806


namespace consecutive_numbers_digit_sum_exists_l1348_134844

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem consecutive_numbers_digit_sum_exists : ∃ (n : ℕ), 
  sumOfDigits n = 52 ∧ 
  sumOfDigits (n + 4) = 20 ∧ 
  n > 0 :=
sorry

end consecutive_numbers_digit_sum_exists_l1348_134844


namespace expand_product_l1348_134869

theorem expand_product (x : ℝ) : (3*x + 4) * (x - 2) * (x + 6) = 3*x^3 + 16*x^2 - 20*x - 48 := by
  sorry

end expand_product_l1348_134869
