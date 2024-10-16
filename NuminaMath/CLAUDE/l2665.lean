import Mathlib

namespace NUMINAMATH_CALUDE_union_of_sets_l2665_266592

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_of_sets :
  ∃ x : ℝ, (B x ∩ A x = {9}) → (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2665_266592


namespace NUMINAMATH_CALUDE_solve_for_y_l2665_266559

theorem solve_for_y (x y : ℝ) (h1 : x^2 = 2*y - 6) (h2 : x = 3) : y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2665_266559


namespace NUMINAMATH_CALUDE_original_number_l2665_266535

theorem original_number (final_number : ℝ) (increase_percentage : ℝ) 
  (h1 : final_number = 210)
  (h2 : increase_percentage = 0.40) : 
  final_number = (1 + increase_percentage) * 150 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2665_266535


namespace NUMINAMATH_CALUDE_max_markers_is_16_l2665_266518

-- Define the prices and quantities for each option
def single_marker_price : ℕ := 2
def pack4_price : ℕ := 6
def pack8_price : ℕ := 10
def pack4_quantity : ℕ := 4
def pack8_quantity : ℕ := 8

-- Define Lisa's budget
def budget : ℕ := 20

-- Define a function to calculate the number of markers for a given combination of purchases
def markers_bought (singles pack4s pack8s : ℕ) : ℕ :=
  singles + pack4s * pack4_quantity + pack8s * pack8_quantity

-- Define a function to calculate the total cost of a combination of purchases
def total_cost (singles pack4s pack8s : ℕ) : ℕ :=
  singles * single_marker_price + pack4s * pack4_price + pack8s * pack8_price

-- Theorem: The maximum number of markers that can be bought with the given budget is 16
theorem max_markers_is_16 :
  ∀ (singles pack4s pack8s : ℕ),
    total_cost singles pack4s pack8s ≤ budget →
    markers_bought singles pack4s pack8s ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_markers_is_16_l2665_266518


namespace NUMINAMATH_CALUDE_graph_equation_two_lines_l2665_266565

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 is equivalent to the union of the lines x = 0 and y = 0 -/
theorem graph_equation_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_equation_two_lines_l2665_266565


namespace NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_56_l2665_266569

def bernardo_win (N : ℕ) : Prop :=
  N ≤ 999 ∧
  3 * N < 1000 ∧
  3 * N + 100 < 1000 ∧
  9 * N + 300 < 1000 ∧
  9 * N + 400 < 1000 ∧
  27 * N + 1200 ≥ 1000

theorem smallest_winning_number :
  ∀ n : ℕ, n < 56 → ¬(bernardo_win n) ∧ bernardo_win 56 :=
sorry

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_56 :
  sum_of_digits 56 = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_sum_of_digits_56_l2665_266569


namespace NUMINAMATH_CALUDE_gumble_words_count_l2665_266503

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def words_with_b (n : ℕ) : ℕ :=
  alphabet_size^n - (alphabet_size - 1)^n

def total_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4 + words_with_b 5

theorem gumble_words_count :
  total_words = 1863701 :=
by sorry

end NUMINAMATH_CALUDE_gumble_words_count_l2665_266503


namespace NUMINAMATH_CALUDE_blood_donation_selection_count_l2665_266512

def male_teachers : ℕ := 3
def female_teachers : ℕ := 6
def total_teachers : ℕ := male_teachers + female_teachers
def selection_size : ℕ := 5

theorem blood_donation_selection_count :
  (Nat.choose total_teachers selection_size) - (Nat.choose female_teachers selection_size) = 120 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_count_l2665_266512


namespace NUMINAMATH_CALUDE_corn_acreage_l2665_266598

/-- Given a total of 1034 acres of land divided among beans, wheat, and corn
    in the ratio of 5:2:4, prove that the number of acres used for corn is 376. -/
theorem corn_acreage (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
    (h1 : total_land = 1034)
    (h2 : beans_ratio = 5)
    (h3 : wheat_ratio = 2)
    (h4 : corn_ratio = 4) :
    (corn_ratio * total_land) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry


end NUMINAMATH_CALUDE_corn_acreage_l2665_266598


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_nine_l2665_266566

theorem smallest_n_multiple_of_nine (x y a : ℤ) : 
  (∃ k₁ k₂ : ℤ, x - a = 9 * k₁ ∧ y + a = 9 * k₂) →
  (∃ n : ℕ, n > 0 ∧ ∃ k : ℤ, x^2 + x*y + y^2 + n = 9 * k ∧
    ∀ m : ℕ, m > 0 → (∃ l : ℤ, x^2 + x*y + y^2 + m = 9 * l) → m ≥ n) →
  (∃ k : ℤ, x^2 + x*y + y^2 + 6 = 9 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_nine_l2665_266566


namespace NUMINAMATH_CALUDE_fabric_cost_theorem_l2665_266524

/-- Represents the cost in livres, sous, and deniers -/
structure Cost :=
  (livres : ℕ)
  (sous : ℕ)
  (deniers : ℚ)

/-- Converts a Cost to deniers -/
def cost_to_deniers (c : Cost) : ℚ :=
  c.livres * 20 * 12 + c.sous * 12 + c.deniers

/-- Converts deniers to a Cost -/
def deniers_to_cost (d : ℚ) : Cost :=
  let total_sous := d / 12
  let livres := (total_sous / 20).floor
  let remaining_sous := total_sous - livres * 20
  { livres := livres.toNat,
    sous := remaining_sous.floor.toNat,
    deniers := d - (livres * 20 * 12 + remaining_sous.floor * 12) }

def ell_cost : Cost := { livres := 42, sous := 17, deniers := 11 }

def fabric_length : ℚ := 15 + 13 / 16

theorem fabric_cost_theorem :
  deniers_to_cost (cost_to_deniers ell_cost * fabric_length) =
  { livres := 682, sous := 15, deniers := 9 + 11 / 16 } := by
  sorry

end NUMINAMATH_CALUDE_fabric_cost_theorem_l2665_266524


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l2665_266574

theorem complex_number_magnitude (a : ℝ) (z : ℂ) : 
  z = (1 + a * Complex.I) / Complex.I → 
  z.re = z.im →
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l2665_266574


namespace NUMINAMATH_CALUDE_coffee_maker_price_l2665_266584

/-- The final price of a coffee maker after applying a discount -/
def final_price (original_price discount : ℕ) : ℕ :=
  original_price - discount

/-- Theorem: The customer pays $70 for a coffee maker with original price $90 and a $20 discount -/
theorem coffee_maker_price :
  final_price 90 20 = 70 := by
  sorry

end NUMINAMATH_CALUDE_coffee_maker_price_l2665_266584


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l2665_266545

/-- Given a quadratic function y = (x - a)² + a - 1, where a is a constant,
    and (m, n) is a point on the graph with m > 0, prove that if m > 2a, then n > -5/4. -/
theorem quadratic_function_inequality (a m n : ℝ) : 
  m > 0 → 
  n = (m - a)^2 + a - 1 → 
  m > 2*a → 
  n > -5/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l2665_266545


namespace NUMINAMATH_CALUDE_correct_num_children_l2665_266531

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 2

/-- The total number of pencils -/
def total_pencils : ℕ := 16

/-- The number of children -/
def num_children : ℕ := total_pencils / pencils_per_child

theorem correct_num_children : num_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_children_l2665_266531


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2665_266576

/-- A quadratic equation qx^2 - 18x + 8 = 0 has only one solution when q = 81/8 -/
theorem unique_solution_quadratic :
  ∃! (x : ℝ), (81/8 : ℝ) * x^2 - 18 * x + 8 = 0 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2665_266576


namespace NUMINAMATH_CALUDE_officer_assignment_count_l2665_266539

-- Define the set of people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carol : Person
| Dave : Person

-- Define the set of officer positions
inductive Position : Type
| President : Position
| Secretary : Position
| Treasurer : Position

-- Define a function to check if a person is qualified for a position
def isQualified (p : Person) (pos : Position) : Prop :=
  match pos with
  | Position.President => p = Person.Dave
  | _ => True

-- Define an assignment of officers
def OfficerAssignment := Position → Person

-- Define a valid assignment
def validAssignment (assignment : OfficerAssignment) : Prop :=
  (∀ pos, isQualified (assignment pos) pos) ∧
  (∀ pos1 pos2, pos1 ≠ pos2 → assignment pos1 ≠ assignment pos2)

-- State the theorem
theorem officer_assignment_count :
  ∃ (assignments : Finset OfficerAssignment),
    (∀ a ∈ assignments, validAssignment a) ∧
    assignments.card = 6 :=
sorry

end NUMINAMATH_CALUDE_officer_assignment_count_l2665_266539


namespace NUMINAMATH_CALUDE_middle_seat_is_A_l2665_266597

/-- Represents the position of a person in the train -/
inductive Position
| first
| second
| third
| fourth
| fifth

/-- Represents a person -/
inductive Person
| A
| B
| C
| D
| E

/-- The seating arrangement in the train -/
def SeatingArrangement := Person → Position

theorem middle_seat_is_A (arrangement : SeatingArrangement) : 
  (arrangement Person.D = Position.fifth) →
  (arrangement Person.A = Position.fourth ∧ arrangement Person.E = Position.second) ∨
  (arrangement Person.A = Position.third ∧ arrangement Person.E = Position.second) →
  (arrangement Person.B = Position.first ∨ arrangement Person.B = Position.second) →
  (arrangement Person.B ≠ arrangement Person.C ∧ 
   arrangement Person.A ≠ arrangement Person.C ∧
   arrangement Person.E ≠ arrangement Person.C) →
  arrangement Person.A = Position.third :=
by sorry

end NUMINAMATH_CALUDE_middle_seat_is_A_l2665_266597


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2665_266582

theorem quadratic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f (f 0) = 0 ∧ f (f 1) = 0 ∧ f 0 ≠ f 1) → f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2665_266582


namespace NUMINAMATH_CALUDE_alex_max_correct_answers_l2665_266544

/-- Represents a math contest with multiple-choice questions. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ

/-- Represents a student's performance in the math contest. -/
structure StudentPerformance where
  contest : MathContest
  total_score : ℤ

/-- Calculates the maximum number of correct answers for a given student performance. -/
def max_correct_answers (perf : StudentPerformance) : ℕ :=
  sorry

/-- The theorem stating the maximum number of correct answers for Alex's performance. -/
theorem alex_max_correct_answers :
  let contest : MathContest := {
    total_questions := 80,
    correct_points := 5,
    blank_points := 0,
    incorrect_points := -2
  }
  let performance : StudentPerformance := {
    contest := contest,
    total_score := 150
  }
  max_correct_answers performance = 44 := by
  sorry

end NUMINAMATH_CALUDE_alex_max_correct_answers_l2665_266544


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l2665_266521

/-- A quadratic function f(x) = x^2 + 4x + c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) : ℝ → ℝ := fun x ↦ f c (f c x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_exactly_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    (g x = 0 ∧ g y = 0 ∧ g z = 0) ∧
    (∀ w : ℝ, g w = 0 → w = x ∨ w = y ∨ w = z)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = 1 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, has_exactly_three_distinct_real_roots (f_comp c) ↔ c = 1 :=
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l2665_266521


namespace NUMINAMATH_CALUDE_intersection_M_N_l2665_266504

def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {-2, 0, 1}

theorem intersection_M_N : M ∩ N = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2665_266504


namespace NUMINAMATH_CALUDE_bookshop_inventory_l2665_266543

/-- Bookshop inventory problem -/
theorem bookshop_inventory (initial_books : ℕ) (saturday_instore : ℕ) (saturday_online : ℕ) 
  (sunday_instore : ℕ) (shipment : ℕ) (final_books : ℕ) 
  (h1 : initial_books = 743)
  (h2 : saturday_instore = 37)
  (h3 : saturday_online = 128)
  (h4 : sunday_instore = 2 * saturday_instore)
  (h5 : shipment = 160)
  (h6 : final_books = 502) :
  ∃ (sunday_online : ℕ), 
    final_books = initial_books - (saturday_instore + saturday_online + sunday_instore + sunday_online) + shipment ∧ 
    sunday_online = saturday_online + 34 :=
by sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l2665_266543


namespace NUMINAMATH_CALUDE_cos_shift_right_l2665_266557

theorem cos_shift_right (x : ℝ) :
  let f (x : ℝ) := Real.cos (2 * x)
  let shift := π / 12
  let g (x : ℝ) := f (x - shift)
  g x = Real.cos (2 * x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_right_l2665_266557


namespace NUMINAMATH_CALUDE_woman_work_time_l2665_266548

/-- Represents the time taken to complete a work unit -/
structure WorkTime where
  men : ℕ
  women : ℕ
  days : ℕ

/-- The work done by one person in one day -/
def work_per_day (gender : String) : ℚ :=
  if gender = "man" then 1 / 100
  else 1 / 225

theorem woman_work_time : ∃ (w : ℚ),
  (10 * work_per_day "man" + 15 * w) * 6 = 1 ∧
  w = work_per_day "woman" ∧
  1 / w = 225 := by
  sorry

#check woman_work_time

end NUMINAMATH_CALUDE_woman_work_time_l2665_266548


namespace NUMINAMATH_CALUDE_boat_speed_problem_l2665_266536

/-- Proves that given a boat traveling 45 miles upstream in 5 hours and having a speed of 12 mph in still water, the speed of the current is 3 mph. -/
theorem boat_speed_problem (distance : ℝ) (time : ℝ) (still_water_speed : ℝ) 
  (h1 : distance = 45) 
  (h2 : time = 5) 
  (h3 : still_water_speed = 12) : 
  still_water_speed - (distance / time) = 3 := by
  sorry

#check boat_speed_problem

end NUMINAMATH_CALUDE_boat_speed_problem_l2665_266536


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2665_266558

def f (a : ℝ) (x : ℝ) := a * x^2 + 2 * a * x + 1

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = -3 ∨ a = 3/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2665_266558


namespace NUMINAMATH_CALUDE_flight_departure_requirement_l2665_266505

/-- The minimum number of people required for the flight to depart -/
def min_required : ℕ := 16

/-- The number of people currently on the plane -/
def current_people : ℕ := 9

/-- The number of additional people needed to board before departure -/
def additional_people : ℕ := min_required - current_people

theorem flight_departure_requirement :
  min_required > 15 ∧ current_people = 9 → additional_people = 7 := by
  sorry

end NUMINAMATH_CALUDE_flight_departure_requirement_l2665_266505


namespace NUMINAMATH_CALUDE_product_mod_23_l2665_266583

theorem product_mod_23 : (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l2665_266583


namespace NUMINAMATH_CALUDE_range_of_m_when_p_true_range_of_m_when_p_or_q_true_and_p_and_q_false_l2665_266529

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 ≥ m

-- Define proposition q
def q (m : ℝ) : Prop := ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (m - 2) * (m + 2) < 0 ∧
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 2) = 1 ↔ (x / a)^2 - (y / b)^2 = 1

-- Theorem 1
theorem range_of_m_when_p_true (m : ℝ) : p m → m ≤ 1 := by sorry

-- Theorem 2
theorem range_of_m_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ≤ -2 ∨ (1 < m ∧ m < 2) := by sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_true_range_of_m_when_p_or_q_true_and_p_and_q_false_l2665_266529


namespace NUMINAMATH_CALUDE_same_terminal_side_l2665_266572

theorem same_terminal_side (π : ℝ) : ∃ (k : ℤ), (7 / 6 * π) - (-5 / 6 * π) = k * (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l2665_266572


namespace NUMINAMATH_CALUDE_undefined_expression_l2665_266575

theorem undefined_expression (x : ℝ) : 
  (x - 1) / (x^2 - 5*x + 6) = 0⁻¹ ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_expression_l2665_266575


namespace NUMINAMATH_CALUDE_hyperbola_focus_l2665_266509

/-- Given a hyperbola with equation x²/a² - y²/2 = 1, where one of its asymptotes
    passes through the point (√2, 1), prove that one of its foci has coordinates (√6, 0) -/
theorem hyperbola_focus (a : ℝ) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 2 = 1) →  -- Hyperbola equation
  (∃ (m : ℝ), m * Real.sqrt 2 = 1 ∧ m = Real.sqrt 2 / a) →  -- Asymptote through (√2, 1)
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 2 = 1 ∧ x = Real.sqrt 6 ∧ y = 0) :=  -- Focus at (√6, 0)
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l2665_266509


namespace NUMINAMATH_CALUDE_student_group_arrangement_l2665_266516

theorem student_group_arrangement (n : ℕ) (p : ℚ) :
  n = 9 ∧ p = 16/21 →
  (∃ (m f : ℕ),
    m + f = n ∧
    (Nat.choose m 2 * Nat.choose f 1 + Nat.choose m 1 * Nat.choose f 2 + Nat.choose f 3) / Nat.choose n 3 = p ∧
    m = 6 ∧ f = 3) ∧
  (∃ (arrangements : ℕ),
    arrangements = 17280 ∧
    arrangements = (Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2) / (6 * 2 * 1) *
                   (3 * 2 * 1) * (4 * 3 * 2) *
                   (2 * 1) * (2 * 1) * (2 * 1)) :=
by sorry

end NUMINAMATH_CALUDE_student_group_arrangement_l2665_266516


namespace NUMINAMATH_CALUDE_total_cans_stored_l2665_266553

/-- Represents the capacity of a closet for storing cans -/
structure Closet where
  cansPerRow : ℕ
  rowsPerShelf : ℕ
  shelves : ℕ

/-- Calculates the total number of cans that can be stored in a closet -/
def closetCapacity (c : Closet) : ℕ :=
  c.cansPerRow * c.rowsPerShelf * c.shelves

/-- The first closet in Jack's emergency bunker -/
def closet1 : Closet :=
  { cansPerRow := 12
    rowsPerShelf := 4
    shelves := 10 }

/-- The second closet in Jack's emergency bunker -/
def closet2 : Closet :=
  { cansPerRow := 15
    rowsPerShelf := 5
    shelves := 8 }

/-- Theorem stating the total number of cans Jack can store in both closets -/
theorem total_cans_stored :
  closetCapacity closet1 + closetCapacity closet2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_stored_l2665_266553


namespace NUMINAMATH_CALUDE_expression_equality_l2665_266510

theorem expression_equality : 201 * 5 + 1220 - 2 * 3 * 5 * 7 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2665_266510


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2665_266534

theorem gcd_lcm_product (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2665_266534


namespace NUMINAMATH_CALUDE_special_collection_loans_l2665_266552

theorem special_collection_loans (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) :
  initial_books = 75 →
  final_books = 66 →
  return_rate = 70 / 100 →
  ∃ (loaned_books : ℕ), loaned_books = 30 ∧ 
    final_books = initial_books - (1 - return_rate) * loaned_books := by
  sorry

end NUMINAMATH_CALUDE_special_collection_loans_l2665_266552


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2665_266593

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ (n = 8 ∧ n > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2665_266593


namespace NUMINAMATH_CALUDE_intersection_point_properties_l2665_266513

/-- Point P where the given lines intersect -/
def P : ℝ × ℝ := (1, 1)

/-- Line perpendicular to 3x + 4y - 15 = 0 passing through P -/
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y - 1 = 0

/-- Line with equal intercepts passing through P -/
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0

theorem intersection_point_properties :
  (∃ (x y : ℝ), 2 * x - y - 1 = 0 ∧ x - 2 * y + 1 = 0 ∧ (x, y) = P) ∧
  (∀ (x y : ℝ), l₁ x y ↔ (4 * x - 3 * y - 1 = 0 ∧ (x, y) = P ∨ (x, y) ≠ P)) ∧
  (∀ (x y : ℝ), l₂ x y ↔ (x + y - 2 = 0 ∧ (x, y) = P ∨ (x, y) ≠ P)) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_properties_l2665_266513


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l2665_266562

/-- Represents a number as a sequence of digits in base 10 -/
def DigitSequence (d : Nat) (n : Nat) : Nat :=
  (10^n - 1) / 9 * d

/-- Calculates the sum of digits of a number in base 10 -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_9ab :
  let a := DigitSequence 9 2023
  let b := DigitSequence 6 2023
  sumOfDigits (9 * a * b) = 28314 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l2665_266562


namespace NUMINAMATH_CALUDE_most_stable_scores_l2665_266560

theorem most_stable_scores (S_A S_B S_C : ℝ) 
  (h1 : S_A = 38) (h2 : S_B = 10) (h3 : S_C = 26) :
  S_B < S_A ∧ S_B < S_C := by
  sorry

end NUMINAMATH_CALUDE_most_stable_scores_l2665_266560


namespace NUMINAMATH_CALUDE_inscribed_sphere_ratio_l2665_266554

/-- A regular tetrahedron with height H and an inscribed sphere of radius R -/
structure RegularTetrahedron where
  H : ℝ
  R : ℝ
  H_pos : H > 0
  R_pos : R > 0

/-- The ratio of the inscribed sphere radius to the tetrahedron height is 1:4 -/
theorem inscribed_sphere_ratio (t : RegularTetrahedron) : t.R / t.H = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_ratio_l2665_266554


namespace NUMINAMATH_CALUDE_solution_difference_l2665_266508

theorem solution_difference : ∃ p₁ p₂ : ℝ, 
  (p₁^2 - 5*p₁ - 11) / (p₁ + 3) = 3*p₁ + 9 ∧
  (p₂^2 - 5*p₂ - 11) / (p₂ + 3) = 3*p₂ + 9 ∧
  p₁ ≠ p₂ ∧
  |p₁ - p₂| = 7.5 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2665_266508


namespace NUMINAMATH_CALUDE_expression_equivalence_l2665_266500

variables (x y : ℝ)

def P : ℝ := 2*x + 3*y
def Q : ℝ := x - 2*y

theorem expression_equivalence :
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (2*x + 3*y) / (2*x + 10*y) :=
by sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2665_266500


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2665_266526

theorem max_product_sum_300 :
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 ∧ ∃ a b : ℤ, a + b = 300 ∧ a * b = 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2665_266526


namespace NUMINAMATH_CALUDE_at_least_one_parabola_has_two_roots_l2665_266527

-- Define the parabolas
def parabola1 (a b c x : ℝ) : ℝ := a * x^2 + 2 * b * x + c
def parabola2 (a b c x : ℝ) : ℝ := b * x^2 + 2 * c * x + a
def parabola3 (a b c x : ℝ) : ℝ := c * x^2 + 2 * a * x + b

-- Define a function to check if a parabola has two distinct roots
def has_two_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0

-- State the theorem
theorem at_least_one_parabola_has_two_roots (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  has_two_distinct_roots (parabola1 a b c) ∨ 
  has_two_distinct_roots (parabola2 a b c) ∨ 
  has_two_distinct_roots (parabola3 a b c) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_parabola_has_two_roots_l2665_266527


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_sold_megan_sold_three_gem_stone_necklaces_l2665_266596

/-- The number of gem stone necklaces sold at a garage sale -/
theorem gem_stone_necklaces_sold (bead_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) : ℕ :=
  let gem_stone_necklaces := (total_earnings - bead_necklaces * price_per_necklace) / price_per_necklace
  gem_stone_necklaces

/-- Proof that Megan sold 3 gem stone necklaces -/
theorem megan_sold_three_gem_stone_necklaces : 
  gem_stone_necklaces_sold 7 9 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_sold_megan_sold_three_gem_stone_necklaces_l2665_266596


namespace NUMINAMATH_CALUDE_triangle_angle_c_two_thirds_pi_l2665_266568

theorem triangle_angle_c_two_thirds_pi
  (A B C : Real) (a b c : Real)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : (a + b + c) * (Real.sin A + Real.sin B - Real.sin C) = a * Real.sin B) :
  C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_two_thirds_pi_l2665_266568


namespace NUMINAMATH_CALUDE_harmonic_arithmetic_mean_inequality_l2665_266589

theorem harmonic_arithmetic_mean_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let h := 3 / ((1 / a) + (1 / b) + (1 / c))
  let m := (a + b + c) / 3
  h ≤ m ∧ (h = m ↔ a = b ∧ b = c) := by
  sorry

#check harmonic_arithmetic_mean_inequality

end NUMINAMATH_CALUDE_harmonic_arithmetic_mean_inequality_l2665_266589


namespace NUMINAMATH_CALUDE_roots_condition_implies_a_equals_neg_nine_l2665_266595

/-- The polynomial p(x) = x³ - 6x² + ax + a, where a is a parameter --/
def p (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + a*x + a

/-- The condition that the sum of cubes of the roots minus 3 is zero --/
def sum_of_cubes_minus_3_is_zero (x₁ x₂ x₃ : ℝ) : Prop :=
  (x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0

theorem roots_condition_implies_a_equals_neg_nine (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 
    (∀ x : ℝ, p a x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    sum_of_cubes_minus_3_is_zero x₁ x₂ x₃) →
  a = -9 :=
sorry

end NUMINAMATH_CALUDE_roots_condition_implies_a_equals_neg_nine_l2665_266595


namespace NUMINAMATH_CALUDE_calculate_Y_l2665_266538

theorem calculate_Y : 
  let P : ℚ := 208 / 4
  let Q : ℚ := P / 2
  let Y : ℚ := P - Q * (10 / 100)
  Y = 49.4 := by sorry

end NUMINAMATH_CALUDE_calculate_Y_l2665_266538


namespace NUMINAMATH_CALUDE_find_a_value_l2665_266515

def A : Set ℝ := {0, 1, 2}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 3}

theorem find_a_value :
  ∀ a : ℝ, (A ∩ B a = {1}) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l2665_266515


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2665_266507

/-- Given two points A(-2, m) and B(m, 4), and that the line AB is perpendicular
    to the line x - 2y = 0, prove that m = -8 -/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (let A : ℝ × ℝ := (-2, m)
   let B : ℝ × ℝ := (m, 4)
   let slope_AB := (B.2 - A.2) / (B.1 - A.1)
   let slope_perpendicular := 1 / 2
   slope_AB * slope_perpendicular = -1) →
  m = -8 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2665_266507


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l2665_266541

/-- The time taken for a monkey to climb a tree, given the tree height, hop distance, slip distance, and final hop distance. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) : ℕ :=
  let net_progress := hop_distance - slip_distance
  let distance_before_final_hop := tree_height - final_hop
  distance_before_final_hop / net_progress + 1

/-- Theorem stating that a monkey climbing a 20 ft tree, hopping 3 ft and slipping 2 ft each hour, with a final 3 ft hop, takes 18 hours to reach the top. -/
theorem monkey_climb_theorem :
  monkey_climb_time 20 3 2 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l2665_266541


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2665_266587

/-- The line kx+y+2=0 intersects the circle (x-1)^2+(y+2)^2=16 for all real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (k * x + y + 2 = 0) ∧ ((x - 1)^2 + (y + 2)^2 = 16) := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2665_266587


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2665_266591

-- Define the radius of the sun in kilometers
def sun_radius_km : ℝ := 696000

-- Define the conversion factor from kilometers to meters
def km_to_m : ℝ := 1000

-- Theorem to prove
theorem sun_radius_scientific_notation :
  sun_radius_km * km_to_m = 6.96 * (10 ^ 8) :=
by sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2665_266591


namespace NUMINAMATH_CALUDE_linear_function_properties_l2665_266570

/-- A linear function passing through two points and intersecting a horizontal line -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_properties 
  (k b : ℝ) 
  (h_k : k ≠ 0)
  (h_point_A : LinearFunction k b 0 = 1)
  (h_point_B : LinearFunction k b 1 = 2)
  (h_intersect : ∃ x, LinearFunction k b x = 4) :
  (k = 1 ∧ b = 1) ∧ 
  (∃ x, x = 3 ∧ LinearFunction k b x = 4) ∧
  (∀ x, x < 3 → (2/3 * x + 2 > LinearFunction k b x ∧ 2/3 * x + 2 < 4)) := by
  sorry

#check linear_function_properties

end NUMINAMATH_CALUDE_linear_function_properties_l2665_266570


namespace NUMINAMATH_CALUDE_document_typing_time_l2665_266563

theorem document_typing_time 
  (total_time : ℝ) 
  (susan_time : ℝ) 
  (jack_time : ℝ) 
  (h1 : total_time = 10) 
  (h2 : susan_time = 30) 
  (h3 : jack_time = 24) : 
  ∃ jonathan_time : ℝ, 
    jonathan_time = 40 ∧ 
    1 / total_time = 1 / jonathan_time + 1 / susan_time + 1 / jack_time :=
by
  sorry

#check document_typing_time

end NUMINAMATH_CALUDE_document_typing_time_l2665_266563


namespace NUMINAMATH_CALUDE_division_ratio_l2665_266594

theorem division_ratio (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : ∃ n : ℕ, divisor = n * remainder)
  (h3 : remainder = 46)
  (h4 : 5290 = divisor * quotient + remainder) :
  divisor / remainder = 5 :=
sorry

end NUMINAMATH_CALUDE_division_ratio_l2665_266594


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2665_266581

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  ∀ c d : ℝ, c > 0 → d > 0 →
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) →
  (∃ x : ℝ, x^2 + 3*d*x + c = 0) →
  a + b ≤ c + d ∧ a + b ≥ 6.5 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2665_266581


namespace NUMINAMATH_CALUDE_journey_time_calculation_l2665_266532

/-- Given a journey with the following conditions:
    - Total distance is 224 km
    - Journey is divided into two equal halves
    - First half is traveled at 21 km/hr
    - Second half is traveled at 24 km/hr
    The total time taken to complete the journey is 10 hours. -/
theorem journey_time_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ) :
  total_distance = 224 →
  first_half_speed = 21 →
  second_half_speed = 24 →
  (total_distance / 2 / first_half_speed) + (total_distance / 2 / second_half_speed) = 10 := by
sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l2665_266532


namespace NUMINAMATH_CALUDE_ricks_ironing_total_l2665_266530

/-- Rick's ironing problem -/
theorem ricks_ironing_total (shirts_per_hour pants_per_hour shirt_hours pant_hours : ℕ) 
  (h1 : shirts_per_hour = 4)
  (h2 : pants_per_hour = 3)
  (h3 : shirt_hours = 3)
  (h4 : pant_hours = 5) :
  shirts_per_hour * shirt_hours + pants_per_hour * pant_hours = 27 := by
  sorry

#check ricks_ironing_total

end NUMINAMATH_CALUDE_ricks_ironing_total_l2665_266530


namespace NUMINAMATH_CALUDE_probability_failed_chinese_given_failed_math_l2665_266523

theorem probability_failed_chinese_given_failed_math 
  (total_students : ℕ) 
  (failed_math : ℕ) 
  (failed_chinese : ℕ) 
  (failed_both : ℕ) 
  (h1 : failed_math = (25 : ℕ) * total_students / 100)
  (h2 : failed_chinese = (10 : ℕ) * total_students / 100)
  (h3 : failed_both = (5 : ℕ) * total_students / 100)
  (h4 : total_students > 0) :
  (failed_both : ℚ) / failed_math = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_failed_chinese_given_failed_math_l2665_266523


namespace NUMINAMATH_CALUDE_square_equation_solution_l2665_266585

theorem square_equation_solution : ∃! (M : ℕ), M > 0 ∧ 16^2 * 40^2 = 20^2 * M^2 ∧ M = 32 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2665_266585


namespace NUMINAMATH_CALUDE_distance_ratio_l2665_266537

/-- Yan's travel scenario --/
structure TravelScenario where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to his office
  y : ℝ  -- Distance from Yan to the concert hall
  h_positive : w > 0 ∧ x > 0 ∧ y > 0  -- Positive distances and speed

/-- The time taken for both travel options is equal --/
def equal_time (s : TravelScenario) : Prop :=
  s.y / s.w = (s.x / s.w + (s.x + s.y) / (5 * s.w))

/-- The theorem stating the ratio of distances --/
theorem distance_ratio (s : TravelScenario) 
  (h_equal_time : equal_time s) : 
  s.x / s.y = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_ratio_l2665_266537


namespace NUMINAMATH_CALUDE_problem_statement_l2665_266549

theorem problem_statement (a b c : ℝ) 
  (h1 : a * b * c ≠ 0) 
  (h2 : a + b + c = 2) 
  (h3 : a^2 + b^2 + c^2 = 2) : 
  (1 - a)^2 / (b * c) + (1 - b)^2 / (c * a) + (1 - c)^2 / (a * b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2665_266549


namespace NUMINAMATH_CALUDE_triangle_problem_l2665_266519

noncomputable def f (x θ : Real) : Real :=
  2 * Real.sin x * (Real.cos (θ / 2))^2 + Real.cos x * Real.sin θ - Real.sin x

theorem triangle_problem (θ : Real) (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : ∀ x, f x θ ≥ f π θ) :
  ∃ (A B C : Real),
    θ = π / 2 ∧
    0 < A ∧ A < π ∧
    0 < B ∧ B < π ∧
    0 < C ∧ C < π ∧
    A + B + C = π ∧
    Real.sin B / Real.sin A = Real.sqrt 2 ∧
    f A (π / 2) = Real.sqrt 3 / 2 ∧
    (C = 7 * π / 12 ∨ C = π / 12) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2665_266519


namespace NUMINAMATH_CALUDE_m_range_l2665_266511

theorem m_range (m : ℝ) : 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
  (∀ x : ℝ, x^2 + (m-2)*x + 1 ≠ 0) ∧
  ¬((∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
    (∀ x : ℝ, x^2 + (m-2)*x + 1 ≠ 0)) →
  m ∈ Set.Ioo 0 2 ∪ Set.Ici 4 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2665_266511


namespace NUMINAMATH_CALUDE_pentomino_circumscribing_rectangle_ratio_l2665_266567

/-- A pentomino is a planar geometric figure formed by joining five equal squares edge to edge. -/
structure Pentomino where
  -- Add necessary fields to represent a pentomino
  -- This is a placeholder and may need to be expanded based on specific requirements

/-- A rectangle that circumscribes a pentomino. -/
structure CircumscribingRectangle (p : Pentomino) where
  width : ℝ
  height : ℝ
  -- Add necessary fields to represent the relationship between the pentomino and the rectangle
  -- This is a placeholder and may need to be expanded based on specific requirements

/-- The theorem stating that for any pentomino inscribed in a rectangle, 
    the ratio of the shorter side to the longer side of the rectangle is 1:2. -/
theorem pentomino_circumscribing_rectangle_ratio (p : Pentomino) 
  (r : CircumscribingRectangle p) : 
  min r.width r.height / max r.width r.height = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pentomino_circumscribing_rectangle_ratio_l2665_266567


namespace NUMINAMATH_CALUDE_coefficient_x4_in_product_l2665_266520

/-- The coefficient of x^4 in the expansion of (2x^3 + 5x^2 - 3x)(3x^3 - 8x^2 + 6x - 9) is -37 -/
theorem coefficient_x4_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 5 * X^2 - 3 * X
  let p₂ : Polynomial ℤ := 3 * X^3 - 8 * X^2 + 6 * X - 9
  (p₁ * p₂).coeff 4 = -37 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_product_l2665_266520


namespace NUMINAMATH_CALUDE_guessing_game_theorem_l2665_266555

/-- The guessing game between Banana and Corona -/
def GuessGame (n k : ℕ) : Prop :=
  n > 0 ∧ k > 0 ∧ k ≤ 2^n

/-- Corona can determine x in finitely many turns -/
def CanDetermine (n k : ℕ) : Prop :=
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ n → ∃ t : ℕ, t > 0 ∧ (∀ y : ℕ, 1 ≤ y ∧ y ≤ n → y = x)

/-- The main theorem: Corona can determine x iff k ≤ 2^(n-1) -/
theorem guessing_game_theorem (n k : ℕ) :
  GuessGame n k → (CanDetermine n k ↔ k ≤ 2^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_guessing_game_theorem_l2665_266555


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2665_266578

/-- Proves that in an election with two candidates, if the first candidate got 60% of the votes
    and the second candidate got 240 votes, then the total number of votes was 600. -/
theorem election_votes_theorem (total_votes : ℕ) (first_candidate_percentage : ℚ) 
    (second_candidate_votes : ℕ) : 
    first_candidate_percentage = 60 / 100 →
    second_candidate_votes = 240 →
    (1 - first_candidate_percentage) * total_votes = second_candidate_votes →
    total_votes = 600 := by
  sorry

#check election_votes_theorem

end NUMINAMATH_CALUDE_election_votes_theorem_l2665_266578


namespace NUMINAMATH_CALUDE_castle_entry_exit_ways_l2665_266522

/-- The number of windows in the castle -/
def num_windows : ℕ := 8

/-- The number of ways to enter and exit the castle through different windows -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: Given a castle with 8 windows, the number of ways to enter through
    one window and exit through a different window is 56 -/
theorem castle_entry_exit_ways :
  num_windows = 8 → num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_castle_entry_exit_ways_l2665_266522


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l2665_266546

/-- For an infinite geometric series with first term 512 and sum 2048, the common ratio is 3/4 -/
theorem infinite_geometric_series_ratio (a : ℝ) (S : ℝ) (r : ℝ) : 
  a = 512 → S = 2048 → S = a / (1 - r) → r = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l2665_266546


namespace NUMINAMATH_CALUDE_equal_vectors_have_equal_magnitudes_l2665_266506

theorem equal_vectors_have_equal_magnitudes {V : Type*} [NormedAddCommGroup V] 
  {a b : V} (h : a = b) : ‖a‖ = ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_equal_vectors_have_equal_magnitudes_l2665_266506


namespace NUMINAMATH_CALUDE_connie_savings_theorem_connie_savings_value_l2665_266586

/-- The amount of money Connie saved up -/
def connie_savings : ℕ := sorry

/-- The cost of the watch -/
def watch_cost : ℕ := 55

/-- The additional amount Connie needs -/
def additional_needed : ℕ := 16

/-- Theorem stating that Connie's savings plus the additional amount needed equals the watch cost -/
theorem connie_savings_theorem : connie_savings + additional_needed = watch_cost := by sorry

/-- Theorem proving that Connie's savings equal $39 -/
theorem connie_savings_value : connie_savings = 39 := by sorry

end NUMINAMATH_CALUDE_connie_savings_theorem_connie_savings_value_l2665_266586


namespace NUMINAMATH_CALUDE_find_A_l2665_266514

def round_down_hundreds (n : ℕ) : ℕ := n / 100 * 100

def is_valid_number (n : ℕ) : Prop := 
  ∃ (A : ℕ), A < 10 ∧ n = 1000 + A * 100 + 77

theorem find_A : 
  ∀ (n : ℕ), is_valid_number n → round_down_hundreds n = 1700 → n = 1777 :=
sorry

end NUMINAMATH_CALUDE_find_A_l2665_266514


namespace NUMINAMATH_CALUDE_original_number_proof_l2665_266547

theorem original_number_proof (x : ℝ) : x * 1.2 = 1080 → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2665_266547


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2665_266571

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem sufficient_not_necessary : 
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ 
  (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2665_266571


namespace NUMINAMATH_CALUDE_odd_sum_probability_l2665_266551

/-- The first 15 prime numbers -/
def first_15_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

/-- The number of ways to select 3 primes from the first 15 primes -/
def total_selections : Nat := Nat.choose 15 3

/-- The number of ways to select 3 primes from the first 15 primes such that their sum is odd -/
def odd_sum_selections : Nat := Nat.choose 14 2

theorem odd_sum_probability :
  (odd_sum_selections : ℚ) / total_selections = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l2665_266551


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2665_266561

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The ninth term of an arithmetic sequence is 32, given that its third term is 20 and its sixth term is 26. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℕ)
  (h_arith : ArithmeticSequence a)
  (h_third : a 3 = 20)
  (h_sixth : a 6 = 26) :
  a 9 = 32 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2665_266561


namespace NUMINAMATH_CALUDE_headlight_cost_is_180_l2665_266502

/-- Represents the scenario of Chris selling his car with two different offers --/
def car_sale_scenario (asking_price : ℝ) (maintenance_cost : ℝ) (headlight_cost : ℝ) : Prop :=
  let tire_cost := 3 * headlight_cost
  let first_offer := asking_price - maintenance_cost
  let second_offer := asking_price - (headlight_cost + tire_cost)
  (maintenance_cost = asking_price / 10) ∧
  (first_offer - second_offer = 200)

/-- Theorem stating that given the conditions, the headlight replacement cost is $180 --/
theorem headlight_cost_is_180 :
  car_sale_scenario 5200 520 180 :=
sorry

end NUMINAMATH_CALUDE_headlight_cost_is_180_l2665_266502


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2665_266580

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2665_266580


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2665_266590

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - k ≠ 0) → k < -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2665_266590


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2665_266517

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ, Real.sqrt (a m * a n) = 2 * Real.sqrt 2 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 11 / 6) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 11 / 6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2665_266517


namespace NUMINAMATH_CALUDE_circle_symmetry_range_l2665_266501

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5*a = 0

-- Define the symmetry line equation
def symmetry_line (x y b : ℝ) : Prop :=
  y = x + 2*b

-- Theorem statement
theorem circle_symmetry_range (a b : ℝ) :
  (∃ x y : ℝ, circle_equation x y a ∧ symmetry_line x y b) →
  a + b < 0 ∧ ∀ c, c < 0 → ∃ a' b', a' + b' = c ∧
    ∃ x y : ℝ, circle_equation x y a' ∧ symmetry_line x y b' :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_range_l2665_266501


namespace NUMINAMATH_CALUDE_smallest_covering_triangular_l2665_266542

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to determine the row of a number on the snaking board -/
def board_row (n : ℕ) : ℕ :=
  if n % 20 ≤ 10 then (n - 1) / 10 + 1 else (n + 9) / 10

/-- Theorem stating that 91 is the smallest triangular number that covers all rows -/
theorem smallest_covering_triangular : 
  (∀ k < 13, ∃ r ≤ 10, ∀ i ≤ 10, board_row (triangular k) ≠ i) ∧
  (∀ i ≤ 10, ∃ k ≤ 13, board_row (triangular k) = i) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_triangular_l2665_266542


namespace NUMINAMATH_CALUDE_simplify_exponential_fraction_l2665_266577

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+3) - 3*(3^n)) / (3*(3^(n+2))) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponential_fraction_l2665_266577


namespace NUMINAMATH_CALUDE_factorial_equality_l2665_266550

theorem factorial_equality : ∃ N : ℕ+, Nat.factorial 7 * Nat.factorial 11 = 18 * Nat.factorial N.val := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l2665_266550


namespace NUMINAMATH_CALUDE_sum_of_integers_problem_l2665_266564

theorem sum_of_integers_problem : ∃ (a b : ℕ), 
  (a > 0) ∧ (b > 0) ∧
  (a * b + a + b - (a - b) = 120) ∧
  (Nat.gcd a b = 1) ∧
  (a < 25) ∧ (b < 25) ∧
  (a + b = 19) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_problem_l2665_266564


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2665_266573

theorem quadratic_root_problem (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*k*x + k - 1 = 0 ∧ x = 0) → 
  (∃ y : ℝ, y^2 + 2*k*y + k - 1 = 0 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2665_266573


namespace NUMINAMATH_CALUDE_charlie_votes_l2665_266525

/-- Represents a candidate in the student council election -/
inductive Candidate
| Alex
| Brenda
| Charlie
| Dana

/-- Represents the vote count for each candidate -/
def VoteCount := Candidate → ℕ

/-- The total number of votes cast in the election -/
def TotalVotes (votes : VoteCount) : ℕ :=
  votes Candidate.Alex + votes Candidate.Brenda + votes Candidate.Charlie + votes Candidate.Dana

theorem charlie_votes (votes : VoteCount) : 
  votes Candidate.Brenda = 40 ∧ 
  4 * votes Candidate.Brenda = TotalVotes votes ∧
  votes Candidate.Charlie = votes Candidate.Dana + 10 →
  votes Candidate.Charlie = 45 := by
  sorry

end NUMINAMATH_CALUDE_charlie_votes_l2665_266525


namespace NUMINAMATH_CALUDE_min_value_theorem_l2665_266556

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y - 9 = 0) :
  2/y + 1/x ≥ 1 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 2*y' - 9 = 0 ∧ 2/y' + 1/x' = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2665_266556


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2665_266588

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : 4 * a - 4 * b + c > 0) 
  (h2 : a + 2 * b + c < 0) : 
  b^2 > a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2665_266588


namespace NUMINAMATH_CALUDE_smallest_m_is_correct_l2665_266599

/-- The smallest positive value of m for which 15x^2 - mx + 630 = 0 has integral solutions -/
def smallest_m : ℕ := 195

/-- The equation 15x^2 - mx + 630 = 0 has integral solutions -/
def has_integral_solutions (m : ℕ) : Prop :=
  ∃ x : ℤ, 15 * x^2 - m * x + 630 = 0

/-- The main theorem: smallest_m is the smallest positive value of m for which
    the equation 15x^2 - mx + 630 = 0 has integral solutions -/
theorem smallest_m_is_correct :
  has_integral_solutions smallest_m ∧
  ∀ m : ℕ, 0 < m → m < smallest_m → ¬(has_integral_solutions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_correct_l2665_266599


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_geq_3_l2665_266533

theorem empty_solution_set_implies_a_geq_3 (a : ℝ) : 
  (∀ x : ℝ, ¬((x - 2) / 5 + 2 > x - 4 / 5 ∧ x > a)) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_geq_3_l2665_266533


namespace NUMINAMATH_CALUDE_five_pq_odd_l2665_266540

theorem five_pq_odd (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (5 * p * q) := by
  sorry

end NUMINAMATH_CALUDE_five_pq_odd_l2665_266540


namespace NUMINAMATH_CALUDE_helga_wrote_250_articles_l2665_266579

/-- Represents Helga's work schedule and article writing capacity --/
structure HelgaWorkSchedule where
  articles_per_half_hour : ℕ
  regular_hours_per_day : ℕ
  regular_days_per_week : ℕ
  extra_hours_thursday : ℕ
  extra_hours_friday : ℕ

/-- Calculates the total number of articles Helga wrote in a week --/
def total_articles_in_week (schedule : HelgaWorkSchedule) : ℕ :=
  let articles_per_hour := schedule.articles_per_half_hour * 2
  let regular_articles := articles_per_hour * schedule.regular_hours_per_day * schedule.regular_days_per_week
  let extra_articles := articles_per_hour * (schedule.extra_hours_thursday + schedule.extra_hours_friday)
  regular_articles + extra_articles

/-- Theorem stating that Helga wrote 250 articles in the given week --/
theorem helga_wrote_250_articles : 
  ∀ (schedule : HelgaWorkSchedule), 
    schedule.articles_per_half_hour = 5 ∧ 
    schedule.regular_hours_per_day = 4 ∧ 
    schedule.regular_days_per_week = 5 ∧ 
    schedule.extra_hours_thursday = 2 ∧ 
    schedule.extra_hours_friday = 3 →
    total_articles_in_week schedule = 250 := by
  sorry

end NUMINAMATH_CALUDE_helga_wrote_250_articles_l2665_266579


namespace NUMINAMATH_CALUDE_equation_solutions_l2665_266528

theorem equation_solutions (x : ℝ) : 2 * x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1/2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2665_266528
