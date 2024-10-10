import Mathlib

namespace treasure_hunt_probability_l1995_199540

def probability_gold : ℚ := 1 / 5
def probability_danger : ℚ := 1 / 10
def probability_neither : ℚ := 4 / 5
def total_caves : ℕ := 5
def gold_caves : ℕ := 2

theorem treasure_hunt_probability :
  (Nat.choose total_caves gold_caves : ℚ) *
  probability_gold ^ gold_caves *
  probability_neither ^ (total_caves - gold_caves) =
  128 / 625 := by sorry

end treasure_hunt_probability_l1995_199540


namespace prove_depletion_rate_l1995_199562

-- Define the initial value of the machine
def initial_value : ℝ := 2500

-- Define the value of the machine after 2 years
def value_after_2_years : ℝ := 2256.25

-- Define the number of years
def years : ℝ := 2

-- Define the depletion rate
def depletion_rate : ℝ := 0.05

-- Theorem to prove that the given depletion rate is correct
theorem prove_depletion_rate : 
  value_after_2_years = initial_value * (1 - depletion_rate) ^ years := by
  sorry


end prove_depletion_rate_l1995_199562


namespace inequality_solution_set_l1995_199500

theorem inequality_solution_set (x : ℝ) : 
  (x^2 + 8*x < 20) ↔ (-10 < x ∧ x < 2) :=
by sorry

end inequality_solution_set_l1995_199500


namespace product_trailing_zeros_l1995_199522

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros :
  trailing_zeros (50 * 720 * 125) = 5 := by sorry

end product_trailing_zeros_l1995_199522


namespace circle_equation_from_center_and_chord_l1995_199520

/-- The equation of a circle given its center and a chord. -/
theorem circle_equation_from_center_and_chord 
  (center_x center_y : ℝ) 
  (line1 : ℝ → ℝ → ℝ) (line2 : ℝ → ℝ → ℝ) (line3 : ℝ → ℝ → ℝ)
  (h1 : line1 center_x center_y = 0)
  (h2 : line2 center_x center_y = 0)
  (h3 : ∃ (A B : ℝ × ℝ), line3 A.1 A.2 = 0 ∧ line3 B.1 B.2 = 0)
  (h4 : ∀ (A B : ℝ × ℝ), line3 A.1 A.2 = 0 → line3 B.1 B.2 = 0 → 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)
  (h5 : line1 x y = x - y - 1)
  (h6 : line2 x y = 2*x - y - 1)
  (h7 : line3 x y = 3*x + 4*y - 11) :
  ∀ (x y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 18 :=
by sorry

end circle_equation_from_center_and_chord_l1995_199520


namespace circle_a_range_l1995_199537

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + a = 0

-- Define what it means for an equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y, circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

-- Theorem statement
theorem circle_a_range (a : ℝ) :
  is_circle a → a < 5 :=
sorry

end circle_a_range_l1995_199537


namespace remaining_pie_portion_l1995_199586

/-- Proves that given Carlos took 60% of a whole pie and Maria took one quarter of the remainder, 
    the portion of the whole pie left is 30%. -/
theorem remaining_pie_portion 
  (carlos_portion : Real) 
  (maria_portion : Real) 
  (h1 : carlos_portion = 0.6) 
  (h2 : maria_portion = 0.25 * (1 - carlos_portion)) : 
  1 - carlos_portion - maria_portion = 0.3 := by
sorry

end remaining_pie_portion_l1995_199586


namespace negation_of_sum_squares_l1995_199542

theorem negation_of_sum_squares (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 := by
  sorry

end negation_of_sum_squares_l1995_199542


namespace initial_number_proof_l1995_199589

theorem initial_number_proof : ∃ (N : ℕ), N > 0 ∧ (N - 10) % 21 = 0 ∧ ∀ (M : ℕ), M > 0 → (M - 10) % 21 = 0 → M ≥ N := by
  sorry

end initial_number_proof_l1995_199589


namespace binomial_expansion_terms_l1995_199574

theorem binomial_expansion_terms (n : ℕ+) : 
  (Finset.range (2*n + 1)).card = 2*n + 1 := by sorry

end binomial_expansion_terms_l1995_199574


namespace point_transformation_l1995_199534

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate_180 c d 2 3
  let (x₂, y₂) := reflect_y_eq_x x₁ y₁
  (x₂ = 2 ∧ y₂ = -1) → d - c = -1 := by
  sorry

end point_transformation_l1995_199534


namespace daniel_and_elsie_crackers_l1995_199501

/-- The amount of crackers Matthew had initially -/
def initial_crackers : ℝ := 27.5

/-- The amount of crackers Ally ate -/
def ally_crackers : ℝ := 3.5

/-- The amount of crackers Bob ate -/
def bob_crackers : ℝ := 4

/-- The amount of crackers Clair ate -/
def clair_crackers : ℝ := 5.5

/-- The amount of crackers Matthew had left after giving to Ally, Bob, and Clair -/
def remaining_crackers : ℝ := 10.5

/-- The theorem stating that Daniel and Elsie ate 4 crackers combined -/
theorem daniel_and_elsie_crackers : 
  initial_crackers - (ally_crackers + bob_crackers + clair_crackers) - remaining_crackers = 4 := by
  sorry

end daniel_and_elsie_crackers_l1995_199501


namespace f_max_min_on_interval_l1995_199516

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem f_max_min_on_interval :
  let a := 1
  let b := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 4 ∧ x_max = 3 ∧ f x_min = 0 ∧ x_min = 2 :=
by sorry

end f_max_min_on_interval_l1995_199516


namespace final_time_and_sum_l1995_199560

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time -/
def addTime (start : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Converts a time to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  sorry

/-- Theorem: Given the starting time and duration, prove the final time and sum -/
theorem final_time_and_sum 
  (start : Time)
  (durationHours durationMinutes durationSeconds : Nat) : 
  start.hours = 3 ∧ start.minutes = 0 ∧ start.seconds = 0 →
  durationHours = 313 ∧ durationMinutes = 45 ∧ durationSeconds = 56 →
  let finalTime := to12HourFormat (addTime start durationHours durationMinutes durationSeconds)
  finalTime.hours = 4 ∧ finalTime.minutes = 45 ∧ finalTime.seconds = 56 ∧
  finalTime.hours + finalTime.minutes + finalTime.seconds = 105 :=
by sorry

end final_time_and_sum_l1995_199560


namespace meaningful_reciprocal_l1995_199567

theorem meaningful_reciprocal (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end meaningful_reciprocal_l1995_199567


namespace lcm_of_12_18_30_l1995_199554

theorem lcm_of_12_18_30 : Nat.lcm (Nat.lcm 12 18) 30 = 180 := by
  sorry

end lcm_of_12_18_30_l1995_199554


namespace sum_of_squares_of_consecutive_even_numbers_l1995_199596

theorem sum_of_squares_of_consecutive_even_numbers : 
  ∃ (a b c d : ℕ), 
    (∃ (n : ℕ), a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) ∧ 
    a + b + c + d = 36 → 
    a^2 + b^2 + c^2 + d^2 = 344 := by
  sorry

end sum_of_squares_of_consecutive_even_numbers_l1995_199596


namespace bamboo_pole_problem_l1995_199599

theorem bamboo_pole_problem (pole_length : ℝ) (point_distance : ℝ) 
  (h_pole_length : pole_length = 24)
  (h_point_distance : point_distance = 7) :
  ∃ (height : ℝ), height = 16 + 4 * Real.sqrt 2 ∨ height = 16 - 4 * Real.sqrt 2 := by
  sorry

end bamboo_pole_problem_l1995_199599


namespace f_max_min_difference_l1995_199582

noncomputable def f (x : ℝ) : ℝ := 4 * Real.pi * Real.arcsin x - (Real.arccos (-x))^2

theorem f_max_min_difference :
  ∃ (M m : ℝ), (∀ x : ℝ, f x ≤ M ∧ f x ≥ m) ∧ M - m = 3 * Real.pi^2 := by
  sorry

end f_max_min_difference_l1995_199582


namespace milk_packs_per_set_l1995_199530

/-- The number of packs in each set of milk -/
def packs_per_set : ℕ := sorry

/-- The cost of a set of milk packs in dollars -/
def cost_per_set : ℚ := 2.5

/-- The cost of an individual milk pack in dollars -/
def cost_per_pack : ℚ := 1.3

/-- The total savings from buying ten sets in dollars -/
def total_savings : ℚ := 1

theorem milk_packs_per_set :
  packs_per_set = 2 ∧
  10 * cost_per_set + total_savings = 10 * packs_per_set * cost_per_pack :=
sorry

end milk_packs_per_set_l1995_199530


namespace largest_n_for_product_2304_l1995_199506

theorem largest_n_for_product_2304 :
  ∀ (d_a d_b : ℤ),
  ∃ (n : ℕ),
  (∀ k : ℕ, (1 + (k - 1) * d_a) * (3 + (k - 1) * d_b) = 2304 → k ≤ n) ∧
  (1 + (n - 1) * d_a) * (3 + (n - 1) * d_b) = 2304 ∧
  n = 20 :=
by sorry

end largest_n_for_product_2304_l1995_199506


namespace system_and_linear_equation_solution_l1995_199517

theorem system_and_linear_equation_solution (a : ℝ) :
  (∃ x y : ℝ, x + y = a ∧ x - y = 4*a ∧ 3*x - 5*y - 90 = 0) → a = 6 := by
  sorry

end system_and_linear_equation_solution_l1995_199517


namespace garrison_provisions_theorem_l1995_199544

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def daysProvisionsLast (initialMen : ℕ) (initialDays : ℕ) (reinforcementMen : ℕ) (daysPassed : ℕ) : ℕ :=
  let totalProvisions := initialMen * initialDays
  let remainingProvisions := totalProvisions - (initialMen * daysPassed)
  let totalMenAfterReinforcement := initialMen + reinforcementMen
  remainingProvisions / totalMenAfterReinforcement

/-- Theorem stating that given the specific conditions, provisions will last 10 more days -/
theorem garrison_provisions_theorem :
  daysProvisionsLast 2000 40 2000 20 = 10 := by
  sorry

#eval daysProvisionsLast 2000 40 2000 20

end garrison_provisions_theorem_l1995_199544


namespace completing_square_transform_l1995_199536

theorem completing_square_transform (x : ℝ) : 
  (x^2 - 2*x = 9) ↔ ((x - 1)^2 = 10) :=
by
  sorry

end completing_square_transform_l1995_199536


namespace unique_solution_triple_l1995_199587

theorem unique_solution_triple (x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y - 1)^2 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_solution_triple_l1995_199587


namespace percentage_problem_l1995_199527

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 200) : 
  (1200 / x) * 100 = 120 := by
sorry

end percentage_problem_l1995_199527


namespace cd_cost_l1995_199502

theorem cd_cost (two_cd_cost : ℝ) (h : two_cd_cost = 36) :
  8 * (two_cd_cost / 2) = 144 := by
sorry

end cd_cost_l1995_199502


namespace backyard_width_calculation_l1995_199543

/-- Given a rectangular backyard with a rectangular shed, calculate the width of the backyard -/
theorem backyard_width_calculation 
  (backyard_length : ℝ) 
  (shed_length shed_width : ℝ) 
  (sod_area : ℝ) :
  backyard_length = 20 →
  shed_length = 3 →
  shed_width = 5 →
  sod_area = 245 →
  ∃ (backyard_width : ℝ), 
    sod_area = backyard_length * backyard_width - shed_length * shed_width ∧ 
    backyard_width = 13 :=
by sorry

end backyard_width_calculation_l1995_199543


namespace additional_marbles_needed_l1995_199591

def friends : ℕ := 12
def current_marbles : ℕ := 50

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem additional_marbles_needed : 
  sum_first_n friends - current_marbles = 28 := by
  sorry

end additional_marbles_needed_l1995_199591


namespace max_handshakes_l1995_199573

theorem max_handshakes (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 := by
  sorry

#check max_handshakes

end max_handshakes_l1995_199573


namespace negative_x_implies_positive_expression_l1995_199545

theorem negative_x_implies_positive_expression (x : ℝ) (h : x < 0) : -3 * x⁻¹ > 0 := by
  sorry

end negative_x_implies_positive_expression_l1995_199545


namespace bookstore_sales_l1995_199568

/-- Given a store that sold 72 books and has a ratio of books to bookmarks sold of 9:2,
    prove that the number of bookmarks sold is 16. -/
theorem bookstore_sales (books_sold : ℕ) (book_ratio : ℕ) (bookmark_ratio : ℕ) 
    (h1 : books_sold = 72)
    (h2 : book_ratio = 9)
    (h3 : bookmark_ratio = 2) :
    (books_sold * bookmark_ratio) / book_ratio = 16 := by
  sorry

end bookstore_sales_l1995_199568


namespace q_function_determination_l1995_199570

theorem q_function_determination (q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →  -- q is quadratic
  q 3 = 0 →                                       -- vertical asymptote at x = 3
  q (-3) = 0 →                                    -- vertical asymptote at x = -3
  q 2 = 18 →                                      -- given condition
  ∀ x, q x = -((18 : ℝ) / 5) * x^2 + (162 : ℝ) / 5 :=
by sorry

end q_function_determination_l1995_199570


namespace conference_attendees_l1995_199579

theorem conference_attendees (total : ℕ) (creators : ℕ) (editors : ℕ) (y : ℕ) :
  total = 200 →
  creators = 80 →
  editors = 65 →
  total = creators + editors - y + 3 * y →
  y ≤ 27 ∧ ∃ (y : ℕ), y = 27 ∧ total = creators + editors - y + 3 * y :=
by sorry

end conference_attendees_l1995_199579


namespace simple_interest_principal_l1995_199549

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℝ) (rate_paise : ℝ) (time_months : ℝ) :
  interest = 23 * (rate_paise / 100) * time_months →
  interest = 3.45 ∧ rate_paise = 5 ∧ time_months = 3 →
  23 = interest / ((rate_paise / 100) * time_months) :=
by sorry

end simple_interest_principal_l1995_199549


namespace valid_paths_count_l1995_199548

/-- Represents the grid dimensions -/
structure GridDimensions where
  width : Nat
  height : Nat

/-- Represents a forbidden vertical segment -/
structure ForbiddenSegment where
  x : Nat
  y_start : Nat
  y_end : Nat

/-- Calculates the number of valid paths on the grid -/
def countValidPaths (grid : GridDimensions) (forbidden : List ForbiddenSegment) : Nat :=
  sorry

/-- The main theorem stating the number of valid paths -/
theorem valid_paths_count :
  let grid := GridDimensions.mk 10 4
  let forbidden := [
    ForbiddenSegment.mk 5 1 3,
    ForbiddenSegment.mk 6 1 3
  ]
  countValidPaths grid forbidden = 329 := by
  sorry

end valid_paths_count_l1995_199548


namespace equation_roots_imply_a_range_l1995_199547

theorem equation_roots_imply_a_range :
  ∀ a : ℝ, (∃ x : ℝ, (2 - 2^(-|x - 3|))^2 = 3 + a) → -2 ≤ a ∧ a < 1 := by
  sorry

end equation_roots_imply_a_range_l1995_199547


namespace remainder_of_B_divided_by_9_l1995_199556

theorem remainder_of_B_divided_by_9 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 := by
  sorry

end remainder_of_B_divided_by_9_l1995_199556


namespace two_digit_number_property_l1995_199583

theorem two_digit_number_property : 
  let n : ℕ := 27
  let tens_digit : ℕ := n / 10
  let units_digit : ℕ := n % 10
  (units_digit = tens_digit + 5) →
  (n * (tens_digit + units_digit) = 243) :=
by
  sorry

end two_digit_number_property_l1995_199583


namespace expression_equals_zero_l1995_199503

theorem expression_equals_zero : 
  |Real.sqrt 3 - 1| + (Real.pi - 3)^0 - Real.tan (Real.pi / 3) = 0 := by
  sorry

end expression_equals_zero_l1995_199503


namespace rectangle_hall_length_l1995_199538

theorem rectangle_hall_length :
  ∀ (length breadth : ℝ),
    length = breadth + 5 →
    length * breadth = 750 →
    length = 30 := by
  sorry

end rectangle_hall_length_l1995_199538


namespace abc_perfect_cube_l1995_199584

theorem abc_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (n : ℤ), (a / b : ℚ) + (b / c : ℚ) + (c / a : ℚ) = n) : 
  ∃ (k : ℤ), a * b * c = k^3 := by
  sorry

end abc_perfect_cube_l1995_199584


namespace sum_reciprocals_l1995_199577

theorem sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / b + b / c + c / a + b / a + c / b + a / c = 9) :
  a / b + b / c + c / a = 4.5 := by
  sorry

end sum_reciprocals_l1995_199577


namespace digit_sum_in_t_shape_l1995_199535

theorem digit_sum_in_t_shape : 
  ∀ (a b c d e f g : ℕ),
  a ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  b ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  c ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  d ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  e ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  f ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g →
  a + b + c = 23 →
  d + e + f + g = 12 →
  b = e →
  a + b + c + d + f + g = 29 := by
sorry

end digit_sum_in_t_shape_l1995_199535


namespace prime_condition_l1995_199559

theorem prime_condition (p : ℕ) : 
  Prime p → Prime (p^4 - 3*p^2 + 9) → p = 2 :=
by sorry

end prime_condition_l1995_199559


namespace restaurant_problem_l1995_199572

theorem restaurant_problem (initial_wings : ℕ) (additional_wings : ℕ) (wings_per_friend : ℕ) 
  (h1 : initial_wings = 9)
  (h2 : additional_wings = 7)
  (h3 : wings_per_friend = 4) :
  (initial_wings + additional_wings) / wings_per_friend = 4 :=
by sorry

end restaurant_problem_l1995_199572


namespace transaction_conservation_l1995_199590

/-- Represents the transaction in the restaurant problem -/
structure RestaurantTransaction where
  initial_payment : ℕ
  people : ℕ
  overcharge : ℕ
  refund_per_person : ℕ
  assistant_kept : ℕ

/-- The actual cost of the meal -/
def actual_cost (t : RestaurantTransaction) : ℕ :=
  t.initial_payment - t.overcharge

/-- The amount effectively paid by the customers -/
def effective_payment (t : RestaurantTransaction) : ℕ :=
  t.people * (t.initial_payment / t.people - t.refund_per_person)

/-- Theorem stating that the total amount involved is conserved -/
theorem transaction_conservation (t : RestaurantTransaction) 
  (h1 : t.initial_payment = 30)
  (h2 : t.people = 3)
  (h3 : t.overcharge = 5)
  (h4 : t.refund_per_person = 1)
  (h5 : t.assistant_kept = 2) :
  effective_payment t + (t.people * t.refund_per_person) + t.assistant_kept = t.initial_payment := by
  sorry

#check transaction_conservation

end transaction_conservation_l1995_199590


namespace vertical_line_slope_angle_l1995_199565

-- Define the line x + 2 = 0
def vertical_line (x : ℝ) : Prop := x + 2 = 0

-- Define the slope angle of a line
def slope_angle (line : ℝ → Prop) : ℝ := sorry

-- Theorem: The slope angle of the line x + 2 = 0 is π/2
theorem vertical_line_slope_angle :
  slope_angle vertical_line = π / 2 := by sorry

end vertical_line_slope_angle_l1995_199565


namespace expression_simplification_l1995_199561

theorem expression_simplification (a b x y : ℝ) (h : b*x + a*y ≠ 0) :
  (b*x*(a^2*x^2 + 2*a^2*y^2 + b^2*y^2) + a*y*(a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x + a*y)
  = (a*x + b*y)^2 := by
sorry

end expression_simplification_l1995_199561


namespace return_speed_calculation_l1995_199598

/-- Proves that given a round trip between two cities 150 miles apart,
    where the outbound speed is 50 mph and the average round trip speed is 60 mph,
    the return speed is 75 mph. -/
theorem return_speed_calculation (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) :
  distance = 150 →
  outbound_speed = 50 →
  average_speed = 60 →
  (2 * distance) / (distance / outbound_speed + distance / (2 * distance / average_speed - distance / outbound_speed)) = average_speed →
  2 * distance / average_speed - distance / outbound_speed = 75 := by
  sorry

end return_speed_calculation_l1995_199598


namespace cubic_expansion_sum_difference_l1995_199557

theorem cubic_expansion_sum_difference (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5*x + 4)^3 = a + a₁*x + a₂*x^2 + a₃*x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by sorry

end cubic_expansion_sum_difference_l1995_199557


namespace equation_roots_l1995_199519

theorem equation_roots : ∀ (x : ℝ), x * (x - 3)^2 * (5 + x) = 0 ↔ x ∈ ({0, 3, -5} : Set ℝ) := by
  sorry

end equation_roots_l1995_199519


namespace speaking_orders_count_l1995_199588

/-- The number of contestants -/
def n : ℕ := 6

/-- The number of positions where contestant A can speak -/
def a_positions : ℕ := n - 2

/-- The number of permutations for the remaining contestants -/
def remaining_permutations : ℕ := Nat.factorial (n - 1)

/-- The total number of different speaking orders -/
def total_orders : ℕ := a_positions * remaining_permutations

theorem speaking_orders_count : total_orders = 480 := by
  sorry

end speaking_orders_count_l1995_199588


namespace carpet_cost_l1995_199523

/-- The cost of carpeting a room with given dimensions and carpet specifications. -/
theorem carpet_cost (room_length room_width carpet_width carpet_cost : ℝ) :
  room_length = 13 ∧
  room_width = 9 ∧
  carpet_width = 0.75 ∧
  carpet_cost = 12 →
  room_length * room_width * carpet_cost = 1404 := by
  sorry

end carpet_cost_l1995_199523


namespace vasya_wins_in_four_moves_l1995_199533

-- Define a polynomial with integer coefficients
def IntPolynomial := ℤ → ℤ

-- Define a function that counts the number of integer solutions for P(x) = a
def countIntegerSolutions (P : IntPolynomial) (a : ℤ) : ℕ :=
  sorry

-- Theorem statement
theorem vasya_wins_in_four_moves :
  ∀ (P : IntPolynomial),
  ∃ (S : Finset ℤ),
  (Finset.card S ≤ 4) ∧
  ∃ (a b : ℤ),
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧
  countIntegerSolutions P a = countIntegerSolutions P b :=
sorry

end vasya_wins_in_four_moves_l1995_199533


namespace square_side_length_range_l1995_199550

theorem square_side_length_range (a : ℝ) : a^2 = 30 → 5.4 < a ∧ a < 5.5 := by
  sorry

end square_side_length_range_l1995_199550


namespace perfect_square_condition_l1995_199597

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 m 16 → m = 8 ∨ m = -8 := by
  sorry

end perfect_square_condition_l1995_199597


namespace product_of_repeating_decimals_l1995_199569

def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_7 : ℚ := 7/9

theorem product_of_repeating_decimals :
  repeating_decimal_4 * repeating_decimal_7 = 28/81 := by
  sorry

end product_of_repeating_decimals_l1995_199569


namespace cost_per_book_is_5_l1995_199553

/-- The cost to produce each book -/
def cost_per_book : ℝ := 5

/-- The selling price of each book -/
def selling_price : ℝ := 20

/-- The total profit -/
def total_profit : ℝ := 120

/-- The number of customers -/
def num_customers : ℕ := 4

/-- The number of books each customer buys -/
def books_per_customer : ℕ := 2

/-- The theorem stating the cost to make each book -/
theorem cost_per_book_is_5 : 
  cost_per_book = 5 :=
by sorry

end cost_per_book_is_5_l1995_199553


namespace cylinder_radius_l1995_199509

/-- 
Given a right circular cylinder with height h and diagonal d (measured from the center of the
circular base to the top edge of the cylinder), this theorem proves that when h = 12 and d = 13,
the radius r of the cylinder is 5.
-/
theorem cylinder_radius (h d : ℝ) (h_pos : h > 0) (d_pos : d > 0) 
  (h_val : h = 12) (d_val : d = 13) : ∃ r : ℝ, r > 0 ∧ r = 5 ∧ r^2 + h^2 = d^2 := by
  sorry

end cylinder_radius_l1995_199509


namespace complementary_angles_imply_right_triangle_l1995_199528

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the Triangle Angle Sum Theorem
axiom triangle_angle_sum (t : Triangle) : t.angle1 + t.angle2 + t.angle3 = 180

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) 
  (h : complementary t.angle1 t.angle2 ∨ complementary t.angle1 t.angle3 ∨ complementary t.angle2 t.angle3) : 
  is_right_triangle t :=
sorry

end complementary_angles_imply_right_triangle_l1995_199528


namespace class_size_l1995_199546

theorem class_size (average_age : ℝ) (new_average : ℝ) (student_leave_age : ℝ) (teacher_age : ℝ)
  (h1 : average_age = 10)
  (h2 : new_average = 11)
  (h3 : student_leave_age = 11)
  (h4 : teacher_age = 41) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * average_age = (n : ℝ) * new_average - teacher_age + student_leave_age :=
by
  sorry

end class_size_l1995_199546


namespace katies_sister_candy_l1995_199575

theorem katies_sister_candy (katie_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  katie_candy = 10 →
  eaten_candy = 9 →
  remaining_candy = 7 →
  ∃ sister_candy : ℕ, sister_candy = 6 ∧ katie_candy + sister_candy = eaten_candy + remaining_candy :=
by sorry

end katies_sister_candy_l1995_199575


namespace tenths_minus_hundredths_l1995_199592

theorem tenths_minus_hundredths : (0.5 : ℝ) - (0.05 : ℝ) = 0.45 := by
  sorry

end tenths_minus_hundredths_l1995_199592


namespace factor_expression_l1995_199505

theorem factor_expression (x : ℝ) : 60 * x + 45 + 9 * x^2 = 3 * (3 * x + 5) * (x + 3) := by
  sorry

end factor_expression_l1995_199505


namespace trapezoid_segment_length_l1995_199541

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  base_diff : ℝ
  midpoint_ratio : ℝ × ℝ
  equal_area_segment : ℝ

/-- The trapezoid satisfying the problem conditions -/
def problem_trapezoid : Trapezoid where
  base_diff := 120
  midpoint_ratio := (3, 4)
  equal_area_segment := x
  where x : ℝ := sorry  -- The actual value of x will be determined in the proof

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  t = problem_trapezoid → ⌊(t.equal_area_segment^2) / 120⌋ = 270 := by
  sorry

end trapezoid_segment_length_l1995_199541


namespace sector_to_cone_l1995_199518

/-- Given a 300° sector of a circle with radius 12, prove it forms a cone with base radius 10 and slant height 12 -/
theorem sector_to_cone (r : ℝ) (angle : ℝ) :
  r = 12 →
  angle = 300 * (π / 180) →
  ∃ (base_radius slant_height : ℝ),
    base_radius = 10 ∧
    slant_height = r ∧
    2 * π * base_radius = angle * r :=
by sorry

end sector_to_cone_l1995_199518


namespace solution_for_E_l1995_199593

/-- The function E as defined in the problem -/
def E (a b c : ℚ) : ℚ := a * b^2 + c

/-- Theorem stating that -1/10 is the solution to E(a,4,5) = E(a,6,7) -/
theorem solution_for_E : 
  ∃ a : ℚ, E a 4 5 = E a 6 7 ∧ a = -1/10 := by
  sorry

end solution_for_E_l1995_199593


namespace unique_root_P_l1995_199576

-- Define the polynomial sequence
def P : ℕ → ℝ → ℝ
  | 0, x => 0
  | 1, x => x
  | (n+2), x => x * P (n+1) x + (1 - x) * P n x

-- State the theorem
theorem unique_root_P (n : ℕ) (hn : n ≥ 1) : 
  ∀ x : ℝ, P n x = 0 ↔ x = 0 := by sorry

end unique_root_P_l1995_199576


namespace person_c_start_time_l1995_199578

/-- Represents a point on the line AB -/
inductive Point : Type
| A : Point
| C : Point
| D : Point
| B : Point

/-- Represents a person walking on the line AB -/
structure Person where
  name : String
  startTime : Nat
  startPoint : Point
  endPoint : Point
  speed : Nat

/-- Represents the problem setup -/
structure ProblemSetup where
  personA : Person
  personB : Person
  personC : Person
  meetingTimeAB : Nat
  meetingTimeAC : Nat

/-- The theorem to prove -/
theorem person_c_start_time (setup : ProblemSetup) : setup.personC.startTime = 16 :=
  by sorry

end person_c_start_time_l1995_199578


namespace probability_same_group_is_one_fourth_l1995_199515

def number_of_groups : ℕ := 4

def probability_same_group : ℚ :=
  (number_of_groups : ℚ) / ((number_of_groups : ℚ) * (number_of_groups : ℚ))

theorem probability_same_group_is_one_fourth :
  probability_same_group = 1/4 := by
  sorry

end probability_same_group_is_one_fourth_l1995_199515


namespace gray_eyed_black_haired_count_l1995_199585

/-- Represents the characteristics of models in a modeling agency. -/
structure ModelAgency where
  total : Nat
  redHaired : Nat
  blackHaired : Nat
  greenEyed : Nat
  grayEyed : Nat
  greenEyedRedHaired : Nat

/-- Conditions for the modeling agency problem. -/
def modelingAgencyConditions : ModelAgency :=
  { total := 60
  , redHaired := 24  -- Derived from total - blackHaired
  , blackHaired := 36
  , greenEyed := 36  -- Derived from total - grayEyed
  , grayEyed := 24
  , greenEyedRedHaired := 22 }

/-- Theorem stating that the number of gray-eyed black-haired models is 10. -/
theorem gray_eyed_black_haired_count (agency : ModelAgency) 
  (h1 : agency = modelingAgencyConditions) : 
  agency.blackHaired + agency.greenEyedRedHaired - agency.greenEyed = 10 := by
  sorry

#check gray_eyed_black_haired_count

end gray_eyed_black_haired_count_l1995_199585


namespace value_of_b_l1995_199581

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end value_of_b_l1995_199581


namespace function_decreasing_interval_l1995_199529

/-- Given a function f(x) = kx³ + 3(k-1)x² - k² + 1 where k > 0,
    and f(x) is decreasing in the interval (0,4),
    prove that k = 4. -/
theorem function_decreasing_interval (k : ℝ) (h1 : k > 0) : 
  (∀ x ∈ Set.Ioo 0 4, 
    (deriv (fun x => k*x^3 + 3*(k-1)*x^2 - k^2 + 1) x) < 0) → 
  k = 4 := by
sorry

end function_decreasing_interval_l1995_199529


namespace right_triangle_sin_identity_l1995_199504

theorem right_triangle_sin_identity (A B C : Real) (h1 : C = Real.pi / 2) (h2 : A + B = Real.pi / 2) :
  Real.sin A * Real.sin B * Real.sin (A - B) + 
  Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + 
  Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
  sorry

end right_triangle_sin_identity_l1995_199504


namespace binomial_100_97_l1995_199531

theorem binomial_100_97 : Nat.choose 100 97 = 161700 := by
  sorry

end binomial_100_97_l1995_199531


namespace jan_extra_miles_l1995_199532

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  ian_speed : ℝ
  ian_time : ℝ
  han_speed : ℝ
  han_time : ℝ
  jan_speed : ℝ
  jan_time : ℝ

/-- The conditions of the driving scenario -/
def scenario_conditions (s : DrivingScenario) : Prop :=
  s.han_speed = s.ian_speed + 10 ∧
  s.han_time = s.ian_time ∧
  s.jan_time = s.ian_time + 3 ∧
  s.jan_speed = s.ian_speed + 15 ∧
  s.han_speed * s.han_time = s.ian_speed * s.ian_time + 90

/-- The theorem to be proved -/
theorem jan_extra_miles (s : DrivingScenario) :
  scenario_conditions s →
  s.jan_speed * s.jan_time - s.ian_speed * s.ian_time = 210 := by
  sorry


end jan_extra_miles_l1995_199532


namespace six_by_six_grid_shaded_percentage_l1995_199507

/-- Represents a square grid --/
structure SquareGrid :=
  (side : ℕ)
  (total_squares : ℕ)
  (shaded_squares : ℕ)

/-- Calculates the percentage of shaded area in a square grid --/
def shaded_percentage (grid : SquareGrid) : ℚ :=
  (grid.shaded_squares : ℚ) / (grid.total_squares : ℚ)

theorem six_by_six_grid_shaded_percentage :
  let grid : SquareGrid := ⟨6, 36, 21⟩
  shaded_percentage grid = 7 / 12 := by
  sorry

#eval (7 : ℚ) / 12 * 100  -- To show the decimal representation

end six_by_six_grid_shaded_percentage_l1995_199507


namespace wall_length_is_800_l1995_199595

-- Define the dimensions of a single brick
def brick_length : ℝ := 100
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the wall dimensions
def wall_height : ℝ := 600 -- 6 m converted to cm
def wall_width : ℝ := 22.5

-- Define the number of bricks
def num_bricks : ℕ := 1600

-- Define the volume of a single brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Define the total volume of all bricks
def total_brick_volume : ℝ := brick_volume * num_bricks

-- Theorem stating the length of the wall
theorem wall_length_is_800 : 
  ∃ (wall_length : ℝ), wall_length * wall_height * wall_width = total_brick_volume ∧ wall_length = 800 := by
sorry

end wall_length_is_800_l1995_199595


namespace binomial_expansion_simplification_l1995_199551

theorem binomial_expansion_simplification (x : ℝ) : 
  (2*x+1)^5 - 5*(2*x+1)^4 + 10*(2*x+1)^3 - 10*(2*x+1)^2 + 5*(2*x+1) - 1 = 32*x^5 := by
  sorry

end binomial_expansion_simplification_l1995_199551


namespace cube_volume_problem_l1995_199539

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 8 → 
  a^3 = 8 := by
sorry

end cube_volume_problem_l1995_199539


namespace factory_equation_holds_l1995_199558

/-- Represents a factory's part processing scenario -/
def factory_scenario (x : ℝ) : Prop :=
  x > 0 ∧ 
  (100 / x) + (400 / (2 * x)) = 6

/-- Theorem stating the equation holds for the given scenario -/
theorem factory_equation_holds : 
  ∀ x : ℝ, x > 0 → (100 / x) + (400 / (2 * x)) = 6 ↔ factory_scenario x :=
by
  sorry

#check factory_equation_holds

end factory_equation_holds_l1995_199558


namespace complex_equation_solution_l1995_199594

theorem complex_equation_solution (z : ℂ) :
  z * (Complex.I - Complex.I^2) = 1 + Complex.I^3 → z = -Complex.I := by
  sorry

end complex_equation_solution_l1995_199594


namespace circle_line_distance_l1995_199514

theorem circle_line_distance (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + 1 = 0) → 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 4) →
  (|a + 1| / Real.sqrt (a^2 + 1) = 1) →
  a = 0 := by
sorry

end circle_line_distance_l1995_199514


namespace fred_cards_l1995_199555

theorem fred_cards (initial_cards torn_cards bought_cards total_cards : ℕ) : 
  initial_cards = 18 →
  torn_cards = 8 →
  bought_cards = 40 →
  total_cards = 84 →
  total_cards = initial_cards - torn_cards + bought_cards + (total_cards - (initial_cards - torn_cards + bought_cards)) →
  total_cards - (initial_cards - torn_cards + bought_cards) = 34 := by
  sorry

end fred_cards_l1995_199555


namespace circle_center_point_is_center_l1995_199508

/-- The center of a circle given by the equation x^2 + 4x + y^2 - 6y = 24 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 4*x + y^2 - 6*y = 24) ↔ ((x + 2)^2 + (y - 3)^2 = 37) :=
by sorry

/-- The point (-2, 3) is the center of the circle -/
theorem point_is_center : 
  ∃! (a b : ℝ), ∀ (x y : ℝ), (x^2 + 4*x + y^2 - 6*y = 24) ↔ ((x - a)^2 + (y - b)^2 = 37) ∧ 
  a = -2 ∧ b = 3 :=
by sorry

end circle_center_point_is_center_l1995_199508


namespace proposition_relationship_l1995_199513

-- Define propositions as variables of type Prop
variable (A B C : Prop)

-- Define the relationships between propositions
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary_and_sufficient (P Q : Prop) : Prop :=
  (P ↔ Q)

def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

-- State the theorem
theorem proposition_relationship :
  sufficient_not_necessary A B →
  necessary_and_sufficient B C →
  necessary_not_sufficient C A :=
by sorry

end proposition_relationship_l1995_199513


namespace number_order_l1995_199526

theorem number_order : 
  (1 * 4^3) < (8 * 9 + 5) ∧ (8 * 9 + 5) < (2 * 6^2 + 1 * 6 + 0) := by
  sorry

end number_order_l1995_199526


namespace not_right_triangle_11_12_15_l1995_199511

/-- A function that checks if three numbers can form a right triangle -/
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- Theorem stating that 11, 12, and 15 cannot form a right triangle -/
theorem not_right_triangle_11_12_15 : ¬ isRightTriangle 11 12 15 := by
  sorry

#check not_right_triangle_11_12_15

end not_right_triangle_11_12_15_l1995_199511


namespace fish_catching_ratio_l1995_199525

/-- The number of fish Blaine caught -/
def blaine_fish : ℕ := 5

/-- The total number of fish caught by Keith and Blaine -/
def total_fish : ℕ := 15

/-- The number of fish Keith caught -/
def keith_fish : ℕ := total_fish - blaine_fish

/-- The ratio of fish Keith caught to fish Blaine caught -/
def fish_ratio : ℚ := keith_fish / blaine_fish

theorem fish_catching_ratio :
  fish_ratio = 2 / 1 := by
  sorry

end fish_catching_ratio_l1995_199525


namespace fliers_remaining_l1995_199521

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h1 : total = 2500)
  (h2 : morning_fraction = 1 / 5)
  (h3 : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * total).floor - (afternoon_fraction * (total - (morning_fraction * total).floor)).floor = 1500 :=
by sorry

end fliers_remaining_l1995_199521


namespace vat_volume_l1995_199512

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The volume of juice in each glass (in pints) -/
def volume_per_glass : ℕ := 30

/-- Theorem: The total volume of orange juice in the vat is 150 pints -/
theorem vat_volume : num_glasses * volume_per_glass = 150 := by
  sorry

end vat_volume_l1995_199512


namespace math_books_same_box_probability_l1995_199566

def total_textbooks : ℕ := 12
def math_textbooks : ℕ := 3
def box_capacities : List ℕ := [3, 4, 5]

def probability_all_math_in_same_box : ℚ :=
  3 / 44

theorem math_books_same_box_probability :
  probability_all_math_in_same_box = 3 / 44 :=
by sorry

end math_books_same_box_probability_l1995_199566


namespace tea_mixture_price_l1995_199524

/-- Given two types of tea with different prices per kg, calculate the price per kg of their mixture when mixed in equal quantities. -/
theorem tea_mixture_price (price_a price_b : ℚ) (h1 : price_a = 65) (h2 : price_b = 70) :
  (price_a + price_b) / 2 = 67.5 := by
  sorry

#check tea_mixture_price

end tea_mixture_price_l1995_199524


namespace horner_rule_V₂_l1995_199552

def f (x : ℝ) : ℝ := 2*x^6 + 3*x^5 + 5*x^3 + 6*x^2 + 7*x + 8

def V₂ (x : ℝ) : ℝ := 2

def V₁ (x : ℝ) : ℝ := V₂ x * x + 3

def V₂_final (x : ℝ) : ℝ := V₁ x * x + 0

theorem horner_rule_V₂ : V₂_final 2 = 14 := by
  sorry

end horner_rule_V₂_l1995_199552


namespace function_relationship_l1995_199510

/-- Given that y-m is directly proportional to 3x+6, where m is a constant,
    and that when x=2, y=4 and when x=3, y=7,
    prove that the function relationship between y and x is y = 3x - 2 -/
theorem function_relationship (m : ℝ) (k : ℝ) :
  (∀ x y, y - m = k * (3 * x + 6)) →
  (4 - m = k * (3 * 2 + 6)) →
  (7 - m = k * (3 * 3 + 6)) →
  ∀ x y, y = 3 * x - 2 := by
sorry


end function_relationship_l1995_199510


namespace y_work_time_l1995_199563

/-- The time it takes for y to complete the work alone, given the conditions -/
def time_y_alone (time_x time_yz time_xz : ℝ) : ℝ :=
  24

/-- Theorem stating that y takes 24 hours to complete the work alone -/
theorem y_work_time (time_x time_yz time_xz : ℝ) 
  (hx : time_x = 8) 
  (hyz : time_yz = 6) 
  (hxz : time_xz = 4) : 
  time_y_alone time_x time_yz time_xz = 24 := by
  sorry

#check y_work_time

end y_work_time_l1995_199563


namespace linear_function_point_relation_l1995_199571

/-- Given a linear function f(x) = -x + b, prove that if P₁(-1, y₁) and P₂(2, y₂) 
    are points on the graph of f, then y₁ > y₂ -/
theorem linear_function_point_relation (b : ℝ) (y₁ y₂ : ℝ) 
    (h₁ : y₁ = -(-1) + b) 
    (h₂ : y₂ = -(2) + b) : 
  y₁ > y₂ := by
  sorry

#check linear_function_point_relation

end linear_function_point_relation_l1995_199571


namespace floor_sum_of_positive_reals_l1995_199564

theorem floor_sum_of_positive_reals (u v w x : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x)
  (h1 : u^2 + v^2 = 3005) (h2 : w^2 + x^2 = 3005)
  (h3 : u * w = 1729) (h4 : v * x = 1729) : 
  ⌊u + v + w + x⌋ = 155 := by
sorry

end floor_sum_of_positive_reals_l1995_199564


namespace abs_sum_inequality_l1995_199580

theorem abs_sum_inequality (a b c d : ℝ) 
  (sum_pos : a + b + c + d > 0)
  (a_gt_c : a > c)
  (b_gt_d : b > d) :
  |a + b| > |c + d| := by sorry

end abs_sum_inequality_l1995_199580
