import Mathlib

namespace unique_number_with_special_divisors_l1578_157829

def has_twelve_divisors (N : ℕ) : Prop :=
  ∃ (d : Fin 12 → ℕ), 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ N) ∧
    (∀ m, m ∣ N → ∃ i, d i = m) ∧
    d 0 = 1 ∧ d 11 = N

theorem unique_number_with_special_divisors :
  ∃! N : ℕ, has_twelve_divisors N ∧
    ∃ (d : Fin 12 → ℕ), 
      (∀ i j, i < j → d i < d j) ∧
      (∀ i, d i ∣ N) ∧
      (d 0 = 1) ∧
      (d (d 3 - 2) = (d 0 + d 1 + d 3) * d 7) ∧
      N = 1989 :=
by sorry

end unique_number_with_special_divisors_l1578_157829


namespace strawberry_sale_revenue_difference_l1578_157839

/-- Represents the sale of strawberries at a supermarket -/
structure StrawberrySale where
  pints_sold : ℕ
  sale_revenue : ℕ
  price_difference : ℕ

/-- Calculates the revenue difference between regular and sale prices -/
def revenue_difference (sale : StrawberrySale) : ℕ :=
  let sale_price := sale.sale_revenue / sale.pints_sold
  let regular_price := sale_price + sale.price_difference
  regular_price * sale.pints_sold - sale.sale_revenue

/-- Theorem stating the revenue difference for the given scenario -/
theorem strawberry_sale_revenue_difference :
  ∃ (sale : StrawberrySale),
    sale.pints_sold = 54 ∧
    sale.sale_revenue = 216 ∧
    sale.price_difference = 2 ∧
    revenue_difference sale = 108 := by
  sorry

end strawberry_sale_revenue_difference_l1578_157839


namespace polynomial_roots_l1578_157858

def f (x : ℝ) : ℝ := 2*x^4 - 5*x^3 - 7*x^2 + 34*x - 24

theorem polynomial_roots :
  (f 1 = 0) ∧
  (∀ x : ℝ, f x = 0 ∧ x ≠ 1 → 2*x^3 - 3*x^2 - 12*x + 10 = 0) :=
by sorry

end polynomial_roots_l1578_157858


namespace complex_fraction_simplification_l1578_157842

theorem complex_fraction_simplification :
  let z : ℂ := (4 - 9*I) / (3 + 4*I)
  z = -24/25 - 43/25*I :=
by sorry

end complex_fraction_simplification_l1578_157842


namespace distance_bound_l1578_157860

/-- Given two points A and B, and their distances to a third point (school),
    prove that the distance between A and B is bounded. -/
theorem distance_bound (dist_A_school dist_B_school d : ℝ) : 
  dist_A_school = 5 →
  dist_B_school = 2 →
  3 ≤ d ∧ d ≤ 7 :=
by
  sorry

#check distance_bound

end distance_bound_l1578_157860


namespace squirrel_count_l1578_157865

theorem squirrel_count (first_count : ℕ) (second_count : ℕ) : 
  first_count = 12 → 
  second_count = first_count + first_count / 3 → 
  first_count + second_count = 28 :=
by
  sorry

end squirrel_count_l1578_157865


namespace power_equality_l1578_157840

theorem power_equality : (8 : ℕ) ^ 8 = (4 : ℕ) ^ 12 ∧ (8 : ℕ) ^ 8 = (2 : ℕ) ^ 24 := by
  sorry

end power_equality_l1578_157840


namespace seven_numbers_even_sum_after_removal_l1578_157859

theorem seven_numbers_even_sum_after_removal (S : Finset ℕ) (h : S.card = 7) :
  ∃ x ∈ S, Even (S.sum id - x) := by
  sorry

end seven_numbers_even_sum_after_removal_l1578_157859


namespace expand_and_simplify_l1578_157887

theorem expand_and_simplify (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_and_simplify_l1578_157887


namespace curve_translation_l1578_157837

-- Define the original curve
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

-- Define the translated curve
def translated_curve (x y : ℝ) : Prop :=
  (y + 1) * Real.sin x + 2 * y + 1 = 0

-- Theorem statement
theorem curve_translation :
  ∀ x y : ℝ, original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
by sorry

end curve_translation_l1578_157837


namespace nonagon_diagonals_l1578_157820

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by sorry

end nonagon_diagonals_l1578_157820


namespace train_length_l1578_157827

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 48 → time_s = 9 → ∃ length_m : ℝ, abs (length_m - 119.97) < 0.01 := by
  sorry

end train_length_l1578_157827


namespace gcd_factorial_bound_l1578_157869

theorem gcd_factorial_bound (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p > q) :
  Nat.gcd (Nat.factorial p - 1) (Nat.factorial q - 1) ≤ p^(5/3) := by
  sorry

end gcd_factorial_bound_l1578_157869


namespace enrollment_increase_l1578_157862

/-- Calculate the total enrollment and percent increase from 1991 to 1995 --/
theorem enrollment_increase (dept_a_1991 dept_b_1991 : ℝ) 
  (increase_a_1992 increase_b_1992 : ℝ)
  (increase_a_1993 increase_b_1993 : ℝ)
  (increase_1994 : ℝ)
  (decrease_1994 : ℝ)
  (campus_c_1994 : ℝ)
  (increase_c_1995 : ℝ) :
  dept_a_1991 = 2000 →
  dept_b_1991 = 1000 →
  increase_a_1992 = 0.25 →
  increase_b_1992 = 0.10 →
  increase_a_1993 = 0.15 →
  increase_b_1993 = 0.20 →
  increase_1994 = 0.10 →
  decrease_1994 = 0.05 →
  campus_c_1994 = 300 →
  increase_c_1995 = 0.50 →
  let dept_a_1995 := dept_a_1991 * (1 + increase_a_1992) * (1 + increase_a_1993) * (1 + increase_1994) * (1 - decrease_1994)
  let dept_b_1995 := dept_b_1991 * (1 + increase_b_1992) * (1 + increase_b_1993) * (1 + increase_1994) * (1 - decrease_1994)
  let campus_c_1995 := campus_c_1994 * (1 + increase_c_1995)
  let total_1995 := dept_a_1995 + dept_b_1995 + campus_c_1995
  let total_1991 := dept_a_1991 + dept_b_1991
  let percent_increase := (total_1995 - total_1991) / total_1991 * 100
  total_1995 = 4833.775 ∧ percent_increase = 61.1258333 := by
  sorry

end enrollment_increase_l1578_157862


namespace blue_marbles_count_l1578_157812

theorem blue_marbles_count (red blue : ℕ) : 
  red + blue = 6000 →
  (red + blue) - (blue - red) = 4800 →
  blue > red →
  blue = 3600 := by
sorry

end blue_marbles_count_l1578_157812


namespace card_deck_size_l1578_157810

theorem card_deck_size (n : ℕ) (h1 : n ≥ 6) 
  (h2 : Nat.choose n 6 = 6 * Nat.choose n 3) : n = 13 :=
by
  sorry

end card_deck_size_l1578_157810


namespace half_recipe_flour_l1578_157817

-- Define the original amount of flour in the recipe
def original_flour : ℚ := 4 + 1/2

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/2

-- Theorem to prove
theorem half_recipe_flour :
  recipe_fraction * original_flour = 2 + 1/4 :=
by sorry

end half_recipe_flour_l1578_157817


namespace initial_trees_count_l1578_157873

/-- The number of dogwood trees initially in the park -/
def initial_trees : ℕ := sorry

/-- The number of trees planted today -/
def trees_planted_today : ℕ := 5

/-- The number of trees planted tomorrow -/
def trees_planted_tomorrow : ℕ := 4

/-- The total number of trees after planting -/
def final_trees : ℕ := 16

/-- The number of workers who finished the work -/
def num_workers : ℕ := 8

theorem initial_trees_count : 
  initial_trees = final_trees - (trees_planted_today + trees_planted_tomorrow) :=
by sorry

end initial_trees_count_l1578_157873


namespace inequality_system_solution_l1578_157845

theorem inequality_system_solution :
  ∀ x : ℝ,
  (x + 1 > 7 - 2*x ∧ x ≤ (4 + 2*x) / 3) ↔ (2 < x ∧ x ≤ 4) :=
by sorry

end inequality_system_solution_l1578_157845


namespace marble_difference_l1578_157889

theorem marble_difference (total_yellow : ℕ) (jar1_red_ratio jar1_yellow_ratio jar2_red_ratio jar2_yellow_ratio : ℕ) :
  total_yellow = 140 →
  jar1_red_ratio = 7 →
  jar1_yellow_ratio = 3 →
  jar2_red_ratio = 3 →
  jar2_yellow_ratio = 2 →
  ∃ (jar1_total jar2_total : ℕ),
    jar1_total = jar2_total ∧
    jar1_total * jar1_yellow_ratio / (jar1_red_ratio + jar1_yellow_ratio) +
    jar2_total * jar2_yellow_ratio / (jar2_red_ratio + jar2_yellow_ratio) = total_yellow ∧
    jar1_total * jar1_red_ratio / (jar1_red_ratio + jar1_yellow_ratio) -
    jar2_total * jar2_red_ratio / (jar2_red_ratio + jar2_yellow_ratio) = 20 :=
by sorry

end marble_difference_l1578_157889


namespace circle_equation_minus_one_two_radius_two_l1578_157894

/-- The standard equation of a circle with center (h, k) and radius r -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The standard equation of a circle with center (-1, 2) and radius 2 -/
theorem circle_equation_minus_one_two_radius_two :
  ∀ x y : ℝ, standard_circle_equation x y (-1) 2 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end circle_equation_minus_one_two_radius_two_l1578_157894


namespace diggers_holes_problem_l1578_157880

/-- Given that three diggers dug three holes in three hours,
    prove that six diggers will dig 10 holes in five hours. -/
theorem diggers_holes_problem (diggers_rate : ℚ) : 
  (diggers_rate = 3 / (3 * 3)) →  -- Rate of digging holes per digger per hour
  (6 * diggers_rate * 5 : ℚ) = 10 := by
  sorry

end diggers_holes_problem_l1578_157880


namespace community_service_arrangements_l1578_157813

def volunteers : ℕ := 8
def service_days : ℕ := 5

def arrangements (n m : ℕ) : ℕ := sorry

theorem community_service_arrangements :
  let total_arrangements := 
    (arrangements 2 1 * arrangements 6 4 * arrangements 5 5) + 
    (arrangements 2 2 * arrangements 6 3 * arrangements 4 2)
  total_arrangements = 5040 := by sorry

end community_service_arrangements_l1578_157813


namespace wine_exchange_equation_l1578_157892

/-- Represents the value of clear wine in terms of grain -/
def clear_wine_value : ℝ := 10

/-- Represents the value of turbid wine in terms of grain -/
def turbid_wine_value : ℝ := 3

/-- Represents the total amount of grain used -/
def total_grain : ℝ := 30

/-- Represents the total amount of wine obtained -/
def total_wine : ℝ := 5

/-- Proves that the equation 10x + 3(5-x) = 30 correctly represents the problem -/
theorem wine_exchange_equation (x : ℝ) : 
  x ≥ 0 ∧ x ≤ total_wine → 
  clear_wine_value * x + turbid_wine_value * (total_wine - x) = total_grain := by
sorry

end wine_exchange_equation_l1578_157892


namespace robotics_camp_age_problem_l1578_157885

theorem robotics_camp_age_problem (total_members : ℕ) (girls : ℕ) (boys : ℕ) (adults : ℕ)
  (overall_avg : ℚ) (girls_avg : ℚ) (boys_avg : ℚ) :
  total_members = 60 →
  girls = 30 →
  boys = 20 →
  adults = 10 →
  overall_avg = 18 →
  girls_avg = 16 →
  boys_avg = 17 →
  (total_members * overall_avg - girls * girls_avg - boys * boys_avg) / adults = 26 :=
by sorry

end robotics_camp_age_problem_l1578_157885


namespace school_students_count_l1578_157871

theorem school_students_count (girls : ℕ) (boys : ℕ) (total : ℕ) : 
  girls = 160 → 
  girls * 8 = boys * 5 → 
  total = girls + boys → 
  total = 416 := by
sorry

end school_students_count_l1578_157871


namespace savings_percentage_l1578_157843

/-- Proves that a person saves 20% of their salary given specific conditions -/
theorem savings_percentage (salary : ℝ) (savings_after_increase : ℝ) 
  (h1 : salary = 6500)
  (h2 : savings_after_increase = 260)
  (h3 : ∃ (original_expenses : ℝ), 
    salary = original_expenses + (salary * 0.2) 
    ∧ savings_after_increase = salary - (original_expenses * 1.2)) :
  (salary - savings_after_increase) / salary * 100 = 80 := by
  sorry

end savings_percentage_l1578_157843


namespace unique_obtuse_consecutive_triangle_l1578_157884

/-- A triangle with consecutive natural number side lengths is obtuse if and only if 
    the square of the longest side is greater than the sum of squares of the other two sides. -/
def IsObtuseConsecutiveTriangle (x : ℕ) : Prop :=
  (x + 2)^2 > x^2 + (x + 1)^2

/-- There exists exactly one obtuse triangle with consecutive natural number side lengths. -/
theorem unique_obtuse_consecutive_triangle :
  ∃! x : ℕ, IsObtuseConsecutiveTriangle x ∧ x > 0 := by
  sorry

end unique_obtuse_consecutive_triangle_l1578_157884


namespace open_book_is_random_event_l1578_157806

/-- Represents the possible classifications of events --/
inductive EventType
  | Certain
  | Random
  | Impossible
  | Determined

/-- Represents a book --/
structure Book where
  grade : Nat
  subject : String
  publisher : String

/-- Represents the event of opening a book to a specific page --/
structure OpenBookEvent where
  book : Book
  page : Nat
  intentional : Bool

/-- Definition of a certain event --/
def is_certain_event (e : OpenBookEvent) : Prop :=
  e.intentional ∧ ∀ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of a random event --/
def is_random_event (e : OpenBookEvent) : Prop :=
  ¬e.intentional ∧ ∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of an impossible event --/
def is_impossible_event (e : OpenBookEvent) : Prop :=
  ¬∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- Definition of a determined event --/
def is_determined_event (e : OpenBookEvent) : Prop :=
  e.intentional ∧ ∃ (b : Book) (p : Nat), e.book = b ∧ e.page = p

/-- The main theorem to prove --/
theorem open_book_is_random_event (e : OpenBookEvent) 
  (h1 : e.book.grade = 9)
  (h2 : e.book.subject = "mathematics")
  (h3 : e.book.publisher = "East China Normal University")
  (h4 : e.page = 50)
  (h5 : ¬e.intentional) :
  is_random_event e :=
sorry

end open_book_is_random_event_l1578_157806


namespace four_digit_divisible_by_9_l1578_157888

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

theorem four_digit_divisible_by_9 (A : ℕ) (h1 : digit A) (h2 : is_divisible_by_9 (3000 + 100 * A + 10 * A + 1)) :
  A = 7 := by sorry

end four_digit_divisible_by_9_l1578_157888


namespace seventh_term_of_geometric_sequence_l1578_157801

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 1 = 3)
  (h_sixth : a 6 = 972) :
  a 7 = 2187 := by
sorry

end seventh_term_of_geometric_sequence_l1578_157801


namespace initial_quarters_l1578_157897

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- The total value of coins in cents -/
def total_value (dimes nickels quarters : ℕ) : ℕ :=
  dimes * coin_value "dime" + nickels * coin_value "nickel" + quarters * coin_value "quarter"

theorem initial_quarters (initial_dimes initial_nickels mom_quarters : ℕ) 
  (total_cents : ℕ) (h1 : initial_dimes = 4) (h2 : initial_nickels = 7) 
  (h3 : mom_quarters = 5) (h4 : total_cents = 300) :
  ∃ initial_quarters : ℕ, 
    total_value initial_dimes initial_nickels (initial_quarters + mom_quarters) = total_cents ∧ 
    initial_quarters = 4 := by
  sorry

end initial_quarters_l1578_157897


namespace sailboat_rental_cost_l1578_157855

/-- The cost to rent a sailboat for 3 hours a day over 2 days -/
def sailboat_cost : ℝ := sorry

/-- The cost per hour to rent a ski boat -/
def ski_boat_cost_per_hour : ℝ := 80

/-- The number of hours per day the boats were rented -/
def hours_per_day : ℕ := 3

/-- The number of days the boats were rented -/
def days_rented : ℕ := 2

/-- The additional cost Aldrich paid for the ski boat compared to Ken's sailboat -/
def additional_cost : ℝ := 120

theorem sailboat_rental_cost :
  sailboat_cost = 360 :=
by
  have ski_boat_total_cost : ℝ := ski_boat_cost_per_hour * (hours_per_day * days_rented)
  have h1 : ski_boat_total_cost = sailboat_cost + additional_cost := by sorry
  sorry

end sailboat_rental_cost_l1578_157855


namespace area_of_quadrilateral_l1578_157851

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 16 + y^2 / 4 = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral (hP : P ∈ C) (hQ : Q ∈ C) 
  (hSymmetric : Q = (-P.1, -P.2)) (hDistance : ‖P - Q‖ = ‖F₁ - F₂‖) :
  ‖P - F₁‖ * ‖P - F₂‖ = 8 := by sorry

end area_of_quadrilateral_l1578_157851


namespace arithmetic_expression_equality_l1578_157890

theorem arithmetic_expression_equality : 1000 + 200 - 10 + 1 = 1191 := by
  sorry

end arithmetic_expression_equality_l1578_157890


namespace no_such_hexagon_exists_l1578_157825

-- Define a hexagon as a collection of 6 points in 2D space
def Hexagon := (Fin 6 → ℝ × ℝ)

-- Define a predicate for convexity
def is_convex (h : Hexagon) : Prop := sorry

-- Define a predicate for a point being inside a hexagon
def is_inside (p : ℝ × ℝ) (h : Hexagon) : Prop := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the length of a hexagon side
def side_length (h : Hexagon) (i : Fin 6) : ℝ := 
  distance (h i) (h ((i + 1) % 6))

-- Theorem statement
theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    is_inside m h ∧
    (∀ i : Fin 6, side_length h i > 1) ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
sorry

end no_such_hexagon_exists_l1578_157825


namespace avalon_quest_probability_l1578_157822

theorem avalon_quest_probability :
  let total_players : ℕ := 10
  let bad_players : ℕ := 4
  let quest_size : ℕ := 3
  let good_players : ℕ := total_players - bad_players
  let total_quests : ℕ := Nat.choose total_players quest_size
  let failed_quests : ℕ := total_quests - Nat.choose good_players quest_size
  let one_bad_quests : ℕ := Nat.choose bad_players 1 * Nat.choose good_players (quest_size - 1)
  (failed_quests > 0) →
  (one_bad_quests : ℚ) / failed_quests = 3 / 5 :=
by sorry

end avalon_quest_probability_l1578_157822


namespace expression_factorization_l1578_157866

theorem expression_factorization (a b c : ℝ) (h : (a - b) + (b - c) + (c - a) ≠ 0) :
  ((a - b)^2 + (b - c)^2 + (c - a)^2) / ((a - b) + (b - c) + (c - a)) = a - b + b - c + c - a :=
by sorry

end expression_factorization_l1578_157866


namespace correct_guess_probability_l1578_157872

/-- Represents a digit in the combination lock --/
def Digit := Fin 10

/-- Represents a three-digit combination --/
structure Combination where
  first : Digit
  second : Digit
  third : Digit

/-- The probability of guessing the correct last digit --/
def probability_guess_last_digit : ℚ := 1 / 10

/-- Theorem stating that the probability of guessing the last digit correctly is 1/10 --/
theorem correct_guess_probability :
  probability_guess_last_digit = 1 / 10 := by sorry

end correct_guess_probability_l1578_157872


namespace triangle_side_calculation_l1578_157883

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) : 
  a = 10 → B = 2 * π / 3 → C = π / 6 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b = 10 * Real.sqrt 3 := by
sorry

end triangle_side_calculation_l1578_157883


namespace intersection_with_complement_l1578_157831

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4}

-- Theorem statement
theorem intersection_with_complement : A ∩ (U \ B) = {1, 2} := by sorry

end intersection_with_complement_l1578_157831


namespace quadratic_function_property_l1578_157846

/-- Given a quadratic function f(x) = ax² + bx + c, 
    if f(0) = f(4) > f(1), then a > 0 and 4a + b = 0 -/
theorem quadratic_function_property (a b c : ℝ) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end quadratic_function_property_l1578_157846


namespace cube_root_eight_times_sixth_root_sixtyfour_equals_four_l1578_157814

theorem cube_root_eight_times_sixth_root_sixtyfour_equals_four :
  (8 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) = 4 := by sorry

end cube_root_eight_times_sixth_root_sixtyfour_equals_four_l1578_157814


namespace circle_condition_l1578_157824

theorem circle_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0) ↔ (k > 4 ∨ k < -1) :=
by sorry

end circle_condition_l1578_157824


namespace area_of_M_l1578_157828

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (abs x + abs (4 - x) ≤ 4) ∧
               ((x^2 - 4*x - 2*y + 2) / (y - x + 3) ≥ 0) ∧
               (0 ≤ x ∧ x ≤ 4)}

-- Define the area function for sets in ℝ²
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_M : area M = 4 := by sorry

end area_of_M_l1578_157828


namespace david_money_left_l1578_157830

/-- Calculates the amount of money David has left after his trip -/
def money_left (initial_amount : ℕ) (difference : ℕ) : ℕ :=
  initial_amount - (initial_amount - difference) / 2

theorem david_money_left :
  money_left 1800 800 = 500 := by
  sorry

end david_money_left_l1578_157830


namespace hazel_drank_one_cup_l1578_157875

def lemonade_problem (total_cups : ℕ) (sold_to_kids : ℕ) : Prop :=
  let sold_to_crew : ℕ := total_cups / 2
  let given_to_friends : ℕ := sold_to_kids / 2
  let remaining_cups : ℕ := total_cups - (sold_to_crew + sold_to_kids + given_to_friends)
  remaining_cups = 1

theorem hazel_drank_one_cup : lemonade_problem 56 18 := by
  sorry

end hazel_drank_one_cup_l1578_157875


namespace squared_sum_inequality_l1578_157815

theorem squared_sum_inequality (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a^3 + b^3 = 2*a*b) :
  a^2 + b^2 ≤ 1 + a*b := by sorry

end squared_sum_inequality_l1578_157815


namespace circle_radius_with_tangent_and_secant_l1578_157823

/-- A circle with a tangent and a secant drawn from an external point -/
structure CircleWithTangentAndSecant where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the tangent -/
  tangent_length : ℝ
  /-- The length of the internal segment of the secant -/
  secant_internal_length : ℝ
  /-- The tangent and secant are mutually perpendicular -/
  perpendicular : True

/-- Theorem: If a circle has a tangent of length 12 and a secant with internal segment of length 10,
    and the tangent and secant are mutually perpendicular, then the radius of the circle is 13 -/
theorem circle_radius_with_tangent_and_secant 
  (c : CircleWithTangentAndSecant) 
  (h1 : c.tangent_length = 12) 
  (h2 : c.secant_internal_length = 10) :
  c.radius = 13 := by
  sorry

end circle_radius_with_tangent_and_secant_l1578_157823


namespace numbers_left_on_board_l1578_157896

theorem numbers_left_on_board : 
  let S := Finset.range 20
  (S.filter (fun n => n % 2 ≠ 0 ∧ n % 5 ≠ 4)).card = 8 := by
  sorry

end numbers_left_on_board_l1578_157896


namespace no_perfect_cube_in_range_l1578_157850

theorem no_perfect_cube_in_range : 
  ¬∃ n : ℤ, 4 ≤ n ∧ n ≤ 12 ∧ ∃ k : ℤ, n^2 + 3*n + 2 = k^3 := by
  sorry

end no_perfect_cube_in_range_l1578_157850


namespace bound_cyclic_fraction_l1578_157853

theorem bound_cyclic_fraction (a b x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < b)
  (h₃ : a ≤ x₁ ∧ x₁ ≤ b) (h₄ : a ≤ x₂ ∧ x₂ ≤ b)
  (h₅ : a ≤ x₃ ∧ x₃ ≤ b) (h₆ : a ≤ x₄ ∧ x₄ ≤ b) :
  1/b ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ 1/a :=
by sorry

end bound_cyclic_fraction_l1578_157853


namespace marble_difference_l1578_157809

theorem marble_difference (seokjin_marbles : ℕ) (yuna_marbles : ℕ) (jimin_marbles : ℕ) : 
  seokjin_marbles = 3 →
  yuna_marbles = seokjin_marbles - 1 →
  jimin_marbles = 2 * seokjin_marbles →
  jimin_marbles - yuna_marbles = 4 := by
sorry

end marble_difference_l1578_157809


namespace suzanna_bike_ride_l1578_157808

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Represents Suzanna's bike ride -/
theorem suzanna_bike_ride (speed : ℚ) (time : ℚ) (h1 : speed = 1 / 6) (h2 : time = 40) :
  distance_traveled speed time = 6 := by
  sorry

#check suzanna_bike_ride

end suzanna_bike_ride_l1578_157808


namespace octagon_handshakes_eight_students_l1578_157856

/-- The number of handshakes in an octagonal arrangement of students -/
def octagon_handshakes (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: In a group of 8 students arranged in an octagonal shape,
    where each student shakes hands once with every other student
    except their two neighbors, the total number of handshakes is 20. -/
theorem octagon_handshakes_eight_students :
  octagon_handshakes 8 = 20 := by
  sorry

end octagon_handshakes_eight_students_l1578_157856


namespace grape_rate_proof_l1578_157805

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The weight of grapes purchased in kg -/
def grape_weight : ℝ := 7

/-- The weight of mangoes purchased in kg -/
def mango_weight : ℝ := 9

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The total amount paid -/
def total_paid : ℝ := 985

theorem grape_rate_proof : 
  grape_rate * grape_weight + mango_rate * mango_weight = total_paid :=
by sorry

end grape_rate_proof_l1578_157805


namespace total_earnings_theorem_l1578_157893

/-- Represents the earnings of investors a, b, and c -/
structure Earnings where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculates the total earnings of a, b, and c -/
def total_earnings (e : Earnings) : ℚ :=
  e.a + e.b + e.c

/-- Theorem stating the total earnings given the investment and return ratios -/
theorem total_earnings_theorem (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let e := Earnings.mk (18*x*y) (20*x*y) (20*x*y)
  2*x*y = 120 → total_earnings e = 3480 := by
  sorry

#check total_earnings_theorem

end total_earnings_theorem_l1578_157893


namespace family_gave_forty_dollars_l1578_157879

/-- Represents the cost and composition of a family's movie outing -/
structure MovieOuting where
  regular_ticket_cost : ℕ
  child_discount : ℕ
  num_adults : ℕ
  num_children : ℕ
  change_received : ℕ

/-- Calculates the total amount given to the cashier for a movie outing -/
def total_amount_given (outing : MovieOuting) : ℕ :=
  let adult_cost := outing.regular_ticket_cost * outing.num_adults
  let child_cost := (outing.regular_ticket_cost - outing.child_discount) * outing.num_children
  let total_cost := adult_cost + child_cost
  total_cost + outing.change_received

/-- Theorem stating that the family gave the cashier $40 in total -/
theorem family_gave_forty_dollars :
  let outing : MovieOuting := {
    regular_ticket_cost := 9,
    child_discount := 2,
    num_adults := 2,
    num_children := 3,
    change_received := 1
  }
  total_amount_given outing = 40 := by sorry

end family_gave_forty_dollars_l1578_157879


namespace compound_interest_problem_l1578_157841

theorem compound_interest_problem (P : ℝ) (t : ℝ) : 
  P * (1 + 0.1)^t = 2420 → 
  P * (1 + 0.1)^(t+3) = 2662 → 
  t = 3 := by
sorry

end compound_interest_problem_l1578_157841


namespace a_minus_b_equals_two_l1578_157819

-- Define the functions f, g, h, and h_inv
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 3

-- State the theorem
theorem a_minus_b_equals_two (a b : ℝ) : 
  (∀ x, h a b x = x - 3) → 
  (∀ x, h a b (h_inv x) = x) → 
  a - b = 2 := by
  sorry


end a_minus_b_equals_two_l1578_157819


namespace smallest_n_with_conditions_l1578_157882

def is_sum_of_identical_digits (n : ℕ) (count : ℕ) : Prop :=
  ∃ (d : ℕ), d ≤ 9 ∧ n = count * d

theorem smallest_n_with_conditions : 
  let n := 6036
  (n > 0) ∧ 
  (n % 2010 = 0) ∧ 
  (n % 2012 = 0) ∧ 
  (n % 2013 = 0) ∧
  (is_sum_of_identical_digits n 2010) ∧
  (is_sum_of_identical_digits n 2012) ∧
  (is_sum_of_identical_digits n 2013) ∧
  (∀ m : ℕ, m > 0 ∧ 
            m % 2010 = 0 ∧ 
            m % 2012 = 0 ∧ 
            m % 2013 = 0 ∧
            is_sum_of_identical_digits m 2010 ∧
            is_sum_of_identical_digits m 2012 ∧
            is_sum_of_identical_digits m 2013 
            → m ≥ n) :=
by sorry

end smallest_n_with_conditions_l1578_157882


namespace square_of_binomial_equivalence_l1578_157811

theorem square_of_binomial_equivalence (x : ℝ) : (-3 - x) * (3 - x) = (x - 3)^2 := by
  sorry

end square_of_binomial_equivalence_l1578_157811


namespace square_root_properties_l1578_157891

theorem square_root_properties (x c d e f : ℝ) : 
  (x^3 - x^2 - 6*x + 2 = 0 → (x^2)^3 - 13*(x^2)^2 + 40*(x^2) - 4 = 0) ∧
  (x^4 + c*x^3 + d*x^2 + e*x + f = 0 → 
    (x^2)^4 + (2*d - c^2)*(x^2)^3 + (d^2 - 2*c*e + 2*f)*(x^2)^2 + (2*d*f - e^2)*(x^2) + f^2 = 0) :=
by sorry

end square_root_properties_l1578_157891


namespace tank_dimension_proof_l1578_157847

/-- Proves that the second dimension of a rectangular tank is 5 feet -/
theorem tank_dimension_proof (w : ℝ) : 
  w > 0 → -- w is positive
  4 * w * 3 > 0 → -- tank volume is positive
  2 * (4 * w + 4 * 3 + w * 3) = 1880 / 20 → -- surface area equation
  w = 5 := by
  sorry

end tank_dimension_proof_l1578_157847


namespace largest_number_on_board_l1578_157836

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

def set_of_numbers : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_number_on_board : 
  ∃ (m : ℕ), m ∈ set_of_numbers ∧ ∀ (n : ℕ), n ∈ set_of_numbers → n ≤ m ∧ m = 84 :=
sorry

end largest_number_on_board_l1578_157836


namespace binomial_coefficient_sum_l1578_157802

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x + 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  (a₁ + 3*a₃ + 5*a₅ + 7*a₇ + 9*a₉)^2 - (2*a₂ + 4*a₄ + 6*a₆ + 8*a₈)^2 = 3^12 := by
  sorry

end binomial_coefficient_sum_l1578_157802


namespace rectangle_perimeter_l1578_157876

theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) :
  length = 3 * breadth →
  area = 432 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 96 := by
  sorry

end rectangle_perimeter_l1578_157876


namespace paula_tickets_l1578_157899

/-- The number of tickets needed for Paula's amusement park rides -/
def tickets_needed (go_kart_rides : ℕ) (bumper_car_rides : ℕ) (go_kart_cost : ℕ) (bumper_car_cost : ℕ) : ℕ :=
  go_kart_rides * go_kart_cost + bumper_car_rides * bumper_car_cost

/-- Theorem: Paula needs 24 tickets for her amusement park rides -/
theorem paula_tickets : tickets_needed 1 4 4 5 = 24 := by
  sorry

end paula_tickets_l1578_157899


namespace expected_profit_is_140000_l1578_157895

/-- The probability of a machine malfunctioning within a day -/
def malfunction_prob : ℝ := 0.2

/-- The loss incurred when a machine malfunctions (in yuan) -/
def malfunction_loss : ℝ := 50000

/-- The profit made when a machine works normally (in yuan) -/
def normal_profit : ℝ := 100000

/-- The number of machines -/
def num_machines : ℕ := 2

/-- The expected profit of two identical machines within a day (in yuan) -/
def expected_profit : ℝ := num_machines * (normal_profit * (1 - malfunction_prob) - malfunction_loss * malfunction_prob)

theorem expected_profit_is_140000 : expected_profit = 140000 := by
  sorry

end expected_profit_is_140000_l1578_157895


namespace min_value_trig_expression_l1578_157854

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 := by
  sorry

end min_value_trig_expression_l1578_157854


namespace min_sum_of_product_l1578_157816

theorem min_sum_of_product (a b : ℤ) (h : a * b = 150) : 
  ∀ x y : ℤ, x * y = 150 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 150 ∧ a₀ + b₀ = -151 :=
by sorry

end min_sum_of_product_l1578_157816


namespace sara_lunch_bill_total_l1578_157833

/-- The total cost of Sara's lunch bill --/
def lunch_bill (hotdog_cost salad_cost drink_cost side_item_cost : ℚ) : ℚ :=
  hotdog_cost + salad_cost + drink_cost + side_item_cost

/-- Theorem stating that Sara's lunch bill totals $16.71 --/
theorem sara_lunch_bill_total :
  lunch_bill 5.36 5.10 2.50 3.75 = 16.71 := by
  sorry

end sara_lunch_bill_total_l1578_157833


namespace vector_equality_l1578_157804

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, x)

theorem vector_equality (x : ℝ) :
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1 - b.1)^2 + (a.2 - b.2)^2 →
  x = 1 ∨ x = -2 := by
sorry

end vector_equality_l1578_157804


namespace problem_1_problem_2_problem_3_problem_4_l1578_157821

-- Problem 1
theorem problem_1 : 3 + (-1) - (-3) + 2 = 10 := by sorry

-- Problem 2
theorem problem_2 : 12 + |(-6)| - (-8) * 3 = 42 := by sorry

-- Problem 3
theorem problem_3 : (2/3 - 1/4 - 3/8) * 24 = 1 := by sorry

-- Problem 4
theorem problem_4 : -1^2021 - (-3 * (2/3)^2 - 4/3 / 2^2) = 2/3 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1578_157821


namespace solve_linear_equation_l1578_157886

theorem solve_linear_equation (x : ℝ) :
  3 + 5 * x = 28 → x = 5 := by
  sorry

end solve_linear_equation_l1578_157886


namespace train_length_l1578_157874

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  bridge_length = 295 →
  (train_speed * crossing_time) - bridge_length = 80 := by
  sorry

end train_length_l1578_157874


namespace junk_mail_delivery_l1578_157826

/-- Calculates the total pieces of junk mail delivered given the number of houses with white and red mailboxes -/
def total_junk_mail (total_houses : ℕ) (white_mailboxes : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ) : ℕ :=
  (white_mailboxes + red_mailboxes) * mail_per_house

/-- Proves that the total junk mail delivered is 30 pieces given the specified conditions -/
theorem junk_mail_delivery :
  total_junk_mail 8 2 3 6 = 30 := by
  sorry

end junk_mail_delivery_l1578_157826


namespace primle_is_79_l1578_157838

def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_in_tens_place (n : ℕ) (d : ℕ) : Prop := n / 10 = d

def digit_in_ones_place (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem primle_is_79 (primle : ℕ) 
  (h1 : is_prime primle)
  (h2 : is_two_digit primle)
  (h3 : digit_in_tens_place primle 7)
  (h4 : ¬ digit_in_ones_place primle 7)
  (h5 : ¬ (digit_in_tens_place primle 1 ∨ digit_in_ones_place primle 1))
  (h6 : ¬ (digit_in_tens_place primle 3 ∨ digit_in_ones_place primle 3))
  (h7 : ¬ (digit_in_tens_place primle 4 ∨ digit_in_ones_place primle 4)) :
  primle = 79 := by
sorry

end primle_is_79_l1578_157838


namespace coupon_savings_difference_l1578_157844

/-- Represents the savings from a coupon given a price -/
def CouponSavings (price : ℝ) : (ℝ → ℝ) → ℝ := fun coupon => coupon price

/-- Coupon A: 20% off the listed price -/
def CouponA (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: $30 off the listed price -/
def CouponB (price : ℝ) : ℝ := 30

/-- Coupon C: 20% off the amount exceeding $100 -/
def CouponC (price : ℝ) : ℝ := 0.2 * (price - 100)

/-- The lowest price where Coupon A saves at least as much as Coupon B or C -/
def x : ℝ := 150

/-- The highest price where Coupon A saves at least as much as Coupon B or C -/
def y : ℝ := 300

theorem coupon_savings_difference :
  ∀ price : ℝ, price > 100 →
  (x ≤ price ∧ price ≤ y) ↔
  (CouponSavings price CouponA ≥ CouponSavings price CouponB ∧
   CouponSavings price CouponA ≥ CouponSavings price CouponC) →
  y - x = 150 := by sorry

end coupon_savings_difference_l1578_157844


namespace expression_evaluation_l1578_157848

theorem expression_evaluation (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2*x^2 = -6 := by
  sorry

end expression_evaluation_l1578_157848


namespace existence_of_xy_sequences_l1578_157857

def sequence_a : ℕ → ℤ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * sequence_a (n + 1) - sequence_a n

theorem existence_of_xy_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n : ℕ,
    sequence_a n = (y n ^ 2 + 7) / (x n - y n) :=
by
  sorry

end existence_of_xy_sequences_l1578_157857


namespace triangle_max_perimeter_l1578_157834

theorem triangle_max_perimeter :
  ∀ (x : ℕ),
  x > 0 →
  x ≤ 6 →
  x + 4*x > 20 →
  4*x + 20 > x →
  x + 20 > 4*x →
  (∀ y : ℕ, y > 0 → y ≤ 6 → y + 4*y > 20 → 4*y + 20 > y → y + 20 > 4*y → x + 4*x + 20 ≥ y + 4*y + 20) →
  x + 4*x + 20 = 50 :=
by sorry

end triangle_max_perimeter_l1578_157834


namespace range_of_a_l1578_157807

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | x ≤ a}
def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

-- State the theorem
theorem range_of_a (a : ℝ) : P a ⊇ Q → a ∈ Set.Ici 1 := by
  sorry

end range_of_a_l1578_157807


namespace proposition_1_proposition_3_l1578_157868

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β γ : Plane)

-- Assume the lines and planes are distinct
variable (h_distinct_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Proposition 1
theorem proposition_1 :
  parallel α β → contains α l → line_parallel_plane l β :=
by sorry

-- Proposition 3
theorem proposition_3 :
  ¬contains α m → contains α n → line_parallel m n → line_parallel_plane m α :=
by sorry

end proposition_1_proposition_3_l1578_157868


namespace arithmetic_equation_l1578_157849

theorem arithmetic_equation : 50 + 5 * 12 / (180 / 3) = 51 := by
  sorry

end arithmetic_equation_l1578_157849


namespace picnic_men_count_l1578_157867

/-- Given a picnic with 240 people, where there are 40 more men than women
    and 40 more adults than children, prove that there are 90 men. -/
theorem picnic_men_count :
  ∀ (men women adults children : ℕ),
    men + women + children = 240 →
    men = women + 40 →
    adults = children + 40 →
    men + women = adults →
    men = 90 := by
  sorry

end picnic_men_count_l1578_157867


namespace triangle_cosine_proof_l1578_157818

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → c = 9 → (Real.sin A) * (Real.sin C) = (Real.sin B)^2 → 
  Real.cos B = 61/72 := by sorry

end triangle_cosine_proof_l1578_157818


namespace slower_train_speed_l1578_157877

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem -/
theorem slower_train_speed
  (train_length : ℝ)
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 75)
  (h2 : faster_train_speed = 46)
  (h3 : passing_time = 54)
  : ∃ (slower_train_speed : ℝ),
    slower_train_speed = 36 ∧
    (2 * train_length) = (faster_train_speed - slower_train_speed) * (5/18) * passing_time :=
by sorry

end slower_train_speed_l1578_157877


namespace derivative_f_at_pi_half_l1578_157800

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem derivative_f_at_pi_half :
  deriv f (π / 2) = 1 := by sorry

end derivative_f_at_pi_half_l1578_157800


namespace jacks_walking_speed_l1578_157870

/-- Proves Jack's walking speed given the conditions of the problem -/
theorem jacks_walking_speed 
  (initial_distance : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : christina_speed = 8)
  (h3 : lindy_speed = 10)
  (h4 : lindy_distance = 100) : 
  ∃ (jack_speed : ℝ), jack_speed = 7 := by
  sorry

end jacks_walking_speed_l1578_157870


namespace half_radius_of_y_l1578_157863

-- Define the circles
variable (x y : ℝ → Prop)

-- Define the radius and area functions
noncomputable def radius (c : ℝ → Prop) : ℝ := sorry
noncomputable def area (c : ℝ → Prop) : ℝ := sorry

-- State the theorem
theorem half_radius_of_y (h1 : area x = area y) (h2 : 2 * π * radius x = 20 * π) :
  radius y / 2 = 5 := by sorry

end half_radius_of_y_l1578_157863


namespace probability_of_point_in_region_l1578_157803

-- Define the lines
def line1 (x : ℝ) : ℝ := -2 * x + 8
def line2 (x : ℝ) : ℝ := -3 * x + 9

-- Define the region of interest
def region_of_interest (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ y ≤ line1 x ∧ y ≥ line2 x

-- Define the total area under line1 in the first quadrant
def total_area : ℝ := 16

-- Define the area of the region of interest
def area_of_interest : ℝ := 14.5

-- Theorem statement
theorem probability_of_point_in_region :
  (area_of_interest / total_area) = 0.90625 :=
sorry

end probability_of_point_in_region_l1578_157803


namespace value_added_to_half_l1578_157864

theorem value_added_to_half : ∃ (v : ℝ), (20 / 2) + v = 17 ∧ v = 7 := by sorry

end value_added_to_half_l1578_157864


namespace john_average_score_l1578_157861

def john_scores : List ℝ := [95, 88, 91, 87, 92, 90]

theorem john_average_score :
  (john_scores.sum / john_scores.length : ℝ) = 90.5 := by
  sorry

end john_average_score_l1578_157861


namespace mapping_result_l1578_157898

def A : Set ℕ := {1, 2}

def f (x : ℕ) : ℕ := x^2

def B : Set ℕ := f '' A

theorem mapping_result : B = {1, 4} := by sorry

end mapping_result_l1578_157898


namespace job_completion_time_l1578_157878

theorem job_completion_time (P Q : ℝ) (h1 : Q = 15) (h2 : 3 / P + 3 / Q + 1 / (5 * P) = 1) : P = 4 := by
  sorry

end job_completion_time_l1578_157878


namespace trees_in_yard_l1578_157852

/-- The number of trees in a yard with given length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem: There are 31 trees in a 360-meter yard with 12-meter spacing -/
theorem trees_in_yard :
  num_trees 360 12 = 31 := by
  sorry

end trees_in_yard_l1578_157852


namespace largest_angle_in_special_triangle_l1578_157881

/-- A triangle with altitudes 9, 12, and 18 -/
structure TriangleWithAltitudes where
  a : ℝ
  b : ℝ
  c : ℝ
  altitude_a : ℝ
  altitude_b : ℝ
  altitude_c : ℝ
  ha : altitude_a = 9
  hb : altitude_b = 12
  hc : altitude_c = 18
  area_eq1 : a * altitude_a = b * altitude_b
  area_eq2 : b * altitude_b = c * altitude_c
  triangle_ineq1 : a + b > c
  triangle_ineq2 : b + c > a
  triangle_ineq3 : c + a > b

/-- The largest angle in a triangle with altitudes 9, 12, and 18 is arccos(-1/4) -/
theorem largest_angle_in_special_triangle (t : TriangleWithAltitudes) :
  ∃ θ : ℝ, θ = Real.arccos (-1/4) ∧ 
  θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))
         (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
              (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry


end largest_angle_in_special_triangle_l1578_157881


namespace some_number_value_l1578_157832

theorem some_number_value (x : ℝ) : x * 6000 = 480 * 10^5 → x = 8000 := by
  sorry

end some_number_value_l1578_157832


namespace kim_math_test_probability_l1578_157835

theorem kim_math_test_probability (p : ℚ) (h : p = 4/7) :
  1 - p = 3/7 := by
  sorry

end kim_math_test_probability_l1578_157835
