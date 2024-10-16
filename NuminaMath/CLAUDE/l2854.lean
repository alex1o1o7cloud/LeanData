import Mathlib

namespace NUMINAMATH_CALUDE_lucky_money_distribution_l2854_285437

/-- Represents a distribution of lucky money among three grandsons -/
structure LuckyMoneyDistribution where
  grandson1 : ℕ
  grandson2 : ℕ
  grandson3 : ℕ

/-- Checks if a distribution satisfies the given conditions -/
def isValidDistribution (d : LuckyMoneyDistribution) : Prop :=
  ∃ (x y z : ℕ),
    (d.grandson1 = 10 * x ∧ d.grandson2 = 20 * y ∧ d.grandson3 = 50 * z) ∧
    (x = y * z) ∧
    (d.grandson1 + d.grandson2 + d.grandson3 = 300)

/-- The theorem stating the only valid distributions -/
theorem lucky_money_distribution :
  ∀ d : LuckyMoneyDistribution,
    isValidDistribution d →
    (d = ⟨100, 100, 100⟩ ∨ d = ⟨90, 60, 150⟩) :=
by sorry


end NUMINAMATH_CALUDE_lucky_money_distribution_l2854_285437


namespace NUMINAMATH_CALUDE_power_inequality_l2854_285453

theorem power_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ^ b > b ^ a) (hbc : b ^ c > c ^ b) : 
  a ^ c > c ^ a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2854_285453


namespace NUMINAMATH_CALUDE_fraction_equality_l2854_285430

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = -3/4) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = 3/4 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2854_285430


namespace NUMINAMATH_CALUDE_total_people_after_hour_l2854_285438

/-- Represents the number of people in each line at the fair -/
structure FairLines where
  ferrisWheel : ℕ
  bumperCars : ℕ
  rollerCoaster : ℕ

/-- Calculates the total number of people across all lines after an hour -/
def totalPeopleAfterHour (initial : FairLines) (x y Z : ℕ) : ℕ :=
  Z + initial.rollerCoaster * 2

/-- Theorem stating the total number of people after an hour -/
theorem total_people_after_hour 
  (initial : FairLines)
  (x y Z : ℕ)
  (h1 : initial.ferrisWheel = 50)
  (h2 : initial.bumperCars = 50)
  (h3 : initial.rollerCoaster = 50)
  (h4 : Z = (50 - x) + (50 + y)) :
  totalPeopleAfterHour initial x y Z = Z + 100 := by
  sorry

#check total_people_after_hour

end NUMINAMATH_CALUDE_total_people_after_hour_l2854_285438


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l2854_285475

variable (n : ℕ)

/-- The number of ways to arrange 3n people into circles with ABC pattern -/
def ball_arrangements (n : ℕ) : ℕ := (3 * n).factorial

/-- Theorem: The number of ball arrangements is (3n)! -/
theorem ball_arrangements_count :
  ball_arrangements n = (3 * n).factorial := by
  sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l2854_285475


namespace NUMINAMATH_CALUDE_gcd_1987_1463_l2854_285489

theorem gcd_1987_1463 : Nat.gcd 1987 1463 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1987_1463_l2854_285489


namespace NUMINAMATH_CALUDE_second_green_probability_l2854_285467

-- Define the contents of each bag
def bag1 : Finset ℕ := {0, 0, 0, 1}  -- 0 represents green, 1 represents red
def bag2 : Finset ℕ := {0, 0, 1, 1}
def bag3 : Finset ℕ := {0, 1, 1, 1}

-- Define the probability of selecting each bag
def bagProb : ℕ → ℚ
  | 1 => 1/3
  | 2 => 1/3
  | 3 => 1/3
  | _ => 0

-- Define the probability of selecting a green candy from a bag
def greenProb : Finset ℕ → ℚ
  | s => (s.filter (· = 0)).card / s.card

-- Define the probability of selecting a red candy from a bag
def redProb : Finset ℕ → ℚ
  | s => (s.filter (· = 1)).card / s.card

-- Define the probability of selecting a green candy as the second candy
def secondGreenProb : ℚ := sorry

theorem second_green_probability : secondGreenProb = 73/144 := by sorry

end NUMINAMATH_CALUDE_second_green_probability_l2854_285467


namespace NUMINAMATH_CALUDE_rectangular_cards_are_squares_l2854_285402

/-- Represents a rectangular card with dimensions width and height -/
structure Card where
  width : ℕ+
  height : ℕ+

/-- Represents the result of a child cutting their card into squares -/
structure CutResult where
  squareCount : ℕ+

theorem rectangular_cards_are_squares
  (n : ℕ+)
  (h_n : n > 1)
  (cards : Fin n → Card)
  (h_identical : ∀ i j : Fin n, cards i = cards j)
  (cuts : Fin n → CutResult)
  (h_prime_total : Nat.Prime (Finset.sum (Finset.range n) (λ i => (cuts i).squareCount))) :
  ∀ i : Fin n, (cards i).width = (cards i).height :=
sorry

end NUMINAMATH_CALUDE_rectangular_cards_are_squares_l2854_285402


namespace NUMINAMATH_CALUDE_lemons_given_away_fraction_l2854_285442

def dozen : ℕ := 12

theorem lemons_given_away_fraction (lemons_left : ℕ) 
  (h1 : lemons_left = 9) : 
  (dozen - lemons_left : ℚ) / dozen = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lemons_given_away_fraction_l2854_285442


namespace NUMINAMATH_CALUDE_g_composition_fixed_points_l2854_285456

def g (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem g_composition_fixed_points :
  {x : ℝ | g (g x) = g x} =
  {x : ℝ | x = 2 + Real.sqrt ((11 + 2*Real.sqrt 21)/2) ∨
           x = 2 - Real.sqrt ((11 + 2*Real.sqrt 21)/2) ∨
           x = 2 + Real.sqrt ((11 - 2*Real.sqrt 21)/2) ∨
           x = 2 - Real.sqrt ((11 - 2*Real.sqrt 21)/2)} :=
by sorry

end NUMINAMATH_CALUDE_g_composition_fixed_points_l2854_285456


namespace NUMINAMATH_CALUDE_card_ratio_problem_l2854_285494

theorem card_ratio_problem (b j m : ℕ) : 
  j = b + 9 →  -- Janet has 9 cards more than Brenda
  m = 150 - 40 →  -- Mara has 40 cards less than 150
  b + j + m = 211 →  -- Total cards is 211
  ∃ (k : ℕ), m = k * j →  -- Mara's cards are a multiple of Janet's
  m / j = 2 :=  -- Ratio of Mara's cards to Janet's cards is 2:1
by
  sorry

end NUMINAMATH_CALUDE_card_ratio_problem_l2854_285494


namespace NUMINAMATH_CALUDE_divide_people_eq_280_l2854_285460

/-- The number of ways to divide 8 people into three groups -/
def divide_people : ℕ :=
  let total_people : ℕ := 8
  let group_1_size : ℕ := 3
  let group_2_size : ℕ := 3
  let group_3_size : ℕ := 2
  let ways_to_choose_group_1 := Nat.choose total_people group_1_size
  let ways_to_choose_group_2 := Nat.choose (total_people - group_1_size) group_2_size
  let ways_to_choose_group_3 := Nat.choose group_3_size group_3_size
  let arrangements_of_identical_groups : ℕ := 2  -- 2! for two identical groups of 3
  (ways_to_choose_group_1 * ways_to_choose_group_2 * ways_to_choose_group_3) / arrangements_of_identical_groups

theorem divide_people_eq_280 : divide_people = 280 := by
  sorry

end NUMINAMATH_CALUDE_divide_people_eq_280_l2854_285460


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2854_285431

theorem arithmetic_calculation : 6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2854_285431


namespace NUMINAMATH_CALUDE_trader_shipment_cost_l2854_285448

/-- The amount needed for the next shipment of wares --/
def amount_needed (total_profit donation excess : ℕ) : ℕ :=
  total_profit / 2 + donation - excess

/-- Theorem stating the amount needed for the next shipment --/
theorem trader_shipment_cost (total_profit donation excess : ℕ)
  (h1 : total_profit = 960)
  (h2 : donation = 310)
  (h3 : excess = 180) :
  amount_needed total_profit donation excess = 610 := by
  sorry

#eval amount_needed 960 310 180

end NUMINAMATH_CALUDE_trader_shipment_cost_l2854_285448


namespace NUMINAMATH_CALUDE_hat_cost_l2854_285425

theorem hat_cost (initial_amount : ℕ) (num_sock_pairs : ℕ) (cost_per_sock_pair : ℕ) (amount_left : ℕ) : 
  initial_amount = 20 ∧ 
  num_sock_pairs = 4 ∧ 
  cost_per_sock_pair = 2 ∧ 
  amount_left = 5 → 
  initial_amount - (num_sock_pairs * cost_per_sock_pair) - amount_left = 7 :=
by sorry

end NUMINAMATH_CALUDE_hat_cost_l2854_285425


namespace NUMINAMATH_CALUDE_smallest_fraction_proof_l2854_285445

def is_natural_number (q : ℚ) : Prop := ∃ (n : ℕ), q = n

theorem smallest_fraction_proof (f : ℚ) : 
  (f ≥ 42/5) →
  (is_natural_number (f / (21/25))) →
  (is_natural_number (f / (14/15))) →
  (∀ g : ℚ, g < f → ¬(is_natural_number (g / (21/25)) ∧ is_natural_number (g / (14/15)))) →
  f = 42/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_proof_l2854_285445


namespace NUMINAMATH_CALUDE_square_side_from_diagonal_difference_l2854_285428

/-- Given the difference between the diagonal and side of a square, 
    the side of the square can be uniquely determined. -/
theorem square_side_from_diagonal_difference (d_minus_a : ℝ) (d_minus_a_pos : 0 < d_minus_a) :
  ∃! a : ℝ, ∃ d : ℝ, 
    0 < a ∧ 
    d = Real.sqrt (2 * a ^ 2) ∧ 
    d - a = d_minus_a :=
by sorry

end NUMINAMATH_CALUDE_square_side_from_diagonal_difference_l2854_285428


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_condition_for_f_geq_g_l2854_285443

-- Define f(x) and g(x)
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x - 1|
def g : ℝ → ℝ := λ x => 2

-- Theorem 1: Solution set when a = 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ g x} = {x : ℝ | x ≤ 2/3} := by sorry

-- Theorem 2: Condition for f(x) ≥ g(x) to always hold
theorem condition_for_f_geq_g (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g x) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_condition_for_f_geq_g_l2854_285443


namespace NUMINAMATH_CALUDE_interior_angles_sum_l2854_285495

theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 2340) →
  (180 * ((n + 3) - 2) = 2880) :=
by sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l2854_285495


namespace NUMINAMATH_CALUDE_vacant_seats_l2854_285463

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 700) (h2 : filled_percentage = 75 / 100) :
  (1 - filled_percentage) * total_seats = 175 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l2854_285463


namespace NUMINAMATH_CALUDE_difference_one_third_and_decimal_l2854_285452

theorem difference_one_third_and_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / 3000 := by sorry

end NUMINAMATH_CALUDE_difference_one_third_and_decimal_l2854_285452


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2854_285466

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2854_285466


namespace NUMINAMATH_CALUDE_sugar_substitute_box_cost_l2854_285414

theorem sugar_substitute_box_cost 
  (packets_per_coffee : ℕ)
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (days_supply : ℕ)
  (total_cost : ℝ)
  (h1 : packets_per_coffee = 1)
  (h2 : coffees_per_day = 2)
  (h3 : packets_per_box = 30)
  (h4 : days_supply = 90)
  (h5 : total_cost = 24) :
  total_cost / (days_supply * coffees_per_day * packets_per_coffee / packets_per_box) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sugar_substitute_box_cost_l2854_285414


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2854_285451

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*a*x + a + 2 ≤ 0

def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | quadratic_inequality a x}

theorem solution_set_characterization (a : ℝ) :
  (solution_set a ⊆ Set.Icc 1 3) ↔ a ∈ Set.Ioo (-1) (11/5) :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2854_285451


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2854_285454

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 18 = 0) → 
  (3 * b^2 + 9 * b - 18 = 0) → 
  (3*a - 2) * (6*b - 9) = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2854_285454


namespace NUMINAMATH_CALUDE_marcos_lap_time_improvement_l2854_285440

/-- Represents the improvement in lap time for Marcos after training -/
theorem marcos_lap_time_improvement :
  let initial_laps : ℕ := 15
  let initial_time : ℕ := 45
  let final_laps : ℕ := 18
  let final_time : ℕ := 42
  let initial_lap_time := initial_time / initial_laps
  let final_lap_time := final_time / final_laps
  let improvement := initial_lap_time - final_lap_time
  improvement = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_marcos_lap_time_improvement_l2854_285440


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l2854_285406

/-- Represents the possible positions on the circle -/
inductive Position : Type
| one : Position
| two : Position
| three : Position
| four : Position
| five : Position
| six : Position
| seven : Position

/-- Determines if a position is odd-numbered -/
def is_odd (p : Position) : Bool :=
  match p with
  | Position.one => true
  | Position.two => false
  | Position.three => true
  | Position.four => false
  | Position.five => true
  | Position.six => false
  | Position.seven => true

/-- Represents a single jump of the bug -/
def jump (p : Position) : Position :=
  match p with
  | Position.one => Position.three
  | Position.two => Position.five
  | Position.three => Position.five
  | Position.four => Position.seven
  | Position.five => Position.seven
  | Position.six => Position.two
  | Position.seven => Position.two

/-- Represents multiple jumps of the bug -/
def multi_jump (p : Position) (n : Nat) : Position :=
  match n with
  | 0 => p
  | n + 1 => jump (multi_jump p n)

/-- The main theorem to prove -/
theorem bug_position_after_2023_jumps :
  multi_jump Position.seven 2023 = Position.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l2854_285406


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l2854_285496

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l2854_285496


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2854_285486

theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  -- ABCDEF is a convex hexagon (sum of angles is 720°)
  A + B + C + D + E + F = 720 →
  -- Angles A, B, C, and D are congruent
  A = B ∧ B = C ∧ C = D →
  -- Angles E and F are congruent
  E = F →
  -- Measure of angle A is 30° less than measure of angle E
  A + 30 = E →
  -- Conclusion: Measure of angle E is 140°
  E = 140 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2854_285486


namespace NUMINAMATH_CALUDE_a_10_value_l2854_285473

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 12 = 16)
  (h_7 : a 7 = 1) :
  a 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_10_value_l2854_285473


namespace NUMINAMATH_CALUDE_simplify_expression_l2854_285450

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2854_285450


namespace NUMINAMATH_CALUDE_unique_element_in_A_not_in_B_l2854_285427

-- Define the sets A and B
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {2, 4, 6}

-- State the theorem
theorem unique_element_in_A_not_in_B :
  ∀ x : ℕ, x ∈ A ∧ x ∉ B → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_in_A_not_in_B_l2854_285427


namespace NUMINAMATH_CALUDE_polynomial_intersection_theorem_l2854_285449

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the theorem
theorem polynomial_intersection_theorem (a b c d : ℝ) : 
  -- f and g are distinct polynomials
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- The graphs intersect at (50, -200)
  f a b 50 = -200 ∧ g c d 50 = -200 →
  -- The minimum value of f is 50 less than the minimum value of g
  (-a^2/4 + b) = (-c^2/4 + d - 50) →
  -- There exists a unique value for a + c
  ∃! x, x = a + c :=
by sorry

end NUMINAMATH_CALUDE_polynomial_intersection_theorem_l2854_285449


namespace NUMINAMATH_CALUDE_equation_solution_l2854_285400

theorem equation_solution : ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2854_285400


namespace NUMINAMATH_CALUDE_afternoon_and_evening_emails_l2854_285490

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 5

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- Theorem: The sum of emails Jack received in the afternoon and evening is 13 -/
theorem afternoon_and_evening_emails :
  afternoon_emails + evening_emails = 13 := by sorry

end NUMINAMATH_CALUDE_afternoon_and_evening_emails_l2854_285490


namespace NUMINAMATH_CALUDE_solution_system_equations_l2854_285499

theorem solution_system_equations (A : ℤ) (hA : A ≠ 0) :
  ∀ x y z : ℤ,
    x + y^2 + z^3 = A ∧
    (1 : ℚ) / x + (1 : ℚ) / y^2 + (1 : ℚ) / z^3 = (1 : ℚ) / A ∧
    x * y^2 * z^3 = A^2 →
    ∃ k : ℤ, A = -k^12 ∧
      ((x = -k^12 ∧ (y = k^3 ∨ y = -k^3) ∧ z = -k^2) ∨
       (x = -k^3 ∧ (y = k^3 ∨ y = -k^3) ∧ z = -k^4)) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2854_285499


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l2854_285447

theorem initial_number_of_persons
  (average_weight_increase : ℝ)
  (weight_of_leaving_person : ℝ)
  (weight_of_new_person : ℝ)
  (h1 : average_weight_increase = 5.5)
  (h2 : weight_of_leaving_person = 68)
  (h3 : weight_of_new_person = 95.5) :
  ∃ N : ℕ, N = 5 ∧ 
  N * average_weight_increase = weight_of_new_person - weight_of_leaving_person :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l2854_285447


namespace NUMINAMATH_CALUDE_price_restoration_l2854_285476

theorem price_restoration (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.8 * original_price
  (reduced_price * 1.25 = original_price) := by sorry

end NUMINAMATH_CALUDE_price_restoration_l2854_285476


namespace NUMINAMATH_CALUDE_classroom_to_total_ratio_is_one_to_four_l2854_285457

/-- Given a class of students with some on the playground and some in the classroom,
    prove that the ratio of students in the classroom to total students is 1:4. -/
theorem classroom_to_total_ratio_is_one_to_four
  (total_students : ℕ)
  (playground_students : ℕ)
  (classroom_students : ℕ)
  (playground_girls : ℕ)
  (h1 : total_students = 20)
  (h2 : total_students = playground_students + classroom_students)
  (h3 : playground_girls = 10)
  (h4 : playground_girls = (2 : ℚ) / 3 * playground_students) :
  (classroom_students : ℚ) / total_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_classroom_to_total_ratio_is_one_to_four_l2854_285457


namespace NUMINAMATH_CALUDE_logan_grocery_budget_l2854_285487

/-- Calculates the amount Logan can spend on groceries annually given his financial parameters. -/
def grocery_budget (current_income : ℕ) (income_increase : ℕ) (rent : ℕ) (gas : ℕ) (desired_savings : ℕ) : ℕ :=
  (current_income + income_increase) - (rent + gas + desired_savings)

/-- Theorem stating that Logan's grocery budget is $5,000 given his financial parameters. -/
theorem logan_grocery_budget :
  grocery_budget 65000 10000 20000 8000 42000 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_logan_grocery_budget_l2854_285487


namespace NUMINAMATH_CALUDE_rational_sum_product_quotient_zero_l2854_285485

theorem rational_sum_product_quotient_zero (a b : ℚ) :
  (a + b) / (a * b) = 0 → a ≠ 0 ∧ b ≠ 0 ∧ a = -b := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_product_quotient_zero_l2854_285485


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l2854_285498

open Complex

theorem imaginary_part_of_fraction (i : ℂ) (h : i * i = -1) :
  (((1 : ℂ) + i) / ((1 : ℂ) - i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l2854_285498


namespace NUMINAMATH_CALUDE_number_problem_l2854_285444

theorem number_problem (x : ℝ) : 0.6667 * x + 0.75 = 1.6667 → x = 1.375 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2854_285444


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2854_285413

theorem quadratic_root_value (d : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + d = 0 ↔ x = (-14 + Real.sqrt 14) / 4 ∨ x = (-14 - Real.sqrt 14) / 4) →
  d = 91/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2854_285413


namespace NUMINAMATH_CALUDE_reflection_result_l2854_285478

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def C : ℝ × ℝ := (3, 3)

theorem reflection_result :
  (reflect_over_x_axis ∘ reflect_over_y_axis) C = (-3, -3) := by
sorry

end NUMINAMATH_CALUDE_reflection_result_l2854_285478


namespace NUMINAMATH_CALUDE_cuboid_third_edge_length_l2854_285423

/-- Given a cuboid with two edges of 4 cm and 5 cm, and a surface area of 148 cm², 
    the length of the third edge is 6 cm. -/
theorem cuboid_third_edge_length : 
  ∀ (x : ℝ), 
    (2 * (4 * 5 + 4 * x + 5 * x) = 148) → 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_third_edge_length_l2854_285423


namespace NUMINAMATH_CALUDE_not_perpendicular_to_y_axis_can_be_perpendicular_to_x_axis_l2854_285483

/-- A line in the form x = ky + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Predicate for a line being perpendicular to the y-axis -/
def perpendicular_to_y_axis (l : Line) : Prop :=
  ∃ c : ℝ, ∀ y : ℝ, l.k * y + l.b = c

/-- Predicate for a line being perpendicular to the x-axis -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  l.k = 0

/-- Theorem stating that lines in the form x = ky + b cannot be perpendicular to the y-axis -/
theorem not_perpendicular_to_y_axis :
  ¬ ∃ l : Line, perpendicular_to_y_axis l :=
sorry

/-- Theorem stating that lines in the form x = ky + b can be perpendicular to the x-axis -/
theorem can_be_perpendicular_to_x_axis :
  ∃ l : Line, perpendicular_to_x_axis l :=
sorry

end NUMINAMATH_CALUDE_not_perpendicular_to_y_axis_can_be_perpendicular_to_x_axis_l2854_285483


namespace NUMINAMATH_CALUDE_function_has_infinitely_many_extreme_points_l2854_285416

/-- The function f(x) = x^2 - 2x cos(x) has infinitely many extreme points -/
theorem function_has_infinitely_many_extreme_points :
  ∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 2*x*(Real.cos x)) ∧
  (∀ n : ℕ, ∃ (S : Finset ℝ), S.card ≥ n ∧ 
    (∀ x ∈ S, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x)) :=
by sorry


end NUMINAMATH_CALUDE_function_has_infinitely_many_extreme_points_l2854_285416


namespace NUMINAMATH_CALUDE_cos_squared_pi_eighth_minus_one_l2854_285401

theorem cos_squared_pi_eighth_minus_one (π : Real) : 2 * Real.cos (π / 8) ^ 2 - 1 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_pi_eighth_minus_one_l2854_285401


namespace NUMINAMATH_CALUDE_f_greater_than_g_l2854_285465

/-- The function f defined as f(x) = 3x^2 - x + 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 - x + 1

/-- The function g defined as g(x) = 2x^2 + x - 1 -/
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

/-- For all real x, f(x) > g(x) -/
theorem f_greater_than_g : ∀ x : ℝ, f x > g x := by sorry

end NUMINAMATH_CALUDE_f_greater_than_g_l2854_285465


namespace NUMINAMATH_CALUDE_linear_system_solution_quadratic_system_result_l2854_285409

-- Define the system of linear equations
def linear_system (x y : ℝ) : Prop :=
  3 * x - 2 * y = 5 ∧ 9 * x - 4 * y = 19

-- Define the system of quadratic equations
def quadratic_system (x y : ℝ) : Prop :=
  3 * x^2 - 2 * x * y + 12 * y^2 = 47 ∧ 2 * x^2 + x * y + 8 * y^2 = 36

-- Theorem for the linear system
theorem linear_system_solution :
  ∃ x y : ℝ, linear_system x y ∧ x = 3 ∧ y = 2 :=
sorry

-- Theorem for the quadratic system
theorem quadratic_system_result :
  ∀ x y : ℝ, quadratic_system x y → x^2 + 4 * y^2 = 17 :=
sorry

end NUMINAMATH_CALUDE_linear_system_solution_quadratic_system_result_l2854_285409


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2854_285433

def satisfies_remainder_conditions (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 ∧
  n % 8 = 7 ∧ n % 9 = 8 ∧ n % 10 = 9 ∧ n % 11 = 10 ∧ n % 12 = 11

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → n % m ≠ 0

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ),
    n = 10079 ∧
    satisfies_remainder_conditions n ∧
    ¬∃ (m : ℕ), m * m = n ∧
    is_prime (sum_of_digits n) ∧
    n > 10000 ∧
    ∀ (k : ℕ), 10000 < k ∧ k < n →
      ¬(satisfies_remainder_conditions k ∧
        ¬∃ (m : ℕ), m * m = k ∧
        is_prime (sum_of_digits k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2854_285433


namespace NUMINAMATH_CALUDE_anya_lost_games_l2854_285462

/-- Represents a girl in the table tennis game -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the number of games played by each girl -/
def games_played (g : Girl) : ℕ :=
  match g with
  | .Anya => 4
  | .Bella => 6
  | .Valya => 7
  | .Galya => 10
  | .Dasha => 11

/-- The total number of games played -/
def total_games : ℕ := 19

/-- Theorem stating that Anya lost in specific games -/
theorem anya_lost_games :
  ∃ (lost_games : List ℕ),
    lost_games = [4, 8, 12, 16] ∧
    (∀ g ∈ lost_games, g ≤ total_games) ∧
    (∀ g ∈ lost_games, ∃ i, g = 4 * i) ∧
    lost_games.length = games_played Girl.Anya := by
  sorry

end NUMINAMATH_CALUDE_anya_lost_games_l2854_285462


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2854_285497

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardCount : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def probAdjacentRed : ℚ := 25 / 51

/-- The expected number of pairs of adjacent red cards in a standard 52-card deck
    dealt in a circle -/
theorem expected_adjacent_red_pairs :
  (redCardCount : ℚ) * probAdjacentRed = 650 / 51 := by sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2854_285497


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2854_285418

theorem simplify_fraction_product : 
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2854_285418


namespace NUMINAMATH_CALUDE_even_odd_sum_zero_l2854_285474

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- Main theorem: If f is even and g(x) = f(x-1) is odd, then f(2009) + f(2011) = 0 -/
theorem even_odd_sum_zero (f : ℝ → ℝ) (g : ℝ → ℝ) 
    (h_even : IsEven f) (h_odd : IsOdd g) (h_g : ∀ x, g x = f (x - 1)) :
    f 2009 + f 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_zero_l2854_285474


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2854_285446

/-- Given a quadratic equation (k-1)x^2 + 6x + k^2 - k = 0 with a root of 0, prove that k = 0 -/
theorem quadratic_root_zero (k : ℝ) : 
  (∃ x, (k - 1) * x^2 + 6 * x + k^2 - k = 0) ∧ 
  ((k - 1) * 0^2 + 6 * 0 + k^2 - k = 0) → 
  k = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2854_285446


namespace NUMINAMATH_CALUDE_final_K_value_l2854_285480

/-- Represents the state of the program at each iteration -/
structure ProgramState :=
  (S : ℕ)
  (K : ℕ)

/-- Defines a single iteration of the loop -/
def iterate (state : ProgramState) : ProgramState :=
  { S := state.S^2 + 1,
    K := state.K + 1 }

/-- Defines the condition for continuing the loop -/
def loopCondition (state : ProgramState) : Prop :=
  state.S < 100

/-- Theorem: The final value of K is 4 -/
theorem final_K_value :
  ∃ (n : ℕ), ∃ (finalState : ProgramState),
    (finalState.K = 4) ∧
    (¬loopCondition finalState) ∧
    (finalState = (iterate^[n] ⟨1, 1⟩)) :=
  sorry

end NUMINAMATH_CALUDE_final_K_value_l2854_285480


namespace NUMINAMATH_CALUDE_motel_rent_reduction_l2854_285404

/-- Represents a motel with rooms rented at two different prices -/
structure Motel :=
  (price1 : ℕ)
  (price2 : ℕ)
  (total_rent : ℕ)
  (room_change : ℕ)

/-- The percentage reduction in total rent when changing room prices -/
def rent_reduction_percentage (m : Motel) : ℚ :=
  ((m.price2 - m.price1) * m.room_change : ℚ) / m.total_rent * 100

/-- Theorem stating that for a motel with specific conditions, 
    changing 10 rooms from $60 to $40 results in a 10% rent reduction -/
theorem motel_rent_reduction 
  (m : Motel) 
  (h1 : m.price1 = 40)
  (h2 : m.price2 = 60)
  (h3 : m.total_rent = 2000)
  (h4 : m.room_change = 10) :
  rent_reduction_percentage m = 10 := by
  sorry

#eval rent_reduction_percentage ⟨40, 60, 2000, 10⟩

end NUMINAMATH_CALUDE_motel_rent_reduction_l2854_285404


namespace NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l2854_285477

theorem min_value_cube_root_plus_inverse_square (x : ℝ) (h : x > 0) :
  3 * x^(1/3) + 4 / x^2 ≥ 7 ∧
  (3 * x^(1/3) + 4 / x^2 = 7 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_cube_root_plus_inverse_square_l2854_285477


namespace NUMINAMATH_CALUDE_vector_problem_l2854_285410

/-- Given vectors and triangle properties, prove vector n coordinates and magnitude range of n + p -/
theorem vector_problem (m n p q : ℝ × ℝ) (A B C : ℝ) : 
  m = (1, 1) →
  q = (1, 0) →
  (m.1 * n.1 + m.2 * n.2) = -1 →
  ∃ (k : ℝ), n = k • q →
  p = (2 * Real.cos (C / 2) ^ 2, Real.cos A) →
  B = π / 3 →
  A + B + C = π →
  (n = (-1, 0) ∧ Real.sqrt 2 / 2 ≤ Real.sqrt ((n.1 + p.1)^2 + (n.2 + p.2)^2) ∧ 
   Real.sqrt ((n.1 + p.1)^2 + (n.2 + p.2)^2) < Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l2854_285410


namespace NUMINAMATH_CALUDE_different_group_choices_l2854_285471

theorem different_group_choices (n : Nat) (h : n = 3) : 
  n^2 - n = 6 := by
  sorry

#check different_group_choices

end NUMINAMATH_CALUDE_different_group_choices_l2854_285471


namespace NUMINAMATH_CALUDE_money_ratio_l2854_285403

theorem money_ratio (bob phil jenna : ℚ) : 
  bob = 60 →
  phil = (1/3) * bob →
  jenna = bob - 20 →
  jenna / phil = 2 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_l2854_285403


namespace NUMINAMATH_CALUDE_combined_rate_is_90_l2854_285408

/-- Represents the fish fillet production scenario -/
structure FishFilletProduction where
  totalRequired : ℕ
  deadline : ℕ
  firstTeamProduction : ℕ
  secondTeamProduction : ℕ
  thirdTeamRate : ℕ

/-- Calculates the combined production rate of the third and fourth teams -/
def combinedRate (p : FishFilletProduction) : ℕ :=
  let remainingPieces := p.totalRequired - (p.firstTeamProduction + p.secondTeamProduction)
  let thirdTeamProduction := p.thirdTeamRate * p.deadline
  let fourthTeamProduction := remainingPieces - thirdTeamProduction
  p.thirdTeamRate + (fourthTeamProduction / p.deadline)

/-- Theorem stating that the combined production rate is 90 pieces per hour -/
theorem combined_rate_is_90 (p : FishFilletProduction)
    (h1 : p.totalRequired = 500)
    (h2 : p.deadline = 2)
    (h3 : p.firstTeamProduction = 189)
    (h4 : p.secondTeamProduction = 131)
    (h5 : p.thirdTeamRate = 45) :
    combinedRate p = 90 := by
  sorry

#eval combinedRate {
  totalRequired := 500,
  deadline := 2,
  firstTeamProduction := 189,
  secondTeamProduction := 131,
  thirdTeamRate := 45
}

end NUMINAMATH_CALUDE_combined_rate_is_90_l2854_285408


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l2854_285412

/-- A quadrilateral with an inscribed circle. -/
structure InscribedCircleQuadrilateral where
  /-- The radius of the inscribed circle. -/
  r : ℝ
  /-- The length of AP. -/
  ap : ℝ
  /-- The length of PB. -/
  pb : ℝ
  /-- The length of CQ. -/
  cq : ℝ
  /-- The length of QD. -/
  qd : ℝ
  /-- The circle is tangent to AB at P and to CD at Q. -/
  tangent_condition : True

/-- The theorem stating that for the given quadrilateral with inscribed circle,
    the square of the radius is 647. -/
theorem inscribed_circle_radius_squared
  (quad : InscribedCircleQuadrilateral)
  (h1 : quad.ap = 19)
  (h2 : quad.pb = 26)
  (h3 : quad.cq = 37)
  (h4 : quad.qd = 23) :
  quad.r ^ 2 = 647 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l2854_285412


namespace NUMINAMATH_CALUDE_symmetric_line_problem_l2854_285439

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to y = x -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (b, a, c)

/-- The equation of the line symmetric to 3x-5y+1=0 with respect to y=x -/
theorem symmetric_line_problem : 
  symmetric_line 3 (-5) 1 = (5, -3, -1) := by sorry

end NUMINAMATH_CALUDE_symmetric_line_problem_l2854_285439


namespace NUMINAMATH_CALUDE_sameColorPairsTheorem_l2854_285419

/-- The number of ways to choose a pair of socks of the same color -/
def sameColorPairs (total white brown green : ℕ) : ℕ :=
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose green 2

/-- Theorem: The number of ways to choose a pair of socks of the same color
    from 12 distinguishable socks (5 white, 5 brown, and 2 green) is 21 -/
theorem sameColorPairsTheorem :
  sameColorPairs 12 5 5 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sameColorPairsTheorem_l2854_285419


namespace NUMINAMATH_CALUDE_orange_harvest_l2854_285455

theorem orange_harvest (total_days : ℕ) (total_sacks : ℕ) (h1 : total_days = 14) (h2 : total_sacks = 56) :
  total_sacks / total_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_l2854_285455


namespace NUMINAMATH_CALUDE_pascal_triangle_specific_number_l2854_285441

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The kth number in the nth row of Pascal's triangle -/
def pascal_number (n k : ℕ) : ℕ := Nat.choose n (k - 1)

theorem pascal_triangle_specific_number :
  pascal_number 50 10 = 2586948580 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_specific_number_l2854_285441


namespace NUMINAMATH_CALUDE_square_difference_fraction_l2854_285479

theorem square_difference_fraction (x y : ℚ) 
  (h1 : x + y = 9/17) (h2 : x - y = 1/51) : x^2 - y^2 = 1/289 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fraction_l2854_285479


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2854_285492

/-- Represents the number of students in each grade and the sample size for grade 10 -/
structure SchoolData where
  grade12 : ℕ
  grade11 : ℕ
  grade10 : ℕ
  sample10 : ℕ

/-- Calculates the total number of students sampled from the entire school using stratified sampling -/
def totalSampleSize (data : SchoolData) : ℕ :=
  (data.sample10 * (data.grade12 + data.grade11 + data.grade10)) / data.grade10

/-- Theorem stating that given the specific school data, the total sample size is 220 -/
theorem stratified_sample_size :
  let data := SchoolData.mk 700 700 800 80
  totalSampleSize data = 220 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l2854_285492


namespace NUMINAMATH_CALUDE_concurrency_condition_l2854_285472

/-- Triangle ABC with sides a, b, and c, where AD is an altitude, BE is an angle bisector, and CF is a median -/
structure Triangle :=
  (a b c : ℝ)
  (ad_is_altitude : Bool)
  (be_is_angle_bisector : Bool)
  (cf_is_median : Bool)

/-- The lines AD, BE, and CF are concurrent -/
def are_concurrent (t : Triangle) : Prop := sorry

/-- Theorem stating the condition for concurrency of AD, BE, and CF -/
theorem concurrency_condition (t : Triangle) : 
  are_concurrent t ↔ t.a^2 * (t.a - t.c) = (t.b^2 - t.c^2) * (t.a + t.c) :=
sorry

end NUMINAMATH_CALUDE_concurrency_condition_l2854_285472


namespace NUMINAMATH_CALUDE_triangle_side_length_l2854_285481

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  b = 2 * Real.sqrt 3 →
  a = 2 →
  B = π / 3 →  -- 60° in radians
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos B →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2854_285481


namespace NUMINAMATH_CALUDE_soda_cost_l2854_285422

/-- Proves that the cost of each soda is $0.87 given the total cost and the cost of sandwiches -/
theorem soda_cost (total_cost : ℚ) (sandwich_cost : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) :
  total_cost = 8.38 →
  sandwich_cost = 2.45 →
  num_sandwiches = 2 →
  num_sodas = 4 →
  (total_cost - num_sandwiches * sandwich_cost) / num_sodas = 0.87 := by
sorry

#eval (8.38 - 2 * 2.45) / 4

end NUMINAMATH_CALUDE_soda_cost_l2854_285422


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2854_285469

theorem price_decrease_percentage (base_price : ℝ) (regular_price : ℝ) (promotional_price : ℝ) : 
  regular_price = base_price * (1 + 0.25) ∧ 
  promotional_price = base_price →
  (regular_price - promotional_price) / regular_price = 0.20 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2854_285469


namespace NUMINAMATH_CALUDE_largest_number_problem_l2854_285482

theorem largest_number_problem (a b c : ℝ) :
  a < b ∧ b < c →
  a + b + c = 82 →
  c - b = 8 →
  b - a = 4 →
  c = 34 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2854_285482


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2854_285417

theorem arithmetic_mean_of_fractions :
  let a : ℚ := 3/8
  let b : ℚ := 5/9
  (a + b) / 2 = 67/144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2854_285417


namespace NUMINAMATH_CALUDE_min_value_of_objective_function_l2854_285415

-- Define the constraint region
def ConstraintRegion (x y : ℝ) : Prop :=
  2 * x - y ≥ 0 ∧ y ≥ x ∧ y ≥ -x + 2

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ := 2 * x + y

-- Theorem statement
theorem min_value_of_objective_function :
  ∃ (min_z : ℝ), min_z = 8/3 ∧
  (∀ (x y : ℝ), ConstraintRegion x y → ObjectiveFunction x y ≥ min_z) ∧
  (∃ (x y : ℝ), ConstraintRegion x y ∧ ObjectiveFunction x y = min_z) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_objective_function_l2854_285415


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_25_l2854_285407

/-- Calculates the net rate of pay for a driver given specific conditions -/
theorem driver_net_pay_rate (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (compensation_rate : ℝ) (fuel_cost : ℝ) : ℝ :=
  let total_distance := travel_time * speed
  let fuel_used := total_distance / fuel_efficiency
  let earnings := compensation_rate * total_distance
  let fuel_expense := fuel_cost * fuel_used
  let net_earnings := earnings - fuel_expense
  let net_rate := net_earnings / travel_time
  net_rate

/-- Proves that the driver's net rate of pay is $25 per hour under the given conditions -/
theorem driver_net_pay_is_25 : 
  driver_net_pay_rate 3 50 25 0.60 2.50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_25_l2854_285407


namespace NUMINAMATH_CALUDE_night_day_crew_ratio_l2854_285434

theorem night_day_crew_ratio (D N : ℝ) (h1 : D > 0) (h2 : N > 0) : 
  (D / (D + 3/4 * N) = 0.64) → (N / D = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_night_day_crew_ratio_l2854_285434


namespace NUMINAMATH_CALUDE_probability_sum_three_l2854_285405

/-- Represents the color of a ball --/
inductive BallColor
  | Red
  | Yellow
  | Blue

/-- Represents the score of a ball --/
def score (color : BallColor) : ℕ :=
  match color with
  | BallColor.Red => 1
  | BallColor.Yellow => 2
  | BallColor.Blue => 3

/-- The total number of balls in the bag --/
def totalBalls : ℕ := 6

/-- The number of possible outcomes when drawing two balls with replacement --/
def totalOutcomes : ℕ := totalBalls * totalBalls

/-- The number of favorable outcomes (sum of scores is 3) --/
def favorableOutcomes : ℕ := 12

/-- Theorem stating that the probability of drawing two balls with a sum of scores equal to 3 is 1/3 --/
theorem probability_sum_three (h : favorableOutcomes = 12) :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_three_l2854_285405


namespace NUMINAMATH_CALUDE_average_permutation_sum_l2854_285468

def permutation_sum (b : Fin 8 → Fin 8) : ℕ :=
  |b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (fun f => Function.Injective f)

theorem average_permutation_sum :
  (Finset.sum all_permutations permutation_sum) / all_permutations.card = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_permutation_sum_l2854_285468


namespace NUMINAMATH_CALUDE_slope_of_line_parallel_lines_solution_l2854_285421

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
theorem slope_of_line (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y, a * x + b * y + c = 0 ↔ y = (-a/b) * x + (-c/b) :=
  sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y, a * x + y - 4 = 0 ↔ x + (a + 3/2) * y + 2 = 0) → a = 1/2 :=
  sorry

end NUMINAMATH_CALUDE_slope_of_line_parallel_lines_solution_l2854_285421


namespace NUMINAMATH_CALUDE_juan_running_time_l2854_285411

theorem juan_running_time (distance : ℝ) (speed : ℝ) (h1 : distance = 250) (h2 : speed = 8) :
  distance / speed = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_time_l2854_285411


namespace NUMINAMATH_CALUDE_sum_of_products_l2854_285426

theorem sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2854_285426


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2854_285461

def arithmetic_sequence (a : ℕ → ℤ) := ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ := 
  n * (a 0 + a (n - 1)) / 2

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum3 : sum_arithmetic_sequence a 3 = 42)
  (h_sum6 : sum_arithmetic_sequence a 6 = 57) :
  (∀ n, a n = 20 - 3 * n) ∧ 
  (∀ n, n ≤ 6 → sum_arithmetic_sequence a n ≤ sum_arithmetic_sequence a 6) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2854_285461


namespace NUMINAMATH_CALUDE_worker_b_time_l2854_285420

/-- Given two workers A and B, where A takes 8 hours to complete a job,
    and together they take 4.8 hours, prove that B takes 12 hours alone. -/
theorem worker_b_time (time_a time_ab : ℝ) (time_a_pos : time_a > 0) (time_ab_pos : time_ab > 0)
  (h1 : time_a = 8) (h2 : time_ab = 4.8) : 
  ∃ time_b : ℝ, time_b > 0 ∧ 1 / time_a + 1 / time_b = 1 / time_ab ∧ time_b = 12 := by
  sorry

#check worker_b_time

end NUMINAMATH_CALUDE_worker_b_time_l2854_285420


namespace NUMINAMATH_CALUDE_correct_distance_equation_l2854_285459

/-- Represents the driving scenario described in the problem -/
structure DrivingScenario where
  rate_before_stop : ℝ
  rate_after_stop : ℝ
  stop_duration : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation describing the total distance traveled is correct -/
theorem correct_distance_equation (scenario : DrivingScenario)
  (h1 : scenario.rate_before_stop = 80)
  (h2 : scenario.rate_after_stop = 100)
  (h3 : scenario.stop_duration = 1/3)
  (h4 : scenario.total_distance = 250)
  (h5 : scenario.total_time = 3) :
  ∃ t : ℝ, scenario.rate_before_stop * t + scenario.rate_after_stop * (scenario.total_time - scenario.stop_duration - t) = scenario.total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_correct_distance_equation_l2854_285459


namespace NUMINAMATH_CALUDE_magnitude_of_one_minus_i_to_eighth_l2854_285484

theorem magnitude_of_one_minus_i_to_eighth : 
  Complex.abs ((1 - Complex.I) ^ 8) = 16 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_one_minus_i_to_eighth_l2854_285484


namespace NUMINAMATH_CALUDE_fraction_equality_l2854_285470

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 12)
  (h3 : s / t = 4) :
  t / q = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2854_285470


namespace NUMINAMATH_CALUDE_min_difference_theorem_l2854_285491

noncomputable def f (x : ℝ) : ℝ := Real.exp (4 * x - 1)

noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.log (2 * x)

theorem min_difference_theorem (m n : ℝ) (h : f m = g n) :
  (∀ m' n', f m' = g n' → n' - m' ≥ (1 + Real.log 2) / 4) ∧
  (∃ m₀ n₀, f m₀ = g n₀ ∧ n₀ - m₀ = (1 + Real.log 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_min_difference_theorem_l2854_285491


namespace NUMINAMATH_CALUDE_babysitting_earnings_proof_l2854_285436

/-- Calculates earnings from babysitting given net profit, lemonade stand revenue, and operating cost -/
def earnings_from_babysitting (net_profit : ℕ) (lemonade_revenue : ℕ) (operating_cost : ℕ) : ℕ :=
  (net_profit + operating_cost) - lemonade_revenue

/-- Proves that earnings from babysitting equal $31 given the specific values -/
theorem babysitting_earnings_proof :
  earnings_from_babysitting 44 47 34 = 31 := by
sorry

end NUMINAMATH_CALUDE_babysitting_earnings_proof_l2854_285436


namespace NUMINAMATH_CALUDE_h_expansion_count_h_expansion_10_l2854_285464

/-- Definition of H expansion sequence -/
def h_expansion_seq (n : ℕ) : ℕ :=
  2^n + 1

/-- Theorem: The number of items after n H expansions is 2^n + 1 -/
theorem h_expansion_count (n : ℕ) :
  h_expansion_seq n = 2^n + 1 :=
by sorry

/-- Corollary: After 10 H expansions, the sequence has 1025 items -/
theorem h_expansion_10 :
  h_expansion_seq 10 = 1025 :=
by sorry

end NUMINAMATH_CALUDE_h_expansion_count_h_expansion_10_l2854_285464


namespace NUMINAMATH_CALUDE_candy_bar_multiple_l2854_285458

/-- 
Given that Max sold 24 candy bars and Seth sold 78 candy bars,
and Seth sold 6 more candy bars than a certain multiple of Max's candy bars,
prove that the multiple is 3.
-/
theorem candy_bar_multiple : ∃ m : ℕ, m * 24 + 6 = 78 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_multiple_l2854_285458


namespace NUMINAMATH_CALUDE_equation_graph_is_axes_l2854_285488

/-- The set of points (x, y) satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of the x-axis and y-axis -/
def XYAxes : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_graph_is_axes : S = XYAxes := by
  sorry

end NUMINAMATH_CALUDE_equation_graph_is_axes_l2854_285488


namespace NUMINAMATH_CALUDE_max_display_sum_l2854_285432

def hour_sum (h : Nat) : Nat :=
  if h < 10 then h
  else if h < 20 then (h / 10) + (h % 10)
  else 2 + (h % 10)

def minute_sum (m : Nat) : Nat :=
  (m / 10) + (m % 10)

def display_sum (h m : Nat) : Nat :=
  hour_sum h + minute_sum m

theorem max_display_sum :
  (∀ h m, h < 24 → m < 60 → display_sum h m ≤ 24) ∧
  (∃ h m, h < 24 ∧ m < 60 ∧ display_sum h m = 24) :=
sorry

end NUMINAMATH_CALUDE_max_display_sum_l2854_285432


namespace NUMINAMATH_CALUDE_sum_of_digits_power_6_13_l2854_285429

def power_6_13 : ℕ := 6^13

def ones_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_of_digits_power_6_13 :
  ones_digit power_6_13 + tens_digit power_6_13 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_6_13_l2854_285429


namespace NUMINAMATH_CALUDE_max_k_inequality_l2854_285435

theorem max_k_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    Real.sqrt (x^2 + k*y^2) + Real.sqrt (y^2 + k*x^2) ≥ x + y + (k-1) * Real.sqrt (x*y)) ∧
  (∀ (k' : ℝ), k' > k → 
    ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      Real.sqrt (x^2 + k'*y^2) + Real.sqrt (y^2 + k'*x^2) < x + y + (k'-1) * Real.sqrt (x*y)) ∧
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_max_k_inequality_l2854_285435


namespace NUMINAMATH_CALUDE_two_left_movements_l2854_285424

-- Define the direction type
inductive Direction
| Left
| Right

-- Define a function to convert direction to sign
def directionToSign (d : Direction) : Int :=
  match d with
  | Direction.Left => -1
  | Direction.Right => 1

-- Define a single movement
def singleMovement (distance : ℝ) (direction : Direction) : ℝ :=
  (directionToSign direction : ℝ) * distance

-- Define the problem statement
theorem two_left_movements (distance : ℝ) :
  distance = 3 →
  (singleMovement distance Direction.Left + singleMovement distance Direction.Left) = -6 :=
by sorry

end NUMINAMATH_CALUDE_two_left_movements_l2854_285424


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_l2854_285493

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- First three terms increasing -/
def first_three_increasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_increasing_iff_first_three (a : ℕ → ℝ) :
  geometric_sequence a →
  (increasing_sequence a ↔ first_three_increasing a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_l2854_285493
