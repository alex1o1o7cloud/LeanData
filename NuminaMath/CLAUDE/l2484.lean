import Mathlib

namespace quentavious_gum_pieces_l2484_248465

/-- Calculates the number of gum pieces received in an exchange. -/
def gum_pieces_received (initial_nickels : ℕ) (gum_per_nickel : ℕ) (remaining_nickels : ℕ) : ℕ :=
  (initial_nickels - remaining_nickels) * gum_per_nickel

/-- Proves that Quentavious received 6 pieces of gum. -/
theorem quentavious_gum_pieces :
  gum_pieces_received 5 2 2 = 6 := by
  sorry

end quentavious_gum_pieces_l2484_248465


namespace triangle_max_area_l2484_248434

theorem triangle_max_area (a b c : Real) (A B C : Real) :
  C = π / 6 →
  a + b = 12 →
  0 < a ∧ 0 < b ∧ 0 < c →
  (∃ (S : Real), S = (1 / 2) * a * b * Real.sin C ∧
    ∀ (S' : Real), S' = (1 / 2) * a * b * Real.sin C → S' ≤ 9) :=
by sorry

end triangle_max_area_l2484_248434


namespace matching_color_probability_l2484_248427

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, red := 2, blue := 1, yellow := 0 }

/-- Clara's jelly bean distribution -/
def clara : JellyBeans :=
  { green := 3, red := 2, blue := 1, yellow := 2 }

/-- Calculates the probability of picking a specific color -/
def prob_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Theorem: The probability of Abe and Clara showing the same color is 11/40 -/
theorem matching_color_probability :
  (prob_color abe abe.green * prob_color clara clara.green) +
  (prob_color abe abe.red * prob_color clara clara.red) +
  (prob_color abe abe.blue * prob_color clara clara.blue) = 11 / 40 := by
  sorry

end matching_color_probability_l2484_248427


namespace min_value_expression_lower_bound_achievable_l2484_248462

theorem min_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := by
  sorry

theorem lower_bound_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 4 := by
  sorry

end min_value_expression_lower_bound_achievable_l2484_248462


namespace product_of_parts_l2484_248422

theorem product_of_parts (z : ℂ) : z = 1 - I → (z.re * z.im = -1) := by
  sorry

end product_of_parts_l2484_248422


namespace smallest_part_of_proportional_division_l2484_248467

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 24 := by
  sorry

end smallest_part_of_proportional_division_l2484_248467


namespace sebastian_grade_size_l2484_248490

/-- The number of students in a grade where a student is ranked both the 70th best and 70th worst -/
def num_students (rank_best : ℕ) (rank_worst : ℕ) : ℕ :=
  (rank_best - 1) + 1 + (rank_worst - 1)

/-- Theorem stating that if a student is ranked both the 70th best and 70th worst, 
    then there are 139 students in total -/
theorem sebastian_grade_size :
  num_students 70 70 = 139 := by
  sorry

end sebastian_grade_size_l2484_248490


namespace derivative_at_one_l2484_248445

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) :
  f' 1 = -2 := by
sorry

end derivative_at_one_l2484_248445


namespace area_of_triangle_PQR_l2484_248400

/-- Given two lines intersecting at point P(2,5) with slopes -1 and -2 respectively,
    and points Q and R on the x-axis, prove that the area of triangle PQR is 6.25 -/
theorem area_of_triangle_PQR : ∃ (Q R : ℝ × ℝ),
  let P : ℝ × ℝ := (2, 5)
  let slope_PQ : ℝ := -1
  let slope_PR : ℝ := -2
  Q.2 = 0 ∧ R.2 = 0 ∧
  (Q.1 - P.1) / (Q.2 - P.2) = slope_PQ ∧
  (R.1 - P.1) / (R.2 - P.2) = slope_PR ∧
  (1/2 : ℝ) * |Q.1 - R.1| * P.2 = 6.25 := by
sorry

end area_of_triangle_PQR_l2484_248400


namespace grasshopper_jump_distance_l2484_248463

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (total_jump : ℕ) 
  (h1 : frog_jump = 35)
  (h2 : total_jump = 66) :
  total_jump - frog_jump = 31 := by
sorry

end grasshopper_jump_distance_l2484_248463


namespace family_reunion_attendance_l2484_248448

/-- The number of people at a family reunion --/
def family_reunion (male_adults female_adults children : ℕ) : ℕ :=
  male_adults + female_adults + children

/-- Theorem: Given the conditions, the family reunion has 750 people --/
theorem family_reunion_attendance :
  ∀ (male_adults female_adults children : ℕ),
  male_adults = 100 →
  female_adults = male_adults + 50 →
  children = 2 * (male_adults + female_adults) →
  family_reunion male_adults female_adults children = 750 :=
by
  sorry

end family_reunion_attendance_l2484_248448


namespace system_solutions_l2484_248496

def equation1 (x y : ℝ) : Prop := (x + 2*y) * (x + 3*y) = x + y

def equation2 (x y : ℝ) : Prop := (2*x + y) * (3*x + y) = -99 * (x + y)

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (-14, 6), (-85/6, 35/6)}

theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set := by
  sorry

end system_solutions_l2484_248496


namespace distinct_ages_count_l2484_248401

def average_age : ℕ := 31
def standard_deviation : ℕ := 5

def lower_bound : ℕ := average_age - standard_deviation
def upper_bound : ℕ := average_age + standard_deviation

theorem distinct_ages_count : 
  (Finset.range (upper_bound - lower_bound + 1)).card = 11 := by
  sorry

end distinct_ages_count_l2484_248401


namespace teachers_not_adjacent_arrangements_l2484_248425

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The total number of people -/
def total_people : ℕ := num_teachers + num_students

/-- The number of arrangements of n elements taken r at a time -/
def arrangements (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where teachers are not adjacent -/
def arrangements_teachers_not_adjacent : ℕ := 
  arrangements num_students num_students * arrangements (num_students + 1) num_teachers

theorem teachers_not_adjacent_arrangements :
  arrangements_teachers_not_adjacent = 480 :=
by sorry

end teachers_not_adjacent_arrangements_l2484_248425


namespace four_digit_number_theorem_l2484_248475

/-- Represents a four-digit number ABCD --/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Checks if a four-digit number contains no nines --/
def no_nines (n : FourDigitNumber) : Prop :=
  ∀ d, d ∈ n.value.digits 10 → d ≠ 9

/-- Extracts the first two digits of a four-digit number --/
def first_two_digits (n : FourDigitNumber) : ℕ := n.value / 100

/-- Extracts the last two digits of a four-digit number --/
def last_two_digits (n : FourDigitNumber) : ℕ := n.value % 100

/-- Extracts the first digit of a four-digit number --/
def first_digit (n : FourDigitNumber) : ℕ := n.value / 1000

/-- Extracts the second digit of a four-digit number --/
def second_digit (n : FourDigitNumber) : ℕ := (n.value / 100) % 10

/-- Extracts the third digit of a four-digit number --/
def third_digit (n : FourDigitNumber) : ℕ := (n.value / 10) % 10

/-- Extracts the fourth digit of a four-digit number --/
def fourth_digit (n : FourDigitNumber) : ℕ := n.value % 10

/-- Checks if a quadratic equation ax² + bx + c = 0 has real roots --/
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem four_digit_number_theorem (n : FourDigitNumber) 
  (h_no_nines : no_nines n)
  (h_eq1 : has_real_roots (first_digit n : ℝ) (second_digit n : ℝ) (last_two_digits n : ℝ))
  (h_eq2 : has_real_roots (first_digit n : ℝ) ((n.value / 10) % 100 : ℝ) (fourth_digit n : ℝ))
  (h_eq3 : has_real_roots (first_two_digits n : ℝ) (third_digit n : ℝ) (fourth_digit n : ℝ))
  (h_leading : first_digit n ≠ 0 ∧ second_digit n ≠ 0) :
  n.value = 1710 ∨ n.value = 1810 := by
  sorry

end four_digit_number_theorem_l2484_248475


namespace simplify_polynomial_l2484_248444

theorem simplify_polynomial (x : ℝ) :
  2 - 4*x - 6*x^2 + 8 + 10*x - 12*x^2 - 14 + 16*x + 18*x^2 = 22*x - 4 := by
  sorry

end simplify_polynomial_l2484_248444


namespace sqrt_expression_equals_sqrt_three_l2484_248437

theorem sqrt_expression_equals_sqrt_three : 
  Real.sqrt 48 - 6 * Real.sqrt (1/3) - Real.sqrt 18 / Real.sqrt 6 = Real.sqrt 3 := by
  sorry

end sqrt_expression_equals_sqrt_three_l2484_248437


namespace robie_chocolates_l2484_248471

/-- Calculates the number of chocolate bags left after a series of purchases and giveaways. -/
def chocolates_left (initial_purchase : ℕ) (given_away : ℕ) (additional_purchase : ℕ) : ℕ :=
  initial_purchase - given_away + additional_purchase

/-- Proves that given the specific scenario, 4 bags of chocolates are left. -/
theorem robie_chocolates : chocolates_left 3 2 3 = 4 := by
  sorry

end robie_chocolates_l2484_248471


namespace school_boys_count_l2484_248438

theorem school_boys_count (girls : ℕ) (difference : ℕ) (boys : ℕ) : 
  girls = 635 → difference = 510 → boys = girls + difference → boys = 1145 := by
  sorry

end school_boys_count_l2484_248438


namespace broken_line_length_bound_l2484_248499

/-- A broken line is represented as a list of points -/
def BrokenLine := List ℝ × ℝ

/-- A rectangle is represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Predicate to check if a broken line is inside a rectangle -/
def isInside (bl : BrokenLine) (rect : Rectangle) : Prop := sorry

/-- Predicate to check if every line parallel to the sides of the rectangle
    intersects the broken line at most once -/
def intersectsAtMostOnce (bl : BrokenLine) (rect : Rectangle) : Prop := sorry

/-- Function to calculate the length of a broken line -/
def length (bl : BrokenLine) : ℝ := sorry

/-- Theorem: If a broken line is inside a rectangle and every line parallel to the sides
    of the rectangle intersects the broken line at most once, then the length of the
    broken line is less than the sum of the lengths of two adjacent sides of the rectangle -/
theorem broken_line_length_bound (bl : BrokenLine) (rect : Rectangle) :
  isInside bl rect →
  intersectsAtMostOnce bl rect →
  length bl < rect.width + rect.height := by
  sorry

end broken_line_length_bound_l2484_248499


namespace taxi_occupancy_l2484_248432

theorem taxi_occupancy (cars : Nat) (car_capacity : Nat) (vans : Nat) (van_capacity : Nat) 
  (taxis : Nat) (total_people : Nat) :
  cars = 3 → car_capacity = 4 → vans = 2 → van_capacity = 5 → taxis = 6 → total_people = 58 →
  ∃ (taxi_capacity : Nat), taxi_capacity = 6 ∧ 
    cars * car_capacity + vans * van_capacity + taxis * taxi_capacity = total_people :=
by sorry

end taxi_occupancy_l2484_248432


namespace binomial_coefficient_two_l2484_248476

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2484_248476


namespace correct_calculation_l2484_248481

theorem correct_calculation (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 := by
  sorry

end correct_calculation_l2484_248481


namespace number_representation_l2484_248488

/-- Represents a number in terms of millions, ten thousands, and thousands -/
structure NumberComposition :=
  (millions : ℕ)
  (ten_thousands : ℕ)
  (thousands : ℕ)

/-- Converts a NumberComposition to its standard integer representation -/
def to_standard (n : NumberComposition) : ℕ :=
  n.millions * 1000000 + n.ten_thousands * 10000 + n.thousands * 1000

/-- Converts a natural number to its representation in ten thousands -/
def to_ten_thousands (n : ℕ) : ℚ :=
  (n : ℚ) / 10000

theorem number_representation (n : NumberComposition) 
  (h : n = ⟨6, 3, 4⟩) : 
  to_standard n = 6034000 ∧ to_ten_thousands (to_standard n) = 603.4 := by
  sorry

end number_representation_l2484_248488


namespace marble_probability_l2484_248446

theorem marble_probability (b : ℕ) : 
  2 * (2 / (2 + b)) * (1 / (1 + b)) = 1/3 → b = 2 := by
  sorry

end marble_probability_l2484_248446


namespace electronic_components_production_ahead_of_schedule_l2484_248498

theorem electronic_components_production_ahead_of_schedule 
  (total_components : ℕ) 
  (planned_days : ℕ) 
  (additional_daily_production : ℕ) : 
  total_components = 15000 → 
  planned_days = 30 → 
  additional_daily_production = 250 → 
  (planned_days - (total_components / ((total_components / planned_days) + additional_daily_production))) = 10 := by
sorry

end electronic_components_production_ahead_of_schedule_l2484_248498


namespace quadratic_point_between_roots_l2484_248424

/-- Given a quadratic function y = x^2 + 2x + c with roots x₁ and x₂ (where x₁ < x₂),
    and a point P(m, n) on the graph, if n < 0, then x₁ < m < x₂. -/
theorem quadratic_point_between_roots
  (c : ℝ) (x₁ x₂ m n : ℝ)
  (h_roots : x₁ < x₂)
  (h_on_graph : n = m^2 + 2*m + c)
  (h_roots_def : x₁^2 + 2*x₁ + c = 0 ∧ x₂^2 + 2*x₂ + c = 0)
  (h_n_neg : n < 0) :
  x₁ < m ∧ m < x₂ :=
by sorry

end quadratic_point_between_roots_l2484_248424


namespace S_equals_T_l2484_248468

def S : Set ℤ := {x | ∃ n : ℤ, x = 2*n + 1}
def T : Set ℤ := {x | ∃ n : ℤ, x = 4*n + 1 ∨ x = 4*n - 1}

theorem S_equals_T : S = T := by sorry

end S_equals_T_l2484_248468


namespace exists_valid_strategy_l2484_248412

/-- Represents a strategy for distributing balls in boxes -/
structure Strategy where
  distribute : Fin 2018 → ℕ

/-- Represents the game setup and rules -/
structure Game where
  boxes : Fin 2018
  pairs : Fin 4032
  pairAssignment : Fin 4032 → Fin 2018 × Fin 2018

/-- Predicate to check if a strategy results in distinct ball counts -/
def isValidStrategy (g : Game) (s : Strategy) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → s.distribute i ≠ s.distribute j

/-- Theorem stating the existence of a valid strategy -/
theorem exists_valid_strategy (g : Game) : ∃ s : Strategy, isValidStrategy g s := by
  sorry


end exists_valid_strategy_l2484_248412


namespace money_left_after_purchase_l2484_248435

def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def initial_dimes : ℕ := 19
def initial_quarters : ℕ := 6

def candy_bars_bought : ℕ := 4
def dimes_per_candy : ℕ := 3

def lollipops_bought : ℕ := 1

theorem money_left_after_purchase : 
  (initial_dimes * dime_value + initial_quarters * quarter_value) - 
  (candy_bars_bought * dimes_per_candy * dime_value + lollipops_bought * quarter_value) = 195 := by
sorry

end money_left_after_purchase_l2484_248435


namespace min_coefficient_value_l2484_248436

theorem min_coefficient_value (a b c d : ℤ) :
  (∃ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40) →
  (∃ (min_box : ℤ), 
    (∃ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40 ∧ box ≥ min_box) ∧
    (∀ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40 → box ≥ min_box) ∧
    min_box = 89) :=
by sorry


end min_coefficient_value_l2484_248436


namespace darcie_age_ratio_l2484_248469

theorem darcie_age_ratio (darcie_age mother_age father_age : ℕ) :
  darcie_age = 4 →
  mother_age = (4 * father_age) / 5 →
  father_age = 30 →
  darcie_age * 6 = mother_age :=
by
  sorry

end darcie_age_ratio_l2484_248469


namespace square_division_perimeters_l2484_248447

theorem square_division_perimeters (p : ℚ) : 
  (∃ a b c d e f : ℚ, 
    a + b + c = 1 ∧ 
    d + e + f = 1 ∧ 
    2 * (a + d) = p ∧ 
    2 * (b + e) = p ∧ 
    2 * (c + f) = p) → 
  (p = 8/3 ∨ p = 5/2) :=
by sorry

end square_division_perimeters_l2484_248447


namespace bug_return_probability_l2484_248443

/-- Probability of the bug being at the starting corner after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting corner on the eighth move -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end bug_return_probability_l2484_248443


namespace two_visit_days_365_l2484_248486

def alice_visits (d : ℕ) : Bool := d % 4 = 0
def bianca_visits (d : ℕ) : Bool := d % 6 = 0
def carmen_visits (d : ℕ) : Bool := d % 8 = 0

def exactly_two_visit (d : ℕ) : Bool :=
  let visit_count := (alice_visits d).toNat + (bianca_visits d).toNat + (carmen_visits d).toNat
  visit_count = 2

def count_two_visit_days (n : ℕ) : ℕ :=
  (List.range n).filter exactly_two_visit |>.length

theorem two_visit_days_365 :
  count_two_visit_days 365 = 45 := by
  sorry

end two_visit_days_365_l2484_248486


namespace athlete_c_most_suitable_l2484_248405

/-- Represents an athlete with their mean jump distance and variance --/
structure Athlete where
  name : String
  mean : ℝ
  variance : ℝ

/-- Determines if one athlete is more suitable than another --/
def moreSuitable (a b : Athlete) : Prop :=
  (a.mean > b.mean) ∨ (a.mean = b.mean ∧ a.variance < b.variance)

/-- Determines if an athlete is the most suitable among a list of athletes --/
def mostSuitable (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, a ≠ b → moreSuitable a b

theorem athlete_c_most_suitable :
  let athletes := [
    Athlete.mk "A" 380 12.5,
    Athlete.mk "B" 360 13.5,
    Athlete.mk "C" 380 2.4,
    Athlete.mk "D" 350 2.7
  ]
  let c := Athlete.mk "C" 380 2.4
  mostSuitable c athletes := by
  sorry

end athlete_c_most_suitable_l2484_248405


namespace regular_octagon_interior_angle_measure_l2484_248442

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def polygon_interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = 
    (polygon_interior_angle_sum octagon_sides) / octagon_sides :=
by sorry

end regular_octagon_interior_angle_measure_l2484_248442


namespace equation_solution_l2484_248483

theorem equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ * (x₁ - 2) + x₁ - 2 = 0) ∧ 
  (x₂ * (x₂ - 2) + x₂ - 2 = 0) ∧ 
  x₁ = 2 ∧ x₂ = -1 ∧ 
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 → (x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l2484_248483


namespace companion_pair_expression_l2484_248482

/-- Definition of a companion pair -/
def is_companion_pair (m n : ℝ) : Prop :=
  m / 2 + n / 3 = (m + n) / 5

/-- Theorem: For any companion pair (m, n), the expression 
    m - (22/3)n - [4m - 2(3n - 1)] equals -2 -/
theorem companion_pair_expression (m n : ℝ) 
  (h : is_companion_pair m n) : 
  m - (22/3) * n - (4 * m - 2 * (3 * n - 1)) = -2 := by
  sorry

end companion_pair_expression_l2484_248482


namespace set_union_problem_l2484_248452

def M (a : ℕ) : Set ℕ := {3, 4^a}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem set_union_problem (a b : ℕ) :
  M a ∩ N a b = {1} → M a ∪ N a b = {1, 2, 3} := by
  sorry

end set_union_problem_l2484_248452


namespace cooper_remaining_pies_l2484_248430

/-- The number of apple pies Cooper makes per day -/
def pies_per_day : ℕ := 7

/-- The number of days Cooper makes pies -/
def days_making_pies : ℕ := 12

/-- The number of pies Ashley eats -/
def pies_eaten : ℕ := 50

/-- The number of pies remaining with Cooper -/
def remaining_pies : ℕ := pies_per_day * days_making_pies - pies_eaten

theorem cooper_remaining_pies : remaining_pies = 34 := by sorry

end cooper_remaining_pies_l2484_248430


namespace smaller_circle_radius_l2484_248410

theorem smaller_circle_radius (R : ℝ) (r : ℝ) :
  R = 10 → -- Radius of the larger circle is 10 meters
  (4 * (2 * r) = 2 * R) → -- Four diameters of smaller circles span the diameter of the larger circle
  r = 2.5 := by sorry

end smaller_circle_radius_l2484_248410


namespace crackers_eaten_equals_180_l2484_248403

/-- Calculates the total number of animal crackers eaten by Mrs. Gable's students -/
def total_crackers_eaten (total_students : ℕ) (students_not_eating : ℕ) (crackers_per_pack : ℕ) : ℕ :=
  (total_students - students_not_eating) * crackers_per_pack

/-- Proves that the total number of animal crackers eaten is 180 -/
theorem crackers_eaten_equals_180 :
  total_crackers_eaten 20 2 10 = 180 := by
  sorry

#eval total_crackers_eaten 20 2 10

end crackers_eaten_equals_180_l2484_248403


namespace sallys_nickels_from_dad_l2484_248426

/-- The number of nickels Sally's dad gave her -/
def dads_nickels (initial_nickels mother_nickels total_nickels : ℕ) : ℕ :=
  total_nickels - (initial_nickels + mother_nickels)

/-- Proof that Sally's dad gave her 9 nickels -/
theorem sallys_nickels_from_dad :
  dads_nickels 7 2 18 = 9 := by
  sorry

end sallys_nickels_from_dad_l2484_248426


namespace solve_for_q_l2484_248456

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/60)
  (h2 : 5/6 = (m+n)/90)
  (h3 : 5/6 = (q-m)/150) : q = 150 := by sorry

end solve_for_q_l2484_248456


namespace geese_in_marsh_l2484_248449

theorem geese_in_marsh (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end geese_in_marsh_l2484_248449


namespace absolute_value_inequality_l2484_248497

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ((-4 ≤ x ∧ x ≤ -1) ∨ (3 ≤ x ∧ x ≤ 6)) := by
  sorry

end absolute_value_inequality_l2484_248497


namespace power_tower_mod_1000_l2484_248402

theorem power_tower_mod_1000 : 3^(3^(3^3)) ≡ 387 [ZMOD 1000] := by sorry

end power_tower_mod_1000_l2484_248402


namespace adam_apples_proof_l2484_248407

def monday_apples : ℕ := 15
def tuesday_multiplier : ℕ := 3
def wednesday_multiplier : ℕ := 4

def total_apples : ℕ := 
  monday_apples + 
  (tuesday_multiplier * monday_apples) + 
  (wednesday_multiplier * tuesday_multiplier * monday_apples)

theorem adam_apples_proof : total_apples = 240 := by
  sorry

end adam_apples_proof_l2484_248407


namespace rain_probability_l2484_248451

theorem rain_probability (p_rain : ℝ) (p_consecutive : ℝ) 
  (h1 : p_rain = 1/3)
  (h2 : p_consecutive = 1/5) :
  p_consecutive / p_rain = 3/5 := by
sorry

end rain_probability_l2484_248451


namespace z_properties_l2484_248459

/-- Complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 2)

/-- Condition for z to be purely imaginary -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Condition for z to lie on the line x + 2y + 1 = 0 -/
def on_line (z : ℂ) : Prop := z.re + 2 * z.im + 1 = 0

theorem z_properties (a : ℝ) :
  (is_purely_imaginary (z a) → a = 2) ∧
  (on_line (z a) → a = -1) := by sorry

end z_properties_l2484_248459


namespace probability_two_slate_rocks_l2484_248458

/-- The probability of selecting two slate rocks from a field with given rock counts -/
theorem probability_two_slate_rocks (slate_count pumice_count granite_count : ℕ) :
  slate_count = 12 →
  pumice_count = 16 →
  granite_count = 8 →
  let total_count := slate_count + pumice_count + granite_count
  (slate_count : ℚ) / total_count * ((slate_count - 1) : ℚ) / (total_count - 1) = 11 / 105 :=
by sorry

end probability_two_slate_rocks_l2484_248458


namespace sequence_general_term_l2484_248409

/-- Given a sequence {aₙ} where a₁ = 1 and aₙ₊₁ - aₙ = 2ⁿ for all n ≥ 1,
    prove that the general term is given by aₙ = 2ⁿ - 1 -/
theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n) : 
    ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 := by
  sorry

end sequence_general_term_l2484_248409


namespace smallest_prime_divisor_of_sum_l2484_248414

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^15 + 11^21) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^15 + 11^21) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l2484_248414


namespace garden_yield_calculation_l2484_248411

/-- Represents the dimensions of a garden section in steps -/
structure GardenSection where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield for an L-shaped garden -/
def expected_potato_yield (section1 : GardenSection) (section2 : GardenSection) 
    (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let area1 := (section1.length * section1.width * step_length ^ 2 : ℝ)
  let area2 := (section2.length * section2.width * step_length ^ 2 : ℝ)
  (area1 + area2) * yield_per_sqft

/-- Theorem stating the expected potato yield for the given garden -/
theorem garden_yield_calculation :
  let section1 : GardenSection := { length := 10, width := 25 }
  let section2 : GardenSection := { length := 10, width := 10 }
  let step_length : ℝ := 1.5
  let yield_per_sqft : ℝ := 0.75
  expected_potato_yield section1 section2 step_length yield_per_sqft = 590.625 := by
  sorry

end garden_yield_calculation_l2484_248411


namespace sum_of_roots_quadratic_l2484_248492

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∀ x, f x = 0 ↔ (x - sum_of_roots / 2)^2 = (sum_of_roots^2 - 4 * (b^2 - 4*a*c) / (4*a)) / 4) →
  sum_of_roots = 5 ↔ a = 1 ∧ b = -5 ∧ c = 6 :=
by sorry

end sum_of_roots_quadratic_l2484_248492


namespace distance_at_speed1_proof_l2484_248457

-- Define the total distance
def total_distance : ℝ := 250

-- Define the two speeds
def speed1 : ℝ := 40
def speed2 : ℝ := 60

-- Define the total time
def total_time : ℝ := 5.2

-- Define the distance covered at speed1 (40 kmph)
def distance_at_speed1 : ℝ := 124

-- Theorem statement
theorem distance_at_speed1_proof :
  let distance_at_speed2 := total_distance - distance_at_speed1
  (distance_at_speed1 / speed1) + (distance_at_speed2 / speed2) = total_time :=
by sorry

end distance_at_speed1_proof_l2484_248457


namespace evening_campers_count_l2484_248440

def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def total_campers : ℕ := 98

theorem evening_campers_count : 
  total_campers - (morning_campers + afternoon_campers) = 49 := by
  sorry

end evening_campers_count_l2484_248440


namespace rhombus_diagonal_l2484_248470

/-- A rhombus with given perimeter and one diagonal -/
structure Rhombus where
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem: In a rhombus with perimeter 52 and one diagonal 10, the other diagonal is 24 -/
theorem rhombus_diagonal (r : Rhombus) (h1 : r.perimeter = 52) (h2 : r.diagonal2 = 10) :
  ∃ (diagonal1 : ℝ), diagonal1 = 24 := by
  sorry


end rhombus_diagonal_l2484_248470


namespace license_plate_count_l2484_248478

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 6

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 20

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The number of special characters available for the license plate. -/
def num_special_chars : ℕ := 2

/-- The total number of possible license plates. -/
def total_license_plates : ℕ := num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

theorem license_plate_count : total_license_plates = 48000 := by
  sorry

end license_plate_count_l2484_248478


namespace max_value_quadratic_function_l2484_248489

theorem max_value_quadratic_function (f : ℝ → ℝ) (h : ∀ x ∈ (Set.Ioo 0 1), f x = x * (1 - x)) :
  ∃ x ∈ (Set.Ioo 0 1), ∀ y ∈ (Set.Ioo 0 1), f x ≥ f y ∧ f x = 1/4 :=
sorry

end max_value_quadratic_function_l2484_248489


namespace sqrt_equation_solution_l2484_248418

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_equation_solution_l2484_248418


namespace lana_extra_tickets_l2484_248429

/-- Calculates the number of extra tickets bought given the ticket price, number of tickets for friends, and total amount spent. -/
def extra_tickets (ticket_price : ℕ) (friends_tickets : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - friends_tickets * ticket_price) / ticket_price

/-- Proves that Lana bought 2 extra tickets given the problem conditions. -/
theorem lana_extra_tickets :
  let ticket_price : ℕ := 6
  let friends_tickets : ℕ := 8
  let total_spent : ℕ := 60
  extra_tickets ticket_price friends_tickets total_spent = 2 := by
  sorry

end lana_extra_tickets_l2484_248429


namespace negation_of_implication_l2484_248480

theorem negation_of_implication (a b c : ℝ) :
  ¬(a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) := by
  sorry

end negation_of_implication_l2484_248480


namespace normal_distribution_equality_l2484_248453

-- Define the random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the normal distribution parameters
variable (μ σ : ℝ)

-- Define the probability measure
variable (P : Set ℝ → ℝ)

-- State the theorem
theorem normal_distribution_equality (h1 : μ = 2) 
  (h2 : P {x | ξ x ≤ 4 - a} = P {x | ξ x ≥ 2 + 3 * a}) : a = -1 := by
  sorry

end normal_distribution_equality_l2484_248453


namespace inverse_mod_53_l2484_248487

theorem inverse_mod_53 (h : (15⁻¹ : ZMod 53) = 31) : (38⁻¹ : ZMod 53) = 22 := by
  sorry

end inverse_mod_53_l2484_248487


namespace absent_children_l2484_248413

theorem absent_children (total_children : ℕ) (bananas : ℕ) (absent : ℕ) : 
  total_children = 840 →
  bananas = 840 * 2 →
  bananas = (840 - absent) * 4 →
  absent = 420 := by
sorry

end absent_children_l2484_248413


namespace beatrix_pages_l2484_248485

theorem beatrix_pages (beatrix cristobal : ℕ) 
  (h1 : cristobal = 3 * beatrix + 15)
  (h2 : cristobal = beatrix + 1423) : 
  beatrix = 704 := by
sorry

end beatrix_pages_l2484_248485


namespace positive_X_value_l2484_248421

-- Define the # operation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- State the theorem
theorem positive_X_value :
  ∃ X : ℝ, X > 0 ∧ hash X 7 = 85 ∧ X = 6 := by
  sorry

end positive_X_value_l2484_248421


namespace class_average_weight_l2484_248415

theorem class_average_weight (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ) :
  students_a = 50 →
  students_b = 50 →
  avg_weight_a = 60 →
  avg_weight_b = 80 →
  (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = 70 :=
by sorry

end class_average_weight_l2484_248415


namespace total_after_discount_rounded_l2484_248472

-- Define the purchases
def purchase1 : ℚ := 215 / 100
def purchase2 : ℚ := 749 / 100
def purchase3 : ℚ := 1285 / 100

-- Define the discount rate
def discount_rate : ℚ := 1 / 10

-- Function to apply discount to the most expensive item
def apply_discount (p1 p2 p3 : ℚ) (rate : ℚ) : ℚ :=
  let max_purchase := max p1 (max p2 p3)
  let discounted_max := max_purchase * (1 - rate)
  if p1 == max_purchase then discounted_max + p2 + p3
  else if p2 == max_purchase then p1 + discounted_max + p3
  else p1 + p2 + discounted_max

-- Function to round to nearest integer
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

-- Theorem statement
theorem total_after_discount_rounded :
  round_to_nearest (apply_discount purchase1 purchase2 purchase3 discount_rate) = 21 := by
  sorry

end total_after_discount_rounded_l2484_248472


namespace smallest_prime_for_divisibility_l2484_248479

theorem smallest_prime_for_divisibility : ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  (11002 + p) % 11 = 0 ∧ 
  (11002 + p) % 7 = 0 ∧
  ∀ (q : ℕ), Nat.Prime q → (11002 + q) % 11 = 0 → (11002 + q) % 7 = 0 → p ≤ q :=
by
  -- The proof would go here
  sorry

end smallest_prime_for_divisibility_l2484_248479


namespace fraction_equality_l2484_248460

theorem fraction_equality (p q : ℝ) (h : (p⁻¹ + q⁻¹) / (p⁻¹ - q⁻¹) = 1009) :
  (p + q) / (p - q) = -1009 := by
  sorry

end fraction_equality_l2484_248460


namespace triangle_cosC_l2484_248464

theorem triangle_cosC (A B C : Real) (a b c : Real) : 
  -- Conditions
  (a = 2) →
  (b = 3) →
  (C = 2 * A) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Law of cosines
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Conclusion
  Real.cos C = 1/4 := by sorry

end triangle_cosC_l2484_248464


namespace sum_of_integers_l2484_248419

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 180) : 
  x.val + y.val = 28 := by
  sorry

end sum_of_integers_l2484_248419


namespace least_subtraction_for_divisibility_problem_solution_l2484_248491

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 5 ∧ (3830 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (3830 - m) % 15 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l2484_248491


namespace modulus_of_complex_l2484_248417

theorem modulus_of_complex (m : ℝ) : 
  let z : ℂ := Complex.mk (m - 2) (m + 1)
  Complex.abs z = Real.sqrt (2 * m^2 - 2 * m + 5) := by sorry

end modulus_of_complex_l2484_248417


namespace kids_total_savings_l2484_248439

-- Define the conversion rate
def pound_to_dollar : ℝ := 1.38

-- Define the savings for each child
def teagan_savings : ℝ := 200 * 0.01 + 15 * 1.00
def rex_savings : ℝ := 100 * 0.05 + 45 * 0.25 + 8 * pound_to_dollar
def toni_savings : ℝ := 330 * 0.10 + 12 * 5.00

-- Define the total savings
def total_savings : ℝ := teagan_savings + rex_savings + toni_savings

-- Theorem statement
theorem kids_total_savings : total_savings = 137.29 := by
  sorry

end kids_total_savings_l2484_248439


namespace ace_distribution_probability_l2484_248493

def num_players : ℕ := 4
def num_cards : ℕ := 32
def num_aces : ℕ := 4
def cards_per_player : ℕ := num_cards / num_players

theorem ace_distribution_probability :
  let remaining_players := num_players - 1
  let remaining_cards := num_cards - cards_per_player
  let p_no_ace_for_one := 1 / num_players
  let p_two_aces_for_others := 
    (Nat.choose remaining_players 1 * Nat.choose num_aces 2 * Nat.choose (remaining_cards - num_aces) (cards_per_player - 2)) /
    (Nat.choose remaining_cards cards_per_player)
  p_two_aces_for_others = 8 / 11 :=
sorry

end ace_distribution_probability_l2484_248493


namespace complex_number_properties_l2484_248431

theorem complex_number_properties (z : ℂ) (h : z * (1 + Complex.I) = 2) :
  (Complex.abs z = Real.sqrt 2) ∧
  (∀ p : ℝ, z^2 - p*z + 2 = 0 → p = 2) := by
  sorry

end complex_number_properties_l2484_248431


namespace exterior_angle_not_sum_of_adjacent_angles_l2484_248494

-- Define a triangle with interior angles A, B, C and exterior angle A_ext at vertex A
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  A_ext : ℝ

-- State the theorem
theorem exterior_angle_not_sum_of_adjacent_angles (t : Triangle) : 
  t.A_ext ≠ t.B + t.C :=
sorry

end exterior_angle_not_sum_of_adjacent_angles_l2484_248494


namespace beckys_necklace_count_l2484_248466

/-- Calculates the final number of necklaces in Becky's collection -/
def final_necklace_count (initial : ℕ) (broken : ℕ) (new : ℕ) (gifted : ℕ) : ℕ :=
  initial - broken + new - gifted

/-- Theorem stating that Becky's final necklace count is 37 -/
theorem beckys_necklace_count :
  final_necklace_count 50 3 5 15 = 37 := by
  sorry

end beckys_necklace_count_l2484_248466


namespace complementary_angles_equal_l2484_248416

/-- Two angles that are complementary to the same angle are equal. -/
theorem complementary_angles_equal (α β γ : Real) (h1 : α + γ = 90) (h2 : β + γ = 90) : α = β := by
  sorry

end complementary_angles_equal_l2484_248416


namespace x_intercept_of_line_l2484_248455

/-- The x-intercept of the line 2x + y - 2 = 0 is at x = 1 -/
theorem x_intercept_of_line (x y : ℝ) : 2*x + y - 2 = 0 → y = 0 → x = 1 := by
  sorry

end x_intercept_of_line_l2484_248455


namespace period_length_l2484_248420

theorem period_length 
  (total_duration : ℕ) 
  (num_periods : ℕ) 
  (break_duration : ℕ) 
  (num_breaks : ℕ) :
  total_duration = 220 →
  num_periods = 5 →
  break_duration = 5 →
  num_breaks = 4 →
  (total_duration - num_breaks * break_duration) / num_periods = 40 :=
by sorry

end period_length_l2484_248420


namespace smallest_k_for_64_power_gt_4_power_20_l2484_248477

theorem smallest_k_for_64_power_gt_4_power_20 : ∃ k : ℕ, k = 7 ∧ (∀ m : ℕ, 64^m > 4^20 → m ≥ k) := by
  sorry

end smallest_k_for_64_power_gt_4_power_20_l2484_248477


namespace three_digit_prime_discriminant_not_square_l2484_248450

theorem three_digit_prime_discriminant_not_square (A B C : ℕ) : 
  (100 * A + 10 * B + C).Prime → 
  ¬∃ (n : ℤ), B^2 - 4*A*C = n^2 := by
sorry

end three_digit_prime_discriminant_not_square_l2484_248450


namespace range_of_a_l2484_248428

def A (a : ℝ) : Set ℝ := {x | |x - 1| ≤ a ∧ a > 0}

def B : Set ℝ := {x | x^2 - 6*x - 7 > 0}

theorem range_of_a (a : ℝ) :
  (A a ∩ B = ∅) → (0 < a ∧ a ≤ 2) :=
by sorry

end range_of_a_l2484_248428


namespace sons_age_l2484_248495

theorem sons_age (father_age : ℕ) (h1 : father_age = 38) : ℕ :=
  let son_age := 14
  let years_ago := 10
  have h2 : father_age - years_ago = 7 * (son_age - years_ago) := by sorry
  son_age

#check sons_age

end sons_age_l2484_248495


namespace no_real_solution_l2484_248441

theorem no_real_solution : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + 1/y = 5 ∧ y + 1/x = 1/3 := by
  sorry

end no_real_solution_l2484_248441


namespace paige_mp3_songs_l2484_248423

/-- Calculates the final number of songs on an mp3 player after deleting and adding songs. -/
def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

/-- Theorem: The final number of songs on Paige's mp3 player is 10. -/
theorem paige_mp3_songs : final_song_count 11 9 8 = 10 := by
  sorry

end paige_mp3_songs_l2484_248423


namespace boat_speed_in_still_water_l2484_248408

/-- The speed of a boat in still water, given its speed with and against the stream -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ)
  (h1 : along_stream = 9)
  (h2 : against_stream = 5) :
  (along_stream + against_stream) / 2 = 7 := by
  sorry

end boat_speed_in_still_water_l2484_248408


namespace division_problem_l2484_248484

theorem division_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 5/3)
  (h3 : c / d = 2) :
  d / a = 1/10 := by
  sorry

end division_problem_l2484_248484


namespace first_discount_percentage_l2484_248454

/-- Proves that given an original price of $199.99999999999997, a final sale price of $144
    after two successive discounts, where the second discount is 20%,
    the first discount percentage is 10%. -/
theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 199.99999999999997)
  (h2 : final_price = 144)
  (h3 : second_discount = 0.2)
  : (original_price - final_price / (1 - second_discount)) / original_price = 0.1 :=
by sorry

end first_discount_percentage_l2484_248454


namespace division_problem_l2484_248404

theorem division_problem (x y z : ℚ) 
  (h1 : x / y = 3) 
  (h2 : y / z = 2 / 5) : 
  z / x = 5 / 6 := by
sorry

end division_problem_l2484_248404


namespace limit_sequence_equals_one_over_e_l2484_248406

theorem limit_sequence_equals_one_over_e :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((2*n - 1) / (2*n + 1))^(n + 1) - 1/Real.exp 1| < ε :=
sorry

end limit_sequence_equals_one_over_e_l2484_248406


namespace lloyd_house_of_cards_solution_l2484_248474

/-- Represents the number of cards in Lloyd's house of cards problem -/
def lloyd_house_of_cards (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) : ℕ :=
  (decks * cards_per_deck) / layers

/-- Theorem stating the number of cards per layer in Lloyd's house of cards -/
theorem lloyd_house_of_cards_solution :
  lloyd_house_of_cards 24 78 48 = 39 := by
  sorry

#eval lloyd_house_of_cards 24 78 48

end lloyd_house_of_cards_solution_l2484_248474


namespace ball_volume_ratio_l2484_248461

theorem ball_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) (h₃ : r₃ = 3 * r₁) :
  (4 / 3 * π * r₃^3) = 3 * ((4 / 3 * π * r₁^3) + (4 / 3 * π * r₂^3)) :=
by sorry

end ball_volume_ratio_l2484_248461


namespace no_winning_strategy_l2484_248473

/-- Represents a player in the game -/
inductive Player
| kezdo
| masodik

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 19
  col : Fin 19

/-- Represents a move in the game -/
structure Move where
  player : Player
  cell : Cell
  value : Fin 2

/-- Represents the state of the game after all moves -/
def GameState := List Move

/-- Calculates the sum of a row -/
def rowSum (state : GameState) (row : Fin 19) : Nat :=
  sorry

/-- Calculates the sum of a column -/
def colSum (state : GameState) (col : Fin 19) : Nat :=
  sorry

/-- Calculates the maximum row sum -/
def maxRowSum (state : GameState) : Nat :=
  sorry

/-- Calculates the maximum column sum -/
def maxColSum (state : GameState) : Nat :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Theorem: No winning strategy exists for either player -/
theorem no_winning_strategy :
  ∀ (kezdo_strategy : Strategy) (masodik_strategy : Strategy),
    ∃ (final_state : GameState),
      (maxRowSum final_state = maxColSum final_state) ∧
      (List.length final_state = 19 * 19) :=
sorry

end no_winning_strategy_l2484_248473


namespace volleyball_practice_start_time_l2484_248433

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define addition of minutes to Time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

theorem volleyball_practice_start_time 
  (start_time : Time) 
  (homework_duration : Nat) 
  (break_duration : Nat) : 
  start_time = { hour := 13, minute := 59 } → 
  homework_duration = 96 → 
  break_duration = 25 → 
  addMinutes (addMinutes start_time homework_duration) break_duration = { hour := 16, minute := 0 } :=
by
  sorry


end volleyball_practice_start_time_l2484_248433
