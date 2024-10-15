import Mathlib

namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3884_388474

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {x ∈ U | 2 < x ∧ x < 6}

-- Theorem statement
theorem complement_A_intersect_B : 
  (U \ A) ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3884_388474


namespace NUMINAMATH_CALUDE_parabola_c_value_l3884_388465

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem parabola_c_value :
  ∀ b c : ℝ,
  Parabola b c 1 = 6 →
  Parabola b c 5 = 10 →
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3884_388465


namespace NUMINAMATH_CALUDE_parabola_vertex_l3884_388435

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 6*y + 4*x - 7 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ x' y', parabola_equation x' y' → y' ≥ y

theorem parabola_vertex :
  is_vertex 4 (-3) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3884_388435


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3884_388439

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  (11 - 3*i) / (1 + 2*i) = 3 - 5*i := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3884_388439


namespace NUMINAMATH_CALUDE_f_increasing_l3884_388403

-- Define the function f(x) = x³ + x + 1
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Theorem statement
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l3884_388403


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3884_388433

-- Define the equation for which a and b are roots
def root_equation (m x : ℝ) : Prop :=
  m * (x^2 - 2*x) + 3*x + 7 = 0

-- Define the condition for m₁ and m₂
def m_condition (m : ℝ) (a b : ℝ) : Prop :=
  a / b + b / a = 7 / 10

theorem root_sum_theorem :
  ∀ (m₁ m₂ a b : ℝ),
  (∃ m, root_equation m a ∧ root_equation m b) →
  m_condition m₁ a b →
  m_condition m₂ a b →
  m₁ / m₂ + m₂ / m₁ = 253 / 36 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3884_388433


namespace NUMINAMATH_CALUDE_income_calculation_l3884_388434

/-- Proves that given an income to expenditure ratio of 7:6 and savings of 3000,
    the income is 21000. -/
theorem income_calculation (income expenditure savings : ℕ) : 
  income * 6 = expenditure * 7 →
  income - expenditure = savings →
  savings = 3000 →
  income = 21000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l3884_388434


namespace NUMINAMATH_CALUDE_unique_solution_system_l3884_388495

theorem unique_solution_system (s t : ℝ) : 
  15 * s + 10 * t = 270 ∧ s = 3 * t - 4 → s = 14 ∧ t = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3884_388495


namespace NUMINAMATH_CALUDE_sector_area_l3884_388491

/-- Given a circular sector with central angle α = 60° and arc length l = 6π,
    prove that the area of the sector is 54π. -/
theorem sector_area (α : Real) (l : Real) (h1 : α = 60 * π / 180) (h2 : l = 6 * π) :
  (1 / 2) * l * (l / α) = 54 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3884_388491


namespace NUMINAMATH_CALUDE_expression_evaluation_l3884_388493

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3884_388493


namespace NUMINAMATH_CALUDE_ticket_price_uniqueness_l3884_388450

theorem ticket_price_uniqueness : ∃! x : ℕ+, 
  (x : ℕ) ∣ 72 ∧ 
  (x : ℕ) ∣ 90 ∧ 
  1 ≤ 72 / (x : ℕ) ∧ 72 / (x : ℕ) ≤ 10 ∧
  1 ≤ 90 / (x : ℕ) ∧ 90 / (x : ℕ) ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_uniqueness_l3884_388450


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3884_388456

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 1 = (x^2 - 4*x + 7) * q + (8*x - 62) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3884_388456


namespace NUMINAMATH_CALUDE_number_equation_l3884_388487

theorem number_equation : ∃ x : ℝ, x / 1500 = 0.016833333333333332 ∧ x = 25.25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3884_388487


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3884_388406

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  n % 2 = 1 ∧ n % 3 = 2 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3884_388406


namespace NUMINAMATH_CALUDE_quadratic_coefficient_not_one_l3884_388401

/-- A quadratic equation in x is of the form px^2 + qx + r = 0 where p ≠ 0 -/
def is_quadratic_equation (p q r : ℝ) : Prop := p ≠ 0

theorem quadratic_coefficient_not_one (a : ℝ) :
  is_quadratic_equation (a - 1) (-1) 7 → a ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_not_one_l3884_388401


namespace NUMINAMATH_CALUDE_remainder_of_b_mod_29_l3884_388422

theorem remainder_of_b_mod_29 :
  let b := (((13⁻¹ : ZMod 29) + (17⁻¹ : ZMod 29) + (19⁻¹ : ZMod 29))⁻¹ : ZMod 29)
  b = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_b_mod_29_l3884_388422


namespace NUMINAMATH_CALUDE_prob_double_is_one_seventh_l3884_388414

/-- The number of integers in the modified domino set -/
def n : ℕ := 13

/-- The total number of domino pairings in the set -/
def total_pairings : ℕ := n * (n + 1) / 2

/-- The number of doubles in the set -/
def num_doubles : ℕ := n

/-- The probability of selecting a double from the modified domino set -/
def prob_double : ℚ := num_doubles / total_pairings

theorem prob_double_is_one_seventh : prob_double = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_double_is_one_seventh_l3884_388414


namespace NUMINAMATH_CALUDE_water_bottle_count_l3884_388438

theorem water_bottle_count (initial bottles_drunk bottles_bought : ℕ) 
  (h1 : initial = 42)
  (h2 : bottles_drunk = 25)
  (h3 : bottles_bought = 30) : 
  initial - bottles_drunk + bottles_bought = 47 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_count_l3884_388438


namespace NUMINAMATH_CALUDE_curve_points_difference_l3884_388429

theorem curve_points_difference (e a b : ℝ) : 
  e > 0 →
  (a^2 + e^2 = 2*e*a + 1) →
  (b^2 + e^2 = 2*e*b + 1) →
  a ≠ b →
  |a - b| = 2 := by
sorry

end NUMINAMATH_CALUDE_curve_points_difference_l3884_388429


namespace NUMINAMATH_CALUDE_circle_tangent_existence_l3884_388454

/-- A line in a 2D plane -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle2D where
  center : Point2D
  radius : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if a circle is tangent to a line at a point -/
def circleTangentToLineAtPoint (c : Circle2D) (l : Line2D) (p : Point2D) : Prop :=
  pointOnLine p l ∧
  (c.center.x - p.x) * l.slope + (c.center.y - p.y) = 0 ∧
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

theorem circle_tangent_existence
  (l : Line2D) (p : Point2D) (r : ℝ) 
  (h_positive : r > 0) 
  (h_on_line : pointOnLine p l) :
  ∃ (c1 c2 : Circle2D), 
    c1.radius = r ∧
    c2.radius = r ∧
    circleTangentToLineAtPoint c1 l p ∧
    circleTangentToLineAtPoint c2 l p ∧
    c1 ≠ c2 :=
  sorry

end NUMINAMATH_CALUDE_circle_tangent_existence_l3884_388454


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3884_388498

theorem stratified_sampling_theorem (elementary middle high : ℕ) 
  (h1 : elementary = 126) 
  (h2 : middle = 280) 
  (h3 : high = 95) 
  (sample_size : ℕ) 
  (h4 : sample_size = 100) : 
  ∃ (adjusted_elementary : ℕ), 
    adjusted_elementary = elementary - 1 ∧ 
    (adjusted_elementary + middle + high) % sample_size = 0 ∧
    (adjusted_elementary / 5 + middle / 5 + high / 5 = sample_size) :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3884_388498


namespace NUMINAMATH_CALUDE_paper_cutting_game_l3884_388428

theorem paper_cutting_game (n : ℕ) (pieces : ℕ) : 
  (pieces = 8 * n + 1) → (pieces = 2009) → (n = 251) := by
  sorry

end NUMINAMATH_CALUDE_paper_cutting_game_l3884_388428


namespace NUMINAMATH_CALUDE_trig_expression_value_quadratic_equation_solutions_quadratic_root_property_l3884_388421

-- Part 1
theorem trig_expression_value : 
  2 * Real.tan (60 * π / 180) * Real.cos (30 * π / 180) - Real.sin (45 * π / 180) ^ 2 = 5/2 := by
sorry

-- Part 2
theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x => 2 * (x + 2)^2 - 3 * (x + 2)
  ∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = -1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

-- Part 3
theorem quadratic_root_property :
  ∀ m : ℝ, m^2 - 5*m - 2 = 0 → 2*m^2 - 10*m + 2023 = 2027 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_value_quadratic_equation_solutions_quadratic_root_property_l3884_388421


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l3884_388426

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum_squares : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l3884_388426


namespace NUMINAMATH_CALUDE_samantha_birth_year_l3884_388490

-- Define the year of the first AMC 8
def first_amc_year : ℕ := 1980

-- Define the function to calculate the year of the nth AMC 8
def amc_year (n : ℕ) : ℕ := first_amc_year + n - 1

-- Define Samantha's age when she took the 9th AMC 8
def samantha_age_at_ninth_amc : ℕ := 14

-- Theorem to prove Samantha's birth year
theorem samantha_birth_year :
  amc_year 9 - samantha_age_at_ninth_amc = 1974 := by
  sorry


end NUMINAMATH_CALUDE_samantha_birth_year_l3884_388490


namespace NUMINAMATH_CALUDE_inequality_solution_l3884_388462

-- Define the function f
def f (x : ℝ) := x^2 - x - 6

-- State the theorem
theorem inequality_solution :
  (∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 3) →
  (∀ x : ℝ, -6 * f (-x) > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3884_388462


namespace NUMINAMATH_CALUDE_min_moves_ten_elements_l3884_388480

/-- Represents a circular arrangement of n distinct elements -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- A single move in the circular arrangement -/
def Move (n : ℕ) (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the arrangement is sorted in ascending order -/
def IsSorted (n : ℕ) (arr : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of moves required to sort the arrangement -/
def MinMoves (n : ℕ) (arr : CircularArrangement n) : ℕ :=
  sorry

/-- Theorem: The minimum number of moves to sort 10 distinct elements in a circle is 8 -/
theorem min_moves_ten_elements :
  ∀ (arr : CircularArrangement 10), MinMoves 10 arr = 8 :=
  sorry

end NUMINAMATH_CALUDE_min_moves_ten_elements_l3884_388480


namespace NUMINAMATH_CALUDE_smallest_square_side_is_14_l3884_388468

/-- The smallest side length of a square composed of equal numbers of unit squares with sides 1, 2, and 3 -/
def smallest_square_side : ℕ := 14

/-- Proposition: The smallest possible side length of a square composed of an equal number of squares with sides 1, 2, and 3 is 14 units -/
theorem smallest_square_side_is_14 :
  ∀ n : ℕ, n > 0 →
  ∃ s : ℕ, s * s = n * (1 * 1 + 2 * 2 + 3 * 3) →
  s ≥ smallest_square_side :=
sorry

end NUMINAMATH_CALUDE_smallest_square_side_is_14_l3884_388468


namespace NUMINAMATH_CALUDE_min_people_hat_glove_not_scarf_l3884_388477

theorem min_people_hat_glove_not_scarf (n : ℕ) 
  (gloves : ℕ) (hats : ℕ) (scarves : ℕ) :
  gloves = (3 * n) / 8 ∧ 
  hats = (5 * n) / 6 ∧ 
  scarves = n / 4 →
  ∃ (x : ℕ), x = hats + gloves - (n - scarves) ∧ 
  x ≥ 11 ∧ 
  (∀ (m : ℕ), m < n → 
    (3 * m) % 8 ≠ 0 ∨ (5 * m) % 6 ≠ 0 ∨ m % 4 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_min_people_hat_glove_not_scarf_l3884_388477


namespace NUMINAMATH_CALUDE_expression_bounds_l3884_388415

theorem expression_bounds (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  14/27 ≤ x^3 + 2*y^2 + (10/3)*z ∧ x^3 + 2*y^2 + (10/3)*z ≤ 10/3 := by
sorry

end NUMINAMATH_CALUDE_expression_bounds_l3884_388415


namespace NUMINAMATH_CALUDE_birds_on_fence_l3884_388453

/-- The number of additional birds that joined the fence -/
def additional_birds : ℕ := sorry

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 2

/-- The number of storks that joined the fence -/
def joined_storks : ℕ := 4

theorem birds_on_fence :
  additional_birds = 5 :=
by
  have h1 : initial_birds + additional_birds = joined_storks + 3 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3884_388453


namespace NUMINAMATH_CALUDE_system_solution_l3884_388425

theorem system_solution :
  ∃ (x y z : ℚ),
    (7 * x - 3 * y + 2 * z = 4) ∧
    (2 * x + 8 * y - z = 1) ∧
    (3 * x - 4 * y + 5 * z = 7) ∧
    (x = 1262 / 913) ∧
    (y = -59 / 83) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3884_388425


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3884_388486

theorem quadratic_roots_property (m n : ℝ) : 
  (∀ x, x^2 - 3*x + 1 = 0 ↔ x = m ∨ x = n) →
  -m - n - m*n = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3884_388486


namespace NUMINAMATH_CALUDE_parallelogram_height_l3884_388466

/-- Given a parallelogram with area 576 square cm and base 12 cm, its height is 48 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 ∧ base = 12 ∧ area = base * height → height = 48 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3884_388466


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l3884_388440

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^2 - 8)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l3884_388440


namespace NUMINAMATH_CALUDE_b_squared_is_zero_matrix_l3884_388471

theorem b_squared_is_zero_matrix (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_squared_is_zero_matrix_l3884_388471


namespace NUMINAMATH_CALUDE_coupon_probability_l3884_388423

def total_coupons : ℕ := 17
def semyon_coupons : ℕ := 9
def temyon_missing : ℕ := 6

theorem coupon_probability : 
  (Nat.choose temyon_missing temyon_missing * Nat.choose (total_coupons - temyon_missing) (semyon_coupons - temyon_missing)) / 
  Nat.choose total_coupons semyon_coupons = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_probability_l3884_388423


namespace NUMINAMATH_CALUDE_find_a_l3884_388475

theorem find_a (x y : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : ∃ a : ℝ, a * x - y = 3) : 
  ∃ a : ℝ, a = 5 ∧ a * x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l3884_388475


namespace NUMINAMATH_CALUDE_book_page_digit_sum_l3884_388441

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Sum of digits for all page numbers from 1 to n -/
def total_digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in page numbers of a 2000-page book is 28002 -/
theorem book_page_digit_sum :
  total_digit_sum 2000 = 28002 := by sorry

end NUMINAMATH_CALUDE_book_page_digit_sum_l3884_388441


namespace NUMINAMATH_CALUDE_scooter_repair_percentage_l3884_388416

theorem scooter_repair_percentage (profit_percentage : ℝ) (profit_amount : ℝ) (repair_cost : ℝ) :
  profit_percentage = 0.2 →
  profit_amount = 1100 →
  repair_cost = 500 →
  (repair_cost / (profit_amount / profit_percentage)) * 100 = 500 / 5500 * 100 := by
sorry

end NUMINAMATH_CALUDE_scooter_repair_percentage_l3884_388416


namespace NUMINAMATH_CALUDE_team_b_average_points_l3884_388446

theorem team_b_average_points (average_first_two : ℝ) : 
  (2 * average_first_two + 47 + 330 > 500) → average_first_two > 61.5 := by
  sorry

end NUMINAMATH_CALUDE_team_b_average_points_l3884_388446


namespace NUMINAMATH_CALUDE_squirrel_difference_l3884_388482

def scotland_squirrels : ℕ := 120000
def scotland_percentage : ℚ := 3/4

theorem squirrel_difference : 
  scotland_squirrels - (scotland_squirrels / scotland_percentage - scotland_squirrels) = 80000 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_difference_l3884_388482


namespace NUMINAMATH_CALUDE_zoo_tickets_cost_l3884_388499

/-- The total cost of zoo tickets for a group of children and adults. -/
def total_cost (num_children num_adults child_ticket_price adult_ticket_price : ℕ) : ℕ :=
  num_children * child_ticket_price + num_adults * adult_ticket_price

/-- Theorem: The total cost of zoo tickets for a group of 6 children and 10 adults is $220,
    given that child tickets cost $10 each and adult tickets cost $16 each. -/
theorem zoo_tickets_cost :
  total_cost 6 10 10 16 = 220 := by
sorry

end NUMINAMATH_CALUDE_zoo_tickets_cost_l3884_388499


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3884_388483

theorem no_solution_for_equation :
  ¬ ∃ (x : ℝ), (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3884_388483


namespace NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l3884_388444

/-- The number of cards Janessa keeps for herself --/
def cards_kept_by_janessa (initial_cards : ℕ) (cards_from_father : ℕ) (cards_ordered : ℕ) 
  (bad_cards : ℕ) (cards_given_to_dexter : ℕ) : ℕ :=
  initial_cards + cards_from_father + cards_ordered - bad_cards - cards_given_to_dexter

/-- Theorem stating that Janessa keeps 20 cards for herself --/
theorem janessa_keeps_twenty_cards : 
  cards_kept_by_janessa 4 13 36 4 29 = 20 := by
  sorry

end NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l3884_388444


namespace NUMINAMATH_CALUDE_max_candy_pieces_l3884_388464

theorem max_candy_pieces (n : ℕ) (mean : ℕ) (min_pieces : ℕ) :
  n = 25 →
  mean = 7 →
  min_pieces = 2 →
  ∃ (max_pieces : ℕ),
    max_pieces = n * mean - (n - 1) * min_pieces ∧
    max_pieces = 127 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l3884_388464


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3884_388448

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3884_388448


namespace NUMINAMATH_CALUDE_undeclared_major_fraction_l3884_388484

theorem undeclared_major_fraction (T : ℝ) (f : ℝ) : 
  T > 0 →
  (1/2 : ℝ) * T = T - (1/2 : ℝ) * T →
  (1/2 : ℝ) * T * (1 - (1/2 : ℝ) * (1 - f)) = (45/100 : ℝ) * T →
  f = 4/5 := by sorry

end NUMINAMATH_CALUDE_undeclared_major_fraction_l3884_388484


namespace NUMINAMATH_CALUDE_sandy_comic_books_l3884_388461

theorem sandy_comic_books (x : ℕ) : 
  (x / 2 : ℚ) - 3 + 6 = 13 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l3884_388461


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l3884_388402

/-- Conversion from polar to rectangular coordinates -/
theorem polar_to_rectangular (r : ℝ) (θ : ℝ) :
  let (x, y) := (r * Real.cos θ, r * Real.sin θ)
  (x, y) = (5 / 2, -5 * Real.sqrt 3 / 2) ↔ r = 5 ∧ θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l3884_388402


namespace NUMINAMATH_CALUDE_f_minimum_at_neg_two_l3884_388430

/-- The function f(x) = |x+1| + |x+2| + |x+3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem f_minimum_at_neg_two :
  f (-2) = 2 ∧ ∀ x : ℝ, f x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_at_neg_two_l3884_388430


namespace NUMINAMATH_CALUDE_peach_boxes_theorem_l3884_388437

/-- Given the initial number of peaches per basket, the number of baskets,
    the number of peaches eaten, and the number of peaches per smaller box,
    calculate the number of smaller boxes of peaches. -/
def number_of_boxes (peaches_per_basket : ℕ) (num_baskets : ℕ) (peaches_eaten : ℕ) (peaches_per_small_box : ℕ) : ℕ :=
  ((peaches_per_basket * num_baskets) - peaches_eaten) / peaches_per_small_box

/-- Prove that under the given conditions, the number of smaller boxes of peaches is 8. -/
theorem peach_boxes_theorem :
  number_of_boxes 25 5 5 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_boxes_theorem_l3884_388437


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3884_388496

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 300 ∧ 
  9 * a = b - 11 ∧ 
  9 * a = c + 15 ∧ 
  a ≤ b ∧ 
  a ≤ c ∧ 
  c ≤ b → 
  a * b * c = 319760 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3884_388496


namespace NUMINAMATH_CALUDE_logarithm_equations_l3884_388400

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm with arbitrary base
noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_equations :
  (lg 4 + lg 500 - lg 2 = 3) ∧
  ((27 : ℝ)^(1/3) + (log 3 2) * (log 2 3) = 4) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equations_l3884_388400


namespace NUMINAMATH_CALUDE_f_properties_l3884_388431

-- Define the function f(x) = (x-2)(x+4)
def f (x : ℝ) : ℝ := (x - 2) * (x + 4)

-- Theorem statement
theorem f_properties :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x > f y) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y → f x < f y) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≥ -9) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = -9) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l3884_388431


namespace NUMINAMATH_CALUDE_function_inequality_l3884_388473

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f
variable (h : ∀ x, x > 0 → DifferentiableAt ℝ f x)

-- Define the condition f(x)/x > f'(x)
variable (cond : ∀ x, x > 0 → (f x) / x > deriv f x)

-- Theorem statement
theorem function_inequality : 2015 * (f 2016) > 2016 * (f 2015) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3884_388473


namespace NUMINAMATH_CALUDE_unfolded_holes_count_l3884_388447

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (original : Paper)
  (center_hole : Bool)
  (upper_right_hole : Bool)

/-- Counts the number of holes when the paper is unfolded -/
def count_holes (fp : FoldedPaper) : ℕ :=
  let center_holes := if fp.center_hole then 4 else 0
  let corner_holes := if fp.upper_right_hole then 4 else 0
  center_holes + corner_holes

/-- Theorem stating that the number of holes when unfolded is 8 -/
theorem unfolded_holes_count (p : Paper) : 
  ∀ (fp : FoldedPaper), 
    fp.original = p → 
    fp.center_hole = true → 
    fp.upper_right_hole = true → 
    count_holes fp = 8 :=
sorry

end NUMINAMATH_CALUDE_unfolded_holes_count_l3884_388447


namespace NUMINAMATH_CALUDE_subsets_containing_six_l3884_388413

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem subsets_containing_six (A : Finset ℕ) (h : A ⊆ S) (h6 : 6 ∈ A) :
  (Finset.filter (fun A => 6 ∈ A) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_six_l3884_388413


namespace NUMINAMATH_CALUDE_distance_to_x_axis_for_point_p_l3884_388419

/-- The distance from a point to the x-axis in a Cartesian coordinate system --/
def distanceToXAxis (x y : ℝ) : ℝ := |y|

/-- Theorem: The distance from point P(3, -2) to the x-axis is 2 --/
theorem distance_to_x_axis_for_point_p :
  distanceToXAxis 3 (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_for_point_p_l3884_388419


namespace NUMINAMATH_CALUDE_centroid_trace_area_l3884_388470

-- Define the circle
def Circle : Type := {p : ℝ × ℝ // (p.1^2 + p.2^2 = 225)}

-- Define points A, B, and C
def A : ℝ × ℝ := (-15, 0)
def B : ℝ × ℝ := (15, 0)

-- Define C as a point on the circle
def C : Circle := sorry

-- Define the centroid of triangle ABC
def centroid (c : Circle) : ℝ × ℝ := sorry

-- Statement to prove
theorem centroid_trace_area :
  ∃ (area : ℝ), area = 25 * Real.pi ∧
  (∀ (c : Circle), c.1 ≠ A ∧ c.1 ≠ B →
    (centroid c).1^2 + (centroid c).2^2 = 25) :=
sorry

end NUMINAMATH_CALUDE_centroid_trace_area_l3884_388470


namespace NUMINAMATH_CALUDE_greatest_common_length_l3884_388479

theorem greatest_common_length (a b c d : ℕ) 
  (ha : a = 48) (hb : b = 60) (hc : c = 72) (hd : d = 120) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l3884_388479


namespace NUMINAMATH_CALUDE_triangle_distance_l3884_388485

/-- Given a triangle ABC with the following properties:
  - AB = x meters
  - BC = 3 meters
  - Angle B = 150°
  - Area of triangle ABC = 3√3/4 m²
  Prove that the length of AC is √3 meters. -/
theorem triangle_distance (x : ℝ) : 
  let a := x
  let b := 3
  let c := (a^2 + b^2 - 2*a*b*Real.cos (150 * π / 180))^(1/2)
  let s := 3 * Real.sqrt 3 / 4
  s = 1/2 * a * b * Real.sin (150 * π / 180) →
  c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_distance_l3884_388485


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l3884_388459

theorem smaller_solution_quadratic_equation :
  ∃ x : ℝ, x^2 + 10*x - 40 = 0 ∧ 
  (∀ y : ℝ, y^2 + 10*y - 40 = 0 → x ≤ y) ∧
  x = -8 := by
sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l3884_388459


namespace NUMINAMATH_CALUDE_bexy_bicycle_speed_l3884_388481

/-- Bexy's round trip problem -/
theorem bexy_bicycle_speed :
  -- Bexy's walking distance and time
  let bexy_walk_distance : ℝ := 5
  let bexy_walk_time : ℝ := 1

  -- Ben's total round trip time in hours
  let ben_total_time : ℝ := 160 / 60

  -- Ben's speed relative to Bexy's average speed
  let ben_speed_ratio : ℝ := 1 / 2

  -- Bexy's bicycle speed
  ∃ bexy_bike_speed : ℝ,
    -- Ben's walking time is twice Bexy's
    let ben_walk_time : ℝ := 2 * bexy_walk_time

    -- Ben's biking time
    let ben_bike_time : ℝ := ben_total_time - ben_walk_time

    -- Ben's biking speed
    let ben_bike_speed : ℝ := bexy_bike_speed * ben_speed_ratio

    -- Distance traveled equals speed times time
    ben_bike_speed * ben_bike_time = bexy_walk_distance ∧
    bexy_bike_speed = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bexy_bicycle_speed_l3884_388481


namespace NUMINAMATH_CALUDE_expression_value_l3884_388418

theorem expression_value : 
  (1 - 2/7) / (0.25 + 3 * (1/4)) + (2 * 0.3) / (1.3 - 0.4) = 29/21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3884_388418


namespace NUMINAMATH_CALUDE_painted_fraction_of_specific_cone_l3884_388467

/-- Represents a cone with given dimensions -/
structure Cone where
  radius : ℝ
  slant_height : ℝ

/-- Calculates the fraction of a cone's surface area covered in paint -/
def painted_fraction (c : Cone) (paint_depth : ℝ) : ℚ :=
  sorry

/-- Theorem stating the correct fraction of painted surface area for the given cone -/
theorem painted_fraction_of_specific_cone :
  let c : Cone := { radius := 3, slant_height := 5 }
  painted_fraction c 2 = 27 / 32 := by
  sorry

end NUMINAMATH_CALUDE_painted_fraction_of_specific_cone_l3884_388467


namespace NUMINAMATH_CALUDE_inequality_properties_l3884_388488

theorem inequality_properties (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ (a^2 > b^2) ∧ (a * b > b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3884_388488


namespace NUMINAMATH_CALUDE_angle_relation_l3884_388424

theorem angle_relation (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) (h4 : A + B + C = π) :
  (Real.cos (A/2))^2 = (Real.cos (B/2))^2 + (Real.cos (C/2))^2 - 2 * Real.cos (B/2) * Real.cos (C/2) * Real.sin (A/2) := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l3884_388424


namespace NUMINAMATH_CALUDE_half_power_inequality_l3884_388412

theorem half_power_inequality (m n : ℝ) (h : m > n) : (1/2 : ℝ)^m < (1/2 : ℝ)^n := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l3884_388412


namespace NUMINAMATH_CALUDE_problem_solution_l3884_388497

theorem problem_solution (x y : ℝ) (h1 : x + y = 4) (h2 : x - y = 6) :
  2 * x^2 - 2 * y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3884_388497


namespace NUMINAMATH_CALUDE_line_segment_length_l3884_388408

/-- Given points A, B, C, and D on a line in that order, prove that CD = 3 cm -/
theorem line_segment_length (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (B - A = 2) →                  -- AB = 2 cm
  (C - A = 5) →                  -- AC = 5 cm
  (D - B = 6) →                  -- BD = 6 cm
  (D - C = 3) :=                 -- CD = 3 cm (to be proved)
by sorry

end NUMINAMATH_CALUDE_line_segment_length_l3884_388408


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_nine_l3884_388494

theorem gcd_factorial_eight_nine : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_nine_l3884_388494


namespace NUMINAMATH_CALUDE_hyperbola_proof_l3884_388410

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- Hyperbola to prove -/
def target_hyperbola (x y : ℝ) : Prop := x^2/3 - y^2/12 = 1

/-- Point that the target hyperbola passes through -/
def point : ℝ × ℝ := (2, 2)

theorem hyperbola_proof :
  (∀ x y : ℝ, given_hyperbola x y ↔ ∃ k : ℝ, x^2 - y^2/4 = k) ∧
  target_hyperbola point.1 point.2 ∧
  (∀ x y : ℝ, given_hyperbola x y ↔ target_hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_proof_l3884_388410


namespace NUMINAMATH_CALUDE_sticker_cost_l3884_388436

theorem sticker_cost (num_packs : ℕ) (stickers_per_pack : ℕ) (james_payment : ℚ) :
  num_packs = 4 →
  stickers_per_pack = 30 →
  james_payment = 6 →
  (2 * james_payment) / (num_packs * stickers_per_pack : ℚ) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_sticker_cost_l3884_388436


namespace NUMINAMATH_CALUDE_coloring_books_per_shelf_l3884_388404

theorem coloring_books_per_shelf 
  (initial_stock : ℕ) 
  (sold : ℕ) 
  (shelves : ℕ) 
  (h1 : initial_stock = 87) 
  (h2 : sold = 33) 
  (h3 : shelves = 9) 
  (h4 : shelves > 0) : 
  (initial_stock - sold) / shelves = 6 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_per_shelf_l3884_388404


namespace NUMINAMATH_CALUDE_dividend_calculation_l3884_388427

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 18 → quotient = 9 → remainder = 3 → 
  dividend = divisor * quotient + remainder →
  dividend = 165 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3884_388427


namespace NUMINAMATH_CALUDE_parity_of_linear_system_solution_l3884_388478

theorem parity_of_linear_system_solution (n m : ℤ) 
  (h_n_odd : Odd n) (h_m_odd : Odd m) :
  ∃ (x y : ℤ), x + 2*y = n ∧ 3*x - y = m → Odd x ∧ Even y := by
  sorry

end NUMINAMATH_CALUDE_parity_of_linear_system_solution_l3884_388478


namespace NUMINAMATH_CALUDE_first_day_exceeding_target_day_exceeding_target_is_tuesday_l3884_388469

/-- Geometric sequence sum function -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- First day of deposit (Sunday) -/
def initialDay : ℕ := 0

/-- Initial deposit amount in cents -/
def initialDeposit : ℚ := 3

/-- Daily deposit multiplier -/
def dailyMultiplier : ℚ := 2

/-- Target amount in cents -/
def targetAmount : ℚ := 2000

/-- Function to calculate the day of the week -/
def dayOfWeek (n : ℕ) : ℕ :=
  (initialDay + n) % 7

/-- Theorem: The 10th deposit day is the first to exceed the target amount -/
theorem first_day_exceeding_target :
  (∀ k < 10, geometricSum initialDeposit dailyMultiplier k ≤ targetAmount) ∧
  geometricSum initialDeposit dailyMultiplier 10 > targetAmount :=
sorry

/-- Corollary: The day when the total first exceeds the target is Tuesday -/
theorem day_exceeding_target_is_tuesday :
  dayOfWeek 10 = 2 :=
sorry

end NUMINAMATH_CALUDE_first_day_exceeding_target_day_exceeding_target_is_tuesday_l3884_388469


namespace NUMINAMATH_CALUDE_compound_interest_time_calculation_l3884_388420

/-- Proves that the time t satisfies the compound interest equation for the given problem --/
theorem compound_interest_time_calculation 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (compounding_frequency : ℝ) 
  (final_amount : ℝ) 
  (h1 : initial_investment = 600)
  (h2 : annual_rate = 0.10)
  (h3 : compounding_frequency = 2)
  (h4 : final_amount = 661.5) :
  ∃ t : ℝ, final_amount = initial_investment * (1 + annual_rate / compounding_frequency) ^ (compounding_frequency * t) :=
sorry

end NUMINAMATH_CALUDE_compound_interest_time_calculation_l3884_388420


namespace NUMINAMATH_CALUDE_rowing_current_velocity_l3884_388417

/-- Proves that the velocity of the current is 1 kmph given the conditions of the rowing problem. -/
theorem rowing_current_velocity 
  (still_water_speed : ℝ) 
  (distance : ℝ) 
  (total_time : ℝ) 
  (h1 : still_water_speed = 5)
  (h2 : distance = 2.4)
  (h3 : total_time = 1) :
  ∃ v : ℝ, v = 1 ∧ total_time = distance / (still_water_speed + v) + distance / (still_water_speed - v) :=
by sorry

end NUMINAMATH_CALUDE_rowing_current_velocity_l3884_388417


namespace NUMINAMATH_CALUDE_system_solution_l3884_388455

-- Define the system of equations
def system (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Define the solution
def solution (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  (y = 2 ∧ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) ∨
  (y ≠ 2 ∧
    ((y^2 + y - 1 ≠ 0 ∧ x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨
     (y^2 + y - 1 = 0 ∧ (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
      x₃ = y * x₂ - x₁ ∧
      x₄ = -y * x₂ - y * x₁ ∧
      x₅ = y * x₁ - x₂)))

-- Theorem statement
theorem system_solution (x₁ x₂ x₃ x₄ x₅ y : ℝ) :
  system x₁ x₂ x₃ x₄ x₅ y ↔ solution x₁ x₂ x₃ x₄ x₅ y := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3884_388455


namespace NUMINAMATH_CALUDE_apple_percentage_after_orange_removal_l3884_388476

/-- Calculates the percentage of apples in a bowl of fruit after removing oranges -/
theorem apple_percentage_after_orange_removal (initial_apples initial_oranges removed_oranges : ℕ) 
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 20)
  (h3 : removed_oranges = 14) :
  (initial_apples : ℚ) / (initial_apples + initial_oranges - removed_oranges) * 100 = 70 := by
  sorry


end NUMINAMATH_CALUDE_apple_percentage_after_orange_removal_l3884_388476


namespace NUMINAMATH_CALUDE_quadruple_equation_solution_l3884_388442

def is_valid_quadruple (a b c d : ℕ) : Prop :=
  a + b = c * d ∧ c + d = a * b

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2, 2, 2, 2), (1, 2, 3, 5), (2, 1, 3, 5), (1, 2, 5, 3), (2, 1, 5, 3),
   (3, 5, 1, 2), (5, 3, 1, 2), (3, 5, 2, 1), (5, 3, 2, 1)}

theorem quadruple_equation_solution :
  {q : ℕ × ℕ × ℕ × ℕ | is_valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solution_l3884_388442


namespace NUMINAMATH_CALUDE_fraction_equality_l3884_388452

theorem fraction_equality (a b c : ℝ) :
  (3 * a^2 + 3 * b^2 - 5 * c^2 + 6 * a * b) / (4 * a^2 + 4 * c^2 - 6 * b^2 + 8 * a * c) =
  ((a + b + Real.sqrt (5 * c^2)) * (a + b - Real.sqrt (5 * c^2))) /
  ((2 * (a + c) + Real.sqrt (6 * b^2)) * (2 * (a + c) - Real.sqrt (6 * b^2))) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3884_388452


namespace NUMINAMATH_CALUDE_final_ratio_is_16_to_9_l3884_388407

/-- Represents the contents of a bin with peanuts and raisins -/
structure BinContents where
  peanuts : ℚ
  raisins : ℚ

/-- Removes an amount from the bin proportionally -/
def removeProportionally (bin : BinContents) (amount : ℚ) : BinContents :=
  let total := bin.peanuts + bin.raisins
  let peanutsProportion := bin.peanuts / total
  let raisinsProportion := bin.raisins / total
  { peanuts := bin.peanuts - (peanutsProportion * amount)
  , raisins := bin.raisins - (raisinsProportion * amount) }

/-- Adds an amount of raisins to the bin -/
def addRaisins (bin : BinContents) (amount : ℚ) : BinContents :=
  { peanuts := bin.peanuts, raisins := bin.raisins + amount }

/-- Theorem stating the final ratio of peanuts to raisins -/
theorem final_ratio_is_16_to_9 :
  let initial_bin : BinContents := { peanuts := 10, raisins := 0 }
  let after_first_operation := addRaisins { peanuts := initial_bin.peanuts - 2, raisins := 0 } 2
  let after_second_operation := addRaisins (removeProportionally after_first_operation 2) 2
  (after_second_operation.peanuts * 9 = after_second_operation.raisins * 16) := by
  sorry

end NUMINAMATH_CALUDE_final_ratio_is_16_to_9_l3884_388407


namespace NUMINAMATH_CALUDE_G_4_g_5_equals_29_l3884_388443

-- Define the functions g and G
def g (x : ℝ) : ℝ := 2 * x - 3
def G (x y : ℝ) : ℝ := x * y + 2 * x - y

-- State the theorem
theorem G_4_g_5_equals_29 : G 4 (g 5) = 29 := by
  sorry

end NUMINAMATH_CALUDE_G_4_g_5_equals_29_l3884_388443


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l3884_388457

theorem abs_inequality_solution (x : ℝ) : 
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l3884_388457


namespace NUMINAMATH_CALUDE_percentage_of_girls_l3884_388460

theorem percentage_of_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 900 →
  boys = 90 →
  girls = total - boys →
  (girls : ℚ) / (total : ℚ) * 100 = 90 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l3884_388460


namespace NUMINAMATH_CALUDE_exam_students_count_l3884_388405

theorem exam_students_count :
  ∀ (n : ℕ) (T : ℝ),
    n > 0 →
    T = n * 90 →
    T - 120 = (n - 3) * 95 →
    n = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l3884_388405


namespace NUMINAMATH_CALUDE_petrol_expenses_l3884_388411

def monthly_salary : ℕ := 23000
def savings_percentage : ℚ := 1/10
def savings : ℕ := 2300
def known_expenses : ℕ := 18700

theorem petrol_expenses : 
  monthly_salary * savings_percentage = savings →
  monthly_salary - savings - known_expenses = 2000 := by
sorry

end NUMINAMATH_CALUDE_petrol_expenses_l3884_388411


namespace NUMINAMATH_CALUDE_square_roots_problem_l3884_388432

theorem square_roots_problem (m a : ℝ) (hm : m > 0) 
  (h1 : (1 - 2*a)^2 = m) (h2 : (a - 5)^2 = m) (h3 : 1 - 2*a ≠ a - 5) : 
  m = 81 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3884_388432


namespace NUMINAMATH_CALUDE_open_box_volume_l3884_388472

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_length : ℝ)
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 3)
  : (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 3780 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l3884_388472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3884_388445

theorem arithmetic_sequence_sum : 
  ∀ (a d n : ℕ) (last : ℕ),
    a = 2 →
    d = 2 →
    last = 20 →
    last = a + (n - 1) * d →
    (n : ℕ) * (a + last) / 2 = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3884_388445


namespace NUMINAMATH_CALUDE_poly_factorable_iff_l3884_388489

/-- A polynomial in x and y with a parameter k -/
def poly (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + k*y - k

/-- A linear factor with integer coefficients -/
structure LinearFactor where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Check if a polynomial can be factored into two linear factors -/
def isFactorable (k : ℤ) : Prop :=
  ∃ (f g : LinearFactor), ∀ (x y : ℤ),
    poly k x y = (f.a * x + f.b * y + f.c) * (g.a * x + g.b * y + g.c)

/-- The main theorem: the polynomial is factorable iff k = 0 or k = 16 -/
theorem poly_factorable_iff (k : ℤ) : isFactorable k ↔ k = 0 ∨ k = 16 :=
sorry

end NUMINAMATH_CALUDE_poly_factorable_iff_l3884_388489


namespace NUMINAMATH_CALUDE_simplify_expression_l3884_388458

theorem simplify_expression (x y : ℝ) : (x - 3*y + 2) * (x + 3*y + 2) = x^2 + 4*x + 4 - 9*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3884_388458


namespace NUMINAMATH_CALUDE_project_hours_difference_l3884_388409

theorem project_hours_difference (total_pay : ℝ) (wage_p wage_q : ℝ) :
  total_pay = 420 ∧ 
  wage_p = wage_q * 1.5 ∧ 
  wage_p = wage_q + 7 →
  (total_pay / wage_q) - (total_pay / wage_p) = 10 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3884_388409


namespace NUMINAMATH_CALUDE_expand_expression_l3884_388463

theorem expand_expression (y : ℝ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3884_388463


namespace NUMINAMATH_CALUDE_total_pens_after_changes_l3884_388449

-- Define the initial number of pens
def initial_red : ℕ := 65
def initial_blue : ℕ := 45
def initial_black : ℕ := 58
def initial_green : ℕ := 36
def initial_purple : ℕ := 27

-- Define the changes in pen quantities
def red_decrease : ℕ := 15
def blue_decrease : ℕ := 20
def black_increase : ℕ := 12
def green_decrease : ℕ := 10
def purple_increase : ℕ := 5

-- Define the theorem
theorem total_pens_after_changes : 
  (initial_red - red_decrease) + 
  (initial_blue - blue_decrease) + 
  (initial_black + black_increase) + 
  (initial_green - green_decrease) + 
  (initial_purple + purple_increase) = 203 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_after_changes_l3884_388449


namespace NUMINAMATH_CALUDE_poem_line_growth_l3884_388451

/-- The number of months required to reach a target number of lines in a poem, 
    given the initial number of lines and the number of lines added per month. -/
def months_to_reach_target (initial_lines : ℕ) (lines_per_month : ℕ) (target_lines : ℕ) : ℕ :=
  (target_lines - initial_lines) / lines_per_month

theorem poem_line_growth : months_to_reach_target 24 3 90 = 22 := by
  sorry

end NUMINAMATH_CALUDE_poem_line_growth_l3884_388451


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_l3884_388492

theorem envelope_addressing_equation (x : ℝ) : x > 0 → (
  let rate1 := 800 / 12  -- rate of first machine
  let rate2 := 800 / x   -- rate of second machine
  let combined_rate := 800 / 3  -- combined rate of both machines
  rate1 + rate2 = combined_rate ↔ 1/12 + 1/x = 1/3
) := by sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_l3884_388492
