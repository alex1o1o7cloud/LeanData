import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l2904_290430

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 127 → 
  divisor = 25 → 
  remainder = 2 → 
  dividend = divisor * quotient + remainder → 
  quotient = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2904_290430


namespace NUMINAMATH_CALUDE_total_subscription_is_50000_l2904_290437

/-- Represents the subscription amounts and profit distribution for a business --/
structure BusinessSubscription where
  /-- C's subscription amount --/
  c : ℕ
  /-- Total profit --/
  total_profit : ℕ
  /-- A's profit share --/
  a_profit : ℕ

/-- Calculates the total subscription amount given the business subscription details --/
def total_subscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c + 14000

/-- Theorem stating that the total subscription amount is 50000 given the problem conditions --/
theorem total_subscription_is_50000 (bs : BusinessSubscription)
  (h1 : bs.total_profit = 36000)
  (h2 : bs.a_profit = 15120)
  (h3 : bs.a_profit * (3 * bs.c + 14000) = bs.total_profit * (bs.c + 9000)) :
  total_subscription bs = 50000 := by
  sorry

#check total_subscription_is_50000

end NUMINAMATH_CALUDE_total_subscription_is_50000_l2904_290437


namespace NUMINAMATH_CALUDE_proposition_contrapositive_equivalence_l2904_290465

theorem proposition_contrapositive_equivalence (P Q : Prop) :
  (P → Q) ↔ (¬Q → ¬P) := by sorry

end NUMINAMATH_CALUDE_proposition_contrapositive_equivalence_l2904_290465


namespace NUMINAMATH_CALUDE_expression_equals_one_tenth_l2904_290470

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the expression
def expression : ℚ :=
  (ceiling ((21 : ℚ) / 8 - ceiling ((35 : ℚ) / 21))) /
  (ceiling ((35 : ℚ) / 8 + ceiling ((8 * 21 : ℚ) / 35)))

-- Theorem statement
theorem expression_equals_one_tenth : expression = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_tenth_l2904_290470


namespace NUMINAMATH_CALUDE_negation_equivalence_l2904_290409

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2904_290409


namespace NUMINAMATH_CALUDE_rock_paper_scissors_tournament_l2904_290403

/-- Represents the number of players in the tournament -/
def n : ℕ := 4

/-- The number of players in the tournament -/
def num_players : ℕ := 2^n

/-- The number of possible moves (rock, paper, scissors) -/
def num_moves : ℕ := 3

/-- The number of matches in the tournament -/
def num_matches : ℕ := 2^n - 1

/-- The number of possible tournament outcomes -/
def tournament_outcomes : ℕ := 2^(2^n - 1)

/-- The total number of possible combinations of player choices
    that lead to a concluded tournament with a winner -/
def total_combinations : ℕ := num_moves * tournament_outcomes

theorem rock_paper_scissors_tournament :
  total_combinations = 3 * 2^15 := by
  sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_tournament_l2904_290403


namespace NUMINAMATH_CALUDE_no_intersection_l2904_290427

/-- The line equation y = 2x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 -/
def line_passes_through_vertex (b : ℝ) : Prop :=
  ∃ x : ℝ, 2 * x + b = x^2 + b^2 + 1

/-- There are no real values of b for which the line passes through the vertex of the parabola -/
theorem no_intersection :
  ¬∃ b : ℝ, line_passes_through_vertex b :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l2904_290427


namespace NUMINAMATH_CALUDE_graph_transformation_l2904_290420

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the reflection operation about x = 1
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

-- Define the left shift operation
def shift_left (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) :
  shift_left (reflect f) = λ x => f (1 - x) := by sorry

end NUMINAMATH_CALUDE_graph_transformation_l2904_290420


namespace NUMINAMATH_CALUDE_base5_to_decimal_conversion_l2904_290494

/-- Converts a base-5 digit to its decimal (base-10) value -/
def base5ToDecimal (digit : Nat) : Nat :=
  digit

/-- Converts a base-5 number to its decimal (base-10) equivalent -/
def convertBase5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + (base5ToDecimal d) * (5 ^ i)) 0

/-- The base-5 representation of the number -/
def base5Number : List Nat := [2, 1, 4, 3, 2]

theorem base5_to_decimal_conversion :
  convertBase5ToDecimal base5Number = 1732 := by
  sorry

end NUMINAMATH_CALUDE_base5_to_decimal_conversion_l2904_290494


namespace NUMINAMATH_CALUDE_homeless_donation_problem_l2904_290463

/-- The amount given to the last set of homeless families -/
def last_set_amount (total spent_on_first_four : ℝ) : ℝ :=
  total - spent_on_first_four

/-- The problem statement -/
theorem homeless_donation_problem (total first second third fourth : ℝ) 
  (h1 : total = 4500)
  (h2 : first = 725)
  (h3 : second = 1100)
  (h4 : third = 950)
  (h5 : fourth = 815) :
  last_set_amount total (first + second + third + fourth) = 910 := by
  sorry

end NUMINAMATH_CALUDE_homeless_donation_problem_l2904_290463


namespace NUMINAMATH_CALUDE_abs_le_2_set_equality_l2904_290453

def set_of_integers_abs_le_2 : Set ℤ := {x | |x| ≤ 2}

theorem abs_le_2_set_equality : set_of_integers_abs_le_2 = {-2, -1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_abs_le_2_set_equality_l2904_290453


namespace NUMINAMATH_CALUDE_basketball_distribution_l2904_290442

theorem basketball_distribution (total : ℕ) (left : ℕ) (classes : ℕ) : 
  total = 54 → left = 5 → classes * ((total - left) / classes) = total - left → classes = 7 := by
  sorry

end NUMINAMATH_CALUDE_basketball_distribution_l2904_290442


namespace NUMINAMATH_CALUDE_least_integer_with_digit_sum_property_l2904_290484

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: 2999999999999 is the least positive integer N such that
    the sum of its digits is 100 and the sum of the digits of 2N is 110 -/
theorem least_integer_with_digit_sum_property : 
  (∀ m : ℕ, m > 0 ∧ m < 2999999999999 → 
    (sum_of_digits m = 100 ∧ sum_of_digits (2 * m) = 110) → False) ∧ 
  sum_of_digits 2999999999999 = 100 ∧ 
  sum_of_digits (2 * 2999999999999) = 110 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_digit_sum_property_l2904_290484


namespace NUMINAMATH_CALUDE_advanced_tablet_price_relationship_l2904_290488

/-- The price of a smartphone in dollars. -/
def smartphone_price : ℕ := 300

/-- The price difference between a personal computer and a smartphone in dollars. -/
def pc_price_difference : ℕ := 500

/-- The total cost of buying one of each product (smartphone, personal computer, and advanced tablet) in dollars. -/
def total_cost : ℕ := 2200

/-- The price of a personal computer in dollars. -/
def pc_price : ℕ := smartphone_price + pc_price_difference

/-- The price of an advanced tablet in dollars. -/
def advanced_tablet_price : ℕ := total_cost - (smartphone_price + pc_price)

theorem advanced_tablet_price_relationship :
  advanced_tablet_price = smartphone_price + pc_price - 400 := by
  sorry

end NUMINAMATH_CALUDE_advanced_tablet_price_relationship_l2904_290488


namespace NUMINAMATH_CALUDE_mistaken_division_l2904_290469

theorem mistaken_division (n : ℕ) (h : 2 * n = 622) : 
  (n / 12) + (n % 12) = 36 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_l2904_290469


namespace NUMINAMATH_CALUDE_ABC_collinear_l2904_290468

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points in the plane -/
def A : Point := ⟨-1, 4⟩
def B : Point := ⟨-3, 2⟩
def C : Point := ⟨0, 5⟩

/-- Definition of collinearity for three points -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Theorem: Points A, B, and C are collinear -/
theorem ABC_collinear : collinear A B C := by
  sorry

end NUMINAMATH_CALUDE_ABC_collinear_l2904_290468


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_increase_l2904_290407

theorem rectangular_prism_volume_increase (a b c : ℝ) : 
  (a * b * c = 8) → 
  ((a + 1) * (b + 1) * (c + 1) = 27) → 
  ((a + 2) * (b + 2) * (c + 2) = 64) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_increase_l2904_290407


namespace NUMINAMATH_CALUDE_value_of_a_l2904_290424

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2904_290424


namespace NUMINAMATH_CALUDE_twoDigitNumberRepresentation_l2904_290419

/-- Represents a two-digit number with x in the tens place and 5 in the ones place -/
def twoDigitNumber (x : ℕ) : ℕ := 10 * x + 5

/-- Proves that a two-digit number with x in the tens place and 5 in the ones place
    can be represented as 10x + 5 -/
theorem twoDigitNumberRepresentation (x : ℕ) (h : x < 10) :
  twoDigitNumber x = 10 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_twoDigitNumberRepresentation_l2904_290419


namespace NUMINAMATH_CALUDE_additional_money_needed_l2904_290435

def new_computer_cost : ℕ := 80
def initial_savings : ℕ := 50
def old_computer_sale : ℕ := 20

theorem additional_money_needed : 
  new_computer_cost - (initial_savings + old_computer_sale) = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2904_290435


namespace NUMINAMATH_CALUDE_right_triangle_max_area_l2904_290462

/-- Given a right triangle with perimeter 2, its maximum area is 3 - 2√2 -/
theorem right_triangle_max_area :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  a + b + Real.sqrt (a^2 + b^2) = 2 →
  (1/2) * a * b ≤ 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_area_l2904_290462


namespace NUMINAMATH_CALUDE_plains_routes_count_l2904_290443

/-- Represents the number of routes between two types of cities -/
structure RouteCount where
  total : ℕ
  mountain : ℕ
  plain : ℕ

/-- Calculates the number of routes between plains cities -/
def plainsRoutes (cities : ℕ × ℕ) (routes : RouteCount) : ℕ :=
  routes.total - routes.mountain - (cities.1 * 3 - 2 * routes.mountain) / 2

/-- Theorem stating the number of routes between plains cities -/
theorem plains_routes_count 
  (cities : ℕ × ℕ) 
  (routes : RouteCount) 
  (h1 : cities.1 + cities.2 = 100)
  (h2 : cities.1 = 30)
  (h3 : cities.2 = 70)
  (h4 : routes.total = 150)
  (h5 : routes.mountain = 21) :
  plainsRoutes cities routes = 81 := by
sorry

end NUMINAMATH_CALUDE_plains_routes_count_l2904_290443


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2904_290478

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  B = π / 4 →
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  A + B + C = π →
  A = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2904_290478


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l2904_290490

/-- Calculates the cost of tax-free items given total spend, tax percentage, and tax rate -/
def cost_of_tax_free_items (total_spend : ℚ) (tax_percentage : ℚ) (tax_rate : ℚ) : ℚ :=
  let taxable_cost := total_spend * (1 - tax_percentage / 100)
  let rounded_tax := (taxable_cost * tax_rate / 100).ceil
  total_spend - (taxable_cost + rounded_tax)

theorem tax_free_items_cost :
  cost_of_tax_free_items 40 30 6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l2904_290490


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_4_5_l2904_290408

theorem smallest_multiple_of_3_4_5 : 
  ∀ n : ℕ, (3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n) → n ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_4_5_l2904_290408


namespace NUMINAMATH_CALUDE_students_remaining_l2904_290405

theorem students_remaining (groups : ℕ) (students_per_group : ℕ) (left_early : ℕ) : 
  groups = 5 → students_per_group = 12 → left_early = 7 → 
  groups * students_per_group - left_early = 53 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_l2904_290405


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2904_290433

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 5 * d - 7 = 0) → 
  (3 * e^2 + 5 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2904_290433


namespace NUMINAMATH_CALUDE_factorial_divisor_sum_l2904_290483

theorem factorial_divisor_sum (n : ℕ) :
  ∀ k : ℕ, k ≤ n.factorial → ∃ (s : Finset ℕ),
    (∀ x ∈ s, x ∣ n.factorial) ∧
    s.card ≤ n ∧
    k = s.sum id :=
sorry

end NUMINAMATH_CALUDE_factorial_divisor_sum_l2904_290483


namespace NUMINAMATH_CALUDE_measure_of_inequality_is_zero_l2904_290474

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (μ : Measure Ω)
variable (ξ η : Ω → ℝ)

theorem measure_of_inequality_is_zero 
  (hξ_integrable : IntegrableOn (|ξ|) Set.univ μ)
  (hη_integrable : IntegrableOn (|η|) Set.univ μ)
  (h_inequality : ∀ (A : Set Ω), MeasurableSet A → ∫ x in A, ξ x ∂μ ≤ ∫ x in A, η x ∂μ) :
  μ {x | ξ x > η x} = 0 := by
  sorry

end NUMINAMATH_CALUDE_measure_of_inequality_is_zero_l2904_290474


namespace NUMINAMATH_CALUDE_problem_1_l2904_290404

theorem problem_1 : (-1)^3 + |2 - Real.sqrt 5| + (Real.pi / 2 - 1.57)^0 + Real.sqrt 20 = 3 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2904_290404


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2904_290452

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : Real.cos (2 * α) + Real.cos α ^ 2 = 0) : 
  Real.tan (α + Real.pi / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2904_290452


namespace NUMINAMATH_CALUDE_range_of_a_l2904_290456

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2904_290456


namespace NUMINAMATH_CALUDE_elizabeth_revenue_is_900_l2904_290498

/-- Represents the revenue and investment data for Mr. Banks and Ms. Elizabeth -/
structure InvestmentData where
  banks_investments : ℕ
  banks_revenue_per_investment : ℕ
  elizabeth_investments : ℕ
  elizabeth_total_revenue_difference : ℕ

/-- Calculates Ms. Elizabeth's revenue per investment given the investment data -/
def elizabeth_revenue_per_investment (data : InvestmentData) : ℕ :=
  (data.banks_investments * data.banks_revenue_per_investment + data.elizabeth_total_revenue_difference) / data.elizabeth_investments

/-- Theorem stating that Ms. Elizabeth's revenue per investment is $900 given the problem conditions -/
theorem elizabeth_revenue_is_900 (data : InvestmentData)
  (h1 : data.banks_investments = 8)
  (h2 : data.banks_revenue_per_investment = 500)
  (h3 : data.elizabeth_investments = 5)
  (h4 : data.elizabeth_total_revenue_difference = 500) :
  elizabeth_revenue_per_investment data = 900 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_revenue_is_900_l2904_290498


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2904_290429

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2904_290429


namespace NUMINAMATH_CALUDE_puzzle_palace_spending_l2904_290492

theorem puzzle_palace_spending (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 90)
  (h2 : remaining_amount = 12) :
  initial_amount - remaining_amount = 78 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_palace_spending_l2904_290492


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l2904_290432

/-- Proves that the initial volume of a 20% salt solution is 90 liters,
    given that adding 30 liters of water dilutes it to a 15% salt solution. -/
theorem salt_solution_dilution (initial_volume : ℝ) : 
  (0.20 * initial_volume = 0.15 * (initial_volume + 30)) → 
  initial_volume = 90 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_dilution_l2904_290432


namespace NUMINAMATH_CALUDE_intersection_M_N_l2904_290457

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2904_290457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2904_290422

/-- An arithmetic sequence with first term 1 and sum of first three terms 9 has general term 2n - 1 -/
theorem arithmetic_sequence_general_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) → -- arithmetic sequence condition
    a 1 = 1 → -- first term condition
    a 1 + a 2 + a 3 = 9 → -- sum of first three terms condition
    ∀ n, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2904_290422


namespace NUMINAMATH_CALUDE_negation_of_existence_square_greater_than_self_negation_l2904_290460

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬ ∃ x : ℕ, P x) ↔ (∀ x : ℕ, ¬ P x) := by sorry

theorem square_greater_than_self_negation :
  (¬ ∃ x : ℕ, x^2 ≤ x) ↔ (∀ x : ℕ, x^2 > x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_greater_than_self_negation_l2904_290460


namespace NUMINAMATH_CALUDE_cone_volume_l2904_290431

theorem cone_volume (central_angle : Real) (sector_area : Real) :
  central_angle = 120 * Real.pi / 180 →
  sector_area = 3 * Real.pi →
  ∃ (volume : Real), volume = (2 * Real.sqrt 2 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2904_290431


namespace NUMINAMATH_CALUDE_square_area_increase_l2904_290496

theorem square_area_increase (s : ℝ) (k : ℝ) (h1 : s > 0) (h2 : k > 0) :
  (k * s)^2 = 25 * s^2 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l2904_290496


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2904_290472

theorem trig_identity_proof : 
  2 * Real.cos (π / 6) - Real.tan (π / 3) + Real.sin (π / 4) * Real.cos (π / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2904_290472


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2904_290459

theorem sum_of_cubes : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2904_290459


namespace NUMINAMATH_CALUDE_exam_total_marks_l2904_290425

theorem exam_total_marks (student_marks : ℝ) (percentage : ℝ) 
  (h1 : student_marks = 450)
  (h2 : percentage = 90)
  (h3 : student_marks = (percentage / 100) * total_marks) :
  total_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_exam_total_marks_l2904_290425


namespace NUMINAMATH_CALUDE_xy_geq_ac_plus_bd_l2904_290406

theorem xy_geq_ac_plus_bd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hx : x = Real.sqrt (a^2 + b^2)) (hy : y = Real.sqrt (c^2 + d^2)) : x * y ≥ a * c + b * d :=
by sorry

end NUMINAMATH_CALUDE_xy_geq_ac_plus_bd_l2904_290406


namespace NUMINAMATH_CALUDE_rachel_picture_book_shelves_l2904_290480

/-- Calculates the number of picture book shelves given the total number of books,
    number of mystery book shelves, and books per shelf. -/
def picture_book_shelves (total_books : ℕ) (mystery_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf

/-- Proves that Rachel has 2 shelves of picture books given the problem conditions. -/
theorem rachel_picture_book_shelves :
  picture_book_shelves 72 6 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rachel_picture_book_shelves_l2904_290480


namespace NUMINAMATH_CALUDE_girls_pass_percentage_l2904_290458

theorem girls_pass_percentage 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (boys_pass_rate : ℚ) 
  (total_fail_rate : ℚ) :
  total_boys = 50 →
  total_girls = 100 →
  boys_pass_rate = 1/2 →
  total_fail_rate = 5667/10000 →
  (total_girls - (((total_boys + total_girls) * total_fail_rate).floor - (total_boys * (1 - boys_pass_rate)).floor)) / total_girls = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_girls_pass_percentage_l2904_290458


namespace NUMINAMATH_CALUDE_exponent_equality_l2904_290415

theorem exponent_equality (n : ℕ) : 4^8 = 4^n → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2904_290415


namespace NUMINAMATH_CALUDE_pizza_party_children_count_l2904_290464

theorem pizza_party_children_count (total : ℕ) (children : ℕ) (adults : ℕ) : 
  total = 120 →
  children = 2 * adults →
  total = children + adults →
  children = 80 := by
sorry

end NUMINAMATH_CALUDE_pizza_party_children_count_l2904_290464


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2904_290487

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle structure -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem statement -/
theorem parabola_line_intersection (C : Parabola) (l : Line) (M N : Point) (directrix : Line) :
  (l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3) →  -- Line equation: y = -√3(x-1)
  (Point.mk (C.p / 2) 0).y = l.m * (Point.mk (C.p / 2) 0).x + l.b →  -- Line passes through focus
  (M.y^2 = 2 * C.p * M.x ∧ N.y^2 = 2 * C.p * N.x) →  -- M and N are on the parabola
  (directrix.m = 0 ∧ directrix.b = -C.p / 2) →  -- Directrix equation: x = -p/2
  (C.p = 2 ∧  -- First conclusion: p = 2
   ∃ (circ : Circle), circ.center = Point.mk ((M.x + N.x) / 2) ((M.y + N.y) / 2) ∧
                      circ.radius = abs ((M.x - N.x) / 2) ∧
                      abs (circ.center.x - (-C.p / 2)) = circ.radius)  -- Second conclusion: Circle tangent to directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2904_290487


namespace NUMINAMATH_CALUDE_joan_has_five_apples_l2904_290441

/-- The number of apples Joan has after giving some away -/
def apples_remaining (initial : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : ℕ :=
  initial - given_to_melanie - given_to_sarah

/-- Proof that Joan has 5 apples remaining -/
theorem joan_has_five_apples :
  apples_remaining 43 27 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_five_apples_l2904_290441


namespace NUMINAMATH_CALUDE_quadratic_solution_l2904_290421

theorem quadratic_solution : ∃ x : ℝ, x^2 - x - 1 = 0 ∧ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2904_290421


namespace NUMINAMATH_CALUDE_candidates_count_l2904_290449

theorem candidates_count (x : ℝ) : 
  (x > 0) →  -- number of candidates is positive
  (0.07 * x = 0.06 * x + 80) →  -- State B had 80 more selected candidates
  (x = 8000) := by
sorry

end NUMINAMATH_CALUDE_candidates_count_l2904_290449


namespace NUMINAMATH_CALUDE_problem_statement_l2904_290423

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/(x - 3)^2 = 23 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2904_290423


namespace NUMINAMATH_CALUDE_inequality_proof_l2904_290461

theorem inequality_proof (a b c d : ℝ) :
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2904_290461


namespace NUMINAMATH_CALUDE_problem_solution_l2904_290481

theorem problem_solution (x : ℝ) (h : Real.sqrt x = 6000 * (1/1000)) :
  (600 - Real.sqrt x)^2 + x = 352872 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2904_290481


namespace NUMINAMATH_CALUDE_wire_cutting_l2904_290466

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 90 →
  ratio = 2 / 7 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 20 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_l2904_290466


namespace NUMINAMATH_CALUDE_min_distance_A_to_E_l2904_290417

/-- Given five points A, B, C, D, and E with specified distances between them,
    prove that the minimum possible distance between A and E is 2 units. -/
theorem min_distance_A_to_E (A B C D E : ℝ) : 
  (∃ (AB BC CD DE : ℝ), 
    AB = 12 ∧ 
    BC = 5 ∧ 
    CD = 3 ∧ 
    DE = 2 ∧ 
    (∀ (AE : ℝ), AE ≥ 2)) → 
  (∃ (AE : ℝ), AE = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_A_to_E_l2904_290417


namespace NUMINAMATH_CALUDE_right_triangle_iff_sum_squares_eq_eight_R_squared_l2904_290473

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  circumradius_positive : 0 < R

/-- Definition of a right triangle -/
def IsRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- Theorem: A triangle satisfies a² + b² + c² = 8R² if and only if it is a right triangle -/
theorem right_triangle_iff_sum_squares_eq_eight_R_squared (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 8 * t.R^2 ↔ IsRightTriangle t := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_iff_sum_squares_eq_eight_R_squared_l2904_290473


namespace NUMINAMATH_CALUDE_nancy_hourly_wage_l2904_290477

/-- Calculates the hourly wage needed to cover remaining expenses for one semester --/
def hourly_wage_needed (tuition housing meal_plan textbooks merit_scholarship need_scholarship work_hours : ℕ) : ℚ :=
  let total_cost := tuition + housing + meal_plan + textbooks
  let parents_contribution := tuition / 2
  let student_loan := 2 * merit_scholarship
  let total_support := parents_contribution + merit_scholarship + need_scholarship + student_loan
  let remaining_expenses := total_cost - total_support
  (remaining_expenses : ℚ) / work_hours

/-- Theorem stating that Nancy needs to earn $49 per hour --/
theorem nancy_hourly_wage :
  hourly_wage_needed 22000 6000 2500 800 3000 1500 200 = 49 := by
  sorry

end NUMINAMATH_CALUDE_nancy_hourly_wage_l2904_290477


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l2904_290467

theorem no_solution_fractional_equation :
  ∀ x : ℝ, (((1 - x) / (x - 2)) + 2 ≠ 1 / (2 - x)) ∨ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l2904_290467


namespace NUMINAMATH_CALUDE_fish_ratio_l2904_290482

theorem fish_ratio (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) (total : ℕ) :
  bass = 32 →
  trout = bass / 4 →
  total = 104 →
  total = bass + trout + blue_gill →
  blue_gill / bass = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_ratio_l2904_290482


namespace NUMINAMATH_CALUDE_bus_trip_speed_l2904_290439

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 360 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (v : ℝ), v > 0 ∧ distance / v - time_decrease = distance / (v + speed_increase) ∧ v = 40 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l2904_290439


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2904_290485

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ 1 →
  (((1 / x - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = 1 / (x * (x - 1))) ∧
  (((1 / 2 - 1 / 3) / ((2^2 - 1) / (2^2 + 2*2 + 1))) = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2904_290485


namespace NUMINAMATH_CALUDE_phone_number_probability_l2904_290414

def first_three_digits : ℕ := 3
def last_five_digits_arrangements : ℕ := 300

theorem phone_number_probability :
  let total_possibilities := first_three_digits * last_five_digits_arrangements
  (1 : ℚ) / total_possibilities = (1 : ℚ) / 900 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l2904_290414


namespace NUMINAMATH_CALUDE_max_three_cards_l2904_290479

theorem max_three_cards (total_cards : ℕ) (sum : ℕ) (cards_chosen : ℕ) : 
  total_cards = 10 →
  sum = 31 →
  cards_chosen = 8 →
  ∃ (threes fours fives : ℕ),
    threes + fours + fives = cards_chosen ∧
    3 * threes + 4 * fours + 5 * fives = sum ∧
    threes ≤ 4 ∧
    ∀ (t f v : ℕ), 
      t + f + v = cards_chosen →
      3 * t + 4 * f + 5 * v = sum →
      t ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_three_cards_l2904_290479


namespace NUMINAMATH_CALUDE_binomial_60_3_l2904_290401

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l2904_290401


namespace NUMINAMATH_CALUDE_inequality_for_positive_reals_l2904_290434

theorem inequality_for_positive_reals : ∀ x : ℝ, x > 0 → x + 4/x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_reals_l2904_290434


namespace NUMINAMATH_CALUDE_definite_integral_sqrt_4_minus_x_squared_minus_2x_l2904_290411

theorem definite_integral_sqrt_4_minus_x_squared_minus_2x : 
  ∫ (x : ℝ) in (0)..(2), (Real.sqrt (4 - x^2) - 2*x) = π - 4 := by sorry

end NUMINAMATH_CALUDE_definite_integral_sqrt_4_minus_x_squared_minus_2x_l2904_290411


namespace NUMINAMATH_CALUDE_continuity_at_one_l2904_290455

def f (x : ℝ) := -5 * x^2 - 7

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_one_l2904_290455


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2904_290402

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 5 + 5 * Complex.I)) = 9 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2904_290402


namespace NUMINAMATH_CALUDE_mod_equivalence_2021_l2904_290413

theorem mod_equivalence_2021 :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2021 [ZMOD 13] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2021_l2904_290413


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_five_l2904_290445

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_five_l2904_290445


namespace NUMINAMATH_CALUDE_parallel_EX_AP_l2904_290444

noncomputable section

-- Define the points on a complex plane
variable (a b c p h e q r x : ℂ)

-- Define the triangle ABC on the unit circle
def on_unit_circle (z : ℂ) : Prop := Complex.abs z = 1

-- Define the orthocenter condition
def is_orthocenter (a b c h : ℂ) : Prop := a + b + c = h

-- Define the circumcircle condition
def on_circumcircle (a b c p : ℂ) : Prop := on_unit_circle p

-- Define the foot of altitude condition
def is_foot_of_altitude (a b c e : ℂ) : Prop :=
  e = (1 / 2) * (a + b + c - (a * c) / b)

-- Define parallelogram conditions
def is_parallelogram_PAQB (a b p q : ℂ) : Prop := q = a + b - p
def is_parallelogram_PARC (a c p r : ℂ) : Prop := r = a + c - p

-- Define the intersection point condition
def is_intersection (a q h r x : ℂ) : Prop :=
  ∃ t₁ t₂ : ℝ, x = a + t₁ * (q - a) ∧ x = h + t₂ * (r - h)

-- Main theorem
theorem parallel_EX_AP (a b c p h e q r x : ℂ) 
  (h_circle : on_unit_circle a ∧ on_unit_circle b ∧ on_unit_circle c)
  (h_orthocenter : is_orthocenter a b c h)
  (h_circumcircle : on_circumcircle a b c p)
  (h_foot : is_foot_of_altitude a b c e)
  (h_para1 : is_parallelogram_PAQB a b p q)
  (h_para2 : is_parallelogram_PARC a c p r)
  (h_intersect : is_intersection a q h r x) :
  ∃ k : ℂ, e - x = k * (a - p) :=
sorry

end NUMINAMATH_CALUDE_parallel_EX_AP_l2904_290444


namespace NUMINAMATH_CALUDE_removed_to_total_ratio_is_one_to_two_l2904_290499

/-- Represents the number of bricks in a course -/
def bricks_per_course : ℕ := 400

/-- Represents the initial number of courses -/
def initial_courses : ℕ := 3

/-- Represents the number of courses added -/
def added_courses : ℕ := 2

/-- Represents the total number of bricks after removal -/
def total_bricks_after_removal : ℕ := 1800

/-- Theorem stating that the ratio of removed bricks to total bricks in the last course is 1:2 -/
theorem removed_to_total_ratio_is_one_to_two :
  let total_courses := initial_courses + added_courses
  let expected_total_bricks := total_courses * bricks_per_course
  let removed_bricks := expected_total_bricks - total_bricks_after_removal
  let last_course_bricks := bricks_per_course
  (removed_bricks : ℚ) / (last_course_bricks : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_removed_to_total_ratio_is_one_to_two_l2904_290499


namespace NUMINAMATH_CALUDE_resort_tips_fraction_l2904_290471

theorem resort_tips_fraction (avg_tips : ℝ) (h : avg_tips > 0) :
  let other_months_tips := 6 * avg_tips
  let august_tips := 6 * avg_tips
  let total_tips := other_months_tips + august_tips
  august_tips / total_tips = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_resort_tips_fraction_l2904_290471


namespace NUMINAMATH_CALUDE_campaign_donation_percentage_l2904_290400

theorem campaign_donation_percentage :
  let max_donation : ℕ := 1200
  let max_donors : ℕ := 500
  let half_donation : ℕ := max_donation / 2
  let half_donors : ℕ := 3 * max_donors
  let total_raised : ℕ := 3750000
  let donation_sum : ℕ := max_donation * max_donors + half_donation * half_donors
  (donation_sum : ℚ) / total_raised * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_campaign_donation_percentage_l2904_290400


namespace NUMINAMATH_CALUDE_committee_probability_grammar_club_committee_probability_l2904_290447

/-- The probability of selecting a committee with at least one boy and one girl -/
theorem committee_probability (total : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) :
  total = boys + girls →
  total = 25 →
  boys = 15 →
  girls = 10 →
  committee_size = 5 →
  (Nat.choose total committee_size - (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
  Nat.choose total committee_size = 195 / 208 := by
  sorry

/-- Main theorem stating the probability for the specific case -/
theorem grammar_club_committee_probability :
  (Nat.choose 25 5 - (Nat.choose 15 5 + Nat.choose 10 5)) / Nat.choose 25 5 = 195 / 208 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_grammar_club_committee_probability_l2904_290447


namespace NUMINAMATH_CALUDE_integral_x_minus_reciprocal_x_l2904_290416

theorem integral_x_minus_reciprocal_x (f : ℝ → ℝ) (hf : ∀ x ∈ Set.Icc 1 2, HasDerivAt f (x - 1/x) x) :
  ∫ x in Set.Icc 1 2, (x - 1/x) = 1 - Real.log 2 := by
sorry

end NUMINAMATH_CALUDE_integral_x_minus_reciprocal_x_l2904_290416


namespace NUMINAMATH_CALUDE_probability_sum_not_less_than_6_l2904_290410

/-- Represents a tetrahedral die with faces numbered 1, 2, 3, 5 -/
def TetrahedralDie : Type := Fin 4

/-- The possible face values of the tetrahedral die -/
def face_values : List ℕ := [1, 2, 3, 5]

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 16

/-- Predicate to check if the sum of two face values is not less than 6 -/
def sum_not_less_than_6 (a b : ℕ) : Prop := a + b ≥ 6

/-- The number of favorable outcomes (sum not less than 6) -/
def favorable_outcomes : ℕ := 8

/-- Theorem stating that the probability of the sum being not less than 6 is 1/2 -/
theorem probability_sum_not_less_than_6 :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_sum_not_less_than_6_l2904_290410


namespace NUMINAMATH_CALUDE_diagonal_angle_tangent_l2904_290412

/-- A convex quadrilateral with given properties -/
structure ConvexQuadrilateral where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  convex : Bool

/-- The measure of the acute angle formed by the diagonals -/
def diagonalAngle (q : ConvexQuadrilateral) : ℝ := sorry

/-- Theorem stating the tangent of the diagonal angle -/
theorem diagonal_angle_tangent (q : ConvexQuadrilateral) 
  (h1 : q.area = 30)
  (h2 : q.side1 = 5)
  (h3 : q.side2 = 6)
  (h4 : q.side3 = 9)
  (h5 : q.side4 = 7)
  (h6 : q.convex = true) :
  Real.tan (diagonalAngle q) = 40 / 7 := by sorry

end NUMINAMATH_CALUDE_diagonal_angle_tangent_l2904_290412


namespace NUMINAMATH_CALUDE_playground_children_count_l2904_290454

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 27) 
  (h2 : girls = 35) : 
  boys + girls = 62 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l2904_290454


namespace NUMINAMATH_CALUDE_trackball_mice_count_l2904_290489

theorem trackball_mice_count (total : ℕ) (wireless : ℕ) (optical : ℕ) (trackball : ℕ) : 
  total = 80 →
  wireless = total / 2 →
  optical = total / 4 →
  trackball = total - (wireless + optical) →
  trackball = 20 := by
sorry

end NUMINAMATH_CALUDE_trackball_mice_count_l2904_290489


namespace NUMINAMATH_CALUDE_existence_of_strictly_decreasing_function_with_inequality_l2904_290497

/-- A strictly decreasing function from (0, +∞) to (0, +∞) -/
def StrictlyDecreasingPositiveFunction :=
  {g : ℝ → ℝ | ∀ x y, 0 < x → 0 < y → x < y → g y < g x}

theorem existence_of_strictly_decreasing_function_with_inequality
  (k : ℝ) (h_k : 0 < k) :
  (∃ g : ℝ → ℝ, g ∈ StrictlyDecreasingPositiveFunction ∧
    ∀ x, 0 < x → 0 < g x ∧ g x ≥ k * g (x + g x)) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_strictly_decreasing_function_with_inequality_l2904_290497


namespace NUMINAMATH_CALUDE_shop_e_tv_sets_l2904_290436

theorem shop_e_tv_sets (shops : Fin 5 → ℕ)
  (ha : shops 0 = 20)
  (hb : shops 1 = 30)
  (hc : shops 2 = 60)
  (hd : shops 3 = 80)
  (havg : (shops 0 + shops 1 + shops 2 + shops 3 + shops 4) / 5 = 48) :
  shops 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_shop_e_tv_sets_l2904_290436


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2904_290475

/-- Given a population size and sample size, prove that the probability of selection
    is equal for simple random sampling, systematic sampling, and stratified sampling. -/
theorem equal_selection_probability
  (N n : ℕ) -- Population size and sample size
  (h_N_pos : N > 0) -- Assumption: Population size is positive
  (h_n_le_N : n ≤ N) -- Assumption: Sample size is not greater than population size
  (P₁ P₂ P₃ : ℚ) -- Probabilities for each sampling method
  (h_P₁ : P₁ = n / N) -- Definition of P₁ for simple random sampling
  (h_P₂ : P₂ = n / N) -- Definition of P₂ for systematic sampling
  (h_P₃ : P₃ = n / N) -- Definition of P₃ for stratified sampling
  : P₁ = P₂ ∧ P₂ = P₃ := by
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2904_290475


namespace NUMINAMATH_CALUDE_computer_price_increase_l2904_290440

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 540) : 
  d * (1 + 0.3) = 351 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2904_290440


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l2904_290486

theorem quadratic_completion_square (a : ℝ) (n : ℝ) : 
  (∀ x, x^2 + a*x + (1/4 : ℝ) = (x + n)^2 + (1/16 : ℝ)) → 
  a < 0 → 
  a = -((3 : ℝ).sqrt / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l2904_290486


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_1_l2904_290438

/-- The quadratic function f(x) = x² - 2mx + 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 5

/-- f(x) is decreasing for all x < 1 -/
def is_decreasing_before_1 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ < 1 → f m x₁ > f m x₂

theorem quadratic_decreasing_implies_m_geq_1 (m : ℝ) :
  is_decreasing_before_1 m → m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_1_l2904_290438


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2904_290495

theorem diophantine_equation_solutions :
  ∀ a b c : ℕ+,
  a * b + b * c + c * a = 2 * (a + b + c) ↔
  ((a = 2 ∧ b = 2 ∧ c = 2) ∨
   (a = 1 ∧ b = 2 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 2) ∨
   (a = 2 ∧ b = 1 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 1) ∨
   (a = 4 ∧ b = 1 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2904_290495


namespace NUMINAMATH_CALUDE_cos_420_plus_sin_330_equals_zero_l2904_290428

theorem cos_420_plus_sin_330_equals_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_420_plus_sin_330_equals_zero_l2904_290428


namespace NUMINAMATH_CALUDE_two_books_different_subjects_l2904_290451

theorem two_books_different_subjects (math_books : ℕ) (chinese_books : ℕ) (english_books : ℕ) :
  math_books = 10 → chinese_books = 9 → english_books = 8 →
  (math_books * chinese_books) + (math_books * english_books) + (chinese_books * english_books) = 242 :=
by sorry

end NUMINAMATH_CALUDE_two_books_different_subjects_l2904_290451


namespace NUMINAMATH_CALUDE_land_plot_area_land_plot_area_is_1267200_l2904_290476

/-- Calculates the total area of a land plot in acres given the dimensions in cm and conversion factors. -/
theorem land_plot_area 
  (triangle_base : ℝ) 
  (triangle_height : ℝ) 
  (rect_length : ℝ) 
  (rect_width : ℝ) 
  (scale_cm_to_miles : ℝ) 
  (acres_per_sq_mile : ℝ) : ℝ :=
  let triangle_area := (1/2) * triangle_base * triangle_height
  let rect_area := rect_length * rect_width
  let total_area_cm2 := triangle_area + rect_area
  let total_area_miles2 := total_area_cm2 * (scale_cm_to_miles^2)
  let total_area_acres := total_area_miles2 * acres_per_sq_mile
  total_area_acres

/-- Proves that the total area of the given land plot is 1267200 acres. -/
theorem land_plot_area_is_1267200 : 
  land_plot_area 20 12 20 5 3 640 = 1267200 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_area_land_plot_area_is_1267200_l2904_290476


namespace NUMINAMATH_CALUDE_function_value_problem_l2904_290450

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3)
  (h2 : f m = 6) : 
  m = -1/4 := by sorry

end NUMINAMATH_CALUDE_function_value_problem_l2904_290450


namespace NUMINAMATH_CALUDE_remaining_eggs_l2904_290426

def initial_eggs : ℕ := 50

def eggs_day1 : ℕ := 5 + 4
def eggs_day2 : ℕ := 8 + 3
def eggs_day3 : ℕ := 6 + 2

def total_eaten : ℕ := eggs_day1 + eggs_day2 + eggs_day3

theorem remaining_eggs : initial_eggs - total_eaten = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_eggs_l2904_290426


namespace NUMINAMATH_CALUDE_optimal_rectangle_area_l2904_290418

theorem optimal_rectangle_area 
  (perimeter : ℝ) 
  (min_length : ℝ) 
  (min_width : ℝ) 
  (h_perimeter : perimeter = 360) 
  (h_min_length : min_length = 90) 
  (h_min_width : min_width = 50) : 
  ∃ (length width : ℝ), 
    length ≥ min_length ∧ 
    width ≥ min_width ∧ 
    2 * (length + width) = perimeter ∧ 
    length * width = 8100 ∧ 
    ∀ (l w : ℝ), 
      l ≥ min_length → 
      w ≥ min_width → 
      2 * (l + w) = perimeter → 
      l * w ≤ 8100 :=
sorry

end NUMINAMATH_CALUDE_optimal_rectangle_area_l2904_290418


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2904_290446

/-- Calculates the total amount after compound interest -/
def totalAmountAfterCompoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem compound_interest_problem (compoundInterest : ℝ) (rate : ℝ) (time : ℕ) 
  (h1 : compoundInterest = 2828.80)
  (h2 : rate = 0.08)
  (h3 : time = 2) :
  ∃ (principal : ℝ), 
    totalAmountAfterCompoundInterest principal rate time - principal = compoundInterest ∧
    totalAmountAfterCompoundInterest principal rate time = 19828.80 := by
  sorry

#eval totalAmountAfterCompoundInterest 17000 0.08 2

end NUMINAMATH_CALUDE_compound_interest_problem_l2904_290446


namespace NUMINAMATH_CALUDE_sqrt2_minus1_power_representation_l2904_290448

theorem sqrt2_minus1_power_representation (n : ℤ) :
  ∃ (N : ℕ), (Real.sqrt 2 - 1) ^ n = Real.sqrt N - Real.sqrt (N - 1) :=
sorry

end NUMINAMATH_CALUDE_sqrt2_minus1_power_representation_l2904_290448


namespace NUMINAMATH_CALUDE_hexagon_area_is_six_l2904_290493

/-- A point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- A polygon defined by its vertices --/
structure Polygon where
  vertices : List GridPoint

/-- Calculate the area of a polygon given its vertices --/
def calculateArea (p : Polygon) : Int :=
  sorry

/-- The 4x4 square on the grid --/
def square : Polygon :=
  { vertices := [
      { x := 0, y := 0 },
      { x := 0, y := 4 },
      { x := 4, y := 4 },
      { x := 4, y := 0 }
    ] }

/-- The hexagon formed by adding two points to the square --/
def hexagon : Polygon :=
  { vertices := [
      { x := 0, y := 0 },
      { x := 0, y := 4 },
      { x := 2, y := 4 },
      { x := 4, y := 4 },
      { x := 4, y := 0 },
      { x := 2, y := 0 }
    ] }

theorem hexagon_area_is_six :
  calculateArea hexagon = 6 :=
sorry

end NUMINAMATH_CALUDE_hexagon_area_is_six_l2904_290493


namespace NUMINAMATH_CALUDE_complex_expression_equals_100_algebraic_expression_simplification_l2904_290491

-- Problem 1
theorem complex_expression_equals_100 :
  (2 * (7 / 9 : ℝ)) ^ (1 / 2 : ℝ) + (1 / 10 : ℝ) ^ (-2 : ℝ) + 
  (2 * (10 / 27 : ℝ)) ^ (-(2 / 3) : ℝ) - 3 * (Real.pi ^ (0 : ℝ)) + 
  (37 / 48 : ℝ) = 100 := by sorry

-- Problem 2
theorem algebraic_expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (a ^ (2 / 3)) * (b ^ (1 / 2))) * (-6 * (a ^ (1 / 2)) * (b ^ (1 / 3))) / 
  (-3 * (a ^ (1 / 6)) * (b ^ (5 / 6))) = 4 * a := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_100_algebraic_expression_simplification_l2904_290491
