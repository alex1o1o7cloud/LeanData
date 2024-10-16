import Mathlib

namespace NUMINAMATH_CALUDE_not_always_reducible_box_dimension_l2966_296695

structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  positive_dimensions : 0 < length ∧ 0 < width ∧ 0 < height

def fits_in (p q : RectangularParallelepiped) : Prop :=
  p.length ≤ q.length ∧ p.width ≤ q.width ∧ p.height ≤ q.height

def is_defective (original defective : RectangularParallelepiped) : Prop :=
  (defective.length < original.length ∧ defective.width = original.width ∧ defective.height = original.height) ∨
  (defective.length = original.length ∧ defective.width < original.width ∧ defective.height = original.height) ∨
  (defective.length = original.length ∧ defective.width = original.width ∧ defective.height < original.height)

theorem not_always_reducible_box_dimension 
  (box : RectangularParallelepiped) 
  (parallelepipeds : List RectangularParallelepiped) 
  (original_parallelepipeds : List RectangularParallelepiped) 
  (h1 : ∀ p ∈ parallelepipeds, fits_in p box)
  (h2 : parallelepipeds.length = original_parallelepipeds.length)
  (h3 : ∀ (i : Fin parallelepipeds.length), is_defective (original_parallelepipeds[i]) (parallelepipeds[i])) :
  ¬ (∀ (reduced_box : RectangularParallelepiped), 
    (reduced_box.length < box.length ∨ reduced_box.width < box.width ∨ reduced_box.height < box.height) → 
    (∀ p ∈ parallelepipeds, fits_in p reduced_box)) :=
by sorry

end NUMINAMATH_CALUDE_not_always_reducible_box_dimension_l2966_296695


namespace NUMINAMATH_CALUDE_nicholas_bottle_caps_l2966_296633

theorem nicholas_bottle_caps :
  let initial_caps : ℕ := 8
  let additional_caps : ℕ := 85
  initial_caps + additional_caps = 93
:= by sorry

end NUMINAMATH_CALUDE_nicholas_bottle_caps_l2966_296633


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2966_296635

theorem constant_term_binomial_expansion :
  (Finset.sum (Finset.range 10) (fun k => Nat.choose 9 k * (1 : ℝ)^k * (1 : ℝ)^(9 - k))) = 84 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2966_296635


namespace NUMINAMATH_CALUDE_intersection_equals_closed_interval_l2966_296647

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ -1}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the closed interval [-1, 3]
def closedInterval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_equals_closed_interval : M ∩ N = closedInterval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_closed_interval_l2966_296647


namespace NUMINAMATH_CALUDE_min_value_inequality_l2966_296657

theorem min_value_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a + 2*b = 1) (h2 : c + 2*d = 1) : 
  1/a + 1/(b*c*d) > 25 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2966_296657


namespace NUMINAMATH_CALUDE_range_of_a_l2966_296658

-- Define propositions p and q
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the set A
def A : Set ℝ := {x | p x}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | q x a}

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a))

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, condition a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2966_296658


namespace NUMINAMATH_CALUDE_average_equation_l2966_296608

theorem average_equation (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 89 → a = 34 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l2966_296608


namespace NUMINAMATH_CALUDE_max_min_sin_cos_combination_l2966_296690

theorem max_min_sin_cos_combination (a b : ℝ) :
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ Real.sqrt (a^2 + b^2)) ∧
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ -Real.sqrt (a^2 + b^2)) ∧
  (∃ θ₁ : ℝ, a * Real.sin θ₁ + b * Real.cos θ₁ = Real.sqrt (a^2 + b^2)) ∧
  (∃ θ₂ : ℝ, a * Real.sin θ₂ + b * Real.cos θ₂ = -Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_min_sin_cos_combination_l2966_296690


namespace NUMINAMATH_CALUDE_kitchen_cleaning_time_l2966_296606

theorem kitchen_cleaning_time (alice_time bob_fraction : ℚ) (h1 : alice_time = 40) (h2 : bob_fraction = 3/8) :
  bob_fraction * alice_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_cleaning_time_l2966_296606


namespace NUMINAMATH_CALUDE_circle_with_rational_center_multiple_lattice_points_l2966_296640

/-- A point in the 2D plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- A circle in the 2D plane -/
structure Circle where
  center : RationalPoint
  radius : ℝ

/-- A lattice point in the 2D plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Check if a lattice point is on the circumference of a circle -/
def isOnCircumference (c : Circle) (p : LatticePoint) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem statement -/
theorem circle_with_rational_center_multiple_lattice_points
  (K : RationalPoint) (c : Circle) (p : LatticePoint) :
  c.center = K → isOnCircumference c p →
  ∃ q : LatticePoint, q ≠ p ∧ isOnCircumference c q :=
by sorry

end NUMINAMATH_CALUDE_circle_with_rational_center_multiple_lattice_points_l2966_296640


namespace NUMINAMATH_CALUDE_sum_of_products_zero_l2966_296602

theorem sum_of_products_zero 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 43) :
  x*y + y*z + x*z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_zero_l2966_296602


namespace NUMINAMATH_CALUDE_estimate_two_sqrt_five_l2966_296644

theorem estimate_two_sqrt_five : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 := by
  sorry

end NUMINAMATH_CALUDE_estimate_two_sqrt_five_l2966_296644


namespace NUMINAMATH_CALUDE_system_negative_solution_l2966_296605

/-- The system of equations has at least one negative solution if and only if a + b + c = 0 -/
theorem system_negative_solution (a b c : ℝ) :
  (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧
    a * x + b * y = c ∧
    b * x + c * y = a ∧
    c * x + a * y = b) ↔
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_negative_solution_l2966_296605


namespace NUMINAMATH_CALUDE_distribute_10_4_l2966_296671

/-- The number of ways to distribute n identical objects among k distinct containers,
    where each container must have at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 10 identical objects among 4 distinct containers,
    where each container must have at least one object, results in 34 unique arrangements. -/
theorem distribute_10_4 : distribute 10 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_distribute_10_4_l2966_296671


namespace NUMINAMATH_CALUDE_total_weight_of_rings_l2966_296603

-- Define the weights of the rings
def orange_weight : ℚ := 0.08
def purple_weight : ℚ := 0.33
def white_weight : ℚ := 0.42

-- Theorem statement
theorem total_weight_of_rings : orange_weight + purple_weight + white_weight = 0.83 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_rings_l2966_296603


namespace NUMINAMATH_CALUDE_exists_always_white_cell_l2966_296629

-- Define the grid plane
def GridPlane := ℤ × ℤ

-- Define the state of a cell (Black or White)
inductive CellState
| Black
| White

-- Define the initial state of the grid
def initial_grid : GridPlane → CellState :=
  sorry

-- Define the polygon M
def M : Set GridPlane :=
  sorry

-- Axiom: M covers more than one cell
axiom M_size : ∃ (c1 c2 : GridPlane), c1 ≠ c2 ∧ c1 ∈ M ∧ c2 ∈ M

-- Define a valid shift of M
def valid_shift (s : GridPlane) : Prop :=
  sorry

-- Define the state of the grid after a shift
def shift_grid (g : GridPlane → CellState) (s : GridPlane) : GridPlane → CellState :=
  sorry

-- Define the state of the grid after any number of shifts
def final_grid : GridPlane → CellState :=
  sorry

-- The theorem to prove
theorem exists_always_white_cell :
  ∃ (c : GridPlane), final_grid c = CellState.White :=
sorry

end NUMINAMATH_CALUDE_exists_always_white_cell_l2966_296629


namespace NUMINAMATH_CALUDE_investment_growth_l2966_296623

-- Define the compound interest function
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

-- Define the problem parameters
def initial_investment : ℝ := 15000
def interest_rate : ℝ := 0.04
def investment_time : ℕ := 10

-- State the theorem
theorem investment_growth :
  round (compound_interest initial_investment interest_rate investment_time) = 22204 := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l2966_296623


namespace NUMINAMATH_CALUDE_weight_of_3_moles_HBrO3_l2966_296666

/-- The molecular weight of a single HBrO3 molecule in g/mol -/
def molecular_weight_HBrO3 : ℝ :=
  1.01 + 79.90 + 3 * 16.00

/-- The weight of 3 moles of HBrO3 in grams -/
def weight_3_moles_HBrO3 : ℝ :=
  3 * molecular_weight_HBrO3

theorem weight_of_3_moles_HBrO3 :
  weight_3_moles_HBrO3 = 386.73 := by sorry

end NUMINAMATH_CALUDE_weight_of_3_moles_HBrO3_l2966_296666


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_k_l2966_296656

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := by sorry

-- Theorem 2: Range of k for |k - 1| < g(x)
theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, |k - 1| < g x) → -3 < k ∧ k < 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_k_l2966_296656


namespace NUMINAMATH_CALUDE_shuffle_32_cards_l2966_296636

/-- The number of ways to shuffle a deck of cards -/
def shuffleWays (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of ways to shuffle a deck of 32 cards is 32! -/
theorem shuffle_32_cards : shuffleWays 32 = Nat.factorial 32 := by
  sorry

end NUMINAMATH_CALUDE_shuffle_32_cards_l2966_296636


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l2966_296665

theorem yellow_highlighters_count (total : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : total = 12) 
  (h2 : pink = 6) 
  (h3 : blue = 4) : 
  total - pink - blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l2966_296665


namespace NUMINAMATH_CALUDE_min_cans_is_281_l2966_296615

/-- The number of liters of Maaza --/
def maaza : ℕ := 50

/-- The number of liters of Pepsi --/
def pepsi : ℕ := 144

/-- The number of liters of Sprite --/
def sprite : ℕ := 368

/-- The function to calculate the minimum number of cans required --/
def min_cans (m p s : ℕ) : ℕ :=
  (m / Nat.gcd m (Nat.gcd p s)) + (p / Nat.gcd m (Nat.gcd p s)) + (s / Nat.gcd m (Nat.gcd p s))

/-- Theorem stating that the minimum number of cans required is 281 --/
theorem min_cans_is_281 : min_cans maaza pepsi sprite = 281 := by
  sorry

end NUMINAMATH_CALUDE_min_cans_is_281_l2966_296615


namespace NUMINAMATH_CALUDE_smallest_multiple_l2966_296696

theorem smallest_multiple (n : ℕ) : n = 459 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 76 * m + 3) ∧ 
  (∀ x : ℕ, x < n → ¬(∃ k : ℕ, x = 17 * k) ∨ ¬(∃ m : ℕ, x = 76 * m + 3)) := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2966_296696


namespace NUMINAMATH_CALUDE_quadruple_solutions_l2966_296638

def is_solution (a b c k : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ k > 0 ∧
  a^2 + b^2 + 16*c^2 = 9*k^2 + 1

theorem quadruple_solutions :
  ∀ a b c k : ℕ,
    is_solution a b c k ↔
      ((a, b, c, k) = (3, 3, 2, 3) ∨
       (a, b, c, k) = (3, 17, 3, 7) ∨
       (a, b, c, k) = (17, 3, 3, 7) ∨
       (a, b, c, k) = (3, 37, 3, 13) ∨
       (a, b, c, k) = (37, 3, 3, 13)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solutions_l2966_296638


namespace NUMINAMATH_CALUDE_function_equality_l2966_296676

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = x^2 - 2*x) :
  ∀ x, f x = x^2 - 1 := by sorry

end NUMINAMATH_CALUDE_function_equality_l2966_296676


namespace NUMINAMATH_CALUDE_same_suit_probability_l2966_296619

theorem same_suit_probability (total_cards : ℕ) (num_suits : ℕ) (cards_per_suit : ℕ) 
  (h1 : total_cards = 52)
  (h2 : num_suits = 4)
  (h3 : cards_per_suit = 13)
  (h4 : total_cards = num_suits * cards_per_suit) :
  (4 : ℚ) / 17 = (num_suits * (cards_per_suit.choose 2)) / (total_cards.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_same_suit_probability_l2966_296619


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l2966_296688

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a)

theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_negative_one_l2966_296688


namespace NUMINAMATH_CALUDE_total_football_games_l2966_296655

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := 9

/-- Theorem: The total number of football games Joan went to is 13 -/
theorem total_football_games : games_this_year + games_last_year = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l2966_296655


namespace NUMINAMATH_CALUDE_original_number_proof_l2966_296687

theorem original_number_proof (x : ℤ) : x = 16 ↔ 
  (∃ k : ℤ, x + 10 = 26 * k) ∧ 
  (∀ y : ℤ, y < 10 → ∀ m : ℤ, x + y ≠ 26 * m) :=
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2966_296687


namespace NUMINAMATH_CALUDE_rally_ticket_cost_l2966_296684

theorem rally_ticket_cost 
  (total_attendance : ℕ)
  (door_ticket_price : ℚ)
  (total_receipts : ℚ)
  (pre_rally_tickets : ℕ)
  (h1 : total_attendance = 750)
  (h2 : door_ticket_price = 2.75)
  (h3 : total_receipts = 1706.25)
  (h4 : pre_rally_tickets = 475) :
  ∃ (pre_rally_price : ℚ), 
    pre_rally_price * pre_rally_tickets + 
    door_ticket_price * (total_attendance - pre_rally_tickets) = total_receipts ∧
    pre_rally_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_rally_ticket_cost_l2966_296684


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2966_296611

theorem quadratic_equal_roots : ∃ x : ℝ, x^2 - x + (1/4 : ℝ) = 0 ∧
  ∀ y : ℝ, y^2 - y + (1/4 : ℝ) = 0 → y = x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2966_296611


namespace NUMINAMATH_CALUDE_solve_equation_l2966_296648

theorem solve_equation (y : ℤ) (h : 7 - y = 13) : y = -6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2966_296648


namespace NUMINAMATH_CALUDE_infinite_intersection_l2966_296664

def sequence_a : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

theorem infinite_intersection :
  Set.Infinite {n : ℕ | ∃ m : ℕ, sequence_a n = sequence_b m} :=
sorry

end NUMINAMATH_CALUDE_infinite_intersection_l2966_296664


namespace NUMINAMATH_CALUDE_angle_theta_trig_values_l2966_296654

/-- An angle θ with vertex at the origin, initial side along positive x-axis, and terminal side on y = 2x -/
structure AngleTheta where
  terminal_side : ∀ (x y : ℝ), y = 2 * x

theorem angle_theta_trig_values (θ : AngleTheta) :
  ∃ (s c : ℝ),
    s^2 + c^2 = 1 ∧
    |s| = 2 * Real.sqrt 5 / 5 ∧
    |c| = Real.sqrt 5 / 5 ∧
    s / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_theta_trig_values_l2966_296654


namespace NUMINAMATH_CALUDE_equation_rewrite_l2966_296659

theorem equation_rewrite (a x y c : ℤ) :
  ∃ (m n p : ℕ), 
    (m = 4 ∧ n = 3 ∧ p = 4) ∧
    (a^8*x*y - a^7*y - a^6*x = a^5*(c^5 - 1)) ↔ 
    ((a^m*x - a^n)*(a^p*y - a^3) = a^5*c^5) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l2966_296659


namespace NUMINAMATH_CALUDE_roots_have_nonzero_imag_l2966_296680

/-- The complex number i such that i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation 5z^2 + 2iz - m = 0 -/
def equation (z : ℂ) (m : ℝ) : Prop := 5 * z^2 + 2 * i * z - m = 0

/-- A complex number has a non-zero imaginary part -/
def has_nonzero_imag (z : ℂ) : Prop := z.im ≠ 0

/-- Both roots of the equation have non-zero imaginary parts for all real m -/
theorem roots_have_nonzero_imag :
  ∀ m : ℝ, ∀ z : ℂ, equation z m → has_nonzero_imag z :=
sorry

end NUMINAMATH_CALUDE_roots_have_nonzero_imag_l2966_296680


namespace NUMINAMATH_CALUDE_nancy_savings_l2966_296677

-- Define the number of quarters in a dozen
def dozen_quarters : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem nancy_savings : (dozen_quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l2966_296677


namespace NUMINAMATH_CALUDE_find_lighter_orange_l2966_296663

/-- Represents a group of objects that can be weighed -/
structure WeightGroup where
  objects : Finset ℕ
  size : ℕ
  h_size : objects.card = size

/-- Represents the result of weighing two groups -/
inductive WeighResult
  | Left
  | Right
  | Equal

/-- Represents a balance scale that can compare two groups -/
def Balance := WeightGroup → WeightGroup → WeighResult

/-- The problem setup with 8 objects, 7 of equal weight and 1 lighter -/
structure OrangeSetup where
  total_objects : ℕ
  h_total : total_objects = 8
  equal_weight_objects : ℕ
  h_equal : equal_weight_objects = 7
  h_lighter : total_objects = equal_weight_objects + 1

/-- The theorem stating that the lighter object can be found in at most 2 measurements -/
theorem find_lighter_orange (setup : OrangeSetup) :
  ∃ (strategy : Balance → Balance → ℕ),
    ∀ (b : Balance), strategy b b < setup.total_objects ∧ 
    (strategy b b) ∈ Finset.range setup.total_objects := by
  sorry


end NUMINAMATH_CALUDE_find_lighter_orange_l2966_296663


namespace NUMINAMATH_CALUDE_max_score_in_range_score_2079_is_in_range_l2966_296673

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_score_in_range :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → score x ≤ score 2079 :=
by sorry

theorem score_2079 : score 2079 = 30 :=
by sorry

theorem is_in_range : 2017 ≤ 2079 ∧ 2079 ≤ 2117 :=
by sorry

end NUMINAMATH_CALUDE_max_score_in_range_score_2079_is_in_range_l2966_296673


namespace NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l2966_296672

/-- Calculates the number of cans of pie filling produced given the total pumpkins,
    price per pumpkin, total money made, and pumpkins per can. -/
def cans_of_pie_filling (total_pumpkins : ℕ) (price_per_pumpkin : ℕ) 
                        (total_made : ℕ) (pumpkins_per_can : ℕ) : ℕ :=
  (total_pumpkins - total_made / price_per_pumpkin) / pumpkins_per_can

/-- Theorem stating that given the specific conditions, 
    the number of cans of pie filling produced is 17. -/
theorem pumpkin_patch_pie_filling : 
  cans_of_pie_filling 83 3 96 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l2966_296672


namespace NUMINAMATH_CALUDE_binomial_inequality_l2966_296670

theorem binomial_inequality (n : ℕ) : 2 ≤ (1 + 1 / n)^n ∧ (1 + 1 / n)^n < 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l2966_296670


namespace NUMINAMATH_CALUDE_specific_theater_seats_l2966_296682

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increment + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with specific seat arrangement has 416 seats -/
theorem specific_theater_seats :
  let t : Theater := {
    first_row_seats := 14,
    seat_increment := 3,
    last_row_seats := 50
  }
  total_seats t = 416 := by
  sorry


end NUMINAMATH_CALUDE_specific_theater_seats_l2966_296682


namespace NUMINAMATH_CALUDE_number_percentage_problem_l2966_296642

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 35 → (40/100 : ℝ) * N = 420 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l2966_296642


namespace NUMINAMATH_CALUDE_inequality_proof_l2966_296650

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2966_296650


namespace NUMINAMATH_CALUDE_even_count_in_pascal_triangle_l2966_296694

/-- Pascal's Triangle coefficient -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Count even numbers in a single row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ :=
  (List.range (row + 1)).filter (fun k => isEven (binomial row k)) |>.length

/-- Count even numbers in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ :=
  (List.range n).map countEvenInRow |>.sum

/-- Theorem: There are 64 even integers in the first 15 rows of Pascal's Triangle -/
theorem even_count_in_pascal_triangle : countEvenInTriangle 15 = 64 := by
  sorry

end NUMINAMATH_CALUDE_even_count_in_pascal_triangle_l2966_296694


namespace NUMINAMATH_CALUDE_sum_of_coordinates_equals_16_l2966_296610

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y - 2)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem sum_of_coordinates_equals_16 :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₁, y₁) ≠ (x₄, y₄) ∧
    (x₂, y₂) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₄, y₄) ∧ (x₃, y₃) ≠ (x₄, y₄) ∧
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_equals_16_l2966_296610


namespace NUMINAMATH_CALUDE_x_less_than_one_iff_x_abs_x_less_than_one_l2966_296674

theorem x_less_than_one_iff_x_abs_x_less_than_one (x : ℝ) : x < 1 ↔ x * |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_one_iff_x_abs_x_less_than_one_l2966_296674


namespace NUMINAMATH_CALUDE_nickel_count_l2966_296614

theorem nickel_count (total_value : ℚ) (nickel_value : ℚ) (quarter_value : ℚ) :
  total_value = 12 →
  nickel_value = 0.05 →
  quarter_value = 0.25 →
  ∃ n : ℕ, n * nickel_value + n * quarter_value = total_value ∧ n = 40 :=
by sorry

end NUMINAMATH_CALUDE_nickel_count_l2966_296614


namespace NUMINAMATH_CALUDE_anne_distance_l2966_296626

/-- Calculates the distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that wandering for 5 hours at 4 miles per hour results in a distance of 20 miles -/
theorem anne_distance : distance 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_anne_distance_l2966_296626


namespace NUMINAMATH_CALUDE_max_savings_l2966_296683

structure Flight where
  airline : String
  basePrice : ℕ
  discountPercentage : ℕ
  layovers : ℕ
  travelTime : ℕ

def calculateDiscountedPrice (flight : Flight) : ℚ :=
  flight.basePrice - (flight.basePrice * flight.discountPercentage / 100)

def flightOptions : List Flight := [
  ⟨"Delta Airlines", 850, 20, 1, 6⟩,
  ⟨"United Airlines", 1100, 30, 1, 7⟩,
  ⟨"American Airlines", 950, 25, 2, 9⟩,
  ⟨"Southwest Airlines", 900, 15, 1, 5⟩,
  ⟨"JetBlue Airways", 1200, 40, 0, 4⟩
]

theorem max_savings (options : List Flight := flightOptions) :
  let discountedPrices := options.map calculateDiscountedPrice
  let minPrice := discountedPrices.minimum?
  let maxPrice := discountedPrices.maximum?
  ∀ min max, minPrice = some min → maxPrice = some max →
    max - min = 90 :=
by sorry

end NUMINAMATH_CALUDE_max_savings_l2966_296683


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l2966_296607

/-- Given that Jason initially has 3 Pokemon cards and Benny buys 2 of them,
    prove that Jason will have 1 Pokemon card left. -/
theorem jason_pokemon_cards (initial_cards : ℕ) (cards_bought : ℕ) 
  (h1 : initial_cards = 3)
  (h2 : cards_bought = 2) :
  initial_cards - cards_bought = 1 := by
  sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l2966_296607


namespace NUMINAMATH_CALUDE_stone_fall_time_exists_stone_fall_time_approx_l2966_296692

theorem stone_fall_time_exists : ∃ s : ℝ, s > 0 ∧ -4.5 * s^2 - 12 * s + 48 = 0 := by
  sorry

theorem stone_fall_time_approx (s : ℝ) (hs : s > 0 ∧ -4.5 * s^2 - 12 * s + 48 = 0) : 
  ∃ ε > 0, |s - 3.82| < ε := by
  sorry

end NUMINAMATH_CALUDE_stone_fall_time_exists_stone_fall_time_approx_l2966_296692


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2966_296662

/-- Given a square of side length 2, divided into a central square and two congruent trapezoids,
    if the areas are equal, then the longer parallel side of a trapezoid is 1. -/
theorem trapezoid_side_length (s : ℝ) : 
  2 > 0 ∧                             -- Square side length is positive
  s > 0 ∧                             -- Central square side length is positive
  s < 2 ∧                             -- Central square fits inside the larger square
  s^2 = (1 + s) / 2 →                 -- Areas are equal
  s = 1 :=                            -- Longer parallel side of trapezoid is 1
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2966_296662


namespace NUMINAMATH_CALUDE_smallest_x_for_quadratic_inequality_l2966_296667

theorem smallest_x_for_quadratic_inequality :
  ∃ x₀ : ℝ, x₀ = 3 ∧
  (∀ x : ℝ, x^2 - 8*x + 15 ≤ 0 → x ≥ x₀) ∧
  (x₀^2 - 8*x₀ + 15 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_quadratic_inequality_l2966_296667


namespace NUMINAMATH_CALUDE_complex_division_by_i_l2966_296620

theorem complex_division_by_i (i : ℂ) (h : i^2 = -1) : 
  (3 + 4*i) / i = 4 - 3*i := by sorry

end NUMINAMATH_CALUDE_complex_division_by_i_l2966_296620


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2966_296612

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 2 → x - 1 > 0) ∧
  ¬(∀ x : ℝ, x - 1 > 0 → x > 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2966_296612


namespace NUMINAMATH_CALUDE_problem_solution_l2966_296646

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_solution : 
  avg3 (avg3 2 3 (-1)) (avg2 1 2) 1 = 23 / 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2966_296646


namespace NUMINAMATH_CALUDE_piglet_straws_l2966_296613

theorem piglet_straws (total_straws : ℕ) (adult_pig_fraction : ℚ) (num_piglets : ℕ) :
  total_straws = 300 →
  adult_pig_fraction = 3/5 →
  num_piglets = 20 →
  (adult_pig_fraction * total_straws : ℚ) = (total_straws - adult_pig_fraction * total_straws : ℚ) →
  (total_straws - adult_pig_fraction * total_straws) / num_piglets = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_piglet_straws_l2966_296613


namespace NUMINAMATH_CALUDE_dot_product_theorem_l2966_296604

variable (a b : ℝ × ℝ)

theorem dot_product_theorem (h1 : a.1 + 2 * b.1 = 0 ∧ a.2 + 2 * b.2 = 0) 
                            (h2 : (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 2) : 
  a.1 * b.1 + a.2 * b.2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l2966_296604


namespace NUMINAMATH_CALUDE_solution_sets_union_l2966_296693

-- Define the solution sets A and B
def A (p q : ℝ) : Set ℝ := {x | x^2 - (p-1)*x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 + (q-1)*x + p = 0}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (∃ (p q : ℝ), A p q ∩ B p q = {-2}) →
  (∃ (p q : ℝ), A p q ∪ B p q = {-2, -1, 1}) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_union_l2966_296693


namespace NUMINAMATH_CALUDE_class_weight_problem_l2966_296675

theorem class_weight_problem (total_boys : Nat) (group_boys : Nat) (group_avg : Real) (total_avg : Real) :
  total_boys = 34 →
  group_boys = 26 →
  group_avg = 50.25 →
  total_avg = 49.05 →
  let remaining_boys := total_boys - group_boys
  let remaining_avg := (total_boys * total_avg - group_boys * group_avg) / remaining_boys
  remaining_avg = 45.15 := by
  sorry

end NUMINAMATH_CALUDE_class_weight_problem_l2966_296675


namespace NUMINAMATH_CALUDE_unwatered_rosebushes_l2966_296691

/-- The number of unwatered rosebushes in Anna and Vitya's garden -/
theorem unwatered_rosebushes 
  (total : ℕ) 
  (vitya_watered : ℕ) 
  (anna_watered : ℕ) 
  (both_watered : ℕ)
  (h1 : total = 2006)
  (h2 : vitya_watered = total / 2)
  (h3 : anna_watered = total / 2)
  (h4 : both_watered = 3) :
  total - (vitya_watered + anna_watered - both_watered) = 3 :=
by sorry

end NUMINAMATH_CALUDE_unwatered_rosebushes_l2966_296691


namespace NUMINAMATH_CALUDE_mikes_score_l2966_296643

theorem mikes_score (max_score : ℕ) (pass_percentage : ℚ) (shortfall : ℕ) (actual_score : ℕ) : 
  max_score = 750 → 
  pass_percentage = 30 / 100 → 
  shortfall = 13 → 
  actual_score = max_score * pass_percentage - shortfall →
  actual_score = 212 :=
by sorry

end NUMINAMATH_CALUDE_mikes_score_l2966_296643


namespace NUMINAMATH_CALUDE_circle_radius_spherical_coordinates_l2966_296660

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/4) is √2/2 --/
theorem circle_radius_spherical_coordinates :
  let r := Real.sqrt ((Real.sin (π/4))^2 + (Real.cos (π/4))^2)
  r = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_spherical_coordinates_l2966_296660


namespace NUMINAMATH_CALUDE_remainder_congruence_l2966_296669

theorem remainder_congruence (x : ℤ) 
  (h1 : (2 + x) % 8 = 9 % 8)
  (h2 : (3 + x) % 27 = 4 % 27)
  (h3 : (11 + x) % 1331 = 49 % 1331) :
  x % 198 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_congruence_l2966_296669


namespace NUMINAMATH_CALUDE_sin_29pi_over_6_l2966_296645

theorem sin_29pi_over_6 : Real.sin (29 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_29pi_over_6_l2966_296645


namespace NUMINAMATH_CALUDE_knight_traversal_coloring_l2966_296637

/-- Represents a chessboard of arbitrary size -/
structure Chessboard where
  size : ℕ
  canBeTraversed : Bool

/-- Represents a position on the chessboard -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents a knight's move -/
def knightMove (p : Position) : Position :=
  sorry

/-- Checks if a position is even in the knight's traversal -/
def isEvenPosition (p : Position) : Bool :=
  sorry

/-- Checks if a position should be colored black in a properly colored chessboard -/
def isBlackInProperColoring (p : Position) : Bool :=
  sorry

/-- The main theorem stating that shading even-numbered squares in a knight's traversal
    reproduces the proper coloring of a chessboard -/
theorem knight_traversal_coloring (board : Chessboard) :
  board.canBeTraversed →
  ∀ p : Position, isEvenPosition p = isBlackInProperColoring p :=
sorry

end NUMINAMATH_CALUDE_knight_traversal_coloring_l2966_296637


namespace NUMINAMATH_CALUDE_limit_cubic_fraction_l2966_296609

theorem limit_cubic_fraction : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((x^3 - 1) / (x - 1)) - 3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_cubic_fraction_l2966_296609


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2966_296634

theorem simplify_and_evaluate :
  ∀ x : ℝ, x ≠ 1 → x ≠ 3 →
  (1 - 2 / (x - 1)) * ((x^2 - x) / (x^2 - 6*x + 9)) = x / (x - 3) ∧
  (2 : ℝ) / ((2 : ℝ) - 3) = -2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2966_296634


namespace NUMINAMATH_CALUDE_choose_two_from_four_with_repetition_l2966_296661

/-- The number of ways to choose r items from n items with repetition allowed -/
def combinationsWithRepetition (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose (n + r - 1) r

theorem choose_two_from_four_with_repetition :
  combinationsWithRepetition 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_four_with_repetition_l2966_296661


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2966_296616

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) =
  -4 * x^4 + x^3 + 3 * x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2966_296616


namespace NUMINAMATH_CALUDE_jake_candy_cost_l2966_296679

/-- The cost of a single candy given Jake's feeding allowance and sharing behavior -/
def candy_cost (feeding_allowance : ℚ) (share_fraction : ℚ) (candies_bought : ℕ) : ℚ :=
  (feeding_allowance * share_fraction) / candies_bought

theorem jake_candy_cost :
  candy_cost 4 (1/4) 5 = 1/5 := by sorry

end NUMINAMATH_CALUDE_jake_candy_cost_l2966_296679


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l2966_296651

theorem quadratic_inequality_implies_range (x : ℝ) : 
  x^2 - 7*x + 12 < 0 → 42 < x^2 + 7*x + 12 ∧ x^2 + 7*x + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_range_l2966_296651


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2966_296621

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = i * (i - 1) → Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2966_296621


namespace NUMINAMATH_CALUDE_fraction_sum_l2966_296622

theorem fraction_sum : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_l2966_296622


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_45_is_smallest_is_four_digit_and_divisible_by_45_l2966_296689

theorem smallest_four_digit_divisible_by_45 : ℕ :=
  let is_four_digit (n : ℕ) := 1000 ≤ n ∧ n ≤ 9999
  let is_divisible_by_45 (n : ℕ) := n % 45 = 0
  1008

theorem is_smallest :
  let is_four_digit (n : ℕ) := 1000 ≤ n ∧ n ≤ 9999
  let is_divisible_by_45 (n : ℕ) := n % 45 = 0
  ∀ n : ℕ, is_four_digit n → is_divisible_by_45 n → smallest_four_digit_divisible_by_45 ≤ n :=
by
  sorry

theorem is_four_digit_and_divisible_by_45 :
  let is_four_digit (n : ℕ) := 1000 ≤ n ∧ n ≤ 9999
  let is_divisible_by_45 (n : ℕ) := n % 45 = 0
  is_four_digit smallest_four_digit_divisible_by_45 ∧ is_divisible_by_45 smallest_four_digit_divisible_by_45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_45_is_smallest_is_four_digit_and_divisible_by_45_l2966_296689


namespace NUMINAMATH_CALUDE_anthony_percentage_more_than_mabel_l2966_296649

theorem anthony_percentage_more_than_mabel :
  ∀ (mabel anthony cal jade : ℕ),
    mabel = 90 →
    cal = (2 * anthony) / 3 →
    jade = cal + 18 →
    jade = 84 →
    (anthony : ℚ) / mabel = 11 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_anthony_percentage_more_than_mabel_l2966_296649


namespace NUMINAMATH_CALUDE_dans_age_l2966_296617

theorem dans_age (x : ℕ) : x + 20 = 7 * (x - 4) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_l2966_296617


namespace NUMINAMATH_CALUDE_whoosit_count_2_l2966_296697

def worker_count_1 : ℕ := 150
def widget_count_1 : ℕ := 450
def whoosit_count_1 : ℕ := 300
def hours_1 : ℕ := 1

def worker_count_2 : ℕ := 90
def widget_count_2 : ℕ := 540
def hours_2 : ℕ := 3

def worker_count_3 : ℕ := 75
def widget_count_3 : ℕ := 300
def whoosit_count_3 : ℕ := 400
def hours_3 : ℕ := 4

def widget_production_rate_1 : ℚ := widget_count_1 / (worker_count_1 * hours_1)
def whoosit_production_rate_1 : ℚ := whoosit_count_1 / (worker_count_1 * hours_1)

def widget_production_rate_3 : ℚ := widget_count_3 / (worker_count_3 * hours_3)
def whoosit_production_rate_3 : ℚ := whoosit_count_3 / (worker_count_3 * hours_3)

theorem whoosit_count_2 (h : 2 * whoosit_production_rate_3 = widget_production_rate_3) :
  ∃ n : ℕ, n = 360 ∧ n / (worker_count_2 * hours_2) = whoosit_production_rate_3 :=
by sorry

end NUMINAMATH_CALUDE_whoosit_count_2_l2966_296697


namespace NUMINAMATH_CALUDE_equation_solution_l2966_296628

theorem equation_solution (x : ℝ) : 
  (x / 4) / 2 = 4 / (x / 2) → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2966_296628


namespace NUMINAMATH_CALUDE_first_year_after_2010_sum_15_is_correct_l2966_296641

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Check if a year is after 2010 -/
def is_after_2010 (year : ℕ) : Prop :=
  year > 2010

/-- First year after 2010 with sum of digits equal to 15 -/
def first_year_after_2010_sum_15 : ℕ :=
  2039

theorem first_year_after_2010_sum_15_is_correct :
  (is_after_2010 first_year_after_2010_sum_15) ∧ 
  (sum_of_digits first_year_after_2010_sum_15 = 15) ∧
  (∀ y : ℕ, is_after_2010 y ∧ y < first_year_after_2010_sum_15 → sum_of_digits y ≠ 15) :=
by sorry

end NUMINAMATH_CALUDE_first_year_after_2010_sum_15_is_correct_l2966_296641


namespace NUMINAMATH_CALUDE_power_of_five_reciprocal_l2966_296699

theorem power_of_five_reciprocal (x y : ℕ) : 
  (2^x : ℕ) ∣ 144 ∧ 
  (∀ k > x, ¬((2^k : ℕ) ∣ 144)) ∧ 
  (3^y : ℕ) ∣ 144 ∧ 
  (∀ k > y, ¬((3^k : ℕ) ∣ 144)) →
  (1/5 : ℚ)^(y - x) = 25 := by
sorry

end NUMINAMATH_CALUDE_power_of_five_reciprocal_l2966_296699


namespace NUMINAMATH_CALUDE_fraction_difference_zero_l2966_296698

theorem fraction_difference_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + b = a * b) : 
  1 / a - 1 / b = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_zero_l2966_296698


namespace NUMINAMATH_CALUDE_probability_six_spades_correct_l2966_296624

/-- The number of cards in a standard deck of poker cards (excluding jokers) -/
def deck_size : ℕ := 52

/-- The number of spades in a standard deck of poker cards -/
def spades_count : ℕ := 13

/-- The number of cards each player receives when 4 people play -/
def cards_per_player : ℕ := deck_size / 4

/-- The probability of a person getting exactly 6 spades when 4 people play with a standard deck -/
def probability_six_spades : ℚ :=
  (Nat.choose spades_count 6 * Nat.choose (deck_size - spades_count) (cards_per_player - 6)) /
  Nat.choose deck_size cards_per_player

theorem probability_six_spades_correct :
  probability_six_spades = (Nat.choose 13 6 * Nat.choose 39 7) / Nat.choose 52 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_spades_correct_l2966_296624


namespace NUMINAMATH_CALUDE_triangle_theorem_l2966_296627

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The perimeter of the triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The condition given in the problem -/
def condition (t : Triangle) : Prop :=
  (4 * area t) / Real.tan t.B = t.a^2 * Real.cos t.B + t.a * t.b * Real.cos t.A

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h : condition t) :
  t.B = π/3 ∧ 
  (∃ (t' : Triangle), t'.b = 3 ∧ 
    (∀ (t'' : Triangle), t''.b = 3 → 
      area t'' / perimeter t'' ≤ area t' / perimeter t' ∧ 
      area t' / perimeter t' = Real.sqrt 3 / 4)) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2966_296627


namespace NUMINAMATH_CALUDE_place_mat_length_l2966_296618

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) :
  r = 5 →
  n = 8 →
  w = 1 →
  (x - w/2)^2 + (w/2)^2 = r^2 →
  x = (3 * Real.sqrt 11 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_place_mat_length_l2966_296618


namespace NUMINAMATH_CALUDE_circular_arrangement_multiple_of_four_l2966_296630

/-- Represents a child in the circular arrangement -/
inductive Child
| Boy
| Girl

/-- Represents the circular arrangement of children -/
def CircularArrangement := List Child

/-- Counts the number of children whose right-hand neighbor is of the same gender -/
def countSameGenderNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of children whose right-hand neighbor is of a different gender -/
def countDifferentGenderNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if the arrangement satisfies the equal neighbor condition -/
def hasEqualNeighbors (arrangement : CircularArrangement) : Prop :=
  countSameGenderNeighbors arrangement = countDifferentGenderNeighbors arrangement

theorem circular_arrangement_multiple_of_four 
  (arrangement : CircularArrangement) 
  (h : hasEqualNeighbors arrangement) :
  ∃ k : Nat, arrangement.length = 4 * k :=
sorry

end NUMINAMATH_CALUDE_circular_arrangement_multiple_of_four_l2966_296630


namespace NUMINAMATH_CALUDE_greatest_perimeter_l2966_296681

/-- A rectangle with whole number side lengths and an area of 12 square metres. -/
structure Rectangle where
  width : ℕ
  length : ℕ
  area_eq : width * length = 12

/-- The perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- The theorem stating that the greatest possible perimeter is 26. -/
theorem greatest_perimeter :
  ∀ r : Rectangle, perimeter r ≤ 26 ∧ ∃ r' : Rectangle, perimeter r' = 26 := by
  sorry

end NUMINAMATH_CALUDE_greatest_perimeter_l2966_296681


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2966_296600

theorem absolute_value_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2966_296600


namespace NUMINAMATH_CALUDE_tom_video_game_spending_l2966_296653

/-- The amount Tom spent on the Batman game -/
def batman_game_cost : ℚ := 13.6

/-- The amount Tom spent on the Superman game -/
def superman_game_cost : ℚ := 5.06

/-- The total amount Tom spent on video games -/
def total_spent : ℚ := batman_game_cost + superman_game_cost

theorem tom_video_game_spending :
  total_spent = 18.66 := by sorry

end NUMINAMATH_CALUDE_tom_video_game_spending_l2966_296653


namespace NUMINAMATH_CALUDE_twelve_integers_with_properties_l2966_296652

theorem twelve_integers_with_properties : ∃ (S : Finset ℤ),
  (Finset.card S = 12) ∧
  (∃ (P : Finset ℤ), P ⊆ S ∧ Finset.card P = 6 ∧ ∀ p ∈ P, Nat.Prime p.natAbs) ∧
  (∃ (O : Finset ℤ), O ⊆ S ∧ Finset.card O = 9 ∧ ∀ n ∈ O, n % 2 ≠ 0) ∧
  (∃ (NN : Finset ℤ), NN ⊆ S ∧ Finset.card NN = 10 ∧ ∀ n ∈ NN, n ≥ 0) ∧
  (∃ (GT : Finset ℤ), GT ⊆ S ∧ Finset.card GT = 7 ∧ ∀ n ∈ GT, n > 10) :=
by
  sorry


end NUMINAMATH_CALUDE_twelve_integers_with_properties_l2966_296652


namespace NUMINAMATH_CALUDE_cube_root_fourth_power_equals_81_l2966_296678

theorem cube_root_fourth_power_equals_81 (y : ℝ) : (y^(1/3))^4 = 81 → y = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fourth_power_equals_81_l2966_296678


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2966_296601

/-- The minimum value of a quadratic function y = (x - a)(x - b) -/
theorem quadratic_minimum_value (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) :
  ∃ x₀, ∀ x, (x - a) * (x - b) ≥ (x₀ - a) * (x₀ - b) ∧ 
  (x₀ - a) * (x₀ - b) = -(|a - b| / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2966_296601


namespace NUMINAMATH_CALUDE_slope_is_two_l2966_296625

/-- A linear function y = kx + b where y increases by 6 when x increases by 3 -/
structure LinearFunction where
  k : ℝ
  b : ℝ
  increase_property : ∀ (x : ℝ), (k * (x + 3) + b) - (k * x + b) = 6

/-- Theorem: The slope k of the linear function is equal to 2 -/
theorem slope_is_two (f : LinearFunction) : f.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_is_two_l2966_296625


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2966_296668

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y a : ℝ) : ℝ := x^2 + y^2 + 2*a*x + 2*y + (a-1)^2

/-- Predicate to check if a point (x, y) is outside the circle -/
def is_outside_circle (x y a : ℝ) : Prop := circle_equation x y a > 0

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) : 
  is_outside_circle 0 0 a :=
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2966_296668


namespace NUMINAMATH_CALUDE_homework_problems_exist_l2966_296685

theorem homework_problems_exist : ∃ (a b c d : ℤ), 
  (a ≤ -1) ∧ (b ≤ -1) ∧ (c ≤ -1) ∧ (d ≤ -1) ∧ 
  (a * b = -(a + b)) ∧ 
  (c * d = -182 * (1 / (c + d))) :=
sorry

end NUMINAMATH_CALUDE_homework_problems_exist_l2966_296685


namespace NUMINAMATH_CALUDE_circle_equations_correct_l2966_296632

-- Define the parallel lines
def line1 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + y + Real.sqrt 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 2 * Real.sqrt 2 * a = 0

-- Define the circle N
def circleN (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 5)^2 + (y + 2)^2 = 49

-- Define point B
def pointB : ℝ × ℝ := (3, -2)

-- Define the line of symmetry
def lineOfSymmetry (x : ℝ) : Prop := x = -1

-- Main theorem
theorem circle_equations_correct :
  ∃ (a : ℝ),
    (∀ x y, line1 a x y ↔ line2 a x y) ∧  -- Lines are parallel
    (∃ r, r > 0 ∧ r = Real.sqrt ((3 - (-5))^2 + (4 - (-2))^2) - 3) ∧  -- Distance between centers minus radius of N
    (∀ x, lineOfSymmetry x → x = -1) ∧
    (∃ c : ℝ × ℝ, c.1 = -5 ∧ c.2 = -2) →  -- Point C exists
  (∀ x y, circleN x y) ∧ (∀ x y, circleC x y) :=
sorry

end NUMINAMATH_CALUDE_circle_equations_correct_l2966_296632


namespace NUMINAMATH_CALUDE_fifteenth_term_of_ap_l2966_296639

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- Theorem: The 15th term of an arithmetic progression with first term 2 and common difference 3 is 44 -/
theorem fifteenth_term_of_ap : arithmeticProgressionTerm 2 3 15 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_ap_l2966_296639


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2966_296686

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2966_296686


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l2966_296631

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 32)
  (h2 : failed_english = 56)
  (h3 : failed_both = 12) :
  100 - (failed_hindi + failed_english - failed_both) = 24 :=
by sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l2966_296631
