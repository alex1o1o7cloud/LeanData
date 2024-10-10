import Mathlib

namespace remainder_2753_div_98_l3940_394079

theorem remainder_2753_div_98 : 2753 % 98 = 9 := by
  sorry

end remainder_2753_div_98_l3940_394079


namespace smallest_sum_of_reciprocals_l3940_394005

theorem smallest_sum_of_reciprocals (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h : (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15) :
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → a + b ≤ x + y ∧ a + b = 64 := by
  sorry

end smallest_sum_of_reciprocals_l3940_394005


namespace not_repeating_decimal_l3940_394098

/-- Definition of the number we're considering -/
def x : ℚ := 3.66666

/-- Definition of a repeating decimal -/
def is_repeating_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ) (c : ℤ), q = (c : ℚ) + (a : ℚ) / (10^b - 1)

/-- Theorem stating that 3.66666 is not a repeating decimal -/
theorem not_repeating_decimal : ¬ is_repeating_decimal x := by
  sorry

end not_repeating_decimal_l3940_394098


namespace percentage_solution_l3940_394003

/-- The percentage that, when applied to 100 and added to 20, results in 100 -/
def percentage_problem (P : ℝ) : Prop :=
  100 * (P / 100) + 20 = 100

/-- The solution to the percentage problem is 80% -/
theorem percentage_solution : ∃ P : ℝ, percentage_problem P ∧ P = 80 := by
  sorry

end percentage_solution_l3940_394003


namespace quadrant_I_condition_l3940_394008

theorem quadrant_I_condition (k : ℝ) :
  (∃ x y : ℝ, x + 2*y = 6 ∧ k*x - y = 2 ∧ x > 0 ∧ y > 0) ↔ k > 1/3 := by
  sorry

end quadrant_I_condition_l3940_394008


namespace sufficient_not_necessary_condition_l3940_394059

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end sufficient_not_necessary_condition_l3940_394059


namespace number_of_balls_in_box_l3940_394076

theorem number_of_balls_in_box : ∃ n : ℕ, n - 44 = 70 - n ∧ n = 57 := by sorry

end number_of_balls_in_box_l3940_394076


namespace golden_fish_catches_l3940_394054

theorem golden_fish_catches (x y z : ℕ) : 
  4 * x + 2 * z = 1000 →
  2 * y + z = 800 →
  x + y + z = 900 :=
by sorry

end golden_fish_catches_l3940_394054


namespace expression_evaluation_l3940_394058

theorem expression_evaluation : (2 + 6 * 3 - 4) + 2^3 * 4 / 2 = 32 := by
  sorry

end expression_evaluation_l3940_394058


namespace rectangular_solid_surface_area_l3940_394062

-- Define a structure for a rectangular solid
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define properties of the rectangular solid
def isPrime (n : ℕ) : Prop := sorry

def volume (r : RectangularSolid) : ℕ :=
  r.length * r.width * r.height

def surfaceArea (r : RectangularSolid) : ℕ :=
  2 * (r.length * r.width + r.width * r.height + r.height * r.length)

-- Theorem statement
theorem rectangular_solid_surface_area :
  ∀ (r : RectangularSolid),
    isPrime r.length ∧ isPrime r.width ∧ isPrime r.height →
    volume r = 1155 →
    surfaceArea r = 142 := by
  sorry

end rectangular_solid_surface_area_l3940_394062


namespace min_n_for_infinite_moves_l3940_394051

/-- A move in the card game -/
structure Move where
  cards : Finset ℕ
  sum_equals_index : ℕ

/-- The card game setup -/
structure CardGame where
  n : ℕ
  card_count : ℕ → ℕ
  card_count_eq_n : ∀ l : ℕ, card_count l = n

/-- An infinite sequence of moves in the game -/
def InfiniteMoveSequence (game : CardGame) : Type :=
  ℕ → Move

/-- The theorem statement -/
theorem min_n_for_infinite_moves :
  ∀ n : ℕ,
  n ≥ 10000 →
  ∃ (game : CardGame) (moves : InfiniteMoveSequence game),
    (∀ k : ℕ, k > 0 → 
      (moves k).cards.card = 100 ∧
      (moves k).sum_equals_index = k) ∧
  ∀ m : ℕ,
  m < 10000 →
  ¬∃ (game : CardGame) (moves : InfiniteMoveSequence game),
    (∀ k : ℕ, k > 0 → 
      (moves k).cards.card = 100 ∧
      (moves k).sum_equals_index = k) :=
sorry

end min_n_for_infinite_moves_l3940_394051


namespace strawberry_plants_l3940_394056

theorem strawberry_plants (initial : ℕ) : 
  (initial * 2 * 2 * 2 - 4 = 20) → initial = 3 :=
by
  sorry

end strawberry_plants_l3940_394056


namespace product_equals_zero_l3940_394094

theorem product_equals_zero (b : ℤ) (h : b = 5) : 
  ((b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * 
   (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b) = 0 := by
sorry

end product_equals_zero_l3940_394094


namespace probability_of_selection_X_l3940_394074

theorem probability_of_selection_X (p_Y p_XY : ℝ) : 
  p_Y = 2/3 → p_XY = 0.13333333333333333 → ∃ p_X : ℝ, p_X = 0.2 ∧ p_XY = p_X * p_Y :=
by sorry

end probability_of_selection_X_l3940_394074


namespace ladybugs_with_spots_count_l3940_394096

/-- Given the total number of ladybugs and the number of ladybugs without spots,
    calculate the number of ladybugs with spots. -/
def ladybugsWithSpots (total : ℕ) (withoutSpots : ℕ) : ℕ :=
  total - withoutSpots

/-- Theorem stating that given 67,082 total ladybugs and 54,912 ladybugs without spots,
    there are 12,170 ladybugs with spots. -/
theorem ladybugs_with_spots_count :
  ladybugsWithSpots 67082 54912 = 12170 := by
  sorry

end ladybugs_with_spots_count_l3940_394096


namespace quadratic_property_l3940_394067

/-- A quadratic function f(x) = ax² + bx + c with specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > c
  h3 : a + b + c = 0

/-- The set A of m where f(m) < 0 -/
def A (f : QuadraticFunction) : Set ℝ :=
  {m | f.a * m^2 + f.b * m + f.c < 0}

/-- Main theorem: For any m in A, f(m+3) > 0 -/
theorem quadratic_property (f : QuadraticFunction) :
  ∀ m ∈ A f, f.a * (m + 3)^2 + f.b * (m + 3) + f.c > 0 := by
  sorry

end quadratic_property_l3940_394067


namespace parallel_lines_c_value_l3940_394009

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c that makes the given lines parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = (5/2) * x + 5 ↔ y = (3 * c) * x + 3) → c = 5/6 :=
by sorry

end parallel_lines_c_value_l3940_394009


namespace cubic_roots_sum_l3940_394095

-- Define the polynomial
def p (x : ℝ) : ℝ := 10 * x^3 + 502 * x + 3010

-- Define the roots
theorem cubic_roots_sum (a b c : ℝ) (ha : p a = 0) (hb : p b = 0) (hc : p c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 := by
  sorry

end cubic_roots_sum_l3940_394095


namespace family_weight_ratio_l3940_394046

/-- Given the weights of three generations in a family, prove the ratio of the child's weight to the grandmother's weight -/
theorem family_weight_ratio :
  ∀ (grandmother daughter child : ℝ),
  grandmother + daughter + child = 160 →
  daughter + child = 60 →
  daughter = 40 →
  child / grandmother = 1 / 5 := by
sorry

end family_weight_ratio_l3940_394046


namespace min_value_reciprocal_sum_l3940_394017

/-- Given a line and a circle with specific properties, prove the minimum value of 1/a + 1/b --/
theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, 2*a*x - b*y + 2 = 0 ∧ (x + 1)^2 + (y - 2)^2 = 4) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    2*a*x₁ - b*y₁ + 2 = 0 ∧ (x₁ + 1)^2 + (y₁ - 2)^2 = 4 ∧
    2*a*x₂ - b*y₂ + 2 = 0 ∧ (x₂ + 1)^2 + (y₂ - 2)^2 = 4 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  (1/a + 1/b) ≥ 2 :=
by sorry


end min_value_reciprocal_sum_l3940_394017


namespace sum_of_x_and_y_l3940_394000

theorem sum_of_x_and_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 4) (hxy : x * y > 0) :
  x + y = 7 ∨ x + y = -7 := by sorry

end sum_of_x_and_y_l3940_394000


namespace set_representation_implies_sum_of_powers_l3940_394004

theorem set_representation_implies_sum_of_powers (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = {a+b, 0, a^2} → a^2010 + b^2010 = 1 := by
  sorry

end set_representation_implies_sum_of_powers_l3940_394004


namespace complex_square_equality_l3940_394071

theorem complex_square_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : (a : ℂ) + b*i - 2*i = 2 - b*i) : 
  (a + b*i)^2 = 3 + 4*i := by
  sorry

end complex_square_equality_l3940_394071


namespace quiz_logic_l3940_394019

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answers_all_correctly : Student → Prop)
variable (passes_quiz : Student → Prop)

-- State the theorem
theorem quiz_logic (s : Student) 
  (h : ∀ x : Student, answers_all_correctly x → passes_quiz x) :
  ¬(passes_quiz s) → ¬(answers_all_correctly s) :=
by
  sorry

end quiz_logic_l3940_394019


namespace polygon_diagonals_l3940_394028

theorem polygon_diagonals (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (n - 3 = 5) →  -- At most 5 diagonals can be drawn from any vertex
  n = 8 := by
sorry

end polygon_diagonals_l3940_394028


namespace car_rental_cost_l3940_394038

/-- The maximum daily rental cost for a car, given budget and mileage constraints -/
theorem car_rental_cost (budget : ℝ) (max_miles : ℝ) (cost_per_mile : ℝ) :
  budget = 88 ∧ max_miles = 190 ∧ cost_per_mile = 0.2 →
  ∃ (daily_rental : ℝ), daily_rental ≤ 50 ∧ daily_rental + max_miles * cost_per_mile ≤ budget :=
by sorry

end car_rental_cost_l3940_394038


namespace hyperbola_equation_l3940_394030

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 20*y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 5)

-- Define the asymptotes of the hyperbola
def hyperbola_asymptotes (x y : ℝ) : Prop := (3*x + 4*y = 0) ∨ (3*x - 4*y = 0)

-- Define the standard form of a hyperbola
def hyperbola_standard_form (a b : ℝ) (x y : ℝ) : Prop := y^2/a^2 - x^2/b^2 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∃ (x y : ℝ), parabola x y ∧
  (∃ (fx fy : ℝ), (fx, fy) = parabola_focus) ∧
  hyperbola_asymptotes x y →
  hyperbola_standard_form 3 4 x y :=
sorry

end hyperbola_equation_l3940_394030


namespace car_part_cost_l3940_394097

/-- Calculates the cost of a car part given the total repair cost, labor time, and hourly rate. -/
theorem car_part_cost (total_cost labor_time hourly_rate : ℝ) : 
  total_cost = 300 ∧ labor_time = 2 ∧ hourly_rate = 75 → 
  total_cost - (labor_time * hourly_rate) = 150 := by
sorry

end car_part_cost_l3940_394097


namespace division_problem_l3940_394083

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 23 →
  divisor = 5 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  quotient = 4 := by
  sorry

end division_problem_l3940_394083


namespace brenda_spay_problem_l3940_394070

/-- The number of cats Brenda needs to spay -/
def num_cats : ℕ := 7

/-- The number of dogs Brenda needs to spay -/
def num_dogs : ℕ := 2 * num_cats

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := 21

theorem brenda_spay_problem :
  num_cats = 7 ∧ num_dogs = 2 * num_cats ∧ num_cats + num_dogs = total_animals :=
sorry

end brenda_spay_problem_l3940_394070


namespace max_m_inequality_l3940_394066

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m : ℝ, ∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b ≥ m/(2*a+b)) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 2/a + 1/b ≥ m/(2*a+b)) → m ≤ 9) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2/a + 1/b = 9/(2*a+b)) :=
by sorry

end max_m_inequality_l3940_394066


namespace borgnine_chimps_count_l3940_394091

/-- The number of chimps Borgnine has seen at the zoo -/
def num_chimps : ℕ := 25

/-- The total number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of lions Borgnine has seen -/
def num_lions : ℕ := 8

/-- The number of lizards Borgnine has seen -/
def num_lizards : ℕ := 5

/-- The number of tarantulas Borgnine needs to see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a lion, lizard, or chimp has -/
def legs_per_mammal_or_reptile : ℕ := 4

/-- The number of legs a tarantula has -/
def legs_per_tarantula : ℕ := 8

theorem borgnine_chimps_count :
  num_chimps * legs_per_mammal_or_reptile +
  num_lions * legs_per_mammal_or_reptile +
  num_lizards * legs_per_mammal_or_reptile +
  num_tarantulas * legs_per_tarantula = total_legs :=
by sorry

end borgnine_chimps_count_l3940_394091


namespace solution_set_is_open_interval_l3940_394040

-- Define the quadratic function
def f (x : ℝ) := -x^2 - 3*x + 4

-- Define the solution set
def S : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_is_open_interval : S = Set.Ioo (-4 : ℝ) 1 := by sorry

end solution_set_is_open_interval_l3940_394040


namespace g_x_squared_properties_l3940_394034

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem g_x_squared_properties
  (g : ℝ → ℝ)
  (h_sym : symmetric_wrt_y_eq_x f g) :
  (∀ x, g (x^2) = g ((-x)^2)) ∧
  (∀ x y, x < y → x < 0 → y < 0 → g (x^2) < g (y^2)) :=
sorry

end g_x_squared_properties_l3940_394034


namespace opposing_team_score_l3940_394064

theorem opposing_team_score (chucks_team_score : ℕ) (lead : ℕ) (opposing_team_score : ℕ) :
  chucks_team_score = 72 →
  lead = 17 →
  chucks_team_score = opposing_team_score + lead →
  opposing_team_score = 55 := by
sorry

end opposing_team_score_l3940_394064


namespace sequence_term_from_sum_l3940_394093

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := n^2 + 3*n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ := 2*n + 2

theorem sequence_term_from_sum (n : ℕ) : 
  n > 0 → S n - S (n-1) = a n :=
by sorry

end sequence_term_from_sum_l3940_394093


namespace sum_of_roots_quadratic_equation_l3940_394049

theorem sum_of_roots_quadratic_equation :
  let a : ℝ := 6 + 3 * Real.sqrt 3
  let b : ℝ := 3 + Real.sqrt 3
  let c : ℝ := -3
  let sum_of_roots := -b / a
  sum_of_roots = -1 + Real.sqrt 3 / 3 :=
by sorry

end sum_of_roots_quadratic_equation_l3940_394049


namespace middle_three_sum_is_twelve_l3940_394022

/-- Represents a card with a color and a number -/
inductive Card
  | red (n : Nat)
  | blue (n : Nat)

/-- Checks if a number divides another number -/
def divides (a b : Nat) : Bool :=
  b % a == 0

/-- Checks if a stack of cards satisfies the alternating color and division rules -/
def validStack (stack : List Card) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | (Card.blue b) :: (Card.red r) :: (Card.blue b') :: rest =>
      divides r b && divides r b' && validStack ((Card.red r) :: (Card.blue b') :: rest)
  | _ => false

/-- Returns the sum of the numbers on the middle three cards -/
def middleThreeSum (stack : List Card) : Nat :=
  let mid := stack.length / 2
  match (stack.get? (mid - 1), stack.get? mid, stack.get? (mid + 1)) with
  | (some (Card.blue b1), some (Card.red r), some (Card.blue b2)) => b1 + r + b2
  | _ => 0

/-- The main theorem -/
theorem middle_three_sum_is_twelve :
  ∃ (stack : List Card),
    stack.length = 9 ∧
    stack.head? = some (Card.blue 2) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 4 → (Card.red n) ∈ stack) ∧
    (∀ n, 2 ≤ n ∧ n ≤ 6 → (Card.blue n) ∈ stack) ∧
    validStack stack ∧
    middleThreeSum stack = 12 :=
  sorry


end middle_three_sum_is_twelve_l3940_394022


namespace triangle_inequality_violation_l3940_394012

theorem triangle_inequality_violation
  (a b c d e : ℝ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e)
  (sum_equality : a^2 + b^2 + c^2 + d^2 + e^2 = a*b + a*c + a*d + a*e + b*c + b*d + b*e + c*d + c*e + d*e) :
  ∃ (x y z : ℝ), (x = a ∧ y = b ∧ z = c) ∨ (x = a ∧ y = b ∧ z = d) ∨ (x = a ∧ y = b ∧ z = e) ∨
                 (x = a ∧ y = c ∧ z = d) ∨ (x = a ∧ y = c ∧ z = e) ∨ (x = a ∧ y = d ∧ z = e) ∨
                 (x = b ∧ y = c ∧ z = d) ∨ (x = b ∧ y = c ∧ z = e) ∨ (x = b ∧ y = d ∧ z = e) ∨
                 (x = c ∧ y = d ∧ z = e) ∧
                 (x + y ≤ z ∨ y + z ≤ x ∨ z + x ≤ y) :=
by sorry

end triangle_inequality_violation_l3940_394012


namespace tetrahedron_altitude_exsphere_relation_l3940_394036

/-- A tetrahedron with its altitudes and exsphere radii -/
structure Tetrahedron where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  r₄ : ℝ
  h_pos : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ h₄ > 0
  r_pos : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0

/-- The theorem about the relationship between altitudes and exsphere radii in a tetrahedron -/
theorem tetrahedron_altitude_exsphere_relation (t : Tetrahedron) :
  2 * (1 / t.h₁ + 1 / t.h₂ + 1 / t.h₃ + 1 / t.h₄) =
  1 / t.r₁ + 1 / t.r₂ + 1 / t.r₃ + 1 / t.r₄ := by
  sorry

end tetrahedron_altitude_exsphere_relation_l3940_394036


namespace magnitude_a_minus_2b_equals_sqrt_17_l3940_394021

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_a_minus_2b_equals_sqrt_17 :
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 17 := by
  sorry

end magnitude_a_minus_2b_equals_sqrt_17_l3940_394021


namespace system_solution_unique_l3940_394087

theorem system_solution_unique :
  ∃! (x y : ℝ), x + 2*y = 5 ∧ 3*x + y = 5 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l3940_394087


namespace hairdresser_cash_register_l3940_394044

theorem hairdresser_cash_register (x : ℝ) : 
  (8 * x - 70 = 0) → x = 8.75 := by
  sorry

end hairdresser_cash_register_l3940_394044


namespace trig_identity_proof_l3940_394032

theorem trig_identity_proof : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (10 * π / 180) * Real.sin (70 * π / 180) = 1 / 2 := by
  sorry

end trig_identity_proof_l3940_394032


namespace triangle_cosine_B_l3940_394047

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_cosine_B (abc : Triangle) 
  (h1 : abc.b * Real.sin abc.B - abc.a * Real.sin abc.A = (1/2) * abc.a * Real.sin abc.C)
  (h2 : (1/2) * abc.a * abc.c * Real.sin abc.B = abc.a^2 * Real.sin abc.B) :
  Real.cos abc.B = 3/4 := by
  sorry

end triangle_cosine_B_l3940_394047


namespace swim_team_girls_count_l3940_394010

theorem swim_team_girls_count (total : ℕ) (ratio : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 96 → 
  ratio = 5 → 
  girls = ratio * boys → 
  total = girls + boys → 
  girls = 80 :=
by
  sorry

end swim_team_girls_count_l3940_394010


namespace wedge_product_formula_l3940_394057

/-- The wedge product of two 2D vectors -/
def wedge_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

/-- Theorem: The wedge product of two 2D vectors (a₁, a₂) and (b₁, b₂) is a₁b₂ - a₂b₁ -/
theorem wedge_product_formula (a b : ℝ × ℝ) : 
  wedge_product a b = a.1 * b.2 - a.2 * b.1 := by
  sorry

end wedge_product_formula_l3940_394057


namespace factory_workers_count_l3940_394014

/-- Represents the number of factory workers in company J -/
def factory_workers : ℕ := sorry

/-- Represents the number of office workers in company J -/
def office_workers : ℕ := 30

/-- Represents the total monthly payroll for factory workers in dollars -/
def factory_payroll : ℕ := 30000

/-- Represents the total monthly payroll for office workers in dollars -/
def office_payroll : ℕ := 75000

/-- Represents the difference in average monthly salary between office and factory workers in dollars -/
def salary_difference : ℕ := 500

theorem factory_workers_count :
  factory_workers = 15 ∧
  factory_workers * (office_payroll / office_workers - salary_difference) = factory_payroll :=
by sorry

end factory_workers_count_l3940_394014


namespace scatter_plot_correlation_l3940_394090

/-- Represents a scatter plot of two variables -/
structure ScatterPlot where
  bottomLeft : Bool
  topRight : Bool

/-- Defines positive correlation between two variables -/
def positivelyCorrelated (x y : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → x a < x b ∧ y a < y b

/-- Theorem: If a scatter plot goes from bottom left to top right, 
    the variables are positively correlated -/
theorem scatter_plot_correlation (plot : ScatterPlot) (x y : ℝ → ℝ) :
  plot.bottomLeft ∧ plot.topRight → positivelyCorrelated x y := by
  sorry


end scatter_plot_correlation_l3940_394090


namespace orange_cost_l3940_394007

/-- If 4 dozen oranges cost $24.00, then 6 dozen oranges at the same rate will cost $36.00. -/
theorem orange_cost (initial_cost : ℝ) (initial_quantity : ℕ) (target_quantity : ℕ) : 
  initial_cost = 24 ∧ initial_quantity = 4 ∧ target_quantity = 6 →
  (target_quantity : ℝ) * (initial_cost / initial_quantity) = 36 := by
sorry

end orange_cost_l3940_394007


namespace different_color_probability_l3940_394020

def shorts_colors : ℕ := 3
def jersey_colors : ℕ := 4

theorem different_color_probability :
  let total_combinations := shorts_colors * jersey_colors
  let different_color_combinations := total_combinations - shorts_colors
  (different_color_combinations : ℚ) / total_combinations = 3 / 4 := by
sorry

end different_color_probability_l3940_394020


namespace divisibility_by_nineteen_l3940_394029

theorem divisibility_by_nineteen (n : ℕ+) :
  ∃ k : ℤ, (5 ^ (2 * n.val - 1) : ℤ) + (3 ^ (n.val - 2) : ℤ) * (2 ^ (n.val - 1) : ℤ) = 19 * k :=
by sorry

end divisibility_by_nineteen_l3940_394029


namespace triple_of_negative_two_l3940_394055

theorem triple_of_negative_two : (3 : ℤ) * (-2 : ℤ) = -6 := by sorry

end triple_of_negative_two_l3940_394055


namespace bernardo_win_smallest_number_l3940_394088

theorem bernardo_win_smallest_number : ∃ M : ℕ, 
  (M ≤ 999) ∧ 
  (900 ≤ 72 * M) ∧ 
  (72 * M ≤ 999) ∧ 
  (∀ n : ℕ, n < M → (n ≤ 999 → 72 * n < 900 ∨ 999 < 72 * n)) ∧
  M = 13 := by
sorry

end bernardo_win_smallest_number_l3940_394088


namespace lens_savings_l3940_394031

theorem lens_savings (original_price : ℝ) (discount_rate : ℝ) (cheaper_price : ℝ) : 
  original_price = 300 ∧ 
  discount_rate = 0.20 ∧ 
  cheaper_price = 220 → 
  original_price * (1 - discount_rate) - cheaper_price = 20 := by
sorry

end lens_savings_l3940_394031


namespace sqrt_sum_rational_form_l3940_394048

theorem sqrt_sum_rational_form :
  ∃ (p q r : ℕ+), 
    (Real.sqrt 6 + (Real.sqrt 6)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (p * Real.sqrt 6 + q * Real.sqrt 8) / r) ∧
    (∀ (p' q' r' : ℕ+), 
      (Real.sqrt 6 + (Real.sqrt 6)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (p' * Real.sqrt 6 + q' * Real.sqrt 8) / r') →
      r ≤ r') ∧
    (p + q + r = 19) :=
by sorry

end sqrt_sum_rational_form_l3940_394048


namespace three_digit_integers_with_specific_remainders_l3940_394065

theorem three_digit_integers_with_specific_remainders :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ 
              n % 7 = 3 ∧ 
              n % 10 = 4 ∧ 
              n % 12 = 8) ∧
    S.card = 3 :=
by sorry

end three_digit_integers_with_specific_remainders_l3940_394065


namespace circle_C_equation_line_MN_equation_l3940_394045

-- Define the circle C
def circle_C (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + m = 0

-- Define the line that the circle is tangent to
def tangent_line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + Real.sqrt 3 - 2 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  x + 2*y = 0

-- Theorem for the equation of circle C
theorem circle_C_equation :
  ∃ m, ∀ x y, circle_C x y m ↔ (x+2)^2 + (y-1)^2 = 4 :=
sorry

-- Theorem for the equation of line MN
theorem line_MN_equation :
  ∃ M N : ℝ × ℝ,
    (∀ x y, circle_C x y 0 → (x, y) = M ∨ (x, y) = N) ∧
    (symmetry_line M.1 M.2 ↔ symmetry_line N.1 N.2) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 12) →
    ∃ c, ∀ x y, (2*x - y + c = 0 ∨ 2*x - y + (10 - c) = 0) ∧ c^2 = 30 :=
sorry

end circle_C_equation_line_MN_equation_l3940_394045


namespace minimum_employees_to_hire_l3940_394077

theorem minimum_employees_to_hire (S H : Finset Nat) 
  (h1 : S.card = 120)
  (h2 : H.card = 90)
  (h3 : (S ∩ H).card = 40) :
  (S ∪ H).card = 170 := by
sorry

end minimum_employees_to_hire_l3940_394077


namespace sunny_cake_candles_l3940_394033

/-- Given the initial number of cakes, number of cakes given away, and total candles used,
    calculate the number of candles on each remaining cake. -/
def candles_per_cake (initial_cakes : ℕ) (cakes_given_away : ℕ) (total_candles : ℕ) : ℕ :=
  total_candles / (initial_cakes - cakes_given_away)

/-- Prove that given the specific values in the problem, 
    the number of candles on each remaining cake is 6. -/
theorem sunny_cake_candles : 
  candles_per_cake 8 2 36 = 6 := by sorry

end sunny_cake_candles_l3940_394033


namespace P_iff_Q_l3940_394025

-- Define a triangle ABC with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the condition P
def condition_P (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2

-- Define the condition Q
def condition_Q (t : Triangle) : Prop :=
  ∃ x : ℝ, (x^2 + 2*t.a*x + t.b^2 = 0) ∧ (x^2 + 2*t.c*x - t.b^2 = 0)

-- State the theorem
theorem P_iff_Q (t : Triangle) : condition_P t ↔ condition_Q t := by
  sorry

end P_iff_Q_l3940_394025


namespace floor_difference_equals_five_l3940_394035

theorem floor_difference_equals_five (n : ℤ) : 
  (Int.floor (n^2 / 4 : ℚ) - Int.floor (n / 2 : ℚ)^2 = 5) → n = 11 := by
  sorry

end floor_difference_equals_five_l3940_394035


namespace quadratic_inequality_range_l3940_394043

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x - 4

-- Define the property that the solution set is ℝ
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x < 0

-- Theorem statement
theorem quadratic_inequality_range :
  ∀ a : ℝ, solution_set_is_reals a → -16 < a ∧ a ≤ 0 :=
sorry

end quadratic_inequality_range_l3940_394043


namespace system_solution_l3940_394024

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 - 3*y + z = -4) ∧ 
  (x - 3*y + z^2 = -10) ∧ 
  (3*x + y^2 - 3*z = 0) ∧ 
  (x = -2) ∧ (y = 3) ∧ (z = 1) := by
  sorry

end system_solution_l3940_394024


namespace hyperbola_intersection_theorem_l3940_394082

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 > 2

-- Main theorem
theorem hyperbola_intersection_theorem (k : ℝ) :
  (hyperbola_C 0 0) ∧  -- Center at origin
  (hyperbola_C 2 0) ∧  -- Right focus at (2,0)
  (hyperbola_C (Real.sqrt 3) 0) ∧  -- Right vertex at (√3,0)
  (intersects_at_two_points k) ∧
  (∀ A B : ℝ × ℝ, hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 → dot_product_condition A B) →
  (-1 < k ∧ k < -Real.sqrt 3 / 3) ∨ (Real.sqrt 3 / 3 < k ∧ k < 1) :=
sorry

end hyperbola_intersection_theorem_l3940_394082


namespace complex_square_equality_l3940_394026

theorem complex_square_equality : (((3 : ℂ) - I) / ((1 : ℂ) + I))^2 = -3 - 4*I := by sorry

end complex_square_equality_l3940_394026


namespace smallest_c_for_unique_solution_l3940_394037

/-- The system of equations -/
def system (x y c : ℝ) : Prop :=
  2 * (x + 7)^2 + (y - 4)^2 = c ∧ (x + 4)^2 + 2 * (y - 7)^2 = c

/-- The system has a unique solution -/
def has_unique_solution (c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, system p.1 p.2 c

/-- The smallest value of c for which the system has a unique solution is 6.0 -/
theorem smallest_c_for_unique_solution :
  (∀ c < 6, ¬ has_unique_solution c) ∧ has_unique_solution 6 :=
sorry

end smallest_c_for_unique_solution_l3940_394037


namespace intersecting_linear_function_k_range_l3940_394061

/-- A linear function passing through (2, 2) and intersecting y = -x + 3 within [0, 3] -/
structure IntersectingLinearFunction where
  k : ℝ
  b : ℝ
  passes_through_2_2 : 2 * k + b = 2
  intersects_in_domain : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ k * x + b = -x + 3

/-- The range of k values for the intersecting linear function -/
def k_range (f : IntersectingLinearFunction) : Prop :=
  (f.k ≤ -2 ∨ f.k ≥ -1/2) ∧ f.k ≠ 0

theorem intersecting_linear_function_k_range (f : IntersectingLinearFunction) :
  k_range f := by sorry

end intersecting_linear_function_k_range_l3940_394061


namespace three_squares_inequality_l3940_394018

/-- Given three equal squares arranged in a specific configuration, 
    this theorem proves that the length of the diagonal spanning two squares (AB) 
    is greater than the length of the diagonal spanning one square 
    and the side of another square (BC). -/
theorem three_squares_inequality (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  Real.sqrt (5 * x^2 + 4 * x * y + y^2) > Real.sqrt (5 * x^2 + 2 * x * y + y^2) := by
  sorry


end three_squares_inequality_l3940_394018


namespace segment_length_segment_length_is_eight_l3940_394053

theorem segment_length : ℝ → Prop :=
  fun length =>
    ∃ x₁ x₂ : ℝ,
      x₁ < x₂ ∧
      |x₁ - (27 : ℝ)^(1/3)| = 4 ∧
      |x₂ - (27 : ℝ)^(1/3)| = 4 ∧
      length = x₂ - x₁ ∧
      length = 8

theorem segment_length_is_eight : segment_length 8 := by
  sorry

end segment_length_segment_length_is_eight_l3940_394053


namespace pentagonal_country_routes_fifty_cities_routes_no_forty_six_routes_l3940_394086

/-- Definition of a pentagonal country -/
def PentagonalCountry (n : ℕ) := n > 0

/-- Number of air routes in a pentagonal country -/
def airRoutes (n : ℕ) : ℕ := (n * 5) / 2

theorem pentagonal_country_routes (n : ℕ) (h : PentagonalCountry n) : 
  airRoutes n = (n * 5) / 2 :=
sorry

theorem fifty_cities_routes : 
  airRoutes 50 = 125 :=
sorry

theorem no_forty_six_routes : 
  ¬ ∃ (n : ℕ), PentagonalCountry n ∧ airRoutes n = 46 :=
sorry

end pentagonal_country_routes_fifty_cities_routes_no_forty_six_routes_l3940_394086


namespace quadratic_roots_reciprocal_sum_l3940_394052

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → 
  x₂^2 - 2*x₂ - 5 = 0 → 
  x₁ ≠ 0 → 
  x₂ ≠ 0 → 
  1/x₁ + 1/x₂ = -2/5 := by
  sorry

end quadratic_roots_reciprocal_sum_l3940_394052


namespace f_properties_l3940_394042

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x, f x ≤ 2) ∧ 
  (f (7 * Real.pi / 12) = 0) := by
  sorry

end f_properties_l3940_394042


namespace circle_equation_is_correct_l3940_394092

/-- A circle C with center (1,2) that is tangent to the line x+2y=0 -/
def CircleC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 5}

/-- The line x+2y=0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2*p.2 = 0}

theorem circle_equation_is_correct :
  (∀ p ∈ CircleC, (p.1 - 1)^2 + (p.2 - 2)^2 = 5) ∧
  (∃ q ∈ CircleC ∩ TangentLine, q = (1, 2)) ∧
  (∀ r ∈ CircleC, r ≠ (1, 2) → r ∉ TangentLine) :=
sorry

end circle_equation_is_correct_l3940_394092


namespace sqrt_300_simplification_l3940_394085

theorem sqrt_300_simplification : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end sqrt_300_simplification_l3940_394085


namespace min_side_length_l3940_394084

/-- Given two triangles PQR and SQR sharing side QR, with PQ = 7 cm, PR = 15 cm, SR = 10 cm, and QS = 25 cm, the least possible integral length of QR is 16 cm. -/
theorem min_side_length (PQ PR SR QS : ℕ) (h1 : PQ = 7) (h2 : PR = 15) (h3 : SR = 10) (h4 : QS = 25) :
  (∃ QR : ℕ, QR > PR - PQ ∧ QR > QS - SR ∧ ∀ x : ℕ, (x > PR - PQ ∧ x > QS - SR) → x ≥ QR) →
  (∃ QR : ℕ, QR = 16 ∧ QR > PR - PQ ∧ QR > QS - SR ∧ ∀ x : ℕ, (x > PR - PQ ∧ x > QS - SR) → x ≥ QR) :=
by sorry

end min_side_length_l3940_394084


namespace no_cube_root_sum_prime_l3940_394027

theorem no_cube_root_sum_prime (x y p : ℕ+) (hp : Nat.Prime p.val) :
  (x.val : ℝ)^(1/3) + (y.val : ℝ)^(1/3) ≠ (p.val : ℝ)^(1/3) := by
  sorry

end no_cube_root_sum_prime_l3940_394027


namespace rectangle_cut_theorem_l3940_394068

theorem rectangle_cut_theorem (m : ℤ) (hm : m > 12) :
  ∃ (x y : ℕ+), (x.val : ℤ) * (y.val : ℤ) > m ∧ (x.val : ℤ) * ((y.val : ℤ) - 1) < m :=
sorry

end rectangle_cut_theorem_l3940_394068


namespace expected_heads_after_flips_l3940_394006

def num_coins : ℕ := 64
def max_flips : ℕ := 4

def prob_heads_single_flip : ℚ := 1 / 2

def prob_heads_multiple_flips (n : ℕ) : ℚ :=
  1 - (1 - prob_heads_single_flip) ^ n

theorem expected_heads_after_flips :
  (num_coins : ℚ) * prob_heads_multiple_flips max_flips = 60 := by
  sorry

end expected_heads_after_flips_l3940_394006


namespace gcd_840_1764_l3940_394060

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l3940_394060


namespace g_at_2_l3940_394023

-- Define the function g
def g (d : ℝ) (x : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

-- State the theorem
theorem g_at_2 (d : ℝ) : g d (-2) = 4 → g d 2 = -84 := by
  sorry

end g_at_2_l3940_394023


namespace walnut_trees_after_planting_l3940_394015

/-- The number of walnut trees in the park after planting -/
def total_walnut_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem stating the total number of walnut trees after planting -/
theorem walnut_trees_after_planting :
  total_walnut_trees 22 33 = 55 := by
  sorry

end walnut_trees_after_planting_l3940_394015


namespace oliver_initial_money_l3940_394011

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- Oliver's initial problem -/
theorem oliver_initial_money 
  (initial_quarters : ℕ) 
  (given_dollars : ℚ) 
  (given_quarters : ℕ) 
  (remaining_total : ℚ) :
  initial_quarters = 200 →
  given_dollars = 5 →
  given_quarters = 120 →
  remaining_total = 55 →
  (initial_quarters : ℚ) * quarter_value + 
    (given_dollars + (given_quarters : ℚ) * quarter_value + remaining_total) = 120 := by
  sorry

#eval quarter_value -- This line is to check if the definition is correct

end oliver_initial_money_l3940_394011


namespace opposite_sides_equal_implies_parallelogram_l3940_394063

/-- A quadrilateral is represented by four points in a 2D plane -/
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)

/-- Definition of a parallelogram -/
def is_parallelogram {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.A - q.D = q.B - q.C

/-- Theorem: If opposite sides of a quadrilateral are equal, it is a parallelogram -/
theorem opposite_sides_equal_implies_parallelogram 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (q : Quadrilateral V) 
  (h1 : q.A - q.B = q.D - q.C) 
  (h2 : q.A - q.D = q.B - q.C) : 
  is_parallelogram q :=
sorry

end opposite_sides_equal_implies_parallelogram_l3940_394063


namespace solve_percentage_equation_l3940_394089

theorem solve_percentage_equation (x : ℝ) : 0.60 * x = (1 / 3) * x + 110 → x = 412.5 := by
  sorry

end solve_percentage_equation_l3940_394089


namespace y1_greater_than_y2_l3940_394081

/-- A linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a linear function -/
def pointOnLinearFunction (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.m * p.x + f.b

/-- The theorem to be proved -/
theorem y1_greater_than_y2
  (f : LinearFunction)
  (A B C : Point)
  (h1 : f.m ≠ 0)
  (h2 : f.b = 4)
  (h3 : A.x = -2)
  (h4 : B.x = 1)
  (h5 : B.y = 3)
  (h6 : C.x = 3)
  (h7 : pointOnLinearFunction A f)
  (h8 : pointOnLinearFunction B f)
  (h9 : pointOnLinearFunction C f) :
  A.y > C.y :=
sorry

end y1_greater_than_y2_l3940_394081


namespace tens_digit_of_3_to_2013_l3940_394050

theorem tens_digit_of_3_to_2013 : ∃ n : ℕ, 3^2013 ≡ 40 + n [ZMOD 100] ∧ n < 10 := by
  sorry

end tens_digit_of_3_to_2013_l3940_394050


namespace jacket_final_price_l3940_394075

def original_price : ℝ := 240
def initial_discount : ℝ := 0.6
def holiday_discount : ℝ := 0.25

theorem jacket_final_price :
  let price_after_initial := original_price * (1 - initial_discount)
  let final_price := price_after_initial * (1 - holiday_discount)
  final_price = 72 := by sorry

end jacket_final_price_l3940_394075


namespace inequality_and_minimum_l3940_394099

theorem inequality_and_minimum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum_prod : x + y + z ≥ x * y * z) : 
  (x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z) ∧ 
  (∃ (u : ℝ), u = x / (y * z) + y / (z * x) + z / (x * y) ∧ 
              u ≥ Real.sqrt 3 ∧ 
              ∀ (v : ℝ), v = x / (y * z) + y / (z * x) + z / (x * y) → v ≥ u) := by
  sorry

end inequality_and_minimum_l3940_394099


namespace rightmost_four_digits_of_7_to_2023_l3940_394073

theorem rightmost_four_digits_of_7_to_2023 :
  7^2023 ≡ 1359 [ZMOD 10000] := by
  sorry

end rightmost_four_digits_of_7_to_2023_l3940_394073


namespace scheme2_more_cost_effective_l3940_394001

/-- The cost of a teapot in yuan -/
def teapot_cost : ℝ := 25

/-- The cost of a tea cup in yuan -/
def teacup_cost : ℝ := 5

/-- The number of teapots the customer needs to buy -/
def num_teapots : ℕ := 4

/-- The discount percentage for Scheme 2 -/
def discount_percentage : ℝ := 0.94

/-- The cost calculation for Scheme 1 -/
def scheme1_cost (x : ℝ) : ℝ := 5 * x + 80

/-- The cost calculation for Scheme 2 -/
def scheme2_cost (x : ℝ) : ℝ := (teapot_cost * num_teapots + teacup_cost * x) * discount_percentage

/-- The number of tea cups for which we want to compare the schemes -/
def x : ℝ := 47

theorem scheme2_more_cost_effective : scheme2_cost x < scheme1_cost x := by
  sorry

end scheme2_more_cost_effective_l3940_394001


namespace largest_remainder_2015_l3940_394069

theorem largest_remainder_2015 : 
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 1000 → (2015 % d) ≤ 671 ∧ ∃ d₀ : ℕ, 1 ≤ d₀ ∧ d₀ ≤ 1000 ∧ 2015 % d₀ = 671 :=
by sorry

end largest_remainder_2015_l3940_394069


namespace fred_bought_two_tickets_l3940_394080

/-- The number of tickets Fred bought -/
def num_tickets : ℕ := 2

/-- The price of each ticket in cents -/
def ticket_price : ℕ := 592

/-- The cost of borrowing a movie in cents -/
def movie_rental : ℕ := 679

/-- The amount Fred paid in cents -/
def amount_paid : ℕ := 2000

/-- The change Fred received in cents -/
def change_received : ℕ := 137

/-- Theorem stating that Fred bought 2 tickets given the conditions -/
theorem fred_bought_two_tickets :
  num_tickets * ticket_price + movie_rental = amount_paid - change_received :=
by sorry

end fred_bought_two_tickets_l3940_394080


namespace unique_solution_l3940_394041

/-- The function f(x) = x^2 + 4x + 3 -/
def f (x : ℤ) : ℤ := x^2 + 4*x + 3

/-- The function g(x) = x^2 + 2x - 1 -/
def g (x : ℤ) : ℤ := x^2 + 2*x - 1

/-- Theorem stating that x = -2 is the only integer solution to f(g(f(x))) = g(f(g(x))) -/
theorem unique_solution :
  ∃! x : ℤ, f (g (f x)) = g (f (g x)) ∧ x = -2 := by
  sorry

end unique_solution_l3940_394041


namespace largest_common_divisor_l3940_394002

theorem largest_common_divisor : 
  let a := 924
  let b := 1386
  let c := 462
  Nat.gcd a (Nat.gcd b c) = 462 := by
sorry

end largest_common_divisor_l3940_394002


namespace complex_equation_solution_l3940_394016

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (x : ℂ), (2 : ℂ) + 3 * i * x = (4 : ℂ) - 5 * i * x ∧ x = i / 4 := by
  sorry

end complex_equation_solution_l3940_394016


namespace calculation_result_l3940_394072

theorem calculation_result : (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 := by
  sorry

end calculation_result_l3940_394072


namespace diane_age_is_16_l3940_394013

/-- Represents the current ages of Diane, Alex, and Allison -/
structure Ages where
  diane : ℕ
  alex : ℕ
  allison : ℕ

/-- Conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alex + ages.allison = 47 ∧
  ages.alex + (30 - ages.diane) = 60 ∧
  ages.allison + (30 - ages.diane) = 15

/-- Theorem stating that Diane's current age is 16 -/
theorem diane_age_is_16 :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.diane = 16 :=
sorry

end diane_age_is_16_l3940_394013


namespace distance_between_Disney_and_London_l3940_394039

/-- The distance between lake Disney and lake London --/
def distance_Disney_London : ℝ := 60

/-- The number of migrating birds --/
def num_birds : ℕ := 20

/-- The distance between lake Jim and lake Disney --/
def distance_Jim_Disney : ℝ := 50

/-- The combined distance traveled by all birds in two seasons --/
def total_distance : ℝ := 2200

theorem distance_between_Disney_and_London :
  distance_Disney_London = 
    (total_distance - num_birds * distance_Jim_Disney) / num_birds :=
by sorry

end distance_between_Disney_and_London_l3940_394039


namespace second_plot_germination_rate_l3940_394078

/-- Calculates the germination rate of the second plot given the number of seeds in each plot,
    the germination rate of the first plot, and the overall germination rate. -/
theorem second_plot_germination_rate 
  (seeds_first_plot : ℕ)
  (seeds_second_plot : ℕ)
  (germination_rate_first_plot : ℚ)
  (overall_germination_rate : ℚ)
  (h1 : seeds_first_plot = 300)
  (h2 : seeds_second_plot = 200)
  (h3 : germination_rate_first_plot = 25 / 100)
  (h4 : overall_germination_rate = 27 / 100)
  : (overall_germination_rate * (seeds_first_plot + seeds_second_plot) - 
     germination_rate_first_plot * seeds_first_plot) / seeds_second_plot = 30 / 100 := by
  sorry

end second_plot_germination_rate_l3940_394078
