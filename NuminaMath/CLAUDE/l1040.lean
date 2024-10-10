import Mathlib

namespace no_perfect_squares_sum_l1040_104058

theorem no_perfect_squares_sum (x y : ℕ) : 
  ¬(∃ (a b : ℕ), x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end no_perfect_squares_sum_l1040_104058


namespace exactly_seven_numbers_satisfy_conditions_l1040_104045

/-- A function that replaces a digit at position k with zero in a natural number n -/
def replace_digit_with_zero (n : ℕ) (k : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with zero -/
def ends_with_zero (n : ℕ) : Prop := sorry

/-- A function that counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ := sorry

/-- The main theorem stating that there are exactly 7 numbers satisfying the conditions -/
theorem exactly_seven_numbers_satisfy_conditions : 
  ∃! (s : Finset ℕ), 
    (s.card = 7) ∧ 
    (∀ n ∈ s, 
      ¬ends_with_zero n ∧ 
      ∃ k, k < digit_count n ∧ 
        9 * replace_digit_with_zero n k = n) :=
by sorry

end exactly_seven_numbers_satisfy_conditions_l1040_104045


namespace bus_passenger_count_l1040_104048

def bus_passengers (initial_passengers : ℕ) (new_passengers : ℕ) : ℕ :=
  initial_passengers + new_passengers

theorem bus_passenger_count : 
  bus_passengers 4 13 = 17 := by sorry

end bus_passenger_count_l1040_104048


namespace three_roots_implies_m_equals_two_l1040_104010

/-- The function f(x) = x^2 - 2|x| + 2 - m -/
def f (x m : ℝ) : ℝ := x^2 - 2 * abs x + 2 - m

/-- The number of roots of f(x) for a given m -/
def num_roots (m : ℝ) : ℕ := sorry

theorem three_roots_implies_m_equals_two :
  ∀ m : ℝ, num_roots m = 3 → m = 2 := by sorry

end three_roots_implies_m_equals_two_l1040_104010


namespace cube_volume_ratio_l1040_104052

theorem cube_volume_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := s₂ * Real.sqrt 3
  (s₁^3) / (s₂^3) = 3 * Real.sqrt 3 := by
  sorry

end cube_volume_ratio_l1040_104052


namespace ones_digit_of_first_prime_in_sequence_l1040_104003

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Definition of an arithmetic sequence of five primes -/
def is_arithmetic_prime_sequence (p q r s t : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ Nat.Prime t ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 ∧ t = s + 10

theorem ones_digit_of_first_prime_in_sequence (p q r s t : ℕ) :
  is_arithmetic_prime_sequence p q r s t → p > 5 → ones_digit p = 1 :=
by sorry

end ones_digit_of_first_prime_in_sequence_l1040_104003


namespace min_value_of_function_equality_condition_l1040_104046

theorem min_value_of_function (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 2) : 
  ∃ x, x > 2 ∧ x + 4 / (x - 2) = 6 := by
  sorry

end min_value_of_function_equality_condition_l1040_104046


namespace dinner_tip_calculation_l1040_104054

/-- Calculates the individual tip amount for a group dinner -/
theorem dinner_tip_calculation (julie_order : ℚ) (letitia_order : ℚ) (anton_order : ℚ) 
  (tip_percentage : ℚ) (num_people : ℕ) : 
  julie_order = 10 ∧ letitia_order = 20 ∧ anton_order = 30 ∧ 
  tip_percentage = 1/5 ∧ num_people = 3 →
  (julie_order + letitia_order + anton_order) * tip_percentage / num_people = 4 := by
  sorry

#check dinner_tip_calculation

end dinner_tip_calculation_l1040_104054


namespace cone_height_l1040_104088

theorem cone_height (s : ℝ) (a : ℝ) (h : s = 13 ∧ a = 65 * Real.pi) :
  Real.sqrt (s^2 - (a / (s * Real.pi))^2) = 12 := by
  sorry

end cone_height_l1040_104088


namespace ellipse_hyperbola_same_foci_l1040_104028

theorem ellipse_hyperbola_same_foci (k : ℝ) : k > 0 →
  (∀ x y : ℝ, x^2/9 + y^2/k^2 = 1 ↔ x^2/k - y^2/3 = 1) →
  k = 2 :=
by sorry

end ellipse_hyperbola_same_foci_l1040_104028


namespace sector_area_l1040_104026

/-- The area of a circular sector with central angle π/3 and radius 2 is 2π/3 -/
theorem sector_area (α : Real) (r : Real) (h1 : α = π / 3) (h2 : r = 2) :
  (1 / 2) * α * r^2 = (2 * π) / 3 := by
  sorry

end sector_area_l1040_104026


namespace units_digit_of_power_difference_l1040_104070

theorem units_digit_of_power_difference : ∃ n : ℕ, (25^2010 - 3^2012) % 10 = 4 := by
  sorry

end units_digit_of_power_difference_l1040_104070


namespace front_wheel_perimeter_front_wheel_perimeter_is_30_l1040_104096

/-- The perimeter of the front wheel of a bicycle, given the perimeter of the back wheel
    and the number of revolutions each wheel makes to cover the same distance. -/
theorem front_wheel_perimeter (back_wheel_perimeter : ℝ) 
    (front_wheel_revolutions : ℝ) (back_wheel_revolutions : ℝ) : ℝ :=
  let front_wheel_perimeter := (back_wheel_perimeter * back_wheel_revolutions) / front_wheel_revolutions
  have back_wheel_perimeter_eq : back_wheel_perimeter = 20 := by sorry
  have front_wheel_revolutions_eq : front_wheel_revolutions = 240 := by sorry
  have back_wheel_revolutions_eq : back_wheel_revolutions = 360 := by sorry
  have equal_distance : front_wheel_perimeter * front_wheel_revolutions = 
                        back_wheel_perimeter * back_wheel_revolutions := by sorry
  30

theorem front_wheel_perimeter_is_30 : front_wheel_perimeter 20 240 360 = 30 := by sorry

end front_wheel_perimeter_front_wheel_perimeter_is_30_l1040_104096


namespace pepperoni_coverage_is_four_ninths_l1040_104051

/-- Represents a circular pizza with pepperoni toppings -/
structure PizzaWithPepperoni where
  pizzaDiameter : ℝ
  pepperoniAcrossDiameter : ℕ
  totalPepperoni : ℕ

/-- Calculates the fraction of the pizza covered by pepperoni -/
def pepperoniCoverage (pizza : PizzaWithPepperoni) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the pizza covered by pepperoni is 4/9 -/
theorem pepperoni_coverage_is_four_ninths (pizza : PizzaWithPepperoni) 
  (h1 : pizza.pizzaDiameter = 18)
  (h2 : pizza.pepperoniAcrossDiameter = 9)
  (h3 : pizza.totalPepperoni = 36) : 
  pepperoniCoverage pizza = 4/9 := by
  sorry

end pepperoni_coverage_is_four_ninths_l1040_104051


namespace william_hot_dogs_l1040_104059

/-- The number of hot dogs William sold during the first three innings -/
def first_innings_sales : ℕ := 19

/-- The number of hot dogs William sold during the next three innings -/
def next_innings_sales : ℕ := 27

/-- The number of hot dogs William had left to sell -/
def remaining_hot_dogs : ℕ := 45

/-- The total number of hot dogs William had at first -/
def total_hot_dogs : ℕ := first_innings_sales + next_innings_sales + remaining_hot_dogs

theorem william_hot_dogs : total_hot_dogs = 91 := by sorry

end william_hot_dogs_l1040_104059


namespace existence_of_even_and_odd_composite_functions_l1040_104080

theorem existence_of_even_and_odd_composite_functions :
  ∃ (p q : ℝ → ℝ),
    (∀ x, p (-x) = p x) ∧
    (∀ x, p (q (-x)) = -(p (q x))) ∧
    (∃ x, p (q x) ≠ 0) :=
by sorry

end existence_of_even_and_odd_composite_functions_l1040_104080


namespace distance_to_right_focus_is_18_l1040_104078

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Axiom: P is on the left branch of the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the distance from P to the left focus
def distance_to_left_focus : ℝ := 10

-- Define the distance from P to the right focus
def distance_to_right_focus : ℝ := sorry

-- Theorem to prove
theorem distance_to_right_focus_is_18 :
  distance_to_right_focus = 18 :=
sorry

end distance_to_right_focus_is_18_l1040_104078


namespace simplify_and_rationalize_l1040_104008

theorem simplify_and_rationalize :
  ∃ (x : ℝ), (x = (Real.sqrt 2 / Real.sqrt 5) * (Real.sqrt 3 / Real.sqrt 7) * (Real.rpow 4 (1/3) / Real.sqrt 6)) ∧
             (x = (Real.rpow 4 (1/3) * Real.sqrt 35) / 35) := by
  sorry

end simplify_and_rationalize_l1040_104008


namespace sqrt_meaningful_range_l1040_104036

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_meaningful_range_l1040_104036


namespace polynomial_identity_sum_l1040_104025

theorem polynomial_identity_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ + a₂ + a₀ = 41 := by
sorry

end polynomial_identity_sum_l1040_104025


namespace shirt_price_l1040_104095

/-- Given a shirt and sweater with a total cost of $80.34, where the shirt costs $7.43 less than the sweater, the price of the shirt is $36.455. -/
theorem shirt_price (total_cost sweater_price shirt_price : ℝ) : 
  total_cost = 80.34 →
  sweater_price = shirt_price + 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by sorry

end shirt_price_l1040_104095


namespace tangent_parallel_to_xy_l1040_104067

-- Define the function f(x) = x^2 - x
def f (x : ℝ) : ℝ := x^2 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem tangent_parallel_to_xy (P : ℝ × ℝ) :
  (P.1 = 1 ∧ P.2 = 0) ↔
  (f' P.1 = 1 ∧ P.2 = f P.1) := by
  sorry

#check tangent_parallel_to_xy

end tangent_parallel_to_xy_l1040_104067


namespace carries_payment_l1040_104073

def clothes_shopping (shirt_quantity : ℕ) (pants_quantity : ℕ) (jacket_quantity : ℕ)
                     (shirt_price : ℕ) (pants_price : ℕ) (jacket_price : ℕ) : ℕ :=
  let total_cost := shirt_quantity * shirt_price + pants_quantity * pants_price + jacket_quantity * jacket_price
  total_cost / 2

theorem carries_payment :
  clothes_shopping 4 2 2 8 18 60 = 94 :=
by
  sorry

end carries_payment_l1040_104073


namespace total_cost_one_large_three_small_l1040_104097

/-- The cost of a large puzzle, in dollars -/
def large_puzzle_cost : ℕ := 15

/-- The cost of a small puzzle and a large puzzle together, in dollars -/
def combined_cost : ℕ := 23

/-- The cost of a small puzzle, in dollars -/
def small_puzzle_cost : ℕ := combined_cost - large_puzzle_cost

theorem total_cost_one_large_three_small :
  large_puzzle_cost + 3 * small_puzzle_cost = 39 := by
  sorry

end total_cost_one_large_three_small_l1040_104097


namespace arithmetic_sequence_sum_l1040_104063

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_property : ∀ n, S n = n * (a 0 + a (n-1)) / 2

/-- Theorem: For an arithmetic sequence with S_3 = 9 and S_6 = 36, S_9 = 81 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 9) (h6 : seq.S 6 = 36) : seq.S 9 = 81 := by
  sorry

end arithmetic_sequence_sum_l1040_104063


namespace monotone_increasing_iff_a_in_range_l1040_104081

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1/2 then a^x else (2*a - 1)*x

theorem monotone_increasing_iff_a_in_range (a : ℝ) :
  Monotone (f a) ↔ a ∈ Set.Ici ((2 + Real.sqrt 3) / 2) :=
sorry

end monotone_increasing_iff_a_in_range_l1040_104081


namespace no_seven_edge_polyhedron_exists_polyhedron_with_2n_and_2n_plus_3_edges_l1040_104090

-- Define a convex polyhedron
structure ConvexPolyhedron where
  edges : ℕ
  is_convex : Bool

-- Theorem 1: A convex polyhedron cannot have exactly 7 edges
theorem no_seven_edge_polyhedron :
  ¬∃ (p : ConvexPolyhedron), p.edges = 7 ∧ p.is_convex = true :=
sorry

-- Theorem 2: For any integer n ≥ 3, there exists a convex polyhedron 
-- with 2n edges and another with 2n + 3 edges
theorem exists_polyhedron_with_2n_and_2n_plus_3_edges (n : ℕ) (h : n ≥ 3) :
  (∃ (p : ConvexPolyhedron), p.edges = 2 * n ∧ p.is_convex = true) ∧
  (∃ (q : ConvexPolyhedron), q.edges = 2 * n + 3 ∧ q.is_convex = true) :=
sorry

end no_seven_edge_polyhedron_exists_polyhedron_with_2n_and_2n_plus_3_edges_l1040_104090


namespace worksheets_graded_l1040_104086

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 16 →
  problems_per_worksheet = 4 →
  problems_left = 32 →
  total_worksheets * problems_per_worksheet - problems_left = 8 * problems_per_worksheet :=
by sorry

end worksheets_graded_l1040_104086


namespace desk_rearrangement_combinations_l1040_104014

/-- The number of choices for each day of the week --/
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 4
def friday_choices : ℕ := 1

/-- The total number of combinations --/
def total_combinations : ℕ := 
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

/-- Theorem stating that the total number of combinations is 60 --/
theorem desk_rearrangement_combinations : total_combinations = 60 := by
  sorry

end desk_rearrangement_combinations_l1040_104014


namespace geometric_sequence_ratio_l1040_104041

/-- A geometric sequence with the given property -/
structure GeometricSequence where
  a : ℕ → ℝ
  has_identical_roots : ∃ x : ℝ, a 1 * x^2 - a 3 * x + a 2 = 0 ∧ 
    ∀ y : ℝ, a 1 * y^2 - a 3 * y + a 2 = 0 → y = x

/-- Sum of the first n terms of a geometric sequence -/
def sum (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  sum seq 9 / sum seq 3 = 21 := by sorry

end geometric_sequence_ratio_l1040_104041


namespace always_true_inequality_l1040_104062

theorem always_true_inequality (a b x : ℝ) (h : a > b) : a * (2 : ℝ)^x > b * (2 : ℝ)^x := by
  sorry

end always_true_inequality_l1040_104062


namespace store_charge_with_interest_l1040_104061

/-- Proves that a principal amount of $35 with 7% simple annual interest results in a total debt of $37.45 after one year -/
theorem store_charge_with_interest (P : ℝ) (interest_rate : ℝ) (total_debt : ℝ) : 
  interest_rate = 0.07 →
  total_debt = 37.45 →
  P * (1 + interest_rate) = total_debt →
  P = 35 := by
sorry

end store_charge_with_interest_l1040_104061


namespace fifty_billion_scientific_notation_l1040_104031

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem fifty_billion_scientific_notation :
  toScientificNotation 50000000000 = ScientificNotation.mk 5 10 (by sorry) :=
sorry

end fifty_billion_scientific_notation_l1040_104031


namespace rainwater_farm_chickens_l1040_104044

/-- Represents the number of animals on Mr. Rainwater's farm -/
structure FarmAnimals where
  cows : Nat
  goats : Nat
  chickens : Nat
  ducks : Nat

/-- Defines the conditions for Mr. Rainwater's farm -/
def valid_farm (f : FarmAnimals) : Prop :=
  f.cows = 9 ∧
  f.goats = 4 * f.cows ∧
  f.goats = 2 * f.chickens ∧
  f.ducks = (3 * f.chickens) / 2 ∧
  (f.ducks - 2 * f.chickens) % 3 = 0 ∧
  f.goats + f.chickens + f.ducks ≤ 100

theorem rainwater_farm_chickens :
  ∀ f : FarmAnimals, valid_farm f → f.chickens = 18 :=
sorry

end rainwater_farm_chickens_l1040_104044


namespace steel_experiment_golden_ratio_l1040_104089

/-- The 0.618 method calculation for a given range -/
def golden_ratio_method (lower_bound upper_bound : ℝ) : ℝ :=
  lower_bound + (upper_bound - lower_bound) * 0.618

/-- Theorem: The 0.618 method for the given steel experiment -/
theorem steel_experiment_golden_ratio :
  let lower_bound : ℝ := 500
  let upper_bound : ℝ := 1000
  golden_ratio_method lower_bound upper_bound = 809 := by
  sorry

end steel_experiment_golden_ratio_l1040_104089


namespace shopkeeper_payment_l1040_104013

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_successive_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem shopkeeper_payment (porcelain_price crystal_price : ℝ)
  (porcelain_discounts crystal_discounts : List ℝ) :
  porcelain_price = 8500 →
  crystal_price = 1500 →
  porcelain_discounts = [0.25, 0.15, 0.05] →
  crystal_discounts = [0.30, 0.10, 0.05] →
  (apply_successive_discounts porcelain_price porcelain_discounts +
   apply_successive_discounts crystal_price crystal_discounts) = 6045.56 := by
  sorry

end shopkeeper_payment_l1040_104013


namespace exchange_theorem_l1040_104038

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def num_exchanges : ℕ := 4

/-- Xiao Zhang's initial number of pencils -/
def zhang_initial_pencils : ℕ := 200

/-- Xiao Li's initial number of fountain pens -/
def li_initial_pens : ℕ := 20

/-- Number of pencils Xiao Zhang gives in each exchange -/
def pencils_per_exchange : ℕ := 6

/-- Number of fountain pens Xiao Li gives in each exchange -/
def pens_per_exchange : ℕ := 1

/-- Xiao Zhang's final number of pencils -/
def zhang_final_pencils : ℕ := zhang_initial_pencils - num_exchanges * pencils_per_exchange

/-- Xiao Li's final number of fountain pens -/
def li_final_pens : ℕ := li_initial_pens - num_exchanges * pens_per_exchange

theorem exchange_theorem :
  zhang_final_pencils = 11 * li_final_pens :=
by sorry

end exchange_theorem_l1040_104038


namespace board_problem_l1040_104039

def board_operation (a b c : ℤ) : ℤ × ℤ × ℤ :=
  (a, b, a + b - c)

def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

def can_reach_sequence (start_a start_b start_c target_a target_b target_c : ℤ) : Prop :=
  ∃ (n : ℕ), ∃ (seq : ℕ → ℤ × ℤ × ℤ),
    seq 0 = (start_a, start_b, start_c) ∧
    (∀ i, i < n → 
      let (a, b, c) := seq i
      seq (i + 1) = board_operation a b c ∨ 
      seq (i + 1) = board_operation a c b ∨ 
      seq (i + 1) = board_operation b c a) ∧
    seq n = (target_a, target_b, target_c)

theorem board_problem :
  can_reach_sequence 3 9 15 2013 2019 2025 ∧
  is_arithmetic_sequence 2013 2019 2025 :=
sorry

end board_problem_l1040_104039


namespace income_data_correction_l1040_104084

theorem income_data_correction (T : ℝ) : 
  let num_families : ℕ := 1200
  let largest_correct_income : ℝ := 102000
  let largest_incorrect_income : ℝ := 1020000
  let processing_fee : ℝ := 500
  let corrected_mean := (T + (largest_correct_income - processing_fee)) / num_families
  let incorrect_mean := (T + (largest_incorrect_income - processing_fee)) / num_families
  incorrect_mean - corrected_mean = 765 := by
sorry

end income_data_correction_l1040_104084


namespace high_season_packs_correct_l1040_104034

/-- Represents the number of tuna packs sold per hour during the high season -/
def high_season_packs : ℕ := 6

/-- Represents the number of tuna packs sold per hour during the low season -/
def low_season_packs : ℕ := 4

/-- Represents the price of each tuna pack in dollars -/
def price_per_pack : ℕ := 60

/-- Represents the number of hours fish are sold per day -/
def hours_per_day : ℕ := 15

/-- Represents the additional revenue in dollars during the high season compared to the low season -/
def additional_revenue : ℕ := 1800

theorem high_season_packs_correct :
  high_season_packs * hours_per_day * price_per_pack =
  low_season_packs * hours_per_day * price_per_pack + additional_revenue :=
by sorry

end high_season_packs_correct_l1040_104034


namespace floor_inequality_l1040_104083

theorem floor_inequality (α β : ℝ) : 
  ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ := by sorry

end floor_inequality_l1040_104083


namespace arithmetic_contains_geometric_l1040_104043

/-- An arithmetic sequence of natural numbers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + (n - 1) * d

theorem arithmetic_contains_geometric (a d : ℕ) (h : d > 0) :
  ∃ (r : ℚ) (f : ℕ → ℕ), 
    (∀ n, f n < f (n + 1)) ∧ 
    (∀ n, ArithmeticSequence a d (f n) * r = ArithmeticSequence a d (f (n + 1))) := by
  sorry

end arithmetic_contains_geometric_l1040_104043


namespace equation_D_is_quadratic_l1040_104068

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation (k²+1)x²-2x+1=0 -/
def equation_D (k : ℝ) (x : ℝ) : Prop :=
  (k^2 + 1) * x^2 - 2*x + 1 = 0

theorem equation_D_is_quadratic (k : ℝ) :
  is_quadratic_in_x (k^2 + 1) (-2) 1 := by sorry

end equation_D_is_quadratic_l1040_104068


namespace ratio_closest_to_five_l1040_104066

theorem ratio_closest_to_five : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |(10^2000 + 10^2002) / (10^2001 + 10^2001) - 5| < ε :=
sorry

end ratio_closest_to_five_l1040_104066


namespace quadratic_inequality_solution_l1040_104030

/-- Given a quadratic inequality a*x^2 + b*x + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that the product ab equals -6. -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, a*x^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a * b = -6 :=
sorry

end quadratic_inequality_solution_l1040_104030


namespace function_difference_l1040_104040

theorem function_difference (m : ℚ) : 
  let f (x : ℚ) := 4 * x^2 - 3 * x + 5
  let g (x : ℚ) := x^2 - m * x - 8
  (f 5 - g 5 = 20) → m = -53/5 := by
sorry

end function_difference_l1040_104040


namespace opposite_numbers_linear_equation_l1040_104037

theorem opposite_numbers_linear_equation :
  ∀ x y : ℝ,
  (2 * x - 3 * y = 10) →
  (y = -x) →
  x = 2 := by
sorry

end opposite_numbers_linear_equation_l1040_104037


namespace average_equation_solution_l1040_104033

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((3*x + 4) + (7*x - 5) + (4*x + 9)) = 5*x - 3 → x = 17 := by
sorry

end average_equation_solution_l1040_104033


namespace fraction_evaluation_l1040_104077

theorem fraction_evaluation : (1 - 2/5) / (1 - 1/4) = 4/5 := by sorry

end fraction_evaluation_l1040_104077


namespace arcsin_plus_arcsin_2x_eq_arccos_l1040_104072

theorem arcsin_plus_arcsin_2x_eq_arccos (x : ℝ) : 
  (Real.arcsin x + Real.arcsin (2*x) = Real.arccos x) ↔ 
  (x = 0 ∨ x = 2/Real.sqrt 5 ∨ x = -2/Real.sqrt 5) := by
sorry

end arcsin_plus_arcsin_2x_eq_arccos_l1040_104072


namespace product_sum_theorem_l1040_104093

theorem product_sum_theorem (a b c : ℤ) : 
  a * b * c = -13 → (a + b + c = -11 ∨ a + b + c = 13) := by
  sorry

end product_sum_theorem_l1040_104093


namespace hallway_area_in_sq_yards_l1040_104001

-- Define the dimensions of the hallway
def hallway_length : ℝ := 15
def hallway_width : ℝ := 4

-- Define the conversion factor from square feet to square yards
def sq_feet_per_sq_yard : ℝ := 9

-- Theorem statement
theorem hallway_area_in_sq_yards :
  (hallway_length * hallway_width) / sq_feet_per_sq_yard = 20 / 3 := by
  sorry

end hallway_area_in_sq_yards_l1040_104001


namespace least_homeowners_l1040_104050

theorem least_homeowners (total_members : ℕ) (men_percentage : ℚ) (women_percentage : ℚ)
  (h_total : total_members = 150)
  (h_men_percentage : men_percentage = 1/10)
  (h_women_percentage : women_percentage = 1/5) :
  ∃ (men women : ℕ),
    men + women = total_members ∧
    ∃ (men_homeowners women_homeowners : ℕ),
      men_homeowners = ⌈men_percentage * men⌉ ∧
      women_homeowners = ⌈women_percentage * women⌉ ∧
      men_homeowners + women_homeowners = 16 ∧
      ∀ (other_men other_women : ℕ),
        other_men + other_women = total_members →
        ∃ (other_men_homeowners other_women_homeowners : ℕ),
          other_men_homeowners = ⌈men_percentage * other_men⌉ ∧
          other_women_homeowners = ⌈women_percentage * other_women⌉ →
          other_men_homeowners + other_women_homeowners ≥ 16 :=
by
  sorry

end least_homeowners_l1040_104050


namespace min_books_borrowed_l1040_104057

theorem min_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat)
  (avg_books : Rat) (max_books : Nat) :
  total_students = 32 →
  zero_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books = 2 →
  max_books = 11 →
  ∃ (min_books : Nat),
    min_books = 4 ∧
    min_books ≤ max_books ∧
    (total_students - zero_books - one_book - two_books) * min_books +
    one_book * 1 + two_books * 2 =
    (total_students : Rat) * avg_books := by
  sorry

end min_books_borrowed_l1040_104057


namespace plumbing_equal_charge_time_l1040_104016

def pauls_visit_fee : ℚ := 55
def pauls_hourly_rate : ℚ := 35
def reliable_visit_fee : ℚ := 75
def reliable_hourly_rate : ℚ := 30

theorem plumbing_equal_charge_time : 
  ∃ h : ℚ, h > 0 ∧ (pauls_visit_fee + pauls_hourly_rate * h = reliable_visit_fee + reliable_hourly_rate * h) ∧ h = 4 := by
  sorry

end plumbing_equal_charge_time_l1040_104016


namespace floor_of_5_7_l1040_104094

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by
  sorry

end floor_of_5_7_l1040_104094


namespace min_value_a_min_value_a_tight_l1040_104000

theorem min_value_a (a : ℝ) : 
  (∀ x > 0, x^2 + a*x + 1 ≥ 0) → a ≥ -2 :=
by sorry

theorem min_value_a_tight : 
  ∃ a : ℝ, (∀ x > 0, x^2 + a*x + 1 ≥ 0) ∧ a = -2 :=
by sorry

end min_value_a_min_value_a_tight_l1040_104000


namespace cone_volume_approximation_l1040_104071

theorem cone_volume_approximation (L h : ℝ) (h1 : L > 0) (h2 : h > 0) :
  (7 / 264 : ℝ) * L^2 * h = (1 / 3 : ℝ) * ((22 / 7 : ℝ) / 4) * L^2 * h := by
  sorry

end cone_volume_approximation_l1040_104071


namespace point_on_line_with_equal_distances_l1040_104055

theorem point_on_line_with_equal_distances (P : ℝ × ℝ) :
  P.1 + 3 * P.2 = 0 →
  (P.1^2 + P.2^2).sqrt = |P.1 + 3 * P.2 - 2| / (1^2 + 3^2).sqrt →
  (P = (3/5, -1/5) ∨ P = (-3/5, 1/5)) :=
by sorry

end point_on_line_with_equal_distances_l1040_104055


namespace triangle_radius_inequality_l1040_104018

/-- For any triangle with circumradius R, inradius r, and semiperimeter p,
    the inequality 27Rr ≤ 2p² ≤ 27R²/2 holds. -/
theorem triangle_radius_inequality (R r p : ℝ) (h_positive : R > 0 ∧ r > 0 ∧ p > 0) 
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    p = (a + b + c) / 2 ∧ 
    R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)) ∧
    r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 := by
  sorry

end triangle_radius_inequality_l1040_104018


namespace unique_solution_to_equation_l1040_104012

theorem unique_solution_to_equation : 
  ∃! (x y : ℕ+), (x.val : ℝ)^6 * (y.val : ℝ)^6 - 19 * (x.val : ℝ)^3 * (y.val : ℝ)^3 + 18 = 0 := by
  sorry

end unique_solution_to_equation_l1040_104012


namespace boat_journey_time_l1040_104021

/-- The boat's journey time given specific conditions -/
theorem boat_journey_time 
  (stream_velocity : ℝ) 
  (boat_speed_still : ℝ) 
  (distance_AB : ℝ) 
  (h1 : stream_velocity = 4)
  (h2 : boat_speed_still = 14)
  (h3 : distance_AB = 180) :
  let downstream_speed := boat_speed_still + stream_velocity
  let upstream_speed := boat_speed_still - stream_velocity
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := (distance_AB / 2) / upstream_speed
  time_downstream + time_upstream = 19 := by
sorry

end boat_journey_time_l1040_104021


namespace base_n_problem_l1040_104027

theorem base_n_problem (n d : ℕ) : 
  n > 0 → 
  d < 10 → 
  3 * n^2 + 2 * n + d = 263 → 
  3 * n^2 + 2 * n + 4 = 396 + 7 * d → 
  n + d = 11 := by sorry

end base_n_problem_l1040_104027


namespace apple_price_calculation_l1040_104002

/-- Calculates the price of each apple given the produce inventory and total worth -/
theorem apple_price_calculation (asparagus_bundles : ℕ) (asparagus_price : ℚ)
  (grape_boxes : ℕ) (grape_price : ℚ) (apple_count : ℕ) (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  grape_price = 5/2 →
  apple_count = 700 →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + grape_boxes * grape_price)) / apple_count = 1/2 := by
  sorry

end apple_price_calculation_l1040_104002


namespace finite_perfect_squares_l1040_104065

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (S : Finset ℤ), ∀ (n : ℤ),
    (∃ (x : ℤ), a * n^2 + b = x^2) ∧ (∃ (y : ℤ), a * (n + 1)^2 + b = y^2) →
    n ∈ S :=
sorry

end finite_perfect_squares_l1040_104065


namespace log11_not_expressible_l1040_104049

-- Define the given logarithmic values
def log5 : ℝ := 0.6990
def log6 : ℝ := 0.7782

-- Define a type for basic logarithmic expressions
inductive LogExpr
| Const : ℝ → LogExpr
| Log5 : LogExpr
| Log6 : LogExpr
| Add : LogExpr → LogExpr → LogExpr
| Sub : LogExpr → LogExpr → LogExpr
| Mul : ℝ → LogExpr → LogExpr

-- Function to evaluate a LogExpr
def eval : LogExpr → ℝ
| LogExpr.Const r => r
| LogExpr.Log5 => log5
| LogExpr.Log6 => log6
| LogExpr.Add e1 e2 => eval e1 + eval e2
| LogExpr.Sub e1 e2 => eval e1 - eval e2
| LogExpr.Mul r e => r * eval e

-- Theorem stating that log 11 cannot be expressed using log 5 and log 6
theorem log11_not_expressible : ∀ e : LogExpr, eval e ≠ Real.log 11 := by
  sorry

end log11_not_expressible_l1040_104049


namespace fish_rice_equivalence_l1040_104091

/-- Represents the value of one fish in terms of bags of rice -/
def fish_to_rice_ratio : ℚ := 21 / 20

theorem fish_rice_equivalence (fish bread rice : ℚ) 
  (h1 : 4 * fish = 3 * bread) 
  (h2 : 5 * bread = 7 * rice) : 
  fish = fish_to_rice_ratio * rice := by
  sorry

#check fish_rice_equivalence

end fish_rice_equivalence_l1040_104091


namespace empty_solution_set_iff_a_in_range_l1040_104007

/-- The quadratic function f(x) = (a²-4)x² + (a+2)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 4) * x^2 + (a + 2) * x - 1

/-- The set of x that satisfy the inequality f(x) ≥ 0 -/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- The theorem stating the range of a for which the solution set is empty -/
theorem empty_solution_set_iff_a_in_range :
  ∀ a : ℝ, solution_set a = ∅ ↔ a ∈ Set.Icc (-2) (6/5) := by sorry

end empty_solution_set_iff_a_in_range_l1040_104007


namespace infinitely_many_lines_through_lattice_points_l1040_104029

/-- A line passing through the point (10, 1/2) -/
structure LineThrough10Half where
  slope : ℤ
  intercept : ℚ
  eq : intercept = 1/2 - 10 * slope

/-- A lattice point is a point with integer coordinates -/
def LatticePoint (x y : ℤ) : Prop := True

/-- A line passes through a lattice point -/
def PassesThroughLatticePoint (line : LineThrough10Half) (x y : ℤ) : Prop :=
  y = line.slope * x + line.intercept

theorem infinitely_many_lines_through_lattice_points :
  ∃ (f : ℕ → LineThrough10Half),
    (∀ n : ℕ, ∃ (x₁ y₁ x₂ y₂ : ℤ), 
      x₁ ≠ x₂ ∧ 
      LatticePoint x₁ y₁ ∧ 
      LatticePoint x₂ y₂ ∧ 
      PassesThroughLatticePoint (f n) x₁ y₁ ∧ 
      PassesThroughLatticePoint (f n) x₂ y₂) ∧
    (∀ m n : ℕ, m ≠ n → f m ≠ f n) :=
  sorry

end infinitely_many_lines_through_lattice_points_l1040_104029


namespace arithmetic_sequence_general_term_l1040_104056

/-- An arithmetic sequence {a_n} with a_1 = 1 and a_3 = a_2^2 - 4 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  a 3 = (a 2)^2 - 4 ∧
  ∀ n m : ℕ, n < m → a n < a m

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l1040_104056


namespace circle_placement_existence_l1040_104075

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Represents a circle --/
structure Circle where
  diameter : ℝ

/-- Checks if a circle intersects a square --/
def circleIntersectsSquare (c : Circle) (s : Square) : Prop :=
  sorry

/-- Theorem: In a 20 by 25 rectangle with 120 unit squares, 
    there exists a point for a circle with diameter 1 that doesn't intersect any square --/
theorem circle_placement_existence 
  (r : Rectangle) 
  (squares : Finset Square) 
  (c : Circle) : 
  r.width = 20 ∧ 
  r.height = 25 ∧ 
  squares.card = 120 ∧ 
  (∀ s ∈ squares, s.side = 1) ∧ 
  c.diameter = 1 →
  ∃ (x y : ℝ), ∀ s ∈ squares, ¬circleIntersectsSquare { diameter := 1 } s :=
sorry

end circle_placement_existence_l1040_104075


namespace diamond_square_counts_l1040_104022

/-- Represents a diamond-shaped arrangement of colored squares -/
structure DiamondArrangement where
  sideLength : ℕ
  totalSquares : ℕ
  greenSquares : ℕ
  whiteSquares : ℕ

/-- Properties of the diamond arrangement -/
def validDiamondArrangement (d : DiamondArrangement) : Prop :=
  d.sideLength = 4 ∧
  d.totalSquares = (2 * d.sideLength + 1)^2 ∧
  d.greenSquares = (d.totalSquares + 1) / 2 ∧
  d.whiteSquares = (d.totalSquares - 1) / 2

theorem diamond_square_counts (d : DiamondArrangement) 
  (h : validDiamondArrangement d) : 
  d.whiteSquares = 40 ∧ 
  d.greenSquares = 41 ∧ 
  100 * d.whiteSquares + d.greenSquares = 4041 := by
  sorry

end diamond_square_counts_l1040_104022


namespace original_equals_scientific_l1040_104087

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 24538

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 2.4538,
  exponent := 4,
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end original_equals_scientific_l1040_104087


namespace library_books_end_of_month_l1040_104069

theorem library_books_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) :
  initial_books = 75 →
  loaned_books = 60 →
  return_rate = 65 / 100 →
  initial_books - loaned_books + (return_rate * loaned_books).floor = 54 :=
by
  sorry

end library_books_end_of_month_l1040_104069


namespace estimate_black_pieces_is_twelve_l1040_104035

/-- Represents the result of drawing chess pieces -/
structure DrawResult where
  total_pieces : ℕ
  total_draws : ℕ
  black_draws : ℕ

/-- Estimates the number of black chess pieces in the bag -/
def estimate_black_pieces (result : DrawResult) : ℚ :=
  result.total_pieces * (result.black_draws : ℚ) / result.total_draws

/-- Theorem: The estimated number of black chess pieces is 12 -/
theorem estimate_black_pieces_is_twelve (result : DrawResult) 
  (h1 : result.total_pieces = 20)
  (h2 : result.total_draws = 100)
  (h3 : result.black_draws = 60) : 
  estimate_black_pieces result = 12 := by
  sorry

#eval estimate_black_pieces ⟨20, 100, 60⟩

end estimate_black_pieces_is_twelve_l1040_104035


namespace haley_gives_away_48_papers_l1040_104011

/-- The number of origami papers Haley gives away -/
def total_papers (num_cousins : ℕ) (papers_per_cousin : ℕ) : ℕ :=
  num_cousins * papers_per_cousin

/-- Theorem stating that Haley gives away 48 origami papers -/
theorem haley_gives_away_48_papers : total_papers 6 8 = 48 := by
  sorry

end haley_gives_away_48_papers_l1040_104011


namespace smallest_sum_reciprocals_l1040_104020

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 81 :=
by sorry

end smallest_sum_reciprocals_l1040_104020


namespace not_corner_2010_l1040_104009

/-- Represents the sequence of corner numbers in the spiral -/
def corner_sequence : ℕ → ℕ
| 0 => 2
| 1 => 4
| n + 2 => corner_sequence (n + 1) + 8 * (n + 1)

/-- Checks if a number is a corner number in the spiral -/
def is_corner_number (n : ℕ) : Prop :=
  ∃ k : ℕ, corner_sequence k = n

/-- The main theorem stating that 2010 is not a corner number -/
theorem not_corner_2010 : ¬ is_corner_number 2010 := by
  sorry

#eval corner_sequence 0  -- Expected: 2
#eval corner_sequence 1  -- Expected: 4
#eval corner_sequence 2  -- Expected: 6
#eval corner_sequence 3  -- Expected: 10

end not_corner_2010_l1040_104009


namespace remaining_pages_l1040_104015

/-- Calculate the remaining pages of books after some are lost -/
theorem remaining_pages (initial_books : ℕ) (pages_per_book : ℕ) (lost_books : ℕ) 
  (h1 : initial_books ≥ lost_books) :
  (initial_books - lost_books) * pages_per_book = 
  initial_books * pages_per_book - lost_books * pages_per_book :=
by sorry

#check remaining_pages

end remaining_pages_l1040_104015


namespace river_flow_volume_l1040_104098

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
theorem river_flow_volume 
  (depth : ℝ) 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (h_depth : depth = 3) 
  (h_width : width = 32) 
  (h_flow_rate : flow_rate_kmph = 2) : 
  depth * width * (flow_rate_kmph * 1000 / 60) = 3200 := by
  sorry

#check river_flow_volume

end river_flow_volume_l1040_104098


namespace sum_of_bases_l1040_104082

/-- Given two bases R₁ and R₂, and two fractions F₁ and F₂, prove that R₁ + R₂ = 21 -/
theorem sum_of_bases (R₁ R₂ : ℕ) (F₁ F₂ : ℚ) : R₁ + R₂ = 21 :=
  by
  have h1 : F₁ = (4 * R₁ + 7) / (R₁^2 - 1) := by sorry
  have h2 : F₂ = (7 * R₁ + 4) / (R₁^2 - 1) := by sorry
  have h3 : F₁ = (R₂ + 6) / (R₂^2 - 1) := by sorry
  have h4 : F₂ = (6 * R₂ + 1) / (R₂^2 - 1) := by sorry
  sorry

end sum_of_bases_l1040_104082


namespace expression_value_l1040_104042

theorem expression_value (x y : ℝ) 
  (eq1 : x - y = -2)
  (eq2 : 2 * x + y = -1) :
  (x - y)^2 - (x - 2*y) * (x + 2*y) = 7 := by
  sorry

end expression_value_l1040_104042


namespace ellipse_properties_l1040_104079

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  y^2 / 4 + x^2 = 1

-- Define the point A
def point_A : ℝ × ℝ := (1, 0)

-- Define the condition for line l
def line_l (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 = 0

-- Define the equality of distances condition
def equal_distances (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 1)^2 + y1^2 = (x2 - 1)^2 + y2^2

-- Main theorem
theorem ellipse_properties :
  -- Given conditions
  (∃ (x y : ℝ), x = 1/2 ∧ y = Real.sqrt 3 ∧ ellipse_C x y) →
  -- Conclusions
  (∀ (k m x1 y1 x2 y2 : ℝ),
    -- Line l intersects ellipse C at M(x1, y1) and N(x2, y2)
    ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
    line_l k m x1 y1 ∧ line_l k m x2 y2 ∧
    -- AM ⊥ AN and |AM| = |AN|
    perpendicular x1 y1 x2 y2 ∧ equal_distances x1 y1 x2 y2 →
    -- Then line l has one of these equations
    (k = Real.sqrt 5 ∧ m = -3/5 * Real.sqrt 5) ∨
    (k = -Real.sqrt 5 ∧ m = -3/5 * Real.sqrt 5) ∨
    (k = 0 ∧ m = -3/5)) ∧
  -- The locus of H
  (∀ (x y : ℝ), x ≠ 1 →
    (∃ (k m x1 y1 x2 y2 : ℝ),
      ellipse_C x1 y1 ∧ ellipse_C x2 y2 ∧
      line_l k m x1 y1 ∧ line_l k m x2 y2 ∧
      perpendicular x1 y1 x2 y2 ∧
      -- H is on the perpendicular from A to MN
      (y - 0) / (x - 1) = -1 / k) ↔
    (x - 1/5)^2 + y^2 = 16/25) :=
sorry

end ellipse_properties_l1040_104079


namespace sun_moon_volume_ratio_l1040_104053

/-- The ratio of the Sun-Earth distance to the Moon-Earth distance -/
def distance_ratio : ℝ := 387

/-- The ratio of the Sun's volume to the Moon's volume -/
def volume_ratio : ℝ := distance_ratio ^ 3

theorem sun_moon_volume_ratio : 
  volume_ratio = distance_ratio ^ 3 := by sorry

end sun_moon_volume_ratio_l1040_104053


namespace quadratic_form_sum_l1040_104085

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (-3 * x^2 + 24 * x + 81 = a * (x + b)^2 + c) ∧ (a + b + c = 122) := by
  sorry

end quadratic_form_sum_l1040_104085


namespace square_difference_value_l1040_104076

theorem square_difference_value (m n : ℝ) (h : m^2 + n^2 = 6*m - 4*n - 13) : 
  m^2 - n^2 = 5 := by
sorry

end square_difference_value_l1040_104076


namespace quadricycles_count_l1040_104099

/-- Given a total number of children and wheels, calculate the number of quadricycles -/
def count_quadricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  let scooter_wheels := 2
  let quadricycle_wheels := 4
  let quadricycles := (total_wheels - scooter_wheels * total_children) / (quadricycle_wheels - scooter_wheels)
  quadricycles

/-- Theorem stating that given 9 children and 30 wheels, there are 6 quadricycles -/
theorem quadricycles_count : count_quadricycles 9 30 = 6 := by
  sorry

#eval count_quadricycles 9 30

end quadricycles_count_l1040_104099


namespace roots_of_equation_l1040_104004

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | f x = 0} = {2, 3, -2} := by
sorry

end roots_of_equation_l1040_104004


namespace kenny_basketball_hours_l1040_104074

/-- Represents the number of hours Kenny spent on different activities -/
structure KennyActivities where
  basketball : ℕ
  running : ℕ
  trumpet : ℕ

/-- Defines the relationships between Kenny's activities -/
def valid_activities (k : KennyActivities) : Prop :=
  k.running = 2 * k.basketball ∧
  k.trumpet = 2 * k.running ∧
  k.trumpet = 40

/-- Theorem: Given the conditions, Kenny played basketball for 10 hours -/
theorem kenny_basketball_hours (k : KennyActivities) 
  (h : valid_activities k) : k.basketball = 10 := by
  sorry


end kenny_basketball_hours_l1040_104074


namespace restaurant_bill_rounding_l1040_104032

theorem restaurant_bill_rounding (people : ℕ) (individual_payment : ℚ) (total_payment : ℚ) :
  people = 9 →
  individual_payment = 3491/100 →
  total_payment = 31419/100 →
  ∃ (original_bill : ℚ), 
    original_bill = 31418/100 ∧
    original_bill * people ≤ total_payment ∧
    total_payment - original_bill * people < people * (1/100) :=
by sorry

end restaurant_bill_rounding_l1040_104032


namespace donut_selections_l1040_104060

theorem donut_selections (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) : 
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end donut_selections_l1040_104060


namespace two_hundred_fiftieth_term_is_331_l1040_104006

/-- The sequence function that generates the nth term of the sequence 
    by omitting perfect squares and multiples of 5 -/
def sequenceFunction (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 250th term of the sequence is 331 -/
theorem two_hundred_fiftieth_term_is_331 : sequenceFunction 250 = 331 := by
  sorry

end two_hundred_fiftieth_term_is_331_l1040_104006


namespace large_duck_cost_large_duck_cost_proof_l1040_104023

/-- The cost of a large size duck given the following conditions:
  * Regular size ducks cost $3.00 each
  * 221 regular size ducks were sold
  * 185 large size ducks were sold
  * Total amount raised is $1588
-/
theorem large_duck_cost : ℝ → Prop :=
  λ large_cost : ℝ =>
    let regular_cost : ℝ := 3
    let regular_sold : ℕ := 221
    let large_sold : ℕ := 185
    let total_raised : ℝ := 1588
    (regular_cost * regular_sold + large_cost * large_sold = total_raised) →
    large_cost = 5

/-- Proof of the large duck cost theorem -/
theorem large_duck_cost_proof : large_duck_cost 5 := by
  sorry

end large_duck_cost_large_duck_cost_proof_l1040_104023


namespace du_chin_meat_pie_business_l1040_104017

/-- Du Chin's meat pie business theorem -/
theorem du_chin_meat_pie_business 
  (pies_baked : ℕ) 
  (price_per_pie : ℚ) 
  (ingredient_cost_ratio : ℚ) 
  (h1 : pies_baked = 200)
  (h2 : price_per_pie = 20)
  (h3 : ingredient_cost_ratio = 3/5) :
  pies_baked * price_per_pie - pies_baked * price_per_pie * ingredient_cost_ratio = 1600 :=
by sorry

end du_chin_meat_pie_business_l1040_104017


namespace solution_set_inequality_l1040_104047

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (1 - 2*x) ≥ 0 ↔ 1/2 ≤ x ∧ x ≤ 2 :=
by sorry

end solution_set_inequality_l1040_104047


namespace fraction_bounds_l1040_104024

theorem fraction_bounds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  0 ≤ (|x + y|^2) / (|x|^2 + |y|^2) ∧ (|x + y|^2) / (|x|^2 + |y|^2) ≤ 2 :=
by sorry

end fraction_bounds_l1040_104024


namespace purchase_group_equation_l1040_104092

/-- A group of people buying an item -/
structure PurchaseGroup where
  price : ℝ  -- Price of the item in coins
  excess_contribution : ℝ := 8  -- Contribution that exceeds the price
  excess_amount : ℝ := 3  -- Amount by which the excess contribution exceeds the price
  shortfall_contribution : ℝ := 7  -- Contribution that falls short of the price
  shortfall_amount : ℝ := 4  -- Amount by which the shortfall contribution falls short of the price

/-- The equation holds for a purchase group -/
theorem purchase_group_equation (g : PurchaseGroup) :
  (g.price + g.excess_amount) / g.excess_contribution = (g.price - g.shortfall_amount) / g.shortfall_contribution :=
sorry

end purchase_group_equation_l1040_104092


namespace smallest_k_satisfying_condition_l1040_104019

def S (n : ℕ) : ℤ := 2 * n^2 - 15 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem smallest_k_satisfying_condition : 
  (∀ k < 6, a k + a (k + 1) ≤ 12) ∧ 
  (a 6 + a 7 > 12) := by sorry

end smallest_k_satisfying_condition_l1040_104019


namespace inequality_solution_l1040_104005

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + x^2)) ↔ x ≥ -12 / 7 :=
by sorry

end inequality_solution_l1040_104005


namespace selling_price_is_14_l1040_104064

/-- Calculates the selling price per bowl given the following conditions:
    * Total number of glass bowls bought: 110
    * Cost per bowl: Rs. 10
    * Number of bowls sold: 100
    * Number of bowls broken: 10 (remaining)
    * Percentage gain: 27.27272727272727%
-/
def calculate_selling_price_per_bowl (total_bowls : ℕ) (cost_per_bowl : ℚ) 
  (bowls_sold : ℕ) (percentage_gain : ℚ) : ℚ :=
  let total_cost : ℚ := total_bowls * cost_per_bowl
  let gain : ℚ := percentage_gain * total_cost
  let total_selling_price : ℚ := total_cost + gain
  total_selling_price / bowls_sold

/-- Theorem stating that the selling price per bowl is 14 given the problem conditions -/
theorem selling_price_is_14 :
  calculate_selling_price_per_bowl 110 10 100 (27.27272727272727 / 100) = 14 := by
  sorry

#eval calculate_selling_price_per_bowl 110 10 100 (27.27272727272727 / 100)

end selling_price_is_14_l1040_104064
