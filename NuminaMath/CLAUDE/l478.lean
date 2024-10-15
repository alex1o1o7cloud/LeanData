import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_equality_l478_47844

theorem complex_number_equality (z : ℂ) : (z - 2) * Complex.I = 1 + Complex.I → z = 3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l478_47844


namespace NUMINAMATH_CALUDE_echo_earnings_l478_47827

-- Define the schools and their parameters
structure School where
  name : String
  students : ℕ
  days : ℕ
  rate_multiplier : ℚ

-- Define the problem parameters
def delta : School := { name := "Delta", students := 8, days := 4, rate_multiplier := 1 }
def echo : School := { name := "Echo", students := 6, days := 6, rate_multiplier := 3/2 }
def foxtrot : School := { name := "Foxtrot", students := 7, days := 7, rate_multiplier := 1 }

def total_payment : ℚ := 1284

-- Function to calculate effective student-days
def effective_student_days (s : School) : ℚ :=
  ↑s.students * ↑s.days * s.rate_multiplier

-- Theorem statement
theorem echo_earnings :
  let total_effective_days := effective_student_days delta + effective_student_days echo + effective_student_days foxtrot
  let daily_wage := total_payment / total_effective_days
  effective_student_days echo * daily_wage = 513.6 := by
sorry

end NUMINAMATH_CALUDE_echo_earnings_l478_47827


namespace NUMINAMATH_CALUDE_average_after_discarding_l478_47843

theorem average_after_discarding (numbers : Finset ℕ) (sum : ℕ) (n : ℕ) :
  Finset.card numbers = 50 →
  sum = Finset.sum numbers id →
  sum / 50 = 38 →
  45 ∈ numbers →
  55 ∈ numbers →
  (sum - 45 - 55) / 48 = 75/2 :=
by
  sorry

end NUMINAMATH_CALUDE_average_after_discarding_l478_47843


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l478_47884

/-- 
Given a rectangular metallic sheet with length 48 meters, 
from which squares of side 8 meters are cut from each corner to form an open box,
if the volume of the resulting box is 5120 cubic meters,
then the width of the original metallic sheet is 36 meters.
-/
theorem metallic_sheet_width : 
  ∀ (w : ℝ), 
    (48 - 2 * 8) * (w - 2 * 8) * 8 = 5120 → 
    w = 36 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l478_47884


namespace NUMINAMATH_CALUDE_has_18_divisors_smallest_with_18_divisors_smallest_integer_with_18_divisors_l478_47800

/-- A function that counts the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- 131072 has exactly 18 positive divisors -/
theorem has_18_divisors : countDivisors 131072 = 18 := sorry

/-- For any positive integer smaller than 131072, 
    the number of its positive divisors is not 18 -/
theorem smallest_with_18_divisors (n : ℕ) : 
  0 < n → n < 131072 → countDivisors n ≠ 18 := sorry

/-- 131072 is the smallest positive integer with exactly 18 positive divisors -/
theorem smallest_integer_with_18_divisors : 
  ∀ n : ℕ, 0 < n → countDivisors n = 18 → n ≥ 131072 := by
  sorry

end NUMINAMATH_CALUDE_has_18_divisors_smallest_with_18_divisors_smallest_integer_with_18_divisors_l478_47800


namespace NUMINAMATH_CALUDE_pizza_pooling_advantage_l478_47878

/-- Represents the size and price of a pizza --/
structure Pizza where
  side : ℕ
  price : ℕ

/-- Calculates the area of a square pizza --/
def pizzaArea (p : Pizza) : ℕ := p.side * p.side

/-- Represents the pizza options and money available --/
structure PizzaShop where
  smallPizza : Pizza
  largePizza : Pizza
  moneyPerPerson : ℕ
  numPeople : ℕ

/-- Calculates the maximum area of pizza that can be bought individually --/
def maxIndividualArea (shop : PizzaShop) : ℕ :=
  let smallArea := (shop.moneyPerPerson / shop.smallPizza.price) * pizzaArea shop.smallPizza
  let largeArea := (shop.moneyPerPerson / shop.largePizza.price) * pizzaArea shop.largePizza
  max smallArea largeArea * shop.numPeople

/-- Calculates the maximum area of pizza that can be bought by pooling money --/
def maxPooledArea (shop : PizzaShop) : ℕ :=
  let totalMoney := shop.moneyPerPerson * shop.numPeople
  let smallArea := (totalMoney / shop.smallPizza.price) * pizzaArea shop.smallPizza
  let largeArea := (totalMoney / shop.largePizza.price) * pizzaArea shop.largePizza
  max smallArea largeArea

theorem pizza_pooling_advantage (shop : PizzaShop) 
    (h1 : shop.smallPizza = ⟨6, 10⟩)
    (h2 : shop.largePizza = ⟨9, 20⟩)
    (h3 : shop.moneyPerPerson = 30)
    (h4 : shop.numPeople = 2) :
  maxPooledArea shop - maxIndividualArea shop = 27 := by
  sorry


end NUMINAMATH_CALUDE_pizza_pooling_advantage_l478_47878


namespace NUMINAMATH_CALUDE_cost_operation_l478_47857

theorem cost_operation (t : ℝ) (b b' : ℝ) : 
  (∀ C, C = t * b^4) →
  (∃ e, e = 16 * t * b^4) →
  (∃ e, e = t * b'^4) →
  b' = 2 * b :=
sorry

end NUMINAMATH_CALUDE_cost_operation_l478_47857


namespace NUMINAMATH_CALUDE_lottery_probability_l478_47874

theorem lottery_probability (total_tickets : Nat) (winning_tickets : Nat) (buyers : Nat) :
  total_tickets = 10 →
  winning_tickets = 3 →
  buyers = 5 →
  let prob_at_least_one_wins := 1 - (Nat.choose (total_tickets - winning_tickets) buyers / Nat.choose total_tickets buyers)
  prob_at_least_one_wins = 77 / 84 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l478_47874


namespace NUMINAMATH_CALUDE_counterexample_exists_l478_47837

theorem counterexample_exists : ∃ n : ℕ, 
  (¬ Nat.Prime n) ∧ (Nat.Prime (n - 3) ∨ Nat.Prime (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l478_47837


namespace NUMINAMATH_CALUDE_journey_theorem_l478_47818

/-- Represents a two-segment journey with different speeds -/
structure Journey where
  time_at_5mph : ℝ
  time_at_15mph : ℝ
  total_time : ℝ
  total_distance : ℝ

/-- The average speed of the entire journey is 10 mph -/
def average_speed (j : Journey) : Prop :=
  j.total_distance / j.total_time = 10

/-- The total time is the sum of time spent at each speed -/
def total_time_sum (j : Journey) : Prop :=
  j.total_time = j.time_at_5mph + j.time_at_15mph

/-- The total distance is the sum of distances covered at each speed -/
def total_distance_sum (j : Journey) : Prop :=
  j.total_distance = 5 * j.time_at_5mph + 15 * j.time_at_15mph

/-- The fraction of time spent at 15 mph is half of the total time -/
def half_time_at_15mph (j : Journey) : Prop :=
  j.time_at_15mph / j.total_time = 1 / 2

theorem journey_theorem (j : Journey) 
  (h1 : average_speed j) 
  (h2 : total_time_sum j) 
  (h3 : total_distance_sum j) : 
  half_time_at_15mph j := by
  sorry

end NUMINAMATH_CALUDE_journey_theorem_l478_47818


namespace NUMINAMATH_CALUDE_probability_h_in_mathematics_l478_47836

def word : String := "Mathematics"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_h_in_mathematics :
  (count_letter word 'h' : ℚ) / word.length = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_h_in_mathematics_l478_47836


namespace NUMINAMATH_CALUDE_sally_earnings_l478_47868

/-- Sally's earnings per house, given her total earnings and number of houses cleaned -/
def earnings_per_house (total_earnings : ℕ) (houses_cleaned : ℕ) : ℚ :=
  (total_earnings : ℚ) / houses_cleaned

/-- Conversion factor from dozens to units -/
def dozens_to_units : ℕ := 12

theorem sally_earnings :
  let total_dozens : ℕ := 200
  let houses_cleaned : ℕ := 96
  earnings_per_house (total_dozens * dozens_to_units) houses_cleaned = 25 := by
sorry

end NUMINAMATH_CALUDE_sally_earnings_l478_47868


namespace NUMINAMATH_CALUDE_expression_value_l478_47845

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x*y)) : 
  1 / (1 + x^2) + 1 / (1 + y^2) + 2 / (1 + x*y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l478_47845


namespace NUMINAMATH_CALUDE_snowman_height_example_l478_47864

/-- The height of a snowman built from three vertically aligned spheres -/
def snowman_height (r1 r2 r3 : ℝ) : ℝ := 2 * (r1 + r2 + r3)

/-- Theorem: The height of a snowman with spheres of radii 10 cm, 20 cm, and 30 cm is 120 cm -/
theorem snowman_height_example : snowman_height 10 20 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_snowman_height_example_l478_47864


namespace NUMINAMATH_CALUDE_parallelogram_split_slope_l478_47821

/-- A parallelogram in a 2D plane --/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- A line in a 2D plane represented by its slope --/
structure Line where
  slope : ℝ

/-- Predicate to check if a line passes through the origin and splits a parallelogram into two congruent parts --/
def splits_congruently (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The main theorem --/
theorem parallelogram_split_slope :
  let p := Parallelogram.mk (8, 50) (8, 120) (30, 160) (30, 90)
  let l := Line.mk (265 / 38)
  splits_congruently p l := by sorry

end NUMINAMATH_CALUDE_parallelogram_split_slope_l478_47821


namespace NUMINAMATH_CALUDE_cookie_count_l478_47866

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l478_47866


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l478_47883

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_atoms hydrogen_atoms oxygen_atoms : ℕ) 
  (carbon_weight hydrogen_weight oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_weight + 
  (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 4 Carbon atoms, 1 Hydrogen atom, and 1 Oxygen atom 
    is equal to 65.048 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 4 1 1 12.01 1.008 16.00 = 65.048 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l478_47883


namespace NUMINAMATH_CALUDE_sarahs_brother_apples_l478_47871

theorem sarahs_brother_apples (sarah_apples : ℕ) (ratio : ℕ) (brother_apples : ℕ) : 
  sarah_apples = 45 → 
  ratio = 5 → 
  sarah_apples = ratio * brother_apples → 
  brother_apples = 9 := by
sorry

end NUMINAMATH_CALUDE_sarahs_brother_apples_l478_47871


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_range_l478_47887

theorem function_inequality_implies_k_range (k : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 3, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 3, 
    2 * x₁^2 + x₁ - k ≤ x₀^3 - 3 * x₀) → 
  k ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_range_l478_47887


namespace NUMINAMATH_CALUDE_bears_distribution_l478_47886

def bears_per_shelf (initial_stock new_shipment num_shelves : ℕ) : ℕ :=
  (initial_stock + new_shipment) / num_shelves

theorem bears_distribution (initial_stock new_shipment num_shelves : ℕ) 
  (h1 : initial_stock = 17)
  (h2 : new_shipment = 10)
  (h3 : num_shelves = 3) :
  bears_per_shelf initial_stock new_shipment num_shelves = 9 := by
  sorry

end NUMINAMATH_CALUDE_bears_distribution_l478_47886


namespace NUMINAMATH_CALUDE_max_value_with_remainder_l478_47822

theorem max_value_with_remainder (A B : ℕ) : 
  A ≠ B → 
  A = 17 * 25 + B → 
  B < 17 → 
  (∀ C : ℕ, C < 17 → 17 * 25 + C ≤ 17 * 25 + B) → 
  A = 441 :=
by sorry

end NUMINAMATH_CALUDE_max_value_with_remainder_l478_47822


namespace NUMINAMATH_CALUDE_largest_difference_l478_47831

def P : ℕ := 3 * 2003^2004
def Q : ℕ := 2003^2004
def R : ℕ := 2002 * 2003^2003
def S : ℕ := 3 * 2003^2003
def T : ℕ := 2003^2003
def U : ℕ := 2003^2002

theorem largest_difference (P Q R S T U : ℕ) 
  (hP : P = 3 * 2003^2004)
  (hQ : Q = 2003^2004)
  (hR : R = 2002 * 2003^2003)
  (hS : S = 3 * 2003^2003)
  (hT : T = 2003^2003)
  (hU : U = 2003^2002) :
  P - Q > Q - R ∧ P - Q > R - S ∧ P - Q > S - T ∧ P - Q > T - U :=
by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l478_47831


namespace NUMINAMATH_CALUDE_angle_problem_l478_47813

theorem angle_problem (A B : ℝ) (h1 : A = 4 * B) (h2 : 90 - B = 4 * (90 - A)) : B = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l478_47813


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l478_47825

theorem smallest_n_for_exact_tax : ∃ (x : ℕ), (105 * x) % 10000 = 0 ∧ 
  (∀ (y : ℕ), y < 21 → (105 * y) % 10000 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l478_47825


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l478_47811

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l478_47811


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l478_47828

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (5, 0)

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3 * y = 4 * x

-- Theorem statement
theorem distance_focus_to_asymptote :
  let F := right_focus
  ∃ (d : ℝ), d = 4 ∧
  ∀ (x y : ℝ), asymptote x y →
    (F.1 - x)^2 + (F.2 - y)^2 = d^2 :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l478_47828


namespace NUMINAMATH_CALUDE_min_value_quadratic_l478_47823

theorem min_value_quadratic (x y : ℝ) (h1 : |y| ≤ 1) (h2 : 2 * x + y = 1) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x' y' : ℝ), |y'| ≤ 1 → 2 * x' + y' = 1 →
    2 * x'^2 + 16 * x' + 3 * y'^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l478_47823


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l478_47892

theorem salary_increase_percentage (S : ℝ) (h1 : S + 0.10 * S = 330) : 
  ∃ P : ℝ, S + (P / 100) * S = 348 ∧ P = 16 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l478_47892


namespace NUMINAMATH_CALUDE_equidistant_function_b_squared_l478_47861

/-- A complex function that is equidistant from z and the origin -/
def equidistant_function (a b : ℝ) : ℂ → ℂ := fun z ↦ (a + b * Complex.I) * z

/-- The property that f(z) is equidistant from z and the origin for all z -/
def is_equidistant (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (f z - z) = Complex.abs (f z)

theorem equidistant_function_b_squared
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_equidistant : is_equidistant (equidistant_function a b))
  (h_norm : Complex.abs (a + b * Complex.I) = 5) :
  b^2 = 99/4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_function_b_squared_l478_47861


namespace NUMINAMATH_CALUDE_division_problem_l478_47830

/-- Proves that given a total amount of 544, if A gets 2/3 of what B gets, and B gets 1/4 of what C gets, then A gets 64. -/
theorem division_problem (total : ℚ) (a b c : ℚ) 
  (h_total : total = 544)
  (h_ab : a = (2/3) * b)
  (h_bc : b = (1/4) * c)
  (h_sum : a + b + c = total) : 
  a = 64 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l478_47830


namespace NUMINAMATH_CALUDE_average_weight_increase_l478_47819

/-- Proves that replacing a person weighing 68 kg with a person weighing 95.5 kg
    in a group of 5 people increases the average weight by 5.5 kg -/
theorem average_weight_increase (initial_average : ℝ) :
  let initial_total := 5 * initial_average
  let new_total := initial_total - 68 + 95.5
  let new_average := new_total / 5
  new_average - initial_average = 5.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l478_47819


namespace NUMINAMATH_CALUDE_max_value_xy_expression_l478_47801

theorem max_value_xy_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x + 5*y < 90) :
  xy*(90 - 4*x - 5*y) ≤ 1350 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4*x₀ + 5*y₀ < 90 ∧ x₀*y₀*(90 - 4*x₀ - 5*y₀) = 1350 :=
by sorry

end NUMINAMATH_CALUDE_max_value_xy_expression_l478_47801


namespace NUMINAMATH_CALUDE_artist_paintings_l478_47829

theorem artist_paintings (paint_per_large : ℕ) (paint_per_small : ℕ) 
  (small_paintings : ℕ) (total_paint : ℕ) :
  paint_per_large = 3 →
  paint_per_small = 2 →
  small_paintings = 4 →
  total_paint = 17 →
  ∃ (large_paintings : ℕ), 
    large_paintings * paint_per_large + small_paintings * paint_per_small = total_paint ∧
    large_paintings = 3 :=
by sorry

end NUMINAMATH_CALUDE_artist_paintings_l478_47829


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l478_47805

theorem arithmetic_expression_evaluation : 3 + 2 * (8 - 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l478_47805


namespace NUMINAMATH_CALUDE_heat_of_formation_C6H6_value_l478_47856

-- Define the heat changes for the given reactions
def heat_change_C2H2 : ℝ := 226.7
def heat_change_3C2H2_to_C6H6 : ℝ := 631.1
def heat_change_C6H6_gas_to_liquid : ℝ := -33.9

-- Define the function to calculate the heat of formation
def heat_of_formation_C6H6 : ℝ :=
  -3 * heat_change_C2H2 + heat_change_3C2H2_to_C6H6 - heat_change_C6H6_gas_to_liquid

-- Theorem statement
theorem heat_of_formation_C6H6_value :
  heat_of_formation_C6H6 = -82.9 := by sorry

end NUMINAMATH_CALUDE_heat_of_formation_C6H6_value_l478_47856


namespace NUMINAMATH_CALUDE_target_probability_l478_47853

/-- The probability of hitting the target in a single shot -/
def p : ℝ := 0.8

/-- The probability of missing the target in a single shot -/
def q : ℝ := 1 - p

/-- The probability of hitting the target at least once in two shots -/
def prob_hit_at_least_once_in_two : ℝ := 1 - q^2

theorem target_probability :
  prob_hit_at_least_once_in_two = 0.96 →
  (5 : ℝ) * p^4 * q = 0.4096 :=
sorry

end NUMINAMATH_CALUDE_target_probability_l478_47853


namespace NUMINAMATH_CALUDE_point_on_line_segment_l478_47898

def A (m : ℝ) : ℝ × ℝ := (m^2, 2)
def B (m : ℝ) : ℝ × ℝ := (2*m^2 + 2, 2)
def M (m : ℝ) : ℝ × ℝ := (-m^2, 2)
def N (m : ℝ) : ℝ × ℝ := (m^2, m^2 + 2)
def P (m : ℝ) : ℝ × ℝ := (m^2 + 1, 2)
def Q (m : ℝ) : ℝ × ℝ := (3*m^2, 2)

theorem point_on_line_segment (m : ℝ) :
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → M m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → N m = ((1 - t) • (A m) + t • (B m))) ∧
  (¬ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → Q m = ((1 - t) • (A m) + t • (B m))) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_segment_l478_47898


namespace NUMINAMATH_CALUDE_union_of_nonnegative_and_less_than_one_is_real_l478_47817

theorem union_of_nonnegative_and_less_than_one_is_real : 
  ({x : ℝ | x ≥ 0} ∪ {x : ℝ | x < 1}) = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_of_nonnegative_and_less_than_one_is_real_l478_47817


namespace NUMINAMATH_CALUDE_standard_deviation_proof_l478_47855

/-- The average age of job applicants -/
def average_age : ℝ := 31

/-- The number of different ages in the acceptable range -/
def different_ages : ℕ := 19

/-- The standard deviation of applicants' ages -/
def standard_deviation : ℝ := 9

/-- Theorem stating that the standard deviation is correct given the problem conditions -/
theorem standard_deviation_proof : 
  (average_age + standard_deviation) - (average_age - standard_deviation) = different_ages - 1 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_proof_l478_47855


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l478_47815

/-- A geometric sequence with specific conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  second_term : a 2 = 6
  sum_condition : 6 * a 1 + a 3 = 30

/-- The general term formula for the geometric sequence -/
def general_term (seq : GeometricSequence) : ℕ → ℝ
| n => (3 * 3^(n - 1) : ℝ)

/-- Alternative general term formula for the geometric sequence -/
def general_term_alt (seq : GeometricSequence) : ℕ → ℝ
| n => (2 * 2^(n - 1) : ℝ)

/-- Theorem stating that one of the general term formulas is correct -/
theorem geometric_sequence_general_term (seq : GeometricSequence) :
  (∀ n, seq.a n = general_term seq n) ∨ (∀ n, seq.a n = general_term_alt seq n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l478_47815


namespace NUMINAMATH_CALUDE_cylindrical_surface_is_cylinder_l478_47803

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSurface c

theorem cylindrical_surface_is_cylinder (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSurface c) := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_surface_is_cylinder_l478_47803


namespace NUMINAMATH_CALUDE_no_prime_satisfies_condition_l478_47870

theorem no_prime_satisfies_condition : ¬∃ (P : ℕ), Prime P ∧ (100 : ℚ) * P = P + (1386 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_satisfies_condition_l478_47870


namespace NUMINAMATH_CALUDE_train_passing_platform_l478_47849

theorem train_passing_platform (train_length platform_length : ℝ) 
  (time_to_pass_point : ℝ) (h1 : train_length = 1400) 
  (h2 : platform_length = 700) (h3 : time_to_pass_point = 100) :
  (train_length + platform_length) / (train_length / time_to_pass_point) = 150 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l478_47849


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_negative_coefficient_inverse_relationship_regression_passes_through_mean_confidence_level_interpretation_l478_47885

-- Define a dataset as a list of real numbers
def Dataset := List Real

-- Define the variance of a dataset
noncomputable def variance (D : Dataset) : Real := sorry

-- 1. Variance remains unchanged when adding a constant
theorem variance_invariant_under_translation (D : Dataset) (c : Real) :
  variance (D.map (· + c)) = variance D := sorry

-- 2. Negative coefficient in regression equation implies inverse relationship
theorem negative_coefficient_inverse_relationship (a b x : Real) (h : b < 0) :
  let y₁ := a + b * x
  let y₂ := a + b * (x + 1)
  y₂ < y₁ := sorry

-- 3. Linear regression equation passes through the mean point
theorem regression_passes_through_mean (a b : Real) (D : Dataset) :
  let x_mean := (D.sum) / D.length
  let y_mean := (D.map (λ x => a + b * x)).sum / D.length
  y_mean = a + b * x_mean := sorry

-- 4. Confidence level interpretation
theorem confidence_level_interpretation (confidence_level : Real) (h : confidence_level = 0.99) :
  ∃ (error_rate : Real), error_rate = 1 - confidence_level := sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_negative_coefficient_inverse_relationship_regression_passes_through_mean_confidence_level_interpretation_l478_47885


namespace NUMINAMATH_CALUDE_max_surface_area_after_cut_l478_47882

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def Cuboid.surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.length * c.height)

/-- Represents the result of cutting a cuboid into two triangular prisms -/
structure CutResult where
  prism1_surface_area : ℝ
  prism2_surface_area : ℝ

/-- Calculates the sum of surface areas after cutting -/
def CutResult.totalSurfaceArea (cr : CutResult) : ℝ :=
  cr.prism1_surface_area + cr.prism2_surface_area

/-- The main theorem stating the maximum sum of surface areas after cutting -/
theorem max_surface_area_after_cut (c : Cuboid) 
  (h1 : c.length = 5) 
  (h2 : c.width = 4) 
  (h3 : c.height = 3) : 
  (∃ (cr : CutResult), ∀ (cr' : CutResult), cr.totalSurfaceArea ≥ cr'.totalSurfaceArea) → 
  (∃ (cr : CutResult), cr.totalSurfaceArea = 144) :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_after_cut_l478_47882


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l478_47850

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l478_47850


namespace NUMINAMATH_CALUDE_even_function_intersection_l478_47809

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem even_function_intersection (ω φ : ℝ) :
  (0 < φ) → (φ < π) →
  (∀ x, f ω φ x = f ω φ (-x)) →
  (∃ x₁ x₂, f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₁ - x₂| = π) →
  ω = 2 ∧ φ = π/2 := by
sorry

end NUMINAMATH_CALUDE_even_function_intersection_l478_47809


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l478_47890

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3) ↔ x = -3/2 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l478_47890


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l478_47893

/-- The standard equation of a parabola with directrix x = 1 -/
theorem parabola_standard_equation (x y : ℝ) :
  (∃ (p : ℝ), p > 0 ∧ 1 = p / 2 ∧ x < -p / 2) →
  (∀ point : ℝ × ℝ, point ∈ {(x, y) | y^2 = -4*x} ↔
    dist point (x, 0) = dist point (1, (point.2))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l478_47893


namespace NUMINAMATH_CALUDE_beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l478_47895

-- Define the "beautiful association number"
def beautiful_association_number (x y a : ℚ) : ℚ :=
  |x - a| + |y - a|

-- Part 1
theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 = 8 := by sorry

-- Part 2
theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 = 4 → x = 6 ∨ x = 0 := by sorry

-- Part 3
theorem beautiful_association_number_part3 (x₀ x₁ x₂ x₃ x₄ x₅ : ℚ) :
  beautiful_association_number x₀ x₁ 1 = 1 →
  beautiful_association_number x₁ x₂ 2 = 1 →
  beautiful_association_number x₂ x₃ 3 = 1 →
  beautiful_association_number x₃ x₄ 4 = 1 →
  beautiful_association_number x₄ x₅ 5 = 1 →
  ∃ (min : ℚ), min = 10 ∧ x₁ + x₂ + x₃ + x₄ ≥ min := by sorry

end NUMINAMATH_CALUDE_beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l478_47895


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l478_47860

theorem max_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = x + 2*y ∧ ∀ (w : ℝ), w = x + 2*y → w ≤ z ∧ z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l478_47860


namespace NUMINAMATH_CALUDE_d_bounds_l478_47894

/-- The maximum number of black squares on an n × n board where each black square
    has exactly two neighboring black squares. -/
def d (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds for d(n) -/
theorem d_bounds (n : ℕ) : 
  (2/3 : ℝ) * n^2 - 8 * n ≤ (d n : ℝ) ∧ (d n : ℝ) ≤ (2/3 : ℝ) * n^2 + 4 * n :=
sorry

end NUMINAMATH_CALUDE_d_bounds_l478_47894


namespace NUMINAMATH_CALUDE_hexagon_area_2016_l478_47812

/-- The area of the hexagon formed by constructing squares on the sides of a right triangle -/
def hexagon_area (a b : ℕ) : ℕ := 2 * (a^2 + b^2 + a*b)

/-- The proposition to be proved -/
theorem hexagon_area_2016 :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ hexagon_area a b = 2016 ∧
  (∀ (x y : ℕ), x > 0 → y > 0 → hexagon_area x y = 2016 → (x = 12 ∧ y = 24) ∨ (x = 24 ∧ y = 12)) :=
sorry

end NUMINAMATH_CALUDE_hexagon_area_2016_l478_47812


namespace NUMINAMATH_CALUDE_bridget_block_collection_l478_47832

/-- The number of groups of blocks in Bridget's collection -/
def num_groups : ℕ := 82

/-- The number of blocks in each group -/
def blocks_per_group : ℕ := 10

/-- The total number of blocks in Bridget's collection -/
def total_blocks : ℕ := num_groups * blocks_per_group

theorem bridget_block_collection :
  total_blocks = 820 :=
by sorry

end NUMINAMATH_CALUDE_bridget_block_collection_l478_47832


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l478_47841

-- Define a parabola
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

-- Define a line
structure Line where
  m : ℝ
  b : ℝ

-- Define the concept of a directrix
def is_directrix (l : Line) (p : Parabola) : Prop := sorry

-- Define the concept of tangency
def is_tangent (l : Line) (p : Parabola) : Prop := sorry

-- Define the concept of intersection
def intersect (l : Line) (p : Parabola) : Finset ℝ := sorry

-- Main theorem
theorem parabola_line_intersection
  (p : Parabola)
  (l1 l2 : Line)
  (h1 : l1.m ≠ l2.m ∨ l1.b ≠ l2.b) -- lines are distinct
  (h2 : is_directrix l1 p)
  (h3 : ¬ is_tangent l1 p)
  (h4 : ¬ is_tangent l2 p) :
  (intersect l1 p).card + (intersect l2 p).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l478_47841


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l478_47859

/-- Given a point A with coordinates (-2,4), this theorem states that the point
    symmetric to A with respect to the y-axis has coordinates (2,4). -/
theorem symmetric_point_wrt_y_axis :
  let A : ℝ × ℝ := (-2, 4)
  let symmetric_point := (- A.1, A.2)
  symmetric_point = (2, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_axis_l478_47859


namespace NUMINAMATH_CALUDE_least_divisible_by_three_smallest_primes_gt_7_l478_47810

def smallest_prime_greater_than_7 : ℕ := 11
def second_smallest_prime_greater_than_7 : ℕ := 13
def third_smallest_prime_greater_than_7 : ℕ := 17

theorem least_divisible_by_three_smallest_primes_gt_7 :
  ∃ n : ℕ, n > 0 ∧ 
  smallest_prime_greater_than_7 ∣ n ∧
  second_smallest_prime_greater_than_7 ∣ n ∧
  third_smallest_prime_greater_than_7 ∣ n ∧
  ∀ m : ℕ, m > 0 → 
    smallest_prime_greater_than_7 ∣ m →
    second_smallest_prime_greater_than_7 ∣ m →
    third_smallest_prime_greater_than_7 ∣ m →
    n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_three_smallest_primes_gt_7_l478_47810


namespace NUMINAMATH_CALUDE_stock_price_return_l478_47851

theorem stock_price_return (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.3
  let decrease_rate := 1 - 1 / 1.3
  increased_price * (1 - decrease_rate) = original_price :=
by
  sorry

#eval (1 - 1 / 1.3) * 100 -- This will output approximately 23.08

end NUMINAMATH_CALUDE_stock_price_return_l478_47851


namespace NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l478_47835

theorem belt_and_road_population_scientific_notation :
  (4600000000 : ℝ) = 4.6 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_belt_and_road_population_scientific_notation_l478_47835


namespace NUMINAMATH_CALUDE_value_of_4x_l478_47880

theorem value_of_4x (x : ℝ) (h : 2 * x - 3 = 10) : 4 * x = 26 := by
  sorry

end NUMINAMATH_CALUDE_value_of_4x_l478_47880


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l478_47842

def f (x : ℝ) := x^3 + x - 5

theorem solution_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → (f 1.5 < 0) →
  ∃ x, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l478_47842


namespace NUMINAMATH_CALUDE_stool_height_l478_47804

-- Define the constants
def ceiling_height : ℝ := 300  -- in cm
def bulb_below_ceiling : ℝ := 15  -- in cm
def alice_height : ℝ := 160  -- in cm
def alice_reach : ℝ := 50  -- in cm

-- Define the theorem
theorem stool_height : 
  ∃ (h : ℝ), 
    h = ceiling_height - bulb_below_ceiling - (alice_height + alice_reach) ∧ 
    h = 75 :=
by sorry

end NUMINAMATH_CALUDE_stool_height_l478_47804


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l478_47891

theorem polynomial_division_remainder (m b : ℤ) : 
  (∃ q : Polynomial ℤ, x^5 - 4*x^4 + 12*x^3 - 14*x^2 + 8*x + 5 = 
    (x^2 - 3*x + m) * q + (2*x + b)) → 
  m = 1 ∧ b = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l478_47891


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l478_47872

theorem smallest_angle_solution (x : Real) : 
  (∀ y : Real, y > 0 ∧ 8 * Real.sin y * Real.cos y^5 - 8 * Real.sin y^5 * Real.cos y = 1 → x ≤ y) ∧ 
  (x > 0 ∧ 8 * Real.sin x * Real.cos x^5 - 8 * Real.sin x^5 * Real.cos x = 1) →
  x = π / 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l478_47872


namespace NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l478_47824

theorem multiple_of_six_is_multiple_of_three (m : ℤ) :
  (∀ k : ℤ, ∃ n : ℤ, k * 6 = n * 3) →
  (∃ l : ℤ, m = l * 6) →
  (∃ j : ℤ, m = j * 3) :=
sorry

end NUMINAMATH_CALUDE_multiple_of_six_is_multiple_of_three_l478_47824


namespace NUMINAMATH_CALUDE_wholesale_price_calculation_l478_47807

/-- The wholesale price of a pair of pants -/
def wholesale_price : ℝ := 20

/-- The retail price of a pair of pants -/
def retail_price : ℝ := 36

/-- The markup percentage as a decimal -/
def markup : ℝ := 0.8

theorem wholesale_price_calculation :
  wholesale_price = retail_price / (1 + markup) :=
by sorry

end NUMINAMATH_CALUDE_wholesale_price_calculation_l478_47807


namespace NUMINAMATH_CALUDE_fill_time_without_leakage_l478_47806

/-- Represents the time to fill a tank with leakage -/
def fill_time_with_leakage : ℝ := 18

/-- Represents the time to empty the tank due to leakage -/
def empty_time_leakage : ℝ := 36

/-- Represents the volume of the tank -/
def tank_volume : ℝ := 1

/-- Theorem stating the time to fill the tank without leakage -/
theorem fill_time_without_leakage :
  let fill_rate := tank_volume / fill_time_with_leakage + tank_volume / empty_time_leakage
  tank_volume / fill_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_fill_time_without_leakage_l478_47806


namespace NUMINAMATH_CALUDE_unique_y_value_l478_47820

theorem unique_y_value (x : ℝ) (h : x^2 + 4 * (x / (x + 3))^2 = 64) : 
  ((x + 3)^2 * (x - 2)) / (2 * x + 3) = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_y_value_l478_47820


namespace NUMINAMATH_CALUDE_probability_x_gt_7y_in_rectangle_l478_47847

/-- The probability of a point (x,y) satisfying x > 7y in a specific rectangle -/
theorem probability_x_gt_7y_in_rectangle : 
  let rectangle_area := 2009 * 2010
  let triangle_area := (1 / 2) * 2009 * (2009 / 7)
  triangle_area / rectangle_area = 287 / 4020 := by
sorry

end NUMINAMATH_CALUDE_probability_x_gt_7y_in_rectangle_l478_47847


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l478_47865

def rectangle_width : ℕ := 3

def odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def rectangle_lengths : List ℕ := odd_numbers.map (λ x => x * x)

def rectangle_areas : List ℕ := rectangle_lengths.map (λ x => rectangle_width * x)

theorem sum_of_rectangle_areas :
  rectangle_areas.sum = 1365 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l478_47865


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l478_47852

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (x - 4) = 10 → x = 104 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l478_47852


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l478_47814

theorem pure_imaginary_fraction (a : ℝ) : 
  (((a : ℂ) - Complex.I) / (1 + Complex.I)).re = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l478_47814


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l478_47889

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + x + 2

-- Define the points on the parabola
def point_a : ℝ × ℝ := (2, f 2)
def point_b : ℝ × ℝ := (-1, f (-1))
def point_c : ℝ × ℝ := (3, f 3)

-- Theorem stating the relationship between a, b, and c
theorem parabola_point_relationship :
  point_c.2 > point_a.2 ∧ point_a.2 > point_b.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l478_47889


namespace NUMINAMATH_CALUDE_product_14_sum_5_or_minus_5_l478_47858

theorem product_14_sum_5_or_minus_5 (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 14 →
  a + b + c + d = 5 ∨ a + b + c + d = -5 := by
sorry

end NUMINAMATH_CALUDE_product_14_sum_5_or_minus_5_l478_47858


namespace NUMINAMATH_CALUDE_number_division_problem_l478_47881

theorem number_division_problem :
  ∃ x : ℝ, (x / 9 + x + 9 = 69) ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l478_47881


namespace NUMINAMATH_CALUDE_kim_candy_bars_saved_l478_47875

/-- The number of candy bars Kim's dad buys her per week -/
def candyBarsPerWeek : ℕ := 2

/-- The number of weeks it takes Kim to eat one candy bar -/
def weeksPerCandyBar : ℕ := 4

/-- The total number of weeks -/
def totalWeeks : ℕ := 16

/-- The number of candy bars Kim saved after the total number of weeks -/
def candyBarsSaved : ℕ := totalWeeks * candyBarsPerWeek - totalWeeks / weeksPerCandyBar

theorem kim_candy_bars_saved : candyBarsSaved = 28 := by
  sorry

end NUMINAMATH_CALUDE_kim_candy_bars_saved_l478_47875


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_four_l478_47834

theorem tangent_slope_at_pi_over_four :
  let f (x : ℝ) := Real.tan x
  (deriv f) (π / 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_four_l478_47834


namespace NUMINAMATH_CALUDE_series_sum_l478_47848

theorem series_sum : 
  let a : ℕ → ℝ := λ n => n / 5^n
  let S := ∑' n, a n
  S = 5/16 := by
sorry

end NUMINAMATH_CALUDE_series_sum_l478_47848


namespace NUMINAMATH_CALUDE_least_seven_digit_binary_l478_47896

theorem least_seven_digit_binary : ∀ n : ℕ, 
  (n > 0 ∧ n < 64) → (Nat.bits n).length < 7 ∧ 
  (Nat.bits 64).length = 7 :=
by sorry

end NUMINAMATH_CALUDE_least_seven_digit_binary_l478_47896


namespace NUMINAMATH_CALUDE_employed_males_percentage_l478_47826

/-- Proves that the percentage of the population that are employed males is 80%,
    given that 120% of the population are employed and 33.33333333333333% of employed people are females. -/
theorem employed_males_percentage (total_employed : Real) (female_employed_ratio : Real) :
  total_employed = 120 →
  female_employed_ratio = 100/3 →
  (1 - female_employed_ratio / 100) * total_employed = 80 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l478_47826


namespace NUMINAMATH_CALUDE_intersection_point_sum_l478_47867

theorem intersection_point_sum (a b : ℚ) : 
  (∃ x y : ℚ, x = (1/4)*y + a ∧ y = (1/4)*x + b ∧ x = 1 ∧ y = 2) →
  a + b = 9/4 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l478_47867


namespace NUMINAMATH_CALUDE_largest_constructible_cube_l478_47899

/-- Represents the dimensions of the cardboard sheet -/
def sheet_length : ℕ := 60
def sheet_width : ℕ := 25

/-- Checks if a cube with given edge length can be constructed from the sheet -/
def can_construct_cube (edge_length : ℕ) : Prop :=
  6 * edge_length^2 ≤ sheet_length * sheet_width ∧ 
  edge_length ≤ sheet_length ∧ 
  edge_length ≤ sheet_width

/-- The largest cube edge length that can be constructed -/
def max_cube_edge : ℕ := 15

/-- Theorem stating that the largest constructible cube has edge length of 15 cm -/
theorem largest_constructible_cube :
  can_construct_cube max_cube_edge ∧
  ∀ (n : ℕ), n > max_cube_edge → ¬(can_construct_cube n) :=
by sorry

#check largest_constructible_cube

end NUMINAMATH_CALUDE_largest_constructible_cube_l478_47899


namespace NUMINAMATH_CALUDE_sin_two_phi_value_l478_47876

theorem sin_two_phi_value (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_phi_value_l478_47876


namespace NUMINAMATH_CALUDE_tan_pi_sevenths_l478_47839

theorem tan_pi_sevenths (y₁ y₂ y₃ : ℝ) 
  (h : y₁^3 - 21*y₁^2 + 35*y₁ - 7 = 0 ∧ 
       y₂^3 - 21*y₂^2 + 35*y₂ - 7 = 0 ∧ 
       y₃^3 - 21*y₃^2 + 35*y₃ - 7 = 0) 
  (h₁ : y₁ = Real.tan (π/7)^2) 
  (h₂ : y₂ = Real.tan (2*π/7)^2) 
  (h₃ : y₃ = Real.tan (3*π/7)^2) : 
  Real.tan (π/7) * Real.tan (2*π/7) * Real.tan (3*π/7) = Real.sqrt 7 ∧
  Real.tan (π/7)^2 + Real.tan (2*π/7)^2 + Real.tan (3*π/7)^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_tan_pi_sevenths_l478_47839


namespace NUMINAMATH_CALUDE_line_circle_intersection_l478_47869

theorem line_circle_intersection (r : ℝ) (A B : ℝ × ℝ) (h_r : r > 0) : 
  (∀ (x y : ℝ), 3*x - 4*y + 5 = 0 → x^2 + y^2 = r^2) →
  (A.1^2 + A.2^2 = r^2) →
  (B.1^2 + B.2^2 = r^2) →
  (3*A.1 - 4*A.2 + 5 = 0) →
  (3*B.1 - 4*B.2 + 5 = 0) →
  (A.1 * B.1 + A.2 * B.2 = -r^2/2) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l478_47869


namespace NUMINAMATH_CALUDE_mollys_age_l478_47862

/-- Molly's birthday candle problem -/
theorem mollys_age (initial_candles additional_candles : ℕ) 
  (h1 : initial_candles = 14)
  (h2 : additional_candles = 6) :
  initial_candles + additional_candles = 20 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_l478_47862


namespace NUMINAMATH_CALUDE_credit_rating_equation_l478_47897

theorem credit_rating_equation (x : ℝ) : 
  (96 : ℝ) = x * (1 + 0.2) ↔ 
  (96 : ℝ) = x + x * 0.2 := by sorry

end NUMINAMATH_CALUDE_credit_rating_equation_l478_47897


namespace NUMINAMATH_CALUDE_jerry_total_games_l478_47888

/-- The total number of video games Jerry has after his birthday -/
def total_games (initial_games birthday_games : ℕ) : ℕ :=
  initial_games + birthday_games

/-- Theorem: Jerry has 9 video games in total -/
theorem jerry_total_games :
  total_games 7 2 = 9 := by sorry

end NUMINAMATH_CALUDE_jerry_total_games_l478_47888


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l478_47802

theorem shopkeeper_profit (cost_price : ℝ) (discount_rate : ℝ) (profit_rate_with_discount : ℝ) 
  (h1 : discount_rate = 0.05)
  (h2 : profit_rate_with_discount = 0.273) :
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := selling_price_with_discount / (1 - discount_rate)
  let profit_rate_without_discount := (marked_price - cost_price) / cost_price
  profit_rate_without_discount = 0.34 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l478_47802


namespace NUMINAMATH_CALUDE_yearly_income_calculation_l478_47854

/-- Calculates the simple interest for a given principal, rate, and time (in years) -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ := 1) : ℚ :=
  principal * rate * time / 100

theorem yearly_income_calculation (totalAmount : ℚ) (part1 : ℚ) (rate1 : ℚ) (rate2 : ℚ) 
  (h1 : totalAmount = 2600)
  (h2 : part1 = 1600)
  (h3 : rate1 = 5)
  (h4 : rate2 = 6) :
  simpleInterest part1 rate1 + simpleInterest (totalAmount - part1) rate2 = 140 := by
  sorry

#eval simpleInterest 1600 5 + simpleInterest 1000 6

end NUMINAMATH_CALUDE_yearly_income_calculation_l478_47854


namespace NUMINAMATH_CALUDE_largest_prime_factor_l478_47816

theorem largest_prime_factor : ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (15^3 + 10^4 - 5^5) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (15^3 + 10^4 - 5^5) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l478_47816


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l478_47879

theorem necessary_not_sufficient_condition :
  (∃ a : ℝ, a > 0 ∧ a^2 - 2*a ≥ 0) ∧
  (∀ a : ℝ, a^2 - 2*a < 0 → a > 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l478_47879


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l478_47873

theorem abs_inequality_solution_set (x : ℝ) : 
  |x + 3| - |x - 3| > 3 ↔ x > 3/2 := by
sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l478_47873


namespace NUMINAMATH_CALUDE_no_integer_in_interval_l478_47840

theorem no_integer_in_interval (n : ℕ) : ¬∃ k : ℤ, (n : ℝ) * Real.sqrt 2 - 1 / (3 * (n : ℝ)) < (k : ℝ) ∧ (k : ℝ) < (n : ℝ) * Real.sqrt 2 + 1 / (3 * (n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_in_interval_l478_47840


namespace NUMINAMATH_CALUDE_y_value_when_x_is_8_l478_47808

theorem y_value_when_x_is_8 (k : ℝ) :
  (∀ x, (x : ℝ) > 0 → k * x^(1/3) = 3 * Real.sqrt 2 → x = 64) →
  k * 8^(1/3) = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_when_x_is_8_l478_47808


namespace NUMINAMATH_CALUDE_kiwi_fraction_l478_47833

theorem kiwi_fraction (total : ℕ) (strawberries : ℕ) (h1 : total = 78) (h2 : strawberries = 52) :
  (total - strawberries : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_kiwi_fraction_l478_47833


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l478_47846

theorem sum_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : y = 18.5) : x + y = 46 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l478_47846


namespace NUMINAMATH_CALUDE_binomial_plus_five_l478_47838

theorem binomial_plus_five : Nat.choose 7 4 + 5 = 40 := by sorry

end NUMINAMATH_CALUDE_binomial_plus_five_l478_47838


namespace NUMINAMATH_CALUDE_inequality_solution_set_l478_47877

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x + 1 < a^2 + x}
  (a > 1 → S = {x : ℝ | x < a + 1}) ∧
  (a < 1 → S = {x : ℝ | x > a + 1}) ∧
  (a = 1 → S = ∅) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l478_47877


namespace NUMINAMATH_CALUDE_daytona_beach_shark_sightings_l478_47863

theorem daytona_beach_shark_sightings :
  let cape_may_sightings : ℕ := 7
  let daytona_beach_sightings : ℕ := 3 * cape_may_sightings + 5
  daytona_beach_sightings = 26 :=
by sorry

end NUMINAMATH_CALUDE_daytona_beach_shark_sightings_l478_47863
