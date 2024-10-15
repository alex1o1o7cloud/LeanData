import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3834_383417

theorem min_value_expression (x : ℝ) (h : x > 0) :
  2 + 3 * x + 4 / x ≥ 2 + 4 * Real.sqrt 3 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  2 + 3 * x + 4 / x = 2 + 4 * Real.sqrt 3 ↔ x = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3834_383417


namespace NUMINAMATH_CALUDE_top_book_cost_l3834_383469

/-- The cost of the "TOP" book -/
def top_cost : ℚ := 8

/-- The cost of the "ABC" book -/
def abc_cost : ℚ := 23

/-- The number of "TOP" books sold -/
def top_sold : ℕ := 13

/-- The number of "ABC" books sold -/
def abc_sold : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books -/
def earnings_difference : ℚ := 12

theorem top_book_cost :
  top_cost * top_sold - abc_cost * abc_sold = earnings_difference :=
sorry

end NUMINAMATH_CALUDE_top_book_cost_l3834_383469


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3834_383430

theorem sin_2alpha_value (α : Real) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3834_383430


namespace NUMINAMATH_CALUDE_graph_is_two_intersecting_lines_l3834_383491

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop := x^2 - y^2 = 0

/-- Definition of two intersecting lines -/
def two_intersecting_lines (f : ℝ → ℝ → Prop) : Prop :=
  ∃ g h : ℝ → ℝ, (∀ x y, f x y ↔ (y = g x ∨ y = h x)) ∧
  (∃ x₀, g x₀ ≠ h x₀)

/-- Theorem stating that the graph of x^2 - y^2 = 0 represents two intersecting lines -/
theorem graph_is_two_intersecting_lines :
  two_intersecting_lines graph_equation := by sorry

end NUMINAMATH_CALUDE_graph_is_two_intersecting_lines_l3834_383491


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3834_383454

theorem trigonometric_identity 
  (α β γ : Real) 
  (a b c : Real) 
  (h1 : 0 < α) (h2 : α < π)
  (h3 : 0 < β) (h4 : β < π)
  (h5 : 0 < γ) (h6 : γ < π)
  (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)
  (h10 : b = c * (Real.cos α + Real.cos β * Real.cos γ) / (Real.sin γ)^2)
  (h11 : a = c * (Real.cos β + Real.cos α * Real.cos γ) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3834_383454


namespace NUMINAMATH_CALUDE_bird_tree_stone_ratio_l3834_383498

theorem bird_tree_stone_ratio :
  let num_stones : ℕ := 40
  let num_trees : ℕ := 3 * num_stones
  let num_birds : ℕ := 400
  let combined_trees_stones : ℕ := num_trees + num_stones
  (num_birds : ℚ) / combined_trees_stones = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bird_tree_stone_ratio_l3834_383498


namespace NUMINAMATH_CALUDE_simplify_expression_l3834_383492

theorem simplify_expression : 
  1 / (2 / (Real.sqrt 3 + 2) + 3 / (Real.sqrt 5 - 2)) = (10 + 2 * Real.sqrt 3 - 3 * Real.sqrt 5) / 43 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3834_383492


namespace NUMINAMATH_CALUDE_warehouse_cleaning_time_l3834_383429

def lara_rate : ℚ := 1/4
def chris_rate : ℚ := 1/6
def break_time : ℚ := 2

theorem warehouse_cleaning_time (t : ℚ) : 
  (lara_rate + chris_rate) * (t - break_time) = 1 ↔ 
  t = (1 / (lara_rate + chris_rate)) + break_time :=
by sorry

end NUMINAMATH_CALUDE_warehouse_cleaning_time_l3834_383429


namespace NUMINAMATH_CALUDE_average_monthly_salary_l3834_383485

/-- Calculates the average monthly salary of five employees given their base salaries and bonus/deduction percentages. -/
theorem average_monthly_salary
  (base_A base_B base_C base_D base_E : ℕ)
  (bonus_A bonus_B1 bonus_D bonus_E : ℚ)
  (deduct_B deduct_D deduct_E : ℚ)
  (h_base_A : base_A = 8000)
  (h_base_B : base_B = 5000)
  (h_base_C : base_C = 16000)
  (h_base_D : base_D = 7000)
  (h_base_E : base_E = 9000)
  (h_bonus_A : bonus_A = 5 / 100)
  (h_bonus_B1 : bonus_B1 = 10 / 100)
  (h_deduct_B : deduct_B = 2 / 100)
  (h_bonus_D : bonus_D = 8 / 100)
  (h_deduct_D : deduct_D = 3 / 100)
  (h_bonus_E : bonus_E = 12 / 100)
  (h_deduct_E : deduct_E = 5 / 100) :
  (base_A * (1 + bonus_A) +
   base_B * (1 + bonus_B1 - deduct_B) +
   base_C +
   base_D * (1 + bonus_D - deduct_D) +
   base_E * (1 + bonus_E - deduct_E)) / 5 = 8756 :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_salary_l3834_383485


namespace NUMINAMATH_CALUDE_frank_allowance_proof_l3834_383405

def frank_allowance (initial_amount spent_amount final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount - spent_amount)

theorem frank_allowance_proof :
  frank_allowance 11 3 22 = 14 := by
  sorry

end NUMINAMATH_CALUDE_frank_allowance_proof_l3834_383405


namespace NUMINAMATH_CALUDE_intersection_equals_A_l3834_383446

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | |x| < 2}

-- Theorem statement
theorem intersection_equals_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_intersection_equals_A_l3834_383446


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l3834_383408

theorem correct_mark_calculation (n : ℕ) (initial_avg : ℚ) (wrong_mark : ℚ) (correct_avg : ℚ) :
  n = 10 →
  initial_avg = 100 →
  wrong_mark = 90 →
  correct_avg = 92 →
  ∃ x : ℚ, (n : ℚ) * initial_avg - wrong_mark + x = (n : ℚ) * correct_avg ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l3834_383408


namespace NUMINAMATH_CALUDE_mary_tim_income_difference_l3834_383400

theorem mary_tim_income_difference (juan tim mary : ℝ) 
  (h1 : tim = 0.5 * juan)
  (h2 : mary = 0.8 * juan) :
  (mary - tim) / tim * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_mary_tim_income_difference_l3834_383400


namespace NUMINAMATH_CALUDE_marble_distribution_l3834_383479

theorem marble_distribution (T : ℝ) (C B O : ℝ) : 
  T > 0 →
  C = 0.40 * T →
  O = (2/5) * T →
  C + B + O = T →
  B = 0.20 * T :=
by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3834_383479


namespace NUMINAMATH_CALUDE_sin_monotone_interval_l3834_383411

theorem sin_monotone_interval (t : ℝ) : 
  (∀ x ∈ Set.Icc (-t) t, StrictMono (fun x ↦ Real.sin (2 * x + π / 6))) ↔ 
  t ∈ Set.Ioo 0 (π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sin_monotone_interval_l3834_383411


namespace NUMINAMATH_CALUDE_frog_eggs_first_day_l3834_383475

/-- Represents the number of eggs laid by a frog over 4 days -/
def frog_eggs (x : ℕ) : ℕ :=
  let day1 := x
  let day2 := 2 * x
  let day3 := 2 * x + 20
  let day4 := 2 * (day1 + day2 + day3)
  day1 + day2 + day3 + day4

/-- Theorem stating that if the frog lays 810 eggs over 4 days following the given pattern,
    then it laid 50 eggs on the first day -/
theorem frog_eggs_first_day :
  ∃ (x : ℕ), frog_eggs x = 810 ∧ x = 50 :=
sorry

end NUMINAMATH_CALUDE_frog_eggs_first_day_l3834_383475


namespace NUMINAMATH_CALUDE_trigonometric_identities_trigonometric_value_l3834_383489

theorem trigonometric_identities (α : Real) :
  (Real.tan (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α - Real.pi) * Real.sin (-Real.pi + α) * Real.cos (α + 5 * Real.pi / 2)) = -1 / Real.sin α :=
by sorry

theorem trigonometric_value (α : Real) (h : Real.tan α = 1/4) :
  1 / (2 * Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α) = 17/20 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_trigonometric_value_l3834_383489


namespace NUMINAMATH_CALUDE_ac_average_usage_time_l3834_383404

/-- Calculates the average usage time for each air conditioner -/
def averageUsageTime (totalAC : ℕ) (maxSimultaneous : ℕ) (hoursPerDay : ℕ) : ℚ :=
  (maxSimultaneous * hoursPerDay : ℚ) / totalAC

/-- Proves that the average usage time for each air conditioner is 20 hours -/
theorem ac_average_usage_time :
  let totalAC : ℕ := 6
  let maxSimultaneous : ℕ := 5
  let hoursPerDay : ℕ := 24
  averageUsageTime totalAC maxSimultaneous hoursPerDay = 20 := by
sorry

end NUMINAMATH_CALUDE_ac_average_usage_time_l3834_383404


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3834_383487

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (Complex.I - 1) / Complex.I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3834_383487


namespace NUMINAMATH_CALUDE_words_with_consonant_count_l3834_383412

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := alphabet \ consonants

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def vowel_only_words : Nat := vowels.card ^ word_length

theorem words_with_consonant_count :
  total_words - vowel_only_words = 7744 :=
sorry

end NUMINAMATH_CALUDE_words_with_consonant_count_l3834_383412


namespace NUMINAMATH_CALUDE_function_inequality_l3834_383426

/-- A function satisfying the given conditions -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, (x * (deriv f x) - f x) ≤ 0

theorem function_inequality
  (f : ℝ → ℝ)
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : Differentiable ℝ f)
  (h_cond : SatisfiesCondition f)
  (m n : ℝ)
  (h_pos_m : m > 0)
  (h_pos_n : n > 0)
  (h_lt : m < n) :
  m * f n ≤ n * f m :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l3834_383426


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3834_383494

/-- Ellipse C₁ -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Parabola C₂ -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 4 * p.1}

/-- Line with slope and y-intercept -/
structure Line where
  k : ℝ
  m : ℝ

/-- Tangent line to both ellipse and parabola -/
def isTangentLine (l : Line) (e : Ellipse) : Prop :=
  ∃ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 ∧
                y = l.k * x + l.m ∧
                (l.k * x + l.m)^2 = 4 * x

theorem tangent_line_equation (e : Ellipse) 
  (h1 : e.a^2 - e.b^2 = e.a^2 / 2)  -- Eccentricity condition
  (h2 : e.a - (e.a^2 - e.b^2).sqrt = Real.sqrt 2 - 1)  -- Minimum distance condition
  : ∃ (l : Line), isTangentLine l e ∧ 
    ((l.k = Real.sqrt 2 / 2 ∧ l.m = Real.sqrt 2) ∨
     (l.k = -Real.sqrt 2 / 2 ∧ l.m = -Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3834_383494


namespace NUMINAMATH_CALUDE_norris_savings_l3834_383437

/-- The amount of money Norris saved in November -/
def november_savings : ℤ := sorry

/-- The amount of money Norris saved in September -/
def september_savings : ℤ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℤ := 25

/-- The amount of money Norris spent on an online game -/
def online_game_cost : ℤ := 75

/-- The amount of money Norris has left -/
def money_left : ℤ := 10

theorem norris_savings : november_savings = 31 := by
  sorry

end NUMINAMATH_CALUDE_norris_savings_l3834_383437


namespace NUMINAMATH_CALUDE_price_of_short_is_13_50_l3834_383466

/-- The price of a single short, given the conditions of Jimmy and Irene's shopping trip -/
def price_of_short (num_shorts : ℕ) (num_shirts : ℕ) (shirt_price : ℚ) 
  (discount_rate : ℚ) (total_paid : ℚ) : ℚ :=
  let shirt_total := num_shirts * shirt_price
  let discounted_shirt_total := shirt_total * (1 - discount_rate)
  let shorts_total := total_paid - discounted_shirt_total
  shorts_total / num_shorts

/-- Theorem stating that the price of each short is $13.50 under the given conditions -/
theorem price_of_short_is_13_50 :
  price_of_short 3 5 17 (1/10) 117 = 27/2 := by sorry

end NUMINAMATH_CALUDE_price_of_short_is_13_50_l3834_383466


namespace NUMINAMATH_CALUDE_decimal_to_binary_119_l3834_383450

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 119

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, true, false, true, true, true]

/-- Theorem stating that the binary representation of 119 is [1,1,1,0,1,1,1] -/
theorem decimal_to_binary_119 : toBinary decimalNumber = expectedBinary := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_119_l3834_383450


namespace NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l3834_383424

theorem tan_beta_minus_2alpha (α β : ℝ) 
  (h1 : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3)
  (h2 : Real.tan (α - β) = 2) : 
  Real.tan (β - 2*α) = 4/3 := by sorry

end NUMINAMATH_CALUDE_tan_beta_minus_2alpha_l3834_383424


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3834_383480

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x - y) / (2*x + 3*y) + (2*x + 3*y) / (x - y) = 2) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 34/15 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3834_383480


namespace NUMINAMATH_CALUDE_specific_trapezoid_height_l3834_383493

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  c : ℝ  -- Length of one non-parallel side
  d : ℝ  -- Length of the other non-parallel side

/-- The height of a trapezoid -/
def trapezoid_height (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with the given dimensions has a height of 12 -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { a := 25, b := 4, c := 20, d := 13 }
  trapezoid_height t = 12 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_height_l3834_383493


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l3834_383481

theorem unique_prime_with_prime_sums : ∀ p : ℕ, 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) → p = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l3834_383481


namespace NUMINAMATH_CALUDE_least_distinct_values_in_list_l3834_383488

theorem least_distinct_values_in_list (list : List ℕ) : 
  list.length = 2030 →
  ∃! m, m ∈ list ∧ (list.count m = 11) ∧ (∀ n ∈ list, n ≠ m → list.count n < 11) →
  (∃ x : ℕ, x = list.toFinset.card ∧ x ≥ 203 ∧ ∀ y : ℕ, y = list.toFinset.card → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_in_list_l3834_383488


namespace NUMINAMATH_CALUDE_hyperbola_and_chord_equation_l3834_383490

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 18 + y^2 / 14 = 1

-- Define the common focal point condition
def common_focal_point (C : (ℝ → ℝ → Prop)) (E : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C x y ∧ E x y

-- Define point A on hyperbola C
def point_A_on_C (C : (ℝ → ℝ → Prop)) : Prop :=
  C 3 (Real.sqrt 7)

-- Define point P as midpoint of chord AB
def point_P_midpoint (C : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ 1 = (x₁ + x₂) / 2 ∧ 2 = (y₁ + y₂) / 2

-- Main theorem
theorem hyperbola_and_chord_equation :
  ∀ (a b : ℝ),
    (hyperbola_C 3 (Real.sqrt 7) a b) →
    (common_focal_point (hyperbola_C · · a b) ellipse) →
    (point_A_on_C (hyperbola_C · · a b)) →
    (point_P_midpoint (hyperbola_C · · a b)) →
    (∀ (x y : ℝ), hyperbola_C x y a b ↔ x^2 / 2 - y^2 / 2 = 1) ∧
    (∃ (m c : ℝ), ∀ (x y : ℝ), (hyperbola_C x y a b ∧ y = m * x + c) → x - 2 * y + 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_chord_equation_l3834_383490


namespace NUMINAMATH_CALUDE_class_size_proof_l3834_383465

theorem class_size_proof (total : ℕ) 
  (h1 : 20 < total ∧ total < 30)
  (h2 : ∃ male : ℕ, total = 3 * male)
  (h3 : ∃ registered unregistered : ℕ, 
    registered + unregistered = total ∧ 
    registered = 3 * unregistered - 1) :
  total = 27 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l3834_383465


namespace NUMINAMATH_CALUDE_max_d_value_l3834_383427

def a (n : ℕ) : ℕ := 99 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (M : ℕ), M = 401 ∧ ∀ (n : ℕ), n > 0 → d n ≤ M ∧ ∃ (k : ℕ), k > 0 ∧ d k = M :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3834_383427


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l3834_383477

theorem dvd_rental_cost (num_dvds : ℕ) (cost_per_dvd : ℚ) (total_cost : ℚ) :
  num_dvds = 4 →
  cost_per_dvd = 6/5 →
  total_cost = num_dvds * cost_per_dvd →
  total_cost = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l3834_383477


namespace NUMINAMATH_CALUDE_bus_capacity_is_90_l3834_383472

/-- The number of people that can sit in a bus with given seat arrangements -/
def bus_capacity (left_seats : ℕ) (right_seat_difference : ℕ) (people_per_seat : ℕ) (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seat_difference
  let total_regular_seats := left_seats + right_seats
  let total_regular_capacity := total_regular_seats * people_per_seat
  total_regular_capacity + back_seat_capacity

/-- Theorem stating that the bus capacity is 90 given the specific conditions -/
theorem bus_capacity_is_90 : 
  bus_capacity 15 3 3 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_is_90_l3834_383472


namespace NUMINAMATH_CALUDE_book_cost_problem_l3834_383414

theorem book_cost_problem (cost_loss : ℝ) (sell_price : ℝ) :
  cost_loss = 262.5 →
  sell_price = cost_loss * 0.85 →
  sell_price = (sell_price / 1.19) * 1.19 →
  cost_loss + (sell_price / 1.19) = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l3834_383414


namespace NUMINAMATH_CALUDE_correct_number_of_groups_l3834_383441

/-- The number of different groups of 3 marbles Tom can choose -/
def number_of_groups : ℕ := 16

/-- The number of red marbles Tom has -/
def red_marbles : ℕ := 1

/-- The number of blue marbles Tom has -/
def blue_marbles : ℕ := 1

/-- The number of black marbles Tom has -/
def black_marbles : ℕ := 1

/-- The number of white marbles Tom has -/
def white_marbles : ℕ := 4

/-- The total number of marbles Tom has -/
def total_marbles : ℕ := red_marbles + blue_marbles + black_marbles + white_marbles

/-- Theorem stating that the number of different groups of 3 marbles Tom can choose is correct -/
theorem correct_number_of_groups :
  number_of_groups = (Nat.choose white_marbles 3) + (Nat.choose 3 2 * Nat.choose white_marbles 1) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_groups_l3834_383441


namespace NUMINAMATH_CALUDE_jones_elementary_population_l3834_383433

theorem jones_elementary_population :
  let total_students : ℕ := 225
  let boys_percentage : ℚ := 40 / 100
  let boys_count : ℕ := 90
  (boys_count : ℚ) / (total_students * boys_percentage) = 1 :=
by sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l3834_383433


namespace NUMINAMATH_CALUDE_next_coincidence_correct_l3834_383496

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Checks if hour and minute hands coincide at given time -/
def handsCoincide (t : Time) : Prop :=
  (t.hours % 12 * 60 + t.minutes) * 11 = t.minutes * 12

/-- The next time after midnight when clock hands coincide -/
def nextCoincidence : Time :=
  { hours := 1, minutes := 5, seconds := 27 }

theorem next_coincidence_correct :
  handsCoincide nextCoincidence ∧
  ∀ t : Time, t.toSeconds < nextCoincidence.toSeconds → ¬handsCoincide t :=
by sorry

end NUMINAMATH_CALUDE_next_coincidence_correct_l3834_383496


namespace NUMINAMATH_CALUDE_max_distinct_squares_sum_l3834_383482

/-- The sum of squares of the first n positive integers -/
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- A function that checks if there exists a set of n distinct positive integers
    whose squares sum to 2531 -/
def exists_distinct_squares_sum (n : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card = n ∧ (∀ x ∈ s, x > 0) ∧ (s.sum (λ x => x^2) = 2531)

theorem max_distinct_squares_sum :
  (∃ n : ℕ, exists_distinct_squares_sum n ∧
    ∀ m : ℕ, m > n → ¬exists_distinct_squares_sum m) ∧
  (∃ n : ℕ, exists_distinct_squares_sum n ∧ n = 18) := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_squares_sum_l3834_383482


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l3834_383436

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- The octal representation of the spaceship's age --/
def spaceship_age_octal : ℕ := 367

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 247 := by sorry

end NUMINAMATH_CALUDE_spaceship_age_conversion_l3834_383436


namespace NUMINAMATH_CALUDE_true_proposition_l3834_383473

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Proposition p: There exists a φ ∈ ℝ such that f(x) = sin(x + φ) is an even function -/
def p : Prop := ∃ φ : ℝ, IsEven (fun x ↦ Real.sin (x + φ))

/-- Proposition q: For all x ∈ ℝ, cos(2x) + 4sin(x) - 3 < 0 -/
def q : Prop := ∀ x : ℝ, Real.cos (2 * x) + 4 * Real.sin x - 3 < 0

/-- The true proposition is p ∨ (¬q) -/
theorem true_proposition : p ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_true_proposition_l3834_383473


namespace NUMINAMATH_CALUDE_simplify_expression_l3834_383457

theorem simplify_expression (x : ℝ) : x^3 * x^2 * x + (x^3)^2 + (-2*x^2)^3 = -6*x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3834_383457


namespace NUMINAMATH_CALUDE_one_third_to_fifth_power_l3834_383458

theorem one_third_to_fifth_power :
  (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_one_third_to_fifth_power_l3834_383458


namespace NUMINAMATH_CALUDE_stratified_sampling_senior_managers_l3834_383434

theorem stratified_sampling_senior_managers 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (senior_managers : ℕ) 
  (h1 : total_population = 200) 
  (h2 : sample_size = 40) 
  (h3 : senior_managers = 10) :
  (sample_size : ℚ) / total_population * senior_managers = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_senior_managers_l3834_383434


namespace NUMINAMATH_CALUDE_existence_of_special_function_l3834_383409

theorem existence_of_special_function :
  ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f (f n) = 1993 * n^1945 :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_function_l3834_383409


namespace NUMINAMATH_CALUDE_nancy_scholarship_amount_l3834_383495

/-- Proves that Nancy's scholarship amount is $3,000 given the tuition costs and other conditions --/
theorem nancy_scholarship_amount : 
  ∀ (tuition : ℕ) 
    (parent_contribution : ℕ) 
    (work_hours : ℕ) 
    (hourly_rate : ℕ) 
    (scholarship : ℕ),
  tuition = 22000 →
  parent_contribution = tuition / 2 →
  work_hours = 200 →
  hourly_rate = 10 →
  scholarship + 2 * scholarship + parent_contribution + work_hours * hourly_rate = tuition →
  scholarship = 3000 := by
sorry


end NUMINAMATH_CALUDE_nancy_scholarship_amount_l3834_383495


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3834_383428

/-- A parallelogram with vertices A, B, C, and D in a real inner product space. -/
structure Parallelogram (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)
  (is_parallelogram : A - B = D - C)

/-- The theorem stating that if BD = 2 and 2(AD • AB) = |BC|^2 in a parallelogram ABCD,
    then |AB| = 2. -/
theorem parallelogram_side_length
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (para : Parallelogram V)
  (h1 : ‖para.B - para.D‖ = 2)
  (h2 : 2 * inner (para.A - para.D) (para.A - para.B) = ‖para.B - para.C‖^2) :
  ‖para.A - para.B‖ = 2 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3834_383428


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3834_383476

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Given conditions for the geometric sequence -/
def SequenceConditions (a : ℕ → ℝ) : Prop :=
  GeometricSequence a ∧ 
  a 2 * a 3 = 2 * a 1 ∧ 
  (a 4 + 2 * a 7) / 2 = 5 / 4

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  SequenceConditions a → a 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3834_383476


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3834_383453

theorem quadratic_equation_root (k l m : ℝ) :
  (2 * (k - l) * 2^2 + 3 * (l - m) * 2 + 4 * (m - k) = 0) →
  (∃ x : ℝ, 2 * (k - l) * x^2 + 3 * (l - m) * x + 4 * (m - k) = 0 ∧ x ≠ 2) →
  (∃ x : ℝ, 2 * (k - l) * x^2 + 3 * (l - m) * x + 4 * (m - k) = 0 ∧ x = (m - k) / (k - l)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3834_383453


namespace NUMINAMATH_CALUDE_particles_tend_to_unit_circle_l3834_383461

/-- Velocity field of the fluid -/
def velocity_field (x y : ℝ) : ℝ × ℝ :=
  (y + 2*x - 2*x^3 - 2*x*y^2, -x)

/-- The rate of change of r^2 with respect to t -/
def r_squared_derivative (x y : ℝ) : ℝ :=
  2*x*(y + 2*x - 2*x^3 - 2*x*y^2) + 2*y*(-x)

/-- Theorem stating that particles tend towards the unit circle as t → ∞ -/
theorem particles_tend_to_unit_circle :
  ∀ (x y : ℝ), x ≠ 0 →
  (r_squared_derivative x y > 0 ↔ x^2 + y^2 < 1) ∧
  (r_squared_derivative x y < 0 ↔ x^2 + y^2 > 1) :=
sorry

end NUMINAMATH_CALUDE_particles_tend_to_unit_circle_l3834_383461


namespace NUMINAMATH_CALUDE_eighteen_cubed_times_nine_cubed_l3834_383435

theorem eighteen_cubed_times_nine_cubed (L M : ℕ) : 18^3 * 9^3 = 2^3 * 3^12 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_cubed_times_nine_cubed_l3834_383435


namespace NUMINAMATH_CALUDE_smallest_multiple_five_satisfies_five_is_smallest_l3834_383407

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 625 ∣ (x * 500) → x ≥ 5 := by
  sorry

theorem five_satisfies : 625 ∣ (5 * 500) := by
  sorry

theorem five_is_smallest : ∀ (x : ℕ), x > 0 ∧ 625 ∣ (x * 500) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_five_satisfies_five_is_smallest_l3834_383407


namespace NUMINAMATH_CALUDE_pizza_fraction_proof_l3834_383449

theorem pizza_fraction_proof (total_slices : ℕ) (whole_slices_eaten : ℕ) (shared_slice_fraction : ℚ) :
  total_slices = 16 →
  whole_slices_eaten = 2 →
  shared_slice_fraction = 1/3 →
  (whole_slices_eaten : ℚ) / total_slices + shared_slice_fraction / total_slices = 7/48 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_proof_l3834_383449


namespace NUMINAMATH_CALUDE_triangle_dot_product_l3834_383455

/-- Given a triangle ABC with area √3 and angle A = π/3, 
    the dot product of vectors AB and AC is equal to 2. -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let S := Real.sqrt 3
  let angleA := π / 3
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let area := Real.sqrt 3
  area = Real.sqrt 3 ∧ 
  angleA = π / 3 →
  AB.1 * AC.1 + AB.2 * AC.2 = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l3834_383455


namespace NUMINAMATH_CALUDE_locus_is_circle_l3834_383425

/-- The locus of points satisfying the given condition is a circle -/
theorem locus_is_circle (k : ℝ) (h : k > 0) :
  ∀ (x y : ℝ), (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x / a + y / b = 1 ∧ 1 / a^2 + 1 / b^2 = 1 / k^2) 
  ↔ x^2 + y^2 = k^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l3834_383425


namespace NUMINAMATH_CALUDE_x_value_l3834_383463

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 :=
by sorry

end NUMINAMATH_CALUDE_x_value_l3834_383463


namespace NUMINAMATH_CALUDE_min_value_of_f_l3834_383416

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then |x + a| + |x - 1| else x^2 - a*x + 2

theorem min_value_of_f (a : ℝ) : 
  (∀ x, f a x ≥ a) ∧ (∃ x, f a x = a) ↔ a ∈ ({-2 - 2*Real.sqrt 3, 2} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3834_383416


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3834_383413

theorem quadratic_roots_condition (p q r : ℝ) : 
  (p^4 * (q - r)^2 + 2 * p^2 * (q + r) + 1 = p^4) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    (x₁^2 + p*x₁ + q = 0) ∧ 
    (x₂^2 + p*x₂ + q = 0) ∧ 
    (y₁^2 - p*y₁ + r = 0) ∧ 
    (y₂^2 - p*y₂ + r = 0) ∧ 
    (x₁*y₁ - x₂*y₂ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3834_383413


namespace NUMINAMATH_CALUDE_milk_quantity_proof_l3834_383447

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1216

/-- The fraction of milk in container B compared to A --/
def b_fraction : ℝ := 0.375

/-- The amount transferred between containers --/
def transfer_amount : ℝ := 152

/-- Theorem stating the initial quantity of milk in container A --/
theorem milk_quantity_proof :
  ∃ (a b c : ℝ),
    a = initial_quantity ∧
    b = b_fraction * a ∧
    c = a - b ∧
    b + transfer_amount = c - transfer_amount :=
by sorry

end NUMINAMATH_CALUDE_milk_quantity_proof_l3834_383447


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3834_383402

open Real

theorem trigonometric_equation_solutions (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    sin (5 * a) * cos x₁ - cos (x₁ + 4 * a) = 0 ∧
    sin (5 * a) * cos x₂ - cos (x₂ + 4 * a) = 0 ∧
    ¬ ∃ k : ℤ, x₁ - x₂ = π * (k : ℝ)) ↔
  ∃ t : ℤ, a = π * ((4 * t + 1 : ℤ) : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3834_383402


namespace NUMINAMATH_CALUDE_first_subject_grade_l3834_383484

/-- 
Given a student's grades in three subjects, prove that if the second subject is 60%,
the third subject is 70%, and the overall average is 60%, then the first subject's grade must be 50%.
-/
theorem first_subject_grade (grade1 : ℝ) (grade2 grade3 overall : ℝ) : 
  grade2 = 60 → grade3 = 70 → overall = 60 → (grade1 + grade2 + grade3) / 3 = overall → grade1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_subject_grade_l3834_383484


namespace NUMINAMATH_CALUDE_angle_equality_l3834_383456

theorem angle_equality (θ : Real) (h1 : Real.sqrt 5 * Real.sin (15 * π / 180) = Real.cos θ + Real.sin θ) 
  (h2 : 0 < θ ∧ θ < π / 2) : θ = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3834_383456


namespace NUMINAMATH_CALUDE_bahs_equal_to_500_yahs_l3834_383415

-- Define the conversion rates
def bah_to_rah : ℚ := 30 / 20
def rah_to_yah : ℚ := 25 / 10

-- Define the target number of yahs
def target_yahs : ℕ := 500

-- Theorem statement
theorem bahs_equal_to_500_yahs :
  ⌊(target_yahs : ℚ) / (rah_to_yah * bah_to_rah)⌋ = 133 := by
  sorry

end NUMINAMATH_CALUDE_bahs_equal_to_500_yahs_l3834_383415


namespace NUMINAMATH_CALUDE_chromatic_number_upper_bound_l3834_383499

-- Define a graph type
structure Graph :=
  (V : Type) -- Vertex set
  (E : V → V → Prop) -- Edge relation

-- Define the number of edges in a graph
def num_edges (G : Graph) : ℕ := sorry

-- Define the chromatic number of a graph
def chromatic_number (G : Graph) : ℕ := sorry

-- State the theorem
theorem chromatic_number_upper_bound (G : Graph) :
  chromatic_number G ≤ (1/2 : ℝ) + Real.sqrt (2 * (num_edges G : ℝ) + 1/4) := by
  sorry

end NUMINAMATH_CALUDE_chromatic_number_upper_bound_l3834_383499


namespace NUMINAMATH_CALUDE_photos_per_album_l3834_383468

theorem photos_per_album 
  (total_photos : ℕ) 
  (num_albums : ℕ) 
  (h1 : total_photos = 2560) 
  (h2 : num_albums = 32) 
  (h3 : total_photos % num_albums = 0) :
  total_photos / num_albums = 80 := by
sorry

end NUMINAMATH_CALUDE_photos_per_album_l3834_383468


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3834_383459

/-- Definition of a point in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

/-- The point (3, -2) -/
def point : ℝ × ℝ := (3, -2)

/-- Theorem: The point (3, -2) is in the fourth quadrant -/
theorem point_in_fourth_quadrant : 
  in_fourth_quadrant point.1 point.2 := by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3834_383459


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l3834_383471

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4761 → 
  min a b = 53 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l3834_383471


namespace NUMINAMATH_CALUDE_sugar_needed_is_six_l3834_383486

/-- Represents the ratios in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of sugar needed in the new recipe --/
def sugar_needed (original : RecipeRatio) (water_new : ℚ) : ℚ :=
  let flour_water_ratio_new := 2 * (original.flour / original.water)
  let flour_sugar_ratio_new := (original.flour / original.sugar) / 2
  let flour_new := flour_water_ratio_new * water_new
  flour_new / flour_sugar_ratio_new

/-- Theorem stating that the amount of sugar needed is 6 cups --/
theorem sugar_needed_is_six :
  let original := RecipeRatio.mk 8 4 3
  let water_new := 2
  sugar_needed original water_new = 6 := by
  sorry

#eval sugar_needed (RecipeRatio.mk 8 4 3) 2

end NUMINAMATH_CALUDE_sugar_needed_is_six_l3834_383486


namespace NUMINAMATH_CALUDE_decimal_addition_l3834_383440

theorem decimal_addition : (5.467 : ℝ) + 3.92 = 9.387 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l3834_383440


namespace NUMINAMATH_CALUDE_carlson_problem_max_candies_l3834_383460

/-- The maximum number of candies that can be eaten in the Carlson problem -/
def max_candies (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating the maximum number of candies for 32 initial ones -/
theorem carlson_problem_max_candies :
  max_candies 32 = 496 := by
  sorry

#eval max_candies 32  -- Should output 496

end NUMINAMATH_CALUDE_carlson_problem_max_candies_l3834_383460


namespace NUMINAMATH_CALUDE_brett_travel_distance_l3834_383432

/-- The distance traveled given a constant speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Brett's travel distance in 12 hours at 75 miles per hour is 900 miles -/
theorem brett_travel_distance : distance_traveled 75 12 = 900 := by
  sorry

end NUMINAMATH_CALUDE_brett_travel_distance_l3834_383432


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l3834_383483

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section from its equation -/
def determineConicType (f : ℝ → ℝ → ℝ) : ConicType :=
  sorry

/-- The equation of the conic section -/
def conicEquation (x y : ℝ) : ℝ :=
  (x - 3)^2 - 2*(y + 1)^2 - 50

theorem conic_is_hyperbola :
  determineConicType conicEquation = ConicType.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l3834_383483


namespace NUMINAMATH_CALUDE_root_implies_h_value_l3834_383401

theorem root_implies_h_value (h : ℝ) :
  ((-3 : ℝ)^3 + h * (-3) - 10 = 0) → h = -37/3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l3834_383401


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3834_383419

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + x^2 + 15) - (x^6 + x^5 - 2 * x^4 + 3 * x^2 + 20) =
  x^6 + 2 * x^5 + 3 * x^4 - 2 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3834_383419


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3834_383410

/-- Given a boat that travels 11 km/hr along a stream and 7 km/hr against the same stream,
    its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 11 →
    boat_speed - stream_speed = 7 →
    boat_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3834_383410


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3834_383452

/-- Represents the number of items of each product type in a sample -/
structure SampleCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total sample size -/
def totalSampleSize (s : SampleCounts) : ℕ :=
  s.typeA + s.typeB + s.typeC

/-- Represents the production ratio of the three product types -/
def productionRatio : Fin 3 → ℕ
| 0 => 1  -- Type A
| 1 => 3  -- Type B
| 2 => 5  -- Type C

theorem stratified_sample_size 
  (s : SampleCounts) 
  (h_ratio : s.typeA * productionRatio 1 = s.typeB * productionRatio 0 ∧ 
             s.typeB * productionRatio 2 = s.typeC * productionRatio 1) 
  (h_typeB : s.typeB = 12) : 
  totalSampleSize s = 36 := by
sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l3834_383452


namespace NUMINAMATH_CALUDE_smallest_positive_resolvable_debt_is_40_l3834_383420

/-- The value of a pig in dollars -/
def pig_value : ℕ := 280

/-- The value of a goat in dollars -/
def goat_value : ℕ := 200

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℤ) : Prop :=
  ∃ (p g : ℤ), debt = p * pig_value + g * goat_value

/-- The smallest positive resolvable debt -/
def smallest_positive_resolvable_debt : ℕ := 40

theorem smallest_positive_resolvable_debt_is_40 :
  (∀ d : ℕ, d < smallest_positive_resolvable_debt → ¬is_resolvable d) ∧
  is_resolvable smallest_positive_resolvable_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_resolvable_debt_is_40_l3834_383420


namespace NUMINAMATH_CALUDE_letter_R_in_13th_space_l3834_383418

/-- The space number where the letter R should be placed on a sign -/
def letter_R_position (total_spaces : ℕ) (word_length : ℕ) : ℕ :=
  (total_spaces - word_length) / 2 + 1

/-- Theorem stating that the letter R should be in the 13th space -/
theorem letter_R_in_13th_space :
  letter_R_position 31 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_letter_R_in_13th_space_l3834_383418


namespace NUMINAMATH_CALUDE_base_of_exponent_l3834_383439

theorem base_of_exponent (a : ℝ) (x : ℝ) (h1 : a^(2*x + 2) = 16^(3*x - 1)) (h2 : x = 1) : a = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_of_exponent_l3834_383439


namespace NUMINAMATH_CALUDE_first_day_exceeding_threshold_l3834_383474

def initial_bacteria : ℕ := 5
def growth_rate : ℕ := 3
def threshold : ℕ := 200

def bacteria_count (n : ℕ) : ℕ := initial_bacteria * growth_rate ^ n

theorem first_day_exceeding_threshold :
  ∃ n : ℕ, bacteria_count n > threshold ∧ ∀ m : ℕ, m < n → bacteria_count m ≤ threshold :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_threshold_l3834_383474


namespace NUMINAMATH_CALUDE_postcard_probability_l3834_383478

/-- The probability of arranging n unique items in a line, such that k specific items are consecutive. -/
def consecutive_probability (n k : ℕ) : ℚ :=
  if k ≤ n ∧ k > 0 then
    (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n
  else
    0

/-- Theorem: The probability of arranging 12 unique postcards in a line, 
    such that 4 specific postcards are consecutive, is 1/55. -/
theorem postcard_probability : consecutive_probability 12 4 = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_postcard_probability_l3834_383478


namespace NUMINAMATH_CALUDE_students_without_A_l3834_383470

theorem students_without_A (total : ℕ) (lit_A : ℕ) (sci_A : ℕ) (both_A : ℕ) : 
  total = 35 → lit_A = 10 → sci_A = 15 → both_A = 5 → 
  total - (lit_A + sci_A - both_A) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l3834_383470


namespace NUMINAMATH_CALUDE_sean_has_more_whistles_l3834_383462

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := 13

/-- The difference in whistle count between Sean and Charles -/
def whistle_difference : ℕ := sean_whistles - charles_whistles

theorem sean_has_more_whistles : whistle_difference = 32 := by
  sorry

end NUMINAMATH_CALUDE_sean_has_more_whistles_l3834_383462


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3834_383422

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  sum_property : a 1 + a 3 = 8
  geometric_mean : (a 4) ^ 2 = (a 2) * (a 9)
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) : seq.a 5 = 13 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3834_383422


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3834_383421

def A : Set Int := {-1, 0, 1, 3, 5}
def B : Set Int := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3834_383421


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l3834_383464

/-- Given a sequence {aₙ} where Sₙ (the sum of the first n terms) is defined as Sₙ = n² + 1,
    prove that the 5th term of the sequence (a₅) is equal to 9. -/
theorem fifth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = n^2 + 1) : a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l3834_383464


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3834_383443

theorem cricket_team_age_difference :
  let team_size : ℕ := 11
  let captain_age : ℕ := 25
  let team_average_age : ℕ := 22
  let remaining_average_age : ℕ := team_average_age - 1
  let wicket_keeper_age := captain_age + x

  (team_size * team_average_age = 
   (team_size - 2) * remaining_average_age + captain_age + wicket_keeper_age) →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3834_383443


namespace NUMINAMATH_CALUDE_flagstaff_shadow_length_l3834_383444

/-- Given a flagstaff and a building casting shadows under similar conditions,
    this theorem proves the length of the flagstaff's shadow. -/
theorem flagstaff_shadow_length
  (h_flagstaff : ℝ)
  (h_building : ℝ)
  (s_building : ℝ)
  (h_flagstaff_pos : h_flagstaff > 0)
  (h_building_pos : h_building > 0)
  (s_building_pos : s_building > 0)
  (h_flagstaff_val : h_flagstaff = 17.5)
  (h_building_val : h_building = 12.5)
  (s_building_val : s_building = 28.75) :
  ∃ s_flagstaff : ℝ, s_flagstaff = 40.15 ∧ h_flagstaff / s_flagstaff = h_building / s_building :=
by
  sorry


end NUMINAMATH_CALUDE_flagstaff_shadow_length_l3834_383444


namespace NUMINAMATH_CALUDE_exists_number_satisfying_equation_l3834_383431

theorem exists_number_satisfying_equation : ∃ x : ℝ, (3.241 * x) / 100 = 0.045374000000000005 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_equation_l3834_383431


namespace NUMINAMATH_CALUDE_unique_square_difference_l3834_383445

theorem unique_square_difference (n : ℕ) : 
  (∃ k m : ℕ, n + 30 = k^2 ∧ n - 17 = m^2) ↔ n = 546 := by
sorry

end NUMINAMATH_CALUDE_unique_square_difference_l3834_383445


namespace NUMINAMATH_CALUDE_friends_bread_slices_l3834_383467

/-- Calculates the number of slices each friend eats given the number of friends and the slices in each loaf -/
def slices_per_friend (n : ℕ) (loaf1 loaf2 loaf3 loaf4 : ℕ) : ℕ :=
  (loaf1 + loaf2 + loaf3 + loaf4)

/-- Theorem stating that each friend eats 78 slices of bread -/
theorem friends_bread_slices (n : ℕ) (h : n > 0) :
  slices_per_friend n 15 18 20 25 = 78 := by
  sorry

#check friends_bread_slices

end NUMINAMATH_CALUDE_friends_bread_slices_l3834_383467


namespace NUMINAMATH_CALUDE_solve_work_problem_l3834_383451

def work_problem (a_days b_days : ℕ) (b_share : ℚ) : Prop :=
  a_days > 0 ∧ b_days > 0 ∧ b_share > 0 →
  let a_rate : ℚ := 1 / a_days
  let b_rate : ℚ := 1 / b_days
  let total_rate : ℚ := a_rate + b_rate
  let a_proportion : ℚ := a_rate / total_rate
  let b_proportion : ℚ := b_rate / total_rate
  let total_amount : ℚ := b_share / b_proportion
  total_amount = 1000

theorem solve_work_problem :
  work_problem 30 20 600 := by
  sorry

end NUMINAMATH_CALUDE_solve_work_problem_l3834_383451


namespace NUMINAMATH_CALUDE_correct_notation_of_expression_l3834_383497

/-- Predicate to check if an expression is correctly written in standard algebraic notation -/
def is_correct_notation : Set ℝ → Prop :=
  sorry

/-- The specific expression we're checking -/
def expression : Set ℝ := {x | ∃ y, y = |4| / 3 ∧ x = y}

/-- Theorem stating that the given expression is correctly notated -/
theorem correct_notation_of_expression : is_correct_notation expression :=
  sorry

end NUMINAMATH_CALUDE_correct_notation_of_expression_l3834_383497


namespace NUMINAMATH_CALUDE_non_adjacent_permutations_five_l3834_383448

def number_of_permutations (n : ℕ) : ℕ := Nat.factorial n

def adjacent_permutations (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

theorem non_adjacent_permutations_five :
  number_of_permutations 5 - adjacent_permutations 5 = 72 := by sorry

end NUMINAMATH_CALUDE_non_adjacent_permutations_five_l3834_383448


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l3834_383403

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x ^ 2 - Real.sin x ^ 2}

-- Define the set N
def N : Set ℝ := {x : ℝ | Complex.abs (2 * x / (1 - Complex.I * Real.sqrt 3)) < 1}

-- State the theorem
theorem M_intersect_N_eq_unit_interval : M ∩ N = Set.Icc 0 1 \ {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_unit_interval_l3834_383403


namespace NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l3834_383406

/-- A polynomial with real coefficients. -/
def RealPolynomial := Polynomial ℝ

/-- Proposition that a polynomial is positive for all positive real numbers. -/
def IsPositiveForPositive (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, x > 0 → p.eval x > 0

/-- Proposition that a polynomial has nonnegative coefficients. -/
def HasNonnegativeCoeffs (p : RealPolynomial) : Prop :=
  ∀ n : ℕ, p.coeff n ≥ 0

/-- Main theorem: For any polynomial P that is positive for all positive real numbers,
    there exist polynomials Q and R with nonnegative coefficients such that
    P(x) = Q(x)/R(x) for all positive real numbers x. -/
theorem positive_polynomial_fraction_representation
  (P : RealPolynomial) (h : IsPositiveForPositive P) :
  ∃ (Q R : RealPolynomial), HasNonnegativeCoeffs Q ∧ HasNonnegativeCoeffs R ∧
    ∀ x : ℝ, x > 0 → P.eval x = (Q.eval x) / (R.eval x) := by
  sorry

end NUMINAMATH_CALUDE_positive_polynomial_fraction_representation_l3834_383406


namespace NUMINAMATH_CALUDE_cash_me_problem_l3834_383438

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def to_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem cash_me_problem :
  ¬∃ (C A S H M E O I D : ℕ),
    is_digit C ∧ is_digit A ∧ is_digit S ∧ is_digit H ∧
    is_digit M ∧ is_digit E ∧ is_digit O ∧ is_digit I ∧ is_digit D ∧
    C ≠ 0 ∧ M ≠ 0 ∧ O ≠ 0 ∧
    to_number C A S H + to_number M E 0 0 = to_number O S I D ∧
    to_number O S I D ≥ 1000 ∧ to_number O S I D < 10000 :=
by sorry

end NUMINAMATH_CALUDE_cash_me_problem_l3834_383438


namespace NUMINAMATH_CALUDE_greatest_length_segment_l3834_383423

theorem greatest_length_segment (AE CD CF AC FD CE : ℝ) : 
  AE = Real.sqrt 106 →
  CD = 5 →
  CF = Real.sqrt 20 →
  AC = 5 →
  FD = Real.sqrt 85 →
  CE = Real.sqrt 29 →
  AC + CE > AE ∧ AC + CE > CD + CF ∧ AC + CE > AC + CF ∧ AC + CE > FD :=
by sorry

end NUMINAMATH_CALUDE_greatest_length_segment_l3834_383423


namespace NUMINAMATH_CALUDE_parabola_focus_l3834_383442

/-- A parabola is defined by the equation x = -1/16 * y^2 + 2 -/
def parabola (x y : ℝ) : Prop := x = -1/16 * y^2 + 2

/-- The focus of a parabola is a point -/
def focus : ℝ × ℝ := (-2, 0)

/-- Theorem: The focus of the parabola defined by x = -1/16 * y^2 + 2 is at (-2, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola x y → focus = (-2, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l3834_383442
