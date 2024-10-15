import Mathlib

namespace NUMINAMATH_CALUDE_candy_bar_cost_proof_l368_36891

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John used -/
def quarters_used : ℕ := 4

/-- The number of dimes John used -/
def dimes_used : ℕ := 3

/-- The number of nickels John used -/
def nickels_used : ℕ := 1

/-- The amount of change John received in cents -/
def change_received : ℕ := 4

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 131

theorem candy_bar_cost_proof :
  (quarters_used * quarter_value + dimes_used * dime_value + nickels_used * nickel_value) - change_received = candy_bar_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_proof_l368_36891


namespace NUMINAMATH_CALUDE_trig_sum_equality_l368_36845

theorem trig_sum_equality : 
  3.423 * Real.sin (10 * π / 180) + Real.sin (20 * π / 180) + Real.sin (30 * π / 180) + 
  Real.sin (40 * π / 180) + Real.sin (50 * π / 180) = 
  Real.sin (25 * π / 180) / (2 * Real.sin (5 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l368_36845


namespace NUMINAMATH_CALUDE_factorization_equality_l368_36834

theorem factorization_equality (a b m : ℝ) : 
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l368_36834


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l368_36809

/-- Given a hyperbola and a circle with specific properties, prove that m = 2 -/
theorem hyperbola_circle_intersection (a b m : ℝ) : 
  a > 0 → b > 0 → m > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →
  (∃ c, c^2 = a^2 + b^2 ∧ c/a = Real.sqrt 2) →
  (∀ x y, (x - m)^2 + y^2 = 4) →
  (∃ x y, x = y ∧ (x - m)^2 + y^2 = 4 ∧ 2 * Real.sqrt (4 - (x - m)^2) = 2 * Real.sqrt 2) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l368_36809


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l368_36830

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - m * x + 3 = 0 ∧ x = 3) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l368_36830


namespace NUMINAMATH_CALUDE_calculate_expression_l368_36832

theorem calculate_expression : 8 * (2 / 16) * 32 - 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l368_36832


namespace NUMINAMATH_CALUDE_paul_initial_pens_l368_36889

/-- The number of pens Paul sold in the garage sale. -/
def pens_sold : ℕ := 92

/-- The number of pens Paul had left after the garage sale. -/
def pens_left : ℕ := 14

/-- The initial number of pens Paul had. -/
def initial_pens : ℕ := pens_sold + pens_left

theorem paul_initial_pens : initial_pens = 106 := by
  sorry

end NUMINAMATH_CALUDE_paul_initial_pens_l368_36889


namespace NUMINAMATH_CALUDE_chocolate_count_l368_36806

/-- The number of boxes of chocolates -/
def num_boxes : ℕ := 6

/-- The number of pieces of chocolate in each box -/
def pieces_per_box : ℕ := 500

/-- The total number of pieces of chocolate -/
def total_pieces : ℕ := num_boxes * pieces_per_box

theorem chocolate_count : total_pieces = 3000 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l368_36806


namespace NUMINAMATH_CALUDE_circle_a_l368_36822

theorem circle_a (x y : ℝ) :
  (x - 3)^2 + (y + 2)^2 = 16 → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (3, -2) ∧ radius = 4 :=
by sorry


end NUMINAMATH_CALUDE_circle_a_l368_36822


namespace NUMINAMATH_CALUDE_chosen_number_proof_l368_36895

theorem chosen_number_proof (x : ℝ) : (x / 2) - 100 = 4 → x = 208 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l368_36895


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l368_36854

theorem largest_n_binomial_sum : 
  (∃ n : ℕ, (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) ∧ 
   (∀ m : ℕ, m > n → Nat.choose 9 4 + Nat.choose 9 5 ≠ Nat.choose 10 m)) → 
  (∃ n : ℕ, n = 5 ∧ (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) ∧ 
   (∀ m : ℕ, m > n → Nat.choose 9 4 + Nat.choose 9 5 ≠ Nat.choose 10 m)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l368_36854


namespace NUMINAMATH_CALUDE_quadratic_root_l368_36881

/-- Given a quadratic equation ax^2 + bx + c = 0 with coefficients defined in terms of p and q,
    if 1 is a root, then -2p / (p - 2) is the other root. -/
theorem quadratic_root (p q : ℝ) : 
  let a := p + q
  let b := p - q
  let c := p * q
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = -2 * p / (p - 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l368_36881


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l368_36849

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 7) % 12 = 0 ∧
  (n - 7) % 16 = 0 ∧
  (n - 7) % 18 = 0 ∧
  (n - 7) % 21 = 0 ∧
  (n - 7) % 28 = 0 ∧
  (n - 7) % 35 = 0 ∧
  (n - 7) % 39 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 65527 ∧
  ∀ m : ℕ, m < 65527 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l368_36849


namespace NUMINAMATH_CALUDE_sock_order_ratio_l368_36808

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
structure SockPrice where
  blue : ℝ

/-- Calculates the total cost of a sock order given the prices -/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  order.black * (3 * price.blue) + order.blue * price.blue

theorem sock_order_ratio : ∀ (original : SockOrder) (price : SockPrice),
  original.black = 6 →
  totalCost { black := original.blue, blue := original.black } price = 1.6 * totalCost original price →
  (original.black : ℝ) / original.blue = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l368_36808


namespace NUMINAMATH_CALUDE_function_decreasing_implies_a_range_a_in_range_l368_36837

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  0 < a ∧ a ≤ 1/4 := by
  sorry

-- Define the set of possible values for a
def a_range : Set ℝ := { a | 0 < a ∧ a ≤ 1/4 }

-- State the final theorem
theorem a_in_range :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ a_range := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_implies_a_range_a_in_range_l368_36837


namespace NUMINAMATH_CALUDE_asian_math_competition_l368_36866

theorem asian_math_competition (total_countries : ℕ) 
  (solved_1 solved_1_2 solved_1_3 solved_1_4 solved_all : ℕ) :
  total_countries = 846 →
  solved_1 = 235 →
  solved_1_2 = 59 →
  solved_1_3 = 29 →
  solved_1_4 = 15 →
  solved_all = 3 →
  ∃ (country : ℕ), country ≤ total_countries ∧ 
    ∃ (students : ℕ), students ≥ 4 ∧
      students ≤ (solved_1 - solved_1_2 - solved_1_3 - solved_1_4 + solved_all) :=
by sorry

end NUMINAMATH_CALUDE_asian_math_competition_l368_36866


namespace NUMINAMATH_CALUDE_william_land_percentage_l368_36870

def total_tax : ℝ := 3840
def tax_percentage : ℝ := 0.75
def william_tax : ℝ := 480

theorem william_land_percentage :
  let total_taxable_income := total_tax / tax_percentage
  let william_percentage := (william_tax / total_taxable_income) * 100
  william_percentage = 9.375 := by sorry

end NUMINAMATH_CALUDE_william_land_percentage_l368_36870


namespace NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l368_36836

theorem gcd_of_powers_minus_one : Nat.gcd (4^8 - 1) (8^12 - 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_minus_one_l368_36836


namespace NUMINAMATH_CALUDE_pattern_equality_l368_36887

theorem pattern_equality (n : ℕ) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l368_36887


namespace NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l368_36878

def raw_materials : ℝ := 35000
def machinery : ℝ := 40000
def total_amount : ℝ := 93750

theorem cash_percentage_is_twenty_percent :
  (total_amount - (raw_materials + machinery)) / total_amount * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l368_36878


namespace NUMINAMATH_CALUDE_partition_naturals_with_property_l368_36896

theorem partition_naturals_with_property : 
  ∃ (partition : ℕ → Fin 100), 
    (∀ i : Fin 100, ∃ n : ℕ, partition n = i) ∧ 
    (∀ a b c : ℕ, a + 99 * b = c → 
      partition a = partition b ∨ 
      partition a = partition c ∨ 
      partition b = partition c) := by sorry

end NUMINAMATH_CALUDE_partition_naturals_with_property_l368_36896


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l368_36856

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9) :
  ∀ n, a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l368_36856


namespace NUMINAMATH_CALUDE_no_valid_triples_l368_36873

theorem no_valid_triples : ¬∃ (a b c : ℤ), 
  (|a + b| + c = 23) ∧ 
  (a * b + |c| = 85) ∧ 
  (∃ k : ℤ, b = 3 * k) := by
sorry

end NUMINAMATH_CALUDE_no_valid_triples_l368_36873


namespace NUMINAMATH_CALUDE_expression_evaluation_l368_36831

theorem expression_evaluation (x : ℝ) (h : x = 6) :
  (1 + 2 / (x + 1)) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l368_36831


namespace NUMINAMATH_CALUDE_afternoon_sales_l368_36852

/-- Represents the amount of pears sold by a salesman in a day -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ
  total : ℕ

/-- Theorem stating the afternoon sales given the conditions -/
theorem afternoon_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning) 
  (h2 : sales.total = sales.morning + sales.afternoon)
  (h3 : sales.total = 510) : 
  sales.afternoon = 340 := by
  sorry

#check afternoon_sales

end NUMINAMATH_CALUDE_afternoon_sales_l368_36852


namespace NUMINAMATH_CALUDE_standing_arrangements_l368_36885

def number_of_people : ℕ := 5

-- Function to calculate the number of ways person A and B can stand next to each other
def ways_next_to_each_other (n : ℕ) : ℕ := sorry

-- Function to calculate the total number of ways n people can stand
def total_ways (n : ℕ) : ℕ := sorry

-- Function to calculate the number of ways person A and B can stand not next to each other
def ways_not_next_to_each_other (n : ℕ) : ℕ := sorry

theorem standing_arrangements :
  (ways_next_to_each_other number_of_people = 48) ∧
  (ways_not_next_to_each_other number_of_people = 72) := by sorry

end NUMINAMATH_CALUDE_standing_arrangements_l368_36885


namespace NUMINAMATH_CALUDE_expression_evaluation_l368_36853

theorem expression_evaluation :
  let x : ℤ := -2
  (x - 2)^2 - 4*x*(x - 1) + (2*x + 1)*(2*x - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l368_36853


namespace NUMINAMATH_CALUDE_tetris_score_calculation_l368_36825

/-- Represents the score calculation for a Tetris game with bonus conditions -/
theorem tetris_score_calculation 
  (single_line_score : ℕ)
  (tetris_score_multiplier : ℕ)
  (single_tetris_bonus_multiplier : ℕ)
  (back_to_back_tetris_bonus : ℕ)
  (single_double_triple_bonus : ℕ)
  (singles_count : ℕ)
  (tetrises_count : ℕ)
  (doubles_count : ℕ)
  (triples_count : ℕ)
  (single_tetris_consecutive_count : ℕ)
  (back_to_back_tetris_count : ℕ)
  (single_double_triple_consecutive_count : ℕ)
  (h1 : single_line_score = 1000)
  (h2 : tetris_score_multiplier = 8)
  (h3 : single_tetris_bonus_multiplier = 2)
  (h4 : back_to_back_tetris_bonus = 5000)
  (h5 : single_double_triple_bonus = 3000)
  (h6 : singles_count = 6)
  (h7 : tetrises_count = 4)
  (h8 : doubles_count = 2)
  (h9 : triples_count = 1)
  (h10 : single_tetris_consecutive_count = 1)
  (h11 : back_to_back_tetris_count = 1)
  (h12 : single_double_triple_consecutive_count = 1) :
  singles_count * single_line_score + 
  tetrises_count * (tetris_score_multiplier * single_line_score) +
  single_tetris_consecutive_count * (single_tetris_bonus_multiplier - 1) * (tetris_score_multiplier * single_line_score) +
  back_to_back_tetris_count * back_to_back_tetris_bonus +
  single_double_triple_consecutive_count * single_double_triple_bonus = 54000 := by
  sorry


end NUMINAMATH_CALUDE_tetris_score_calculation_l368_36825


namespace NUMINAMATH_CALUDE_even_digits_in_base8_523_l368_36818

/-- Converts a natural number to its base-8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-8 representation of 523₁₀ is 1 -/
theorem even_digits_in_base8_523 :
  countEvenDigits (toBase8 523) = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base8_523_l368_36818


namespace NUMINAMATH_CALUDE_f_even_iff_a_eq_zero_l368_36838

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax for some a ∈ ℝ -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x

/-- Theorem: f is an even function if and only if a = 0 -/
theorem f_even_iff_a_eq_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_f_even_iff_a_eq_zero_l368_36838


namespace NUMINAMATH_CALUDE_make_up_average_is_95_percent_l368_36813

/-- Represents the average score of students who took the exam on the make-up date -/
def make_up_average (total_students : ℕ) (assigned_day_percent : ℚ) (assigned_day_average : ℚ) (overall_average : ℚ) : ℚ :=
  (overall_average * total_students - assigned_day_average * (assigned_day_percent * total_students)) / ((1 - assigned_day_percent) * total_students)

/-- Theorem stating the average score of students who took the exam on the make-up date -/
theorem make_up_average_is_95_percent :
  make_up_average 100 (70/100) (65/100) (74/100) = 95/100 := by
  sorry

end NUMINAMATH_CALUDE_make_up_average_is_95_percent_l368_36813


namespace NUMINAMATH_CALUDE_vector_addition_l368_36892

theorem vector_addition (a b : ℝ × ℝ) :
  a = (5, -3) → b = (-6, 4) → a + b = (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l368_36892


namespace NUMINAMATH_CALUDE_inverse_proportion_l368_36862

/-- Given that the product of x and y is constant, and x = 30 when y = 10,
    prove that x = 60 when y = 5 and the relationship doesn't hold for x = 48 and y = 15 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 30 * 10 = k) :
  (5 * 60 = k) ∧ ¬(48 * 15 = k) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l368_36862


namespace NUMINAMATH_CALUDE_BEE_has_largest_value_l368_36865

def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | _   => 0

def word_value (w : String) : ℕ :=
  w.toList.map letter_value |>.sum

theorem BEE_has_largest_value :
  let BAD := "BAD"
  let CAB := "CAB"
  let DAD := "DAD"
  let BEE := "BEE"
  let BED := "BED"
  (word_value BEE > word_value BAD) ∧
  (word_value BEE > word_value CAB) ∧
  (word_value BEE > word_value DAD) ∧
  (word_value BEE > word_value BED) := by
  sorry

end NUMINAMATH_CALUDE_BEE_has_largest_value_l368_36865


namespace NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l368_36804

theorem nested_sqrt_fourth_power : 
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l368_36804


namespace NUMINAMATH_CALUDE_expand_product_l368_36821

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l368_36821


namespace NUMINAMATH_CALUDE_semicircle_function_max_point_max_value_max_point_trig_l368_36871

noncomputable section

variables (R : ℝ) (x : ℝ)

def semicircle_point (R x : ℝ) : ℝ × ℝ :=
  (x, Real.sqrt (4 * R^2 - x^2))

def y (R x : ℝ) : ℝ :=
  2 * x + 3 * (2 * R - x^2 / (2 * R))

theorem semicircle_function (R : ℝ) (h : R > 0) :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * R →
  y R x = -3 / (2 * R) * x^2 + 2 * x + 6 * R :=
sorry

theorem max_point (R : ℝ) (h : R > 0) :
  ∃ x_max, x_max = 2 * R / 3 ∧
  ∀ x, 0 ≤ x ∧ x ≤ 2 * R → y R x ≤ y R x_max :=
sorry

theorem max_value (R : ℝ) (h : R > 0) :
  y R (2 * R / 3) = 20 * R / 3 :=
sorry

theorem max_point_trig (R : ℝ) (h : R > 0) :
  let x_max := 2 * R / 3
  let α := Real.arccos (1 - x_max^2 / (2 * R^2))
  Real.cos α = 7 / 9 ∧ Real.sin α = 4 * Real.sqrt 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_semicircle_function_max_point_max_value_max_point_trig_l368_36871


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l368_36814

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sqrt (3 + 4 * Real.sqrt 6 - (16 * Real.sqrt 3 - 8 * Real.sqrt 2) * Real.sin x) = 4 * Real.sin x - Real.sqrt 3) ↔ 
  ∃ k : ℤ, x = (-1)^k * (π / 4) + 2 * k * π :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l368_36814


namespace NUMINAMATH_CALUDE_probabilities_ascending_order_order_matches_sequence_l368_36880

-- Define the probabilities of each event
def prob_event1 : ℚ := 2/3
def prob_event2 : ℚ := 1
def prob_event3 : ℚ := 1/3
def prob_event4 : ℚ := 1/2
def prob_event5 : ℚ := 0

-- Define a function to represent the correct order
def correct_order : Fin 5 → ℚ
  | 0 => prob_event5
  | 1 => prob_event3
  | 2 => prob_event4
  | 3 => prob_event1
  | 4 => prob_event2

-- Theorem stating that the probabilities are in ascending order
theorem probabilities_ascending_order :
  ∀ i j : Fin 5, i < j → correct_order i ≤ correct_order j :=
by sorry

-- Theorem stating that this order matches the given sequence (5) (3) (4) (1) (2)
theorem order_matches_sequence :
  correct_order 0 = prob_event5 ∧
  correct_order 1 = prob_event3 ∧
  correct_order 2 = prob_event4 ∧
  correct_order 3 = prob_event1 ∧
  correct_order 4 = prob_event2 :=
by sorry

end NUMINAMATH_CALUDE_probabilities_ascending_order_order_matches_sequence_l368_36880


namespace NUMINAMATH_CALUDE_k_range_theorem_l368_36827

/-- Proposition p: The equation represents an ellipse with foci on the y-axis -/
def p (k : ℝ) : Prop := 3 < k ∧ k < 9/2

/-- Proposition q: The equation represents a hyperbola with eccentricity e in (√3, 2) -/
def q (k : ℝ) : Prop := 4 < k ∧ k < 6

/-- The range of real values for k -/
def k_range (k : ℝ) : Prop := (3 < k ∧ k ≤ 4) ∨ (9/2 ≤ k ∧ k < 6)

theorem k_range_theorem (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → k_range k := by
  sorry

end NUMINAMATH_CALUDE_k_range_theorem_l368_36827


namespace NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l368_36844

theorem multiples_of_four_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0) (Finset.range 350 \ Finset.range 100)).card = 62 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_between_100_and_350_l368_36844


namespace NUMINAMATH_CALUDE_product_of_sequence_a_l368_36864

def sequence_a : ℕ → ℚ
  | 0 => 3/2
  | n + 1 => 3 + (sequence_a n - 2)^2

def infinite_product (f : ℕ → ℚ) : ℚ := sorry

theorem product_of_sequence_a :
  infinite_product sequence_a = 4/3 := by sorry

end NUMINAMATH_CALUDE_product_of_sequence_a_l368_36864


namespace NUMINAMATH_CALUDE_prove_vector_sum_with_scalar_multiple_l368_36869

def vector_sum_with_scalar_multiple : Prop :=
  let v1 : Fin 3 → ℝ := ![3, -2, 5]
  let v2 : Fin 3 → ℝ := ![-1, 4, -3]
  let result : Fin 3 → ℝ := ![1, 6, -1]
  v1 + 2 • v2 = result

theorem prove_vector_sum_with_scalar_multiple : vector_sum_with_scalar_multiple := by
  sorry

end NUMINAMATH_CALUDE_prove_vector_sum_with_scalar_multiple_l368_36869


namespace NUMINAMATH_CALUDE_equation_solution_l368_36899

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)*(3*x + 1)*(5*x + 1)*(30*x + 1)
  ∀ x : ℝ, f x = 10 ↔ x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l368_36899


namespace NUMINAMATH_CALUDE_root_sum_theorem_l368_36833

theorem root_sum_theorem (a b c : ℝ) : 
  (a^3 - 6*a^2 + 8*a - 3 = 0) → 
  (b^3 - 6*b^2 + 8*b - 3 = 0) → 
  (c^3 - 6*c^2 + 8*c - 3 = 0) → 
  (a/(b*c + 2) + b/(a*c + 2) + c/(a*b + 2) = 0) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l368_36833


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l368_36859

/-- Parabola function -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

/-- Point A on the parabola -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the parabola -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the parabola -/
def C : ℝ × ℝ := (2, f 2)

/-- Theorem stating the relationship between y-coordinates of A, B, and C -/
theorem parabola_point_relationship : A.2 > B.2 ∧ B.2 > C.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l368_36859


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l368_36811

theorem consecutive_integers_product_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 336) → (a + b + c = 21) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l368_36811


namespace NUMINAMATH_CALUDE_range_of_a_l368_36868

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : ∀ x, -1 < x ∧ x < 1 → ∃ y, f x = y)  -- f is defined on (-1, 1)
  (h2 : ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y)  -- f is decreasing on (-1, 1)
  (h3 : f (a - 1) > f (2 * a))  -- f(a-1) > f(2a)
  (h4 : -1 < a - 1 ∧ a - 1 < 1)  -- -1 < a-1 < 1
  (h5 : -1 < 2 * a ∧ 2 * a < 1)  -- -1 < 2a < 1
  : 0 < a ∧ a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l368_36868


namespace NUMINAMATH_CALUDE_george_eggs_boxes_l368_36820

/-- Given a total number of eggs and eggs per box, calculates the number of boxes required. -/
def calculate_boxes (total_eggs : ℕ) (eggs_per_box : ℕ) : ℕ :=
  total_eggs / eggs_per_box

theorem george_eggs_boxes :
  let total_eggs : ℕ := 15
  let eggs_per_box : ℕ := 3
  calculate_boxes total_eggs eggs_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_george_eggs_boxes_l368_36820


namespace NUMINAMATH_CALUDE_ariella_meetings_percentage_l368_36888

theorem ariella_meetings_percentage : 
  let work_day_hours : ℝ := 8
  let first_meeting_minutes : ℝ := 60
  let second_meeting_factor : ℝ := 1.5
  let work_day_minutes : ℝ := work_day_hours * 60
  let second_meeting_minutes : ℝ := second_meeting_factor * first_meeting_minutes
  let total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes
  let meeting_percentage : ℝ := (total_meeting_minutes / work_day_minutes) * 100
  meeting_percentage = 31.25 := by sorry

end NUMINAMATH_CALUDE_ariella_meetings_percentage_l368_36888


namespace NUMINAMATH_CALUDE_cereal_spending_ratio_is_two_to_one_l368_36876

/-- The ratio of Snap's spending to Crackle's spending on cereal -/
def cereal_spending_ratio : ℚ :=
  let total_spent : ℚ := 150
  let pop_spent : ℚ := 15
  let crackle_spent : ℚ := 3 * pop_spent
  let snap_spent : ℚ := total_spent - crackle_spent - pop_spent
  snap_spent / crackle_spent

/-- Theorem stating that the ratio of Snap's spending to Crackle's spending is 2:1 -/
theorem cereal_spending_ratio_is_two_to_one :
  cereal_spending_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_cereal_spending_ratio_is_two_to_one_l368_36876


namespace NUMINAMATH_CALUDE_M_equals_divisors_of_151_l368_36840

def M : Set Nat :=
  {d | ∃ m n : Nat, d = Nat.gcd (2*n + 3*m + 13) (Nat.gcd (3*n + 5*m + 1) (6*n + 6*m - 1))}

theorem M_equals_divisors_of_151 : M = {d : Nat | d > 0 ∧ d ∣ 151} := by
  sorry

end NUMINAMATH_CALUDE_M_equals_divisors_of_151_l368_36840


namespace NUMINAMATH_CALUDE_distance_walked_calculation_l368_36810

/-- Calculates the distance walked given the walking time and speed. -/
def distance_walked (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem: The distance walked is 499.98 meters given the specified conditions. -/
theorem distance_walked_calculation :
  let time : ℝ := 6
  let speed : ℝ := 83.33
  distance_walked time speed = 499.98 := by sorry

end NUMINAMATH_CALUDE_distance_walked_calculation_l368_36810


namespace NUMINAMATH_CALUDE_event_A_subset_event_B_l368_36801

-- Define the sample space for tossing two coins
inductive CoinOutcome
  | HH -- Both heads
  | HT -- First head, second tail
  | TH -- First tail, second head
  | TT -- Both tails

-- Define the probability space
def coin_toss_space : Type := CoinOutcome

-- Define the events A and B
def event_A : Set coin_toss_space := {CoinOutcome.HH}
def event_B : Set coin_toss_space := {CoinOutcome.HH, CoinOutcome.TT}

-- State the theorem
theorem event_A_subset_event_B : event_A ⊆ event_B := by sorry

end NUMINAMATH_CALUDE_event_A_subset_event_B_l368_36801


namespace NUMINAMATH_CALUDE_infinite_intersecting_lines_l368_36817

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry  -- Definition of skew lines

/-- A set of three pairwise skew lines -/
structure SkewLineTriple where
  a : Line3D
  b : Line3D
  c : Line3D
  skew_ab : are_skew a b
  skew_bc : are_skew b c
  skew_ca : are_skew c a

/-- The set of lines intersecting all three lines in a SkewLineTriple -/
def intersecting_lines (triple : SkewLineTriple) : Set Line3D :=
  sorry  -- Definition of the set of intersecting lines

/-- Theorem stating that there are infinitely many intersecting lines -/
theorem infinite_intersecting_lines (triple : SkewLineTriple) :
  Set.Infinite (intersecting_lines triple) :=
sorry

end NUMINAMATH_CALUDE_infinite_intersecting_lines_l368_36817


namespace NUMINAMATH_CALUDE_min_additional_bureaus_for_192_and_36_l368_36805

/-- Given a number of bureaus and offices, calculates the minimum number of additional
    bureaus needed to ensure each office gets an equal number of bureaus. -/
def min_additional_bureaus (total_bureaus : ℕ) (num_offices : ℕ) : ℕ :=
  let bureaus_per_office := (total_bureaus + num_offices - 1) / num_offices
  bureaus_per_office * num_offices - total_bureaus

/-- Proves that for 192 bureaus and 36 offices, the minimum number of additional
    bureaus needed is 24. -/
theorem min_additional_bureaus_for_192_and_36 :
  min_additional_bureaus 192 36 = 24 := by
  sorry

#eval min_additional_bureaus 192 36

end NUMINAMATH_CALUDE_min_additional_bureaus_for_192_and_36_l368_36805


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l368_36819

theorem rectangle_perimeter (a b : ℤ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  a * b = 4 * (2 * a + 2 * b) - 12 → 
  2 * (a + b) = 72 ∨ 2 * (a + b) = 100 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l368_36819


namespace NUMINAMATH_CALUDE_min_value_a_l368_36861

theorem min_value_a (h : ∀ x y : ℝ, x > 0 → y > 0 → x + y ≥ 9) :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, x > 0 → x + a ≥ 9) ∧
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, x > 0 → x + b ≥ 9) → b ≥ a) ∧
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l368_36861


namespace NUMINAMATH_CALUDE_triangle_problem_l368_36803

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b² + c² - a²) / cos A = 2 and (a cos B - b cos A) / (a cos B + b cos A) - b / c = 1,
    then bc = 1 and the area of triangle ABC is √3 / 4 -/
theorem triangle_problem (a b c A B C : ℝ) (h1 : (b^2 + c^2 - a^2) / Real.cos A = 2)
    (h2 : (a * Real.cos B - b * Real.cos A) / (a * Real.cos B + b * Real.cos A) - b / c = 1) :
    b * c = 1 ∧ (1/2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l368_36803


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l368_36863

/-- If a square is cut into two identical rectangles, each with a perimeter of 24 cm,
    then the area of the original square is 64 cm². -/
theorem square_area_from_rectangle_perimeter :
  ∀ (side : ℝ), side > 0 →
  (2 * (side + side / 2) = 24) →
  side * side = 64 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l368_36863


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l368_36850

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the expected strawberry harvest based on garden dimensions and planting information -/
def expectedStrawberryHarvest (dimensions : GardenDimensions) (plantsPerSquareFoot : ℝ) (strawberriesPerPlant : ℝ) : ℝ :=
  dimensions.length * dimensions.width * plantsPerSquareFoot * strawberriesPerPlant

/-- Theorem stating that Carrie's garden will yield 1920 strawberries -/
theorem carries_strawberry_harvest :
  let dimensions : GardenDimensions := { length := 6, width := 8 }
  let plantsPerSquareFoot : ℝ := 4
  let strawberriesPerPlant : ℝ := 10
  expectedStrawberryHarvest dimensions plantsPerSquareFoot strawberriesPerPlant = 1920 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l368_36850


namespace NUMINAMATH_CALUDE_solve_for_m_l368_36815

theorem solve_for_m (x : ℝ) (m : ℝ) : 
  (-3 * x = -5 * x + 4) → 
  (m^x - 9 = 0) → 
  (m = 3 ∨ m = -3) := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l368_36815


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_149_l368_36894

theorem first_nonzero_digit_after_decimal_1_149 : ∃ (n : ℕ) (d : ℕ),
  (1 : ℚ) / 149 = (n : ℚ) / 10^(d + 1) + (7 : ℚ) / 10^(d + 2) + (r : ℚ)
  ∧ 0 ≤ r
  ∧ r < 1 / 10^(d + 2)
  ∧ n < 10^(d + 1) :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_149_l368_36894


namespace NUMINAMATH_CALUDE_smallest_factor_of_32_not_8_l368_36884

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ 
  (32 % n = 0) ∧ (8 % n ≠ 0) ∧ 
  (∀ m : ℕ, m < n → (32 % m = 0 → 8 % m = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_32_not_8_l368_36884


namespace NUMINAMATH_CALUDE_quadratic_inequality_l368_36858

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots -1 and 4, and a < 0,
    prove that ax^2 + bx + c < 0 when x < -1 or x > 4 -/
theorem quadratic_inequality (a b c : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = -1 ∨ x = 4) :
  ∀ x, a * x^2 + b * x + c < 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l368_36858


namespace NUMINAMATH_CALUDE_wenzhou_population_scientific_notation_l368_36857

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (h : x > 0) : ScientificNotation :=
  sorry

theorem wenzhou_population_scientific_notation :
  toScientificNotation 9570000 (by norm_num) =
    ScientificNotation.mk 9.57 6 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_wenzhou_population_scientific_notation_l368_36857


namespace NUMINAMATH_CALUDE_max_log_sum_l368_36826

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 4*y = 40) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + 4*b = 40 → Real.log a + Real.log b ≤ Real.log x + Real.log y) →
  Real.log x + Real.log y = 2 := by
sorry

end NUMINAMATH_CALUDE_max_log_sum_l368_36826


namespace NUMINAMATH_CALUDE_A_minus_B_equals_1790_l368_36828

/-- Calculates the value of A based on the given groups -/
def calculate_A : ℕ := 1 * 1000 + 16 * 100 + 28 * 10

/-- Calculates the value of B based on the given jumps and interval -/
def calculate_B : ℕ := 355 + 3 * 245

/-- Proves that A - B equals 1790 -/
theorem A_minus_B_equals_1790 : calculate_A - calculate_B = 1790 := by
  sorry

end NUMINAMATH_CALUDE_A_minus_B_equals_1790_l368_36828


namespace NUMINAMATH_CALUDE_projection_onto_common_vector_l368_36802

/-- Given two vectors v1 and v2 in ℝ², prove that their projection onto a common vector u results in the vector q. -/
theorem projection_onto_common_vector (v1 v2 u q : ℝ × ℝ) : 
  v1 = (3, 2) → 
  v2 = (2, 5) → 
  q = (27/8, 7/8) → 
  ∃ (t : ℝ), q = v1 + t • (v2 - v1) ∧ 
  (q - v1) • (v2 - v1) = 0 ∧ 
  (q - v2) • (v2 - v1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_projection_onto_common_vector_l368_36802


namespace NUMINAMATH_CALUDE_min_value_theorem_l368_36824

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y = 3) :
  (1/x + 2/y) ≥ 8/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l368_36824


namespace NUMINAMATH_CALUDE_cube_surface_area_l368_36872

/-- Given a cube with the sum of edge lengths equal to 36 and space diagonal length equal to 3√3,
    the total surface area is 54. -/
theorem cube_surface_area (s : ℝ) 
  (h1 : 12 * s = 36) 
  (h2 : s * Real.sqrt 3 = 3 * Real.sqrt 3) : 
  6 * s^2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l368_36872


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l368_36874

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 202 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 202 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l368_36874


namespace NUMINAMATH_CALUDE_vertex_ordinate_zero_l368_36890

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The number of solutions to the equation (f x)^3 - f x = 0 -/
def numSolutions (f : ℝ → ℝ) : ℕ := sorry

/-- The ordinate (y-coordinate) of the vertex of a quadratic polynomial -/
def vertexOrdinate (f : ℝ → ℝ) : ℝ := sorry

/-- 
If f is a quadratic polynomial and (f x)^3 - f x = 0 has exactly three solutions,
then the ordinate of the vertex of f is 0
-/
theorem vertex_ordinate_zero 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = QuadraticPolynomial a b c) 
  (h_solutions : numSolutions f = 3) : 
  vertexOrdinate f = 0 := by sorry

end NUMINAMATH_CALUDE_vertex_ordinate_zero_l368_36890


namespace NUMINAMATH_CALUDE_area_ratio_of_rectangles_l368_36882

/-- A structure representing a rectangle with width and length --/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- A structure representing a square composed of five rectangles --/
structure SquareOfRectangles where
  shaded : Rectangle
  unshaded : Rectangle
  total_width : ℝ
  total_height : ℝ

/-- The theorem stating the ratio of areas of shaded to unshaded rectangles --/
theorem area_ratio_of_rectangles (s : SquareOfRectangles) 
  (h1 : s.shaded.width + s.shaded.width + s.unshaded.width = s.total_width)
  (h2 : s.shaded.length = s.total_height)
  (h3 : 2 * (s.shaded.width + s.shaded.length) = 2 * (s.unshaded.width + s.unshaded.length))
  (h4 : s.shaded.width > 0)
  (h5 : s.shaded.length > 0)
  (h6 : s.unshaded.width > 0)
  (h7 : s.unshaded.length > 0) :
  (s.shaded.width * s.shaded.length) / (s.unshaded.width * s.unshaded.length) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_rectangles_l368_36882


namespace NUMINAMATH_CALUDE_problem_statements_l368_36807

theorem problem_statements :
  -- Statement 1
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)) ∧
  -- Statement 2
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x ≤ 2 ∧ x^2 - 3*x + 2 > 0) ∧
  -- Statement 3
  (∃ p q : Prop, ¬(p ∧ q) ∧ (p ∨ q)) ∧
  -- Statement 4
  (¬(∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l368_36807


namespace NUMINAMATH_CALUDE_complex_number_location_l368_36846

/-- The complex number z = (2+i)/(1+i) is located in Quadrant IV -/
theorem complex_number_location :
  let z : ℂ := (2 + I) / (1 + I)
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l368_36846


namespace NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l368_36835

theorem inverse_false_implies_negation_false (p : Prop) :
  (¬p → False) → (¬p = False) := by
  sorry

end NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l368_36835


namespace NUMINAMATH_CALUDE_candy_per_package_l368_36848

/-- Given that Robin has 45 packages of candy and 405 pieces of candies in total,
    prove that there are 9 pieces of candy in each package. -/
theorem candy_per_package (packages : ℕ) (total_pieces : ℕ) 
    (h1 : packages = 45) (h2 : total_pieces = 405) : 
    total_pieces / packages = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_per_package_l368_36848


namespace NUMINAMATH_CALUDE_mike_weekly_pullups_l368_36841

/-- Calculates the number of pull-ups Mike does in a week -/
def weekly_pullups (pullups_per_entry : ℕ) (entries_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  pullups_per_entry * entries_per_day * days_per_week

/-- Proves that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups :
  weekly_pullups 2 5 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_pullups_l368_36841


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l368_36847

theorem continued_fraction_equality : 
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_equality_l368_36847


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l368_36800

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldl (λ sum (i, d) => sum + d * b^i) 0

theorem base_conversion_theorem :
  let base_5_123 := to_base_10 [3, 2, 1] 5
  let base_8_107 := to_base_10 [7, 0, 1] 8
  let base_9_4321 := to_base_10 [1, 2, 3, 4] 9
  (2468 / base_5_123) * base_8_107 + base_9_4321 = 7789 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l368_36800


namespace NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l368_36886

theorem abs_neg_2023_eq_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l368_36886


namespace NUMINAMATH_CALUDE_shortest_side_length_l368_36829

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, the shortest side has length 1. -/
theorem shortest_side_length (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a * c + c^2 = b^2 - a^2 →  -- given condition
  b = Real.sqrt 7 →  -- longest side is √7
  Real.sin C = 2 * Real.sin A →  -- given condition
  b ≥ a ∧ b ≥ c →  -- b is the longest side
  min a c = 1 :=  -- the shortest side has length 1
by sorry

end NUMINAMATH_CALUDE_shortest_side_length_l368_36829


namespace NUMINAMATH_CALUDE_perfect_square_difference_l368_36875

theorem perfect_square_difference (x y : ℝ) : (x - y)^2 = x^2 - 2*x*y + y^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l368_36875


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l368_36883

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 18 * y = 5 ∧ 4 * y - x = 6 ∧ x = -64/3 ∧ y = -23/6 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l368_36883


namespace NUMINAMATH_CALUDE_cyclist_pedestrian_speed_ratio_l368_36855

/-- Represents the speed of a person -/
structure Speed :=
  (value : ℝ)

/-- Represents a point in time -/
structure Time :=
  (hours : ℝ)

/-- Represents a distance between two points -/
structure Distance :=
  (value : ℝ)

/-- The problem setup -/
structure ProblemSetup :=
  (pedestrian_start : Time)
  (cyclist_start : Time)
  (meetup_time : Time)
  (cyclist_return : Time)
  (final_meetup : Time)
  (distance_AB : Distance)

/-- The theorem to be proved -/
theorem cyclist_pedestrian_speed_ratio 
  (setup : ProblemSetup)
  (pedestrian_speed : Speed)
  (cyclist_speed : Speed)
  (h1 : setup.pedestrian_start.hours = 12)
  (h2 : setup.meetup_time.hours = 13)
  (h3 : setup.final_meetup.hours = 16)
  (h4 : setup.pedestrian_start.hours < setup.cyclist_start.hours)
  (h5 : setup.cyclist_start.hours < setup.meetup_time.hours) :
  cyclist_speed.value / pedestrian_speed.value = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_pedestrian_speed_ratio_l368_36855


namespace NUMINAMATH_CALUDE_sector_area_l368_36843

theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 2) :
  (1 / 2) * θ * r^2 = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l368_36843


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l368_36823

theorem min_value_sum_reciprocals (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → a / b + b / a ≤ x / y + y / x) ∧ 
  (a / b + b / a = 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l368_36823


namespace NUMINAMATH_CALUDE_linked_rings_length_l368_36877

/-- Represents a sequence of linked rings with specific properties. -/
structure LinkedRings where
  ringThickness : ℝ
  topRingDiameter : ℝ
  bottomRingDiameter : ℝ
  diameterDecrease : ℝ

/-- Calculates the total length of the linked rings. -/
def totalLength (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total length of the linked rings with given properties is 342 cm. -/
theorem linked_rings_length :
  let rings : LinkedRings := {
    ringThickness := 2,
    topRingDiameter := 40,
    bottomRingDiameter := 4,
    diameterDecrease := 2
  }
  totalLength rings = 342 := by sorry

end NUMINAMATH_CALUDE_linked_rings_length_l368_36877


namespace NUMINAMATH_CALUDE_officers_count_l368_36812

/-- The number of ways to choose 4 distinct officers from a group of n people -/
def choose_officers (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3)

/-- The number of club members -/
def club_members : ℕ := 12

/-- Theorem stating that choosing 4 officers from 12 members results in 11880 possibilities -/
theorem officers_count : choose_officers club_members = 11880 := by
  sorry

end NUMINAMATH_CALUDE_officers_count_l368_36812


namespace NUMINAMATH_CALUDE_sequence_convergence_l368_36897

theorem sequence_convergence (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : 
  a 1 = 0 ∨ a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l368_36897


namespace NUMINAMATH_CALUDE_triangle_properties_l368_36816

theorem triangle_properties (A B C : ℝ) (a b c R : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  R > 0 →
  a = Real.sqrt 3 →
  A = π/3 →
  2 * R = a / Real.sin A →
  2 * R = b / Real.sin B →
  2 * R = c / Real.sin C →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  R = 1 ∧ ∀ (b' c' : ℝ), b' * c' ≤ 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l368_36816


namespace NUMINAMATH_CALUDE_min_value_of_f_l368_36879

open Real

noncomputable def f (x : ℝ) : ℝ := (log x)^2 / x

theorem min_value_of_f :
  ∀ x > 0, f x ≥ 0 ∧ ∃ x₀ > 0, f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l368_36879


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l368_36851

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l368_36851


namespace NUMINAMATH_CALUDE_probability_x_greater_than_3y_l368_36867

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  y_min : ℝ
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is inside a rectangle --/
def Point.insideRectangle (p : Point) (r : Rectangle) : Prop :=
  r.x_min ≤ p.x ∧ p.x ≤ r.x_max ∧ r.y_min ≤ p.y ∧ p.y ≤ r.y_max

/-- The probability of an event occurring for a point randomly picked from a rectangle --/
def probability (r : Rectangle) (event : Point → Prop) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle :=
  { x_min := 0, y_min := 0, x_max := 3000, y_max := 3000 }

/-- The event x > 3y --/
def xGreaterThan3y (p : Point) : Prop :=
  p.x > 3 * p.y

theorem probability_x_greater_than_3y :
  probability problemRectangle xGreaterThan3y = 1/6 :=
sorry

end NUMINAMATH_CALUDE_probability_x_greater_than_3y_l368_36867


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l368_36842

theorem regular_polygon_interior_angle (n : ℕ) (n_ge_3 : n ≥ 3) :
  let interior_angle := (n - 2) * 180 / n
  interior_angle = 135 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l368_36842


namespace NUMINAMATH_CALUDE_reflection_over_x_axis_l368_36893

/-- Reflects a point over the x-axis -/
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The reflection of (-4, 3) over the x-axis is (-4, -3) -/
theorem reflection_over_x_axis :
  reflect_over_x_axis (-4, 3) = (-4, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_over_x_axis_l368_36893


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l368_36839

open Set

def A : Set ℝ := {x | x^2 - 1 ≤ 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_equality : A ∩ (𝒰 \ B) = {x | x = 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l368_36839


namespace NUMINAMATH_CALUDE_crucian_carp_cultivation_optimal_l368_36898

/-- Represents the seafood wholesaler's crucian carp cultivation problem -/
structure CrucianCarpProblem where
  initialWeight : ℝ  -- Initial weight of crucian carp in kg
  initialPrice : ℝ   -- Initial price per kg in yuan
  priceIncrease : ℝ  -- Daily price increase per kg in yuan
  maxDays : ℕ        -- Maximum culture period in days
  dailyLoss : ℝ      -- Daily weight loss due to oxygen deficiency in kg
  lossPrice : ℝ      -- Price of oxygen-deficient carp per kg in yuan
  dailyExpense : ℝ   -- Daily expenses during culture in yuan

/-- Calculates the profit for a given number of culture days -/
def profit (p : CrucianCarpProblem) (days : ℝ) : ℝ :=
  p.dailyLoss * days * (p.lossPrice - p.initialPrice) +
  (p.initialWeight - p.dailyLoss * days) * (p.initialPrice + p.priceIncrease * days) -
  p.initialWeight * p.initialPrice -
  p.dailyExpense * days

/-- The main theorem to be proved -/
theorem crucian_carp_cultivation_optimal (p : CrucianCarpProblem)
  (h1 : p.initialWeight = 1000)
  (h2 : p.initialPrice = 10)
  (h3 : p.priceIncrease = 1)
  (h4 : p.maxDays = 20)
  (h5 : p.dailyLoss = 10)
  (h6 : p.lossPrice = 5)
  (h7 : p.dailyExpense = 450) :
  (∃ x : ℝ, x ≤ p.maxDays ∧ profit p x = 8500 ∧ x = 10) ∧
  (∀ x : ℝ, x ≤ p.maxDays → profit p x ≤ 6000) ∧
  (∃ x : ℝ, x ≤ p.maxDays ∧ profit p x = 6000) := by
  sorry


end NUMINAMATH_CALUDE_crucian_carp_cultivation_optimal_l368_36898


namespace NUMINAMATH_CALUDE_xyz_sum_l368_36860

theorem xyz_sum (x y z : ℝ) (eq1 : 2*x + 3*y + 4*z = 10) (eq2 : y + 2*z = 2) : 
  x + y + z = 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l368_36860
