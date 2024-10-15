import Mathlib

namespace NUMINAMATH_CALUDE_spy_is_A_l2031_203135

/-- Represents the three defendants -/
inductive Defendant : Type
  | A
  | B
  | C

/-- Represents the role of each defendant -/
inductive Role : Type
  | Spy
  | Knight
  | Liar

/-- The statement made by each defendant -/
def statement (d : Defendant) : Prop :=
  match d with
  | Defendant.A => ∃ r, r = Role.Spy
  | Defendant.B => ∃ r, r = Role.Knight
  | Defendant.C => ∃ r, r = Role.Spy

/-- The role assigned to each defendant -/
def assigned_role : Defendant → Role := sorry

/-- A defendant tells the truth if they are the Knight or if they are the Spy and claim to be the Spy -/
def tells_truth (d : Defendant) : Prop :=
  (assigned_role d = Role.Knight) ∨
  (assigned_role d = Role.Spy ∧ statement d)

theorem spy_is_A :
  (∃! d : Defendant, assigned_role d = Role.Spy) ∧
  (∃! d : Defendant, assigned_role d = Role.Knight) ∧
  (∃! d : Defendant, assigned_role d = Role.Liar) ∧
  (tells_truth Defendant.B) →
  assigned_role Defendant.A = Role.Spy := by
  sorry


end NUMINAMATH_CALUDE_spy_is_A_l2031_203135


namespace NUMINAMATH_CALUDE_scissors_count_l2031_203195

/-- The total number of scissors after addition -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of scissors is 52 -/
theorem scissors_count : total_scissors 39 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l2031_203195


namespace NUMINAMATH_CALUDE_x_over_y_value_l2031_203197

theorem x_over_y_value (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) 
  (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_value_l2031_203197


namespace NUMINAMATH_CALUDE_integer_representation_l2031_203153

theorem integer_representation (N : ℕ+) : 
  ∃ (p q u v : ℤ), (N : ℤ) = p * q + u * v ∧ u - v = 2 * (p - q) := by
  sorry

end NUMINAMATH_CALUDE_integer_representation_l2031_203153


namespace NUMINAMATH_CALUDE_mean_squared_sum_l2031_203194

theorem mean_squared_sum (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_mean_squared_sum_l2031_203194


namespace NUMINAMATH_CALUDE_min_unsuccessful_placements_l2031_203122

/-- Represents a cell in the grid -/
inductive Cell
| Plus : Cell
| Minus : Cell

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Cell

/-- Represents a T-shaped figure -/
structure TShape where
  row : Fin 8
  col : Fin 8
  orientation : Bool  -- True for horizontal, False for vertical

/-- Calculates the sum of a T-shape on the grid -/
def tShapeSum (g : Grid) (t : TShape) : Int :=
  sorry

/-- Counts the number of unsuccessful T-shape placements -/
def countUnsuccessful (g : Grid) : Nat :=
  sorry

/-- Theorem: The minimum number of unsuccessful T-shape placements is 132 -/
theorem min_unsuccessful_placements :
  ∀ g : Grid, countUnsuccessful g ≥ 132 :=
sorry

end NUMINAMATH_CALUDE_min_unsuccessful_placements_l2031_203122


namespace NUMINAMATH_CALUDE_num_divisors_2_pow_7_num_divisors_5_pow_4_num_divisors_2_pow_7_mul_5_pow_4_num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k_num_divisors_3600_num_divisors_42_pow_5_l2031_203119

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorem for 2^7
theorem num_divisors_2_pow_7 : num_divisors (2^7) = 8 := by sorry

-- Theorem for 5^4
theorem num_divisors_5_pow_4 : num_divisors (5^4) = 5 := by sorry

-- Theorem for 2^7 * 5^4
theorem num_divisors_2_pow_7_mul_5_pow_4 : num_divisors (2^7 * 5^4) = 40 := by sorry

-- Theorem for 2^m * 5^n * 3^k
theorem num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k (m n k : ℕ) :
  num_divisors (2^m * 5^n * 3^k) = (m + 1) * (n + 1) * (k + 1) := by sorry

-- Theorem for 3600
theorem num_divisors_3600 : num_divisors 3600 = 45 := by sorry

-- Theorem for 42^5
theorem num_divisors_42_pow_5 : num_divisors (42^5) = 216 := by sorry

end NUMINAMATH_CALUDE_num_divisors_2_pow_7_num_divisors_5_pow_4_num_divisors_2_pow_7_mul_5_pow_4_num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k_num_divisors_3600_num_divisors_42_pow_5_l2031_203119


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l2031_203157

theorem average_of_four_numbers (r s t u : ℝ) :
  (5 / 2) * (r + s + t + u) = 25 → (r + s + t + u) / 4 = 2.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l2031_203157


namespace NUMINAMATH_CALUDE_triangles_forming_square_even_l2031_203115

theorem triangles_forming_square_even (n : ℕ) (a : ℕ) : 
  (n * 6 = a * a) → Even n := by sorry

end NUMINAMATH_CALUDE_triangles_forming_square_even_l2031_203115


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2031_203129

/-- Represents the shaded area calculation problem on a grid with circles --/
theorem shaded_area_calculation (grid_size : ℕ) (small_circle_radius : ℝ) (large_circle_radius : ℝ) 
  (small_circle_count : ℕ) (large_circle_count : ℕ) :
  grid_size = 6 ∧ 
  small_circle_radius = 0.5 ∧ 
  large_circle_radius = 1 ∧
  small_circle_count = 4 ∧
  large_circle_count = 2 →
  ∃ (A C : ℝ), 
    (A - C * Real.pi = grid_size^2 - (small_circle_count * small_circle_radius^2 + large_circle_count * large_circle_radius^2) * Real.pi) ∧
    A + C = 39 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2031_203129


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l2031_203126

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  (Real.tan ((150 : ℝ) - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) ∧
  x = 120 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l2031_203126


namespace NUMINAMATH_CALUDE_sector_area_l2031_203120

/-- The area of a sector with central angle 150° and radius 3 is 15π/4 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 150 * π / 180) (h2 : r = 3) :
  (1/2) * r^2 * θ = 15 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2031_203120


namespace NUMINAMATH_CALUDE_sin_2alpha_equals_3_5_l2031_203150

theorem sin_2alpha_equals_3_5 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : 
  Real.sin (2*α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_sin_2alpha_equals_3_5_l2031_203150


namespace NUMINAMATH_CALUDE_base_3_minus_base_8_digits_of_2048_l2031_203103

theorem base_3_minus_base_8_digits_of_2048 : 
  (Nat.log 3 2048 + 1) - (Nat.log 8 2048 + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_3_minus_base_8_digits_of_2048_l2031_203103


namespace NUMINAMATH_CALUDE_alphametic_puzzle_solution_l2031_203107

theorem alphametic_puzzle_solution :
  ∃! (A R K : Nat),
    A < 10 ∧ R < 10 ∧ K < 10 ∧
    A ≠ R ∧ A ≠ K ∧ R ≠ K ∧
    1000 * A + 100 * R + 10 * K + A +
    100 * R + 10 * K + A +
    10 * K + A +
    A = 2014 ∧
    A = 1 ∧ R = 4 ∧ K = 7 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_solution_l2031_203107


namespace NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l2031_203136

/-- Calculates the total time to assemble furniture -/
def total_assembly_time (chairs : ℕ) (tables : ℕ) (time_per_piece : ℕ) : ℕ :=
  (chairs + tables) * time_per_piece

/-- Theorem: The total assembly time for Rachel's furniture -/
theorem rachel_furniture_assembly_time :
  total_assembly_time 7 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l2031_203136


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2031_203101

theorem unique_four_digit_number :
  ∃! x : ℕ, 
    1000 ≤ x ∧ x < 10000 ∧
    x + (x % 10) = 5574 ∧
    x + ((x / 10) % 10) = 557 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2031_203101


namespace NUMINAMATH_CALUDE_walnut_trees_remaining_l2031_203155

/-- The number of walnut trees remaining after some are cut down -/
def remaining_walnut_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining walnut trees is 29 -/
theorem walnut_trees_remaining :
  remaining_walnut_trees 42 13 = 29 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_remaining_l2031_203155


namespace NUMINAMATH_CALUDE_square_table_correctness_l2031_203167

/-- Converts a base 60 number to base 10 -/
def base60ToBase10 (x : List Nat) : Nat :=
  x.enum.foldl (fun acc (i, digit) => acc + digit * (60 ^ i)) 0

/-- Converts a base 10 number to base 60 -/
def base10ToBase60 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 60) ((m % 60) :: acc)
    aux n []

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Represents the table of squares in base 60 -/
def squareTable : Nat → List Nat := sorry

theorem square_table_correctness :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 60 →
    base60ToBase10 (squareTable n) = n * n ∧
    isPerfectSquare (base60ToBase10 (squareTable n)) := by
  sorry

end NUMINAMATH_CALUDE_square_table_correctness_l2031_203167


namespace NUMINAMATH_CALUDE_triple_root_at_zero_l2031_203177

/-- The polynomial representing the difference between the two functions -/
def P (a b c d m n : ℝ) (x : ℝ) : ℝ :=
  x^7 - 9*x^6 + 27*x^5 + a*x^4 + b*x^3 + c*x^2 + d*x - m*x - n

/-- Theorem stating that the polynomial has a triple root at x = 0 -/
theorem triple_root_at_zero (a b c d m n : ℝ) : 
  ∃ (p q : ℝ), p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧
  ∀ (x : ℝ), P a b c d m n x = (x - p)^2 * (x - q)^2 * x^3 :=
sorry

end NUMINAMATH_CALUDE_triple_root_at_zero_l2031_203177


namespace NUMINAMATH_CALUDE_billys_age_l2031_203125

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l2031_203125


namespace NUMINAMATH_CALUDE_complex_real_implies_m_equals_five_l2031_203108

theorem complex_real_implies_m_equals_five (m : ℝ) (z : ℂ) :
  z = Complex.I * (m^2 - 2*m - 15) → z.im = 0 → m = 5 := by sorry

end NUMINAMATH_CALUDE_complex_real_implies_m_equals_five_l2031_203108


namespace NUMINAMATH_CALUDE_square_diagonal_and_area_l2031_203143

/-- Given a square with side length 30√3 cm, this theorem proves the length of its diagonal and its area. -/
theorem square_diagonal_and_area :
  let side_length : ℝ := 30 * Real.sqrt 3
  let diagonal : ℝ := side_length * Real.sqrt 2
  let area : ℝ := side_length ^ 2
  diagonal = 30 * Real.sqrt 6 ∧ area = 2700 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_and_area_l2031_203143


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2031_203182

/-- The complex number z = (1-3i)(2+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant : 
  let z : ℂ := (1 - 3*Complex.I) * (2 + Complex.I)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2031_203182


namespace NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l2031_203176

/-- The number of rectangles on a 6x6 checkerboard -/
def num_rectangles : ℕ := 441

/-- The number of squares on a 6x6 checkerboard -/
def num_squares : ℕ := 91

/-- Theorem stating that the ratio of squares to rectangles on a 6x6 checkerboard is 13/63 -/
theorem squares_to_rectangles_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = 13 / 63 := by sorry

end NUMINAMATH_CALUDE_squares_to_rectangles_ratio_l2031_203176


namespace NUMINAMATH_CALUDE_inverse_proportion_wrench_force_l2031_203168

/-- Proof that for inversely proportional quantities, if F₁ * L₁ = k and F₂ * L₂ = k,
    where F₁ = 300, L₁ = 12, and L₂ = 18, then F₂ = 200. -/
theorem inverse_proportion_wrench_force (k : ℝ) (F₁ F₂ L₁ L₂ : ℝ) 
    (h1 : F₁ * L₁ = k)
    (h2 : F₂ * L₂ = k)
    (h3 : F₁ = 300)
    (h4 : L₁ = 12)
    (h5 : L₂ = 18) :
    F₂ = 200 := by
  sorry

#check inverse_proportion_wrench_force

end NUMINAMATH_CALUDE_inverse_proportion_wrench_force_l2031_203168


namespace NUMINAMATH_CALUDE_problem_statement_l2031_203106

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then (x - 1)^3 else |x - 1|

theorem problem_statement :
  (∃ a : ℝ, ∀ y : ℝ, ∃ x : ℝ, f a x < y) ∧
  (∀ a : ℝ, ∃ x : ℝ, f a x = 0) ∧
  (∀ a : ℝ, a > 1 → a < 2 → ∃ m : ℝ, ∃ x₁ x₂ x₃ : ℝ,
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2031_203106


namespace NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l2031_203154

/-- Represents the profit percentage calculation for a shopkeeper's sale --/
theorem shopkeeper_profit_calculation 
  (cost_price : ℝ) 
  (discount_percent : ℝ) 
  (profit_with_discount : ℝ) 
  (h_positive_cp : cost_price > 0)
  (h_discount : discount_percent = 5)
  (h_profit : profit_with_discount = 20.65) :
  let selling_price_with_discount := cost_price * (1 - discount_percent / 100)
  let selling_price_no_discount := cost_price * (1 + profit_with_discount / 100)
  let profit_no_discount := (selling_price_no_discount - cost_price) / cost_price * 100
  profit_no_discount = profit_with_discount := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l2031_203154


namespace NUMINAMATH_CALUDE_apples_in_shop_l2031_203183

/-- Given a ratio of fruits and the number of mangoes, calculate the number of apples -/
def calculate_apples (mango_ratio : ℕ) (orange_ratio : ℕ) (apple_ratio : ℕ) (mango_count : ℕ) : ℕ :=
  (mango_count / mango_ratio) * apple_ratio

/-- Theorem: Given the ratio 10:2:3 for mangoes:oranges:apples and 120 mangoes, there are 36 apples -/
theorem apples_in_shop :
  calculate_apples 10 2 3 120 = 36 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_shop_l2031_203183


namespace NUMINAMATH_CALUDE_probability_second_science_question_l2031_203188

/-- Given a set of questions with science and humanities questions,
    prove the probability of drawing a second science question
    after drawing a science question first. -/
theorem probability_second_science_question
  (total_questions : ℕ)
  (science_questions : ℕ)
  (humanities_questions : ℕ)
  (h1 : total_questions = 6)
  (h2 : science_questions = 4)
  (h3 : humanities_questions = 2)
  (h4 : total_questions = science_questions + humanities_questions)
  (h5 : science_questions > 0) :
  (science_questions - 1 : ℚ) / (total_questions - 1) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_probability_second_science_question_l2031_203188


namespace NUMINAMATH_CALUDE_a_minus_b_equals_15_l2031_203130

/-- Represents the division of money among A, B, and C -/
structure MoneyDivision where
  a : ℝ  -- Amount received by A
  b : ℝ  -- Amount received by B
  c : ℝ  -- Amount received by C

/-- Conditions for the money division problem -/
def validDivision (d : MoneyDivision) : Prop :=
  d.a = (1/3) * (d.b + d.c) ∧
  d.b = (2/7) * (d.a + d.c) ∧
  d.a > d.b ∧
  d.a + d.b + d.c = 540

/-- Theorem stating that A receives $15 more than B -/
theorem a_minus_b_equals_15 (d : MoneyDivision) (h : validDivision d) :
  d.a - d.b = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_15_l2031_203130


namespace NUMINAMATH_CALUDE_tan_inequality_l2031_203178

theorem tan_inequality (n : ℕ) (x : ℝ) (h1 : 0 < x) (h2 : x < π / (2 * n)) :
  (1/2) * (Real.tan x + Real.tan (n * x) - Real.tan ((n - 1) * x)) > (1/n) * Real.tan (n * x) := by
  sorry

end NUMINAMATH_CALUDE_tan_inequality_l2031_203178


namespace NUMINAMATH_CALUDE_flight_time_theorem_l2031_203116

/-- Represents the flight time between two towns -/
structure FlightTime where
  against_wind : ℝ
  with_wind : ℝ
  no_wind : ℝ

/-- The flight time satisfies the given conditions -/
def satisfies_conditions (ft : FlightTime) : Prop :=
  ft.against_wind = 84 ∧ ft.with_wind = ft.no_wind - 9

/-- The theorem to be proved -/
theorem flight_time_theorem (ft : FlightTime) 
  (h : satisfies_conditions ft) : 
  ft.with_wind = 63 ∨ ft.with_wind = 12 := by
  sorry

end NUMINAMATH_CALUDE_flight_time_theorem_l2031_203116


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_2800_l2031_203105

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_of_2800 :
  largest_perfect_square_factor 2800 = 400 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_2800_l2031_203105


namespace NUMINAMATH_CALUDE_square_fence_poles_l2031_203138

/-- Given a square fence with a total of 104 poles, prove that the number of poles on each side is 26. -/
theorem square_fence_poles (total_poles : ℕ) (h1 : total_poles = 104) :
  ∃ (side_poles : ℕ), side_poles * 4 = total_poles ∧ side_poles = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_poles_l2031_203138


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2031_203179

/-- Line equation: ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  a * x + y - 2 = 0

/-- Circle equation: (x - 1)^2 + (y - a)^2 = 16/3 -/
def circle_equation (a x y : ℝ) : Prop :=
  (x - 1)^2 + (y - a)^2 = 16/3

/-- Circle center: C(1, a) -/
def circle_center (a : ℝ) : ℝ × ℝ :=
  (1, a)

/-- Triangle ABC is equilateral -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

theorem line_circle_intersection (a : ℝ) :
  ∃ A B : ℝ × ℝ,
    line_equation a A.1 A.2 ∧
    line_equation a B.1 B.2 ∧
    circle_equation a A.1 A.2 ∧
    circle_equation a B.1 B.2 ∧
    is_equilateral_triangle A B (circle_center a) →
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2031_203179


namespace NUMINAMATH_CALUDE_exists_digit_sum_div_11_l2031_203123

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 consecutive natural numbers, there is always at least one number
    whose sum of digits is divisible by 11. -/
theorem exists_digit_sum_div_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digit_sum (n + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_exists_digit_sum_div_11_l2031_203123


namespace NUMINAMATH_CALUDE_vincent_book_cost_l2031_203114

/-- Calculates the total cost of Vincent's books --/
def total_cost (animal_books train_books history_books cooking_books : ℕ) 
  (animal_price outer_space_price train_price history_price cooking_price : ℕ) : ℕ :=
  animal_books * animal_price + 
  1 * outer_space_price + 
  train_books * train_price + 
  history_books * history_price + 
  cooking_books * cooking_price

/-- Theorem stating that Vincent's total book cost is $356 --/
theorem vincent_book_cost : 
  total_cost 10 3 5 2 16 20 14 18 22 = 356 := by
  sorry


end NUMINAMATH_CALUDE_vincent_book_cost_l2031_203114


namespace NUMINAMATH_CALUDE_fraction_ordering_l2031_203104

theorem fraction_ordering : (4 : ℚ) / 17 < 6 / 25 ∧ 6 / 25 < 8 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2031_203104


namespace NUMINAMATH_CALUDE_scientific_notation_75500000_l2031_203196

theorem scientific_notation_75500000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 75500000 = a * (10 : ℝ) ^ n ∧ a = 7.55 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_75500000_l2031_203196


namespace NUMINAMATH_CALUDE_inequality_proof_l2031_203140

theorem inequality_proof (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) :
  Real.exp x₂ * Real.log x₁ < Real.exp x₁ * Real.log x₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2031_203140


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2031_203146

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2031_203146


namespace NUMINAMATH_CALUDE_conic_is_parabola_l2031_203156

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + y^2)

-- Define what it means for an equation to describe a parabola
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ) (h : a ≠ 0), 
    ∀ x y, f x y ↔ y = a * x^2 + b * x + c ∨ x = a * y^2 + b * y + d

-- Theorem statement
theorem conic_is_parabola : is_parabola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l2031_203156


namespace NUMINAMATH_CALUDE_equation_solution_for_all_y_l2031_203172

theorem equation_solution_for_all_y :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 4 * x - 6 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_for_all_y_l2031_203172


namespace NUMINAMATH_CALUDE_swim_team_total_l2031_203148

theorem swim_team_total (girls : ℕ) (boys : ℕ) : 
  girls = 80 → girls = 5 * boys → girls + boys = 96 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_total_l2031_203148


namespace NUMINAMATH_CALUDE_intersection_M_N_l2031_203133

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | x - 1 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2031_203133


namespace NUMINAMATH_CALUDE_pencil_count_l2031_203152

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 2

/-- The number of pencils Tim added to the drawer -/
def added_pencils : ℕ := 3

/-- The total number of pencils in the drawer after Tim's action -/
def total_pencils : ℕ := initial_pencils + added_pencils

theorem pencil_count : total_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2031_203152


namespace NUMINAMATH_CALUDE_quadratic_with_rational_roots_has_even_coefficient_l2031_203132

theorem quadratic_with_rational_roots_has_even_coefficient
  (a b c : ℕ+) -- a, b, c are positive integers
  (h_rational_roots : ∃ (p q r s : ℤ), (p * r ≠ 0 ∧ q * s ≠ 0) ∧
    (a * (p * s)^2 + b * (p * s) * (q * r) + c * (q * r)^2 = 0)) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_with_rational_roots_has_even_coefficient_l2031_203132


namespace NUMINAMATH_CALUDE_cookies_problem_l2031_203180

/-- Calculates the number of cookies taken out in four days given the initial count,
    remaining count after a week, and assuming equal daily removal. -/
def cookies_taken_in_four_days (initial : ℕ) (remaining : ℕ) : ℕ :=
  let total_taken := initial - remaining
  let daily_taken := total_taken / 7
  4 * daily_taken

/-- Proves that given 70 initial cookies and 28 remaining after a week,
    Paul took out 24 cookies in four days. -/
theorem cookies_problem :
  cookies_taken_in_four_days 70 28 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookies_problem_l2031_203180


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l2031_203193

/-- Given a geometric sequence with first term 120, second term b, and third term 60/24,
    prove that b = 10√3 when b is positive. -/
theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
    (h2 : ∃ (r : ℝ), 120 * r = b ∧ b * r = 60 / 24) : b = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l2031_203193


namespace NUMINAMATH_CALUDE_king_not_right_mind_queen_indeterminate_l2031_203192

-- Define the mental states
inductive MentalState
| RightMind
| NotRightMind

-- Define the royals
structure Royal where
  name : String
  state : MentalState

-- Define the belief function
def believes (r : Royal) (p : Prop) : Prop := sorry

-- Define the King and Queen of Spades
def King : Royal := ⟨"King of Spades", MentalState.NotRightMind⟩
def Queen : Royal := ⟨"Queen of Spades", MentalState.NotRightMind⟩

-- The main theorem
theorem king_not_right_mind_queen_indeterminate :
  believes Queen (believes King (Queen.state = MentalState.NotRightMind)) →
  (King.state = MentalState.NotRightMind) ∧
  ((Queen.state = MentalState.RightMind) ∨ (Queen.state = MentalState.NotRightMind)) :=
by sorry

end NUMINAMATH_CALUDE_king_not_right_mind_queen_indeterminate_l2031_203192


namespace NUMINAMATH_CALUDE_sector_max_area_l2031_203110

/-- Given a sector with circumference 12 cm, its maximum area is 9 cm². -/
theorem sector_max_area (r l : ℝ) (h_circumference : 2 * r + l = 12) :
  (1/2 : ℝ) * l * r ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l2031_203110


namespace NUMINAMATH_CALUDE_range_of_a_l2031_203185

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ y : ℝ, y ≥ a ∧ ¬(|y - 1| < 1)) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2031_203185


namespace NUMINAMATH_CALUDE_angle_between_a_and_b_l2031_203171

/-- The angle between two 3D vectors -/
def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ := by sorry

/-- The vector a -/
def a : ℝ × ℝ × ℝ := (1, 1, -4)

/-- The vector b -/
def b : ℝ × ℝ × ℝ := (1, -2, 2)

/-- The theorem stating that the angle between vectors a and b is 135 degrees -/
theorem angle_between_a_and_b : 
  angle_between_vectors a b = 135 * Real.pi / 180 := by sorry

end NUMINAMATH_CALUDE_angle_between_a_and_b_l2031_203171


namespace NUMINAMATH_CALUDE_maximize_fruit_yield_l2031_203145

/-- Maximizing fruit yield in an orchard --/
theorem maximize_fruit_yield (x : ℝ) :
  let initial_trees : ℝ := 100
  let initial_yield_per_tree : ℝ := 600
  let yield_decrease_per_tree : ℝ := 5
  let total_trees : ℝ := x + initial_trees
  let new_yield_per_tree : ℝ := initial_yield_per_tree - yield_decrease_per_tree * x
  let total_yield : ℝ := total_trees * new_yield_per_tree
  (∀ z : ℝ, total_yield ≥ (z + initial_trees) * (initial_yield_per_tree - yield_decrease_per_tree * z)) →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_maximize_fruit_yield_l2031_203145


namespace NUMINAMATH_CALUDE_percentage_decrease_l2031_203141

theorem percentage_decrease (w : ℝ) (x : ℝ) (h1 : w = 80) (h2 : w * (1 + 0.125) - w * (1 - x / 100) = 30) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l2031_203141


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_eight_l2031_203147

theorem sum_of_A_and_C_is_eight :
  ∀ (A B C D : ℕ),
    A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A : ℚ) / B - (C : ℚ) / D = 2 →
    A + C = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_A_and_C_is_eight_l2031_203147


namespace NUMINAMATH_CALUDE_store_discount_percentage_l2031_203158

/-- Proves that the discount percentage is 9% given the specified markups and profit -/
theorem store_discount_percentage (C : ℝ) (h : C > 0) : 
  let initial_price := 1.20 * C
  let marked_up_price := 1.25 * initial_price
  let final_profit := 0.365 * C
  ∃ (D : ℝ), 
    marked_up_price * (1 - D) - C = final_profit ∧ 
    D = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l2031_203158


namespace NUMINAMATH_CALUDE_range_of_g_l2031_203142

-- Define the function f(x) = |x|
def f (x : ℝ) : ℝ := |x|

-- Define the domain
def domain : Set ℝ := {x | -4 ≤ x ∧ x ≤ 4}

-- Define the function g(x) = f(x) - x
def g (x : ℝ) : ℝ := f x - x

-- Theorem statement
theorem range_of_g :
  {y | ∃ x ∈ domain, g x = y} = {y | 0 ≤ y ∧ y ≤ 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2031_203142


namespace NUMINAMATH_CALUDE_number_ordering_l2031_203118

theorem number_ordering (a b c : ℝ) : 
  a = 9^(1/3) → b = 3^(2/5) → c = 4^(1/5) → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2031_203118


namespace NUMINAMATH_CALUDE_square_area_error_l2031_203163

theorem square_area_error (s : ℝ) (s' : ℝ) (h : s' = s * (1 + 0.02)) :
  (s'^2 - s^2) / s^2 * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l2031_203163


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2031_203139

-- Define the quadratic function
def f (a b x : ℝ) := x^2 + b*x + a

-- State the theorem
theorem quadratic_inequality_solution (a b : ℝ) 
  (h : ∀ x, f a b x > 0 ↔ x ∈ Set.Iio 1 ∪ Set.Ioi 5) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2031_203139


namespace NUMINAMATH_CALUDE_hall_dimension_difference_l2031_203109

/-- Represents the dimensions and volume of a rectangular hall -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- The width is half the length, the height is one-third of the width, 
    and the volume is 600 cubic meters -/
def hall_constraints (hall : RectangularHall) : Prop :=
  hall.width = hall.length / 2 ∧
  hall.height = hall.width / 3 ∧
  hall.volume = 600

/-- The theorem stating the difference between length, width, and height -/
theorem hall_dimension_difference (hall : RectangularHall) 
  (h : hall_constraints hall) : 
  ∃ ε > 0, |hall.length - hall.width - hall.height - 6.43| < ε :=
sorry

end NUMINAMATH_CALUDE_hall_dimension_difference_l2031_203109


namespace NUMINAMATH_CALUDE_shaded_area_equals_36_plus_18pi_l2031_203131

-- Define the circle and its properties
def circle_radius : ℝ := 6

-- Define the triangles and their properties
def triangle_OAC_isosceles_right : Prop := sorry
def triangle_OBD_right : Prop := sorry

-- Define the areas
def area_triangle_OAC : ℝ := sorry
def area_triangle_OBD : ℝ := sorry
def area_sector_OAB : ℝ := sorry
def area_sector_OCD : ℝ := sorry

-- Theorem statement
theorem shaded_area_equals_36_plus_18pi :
  triangle_OAC_isosceles_right →
  triangle_OBD_right →
  area_triangle_OAC + area_triangle_OBD + area_sector_OAB + area_sector_OCD = 36 + 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_equals_36_plus_18pi_l2031_203131


namespace NUMINAMATH_CALUDE_f_properties_l2031_203159

def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

theorem f_properties (m : ℝ) :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x m > f y m) ∧
  (∃ x₀ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x m ≤ f x₀ m) ∧
  f x₀ m = 20 →
  ∃ x₁ ∈ Set.Icc (-2) 2, ∀ x ∈ Set.Icc (-2) 2, f x m ≥ f x₁ m ∧ f x₁ m = -7 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2031_203159


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2031_203144

/-- 
Given a rectangular plot where:
- The area is 18 times the breadth
- The length is 10 meters more than the breadth
Prove that the breadth is 8 meters
-/
theorem rectangular_plot_breadth (b : ℝ) (l : ℝ) (A : ℝ) : 
  A = 18 * b →
  l = b + 10 →
  A = l * b →
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2031_203144


namespace NUMINAMATH_CALUDE_max_d_is_401_l2031_203121

/-- The sequence a_n defined as n^2 + 100 -/
def a (n : ℕ+) : ℕ := n^2 + 100

/-- The sequence d_n defined as the gcd of a_n and a_{n+1} -/
def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating that the maximum value of d_n is 401 -/
theorem max_d_is_401 : ∃ (n : ℕ+), d n = 401 ∧ ∀ (m : ℕ+), d m ≤ 401 := by
  sorry

end NUMINAMATH_CALUDE_max_d_is_401_l2031_203121


namespace NUMINAMATH_CALUDE_rowing_speed_in_still_water_l2031_203117

/-- The speed of a man rowing in still water, given downstream conditions -/
theorem rowing_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 8.5)
  (h2 : distance = 45.5)
  (h3 : time = 9.099272058235341)
  : ∃ (still_water_speed : ℝ), still_water_speed = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_in_still_water_l2031_203117


namespace NUMINAMATH_CALUDE_carousel_horses_l2031_203124

theorem carousel_horses (blue purple green gold : ℕ) : 
  purple = 3 * blue →
  green = 2 * purple →
  gold = green / 6 →
  blue + purple + green + gold = 33 →
  blue = 3 := by
sorry

end NUMINAMATH_CALUDE_carousel_horses_l2031_203124


namespace NUMINAMATH_CALUDE_arithmetic_sequence_characterization_l2031_203100

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem arithmetic_sequence_characterization (a : ℕ+ → ℝ) :
  is_arithmetic_sequence a ↔ ∀ n : ℕ+, 2 * a (n + 1) = a n + a (n + 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_characterization_l2031_203100


namespace NUMINAMATH_CALUDE_incorrect_inequality_l2031_203166

theorem incorrect_inequality (m n : ℝ) (h : m > n) : ¬(-2 * m > -2 * n) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l2031_203166


namespace NUMINAMATH_CALUDE_f_monotonicity_and_positivity_l2031_203102

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x - k * Real.log x

-- State the theorem
theorem f_monotonicity_and_positivity (k : ℝ) (h_k : k > 0) :
  (∀ x > k, ∀ y > k, x < y → f k x < f k y) ∧ 
  (∀ x ∈ Set.Ioo 0 k, ∀ y ∈ Set.Ioo 0 k, x < y → f k x > f k y) ∧
  (∀ x ≥ 1, f k x > 0) → 
  0 < k ∧ k < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_positivity_l2031_203102


namespace NUMINAMATH_CALUDE_notebook_duration_example_l2031_203169

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, using 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_duration_example_l2031_203169


namespace NUMINAMATH_CALUDE_tangent_line_coincidence_l2031_203189

/-- Given a differentiable function f where the tangent line of y = f(x) at (0,0) 
    coincides with the tangent line of y = f(x)/x at (2,1), prove that f'(2) = 2 -/
theorem tangent_line_coincidence (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, x ≠ 0 → (f x) / x = ((f 0) + (deriv f 0) * x)) →
  (f 2) / 2 = 1 →
  deriv f 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_coincidence_l2031_203189


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l2031_203134

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |> List.sum

theorem units_digit_sum_factorials_500 :
  units_digit (sum_factorials 500) = units_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4) :=
by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_500_l2031_203134


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2031_203199

/-- A geometric sequence with first term x, second term 3x+3, and third term 6x+6 has fourth term -24 -/
theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  let a₁ : ℝ := x
  let a₂ : ℝ := 3*x + 3
  let a₃ : ℝ := 6*x + 6
  let r : ℝ := a₂ / a₁
  (a₂ = r * a₁) ∧ (a₃ = r * a₂) →
  r * a₃ = -24 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2031_203199


namespace NUMINAMATH_CALUDE_gumdrop_cost_l2031_203161

/-- Given 80 cents to buy 20 gumdrops, prove that each gumdrop costs 4 cents. -/
theorem gumdrop_cost (total_money : ℕ) (num_gumdrops : ℕ) (cost_per_gumdrop : ℕ) :
  total_money = 80 ∧ num_gumdrops = 20 ∧ total_money = num_gumdrops * cost_per_gumdrop →
  cost_per_gumdrop = 4 := by
sorry

end NUMINAMATH_CALUDE_gumdrop_cost_l2031_203161


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l2031_203181

/-- The volume of a rectangular solid with side lengths 1 m, 20 cm, and 50 cm is 100000 cm³ -/
theorem rectangular_solid_volume : 
  let length_m : ℝ := 1
  let width_cm : ℝ := 20
  let height_cm : ℝ := 50
  let m_to_cm : ℝ := 100
  (length_m * m_to_cm * width_cm * height_cm) = 100000 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l2031_203181


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2031_203151

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 + 2 * p^3 - 7 * p^2 + 3 * p - 2) + (-3 * p^4 + 4 * p^3 + 8 * p^2 - 2 * p + 6) =
  2 * p^4 + 6 * p^3 + p^2 + p + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2031_203151


namespace NUMINAMATH_CALUDE_odd_function_with_period_two_negation_at_six_l2031_203160

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_negation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem odd_function_with_period_two_negation_at_six
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period_two_negation f) :
  f 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_with_period_two_negation_at_six_l2031_203160


namespace NUMINAMATH_CALUDE_periodic_function_theorem_l2031_203191

/-- A function f: ℝ → ℝ is periodic if there exists a positive real number p such that
    for all x ∈ ℝ, f(x + p) = f(x) -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- The main theorem: if f satisfies the given functional equation,
    then f is periodic with period 2a -/
theorem periodic_function_theorem (f : ℝ → ℝ) (a : ℝ) 
    (h : a > 0) 
    (eq : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  IsPeriodic f ∧ ∃ p : ℝ, p = 2 * a ∧ ∀ x : ℝ, f (x + p) = f x :=
by sorry

end NUMINAMATH_CALUDE_periodic_function_theorem_l2031_203191


namespace NUMINAMATH_CALUDE_largest_bundle_size_correct_l2031_203149

def largest_bundle_size (john_notebooks emily_notebooks min_bundle_size : ℕ) : ℕ :=
  Nat.gcd john_notebooks emily_notebooks

theorem largest_bundle_size_correct 
  (john_notebooks : ℕ) 
  (emily_notebooks : ℕ) 
  (min_bundle_size : ℕ) 
  (h1 : john_notebooks = 36) 
  (h2 : emily_notebooks = 45) 
  (h3 : min_bundle_size = 5) :
  largest_bundle_size john_notebooks emily_notebooks min_bundle_size = 9 ∧ 
  largest_bundle_size john_notebooks emily_notebooks min_bundle_size > min_bundle_size := by
  sorry

#eval largest_bundle_size 36 45 5

end NUMINAMATH_CALUDE_largest_bundle_size_correct_l2031_203149


namespace NUMINAMATH_CALUDE_percentage_problem_l2031_203186

theorem percentage_problem (P : ℝ) (x : ℝ) (h1 : x = 264) (h2 : (P / 100) * x = (1 / 3) * x + 110) : P = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2031_203186


namespace NUMINAMATH_CALUDE_g_negative_three_value_l2031_203137

theorem g_negative_three_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g (5 * x - 7) = 8 * x + 2) :
  g (-3) = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_three_value_l2031_203137


namespace NUMINAMATH_CALUDE_solution_existence_l2031_203165

/-- The set of real solutions (x, y) satisfying both equations -/
def SolutionSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 9 = 0 ∧ p.1^2 - 2*p.2 + 6 = 0}

/-- Theorem stating that real solutions exist if and only if y = -5 or y = 3 -/
theorem solution_existence : 
  ∃ (x : ℝ), (x, y) ∈ SolutionSet ↔ y = -5 ∨ y = 3 :=
sorry

end NUMINAMATH_CALUDE_solution_existence_l2031_203165


namespace NUMINAMATH_CALUDE_lamp_sales_theorem_l2031_203112

/-- The monthly average growth rate of lamp sales -/
def monthly_growth_rate : ℝ := 0.2

/-- The price of lamps in April to achieve the target profit -/
def april_price : ℝ := 38

/-- Initial sales volume in January -/
def january_sales : ℕ := 400

/-- Sales volume in March -/
def march_sales : ℕ := 576

/-- Purchase cost per lamp -/
def purchase_cost : ℝ := 30

/-- Initial selling price -/
def initial_price : ℝ := 40

/-- Increase in sales volume per 0.5 yuan price reduction -/
def sales_increase_per_half_yuan : ℕ := 6

/-- Target profit in April -/
def target_profit : ℝ := 4800

/-- Theorem stating the correctness of the monthly growth rate and April price -/
theorem lamp_sales_theorem :
  (january_sales * (1 + monthly_growth_rate)^2 = march_sales) ∧
  ((april_price - purchase_cost) *
    (march_sales + 2 * sales_increase_per_half_yuan * (initial_price - april_price)) = target_profit) := by
  sorry


end NUMINAMATH_CALUDE_lamp_sales_theorem_l2031_203112


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l2031_203170

/-- The amount of money Jerry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $72 to finish his collection -/
theorem jerry_collection_cost : money_needed 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l2031_203170


namespace NUMINAMATH_CALUDE_plan_A_cost_per_text_l2031_203173

/-- The cost per text message for Plan A, in dollars -/
def cost_per_text_A : ℝ := 0.25

/-- The monthly fee for Plan A, in dollars -/
def monthly_fee_A : ℝ := 9

/-- The cost per text message for Plan B, in dollars -/
def cost_per_text_B : ℝ := 0.40

/-- The number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem plan_A_cost_per_text :
  cost_per_text_A * equal_cost_messages + monthly_fee_A =
  cost_per_text_B * equal_cost_messages :=
by sorry

end NUMINAMATH_CALUDE_plan_A_cost_per_text_l2031_203173


namespace NUMINAMATH_CALUDE_pairball_playing_time_l2031_203127

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) (h1 : total_time = 120) (h2 : num_children = 6) : 
  (2 * total_time) / num_children = 40 :=
by sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l2031_203127


namespace NUMINAMATH_CALUDE_quinn_free_donuts_l2031_203111

/-- The number of free donuts Quinn is eligible for based on his summer reading --/
def free_donuts (books_per_donut : ℕ) (books_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  (books_per_week * num_weeks) / books_per_donut

/-- Theorem stating that Quinn is eligible for 4 free donuts --/
theorem quinn_free_donuts :
  free_donuts 5 2 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_free_donuts_l2031_203111


namespace NUMINAMATH_CALUDE_money_needed_for_perfume_l2031_203190

def perfume_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def yards_mowed : ℕ := 4
def yard_charge : ℕ := 5
def dogs_walked : ℕ := 6
def dog_charge : ℕ := 2

theorem money_needed_for_perfume :
  perfume_cost - (christian_initial + sue_initial + yards_mowed * yard_charge + dogs_walked * dog_charge) = 6 := by
  sorry

end NUMINAMATH_CALUDE_money_needed_for_perfume_l2031_203190


namespace NUMINAMATH_CALUDE_additional_surcharge_l2031_203198

/-- Calculates the additional surcharge for a special project given the tax information --/
theorem additional_surcharge (tax_1995 tax_1996 : ℕ) (increase_rate : ℚ) : 
  tax_1995 = 1800 →
  increase_rate = 6 / 100 →
  tax_1996 = 2108 →
  (tax_1996 : ℚ) = (tax_1995 : ℚ) * (1 + increase_rate) + 200 := by
  sorry

end NUMINAMATH_CALUDE_additional_surcharge_l2031_203198


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2031_203162

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_simplification :
  (5 - 3 * i)^2 = 16 - 30 * i :=
by
  -- The proof would go here, but we're skipping it as per instructions
  sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2031_203162


namespace NUMINAMATH_CALUDE_percentage_increase_l2031_203164

theorem percentage_increase (x : ℝ) (h1 : x = 14.4) (h2 : x > 12) :
  (x - 12) / 12 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2031_203164


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2031_203187

theorem inequality_system_solution_set
  (x : ℝ) :
  (2 * x ≤ -2 ∧ x + 3 < 4) ↔ x ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2031_203187


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2031_203128

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2031_203128


namespace NUMINAMATH_CALUDE_book_sale_result_l2031_203184

theorem book_sale_result (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  selling_price = 4.5 ∧ 
  profit_percent = 25 ∧ 
  loss_percent = 25 →
  (selling_price * 2) - (selling_price / (1 + profit_percent / 100) + selling_price / (1 - loss_percent / 100)) = -0.6 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_result_l2031_203184


namespace NUMINAMATH_CALUDE_parabola_focus_l2031_203175

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * y^2

-- Define the focus of a parabola
def focus (p q : ℝ) : Prop := ∀ x y : ℝ, parabola_equation x y → (x + 1)^2 + y^2 = 1

-- Theorem statement
theorem parabola_focus : focus (-1) 0 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2031_203175


namespace NUMINAMATH_CALUDE_water_depth_relationship_l2031_203174

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ
  horizontalWaterDepth : ℝ

/-- Calculates the water depth when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) : ℝ :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_relationship (tank : WaterTank) 
  (h : tank.height = 20 ∧ tank.baseDiameter = 6 ∧ tank.horizontalWaterDepth = 2) :
  abs (verticalWaterDepth tank - 7.0) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_relationship_l2031_203174


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_property_l2031_203113

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the property of a_4 and a_12 being roots of x^2 + 3x + 1 = 0
def roots_property (a : ℕ → ℝ) : Prop :=
  a 4 + a 12 = -3 ∧ a 4 * a 12 = 1

theorem geometric_sequence_roots_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (roots_property a → a 8 = -1) ∧
  ¬(a 8 = -1 → roots_property a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_property_l2031_203113
