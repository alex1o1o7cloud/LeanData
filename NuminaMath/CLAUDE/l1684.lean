import Mathlib

namespace NUMINAMATH_CALUDE_trip_expenses_l1684_168409

def david_initial : ℝ := 1800
def emma_initial : ℝ := 2400
def john_initial : ℝ := 1200

def david_spend_percent : ℝ := 0.60
def emma_spend_percent : ℝ := 0.75
def john_spend_percent : ℝ := 0.50

def david_remaining : ℝ := david_initial * (1 - david_spend_percent)
def emma_spent : ℝ := emma_initial * emma_spend_percent
def emma_remaining : ℝ := emma_spent - 800
def john_remaining : ℝ := john_initial * (1 - john_spend_percent)

theorem trip_expenses :
  david_remaining = 720 ∧
  emma_remaining = 1400 ∧
  john_remaining = 600 ∧
  emma_remaining = emma_spent - 800 :=
by sorry

end NUMINAMATH_CALUDE_trip_expenses_l1684_168409


namespace NUMINAMATH_CALUDE_tangent_circle_center_l1684_168459

/-- A circle passes through (0,3) and is tangent to y = x^2 at (1,1) -/
structure TangentCircle where
  center : ℝ × ℝ
  passes_through : center.1^2 + (center.2 - 3)^2 = (center.1 - 0)^2 + (center.2 - 3)^2
  tangent_at : center.1^2 + (center.2 - 1)^2 = (center.1 - 1)^2 + (center.2 - 1)^2
  on_parabola : 1 = 1^2

/-- The center of the circle is (0, 3/2) -/
theorem tangent_circle_center : ∀ c : TangentCircle, c.center = (0, 3/2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_center_l1684_168459


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l1684_168433

/-- Given vectors a and b in R², prove that if 2a + b is parallel to a - 2b, then m = -1/2 --/
theorem parallel_vectors_imply_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (2, -1)
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (a - 2 • b)) →
  m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l1684_168433


namespace NUMINAMATH_CALUDE_art_fair_customers_l1684_168487

theorem art_fair_customers (group1 group2 group3 : ℕ) 
  (paintings_per_customer1 paintings_per_customer2 paintings_per_customer3 : ℕ) 
  (total_paintings : ℕ) : 
  group1 = 4 → 
  group2 = 12 → 
  group3 = 4 → 
  paintings_per_customer1 = 2 → 
  paintings_per_customer2 = 1 → 
  paintings_per_customer3 = 4 → 
  total_paintings = 36 → 
  group1 * paintings_per_customer1 + 
  group2 * paintings_per_customer2 + 
  group3 * paintings_per_customer3 = total_paintings → 
  group1 + group2 + group3 = 20 := by
sorry

end NUMINAMATH_CALUDE_art_fair_customers_l1684_168487


namespace NUMINAMATH_CALUDE_inverse_sum_of_cube_function_l1684_168489

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum_of_cube_function :
  g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_of_cube_function_l1684_168489


namespace NUMINAMATH_CALUDE_doubled_added_tripled_l1684_168493

theorem doubled_added_tripled (y : ℝ) : 3 * (2 * 7 + y) = 69 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_doubled_added_tripled_l1684_168493


namespace NUMINAMATH_CALUDE_triangle_side_length_expression_l1684_168482

/-- For any triangle with side lengths a, b, and c, |a+b-c|-|a-b-c| = 2a-2c -/
theorem triangle_side_length_expression (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  |a + b - c| - |a - b - c| = 2*a - 2*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_expression_l1684_168482


namespace NUMINAMATH_CALUDE_tank_fill_time_AB_l1684_168404

/-- Given three valves A, B, and C that fill a tank, this theorem proves the time required
    to fill the tank with only valves A and B open, based on given filling times for
    different valve combinations. -/
theorem tank_fill_time_AB (fill_time_ABC : ℝ) (fill_time_AC : ℝ) (fill_time_BC : ℝ)
    (h1 : fill_time_ABC = 1)
    (h2 : fill_time_AC = 1.5)
    (h3 : fill_time_BC = 2) :
    ∃ (fill_time_AB : ℝ), fill_time_AB = 1.2 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_time_AB_l1684_168404


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l1684_168410

theorem subtraction_puzzle (A B : Nat) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  2000 + 100 * A + 32 - (100 * B + 10 * B + B) = 1000 + 100 * B + 10 * B + B → 
  B - A = 3 := by
sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l1684_168410


namespace NUMINAMATH_CALUDE_lcm_factor_42_l1684_168401

theorem lcm_factor_42 (A B : ℕ+) : 
  Nat.gcd A B = 42 → 
  max A B = 840 → 
  42 ∣ Nat.lcm A B :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_42_l1684_168401


namespace NUMINAMATH_CALUDE_lea_purchases_cost_l1684_168476

/-- The cost of a single book -/
def book_cost : ℕ := 16

/-- The cost of a single binder -/
def binder_cost : ℕ := 2

/-- The number of binders bought -/
def num_binders : ℕ := 3

/-- The cost of a single notebook -/
def notebook_cost : ℕ := 1

/-- The number of notebooks bought -/
def num_notebooks : ℕ := 6

/-- The total cost of Léa's purchases -/
def total_cost : ℕ := book_cost + (binder_cost * num_binders) + (notebook_cost * num_notebooks)

theorem lea_purchases_cost : total_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_lea_purchases_cost_l1684_168476


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l1684_168400

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l1684_168400


namespace NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l1684_168439

theorem sum_and_ratio_implies_difference (x y : ℝ) 
  (sum_eq : x + y = 540)
  (ratio_eq : x / y = 4 / 5) :
  y - x = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l1684_168439


namespace NUMINAMATH_CALUDE_tax_reduction_percentage_l1684_168486

/-- Given a commodity with tax and consumption, proves that if the tax is reduced by a certain percentage,
    consumption increases by 15%, and revenue decreases by 8%, then the tax reduction percentage is 20%. -/
theorem tax_reduction_percentage
  (T : ℝ) -- Original tax
  (C : ℝ) -- Original consumption
  (X : ℝ) -- Percentage by which tax is diminished
  (h1 : T > 0)
  (h2 : C > 0)
  (h3 : X ≥ 0)
  (h4 : X ≤ 100)
  (h5 : T * (1 - X / 100) * C * 1.15 = 0.92 * T * C) -- Revenue equation
  : X = 20 := by sorry

end NUMINAMATH_CALUDE_tax_reduction_percentage_l1684_168486


namespace NUMINAMATH_CALUDE_multiply_add_distribute_l1684_168412

theorem multiply_add_distribute : 57 * 33 + 13 * 33 = 2310 := by
  sorry

end NUMINAMATH_CALUDE_multiply_add_distribute_l1684_168412


namespace NUMINAMATH_CALUDE_rearrange_segments_sum_l1684_168402

theorem rearrange_segments_sum (a b : ℕ) : ∃ (f g : Fin 1961 → Fin 1961),
  ∀ (i : Fin 1961), ∃ (k : ℕ),
    (a + (f i : ℕ)) + (b + (g i : ℕ)) = k + i ∧ 
    k + 1961 > k + i ∧
    k + i ≥ k :=
sorry

end NUMINAMATH_CALUDE_rearrange_segments_sum_l1684_168402


namespace NUMINAMATH_CALUDE_seashells_needed_l1684_168449

def current_seashells : ℕ := 19
def goal_seashells : ℕ := 25

theorem seashells_needed : goal_seashells - current_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashells_needed_l1684_168449


namespace NUMINAMATH_CALUDE_line_through_parabola_focus_l1684_168480

/-- The focus of a parabola y² = 4x is the point (1, 0) -/
def focus_of_parabola : ℝ × ℝ := (1, 0)

/-- A line passing through a point (x, y) is represented by the equation ax - y + 1 = 0 -/
def line_passes_through (a : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 - p.2 + 1 = 0

theorem line_through_parabola_focus (a : ℝ) :
  line_passes_through a focus_of_parabola → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_parabola_focus_l1684_168480


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1684_168430

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1684_168430


namespace NUMINAMATH_CALUDE_distinct_roots_condition_roots_when_k_is_one_l1684_168415

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 + (2*k + 3)*x + k^2 + 5*k

-- Theorem for part 1
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   quadratic_equation k x = 0 ∧ 
   quadratic_equation k y = 0) →
  k < 9/8 :=
sorry

-- Theorem for part 2
theorem roots_when_k_is_one :
  quadratic_equation 1 (-2) = 0 ∧ 
  quadratic_equation 1 (-3) = 0 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_roots_when_k_is_one_l1684_168415


namespace NUMINAMATH_CALUDE_allen_reading_speed_l1684_168418

/-- The number of pages in Allen's book -/
def total_pages : ℕ := 120

/-- The number of days Allen took to read the book -/
def days_to_read : ℕ := 12

/-- The number of pages Allen read per day -/
def pages_per_day : ℕ := total_pages / days_to_read

/-- Theorem stating that Allen read 10 pages per day -/
theorem allen_reading_speed : pages_per_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_allen_reading_speed_l1684_168418


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1684_168473

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 4))) = 11 / 29 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1684_168473


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1684_168458

/-- Given a line with equation ax + by + c = 0, returns the equation of the line
    symmetric to it with respect to y = x as a triple (a', b', c') representing
    a'x + b'y + c' = 0 -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (b, a, c)

theorem symmetric_line_correct :
  let original_line := (1, -3, 5)  -- Represents x - 3y + 5 = 0
  let symm_line := symmetric_line 1 (-3) 5
  symm_line = (3, -1, -5)  -- Represents 3x - y - 5 = 0
  := by sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1684_168458


namespace NUMINAMATH_CALUDE_optimal_distribution_second_day_distribution_l1684_168453

/-- Represents a production line with its processing characteristics -/
structure ProductionLine where
  name : String
  process_time : ℝ → ℝ
  tonnage : ℝ

/-- The company with two production lines -/
structure Company where
  line_a : ProductionLine
  line_b : ProductionLine

/-- Defines the company with given production line characteristics -/
def our_company : Company :=
  { line_a := { name := "A", process_time := (λ a ↦ 4 * a + 1), tonnage := 0 },
    line_b := { name := "B", process_time := (λ b ↦ 2 * b + 3), tonnage := 0 } }

/-- Total raw materials allocated to both production lines -/
def total_raw_materials : ℝ := 5

/-- Theorem stating the optimal distribution of raw materials -/
theorem optimal_distribution (c : Company) (h : c = our_company) :
  ∃ (a b : ℝ),
    a + b = total_raw_materials ∧
    c.line_a.process_time a = c.line_b.process_time b ∧
    a = 2 ∧ b = 3 := by
  sorry

/-- Theorem stating the relationship between m and n for the second day -/
theorem second_day_distribution (m n : ℝ) (h : m + n = 6) :
  2 * m = n → m = 2 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_optimal_distribution_second_day_distribution_l1684_168453


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_proof_l1684_168497

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- The smallest prime whose digits sum to 20 -/
def smallest_prime_digit_sum_20 : ℕ := 299

theorem smallest_prime_digit_sum_20_proof :
  (digit_sum smallest_prime_digit_sum_20 = 20) ∧
  (is_prime smallest_prime_digit_sum_20) ∧
  (∀ n : ℕ, n < smallest_prime_digit_sum_20 →
    digit_sum n = 20 → ¬is_prime n) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_proof_l1684_168497


namespace NUMINAMATH_CALUDE_chemistry_question_ratio_l1684_168485

theorem chemistry_question_ratio 
  (total_multiple_choice : ℕ) 
  (total_problem_solving : ℕ) 
  (problem_solving_fraction_written : ℚ) 
  (remaining_questions : ℕ) : 
  total_multiple_choice = 35 →
  total_problem_solving = 15 →
  problem_solving_fraction_written = 1/3 →
  remaining_questions = 31 →
  (total_multiple_choice - remaining_questions + 
   (total_problem_solving - ⌊total_problem_solving * problem_solving_fraction_written⌋)) / 
   total_multiple_choice = 9/35 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_question_ratio_l1684_168485


namespace NUMINAMATH_CALUDE_eggs_donated_to_charity_l1684_168407

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs in a week --/
def collection_days : ℕ := 2

/-- Represents the number of dozen eggs Mortdecai collects per collection day --/
def eggs_collected_per_day : ℕ := 8

/-- Represents the number of dozen eggs Mortdecai delivers to the market --/
def eggs_to_market : ℕ := 3

/-- Represents the number of dozen eggs Mortdecai delivers to the mall --/
def eggs_to_mall : ℕ := 5

/-- Represents the number of dozen eggs Mortdecai uses for pie --/
def eggs_for_pie : ℕ := 4

/-- Theorem stating the number of eggs Mortdecai donates to charity --/
theorem eggs_donated_to_charity : 
  (collection_days * eggs_collected_per_day - (eggs_to_market + eggs_to_mall + eggs_for_pie)) * dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_eggs_donated_to_charity_l1684_168407


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l1684_168448

theorem tip_percentage_calculation (total_bill : ℝ) (billy_tip : ℝ) (billy_percentage : ℝ) :
  total_bill = 50 →
  billy_tip = 8 →
  billy_percentage = 0.8 →
  (billy_tip / billy_percentage) / total_bill = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l1684_168448


namespace NUMINAMATH_CALUDE_triangle_theorem_l1684_168424

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C

theorem triangle_theorem (t : Triangle) (h : condition t) :
  t.A = π / 3 ∧ (t.a = 4 → ∃ (max_area : ℝ), max_area = 4 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1684_168424


namespace NUMINAMATH_CALUDE_expression_evaluation_l1684_168413

theorem expression_evaluation : 80 + 5 * 12 / (180 / 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1684_168413


namespace NUMINAMATH_CALUDE_seconds_in_minutes_l1684_168452

/-- The number of seconds in one minute -/
def seconds_per_minute : ℝ := 60

/-- The number of minutes we want to convert to seconds -/
def minutes : ℝ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_minutes : minutes * seconds_per_minute = 750 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_minutes_l1684_168452


namespace NUMINAMATH_CALUDE_f_50_solutions_l1684_168447

-- Define f_0
def f_0 (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

-- Define f_n recursively
def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => f_0 x
  | n + 1 => |f n x| - 1

-- Theorem statement
theorem f_50_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f 50 x = 0) ∧ 
                    (∀ x ∉ S, f 50 x ≠ 0) ∧ 
                    Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_f_50_solutions_l1684_168447


namespace NUMINAMATH_CALUDE_total_players_on_ground_l1684_168445

/-- The number of cricket players -/
def cricket_players : ℕ := 15

/-- The number of hockey players -/
def hockey_players : ℕ := 12

/-- The number of football players -/
def football_players : ℕ := 13

/-- The number of softball players -/
def softball_players : ℕ := 15

/-- Theorem stating the total number of players on the ground -/
theorem total_players_on_ground :
  cricket_players + hockey_players + football_players + softball_players = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l1684_168445


namespace NUMINAMATH_CALUDE_place_mat_side_length_l1684_168437

theorem place_mat_side_length (r : ℝ) (n : ℕ) (x : ℝ) : 
  r = 5 →
  n = 8 →
  x = 2 * r * Real.sin (π / (2 * n)) →
  x = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_place_mat_side_length_l1684_168437


namespace NUMINAMATH_CALUDE_prove_necklace_sum_l1684_168438

def necklace_sum (H J x S : ℕ) : Prop :=
  (H = J + 5) ∧ 
  (x = J / 2) ∧ 
  (S = 2 * H) ∧ 
  (H = 25) →
  H + J + x + S = 105

theorem prove_necklace_sum : 
  ∀ (H J x S : ℕ), necklace_sum H J x S :=
by
  sorry

end NUMINAMATH_CALUDE_prove_necklace_sum_l1684_168438


namespace NUMINAMATH_CALUDE_total_cans_eq_319_l1684_168405

/-- The number of cans collected by five people given certain relationships between their collections. -/
def total_cans (solomon : ℕ) : ℕ :=
  let juwan := solomon / 3
  let levi := juwan / 2
  let gaby := (5 * solomon) / 2
  let michelle := gaby / 3
  solomon + juwan + levi + gaby + michelle

/-- Theorem stating that when Solomon collects 66 cans, the total number of cans collected by all five people is 319. -/
theorem total_cans_eq_319 : total_cans 66 = 319 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_eq_319_l1684_168405


namespace NUMINAMATH_CALUDE_lunch_spending_solution_l1684_168435

def lunch_spending (your_spending : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  (your_spending, 
   your_spending + 15, 
   your_spending - 20, 
   2 * your_spending)

theorem lunch_spending_solution : 
  ∃! (your_spending : ℚ), 
    let (you, friend1, friend2, friend3) := lunch_spending your_spending
    you + friend1 + friend2 + friend3 = 150 ∧
    friend1 = you + 15 ∧
    friend2 = you - 20 ∧
    friend3 = 2 * you :=
by
  sorry

#eval lunch_spending 31

end NUMINAMATH_CALUDE_lunch_spending_solution_l1684_168435


namespace NUMINAMATH_CALUDE_value_of_x_l1684_168484

theorem value_of_x (x y z : ℝ) 
  (h1 : x = y / 4)
  (h2 : y = z / 3)
  (h3 : z = 90) :
  x = 7.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l1684_168484


namespace NUMINAMATH_CALUDE_number_in_set_l1684_168456

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a 3-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a 3-digit number -/
def reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The theorem to be proved -/
theorem number_in_set (numbers : List ThreeDigitNumber) (reversed : ThreeDigitNumber) 
  (h_reversed : reversed ∈ numbers)
  (h_diff : reversed.units - reversed.hundreds = 2)
  (h_average_increase : (reversed_value reversed - value reversed : ℚ) / numbers.length = 198/10) :
  numbers.length = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_in_set_l1684_168456


namespace NUMINAMATH_CALUDE_phone_inventory_and_profit_optimization_l1684_168472

/-- Represents a phone model with purchase and selling prices -/
structure PhoneModel where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the inventory and financial data of a business hall -/
structure BusinessHall where
  modelA : PhoneModel
  modelB : PhoneModel
  totalSpent : ℕ
  totalProfit : ℕ

/-- Theorem stating the correct number of units purchased and maximum profit -/
theorem phone_inventory_and_profit_optimization 
  (hall : BusinessHall) 
  (hall_data : hall.modelA.purchasePrice = 3000 ∧ 
               hall.modelA.sellingPrice = 3400 ∧
               hall.modelB.purchasePrice = 3500 ∧ 
               hall.modelB.sellingPrice = 4000 ∧
               hall.totalSpent = 32000 ∧ 
               hall.totalProfit = 4400) :
  (∃ (a b : ℕ), 
    a * hall.modelA.purchasePrice + b * hall.modelB.purchasePrice = hall.totalSpent ∧
    a * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    b * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) = hall.totalProfit ∧
    a = 6 ∧ b = 4) ∧
  (∃ (x : ℕ), 
    x ≥ 10 ∧ 30 - x ≤ 2 * x ∧
    ∀ y : ℕ, y ≥ 10 → 30 - y ≤ 2 * y → 
    x * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - x) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) ≥
    y * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - y) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) ∧
    x * (hall.modelA.sellingPrice - hall.modelA.purchasePrice) + 
    (30 - x) * (hall.modelB.sellingPrice - hall.modelB.purchasePrice) = 14000) :=
by sorry

end NUMINAMATH_CALUDE_phone_inventory_and_profit_optimization_l1684_168472


namespace NUMINAMATH_CALUDE_sin_minus_cos_eq_one_l1684_168479

theorem sin_minus_cos_eq_one (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  (Real.sin x - Real.cos x = 1 ↔ x = Real.pi / 2 ∨ x = Real.pi) := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_eq_one_l1684_168479


namespace NUMINAMATH_CALUDE_exists_divisible_power_minus_one_l1684_168443

theorem exists_divisible_power_minus_one (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k > 0 ∧ k < n ∧ (n ∣ 2^k - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_power_minus_one_l1684_168443


namespace NUMINAMATH_CALUDE_three_cyclic_equations_l1684_168423

theorem three_cyclic_equations (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    a = x + 1/y ∧ a = y + 1/z ∧ a = z + 1/x) ↔ 
  (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_three_cyclic_equations_l1684_168423


namespace NUMINAMATH_CALUDE_linear_function_value_l1684_168461

theorem linear_function_value (k b : ℝ) :
  ((-1 : ℝ) * k + b = 1) →
  (2 * k + b = -2) →
  (1 : ℝ) * k + b = -1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l1684_168461


namespace NUMINAMATH_CALUDE_bus_passengers_l1684_168498

theorem bus_passengers (total_seats : ℕ) (net_increase : ℕ) (empty_seats : ℕ) :
  total_seats = 92 →
  net_increase = 19 →
  empty_seats = 57 →
  total_seats - empty_seats - net_increase = 16 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l1684_168498


namespace NUMINAMATH_CALUDE_probability_three_one_color_l1684_168427

/-- The probability of drawing 3 balls of one color and 1 of another color
    from a set of 20 balls (12 black, 8 white) when 4 are drawn at random -/
theorem probability_three_one_color (black_balls white_balls total_balls drawn : ℕ) 
  (h1 : black_balls = 12)
  (h2 : white_balls = 8)
  (h3 : total_balls = black_balls + white_balls)
  (h4 : drawn = 4) :
  (Nat.choose black_balls 3 * Nat.choose white_balls 1 +
   Nat.choose black_balls 1 * Nat.choose white_balls 3) / 
  Nat.choose total_balls drawn = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_three_one_color_l1684_168427


namespace NUMINAMATH_CALUDE_shannons_to_olivias_scoops_ratio_l1684_168481

/-- Represents the number of scoops in a carton of ice cream -/
def scoops_per_carton : ℕ := 10

/-- Represents the number of cartons Mary has -/
def marys_cartons : ℕ := 3

/-- Represents the number of scoops Ethan wants -/
def ethans_scoops : ℕ := 2

/-- Represents the number of scoops Lucas, Danny, and Connor want in total -/
def lucas_danny_connor_scoops : ℕ := 6

/-- Represents the number of scoops Olivia wants -/
def olivias_scoops : ℕ := 2

/-- Represents the number of scoops left -/
def scoops_left : ℕ := 16

/-- Theorem stating that the ratio of Shannon's scoops to Olivia's scoops is 2:1 -/
theorem shannons_to_olivias_scoops_ratio : 
  ∃ (shannons_scoops : ℕ), 
    shannons_scoops = marys_cartons * scoops_per_carton - 
      (ethans_scoops + lucas_danny_connor_scoops + olivias_scoops + scoops_left) ∧
    shannons_scoops = 2 * olivias_scoops :=
by sorry

end NUMINAMATH_CALUDE_shannons_to_olivias_scoops_ratio_l1684_168481


namespace NUMINAMATH_CALUDE_triangle_properties_l1684_168466

/-- Given a triangle ABC with acute angles A and B, prove the following:
    1. If ∠C = π/3 and c = 2, then 2 + 2√3 < perimeter ≤ 6
    2. If sin²A + sin²B > sin²C, then sin²A + sin²B > 1 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (C = π/3 ∧ c = 2 → 2 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c ≤ 6) ∧
  (Real.sin A ^ 2 + Real.sin B ^ 2 > Real.sin C ^ 2 → Real.sin A ^ 2 + Real.sin B ^ 2 > 1) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1684_168466


namespace NUMINAMATH_CALUDE_sixth_power_sum_l1684_168492

theorem sixth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 23)
  (h4 : a * x^4 + b * y^4 = 50)
  (h5 : a * x^5 + b * y^5 = 106) :
  a * x^6 + b * y^6 = 238 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l1684_168492


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1684_168421

theorem sqrt_inequality : Real.sqrt 7 - 1 > Real.sqrt 11 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1684_168421


namespace NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_six_l1684_168474

theorem closest_integer_to_two_plus_sqrt_six (x : ℝ) : 
  x = 2 + Real.sqrt 6 → 
  ∃ (n : ℕ), n = 4 ∧ ∀ (m : ℕ), m ≠ 4 → |x - ↑n| < |x - ↑m| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_six_l1684_168474


namespace NUMINAMATH_CALUDE_a_explicit_formula_l1684_168495

/-- Sequence {a_n} defined recursively --/
def a : ℕ → ℚ
  | 0 => 0
  | n + 1 => a n + (n + 1)^3

/-- Theorem stating the explicit formula for a_n --/
theorem a_explicit_formula (n : ℕ) : a n = n^2 * (n + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_a_explicit_formula_l1684_168495


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1684_168411

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) → c = 5 ∨ c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1684_168411


namespace NUMINAMATH_CALUDE_rectangle_Q_coordinates_l1684_168464

/-- A rectangle in a 2D plane --/
structure Rectangle where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The specific rectangle from the problem --/
def problemRectangle : Rectangle where
  O := (0, 0)
  P := (0, 3)
  R := (5, 0)
  Q := (5, 3)  -- We'll prove this is correct

/-- Predicate to check if four points form a rectangle --/
def isRectangle (rect : Rectangle) : Prop :=
  -- Opposite sides are parallel and equal in length
  (rect.O.1 = rect.P.1 ∧ rect.Q.1 = rect.R.1) ∧
  (rect.O.2 = rect.R.2 ∧ rect.P.2 = rect.Q.2) ∧
  (rect.P.1 - rect.O.1)^2 + (rect.P.2 - rect.O.2)^2 =
  (rect.Q.1 - rect.R.1)^2 + (rect.Q.2 - rect.R.2)^2 ∧
  (rect.R.1 - rect.O.1)^2 + (rect.R.2 - rect.O.2)^2 =
  (rect.Q.1 - rect.P.1)^2 + (rect.Q.2 - rect.P.2)^2

theorem rectangle_Q_coordinates :
  isRectangle problemRectangle →
  problemRectangle.Q = (5, 3) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_Q_coordinates_l1684_168464


namespace NUMINAMATH_CALUDE_jimmy_cookies_count_l1684_168488

/-- Given that:
  - Crackers contain 15 calories each
  - Cookies contain 50 calories each
  - Jimmy eats 10 crackers
  - Jimmy consumes a total of 500 calories
  Prove that Jimmy eats 7 cookies -/
theorem jimmy_cookies_count :
  let cracker_calories : ℕ := 15
  let cookie_calories : ℕ := 50
  let crackers_eaten : ℕ := 10
  let total_calories : ℕ := 500
  let cookies_eaten : ℕ := (total_calories - cracker_calories * crackers_eaten) / cookie_calories
  cookies_eaten = 7 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_cookies_count_l1684_168488


namespace NUMINAMATH_CALUDE_jinas_koala_bears_l1684_168478

theorem jinas_koala_bears :
  let initial_teddies : ℕ := 5
  let bunny_multiplier : ℕ := 3
  let additional_teddies_per_bunny : ℕ := 2
  let total_mascots : ℕ := 51
  let bunnies : ℕ := initial_teddies * bunny_multiplier
  let additional_teddies : ℕ := bunnies * additional_teddies_per_bunny
  let total_teddies : ℕ := initial_teddies + additional_teddies
  let koala_bears : ℕ := total_mascots - (total_teddies + bunnies)
  koala_bears = 1 := by
  sorry

end NUMINAMATH_CALUDE_jinas_koala_bears_l1684_168478


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1684_168477

/-- A type representing the colors of segments -/
inductive Color
| Red
| Blue

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A function type representing the coloring of segments -/
def Coloring := (Point × Point) → Color

/-- Theorem: Given 6 points in a plane with all segments colored either red or blue,
    there exists a triangle whose sides are all the same color -/
theorem monochromatic_triangle_exists (points : Fin 6 → Point) (coloring : Coloring) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (coloring (points i, points j) = coloring (points j, points k) ∧
     coloring (points j, points k) = coloring (points k, points i)) :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1684_168477


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1684_168414

theorem rectangle_area_change (initial_area : ℝ) : 
  initial_area = 540 →
  (0.8 * 1.15) * initial_area = 496.8 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1684_168414


namespace NUMINAMATH_CALUDE_company_production_l1684_168429

/-- The number of bottles produced daily by a company -/
def bottles_produced (cases_required : ℕ) (bottles_per_case : ℕ) : ℕ :=
  cases_required * bottles_per_case

/-- Theorem stating the company's daily water bottle production -/
theorem company_production :
  bottles_produced 10000 12 = 120000 := by
  sorry

end NUMINAMATH_CALUDE_company_production_l1684_168429


namespace NUMINAMATH_CALUDE_gpa_probability_l1684_168499

structure GradeSystem where
  a_points : ℕ := 4
  b_points : ℕ := 3
  c_points : ℕ := 2
  d_points : ℕ := 1

structure CourseGrades where
  math_grade : ℕ
  science_grade : ℕ
  english_grade : ℕ
  history_grade : ℕ

def calculate_gpa (gs : GradeSystem) (cg : CourseGrades) : ℚ :=
  (cg.math_grade + cg.science_grade + cg.english_grade + cg.history_grade : ℚ) / 4

def english_prob_a : ℚ := 1/3
def english_prob_b : ℚ := 1/4
def english_prob_c : ℚ := 1 - english_prob_a - english_prob_b

def history_prob_a : ℚ := 1/5
def history_prob_b : ℚ := 2/5
def history_prob_c : ℚ := 1 - history_prob_a - history_prob_b

def prob_gpa_at_least (target_gpa : ℚ) (gs : GradeSystem) : ℚ :=
  let prob_aa := english_prob_a * history_prob_a
  let prob_ab := english_prob_a * history_prob_b
  let prob_ba := english_prob_b * history_prob_a
  prob_aa + prob_ab + prob_ba

theorem gpa_probability (gs : GradeSystem) :
  prob_gpa_at_least (15/4) gs = 1/4 :=
sorry

end NUMINAMATH_CALUDE_gpa_probability_l1684_168499


namespace NUMINAMATH_CALUDE_equation_proof_l1684_168420

theorem equation_proof : 529 + 2 * 23 * 3 + 9 = 676 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1684_168420


namespace NUMINAMATH_CALUDE_china_mobile_charges_l1684_168419

/-- Represents a mobile plan with a base fee and an excess charge per minute -/
structure MobilePlan where
  base_fee : ℝ
  excess_charge : ℝ

/-- Calculates the total call charges for a given mobile plan and excess minutes -/
def total_charges (plan : MobilePlan) (excess_minutes : ℝ) : ℝ :=
  plan.base_fee + plan.excess_charge * excess_minutes

/-- Theorem stating the relationship between total charges and excess minutes for the specific plan -/
theorem china_mobile_charges (x : ℝ) :
  let plan := MobilePlan.mk 39 0.19
  total_charges plan x = 0.19 * x + 39 := by
  sorry


end NUMINAMATH_CALUDE_china_mobile_charges_l1684_168419


namespace NUMINAMATH_CALUDE_square_side_length_l1684_168436

theorem square_side_length (overlap1 overlap2 overlap3 non_overlap_total : ℝ) 
  (h1 : overlap1 = 2)
  (h2 : overlap2 = 5)
  (h3 : overlap3 = 8)
  (h4 : non_overlap_total = 117)
  (h5 : overlap1 > 0 ∧ overlap2 > 0 ∧ overlap3 > 0 ∧ non_overlap_total > 0) :
  ∃ (side_length : ℝ), 
    side_length = 7 ∧ 
    3 * side_length ^ 2 = non_overlap_total + 2 * (overlap1 + overlap2 + overlap3) :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l1684_168436


namespace NUMINAMATH_CALUDE_fifth_term_is_eight_l1684_168442

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fibonacci_like n + fibonacci_like (n + 1)

theorem fifth_term_is_eight : fibonacci_like 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_eight_l1684_168442


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1684_168434

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3) :
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - m := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1684_168434


namespace NUMINAMATH_CALUDE_haleys_stickers_haleys_stickers_specific_l1684_168471

/-- Haley's sticker distribution problem -/
theorem haleys_stickers (num_friends : ℕ) (stickers_per_friend : ℕ) : 
  num_friends * stickers_per_friend = num_friends * stickers_per_friend := by
  sorry

/-- The specific case of Haley's problem -/
theorem haleys_stickers_specific : 9 * 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_haleys_stickers_haleys_stickers_specific_l1684_168471


namespace NUMINAMATH_CALUDE_determinant_equality_l1684_168460

theorem determinant_equality (x y z w : ℝ) : 
  x * w - y * z = 7 → (x + 2*z) * w - (y + 2*w) * z = 7 := by sorry

end NUMINAMATH_CALUDE_determinant_equality_l1684_168460


namespace NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l1684_168406

theorem no_consecutive_integers_with_square_diff_2000 :
  ¬ ∃ (a : ℤ), (a + 1)^2 - a^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l1684_168406


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1684_168403

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1684_168403


namespace NUMINAMATH_CALUDE_raj_house_bedrooms_l1684_168454

/-- Represents the floor plan of Raj's house -/
structure RajHouse where
  total_area : ℕ
  bedroom_side : ℕ
  bathroom_length : ℕ
  bathroom_width : ℕ
  num_bathrooms : ℕ
  kitchen_area : ℕ

/-- Calculates the number of bedrooms in Raj's house -/
def num_bedrooms (house : RajHouse) : ℕ :=
  let bathroom_area := house.bathroom_length * house.bathroom_width * house.num_bathrooms
  let kitchen_living_area := 2 * house.kitchen_area
  let non_bedroom_area := bathroom_area + kitchen_living_area
  let bedroom_area := house.total_area - non_bedroom_area
  bedroom_area / (house.bedroom_side * house.bedroom_side)

/-- Theorem stating that Raj's house has 4 bedrooms -/
theorem raj_house_bedrooms :
  let house : RajHouse := {
    total_area := 1110,
    bedroom_side := 11,
    bathroom_length := 6,
    bathroom_width := 8,
    num_bathrooms := 2,
    kitchen_area := 265
  }
  num_bedrooms house = 4 := by
  sorry


end NUMINAMATH_CALUDE_raj_house_bedrooms_l1684_168454


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l1684_168428

theorem solution_implies_m_value (m : ℚ) : 
  (m * (-3) - 8 = 15 + m) → m = -23/4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l1684_168428


namespace NUMINAMATH_CALUDE_trajectory_equation_l1684_168450

/-- Given three distinct points A(-2, y), B(0, y/2), and C(x, y) on a plane,
    if AB is perpendicular to BC, then the trajectory equation of point C is y^2 = 8x (x ≠ 0) -/
theorem trajectory_equation (x y : ℝ) (h : x ≠ 0) :
  let A : ℝ × ℝ := (-2, y)
  let B : ℝ × ℝ := (0, y/2)
  let C : ℝ × ℝ := (x, y)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  AB.1 * BC.1 + AB.2 * BC.2 = 0 → y^2 = 8*x :=
by
  sorry


end NUMINAMATH_CALUDE_trajectory_equation_l1684_168450


namespace NUMINAMATH_CALUDE_eight_b_value_l1684_168468

theorem eight_b_value (a b : ℚ) 
  (eq1 : 6 * a + 3 * b = 3) 
  (eq2 : b = 2 * a - 3) : 
  8 * b = -8 := by
sorry

end NUMINAMATH_CALUDE_eight_b_value_l1684_168468


namespace NUMINAMATH_CALUDE_part1_part2_part3_l1684_168496

-- Define the equation
def equation (x a : ℝ) : Prop :=
  (x - a) / (x - 2) - 5 / x = 1

-- Part 1: x = 5 is a root
theorem part1 :
  ∀ a : ℝ, equation 5 a → a = -1 := by sorry

-- Part 2: Double root
theorem part2 :
  ∀ a : ℝ, (∃ x : ℝ, x ≠ 2 ∧ equation x a ∧ (∀ y : ℝ, y ≠ x → ¬equation y a)) → a = 2 := by sorry

-- Part 3: No solution
theorem part3 :
  ∀ a : ℝ, (∀ x : ℝ, ¬equation x a) → (a = -3 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l1684_168496


namespace NUMINAMATH_CALUDE_point_N_coordinates_l1684_168426

def M : ℝ × ℝ := (3, -4)
def a : ℝ × ℝ := (1, -2)

theorem point_N_coordinates : 
  ∀ N : ℝ × ℝ, 
  (N.1 - M.1, N.2 - M.2) = (-2 * a.1, -2 * a.2) → 
  N = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l1684_168426


namespace NUMINAMATH_CALUDE_two_a_plus_b_values_l1684_168457

theorem two_a_plus_b_values (a b : ℝ) 
  (h1 : |a - 1| = 4)
  (h2 : |-b| = |-7|)
  (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_two_a_plus_b_values_l1684_168457


namespace NUMINAMATH_CALUDE_final_employee_count_l1684_168483

/-- Represents the workforce of Company X throughout the year --/
structure CompanyWorkforce where
  initial_total : ℕ
  initial_female : ℕ
  second_quarter_total : ℕ
  second_quarter_female : ℕ
  third_quarter_total : ℕ
  third_quarter_female : ℕ
  final_total : ℕ
  final_female : ℕ

/-- Theorem stating the final number of employees given the workforce changes --/
theorem final_employee_count (w : CompanyWorkforce) : w.final_total = 700 :=
  by
  have h1 : w.initial_female = (60 : ℚ) / 100 * w.initial_total := by sorry
  have h2 : w.second_quarter_total = w.initial_total + 30 := by sorry
  have h3 : w.second_quarter_female = w.initial_female := by sorry
  have h4 : w.second_quarter_female = (57 : ℚ) / 100 * w.second_quarter_total := by sorry
  have h5 : w.third_quarter_total = w.second_quarter_total + 50 := by sorry
  have h6 : w.third_quarter_female = w.second_quarter_female + 50 := by sorry
  have h7 : w.third_quarter_female = (62 : ℚ) / 100 * w.third_quarter_total := by sorry
  have h8 : w.final_total = w.third_quarter_total + 50 := by sorry
  have h9 : w.final_female = w.third_quarter_female + 10 := by sorry
  have h10 : w.final_female = (58 : ℚ) / 100 * w.final_total := by sorry
  sorry


end NUMINAMATH_CALUDE_final_employee_count_l1684_168483


namespace NUMINAMATH_CALUDE_max_value_x_minus_2z_l1684_168417

theorem max_value_x_minus_2z (x y z : ℝ) :
  x^2 + y^2 + z^2 = 16 →
  ∃ (max : ℝ), max = 4 * Real.sqrt 5 ∧ ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 16 → x' - 2*z' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2z_l1684_168417


namespace NUMINAMATH_CALUDE_total_tasters_l1684_168431

/-- Represents the number of apple pies Sedrach has -/
def num_pies : ℕ := 13

/-- Represents the number of halves each pie can be divided into -/
def halves_per_pie : ℕ := 2

/-- Represents the number of bite-size samples each half can be split into -/
def samples_per_half : ℕ := 5

/-- Theorem stating the total number of people who can taste Sedrach's apple pies -/
theorem total_tasters : num_pies * halves_per_pie * samples_per_half = 130 := by
  sorry

end NUMINAMATH_CALUDE_total_tasters_l1684_168431


namespace NUMINAMATH_CALUDE_stream_speed_l1684_168441

/-- Proves that the speed of the stream is 8 kmph given the conditions -/
theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 24 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 8 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l1684_168441


namespace NUMINAMATH_CALUDE_right_triangle_area_l1684_168463

theorem right_triangle_area (a b c : ℝ) (h1 : a = (4/3) * b) (h2 : a = (2/3) * c) 
  (h3 : a^2 + b^2 = c^2) : (1/2) * a * b = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1684_168463


namespace NUMINAMATH_CALUDE_deck_width_l1684_168491

/-- Given a rectangular pool of dimensions 10 feet by 12 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 360 square feet, then the width of the deck is 4 feet. -/
theorem deck_width (w : ℝ) : 
  (10 + 2*w) * (12 + 2*w) = 360 → w = 4 := by sorry

end NUMINAMATH_CALUDE_deck_width_l1684_168491


namespace NUMINAMATH_CALUDE_greatest_satisfying_n_l1684_168425

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the factorial of n
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the primality check
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the condition for n
def satisfies_condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  is_prime (n + 2) ∧
  ¬(factorial n % sum_first_n n = 0)

-- Theorem statement
theorem greatest_satisfying_n :
  satisfies_condition 995 ∧
  ∀ m, satisfies_condition m → m ≤ 995 :=
sorry

end NUMINAMATH_CALUDE_greatest_satisfying_n_l1684_168425


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l1684_168451

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 35 →
    2 * chickens + 4 * rabbits = 94 →
    chickens = 23 ∧ rabbits = 12 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l1684_168451


namespace NUMINAMATH_CALUDE_carpet_square_cost_l1684_168475

/-- The cost of each carpet square given floor and carpet dimensions and total cost -/
theorem carpet_square_cost
  (floor_length : ℝ)
  (floor_width : ℝ)
  (square_side : ℝ)
  (total_cost : ℝ)
  (h1 : floor_length = 6)
  (h2 : floor_width = 10)
  (h3 : square_side = 2)
  (h4 : total_cost = 225) :
  (total_cost / ((floor_length * floor_width) / (square_side * square_side))) = 15 := by
  sorry

#check carpet_square_cost

end NUMINAMATH_CALUDE_carpet_square_cost_l1684_168475


namespace NUMINAMATH_CALUDE_area_of_specific_region_l1684_168440

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
def areaOfRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_of_specific_region :
  let c1 : Circle := { center := (5, 5), radius := 5 }
  let c2 : Circle := { center := (10, 5), radius := 3 }
  areaOfRegion c1 c2 = 25 - 17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_specific_region_l1684_168440


namespace NUMINAMATH_CALUDE_problem_solution_l1684_168422

open Real

-- Define the given condition
def alpha_condition (α : ℝ) : Prop := 2 * sin α = cos α

-- Define that α is in the third quadrant
def third_quadrant (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

theorem problem_solution (α : ℝ) 
  (h1 : alpha_condition α) 
  (h2 : third_quadrant α) : 
  (cos (π - α) = 2 * sqrt 5 / 5) ∧ 
  ((1 + 2 * sin α * sin (π / 2 - α)) / (sin α ^ 2 - cos α ^ 2) = -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1684_168422


namespace NUMINAMATH_CALUDE_dollar_three_neg_one_l1684_168467

-- Define the $ operation
def dollar (a b : ℤ) : ℤ := a * (b + 2) + a * (b + 1)

-- Theorem to prove
theorem dollar_three_neg_one : dollar 3 (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_neg_one_l1684_168467


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1684_168494

/-- Given a polynomial ax³ + bx - 3 where a and b are constants,
    if the value of the polynomial is 15 when x = 2,
    then the value of the polynomial is -21 when x = -2. -/
theorem polynomial_value_theorem (a b : ℝ) : 
  (8 * a + 2 * b - 3 = 15) → (-8 * a - 2 * b - 3 = -21) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1684_168494


namespace NUMINAMATH_CALUDE_water_pressure_force_on_trapezoidal_dam_l1684_168432

/-- The force of water pressure on a trapezoidal dam -/
theorem water_pressure_force_on_trapezoidal_dam 
  (ρ : Real) (g : Real) (a b h : Real) : 
  ρ = 1000 →
  g = 10 →
  a = 6.9 →
  b = 11.4 →
  h = 5.0 →
  ρ * g * h^2 * (b / 2 - (b - a) * h / (6 * h)) = 1050000 := by
  sorry

end NUMINAMATH_CALUDE_water_pressure_force_on_trapezoidal_dam_l1684_168432


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1684_168470

theorem algebraic_expression_value (a b : ℝ) :
  2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18 →
  9 * b - 6 * a + 2 = 32 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1684_168470


namespace NUMINAMATH_CALUDE_infinite_triples_with_gcd_one_l1684_168444

theorem infinite_triples_with_gcd_one (m n : ℕ+) :
  ∃ (a b c : ℕ+),
    a = m^2 + m * n + n^2 ∧
    b = m^2 - m * n ∧
    c = n^2 - m * n ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    a^2 = b^2 + c^2 + b * c :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_with_gcd_one_l1684_168444


namespace NUMINAMATH_CALUDE_power_inequality_l1684_168490

/-- Proof of inequality involving powers -/
theorem power_inequality (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 5) :
  a^n + b^n + c^n ≥ a^(n-5) * b^3 * c^2 + b^(n-5) * c^3 * a^2 + c^(n-5) * a^3 * b^2 := by
  sorry

#check power_inequality

end NUMINAMATH_CALUDE_power_inequality_l1684_168490


namespace NUMINAMATH_CALUDE_gumballs_per_pair_is_9_l1684_168416

/-- The number of gumballs Kim gets for each pair of earrings --/
def gumballs_per_pair : ℕ :=
  let earrings_day1 : ℕ := 3
  let earrings_day2 : ℕ := 2 * earrings_day1
  let earrings_day3 : ℕ := earrings_day2 - 1
  let total_earrings : ℕ := earrings_day1 + earrings_day2 + earrings_day3
  let gumballs_per_day : ℕ := 3
  let total_days : ℕ := 42
  let total_gumballs : ℕ := gumballs_per_day * total_days
  total_gumballs / total_earrings

theorem gumballs_per_pair_is_9 : gumballs_per_pair = 9 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_per_pair_is_9_l1684_168416


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1684_168465

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1684_168465


namespace NUMINAMATH_CALUDE_papaya_production_l1684_168455

/-- The number of papaya trees -/
def papaya_trees : ℕ := 2

/-- The number of mango trees -/
def mango_trees : ℕ := 3

/-- The number of mangos each mango tree produces -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits -/
def total_fruits : ℕ := 80

/-- The number of papayas each papaya tree produces -/
def papayas_per_tree : ℕ := 10

theorem papaya_production :
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits :=
by sorry

end NUMINAMATH_CALUDE_papaya_production_l1684_168455


namespace NUMINAMATH_CALUDE_crease_length_l1684_168408

theorem crease_length (width : Real) (θ : Real) : width = 10 → 
  let crease_length := width / 2 * Real.tan θ
  crease_length = 5 * Real.tan θ := by sorry

end NUMINAMATH_CALUDE_crease_length_l1684_168408


namespace NUMINAMATH_CALUDE_common_solution_iff_y_eq_one_l1684_168469

/-- The first equation: x^2 + y^2 - 4 = 0 -/
def equation1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

/-- The second equation: x^2 - 4y + y^2 = 0 -/
def equation2 (x y : ℝ) : Prop := x^2 - 4*y + y^2 = 0

/-- The theorem stating that the equations have common real solutions iff y = 1 -/
theorem common_solution_iff_y_eq_one :
  (∃ x : ℝ, equation1 x 1 ∧ equation2 x 1) ∧
  (∀ y : ℝ, y ≠ 1 → ¬∃ x : ℝ, equation1 x y ∧ equation2 x y) :=
sorry

end NUMINAMATH_CALUDE_common_solution_iff_y_eq_one_l1684_168469


namespace NUMINAMATH_CALUDE_max_value_of_f_on_I_l1684_168462

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Define the closed interval [0, 3]
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Statement of the theorem
theorem max_value_of_f_on_I :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_I_l1684_168462


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_original_solution_l1684_168446

theorem alcohol_percentage_in_original_solution 
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_mixture_percentage : ℝ)
  (h1 : original_volume = 3)
  (h2 : added_water = 1)
  (h3 : new_mixture_percentage = 24.75) :
  let new_volume := original_volume + added_water
  let alcohol_amount := (new_mixture_percentage / 100) * new_volume
  (alcohol_amount / original_volume) * 100 = 33 := by
sorry


end NUMINAMATH_CALUDE_alcohol_percentage_in_original_solution_l1684_168446
