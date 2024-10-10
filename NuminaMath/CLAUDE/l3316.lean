import Mathlib

namespace initial_stock_proof_l3316_331620

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 1400

/-- The number of books sold over 5 days -/
def books_sold : ℕ := 402

/-- The percentage of books sold, expressed as a real number between 0 and 1 -/
def percentage_sold : ℝ := 0.2871428571428571

theorem initial_stock_proof :
  (books_sold : ℝ) / initial_books = percentage_sold :=
by sorry

end initial_stock_proof_l3316_331620


namespace intersection_when_a_half_range_of_a_for_empty_intersection_l3316_331667

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem for part I
theorem intersection_when_a_half : 
  A (1/2) ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem for part II
theorem range_of_a_for_empty_intersection :
  ∀ a : ℝ, (A a).Nonempty → (A a ∩ B = ∅) → 
    ((-2 < a ∧ a ≤ -1/2) ∨ a ≥ 2) := by sorry

end intersection_when_a_half_range_of_a_for_empty_intersection_l3316_331667


namespace sector_area_l3316_331614

/-- The area of a sector with radius 2 and central angle π/4 is π/2 -/
theorem sector_area (r : ℝ) (α : ℝ) (S : ℝ) : 
  r = 2 → α = π / 4 → S = (1 / 2) * r^2 * α → S = π / 2 := by
  sorry

end sector_area_l3316_331614


namespace units_digit_of_sum_of_squares_2007_odd_integers_l3316_331637

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square (n : ℕ) : ℕ := n * n

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map square |> List.sum

theorem units_digit_of_sum_of_squares_2007_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 2007)) = 5 := by
  sorry

end units_digit_of_sum_of_squares_2007_odd_integers_l3316_331637


namespace laura_charge_account_balance_l3316_331636

/-- Calculates the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating that the total amount owed is $37.10 given the problem conditions --/
theorem laura_charge_account_balance : 
  total_amount_owed 35 0.06 1 = 37.10 := by
  sorry

end laura_charge_account_balance_l3316_331636


namespace janet_total_earnings_l3316_331612

/-- Calculates Janet's total earnings from exterminator work and sculpture sales -/
def janet_earnings (exterminator_rate : ℕ) (sculpture_rate : ℕ) (hours_worked : ℕ) (sculpture1_weight : ℕ) (sculpture2_weight : ℕ) : ℕ :=
  exterminator_rate * hours_worked + sculpture_rate * (sculpture1_weight + sculpture2_weight)

theorem janet_total_earnings :
  janet_earnings 70 20 20 5 7 = 1640 := by
  sorry

end janet_total_earnings_l3316_331612


namespace canned_food_bins_l3316_331600

theorem canned_food_bins (soup vegetables pasta : Real) 
  (h1 : soup = 0.125)
  (h2 : vegetables = 0.125)
  (h3 : pasta = 0.5) :
  soup + vegetables + pasta = 0.75 := by
sorry

end canned_food_bins_l3316_331600


namespace sqrt_gt_3x_iff_l3316_331660

theorem sqrt_gt_3x_iff (x : ℝ) (h : x > 0) : 
  Real.sqrt x > 3 * x ↔ 0 < x ∧ x < 1/9 := by
  sorry

end sqrt_gt_3x_iff_l3316_331660


namespace positive_integer_pairs_l3316_331659

theorem positive_integer_pairs (a b : ℕ+) :
  (∃ k : ℤ, (a.val^3 * b.val - 1) = k * (a.val + 1)) ∧
  (∃ m : ℤ, (b.val^3 * a.val + 1) = m * (b.val - 1)) →
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
by sorry

end positive_integer_pairs_l3316_331659


namespace original_class_size_l3316_331669

/-- Proves that the original number of students in a class is 12, given the conditions of the problem. -/
theorem original_class_size (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ),
    original_size * initial_avg + new_students * new_avg = (original_size + new_students) * (initial_avg - avg_decrease) ∧
    original_size = 12 := by
  sorry

end original_class_size_l3316_331669


namespace subscription_total_l3316_331685

-- Define the subscription amounts for a, b, and c
def subscription (a b c : ℕ) : Prop :=
  a = b + 4000 ∧ b = c + 5000

-- Define the total profit and b's share
def profit_share (total_profit b_profit : ℕ) : Prop :=
  total_profit = 30000 ∧ b_profit = 10200

-- Define the total subscription
def total_subscription (a b c : ℕ) : ℕ :=
  a + b + c

-- Theorem statement
theorem subscription_total 
  (a b c : ℕ) 
  (h1 : subscription a b c) 
  (h2 : profit_share 30000 10200) :
  total_subscription a b c = 14036 :=
sorry

end subscription_total_l3316_331685


namespace final_output_is_25_l3316_331604

def algorithm_output : ℕ → ℕ
| 0 => 25
| (n+1) => if 2*n + 1 < 10 then algorithm_output n else 2*(2*n + 1) + 3

theorem final_output_is_25 : algorithm_output 0 = 25 := by
  sorry

end final_output_is_25_l3316_331604


namespace pipe_fill_time_l3316_331607

/-- Given three pipes P, Q, and R that can fill a tank, this theorem proves
    that if P fills the tank in 2 hours, Q in 4 hours, and all pipes together
    in 1.2 hours, then R fills the tank in 12 hours. -/
theorem pipe_fill_time (fill_rate_P fill_rate_Q fill_rate_R : ℝ) : 
  fill_rate_P = 1 / 2 →
  fill_rate_Q = 1 / 4 →
  fill_rate_P + fill_rate_Q + fill_rate_R = 1 / 1.2 →
  fill_rate_R = 1 / 12 :=
by sorry

end pipe_fill_time_l3316_331607


namespace table_permutation_exists_l3316_331692

/-- Represents a 2 × n table of real numbers -/
def Table (n : ℕ) := Fin 2 → Fin n → ℝ

/-- Calculates the sum of a column in the table -/
def columnSum (t : Table n) (j : Fin n) : ℝ :=
  (t 0 j) + (t 1 j)

/-- Calculates the sum of a row in the table -/
def rowSum (t : Table n) (i : Fin 2) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin n)) (λ j => t i j)

/-- States that all column sums in a table are different -/
def distinctColumnSums (t : Table n) : Prop :=
  ∀ j k : Fin n, j ≠ k → columnSum t j ≠ columnSum t k

/-- Represents a permutation of table elements -/
def tablePermutation (n : ℕ) := Fin 2 → Fin n → Fin 2 × Fin n

/-- Applies a permutation to a table -/
def applyPermutation (t : Table n) (p : tablePermutation n) : Table n :=
  λ i j => let (i', j') := p i j; t i' j'

theorem table_permutation_exists (n : ℕ) (h : n > 2) (t : Table n) 
  (hd : distinctColumnSums t) :
  ∃ p : tablePermutation n, 
    distinctColumnSums (applyPermutation t p) ∧ 
    rowSum (applyPermutation t p) 0 ≠ rowSum (applyPermutation t p) 1 :=
  sorry

end table_permutation_exists_l3316_331692


namespace supermarket_theorem_l3316_331627

/-- Represents the supermarket's agricultural product distribution problem -/
structure SupermarketProblem where
  total_boxes : ℕ
  brand_a_cost : ℝ
  brand_a_price : ℝ
  brand_b_cost : ℝ
  brand_b_price : ℝ
  total_expenditure : ℝ
  min_total_profit : ℝ

/-- Theorem for the supermarket problem -/
theorem supermarket_theorem (p : SupermarketProblem)
  (h_total : p.total_boxes = 100)
  (h_a_cost : p.brand_a_cost = 80)
  (h_a_price : p.brand_a_price = 120)
  (h_b_cost : p.brand_b_cost = 130)
  (h_b_price : p.brand_b_price = 200)
  (h_expenditure : p.total_expenditure = 10000)
  (h_min_profit : p.min_total_profit = 5600) :
  (∃ (x y : ℕ), x + y = p.total_boxes ∧ 
    p.brand_a_cost * x + p.brand_b_cost * y = p.total_expenditure ∧
    x = 60 ∧ y = 40) ∧
  (∃ (z : ℕ), z ≥ 54 ∧
    (p.brand_a_price - p.brand_a_cost) * (p.total_boxes - z) +
    (p.brand_b_price - p.brand_b_cost) * z ≥ p.min_total_profit) :=
by sorry


end supermarket_theorem_l3316_331627


namespace brown_eggs_survived_l3316_331656

/-- Given that Linda initially had three times as many white eggs as brown eggs,
    and after dropping her basket she ended up with a dozen eggs in total,
    prove that 3 brown eggs survived the fall. -/
theorem brown_eggs_survived (white_eggs brown_eggs : ℕ) : 
  white_eggs = 3 * brown_eggs →  -- Initial condition
  white_eggs + brown_eggs = 12 →  -- Total eggs after the fall
  brown_eggs > 0 →  -- Some brown eggs survived
  brown_eggs = 3 := by
  sorry

end brown_eggs_survived_l3316_331656


namespace sum_x_y_l3316_331630

/-- The smallest positive integer x such that 480x is a perfect square -/
def x : ℕ := 30

/-- The smallest positive integer y such that 480y is a perfect cube -/
def y : ℕ := 450

/-- 480 * x is a perfect square -/
axiom x_square : ∃ n : ℕ, 480 * x = n^2

/-- 480 * y is a perfect cube -/
axiom y_cube : ∃ n : ℕ, 480 * y = n^3

/-- x is the smallest positive integer such that 480x is a perfect square -/
axiom x_smallest : ∀ z : ℕ, z > 0 → z < x → ¬∃ n : ℕ, 480 * z = n^2

/-- y is the smallest positive integer such that 480y is a perfect cube -/
axiom y_smallest : ∀ z : ℕ, z > 0 → z < y → ¬∃ n : ℕ, 480 * z = n^3

theorem sum_x_y : x + y = 480 := by sorry

end sum_x_y_l3316_331630


namespace inscribed_circle_radius_l3316_331658

/-- For a right triangle with legs a and b, hypotenuse c, and an inscribed circle of radius r -/
def RightTriangleWithInscribedCircle (a b c r : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ a^2 + b^2 = c^2

/-- The radius of the inscribed circle in a right triangle is equal to (a + b - c) / 2 -/
theorem inscribed_circle_radius 
  (a b c r : ℝ) 
  (h : RightTriangleWithInscribedCircle a b c r) : 
  r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_l3316_331658


namespace area_of_similar_rectangle_l3316_331648

/-- Given a rectangle R1 with one side of 4 inches and an area of 32 square inches,
    and a similar rectangle R2 with a diagonal of 10 inches,
    the area of R2 is 40 square inches. -/
theorem area_of_similar_rectangle (side_R1 area_R1 diagonal_R2 : ℝ) :
  side_R1 = 4 →
  area_R1 = 32 →
  diagonal_R2 = 10 →
  ∃ (side_a_R2 side_b_R2 : ℝ),
    side_a_R2 * side_b_R2 = 40 ∧
    side_a_R2^2 + side_b_R2^2 = diagonal_R2^2 ∧
    side_b_R2 / side_a_R2 = area_R1 / side_R1^2 :=
by sorry


end area_of_similar_rectangle_l3316_331648


namespace geometric_sequence_first_term_l3316_331699

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_decreasing : is_decreasing_sequence a)
  (h_third_term : a 3 = 18)
  (h_fourth_term : a 4 = 12) :
  a 1 = 40.5 :=
sorry

end geometric_sequence_first_term_l3316_331699


namespace decimal_to_fraction_l3316_331611

theorem decimal_to_fraction :
  (0.36 : ℚ) = 9 / 25 := by sorry

end decimal_to_fraction_l3316_331611


namespace store_a_cheaper_for_300_l3316_331642

def cost_store_a (x : ℕ) : ℝ :=
  if x ≤ 100 then 5 * x else 4 * x + 100

def cost_store_b (x : ℕ) : ℝ :=
  4.5 * x

theorem store_a_cheaper_for_300 :
  cost_store_a 300 < cost_store_b 300 :=
sorry

end store_a_cheaper_for_300_l3316_331642


namespace parallelogram_area_from_rectangle_l3316_331668

theorem parallelogram_area_from_rectangle (rectangle_width rectangle_length parallelogram_height : ℝ) 
  (hw : rectangle_width = 8)
  (hl : rectangle_length = 10)
  (hh : parallelogram_height = 9) :
  rectangle_width * parallelogram_height = 72 := by
  sorry

#check parallelogram_area_from_rectangle

end parallelogram_area_from_rectangle_l3316_331668


namespace integer_pair_gcd_equation_l3316_331651

theorem integer_pair_gcd_equation :
  ∀ x y : ℕ+, 
    (x.val * y.val * Nat.gcd x.val y.val = x.val + y.val + (Nat.gcd x.val y.val)^2) ↔ 
    ((x, y) = (2, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (3, 2)) := by
  sorry

end integer_pair_gcd_equation_l3316_331651


namespace rationalize_denominator_1_l3316_331687

theorem rationalize_denominator_1 (a b c : ℝ) :
  a / (b - Real.sqrt c + a) = (a * (b + a + Real.sqrt c)) / ((b + a)^2 - c) :=
sorry

end rationalize_denominator_1_l3316_331687


namespace tangent_and_trigonometric_identity_l3316_331661

theorem tangent_and_trigonometric_identity (α : ℝ) 
  (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) : 
  (Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3) ∧ 
  (2 * Real.sin α ^ 2 - Real.cos α ^ 2 = -43 / 52) := by
  sorry

end tangent_and_trigonometric_identity_l3316_331661


namespace max_value_sum_of_roots_l3316_331613

theorem max_value_sum_of_roots (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  (Real.sqrt (49 + x) + Real.sqrt (49 - x) = 14 ↔ x = 0) :=
by sorry

end max_value_sum_of_roots_l3316_331613


namespace sum_of_reciprocals_l3316_331609

theorem sum_of_reciprocals (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) :
  x + y = 4/3 := by sorry

end sum_of_reciprocals_l3316_331609


namespace loan_sum_proof_l3316_331619

theorem loan_sum_proof (x y : ℝ) : 
  x * (3 / 100) * 5 = y * (5 / 100) * 3 →
  y = 1332.5 →
  x + y = 2665 := by
sorry

end loan_sum_proof_l3316_331619


namespace intersection_condition_l3316_331610

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem intersection_condition (a : ℝ) :
  (A a ∩ B).Nonempty → 1/3 < a ∧ a < 1 := by sorry

end intersection_condition_l3316_331610


namespace absolute_value_equation_product_l3316_331682

theorem absolute_value_equation_product (x₁ x₂ : ℝ) : 
  (|3 * x₁ - 5| = 40) ∧ (|3 * x₂ - 5| = 40) ∧ (x₁ ≠ x₂) →
  x₁ * x₂ = -175 := by
sorry

end absolute_value_equation_product_l3316_331682


namespace total_money_made_l3316_331665

/-- Represents the amount of water collected per inch of rain -/
def gallons_per_inch : ℝ := 15

/-- Represents the rainfall on Monday in inches -/
def monday_rain : ℝ := 4

/-- Represents the rainfall on Tuesday in inches -/
def tuesday_rain : ℝ := 3

/-- Represents the rainfall on Wednesday in inches -/
def wednesday_rain : ℝ := 2.5

/-- Represents the selling price per gallon on Monday -/
def monday_price : ℝ := 1.2

/-- Represents the selling price per gallon on Tuesday -/
def tuesday_price : ℝ := 1.5

/-- Represents the selling price per gallon on Wednesday -/
def wednesday_price : ℝ := 0.8

/-- Theorem stating the total money James made from selling water -/
theorem total_money_made : 
  (gallons_per_inch * monday_rain * monday_price) +
  (gallons_per_inch * tuesday_rain * tuesday_price) +
  (gallons_per_inch * wednesday_rain * wednesday_price) = 169.5 := by
  sorry

end total_money_made_l3316_331665


namespace five_digit_division_sum_l3316_331603

theorem five_digit_division_sum (ABCDE : ℕ) : 
  ABCDE ≥ 10000 ∧ ABCDE < 100000 ∧ ABCDE % 6 = 0 ∧ ABCDE / 6 = 13579 →
  (ABCDE / 100) + (ABCDE % 100) = 888 := by
sorry

end five_digit_division_sum_l3316_331603


namespace matrix_cube_computation_l3316_331643

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_cube_computation :
  A ^ 3 = !![(-4), 2; (-2), 1] := by sorry

end matrix_cube_computation_l3316_331643


namespace circle_area_through_points_l3316_331666

/-- The area of a circle with center P(-5, 3) passing through Q(7, -2) is 169π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (-5, 3)
  let Q : ℝ × ℝ := (7, -2)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 169 * π := by
  sorry

end circle_area_through_points_l3316_331666


namespace circle_radius_l3316_331672

theorem circle_radius (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 - 4 * x + 6 * y = 3/2) → 
  ∃ (h k r : ℝ), r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
sorry

end circle_radius_l3316_331672


namespace f_max_min_l3316_331673

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem f_max_min :
  ∀ x ∈ Set.Icc (-1 : ℝ) 0,
    f x ≤ 4/3 ∧ f x ≥ 1 ∧
    (∃ x₁ ∈ Set.Icc (-1 : ℝ) 0, f x₁ = 4/3) ∧
    (∃ x₂ ∈ Set.Icc (-1 : ℝ) 0, f x₂ = 1) :=
by sorry

end f_max_min_l3316_331673


namespace triangle_abc_proof_l3316_331639

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  (b - 2*a) * Real.cos C + c * Real.cos B = 0 →
  c = 2 →
  S = Real.sqrt 3 →
  S = 1/2 * a * b * Real.sin C →
  a^2 + b^2 - c^2 = 2*a*b * Real.cos C →
  C = π/3 ∧ a = 2 ∧ b = 2 := by sorry

end triangle_abc_proof_l3316_331639


namespace complex_product_real_l3316_331608

theorem complex_product_real (z₁ z₂ : ℂ) :
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 ↔ z₂ = 4 + 2 * Complex.I :=
by sorry

end complex_product_real_l3316_331608


namespace computer_table_cost_price_l3316_331646

/-- The cost price of a computer table, given the selling price and markup percentage. -/
def cost_price (selling_price : ℚ) (markup_percentage : ℚ) : ℚ :=
  selling_price / (1 + markup_percentage)

/-- Theorem: The cost price of the computer table is 6500, given the conditions. -/
theorem computer_table_cost_price :
  let selling_price : ℚ := 8450
  let markup_percentage : ℚ := 0.30
  cost_price selling_price markup_percentage = 6500 := by
sorry

end computer_table_cost_price_l3316_331646


namespace students_favoring_both_proposals_l3316_331618

theorem students_favoring_both_proposals 
  (total : ℕ) 
  (favor_A : ℕ) 
  (favor_B : ℕ) 
  (against_both : ℕ) 
  (h1 : total = 232)
  (h2 : favor_A = 172)
  (h3 : favor_B = 143)
  (h4 : against_both = 37) :
  favor_A + favor_B - (total - against_both) = 120 := by
  sorry

end students_favoring_both_proposals_l3316_331618


namespace solve_for_m_l3316_331601

-- Define the functions f and g
def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m
def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

-- State the theorem
theorem solve_for_m : 
  ∀ m : ℚ, 3 * (f m 5) = 2 * (g m 5) → m = 10/7 := by sorry

end solve_for_m_l3316_331601


namespace divisible_count_theorem_l3316_331681

def count_divisible (n : ℕ) : ℕ :=
  let div2 := n / 2
  let div3 := n / 3
  let div5 := n / 5
  let div6 := n / 6
  let div10 := n / 10
  let div15 := n / 15
  let div30 := n / 30
  (div2 + div3 + div5 - div6 - div10 - div15 + div30) - div6

theorem divisible_count_theorem :
  count_divisible 1000 = 568 := by sorry

end divisible_count_theorem_l3316_331681


namespace variance_of_five_numbers_l3316_331629

theorem variance_of_five_numbers (m : ℝ) 
  (h : (1 + 2 + 3 + 4 + m) / 5 = 3) : 
  ((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (4 - 3)^2 + (m - 3)^2) / 5 = 2 := by
  sorry

end variance_of_five_numbers_l3316_331629


namespace student_score_problem_l3316_331638

theorem student_score_problem (total_questions : ℕ) (student_score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : student_score = 61) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 87 ∧ 
    student_score = correct_answers - 2 * (total_questions - correct_answers) :=
by
  sorry

end student_score_problem_l3316_331638


namespace secret_number_count_l3316_331653

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def secret_number (n : ℕ) : Prop :=
  is_two_digit n ∧
  Odd (tens_digit n) ∧
  Even (units_digit n) ∧
  n > 75 ∧
  n % 3 = 0

theorem secret_number_count : 
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ secret_number n :=
sorry

end secret_number_count_l3316_331653


namespace line_tangent_to_parabola_l3316_331624

/-- A line with equation x - 2y = r is tangent to a parabola with equation y = x^2 - r
    if and only if r = -1/8 -/
theorem line_tangent_to_parabola (r : ℝ) :
  (∃ x y, x - 2*y = r ∧ y = x^2 - r ∧
    ∀ x' y', x' - 2*y' = r ∧ y' = x'^2 - r → (x', y') = (x, y)) ↔
  r = -1/8 := by
sorry

end line_tangent_to_parabola_l3316_331624


namespace functional_equation_implies_odd_l3316_331695

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) = y * f x

/-- Theorem stating that f(-x) = -f(x) for functions satisfying the functional equation -/
theorem functional_equation_implies_odd (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end functional_equation_implies_odd_l3316_331695


namespace calculation_proof_l3316_331621

theorem calculation_proof : 3 * 16 + 3 * 17 + 3 * 20 + 11 = 170 := by
  sorry

end calculation_proof_l3316_331621


namespace mings_estimate_smaller_l3316_331691

theorem mings_estimate_smaller (x y δ : ℝ) (hx : x > y) (hy : y > 0) (hδ : δ > 0) :
  (x + δ) - (y + 2*δ) < x - y := by
  sorry

end mings_estimate_smaller_l3316_331691


namespace expression_evaluation_l3316_331623

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end expression_evaluation_l3316_331623


namespace cube_sum_solutions_l3316_331645

def is_cube_sum (a b c : ℕ+) : Prop :=
  ∃ n : ℕ+, 2^(Nat.factorial a.val) + 2^(Nat.factorial b.val) + 2^(Nat.factorial c.val) = n^3

theorem cube_sum_solutions :
  ∀ a b c : ℕ+, is_cube_sum a b c ↔ 
    ((a, b, c) = (1, 1, 2) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (2, 1, 1)) :=
by sorry

end cube_sum_solutions_l3316_331645


namespace five_sided_polygon_angle_sum_l3316_331641

theorem five_sided_polygon_angle_sum 
  (A B C x y : ℝ) 
  (h1 : A = 28)
  (h2 : B = 74)
  (h3 : C = 26)
  (h4 : A + B + (360 - x) + 90 + (116 - y) = 540) :
  x + y = 128 := by
  sorry

end five_sided_polygon_angle_sum_l3316_331641


namespace max_value_of_quadratic_l3316_331697

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 3/2) :
  x * (3 - 2*x) ≤ 9/8 :=
by sorry

end max_value_of_quadratic_l3316_331697


namespace determinant_of_2x2_matrix_l3316_331683

theorem determinant_of_2x2_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![9, 5; -3, 4]
  Matrix.det A = 51 := by
sorry

end determinant_of_2x2_matrix_l3316_331683


namespace outfit_count_l3316_331628

/-- The number of different outfits with different colored shirt and hat -/
def number_of_outfits (blue_shirts green_shirts pants blue_hats green_hats : ℕ) : ℕ :=
  (blue_shirts * green_hats * pants) + (green_shirts * blue_hats * pants)

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfit_count :
  number_of_outfits 7 6 7 10 9 = 861 := by
  sorry

end outfit_count_l3316_331628


namespace repeated_digit_sum_tower_exp_l3316_331679

-- Define the function for the tower of exponents
def tower_exp (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 6^(tower_exp n)

-- Define the repeated digit sum operation (conceptually)
def repeated_digit_sum (n : ℕ) : ℕ := n % 11

-- State the theorem
theorem repeated_digit_sum_tower_exp : 
  repeated_digit_sum (7^(tower_exp 5)) = 4 := by sorry

end repeated_digit_sum_tower_exp_l3316_331679


namespace tangent_circles_existence_l3316_331654

-- Define the necessary geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the tangency relations
def isTangentToCircle (c1 c2 : Circle) : Prop :=
  sorry

def isTangentToLine (c : Circle) (l : Line) : Prop :=
  sorry

def isOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  sorry

-- Theorem statement
theorem tangent_circles_existence
  (C : Circle) (l : Line) (M : ℝ × ℝ) 
  (h : isOnLine M l) :
  ∃ (C' C'' : Circle),
    (isTangentToCircle C' C ∧ isTangentToLine C' l ∧ isOnLine M l) ∧
    (isTangentToCircle C'' C ∧ isTangentToLine C'' l ∧ isOnLine M l) ∧
    (C' ≠ C'') :=
by sorry

end tangent_circles_existence_l3316_331654


namespace fraction_inequality_l3316_331632

theorem fraction_inequality (a : ℝ) : a > 1 → (2*a + 1)/(a - 1) > 2 := by
  sorry

end fraction_inequality_l3316_331632


namespace coat_duration_proof_l3316_331652

/-- The duration (in years) for which the more expensive coat lasts -/
def duration_expensive_coat : ℕ := sorry

/-- The cost of the more expensive coat -/
def cost_expensive_coat : ℕ := 300

/-- The cost of the cheaper coat -/
def cost_cheaper_coat : ℕ := 120

/-- The duration (in years) for which the cheaper coat lasts -/
def duration_cheaper_coat : ℕ := 5

/-- The time period (in years) over which savings are calculated -/
def savings_period : ℕ := 30

/-- The amount saved over the savings period by choosing the more expensive coat -/
def savings_amount : ℕ := 120

theorem coat_duration_proof :
  duration_expensive_coat = 15 ∧
  cost_expensive_coat * savings_period / duration_expensive_coat +
    savings_amount =
  cost_cheaper_coat * savings_period / duration_cheaper_coat :=
by sorry

end coat_duration_proof_l3316_331652


namespace line_passes_through_fixed_point_l3316_331640

/-- The line ax + y + a + 1 = 0 always passes through the point (-1, -1) for all values of a. -/
theorem line_passes_through_fixed_point (a : ℝ) : a * (-1) + (-1) + a + 1 = 0 := by
  sorry

end line_passes_through_fixed_point_l3316_331640


namespace expression_simplification_and_evaluation_l3316_331670

theorem expression_simplification_and_evaluation (a b : ℚ) 
  (ha : a = -2) (hb : b = 3/2) : 
  1/2 * a - 2 * (a - 1/2 * b^2) - (3/2 * a - 1/3 * b^2) = 9 := by
  sorry

end expression_simplification_and_evaluation_l3316_331670


namespace gcd_442872_312750_l3316_331633

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 := by
  sorry

end gcd_442872_312750_l3316_331633


namespace solution_set_part_i_range_of_a_part_ii_l3316_331622

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 3| > 0} = Set.Ioi 2 ∪ Set.Iic (2/3) :=
sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | ∀ x, f a x < |x - 3|} = Set.Ioo 1 5 :=
sorry

end solution_set_part_i_range_of_a_part_ii_l3316_331622


namespace no_formula_matches_l3316_331635

def x : List ℕ := [1, 2, 3, 4, 5]
def y : List ℕ := [5, 15, 33, 61, 101]

def formula_a (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_b (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_d (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem no_formula_matches : 
  (∃ i, List.get! x i ≠ 0 ∧ formula_a (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_b (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_c (List.get! x i) ≠ List.get! y i) ∧
  (∃ i, List.get! x i ≠ 0 ∧ formula_d (List.get! x i) ≠ List.get! y i) :=
by sorry

end no_formula_matches_l3316_331635


namespace f_decreasing_interval_a_upper_bound_l3316_331684

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the monotonically decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (-1)),
  StrictMonoOn f (Set.Ioo 0 (Real.exp (-1))) :=
sorry

-- Theorem for the range of a
theorem a_upper_bound
  (h : ∀ x > 0, f x ≥ -x^2 + a*x - 6) :
  a ≤ 5 + Real.log 2 :=
sorry

end f_decreasing_interval_a_upper_bound_l3316_331684


namespace travel_fraction_proof_l3316_331615

def initial_amount : ℚ := 750
def clothes_fraction : ℚ := 1/3
def food_fraction : ℚ := 1/5
def final_amount : ℚ := 300

theorem travel_fraction_proof :
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  let spent_on_travel := remaining_after_food - final_amount
  spent_on_travel / remaining_after_food = 1/4 := by sorry

end travel_fraction_proof_l3316_331615


namespace smallest_prime_8_less_than_odd_square_l3316_331647

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem smallest_prime_8_less_than_odd_square : 
  ∀ n : ℕ, 
    n > 0 → 
    is_prime n → 
    (∃ m : ℕ, 
      m ≥ 16 ∧ 
      is_perfect_square (n + 8) ∧ 
      is_odd (n + 8) ∧ 
      n + 8 = m * m) → 
    n ≥ 17 :=
sorry

end smallest_prime_8_less_than_odd_square_l3316_331647


namespace sum_of_coefficients_zero_l3316_331655

theorem sum_of_coefficients_zero (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x, (1 - 4*x)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                       a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁/2 + a₂/2^2 + a₃/2^3 + a₄/2^4 + a₅/2^5 + a₆/2^6 + a₇/2^7 + a₈/2^8 + a₉/2^9 + a₁₀/2^10 = 0 :=
by sorry

end sum_of_coefficients_zero_l3316_331655


namespace louise_oranges_l3316_331674

theorem louise_oranges (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : num_boxes = 7) 
  (h2 : oranges_per_box = 6) : 
  num_boxes * oranges_per_box = 42 := by
  sorry

end louise_oranges_l3316_331674


namespace fraction_calculation_l3316_331657

theorem fraction_calculation : 
  (2 / 7 + 5 / 8 * 1 / 3) / (3 / 4 - 2 / 9) = 15 / 16 := by
  sorry

end fraction_calculation_l3316_331657


namespace loan_principal_calculation_l3316_331617

/-- Simple interest calculation for a loan -/
theorem loan_principal_calculation 
  (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  rate = 0.12 →
  time = 3 →
  interest = 5400 →
  principal * rate * time = interest →
  principal = 15000 := by
  sorry

end loan_principal_calculation_l3316_331617


namespace intersection_point_value_l3316_331678

theorem intersection_point_value (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a*x - 2) ≥ 0) → a = 1 := by
  sorry

end intersection_point_value_l3316_331678


namespace concert_songs_theorem_l3316_331690

/-- Represents the number of songs sung by each girl -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  lucy : ℕ

/-- The total number of songs sung by the trios -/
def total_songs (s : SongCount) : ℕ :=
  (s.mary + s.alina + s.tina + s.hanna + s.lucy) / 3

/-- The conditions given in the problem -/
def satisfies_conditions (s : SongCount) : Prop :=
  s.hanna = 9 ∧
  s.lucy = 5 ∧
  s.mary > s.lucy ∧ s.mary < s.hanna ∧
  s.alina > s.lucy ∧ s.alina < s.hanna ∧
  s.tina > s.lucy ∧ s.tina < s.hanna

theorem concert_songs_theorem (s : SongCount) :
  satisfies_conditions s → total_songs s = 11 := by
  sorry

end concert_songs_theorem_l3316_331690


namespace vector_simplification_l3316_331677

/-- Given four points A, B, C, and D in a vector space, 
    prove that the vector AB minus DC minus CB equals AD -/
theorem vector_simplification (V : Type*) [AddCommGroup V] 
  (A B C D : V) : 
  (B - A) - (C - D) - (B - C) = D - A := by sorry

end vector_simplification_l3316_331677


namespace root_shift_theorem_l3316_331696

/-- Given a, b, and c are roots of x³ - 5x + 7 = 0, prove that a+3, b+3, and c+3 are roots of x³ - 9x² + 22x - 5 = 0 -/
theorem root_shift_theorem (a b c : ℝ) : 
  (a^3 - 5*a + 7 = 0) → 
  (b^3 - 5*b + 7 = 0) → 
  (c^3 - 5*c + 7 = 0) → 
  ((a+3)^3 - 9*(a+3)^2 + 22*(a+3) - 5 = 0) ∧
  ((b+3)^3 - 9*(b+3)^2 + 22*(b+3) - 5 = 0) ∧
  ((c+3)^3 - 9*(c+3)^2 + 22*(c+3) - 5 = 0) := by
  sorry


end root_shift_theorem_l3316_331696


namespace not_product_of_two_primes_l3316_331606

theorem not_product_of_two_primes (n : ℕ) (h : n ≥ 2) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  (a * b * c ∣ 2^(4*n + 2) + 1) :=
sorry

end not_product_of_two_primes_l3316_331606


namespace min_value_when_a_is_quarter_range_of_a_for_full_range_l3316_331605

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - 4*a) * a^x + a else Real.log x

-- Theorem 1: Minimum value of f(x) when a = 1/4 is 0
theorem min_value_when_a_is_quarter :
  ∀ x : ℝ, f (1/4) x ≥ 0 ∧ ∃ x₀ : ℝ, f (1/4) x₀ = 0 :=
sorry

-- Theorem 2: Range of f(x) is R iff 1/2 < a ≤ 3/4
theorem range_of_a_for_full_range :
  ∀ a : ℝ, (a > 0 ∧ a ≠ 1) →
    (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ (1/2 < a ∧ a ≤ 3/4) :=
sorry

end min_value_when_a_is_quarter_range_of_a_for_full_range_l3316_331605


namespace clean_city_people_l3316_331663

/-- The number of people working together to clean the city -/
def total_people (group_A group_B group_C group_D group_E group_F group_G group_H : ℕ) : ℕ :=
  group_A + group_B + group_C + group_D + group_E + group_F + group_G + group_H

/-- Theorem stating the total number of people cleaning the city -/
theorem clean_city_people :
  ∃ (group_A group_B group_C group_D group_E group_F group_G group_H : ℕ),
    group_A = 54 ∧
    group_B = group_A - 17 ∧
    group_C = 2 * group_B ∧
    group_D = group_A / 3 ∧
    group_E = group_C + (group_C / 4) ∧
    group_F = group_D / 2 ∧
    group_G = (group_A + group_B + group_C) - ((group_A + group_B + group_C) * 3 / 10) ∧
    group_H = group_F + group_G ∧
    total_people group_A group_B group_C group_D group_E group_F group_G group_H = 523 :=
by sorry

end clean_city_people_l3316_331663


namespace num_paths_upper_bound_l3316_331693

/-- Represents a rectangular grid city -/
structure City where
  length : ℕ
  width : ℕ

/-- The number of possible paths from southwest to northeast corner -/
def num_paths (c : City) : ℕ := sorry

/-- The theorem to be proved -/
theorem num_paths_upper_bound (c : City) :
  num_paths c ≤ 2^(c.length * c.width) := by sorry

end num_paths_upper_bound_l3316_331693


namespace product_sum_fractions_l3316_331689

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end product_sum_fractions_l3316_331689


namespace smallest_common_multiple_l3316_331650

theorem smallest_common_multiple : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ 6 ∣ m ∧ 8 ∣ m ∧ 12 ∣ m → n ≤ m) ∧ 
  6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n := by
  sorry

end smallest_common_multiple_l3316_331650


namespace xyz_value_l3316_331664

-- Define a geometric sequence of 5 terms
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q ∧ d = c * q ∧ e = d * q

-- State the theorem
theorem xyz_value (x y z : ℝ) 
  (h : is_geometric_sequence (-1) x y z (-4)) : x * y * z = -8 := by
  sorry

end xyz_value_l3316_331664


namespace linear_system_solution_l3316_331676

-- Define the determinant function
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the system of linear equations
def system (x y : ℝ) : Prop := 2 * x + y = 1 ∧ 3 * x - 2 * y = 12

-- State the theorem
theorem linear_system_solution :
  let D := det2x2 2 1 3 (-2)
  let Dx := det2x2 1 1 12 (-2)
  let Dy := det2x2 2 1 3 12
  D = -7 ∧ Dx = -14 ∧ Dy = 21 ∧ system (Dx / D) (Dy / D) ∧ system 2 (-3) := by
  sorry


end linear_system_solution_l3316_331676


namespace intersection_of_A_and_B_l3316_331644

def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l3316_331644


namespace multiple_of_eleven_with_specific_digits_l3316_331631

theorem multiple_of_eleven_with_specific_digits : ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧
  (85 * 10^5 + A * 10^4 + 3 * 10^3 + 6 * 10^2 + B * 10 + 4) % 11 = 0 ∧
  (9 * 10^6 + 1 * 10^5 + 7 * 10^4 + B * 10^3 + A * 10^2 + 5 * 10 + 0) % 11 = 0 :=
by sorry

end multiple_of_eleven_with_specific_digits_l3316_331631


namespace equation_solution_l3316_331698

theorem equation_solution (x : ℝ) : 
  (21 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 5 ∨ x = -3) :=
by sorry

end equation_solution_l3316_331698


namespace property_P_theorems_l3316_331626

/-- Property (P): A number n ≥ 2 has property (P) if in its prime factorization,
    at least one of the factors has an exponent of 3 -/
def has_property_P (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∃ p : ℕ, Prime p ∧ (∃ k : ℕ, n = p^(3*k+3) * (n / p^(3*k+3)))

/-- The smallest N such that any N consecutive natural numbers contain
    at least one number with property (P) -/
def smallest_N : ℕ := 16

/-- The smallest 15 consecutive numbers without property (P) such that
    their sum multiplied by 5 has property (P) -/
def smallest_15_consecutive : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

theorem property_P_theorems :
  (∀ k : ℕ, ∃ n ∈ List.range smallest_N, has_property_P (k + n)) ∧
  (∀ n ∈ smallest_15_consecutive, ¬ has_property_P n) ∧
  has_property_P (5 * smallest_15_consecutive.sum) := by
  sorry

end property_P_theorems_l3316_331626


namespace parts_per_day_calculation_l3316_331634

/-- The number of parts initially planned per day -/
def initial_parts_per_day : ℕ := 142

/-- The number of days with initial production rate -/
def initial_days : ℕ := 3

/-- The increase in parts per day after the initial days -/
def increase_in_parts : ℕ := 5

/-- The total number of parts produced -/
def total_parts : ℕ := 675

/-- The number of extra parts produced compared to the plan -/
def extra_parts : ℕ := 100

/-- The number of days after the initial period -/
def additional_days : ℕ := 1

theorem parts_per_day_calculation :
  initial_parts_per_day * initial_days + 
  (initial_parts_per_day + increase_in_parts) * additional_days = 
  total_parts - extra_parts :=
by sorry

#check parts_per_day_calculation

end parts_per_day_calculation_l3316_331634


namespace fraction_of_married_men_l3316_331649

theorem fraction_of_married_men (total : ℕ) (h1 : total > 0) : 
  let women := (60 : ℚ) / 100 * total
  let men := total - women
  let married := (60 : ℚ) / 100 * total
  let single_men := (3 : ℚ) / 4 * men
  (men - single_men) / men = (1 : ℚ) / 4 :=
by sorry

end fraction_of_married_men_l3316_331649


namespace first_competitor_distance_l3316_331694

/-- The long jump competition with four competitors -/
structure LongJumpCompetition where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the long jump competition -/
def validCompetition (c : LongJumpCompetition) : Prop :=
  c.second = c.first + 1 ∧
  c.third = c.second - 2 ∧
  c.fourth = c.third + 3 ∧
  c.fourth = 24

/-- Theorem: In a valid long jump competition, the first competitor jumped 22 feet -/
theorem first_competitor_distance (c : LongJumpCompetition) 
  (h : validCompetition c) : c.first = 22 := by
  sorry

#check first_competitor_distance

end first_competitor_distance_l3316_331694


namespace set_operations_l3316_331680

open Set

def A : Set ℝ := {x | x ≤ 5}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 8}

theorem set_operations :
  (A ∩ B = {x | -3 < x ∧ x ≤ 5}) ∧
  (A ∪ B = {x | x ≤ 8}) ∧
  (A ∪ (𝒰 \ B) = {x | x ≤ 5 ∨ x > 8}) := by
  sorry

end set_operations_l3316_331680


namespace min_value_theorem_l3316_331675

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (heq : a * b = 1 / 2) :
  (4 * a^2 + b^2 + 1) / (2 * a - b) ≥ 2 * Real.sqrt 3 :=
by sorry

end min_value_theorem_l3316_331675


namespace cherry_trees_planted_l3316_331616

/-- The number of trees planted by each group in a tree-planting event --/
structure TreePlanting where
  apple : ℕ
  orange : ℕ
  cherry : ℕ

/-- The conditions of the tree-planting event --/
def tree_planting_conditions (t : TreePlanting) : Prop :=
  t.apple = 2 * t.orange ∧
  t.orange = t.apple - 15 ∧
  t.cherry = t.apple + t.orange - 10 ∧
  t.apple = 47 ∧
  t.orange = 27

/-- Theorem stating that under the given conditions, 64 cherry trees were planted --/
theorem cherry_trees_planted (t : TreePlanting) 
  (h : tree_planting_conditions t) : t.cherry = 64 := by
  sorry


end cherry_trees_planted_l3316_331616


namespace diamond_equation_solution_l3316_331671

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

-- State the theorem
theorem diamond_equation_solution :
  ∀ y : ℚ, diamond 4 y = 50 → y = 58 / 7 := by
  sorry

end diamond_equation_solution_l3316_331671


namespace sin_cos_identity_l3316_331686

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 / 2 = 
  Real.sin (2 * x + π / 3) := by
  sorry

end sin_cos_identity_l3316_331686


namespace quadratic_function_range_l3316_331688

/-- Given a quadratic function f(x) = x^2 + ax + 5 that is symmetric about x = -2
    and has a range of [1, 5] on the interval [m, 0], prove that -4 ≤ m ≤ -2. -/
theorem quadratic_function_range (a : ℝ) (m : ℝ) (h_m : m < 0) :
  (∀ x, ((-2 + x)^2 + a*(-2 + x) + 5 = (-2 - x)^2 + a*(-2 - x) + 5)) →
  (∀ x ∈ Set.Icc m 0, 1 ≤ x^2 + a*x + 5 ∧ x^2 + a*x + 5 ≤ 5) →
  -4 ≤ m ∧ m ≤ -2 := by sorry

end quadratic_function_range_l3316_331688


namespace average_speed_ratio_l3316_331662

/-- Represents the average speed ratio problem -/
theorem average_speed_ratio 
  (distance_eddy : ℝ) 
  (distance_freddy : ℝ) 
  (time_eddy : ℝ) 
  (time_freddy : ℝ) 
  (h1 : distance_eddy = 600) 
  (h2 : distance_freddy = 460) 
  (h3 : time_eddy = 3) 
  (h4 : time_freddy = 4) : 
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 := by
  sorry

end average_speed_ratio_l3316_331662


namespace man_son_age_ratio_l3316_331602

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the man is 35 years older than his son and the son's present age is 33. -/
theorem man_son_age_ratio :
  let son_age : ℕ := 33
  let man_age : ℕ := son_age + 35
  let son_age_in_two_years : ℕ := son_age + 2
  let man_age_in_two_years : ℕ := man_age + 2
  man_age_in_two_years = 2 * son_age_in_two_years := by
  sorry

#check man_son_age_ratio

end man_son_age_ratio_l3316_331602


namespace solution_set_l3316_331625

def system_solution (x : ℝ) : Prop :=
  x / 3 ≥ -1 ∧ 3 * x + 4 < 1

theorem solution_set : ∀ x : ℝ, system_solution x ↔ -3 ≤ x ∧ x < -1 := by sorry

end solution_set_l3316_331625
