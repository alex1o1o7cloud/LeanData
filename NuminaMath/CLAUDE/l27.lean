import Mathlib

namespace factorization_problem_l27_2773

theorem factorization_problem (A B : ℤ) :
  (∀ x : ℝ, 10 * x^2 - 31 * x + 21 = (A * x - 7) * (B * x - 3)) →
  A * B + A = 15 := by
sorry

end factorization_problem_l27_2773


namespace problem_statement_l27_2793

theorem problem_statement (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) :
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := by
  sorry

end problem_statement_l27_2793


namespace distance_between_vertices_l27_2778

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 1| = 3

/-- The first parabola equation when y ≥ 1 -/
def parabola1 (x y : ℝ) : Prop :=
  y = -1/8 * x^2 + 2

/-- The second parabola equation when y < 1 -/
def parabola2 (x y : ℝ) : Prop :=
  y = 1/4 * x^2 - 1

/-- The vertex of the first parabola -/
def vertex1 : ℝ × ℝ := (0, 2)

/-- The vertex of the second parabola -/
def vertex2 : ℝ × ℝ := (0, -1)

theorem distance_between_vertices :
  |vertex1.2 - vertex2.2| = 3 :=
sorry

end distance_between_vertices_l27_2778


namespace least_subtraction_for_divisibility_l27_2772

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 2 ∧ 
  (13 ∣ (964807 - x)) ∧ 
  ∀ (y : ℕ), y < x → ¬(13 ∣ (964807 - y)) :=
sorry

end least_subtraction_for_divisibility_l27_2772


namespace vector_b_solution_l27_2742

def a : ℝ × ℝ := (1, -2)

theorem vector_b_solution (b : ℝ × ℝ) :
  (b.1^2 + b.2^2 = 20) →  -- |b| = 2√5
  (a.1 * b.2 = a.2 * b.1) →  -- a ∥ b
  (b = (2, -4) ∨ b = (-2, 4)) :=
by sorry

end vector_b_solution_l27_2742


namespace meiosis_fertilization_result_l27_2725

/-- Represents a genetic combination -/
structure GeneticCombination where
  -- Add necessary fields

/-- Represents a gamete -/
structure Gamete where
  -- Add necessary fields

/-- Represents an organism -/
structure Organism where
  genetic_combination : GeneticCombination

/-- Meiosis process -/
def meiosis (parent : Organism) : List Gamete :=
  sorry

/-- Fertilization process -/
def fertilization (gamete1 gamete2 : Gamete) : Organism :=
  sorry

/-- Predicate to check if two genetic combinations are different -/
def are_different (gc1 gc2 : GeneticCombination) : Prop :=
  sorry

theorem meiosis_fertilization_result 
  (parent1 parent2 : Organism) : 
  ∃ (offspring : Organism), 
    (∃ (g1 : Gamete) (g2 : Gamete), 
      g1 ∈ meiosis parent1 ∧ 
      g2 ∈ meiosis parent2 ∧ 
      offspring = fertilization g1 g2) ∧
    are_different offspring.genetic_combination parent1.genetic_combination ∧
    are_different offspring.genetic_combination parent2.genetic_combination :=
  sorry

end meiosis_fertilization_result_l27_2725


namespace bruce_books_purchased_l27_2774

def bruce_purchase (num_books : ℕ) : Prop :=
  let crayon_cost : ℕ := 5 * 5
  let calculator_cost : ℕ := 3 * 5
  let total_cost : ℕ := crayon_cost + calculator_cost + num_books * 5
  let remaining_money : ℕ := 200 - total_cost
  remaining_money = 11 * 10

theorem bruce_books_purchased : ∃ (num_books : ℕ), bruce_purchase num_books ∧ num_books = 10 := by
  sorry

end bruce_books_purchased_l27_2774


namespace expression_value_l27_2700

theorem expression_value : (1/3 * 9 * 1/27 * 81 * 1/243 * 729)^2 = 729 := by
  sorry

end expression_value_l27_2700


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l27_2753

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def sum_of_consecutive_integers (start : ℕ) (count : ℕ) : ℕ :=
  count * (2 * start + count - 1) / 2

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∀ n : ℕ, 
    (∃ start : ℕ, sum_of_consecutive_integers start 20 = n ∧ is_perfect_square n) →
    n ≥ 490 :=
by sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l27_2753


namespace dagger_example_l27_2715

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (2 * q / n)

-- Theorem statement
theorem dagger_example : dagger (5/9) (7/6) = 140/3 := by
  sorry

end dagger_example_l27_2715


namespace carpet_reconstruction_l27_2704

theorem carpet_reconstruction (original_length original_width cut_length cut_width new_side : ℝ) 
  (h1 : original_length = 12)
  (h2 : original_width = 9)
  (h3 : cut_length = 8)
  (h4 : cut_width = 1)
  (h5 : new_side = 10) :
  original_length * original_width - cut_length * cut_width = new_side * new_side := by
sorry

end carpet_reconstruction_l27_2704


namespace pascal_triangle_45th_number_51_entries_l27_2768

theorem pascal_triangle_45th_number_51_entries : 
  let n : ℕ := 50  -- The row number (0-indexed) with 51 entries
  let k : ℕ := 44  -- The position (0-indexed) of the 45th number
  Nat.choose n k = 19380000 := by sorry

end pascal_triangle_45th_number_51_entries_l27_2768


namespace min_rectangles_for_problem_figure_l27_2707

/-- Represents a corner in the figure -/
inductive Corner
| Type1
| Type2

/-- Represents a set of three Type2 corners -/
structure CornerSet :=
  (corners : Fin 3 → Corner)
  (all_type2 : ∀ i, corners i = Corner.Type2)

/-- The figure with its corner structure -/
structure Figure :=
  (total_corners : Nat)
  (type1_corners : Nat)
  (type2_corners : Nat)
  (corner_sets : Nat)
  (valid_total : total_corners = type1_corners + type2_corners)
  (valid_type2 : type2_corners = 3 * corner_sets)

/-- The minimum number of rectangles needed to cover the figure -/
def min_rectangles (f : Figure) : Nat :=
  f.type1_corners + f.corner_sets

/-- The specific figure from the problem -/
def problem_figure : Figure :=
  { total_corners := 24
  , type1_corners := 12
  , type2_corners := 12
  , corner_sets := 4
  , valid_total := by rfl
  , valid_type2 := by rfl }

theorem min_rectangles_for_problem_figure :
  min_rectangles problem_figure = 12 := by sorry

end min_rectangles_for_problem_figure_l27_2707


namespace infinitely_many_coprime_binomials_l27_2794

theorem infinitely_many_coprime_binomials (k l : ℕ+) :
  ∃ (S : Set ℕ), (∀ (m : ℕ), m ∈ S → m ≥ k) ∧
                 (Set.Infinite S) ∧
                 (∀ (m : ℕ), m ∈ S → Nat.gcd (Nat.choose m k) l = 1) :=
sorry

end infinitely_many_coprime_binomials_l27_2794


namespace simplify_trig_expression_l27_2791

theorem simplify_trig_expression (x : ℝ) : 
  Real.sqrt 2 * Real.cos x + Real.sqrt 6 * Real.sin x = 
  2 * Real.sqrt 2 * Real.cos (π / 3 - x) := by
  sorry

end simplify_trig_expression_l27_2791


namespace min_value_and_nonexistence_l27_2758

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) * Real.sqrt (a * b) = 1) :
  (∀ x y, x > 0 → y > 0 → (x + y) * Real.sqrt (x * y) = 1 → 1 / x^3 + 1 / y^3 ≥ 1 / a^3 + 1 / b^3) ∧
  1 / a^3 + 1 / b^3 = 4 * Real.sqrt 2 ∧
  ¬∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 1 / (2 * c) + 1 / (3 * d) = Real.sqrt 6 / 3 := by
  sorry

end min_value_and_nonexistence_l27_2758


namespace experiment_sequences_l27_2720

/-- The number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the possible positions for procedure A -/
inductive ProcedureAPosition
| First
| Last

/-- Represents a pair of adjacent procedures (C and D) -/
structure AdjacentPair where
  first : Fin num_procedures
  second : Fin num_procedures
  adjacent : first.val + 1 = second.val

/-- The total number of possible sequences in the experiment -/
def num_sequences : ℕ := 24

/-- Theorem stating the number of possible sequences in the experiment -/
theorem experiment_sequences :
  ∀ (a_pos : ProcedureAPosition) (cd_pair : AdjacentPair),
  num_sequences = 24 :=
sorry

end experiment_sequences_l27_2720


namespace cut_square_theorem_l27_2765

/-- Represents the dimensions of the original square -/
def original_size : ℕ := 8

/-- Represents the total length of cuts -/
def total_cut_length : ℕ := 54

/-- Represents the width of a rectangular piece -/
def rect_width : ℕ := 1

/-- Represents the length of a rectangular piece -/
def rect_length : ℕ := 4

/-- Represents the side length of a square piece -/
def square_side : ℕ := 2

/-- Represents the perimeter of the original square -/
def original_perimeter : ℕ := 4 * original_size

/-- Represents the total number of cells in the original square -/
def total_cells : ℕ := original_size * original_size

/-- Represents the number of cells covered by each piece (both rectangle and square) -/
def cells_per_piece : ℕ := square_side * square_side

theorem cut_square_theorem (num_rectangles num_squares : ℕ) :
  (num_rectangles + num_squares = total_cells / cells_per_piece) ∧
  (2 * total_cut_length + original_perimeter = 
   num_rectangles * (2 * (rect_width + rect_length)) + 
   num_squares * (4 * square_side)) →
  num_rectangles = 6 ∧ num_squares = 10 := by
  sorry

end cut_square_theorem_l27_2765


namespace josh_wallet_amount_l27_2782

def calculate_final_wallet_amount (initial_wallet : ℝ) (investment : ℝ) (debt : ℝ)
  (stock_a_percent : ℝ) (stock_b_percent : ℝ) (stock_c_percent : ℝ)
  (stock_a_change : ℝ) (stock_b_change : ℝ) (stock_c_change : ℝ) : ℝ :=
  let stock_a_value := investment * stock_a_percent * (1 + stock_a_change)
  let stock_b_value := investment * stock_b_percent * (1 + stock_b_change)
  let stock_c_value := investment * stock_c_percent * (1 + stock_c_change)
  let total_stock_value := stock_a_value + stock_b_value + stock_c_value
  let remaining_after_debt := total_stock_value - debt
  initial_wallet + remaining_after_debt

theorem josh_wallet_amount :
  calculate_final_wallet_amount 300 2000 500 0.4 0.3 0.3 0.2 0.3 (-0.1) = 2080 := by
  sorry

end josh_wallet_amount_l27_2782


namespace fib_999_1001_minus_1000_squared_l27_2717

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem stating that F₉₉₉ * F₁₀₀₁ - F₁₀₀₀² = 1 for the Fibonacci sequence -/
theorem fib_999_1001_minus_1000_squared :
  fib 999 * fib 1001 - fib 1000 * fib 1000 = 1 := by
  sorry

end fib_999_1001_minus_1000_squared_l27_2717


namespace coefficient_of_P_equals_30_l27_2714

/-- The generating function P as described in the problem -/
def P (x : Fin 6 → ℚ) : ℚ :=
  (1 / 24) * (
    (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^6 +
    6 * (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^2 * (x 0^4 + x 1^4 + x 2^4 + x 3^4 + x 4^4 + x 5^4) +
    3 * (x 0 + x 1 + x 2 + x 3 + x 4 + x 5)^2 * (x 0^2 + x 1^2 + x 2^2 + x 3^2 + x 4^2 + x 5^2)^2 +
    6 * (x 0^2 + x 1^2 + x 2^2 + x 3^2 + x 4^2 + x 5^2)^3 +
    8 * (x 0^3 + x 1^3 + x 2^3 + x 3^3 + x 4^3 + x 5^3)^2
  )

/-- The coefficient of x₁x₂x₃x₄x₅x₆ in the generating function P -/
def coefficient_x1x2x3x4x5x6 (P : (Fin 6 → ℚ) → ℚ) : ℚ :=
  sorry  -- Definition of how to extract the coefficient

theorem coefficient_of_P_equals_30 :
  coefficient_x1x2x3x4x5x6 P = 30 := by
  sorry

end coefficient_of_P_equals_30_l27_2714


namespace divisibility_problem_l27_2736

theorem divisibility_problem (a b c : ℕ+) 
  (h1 : a ∣ b^4) 
  (h2 : b ∣ c^4) 
  (h3 : c ∣ a^4) : 
  (a * b * c) ∣ (a + b + c)^21 := by
  sorry

end divisibility_problem_l27_2736


namespace rhombus_count_in_divided_equilateral_triangle_l27_2724

/-- Given an equilateral triangle ABC with each side divided into n equal parts,
    and parallel lines drawn through each division point to form a grid of smaller
    equilateral triangles, the number of rhombuses with side length 1/n in this grid
    is equal to 3 * C(n,2), where C(n,2) is the binomial coefficient. -/
theorem rhombus_count_in_divided_equilateral_triangle (n : ℕ) :
  let num_rhombuses := 3 * (n.choose 2)
  num_rhombuses = 3 * (n * (n - 1)) / 2 := by
  sorry

end rhombus_count_in_divided_equilateral_triangle_l27_2724


namespace school_pizza_profit_l27_2721

theorem school_pizza_profit :
  let num_pizzas : ℕ := 55
  let pizza_cost : ℚ := 685 / 100
  let slices_per_pizza : ℕ := 8
  let slice_price : ℚ := 1
  let total_revenue : ℚ := num_pizzas * slices_per_pizza * slice_price
  let total_cost : ℚ := num_pizzas * pizza_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 6325 / 100 := by
  sorry

end school_pizza_profit_l27_2721


namespace solution_set_f_range_of_m_l27_2757

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x + 1|

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m^2 + 3*m + 2*f x ≥ 0) ↔ (m ≤ -2 ∨ m ≥ -1) := by sorry

end solution_set_f_range_of_m_l27_2757


namespace total_short_trees_correct_park_short_trees_after_planting_l27_2799

/-- Calculates the total number of short trees after planting -/
def total_short_trees (initial_short_trees planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + planted_short_trees

/-- Proves that the total number of short trees after planting is correct -/
theorem total_short_trees_correct (initial_short_trees planted_short_trees : ℕ) :
  total_short_trees initial_short_trees planted_short_trees = initial_short_trees + planted_short_trees :=
by sorry

/-- Proves that the specific case in the problem is correct -/
theorem park_short_trees_after_planting :
  total_short_trees 41 57 = 98 :=
by sorry

end total_short_trees_correct_park_short_trees_after_planting_l27_2799


namespace not_all_squares_congruent_l27_2789

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent :
  ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry


end not_all_squares_congruent_l27_2789


namespace product_equals_zero_l27_2709

theorem product_equals_zero (a : ℤ) (h : a = 3) : 
  (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 := by
  sorry

end product_equals_zero_l27_2709


namespace salesman_profit_l27_2743

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit (total_backpacks : ℕ) (case_cost : ℕ)
  (swap_meet_qty : ℕ) (swap_meet_price : ℕ)
  (dept_store_qty : ℕ) (dept_store_price : ℕ)
  (online_qty : ℕ) (online_price : ℕ)
  (online_shipping : ℕ) (local_market_price : ℕ) :
  total_backpacks = 72 →
  case_cost = 1080 →
  swap_meet_qty = 25 →
  swap_meet_price = 20 →
  dept_store_qty = 18 →
  dept_store_price = 30 →
  online_qty = 12 →
  online_price = 28 →
  online_shipping = 40 →
  local_market_price = 24 →
  (swap_meet_qty * swap_meet_price +
   dept_store_qty * dept_store_price +
   online_qty * online_price - online_shipping +
   (total_backpacks - swap_meet_qty - dept_store_qty - online_qty) * local_market_price) -
  case_cost = 664 :=
by sorry

end salesman_profit_l27_2743


namespace quadratic_equation_roots_l27_2777

theorem quadratic_equation_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, x^2 + a*x + b = 0 ↔ x = -2*a ∨ x = b) →
  (b = -2*(-2*a) ∨ -2*a = -2*b) →
  a = -1/2 ∧ b = -1/2 := by
  sorry

end quadratic_equation_roots_l27_2777


namespace triangle_interior_angle_mean_l27_2759

theorem triangle_interior_angle_mean :
  ∀ (triangle_sum : ℝ) (num_angles : ℕ),
    triangle_sum = 180 →
    num_angles = 3 →
    triangle_sum / num_angles = 60 := by
  sorry

end triangle_interior_angle_mean_l27_2759


namespace waldo_puzzles_per_book_l27_2732

theorem waldo_puzzles_per_book 
  (num_books : ℕ) 
  (minutes_per_puzzle : ℕ) 
  (total_minutes : ℕ) 
  (h1 : num_books = 15)
  (h2 : minutes_per_puzzle = 3)
  (h3 : total_minutes = 1350) :
  total_minutes / minutes_per_puzzle / num_books = 30 := by
  sorry

end waldo_puzzles_per_book_l27_2732


namespace max_product_under_constraint_l27_2747

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4*b = 8) :
  ab ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4*b₀ = 8 ∧ a₀*b₀ = 4 :=
sorry

end max_product_under_constraint_l27_2747


namespace second_chord_length_l27_2754

/-- Represents a chord in a circle -/
structure Chord :=
  (length : ℝ)
  (segment1 : ℝ)
  (segment2 : ℝ)
  (valid : segment1 > 0 ∧ segment2 > 0 ∧ length = segment1 + segment2)

/-- Theorem: Length of the second chord given intersecting chords -/
theorem second_chord_length
  (chord1 : Chord)
  (chord2 : Chord)
  (h1 : chord1.segment1 = 12 ∧ chord1.segment2 = 18)
  (h2 : chord2.segment1 / chord2.segment2 = 3 / 8)
  (h3 : chord1.segment1 * chord1.segment2 = chord2.segment1 * chord2.segment2) :
  chord2.length = 33 :=
sorry

end second_chord_length_l27_2754


namespace sqrt_sum_equality_l27_2739

theorem sqrt_sum_equality : Real.sqrt 1 + Real.sqrt 9 = 4 := by
  sorry

end sqrt_sum_equality_l27_2739


namespace alexandra_magazines_l27_2755

/-- Alexandra's magazine problem -/
theorem alexandra_magazines :
  let friday_magazines : ℕ := 15
  let saturday_magazines : ℕ := 20
  let sunday_magazines : ℕ := 4 * friday_magazines
  let chewed_magazines : ℕ := 8
  let total_magazines : ℕ := friday_magazines + saturday_magazines + sunday_magazines - chewed_magazines
  total_magazines = 87 := by sorry

end alexandra_magazines_l27_2755


namespace pond_a_twice_pond_b_total_frogs_is_48_l27_2722

/-- The number of frogs in Pond A -/
def frogs_in_pond_a : ℕ := 32

/-- The number of frogs in Pond B -/
def frogs_in_pond_b : ℕ := frogs_in_pond_a / 2

/-- Pond A has twice as many frogs as Pond B -/
theorem pond_a_twice_pond_b : frogs_in_pond_a = 2 * frogs_in_pond_b := by sorry

/-- The total number of frogs in both ponds -/
def total_frogs : ℕ := frogs_in_pond_a + frogs_in_pond_b

/-- Theorem: The total number of frogs in both ponds is 48 -/
theorem total_frogs_is_48 : total_frogs = 48 := by sorry

end pond_a_twice_pond_b_total_frogs_is_48_l27_2722


namespace wilted_flowers_count_l27_2785

def flower_problem (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (remaining_bouquets : ℕ) : ℕ :=
  initial_flowers - (remaining_bouquets * flowers_per_bouquet)

theorem wilted_flowers_count :
  flower_problem 53 7 5 = 18 := by
  sorry

end wilted_flowers_count_l27_2785


namespace michaels_coins_value_l27_2734

theorem michaels_coins_value (p n : ℕ) : 
  p + n = 15 ∧ 
  n + 2 = 2 * (p - 2) →
  p * 1 + n * 5 = 47 :=
by sorry

end michaels_coins_value_l27_2734


namespace part1_part2_part3_l27_2756

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

-- Part 1
theorem part1 (m : ℝ) : (∀ x, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∀ x, f m x ≥ (m + 1) * x) ↔
  (m = -1 ∧ ∀ x, x ≥ 1) ∨
  (m > -1 ∧ ∀ x, x ≤ (m - 1) / (m + 1) ∨ x ≥ 1) ∨
  (m < -1 ∧ ∀ x, 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) :=
sorry

-- Part 3
theorem part3 (m : ℝ) : (∀ x ∈ Set.Icc (-1/2) (1/2), f m x ≥ 0) ↔ m ≥ 1 :=
sorry

end part1_part2_part3_l27_2756


namespace certain_number_proof_l27_2703

theorem certain_number_proof : ∃ x : ℤ, (287^2 : ℤ) + x^2 - 2*287*x = 324 ∧ x = 269 := by
  sorry

end certain_number_proof_l27_2703


namespace system_solution_l27_2701

theorem system_solution (x y z : ℝ) : 
  (x * (y^2 + z) = z * (z + x*y)) ∧ 
  (y * (z^2 + x) = x * (x + y*z)) ∧ 
  (z * (x^2 + y) = y * (y + x*z)) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) := by
  sorry

end system_solution_l27_2701


namespace line_inclination_angle_l27_2745

/-- The inclination angle of a line given by the equation x*cos(140°) + y*sin(40°) + 1 = 0 is 50°. -/
theorem line_inclination_angle (x y : ℝ) :
  x * Real.cos (140 * π / 180) + y * Real.sin (40 * π / 180) + 1 = 0 →
  Real.arctan (Real.tan (50 * π / 180)) = 50 * π / 180 :=
by sorry

end line_inclination_angle_l27_2745


namespace marley_fruit_count_l27_2790

/-- Represents the number of fruits a person has -/
structure FruitCount where
  oranges : ℕ
  apples : ℕ

/-- Calculates the total number of fruits -/
def totalFruits (fc : FruitCount) : ℕ :=
  fc.oranges + fc.apples

/-- The problem statement -/
theorem marley_fruit_count :
  let louis : FruitCount := ⟨5, 3⟩
  let samantha : FruitCount := ⟨8, 7⟩
  let marley : FruitCount := ⟨2 * louis.oranges, 3 * samantha.apples⟩
  totalFruits marley = 31 := by
  sorry


end marley_fruit_count_l27_2790


namespace geese_percentage_among_non_herons_l27_2735

theorem geese_percentage_among_non_herons :
  let total_birds : ℝ := 100
  let geese_percentage : ℝ := 30
  let swans_percentage : ℝ := 25
  let herons_percentage : ℝ := 20
  let ducks_percentage : ℝ := 25
  let non_heron_percentage : ℝ := total_birds - herons_percentage
  geese_percentage / non_heron_percentage * 100 = 37.5 := by
  sorry

end geese_percentage_among_non_herons_l27_2735


namespace candy_distribution_theorem_l27_2787

def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ 120 % (2 * n) = 0

theorem candy_distribution_theorem :
  ∀ n : ℕ, is_valid_student_count n ↔ n ∈ ({5, 6, 10, 12, 15} : Finset ℕ) :=
by sorry

end candy_distribution_theorem_l27_2787


namespace clock_cost_price_l27_2710

theorem clock_cost_price (total_clocks : ℕ) (clocks_sold_10_percent : ℕ) (clocks_sold_20_percent : ℕ)
  (profit_10_percent : ℝ) (profit_20_percent : ℝ) (uniform_profit : ℝ) (price_difference : ℝ) :
  total_clocks = 90 →
  clocks_sold_10_percent = 40 →
  clocks_sold_20_percent = 50 →
  profit_10_percent = 0.1 →
  profit_20_percent = 0.2 →
  uniform_profit = 0.15 →
  price_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price * (clocks_sold_10_percent * (1 + profit_10_percent) + 
      clocks_sold_20_percent * (1 + profit_20_percent)) - 
    cost_price * total_clocks * (1 + uniform_profit) = price_difference ∧
    cost_price = 80 :=
by sorry


end clock_cost_price_l27_2710


namespace roy_daily_sports_hours_l27_2750

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of hours Roy spends on sports in a week when he misses 2 days -/
def sports_hours_with_missed_days : ℕ := 6

/-- The number of days Roy misses in a week -/
def missed_days : ℕ := 2

/-- The number of hours Roy spends on sports activities in school every day -/
def daily_sports_hours : ℚ := 2

theorem roy_daily_sports_hours :
  daily_sports_hours = sports_hours_with_missed_days / (school_days_per_week - missed_days) :=
by sorry

end roy_daily_sports_hours_l27_2750


namespace sqrt_equation_solution_l27_2729

theorem sqrt_equation_solution :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x + Real.sqrt (x + 1) - Real.sqrt (x + 2) = 0 ∧ x = -1 + (2 * Real.sqrt 3) / 3 := by
  sorry

end sqrt_equation_solution_l27_2729


namespace xyz_product_zero_l27_2708

theorem xyz_product_zero (x y z : ℝ) 
  (eq1 : x + 1/y = 1) 
  (eq2 : y + 1/z = 1) 
  (eq3 : z + 1/x = 1) : 
  x * y * z = 0 := by
  sorry

end xyz_product_zero_l27_2708


namespace gcd_sum_ten_l27_2730

theorem gcd_sum_ten (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℕ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
sorry

end gcd_sum_ten_l27_2730


namespace purely_imaginary_complex_number_l27_2766

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 3*a + 2) + Complex.I * (a - 1)) → a = 2 :=
by sorry

end purely_imaginary_complex_number_l27_2766


namespace gcd_102_238_l27_2763

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l27_2763


namespace locus_of_C1_l27_2784

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define a chord parallel to x-axis
structure Chord :=
  (a : ℝ)
  (property : parabola a = parabola (-a))

-- Define a point on the parabola
structure ParabolaPoint :=
  (x : ℝ)
  (y : ℝ)
  (on_parabola : y = parabola x)

-- Define the circumcircle of a triangle
def circumcircle (A B C : ParabolaPoint) : Set (ℝ × ℝ) := sorry

-- Define a point on the circumcircle with the same x-coordinate as C
def C1 (C : ParabolaPoint) (circle : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- The main theorem
theorem locus_of_C1 (AB : Chord) (C : ParabolaPoint) 
  (hC : C.x ≠ AB.a ∧ C.x ≠ -AB.a) :
  let A := ⟨AB.a, parabola AB.a, rfl⟩
  let B := ⟨-AB.a, parabola (-AB.a), rfl⟩
  let circle := circumcircle A B C
  let c1 := C1 C circle
  c1.2 = 1 + AB.a^2 := by sorry

end locus_of_C1_l27_2784


namespace speech_competition_probability_l27_2798

theorem speech_competition_probability (n : ℕ) (h : n = 5) : 
  let total_arrangements := n.factorial
  let favorable_arrangements := (n - 1).factorial
  let prob_A_before_B := (total_arrangements / 2 : ℚ) / total_arrangements
  let prob_adjacent_and_A_before_B := (favorable_arrangements : ℚ) / total_arrangements
  (prob_adjacent_and_A_before_B / prob_A_before_B) = 2 / 5 := by
  sorry

end speech_competition_probability_l27_2798


namespace combined_tennis_percentage_l27_2744

theorem combined_tennis_percentage :
  let north_students : ℕ := 1800
  let south_students : ℕ := 2200
  let north_tennis_percentage : ℚ := 25 / 100
  let south_tennis_percentage : ℚ := 35 / 100
  let total_students := north_students + south_students
  let north_tennis_students := (north_students : ℚ) * north_tennis_percentage
  let south_tennis_students := (south_students : ℚ) * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let combined_percentage := total_tennis_students / (total_students : ℚ) * 100
  ⌊combined_percentage⌋ = 31 := by sorry

end combined_tennis_percentage_l27_2744


namespace minimum_class_size_minimum_class_size_is_21_l27_2776

theorem minimum_class_size : ℕ → Prop :=
  fun n =>
    ∃ (boys girls : ℕ),
      boys > 0 ∧ girls > 0 ∧
      (3 * boys = 4 * ((2 * girls) / 3)) ∧
      n = boys + girls + 4 ∧
      ∀ m, m < n →
        ¬∃ (b g : ℕ),
          b > 0 ∧ g > 0 ∧
          (3 * b = 4 * ((2 * g) / 3)) ∧
          m = b + g + 4

theorem minimum_class_size_is_21 :
  minimum_class_size 21 := by
  sorry

end minimum_class_size_minimum_class_size_is_21_l27_2776


namespace min_distance_sum_l27_2741

theorem min_distance_sum (x y z : ℝ) :
  Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) ≥ Real.sqrt 6 :=
by sorry

end min_distance_sum_l27_2741


namespace expression_equals_nine_l27_2726

theorem expression_equals_nine : 3 * 3 - 3 + 3 = 9 := by
  sorry

end expression_equals_nine_l27_2726


namespace x_value_l27_2713

theorem x_value : 
  let x := 98 * (1 + 20 / 100)
  x = 117.6 := by sorry

end x_value_l27_2713


namespace find_a_l27_2797

theorem find_a (b w : ℝ) (h1 : b = 2120) (h2 : w = 0.5) : ∃ a : ℝ, w = a / b ∧ a = 1060 := by
  sorry

end find_a_l27_2797


namespace special_function_property_l27_2770

/-- A function satisfying the given property for all real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, c^2 * g d = d^2 * g c

theorem special_function_property (g : ℝ → ℝ) (h1 : SatisfiesProperty g) (h2 : g 4 ≠ 0) :
  (g 7 - g 3) / g 4 = 2.5 := by
  sorry

end special_function_property_l27_2770


namespace student_congress_sample_size_l27_2749

/-- Calculates the sample size for a Student Congress given the number of classes and students selected per class. -/
def sampleSize (numClasses : ℕ) (studentsPerClass : ℕ) : ℕ :=
  numClasses * studentsPerClass

/-- Theorem stating that for a school with 40 classes, where each class selects 3 students
    for the Student Congress, the sample size is 120 students. -/
theorem student_congress_sample_size :
  sampleSize 40 3 = 120 := by
  sorry

#eval sampleSize 40 3

end student_congress_sample_size_l27_2749


namespace divisible_by_42_l27_2769

theorem divisible_by_42 (n : ℕ) : ∃ k : ℤ, (n ^ 3 * (n ^ 6 - 1) : ℤ) = 42 * k := by
  sorry

end divisible_by_42_l27_2769


namespace justin_age_proof_l27_2764

/-- Angelina's age in 5 years -/
def angelina_future_age : ℕ := 40

/-- Number of years until Angelina reaches her future age -/
def years_until_future : ℕ := 5

/-- Age difference between Angelina and Justin -/
def age_difference : ℕ := 4

/-- Justin's current age -/
def justin_current_age : ℕ := angelina_future_age - years_until_future - age_difference

theorem justin_age_proof : justin_current_age = 31 := by
  sorry

end justin_age_proof_l27_2764


namespace isosceles_right_triangle_leg_length_l27_2796

/-- An isosceles right triangle with a median to the hypotenuse of length 15 units has legs of length 15√2 units. -/
theorem isosceles_right_triangle_leg_length :
  ∀ (a b c m : ℝ),
  a = b →                          -- The triangle is isosceles
  a^2 + b^2 = c^2 →                -- The triangle is right-angled (Pythagorean theorem)
  m = 15 →                         -- The median to the hypotenuse is 15 units
  m = c / 2 →                      -- The median to the hypotenuse is half the hypotenuse length
  a = 15 * Real.sqrt 2 :=           -- The leg length is 15√2
by sorry

end isosceles_right_triangle_leg_length_l27_2796


namespace linear_equation_exponent_l27_2762

theorem linear_equation_exponent (k : ℕ) : 
  (∀ x, ∃ a b, x^(k-1) + 3 = a*x + b) → k = 2 :=
by sorry

end linear_equation_exponent_l27_2762


namespace smallest_coin_count_fifty_seven_satisfies_conditions_smallest_coin_count_is_57_l27_2786

theorem smallest_coin_count (n : ℕ) : 
  (n % 5 = 2) ∧ (n % 4 = 1) ∧ (n % 3 = 0) → n ≥ 57 :=
by
  sorry

theorem fifty_seven_satisfies_conditions : 
  (57 % 5 = 2) ∧ (57 % 4 = 1) ∧ (57 % 3 = 0) :=
by
  sorry

theorem smallest_coin_count_is_57 : 
  ∃ (n : ℕ), (n % 5 = 2) ∧ (n % 4 = 1) ∧ (n % 3 = 0) ∧ 
  (∀ (m : ℕ), (m % 5 = 2) ∧ (m % 4 = 1) ∧ (m % 3 = 0) → m ≥ n) ∧
  n = 57 :=
by
  sorry

end smallest_coin_count_fifty_seven_satisfies_conditions_smallest_coin_count_is_57_l27_2786


namespace planter_cost_theorem_l27_2728

/-- Represents the cost and quantity of a type of plant in a planter --/
structure PlantInfo where
  quantity : ℕ
  price : ℚ

/-- Calculates the total cost for a rectangle-shaped pool's corner planters --/
def total_cost (palm_fern : PlantInfo) (creeping_jenny : PlantInfo) (geranium : PlantInfo) : ℚ :=
  let cost_per_pot := palm_fern.quantity * palm_fern.price + 
                      creeping_jenny.quantity * creeping_jenny.price + 
                      geranium.quantity * geranium.price
  4 * cost_per_pot

/-- Theorem stating the total cost for the planters --/
theorem planter_cost_theorem (palm_fern : PlantInfo) (creeping_jenny : PlantInfo) (geranium : PlantInfo)
  (h1 : palm_fern.quantity = 1)
  (h2 : palm_fern.price = 15)
  (h3 : creeping_jenny.quantity = 4)
  (h4 : creeping_jenny.price = 4)
  (h5 : geranium.quantity = 4)
  (h6 : geranium.price = 7/2) :
  total_cost palm_fern creeping_jenny geranium = 180 := by
  sorry

end planter_cost_theorem_l27_2728


namespace r_daily_earnings_l27_2723

/-- Represents the daily earnings of individuals p, q, and r -/
structure Earnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : Earnings) : Prop :=
  9 * (e.p + e.q + e.r) = 1620 ∧
  5 * (e.p + e.r) = 600 ∧
  7 * (e.q + e.r) = 910

/-- The theorem stating that given the problem conditions, r's daily earnings are 70 -/
theorem r_daily_earnings (e : Earnings) : 
  problem_conditions e → e.r = 70 := by
  sorry

#check r_daily_earnings

end r_daily_earnings_l27_2723


namespace firewood_per_log_l27_2731

/-- Calculates the number of pieces of firewood per log -/
def piecesPerLog (totalPieces : ℕ) (totalTrees : ℕ) (logsPerTree : ℕ) : ℚ :=
  totalPieces / (totalTrees * logsPerTree)

theorem firewood_per_log :
  piecesPerLog 500 25 4 = 5 := by
  sorry

end firewood_per_log_l27_2731


namespace calvin_prevents_hobbes_win_l27_2733

/-- Represents a position on the integer lattice -/
structure Position where
  x : ℤ
  y : ℤ

/-- The game state -/
structure GameState where
  position : Position
  chosenIntegers : Set ℤ

/-- Calvin's strategy function -/
def calvinsStrategy (state : GameState) : Position := sorry

/-- Theorem stating Calvin can always prevent Hobbes from winning -/
theorem calvin_prevents_hobbes_win :
  ∀ (state : GameState),
  let newPos := calvinsStrategy state
  ∀ a b : ℤ,
    a ∉ state.chosenIntegers →
    b ∉ state.chosenIntegers →
    a ≠ (newPos.x - state.position.x) →
    b ≠ (newPos.y - state.position.y) →
    Position.mk (newPos.x + a) (newPos.y + b) ≠ Position.mk 0 0 :=
by sorry

end calvin_prevents_hobbes_win_l27_2733


namespace divisibility_implication_l27_2761

theorem divisibility_implication (n : ℕ) (h : n > 0) :
  (13 ∣ n^2 + 3*n + 51) → (169 ∣ 21*n^2 + 89*n + 44) := by
  sorry

end divisibility_implication_l27_2761


namespace spring_decrease_percentage_l27_2719

theorem spring_decrease_percentage (initial_increase : ℝ) (total_change : ℝ) : 
  initial_increase = 0.05 →
  total_change = -0.1495 →
  ∃ spring_decrease : ℝ, 
    (1 + initial_increase) * (1 - spring_decrease) = 1 + total_change ∧
    spring_decrease = 0.19 :=
by sorry

end spring_decrease_percentage_l27_2719


namespace hiking_problem_l27_2752

/-- Hiking Problem -/
theorem hiking_problem (up_rate : ℝ) (up_time : ℝ) (down_dist : ℝ) (rate_ratio : ℝ) :
  up_time = 2 →
  down_dist = 18 →
  rate_ratio = 1.5 →
  up_rate * up_time = down_dist / rate_ratio →
  up_rate = 6 := by
  sorry

end hiking_problem_l27_2752


namespace simplified_root_expression_l27_2737

theorem simplified_root_expression : 
  ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ 
  (3^5 * 5^4)^(1/4) = a * b^(1/4) ∧ 
  a + b = 18 := by sorry

end simplified_root_expression_l27_2737


namespace min_sum_arc_lengths_l27_2716

/-- A set of points on a circle consisting of n arcs -/
structure CircleSet (n : ℕ) where
  arcs : Fin n → Set ℝ
  sum_lengths : ℝ

/-- Rotation of a set of points on a circle -/
def rotate (α : ℝ) (F : Set ℝ) : Set ℝ := sorry

/-- Property that for any rotation, the rotated set intersects with the original set -/
def intersects_all_rotations (F : Set ℝ) : Prop :=
  ∀ α : ℝ, (rotate α F ∩ F).Nonempty

/-- Theorem stating the minimum sum of arc lengths -/
theorem min_sum_arc_lengths (n : ℕ) (F : CircleSet n) 
  (h : intersects_all_rotations (⋃ i, F.arcs i)) :
  F.sum_lengths ≥ 180 / n := sorry

end min_sum_arc_lengths_l27_2716


namespace home_run_difference_l27_2746

theorem home_run_difference (aaron_hr winfield_hr : ℕ) : 
  aaron_hr = 755 → winfield_hr = 465 → 2 * winfield_hr - aaron_hr = 175 := by
  sorry

end home_run_difference_l27_2746


namespace two_digit_integer_less_than_multiple_l27_2788

theorem two_digit_integer_less_than_multiple (n : ℕ) : n = 83 ↔ 
  (10 ≤ n ∧ n < 100) ∧ 
  (∃ k : ℕ, n + 1 = 3 * k) ∧ 
  (∃ k : ℕ, n + 1 = 4 * k) ∧ 
  (∃ k : ℕ, n + 1 = 7 * k) :=
by sorry

end two_digit_integer_less_than_multiple_l27_2788


namespace triangle_inequality_bound_l27_2718

theorem triangle_inequality_bound (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : b = 2 * a) :
  (a^2 + b^2) / c^2 > 5/9 := by
  sorry

end triangle_inequality_bound_l27_2718


namespace max_value_of_g_l27_2712

/-- Definition of the function f --/
def f (n : ℕ+) : ℕ := 70 + n^2

/-- Definition of the function g --/
def g (n : ℕ+) : ℕ := Nat.gcd (f n) (f (n + 1))

/-- Theorem stating the maximum value of g(n) --/
theorem max_value_of_g :
  ∃ (m : ℕ+), ∀ (n : ℕ+), g n ≤ g m ∧ g m = 281 :=
sorry

end max_value_of_g_l27_2712


namespace arithmetic_expression_equality_l27_2727

theorem arithmetic_expression_equality : 10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3 = -4 := by
  sorry

end arithmetic_expression_equality_l27_2727


namespace equations_solution_set_l27_2706

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(0,0,0,0), (2,2,2,2), (1,5,2,3), (5,1,2,3), (1,5,3,2), (5,1,3,2),
   (2,3,1,5), (2,3,5,1), (3,2,1,5), (3,2,5,1)}

def satisfies_equations (x y z t : ℕ) : Prop :=
  x + y = z + t ∧ z + t = x * y

theorem equations_solution_set :
  ∀ x y z t : ℕ, satisfies_equations x y z t ↔ (x, y, z, t) ∈ solution_set := by
  sorry

end equations_solution_set_l27_2706


namespace martha_initial_blocks_l27_2792

/-- Given that Martha finds 80 blocks and ends up with 84 blocks, 
    prove that she initially had 4 blocks. -/
theorem martha_initial_blocks : 
  ∀ (initial_blocks found_blocks final_blocks : ℕ),
    found_blocks = 80 →
    final_blocks = 84 →
    final_blocks = initial_blocks + found_blocks →
    initial_blocks = 4 :=
by
  sorry

end martha_initial_blocks_l27_2792


namespace opposite_unit_vector_l27_2779

def a : ℝ × ℝ := (12, 5)

theorem opposite_unit_vector :
  let magnitude := Real.sqrt (a.1^2 + a.2^2)
  (-a.1 / magnitude, -a.2 / magnitude) = (-12/13, -5/13) := by
  sorry

end opposite_unit_vector_l27_2779


namespace non_chihuahua_male_dogs_l27_2705

theorem non_chihuahua_male_dogs (total_dogs : ℕ) (male_ratio : ℚ) (chihuahua_ratio : ℚ) :
  total_dogs = 32 →
  male_ratio = 5/8 →
  chihuahua_ratio = 3/4 →
  (total_dogs : ℚ) * male_ratio * (1 - chihuahua_ratio) = 5 := by
  sorry

end non_chihuahua_male_dogs_l27_2705


namespace units_digit_of_product_division_l27_2767

theorem units_digit_of_product_division : 
  (30 * 31 * 32 * 33 * 34 * 35) / 2500 % 10 = 2 := by sorry

end units_digit_of_product_division_l27_2767


namespace exponential_equation_sum_of_reciprocals_l27_2781

theorem exponential_equation_sum_of_reciprocals (x y : ℝ) 
  (h1 : 3^x = Real.sqrt 12) 
  (h2 : 4^y = Real.sqrt 12) : 
  1/x + 1/y = 2 := by
  sorry

end exponential_equation_sum_of_reciprocals_l27_2781


namespace decimal_rep_denominators_num_possible_denominators_l27_2783

-- Define a digit as a natural number between 0 and 9
def Digit := {n : ℕ // n ≤ 9}

-- Define the decimal representation
def DecimalRep (a b c : Digit) : ℚ := (100 * a.val + 10 * b.val + c.val : ℕ) / 999

-- Define the condition that not all digits are nine
def NotAllNine (a b c : Digit) : Prop :=
  ¬(a.val = 9 ∧ b.val = 9 ∧ c.val = 9)

-- Define the condition that not all digits are zero
def NotAllZero (a b c : Digit) : Prop :=
  ¬(a.val = 0 ∧ b.val = 0 ∧ c.val = 0)

-- Define the set of possible denominators
def PossibleDenominators : Finset ℕ :=
  {3, 9, 27, 37, 111, 333, 999}

-- The main theorem
theorem decimal_rep_denominators (a b c : Digit) 
  (h1 : NotAllNine a b c) (h2 : NotAllZero a b c) :
  (DecimalRep a b c).den ∈ PossibleDenominators := by
  sorry

-- The final result
theorem num_possible_denominators :
  Finset.card PossibleDenominators = 7 := by
  sorry

end decimal_rep_denominators_num_possible_denominators_l27_2783


namespace books_read_l27_2711

/-- The number of books read in the 'crazy silly school' series -/
theorem books_read (total_books : ℕ) (unread_books : ℕ) (h1 : total_books = 20) (h2 : unread_books = 5) :
  total_books - unread_books = 15 := by
  sorry

end books_read_l27_2711


namespace function_value_at_negative_two_l27_2738

/-- Given a function f(x) = ax^4 + bx^2 - x + 1 where a and b are real numbers,
    if f(2) = 9, then f(-2) = 13 -/
theorem function_value_at_negative_two
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 - x + 1)
  (h2 : f 2 = 9) :
  f (-2) = 13 := by
sorry

end function_value_at_negative_two_l27_2738


namespace discontinuous_when_limit_not_equal_value_l27_2748

-- Define a multivariable function type
def MultivariableFunction (α : Type*) (β : Type*) := α → β

-- Define the concept of a limit for a multivariable function
def HasLimit (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - L) < ε

-- Define continuity for a multivariable function
def IsContinuousAt (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) : Prop :=
  ∃ L, HasLimit f x₀ L ∧ f x₀ = L

-- Theorem statement
theorem discontinuous_when_limit_not_equal_value
  (f : MultivariableFunction ℝ ℝ) (x₀ : ℝ) (L : ℝ) :
  HasLimit f x₀ L → f x₀ ≠ L → ¬(IsContinuousAt f x₀) :=
sorry

end discontinuous_when_limit_not_equal_value_l27_2748


namespace largest_stamps_per_page_l27_2780

theorem largest_stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 1520)
  (h2 : book2 = 1900)
  (h3 : book3 = 2280) :
  Nat.gcd book1 (Nat.gcd book2 book3) = 380 := by
  sorry

end largest_stamps_per_page_l27_2780


namespace puzzle_solution_l27_2760

theorem puzzle_solution :
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    1448 = 282 * a + 10 * a + b ∧
    423 * (c / 3) = 282 ∧
    47 * 9 = 423 ∧
    423 * (2 / 3) = 282 ∧
    282 * 5 = 1410 ∧
    1410 + 38 = 1448 ∧
    705 + 348 = 1053 := by
  sorry

end puzzle_solution_l27_2760


namespace olivia_earnings_l27_2740

def hourly_wage : ℕ := 9
def monday_hours : ℕ := 4
def wednesday_hours : ℕ := 3
def friday_hours : ℕ := 6

theorem olivia_earnings : 
  hourly_wage * (monday_hours + wednesday_hours + friday_hours) = 117 := by
  sorry

end olivia_earnings_l27_2740


namespace inequality_equiv_range_l27_2775

/-- The function f(x) = x³ + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- The theorem stating the equivalence between the inequality and the range of x -/
theorem inequality_equiv_range :
  ∀ x : ℝ, (f (1 - x) + f (2 * x) > 2) ↔ x > -1 :=
sorry

end inequality_equiv_range_l27_2775


namespace farm_feet_count_l27_2771

/-- Given a farm with hens and cows, calculate the total number of feet -/
theorem farm_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 46 → hen_count = 22 → (hen_count * 2 + (total_heads - hen_count) * 4 = 140) :=
by
  sorry

#check farm_feet_count

end farm_feet_count_l27_2771


namespace inverse_proposition_l27_2702

theorem inverse_proposition :
  (∀ x a b : ℝ, x ≥ a^2 + b^2 → x ≥ 2*a*b) →
  (∀ x a b : ℝ, x ≥ 2*a*b → x ≥ a^2 + b^2) :=
by sorry

end inverse_proposition_l27_2702


namespace quadratic_roots_properties_l27_2795

theorem quadratic_roots_properties (b c x₁ x₂ : ℝ) 
  (h_eq : ∀ x, x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂)
  (h_distinct : x₁ ≠ x₂)
  (h_order : x₁ < x₂)
  (h_x₁_range : -1 < x₁ ∧ x₁ < 0) :
  (x₂ > 0 → c < 0) ∧
  (|x₂ - x₁| = 2 → |1 - b + c| - |1 + b + c| > 2*|4 + 2*b + c| - 6) := by
sorry

end quadratic_roots_properties_l27_2795


namespace jim_total_cost_l27_2751

def total_cost (lamp_price bulb_price bedside_table_price decorative_item_price : ℝ)
               (lamp_quantity bulb_quantity bedside_table_quantity decorative_item_quantity : ℕ)
               (lamp_discount bulb_discount bedside_table_discount decorative_item_discount : ℝ)
               (lamp_tax_rate bulb_tax_rate bedside_table_tax_rate decorative_item_tax_rate : ℝ) : ℝ :=
  let lamp_cost := lamp_quantity * lamp_price * (1 - lamp_discount) * (1 + lamp_tax_rate)
  let bulb_cost := bulb_quantity * bulb_price * (1 - bulb_discount) * (1 + bulb_tax_rate)
  let bedside_table_cost := bedside_table_quantity * bedside_table_price * (1 - bedside_table_discount) * (1 + bedside_table_tax_rate)
  let decorative_item_cost := decorative_item_quantity * decorative_item_price * (1 - decorative_item_discount) * (1 + decorative_item_tax_rate)
  lamp_cost + bulb_cost + bedside_table_cost + decorative_item_cost

theorem jim_total_cost :
  total_cost 12 8 25 10 2 6 3 4 0.2 0.3 0 0.15 0.05 0.05 0.06 0.04 = 170.30 := by
  sorry

end jim_total_cost_l27_2751
