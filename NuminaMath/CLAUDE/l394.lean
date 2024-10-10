import Mathlib

namespace equal_sum_product_square_diff_l394_39420

theorem equal_sum_product_square_diff : ∃ (x y : ℝ),
  (x + y = x * y) ∧ (x + y = x^2 - y^2) ∧
  ((x = (3 + Real.sqrt 5) / 2 ∧ y = (1 + Real.sqrt 5) / 2) ∨
   (x = (3 - Real.sqrt 5) / 2 ∧ y = (1 - Real.sqrt 5) / 2) ∨
   (x = 0 ∧ y = 0)) :=
by sorry


end equal_sum_product_square_diff_l394_39420


namespace mike_pens_l394_39423

/-- The number of pens Mike gave -/
def M : ℕ := sorry

/-- The initial number of pens -/
def initial_pens : ℕ := 5

/-- The number of pens given away -/
def pens_given_away : ℕ := 19

/-- The final number of pens -/
def final_pens : ℕ := 31

theorem mike_pens : 
  2 * (initial_pens + M) - pens_given_away = final_pens ∧ M = 20 := by sorry

end mike_pens_l394_39423


namespace equation_solution_l394_39431

theorem equation_solution : ∃ (x y z : ℝ), 
  2 * Real.sqrt (x - 4) + 3 * Real.sqrt (y - 9) + 4 * Real.sqrt (z - 16) = (1/2) * (x + y + z) ∧
  x = 8 ∧ y = 18 ∧ z = 32 := by
  sorry

end equation_solution_l394_39431


namespace two_true_statements_l394_39437

theorem two_true_statements : 
  let original := ∀ a : ℝ, a > -5 → a > -8
  let converse := ∀ a : ℝ, a > -8 → a > -5
  let inverse := ∀ a : ℝ, a ≤ -5 → a ≤ -8
  let contrapositive := ∀ a : ℝ, a ≤ -8 → a ≤ -5
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by
  sorry

end two_true_statements_l394_39437


namespace lagrange_four_square_theorem_l394_39426

-- Define the property of being expressible as the sum of four squares
def SumOfFourSquares (n : ℕ) : Prop :=
  ∃ a b c d : ℤ, n = a^2 + b^2 + c^2 + d^2

-- State Lagrange's Four Square Theorem
theorem lagrange_four_square_theorem :
  ∀ n : ℕ, SumOfFourSquares n :=
by
  sorry

-- State the given conditions
axiom odd_prime_four_squares :
  ∀ p : ℕ, Nat.Prime p → p % 2 = 1 → SumOfFourSquares p

axiom two_four_squares : SumOfFourSquares 2

axiom product_four_squares :
  ∀ a b : ℕ, SumOfFourSquares a → SumOfFourSquares b → SumOfFourSquares (a * b)

end lagrange_four_square_theorem_l394_39426


namespace min_reciprocal_sum_l394_39425

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 4) :
  1/x + 1/y ≥ 1 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧ 1/a + 1/b = 1 :=
by sorry

end min_reciprocal_sum_l394_39425


namespace vector_sum_squared_norms_l394_39408

theorem vector_sum_squared_norms (a b : ℝ × ℝ) :
  let m : ℝ × ℝ := (4, 10)  -- midpoint
  (∀ (x : ℝ) (y : ℝ), m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) →  -- midpoint condition
  (a.1 * b.1 + a.2 * b.2 = 12) →  -- dot product condition
  (a.1^2 + a.2^2) + (b.1^2 + b.2^2) = 440 := by
  sorry

end vector_sum_squared_norms_l394_39408


namespace largest_int_less_100_rem_5_div_8_l394_39446

theorem largest_int_less_100_rem_5_div_8 : 
  ∃ (n : ℕ), n < 100 ∧ n % 8 = 5 ∧ ∀ (m : ℕ), m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_int_less_100_rem_5_div_8_l394_39446


namespace walking_time_difference_l394_39454

/-- The speed of person A in km/h -/
def speed_A : ℝ := 4

/-- The speed of person B in km/h -/
def speed_B : ℝ := 4.555555555555555

/-- The time in hours after which B overtakes A -/
def overtake_time : ℝ := 1.8

/-- The time in hours after A started that B starts walking -/
def time_diff : ℝ := 0.25

theorem walking_time_difference :
  speed_A * (time_diff + overtake_time) = speed_B * overtake_time := by
  sorry

end walking_time_difference_l394_39454


namespace max_sum_given_constraints_l394_39453

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 98) 
  (h2 : x * y = 40) : 
  x + y ≤ Real.sqrt 178 := by
sorry

end max_sum_given_constraints_l394_39453


namespace locus_and_circle_existence_l394_39415

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 18
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2

-- Define the locus of the center of circle M
def locus (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the circle centered at the origin
def origin_circle (x y : ℝ) : Prop := x^2 + y^2 = 8 / 3

-- Define the tangency and orthogonality conditions
def tangent_intersects_locus (t m n : ℝ × ℝ) : Prop :=
  locus m.1 m.2 ∧ locus n.1 n.2 ∧ 
  (∃ k b : ℝ, t.2 = k * t.1 + b ∧ origin_circle t.1 t.2)

def orthogonal (o m n : ℝ × ℝ) : Prop :=
  (m.1 - o.1) * (n.1 - o.1) + (m.2 - o.2) * (n.2 - o.2) = 0

-- Main theorem
theorem locus_and_circle_existence :
  (∀ x y : ℝ, C₁ x y ∨ C₂ x y → 
    ∃ m : ℝ × ℝ, locus m.1 m.2 ∧
    (∀ t : ℝ × ℝ, origin_circle t.1 t.2 → 
      ∃ n : ℝ × ℝ, tangent_intersects_locus t m n ∧ 
        orthogonal (0, 0) m n)) :=
sorry

end locus_and_circle_existence_l394_39415


namespace fifteen_segments_two_monochromatic_triangles_fourteen_segments_no_monochromatic_triangle_possible_l394_39452

/-- Represents a segment (edge or diagonal) in a regular hexagon --/
inductive Segment
| Edge : Fin 6 → Fin 6 → Segment
| Diagonal : Fin 6 → Fin 6 → Segment

/-- Represents the color of a segment --/
inductive Color
| Red
| Blue

/-- Represents a coloring of segments in a regular hexagon --/
def Coloring := Segment → Option Color

/-- Checks if a triangle is monochromatic --/
def isMonochromatic (c : Coloring) (v1 v2 v3 : Fin 6) : Bool :=
  sorry

/-- Counts the number of colored segments in a coloring --/
def countColoredSegments (c : Coloring) : Nat :=
  sorry

/-- Counts the number of monochromatic triangles in a coloring --/
def countMonochromaticTriangles (c : Coloring) : Nat :=
  sorry

/-- Theorem: If 15 segments are colored, there are at least two monochromatic triangles --/
theorem fifteen_segments_two_monochromatic_triangles (c : Coloring) :
  countColoredSegments c = 15 → countMonochromaticTriangles c ≥ 2 :=
  sorry

/-- Theorem: It's possible to color 14 segments without forming a monochromatic triangle --/
theorem fourteen_segments_no_monochromatic_triangle_possible :
  ∃ c : Coloring, countColoredSegments c = 14 ∧ countMonochromaticTriangles c = 0 :=
  sorry

end fifteen_segments_two_monochromatic_triangles_fourteen_segments_no_monochromatic_triangle_possible_l394_39452


namespace average_age_increase_l394_39444

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℚ) (teacher_age : ℕ) : 
  num_students = 25 →
  student_avg_age = 26 →
  teacher_age = 52 →
  (student_avg_age * num_students + teacher_age) / (num_students + 1) - student_avg_age = 1 := by
  sorry

end average_age_increase_l394_39444


namespace shooting_probability_l394_39469

/-- The probability of person A hitting the target -/
def prob_A : ℚ := 3/4

/-- The probability of person B hitting the target -/
def prob_B : ℚ := 4/5

/-- The probability of the event where A has shot twice when they stop -/
def prob_A_shoots_twice : ℚ := 19/400

theorem shooting_probability :
  let prob_A_miss := 1 - prob_A
  let prob_B_miss := 1 - prob_B
  prob_A_shoots_twice = 
    (prob_A_miss * prob_B_miss * prob_A) + 
    (prob_A_miss * prob_B_miss * prob_A_miss * prob_B) :=
by sorry

end shooting_probability_l394_39469


namespace external_tangent_circle_l394_39429

/-- Given circle C with equation (x-2)^2 + (y+1)^2 = 4 and point A(4, -1) on C,
    prove that the circle with equation (x-5)^2 + (y+1)^2 = 1 is externally
    tangent to C at A and has radius 1. -/
theorem external_tangent_circle
  (C : Set (ℝ × ℝ))
  (A : ℝ × ℝ)
  (hC : C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4})
  (hA : A = (4, -1))
  (hA_on_C : A ∈ C)
  : ∃ (M : Set (ℝ × ℝ)),
    M = {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 1)^2 = 1} ∧
    (∀ p ∈ M, ∃ q ∈ C, (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1) ∧
    A ∈ M ∧
    (∀ p ∈ M, (p.1 - 5)^2 + (p.2 + 1)^2 = 1) :=
sorry

end external_tangent_circle_l394_39429


namespace total_bales_in_barn_l394_39455

def initial_bales : ℕ := 54
def added_bales : ℕ := 28

theorem total_bales_in_barn : initial_bales + added_bales = 82 := by
  sorry

end total_bales_in_barn_l394_39455


namespace square_difference_inapplicable_l394_39458

/-- The square difference formula cannot be applied to (2x+3y)(-3y-2x) -/
theorem square_difference_inapplicable (x y : ℝ) :
  ¬ ∃ (a b : ℝ), (∃ (c₁ c₂ c₃ c₄ : ℝ), a = c₁ * x + c₂ * y ∧ b = c₃ * x + c₄ * y) ∧
    (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) := by
  sorry

end square_difference_inapplicable_l394_39458


namespace quadratic_inequality_solution_quadratic_inequality_empty_solution_l394_39484

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 - 2 * x + 3 * k < 0

def solution_set_case1 (x : ℝ) : Prop :=
  x < -3 ∨ x > -1

def solution_set_case2 : Set ℝ :=
  ∅

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x, quadratic_inequality k x ↔ solution_set_case1 x) → k = -1/2 :=
sorry

theorem quadratic_inequality_empty_solution (k : ℝ) :
  (∀ x, ¬ quadratic_inequality k x) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end quadratic_inequality_solution_quadratic_inequality_empty_solution_l394_39484


namespace count_l_shapes_l394_39428

/-- The number of ways to select an L-shaped piece from an m × n chessboard -/
def lShapeCount (m n : ℕ) : ℕ :=
  4 * (m - 1) * (n - 1)

/-- Theorem stating that the number of ways to select an L-shaped piece
    from an m × n chessboard is equal to 4(m-1)(n-1) -/
theorem count_l_shapes (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  lShapeCount m n = 4 * (m - 1) * (n - 1) := by
  sorry

#check count_l_shapes

end count_l_shapes_l394_39428


namespace line_through_quadrants_l394_39406

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point (x, y) is in the first quadrant -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Predicate to check if a line passes through a given quadrant -/
def passes_through_quadrant (l : Line) (quad : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, quad x y ∧ l.a * x + l.b * y + l.c = 0

/-- Main theorem: If a line passes through the first, second, and fourth quadrants,
    then ac > 0 and bc < 0 -/
theorem line_through_quadrants (l : Line) :
  passes_through_quadrant l in_first_quadrant ∧
  passes_through_quadrant l in_second_quadrant ∧
  passes_through_quadrant l in_fourth_quadrant →
  l.a * l.c > 0 ∧ l.b * l.c < 0 := by
  sorry

end line_through_quadrants_l394_39406


namespace georgia_carnation_cost_l394_39462

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 1/2

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia sent carnations to -/
def num_teachers : ℕ := 5

/-- The number of friends Georgia bought carnations for -/
def num_friends : ℕ := 14

/-- The total cost of carnations Georgia would spend -/
def total_cost : ℚ := num_teachers * dozen_carnation_cost + num_friends * single_carnation_cost

theorem georgia_carnation_cost : total_cost = 27 := by sorry

end georgia_carnation_cost_l394_39462


namespace farmer_radishes_per_row_l394_39445

/-- Represents the farmer's planting scenario -/
structure FarmerPlanting where
  bean_seedlings : ℕ
  bean_per_row : ℕ
  pumpkin_seeds : ℕ
  pumpkin_per_row : ℕ
  total_radishes : ℕ
  rows_per_bed : ℕ
  total_beds : ℕ

/-- Calculates the number of radishes per row -/
def radishes_per_row (fp : FarmerPlanting) : ℕ :=
  let bean_rows := fp.bean_seedlings / fp.bean_per_row
  let pumpkin_rows := fp.pumpkin_seeds / fp.pumpkin_per_row
  let total_rows := fp.rows_per_bed * fp.total_beds
  let radish_rows := total_rows - (bean_rows + pumpkin_rows)
  fp.total_radishes / radish_rows

/-- Theorem stating that given the farmer's planting conditions, 
    the number of radishes per row is 6 -/
theorem farmer_radishes_per_row :
  let fp : FarmerPlanting := {
    bean_seedlings := 64,
    bean_per_row := 8,
    pumpkin_seeds := 84,
    pumpkin_per_row := 7,
    total_radishes := 48,
    rows_per_bed := 2,
    total_beds := 14
  }
  radishes_per_row fp = 6 := by
  sorry

end farmer_radishes_per_row_l394_39445


namespace decimal_equivalences_l394_39447

-- Define the decimal number
def decimal_number : ℚ := 209 / 100

-- Theorem to prove the equivalence
theorem decimal_equivalences :
  -- Percentage equivalence
  (decimal_number * 100 : ℚ) = 209 ∧
  -- Simplified fraction equivalence
  decimal_number = 209 / 100 ∧
  -- Mixed number equivalence
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    whole = 2 ∧
    numerator = 9 ∧
    denominator = 100 ∧
    decimal_number = whole + (numerator : ℚ) / denominator :=
by
  sorry

end decimal_equivalences_l394_39447


namespace intersection_of_M_and_N_l394_39443

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l394_39443


namespace boltons_class_size_l394_39419

theorem boltons_class_size :
  ∀ (S : ℚ),
  (2 / 5 : ℚ) * S + (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S) + ((3 / 5 : ℚ) * S - (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S)) = S →
  (2 / 5 : ℚ) * S + ((3 / 5 : ℚ) * S - (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S)) = 20 →
  S = 25 := by
  sorry

end boltons_class_size_l394_39419


namespace percentage_difference_in_gain_l394_39436

def cost_price : ℝ := 400
def selling_price1 : ℝ := 360
def selling_price2 : ℝ := 340

def gain1 : ℝ := selling_price1 - cost_price
def gain2 : ℝ := selling_price2 - cost_price

def difference_in_gain : ℝ := gain1 - gain2

theorem percentage_difference_in_gain :
  (difference_in_gain / cost_price) * 100 = 5 := by sorry

end percentage_difference_in_gain_l394_39436


namespace solve_equation_l394_39449

theorem solve_equation (B : ℝ) : 
  80 - (5 - (6 + 2 * (B - 8 - 5))) = 89 ↔ B = 17 := by
  sorry

end solve_equation_l394_39449


namespace polynomial_equality_sum_l394_39485

theorem polynomial_equality_sum (a b c d : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - x^2 + 18*x + 24) →
  a + b + c + d = 12 := by
  sorry

end polynomial_equality_sum_l394_39485


namespace odd_function_with_conditions_l394_39410

def f (a b c x : ℤ) : ℚ := (a * x^2 + 1) / (b * x + c)

theorem odd_function_with_conditions (a b c : ℤ) :
  (∀ x, f a b c (-x) = -f a b c x) →  -- f is an odd function
  f a b c 1 = 2 →                     -- f(1) = 2
  f a b c 2 < 3 →                     -- f(2) < 3
  a = 1 ∧ b = 1 ∧ c = 0 :=            -- conclusion: a = 1, b = 1, c = 0
by sorry

end odd_function_with_conditions_l394_39410


namespace only_option_C_is_well_defined_set_l394_39487

-- Define the universe of discourse
def Universe : Type := String

-- Define the property of being a well-defined set
def is_well_defined_set (S : Set Universe) : Prop :=
  ∀ x, ∃ (decision : Prop), (x ∈ S ↔ decision)

-- Define the four options
def option_A : Set Universe := {x | x = "Tall students in the first grade of Fengdu Middle School in January 2013"}
def option_B : Set Universe := {x | x = "Tall trees in the campus"}
def option_C : Set Universe := {x | x = "Students in the first grade of Fengdu Middle School in January 2013"}
def option_D : Set Universe := {x | x = "Students with high basketball skills in the school"}

-- Theorem statement
theorem only_option_C_is_well_defined_set :
  is_well_defined_set option_C ∧
  ¬is_well_defined_set option_A ∧
  ¬is_well_defined_set option_B ∧
  ¬is_well_defined_set option_D :=
sorry

end only_option_C_is_well_defined_set_l394_39487


namespace nancy_crayon_packs_l394_39482

/-- The number of crayons Nancy bought -/
def total_crayons : ℕ := 615

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def number_of_packs : ℕ := total_crayons / crayons_per_pack

theorem nancy_crayon_packs : number_of_packs = 41 := by
  sorry

end nancy_crayon_packs_l394_39482


namespace arctan_equation_solutions_l394_39466

theorem arctan_equation_solutions (x : ℝ) : 
  (Real.arctan (2 / x) + Real.arctan (1 / x^2) = π / 4) ↔ 
  (x = 3 ∨ x = (-3 + Real.sqrt 5) / 2) :=
sorry

end arctan_equation_solutions_l394_39466


namespace rent_expense_calculation_l394_39467

def monthly_salary : ℕ := 23500
def savings_percentage : ℚ := 1/10
def savings : ℕ := 2350
def milk_expense : ℕ := 1500
def groceries_expense : ℕ := 4500
def education_expense : ℕ := 2500
def petrol_expense : ℕ := 2000
def miscellaneous_expense : ℕ := 5650

theorem rent_expense_calculation :
  let total_expenses := milk_expense + groceries_expense + education_expense + petrol_expense + miscellaneous_expense
  let rent := monthly_salary - savings - total_expenses
  rent = 4850 := by sorry

end rent_expense_calculation_l394_39467


namespace swan_percentage_among_non_ducks_l394_39479

theorem swan_percentage_among_non_ducks (total : ℝ) (ducks swans herons geese : ℝ) :
  total = 100 →
  ducks = 35 →
  swans = 30 →
  herons = 20 →
  geese = 15 →
  (swans / (total - ducks)) * 100 = 46.15 := by
  sorry

end swan_percentage_among_non_ducks_l394_39479


namespace smallest_factor_l394_39493

theorem smallest_factor (n : ℕ) : n = 900 ↔ 
  (∀ m : ℕ, m > 0 → m < n → ¬(2^5 ∣ (936 * m) ∧ 3^3 ∣ (936 * m) ∧ 10^2 ∣ (936 * m))) ∧
  (2^5 ∣ (936 * n) ∧ 3^3 ∣ (936 * n) ∧ 10^2 ∣ (936 * n)) ∧
  (n > 0) :=
by sorry

end smallest_factor_l394_39493


namespace triangle_345_not_triangle_123_not_triangle_384_not_triangle_5510_l394_39409

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_345 : can_form_triangle 3 4 5 := by sorry

/-- Theorem stating that line segments of lengths 1, 2, and 3 cannot form a triangle -/
theorem not_triangle_123 : ¬can_form_triangle 1 2 3 := by sorry

/-- Theorem stating that line segments of lengths 3, 8, and 4 cannot form a triangle -/
theorem not_triangle_384 : ¬can_form_triangle 3 8 4 := by sorry

/-- Theorem stating that line segments of lengths 5, 5, and 10 cannot form a triangle -/
theorem not_triangle_5510 : ¬can_form_triangle 5 5 10 := by sorry

end triangle_345_not_triangle_123_not_triangle_384_not_triangle_5510_l394_39409


namespace three_digit_numbers_with_repeats_eq_252_l394_39442

/-- The number of three-digit numbers with repeated digits using digits 0 to 9 -/
def three_digit_numbers_with_repeats : ℕ :=
  let total_three_digit_numbers := 9 * 10 * 10  -- First digit can't be 0
  let three_digit_numbers_without_repeats := 9 * 9 * 8
  total_three_digit_numbers - three_digit_numbers_without_repeats

/-- Theorem stating that the number of three-digit numbers with repeated digits is 252 -/
theorem three_digit_numbers_with_repeats_eq_252 : 
  three_digit_numbers_with_repeats = 252 := by
  sorry

end three_digit_numbers_with_repeats_eq_252_l394_39442


namespace workshop_assignment_l394_39480

theorem workshop_assignment (total_workers : ℕ) 
  (type_a_rate type_b_rate : ℕ) (ratio_a ratio_b : ℕ) 
  (type_a_workers : ℕ) (type_b_workers : ℕ) : 
  total_workers = 90 →
  type_a_rate = 15 →
  type_b_rate = 8 →
  ratio_a = 3 →
  ratio_b = 2 →
  type_a_workers = 40 →
  type_b_workers = 50 →
  total_workers = type_a_workers + type_b_workers →
  ratio_a * (type_b_rate * type_b_workers) = ratio_b * (type_a_rate * type_a_workers) := by
  sorry

#check workshop_assignment

end workshop_assignment_l394_39480


namespace arithmetic_sequence_ratio_l394_39427

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n a n / sum_n b n = (7 * n + 1 : ℚ) / (4 * n + 27)) →
  a.a 6 / b.a 6 = 78 / 71 := by
  sorry

end arithmetic_sequence_ratio_l394_39427


namespace octagon_side_length_l394_39421

theorem octagon_side_length (square_side : ℝ) (h : square_side = 1) :
  let octagon_side := square_side - 2 * ((square_side * (1 - 1 / Real.sqrt 2)) / 2)
  octagon_side = 1 - Real.sqrt 2 / 2 := by
  sorry

end octagon_side_length_l394_39421


namespace cubic_function_properties_l394_39476

/-- A cubic function with a maximum at x = -1 and a minimum at x = 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c x ≤ f a b c (-1)) ∧
  (f a b c (-1) = 7) ∧
  (f' a b (-1) = 0) ∧
  (f' a b 3 = 0) →
  a = -3 ∧ b = -9 ∧ c = 2 ∧ f a b c 3 = -25 := by
  sorry

#check cubic_function_properties

end cubic_function_properties_l394_39476


namespace theater_tickets_sold_l394_39488

/-- Theorem: Total number of tickets sold in a theater -/
theorem theater_tickets_sold 
  (orchestra_price : ℕ) 
  (balcony_price : ℕ) 
  (total_cost : ℕ) 
  (extra_balcony : ℕ) : 
  orchestra_price = 12 →
  balcony_price = 8 →
  total_cost = 3320 →
  extra_balcony = 190 →
  ∃ (orchestra : ℕ) (balcony : ℕ),
    orchestra_price * orchestra + balcony_price * balcony = total_cost ∧
    balcony = orchestra + extra_balcony ∧
    orchestra + balcony = 370 := by
  sorry

end theater_tickets_sold_l394_39488


namespace sum_of_x_and_y_on_circle_l394_39430

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end sum_of_x_and_y_on_circle_l394_39430


namespace intersection_of_A_and_B_l394_39441

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l394_39441


namespace puzzle_e_count_l394_39418

/-- Represents the types of puzzle pieces -/
inductive PieceType
| A  -- Corner piece
| B  -- Edge piece
| C  -- Special edge piece
| D  -- Internal piece with 3 indentations
| E  -- Internal piece with 2 indentations

/-- Structure representing a rectangular puzzle -/
structure Puzzle where
  width : ℕ
  height : ℕ
  total_pieces : ℕ
  a_count : ℕ
  b_count : ℕ
  c_count : ℕ
  d_count : ℕ
  balance_equation : 2 * a_count + b_count + c_count + 3 * d_count = 2 * b_count + 2 * c_count + d_count

/-- Theorem stating the number of E-type pieces in the puzzle -/
theorem puzzle_e_count (p : Puzzle) 
  (h_dim : p.width = 23 ∧ p.height = 37)
  (h_total : p.total_pieces = 851)
  (h_a : p.a_count = 4)
  (h_b : p.b_count = 108)
  (h_c : p.c_count = 4)
  (h_d : p.d_count = 52) :
  p.total_pieces - (p.a_count + p.b_count + p.c_count + p.d_count) = 683 := by
  sorry

end puzzle_e_count_l394_39418


namespace equal_sandwiched_segments_imply_parallel_or_intersecting_planes_l394_39468

-- Define a plane
structure Plane where
  -- Add necessary fields for a plane

-- Define a line segment
structure LineSegment where
  -- Add necessary fields for a line segment

-- Define the property of being sandwiched between two planes
def sandwichedBetween (l : LineSegment) (p1 p2 : Plane) : Prop :=
  sorry

-- Define the property of line segments being parallel
def areParallel (l1 l2 l3 : LineSegment) : Prop :=
  sorry

-- Define the property of line segments being equal
def areEqual (l1 l2 l3 : LineSegment) : Prop :=
  sorry

-- Define the property of planes being parallel
def arePlanesParallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define the property of planes intersecting
def arePlanesIntersecting (p1 p2 : Plane) : Prop :=
  sorry

-- The main theorem
theorem equal_sandwiched_segments_imply_parallel_or_intersecting_planes 
  (p1 p2 : Plane) (l1 l2 l3 : LineSegment) :
  sandwichedBetween l1 p1 p2 →
  sandwichedBetween l2 p1 p2 →
  sandwichedBetween l3 p1 p2 →
  areParallel l1 l2 l3 →
  areEqual l1 l2 l3 →
  arePlanesParallel p1 p2 ∨ arePlanesIntersecting p1 p2 :=
sorry

end equal_sandwiched_segments_imply_parallel_or_intersecting_planes_l394_39468


namespace a_minus_b_value_l394_39416

/-- Given an equation y = a + b/x, where a and b are constants, 
    prove that a - b = 19/2 when y = 2 for x = 2 and y = 7 for x = -2 -/
theorem a_minus_b_value (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 2 ↔ x = 2) ∧ (a + b / x = 7 ↔ x = -2)) → 
  a - b = 19 / 2 := by
sorry

end a_minus_b_value_l394_39416


namespace jack_and_jill_speed_l394_39486

/-- Jack's speed function in miles per hour -/
def jack_speed (x : ℝ) : ℝ := x^2 - 7*x - 18

/-- Jill's distance function in miles -/
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72

/-- Jill's time function in hours -/
def jill_time (x : ℝ) : ℝ := x + 8

/-- Theorem stating that under given conditions, Jack and Jill's speed is 2 miles per hour -/
theorem jack_and_jill_speed :
  ∃ x : ℝ, 
    jack_speed x = jill_distance x / jill_time x ∧ 
    jack_speed x = 2 :=
by sorry

end jack_and_jill_speed_l394_39486


namespace circle_symmetry_l394_39435

/-- Given a circle symmetrical to circle C with respect to the line x-y+1=0,
    prove that the equation of circle C is x^2 + (y-2)^2 = 1 -/
theorem circle_symmetry (x y : ℝ) :
  ((x - 1)^2 + (y - 1)^2 = 1) →  -- Equation of the symmetrical circle
  (x - y + 1 = 0 →               -- Equation of the line of symmetry
   (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = 1 ∧  -- Existence of circle C
    (a^2 + (b - 2)^2 = 1)))      -- Equation of circle C
:= by sorry

end circle_symmetry_l394_39435


namespace min_value_quadratic_fraction_l394_39475

theorem min_value_quadratic_fraction (x : ℝ) (h : x ≥ 0) :
  (9 * x^2 + 17 * x + 15) / (5 * (x + 2)) ≥ 18 * Real.sqrt 3 / 5 ∧
  ∃ y ≥ 0, (9 * y^2 + 17 * y + 15) / (5 * (y + 2)) = 18 * Real.sqrt 3 / 5 :=
by sorry

end min_value_quadratic_fraction_l394_39475


namespace cubic_factorization_l394_39422

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end cubic_factorization_l394_39422


namespace calculation_proof_l394_39402

theorem calculation_proof :
  ((-3/4 - 5/8 + 9/12) * (-24) = 15) ∧
  (-1^6 + |(-2)^3 - 10| - (-3) / (-1)^2023 = 14) := by
  sorry

end calculation_proof_l394_39402


namespace andys_school_distance_l394_39450

/-- The distance between Andy's house and school, given the total distance walked and the distance to the market. -/
theorem andys_school_distance (total_distance : ℕ) (market_distance : ℕ) (h1 : total_distance = 140) (h2 : market_distance = 40) : 
  let school_distance := (total_distance - market_distance) / 2
  school_distance = 50 := by sorry

end andys_school_distance_l394_39450


namespace circle_tangent_and_secant_l394_39472

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (4, -8)

-- Define the length of AB
def length_AB : ℝ := 4

theorem circle_tangent_and_secant :
  ∃ (tangent_length : ℝ) (line_DE : ℝ → ℝ → Prop) (line_AB : ℝ → ℝ → Prop),
    -- The length of the tangent from M to C is 3√5
    tangent_length = 3 * Real.sqrt 5 ∧
    -- The equation of line DE is 2x-7y-19=0
    (∀ x y, line_DE x y ↔ 2*x - 7*y - 19 = 0) ∧
    -- The equation of line AB is either 45x+28y+44=0 or x=4
    (∀ x y, line_AB x y ↔ (45*x + 28*y + 44 = 0 ∨ x = 4)) :=
by sorry

end circle_tangent_and_secant_l394_39472


namespace complex_magnitude_l394_39491

theorem complex_magnitude (a b : ℝ) :
  (Complex.I + a) * (1 - b * Complex.I) = 2 * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 2 := by
  sorry

end complex_magnitude_l394_39491


namespace school_gender_ratio_l394_39440

theorem school_gender_ratio (boys girls : ℕ) : 
  boys * 13 = girls * 5 →  -- ratio of boys to girls is 5:13
  girls = boys + 80 →      -- there are 80 more girls than boys
  boys = 50 :=             -- prove that the number of boys is 50
by sorry

end school_gender_ratio_l394_39440


namespace remainder_sum_of_powers_l394_39400

theorem remainder_sum_of_powers (n : ℕ) : (9^4 + 8^5 + 7^6) % 7 = 3 := by
  sorry

end remainder_sum_of_powers_l394_39400


namespace division_property_l394_39494

theorem division_property (a b : ℕ+) :
  (∃ (q r : ℕ), a.val^2 + b.val^2 = q * (a.val + b.val) + r ∧
                0 ≤ r ∧ r < a.val + b.val ∧
                q^2 + r = 1977) →
  ((a.val = 50 ∧ b.val = 37) ∨
   (a.val = 50 ∧ b.val = 7) ∨
   (a.val = 37 ∧ b.val = 50) ∨
   (a.val = 7 ∧ b.val = 50)) :=
by sorry

end division_property_l394_39494


namespace special_property_implies_units_nine_l394_39460

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ units ∧ units ≤ 9

/-- The property described in the problem -/
def has_special_property (n : TwoDigitNumber) : Prop :=
  n.tens + n.units + n.tens * n.units = 10 * n.tens + n.units

theorem special_property_implies_units_nine :
  ∀ n : TwoDigitNumber, has_special_property n → n.units = 9 := by
  sorry

end special_property_implies_units_nine_l394_39460


namespace quadratic_function_bounds_l394_39483

theorem quadratic_function_bounds (a c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - c
  (-4 : ℝ) ≤ f 1 ∧ f 1 ≤ -1 ∧ (-1 : ℝ) ≤ f 2 ∧ f 2 ≤ 5 →
  (-1 : ℝ) ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end quadratic_function_bounds_l394_39483


namespace only_B_suitable_l394_39492

-- Define the structure for a sampling experiment
structure SamplingExperiment where
  totalSize : ℕ
  sampleSize : ℕ
  isWellMixed : Bool

-- Define the conditions for lottery method suitability
def isLotteryMethodSuitable (experiment : SamplingExperiment) : Prop :=
  experiment.totalSize ≤ 100 ∧ 
  experiment.sampleSize ≤ 10 ∧ 
  experiment.isWellMixed

-- Define the given sampling experiments
def experimentA : SamplingExperiment := ⟨5000, 600, true⟩
def experimentB : SamplingExperiment := ⟨36, 6, true⟩
def experimentC : SamplingExperiment := ⟨36, 6, false⟩
def experimentD : SamplingExperiment := ⟨5000, 10, true⟩

-- Theorem statement
theorem only_B_suitable : 
  ¬(isLotteryMethodSuitable experimentA) ∧
  (isLotteryMethodSuitable experimentB) ∧
  ¬(isLotteryMethodSuitable experimentC) ∧
  ¬(isLotteryMethodSuitable experimentD) :=
by sorry

end only_B_suitable_l394_39492


namespace cube_sum_is_42_l394_39498

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  /-- The smallest number on the cube's faces -/
  smallest : ℕ
  /-- Proof that the smallest number is even -/
  smallest_even : Even smallest

/-- The sum of numbers on opposite faces of the cube -/
def opposite_face_sum (cube : NumberedCube) : ℕ :=
  2 * cube.smallest + 10

/-- The sum of all numbers on the cube's faces -/
def total_sum (cube : NumberedCube) : ℕ :=
  6 * cube.smallest + 30

/-- Theorem stating that the sum of numbers on a cube with the given properties is 42 -/
theorem cube_sum_is_42 (cube : NumberedCube) 
  (h : ∀ (i : Fin 3), opposite_face_sum cube = 2 * cube.smallest + 2 * i + 10) :
  total_sum cube = 42 := by
  sorry


end cube_sum_is_42_l394_39498


namespace quadratic_equation_properties_l394_39463

theorem quadratic_equation_properties (m : ℝ) :
  -- Part 1: The equation always has real roots
  ∃ x₁ x₂ : ℝ, x₁^2 - (m+3)*x₁ + 2*(m+1) = 0 ∧ x₂^2 - (m+3)*x₂ + 2*(m+1) = 0 ∧
  -- Part 2: If x₁² + x₂² = 5, then m = 0 or m = -2
  (∀ x₁ x₂ : ℝ, x₁^2 - (m+3)*x₁ + 2*(m+1) = 0 → x₂^2 - (m+3)*x₂ + 2*(m+1) = 0 → x₁^2 + x₂^2 = 5 → m = 0 ∨ m = -2) :=
by sorry

end quadratic_equation_properties_l394_39463


namespace bus_speed_proof_l394_39448

/-- The speed of Bus A in miles per hour -/
def speed_A : ℝ := 45

/-- The speed of Bus B in miles per hour -/
def speed_B : ℝ := speed_A - 15

/-- The initial distance between Bus A and Bus B in miles -/
def initial_distance : ℝ := 150

/-- The time it takes for Bus A to overtake Bus B when driving in the same direction, in hours -/
def overtake_time : ℝ := 10

/-- The time it would take for the buses to meet if driving towards each other, in hours -/
def meet_time : ℝ := 2

theorem bus_speed_proof :
  (speed_A - speed_B) * overtake_time = initial_distance ∧
  (speed_A + speed_B) * meet_time = initial_distance ∧
  speed_A = 45 := by
  sorry

end bus_speed_proof_l394_39448


namespace sandwich_combinations_l394_39403

theorem sandwich_combinations (salami_types : Nat) (cheese_types : Nat) (sauce_types : Nat) :
  salami_types = 8 →
  cheese_types = 7 →
  sauce_types = 3 →
  (salami_types * Nat.choose cheese_types 2 * sauce_types) = 504 := by
  sorry

end sandwich_combinations_l394_39403


namespace pure_imaginary_fraction_l394_39489

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - 2 * Complex.I) = Complex.I * b) → a = 2 := by
  sorry

end pure_imaginary_fraction_l394_39489


namespace fraction_equality_l394_39433

theorem fraction_equality : (18 : ℚ) / (5 * 107 + 3) = 18 / 538 := by sorry

end fraction_equality_l394_39433


namespace planted_fraction_specific_case_l394_39434

/-- Represents a right triangle field with an unplanted area -/
structure FieldWithUnplantedArea where
  /-- Length of the first leg of the right triangle field -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle field -/
  leg2 : ℝ
  /-- Shortest distance from the base of the unplanted triangle to the hypotenuse -/
  unplanted_distance : ℝ

/-- Calculates the fraction of the planted area in the field -/
def planted_fraction (field : FieldWithUnplantedArea) : ℝ :=
  -- Implementation details omitted
  sorry

theorem planted_fraction_specific_case :
  let field := FieldWithUnplantedArea.mk 5 12 3
  planted_fraction field = 2665 / 2890 := by
  sorry

end planted_fraction_specific_case_l394_39434


namespace max_area_of_triangle_l394_39412

noncomputable section

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sine law holds for the triangle. -/
axiom sine_law (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The given condition in the problem. -/
axiom problem_condition (t : Triangle) : Real.sin t.A / t.a = Real.sqrt 3 * Real.cos t.B / t.b

/-- The side b is equal to 2. -/
axiom side_b_is_2 (t : Triangle) : t.b = 2

/-- The area of a triangle. -/
def triangle_area (t : Triangle) : ℝ := 1/2 * t.a * t.c * Real.sin t.B

/-- The theorem to be proved. -/
theorem max_area_of_triangle (t : Triangle) : 
  (∀ t' : Triangle, triangle_area t' ≤ triangle_area t) → triangle_area t = Real.sqrt 3 := by
  sorry

end max_area_of_triangle_l394_39412


namespace adults_fed_is_eight_l394_39478

/-- Represents the number of adults that can be fed with one can of soup -/
def adults_per_can : ℕ := 4

/-- Represents the number of children that can be fed with one can of soup -/
def children_per_can : ℕ := 6

/-- Represents the total number of cans available -/
def total_cans : ℕ := 8

/-- Represents the number of children fed -/
def children_fed : ℕ := 24

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed : ℕ :=
  let cans_used_for_children := children_fed / children_per_can
  let remaining_cans := total_cans - cans_used_for_children
  let usable_cans := remaining_cans / 2
  usable_cans * adults_per_can

theorem adults_fed_is_eight : adults_fed = 8 := by
  sorry

end adults_fed_is_eight_l394_39478


namespace triangle_problem_l394_39496

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Condition: sin²A + sin A sin C + sin²C + cos²B = 1
  (Real.sin A)^2 + (Real.sin A) * (Real.sin C) + (Real.sin C)^2 + (Real.cos B)^2 = 1 →
  -- Condition: a = 5
  a = 5 →
  -- Condition: b = 7
  b = 7 →
  -- Prove: B = 2π/3
  B = 2 * Real.pi / 3 ∧
  -- Prove: sin C = 3√3/14
  Real.sin C = 3 * Real.sqrt 3 / 14 := by
  sorry


end triangle_problem_l394_39496


namespace complex_number_point_l394_39481

theorem complex_number_point (z : ℂ) : z = Complex.I * (2 + Complex.I) → z.re = -1 ∧ z.im = 2 := by
  sorry

end complex_number_point_l394_39481


namespace reflection_line_sum_l394_39459

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line
    is (4,9), then m + b = 20/3 -/
theorem reflection_line_sum (m b : ℚ) : 
  (∀ (x y : ℚ), y = m * x + b →
    (2 + (2 * m * (m * 2 + b - 3) / (1 + m^2)) = 4 ∧
     3 + (2 * (m * 2 + b - 3) / (1 + m^2)) = 9)) →
  m + b = 20/3 := by sorry

end reflection_line_sum_l394_39459


namespace weight_of_new_person_l394_39407

/-- Proves that if the average weight of 6 persons increases by 1.5 kg when a person
    weighing 65 kg is replaced by a new person, then the weight of the new person is 74 kg. -/
theorem weight_of_new_person
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (old_weight : ℝ)
  (new_weight : ℝ)
  (h1 : num_persons = 6)
  (h2 : avg_increase = 1.5)
  (h3 : old_weight = 65)
  (h4 : new_weight = num_persons * avg_increase + old_weight) :
  new_weight = 74 := by
  sorry

end weight_of_new_person_l394_39407


namespace jake_initial_balloons_l394_39414

/-- The number of balloons Jake initially brought to the park -/
def jake_initial : ℕ := 2

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of additional balloons Jake bought at the park -/
def jake_additional : ℕ := 3

theorem jake_initial_balloons :
  jake_initial = 2 :=
by
  have h1 : allan_balloons = jake_initial + jake_additional + 1 :=
    sorry
  sorry

end jake_initial_balloons_l394_39414


namespace geometric_sequence_a5_l394_39474

/-- A geometric sequence {a_n} where a_3 * a_7 = 64 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 := by
  sorry

end geometric_sequence_a5_l394_39474


namespace triangle_arithmetic_sequence_properties_l394_39495

/-- Given a triangle with sides a > b > c forming an arithmetic sequence with difference d,
    and inscribed circle radius r, prove the following properties. -/
theorem triangle_arithmetic_sequence_properties
  (a b c d r : ℝ)
  (α γ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a - b = d)
  (h4 : b - c = d)
  (h5 : r > 0)
  (h6 : α > 0)
  (h7 : γ > 0) :
  (Real.tan (α / 2) * Real.tan (γ / 2) = 1 / 3) ∧
  (r = 2 * d / (3 * (Real.tan (α / 2) - Real.tan (γ / 2)))) :=
by sorry

end triangle_arithmetic_sequence_properties_l394_39495


namespace complement_union_theorem_l394_39465

/-- The universal set I -/
def I : Set ℕ := {0, 1, 2, 3, 4}

/-- Set A -/
def A : Set ℕ := {0, 1, 2, 3}

/-- Set B -/
def B : Set ℕ := {2, 3, 4}

/-- The main theorem -/
theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by sorry

end complement_union_theorem_l394_39465


namespace problem_statement_l394_39451

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) :
  b < 0 ∧ |b| > |a| := by
  sorry

end problem_statement_l394_39451


namespace nonreal_cube_root_sum_l394_39413

/-- Given ω is a nonreal complex cube root of unity, 
    prove that (1 - ω + ω^2)^4 + (1 + ω - ω^2)^4 = -16 -/
theorem nonreal_cube_root_sum (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (1 - ω + ω^2)^4 + (1 + ω - ω^2)^4 = -16 := by
  sorry

end nonreal_cube_root_sum_l394_39413


namespace simple_interest_rate_calculation_l394_39417

theorem simple_interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (t : ℝ)  -- Time period in years
  (SI : ℝ) -- Simple interest
  (h1 : P = 2800)
  (h2 : t = 5)
  (h3 : SI = P - 2240)
  (h4 : SI = (P * r * t) / 100) -- r is the interest rate
  : r = 4 := by
sorry

end simple_interest_rate_calculation_l394_39417


namespace grade_ratio_is_two_to_one_l394_39464

/-- The ratio of students in the third grade to the second grade -/
def grade_ratio (boys_2nd : ℕ) (girls_2nd : ℕ) (total_students : ℕ) : ℚ :=
  let students_2nd := boys_2nd + girls_2nd
  let students_3rd := total_students - students_2nd
  students_3rd / students_2nd

/-- Theorem stating the ratio of students in the third grade to the second grade -/
theorem grade_ratio_is_two_to_one :
  grade_ratio 20 11 93 = 2 := by
  sorry

#eval grade_ratio 20 11 93

end grade_ratio_is_two_to_one_l394_39464


namespace not_both_odd_with_equal_product_l394_39404

/-- Represents a mapping of letters to digits -/
def DigitMapping := Char → Fin 10

/-- Represents a number as a string of letters -/
def NumberWord := String

/-- Calculate the product of digits in a number word given a digit mapping -/
def digitProduct (mapping : DigitMapping) (word : NumberWord) : ℕ :=
  word.foldl (λ acc c => acc * (mapping c).val.succ) 1

/-- Check if a number word represents an odd number given a digit mapping -/
def isOdd (mapping : DigitMapping) (word : NumberWord) : Prop :=
  (mapping word.back).val % 2 = 1

theorem not_both_odd_with_equal_product (mapping : DigitMapping) 
    (word1 word2 : NumberWord) 
    (h_distinct : ∀ (c1 c2 : Char), c1 ≠ c2 → mapping c1 ≠ mapping c2)
    (h_equal_product : digitProduct mapping word1 = digitProduct mapping word2) :
    ¬(isOdd mapping word1 ∧ isOdd mapping word2) := by
  sorry

end not_both_odd_with_equal_product_l394_39404


namespace sum_in_interval_l394_39461

theorem sum_in_interval : 
  let a : ℚ := 4 + 5/9
  let b : ℚ := 5 + 3/4
  let c : ℚ := 7 + 8/17
  17.5 < a + b + c ∧ a + b + c < 18 :=
by sorry

end sum_in_interval_l394_39461


namespace cube_monotonically_increasing_l394_39477

/-- A function f: ℝ → ℝ is monotonically increasing if for all x₁ < x₂, f(x₁) ≤ f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂

/-- The cube function -/
def cube (x : ℝ) : ℝ := x^3

/-- The cube function is monotonically increasing on ℝ -/
theorem cube_monotonically_increasing : MonotonicallyIncreasing cube := by
  sorry


end cube_monotonically_increasing_l394_39477


namespace lines_intersection_l394_39424

/-- Two lines intersect at a unique point (-2/7, 5/7) -/
theorem lines_intersection :
  ∃! (p : ℝ × ℝ), 
    (∃ s : ℝ, p = (2 + 3*s, 3 + 4*s)) ∧ 
    (∃ v : ℝ, p = (-1 + v, 2 - v)) ∧
    p = (-2/7, 5/7) := by
  sorry


end lines_intersection_l394_39424


namespace grid_rectangles_l394_39401

/-- The number of rectangles formed in a grid of parallel lines -/
def rectangles_in_grid (lines1 : ℕ) (lines2 : ℕ) : ℕ :=
  (lines1 - 1) * (lines2 - 1)

/-- Theorem: In a grid formed by 8 parallel lines intersected by 10 parallel lines, 
    the total number of rectangles formed is 63 -/
theorem grid_rectangles :
  rectangles_in_grid 8 10 = 63 := by
  sorry

end grid_rectangles_l394_39401


namespace swim_time_proof_l394_39456

/-- Proves that the time taken to swim downstream and upstream is 6 hours each -/
theorem swim_time_proof (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (still_water_speed : ℝ) (h1 : downstream_distance = 30) 
  (h2 : upstream_distance = 18) (h3 : still_water_speed = 4) :
  ∃ (t : ℝ) (current_speed : ℝ), 
    t = downstream_distance / (still_water_speed + current_speed) ∧
    t = upstream_distance / (still_water_speed - current_speed) ∧
    t = 6 := by
  sorry

#check swim_time_proof

end swim_time_proof_l394_39456


namespace boat_downstream_distance_l394_39411

theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 5) 
  (h3 : time = 4) : 
  boat_speed + stream_speed * time = 84 :=
by sorry

end boat_downstream_distance_l394_39411


namespace box_maker_is_bellini_l394_39497

-- Define the possible makers of the box
inductive Maker
  | Bellini
  | Cellini
  | BelliniSon
  | CelliniSon

-- Define the inscription on the box
def inscription (maker : Maker) : Prop :=
  maker ≠ Maker.BelliniSon

-- Define the condition that the box was made by Bellini, Cellini, or one of their sons
def possibleMakers (maker : Maker) : Prop :=
  maker = Maker.Bellini ∨ maker = Maker.Cellini ∨ maker = Maker.BelliniSon ∨ maker = Maker.CelliniSon

-- Theorem: The maker of the box is Bellini
theorem box_maker_is_bellini :
  ∃ (maker : Maker), possibleMakers maker ∧ inscription maker → maker = Maker.Bellini :=
sorry

end box_maker_is_bellini_l394_39497


namespace gcd_lcm_sum_45_4050_l394_39405

theorem gcd_lcm_sum_45_4050 : Nat.gcd 45 4050 + Nat.lcm 45 4050 = 4095 := by
  sorry

end gcd_lcm_sum_45_4050_l394_39405


namespace max_cubes_is_six_l394_39471

/-- Represents a stack of identical wooden cubes -/
structure CubeStack where
  front_view : Nat
  side_view : Nat
  top_view : Nat

/-- The maximum number of cubes in a stack given its views -/
def max_cubes (stack : CubeStack) : Nat :=
  2 * stack.top_view

/-- Theorem stating that the maximum number of cubes in the stack is 6 -/
theorem max_cubes_is_six (stack : CubeStack) 
  (h_top : stack.top_view = 3) : max_cubes stack = 6 := by
  sorry

end max_cubes_is_six_l394_39471


namespace line_slope_point_sum_l394_39432

/-- Given a line with slope 5 passing through (5, 3), prove m + b^2 = 489 --/
theorem line_slope_point_sum (m b : ℝ) : 
  m = 5 →                   -- The slope is 5
  3 = 5 * 5 + b →           -- The line passes through (5, 3)
  m + b^2 = 489 :=          -- Prove that m + b^2 = 489
by sorry

end line_slope_point_sum_l394_39432


namespace inequality_theorem_l394_39457

theorem inequality_theorem (a b : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → x^2 + 1 ≥ a*x + b ∧ a*x + b ≥ (3/2)*x^(2/3)) →
  ((2 - Real.sqrt 2) / 4 ≤ b ∧ b ≤ (2 + Real.sqrt 2) / 4) ∧
  (1 / Real.sqrt (2*b) ≤ a ∧ a ≤ 2 * Real.sqrt (1 - b)) :=
by sorry

end inequality_theorem_l394_39457


namespace physics_marks_calculation_l394_39438

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem physics_marks_calculation :
  let known_subjects_total : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let total_marks : ℕ := average_marks * total_subjects
  total_marks - known_subjects_total = 82 := by sorry

end physics_marks_calculation_l394_39438


namespace subtraction_problem_l394_39439

theorem subtraction_problem : 240 - (35 * 4 + 6 * 3) = 82 := by
  sorry

end subtraction_problem_l394_39439


namespace fruit_cost_calculation_l394_39490

/-- The cost of a single orange in dollars -/
def orange_cost : ℚ := (1.27 - 2 * 0.21) / 5

/-- The total cost of six apples and three oranges in dollars -/
def total_cost : ℚ := 6 * 0.21 + 3 * orange_cost

theorem fruit_cost_calculation :
  (2 * 0.21 + 5 * orange_cost = 1.27) →
  total_cost = 1.77 := by
  sorry

end fruit_cost_calculation_l394_39490


namespace equation_solution_l394_39470

theorem equation_solution : 
  ∃ (x : ℚ), (x + 2) / 4 - 1 = (2 * x + 1) / 3 ∧ x = -2 := by
  sorry

end equation_solution_l394_39470


namespace goldfish_price_theorem_l394_39499

/-- Represents the selling price of a goldfish -/
def selling_price : ℝ := sorry

/-- Represents the cost price of a goldfish -/
def cost_price : ℝ := 0.25

/-- Represents the price of the new tank -/
def tank_price : ℝ := 100

/-- Represents the number of goldfish sold -/
def goldfish_sold : ℕ := 110

/-- Represents the percentage short of the tank price -/
def percentage_short : ℝ := 0.45

theorem goldfish_price_theorem :
  selling_price = 0.75 :=
by
  sorry

end goldfish_price_theorem_l394_39499


namespace symmetric_circle_equation_l394_39473

/-- The equation of a circle symmetric to (x+2)^2 + y^2 = 5 with respect to y = x -/
theorem symmetric_circle_equation :
  let original_circle := (fun (x y : ℝ) => (x + 2)^2 + y^2 = 5)
  let symmetry_line := (fun (x y : ℝ) => y = x)
  let symmetric_circle := (fun (x y : ℝ) => x^2 + (y + 2)^2 = 5)
  ∀ x y : ℝ, symmetric_circle x y ↔ 
    ∃ x' y' : ℝ, original_circle x' y' ∧ 
    ((x + y = x' + y') ∧ (y - x = x' - y')) :=
by sorry


end symmetric_circle_equation_l394_39473
