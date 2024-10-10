import Mathlib

namespace min_value_x_plus_2y_l3545_354560

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y - x*y = 0 ∧ x + 2*y = 8 :=
by sorry

end min_value_x_plus_2y_l3545_354560


namespace dwarf_milk_problem_l3545_354529

/-- Represents the amount of milk in each cup after a dwarf pours -/
def milk_distribution (initial_amount : ℚ) (k : Fin 7) : ℚ :=
  initial_amount * k / 6

/-- The total amount of milk after all distributions -/
def total_milk (initial_amount : ℚ) : ℚ :=
  (Finset.sum Finset.univ (milk_distribution initial_amount)) + initial_amount

theorem dwarf_milk_problem (initial_amount : ℚ) :
  (∀ (k : Fin 7), milk_distribution initial_amount k ≤ initial_amount) →
  total_milk initial_amount = 3 →
  initial_amount = 3 / 7 := by
  sorry

end dwarf_milk_problem_l3545_354529


namespace smallest_integer_satisfying_inequality_l3545_354567

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 14*m + 40 ≤ 0 → n ≤ m) ∧ n^2 - 14*n + 40 ≤ 0 ∧ n = 4 := by
  sorry

end smallest_integer_satisfying_inequality_l3545_354567


namespace odd_binomials_count_l3545_354523

/-- The number of 1's in the binary representation of a natural number -/
def numOnes (n : ℕ) : ℕ := sorry

/-- The number of odd binomial coefficients in the n-th row of Pascal's triangle -/
def numOddBinomials (n : ℕ) : ℕ := sorry

/-- Theorem: The number of odd binomial coefficients in the n-th row of Pascal's triangle
    is equal to 2^k, where k is the number of 1's in the binary representation of n -/
theorem odd_binomials_count (n : ℕ) : numOddBinomials n = 2^(numOnes n) := by sorry

end odd_binomials_count_l3545_354523


namespace car_y_win_probability_l3545_354592

/-- The probability of car Y winning a race given specific conditions -/
theorem car_y_win_probability (total_cars : ℕ) (prob_x prob_z prob_xyz : ℝ) : 
  total_cars = 15 →
  prob_x = 1/4 →
  prob_z = 1/12 →
  prob_xyz = 0.4583333333333333 →
  ∃ (prob_y : ℝ), prob_y = 1/8 ∧ prob_x + prob_y + prob_z = prob_xyz :=
by sorry

end car_y_win_probability_l3545_354592


namespace equation_solution_l3545_354590

theorem equation_solution (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) ↔ x = 3 / 2 := by
sorry

end equation_solution_l3545_354590


namespace arithmetic_sequence_length_l3545_354517

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  /-- The sequence terms -/
  a : ℕ → ℝ
  /-- The number of terms -/
  n : ℕ
  /-- Sum of first 3 terms is 20 -/
  first_three_sum : a 1 + a 2 + a 3 = 20
  /-- Sum of last 3 terms is 130 -/
  last_three_sum : a (n - 2) + a (n - 1) + a n = 130
  /-- Sum of all terms is 200 -/
  total_sum : (Finset.range n).sum a = 200

/-- The number of terms in the arithmetic sequence is 8 -/
theorem arithmetic_sequence_length (seq : ArithmeticSequence) : seq.n = 8 := by
  sorry

end arithmetic_sequence_length_l3545_354517


namespace dave_book_spending_l3545_354565

/-- The total amount Dave spent on books -/
def total_spent (animal_books animal_price space_books space_price train_books train_price history_books history_price science_books science_price : ℕ) : ℕ :=
  animal_books * animal_price + space_books * space_price + train_books * train_price + history_books * history_price + science_books * science_price

/-- Theorem stating the total amount Dave spent on books -/
theorem dave_book_spending :
  total_spent 8 10 6 12 9 8 4 15 5 18 = 374 := by
  sorry

end dave_book_spending_l3545_354565


namespace quadratic_two_roots_condition_l3545_354578

theorem quadratic_two_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) ↔ k < 9 :=
by sorry

end quadratic_two_roots_condition_l3545_354578


namespace roberts_extra_chocolates_l3545_354516

theorem roberts_extra_chocolates (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 9) 
  (h2 : nickel_chocolates = 2) : 
  robert_chocolates - nickel_chocolates = 7 := by
sorry

end roberts_extra_chocolates_l3545_354516


namespace concrete_mixture_cement_percentage_l3545_354562

/-- Proves that given two types of concrete mixed in equal amounts to create a total mixture with a specific cement percentage, if one type has a known cement percentage, then the other type's cement percentage can be determined. -/
theorem concrete_mixture_cement_percentage 
  (total_weight : ℝ) 
  (final_cement_percentage : ℝ) 
  (weight_each_type : ℝ) 
  (cement_percentage_type1 : ℝ) :
  total_weight = 4500 →
  final_cement_percentage = 10.8 →
  weight_each_type = 1125 →
  cement_percentage_type1 = 10.8 →
  ∃ (cement_percentage_type2 : ℝ),
    cement_percentage_type2 = 32.4 ∧
    weight_each_type * cement_percentage_type1 / 100 + 
    weight_each_type * cement_percentage_type2 / 100 = 
    total_weight * final_cement_percentage / 100 :=
by sorry

end concrete_mixture_cement_percentage_l3545_354562


namespace cube_preserves_order_l3545_354570

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by sorry

end cube_preserves_order_l3545_354570


namespace regression_y_change_l3545_354504

/-- Represents a simple linear regression equation -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- Represents the change in y for a unit change in x -/
def yChange (reg : LinearRegression) : ℝ := -reg.slope

theorem regression_y_change (reg : LinearRegression) 
  (h : reg = { intercept := 3, slope := 5 }) : 
  yChange reg = -5 := by sorry

end regression_y_change_l3545_354504


namespace locus_equation_rectangle_perimeter_bound_l3545_354538

-- Define the locus W
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = Real.sqrt (p.1^2 + (p.2 - 1/2)^2)}

-- Define a rectangle with three vertices on W
structure RectangleOnW where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ
  h_a_on_w : a ∈ W
  h_b_on_w : b ∈ W
  h_c_on_w : c ∈ W
  h_is_rectangle : (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0 ∧
                   (a.1 - d.1) * (c.1 - d.1) + (a.2 - d.2) * (c.2 - d.2) = 0

-- Theorem statements
theorem locus_equation (p : ℝ × ℝ) :
  p ∈ W ↔ p.2 = p.1^2 + 1/4 := by sorry

theorem rectangle_perimeter_bound (rect : RectangleOnW) :
  let perimeter := 2 * (Real.sqrt ((rect.a.1 - rect.b.1)^2 + (rect.a.2 - rect.b.2)^2) +
                        Real.sqrt ((rect.b.1 - rect.c.1)^2 + (rect.b.2 - rect.c.2)^2))
  perimeter > 3 * Real.sqrt 3 := by sorry

end locus_equation_rectangle_perimeter_bound_l3545_354538


namespace otimes_neg_two_three_otimes_commutative_four_neg_two_l3545_354532

-- Define the ⊗ operation for rational numbers
def otimes (a b : ℚ) : ℚ := a * b - a - b - 2

-- Theorem 1: (-2) ⊗ 3 = -9
theorem otimes_neg_two_three : otimes (-2) 3 = -9 := by sorry

-- Theorem 2: 4 ⊗ (-2) = (-2) ⊗ 4
theorem otimes_commutative_four_neg_two : otimes 4 (-2) = otimes (-2) 4 := by sorry

end otimes_neg_two_three_otimes_commutative_four_neg_two_l3545_354532


namespace final_grasshoppers_count_l3545_354554

/-- Represents the state of the cage --/
structure CageState where
  crickets : ℕ
  grasshoppers : ℕ

/-- Represents a magician's trick --/
inductive Trick
  | Red
  | Green

/-- Applies a single trick to the cage state --/
def applyTrick (state : CageState) (trick : Trick) : CageState :=
  match trick with
  | Trick.Red => CageState.mk (state.crickets + 1) (state.grasshoppers - 4)
  | Trick.Green => CageState.mk (state.crickets - 5) (state.grasshoppers + 2)

/-- Applies a sequence of tricks to the cage state --/
def applyTricks (state : CageState) (tricks : List Trick) : CageState :=
  tricks.foldl applyTrick state

theorem final_grasshoppers_count (tricks : List Trick) :
  tricks.length = 18 →
  (applyTricks (CageState.mk 30 30) tricks).crickets = 0 →
  (applyTricks (CageState.mk 30 30) tricks).grasshoppers = 6 :=
by sorry

end final_grasshoppers_count_l3545_354554


namespace pilot_miles_theorem_l3545_354515

theorem pilot_miles_theorem (tuesday_miles : ℕ) (total_miles : ℕ) :
  tuesday_miles = 1134 →
  total_miles = 7827 →
  ∃ (thursday_miles : ℕ),
    3 * (tuesday_miles + thursday_miles) = total_miles ∧
    thursday_miles = 1475 :=
by
  sorry

end pilot_miles_theorem_l3545_354515


namespace complex_expression_simplification_l3545_354546

theorem complex_expression_simplification :
  let c : ℂ := 3 + 2*I
  let d : ℂ := -2 - I
  3*c + 4*d = 1 + 2*I :=
by sorry

end complex_expression_simplification_l3545_354546


namespace other_root_is_one_l3545_354507

-- Define the quadratic equation
def quadratic (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

-- Theorem statement
theorem other_root_is_one (b : ℝ) :
  (∃ x : ℝ, quadratic b x = 0 ∧ x = 3) →
  (∃ y : ℝ, y ≠ 3 ∧ quadratic b y = 0 ∧ y = 1) :=
by sorry

end other_root_is_one_l3545_354507


namespace circle_radius_is_two_l3545_354588

theorem circle_radius_is_two (r : ℝ) : r > 0 →
  3 * (2 * Real.pi * r) = 3 * (Real.pi * r^2) → r = 2 := by
  sorry

end circle_radius_is_two_l3545_354588


namespace substitution_remainder_l3545_354513

/-- Number of players in a soccer team --/
def total_players : ℕ := 22

/-- Number of starting players --/
def starting_players : ℕ := 11

/-- Number of substitute players --/
def substitute_players : ℕ := 11

/-- Maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Function to calculate the number of ways to make k substitutions --/
def substitution_ways (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => starting_players * substitute_players
  | k+1 => starting_players * (substitute_players - k) * substitution_ways k

/-- Total number of substitution scenarios --/
def total_scenarios : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

/-- Theorem stating the remainder when total scenarios is divided by 2000 --/
theorem substitution_remainder :
  total_scenarios % 2000 = 942 := by sorry

end substitution_remainder_l3545_354513


namespace fraction_simplification_l3545_354531

theorem fraction_simplification :
  (3 : ℝ) / (Real.sqrt 75 + Real.sqrt 48 + Real.sqrt 18) = Real.sqrt 3 / 12 := by
  sorry

end fraction_simplification_l3545_354531


namespace collinearity_iff_sum_one_l3545_354561

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define points
variable (O A B P : V)

-- Define real numbers m and n
variable (m n : ℝ)

-- Define the condition that O, A, B are not collinear
def not_collinear (O A B : V) : Prop := 
  ∀ (t : ℝ), (B - O) ≠ t • (A - O)

-- Define the vector equation
def vector_equation (O A B P : V) (m n : ℝ) : Prop :=
  (P - O) = m • (A - O) + n • (B - O)

-- Define collinearity of points A, P, B
def collinear (A P B : V) : Prop :=
  ∃ (t : ℝ), (P - A) = t • (B - A)

-- State the theorem
theorem collinearity_iff_sum_one
  (h₁ : not_collinear O A B)
  (h₂ : vector_equation O A B P m n) :
  collinear A P B ↔ m + n = 1 := by sorry

end collinearity_iff_sum_one_l3545_354561


namespace polynomial_factorization_l3545_354501

theorem polynomial_factorization (x y : ℝ) :
  -2 * x^2 * y + 8 * x * y - 6 * y = -2 * y * (x - 1) * (x - 3) :=
by sorry

end polynomial_factorization_l3545_354501


namespace number_equals_eight_l3545_354500

theorem number_equals_eight (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 := by
  sorry

end number_equals_eight_l3545_354500


namespace sum_of_A_and_B_sum_of_A_and_B_proof_l3545_354537

theorem sum_of_A_and_B : ℕ → ℕ → Prop :=
  fun A B =>
    (A < 10 ∧ B < 10) →  -- A and B are single digit numbers
    (A = 2 + 4) →        -- A is 4 greater than 2
    (B - 3 = 1) →        -- 3 less than B is 1
    A + B = 10           -- The sum of A and B is 10

-- Proof
theorem sum_of_A_and_B_proof : sum_of_A_and_B 6 4 := by
  sorry

end sum_of_A_and_B_sum_of_A_and_B_proof_l3545_354537


namespace complement_union_M_N_l3545_354545

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by sorry

end complement_union_M_N_l3545_354545


namespace equation_solution_set_l3545_354558

-- Define the equation
def equation (x : ℝ) : Prop := Real.log (Real.sqrt 3 * Real.sin x) = Real.log (-Real.cos x)

-- Define the solution set
def solution_set : Set ℝ := {x | ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6}

-- Theorem statement
theorem equation_solution_set : {x : ℝ | equation x} = solution_set := by sorry

end equation_solution_set_l3545_354558


namespace divisibility_condition_l3545_354506

theorem divisibility_condition (M : ℕ) : 
  M > 0 ∧ M < 10 →
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1 ∨ M = 4) :=
by sorry

end divisibility_condition_l3545_354506


namespace shortest_distance_line_to_circle_l3545_354596

/-- The shortest distance from a point on the line y=x-1 to the circle x^2+y^2+4x-2y+4=0 is 2√2 - 1 -/
theorem shortest_distance_line_to_circle : ∃ d : ℝ, d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (x y : ℝ),
    (y = x - 1) →
    (x^2 + y^2 + 4*x - 2*y + 4 = 0) →
    d ≤ Real.sqrt ((x - 0)^2 + (y - 0)^2) :=
by sorry

end shortest_distance_line_to_circle_l3545_354596


namespace modular_congruence_l3545_354553

theorem modular_congruence (n : ℕ) : 
  0 ≤ n ∧ n < 31 ∧ (3 * n) % 31 = 1 → 
  (((2^n) ^ 3) - 2) % 31 = 6 := by
  sorry

end modular_congruence_l3545_354553


namespace hours_worked_per_day_l3545_354594

theorem hours_worked_per_day 
  (total_hours : ℕ) 
  (weeks_worked : ℕ) 
  (h1 : total_hours = 140) 
  (h2 : weeks_worked = 4) :
  (total_hours : ℚ) / (weeks_worked * 7) = 5 := by
  sorry

end hours_worked_per_day_l3545_354594


namespace range_of_a_l3545_354535

/-- Proposition p: The solution set of the inequality x^2 + (a-1)x + a^2 < 0 regarding x is an empty set. -/
def proposition_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a-1)*x + a^2 ≥ 0

/-- Quadratic function f(x) = x^2 - mx + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- Proposition q: Given the quadratic function f(x) = x^2 - mx + 2 satisfies f(3/2 + x) = f(3/2 - x),
    and its maximum value is 2 when x ∈ [0,a]. -/
def proposition_q (a : ℝ) : Prop :=
  ∃ m, (∀ x, f m (3/2 + x) = f m (3/2 - x)) ∧
       (∀ x ∈ Set.Icc 0 a, f m x ≤ 2) ∧
       (∃ x ∈ Set.Icc 0 a, f m x = 2)

/-- The range of a given the logical conditions on p and q -/
theorem range_of_a :
  ∀ a : ℝ, (¬(proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a)) ↔
            a ∈ Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3 :=
sorry

end range_of_a_l3545_354535


namespace second_polygon_sides_l3545_354551

/-- A regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ := p.sides * p.sideLength

theorem second_polygon_sides 
  (p1 p2 : RegularPolygon) 
  (h1 : p1.sides = 42)
  (h2 : p1.sideLength = 3 * p2.sideLength)
  (h3 : perimeter p1 = perimeter p2) :
  p2.sides = 126 := by
sorry

end second_polygon_sides_l3545_354551


namespace distance_to_reflection_over_x_axis_l3545_354591

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis 
  (D : ℝ × ℝ) -- Point D in the plane
  (h : D = (3, -2)) -- D has coordinates (3, -2)
  : ‖D - (D.1, -D.2)‖ = 4 := by
  sorry


end distance_to_reflection_over_x_axis_l3545_354591


namespace perfect_square_trinomial_l3545_354586

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*(m-3)*x + 16 = (x - a)^2) → 
  (m = 7 ∨ m = -1) :=
by sorry

end perfect_square_trinomial_l3545_354586


namespace farmers_children_l3545_354564

/-- Represents the problem of determining the number of farmer's children --/
theorem farmers_children (apples_per_child : ℕ) (apples_eaten : ℕ) (apples_sold : ℕ) (apples_left : ℕ) : 
  apples_per_child = 15 → 
  apples_eaten = 8 → 
  apples_sold = 7 → 
  apples_left = 60 → 
  (apples_left + apples_eaten + apples_sold) / apples_per_child = 5 := by
  sorry

#check farmers_children

end farmers_children_l3545_354564


namespace path_count_theorem_l3545_354540

/-- The number of paths on a grid from point C to point D, where D is 6 units right and 2 units up from C, and the path consists of exactly 8 steps. -/
def number_of_paths : ℕ := 28

/-- The horizontal distance between points C and D on the grid. -/
def horizontal_distance : ℕ := 6

/-- The vertical distance between points C and D on the grid. -/
def vertical_distance : ℕ := 2

/-- The total number of steps in the path. -/
def total_steps : ℕ := 8

theorem path_count_theorem :
  number_of_paths = Nat.choose total_steps vertical_distance :=
by sorry

end path_count_theorem_l3545_354540


namespace jose_initial_caps_l3545_354508

/-- The number of bottle caps Jose started with -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Jose received from Rebecca -/
def received_caps : ℕ := 2

/-- The total number of bottle caps Jose ended up with -/
def total_caps : ℕ := 9

/-- Theorem stating that Jose started with 7 bottle caps -/
theorem jose_initial_caps : initial_caps = 7 := by
  sorry

end jose_initial_caps_l3545_354508


namespace apollo_total_cost_l3545_354539

/-- Represents the cost structure for a blacksmith --/
structure BlacksmithCost where
  monthly_rates : List ℕ
  installation_fee : ℕ
  installation_frequency : ℕ

/-- Calculates the total cost for a blacksmith for a year --/
def calculate_blacksmith_cost (cost : BlacksmithCost) : ℕ :=
  (cost.monthly_rates.sum) + 
  (12 / cost.installation_frequency * cost.installation_fee)

/-- Hephaestus's cost structure --/
def hephaestus_cost : BlacksmithCost := {
  monthly_rates := [3, 3, 3, 3, 6, 6, 6, 6, 9, 9, 9, 9],
  installation_fee := 2,
  installation_frequency := 1
}

/-- Athena's cost structure --/
def athena_cost : BlacksmithCost := {
  monthly_rates := [5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7],
  installation_fee := 10,
  installation_frequency := 12
}

/-- Ares's cost structure --/
def ares_cost : BlacksmithCost := {
  monthly_rates := [4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8],
  installation_fee := 3,
  installation_frequency := 3
}

/-- The total cost for Apollo's chariot wheels for a year --/
theorem apollo_total_cost : 
  calculate_blacksmith_cost hephaestus_cost + 
  calculate_blacksmith_cost athena_cost + 
  calculate_blacksmith_cost ares_cost = 265 := by
  sorry

end apollo_total_cost_l3545_354539


namespace james_pizza_fraction_l3545_354543

theorem james_pizza_fraction (num_pizzas : ℕ) (slices_per_pizza : ℕ) (james_slices : ℕ) :
  num_pizzas = 2 →
  slices_per_pizza = 6 →
  james_slices = 8 →
  (james_slices : ℚ) / (num_pizzas * slices_per_pizza : ℚ) = 2 / 3 := by
  sorry

end james_pizza_fraction_l3545_354543


namespace v_3003_equals_3_l3545_354569

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 1
| 4 => 2
| 5 => 4
| _ => 0  -- Default case for inputs not in the table

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| n + 1 => g (v n)

-- Theorem to prove
theorem v_3003_equals_3 : v 3003 = 3 := by
  sorry

end v_3003_equals_3_l3545_354569


namespace siwoo_cranes_per_hour_l3545_354536

/-- The number of cranes Siwoo folds in 30 minutes -/
def cranes_per_30_min : ℕ := 180

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the number of cranes Siwoo folds in 1 hour -/
def cranes_per_hour : ℕ := cranes_per_30_min * (minutes_per_hour / 30)

/-- Theorem stating that Siwoo folds 360 cranes in 1 hour -/
theorem siwoo_cranes_per_hour :
  cranes_per_hour = 360 := by
  sorry

end siwoo_cranes_per_hour_l3545_354536


namespace sumata_vacation_miles_l3545_354577

/-- The total miles driven on a vacation -/
def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

/-- Proof that the Sumata family drove 1250 miles on their vacation -/
theorem sumata_vacation_miles : 
  total_miles_driven 5.0 250 = 1250 := by
  sorry

end sumata_vacation_miles_l3545_354577


namespace energy_saving_product_analysis_l3545_354527

/-- Represents the sales volume in ten thousand items -/
def y (x : ℝ) : ℝ := -x + 120

/-- Represents the profit in ten thousand dollars -/
def W (x : ℝ) : ℝ := -(x - 100)^2 - 80

/-- Represents the profit in the second year considering donations -/
def W2 (x : ℝ) : ℝ := (x - 82) * (-x + 120)

theorem energy_saving_product_analysis :
  (∀ x, 90 ≤ x → x ≤ 110 → y x = -x + 120) ∧
  (∀ x, 90 ≤ x ∧ x ≤ 110 → W x ≤ 0) ∧
  (∃ x, 90 ≤ x ∧ x ≤ 110 ∧ W x = -80) ∧
  (∃ x, 92 ≤ x ∧ x ≤ 110 ∧ W2 x ≥ 280 ∧
    ∀ x', 92 ≤ x' ∧ x' ≤ 110 → W2 x' ≤ W2 x) :=
by sorry

end energy_saving_product_analysis_l3545_354527


namespace find_starting_number_l3545_354541

theorem find_starting_number :
  ∀ n : ℤ,
  (300 : ℝ) = (n + 200 : ℝ) / 2 + 150 →
  n = 100 :=
by sorry

end find_starting_number_l3545_354541


namespace equation_solutions_l3545_354568

open Real

-- Define the tangent function
noncomputable def tg (x : ℝ) : ℝ := tan x

-- Define the equation
def equation (x : ℝ) : Prop := tg x + tg (2*x) + tg (3*x) + tg (4*x) = 0

-- Define the set of solutions
def solution_set : Set ℝ := {0, π/7.2, π/5, π/3.186, π/2.5, -π/7.2, -π/5, -π/3.186, -π/2.5}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set :=
sorry

end equation_solutions_l3545_354568


namespace old_computer_wattage_is_1500_l3545_354549

/-- The wattage of John's old computer --/
def old_computer_wattage : ℝ := 1500

/-- The price increase of electricity --/
def electricity_price_increase : ℝ := 0.25

/-- The wattage increase of the new computer compared to the old one --/
def new_computer_wattage_increase : ℝ := 0.5

/-- The old price of electricity in dollars per kilowatt-hour --/
def old_electricity_price : ℝ := 0.12

/-- The cost to run the old computer for 50 hours in dollars --/
def old_computer_cost_50_hours : ℝ := 9

/-- The number of hours the old computer runs --/
def run_hours : ℝ := 50

/-- Theorem stating that the old computer's wattage is 1500 watts --/
theorem old_computer_wattage_is_1500 :
  old_computer_wattage = 
    (old_computer_cost_50_hours / run_hours) / old_electricity_price * 1000 :=
by sorry

end old_computer_wattage_is_1500_l3545_354549


namespace range_of_a_lower_bound_of_f_l3545_354503

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : f 1 a < 3 → -2/3 < a ∧ a < 4/3 := by sorry

-- Theorem for the lower bound of f(x)
theorem lower_bound_of_f (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := by sorry

end range_of_a_lower_bound_of_f_l3545_354503


namespace parabola_standard_equation_l3545_354519

/-- A parabola with directrix y = 1/2 has the standard equation x^2 = -2y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  let directrix : ℝ → ℝ := λ y => 1/2
  let parabola : ℝ → ℝ → Prop := λ x y => x^2 = -2*p*y
  (∀ x y, parabola x y ↔ y = -(x^2)/(2*p)) ∧ p = 1 → 
  ∀ x y, parabola x y ↔ x^2 = -2*y :=
by sorry

end parabola_standard_equation_l3545_354519


namespace angle_B_is_45_degrees_l3545_354566

theorem angle_B_is_45_degrees (A B : ℝ) 
  (h : 90 - (A + B) = 180 - (A - B)) : B = 45 := by
  sorry

end angle_B_is_45_degrees_l3545_354566


namespace parabola_decreasing_for_positive_x_l3545_354544

theorem parabola_decreasing_for_positive_x (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) :
  -x₂^2 + 3 < -x₁^2 + 3 :=
by sorry

end parabola_decreasing_for_positive_x_l3545_354544


namespace chess_group_age_sum_l3545_354585

/-- Given 4 children and 2 coaches with specific age relationships, 
    prove that if the sum of squares of their ages is 2796, 
    then the sum of their ages is 94. -/
theorem chess_group_age_sum 
  (a : ℕ) -- age of the youngest child
  (b : ℕ) -- age of the younger coach
  (h1 : a^2 + (a+2)^2 + (a+4)^2 + (a+6)^2 + b^2 + (b+2)^2 = 2796) :
  a + (a+2) + (a+4) + (a+6) + b + (b+2) = 94 := by
sorry

end chess_group_age_sum_l3545_354585


namespace hyperbola_circle_intersection_l3545_354587

/-- Given a hyperbola and a circle, prove that the x-coordinate of their intersection point in the first quadrant is (√3 + 1) / 2 -/
theorem hyperbola_circle_intersection (b c : ℝ) (P : ℝ × ℝ) : 
  let (x, y) := P
  (x^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (x^2 + y^2 = c^2) →      -- Circle equation
  (x > 0 ∧ y > 0) →        -- P is in the first quadrant
  ((x - c)^2 + y^2 = (c + 2)^2) →  -- |PF1| = c + 2
  (x = (Real.sqrt 3 + 1) / 2) :=
by sorry

end hyperbola_circle_intersection_l3545_354587


namespace largest_multiple_of_nine_less_than_hundred_l3545_354552

theorem largest_multiple_of_nine_less_than_hundred : 
  ∀ n : ℕ, n % 9 = 0 ∧ n < 100 → n ≤ 99 :=
by
  sorry

end largest_multiple_of_nine_less_than_hundred_l3545_354552


namespace tangent_slope_at_point_l3545_354599

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem tangent_slope_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -5/3
  let slope : ℝ := deriv f x₀
  (f x₀ = y₀) ∧ (slope = 1) := by sorry

end tangent_slope_at_point_l3545_354599


namespace doug_money_l3545_354580

/-- Represents the amount of money each person has -/
structure Money where
  josh : ℚ
  doug : ℚ
  brad : ℚ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.josh + m.doug + m.brad = 68 ∧
  m.josh = 2 * m.brad ∧
  m.josh = 3/4 * m.doug

/-- The theorem to prove -/
theorem doug_money (m : Money) (h : problem_conditions m) : m.doug = 32 := by
  sorry

end doug_money_l3545_354580


namespace medium_pizzas_ordered_l3545_354518

/-- Represents the number of slices in different pizza sizes --/
structure PizzaSlices where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of pizzas ordered --/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of slices for a given order --/
def totalSlices (slices : PizzaSlices) (order : PizzaOrder) : Nat :=
  slices.small * order.small + slices.medium * order.medium + slices.large * order.large

/-- The main theorem to prove --/
theorem medium_pizzas_ordered 
  (slices : PizzaSlices) 
  (order : PizzaOrder) 
  (h1 : slices.small = 6)
  (h2 : slices.medium = 8)
  (h3 : slices.large = 12)
  (h4 : order.small + order.medium + order.large = 15)
  (h5 : order.small = 4)
  (h6 : totalSlices slices order = 136) :
  order.medium = 5 := by
  sorry

end medium_pizzas_ordered_l3545_354518


namespace probability_theorem_l3545_354589

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_assignment (al bill cal : ℕ) : Prop :=
  1 ≤ al ∧ al ≤ 12 ∧
  1 ≤ bill ∧ bill ≤ 12 ∧
  1 ≤ cal ∧ cal ≤ 12 ∧
  al ≠ bill ∧ bill ≠ cal ∧ al ≠ cal

def satisfies_conditions (al bill cal : ℕ) : Prop :=
  is_multiple al bill ∧ is_multiple bill cal ∧ is_even (al + bill + cal)

def total_assignments : ℕ := 12 * 11 * 10

theorem probability_theorem :
  (∃ valid_count : ℕ,
    (∀ al bill cal : ℕ, valid_assignment al bill cal → satisfies_conditions al bill cal →
      valid_count > 0) ∧
    (valid_count : ℚ) / total_assignments = 2 / 110) :=
sorry

end probability_theorem_l3545_354589


namespace sin_cos_difference_l3545_354555

theorem sin_cos_difference (x y : Real) : 
  Real.sin (75 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end sin_cos_difference_l3545_354555


namespace joshua_and_justin_shared_money_l3545_354530

/-- Given that Joshua's share is $30 and it is thrice as much as Justin's share,
    prove that the total amount of money shared by Joshua and Justin is $40. -/
theorem joshua_and_justin_shared_money (joshua_share : ℕ) (justin_share : ℕ) : 
  joshua_share = 30 → joshua_share = 3 * justin_share → joshua_share + justin_share = 40 := by
  sorry

end joshua_and_justin_shared_money_l3545_354530


namespace anns_shopping_trip_l3545_354598

/-- Calculates the cost of each top in Ann's shopping trip -/
def cost_per_top (total_spent : ℚ) (num_shorts : ℕ) (price_shorts : ℚ) 
  (num_shoes : ℕ) (price_shoes : ℚ) (num_tops : ℕ) : ℚ :=
  let total_shorts := num_shorts * price_shorts
  let total_shoes := num_shoes * price_shoes
  let total_tops := total_spent - total_shorts - total_shoes
  total_tops / num_tops

/-- Proves that the cost per top is $5 given the conditions of Ann's shopping trip -/
theorem anns_shopping_trip : 
  cost_per_top 75 5 7 2 10 4 = 5 := by
  sorry

end anns_shopping_trip_l3545_354598


namespace segments_can_be_commensurable_l3545_354510

/-- Represents a geometric segment -/
structure Segment where
  length : ℝ
  pos : length > 0

/-- Two segments are commensurable if their ratio is rational -/
def commensurable (a b : Segment) : Prop :=
  ∃ (q : ℚ), a.length = q * b.length

/-- Segment m fits into a an integer number of times -/
def fits_integer_times (m a : Segment) : Prop :=
  ∃ (k : ℤ), a.length = k * m.length

/-- No segment m/(10^n) fits into b an integer number of times -/
def no_submultiple_fits (m b : Segment) : Prop :=
  ∀ (n : ℕ), ¬∃ (j : ℤ), b.length = j * (m.length / (10^n : ℝ))

theorem segments_can_be_commensurable
  (a b m : Segment)
  (h1 : fits_integer_times m a)
  (h2 : no_submultiple_fits m b) :
  commensurable a b :=
sorry

end segments_can_be_commensurable_l3545_354510


namespace symmetry_implies_periodicity_l3545_354582

def is_symmetrical_about (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x = 2 * b - f (2 * a - x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity 
  (f : ℝ → ℝ) (a b c d : ℝ) (h1 : a ≠ c) 
  (h2 : is_symmetrical_about f a b) 
  (h3 : is_symmetrical_about f c d) : 
  is_periodic f (2 * |a - c|) :=
sorry

end symmetry_implies_periodicity_l3545_354582


namespace gcd_of_powers_of_two_l3545_354542

theorem gcd_of_powers_of_two : Nat.gcd (2^2025 - 1) (2^2016 - 1) = 2^9 - 1 := by
  sorry

end gcd_of_powers_of_two_l3545_354542


namespace circle_area_equality_l3545_354576

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 24) (h₂ : r₂ = 34) :
  ∃ r : ℝ, π * r^2 = π * (r₂^2 - r₁^2) ∧ r = 2 * Real.sqrt 145 := by
  sorry

end circle_area_equality_l3545_354576


namespace matematika_arrangements_l3545_354573

/-- The number of distinct letters in "MATEMATIKA" excluding "A" -/
def n : ℕ := 7

/-- The number of repeated letters (M and T) -/
def r : ℕ := 2

/-- The number of "A"s in "MATEMATIKA" -/
def a : ℕ := 3

/-- The number of positions to place "A"s -/
def p : ℕ := n + 1

theorem matematika_arrangements : 
  (n.factorial / (r.factorial * r.factorial)) * Nat.choose p a = 70560 := by
  sorry

end matematika_arrangements_l3545_354573


namespace tangent_line_parallel_and_inequality_l3545_354559

noncomputable def f (x : ℝ) := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) := f x + a / x - 1

theorem tangent_line_parallel_and_inequality (a : ℝ) :
  (∃ (m : ℝ), m = (1 / 2 : ℝ) - a / 4 ∧ m = -(1 / 2 : ℝ)) ∧
  (∀ (m n : ℝ), m > n → n > 0 → (m - n) / (m + n) < (Real.log m - Real.log n) / 2) :=
sorry

end tangent_line_parallel_and_inequality_l3545_354559


namespace inequality_system_solution_l3545_354521

theorem inequality_system_solution (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1) ∧ (1/2) * x - 1 ≤ 7 - (3/2) * x) ↔ (2 < x ∧ x ≤ 4) := by
  sorry

end inequality_system_solution_l3545_354521


namespace ages_ratio_years_ago_sum_of_ages_correct_years_ago_l3545_354511

/-- The number of years ago when the ages of A, B, and C were in the ratio 1 : 2 : 3 -/
def years_ago : ℕ := 3

/-- The present age of A -/
def A_age : ℕ := 11

/-- The present age of B -/
def B_age : ℕ := 22

/-- The present age of C -/
def C_age : ℕ := 24

/-- The theorem stating that the ages were in ratio 1:2:3 some years ago -/
theorem ages_ratio_years_ago : 
  (A_age - years_ago) * 2 = B_age - years_ago ∧
  (A_age - years_ago) * 3 = C_age - years_ago :=
sorry

/-- The theorem stating that the sum of present ages is 57 -/
theorem sum_of_ages : A_age + B_age + C_age = 57 :=
sorry

/-- The main theorem proving that 'years_ago' is correct -/
theorem correct_years_ago : 
  ∃ (y : ℕ), y = years_ago ∧
  (A_age - y) * 2 = B_age - y ∧
  (A_age - y) * 3 = C_age - y ∧
  A_age + B_age + C_age = 57 ∧
  A_age = 11 :=
sorry

end ages_ratio_years_ago_sum_of_ages_correct_years_ago_l3545_354511


namespace product_of_sum_and_sum_of_cubes_l3545_354583

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 3) 
  (h2 : a^3 + b^3 = 81) : 
  a * b = -6 := by
sorry

end product_of_sum_and_sum_of_cubes_l3545_354583


namespace root_of_two_quadratics_l3545_354572

theorem root_of_two_quadratics (a b c d : ℂ) (k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^2 + b * k + c = 0)
  (hk2 : b * k^2 + c * k + d = 0) :
  k = 1 ∨ k = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ k = (-1 - Complex.I * Real.sqrt 3) / 2 :=
sorry

end root_of_two_quadratics_l3545_354572


namespace cube_paint_theorem_l3545_354550

/-- Represents a cube with edge length n -/
structure Cube (n : ℕ) where
  edge_length : n > 0

/-- Count of unit cubes with one red face -/
def one_face_count (n : ℕ) : ℕ := 6 * (n - 2)^2

/-- Count of unit cubes with two red faces -/
def two_face_count (n : ℕ) : ℕ := 12 * (n - 2)

/-- The main theorem stating the condition for n = 26 -/
theorem cube_paint_theorem (n : ℕ) (c : Cube n) :
  one_face_count n = 12 * two_face_count n ↔ n = 26 := by
  sorry

#check cube_paint_theorem

end cube_paint_theorem_l3545_354550


namespace sin_cubed_identity_l3545_354520

theorem sin_cubed_identity (θ : Real) : 
  Real.sin θ ^ 3 = -1/4 * Real.sin (3 * θ) + 3/4 * Real.sin θ := by
  sorry

end sin_cubed_identity_l3545_354520


namespace square_area_is_400_l3545_354579

/-- A square cut into five rectangles of equal area -/
structure CutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of one of the rectangles -/
  rect_width : ℝ
  /-- The number of rectangles the square is cut into -/
  num_rectangles : ℕ
  /-- The rectangles have equal area -/
  equal_area : ℝ
  /-- The given width of one rectangle -/
  given_width : ℝ
  /-- Condition: The number of rectangles is 5 -/
  h1 : num_rectangles = 5
  /-- Condition: The given width is 5 -/
  h2 : given_width = 5
  /-- Condition: The area of each rectangle is the total area divided by the number of rectangles -/
  h3 : equal_area = side^2 / num_rectangles
  /-- Condition: One of the rectangles has the given width -/
  h4 : rect_width = given_width

/-- The area of the square is 400 -/
theorem square_area_is_400 (s : CutSquare) : s.side^2 = 400 := by
  sorry

end square_area_is_400_l3545_354579


namespace seryozha_healthy_eating_days_l3545_354525

/-- Represents the daily cookie consumption pattern -/
structure DailyCookies where
  chocolate : ℕ
  sugarFree : ℕ

/-- Represents the total cookie consumption over a period -/
structure TotalCookies where
  chocolate : ℕ
  sugarFree : ℕ

/-- Calculates the total cookies consumed over a period given the initial and final daily consumption -/
def calculateTotalCookies (initial final : DailyCookies) (days : ℕ) : TotalCookies :=
  { chocolate := (initial.chocolate + final.chocolate) * days / 2,
    sugarFree := (initial.sugarFree + final.sugarFree) * days / 2 }

/-- Theorem stating the number of days in Seryozha's healthy eating regimen -/
theorem seryozha_healthy_eating_days : 
  ∃ (initial : DailyCookies) (days : ℕ),
    let final : DailyCookies := ⟨initial.chocolate - (days - 1), initial.sugarFree + (days - 1)⟩
    let total : TotalCookies := calculateTotalCookies initial final days
    total.chocolate = 264 ∧ total.sugarFree = 187 ∧ days = 11 := by
  sorry


end seryozha_healthy_eating_days_l3545_354525


namespace min_perimeter_rectangle_l3545_354556

/-- Given a positive real number S representing the area of a rectangle,
    prove that the square with side length √S has the smallest perimeter
    among all rectangles with area S, and this minimum perimeter is 4√S. -/
theorem min_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x * y = S ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → a * b = S → 2*(x + y) ≤ 2*(a + b)) ∧
    x = Real.sqrt S ∧ y = Real.sqrt S ∧
    2*(x + y) = 4 * Real.sqrt S :=
sorry

end min_perimeter_rectangle_l3545_354556


namespace tournament_properties_l3545_354526

structure Tournament :=
  (teams : ℕ)
  (scores : List ℕ)
  (win_points : ℕ)
  (draw_points : ℕ)
  (loss_points : ℕ)

def round_robin (t : Tournament) : Prop :=
  t.teams = 10 ∧ t.scores.length = 10 ∧ t.win_points = 3 ∧ t.draw_points = 1 ∧ t.loss_points = 0

theorem tournament_properties (t : Tournament) (h : round_robin t) :
  (∃ k : ℕ, (t.scores.filter (λ x => x % 2 = 1)).length = 2 * k) ∧
  (∃ k : ℕ, (t.scores.filter (λ x => x % 2 = 0)).length = 2 * k) ∧
  ¬(∃ a b c : ℕ, a < b ∧ b < c ∧ c < t.scores.length ∧ t.scores[a]! = 0 ∧ t.scores[b]! = 0 ∧ t.scores[c]! = 0) ∧
  (∃ scores : List ℕ, scores.length = 10 ∧ scores.sum < 135 ∧ round_robin ⟨10, scores, 3, 1, 0⟩) ∧
  (∃ m : ℕ, m ≥ 15 ∧ m ∈ t.scores) :=
by sorry

end tournament_properties_l3545_354526


namespace nell_gave_136_cards_to_jeff_l3545_354533

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff (original_cards : ℕ) (cards_left : ℕ) : ℕ :=
  original_cards - cards_left

/-- Proof that Nell gave 136 cards to Jeff -/
theorem nell_gave_136_cards_to_jeff :
  cards_given_to_jeff 242 106 = 136 := by
  sorry

end nell_gave_136_cards_to_jeff_l3545_354533


namespace graph_of_2x_plus_5_is_straight_line_l3545_354524

-- Define what it means for a function to be linear
def is_linear_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define what it means for a graph to be a straight line
def is_straight_line (f : ℝ → ℝ) : Prop := 
  ∃ m b : ℝ, ∀ x, f x = m * x + b

-- Define our specific function
def f : ℝ → ℝ := λ x => 2 * x + 5

-- State the theorem
theorem graph_of_2x_plus_5_is_straight_line :
  (∀ g : ℝ → ℝ, is_linear_function g → is_straight_line g) →
  is_linear_function f →
  is_straight_line f := by
  sorry

end graph_of_2x_plus_5_is_straight_line_l3545_354524


namespace milk_butterfat_calculation_l3545_354581

/-- Represents the butterfat percentage as a real number between 0 and 100 -/
def ButterfatPercentage := { x : ℝ // 0 ≤ x ∧ x ≤ 100 }

/-- Calculates the initial butterfat percentage of milk given the conditions -/
def initial_butterfat_percentage (
  initial_volume : ℝ) 
  (cream_volume : ℝ) 
  (cream_butterfat : ButterfatPercentage) 
  (final_butterfat : ButterfatPercentage) : ButterfatPercentage :=
  sorry

theorem milk_butterfat_calculation :
  let initial_volume : ℝ := 1000
  let cream_volume : ℝ := 50
  let cream_butterfat : ButterfatPercentage := ⟨23, by norm_num⟩
  let final_butterfat : ButterfatPercentage := ⟨3, by norm_num⟩
  let result := initial_butterfat_percentage initial_volume cream_volume cream_butterfat final_butterfat
  result.val = 4 := by sorry

end milk_butterfat_calculation_l3545_354581


namespace correct_writers_l3545_354505

/-- Represents the group of students and their writing task -/
structure StudentGroup where
  total : Nat
  cat_writers : Nat
  rat_writers : Nat
  crocodile_writers : Nat
  correct_cat : Nat
  correct_rat : Nat

/-- Theorem stating the number of students who wrote their word correctly -/
theorem correct_writers (group : StudentGroup) 
  (h1 : group.total = 50)
  (h2 : group.cat_writers = 10)
  (h3 : group.rat_writers = 18)
  (h4 : group.crocodile_writers = group.total - group.cat_writers - group.rat_writers)
  (h5 : group.correct_cat = 15)
  (h6 : group.correct_rat = 15) :
  group.correct_cat + group.correct_rat - (group.cat_writers + group.rat_writers) + group.crocodile_writers = 8 := by
  sorry

end correct_writers_l3545_354505


namespace ellipse_other_x_intercept_l3545_354547

/-- Definition of an ellipse with given foci and one x-intercept -/
def Ellipse (f1 f2 x1 : ℝ × ℝ) : Prop :=
  let d1 (x y : ℝ) := Real.sqrt ((x - f1.1)^2 + (y - f1.2)^2)
  let d2 (x y : ℝ) := Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2)
  ∀ x y : ℝ, d1 x y + d2 x y = d1 x1.1 x1.2 + d2 x1.1 x1.2

/-- The main theorem -/
theorem ellipse_other_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  let x1 : ℝ × ℝ := (1, 0)
  let x2 : ℝ × ℝ := ((13 - 14 * Real.sqrt 10) / (2 * Real.sqrt 10 + 14), 0)
  Ellipse f1 f2 x1 → x2.1 ≠ x1.1 → Ellipse f1 f2 x2 := by
  sorry

end ellipse_other_x_intercept_l3545_354547


namespace correlation_theorem_l3545_354522

-- Define the relation between x and y
def relation (x y : ℝ) : Prop := y = -0.1 * x + 1

-- Define positive correlation
def positively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → a x < a y ∧ b x < b y

-- Define negative correlation
def negatively_correlated (a b : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → a x > a y ∧ b x < b y

-- The main theorem
theorem correlation_theorem (x y z : ℝ → ℝ) 
  (h1 : ∀ t, relation (x t) (y t))
  (h2 : positively_correlated y z) :
  negatively_correlated x y ∧ negatively_correlated x z := by
  sorry

end correlation_theorem_l3545_354522


namespace average_first_16_even_numbers_l3545_354574

theorem average_first_16_even_numbers : 
  let first_16_even : List ℕ := List.range 16 |>.map (fun n => 2 * (n + 1))
  (first_16_even.sum / first_16_even.length : ℚ) = 17 := by
sorry

end average_first_16_even_numbers_l3545_354574


namespace subtraction_result_l3545_354563

theorem subtraction_result : 3.05 - 5.678 = -2.628 := by sorry

end subtraction_result_l3545_354563


namespace xyz_product_one_l3545_354595

theorem xyz_product_one (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5) (eq2 : y + 1/z = 2) (eq3 : z + 1/x = 3) :
  x * y * z = 1 := by
sorry

end xyz_product_one_l3545_354595


namespace sqrt_and_pi_comparisons_l3545_354593

theorem sqrt_and_pi_comparisons : 
  (Real.sqrt 2 < Real.sqrt 3) ∧ (3.14 < Real.pi) := by sorry

end sqrt_and_pi_comparisons_l3545_354593


namespace a_in_range_l3545_354502

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem a_in_range (a : ℝ) (h : a ∈ A) : -1 ≤ a ∧ a ≤ 3 := by
  sorry

end a_in_range_l3545_354502


namespace sequence_seventh_term_l3545_354512

theorem sequence_seventh_term : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 ∧ 
    (∀ n : ℕ, a (n + 1) = 2 * a n + 2) → 
    a 7 = 190 := by
  sorry

end sequence_seventh_term_l3545_354512


namespace problems_left_to_grade_l3545_354509

/-- Calculates the number of problems left to grade for a teacher grading worksheets from three subjects. -/
theorem problems_left_to_grade
  (math_problems_per_sheet : ℕ)
  (science_problems_per_sheet : ℕ)
  (english_problems_per_sheet : ℕ)
  (total_math_sheets : ℕ)
  (total_science_sheets : ℕ)
  (total_english_sheets : ℕ)
  (graded_math_sheets : ℕ)
  (graded_science_sheets : ℕ)
  (graded_english_sheets : ℕ)
  (h_math : math_problems_per_sheet = 5)
  (h_science : science_problems_per_sheet = 3)
  (h_english : english_problems_per_sheet = 7)
  (h_total_math : total_math_sheets = 10)
  (h_total_science : total_science_sheets = 15)
  (h_total_english : total_english_sheets = 12)
  (h_graded_math : graded_math_sheets = 6)
  (h_graded_science : graded_science_sheets = 10)
  (h_graded_english : graded_english_sheets = 5) :
  (total_math_sheets * math_problems_per_sheet - graded_math_sheets * math_problems_per_sheet) +
  (total_science_sheets * science_problems_per_sheet - graded_science_sheets * science_problems_per_sheet) +
  (total_english_sheets * english_problems_per_sheet - graded_english_sheets * english_problems_per_sheet) = 84 :=
by sorry

end problems_left_to_grade_l3545_354509


namespace coin_value_equality_l3545_354557

/-- The value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The theorem stating the equality of coin values -/
theorem coin_value_equality (n : ℕ) : 
  15 * coin_value "quarter" + 20 * coin_value "dime" = 
  10 * coin_value "quarter" + n * coin_value "dime" + 5 * coin_value "nickel" → 
  n = 30 := by
  sorry

#check coin_value_equality

end coin_value_equality_l3545_354557


namespace circle_equation_from_diameter_l3545_354534

/-- The equation of a circle given the endpoints of its diameter -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  A = (-3, -1) →
  B = (5, 5) →
  ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 25 ↔ 
    ∃ t : ℝ, (x, y) = ((1 - t) • A.1 + t • B.1, (1 - t) • A.2 + t • B.2) ∧ 0 ≤ t ∧ t ≤ 1 :=
by sorry

end circle_equation_from_diameter_l3545_354534


namespace two_layer_coverage_is_zero_l3545_354514

/-- Represents the area covered by rugs with different layers of overlap -/
structure RugCoverage where
  total_rug_area : ℝ
  total_floor_coverage : ℝ
  multilayer_coverage : ℝ
  three_layer_coverage : ℝ

/-- Calculates the area covered by exactly two layers of rug -/
def two_layer_coverage (rc : RugCoverage) : ℝ :=
  rc.multilayer_coverage - rc.three_layer_coverage

/-- Theorem stating that under the given conditions, the area covered by exactly two layers of rug is 0 -/
theorem two_layer_coverage_is_zero (rc : RugCoverage)
  (h1 : rc.total_rug_area = 212)
  (h2 : rc.total_floor_coverage = 140)
  (h3 : rc.multilayer_coverage = 24)
  (h4 : rc.three_layer_coverage = 24) :
  two_layer_coverage rc = 0 := by
  sorry

end two_layer_coverage_is_zero_l3545_354514


namespace direction_vector_value_l3545_354548

/-- A line with direction vector (a, 4) passing through points (-2, 3) and (3, 5) has a = 10 -/
theorem direction_vector_value (a : ℝ) : 
  let v : ℝ × ℝ := (a, 4)
  let p₁ : ℝ × ℝ := (-2, 3)
  let p₂ : ℝ × ℝ := (3, 5)
  (∃ (t : ℝ), p₂ = p₁ + t • v) → a = 10 := by
sorry

end direction_vector_value_l3545_354548


namespace proposition_b_l3545_354571

theorem proposition_b (a : ℝ) : 0 < a → a < 1 → a^3 < a := by
  sorry

end proposition_b_l3545_354571


namespace log_equation_solution_l3545_354584

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ log x 81 = 4/2 → x = 9 :=
by
  sorry

end log_equation_solution_l3545_354584


namespace percentage_calculation_l3545_354575

theorem percentage_calculation (x y : ℝ) (P : ℝ) 
  (h1 : x / y = 4)
  (h2 : 0.8 * x = P / 100 * y) :
  P = 320 := by
sorry

end percentage_calculation_l3545_354575


namespace smallest_integers_difference_smallest_integers_difference_exists_l3545_354597

theorem smallest_integers_difference : ℕ → Prop := fun n =>
  (∃ m : ℕ, m > 1 ∧ 
    (∀ k : ℕ, 2 ≤ k → k ≤ 13 → m % k = 1) ∧
    (∀ j : ℕ, j > 1 → 
      (∀ k : ℕ, 2 ≤ k → k ≤ 13 → j % k = 1) → 
      j ≥ m) ∧
    (∃ p : ℕ, p > m ∧ 
      (∀ k : ℕ, 2 ≤ k → k ≤ 13 → p % k = 1) ∧
      (∀ q : ℕ, q > m → 
        (∀ k : ℕ, 2 ≤ k → k ≤ 13 → q % k = 1) → 
        q ≥ p) ∧
      p - m = n)) →
  n = 360360

theorem smallest_integers_difference_exists : 
  ∃ n : ℕ, smallest_integers_difference n := by sorry

end smallest_integers_difference_smallest_integers_difference_exists_l3545_354597


namespace complex_difference_on_unit_circle_l3545_354528

theorem complex_difference_on_unit_circle (z₁ z₂ : ℂ) : 
  (∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I) ≤ Complex.abs (z₁ + 1 + Complex.I)) →
  (∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I) ≥ Complex.abs (z₂ + 1 + Complex.I)) →
  Complex.abs z₁ = 1 →
  Complex.abs z₂ = 1 →
  z₁ - z₂ = Complex.mk (Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end complex_difference_on_unit_circle_l3545_354528
