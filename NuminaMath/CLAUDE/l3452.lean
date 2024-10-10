import Mathlib

namespace sequence_difference_l3452_345274

def sequence_property (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n ≠ 0) ∧ 
  (∀ n : ℕ+, a n * a (n + 1) = S n)

theorem sequence_difference (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : sequence_property a S) : a 3 - a 1 = 1 := by
  sorry

end sequence_difference_l3452_345274


namespace function_inequality_implies_constant_l3452_345216

/-- A function f: ℝ → ℝ satisfying f(x+y) ≤ f(x^2+y) for all x, y ∈ ℝ is constant. -/
theorem function_inequality_implies_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end function_inequality_implies_constant_l3452_345216


namespace molecular_weight_AlI3_correct_l3452_345249

/-- The molecular weight of AlI3 in grams per mole -/
def molecular_weight_AlI3 : ℝ := 408

/-- The number of moles given in the problem -/
def num_moles : ℝ := 8

/-- The total weight of the given number of moles in grams -/
def total_weight : ℝ := 3264

/-- Theorem stating that the molecular weight of AlI3 is correct -/
theorem molecular_weight_AlI3_correct : 
  molecular_weight_AlI3 = total_weight / num_moles :=
sorry

end molecular_weight_AlI3_correct_l3452_345249


namespace expression_result_l3452_345254

theorem expression_result : 
  (0.66 : ℝ)^3 - (0.1 : ℝ)^3 / (0.66 : ℝ)^2 + 0.066 + (0.1 : ℝ)^2 = 0.3612 := by
  sorry

end expression_result_l3452_345254


namespace imaginary_part_of_z_l3452_345296

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 - Complex.I) = 3 + Complex.I) : 
  z.im = -2 := by sorry

end imaginary_part_of_z_l3452_345296


namespace triangle_properties_l3452_345286

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 6)
  (h2 : Real.sin t.A - Real.sin t.C = Real.sin (t.A - t.B))
  (h3 : t.b = 2 * Real.sqrt 7) :
  t.B = π / 3 ∧ 
  (t.a * t.c * Real.sin t.B / 2 = 3 * Real.sqrt 3 ∨ 
   t.a * t.c * Real.sin t.B / 2 = 6 * Real.sqrt 3) := by
  sorry

end triangle_properties_l3452_345286


namespace smallest_prime_divisor_of_sum_l3452_345294

theorem smallest_prime_divisor_of_sum (p : Nat) :
  Prime p ∧ p ∣ (3^15 + 11^9) ∧ ∀ q < p, Prime q → ¬(q ∣ (3^15 + 11^9)) →
  p = 2 :=
sorry

end smallest_prime_divisor_of_sum_l3452_345294


namespace charlie_data_overage_cost_l3452_345257

/-- Charlie's cell phone plan data usage and cost calculation --/
theorem charlie_data_overage_cost
  (data_limit : ℕ)
  (week1_usage week2_usage week3_usage week4_usage : ℕ)
  (overage_charge : ℕ)
  (h1 : data_limit = 8)
  (h2 : week1_usage = 2)
  (h3 : week2_usage = 3)
  (h4 : week3_usage = 5)
  (h5 : week4_usage = 10)
  (h6 : overage_charge = 120) :
  let total_usage := week1_usage + week2_usage + week3_usage + week4_usage
  let overage := total_usage - data_limit
  overage_charge / overage = 10 := by sorry

end charlie_data_overage_cost_l3452_345257


namespace gcd_15n_plus_4_9n_plus_2_max_2_l3452_345293

theorem gcd_15n_plus_4_9n_plus_2_max_2 :
  (∃ n : ℕ+, Nat.gcd (15 * n + 4) (9 * n + 2) = 2) ∧
  (∀ n : ℕ+, Nat.gcd (15 * n + 4) (9 * n + 2) ≤ 2) :=
by sorry

end gcd_15n_plus_4_9n_plus_2_max_2_l3452_345293


namespace distribute_five_balls_three_boxes_l3452_345264

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute_balls 5 3 = 21 := by
  sorry

end distribute_five_balls_three_boxes_l3452_345264


namespace highlighter_expense_is_30_l3452_345203

/-- The amount of money Heaven's brother spent on highlighters -/
def highlighter_expense (total_money : ℕ) (sharpener_price : ℕ) (notebook_price : ℕ) 
  (eraser_price : ℕ) (num_sharpeners : ℕ) (num_notebooks : ℕ) (num_erasers : ℕ) : ℕ :=
  total_money - (sharpener_price * num_sharpeners + notebook_price * num_notebooks + eraser_price * num_erasers)

/-- Theorem stating the amount spent on highlighters -/
theorem highlighter_expense_is_30 : 
  highlighter_expense 100 5 5 4 2 4 10 = 30 := by
  sorry

end highlighter_expense_is_30_l3452_345203


namespace kiera_muffins_count_l3452_345211

/-- Represents the number of items in an order -/
structure Order :=
  (muffins : ℕ)
  (fruitCups : ℕ)

/-- Calculates the cost of an order given the prices -/
def orderCost (order : Order) (muffinPrice fruitCupPrice : ℕ) : ℕ :=
  order.muffins * muffinPrice + order.fruitCups * fruitCupPrice

theorem kiera_muffins_count 
  (muffinPrice fruitCupPrice : ℕ)
  (francis : Order)
  (kiera : Order)
  (h1 : muffinPrice = 2)
  (h2 : fruitCupPrice = 3)
  (h3 : francis.muffins = 2)
  (h4 : francis.fruitCups = 2)
  (h5 : kiera.fruitCups = 1)
  (h6 : orderCost francis muffinPrice fruitCupPrice + 
        orderCost kiera muffinPrice fruitCupPrice = 17) :
  kiera.muffins = 2 := by
sorry

end kiera_muffins_count_l3452_345211


namespace lcm_gcd_product_10_15_l3452_345269

theorem lcm_gcd_product_10_15 :
  Nat.lcm 10 15 * Nat.gcd 10 15 = 150 := by
  sorry

end lcm_gcd_product_10_15_l3452_345269


namespace max_square_plots_l3452_345230

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Represents the available internal fencing -/
def availableFencing : ℝ := 2400

/-- Calculates the number of square plots given the number of plots along the width -/
def numPlots (n : ℕ) : ℕ := n * n * 2

/-- Calculates the amount of internal fencing needed for a given number of plots along the width -/
def fencingNeeded (n : ℕ) (field : FieldDimensions) : ℝ :=
  (2 * n - 1) * field.width + (n - 1) * field.length

/-- The main theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
    (h_length : field.length = 60) 
    (h_width : field.width = 30) :
    ∃ (n : ℕ), 
      numPlots n = 400 ∧ 
      fencingNeeded n field ≤ availableFencing ∧
      ∀ (m : ℕ), fencingNeeded m field ≤ availableFencing → numPlots m ≤ numPlots n := by
  sorry


end max_square_plots_l3452_345230


namespace largest_non_expressible_l3452_345206

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 36 * a + b ∧ a > 0 ∧ is_power_of_two b

theorem largest_non_expressible : 
  (∀ n > 104, expressible n) ∧ ¬(expressible 104) := by sorry

end largest_non_expressible_l3452_345206


namespace solutions_equation1_solutions_equation2_solutions_equation3_l3452_345252

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := 4 * (x - 1)^2 - 36 = 0
def equation2 (x : ℝ) : Prop := x^2 + 2*x - 3 = 0
def equation3 (x : ℝ) : Prop := x*(x - 4) = 8 - 2*x

-- Theorem stating the solutions for equation1
theorem solutions_equation1 : 
  ∀ x : ℝ, equation1 x ↔ (x = 4 ∨ x = -2) :=
sorry

-- Theorem stating the solutions for equation2
theorem solutions_equation2 : 
  ∀ x : ℝ, equation2 x ↔ (x = -3 ∨ x = 1) :=
sorry

-- Theorem stating the solutions for equation3
theorem solutions_equation3 : 
  ∀ x : ℝ, equation3 x ↔ (x = 4 ∨ x = -2) :=
sorry

end solutions_equation1_solutions_equation2_solutions_equation3_l3452_345252


namespace expand_polynomial_l3452_345256

theorem expand_polynomial (x : ℝ) : (5*x^2 + 7*x + 2) * 3*x = 15*x^3 + 21*x^2 + 6*x := by
  sorry

end expand_polynomial_l3452_345256


namespace no_bounded_sequences_with_property_l3452_345226

theorem no_bounded_sequences_with_property :
  ¬ ∃ (a b : ℕ → ℝ),
    (∃ M : ℝ, ∀ n, |a n| ≤ M ∧ |b n| ≤ M) ∧
    (∀ n m : ℕ, m > n → |a m - a n| > 1 / Real.sqrt n ∨ |b m - b n| > 1 / Real.sqrt n) :=
sorry

end no_bounded_sequences_with_property_l3452_345226


namespace annular_sector_area_l3452_345259

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  R : ℝ
  r : ℝ
  h : R > r

/-- A point on the larger circle of the annulus. -/
structure PointOnLargerCircle (A : Annulus) where
  P : ℝ × ℝ
  h : (P.1 - 0)^2 + (P.2 - 0)^2 = A.R^2

/-- A point on the smaller circle of the annulus. -/
structure PointOnSmallerCircle (A : Annulus) where
  Q : ℝ × ℝ
  h : (Q.1 - 0)^2 + (Q.2 - 0)^2 = A.r^2

/-- A tangent line to the smaller circle. -/
def IsTangent (A : Annulus) (P : PointOnLargerCircle A) (Q : PointOnSmallerCircle A) : Prop :=
  (P.P.1 - Q.Q.1)^2 + (P.P.2 - Q.Q.2)^2 = A.R^2 - A.r^2

/-- The theorem stating the area of the annular sector. -/
theorem annular_sector_area (A : Annulus) (P : PointOnLargerCircle A) (Q : PointOnSmallerCircle A)
    (θ : ℝ) (t : ℝ) (h_tangent : IsTangent A P Q) (h_t : t^2 = A.R^2 - A.r^2) :
    (θ/2 - π) * A.r^2 + θ * t^2 / 2 = θ * A.R^2 / 2 - π * A.r^2 := by
  sorry

#check annular_sector_area

end annular_sector_area_l3452_345259


namespace other_diagonal_length_l3452_345295

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

/-- Given a rhombus with area 330 and one diagonal 30, the other diagonal is 22 -/
theorem other_diagonal_length :
  ∀ (r : Rhombus), r.area = 330 ∧ r.d1 = 30 → r.d2 = 22 := by
  sorry

end other_diagonal_length_l3452_345295


namespace ratio_of_sum_to_difference_l3452_345283

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end ratio_of_sum_to_difference_l3452_345283


namespace quadratic_equation_complete_square_l3452_345251

theorem quadratic_equation_complete_square (m n : ℝ) : 
  (∀ x, 15 * x^2 - 30 * x - 45 = 0 ↔ (x + m)^2 = n) → m + n = 3 := by
  sorry

end quadratic_equation_complete_square_l3452_345251


namespace pear_problem_l3452_345262

theorem pear_problem (alyssa_pears nancy_pears carlos_pears given_away : ℕ) 
  (h1 : alyssa_pears = 42)
  (h2 : nancy_pears = 17)
  (h3 : carlos_pears = 25)
  (h4 : given_away = 5) :
  alyssa_pears + nancy_pears + carlos_pears - 3 * given_away = 69 := by
  sorry

end pear_problem_l3452_345262


namespace complex_fraction_simplification_l3452_345245

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - i) / (1 + i) = (1 : ℂ) / 2 - (3 : ℂ) / 2 * i := by sorry

end complex_fraction_simplification_l3452_345245


namespace fruit_drink_volume_l3452_345229

/-- Represents a fruit drink composed of grapefruit, lemon, and orange juice -/
structure FruitDrink where
  total : ℝ
  grapefruit : ℝ
  lemon : ℝ
  orange : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.grapefruit = 0.25 * drink.total)
  (h2 : drink.lemon = 0.35 * drink.total)
  (h3 : drink.orange = 20)
  (h4 : drink.total = drink.grapefruit + drink.lemon + drink.orange) :
  drink.total = 50 := by
  sorry


end fruit_drink_volume_l3452_345229


namespace range_of_4a_minus_2b_l3452_345210

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := by
  sorry

end range_of_4a_minus_2b_l3452_345210


namespace min_books_borrowed_l3452_345276

/-- Represents the minimum number of books borrowed by the remaining students -/
def min_books_remaining : ℕ := 4

theorem min_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 38) (h2 : no_books = 2) (h3 : one_book = 12) 
  (h4 : two_books = 10) (h5 : avg_books = 2) : 
  min_books_remaining = 4 := by
  sorry

#check min_books_borrowed

end min_books_borrowed_l3452_345276


namespace banana_count_l3452_345214

theorem banana_count (bananas apples oranges : ℕ) : 
  apples = 2 * bananas →
  oranges = 6 →
  bananas + apples + oranges = 12 →
  bananas = 2 := by
sorry

end banana_count_l3452_345214


namespace vector_sum_collinear_points_l3452_345277

/-- Given points A, B, C are collinear, O is not on their line, and 
    p⃗OA + q⃗OB + r⃗OC = 0⃗, then p + q + r = 0 -/
theorem vector_sum_collinear_points 
  (O A B C : EuclideanSpace ℝ (Fin 3))
  (p q r : ℝ) :
  Collinear ℝ ({A, B, C} : Set (EuclideanSpace ℝ (Fin 3))) →
  O ∉ affineSpan ℝ {A, B, C} →
  p • (A - O) + q • (B - O) + r • (C - O) = 0 →
  p + q + r = 0 := by
  sorry

end vector_sum_collinear_points_l3452_345277


namespace dans_remaining_money_l3452_345239

def remaining_money (initial_amount spending : ℚ) : ℚ :=
  initial_amount - spending

theorem dans_remaining_money :
  remaining_money 4 3 = 1 :=
by sorry

end dans_remaining_money_l3452_345239


namespace sufficient_not_necessary_condition_l3452_345225

/-- 
A proposition stating that "a>2 and b>2" is a sufficient but not necessary condition 
for "a+b>4 and ab>4" for real numbers a and b.
-/
theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∃ x y : ℝ, x > 2 ∧ y > 2 → x + y > 4 ∧ x * y > 4) ∧ 
  (∃ p q : ℝ, p + q > 4 ∧ p * q > 4 ∧ ¬(p > 2 ∧ q > 2)) :=
by sorry

end sufficient_not_necessary_condition_l3452_345225


namespace repeated_root_fraction_equation_l3452_345242

theorem repeated_root_fraction_equation (x m : ℝ) : 
  (∃ x, (x / (x - 3) + 1 = m / (x - 3)) ∧ 
        (∀ y, y ≠ x → y / (y - 3) + 1 ≠ m / (y - 3))) → 
  m = 3 := by
sorry

end repeated_root_fraction_equation_l3452_345242


namespace manuscript_completion_time_l3452_345204

theorem manuscript_completion_time 
  (original_computers : ℕ) 
  (original_time : ℚ) 
  (reduced_time_ratio : ℚ) 
  (additional_time : ℚ) :
  (original_computers : ℚ) / (original_computers + 3) = reduced_time_ratio →
  (original_computers : ℚ) / (original_computers - 3) = original_time / (original_time + additional_time) →
  reduced_time_ratio = 3/4 →
  additional_time = 5/6 →
  original_time = 5/3 := by
sorry

end manuscript_completion_time_l3452_345204


namespace amp_composition_l3452_345289

-- Define the & operation (postfix)
def postAmp (x : ℝ) : ℝ := 9 - x

-- Define the & operation (prefix)
def preAmp (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_composition : preAmp (postAmp 15) = -15 := by
  sorry

end amp_composition_l3452_345289


namespace math_team_count_is_480_l3452_345253

/-- The number of ways to form a six-member math team with 3 girls and 3 boys 
    from a club of 4 girls and 6 boys, where one team member is selected as captain -/
def math_team_count : ℕ := sorry

/-- The number of girls in the math club -/
def girls_in_club : ℕ := 4

/-- The number of boys in the math club -/
def boys_in_club : ℕ := 6

/-- The number of girls required in the team -/
def girls_in_team : ℕ := 3

/-- The number of boys required in the team -/
def boys_in_team : ℕ := 3

/-- The total number of team members -/
def team_size : ℕ := girls_in_team + boys_in_team

theorem math_team_count_is_480 : 
  math_team_count = (Nat.choose girls_in_club girls_in_team) * 
                    (Nat.choose boys_in_club boys_in_team) * 
                    team_size := by sorry

end math_team_count_is_480_l3452_345253


namespace no_negative_one_in_sequence_l3452_345221

def recurrence_sequence (p : ℕ) : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * recurrence_sequence p (n + 1) - p * recurrence_sequence p n

theorem no_negative_one_in_sequence (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) (h_not_five : p ≠ 5) :
  ∀ n, recurrence_sequence p n ≠ -1 :=
by sorry

end no_negative_one_in_sequence_l3452_345221


namespace fractional_equation_root_l3452_345232

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 4 ∧ (3 / (x - 4) + (x + m) / (4 - x) = 1)) → m = -1 := by
  sorry

end fractional_equation_root_l3452_345232


namespace probability_of_marked_section_on_top_l3452_345290

theorem probability_of_marked_section_on_top (n : ℕ) (h : n = 8) : 
  (1 : ℚ) / n = (1 : ℚ) / 8 := by
  sorry

#check probability_of_marked_section_on_top

end probability_of_marked_section_on_top_l3452_345290


namespace functional_equation_solution_l3452_345233

/-- A continuous function from positive reals to positive reals satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x, x > 0 → f x > 0) ∧
  ∀ x y, x > 0 → y > 0 → f (x + y) * (f x + f y) = f x * f y

/-- The theorem stating that any function satisfying the functional equation has the form f(x) = 1/(αx) for some α > 0 -/
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) :
  ∃ α : ℝ, α > 0 ∧ ∀ x, x > 0 → f x = 1 / (α * x) :=
sorry

end functional_equation_solution_l3452_345233


namespace seashells_given_l3452_345297

theorem seashells_given (initial_seashells current_seashells : ℕ) 
  (h1 : initial_seashells = 5)
  (h2 : current_seashells = 3) : 
  initial_seashells - current_seashells = 2 := by
  sorry

end seashells_given_l3452_345297


namespace ring_toss_total_earnings_l3452_345287

theorem ring_toss_total_earnings (first_44_days : ℕ) (remaining_10_days : ℕ) (total : ℕ) :
  first_44_days = 382 →
  remaining_10_days = 374 →
  total = first_44_days + remaining_10_days →
  total = 756 := by sorry

end ring_toss_total_earnings_l3452_345287


namespace f_monotonicity_and_minimum_l3452_345220

def f (a x : ℝ) := x^2 + 2*a*x + 2

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem f_monotonicity_and_minimum (a : ℝ) :
  (is_monotonic (f a) (-5) 5 ↔ a ≥ 5 ∨ a ≤ -5) ∧
  (∀ x ∈ Set.Icc (-5) 5, f a x ≥
    if a ≥ 5 then 27 - 10*a
    else if a ≥ -5 then 2 - a^2
    else 27 + 10*a) :=
  sorry

end f_monotonicity_and_minimum_l3452_345220


namespace condition_p_necessary_not_sufficient_l3452_345292

theorem condition_p_necessary_not_sufficient :
  (∀ a : ℝ, (|a| ≤ 1 → a ≤ 1)) ∧
  (∃ a : ℝ, a ≤ 1 ∧ ¬(|a| ≤ 1)) :=
by sorry

end condition_p_necessary_not_sufficient_l3452_345292


namespace floor_paving_cost_l3452_345285

/-- Calculates the total cost of paving a floor with different types of slabs -/
theorem floor_paving_cost (room_length room_width : ℝ)
  (square_slab_side square_slab_cost square_slab_percentage : ℝ)
  (rect_slab_length rect_slab_width rect_slab_cost rect_slab_percentage : ℝ)
  (tri_slab_height tri_slab_base tri_slab_cost tri_slab_percentage : ℝ) :
  room_length = 5.5 →
  room_width = 3.75 →
  square_slab_side = 1 →
  square_slab_cost = 800 →
  square_slab_percentage = 0.4 →
  rect_slab_length = 1.5 →
  rect_slab_width = 1 →
  rect_slab_cost = 1000 →
  rect_slab_percentage = 0.35 →
  tri_slab_height = 1 →
  tri_slab_base = 1 →
  tri_slab_cost = 1200 →
  tri_slab_percentage = 0.25 →
  square_slab_percentage + rect_slab_percentage + tri_slab_percentage = 1 →
  (room_length * room_width) * 
    (square_slab_percentage * square_slab_cost + 
     rect_slab_percentage * rect_slab_cost + 
     tri_slab_percentage * tri_slab_cost) = 20006.25 := by
  sorry

end floor_paving_cost_l3452_345285


namespace unique_triple_l3452_345265

theorem unique_triple : ∃! (x y z : ℕ), 
  100 ≤ x ∧ x < y ∧ y < z ∧ z < 1000 ∧ 
  (y - x = z - y) ∧ 
  (y * y = x * (z + 1000)) ∧
  x = 160 ∧ y = 560 ∧ z = 960 := by
  sorry

end unique_triple_l3452_345265


namespace smallest_three_square_representations_l3452_345281

/-- A function that represents the number of ways a positive integer can be expressed as the sum of three squares -/
def numThreeSquareRepresentations (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is expressible as the sum of three squares in three different ways -/
def hasThreeRepresentations (n : ℕ) : Prop :=
  numThreeSquareRepresentations n = 3

/-- Theorem stating that 30 is the smallest positive integer with three different representations as the sum of three squares -/
theorem smallest_three_square_representations :
  (∀ m : ℕ, m > 0 → m < 30 → ¬(hasThreeRepresentations m)) ∧
  hasThreeRepresentations 30 := by sorry

end smallest_three_square_representations_l3452_345281


namespace sqrt_13_parts_sum_l3452_345240

theorem sqrt_13_parts_sum (x y : ℝ) : 
  (x = ⌊Real.sqrt 13⌋) → 
  (y = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  (2 * x - y + Real.sqrt 13 = 9) := by
sorry

end sqrt_13_parts_sum_l3452_345240


namespace ellipse_hyperbola_same_foci_l3452_345261

/-- Given an ellipse and a hyperbola with the same foci, prove that the parameter a of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → a > 0) → -- Ellipse equation condition
  (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1) → -- Hyperbola equation
  (∃ c : ℝ, c^2 = 7 ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → x^2 + y^2 = a^2 + c^2) ∧ 
    (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1 → x^2 - y^2 = 4 + c^2)) → -- Same foci condition
  a = 4 := by
sorry

end ellipse_hyperbola_same_foci_l3452_345261


namespace smallest_divisible_by_8_11_15_l3452_345255

theorem smallest_divisible_by_8_11_15 : ∃! n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(8 ∣ m ∧ 11 ∣ m ∧ 15 ∣ m)) ∧ 
  (8 ∣ n) ∧ (11 ∣ n) ∧ (15 ∣ n) := by
  sorry

end smallest_divisible_by_8_11_15_l3452_345255


namespace inequality_solution_set_l3452_345272

theorem inequality_solution_set (a b : ℝ) (d : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x^2 + a*x + b > 0 ↔ x ≠ d) :
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x - b < 0 ↔ x₁ < x ∧ x < x₂) ∧ x₁ * x₂ ≤ 0 := by
  sorry

end inequality_solution_set_l3452_345272


namespace runners_journey_l3452_345207

/-- A runner's journey with changing speeds -/
theorem runners_journey (initial_speed : ℝ) (tired_speed : ℝ) (total_distance : ℝ) (total_time : ℝ)
  (h1 : initial_speed = 15)
  (h2 : tired_speed = 10)
  (h3 : total_distance = 100)
  (h4 : total_time = 9) :
  ∃ (initial_time : ℝ), initial_time = 2 ∧ 
    initial_speed * initial_time + tired_speed * (total_time - initial_time) = total_distance := by
  sorry

end runners_journey_l3452_345207


namespace smallest_fraction_greater_than_three_fifths_l3452_345258

theorem smallest_fraction_greater_than_three_fifths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 3 / 5 →
    (59 : ℚ) / 98 ≤ (a : ℚ) / b :=
by sorry

end smallest_fraction_greater_than_three_fifths_l3452_345258


namespace second_chapter_pages_l3452_345227

theorem second_chapter_pages (total_chapters : ℕ) (total_pages : ℕ) (second_chapter_length : ℕ) :
  total_chapters = 2 →
  total_pages = 81 →
  second_chapter_length = 68 →
  second_chapter_length = 68 := by
  sorry

end second_chapter_pages_l3452_345227


namespace prob_not_at_ends_eight_chairs_l3452_345215

/-- The number of chairs in the row -/
def n : ℕ := 8

/-- The probability of two people not sitting at either end when randomly choosing seats in a row of n chairs -/
def prob_not_at_ends (n : ℕ) : ℚ :=
  1 - (2 + 2 * (n - 2)) / (n.choose 2)

theorem prob_not_at_ends_eight_chairs :
  prob_not_at_ends n = 3/7 := by sorry

end prob_not_at_ends_eight_chairs_l3452_345215


namespace min_distance_midpoint_to_origin_min_distance_midpoint_to_origin_is_5sqrt2_l3452_345270

/-- The minimum distance from the midpoint of two points on parallel lines x-y-5=0 and x-y-15=0 to the origin is 5√2. -/
theorem min_distance_midpoint_to_origin : ℝ → Prop := 
  fun d => ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - y₁ = 5) →
    (x₂ - y₂ = 15) →
    let midpoint_x := (x₁ + x₂) / 2
    let midpoint_y := (y₁ + y₂) / 2
    d = Real.sqrt 50

-- The proof is omitted
theorem min_distance_midpoint_to_origin_is_5sqrt2 : 
  min_distance_midpoint_to_origin (5 * Real.sqrt 2) :=
sorry

end min_distance_midpoint_to_origin_min_distance_midpoint_to_origin_is_5sqrt2_l3452_345270


namespace cost_price_calculation_l3452_345284

/-- Represents a type of cloth with its sales information -/
structure ClothType where
  quantity : ℕ      -- Quantity sold in meters
  totalPrice : ℕ    -- Total selling price in Rs.
  profitPerMeter : ℕ -- Profit per meter in Rs.

/-- Calculates the cost price per meter for a given cloth type -/
def costPricePerMeter (cloth : ClothType) : ℕ :=
  cloth.totalPrice / cloth.quantity - cloth.profitPerMeter

/-- The trader's cloth inventory -/
def traderInventory : List ClothType :=
  [
    { quantity := 85, totalPrice := 8500, profitPerMeter := 15 },  -- Type A
    { quantity := 120, totalPrice := 10200, profitPerMeter := 12 }, -- Type B
    { quantity := 60, totalPrice := 4200, profitPerMeter := 10 }   -- Type C
  ]

theorem cost_price_calculation (inventory : List ClothType) :
  ∀ cloth ∈ inventory,
    costPricePerMeter cloth =
      cloth.totalPrice / cloth.quantity - cloth.profitPerMeter :=
by
  sorry

#eval traderInventory.map costPricePerMeter

end cost_price_calculation_l3452_345284


namespace ferry_problem_l3452_345246

/-- The ferry problem -/
theorem ferry_problem (speed_p speed_q : ℝ) (time_p distance_q : ℝ) :
  speed_p = 8 →
  time_p = 2 →
  distance_q = 3 * speed_p * time_p →
  speed_q = speed_p + 4 →
  distance_q / speed_q - time_p = 2 := by
  sorry

end ferry_problem_l3452_345246


namespace chicken_rabbit_equations_correct_l3452_345238

/-- Represents the "chicken-rabbit in the same cage" problem -/
structure ChickenRabbitProblem where
  total_heads : ℕ
  total_feet : ℕ
  chickens : ℕ
  rabbits : ℕ

/-- The system of equations for the chicken-rabbit problem -/
def correct_equations (problem : ChickenRabbitProblem) : Prop :=
  problem.chickens + problem.rabbits = problem.total_heads ∧
  2 * problem.chickens + 4 * problem.rabbits = problem.total_feet

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem chicken_rabbit_equations_correct (problem : ChickenRabbitProblem) 
  (h1 : problem.total_heads = 35)
  (h2 : problem.total_feet = 94) :
  correct_equations problem :=
sorry

end chicken_rabbit_equations_correct_l3452_345238


namespace probability_neither_mix_l3452_345273

/-- Represents the set of all buyers -/
def TotalBuyers : ℕ := 100

/-- Represents the number of buyers who purchase cake mix -/
def CakeMixBuyers : ℕ := 50

/-- Represents the number of buyers who purchase muffin mix -/
def MuffinMixBuyers : ℕ := 40

/-- Represents the number of buyers who purchase both cake mix and muffin mix -/
def BothMixesBuyers : ℕ := 19

/-- Theorem stating the probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem probability_neither_mix (TotalBuyers CakeMixBuyers MuffinMixBuyers BothMixesBuyers : ℕ) 
  (h1 : TotalBuyers = 100)
  (h2 : CakeMixBuyers = 50)
  (h3 : MuffinMixBuyers = 40)
  (h4 : BothMixesBuyers = 19) :
  (TotalBuyers - (CakeMixBuyers + MuffinMixBuyers - BothMixesBuyers)) / TotalBuyers = 29 / 100 := by
  sorry

end probability_neither_mix_l3452_345273


namespace investment_growth_l3452_345209

/-- Calculates the final amount after simple interest --/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the conditions, the final amount after 6 years is $380 --/
theorem investment_growth (principal : ℝ) (amount_after_2_years : ℝ) :
  principal = 200 →
  amount_after_2_years = 260 →
  final_amount principal ((amount_after_2_years - principal) / (principal * 2)) 6 = 380 :=
by sorry

end investment_growth_l3452_345209


namespace fraction_equality_l3452_345237

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := by
  sorry

end fraction_equality_l3452_345237


namespace solution_equivalence_l3452_345280

-- Define the set of real numbers greater than 1
def greaterThanOne : Set ℝ := {x | x > 1}

-- Define the solution set of ax - b > 0
def solutionSet (a b : ℝ) : Set ℝ := {x | a * x - b > 0}

-- Define the set (-∞,-1)∪(2,+∞)
def targetSet : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem statement
theorem solution_equivalence (a b : ℝ) :
  solutionSet a b = greaterThanOne →
  {x : ℝ | (a * x + b) / (x - 2) > 0} = targetSet := by
  sorry

end solution_equivalence_l3452_345280


namespace distinct_numbers_count_l3452_345223

/-- Represents the possible states of a matchstick (present or removed) --/
inductive MatchstickState
| Present
| Removed

/-- Represents the configuration of matchsticks in the symbol --/
structure MatchstickConfiguration :=
(top : MatchstickState)
(bottom : MatchstickState)
(left : MatchstickState)
(right : MatchstickState)

/-- Defines the set of valid number representations --/
def ValidNumberRepresentations : Set MatchstickConfiguration := sorry

/-- Counts the number of distinct valid number representations --/
def CountDistinctNumbers : Nat := sorry

/-- Theorem stating that the number of distinct numbers obtainable is 5 --/
theorem distinct_numbers_count :
  CountDistinctNumbers = 5 := by sorry

end distinct_numbers_count_l3452_345223


namespace inequality_holds_C_is_maximum_l3452_345278

noncomputable def C : ℝ := (Real.sqrt (13 + 16 * Real.sqrt 2) - 1) / 2

theorem inequality_holds (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x^3 + y^3 + z^3 + C * (x*y^2 + y*z^2 + z*x^2) ≥ (C + 1) * (x^2*y + y^2*z + z^2*x) :=
by sorry

theorem C_is_maximum : 
  ∀ D : ℝ, D > C → ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^3 + y^3 + z^3 + D * (x*y^2 + y*z^2 + z*x^2) < (D + 1) * (x^2*y + y^2*z + z^2*x) :=
by sorry

end inequality_holds_C_is_maximum_l3452_345278


namespace johns_presents_worth_l3452_345243

/-- The total worth of John's presents to his fiancee -/
def total_worth (ring_cost car_cost brace_cost : ℕ) : ℕ :=
  ring_cost + car_cost + brace_cost

/-- Theorem stating the total worth of John's presents -/
theorem johns_presents_worth :
  ∃ (ring_cost car_cost brace_cost : ℕ),
    ring_cost = 4000 ∧
    car_cost = 2000 ∧
    brace_cost = 2 * ring_cost ∧
    total_worth ring_cost car_cost brace_cost = 14000 := by
  sorry

end johns_presents_worth_l3452_345243


namespace sqrt_inequality_l3452_345212

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + Real.sqrt (x - 1) := by
  sorry

end sqrt_inequality_l3452_345212


namespace company_price_ratio_l3452_345205

/-- Given companies A, B, and KW where:
    - KW's price is 30% more than A's assets
    - KW's price is 100% more than B's assets
    Prove that KW's price is approximately 78.79% of A and B's combined assets -/
theorem company_price_ratio (P A B : ℝ) 
  (h1 : P = A + 0.3 * A) 
  (h2 : P = B + B) : 
  ∃ ε > 0, abs (P / (A + B) - 0.7879) < ε :=
sorry

end company_price_ratio_l3452_345205


namespace fence_cost_per_foot_l3452_345219

theorem fence_cost_per_foot 
  (plot_area : ℝ) 
  (total_cost : ℝ) 
  (h1 : plot_area = 289) 
  (h2 : total_cost = 3808) : 
  total_cost / (4 * Real.sqrt plot_area) = 56 := by
sorry

end fence_cost_per_foot_l3452_345219


namespace average_age_after_leaving_l3452_345244

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age1 : ℕ) (leaving_age2 : ℕ) (remaining_people : ℕ) :
  initial_people = 6 →
  initial_average = 25 →
  leaving_age1 = 20 →
  leaving_age2 = 22 →
  remaining_people = 4 →
  (initial_people : ℚ) * initial_average - (leaving_age1 + leaving_age2 : ℚ) = 
    (remaining_people : ℚ) * 27 := by
  sorry

#check average_age_after_leaving

end average_age_after_leaving_l3452_345244


namespace expression_value_l3452_345218

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := by
  sorry

end expression_value_l3452_345218


namespace x_value_proof_l3452_345298

theorem x_value_proof : ∃ X : ℝ, 
  X * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  abs (X - 0.6) < 0.0000000000000001 := by
  sorry

end x_value_proof_l3452_345298


namespace coefficient_d_nonzero_l3452_345299

-- Define the polynomial Q(x)
def Q (a b c d e f : ℝ) (x : ℝ) : ℝ :=
  x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

-- Define the property of having six distinct x-intercepts
def has_six_distinct_intercepts (a b c d e f : ℝ) : Prop :=
  ∃ p q r s t : ℝ, 
    (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ (p ≠ 0) ∧
    (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ (q ≠ 0) ∧
    (r ≠ s) ∧ (r ≠ t) ∧ (r ≠ 0) ∧
    (s ≠ t) ∧ (s ≠ 0) ∧
    (t ≠ 0) ∧
    (Q a b c d e f p = 0) ∧ (Q a b c d e f q = 0) ∧ 
    (Q a b c d e f r = 0) ∧ (Q a b c d e f s = 0) ∧ 
    (Q a b c d e f t = 0) ∧ (Q a b c d e f 0 = 0)

-- Theorem statement
theorem coefficient_d_nonzero 
  (a b c d e f : ℝ) 
  (h : has_six_distinct_intercepts a b c d e f) : 
  d ≠ 0 := by
  sorry

end coefficient_d_nonzero_l3452_345299


namespace solutions_equation_1_solutions_equation_2_l3452_345224

-- Equation 1
theorem solutions_equation_1 : 
  ∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4 := by sorry

-- Equation 2
theorem solutions_equation_2 : 
  ∀ x : ℝ, x*(x-1) + 2*(x-1) = 0 ↔ x = 1 ∨ x = -2 := by sorry

end solutions_equation_1_solutions_equation_2_l3452_345224


namespace rod_cutting_l3452_345241

/-- Given a rod of length 34 meters that can be cut into 40 equal pieces,
    prove that each piece is 0.85 meters long. -/
theorem rod_cutting (rod_length : ℝ) (num_pieces : ℕ) (piece_length : ℝ) 
  (h1 : rod_length = 34)
  (h2 : num_pieces = 40)
  (h3 : piece_length * num_pieces = rod_length) :
  piece_length = 0.85 := by
  sorry

end rod_cutting_l3452_345241


namespace circle_center_radius_sum_l3452_345217

/-- Given a circle D with equation x^2 + 20x + y^2 + 18y = -36,
    prove that its center coordinates (p, q) and radius s satisfy p + q + s = -19 + Real.sqrt 145 -/
theorem circle_center_radius_sum (x y : ℝ) :
  x^2 + 20*x + y^2 + 18*y = -36 →
  ∃ (p q s : ℝ), (∀ (x y : ℝ), (x - p)^2 + (y - q)^2 = s^2 ↔ x^2 + 20*x + y^2 + 18*y = -36) ∧
                 p + q + s = -19 + Real.sqrt 145 := by
  sorry

end circle_center_radius_sum_l3452_345217


namespace cubic_polynomial_roots_l3452_345222

theorem cubic_polynomial_roots (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 1) →
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 1 ∧ x ≠ 2 - Real.sqrt 5) :=
by sorry

end cubic_polynomial_roots_l3452_345222


namespace no_roots_implies_non_integer_l3452_345236

theorem no_roots_implies_non_integer (a b : ℝ) : 
  a ≠ b →
  (∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) →
  ¬(∃ n : ℤ, 20*(b-a) = n) :=
by sorry

end no_roots_implies_non_integer_l3452_345236


namespace geometric_progression_solution_l3452_345288

theorem geometric_progression_solution :
  ∃! x : ℚ, ((-10 + x)^2 = (-30 + x) * (40 + x)) ∧ x = 130/3 := by
  sorry

end geometric_progression_solution_l3452_345288


namespace team_a_builds_30m_per_day_l3452_345271

/-- Represents the daily road-building rate of Team A in meters -/
def team_a_rate : ℝ := 30

/-- Represents the daily road-building rate of Team B in meters -/
def team_b_rate : ℝ := team_a_rate + 10

/-- Represents the total length of road built by Team A in meters -/
def team_a_total : ℝ := 120

/-- Represents the total length of road built by Team B in meters -/
def team_b_total : ℝ := 160

/-- Theorem stating that Team A's daily rate is 30m, given the problem conditions -/
theorem team_a_builds_30m_per_day :
  (team_a_total / team_a_rate = team_b_total / team_b_rate) ∧
  (team_b_rate = team_a_rate + 10) ∧
  (team_a_rate = 30) := by sorry

end team_a_builds_30m_per_day_l3452_345271


namespace system_solution_l3452_345263

theorem system_solution (x y z : ℝ) : 
  (x * y + z = 40 ∧ x * z + y = 51 ∧ x + y + z = 19) ↔ 
  ((x = 12 ∧ y = 3 ∧ z = 4) ∨ (x = 6 ∧ y = 5.4 ∧ z = 7.6)) := by
sorry

end system_solution_l3452_345263


namespace statue_of_liberty_model_height_l3452_345208

/-- The scale ratio of the model to the actual size -/
def scaleRatio : ℚ := 1 / 25

/-- The actual height of the Statue of Liberty in feet -/
def actualHeight : ℕ := 305

/-- Rounds a rational number to the nearest integer -/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The height of the scale model in feet -/
def modelHeight : ℚ := actualHeight * scaleRatio

theorem statue_of_liberty_model_height :
  roundToNearest modelHeight = 12 := by
  sorry

end statue_of_liberty_model_height_l3452_345208


namespace eight_digit_permutations_eq_1680_l3452_345247

/-- The number of different positive, eight-digit integers that can be formed
    using the digits 2, 2, 2, 5, 5, 7, 9, and 9 -/
def eight_digit_permutations : ℕ :=
  Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, eight-digit integers
    that can be formed using the digits 2, 2, 2, 5, 5, 7, 9, and 9 is 1680 -/
theorem eight_digit_permutations_eq_1680 :
  eight_digit_permutations = 1680 := by
  sorry

end eight_digit_permutations_eq_1680_l3452_345247


namespace jessica_age_problem_l3452_345200

/-- Jessica's age problem -/
theorem jessica_age_problem (jessica_age_at_death : ℕ) (mother_age_at_death : ℕ) (years_since_death : ℕ) :
  jessica_age_at_death = mother_age_at_death / 2 →
  mother_age_at_death + years_since_death = 70 →
  years_since_death = 10 →
  jessica_age_at_death + years_since_death = 40 :=
by
  sorry

#check jessica_age_problem

end jessica_age_problem_l3452_345200


namespace quadratic_equation_roots_l3452_345234

theorem quadratic_equation_roots (θ : Real) (m : Real) :
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  (∃ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0) →
  (∃ x, x = Real.sin θ ∨ x = Real.cos θ) →
  (Real.sin θ / (1 - Real.cos θ) + Real.cos θ / (1 - Real.tan θ) = (3 + 5 * Real.sqrt 3) / 4) ∧
  (m = Real.sqrt 3 / 4) ∧
  ((Real.sin θ = Real.sqrt 3 / 2 ∧ Real.cos θ = 1 / 2 ∧ θ = Real.pi / 3) ∨
   (Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 ∧ θ = Real.pi / 6)) :=
by sorry

end quadratic_equation_roots_l3452_345234


namespace converse_of_square_equals_one_l3452_345275

theorem converse_of_square_equals_one (a : ℝ) : 
  (∀ a, a = 1 → a^2 = 1) → (∀ a, a^2 = 1 → a = 1) := by
  sorry

end converse_of_square_equals_one_l3452_345275


namespace house_of_cards_height_l3452_345266

/-- Given a triangular-shaped house of cards with base 40 cm,
    prove that if the total area of three similar houses is 1200 cm²,
    then the height of each house is 20 cm. -/
theorem house_of_cards_height
  (base : ℝ)
  (total_area : ℝ)
  (num_houses : ℕ)
  (h_base : base = 40)
  (h_total_area : total_area = 1200)
  (h_num_houses : num_houses = 3) :
  let area := total_area / num_houses
  let height := 2 * area / base
  height = 20 := by
sorry

end house_of_cards_height_l3452_345266


namespace nested_fraction_equation_l3452_345213

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225 / 68 → x = -102 / 19 := by
sorry

end nested_fraction_equation_l3452_345213


namespace checkerboard_existence_l3452_345201

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is on the boundary of the board -/
def isBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic (b : Board) (i j : Fin 100) : Prop :=
  b i j = b (i+1) j ∧ b i j = b i (j+1) ∧ b i j = b (i+1) (j+1)

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard (b : Board) (i j : Fin 100) : Prop :=
  (b i j = b (i+1) (j+1) ∧ b (i+1) j = b i (j+1) ∧ b i j ≠ b (i+1) j)

theorem checkerboard_existence (b : Board) 
  (h1 : ∀ i j, isBoundary i j → b i j = Color.Black)
  (h2 : ∀ i j, ¬isMonochromatic b i j) :
  ∃ i j, isCheckerboard b i j :=
sorry

end checkerboard_existence_l3452_345201


namespace weight_difference_l3452_345267

/-- Proves that given Robbie's weight, Patty's initial weight relative to Robbie's, and Patty's weight loss, 
    the difference between Patty's current weight and Robbie's weight is 115 pounds. -/
theorem weight_difference (robbie_weight : ℝ) (patty_initial_factor : ℝ) (patty_weight_loss : ℝ) : 
  robbie_weight = 100 → 
  patty_initial_factor = 4.5 → 
  patty_weight_loss = 235 → 
  patty_initial_factor * robbie_weight - patty_weight_loss - robbie_weight = 115 := by
  sorry


end weight_difference_l3452_345267


namespace boxer_win_ratio_is_one_l3452_345248

/-- Represents a boxer's career statistics -/
structure BoxerStats where
  wins_before_first_loss : ℕ
  total_losses : ℕ
  win_loss_difference : ℕ

/-- Calculates the ratio of wins after first loss to wins before first loss -/
def win_ratio (stats : BoxerStats) : ℚ :=
  let wins_after_first_loss := stats.win_loss_difference + stats.total_losses - stats.wins_before_first_loss
  wins_after_first_loss / stats.wins_before_first_loss

/-- Theorem stating that for a boxer with given statistics, the win ratio is 1 -/
theorem boxer_win_ratio_is_one (stats : BoxerStats)
  (h1 : stats.wins_before_first_loss = 15)
  (h2 : stats.total_losses = 2)
  (h3 : stats.win_loss_difference = 28) :
  win_ratio stats = 1 := by
  sorry

end boxer_win_ratio_is_one_l3452_345248


namespace smallest_among_given_numbers_l3452_345235

theorem smallest_among_given_numbers : 
  let a := Real.sqrt 3
  let b := -(1/3 : ℝ)
  let c := -2
  let d := 0
  c < b ∧ c < d ∧ c < a :=
by sorry

end smallest_among_given_numbers_l3452_345235


namespace stratified_optimal_survey1_simple_random_optimal_survey2_l3452_345260

/-- Represents the income level of a family -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing the conditions of Survey 1 -/
structure Survey1 where
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

/-- Structure representing the conditions of Survey 2 -/
structure Survey2 where
  totalStudents : Nat
  sampleSize : Nat

/-- Function to determine the optimal sampling method for Survey 1 -/
def optimalMethodSurvey1 (s : Survey1) : SamplingMethod := sorry

/-- Function to determine the optimal sampling method for Survey 2 -/
def optimalMethodSurvey2 (s : Survey2) : SamplingMethod := sorry

/-- Theorem stating that stratified sampling is optimal for Survey 1 -/
theorem stratified_optimal_survey1 (s : Survey1) :
  s.highIncomeFamilies = 125 →
  s.middleIncomeFamilies = 200 →
  s.lowIncomeFamilies = 95 →
  s.sampleSize = 100 →
  optimalMethodSurvey1 s = SamplingMethod.Stratified :=
by sorry

/-- Theorem stating that simple random sampling is optimal for Survey 2 -/
theorem simple_random_optimal_survey2 (s : Survey2) :
  s.totalStudents = 5 →
  s.sampleSize = 3 →
  optimalMethodSurvey2 s = SamplingMethod.SimpleRandom :=
by sorry

end stratified_optimal_survey1_simple_random_optimal_survey2_l3452_345260


namespace second_floor_bedrooms_l3452_345282

theorem second_floor_bedrooms (total_bedrooms first_floor_bedrooms : ℕ) 
  (h1 : total_bedrooms = 10)
  (h2 : first_floor_bedrooms = 8) :
  total_bedrooms - first_floor_bedrooms = 2 := by
  sorry

end second_floor_bedrooms_l3452_345282


namespace train_length_l3452_345250

/-- Given a train traveling at 72 km/hr crossing a 270 m long platform in 26 seconds,
    the length of the train is 250 meters. -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * 1000 / 3600 →
  platform_length = 270 →
  crossing_time = 26 →
  speed * crossing_time - platform_length = 250 :=
by sorry

end train_length_l3452_345250


namespace equidistant_point_y_coord_l3452_345202

/-- The y-coordinate of the point on the y-axis equidistant from A(3, 0) and B(1, -6) is -7/3 -/
theorem equidistant_point_y_coord :
  ∃ y : ℝ, (3^2 + y^2 = 1^2 + (y + 6)^2) ∧ y = -7/3 := by
sorry

end equidistant_point_y_coord_l3452_345202


namespace weeds_never_cover_entire_field_l3452_345291

/-- Represents a 10x10 grid -/
def Grid := Fin 10 → Fin 10 → Bool

/-- The initial state of the grid with 9 occupied cells -/
def initial_state : Grid := sorry

/-- Checks if a cell is adjacent to at least two occupied cells -/
def has_two_adjacent_occupied (g : Grid) (i j : Fin 10) : Bool := sorry

/-- The next state of the grid after one step of spreading -/
def next_state (g : Grid) : Grid := sorry

/-- The state of the grid after n steps of spreading -/
def state_after_n_steps (n : ℕ) : Grid := sorry

/-- Counts the number of occupied cells in the grid -/
def count_occupied (g : Grid) : ℕ := sorry

theorem weeds_never_cover_entire_field :
  ∀ n : ℕ, count_occupied (state_after_n_steps n) < 100 := by sorry

end weeds_never_cover_entire_field_l3452_345291


namespace water_needed_to_fill_glasses_l3452_345231

theorem water_needed_to_fill_glasses (num_glasses : ℕ) (glass_capacity : ℚ) (current_fullness : ℚ) :
  num_glasses = 10 →
  glass_capacity = 6 →
  current_fullness = 4/5 →
  (num_glasses : ℚ) * glass_capacity * (1 - current_fullness) = 12 := by
  sorry

end water_needed_to_fill_glasses_l3452_345231


namespace pizza_theorem_l3452_345228

def pizza_problem (total_pepperoni : ℕ) (fallen_pepperoni : ℕ) : Prop :=
  let half_pizza_pepperoni := total_pepperoni / 2
  let quarter_pizza_pepperoni := half_pizza_pepperoni / 2
  quarter_pizza_pepperoni - fallen_pepperoni = 9

theorem pizza_theorem : pizza_problem 40 1 := by
  sorry

end pizza_theorem_l3452_345228


namespace alternate_arrangement_probability_l3452_345279

/-- The number of male fans -/
def num_male : ℕ := 3

/-- The number of female fans -/
def num_female : ℕ := 3

/-- The total number of fans -/
def total_fans : ℕ := num_male + num_female

/-- The number of ways to arrange fans alternately -/
def alternate_arrangements : ℕ := 2 * (Nat.factorial num_male) * (Nat.factorial num_female)

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := Nat.factorial total_fans

/-- The probability of arranging fans alternately -/
def prob_alternate : ℚ := alternate_arrangements / total_arrangements

theorem alternate_arrangement_probability :
  prob_alternate = 1 / 10 := by
  sorry

end alternate_arrangement_probability_l3452_345279


namespace abs_T_equals_128_sqrt_2_l3452_345268

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^15 - (1 - i)^15

-- Theorem statement
theorem abs_T_equals_128_sqrt_2 : Complex.abs T = 128 * Real.sqrt 2 := by
  sorry

end abs_T_equals_128_sqrt_2_l3452_345268
