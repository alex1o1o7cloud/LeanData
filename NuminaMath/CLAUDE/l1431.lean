import Mathlib

namespace at_most_one_negative_l1431_143173

theorem at_most_one_negative (a b c : ℝ) (sum_nonneg : a + b + c ≥ 0) (product_nonpos : a * b * c ≤ 0) :
  ¬(((a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))) := by
  sorry

end at_most_one_negative_l1431_143173


namespace binomial_square_proof_l1431_143171

theorem binomial_square_proof : 
  ∃ (r s : ℚ), (r * X + s) ^ 2 = (196 / 9 : ℚ) * X ^ 2 + 28 * X + 9 := by
  sorry

end binomial_square_proof_l1431_143171


namespace point_A_coordinates_l1431_143115

/-- A point in the second quadrant of the Cartesian coordinate system with coordinates dependent on an integer m -/
def point_A (m : ℤ) : ℝ × ℝ := (7 - 2*m, 5 - m)

/-- Predicate to check if a point is in the second quadrant -/
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- Theorem stating that if A(7-2m, 5-m) is in the second quadrant and m is an integer, then A(-1, 1) is the only solution -/
theorem point_A_coordinates : 
  ∃! m : ℤ, in_second_quadrant (point_A m) ∧ point_A m = (-1, 1) :=
sorry

end point_A_coordinates_l1431_143115


namespace average_is_three_l1431_143180

/-- Given four real numbers A, B, C, and D satisfying certain conditions,
    prove that their average is 3. -/
theorem average_is_three (A B C D : ℝ) 
    (eq1 : 501 * C - 2004 * A = 3006)
    (eq2 : 2502 * B + 6006 * A = 10010)
    (eq3 : D = A + 2) :
    (A + B + C + D) / 4 = 3 := by
  sorry

end average_is_three_l1431_143180


namespace repeating_decimal_equiv_l1431_143154

-- Define the repeating decimal 0.4̄13
def repeating_decimal : ℚ := 409 / 990

-- Theorem statement
theorem repeating_decimal_equiv : 
  repeating_decimal = 409 / 990 ∧ 
  (∀ n d : ℕ, n / d = 409 / 990 → d ≤ 990) :=
by sorry

end repeating_decimal_equiv_l1431_143154


namespace complex_square_roots_l1431_143141

theorem complex_square_roots (z : ℂ) : z^2 = -77 - 36*I ↔ z = 2 - 9*I ∨ z = -2 + 9*I := by
  sorry

end complex_square_roots_l1431_143141


namespace prime_pairs_problem_l1431_143185

theorem prime_pairs_problem :
  ∀ p q : ℕ,
    1 < p → p < 100 →
    1 < q → q < 100 →
    Prime p →
    Prime q →
    Prime (p + 6) →
    Prime (p + 10) →
    Prime (q + 4) →
    Prime (q + 10) →
    Prime (p + q + 1) →
    ((p = 7 ∧ q = 3) ∨ (p = 13 ∧ q = 3) ∨ (p = 37 ∧ q = 3) ∨ (p = 97 ∧ q = 3)) :=
by sorry

end prime_pairs_problem_l1431_143185


namespace original_cost_price_satisfies_conditions_l1431_143191

/-- The original cost price of a computer satisfying given conditions -/
def original_cost_price : ℝ := 40

/-- The selling price of the computer -/
def selling_price : ℝ := 48

/-- The decrease rate of the cost price -/
def cost_decrease_rate : ℝ := 0.04

/-- The increase rate of the profit margin -/
def profit_margin_increase_rate : ℝ := 0.05

/-- Theorem stating that the original cost price satisfies all given conditions -/
theorem original_cost_price_satisfies_conditions :
  let new_cost_price := original_cost_price * (1 - cost_decrease_rate)
  let original_profit_margin := (selling_price - original_cost_price) / original_cost_price
  let new_profit_margin := (selling_price - new_cost_price) / new_cost_price
  new_profit_margin = original_profit_margin + profit_margin_increase_rate := by
  sorry


end original_cost_price_satisfies_conditions_l1431_143191


namespace complex_fraction_square_l1431_143118

theorem complex_fraction_square (m n : ℝ) (h : m * (1 + Complex.I) = 1 + n * Complex.I) :
  ((m + n * Complex.I) / (m - n * Complex.I))^2 = -1 := by
  sorry

end complex_fraction_square_l1431_143118


namespace pumpkin_pie_cost_pumpkin_pie_cost_proof_l1431_143121

/-- The cost to make a pumpkin pie given the following conditions:
  * 10 pumpkin pies and 12 cherry pies are made
  * Cherry pies cost $5 each to make
  * The total profit is $20
  * Each pie is sold for $5
-/
theorem pumpkin_pie_cost : ℝ :=
  let num_pumpkin_pies : ℕ := 10
  let num_cherry_pies : ℕ := 12
  let cherry_pie_cost : ℝ := 5
  let profit : ℝ := 20
  let selling_price : ℝ := 5
  3

/-- Proof that the cost to make each pumpkin pie is $3 -/
theorem pumpkin_pie_cost_proof :
  let num_pumpkin_pies : ℕ := 10
  let num_cherry_pies : ℕ := 12
  let cherry_pie_cost : ℝ := 5
  let profit : ℝ := 20
  let selling_price : ℝ := 5
  pumpkin_pie_cost = 3 := by
  sorry

end pumpkin_pie_cost_pumpkin_pie_cost_proof_l1431_143121


namespace z_in_second_quadrant_l1431_143122

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1-i)z = 2i
def equation (z : ℂ) : Prop := (1 - i) * z = 2 * i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end z_in_second_quadrant_l1431_143122


namespace stability_comparison_l1431_143197

/-- Represents a student's performance in a series of matches -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines the stability of scores based on variance -/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_avg : student_A.average_score = student_B.average_score)
  (h_var_A : student_A.variance = 0.2)
  (h_var_B : student_B.variance = 0.8) :
  more_stable student_A student_B :=
sorry

end stability_comparison_l1431_143197


namespace max_volume_at_10cm_l1431_143190

/-- The length of the original sheet in centimeters -/
def sheet_length : ℝ := 90

/-- The width of the original sheet in centimeters -/
def sheet_width : ℝ := 48

/-- The side length of the cut-out squares in centimeters -/
def cut_length : ℝ := 10

/-- The volume of the container as a function of the cut length -/
def container_volume (x : ℝ) : ℝ := (sheet_length - 2*x) * (sheet_width - 2*x) * x

theorem max_volume_at_10cm :
  ∀ x, 0 < x → x < sheet_width/2 → x < sheet_length/2 →
  container_volume x ≤ container_volume cut_length :=
sorry

end max_volume_at_10cm_l1431_143190


namespace equation_solutions_l1431_143167

theorem equation_solutions :
  (∀ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 →
    ¬(y.val = x.val + 1 ∧ z.val = y.val + 1 ∧ w.val = z.val + 1)) ∧
  (∃ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 ∧
    y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2 ∧
    Even x.val ∧ Even y.val ∧ Even z.val ∧ Even w.val) ∧
  (∀ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 →
    ¬(y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2 ∧
      Odd x.val ∧ Odd y.val ∧ Odd z.val ∧ Odd w.val)) ∧
  (∃ x y z w : ℕ+, x.val + y.val + z.val + w.val = 60 ∧
    y.val = x.val + 2 ∧ z.val = y.val + 2 ∧ w.val = z.val + 2) :=
by sorry

end equation_solutions_l1431_143167


namespace homecoming_ticket_sales_l1431_143111

theorem homecoming_ticket_sales
  (single_price : ℕ)
  (couple_price : ℕ)
  (total_attendance : ℕ)
  (couple_tickets_sold : ℕ)
  (h1 : single_price = 20)
  (h2 : couple_price = 35)
  (h3 : total_attendance = 128)
  (h4 : couple_tickets_sold = 16) :
  single_price * (total_attendance - 2 * couple_tickets_sold) +
  couple_price * couple_tickets_sold = 2480 := by
sorry


end homecoming_ticket_sales_l1431_143111


namespace irregular_polygon_rotation_implies_composite_l1431_143143

/-- An n-gon inscribed in a circle -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Rotation of a point around the origin by an angle -/
def rotate (p : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

/-- A polygon is irregular if not all sides have the same length -/
def irregular (P : Polygon n) : Prop := sorry

/-- A polygon coincides with itself after rotation -/
def coincides_after_rotation (P : Polygon n) (angle : ℝ) : Prop := sorry

/-- A number is composite if it's not prime and greater than 1 -/
def composite (n : ℕ) : Prop := ¬ Nat.Prime n ∧ n > 1

/-- Main theorem -/
theorem irregular_polygon_rotation_implies_composite
  (n : ℕ) (P : Polygon n) (α : ℝ) :
  irregular P →
  α ≠ 2 * Real.pi →
  coincides_after_rotation P α →
  composite n := by sorry

end irregular_polygon_rotation_implies_composite_l1431_143143


namespace exponent_problem_l1431_143100

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = 10) :
  x^(2*m - n) = 5/2 := by
  sorry

end exponent_problem_l1431_143100


namespace polynomial_simplification_l1431_143125

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + x^3 + 5) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 7) =
  x^6 - x^5 - x^4 + 2 * x^3 - 2 := by
  sorry

end polynomial_simplification_l1431_143125


namespace box_volume_l1431_143174

/-- A rectangular box with specific proportions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * width = 0.5 * (length * height)
  top_one_half_side : length * height = 1.5 * (width * height)
  side_area : width * height = 200

/-- The volume of a box is the product of its length, width, and height -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- Theorem stating that a box with the given proportions has a volume of 3000 -/
theorem box_volume (b : Box) : volume b = 3000 := by
  sorry

end box_volume_l1431_143174


namespace complement_union_equals_five_l1431_143147

def U : Set Nat := {1, 3, 5, 9}
def A : Set Nat := {1, 3, 9}
def B : Set Nat := {1, 9}

theorem complement_union_equals_five : (U \ (A ∪ B)) = {5} := by sorry

end complement_union_equals_five_l1431_143147


namespace equation_solution_l1431_143135

theorem equation_solution : ∃ k : ℝ, (5/9 * (k^2 - 32))^3 = 150 := by
  sorry

end equation_solution_l1431_143135


namespace stratified_sample_size_l1431_143128

theorem stratified_sample_size
  (ratio_A ratio_B ratio_C : ℕ)
  (sample_A : ℕ)
  (h_ratio : ratio_A = 3 ∧ ratio_B = 4 ∧ ratio_C = 7)
  (h_sample_A : sample_A = 15) :
  ∃ n : ℕ, n = sample_A * (ratio_A + ratio_B + ratio_C) / ratio_A ∧ n = 70 :=
by sorry

end stratified_sample_size_l1431_143128


namespace diana_remaining_paint_l1431_143144

/-- The amount of paint required for one statue in gallons -/
def paint_per_statue : ℚ := 1/16

/-- The number of statues Diana can paint with the remaining paint -/
def statues_to_paint : ℕ := 7

/-- The amount of paint Diana has remaining in gallons -/
def remaining_paint : ℚ := paint_per_statue * statues_to_paint

theorem diana_remaining_paint :
  remaining_paint = 7/16 := by sorry

end diana_remaining_paint_l1431_143144


namespace lauren_change_l1431_143126

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  meat_price : ℝ
  meat_weight : ℝ
  buns_price : ℝ
  lettuce_price : ℝ
  tomato_price : ℝ
  tomato_weight : ℝ
  pickles_price : ℝ
  pickle_coupon : ℝ

/-- Calculates the total cost of the grocery items --/
def total_cost (items : GroceryItems) : ℝ :=
  items.meat_price * items.meat_weight +
  items.buns_price +
  items.lettuce_price +
  items.tomato_price * items.tomato_weight +
  (items.pickles_price - items.pickle_coupon)

/-- Calculates the change from a given payment --/
def calculate_change (items : GroceryItems) (payment : ℝ) : ℝ :=
  payment - total_cost items

/-- Theorem stating that Lauren's change is $6.00 --/
theorem lauren_change :
  let items : GroceryItems := {
    meat_price := 3.5,
    meat_weight := 2,
    buns_price := 1.5,
    lettuce_price := 1,
    tomato_price := 2,
    tomato_weight := 1.5,
    pickles_price := 2.5,
    pickle_coupon := 1
  }
  calculate_change items 20 = 6 := by sorry

end lauren_change_l1431_143126


namespace translation_right_3_units_l1431_143194

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_3_units :
  let A : Point := { x := 1, y := 2 }
  let B : Point := translateRight A 3
  B = { x := 4, y := 2 } := by
  sorry

end translation_right_3_units_l1431_143194


namespace science_books_count_l1431_143139

theorem science_books_count (total : ℕ) (storybooks science picture dictionaries : ℕ) :
  total = 35 →
  total = storybooks + science + picture + dictionaries →
  storybooks + science = 17 →
  science + picture = 16 →
  storybooks ≠ science →
  storybooks ≠ picture →
  storybooks ≠ dictionaries →
  science ≠ picture →
  science ≠ dictionaries →
  picture ≠ dictionaries →
  (storybooks = 9 ∨ science = 9 ∨ picture = 9 ∨ dictionaries = 9) →
  science = 9 :=
by sorry

end science_books_count_l1431_143139


namespace white_towels_count_l1431_143137

def green_towels : ℕ := 35
def towels_given_away : ℕ := 34
def towels_remaining : ℕ := 22

theorem white_towels_count : ℕ := by
  sorry

end white_towels_count_l1431_143137


namespace initial_mixture_volume_l1431_143145

/-- Proves that given a mixture with 20% alcohol, if 3 liters of water are added
    and the resulting mixture has 17.14285714285715% alcohol, 
    then the initial amount of mixture was 18 liters. -/
theorem initial_mixture_volume (initial_volume : ℝ) : 
  initial_volume > 0 →
  (0.2 * initial_volume) / (initial_volume + 3) = 17.14285714285715 / 100 →
  initial_volume = 18 := by
sorry

end initial_mixture_volume_l1431_143145


namespace pascal_triangle_42nd_number_in_45_number_row_l1431_143188

theorem pascal_triangle_42nd_number_in_45_number_row : 
  let n : ℕ := 44  -- The row number (0-indexed) that contains 45 numbers
  let k : ℕ := 41  -- The position (0-indexed) of the 42nd number in the row
  (n.choose k) = 13254 := by sorry

end pascal_triangle_42nd_number_in_45_number_row_l1431_143188


namespace cos_difference_l1431_143179

theorem cos_difference (α : Real) (h : α = 2 * Real.pi / 3) :
  Real.cos (α + Real.pi / 2) - Real.cos (Real.pi + α) = -(Real.sqrt 3 + 1) / 2 := by
  sorry

end cos_difference_l1431_143179


namespace janet_action_figures_l1431_143120

/-- Calculates the final number of action figures Janet has -/
def final_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  let after_selling := initial - sold
  let after_buying := after_selling + bought
  let brothers_collection := 2 * after_buying
  after_buying + brothers_collection

/-- Proves that Janet ends up with 24 action figures given the initial conditions -/
theorem janet_action_figures :
  final_action_figures 10 6 4 = 24 := by
  sorry

end janet_action_figures_l1431_143120


namespace max_value_function_l1431_143178

theorem max_value_function (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 ≤ 14) → 
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 = 14) → 
  a = 1/3 ∨ a = 3 := by
sorry

end max_value_function_l1431_143178


namespace unique_solution_l1431_143192

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - (floor x : ℝ)

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  2 * (floor x : ℝ) * frac x = x^2 - 3/2 * x - 11/16

theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 9/4 :=
sorry

end unique_solution_l1431_143192


namespace maintenance_check_increase_l1431_143183

theorem maintenance_check_increase (original_time : ℝ) (increase_percentage : ℝ) (new_time : ℝ) :
  original_time = 25 →
  increase_percentage = 20 →
  new_time = original_time * (1 + increase_percentage / 100) →
  new_time = 30 := by
  sorry

end maintenance_check_increase_l1431_143183


namespace inequality_proof_l1431_143172

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 := by
  sorry

end inequality_proof_l1431_143172


namespace binomial_variance_10_07_l1431_143151

/-- The variance of a binomial distribution with 10 trials and 0.7 probability of success is 2.1 -/
theorem binomial_variance_10_07 :
  let n : ℕ := 10
  let p : ℝ := 0.7
  let variance := n * p * (1 - p)
  variance = 2.1 := by
  sorry

end binomial_variance_10_07_l1431_143151


namespace inequality_proof_l1431_143160

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z) / (x^3 + y^3 + x * y * z) +
  (x * y * z) / (y^3 + z^3 + x * y * z) +
  (x * y * z) / (z^3 + x^3 + x * y * z) ≤ 1 :=
by sorry

end inequality_proof_l1431_143160


namespace common_solution_y_value_l1431_143108

theorem common_solution_y_value : 
  ∃! y : ℝ, ∃ x : ℝ, (x^2 + y^2 - 4 = 0) ∧ (x^2 - 4*y + 8 = 0) ∧ (y = 2) :=
by sorry

end common_solution_y_value_l1431_143108


namespace parabola_perpendicular_point_range_l1431_143138

/-- Given points A, B, C where B and C are on a parabola and AB is perpendicular to BC,
    the y-coordinate of C satisfies y ≤ 0 or y ≥ 4 -/
theorem parabola_perpendicular_point_range 
  (A B C : ℝ × ℝ)
  (h_A : A = (0, 2))
  (h_B : B.1 = B.2^2 - 4)
  (h_C : C.1 = C.2^2 - 4)
  (h_perp : (B.2 - 2) * (C.2 - B.2) = -(B.1 - 0) * (C.1 - B.1)) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 :=
sorry

end parabola_perpendicular_point_range_l1431_143138


namespace selection_probabilities_correct_l1431_143184

/-- Given a group of 3 boys and 2 girls, this function calculates various probabilities
    when selecting two people from the group. -/
def selection_probabilities (num_boys : ℕ) (num_girls : ℕ) : ℚ × ℚ × ℚ :=
  let total := num_boys + num_girls
  let total_combinations := (total.choose 2 : ℚ)
  let two_boys := (num_boys.choose 2 : ℚ) / total_combinations
  let one_girl := (num_boys * num_girls : ℚ) / total_combinations
  let at_least_one_girl := 1 - two_boys
  (two_boys, one_girl, at_least_one_girl)

theorem selection_probabilities_correct :
  selection_probabilities 3 2 = (3/10, 3/5, 7/10) := by
  sorry

end selection_probabilities_correct_l1431_143184


namespace problem_solution_l1431_143149

def is_product_of_three_primes_less_than_10 (n : ℕ) : Prop :=
  ∃ p q r, p < 10 ∧ q < 10 ∧ r < 10 ∧ Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ n = p * q * r

def all_primes_less_than_10_present (a b : ℕ) : Prop :=
  ∀ p, p < 10 → Nat.Prime p → (p ∣ a ∨ p ∣ b)

theorem problem_solution (a b : ℕ) :
  is_product_of_three_primes_less_than_10 a ∧
  is_product_of_three_primes_less_than_10 b ∧
  all_primes_less_than_10_present a b ∧
  Nat.gcd a b = Nat.gcd (a / 15) b ∧
  Nat.gcd a b = 2 * Nat.gcd a (b / 4) →
  a = 30 ∧ b = 28 := by
  sorry

end problem_solution_l1431_143149


namespace largest_mersenne_prime_under_300_l1431_143131

def mersenne_number (n : ℕ) : ℕ := 2^n - 1

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ m = mersenne_number n ∧ Prime m

theorem largest_mersenne_prime_under_300 :
  ∀ m : ℕ, is_mersenne_prime m → m < 300 → m ≤ 127 :=
by sorry

end largest_mersenne_prime_under_300_l1431_143131


namespace souvenir_shop_optimal_solution_souvenir_shop_max_profit_l1431_143117

/-- Represents the cost and profit structure for souvenir types A and B -/
structure SouvenirShop where
  cost_A : ℝ
  cost_B : ℝ
  profit_A : ℝ
  profit_B : ℝ

/-- Theorem stating the optimal solution for the souvenir shop problem -/
theorem souvenir_shop_optimal_solution (shop : SouvenirShop) 
  (h1 : 7 * shop.cost_A + 8 * shop.cost_B = 380)
  (h2 : 10 * shop.cost_A + 6 * shop.cost_B = 380)
  (h3 : shop.profit_A = 5)
  (h4 : shop.profit_B = 7) : 
  (shop.cost_A = 20 ∧ shop.cost_B = 30) ∧ 
  (∀ a b : ℕ, a + b = 40 → a * shop.cost_A + b * shop.cost_B ≤ 900 → 
    a * shop.profit_A + b * shop.profit_B ≥ 216 → 
    a * shop.profit_A + b * shop.profit_B ≤ 30 * shop.profit_A + 10 * shop.profit_B) :=
sorry

/-- Corollary stating the maximum profit -/
theorem souvenir_shop_max_profit (shop : SouvenirShop) 
  (h : shop.cost_A = 20 ∧ shop.cost_B = 30 ∧ shop.profit_A = 5 ∧ shop.profit_B = 7) :
  30 * shop.profit_A + 10 * shop.profit_B = 220 :=
sorry

end souvenir_shop_optimal_solution_souvenir_shop_max_profit_l1431_143117


namespace largest_power_of_six_divisor_l1431_143106

theorem largest_power_of_six_divisor : 
  (∃ k : ℕ, 6^k ∣ (8 * 48 * 81) ∧ 
   ∀ m : ℕ, m > k → ¬(6^m ∣ (8 * 48 * 81))) → 
  (∃ k : ℕ, k = 5 ∧ 6^k ∣ (8 * 48 * 81) ∧ 
   ∀ m : ℕ, m > k → ¬(6^m ∣ (8 * 48 * 81))) := by
sorry

end largest_power_of_six_divisor_l1431_143106


namespace card_length_is_three_inches_l1431_143162

-- Define the poster board size in inches
def posterBoardSize : ℕ := 12

-- Define the width of the cards in inches
def cardWidth : ℕ := 2

-- Define the maximum number of cards that can be made
def maxCards : ℕ := 24

-- Theorem statement
theorem card_length_is_three_inches :
  ∀ (cardLength : ℕ),
    (posterBoardSize / cardWidth) * (posterBoardSize / cardLength) = maxCards →
    cardLength = 3 := by
  sorry

end card_length_is_three_inches_l1431_143162


namespace cistern_width_is_six_l1431_143103

/-- Represents the dimensions and properties of a rectangular cistern --/
structure Cistern where
  length : ℝ
  width : ℝ
  waterDepth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterDepth + 2 * c.width * c.waterDepth

/-- Theorem stating that a cistern with given dimensions has a width of 6 meters --/
theorem cistern_width_is_six (c : Cistern) 
  (h1 : c.length = 9)
  (h2 : c.waterDepth = 2.25)
  (h3 : c.wetSurfaceArea = 121.5)
  (h4 : totalWetSurfaceArea c = c.wetSurfaceArea) : 
  c.width = 6 := by
  sorry

end cistern_width_is_six_l1431_143103


namespace power_of_product_equality_l1431_143181

theorem power_of_product_equality (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end power_of_product_equality_l1431_143181


namespace tan_2x_value_l1431_143102

theorem tan_2x_value (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, f x = Real.sin x + Real.cos x)
  (h2 : ∀ x, deriv f x = 3 * f x) : 
  Real.tan (2 * x) = -4/3 := by
  sorry

end tan_2x_value_l1431_143102


namespace find_y_l1431_143170

theorem find_y : ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end find_y_l1431_143170


namespace fourth_quadrant_condition_l1431_143109

def complex_number (b : ℝ) : ℂ := (1 + b * Complex.I) * (2 + Complex.I)

def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

theorem fourth_quadrant_condition (b : ℝ) : 
  in_fourth_quadrant (complex_number b) ↔ b < -1/2 := by sorry

end fourth_quadrant_condition_l1431_143109


namespace new_person_weight_is_109_5_l1431_143105

/-- Calculates the weight of a new person in a group when the average weight changes --/
def newPersonWeight (numPersons : ℕ) (avgWeightIncrease : ℝ) (oldPersonWeight : ℝ) : ℝ :=
  oldPersonWeight + numPersons * avgWeightIncrease

/-- Theorem: The weight of the new person is 109.5 kg --/
theorem new_person_weight_is_109_5 :
  newPersonWeight 15 2.3 75 = 109.5 := by
  sorry

end new_person_weight_is_109_5_l1431_143105


namespace restaurant_students_l1431_143158

theorem restaurant_students (burger_count : ℕ) (hotdog_count : ℕ) :
  burger_count = 30 →
  burger_count = 2 * hotdog_count →
  burger_count + hotdog_count = 45 := by
  sorry

end restaurant_students_l1431_143158


namespace sqrt_problem_l1431_143187

theorem sqrt_problem : 
  (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2 = 0) ∧
  (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5 = 9 * Real.sqrt 6) := by
sorry

end sqrt_problem_l1431_143187


namespace inequality_proof_l1431_143175

theorem inequality_proof (a x : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hx : 0 ≤ x ∧ x ≤ π) :
  (2 * a - 1) * Real.sin x + (1 - a) * Real.sin ((1 - a) * x) ≥ 0 := by
  sorry

end inequality_proof_l1431_143175


namespace tommy_makes_twelve_loaves_l1431_143107

/-- Represents the number of loaves Tommy can make given the flour costs and his budget --/
def tommys_loaves (flour_per_loaf : ℝ) (small_bag_weight : ℝ) (small_bag_cost : ℝ) 
  (large_bag_weight : ℝ) (large_bag_cost : ℝ) (budget : ℝ) : ℕ :=
  sorry

/-- Theorem stating that Tommy can make 12 loaves of bread --/
theorem tommy_makes_twelve_loaves :
  tommys_loaves 4 10 10 12 13 50 = 12 := by
  sorry

end tommy_makes_twelve_loaves_l1431_143107


namespace conic_section_is_hyperbola_l1431_143168

/-- The equation (x-3)^2 = 3(2y+4)^2 - 75 represents a hyperbola -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (x - 3)^2 = 3*(2*y + 4)^2 - 75 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0) ∧
    a > 0 ∧ b < 0 := by
  sorry

end conic_section_is_hyperbola_l1431_143168


namespace combined_population_theorem_l1431_143198

/-- The combined population of New York and New England -/
def combined_population (new_england_population : ℕ) : ℕ :=
  new_england_population + (2 * new_england_population) / 3

/-- Theorem stating the combined population of New York and New England -/
theorem combined_population_theorem :
  combined_population 2100000 = 3500000 := by
  sorry

#eval combined_population 2100000

end combined_population_theorem_l1431_143198


namespace line_m_equation_line_n_equation_l1431_143146

-- Define the point A
def A : ℝ × ℝ := (-2, 1)

-- Define the line l
def l (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Define parallelism
def parallel (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l₂ (k * x) (k * y)

-- Define perpendicularity
def perpendicular (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ x y, l₁ x y ↔ l₂ y (-x)

-- Theorem for line m
theorem line_m_equation :
  ∃ (m : ℝ → ℝ → Prop),
    (m A.1 A.2) ∧
    (parallel l m) ∧
    (∀ x y, m x y ↔ 2 * x - y + 5 = 0) :=
sorry

-- Theorem for line n
theorem line_n_equation :
  ∃ (n : ℝ → ℝ → Prop),
    (n A.1 A.2) ∧
    (perpendicular l n) ∧
    (∀ x y, n x y ↔ x + 2 * y = 0) :=
sorry

end line_m_equation_line_n_equation_l1431_143146


namespace diminished_value_is_seven_l1431_143153

def smallest_number : ℕ := 1015

def divisors : List ℕ := [12, 16, 18, 21, 28]

theorem diminished_value_is_seven :
  ∃ (k : ℕ), k = 7 ∧
  ∀ d ∈ divisors, (smallest_number - k) % d = 0 ∧
  ∀ m < k, ∃ d ∈ divisors, (smallest_number - m) % d ≠ 0 :=
sorry

end diminished_value_is_seven_l1431_143153


namespace triangle_circumscribed_circle_radius_l1431_143127

theorem triangle_circumscribed_circle_radius 
  (α : Real) (a b : Real) (R : Real) : 
  α = π / 3 →  -- 60° in radians
  a = 6 → 
  b = 2 → 
  R = (2 * Real.sqrt 21) / 3 → 
  2 * R = (Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos α))) / (Real.sin α) := by
  sorry

end triangle_circumscribed_circle_radius_l1431_143127


namespace complement_intersection_l1431_143159

def A : Set ℕ := {2, 3, 4}
def B (a : ℕ) : Set ℕ := {a + 2, a}

theorem complement_intersection (a : ℕ) (h : A ∩ B a = B a) : (Aᶜ ∩ B a) = {3} := by
  sorry

end complement_intersection_l1431_143159


namespace monroe_family_children_l1431_143101

/-- Given the total number of granola bars, the number eaten by parents, and the number given to each child,
    calculate the number of children in the family. -/
def number_of_children (total_bars : ℕ) (eaten_by_parents : ℕ) (bars_per_child : ℕ) : ℕ :=
  (total_bars - eaten_by_parents) / bars_per_child

/-- Theorem stating that the number of children in Monroe's family is 6. -/
theorem monroe_family_children :
  number_of_children 200 80 20 = 6 := by
  sorry

end monroe_family_children_l1431_143101


namespace square_function_properties_l1431_143164

-- Define the function f(x) = x^2 on (0, +∞)
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem square_function_properties :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    (f (x₁ * x₂) = f x₁ * f x₂) ∧
    ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
    (f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2) :=
by sorry

end square_function_properties_l1431_143164


namespace fraction_transformation_l1431_143134

theorem fraction_transformation (x : ℚ) : 
  (3 + x) / (11 + x) = 5 / 9 → x = 7 := by
  sorry

end fraction_transformation_l1431_143134


namespace kyler_wins_one_game_l1431_143150

/-- Represents a player in the chess tournament -/
inductive Player : Type
| Peter : Player
| Emma : Player
| Kyler : Player

/-- Represents the outcome of a chess game -/
inductive Outcome : Type
| Win : Outcome
| Loss : Outcome
| Draw : Outcome

/-- The number of games each player played -/
def games_per_player : ℕ := 6

/-- The total number of game outcomes recorded -/
def total_outcomes : ℕ := 18

/-- Function to get the number of wins for a player -/
def wins (p : Player) : ℕ :=
  match p with
  | Player.Peter => 3
  | Player.Emma => 2
  | Player.Kyler => 0  -- We'll prove this is actually 1

/-- Function to get the number of losses for a player -/
def losses (p : Player) : ℕ :=
  match p with
  | Player.Peter => 2
  | Player.Emma => 2
  | Player.Kyler => 3

/-- Function to get the number of draws for a player -/
def draws (p : Player) : ℕ :=
  match p with
  | Player.Peter => 1
  | Player.Emma => 2
  | Player.Kyler => 2

theorem kyler_wins_one_game :
  wins Player.Kyler = 1 :=
by
  sorry


end kyler_wins_one_game_l1431_143150


namespace lucky_larry_problem_l1431_143157

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 2 → b = 5 → c = 3 → d = 4 → 
  (a + b - c - d * e = a + (b - (c - (d * e)))) → e = 0 := by
  sorry

end lucky_larry_problem_l1431_143157


namespace production_average_problem_l1431_143193

theorem production_average_problem (n : ℕ) : 
  (n * 50 + 105) / (n + 1) = 55 → n = 10 := by sorry

end production_average_problem_l1431_143193


namespace apps_added_l1431_143169

theorem apps_added (initial_apps final_apps : ℕ) 
  (h1 : initial_apps = 17) 
  (h2 : final_apps = 18) : 
  final_apps - initial_apps = 1 := by
  sorry

end apps_added_l1431_143169


namespace other_number_proof_l1431_143104

/-- Given two positive integers with specific HCF, LCM, and one known value, prove the other value -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 132 := by
  sorry

end other_number_proof_l1431_143104


namespace game_winner_parity_l1431_143124

/-- The game state representing the current rectangle -/
structure GameState where
  width : ℕ
  height : ℕ
  area : ℕ
  h_width : width > 1
  h_height : height > 1
  h_area : area = width * height

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The game rules and win condition -/
def game_rules (initial_state : GameState) : GameResult :=
  if initial_state.area % 2 = 1 then
    GameResult.FirstPlayerWins
  else
    GameResult.SecondPlayerWins

/-- The main theorem stating the winning condition based on initial area parity -/
theorem game_winner_parity (m n : ℕ) (h_m : m > 1) (h_n : n > 1) :
  let initial_state : GameState := {
    width := m,
    height := n,
    area := m * n,
    h_width := h_m,
    h_height := h_n,
    h_area := rfl
  }
  game_rules initial_state =
    if m * n % 2 = 1 then
      GameResult.FirstPlayerWins
    else
      GameResult.SecondPlayerWins :=
sorry

end game_winner_parity_l1431_143124


namespace rectangle_dimensions_l1431_143163

theorem rectangle_dimensions (x : ℝ) : 
  (x + 3) * (3 * x - 4) = 5 * x + 14 → x = Real.sqrt 78 / 3 := by
sorry

end rectangle_dimensions_l1431_143163


namespace line_equation_proof_l1431_143166

theorem line_equation_proof (m c : ℝ) (h1 : c ≠ 0) (h2 : m = 4 + 2 * Real.sqrt 7) (h3 : c = 2 - 2 * Real.sqrt 7) :
  ∃ k : ℝ, 
    (∀ k' : ℝ, k' ≠ k → 
      (abs ((k'^2 + 4*k' + 3) - (m*k' + c)) ≠ 7 ∨ 
       ¬∃ y1 y2 : ℝ, y1 = k'^2 + 4*k' + 3 ∧ y2 = m*k' + c ∧ y1 ≠ y2)) ∧
    (∃ y1 y2 : ℝ, y1 = k^2 + 4*k + 3 ∧ y2 = m*k + c ∧ y1 ≠ y2 ∧ abs (y1 - y2) = 7) ∧
    m * 1 + c = 6 := by
  sorry

end line_equation_proof_l1431_143166


namespace allocation_problem_l1431_143123

/-- The number of ways to allocate doctors and nurses to schools --/
def allocations (doctors nurses schools : ℕ) : ℕ :=
  (doctors.factorial * nurses.choose (2 * schools)) / (schools.factorial * (2 ^ schools))

/-- Theorem stating the number of allocations for the given problem --/
theorem allocation_problem :
  allocations 3 6 3 = 540 :=
by sorry

end allocation_problem_l1431_143123


namespace fred_card_purchase_l1431_143114

/-- The number of packs of football cards Fred bought -/
def football_packs : ℕ := 2

/-- The cost of one pack of football cards -/
def football_cost : ℚ := 273/100

/-- The cost of the pack of Pokemon cards -/
def pokemon_cost : ℚ := 401/100

/-- The cost of the deck of baseball cards -/
def baseball_cost : ℚ := 895/100

/-- The total amount Fred spent on cards -/
def total_spent : ℚ := 1842/100

theorem fred_card_purchase :
  (football_packs : ℚ) * football_cost + pokemon_cost + baseball_cost = total_spent :=
sorry

end fred_card_purchase_l1431_143114


namespace equation_solutions_l1431_143177

theorem equation_solutions :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → x + 1 ≠ 7 * (x - 1) - x^2) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → (x + 1 = x * (10 - x) - 7 ↔ x = 8 ∨ x = 1)) ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 ∧ x ≠ 4 → (x + 1 = x * (7 - x) + 1 ↔ x = 6)) :=
by sorry

end equation_solutions_l1431_143177


namespace no_special_numbers_l1431_143156

/-- A number is prime if it's greater than 1 and has no divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A number is composite if it has more than two factors -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

/-- A number is a perfect square if it's the square of an integer -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The set of integers from 1 to 1000 -/
def numberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 1000}

theorem no_special_numbers : ∀ n ∈ numberSet, isPrime n ∨ isComposite n ∨ isPerfectSquare n := by
  sorry

end no_special_numbers_l1431_143156


namespace parallel_planes_from_intersecting_parallel_lines_l1431_143110

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Define the property of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the property of two lines intersecting
variable (lines_intersect : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_intersecting_parallel_lines
  (α β : Plane) (l₁ l₂ : Line)
  (h1 : line_in_plane l₁ α)
  (h2 : line_in_plane l₂ α)
  (h3 : lines_intersect l₁ l₂)
  (h4 : line_parallel_to_plane l₁ β)
  (h5 : line_parallel_to_plane l₂ β) :
  plane_parallel α β :=
sorry

end parallel_planes_from_intersecting_parallel_lines_l1431_143110


namespace numerical_expression_problem_l1431_143140

theorem numerical_expression_problem :
  ∃ (A B C D : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    20180 ≤ 2018 * 10 + A ∧ 2018 * 10 + A < 20190 ∧
    100 ≤ B * 100 + C * 10 + D ∧ B * 100 + C * 10 + D < 1000 ∧
    (2018 * 10 + A) / (B * 100 + C * 10 + D) = 10 * A + A ∧
    A = 5 ∧ B = 3 ∧ C = 6 ∧ D = 7 :=
by sorry

end numerical_expression_problem_l1431_143140


namespace f_max_value_l1431_143152

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x^2)

theorem f_max_value :
  ∃ (x_max : ℝ), x_max > 0 ∧
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  x_max = Real.sqrt (Real.exp 1) ∧
  f x_max = 1 / (2 * Real.exp 1) :=
sorry

end f_max_value_l1431_143152


namespace coin_weight_verification_l1431_143176

theorem coin_weight_verification (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  x + y = 3 ∧ 
  x + (x + y) + (x + 2*y) = 9 ∧ 
  y + (x + y) + (x + 2*y) = x + 9 → 
  x = 1 ∧ y = 2 := by sorry

end coin_weight_verification_l1431_143176


namespace correct_reading_growth_equation_l1431_143199

/-- Represents the growth of average reading amount per student over 2 years -/
def reading_growth (x : ℝ) : Prop :=
  let initial_amount : ℝ := 1
  let final_amount : ℝ := 1.21
  let growth_period : ℕ := 2
  100 * (1 + x)^growth_period = 121

/-- Proves that the equation correctly represents the reading growth -/
theorem correct_reading_growth_equation :
  ∃ x : ℝ, reading_growth x ∧ x > 0 := by sorry

end correct_reading_growth_equation_l1431_143199


namespace number_wall_solution_l1431_143116

/-- Represents a number wall with four layers --/
structure NumberWall :=
  (bottom_row : Fin 4 → ℕ)
  (second_row : Fin 3 → ℕ)
  (third_row : Fin 2 → ℕ)
  (top : ℕ)

/-- Checks if a number wall follows the addition rule --/
def is_valid_wall (wall : NumberWall) : Prop :=
  (∀ i : Fin 3, wall.second_row i = wall.bottom_row i + wall.bottom_row (i + 1)) ∧
  (∀ i : Fin 2, wall.third_row i = wall.second_row i + wall.second_row (i + 1)) ∧
  (wall.top = wall.third_row 0 + wall.third_row 1)

/-- The theorem to be proved --/
theorem number_wall_solution (m : ℕ) : 
  (∃ wall : NumberWall, 
    wall.bottom_row = ![m, 4, 10, 9] ∧ 
    wall.top = 52 ∧ 
    is_valid_wall wall) → 
  m = 1 := by
  sorry

end number_wall_solution_l1431_143116


namespace eight_people_seating_theorem_l1431_143148

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seating_arrangements (n : ℕ) (restricted_pairs : ℕ) : ℕ :=
  factorial n - (2 * restricted_pairs * factorial (n - 1) * 2 - factorial (n - 2) * 4)

theorem eight_people_seating_theorem :
  seating_arrangements 8 2 = 23040 := by sorry

end eight_people_seating_theorem_l1431_143148


namespace product_multiple_in_consecutive_integers_l1431_143142

theorem product_multiple_in_consecutive_integers (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∃ (start : ℤ) (x y : ℤ), 
    x ≠ y ∧ 
    start ≤ x ∧ x < start + b ∧
    start ≤ y ∧ y < start + b ∧
    (x * y) % (a * b) = 0 :=
by sorry

end product_multiple_in_consecutive_integers_l1431_143142


namespace least_positive_integer_multiple_of_53_l1431_143119

theorem least_positive_integer_multiple_of_53 :
  ∀ x : ℕ+, x < 21 → ¬(∃ k : ℤ, (3*x)^2 + 2*43*3*x + 43^2 = 53*k) ∧
  ∃ k : ℤ, (3*21)^2 + 2*43*3*21 + 43^2 = 53*k :=
by sorry

end least_positive_integer_multiple_of_53_l1431_143119


namespace shopkeeper_profit_calculation_l1431_143155

theorem shopkeeper_profit_calculation 
  (C L S : ℝ)
  (h1 : L = C * (1 + intended_profit_percentage))
  (h2 : S = 0.9 * L)
  (h3 : S = 1.35 * C)
  : intended_profit_percentage = 0.5 :=
by sorry


end shopkeeper_profit_calculation_l1431_143155


namespace shipment_box_count_l1431_143132

theorem shipment_box_count :
  ∀ (x y : ℕ),
  (10 * x + 20 * y) / (x + y) = 18 →
  (10 * x + 20 * (y - 15)) / (x + y - 15) = 16 →
  x + y = 30 := by
sorry

end shipment_box_count_l1431_143132


namespace yard_trees_l1431_143113

/-- Calculates the number of trees in a yard given the yard length and distance between trees. -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating that in a 325-meter yard with trees 13 meters apart, there are 26 trees. -/
theorem yard_trees : num_trees 325 13 = 26 := by
  sorry

end yard_trees_l1431_143113


namespace bottles_left_on_shelf_prove_bottles_left_l1431_143186

/-- Calculates the number of bottles left on a shelf after two customers make purchases with specific discounts --/
theorem bottles_left_on_shelf (initial_bottles : ℕ) 
  (jason_bottles : ℕ) (harry_bottles : ℕ) : ℕ :=
  let jason_effective_bottles := jason_bottles
  let harry_effective_bottles := harry_bottles + 1
  initial_bottles - (jason_effective_bottles + harry_effective_bottles)

/-- Proves that given the specific conditions, there are 23 bottles left on the shelf --/
theorem prove_bottles_left : 
  bottles_left_on_shelf 35 5 6 = 23 := by
  sorry

end bottles_left_on_shelf_prove_bottles_left_l1431_143186


namespace negation_of_existence_negation_of_quadratic_inequality_l1431_143165

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1431_143165


namespace magic_square_d_plus_e_l1431_143196

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  sum : ℤ
  row1_sum : 30 + b + 22 = sum
  row2_sum : 19 + c + d = sum
  row3_sum : a + 28 + f = sum
  col1_sum : 30 + 19 + a = sum
  col2_sum : b + c + 28 = sum
  col3_sum : 22 + d + f = sum
  diag1_sum : 30 + c + f = sum
  diag2_sum : a + c + 22 = sum

/-- The sum of d and e in the magic square is 54 -/
theorem magic_square_d_plus_e (ms : MagicSquare) : ms.d + ms.e = 54 := by
  sorry

end magic_square_d_plus_e_l1431_143196


namespace sum_congruence_and_parity_l1431_143182

def sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruence_and_parity :
  (sum % 9 = 6) ∧ Even 6 := by sorry

end sum_congruence_and_parity_l1431_143182


namespace max_median_value_l1431_143130

theorem max_median_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  t = 20 →
  r ≤ 8 := by
sorry

end max_median_value_l1431_143130


namespace complement_union_A_B_l1431_143129

def U : Set ℕ := {0, 1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 7*x + 12 = 0}

def B : Set ℕ := {1, 3, 5}

theorem complement_union_A_B : (U \ (A ∪ B)) = {0, 2} := by
  sorry

end complement_union_A_B_l1431_143129


namespace no_real_solutions_l1431_143133

theorem no_real_solutions : ¬∃ y : ℝ, (y - 4*y + 10)^2 + 4 = -2*abs y := by
  sorry

end no_real_solutions_l1431_143133


namespace line_equation_proof_l1431_143189

def P : ℝ × ℝ := (1, 2)
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -5)

def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

def DistancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Definition of distance from a point to a line

theorem line_equation_proof :
  ∃ (l : Set (ℝ × ℝ)),
    P ∈ l ∧
    DistancePointToLine A l = DistancePointToLine B l ∧
    (l = Line 4 1 (-6) ∨ l = Line 3 2 (-7)) :=
by sorry

end line_equation_proof_l1431_143189


namespace sphere_volume_from_cube_surface_l1431_143112

theorem sphere_volume_from_cube_surface (cube_side : ℝ) (sphere_radius : ℝ) : 
  cube_side = 3 → 
  (6 * cube_side^2 : ℝ) = 4 * π * sphere_radius^2 → 
  (4 / 3 : ℝ) * π * sphere_radius^3 = 54 * Real.sqrt 6 / Real.sqrt π := by
  sorry

#check sphere_volume_from_cube_surface

end sphere_volume_from_cube_surface_l1431_143112


namespace total_cost_calculation_l1431_143161

def snake_toy_price : ℚ := 1176 / 100
def cage_price : ℚ := 1454 / 100
def heat_lamp_price : ℚ := 625 / 100
def cage_discount_rate : ℚ := 10 / 100
def sales_tax_rate : ℚ := 8 / 100
def found_money : ℚ := 1

def total_cost : ℚ :=
  let discounted_cage_price := cage_price * (1 - cage_discount_rate)
  let subtotal := snake_toy_price + discounted_cage_price + heat_lamp_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax - found_money

theorem total_cost_calculation :
  (total_cost * 100).floor / 100 = 3258 / 100 := by sorry

end total_cost_calculation_l1431_143161


namespace long_furred_brown_dogs_l1431_143136

theorem long_furred_brown_dogs 
  (total : ℕ) 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (h1 : total = 45) 
  (h2 : long_furred = 29) 
  (h3 : brown = 17) 
  (h4 : neither = 8) : 
  long_furred + brown - (total - neither) = 9 := by
sorry

end long_furred_brown_dogs_l1431_143136


namespace rectangle_area_increase_l1431_143195

theorem rectangle_area_increase :
  ∀ (l w : ℝ), l > 0 → w > 0 →
  let new_length := 1.25 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.4375 := by
sorry

end rectangle_area_increase_l1431_143195
