import Mathlib

namespace yellow_balls_count_l28_2815

theorem yellow_balls_count (total : ℕ) (red : ℕ) (prob_red : ℚ) : 
  red = 10 → prob_red = 1/3 → total - red = 20 :=
by
  sorry

end yellow_balls_count_l28_2815


namespace cos_eight_arccos_one_fifth_l28_2804

theorem cos_eight_arccos_one_fifth :
  Real.cos (8 * Real.arccos (1/5)) = -15647/390625 := by
  sorry

end cos_eight_arccos_one_fifth_l28_2804


namespace comic_book_arrangement_l28_2816

def arrange_comic_books (batman : Nat) (superman : Nat) (wonder_woman : Nat) (flash : Nat) : Nat :=
  (Nat.factorial batman) * (Nat.factorial superman) * (Nat.factorial wonder_woman) * (Nat.factorial flash) * (Nat.factorial 4)

theorem comic_book_arrangement :
  arrange_comic_books 8 7 6 5 = 421275894176000 := by
  sorry

end comic_book_arrangement_l28_2816


namespace at_least_one_non_negative_l28_2805

theorem at_least_one_non_negative (a b c d e f g h : ℝ) :
  (max (a*c + b*d) (max (a*e + b*f) (max (a*g + b*h) (max (c*e + d*f) (max (c*g + d*h) (e*g + f*h)))))) ≥ 0 := by
  sorry

end at_least_one_non_negative_l28_2805


namespace fraction_meaningful_condition_l28_2800

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, (x + 2) / (x - 1) = y) ↔ x ≠ 1 :=
by sorry

end fraction_meaningful_condition_l28_2800


namespace crayons_left_l28_2802

theorem crayons_left (initial_crayons : ℕ) (percentage_lost : ℚ) : 
  initial_crayons = 253 → 
  percentage_lost = 35.5 / 100 →
  ↑⌊initial_crayons - percentage_lost * initial_crayons⌋ = 163 := by
  sorry

end crayons_left_l28_2802


namespace distance_after_two_hours_l28_2866

-- Define the speeds and time
def alice_speed : ℚ := 1 / 12  -- miles per minute
def bob_speed : ℚ := 3 / 20    -- miles per minute
def duration : ℚ := 120        -- minutes (2 hours)

-- Theorem statement
theorem distance_after_two_hours :
  let alice_distance := alice_speed * duration
  let bob_distance := bob_speed * duration
  alice_distance + bob_distance = 28 := by
sorry


end distance_after_two_hours_l28_2866


namespace warren_guests_proof_l28_2897

/-- The number of tables Warren has -/
def num_tables : ℝ := 252.0

/-- The number of guests each table can hold -/
def guests_per_table : ℝ := 4.0

/-- The total number of guests Warren can accommodate -/
def total_guests : ℝ := num_tables * guests_per_table

theorem warren_guests_proof : total_guests = 1008.0 := by
  sorry

end warren_guests_proof_l28_2897


namespace stomachion_gray_area_l28_2877

/-- A square with side length 12 cm divided into 14 polygons -/
structure StomachionPuzzle where
  side_length : ℝ
  num_polygons : ℕ
  h_side : side_length = 12
  h_polygons : num_polygons = 14

/-- A quadrilateral in the Stomachion puzzle -/
structure Quadrilateral (puzzle : StomachionPuzzle) where
  area : ℝ

/-- There exists a quadrilateral in the Stomachion puzzle with an area of 12 cm² -/
theorem stomachion_gray_area (puzzle : StomachionPuzzle) :
  ∃ (q : Quadrilateral puzzle), q.area = 12 := by
  sorry

end stomachion_gray_area_l28_2877


namespace carnival_snack_booth_sales_ratio_l28_2849

-- Define the constants from the problem
def daily_popcorn_sales : ℚ := 50
def num_days : ℕ := 5
def rent : ℚ := 30
def ingredient_cost : ℚ := 75
def total_earnings : ℚ := 895

-- Define the theorem
theorem carnival_snack_booth_sales_ratio :
  ∃ (daily_cotton_candy_sales : ℚ),
    (daily_cotton_candy_sales * num_days + daily_popcorn_sales * num_days - (rent + ingredient_cost) = total_earnings) ∧
    (daily_cotton_candy_sales / daily_popcorn_sales = 3 / 1) := by
  sorry

end carnival_snack_booth_sales_ratio_l28_2849


namespace opposite_of_negative_2023_l28_2880

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l28_2880


namespace square_of_arithmetic_mean_le_arithmetic_mean_of_squares_l28_2876

theorem square_of_arithmetic_mean_le_arithmetic_mean_of_squares
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b) / 2) ^ 2 ≤ (a ^ 2 + b ^ 2) / 2 := by
  sorry

end square_of_arithmetic_mean_le_arithmetic_mean_of_squares_l28_2876


namespace complex_equation_solution_l28_2879

theorem complex_equation_solution (x y : ℕ+) 
  (h : (x - Complex.I * y) ^ 2 = 15 - 20 * Complex.I) : 
  x - Complex.I * y = 5 - 2 * Complex.I :=
by sorry

end complex_equation_solution_l28_2879


namespace zucchini_weight_l28_2855

/-- Proves that the weight of zucchini installed is 13 kg -/
theorem zucchini_weight (carrots broccoli half_sold : ℝ) (h1 : carrots = 15) (h2 : broccoli = 8) (h3 : half_sold = 18) :
  ∃ zucchini : ℝ, (carrots + zucchini + broccoli) / 2 = half_sold ∧ zucchini = 13 := by
  sorry

end zucchini_weight_l28_2855


namespace inequalities_hold_l28_2899

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a) (h2 : y^2 < b) (h3 : z^2 < c) : 
  (x^2*y^2 + y^2*z^2 + z^2*x^2 < a*b + b*c + c*a) ∧ 
  (x^4 + y^4 + z^4 < a^2 + b^2 + c^2) ∧ 
  (x^2*y^2*z^2 < a*b*c) := by
  sorry

end inequalities_hold_l28_2899


namespace white_marbles_count_l28_2893

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (white : ℕ) 
  (h1 : total = 20)
  (h2 : blue = 5)
  (h3 : red = 7)
  (h4 : total = blue + red + white)
  (h5 : (red + white : ℚ) / total = 3/4) : 
  white = 8 := by
sorry

end white_marbles_count_l28_2893


namespace fruit_basket_count_l28_2819

/-- The number of fruit baskets -/
def num_baskets : ℕ := 4

/-- The number of apples in each of the first three baskets -/
def apples_per_basket : ℕ := 9

/-- The number of oranges in each of the first three baskets -/
def oranges_per_basket : ℕ := 15

/-- The number of bananas in each of the first three baskets -/
def bananas_per_basket : ℕ := 14

/-- The number of fruits reduced in the fourth basket -/
def reduction : ℕ := 2

/-- The total number of fruits in all baskets -/
def total_fruits : ℕ := 146

theorem fruit_basket_count :
  (3 * (apples_per_basket + oranges_per_basket + bananas_per_basket)) +
  ((apples_per_basket - reduction) + (oranges_per_basket - reduction) + (bananas_per_basket - reduction)) = total_fruits :=
by sorry

end fruit_basket_count_l28_2819


namespace root_product_equals_27_l28_2835

theorem root_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end root_product_equals_27_l28_2835


namespace line_ellipse_intersection_slope_condition_l28_2839

/-- The slope of a line intersecting an ellipse satisfies a certain condition -/
theorem line_ellipse_intersection_slope_condition 
  (m : ℝ) -- slope of the line
  (h : ∃ (x y : ℝ), y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100) -- line intersects ellipse
  : m^2 ≥ 1/624 := by
  sorry

#check line_ellipse_intersection_slope_condition

end line_ellipse_intersection_slope_condition_l28_2839


namespace two_propositions_are_true_l28_2823

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel : Line → Line → Prop := sorry
def perpendicular : Line → Line → Prop := sorry
def planeParallel : Plane → Plane → Prop := sorry
def planePerpendicular : Plane → Plane → Prop := sorry
def lineParallelToPlane : Line → Plane → Prop := sorry
def linePerpendicularToPlane : Line → Plane → Prop := sorry

-- Define the propositions
def prop1 (α β : Plane) (c : Line) : Prop :=
  planeParallel α β ∧ linePerpendicularToPlane c α → linePerpendicularToPlane c β

def prop2 (α : Plane) (b : Line) (γ : Plane) : Prop :=
  lineParallelToPlane b α ∧ planePerpendicular α γ → linePerpendicularToPlane b γ

def prop3 (a : Line) (β γ : Plane) : Prop :=
  lineParallelToPlane a β ∧ linePerpendicularToPlane a γ → planePerpendicular β γ

-- The main theorem
theorem two_propositions_are_true :
  ∃ (α β γ : Plane) (a b c : Line),
    (prop1 α β c ∧ prop3 a β γ) ∧ ¬prop2 α b γ :=
sorry

end two_propositions_are_true_l28_2823


namespace smallest_solution_absolute_value_equation_l28_2885

theorem smallest_solution_absolute_value_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  ∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y ∧
  x = -2 :=
by sorry

end smallest_solution_absolute_value_equation_l28_2885


namespace rabbit_speed_l28_2854

def rabbit_speed_equation (x : ℝ) : Prop :=
  2 * (2 * x + 4) = 188

theorem rabbit_speed : ∃ (x : ℝ), rabbit_speed_equation x ∧ x = 45 := by
  sorry

end rabbit_speed_l28_2854


namespace sin_difference_quotient_zero_l28_2840

theorem sin_difference_quotient_zero (x y : ℝ) 
  (hx : Real.tan x = x) 
  (hy : Real.tan y = y) 
  (hxy : |x| ≠ |y|) : 
  (Real.sin (x + y)) / (x + y) - (Real.sin (x - y)) / (x - y) = 0 := by
  sorry

end sin_difference_quotient_zero_l28_2840


namespace clubsuit_computation_l28_2827

-- Define the ♣ operation
def clubsuit (a c b : ℚ) : ℚ := (2 * a + c) / b

-- State the theorem
theorem clubsuit_computation : 
  clubsuit (clubsuit 6 1 (clubsuit 4 2 3)) 2 2 = 49 / 10 := by
  sorry

end clubsuit_computation_l28_2827


namespace vw_toyota_ratio_l28_2832

/-- The number of Dodge trucks in the parking lot -/
def dodge_trucks : ℕ := 60

/-- The number of Volkswagen Bugs in the parking lot -/
def vw_bugs : ℕ := 5

/-- The number of Ford trucks in the parking lot -/
def ford_trucks : ℕ := dodge_trucks / 3

/-- The number of Toyota trucks in the parking lot -/
def toyota_trucks : ℕ := ford_trucks / 2

/-- The ratio of Volkswagen Bugs to Toyota trucks is 1:2 -/
theorem vw_toyota_ratio : 
  (vw_bugs : ℚ) / toyota_trucks = 1 / 2 := by sorry

end vw_toyota_ratio_l28_2832


namespace no_seven_edge_polyhedron_l28_2843

/-- A polyhedron is a structure with vertices, edges, and faces. -/
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces

/-- Euler's formula for polyhedra -/
axiom euler_formula (p : Polyhedron) : p.V - p.E + p.F = 2

/-- Each vertex in a polyhedron has at least 3 edges -/
axiom vertex_edge_count (p : Polyhedron) : p.E * 2 ≥ p.V * 3

/-- A polyhedron must have at least 4 vertices -/
axiom min_vertices (p : Polyhedron) : p.V ≥ 4

/-- Theorem: No polyhedron can have exactly 7 edges -/
theorem no_seven_edge_polyhedron :
  ¬∃ (p : Polyhedron), p.E = 7 := by sorry

end no_seven_edge_polyhedron_l28_2843


namespace box_third_dimension_l28_2857

/-- Proves that the third dimension of a rectangular box is 6 cm, given specific conditions -/
theorem box_third_dimension (num_cubes : ℕ) (cube_volume : ℝ) (length width : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  length = 9 →
  width = 12 →
  (num_cubes : ℝ) * cube_volume = length * width * 6 :=
by sorry

end box_third_dimension_l28_2857


namespace infinitely_many_primes_l28_2828

theorem infinitely_many_primes : ∀ S : Finset Nat, (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end infinitely_many_primes_l28_2828


namespace complex_fraction_imaginary_l28_2814

theorem complex_fraction_imaginary (a : ℝ) : 
  (∃ (b : ℝ), b ≠ 0 ∧ (a - Complex.I) / (2 + Complex.I) = Complex.I * b) → a = 1/2 := by
  sorry

end complex_fraction_imaginary_l28_2814


namespace sales_tax_difference_l28_2852

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) 
  (h1 : price = 50)
  (h2 : tax_rate1 = 0.0725)
  (h3 : tax_rate2 = 0.0675) : 
  (tax_rate1 - tax_rate2) * price = 0.25 := by
  sorry

end sales_tax_difference_l28_2852


namespace f_properties_l28_2870

noncomputable def f (x : ℝ) : ℝ := x - Real.log x - 1

theorem f_properties :
  (∀ x > 0, f x ≥ 0) ∧
  (∀ p : ℝ, (∀ x ≥ 1, f (1/x) ≥ (Real.log x)^2 / (p + Real.log x)) ↔ p ≥ 2) := by
  sorry

end f_properties_l28_2870


namespace trig_expression_equals_half_l28_2889

theorem trig_expression_equals_half : 
  Real.sin (π / 3) - Real.sqrt 3 * Real.cos (π / 3) + (1 / 2) * Real.tan (π / 4) = 1 / 2 := by
  sorry

end trig_expression_equals_half_l28_2889


namespace max_profit_at_110_unique_max_profit_at_110_l28_2881

/-- Represents the profit function for a new energy company -/
def profit (x : ℕ+) : ℚ :=
  if x < 100 then
    -1/2 * x^2 + 90 * x - 600
  else
    -2 * x - 24200 / x + 4100

/-- Theorem stating the maximum profit occurs at x = 110 -/
theorem max_profit_at_110 :
  ∀ x : ℕ+, profit x ≤ profit 110 ∧ profit 110 = 3660 := by
  sorry

/-- Theorem stating that 110 is the unique maximizer of the profit function -/
theorem unique_max_profit_at_110 :
  ∀ x : ℕ+, x ≠ 110 → profit x < profit 110 := by
  sorry

end max_profit_at_110_unique_max_profit_at_110_l28_2881


namespace arithmetic_sequence_formula_l28_2825

/-- An arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + 2 * (n - 1)

/-- The general formula for the n-th term of the arithmetic sequence -/
theorem arithmetic_sequence_formula (n : ℕ) :
  arithmetic_sequence n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_formula_l28_2825


namespace old_refrigerator_cost_proof_l28_2896

/-- The daily cost of Kurt's new refrigerator in dollars -/
def new_refrigerator_cost : ℝ := 0.45

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The amount Kurt saves in a month with his new refrigerator in dollars -/
def monthly_savings : ℝ := 12

/-- The daily cost of Kurt's old refrigerator in dollars -/
def old_refrigerator_cost : ℝ := 0.85

theorem old_refrigerator_cost_proof : 
  old_refrigerator_cost * days_in_month - new_refrigerator_cost * days_in_month = monthly_savings :=
sorry

end old_refrigerator_cost_proof_l28_2896


namespace clown_mobiles_count_l28_2856

theorem clown_mobiles_count (clowns_per_mobile : ℕ) (total_clowns : ℕ) (mobiles_count : ℕ) : 
  clowns_per_mobile = 28 → 
  total_clowns = 140 → 
  mobiles_count * clowns_per_mobile = total_clowns →
  mobiles_count = 5 :=
by sorry

end clown_mobiles_count_l28_2856


namespace g_value_at_pi_over_4_l28_2891

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - Real.sqrt 3 * (sin x)^2

noncomputable def g (x : ℝ) : ℝ := f (x - π / 12) + Real.sqrt 3 / 2

theorem g_value_at_pi_over_4 : g (π / 4) = Real.sqrt 3 / 2 := by
  sorry

end g_value_at_pi_over_4_l28_2891


namespace quadratic_with_real_roots_l28_2837

/-- 
Given a quadratic equation with complex coefficients that has real roots, 
prove that the value of the real parameter m is 1/12.
-/
theorem quadratic_with_real_roots (i : ℂ) :
  (∃ x : ℝ, x^2 - (2*i - 1)*x + 3*m - i = 0) → m = 1/12 :=
by
  sorry

end quadratic_with_real_roots_l28_2837


namespace result_of_operation_l28_2829

theorem result_of_operation (n : ℕ) (h : n = 95) : (n / 5 + 23 : ℚ) = 42 := by
  sorry

end result_of_operation_l28_2829


namespace circumcircle_radius_of_triangle_l28_2810

theorem circumcircle_radius_of_triangle (a b c : ℚ) :
  a = 15/2 ∧ b = 10 ∧ c = 25/2 →
  a^2 + b^2 = c^2 →
  (c/2 : ℚ) = 25/4 := by
  sorry

end circumcircle_radius_of_triangle_l28_2810


namespace product_of_numbers_with_given_sum_and_difference_l28_2874

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 ∧ x - y = 10 → x * y = 875 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l28_2874


namespace decimal_equivalent_one_tenth_squared_l28_2853

theorem decimal_equivalent_one_tenth_squared : (1 / 10 : ℚ) ^ 2 = 0.01 := by
  sorry

end decimal_equivalent_one_tenth_squared_l28_2853


namespace square_difference_l28_2872

theorem square_difference (x a : ℝ) : (2*x + a)^2 - (2*x - a)^2 = 8*a*x := by
  sorry

end square_difference_l28_2872


namespace chris_earnings_june_l28_2817

/-- Chris's earnings for the first two weeks of June --/
def chrisEarnings (hoursWeek1 hoursWeek2 : ℕ) (extraEarnings : ℚ) : ℚ :=
  let hourlyWage := extraEarnings / (hoursWeek2 - hoursWeek1)
  hourlyWage * (hoursWeek1 + hoursWeek2)

/-- Theorem stating Chris's earnings for the first two weeks of June --/
theorem chris_earnings_june :
  chrisEarnings 18 30 (65.40 : ℚ) = (261.60 : ℚ) := by
  sorry

#eval chrisEarnings 18 30 (65.40 : ℚ)

end chris_earnings_june_l28_2817


namespace exists_same_color_neighbors_l28_2858

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- Represents a position in the grid -/
structure Position where
  x : Fin 50
  y : Fin 50

/-- Represents the coloring of the grid -/
def GridColoring := Position → Color

/-- Checks if a position is valid in the 50x50 grid -/
def isValidPosition (p : Position) : Prop :=
  p.x < 50 ∧ p.y < 50

/-- Gets the color of a cell at a given position -/
def getColor (g : GridColoring) (p : Position) : Color :=
  g p

/-- Checks if a cell has the same color as its four adjacent cells -/
def hasSameColorNeighbors (g : GridColoring) (p : Position) : Prop :=
  isValidPosition p ∧
  isValidPosition ⟨p.x - 1, p.y⟩ ∧
  isValidPosition ⟨p.x + 1, p.y⟩ ∧
  isValidPosition ⟨p.x, p.y - 1⟩ ∧
  isValidPosition ⟨p.x, p.y + 1⟩ ∧
  getColor g p = getColor g ⟨p.x - 1, p.y⟩ ∧
  getColor g p = getColor g ⟨p.x + 1, p.y⟩ ∧
  getColor g p = getColor g ⟨p.x, p.y - 1⟩ ∧
  getColor g p = getColor g ⟨p.x, p.y + 1⟩

/-- Theorem: There exists a cell with four cells on its sides of the same color -/
theorem exists_same_color_neighbors :
  ∀ (g : GridColoring), ∃ (p : Position), hasSameColorNeighbors g p :=
by sorry

end exists_same_color_neighbors_l28_2858


namespace adjacent_probability_l28_2824

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a type for arrangements
def Arrangement := List Person

-- Define a function to check if A and B are adjacent in an arrangement
def areAdjacent (arr : Arrangement) : Prop :=
  ∃ i, (arr.get? i = some Person.A ∧ arr.get? (i+1) = some Person.B) ∨
       (arr.get? i = some Person.B ∧ arr.get? (i+1) = some Person.A)

-- Define the set of all possible arrangements
def allArrangements : Finset Arrangement :=
  sorry

-- Define the set of arrangements where A and B are adjacent
def adjacentArrangements : Finset Arrangement :=
  sorry

-- State the theorem
theorem adjacent_probability :
  (adjacentArrangements.card : ℚ) / (allArrangements.card : ℚ) = 2 / 3 :=
sorry

end adjacent_probability_l28_2824


namespace inequality_proof_l28_2836

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end inequality_proof_l28_2836


namespace solution_set_of_inequality_l28_2803

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x, f x + (deriv f) x > 1) (h2 : f 0 = 4) :
  {x : ℝ | f x > 3 / Real.exp x + 1} = {x : ℝ | x > 0} := by sorry

end solution_set_of_inequality_l28_2803


namespace polygon_angle_sum_l28_2878

theorem polygon_angle_sum (n : ℕ) : (n - 2) * 180 = 2 * 360 ↔ n = 6 := by sorry

end polygon_angle_sum_l28_2878


namespace min_formula_l28_2822

theorem min_formula (a b : ℝ) : min a b = (a + b - Real.sqrt ((a - b)^2)) / 2 := by
  sorry

end min_formula_l28_2822


namespace trouser_cost_calculation_final_cost_is_correct_l28_2860

/-- Calculate the final cost in GBP for three trousers with given prices, discounts, taxes, and fees -/
theorem trouser_cost_calculation (price1 price2 price3 : ℝ) 
  (discount1 discount2 discount3 : ℝ) (global_discount : ℝ) 
  (sales_tax handling_fee conversion_rate : ℝ) : ℝ :=
  let discounted_price1 := price1 * (1 - discount1)
  let discounted_price2 := price2 * (1 - discount2)
  let discounted_price3 := price3 * (1 - discount3)
  let total_discounted := discounted_price1 + discounted_price2 + discounted_price3
  let after_global_discount := total_discounted * (1 - global_discount)
  let after_tax := after_global_discount * (1 + sales_tax)
  let final_usd := after_tax + 3 * handling_fee
  let final_gbp := final_usd * conversion_rate
  final_gbp

/-- The final cost in GBP for the given trouser prices and conditions is £271.87 -/
theorem final_cost_is_correct : 
  trouser_cost_calculation 100 150 200 0.20 0.15 0.25 0.10 0.08 5 0.75 = 271.87 := by
  sorry


end trouser_cost_calculation_final_cost_is_correct_l28_2860


namespace expression_equals_159_l28_2887

def numerator : List ℕ := [12, 24, 36, 48, 60]
def denominator : List ℕ := [6, 18, 30, 42, 54]

def term (x : ℕ) : ℕ := x^4 + 375

def expression : ℚ :=
  (numerator.map term).prod / (denominator.map term).prod

theorem expression_equals_159 : expression = 159 := by sorry

end expression_equals_159_l28_2887


namespace f_properties_l28_2851

/-- The function f(x) = x^3 - ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

/-- Theorem stating the range of a and the fixed point property -/
theorem f_properties (a : ℝ) (x₀ : ℝ) 
  (ha : a > 0)
  (hf : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y)
  (hx₀ : x₀ ≥ 1)
  (hfx₀ : f a x₀ ≥ 1)
  (hffx₀ : f a (f a x₀) = x₀) :
  (0 < a ∧ a ≤ 3) ∧ f a x₀ = x₀ := by
  sorry

end f_properties_l28_2851


namespace car_speed_adjustment_l28_2830

/-- Given a car traveling a fixed distance D at 2 mph for T hours,
    prove that to cover the same distance in 5.0 hours, its speed S should be (2T)/5 mph. -/
theorem car_speed_adjustment (T : ℝ) (h : T > 0) : 
  let D := 2 * T  -- Distance covered at 2 mph for T hours
  let S := (2 * T) / 5  -- New speed to cover the same distance in 5 hours
  D = S * 5 := by sorry

end car_speed_adjustment_l28_2830


namespace necessary_but_not_sufficient_l28_2844

-- Define the propositions
def proposition_A (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

def proposition_B (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  (∃ a : ℝ, proposition_A a ∧ ¬proposition_B a) :=
by sorry

end necessary_but_not_sufficient_l28_2844


namespace max_value_implies_a_f_leq_g_implies_a_range_l28_2808

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + 2 * a * x + 1
def g (x : ℝ) : ℝ := x * (Real.exp x + 1)

-- Part (Ⅰ)
theorem max_value_implies_a (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y) ∧ (∃ x > 0, f a x = 0) →
  a = -1/2 :=
sorry

-- Part (Ⅱ)
theorem f_leq_g_implies_a_range (a : ℝ) :
  (∀ x > 0, f a x ≤ g x) →
  a ≤ 1 :=
sorry

end max_value_implies_a_f_leq_g_implies_a_range_l28_2808


namespace nancy_garden_seeds_l28_2847

theorem nancy_garden_seeds (total_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : big_garden_seeds = 28)
  (h3 : small_gardens = 6)
  (h4 : big_garden_seeds ≤ total_seeds) :
  (total_seeds - big_garden_seeds) / small_gardens = 4 := by
  sorry

end nancy_garden_seeds_l28_2847


namespace lance_cents_l28_2833

/-- Represents the amount of cents each person has -/
structure Cents where
  lance : ℕ
  margaret : ℕ
  guy : ℕ
  bill : ℕ

/-- The problem statement -/
theorem lance_cents (c : Cents) : 
  c.margaret = 75 → -- Margaret has three-fourths of a dollar (75 cents)
  c.guy = 60 → -- Guy has two quarters (50 cents) and a dime (10 cents)
  c.bill = 60 → -- Bill has six dimes (6 * 10 cents)
  c.lance + c.margaret + c.guy + c.bill = 265 → -- Total combined cents
  c.lance = 70 := by
  sorry


end lance_cents_l28_2833


namespace pi_fourth_in_range_of_g_l28_2868

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - x) / (2 + x))

theorem pi_fourth_in_range_of_g : ∃ (x : ℝ), g x = π / 4 := by sorry

end pi_fourth_in_range_of_g_l28_2868


namespace alan_market_spend_l28_2864

/-- The total amount spent by Alan at the market -/
def total_spent (num_eggs : ℕ) (price_per_egg : ℕ) (num_chickens : ℕ) (price_per_chicken : ℕ) : ℕ :=
  num_eggs * price_per_egg + num_chickens * price_per_chicken

/-- Theorem: Alan spent $88 at the market -/
theorem alan_market_spend :
  total_spent 20 2 6 8 = 88 := by
  sorry

end alan_market_spend_l28_2864


namespace factor_sum_l28_2892

/-- If x^2 + 2√2x + 5 is a factor of x^4 + Px^2 + Q, then P + Q = 27 -/
theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (x^2 + 2 * Real.sqrt 2 * x + 5) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  P + Q = 27 := by
  sorry

end factor_sum_l28_2892


namespace exists_n_f_div_g_eq_2012_l28_2809

/-- The number of divisors of n which are perfect squares -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n which are perfect cubes -/
def g (n : ℕ+) : ℕ := sorry

/-- There exists a positive integer n such that f(n) / g(n) = 2012 -/
theorem exists_n_f_div_g_eq_2012 : ∃ n : ℕ+, (f n : ℚ) / (g n : ℚ) = 2012 := by
  sorry

end exists_n_f_div_g_eq_2012_l28_2809


namespace arc_measure_constant_l28_2894

/-- A right isosceles triangle with a rotating circle -/
structure RightIsoscelesWithCircle where
  -- The side length of the right isosceles triangle
  s : ℝ
  -- Ensure s is positive
  s_pos : 0 < s

/-- The measure of arc MBM' in degrees -/
def arcMeasure (t : RightIsoscelesWithCircle) : ℝ := 180

/-- Theorem: The arc MBM' always measures 180° -/
theorem arc_measure_constant (t : RightIsoscelesWithCircle) :
  arcMeasure t = 180 := by
  sorry

end arc_measure_constant_l28_2894


namespace correct_calculation_l28_2821

theorem correct_calculation (x : ℝ) : x / 12 = 8 → x * 12 = 1152 := by
  sorry

end correct_calculation_l28_2821


namespace solution_set_equality_l28_2859

open Set

def S : Set ℝ := {x | |x + 1| + |x - 4| ≥ 7}

theorem solution_set_equality : S = Iic (-2) ∪ Ici 5 := by sorry

end solution_set_equality_l28_2859


namespace zero_function_inequality_l28_2807

theorem zero_function_inequality (f : ℝ → ℝ) :
  (∀ (x y : ℝ), x ≠ 0 → f (x^2 + y) ≥ (1/x + 1) * f y) →
  ∀ x, f x = 0 := by
sorry

end zero_function_inequality_l28_2807


namespace arithmetic_progression_sum_l28_2806

theorem arithmetic_progression_sum (x y z d k : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  y * (z - x) - x * (y - z) = d ∧
  z * (x - y) - y * (z - x) = d ∧
  x * (y - z) + y * (z - x) + z * (x - y) = k
  → d = k / 3 := by sorry

end arithmetic_progression_sum_l28_2806


namespace quadratic_no_real_roots_l28_2834

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 4*x + 5 ≠ 0 := by
  sorry

end quadratic_no_real_roots_l28_2834


namespace apples_for_pies_l28_2886

/-- Calculates the number of apples needed to make a given number of pies. -/
def apples_needed (apples_per_pie : ℝ) (num_pies : ℕ) : ℝ :=
  apples_per_pie * (num_pies : ℝ)

/-- Theorem stating that 504 apples are needed to make 126 pies,
    given that it takes 4.0 apples to make 1.0 pie. -/
theorem apples_for_pies :
  apples_needed 4.0 126 = 504 := by
  sorry

end apples_for_pies_l28_2886


namespace pages_used_l28_2842

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def old_cards : ℕ := 16

theorem pages_used (total_cards : ℕ) (h : total_cards = new_cards + old_cards) :
  total_cards / cards_per_page = 8 :=
sorry

end pages_used_l28_2842


namespace product_sum_inequality_l28_2883

theorem product_sum_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := by
  sorry

end product_sum_inequality_l28_2883


namespace sqrt_expression_equality_l28_2888

theorem sqrt_expression_equality : 
  Real.sqrt 6 * (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6) - abs (3 * Real.sqrt 2 - 6) = 2 * Real.sqrt 3 := by
  sorry

end sqrt_expression_equality_l28_2888


namespace largest_number_l28_2875

theorem largest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -1) (hc : c = -2) (hd : d = 1) :
  d = max a (max b (max c d)) :=
by sorry

end largest_number_l28_2875


namespace part_one_part_two_l28_2838

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := -x^2 + 4*a*x - 3*a^2

-- Define the set q
def q : Set ℝ := {x | -x^2 + 11*x - 18 ≥ 0}

-- Part 1
theorem part_one : 
  {x : ℝ | f x 1 > 0} ∩ q = Set.Icc 2 3 := by sorry

-- Part 2
theorem part_two : 
  {a : ℝ | a > 0 ∧ ∀ x, f x a > 0 → x ∈ Set.Ioo 2 9} = Set.Icc 2 3 := by sorry

end part_one_part_two_l28_2838


namespace a_8_equals_3_l28_2869

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def IsArithmeticSequence (b : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d

theorem a_8_equals_3
  (a b : Sequence)
  (h1 : a 1 = 3)
  (h2 : IsArithmeticSequence b)
  (h3 : ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n)
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 8 = 3 := by
  sorry

end a_8_equals_3_l28_2869


namespace y_value_at_243_l28_2863

-- Define the function y in terms of k and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/5)

-- State the theorem
theorem y_value_at_243 (k : ℝ) :
  y k 32 = 4 → y k 243 = 6 := by
  sorry

end y_value_at_243_l28_2863


namespace triangle_area_heron_l28_2831

/-- Given a triangle ABC with sides a, b, c and area S, prove that under certain conditions, 
    the area S calculated using Heron's formula is equal to 15√7/4 -/
theorem triangle_area_heron (a b c : ℝ) (S : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 * Real.sin C = 24 * Real.sin A →
  a * (Real.sin C - Real.sin B) * (c + b) = (27 - a^2) * Real.sin A →
  S = Real.sqrt ((1/4) * (a^2 * c^2 - ((a^2 + c^2 - b^2) / 2)^2)) →
  S = 15 * Real.sqrt 7 / 4 := by
  sorry

end triangle_area_heron_l28_2831


namespace shaded_area_square_with_circles_l28_2818

/-- The shaded area of a square with side length 36 inches containing 9 tangent circles -/
theorem shaded_area_square_with_circles : 
  let square_side : ℝ := 36
  let num_circles : ℕ := 9
  let circle_radius : ℝ := square_side / 6

  let square_area : ℝ := square_side ^ 2
  let total_circles_area : ℝ := num_circles * Real.pi * circle_radius ^ 2
  let shaded_area : ℝ := square_area - total_circles_area

  shaded_area = 1296 - 324 * Real.pi :=
by
  sorry


end shaded_area_square_with_circles_l28_2818


namespace circle_intersection_range_l28_2865

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - 2*a)^2 + (y - (a + 3))^2 = 4 ∧ x^2 + y^2 = 1) →
  -6/5 < a ∧ a < 0 :=
by sorry

end circle_intersection_range_l28_2865


namespace penumbra_ring_area_l28_2871

/-- The area of a ring formed between two concentric circles --/
theorem penumbra_ring_area (r_umbra : ℝ) (r_penumbra : ℝ) (h1 : r_umbra = 40) (h2 : r_penumbra = 3 * r_umbra) :
  let a_ring := π * r_penumbra^2 - π * r_umbra^2
  a_ring = 12800 * π := by sorry

end penumbra_ring_area_l28_2871


namespace arithmetic_square_root_of_sqrt_16_l28_2848

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end arithmetic_square_root_of_sqrt_16_l28_2848


namespace train_crossing_time_l28_2826

-- Define the given values
def train_length : Real := 210  -- meters
def train_speed : Real := 25    -- km/h
def man_speed : Real := 2       -- km/h

-- Define the theorem
theorem train_crossing_time :
  let relative_speed : Real := train_speed + man_speed
  let relative_speed_mps : Real := relative_speed * 1000 / 3600
  let time : Real := train_length / relative_speed_mps
  time = 28 := by
  sorry


end train_crossing_time_l28_2826


namespace parallelogram_height_l28_2862

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) 
    (h1 : area = 384) 
    (h2 : base = 24) 
    (h3 : area = base * height) : height = 16 := by
  sorry

end parallelogram_height_l28_2862


namespace sine_transformation_l28_2841

theorem sine_transformation (x : ℝ) : 
  Real.sin (2 * x + (2 * Real.pi) / 3) = Real.sin (2 * (x + Real.pi / 3)) := by
  sorry

end sine_transformation_l28_2841


namespace cylinder_lateral_area_l28_2884

/-- The lateral area of a cylinder with base diameter and height both equal to 4 cm is 16π cm². -/
theorem cylinder_lateral_area (π : ℝ) (h : π > 0) : 
  let d : ℝ := 4 -- diameter
  let h : ℝ := 4 -- height
  let r : ℝ := d / 2 -- radius
  let lateral_area : ℝ := 2 * π * r * h
  lateral_area = 16 * π := by sorry

end cylinder_lateral_area_l28_2884


namespace abigail_score_l28_2850

theorem abigail_score (n : ℕ) (initial_avg final_avg : ℚ) (abigail_score : ℚ) :
  n = 20 →
  initial_avg = 85 →
  final_avg = 86 →
  (n : ℚ) * initial_avg + abigail_score = (n + 1 : ℚ) * final_avg →
  abigail_score = 106 :=
by sorry

end abigail_score_l28_2850


namespace total_books_l28_2898

def books_per_shelf : ℕ := 15
def mystery_shelves : ℕ := 8
def picture_shelves : ℕ := 4
def biography_shelves : ℕ := 3
def scifi_shelves : ℕ := 5

theorem total_books : 
  books_per_shelf * (mystery_shelves + picture_shelves + biography_shelves + scifi_shelves) = 300 := by
  sorry

end total_books_l28_2898


namespace triangle_angle_ranges_l28_2801

def triangle_angles (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

theorem triangle_angle_ranges (α β γ : Real) (h : triangle_angles α β γ) :
  60 ≤ max α (max β γ) ∧ max α (max β γ) < 180 ∧
  0 < min α (min β γ) ∧ min α (min β γ) ≤ 60 ∧
  0 < (max (min α β) (min (max α β) γ)) ∧ (max (min α β) (min (max α β) γ)) < 90 := by
  sorry

end triangle_angle_ranges_l28_2801


namespace last_installment_value_is_3300_l28_2861

/-- Represents the payment structure for a TV set purchase -/
structure TVPayment where
  price : ℕ               -- Price of the TV set in Rupees
  num_installments : ℕ    -- Number of installments
  installment_amount : ℕ  -- Amount of each installment in Rupees
  interest_rate : ℚ       -- Annual interest rate as a rational number
  processing_fee : ℕ      -- Processing fee in Rupees

/-- Calculates the value of the last installment for a TV payment plan -/
def last_installment_value (payment : TVPayment) : ℕ :=
  payment.installment_amount + payment.processing_fee

/-- Theorem stating that the last installment value for the given TV payment plan is 3300 Rupees -/
theorem last_installment_value_is_3300 (payment : TVPayment) 
  (h1 : payment.price = 35000)
  (h2 : payment.num_installments = 36)
  (h3 : payment.installment_amount = 2300)
  (h4 : payment.interest_rate = 9 / 100)
  (h5 : payment.processing_fee = 1000) :
  last_installment_value payment = 3300 := by
  sorry

#eval last_installment_value { 
  price := 35000, 
  num_installments := 36, 
  installment_amount := 2300, 
  interest_rate := 9 / 100, 
  processing_fee := 1000 
}

end last_installment_value_is_3300_l28_2861


namespace brother_age_twice_sister_l28_2882

def brother_age_2005 : ℕ := 16
def sister_age_2005 : ℕ := 10
def reference_year : ℕ := 2005

theorem brother_age_twice_sister : 
  ∃ (year : ℕ), year = reference_year - (brother_age_2005 - 2 * sister_age_2005) ∧ year = 2001 :=
sorry

end brother_age_twice_sister_l28_2882


namespace sum_of_solutions_l28_2813

-- Define the equation
def equation (x : ℝ) : Prop :=
  2 * Real.cos (2 * x) * (Real.cos (2 * x) - Real.cos (2000 * Real.pi ^ 2 / x)) = Real.cos (4 * x) - 1

-- Define the set of all positive real solutions
def solution_set : Set ℝ := {x | x > 0 ∧ equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, x ∈ solution_set) ∧
                    (∀ x ∈ solution_set, x ∈ S) ∧
                    (Finset.sum S id = 136 * Real.pi) := by
  sorry

end sum_of_solutions_l28_2813


namespace complex_star_angle_sum_l28_2895

/-- An n-pointed complex star is formed from a regular n-gon by extending every third side --/
structure ComplexStar where
  n : ℕ
  is_even : Even n
  n_ge_6 : n ≥ 6

/-- The sum of interior angles at the n intersections of a complex star --/
def interior_angle_sum (star : ComplexStar) : ℝ :=
  180 * (star.n - 6)

/-- Theorem: The sum of interior angles at the n intersections of a complex star is 180° * (n-6) --/
theorem complex_star_angle_sum (star : ComplexStar) :
  interior_angle_sum star = 180 * (star.n - 6) := by
  sorry

end complex_star_angle_sum_l28_2895


namespace round_trip_distance_boy_school_distance_l28_2811

/-- Calculates the distance between two points given the speeds and total time of a round trip -/
theorem round_trip_distance (outbound_speed return_speed : ℝ) (total_time : ℝ) : 
  outbound_speed > 0 → return_speed > 0 → total_time > 0 →
  (1 / outbound_speed + 1 / return_speed) * (outbound_speed * return_speed * total_time / (outbound_speed + return_speed)) = total_time := by
  sorry

/-- The distance between the boy's house and school -/
theorem boy_school_distance : 
  let outbound_speed : ℝ := 3
  let return_speed : ℝ := 2
  let total_time : ℝ := 5
  (outbound_speed * return_speed * total_time) / (outbound_speed + return_speed) = 6 := by
  sorry

end round_trip_distance_boy_school_distance_l28_2811


namespace unique_solution_quadratic_with_square_root_l28_2845

theorem unique_solution_quadratic_with_square_root :
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 4) = 31 :=
by
  -- The unique solution is (11 - 3√5) / 2
  use (11 - 3 * Real.sqrt 5) / 2
  sorry

end unique_solution_quadratic_with_square_root_l28_2845


namespace geometric_sequence_ratio_l28_2812

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n
  prop1 : a 7 * a 11 = 6
  prop2 : a 4 + a 14 = 5

/-- The main theorem stating the possible values of a_20 / a_10 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 10 = 2/3 ∨ seq.a 20 / seq.a 10 = 3/2 := by
  sorry

end geometric_sequence_ratio_l28_2812


namespace binomial_coefficient_21_14_l28_2873

theorem binomial_coefficient_21_14 : Nat.choose 21 14 = 116280 :=
by
  have h1 : Nat.choose 20 13 = 77520 := by sorry
  have h2 : Nat.choose 20 14 = 38760 := by sorry
  sorry

end binomial_coefficient_21_14_l28_2873


namespace binomial_sum_problem_l28_2820

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem binomial_sum_problem : 
  (binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006) ∧ 
  (binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32) :=
by sorry

end binomial_sum_problem_l28_2820


namespace pet_store_snakes_l28_2867

/-- The number of snakes in a pet store -/
theorem pet_store_snakes (num_cages : ℕ) (snakes_per_cage : ℕ) : 
  num_cages = 2 → snakes_per_cage = 2 → num_cages * snakes_per_cage = 4 := by
  sorry

#check pet_store_snakes

end pet_store_snakes_l28_2867


namespace first_group_size_first_group_size_is_16_l28_2890

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℝ := 25

/-- The number of men in the second group -/
def men_second_group : ℝ := 15

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℝ := 26.666666666666668

/-- The work done is inversely proportional to the number of days taken -/
axiom work_time_inverse_proportion {m1 m2 d1 d2 : ℝ} :
  m1 * d1 = m2 * d2

theorem first_group_size : ℝ := by
  sorry

theorem first_group_size_is_16 : first_group_size = 16 := by
  sorry

end first_group_size_first_group_size_is_16_l28_2890


namespace triangle_side_b_l28_2846

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_side_b (t : Triangle) : 
  t.B = π / 6 → t.a = Real.sqrt 3 → t.c = 1 → t.b = 1 := by
  sorry

end triangle_side_b_l28_2846
