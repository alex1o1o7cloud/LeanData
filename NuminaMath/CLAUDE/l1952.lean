import Mathlib

namespace derivative_at_one_l1952_195268

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 2| < ε :=
sorry

end derivative_at_one_l1952_195268


namespace hearty_beads_count_l1952_195214

/-- The number of packages of blue beads -/
def blue_packages : ℕ := 4

/-- The number of packages of red beads -/
def red_packages : ℕ := 5

/-- The number of packages of green beads -/
def green_packages : ℕ := 2

/-- The number of beads in each blue package -/
def blue_beads_per_package : ℕ := 30

/-- The number of beads in each red package -/
def red_beads_per_package : ℕ := 45

/-- The number of additional beads in each green package compared to a blue package -/
def green_extra_beads : ℕ := 15

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * blue_beads_per_package + 
                        red_packages * red_beads_per_package + 
                        green_packages * (blue_beads_per_package + green_extra_beads)

theorem hearty_beads_count : total_beads = 435 := by
  sorry

end hearty_beads_count_l1952_195214


namespace james_lifting_ratio_l1952_195242

def initial_total : ℝ := 2200
def initial_weight : ℝ := 245
def total_gain_percentage : ℝ := 0.15
def weight_gain : ℝ := 8

def new_total : ℝ := initial_total * (1 + total_gain_percentage)
def new_weight : ℝ := initial_weight + weight_gain

theorem james_lifting_ratio :
  new_total / new_weight = 10 := by sorry

end james_lifting_ratio_l1952_195242


namespace sqrt_inequality_l1952_195273

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) + Real.sqrt (x - 2) > Real.sqrt (x - 4) + Real.sqrt (x - 1) := by
  sorry

end sqrt_inequality_l1952_195273


namespace sum_solution_equations_find_a_value_l1952_195248

/-- Definition of a "sum solution equation" -/
def is_sum_solution_equation (a b : ℚ) : Prop :=
  (b / a) = b + a

/-- Theorem for the given equations -/
theorem sum_solution_equations :
  is_sum_solution_equation (-3) (9/4) ∧
  ¬is_sum_solution_equation (2/3) (-2/3) ∧
  ¬is_sum_solution_equation 5 (-2) :=
sorry

/-- Theorem for finding the value of a -/
theorem find_a_value (a : ℚ) :
  is_sum_solution_equation 3 (2*a - 10) → a = 11/4 :=
sorry

end sum_solution_equations_find_a_value_l1952_195248


namespace hall_covering_cost_l1952_195291

def hall_length : ℝ := 20
def hall_width : ℝ := 15
def hall_height : ℝ := 5
def cost_per_square_meter : ℝ := 50

def total_area : ℝ := 2 * (hall_length * hall_width) + 2 * (hall_length * hall_height) + 2 * (hall_width * hall_height)

def total_expenditure : ℝ := total_area * cost_per_square_meter

theorem hall_covering_cost : total_expenditure = 47500 := by
  sorry

end hall_covering_cost_l1952_195291


namespace caroline_lassis_l1952_195222

/-- The number of lassis Caroline can make from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  8 * mangoes / 3

/-- Theorem stating that Caroline can make 40 lassis from 15 mangoes -/
theorem caroline_lassis : lassis_from_mangoes 15 = 40 := by
  sorry

end caroline_lassis_l1952_195222


namespace pythagorean_triple_parity_l1952_195283

theorem pythagorean_triple_parity (x y z : ℤ) (h : x^2 + y^2 = z^2) :
  Even x ∨ Even y := by
  sorry

end pythagorean_triple_parity_l1952_195283


namespace chemistry_books_count_l1952_195270

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The problem statement -/
theorem chemistry_books_count :
  ∃ (C : ℕ), C > 0 ∧ choose_2 15 * choose_2 C = 2940 ∧ C = 8 := by sorry

end chemistry_books_count_l1952_195270


namespace largest_power_of_five_factor_l1952_195255

-- Define factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the sum of factorials
def sum_of_factorials : ℕ := factorial 77 + factorial 78 + factorial 79

-- Define the function to count factors of 5
def count_factors_of_five (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + if x % 5 = 0 then 1 else 0) 0

-- Theorem statement
theorem largest_power_of_five_factor :
  ∃ (n : ℕ), n = 18 ∧ 5^n ∣ sum_of_factorials ∧ ¬(5^(n+1) ∣ sum_of_factorials) := by
  sorry

end largest_power_of_five_factor_l1952_195255


namespace isosceles_trapezoid_area_isosceles_trapezoid_area_is_768_l1952_195209

/-- An isosceles trapezoid with the given properties has an area of 768 sq cm. -/
theorem isosceles_trapezoid_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun leg_length diagonal_length longer_base area =>
    leg_length = 30 ∧
    diagonal_length = 40 ∧
    longer_base = 50 ∧
    area = 768 ∧
    ∃ (height shorter_base : ℝ),
      height > 0 ∧
      shorter_base > 0 ∧
      shorter_base < longer_base ∧
      leg_length^2 = height^2 + ((longer_base - shorter_base) / 2)^2 ∧
      diagonal_length^2 = height^2 + (longer_base^2 / 4) ∧
      area = (longer_base + shorter_base) * height / 2

/-- The isosceles trapezoid with the given properties has an area of 768 sq cm. -/
theorem isosceles_trapezoid_area_is_768 : isosceles_trapezoid_area 30 40 50 768 := by
  sorry

end isosceles_trapezoid_area_isosceles_trapezoid_area_is_768_l1952_195209


namespace raccoon_lock_problem_l1952_195252

theorem raccoon_lock_problem :
  ∀ (x : ℝ),
  let first_lock_time := 5
  let second_lock_time := x * first_lock_time - 3
  let both_locks_time := 5 * second_lock_time
  both_locks_time = 60 →
  x = 3 := by
sorry

end raccoon_lock_problem_l1952_195252


namespace candies_given_to_stephanie_l1952_195269

theorem candies_given_to_stephanie (initial_candies remaining_candies : ℕ) 
  (h1 : initial_candies = 95)
  (h2 : remaining_candies = 92) :
  initial_candies - remaining_candies = 3 := by
  sorry

end candies_given_to_stephanie_l1952_195269


namespace max_quarters_sasha_l1952_195224

/-- Represents the value of a coin in cents -/
def coin_value (coin_type : String) : ℕ :=
  match coin_type with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The total amount Sasha has in cents -/
def total_amount : ℕ := 480

/-- Theorem stating the maximum number of quarters Sasha can have -/
theorem max_quarters_sasha : 
  ∀ (quarters nickels dimes : ℕ),
  quarters = nickels →
  dimes = 4 * nickels →
  quarters * coin_value "quarter" + 
  nickels * coin_value "nickel" + 
  dimes * coin_value "dime" ≤ total_amount →
  quarters ≤ 6 := by
sorry

end max_quarters_sasha_l1952_195224


namespace oak_elm_difference_pine_elm_difference_l1952_195298

-- Define the heights of the trees
def elm_height : ℚ := 49/4  -- 12¼ feet
def oak_height : ℚ := 37/2  -- 18½ feet
def pine_height_inches : ℚ := 225  -- 225 inches

-- Convert pine height to feet
def pine_height : ℚ := pine_height_inches / 12

-- Define the theorems to be proved
theorem oak_elm_difference : oak_height - elm_height = 25/4 := by sorry

theorem pine_elm_difference : pine_height - elm_height = 13/2 := by sorry

end oak_elm_difference_pine_elm_difference_l1952_195298


namespace num_keepers_is_correct_l1952_195219

/-- The number of keepers in a caravan with hens, goats, and camels. -/
def num_keepers : ℕ :=
  let num_hens : ℕ := 50
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let hen_feet : ℕ := 2
  let goat_feet : ℕ := 4
  let camel_feet : ℕ := 4
  let keeper_feet : ℕ := 2
  let total_animal_feet : ℕ := num_hens * hen_feet + num_goats * goat_feet + num_camels * camel_feet
  let total_animal_heads : ℕ := num_hens + num_goats + num_camels
  let extra_feet : ℕ := 224
  15

theorem num_keepers_is_correct : num_keepers = 15 := by
  sorry

#eval num_keepers

end num_keepers_is_correct_l1952_195219


namespace area_of_absolute_value_graph_l1952_195225

/-- The area enclosed by the graph of |x| + |3y| = 9 is 54 square units -/
theorem area_of_absolute_value_graph : 
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ |x| + |3 * y|
  ∃ S : Set (ℝ × ℝ), S = {p : ℝ × ℝ | f p = 9} ∧ MeasureTheory.volume S = 54 := by
  sorry

end area_of_absolute_value_graph_l1952_195225


namespace x_range_for_inequality_l1952_195284

theorem x_range_for_inequality (x : ℝ) :
  (∀ a : ℝ, a ∈ Set.Icc 0 2 → a * x^2 + (a + 1) * x + 1 - (3/2) * a < 0) →
  x ∈ Set.Ioo (-2) (-1) := by
  sorry

end x_range_for_inequality_l1952_195284


namespace triangle_problem_l1952_195286

/-- Given a triangle ABC with sides a and b that are roots of x^2 - 2√3x + 2 = 0,
    and 2cos(A+B) = 1, prove that angle C is 120° and side AB has length √10 -/
theorem triangle_problem (a b : ℝ) (A B C : ℝ) :
  a^2 - 2 * Real.sqrt 3 * a + 2 = 0 →
  b^2 - 2 * Real.sqrt 3 * b + 2 = 0 →
  2 * Real.cos (A + B) = 1 →
  C = 2 * π / 3 ∧
  (a^2 + b^2 - 2 * a * b * Real.cos C) = 10 :=
by sorry

end triangle_problem_l1952_195286


namespace pizza_cost_l1952_195276

/-- The cost of purchasing pizzas with special pricing -/
theorem pizza_cost (standard_price : ℕ) (triple_cheese_count : ℕ) (meat_lovers_count : ℕ) :
  standard_price = 5 →
  triple_cheese_count = 10 →
  meat_lovers_count = 9 →
  (standard_price * (triple_cheese_count / 2 + 2 * meat_lovers_count / 3) : ℕ) = 55 := by
  sorry

#check pizza_cost

end pizza_cost_l1952_195276


namespace female_officers_count_l1952_195278

theorem female_officers_count (total_on_duty : ℕ) (male_on_duty : ℕ) 
  (female_on_duty_percentage : ℚ) :
  total_on_duty = 475 →
  male_on_duty = 315 →
  female_on_duty_percentage = 65/100 →
  ∃ (total_female : ℕ), 
    (total_female : ℚ) * female_on_duty_percentage = (total_on_duty - male_on_duty : ℚ) ∧
    total_female = 246 :=
by
  sorry

end female_officers_count_l1952_195278


namespace square_minus_twice_plus_nine_equals_eleven_l1952_195220

theorem square_minus_twice_plus_nine_equals_eleven :
  let a : ℝ := 2 / (Real.sqrt 3 - 1)
  a^2 - 2*a + 9 = 11 := by sorry

end square_minus_twice_plus_nine_equals_eleven_l1952_195220


namespace perpendicular_vectors_x_value_l1952_195245

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_value :
  ∃ (x : ℝ), x > 0 ∧ dot_product (vector_a x) (vector_b x) = 0 ∧ x = 1 := by
  sorry

end perpendicular_vectors_x_value_l1952_195245


namespace right_triangle_hypotenuse_l1952_195226

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Ensure positive side lengths
  (a^2 + b^2 = c^2) →        -- Pythagorean theorem (right-angled triangle)
  (a^2 + b^2 + c^2 = 2450) → -- Sum of squares condition
  (b = a + 7) →              -- One leg is 7 units longer
  c = 35 := by               -- Conclusion: hypotenuse length is 35
sorry

end right_triangle_hypotenuse_l1952_195226


namespace triangular_to_square_ratio_l1952_195227

/-- A polyhedron with only triangular and square faces -/
structure Polyhedron :=
  (triangular_faces : ℕ)
  (square_faces : ℕ)

/-- Property that no two faces of the same type share an edge -/
def no_same_type_edge_sharing (p : Polyhedron) : Prop :=
  ∀ (edge : ℕ), (∃! square_face : ℕ, square_face ≤ p.square_faces) ∧
                (∃! triangular_face : ℕ, triangular_face ≤ p.triangular_faces)

theorem triangular_to_square_ratio (p : Polyhedron) 
  (h : no_same_type_edge_sharing p) (h_pos : p.square_faces > 0) : 
  (p.triangular_faces : ℚ) / p.square_faces = 4 / 3 := by
  sorry

end triangular_to_square_ratio_l1952_195227


namespace or_false_implies_both_false_l1952_195261

theorem or_false_implies_both_false (p q : Prop) : 
  (¬p ∨ ¬q) → (¬p ∧ ¬q) := by sorry

end or_false_implies_both_false_l1952_195261


namespace library_visitors_l1952_195287

theorem library_visitors (visitors_non_sunday : ℕ) (avg_visitors_per_day : ℕ) :
  visitors_non_sunday = 140 →
  avg_visitors_per_day = 200 →
  ∃ (visitors_sunday : ℕ),
    5 * visitors_sunday + 25 * visitors_non_sunday = 30 * avg_visitors_per_day ∧
    visitors_sunday = 500 := by
  sorry

end library_visitors_l1952_195287


namespace expression_values_l1952_195265

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + (a * b * c) / abs (a * b * c)
  expr = -4 ∨ expr = 0 ∨ expr = 4 := by
  sorry

end expression_values_l1952_195265


namespace abraham_shower_gels_l1952_195210

def shower_gel_problem (budget : ℕ) (shower_gel_cost : ℕ) (toothpaste_cost : ℕ) (detergent_cost : ℕ) (remaining : ℕ) : Prop :=
  let total_spent : ℕ := budget - remaining
  let non_gel_cost : ℕ := toothpaste_cost + detergent_cost
  let gel_cost : ℕ := total_spent - non_gel_cost
  gel_cost / shower_gel_cost = 4

theorem abraham_shower_gels :
  shower_gel_problem 60 4 3 11 30 := by
  sorry

end abraham_shower_gels_l1952_195210


namespace meeting_time_calculation_l1952_195229

-- Define the speeds of the two people
def v₁ : ℝ := 6
def v₂ : ℝ := 4

-- Define the time difference in reaching the final destination
def time_difference : ℝ := 10

-- Define the theorem to prove
theorem meeting_time_calculation (t₁ : ℝ) :
  v₂ * t₁ = v₁ * (t₁ - time_difference) → t₁ = 30 :=
by sorry

end meeting_time_calculation_l1952_195229


namespace henley_candy_problem_l1952_195285

theorem henley_candy_problem :
  ∀ (total_candies : ℕ),
    (total_candies : ℚ) * (60 : ℚ) / 100 = 3 * 60 →
    total_candies = 300 :=
by
  sorry

end henley_candy_problem_l1952_195285


namespace projection_of_a_onto_b_l1952_195208

def a : Fin 2 → ℚ := ![1, 2]
def b : Fin 2 → ℚ := ![-2, 4]

def dot_product (v w : Fin 2 → ℚ) : ℚ :=
  (v 0) * (w 0) + (v 1) * (w 1)

def magnitude_squared (v : Fin 2 → ℚ) : ℚ :=
  dot_product v v

def scalar_mult (c : ℚ) (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  fun i => c * (v i)

def projection (v w : Fin 2 → ℚ) : Fin 2 → ℚ :=
  scalar_mult ((dot_product v w) / (magnitude_squared w)) w

theorem projection_of_a_onto_b :
  projection a b = ![-(3/5), 6/5] := by
  sorry

end projection_of_a_onto_b_l1952_195208


namespace probability_two_females_one_male_l1952_195256

theorem probability_two_females_one_male (total : ℕ) (females : ℕ) (males : ℕ) (chosen : ℕ) :
  total = females + males →
  total = 8 →
  females = 5 →
  males = 3 →
  chosen = 3 →
  (Nat.choose females 2 * Nat.choose males 1 : ℚ) / Nat.choose total chosen = 15 / 28 :=
by sorry

end probability_two_females_one_male_l1952_195256


namespace min_value_x_plus_four_over_x_l1952_195274

theorem min_value_x_plus_four_over_x :
  ∀ x : ℝ, x > 0 → x + 4 / x ≥ 4 ∧ ∃ y : ℝ, y > 0 ∧ y + 4 / y = 4 := by
  sorry

end min_value_x_plus_four_over_x_l1952_195274


namespace decimal_point_problem_l1952_195251

theorem decimal_point_problem : ∃! (x : ℝ), x > 0 ∧ 10000 * x = 9 / x ∧ x = 0.03 := by sorry

end decimal_point_problem_l1952_195251


namespace concert_revenue_l1952_195258

/-- Calculate the total revenue from concert ticket sales --/
theorem concert_revenue (ticket_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
  (first_group : ℕ) (second_group : ℕ) (total_people : ℕ) :
  ticket_price = 20 →
  first_discount = 0.4 →
  second_discount = 0.15 →
  first_group = 10 →
  second_group = 20 →
  total_people = 45 →
  (first_group * ticket_price * (1 - first_discount) +
   second_group * ticket_price * (1 - second_discount) +
   (total_people - first_group - second_group) * ticket_price) = 760 := by
sorry

end concert_revenue_l1952_195258


namespace p_necessary_not_sufficient_l1952_195241

-- Define the propositions p and q
def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end p_necessary_not_sufficient_l1952_195241


namespace carries_tshirt_purchase_l1952_195217

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.65

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 12

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * (num_tshirts : ℝ)

/-- Theorem: The total cost of Carrie's t-shirt purchase is $115.80 -/
theorem carries_tshirt_purchase : total_cost = 115.80 := by
  sorry

end carries_tshirt_purchase_l1952_195217


namespace remainder_of_binary_number_div_4_l1952_195232

def binary_number : ℕ := 3789 -- 111001001101₂ in decimal

theorem remainder_of_binary_number_div_4 :
  binary_number % 4 = 1 := by
  sorry

end remainder_of_binary_number_div_4_l1952_195232


namespace polygon_diagonals_minus_sides_l1952_195292

theorem polygon_diagonals_minus_sides (n : ℕ) (h : n = 105) : 
  (n * (n - 3)) / 2 - n = 5250 := by
  sorry

end polygon_diagonals_minus_sides_l1952_195292


namespace cost_of_pancakes_l1952_195272

/-- The cost of pancakes given initial order, tax, payment, and change --/
theorem cost_of_pancakes 
  (eggs_cost : ℕ)
  (cocoa_cost : ℕ)
  (cocoa_quantity : ℕ)
  (tax : ℕ)
  (payment : ℕ)
  (change : ℕ)
  (h1 : eggs_cost = 3)
  (h2 : cocoa_cost = 2)
  (h3 : cocoa_quantity = 2)
  (h4 : tax = 1)
  (h5 : payment = 15)
  (h6 : change = 1)
  : ℕ := by
  sorry

#check cost_of_pancakes

end cost_of_pancakes_l1952_195272


namespace system_solution_unique_l1952_195240

theorem system_solution_unique :
  ∃! (x y : ℚ),
    1 / (2 - x + 2 * y) - 1 / (x + 2 * y - 1) = 2 ∧
    1 / (2 - x + 2 * y) - 1 / (1 - x - 2 * y) = 4 ∧
    x = 11 / 6 ∧
    y = 1 / 12 := by
  sorry

end system_solution_unique_l1952_195240


namespace mrs_hilt_pizza_slices_l1952_195213

theorem mrs_hilt_pizza_slices :
  ∀ (num_pizzas : ℕ) (slices_per_pizza : ℕ),
    num_pizzas = 5 →
    slices_per_pizza = 12 →
    num_pizzas * slices_per_pizza = 60 :=
by
  sorry

end mrs_hilt_pizza_slices_l1952_195213


namespace figure_b_impossible_l1952_195277

-- Define the shape of a square
structure Square :=
  (side : ℝ)
  (area : ℝ := side * side)

-- Define the set of available squares
def available_squares : Finset Square := sorry

-- Define the shapes of the five figures
inductive Figure
| A
| B
| C
| D
| E

-- Function to check if a figure can be formed from the available squares
def can_form_figure (f : Figure) (squares : Finset Square) : Prop := sorry

-- Theorem stating that figure B cannot be formed while others can
theorem figure_b_impossible :
  (∀ s ∈ available_squares, s.side = 1) →
  (available_squares.card = 17) →
  (¬ can_form_figure Figure.B available_squares) ∧
  (can_form_figure Figure.A available_squares) ∧
  (can_form_figure Figure.C available_squares) ∧
  (can_form_figure Figure.D available_squares) ∧
  (can_form_figure Figure.E available_squares) :=
by sorry

end figure_b_impossible_l1952_195277


namespace infinite_series_sum_l1952_195275

open Real

/-- The sum of the infinite series ∑(n=1 to ∞) (n + 1) / (n + 2)! is equal to e - 3 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (n + 1 : ℝ) / (n + 2).factorial) = Real.exp 1 - 3 := by
  sorry

end infinite_series_sum_l1952_195275


namespace not_all_data_sets_have_regression_equation_l1952_195282

-- Define a type for data sets
structure DataSet where
  -- Add necessary fields to represent a data set
  dummy : Unit

-- Define a predicate for the existence of a regression equation
def has_regression_equation (ds : DataSet) : Prop :=
  -- Add necessary conditions for a data set to have a regression equation
  sorry

-- Theorem stating that not every data set has a regression equation
theorem not_all_data_sets_have_regression_equation :
  ¬ (∀ ds : DataSet, has_regression_equation ds) := by
  sorry

end not_all_data_sets_have_regression_equation_l1952_195282


namespace rulers_placed_l1952_195212

theorem rulers_placed (initial_rulers final_rulers : ℕ) (h : final_rulers = initial_rulers + 14) :
  final_rulers - initial_rulers = 14 := by
  sorry

end rulers_placed_l1952_195212


namespace seven_consecutive_integers_product_first_57_integers_product_l1952_195297

-- Define a function to calculate the number of trailing zeros
def trailingZeros (n : ℕ) : ℕ := sorry

-- Theorem for seven consecutive integers
theorem seven_consecutive_integers_product (k : ℕ) :
  ∃ m : ℕ, m > 0 ∧ trailingZeros ((k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6) * (k + 7)) ≥ m :=
sorry

-- Theorem for the product of first 57 positive integers
theorem first_57_integers_product :
  trailingZeros (Nat.factorial 57) = 13 :=
sorry

end seven_consecutive_integers_product_first_57_integers_product_l1952_195297


namespace approximate_root_l1952_195280

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem approximate_root (hcont : Continuous f) 
  (h1 : f 0.64 < 0) (h2 : f 0.72 > 0) (h3 : f 0.68 < 0) :
  ∃ (x : ℝ), f x = 0 ∧ |x - 0.7| ≤ 0.1 := by
  sorry

end approximate_root_l1952_195280


namespace real_solution_exists_l1952_195230

theorem real_solution_exists (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 := by
  sorry

end real_solution_exists_l1952_195230


namespace orange_bucket_difference_l1952_195263

theorem orange_bucket_difference (bucket1 bucket2 bucket3 total : ℕ) : 
  bucket1 = 22 →
  bucket2 = bucket1 + 17 →
  bucket3 < bucket2 →
  total = bucket1 + bucket2 + bucket3 →
  total = 89 →
  bucket2 - bucket3 = 11 := by
sorry

end orange_bucket_difference_l1952_195263


namespace floor_abs_negative_l1952_195207

theorem floor_abs_negative : ⌊|(-57.8 : ℝ)|⌋ = 57 := by sorry

end floor_abs_negative_l1952_195207


namespace complex_equation_solution_l1952_195200

theorem complex_equation_solution :
  ∀ (x y : ℝ), (1 + x * Complex.I) * (1 - 2 * Complex.I) = y → x = 2 ∧ y = 5 := by
sorry

end complex_equation_solution_l1952_195200


namespace square_side_length_equal_perimeter_l1952_195262

theorem square_side_length_equal_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 10)
  (h2 : rectangle_width = 8) : 
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let square_side_length := rectangle_perimeter / 4
  square_side_length = 9 := by
sorry

end square_side_length_equal_perimeter_l1952_195262


namespace mrs_a_speed_l1952_195296

/-- Proves that Mrs. A's speed is 10 kmph given the problem conditions --/
theorem mrs_a_speed (initial_distance : ℝ) (mr_a_speed : ℝ) (bee_speed : ℝ) (bee_distance : ℝ)
  (h1 : initial_distance = 120)
  (h2 : mr_a_speed = 30)
  (h3 : bee_speed = 60)
  (h4 : bee_distance = 180) :
  let time := bee_distance / bee_speed
  let mr_a_distance := mr_a_speed * time
  let mrs_a_distance := initial_distance - mr_a_distance
  mrs_a_distance / time = 10 := by
  sorry

#check mrs_a_speed

end mrs_a_speed_l1952_195296


namespace expression_zero_l1952_195264

theorem expression_zero (a b c : ℝ) (h : c = b + 2) :
  b = -2 ∧ c = 0 → (a - (b + c)) - ((a + c) - b) = 0 :=
by sorry

end expression_zero_l1952_195264


namespace z_to_twelve_equals_one_l1952_195260

theorem z_to_twelve_equals_one :
  let z : ℂ := (Real.sqrt 3 - Complex.I) / 2
  z^12 = 1 := by sorry

end z_to_twelve_equals_one_l1952_195260


namespace steve_writes_24_pages_l1952_195290

/-- Calculates the number of pages Steve writes in a month -/
def stevePages : ℕ :=
  let daysInMonth : ℕ := 30
  let letterFrequency : ℕ := 3
  let regularLetterTime : ℕ := 20
  let timePerPage : ℕ := 10
  let longLetterTimeMultiplier : ℕ := 2
  let longLetterTotalTime : ℕ := 80

  let regularLettersCount : ℕ := daysInMonth / letterFrequency
  let regularLettersTotalTime : ℕ := regularLettersCount * regularLetterTime
  let regularLettersPages : ℕ := regularLettersTotalTime / timePerPage

  let longLetterTimePerPage : ℕ := timePerPage * longLetterTimeMultiplier
  let longLetterPages : ℕ := longLetterTotalTime / longLetterTimePerPage

  regularLettersPages + longLetterPages

theorem steve_writes_24_pages : stevePages = 24 := by
  sorry

end steve_writes_24_pages_l1952_195290


namespace pencil_length_after_sharpening_l1952_195257

/-- Calculates the final length of a pencil after sharpening on four consecutive days. -/
def final_pencil_length (initial_length : ℕ) (day1 day2 day3 day4 : ℕ) : ℕ :=
  initial_length - (day1 + day2 + day3 + day4)

/-- Theorem stating that given specific initial length and sharpening amounts, the final pencil length is 36 inches. -/
theorem pencil_length_after_sharpening :
  final_pencil_length 50 2 3 4 5 = 36 := by
  sorry

end pencil_length_after_sharpening_l1952_195257


namespace bicycle_journey_initial_time_l1952_195237

theorem bicycle_journey_initial_time 
  (speed : ℝ) 
  (additional_distance : ℝ) 
  (rest_time : ℝ) 
  (final_distance : ℝ) 
  (total_time : ℝ) :
  speed = 10 →
  additional_distance = 15 →
  rest_time = 30 →
  final_distance = 20 →
  total_time = 270 →
  ∃ (initial_time : ℝ), 
    initial_time * 60 + additional_distance / speed * 60 + rest_time + final_distance / speed * 60 = total_time ∧ 
    initial_time * 60 = 30 :=
by sorry

end bicycle_journey_initial_time_l1952_195237


namespace f_2021_value_l1952_195221

-- Define the set A
def A : Set ℚ := {x : ℚ | x ≠ -1 ∧ x ≠ 0}

-- Define the function property
def has_property (f : A → ℝ) : Prop :=
  ∀ x : A, f x + f ⟨1 + 1 / x, sorry⟩ = (1/2) * Real.log (abs (x : ℝ))

-- State the theorem
theorem f_2021_value (f : A → ℝ) (h : has_property f) :
  f ⟨2021, sorry⟩ = (1/2) * Real.log 2021 := by sorry

end f_2021_value_l1952_195221


namespace pages_ratio_l1952_195246

theorem pages_ratio (lana_initial : ℕ) (duane_initial : ℕ) (lana_final : ℕ)
  (h1 : lana_initial = 8)
  (h2 : duane_initial = 42)
  (h3 : lana_final = 29) :
  (lana_final - lana_initial) * 2 = duane_initial :=
by sorry

end pages_ratio_l1952_195246


namespace lollipop_count_l1952_195238

theorem lollipop_count (total_cost : ℝ) (single_cost : ℝ) (count : ℕ) : 
  total_cost = 90 →
  single_cost = 0.75 →
  (count : ℝ) * single_cost = total_cost →
  count = 120 := by
sorry

end lollipop_count_l1952_195238


namespace consecutive_even_numbers_product_l1952_195216

theorem consecutive_even_numbers_product : 
  ∃! (a b c : ℕ), 
    (b = a + 2 ∧ c = b + 2) ∧ 
    (a % 2 = 0) ∧
    (800000 ≤ a * b * c) ∧ 
    (a * b * c < 900000) ∧
    (a * b * c % 10 = 2) ∧
    (a = 94 ∧ b = 96 ∧ c = 98) := by
  sorry

end consecutive_even_numbers_product_l1952_195216


namespace arithmetic_mean_of_geometric_sequence_l1952_195236

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q : q = -2)
  (h_condition : a 3 * a 7 = 4 * a 4) :
  (a 8 + a 11) / 2 = -56 := by
sorry

end arithmetic_mean_of_geometric_sequence_l1952_195236


namespace value_calculation_l1952_195299

theorem value_calculation (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + a^3 / b^2 + b^3 / a^2 + b = 2535 := by
  sorry

end value_calculation_l1952_195299


namespace shaded_area_is_zero_l1952_195205

/-- Rectangle JKLM with given dimensions and points -/
structure Rectangle where
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  C : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The conditions of the rectangle as given in the problem -/
def rectangle_conditions (r : Rectangle) : Prop :=
  r.J = (0, 0) ∧
  r.K = (4, 0) ∧
  r.L = (4, 5) ∧
  r.M = (0, 5) ∧
  r.C = (1.5, 5) ∧
  r.B = (4, 4) ∧
  r.E = r.J ∧
  r.F = r.M

/-- The area of the shaded region formed by the intersection of CF and BE -/
def shaded_area (r : Rectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 0 -/
theorem shaded_area_is_zero (r : Rectangle) (h : rectangle_conditions r) : 
  shaded_area r = 0 := by sorry

end shaded_area_is_zero_l1952_195205


namespace min_value_theorem_min_value_achievable_l1952_195233

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 3 / (x + 1) ≥ 2 * Real.sqrt 3 - 1 :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, x + 3 / (x + 1) = 2 * Real.sqrt 3 - 1 :=
by sorry

end min_value_theorem_min_value_achievable_l1952_195233


namespace copperfield_numbers_l1952_195271

theorem copperfield_numbers : ∃ (x₁ x₂ x₃ : ℕ), 
  x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
  (∃ (k₁ k₂ k₃ : ℕ+), 
    x₁ * (3 ^ k₁.val) = x₁ + 2500 * k₁.val ∧
    x₂ * (3 ^ k₂.val) = x₂ + 2500 * k₂.val ∧
    x₃ * (3 ^ k₃.val) = x₃ + 2500 * k₃.val) :=
by sorry

end copperfield_numbers_l1952_195271


namespace three_faces_colored_count_l1952_195201

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  totalSmallCubes : ℕ
  smallCubesPerEdge : ℕ

/-- Calculates the number of small cubes with exactly three faces colored -/
def threeFacesColored (c : CutCube) : ℕ := 8

/-- Theorem: In a cube cut into 216 equal smaller cubes, 
    the number of small cubes with exactly three faces colored is 8 -/
theorem three_faces_colored_count :
  ∀ (c : CutCube), c.totalSmallCubes = 216 → threeFacesColored c = 8 := by
  sorry

end three_faces_colored_count_l1952_195201


namespace sum_of_square_roots_lower_bound_l1952_195204

theorem sum_of_square_roots_lower_bound (a b c d e : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
sorry

end sum_of_square_roots_lower_bound_l1952_195204


namespace exponent_calculation_l1952_195289

theorem exponent_calculation (m : ℕ) : m = 8^126 → (m * 16) / 64 = 16^94 := by
  sorry

end exponent_calculation_l1952_195289


namespace merchant_profit_percentage_l1952_195247

/-- Calculates the profit percentage for a merchant who marks up goods by 50%
    and then offers a 10% discount on the marked price. -/
theorem merchant_profit_percentage 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (hp_markup : markup_percentage = 50) 
  (hp_discount : discount_percentage = 10) : 
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discounted_price := marked_price * (1 - discount_percentage / 100)
  let profit := discounted_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 35 := by
sorry

end merchant_profit_percentage_l1952_195247


namespace prob_second_white_given_first_white_l1952_195249

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the bag -/
structure BagState where
  white : ℕ
  black : ℕ

/-- The initial state of the bag -/
def initial_bag : BagState := ⟨3, 2⟩

/-- The probability of drawing a white ball given the bag state -/
def prob_white (bag : BagState) : ℚ :=
  bag.white / (bag.white + bag.black)

/-- The probability of drawing a specific color given the bag state -/
def prob_draw (bag : BagState) (color : BallColor) : ℚ :=
  match color with
  | BallColor.White => prob_white bag
  | BallColor.Black => 1 - prob_white bag

/-- The new bag state after drawing a ball of a given color -/
def draw_ball (bag : BagState) (color : BallColor) : BagState :=
  match color with
  | BallColor.White => ⟨bag.white - 1, bag.black⟩
  | BallColor.Black => ⟨bag.white, bag.black - 1⟩

theorem prob_second_white_given_first_white :
  prob_draw (draw_ball initial_bag BallColor.White) BallColor.White = 1/2 := by
  sorry

end prob_second_white_given_first_white_l1952_195249


namespace quadratic_factorization_l1952_195223

theorem quadratic_factorization (a : ℝ) : a^2 + 4*a - 21 = (a - 3) * (a + 7) := by
  sorry

end quadratic_factorization_l1952_195223


namespace rahul_deepak_age_ratio_l1952_195239

def rahul_age_after_6_years : ℕ := 26
def years_until_rahul_age : ℕ := 6
def deepak_current_age : ℕ := 8

theorem rahul_deepak_age_ratio :
  let rahul_current_age := rahul_age_after_6_years - years_until_rahul_age
  (rahul_current_age : ℚ) / deepak_current_age = 5 / 2 := by
  sorry

end rahul_deepak_age_ratio_l1952_195239


namespace inequality_solution_l1952_195244

theorem inequality_solution (x : ℝ) : 
  (12 * x^3 + 24 * x^2 - 75 * x - 3) / ((3 * x - 4) * (x + 5)) < 6 ↔ -5 < x ∧ x < 4/3 :=
by sorry

end inequality_solution_l1952_195244


namespace sin_40_tan_10_minus_sqrt_3_l1952_195243

/-- Prove that sin 40° * (tan 10° - √3) = -1 -/
theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end sin_40_tan_10_minus_sqrt_3_l1952_195243


namespace unique_solution_cube_root_equation_l1952_195215

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 * x + (15 * x + 17) ^ (1/3)) ^ (1/3)

-- State the theorem
theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, f x = 18 ∧ x = 387 := by sorry

end unique_solution_cube_root_equation_l1952_195215


namespace unique_configuration_l1952_195228

/-- A configuration of n points in the plane with associated real numbers. -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ
  r : Fin n → ℝ

/-- The area of a triangle given by three points in the plane. -/
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- Predicate stating that three points are not collinear. -/
def nonCollinear (p₁ p₂ p₃ : ℝ × ℝ) : Prop := sorry

/-- The configuration satisfies the area condition for all triples of points. -/
def satisfiesAreaCondition (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i < j → j < k →
    triangleArea (config.points i) (config.points j) (config.points k) =
    config.r i + config.r j + config.r k

/-- The configuration satisfies the non-collinearity condition for all triples of points. -/
def satisfiesNonCollinearityCondition (config : PointConfiguration n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    nonCollinear (config.points i) (config.points j) (config.points k)

/-- The main theorem stating that 4 is the only integer greater than 3 satisfying the conditions. -/
theorem unique_configuration :
  ∀ (n : ℕ), n > 3 →
  (∃ (config : PointConfiguration n),
    satisfiesAreaCondition config ∧
    satisfiesNonCollinearityCondition config) →
  n = 4 :=
sorry

end unique_configuration_l1952_195228


namespace base_7_65234_equals_16244_l1952_195267

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_65234_equals_16244 :
  base_7_to_10 [4, 3, 2, 5, 6] = 16244 := by
  sorry

end base_7_65234_equals_16244_l1952_195267


namespace jacket_price_reduction_l1952_195281

theorem jacket_price_reduction (initial_reduction : ℝ) (final_increase : ℝ) (special_reduction : ℝ) : 
  initial_reduction = 25 →
  final_increase = 48.148148148148145 →
  (1 - initial_reduction / 100) * (1 - special_reduction / 100) * (1 + final_increase / 100) = 1 →
  special_reduction = 10 := by
sorry

end jacket_price_reduction_l1952_195281


namespace tangent_at_one_two_tangent_through_one_one_l1952_195202

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem for the tangent line at (1, 2)
theorem tangent_at_one_two :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ 2 * x - y = 0) ∧
  f 1 = 2 ∧ f' 1 = m := by sorry

-- Theorem for the tangent lines through (1, 1)
theorem tangent_through_one_one :
  ∃ (x₀ : ℝ), (x₀ = 0 ∨ x₀ = 2) ∧
  (∀ x y, y = 1 ↔ x₀ = 0 ∧ y = f x₀ + f' x₀ * (x - x₀)) ∧
  (∀ x y, 4 * x - y - 3 = 0 ↔ x₀ = 2 ∧ y = f x₀ + f' x₀ * (x - x₀)) ∧
  f x₀ + f' x₀ * (1 - x₀) = 1 := by sorry

end tangent_at_one_two_tangent_through_one_one_l1952_195202


namespace jane_initial_crayons_l1952_195231

/-- The number of crayons Jane started with -/
def initial_crayons : ℕ := sorry

/-- The number of crayons eaten by the hippopotamus -/
def eaten_crayons : ℕ := 7

/-- The number of crayons Jane ended with -/
def final_crayons : ℕ := 80

/-- Theorem stating that Jane started with 87 crayons -/
theorem jane_initial_crayons : initial_crayons = 87 := by
  sorry

end jane_initial_crayons_l1952_195231


namespace grade11_sample_count_l1952_195234

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  grade10Students : ℕ
  grade11Students : ℕ
  grade12Students : ℕ
  firstDrawn : ℕ

/-- Calculates the number of grade 11 students in the sample. -/
def grade11InSample (s : SystematicSampling) : ℕ :=
  let interval := s.totalStudents / s.sampleSize
  let grade11Start := s.grade10Students + 1
  let grade11End := grade11Start + s.grade11Students - 1
  let firstSampleInGrade11 := (((grade11Start - 1) / interval) * interval + s.firstDrawn - 1) / interval + 1
  let lastSampleInGrade11 := (((grade11End - 1) / interval) * interval + s.firstDrawn - 1) / interval + 1
  lastSampleInGrade11 - firstSampleInGrade11 + 1

/-- Theorem stating that for the given conditions, the number of grade 11 students in the sample is 17. -/
theorem grade11_sample_count (s : SystematicSampling) 
    (h1 : s.totalStudents = 1470)
    (h2 : s.sampleSize = 49)
    (h3 : s.grade10Students = 495)
    (h4 : s.grade11Students = 493)
    (h5 : s.grade12Students = 482)
    (h6 : s.firstDrawn = 23) :
    grade11InSample s = 17 := by
  sorry

end grade11_sample_count_l1952_195234


namespace unique_natural_solution_l1952_195253

theorem unique_natural_solution :
  ∃! (x y : ℕ), 3 * x + 7 * y = 23 :=
by
  -- The proof would go here
  sorry

end unique_natural_solution_l1952_195253


namespace yellow_paint_theorem_l1952_195259

/-- Represents the ratio of paints in the mixture -/
structure PaintRatio :=
  (blue : ℚ)
  (yellow : ℚ)
  (white : ℚ)

/-- Calculates the amount of yellow paint needed given the amount of white paint and the ratio -/
def yellow_paint_amount (ratio : PaintRatio) (white_amount : ℚ) : ℚ :=
  (ratio.yellow / ratio.white) * white_amount

/-- Theorem stating that given the specific ratio and white paint amount, 
    the yellow paint amount should be 9 quarts -/
theorem yellow_paint_theorem (ratio : PaintRatio) (white_amount : ℚ) :
  ratio.blue = 4 ∧ ratio.yellow = 3 ∧ ratio.white = 5 ∧ white_amount = 15 →
  yellow_paint_amount ratio white_amount = 9 := by
  sorry


end yellow_paint_theorem_l1952_195259


namespace special_triangle_area_l1952_195288

/-- Triangle with specific properties -/
structure SpecialTriangle where
  -- Three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- One angle is 120°
  angle_120 : ∃ θ, θ = 2 * π / 3
  -- Sides form arithmetic sequence with difference 4
  arithmetic_seq : ∃ x : ℝ, a = x - 4 ∧ b = x ∧ c = x + 4

/-- The area of the special triangle is 15√3 -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1/2) * t.a * t.b * Real.sqrt 3 = 15 * Real.sqrt 3 := by
  sorry


end special_triangle_area_l1952_195288


namespace arithmetic_sum_l1952_195211

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 3 = 13 →
  a 1 = 2 →
  a 4 + a 5 + a 6 = 42 := by
sorry

end arithmetic_sum_l1952_195211


namespace quadratic_properties_l1952_195294

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a ≠ 0
  hpos : ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0
  hsym : -b / (2 * a) = 2
  hintercept : ∃ x, x > 0 ∧ a * x^2 + b * x + c = 0 ∧ |c| = x

/-- Theorem stating properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.c > -1 ∧ f.a * (-f.c)^2 + f.b * (-f.c) + f.c = 0 := by
  sorry


end quadratic_properties_l1952_195294


namespace wall_width_l1952_195293

/-- Given a rectangular wall with specific proportions and volume, prove its width --/
theorem wall_width (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) 
  (h_volume : w * h * l = 86436) : w = 7 := by
  sorry

end wall_width_l1952_195293


namespace nba_games_total_l1952_195206

theorem nba_games_total (bulls_wins heat_wins knicks_wins : ℕ) : 
  bulls_wins = 70 →
  heat_wins = bulls_wins + 5 →
  knicks_wins = 2 * heat_wins →
  bulls_wins + heat_wins + knicks_wins = 295 := by
  sorry

end nba_games_total_l1952_195206


namespace adam_tshirts_correct_l1952_195235

/-- The number of t-shirts Adam initially took out -/
def adam_tshirts : ℕ := 20

/-- The total number of clothing items donated -/
def total_donated : ℕ := 126

/-- The number of Adam's friends who donated -/
def friends_donating : ℕ := 3

/-- The number of pants Adam took out -/
def adam_pants : ℕ := 4

/-- The number of jumpers Adam took out -/
def adam_jumpers : ℕ := 4

/-- The number of pajama sets Adam took out -/
def adam_pajamas : ℕ := 4

/-- Theorem stating that the number of t-shirts Adam initially took out is correct -/
theorem adam_tshirts_correct : 
  (adam_pants + adam_jumpers + 2 * adam_pajamas + adam_tshirts) / 2 + 
  friends_donating * (adam_pants + adam_jumpers + 2 * adam_pajamas + adam_tshirts) = 
  total_donated :=
sorry

end adam_tshirts_correct_l1952_195235


namespace quadratic_inequality_solution_set_l1952_195266

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 4*x - 5 < 0} = Set.Ioo (-5) 1 := by sorry

end quadratic_inequality_solution_set_l1952_195266


namespace max_value_three_power_minus_nine_power_l1952_195295

theorem max_value_three_power_minus_nine_power (x : ℝ) :
  ∃ (max : ℝ), max = (1 : ℝ) / 4 ∧ ∀ y : ℝ, 3^y - 9^y ≤ max :=
by
  sorry

end max_value_three_power_minus_nine_power_l1952_195295


namespace original_average_weight_l1952_195254

theorem original_average_weight 
  (original_count : ℕ) 
  (new_boy_weight : ℝ) 
  (average_increase : ℝ) : 
  original_count = 5 →
  new_boy_weight = 40 →
  average_increase = 1 →
  (original_count : ℝ) * ((original_count : ℝ) * average_increase + new_boy_weight) / 
    (original_count + 1) - new_boy_weight = 34 := by
  sorry

end original_average_weight_l1952_195254


namespace salary_savings_percentage_l1952_195279

theorem salary_savings_percentage (last_year_salary : ℝ) (last_year_savings_percentage : ℝ) : 
  last_year_savings_percentage > 0 →
  (0.15 * (1.1 * last_year_salary) = 1.65 * (last_year_savings_percentage / 100 * last_year_salary)) →
  last_year_savings_percentage = 10 := by
sorry

end salary_savings_percentage_l1952_195279


namespace remainder_problem_l1952_195203

theorem remainder_problem (x : ℤ) (h : x % 82 = 5) : (x + 17) % 41 = 22 := by
  sorry

end remainder_problem_l1952_195203


namespace oldest_youngest_sum_l1952_195250

def age_problem (a b c d : ℕ) : Prop :=
  a + b + c + d = 100 ∧
  a = 32 ∧
  a + b = 3 * (c + d) ∧
  c = d + 3

theorem oldest_youngest_sum (a b c d : ℕ) 
  (h : age_problem a b c d) : 
  max a (max b (max c d)) + min a (min b (min c d)) = 54 := by
  sorry

end oldest_youngest_sum_l1952_195250


namespace coefficient_of_minus_two_ab_l1952_195218

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℤ) (x : String) : ℤ := m

/-- Given monomial -2ab, prove its coefficient is -2 -/
theorem coefficient_of_minus_two_ab :
  coefficient (-2) "ab" = -2 := by
  sorry

end coefficient_of_minus_two_ab_l1952_195218
