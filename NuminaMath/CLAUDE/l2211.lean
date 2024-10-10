import Mathlib

namespace equation_solution_l2211_221105

theorem equation_solution (x : ℝ) :
  (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4) →
  x = 257 / 16 :=
by sorry

end equation_solution_l2211_221105


namespace fraction_evaluation_l2211_221147

theorem fraction_evaluation : (3 - (-3)) / (2 - 1) = 6 := by
  sorry

end fraction_evaluation_l2211_221147


namespace special_triangle_third_side_l2211_221139

/-- A triangle with two sides of lengths 2 and 3, and the third side length satisfying a quadratic equation. -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a = 2
  h2 : b = 3
  h3 : c^2 - 10*c + 21 = 0
  h4 : a + b > c ∧ b + c > a ∧ c + a > b  -- Triangle inequality

/-- The third side of the SpecialTriangle has length 3. -/
theorem special_triangle_third_side (t : SpecialTriangle) : t.c = 3 := by
  sorry

end special_triangle_third_side_l2211_221139


namespace right_triangle_area_l2211_221110

theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) : 
  h = 12 →
  α = 30 * π / 180 →
  area = 18 * Real.sqrt 3 →
  area = (1 / 2) * h * h * Real.sin α * Real.cos α :=
by sorry

end right_triangle_area_l2211_221110


namespace intersection_with_complement_l2211_221118

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_with_complement : P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_with_complement_l2211_221118


namespace extremum_properties_l2211_221155

noncomputable section

variable (x : ℝ)

def f (x : ℝ) : ℝ := x * Real.log x + (1/2) * x^2

def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, x > 0 → f x ≤ f x₀ ∨ f x ≥ f x₀

theorem extremum_properties (x₀ : ℝ) 
  (h₁ : x₀ > 0) 
  (h₂ : is_extremum_point f x₀) : 
  (0 < x₀ ∧ x₀ < Real.exp (-1)) ∧ 
  (f x₀ + x₀ < 0) := by
  sorry

end

end extremum_properties_l2211_221155


namespace largest_circle_radius_on_chessboard_l2211_221112

/-- Represents a chessboard with the usual coloring of fields -/
structure Chessboard where
  size : Nat
  is_black : Nat → Nat → Bool

/-- Represents a circle on the chessboard -/
structure Circle where
  center : Nat × Nat
  radius : Real

/-- Check if a circle intersects a white field on the chessboard -/
def intersects_white (board : Chessboard) (circle : Circle) : Bool :=
  sorry

/-- The largest possible circle radius on a chessboard without intersecting white fields -/
def largest_circle_radius (board : Chessboard) : Real :=
  sorry

/-- Theorem stating the largest possible circle radius on a standard chessboard -/
theorem largest_circle_radius_on_chessboard :
  ∀ (board : Chessboard),
    board.size = 8 →
    (∀ i j, board.is_black i j = ((i + j) % 2 = 1)) →
    largest_circle_radius board = (1 / 2) * Real.sqrt 10 :=
  sorry

end largest_circle_radius_on_chessboard_l2211_221112


namespace uncovered_area_of_rectangles_l2211_221146

theorem uncovered_area_of_rectangles (small_length small_width large_length large_width : ℝ) 
  (h1 : small_length = 4)
  (h2 : small_width = 2)
  (h3 : large_length = 10)
  (h4 : large_width = 6)
  (h5 : small_length ≤ large_length)
  (h6 : small_width ≤ large_width) :
  large_length * large_width - small_length * small_width = 52 := by
sorry

end uncovered_area_of_rectangles_l2211_221146


namespace hypotenuse_length_hypotenuse_is_four_l2211_221151

-- Define a right triangle with one angle of 15 degrees and altitude to hypotenuse of 1 cm
structure RightTriangle where
  -- One angle is 15 degrees (π/12 radians)
  angle : Real
  angle_eq : angle = Real.pi / 12
  -- The altitude to the hypotenuse is 1 cm
  altitude : Real
  altitude_eq : altitude = 1
  -- It's a right triangle (one angle is 90 degrees)
  is_right : Bool
  is_right_eq : is_right = true

-- Theorem: The hypotenuse of this triangle is 4 cm
theorem hypotenuse_length (t : RightTriangle) : Real :=
  4

-- The proof
theorem hypotenuse_is_four (t : RightTriangle) : hypotenuse_length t = 4 := by
  sorry

end hypotenuse_length_hypotenuse_is_four_l2211_221151


namespace math_team_selection_ways_l2211_221176

/-- The number of ways to select r items from n items --/
def binomial (n r : ℕ) : ℕ := Nat.choose n r

/-- The total number of students in the math club --/
def total_students : ℕ := 14

/-- The number of students to be selected for the team --/
def team_size : ℕ := 6

/-- Theorem stating that the number of ways to select the team is 3003 --/
theorem math_team_selection_ways :
  binomial total_students team_size = 3003 := by sorry

end math_team_selection_ways_l2211_221176


namespace fred_remaining_cards_l2211_221129

/-- Given that Fred initially had 40 baseball cards and Keith bought 22 of them,
    prove that Fred now has 18 baseball cards. -/
theorem fred_remaining_cards (initial_cards : ℕ) (cards_bought : ℕ) (h1 : initial_cards = 40) (h2 : cards_bought = 22) :
  initial_cards - cards_bought = 18 := by
  sorry

end fred_remaining_cards_l2211_221129


namespace sum_reciprocals_bound_l2211_221124

theorem sum_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) :
  1/x + 1/y ≥ 8 ∧ ∀ ε > 0, ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 + y'^2 = 1 ∧ 1/x' + 1/y' > 1/ε :=
sorry

end sum_reciprocals_bound_l2211_221124


namespace milk_yogurt_quantities_l2211_221120

/-- Represents the quantities and prices of milk and yogurt --/
structure MilkYogurtData where
  milk_cost : ℝ
  yogurt_cost : ℝ
  yogurt_quantity_ratio : ℝ
  price_difference : ℝ
  milk_selling_price : ℝ
  yogurt_markup : ℝ
  yogurt_discount : ℝ
  total_profit : ℝ

/-- Theorem stating the quantities of milk and discounted yogurt --/
theorem milk_yogurt_quantities (data : MilkYogurtData) 
  (h_milk_cost : data.milk_cost = 2000)
  (h_yogurt_cost : data.yogurt_cost = 4800)
  (h_ratio : data.yogurt_quantity_ratio = 1.5)
  (h_price_diff : data.price_difference = 30)
  (h_milk_price : data.milk_selling_price = 80)
  (h_yogurt_markup : data.yogurt_markup = 0.25)
  (h_yogurt_discount : data.yogurt_discount = 0.1)
  (h_total_profit : data.total_profit = 2150) :
  ∃ (milk_quantity yogurt_discounted : ℕ),
    milk_quantity = 40 ∧ yogurt_discounted = 25 := by
  sorry

end milk_yogurt_quantities_l2211_221120


namespace necessary_not_sufficient_condition_l2211_221179

theorem necessary_not_sufficient_condition :
  ∃ (x : ℝ), x ≠ 0 ∧ ¬(|2*x + 5| ≥ 7) ∧
  ∀ (y : ℝ), |2*y + 5| ≥ 7 → y ≠ 0 :=
by sorry

end necessary_not_sufficient_condition_l2211_221179


namespace pizzeria_small_pizzas_sold_l2211_221150

/-- Calculates the number of small pizzas sold given the prices, total sales, and number of large pizzas sold. -/
def small_pizzas_sold (small_price large_price total_sales : ℕ) (large_pizzas_sold : ℕ) : ℕ :=
  (total_sales - large_price * large_pizzas_sold) / small_price

/-- Theorem stating that the number of small pizzas sold is 8 under the given conditions. -/
theorem pizzeria_small_pizzas_sold :
  small_pizzas_sold 2 8 40 3 = 8 := by
  sorry

end pizzeria_small_pizzas_sold_l2211_221150


namespace storm_rainfall_calculation_l2211_221148

/-- Represents the rainfall during a storm -/
structure StormRainfall where
  first_30min : ℝ
  second_30min : ℝ
  last_hour : ℝ
  average_total : ℝ
  duration : ℝ

/-- Theorem about the rainfall during a specific storm -/
theorem storm_rainfall_calculation (storm : StormRainfall) 
  (h1 : storm.first_30min = 5)
  (h2 : storm.second_30min = storm.first_30min / 2)
  (h3 : storm.duration = 2)
  (h4 : storm.average_total = 4) :
  storm.last_hour = 0.5 := by
  sorry

end storm_rainfall_calculation_l2211_221148


namespace arithmetic_sequence_sum_equals_five_l2211_221185

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_equals_five (k : ℕ) :
  k > 0 ∧ sum_arithmetic_sequence (-3) 2 k = 5 → k = 5 := by
  sorry

end arithmetic_sequence_sum_equals_five_l2211_221185


namespace parabola_reflection_translation_sum_l2211_221187

/-- Original parabola function -/
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Reflected parabola function -/
def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := -(a * x^2 + b * x + c)

/-- Translated original parabola (3 units right) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c (x - 3)

/-- Translated reflected parabola (4 units left) -/
def g (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 4)

/-- Sum of translated original and reflected parabolas -/
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation_sum (a b c : ℝ) :
  ∀ x, f_plus_g a b c x = -14 * a * x - 19 * a - 7 * b :=
by sorry

end parabola_reflection_translation_sum_l2211_221187


namespace jake_sausage_cost_l2211_221121

/-- Calculates the total cost of sausages given the weight per package, number of packages, and price per pound -/
def total_cost (weight_per_package : ℕ) (num_packages : ℕ) (price_per_pound : ℕ) : ℕ :=
  weight_per_package * num_packages * price_per_pound

/-- Theorem: The total cost of Jake's sausage purchase is $24 -/
theorem jake_sausage_cost : total_cost 2 3 4 = 24 := by
  sorry

end jake_sausage_cost_l2211_221121


namespace coin_combination_difference_l2211_221133

def coin_values : List Nat := [5, 10, 20, 25]

def total_amount : Nat := 45

def is_valid_combination (combination : List Nat) : Prop :=
  combination.all (λ x => x ∈ coin_values) ∧
  combination.sum = total_amount

def num_coins (combination : List Nat) : Nat :=
  combination.length

theorem coin_combination_difference :
  ∃ (min_combination max_combination : List Nat),
    is_valid_combination min_combination ∧
    is_valid_combination max_combination ∧
    (∀ c, is_valid_combination c → 
      num_coins min_combination ≤ num_coins c ∧
      num_coins c ≤ num_coins max_combination) ∧
    num_coins max_combination - num_coins min_combination = 7 :=
  sorry

end coin_combination_difference_l2211_221133


namespace complex_equation_product_l2211_221189

theorem complex_equation_product (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a * b = -2 := by
  sorry

end complex_equation_product_l2211_221189


namespace cloth_length_proof_l2211_221171

/-- The length of a piece of cloth satisfying given cost conditions -/
def cloth_length : ℝ := 10

/-- The cost of the cloth -/
def total_cost : ℝ := 35

/-- The additional length in the hypothetical scenario -/
def additional_length : ℝ := 4

/-- The price reduction per meter in the hypothetical scenario -/
def price_reduction : ℝ := 1

theorem cloth_length_proof :
  cloth_length > 0 ∧
  total_cost = cloth_length * (total_cost / cloth_length) ∧
  total_cost = (cloth_length + additional_length) * (total_cost / cloth_length - price_reduction) :=
by sorry

end cloth_length_proof_l2211_221171


namespace side_b_value_triangle_area_l2211_221108

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  -- Add conditions here
  a = 3 ∧ 
  Real.cos A = Real.sqrt 6 / 3 ∧
  B = A + Real.pi / 2

-- Theorem for the value of side b
theorem side_b_value (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : b = 3 * Real.sqrt 2 := by
  sorry

-- Theorem for the area of triangle ABC
theorem triangle_area (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : 
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 := by
  sorry

end side_b_value_triangle_area_l2211_221108


namespace tom_younger_than_bob_by_three_l2211_221130

/-- Represents the ages of four siblings -/
structure SiblingAges where
  susan : ℕ
  arthur : ℕ
  tom : ℕ
  bob : ℕ

/-- The age difference between Bob and Tom -/
def ageDifference (ages : SiblingAges) : ℕ :=
  ages.bob - ages.tom

theorem tom_younger_than_bob_by_three (ages : SiblingAges) 
  (susan_age : ages.susan = 15)
  (arthur_age : ages.arthur = ages.susan + 2)
  (bob_age : ages.bob = 11)
  (total_age : ages.susan + ages.arthur + ages.tom + ages.bob = 51) :
  ageDifference ages = 3 := by
sorry

end tom_younger_than_bob_by_three_l2211_221130


namespace x_value_proof_l2211_221195

theorem x_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) : x = 7 := by
  sorry

end x_value_proof_l2211_221195


namespace exists_non_prime_combination_l2211_221152

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number has distinct digits (excluding 7)
def hasDistinctDigitsNo7 (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 9 ∧ 
  7 ∉ digits ∧
  digits.toFinset.card = 9

-- Define a function to check if any three-digit combination is prime
def anyThreeDigitsPrime (n : Nat) : Prop :=
  ∀ i j k, 0 ≤ i ∧ i < j ∧ j < k ∧ k < 9 →
    isPrime (100 * (n.digits 10).get ⟨i, by sorry⟩ + 
             10 * (n.digits 10).get ⟨j, by sorry⟩ + 
             (n.digits 10).get ⟨k, by sorry⟩)

-- The main theorem
theorem exists_non_prime_combination :
  ∃ n : Nat, hasDistinctDigitsNo7 n ∧ ¬(anyThreeDigitsPrime n) :=
sorry

end exists_non_prime_combination_l2211_221152


namespace algebraic_identities_l2211_221167

theorem algebraic_identities (a b x : ℝ) : 
  ((3 * a * b^3)^2 = 9 * a^2 * b^6) ∧ 
  (x * x^3 + x^2 * x^2 = 2 * x^4) ∧ 
  ((12 * x^4 - 6 * x^3) / (3 * x^2) = 4 * x^2 - 2 * x) := by
  sorry

end algebraic_identities_l2211_221167


namespace unique_modular_equivalence_l2211_221169

theorem unique_modular_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end unique_modular_equivalence_l2211_221169


namespace class_fund_problem_l2211_221173

theorem class_fund_problem (m : ℕ) (x : ℕ) (y : ℚ) :
  m < 400 →
  38 ≤ x →
  x < 50 →
  x * y = m →
  (x + 12) * (y - 2) = m →
  x = 42 ∧ y = 9 := by
  sorry

end class_fund_problem_l2211_221173


namespace james_total_matches_l2211_221141

/-- The number of boxes in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def james_dozens : ℕ := 5

/-- The number of matches in each box -/
def matches_per_box : ℕ := 20

/-- Theorem: James has 1200 matches in total -/
theorem james_total_matches : james_dozens * dozen * matches_per_box = 1200 := by
  sorry

end james_total_matches_l2211_221141


namespace nice_number_characterization_l2211_221142

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ := n / 1000
def second_digit (n : ℕ) : ℕ := (n / 100) % 10
def third_digit (n : ℕ) : ℕ := (n / 10) % 10
def fourth_digit (n : ℕ) : ℕ := n % 10

def digit_product (n : ℕ) : ℕ :=
  (first_digit n) * (second_digit n) * (third_digit n) * (fourth_digit n)

def is_nice (n : ℕ) : Prop :=
  is_four_digit n ∧
  first_digit n = third_digit n ∧
  second_digit n = fourth_digit n ∧
  (n * n) % (digit_product n) = 0

def nice_numbers : List ℕ := [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1212, 2424, 3636, 4848, 1515]

theorem nice_number_characterization (n : ℕ) :
  is_nice n ↔ n ∈ nice_numbers := by sorry

end nice_number_characterization_l2211_221142


namespace x_gt_y_necessary_not_sufficient_l2211_221104

theorem x_gt_y_necessary_not_sufficient (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ 
  (∃ y, x > y ∧ ¬(x > |y|)) := by
  sorry

end x_gt_y_necessary_not_sufficient_l2211_221104


namespace stamp_solution_l2211_221165

/-- Represents the stamp collection and sale problem --/
def stamp_problem (red_count blue_count : ℕ) (red_price blue_price yellow_price : ℚ) (total_goal : ℚ) : Prop :=
  let red_earnings := red_count * red_price
  let blue_earnings := blue_count * blue_price
  let remaining_earnings := total_goal - (red_earnings + blue_earnings)
  ∃ yellow_count : ℕ, yellow_count * yellow_price = remaining_earnings

/-- Theorem stating the solution to the stamp problem --/
theorem stamp_solution :
  stamp_problem 20 80 1.1 0.8 2 100 → ∃ yellow_count : ℕ, yellow_count = 7 :=
by
  sorry


end stamp_solution_l2211_221165


namespace bob_school_year_hours_l2211_221144

/-- Calculates the hours per week Bob needs to work during the school year --/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := school_year_earnings / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that Bob needs to work 15 hours per week during the school year --/
theorem bob_school_year_hours : 
  school_year_hours_per_week 8 45 3600 24 3600 = 15 := by sorry

end bob_school_year_hours_l2211_221144


namespace bianca_carrots_l2211_221101

/-- The number of carrots Bianca picked on the first day -/
def first_day_carrots : ℕ := 23

/-- The number of carrots Bianca threw out after the first day -/
def thrown_out_carrots : ℕ := 10

/-- The number of carrots Bianca picked on the second day -/
def second_day_carrots : ℕ := 47

/-- The total number of carrots Bianca has at the end -/
def total_carrots : ℕ := 60

theorem bianca_carrots :
  first_day_carrots - thrown_out_carrots + second_day_carrots = total_carrots :=
by sorry

end bianca_carrots_l2211_221101


namespace pie_eating_contest_l2211_221109

theorem pie_eating_contest (first_student_total : ℚ) (second_student_total : ℚ) 
  (third_student_first_pie : ℚ) (third_student_second_pie : ℚ) :
  first_student_total = 7/6 ∧ 
  second_student_total = 4/3 ∧ 
  third_student_first_pie = 1/2 ∧ 
  third_student_second_pie = 1/3 →
  (first_student_total - third_student_first_pie * first_student_total = 2/3) ∧
  (second_student_total - third_student_second_pie * second_student_total = 1) ∧
  (third_student_first_pie * first_student_total + third_student_second_pie * second_student_total = 5/6) :=
by sorry

end pie_eating_contest_l2211_221109


namespace regular_polygon_sides_l2211_221172

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) / n = 150 → n = 12 := by
  sorry

end regular_polygon_sides_l2211_221172


namespace money_division_l2211_221181

theorem money_division (a b c : ℚ) :
  a = (1/2 : ℚ) * (b + c) →
  b = (2/3 : ℚ) * (a + c) →
  a = 122 →
  a + b + c = 366 := by
sorry

end money_division_l2211_221181


namespace smallest_base_for_62_l2211_221166

theorem smallest_base_for_62 : 
  ∃ (b : ℕ), b = 4 ∧ 
  (∀ (x : ℕ), x < b → ¬(b^2 ≤ 62 ∧ 62 < b^3)) ∧
  (b^2 ≤ 62 ∧ 62 < b^3) := by
sorry

end smallest_base_for_62_l2211_221166


namespace complete_square_with_integer_l2211_221174

theorem complete_square_with_integer (x : ℝ) : 
  ∃ (k : ℤ) (a : ℝ), x^2 - 6*x + 20 = (x - a)^2 + k ∧ k = 11 := by
  sorry

end complete_square_with_integer_l2211_221174


namespace expression_bounds_l2211_221168

theorem expression_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) 
    (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry

end expression_bounds_l2211_221168


namespace lemonade_theorem_l2211_221182

/-- Represents the number of glasses of lemonade that can be made -/
def lemonade_glasses (lemons oranges grapefruits : ℕ) : ℕ :=
  let lemon_glasses := lemons / 2
  let orange_glasses := oranges
  let citrus_glasses := min lemon_glasses orange_glasses
  let grapefruit_glasses := grapefruits
  citrus_glasses + grapefruit_glasses

/-- Theorem stating that with given ingredients, 15 glasses of lemonade can be made -/
theorem lemonade_theorem : lemonade_glasses 18 10 6 = 15 := by
  sorry

end lemonade_theorem_l2211_221182


namespace additional_cars_needed_l2211_221114

def cars_per_row : ℕ := 8
def current_cars : ℕ := 37

theorem additional_cars_needed : 
  ∃ (n : ℕ), (n > 0) ∧ (cars_per_row * n ≥ current_cars) ∧ 
  (cars_per_row * n - current_cars = 3) ∧
  (∀ m : ℕ, m > 0 → cars_per_row * m ≥ current_cars → 
    cars_per_row * m - current_cars ≥ 3) :=
by sorry

end additional_cars_needed_l2211_221114


namespace geometric_sequence_term_count_l2211_221198

theorem geometric_sequence_term_count (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 1/2 →
  q = 1/2 →
  aₙ = 1/32 →
  aₙ = a₁ * q^(n-1) →
  n = 5 := by
sorry

end geometric_sequence_term_count_l2211_221198


namespace journey_time_increase_l2211_221191

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 ∧ 
  first_half_speed = 80 ∧ 
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time = 2 := by
sorry

end journey_time_increase_l2211_221191


namespace pure_imaginary_complex_number_l2211_221194

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x^2 + 3*x + 2)*I = (0 : ℂ) + y*I ∧ y ≠ 0) → x = 1 :=
by sorry

end pure_imaginary_complex_number_l2211_221194


namespace sum_and_diff_expectations_l2211_221190

variable (X Y : ℝ → ℝ)

-- Define the expectation operator
def expectation (Z : ℝ → ℝ) : ℝ := sorry

-- Given conditions
axiom X_expectation : expectation X = 3
axiom Y_expectation : expectation Y = 2

-- Linearity of expectation
axiom expectation_sum (Z W : ℝ → ℝ) : expectation (Z + W) = expectation Z + expectation W
axiom expectation_diff (Z W : ℝ → ℝ) : expectation (Z - W) = expectation Z - expectation W

-- Theorem to prove
theorem sum_and_diff_expectations :
  expectation (X + Y) = 5 ∧ expectation (X - Y) = 1 := by sorry

end sum_and_diff_expectations_l2211_221190


namespace compass_leg_swap_impossible_l2211_221170

/-- Represents a point on the integer grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the state of the compass -/
structure CompassState where
  leg1 : GridPoint
  leg2 : GridPoint

/-- Calculates the squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Defines a valid move of the compass -/
def isValidMove (s1 s2 : CompassState) : Prop :=
  (s1.leg1 = s2.leg1 ∧ squaredDistance s1.leg1 s1.leg2 = squaredDistance s2.leg1 s2.leg2) ∨
  (s1.leg2 = s2.leg2 ∧ squaredDistance s1.leg1 s1.leg2 = squaredDistance s2.leg1 s2.leg2)

/-- Defines a sequence of valid moves -/
def isValidMoveSequence : List CompassState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

theorem compass_leg_swap_impossible (start finish : CompassState) 
  (h_start_distance : squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 finish.leg2)
  (h_swap : start.leg1 = finish.leg2 ∧ start.leg2 = finish.leg1) :
  ¬∃ (moves : List CompassState), isValidMoveSequence (start :: moves ++ [finish]) :=
sorry

end compass_leg_swap_impossible_l2211_221170


namespace geometric_series_common_ratio_l2211_221180

/-- The common ratio of the infinite geometric series 8/10 - 6/15 + 36/225 - ... is -1/2 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -6 / 15
  let a₃ : ℚ := 36 / 225
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -1 / 2 := by
  sorry

end geometric_series_common_ratio_l2211_221180


namespace coffee_shrink_theorem_l2211_221164

/-- Represents the shrink ray effect on volume --/
def shrinkEffect : ℝ := 0.5

/-- Number of coffee cups --/
def numCups : ℕ := 5

/-- Initial volume of coffee in each cup (in ounces) --/
def initialVolume : ℝ := 8

/-- Calculates the total volume of coffee after shrinking --/
def totalVolumeAfterShrink (shrinkEffect : ℝ) (numCups : ℕ) (initialVolume : ℝ) : ℝ :=
  (shrinkEffect * initialVolume) * numCups

/-- Theorem stating that the total volume of coffee after shrinking is 20 ounces --/
theorem coffee_shrink_theorem : 
  totalVolumeAfterShrink shrinkEffect numCups initialVolume = 20 := by
  sorry

end coffee_shrink_theorem_l2211_221164


namespace turtle_combination_probability_l2211_221186

/- Define the number of initial turtles -/
def initial_turtles : ℕ := 2017

/- Define the number of combinations -/
def combinations : ℕ := 2015

/- Function to calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/- Probability that a specific turtle is never chosen -/
def prob_never_chosen : ℚ := 1 / (binomial initial_turtles 2)

/- Theorem statement -/
theorem turtle_combination_probability :
  (initial_turtles : ℚ) * prob_never_chosen = 1 / 1008 :=
sorry

end turtle_combination_probability_l2211_221186


namespace circle_center_coordinates_l2211_221117

-- Define the lines and circle
def line_l2 (x y : ℝ) : Prop := x + 3*y + 1 = 0
def line_l1 (x y : ℝ) : Prop := 3*x - y = 0
def circle_C (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y = 1 - 2*a^2

-- Define the theorem
theorem circle_center_coordinates (a : ℝ) :
  a > 0 →
  (∃ M N : ℝ × ℝ, line_l1 M.1 M.2 ∧ line_l1 N.1 N.2 ∧ circle_C M.1 M.2 a ∧ circle_C N.1 N.2 a) →
  (∀ x y : ℝ, line_l2 x y → (∀ u v : ℝ, line_l1 u v → u*x + v*y = 0)) →
  (∃ C : ℝ × ℝ, C.1 = a ∧ C.2 = a ∧ circle_C C.1 C.2 a) →
  (∃ M N : ℝ × ℝ, line_l1 M.1 M.2 ∧ line_l1 N.1 N.2 ∧ circle_C M.1 M.2 a ∧ circle_C N.1 N.2 a ∧
    (M.1 - a) * (N.1 - a) + (M.2 - a) * (N.2 - a) = 0) →
  a = Real.sqrt 5 / 2 :=
sorry

end circle_center_coordinates_l2211_221117


namespace betting_game_result_l2211_221128

theorem betting_game_result (initial_amount : ℚ) (num_bets num_wins num_losses : ℕ) 
  (h1 : initial_amount = 64)
  (h2 : num_bets = 6)
  (h3 : num_wins = 3)
  (h4 : num_losses = 3)
  (h5 : num_wins + num_losses = num_bets) :
  let final_amount := initial_amount * (3/2)^num_wins * (1/2)^num_losses
  final_amount = 27 ∧ initial_amount - final_amount = 37 := by
  sorry

#eval (64 : ℚ) * (3/2)^3 * (1/2)^3

end betting_game_result_l2211_221128


namespace root_values_l2211_221107

theorem root_values (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^4 + b * k^3 + c * k^2 + d * k + a = 0)
  (h2 : a * k^3 + b * k^2 + c * k + d = 0) :
  k = Complex.I^(1/4) ∨ k = -Complex.I^(1/4) ∨ k = Complex.I^(3/4) ∨ k = -Complex.I^(3/4) :=
sorry

end root_values_l2211_221107


namespace child_ticket_cost_l2211_221161

theorem child_ticket_cost (total_seats : ℕ) (adult_ticket_cost : ℚ) 
  (num_children : ℕ) (total_revenue : ℚ) :
  total_seats = 250 →
  adult_ticket_cost = 6 →
  num_children = 188 →
  total_revenue = 1124 →
  ∃ (child_ticket_cost : ℚ),
    child_ticket_cost * num_children + 
    adult_ticket_cost * (total_seats - num_children) = total_revenue ∧
    child_ticket_cost = 4 :=
by sorry

end child_ticket_cost_l2211_221161


namespace permutations_5_3_l2211_221113

/-- The number of permutations of k elements chosen from n elements -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem: The number of permutations A_5^3 equals 60 -/
theorem permutations_5_3 : permutations 5 3 = 60 := by
  sorry

end permutations_5_3_l2211_221113


namespace book_price_l2211_221163

theorem book_price (price : ℝ) : price = 1 + (1/3) * price → price = 1.5 := by
  sorry

end book_price_l2211_221163


namespace sequence_difference_theorem_l2211_221197

theorem sequence_difference_theorem (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) 
  (h3 : ∀ n : ℕ, a n < a (n + 1)) :
  ∀ n : ℕ, ∃ p q : ℕ, a p - a q = n :=
by sorry

end sequence_difference_theorem_l2211_221197


namespace line_ellipse_intersection_l2211_221122

/-- The line y - kx - 1 = 0 always has a common point with the ellipse x²/5 + y²/m = 1 
    for all real k if and only if m ∈ [1,5) ∪ (5,+∞) -/
theorem line_ellipse_intersection (m : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y - k*x - 1 = 0 ∧ x^2/5 + y^2/m = 1) ↔ 
  (m ∈ Set.Icc 1 5 ∪ Set.Ioi 5) ∧ m ≠ 5 :=
sorry

end line_ellipse_intersection_l2211_221122


namespace invertible_product_l2211_221106

def is_invertible (f : ℕ → Bool) : Prop := f 1 = false ∧ f 2 = true ∧ f 3 = true ∧ f 4 = true

theorem invertible_product (f : ℕ → Bool) (h : is_invertible f) :
  (List.filter (λ i => f i) [1, 2, 3, 4]).prod = 24 := by
  sorry

end invertible_product_l2211_221106


namespace product_divisibility_probability_l2211_221192

/-- The number of dice rolled -/
def n : ℕ := 8

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The probability that a single die roll is even -/
def p_even : ℚ := 1/2

/-- The probability that a single die roll is divisible by 3 -/
def p_div3 : ℚ := 1/3

/-- The probability that the product of n dice rolls is divisible by both 4 and 3 -/
def prob_div_4_and_3 : ℚ := 1554975/1679616

theorem product_divisibility_probability :
  (1 - (1 - (1 - (1 - p_even)^n - n * (1 - p_even)^(n-1) * p_even))) *
  (1 - (1 - p_div3)^n) = prob_div_4_and_3 := by
  sorry

end product_divisibility_probability_l2211_221192


namespace stream_speed_calculation_l2211_221156

/-- Given a boat traveling downstream, calculate the speed of the stream. -/
theorem stream_speed_calculation 
  (boat_speed : ℝ)           -- Speed of the boat in still water
  (distance : ℝ)             -- Distance traveled downstream
  (time : ℝ)                 -- Time taken to travel downstream
  (h1 : boat_speed = 5)      -- Boat speed is 5 km/hr
  (h2 : distance = 100)      -- Distance is 100 km
  (h3 : time = 10)           -- Time taken is 10 hours
  : ∃ (stream_speed : ℝ), 
    stream_speed = 5 ∧ 
    distance = (boat_speed + stream_speed) * time :=
by sorry


end stream_speed_calculation_l2211_221156


namespace f_geq_a_implies_a_leq_2_l2211_221162

/-- The function f(x) = x^2 - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

/-- The theorem stating that if f(x) ≥ a for all x ∈ [-1, +∞), then a ≤ 2 -/
theorem f_geq_a_implies_a_leq_2 (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → a ≤ 2 :=
by sorry

end f_geq_a_implies_a_leq_2_l2211_221162


namespace circle_ratio_l2211_221184

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * (π * r₁^2)) :
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end circle_ratio_l2211_221184


namespace boys_to_girls_ratio_l2211_221127

/-- Proves that in a class of 27 students with 15 girls, the ratio of boys to girls is 4:5 -/
theorem boys_to_girls_ratio (total_students : Nat) (girls : Nat) 
  (h1 : total_students = 27) 
  (h2 : girls = 15) : 
  (total_students - girls) * 5 = girls * 4 := by
  sorry

end boys_to_girls_ratio_l2211_221127


namespace recipe_cereal_cups_l2211_221119

/-- Given a recipe calling for 18.0 servings of cereal, where each serving is 2.0 cups,
    the total number of cups needed is 36.0. -/
theorem recipe_cereal_cups : 
  let servings : ℝ := 18.0
  let cups_per_serving : ℝ := 2.0
  servings * cups_per_serving = 36.0 := by
sorry

end recipe_cereal_cups_l2211_221119


namespace log_equation_solution_l2211_221193

theorem log_equation_solution (x : ℝ) :
  Real.log (x + 8) / Real.log 8 = 3/2 → x = 8 * (2 * Real.sqrt 2 - 1) := by
  sorry

end log_equation_solution_l2211_221193


namespace proposition_false_iff_a_in_range_l2211_221134

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) ↔ (a < -1 ∨ a > 3) := by
  sorry

end proposition_false_iff_a_in_range_l2211_221134


namespace four_digit_decimal_problem_l2211_221137

theorem four_digit_decimal_problem :
  ∃ (x : ℕ), 
    (1000 ≤ x ∧ x < 10000) ∧
    ((x : ℝ) - (x : ℝ) / 10 = 2059.2 ∨ (x : ℝ) - (x : ℝ) / 100 = 2059.2) ∧
    (x = 2288 ∨ x = 2080) :=
by sorry

end four_digit_decimal_problem_l2211_221137


namespace tree_increase_l2211_221158

theorem tree_increase (initial_trees : ℕ) (increase_percentage : ℚ) : 
  initial_trees = 120 →
  increase_percentage = 5.5 / 100 →
  initial_trees + ⌊(increase_percentage * initial_trees : ℚ)⌋ = 126 := by
sorry

end tree_increase_l2211_221158


namespace cake_muffin_buyers_l2211_221116

theorem cake_muffin_buyers (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : cake_buyers = 50) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 16) (h4 : prob_neither = 26/100) : 
  ∃ total_buyers : ℕ, 
    (total_buyers : ℚ) - ((cake_buyers : ℚ) + (muffin_buyers : ℚ) - (both_buyers : ℚ)) = 
    prob_neither * (total_buyers : ℚ) ∧ total_buyers = 100 := by
  sorry

end cake_muffin_buyers_l2211_221116


namespace triangle_transformation_l2211_221159

-- Define the points of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (8, 9)
def C : ℝ × ℝ := (-3, 7)

-- Define the points of the transformed triangle
def A' : ℝ × ℝ := (-2, -6)
def B' : ℝ × ℝ := (-7, -11)
def C' : ℝ × ℝ := (2, -9)

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-(x - 0.5) - 5.5, -y - 2)

-- Theorem stating that the transformation maps the original triangle to the new one
theorem triangle_transformation :
  transform A = A' ∧ transform B = B' ∧ transform C = C' := by
  sorry


end triangle_transformation_l2211_221159


namespace mod_sum_powers_seven_l2211_221196

theorem mod_sum_powers_seven : (45^1234 + 27^1234) % 7 = 5 := by
  sorry

end mod_sum_powers_seven_l2211_221196


namespace am_gm_inequality_l2211_221154

theorem am_gm_inequality (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c < d) :
  ((c + d) / 2 - Real.sqrt (c * d)) < (d - c)^3 / (8 * c) := by
  sorry

end am_gm_inequality_l2211_221154


namespace percentage_problem_l2211_221138

/-- The percentage P that satisfies the equation (1/10 * 8000) - (P/100 * 8000) = 796 -/
theorem percentage_problem (P : ℝ) : (1/10 * 8000) - (P/100 * 8000) = 796 ↔ P = 5 := by
  sorry

end percentage_problem_l2211_221138


namespace race_time_A_l2211_221136

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runnerA : Runner
  runnerB : Runner
  timeDiff : ℝ
  distanceDiff : ℝ

/-- The main theorem that proves the race time for runner A -/
theorem race_time_A (race : Race) (h1 : race.distance = 1000) 
    (h2 : race.timeDiff = 10) (h3 : race.distanceDiff = 25) : 
    race.distance / race.runnerA.speed = 390 := by
  sorry

#check race_time_A

end race_time_A_l2211_221136


namespace ellipse_equation_correct_l2211_221103

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if two points are foci of an ellipse -/
def areFoci (f1 f2 : Point) (e : Ellipse) : Prop :=
  (f2.x - f1.x)^2 / 4 = e.a^2 - e.b^2

theorem ellipse_equation_correct (P A B : Point) (E : Ellipse) :
  P.x = 5/2 ∧ P.y = -3/2 ∧
  A.x = -2 ∧ A.y = 0 ∧
  B.x = 2 ∧ B.y = 0 ∧
  E.a^2 = 10 ∧ E.b^2 = 6 →
  pointOnEllipse P E ∧ areFoci A B E := by
  sorry

#check ellipse_equation_correct

end ellipse_equation_correct_l2211_221103


namespace base7_to_base10_5326_l2211_221160

def base7ToBase10 (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem base7_to_base10_5326 : base7ToBase10 5 3 2 6 = 1882 := by
  sorry

end base7_to_base10_5326_l2211_221160


namespace average_of_eleven_numbers_l2211_221177

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 88 →
  last_six_avg = 65 →
  sixth_number = 258 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 := by
  sorry

end average_of_eleven_numbers_l2211_221177


namespace least_addition_for_divisibility_l2211_221188

theorem least_addition_for_divisibility (n m k : ℕ) (h : n + k = m * 29) : 
  ∀ j : ℕ, j < k → ¬(∃ l : ℕ, n + j = l * 29) :=
by
  sorry

#check least_addition_for_divisibility 1056 37 17

end least_addition_for_divisibility_l2211_221188


namespace complex_magnitude_inequality_l2211_221125

theorem complex_magnitude_inequality (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := -2 + Complex.I
  Complex.abs z₁ < Complex.abs z₂ → -1 < a ∧ a < 1 := by
sorry

end complex_magnitude_inequality_l2211_221125


namespace grape_rate_proof_l2211_221149

/-- The rate of grapes per kilogram -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kilograms -/
def grape_amount : ℝ := 8

/-- The rate of mangoes per kilogram -/
def mango_rate : ℝ := 60

/-- The amount of mangoes purchased in kilograms -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 1100

theorem grape_rate_proof : 
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end grape_rate_proof_l2211_221149


namespace first_expression_value_l2211_221115

theorem first_expression_value (E a : ℝ) : 
  (E + (3 * a - 8)) / 2 = 84 → a = 32 → E = 80 := by
  sorry

end first_expression_value_l2211_221115


namespace weekend_rain_probability_l2211_221145

theorem weekend_rain_probability (p_sat p_sun : ℝ) 
  (h_sat : p_sat = 0.6) 
  (h_sun : p_sun = 0.7) 
  (h_independent : True) -- We don't need to express independence in the statement
  : 1 - (1 - p_sat) * (1 - p_sun) = 0.88 := by
  sorry

end weekend_rain_probability_l2211_221145


namespace maria_sheets_problem_l2211_221178

/-- The number of sheets in Maria's desk -/
def sheets_in_desk : ℕ := sorry

/-- The number of sheets in Maria's backpack -/
def sheets_in_backpack : ℕ := sorry

/-- The total number of sheets Maria has -/
def total_sheets : ℕ := 91

theorem maria_sheets_problem :
  (sheets_in_backpack = sheets_in_desk + 41) →
  (total_sheets = sheets_in_desk + sheets_in_backpack) →
  sheets_in_desk = 25 := by sorry

end maria_sheets_problem_l2211_221178


namespace milk_liters_bought_l2211_221199

/-- Given the costs of ingredients and the total cost, prove the number of liters of milk bought. -/
theorem milk_liters_bought (flour_boxes : ℕ) (flour_cost : ℕ) (egg_trays : ℕ) (egg_cost : ℕ)
  (milk_cost : ℕ) (soda_boxes : ℕ) (soda_cost : ℕ) (total_cost : ℕ)
  (h1 : flour_boxes = 3) (h2 : flour_cost = 3) (h3 : egg_trays = 3) (h4 : egg_cost = 10)
  (h5 : milk_cost = 5) (h6 : soda_boxes = 2) (h7 : soda_cost = 3) (h8 : total_cost = 80) :
  (total_cost - (flour_boxes * flour_cost + egg_trays * egg_cost + soda_boxes * soda_cost)) / milk_cost = 7 := by
  sorry

end milk_liters_bought_l2211_221199


namespace wood_length_problem_l2211_221126

theorem wood_length_problem (first_set second_set : ℝ) :
  second_set = 5 * first_set →
  second_set = 20 →
  first_set = 4 :=
by
  sorry

end wood_length_problem_l2211_221126


namespace equal_output_day_l2211_221132

def initial_output_A : ℝ := 200
def daily_output_A : ℝ := 20
def daily_output_B : ℝ := 30

def total_output_A (days : ℝ) : ℝ := initial_output_A + daily_output_A * days
def total_output_B (days : ℝ) : ℝ := daily_output_B * days

theorem equal_output_day : 
  ∃ (day : ℝ), day > 0 ∧ total_output_A day = total_output_B day ∧ day = 20 :=
sorry

end equal_output_day_l2211_221132


namespace expected_squares_under_attack_l2211_221102

/-- The number of squares on a chessboard -/
def board_size : ℕ := 64

/-- The number of rooks placed on the board -/
def num_rooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def prob_not_attacked_by_one : ℚ := 49 / 64

/-- The expected number of squares under attack by three randomly placed rooks on a chessboard -/
theorem expected_squares_under_attack :
  let prob_attacked := 1 - prob_not_attacked_by_one ^ num_rooks
  (board_size : ℚ) * prob_attacked = 64 * (1 - (49/64)^3) :=
sorry

end expected_squares_under_attack_l2211_221102


namespace existence_of_two_integers_l2211_221183

theorem existence_of_two_integers (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ q₁ q₂ : ℕ, q₁ ≠ q₂ ∧
    1 ≤ q₁ ∧ q₁ ≤ p - 1 ∧
    1 ≤ q₂ ∧ q₂ ≤ p - 1 ∧
    (q₁^(p-1) : ℤ) % p^2 = 1 ∧
    (q₂^(p-1) : ℤ) % p^2 = 1 :=
by sorry

end existence_of_two_integers_l2211_221183


namespace train_length_calculation_l2211_221153

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 180

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Theorem stating that under given conditions, the train length is 1500 meters -/
theorem train_length_calculation (train_length platform_length : ℝ) 
  (h1 : train_length = platform_length) 
  (h2 : train_speed * (1000 / 60) * crossing_time = 2 * train_length) : 
  train_length = 1500 := by
  sorry

end train_length_calculation_l2211_221153


namespace fundraising_total_l2211_221111

def total_donations (initial_donors : ℕ) (initial_average : ℕ) (days : ℕ) : ℕ :=
  let donor_counts := List.range days |>.map (fun i => initial_donors * 2^i)
  let daily_averages := List.range days |>.map (fun i => initial_average + 5 * i)
  (List.zip donor_counts daily_averages).map (fun (d, a) => d * a) |>.sum

theorem fundraising_total :
  total_donations 10 10 5 = 8000 := by
  sorry

end fundraising_total_l2211_221111


namespace remainder_5032_div_28_l2211_221175

theorem remainder_5032_div_28 : 5032 % 28 = 20 := by
  sorry

end remainder_5032_div_28_l2211_221175


namespace forty_platforms_required_l2211_221157

/-- The minimum number of platforms required to transport granite slabs -/
def min_platforms (num_slabs_7ton : ℕ) (num_slabs_9ton : ℕ) (max_platform_capacity : ℕ) : ℕ :=
  let total_weight := num_slabs_7ton * 7 + num_slabs_9ton * 9
  (total_weight + max_platform_capacity - 1) / max_platform_capacity

/-- Theorem stating that 40 platforms are required for the given conditions -/
theorem forty_platforms_required :
  min_platforms 120 80 40 = 40 ∧
  ∀ n : ℕ, n < 40 → ¬ (120 * 7 + 80 * 9 ≤ n * 40) :=
by sorry

end forty_platforms_required_l2211_221157


namespace pie_fraction_not_eaten_l2211_221123

theorem pie_fraction_not_eaten
  (lara_ate : ℚ)
  (ryan_ate : ℚ)
  (cassie_ate_remaining : ℚ)
  (h1 : lara_ate = 1/4)
  (h2 : ryan_ate = 3/10)
  (h3 : cassie_ate_remaining = 2/3)
  : 1 - (lara_ate + ryan_ate + cassie_ate_remaining * (1 - lara_ate - ryan_ate)) = 3/20 := by
  sorry

#check pie_fraction_not_eaten

end pie_fraction_not_eaten_l2211_221123


namespace pencil_notebook_cost_l2211_221140

/-- The cost of pencils and notebooks -/
theorem pencil_notebook_cost : 
  ∀ (p n : ℝ), 
  3 * p + 4 * n = 60 →
  p + n = 15.512820512820513 →
  96 * p + 24 * n = 520 := by
sorry

end pencil_notebook_cost_l2211_221140


namespace sine_product_identity_l2211_221143

theorem sine_product_identity : 
  (Real.sin (10 * π / 180)) * (Real.sin (30 * π / 180)) * 
  (Real.sin (50 * π / 180)) * (Real.sin (70 * π / 180)) = 1/16 := by
sorry

end sine_product_identity_l2211_221143


namespace new_person_weight_l2211_221100

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 65 ∧ 
  avg_increase = 2.5 →
  replaced_weight + n * avg_increase = 85 := by
  sorry

end new_person_weight_l2211_221100


namespace triangle_translation_l2211_221135

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation2D) (p : Point2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem triangle_translation :
  let A : Point2D := { x := 0, y := 2 }
  let B : Point2D := { x := 2, y := -1 }
  let A' : Point2D := { x := -1, y := 0 }
  let t : Translation2D := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' : Point2D := applyTranslation t B
  B'.x = 1 ∧ B'.y = -3 := by sorry

end triangle_translation_l2211_221135


namespace product_plus_one_is_square_l2211_221131

theorem product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 := by
  sorry

end product_plus_one_is_square_l2211_221131
