import Mathlib

namespace largest_cube_surface_area_l311_31155

/-- Given a cuboid with dimensions 12 cm, 16 cm, and 14 cm, 
    the surface area of the largest cube that can be cut from it is 864 cm^2 -/
theorem largest_cube_surface_area 
  (width : ℝ) (length : ℝ) (height : ℝ)
  (h_width : width = 12)
  (h_length : length = 16)
  (h_height : height = 14) :
  6 * (min width (min length height))^2 = 864 := by
  sorry

end largest_cube_surface_area_l311_31155


namespace cubic_equation_roots_l311_31184

theorem cubic_equation_roots (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 - Complex.I : ℂ) ^ 3 + p * (2 - Complex.I : ℂ) ^ 2 + q * (2 - Complex.I : ℂ) - 6 = 0 →
  p = -26/5 ∧ q = 49/5 := by
  sorry

end cubic_equation_roots_l311_31184


namespace xiaoming_walking_speed_l311_31199

theorem xiaoming_walking_speed (distance : ℝ) (min_time max_time : ℝ) (h1 : distance = 3500)
  (h2 : min_time = 40) (h3 : max_time = 50) :
  let speed_range := {x : ℝ | distance / max_time ≤ x ∧ x ≤ distance / min_time}
  ∀ x ∈ speed_range, 70 ≤ x ∧ x ≤ 87.5 :=
by sorry

end xiaoming_walking_speed_l311_31199


namespace rectangular_garden_area_l311_31156

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- The area of a rectangular garden -/
def area (g : RectangularGarden) : ℝ := g.width * g.length

/-- Theorem: The area of a rectangular garden with width 16 meters and length three times its width is 768 square meters -/
theorem rectangular_garden_area : 
  ∀ (g : RectangularGarden), 
  g.width = 16 → 
  g.length = 3 * g.width → 
  area g = 768 := by
sorry

end rectangular_garden_area_l311_31156


namespace claires_weight_l311_31174

theorem claires_weight (alice_weight claire_weight : ℚ) : 
  alice_weight + claire_weight = 200 →
  claire_weight - alice_weight = claire_weight / 3 →
  claire_weight = 1400 / 9 := by
sorry

end claires_weight_l311_31174


namespace parallel_vectors_x_value_l311_31119

/-- Given two parallel vectors a and b, prove that x = 1/2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (∃ (k : ℝ), a = k • b) →
  x = 1/2 := by
sorry

end parallel_vectors_x_value_l311_31119


namespace unique_prime_product_sum_l311_31102

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def distinct_primes (p q r : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r

theorem unique_prime_product_sum (p q r : ℕ) : 
  5401 = p * q * r → 
  distinct_primes p q r →
  ∃! n : ℕ, ∃ p1 p2 p3 : ℕ, 
    n = p1 * p2 * p3 ∧ 
    distinct_primes p1 p2 p3 ∧ 
    p1 + p2 + p3 = p + q + r ∧
    n ≠ 5401 :=
sorry

end unique_prime_product_sum_l311_31102


namespace equation_solutions_l311_31121

theorem equation_solutions :
  (∀ x y : ℤ, y^4 + 2*x^4 + 1 = 4*x^2*y ↔ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = 1)) ∧
  (∀ x y z : ℕ+, 5*(x*y + y*z + z*x) = 4*x*y*z ↔
    ((x = 5 ∧ y = 10 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 10) ∨
     (x = 10 ∧ y = 5 ∧ z = 2) ∨ (x = 10 ∧ y = 2 ∧ z = 5) ∨
     (x = 2 ∧ y = 10 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 10) ∨
     (x = 4 ∧ y = 20 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 20) ∨
     (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 20 ∧ z = 4) ∨
     (x = 20 ∧ y = 2 ∧ z = 4) ∨ (x = 20 ∧ y = 4 ∧ z = 2))) := by
  sorry


end equation_solutions_l311_31121


namespace min_sum_of_squares_with_diff_l311_31134

theorem min_sum_of_squares_with_diff (x y : ℕ+) : 
  x.val^2 - y.val^2 = 145 → x.val^2 + y.val^2 ≥ 433 := by
sorry

end min_sum_of_squares_with_diff_l311_31134


namespace non_holiday_rate_correct_l311_31124

/-- The number of customers per hour during the non-holiday season -/
def non_holiday_rate : ℕ := 175

/-- The number of customers per hour during the holiday season -/
def holiday_rate : ℕ := non_holiday_rate * 2

/-- The total number of customers during the holiday season -/
def total_customers : ℕ := 2800

/-- The number of hours observed during the holiday season -/
def observation_hours : ℕ := 8

/-- Theorem stating that the non-holiday rate is correct given the conditions -/
theorem non_holiday_rate_correct : 
  holiday_rate * observation_hours = total_customers ∧
  non_holiday_rate = 175 := by
  sorry

end non_holiday_rate_correct_l311_31124


namespace printer_cost_l311_31157

/-- The cost of a single printer given the total cost of keyboards and printers, 
    the number of each item, and the cost of a single keyboard. -/
theorem printer_cost 
  (total_cost : ℕ) 
  (num_keyboards num_printers : ℕ) 
  (keyboard_cost : ℕ) 
  (h1 : total_cost = 2050)
  (h2 : num_keyboards = 15)
  (h3 : num_printers = 25)
  (h4 : keyboard_cost = 20) :
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 :=
by sorry

end printer_cost_l311_31157


namespace johnson_family_seating_l311_31170

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem johnson_family_seating (boys girls : ℕ) (total : ℕ) :
  boys = 5 →
  girls = 4 →
  total = boys + girls →
  factorial total - (factorial boys * factorial girls) = 360000 :=
by
  sorry

end johnson_family_seating_l311_31170


namespace opposite_of_negative_mixed_number_l311_31160

theorem opposite_of_negative_mixed_number : 
  -(-(7/4)) = 7/4 := by sorry

end opposite_of_negative_mixed_number_l311_31160


namespace perfect_square_condition_l311_31129

theorem perfect_square_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m*x + 1 = y^2) → (m = 2 ∨ m = -2) := by
  sorry

end perfect_square_condition_l311_31129


namespace function_transformation_l311_31130

-- Define the given function
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem function_transformation :
  (∀ x, f (2 * x + 1) = x^2 - 2*x) →
  (∀ x, f x = x^2 / 4 - (3/2) * x + 5/4) := by sorry

end function_transformation_l311_31130


namespace laborer_income_proof_l311_31163

/-- Represents the monthly income of a laborer -/
def monthly_income : ℝ := 75

/-- Represents the debt after 6 months -/
def debt : ℝ := 30

theorem laborer_income_proof :
  let initial_period := 6
  let initial_monthly_expenditure := 80
  let later_period := 4
  let later_monthly_expenditure := 60
  let savings := 30
  (initial_period * monthly_income < initial_period * initial_monthly_expenditure) ∧
  (later_period * monthly_income = later_period * later_monthly_expenditure + debt + savings) →
  monthly_income = 75 := by
sorry

end laborer_income_proof_l311_31163


namespace first_prime_of_nine_sum_100_l311_31140

theorem first_prime_of_nine_sum_100 (primes : List Nat) : 
  primes.length = 9 ∧ 
  (∀ p ∈ primes, Nat.Prime p) ∧ 
  primes.sum = 100 →
  primes.head? = some 2 := by
sorry

end first_prime_of_nine_sum_100_l311_31140


namespace f_8_equals_60_l311_31154

-- Define the function f
def f (n : ℤ) : ℤ := n^2 - 3*n + 20

-- Theorem statement
theorem f_8_equals_60 : f 8 = 60 := by
  sorry

end f_8_equals_60_l311_31154


namespace inverse_variation_proof_l311_31138

/-- Given that x varies inversely as the square of y, prove that y = 6 when x = 0.1111111111111111,
    given that y = 2 when x = 1. -/
theorem inverse_variation_proof (x y : ℝ) (k : ℝ) 
    (h1 : ∀ x y, x = k / (y * y))  -- x varies inversely as square of y
    (h2 : 1 = k / (2 * 2))         -- y = 2 when x = 1
    : y = 6 ↔ x = 0.1111111111111111 :=
by sorry

end inverse_variation_proof_l311_31138


namespace mushroom_problem_solution_l311_31168

/-- Represents a basket of mushrooms with two types: ryzhiki and gruzdi -/
structure MushroomBasket where
  total : ℕ
  ryzhiki : ℕ
  gruzdi : ℕ
  sum_eq_total : ryzhiki + gruzdi = total

/-- Predicate to check if the basket satisfies the ryzhiki condition -/
def has_ryzhik_in_12 (b : MushroomBasket) : Prop :=
  ∀ n : ℕ, n ≤ b.total - 12 → b.ryzhiki > n

/-- Predicate to check if the basket satisfies the gruzdi condition -/
def has_gruzd_in_20 (b : MushroomBasket) : Prop :=
  ∀ n : ℕ, n ≤ b.total - 20 → b.gruzdi > n

/-- Theorem stating the solution to the mushroom problem -/
theorem mushroom_problem_solution :
  ∀ b : MushroomBasket,
  b.total = 30 →
  has_ryzhik_in_12 b →
  has_gruzd_in_20 b →
  b.ryzhiki = 19 ∧ b.gruzdi = 11 := by
  sorry


end mushroom_problem_solution_l311_31168


namespace part_one_part_two_l311_31117

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ :=
  if a ≥ b then b^2 else 2*a - b

-- Theorem for part (1)
theorem part_one : triangle (-3) (-4) = 16 := by sorry

-- Theorem for part (2)
theorem part_two : triangle (triangle (-2) 3) (-8) = 64 := by sorry

end part_one_part_two_l311_31117


namespace circle_reassembly_possible_l311_31111

/-- A circle with a marked point -/
structure MarkedCircle where
  center : ℝ × ℝ
  radius : ℝ
  marked_point : ℝ × ℝ

/-- A piece of a circle -/
structure CirclePiece

/-- Represents the process of cutting a circle into pieces -/
def cut_circle (c : MarkedCircle) (n : ℕ) : List CirclePiece :=
  sorry

/-- Represents the process of assembling pieces into a new circle -/
def assemble_circle (pieces : List CirclePiece) : MarkedCircle :=
  sorry

/-- Theorem stating that it's possible to cut and reassemble the circle as required -/
theorem circle_reassembly_possible (c : MarkedCircle) :
  ∃ (pieces : List CirclePiece),
    (pieces.length = 3) ∧
    (∃ (new_circle : MarkedCircle),
      (assemble_circle pieces = new_circle) ∧
      (new_circle.marked_point = new_circle.center)) :=
  sorry

end circle_reassembly_possible_l311_31111


namespace drama_club_theorem_l311_31166

theorem drama_club_theorem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 50)
  (h2 : math = 36)
  (h3 : physics = 27)
  (h4 : both = 20) :
  total - (math - both + physics - both + both) = 7 :=
by sorry

end drama_club_theorem_l311_31166


namespace angle_sum_is_345_l311_31196

def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

theorem angle_sum_is_345 (α β γ : ℝ) 
  (h1 : is_acute_angle α ∨ is_acute_angle β ∨ is_acute_angle γ)
  (h2 : is_acute_angle α ∨ is_acute_angle β ∨ is_acute_angle γ)
  (h3 : is_obtuse_angle α ∨ is_obtuse_angle β ∨ is_obtuse_angle γ)
  (h4 : (α + β + γ) / 15 = 23 ∨ (α + β + γ) / 15 = 24 ∨ (α + β + γ) / 15 = 25) :
  α + β + γ = 345 := by
  sorry

end angle_sum_is_345_l311_31196


namespace ac_length_l311_31133

/-- Given a quadrilateral ABCD with specified side lengths, prove the length of AC --/
theorem ac_length (AB DC AD : ℝ) (h1 : AB = 13) (h2 : DC = 15) (h3 : AD = 12) :
  ∃ (AC : ℝ), abs (AC - Real.sqrt (369 + 240 * Real.sqrt 2)) < 0.05 := by
  sorry

end ac_length_l311_31133


namespace expected_value_girls_selected_l311_31190

/-- The expected value of girls selected in a hypergeometric distribution -/
theorem expected_value_girls_selected (total : ℕ) (girls : ℕ) (sample : ℕ)
  (h_total : total = 8)
  (h_girls : girls = 3)
  (h_sample : sample = 2) :
  (girls : ℚ) / total * sample = 3 / 4 := by
  sorry

end expected_value_girls_selected_l311_31190


namespace fred_money_last_week_l311_31113

theorem fred_money_last_week 
  (fred_now : ℕ)
  (jason_now : ℕ)
  (jason_earned : ℕ)
  (jason_last_week : ℕ)
  (h1 : fred_now = 112)
  (h2 : jason_now = 63)
  (h3 : jason_earned = 60)
  (h4 : jason_last_week = 3)
  : fred_now - (jason_earned + jason_last_week) = 3 := by
  sorry

end fred_money_last_week_l311_31113


namespace linear_function_properties_l311_31169

def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_properties :
  ∃ (k b : ℝ), 
    k ≠ 0 ∧
    linear_function k b 1 = 2 ∧
    linear_function k b (-1) = 4 ∧
    linear_function (-1) 3 = linear_function k b ∧
    linear_function (-1) 3 2 ≠ 3 ∧
    linear_function (-1) 3 3 = 0 := by
  sorry

end linear_function_properties_l311_31169


namespace eight_weavers_eight_days_l311_31188

/-- Represents the number of mats woven by a given number of mat-weavers in a given number of days. -/
def mats_woven (weavers : ℕ) (days : ℕ) : ℕ := sorry

/-- The rate at which mat-weavers work is constant. -/
axiom constant_rate : mats_woven 4 4 = 4

/-- Theorem stating that 8 mat-weavers can weave 16 mats in 8 days. -/
theorem eight_weavers_eight_days : mats_woven 8 8 = 16 := by sorry

end eight_weavers_eight_days_l311_31188


namespace system_equations_properties_l311_31114

theorem system_equations_properties (a x y : ℝ) 
  (eq1 : x + y = 1 - a) 
  (eq2 : x - y = 3 * a + 5) 
  (x_pos : x > 0) 
  (y_nonneg : y ≥ 0) : 
  (a = -5/3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := by
  sorry

end system_equations_properties_l311_31114


namespace range_of_sin_cos_function_l311_31161

theorem range_of_sin_cos_function : 
  ∀ x : ℝ, 3/4 ≤ Real.sin x ^ 4 + Real.cos x ^ 2 ∧ 
  Real.sin x ^ 4 + Real.cos x ^ 2 ≤ 1 ∧
  ∃ y z : ℝ, Real.sin y ^ 4 + Real.cos y ^ 2 = 3/4 ∧
            Real.sin z ^ 4 + Real.cos z ^ 2 = 1 := by
  sorry

end range_of_sin_cos_function_l311_31161


namespace power_of_i_2023_l311_31182

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_2023 : i ^ 2023 = -i := by
  sorry

end power_of_i_2023_l311_31182


namespace banana_cost_l311_31147

-- Define the rate of bananas
def banana_rate : ℚ := 3 / 4

-- Define the amount of bananas to buy
def banana_amount : ℚ := 20

-- Theorem to prove
theorem banana_cost : banana_amount * banana_rate = 15 := by
  sorry

end banana_cost_l311_31147


namespace power_sum_integer_l311_31115

theorem power_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1 / a = k) :
  ∀ n : ℕ, ∃ m : ℤ, a^n + 1 / a^n = m :=
by sorry

end power_sum_integer_l311_31115


namespace mikeys_jelly_beans_l311_31137

theorem mikeys_jelly_beans (napoleon : ℕ) (sedrich : ℕ) (mikey : ℕ) : 
  napoleon = 17 →
  sedrich = napoleon + 4 →
  2 * (napoleon + sedrich) = 4 * mikey →
  mikey = 19 := by
sorry

end mikeys_jelly_beans_l311_31137


namespace prank_combinations_l311_31145

/-- Represents the number of choices for each day of the week --/
def choices : List Nat := [1, 2, 6, 3, 1]

/-- Calculates the total number of combinations --/
def totalCombinations (choices : List Nat) : Nat :=
  choices.prod

/-- Theorem: The total number of combinations for the given choices is 36 --/
theorem prank_combinations :
  totalCombinations choices = 36 := by
  sorry

end prank_combinations_l311_31145


namespace coin_found_in_33_moves_l311_31141

/-- Represents the game state with thimbles and a hidden coin. -/
structure ThimbleGame where
  num_thimbles : Nat
  thimbles_per_move : Nat

/-- Calculates the maximum number of moves needed to guarantee finding the coin. -/
def max_moves_to_find_coin (game : ThimbleGame) : Nat :=
  sorry

/-- Theorem stating that for 100 thimbles and 4 checks per move, 33 moves are sufficient. -/
theorem coin_found_in_33_moves :
  let game : ThimbleGame := { num_thimbles := 100, thimbles_per_move := 4 }
  max_moves_to_find_coin game ≤ 33 := by
  sorry

end coin_found_in_33_moves_l311_31141


namespace quadratic_equation_implication_l311_31136

theorem quadratic_equation_implication (x : ℝ) : 2 * x^2 + 1 = 17 → 4 * x^2 + 1 = 33 := by
  sorry

end quadratic_equation_implication_l311_31136


namespace compare_quadratic_expressions_l311_31165

theorem compare_quadratic_expressions (x : ℝ) : 2*x^2 - 2*x + 1 > x^2 - 2*x := by
  sorry

end compare_quadratic_expressions_l311_31165


namespace rationalize_denominator_l311_31107

-- Define the original expression
def original_expression := (4 : ℚ) / (3 * (7 : ℚ)^(1/4))

-- Define the rationalized expression
def rationalized_expression := (4 * (343 : ℚ)^(1/4)) / 21

-- State the theorem
theorem rationalize_denominator :
  original_expression = rationalized_expression ∧
  ¬ (∃ (p : ℕ), Prime p ∧ (343 : ℕ) % p^4 = 0) :=
by sorry

end rationalize_denominator_l311_31107


namespace power_values_l311_31109

-- Define variables
variable (a m n : ℝ)

-- State the theorem
theorem power_values (h1 : a^m = 2) (h2 : a^n = 3) :
  a^(4*m + 3*n) = 432 ∧ a^(5*m - 2*n) = 32/9 := by
  sorry

end power_values_l311_31109


namespace leaves_per_frond_l311_31181

theorem leaves_per_frond (num_ferns : ℕ) (fronds_per_fern : ℕ) (total_leaves : ℕ) :
  num_ferns = 6 →
  fronds_per_fern = 7 →
  total_leaves = 1260 →
  total_leaves / (num_ferns * fronds_per_fern) = 30 :=
by sorry

end leaves_per_frond_l311_31181


namespace rahim_average_book_price_l311_31120

/-- The average price of books bought by Rahim -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2 : ℚ)

/-- Theorem: The average price Rahim paid per book is 85 rupees -/
theorem rahim_average_book_price :
  average_price 65 35 6500 2000 = 85 := by
  sorry

end rahim_average_book_price_l311_31120


namespace parabola_b_value_l311_31123

/-- Given a parabola y = ax^2 + bx + c with vertex (p, p) and y-intercept (0, -p), where p ≠ 0, 
    prove that b = 4. -/
theorem parabola_b_value (a b c p : ℝ) (hp : p ≠ 0) 
  (h_vertex : ∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p)
  (h_y_intercept : c = -p) : b = 4 := by
  sorry

end parabola_b_value_l311_31123


namespace train_length_l311_31125

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 144 → crossing_time = 5 → 
  speed_kmh * (1000 / 3600) * crossing_time = 200 := by
  sorry

end train_length_l311_31125


namespace simple_interest_problem_l311_31183

/-- Given a principal amount P and an interest rate R (as a percentage),
    if the amount after 2 years is 720 and after 7 years is 1020,
    then the principal amount P is 600. -/
theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 720 →
  P + (P * R * 7) / 100 = 1020 →
  P = 600 := by
sorry

end simple_interest_problem_l311_31183


namespace has_extremum_if_a_less_than_neg_one_l311_31186

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

/-- Theorem stating that if a < -1, then f has an extremum -/
theorem has_extremum_if_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  ∃ x : ℝ, f_derivative a x = 0 :=
sorry

end has_extremum_if_a_less_than_neg_one_l311_31186


namespace x_value_proof_l311_31153

theorem x_value_proof (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 3) = x) : x = 5 / 3 := by
  sorry

end x_value_proof_l311_31153


namespace average_weight_b_c_l311_31177

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 43 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 42 →  -- The average weight of a, b, and c is 42 kg
  (a + b) / 2 = 40 →      -- The average weight of a and b is 40 kg
  b = 40 →                -- The weight of b is 40 kg
  (b + c) / 2 = 43 :=     -- The average weight of b and c is 43 kg
by sorry

end average_weight_b_c_l311_31177


namespace maintenance_check_time_l311_31198

theorem maintenance_check_time (initial_time : ℝ) : 
  (initial_time + (1/3) * initial_time = 60) → initial_time = 45 := by
  sorry

end maintenance_check_time_l311_31198


namespace suspension_days_per_instance_l311_31106

/-- The number of fingers and toes a typical person has -/
def typical_fingers_and_toes : ℕ := 20

/-- The number of instances of bullying Kris is responsible for -/
def bullying_instances : ℕ := 20

/-- The total number of days Kris was suspended -/
def total_suspension_days : ℕ := 3 * typical_fingers_and_toes

/-- The number of days Kris was suspended for each instance of bullying -/
def days_per_instance : ℚ := total_suspension_days / bullying_instances

theorem suspension_days_per_instance :
  days_per_instance = 3 := by sorry

end suspension_days_per_instance_l311_31106


namespace product_of_sums_equals_difference_of_powers_l311_31118

theorem product_of_sums_equals_difference_of_powers : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end product_of_sums_equals_difference_of_powers_l311_31118


namespace polyhedron_volume_l311_31132

theorem polyhedron_volume (prism_volume pyramid_volume : ℝ) 
  (h1 : prism_volume = Real.sqrt 2 - 1)
  (h2 : pyramid_volume = 1/6) :
  prism_volume + 2 * pyramid_volume = Real.sqrt 2 - 2/3 := by
  sorry

end polyhedron_volume_l311_31132


namespace hyperbola_center_l311_31191

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (2, 0) ∧ f2 = (8, 6) →
  center = (5, 3) := by
  sorry

end hyperbola_center_l311_31191


namespace cos_2x_min_value_in_interval_l311_31164

theorem cos_2x_min_value_in_interval :
  ∃ x ∈ Set.Ioo 0 π, ∀ y ∈ Set.Ioo 0 π, Real.cos (2 * x) ≤ Real.cos (2 * y) ∧
  Real.cos (2 * x) = -1 :=
sorry

end cos_2x_min_value_in_interval_l311_31164


namespace wire_cut_ratio_l311_31103

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (a^2 / 16 = b^2 / (4 * Real.pi)) → a / b = 2 / Real.sqrt Real.pi := by
  sorry

end wire_cut_ratio_l311_31103


namespace agno3_mass_fraction_l311_31108

/-- Given the number of moles, molar mass, and total solution mass of AgNO₃,
    prove that its mass fraction in the solution is 8%. -/
theorem agno3_mass_fraction :
  ∀ (n M m_total : ℝ),
  n = 0.12 →
  M = 170 →
  m_total = 255 →
  let m := n * M
  let ω := m * 100 / m_total
  ω = 8 := by
sorry

end agno3_mass_fraction_l311_31108


namespace smallest_three_digit_prime_with_prime_reverse_l311_31175

/-- A function that reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has three digits -/
def hasThreeDigits (n : ℕ) : Prop := sorry

theorem smallest_three_digit_prime_with_prime_reverse : 
  (∀ n : ℕ, hasThreeDigits n → isPrime n → isPrime (reverseDigits n) → 107 ≤ n) ∧ 
  hasThreeDigits 107 ∧ 
  isPrime 107 ∧ 
  isPrime (reverseDigits 107) := by sorry

end smallest_three_digit_prime_with_prime_reverse_l311_31175


namespace arithmetic_expression_evaluation_l311_31152

theorem arithmetic_expression_evaluation : 8 + 15 / 3 - 2^3 = 5 := by
  sorry

end arithmetic_expression_evaluation_l311_31152


namespace sector_area_l311_31197

theorem sector_area (θ : ℝ) (p : ℝ) (h1 : θ = 2) (h2 : p = 4) :
  let r := (p - θ) / 2
  let area := r^2 * θ / 2
  area = 1 := by sorry

end sector_area_l311_31197


namespace square_pattern_properties_l311_31176

/-- Represents the number of squares in Figure n of the pattern --/
def num_squares (n : ℕ+) : ℕ := 3 + 2 * (n - 1)

/-- Represents the perimeter of Figure n of the pattern --/
def perimeter (n : ℕ+) : ℕ := 8 + 4 * (n - 1)

/-- Theorem stating the properties of the square pattern --/
theorem square_pattern_properties (n : ℕ+) :
  (num_squares n = 3 + 2 * (n - 1)) ∧ (perimeter n = 8 + 4 * (n - 1)) := by
  sorry

end square_pattern_properties_l311_31176


namespace quadratic_equation_solution_l311_31150

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -1 + Real.sqrt 2 ∧ 
  x₂ = -1 - Real.sqrt 2 ∧ 
  x₁^2 + 2*x₁ - 1 = 0 ∧ 
  x₂^2 + 2*x₂ - 1 = 0 := by
  sorry

end quadratic_equation_solution_l311_31150


namespace race_overtake_points_l311_31151

-- Define the race parameters
def kelly_head_start : ℝ := 3
def kelly_speed : ℝ := 9
def abel_speed : ℝ := 9.5
def chris_speed : ℝ := 10
def chris_start_behind : ℝ := 2
def abel_loss_distance : ℝ := 0.75

-- Define the overtake points
def abel_overtake_kelly : ℝ := 54.75
def chris_overtake_both : ℝ := 56

-- Theorem statement
theorem race_overtake_points : 
  kelly_head_start = 3 ∧ 
  kelly_speed = 9 ∧ 
  abel_speed = 9.5 ∧ 
  chris_speed = 10 ∧ 
  chris_start_behind = 2 ∧
  abel_loss_distance = 0.75 →
  (abel_overtake_kelly = 54.75 ∧ chris_overtake_both = 56) := by
  sorry

end race_overtake_points_l311_31151


namespace smaller_bill_value_l311_31105

theorem smaller_bill_value (total_bills : ℕ) (total_value : ℕ) (small_bills : ℕ) (ten_bills : ℕ) 
  (h1 : total_bills = 12)
  (h2 : total_value = 100)
  (h3 : small_bills = 4)
  (h4 : ten_bills = 8)
  (h5 : total_bills = small_bills + ten_bills)
  (h6 : total_value = small_bills * x + ten_bills * 10) :
  x = 5 := by
  sorry

#check smaller_bill_value

end smaller_bill_value_l311_31105


namespace complex_conjugate_roots_l311_31162

/-- The quadratic equation z^2 + (12 + ci)z + (45 + di) = 0 has complex conjugate roots if and only if c = 0 and d = 0 -/
theorem complex_conjugate_roots (c d : ℝ) : 
  (∀ z : ℂ, z^2 + (12 + c * Complex.I) * z + (45 + d * Complex.I) = 0 → 
    ∃ x y : ℝ, z = x + y * Complex.I ∧ x - y * Complex.I ∈ {w : ℂ | w^2 + (12 + c * Complex.I) * w + (45 + d * Complex.I) = 0}) ↔ 
  c = 0 ∧ d = 0 := by
sorry

end complex_conjugate_roots_l311_31162


namespace triangular_array_coin_sum_l311_31180

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem triangular_array_coin_sum :
  ∃ (N : ℕ), triangular_sum N = 5050 ∧ sum_of_digits N = 1 :=
sorry

end triangular_array_coin_sum_l311_31180


namespace hippocrates_lunes_l311_31189

theorem hippocrates_lunes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 + b^2 = c^2) :
  let semicircle_area (d : ℝ) := π * d^2 / 8
  let triangle_area := a * b / 2
  let lunes_area := semicircle_area a + semicircle_area b - (semicircle_area c - triangle_area)
  lunes_area = triangle_area := by
sorry

end hippocrates_lunes_l311_31189


namespace unique_quadratic_solution_l311_31122

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 25 * x + 9 = 0) :
  ∃ x, b * x^2 + 25 * x + 9 = 0 ∧ x = -18/25 := by
sorry

end unique_quadratic_solution_l311_31122


namespace toms_original_amount_l311_31195

theorem toms_original_amount (tom sara jim : ℝ) : 
  tom + sara + jim = 1200 →
  (tom - 200) + (3 * sara) + (2 * jim) = 1800 →
  tom = 400 := by
sorry

end toms_original_amount_l311_31195


namespace age_difference_l311_31139

theorem age_difference (ana_age bonita_age : ℕ) : 
  (ana_age - 1 = 3 * (bonita_age - 1)) →  -- Last year's condition
  (ana_age = 2 * bonita_age + 3) →        -- This year's condition
  (ana_age - bonita_age = 8) :=           -- Age difference is 8
by
  sorry


end age_difference_l311_31139


namespace area_comparison_l311_31126

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define a function to check if a triangle is inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define a function to find the points where angle bisectors meet the circle
def angleBisectorPoints (t : Triangle) (c : Circle) : Triangle := sorry

-- Theorem statement
theorem area_comparison 
  (t : Triangle) (c : Circle) 
  (h : isInscribed t c) : 
  let t' := angleBisectorPoints t c
  triangleArea t ≤ triangleArea t' := by sorry

end area_comparison_l311_31126


namespace complex_number_property_l311_31135

theorem complex_number_property : 
  let z : ℂ := (-2 * Complex.I) / (1 + Complex.I)
  (z + 1).im ≠ 0 ∧ (z + 1).re = 0 := by
  sorry

end complex_number_property_l311_31135


namespace repeating_decimal_three_six_equals_eleven_thirtieths_l311_31172

def repeating_decimal (a b : ℕ) : ℚ :=
  (a : ℚ) / 10 + (b : ℚ) / (9 * 10)

theorem repeating_decimal_three_six_equals_eleven_thirtieths :
  repeating_decimal 3 6 = 11 / 30 := by
  sorry

end repeating_decimal_three_six_equals_eleven_thirtieths_l311_31172


namespace sequence_existence_iff_k_in_range_l311_31193

theorem sequence_existence_iff_k_in_range (n : ℕ) :
  (∃ (x : ℕ → ℕ), (∀ i j, i < j → i ≤ n → j ≤ n → x i < x j)) ↔
  (∀ k : ℕ, k ≤ n → ∃ (x : ℕ → ℕ), (∀ i j, i < j → i ≤ k → j ≤ k → x i < x j)) :=
by sorry

end sequence_existence_iff_k_in_range_l311_31193


namespace orchestra_size_l311_31128

def percussion_count : ℕ := 3
def brass_count : ℕ := 5 + 4 + 2 + 2
def strings_count : ℕ := 7 + 5 + 4 + 2
def woodwinds_count : ℕ := 3 + 4 + 2 + 1
def keyboards_harp_count : ℕ := 1 + 1
def conductor_count : ℕ := 1

theorem orchestra_size :
  percussion_count + brass_count + strings_count + woodwinds_count + keyboards_harp_count + conductor_count = 47 := by
  sorry

end orchestra_size_l311_31128


namespace pretty_numbers_characterization_l311_31100

def is_pretty (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ k ℓ : ℕ, 0 < k ∧ k < n ∧ 0 < ℓ ∧ ℓ < n ∧ k ∣ n ∧ ℓ ∣ n →
    (2 * k - ℓ) ∣ n ∨ (2 * ℓ - k) ∣ n

theorem pretty_numbers_characterization (n : ℕ) :
  is_pretty n ↔ Nat.Prime n ∨ n = 6 ∨ n = 9 ∨ n = 15 :=
sorry

end pretty_numbers_characterization_l311_31100


namespace log_relation_l311_31171

theorem log_relation (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
  sorry

end log_relation_l311_31171


namespace volunteer_allocation_l311_31131

theorem volunteer_allocation (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 84 := by
  sorry

end volunteer_allocation_l311_31131


namespace exists_special_sequence_l311_31104

/-- An infinite increasing sequence of natural numbers -/
def IncreasingSeq : ℕ → ℕ := sorry

/-- The property that the sequence is increasing -/
axiom seq_increasing : ∀ n : ℕ, IncreasingSeq n < IncreasingSeq (n + 1)

/-- The coprimality property of the sequence -/
axiom seq_coprime : ∀ i j p q r : ℕ, 
  i ≠ j → i ≠ p → i ≠ q → i ≠ r → j ≠ p → j ≠ q → j ≠ r → p ≠ q → p ≠ r → q ≠ r →
  Nat.gcd (IncreasingSeq i + IncreasingSeq j) (IncreasingSeq p + IncreasingSeq q + IncreasingSeq r) = 1

/-- The main theorem: existence of the sequence with the required properties -/
theorem exists_special_sequence : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n : ℕ, seq n < seq (n + 1)) ∧ 
    (∀ i j p q r : ℕ, 
      i ≠ j → i ≠ p → i ≠ q → i ≠ r → j ≠ p → j ≠ q → j ≠ r → p ≠ q → p ≠ r → q ≠ r →
      Nat.gcd (seq i + seq j) (seq p + seq q + seq r) = 1) :=
sorry

end exists_special_sequence_l311_31104


namespace swan_population_after_ten_years_l311_31146

/-- The number of swans after a given number of years, given an initial population and a doubling period -/
def swan_population (initial_population : ℕ) (doubling_period : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / doubling_period)

/-- Theorem stating that the swan population after 10 years will be 480, given the initial conditions -/
theorem swan_population_after_ten_years :
  swan_population 15 2 10 = 480 := by
sorry

end swan_population_after_ten_years_l311_31146


namespace at_least_one_not_less_than_two_l311_31194

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l311_31194


namespace tan_fifteen_to_sqrt_three_l311_31101

theorem tan_fifteen_to_sqrt_three : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_fifteen_to_sqrt_three_l311_31101


namespace average_age_of_nine_students_l311_31167

theorem average_age_of_nine_students 
  (total_students : ℕ)
  (total_average : ℚ)
  (five_students : ℕ)
  (five_average : ℚ)
  (fifteenth_student_age : ℕ)
  (h1 : total_students = 15)
  (h2 : total_average = 15)
  (h3 : five_students = 5)
  (h4 : five_average = 13)
  (h5 : fifteenth_student_age = 16) :
  (total_students * total_average - five_students * five_average - fifteenth_student_age) / (total_students - five_students - 1) = 16 := by
  sorry

end average_age_of_nine_students_l311_31167


namespace sum_of_squares_divisible_by_seven_l311_31179

theorem sum_of_squares_divisible_by_seven (a b : ℤ) :
  (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end sum_of_squares_divisible_by_seven_l311_31179


namespace strawberry_cake_cost_l311_31149

/-- Proves that the cost of each strawberry cake is $22 given the order details --/
theorem strawberry_cake_cost
  (num_chocolate : ℕ)
  (price_chocolate : ℕ)
  (num_strawberry : ℕ)
  (total_cost : ℕ)
  (h1 : num_chocolate = 3)
  (h2 : price_chocolate = 12)
  (h3 : num_strawberry = 6)
  (h4 : total_cost = 168)
  : (total_cost - num_chocolate * price_chocolate) / num_strawberry = 22 := by
  sorry

end strawberry_cake_cost_l311_31149


namespace heartsuit_three_five_l311_31158

-- Define the ⊛ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end heartsuit_three_five_l311_31158


namespace combined_work_theorem_l311_31173

/-- The number of days it takes for three workers to complete a task together,
    given their individual completion times. -/
def combinedWorkDays (raviDays prakashDays seemaDays : ℚ) : ℚ :=
  1 / (1 / raviDays + 1 / prakashDays + 1 / seemaDays)

/-- Theorem stating that if Ravi can do the work in 50 days, Prakash in 75 days,
    and Seema in 60 days, they will finish the work together in 20 days. -/
theorem combined_work_theorem :
  combinedWorkDays 50 75 60 = 20 := by sorry

end combined_work_theorem_l311_31173


namespace weight_gain_ratio_l311_31112

/-- The weight gain problem at the family reunion --/
theorem weight_gain_ratio (orlando jose fernando : ℕ) : 
  orlando = 5 →
  jose = 2 * orlando + 2 →
  orlando + jose + fernando = 20 →
  fernando * 4 = jose := by
  sorry

end weight_gain_ratio_l311_31112


namespace cube_root_unity_sum_of_powers_l311_31127

theorem cube_root_unity_sum_of_powers : 
  let ω : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  (ω ^ 3 = 1) → (ω ≠ 1) →
  (ω ^ 8 + (ω ^ 2) ^ 8 = -2) := by
sorry

end cube_root_unity_sum_of_powers_l311_31127


namespace pentagon_square_angle_sum_l311_31159

theorem pentagon_square_angle_sum : 
  ∀ (pentagon_angle square_angle : ℝ),
  (pentagon_angle = 180 * (5 - 2) / 5) →
  (square_angle = 180 * (4 - 2) / 4) →
  pentagon_angle + square_angle = 198 := by
  sorry

end pentagon_square_angle_sum_l311_31159


namespace bag_of_balls_l311_31142

theorem bag_of_balls (num_black : ℕ) (prob_black : ℚ) (total : ℕ) : 
  num_black = 4 → prob_black = 1/3 → total = num_black / prob_black → total = 12 := by
sorry

end bag_of_balls_l311_31142


namespace picnic_cost_is_60_l311_31148

/-- Calculates the total cost of a picnic basket given the number of people and item costs. -/
def picnic_cost (num_people : ℕ) (sandwich_cost fruit_salad_cost soda_cost snack_cost : ℕ) 
  (num_sodas_per_person num_snack_bags : ℕ) : ℕ :=
  num_people * (sandwich_cost + fruit_salad_cost + num_sodas_per_person * soda_cost) + 
  num_snack_bags * snack_cost

/-- Theorem stating that the total cost of the picnic basket is $60. -/
theorem picnic_cost_is_60 : 
  picnic_cost 4 5 3 2 4 2 3 = 60 := by
  sorry

end picnic_cost_is_60_l311_31148


namespace fourth_post_length_l311_31144

/-- Given a total rope length and the lengths used for the first three posts,
    calculate the length of rope used for the fourth post. -/
def rope_for_fourth_post (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) : ℕ :=
  total - (first + second + third)

/-- Theorem stating that given the specific lengths in the problem,
    the rope used for the fourth post is 12 inches. -/
theorem fourth_post_length :
  rope_for_fourth_post 70 24 20 14 = 12 := by
  sorry

end fourth_post_length_l311_31144


namespace even_sum_sufficient_not_necessary_l311_31110

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define the sum of two functions
def SumFunc (f g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f x + g x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (SumFunc f g)) ∧ 
  (∃ f g : ℝ → ℝ, IsEven (SumFunc f g) ∧ ¬(IsEven f) ∧ ¬(IsEven g)) :=
sorry

end even_sum_sufficient_not_necessary_l311_31110


namespace compare_powers_l311_31143

theorem compare_powers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < 1) (hba : 1 < b) :
  a^4 < 1 ∧ 1 < b^(1/4) := by
  sorry

end compare_powers_l311_31143


namespace alternative_plan_savings_l311_31185

/-- Proves that the alternative phone plan is $1 cheaper than the current plan -/
theorem alternative_plan_savings :
  ∀ (current_plan_cost : ℚ)
    (texts_sent : ℕ)
    (call_minutes : ℕ)
    (text_package_size : ℕ)
    (call_package_size : ℕ)
    (text_package_cost : ℚ)
    (call_package_cost : ℚ),
  current_plan_cost = 12 →
  texts_sent = 60 →
  call_minutes = 60 →
  text_package_size = 30 →
  call_package_size = 20 →
  text_package_cost = 1 →
  call_package_cost = 3 →
  current_plan_cost - 
    ((texts_sent / text_package_size : ℚ) * text_package_cost +
     (call_minutes / call_package_size : ℚ) * call_package_cost) = 1 :=
by
  sorry

#check alternative_plan_savings

end alternative_plan_savings_l311_31185


namespace boys_girls_arrangement_l311_31187

/-- The number of ways to arrange boys and girls in a row with alternating genders -/
def alternating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (Nat.factorial num_boys) * (Nat.factorial num_girls)

/-- Theorem stating that there are 144 ways to arrange 4 boys and 3 girls
    in a row such that no two boys or two girls stand next to each other -/
theorem boys_girls_arrangement :
  alternating_arrangements 4 3 = 144 := by
  sorry

#check boys_girls_arrangement

end boys_girls_arrangement_l311_31187


namespace geometric_sequence_property_l311_31192

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a → a 2 * a 6 = 4 → (a 4 = 2 ∨ a 4 = -2) :=
by
  sorry


end geometric_sequence_property_l311_31192


namespace complex_equation_solution_complex_inequality_range_l311_31178

-- Problem 1
theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + 8 * I → z = -15 + 8 * I := by sorry

-- Problem 2
theorem complex_inequality_range (a : ℝ) :
  Complex.abs (3 + a * I) < 4 → -Real.sqrt 7 < a ∧ a < Real.sqrt 7 := by sorry

end complex_equation_solution_complex_inequality_range_l311_31178


namespace odd_m_triple_g_35_l311_31116

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n  -- This case is not specified in the original problem, but needed for completeness

theorem odd_m_triple_g_35 (m : ℤ) (h_odd : m % 2 = 1) (h_triple_g : g (g (g m)) = 35) : m = 85 := by
  sorry

end odd_m_triple_g_35_l311_31116
