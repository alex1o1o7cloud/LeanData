import Mathlib

namespace square_cut_impossible_l853_85377

/-- Proves that a square with perimeter 40 cannot be cut into two identical rectangles with perimeter 20 each -/
theorem square_cut_impossible (square_perimeter : ℝ) (rect_perimeter : ℝ) : 
  square_perimeter = 40 → rect_perimeter = 20 → 
  ¬ ∃ (square_side rect_length rect_width : ℝ),
    (square_side * 4 = square_perimeter) ∧ 
    (rect_length + rect_width = square_side) ∧
    (2 * (rect_length + rect_width) = rect_perimeter) :=
by
  sorry

#check square_cut_impossible

end square_cut_impossible_l853_85377


namespace grape_rate_calculation_l853_85374

/-- The rate per kg for grapes that Andrew purchased -/
def grape_rate : ℝ := 74

/-- The amount of grapes Andrew purchased in kg -/
def grape_amount : ℝ := 6

/-- The rate per kg for mangoes that Andrew purchased -/
def mango_rate : ℝ := 59

/-- The amount of mangoes Andrew purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount Andrew paid to the shopkeeper -/
def total_paid : ℝ := 975

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
sorry

end grape_rate_calculation_l853_85374


namespace min_value_constraint_l853_85363

theorem min_value_constraint (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + 2*y = 1) :
  2*x + 3*y^2 ≥ 0.75 := by
  sorry

end min_value_constraint_l853_85363


namespace division_sum_theorem_l853_85314

theorem division_sum_theorem (quotient divisor remainder : ℝ) :
  quotient = 450 →
  divisor = 350.7 →
  remainder = 287.9 →
  (divisor * quotient) + remainder = 158102.9 := by
  sorry

end division_sum_theorem_l853_85314


namespace equation_solution_l853_85339

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x * (x - 2) - 2 * (x + 1)
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 6 ∧ x₂ = 2 - Real.sqrt 6 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end equation_solution_l853_85339


namespace f_odd_and_increasing_l853_85310

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Theorem stating that f is an odd function and an increasing function
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end f_odd_and_increasing_l853_85310


namespace train_speed_and_length_l853_85323

def bridge_length : ℝ := 1260
def bridge_time : ℝ := 60
def tunnel_length : ℝ := 2010
def tunnel_time : ℝ := 90

theorem train_speed_and_length :
  ∃ (speed length : ℝ),
    (bridge_length + length) / bridge_time = (tunnel_length + length) / tunnel_time ∧
    speed = (bridge_length + length) / bridge_time ∧
    speed = 25 ∧
    length = 240 := by sorry

end train_speed_and_length_l853_85323


namespace expression_simplification_l853_85315

theorem expression_simplification (p : ℝ) 
  (h1 : p^3 - p^2 + 2*p + 16 ≠ 0) 
  (h2 : p^2 + 2*p + 6 ≠ 0) : 
  (p^3 + 4*p^2 + 10*p + 12) / (p^3 - p^2 + 2*p + 16) * 
  (p^3 - 3*p^2 + 8*p) / (p^2 + 2*p + 6) = p := by
  sorry

end expression_simplification_l853_85315


namespace equation_solutions_l853_85368

theorem equation_solutions : ∃ (x₁ x₂ : ℝ) (z₁ z₂ : ℂ),
  (∀ x : ℝ, (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 48 ↔ x = x₁ ∨ x = x₂) ∧
  (∀ z : ℂ, (15*z - z^2)/(z + 2) * (z + (15 - z)/(z + 2)) = 48 ↔ z = z₁ ∨ z = z₂) ∧
  x₁ = 4 ∧ x₂ = -3 ∧ z₁ = 3 + Complex.I * Real.sqrt 2 ∧ z₂ = 3 - Complex.I * Real.sqrt 2 :=
by sorry

end equation_solutions_l853_85368


namespace tray_trips_l853_85303

theorem tray_trips (capacity : ℕ) (total_trays : ℕ) (h1 : capacity = 8) (h2 : total_trays = 16) :
  (total_trays + capacity - 1) / capacity = 2 := by
  sorry

end tray_trips_l853_85303


namespace square_sum_and_product_l853_85394

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 4) 
  (h2 : (x + y)^2 = 64) : 
  x^2 + y^2 = 34 ∧ x * y = 15 := by
  sorry

end square_sum_and_product_l853_85394


namespace prime_sum_product_l853_85300

theorem prime_sum_product (p₁ p₂ p₃ p₄ : ℕ) 
  (h_prime₁ : Nat.Prime p₁) (h_prime₂ : Nat.Prime p₂) 
  (h_prime₃ : Nat.Prime p₃) (h_prime₄ : Nat.Prime p₄)
  (h_order : p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄)
  (h_sum : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  p₁ = 7 ∧ p₂ = 11 ∧ p₃ = 13 ∧ p₄ = 17 := by
  sorry

end prime_sum_product_l853_85300


namespace units_digit_27_times_36_l853_85302

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_27_times_36 : units_digit (27 * 36) = 2 := by
  sorry

end units_digit_27_times_36_l853_85302


namespace product_of_decimals_l853_85353

theorem product_of_decimals : (0.5 : ℝ) * 0.3 = 0.15 := by sorry

end product_of_decimals_l853_85353


namespace min_c_value_l853_85366

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

theorem min_c_value (a b c d e : ℕ) : 
  (a + 1 = b) → 
  (b + 1 = c) → 
  (c + 1 = d) → 
  (d + 1 = e) → 
  (is_perfect_square (b + c + d)) →
  (is_perfect_cube (a + b + c + d + e)) →
  (c ≥ 675 ∧ ∀ x < c, ¬(is_perfect_square (x - 1 + x + x + 1) ∧ 
                        is_perfect_cube (x - 2 + x - 1 + x + x + 1 + x + 2))) :=
by sorry

end min_c_value_l853_85366


namespace vector_addition_problem_l853_85382

theorem vector_addition_problem (a b : ℝ × ℝ) :
  a = (2, -1) → b = (-3, 4) → 2 • a + b = (1, 2) := by sorry

end vector_addition_problem_l853_85382


namespace negative_fractions_comparison_l853_85354

theorem negative_fractions_comparison : -1/3 < -1/4 := by
  sorry

end negative_fractions_comparison_l853_85354


namespace orange_weight_l853_85331

theorem orange_weight (apple_weight orange_weight : ℝ) 
  (h1 : orange_weight = 5 * apple_weight) 
  (h2 : apple_weight + orange_weight = 12) : 
  orange_weight = 10 := by
sorry

end orange_weight_l853_85331


namespace inequality_proof_l853_85356

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 5/2) (hy : y ≥ 5/2) (hz : z ≥ 5/2) :
  (1 + 1/(2+x)) * (1 + 1/(2+y)) * (1 + 1/(2+z)) ≥ (1 + 1/(2 + (x*y*z)^(1/3)))^3 := by
  sorry

end inequality_proof_l853_85356


namespace garden_tulips_count_l853_85344

/-- Represents the garden scenario with tulips and sunflowers -/
structure Garden where
  tulip_ratio : ℕ
  sunflower_ratio : ℕ
  initial_sunflowers : ℕ
  added_sunflowers : ℕ

/-- Calculates the final number of tulips in the garden -/
def final_tulips (g : Garden) : ℕ :=
  let final_sunflowers := g.initial_sunflowers + g.added_sunflowers
  let ratio_units := final_sunflowers / g.sunflower_ratio
  ratio_units * g.tulip_ratio

/-- Theorem stating that given the garden conditions, the final number of tulips is 30 -/
theorem garden_tulips_count (g : Garden) 
  (h1 : g.tulip_ratio = 3)
  (h2 : g.sunflower_ratio = 7)
  (h3 : g.initial_sunflowers = 42)
  (h4 : g.added_sunflowers = 28) : 
  final_tulips g = 30 := by
  sorry

end garden_tulips_count_l853_85344


namespace absolute_value_problem_l853_85398

theorem absolute_value_problem (m n : ℤ) 
  (hm : |m| = 4) (hn : |n| = 3) : 
  ((m * n > 0 → |m - n| = 1) ∧ 
   (m * n < 0 → |m + n| = 1)) := by
  sorry

end absolute_value_problem_l853_85398


namespace mrs_lim_milk_revenue_l853_85311

/-- Calculates the revenue from milk sales given the conditions of Mrs. Lim's milk production and sales --/
theorem mrs_lim_milk_revenue :
  let yesterday_morning : ℕ := 68
  let yesterday_evening : ℕ := 82
  let this_morning_difference : ℕ := 18
  let remaining_milk : ℕ := 24
  let price_per_gallon : ℚ := 7/2

  let this_morning : ℕ := yesterday_morning - this_morning_difference
  let total_milk : ℕ := yesterday_morning + yesterday_evening + this_morning
  let sold_milk : ℕ := total_milk - remaining_milk
  let revenue : ℚ := price_per_gallon * sold_milk

  revenue = 616 := by sorry

end mrs_lim_milk_revenue_l853_85311


namespace six_people_arrangement_l853_85396

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of positions where person A can stand (not at ends) -/
def validPositionsForA (n : ℕ) : ℕ := n - 2

/-- The number of ways to arrange the remaining people after placing A -/
def remainingArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The total number of valid arrangements -/
def validArrangements (n : ℕ) : ℕ :=
  (validPositionsForA n) * (remainingArrangements n)

theorem six_people_arrangement :
  validArrangements 6 = 480 := by
  sorry

end six_people_arrangement_l853_85396


namespace max_difference_reverse_digits_l853_85318

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Main theorem -/
theorem max_difference_reverse_digits (q r : ℕ) :
  TwoDigitInt q ∧ TwoDigitInt r ∧
  r = reverseDigits q ∧
  q > r ∧
  q - r < 20 →
  q - r ≤ 18 :=
sorry

end max_difference_reverse_digits_l853_85318


namespace students_per_group_l853_85338

theorem students_per_group 
  (total_students : ℕ) 
  (students_not_picked : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : students_not_picked = 36) 
  (h3 : num_groups = 4) : 
  (total_students - students_not_picked) / num_groups = 7 := by
sorry

end students_per_group_l853_85338


namespace total_books_is_283_l853_85327

/-- The number of books borrowed on a given day -/
def books_borrowed (day : Nat) : Nat :=
  match day with
  | 1 => 40  -- Monday
  | 2 => 42  -- Tuesday
  | 3 => 44  -- Wednesday
  | 4 => 46  -- Thursday
  | 5 => 64  -- Friday
  | _ => 0   -- Weekend (handled separately)

/-- The total number of books borrowed during weekdays -/
def weekday_total : Nat :=
  (List.range 5).map books_borrowed |>.sum

/-- The number of books borrowed during the weekend -/
def weekend_books : Nat :=
  (weekday_total / 10) * 2

/-- The total number of books borrowed over the week -/
def total_books : Nat :=
  weekday_total + weekend_books

theorem total_books_is_283 : total_books = 283 := by
  sorry

end total_books_is_283_l853_85327


namespace particle_probability_l853_85340

/-- The probability of a particle reaching point (2,3) after 5 moves -/
theorem particle_probability (n : ℕ) (k : ℕ) (p : ℝ) : 
  n = 5 → k = 2 → p = 1/2 → 
  Nat.choose n k * p^n = Nat.choose 5 2 * (1/2)^5 :=
by sorry

end particle_probability_l853_85340


namespace package_weight_problem_l853_85329

theorem package_weight_problem (x y z w : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : y + z + w = 160)
  (h3 : z + w + x = 170) :
  x + y + z + w = 160 := by
sorry

end package_weight_problem_l853_85329


namespace black_equals_sum_of_whites_l853_85332

/-- Definition of a white number -/
def is_white_number (x : ℝ) : Prop :=
  ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ x = Real.sqrt (a + b * Real.sqrt 2)

/-- Definition of a black number -/
def is_black_number (x : ℝ) : Prop :=
  ∃ (c d : ℤ), c ≠ 0 ∧ d ≠ 0 ∧ x = Real.sqrt (c + d * Real.sqrt 7)

/-- Theorem stating that a black number can be equal to the sum of two white numbers -/
theorem black_equals_sum_of_whites :
  ∃ (x y z : ℝ), is_white_number x ∧ is_white_number y ∧ is_black_number z ∧ z = x + y :=
sorry

end black_equals_sum_of_whites_l853_85332


namespace intersection_point_correct_l853_85393

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (-9/13, 32/13)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = -10x - 2 -/
def line2 (x y : ℚ) : Prop := 2 * y = -10 * x - 2

theorem intersection_point_correct :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

end intersection_point_correct_l853_85393


namespace gcd_12569_36975_l853_85391

theorem gcd_12569_36975 : Nat.gcd 12569 36975 = 1 := by
  sorry

end gcd_12569_36975_l853_85391


namespace marble_distribution_l853_85334

theorem marble_distribution (total_marbles : ℕ) (num_groups : ℕ) (marbles_per_group : ℕ) :
  total_marbles = 64 →
  num_groups = 32 →
  total_marbles = num_groups * marbles_per_group →
  marbles_per_group = 2 := by
  sorry

end marble_distribution_l853_85334


namespace three_isosceles_triangles_l853_85345

-- Define a point on the grid
structure GridPoint where
  x : Int
  y : Int

-- Define a triangle on the grid
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : GridTriangle) : Bool :=
  let d1 := squaredDistance t.p1 t.p2
  let d2 := squaredDistance t.p2 t.p3
  let d3 := squaredDistance t.p3 t.p1
  d1 = d2 || d2 = d3 || d3 = d1

-- Define the five triangles
def triangle1 : GridTriangle := ⟨⟨0, 8⟩, ⟨4, 8⟩, ⟨2, 5⟩⟩
def triangle2 : GridTriangle := ⟨⟨2, 2⟩, ⟨2, 5⟩, ⟨6, 2⟩⟩
def triangle3 : GridTriangle := ⟨⟨1, 1⟩, ⟨5, 4⟩, ⟨9, 1⟩⟩
def triangle4 : GridTriangle := ⟨⟨7, 7⟩, ⟨6, 9⟩, ⟨10, 7⟩⟩
def triangle5 : GridTriangle := ⟨⟨3, 1⟩, ⟨4, 4⟩, ⟨6, 0⟩⟩

-- List of all triangles
def allTriangles : List GridTriangle := [triangle1, triangle2, triangle3, triangle4, triangle5]

-- Theorem: Exactly 3 out of the 5 given triangles are isosceles
theorem three_isosceles_triangles :
  (allTriangles.filter isIsosceles).length = 3 := by
  sorry


end three_isosceles_triangles_l853_85345


namespace village_seniors_l853_85325

/-- Proves the number of seniors in a village given the population distribution -/
theorem village_seniors (total_population : ℕ) 
  (h1 : total_population * 60 / 100 = 23040)  -- 60% of population are adults
  (h2 : total_population * 30 / 100 = total_population * 3 / 10) -- 30% are children
  : total_population * 10 / 100 = 3840 := by
  sorry

end village_seniors_l853_85325


namespace integer_roots_of_cubic_l853_85373

theorem integer_roots_of_cubic (x : ℤ) :
  x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = -4 ∨ x = 3 ∨ x = 8 := by
  sorry

end integer_roots_of_cubic_l853_85373


namespace square_is_one_l853_85342

/-- Represents a digit in base-7 --/
def Base7Digit := Fin 7

/-- The addition problem in base-7 --/
def addition_problem (square : Base7Digit) : Prop :=
  ∃ (carry1 carry2 carry3 : Nat),
    (square.val + 1 + 3 + 2) % 7 = 0 ∧
    (carry1 + square.val + 5 + square.val + 1) % 7 = square.val ∧
    (carry2 + 4 + carry3) % 7 = 5 ∧
    carry1 = (square.val + 1 + 3 + 2) / 7 ∧
    carry2 = (carry1 + square.val + 5 + square.val + 1) / 7 ∧
    carry3 = (square.val + 5 + 1) / 7

theorem square_is_one :
  ∃ (square : Base7Digit), addition_problem square ∧ square.val = 1 := by sorry

end square_is_one_l853_85342


namespace tangent_line_at_zero_two_l853_85333

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x + 1

theorem tangent_line_at_zero_two :
  let f : ℝ → ℝ := λ x ↦ Real.exp x + 2 * x + 1
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3 * x - y + 2 = 0) :=
by
  sorry

end tangent_line_at_zero_two_l853_85333


namespace smallest_y_for_perfect_cube_l853_85324

def x : ℕ := 5 * 24 * 36

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube : 
  (∀ y < 50, ¬ is_perfect_cube (x * y)) ∧ is_perfect_cube (x * 50) := by
  sorry

end smallest_y_for_perfect_cube_l853_85324


namespace coffee_decaf_percentage_l853_85378

theorem coffee_decaf_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (additional_purchase : ℝ) 
  (final_decaf_percent : ℝ) :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  additional_purchase = 100 →
  final_decaf_percent = 26 →
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := initial_stock * (initial_decaf_percent / 100)
  let final_decaf := total_stock * (final_decaf_percent / 100)
  let additional_decaf := final_decaf - initial_decaf
  (additional_decaf / additional_purchase) * 100 = 50 := by
sorry

end coffee_decaf_percentage_l853_85378


namespace vector_properties_l853_85375

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Parallel vectors condition -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- Perpendicular vectors condition -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Vector addition -/
def add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

/-- Scalar multiplication -/
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

/-- Vector subtraction -/
def sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  add v (smul (-1) w)

/-- Squared norm of a vector -/
def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

theorem vector_properties (m : ℝ) :
  (parallel a (b m) ↔ m = -4) ∧
  (perpendicular a (b m) ↔ m = 1) ∧
  ¬(norm_sq (sub (smul 2 a) (b m)) = norm_sq (add a (b m)) → m = 1) ∧
  ¬(norm_sq (add a (b m)) = norm_sq a → m = -4) := by
  sorry

end vector_properties_l853_85375


namespace optimal_price_maximizes_profit_l853_85357

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 220 * x - 960

/-- Represents the optimal selling price that maximizes profit -/
def optimal_price : ℝ := 11

theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

#check optimal_price_maximizes_profit

end optimal_price_maximizes_profit_l853_85357


namespace total_tax_percentage_l853_85320

/-- Calculate the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.60)
  (h2 : food_percent = 0.10)
  (h3 : other_percent = 0.30)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.08) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.048 := by
sorry

end total_tax_percentage_l853_85320


namespace units_digit_factorial_sum_2010_l853_85319

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_2010 : 
  units_digit (factorial_sum 2010) = 3 := by sorry

end units_digit_factorial_sum_2010_l853_85319


namespace train_speed_l853_85346

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250.00000000000003)
  (h2 : time = 15) : 
  ∃ (speed : ℝ), abs (speed - 60) < 0.00000000000001 :=
by
  sorry

end train_speed_l853_85346


namespace race_distance_l853_85365

theorem race_distance (d : ℝ) (a b c : ℝ) : 
  (d > 0) →
  (d / a = (d - 30) / b) →
  (d / b = (d - 15) / c) →
  (d / a = (d - 40) / c) →
  d = 90 := by
sorry

end race_distance_l853_85365


namespace right_triangle_relation_l853_85352

theorem right_triangle_relation (a b c h : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hh : h > 0)
  (right_triangle : a^2 + b^2 = c^2) (height_relation : 2 * h * c = a * b) :
  1 / a^2 + 1 / b^2 = 1 / h^2 := by
  sorry

end right_triangle_relation_l853_85352


namespace strawberry_harvest_l853_85381

theorem strawberry_harvest (garden_length : ℝ) (garden_width : ℝ) 
  (plantable_percentage : ℝ) (plants_per_sqft : ℝ) (strawberries_per_plant : ℝ) : ℝ :=
  by
  have garden_length_eq : garden_length = 10 := by sorry
  have garden_width_eq : garden_width = 12 := by sorry
  have plantable_percentage_eq : plantable_percentage = 0.9 := by sorry
  have plants_per_sqft_eq : plants_per_sqft = 4 := by sorry
  have strawberries_per_plant_eq : strawberries_per_plant = 8 := by sorry
  
  have total_area : ℝ := garden_length * garden_width
  have plantable_area : ℝ := total_area * plantable_percentage
  have total_plants : ℝ := plantable_area * plants_per_sqft
  have total_strawberries : ℝ := total_plants * strawberries_per_plant
  
  exact total_strawberries

end strawberry_harvest_l853_85381


namespace no_rectangle_with_half_perimeter_and_area_l853_85301

theorem no_rectangle_with_half_perimeter_and_area 
  (a b : ℝ) (h_ab : 0 < a ∧ a < b) : 
  ¬∃ (x y : ℝ), 
    0 < x ∧ x < b ∧
    0 < y ∧ y < b ∧
    x + y = a + b ∧
    x * y = (a * b) / 2 := by
sorry

end no_rectangle_with_half_perimeter_and_area_l853_85301


namespace harrison_croissant_cost_l853_85349

/-- The cost of croissants for Harrison in a year -/
def croissant_cost (regular_price almond_price : ℚ) (weeks_per_year : ℕ) : ℚ :=
  weeks_per_year * (regular_price + almond_price)

/-- Theorem: Harrison spends $468.00 on croissants in a year -/
theorem harrison_croissant_cost :
  croissant_cost (35/10) (55/10) 52 = 468 :=
sorry

end harrison_croissant_cost_l853_85349


namespace intersection_equality_l853_85348

def M : Set ℤ := {-1, 0, 1}

def N (a : ℤ) : Set ℤ := {a, a^2}

theorem intersection_equality (a : ℤ) : M ∩ N a = N a ↔ a = -1 := by
  sorry

end intersection_equality_l853_85348


namespace percentage_calculation_l853_85389

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 6000 → 
  P / 100 * (30 / 100) * (50 / 100) * N = 90 → 
  P = 10 := by
sorry

end percentage_calculation_l853_85389


namespace parallel_lines_a_equals_four_l853_85347

/-- A line in 2D space defined by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : ParametricLine) : Prop :=
  ∃ (k : ℝ), ∀ (t : ℝ), 
    (l1.x t - l1.x 0) * (l2.y t - l2.y 0) = k * (l1.y t - l1.y 0) * (l2.x t - l2.x 0)

/-- The first line l₁ --/
def l1 : ParametricLine where
  x := λ s => 2 * s + 1
  y := λ s => s

/-- The second line l₂ --/
def l2 (a : ℝ) : ParametricLine where
  x := λ t => a * t
  y := λ t => 2 * t - 1

/-- Theorem: If l₁ and l₂ are parallel, then a = 4 --/
theorem parallel_lines_a_equals_four :
  are_parallel l1 (l2 a) → a = 4 := by
  sorry

end parallel_lines_a_equals_four_l853_85347


namespace average_bowling_score_l853_85305

-- Define the players and their scores
def gretchen_score : ℕ := 120
def mitzi_score : ℕ := 113
def beth_score : ℕ := 85

-- Define the number of players
def num_players : ℕ := 3

-- Define the total score
def total_score : ℕ := gretchen_score + mitzi_score + beth_score

-- Theorem to prove
theorem average_bowling_score :
  (total_score : ℚ) / num_players = 106 := by
  sorry

end average_bowling_score_l853_85305


namespace probability_12_draws_10_red_l853_85369

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls

/-- The probability of drawing a red ball -/
def p_red : ℚ := red_balls / total_balls

/-- The probability of drawing a yellow ball -/
def p_yellow : ℚ := yellow_balls / total_balls

/-- The number of red balls needed to stop the process -/
def red_balls_needed : ℕ := 10

/-- The number of draws when the process stops -/
def total_draws : ℕ := 12

/-- The probability of drawing exactly 12 balls to get 10 red balls -/
theorem probability_12_draws_10_red (ξ : ℕ → ℚ) : 
  ξ total_draws = (Nat.choose (total_draws - 1) (red_balls_needed - 1)) * 
                  (p_red ^ red_balls_needed) * 
                  (p_yellow ^ (total_draws - red_balls_needed)) := by
  sorry

end probability_12_draws_10_red_l853_85369


namespace tenth_term_of_arithmetic_sequence_l853_85328

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def nthTerm (a : ℕ → ℝ) (n : ℕ) : ℝ := a n

theorem tenth_term_of_arithmetic_sequence
    (a : ℕ → ℝ)
    (h_arith : ArithmeticSequence a)
    (h_4th : nthTerm a 4 = 23)
    (h_6th : nthTerm a 6 = 43) :
  nthTerm a 10 = 83 := by
  sorry

end tenth_term_of_arithmetic_sequence_l853_85328


namespace rectangular_solid_surface_area_l853_85387

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a * b * c = 273 → 
  2 * (a * b + b * c + c * a) = 302 := by sorry

end rectangular_solid_surface_area_l853_85387


namespace constant_term_binomial_expansion_constant_term_binomial_expansion_proof_l853_85370

/-- The constant term in the binomial expansion of (2x - 1/√x)^6 is 60 -/
theorem constant_term_binomial_expansion : ℕ :=
  let n : ℕ := 6
  let a : ℝ → ℝ := λ x ↦ 2 * x
  let b : ℝ → ℝ := λ x ↦ -1 / Real.sqrt x
  let expansion : ℝ → ℝ := λ x ↦ (a x + b x) ^ n
  let constant_term : ℕ := 60
  constant_term

/-- Proof of the theorem -/
theorem constant_term_binomial_expansion_proof : 
  constant_term_binomial_expansion = 60 := by sorry

end constant_term_binomial_expansion_constant_term_binomial_expansion_proof_l853_85370


namespace curve_properties_l853_85383

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parametric curve C -/
def C (a : ℝ) (t : ℝ) : Point where
  x := 1 + 2 * t
  y := a * t^2

/-- The point M lies on the curve C -/
def M : Point := ⟨5, 4⟩

theorem curve_properties (a : ℝ) :
  (∃ t, C a t = M) →
  (a = 1 ∧ ∀ x y, (C 1 ((x - 1) / 2)).y = y ↔ 4 * y = (x - 1)^2) := by
  sorry


end curve_properties_l853_85383


namespace opposite_numbers_subtraction_not_always_smaller_l853_85392

-- Statement 1
theorem opposite_numbers (a b : ℝ) : a + b = 0 → a = -b := by sorry

-- Statement 2
theorem subtraction_not_always_smaller : ∃ x y : ℚ, x - y > y := by sorry

end opposite_numbers_subtraction_not_always_smaller_l853_85392


namespace inequality_proof_l853_85399

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end inequality_proof_l853_85399


namespace probability_three_heads_out_of_five_probability_three_heads_proof_l853_85385

/-- The probability of three specific coins out of five coming up heads when all five are flipped simultaneously -/
theorem probability_three_heads_out_of_five : ℚ :=
  1 / 8

/-- The total number of possible outcomes when flipping five coins -/
def total_outcomes : ℕ := 2^5

/-- The number of successful outcomes where three specific coins are heads -/
def successful_outcomes : ℕ := 2^2

theorem probability_three_heads_proof :
  (successful_outcomes : ℚ) / total_outcomes = probability_three_heads_out_of_five :=
sorry

end probability_three_heads_out_of_five_probability_three_heads_proof_l853_85385


namespace sufficient_but_not_necessary_l853_85380

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, |x| < 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ ¬(|x| < 1)) := by
  sorry

end sufficient_but_not_necessary_l853_85380


namespace inverse_proportion_ratio_l853_85364

/-- Given that x is inversely proportional to y, this theorem proves that
    if x₁/x₂ = 3/5, then y₁/y₂ = 5/3, where y₁ and y₂ are the corresponding
    y values for x₁ and x₂. -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y, x * y = k) →  -- x is inversely proportional to y
  x₁ / x₂ = 3 / 5 →
  y₁ / y₂ = 5 / 3 := by
sorry

end inverse_proportion_ratio_l853_85364


namespace square_circle_area_ratio_l853_85343

theorem square_circle_area_ratio :
  ∀ (r : ℝ) (s : ℝ),
    r > 0 →
    s > 0 →
    s = r * Real.sqrt 15 / 2 →
    (s^2) / (π * r^2) = 15 / (4 * π) := by
  sorry

end square_circle_area_ratio_l853_85343


namespace total_outlets_is_42_l853_85386

/-- The number of rooms in the house -/
def num_rooms : ℕ := 7

/-- The number of outlets required per room -/
def outlets_per_room : ℕ := 6

/-- The total number of outlets needed for the house -/
def total_outlets : ℕ := num_rooms * outlets_per_room

/-- Theorem stating that the total number of outlets needed is 42 -/
theorem total_outlets_is_42 : total_outlets = 42 := by
  sorry

end total_outlets_is_42_l853_85386


namespace inspection_arrangements_l853_85313

/-- Represents the number of liberal arts classes -/
def liberal_arts_classes : ℕ := 2

/-- Represents the number of science classes -/
def science_classes : ℕ := 4

/-- Represents the total number of classes -/
def total_classes : ℕ := liberal_arts_classes + science_classes

/-- Represents the number of ways to choose inspectors from science classes for liberal arts classes -/
def science_to_liberal_arts : ℕ := science_classes * (science_classes - 1)

/-- Represents the number of ways to arrange inspections within science classes -/
def science_arrangements : ℕ := 
  liberal_arts_classes * (liberal_arts_classes - 1) * (science_classes - 2) * (science_classes - 3) +
  liberal_arts_classes * (liberal_arts_classes - 1) +
  liberal_arts_classes * liberal_arts_classes * (science_classes - 2)

/-- The main theorem stating the total number of inspection arrangements -/
theorem inspection_arrangements : 
  science_to_liberal_arts * science_arrangements = 168 := by
  sorry

end inspection_arrangements_l853_85313


namespace sin_of_arcsin_plus_arctan_l853_85361

theorem sin_of_arcsin_plus_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan 1) = 7 * Real.sqrt 2 / 10 := by
  sorry

end sin_of_arcsin_plus_arctan_l853_85361


namespace factorization_cubic_minus_linear_l853_85359

theorem factorization_cubic_minus_linear (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end factorization_cubic_minus_linear_l853_85359


namespace max_product_of_tangent_circles_l853_85351

/-- Two circles C₁ and C₂ are externally tangent -/
def externally_tangent (a b : ℝ) : Prop :=
  a + b = 3

/-- The product of a and b -/
def product (a b : ℝ) : ℝ := a * b

/-- The theorem stating the maximum value of ab -/
theorem max_product_of_tangent_circles (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_tangent : externally_tangent a b) :
  product a b ≤ 9/4 :=
sorry

end max_product_of_tangent_circles_l853_85351


namespace james_printing_problem_l853_85376

/-- Calculates the minimum number of sheets required for printing books -/
def sheets_required (num_books : ℕ) (pages_per_book : ℕ) (sides_per_sheet : ℕ) (pages_per_side : ℕ) : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := sides_per_sheet * pages_per_side
  (total_pages + pages_per_sheet - 1) / pages_per_sheet

theorem james_printing_problem :
  sheets_required 5 800 3 6 = 223 := by
  sorry

end james_printing_problem_l853_85376


namespace camp_wonka_marshmallows_l853_85390

theorem camp_wonka_marshmallows : 
  ∀ (total_campers : ℕ) 
    (boys_fraction girls_fraction : ℚ) 
    (boys_toast_percent girls_toast_percent : ℚ),
  total_campers = 96 →
  boys_fraction = 2/3 →
  girls_fraction = 1/3 →
  boys_toast_percent = 1/2 →
  girls_toast_percent = 3/4 →
  (boys_fraction * ↑total_campers * boys_toast_percent + 
   girls_fraction * ↑total_campers * girls_toast_percent : ℚ) = 56 := by
sorry

end camp_wonka_marshmallows_l853_85390


namespace intersection_of_M_and_N_l853_85358

-- Define the sets M and N
def M : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def N : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end intersection_of_M_and_N_l853_85358


namespace expression_evaluation_l853_85372

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^3 + 1) / x * (y^3 + 1) / y + (x^3 - 1) / y * (y^3 - 1) / x = 2 * x^2 * y^2 + 2 / (x * y) := by
  sorry

end expression_evaluation_l853_85372


namespace coin_flip_probability_difference_l853_85330

/-- The probability of getting exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The theorem statement -/
theorem coin_flip_probability_difference :
  let p_four_heads := binomial_probability 5 4 (1/2)
  let p_five_heads := binomial_probability 5 5 (1/2)
  |p_four_heads - p_five_heads| = 1/8 := by
sorry

end coin_flip_probability_difference_l853_85330


namespace ratio_problem_l853_85321

theorem ratio_problem (a b : ℝ) 
  (h1 : b / a = 2) 
  (h2 : b = 15 - 4 * a) : 
  a = 5 / 2 := by sorry

end ratio_problem_l853_85321


namespace volume_of_S_l853_85367

-- Define the solid S' in the first octant
def S' : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
                   x + 2*y ≤ 1 ∧ 2*x + z ≤ 1 ∧ y + 2*z ≤ 1}

-- State the theorem about the volume of S'
theorem volume_of_S' : MeasureTheory.volume S' = 1/48 := by
  sorry

end volume_of_S_l853_85367


namespace sweater_discount_percentage_l853_85316

/-- Proves that the discount percentage is approximately 15.5% given the conditions -/
theorem sweater_discount_percentage (markup : ℝ) (profit : ℝ) :
  markup = 0.5384615384615385 →
  profit = 0.3 →
  let normal_price := 1 + markup
  let discounted_price := 1 + profit
  let discount := (normal_price - discounted_price) / normal_price
  abs (discount - 0.155) < 0.001 := by
sorry

end sweater_discount_percentage_l853_85316


namespace line_arrangement_with_restriction_l853_85388

def number_of_students : ℕ := 5

def total_arrangements (n : ℕ) : ℕ := n.factorial

def restricted_arrangements (n : ℕ) : ℕ := 
  (n - 1).factorial * 2

theorem line_arrangement_with_restriction :
  total_arrangements number_of_students - restricted_arrangements number_of_students = 72 := by
  sorry

end line_arrangement_with_restriction_l853_85388


namespace square_sum_of_special_integers_l853_85350

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 71)
  (h2 : x^2 * y + x * y^2 = 880) : 
  x^2 + y^2 = 146 := by sorry

end square_sum_of_special_integers_l853_85350


namespace divisible_by_65_l853_85312

theorem divisible_by_65 (n : ℕ) : ∃ k : ℤ, 5^n * (2^(2*n) - 3^n) + 2^n - 7^n = 65 * k := by
  sorry

end divisible_by_65_l853_85312


namespace regular_pentagon_diagonal_l853_85308

/-- For a regular pentagon with side length a, its diagonal d satisfies d = (√5 + 1)/2 * a -/
theorem regular_pentagon_diagonal (a : ℝ) (h : a > 0) :
  ∃ d : ℝ, d > 0 ∧ d = (Real.sqrt 5 + 1) / 2 * a := by
  sorry

end regular_pentagon_diagonal_l853_85308


namespace infinite_inequality_occurrences_l853_85317

theorem infinite_inequality_occurrences (a : ℕ → ℕ+) : 
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    ∀ n ∈ S, (1 : ℝ) + a n > (a (n-1) : ℝ) * (2 : ℝ) ^ (1 / n) :=
sorry

end infinite_inequality_occurrences_l853_85317


namespace sum_of_numbers_l853_85360

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) 
  (h4 : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end sum_of_numbers_l853_85360


namespace product_complex_polar_form_l853_85371

/-- The product of two complex numbers in polar form results in another complex number in polar form -/
theorem product_complex_polar_form 
  (z₁ : ℂ) (z₂ : ℂ) (r₁ θ₁ r₂ θ₂ : ℝ) :
  z₁ = r₁ * Complex.exp (θ₁ * Complex.I) →
  z₂ = r₂ * Complex.exp (θ₂ * Complex.I) →
  r₁ = 4 →
  r₂ = 5 →
  θ₁ = 45 * π / 180 →
  θ₂ = 72 * π / 180 →
  ∃ (r θ : ℝ), 
    z₁ * z₂ = r * Complex.exp (θ * Complex.I) ∧
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * π ∧
    r = 20 ∧
    θ = 297 * π / 180 := by
  sorry


end product_complex_polar_form_l853_85371


namespace worker_task_completion_time_l853_85336

theorem worker_task_completion_time 
  (x y : ℝ) -- x and y represent the time taken by the first and second worker respectively
  (h1 : (1/x) + (2/x + 2/y) = 11/20) -- Work completed in 3 hours
  (h2 : (1/x) + (1/y) = 1/2) -- Each worker completes half the task
  : x = 10 ∧ y = 8 := by
  sorry

end worker_task_completion_time_l853_85336


namespace start_time_is_6am_l853_85395

/-- Represents the hiking scenario with two hikers --/
structure HikingScenario where
  meetTime : ℝ       -- Time when hikers meet (in hours after midnight)
  rychlyEndTime : ℝ  -- Time when Mr. Rychlý finishes (in hours after midnight)
  loudaEndTime : ℝ   -- Time when Mr. Louda finishes (in hours after midnight)

/-- Calculates the start time of the hike given a HikingScenario --/
def calculateStartTime (scenario : HikingScenario) : ℝ :=
  scenario.meetTime - (scenario.rychlyEndTime - scenario.meetTime)

/-- Theorem stating that the start time is 6 AM (6 hours after midnight) --/
theorem start_time_is_6am (scenario : HikingScenario) 
  (h1 : scenario.meetTime = 10)
  (h2 : scenario.rychlyEndTime = 12)
  (h3 : scenario.loudaEndTime = 18) :
  calculateStartTime scenario = 6 := by
  sorry

#eval calculateStartTime { meetTime := 10, rychlyEndTime := 12, loudaEndTime := 18 }

end start_time_is_6am_l853_85395


namespace man_travel_distance_l853_85341

theorem man_travel_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 39 → distance = speed * time → distance = 78 := by
  sorry

end man_travel_distance_l853_85341


namespace simplify_expression_l853_85326

theorem simplify_expression (a b : ℝ) : 2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b := by
  sorry

end simplify_expression_l853_85326


namespace arthur_reading_time_ben_reading_time_l853_85322

-- Define the reading speed of the narrator
def narrator_speed : ℝ := 1

-- Define the time it takes the narrator to read the book (in hours)
def narrator_time : ℝ := 3

-- Define Arthur's reading speed relative to the narrator
def arthur_speed : ℝ := 3 * narrator_speed

-- Define Ben's reading speed relative to the narrator
def ben_speed : ℝ := 4 * narrator_speed

-- Theorem for Arthur's reading time
theorem arthur_reading_time :
  (narrator_time * narrator_speed) / arthur_speed = 1 := by sorry

-- Theorem for Ben's reading time
theorem ben_reading_time :
  (narrator_time * narrator_speed) / ben_speed = 3/4 := by sorry

end arthur_reading_time_ben_reading_time_l853_85322


namespace joyce_apples_l853_85337

def initial_apples : ℕ := 75
def apples_given : ℕ := 52
def apples_left : ℕ := 23

theorem joyce_apples : initial_apples = apples_given + apples_left := by
  sorry

end joyce_apples_l853_85337


namespace linear_function_problem_l853_85306

-- Define a linear function f
def f (x : ℝ) : ℝ := sorry

-- Define the inverse function of f
def f_inv (x : ℝ) : ℝ := sorry

-- State the theorem
theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 3 * f_inv x + 9) →         -- f(x) = 3f^(-1)(x) + 9
  f 3 = 6 →                                  -- f(3) = 6
  f 6 = 10.5 * Real.sqrt 3 - 4.5 :=          -- f(6) = 10.5√3 - 4.5
by sorry

end linear_function_problem_l853_85306


namespace flow_rate_is_twelve_l853_85304

/-- Represents the flow rate problem described in the question -/
def flow_rate_problem (tub_capacity : ℕ) (leak_rate : ℕ) (fill_time : ℕ) : ℕ :=
  let cycles := fill_time / 2
  let net_fill_per_cycle := (tub_capacity / cycles) + (2 * leak_rate)
  net_fill_per_cycle

/-- Theorem stating that the flow rate is 12 liters per minute under the given conditions -/
theorem flow_rate_is_twelve :
  flow_rate_problem 120 1 24 = 12 := by
  sorry

end flow_rate_is_twelve_l853_85304


namespace student_council_selections_l853_85362

-- Define the number of students
def n : ℕ := 6

-- Define the number of ways to select a two-person team
def two_person_selections : ℕ := 15

-- Define the number of ways to select a three-person team
def three_person_selections : ℕ := 20

-- Theorem statement
theorem student_council_selections :
  (Nat.choose n 2 = two_person_selections) →
  (Nat.choose n 3 = three_person_selections) :=
by sorry

end student_council_selections_l853_85362


namespace blanket_folding_ratio_l853_85309

theorem blanket_folding_ratio (initial_thickness final_thickness : ℝ) 
  (num_folds : ℕ) (ratio : ℝ) 
  (h1 : initial_thickness = 3)
  (h2 : final_thickness = 48)
  (h3 : num_folds = 4)
  (h4 : final_thickness = initial_thickness * ratio ^ num_folds) :
  ratio = 2 := by
sorry

end blanket_folding_ratio_l853_85309


namespace inequality_proof_l853_85307

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h_prod : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1 / a + 1 / b + 1 / c :=
by sorry

end inequality_proof_l853_85307


namespace addition_problem_l853_85335

def base_8_to_10 (n : ℕ) : ℕ := 
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem addition_problem (X Y : ℕ) (h : X < 8 ∧ Y < 8) :
  base_8_to_10 (500 + 10 * X + Y) + base_8_to_10 32 = base_8_to_10 (600 + 40 + X) →
  X + Y = 16 := by
  sorry

end addition_problem_l853_85335


namespace binomial_divisibility_iff_prime_l853_85355

theorem binomial_divisibility_iff_prime (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, m / 3 ≤ n ∧ n ≤ m / 2 → n ∣ Nat.choose n (m - 2*n)) ↔ Nat.Prime m :=
sorry

end binomial_divisibility_iff_prime_l853_85355


namespace square_sum_equals_one_l853_85384

theorem square_sum_equals_one (a b : ℝ) 
  (h : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1) : 
  a^2 + b^2 = 1 := by
  sorry

end square_sum_equals_one_l853_85384


namespace arithmetic_sequence_75th_term_l853_85379

/-- Given an arithmetic sequence with first term 2 and common difference 4,
    the 75th term of this sequence is 298. -/
theorem arithmetic_sequence_75th_term : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 4   -- Common difference
  let n : ℕ := 75  -- Term number we're looking for
  a₁ + (n - 1) * d = 298 := by
  sorry

end arithmetic_sequence_75th_term_l853_85379


namespace work_done_equals_21_l853_85397

def force : ℝ × ℝ := (5, 2)
def point_A : ℝ × ℝ := (-1, 3)
def point_B : ℝ × ℝ := (2, 6)

theorem work_done_equals_21 : 
  let displacement := (point_B.1 - point_A.1, point_B.2 - point_A.2)
  (force.1 * displacement.1 + force.2 * displacement.2) = 21 := by
  sorry

end work_done_equals_21_l853_85397
