import Mathlib

namespace no_reverse_multiply_all_ones_l994_99495

/-- Given a natural number, return the number with its digits reversed -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is composed of only ones -/
def all_ones (n : ℕ) : Prop := sorry

theorem no_reverse_multiply_all_ones :
  ∀ n : ℕ, n > 1 → ¬(all_ones (n * reverse_digits n)) := by
  sorry

end no_reverse_multiply_all_ones_l994_99495


namespace smallest_sum_B_plus_c_l994_99486

-- Define a digit in base 5
def is_base5_digit (B : ℕ) : Prop := 0 ≤ B ∧ B < 5

-- Define a base greater than or equal to 6
def is_valid_base (c : ℕ) : Prop := c ≥ 6

-- Define the equality BBB_5 = 44_c
def number_equality (B c : ℕ) : Prop := 31 * B = 4 * (c + 1)

-- Theorem statement
theorem smallest_sum_B_plus_c :
  ∀ B c : ℕ,
  is_base5_digit B →
  is_valid_base c →
  number_equality B c →
  (∀ B' c' : ℕ, is_base5_digit B' → is_valid_base c' → number_equality B' c' → B + c ≤ B' + c') →
  B + c = 8 :=
sorry

end smallest_sum_B_plus_c_l994_99486


namespace special_multiplication_pattern_l994_99479

theorem special_multiplication_pattern (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) := by
  sorry

end special_multiplication_pattern_l994_99479


namespace gain_amount_theorem_l994_99447

/-- The amount on which a gain was made, given the gain and gain percent -/
def amount_with_gain (gain : ℚ) (gain_percent : ℚ) : ℚ :=
  gain / (gain_percent / 100)

/-- Theorem: The amount on which a gain of 0.70 rupees was made, given a gain percent of 1%, is equal to 70 rupees -/
theorem gain_amount_theorem : amount_with_gain 0.70 1 = 70 := by
  sorry

end gain_amount_theorem_l994_99447


namespace complement_of_B_l994_99457

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the universal set U
def U (x : ℝ) : Set ℝ := A x ∪ B x

-- State the theorem
theorem complement_of_B (x : ℝ) :
  (B x ∪ (U x \ B x) = A x) →
  ((x = 0 → U x \ B x = {3}) ∧
   (x = Real.sqrt 3 → U x \ B x = {Real.sqrt 3}) ∧
   (x = -Real.sqrt 3 → U x \ B x = {-Real.sqrt 3})) :=
by sorry

end complement_of_B_l994_99457


namespace max_x_value_l994_99410

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |2*x - a|

-- State the theorem
theorem max_x_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∃ a : ℝ, ∀ x : ℝ, f x a ≤ 1/m + 4/n) →
  (∃ x : ℝ, ∀ y : ℝ, |y| ≤ |x| → ∃ a : ℝ, f y a ≤ 1/m + 4/n) ∧
  (∀ x : ℝ, (∀ y : ℝ, |y| ≤ |x| → ∃ a : ℝ, f y a ≤ 1/m + 4/n) → |x| ≤ 3) :=
by sorry


end max_x_value_l994_99410


namespace complex_cube_equals_negative_one_l994_99428

theorem complex_cube_equals_negative_one : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1/2 - Complex.I * (Real.sqrt 3)/2) ^ 3 = -1 :=
by sorry

end complex_cube_equals_negative_one_l994_99428


namespace tangent_line_problem_range_problem_l994_99463

noncomputable section

-- Define the function f(x) = x - ln x
def f (x : ℝ) : ℝ := x - Real.log x

-- Define the function g(x) = (e-1)x
def g (x : ℝ) : ℝ := (Real.exp 1 - 1) * x

-- Define the piecewise function F(x)
def F (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then f x else g x

-- Theorem for the tangent line problem
theorem tangent_line_problem (x₀ : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, k * x = f x + (x - x₀) * (1 - 1 / x₀)) →
  (x₀ = Real.exp 1 ∧ ∃ k : ℝ, k = 1 - 1 / Real.exp 1) :=
sorry

-- Theorem for the range problem
theorem range_problem (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, F a x = y) →
  a ≥ 1 / (Real.exp 1 - 1) :=
sorry

end tangent_line_problem_range_problem_l994_99463


namespace intersection_of_A_and_B_l994_99449

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2

-- Define set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem intersection_of_A_and_B 
  (A : Set ℝ) 
  (h : f '' A ⊆ B) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) :=
sorry

end intersection_of_A_and_B_l994_99449


namespace total_amount_after_stock_sale_l994_99412

def initial_wallet_amount : ℝ := 300
def initial_investment : ℝ := 2000
def stock_price_increase_percentage : ℝ := 0.30

theorem total_amount_after_stock_sale :
  initial_wallet_amount +
  initial_investment * (1 + stock_price_increase_percentage) =
  2900 :=
by sorry

end total_amount_after_stock_sale_l994_99412


namespace equal_book_distribution_l994_99408

theorem equal_book_distribution (total_students : ℕ) (girls : ℕ) (boys : ℕ) 
  (total_books : ℕ) (girls_books : ℕ) :
  total_students = girls + boys →
  total_books = 375 →
  girls = 15 →
  boys = 10 →
  girls_books = 225 →
  ∃ (books_per_student : ℕ), 
    books_per_student = 15 ∧
    girls_books = girls * books_per_student ∧
    total_books = total_students * books_per_student :=
by sorry

end equal_book_distribution_l994_99408


namespace parentheses_removal_correct_l994_99490

theorem parentheses_removal_correct (a b : ℤ) : -2*a + 3*(b - 1) = -2*a + 3*b - 3 := by
  sorry

end parentheses_removal_correct_l994_99490


namespace arithmetic_sequence_formula_l994_99426

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 2 = -1 ∧ a 4 = 3 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The general formula for the arithmetic sequence -/
def GeneralFormula (n : ℕ) : ℤ := 2 * n - 5

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  ArithmeticSequence a → ∀ n : ℕ, a n = GeneralFormula n := by
  sorry

end arithmetic_sequence_formula_l994_99426


namespace preston_received_correct_amount_l994_99461

/-- Calculates the total amount Preston received from Abra Company's order --/
def prestonReceived (
  sandwichPrice : ℚ)
  (sideDishPrice : ℚ)
  (drinkPrice : ℚ)
  (deliveryFee : ℚ)
  (sandwichCount : ℕ)
  (sideDishCount : ℕ)
  (drinkCount : ℕ)
  (tipPercentage : ℚ)
  (discountPercentage : ℚ) : ℚ :=
  let foodCost := sandwichPrice * sandwichCount + sideDishPrice * sideDishCount
  let drinkCost := drinkPrice * drinkCount
  let discountAmount := discountPercentage * foodCost
  let subtotal := foodCost + drinkCost - discountAmount + deliveryFee
  let tipAmount := tipPercentage * subtotal
  subtotal + tipAmount

/-- Theorem stating that Preston received $158.95 from Abra Company's order --/
theorem preston_received_correct_amount :
  prestonReceived 5 3 (3/2) 20 18 10 15 (1/10) (15/100) = 15895/100 := by
  sorry

end preston_received_correct_amount_l994_99461


namespace complex_fraction_simplification_l994_99487

theorem complex_fraction_simplification :
  (2 / 5 + 3 / 4) / (4 / 9 + 1 / 6) = 207 / 110 := by
  sorry

end complex_fraction_simplification_l994_99487


namespace smallest_n_congruence_l994_99462

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (3 * n) % 26 = 1367 % 26 ∧
  ∀ (m : ℕ), m > 0 ∧ (3 * m) % 26 = 1367 % 26 → n ≤ m :=
by sorry

end smallest_n_congruence_l994_99462


namespace ellipse_eccentricity_l994_99437

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a line l passing through a vertex (0, b) and a focus (c, 0),
    if the distance from the center to l is b/4,
    then the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1 / Real.sqrt ((1 / c^2) + (1 / b^2)) = b / 4) →
  c / a = 1 / 2 :=
sorry

end ellipse_eccentricity_l994_99437


namespace angle_value_in_plane_figure_l994_99400

theorem angle_value_in_plane_figure (x : ℝ) : 
  x > 0 ∧ 
  x + x + 140 = 360 → 
  x = 110 := by
sorry

end angle_value_in_plane_figure_l994_99400


namespace shopping_trip_result_l994_99438

def shopping_trip (initial_amount : ℝ) (video_game_price : ℝ) (video_game_discount : ℝ)
  (goggles_percent : ℝ) (goggles_tax : ℝ) (jacket_price : ℝ) (jacket_discount : ℝ)
  (book_percent : ℝ) (book_tax : ℝ) (gift_card : ℝ) : ℝ :=
  let video_game_cost := video_game_price * (1 - video_game_discount)
  let remaining_after_game := initial_amount - video_game_cost
  let goggles_cost := remaining_after_game * goggles_percent * (1 + goggles_tax)
  let remaining_after_goggles := remaining_after_game - goggles_cost
  let jacket_cost := jacket_price * (1 - jacket_discount)
  let remaining_after_jacket := remaining_after_goggles - jacket_cost
  let book_cost := remaining_after_jacket * book_percent * (1 + book_tax)
  remaining_after_jacket - book_cost

theorem shopping_trip_result :
  shopping_trip 200 60 0.15 0.20 0.08 80 0.25 0.10 0.05 20 = 50.85 := by
  sorry

#eval shopping_trip 200 60 0.15 0.20 0.08 80 0.25 0.10 0.05 20

end shopping_trip_result_l994_99438


namespace special_line_equation_l994_99464

/-- A line passing through point M(3, -4) with intercepts on the coordinate axes that are opposite numbers -/
structure SpecialLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The line passes through point M(3, -4)
  passes_through_M : slope * 3 + y_intercept = -4
  -- The intercepts on the coordinate axes are opposite numbers
  opposite_intercepts : (y_intercept = 0 ∧ -y_intercept / slope = 0) ∨ 
                        (y_intercept ≠ 0 ∧ -y_intercept / slope = -y_intercept)

/-- The equation of the special line is either x + y = -1 or 4x + 3y = 0 -/
theorem special_line_equation (l : SpecialLine) : 
  (l.slope = -1 ∧ l.y_intercept = -1) ∨ (l.slope = -4/3 ∧ l.y_intercept = 0) := by
  sorry

#check special_line_equation

end special_line_equation_l994_99464


namespace proof_by_contradiction_assumption_l994_99404

theorem proof_by_contradiction_assumption (a b : ℤ) : 
  (5 ∣ a * b) → (5 ∣ a ∨ 5 ∣ b) ↔ 
  (¬(5 ∣ a) ∧ ¬(5 ∣ b)) → False :=
by sorry

end proof_by_contradiction_assumption_l994_99404


namespace total_spent_is_two_dollars_l994_99419

/-- The price of a single pencil in cents -/
def pencil_price : ℕ := 20

/-- The number of pencils Tolu wants -/
def tolu_pencils : ℕ := 3

/-- The number of pencils Robert wants -/
def robert_pencils : ℕ := 5

/-- The number of pencils Melissa wants -/
def melissa_pencils : ℕ := 2

/-- Theorem: The total amount spent by the students is $2.00 -/
theorem total_spent_is_two_dollars :
  (tolu_pencils * pencil_price + robert_pencils * pencil_price + melissa_pencils * pencil_price) / 100 = 2 := by
  sorry

end total_spent_is_two_dollars_l994_99419


namespace simplify_linear_expression_l994_99467

theorem simplify_linear_expression (x : ℝ) : 5*x + 2*x + 7*x = 14*x := by
  sorry

end simplify_linear_expression_l994_99467


namespace archer_probabilities_l994_99407

/-- Represents the probability of an archer hitting a target -/
def hit_probability : ℚ := 2/3

/-- Represents the number of shots taken -/
def num_shots : ℕ := 5

/-- Calculates the probability of hitting the target exactly k times in n shots -/
def prob_exact_hits (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

/-- Calculates the probability of hitting the target k times in a row and missing n-k times in n shots -/
def prob_consecutive_hits (n k : ℕ) : ℚ :=
  (n - k + 1 : ℚ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

theorem archer_probabilities :
  (prob_exact_hits num_shots 2 = 40/243) ∧
  (prob_consecutive_hits num_shots 3 = 8/81) := by
  sorry

end archer_probabilities_l994_99407


namespace sine_addition_formula_l994_99405

theorem sine_addition_formula (α β : Real) : 
  Real.sin (α - β) * Real.cos β + Real.cos (α - β) * Real.sin β = Real.sin α := by
  sorry

end sine_addition_formula_l994_99405


namespace largest_sum_of_digits_for_reciprocal_fraction_l994_99442

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc -/
def DecimalABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem statement -/
theorem largest_sum_of_digits_for_reciprocal_fraction :
  ∀ (a b c : Digit) (y : ℕ+),
    (0 < y.val) → (y.val ≤ 16) →
    (DecimalABC a b c = 1 / y) →
    (∀ (a' b' c' : Digit) (y' : ℕ+),
      (0 < y'.val) → (y'.val ≤ 16) →
      (DecimalABC a' b' c' = 1 / y') →
      (a.val + b.val + c.val ≥ a'.val + b'.val + c'.val)) →
    (a.val + b.val + c.val = 8) :=
by sorry

end largest_sum_of_digits_for_reciprocal_fraction_l994_99442


namespace milk_production_days_l994_99474

/-- Given that x cows produce x+1 cans of milk in x+2 days, 
    this theorem proves the number of days it takes x+3 cows to produce x+5 cans of milk. -/
theorem milk_production_days (x : ℝ) (h : x > 0) : 
  let initial_cows := x
  let initial_milk := x + 1
  let initial_days := x + 2
  let new_cows := x + 3
  let new_milk := x + 5
  let daily_production_per_cow := initial_milk / (initial_cows * initial_days)
  let days_for_new_production := new_milk / (new_cows * daily_production_per_cow)
  days_for_new_production = x * (x + 2) * (x + 5) / ((x + 1) * (x + 3)) :=
by sorry

end milk_production_days_l994_99474


namespace max_sphere_radius_squared_l994_99468

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- The specific problem setup -/
def problemSetup : ConeSphereProblem :=
  { cone1 := { baseRadius := 4, height := 10 }
  , cone2 := { baseRadius := 4, height := 10 }
  , intersectionDistance := 2
  , sphereRadius := 0  -- To be maximized
  }

/-- The theorem stating the maximal value of r^2 -/
theorem max_sphere_radius_squared (setup : ConeSphereProblem) :
  setup.cone1 = setup.cone2 →
  setup.cone1.baseRadius = 4 →
  setup.cone1.height = 10 →
  setup.intersectionDistance = 2 →
  ∃ (r : ℝ), r^2 ≤ 144/29 ∧
    ∀ (s : ℝ), (∃ (config : ConeSphereProblem),
      config.cone1 = setup.cone1 ∧
      config.cone2 = setup.cone2 ∧
      config.intersectionDistance = setup.intersectionDistance ∧
      config.sphereRadius = s) →
    s^2 ≤ r^2 :=
by
  sorry


end max_sphere_radius_squared_l994_99468


namespace horner_method_for_f_l994_99492

def f (x : ℝ) : ℝ := x^6 + 2*x^5 + 4*x^3 + 5*x^2 + 6*x + 12

theorem horner_method_for_f :
  f 3 = 588 := by sorry

end horner_method_for_f_l994_99492


namespace polygon_interior_angles_l994_99481

theorem polygon_interior_angles (P : ℕ) (h1 : P > 2) : 
  (∃ (a d : ℝ), 
    a = 20 ∧ 
    a + (P - 1) * d = 160 ∧ 
    (P / 2 : ℝ) * (a + (a + (P - 1) * d)) = 180 * (P - 2)) → 
  P = 4 := by
sorry

end polygon_interior_angles_l994_99481


namespace remaining_fuel_fraction_l994_99430

def tank_capacity : ℚ := 12
def round_trip_distance : ℚ := 20
def miles_per_gallon : ℚ := 5

theorem remaining_fuel_fraction :
  (tank_capacity - round_trip_distance / miles_per_gallon) / tank_capacity = 2/3 := by
  sorry

end remaining_fuel_fraction_l994_99430


namespace spider_legs_proof_l994_99415

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size (L : ℕ) : ℕ := L / 2 + 10

theorem spider_legs_proof :
  (∀ L : ℕ, group_size L * L = 112) → spider_legs = 8 := by
  sorry

end spider_legs_proof_l994_99415


namespace polynomial_remainder_l994_99455

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 8*x^3 + 12*x^2 + 20*x - 10
  let g : ℝ → ℝ := λ x => x - 2
  ∃ q : ℝ → ℝ, f x = g x * q x + 30 := by
sorry

end polynomial_remainder_l994_99455


namespace arithmetic_geometric_sequence_problem_l994_99482

theorem arithmetic_geometric_sequence_problem (a b : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) - a n = d) →  -- a_n is arithmetic with common difference d
  d ≠ 0 →  -- d is not equal to 0
  a 2046 + a 1978 - (a 2012)^2 = 0 →  -- given condition
  (∃ r, ∀ n, b (n + 1) = r * b n) →  -- b_n is geometric
  b 2012 = a 2012 →  -- given condition
  b 2010 * b 2014 = 4 := by
sorry

end arithmetic_geometric_sequence_problem_l994_99482


namespace cylinder_cross_section_area_coefficient_sum_l994_99441

/-- The area of a cross-section in a cylinder --/
theorem cylinder_cross_section_area :
  ∀ (r : ℝ) (θ : ℝ),
  r = 8 →
  θ = π / 2 →
  ∃ (A : ℝ),
  A = 16 * Real.sqrt 3 * π + 16 * Real.sqrt 6 ∧
  A = (r^2 * θ / 4 + r^2 * Real.sin (θ / 2) * Real.cos (θ / 2)) * Real.sqrt 3 :=
by sorry

/-- The sum of coefficients in the area expression --/
theorem coefficient_sum :
  ∃ (d e : ℝ) (f : ℕ),
  16 * Real.sqrt 3 * π + 16 * Real.sqrt 6 = d * π + e * Real.sqrt f ∧
  d + e + f = 38 :=
by sorry

end cylinder_cross_section_area_coefficient_sum_l994_99441


namespace sphere_surface_area_ratio_l994_99401

theorem sphere_surface_area_ratio (r1 r2 : ℝ) (h1 : r1 = 40) (h2 : r2 = 10) :
  (4 * π * r1^2) / (4 * π * r2^2) = 16 := by
  sorry

end sphere_surface_area_ratio_l994_99401


namespace unique_x_for_all_y_l994_99459

theorem unique_x_for_all_y : ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 18 * y + x - 2 = 0 :=
by
  -- The proof goes here
  sorry

end unique_x_for_all_y_l994_99459


namespace quadratic_equation_roots_l994_99418

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 2*x + 1 = 0 ∧ a * y^2 - 2*y + 1 = 0) → 
  (a < 1 ∧ a ≠ 0) :=
by sorry

end quadratic_equation_roots_l994_99418


namespace triangle_is_equilateral_l994_99460

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b

def condition2 (t : Triangle) : Prop :=
  2 * Real.cos t.A * Real.sin t.B = Real.sin t.C

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.C = Real.pi / 3 :=
sorry

end triangle_is_equilateral_l994_99460


namespace dave_apps_unchanged_l994_99453

theorem dave_apps_unchanged (initial_files final_files deleted_files final_apps : ℕ) :
  initial_files = final_files + deleted_files →
  initial_files = 24 →
  final_files = 21 →
  deleted_files = 3 →
  final_apps = 17 →
  initial_apps = final_apps :=
by
  sorry

#check dave_apps_unchanged

end dave_apps_unchanged_l994_99453


namespace sequence_property_l994_99432

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a n - a (n + 1) = (a n * a (n + 1)) / (2^(n - 1))

theorem sequence_property (a : ℕ → ℚ) (k : ℕ) 
  (h1 : RecurrenceSequence a) 
  (h2 : a 2 = -1)
  (h3 : a k = 16 * a 8)
  (h4 : k > 0) : 
  k = 12 := by
  sorry

end sequence_property_l994_99432


namespace soccer_ball_cost_l994_99489

theorem soccer_ball_cost (F S : ℝ) 
  (eq1 : 3 * F + S = 155) 
  (eq2 : 2 * F + 3 * S = 220) : 
  S = 50 := by
sorry

end soccer_ball_cost_l994_99489


namespace b_over_a_range_l994_99475

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if the roots represent eccentricities of conic sections -/
def has_conic_eccentricities (eq : CubicEquation) : Prop :=
  ∃ (e₁ e₂ e₃ : ℝ), 
    e₁^3 + eq.a * e₁^2 + eq.b * e₁ + eq.c = 0 ∧
    e₂^3 + eq.a * e₂^2 + eq.b * e₂ + eq.c = 0 ∧
    e₃^3 + eq.a * e₃^2 + eq.b * e₃ + eq.c = 0 ∧
    (0 ≤ e₁ ∧ e₁ < 1) ∧  -- ellipse eccentricity
    (e₂ > 1) ∧           -- hyperbola eccentricity
    (e₃ = 1)             -- parabola eccentricity

/-- The main theorem stating the range of b/a -/
theorem b_over_a_range (eq : CubicEquation) 
  (h : has_conic_eccentricities eq) : 
  -2 < eq.b / eq.a ∧ eq.b / eq.a < -1/2 :=
by sorry

end b_over_a_range_l994_99475


namespace max_profit_theorem_l994_99422

/-- Represents the profit function for a product sale scenario. -/
def profit_function (x : ℝ) : ℝ := -160 * x^2 + 560 * x + 3120

/-- Represents the factory price of the product. -/
def factory_price : ℝ := 3

/-- Represents the initial retail price. -/
def initial_retail_price : ℝ := 4

/-- Represents the initial monthly sales volume. -/
def initial_sales_volume : ℝ := 400

/-- Represents the change in sales volume for every 0.5 CNY price change. -/
def sales_volume_change : ℝ := 40

/-- Theorem stating the maximum profit and the corresponding selling prices. -/
theorem max_profit_theorem :
  (∃ (x : ℝ), x = 1.5 ∨ x = 2) ∧
  (∀ (y : ℝ), y ≤ 3600 → ∃ (x : ℝ), profit_function x = y) ∧
  profit_function 1.5 = 3600 ∧
  profit_function 2 = 3600 := by sorry

end max_profit_theorem_l994_99422


namespace number_transformation_l994_99424

theorem number_transformation (x : ℝ) : 
  x + 0.40 * x = 1680 → x * 0.80 * 1.15 = 1104 := by
  sorry

end number_transformation_l994_99424


namespace intersection_sum_l994_99421

/-- 
Given two lines y = 2x + c and y = 4x + d that intersect at the point (8, 12),
prove that c + d = -24
-/
theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, y = 2*x + c) →
  (∀ x y : ℝ, y = 4*x + d) →
  12 = 2*8 + c →
  12 = 4*8 + d →
  c + d = -24 := by
sorry

end intersection_sum_l994_99421


namespace expression_evaluation_l994_99491

theorem expression_evaluation :
  (∀ a : ℤ, a = -3 → (a + 3)^2 + (2 + a) * (2 - a) = -5) ∧
  (∀ x : ℤ, x = -3 → 2 * x * (3 * x^2 - 4 * x + 1) - 3 * x^2 * (x - 3) = -78) :=
by sorry

end expression_evaluation_l994_99491


namespace digit_sum_property_l994_99440

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : sum_of_digits n = 100)
  (h2 : sum_of_digits (44 * n) = 800) :
  sum_of_digits (3 * n) = 300 := by sorry

end digit_sum_property_l994_99440


namespace line_circle_intersection_l994_99466

/-- A line in 2D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if a line intersects a circle at exactly one point -/
def intersectsAtOnePoint (l : ParametricLine) (c : Circle) : Prop := sorry

/-- The main theorem -/
theorem line_circle_intersection (m : ℝ) :
  let l : ParametricLine := { x := λ t => 3 * t, y := λ t => 4 * t + m }
  let c : Circle := { center := (1, 0), radius := 1 }
  intersectsAtOnePoint l c → m = 1/3 ∨ m = -3 := by
  sorry


end line_circle_intersection_l994_99466


namespace james_heavy_lifting_days_l994_99413

/-- Calculates the number of days until James can lift heavy again after an injury. -/
def daysUntilHeavyLifting (painSubsideDays : ℕ) (healingMultiplier : ℕ) (waitAfterHealingDays : ℕ) (waitBeforeHeavyLiftingWeeks : ℕ) : ℕ :=
  let healingDays := painSubsideDays * healingMultiplier
  let totalDaysBeforeWorkout := healingDays + waitAfterHealingDays
  let waitBeforeHeavyLiftingDays := waitBeforeHeavyLiftingWeeks * 7
  totalDaysBeforeWorkout + waitBeforeHeavyLiftingDays

/-- Theorem stating that James can lift heavy again after 39 days given the specific conditions. -/
theorem james_heavy_lifting_days :
  daysUntilHeavyLifting 3 5 3 3 = 39 := by
  sorry

#eval daysUntilHeavyLifting 3 5 3 3

end james_heavy_lifting_days_l994_99413


namespace sum_of_cube_edges_l994_99436

-- Define a cube with edge length 15
def cube_edge_length : ℝ := 15

-- Define the number of edges in a cube
def cube_num_edges : ℕ := 12

-- Theorem: The sum of all edge lengths in the cube is 180
theorem sum_of_cube_edges :
  cube_edge_length * cube_num_edges = 180 := by
  sorry

end sum_of_cube_edges_l994_99436


namespace garden_bugs_l994_99403

theorem garden_bugs (B : ℕ) : 0.8 * (B : ℝ) - 12 * 7 = 236 → B = 400 := by
  sorry

end garden_bugs_l994_99403


namespace m_range_theorem_l994_99435

-- Define the statements p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

-- Define the set A (where p is true)
def A : Set ℝ := {x | p x}

-- Define the set B (where q is true)
def B (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem m_range_theorem :
  ∀ m : ℝ, 
    (∀ x : ℝ, x ∈ A → x ∈ B m) ∧  -- p implies q
    (∃ x : ℝ, x ∈ B m ∧ x ∉ A) ∧  -- q does not imply p
    m ≥ 40 ∧ m < 50               -- m is in [40, 50)
  ↔ m ∈ Set.Icc 40 50 := by sorry

end m_range_theorem_l994_99435


namespace smallest_satisfying_number_l994_99471

def satisfiesConditions (n : Nat) : Prop :=
  n % 43 = 0 ∧
  n < 43 * 9 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1

theorem smallest_satisfying_number :
  satisfiesConditions 301 ∧
  ∀ m : Nat, m < 301 → ¬satisfiesConditions m :=
by sorry

end smallest_satisfying_number_l994_99471


namespace pirate_count_l994_99473

theorem pirate_count : ∃ p : ℕ, 
  p > 0 ∧ 
  (∃ (participants : ℕ), participants = p - 10 ∧ 
    (54 : ℚ) / 100 * participants = (↑⌊(54 : ℚ) / 100 * participants⌋ : ℚ) ∧ 
    (34 : ℚ) / 100 * participants = (↑⌊(34 : ℚ) / 100 * participants⌋ : ℚ) ∧ 
    (2 : ℚ) / 3 * p = (↑⌊(2 : ℚ) / 3 * p⌋ : ℚ)) ∧ 
  p = 60 := by
  sorry

end pirate_count_l994_99473


namespace black_lambs_count_l994_99458

theorem black_lambs_count (total : ℕ) (white : ℕ) (h1 : total = 6048) (h2 : white = 193) :
  total - white = 5855 := by
  sorry

end black_lambs_count_l994_99458


namespace simplify_expression_l994_99498

theorem simplify_expression (n : ℕ) : (2^(n+5) - 3*(2^n)) / (3*(2^(n+3))) = 29 / 24 := by
  sorry

end simplify_expression_l994_99498


namespace area_curve_C_m_1_intersection_points_l994_99427

-- Define the curve C
def curve_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = m}

-- Define the ellipse
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1}

-- Theorem 1: Area enclosed by curve C when m = 1
theorem area_curve_C_m_1 :
  MeasureTheory.volume (curve_C 1) = 2 := by sorry

-- Theorem 2: Intersection points of curve C and ellipse
theorem intersection_points (m : ℝ) :
  (∃ (a b c d : ℝ × ℝ), a ∈ curve_C m ∩ ellipse ∧
                         b ∈ curve_C m ∩ ellipse ∧
                         c ∈ curve_C m ∩ ellipse ∧
                         d ∈ curve_C m ∩ ellipse ∧
                         a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔
  (2 < m ∧ m < 3) ∨ m = Real.sqrt 13 := by sorry

end area_curve_C_m_1_intersection_points_l994_99427


namespace bakers_cakes_l994_99493

/-- Baker's cake selling problem -/
theorem bakers_cakes (initial_cakes : ℕ) (cakes_left : ℕ) (h1 : initial_cakes = 48) (h2 : cakes_left = 4) :
  initial_cakes - cakes_left = 44 := by
  sorry

end bakers_cakes_l994_99493


namespace laundry_problem_solution_l994_99499

/-- Represents the laundry shop scenario --/
structure LaundryShop where
  price_per_kilo : ℝ
  kilos_two_days_ago : ℝ
  total_earnings : ℝ

/-- Calculates the total kilos of laundry for three days --/
def total_kilos (shop : LaundryShop) : ℝ :=
  shop.kilos_two_days_ago + 
  (shop.kilos_two_days_ago + 5) + 
  2 * (shop.kilos_two_days_ago + 5)

/-- Theorem stating the solution to the laundry problem --/
theorem laundry_problem_solution (shop : LaundryShop) 
  (h1 : shop.price_per_kilo = 2)
  (h2 : shop.total_earnings = 70) :
  shop.kilos_two_days_ago = 5 :=
by
  sorry


end laundry_problem_solution_l994_99499


namespace unique_p_l994_99402

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : Prop :=
  5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p

/-- A number is a natural root of the cubic equation -/
def is_natural_root (p : ℝ) (x : ℕ) : Prop :=
  cubic_equation p (x : ℝ)

/-- The cubic equation has exactly three natural roots -/
def has_three_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_natural_root p a ∧ is_natural_root p b ∧ is_natural_root p c ∧
    ∀ (x : ℕ), is_natural_root p x → (x = a ∨ x = b ∨ x = c)

/-- The main theorem: 76 is the only real number satisfying the conditions -/
theorem unique_p : ∀ (p : ℝ), has_three_natural_roots p ↔ p = 76 := by
  sorry

end unique_p_l994_99402


namespace star_value_l994_99425

theorem star_value (star : ℝ) : star * 12^2 = 12^7 → star = 12^5 := by
  sorry

end star_value_l994_99425


namespace find_A_l994_99470

/-- Represents a three-digit number of the form 2A3 where A is a single digit -/
def threeDigitNumber (A : Nat) : Nat :=
  200 + 10 * A + 3

/-- Condition that A is a single digit -/
def isSingleDigit (A : Nat) : Prop :=
  A ≥ 0 ∧ A ≤ 9

theorem find_A :
  ∀ A : Nat,
    isSingleDigit A →
    (threeDigitNumber A).mod 11 = 0 →
    A = 5 :=
by
  sorry

end find_A_l994_99470


namespace no_given_factors_l994_99478

def f (x : ℝ) : ℝ := x^5 + 3*x^3 - 4*x^2 + 12*x + 8

theorem no_given_factors :
  (∀ x, f x ≠ 0 → x + 1 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 + 1 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 - 2 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 + 3 ≠ 0) := by
  sorry

end no_given_factors_l994_99478


namespace sale_price_for_55_percent_profit_l994_99411

/-- Proves that the sale price for making a 55% profit is $2792, given the conditions. -/
theorem sale_price_for_55_percent_profit 
  (equal_profit_loss : ∀ (cp sp_profit : ℝ), sp_profit - cp = cp - 448)
  (profit_amount : ∀ (cp : ℝ), 0.55 * cp = 992) :
  ∃ (cp : ℝ), cp + 992 = 2792 :=
by sorry

end sale_price_for_55_percent_profit_l994_99411


namespace david_average_marks_l994_99472

def david_marks : List ℕ := [74, 65, 82, 67, 90]

theorem david_average_marks :
  (david_marks.sum : ℚ) / david_marks.length = 75.6 := by sorry

end david_average_marks_l994_99472


namespace darren_fergie_equal_debt_l994_99416

/-- Represents the amount owed after t days with simple interest -/
def amountOwed (principal : ℝ) (rate : ℝ) (days : ℝ) : ℝ :=
  principal * (1 + rate * days)

/-- The problem statement -/
theorem darren_fergie_equal_debt : ∃ t : ℝ, 
  t = 20 ∧ 
  amountOwed 100 0.10 t = amountOwed 150 0.05 t :=
sorry

end darren_fergie_equal_debt_l994_99416


namespace equation_to_lines_l994_99480

/-- The set of points satisfying the given equation is equivalent to the union of two lines -/
theorem equation_to_lines : 
  ∀ x y : ℝ, (2*x^2 + y^2 + 3*x*y + 3*x + y = 2) ↔ 
  (y = -x - 2 ∨ y = -2*x + 1) := by sorry

end equation_to_lines_l994_99480


namespace car_payment_months_l994_99417

def car_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem car_payment_months : 
  (car_price - initial_payment) / monthly_payment = 19 := by
  sorry

end car_payment_months_l994_99417


namespace sufficient_condition_for_existence_l994_99456

theorem sufficient_condition_for_existence (a : ℝ) :
  (a ≥ 2) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ x₀^2 - a ≤ 0) ∧
  ¬((∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ x₀^2 - a ≤ 0) → (a ≥ 2)) :=
by sorry

end sufficient_condition_for_existence_l994_99456


namespace S_is_empty_l994_99423

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ r : ℝ, (2 + 5*i)*z = r ∧ z.re = 2*z.im}

-- Theorem statement
theorem S_is_empty : S = ∅ := by sorry

end S_is_empty_l994_99423


namespace jellybean_problem_l994_99445

theorem jellybean_problem :
  ∃ (n : ℕ), n = 151 ∧ 
  (∀ m : ℕ, m ≥ 150 ∧ m % 17 = 15 → m ≥ n) ∧ 
  n ≥ 150 ∧ 
  n % 17 = 15 := by
sorry

end jellybean_problem_l994_99445


namespace f_decreasing_on_interval_l994_99465

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x y, -1 < x ∧ x < y ∧ y < 11 → f x > f y := by
  sorry

end f_decreasing_on_interval_l994_99465


namespace rhombus_and_rectangle_diagonals_bisect_l994_99454

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect_each_other (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem rhombus_and_rectangle_diagonals_bisect :
  ∀ q : Quadrilateral, 
    (is_rhombus q ∨ is_rectangle q) → diagonals_bisect_each_other q :=
by sorry

end rhombus_and_rectangle_diagonals_bisect_l994_99454


namespace bob_has_winning_strategy_l994_99497

/-- Represents a cell in the grid -/
structure Cell :=
  (row : ℕ)
  (col : ℕ)
  (value : ℚ)

/-- Represents the game state -/
structure GameState :=
  (grid : List (List Cell))
  (current_player : Bool)  -- true for Alice, false for Bob

/-- Checks if a cell is part of a continuous path from top to bottom -/
def is_part_of_path (grid : List (List Cell)) (cell : Cell) : Prop :=
  sorry

/-- Determines if there exists a winning path for Alice -/
def exists_winning_path (state : GameState) : Prop :=
  sorry

/-- Represents a player's strategy -/
def Strategy := GameState → Cell

/-- Determines if a strategy is winning for Bob -/
def is_winning_strategy_for_bob (strategy : Strategy) : Prop :=
  ∀ (state : GameState), 
    (state.current_player = false) → 
    ¬(exists_winning_path (state))

/-- The main theorem stating that Bob has a winning strategy -/
theorem bob_has_winning_strategy : 
  ∃ (strategy : Strategy), is_winning_strategy_for_bob strategy :=
sorry

end bob_has_winning_strategy_l994_99497


namespace sara_remaining_marbles_l994_99477

def initial_black_marbles : ℕ := 792
def marbles_taken : ℕ := 233

theorem sara_remaining_marbles :
  initial_black_marbles - marbles_taken = 559 :=
by sorry

end sara_remaining_marbles_l994_99477


namespace elliptic_curve_solutions_l994_99496

theorem elliptic_curve_solutions (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  ∀ (x y : ℤ), y^2 = x^3 - p^2*x ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p ∧ y = 0) ∨ 
    (x = -p ∧ y = 0) ∨ 
    (x = (p^2 + 1)/2 ∧ (y = ((p^2 - 1)/2)*p ∨ y = -((p^2 - 1)/2)*p)) :=
by sorry

end elliptic_curve_solutions_l994_99496


namespace mrs_hilt_bug_count_l994_99484

theorem mrs_hilt_bug_count (flowers_per_bug : ℕ) (total_flowers : ℕ) (num_bugs : ℕ) : 
  flowers_per_bug = 2 →
  total_flowers = 6 →
  num_bugs * flowers_per_bug = total_flowers →
  num_bugs = 3 := by
sorry

end mrs_hilt_bug_count_l994_99484


namespace arithmetic_sequence_2015th_term_l994_99450

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_eq_1 : a 1 = 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  d_neq_0 : a 2 - a 1 ≠ 0
  geometric_subset : (a 2 / a 1) = (a 5 / a 2)

/-- The 2015th term of the arithmetic sequence is 4029 -/
theorem arithmetic_sequence_2015th_term (seq : ArithmeticSequence) : seq.a 2015 = 4029 := by
  sorry

end arithmetic_sequence_2015th_term_l994_99450


namespace alloy_mixture_l994_99443

/-- Proves that the amount of the first alloy used is 15 kg given the specified conditions -/
theorem alloy_mixture (x : ℝ) 
  (h1 : 0.12 * x + 0.08 * 35 = 0.092 * (x + 35)) : x = 15 := by
  sorry

#check alloy_mixture

end alloy_mixture_l994_99443


namespace least_value_quadratic_l994_99406

theorem least_value_quadratic (a : ℝ) :
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → a ≤ x) ↔ a = 5 := by sorry

end least_value_quadratic_l994_99406


namespace pokemon_cards_remaining_l994_99488

theorem pokemon_cards_remaining (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 13 → given_away = 9 → remaining = initial - given_away → remaining = 4 := by
  sorry

end pokemon_cards_remaining_l994_99488


namespace factor_difference_of_squares_l994_99476

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_difference_of_squares_l994_99476


namespace lines_per_page_l994_99434

theorem lines_per_page (total_words : ℕ) (words_per_line : ℕ) (pages_filled : ℚ) (words_left : ℕ) : 
  total_words = 400 →
  words_per_line = 10 →
  pages_filled = 3/2 →
  words_left = 100 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := by
sorry

end lines_per_page_l994_99434


namespace linear_equation_solution_comparison_l994_99448

theorem linear_equation_solution_comparison
  (c c' d d' : ℝ)
  (hc_pos : c > 0)
  (hc'_pos : c' > 0)
  (hc_gt_c' : c > c') :
  ((-d) / c < (-d') / c') ↔ (c * d' < c' * d) := by
sorry

end linear_equation_solution_comparison_l994_99448


namespace job_selection_probability_l994_99409

theorem job_selection_probability 
  (jamie_prob : ℚ) 
  (tom_prob : ℚ) 
  (h1 : jamie_prob = 2/3) 
  (h2 : tom_prob = 5/7) : 
  jamie_prob * tom_prob = 10/21 := by
sorry

end job_selection_probability_l994_99409


namespace tan_alpha_two_implies_expression_value_l994_99420

theorem tan_alpha_two_implies_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16 / 5 := by
  sorry

end tan_alpha_two_implies_expression_value_l994_99420


namespace solve_system_l994_99444

-- Define the system of equations and the condition
def system_equations (x y m : ℝ) : Prop :=
  (4 * x + 2 * y = 3 * m) ∧ (3 * x + y = m + 2)

def opposite_sign (x y : ℝ) : Prop :=
  y = -x

-- Theorem statement
theorem solve_system :
  ∀ x y m : ℝ, system_equations x y m → opposite_sign x y → m = 1 :=
by
  sorry

end solve_system_l994_99444


namespace mary_peaches_cost_l994_99494

/-- The amount Mary paid for berries in dollars -/
def berries_cost : ℚ := 7.19

/-- The total amount Mary paid with in dollars -/
def total_paid : ℚ := 20

/-- The amount Mary received as change in dollars -/
def change_received : ℚ := 5.98

/-- The amount Mary paid for peaches in dollars -/
def peaches_cost : ℚ := total_paid - change_received - berries_cost

theorem mary_peaches_cost : peaches_cost = 6.83 := by sorry

end mary_peaches_cost_l994_99494


namespace lynne_book_cost_l994_99431

/-- Proves that the cost of each book is $7 given the conditions of Lynne's purchase -/
theorem lynne_book_cost (num_books : ℕ) (num_magazines : ℕ) (magazine_cost : ℚ) (total_spent : ℚ) :
  num_books = 9 →
  num_magazines = 3 →
  magazine_cost = 4 →
  total_spent = 75 →
  (num_books * (total_spent - num_magazines * magazine_cost) / num_books : ℚ) = 7 := by
  sorry

end lynne_book_cost_l994_99431


namespace acute_angle_specific_circles_l994_99483

/-- The acute angle formed by two lines intersecting three concentric circles -/
def acute_angle_concentric_circles (r1 r2 r3 : ℝ) (shaded_ratio : ℝ) : ℝ :=
  sorry

/-- The theorem stating the acute angle for the given problem -/
theorem acute_angle_specific_circles :
  acute_angle_concentric_circles 5 3 1 (10/17) = 107/459 := by
  sorry

end acute_angle_specific_circles_l994_99483


namespace equal_cupcake_distribution_l994_99452

theorem equal_cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end equal_cupcake_distribution_l994_99452


namespace e_is_random_error_l994_99439

/-- Linear regression model -/
structure LinearRegressionModel where
  x : ℝ
  y : ℝ
  a : ℝ
  b : ℝ
  e : ℝ
  model_equation : y = b * x + a + e

/-- Definition of random error in linear regression -/
def is_random_error (model : LinearRegressionModel) : Prop :=
  ∃ (error_term : ℝ), 
    error_term = model.e ∧ 
    model.y = model.b * model.x + model.a + error_term

/-- Theorem: In the linear regression model, e is the random error -/
theorem e_is_random_error (model : LinearRegressionModel) : 
  is_random_error model :=
sorry

end e_is_random_error_l994_99439


namespace negation_equivalence_l994_99469

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
by sorry

end negation_equivalence_l994_99469


namespace divisibility_condition_l994_99451

theorem divisibility_condition (a : ℕ) : 
  (a^2 + a + 1) ∣ (a^7 + 3*a^6 + 3*a^5 + 3*a^4 + a^3 + a^2 + 3) ↔ a = 0 ∨ a = 1 := by
  sorry

end divisibility_condition_l994_99451


namespace crayons_in_drawer_l994_99414

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_in_drawer : total_crayons = 12 := by
  sorry

end crayons_in_drawer_l994_99414


namespace mary_shirts_left_l994_99485

/-- Calculates the number of shirts Mary has left after giving away fractions of each color --/
def shirts_left (blue brown red yellow green : ℕ) : ℕ :=
  let blue_left := blue - (4 * blue / 5)
  let brown_left := brown - (5 * brown / 6)
  let red_left := red - (2 * red / 3)
  let yellow_left := yellow - (3 * yellow / 4)
  let green_left := green - (green / 3)
  blue_left + brown_left + red_left + yellow_left + green_left

/-- The theorem stating that Mary has 45 shirts left --/
theorem mary_shirts_left :
  shirts_left 35 48 27 36 18 = 45 := by sorry

end mary_shirts_left_l994_99485


namespace hyperbola_imaginary_axis_length_l994_99429

/-- A hyperbola with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  e_def : e = (a^2 + b^2).sqrt / a

theorem hyperbola_imaginary_axis_length 
  (h : Hyperbola) 
  (dist_foci : ∃ (p : ℝ × ℝ), p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1 ∧ 
    ∃ (f₁ f₂ : ℝ × ℝ), (p.1 - f₁.1)^2 + (p.2 - f₁.2)^2 = 100 ∧ 
                        (p.1 - f₂.1)^2 + (p.2 - f₂.2)^2 = 16) 
  (h_e : h.e = 2) : 
  2 * h.b = 6 * Real.sqrt 3 := by
  sorry

end hyperbola_imaginary_axis_length_l994_99429


namespace lowest_number_of_students_smallest_multiple_is_120_l994_99446

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 15 ∣ n ∧ 24 ∣ n → n ≥ 120 := by
  sorry

theorem smallest_multiple_is_120 : ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 24 ∣ n ∧ n = 120 := by
  sorry

end lowest_number_of_students_smallest_multiple_is_120_l994_99446


namespace rhombus_diagonal_l994_99433

/-- Given a rhombus with area 80 cm² and one diagonal 16 cm, prove the other diagonal is 10 cm -/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 80 → d1 = 16 → area = (d1 * d2) / 2 → d2 = 10 := by
  sorry

end rhombus_diagonal_l994_99433
