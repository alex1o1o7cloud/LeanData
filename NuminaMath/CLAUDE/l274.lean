import Mathlib

namespace green_blue_difference_l274_27403

/-- Represents the number of disks of each color -/
structure DiskCounts where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  purple : ℕ

/-- The ratio of disks for each color -/
def diskRatio : DiskCounts := {
  blue := 3,
  yellow := 7,
  green := 8,
  red := 4,
  purple := 5
}

/-- The total number of disks in the bag -/
def totalDisks : ℕ := 360

/-- Calculates the total ratio parts -/
def totalRatioParts (ratio : DiskCounts) : ℕ :=
  ratio.blue + ratio.yellow + ratio.green + ratio.red + ratio.purple

/-- Calculates the number of disks for each ratio part -/
def disksPerPart (total : ℕ) (ratioParts : ℕ) : ℕ :=
  total / ratioParts

/-- Calculates the actual disk counts based on the ratio and total disks -/
def actualDiskCounts (ratio : DiskCounts) (total : ℕ) : DiskCounts :=
  let parts := totalRatioParts ratio
  let perPart := disksPerPart total parts
  {
    blue := ratio.blue * perPart,
    yellow := ratio.yellow * perPart,
    green := ratio.green * perPart,
    red := ratio.red * perPart,
    purple := ratio.purple * perPart
  }

theorem green_blue_difference :
  let counts := actualDiskCounts diskRatio totalDisks
  counts.green - counts.blue = 65 := by sorry

end green_blue_difference_l274_27403


namespace specific_student_not_front_l274_27477

/-- The number of ways to arrange n students in a line. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a line with a specific student at the front. -/
def arrangementsWithSpecificFront (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of students. -/
def numStudents : ℕ := 5

theorem specific_student_not_front :
  arrangements numStudents - arrangementsWithSpecificFront numStudents = 96 :=
sorry

end specific_student_not_front_l274_27477


namespace min_area_quadrilateral_l274_27456

/-- Given a rectangle ABCD with points A₁, B₁, C₁, D₁ on the rays AB, BC, CD, DA respectively,
    such that AA₁/AB = BB₁/BC = CC₁/CD = DD₁/DA = k > 0,
    prove that the area of quadrilateral A₁B₁C₁D₁ is minimized when k = 1/2 -/
theorem min_area_quadrilateral (a b : ℝ) (k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  let area := a * b * (1 - k + k^2)
  (∀ k' > 0, area ≤ a * b * (1 - k' + k'^2)) ↔ k = 1/2 := by
  sorry


end min_area_quadrilateral_l274_27456


namespace probability_genuine_after_defective_l274_27432

theorem probability_genuine_after_defective :
  ∀ (total genuine defective : ℕ),
    total = genuine + defective →
    total = 7 →
    genuine = 4 →
    defective = 3 →
    (genuine : ℚ) / (total - 1 : ℚ) = 2 / 3 := by
  sorry

end probability_genuine_after_defective_l274_27432


namespace fraction_equality_l274_27427

theorem fraction_equality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
  sorry

end fraction_equality_l274_27427


namespace quarters_percentage_l274_27486

theorem quarters_percentage (num_dimes : ℕ) (num_quarters : ℕ) : num_dimes = 70 → num_quarters = 30 → 
  (num_quarters * 25 : ℚ) / ((num_dimes * 10 + num_quarters * 25) : ℚ) * 100 = 51724 / 1000 := by
  sorry

end quarters_percentage_l274_27486


namespace solution_set_correct_l274_27424

/-- The solution set of the inequality -x^2 + 2x > 0 -/
def SolutionSet : Set ℝ := {x | 0 < x ∧ x < 2}

/-- Theorem stating that SolutionSet is the correct solution to the inequality -/
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ SolutionSet ↔ -x^2 + 2*x > 0 := by
  sorry

end solution_set_correct_l274_27424


namespace karl_net_income_l274_27452

/-- Represents the sale of boots and subsequent transactions -/
structure BootSale where
  initial_price : ℚ
  actual_sale_price : ℚ
  reduced_price : ℚ
  refund_amount : ℚ
  candy_expense : ℚ
  actual_refund : ℚ

/-- Calculates the net income from a boot sale -/
def net_income (sale : BootSale) : ℚ :=
  sale.actual_sale_price * 2 - sale.refund_amount

/-- Theorem stating that Karl's net income is 20 talers -/
theorem karl_net_income (sale : BootSale) 
  (h1 : sale.initial_price = 25)
  (h2 : sale.actual_sale_price = 12.5)
  (h3 : sale.reduced_price = 10)
  (h4 : sale.refund_amount = 5)
  (h5 : sale.candy_expense = 3)
  (h6 : sale.actual_refund = 1) :
  net_income sale = 20 := by
  sorry


end karl_net_income_l274_27452


namespace least_k_value_l274_27442

theorem least_k_value (a b c d : ℝ) : 
  ∃ k : ℝ, k = 4 ∧ 
  (∀ a b c d : ℝ, 
    Real.sqrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) + 
    Real.sqrt ((b^2 + 1) * (c^2 + 1) * (d^2 + 1)) + 
    Real.sqrt ((c^2 + 1) * (d^2 + 1) * (a^2 + 1)) + 
    Real.sqrt ((d^2 + 1) * (a^2 + 1) * (b^2 + 1)) ≥ 
    2 * (a*b + b*c + c*d + d*a + a*c + b*d) - k) ∧
  (∀ k' : ℝ, k' < k → 
    ∃ a b c d : ℝ, 
      Real.sqrt ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) + 
      Real.sqrt ((b^2 + 1) * (c^2 + 1) * (d^2 + 1)) + 
      Real.sqrt ((c^2 + 1) * (d^2 + 1) * (a^2 + 1)) + 
      Real.sqrt ((d^2 + 1) * (a^2 + 1) * (b^2 + 1)) < 
      2 * (a*b + b*c + c*d + d*a + a*c + b*d) - k') :=
by sorry

end least_k_value_l274_27442


namespace greatest_three_digit_multiple_of_17_l274_27466

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 969 := by sorry

end greatest_three_digit_multiple_of_17_l274_27466


namespace arithmetic_sequence_sum_l274_27425

/-- Given an arithmetic sequence a with S₃ = 6, prove that 5a₁ + a₇ = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  (3 * a 1 + 3 * d = 6) →       -- S₃ = 6 condition
  5 * a 1 + a 7 = 12 := by
sorry

end arithmetic_sequence_sum_l274_27425


namespace gasoline_reduction_l274_27437

theorem gasoline_reduction (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 1.20 * P
  let new_total_cost := 1.08 * (P * Q)
  let new_quantity := new_total_cost / new_price
  (Q - new_quantity) / Q = 0.10 := by
  sorry

end gasoline_reduction_l274_27437


namespace problem_statement_l274_27401

theorem problem_statement :
  (¬ (∃ x : ℝ, x^2 - x + 1 < 0)) ∧
  (¬ (∀ x : ℝ, x^2 - 4 ≠ 0)) := by
  sorry

end problem_statement_l274_27401


namespace circle_area_8m_diameter_circle_area_8m_diameter_proof_l274_27457

/-- The area of a circle with diameter 8 meters, in square centimeters -/
theorem circle_area_8m_diameter (π : ℝ) : ℝ :=
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area_sq_meters : ℝ := π * radius ^ 2
  let sq_cm_per_sq_meter : ℝ := 10000
  160000 * π

/-- Proof that the area of a circle with diameter 8 meters is 160000π square centimeters -/
theorem circle_area_8m_diameter_proof (π : ℝ) :
  circle_area_8m_diameter π = 160000 * π :=
by sorry

end circle_area_8m_diameter_circle_area_8m_diameter_proof_l274_27457


namespace two_cakes_left_l274_27480

/-- The number of cakes left at a restaurant -/
def cakes_left (baked_today baked_yesterday sold : ℕ) : ℕ :=
  baked_today + baked_yesterday - sold

/-- Theorem: Given the conditions, prove that 2 cakes are left -/
theorem two_cakes_left : cakes_left 5 3 6 = 2 := by
  sorry

end two_cakes_left_l274_27480


namespace cost_price_percentage_l274_27420

theorem cost_price_percentage (cost_price selling_price : ℝ) 
  (h : selling_price = 4 * cost_price) : 
  cost_price / selling_price = 1 / 4 := by
  sorry

#check cost_price_percentage

end cost_price_percentage_l274_27420


namespace coffee_shop_spending_coffee_shop_spending_proof_l274_27410

theorem coffee_shop_spending : ℝ → ℝ → Prop :=
  fun b d =>
    (d = 0.6 * b) →  -- David spent 40 cents less for each dollar Ben spent
    (b = d + 14) →   -- Ben paid $14 more than David
    (b + d = 56)     -- Their total spending

-- The proof is omitted
theorem coffee_shop_spending_proof : ∃ b d : ℝ, coffee_shop_spending b d := by sorry

end coffee_shop_spending_coffee_shop_spending_proof_l274_27410


namespace dans_remaining_money_l274_27406

/-- Calculates the remaining money after purchases. -/
def remaining_money (initial : ℕ) (candy_price : ℕ) (chocolate_price : ℕ) : ℕ :=
  initial - (candy_price + chocolate_price)

/-- Proves that Dan has $2 left after his purchases. -/
theorem dans_remaining_money :
  remaining_money 7 2 3 = 2 := by
  sorry

end dans_remaining_money_l274_27406


namespace arithmetic_calculations_l274_27415

theorem arithmetic_calculations :
  (0.25 + (-9) + (-1/4) - 11 = -20) ∧
  (-15 + 5 + 1/3 * (-6) = -12) ∧
  ((-3/8 - 1/6 + 3/4) * 24 = 5) := by
sorry

end arithmetic_calculations_l274_27415


namespace angle_bisector_vector_l274_27465

/-- Given points A and B in a Cartesian coordinate system, 
    and a point C on the angle bisector of ∠AOB with |OC| = 2, 
    prove that OC has specific coordinates. -/
theorem angle_bisector_vector (A B C : ℝ × ℝ) : 
  A = (0, 1) →
  B = (-3, 4) →
  (C.1 * A.2 = C.2 * A.1 ∧ C.1 * B.2 = C.2 * B.1) → -- C is on angle bisector
  C.1^2 + C.2^2 = 4 → -- |OC| = 2
  C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) := by
  sorry

end angle_bisector_vector_l274_27465


namespace expected_value_is_1866_l274_27405

/-- Represents the available keys on the calculator -/
inductive Key
| One
| Two
| Three
| Plus
| Minus

/-- A sequence of 5 keystrokes -/
def Sequence := Vector Key 5

/-- Evaluates a sequence of keystrokes according to the problem rules -/
def evaluate : Sequence → ℤ := sorry

/-- The probability of pressing any specific key -/
def keyProbability : ℚ := 1 / 5

/-- The expected value of the result after evaluating a random sequence -/
def expectedValue : ℚ := sorry

/-- Theorem stating that the expected value is 1866 -/
theorem expected_value_is_1866 : expectedValue = 1866 := by sorry

end expected_value_is_1866_l274_27405


namespace field_length_difference_l274_27438

/-- 
Given a rectangular field with length 24 meters and width 13.5 meters,
prove that the difference between twice the width and the length is 3 meters.
-/
theorem field_length_difference (length width : ℝ) 
  (h1 : length = 24)
  (h2 : width = 13.5) :
  2 * width - length = 3 := by
  sorry

end field_length_difference_l274_27438


namespace hyperbola_to_ellipse_l274_27484

/-- Given a hyperbola with equation y²/12 - x²/4 = 1, prove that the equation of the ellipse
    that has the foci of the hyperbola as its vertices and the vertices of the hyperbola as its foci
    is y²/16 + x²/4 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (y^2 / 12 - x^2 / 4 = 1) →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b) ∧
    (y^2 / a^2 + x^2 / b^2 = 1) ∧
    a = 4 ∧ b^2 = 4 :=
by sorry

end hyperbola_to_ellipse_l274_27484


namespace x_fourth_plus_inverse_x_fourth_l274_27453

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 := by
  sorry

end x_fourth_plus_inverse_x_fourth_l274_27453


namespace wyatt_envelopes_l274_27469

/-- The number of blue envelopes Wyatt has -/
def blue_envelopes : ℕ := 10

/-- The difference between blue and yellow envelopes -/
def envelope_difference : ℕ := 4

/-- The total number of envelopes Wyatt has -/
def total_envelopes : ℕ := blue_envelopes + (blue_envelopes - envelope_difference)

/-- Theorem stating the total number of envelopes Wyatt has -/
theorem wyatt_envelopes : total_envelopes = 16 := by sorry

end wyatt_envelopes_l274_27469


namespace new_average_age_l274_27490

theorem new_average_age
  (initial_students : ℕ)
  (initial_average : ℚ)
  (new_student_age : ℕ)
  (h1 : initial_students = 8)
  (h2 : initial_average = 15)
  (h3 : new_student_age = 17) :
  let total_age : ℚ := initial_students * initial_average + new_student_age
  let new_total_students : ℕ := initial_students + 1
  total_age / new_total_students = 137 / 9 :=
by sorry

end new_average_age_l274_27490


namespace perpendicular_condition_acute_angle_condition_l274_27412

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

/-- Dot product of two 2D vectors -/
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

/-- The angle between two vectors is acute if their dot product is positive -/
def acute_angle (u v : Fin 2 → ℝ) : Prop := dot_product u v > 0

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (u v : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

theorem perpendicular_condition (x : ℝ) : 
  perpendicular (λ i => a i + 2 * b x i) (λ i => 2 * a i - b x i) ↔ x = -2 ∨ x = 7/2 := by
  sorry

theorem acute_angle_condition (x : ℝ) :
  acute_angle a (b x) ∧ ¬ parallel a (b x) ↔ x > -2 ∧ x ≠ 1/2 := by
  sorry

end perpendicular_condition_acute_angle_condition_l274_27412


namespace rectangle_diagonal_l274_27402

theorem rectangle_diagonal (perimeter : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) :
  perimeter = 72 →
  ratio_length = 3 →
  ratio_width = 2 →
  let length := (perimeter / 2) * (ratio_length / (ratio_length + ratio_width))
  let width := (perimeter / 2) * (ratio_width / (ratio_length + ratio_width))
  (length ^ 2 + width ^ 2) = 673.92 := by
  sorry

end rectangle_diagonal_l274_27402


namespace students_liking_both_l274_27421

theorem students_liking_both (total : ℕ) (fries : ℕ) (burgers : ℕ) (neither : ℕ) :
  total = 25 →
  fries = 15 →
  burgers = 10 →
  neither = 6 →
  ∃ (both : ℕ), both = 12 ∧ total = fries + burgers - both + neither :=
by sorry

end students_liking_both_l274_27421


namespace circle_area_with_complex_conditions_l274_27448

theorem circle_area_with_complex_conditions (z₁ z₂ : ℂ) 
  (h1 : z₁^2 - 4*z₁*z₂ + 4*z₂^2 = 0)
  (h2 : Complex.abs z₂ = 2) :
  Real.pi * (Complex.abs z₁ / 2)^2 = 4 * Real.pi := by
  sorry

end circle_area_with_complex_conditions_l274_27448


namespace certain_amount_problem_l274_27426

theorem certain_amount_problem (first_number : ℕ) (certain_amount : ℕ) : 
  first_number = 5 →
  first_number + (11 + certain_amount) = 19 →
  certain_amount = 3 := by
sorry

end certain_amount_problem_l274_27426


namespace point_inside_circle_l274_27499

-- Define the ellipse parameters
variable (a b c : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the conditions
theorem point_inside_circle
  (h_positive : a > 0 ∧ b > 0)
  (h_eccentricity : c / a = 1 / 2)
  (h_ellipse : b^2 = a^2 - c^2)
  (h_roots : x₁ + x₂ = -b / a ∧ x₁ * x₂ = -c / a) :
  x₁^2 + x₂^2 < 2 := by
  sorry

end point_inside_circle_l274_27499


namespace mothers_salary_l274_27409

theorem mothers_salary (mother_salary : ℝ) : 
  let father_salary := 1.3 * mother_salary
  let combined_salary := mother_salary + father_salary
  let method1_savings := (combined_salary / 10) * 6
  let method2_savings := (combined_salary / 2) * (1 + 0.03 * 10)
  method1_savings = method2_savings - 2875 →
  mother_salary = 25000 := by
sorry

end mothers_salary_l274_27409


namespace jerry_remaining_money_l274_27464

/-- Calculates the remaining money after grocery shopping --/
def remaining_money (initial_amount : ℝ) (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
  (pasta_price : ℝ) (pasta_quantity : ℝ) (sauce_price : ℝ) (sauce_quantity : ℝ) : ℝ :=
  initial_amount - (mustard_oil_price * mustard_oil_quantity + pasta_price * pasta_quantity + sauce_price * sauce_quantity)

/-- Theorem stating that Jerry will have $7 after shopping --/
theorem jerry_remaining_money :
  remaining_money 50 13 2 4 3 5 1 = 7 := by
  sorry

end jerry_remaining_money_l274_27464


namespace fraction_addition_l274_27433

theorem fraction_addition (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) :
  (a + b) / b = 7 / 4 := by
  sorry

end fraction_addition_l274_27433


namespace circle_radius_tangent_to_square_extensions_l274_27417

/-- The radius of a circle tangent to the extensions of two sides of a square,
    where two tangents from the opposite corner form a specific angle. -/
theorem circle_radius_tangent_to_square_extensions 
  (side_length : ℝ) 
  (tangent_angle : ℝ) 
  (sin_half_angle : ℝ) :
  side_length = 6 + 2 * Real.sqrt 5 →
  tangent_angle = 36 →
  sin_half_angle = (Real.sqrt 5 - 1) / 4 →
  ∃ (radius : ℝ), 
    radius = 2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) ∧
    radius = side_length * Real.sqrt 2 / 
      ((4 / (Real.sqrt 5 - 1)) - Real.sqrt 2) :=
by sorry

end circle_radius_tangent_to_square_extensions_l274_27417


namespace fruit_basket_count_l274_27498

/-- The number of ways to choose k items from n identical items -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := (n + k - 1).choose k

/-- The number of possible fruit baskets given the number of bananas and pears -/
def fruit_baskets (bananas : ℕ) (pears : ℕ) : ℕ :=
  (choose_with_repetition (bananas + 1) 1) * (choose_with_repetition (pears + 1) 1) - 1

theorem fruit_basket_count :
  fruit_baskets 6 9 = 69 := by
  sorry

end fruit_basket_count_l274_27498


namespace mary_potatoes_l274_27418

/-- The number of potatoes Mary initially had -/
def initial_potatoes : ℕ := 8

/-- The number of potatoes eaten by rabbits -/
def eaten_potatoes : ℕ := 3

/-- The number of potatoes Mary has now -/
def remaining_potatoes : ℕ := initial_potatoes - eaten_potatoes

theorem mary_potatoes : remaining_potatoes = 5 := by
  sorry

end mary_potatoes_l274_27418


namespace arithmetic_sequence_sum_l274_27444

/-- Given an arithmetic sequence where:
    - n is a positive integer
    - The sum of the first n terms is 48
    - The sum of the first 2n terms is 60
    This theorem states that the sum of the first 3n terms is 36 -/
theorem arithmetic_sequence_sum (n : ℕ+) 
  (sum_n : ℕ) (sum_2n : ℕ) (h1 : sum_n = 48) (h2 : sum_2n = 60) :
  ∃ (sum_3n : ℕ), sum_3n = 36 := by
  sorry

end arithmetic_sequence_sum_l274_27444


namespace garment_pricing_problem_l274_27470

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 400

-- Define the profit function without donation
def profit_function (x : ℝ) : ℝ := (x - 60) * (sales_function x)

-- Define the profit function with donation
def profit_function_with_donation (x : ℝ) : ℝ := (x - 70) * (sales_function x)

theorem garment_pricing_problem :
  -- The linear function fits the given data points
  (sales_function 80 = 240) ∧
  (sales_function 90 = 220) ∧
  (sales_function 100 = 200) ∧
  (sales_function 110 = 180) ∧
  -- The smaller solution to the profit equation is 100
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    profit_function x₁ = 8000 ∧ 
    profit_function x₂ = 8000 ∧ 
    x₁ = 100) ∧
  -- The profit function with donation has a maximum at 135
  (∃ max_profit : ℝ, 
    profit_function_with_donation 135 = max_profit ∧
    ∀ x : ℝ, profit_function_with_donation x ≤ max_profit) :=
by sorry

end garment_pricing_problem_l274_27470


namespace prime_average_count_l274_27455

theorem prime_average_count : 
  ∃ (p₁ p₂ p₃ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ > 20 ∧ p₂ > 20 ∧ p₃ > 20 ∧
    (p₁ + p₂ + p₃) / 3 = 83 / 3 ∧
    ∀ (q₁ q₂ q₃ q₄ : ℕ), 
      Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
      q₁ > 20 ∧ q₂ > 20 ∧ q₃ > 20 ∧ q₄ > 20 →
      (q₁ + q₂ + q₃ + q₄) / 4 ≠ 83 / 3 :=
by sorry

end prime_average_count_l274_27455


namespace complex_power_sum_l274_27429

theorem complex_power_sum (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := by
  sorry

end complex_power_sum_l274_27429


namespace original_bananas_count_l274_27483

/-- The number of bananas originally in the jar. -/
def original_bananas : ℕ := sorry

/-- The number of bananas removed from the jar. -/
def removed_bananas : ℕ := 5

/-- The number of bananas left in the jar after removal. -/
def remaining_bananas : ℕ := 41

/-- Theorem: The original number of bananas is equal to 46. -/
theorem original_bananas_count : original_bananas = 46 := by
  sorry

end original_bananas_count_l274_27483


namespace houses_with_neither_amenity_l274_27434

/-- Given a development with houses, some of which have a two-car garage and/or an in-the-ground swimming pool, 
    this theorem proves the number of houses with neither amenity. -/
theorem houses_with_neither_amenity 
  (total : ℕ) 
  (garage : ℕ) 
  (pool : ℕ) 
  (both : ℕ) 
  (h1 : total = 90) 
  (h2 : garage = 50) 
  (h3 : pool = 40) 
  (h4 : both = 35) : 
  total - (garage + pool - both) = 35 := by
  sorry


end houses_with_neither_amenity_l274_27434


namespace weekly_egg_supply_l274_27461

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of dozens of eggs supplied to the first store daily -/
def store1_supply : ℕ := 5

/-- The number of eggs supplied to the second store daily -/
def store2_supply : ℕ := 30

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem weekly_egg_supply : 
  (store1_supply * dozen + store2_supply) * days_in_week = 630 := by
  sorry

end weekly_egg_supply_l274_27461


namespace right_triangle_equations_l274_27447

/-- A right-angled triangle ABC with specified coordinates -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : B.1 = 1 ∧ B.2 = Real.sqrt 3
  A_on_x_axis : A = (-2, 0)
  C_on_x_axis : C.2 = 0

/-- The equation of line BC in the form ax + by + c = 0 -/
def line_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- The equation of line OB (median to hypotenuse) in the form y = kx -/
def median_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

theorem right_triangle_equations (t : RightTriangle) :
  (∃ (a b c : ℝ), a = Real.sqrt 3 ∧ b = 1 ∧ c = -2 * Real.sqrt 3 ∧
    ∀ (x y : ℝ), line_equation a b c x y ↔ (x, y) ∈ ({t.B, t.C} : Set (ℝ × ℝ))) ∧
  (∃ (k : ℝ), k = Real.sqrt 3 ∧
    ∀ (x y : ℝ), median_equation k x y ↔ (x, y) ∈ ({(0, 0), t.B} : Set (ℝ × ℝ))) :=
sorry

end right_triangle_equations_l274_27447


namespace coworker_repair_ratio_l274_27450

/-- The ratio of phones a coworker fixes to the total number of damaged phones -/
theorem coworker_repair_ratio : 
  ∀ (initial_phones repaired_phones new_phones phones_per_person : ℕ),
    initial_phones = 15 →
    repaired_phones = 3 →
    new_phones = 6 →
    phones_per_person = 9 →
    (phones_per_person : ℚ) / ((initial_phones - repaired_phones + new_phones) : ℚ) = 1 / 2 := by
  sorry

end coworker_repair_ratio_l274_27450


namespace eliana_refills_l274_27467

theorem eliana_refills (total_spent : ℕ) (cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) :
  total_spent / cost_per_refill = 3 := by
  sorry

end eliana_refills_l274_27467


namespace gcd_143_100_l274_27474

theorem gcd_143_100 : Nat.gcd 143 100 = 1 := by
  sorry

end gcd_143_100_l274_27474


namespace meet_on_same_side_time_l274_27493

/-- The time when two points moving on a square meet on the same side for the first time -/
def time_to_meet_on_same_side (side_length : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  35

/-- Theorem stating that the time to meet on the same side is 35 seconds under given conditions -/
theorem meet_on_same_side_time :
  time_to_meet_on_same_side 100 5 10 = 35 := by
  sorry

end meet_on_same_side_time_l274_27493


namespace probability_at_least_one_grade_12_l274_27422

def total_sample_size : ℕ := 6
def grade_10_size : ℕ := 54
def grade_11_size : ℕ := 18
def grade_12_size : ℕ := 36

def grade_10_selected : ℕ := 3
def grade_11_selected : ℕ := 1
def grade_12_selected : ℕ := 2

def selected_size : ℕ := 3

theorem probability_at_least_one_grade_12 :
  let total_combinations := Nat.choose total_sample_size selected_size
  let favorable_combinations := total_combinations - Nat.choose (total_sample_size - grade_12_selected) selected_size
  (favorable_combinations : ℚ) / total_combinations = 4 / 5 := by
  sorry

end probability_at_least_one_grade_12_l274_27422


namespace expression_simplification_l274_27475

theorem expression_simplification (x y : ℝ) :
  x * (4 * x^3 - 3 * x^2 + 2 * y) - 6 * (x^3 - 3 * x^2 + 2 * x + 8) =
  4 * x^4 - 9 * x^3 + 18 * x^2 + 2 * x * y - 12 * x - 48 := by
  sorry

end expression_simplification_l274_27475


namespace percentage_calculation_l274_27468

theorem percentage_calculation : 
  let initial_value : ℝ := 180
  let percentage : ℝ := 1/3
  let divisor : ℝ := 6
  (initial_value * (percentage / 100)) / divisor = 0.1 := by sorry

end percentage_calculation_l274_27468


namespace sequence_problem_l274_27441

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) 
    (h_geo : geometric_sequence a)
    (h_arith : arithmetic_sequence b)
    (h_a : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
    (h_b : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end sequence_problem_l274_27441


namespace polynomial_factorization_l274_27497

theorem polynomial_factorization (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end polynomial_factorization_l274_27497


namespace abc_inequality_l274_27413

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 - a*b = c^2) : (a - c) * (b - c) ≤ 0 := by
  sorry

end abc_inequality_l274_27413


namespace polygon_sides_count_l274_27489

theorem polygon_sides_count : ∃ n : ℕ, 
  n > 2 ∧ 
  (n * (n - 3) / 2 : ℚ) = 2 * n ∧ 
  n = 7 := by
  sorry

end polygon_sides_count_l274_27489


namespace cantaloupes_total_l274_27491

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_total : total_cantaloupes = 82 := by
  sorry

end cantaloupes_total_l274_27491


namespace inequality_range_l274_27428

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by
  sorry

end inequality_range_l274_27428


namespace candidate_vote_percentage_l274_27463

/-- Calculates the percentage of valid votes a candidate received in an election. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_vote_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_vote_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 380800) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_vote_percentage) * total_votes) = 80 / 100 := by
sorry


end candidate_vote_percentage_l274_27463


namespace pins_purchased_proof_l274_27485

/-- Calculates the number of pins purchased given the original price, discount percentage, and total amount spent. -/
def calculate_pins_purchased (original_price : ℚ) (discount_percent : ℚ) (total_spent : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_percent / 100)
  total_spent / discounted_price

/-- Proves that purchasing pins at a 15% discount from $20 each, spending $170 results in 10 pins. -/
theorem pins_purchased_proof :
  calculate_pins_purchased 20 15 170 = 10 := by
  sorry

end pins_purchased_proof_l274_27485


namespace perpendicular_vectors_l274_27495

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (t : ℝ) :
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) → t = -5 :=
by sorry

end perpendicular_vectors_l274_27495


namespace cardinality_difference_constant_l274_27400

/-- Given a finite set of positive integers, S_n is the set of all sums of exactly n elements from the set -/
def S_n (A : Finset Nat) (n : Nat) : Finset Nat :=
  sorry

/-- The main theorem stating the existence of N and k -/
theorem cardinality_difference_constant (A : Finset Nat) :
  ∃ (N k : Nat), ∀ n ≥ N, (S_n A (n + 1)).card = (S_n A n).card + k :=
sorry

end cardinality_difference_constant_l274_27400


namespace college_ratio_theorem_l274_27487

/-- Represents the ratio of boys to girls in a college -/
structure CollegeRatio where
  boys : ℕ
  girls : ℕ

/-- Given the total number of students and the number of girls, calculate the ratio of boys to girls -/
def calculateRatio (totalStudents : ℕ) (numGirls : ℕ) : CollegeRatio :=
  { boys := totalStudents - numGirls,
    girls := numGirls }

/-- Theorem stating that for a college with 240 total students and 140 girls, the ratio of boys to girls is 5:7 -/
theorem college_ratio_theorem :
  let ratio := calculateRatio 240 140
  ratio.boys = 5 ∧ ratio.girls = 7 := by
  sorry


end college_ratio_theorem_l274_27487


namespace negative_deviation_notation_l274_27458

/-- Represents a height deviation from the average. -/
structure HeightDeviation where
  value : ℝ

/-- The average height of the team. -/
def averageHeight : ℝ := 175

/-- Notation for height deviations. -/
def denoteDeviation (d : HeightDeviation) : ℝ := d.value

/-- Axiom: Positive deviation is denoted by a positive number. -/
axiom positive_deviation_notation (d : HeightDeviation) :
  d.value > 0 → denoteDeviation d > 0

/-- Theorem: Negative deviation should be denoted by a negative number. -/
theorem negative_deviation_notation (d : HeightDeviation) :
  d.value < 0 → denoteDeviation d < 0 :=
sorry

end negative_deviation_notation_l274_27458


namespace largest_triangle_perimeter_l274_27472

theorem largest_triangle_perimeter : ∀ x : ℤ,
  (8 : ℝ) + 11 > (x : ℝ) ∧ 
  (8 : ℝ) + (x : ℝ) > 11 ∧ 
  (11 : ℝ) + (x : ℝ) > 8 →
  (8 : ℝ) + 11 + (x : ℝ) ≤ 37 :=
by
  sorry

end largest_triangle_perimeter_l274_27472


namespace population_trend_l274_27494

theorem population_trend (P k : ℝ) (h1 : P > 0) (h2 : -1 < k) (h3 : k < 0) :
  ∀ n : ℕ, (P * (1 + k)^(n + 1)) < (P * (1 + k)^n) := by
  sorry

end population_trend_l274_27494


namespace one_hundred_ten_billion_scientific_notation_l274_27492

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem one_hundred_ten_billion_scientific_notation :
  toScientificNotation 110000000000 = ScientificNotation.mk 1.1 11 (by norm_num) :=
sorry

end one_hundred_ten_billion_scientific_notation_l274_27492


namespace square_side_length_for_unit_area_l274_27488

theorem square_side_length_for_unit_area (s : ℝ) :
  s > 0 → s * s = 1 → s = 1 := by sorry

end square_side_length_for_unit_area_l274_27488


namespace valid_rental_plans_l274_27407

/-- Represents a bus rental plan --/
structure RentalPlan where
  typeA : Nat  -- Number of Type A buses
  typeB : Nat  -- Number of Type B buses

/-- Checks if a rental plan can accommodate exactly the given number of students --/
def isValidPlan (plan : RentalPlan) (totalStudents : Nat) (typeACapacity : Nat) (typeBCapacity : Nat) : Prop :=
  plan.typeA * typeACapacity + plan.typeB * typeBCapacity = totalStudents

/-- Theorem stating that the three given rental plans are valid for 37 students --/
theorem valid_rental_plans :
  let totalStudents := 37
  let typeACapacity := 8
  let typeBCapacity := 4
  let plan1 : RentalPlan := ⟨2, 6⟩
  let plan2 : RentalPlan := ⟨3, 4⟩
  let plan3 : RentalPlan := ⟨4, 2⟩
  isValidPlan plan1 totalStudents typeACapacity typeBCapacity ∧
  isValidPlan plan2 totalStudents typeACapacity typeBCapacity ∧
  isValidPlan plan3 totalStudents typeACapacity typeBCapacity :=
by sorry


end valid_rental_plans_l274_27407


namespace quadratic_form_ratio_l274_27478

theorem quadratic_form_ratio (x : ℝ) : ∃ b c : ℝ, 
  x^2 + 500*x + 1000 = (x + b)^2 + c ∧ c / b = -246 := by
sorry

end quadratic_form_ratio_l274_27478


namespace towel_packs_l274_27411

theorem towel_packs (towels_per_pack : ℕ) (total_towels : ℕ) (num_packs : ℕ) :
  towels_per_pack = 3 →
  total_towels = 27 →
  num_packs * towels_per_pack = total_towels →
  num_packs = 9 := by
  sorry

end towel_packs_l274_27411


namespace sine_function_vertical_shift_l274_27423

/-- Given a sine function y = a * sin(b * x) + d with positive constants a, b, and d,
    if the maximum value of y is 4 and the minimum value of y is -2, then d = 1. -/
theorem sine_function_vertical_shift 
  (a b d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hmax : ∀ x, a * Real.sin (b * x) + d ≤ 4)
  (hmin : ∀ x, a * Real.sin (b * x) + d ≥ -2)
  (hex_max : ∃ x, a * Real.sin (b * x) + d = 4)
  (hex_min : ∃ x, a * Real.sin (b * x) + d = -2) : 
  d = 1 := by
  sorry

end sine_function_vertical_shift_l274_27423


namespace perimeter_of_problem_pentagon_l274_27496

/-- A pentagon ABCDE with given side lengths and a right angle -/
structure Pentagon :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DE : ℝ)
  (AE : ℝ)
  (right_angle_AED : AE^2 + DE^2 = AB^2 + BC^2 + DE^2)

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ :=
  p.AB + p.BC + p.CD + p.DE + p.AE

/-- The specific pentagon from the problem -/
def problem_pentagon : Pentagon :=
  { AB := 4
  , BC := 2
  , CD := 2
  , DE := 6
  , AE := 6
  , right_angle_AED := by sorry }

/-- Theorem: The perimeter of the problem pentagon is 14 + 6√2 -/
theorem perimeter_of_problem_pentagon :
  perimeter problem_pentagon = 14 + 6 * Real.sqrt 2 := by
  sorry

end perimeter_of_problem_pentagon_l274_27496


namespace card_probability_ratio_l274_27446

/-- Given a box of 60 cards numbered 1 to 12, with 5 cards for each number,
    prove that the ratio of probabilities q/p is 275, where:
    p = probability of drawing 5 cards with the same number
    q = probability of drawing 4 cards with one number and 1 card with a different number -/
theorem card_probability_ratio :
  let total_cards : ℕ := 60
  let num_values : ℕ := 12
  let cards_per_value : ℕ := 5
  let draw_size : ℕ := 5
  let p := (num_values * Nat.choose cards_per_value draw_size) / Nat.choose total_cards draw_size
  let q := (num_values * (num_values - 1) * Nat.choose cards_per_value 4 * Nat.choose cards_per_value 1) / Nat.choose total_cards draw_size
  q / p = 275 := by
  sorry

end card_probability_ratio_l274_27446


namespace x_equals_y_l274_27476

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : y = x := by
  sorry

end x_equals_y_l274_27476


namespace ellipse_hyperbola_product_l274_27451

-- Define the constants for the foci locations
def ellipse_focus : ℝ := 5
def hyperbola_focus : ℝ := 8

-- Define the theorem
theorem ellipse_hyperbola_product (c d : ℝ) : 
  (d^2 - c^2 = ellipse_focus^2) →   -- Condition for ellipse foci
  (c^2 + d^2 = hyperbola_focus^2) → -- Condition for hyperbola foci
  |c * d| = Real.sqrt ((39 * 89) / 4) := by
sorry

end ellipse_hyperbola_product_l274_27451


namespace rectangle_from_right_triangle_l274_27454

theorem rectangle_from_right_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, 
    x + y = c ∧ 
    x * y = a * b / 2 ∧
    x = (c + a - b) / 2 ∧ 
    y = (c - a + b) / 2 := by
  sorry


end rectangle_from_right_triangle_l274_27454


namespace subtraction_problem_l274_27443

theorem subtraction_problem (x : ℤ) : 
  (x - 48 = 22) → (x - 32 = 38) := by
  sorry

end subtraction_problem_l274_27443


namespace odd_prime_fifth_power_difference_l274_27416

theorem odd_prime_fifth_power_difference (p : ℕ) (h_prime : Prime p) (h_odd : Odd p)
  (h_fifth_power_diff : ∃ (a b : ℕ), p = a^5 - b^5) :
  ∃ (n : ℕ), Odd n ∧ (((4 * p + 1) : ℚ) / 5).sqrt = ((n^2 + 1) : ℚ) / 2 := by
  sorry

end odd_prime_fifth_power_difference_l274_27416


namespace tangent_line_to_two_curves_l274_27445

/-- A line y = kx + t is tangent to both curves y = exp x + 2 and y = exp (x + 1) -/
theorem tangent_line_to_two_curves (k t : ℝ) : 
  (∃ x₁ : ℝ, k * x₁ + t = Real.exp x₁ + 2 ∧ k = Real.exp x₁) →
  (∃ x₂ : ℝ, k * x₂ + t = Real.exp (x₂ + 1) ∧ k = Real.exp (x₂ + 1)) →
  t = 4 - 2 * Real.log 2 := by
sorry


end tangent_line_to_two_curves_l274_27445


namespace angle_value_l274_27435

def A (θ : ℝ) : Set ℝ := {1, Real.cos θ}
def B : Set ℝ := {0, 1/2, 1}

theorem angle_value (θ : ℝ) (h1 : A θ ⊆ B) (h2 : 0 < θ ∧ θ < π / 2) : θ = π / 3 := by
  sorry

end angle_value_l274_27435


namespace joes_test_count_l274_27459

/-- Given Joe's test scores, prove the number of initial tests --/
theorem joes_test_count (initial_avg : ℚ) (lowest_score : ℚ) (new_avg : ℚ) :
  initial_avg = 40 →
  lowest_score = 25 →
  new_avg = 45 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg = ((n : ℚ) - 1) * new_avg + lowest_score ∧
    n = 5 := by
  sorry

end joes_test_count_l274_27459


namespace bombardment_percentage_l274_27404

/-- Proves that the percentage of people who died by bombardment is 5% --/
theorem bombardment_percentage (initial_population : ℕ) (final_population : ℕ) 
  (h1 : initial_population = 4675)
  (h2 : final_population = 3553) :
  ∃ (x : ℝ), x = 5 ∧ 
  (initial_population : ℝ) * ((100 - x) / 100) * 0.8 = final_population := by
  sorry

end bombardment_percentage_l274_27404


namespace quadrilateral_circle_condition_l274_27414

-- Define the lines
def line1 (a x y : ℝ) : Prop := (a + 2) * x + (1 - a) * y - 3 = 0
def line2 (a x y : ℝ) : Prop := (a - 1) * x + (2 * a + 3) * y + 2 = 0

-- Define the property of forming a quadrilateral with coordinate axes
def forms_quadrilateral (a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
    line1 a x1 0 ∧ line1 a 0 y1 ∧ line2 a x2 0 ∧ line2 a 0 y2

-- Define the property of having a circumscribed circle
def has_circumscribed_circle (a : ℝ) : Prop :=
  forms_quadrilateral a → 
    (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0

-- The theorem to prove
theorem quadrilateral_circle_condition (a : ℝ) :
  forms_quadrilateral a → has_circumscribed_circle a → (a = 1 ∨ a = -1) :=
sorry

end quadrilateral_circle_condition_l274_27414


namespace inequality1_solution_inequality2_solution_l274_27481

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 < 0
def inequality2 (x : ℝ) : Prop := 2 * x / (x + 1) ≥ 1

-- Define the solution sets
def solution_set1 : Set ℝ := {x | 1/2 < x ∧ x < 1}
def solution_set2 : Set ℝ := {x | x < -1 ∨ x ≥ 1}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 :=
sorry

theorem inequality2_solution : 
  ∀ x : ℝ, x ≠ -1 → (inequality2 x ↔ x ∈ solution_set2) :=
sorry

end inequality1_solution_inequality2_solution_l274_27481


namespace sufficiency_not_necessity_a_squared_greater_b_squared_l274_27471

theorem sufficiency_not_necessity_a_squared_greater_b_squared (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end sufficiency_not_necessity_a_squared_greater_b_squared_l274_27471


namespace sphere_surface_area_equal_volume_cone_l274_27431

/-- Given a cone with radius 2 inches and height 6 inches, 
    prove that the surface area of a sphere with the same volume 
    is 4π(6^(2/3)) square inches. -/
theorem sphere_surface_area_equal_volume_cone (π : ℝ) : 
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 6
  let cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
  let sphere_radius : ℝ := (3 * cone_volume / (4 * π))^(1/3)
  let sphere_surface_area : ℝ := 4 * π * sphere_radius^2
  sphere_surface_area = 4 * π * 6^(2/3) := by
sorry

end sphere_surface_area_equal_volume_cone_l274_27431


namespace complex_fraction_simplification_l274_27449

theorem complex_fraction_simplification :
  1007 * ((7/4 / (3/4) + 3 / (9/4) + 1/3) / ((1+2+3+4+5) * 5 - 22)) / 19 = 4 := by
  sorry

end complex_fraction_simplification_l274_27449


namespace equation_solution_property_l274_27430

theorem equation_solution_property (m n : ℝ) : 
  (∃ x : ℝ, m * x + n - 2 = 0 ∧ x = 2) → 2 * m + n + 1 = 3 := by
  sorry

end equation_solution_property_l274_27430


namespace simplify_expression_l274_27436

-- Define the expression
def expression (x y : ℝ) : ℝ := (15*x + 45*y) + (7*x + 18*y) - (6*x + 35*y)

-- State the theorem
theorem simplify_expression :
  ∀ x y : ℝ, expression x y = 16*x + 28*y := by
  sorry

end simplify_expression_l274_27436


namespace gcd_values_count_l274_27439

theorem gcd_values_count (a b : ℕ+) (h : Nat.gcd a.val b.val * Nat.lcm a.val b.val = 180) :
  ∃ S : Finset ℕ+, (∀ x ∈ S, x = Nat.gcd a.val b.val) ∧ S.card = 4 := by
  sorry

end gcd_values_count_l274_27439


namespace inequality_theorem_l274_27479

theorem inequality_theorem (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9*x*y*z ≥ 9*(x*y + y*z + z*x) ∧
  ((x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9*x*y*z = 9*(x*y + y*z + z*x) ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end inequality_theorem_l274_27479


namespace increasing_function_inequality_l274_27408

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ)
  (h_increasing : IncreasingFunction f) (h_sum_positive : a + b > 0) :
  f a + f b > f (-a) + f (-b) := by
  sorry

end increasing_function_inequality_l274_27408


namespace gcd_power_three_l274_27473

theorem gcd_power_three : Nat.gcd (3^600 - 1) (3^612 - 1) = 3^12 - 1 := by
  sorry

end gcd_power_three_l274_27473


namespace f_min_value_solution_set_characterization_l274_27419

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem 1: The minimum value of f(x) is -3
theorem f_min_value : ∀ x : ℝ, f x ≥ -3 := by sorry

-- Theorem 2: Characterization of the solution set for the inequality
theorem solution_set_characterization :
  ∀ x : ℝ, x^2 - 8*x + 15 + f x < 0 ↔ (5 - Real.sqrt 3 < x ∧ x < 5) ∨ (5 < x ∧ x < 6) := by sorry

end f_min_value_solution_set_characterization_l274_27419


namespace max_rabbits_with_traits_l274_27482

theorem max_rabbits_with_traits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both_traits : ℕ) :
  long_ears = 13 →
  jump_far = 17 →
  both_traits ≥ 3 →
  N ≤ 27 :=
by
  sorry

end max_rabbits_with_traits_l274_27482


namespace prob_two_blue_balls_l274_27462

/-- The probability of drawing two blue balls from an urn --/
theorem prob_two_blue_balls (total : ℕ) (blue : ℕ) (h1 : total = 10) (h2 : blue = 5) :
  (blue.choose 2 : ℚ) / total.choose 2 = 2 / 9 := by
  sorry

end prob_two_blue_balls_l274_27462


namespace quadratic_equation_roots_l274_27460

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 5 * x + c = 0 ↔ x = (-5 + Real.sqrt 21) / 4 ∨ x = (-5 - Real.sqrt 21) / 4) →
  c = 1 / 2 := by
  sorry

end quadratic_equation_roots_l274_27460


namespace horner_method_operations_l274_27440

/-- The number of arithmetic operations required to evaluate a polynomial using Horner's method -/
def horner_operations (n : ℕ) : ℕ := 2 * n

/-- Theorem: For a polynomial of degree n, Horner's method requires 2n arithmetic operations -/
theorem horner_method_operations (n : ℕ) :
  horner_operations n = 2 * n :=
by sorry

end horner_method_operations_l274_27440
