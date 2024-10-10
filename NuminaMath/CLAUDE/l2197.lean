import Mathlib

namespace emmy_lost_ipods_l2197_219758

/-- The number of iPods Emmy lost -/
def ipods_lost : ℕ := sorry

/-- The number of iPods Rosa has -/
def rosa_ipods : ℕ := sorry

theorem emmy_lost_ipods : ipods_lost = 6 :=
  by
  have h1 : 14 - ipods_lost = 2 * rosa_ipods := sorry
  have h2 : (14 - ipods_lost) + rosa_ipods = 12 := sorry
  sorry

#check emmy_lost_ipods

end emmy_lost_ipods_l2197_219758


namespace complex_equation_solution_l2197_219737

theorem complex_equation_solution (a b : ℝ) (h : (3 + 4*I) * (1 + a*I) = b*I) : a = 3/4 := by
  sorry

end complex_equation_solution_l2197_219737


namespace martin_oranges_l2197_219746

/-- Represents the number of fruits Martin has initially -/
def initial_fruits : ℕ := 150

/-- Represents the number of oranges Martin has after eating half of his fruits -/
def oranges : ℕ := 50

/-- Represents the number of limes Martin has after eating half of his fruits -/
def limes : ℕ := 25

/-- Proves that Martin has 50 oranges after eating half of his fruits -/
theorem martin_oranges :
  (oranges + limes = initial_fruits / 2) ∧
  (oranges = 2 * limes) ∧
  (oranges = 50) :=
sorry

end martin_oranges_l2197_219746


namespace first_day_over_200_is_thursday_l2197_219717

def paperclips (n : Nat) : Nat := 5 * 3^n

def days : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

theorem first_day_over_200_is_thursday :
  days[4] = "Thursday" ∧
  (∀ k < 4, paperclips k ≤ 200) ∧
  paperclips 4 > 200 := by
sorry

end first_day_over_200_is_thursday_l2197_219717


namespace student_sister_weight_l2197_219723

theorem student_sister_weight (student_weight : ℝ) (weight_loss : ℝ) :
  student_weight = 90 ∧
  (student_weight - weight_loss) = 2 * ((student_weight - weight_loss) / 2) ∧
  weight_loss = 6 →
  student_weight + ((student_weight - weight_loss) / 2) = 132 :=
by sorry

end student_sister_weight_l2197_219723


namespace equation_solutions_l2197_219793

def equation (x y n : ℤ) : Prop :=
  x^3 - 3*x*y^2 + y^3 = n

theorem equation_solutions (n : ℤ) (hn : n > 0) :
  (∃ (x y : ℤ), equation x y n → 
    equation (y - x) (-x) n ∧ equation (-y) (x - y) n) ∧
  (n = 2891 → ¬∃ (x y : ℤ), equation x y n) :=
sorry

end equation_solutions_l2197_219793


namespace geometric_sequence_second_term_l2197_219709

theorem geometric_sequence_second_term (a₁ a₃ : ℝ) (h₁ : a₁ = 120) (h₃ : a₃ = 27/16) :
  ∃ b : ℝ, b > 0 ∧ b * b = a₁ * a₃ ∧ b = 15 := by
  sorry

end geometric_sequence_second_term_l2197_219709


namespace characterization_of_special_numbers_l2197_219785

/-- A natural number n > 1 satisfies the given condition if and only if it's prime or a square of a prime -/
theorem characterization_of_special_numbers (n : ℕ) (h : n > 1) :
  (∀ d : ℕ, d > 1 → d ∣ n → (d - 1) ∣ (n - 1)) ↔ 
  (Nat.Prime n ∨ ∃ p : ℕ, Nat.Prime p ∧ n = p^2) := by
  sorry

end characterization_of_special_numbers_l2197_219785


namespace largest_root_is_two_l2197_219705

/-- A polynomial of degree 6 with specific coefficients and three parameters -/
def P (a b c : ℝ) (x : ℝ) : ℝ :=
  x^6 - 6*x^5 + 17*x^4 + 6*x^3 + a*x^2 - b*x - c

/-- The theorem stating that if P has exactly three distinct double roots, the largest root is 2 -/
theorem largest_root_is_two (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    (∀ x : ℝ, P a b c x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
    (∀ x : ℝ, (x - r₁)^2 * (x - r₂)^2 * (x - r₃)^2 = P a b c x)) →
  (∃ r : ℝ, r = 2 ∧ P a b c r = 0 ∧ ∀ s : ℝ, P a b c s = 0 → s ≤ r) :=
sorry

end largest_root_is_two_l2197_219705


namespace smaller_rectangle_dimensions_l2197_219749

theorem smaller_rectangle_dimensions (square_side : ℝ) (small_width : ℝ) :
  square_side = 10 →
  small_width > 0 →
  small_width < square_side →
  small_width * square_side = (1 / 3) * square_side * square_side →
  (small_width, square_side) = (10 / 3, 10) :=
by sorry

end smaller_rectangle_dimensions_l2197_219749


namespace same_grade_percentage_l2197_219772

/-- Represents the grade distribution table -/
def gradeDistribution : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![4, 3, 2, 1],
    ![1, 6, 2, 0],
    ![3, 1, 3, 2],
    ![0, 1, 2, 2]]

/-- Total number of students -/
def totalStudents : ℕ := 36

/-- Sum of diagonal elements in the grade distribution table -/
def sameGradeCount : ℕ := (gradeDistribution 0 0) + (gradeDistribution 1 1) + (gradeDistribution 2 2) + (gradeDistribution 3 3)

/-- Theorem stating the percentage of students who received the same grade on both tests -/
theorem same_grade_percentage :
  (sameGradeCount : ℚ) / totalStudents = 5 / 12 := by sorry

end same_grade_percentage_l2197_219772


namespace canoe_downstream_speed_l2197_219731

/-- Given a canoe rowing upstream at 9 km/hr and a stream speed of 1.5 km/hr,
    the speed of the canoe when rowing downstream is 12 km/hr. -/
theorem canoe_downstream_speed :
  let upstream_speed : ℝ := 9
  let stream_speed : ℝ := 1.5
  let canoe_speed : ℝ := upstream_speed + stream_speed
  let downstream_speed : ℝ := canoe_speed + stream_speed
  downstream_speed = 12 := by sorry

end canoe_downstream_speed_l2197_219731


namespace circle_properties_l2197_219726

/-- The circle C is defined by the equation (x+1)^2 + (y-2)^2 = 4 -/
def C : Set (ℝ × ℝ) := {p | (p.1 + 1)^2 + (p.2 - 2)^2 = 4}

/-- The center of circle C -/
def center : ℝ × ℝ := (-1, 2)

/-- The radius of circle C -/
def radius : ℝ := 2

theorem circle_properties :
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 := by
  sorry

end circle_properties_l2197_219726


namespace binomial_coefficient_20_7_times_3_l2197_219789

theorem binomial_coefficient_20_7_times_3 : 3 * (Nat.choose 20 7) = 16608 := by
  sorry

end binomial_coefficient_20_7_times_3_l2197_219789


namespace problem_solution_l2197_219765

def arithmetic_sum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem problem_solution : 
  ∃ x : ℚ, 
    let n : ℕ := (196 - 2) / 2 + 1
    let S : ℕ := arithmetic_sum 2 n 2
    (S + x) / (n + 1 : ℚ) = 50 * x ∧ x = 2 := by sorry

end problem_solution_l2197_219765


namespace pen_transaction_profit_l2197_219748

/-- Calculates the profit percentage for a given transaction -/
def profit_percent (items_bought : ℕ) (price_paid : ℕ) (discount_percent : ℚ) : ℚ :=
  let cost_per_item : ℚ := price_paid / items_bought
  let selling_price_per_item : ℚ := 1 - (discount_percent / 100)
  let total_revenue : ℚ := items_bought * selling_price_per_item
  let profit : ℚ := total_revenue - price_paid
  (profit / price_paid) * 100

/-- The profit percent for the given transaction is approximately 20.52% -/
theorem pen_transaction_profit :
  ∃ ε > 0, |profit_percent 56 46 1 - 20.52| < ε :=
sorry

end pen_transaction_profit_l2197_219748


namespace solution_sum_l2197_219729

theorem solution_sum (p q : ℝ) : 
  (2^2 - 2*p + 6 = 0) → 
  (2^2 + 6*2 - q = 0) → 
  p + q = 21 := by
sorry

end solution_sum_l2197_219729


namespace geometric_sequence_product_l2197_219719

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (m : ℕ) :
  geometric_sequence a q →
  a 1 = 1 →
  q ≠ 1 →
  q ≠ -1 →
  (∃ m : ℕ, a m = a 1 * a 2 * a 3 * a 4 * a 5) →
  m = 11 := by
  sorry

end geometric_sequence_product_l2197_219719


namespace remainder_444_power_444_mod_13_l2197_219739

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l2197_219739


namespace unique_solution_set_l2197_219763

-- Define the set A
def A : Set ℝ := {a | ∃! x, (x^2 - 4) / (x + a) = 1}

-- Theorem statement
theorem unique_solution_set : A = {-17/4, -2, 2} := by
  sorry

end unique_solution_set_l2197_219763


namespace quadratic_function_properties_l2197_219708

-- Define the quadratic function
def f (x b c : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  f 1 b c = 0 →
  f 3 b c = 0 →
  f (-1) b c = 8 ∧
  (∀ x ∈ Set.Icc 2 4, f x b c ≤ 3) ∧
  (∃ x ∈ Set.Icc 2 4, f x b c = 3) ∧
  (∀ x ∈ Set.Icc 2 4, f x b c ≥ -1) ∧
  (∃ x ∈ Set.Icc 2 4, f x b c = -1) :=
by sorry

end quadratic_function_properties_l2197_219708


namespace almond_butter_ratio_is_one_third_l2197_219743

/-- The cost of a jar of peanut butter in dollars -/
def peanut_butter_cost : ℚ := 3

/-- The cost of a jar of almond butter in dollars -/
def almond_butter_cost : ℚ := 3 * peanut_butter_cost

/-- The additional cost per batch for almond butter cookies compared to peanut butter cookies -/
def additional_cost_per_batch : ℚ := 3

/-- The ratio of almond butter needed for a batch to the amount in a jar -/
def almond_butter_ratio : ℚ := additional_cost_per_batch / almond_butter_cost

theorem almond_butter_ratio_is_one_third :
  almond_butter_ratio = 1 / 3 := by sorry

end almond_butter_ratio_is_one_third_l2197_219743


namespace tank_filling_time_l2197_219780

theorem tank_filling_time (fast_rate slow_rate : ℝ) (combined_time : ℝ) : 
  fast_rate = 4 * slow_rate →
  1 / combined_time = fast_rate + slow_rate →
  combined_time = 40 →
  1 / slow_rate = 200 := by
sorry

end tank_filling_time_l2197_219780


namespace female_officers_count_l2197_219794

theorem female_officers_count (total_on_duty : ℕ) (female_percentage : ℚ) 
  (h1 : total_on_duty = 180)
  (h2 : female_percentage = 18 / 100)
  (h3 : (total_on_duty / 2 : ℚ) = female_percentage * (female_officers_total : ℚ)) :
  female_officers_total = 500 :=
by sorry

end female_officers_count_l2197_219794


namespace unique_solution_quadratic_l2197_219751

def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + 3 * x + k^2 - 4

theorem unique_solution_quadratic (k : ℝ) :
  (quadratic_equation k 0 = 0) →
  (∃! x, quadratic_equation k x = 0) →
  k = -2 := by
  sorry

end unique_solution_quadratic_l2197_219751


namespace simplify_expression_l2197_219740

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 9) - (x + 6)*(3*x + 2) = x - 66 := by
  sorry

end simplify_expression_l2197_219740


namespace next_two_numbers_after_one_l2197_219771

def square_sum (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def satisfies_condition (n : ℕ) : Prop :=
  is_perfect_square (square_sum n / n)

theorem next_two_numbers_after_one (n : ℕ) : 
  (n > 1 ∧ n < 337 → ¬satisfies_condition n) ∧
  satisfies_condition 337 ∧
  (n > 337 ∧ n < 65521 → ¬satisfies_condition n) ∧
  satisfies_condition 65521 :=
sorry

end next_two_numbers_after_one_l2197_219771


namespace negation_of_universal_proposition_l2197_219773

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x < 0 → x + 1 / x ≤ -2)) ↔ (∃ x : ℝ, x < 0 ∧ x + 1 / x > -2) := by
  sorry

end negation_of_universal_proposition_l2197_219773


namespace remaining_weight_calculation_l2197_219759

/-- Calculates the total remaining weight of groceries after an accident --/
theorem remaining_weight_calculation (green_beans_weight : ℝ) : 
  green_beans_weight = 60 →
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_remaining := rice_weight * (2/3)
  let sugar_remaining := sugar_weight * (4/5)
  rice_remaining + sugar_remaining + green_beans_weight = 120 :=
by
  sorry


end remaining_weight_calculation_l2197_219759


namespace jellybean_problem_l2197_219761

theorem jellybean_problem (initial_quantity : ℕ) : 
  (initial_quantity : ℝ) * (0.75^3) = 27 → initial_quantity = 64 := by
  sorry

end jellybean_problem_l2197_219761


namespace book_purchase_total_price_l2197_219734

theorem book_purchase_total_price : 
  let total_books : ℕ := 90
  let math_books : ℕ := 53
  let math_book_price : ℕ := 4
  let history_book_price : ℕ := 5
  let history_books : ℕ := total_books - math_books
  let total_price : ℕ := math_books * math_book_price + history_books * history_book_price
  total_price = 397 := by
sorry

end book_purchase_total_price_l2197_219734


namespace inequality_equivalence_l2197_219797

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem inequality_equivalence :
  ∀ x : ℝ, f (x^2 - 4) + f (3 * x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end inequality_equivalence_l2197_219797


namespace trig_product_equals_one_l2197_219764

theorem trig_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
sorry

end trig_product_equals_one_l2197_219764


namespace non_sunday_average_is_240_l2197_219700

/-- Represents the average number of visitors to a library on different days. -/
structure LibraryVisitors where
  sunday : ℕ
  otherDays : ℕ
  monthlyAverage : ℕ

/-- Calculates the average number of visitors on non-Sunday days given the conditions. -/
def calculateNonSundayAverage (v : LibraryVisitors) : ℕ :=
  ((v.monthlyAverage * 30) - (v.sunday * 5)) / 25

/-- Theorem stating that under the given conditions, the average number of visitors
    on non-Sunday days is 240. -/
theorem non_sunday_average_is_240 (v : LibraryVisitors)
  (h1 : v.sunday = 600)
  (h2 : v.monthlyAverage = 300) :
  calculateNonSundayAverage v = 240 := by
  sorry

#eval calculateNonSundayAverage ⟨600, 0, 300⟩

end non_sunday_average_is_240_l2197_219700


namespace hyperbola_from_ellipse_foci_l2197_219777

/-- Given an ellipse with equation x²/4 + y² = 1, prove that the hyperbola 
    with equation x²/2 - y² = 1 shares the same foci as the ellipse and 
    passes through the point (2,1) -/
theorem hyperbola_from_ellipse_foci (x y : ℝ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   (x^2 / (4 : ℝ) + y^2 = 1) ∧ 
   (c^2 = a^2 + b^2) ∧
   (a^2 = 2) ∧ 
   (b^2 = 1) ∧ 
   (c^2 = 3)) →
  (x^2 / (2 : ℝ) - y^2 = 1) ∧ 
  ((2 : ℝ)^2 / (2 : ℝ) - 1^2 = 1) :=
by sorry


end hyperbola_from_ellipse_foci_l2197_219777


namespace total_subscription_amount_l2197_219720

/-- Prove that the total subscription amount is 50000 given the conditions of the problem -/
theorem total_subscription_amount (c b a : ℕ) 
  (h1 : b = c + 5000)  -- B subscribes 5000 more than C
  (h2 : a = b + 4000)  -- A subscribes 4000 more than B
  (h3 : 14700 * (a + b + c) = 35000 * a)  -- A's profit proportion
  : a + b + c = 50000 := by
  sorry

end total_subscription_amount_l2197_219720


namespace average_of_rst_l2197_219742

theorem average_of_rst (r s t : ℝ) (h : (4 / 3) * (r + s + t) = 12) : 
  (r + s + t) / 3 = 3 := by sorry

end average_of_rst_l2197_219742


namespace snack_eaters_final_count_l2197_219724

/-- Calculates the number of remaining snack eaters after a series of events -/
def remaining_snack_eaters (initial_people : ℕ) (initial_snack_eaters : ℕ) 
  (first_new_outsiders : ℕ) (second_new_outsiders : ℕ) (second_group_leaving : ℕ) : ℕ :=
  let total_after_first_join := initial_snack_eaters + first_new_outsiders
  let remaining_after_first_leave := total_after_first_join / 2
  let total_after_second_join := remaining_after_first_leave + second_new_outsiders
  let remaining_after_second_leave := total_after_second_join - second_group_leaving
  remaining_after_second_leave / 2

theorem snack_eaters_final_count 
  (h1 : initial_people = 200)
  (h2 : initial_snack_eaters = 100)
  (h3 : first_new_outsiders = 20)
  (h4 : second_new_outsiders = 10)
  (h5 : second_group_leaving = 30) :
  remaining_snack_eaters initial_people initial_snack_eaters first_new_outsiders second_new_outsiders second_group_leaving = 20 := by
  sorry

end snack_eaters_final_count_l2197_219724


namespace root_relationship_l2197_219756

/-- Given two functions f and g, and their respective roots x₁ and x₂, prove that x₁ < x₂ -/
theorem root_relationship (f g : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (f = λ x => x + 2^x) →
  (g = λ x => x + Real.log x) →
  f x₁ = 0 →
  g x₂ = 0 →
  x₁ < x₂ := by
  sorry

end root_relationship_l2197_219756


namespace bookstore_sales_percentage_l2197_219788

theorem bookstore_sales_percentage (book_sales magazine_sales other_sales : ℝ) :
  book_sales = 45 →
  magazine_sales = 25 →
  book_sales + magazine_sales + other_sales = 100 →
  other_sales = 30 :=
by
  sorry

end bookstore_sales_percentage_l2197_219788


namespace parallel_vectors_sum_l2197_219728

/-- Given plane vectors a and b, where a is parallel to b, prove that 2a + 3b = (-4, -8) -/
theorem parallel_vectors_sum (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →  -- a is parallel to b
  (2 • a + 3 • b) = ![(-4 : ℝ), -8] := by
sorry

end parallel_vectors_sum_l2197_219728


namespace weight_placement_theorem_l2197_219770

def factorial_double (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (2 * k + 1) * factorial_double k

def weight_placement_ways (n : ℕ) : ℕ :=
  factorial_double n

theorem weight_placement_theorem (n : ℕ) (h : n > 0) :
  weight_placement_ways n = factorial_double n :=
by
  sorry

end weight_placement_theorem_l2197_219770


namespace car_to_stream_distance_l2197_219781

/-- The distance from the car to the stream in miles -/
def distance_car_to_stream : ℝ := 0.2

/-- The total distance hiked in miles -/
def total_distance : ℝ := 0.7

/-- The distance from the stream to the meadow in miles -/
def distance_stream_to_meadow : ℝ := 0.4

/-- The distance from the meadow to the campsite in miles -/
def distance_meadow_to_campsite : ℝ := 0.1

theorem car_to_stream_distance :
  distance_car_to_stream = total_distance - distance_stream_to_meadow - distance_meadow_to_campsite :=
by sorry

end car_to_stream_distance_l2197_219781


namespace probability_non_defective_pencils_l2197_219755

theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 8
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 5 / 14 :=
by sorry

end probability_non_defective_pencils_l2197_219755


namespace vectors_form_basis_l2197_219795

def v1 : Fin 2 → ℝ := ![2, 3]
def v2 : Fin 2 → ℝ := ![-4, 6]

theorem vectors_form_basis : LinearIndependent ℝ ![v1, v2] :=
sorry

end vectors_form_basis_l2197_219795


namespace intersection_point_of_AB_CD_l2197_219786

def A : ℝ × ℝ × ℝ := (3, -2, 5)
def B : ℝ × ℝ × ℝ := (13, -12, 10)
def C : ℝ × ℝ × ℝ := (-2, 5, -8)
def D : ℝ × ℝ × ℝ := (3, -1, 12)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_point_of_AB_CD :
  line_intersection A B C D = (-1/11, 1/11, 15/11) := by
  sorry

end intersection_point_of_AB_CD_l2197_219786


namespace inequality_solution_l2197_219732

theorem inequality_solution (x : ℝ) : 2 * (2 * x - 1) > 3 * x - 1 ↔ x > 1 := by
  sorry

end inequality_solution_l2197_219732


namespace line_through_points_l2197_219712

theorem line_through_points (a : ℝ) : 
  a > 0 ∧ 
  (∃ m b : ℝ, m = 2 ∧ b = 0 ∧ 
    (5 = m * a + b) ∧ 
    (a = m * 2 + b)) →
  a = 3 :=
by sorry

end line_through_points_l2197_219712


namespace stability_comparison_l2197_219713

/-- Represents the variance of a student's performance -/
structure StudentVariance where
  value : ℝ
  positive : value > 0

/-- Defines the concept of stability based on variance -/
def more_stable (a b : StudentVariance) : Prop :=
  a.value < b.value

theorem stability_comparison 
  (variance_A variance_B : StudentVariance)
  (h1 : variance_A.value = 0.05)
  (h2 : variance_B.value = 0.06) :
  more_stable variance_A variance_B :=
sorry

end stability_comparison_l2197_219713


namespace reasoning_is_deductive_l2197_219730

-- Define the set of all substances
variable (Substance : Type)

-- Define the property of being a metal
variable (is_metal : Substance → Prop)

-- Define the property of conducting electricity
variable (conducts_electricity : Substance → Prop)

-- Define iron as a specific substance
variable (iron : Substance)

-- Theorem stating that the given reasoning is deductive
theorem reasoning_is_deductive 
  (h1 : ∀ x, is_metal x → conducts_electricity x)  -- All metals can conduct electricity
  (h2 : is_metal iron)                             -- Iron is a metal
  (h3 : conducts_electricity iron)                 -- Iron can conduct electricity
  : Prop :=
sorry

end reasoning_is_deductive_l2197_219730


namespace factors_of_8_cube_5_fifth_7_square_l2197_219741

def number_of_factors (n : ℕ) : ℕ := sorry

theorem factors_of_8_cube_5_fifth_7_square :
  number_of_factors (8^3 * 5^5 * 7^2) = 180 := by sorry

end factors_of_8_cube_5_fifth_7_square_l2197_219741


namespace max_temperature_range_l2197_219779

/-- Given weather conditions and temperatures, calculate the maximum temperature range --/
theorem max_temperature_range 
  (avg_temp : ℝ) 
  (lowest_temp : ℝ) 
  (temp_fluctuation : ℝ) 
  (h1 : avg_temp = 50)
  (h2 : lowest_temp = 45)
  (h3 : temp_fluctuation = 5) :
  (avg_temp + temp_fluctuation) - lowest_temp = 10 := by
sorry

end max_temperature_range_l2197_219779


namespace product_equals_one_l2197_219782

/-- A geometric sequence with a specific property -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  root_property : a 3 * a 15 = 1 ∧ a 3 + a 15 = 6

/-- The product of five consecutive terms equals 1 -/
theorem product_equals_one (seq : GeometricSequence) :
  seq.a 7 * seq.a 8 * seq.a 9 * seq.a 10 * seq.a 11 = 1 := by
  sorry

end product_equals_one_l2197_219782


namespace equal_apple_distribution_l2197_219745

theorem equal_apple_distribution (total_apples : Nat) (num_students : Nat) 
  (h1 : total_apples = 360) (h2 : num_students = 60) :
  total_apples / num_students = 6 := by
  sorry

end equal_apple_distribution_l2197_219745


namespace andrey_stamps_problem_l2197_219760

theorem andrey_stamps_problem :
  ∃! x : ℕ, x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ 150 < x ∧ x ≤ 300 ∧ x = 208 := by
  sorry

end andrey_stamps_problem_l2197_219760


namespace tangent_line_at_point_l2197_219725

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  tangent_line point.1 (f point.1) ∧
  ∀ x : ℝ, (tangent_line x (f point.1 + (x - point.1) * (3 * point.1^2 - 1))) :=
by sorry

end tangent_line_at_point_l2197_219725


namespace browser_tabs_l2197_219733

theorem browser_tabs (T : ℚ) : 
  (9 / 40 : ℚ) * T = 90 → T = 400 := by
  sorry

end browser_tabs_l2197_219733


namespace pool_depths_l2197_219784

/-- Depths of pools problem -/
theorem pool_depths (john_depth sarah_depth susan_depth : ℝ) : 
  john_depth = 2 * sarah_depth + 5 →
  susan_depth = john_depth + sarah_depth - 3 →
  john_depth = 15 →
  sarah_depth = 5 ∧ susan_depth = 17 := by
  sorry

end pool_depths_l2197_219784


namespace sufficient_not_necessary_l2197_219753

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) := by sorry

end sufficient_not_necessary_l2197_219753


namespace abs_sum_inequality_solution_set_square_sum_geq_sqrt_product_sum_l2197_219736

-- Part Ⅰ
theorem abs_sum_inequality_solution_set (x : ℝ) :
  (|2 + x| + |2 - x| ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) := by sorry

-- Part Ⅱ
theorem square_sum_geq_sqrt_product_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  a^2 + b^2 ≥ Real.sqrt (a * b) * (a + b) := by sorry

end abs_sum_inequality_solution_set_square_sum_geq_sqrt_product_sum_l2197_219736


namespace polynomial_identity_l2197_219744

theorem polynomial_identity (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end polynomial_identity_l2197_219744


namespace problem_solution_l2197_219710

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^3 / y^2 + y^3 / x^2 + y = 5 + 1520 / 9 := by
  sorry

end problem_solution_l2197_219710


namespace tangent_line_y_intercept_l2197_219704

theorem tangent_line_y_intercept (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x - Real.log x
  let f' : ℝ → ℝ := λ x ↦ a - 1 / x
  let tangent_slope : ℝ := f' 1
  let tangent_intercept : ℝ := f 1 - tangent_slope * 1
  tangent_intercept = 1 := by sorry

end tangent_line_y_intercept_l2197_219704


namespace caco3_decomposition_spontaneity_l2197_219774

/-- Represents the thermodynamic properties of a chemical reaction -/
structure ThermodynamicProperties where
  ΔH : ℝ  -- Enthalpy change
  ΔS : ℝ  -- Entropy change

/-- Calculates the Gibbs free energy change for a given temperature -/
def gibbsFreeEnergyChange (props : ThermodynamicProperties) (T : ℝ) : ℝ :=
  props.ΔH - T * props.ΔS

/-- Theorem: For the CaCO₃ decomposition reaction, there exists a temperature
    above which the reaction becomes spontaneous -/
theorem caco3_decomposition_spontaneity 
    (props : ThermodynamicProperties) 
    (h_endothermic : props.ΔH > 0) 
    (h_disorder_increase : props.ΔS > 0) : 
    ∃ T₀ : ℝ, ∀ T > T₀, gibbsFreeEnergyChange props T < 0 := by
  sorry

end caco3_decomposition_spontaneity_l2197_219774


namespace sum_of_reciprocals_squared_l2197_219767

theorem sum_of_reciprocals_squared (a b c d : ℝ) :
  a = 2 * Real.sqrt 2 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  b = -2 * Real.sqrt 2 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  c = 2 * Real.sqrt 2 - 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  d = -2 * Real.sqrt 2 - 2 * Real.sqrt 3 + 2 * Real.sqrt 5 →
  (1/a + 1/b + 1/c + 1/d)^2 = 4/45 := by
  sorry

end sum_of_reciprocals_squared_l2197_219767


namespace quadratic_polynomial_negative_root_l2197_219722

/-- A quadratic polynomial with two distinct real roots -/
structure QuadraticPolynomial where
  P : ℝ → ℝ
  is_quadratic : ∃ (a b c : ℝ), ∀ x, P x = a * x^2 + b * x + c
  has_distinct_roots : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ P r₁ = 0 ∧ P r₂ = 0

/-- The inequality condition for the polynomial -/
def SatisfiesInequality (P : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), (abs a ≥ 2017 ∧ abs b ≥ 2017) → P (a^2 + b^2) ≥ P (2*a*b)

/-- The main theorem -/
theorem quadratic_polynomial_negative_root (p : QuadraticPolynomial) 
    (h : SatisfiesInequality p.P) : 
    ∃ (x : ℝ), x < 0 ∧ p.P x = 0 := by
  sorry

end quadratic_polynomial_negative_root_l2197_219722


namespace unit_circle_y_coordinate_l2197_219769

theorem unit_circle_y_coordinate 
  (α : Real) 
  (h1 : -3*π/2 < α ∧ α < 0) 
  (h2 : Real.cos (α - π/3) = -Real.sqrt 3 / 3) : 
  ∃ (x₀ y₀ : Real), 
    x₀^2 + y₀^2 = 1 ∧ 
    y₀ = Real.sin α ∧
    y₀ = (-Real.sqrt 6 - 3) / 6 :=
by sorry

end unit_circle_y_coordinate_l2197_219769


namespace range_of_m_l2197_219702

/-- The proposition p: x^2 - 7x + 10 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 7*x + 10 ≤ 0

/-- The proposition q: m ≤ x ≤ m + 1 -/
def q (m x : ℝ) : Prop := m ≤ x ∧ x ≤ m + 1

/-- q is a sufficient condition for p -/
def q_sufficient_for_p (m : ℝ) : Prop := ∀ x, q m x → p x

theorem range_of_m (m : ℝ) : 
  q_sufficient_for_p m → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end range_of_m_l2197_219702


namespace sum_of_cubes_l2197_219792

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end sum_of_cubes_l2197_219792


namespace john_max_books_l2197_219718

def john_money : ℕ := 2545  -- in cents
def initial_book_price : ℕ := 285  -- in cents
def discounted_book_price : ℕ := 250  -- in cents
def discount_threshold : ℕ := 10

def max_books_buyable (money : ℕ) (price : ℕ) (discount_price : ℕ) (threshold : ℕ) : ℕ :=
  if money < threshold * price then
    money / price
  else
    threshold + (money - threshold * price) / discount_price

theorem john_max_books :
  max_books_buyable john_money initial_book_price discounted_book_price discount_threshold = 8 :=
sorry

end john_max_books_l2197_219718


namespace books_returned_on_wednesday_l2197_219766

theorem books_returned_on_wednesday (initial_books : ℕ) (tuesday_out : ℕ) (thursday_out : ℕ) (final_books : ℕ) : 
  initial_books = 250 → 
  tuesday_out = 120 → 
  thursday_out = 15 → 
  final_books = 150 → 
  initial_books - tuesday_out + (initial_books - tuesday_out - final_books + thursday_out) - thursday_out = final_books := by
  sorry

end books_returned_on_wednesday_l2197_219766


namespace geometric_sequence_a9_l2197_219757

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a9 (a : ℕ → ℤ) :
  is_geometric_sequence a →
  a 2 * a 5 = -32 →
  a 3 + a 4 = 4 →
  (∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 9 = -256 := by
  sorry

end geometric_sequence_a9_l2197_219757


namespace canada_population_1998_l2197_219791

theorem canada_population_1998 : 
  (30.3 : ℝ) * 1000000 = 30300000 := by
  sorry

end canada_population_1998_l2197_219791


namespace probability_equals_three_fourths_l2197_219762

/-- The set S in R^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | -2 ≤ p.2 ∧ p.2 ≤ |p.1| ∧ -2 ≤ p.1 ∧ p.1 ≤ 2}

/-- The subset of S where |x| + |y| < 2 -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ S ∧ |p.1| + |p.2| < 2}

/-- The area of a set in R^2 -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem probability_equals_three_fourths :
  area T / area S = 3/4 := by sorry

end probability_equals_three_fourths_l2197_219762


namespace final_sum_is_130_l2197_219768

/-- Represents the financial state of Earl, Fred, and Greg --/
structure FinancialState where
  earl : Int
  fred : Int
  greg : Int

/-- Represents the debts between Earl, Fred, and Greg --/
structure Debts where
  earl_to_fred : Int
  fred_to_greg : Int
  greg_to_earl : Int

/-- Calculates the final amounts for Earl and Greg after settling all debts --/
def settle_debts (initial : FinancialState) (debts : Debts) : Int × Int :=
  let earl_final := initial.earl - debts.earl_to_fred + debts.greg_to_earl
  let greg_final := initial.greg + debts.fred_to_greg - debts.greg_to_earl
  (earl_final, greg_final)

/-- Theorem stating that Greg and Earl will have $130 together after settling all debts --/
theorem final_sum_is_130 (initial : FinancialState) (debts : Debts) :
  initial.earl = 90 →
  initial.fred = 48 →
  initial.greg = 36 →
  debts.earl_to_fred = 28 →
  debts.fred_to_greg = 32 →
  debts.greg_to_earl = 40 →
  let (earl_final, greg_final) := settle_debts initial debts
  earl_final + greg_final = 130 := by
  sorry

#check final_sum_is_130

end final_sum_is_130_l2197_219768


namespace min_value_theorem_l2197_219754

def f (x : ℝ) := |x - 2| + |x + 1|

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_min : ∀ x, f x ≥ m + n) (h_exists : ∃ x, f x = m + n) :
  (∃ x, f x = 3) ∧ 
  (m^2 + n^2 ≥ 9/2) ∧
  (m^2 + n^2 = 9/2 ↔ m = 3/2 ∧ n = 3/2) := by
sorry

end min_value_theorem_l2197_219754


namespace balanced_quadruple_inequality_l2197_219775

def balanced (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem balanced_quadruple_inequality (x : ℝ) :
  (∀ a b c d : ℝ, balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔
  x ≥ 3/2 := by sorry

end balanced_quadruple_inequality_l2197_219775


namespace problem_solution_l2197_219783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

theorem problem_solution :
  (∀ x : ℝ, x > 0 → 3*x + (f (-1) x) - 4 = 0) ∧
  (∀ a : ℝ, a > 0 → (∃! x : ℝ, g a x = 0) → a = 1) ∧
  (∀ x : ℝ, Real.exp (-2) < x → x < Real.exp 1 → g 1 x ≤ 2 * Real.exp 2 - 3 * Real.exp 1) :=
by sorry

end problem_solution_l2197_219783


namespace rectangle_dimension_change_l2197_219750

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.05 * L) (h2 : B' * L' = 1.2075 * (B * L)) : B' = 1.15 * B := by
  sorry

end rectangle_dimension_change_l2197_219750


namespace johns_calculation_l2197_219747

theorem johns_calculation (y : ℝ) : (y - 15) / 7 = 25 → (y - 7) / 5 = 36 := by
  sorry

end johns_calculation_l2197_219747


namespace f_min_correct_l2197_219735

noncomputable section

/-- The function f(x) = x^2 - 4x + (2-a)ln(x) where a ∈ ℝ and a ≠ 0 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + (2-a)*Real.log x

/-- The minimum value of f(x) on the interval [e, e^2] -/
def f_min (a : ℝ) : ℝ :=
  if a ≥ 2*(Real.exp 2 - 1)^2 then
    Real.exp 4 - 4*Real.exp 2 + 4 - 2*a
  else if 2*(Real.exp 1 - 1)^2 < a ∧ a < 2*(Real.exp 2 - 1)^2 then
    a/2 - Real.sqrt (2*a) - 3 + (2-a)*Real.log (1 + Real.sqrt (2*a)/2)
  else
    Real.exp 2 - 4*Real.exp 1 + 2 - a

theorem f_min_correct (a : ℝ) (h : a ≠ 0) :
  ∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f_min a ≤ f a x :=
sorry

end

end f_min_correct_l2197_219735


namespace square_minus_product_equals_one_l2197_219721

theorem square_minus_product_equals_one : 2002^2 - 2001 * 2003 = 1 := by
  sorry

end square_minus_product_equals_one_l2197_219721


namespace fifteenth_student_age_l2197_219715

theorem fifteenth_student_age
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat)
  (avg_age_group1 : ℝ)
  (num_group2 : Nat)
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : num_group1 = 6)
  (h4 : avg_age_group1 = 14)
  (h5 : num_group2 = 8)
  (h6 : avg_age_group2 = 16)
  (h7 : num_group1 + num_group2 + 1 = total_students) :
  total_students * avg_age_all - (num_group1 * avg_age_group1 + num_group2 * avg_age_group2) = 13 := by
  sorry

end fifteenth_student_age_l2197_219715


namespace system_solution_l2197_219776

theorem system_solution (x y : ℝ) : 
  x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2 → 
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by sorry

end system_solution_l2197_219776


namespace death_rate_is_three_per_two_seconds_l2197_219701

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per second -/
def birth_rate : ℚ := 3

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 129600

/-- Calculates the death rate in people per second -/
def death_rate : ℚ := birth_rate - (net_increase_per_day : ℚ) / seconds_per_day

/-- Theorem stating that the death rate is 3 people every two seconds -/
theorem death_rate_is_three_per_two_seconds : 
  death_rate * 2 = 3 := by sorry

end death_rate_is_three_per_two_seconds_l2197_219701


namespace remainder_seven_205_mod_12_l2197_219790

theorem remainder_seven_205_mod_12 : 7^205 % 12 = 7 := by
  sorry

end remainder_seven_205_mod_12_l2197_219790


namespace arielle_age_l2197_219738

theorem arielle_age (elvie_age arielle_age : ℕ) : 
  elvie_age = 10 → 
  elvie_age + arielle_age + elvie_age * arielle_age = 131 → 
  arielle_age = 11 := by
sorry

end arielle_age_l2197_219738


namespace no_real_roots_quadratic_l2197_219711

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + k ≠ 0) → k > 25/4 := by
  sorry

end no_real_roots_quadratic_l2197_219711


namespace smallest_prime_8_less_than_square_l2197_219778

theorem smallest_prime_8_less_than_square : 
  ∃ (n : ℕ), 17 = n^2 - 8 ∧ 
  Prime 17 ∧ 
  ∀ (m : ℕ) (p : ℕ), m < n → p = m^2 - 8 → p ≤ 0 ∨ ¬ Prime p :=
by sorry

end smallest_prime_8_less_than_square_l2197_219778


namespace sum_of_roots_even_function_l2197_219706

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f has exactly four roots if there exist exactly four distinct real numbers that make f(x) = 0 -/
def HasFourRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    (∀ x, f x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d)

theorem sum_of_roots_even_function (f : ℝ → ℝ) (heven : IsEven f) (hroots : HasFourRoots f) :
  ∃ (a b c d : ℝ), f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧ a + b + c + d = 0 := by
  sorry

end sum_of_roots_even_function_l2197_219706


namespace unique_positive_solution_l2197_219716

/-- The polynomial function f(x) = x^8 + 3x^7 + 6x^6 + 2023x^5 - 2000x^4 -/
def f (x : ℝ) : ℝ := x^8 + 3*x^7 + 6*x^6 + 2023*x^5 - 2000*x^4

/-- The theorem stating that f(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end unique_positive_solution_l2197_219716


namespace parabola_equation_l2197_219798

/-- A parabola with vertex at the origin and directrix x = -2 has the equation y^2 = 8x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ y^2 = 8*x) ↔ 
  (∀ x y, p (x, y) → (x, y) ≠ (0, 0)) ∧ 
  (∀ x, x = -2 → ∀ y, ¬p (x, y)) := by
sorry

end parabola_equation_l2197_219798


namespace price_difference_pants_belt_l2197_219707

/-- Given the total cost of pants and belt, and the price of pants, 
    calculate the difference in price between the belt and the pants. -/
theorem price_difference_pants_belt 
  (total_cost : ℝ) 
  (pants_price : ℝ) 
  (h1 : total_cost = 70.93)
  (h2 : pants_price = 34.00)
  (h3 : pants_price < total_cost - pants_price) :
  total_cost - pants_price - pants_price = 2.93 := by
  sorry


end price_difference_pants_belt_l2197_219707


namespace two_month_discount_l2197_219799

/-- Calculates the final price of an item after two consecutive percentage discounts --/
theorem two_month_discount (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  initial_price = 1000 ∧ discount1 = 10 ∧ discount2 = 20 →
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100) = 720 := by
sorry


end two_month_discount_l2197_219799


namespace mix_alloys_theorem_l2197_219714

/-- Represents an alloy of copper and zinc -/
structure Alloy where
  copper : ℝ
  zinc : ℝ

/-- The first alloy with twice as much copper as zinc -/
def alloy1 : Alloy := { copper := 2, zinc := 1 }

/-- The second alloy with five times less copper than zinc -/
def alloy2 : Alloy := { copper := 1, zinc := 5 }

/-- Mixing two alloys in a given ratio -/
def mixAlloys (a b : Alloy) (ratio : ℝ) : Alloy :=
  { copper := ratio * a.copper + b.copper,
    zinc := ratio * a.zinc + b.zinc }

/-- Theorem stating that mixing alloy1 and alloy2 in 1:2 ratio results in an alloy with twice as much zinc as copper -/
theorem mix_alloys_theorem :
  let mixedAlloy := mixAlloys alloy1 alloy2 0.5
  mixedAlloy.zinc = 2 * mixedAlloy.copper := by sorry

end mix_alloys_theorem_l2197_219714


namespace servant_pay_problem_l2197_219703

/-- The amount of money a servant receives for partial work -/
def servant_pay (full_year_pay : ℕ) (uniform_cost : ℕ) (months_worked : ℕ) : ℕ :=
  (full_year_pay * months_worked / 12) + uniform_cost

theorem servant_pay_problem :
  let full_year_pay : ℕ := 900
  let uniform_cost : ℕ := 100
  let months_worked : ℕ := 9
  servant_pay full_year_pay uniform_cost months_worked = 775 := by
sorry

#eval servant_pay 900 100 9

end servant_pay_problem_l2197_219703


namespace managers_salary_l2197_219727

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) :
  num_employees = 20 ∧ 
  avg_salary = 1500 ∧ 
  avg_increase = 100 →
  (num_employees + 1) * (avg_salary + avg_increase) - num_employees * avg_salary = 3600 :=
by sorry

end managers_salary_l2197_219727


namespace pythagorean_reciprocal_perimeter_l2197_219787

theorem pythagorean_reciprocal_perimeter 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (pythagorean_reciprocal : (a + b) / c = Real.sqrt 2) 
  (area : a * b / 2 = 4) : 
  a + b + c = 4 * Real.sqrt 2 + 4 := by
  sorry

end pythagorean_reciprocal_perimeter_l2197_219787


namespace candy_cost_l2197_219796

/-- The cost of candy given initial amounts and final amount after transaction -/
theorem candy_cost (michael_initial : ℕ) (brother_initial : ℕ) (brother_final : ℕ) 
    (h1 : michael_initial = 42)
    (h2 : brother_initial = 17)
    (h3 : brother_final = 35) :
    michael_initial / 2 + brother_initial - brother_final = 3 :=
by sorry

end candy_cost_l2197_219796


namespace hill_height_l2197_219752

/-- The height of a hill given its base depth and proportion to total vertical distance -/
theorem hill_height (base_depth : ℝ) (total_distance : ℝ) 
  (h1 : base_depth = 300)
  (h2 : base_depth = (1/4) * total_distance) : 
  total_distance - base_depth = 900 := by
  sorry

end hill_height_l2197_219752
