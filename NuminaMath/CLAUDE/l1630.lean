import Mathlib

namespace parabola_intersection_value_l1630_163058

theorem parabola_intersection_value (m : ℝ) : m^2 - m - 1 = 0 → m^2 - m + 2017 = 2018 := by
  sorry

end parabola_intersection_value_l1630_163058


namespace minimum_commission_rate_l1630_163061

/-- The minimum commission rate problem -/
theorem minimum_commission_rate 
  (old_salary : ℝ) 
  (new_base_salary : ℝ) 
  (sale_value : ℝ) 
  (min_sales : ℝ) 
  (h1 : old_salary = 75000)
  (h2 : new_base_salary = 45000)
  (h3 : sale_value = 750)
  (h4 : min_sales = 266.67)
  : ∃ (commission_rate : ℝ), 
    commission_rate ≥ (old_salary - new_base_salary) / min_sales ∧ 
    commission_rate ≥ 112.50 :=
sorry

end minimum_commission_rate_l1630_163061


namespace only_rational_root_l1630_163067

def f (x : ℚ) : ℚ := 3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

theorem only_rational_root :
  ∀ (x : ℚ), f x = 0 ↔ x = 1 := by sorry

end only_rational_root_l1630_163067


namespace theta_half_quadrants_l1630_163010

theorem theta_half_quadrants (θ : Real) 
  (h1 : |Real.cos θ| = Real.cos θ) 
  (h2 : |Real.tan θ| = -Real.tan θ) : 
  (∃ (k : ℤ), 2 * k * Real.pi + Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ 2 * k * Real.pi + Real.pi) ∨ 
  (∃ (k : ℤ), 2 * k * Real.pi + 3 * Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ 2 * k * Real.pi + 2 * Real.pi) ∨
  (∃ (k : ℤ), θ / 2 = k * Real.pi) :=
by sorry

end theta_half_quadrants_l1630_163010


namespace expression_evaluation_l1630_163015

theorem expression_evaluation : 
  -|-(3 + 3/5)| - (-(2 + 2/5)) + 4/5 = -2/5 := by sorry

end expression_evaluation_l1630_163015


namespace factorization_problem_value_problem_l1630_163037

-- Problem 1
theorem factorization_problem (a : ℝ) : 
  a^3 - 3*a^2 - 4*a + 12 = (a - 3) * (a - 2) * (a + 2) := by sorry

-- Problem 2
theorem value_problem (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) : 
  m^2 - n^2 + 2*m - 2*n = 7 := by sorry

end factorization_problem_value_problem_l1630_163037


namespace clarence_initial_oranges_l1630_163088

/-- Proves that Clarence's initial number of oranges is 5 -/
theorem clarence_initial_oranges :
  ∀ (initial total from_joyce : ℕ),
    initial + from_joyce = total →
    from_joyce = 3 →
    total = 8 →
    initial = 5 := by
  sorry

end clarence_initial_oranges_l1630_163088


namespace complex_modulus_theorem_l1630_163096

theorem complex_modulus_theorem (z : ℂ) (h : z + 3 / z = 0) : Complex.abs z = Real.sqrt 3 := by
  sorry

end complex_modulus_theorem_l1630_163096


namespace uniform_price_is_200_l1630_163006

/-- Represents the agreement between a man and his servant --/
structure Agreement where
  full_year_salary : ℕ
  service_duration : ℕ
  actual_duration : ℕ
  partial_payment : ℕ

/-- Calculates the price of the uniform given the agreement details --/
def uniform_price (a : Agreement) : ℕ :=
  let expected_payment := a.full_year_salary * a.actual_duration / a.service_duration
  expected_payment - a.partial_payment

/-- Theorem stating that the price of the uniform is 200 given the problem conditions --/
theorem uniform_price_is_200 : 
  uniform_price { full_year_salary := 800
                , service_duration := 12
                , actual_duration := 9
                , partial_payment := 400 } = 200 := by
  sorry

end uniform_price_is_200_l1630_163006


namespace salary_change_l1630_163089

theorem salary_change (original_salary : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : 
  increase_rate = 0.25 ∧ decrease_rate = 0.25 →
  (1 - decrease_rate) * (1 + increase_rate) * original_salary - original_salary = -0.0625 * original_salary := by
sorry

end salary_change_l1630_163089


namespace right_triangle_third_side_l1630_163045

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  ((a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4)) →
  c = 5 ∨ c = Real.sqrt 7 := by
sorry

end right_triangle_third_side_l1630_163045


namespace negation_of_implication_l1630_163090

theorem negation_of_implication (x y : ℝ) :
  ¬(x + y = 1 → x * y ≤ 1) ↔ (x + y = 1 → x * y > 1) := by sorry

end negation_of_implication_l1630_163090


namespace triangle_abc_solutions_l1630_163086

theorem triangle_abc_solutions (b c : ℝ) (angle_B : ℝ) :
  b = 3 → c = 3 * Real.sqrt 3 → angle_B = π / 6 →
  ∃ (a angle_A angle_C : ℝ),
    ((angle_A = π / 2 ∧ angle_C = π / 3 ∧ a = Real.sqrt 21) ∨
     (angle_A = π / 6 ∧ angle_C = 2 * π / 3 ∧ a = 3)) ∧
    angle_A + angle_B + angle_C = π ∧
    a / (Real.sin angle_A) = b / (Real.sin angle_B) ∧
    b / (Real.sin angle_B) = c / (Real.sin angle_C) :=
by sorry

end triangle_abc_solutions_l1630_163086


namespace f_extrema_and_monotonicity_l1630_163042

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + a)

theorem f_extrema_and_monotonicity :
  (∃ (x_max x_min : ℝ), f (-3) x_max = 6 * Real.exp (-3) ∧
                        f (-3) x_min = -2 * Real.exp 1 ∧
                        ∀ x, f (-3) x ≤ f (-3) x_max ∧
                              f (-3) x ≥ f (-3) x_min) ∧
  (∀ a, (∀ x y, x < y → f a x < f a y) → a ≥ 1) :=
by sorry

end f_extrema_and_monotonicity_l1630_163042


namespace acute_triangle_inequality_l1630_163047

theorem acute_triangle_inequality (A B C : Real) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hABC : A + B + C = π) (hAcute : A < π/2 ∧ B < π/2 ∧ C < π/2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C)
  ≤ π * (1 / A + 1 / B + 1 / C) := by
  sorry

end acute_triangle_inequality_l1630_163047


namespace searchlight_dark_time_l1630_163005

/-- The number of revolutions per minute for the searchlight -/
def revolutions_per_minute : ℝ := 4

/-- The probability of staying in the dark for at least a certain number of seconds -/
def probability : ℝ := 0.6666666666666667

/-- The time in seconds for which the probability applies -/
def dark_time : ℝ := 10

theorem searchlight_dark_time :
  revolutions_per_minute = 4 ∧ probability = 0.6666666666666667 →
  dark_time = 10 := by sorry

end searchlight_dark_time_l1630_163005


namespace problem_distribution_l1630_163082

theorem problem_distribution (n m : ℕ) (hn : n = 10) (hm : m = 7) :
  (Nat.choose n m * Nat.factorial m * n^(n - m) : ℕ) = 712800000 :=
by sorry

end problem_distribution_l1630_163082


namespace valid_seats_29x29_l1630_163046

/-- Represents a grid of seats -/
def Grid (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if two positions in the grid are adjacent -/
def adjacent (n : ℕ) (p q : Fin n × Fin n) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Counts the number of valid seats in an n x n grid -/
def validSeats (n : ℕ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem valid_seats_29x29 :
  validSeats 29 = 421 :=
sorry

end valid_seats_29x29_l1630_163046


namespace remainder_problem_l1630_163040

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : 90 % (k^2) = 10) : 150 % k = 2 := by
  sorry

end remainder_problem_l1630_163040


namespace count_propositions_and_true_propositions_l1630_163009

-- Define the type for statements
inductive Statement
| RhetoricalQuestion
| Question
| Proposition (isTrue : Bool)
| ExclamatoryStatement
| ConstructionLanguage

-- Define the list of statements
def statements : List Statement := [
  Statement.RhetoricalQuestion,
  Statement.Question,
  Statement.Proposition false,
  Statement.ExclamatoryStatement,
  Statement.Proposition false,
  Statement.ConstructionLanguage
]

-- Theorem to prove
theorem count_propositions_and_true_propositions :
  (statements.filter (fun s => match s with
    | Statement.Proposition _ => true
    | _ => false
  )).length = 2 ∧
  (statements.filter (fun s => match s with
    | Statement.Proposition true => true
    | _ => false
  )).length = 0 := by
  sorry

end count_propositions_and_true_propositions_l1630_163009


namespace triangle_isosceles_theorem_l1630_163060

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A triangle is isosceles if at least two of its sides are equal. -/
def isIsosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c

theorem triangle_isosceles_theorem (a b c : ℕ) 
  (ha : isPrime a) (hb : isPrime b) (hc : isPrime c) 
  (hsum : a + b + c = 16) : 
  isIsosceles a b c := by
  sorry

#check triangle_isosceles_theorem

end triangle_isosceles_theorem_l1630_163060


namespace adjacent_book_left_of_middle_adjacent_book_not_right_of_middle_l1630_163081

/-- Represents the price of a book at a given position. -/
def book_price (c : ℕ) (n : ℕ) : ℕ := c + 2 * (n - 1)

/-- The theorem stating that the adjacent book is to the left of the middle book. -/
theorem adjacent_book_left_of_middle (c : ℕ) : 
  book_price c 31 = book_price c 16 + book_price c 15 :=
sorry

/-- The theorem stating that the adjacent book cannot be to the right of the middle book. -/
theorem adjacent_book_not_right_of_middle (c : ℕ) : 
  book_price c 31 ≠ book_price c 16 + book_price c 17 :=
sorry

end adjacent_book_left_of_middle_adjacent_book_not_right_of_middle_l1630_163081


namespace recurrence_sequence_property_l1630_163019

/-- An integer sequence satisfying the given recurrence relation -/
def RecurrenceSequence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

/-- The main theorem -/
theorem recurrence_sequence_property
  (m : ℤ) (a : ℕ → ℤ) (h_m : |m| ≥ 2)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_recurrence : RecurrenceSequence m a)
  (r s : ℕ) (h_rs : r > s ∧ s ≥ 2)
  (h_equal : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| := by sorry

end recurrence_sequence_property_l1630_163019


namespace fraction_meaningful_iff_not_three_l1630_163062

theorem fraction_meaningful_iff_not_three (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by
sorry

end fraction_meaningful_iff_not_three_l1630_163062


namespace charlie_flutes_l1630_163091

theorem charlie_flutes (charlie_flutes : ℕ) (charlie_horns : ℕ) (charlie_harps : ℕ) 
  (carli_flutes : ℕ) (carli_horns : ℕ) (carli_harps : ℕ) : 
  charlie_horns = 2 →
  charlie_harps = 1 →
  carli_flutes = 2 * charlie_flutes →
  carli_horns = charlie_horns / 2 →
  carli_harps = 0 →
  charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns + carli_harps = 7 →
  charlie_flutes = 1 := by
sorry

end charlie_flutes_l1630_163091


namespace mod_23_equivalence_l1630_163077

theorem mod_23_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 39548 ≡ n [ZMOD 23] ∧ n = 13 := by
  sorry

end mod_23_equivalence_l1630_163077


namespace car_demand_and_profit_l1630_163074

-- Define the total demand function
def R (x : ℕ) : ℚ := (1/2) * x * (x + 1) * (39 - 2*x)

-- Define the purchase price function
def W (x : ℕ) : ℚ := 150000 + 2000*x

-- Define the constraints
def valid_x (x : ℕ) : Prop := x > 0 ∧ x ≤ 6

-- Define the demand function
def g (x : ℕ) : ℚ := -3*x^2 + 40*x

-- Define the monthly profit function
def f (x : ℕ) : ℚ := (185000 - W x) * g x

theorem car_demand_and_profit 
  (h : ∀ x, valid_x x → R x - R (x-1) = g x) :
  (∀ x, valid_x x → g x = -3*x^2 + 40*x) ∧ 
  (∀ x, valid_x x → f x ≤ f 5) ∧
  (f 5 = 3125000) := by
  sorry


end car_demand_and_profit_l1630_163074


namespace inequality_solution_l1630_163055

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  ((1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) ∨ (8 < x ∧ x < 10)) :=
by sorry

end inequality_solution_l1630_163055


namespace bob_and_bill_transfer_probability_l1630_163032

theorem bob_and_bill_transfer_probability (total_students : ℕ) (transfer_students : ℕ) (num_classes : ℕ) :
  total_students = 32 →
  transfer_students = 2 →
  num_classes = 2 →
  (1 : ℚ) / (Nat.choose total_students transfer_students * num_classes) = 1 / 992 :=
by sorry

end bob_and_bill_transfer_probability_l1630_163032


namespace function_passes_through_point_l1630_163052

theorem function_passes_through_point (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x - 1)
  f 1 = 2 := by sorry

end function_passes_through_point_l1630_163052


namespace difference_of_squares_times_three_l1630_163070

theorem difference_of_squares_times_three :
  (650^2 - 350^2) * 3 = 900000 := by
  sorry

end difference_of_squares_times_three_l1630_163070


namespace non_sibling_probability_l1630_163093

/-- Represents a person in the room -/
structure Person where
  siblings : Nat

/-- Represents the room with people -/
def Room : Type := List Person

/-- The number of people in the room -/
def room_size : Nat := 7

/-- The condition that 4 people have exactly 1 sibling -/
def one_sibling_count (room : Room) : Prop :=
  (room.filter (fun p => p.siblings = 1)).length = 4

/-- The condition that 3 people have exactly 2 siblings -/
def two_siblings_count (room : Room) : Prop :=
  (room.filter (fun p => p.siblings = 2)).length = 3

/-- The probability of selecting two non-siblings -/
def prob_non_siblings (room : Room) : ℚ :=
  16 / 21

/-- The main theorem -/
theorem non_sibling_probability (room : Room) :
  room.length = room_size →
  one_sibling_count room →
  two_siblings_count room →
  prob_non_siblings room = 16 / 21 := by
  sorry


end non_sibling_probability_l1630_163093


namespace inverse_proportion_point_difference_l1630_163025

/-- 
Given two points A(x₁, y₁) and B(x₂, y₂) on the graph of y = -2/x,
where x₁ < 0 < x₂, prove that y₁ - y₂ > 0.
-/
theorem inverse_proportion_point_difference (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = -2 / x₁)
  (h2 : y₂ = -2 / x₂)
  (h3 : x₁ < 0)
  (h4 : 0 < x₂) : 
  y₁ - y₂ > 0 := by
  sorry

end inverse_proportion_point_difference_l1630_163025


namespace surface_dots_eq_105_l1630_163073

/-- Represents a standard die -/
structure Die where
  faces : Fin 6 → Nat
  sum_21 : (faces 0) + (faces 1) + (faces 2) + (faces 3) + (faces 4) + (faces 5) = 21

/-- Represents the solid made of glued dice -/
structure DiceSolid where
  dice : Fin 7 → Die
  glued_faces_same : ∀ (i j : Fin 7) (f1 f2 : Fin 6), 
    (dice i).faces f1 = (dice j).faces f2 → i ≠ j

def surface_dots (solid : DiceSolid) : Nat :=
  sorry

theorem surface_dots_eq_105 (solid : DiceSolid) : 
  surface_dots solid = 105 := by
  sorry

end surface_dots_eq_105_l1630_163073


namespace evaluate_complex_exponential_l1630_163029

theorem evaluate_complex_exponential : (3^2)^(3^(3^2)) = 9^19683 := by
  sorry

end evaluate_complex_exponential_l1630_163029


namespace five_pairs_l1630_163051

/-- The number of ordered pairs (b,c) of positive integers satisfying the given conditions -/
def count_pairs : ℕ := 
  (Finset.filter (fun p : ℕ × ℕ => 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧ b^2 ≤ 9*c ∧ c^2 ≤ 9*b) 
  (Finset.product (Finset.range 4) (Finset.range 4))).card

/-- The theorem stating that there are exactly 5 such pairs -/
theorem five_pairs : count_pairs = 5 := by sorry

end five_pairs_l1630_163051


namespace ellipse_m_range_l1630_163030

-- Define the equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1

-- State the theorem
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ioo (-3/2 : ℝ) (-1) :=
by sorry

end ellipse_m_range_l1630_163030


namespace log_system_solutions_l1630_163038

noncomputable def solve_log_system (x y : ℝ) : Prop :=
  x > y ∧ y > 0 ∧
  Real.log (x - y) + Real.log 2 = (1 / 2) * (Real.log x - Real.log y) ∧
  Real.log (x + y) - Real.log 3 = (1 / 2) * (Real.log y - Real.log x)

theorem log_system_solutions :
  (∃ (x y : ℝ), solve_log_system x y ∧ 
    ((x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) ∨ 
     (x = 3 * Real.sqrt 3 / 4 ∧ y = Real.sqrt 3 / 4))) ∧
  (∀ (x y : ℝ), solve_log_system x y → 
    ((x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) ∨ 
     (x = 3 * Real.sqrt 3 / 4 ∧ y = Real.sqrt 3 / 4))) :=
by sorry

end log_system_solutions_l1630_163038


namespace scalene_triangle_distinct_lines_l1630_163094

/-- A scalene triangle is a triangle where all sides and angles are different -/
structure ScaleneTriangle where
  -- We don't need to define the specific properties here, just the existence of the triangle
  exists_triangle : True

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side -/
def altitude (t : ScaleneTriangle) : ℕ := 3

/-- A median of a triangle is a line segment from a vertex to the midpoint of the opposite side -/
def median (t : ScaleneTriangle) : ℕ := 3

/-- An angle bisector of a triangle is a line that divides an angle into two equal parts -/
def angle_bisector (t : ScaleneTriangle) : ℕ := 3

/-- The total number of distinct lines in a scalene triangle -/
def total_distinct_lines (t : ScaleneTriangle) : ℕ :=
  altitude t + median t + angle_bisector t

theorem scalene_triangle_distinct_lines (t : ScaleneTriangle) :
  total_distinct_lines t = 9 := by
  sorry

end scalene_triangle_distinct_lines_l1630_163094


namespace sixth_number_is_eight_l1630_163057

/-- A structure representing an increasing list of consecutive integers -/
structure ConsecutiveIntegerList where
  start : ℤ
  length : ℕ
  increasing : 0 < length

/-- The nth number in the list -/
def ConsecutiveIntegerList.nthNumber (list : ConsecutiveIntegerList) (n : ℕ) : ℤ :=
  list.start + n - 1

/-- The property that the sum of the 3rd and 4th numbers is 11 -/
def sumProperty (list : ConsecutiveIntegerList) : Prop :=
  list.nthNumber 3 + list.nthNumber 4 = 11

theorem sixth_number_is_eight (list : ConsecutiveIntegerList) 
    (h : sumProperty list) : list.nthNumber 6 = 8 := by
  sorry

end sixth_number_is_eight_l1630_163057


namespace weight_replacement_l1630_163017

/-- Given 5 people, if replacing one person with a new person weighing 70 kg
    increases the average weight by 4 kg, then the replaced person weighed 50 kg. -/
theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (new_weight : ℝ) :
  initial_count = 5 →
  weight_increase = 4 →
  new_weight = 70 →
  (initial_count : ℝ) * weight_increase = new_weight - 50 := by
  sorry

end weight_replacement_l1630_163017


namespace unique_three_digit_number_l1630_163020

/-- A positive integer is a multiple of 3, 5, 7, and 9 if and only if it's a multiple of their LCM -/
axiom multiple_of_3_5_7_9 (n : ℕ) : (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ (9 ∣ n) ↔ 315 ∣ n

/-- The theorem stating that 314 is the unique three-digit positive integer
    that is one less than a multiple of 3, 5, 7, and 9 -/
theorem unique_three_digit_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ ∃ m : ℕ, n + 1 = 315 * m :=
sorry

end unique_three_digit_number_l1630_163020


namespace yangtze_length_scientific_notation_l1630_163049

/-- The length of the Yangtze River in meters -/
def yangtze_length : ℕ := 6300000

/-- The scientific notation representation of the Yangtze River's length -/
def yangtze_scientific : ℝ := 6.3 * (10 ^ 6)

theorem yangtze_length_scientific_notation : 
  (yangtze_length : ℝ) = yangtze_scientific := by sorry

end yangtze_length_scientific_notation_l1630_163049


namespace sally_received_quarters_l1630_163011

/-- The number of quarters Sally initially had -/
def initial_quarters : ℕ := 760

/-- The number of quarters Sally now has -/
def final_quarters : ℕ := 1178

/-- The number of quarters Sally received -/
def received_quarters : ℕ := final_quarters - initial_quarters

theorem sally_received_quarters : received_quarters = 418 := by
  sorry

end sally_received_quarters_l1630_163011


namespace new_light_wattage_l1630_163035

theorem new_light_wattage (original_wattage : ℝ) (increase_percentage : ℝ) : 
  original_wattage = 110 → 
  increase_percentage = 30 → 
  original_wattage * (1 + increase_percentage / 100) = 143 := by
sorry

end new_light_wattage_l1630_163035


namespace find_number_multiplied_by_9999_l1630_163001

theorem find_number_multiplied_by_9999 :
  ∃! x : ℤ, x * 9999 = 724807415 :=
by
  sorry

end find_number_multiplied_by_9999_l1630_163001


namespace root_quadratic_equation_l1630_163034

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - m - 110 = 0 → (m - 1)^2 + m = 111 := by
  sorry

end root_quadratic_equation_l1630_163034


namespace bus_passengers_problem_l1630_163084

/-- Given a bus with an initial number of passengers and a number of passengers who got off,
    calculate the number of passengers remaining on the bus. -/
def passengers_remaining (initial : ℕ) (got_off : ℕ) : ℕ :=
  initial - got_off

/-- Theorem stating that given 90 initial passengers and 47 passengers who got off,
    the number of remaining passengers is 43. -/
theorem bus_passengers_problem :
  passengers_remaining 90 47 = 43 := by
  sorry

end bus_passengers_problem_l1630_163084


namespace arithmetic_sequence_sum_l1630_163069

/-- The sum of an arithmetic sequence with first term 2, common difference 4, and 15 terms -/
def arithmetic_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (2 + 4 * n) + arithmetic_sum n

theorem arithmetic_sequence_sum : arithmetic_sum 15 = 450 := by
  sorry

end arithmetic_sequence_sum_l1630_163069


namespace arithmetic_sequence_problem_l1630_163000

/-- An arithmetic sequence with first term 2 and sum of first 3 terms equal to 12 has its 6th term equal to 12 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 1 + a 2 + a 3 = 12 →                -- sum of first 3 terms is 12
  a 6 = 12 := by                        -- 6th term is 12
sorry

end arithmetic_sequence_problem_l1630_163000


namespace geometric_sequence_iff_a_eq_plus_minus_six_l1630_163063

/-- A sequence of three real numbers is geometric if the ratio between consecutive terms is constant. -/
def IsGeometricSequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

/-- The main theorem stating that the sequence 4, a, 9 is geometric if and only if a = 6 or a = -6 -/
theorem geometric_sequence_iff_a_eq_plus_minus_six :
  ∀ a : ℝ, IsGeometricSequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by sorry

end geometric_sequence_iff_a_eq_plus_minus_six_l1630_163063


namespace sum_of_digits_of_power_of_two_unbounded_l1630_163056

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: For any positive real number M, there exists a positive integer n 
    such that the sum of the digits of 2^n is greater than M -/
theorem sum_of_digits_of_power_of_two_unbounded :
  ∀ M : ℝ, M > 0 → ∃ n : ℕ, (sumOfDigits (2^n : ℕ)) > M := by sorry

end sum_of_digits_of_power_of_two_unbounded_l1630_163056


namespace gcd_6Tn_nplus1_eq_one_l1630_163044

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- Theorem: The GCD of 6T_n and n+1 is always 1 for positive integers n -/
theorem gcd_6Tn_nplus1_eq_one (n : ℕ+) : Nat.gcd (6 * T n) (n + 1) = 1 := by
  sorry

end gcd_6Tn_nplus1_eq_one_l1630_163044


namespace seashells_given_to_jason_l1630_163039

theorem seashells_given_to_jason (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 66) (h2 : remaining_seashells = 14) : 
  initial_seashells - remaining_seashells = 52 := by
  sorry

end seashells_given_to_jason_l1630_163039


namespace constant_value_l1630_163099

/-- The function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The theorem stating that if the equation has a solution x = 0.4, then c = 1 -/
theorem constant_value (c : ℝ) :
  (∃ x : ℝ, x = 0.4 ∧ (3 * f (x - 2)) / f 0 + 4 = f (2 * x + c)) → c = 1 := by
  sorry

end constant_value_l1630_163099


namespace arithmetic_series_sum_l1630_163095

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℤ := k^2 - 1
  let d : ℤ := 1
  let n : ℕ := 2 * k
  let S := (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2
  S = 2 * k^3 + 2 * k^2 - 3 * k :=
by sorry

end arithmetic_series_sum_l1630_163095


namespace average_speed_round_trip_l1630_163016

/-- Calculates the average speed of a round trip journey given the distance and times for each leg. -/
theorem average_speed_round_trip 
  (uphill_distance : ℝ) 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) : 
  (2 * uphill_distance) / (uphill_time + downhill_time) = 4 :=
by
  sorry

#check average_speed_round_trip 2 (45/60) (15/60)

end average_speed_round_trip_l1630_163016


namespace arithmetic_mean_root_mean_square_inequality_l1630_163048

theorem arithmetic_mean_root_mean_square_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
  ((a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) :=
by sorry

end arithmetic_mean_root_mean_square_inequality_l1630_163048


namespace interest_credited_proof_l1630_163050

/-- The interest rate per annum as a decimal -/
def interest_rate : ℚ := 5 / 100

/-- The time period in years -/
def time_period : ℚ := 2 / 12

/-- The total amount after interest -/
def total_amount : ℚ := 255.31

/-- The simple interest formula -/
def simple_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate * time_period)

theorem interest_credited_proof :
  ∃ (principal : ℚ),
    simple_interest principal = total_amount ∧
    (total_amount - principal) * 100 = 210 := by
  sorry

end interest_credited_proof_l1630_163050


namespace factorization_identities_l1630_163004

theorem factorization_identities (x y m n p : ℝ) : 
  (x^2 + 2*x + 1 - y^2 = (x + y + 1)*(x - y + 1)) ∧ 
  (m^2 - n^2 - 2*n*p - p^2 = (m + n + p)*(m - n - p)) := by
  sorry

end factorization_identities_l1630_163004


namespace solutions_sum_and_product_l1630_163027

theorem solutions_sum_and_product : ∃ (x₁ x₂ : ℝ),
  (x₁ - 6)^2 = 49 ∧
  (x₂ - 6)^2 = 49 ∧
  x₁ + x₂ = 12 ∧
  x₁ * x₂ = -13 :=
by sorry

end solutions_sum_and_product_l1630_163027


namespace unique_rectangles_l1630_163078

/-- A rectangle with integer dimensions satisfying area and perimeter conditions -/
structure Rectangle where
  w : ℕ+  -- width
  l : ℕ+  -- length
  area_eq : w * l = 18
  perimeter_eq : 2 * w + 2 * l = 18

/-- The theorem stating that only two rectangles satisfy the conditions -/
theorem unique_rectangles : 
  ∀ r : Rectangle, (r.w = 3 ∧ r.l = 6) ∨ (r.w = 6 ∧ r.l = 3) :=
sorry

end unique_rectangles_l1630_163078


namespace unique_function_theorem_l1630_163031

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10^10}

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x ∈ S, f x ∈ S) ∧
  (∀ x ∈ S, f (x + 1) ≡ f (f x) + 1 [MOD 10^10]) ∧
  (f (10^10 + 1) = f 1)

theorem unique_function_theorem :
  ∀ f : ℕ → ℕ, is_valid_function f →
    ∀ x ∈ S, f x ≡ x [MOD 10^10] :=
by sorry

end unique_function_theorem_l1630_163031


namespace polynomial_coefficient_sum_l1630_163002

theorem polynomial_coefficient_sum (a b c d : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 3*x^2 + 7*x - 6) →
  a + b + c + d = 7 := by
sorry

end polynomial_coefficient_sum_l1630_163002


namespace ellipse_equation_from_conditions_l1630_163065

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- The focal distance -/
  c : ℝ
  /-- Assumption that a > b > 0 -/
  h₁ : a > b ∧ b > 0
  /-- Relationship between a, b, and c -/
  h₂ : c^2 = a^2 - b^2

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.b^2 + y^2 / e.a^2 = 1

/-- Theorem stating the equation of the ellipse given the conditions -/
theorem ellipse_equation_from_conditions
  (e : Ellipse)
  (focus_on_y_axis : e.c = e.a * (1/2))
  (focal_length : 2 * e.c = 8) :
  ellipse_equation e = fun x y ↦ x^2 / 48 + y^2 / 64 = 1 := by
  sorry

end ellipse_equation_from_conditions_l1630_163065


namespace evans_books_multiple_l1630_163024

/-- Proves the multiple of Evan's current books in 5 years --/
theorem evans_books_multiple (books_two_years_ago : ℕ) (books_decrease : ℕ) (books_in_five_years : ℕ) : 
  books_two_years_ago = 200 →
  books_decrease = 40 →
  books_in_five_years = 860 →
  ∃ (current_books : ℕ) (multiple : ℕ),
    current_books = books_two_years_ago - books_decrease ∧
    books_in_five_years = multiple * current_books + 60 ∧
    multiple = 5 := by
  sorry

#check evans_books_multiple

end evans_books_multiple_l1630_163024


namespace taylors_pets_l1630_163059

theorem taylors_pets (taylor_pets : ℕ) (total_pets : ℕ) : 
  (3 * (2 * taylor_pets) + 2 * 2 + taylor_pets = total_pets) →
  (total_pets = 32) →
  (taylor_pets = 4) := by
sorry

end taylors_pets_l1630_163059


namespace complex_arithmetic_expression_l1630_163014

theorem complex_arithmetic_expression : 
  ((520 * 0.43) / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := by
  sorry

end complex_arithmetic_expression_l1630_163014


namespace book_price_calculation_l1630_163097

/-- Given a book with a suggested retail price, this theorem proves that
    if the marked price is 60% of the suggested retail price, and a customer
    pays 60% of the marked price, then the customer pays 36% of the
    suggested retail price. -/
theorem book_price_calculation (suggested_retail_price : ℝ) :
  let marked_price := 0.6 * suggested_retail_price
  let customer_price := 0.6 * marked_price
  customer_price = 0.36 * suggested_retail_price := by
  sorry

#check book_price_calculation

end book_price_calculation_l1630_163097


namespace parents_age_when_mark_born_l1630_163092

/-- Given the ages of Mark and John, and their parents' current age relative to John's,
    prove the age of the parents when Mark was born. -/
theorem parents_age_when_mark_born
  (mark_age : ℕ)
  (john_age_diff : ℕ)
  (parents_age_factor : ℕ)
  (h1 : mark_age = 18)
  (h2 : john_age_diff = 10)
  (h3 : parents_age_factor = 5) :
  mark_age - (parents_age_factor * (mark_age - john_age_diff)) = 22 :=
by sorry

end parents_age_when_mark_born_l1630_163092


namespace unique_age_pair_l1630_163071

/-- The set of possible ages for X's sons -/
def AgeSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Predicate for pairs of ages that satisfy the product condition -/
def ProductCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∃ (c d : ℕ), c ≠ a ∧ d ≠ b ∧ c * d = a * b ∧ c ∈ AgeSet ∧ d ∈ AgeSet

/-- Predicate for pairs of ages that satisfy the ratio condition -/
def RatioCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∃ (c d : ℕ), c ≠ a ∧ d ≠ b ∧ c * b = a * d ∧ c ∈ AgeSet ∧ d ∈ AgeSet

/-- Predicate for pairs of ages that satisfy the difference condition -/
def DifferenceCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∀ (c d : ℕ), c ∈ AgeSet → d ∈ AgeSet → c - d = a - b → (c = a ∧ d = b) ∨ (c = b ∧ d = a)

/-- Theorem stating that (8, 2) is the only pair satisfying all conditions -/
theorem unique_age_pair : ∀ (a b : ℕ), 
  ProductCondition a b ∧ RatioCondition a b ∧ DifferenceCondition a b ↔ (a = 8 ∧ b = 2) ∨ (a = 2 ∧ b = 8) :=
sorry

end unique_age_pair_l1630_163071


namespace pairs_sold_proof_l1630_163008

def total_amount : ℝ := 588
def average_price : ℝ := 9.8

theorem pairs_sold_proof :
  total_amount / average_price = 60 :=
by sorry

end pairs_sold_proof_l1630_163008


namespace student_number_factor_l1630_163068

theorem student_number_factor (f : ℝ) : 120 * f - 138 = 102 → f = 2 := by
  sorry

end student_number_factor_l1630_163068


namespace problem_solution_l1630_163054

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ab ≤ 4 ∧ Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2 ∧ a^2 + b^2 ≥ 8 := by
  sorry

end problem_solution_l1630_163054


namespace chiquita_height_l1630_163064

theorem chiquita_height :
  ∀ (chiquita_height martinez_height : ℝ),
    martinez_height = chiquita_height + 2 →
    chiquita_height + martinez_height = 12 →
    chiquita_height = 5 := by
  sorry

end chiquita_height_l1630_163064


namespace circle_k_range_l1630_163018

-- Define the equation
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 2*y + 4*k + 1 = 0

-- Define what it means for the equation to represent a circle
def is_circle (k : ℝ) : Prop :=
  ∃ (h r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y k ↔ (x + 2)^2 + (y + 1)^2 = r^2

-- Theorem statement
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end circle_k_range_l1630_163018


namespace probability_masters_degree_expected_value_bachelors_or_higher_male_education_greater_than_female_l1630_163022

/-- Represents the education levels in the census data -/
inductive EducationLevel
  | NoSchooling
  | PrimarySchool
  | JuniorHighSchool
  | HighSchool
  | CollegeAssociate
  | CollegeBachelor
  | MastersDegree
  | DoctoralDegree

/-- Represents the gender in the census data -/
inductive Gender
  | Male
  | Female

/-- Census data for City Z -/
def censusData : Gender → EducationLevel → Float
  | Gender.Male, EducationLevel.NoSchooling => 0.00
  | Gender.Male, EducationLevel.PrimarySchool => 0.03
  | Gender.Male, EducationLevel.JuniorHighSchool => 0.14
  | Gender.Male, EducationLevel.HighSchool => 0.11
  | Gender.Male, EducationLevel.CollegeAssociate => 0.07
  | Gender.Male, EducationLevel.CollegeBachelor => 0.11
  | Gender.Male, EducationLevel.MastersDegree => 0.03
  | Gender.Male, EducationLevel.DoctoralDegree => 0.01
  | Gender.Female, EducationLevel.NoSchooling => 0.01
  | Gender.Female, EducationLevel.PrimarySchool => 0.04
  | Gender.Female, EducationLevel.JuniorHighSchool => 0.11
  | Gender.Female, EducationLevel.HighSchool => 0.11
  | Gender.Female, EducationLevel.CollegeAssociate => 0.08
  | Gender.Female, EducationLevel.CollegeBachelor => 0.12
  | Gender.Female, EducationLevel.MastersDegree => 0.03
  | Gender.Female, EducationLevel.DoctoralDegree => 0.00

/-- Proportion of residents aged 15 and above in City Z -/
def proportionAged15AndAbove : Float := 0.85

/-- Theorem 1: Probability of selecting a person aged 15 and above with a Master's degree -/
theorem probability_masters_degree : 
  proportionAged15AndAbove * (censusData Gender.Male EducationLevel.MastersDegree + 
  censusData Gender.Female EducationLevel.MastersDegree) = 0.051 := by sorry

/-- Theorem 2: Expected value of X (number of people with Bachelor's degree or higher among two randomly selected residents aged 15 and above) -/
theorem expected_value_bachelors_or_higher : 
  let p := censusData Gender.Male EducationLevel.CollegeBachelor + 
           censusData Gender.Female EducationLevel.CollegeBachelor +
           censusData Gender.Male EducationLevel.MastersDegree + 
           censusData Gender.Female EducationLevel.MastersDegree +
           censusData Gender.Male EducationLevel.DoctoralDegree + 
           censusData Gender.Female EducationLevel.DoctoralDegree
  2 * p * (1 - p) + 2 * p * p = 0.6 := by sorry

/-- Theorem 3: Relationship between average years of education for male and female residents -/
theorem male_education_greater_than_female :
  let male_avg := 0 * censusData Gender.Male EducationLevel.NoSchooling +
                  6 * censusData Gender.Male EducationLevel.PrimarySchool +
                  9 * censusData Gender.Male EducationLevel.JuniorHighSchool +
                  12 * censusData Gender.Male EducationLevel.HighSchool +
                  16 * (censusData Gender.Male EducationLevel.CollegeAssociate +
                        censusData Gender.Male EducationLevel.CollegeBachelor +
                        censusData Gender.Male EducationLevel.MastersDegree +
                        censusData Gender.Male EducationLevel.DoctoralDegree)
  let female_avg := 0 * censusData Gender.Female EducationLevel.NoSchooling +
                    6 * censusData Gender.Female EducationLevel.PrimarySchool +
                    9 * censusData Gender.Female EducationLevel.JuniorHighSchool +
                    12 * censusData Gender.Female EducationLevel.HighSchool +
                    16 * (censusData Gender.Female EducationLevel.CollegeAssociate +
                          censusData Gender.Female EducationLevel.CollegeBachelor +
                          censusData Gender.Female EducationLevel.MastersDegree +
                          censusData Gender.Female EducationLevel.DoctoralDegree)
  male_avg > female_avg := by sorry

end probability_masters_degree_expected_value_bachelors_or_higher_male_education_greater_than_female_l1630_163022


namespace factorial_division_l1630_163076

theorem factorial_division :
  (9 : ℕ).factorial / (4 : ℕ).factorial = 15120 :=
by
  have h1 : (9 : ℕ).factorial = 362880 := by sorry
  sorry

end factorial_division_l1630_163076


namespace sum_of_squares_of_roots_l1630_163033

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 + 5 * x₁ - 4 = 0) → 
  (6 * x₂^2 + 5 * x₂ - 4 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 73/36) := by
sorry

end sum_of_squares_of_roots_l1630_163033


namespace geometric_sequence_third_term_l1630_163026

/-- Given a geometric sequence with first term 1000 and sixth term 125,
    prove that the third term is equal to 301. -/
theorem geometric_sequence_third_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 1000 →
    a * r^5 = 125 →
    a * r^2 = 301 :=
by sorry

end geometric_sequence_third_term_l1630_163026


namespace smallest_block_size_l1630_163036

/-- Given a rectangular block of dimensions l × m × n formed by N unit cubes,
    where (l - 1) × (m - 1) × (n - 1) = 143, the smallest possible value of N is 336. -/
theorem smallest_block_size (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 143) :
  ∃ (N : ℕ), N = l * m * n ∧ N = 336 ∧ ∀ (l' m' n' : ℕ), 
    ((l' - 1) * (m' - 1) * (n' - 1) = 143) → l' * m' * n' ≥ N :=
by sorry

end smallest_block_size_l1630_163036


namespace quadratic_roots_expression_l1630_163083

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  (-2 * x₁^2 + x₁ + 5 = 0) → 
  (-2 * x₂^2 + x₂ + 5 = 0) → 
  x₁^2 * x₂ + x₁ * x₂^2 = -5/4 := by
  sorry

end quadratic_roots_expression_l1630_163083


namespace correct_sum_after_misreading_l1630_163043

/-- Given a three-digit number ABC where C was misread as 6 instead of 9,
    and the sum of AB6 and 57 is 823, prove that the correct sum of ABC and 57 is 826 -/
theorem correct_sum_after_misreading (A B : Nat) : 
  (100 * A + 10 * B + 6 + 57 = 823) → 
  (100 * A + 10 * B + 9 + 57 = 826) :=
by sorry

end correct_sum_after_misreading_l1630_163043


namespace checkerboard_valid_squares_l1630_163079

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- The checkerboard -/
def Checkerboard : Type := Array (Array Bool)

/-- Creates a 10x10 checkerboard with alternating black and white squares -/
def create_checkerboard : Checkerboard := sorry

/-- Checks if a square contains at least 8 black squares -/
def has_at_least_8_black (board : Checkerboard) (square : Square) : Bool := sorry

/-- Counts the number of valid squares on the board -/
def count_valid_squares (board : Checkerboard) : Nat := sorry

theorem checkerboard_valid_squares :
  let board := create_checkerboard
  count_valid_squares board = 140 := by sorry

end checkerboard_valid_squares_l1630_163079


namespace select_students_count_l1630_163087

/-- The number of ways to select 3 students from 5 boys and 3 girls, including both genders -/
def select_students : ℕ :=
  Nat.choose 3 1 * Nat.choose 5 2 + Nat.choose 3 2 * Nat.choose 5 1

/-- Theorem stating that the number of ways to select the students is 45 -/
theorem select_students_count : select_students = 45 := by
  sorry

#eval select_students

end select_students_count_l1630_163087


namespace modulus_of_complex_number_l1630_163021

open Complex

theorem modulus_of_complex_number : ∃ z : ℂ, z = (2 - I)^2 / I ∧ Complex.abs z = 5 := by
  sorry

end modulus_of_complex_number_l1630_163021


namespace product_of_base8_digits_8675_l1630_163080

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8675 (base 10) is 0 -/
theorem product_of_base8_digits_8675 :
  productOfList (toBase8 8675) = 0 := by
  sorry

end product_of_base8_digits_8675_l1630_163080


namespace coin_regrouping_l1630_163066

/-- The total number of coins remains the same after regrouping -/
theorem coin_regrouping (x : ℕ) : 
  (12 + 17 + 23 + 8 : ℕ) = 60 ∧ 
  x > 0 ∧
  60 % x = 0 →
  60 = 60 := by
  sorry

end coin_regrouping_l1630_163066


namespace circle_and_intersection_conditions_l1630_163085

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_and_intersection_conditions (m : ℝ) :
  (∀ x y, circle_equation x y m → m < 5) ∧
  (∃ x1 y1 x2 y2, 
    circle_equation x1 y1 m ∧
    circle_equation x2 y2 m ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 ∧
    perpendicular x1 y1 x2 y2 →
    m = 8/5) :=
by sorry

end circle_and_intersection_conditions_l1630_163085


namespace track_walking_speed_l1630_163041

theorem track_walking_speed 
  (track_width : ℝ) 
  (time_difference : ℝ) 
  (inner_length : ℝ → ℝ → ℝ) 
  (outer_length : ℝ → ℝ → ℝ) :
  track_width = 6 →
  time_difference = 48 →
  (∀ a b, inner_length a b = 2 * a + 2 * π * b) →
  (∀ a b, outer_length a b = 2 * a + 2 * π * (b + track_width)) →
  ∃ s a b, 
    outer_length a b / s = inner_length a b / s + time_difference ∧
    s = π / 4 :=
by sorry

end track_walking_speed_l1630_163041


namespace baymax_testing_system_l1630_163003

theorem baymax_testing_system (x y : ℕ) : 
  (200 * y = x + 18 ∧ 180 * y = x - 42) ↔ 
  (∀ (z : ℕ), z = 200 → z * y = x + 18) ∧ 
  (∀ (w : ℕ), w = 180 → w * y + 42 = x) :=
sorry

end baymax_testing_system_l1630_163003


namespace count_different_numerators_l1630_163053

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a recurring decimal in the form 0.ẋyż -/
structure RecurringDecimal where
  x : Digit
  y : Digit
  z : Digit

/-- Converts a RecurringDecimal to a rational number -/
def toRational (d : RecurringDecimal) : ℚ :=
  (d.x.val * 100 + d.y.val * 10 + d.z.val : ℕ) / 999

/-- The set of all possible RecurringDecimals -/
def allRecurringDecimals : Finset RecurringDecimal :=
  sorry

/-- The set of all possible numerators when converting RecurringDecimals to lowest terms -/
def allNumerators : Finset ℕ :=
  sorry

theorem count_different_numerators :
  Finset.card allNumerators = 660 :=
sorry

end count_different_numerators_l1630_163053


namespace stating_cube_coloring_theorem_l1630_163023

/-- Represents the number of faces on a cube -/
def num_faces : ℕ := 6

/-- Represents the number of available colors -/
def num_colors : ℕ := 8

/-- Represents the number of rotational symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- 
Calculates the number of distinguishable ways to paint a cube
given the number of faces, colors, and rotational symmetries
-/
def distinguishable_colorings (faces : ℕ) (colors : ℕ) (symmetries : ℕ) : ℕ :=
  faces * (Nat.factorial (colors - 1)) / symmetries

/-- 
Theorem stating that the number of distinguishable ways to paint a cube
with 8 different colors, where each face is painted a different color, is 1260
-/
theorem cube_coloring_theorem : 
  distinguishable_colorings num_faces num_colors cube_symmetries = 1260 := by
  sorry

end stating_cube_coloring_theorem_l1630_163023


namespace point_on_graph_l1630_163098

/-- The linear function f(x) = 3x + 1 -/
def f (x : ℝ) : ℝ := 3 * x + 1

/-- The point (2, 7) -/
def point : ℝ × ℝ := (2, 7)

/-- Theorem: The point (2, 7) lies on the graph of f(x) = 3x + 1 -/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end point_on_graph_l1630_163098


namespace average_temperature_l1630_163012

def temperatures : List ℝ := [55, 59, 60, 57, 64]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 59.0 := by
  sorry

end average_temperature_l1630_163012


namespace smallest_four_digit_multiple_of_three_l1630_163007

theorem smallest_four_digit_multiple_of_three :
  ∃ n : ℕ, n = 1002 ∧ 
    (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 3 = 0 → n ≤ m) ∧
    1000 ≤ n ∧ n < 10000 ∧ n % 3 = 0 :=
by sorry

end smallest_four_digit_multiple_of_three_l1630_163007


namespace aron_cleaning_time_l1630_163075

/-- Calculates the total cleaning time per week for Aron -/
def total_cleaning_time (vacuum_time : ℕ) (vacuum_freq : ℕ) (dust_time : ℕ) (dust_freq : ℕ) : ℕ :=
  vacuum_time * vacuum_freq + dust_time * dust_freq

/-- Proves that Aron spends 130 minutes per week cleaning -/
theorem aron_cleaning_time :
  total_cleaning_time 30 3 20 2 = 130 := by
  sorry

end aron_cleaning_time_l1630_163075


namespace katrina_cookies_l1630_163013

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of cookies Katrina sold in the morning -/
def morning_sales : ℕ := 3 * dozen

/-- The number of cookies Katrina sold during lunch rush -/
def lunch_sales : ℕ := 57

/-- The number of cookies Katrina sold in the afternoon -/
def afternoon_sales : ℕ := 16

/-- The number of cookies Katrina has left to take home -/
def cookies_left : ℕ := 11

/-- The total number of cookies Katrina had initially -/
def initial_cookies : ℕ := morning_sales + lunch_sales + afternoon_sales + cookies_left

theorem katrina_cookies : initial_cookies = 120 := by sorry

end katrina_cookies_l1630_163013


namespace cassie_water_refills_l1630_163028

/-- Represents the number of cups of water Cassie aims to drink daily -/
def daily_cups : ℕ := 12

/-- Represents the capacity of Cassie's water bottle in ounces -/
def bottle_capacity : ℕ := 16

/-- Represents the number of ounces in a cup -/
def ounces_per_cup : ℕ := 8

/-- Represents the number of times Cassie needs to refill her water bottle -/
def refills : ℕ := 6

/-- Theorem stating that Cassie needs to refill her water bottle 6 times
    to meet her daily water intake goal -/
theorem cassie_water_refills :
  (daily_cups * ounces_per_cup) / bottle_capacity = refills :=
by sorry

end cassie_water_refills_l1630_163028


namespace polynomial_without_cubic_and_linear_terms_l1630_163072

theorem polynomial_without_cubic_and_linear_terms 
  (a b : ℝ) 
  (h1 : a - 3 = 0)  -- Coefficient of x^3 is zero
  (h2 : 4 - b = 0)  -- Coefficient of x is zero
  : (a - b) ^ 2023 = -1 := by
  sorry

end polynomial_without_cubic_and_linear_terms_l1630_163072
