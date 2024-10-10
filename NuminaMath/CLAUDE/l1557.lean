import Mathlib

namespace exists_trapezoid_in_selected_vertices_l1557_155770

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of selected vertices from a regular polygon -/
def SelectedVertices (n k : ℕ) (p : RegularPolygon n) :=
  {s : Finset (Fin n) // s.card = k}

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
def IsTrapezoid (v1 v2 v3 v4 : ℝ × ℝ) : Prop :=
  (v1.1 - v2.1) * (v3.2 - v4.2) = (v1.2 - v2.2) * (v3.1 - v4.1) ∨
  (v1.1 - v3.1) * (v2.2 - v4.2) = (v1.2 - v3.2) * (v2.1 - v4.1) ∨
  (v1.1 - v4.1) * (v2.2 - v3.2) = (v1.2 - v4.2) * (v2.1 - v3.1)

/-- Main theorem: There exists a trapezoid among 64 selected vertices of a regular 1981-gon -/
theorem exists_trapezoid_in_selected_vertices 
  (p : RegularPolygon 1981) (s : SelectedVertices 1981 64 p) :
  ∃ (a b c d : Fin 1981), a ∈ s.val ∧ b ∈ s.val ∧ c ∈ s.val ∧ d ∈ s.val ∧
    IsTrapezoid (p.vertices a) (p.vertices b) (p.vertices c) (p.vertices d) :=
by
  sorry

end exists_trapezoid_in_selected_vertices_l1557_155770


namespace cubic_sum_over_product_l1557_155720

theorem cubic_sum_over_product (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x^3 + 1/(y+2016) = y^3 + 1/(z+2016) ∧ 
  y^3 + 1/(z+2016) = z^3 + 1/(x+2016) → 
  (x^3 + y^3 + z^3) / (x*y*z) = 3 := by
sorry

end cubic_sum_over_product_l1557_155720


namespace parallel_line_construction_l1557_155741

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Predicate to check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Theorem: Given a line and a point not on the line, 
    it's possible to construct a parallel line through the point 
    using only compass and straightedge -/
theorem parallel_line_construction 
  (l : Line) (A : Point) (h : ¬A.onLine l) :
  ∃ (l' : Line), A.onLine l' ∧ l.parallel l' :=
sorry

end parallel_line_construction_l1557_155741


namespace trip_distance_proof_l1557_155736

/-- Represents the total length of the trip in miles. -/
def total_distance : ℝ := 150

/-- Represents the distance traveled on battery power in miles. -/
def battery_distance : ℝ := 50

/-- Represents the fuel consumption rate in gallons per mile. -/
def fuel_rate : ℝ := 0.03

/-- Represents the average fuel efficiency for the entire trip in miles per gallon. -/
def avg_efficiency : ℝ := 50

theorem trip_distance_proof :
  (total_distance / (fuel_rate * (total_distance - battery_distance)) = avg_efficiency) ∧
  (total_distance > battery_distance) :=
by sorry

end trip_distance_proof_l1557_155736


namespace not_p_sufficient_not_necessary_for_not_p_and_q_l1557_155757

theorem not_p_sufficient_not_necessary_for_not_p_and_q
  (p q : Prop) :
  (∀ (h : ¬p), ¬(p ∧ q)) ∧
  ¬(∀ (h : ¬(p ∧ q)), ¬p) :=
by sorry

end not_p_sufficient_not_necessary_for_not_p_and_q_l1557_155757


namespace det_of_specific_matrix_l1557_155751

/-- The determinant of the matrix [5 -2; 4 3] is 23. -/
theorem det_of_specific_matrix :
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 4, 3]
  Matrix.det M = 23 := by
  sorry

end det_of_specific_matrix_l1557_155751


namespace at_least_one_negative_l1557_155710

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1)
  (sum_cd : c + d = 1)
  (product_sum : a * c + b * d > 1) :
  ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by sorry

end at_least_one_negative_l1557_155710


namespace trig_identity_l1557_155786

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end trig_identity_l1557_155786


namespace exactly_one_integer_n_for_n_plus_i_sixth_power_integer_l1557_155761

theorem exactly_one_integer_n_for_n_plus_i_sixth_power_integer :
  ∃! (n : ℤ), ∃ (m : ℤ), (n + Complex.I) ^ 6 = m := by sorry

end exactly_one_integer_n_for_n_plus_i_sixth_power_integer_l1557_155761


namespace polynomial_division_remainder_l1557_155709

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ,
  x^50 + x^40 + x^30 + x^20 + x^10 + 1 = 
  (x^5 + x^4 + x^3 + x^2 + x + 1) * q + (-2 : Polynomial ℤ) := by
  sorry

end polynomial_division_remainder_l1557_155709


namespace beef_weight_is_fifteen_l1557_155745

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The weight of each steak in ounces -/
def steak_weight : ℕ := 12

/-- The number of steaks Matt gets from the beef -/
def number_of_steaks : ℕ := 20

/-- The total weight of beef in pounds -/
def total_beef_weight : ℚ :=
  (steak_weight * number_of_steaks : ℚ) / ounces_per_pound

theorem beef_weight_is_fifteen :
  total_beef_weight = 15 := by sorry

end beef_weight_is_fifteen_l1557_155745


namespace sharon_coffee_pods_l1557_155700

/-- Calculates the number of pods in a box given vacation details and spending -/
def pods_per_box (vacation_days : ℕ) (daily_pods : ℕ) (total_spent : ℕ) (price_per_box : ℕ) : ℕ :=
  let total_pods := vacation_days * daily_pods
  let boxes_bought := total_spent / price_per_box
  total_pods / boxes_bought

/-- Proves that the number of pods in a box is 30 given the specific vacation details -/
theorem sharon_coffee_pods :
  pods_per_box 40 3 32 8 = 30 := by
  sorry

end sharon_coffee_pods_l1557_155700


namespace arithmetic_sequence_ratio_l1557_155747

/-- Two arithmetic sequences and their sum sequences -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_ratio : 
  ∀ (a b : ℕ → ℚ) (S T : ℕ → ℚ),
  arithmetic_sequence a →
  arithmetic_sequence b →
  (∀ n : ℕ, S n = (n : ℚ) * a n - (n - 1 : ℚ) / 2 * (a n - a 1)) →
  (∀ n : ℕ, T n = (n : ℚ) * b n - (n - 1 : ℚ) / 2 * (b n - b 1)) →
  (∀ n : ℕ, n > 0 → S n / T n = (5 * n - 3 : ℚ) / (2 * n + 1)) →
  a 20 / b 7 = 64 / 9 := by
sorry

end arithmetic_sequence_ratio_l1557_155747


namespace characterize_valid_functions_l1557_155703

def is_valid_function (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a + f b + f (f c) = 0 →
    f a ^ 3 + b * (f b) ^ 2 + c ^ 2 * f c = 3 * a * b * c

theorem characterize_valid_functions :
  ∀ f : ℝ → ℝ, is_valid_function f →
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end characterize_valid_functions_l1557_155703


namespace work_day_meetings_percentage_l1557_155799

/-- Proves that given a 10-hour work day and two meetings, where the first meeting is 60 minutes long
    and the second is three times as long, the percentage of the work day spent in meetings is 40%. -/
theorem work_day_meetings_percentage (work_day_hours : ℕ) (first_meeting_minutes : ℕ) :
  work_day_hours = 10 →
  first_meeting_minutes = 60 →
  let work_day_minutes : ℕ := work_day_hours * 60
  let second_meeting_minutes : ℕ := 3 * first_meeting_minutes
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  let meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100
  meeting_percentage = 40 := by
  sorry


end work_day_meetings_percentage_l1557_155799


namespace largest_subset_size_l1557_155771

/-- A function that returns the size of the largest subset of {1,2,...,n} where no two elements differ by 5 or 8 -/
def maxSubsetSize (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the largest subset of {1,2,3,...,2023} where no two elements differ by 5 or 8 has 780 elements -/
theorem largest_subset_size :
  maxSubsetSize 2023 = 780 :=
sorry

end largest_subset_size_l1557_155771


namespace sector_central_angle_l1557_155701

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 6) (h2 : s.area = 2) :
  ∃ θ : ℝ, (θ = 1 ∨ θ = 4) ∧ 
  (∃ r : ℝ, r > 0 ∧ θ * r + 2 * r = s.perimeter ∧ 1/2 * r^2 * θ = s.area) :=
sorry

end sector_central_angle_l1557_155701


namespace solution_set_equality_l1557_155716

/-- The set of real numbers a for which the solution set of |x - 2| < a is a subset of (-2, 1] -/
def A : Set ℝ := {a | ∀ x, |x - 2| < a → -2 < x ∧ x ≤ 1}

/-- The theorem stating that A is equal to (-∞, 0] -/
theorem solution_set_equality : A = Set.Iic 0 := by sorry

end solution_set_equality_l1557_155716


namespace square_ending_in_five_l1557_155729

theorem square_ending_in_five (a : ℕ) :
  let n : ℕ := 10 * a + 5
  ∃ (m : ℕ), n^2 = m^2 → a % 10 = 2 :=
by sorry

end square_ending_in_five_l1557_155729


namespace girls_in_classroom_l1557_155704

theorem girls_in_classroom (boys : ℕ) (ratio : ℚ) (girls : ℕ) : 
  boys = 20 → ratio = 1/2 → (girls : ℚ) / boys = ratio → girls = 10 := by
  sorry

end girls_in_classroom_l1557_155704


namespace carlos_class_size_l1557_155719

theorem carlos_class_size (n : ℕ) (carlos : ℕ) :
  (carlos = 75) →
  (n - carlos = 74) →
  (carlos - 1 = 74) →
  n = 149 := by
  sorry

end carlos_class_size_l1557_155719


namespace initial_flow_rate_is_two_l1557_155754

/-- Represents the flow rate of cleaner through a pipe over time -/
structure FlowRate where
  initial : ℝ
  after15min : ℝ
  after25min : ℝ

/-- Calculates the total amount of cleaner used given a flow rate profile -/
def totalCleanerUsed (flow : FlowRate) : ℝ :=
  15 * flow.initial + 10 * flow.after15min + 5 * flow.after25min

/-- Theorem stating that the initial flow rate is 2 ounces per minute -/
theorem initial_flow_rate_is_two :
  ∃ (flow : FlowRate),
    flow.after15min = 3 ∧
    flow.after25min = 4 ∧
    totalCleanerUsed flow = 80 ∧
    flow.initial = 2 := by
  sorry

end initial_flow_rate_is_two_l1557_155754


namespace sum_of_digits_after_addition_l1557_155781

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Number of carries in addition -/
def carries_in_addition (a b : ℕ) : ℕ := sorry

theorem sum_of_digits_after_addition (A B : ℕ) 
  (hA : A > 0) 
  (hB : B > 0) 
  (hSumA : sum_of_digits A = 19) 
  (hSumB : sum_of_digits B = 20) 
  (hCarries : carries_in_addition A B = 2) : 
  sum_of_digits (A + B) = 21 := by sorry

end sum_of_digits_after_addition_l1557_155781


namespace geometric_series_ratio_l1557_155740

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
sorry

end geometric_series_ratio_l1557_155740


namespace k_range_for_three_roots_l1557_155763

/-- The cubic function f(x) = x³ - x² - x + k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x + k

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

/-- Theorem stating the range of k for f(x) to have exactly three roots -/
theorem k_range_for_three_roots :
  ∀ k : ℝ, (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f k x₁ = 0 ∧ f k x₂ = 0 ∧ f k x₃ = 0) ↔ 
  -5/27 < k ∧ k < 1 :=
sorry

end k_range_for_three_roots_l1557_155763


namespace day_300_is_tuesday_l1557_155784

/-- If the 26th day of a 366-day year falls on a Monday, then the 300th day of that year falls on a Tuesday. -/
theorem day_300_is_tuesday (year_length : ℕ) (day_26_weekday : ℕ) :
  year_length = 366 →
  day_26_weekday = 1 →
  (300 - 26) % 7 + day_26_weekday ≡ 2 [MOD 7] :=
by sorry

end day_300_is_tuesday_l1557_155784


namespace good_numbers_characterization_l1557_155727

/-- A number n > 3 is 'good' if the set of weights {1, 2, 3, ..., n} can be divided into three piles of equal mass -/
def is_good (n : ℕ) : Prop :=
  n > 3 ∧ ∃ (a b c : Finset ℕ), a ∪ b ∪ c = Finset.range n ∧ 
    a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    a.sum id = b.sum id ∧ b.sum id = c.sum id

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem good_numbers_characterization (n : ℕ) :
  is_good n ↔ (∃ k : ℕ, k ≥ 1 ∧ (n = 3 * k ∨ n = 3 * k + 2)) :=
sorry

end good_numbers_characterization_l1557_155727


namespace sum_positive_not_sufficient_nor_necessary_for_product_positive_l1557_155714

theorem sum_positive_not_sufficient_nor_necessary_for_product_positive :
  ∃ (a b : ℝ), (a + b > 0 ∧ a * b ≤ 0) ∧ ∃ (c d : ℝ), (c + d ≤ 0 ∧ c * d > 0) := by
  sorry

end sum_positive_not_sufficient_nor_necessary_for_product_positive_l1557_155714


namespace expected_pairs_value_l1557_155702

/-- The number of boys in the lineup -/
def num_boys : ℕ := 9

/-- The number of girls in the lineup -/
def num_girls : ℕ := 15

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the lineup -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl or girl-boy pair at any given adjacent position -/
def pair_probability : ℚ := 
  (num_boys * num_girls + num_girls * num_boys) / (total_people * (total_people - 1))

/-- The expected number of boy-girl or girl-boy pairs in a random permutation -/
def expected_pairs : ℚ := num_pairs * pair_probability

theorem expected_pairs_value : expected_pairs = 3105 / 276 := by sorry

end expected_pairs_value_l1557_155702


namespace solve_equation_l1557_155735

theorem solve_equation (x : ℚ) : (3 * x - 7) / 4 = 15 → x = 67 / 3 := by
  sorry

end solve_equation_l1557_155735


namespace hallie_paintings_sold_l1557_155749

/-- The number of paintings Hallie sold -/
def paintings_sold (prize : ℕ) (painting_price : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings - prize) / painting_price

theorem hallie_paintings_sold :
  paintings_sold 150 50 300 = 3 := by
  sorry

end hallie_paintings_sold_l1557_155749


namespace debt_payment_average_l1557_155706

theorem debt_payment_average : 
  let total_payments : ℕ := 52
  let first_payment_count : ℕ := 25
  let first_payment_amount : ℚ := 500
  let additional_amount : ℚ := 100
  let second_payment_count : ℕ := total_payments - first_payment_count
  let second_payment_amount : ℚ := first_payment_amount + additional_amount
  let total_amount : ℚ := first_payment_count * first_payment_amount + 
                          second_payment_count * second_payment_amount
  let average_payment : ℚ := total_amount / total_payments
  average_payment = 551.92 := by
sorry

end debt_payment_average_l1557_155706


namespace isosceles_right_triangle_area_and_perimeter_l1557_155731

/-- An isosceles right triangle with hypotenuse 6√2 -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  is_isosceles_right : hypotenuse = 6 * Real.sqrt 2

theorem isosceles_right_triangle_area_and_perimeter 
  (t : IsoscelesRightTriangle) : 
  ∃ (leg : ℝ), 
    leg^2 + leg^2 = t.hypotenuse^2 ∧ 
    (1/2 * leg * leg = 18) ∧ 
    (leg + leg + t.hypotenuse = 12 + 6 * Real.sqrt 2) := by
  sorry

#check isosceles_right_triangle_area_and_perimeter

end isosceles_right_triangle_area_and_perimeter_l1557_155731


namespace existence_of_monochromatic_right_angled_pentagon_l1557_155732

-- Define a color type
inductive Color
| Red
| Yellow

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a convex pentagon
def ConvexPentagon (p₁ p₂ p₃ p₄ p₅ : Point) : Prop := sorry

-- Define a right angle
def RightAngle (p₁ p₂ p₃ : Point) : Prop := sorry

-- Define the theorem
theorem existence_of_monochromatic_right_angled_pentagon :
  ∃ (p₁ p₂ p₃ p₄ p₅ : Point),
    ConvexPentagon p₁ p₂ p₃ p₄ p₅ ∧
    RightAngle p₁ p₂ p₃ ∧
    RightAngle p₂ p₃ p₄ ∧
    RightAngle p₃ p₄ p₅ ∧
    ((coloring p₁ = Color.Red ∧ coloring p₂ = Color.Red ∧ coloring p₃ = Color.Red ∧ 
      coloring p₄ = Color.Red ∧ coloring p₅ = Color.Red) ∨
     (coloring p₁ = Color.Yellow ∧ coloring p₂ = Color.Yellow ∧ coloring p₃ = Color.Yellow ∧ 
      coloring p₄ = Color.Yellow ∧ coloring p₅ = Color.Yellow)) :=
by sorry


end existence_of_monochromatic_right_angled_pentagon_l1557_155732


namespace remainder_after_adding_4500_l1557_155764

theorem remainder_after_adding_4500 (n : ℤ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := by
  sorry

end remainder_after_adding_4500_l1557_155764


namespace equal_debt_after_calculated_days_l1557_155752

/-- The number of days until Darren and Fergie owe the same amount -/
def days_until_equal_debt : ℝ := 53.75

/-- Darren's initial borrowed amount -/
def darren_initial_borrowed : ℝ := 200

/-- Fergie's initial borrowed amount -/
def fergie_initial_borrowed : ℝ := 300

/-- Darren's initial daily interest rate -/
def darren_initial_rate : ℝ := 0.08

/-- Darren's reduced daily interest rate after 10 days -/
def darren_reduced_rate : ℝ := 0.06

/-- Fergie's daily interest rate -/
def fergie_rate : ℝ := 0.04

/-- The number of days after which Darren's interest rate changes -/
def rate_change_days : ℝ := 10

/-- Theorem stating that Darren and Fergie owe the same amount after the calculated number of days -/
theorem equal_debt_after_calculated_days :
  let darren_debt := if days_until_equal_debt ≤ rate_change_days
    then darren_initial_borrowed * (1 + darren_initial_rate * days_until_equal_debt)
    else darren_initial_borrowed * (1 + darren_initial_rate * rate_change_days) *
      (1 + darren_reduced_rate * (days_until_equal_debt - rate_change_days))
  let fergie_debt := fergie_initial_borrowed * (1 + fergie_rate * days_until_equal_debt)
  darren_debt = fergie_debt := by sorry


end equal_debt_after_calculated_days_l1557_155752


namespace fraction_equality_l1557_155790

theorem fraction_equality : (18 * 3 + 12) / (6 - 4) = 33 := by
  sorry

end fraction_equality_l1557_155790


namespace triangle_theorem_l1557_155737

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A ∧
  Real.cos t.B = 3/4 ∧
  t.c = 5

-- Theorem to prove
theorem triangle_theorem (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = 2 * t.B ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (15/4) * Real.sqrt 7 := by
  sorry


end triangle_theorem_l1557_155737


namespace binomial_coefficient_problem_l1557_155721

theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 7 6 : ℝ) * a = 7 → a = 1 := by
  sorry

end binomial_coefficient_problem_l1557_155721


namespace tv_watching_time_conversion_l1557_155734

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of hours Logan watched TV
def hours_watched : ℕ := 5

-- Theorem to prove
theorem tv_watching_time_conversion :
  hours_watched * minutes_per_hour = 300 := by
  sorry

end tv_watching_time_conversion_l1557_155734


namespace spearman_correlation_approx_l1557_155778

def scores_A : List ℝ := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
def scores_B : List ℝ := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70]

def spearman_rank_correlation (x y : List ℝ) : ℝ :=
  sorry

theorem spearman_correlation_approx :
  ∃ ε > 0, ε < 0.01 ∧ |spearman_rank_correlation scores_A scores_B - 0.64| < ε :=
by sorry

end spearman_correlation_approx_l1557_155778


namespace beth_twice_sister_age_l1557_155779

/-- 
Given:
- Beth is currently 18 years old
- Beth's sister is currently 5 years old

Prove that the number of years until Beth is twice her sister's age is 8.
-/
theorem beth_twice_sister_age (beth_age : ℕ) (sister_age : ℕ) : 
  beth_age = 18 → sister_age = 5 → (beth_age + 8 = 2 * (sister_age + 8)) := by
  sorry

end beth_twice_sister_age_l1557_155779


namespace g_of_neg_four_l1557_155794

/-- Given a function g(x) = 5x - 2, prove that g(-4) = -22 -/
theorem g_of_neg_four (g : ℝ → ℝ) (h : ∀ x, g x = 5 * x - 2) : g (-4) = -22 := by
  sorry

end g_of_neg_four_l1557_155794


namespace two_thirds_squared_l1557_155795

theorem two_thirds_squared : (2 / 3 : ℚ) ^ 2 = 4 / 9 := by
  sorry

end two_thirds_squared_l1557_155795


namespace heather_bicycle_speed_l1557_155722

/-- Heather's bicycle problem -/
theorem heather_bicycle_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 40 ∧ time = 5 ∧ speed = distance / time → speed = 8 := by
  sorry

end heather_bicycle_speed_l1557_155722


namespace linear_function_inequality_solution_l1557_155713

/-- Given a linear function y = kx + b, prove that under certain conditions, 
    the solution set of an inequality is x < 1 -/
theorem linear_function_inequality_solution 
  (k b n : ℝ) 
  (h_k : k ≠ 0)
  (h_n : n > 2)
  (h_y_neg1 : k * (-1) + b = n)
  (h_y_1 : k * 1 + b = 2) :
  {x : ℝ | (k - 2) * x + b > 0} = {x : ℝ | x < 1} := by
sorry

end linear_function_inequality_solution_l1557_155713


namespace largest_digit_change_corrects_addition_l1557_155726

def original_sum : ℕ := 735 + 468 + 281
def given_result : ℕ := 1584
def correct_first_addend : ℕ := 835

theorem largest_digit_change_corrects_addition :
  (original_sum ≠ given_result) →
  (correct_first_addend + 468 + 281 = given_result) →
  ∀ (d : ℕ), d ≤ 9 →
    (d > 7 → 
      ¬∃ (a b c : ℕ), a ≤ 999 ∧ b ≤ 999 ∧ c ≤ 999 ∧
        (a + b + c = given_result) ∧
        (a = 735 + d * 100 - 700 ∨
         b = 468 + d * 100 - 400 ∨
         c = 281 + d * 100 - 200)) :=
sorry

end largest_digit_change_corrects_addition_l1557_155726


namespace number_problem_l1557_155739

theorem number_problem (x : ℝ) : 
  (0.25 * x = 0.20 * 650 + 190) → x = 1280 := by
sorry

end number_problem_l1557_155739


namespace arithmetic_is_linear_geometric_is_exponential_l1557_155765

/-- Definition of an arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Definition of a geometric sequence -/
def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- Theorem: The n-th term of an arithmetic sequence is a linear function of n -/
theorem arithmetic_is_linear (a₁ d : ℝ) :
  ∃ m b : ℝ, ∀ n : ℕ, arithmetic_sequence a₁ d n = m * n + b :=
sorry

/-- Theorem: The n-th term of a geometric sequence is an exponential function of n -/
theorem geometric_is_exponential (a₁ r : ℝ) :
  ∃ A B : ℝ, ∀ n : ℕ, geometric_sequence a₁ r n = A * B^n :=
sorry

end arithmetic_is_linear_geometric_is_exponential_l1557_155765


namespace train_passing_time_l1557_155769

/-- Time for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 110 →
  train_speed = 90 * (1000 / 3600) →
  man_speed = 9 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 4 := by
  sorry

#check train_passing_time

end train_passing_time_l1557_155769


namespace minimum_students_l1557_155782

theorem minimum_students (b g : ℕ) : 
  (3 * b = 5 * g) →  -- Same number of boys and girls passed
  (b ≥ 5) →          -- At least 5 boys (for 3/5 to be meaningful)
  (g ≥ 6) →          -- At least 6 girls (for 5/6 to be meaningful)
  (∀ b' g', (3 * b' = 5 * g') → (b' ≥ 5) → (g' ≥ 6) → (b' + g' ≥ b + g)) →
  b + g = 43 :=
by sorry

#check minimum_students

end minimum_students_l1557_155782


namespace pr_qs_ratio_l1557_155744

-- Define the points and distances
def P : ℝ := 0
def Q : ℝ := 3
def R : ℝ := 10
def S : ℝ := 18

-- State the theorem
theorem pr_qs_ratio :
  (R - P) / (S - Q) = 2 / 3 := by sorry

end pr_qs_ratio_l1557_155744


namespace min_value_theorem_l1557_155759

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : 2*x + 2*y + 3*z = 3) : 
  (2*(x + y)) / (x*y*z) ≥ 14.2222 := by
sorry

#eval (8 : ℚ) / (9 : ℚ) * 16

end min_value_theorem_l1557_155759


namespace tv_selection_combinations_l1557_155785

def num_type_a : ℕ := 4
def num_type_b : ℕ := 5
def num_to_choose : ℕ := 3

def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem tv_selection_combinations : 
  (combinations num_type_a 2 * combinations num_type_b 1) + 
  (combinations num_type_a 1 * combinations num_type_b 2) = 70 := by
  sorry

end tv_selection_combinations_l1557_155785


namespace sin_75_cos_15_minus_1_l1557_155708

theorem sin_75_cos_15_minus_1 : 
  2 * Real.sin (75 * π / 180) * Real.cos (15 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end sin_75_cos_15_minus_1_l1557_155708


namespace cosine_derivative_at_pi_sixth_l1557_155767

theorem cosine_derivative_at_pi_sixth :
  let f : ℝ → ℝ := λ x ↦ Real.cos x
  (deriv f) (π / 6) = - (1 / 2) := by
  sorry

end cosine_derivative_at_pi_sixth_l1557_155767


namespace square_of_difference_l1557_155760

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end square_of_difference_l1557_155760


namespace fraction_division_simplify_fraction_division_l1557_155768

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (5 : ℚ) / 6 / ((7 : ℚ) / 12) = 10 / 7 :=
by sorry

end fraction_division_simplify_fraction_division_l1557_155768


namespace efficiency_ratio_l1557_155712

-- Define the efficiencies of workers a, b, c, and d
def efficiency_a : ℚ := 1 / 18
def efficiency_b : ℚ := 1 / 36
def efficiency_c : ℚ := 1 / 20
def efficiency_d : ℚ := 1 / 30

-- Theorem statement
theorem efficiency_ratio :
  -- a and b together have the same efficiency as c and d together
  efficiency_a + efficiency_b = efficiency_c + efficiency_d →
  -- The ratio of a's efficiency to b's efficiency is 2:1
  efficiency_a / efficiency_b = 2 := by
sorry

end efficiency_ratio_l1557_155712


namespace greatest_integer_of_a_l1557_155775

def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => 1994^2 / (a n + 1)

theorem greatest_integer_of_a (n : ℕ) (h : n ≤ 998) :
  ⌊a n⌋ = 1994 - n := by sorry

end greatest_integer_of_a_l1557_155775


namespace parabola_equation_and_chord_length_l1557_155742

/-- Parabola with vertex at origin, focus on positive y-axis, and focus-directrix distance 2 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  focus_on_y_axis : ∃ y > 0, equation 0 y
  focus_directrix_distance : ℝ

/-- Line defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y = m * x + b

theorem parabola_equation_and_chord_length 
  (p : Parabola) 
  (h_dist : p.focus_directrix_distance = 2) 
  (l : Line) 
  (h_line : l.m = 2 ∧ l.b = 1) :
  (∀ x y, p.equation x y ↔ x^2 = 4*y) ∧
  (∃ A B : ℝ × ℝ, 
    p.equation A.1 A.2 ∧ 
    p.equation B.1 B.2 ∧ 
    l.equation A.1 A.2 ∧ 
    l.equation B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20) := by
  sorry

end parabola_equation_and_chord_length_l1557_155742


namespace g_expression_l1557_155718

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the relationship between f and g
def g_relation (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_expression (g : ℝ → ℝ) (h : g_relation g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end g_expression_l1557_155718


namespace quadratic_through_origin_l1557_155788

/-- If the graph of the quadratic function y = mx^2 + x + m(m-3) passes through the origin, then m = 3 -/
theorem quadratic_through_origin (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2 + x + m * (m - 3)) → 
  (0 = m * 0^2 + 0 + m * (m - 3)) → 
  m = 3 := by sorry

end quadratic_through_origin_l1557_155788


namespace max_large_chips_l1557_155723

theorem max_large_chips (total : ℕ) (small large : ℕ → ℕ) (p : ℕ → ℕ) :
  total = 70 →
  (∀ n, total = small n + large n) →
  (∀ n, Prime (p n)) →
  (∀ n, small n = large n + p n) →
  (∀ n, large n ≤ 34) ∧ (∃ n, large n = 34) :=
by sorry

end max_large_chips_l1557_155723


namespace square_root_one_ninth_l1557_155753

theorem square_root_one_ninth : Real.sqrt (1/9) = 1/3 ∨ Real.sqrt (1/9) = -1/3 := by
  sorry

end square_root_one_ninth_l1557_155753


namespace sine_function_expression_l1557_155746

theorem sine_function_expression 
  (y : ℝ → ℝ) 
  (A ω : ℝ) 
  (h1 : A > 0)
  (h2 : ω > 0)
  (h3 : ∀ x, y x = A * Real.sin (ω * x + φ))
  (h4 : A = 2)
  (h5 : 2 * Real.pi / ω = Real.pi / 2)
  (h6 : φ = -3) :
  ∀ x, y x = 2 * Real.sin (4 * x - 3) := by
sorry

end sine_function_expression_l1557_155746


namespace trigonometric_simplification_l1557_155792

theorem trigonometric_simplification (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) /
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) =
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by sorry

end trigonometric_simplification_l1557_155792


namespace fixed_point_sum_l1557_155724

theorem fixed_point_sum (a : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^(x - 1) + 2 : ℝ → ℝ) m = n → m + n = 4 := by
  sorry

end fixed_point_sum_l1557_155724


namespace square_area_perimeter_relationship_l1557_155766

/-- The relationship between the area and perimeter of a square is quadratic -/
theorem square_area_perimeter_relationship (x y : ℝ) (h_pos : x > 0) :
  ∃ k : ℝ, y = k * x^2 ↔ 
  (∃ a : ℝ, a > 0 ∧ x = 4 * a ∧ y = a^2) :=
by sorry

end square_area_perimeter_relationship_l1557_155766


namespace air_conditioner_installation_rates_l1557_155776

theorem air_conditioner_installation_rates 
  (total_A : ℕ) (total_B : ℕ) (diff : ℕ) :
  total_A = 66 →
  total_B = 60 →
  diff = 2 →
  ∃ (days : ℕ) (rate_A : ℕ) (rate_B : ℕ),
    rate_A = rate_B + diff ∧
    rate_A * days = total_A ∧
    rate_B * days = total_B ∧
    rate_A = 22 ∧
    rate_B = 20 :=
by sorry

end air_conditioner_installation_rates_l1557_155776


namespace key_chain_profit_percentage_l1557_155793

theorem key_chain_profit_percentage 
  (P : ℝ) 
  (h1 : P = 100) 
  (h2 : P - 50 = 0.5 * P) 
  (h3 : 70 < P) : 
  (P - 70) / P = 0.3 := by
  sorry

end key_chain_profit_percentage_l1557_155793


namespace evaluate_expression_l1557_155787

theorem evaluate_expression (x : ℝ) (hx : x ≠ 0) :
  (20 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (5/2) * x^2 := by
  sorry

end evaluate_expression_l1557_155787


namespace divisibility_of_sum_of_fifth_powers_l1557_155711

theorem divisibility_of_sum_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end divisibility_of_sum_of_fifth_powers_l1557_155711


namespace zhang_san_not_losing_probability_l1557_155783

theorem zhang_san_not_losing_probability
  (p_win : ℚ) (p_draw : ℚ)
  (h_win : p_win = 1 / 3)
  (h_draw : p_draw = 1 / 4) :
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l1557_155783


namespace black_ants_count_l1557_155730

theorem black_ants_count (total_ants red_ants : ℕ) 
  (h1 : total_ants = 900) 
  (h2 : red_ants = 413) : 
  total_ants - red_ants = 487 := by
  sorry

end black_ants_count_l1557_155730


namespace tabitha_money_proof_l1557_155791

def calculate_remaining_money (initial_amount : ℚ) (given_away : ℚ) (investment_percentage : ℚ) (num_items : ℕ) (item_cost : ℚ) : ℚ :=
  let remaining_after_giving := initial_amount - given_away
  let investment_amount := (investment_percentage / 100) * remaining_after_giving
  let remaining_after_investment := remaining_after_giving - investment_amount
  let spent_on_items := (num_items : ℚ) * item_cost
  remaining_after_investment - spent_on_items

theorem tabitha_money_proof :
  calculate_remaining_money 45 10 60 12 0.75 = 5 := by
  sorry

end tabitha_money_proof_l1557_155791


namespace inscribed_circle_area_ratio_l1557_155750

theorem inscribed_circle_area_ratio (a : ℝ) (ha : a > 0) :
  let square_area := a^2
  let circle_radius := a / 2
  let circle_area := π * circle_radius^2
  circle_area / square_area = π / 4 := by
sorry

end inscribed_circle_area_ratio_l1557_155750


namespace cement_truck_loads_l1557_155717

theorem cement_truck_loads (total material_truck_loads sand_truck_loads dirt_truck_loads : ℚ)
  (h1 : total = 0.67)
  (h2 : sand_truck_loads = 0.17)
  (h3 : dirt_truck_loads = 0.33)
  : total - (sand_truck_loads + dirt_truck_loads) = 0.17 := by
  sorry

end cement_truck_loads_l1557_155717


namespace rectangle_area_change_l1557_155796

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let new_length := 1.4 * L
  let new_width := 0.5 * W
  let original_area := L * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = -0.3 := by
sorry

end rectangle_area_change_l1557_155796


namespace pet_food_difference_l1557_155798

theorem pet_food_difference (dog_food cat_food : ℕ) 
  (h1 : dog_food = 600) (h2 : cat_food = 327) : 
  dog_food - cat_food = 273 := by
  sorry

end pet_food_difference_l1557_155798


namespace total_crayons_count_l1557_155707

/-- The number of children -/
def num_children : ℕ := 7

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 8

/-- The total number of crayons -/
def total_crayons : ℕ := num_children * crayons_per_child

theorem total_crayons_count : total_crayons = 56 := by
  sorry

end total_crayons_count_l1557_155707


namespace min_value_theorem_l1557_155738

theorem min_value_theorem (x : ℝ) (h : x > 2) : x + 4 / (x - 2) ≥ 6 ∧ (x + 4 / (x - 2) = 6 ↔ x = 4) := by
  sorry

end min_value_theorem_l1557_155738


namespace complex_pure_imaginary_condition_l1557_155755

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a^2 - 1) + (a - 1) * Complex.I) → a = -1 := by
  sorry

end complex_pure_imaginary_condition_l1557_155755


namespace triangles_in_regular_decagon_l1557_155748

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def triangles_in_decagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

theorem triangles_in_regular_decagon : 
  triangles_in_decagon = Nat.choose decagon_vertices triangle_vertices := by
  sorry

end triangles_in_regular_decagon_l1557_155748


namespace thirtieth_set_sum_l1557_155715

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of consecutive integers from a to b, inclusive -/
def sum_consecutive (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ :=
  let first := triangular_number (n - 1) + 1
  let last := triangular_number n
  sum_consecutive first last

theorem thirtieth_set_sum : S 30 = 13515 := by
  sorry

end thirtieth_set_sum_l1557_155715


namespace solve_system_for_q_l1557_155733

theorem solve_system_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
sorry

end solve_system_for_q_l1557_155733


namespace max_k_value_l1557_155728

theorem max_k_value (k : ℝ) : (∀ x : ℝ, Real.exp x ≥ k + x) → k ≤ 1 := by
  sorry

end max_k_value_l1557_155728


namespace sufficient_but_not_necessary_l1557_155772

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

-- Define the intersection condition
def intersection_condition (m : ℝ) : Prop := A m ∩ B = {4}

-- Define sufficiency
def is_sufficient (m : ℝ) : Prop := m = -2 → intersection_condition m

-- Define non-necessity
def is_not_necessary (m : ℝ) : Prop := ∃ x : ℝ, x ≠ -2 ∧ intersection_condition x

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, is_sufficient m) ∧ (∃ m : ℝ, is_not_necessary m) := by
  sorry

end sufficient_but_not_necessary_l1557_155772


namespace two_digit_swap_difference_l1557_155773

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : ℕ
  units : ℕ
  tens_valid : tens < 10
  units_valid : units < 10

/-- Calculates the value of a two-digit number -/
def value (n : TwoDigitNumber) : ℕ := 10 * n.tens + n.units

/-- Swaps the digits of a two-digit number -/
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber := {
  tens := n.units,
  units := n.tens,
  tens_valid := n.units_valid,
  units_valid := n.tens_valid
}

/-- 
Theorem: The difference between a two-digit number with its digits swapped
and the original number is equal to -9x + 9y, where x is the tens digit
and y is the units digit of the original number.
-/
theorem two_digit_swap_difference (n : TwoDigitNumber) :
  (value (swap_digits n) : ℤ) - (value n : ℤ) = -9 * (n.tens : ℤ) + 9 * (n.units : ℤ) := by
  sorry

end two_digit_swap_difference_l1557_155773


namespace max_product_constrained_max_product_value_max_product_achieved_l1557_155762

theorem max_product_constrained (m n : ℝ) : 
  m = 8 - n → m > 0 → n > 0 → ∀ x y : ℝ, x = 8 - y → x > 0 → y > 0 → x * y ≤ m * n := by
  sorry

theorem max_product_value (m n : ℝ) :
  m = 8 - n → m > 0 → n > 0 → m * n ≤ 16 := by
  sorry

theorem max_product_achieved (m n : ℝ) :
  m = 8 - n → m > 0 → n > 0 → ∃ x y : ℝ, x = 8 - y ∧ x > 0 ∧ y > 0 ∧ x * y = 16 := by
  sorry

end max_product_constrained_max_product_value_max_product_achieved_l1557_155762


namespace other_asymptote_equation_l1557_155774

/-- A hyperbola with one known asymptote and foci on a vertical line -/
structure Hyperbola where
  /-- The slope of the known asymptote -/
  known_asymptote_slope : ℝ
  /-- The x-coordinate of the line containing the foci -/
  foci_x : ℝ

/-- The equation of the other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => y = (-h.known_asymptote_slope) * x + (h.known_asymptote_slope + 1) * h.foci_x * 2

theorem other_asymptote_equation (h : Hyperbola) 
    (h_slope : h.known_asymptote_slope = 4) 
    (h_foci : h.foci_x = 3) :
    other_asymptote h = fun x y => y = -4 * x + 24 := by
  sorry

end other_asymptote_equation_l1557_155774


namespace patricia_candy_count_l1557_155789

theorem patricia_candy_count (initial_candy : ℕ) (taken_candy : ℕ) : 
  initial_candy = 76 → taken_candy = 5 → initial_candy - taken_candy = 71 := by
  sorry

end patricia_candy_count_l1557_155789


namespace cosine_period_l1557_155756

/-- The period of the cosine function with a modified argument -/
theorem cosine_period (f : ℝ → ℝ) (h : f = λ x => Real.cos ((3 * x) / 4 + π / 6)) :
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = 8 * π / 3 :=
sorry

end cosine_period_l1557_155756


namespace a_plus_b_equals_one_l1557_155743

theorem a_plus_b_equals_one (a b : ℝ) (h : |a^3 - 27| + (b + 2)^2 = 0) : a + b = 1 := by
  sorry

end a_plus_b_equals_one_l1557_155743


namespace fifteen_sided_polygon_diagonals_l1557_155725

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 15 sides has 90 diagonals -/
theorem fifteen_sided_polygon_diagonals :
  num_diagonals 15 = 90 := by
  sorry

end fifteen_sided_polygon_diagonals_l1557_155725


namespace q_coordinates_is_rectangle_l1557_155780

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by four points -/
structure Rectangle where
  O : Point
  P : Point
  Q : Point
  R : Point

/-- Definition of our specific rectangle -/
def our_rectangle : Rectangle :=
  { O := { x := 0, y := 0 }
  , P := { x := 0, y := 3 }
  , R := { x := 5, y := 0 }
  , Q := { x := 5, y := 3 } }

/-- Theorem: The coordinates of Q in our_rectangle are (5,3) -/
theorem q_coordinates :
  our_rectangle.Q.x = 5 ∧ our_rectangle.Q.y = 3 := by
  sorry

/-- Theorem: our_rectangle is indeed a rectangle -/
theorem is_rectangle (rect : Rectangle) : 
  (rect.O.x = rect.P.x ∧ rect.O.y = rect.R.y) →
  (rect.Q.x = rect.R.x ∧ rect.Q.y = rect.P.y) →
  (rect.P.x - rect.O.x)^2 + (rect.P.y - rect.O.y)^2 =
  (rect.R.x - rect.O.x)^2 + (rect.R.y - rect.O.y)^2 →
  True := by
  sorry

end q_coordinates_is_rectangle_l1557_155780


namespace prime_greater_than_five_form_l1557_155797

theorem prime_greater_than_five_form (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ∃ k : ℕ, p = 6 * k + 1 := by
sorry

end prime_greater_than_five_form_l1557_155797


namespace broccoli_area_l1557_155705

theorem broccoli_area (current_production : ℕ) (increase : ℕ) : 
  current_production = 2601 →
  increase = 101 →
  ∃ (previous_side : ℕ) (current_side : ℕ),
    previous_side ^ 2 + increase = current_side ^ 2 ∧
    current_side ^ 2 = current_production ∧
    (current_side ^ 2 : ℚ) / current_production = 1 :=
by sorry

end broccoli_area_l1557_155705


namespace remainder_theorem_l1557_155777

theorem remainder_theorem : (7 * 10^15 + 3^15) % 9 = 7 := by
  sorry

end remainder_theorem_l1557_155777


namespace shortest_side_of_octagon_l1557_155758

theorem shortest_side_of_octagon (x : ℝ) : 
  x > 0 →                             -- x is positive
  x^2 = 100 →                         -- combined area of cut-off triangles
  20 - x = 10 :=                      -- shortest side of octagon
by sorry

end shortest_side_of_octagon_l1557_155758
