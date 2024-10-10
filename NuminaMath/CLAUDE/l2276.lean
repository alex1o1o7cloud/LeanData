import Mathlib

namespace conference_handshakes_l2276_227627

/-- The number of handshakes in a conference with n participants -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that a conference with 10 participants results in 45 handshakes -/
theorem conference_handshakes : handshakes 10 = 45 := by
  sorry

end conference_handshakes_l2276_227627


namespace speed_time_relationship_l2276_227633

-- Define the initial speed and time
variable (x y : ℝ)

-- Define the percentage increases/decreases
variable (a b : ℝ)

-- Condition: x and y are positive (speed and time can't be negative or zero)
variable (hx : x > 0)
variable (hy : y > 0)

-- Condition: a and b are percentages (between 0 and 100)
variable (ha : 0 ≤ a ∧ a ≤ 100)
variable (hb : 0 ≤ b ∧ b ≤ 100)

-- Theorem stating the relationship between a and b
theorem speed_time_relationship : 
  x * y = x * (1 + a / 100) * (y * (1 - b / 100)) → 
  b = (100 * a) / (100 + a) := by
sorry

end speed_time_relationship_l2276_227633


namespace fraction_scaling_l2276_227669

theorem fraction_scaling (x y : ℝ) :
  (3*x + 3*y) / ((3*x)^2 + (3*y)^2) = (1/3) * ((x + y) / (x^2 + y^2)) :=
by sorry

end fraction_scaling_l2276_227669


namespace P_zero_for_floor_l2276_227665

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The polynomial P(x,y) -/
def P (x y : ℤ) : ℤ :=
  (y - 2*x) * (y - 2*x - 1)

/-- Theorem stating that P(⌊a⌋, ⌊2a⌋) = 0 for all real numbers a -/
theorem P_zero_for_floor (a : ℝ) : P (floor a) (floor (2*a)) = 0 := by
  sorry

end P_zero_for_floor_l2276_227665


namespace hyperbola_inequality_l2276_227649

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 = 1

-- Define the line
def line (x : ℝ) : Prop := x = 3

-- Define the intersection points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (3, -1)

-- Define a point P on the hyperbola
def P (x y : ℝ) : Prop := hyperbola x y

-- Define the vector representation of OP
def OP (a b : ℝ) : ℝ × ℝ := (2*a + 2*b, a - b)

theorem hyperbola_inequality (a b : ℝ) :
  (∀ x y, P x y → OP a b = (x, y)) → |a + b| ≥ 1 := by sorry

end hyperbola_inequality_l2276_227649


namespace line_slope_intercept_sum_l2276_227620

/-- Given points A, B, C, and D where D is the midpoint of AB, 
    prove that the sum of the slope and y-intercept of line CD is 27/10 -/
theorem line_slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, -2) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (C.2 - D.2) / (C.1 - D.1)
  let y_intercept := D.2
  slope + y_intercept = 27 / 10 := by
sorry

end line_slope_intercept_sum_l2276_227620


namespace intersection_equality_subset_condition_l2276_227612

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 < 2*x + 1 ∧ 2*x + 1 < 7}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | 3*a - 2 < x ∧ x < a + 1}

-- Statement 1: A ∩ (C_R B) = {x | -2 < x ≤ 2}
theorem intersection_equality : A ∩ (Set.Icc (-4) 2) = {x : ℝ | -2 < x ∧ x ≤ 2} := by sorry

-- Statement 2: C_R (A∪B) ⊆ C if and only if -3 < a < -2/3
theorem subset_condition (a : ℝ) : Set.Icc (-4) 2 ⊆ C a ↔ -3 < a ∧ a < -2/3 := by sorry

end intersection_equality_subset_condition_l2276_227612


namespace inverse_proportion_problem_l2276_227651

/-- Given that x and y are inversely proportional, prove that y = -27 when x = -9,
    given that x = 3y when x + y = 36. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
    (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 36 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) : 
    x = -9 → y = -27 := by
  sorry

end inverse_proportion_problem_l2276_227651


namespace largest_n_for_equation_l2276_227643

theorem largest_n_for_equation : 
  (∀ n : ℕ, n > 4 → ¬∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) ∧
  (∃ x y z : ℕ+, 4^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) := by
  sorry

end largest_n_for_equation_l2276_227643


namespace pauls_erasers_l2276_227682

/-- The number of erasers Paul got for his birthday -/
def erasers : ℕ := 0  -- We'll prove this is actually 457

/-- The number of crayons Paul got for his birthday -/
def initial_crayons : ℕ := 617

/-- The number of crayons Paul had left at the end of the school year -/
def remaining_crayons : ℕ := 523

/-- The difference between the number of crayons and erasers left -/
def crayon_eraser_difference : ℕ := 66

theorem pauls_erasers : 
  erasers = 457 ∧ 
  initial_crayons = 617 ∧
  remaining_crayons = 523 ∧
  crayon_eraser_difference = 66 ∧
  remaining_crayons = erasers + crayon_eraser_difference :=
sorry

end pauls_erasers_l2276_227682


namespace range_of_x_l2276_227640

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval [-1, 1]
def I : Set ℝ := Set.Icc (-1 : ℝ) 1

-- Define the theorem
theorem range_of_x (h1 : Monotone f) (h2 : Set.MapsTo f I I) 
  (h3 : ∀ x, f (x - 2) < f (1 - x)) :
  ∃ S : Set ℝ, S = Set.Ico 1 (3/2) ∧ ∀ x, x ∈ S ↔ 
    (x - 2 ∈ I ∧ 1 - x ∈ I ∧ f (x - 2) < f (1 - x)) :=
sorry

end range_of_x_l2276_227640


namespace greatest_integer_satisfying_inequality_l2276_227690

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, x ≤ 4 ↔ (x : ℝ)^4 / (x : ℝ)^2 < 18 :=
by sorry

end greatest_integer_satisfying_inequality_l2276_227690


namespace batsman_average_l2276_227683

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  (previous_total = 16 * previous_average) →
  (previous_total + 87) / 17 = previous_average + 3 →
  (previous_total + 87) / 17 = 39 := by
sorry

end batsman_average_l2276_227683


namespace winning_candidate_percentage_l2276_227646

/-- Given an election with two candidates, prove that the winning candidate
    received 60% of the votes under the given conditions. -/
theorem winning_candidate_percentage
  (total_votes : ℕ)
  (winning_margin : ℕ)
  (h_total : total_votes = 1400)
  (h_margin : winning_margin = 280) :
  (winning_votes : ℕ) →
  (losing_votes : ℕ) →
  (winning_votes + losing_votes = total_votes) →
  (winning_votes = losing_votes + winning_margin) →
  (winning_votes : ℚ) / total_votes = 60 / 100 :=
by sorry

end winning_candidate_percentage_l2276_227646


namespace cafe_visits_l2276_227635

/-- The number of people in the club -/
def n : ℕ := 9

/-- The number of people who visit the cafe each day -/
def k : ℕ := 3

/-- The number of days -/
def days : ℕ := 360

/-- The number of times each pair visits the cafe -/
def x : ℕ := 30

theorem cafe_visits :
  (n.choose 2) * x = days * (k.choose 2) := by sorry

end cafe_visits_l2276_227635


namespace range_of_a_l2276_227667

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, (1/a) - (1/x) ≤ 2*x) : a ≥ Real.sqrt 2 / 4 := by
  sorry

end range_of_a_l2276_227667


namespace range_of_a_l2276_227644

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4*x + a ≤ 0

theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) :
  Real.exp 1 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l2276_227644


namespace four_integers_sum_l2276_227653

theorem four_integers_sum (a b c d : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧
  a + b + c = 6 ∧
  a + b + d = 7 ∧
  a + c + d = 8 ∧
  b + c + d = 9 →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
sorry

end four_integers_sum_l2276_227653


namespace two_x_eq_zero_is_linear_l2276_227641

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 2x = 0 -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem stating that 2x = 0 is a linear equation -/
theorem two_x_eq_zero_is_linear : is_linear_equation f := by
  sorry


end two_x_eq_zero_is_linear_l2276_227641


namespace function_solution_l2276_227607

open Real

-- Define the function property
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x ≠ 0, f (1 / x) + (5 / x) * f x = 3 / x^3

-- State the theorem
theorem function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
    ∀ x ≠ 0, f x = 5 / (8 * x^2) - x^3 / 8 := by
  sorry

end function_solution_l2276_227607


namespace circle_symmetry_l2276_227686

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

-- Define Circle C₁
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define Circle C₂
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

-- Define symmetry with respect to a line
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_of_symmetry ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  (x2 - x1) * (x2 - x1) = (y2 - y1) * (y2 - y1)

-- Theorem statement
theorem circle_symmetry :
  ∀ x1 y1 x2 y2 : ℝ,
  circle_C1 x1 y1 →
  circle_C2 x2 y2 →
  symmetric_points x1 y1 x2 y2 :=
sorry

end circle_symmetry_l2276_227686


namespace unique_k_solution_l2276_227609

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem unique_k_solution (k : ℤ) : 
  k % 2 = 1 ∧ f (f (f k)) = 35 → k = 55 :=
by sorry

end unique_k_solution_l2276_227609


namespace a_share_is_288_l2276_227606

/-- Calculates the share of profit for investor A given the initial investments,
    changes after 8 months, and total profit over a year. -/
def calculate_share_a (a_initial : ℕ) (b_initial : ℕ) (a_change : ℕ) (b_change : ℕ) (total_profit : ℕ) : ℕ :=
  let a_investment_months := a_initial * 8 + (a_initial - a_change) * 4
  let b_investment_months := b_initial * 8 + (b_initial + b_change) * 4
  let total_investment_months := a_investment_months + b_investment_months
  let a_ratio := a_investment_months * total_profit / total_investment_months
  a_ratio

/-- Theorem stating that A's share of the profit is 288 Rs given the problem conditions. -/
theorem a_share_is_288 :
  calculate_share_a 3000 4000 1000 1000 756 = 288 := by
  sorry

end a_share_is_288_l2276_227606


namespace like_terms_sum_exponents_l2276_227645

/-- Two monomials are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (a b : ℕ) (m n : ℤ) : Prop :=
  m + 1 = 1 ∧ 3 = n

/-- If 5x^(m+1)y^3 and -3xy^n are like terms, then m + n = 3. -/
theorem like_terms_sum_exponents (m n : ℤ) :
  are_like_terms 5 3 m n → m + n = 3 := by
  sorry

end like_terms_sum_exponents_l2276_227645


namespace necessary_but_not_sufficient_l2276_227689

-- Define the original inequality
def original_inequality (x : ℝ) : Prop := 2 * x^2 - 5*x - 3 ≥ 0

-- Define the solution to the original inequality
def solution_inequality (x : ℝ) : Prop := x ≤ -1/2 ∨ x ≥ 3

-- Define the proposed necessary but not sufficient condition
def proposed_condition (x : ℝ) : Prop := x < -1 ∨ x > 4

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ x : ℝ, original_inequality x ↔ solution_inequality x) →
  (∀ x : ℝ, solution_inequality x → proposed_condition x) ∧
  ¬(∀ x : ℝ, proposed_condition x → solution_inequality x) :=
by sorry

end necessary_but_not_sufficient_l2276_227689


namespace triangle_property_l2276_227661

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given conditions -/
theorem triangle_property (t : Triangle) 
  (h1 : Real.sin t.A + Real.sin t.B = 5/4 * Real.sin t.C)
  (h2 : t.a + t.b + t.c = 9)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sin t.C) :
  t.c = 4 ∧ Real.cos t.C = -1/4 := by
  sorry

end triangle_property_l2276_227661


namespace function_identity_l2276_227619

theorem function_identity (f : ℕ → ℕ) 
  (h1 : ∀ (m n : ℕ), f (m^2 + n^2) = (f m)^2 + (f n)^2) 
  (h2 : f 1 > 0) : 
  ∀ (n : ℕ), f n = n := by
  sorry

end function_identity_l2276_227619


namespace example_is_fractional_equation_l2276_227685

/-- Definition of a fractional equation -/
def is_fractional_equation (eq : Prop) : Prop :=
  ∃ (x : ℝ) (f g : ℝ → ℝ) (h : ℝ → ℝ), 
    (∀ y, f y ≠ 0 ∧ g y ≠ 0) ∧ 
    eq ↔ (h x / f x - 3 / g x = 1) ∧
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ f x = a * x + b) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ g x = c * x + d)

/-- The equation (x / (2x - 1)) - (3 / (2x + 1)) = 1 is a fractional equation -/
theorem example_is_fractional_equation : 
  is_fractional_equation (∃ x : ℝ, x / (2 * x - 1) - 3 / (2 * x + 1) = 1) :=
sorry

end example_is_fractional_equation_l2276_227685


namespace distribution_count_is_18_l2276_227678

/-- The number of ways to distribute 6 numbered balls into 3 boxes -/
def distributionCount : ℕ :=
  let totalBalls : ℕ := 6
  let numBoxes : ℕ := 3
  let ballsPerBox : ℕ := 2
  let fixedPair : Fin totalBalls := 2  -- Represents balls 1 and 2 as a fixed pair
  18

/-- Theorem stating that the number of distributions is 18 -/
theorem distribution_count_is_18 : distributionCount = 18 := by
  sorry

end distribution_count_is_18_l2276_227678


namespace complex_fraction_sum_l2276_227638

theorem complex_fraction_sum (a b : ℂ) (h1 : a = 5 + 7*I) (h2 : b = 5 - 7*I) : 
  a / b + b / a = -23 / 37 := by
  sorry

end complex_fraction_sum_l2276_227638


namespace no_base_all_prime_l2276_227670

/-- For any base b ≥ 2, there exists a number of the form 11...1 
    with (b^2 - 1) ones in base b that is not prime. -/
theorem no_base_all_prime (b : ℕ) (hb : b ≥ 2) : 
  ∃ N : ℕ, (∃ k : ℕ, N = (b^(2*k) - 1) / (b^2 - 1)) ∧ ¬ Prime N := by
  sorry

end no_base_all_prime_l2276_227670


namespace star_polygon_angle_sum_l2276_227628

/-- A star polygon created from a regular n-gon --/
structure StarPolygon where
  n : ℕ
  n_ge_6 : n ≥ 6

/-- The sum of interior angles of a star polygon --/
def sum_interior_angles (s : StarPolygon) : ℝ :=
  180 * (s.n - 2)

/-- Theorem: The sum of interior angles of a star polygon is 180°(n-2) --/
theorem star_polygon_angle_sum (s : StarPolygon) :
  sum_interior_angles s = 180 * (s.n - 2) :=
by sorry

end star_polygon_angle_sum_l2276_227628


namespace store_discount_is_ten_percent_l2276_227659

/-- Calculates the discount percentage given the number of items, cost per item, 
    discount threshold, and final cost after discount. -/
def discount_percentage (num_items : ℕ) (cost_per_item : ℚ) 
  (discount_threshold : ℚ) (final_cost : ℚ) : ℚ :=
  let total_cost := num_items * cost_per_item
  let discount_amount := total_cost - final_cost
  let eligible_amount := total_cost - discount_threshold
  (discount_amount / eligible_amount) * 100

/-- Proves that the discount percentage is 10% for the given scenario. -/
theorem store_discount_is_ten_percent :
  discount_percentage 7 200 1000 1360 = 10 := by
  sorry

end store_discount_is_ten_percent_l2276_227659


namespace mike_seashell_count_l2276_227608

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of seashells Mike found later -/
def later_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := initial_seashells + later_seashells

theorem mike_seashell_count : total_seashells = 10.0 := by
  sorry

end mike_seashell_count_l2276_227608


namespace minimum_additional_candies_l2276_227696

theorem minimum_additional_candies 
  (initial_candies : ℕ) 
  (num_students : ℕ) 
  (additional_candies : ℕ) : 
  initial_candies = 237 →
  num_students = 31 →
  additional_candies = 11 →
  (∃ (candies_per_student : ℕ), 
    (initial_candies + additional_candies) = num_students * candies_per_student) ∧
  (∀ (x : ℕ), x < additional_candies →
    ¬(∃ (y : ℕ), (initial_candies + x) = num_students * y)) :=
by sorry

end minimum_additional_candies_l2276_227696


namespace two_ways_to_combine_fractions_l2276_227677

theorem two_ways_to_combine_fractions : ∃ (f g : ℚ → ℚ → ℚ → ℚ),
  f (1/8) (1/9) (1/28) = 1/2016 ∧
  g (1/8) (1/9) (1/28) = 1/2016 ∧
  f ≠ g :=
by sorry

end two_ways_to_combine_fractions_l2276_227677


namespace age_of_other_man_is_21_l2276_227664

/-- The age of the other replaced man in a group replacement scenario -/
def age_of_other_replaced_man (initial_count : ℕ) (age_increase : ℝ) (age_of_one_replaced : ℕ) (avg_age_new_men : ℝ) : ℝ :=
  let total_age_increase := initial_count * age_increase
  let total_age_new_men := 2 * avg_age_new_men
  total_age_new_men - total_age_increase - age_of_one_replaced

/-- Theorem: The age of the other replaced man is 21 years -/
theorem age_of_other_man_is_21 :
  age_of_other_replaced_man 10 2 23 32 = 21 := by
  sorry

#eval age_of_other_replaced_man 10 2 23 32

end age_of_other_man_is_21_l2276_227664


namespace purely_imaginary_complex_number_l2276_227613

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m + 2 + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = -2 := by
  sorry

end purely_imaginary_complex_number_l2276_227613


namespace bricks_used_total_bricks_used_l2276_227650

/-- Calculates the total number of bricks used in a construction project -/
theorem bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) (incomplete_courses : ℕ) : ℕ :=
  let complete_walls := total_walls - 1
  let complete_wall_bricks := courses_per_wall * bricks_per_course
  let incomplete_wall_courses := courses_per_wall - incomplete_courses
  let incomplete_wall_bricks := incomplete_wall_courses * bricks_per_course
  complete_walls * complete_wall_bricks + incomplete_wall_bricks

/-- Proves that the total number of bricks used is 1140 given the specific conditions -/
theorem total_bricks_used :
  bricks_used 10 20 6 3 = 1140 := by
  sorry

end bricks_used_total_bricks_used_l2276_227650


namespace difference_implies_70_l2276_227600

/-- Represents a two-digit numeral -/
structure TwoDigitNumeral where
  tens : Nat
  ones : Nat
  tens_lt_10 : tens < 10
  ones_lt_10 : ones < 10

/-- The place value of a digit in a two-digit numeral -/
def placeValue (n : TwoDigitNumeral) (d : Nat) : Nat :=
  if d = n.tens then 10 * n.tens else n.ones

/-- The face value of a digit -/
def faceValue (d : Nat) : Nat := d

/-- The theorem stating that if the difference between the place value and face value
    of 7 in a two-digit numeral is 63, then the numeral is 70 -/
theorem difference_implies_70 (n : TwoDigitNumeral) :
  placeValue n 7 - faceValue 7 = 63 → n.tens = 7 ∧ n.ones = 0 := by
  sorry

#check difference_implies_70

end difference_implies_70_l2276_227600


namespace modulus_of_z_l2276_227602

theorem modulus_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1 / 5 := by
  sorry

end modulus_of_z_l2276_227602


namespace gcf_120_180_300_l2276_227631

theorem gcf_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end gcf_120_180_300_l2276_227631


namespace two_digit_number_property_l2276_227630

theorem two_digit_number_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (∃ x y : ℕ, 
    n = 10 * x + y ∧ 
    x < 10 ∧ 
    y < 10 ∧ 
    n = x^3 + y^2) :=
by
  sorry

end two_digit_number_property_l2276_227630


namespace binary_multiplication_subtraction_l2276_227660

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to a binary representation as a list of bits. -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_subtraction :
  let a := binary_to_nat [true, false, true, true]  -- 1101₂
  let b := binary_to_nat [true, true, true]         -- 111₂
  let c := binary_to_nat [true, false, true]        -- 101₂
  nat_to_binary ((a * b) - c) = [false, false, false, true, false, false, true] -- 1001000₂
:= by sorry

end binary_multiplication_subtraction_l2276_227660


namespace symmetric_point_correct_l2276_227623

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_correct :
  let P : Point := (-2, 3)
  symmetricToXAxis P = (-2, -3) := by sorry

end symmetric_point_correct_l2276_227623


namespace count_integer_pairs_l2276_227632

theorem count_integer_pairs : 
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + 2*p.2 < 40) (Finset.product (Finset.range 40) (Finset.range 40))).card = 72 := by
  sorry

end count_integer_pairs_l2276_227632


namespace bird_families_left_l2276_227676

theorem bird_families_left (total : ℕ) (to_africa : ℕ) (to_asia : ℕ) 
  (h1 : total = 85) (h2 : to_africa = 23) (h3 : to_asia = 37) : 
  total - (to_africa + to_asia) = 25 := by
  sorry

end bird_families_left_l2276_227676


namespace ab2_minus_41_equals_591_l2276_227639

/-- Given two single-digit numbers A and B, where AB2 is a three-digit number,
    prove that when A = 6 and B = 2, the equation AB2 - 41 = 591 is valid. -/
theorem ab2_minus_41_equals_591 (A B : Nat) : 
  A < 10 → B < 10 → 100 ≤ A * 100 + B * 10 + 2 → A * 100 + B * 10 + 2 < 1000 →
  A = 6 → B = 2 → A * 100 + B * 10 + 2 - 41 = 591 := by
sorry

end ab2_minus_41_equals_591_l2276_227639


namespace train_length_train_length_is_240_l2276_227615

/-- The length of a train crossing a bridge -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - bridge_length

/-- Proof that the train length is 240 meters -/
theorem train_length_is_240 :
  train_length 150 20 70.2 = 240 := by
  sorry

end train_length_train_length_is_240_l2276_227615


namespace wrapping_paper_fraction_l2276_227684

theorem wrapping_paper_fraction (total_fraction : Rat) (num_presents : Nat) 
  (h1 : total_fraction = 5 / 12)
  (h2 : num_presents = 4) :
  total_fraction / num_presents = 5 / 48 := by
sorry

end wrapping_paper_fraction_l2276_227684


namespace percentage_to_decimal_two_percent_to_decimal_l2276_227656

theorem percentage_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem two_percent_to_decimal : (2 : ℚ) / 100 = 0.02 := by sorry

end percentage_to_decimal_two_percent_to_decimal_l2276_227656


namespace inequality_implication_l2276_227681

theorem inequality_implication (a b c : ℝ) (h1 : a / c^2 > b / c^2) (h2 : c ≠ 0) : a^2 > b^2 := by
  sorry

end inequality_implication_l2276_227681


namespace right_handed_players_count_l2276_227675

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 120)
  (h2 : throwers = 45)
  (h3 : throwers ≤ total_players)
  (h4 : 5 * (total_players - throwers) % 5 = 0) -- Ensures divisibility by 5
  : (throwers + (3 * (total_players - throwers) / 5) : ℕ) = 90 := by
  sorry

end right_handed_players_count_l2276_227675


namespace train_length_l2276_227626

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 →
  crossing_time = 29.997600191984642 →
  bridge_length = 390 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 := by
  sorry

end train_length_l2276_227626


namespace polynomial_division_quotient_l2276_227634

theorem polynomial_division_quotient :
  ∀ x : ℝ, x ≠ 1 →
  (x^6 + 5) / (x - 1) = x^5 + x^4 + x^3 + x^2 + x + 1 := by
sorry

end polynomial_division_quotient_l2276_227634


namespace root_product_equation_l2276_227614

theorem root_product_equation (m p q : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 6 = 0) → 
  (b^2 - m*b + 6 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 49/6 := by
sorry

end root_product_equation_l2276_227614


namespace quadratic_root_value_l2276_227654

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 7 * x + k = 0 ↔ x = (7 + Real.sqrt 17) / 4 ∨ x = (7 - Real.sqrt 17) / 4) →
  k = 4 := by
sorry

end quadratic_root_value_l2276_227654


namespace largest_roots_ratio_l2276_227621

/-- The polynomial f(x) = 1 - x - 4x² + x⁴ -/
def f (x : ℝ) : ℝ := 1 - x - 4*x^2 + x^4

/-- The polynomial g(x) = 16 - 8x - 16x² + x⁴ -/
def g (x : ℝ) : ℝ := 16 - 8*x - 16*x^2 + x^4

/-- x₁ is the largest root of f -/
def x₁ : ℝ := sorry

/-- x₂ is the largest root of g -/
def x₂ : ℝ := sorry

theorem largest_roots_ratio :
  x₁ / x₂ = 1 / 2 := by sorry

end largest_roots_ratio_l2276_227621


namespace rational_root_of_cubic_l2276_227672

/-- Given a cubic polynomial with rational coefficients, if 3 + √5 is a root
    and another root is rational, then the rational root is -6 -/
theorem rational_root_of_cubic (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 3 + Real.sqrt 5) →
  (∃ r : ℚ, r^3 + a*r^2 + b*r + c = 0) →
  (∃ r : ℚ, r^3 + a*r^2 + b*r + c = 0 ∧ r = -6) :=
by sorry

end rational_root_of_cubic_l2276_227672


namespace func_f_properties_l2276_227601

/-- A function satisfying the given functional equation -/
noncomputable def FuncF (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * f b) ∧ 
  (f 0 ≠ 0) ∧
  (∃ c : ℝ, c > 0 ∧ f (c / 2) = 0)

theorem func_f_properties (f : ℝ → ℝ) (h : FuncF f) :
  (f 0 = 1) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, f (x + 2 * c) = f x) :=
by sorry

end func_f_properties_l2276_227601


namespace candy_average_l2276_227662

theorem candy_average (eunji_candies : ℕ) (jimin_diff : ℕ) (jihyun_diff : ℕ) : 
  eunji_candies = 35 →
  jimin_diff = 6 →
  jihyun_diff = 3 →
  (eunji_candies + (eunji_candies + jimin_diff) + (eunji_candies - jihyun_diff)) / 3 = 36 := by
  sorry

end candy_average_l2276_227662


namespace solution_set_of_inequality_l2276_227629

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 ≥ 2*x ↔ x ∈ Set.Iic 0 ∪ Set.Ici 2 := by
  sorry

end solution_set_of_inequality_l2276_227629


namespace quadratic_discriminant_l2276_227648

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of the quadratic equation 5x^2 - 11x + 4 is 41 -/
theorem quadratic_discriminant : discriminant 5 (-11) 4 = 41 := by
  sorry

end quadratic_discriminant_l2276_227648


namespace point_transformation_to_third_quadrant_l2276_227674

/-- Given a point (a, b) in the fourth quadrant, prove that (a/b, 2b-a) is in the third quadrant -/
theorem point_transformation_to_third_quadrant (a b : ℝ) 
  (h1 : a > 0) (h2 : b < 0) : (a / b < 0) ∧ (2 * b - a < 0) := by
  sorry

end point_transformation_to_third_quadrant_l2276_227674


namespace tree_branches_count_l2276_227666

/-- Proves that a tree with the given characteristics has 30 branches -/
theorem tree_branches_count : 
  ∀ (total_leaves : ℕ) (twigs_per_branch : ℕ) 
    (four_leaf_twig_percent : ℚ) (five_leaf_twig_percent : ℚ),
  total_leaves = 12690 →
  twigs_per_branch = 90 →
  four_leaf_twig_percent = 30 / 100 →
  five_leaf_twig_percent = 70 / 100 →
  four_leaf_twig_percent + five_leaf_twig_percent = 1 →
  ∃ (branches : ℕ),
    branches * (four_leaf_twig_percent * twigs_per_branch * 4 + 
                five_leaf_twig_percent * twigs_per_branch * 5) = total_leaves ∧
    branches = 30 := by
  sorry

end tree_branches_count_l2276_227666


namespace largest_four_digit_divisible_by_88_l2276_227624

theorem largest_four_digit_divisible_by_88 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 88 = 0 → n ≤ 9944 :=
by sorry

end largest_four_digit_divisible_by_88_l2276_227624


namespace first_day_rainfall_is_26_l2276_227603

/-- Rainfall data for May -/
structure RainfallData where
  day2 : ℝ
  day3_diff : ℝ
  normal_average : ℝ
  less_than_average : ℝ

/-- Calculate the rainfall on the first day -/
def calculate_first_day_rainfall (data : RainfallData) : ℝ :=
  3 * data.normal_average - data.less_than_average - data.day2 - (data.day2 - data.day3_diff)

/-- Theorem stating that the rainfall on the first day is 26 cm -/
theorem first_day_rainfall_is_26 (data : RainfallData)
  (h1 : data.day2 = 34)
  (h2 : data.day3_diff = 12)
  (h3 : data.normal_average = 140)
  (h4 : data.less_than_average = 58) :
  calculate_first_day_rainfall data = 26 := by
  sorry

#eval calculate_first_day_rainfall ⟨34, 12, 140, 58⟩

end first_day_rainfall_is_26_l2276_227603


namespace triangle_ratio_equals_two_l2276_227616

/-- In triangle ABC, if angle A is 60 degrees and side a is √3, 
    then (a + b) / (sin A + sin B) = 2 -/
theorem triangle_ratio_equals_two (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  A = π / 3 ∧ 
  a = Real.sqrt 3 ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  (a + b) / (Real.sin A + Real.sin B) = 2 := by
sorry

end triangle_ratio_equals_two_l2276_227616


namespace total_students_proof_l2276_227625

/-- The number of students who knew about the event -/
def students_who_knew : ℕ := 40

/-- The frequency of students who knew about the event -/
def frequency : ℚ := 8/10

/-- The total number of students participating in the competition -/
def total_students : ℕ := 50

/-- Theorem stating that the total number of students is 50 given the conditions -/
theorem total_students_proof : 
  (students_who_knew : ℚ) / frequency = total_students := by sorry

end total_students_proof_l2276_227625


namespace probability_two_green_balls_l2276_227642

def total_balls : ℕ := 12
def red_balls : ℕ := 3
def yellow_balls : ℕ := 5
def green_balls : ℕ := 4
def drawn_balls : ℕ := 3

theorem probability_two_green_balls :
  (Nat.choose green_balls 2 * Nat.choose (total_balls - green_balls) 1) /
  Nat.choose total_balls drawn_balls = 12 / 55 :=
by sorry

end probability_two_green_balls_l2276_227642


namespace largest_c_for_range_containing_negative_five_l2276_227611

theorem largest_c_for_range_containing_negative_five :
  let f (x c : ℝ) := x^2 + 5*x + c
  ∃ (c_max : ℝ), c_max = 5/4 ∧
    (∀ c : ℝ, (∃ x : ℝ, f x c = -5) → c ≤ c_max) ∧
    (∃ x : ℝ, f x c_max = -5) :=
by sorry

end largest_c_for_range_containing_negative_five_l2276_227611


namespace quadratic_function_unique_l2276_227658

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_unique :
  ∀ a b c : ℝ,
  (f a b c (-1) = 0) →
  (∀ x : ℝ, x ≤ f a b c x) →
  (∀ x : ℝ, f a b c x ≤ (1 + x^2) / 2) →
  (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
by sorry

end quadratic_function_unique_l2276_227658


namespace point_moved_upwards_l2276_227604

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point upwards by a given distance -/
def moveUpwards (p : Point) (distance : ℝ) : Point :=
  { x := p.x, y := p.y + distance }

theorem point_moved_upwards (P : Point) (Q : Point) :
  P.x = -3 ∧ P.y = 1 ∧ Q = moveUpwards P 2 → Q.x = -3 ∧ Q.y = 3 := by
  sorry

end point_moved_upwards_l2276_227604


namespace complement_B_union_A_C_subset_B_implies_a_range_l2276_227657

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem complement_B_union_A :
  (Set.univ \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x} :=
sorry

-- Theorem for part (2)
theorem C_subset_B_implies_a_range (a : ℝ) :
  C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
sorry

end complement_B_union_A_C_subset_B_implies_a_range_l2276_227657


namespace bowtie_equation_solution_l2276_227636

-- Define the operation ⊙
noncomputable def bowtie (x y : ℝ) : ℝ := x + Real.sqrt (y + Real.sqrt (y + Real.sqrt (y + Real.sqrt y)))

-- State the theorem
theorem bowtie_equation_solution (h : ℝ) : 
  bowtie 8 h = 12 → h = 12 := by sorry

end bowtie_equation_solution_l2276_227636


namespace gcf_of_2835_and_8960_l2276_227655

theorem gcf_of_2835_and_8960 : Nat.gcd 2835 8960 = 35 := by
  sorry

end gcf_of_2835_and_8960_l2276_227655


namespace small_painting_price_l2276_227697

/-- Represents the price of paintings and sales data for Noah's art business -/
structure PaintingSales where
  large_price : ℕ
  small_price : ℕ
  last_month_large : ℕ
  last_month_small : ℕ
  this_month_total : ℕ

/-- Theorem stating that given the conditions, the price of a small painting is $30 -/
theorem small_painting_price (sales : PaintingSales) 
  (h1 : sales.large_price = 60)
  (h2 : sales.last_month_large = 8)
  (h3 : sales.last_month_small = 4)
  (h4 : sales.this_month_total = 1200) :
  sales.small_price = 30 := by
  sorry

#check small_painting_price

end small_painting_price_l2276_227697


namespace trig_identity_l2276_227679

theorem trig_identity (θ : Real) (h : Real.tan (θ - Real.pi) = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end trig_identity_l2276_227679


namespace complex_fraction_simplification_l2276_227695

theorem complex_fraction_simplification :
  (2 + 4 * Complex.I) / ((1 + Complex.I)^2) = 2 - Complex.I :=
by sorry

end complex_fraction_simplification_l2276_227695


namespace counterexample_exists_l2276_227688

theorem counterexample_exists :
  ∃ (n : ℕ), n ≥ 2 ∧ ¬(∃ (k : ℕ), (2^(2^n) % (2^n - 1)) = 4^k) := by
  sorry

end counterexample_exists_l2276_227688


namespace exists_n_divides_1991_l2276_227647

theorem exists_n_divides_1991 : ∃ n : ℕ, n > 2 ∧ (2 * 10^(n+1) - 9) % 1991 = 0 := by
  sorry

end exists_n_divides_1991_l2276_227647


namespace factorial_ratio_l2276_227617

theorem factorial_ratio (n : ℕ) (h : n > 0) : (Nat.factorial n) / (Nat.factorial (n - 1)) = n := by
  sorry

end factorial_ratio_l2276_227617


namespace pencils_indeterminate_l2276_227693

/-- Represents the contents of a drawer -/
structure Drawer where
  initial_crayons : ℕ
  added_crayons : ℕ
  final_crayons : ℕ
  pencils : ℕ

/-- Theorem stating that the number of pencils cannot be determined -/
theorem pencils_indeterminate (d : Drawer) 
  (h1 : d.initial_crayons = 41)
  (h2 : d.added_crayons = 12)
  (h3 : d.final_crayons = 53)
  : ¬ ∃ (n : ℕ), ∀ (d' : Drawer), 
    d'.initial_crayons = d.initial_crayons ∧ 
    d'.added_crayons = d.added_crayons ∧ 
    d'.final_crayons = d.final_crayons → 
    d'.pencils = n :=
by
  sorry

end pencils_indeterminate_l2276_227693


namespace one_point_inside_circle_l2276_227652

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a predicate for three points being collinear
def collinear (p q r : Point) : Prop := sorry

-- Define a predicate for a point being on a circle determined by three other points
def onCircle (p q r s : Point) : Prop := sorry

-- Define a predicate for a point being inside a circle determined by three other points
def insideCircle (p q r s : Point) : Prop := sorry

-- Theorem statement
theorem one_point_inside_circle (A B C D : Point) 
  (h_not_collinear : ¬(collinear A B C) ∧ ¬(collinear A B D) ∧ ¬(collinear A C D) ∧ ¬(collinear B C D))
  (h_not_on_circle : ¬(onCircle A B C D) ∧ ¬(onCircle A B D C) ∧ ¬(onCircle A C D B) ∧ ¬(onCircle B C D A)) :
  insideCircle A B C D ∨ insideCircle A B D C ∨ insideCircle A C D B ∨ insideCircle B C D A :=
by sorry

end one_point_inside_circle_l2276_227652


namespace certain_number_value_l2276_227692

theorem certain_number_value : ∃ x : ℝ, 
  (3 - (1/5) * 390 = x - (1/7) * 210 + 114) ∧ 
  (3 - (1/5) * 390 - (x - (1/7) * 210) = 114) → 
  x = -159 := by
sorry

end certain_number_value_l2276_227692


namespace rabbits_distance_specific_rabbits_distance_l2276_227698

/-- The distance between two rabbits' homes given their resting patterns --/
theorem rabbits_distance (white_rest_interval : ℕ) (gray_rest_interval : ℕ) 
  (rest_difference : ℕ) : ℕ :=
  let meeting_point := white_rest_interval * gray_rest_interval * rest_difference / 
    (white_rest_interval - gray_rest_interval)
  2 * meeting_point

/-- Proof of the specific rabbit problem --/
theorem specific_rabbits_distance : 
  rabbits_distance 30 20 15 = 1800 := by
  sorry

end rabbits_distance_specific_rabbits_distance_l2276_227698


namespace elevator_force_theorem_gavrila_force_l2276_227622

/-- The force exerted by a person on the floor of a decelerating elevator -/
def elevatorForce (m : ℝ) (g a : ℝ) : ℝ := m * (g - a)

/-- Theorem: The force exerted by a person on the floor of a decelerating elevator
    is equal to the person's mass multiplied by the difference between
    gravitational acceleration and the elevator's deceleration -/
theorem elevator_force_theorem (m g a : ℝ) :
  elevatorForce m g a = m * (g - a) := by
  sorry

/-- Corollary: For Gavrila's specific case -/
theorem gavrila_force :
  elevatorForce 70 10 5 = 350 := by
  sorry

end elevator_force_theorem_gavrila_force_l2276_227622


namespace process_termination_and_difference_l2276_227618

-- Define the lists and their properties
def List1 : Type := {l : List ℕ // ∀ x ∈ l, x % 5 = 1}
def List2 : Type := {l : List ℕ // ∀ x ∈ l, x % 5 = 4}

-- Define the operation
def operation (l1 : List1) (l2 : List2) : List1 × List2 :=
  sorry

-- Define the termination condition
def is_terminated (l1 : List1) (l2 : List2) : Prop :=
  l1.val.length = 1 ∧ l2.val.length = 1

-- Theorem statement
theorem process_termination_and_difference 
  (l1_init : List1) (l2_init : List2) : 
  ∃ (l1_final : List1) (l2_final : List2),
    (is_terminated l1_final l2_final) ∧ 
    (l1_final.val.head? ≠ l2_final.val.head?) :=
  sorry

end process_termination_and_difference_l2276_227618


namespace slope_theorem_l2276_227694

/-- Given two points A(-3, 8) and B(5, y) in a coordinate plane, 
    if the slope of the line through A and B is -1/2, then y = 4. -/
theorem slope_theorem (y : ℝ) : 
  let A : ℝ × ℝ := (-3, 8)
  let B : ℝ × ℝ := (5, y)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -1/2 → y = 4 := by
sorry


end slope_theorem_l2276_227694


namespace smallest_n_for_sqrt_12n_integer_l2276_227680

theorem smallest_n_for_sqrt_12n_integer :
  ∀ n : ℕ+, (∃ k : ℕ+, (12 * n : ℕ) = k ^ 2) → n ≥ 3 :=
by
  sorry

end smallest_n_for_sqrt_12n_integer_l2276_227680


namespace adam_basswood_blocks_l2276_227699

/-- The number of figurines that can be created from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be created from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- Theorem stating that Adam owns 15 blocks of basswood -/
theorem adam_basswood_blocks : 
  ∃ (basswood_blocks : ℕ), 
    basswood_blocks * basswood_figurines + 
    butternut_blocks * butternut_figurines + 
    aspen_blocks * aspen_figurines = total_figurines ∧ 
    basswood_blocks = 15 := by
  sorry

end adam_basswood_blocks_l2276_227699


namespace linda_money_l2276_227671

/-- Represents the price of a single jean in dollars -/
def jean_price : ℕ := 11

/-- Represents the price of a single tee in dollars -/
def tee_price : ℕ := 8

/-- Represents the number of tees sold in a day -/
def tees_sold : ℕ := 7

/-- Represents the number of jeans sold in a day -/
def jeans_sold : ℕ := 4

/-- Calculates the total money Linda had at the end of the day -/
def total_money : ℕ := jean_price * jeans_sold + tee_price * tees_sold

theorem linda_money : total_money = 100 := by
  sorry

end linda_money_l2276_227671


namespace number_fraction_problem_l2276_227637

theorem number_fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (1/3 : ℝ) * (2/5 : ℝ) * N = 64 := by
  sorry

end number_fraction_problem_l2276_227637


namespace license_plate_count_l2276_227691

/-- The number of possible letters for each position in the license plate -/
def num_letters : ℕ := 26

/-- The number of possible odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of possible even digits -/
def num_even_digits : ℕ := 5

/-- The total number of license plates with the given conditions -/
def total_license_plates : ℕ := num_letters^3 * num_odd_digits * (num_odd_digits * num_even_digits + num_even_digits * num_odd_digits)

theorem license_plate_count : total_license_plates = 455625 := by
  sorry

end license_plate_count_l2276_227691


namespace no_real_solutions_count_l2276_227668

theorem no_real_solutions_count : 
  ∀ b c : ℕ+, 
  (∃ x : ℝ, x^2 + (b:ℝ)*x + (c:ℝ) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c:ℝ)*x + (b:ℝ) = 0) :=
by sorry

end no_real_solutions_count_l2276_227668


namespace telephone_probability_l2276_227663

theorem telephone_probability (p1 p2 : ℝ) (h1 : p1 = 0.2) (h2 : p2 = 0.3) :
  p1 + p2 = 0.5 := by
  sorry

end telephone_probability_l2276_227663


namespace last_two_digits_product_l2276_227605

/-- Given an integer n, returns its last two digits as a pair of natural numbers -/
def lastTwoDigits (n : ℤ) : ℕ × ℕ :=
  let tens := (n % 100 / 10).toNat
  let units := (n % 10).toNat
  (tens, units)

/-- Theorem: For any integer divisible by 4 with the sum of its last two digits equal to 17,
    the product of its last two digits is 72 -/
theorem last_two_digits_product (n : ℤ) 
  (div_by_4 : 4 ∣ n) 
  (sum_17 : (lastTwoDigits n).1 + (lastTwoDigits n).2 = 17) : 
  (lastTwoDigits n).1 * (lastTwoDigits n).2 = 72 :=
by
  sorry

#check last_two_digits_product

end last_two_digits_product_l2276_227605


namespace meeting_arrangements_l2276_227687

def number_of_schools : ℕ := 3
def members_per_school : ℕ := 6
def total_members : ℕ := 18
def host_representatives : ℕ := 3
def other_representatives : ℕ := 1

def arrange_meeting : ℕ := 
  number_of_schools * 
  (Nat.choose members_per_school host_representatives) * 
  (Nat.choose members_per_school other_representatives) * 
  (Nat.choose members_per_school other_representatives)

theorem meeting_arrangements :
  arrange_meeting = 2160 :=
sorry

end meeting_arrangements_l2276_227687


namespace simplified_expression_equals_22_5_l2276_227610

theorem simplified_expression_equals_22_5 : 
  1.5 * (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9)) = 22.5 := by
  sorry

end simplified_expression_equals_22_5_l2276_227610


namespace quadratic_roots_sum_squares_minimum_l2276_227673

theorem quadratic_roots_sum_squares_minimum (a : ℝ) 
  (x₁ x₂ : ℝ) (h₁ : x₁^2 + 2*a*x₁ + a^2 + 4*a - 2 = 0) 
  (h₂ : x₂^2 + 2*a*x₂ + a^2 + 4*a - 2 = 0) 
  (h₃ : x₁ ≠ x₂) :
  x₁^2 + x₂^2 ≥ 1/2 ∧ 
  (x₁^2 + x₂^2 = 1/2 ↔ a = 1/2) :=
sorry

end quadratic_roots_sum_squares_minimum_l2276_227673
