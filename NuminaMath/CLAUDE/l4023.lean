import Mathlib

namespace prob_same_length_is_11_35_l4023_402323

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of shorter diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 6

/-- The number of longer diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 3

/-- The total number of segments in a regular hexagon -/
def total_segments : ℕ := num_sides + num_short_diagonals + num_long_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_short_diagonals * (num_short_diagonals - 1) + num_long_diagonals * (num_long_diagonals - 1)) /
  (total_segments * (total_segments - 1))

theorem prob_same_length_is_11_35 : prob_same_length = 11 / 35 := by sorry

end prob_same_length_is_11_35_l4023_402323


namespace symmetric_point_y_axis_of_2_1_l4023_402302

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The operation of finding the symmetric point with respect to the y-axis -/
def symmetricPointYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Theorem stating that the symmetric point of (2,1) with respect to the y-axis is (-2,1) -/
theorem symmetric_point_y_axis_of_2_1 :
  let P : Point := { x := 2, y := 1 }
  symmetricPointYAxis P = { x := -2, y := 1 } := by
  sorry

end symmetric_point_y_axis_of_2_1_l4023_402302


namespace m_range_l4023_402385

def p (x m : ℝ) : Prop := x^2 + 2*x - m > 0

theorem m_range :
  (∀ m : ℝ, ¬(p 1 m) ∧ (p 2 m)) ↔ (∀ m : ℝ, 3 ≤ m ∧ m < 8) :=
by sorry

end m_range_l4023_402385


namespace larger_number_proof_l4023_402378

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 5382) →
  (∃ (x y : ℕ+), x * y = 234 ∧ (x = 13 ∨ x = 18) ∧ (y = 13 ∨ y = 18)) →
  (max a b = 414) := by
sorry

end larger_number_proof_l4023_402378


namespace unique_solution_two_and_five_l4023_402300

theorem unique_solution_two_and_five (x : ℝ) : (x - 2) * (x - 5) = 0 ↔ x = 2 ∨ x = 5 := by
  sorry

end unique_solution_two_and_five_l4023_402300


namespace golf_rounds_l4023_402392

theorem golf_rounds (n : ℕ) (average_score : ℚ) (new_score : ℚ) (drop : ℚ) : 
  average_score = 78 →
  new_score = 68 →
  drop = 2 →
  (n * average_score + new_score) / (n + 1) = average_score - drop →
  n = 4 := by
sorry

end golf_rounds_l4023_402392


namespace prob_one_boy_correct_dist_X_correct_dist_X_sum_to_one_l4023_402309

/-- Represents the probability distribution of a discrete random variable -/
def ProbabilityDistribution (α : Type*) := α → ℚ

/-- The total number of students in the group -/
def total_students : ℕ := 5

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The number of students selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting exactly one boy when choosing two students -/
def prob_one_boy : ℚ := 3/5

/-- Represents the number of boys selected -/
inductive X where
  | zero : X
  | one : X
  | two : X

/-- The probability distribution of X (number of boys selected) -/
def dist_X : ProbabilityDistribution X :=
  fun x => match x with
    | X.zero => 1/10
    | X.one  => 3/5
    | X.two  => 3/10

/-- Theorem stating the probability of selecting exactly one boy is correct -/
theorem prob_one_boy_correct :
  prob_one_boy = 3/5 := by sorry

/-- Theorem stating the probability distribution of X is correct -/
theorem dist_X_correct :
  dist_X X.zero = 1/10 ∧
  dist_X X.one  = 3/5  ∧
  dist_X X.two  = 3/10 := by sorry

/-- Theorem stating the sum of probabilities in the distribution equals 1 -/
theorem dist_X_sum_to_one :
  dist_X X.zero + dist_X X.one + dist_X X.two = 1 := by sorry

end prob_one_boy_correct_dist_X_correct_dist_X_sum_to_one_l4023_402309


namespace expression_equals_one_l4023_402322

def numerator : ℕ → ℚ
  | 0 => 1
  | n + 1 => numerator n * (1 + 18 / (n + 1))

def denominator : ℕ → ℚ
  | 0 => 1
  | n + 1 => denominator n * (1 + 20 / (n + 1))

theorem expression_equals_one :
  (numerator 20) / (denominator 18) = 1 := by
  sorry

end expression_equals_one_l4023_402322


namespace f_monotone_decreasing_implies_m_range_l4023_402358

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of monotonically decreasing
def monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem f_monotone_decreasing_implies_m_range (m : ℝ) :
  monotone_decreasing f (2*m) (m+1) → m ∈ Set.Icc (-1) 1 :=
sorry

end f_monotone_decreasing_implies_m_range_l4023_402358


namespace base8_253_to_base10_l4023_402381

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds * 8^2 + tens * 8^1 + units * 8^0

-- Theorem statement
theorem base8_253_to_base10 : base8ToBase10 253 = 171 := by
  sorry

end base8_253_to_base10_l4023_402381


namespace inequality_proof_l4023_402370

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ b*c/(b+c) + c*a/(c+a) + a*b/(a+b) + (1/2) * (b*c/a + c*a/b + a*b/c) := by
  sorry

end inequality_proof_l4023_402370


namespace base5_arithmetic_sequence_implies_xyz_decimal_l4023_402369

/-- Converts a base-5 number to decimal -/
def toDecimal (a b c : Nat) : Nat :=
  a * 25 + b * 5 + c

/-- Checks if a number is a valid base-5 digit -/
def isBase5Digit (n : Nat) : Prop :=
  n ≥ 0 ∧ n < 5

theorem base5_arithmetic_sequence_implies_xyz_decimal (V W X Y Z : Nat) :
  isBase5Digit V ∧ isBase5Digit W ∧ isBase5Digit X ∧ isBase5Digit Y ∧ isBase5Digit Z →
  toDecimal V Y X = toDecimal V Y Z + 1 →
  toDecimal V V W = toDecimal V Y X + 1 →
  toDecimal X Y Z = 108 := by
  sorry

end base5_arithmetic_sequence_implies_xyz_decimal_l4023_402369


namespace interest_is_37_cents_l4023_402345

/-- Calculates the interest in cents given the initial principal and final amount after interest --/
def interest_in_cents (principal : ℚ) (final_amount : ℚ) : ℕ :=
  let interest_rate : ℚ := 3 / 100
  let time : ℚ := 1 / 4
  let interest : ℚ := final_amount - principal
  (interest * 100).floor.toNat

/-- Theorem stating that for some initial amount resulting in $310.45 after 3% annual simple interest for 3 months, the interest in cents is 37 --/
theorem interest_is_37_cents :
  ∃ (principal : ℚ),
    let final_amount : ℚ := 310.45
    interest_in_cents principal final_amount = 37 :=
sorry

end interest_is_37_cents_l4023_402345


namespace perpendicular_vectors_x_value_l4023_402340

theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![(-3), 1]
  let b : Fin 2 → ℝ := ![x, 6]
  (∀ (i j : Fin 2), i.val + j.val = 1 → a i * b j = 0) →
  x = 2 :=
by
  sorry

end perpendicular_vectors_x_value_l4023_402340


namespace jean_is_cyclist_l4023_402374

/-- Represents a traveler's journey --/
structure Traveler where
  distanceTraveled : ℝ
  distanceRemaining : ℝ

/-- Jean's travel condition --/
def jeanCondition (j : Traveler) : Prop :=
  3 * j.distanceTraveled + 2 * j.distanceRemaining = j.distanceTraveled + j.distanceRemaining

/-- Jules' travel condition --/
def julesCondition (j : Traveler) : Prop :=
  (1/2) * j.distanceTraveled + 3 * j.distanceRemaining = j.distanceTraveled + j.distanceRemaining

/-- The theorem to prove --/
theorem jean_is_cyclist (jean jules : Traveler) 
  (hj : jeanCondition jean) (hk : julesCondition jules) : 
  jean.distanceTraveled / (jean.distanceTraveled + jean.distanceRemaining) < 
  jules.distanceTraveled / (jules.distanceTraveled + jules.distanceRemaining) :=
sorry

end jean_is_cyclist_l4023_402374


namespace x_positive_sufficient_not_necessary_for_abs_x_positive_l4023_402326

theorem x_positive_sufficient_not_necessary_for_abs_x_positive :
  (∃ x : ℝ, x > 0 → |x| > 0) ∧
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := by
  sorry

end x_positive_sufficient_not_necessary_for_abs_x_positive_l4023_402326


namespace four_lines_max_regions_l4023_402359

/-- The maximum number of regions a plane can be divided into by n lines -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions a plane can be divided into by four lines is 11 -/
theorem four_lines_max_regions : max_regions 4 = 11 := by
  sorry

end four_lines_max_regions_l4023_402359


namespace area_of_square_on_hypotenuse_l4023_402308

/-- Represents a right-angled isosceles triangle with squares on its sides -/
structure IsoscelesRightTriangle where
  /-- Length of the equal sides -/
  side : ℝ
  /-- Sum of the areas of squares on all sides -/
  squaresSum : ℝ
  /-- The sum of squares is 450 -/
  sum_eq_450 : squaresSum = 450

/-- The area of the square on the hypotenuse of an isosceles right triangle -/
def squareOnHypotenuse (t : IsoscelesRightTriangle) : ℝ := 2 * t.side^2

theorem area_of_square_on_hypotenuse (t : IsoscelesRightTriangle) :
  squareOnHypotenuse t = 225 := by
  sorry

end area_of_square_on_hypotenuse_l4023_402308


namespace jimmy_stair_climbing_l4023_402351

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stair_climbing : arithmetic_sum 25 7 6 = 255 := by
  sorry

end jimmy_stair_climbing_l4023_402351


namespace djibo_age_problem_l4023_402325

/-- Represents the problem of finding when Djibo and his sister's ages summed to 35 --/
theorem djibo_age_problem (djibo_current_age sister_current_age past_sum : ℕ) 
  (h1 : djibo_current_age = 17)
  (h2 : sister_current_age = 28)
  (h3 : past_sum = 35) :
  ∃ (years_ago : ℕ), 
    (djibo_current_age - years_ago) + (sister_current_age - years_ago) = past_sum ∧ 
    years_ago = 5 := by
  sorry


end djibo_age_problem_l4023_402325


namespace alpha_value_l4023_402397

theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - Real.pi / 18) = Real.sqrt 3 / 2) : 
  α = Real.pi * 7 / 18 := by
  sorry

end alpha_value_l4023_402397


namespace family_savings_l4023_402339

def income : ℕ := 509600
def expenses : ℕ := 276000
def initial_savings : ℕ := 1147240

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end family_savings_l4023_402339


namespace somus_age_l4023_402373

theorem somus_age (somu father : ℕ) : 
  somu = father / 3 → 
  (somu - 7) = (father - 7) / 5 → 
  somu = 14 := by
sorry

end somus_age_l4023_402373


namespace min_sqrt_equality_l4023_402352

theorem min_sqrt_equality {a b c : ℝ} (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a*b + 1)/(a*b*c))) (min (Real.sqrt ((b*c + 1)/(a*b*c))) (Real.sqrt ((c*a + 1)/(a*b*c)))) =
    Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔
  ∃ w : ℝ, w > 0 ∧ a = w^2/(1+(w^2+1)^2) ∧ b = w^2/(1+w^2) ∧ c = 1/(1+w^2) :=
by sorry

end min_sqrt_equality_l4023_402352


namespace intersection_point_l4023_402338

/-- The slope of the first line -/
def m : ℚ := 3

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = m * x + 2

/-- The point through which the perpendicular line passes -/
def point : ℚ × ℚ := (3, 4)

/-- The slope of the perpendicular line -/
def m_perp : ℚ := -1 / m

/-- The perpendicular line equation -/
def line2 (x y : ℚ) : Prop := y - point.2 = m_perp * (x - point.1)

/-- The intersection point -/
def intersection : ℚ × ℚ := (9/10, 47/10)

theorem intersection_point : 
  line1 intersection.1 intersection.2 ∧ 
  line2 intersection.1 intersection.2 := by sorry

end intersection_point_l4023_402338


namespace f_has_root_in_interval_l4023_402376

-- Define the function f(x) = x³ - 2x² + 2
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

-- Theorem statement
theorem f_has_root_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) (-1/2 : ℝ), f x = 0 :=
by
  sorry


end f_has_root_in_interval_l4023_402376


namespace katy_brownies_l4023_402395

/-- The number of brownies Katy eats on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy makes in total -/
def total_brownies : ℕ := monday_brownies + 2 * monday_brownies

theorem katy_brownies : 
  total_brownies = 15 := by sorry

end katy_brownies_l4023_402395


namespace pedros_plums_l4023_402382

theorem pedros_plums (total_fruits : ℕ) (total_cost : ℕ) (plum_cost : ℕ) (peach_cost : ℕ) 
  (h1 : total_fruits = 32)
  (h2 : total_cost = 52)
  (h3 : plum_cost = 2)
  (h4 : peach_cost = 1) :
  ∃ (plums peaches : ℕ), 
    plums + peaches = total_fruits ∧ 
    plum_cost * plums + peach_cost * peaches = total_cost ∧
    plums = 20 :=
by sorry

end pedros_plums_l4023_402382


namespace ratio_of_shares_l4023_402305

/-- Given a total amount divided among three persons, prove the ratio of the first person's share to the second person's share. -/
theorem ratio_of_shares (total : ℕ) (r_share : ℕ) (q_to_r_ratio : Rat) :
  total = 1210 →
  r_share = 400 →
  q_to_r_ratio = 9 / 10 →
  ∃ (p_share q_share : ℕ),
    p_share + q_share + r_share = total ∧
    q_share = (q_to_r_ratio * r_share).num ∧
    p_share * 4 = q_share * 5 :=
by sorry

end ratio_of_shares_l4023_402305


namespace largest_divisor_of_expression_l4023_402365

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  ∃ (k : ℤ), (8*x + 2) * (8*x + 4) * (4*x + 2) = 240 * k ∧
  ∀ (m : ℤ), m > 240 → ∃ (y : ℤ), Even y ∧ ¬∃ (l : ℤ), (8*y + 2) * (8*y + 4) * (4*y + 2) = m * l :=
sorry

end largest_divisor_of_expression_l4023_402365


namespace vector_magnitude_l4023_402304

theorem vector_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a - b = (Real.sqrt 3, Real.sqrt 2) →
  ‖a + 2 • b‖ = Real.sqrt 17 := by
sorry

end vector_magnitude_l4023_402304


namespace complex_multiplication_complex_division_l4023_402346

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem 1
theorem complex_multiplication :
  (4 - i) * (6 + 2 * i^3) = 22 - 14 * i :=
by sorry

-- Theorem 2
theorem complex_division :
  (5 * (4 + i)^2) / (i * (2 + i)) = 1 - 38 * i :=
by sorry

end complex_multiplication_complex_division_l4023_402346


namespace problem_solution_l4023_402398

theorem problem_solution :
  ∀ (a b c : ℕ+) (x y z : ℤ),
    x = -2272 →
    y = 1000 + 100 * c.val + 10 * b.val + a.val →
    z = 1 →
    a.val * x + b.val * y + c.val * z = 1 →
    a < b →
    b < c →
    y = 1987 := by
  sorry

end problem_solution_l4023_402398


namespace nineteenth_row_red_squares_l4023_402316

/-- Represents the number of squares in the nth row of a stair-step figure -/
def num_squares (n : ℕ) : ℕ := 3 * n - 1

/-- Represents the number of red squares in the nth row of a stair-step figure -/
def num_red_squares (n : ℕ) : ℕ := (num_squares n) / 2

theorem nineteenth_row_red_squares :
  num_red_squares 19 = 28 := by sorry

end nineteenth_row_red_squares_l4023_402316


namespace inequality_equivalence_l4023_402389

theorem inequality_equivalence (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by sorry

end inequality_equivalence_l4023_402389


namespace window_area_ratio_l4023_402348

theorem window_area_ratio :
  ∀ (ad ab : ℝ),
  ad / ab = 4 / 3 →
  ab = 36 →
  let r := ab / 2
  let rectangle_area := ad * ab
  let semicircles_area := π * r^2
  rectangle_area / semicircles_area = 16 / (3 * π) := by
sorry

end window_area_ratio_l4023_402348


namespace subtract_inequality_preserves_order_l4023_402343

theorem subtract_inequality_preserves_order (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end subtract_inequality_preserves_order_l4023_402343


namespace probability_one_class_no_spot_l4023_402315

/-- The number of spots for top students -/
def num_spots : ℕ := 6

/-- The number of classes -/
def num_classes : ℕ := 3

/-- The number of ways to distribute spots such that exactly one class doesn't receive a spot -/
def favorable_outcomes : ℕ := (num_classes.choose 2) * ((num_spots - 1).choose 1)

/-- The total number of ways to distribute spots among classes -/
def total_outcomes : ℕ := 
  (num_classes.choose 1) + 
  (num_classes.choose 2) * ((num_spots - 1).choose 1) + 
  (num_classes.choose 3) * ((num_spots - 1).choose 2)

/-- The probability that exactly one class does not receive a spot -/
theorem probability_one_class_no_spot : 
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 28 := by
  sorry

end probability_one_class_no_spot_l4023_402315


namespace intersection_sum_zero_l4023_402371

/-- The sum of x-coordinates and y-coordinates of the intersection points of two parabolas -/
theorem intersection_sum_zero (x y : ℝ → ℝ) : 
  (∀ t, y t = (x t - 2)^2) →
  (∀ t, x t + 3 = (y t + 2)^2) →
  (∃ a b c d : ℝ, 
    (y a = (x a - 2)^2 ∧ x a + 3 = (y a + 2)^2) ∧
    (y b = (x b - 2)^2 ∧ x b + 3 = (y b + 2)^2) ∧
    (y c = (x c - 2)^2 ∧ x c + 3 = (y c + 2)^2) ∧
    (y d = (x d - 2)^2 ∧ x d + 3 = (y d + 2)^2) ∧
    (∀ t, y t = (x t - 2)^2 ∧ x t + 3 = (y t + 2)^2 → t = a ∨ t = b ∨ t = c ∨ t = d)) →
  x a + x b + x c + x d + y a + y b + y c + y d = 0 := by
sorry

end intersection_sum_zero_l4023_402371


namespace marbles_difference_l4023_402317

def initial_marbles : ℕ := 7
def lost_marbles : ℕ := 8
def found_marbles : ℕ := 10

theorem marbles_difference : found_marbles - lost_marbles = 2 := by
  sorry

end marbles_difference_l4023_402317


namespace isosceles_triangle_side_length_l4023_402353

theorem isosceles_triangle_side_length 
  (equilateral_side : ℝ) 
  (isosceles_base : ℝ) 
  (equilateral_area : ℝ) 
  (isosceles_area : ℝ) :
  equilateral_side = 1 →
  isosceles_base = 1/3 →
  equilateral_area = Real.sqrt 3 / 4 →
  isosceles_area = equilateral_area / 3 →
  ∃ (isosceles_side : ℝ), 
    isosceles_side = Real.sqrt 3 / 3 ∧ 
    isosceles_side^2 = (isosceles_base/2)^2 + (2 * isosceles_area / isosceles_base)^2 :=
by sorry

end isosceles_triangle_side_length_l4023_402353


namespace trig_values_equal_for_same_terminal_side_l4023_402388

-- Define what it means for two angles to have the same terminal side
def same_terminal_side (α β : Real) : Prop := sorry

-- Define a general trigonometric function
def trig_function (α : Real) : Real := sorry

theorem trig_values_equal_for_same_terminal_side :
  ∀ (α β : Real) (f : Real → Real),
  same_terminal_side α β →
  f = trig_function →
  f α = f β :=
sorry

end trig_values_equal_for_same_terminal_side_l4023_402388


namespace A_intersect_B_l4023_402391

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end A_intersect_B_l4023_402391


namespace triangle_properties_l4023_402324

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  b = c →
  2 * Real.sin B = Real.sqrt 3 * Real.sin A →
  0 < B →
  B < π / 2 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  (Real.sin B = Real.sqrt 6 / 3) ∧
  (Real.cos (2 * B + π / 3) = -(1 + 2 * Real.sqrt 6) / 6) ∧
  (b = 2 → (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 2 / 3) :=
by sorry

end triangle_properties_l4023_402324


namespace distance_to_focus_is_three_l4023_402331

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between a point and a vertical line -/
def distanceToVerticalLine (P : Point) (x₀ : ℝ) : ℝ :=
  |P.x - x₀|

/-- Check if a point is on the parabola -/
def isOnParabola (P : Point) (p : Parabola) : Prop :=
  P.y^2 = 4 * p.a * P.x

/-- Distance from a point to the focus of the parabola -/
noncomputable def distanceToFocus (P : Point) (p : Parabola) : ℝ :=
  sorry

/-- Main theorem -/
theorem distance_to_focus_is_three
  (p : Parabola)
  (P : Point)
  (h_on_parabola : isOnParabola P p)
  (h_distance : distanceToVerticalLine P (-3) = 5)
  : distanceToFocus P p = 3 := by
  sorry

end distance_to_focus_is_three_l4023_402331


namespace field_trip_attendance_l4023_402307

theorem field_trip_attendance :
  let num_vans : ℕ := 6
  let num_buses : ℕ := 8
  let people_per_van : ℕ := 6
  let people_per_bus : ℕ := 18
  let total_people : ℕ := num_vans * people_per_van + num_buses * people_per_bus
  total_people = 180 := by
sorry

end field_trip_attendance_l4023_402307


namespace difference_between_number_and_fraction_l4023_402333

theorem difference_between_number_and_fraction (x : ℝ) (h : x = 155) : x - (3/5 * x) = 62 := by
  sorry

end difference_between_number_and_fraction_l4023_402333


namespace triangle_side_length_l4023_402349

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 10 →
  b = 5 * Real.sqrt 6 := by
sorry

end triangle_side_length_l4023_402349


namespace sin_minus_cos_105_deg_l4023_402379

theorem sin_minus_cos_105_deg : 
  Real.sin (105 * π / 180) - Real.cos (105 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end sin_minus_cos_105_deg_l4023_402379


namespace roberto_final_salary_l4023_402360

/-- Calculates the final salary after raises, bonus, and taxes -/
def final_salary (starting_salary : ℝ) (first_raise_percent : ℝ) (second_raise_percent : ℝ) (bonus : ℝ) (tax_rate : ℝ) : ℝ :=
  let previous_salary := starting_salary * (1 + first_raise_percent)
  let current_salary := previous_salary * (1 + second_raise_percent)
  let total_income := current_salary + bonus
  let taxes := total_income * tax_rate
  total_income - taxes

/-- Theorem stating that Roberto's final salary is $104,550 -/
theorem roberto_final_salary :
  final_salary 80000 0.4 0.2 5000 0.25 = 104550 := by
  sorry

end roberto_final_salary_l4023_402360


namespace larger_part_of_sum_and_product_l4023_402342

theorem larger_part_of_sum_and_product (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ x + y = 20 ∧ x * y = 96 → max x y = 12 := by
  sorry

end larger_part_of_sum_and_product_l4023_402342


namespace continuous_stripe_probability_l4023_402320

/-- Represents the orientation of a stripe on a cube face -/
inductive StripeOrientation
  | EdgeToEdge1
  | EdgeToEdge2
  | Diagonal1
  | Diagonal2

/-- Represents a cube with stripes on its faces -/
structure StripedCube :=
  (faces : Fin 6 → StripeOrientation)

/-- Checks if a given StripedCube has a continuous stripe encircling it -/
def hasContinuousStripe (cube : StripedCube) : Bool :=
  sorry

/-- The total number of possible stripe combinations -/
def totalCombinations : Nat :=
  4^6

/-- The number of stripe combinations that result in a continuous stripe -/
def favorableCombinations : Nat :=
  3 * 4

/-- The probability of a continuous stripe encircling the cube -/
def probabilityOfContinuousStripe : Rat :=
  favorableCombinations / totalCombinations

theorem continuous_stripe_probability :
  probabilityOfContinuousStripe = 3 / 1024 :=
sorry

end continuous_stripe_probability_l4023_402320


namespace f_monotone_decreasing_implies_a_range_l4023_402327

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + (4*a - 3)*x + 3*a
  else if 0 ≤ x ∧ x < Real.pi/2 then -Real.sin x
  else 0  -- undefined for x ≥ π/2

/-- The domain of f(x) -/
def dom (x : ℝ) : Prop := x < Real.pi/2

/-- f(x) is monotonically decreasing in its domain -/
def monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, dom x → dom y → x < y → f a x > f a y

/-- Theorem: If f(x) is monotonically decreasing, then a ∈ [0, 4/3] -/
theorem f_monotone_decreasing_implies_a_range (a : ℝ) :
  monotone_decreasing a → 0 ≤ a ∧ a ≤ 4/3 := by sorry

end f_monotone_decreasing_implies_a_range_l4023_402327


namespace linear_equation_condition_l4023_402390

/-- Given that (a-3)x^|a-2| + 4 = 0 is a linear equation in x and a-3 ≠ 0, prove that a = 1 -/
theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k, (a - 3) * x^(|a - 2|) + 4 = k * x + 4) ∧ 
  (a - 3 ≠ 0) → 
  a = 1 := by sorry

end linear_equation_condition_l4023_402390


namespace compute_expression_l4023_402347

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16) ^ 2 = 16 := by
  sorry

end compute_expression_l4023_402347


namespace probability_nine_matches_zero_l4023_402310

/-- A matching problem with n pairs -/
structure MatchingProblem (n : ℕ) where
  /-- The number of pairs to match -/
  pairs : ℕ
  /-- Assertion that the number of pairs is n -/
  pairs_eq : pairs = n

/-- The probability of correctly matching exactly k pairs in a matching problem with n pairs by random selection -/
noncomputable def probability_exact_matches (n k : ℕ) (problem : MatchingProblem n) : ℝ :=
  sorry

/-- Theorem: In a matching problem with 10 pairs, the probability of correctly matching exactly 9 pairs by random selection is 0 -/
theorem probability_nine_matches_zero :
  ∀ (problem : MatchingProblem 10), probability_exact_matches 10 9 problem = 0 :=
sorry

end probability_nine_matches_zero_l4023_402310


namespace largest_number_proof_l4023_402318

theorem largest_number_proof (a b : ℕ+) 
  (hcf_cond : Nat.gcd a b = 42)
  (lcm_cond : Nat.lcm a b = 42 * 11 * 12) :
  max a b = 504 := by
sorry

end largest_number_proof_l4023_402318


namespace max_degree_polynomial_l4023_402377

theorem max_degree_polynomial (p : ℕ) (hp : Nat.Prime p) :
  ∃ (d : ℕ), d = p - 2 ∧
  (∃ (T : Polynomial ℤ), (Polynomial.degree T = d) ∧
    (∀ (m n : ℤ), T.eval m ≡ T.eval n [ZMOD p] → m ≡ n [ZMOD p])) ∧
  (∀ (d' : ℕ), d' > d →
    ¬∃ (T : Polynomial ℤ), (Polynomial.degree T = d') ∧
      (∀ (m n : ℤ), T.eval m ≡ T.eval n [ZMOD p] → m ≡ n [ZMOD p])) := by
  sorry

end max_degree_polynomial_l4023_402377


namespace number_of_girls_l4023_402335

theorem number_of_girls (total_children happy_children sad_children neutral_children boys happy_boys sad_girls neutral_boys : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : boys = 22)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 10)
  (h9 : happy_children + sad_children + neutral_children = total_children)
  : total_children - boys = 38 := by
  sorry

end number_of_girls_l4023_402335


namespace average_of_data_l4023_402332

def data : List ℝ := [2, 5, 5, 6, 7]

theorem average_of_data : (data.sum / data.length : ℝ) = 5 := by
  sorry

end average_of_data_l4023_402332


namespace section_plane_angle_cosine_l4023_402355

/-- Regular hexagonal pyramid with given properties -/
structure HexagonalPyramid where
  -- Base side length
  a : ℝ
  -- Distance from apex to section plane
  d : ℝ
  -- Base is a regular hexagon
  is_regular_hexagon : a > 0
  -- Section plane properties
  section_plane_properties : True
  -- Given distance
  distance_constraint : d = 1
  -- Given base side length
  base_side_length : a = 2

/-- The angle between the section plane and the base plane -/
def section_angle (pyramid : HexagonalPyramid) : ℝ := sorry

/-- Theorem stating the cosine of the angle between the section plane and base plane -/
theorem section_plane_angle_cosine (pyramid : HexagonalPyramid) : 
  Real.cos (section_angle pyramid) = 3/4 := by sorry

end section_plane_angle_cosine_l4023_402355


namespace sequence_fixed_points_l4023_402372

theorem sequence_fixed_points 
  (a b c d : ℝ) 
  (h1 : c ≠ 0) 
  (h2 : a * d - b * c ≠ 0) 
  (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n (n + 1) = (a * a_n n + b) / (c * a_n n + d)) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (a * x₁ + b) / (c * x₁ + d) = x₁ ∧ 
    (a * x₂ + b) / (c * x₂ + d) = x₂ →
    ∀ n, (a_n (n + 1) - x₁) / (a_n (n + 1) - x₂) = 
         ((a - c * x₁) / (a - c * x₂)) * ((a_n n - x₁) / (a_n n - x₂))) ∧
  (∃ x₀, (a * x₀ + b) / (c * x₀ + d) = x₀ ∧ a ≠ -d →
    ∀ n, 1 / (a_n (n + 1) - x₀) = (2 * c) / (a + d) + 1 / (a_n n - x₀)) :=
by sorry

end sequence_fixed_points_l4023_402372


namespace quadratic_equation_completion_square_l4023_402330

theorem quadratic_equation_completion_square :
  ∃ (d e : ℤ), (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + d)^2 = e) ∧ d + e = 10 := by
  sorry

end quadratic_equation_completion_square_l4023_402330


namespace hanks_route_length_l4023_402311

theorem hanks_route_length :
  ∀ (route_length : ℝ) (monday_speed tuesday_speed : ℝ) (time_diff : ℝ),
    monday_speed = 70 →
    tuesday_speed = 75 →
    time_diff = 1/30 →
    route_length / monday_speed - route_length / tuesday_speed = time_diff →
    route_length = 35 := by
  sorry

end hanks_route_length_l4023_402311


namespace hyperbola_focus_to_parabola_l4023_402313

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16*x

/-- 
Given a hyperbola with equation x^2 - y^2/3 = 1 and right focus F(2,0),
the standard equation of the parabola with focus F is y^2 = 16x.
-/
theorem hyperbola_focus_to_parabola :
  ∀ x y : ℝ, hyperbola x y → parabola x y :=
sorry

end hyperbola_focus_to_parabola_l4023_402313


namespace max_value_of_s_l4023_402380

-- Define the function s
def s (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem max_value_of_s :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x y : ℝ), s x y ≤ M :=
sorry

end max_value_of_s_l4023_402380


namespace units_digit_sum_powers_l4023_402366

theorem units_digit_sum_powers : (2^20 + 3^21 + 7^20) % 10 = 0 := by
  sorry

end units_digit_sum_powers_l4023_402366


namespace unique_multiple_of_72_l4023_402341

def is_multiple_of_72 (n : ℕ) : Prop := ∃ k : ℕ, n = 72 * k

def is_form_a679b (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b

theorem unique_multiple_of_72 :
  ∀ n : ℕ, is_form_a679b n ∧ is_multiple_of_72 n ↔ n = 36792 :=
by sorry

end unique_multiple_of_72_l4023_402341


namespace adam_apple_purchase_l4023_402321

/-- The total quantity of apples Adam bought over three days -/
def total_apples (monday_apples : ℝ) : ℝ :=
  let tuesday_apples := monday_apples * 3.2
  let wednesday_apples := tuesday_apples * 1.05
  monday_apples + tuesday_apples + wednesday_apples

/-- Theorem stating the total quantity of apples Adam bought -/
theorem adam_apple_purchase :
  total_apples 15.5 = 117.18 := by
  sorry

end adam_apple_purchase_l4023_402321


namespace equation_solution_l4023_402396

theorem equation_solution : ∃! (x : ℝ), x ≠ 0 ∧ (6 * x)^18 = (12 * x)^9 ∧ x = 1/3 := by
  sorry

end equation_solution_l4023_402396


namespace sin_double_angle_shift_graph_shift_equivalent_graphs_l4023_402361

theorem sin_double_angle_shift (x : ℝ) :
  2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * (x + π / 6)) := by sorry

theorem graph_shift (x : ℝ) :
  2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * x + π / 3) := by sorry

theorem equivalent_graphs :
  ∀ x : ℝ, 2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * (x + π / 6)) := by sorry

end sin_double_angle_shift_graph_shift_equivalent_graphs_l4023_402361


namespace isosceles_triangle_perimeter_l4023_402344

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 4 ∧ b = 8 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 8 ∧ c = 4) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 20 :=
by sorry


end isosceles_triangle_perimeter_l4023_402344


namespace total_goals_is_fifteen_l4023_402394

def soccer_match_goals : ℕ := by
  -- Define the goals scored by The Kickers in the first period
  let kickers_first_period : ℕ := 2

  -- Define the goals scored by The Kickers in the second period
  let kickers_second_period : ℕ := 2 * kickers_first_period

  -- Define the goals scored by The Spiders in the first period
  let spiders_first_period : ℕ := kickers_first_period / 2

  -- Define the goals scored by The Spiders in the second period
  let spiders_second_period : ℕ := 2 * kickers_second_period

  -- Calculate the total goals
  let total_goals : ℕ := kickers_first_period + kickers_second_period + 
                         spiders_first_period + spiders_second_period

  -- Prove that the total goals equal 15
  have : total_goals = 15 := by sorry

  exact total_goals

-- Theorem stating that the total number of goals is 15
theorem total_goals_is_fifteen : soccer_match_goals = 15 := by sorry

end total_goals_is_fifteen_l4023_402394


namespace convergence_trap_equivalence_l4023_402383

open Set Filter Topology Metric

variable {X : Type*} [MetricSpace X]
variable (x : ℕ → X) (a : X)

def is_trap (s : Set X) (x : ℕ → X) : Prop :=
  ∃ N, ∀ n ≥ N, x n ∈ s

theorem convergence_trap_equivalence :
  (Tendsto x atTop (𝓝 a)) ↔
  (∀ ε > 0, is_trap (ball a ε) x) :=
sorry

end convergence_trap_equivalence_l4023_402383


namespace speed_increase_ratio_l4023_402303

theorem speed_increase_ratio (v : ℝ) (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end speed_increase_ratio_l4023_402303


namespace brother_twice_sister_age_l4023_402334

theorem brother_twice_sister_age (brother_age_2010 sister_age_2010 : ℕ) : 
  brother_age_2010 = 16 →
  sister_age_2010 = 10 →
  ∃ (year : ℕ), year = 2006 ∧ 
    brother_age_2010 - (2010 - year) = 2 * (sister_age_2010 - (2010 - year)) :=
by sorry

end brother_twice_sister_age_l4023_402334


namespace expense_equalization_l4023_402350

/-- Given three people's expenses A, B, and C, where A < B < C, 
    prove that the amount the person who paid A needs to give to each of the others 
    to equalize the costs is (B + C - 2A) / 3 -/
theorem expense_equalization (A B C : ℝ) (h1 : A < B) (h2 : B < C) :
  let total := A + B + C
  let equal_share := total / 3
  let amount_to_give := equal_share - A
  amount_to_give = (B + C - 2 * A) / 3 := by
  sorry

end expense_equalization_l4023_402350


namespace creature_perimeter_l4023_402399

/-- The perimeter of a circular creature with an open mouth -/
theorem creature_perimeter (r : ℝ) (central_angle : ℝ) : 
  r = 2 → central_angle = 270 → 
  (central_angle / 360) * (2 * π * r) + 2 * r = 3 * π + 4 :=
by sorry

end creature_perimeter_l4023_402399


namespace triangle_perimeter_l4023_402354

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 3 → Odd c → a + b > c → b + c > a → c + a > b → a + b + c = 8 := by
  sorry

end triangle_perimeter_l4023_402354


namespace parabola_circle_intersection_l4023_402336

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- The parabola equation -/
def parabola_eq (x y p : ℝ) : Prop := y^2 = 2*p*x

/-- The directrix equation of the parabola -/
def directrix_eq (x p : ℝ) : Prop := x = -p/2

/-- The length of the line segment cut by the circle on the directrix -/
def segment_length (p : ℝ) : ℝ := 4

/-- The theorem to be proved -/
theorem parabola_circle_intersection (p : ℝ) 
  (h_p_pos : p > 0) 
  (h_segment : segment_length p = 4) : p = 2 := by
  sorry

end parabola_circle_intersection_l4023_402336


namespace binary_representation_of_41_l4023_402362

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- The binary representation of 41 -/
def binary41 : List ℕ := [1, 0, 1, 0, 0, 1]

/-- Theorem stating that the binary representation of 41 is [1, 0, 1, 0, 0, 1] -/
theorem binary_representation_of_41 : toBinary 41 = binary41 := by
  sorry

end binary_representation_of_41_l4023_402362


namespace ones_count_l4023_402384

theorem ones_count (hundreds tens total : ℕ) (h1 : hundreds = 3) (h2 : tens = 8) (h3 : total = 383) :
  total - (hundreds * 100 + tens * 10) = 3 := by
  sorry

end ones_count_l4023_402384


namespace one_non_negative_root_l4023_402368

theorem one_non_negative_root (a : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ (x = a + Real.sqrt (a^2 - 4*a + 3) ∨ x = a - Real.sqrt (a^2 - 4*a + 3)) ∧
   ¬∃ y : ℝ, y ≠ x ∧ y ≥ 0 ∧ (y = a + Real.sqrt (a^2 - 4*a + 3) ∨ y = a - Real.sqrt (a^2 - 4*a + 3))) ↔ 
  ((3/4 ≤ a ∧ a < 1) ∨ (a > 3) ∨ (0 < a ∧ a < 3/4)) :=
by sorry

end one_non_negative_root_l4023_402368


namespace sequence_a_property_l4023_402312

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | (n+1) => sequence_a n + (sequence_a n)^2 / 2023

theorem sequence_a_property : sequence_a 2023 < 1 ∧ 1 < sequence_a 2024 := by
  sorry

end sequence_a_property_l4023_402312


namespace proportional_segments_l4023_402337

/-- Given four proportional line segments a, b, c, d, where b = 3, c = 4, and d = 6,
    prove that the length of line segment a is 2. -/
theorem proportional_segments (a b c d : ℝ) 
  (h_prop : a / b = c / d)
  (h_b : b = 3)
  (h_c : c = 4)
  (h_d : d = 6) :
  a = 2 := by
  sorry

end proportional_segments_l4023_402337


namespace student_count_pedro_grade_count_l4023_402329

/-- If a student is ranked both n-th best and n-th worst in a grade,
    then the total number of students in that grade is 2n - 1. -/
theorem student_count (n : ℕ) (h : n > 0) :
  ∃ (total : ℕ), total = 2 * n - 1 := by sorry

/-- There are 59 students in Pedro's grade. -/
theorem pedro_grade_count :
  ∃ (total : ℕ), total = 59 := by
  apply student_count 30
  norm_num

end student_count_pedro_grade_count_l4023_402329


namespace sum_seven_probability_l4023_402314

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to get a sum of 7 when tossing two dice -/
def favorableOutcomes : ℕ := 6

/-- The probability of getting a sum of 7 when tossing two dice -/
def probabilitySumSeven : ℚ := favorableOutcomes / totalOutcomes

theorem sum_seven_probability :
  probabilitySumSeven = 1 / 6 := by
  sorry

end sum_seven_probability_l4023_402314


namespace theta_range_l4023_402301

theorem theta_range (θ : Real) : 
  (∀ x : Real, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) → 
  π/12 < θ ∧ θ < 5*π/12 := by
sorry

end theta_range_l4023_402301


namespace expression_equality_l4023_402367

theorem expression_equality (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 2*x + y/2 ≠ 0) : 
  (2*x + y/2)⁻¹ * ((2*x)⁻¹ + (y/2)⁻¹) = (x*y)⁻¹ := by
  sorry

end expression_equality_l4023_402367


namespace divisor_sum_360_l4023_402375

/-- Sum of positive divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem divisor_sum_360 (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 360 → i + j + k = 6 := by
  sorry

end divisor_sum_360_l4023_402375


namespace modular_inverse_of_5_mod_23_l4023_402387

theorem modular_inverse_of_5_mod_23 :
  ∃ a : ℕ, a ≤ 22 ∧ (5 * a) % 23 = 1 ∧ a = 14 := by
  sorry

end modular_inverse_of_5_mod_23_l4023_402387


namespace lunks_needed_for_dozen_apples_l4023_402393

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks : ℚ := 4 / 7

/-- Exchange rate between kunks and apples -/
def kunks_to_apples : ℚ := 5 / 3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 12

/-- Theorem stating the number of lunks needed to buy 12 apples -/
theorem lunks_needed_for_dozen_apples :
  ⌈(apples_to_buy : ℚ) / kunks_to_apples / lunks_to_kunks⌉ = 14 := by sorry

end lunks_needed_for_dozen_apples_l4023_402393


namespace cube_difference_equals_negative_875_l4023_402328

theorem cube_difference_equals_negative_875 (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 2 * x + y = 20) : 
  x^3 - y^3 = -875 := by sorry

end cube_difference_equals_negative_875_l4023_402328


namespace f_is_quadratic_l4023_402386

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l4023_402386


namespace platform_length_l4023_402363

/-- Given a train and platform with specific properties, prove the length of the platform -/
theorem platform_length 
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 36)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 300 := by
sorry


end platform_length_l4023_402363


namespace factorization_of_2x_squared_minus_8_l4023_402319

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l4023_402319


namespace unique_zero_of_exp_plus_linear_l4023_402356

/-- The function f(x) = e^x + 3x has exactly one zero. -/
theorem unique_zero_of_exp_plus_linear : ∃! x : ℝ, Real.exp x + 3 * x = 0 := by sorry

end unique_zero_of_exp_plus_linear_l4023_402356


namespace sum_has_no_real_roots_l4023_402364

/-- A quadratic polynomial with integer coefficients. -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate for an acceptable quadratic polynomial. -/
def is_acceptable (p : QuadraticPolynomial) : Prop :=
  abs p.a ≤ 2013 ∧ abs p.b ≤ 2013 ∧ abs p.c ≤ 2013 ∧
  ∃ (r₁ r₂ : ℤ), p.a * r₁^2 + p.b * r₁ + p.c = 0 ∧ p.a * r₂^2 + p.b * r₂ + p.c = 0

/-- The set of all acceptable quadratic polynomials. -/
def acceptable_polynomials : Set QuadraticPolynomial :=
  {p : QuadraticPolynomial | is_acceptable p}

/-- The sum of all acceptable quadratic polynomials. -/
noncomputable def sum_of_acceptable_polynomials : QuadraticPolynomial :=
  sorry

/-- Theorem stating that the sum of all acceptable quadratic polynomials has no real roots. -/
theorem sum_has_no_real_roots :
  ∃ (A C : ℤ), A > 0 ∧ C > 0 ∧
  sum_of_acceptable_polynomials.a = A ∧
  sum_of_acceptable_polynomials.b = 0 ∧
  sum_of_acceptable_polynomials.c = C :=
sorry

end sum_has_no_real_roots_l4023_402364


namespace maggie_bought_ten_magazines_l4023_402306

/-- The number of science magazines Maggie bought -/
def num_magazines : ℕ := 10

/-- The number of books Maggie bought -/
def num_books : ℕ := 10

/-- The cost of each book in dollars -/
def book_cost : ℕ := 15

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 2

/-- The total amount Maggie spent in dollars -/
def total_spent : ℕ := 170

/-- Proof that Maggie bought 10 science magazines -/
theorem maggie_bought_ten_magazines :
  num_magazines = 10 ∧
  num_books * book_cost + num_magazines * magazine_cost = total_spent :=
sorry

end maggie_bought_ten_magazines_l4023_402306


namespace every_algorithm_relies_on_sequential_structure_l4023_402357

/-- Represents the basic structures used in algorithms -/
inductive AlgorithmStructure
  | Logical
  | Conditional
  | Loop
  | Sequential

/-- Represents an algorithm with its characteristics -/
structure Algorithm where
  input : Nat
  output : Nat
  steps : List AlgorithmStructure
  isDefinite : Bool
  isFinite : Bool
  isEffective : Bool

/-- Theorem stating that every algorithm relies on the Sequential structure -/
theorem every_algorithm_relies_on_sequential_structure (a : Algorithm) :
  AlgorithmStructure.Sequential ∈ a.steps :=
sorry

end every_algorithm_relies_on_sequential_structure_l4023_402357
