import Mathlib

namespace bisecting_line_projection_ratio_l389_38974

/-- A convex polygon type -/
structure ConvexPolygon where
  -- Add necessary fields

/-- A line type -/
structure Line where
  -- Add necessary fields

/-- Represents the projection of a polygon onto a line -/
structure Projection where
  -- Add necessary fields

/-- Checks if a line bisects the area of a polygon -/
def bisects_area (l : Line) (p : ConvexPolygon) : Prop :=
  sorry

/-- Gets the projection of a polygon onto a line perpendicular to the given line -/
def get_perpendicular_projection (p : ConvexPolygon) (l : Line) : Projection :=
  sorry

/-- Gets the ratio of the segments created by a line on a projection -/
def projection_ratio (proj : Projection) (l : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem bisecting_line_projection_ratio 
  (p : ConvexPolygon) (l : Line) :
  bisects_area l p →
  projection_ratio (get_perpendicular_projection p l) l ≤ 1 + Real.sqrt 2 :=
by
  sorry

end bisecting_line_projection_ratio_l389_38974


namespace football_game_spectators_l389_38979

theorem football_game_spectators (total_wristbands : ℕ) 
  (wristbands_per_person : ℕ) (h1 : total_wristbands = 234) 
  (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end football_game_spectators_l389_38979


namespace untouched_produce_count_l389_38965

/-- The number of untouched tomatoes and cucumbers after processing -/
def untouched_produce (tomato_plants : ℕ) (tomatoes_per_plant : ℕ) (cucumbers : ℕ) : ℕ :=
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let dried_tomatoes := (2 * total_tomatoes) / 3
  let remaining_tomatoes := total_tomatoes - dried_tomatoes
  let sauce_tomatoes := remaining_tomatoes / 2
  let untouched_tomatoes := remaining_tomatoes - sauce_tomatoes
  let pickled_cucumbers := cucumbers / 4
  let untouched_cucumbers := cucumbers - pickled_cucumbers
  untouched_tomatoes + untouched_cucumbers

/-- Theorem stating the number of untouched produce given the conditions -/
theorem untouched_produce_count :
  untouched_produce 50 15 25 = 143 := by
  sorry


end untouched_produce_count_l389_38965


namespace money_distribution_l389_38982

theorem money_distribution (a b c total : ℕ) : 
  a + b + c = total →
  2 * b = 3 * a →
  4 * b = 3 * c →
  b = 600 →
  total = 1800 := by
sorry

end money_distribution_l389_38982


namespace correct_negation_l389_38967

theorem correct_negation :
  (¬ ∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) := by sorry

end correct_negation_l389_38967


namespace contest_possible_orders_l389_38941

/-- The number of questions in the contest -/
def num_questions : ℕ := 10

/-- The number of possible orders to answer the questions -/
def num_possible_orders : ℕ := 512

/-- Theorem stating that the number of possible orders is correct -/
theorem contest_possible_orders :
  (2 ^ (num_questions - 1) : ℕ) = num_possible_orders := by
  sorry

end contest_possible_orders_l389_38941


namespace pants_discount_percentage_l389_38972

theorem pants_discount_percentage (cost : ℝ) (profit_percentage : ℝ) (marked_price : ℝ) :
  cost = 80 →
  profit_percentage = 0.3 →
  marked_price = 130 →
  let profit := cost * profit_percentage
  let selling_price := cost + profit
  let discount := marked_price - selling_price
  let discount_percentage := (discount / marked_price) * 100
  discount_percentage = 20 := by sorry

end pants_discount_percentage_l389_38972


namespace birds_on_fence_l389_38902

theorem birds_on_fence (num_birds : ℕ) (h : num_birds = 20) : 
  2 * num_birds + 10 = 50 := by
  sorry

end birds_on_fence_l389_38902


namespace collinear_points_b_value_l389_38940

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that given the collinearity of the points (4, -10), (-b + 4, 6), and (3b + 6, 4),
    the value of b must be -16/31 -/
theorem collinear_points_b_value :
  ∀ b : ℚ, collinear 4 (-10) (-b + 4) 6 (3*b + 6) 4 → b = -16/31 := by
  sorry

end collinear_points_b_value_l389_38940


namespace geometric_sequence_sum_l389_38932

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (a 1 + a 2 = 40) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = 135) := by
  sorry

end geometric_sequence_sum_l389_38932


namespace shop_dimension_example_l389_38901

/-- Calculates the dimension of a shop given its monthly rent and annual rent per square foot. -/
def shopDimension (monthlyRent : ℕ) (annualRentPerSqFt : ℕ) : ℕ :=
  (monthlyRent * 12) / annualRentPerSqFt

/-- Theorem stating that for a shop with a monthly rent of 1300 and an annual rent per square foot of 156, the dimension is 100 square feet. -/
theorem shop_dimension_example : shopDimension 1300 156 = 100 := by
  sorry

end shop_dimension_example_l389_38901


namespace stating_total_seats_is_680_l389_38917

/-- 
Calculates the total number of seats in a theater given the following conditions:
- The first row has 15 seats
- Each row has 2 more seats than the previous row
- The last row has 53 seats
-/
def theaterSeats : ℕ := by
  -- Define the number of seats in the first row
  let firstRow : ℕ := 15
  -- Define the increase in seats per row
  let seatIncrease : ℕ := 2
  -- Define the number of seats in the last row
  let lastRow : ℕ := 53
  
  -- Calculate the number of rows
  let numRows : ℕ := (lastRow - firstRow) / seatIncrease + 1
  
  -- Calculate the total number of seats
  let totalSeats : ℕ := numRows * (firstRow + lastRow) / 2
  
  exact totalSeats

/-- 
Theorem stating that the total number of seats in the theater is 680
-/
theorem total_seats_is_680 : theaterSeats = 680 := by
  sorry

end stating_total_seats_is_680_l389_38917


namespace student_sequences_count_l389_38915

/-- The number of ways to select 5 students from a group of 15 students,
    where order matters and no student can be selected more than once. -/
def student_sequences : ℕ :=
  Nat.descFactorial 15 5

theorem student_sequences_count : student_sequences = 360360 := by
  sorry

end student_sequences_count_l389_38915


namespace prime_comparison_l389_38951

theorem prime_comparison (x y : ℕ) (hx : Prime x) (hy : Prime y) 
  (hlcm : Nat.lcm x y = 10) (heq : 2 * x + y = 12) : x > y := by
  sorry

end prime_comparison_l389_38951


namespace x_plus_y_value_l389_38912

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 := by
  sorry

end x_plus_y_value_l389_38912


namespace right_triangle_area_l389_38955

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- The angle at C is a right angle -/
  right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  /-- The length of hypotenuse AB is 50 -/
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 50^2
  /-- The median through A lies along the line y = x - 2 -/
  median_A : ∃ (t : ℝ), A.2 = A.1 - 2 ∧ ((B.1 + C.1) / 2 = A.1 + t) ∧ ((B.2 + C.2) / 2 = A.2 + t)
  /-- The median through B lies along the line y = 3x + 1 -/
  median_B : ∃ (t : ℝ), B.2 = 3 * B.1 + 1 ∧ ((A.1 + C.1) / 2 = B.1 + t) ∧ ((A.2 + C.2) / 2 = B.2 + 3 * t)

/-- The area of a right triangle ABC with the given properties is 3750/59 -/
theorem right_triangle_area (t : RightTriangle) : 
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 3750 / 59 :=
sorry

end right_triangle_area_l389_38955


namespace smallest_marble_count_l389_38908

theorem smallest_marble_count : ∃ N : ℕ, 
  N > 1 ∧ 
  N % 9 = 1 ∧ 
  N % 10 = 1 ∧ 
  N % 11 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 9 = 1 ∧ m % 10 = 1 ∧ m % 11 = 1 → m ≥ N) ∧
  N = 991 := by
sorry

end smallest_marble_count_l389_38908


namespace even_function_properties_l389_38973

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

-- Define what it means for f to be even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_properties (b : ℝ) 
  (h : is_even_function (f b)) :
  (b = 0) ∧ 
  (Set.Ioo 1 2 = {x | f b (x - 1) < x}) :=
by sorry

end even_function_properties_l389_38973


namespace quadratic_function_min_value_l389_38949

/-- Given a quadratic function f(x) = ax^2 + 2x + c with range [0, +∞),
    the minimum value of (a+1)/c + (c+1)/a is 4 -/
theorem quadratic_function_min_value (a c : ℝ) :
  (∀ x, a * x^2 + 2 * x + c ≥ 0) →
  a > 0 →
  c > 0 →
  (∃ x, a * x^2 + 2 * x + c = 0) →
  (a + 1) / c + (c + 1) / a ≥ 4 :=
by sorry

end quadratic_function_min_value_l389_38949


namespace complex_equation_solution_l389_38933

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = -3 + 4*I → z = 1 + 2*I := by
  sorry

end complex_equation_solution_l389_38933


namespace product_of_sums_equals_difference_l389_38911

theorem product_of_sums_equals_difference (n : ℕ) :
  (5 + 1) * (5^2 + 1^2) * (5^4 + 1^4) * (5^8 + 1^8) * (5^16 + 1^16) * (5^32 + 1^32) * (5^64 + 1^64) = 5^128 - 1^128 := by
  sorry

end product_of_sums_equals_difference_l389_38911


namespace partial_fraction_decomposition_l389_38928

theorem partial_fraction_decomposition :
  ∀ (x : ℝ) (P Q R : ℚ),
    P = -8/15 ∧ Q = -7/6 ∧ R = 27/10 →
    x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
    (x^2 - 9) / ((x - 1)*(x - 4)*(x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6) :=
by sorry

end partial_fraction_decomposition_l389_38928


namespace jellybean_count_l389_38907

theorem jellybean_count (total blue purple red orange : ℕ) : 
  total = 200 →
  blue = 14 →
  purple = 26 →
  red = 120 →
  total = blue + purple + red + orange →
  orange = 40 := by
  sorry

end jellybean_count_l389_38907


namespace negative_fraction_comparison_l389_38981

theorem negative_fraction_comparison : -3/5 > -3/4 := by
  sorry

end negative_fraction_comparison_l389_38981


namespace correct_inscription_l389_38950

-- Define the type for box makers
inductive BoxMaker
| Bellini
| Cellini
| Other

-- Define a box
structure Box where
  maker : BoxMaker
  inscription : String

-- Define the problem conditions
def validInscription (inscription : String) : Prop :=
  ∃ (box1 box2 : Box),
    (box1.inscription = inscription) ∧
    (box2.inscription = inscription) ∧
    (box1.maker = BoxMaker.Bellini ∧ box2.maker = BoxMaker.Bellini) ∧
    (∀ (b1 b2 : Box), b1.inscription = inscription → b2.inscription = inscription →
      ((b1.maker = BoxMaker.Bellini ∧ b2.maker = BoxMaker.Bellini) ∨
       (b1.maker = BoxMaker.Cellini ∨ b2.maker = BoxMaker.Cellini)))

-- The theorem to be proved
theorem correct_inscription :
  validInscription "Either both caskets are made by Bellini, or at least one of them is made by a member of the Cellini family" :=
sorry

end correct_inscription_l389_38950


namespace sequence_increasing_l389_38929

def a (n : ℕ) : ℚ := (n - 1) / (n + 1)

theorem sequence_increasing : ∀ n : ℕ, n ≥ 1 → a n < a (n + 1) := by
  sorry

end sequence_increasing_l389_38929


namespace complement_of_intersection_l389_38985

def U : Finset ℕ := {1, 2, 3, 4, 6}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_of_intersection (U A B : Finset ℕ) 
  (hU : U = {1, 2, 3, 4, 6})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  U \ (A ∩ B) = {1, 4, 6} := by
  sorry

end complement_of_intersection_l389_38985


namespace fruit_arrangement_unique_l389_38987

-- Define the fruits
inductive Fruit
| Apple
| Pear
| Orange
| Banana

-- Define a type for box numbers
inductive BoxNumber
| One
| Two
| Three
| Four

-- Define a function type for fruit arrangements
def Arrangement := BoxNumber → Fruit

-- Define a predicate for the correctness of labels
def LabelIncorrect (arr : Arrangement) : Prop :=
  arr BoxNumber.One ≠ Fruit.Orange ∧
  arr BoxNumber.Two ≠ Fruit.Pear ∧
  (arr BoxNumber.One = Fruit.Banana → arr BoxNumber.Three ≠ Fruit.Apple ∧ arr BoxNumber.Three ≠ Fruit.Pear) ∧
  arr BoxNumber.Four ≠ Fruit.Apple

-- Define the correct arrangement
def CorrectArrangement : Arrangement :=
  fun b => match b with
  | BoxNumber.One => Fruit.Banana
  | BoxNumber.Two => Fruit.Apple
  | BoxNumber.Three => Fruit.Orange
  | BoxNumber.Four => Fruit.Pear

-- Theorem statement
theorem fruit_arrangement_unique :
  ∀ (arr : Arrangement),
    (∀ (b : BoxNumber), ∃! (f : Fruit), arr b = f) →
    LabelIncorrect arr →
    arr = CorrectArrangement :=
sorry

end fruit_arrangement_unique_l389_38987


namespace difference_of_squares_l389_38945

theorem difference_of_squares (m : ℝ) : m^2 - 9 = (m + 3) * (m - 3) := by
  sorry

end difference_of_squares_l389_38945


namespace negative_abs_two_squared_equals_two_l389_38994

theorem negative_abs_two_squared_equals_two : (-|2|)^2 = 2 := by
  sorry

end negative_abs_two_squared_equals_two_l389_38994


namespace books_obtained_l389_38934

/-- Given an initial number of books and a final number of books,
    calculate the number of additional books obtained. -/
def additional_books (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that for the given initial and final book counts,
    the number of additional books is 23. -/
theorem books_obtained (initial : ℕ) (final : ℕ)
    (h1 : initial = 54)
    (h2 : final = 77) :
    additional_books initial final = 23 := by
  sorry

end books_obtained_l389_38934


namespace second_wheat_rate_l389_38947

-- Define the quantities and rates
def wheat1_quantity : ℝ := 30
def wheat1_rate : ℝ := 11.50
def wheat2_quantity : ℝ := 20
def profit_percentage : ℝ := 0.10
def mixture_sell_rate : ℝ := 13.86

-- Define the theorem
theorem second_wheat_rate (wheat2_rate : ℝ) : 
  wheat1_quantity * wheat1_rate + wheat2_quantity * wheat2_rate = 
  (wheat1_quantity + wheat2_quantity) * mixture_sell_rate / (1 + profit_percentage) →
  wheat2_rate = 14.25 := by
sorry

end second_wheat_rate_l389_38947


namespace sin_2alpha_value_l389_38942

theorem sin_2alpha_value (α : Real) (h : Real.tan (π/4 + α) = 2) : 
  Real.sin (2 * α) = 3/5 := by
  sorry

end sin_2alpha_value_l389_38942


namespace line_passes_through_fixed_point_l389_38914

/-- Given that a and b satisfy a + 2*b = 1, prove that the line ax + 3y + b = 0 passes through the point (1/2, -1/6) -/
theorem line_passes_through_fixed_point (a b : ℝ) (h : a + 2*b = 1) :
  a * (1/2 : ℝ) + 3 * (-1/6 : ℝ) + b = 0 := by
  sorry

end line_passes_through_fixed_point_l389_38914


namespace stratified_sample_proportion_l389_38958

/-- Calculates the number of teachers under 40 in a stratified sample -/
def teachersUnder40InSample (totalTeachers : ℕ) (under40Teachers : ℕ) (sampleSize : ℕ) : ℕ :=
  (under40Teachers * sampleSize) / totalTeachers

theorem stratified_sample_proportion 
  (totalTeachers : ℕ) 
  (under40Teachers : ℕ) 
  (over40Teachers : ℕ) 
  (sampleSize : ℕ) :
  totalTeachers = 490 →
  under40Teachers = 350 →
  over40Teachers = 140 →
  sampleSize = 70 →
  totalTeachers = under40Teachers + over40Teachers →
  teachersUnder40InSample totalTeachers under40Teachers sampleSize = 50 :=
by sorry

end stratified_sample_proportion_l389_38958


namespace favorite_numbers_exist_l389_38923

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem favorite_numbers_exist : ∃ (a b c : ℕ), 
  a * b * c = 71668 ∧ 
  a * sum_of_digits a = 10 * a ∧
  b * sum_of_digits b = 10 * b ∧
  c * sum_of_digits c = 10 * c :=
sorry

end favorite_numbers_exist_l389_38923


namespace new_average_age_with_teacher_l389_38993

theorem new_average_age_with_teacher (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℕ) :
  num_students = 40 →
  student_avg_age = 15 →
  teacher_age = 56 →
  (num_students : ℝ) * student_avg_age + teacher_age = 16 * (num_students + 1) := by
  sorry


end new_average_age_with_teacher_l389_38993


namespace quadratic_roots_expression_l389_38936

theorem quadratic_roots_expression (a b : ℝ) : 
  (a^2 - a - 1 = 0) → (b^2 - b - 1 = 0) → (3*a^2 + 2*b^2 - 3*a - 2*b = 5) := by
  sorry

end quadratic_roots_expression_l389_38936


namespace cube_root_problem_l389_38991

theorem cube_root_problem :
  ∃ (a b : ℤ) (c : ℚ),
    (5 * a - 2 : ℚ) = -27 ∧
    b = Int.floor (Real.sqrt 22) ∧
    c = -(4 : ℚ)/25 ∧
    a = -5 ∧
    b = 4 ∧
    c = -(2 : ℚ)/5 ∧
    Real.sqrt ((4 : ℚ) * a * c + 7 * b) = 6 :=
by sorry

end cube_root_problem_l389_38991


namespace derivative_of_one_plus_cos_2x_squared_l389_38948

theorem derivative_of_one_plus_cos_2x_squared (x : ℝ) :
  let y : ℝ → ℝ := λ x => (1 + Real.cos (2 * x))^2
  deriv y x = -4 * Real.sin (2 * x) - 2 * Real.sin (4 * x) := by
  sorry

end derivative_of_one_plus_cos_2x_squared_l389_38948


namespace complex_equality_implies_a_equals_three_l389_38935

theorem complex_equality_implies_a_equals_three (a : ℝ) : 
  Complex.re ((1 + a * Complex.I) * (2 - Complex.I)) = Complex.im ((1 + a * Complex.I) * (2 - Complex.I)) →
  a = 3 :=
by sorry

end complex_equality_implies_a_equals_three_l389_38935


namespace exists_modular_inverse_l389_38909

theorem exists_modular_inverse :
  ∃ n : ℤ, 21 * n ≡ 1 [ZMOD 74] := by
  sorry

end exists_modular_inverse_l389_38909


namespace jeff_scores_mean_l389_38946

def jeff_scores : List ℝ := [85, 94, 87, 93, 95, 88, 90]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℝ) = 90.2857142857 := by
  sorry

end jeff_scores_mean_l389_38946


namespace inequality_proof_l389_38960

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  a^3 / (b+c+d) + b^3 / (a+c+d) + c^3 / (a+b+d) + d^3 / (a+b+c) ≥ 1/3 := by
  sorry

end inequality_proof_l389_38960


namespace polynomial_coefficient_sum_l389_38910

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end polynomial_coefficient_sum_l389_38910


namespace polynomial_factorization_l389_38919

theorem polynomial_factorization (a x y : ℝ) :
  3 * a * x^2 - 3 * a * y^2 = 3 * a * (x + y) * (x - y) := by
  sorry

end polynomial_factorization_l389_38919


namespace volleyball_ticket_sales_l389_38921

theorem volleyball_ticket_sales (total_tickets : ℕ) (jude_tickets : ℕ) (left_tickets : ℕ) 
  (h1 : total_tickets = 100)
  (h2 : jude_tickets = 16)
  (h3 : left_tickets = 40)
  : total_tickets - left_tickets - 2 * jude_tickets - jude_tickets = jude_tickets - 4 :=
by
  sorry

end volleyball_ticket_sales_l389_38921


namespace game_strategies_l389_38978

def game_state (n : ℕ) : Prop := n > 0

def player_A_move (n m : ℕ) : Prop := n ≤ m ∧ m ≤ n^2

def player_B_move (n m : ℕ) : Prop := ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = m * p^k

def A_wins (n : ℕ) : Prop := n = 1990

def B_wins (n : ℕ) : Prop := n = 1

def A_has_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ n₀ ≥ 8

def B_has_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ n₀ ≤ 5

def no_guaranteed_winning_strategy (n₀ : ℕ) : Prop := game_state n₀ ∧ (n₀ = 6 ∨ n₀ = 7)

theorem game_strategies :
  ∀ n₀ : ℕ, game_state n₀ →
    (A_has_winning_strategy n₀ ↔ n₀ ≥ 8) ∧
    (B_has_winning_strategy n₀ ↔ n₀ ≤ 5) ∧
    (no_guaranteed_winning_strategy n₀ ↔ (n₀ = 6 ∨ n₀ = 7)) :=
  sorry

end game_strategies_l389_38978


namespace complex_absolute_value_l389_38977

theorem complex_absolute_value (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = (2*i)/(1+i) → Complex.abs (z - 2) = Real.sqrt 2 := by
  sorry

end complex_absolute_value_l389_38977


namespace rhombus_in_rectangle_perimeter_l389_38905

-- Define the points of the rectangle and rhombus
variable (I J K L E F G H : ℝ × ℝ)

-- Define the properties of the rectangle and rhombus
def is_rectangle (I J K L : ℝ × ℝ) : Prop := sorry
def is_rhombus (E F G H : ℝ × ℝ) : Prop := sorry
def inscribed (E F G H I J K L : ℝ × ℝ) : Prop := sorry
def interior_point (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem rhombus_in_rectangle_perimeter 
  (h_rectangle : is_rectangle I J K L)
  (h_rhombus : is_rhombus E F G H)
  (h_inscribed : inscribed E F G H I J K L)
  (h_E : interior_point E I J)
  (h_F : interior_point F J K)
  (h_G : interior_point G K L)
  (h_H : interior_point H L I)
  (h_IE : distance I E = 12)
  (h_EJ : distance E J = 25)
  (h_EG : distance E G = 35)
  (h_FH : distance F H = 42) :
  distance I J + distance J K + distance K L + distance L I = 110 := by
  sorry

end rhombus_in_rectangle_perimeter_l389_38905


namespace vladimir_investment_opportunity_l389_38925

/-- Represents the value of 1 kg of buckwheat in rubles -/
def buckwheat_value : ℝ := 85

/-- Represents the initial price of 1 kg of buckwheat in rubles -/
def initial_price : ℝ := 70

/-- Calculates the value after a one-year deposit at the given rate -/
def one_year_deposit (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

/-- Calculates the value after a two-year deposit at the given rate -/
def two_year_deposit (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate) * (1 + rate)

/-- Represents the annual deposit rate for 2015 -/
def rate_2015 : ℝ := 0.16

/-- Represents the annual deposit rate for 2016 -/
def rate_2016 : ℝ := 0.10

/-- Represents the two-year deposit rate starting from 2015 -/
def rate_2015_2016 : ℝ := 0.15

theorem vladimir_investment_opportunity : 
  let option1 := one_year_deposit (one_year_deposit initial_price rate_2015) rate_2016
  let option2 := two_year_deposit initial_price rate_2015_2016
  max option1 option2 > buckwheat_value := by sorry

end vladimir_investment_opportunity_l389_38925


namespace dihedral_angle_in_unit_cube_l389_38997

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculates the dihedral angle between two planes in a cube -/
def dihedralAngle (cube : Cube) : ℝ :=
  sorry

/-- Theorem: The dihedral angle between planes ABD₁ and A₁B₁C₁ in a unit cube is 60° -/
theorem dihedral_angle_in_unit_cube :
  ∀ (cube : Cube),
    (cube.A.x = 0 ∧ cube.A.y = 0 ∧ cube.A.z = 0) →
    (cube.B.x = 1 ∧ cube.B.y = 0 ∧ cube.B.z = 0) →
    (cube.C.x = 1 ∧ cube.C.y = 1 ∧ cube.C.z = 0) →
    (cube.D.x = 0 ∧ cube.D.y = 1 ∧ cube.D.z = 0) →
    (cube.E₁.x = 0 ∧ cube.E₁.y = 0 ∧ cube.E₁.z = 1) →
    (cube.B₁.x = 1 ∧ cube.B₁.y = 0 ∧ cube.B₁.z = 1) →
    (cube.C₁.x = 1 ∧ cube.C₁.y = 1 ∧ cube.C₁.z = 1) →
    (cube.D₁.x = 0 ∧ cube.D₁.y = 1 ∧ cube.D₁.z = 1) →
    dihedralAngle cube = 60 * π / 180 :=
  by sorry

end dihedral_angle_in_unit_cube_l389_38997


namespace num_chords_from_nine_points_l389_38924

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 9

/-- A function to calculate the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of distinct chords from 9 points is 36 -/
theorem num_chords_from_nine_points : 
  choose_two num_points = 36 := by sorry

end num_chords_from_nine_points_l389_38924


namespace quadratic_inequality_solution_set_l389_38963

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - (a + 2)*x + 2*a > 0}
  (a > 2 → solution_set = {x : ℝ | x < 2 ∨ x > a}) ∧
  (a = 2 → solution_set = {x : ℝ | x ≠ 2}) ∧
  (a < 2 → solution_set = {x : ℝ | x < a ∨ x > 2}) :=
by sorry

end quadratic_inequality_solution_set_l389_38963


namespace consortium_psychology_majors_l389_38975

theorem consortium_psychology_majors 
  (total : ℝ) 
  (college_A_percent : ℝ) 
  (college_B_percent : ℝ) 
  (college_C_percent : ℝ) 
  (college_A_freshmen : ℝ) 
  (college_B_freshmen : ℝ) 
  (college_C_freshmen : ℝ) 
  (college_A_liberal_arts : ℝ) 
  (college_B_liberal_arts : ℝ) 
  (college_C_liberal_arts : ℝ) 
  (college_A_psychology : ℝ) 
  (college_B_psychology : ℝ) 
  (college_C_psychology : ℝ) 
  (h1 : college_A_percent = 0.40) 
  (h2 : college_B_percent = 0.35) 
  (h3 : college_C_percent = 0.25) 
  (h4 : college_A_freshmen = 0.80) 
  (h5 : college_B_freshmen = 0.70) 
  (h6 : college_C_freshmen = 0.60) 
  (h7 : college_A_liberal_arts = 0.60) 
  (h8 : college_B_liberal_arts = 0.50) 
  (h9 : college_C_liberal_arts = 0.40) 
  (h10 : college_A_psychology = 0.50) 
  (h11 : college_B_psychology = 0.40) 
  (h12 : college_C_psychology = 0.30) : 
  (college_A_percent * college_A_freshmen * college_A_liberal_arts * college_A_psychology + 
   college_B_percent * college_B_freshmen * college_B_liberal_arts * college_B_psychology + 
   college_C_percent * college_C_freshmen * college_C_liberal_arts * college_C_psychology) * 100 = 16.3 := by
sorry

end consortium_psychology_majors_l389_38975


namespace circular_film_radius_l389_38939

/-- The radius of a circular film formed by a liquid --/
theorem circular_film_radius 
  (volume : ℝ) 
  (thickness : ℝ) 
  (radius : ℝ) 
  (h1 : volume = 320) 
  (h2 : thickness = 0.05) 
  (h3 : π * radius^2 * thickness = volume) : 
  radius = Real.sqrt (6400 / π) := by
sorry

end circular_film_radius_l389_38939


namespace all_star_seating_arrangements_l389_38971

/-- Represents the number of ways to arrange All-Stars from different teams in a row --/
def allStarArrangements (total : Nat) (team1 : Nat) (team2 : Nat) (team3 : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial team1 * Nat.factorial team2 * Nat.factorial team3

/-- Theorem stating the number of arrangements for 8 All-Stars from 3 teams --/
theorem all_star_seating_arrangements :
  allStarArrangements 8 3 3 2 = 432 := by
  sorry

#eval allStarArrangements 8 3 3 2

end all_star_seating_arrangements_l389_38971


namespace steven_more_apples_l389_38922

/-- The number of apples Steven has -/
def steven_apples : ℕ := 19

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference between Steven's apples and peaches -/
def apple_peach_difference : ℤ := steven_apples - steven_peaches

theorem steven_more_apples : apple_peach_difference = 4 := by
  sorry

end steven_more_apples_l389_38922


namespace f_even_l389_38989

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom functional_equation : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b

-- State the theorem to be proved
theorem f_even : ∀ x, f (-x) = f x := by sorry

end f_even_l389_38989


namespace trivia_team_groups_l389_38998

/-- Given a total number of students, number of students not picked, and students per group,
    calculate the number of groups formed. -/
def calculate_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) : ℕ :=
  (total_students - not_picked) / students_per_group

/-- Theorem stating that with 17 total students, 5 not picked, and 4 per group, 3 groups are formed. -/
theorem trivia_team_groups : calculate_groups 17 5 4 = 3 := by
  sorry

end trivia_team_groups_l389_38998


namespace equivalence_conditions_l389_38900

theorem equivalence_conditions (n : ℕ) :
  (∀ (a : ℕ+), n ∣ a^n - a) ↔
  (∀ (p : ℕ), Prime p → p ∣ n → (¬(p^2 ∣ n) ∧ (p - 1 ∣ n - 1))) :=
by sorry

end equivalence_conditions_l389_38900


namespace floor_length_l389_38962

/-- Represents the dimensions of a rectangular floor -/
structure FloorDimensions where
  breadth : ℝ
  length : ℝ

/-- The properties of the floor as given in the problem -/
def FloorProperties (d : FloorDimensions) : Prop :=
  d.length = 3 * d.breadth ∧ d.length * d.breadth = 156

/-- Theorem stating the length of the floor -/
theorem floor_length (d : FloorDimensions) (h : FloorProperties d) : 
  d.length = 6 * Real.sqrt 13 := by
  sorry


end floor_length_l389_38962


namespace power_product_equality_l389_38980

theorem power_product_equality (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end power_product_equality_l389_38980


namespace baker_earnings_calculation_l389_38995

def cakes_sold : ℕ := 453
def cake_price : ℕ := 12
def pies_sold : ℕ := 126
def pie_price : ℕ := 7

def baker_earnings : ℕ := cakes_sold * cake_price + pies_sold * pie_price

theorem baker_earnings_calculation : baker_earnings = 6318 := by
  sorry

end baker_earnings_calculation_l389_38995


namespace bowl_delivery_fee_l389_38920

/-- The problem of calculating the initial fee for a bowl delivery service -/
theorem bowl_delivery_fee
  (total_bowls : ℕ)
  (safe_delivery_pay : ℕ)
  (loss_penalty : ℕ)
  (lost_bowls : ℕ)
  (broken_bowls : ℕ)
  (total_payment : ℕ)
  (h1 : total_bowls = 638)
  (h2 : safe_delivery_pay = 3)
  (h3 : loss_penalty = 4)
  (h4 : lost_bowls = 12)
  (h5 : broken_bowls = 15)
  (h6 : total_payment = 1825) :
  ∃ (initial_fee : ℕ),
    initial_fee = 100 ∧
    total_payment = initial_fee +
      (total_bowls - lost_bowls - broken_bowls) * safe_delivery_pay -
      (lost_bowls + broken_bowls) * loss_penalty :=
by sorry

end bowl_delivery_fee_l389_38920


namespace cuboids_on_diagonal_of_90_cube_l389_38904

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  side : ℕ

/-- Calculates the number of cuboids a diagonal of a cube passes through -/
def cuboids_on_diagonal (cube : Cube) (cuboid : Cuboid) : ℕ :=
  let n1 := cube.side / cuboid.height - 1
  let n2 := cube.side / cuboid.width - 1
  let n3 := cube.side / cuboid.length - 1
  let i12 := cube.side / (cuboid.height * cuboid.width) - 1
  let i23 := cube.side / (cuboid.width * cuboid.length) - 1
  let i13 := cube.side / (cuboid.height * cuboid.length) - 1
  let i123 := cube.side / (cuboid.height * cuboid.width * cuboid.length) - 1
  n1 + n2 + n3 - (i12 + i23 + i13) + i123

/-- The main theorem to be proved -/
theorem cuboids_on_diagonal_of_90_cube (c : Cube) (b : Cuboid) :
  c.side = 90 ∧ b.length = 2 ∧ b.width = 3 ∧ b.height = 5 →
  cuboids_on_diagonal c b = 65 := by
  sorry

end cuboids_on_diagonal_of_90_cube_l389_38904


namespace T_is_three_rays_with_common_endpoint_l389_38952

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 2 < 5) ∨
               (5 = y - 2 ∧ x + 3 < 5) ∨
               (x + 3 = y - 2 ∧ 5 < x + 3)}

-- Define a ray
def Ray (start : ℝ × ℝ) (dir : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * dir.1, start.2 + t * dir.2)}

-- Theorem statement
theorem T_is_three_rays_with_common_endpoint :
  ∃ (start : ℝ × ℝ) (dir1 dir2 dir3 : ℝ × ℝ),
    T = Ray start dir1 ∪ Ray start dir2 ∪ Ray start dir3 ∧
    dir1 ≠ dir2 ∧ dir1 ≠ dir3 ∧ dir2 ≠ dir3 :=
  sorry

end T_is_three_rays_with_common_endpoint_l389_38952


namespace square_diagonal_perimeter_ratio_l389_38956

theorem square_diagonal_perimeter_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a * Real.sqrt 2) / (b * Real.sqrt 2) = 5/2 → (4 * a) / (4 * b) = 5/2 := by
sorry

end square_diagonal_perimeter_ratio_l389_38956


namespace hockey_championship_points_l389_38966

/-- Represents the number of points a team receives for winning a game. -/
def win_points : ℕ := 2

/-- Represents the number of games tied. -/
def games_tied : ℕ := 12

/-- Represents the number of games won. -/
def games_won : ℕ := games_tied + 12

/-- Represents the points received for a tie. -/
def tie_points : ℕ := 1

theorem hockey_championship_points :
  win_points * games_won + tie_points * games_tied = 60 :=
sorry

end hockey_championship_points_l389_38966


namespace constant_grid_function_l389_38916

/-- A function from integer pairs to non-negative integers -/
def GridFunction := ℤ × ℤ → ℕ

/-- The property that each value is the average of its four neighbors -/
def IsAverageOfNeighbors (f : GridFunction) : Prop :=
  ∀ x y : ℤ, 4 * f (x, y) = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)

/-- Theorem stating that if a grid function satisfies the average property, it is constant -/
theorem constant_grid_function (f : GridFunction) (h : IsAverageOfNeighbors f) :
  ∀ x₁ y₁ x₂ y₂ : ℤ, f (x₁, y₁) = f (x₂, y₂) := by
  sorry


end constant_grid_function_l389_38916


namespace function_inverse_fraction_l389_38968

/-- Given a function f : ℝ \ {-1} → ℝ satisfying f((1-x)/(1+x)) = x for all x ≠ -1,
    prove that f(x) = (1-x)/(1+x) for all x ≠ -1 -/
theorem function_inverse_fraction (f : ℝ → ℝ) 
    (h : ∀ x ≠ -1, f ((1 - x) / (1 + x)) = x) :
    ∀ x ≠ -1, f x = (1 - x) / (1 + x) := by
  sorry

end function_inverse_fraction_l389_38968


namespace oliver_bumper_car_rides_l389_38913

def carnival_rides (ferris_wheel_rides : ℕ) (tickets_per_ride : ℕ) (total_tickets : ℕ) : ℕ :=
  (total_tickets - ferris_wheel_rides * tickets_per_ride) / tickets_per_ride

theorem oliver_bumper_car_rides :
  carnival_rides 5 7 63 = 4 := by
  sorry

end oliver_bumper_car_rides_l389_38913


namespace unique_solution_for_equation_l389_38983

theorem unique_solution_for_equation : ∃! (x y z : ℕ), 
  x < 10 ∧ y < 10 ∧ z < 10 ∧ 
  (10 * x + 5) * (300 + 10 * y + z) = 7850 ∧
  x = 2 ∧ y = 1 ∧ z = 4 := by sorry

end unique_solution_for_equation_l389_38983


namespace sqrt_difference_inequality_l389_38906

theorem sqrt_difference_inequality (n : ℝ) (hn : n ≥ 0) :
  Real.sqrt (n + 2) - Real.sqrt (n + 1) < Real.sqrt (n + 1) - Real.sqrt n := by
  sorry

end sqrt_difference_inequality_l389_38906


namespace min_value_trig_function_min_value_trig_function_achievable_l389_38930

theorem min_value_trig_function (α : Real) (h : α ∈ Set.Ioo 0 (π / 2)) :
  1 / (Real.sin α)^2 + 3 / (Real.cos α)^2 ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

theorem min_value_trig_function_achievable :
  ∃ α : Real, α ∈ Set.Ioo 0 (π / 2) ∧
  1 / (Real.sin α)^2 + 3 / (Real.cos α)^2 = 4 + 2 * Real.sqrt 3 := by
  sorry

end min_value_trig_function_min_value_trig_function_achievable_l389_38930


namespace student_percentage_theorem_l389_38954

theorem student_percentage_theorem (total : ℝ) (h_total_pos : total > 0) : 
  let third_year_percent : ℝ := 0.30
  let not_third_second_ratio : ℝ := 1/7
  let third_year : ℝ := third_year_percent * total
  let not_third_year : ℝ := total - third_year
  let second_year_not_third : ℝ := not_third_second_ratio * not_third_year
  let not_second_year : ℝ := total - second_year_not_third
  (not_second_year / total) * 100 = 90
:= by sorry

end student_percentage_theorem_l389_38954


namespace cough_ratio_l389_38927

-- Define the number of coughs per minute for Georgia
def georgia_coughs_per_minute : ℕ := 5

-- Define the total number of coughs after 20 minutes
def total_coughs_after_20_minutes : ℕ := 300

-- Define Robert's coughs per minute
def robert_coughs_per_minute : ℕ := (total_coughs_after_20_minutes - georgia_coughs_per_minute * 20) / 20

-- Theorem stating the ratio of Robert's coughs to Georgia's coughs is 2:1
theorem cough_ratio :
  robert_coughs_per_minute / georgia_coughs_per_minute = 2 ∧
  robert_coughs_per_minute > georgia_coughs_per_minute :=
by sorry

end cough_ratio_l389_38927


namespace no_equilateral_with_100_degree_angle_l389_38988

-- Define what an equilateral triangle is
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c

-- Define the sum of angles in a triangle
axiom triangle_angle_sum (a b c : ℝ) : a + b + c = 180

-- Theorem: An equilateral triangle cannot have an angle of 100 degrees
theorem no_equilateral_with_100_degree_angle (a b c : ℝ) :
  is_equilateral a b c → ¬(a = 100 ∨ b = 100 ∨ c = 100) :=
by sorry

end no_equilateral_with_100_degree_angle_l389_38988


namespace sally_balloons_l389_38969

def initial_orange_balloons : ℕ := sorry

def lost_balloons : ℕ := 2

def current_orange_balloons : ℕ := 7

theorem sally_balloons : initial_orange_balloons = current_orange_balloons + lost_balloons :=
by sorry

end sally_balloons_l389_38969


namespace sum_first_tenth_l389_38918

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_4_7 : a 4 + a 7 = 2
  prod_5_6 : a 5 * a 6 = -8

/-- The sum of the first and tenth terms of the geometric sequence is -7 -/
theorem sum_first_tenth (seq : GeometricSequence) : seq.a 1 + seq.a 10 = -7 := by
  sorry

end sum_first_tenth_l389_38918


namespace polynomial_intersection_l389_38961

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (b - a^2/4 = d - c^2/4) →
  -- The graphs of f and g intersect at the point (2012, -2012)
  f a b 2012 = -2012 ∧ g c d 2012 = -2012 →
  -- Conclusion
  a + c = -8048 := by
  sorry

end polynomial_intersection_l389_38961


namespace jack_and_jill_meeting_point_l389_38944

/-- Represents the meeting point of Jack and Jill on their hill run. -/
structure MeetingPoint where
  /-- The time at which Jack and Jill meet, measured from Jill's start time. -/
  time : ℝ
  /-- The distance from the start point where Jack and Jill meet. -/
  distance : ℝ

/-- Calculates the meeting point of Jack and Jill given their running conditions. -/
def calculateMeetingPoint (totalDistance jackHeadStart uphillDistance : ℝ)
                          (jackUphillSpeed jackDownhillSpeed : ℝ)
                          (jillUphillSpeed jillDownhillSpeed : ℝ) : MeetingPoint :=
  sorry

/-- Theorem stating that Jack and Jill meet 2 km from the top of the hill. -/
theorem jack_and_jill_meeting_point :
  let meetingPoint := calculateMeetingPoint 12 (2/15) 7 12 18 14 20
  meetingPoint.distance = 5 ∧ uphillDistance - meetingPoint.distance = 2 := by
  sorry

end jack_and_jill_meeting_point_l389_38944


namespace proportion_not_greater_than_30_proportion_as_percentage_l389_38926

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of data points greater than 30
def data_greater_than_30 : ℕ := 3

-- Define the proportion calculation function
def calculate_proportion (total : ℕ) (part : ℕ) : ℚ :=
  (total - part : ℚ) / total

-- Theorem statement
theorem proportion_not_greater_than_30 :
  calculate_proportion sample_size data_greater_than_30 = 47/50 :=
by
  sorry

-- Additional theorem to show the decimal representation
theorem proportion_as_percentage :
  (calculate_proportion sample_size data_greater_than_30 * 100 : ℚ) = 94 :=
by
  sorry

end proportion_not_greater_than_30_proportion_as_percentage_l389_38926


namespace points_on_line_procedure_l389_38984

theorem points_on_line_procedure (n : ℕ) : ∃ n, 9 * n - 8 = 82 := by
  sorry

end points_on_line_procedure_l389_38984


namespace inequality_system_solution_range_l389_38938

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (3*(x-1) > x-6 ∧ 8-2*x+2*a ≥ 0) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  -3 ≤ a ∧ a < -2 :=
by sorry

end inequality_system_solution_range_l389_38938


namespace max_salary_460000_l389_38996

/-- Represents a hockey team -/
structure HockeyTeam where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player in a hockey team -/
def maxPlayerSalary (team : HockeyTeam) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a player in the given conditions -/
theorem max_salary_460000 (team : HockeyTeam) 
  (h1 : team.players = 18)
  (h2 : team.minSalary = 20000)
  (h3 : team.maxTotalSalary = 800000) : 
  maxPlayerSalary team = 460000 := by
sorry

#eval maxPlayerSalary { players := 18, minSalary := 20000, maxTotalSalary := 800000 }

end max_salary_460000_l389_38996


namespace base_difference_equals_7422_l389_38999

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The main theorem --/
theorem base_difference_equals_7422 :
  let base_7_num := to_base_10 [3, 4, 1, 2, 5] 7
  let base_8_num := to_base_10 [5, 4, 3, 2, 1] 8
  base_7_num - base_8_num = 7422 := by sorry

end base_difference_equals_7422_l389_38999


namespace foreign_language_selection_l389_38943

theorem foreign_language_selection (total : ℕ) (english_speakers : ℕ) (japanese_speakers : ℕ) :
  total = 9 ∧ english_speakers = 5 ∧ japanese_speakers = 4 →
  english_speakers * japanese_speakers = 20 := by
sorry

end foreign_language_selection_l389_38943


namespace inequality_system_solution_l389_38970

theorem inequality_system_solution (x : ℝ) :
  (5 * x + 1 ≥ 3 * (x - 1)) →
  (1 - (x + 3) / 3 ≤ x) →
  x ≥ 0 := by sorry

end inequality_system_solution_l389_38970


namespace equal_area_rectangle_width_l389_38903

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: Given two rectangles of equal area, where one rectangle has dimensions 8 by x,
    and the other has dimensions 4 by 30, the value of x is 15 -/
theorem equal_area_rectangle_width :
  ∀ (x : ℝ),
  let r1 := Rectangle.mk 8 x
  let r2 := Rectangle.mk 4 30
  area r1 = area r2 → x = 15 := by
sorry


end equal_area_rectangle_width_l389_38903


namespace sum_of_four_digit_even_and_multiples_of_three_l389_38957

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit multiples of 3 -/
def D : ℕ := 3000

/-- Theorem stating that the sum of four-digit even numbers and four-digit multiples of 3 is 7500 -/
theorem sum_of_four_digit_even_and_multiples_of_three :
  C + D = 7500 := by sorry

end sum_of_four_digit_even_and_multiples_of_three_l389_38957


namespace free_throw_probability_l389_38937

/-- The probability of making a single shot -/
def p : ℝ := sorry

/-- The probability of passing the test (making at least one shot out of three chances) -/
def prob_pass : ℝ := p + p * (1 - p) + p * (1 - p)^2

/-- Theorem stating that if the probability of passing is 0.784, then p is 0.4 -/
theorem free_throw_probability : prob_pass = 0.784 → p = 0.4 := by sorry

end free_throw_probability_l389_38937


namespace line_through_P_perpendicular_to_given_line_l389_38976

-- Define the point P
def P : ℝ × ℝ := (4, -1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Define the equation of the line we're looking for
def target_line (x y : ℝ) : Prop := 4 * x + 3 * y - 13 = 0

-- Theorem statement
theorem line_through_P_perpendicular_to_given_line :
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), m * x + y + b = 0 ↔ target_line x y) ∧
    (m * P.1 + P.2 + b = 0) ∧
    (m * 4 + 3 = 0) :=
sorry

end line_through_P_perpendicular_to_given_line_l389_38976


namespace appetizer_cost_l389_38959

/-- Proves that the cost of the appetizer is $10 given the conditions of the restaurant bill --/
theorem appetizer_cost (entree_cost : ℝ) (entree_count : ℕ) (tip_rate : ℝ) (total_spent : ℝ) :
  entree_cost = 20 →
  entree_count = 4 →
  tip_rate = 0.2 →
  total_spent = 108 →
  ∃ (appetizer_cost : ℝ),
    appetizer_cost + entree_cost * entree_count + tip_rate * (appetizer_cost + entree_cost * entree_count) = total_spent ∧
    appetizer_cost = 10 := by
  sorry


end appetizer_cost_l389_38959


namespace total_cost_is_87_60_l389_38931

/-- Calculate the total cost of T-shirts bought by Dave -/
def calculate_total_cost : ℝ :=
  let white_packs := 3
  let blue_packs := 2
  let red_packs := 4
  let green_packs := 1

  let white_price := 12
  let blue_price := 8
  let red_price := 10
  let green_price := 6

  let white_discount := 0.10
  let blue_discount := 0.05
  let red_discount := 0.15
  let green_discount := 0

  let white_cost := white_packs * white_price * (1 - white_discount)
  let blue_cost := blue_packs * blue_price * (1 - blue_discount)
  let red_cost := red_packs * red_price * (1 - red_discount)
  let green_cost := green_packs * green_price * (1 - green_discount)

  white_cost + blue_cost + red_cost + green_cost

/-- The total cost of T-shirts bought by Dave is $87.60 -/
theorem total_cost_is_87_60 : calculate_total_cost = 87.60 := by
  sorry

end total_cost_is_87_60_l389_38931


namespace triangle_with_given_altitudes_exists_l389_38992

theorem triangle_with_given_altitudes_exists (m_a m_b : ℝ) 
  (h1 : 0 < m_a) (h2 : 0 < m_b) (h3 : m_a ≤ m_b) :
  (∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    m_a = (2 * (a * b * c / (a + b + c))) / a ∧
    m_b = (2 * (a * b * c / (a + b + c))) / b ∧
    m_a + m_b = (2 * (a * b * c / (a + b + c))) / c) ↔
  (m_a / m_b)^2 + (m_a / m_b) > 1 :=
by sorry

end triangle_with_given_altitudes_exists_l389_38992


namespace book_distribution_l389_38990

theorem book_distribution (x : ℕ) : 
  (3 * x + 20 = 4 * x - 25) ↔ 
  (∃ (total_books : ℕ), 
    (total_books = 3 * x + 20) ∧ 
    (total_books = 4 * x - 25)) :=
by sorry

end book_distribution_l389_38990


namespace remainder_theorem_l389_38986

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 - 2*x^3 + 4*x^2 + x + 5

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, p = λ x => (x + 2) * q x + 3 :=
sorry

end remainder_theorem_l389_38986


namespace dartboard_probability_l389_38953

theorem dartboard_probability :
  -- Define the probabilities for each sector
  ∀ (prob_E prob_F prob_G prob_H prob_I : ℚ),
  -- Conditions
  prob_E = 1/5 →
  prob_F = 2/5 →
  prob_G = prob_H →
  prob_G = prob_I →
  -- Sum of all probabilities is 1
  prob_E + prob_F + prob_G + prob_H + prob_I = 1 →
  -- Conclusion: probability of landing on sector G is 2/15
  prob_G = 2/15 :=
by sorry

end dartboard_probability_l389_38953


namespace jake_read_225_pages_l389_38964

/-- The number of pages Jake read in a week -/
def pages_read : ℕ :=
  let day1 : ℕ := 45
  let day2 : ℕ := day1 / 3
  let day3 : ℕ := 58 - 12
  let day4 : ℕ := (day1 + 1) / 2  -- Rounding up
  let day5 : ℕ := (3 * day3 + 3) / 4  -- Rounding up
  let day6 : ℕ := day2
  let day7 : ℕ := 2 * day4
  day1 + day2 + day3 + day4 + day5 + day6 + day7

/-- Theorem stating that Jake read 225 pages in total -/
theorem jake_read_225_pages : pages_read = 225 := by
  sorry

end jake_read_225_pages_l389_38964
