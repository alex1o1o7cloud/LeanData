import Mathlib

namespace correct_statements_about_squares_l4011_401158

theorem correct_statements_about_squares :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧
  (∀ x : ℝ, x < -1 → x^2 > x) :=
by sorry

end correct_statements_about_squares_l4011_401158


namespace polynomial_factor_sum_l4011_401146

theorem polynomial_factor_sum (a b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^3 + a*x^2 + b*x + 8 = (x + 1) * (x + 2) * (x + c)) → 
  a + b = 21 := by
  sorry

end polynomial_factor_sum_l4011_401146


namespace amp_example_l4011_401116

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem amp_example : 50 - amp 8 5 = 11 := by
  sorry

end amp_example_l4011_401116


namespace complex_fraction_sum_l4011_401181

theorem complex_fraction_sum : 
  (Complex.I + 1)^2 / (Complex.I * 2 + 1) + (1 - Complex.I)^2 / (2 - Complex.I) = (6 - Complex.I * 2) / 5 := by
  sorry

end complex_fraction_sum_l4011_401181


namespace wire_cut_ratio_l4011_401121

theorem wire_cut_ratio (a b : ℝ) : 
  a > 0 → b > 0 → (∃ (r : ℝ), a = 2 * Real.pi * r) → (∃ (s : ℝ), b = 4 * s) → a = b → a / b = 1 := by
  sorry

end wire_cut_ratio_l4011_401121


namespace M_intersect_N_is_empty_l4011_401195

-- Define set M
def M : Set ℝ := {y | ∃ x, y = x + 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ (N.image Prod.snd) = ∅ := by
  sorry

end M_intersect_N_is_empty_l4011_401195


namespace two_rectangle_formations_l4011_401109

def square_sides : List ℕ := [3, 5, 9, 11, 14, 19, 20, 24, 31, 33, 36, 39, 42]

def rectangle_width : ℕ := 75
def rectangle_height : ℕ := 112

def forms_rectangle (subset : List ℕ) : Prop :=
  (subset.map (λ x => x^2)).sum = rectangle_width * rectangle_height

theorem two_rectangle_formations :
  ∃ (subset1 subset2 : List ℕ),
    subset1 ⊆ square_sides ∧
    subset2 ⊆ square_sides ∧
    subset1 ∩ subset2 = ∅ ∧
    forms_rectangle subset1 ∧
    forms_rectangle subset2 :=
sorry

end two_rectangle_formations_l4011_401109


namespace f_2009_eq_zero_l4011_401148

/-- An even function on ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- An odd function on ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

/-- Main theorem -/
theorem f_2009_eq_zero
  (f g : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_f_1 : f 1 = 0)
  (h_odd : OddFunction g)
  (h_g_def : ∀ x, g x = f (x - 1)) :
  f 2009 = 0 := by
sorry

end f_2009_eq_zero_l4011_401148


namespace expand_product_l4011_401119

theorem expand_product (x : ℝ) : (5*x^2 + 7) * (3*x^3 + 4*x + 1) = 15*x^5 + 41*x^3 + 5*x^2 + 28*x + 7 := by
  sorry

end expand_product_l4011_401119


namespace smallest_z_for_inequality_l4011_401124

theorem smallest_z_for_inequality : ∃! z : ℕ, (∀ w : ℕ, 27^w > 3^24 → w ≥ z) ∧ 27^z > 3^24 :=
by
  -- The proof would go here
  sorry

end smallest_z_for_inequality_l4011_401124


namespace product_sum_fractions_l4011_401179

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end product_sum_fractions_l4011_401179


namespace male_democrat_ratio_l4011_401105

/-- Proves the ratio of male democrats to total male participants in a meeting --/
theorem male_democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  female_democrats * 2 ≤ total_participants →
  3 * female_democrats * 2 = total_participants →
  (total_participants / 3 - female_democrats) * 4 = total_participants - female_democrats * 2 :=
by
  sorry

#check male_democrat_ratio

end male_democrat_ratio_l4011_401105


namespace worker_count_l4011_401168

theorem worker_count (total : ℕ) (extra_total : ℕ) (extra_contribution : ℕ) :
  total = 300000 →
  extra_total = 375000 →
  extra_contribution = 50 →
  ∃ n : ℕ, 
    n * (total / n) = total ∧
    n * (total / n + extra_contribution) = extra_total ∧
    n = 1500 :=
by
  sorry

end worker_count_l4011_401168


namespace five_consecutive_not_square_l4011_401157

theorem five_consecutive_not_square (n : ℤ) : ¬∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = m ^ 2 := by
  sorry

end five_consecutive_not_square_l4011_401157


namespace ball_bounce_distance_l4011_401171

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounces : ℕ) : ℝ :=
  3 * initialHeight - 2^(2 - bounces) * initialHeight

/-- Theorem: A ball dropped from 128 meters, bouncing to half its previous height each time,
    travels 383 meters after 9 bounces -/
theorem ball_bounce_distance :
  totalDistance 128 9 = 383 := by
  sorry

#eval totalDistance 128 9

end ball_bounce_distance_l4011_401171


namespace sister_dolls_count_hannah_dolls_relation_total_dolls_sum_l4011_401115

/-- The number of dolls Hannah's sister has -/
def sister_dolls : ℕ := 8

/-- The number of dolls Hannah has -/
def hannah_dolls : ℕ := 5 * sister_dolls

/-- The total number of dolls Hannah and her sister have -/
def total_dolls : ℕ := 48

theorem sister_dolls_count : sister_dolls = 8 :=
  by sorry

theorem hannah_dolls_relation : hannah_dolls = 5 * sister_dolls :=
  by sorry

theorem total_dolls_sum : sister_dolls + hannah_dolls = total_dolls :=
  by sorry

end sister_dolls_count_hannah_dolls_relation_total_dolls_sum_l4011_401115


namespace line_intersection_x_axis_l4011_401117

/-- A line passing through two points (2, 3) and (6, 7) intersects the x-axis at (-1, 0). -/
theorem line_intersection_x_axis :
  let line := (fun x => x + 1)  -- Define the line equation y = x + 1
  ∀ x y : ℝ,
    (x = 2 ∧ y = 3) ∨ (x = 6 ∧ y = 7) →  -- The line passes through (2, 3) and (6, 7)
    y = line x →  -- The point (x, y) is on the line
    (line (-1) = 0)  -- The line intersects the x-axis at x = -1
    ∧ (∀ t : ℝ, t ≠ -1 → line t ≠ 0)  -- The intersection point is unique
    := by sorry

end line_intersection_x_axis_l4011_401117


namespace power_two_ge_two_times_l4011_401100

theorem power_two_ge_two_times (n : ℕ) : 2^n ≥ 2*n := by
  sorry

end power_two_ge_two_times_l4011_401100


namespace jasons_treats_cost_l4011_401197

/-- Represents the quantity and price of a treat type -/
structure Treat where
  quantity : ℕ  -- quantity in dozens
  price : ℕ     -- price per dozen in dollars
  deriving Repr

/-- Calculates the total cost of treats -/
def totalCost (treats : List Treat) : ℕ :=
  treats.foldl (fun acc t => acc + t.quantity * t.price) 0

theorem jasons_treats_cost (cupcakes cookies brownies : Treat)
    (h1 : cupcakes = { quantity := 4, price := 10 })
    (h2 : cookies = { quantity := 3, price := 8 })
    (h3 : brownies = { quantity := 2, price := 12 }) :
    totalCost [cupcakes, cookies, brownies] = 88 := by
  sorry


end jasons_treats_cost_l4011_401197


namespace expected_weekly_rainfall_l4011_401110

/-- The probability of no rain on a given day -/
def prob_no_rain : ℝ := 0.3

/-- The probability of 5 inches of rain on a given day -/
def prob_5_inches : ℝ := 0.4

/-- The probability of 12 inches of rain on a given day -/
def prob_12_inches : ℝ := 0.3

/-- The amount of rainfall in inches when it rains 5 inches -/
def rain_5_inches : ℝ := 5

/-- The amount of rainfall in inches when it rains 12 inches -/
def rain_12_inches : ℝ := 12

/-- The number of days in the week -/
def days_in_week : ℕ := 7

/-- The expected rainfall for one day -/
def expected_daily_rainfall : ℝ :=
  prob_no_rain * 0 + prob_5_inches * rain_5_inches + prob_12_inches * rain_12_inches

/-- Theorem: The expected total rainfall for the week is 39.2 inches -/
theorem expected_weekly_rainfall :
  (days_in_week : ℝ) * expected_daily_rainfall = 39.2 := by
  sorry

end expected_weekly_rainfall_l4011_401110


namespace solution_set_quadratic_inequality_l4011_401188

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 2) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end solution_set_quadratic_inequality_l4011_401188


namespace racecourse_length_l4011_401178

/-- The length of a racecourse where two runners A and B finish simultaneously,
    given that A runs twice as fast as B and B starts 42 meters ahead. -/
theorem racecourse_length : ℝ := by
  /- Let v be B's speed -/
  let v : ℝ := 1
  /- A's speed is twice B's speed -/
  let speed_A : ℝ := 2 * v
  /- B starts 42 meters ahead -/
  let head_start : ℝ := 42
  /- d is the length of the racecourse -/
  let d : ℝ := 84
  /- Time for A to finish the race -/
  let time_A : ℝ := d / speed_A
  /- Time for B to finish the race -/
  let time_B : ℝ := (d - head_start) / v
  /- A and B reach the finish line simultaneously -/
  have h : time_A = time_B := by sorry
  /- The racecourse length is 84 meters -/
  exact d

end racecourse_length_l4011_401178


namespace store_rooms_problem_l4011_401184

/-- The number of rooms in Li Sangong's store -/
def num_rooms : ℕ := 8

/-- The total number of people visiting the store -/
def total_people : ℕ := 7 * num_rooms + 7

theorem store_rooms_problem :
  (total_people = 7 * num_rooms + 7) ∧
  (total_people = 9 * (num_rooms - 1)) ∧
  (num_rooms = 8) := by
  sorry

end store_rooms_problem_l4011_401184


namespace perimeter_of_modified_square_l4011_401159

-- Define the square and triangle
def square_side_length : ℝ := 16
def triangle_leg_length : ℝ := 8

-- Define the theorem
theorem perimeter_of_modified_square :
  let square_perimeter := 4 * square_side_length
  let triangle_hypotenuse := Real.sqrt (2 * triangle_leg_length ^ 2)
  let new_figure_perimeter := square_perimeter - triangle_leg_length + triangle_hypotenuse
  new_figure_perimeter = 64 + 8 * Real.sqrt 2 := by
  sorry


end perimeter_of_modified_square_l4011_401159


namespace smallest_whole_multiple_651_l4011_401199

def is_whole_multiple (n m : ℕ) : Prop := n % m = 0

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def P (n : ℕ) : ℕ := max ((n / 100) * 10 + (n / 10) % 10) (max ((n / 100) * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

def Q (n : ℕ) : ℕ := min ((n / 100) * 10 + (n / 10) % 10) (min ((n / 100) * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

theorem smallest_whole_multiple_651 : 
  ∃ (A : ℕ), 
    100 ≤ A ∧ A < 1000 ∧ 
    is_whole_multiple A 12 ∧
    digit_sum A = 12 ∧ 
    (A % 10 < (A / 10) % 10) ∧ ((A / 10) % 10 < A / 100) ∧
    ((P A + Q A) % 2 = 0) ∧
    (∀ (B : ℕ), 
      100 ≤ B ∧ B < 1000 ∧ 
      is_whole_multiple B 12 ∧
      digit_sum B = 12 ∧ 
      (B % 10 < (B / 10) % 10) ∧ ((B / 10) % 10 < B / 100) ∧
      ((P B + Q B) % 2 = 0) →
      A ≤ B) ∧
    A = 651 := by
  sorry

end smallest_whole_multiple_651_l4011_401199


namespace max_quarters_in_box_l4011_401143

/-- Represents the number of coins of each type in the coin box -/
structure CoinBox where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the box -/
def total_coins (box : CoinBox) : ℕ :=
  box.nickels + box.dimes + box.quarters

/-- The total value of coins in cents -/
def total_value (box : CoinBox) : ℕ :=
  5 * box.nickels + 10 * box.dimes + 25 * box.quarters

/-- Theorem stating the maximum number of quarters possible -/
theorem max_quarters_in_box :
  ∃ (box : CoinBox),
    total_coins box = 120 ∧
    total_value box = 1000 ∧
    (∀ (other_box : CoinBox),
      total_coins other_box = 120 →
      total_value other_box = 1000 →
      other_box.quarters ≤ box.quarters) ∧
    box.quarters = 20 := by
  sorry

end max_quarters_in_box_l4011_401143


namespace unit_vector_of_AB_l4011_401144

/-- Given a plane vector AB = (-1, 2), prove that its unit vector is (-√5/5, 2√5/5) -/
theorem unit_vector_of_AB (AB : ℝ × ℝ) (h : AB = (-1, 2)) :
  let magnitude := Real.sqrt ((AB.1)^2 + (AB.2)^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (-Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) := by
  sorry

end unit_vector_of_AB_l4011_401144


namespace eighth_term_value_l4011_401137

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first six terms
  sum_six : ℚ
  -- The seventh term
  seventh_term : ℚ

/-- Theorem: Given an arithmetic sequence where the sum of the first six terms is 21
    and the seventh term is 8, the eighth term is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence)
    (h1 : seq.sum_six = 21)
    (h2 : seq.seventh_term = 8) :
    ∃ (a d : ℚ), a + 7 * d = 65 / 7 ∧
                 6 * a + 15 * d = 21 ∧
                 a + 6 * d = 8 :=
  sorry

end eighth_term_value_l4011_401137


namespace junior_score_l4011_401150

theorem junior_score (total : ℕ) (junior_score : ℝ) : 
  total > 0 →
  let junior_count : ℝ := 0.2 * total
  let senior_count : ℝ := 0.8 * total
  let overall_avg : ℝ := 86
  let senior_avg : ℝ := 85
  (junior_count * junior_score + senior_count * senior_avg) / total = overall_avg →
  junior_score = 90 := by
sorry

end junior_score_l4011_401150


namespace original_ticket_price_l4011_401103

/-- Proves that the original ticket price is $7 given the problem conditions --/
theorem original_ticket_price (num_tickets : ℕ) (discount_percent : ℚ) (total_cost : ℚ) : 
  num_tickets = 24 → 
  discount_percent = 1/2 → 
  total_cost = 84 → 
  (1 - discount_percent) * (num_tickets : ℚ) * (7 : ℚ) = total_cost := by
sorry

end original_ticket_price_l4011_401103


namespace tan_one_iff_quarter_pi_plus_multiple_pi_l4011_401142

theorem tan_one_iff_quarter_pi_plus_multiple_pi (x : ℝ) : 
  Real.tan x = 1 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 := by
  sorry

end tan_one_iff_quarter_pi_plus_multiple_pi_l4011_401142


namespace min_value_theorem_l4011_401170

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  ∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_theorem_l4011_401170


namespace square_perimeter_l4011_401133

/-- Given a rectangle with length 50 cm and width 10 cm, and a square with an area
    five times that of the rectangle, prove that the perimeter of the square is 200 cm. -/
theorem square_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) (square_area : ℝ) :
  rectangle_length = 50 ∧ 
  rectangle_width = 10 ∧ 
  square_area = 5 * (rectangle_length * rectangle_width) →
  4 * Real.sqrt square_area = 200 := by
sorry

end square_perimeter_l4011_401133


namespace no_factors_l4011_401193

def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 16

def factor1 (x : ℝ) : ℝ := x^2 - 4
def factor2 (x : ℝ) : ℝ := x + 2
def factor3 (x : ℝ) : ℝ := x^2 + 4*x + 4
def factor4 (x : ℝ) : ℝ := x^2 + 1

theorem no_factors :
  (∃ (x : ℝ), p x ≠ 0 ∧ factor1 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor2 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor3 x = 0) ∧
  (∃ (x : ℝ), p x ≠ 0 ∧ factor4 x = 0) :=
by sorry

end no_factors_l4011_401193


namespace a_8_equals_15_l4011_401138

/-- The sum of the first n terms of the sequence {aₙ} -/
def S (n : ℕ) : ℕ := n^2

/-- The n-th term of the sequence {aₙ} -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Theorem stating that the 8th term of the sequence is 15 -/
theorem a_8_equals_15 : a 8 = 15 := by
  sorry

end a_8_equals_15_l4011_401138


namespace math_class_size_l4011_401153

theorem math_class_size (total_students : ℕ) (both_subjects : ℕ) :
  total_students = 75 →
  both_subjects = 15 →
  ∃ (math_only physics_only : ℕ),
    total_students = math_only + physics_only + both_subjects ∧
    math_only + both_subjects = 2 * (physics_only + both_subjects) →
  math_only + both_subjects = 60 := by
  sorry

end math_class_size_l4011_401153


namespace smallest_area_triangle_l4011_401101

-- Define the angle XAY
def Angle (X A Y : Point) : Prop := sorry

-- Define a point O inside the angle XAY
def InsideAngle (O X A Y : Point) : Prop := sorry

-- Define symmetry of angles with respect to a point
def SymmetricAngle (X A Y X' A' Y' O : Point) : Prop := sorry

-- Define the intersection points B and C
def IntersectionPoints (B C X A Y X' A' Y' O : Point) : Prop := sorry

-- Define a line passing through three points
def LineThroughPoints (P Q R : Point) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem smallest_area_triangle 
  (X A Y O : Point) 
  (h1 : Angle X A Y) 
  (h2 : InsideAngle O X A Y) 
  (X' A' Y' : Point) 
  (h3 : SymmetricAngle X A Y X' A' Y' O) 
  (B C : Point) 
  (h4 : IntersectionPoints B C X A Y X' A' Y' O) 
  (h5 : LineThroughPoints B O C) :
  ∀ P Q : Point, 
    LineThroughPoints P O Q → 
    TriangleArea A P Q ≥ TriangleArea A B C := 
by sorry

end smallest_area_triangle_l4011_401101


namespace number_of_book_pairs_l4011_401163

/-- Represents the number of books in each genre --/
structure BookCollection where
  mystery : Nat
  fantasy : Nat
  biography : Nat

/-- Represents the condition that "Mystery Masterpiece" must be included --/
def mustIncludeMysteryMasterpiece : Bool := true

/-- Calculates the number of possible book pairs --/
def calculatePossiblePairs (books : BookCollection) : Nat :=
  books.fantasy + books.biography

/-- Theorem stating the number of possible book pairs --/
theorem number_of_book_pairs :
  let books : BookCollection := ⟨4, 3, 3⟩
  calculatePossiblePairs books = 6 := by sorry

end number_of_book_pairs_l4011_401163


namespace intersection_of_M_and_N_l4011_401107

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | ∃ x, y = -x^2 + 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Icc 0 1 := by sorry

end intersection_of_M_and_N_l4011_401107


namespace rabbit_speed_theorem_l4011_401166

/-- Given a rabbit's speed, double it, add 4, and double again -/
def rabbit_speed_operation (speed : ℕ) : ℕ :=
  ((speed * 2) + 4) * 2

/-- Theorem stating that the rabbit speed operation on 45 results in 188 -/
theorem rabbit_speed_theorem : rabbit_speed_operation 45 = 188 := by
  sorry

#eval rabbit_speed_operation 45  -- This will evaluate to 188

end rabbit_speed_theorem_l4011_401166


namespace journey_average_speed_l4011_401173

/-- Proves that the average speed of a two-segment journey is 54.4 miles per hour -/
theorem journey_average_speed :
  let distance1 : ℝ := 200  -- miles
  let time1 : ℝ := 4.5      -- hours
  let distance2 : ℝ := 480  -- miles
  let time2 : ℝ := 8        -- hours
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 54.4      -- miles per hour
:= by sorry

end journey_average_speed_l4011_401173


namespace second_month_sale_correct_l4011_401149

/-- Calculates the sale in the second month given the sales figures for other months and the average sale -/
def calculate_second_month_sale (first_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (average_sale : ℕ) : ℕ :=
  5 * average_sale - (first_month + third_month + fourth_month + fifth_month)

/-- Theorem stating that the calculated second month sale is correct -/
theorem second_month_sale_correct (first_month third_month fourth_month fifth_month average_sale : ℕ) :
  first_month = 5700 →
  third_month = 6855 →
  fourth_month = 3850 →
  fifth_month = 14045 →
  average_sale = 7800 →
  calculate_second_month_sale first_month third_month fourth_month fifth_month average_sale = 7550 :=
by
  sorry

#eval calculate_second_month_sale 5700 6855 3850 14045 7800

end second_month_sale_correct_l4011_401149


namespace simplify_expression_l4011_401122

theorem simplify_expression (x : ℝ) : 7*x + 9 - 2*x + 15 = 5*x + 24 := by
  sorry

end simplify_expression_l4011_401122


namespace sin_cos_identity_l4011_401104

theorem sin_cos_identity : 
  Real.sin (160 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (200 * π / 180) * Real.cos (80 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l4011_401104


namespace gvidon_descendants_l4011_401112

/-- Represents the genealogy of King Gvidon's descendants -/
structure GvidonGenealogy where
  sons : Nat
  descendants_with_sons : Nat
  sons_per_descendant : Nat

/-- Calculates the total number of descendants in Gvidon's genealogy -/
def total_descendants (g : GvidonGenealogy) : Nat :=
  g.sons + g.descendants_with_sons * g.sons_per_descendant

/-- Theorem stating that King Gvidon's total descendants is 305 -/
theorem gvidon_descendants (g : GvidonGenealogy)
  (h1 : g.sons = 5)
  (h2 : g.descendants_with_sons = 100)
  (h3 : g.sons_per_descendant = 3) :
  total_descendants g = 305 := by
  sorry

#check gvidon_descendants

end gvidon_descendants_l4011_401112


namespace trig_identity_l4011_401140

theorem trig_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end trig_identity_l4011_401140


namespace complex_simplification_l4011_401185

theorem complex_simplification :
  (5 - 3*Complex.I) + (-2 + 6*Complex.I) - (7 - 2*Complex.I) = -4 + 5*Complex.I :=
by sorry

end complex_simplification_l4011_401185


namespace milk_remaining_l4011_401182

theorem milk_remaining (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) : 
  initial_milk = 5 → given_milk = 18/7 → remaining_milk = initial_milk - given_milk → remaining_milk = 17/7 := by
  sorry

end milk_remaining_l4011_401182


namespace imaginary_part_of_complex_number_l4011_401180

theorem imaginary_part_of_complex_number : 
  Complex.im (1 - Complex.I * Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end imaginary_part_of_complex_number_l4011_401180


namespace lcm_24_36_42_l4011_401186

theorem lcm_24_36_42 : Nat.lcm (Nat.lcm 24 36) 42 = 504 := by sorry

end lcm_24_36_42_l4011_401186


namespace f_properties_l4011_401189

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a < 0 → ∀ x : ℝ, 0 < x → f a x > 0) ∧
  (0 < a → ∀ x : ℝ, 0 < x → (f a x > 0 ↔ x < 2*a)) :=
by sorry

end f_properties_l4011_401189


namespace binomial_coefficient_two_l4011_401160

theorem binomial_coefficient_two (n : ℕ) (h : n > 1) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l4011_401160


namespace polynomial_divisibility_l4011_401194

theorem polynomial_divisibility (m n : ℕ) :
  ∃ q : Polynomial ℤ, (X^2 + X + 1) * q = X^(3*m+2) + (-X^2 - 1)^(3*n+1) + 1 := by
  sorry

end polynomial_divisibility_l4011_401194


namespace max_vertex_product_sum_l4011_401123

/-- The set of numbers to be assigned to the cube faces -/
def CubeNumbers : Finset ℕ := {1, 2, 3, 8, 9, 10}

/-- A valid assignment of numbers to cube faces -/
structure CubeAssignment where
  assignment : Fin 6 → ℕ
  valid : ∀ i, assignment i ∈ CubeNumbers
  distinct : Function.Injective assignment

/-- The sum of products at vertices for a given assignment -/
def vertexProductSum (a : CubeAssignment) : ℕ :=
  let faces := a.assignment
  (faces 0 + faces 1) * (faces 2 + faces 3) * (faces 4 + faces 5)

/-- Theorem stating the maximum sum of vertex products -/
theorem max_vertex_product_sum :
  ∀ a : CubeAssignment, vertexProductSum a ≤ 1331 :=
sorry

end max_vertex_product_sum_l4011_401123


namespace root_sum_symmetric_function_l4011_401113

theorem root_sum_symmetric_function (g : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃! (r : Finset ℝ), r.card = 6 ∧ ∀ x ∈ r, g x = 0 ∧ ∀ y, g y = 0 → y ∈ r) :
  ∃ (r : Finset ℝ), r.card = 6 ∧ (∀ x ∈ r, g x = 0) ∧ (r.sum id = 18) := by
  sorry

end root_sum_symmetric_function_l4011_401113


namespace total_sticks_is_326_l4011_401177

/-- The number of sticks needed for four rafts given specific conditions -/
def total_sticks : ℕ :=
  let simon := 45
  let gerry := (3 * simon) / 5
  let micky := simon + gerry + 15
  let darryl := 2 * micky - 7
  simon + gerry + micky + darryl

/-- Theorem stating that the total number of sticks needed is 326 -/
theorem total_sticks_is_326 : total_sticks = 326 := by
  sorry

end total_sticks_is_326_l4011_401177


namespace imaginary_part_of_i_l4011_401135

theorem imaginary_part_of_i : Complex.im Complex.I = 1 := by
  sorry

end imaginary_part_of_i_l4011_401135


namespace not_sufficient_nor_necessary_l4011_401145

theorem not_sufficient_nor_necessary : ¬(∀ x : ℝ, (x - 2) * (x - 1) > 0 → (x - 2 > 0 ∨ x - 1 > 0)) ∧
                                       ¬(∀ x : ℝ, (x - 2 > 0 ∨ x - 1 > 0) → (x - 2) * (x - 1) > 0) := by
  sorry

end not_sufficient_nor_necessary_l4011_401145


namespace slate_rock_count_l4011_401106

theorem slate_rock_count :
  let pumice_count : ℕ := 11
  let granite_count : ℕ := 4
  let total_count (slate_count : ℕ) : ℕ := slate_count + pumice_count + granite_count
  let prob_two_slate (slate_count : ℕ) : ℚ :=
    (slate_count : ℚ) / (total_count slate_count : ℚ) *
    ((slate_count - 1 : ℚ) / (total_count slate_count - 1 : ℚ))
  ∃ (slate_count : ℕ),
    prob_two_slate slate_count = 15 / 100 ∧
    slate_count = 10 :=
by sorry

end slate_rock_count_l4011_401106


namespace exponent_multiplication_l4011_401131

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l4011_401131


namespace inequality_system_solution_l4011_401129

theorem inequality_system_solution (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) → -2 < x ∧ x < 1 := by
  sorry

end inequality_system_solution_l4011_401129


namespace creature_dressing_order_l4011_401108

-- Define the number of arms
def num_arms : ℕ := 6

-- Define the number of items per arm
def items_per_arm : ℕ := 3

-- Define the total number of items
def total_items : ℕ := num_arms * items_per_arm

-- Define the number of valid permutations per arm (1 out of 6)
def valid_perm_per_arm : ℕ := 1

-- Define the total number of permutations per arm
def total_perm_per_arm : ℕ := Nat.factorial items_per_arm

-- Theorem statement
theorem creature_dressing_order :
  (Nat.factorial total_items) / (total_perm_per_arm ^ num_arms) =
  (Nat.factorial total_items) / (3 ^ num_arms) :=
sorry

end creature_dressing_order_l4011_401108


namespace pet_store_birds_l4011_401152

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 4

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 8

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 40 := by
  sorry

end pet_store_birds_l4011_401152


namespace sin_seventeen_pi_fourths_l4011_401172

theorem sin_seventeen_pi_fourths : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end sin_seventeen_pi_fourths_l4011_401172


namespace rectangle_max_area_l4011_401139

/-- Given a rectangle with perimeter 40, its maximum area is 100 -/
theorem rectangle_max_area :
  ∀ w l : ℝ,
  w > 0 → l > 0 →
  2 * (w + l) = 40 →
  ∀ w' l' : ℝ,
  w' > 0 → l' > 0 →
  2 * (w' + l') = 40 →
  w * l ≤ 100 :=
by sorry

end rectangle_max_area_l4011_401139


namespace curve_translation_l4011_401114

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop := y * Real.cos x + 2 * y - 1 = 0

/-- The translated curve equation -/
def translated_curve (x y : ℝ) : Prop := (y + 1) * Real.sin x + 2 * y + 1 = 0

/-- Theorem stating that the translation of the original curve results in the translated curve -/
theorem curve_translation (x y : ℝ) : 
  original_curve (x - π/2) (y + 1) ↔ translated_curve x y :=
sorry

end curve_translation_l4011_401114


namespace eighth_term_of_happy_sequence_l4011_401196

def happy_sequence (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / 2^n

theorem eighth_term_of_happy_sequence :
  happy_sequence 8 = 1/32 := by sorry

end eighth_term_of_happy_sequence_l4011_401196


namespace cloth_cost_price_l4011_401147

/-- Given the following conditions for a cloth sale:
    - Total cloth length: 400 meters
    - Total selling price: Rs. 18,000
    - Loss per meter: Rs. 5
    Prove that the cost price for one meter of cloth is Rs. 50. -/
theorem cloth_cost_price 
  (total_length : ℝ) 
  (total_selling_price : ℝ) 
  (loss_per_meter : ℝ) 
  (h1 : total_length = 400)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  (total_selling_price / total_length) + loss_per_meter = 50 := by
  sorry

end cloth_cost_price_l4011_401147


namespace midpoint_octagon_area_ratio_l4011_401192

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of the midpoint octagon is 1/4 of the original octagon's area -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o :=
sorry

end midpoint_octagon_area_ratio_l4011_401192


namespace spoiled_apples_count_l4011_401111

def total_apples : ℕ := 7
def prob_at_least_one_spoiled : ℚ := 2857142857142857 / 10000000000000000

theorem spoiled_apples_count (S : ℕ) : 
  S < total_apples → 
  (1 : ℚ) - (↑(total_apples - S) / ↑total_apples) * (↑(total_apples - S - 1) / ↑(total_apples - 1)) = prob_at_least_one_spoiled → 
  S = 1 := by sorry

end spoiled_apples_count_l4011_401111


namespace proposition_truth_l4011_401165

theorem proposition_truth (p q : Prop) (hp : p) (hq : ¬q) : (¬p) ∨ (¬q) := by
  sorry

end proposition_truth_l4011_401165


namespace robot_staircase_l4011_401130

theorem robot_staircase (a b : ℕ+) : 
  ∃ n : ℕ, n = a + b - Nat.gcd a b ∧ 
  (∀ m : ℕ, m < n → ¬∃ (k l : ℕ), k * a = m + l * b) ∧
  (∃ (k l : ℕ), k * a = n + l * b) := by
  sorry

end robot_staircase_l4011_401130


namespace base_sequences_count_l4011_401154

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of different base sequences that can be formed from one base A, two bases C, and three bases G --/
def base_sequences : ℕ :=
  choose 6 1 * choose 5 2 * choose 3 3

theorem base_sequences_count : base_sequences = 60 := by sorry

end base_sequences_count_l4011_401154


namespace roof_area_difference_l4011_401141

/-- Proves the difference in area between two rectangular roofs -/
theorem roof_area_difference (w : ℝ) (h1 : w > 0) (h2 : 4 * w * w = 784) : 
  5 * w * w - 4 * w * w = 196 := by
  sorry

end roof_area_difference_l4011_401141


namespace fliers_left_for_next_day_l4011_401156

def total_fliers : ℕ := 2500
def morning_fraction : ℚ := 1/5
def afternoon_fraction : ℚ := 1/4

theorem fliers_left_for_next_day :
  let morning_sent := total_fliers * morning_fraction
  let remaining_after_morning := total_fliers - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  total_fliers - morning_sent - afternoon_sent = 1500 := by sorry

end fliers_left_for_next_day_l4011_401156


namespace acme_soup_words_count_l4011_401162

/-- The number of possible words of length n formed from a set of k distinct letters,
    where each letter appears at least n times. -/
def word_count (n k : ℕ) : ℕ := k^n

/-- The specific case for 6-letter words formed from 6 distinct letters. -/
def acme_soup_words : ℕ := word_count 6 6

theorem acme_soup_words_count :
  acme_soup_words = 46656 := by
  sorry

end acme_soup_words_count_l4011_401162


namespace rectangle_dimension_change_l4011_401161

theorem rectangle_dimension_change (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let A := L * W
  let W' := 0.4 * W
  let A' := 1.36 * A
  ∃ L', L' = 3.4 * L ∧ A' = L' * W' :=
sorry

end rectangle_dimension_change_l4011_401161


namespace area_of_triangle_RZX_l4011_401190

-- Define the square WXYZ
def Square (W X Y Z : ℝ × ℝ) : Prop :=
  -- Add conditions for a square here
  sorry

-- Define the area of a shape
def Area (shape : Set (ℝ × ℝ)) : ℝ :=
  sorry

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) (ratio : ℝ) : Prop :=
  -- P is on AB with AP:PB = ratio:(1-ratio)
  sorry

-- Define the midpoint of a line segment
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  -- M is the midpoint of AB
  sorry

theorem area_of_triangle_RZX 
  (W X Y Z : ℝ × ℝ)
  (P Q R : ℝ × ℝ)
  (h_square : Square W X Y Z)
  (h_area_WXYZ : Area {W, X, Y, Z} = 144)
  (h_P_on_YZ : PointOnSegment P Y Z (1/3))
  (h_Q_mid_WP : Midpoint Q W P)
  (h_R_mid_XP : Midpoint R X P)
  (h_area_YPRQ : Area {Y, P, R, Q} = 30)
  : Area {R, Z, X} = 24 := by
  sorry

end area_of_triangle_RZX_l4011_401190


namespace ernies_original_income_l4011_401102

theorem ernies_original_income
  (ernies_original : ℝ)
  (ernies_current : ℝ)
  (jacks_current : ℝ)
  (h1 : ernies_current = 4/5 * ernies_original)
  (h2 : jacks_current = 2 * ernies_original)
  (h3 : ernies_current + jacks_current = 16800) :
  ernies_original = 6000 := by
sorry

end ernies_original_income_l4011_401102


namespace fred_car_washing_earnings_l4011_401167

/-- Fred's earnings from various activities -/
structure FredEarnings where
  total : ℕ
  newspaper : ℕ
  car_washing : ℕ

/-- Theorem stating that Fred's car washing earnings are 74 dollars -/
theorem fred_car_washing_earnings (e : FredEarnings) 
  (h1 : e.total = 90)
  (h2 : e.newspaper = 16)
  (h3 : e.total = e.newspaper + e.car_washing) :
  e.car_washing = 74 := by
  sorry

end fred_car_washing_earnings_l4011_401167


namespace coin_puzzle_solution_l4011_401126

/-- Represents the number of coins in each pile -/
structure CoinPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Represents the coin movement operation -/
def moveCoins (piles : CoinPiles) : CoinPiles :=
  { first := piles.first - piles.second + piles.third,
    second := 2 * piles.second - piles.third,
    third := piles.third + piles.second - piles.first }

/-- Theorem stating that if after moving coins each pile has 16 coins, 
    then the initial number in the first pile was 22 -/
theorem coin_puzzle_solution (initial : CoinPiles) :
  (moveCoins initial).first = 16 ∧
  (moveCoins initial).second = 16 ∧
  (moveCoins initial).third = 16 →
  initial.first = 22 :=
sorry

end coin_puzzle_solution_l4011_401126


namespace six_and_negative_six_are_opposite_l4011_401125

/-- Two real numbers are opposite if one is the negative of the other -/
def are_opposite (a b : ℝ) : Prop := b = -a

/-- 6 and -6 are opposite numbers -/
theorem six_and_negative_six_are_opposite : are_opposite 6 (-6) := by
  sorry

end six_and_negative_six_are_opposite_l4011_401125


namespace number_difference_l4011_401198

theorem number_difference (a b : ℕ) : 
  a + b = 20000 → 7 * a = b → b - a = 15000 := by
  sorry

end number_difference_l4011_401198


namespace sheep_to_horse_ratio_l4011_401176

theorem sheep_to_horse_ratio :
  let horse_food_per_day : ℕ := 230
  let total_horse_food : ℕ := 12880
  let num_sheep : ℕ := 16
  let num_horses : ℕ := total_horse_food / horse_food_per_day
  (num_sheep : ℚ) / (num_horses : ℚ) = 2 / 7 :=
by sorry

end sheep_to_horse_ratio_l4011_401176


namespace teagan_total_payment_l4011_401132

def original_shirt_price : ℚ := 60
def original_jacket_price : ℚ := 90
def price_reduction : ℚ := 20 / 100
def num_shirts : ℕ := 5
def num_jackets : ℕ := 10

def reduced_price (original_price : ℚ) : ℚ :=
  original_price * (1 - price_reduction)

def total_cost (item_price : ℚ) (quantity : ℕ) : ℚ :=
  item_price * quantity

theorem teagan_total_payment :
  total_cost (reduced_price original_shirt_price) num_shirts +
  total_cost (reduced_price original_jacket_price) num_jackets = 960 := by
  sorry

end teagan_total_payment_l4011_401132


namespace probability_two_girls_five_tickets_l4011_401169

/-- The probability of selecting 2 girls when drawing 5 tickets from a group of 25 students, of which 10 are girls, is 195/506. -/
theorem probability_two_girls_five_tickets (total_students : Nat) (girls : Nat) (tickets : Nat) :
  total_students = 25 →
  girls = 10 →
  tickets = 5 →
  (Nat.choose girls 2 * Nat.choose (total_students - girls) (tickets - 2)) / Nat.choose total_students tickets = 195 / 506 := by
  sorry

end probability_two_girls_five_tickets_l4011_401169


namespace triangle_count_l4011_401174

/-- A triangle with integral side lengths. -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if the given integers form a valid triangle. -/
def is_valid_triangle (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Check if the triangle has a perimeter of 9. -/
def has_perimeter_9 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 9

/-- Two triangles are considered different if they are not congruent. -/
def are_different (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

theorem triangle_count : 
  ∃ (t1 t2 : IntTriangle), 
    is_valid_triangle t1 ∧ 
    is_valid_triangle t2 ∧ 
    has_perimeter_9 t1 ∧ 
    has_perimeter_9 t2 ∧ 
    are_different t1 t2 ∧
    (∀ (t3 : IntTriangle), 
      is_valid_triangle t3 → 
      has_perimeter_9 t3 → 
      (t3 = t1 ∨ t3 = t2)) := by
  sorry

end triangle_count_l4011_401174


namespace prime_squares_and_fourth_powers_l4011_401120

theorem prime_squares_and_fourth_powers (p : ℕ) : 
  Prime p ↔ 
  (p = 2 ∨ p = 3) ∧ 
  (∃ (a b c k : ℤ), a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) :=
sorry

end prime_squares_and_fourth_powers_l4011_401120


namespace spelling_badges_l4011_401118

theorem spelling_badges (H L C : ℕ) : 
  H + L + C = 83 → H = 14 → L = 17 → C = 52 := by
  sorry

end spelling_badges_l4011_401118


namespace compare_negative_decimals_l4011_401187

theorem compare_negative_decimals : -3.3 < -3.14 := by
  sorry

end compare_negative_decimals_l4011_401187


namespace half_power_inequality_l4011_401136

theorem half_power_inequality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end half_power_inequality_l4011_401136


namespace divisibility_by_1989_l4011_401155

theorem divisibility_by_1989 (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℤ, n^(n^(n^n)) - n^(n^n) = 1989 * k := by sorry

end divisibility_by_1989_l4011_401155


namespace least_product_for_divisibility_least_product_for_cross_divisibility_l4011_401191

theorem least_product_for_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a.val^a.val * b.val^b.val = 2000 * k) →
  (∀ c d : ℕ+, (∃ l : ℕ, c.val^c.val * d.val^d.val = 2000 * l) → c.val * d.val ≥ 10) ∧
  (∃ m n : ℕ+, m.val * n.val = 10 ∧ ∃ p : ℕ, m.val^m.val * n.val^n.val = 2000 * p) :=
sorry

theorem least_product_for_cross_divisibility (a b : ℕ+) :
  (∃ k : ℕ, a.val^b.val * b.val^a.val = 2000 * k) →
  (∀ c d : ℕ+, (∃ l : ℕ, c.val^d.val * d.val^c.val = 2000 * l) → c.val * d.val ≥ 20) ∧
  (∃ m n : ℕ+, m.val * n.val = 20 ∧ ∃ p : ℕ, m.val^n.val * n.val^m.val = 2000 * p) :=
sorry

end least_product_for_divisibility_least_product_for_cross_divisibility_l4011_401191


namespace complex_magnitude_l4011_401164

theorem complex_magnitude (z : ℂ) (h : z - 2 + Complex.I = 1) : Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_magnitude_l4011_401164


namespace oranges_harvested_proof_l4011_401151

/-- The number of oranges harvested per day that are not discarded -/
def oranges_kept (sacks_harvested : ℕ) (sacks_discarded : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  (sacks_harvested - sacks_discarded) * oranges_per_sack

/-- Proof that the number of oranges harvested per day that are not discarded is 600 -/
theorem oranges_harvested_proof :
  oranges_kept 76 64 50 = 600 := by
  sorry

end oranges_harvested_proof_l4011_401151


namespace removed_player_height_l4011_401127

/-- The height of the removed player given the initial and final average heights -/
def height_of_removed_player (initial_avg : ℝ) (final_avg : ℝ) : ℝ :=
  11 * initial_avg - 10 * final_avg

/-- Theorem stating the height of the removed player -/
theorem removed_player_height :
  height_of_removed_player 182 181 = 192 := by
  sorry

#eval height_of_removed_player 182 181

end removed_player_height_l4011_401127


namespace equation_value_l4011_401134

theorem equation_value (x y : ℝ) (h : x + 2 * y = 30) :
  x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3 = 16 := by
  sorry

end equation_value_l4011_401134


namespace trigonometric_problem_l4011_401128

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : 0 < β) (h4 : β < π/2) 
  (h5 : Real.sin (π/3 - α) = 3/5) 
  (h6 : Real.cos (β/2 - π/3) = 2*Real.sqrt 5/5) : 
  (Real.sin α = (4*Real.sqrt 3 - 3)/10) ∧ 
  (Real.cos (β/2 - α) = 11*Real.sqrt 5/25) := by
  sorry

end trigonometric_problem_l4011_401128


namespace probability_at_least_one_male_l4011_401183

theorem probability_at_least_one_male (male_count female_count : ℕ) 
  (h1 : male_count = 3) (h2 : female_count = 2) : 
  1 - (Nat.choose female_count 2 : ℚ) / (Nat.choose (male_count + female_count) 2 : ℚ) = 9/10 := by
  sorry

end probability_at_least_one_male_l4011_401183


namespace max_chain_length_is_optimal_l4011_401175

/-- Represents a triangular grid formed by dividing an equilateral triangle --/
structure TriangularGrid where
  n : ℕ
  total_triangles : ℕ := n^2

/-- Represents a chain of triangles in the grid --/
structure TriangleChain (grid : TriangularGrid) where
  length : ℕ
  is_valid : length ≤ grid.total_triangles

/-- The maximum length of a valid triangle chain in a given grid --/
def max_chain_length (grid : TriangularGrid) : ℕ :=
  grid.n^2 - grid.n + 1

/-- Theorem stating that the maximum chain length is n^2 - n + 1 --/
theorem max_chain_length_is_optimal (grid : TriangularGrid) :
  ∀ (chain : TriangleChain grid), chain.length ≤ max_chain_length grid :=
by sorry

end max_chain_length_is_optimal_l4011_401175
