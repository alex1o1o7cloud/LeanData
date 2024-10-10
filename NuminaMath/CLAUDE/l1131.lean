import Mathlib

namespace exists_a_greater_than_bound_l1131_113153

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1/3
  | (n+2) => (2 * a (n+1)) / 3 - a n

theorem exists_a_greater_than_bound : ∃ n : ℕ, a n > 999/1000 := by
  sorry

end exists_a_greater_than_bound_l1131_113153


namespace joe_list_count_l1131_113123

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of times Joe draws a ball -/
def draws : ℕ := 4

/-- The number of numbers Joe selects for the final list -/
def selected : ℕ := 3

/-- The number of different possible lists Joe can create -/
def num_lists : ℕ := n^draws * (draws.choose selected)

theorem joe_list_count :
  num_lists = 202500 := by
  sorry

end joe_list_count_l1131_113123


namespace sum_lent_is_2000_l1131_113159

/-- Prove that the sum lent is 2000, given the conditions of the loan --/
theorem sum_lent_is_2000 
  (interest_rate : ℝ) 
  (loan_duration : ℝ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.03) 
  (h2 : loan_duration = 3) 
  (h3 : ∀ sum_lent : ℝ, sum_lent * interest_rate * loan_duration = sum_lent - interest_difference) 
  (h4 : interest_difference = 1820) : 
  ∃ sum_lent : ℝ, sum_lent = 2000 := by
  sorry


end sum_lent_is_2000_l1131_113159


namespace parallel_line_slope_l1131_113138

/-- Given a line parallel to 3x - 6y = 12, its slope is 1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℚ) (b : ℚ), (∃ (k : ℚ), 3 * x - 6 * (m * x + b) = k) → m = 1/2 := by
  sorry

end parallel_line_slope_l1131_113138


namespace triangle_division_regions_l1131_113172

/-- Given a triangle ABC and a positive integer n, with the sides divided into 2^n equal parts
    and cevians drawn as described, the number of regions into which the triangle is divided
    is equal to 3 · 2^(2n) - 6 · 2^n + 6. -/
theorem triangle_division_regions (n : ℕ+) : ℕ := by
  sorry

end triangle_division_regions_l1131_113172


namespace only_fourteen_satisfies_l1131_113178

-- Define a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the operation of increasing digits
def increase_digits (n : ℕ) : Set ℕ :=
  { m : ℕ | ∃ (a b : ℕ), n = 10 * a + b ∧ 
    m = 10 * (a + 2) + (b + 2) ∨ 
    m = 10 * (a + 2) + (b + 4) ∨ 
    m = 10 * (a + 4) + (b + 2) ∨ 
    m = 10 * (a + 4) + (b + 4) }

-- The main theorem
theorem only_fourteen_satisfies : 
  ∃! (n : ℕ), is_two_digit n ∧ (4 * n) ∈ increase_digits n :=
by
  -- The proof goes here
  sorry

end only_fourteen_satisfies_l1131_113178


namespace hyperbola_eccentricity_l1131_113139

/-- Given two hyperbolas l and C, prove that the eccentricity of C is 3 -/
theorem hyperbola_eccentricity (k a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), k * x + y - Real.sqrt 2 * k = 0) →  -- Hyperbola l
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola C
  (abs k = b / a) →  -- Parallel asymptotes condition
  (Real.sqrt 2 * k / Real.sqrt (1 + k^2) = 4 / 3) →  -- Distance between asymptotes
  Real.sqrt (1 + b^2 / a^2) = 3 :=  -- Eccentricity of C
by sorry

end hyperbola_eccentricity_l1131_113139


namespace square_difference_501_499_l1131_113180

theorem square_difference_501_499 : 501^2 - 499^2 = 2000 := by
  sorry

end square_difference_501_499_l1131_113180


namespace polynomial_factorization_l1131_113165

theorem polynomial_factorization (u : ℝ) : 
  u^4 - 81*u^2 + 144 = (u^2 - 72)*(u - 3)*(u + 3) := by
  sorry

end polynomial_factorization_l1131_113165


namespace simple_interest_calculation_l1131_113110

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 17.5)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 70 := by
  sorry

end simple_interest_calculation_l1131_113110


namespace top_quality_soccer_balls_l1131_113188

/-- Given a batch of soccer balls, calculate the number of top-quality balls -/
theorem top_quality_soccer_balls 
  (total : ℕ) 
  (frequency : ℝ) 
  (h_total : total = 10000)
  (h_frequency : frequency = 0.975) :
  ⌊(total : ℝ) * frequency⌋ = 9750 := by
  sorry

end top_quality_soccer_balls_l1131_113188


namespace second_polygon_sides_l1131_113164

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by sorry

end second_polygon_sides_l1131_113164


namespace kamal_english_marks_l1131_113116

/-- Kamal's marks in English given his other marks and average --/
theorem kamal_english_marks (math physics chem bio : ℕ) (avg : ℚ) :
  math = 65 →
  physics = 82 →
  chem = 67 →
  bio = 85 →
  avg = 75 →
  (math + physics + chem + bio + english : ℚ) / 5 = avg →
  english = 76 := by
  sorry

end kamal_english_marks_l1131_113116


namespace min_reciprocal_sum_l1131_113141

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 :=
by sorry

end min_reciprocal_sum_l1131_113141


namespace xyz_sum_of_squares_l1131_113100

theorem xyz_sum_of_squares (x y z : ℝ) 
  (h1 : (2*x + 2*y + 3*z) / 7 = 9)
  (h2 : (x^2 * y^2 * z^3)^(1/7) = 6)
  (h3 : 7 / ((2/x) + (2/y) + (3/z)) = 4) :
  x^2 + y^2 + z^2 = 351 := by
  sorry

end xyz_sum_of_squares_l1131_113100


namespace p_necessary_not_sufficient_for_q_l1131_113150

-- Define the conditions
def condition_p (m : ℝ) : Prop := -1 < m ∧ m < 5

def condition_q (m : ℝ) : Prop :=
  ∀ x, x^2 - 2*m*x + m^2 - 1 = 0 → -2 < x ∧ x < 4

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ m, condition_q m → condition_p m) ∧
  (∃ m, condition_p m ∧ ¬condition_q m) :=
sorry

end p_necessary_not_sufficient_for_q_l1131_113150


namespace absolute_difference_simplification_l1131_113176

theorem absolute_difference_simplification (a b : ℝ) 
  (ha : a < 0) (hab : a * b < 0) : 
  |a - b - 3| - |4 + b - a| = -1 := by
sorry

end absolute_difference_simplification_l1131_113176


namespace alternating_sum_fraction_equals_two_l1131_113183

theorem alternating_sum_fraction_equals_two :
  (15 - 14 + 13 - 12 + 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7) = 2 := by
  sorry

end alternating_sum_fraction_equals_two_l1131_113183


namespace expression_value_l1131_113134

theorem expression_value : 
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 7
  x^2 + y^2 + z^2 - 2*x*y = 74 := by sorry

end expression_value_l1131_113134


namespace hamburger_price_correct_l1131_113133

/-- The price of a hamburger that satisfies the given conditions -/
def hamburger_price : ℝ := 3.125

/-- The number of hamburgers already sold -/
def hamburgers_sold : ℕ := 12

/-- The number of additional hamburgers needed to be sold -/
def additional_hamburgers : ℕ := 4

/-- The total revenue target -/
def total_revenue : ℝ := 50

/-- Theorem stating that the hamburger price satisfies the given conditions -/
theorem hamburger_price_correct : 
  hamburger_price * (hamburgers_sold + additional_hamburgers) = total_revenue :=
by sorry

end hamburger_price_correct_l1131_113133


namespace prime_sum_and_product_l1131_113158

def smallest_one_digit_prime : ℕ := 2
def second_smallest_two_digit_prime : ℕ := 13
def smallest_three_digit_prime : ℕ := 101

theorem prime_sum_and_product :
  (smallest_one_digit_prime + second_smallest_two_digit_prime + smallest_three_digit_prime = 116) ∧
  (smallest_one_digit_prime * second_smallest_two_digit_prime * smallest_three_digit_prime = 2626) := by
  sorry

end prime_sum_and_product_l1131_113158


namespace ellipse_tangent_intersection_l1131_113118

-- Define the ellipse C
structure Ellipse :=
  (center : ℝ × ℝ)
  (a b : ℝ)
  (eccentricity : ℝ)

-- Define the parabola
def Parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}

-- Define a point on the ellipse
structure PointOnEllipse (C : Ellipse) :=
  (point : ℝ × ℝ)
  (on_ellipse : (point.1 - C.center.1)^2 / C.a^2 + (point.2 - C.center.2)^2 / C.b^2 = 1)

-- Define a tangent line to the ellipse
structure TangentLine (C : Ellipse) :=
  (point : PointOnEllipse C)
  (slope : ℝ)

-- Theorem statement
theorem ellipse_tangent_intersection 
  (C : Ellipse)
  (h1 : C.center = (0, 0))
  (h2 : C.eccentricity = Real.sqrt 2 / 2)
  (h3 : ∃ (f : ℝ × ℝ), f ∈ Parabola ∧ f ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 / C.a^2 + (p.2 - C.center.2)^2 / C.b^2 = C.eccentricity^2})
  (A : PointOnEllipse C)
  (tAB tAC : TangentLine C)
  (h4 : tAB.point = A ∧ tAC.point = A)
  (h5 : tAB.slope * tAC.slope = 1/4) :
  ∃ (P : ℝ × ℝ), P = (0, 3) ∧ 
    ∀ (B C : ℝ × ℝ), 
      (B.2 - A.point.2 = tAB.slope * (B.1 - A.point.1)) → 
      (C.2 - A.point.2 = tAC.slope * (C.1 - A.point.1)) → 
      (P.2 - B.2) / (P.1 - B.1) = (C.2 - B.2) / (C.1 - B.1) := by
  sorry

end ellipse_tangent_intersection_l1131_113118


namespace add_36_15_l1131_113104

theorem add_36_15 : 36 + 15 = 51 := by
  sorry

end add_36_15_l1131_113104


namespace expression_simplification_l1131_113175

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x^2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l1131_113175


namespace diana_erasers_l1131_113137

/-- Given that Diana shares her erasers among 48 friends and each friend gets 80 erasers,
    prove that Diana has 3840 erasers. -/
theorem diana_erasers : ℕ → ℕ → ℕ → Prop :=
  fun num_friends erasers_per_friend total_erasers =>
    (num_friends = 48) →
    (erasers_per_friend = 80) →
    (total_erasers = num_friends * erasers_per_friend) →
    total_erasers = 3840

/-- Proof of the theorem -/
lemma diana_erasers_proof : diana_erasers 48 80 3840 := by
  sorry

end diana_erasers_l1131_113137


namespace reflection_distance_A_l1131_113177

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_distance (x y : ℝ) : ℝ :=
  2 * |y|

/-- Theorem: The length of the segment from A(2, 4) to its reflection A' over the x-axis is 8 --/
theorem reflection_distance_A : reflection_distance 2 4 = 8 := by
  sorry

end reflection_distance_A_l1131_113177


namespace race_distance_proof_l1131_113162

/-- The distance between two runners at the end of a race --/
def distance_between_runners (race_length : ℕ) (arianna_position : ℕ) : ℕ :=
  race_length - arianna_position

theorem race_distance_proof :
  let race_length : ℕ := 1000  -- 1 km in meters
  let arianna_position : ℕ := 184
  distance_between_runners race_length arianna_position = 816 := by
  sorry

end race_distance_proof_l1131_113162


namespace homework_theorem_l1131_113174

/-- The number of possible homework situations for a given number of teachers and students -/
def homework_situations (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
  num_teachers ^ num_students

/-- Theorem: With 3 teachers and 4 students, there are 3^4 possible homework situations -/
theorem homework_theorem :
  homework_situations 3 4 = 3^4 := by
  sorry

end homework_theorem_l1131_113174


namespace crayons_lost_theorem_l1131_113131

/-- The number of crayons lost or given away -/
def crayons_lost_or_given_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the number of crayons lost or given away is correct -/
theorem crayons_lost_theorem (initial : ℕ) (remaining : ℕ) 
  (h : initial ≥ remaining) : 
  crayons_lost_or_given_away initial remaining = initial - remaining :=
by
  sorry

#eval crayons_lost_or_given_away 479 134

end crayons_lost_theorem_l1131_113131


namespace sum_of_squares_given_means_l1131_113129

theorem sum_of_squares_given_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 2 * Real.sqrt 5 →
  a^2 + b^2 = 216 := by
sorry

end sum_of_squares_given_means_l1131_113129


namespace exist_four_distinct_naturals_perfect_squares_l1131_113143

theorem exist_four_distinct_naturals_perfect_squares :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 :=
by
  sorry

end exist_four_distinct_naturals_perfect_squares_l1131_113143


namespace candy_container_count_l1131_113166

theorem candy_container_count : ℕ := by
  -- Define the number of people
  let people : ℕ := 157

  -- Define the number of candies each person receives
  let candies_per_person : ℕ := 235

  -- Define the number of candies left after distribution
  let leftover_candies : ℕ := 98

  -- Define the total number of candies
  let total_candies : ℕ := people * candies_per_person + leftover_candies

  -- Prove that the total number of candies is 36,993
  have h : total_candies = 36993 := by sorry

  -- Return the result
  exact 36993

end candy_container_count_l1131_113166


namespace equation_solution_l1131_113112

theorem equation_solution (a : ℤ) : 
  (∃ x : ℕ, a * (x : ℤ) = 3) → (a = 1 ∨ a = 3) := by
sorry

end equation_solution_l1131_113112


namespace sarah_flour_amount_l1131_113107

/-- The amount of rye flour Sarah bought in pounds -/
def rye_flour : ℕ := 5

/-- The amount of whole-wheat bread flour Sarah bought in pounds -/
def wheat_bread_flour : ℕ := 10

/-- The amount of chickpea flour Sarah bought in pounds -/
def chickpea_flour : ℕ := 3

/-- The amount of whole-wheat pastry flour Sarah already had at home in pounds -/
def pastry_flour : ℕ := 2

/-- The total amount of flour Sarah now has in pounds -/
def total_flour : ℕ := rye_flour + wheat_bread_flour + chickpea_flour + pastry_flour

theorem sarah_flour_amount : total_flour = 20 := by
  sorry

end sarah_flour_amount_l1131_113107


namespace S_min_value_l1131_113120

/-- The area function S(a) for a > 1 -/
noncomputable def S (a : ℝ) : ℝ := a^2 / Real.sqrt (a^2 - 1)

/-- Theorem stating the minimum value of S(a) -/
theorem S_min_value (a : ℝ) (h : a > 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ S (Real.sqrt 2) = min_val ∧ ∀ x > 1, S x ≥ min_val :=
by sorry

end S_min_value_l1131_113120


namespace percentage_calculation_l1131_113128

theorem percentage_calculation (number : ℝ) (p : ℝ) 
  (h1 : (4/5) * (3/8) * number = 24) 
  (h2 : p * number / 100 = 199.99999999999997) : 
  p = 250 := by sorry

end percentage_calculation_l1131_113128


namespace clubsuit_not_commutative_l1131_113145

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Define the clubsuit operation
def clubsuit (x y : ℝ) : ℝ := heartsuit x (y + 1)

-- Theorem stating that the equality is false
theorem clubsuit_not_commutative : ¬ (∀ x y : ℝ, clubsuit x y = clubsuit y x) := by
  sorry

end clubsuit_not_commutative_l1131_113145


namespace factors_of_N_squared_not_dividing_N_l1131_113106

theorem factors_of_N_squared_not_dividing_N : ∃ (S : Finset ℕ), 
  (∀ d ∈ S, d ∣ (2019^2 - 1)^2 ∧ ¬(d ∣ (2019^2 - 1))) ∧ 
  (∀ d : ℕ, d ∣ (2019^2 - 1)^2 ∧ ¬(d ∣ (2019^2 - 1)) → d ∈ S) ∧ 
  S.card = 157 := by
  sorry

end factors_of_N_squared_not_dividing_N_l1131_113106


namespace total_amount_divided_l1131_113101

/-- The total amount divided among A, B, and C is 3366.00000000000006 given the conditions. -/
theorem total_amount_divided (a b c : ℝ) 
  (h1 : a = (2/3) * b)
  (h2 : b = (1/4) * c)
  (h3 : a = 396.00000000000006) : 
  a + b + c = 3366.00000000000006 := by
  sorry

end total_amount_divided_l1131_113101


namespace scientists_from_usa_l1131_113119

theorem scientists_from_usa (total : ℕ) (europe : ℕ) (canada : ℕ) (usa : ℕ)
  (h1 : total = 70)
  (h2 : europe = total / 2)
  (h3 : canada = total / 5)
  (h4 : usa = total - (europe + canada)) :
  usa = 21 := by
  sorry

end scientists_from_usa_l1131_113119


namespace students_in_both_competitions_l1131_113170

/-- Given a class of students and information about their participation in two competitions,
    calculate the number of students who participated in both competitions. -/
theorem students_in_both_competitions
  (total : ℕ)
  (volleyball : ℕ)
  (track_field : ℕ)
  (none : ℕ)
  (h1 : total = 45)
  (h2 : volleyball = 12)
  (h3 : track_field = 20)
  (h4 : none = 19)
  : volleyball + track_field - (total - none) = 6 := by
  sorry

#check students_in_both_competitions

end students_in_both_competitions_l1131_113170


namespace carolyn_practice_ratio_l1131_113114

/-- Represents Carolyn's music practice schedule and calculates the ratio of violin to piano practice time -/
theorem carolyn_practice_ratio :
  let piano_daily := 20 -- minutes of piano practice per day
  let days_per_week := 6 -- number of practice days per week
  let weeks_per_month := 4 -- number of weeks in a month
  let total_monthly := 1920 -- total practice time in minutes per month

  let piano_monthly := piano_daily * days_per_week * weeks_per_month
  let violin_monthly := total_monthly - piano_monthly
  let violin_daily := violin_monthly / (days_per_week * weeks_per_month)

  (violin_daily : ℚ) / piano_daily = 3 / 1 := by
  sorry

end carolyn_practice_ratio_l1131_113114


namespace tangent_intersection_monotonicity_intervals_m_range_l1131_113192

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / x

theorem tangent_intersection (m : ℝ) :
  (∃ y, y = f m 1 ∧ y - f m 1 = (1 - m) * (0 - 1) ∧ y = 1) → m = 1 := by sorry

theorem monotonicity_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∧
  (m > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < m → f m x₁ > f m x₂) ∧
           (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂)) := by sorry

theorem m_range (m : ℝ) :
  (∀ a b, 0 < a ∧ a < b → (f m b - f m a) / (b - a) < 1) → m ≥ 1/4 := by sorry

end tangent_intersection_monotonicity_intervals_m_range_l1131_113192


namespace same_color_combination_probability_l1131_113111

def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8

def lucy_picks : ℕ := 2
def john_picks : ℕ := 2

theorem same_color_combination_probability :
  let probability_same_combination := (2 * (Nat.choose red_candies 2 * Nat.choose (red_candies - 2) 2 +
                                            Nat.choose blue_candies 2 * Nat.choose (blue_candies - 2) 2) +
                                       Nat.choose red_candies 2 * Nat.choose blue_candies 2 +
                                       Nat.choose blue_candies 2 * Nat.choose red_candies 2) /
                                      (Nat.choose total_candies 2 * Nat.choose (total_candies - 2) 2)
  probability_same_combination = 184 / 323 := by
  sorry

end same_color_combination_probability_l1131_113111


namespace quadrilateral_triangle_product_l1131_113115

/-- Represents a convex quadrilateral with its four triangles formed by diagonals -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by diagonals -/
  triangle_areas : Fin 4 → ℕ

/-- Theorem stating that the product of the four triangle areas in a convex quadrilateral
    cannot be congruent to 2014 modulo 10000 -/
theorem quadrilateral_triangle_product (q : ConvexQuadrilateral) :
  (q.triangle_areas 0 * q.triangle_areas 1 * q.triangle_areas 2 * q.triangle_areas 3) % 10000 ≠ 2014 := by
  sorry

end quadrilateral_triangle_product_l1131_113115


namespace journey_speed_l1131_113198

theorem journey_speed (D : ℝ) (V : ℝ) (h1 : D > 0) (h2 : V > 0) : 
  (2 * D) / (D / V + D / 30) = 40 → V = 60 := by
  sorry

end journey_speed_l1131_113198


namespace number_problem_l1131_113173

theorem number_problem : ∃ x : ℝ, (0.2 * x = 0.4 * 140 + 80) ∧ (x = 680) := by
  sorry

end number_problem_l1131_113173


namespace floor_length_is_ten_l1131_113197

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem: Given the conditions, the floor length is 10 meters -/
theorem floor_length_is_ten (floor : FloorWithRug)
  (h1 : floor.width = 8)
  (h2 : floor.strip_width = 2)
  (h3 : floor.rug_area = 24)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.length = 10 := by
  sorry

#check floor_length_is_ten

end floor_length_is_ten_l1131_113197


namespace sin_four_arcsin_l1131_113122

theorem sin_four_arcsin (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) :
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) := by
  sorry

end sin_four_arcsin_l1131_113122


namespace distance_difference_l1131_113154

/-- Clara's travel rate in miles per hour -/
def clara_rate : ℝ := 3.75

/-- Daniel's travel rate in miles per hour -/
def daniel_rate : ℝ := 3

/-- Time period in hours -/
def time : ℝ := 5

/-- Theorem stating the difference in distance traveled -/
theorem distance_difference : clara_rate * time - daniel_rate * time = 3.75 := by
  sorry

end distance_difference_l1131_113154


namespace computer_profit_percentage_l1131_113117

-- Define the computer's cost
variable (cost : ℝ)

-- Define the two selling prices
def selling_price_1 : ℝ := 2240
def selling_price_2 : ℝ := 2400

-- Define the profit percentages
def profit_percentage_1 : ℝ := 0.4  -- 40%
def profit_percentage_2 : ℝ := 0.5  -- 50%

-- Theorem statement
theorem computer_profit_percentage :
  (selling_price_2 - cost = profit_percentage_2 * cost) →
  (selling_price_1 - cost = profit_percentage_1 * cost) :=
by sorry

end computer_profit_percentage_l1131_113117


namespace class_size_l1131_113156

theorem class_size (n : ℕ) 
  (h1 : n < 50) 
  (h2 : n % 8 = 5) 
  (h3 : n % 6 = 4) : 
  n = 13 := by
sorry

end class_size_l1131_113156


namespace infiniteContinuedFraction_eq_infiniteContinuedFraction_value_l1131_113105

/-- The value of the infinite continued fraction 1 / (1 + 1 / (1 + ...)) -/
noncomputable def infiniteContinuedFraction : ℝ :=
  Real.sqrt 5 / 2 + 1 / 2

/-- The infinite continued fraction satisfies the equation x = 1 + 1/x -/
theorem infiniteContinuedFraction_eq : 
  infiniteContinuedFraction = 1 + 1 / infiniteContinuedFraction := by
sorry

/-- The infinite continued fraction 1 / (1 + 1 / (1 + ...)) is equal to (√5 + 1) / 2 -/
theorem infiniteContinuedFraction_value : 
  infiniteContinuedFraction = Real.sqrt 5 / 2 + 1 / 2 := by
sorry

end infiniteContinuedFraction_eq_infiniteContinuedFraction_value_l1131_113105


namespace bench_press_theorem_l1131_113124

def bench_press_problem (initial_weight : ℝ) (injury_reduction : ℝ) (training_multiplier : ℝ) : Prop :=
  let after_injury := initial_weight * (1 - injury_reduction)
  let final_weight := after_injury * training_multiplier
  final_weight = 300

theorem bench_press_theorem :
  bench_press_problem 500 0.8 3 := by
  sorry

end bench_press_theorem_l1131_113124


namespace ellipse_axis_ratio_l1131_113185

/-- Given an ellipse with equation x²/9 + y²/m² = 1 where 0 < m < 3,
    if the length of its major axis is twice that of its minor axis,
    then m = 3/2 -/
theorem ellipse_axis_ratio (m : ℝ) 
  (h1 : 0 < m) (h2 : m < 3) 
  (h3 : ∀ x y : ℝ, x^2/9 + y^2/m^2 = 1 → 6 = 2*(2*m)) : 
  m = 3/2 := by
sorry

end ellipse_axis_ratio_l1131_113185


namespace geometric_sequence_solution_l1131_113108

def isGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_solution (a : ℝ) (h : a > 0) 
  (h_seq : isGeometricSequence 280 a (180/49)) : 
  a = Real.sqrt (50400/49) := by
  sorry

end geometric_sequence_solution_l1131_113108


namespace total_toads_l1131_113195

theorem total_toads (in_pond : ℕ) (outside_pond : ℕ) 
  (h1 : in_pond = 12) (h2 : outside_pond = 6) : 
  in_pond + outside_pond = 18 := by
sorry

end total_toads_l1131_113195


namespace count_positive_area_triangles_l1131_113199

/-- The number of points in each row or column of the grid -/
def gridSize : ℕ := 6

/-- The total number of points in the grid -/
def totalPoints : ℕ := gridSize * gridSize

/-- The number of ways to choose 3 points from the total points -/
def totalCombinations : ℕ := Nat.choose totalPoints 3

/-- The number of ways to choose 3 points from a single row or column -/
def lineCombo : ℕ := Nat.choose gridSize 3

/-- The number of straight lines (rows and columns) -/
def numLines : ℕ := 2 * gridSize

/-- The number of main diagonals -/
def numMainDiagonals : ℕ := 2

/-- The number of triangles with positive area on the grid -/
def positiveAreaTriangles : ℕ := 
  totalCombinations - (numLines * lineCombo) - (numMainDiagonals * lineCombo)

theorem count_positive_area_triangles : positiveAreaTriangles = 6860 := by
  sorry

end count_positive_area_triangles_l1131_113199


namespace equation_solution_l1131_113151

theorem equation_solution : ∃ x : ℝ, (x - 5) ^ 4 = (1 / 16)⁻¹ ∧ x = 7 := by
  sorry

end equation_solution_l1131_113151


namespace train_crossing_time_l1131_113179

/-- The time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 120 →
  train_speed = 67 * (1000 / 3600) →
  man_speed = 5 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 6 := by
sorry


end train_crossing_time_l1131_113179


namespace equation_solution_l1131_113149

theorem equation_solution : 
  ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 57 ∧ x = 92 := by sorry

end equation_solution_l1131_113149


namespace spells_conversion_l1131_113132

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The number of spells in each book in base-9 --/
def spellsPerBook : List Nat := [5, 3, 6]

theorem spells_conversion :
  base9ToBase10 spellsPerBook = 518 := by
  sorry

#eval base9ToBase10 spellsPerBook

end spells_conversion_l1131_113132


namespace parabolas_intersection_l1131_113189

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 10 * x - 10
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-2, 18), (8, 38)}

-- Theorem statement
theorem parabolas_intersection :
  ∀ x y : ℝ, parabola1 x = parabola2 x ∧ y = parabola1 x ↔ (x, y) ∈ intersection_points :=
by sorry

end parabolas_intersection_l1131_113189


namespace largest_s_proof_l1131_113160

/-- The largest possible value of s for which there exist regular polygons P1 (r-gon) and P2 (s-gon)
    satisfying the given conditions -/
def largest_s : ℕ := 117

theorem largest_s_proof (r s : ℕ) : 
  r ≥ s → 
  s ≥ 3 → 
  (r - 2) * s * 60 = (s - 2) * r * 59 → 
  s ≤ largest_s := by
  sorry

#check largest_s_proof

end largest_s_proof_l1131_113160


namespace total_surveyed_is_185_l1131_113127

/-- Represents the total number of students surveyed in a stratified sampling method -/
def total_surveyed (grade10_total : ℕ) (grade11_total : ℕ) (grade12_total : ℕ) (grade12_surveyed : ℕ) : ℕ :=
  let grade10_surveyed := (grade10_total * grade12_surveyed) / grade12_total
  let grade11_surveyed := (grade11_total * grade12_surveyed) / grade12_total
  grade10_surveyed + grade11_surveyed + grade12_surveyed

/-- Theorem stating that the total number of students surveyed is 185 given the problem conditions -/
theorem total_surveyed_is_185 :
  total_surveyed 1000 1200 1500 75 = 185 := by
  sorry

end total_surveyed_is_185_l1131_113127


namespace arithmetic_statement_not_basic_unique_non_basic_statement_l1131_113161

/-- The set of basic algorithmic statements -/
def BasicAlgorithmicStatements : Set String :=
  {"input statement", "output statement", "assignment statement", "conditional statement", "loop statement"}

/-- The list of options given in the problem -/
def Options : List String :=
  ["assignment statement", "arithmetic statement", "conditional statement", "loop statement"]

/-- Theorem: The arithmetic statement is not a member of the set of basic algorithmic statements -/
theorem arithmetic_statement_not_basic : "arithmetic statement" ∉ BasicAlgorithmicStatements := by
  sorry

/-- Theorem: The arithmetic statement is the only option not in the set of basic algorithmic statements -/
theorem unique_non_basic_statement :
  ∀ s ∈ Options, s ∉ BasicAlgorithmicStatements → s = "arithmetic statement" := by
  sorry

end arithmetic_statement_not_basic_unique_non_basic_statement_l1131_113161


namespace three_zeros_implies_m_in_open_interval_l1131_113196

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + m

/-- The theorem stating that if f has three zeros, then m is in the open interval (-4, 4) -/
theorem three_zeros_implies_m_in_open_interval (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0) →
  m ∈ Set.Ioo (-4 : ℝ) 4 :=
by sorry

end three_zeros_implies_m_in_open_interval_l1131_113196


namespace divisibility_by_2016_l1131_113171

theorem divisibility_by_2016 (n : ℕ) : 
  2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := by
  sorry

end divisibility_by_2016_l1131_113171


namespace expression_evaluation_l1131_113135

theorem expression_evaluation (x y : ℚ) (hx : x = 5) (hy : y = 6) :
  (2 / y) / (2 / x) * 3 = 5 / 2 := by
  sorry

end expression_evaluation_l1131_113135


namespace restaurant_tip_percentage_l1131_113125

theorem restaurant_tip_percentage : 
  let james_meal : ℚ := 16
  let friend_meal : ℚ := 14
  let total_bill : ℚ := james_meal + friend_meal
  let james_paid : ℚ := 21
  let friend_paid : ℚ := total_bill / 2
  let tip : ℚ := james_paid - friend_paid
  tip / total_bill = 1/5 := by sorry

end restaurant_tip_percentage_l1131_113125


namespace working_days_is_twenty_main_theorem_l1131_113142

/-- Represents the commute data for a period of working days -/
structure CommuteData where
  car_to_work : ℕ
  train_from_work : ℕ
  total_train_trips : ℕ

/-- Calculates the total number of working days based on commute data -/
def calculate_working_days (data : CommuteData) : ℕ :=
  data.car_to_work + data.total_train_trips

/-- Theorem stating that the number of working days is 20 given the specific commute data -/
theorem working_days_is_twenty (data : CommuteData) 
  (h1 : data.car_to_work = 12)
  (h2 : data.train_from_work = 11)
  (h3 : data.total_train_trips = 8)
  (h4 : data.car_to_work = data.train_from_work + 1) :
  calculate_working_days data = 20 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ (data : CommuteData), 
  data.car_to_work = 12 ∧ 
  data.train_from_work = 11 ∧ 
  data.total_train_trips = 8 ∧ 
  data.car_to_work = data.train_from_work + 1 ∧
  calculate_working_days data = 20 := by
  sorry

end working_days_is_twenty_main_theorem_l1131_113142


namespace range_of_m_l1131_113190

def P (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m : ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ∈ Set.Ioc 1 2 ∪ Set.Ici 3 := by sorry

end range_of_m_l1131_113190


namespace farm_animals_l1131_113157

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 8) 
  (h2 : total_legs = 24) : 
  ∃ (ducks dogs : ℕ), 
    ducks + dogs = total_animals ∧ 
    2 * ducks + 4 * dogs = total_legs ∧ 
    ducks = 4 := by
  sorry

end farm_animals_l1131_113157


namespace r_six_times_thirty_l1131_113103

/-- The function r as defined in the problem -/
def r (θ : ℚ) : ℚ := 1 / (2 - θ)

/-- The composition of r with itself n times -/
def r_n (n : ℕ) (θ : ℚ) : ℚ :=
  match n with
  | 0 => θ
  | n + 1 => r (r_n n θ)

/-- The main theorem stating that applying r six times to 30 results in 22/23 -/
theorem r_six_times_thirty : r_n 6 30 = 22 / 23 := by
  sorry

end r_six_times_thirty_l1131_113103


namespace prime_pairs_dividing_sum_of_powers_l1131_113126

theorem prime_pairs_dividing_sum_of_powers (p q : ℕ) : 
  Prime p → Prime q → (p * q) ∣ (2^p + 2^q) → 
  ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end prime_pairs_dividing_sum_of_powers_l1131_113126


namespace equal_numbers_product_l1131_113109

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 ∧ 
  a = 22 ∧ 
  b = 34 ∧ 
  c = d → 
  c * d = 144 := by
sorry

end equal_numbers_product_l1131_113109


namespace ratio_of_13th_terms_l1131_113147

/-- Two arithmetic sequences with sums U_n and V_n for the first n terms -/
def arithmetic_sequences (U V : ℕ → ℚ) : Prop :=
  ∃ (a b c d : ℚ), ∀ n : ℕ,
    U n = n * (2 * a + (n - 1) * b) / 2 ∧
    V n = n * (2 * c + (n - 1) * d) / 2

/-- The ratio condition for U_n and V_n -/
def ratio_condition (U V : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, U n * (3 * n + 17) = V n * (5 * n + 3)

/-- The 13th term of an arithmetic sequence -/
def term_13 (seq : ℕ → ℚ) : ℚ :=
  seq 13 - seq 12

/-- Main theorem -/
theorem ratio_of_13th_terms
  (U V : ℕ → ℚ)
  (h1 : arithmetic_sequences U V)
  (h2 : ratio_condition U V) :
  term_13 U / term_13 V = 52 / 89 := by
  sorry

end ratio_of_13th_terms_l1131_113147


namespace negation_of_universal_proposition_l1131_113102

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 1) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 1) :=
by sorry

end negation_of_universal_proposition_l1131_113102


namespace function_equality_implies_zero_l1131_113140

/-- Given a function f(x, y) = kx + 1/y, prove that if f(a, b) = f(b, a) for a ≠ b, then f(ab, 1) = 0 -/
theorem function_equality_implies_zero (k : ℝ) (a b : ℝ) (h1 : a ≠ b) :
  (k * a + 1 / b = k * b + 1 / a) → (k * (a * b) + 1 = 0) :=
by sorry

end function_equality_implies_zero_l1131_113140


namespace chip_price_reduction_l1131_113148

/-- Represents the price reduction process for a chip -/
theorem chip_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 256) 
  (h2 : final_price = 196) 
  (x : ℝ) -- x represents the percentage of each price reduction
  (h3 : 0 ≤ x ∧ x < 1) -- ensure x is a valid percentage
  : initial_price * (1 - x)^2 = final_price ↔ 
    initial_price * (1 - x)^2 = 196 ∧ initial_price = 256 :=
by sorry

end chip_price_reduction_l1131_113148


namespace circle_radius_with_tangents_l1131_113144

/-- Given a circle with parallel tangents and a third tangent, prove the radius. -/
theorem circle_radius_with_tangents 
  (AB CD DE : ℝ) 
  (h_AB : AB = 7)
  (h_CD : CD = 12)
  (h_DE : DE = 3) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 5 := by
sorry


end circle_radius_with_tangents_l1131_113144


namespace triangle_sin_c_l1131_113181

theorem triangle_sin_c (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a = 1 →
  b = Real.sqrt 2 →
  A + C = 2 * B →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sin C = 1 / Real.sqrt 2 := by
  sorry

end triangle_sin_c_l1131_113181


namespace marathon_average_time_l1131_113167

/-- Given Casey's time to complete a marathon and Zendaya's relative time compared to Casey,
    calculate the average time for both to complete the race. -/
theorem marathon_average_time (casey_time : ℝ) (zendaya_relative_time : ℝ) :
  casey_time = 6 →
  zendaya_relative_time = 1/3 →
  let zendaya_time := casey_time + zendaya_relative_time * casey_time
  (casey_time + zendaya_time) / 2 = 7 := by
  sorry


end marathon_average_time_l1131_113167


namespace triangle_properties_l1131_113130

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (a * Real.cos B - b * Real.cos A = c - b) →
  (Real.tan A + Real.tan B + Real.tan C - Real.sqrt 3 * Real.tan B * Real.tan C = 0) →
  ((1/2) * a * (b * Real.sin B + c * Real.sin C - a * Real.sin A) = (1/2) * a * b * Real.sin C) →
  -- Conclusions
  (A = π/3) ∧
  (a = 8 → (1/2) * a * b * Real.sin C = 11 * Real.sqrt 3) :=
by sorry

end triangle_properties_l1131_113130


namespace chocolate_bars_to_sell_l1131_113193

theorem chocolate_bars_to_sell (initial : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) 
  (h1 : initial = 18)
  (h2 : sold_week1 = 5)
  (h3 : sold_week2 = 7) :
  initial - (sold_week1 + sold_week2) = 6 := by
  sorry

end chocolate_bars_to_sell_l1131_113193


namespace max_area_inscribed_rectangle_optimal_rectangle_sides_l1131_113163

/-- The maximum area of a rectangle inscribed in a right triangle -/
theorem max_area_inscribed_rectangle (h : Real) (α : Real) (x y : Real) :
  h = 24 →                   -- Hypotenuse is 24 cm
  α = π / 3 →                -- One angle is 60°
  0 < x →                    -- Length of rectangle is positive
  0 < y →                    -- Width of rectangle is positive
  y = h - (4 * x * Real.sqrt 3) / 3 →  -- Relationship between x and y
  x * y ≤ 12 * 3 * Real.sqrt 3 :=
by sorry

/-- The sides of the rectangle that achieve maximum area -/
theorem optimal_rectangle_sides (h : Real) (α : Real) (x y : Real) :
  h = 24 →                   -- Hypotenuse is 24 cm
  α = π / 3 →                -- One angle is 60°
  0 < x →                    -- Length of rectangle is positive
  0 < y →                    -- Width of rectangle is positive
  y = h - (4 * x * Real.sqrt 3) / 3 →  -- Relationship between x and y
  x * y = 12 * 3 * Real.sqrt 3 →       -- Maximum area condition
  x = 3 * Real.sqrt 3 ∧ y = 12 :=
by sorry

end max_area_inscribed_rectangle_optimal_rectangle_sides_l1131_113163


namespace inequality_proof_l1131_113152

theorem inequality_proof (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  0 < (x - Real.sin x) / (Real.tan x - Real.sin x) ∧
  (x - Real.sin x) / (Real.tan x - Real.sin x) < 1 / 3 := by
  sorry

end inequality_proof_l1131_113152


namespace gondor_laptop_repair_fee_l1131_113184

/-- The amount Gondor earns from repairing a phone -/
def phone_repair_fee : ℝ := 10

/-- The number of phones Gondor repaired on Monday -/
def monday_phones : ℕ := 3

/-- The number of phones Gondor repaired on Tuesday -/
def tuesday_phones : ℕ := 5

/-- The number of laptops Gondor repaired on Wednesday -/
def wednesday_laptops : ℕ := 2

/-- The number of laptops Gondor repaired on Thursday -/
def thursday_laptops : ℕ := 4

/-- The total amount Gondor earned -/
def total_earnings : ℝ := 200

/-- The amount Gondor earns from repairing a laptop -/
def laptop_repair_fee : ℝ := 20

theorem gondor_laptop_repair_fee :
  laptop_repair_fee = 20 ∧
  (monday_phones + tuesday_phones) * phone_repair_fee +
  (wednesday_laptops + thursday_laptops) * laptop_repair_fee = total_earnings :=
by sorry

end gondor_laptop_repair_fee_l1131_113184


namespace binomial_30_3_l1131_113187

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by sorry

end binomial_30_3_l1131_113187


namespace waterpark_total_cost_calculation_l1131_113121

def waterpark_total_cost (adult_price child_price teen_price : ℚ)
                         (num_adults num_children num_teens : ℕ)
                         (activity_discount coupon_discount : ℚ)
                         (soda_price : ℚ) (num_sodas : ℕ) : ℚ :=
  let base_cost := adult_price * num_adults + child_price * num_children + teen_price * num_teens
  let discounted_cost := base_cost * (1 - activity_discount) * (1 - coupon_discount)
  let soda_cost := soda_price * num_sodas
  discounted_cost + soda_cost

theorem waterpark_total_cost_calculation :
  waterpark_total_cost 30 15 20 4 2 4 (1/10) (1/20) 5 5 = 221.65 := by
  sorry

end waterpark_total_cost_calculation_l1131_113121


namespace coronavirus_recoveries_l1131_113169

/-- Calculates the number of recoveries on the third day of a coronavirus outbreak --/
theorem coronavirus_recoveries 
  (initial_cases : ℕ) 
  (second_day_new_cases : ℕ) 
  (second_day_recoveries : ℕ) 
  (third_day_new_cases : ℕ) 
  (final_total_cases : ℕ) 
  (h1 : initial_cases = 2000)
  (h2 : second_day_new_cases = 500)
  (h3 : second_day_recoveries = 50)
  (h4 : third_day_new_cases = 1500)
  (h5 : final_total_cases = 3750) :
  initial_cases + second_day_new_cases - second_day_recoveries + third_day_new_cases - final_total_cases = 200 :=
by
  sorry

#check coronavirus_recoveries

end coronavirus_recoveries_l1131_113169


namespace all_divisible_by_nine_l1131_113194

/-- A five-digit number represented as a tuple of five natural numbers -/
def FiveDigitNumber := (ℕ × ℕ × ℕ × ℕ × ℕ)

/-- The sum of digits in a five-digit number -/
def digitSum (n : FiveDigitNumber) : ℕ :=
  n.1 + n.2.1 + n.2.2.1 + n.2.2.2.1 + n.2.2.2.2

/-- Predicate for a valid five-digit number -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  1 ≤ n.1 ∧ n.1 ≤ 9 ∧ 0 ≤ n.2.1 ∧ n.2.1 ≤ 9 ∧
  0 ≤ n.2.2.1 ∧ n.2.2.1 ≤ 9 ∧ 0 ≤ n.2.2.2.1 ∧ n.2.2.2.1 ≤ 9 ∧
  0 ≤ n.2.2.2.2 ∧ n.2.2.2.2 ≤ 9

/-- The set of all valid five-digit numbers with digit sum 36 -/
def S : Set FiveDigitNumber :=
  {n | isValidFiveDigitNumber n ∧ digitSum n = 36}

/-- The numeric value of a five-digit number -/
def numericValue (n : FiveDigitNumber) : ℕ :=
  10000 * n.1 + 1000 * n.2.1 + 100 * n.2.2.1 + 10 * n.2.2.2.1 + n.2.2.2.2

theorem all_divisible_by_nine :
  ∀ n ∈ S, (numericValue n) % 9 = 0 := by
  sorry

end all_divisible_by_nine_l1131_113194


namespace fraction_to_decimal_decimal_representation_main_result_l1131_113191

theorem fraction_to_decimal : (47 : ℚ) / (2 * 5^4) = (376 : ℚ) / 10000 := by sorry

theorem decimal_representation : (376 : ℚ) / 10000 = 0.0376 := by sorry

theorem main_result : (47 : ℚ) / (2 * 5^4) = 0.0376 := by sorry

end fraction_to_decimal_decimal_representation_main_result_l1131_113191


namespace hcf_from_lcm_and_product_l1131_113182

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750) 
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
sorry

end hcf_from_lcm_and_product_l1131_113182


namespace remainder_problem_l1131_113155

theorem remainder_problem (G : ℕ) (h1 : G = 144) (h2 : 6215 % G = 23) : 7373 % G = 29 := by
  sorry

end remainder_problem_l1131_113155


namespace geometric_sequence_sum_l1131_113146

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 1 + a 2 + a 3 + a 4 = 1) →  -- sum of first 4 terms is 1
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17) := by
sorry

end geometric_sequence_sum_l1131_113146


namespace worker_efficiency_l1131_113113

theorem worker_efficiency (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  p = 1 / 22 → p + q = 1 / 12 → p / q = 6 / 5 := by
  sorry

end worker_efficiency_l1131_113113


namespace trig_values_for_point_l1131_113136

/-- Given a point P(-√3, m) on the terminal side of angle α, where m ≠ 0 and sin α = (√2 * m) / 4,
    prove the values of m, cos α, and tan α. -/
theorem trig_values_for_point (m : ℝ) (α : ℝ) (h1 : m ≠ 0) (h2 : Real.sin α = (Real.sqrt 2 * m) / 4) :
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  (m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
  (m < 0 → Real.tan α = Real.sqrt 15 / 3) := by
  sorry

end trig_values_for_point_l1131_113136


namespace real_part_of_z_l1131_113186

theorem real_part_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : 
  z.re = 3 / 25 := by
sorry

end real_part_of_z_l1131_113186


namespace louis_current_age_l1131_113168

/-- Carla's current age -/
def carla_age : ℕ := 30 - 6

/-- Louis's current age -/
def louis_age : ℕ := 55 - carla_age

theorem louis_current_age : louis_age = 31 := by
  sorry

end louis_current_age_l1131_113168
