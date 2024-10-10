import Mathlib

namespace simplify_power_expression_l1580_158058

theorem simplify_power_expression (x : ℝ) : (3 * (2 * x)^5)^4 = 84934656 * x^20 := by
  sorry

end simplify_power_expression_l1580_158058


namespace max_cookies_andy_l1580_158004

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  andy : Nat
  alexa : Nat
  john : Nat

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.andy + d.alexa + d.john = 36 ∧
  d.andy % d.alexa = 0 ∧
  d.andy % d.john = 0 ∧
  d.alexa > 0 ∧
  d.john > 0

/-- Theorem stating the maximum number of cookies Andy could have eaten -/
theorem max_cookies_andy :
  ∀ d : CookieDistribution, isValidDistribution d → d.andy ≤ 30 :=
sorry

end max_cookies_andy_l1580_158004


namespace parabola_shift_up_two_l1580_158087

/-- Represents a vertical shift transformation of a parabola -/
def verticalShift (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f x + k

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := λ x => x^2

/-- Theorem: Shifting the parabola y = x^2 up by 2 units results in y = x^2 + 2 -/
theorem parabola_shift_up_two :
  verticalShift originalParabola 2 = λ x => x^2 + 2 := by
  sorry

end parabola_shift_up_two_l1580_158087


namespace no_prime_pairs_with_integer_ratios_l1580_158034

theorem no_prime_pairs_with_integer_ratios : 
  ¬ ∃ (x y : ℕ), Prime x ∧ Prime y ∧ y < x ∧ x ≤ 200 ∧ 
  (x / y : ℚ).isInt ∧ ((x + 1) / (y + 1) : ℚ).isInt := by
  sorry

end no_prime_pairs_with_integer_ratios_l1580_158034


namespace percentage_of_male_students_l1580_158086

theorem percentage_of_male_students
  (total_percentage : ℝ)
  (male_percentage : ℝ)
  (female_percentage : ℝ)
  (male_older_25 : ℝ)
  (female_older_25 : ℝ)
  (prob_younger_25 : ℝ)
  (h1 : total_percentage = male_percentage + female_percentage)
  (h2 : total_percentage = 100)
  (h3 : male_older_25 = 40)
  (h4 : female_older_25 = 20)
  (h5 : prob_younger_25 = 0.72)
  (h6 : prob_younger_25 = (1 - male_older_25 / 100) * male_percentage / 100 +
                          (1 - female_older_25 / 100) * female_percentage / 100) :
  male_percentage = 40 := by
sorry

end percentage_of_male_students_l1580_158086


namespace integer_expression_l1580_158043

theorem integer_expression (m : ℤ) : ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = k := by
  sorry

end integer_expression_l1580_158043


namespace trailing_zeros_remainder_l1580_158023

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the product of factorials from 1 to 120
def productOfFactorials : ℕ := (List.range 120).foldl (λ acc i => acc * factorial (i + 1)) 1

-- Define the function to count trailing zeros
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailingZeros (n / 10)
  else 0

-- Theorem statement
theorem trailing_zeros_remainder :
  (trailingZeros productOfFactorials) % 1000 = 224 := by sorry

end trailing_zeros_remainder_l1580_158023


namespace sum_three_consecutive_not_prime_l1580_158055

theorem sum_three_consecutive_not_prime (n : ℕ) : ¬ Prime (3 * (n + 1)) := by
  sorry

#check sum_three_consecutive_not_prime

end sum_three_consecutive_not_prime_l1580_158055


namespace farmer_land_ownership_l1580_158065

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.8 + 0.9 * total_land * 0.1 + 90 = 0.9 * total_land) →
  total_land = 1000 := by
sorry

end farmer_land_ownership_l1580_158065


namespace square_EFGH_side_length_l1580_158038

/-- Square ABCD with side length 10 cm -/
def square_ABCD : Real := 10

/-- Distance of line p from side AB -/
def line_p_distance : Real := 6.5

/-- Area difference between the two parts divided by line p -/
def area_difference : Real := 13.8

/-- Side length of square EFGH -/
def square_EFGH_side : Real := 5.4

theorem square_EFGH_side_length :
  ∃ (square_EFGH : Real),
    square_EFGH = square_EFGH_side ∧
    square_EFGH > 0 ∧
    square_EFGH < square_ABCD ∧
    (square_ABCD - square_EFGH) * line_p_distance = area_difference / 2 ∧
    (square_ABCD - square_EFGH) * (square_ABCD - line_p_distance) = area_difference / 2 :=
by sorry

end square_EFGH_side_length_l1580_158038


namespace y_to_x_equals_one_l1580_158008

theorem y_to_x_equals_one (x y : ℝ) (h : (y + 1)^2 + Real.sqrt (x - 2) = 0) : y^x = 1 := by
  sorry

end y_to_x_equals_one_l1580_158008


namespace store_pricing_l1580_158067

theorem store_pricing (h n : ℝ) 
  (eq1 : 4 * h + 5 * n = 10.45)
  (eq2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 := by
  sorry

end store_pricing_l1580_158067


namespace probability_two_pairs_l1580_158026

def total_socks : ℕ := 10
def drawn_socks : ℕ := 4
def distinct_pairs : ℕ := 5

theorem probability_two_pairs : 
  (Nat.choose distinct_pairs 2) / (Nat.choose total_socks drawn_socks) = 1 / 21 := by
  sorry

end probability_two_pairs_l1580_158026


namespace draw_three_cards_not_same_color_l1580_158011

/-- Given a set of 16 cards with 4 of each color (red, yellow, blue, green),
    this theorem states that the number of ways to draw 3 cards such that
    they are not all the same color is equal to C(16,3) - 4 * C(4,3). -/
theorem draw_three_cards_not_same_color (total_cards : ℕ) (cards_per_color : ℕ) 
  (num_colors : ℕ) (draw : ℕ) (h1 : total_cards = 16) (h2 : cards_per_color = 4) 
  (h3 : num_colors = 4) (h4 : draw = 3) :
  (Nat.choose total_cards draw) - (num_colors * Nat.choose cards_per_color draw) = 544 := by
  sorry

end draw_three_cards_not_same_color_l1580_158011


namespace discounted_price_theorem_l1580_158001

def original_price : ℝ := 760
def discount_percentage : ℝ := 75

theorem discounted_price_theorem :
  original_price * (1 - discount_percentage / 100) = 570 := by
  sorry

end discounted_price_theorem_l1580_158001


namespace min_value_expression_l1580_158032

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_xyz : x * y * z = 2/3) : 
  x^2 + 6*x*y + 18*y^2 + 12*y*z + 4*z^2 ≥ 18 := by
sorry

end min_value_expression_l1580_158032


namespace remaining_donuts_l1580_158021

theorem remaining_donuts (initial_donuts : ℕ) (missing_percentage : ℚ) 
  (h1 : initial_donuts = 30)
  (h2 : missing_percentage = 70/100) :
  ↑initial_donuts * (1 - missing_percentage) = 9 :=
by sorry

end remaining_donuts_l1580_158021


namespace positive_solutions_conditions_l1580_158099

theorem positive_solutions_conditions (a m x y z : ℝ) : 
  (x + y - z = 2 * a) →
  (x^2 + y^2 = z^2) →
  (m * (x + y) = x * y) →
  (x > 0 ∧ y > 0 ∧ z > 0) ↔ 
  (a / 2 * (2 + Real.sqrt 2) ≤ m ∧ m ≤ 2 * a ∧ a > 0) :=
by sorry

end positive_solutions_conditions_l1580_158099


namespace average_of_remaining_numbers_l1580_158068

theorem average_of_remaining_numbers
  (n : ℕ)
  (total_avg : ℚ)
  (first_three_avg : ℚ)
  (next_three_avg : ℚ)
  (h1 : n = 8)
  (h2 : total_avg = 4.5)
  (h3 : first_three_avg = 5.2)
  (h4 : next_three_avg = 3.6) :
  (n * total_avg - 3 * first_three_avg - 3 * next_three_avg) / 2 = 4.8 := by
  sorry

end average_of_remaining_numbers_l1580_158068


namespace percentage_difference_l1580_158095

theorem percentage_difference (x y : ℝ) (h : x = 7 * y) :
  (x - y) / x * 100 = (6 / 7) * 100 := by
  sorry

end percentage_difference_l1580_158095


namespace solution_set_for_m_eq_3_min_m_value_l1580_158069

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |m * x + 1| + |2 * x - 1|

-- Part I
theorem solution_set_for_m_eq_3 :
  {x : ℝ | f 3 x > 4} = {x : ℝ | x < -4/5 ∨ x > 4/5} :=
sorry

-- Part II
theorem min_m_value (m : ℝ) (h1 : 0 < m) (h2 : m < 2) 
  (h3 : ∀ x : ℝ, f m x ≥ 3 / (2 * m)) :
  m ≥ 1 :=
sorry

end solution_set_for_m_eq_3_min_m_value_l1580_158069


namespace decimal_sum_l1580_158052

/-- The sum of 0.403, 0.0007, and 0.07 is equal to 0.4737 -/
theorem decimal_sum : 0.403 + 0.0007 + 0.07 = 0.4737 := by
  sorry

end decimal_sum_l1580_158052


namespace probability_two_A_grades_l1580_158066

/-- The probability of achieving an A grade in exactly two out of three subjects. -/
theorem probability_two_A_grades
  (p_politics : ℝ)
  (p_history : ℝ)
  (p_geography : ℝ)
  (hp_politics : p_politics = 4/5)
  (hp_history : p_history = 3/5)
  (hp_geography : p_geography = 2/5)
  (hprob_politics : 0 ≤ p_politics ∧ p_politics ≤ 1)
  (hprob_history : 0 ≤ p_history ∧ p_history ≤ 1)
  (hprob_geography : 0 ≤ p_geography ∧ p_geography ≤ 1) :
  p_politics * p_history * (1 - p_geography) +
  p_politics * (1 - p_history) * p_geography +
  (1 - p_politics) * p_history * p_geography = 58/125 := by
sorry

end probability_two_A_grades_l1580_158066


namespace age_difference_l1580_158031

/-- The difference in ages between two people given a ratio and one person's age -/
theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 24.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 7 := by
sorry

end age_difference_l1580_158031


namespace square_sum_xy_l1580_158051

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 40)
  (h2 : y * (x + y) = 90)
  (h3 : x - y = 5) :
  (x + y)^2 = 130 := by
sorry

end square_sum_xy_l1580_158051


namespace mans_age_l1580_158047

theorem mans_age (P : ℝ) 
  (h1 : P = 1.25 * (P - 10)) 
  (h2 : P = (250 / 300) * (P + 10)) : 
  P = 50 := by
  sorry

end mans_age_l1580_158047


namespace absolute_value_equation_solution_l1580_158079

theorem absolute_value_equation_solution :
  ∃! n : ℝ, |2 * n + 8| = 3 * n - 4 := by
  sorry

end absolute_value_equation_solution_l1580_158079


namespace point_coordinates_l1580_158064

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (P : Point) 
  (h1 : is_in_second_quadrant P)
  (h2 : distance_to_x_axis P = 2)
  (h3 : distance_to_y_axis P = 3) :
  P.x = -3 ∧ P.y = 2 :=
by sorry

end point_coordinates_l1580_158064


namespace correct_probability_l1580_158048

-- Define the set of balls
inductive Ball : Type
| Red1 : Ball
| Red2 : Ball
| Red3 : Ball
| White2 : Ball
| White3 : Ball

-- Define a function to check if two balls have different colors and numbers
def differentColorAndNumber (b1 b2 : Ball) : Prop :=
  match b1, b2 with
  | Ball.Red1, Ball.White2 => True
  | Ball.Red1, Ball.White3 => True
  | Ball.Red2, Ball.White3 => True
  | Ball.Red3, Ball.White2 => True
  | _, _ => False

-- Define the probability of drawing two balls with different colors and numbers
def probabilityDifferentColorAndNumber : ℚ :=
  2 / 5

-- State the theorem
theorem correct_probability :
  probabilityDifferentColorAndNumber = 2 / 5 := by
  sorry


end correct_probability_l1580_158048


namespace line_plane_perpendicularity_l1580_158091

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel m n → perpendicular n α :=
sorry

end line_plane_perpendicularity_l1580_158091


namespace sin_2alpha_plus_beta_l1580_158018

theorem sin_2alpha_plus_beta (p α β : ℝ) : 
  (∀ x, x^2 - 4*p*x - 2 = 1 → x = Real.tan α ∨ x = Real.tan β) →
  Real.sin (2 * (α + β)) = (2 * p) / (p^2 + 1) := by
sorry

end sin_2alpha_plus_beta_l1580_158018


namespace maggie_picked_40_apples_l1580_158028

/-- The number of apples Kelsey picked -/
def kelsey_apples : ℕ := 28

/-- The number of apples Layla picked -/
def layla_apples : ℕ := 22

/-- The average number of apples picked by the three -/
def average_apples : ℕ := 30

/-- The number of people who picked apples -/
def num_people : ℕ := 3

/-- The number of apples Maggie picked -/
def maggie_apples : ℕ := 40

theorem maggie_picked_40_apples :
  kelsey_apples + layla_apples + maggie_apples = average_apples * num_people :=
by sorry

end maggie_picked_40_apples_l1580_158028


namespace non_red_percentage_is_27_percent_l1580_158042

/-- Represents the car population data for a city --/
structure CarPopulation where
  total : ℕ
  honda : ℕ
  toyota : ℕ
  nissan : ℕ
  honda_red_ratio : ℚ
  toyota_red_ratio : ℚ
  nissan_red_ratio : ℚ

/-- Calculate the percentage of non-red cars in the given car population --/
def non_red_percentage (pop : CarPopulation) : ℚ :=
  let total_red := pop.honda * pop.honda_red_ratio +
                   pop.toyota * pop.toyota_red_ratio +
                   pop.nissan * pop.nissan_red_ratio
  let total_non_red := pop.total - total_red
  (total_non_red / pop.total) * 100

/-- The theorem stating that the percentage of non-red cars is 27% --/
theorem non_red_percentage_is_27_percent (pop : CarPopulation)
  (h1 : pop.total = 30000)
  (h2 : pop.honda = 12000)
  (h3 : pop.toyota = 10000)
  (h4 : pop.nissan = 8000)
  (h5 : pop.honda_red_ratio = 80 / 100)
  (h6 : pop.toyota_red_ratio = 75 / 100)
  (h7 : pop.nissan_red_ratio = 60 / 100) :
  non_red_percentage pop = 27 := by
  sorry

end non_red_percentage_is_27_percent_l1580_158042


namespace non_student_ticket_price_l1580_158010

/-- Proves that the price of a non-student ticket was $8 -/
theorem non_student_ticket_price :
  let total_tickets : ℕ := 150
  let student_ticket_price : ℕ := 5
  let total_revenue : ℕ := 930
  let student_tickets_sold : ℕ := 90
  let non_student_tickets_sold : ℕ := 60
  let non_student_ticket_price : ℕ := (total_revenue - student_ticket_price * student_tickets_sold) / non_student_tickets_sold
  non_student_ticket_price = 8 := by
  sorry

end non_student_ticket_price_l1580_158010


namespace xian_temp_difference_l1580_158041

/-- Given the highest and lowest temperatures on a day, calculate the maximum temperature difference. -/
def max_temp_difference (highest lowest : ℝ) : ℝ :=
  highest - lowest

/-- Theorem: The maximum temperature difference on January 1, 2008 in Xi'an was 6°C. -/
theorem xian_temp_difference :
  let highest : ℝ := 3
  let lowest : ℝ := -3
  max_temp_difference highest lowest = 6 := by
  sorry

end xian_temp_difference_l1580_158041


namespace min_value_theorem_l1580_158005

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + 2*y = 3) :
  ∃ (min_val : ℝ), min_val = 8/3 ∧ 
  ∀ (z : ℝ), z = (1 / (x - y)) + (9 / (x + 5*y)) → z ≥ min_val :=
sorry

end min_value_theorem_l1580_158005


namespace square_of_negative_half_a_squared_b_l1580_158002

theorem square_of_negative_half_a_squared_b (a b : ℝ) :
  (- (1/2 : ℝ) * a^2 * b)^2 = (1/4 : ℝ) * a^4 * b^2 := by
  sorry

end square_of_negative_half_a_squared_b_l1580_158002


namespace girls_to_boys_ratio_l1580_158057

theorem girls_to_boys_ratio (total_students : ℕ) 
  (girls boys : ℕ) 
  (girls_with_dogs : ℚ) 
  (boys_with_dogs : ℚ) 
  (total_with_dogs : ℕ) :
  total_students = 100 →
  girls + boys = total_students →
  girls_with_dogs = 1/5 →
  boys_with_dogs = 1/10 →
  total_with_dogs = 15 →
  girls_with_dogs * girls + boys_with_dogs * boys = total_with_dogs →
  girls = boys :=
by sorry

end girls_to_boys_ratio_l1580_158057


namespace add_negative_and_positive_l1580_158094

theorem add_negative_and_positive : -3 + 5 = 2 := by
  sorry

end add_negative_and_positive_l1580_158094


namespace fundraiser_hourly_rate_l1580_158062

/-- Proves that if 8 volunteers working 40 hours each at $18 per hour raise the same total amount
    as 12 volunteers working 32 hours each, then the hourly rate for the second group is $15. -/
theorem fundraiser_hourly_rate
  (volunteers_last_week : ℕ)
  (hours_last_week : ℕ)
  (rate_last_week : ℚ)
  (volunteers_this_week : ℕ)
  (hours_this_week : ℕ)
  (h1 : volunteers_last_week = 8)
  (h2 : hours_last_week = 40)
  (h3 : rate_last_week = 18)
  (h4 : volunteers_this_week = 12)
  (h5 : hours_this_week = 32)
  (h6 : volunteers_last_week * hours_last_week * rate_last_week =
        volunteers_this_week * hours_this_week * (15 : ℚ)) :
  15 = (volunteers_last_week * hours_last_week * rate_last_week) /
       (volunteers_this_week * hours_this_week) :=
by sorry

end fundraiser_hourly_rate_l1580_158062


namespace math_club_election_l1580_158072

theorem math_club_election (total_candidates : ℕ) (positions : ℕ) (past_officers : ℕ) 
  (h1 : total_candidates = 20)
  (h2 : positions = 5)
  (h3 : past_officers = 10) :
  (Nat.choose total_candidates positions) - (Nat.choose (total_candidates - past_officers) positions) = 15252 := by
sorry

end math_club_election_l1580_158072


namespace cosine_product_equality_l1580_158033

theorem cosine_product_equality : 
  3.416 * Real.cos (π/33) * Real.cos (2*π/33) * Real.cos (4*π/33) * Real.cos (8*π/33) * Real.cos (16*π/33) = 1/32 := by
  sorry

end cosine_product_equality_l1580_158033


namespace jonathan_typing_time_l1580_158020

/-- Represents the time it takes for Jonathan to type the document alone -/
def jonathan_time : ℝ := 40

/-- Represents the time it takes for Susan to type the document alone -/
def susan_time : ℝ := 30

/-- Represents the time it takes for Jack to type the document alone -/
def jack_time : ℝ := 24

/-- Represents the time it takes for all three to type the document together -/
def combined_time : ℝ := 10

/-- Theorem stating that Jonathan's individual typing time satisfies the given conditions -/
theorem jonathan_typing_time :
  1 / jonathan_time + 1 / susan_time + 1 / jack_time = 1 / combined_time :=
by sorry

end jonathan_typing_time_l1580_158020


namespace percentage_relation_l1580_158074

theorem percentage_relation (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end percentage_relation_l1580_158074


namespace sector_angle_l1580_158056

/-- A circular sector with area 1 cm² and perimeter 4 cm has a central angle of 2 radians. -/
theorem sector_angle (r : ℝ) (α : ℝ) : 
  (1/2 * α * r^2 = 1) → (2*r + α*r = 4) → α = 2 := by
  sorry

end sector_angle_l1580_158056


namespace is_ellipse_l1580_158060

/-- The equation √((x-2)² + (y+2)²) + √((x-6)² + y²) = 12 represents an ellipse -/
theorem is_ellipse (x y : ℝ) : 
  (∃ (f₁ f₂ : ℝ × ℝ), f₁ ≠ f₂ ∧ 
  (∀ (p : ℝ × ℝ), Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
                   Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 12) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1) :=
by sorry

end is_ellipse_l1580_158060


namespace three_lines_determine_plane_l1580_158024

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for planes in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Function to check if two lines intersect
def linesIntersect (l1 l2 : Line3D) : Prop := sorry

-- Function to check if three lines intersect at the same point
def threeLinesSameIntersection (l1 l2 l3 : Line3D) : Prop := sorry

-- Function to determine if three lines define a unique plane
def defineUniquePlane (l1 l2 l3 : Line3D) : Prop := sorry

-- Theorem stating that three lines intersecting pairwise but not at the same point determine a unique plane
theorem three_lines_determine_plane (l1 l2 l3 : Line3D) :
  linesIntersect l1 l2 ∧ linesIntersect l2 l3 ∧ linesIntersect l3 l1 ∧
  ¬threeLinesSameIntersection l1 l2 l3 →
  defineUniquePlane l1 l2 l3 := by sorry

end three_lines_determine_plane_l1580_158024


namespace alyssa_fruit_expenses_l1580_158090

theorem alyssa_fruit_expenses : 
  let grapes_cost : ℚ := 12.08
  let cherries_cost : ℚ := 9.85
  grapes_cost + cherries_cost = 21.93 := by sorry

end alyssa_fruit_expenses_l1580_158090


namespace volleyball_team_selection_l1580_158037

theorem volleyball_team_selection (n : ℕ) (k : ℕ) : n = 16 ∧ k = 7 → Nat.choose n k = 11440 := by
  sorry

end volleyball_team_selection_l1580_158037


namespace geometric_series_first_term_l1580_158030

theorem geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -3/7) (h2 : S = 20) :
  S = a / (1 - r) → a = 200/7 :=
by
  sorry

end geometric_series_first_term_l1580_158030


namespace minimum_pastries_for_trick_l1580_158053

/-- Represents a pastry with two fillings -/
structure Pastry where
  filling1 : Fin 10
  filling2 : Fin 10
  h : filling1 ≠ filling2

/-- The set of all possible pastries -/
def allPastries : Finset Pastry :=
  sorry

theorem minimum_pastries_for_trick :
  ∀ n : ℕ,
    (n < 36 →
      ∃ (remaining : Finset Pastry),
        remaining ⊆ allPastries ∧
        remaining.card = 45 - n ∧
        ∀ (p : Pastry),
          p ∈ remaining →
            ∃ (q : Pastry),
              q ∈ remaining ∧ q ≠ p ∧
              (p.filling1 = q.filling1 ∨ p.filling1 = q.filling2 ∨
               p.filling2 = q.filling1 ∨ p.filling2 = q.filling2)) ∧
    (n = 36 →
      ∀ (remaining : Finset Pastry),
        remaining ⊆ allPastries →
        remaining.card = 45 - n →
        ∀ (p : Pastry),
          p ∈ remaining →
            ∃ (broken : Finset Pastry),
              broken ⊆ allPastries ∧
              broken.card = n ∧
              (p.filling1 ∈ broken.image Pastry.filling1 ∪ broken.image Pastry.filling2 ∨
               p.filling2 ∈ broken.image Pastry.filling1 ∪ broken.image Pastry.filling2)) :=
by sorry

end minimum_pastries_for_trick_l1580_158053


namespace quadratic_equation_for_complex_roots_l1580_158036

theorem quadratic_equation_for_complex_roots (ω : ℂ) (α β : ℂ) 
  (h1 : ω^8 = 1) 
  (h2 : ω ≠ 1) 
  (h3 : α = ω + ω^3 + ω^5) 
  (h4 : β = ω^2 + ω^4 + ω^6 + ω^7) :
  α^2 + α + 3 = 0 ∧ β^2 + β + 3 = 0 := by
  sorry

end quadratic_equation_for_complex_roots_l1580_158036


namespace min_value_cos_sin_min_value_cos_sin_achieved_l1580_158093

theorem min_value_cos_sin (x : ℝ) : 2 * (Real.cos x)^2 - Real.sin (2 * x) ≥ 1 - Real.sqrt 2 := by
  sorry

theorem min_value_cos_sin_achieved : ∃ x : ℝ, 2 * (Real.cos x)^2 - Real.sin (2 * x) = 1 - Real.sqrt 2 := by
  sorry

end min_value_cos_sin_min_value_cos_sin_achieved_l1580_158093


namespace intersection_of_M_and_N_l1580_158025

def M : Set ℤ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end intersection_of_M_and_N_l1580_158025


namespace fiona_probability_l1580_158059

/-- Represents a lily pad with its number and whether it contains a predator -/
structure LilyPad where
  number : Nat
  hasPredator : Bool

/-- Represents Fiona's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the frog's journey -/
def FrogJourney := List Move

def numPads : Nat := 12

def predatorPads : List Nat := [3, 6]

def foodPad : Nat := 10

def startPad : Nat := 0

def moveProb : Rat := 1/2

/-- Calculates the final position after a sequence of moves -/
def finalPosition (journey : FrogJourney) : Nat :=
  journey.foldl (fun pos move =>
    match move with
    | Move.Hop => min (pos + 1) (numPads - 1)
    | Move.Jump => min (pos + 2) (numPads - 1)
  ) startPad

/-- Checks if a journey is safe (doesn't land on predator pads) -/
def isSafeJourney (journey : FrogJourney) : Bool :=
  let positions := List.scanl (fun pos move =>
    match move with
    | Move.Hop => min (pos + 1) (numPads - 1)
    | Move.Jump => min (pos + 2) (numPads - 1)
  ) startPad journey
  positions.all (fun pos => pos ∉ predatorPads)

/-- Calculates the probability of a specific journey -/
def journeyProbability (journey : FrogJourney) : Rat :=
  (moveProb ^ journey.length)

theorem fiona_probability :
  ∃ (successfulJourneys : List FrogJourney),
    (∀ j ∈ successfulJourneys, finalPosition j = foodPad ∧ isSafeJourney j) ∧
    (successfulJourneys.map journeyProbability).sum = 15/256 := by
  sorry

end fiona_probability_l1580_158059


namespace fraction_problem_l1580_158029

theorem fraction_problem (f : ℝ) : 
  (f * 8.0 = 0.25 * 8.0 + 2) → f = 0.5 := by
  sorry

end fraction_problem_l1580_158029


namespace insulation_project_proof_l1580_158085

/-- The daily completion rate of Team A in square meters -/
def team_a_rate : ℝ := 200

/-- The daily completion rate of Team B in square meters -/
def team_b_rate : ℝ := 1.5 * team_a_rate

/-- The total area to be insulated in square meters -/
def total_area : ℝ := 9000

/-- The difference in completion time between Team A and Team B in days -/
def time_difference : ℝ := 15

theorem insulation_project_proof :
  (total_area / team_a_rate) - (total_area / team_b_rate) = time_difference :=
by sorry

end insulation_project_proof_l1580_158085


namespace car_travel_distance_l1580_158054

theorem car_travel_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time_minutes : ℝ) :
  train_speed = 90 →
  car_speed_ratio = 5/6 →
  travel_time_minutes = 45 →
  let car_speed := car_speed_ratio * train_speed
  let travel_time_hours := travel_time_minutes / 60
  car_speed * travel_time_hours = 56.25 := by
sorry

end car_travel_distance_l1580_158054


namespace largest_prime_factor_of_989_l1580_158015

theorem largest_prime_factor_of_989 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 989 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 989 → q ≤ p :=
by sorry

end largest_prime_factor_of_989_l1580_158015


namespace part_one_part_two_l1580_158082

-- Define the inequalities p and q
def p (x a : ℝ) : Prop := x^2 - 6*a*x + 8*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 4*x + 3 ≤ 0

-- Theorem for part 1
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x : ℝ, p x a → q x) ∧ (∃ x : ℝ, q x ∧ ¬(p x a))

-- Theorem for part 2
theorem part_two :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
sorry

end part_one_part_two_l1580_158082


namespace paths_AC_count_l1580_158009

/-- The number of paths from A to B -/
def paths_AB : Nat := 2

/-- The number of paths from B to C -/
def paths_BC : Nat := 2

/-- The number of direct paths from A to C -/
def direct_paths_AC : Nat := 1

/-- The total number of paths from A to C -/
def total_paths_AC : Nat := paths_AB * paths_BC + direct_paths_AC

theorem paths_AC_count : total_paths_AC = 5 := by
  sorry

end paths_AC_count_l1580_158009


namespace arithmetic_mean_of_40_integers_from_7_l1580_158098

theorem arithmetic_mean_of_40_integers_from_7 :
  let start : ℕ := 7
  let count : ℕ := 40
  let sequence := (fun i => start + i - 1)
  let sum := (sequence 1 + sequence count) * count / 2
  (sum : ℚ) / count = 26.5 := by
  sorry

end arithmetic_mean_of_40_integers_from_7_l1580_158098


namespace square_difference_fourth_power_l1580_158080

theorem square_difference_fourth_power : (6^2 - 3^2)^4 = 531441 := by
  sorry

end square_difference_fourth_power_l1580_158080


namespace arrangement_count_l1580_158075

/-- The number of volunteers --/
def num_volunteers : ℕ := 5

/-- The number of elderly people --/
def num_elderly : ℕ := 2

/-- The total number of units to arrange (volunteers + elderly unit) --/
def total_units : ℕ := num_volunteers + 1

/-- The number of possible positions for the elderly unit --/
def elderly_positions : ℕ := total_units - 2

theorem arrangement_count :
  (elderly_positions * Nat.factorial num_volunteers * Nat.factorial num_elderly) = 960 := by
  sorry

end arrangement_count_l1580_158075


namespace math_competition_score_xiao_hua_correct_answers_l1580_158071

theorem math_competition_score (total_questions : Nat) (correct_points : Int) (wrong_points : Int) (total_score : Int) : Int :=
  let attempted_questions := total_questions
  let hypothetical_score := total_questions * correct_points
  let score_difference := hypothetical_score - total_score
  let points_per_wrong_answer := correct_points + wrong_points
  let wrong_answers := score_difference / points_per_wrong_answer
  total_questions - wrong_answers

theorem xiao_hua_correct_answers : 
  math_competition_score 15 8 (-4) 72 = 11 := by
  sorry

end math_competition_score_xiao_hua_correct_answers_l1580_158071


namespace john_purchase_proof_l1580_158070

def john_purchase (q : ℝ) : Prop :=
  let initial_money : ℝ := 50
  let drink_cost : ℝ := q
  let small_pizza_cost : ℝ := 1.5 * q
  let medium_pizza_cost : ℝ := 2.5 * q
  let total_cost : ℝ := 2 * drink_cost + small_pizza_cost + medium_pizza_cost
  let money_left : ℝ := initial_money - total_cost
  money_left = 50 - 6 * q

theorem john_purchase_proof (q : ℝ) : john_purchase q := by
  sorry

end john_purchase_proof_l1580_158070


namespace pipe_b_fill_time_l1580_158039

/-- Given a tank and three pipes A, B, and C, prove that pipe B fills the tank in 4 hours. -/
theorem pipe_b_fill_time (fill_time_A fill_time_B empty_time_C all_pipes_time : ℝ) 
  (h1 : fill_time_A = 3)
  (h2 : empty_time_C = 4)
  (h3 : all_pipes_time = 3.000000000000001)
  (h4 : 1 / fill_time_A + 1 / fill_time_B - 1 / empty_time_C = 1 / all_pipes_time) :
  fill_time_B = 4 := by
sorry

end pipe_b_fill_time_l1580_158039


namespace cycling_speed_rectangular_park_l1580_158017

/-- Calculates the cycling speed around a rectangular park -/
theorem cycling_speed_rectangular_park 
  (L B : ℝ) 
  (h1 : B = 3 * L) 
  (h2 : L * B = 120000) 
  (h3 : (2 * L + 2 * B) / 8 = 200) : 
  (200 : ℝ) * 60 / 1000 = 10 / 3 := by
  sorry

end cycling_speed_rectangular_park_l1580_158017


namespace max_value_of_trigonometric_expression_l1580_158035

theorem max_value_of_trigonometric_expression :
  let y : ℝ → ℝ := λ x => Real.tan (x + 5 * Real.pi / 6) - Real.tan (x + Real.pi / 3) + Real.sin (x + Real.pi / 3)
  let max_value := (4 + Real.sqrt 3) / (2 * Real.sqrt 3)
  ∀ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6), y x ≤ max_value ∧
  ∃ x₀ ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6), y x₀ = max_value := by
sorry

end max_value_of_trigonometric_expression_l1580_158035


namespace fortieth_term_is_210_l1580_158077

/-- A function that checks if a number contains the digit 2 --/
def containsTwo (n : ℕ) : Bool :=
  sorry

/-- A function that generates the sequence of positive multiples of 3 containing at least one digit 2 --/
def sequenceGenerator : ℕ → ℕ :=
  sorry

/-- The theorem stating that the 40th term of the sequence is 210 --/
theorem fortieth_term_is_210 : sequenceGenerator 40 = 210 := by
  sorry

end fortieth_term_is_210_l1580_158077


namespace runner_b_lap_time_l1580_158006

/-- A runner on a circular track -/
structure Runner where
  lap_time : ℝ
  speed : ℝ

/-- The circular track -/
structure Track where
  circumference : ℝ

/-- The scenario of two runners on a circular track -/
structure RunningScenario where
  track : Track
  runner_a : Runner
  runner_b : Runner
  meeting_time : ℝ
  b_time_to_start : ℝ

/-- The theorem stating that under given conditions, runner B takes 12 minutes to complete a lap -/
theorem runner_b_lap_time (scenario : RunningScenario) :
  scenario.runner_a.lap_time = 6 ∧
  scenario.b_time_to_start = 8 ∧
  scenario.runner_a.speed = scenario.track.circumference / scenario.runner_a.lap_time ∧
  scenario.runner_b.speed = scenario.track.circumference / scenario.runner_b.lap_time ∧
  scenario.meeting_time = scenario.track.circumference / (scenario.runner_a.speed + scenario.runner_b.speed) ∧
  scenario.runner_b.lap_time = scenario.meeting_time + scenario.b_time_to_start
  →
  scenario.runner_b.lap_time = 12 := by
sorry

end runner_b_lap_time_l1580_158006


namespace square_area_five_equal_rectangles_l1580_158003

/-- A square divided into five rectangles of equal area, where one rectangle has a width of 5, has a total area of 400. -/
theorem square_area_five_equal_rectangles (s : ℝ) (w : ℝ) : 
  s > 0 ∧ w > 0 ∧ w = 5 ∧ 
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  a = b ∧ b = c ∧ c = d ∧ d = e ∧
  s * s = a + b + c + d + e ∧
  w * (s - w) = a →
  s * s = 400 := by
  sorry

end square_area_five_equal_rectangles_l1580_158003


namespace absolute_value_sum_l1580_158063

theorem absolute_value_sum (a : ℝ) (h1 : -2 < a) (h2 : a < 0) :
  |a| + |a + 2| = 2 := by
  sorry

end absolute_value_sum_l1580_158063


namespace manuscript_cost_is_860_l1580_158046

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptTypingCost (totalPages : ℕ) (revisedOnce : ℕ) (revisedTwice : ℕ) 
  (firstTypeCost : ℕ) (revisionCost : ℕ) : ℕ :=
  totalPages * firstTypeCost + revisedOnce * revisionCost + revisedTwice * 2 * revisionCost

/-- Proves that the total cost of typing a 100-page manuscript with given revision parameters is $860. -/
theorem manuscript_cost_is_860 : 
  manuscriptTypingCost 100 35 15 6 4 = 860 := by
  sorry

#eval manuscriptTypingCost 100 35 15 6 4

end manuscript_cost_is_860_l1580_158046


namespace curve_symmetry_l1580_158050

theorem curve_symmetry (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) ↔ x = (p * (-y) + q) / (r * (-y) + s)) →
  p = s :=
sorry

end curve_symmetry_l1580_158050


namespace kevins_age_exists_and_unique_l1580_158040

theorem kevins_age_exists_and_unique :
  ∃! x : ℕ, 
    0 < x ∧ 
    x ≤ 120 ∧ 
    ∃ y : ℕ, x - 2 = y^2 ∧
    ∃ z : ℕ, x + 2 = z^3 := by
  sorry

end kevins_age_exists_and_unique_l1580_158040


namespace quadratic_root_relation_l1580_158076

theorem quadratic_root_relation (a b : ℝ) (h : a ≠ 0) :
  (a * 2019^2 + b * 2019 - 1 = 0) →
  (a * (2020 - 1)^2 + b * (2020 - 1) = 1) := by
  sorry

end quadratic_root_relation_l1580_158076


namespace expansion_term_count_l1580_158097

/-- The number of terms in the expansion of (a+b+c)(a+d+e+f+g) -/
def expansion_terms : ℕ := 15

/-- The first polynomial (a+b+c) has 3 terms -/
def first_poly_terms : ℕ := 3

/-- The second polynomial (a+d+e+f+g) has 5 terms -/
def second_poly_terms : ℕ := 5

/-- Theorem stating that the expansion of (a+b+c)(a+d+e+f+g) has 15 terms -/
theorem expansion_term_count :
  expansion_terms = first_poly_terms * second_poly_terms := by
  sorry

end expansion_term_count_l1580_158097


namespace milkman_A_grazing_period_l1580_158044

/-- Represents the rental arrangement for a pasture shared by four milkmen. -/
structure PastureRental where
  /-- Number of cows grazed by milkman A -/
  cows_A : ℕ
  /-- Number of months milkman A grazed his cows (to be determined) -/
  months_A : ℕ
  /-- Number of cows grazed by milkman B -/
  cows_B : ℕ
  /-- Number of months milkman B grazed his cows -/
  months_B : ℕ
  /-- Number of cows grazed by milkman C -/
  cows_C : ℕ
  /-- Number of months milkman C grazed his cows -/
  months_C : ℕ
  /-- Number of cows grazed by milkman D -/
  cows_D : ℕ
  /-- Number of months milkman D grazed his cows -/
  months_D : ℕ
  /-- A's share of the rent in Rupees -/
  share_A : ℕ
  /-- Total rent of the field in Rupees -/
  total_rent : ℕ

/-- Theorem stating that given the conditions of the pasture rental,
    milkman A grazed his cows for 3 months. -/
theorem milkman_A_grazing_period (r : PastureRental)
  (h1 : r.cows_A = 24)
  (h2 : r.cows_B = 10)
  (h3 : r.months_B = 5)
  (h4 : r.cows_C = 35)
  (h5 : r.months_C = 4)
  (h6 : r.cows_D = 21)
  (h7 : r.months_D = 3)
  (h8 : r.share_A = 1440)
  (h9 : r.total_rent = 6500) :
  r.months_A = 3 := by
  sorry

end milkman_A_grazing_period_l1580_158044


namespace defective_units_shipped_for_sale_l1580_158073

/-- 
Given that 4% of units produced are defective and 0.16% of units produced
are defective units shipped for sale, prove that 4% of defective units
are shipped for sale.
-/
theorem defective_units_shipped_for_sale 
  (total_units : ℝ) 
  (defective_rate : ℝ) 
  (defective_shipped_rate : ℝ) 
  (h1 : defective_rate = 0.04) 
  (h2 : defective_shipped_rate = 0.0016) : 
  defective_shipped_rate / defective_rate = 0.04 := by
  sorry

end defective_units_shipped_for_sale_l1580_158073


namespace farmer_tomatoes_l1580_158049

/-- A farmer picks tomatoes from his garden. -/
theorem farmer_tomatoes (initial : ℕ) (remaining : ℕ) (picked : ℕ)
    (h1 : initial = 97)
    (h2 : remaining = 14)
    (h3 : picked = initial - remaining) :
  picked = 83 := by
  sorry

end farmer_tomatoes_l1580_158049


namespace second_fund_interest_rate_l1580_158019

/-- Proves that the interest rate of the second fund is 8.5% given the problem conditions --/
theorem second_fund_interest_rate : 
  ∀ (total_investment : ℝ) 
    (fund1_rate : ℝ) 
    (annual_interest : ℝ) 
    (fund1_investment : ℝ),
  total_investment = 50000 →
  fund1_rate = 8 →
  annual_interest = 4120 →
  fund1_investment = 26000 →
  ∃ (fund2_rate : ℝ),
    fund2_rate = 8.5 ∧
    annual_interest = (fund1_investment * fund1_rate / 100) + 
                      ((total_investment - fund1_investment) * fund2_rate / 100) :=
by
  sorry


end second_fund_interest_rate_l1580_158019


namespace annie_gives_25_crayons_to_mary_l1580_158092

/-- Calculates the number of crayons Annie gives to Mary -/
def crayons_given_to_mary (new_pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := new_pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

/-- Proves that Annie gives 25 crayons to Mary under the given conditions -/
theorem annie_gives_25_crayons_to_mary :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

#eval crayons_given_to_mary 21 36

end annie_gives_25_crayons_to_mary_l1580_158092


namespace arithmetic_sequence_cos_property_l1580_158078

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cos_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end arithmetic_sequence_cos_property_l1580_158078


namespace students_behind_hoseok_l1580_158045

/-- Given a line of students with the following properties:
  * There are 20 students in total
  * 11 students are in front of Yoongi
  * Hoseok is directly behind Yoongi
  Prove that there are 7 students behind Hoseok -/
theorem students_behind_hoseok (total : ℕ) (front_yoongi : ℕ) (hoseok_pos : ℕ) : 
  total = 20 → front_yoongi = 11 → hoseok_pos = front_yoongi + 2 → 
  total - hoseok_pos = 7 := by sorry

end students_behind_hoseok_l1580_158045


namespace sesame_seed_weight_scientific_notation_l1580_158096

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem sesame_seed_weight_scientific_notation :
  toScientificNotation 0.00000201 = ScientificNotation.mk 2.01 (-6) (by sorry) :=
sorry

end sesame_seed_weight_scientific_notation_l1580_158096


namespace inverse_proposition_is_false_l1580_158007

theorem inverse_proposition_is_false : 
  ¬(∀ a : ℝ, |a| = |6| → a = 6) := by
sorry

end inverse_proposition_is_false_l1580_158007


namespace complex_equation_solution_l1580_158084

theorem complex_equation_solution :
  ∀ b : ℝ, (6 - b * I) / (1 + 2 * I) = 2 - 2 * I → b = -2 :=
by
  sorry

end complex_equation_solution_l1580_158084


namespace four_propositions_correct_l1580_158022

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define symmetry about a point
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x) + 2 * b

-- Define symmetry about a line
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Define periodicity
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem four_propositions_correct (f : ℝ → ℝ) :
  (IsOdd f → SymmetricAboutPoint (fun x => f (x - 1)) 1 0) ∧
  (SymmetricAboutLine (fun x => f (x - 1)) 1 → IsEven f) ∧
  ((∀ x, f (x - 1) = -f x) → HasPeriod f 2) ∧
  (SymmetricAboutLine (fun x => f (x - 1)) 1 ∧ SymmetricAboutLine (fun x => f (1 - x)) 1) :=
by sorry

end four_propositions_correct_l1580_158022


namespace division_problem_l1580_158088

theorem division_problem (x : ℚ) : 
  (2976 / x - 240 = 8) → x = 12 := by
  sorry

end division_problem_l1580_158088


namespace salt_mixture_proof_l1580_158081

/-- Proves that mixing 28 ounces of 40% salt solution with 112 ounces of 90% salt solution
    results in a 140-ounce mixture that is 80% salt -/
theorem salt_mixture_proof :
  let solution_a_amount : ℝ := 28
  let solution_b_amount : ℝ := 112
  let solution_a_concentration : ℝ := 0.4
  let solution_b_concentration : ℝ := 0.9
  let total_amount : ℝ := solution_a_amount + solution_b_amount
  let target_concentration : ℝ := 0.8
  let mixture_salt_amount : ℝ := solution_a_amount * solution_a_concentration +
                                  solution_b_amount * solution_b_concentration
  (total_amount = 140) ∧
  (mixture_salt_amount / total_amount = target_concentration) :=
by
  sorry


end salt_mixture_proof_l1580_158081


namespace problem_solution_l1580_158013

theorem problem_solution : let M := 2021 / 3
                           let N := M / 4
                           let Y := M + N
                           Y = 843 := by
  sorry

end problem_solution_l1580_158013


namespace circles_separate_l1580_158027

theorem circles_separate (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂) :
  (∃ x : ℝ, x^2 - 2*R₁*x + R₂^2 - d*(R₂ - R₁) = 0 ∧
   ∀ y : ℝ, y^2 - 2*R₁*y + R₂^2 - d*(R₂ - R₁) = 0 → y = x) →
  R₁ + R₂ = d ∧ d > R₁ + R₂ := by
sorry

end circles_separate_l1580_158027


namespace min_distance_ellipse_line_l1580_158061

/-- The minimum distance between an ellipse and a line -/
theorem min_distance_ellipse_line : 
  ∃ (d : ℝ), d = (15 : ℝ) / Real.sqrt 41 ∧
  ∀ (x y : ℝ), 
    (x^2 / 25 + y^2 / 9 = 1) →
    (∀ (x' y' : ℝ), (4*x' - 5*y' + 40 = 0) → 
      d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) :=
sorry

end min_distance_ellipse_line_l1580_158061


namespace loss_percent_example_l1580_158014

/-- Calculate the loss percent given the cost price and selling price -/
def loss_percent (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percent is 100/3% when an article is bought for 1200 and sold for 800 -/
theorem loss_percent_example : loss_percent 1200 800 = 100 / 3 := by
  sorry

end loss_percent_example_l1580_158014


namespace climb_10_stairs_l1580_158016

/-- The number of ways to climb n stairs -/
def climbWays : ℕ → ℕ
  | 0 => 1  -- base case for 0 stairs
  | 1 => 1  -- given condition
  | 2 => 2  -- given condition
  | (n + 3) => climbWays (n + 2) + climbWays (n + 1)

/-- Theorem stating that there are 89 ways to climb 10 stairs -/
theorem climb_10_stairs : climbWays 10 = 89 := by
  sorry

/-- Lemma: The number of ways to climb n stairs is the sum of ways to climb (n-1) and (n-2) stairs -/
lemma climb_recursive (n : ℕ) (h : n ≥ 3) : climbWays n = climbWays (n - 1) + climbWays (n - 2) := by
  sorry

end climb_10_stairs_l1580_158016


namespace salt_teaspoons_in_recipe_l1580_158089

/-- Represents the recipe and sodium reduction problem -/
theorem salt_teaspoons_in_recipe : 
  ∀ (S : ℝ) 
    (parmesan_oz : ℝ) 
    (salt_sodium_per_tsp : ℝ) 
    (parmesan_sodium_per_oz : ℝ) 
    (parmesan_reduction : ℝ),
  parmesan_oz = 8 →
  salt_sodium_per_tsp = 50 →
  parmesan_sodium_per_oz = 25 →
  parmesan_reduction = 4 →
  (2 / 3) * (salt_sodium_per_tsp * S + parmesan_sodium_per_oz * parmesan_oz) = 
    salt_sodium_per_tsp * S + parmesan_sodium_per_oz * (parmesan_oz - parmesan_reduction) →
  S = 2 := by
  sorry

end salt_teaspoons_in_recipe_l1580_158089


namespace wood_measurement_correct_l1580_158012

/-- Represents the system of equations for the wood measurement problem from "The Mathematical Classic of Sunzi" --/
def wood_measurement_system (x y : ℝ) : Prop :=
  (x - y = 4.5) ∧ (y - (1/2) * x = 1)

/-- Theorem stating that the system of equations correctly represents the wood measurement problem --/
theorem wood_measurement_correct (x y : ℝ) :
  (x > y) ∧                         -- rope is longer than wood
  (x - y = 4.5) ∧                   -- 4.5 feet of rope left when measuring
  (y > (1/2) * x) ∧                 -- wood is longer than half the rope
  (y - (1/2) * x = 1) →             -- rope falls short by 1 foot when folded
  wood_measurement_system x y := by
  sorry


end wood_measurement_correct_l1580_158012


namespace polynomial_factorization_l1580_158083

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l1580_158083


namespace point_inside_circle_m_range_l1580_158000

/-- A point (x, y) is inside a circle with center (a, b) and radius r if the square of the distance
    from the point to the center is less than r^2 -/
def IsInsideCircle (x y a b : ℝ) (r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 < r^2

theorem point_inside_circle_m_range :
  ∀ m : ℝ, IsInsideCircle 1 (-3) 2 (-1) (m^(1/2)) → m > 5 := by
  sorry

end point_inside_circle_m_range_l1580_158000
