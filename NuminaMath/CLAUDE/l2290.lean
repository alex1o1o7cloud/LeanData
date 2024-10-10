import Mathlib

namespace pentagon_hexagon_side_difference_l2290_229030

theorem pentagon_hexagon_side_difference (e : ℕ) : 
  (∃ (p h : ℝ), 5 * p - 6 * h = 1240 ∧ p - h = e ∧ 5 * p > 0 ∧ 6 * h > 0) ↔ e > 248 :=
sorry

end pentagon_hexagon_side_difference_l2290_229030


namespace smaller_pyramid_volume_theorem_l2290_229043

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ

/-- A plane cutting the pyramid parallel to its base -/
structure CuttingPlane where
  height : ℝ

/-- The volume of the smaller pyramid cut off by the plane -/
def smaller_pyramid_volume (p : RightSquarePyramid) (c : CuttingPlane) : ℝ :=
  sorry

/-- Theorem stating the volume of the smaller pyramid -/
theorem smaller_pyramid_volume_theorem (p : RightSquarePyramid) (c : CuttingPlane) :
  p.base_edge = 12 * Real.sqrt 2 →
  p.slant_edge = 15 →
  c.height = 5 →
  smaller_pyramid_volume p c = 24576 / 507 :=
sorry

end smaller_pyramid_volume_theorem_l2290_229043


namespace arithmetic_sequence_sum_property_l2290_229091

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 400) :
  a 2 + a 8 = 160 := by
sorry

end arithmetic_sequence_sum_property_l2290_229091


namespace johns_total_spending_johns_spending_proof_l2290_229098

/-- Calculate John's total spending on a phone and accessories, including sales tax -/
theorem johns_total_spending (online_price : ℝ) (price_increase_rate : ℝ) 
  (accessory_discount_rate : ℝ) (case_price : ℝ) (protector_price : ℝ) 
  (sales_tax_rate : ℝ) : ℝ :=
  let store_phone_price := online_price * (1 + price_increase_rate)
  let accessories_regular_price := case_price + protector_price
  let accessories_discounted_price := accessories_regular_price * (1 - accessory_discount_rate)
  let subtotal := store_phone_price + accessories_discounted_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- Proof that John's total spending is $2212.75 -/
theorem johns_spending_proof : 
  johns_total_spending 2000 0.02 0.05 35 15 0.06 = 2212.75 := by
  sorry

end johns_total_spending_johns_spending_proof_l2290_229098


namespace pauls_remaining_crayons_l2290_229077

/-- The number of crayons Paul had initially -/
def initial_crayons : ℕ := 479

/-- The number of crayons Paul lost or gave away -/
def lost_crayons : ℕ := 345

/-- The number of crayons Paul had left -/
def remaining_crayons : ℕ := initial_crayons - lost_crayons

theorem pauls_remaining_crayons : remaining_crayons = 134 := by
  sorry

end pauls_remaining_crayons_l2290_229077


namespace shopping_expense_l2290_229067

theorem shopping_expense (total_spent shirt_cost : ℕ) (h1 : total_spent = 300) (h2 : shirt_cost = 97) :
  ∃ (shoe_cost : ℕ), 
    shoe_cost > 2 * shirt_cost ∧ 
    shirt_cost + shoe_cost = total_spent ∧ 
    shoe_cost - 2 * shirt_cost = 9 :=
by sorry

end shopping_expense_l2290_229067


namespace dog_catches_fox_dog_catches_fox_specific_l2290_229005

/-- The distance at which a dog catches a fox given initial conditions -/
theorem dog_catches_fox (initial_distance : ℝ) (dog_leap : ℝ) (fox_leap : ℝ) 
  (dog_leaps_per_unit : ℕ) (fox_leaps_per_unit : ℕ) : ℝ :=
  let dog_distance_per_unit := dog_leap * dog_leaps_per_unit
  let fox_distance_per_unit := fox_leap * fox_leaps_per_unit
  let relative_distance_per_unit := dog_distance_per_unit - fox_distance_per_unit
  let time_units_to_catch := initial_distance / relative_distance_per_unit
  time_units_to_catch * dog_distance_per_unit

/-- The specific case of the dog catching the fox problem -/
theorem dog_catches_fox_specific : 
  dog_catches_fox 30 2 1 2 3 = 120 := by
  sorry

end dog_catches_fox_dog_catches_fox_specific_l2290_229005


namespace sprinkles_problem_l2290_229073

theorem sprinkles_problem (initial_cans : ℕ) : 
  (initial_cans / 2 - 3 = 3) → initial_cans = 12 := by
  sorry

end sprinkles_problem_l2290_229073


namespace not_unique_perpendicular_l2290_229057

/-- A line in a plane --/
structure Line where
  -- We don't need to define the internals of a line for this statement
  mk :: 

/-- A plane --/
structure Plane where
  -- We don't need to define the internals of a plane for this statement
  mk ::

/-- Perpendicularity relation between two lines --/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- The statement to be proven false --/
def unique_perpendicular (p : Plane) : Prop :=
  ∃! (l : Line), ∀ (m : Line), perpendicular l m

/-- The theorem stating that the unique perpendicular line statement is false --/
theorem not_unique_perpendicular :
  ∃ (p : Plane), ¬(unique_perpendicular p) :=
sorry

end not_unique_perpendicular_l2290_229057


namespace student_sums_l2290_229074

theorem student_sums (total : ℕ) (right : ℕ) (wrong : ℕ) : 
  total = 48 → 
  wrong = 3 * right → 
  total = right + wrong → 
  wrong = 36 := by sorry

end student_sums_l2290_229074


namespace leapYearsIn123Years_l2290_229018

/-- In a calendrical system where leap years occur every three years, 
    this function calculates the number of leap years in a given period. -/
def leapYearsCount (periodLength : ℕ) : ℕ :=
  periodLength / 3

/-- Theorem stating that in a 123-year period, the number of leap years is 41. -/
theorem leapYearsIn123Years : leapYearsCount 123 = 41 := by
  sorry

end leapYearsIn123Years_l2290_229018


namespace square_sequence_theorem_l2290_229092

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 4 * n^2 + 1

/-- Theorem stating the properties of the sequence and the value for figure 100 -/
theorem square_sequence_theorem :
  (f 0 = 1) ∧
  (f 1 = 5) ∧
  (f 2 = 17) ∧
  (f 3 = 37) ∧
  (f 100 = 40001) := by
  sorry

end square_sequence_theorem_l2290_229092


namespace milk_production_l2290_229015

/-- Milk production calculation -/
theorem milk_production
  (a b c d e f : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
  (h_efficiency : 0 < f ∧ f ≤ 100)
  (h_initial : b = (a * c) * (b / (a * c)))  -- Initial production rate
  : (d * e) * ((b / (a * c)) * (f / 100)) = b * d * e * f / (100 * a * c) :=
by sorry

#check milk_production

end milk_production_l2290_229015


namespace tan_alpha_value_l2290_229065

theorem tan_alpha_value (α : ℝ) 
  (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1/4) : 
  Real.tan α = -2 := by
  sorry

end tan_alpha_value_l2290_229065


namespace complex_number_location_l2290_229045

theorem complex_number_location : 
  let z : ℂ := 1 - (1 / Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end complex_number_location_l2290_229045


namespace zuminglish_word_count_mod_500_l2290_229047

/-- Represents the alphabet of Zuminglish --/
inductive ZuminglishLetter
| M
| O
| P

/-- Represents whether a letter is a vowel or consonant --/
def isVowel (l : ZuminglishLetter) : Bool :=
  match l with
  | ZuminglishLetter.O => true
  | _ => false

/-- A Zuminglish word is a list of ZuminglishLetters --/
def ZuminglishWord := List ZuminglishLetter

/-- Check if a Zuminglish word is valid --/
def isValidWord (w : ZuminglishWord) : Bool :=
  sorry

/-- Count the number of valid 10-letter Zuminglish words --/
def countValidWords : Nat :=
  sorry

/-- The main theorem to prove --/
theorem zuminglish_word_count_mod_500 :
  countValidWords % 500 = 160 :=
sorry

end zuminglish_word_count_mod_500_l2290_229047


namespace square_sum_given_difference_and_product_l2290_229036

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end square_sum_given_difference_and_product_l2290_229036


namespace geometric_progression_ratio_equation_l2290_229075

theorem geometric_progression_ratio_equation 
  (x y z r : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (hgp : ∃ a : ℝ, a ≠ 0 ∧ 
    y * (z + x) = r * (x * (y + z)) ∧ 
    z * (x + y) = r * (y * (z + x))) : 
  r^2 + r + 1 = 0 := by
sorry

end geometric_progression_ratio_equation_l2290_229075


namespace candies_per_friend_l2290_229058

/-- Given 36 candies shared equally among 9 friends, prove that each friend receives 4 candies. -/
theorem candies_per_friend (total_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) :
  total_candies = 36 →
  num_friends = 9 →
  candies_per_friend = total_candies / num_friends →
  candies_per_friend = 4 := by
  sorry

end candies_per_friend_l2290_229058


namespace tangent_circles_ratio_l2290_229029

/-- Two circles touching internally with specific tangent properties -/
structure TangentCircles where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  touch_internally : R > r  -- Circles touch internally
  radii_angle : ℝ  -- Angle between the two radii of the larger circle
  radii_tangent : Bool  -- The two radii are tangent to the smaller circle

/-- Theorem stating the ratio of radii for circles with specific tangent properties -/
theorem tangent_circles_ratio 
  (c : TangentCircles) 
  (h1 : c.radii_angle = 60) 
  (h2 : c.radii_tangent = true) : 
  c.R / c.r = 3 := by sorry

end tangent_circles_ratio_l2290_229029


namespace xyz_value_l2290_229063

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 5 := by sorry

end xyz_value_l2290_229063


namespace nested_average_equals_25_18_l2290_229012

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem to prove -/
theorem nested_average_equals_25_18 :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 := by sorry

end nested_average_equals_25_18_l2290_229012


namespace smallest_y_for_cube_l2290_229096

theorem smallest_y_for_cube (y : ℕ+) (M : ℤ) : 
  (∀ k : ℕ+, k < y → ¬∃ N : ℤ, 2520 * k = N^3) → 
  (∃ N : ℤ, 2520 * y = N^3) → 
  y = 3675 := by
sorry

end smallest_y_for_cube_l2290_229096


namespace product_of_difference_of_squares_l2290_229061

theorem product_of_difference_of_squares (a b x1 y1 x2 y2 : ℤ) 
  (ha : a = x1^2 - 5*y1^2) (hb : b = x2^2 - 5*y2^2) :
  ∃ u v : ℤ, a * b = u^2 - 5*v^2 := by
sorry

end product_of_difference_of_squares_l2290_229061


namespace symmetric_points_difference_l2290_229039

/-- Two points are symmetric about the y-axis if their y-coordinates are equal and their x-coordinates are opposite -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = y2 ∧ x1 = -x2

/-- The problem statement -/
theorem symmetric_points_difference (m n : ℝ) :
  symmetric_about_y_axis 3 m n 4 → m - n = 7 := by
  sorry

end symmetric_points_difference_l2290_229039


namespace opposite_sides_difference_equal_l2290_229056

/-- An equiangular hexagon with sides a, b, c, d, e, f in order -/
structure EquiangularHexagon where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  equiangular : True  -- This represents the equiangular property

/-- The differences between opposite sides in an equiangular hexagon are equal -/
theorem opposite_sides_difference_equal (h : EquiangularHexagon) :
  h.a - h.d = h.e - h.b ∧ h.e - h.b = h.c - h.f :=
by sorry

end opposite_sides_difference_equal_l2290_229056


namespace no_integer_root_2016_l2290_229007

theorem no_integer_root_2016 (a b c d : ℤ) (p : ℤ → ℤ) :
  (∀ x : ℤ, p x = a * x^3 + b * x^2 + c * x + d) →
  p 1 = 2015 →
  p 2 = 2017 →
  ∀ x : ℤ, p x ≠ 2016 := by
sorry

end no_integer_root_2016_l2290_229007


namespace greatest_n_for_inequality_l2290_229080

theorem greatest_n_for_inequality (n : ℤ) (h : 101 * n^2 ≤ 3600) : n ≤ 5 ∧ ∃ (m : ℤ), m = 5 ∧ 101 * m^2 ≤ 3600 :=
sorry

end greatest_n_for_inequality_l2290_229080


namespace covid_cases_after_growth_l2290_229055

/-- Calculates the total number of COVID-19 cases in New York, California, and Texas after one month of growth --/
theorem covid_cases_after_growth (new_york_initial : ℕ) 
  (h1 : new_york_initial = 2000)
  (h2 : ∃ california_initial : ℕ, california_initial = new_york_initial / 2)
  (h3 : ∃ texas_initial : ℕ, ∃ california_initial : ℕ, 
    california_initial = new_york_initial / 2 ∧ 
    california_initial = texas_initial + 400)
  (h4 : ∃ new_york_growth : ℚ, new_york_growth = 25 / 100)
  (h5 : ∃ california_growth : ℚ, california_growth = 15 / 100)
  (h6 : ∃ texas_growth : ℚ, texas_growth = 30 / 100) :
  ∃ total_cases : ℕ, total_cases = 4430 := by
sorry

end covid_cases_after_growth_l2290_229055


namespace andrews_cat_catch_l2290_229040

theorem andrews_cat_catch (martha_cat cara_cat T : ℕ) : 
  martha_cat = 10 →
  cara_cat = 47 →
  T = martha_cat + cara_cat →
  T^2 + 2 = 3251 :=
by
  sorry

end andrews_cat_catch_l2290_229040


namespace fruit_arrangement_l2290_229076

theorem fruit_arrangement (n a o b p : ℕ) 
  (total : n = a + o + b + p)
  (apple : a = 4)
  (orange : o = 2)
  (banana : b = 2)
  (pear : p = 1) :
  Nat.factorial n / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial p) = 3780 := by
  sorry

end fruit_arrangement_l2290_229076


namespace range_of_a_l2290_229022

-- Define the propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x a : ℝ) : Prop := abs x > a

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a))

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x, p x → q x a) ∧ necessary_not_sufficient a ↔ a ≤ 1 :=
sorry

end range_of_a_l2290_229022


namespace path_width_calculation_l2290_229059

theorem path_width_calculation (field_length field_width path_area : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 15)
  (h3 : path_area = 246)
  (h4 : field_length > 0)
  (h5 : field_width > 0)
  (h6 : path_area > 0) :
  ∃ (path_width : ℝ),
    path_width > 0 ∧
    (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width = path_area ∧
    path_width = 3 := by
  sorry

end path_width_calculation_l2290_229059


namespace cafe_customers_l2290_229027

/-- The number of sandwiches ordered by offices -/
def office_sandwiches : ℕ := 30

/-- The number of sandwiches each ordering customer in the group ordered -/
def sandwiches_per_customer : ℕ := 4

/-- The total number of sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- The fraction of the group that ordered sandwiches -/
def ordering_fraction : ℚ := 1/2

theorem cafe_customers : ℕ :=
  let group_sandwiches := total_sandwiches - office_sandwiches
  let ordering_customers := group_sandwiches / sandwiches_per_customer
  let total_customers := ordering_customers / ordering_fraction
  12

#check cafe_customers

end cafe_customers_l2290_229027


namespace moving_circle_center_trajectory_l2290_229051

/-- A moving circle that passes through (1, 0) and is tangent to x = -1 -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_one_zero : (center.1 - 1)^2 + center.2^2 = (center.1 + 1)^2
  tangent_to_neg_one : True  -- This condition is implied by the equation above

/-- The trajectory of the center of the moving circle is y² = 4x -/
theorem moving_circle_center_trajectory (M : MovingCircle) : 
  M.center.2^2 = 4 * M.center.1 := by
  sorry

end moving_circle_center_trajectory_l2290_229051


namespace imaginary_part_of_i_over_i_plus_one_l2290_229088

theorem imaginary_part_of_i_over_i_plus_one :
  Complex.im (Complex.I / (Complex.I + 1)) = 1 / 2 := by sorry

end imaginary_part_of_i_over_i_plus_one_l2290_229088


namespace bus_cyclist_speed_problem_l2290_229032

/-- Proves that given the problem conditions, the speeds of the bus and cyclist are 35 km/h and 15 km/h respectively. -/
theorem bus_cyclist_speed_problem (distance : ℝ) (first_meeting_time : ℝ) (bus_stop_time : ℝ) (overtake_time : ℝ)
  (h1 : distance = 70)
  (h2 : first_meeting_time = 7/5)
  (h3 : bus_stop_time = 1/3)
  (h4 : overtake_time = 161/60) :
  ∃ (bus_speed cyclist_speed : ℝ),
    bus_speed = 35 ∧
    cyclist_speed = 15 ∧
    first_meeting_time * (bus_speed + cyclist_speed) = distance ∧
    (first_meeting_time + overtake_time - bus_stop_time) * bus_speed - (first_meeting_time + overtake_time) * cyclist_speed = distance :=
by sorry

end bus_cyclist_speed_problem_l2290_229032


namespace last_segment_speed_l2290_229033

/-- Represents the average speed during a journey segment -/
structure JourneySegment where
  duration : ℚ  -- Duration in hours
  speed : ℚ     -- Average speed in mph
  distance : ℚ  -- Distance traveled in miles

/-- Represents a complete journey -/
structure Journey where
  totalDistance : ℚ
  totalTime : ℚ
  segments : List JourneySegment

/-- Calculates the average speed for a given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

theorem last_segment_speed (j : Journey) 
  (h1 : j.totalDistance = 120)
  (h2 : j.totalTime = 2)
  (h3 : j.segments.length = 3)
  (h4 : j.segments[0].duration = 2/3)
  (h5 : j.segments[0].speed = 50)
  (h6 : j.segments[1].duration = 5/6)
  (h7 : j.segments[1].speed = 60)
  (h8 : j.segments[2].duration = 1/2) :
  averageSpeed j.segments[2].distance j.segments[2].duration = 220/3 := by
  sorry

#eval (220 : ℚ) / 3  -- To verify the result is approximately 73.33

end last_segment_speed_l2290_229033


namespace ellipse_major_axis_length_l2290_229001

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Checks if an ellipse is tangent to both x and y axes -/
def isTangentToAxes (e : Ellipse) : Prop := sorry

/-- Calculates the length of the major axis of an ellipse -/
def majorAxisLength (e : Ellipse) : ℝ := sorry

/-- Main theorem: The length of the major axis of the given ellipse is 10 -/
theorem ellipse_major_axis_length :
  ∀ (e : Ellipse),
    e.focus1 = ⟨3, -5 + 2 * Real.sqrt 2⟩ ∧
    e.focus2 = ⟨3, -5 - 2 * Real.sqrt 2⟩ ∧
    isTangentToAxes e →
    majorAxisLength e = 10 := by
  sorry

end ellipse_major_axis_length_l2290_229001


namespace repeating_decimal_sum_l2290_229081

theorem repeating_decimal_sum : 
  (4 : ℚ) / 33 + 34 / 999 + 567 / 99999 = 134255 / 32929667 := by
  sorry

end repeating_decimal_sum_l2290_229081


namespace no_intersection_implies_k_plus_minus_one_l2290_229048

theorem no_intersection_implies_k_plus_minus_one (k : ℤ) :
  (∀ x y : ℝ, x^2 + y^2 = k^2 → x * y ≠ k) →
  k = 1 ∨ k = -1 := by
sorry

end no_intersection_implies_k_plus_minus_one_l2290_229048


namespace quadratic_roots_properties_l2290_229008

theorem quadratic_roots_properties (m : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*(m+1)*x + m^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  (m ≥ 1 ∧ ∃ m', m' ≥ 1 ∧ (x₁ - 1)*(x₂ - 1) = m' + 6 ∧ m' = 4) := by
  sorry

end quadratic_roots_properties_l2290_229008


namespace p_sufficient_not_necessary_for_q_l2290_229050

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := 0 < x ∧ x < 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry

end p_sufficient_not_necessary_for_q_l2290_229050


namespace evaluate_power_l2290_229090

theorem evaluate_power (x : ℝ) (h : x = 81) : x^(5/4) = 243 := by
  sorry

end evaluate_power_l2290_229090


namespace lauren_reaches_andrea_l2290_229021

/-- The initial distance between Andrea and Lauren in kilometers -/
def initial_distance : ℝ := 30

/-- The rate at which the distance between Andrea and Lauren decreases in km/min -/
def distance_decrease_rate : ℝ := 2

/-- The duration of initial biking in minutes -/
def initial_biking_time : ℝ := 10

/-- The duration of the stop in minutes -/
def stop_time : ℝ := 5

/-- Andrea's speed in km/h -/
def andrea_speed : ℝ := 40

/-- Lauren's speed in km/h -/
def lauren_speed : ℝ := 80

/-- The total time it takes for Lauren to reach Andrea -/
def total_time : ℝ := 22.5

theorem lauren_reaches_andrea :
  let distance_covered := distance_decrease_rate * initial_biking_time
  let remaining_distance := initial_distance - distance_covered
  let lauren_final_time := remaining_distance / (lauren_speed / 60)
  total_time = initial_biking_time + stop_time + lauren_final_time :=
by
  sorry

#check lauren_reaches_andrea

end lauren_reaches_andrea_l2290_229021


namespace product_equality_l2290_229072

noncomputable def P : ℝ := Real.sqrt 1011 + Real.sqrt 1012
noncomputable def Q : ℝ := -Real.sqrt 1011 - Real.sqrt 1012
noncomputable def R : ℝ := Real.sqrt 1011 - Real.sqrt 1012
noncomputable def S : ℝ := Real.sqrt 1012 - Real.sqrt 1011

theorem product_equality : (P * Q)^2 * R * S = 8136957 := by
  sorry

end product_equality_l2290_229072


namespace perfect_square_factors_of_7200_eq_12_l2290_229069

/-- The number of factors of 7200 that are perfect squares -/
def perfect_square_factors_of_7200 : ℕ :=
  let n := 7200
  let factorization := [(2, 4), (3, 2), (5, 2)]
  (List.map (fun (p : ℕ × ℕ) => (p.2 / 2 + 1)) factorization).prod

/-- Theorem stating that the number of factors of 7200 that are perfect squares is 12 -/
theorem perfect_square_factors_of_7200_eq_12 :
  perfect_square_factors_of_7200 = 12 := by sorry

end perfect_square_factors_of_7200_eq_12_l2290_229069


namespace equation_represents_three_non_concurrent_lines_l2290_229034

/-- The equation represents three lines that do not all pass through a common point -/
theorem equation_represents_three_non_concurrent_lines :
  ∃ (l₁ l₂ l₃ : ℝ → ℝ → Prop),
    (∀ x y, (x^2 - 3*y)*(x - y + 1) = (y^2 - 3*x)*(x - y + 1) ↔ l₁ x y ∨ l₂ x y ∨ l₃ x y) ∧
    (∃ x₁ y₁, l₁ x₁ y₁ ∧ l₂ x₁ y₁ ∧ ¬l₃ x₁ y₁) ∧
    (∃ x₂ y₂, l₁ x₂ y₂ ∧ ¬l₂ x₂ y₂ ∧ l₃ x₂ y₂) ∧
    (∃ x₃ y₃, ¬l₁ x₃ y₃ ∧ l₂ x₃ y₃ ∧ l₃ x₃ y₃) ∧
    (∀ x y, ¬(l₁ x y ∧ l₂ x y ∧ l₃ x y)) :=
by
  sorry


end equation_represents_three_non_concurrent_lines_l2290_229034


namespace piano_lessons_cost_l2290_229089

theorem piano_lessons_cost (piano_cost : ℝ) (num_lessons : ℕ) (lesson_cost : ℝ) (discount_rate : ℝ) :
  piano_cost = 500 →
  num_lessons = 20 →
  lesson_cost = 40 →
  discount_rate = 0.25 →
  piano_cost + (num_lessons : ℝ) * lesson_cost * (1 - discount_rate) = 1100 := by
  sorry

end piano_lessons_cost_l2290_229089


namespace both_hit_probability_l2290_229054

def prob_both_hit (prob_A prob_B : ℝ) : ℝ := prob_A * prob_B

theorem both_hit_probability :
  let prob_A : ℝ := 0.8
  let prob_B : ℝ := 0.7
  prob_both_hit prob_A prob_B = 0.56 := by
  sorry

end both_hit_probability_l2290_229054


namespace sum_of_variables_l2290_229046

theorem sum_of_variables (a b c d e : ℝ) 
  (eq1 : 3*a + 2*b + 4*d = 10)
  (eq2 : 6*a + 5*b + 4*c + 3*d + 2*e = 8)
  (eq3 : a + b + 2*c + 5*e = 3)
  (eq4 : 2*c + 3*d + 3*e = 4)
  (eq5 : a + 2*b + 3*c + d = 7) :
  a + b + c + d + e = 4 := by
  sorry

end sum_of_variables_l2290_229046


namespace arithmetic_sequence_common_difference_l2290_229085

/-- An arithmetic sequence {a_n} with specified properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 7 - 2 * a 4 = -1)
  (h_third : a 3 = 0) :
  ∃ d : ℚ, (∀ n, a n = a 1 + (n - 1) * d) ∧ d = -1/2 := by
sorry

end arithmetic_sequence_common_difference_l2290_229085


namespace matching_socks_probability_l2290_229014

/-- The number of blue-bottomed socks -/
def blue_socks : ℕ := 12

/-- The number of red-bottomed socks -/
def red_socks : ℕ := 10

/-- The number of green-bottomed socks -/
def green_socks : ℕ := 6

/-- The total number of socks -/
def total_socks : ℕ := blue_socks + red_socks + green_socks

/-- The number of ways to choose 2 socks from n socks -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of picking a matching pair of socks -/
theorem matching_socks_probability : 
  (choose_two blue_socks + choose_two red_socks + choose_two green_socks) / choose_two total_socks = 1 / 3 := by
  sorry

end matching_socks_probability_l2290_229014


namespace exponential_function_fixed_point_l2290_229041

theorem exponential_function_fixed_point (a b : ℝ) (ha : a > 0) :
  (∀ x, (a^(x - b) + 1 = 2) ↔ (x = 1)) → b = 1 := by
  sorry

end exponential_function_fixed_point_l2290_229041


namespace parking_ticket_ratio_l2290_229006

/-- Represents the number of tickets for each person -/
structure Tickets where
  parking : ℕ
  speeding : ℕ

/-- The problem setup -/
def ticketProblem (mark sarah : Tickets) : Prop :=
  mark.speeding = sarah.speeding ∧
  sarah.speeding = 6 ∧
  mark.parking = 8 ∧
  mark.parking + mark.speeding + sarah.parking + sarah.speeding = 24

/-- The theorem to prove -/
theorem parking_ticket_ratio (mark sarah : Tickets) 
  (h : ticketProblem mark sarah) : 
  mark.parking * 1 = sarah.parking * 2 := by
  sorry


end parking_ticket_ratio_l2290_229006


namespace lab_items_per_tech_l2290_229093

/-- Given the number of uniforms in a lab, calculate the total number of coats and uniforms per lab tech. -/
def total_per_lab_tech (num_uniforms : ℕ) : ℕ :=
  let num_coats := 6 * num_uniforms
  let total_items := num_coats + num_uniforms
  let num_lab_techs := num_uniforms / 2
  total_items / num_lab_techs

/-- Theorem stating that given 12 uniforms, each lab tech gets 14 coats and uniforms in total. -/
theorem lab_items_per_tech :
  total_per_lab_tech 12 = 14 := by
  sorry

#eval total_per_lab_tech 12

end lab_items_per_tech_l2290_229093


namespace rectangleA_max_sum_l2290_229064

-- Define a structure for rectangles
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def rectangleA : Rectangle := ⟨8, 2, 9, 5⟩
def rectangleB : Rectangle := ⟨2, 1, 5, 8⟩
def rectangleC : Rectangle := ⟨6, 9, 4, 3⟩
def rectangleD : Rectangle := ⟨4, 6, 2, 9⟩
def rectangleE : Rectangle := ⟨9, 5, 6, 1⟩

-- Define a list of all rectangles
def rectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Define a function to calculate the sum of w and y
def sumWY (r : Rectangle) : Int := r.w + r.y

-- Theorem: Rectangle A has the maximum sum of w and y
theorem rectangleA_max_sum :
  ∀ r ∈ rectangles, sumWY rectangleA ≥ sumWY r := by
  sorry

end rectangleA_max_sum_l2290_229064


namespace hockey_league_season_games_l2290_229062

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 16 teams, where each team faces all other teams 10 times,
    the total number of games played in the season is 1200. -/
theorem hockey_league_season_games :
  hockey_league_games 16 10 = 1200 := by
  sorry

end hockey_league_season_games_l2290_229062


namespace min_transport_time_l2290_229038

/-- The minimum time required for transporting goods between two cities --/
theorem min_transport_time (distance : ℝ) (num_trains : ℕ) (speed : ℝ) 
  (h1 : distance = 400)
  (h2 : num_trains = 17)
  (h3 : speed > 0) :
  (distance / speed + (num_trains - 1) * (speed / 20)^2 / speed) ≥ 8 := by
  sorry

end min_transport_time_l2290_229038


namespace right_triangle_and_multiplicative_inverse_l2290_229009

theorem right_triangle_and_multiplicative_inverse :
  (30^2 + 272^2 = 278^2) ∧
  ((550 * 6) % 4079 = 1) ∧
  (0 ≤ 6 ∧ 6 < 4079) := by
  sorry

end right_triangle_and_multiplicative_inverse_l2290_229009


namespace negation_of_universal_proposition_l2290_229060

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 < 0) :=
by sorry

end negation_of_universal_proposition_l2290_229060


namespace boat_speed_in_still_water_l2290_229079

/-- Given a boat that travels 6 km/hr along a stream and 2 km/hr against the same stream,
    its speed in still water is 4 km/hr. -/
theorem boat_speed_in_still_water (b s : ℝ) 
    (h1 : b + s = 6)  -- Speed along the stream
    (h2 : b - s = 2)  -- Speed against the stream
    : b = 4 := by
  sorry

end boat_speed_in_still_water_l2290_229079


namespace dime_probability_l2290_229000

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℚ
  | _ => 1250

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Penny

/-- The probability of selecting a dime from the jar -/
def probDime : ℚ := coinCount Coin.Dime / totalCoins

theorem dime_probability : probDime = 5 / 57 := by
  sorry

end dime_probability_l2290_229000


namespace cold_virus_diameter_scientific_notation_l2290_229011

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem cold_virus_diameter_scientific_notation :
  to_scientific_notation 0.00000036 = ScientificNotation.mk 3.6 (-7) sorry := by
  sorry

end cold_virus_diameter_scientific_notation_l2290_229011


namespace last_digit_of_special_number_l2290_229044

/-- A function that returns the last element of a list -/
def lastDigit (digits : List Nat) : Nat :=
  match digits.reverse with
  | [] => 0  -- Default value for empty list
  | d :: _ => d

/-- Check if a two-digit number is divisible by 13 -/
def isDivisibleBy13 (n : Nat) : Prop :=
  n % 13 = 0

theorem last_digit_of_special_number :
  ∀ (digits : List Nat),
    digits.length = 2019 →
    digits.head? = some 6 →
    (∀ i, i < digits.length - 1 →
      isDivisibleBy13 (digits[i]! * 10 + digits[i+1]!)) →
    lastDigit digits = 2 := by
  sorry

#check last_digit_of_special_number

end last_digit_of_special_number_l2290_229044


namespace smallest_tax_price_integer_l2290_229084

theorem smallest_tax_price_integer (n : ℕ) : n = 21 ↔ 
  (n > 0 ∧ ∀ m : ℕ, m > 0 → m < n → 
    ¬∃ x : ℕ, (105 * x : ℚ) / 100 = m) ∧
  ∃ x : ℕ, (105 * x : ℚ) / 100 = n :=
by sorry

end smallest_tax_price_integer_l2290_229084


namespace square_area_comparison_l2290_229035

theorem square_area_comparison (a b : ℝ) (h : b = 4 * a) :
  b ^ 2 = 16 * a ^ 2 := by
  sorry

end square_area_comparison_l2290_229035


namespace maltese_cross_to_square_l2290_229052

/-- Represents a piece of the Maltese cross -/
structure Piece where
  area : ℝ

/-- Represents the Maltese cross -/
structure MalteseCross where
  pieces : Finset Piece
  total_area : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- A function that checks if a set of pieces can form a square -/
def can_form_square (pieces : Finset Piece) : Prop :=
  ∃ (s : Square), s.side_length^2 = (pieces.sum (λ p => p.area))

theorem maltese_cross_to_square (cross : MalteseCross) : 
  cross.total_area = 17 → 
  (∃ (cut_pieces : Finset Piece), 
    cut_pieces.card = 7 ∧ 
    (cut_pieces.sum (λ p => p.area) = cross.total_area) ∧
    can_form_square cut_pieces) := by
  sorry

end maltese_cross_to_square_l2290_229052


namespace subset_implies_m_leq_two_l2290_229003

def A : Set ℝ := {x | x < 2}
def B (m : ℝ) : Set ℝ := {x | x < m}

theorem subset_implies_m_leq_two (m : ℝ) : B m ⊆ A → m ≤ 2 := by
  sorry

end subset_implies_m_leq_two_l2290_229003


namespace sufficient_not_necessary_l2290_229053

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ 
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) := by
  sorry

end sufficient_not_necessary_l2290_229053


namespace arithmetic_and_geometric_sequence_l2290_229070

theorem arithmetic_and_geometric_sequence (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence
  (∃ q : ℝ, q = 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n) := by
sorry


end arithmetic_and_geometric_sequence_l2290_229070


namespace exists_min_n_all_rows_shaded_l2290_229042

/-- Calculates the square number of the nth shaded square -/
def shadedSquareNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Calculates the row number for a given square number -/
def squareToRow (square : ℕ) : ℕ :=
  (square - 1) / 5 + 1

/-- Checks if all rows are shaded up to the nth shaded square -/
def allRowsShaded (n : ℕ) : Prop :=
  ∀ row : ℕ, row ≤ 10 → ∃ k : ℕ, k ≤ n ∧ squareToRow (shadedSquareNumber k) = row

/-- The main theorem stating the existence of a minimum n that shades all rows -/
theorem exists_min_n_all_rows_shaded :
  ∃ n : ℕ, allRowsShaded n ∧ ∀ m : ℕ, m < n → ¬allRowsShaded m :=
sorry

end exists_min_n_all_rows_shaded_l2290_229042


namespace abc_product_l2290_229002

theorem abc_product (a b c : ℝ) 
  (eq1 : a + b = 23)
  (eq2 : b + c = 25)
  (eq3 : c + a = 30) :
  a * b * c = 2016 := by
  sorry

end abc_product_l2290_229002


namespace marble_fraction_after_tripling_l2290_229019

theorem marble_fraction_after_tripling (total : ℚ) (h : total > 0) :
  let initial_blue : ℚ := (4 / 7) * total
  let initial_red : ℚ := total - initial_blue
  let new_red : ℚ := 3 * initial_red
  let new_total : ℚ := initial_blue + new_red
  new_red / new_total = 9 / 13 :=
by sorry

end marble_fraction_after_tripling_l2290_229019


namespace invoice_error_correction_l2290_229017

/-- Two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The proposition to be proved -/
theorem invoice_error_correction (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : TwoDigitNumber y)
  (h_diff : 100 * x + y - (100 * y + x) = 3654) :
  x = 63 ∧ y = 26 := by
  sorry

end invoice_error_correction_l2290_229017


namespace intersection_A_B_complement_union_A_B_l2290_229028

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by sorry

-- Theorem for the complement of the union of A and B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x > 3} := by sorry

end intersection_A_B_complement_union_A_B_l2290_229028


namespace repeating_decimal_fraction_l2290_229094

def repeating_decimal : ℚ := 7 + 17 / 99

theorem repeating_decimal_fraction :
  repeating_decimal = 710 / 99 ∧
  (Nat.gcd 710 99 = 1) ∧
  (710 + 99 = 809) := by
  sorry

#eval repeating_decimal

end repeating_decimal_fraction_l2290_229094


namespace greatest_x_satisfying_equation_l2290_229024

theorem greatest_x_satisfying_equation : 
  ∃ (x : ℝ), x = -3 ∧ 
  (∀ y : ℝ, y ≠ 6 → y ≠ -4 → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧
  x ≠ 6 ∧ x ≠ -4 := by
sorry

end greatest_x_satisfying_equation_l2290_229024


namespace division_theorem_l2290_229083

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 132)
  (h2 : divisor = 16)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 8 := by
  sorry

end division_theorem_l2290_229083


namespace specific_prism_volume_max_prism_volume_max_volume_achievable_l2290_229026

/-- Regular quadrangular pyramid with inscribed regular triangular prism -/
structure PyramidWithPrism where
  /-- Volume of the pyramid -/
  V : ℝ
  /-- Angle between lateral edge and base plane (in radians) -/
  angle : ℝ
  /-- Ratio of the division of the pyramid's height by the prism's face -/
  ratio : ℝ × ℝ
  /-- Volume of the inscribed prism -/
  prismVolume : ℝ
  /-- Constraint: angle is 30 degrees (π/6 radians) -/
  angle_is_30_deg : angle = Real.pi / 6
  /-- Constraint: ratio is valid (both parts positive, sum > 0) -/
  ratio_valid : ratio.1 > 0 ∧ ratio.2 > 0 ∧ ratio.1 + ratio.2 > 0
  /-- Constraint: prism volume is positive and less than pyramid volume -/
  volume_valid : 0 < prismVolume ∧ prismVolume < V

/-- Theorem for the volume of the specific prism -/
theorem specific_prism_volume (p : PyramidWithPrism) (h : p.ratio = (2, 3)) :
  p.prismVolume = 9/250 * p.V := by sorry

/-- Theorem for the maximum volume of any such prism -/
theorem max_prism_volume (p : PyramidWithPrism) :
  p.prismVolume ≤ 1/12 * p.V := by sorry

/-- Theorem that 1/12 is achievable -/
theorem max_volume_achievable (V : ℝ) (h : V > 0) :
  ∃ p : PyramidWithPrism, p.V = V ∧ p.prismVolume = 1/12 * V := by sorry

end specific_prism_volume_max_prism_volume_max_volume_achievable_l2290_229026


namespace quadratic_roots_reciprocal_l2290_229037

theorem quadratic_roots_reciprocal (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - b*x₁ + 1 = 0 ∧ x₂^2 - b*x₂ + 1 = 0 →
  (x₂ = 1 / x₁ ∨ (b = 2 ∧ x₁ = 1 ∧ x₂ = 1) ∨ (b = -2 ∧ x₁ = -1 ∧ x₂ = -1)) :=
by sorry

end quadratic_roots_reciprocal_l2290_229037


namespace shaded_fraction_is_one_twelfth_l2290_229031

/-- A point in a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A rectangle defined by its top-left and bottom-right corners -/
structure Rectangle where
  topLeft : GridPoint
  bottomRight : GridPoint

/-- The 6x6 grid -/
def gridSize : ℕ := 6

/-- The rectangle in question -/
def shadedRectangle : Rectangle := {
  topLeft := { x := 2, y := 5 }
  bottomRight := { x := 3, y := 2 }
}

/-- Calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  (r.bottomRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomRight.y)

/-- Calculate the area of the entire grid -/
def gridArea : ℕ := gridSize * gridSize

/-- The fraction of the grid occupied by the shaded rectangle -/
def shadedFraction : ℚ :=
  (rectangleArea shadedRectangle : ℚ) / gridArea

/-- Theorem: The shaded fraction is equal to 1/12 -/
theorem shaded_fraction_is_one_twelfth : shadedFraction = 1 / 12 := by
  sorry

end shaded_fraction_is_one_twelfth_l2290_229031


namespace even_function_sum_l2290_229023

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_sum (f : ℝ → ℝ) (h_even : is_even_function f) (h_f4 : f 4 = 5) :
  f 4 + f (-4) = 10 := by
  sorry

end even_function_sum_l2290_229023


namespace sum_inequality_l2290_229016

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) : a + b + c ≤ 3 := by
  sorry

end sum_inequality_l2290_229016


namespace range_of_2a_minus_b_l2290_229068

theorem range_of_2a_minus_b (a b : ℝ) (ha : -1 ≤ a ∧ a ≤ 3) (hb : 2 ≤ b ∧ b ≤ 4) :
  (∀ x, 2 * a - b ≤ x → x ≤ 4) ∧ (∀ y, -6 ≤ y → y ≤ 2 * a - b) :=
by sorry

end range_of_2a_minus_b_l2290_229068


namespace power_function_through_point_l2290_229049

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x ^ b)

-- Define the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 4 = 2 := by
  sorry

end power_function_through_point_l2290_229049


namespace lines_coplanar_iff_k_eq_neg_half_l2290_229087

/-- First line parameterization --/
def line1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (2 + s, 4 - k*s, -1 + k*s)

/-- Second line parameterization --/
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (2*t, 2 + t, 3 - t)

/-- Direction vector of the first line --/
def dir1 (k : ℝ) : ℝ × ℝ × ℝ := (1, -k, k)

/-- Direction vector of the second line --/
def dir2 : ℝ × ℝ × ℝ := (2, 1, -1)

/-- Two lines are coplanar if and only if k = -1/2 --/
theorem lines_coplanar_iff_k_eq_neg_half :
  (∃ (a b : ℝ), a • dir1 k + b • dir2 = (0, 0, 0)) ↔ k = -1/2 := by sorry

end lines_coplanar_iff_k_eq_neg_half_l2290_229087


namespace opposite_of_negative_2023_l2290_229097

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Prove that the opposite of -2023 is 2023. -/
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l2290_229097


namespace ribbon_length_proof_l2290_229082

theorem ribbon_length_proof (R : ℝ) : 
  (R / 2 + 2000 = R - ((R / 2 - 2000) / 2 + 2000)) → R = 12000 := by
  sorry

end ribbon_length_proof_l2290_229082


namespace divisors_of_210_l2290_229071

theorem divisors_of_210 : Finset.card (Nat.divisors 210) = 16 := by
  sorry

end divisors_of_210_l2290_229071


namespace difference_of_squares_l2290_229010

theorem difference_of_squares (m n : ℝ) : (3*m + n) * (3*m - n) = (3*m)^2 - n^2 := by
  sorry

end difference_of_squares_l2290_229010


namespace problem_statement_l2290_229095

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (a : ℝ) (x : ℝ) := a * (x - 1) / (x + 1)
noncomputable def h (a : ℝ) (x : ℝ) := f x - g a x

theorem problem_statement :
  (∀ x > 1, f x > g 2 x) ∧
  (∀ a ≤ 2, StrictMono (h a)) ∧
  (∀ a > 2, ∃ x y, x < y ∧ IsLocalMax (h a) x ∧ IsLocalMin (h a) y) ∧
  (∀ x > 0, f (x + 1) > x^2 / (exp x - 1)) := by
sorry

end problem_statement_l2290_229095


namespace kevin_ran_17_miles_l2290_229004

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Kevin's total running distance -/
def kevin_total_distance : ℝ :=
  let segment1 := distance 10 0.5
  let segment2 := distance 20 0.5
  let segment3 := distance 8 0.25
  segment1 + segment2 + segment3

/-- Theorem stating that Kevin's total running distance is 17 miles -/
theorem kevin_ran_17_miles : kevin_total_distance = 17 := by
  sorry

end kevin_ran_17_miles_l2290_229004


namespace no_valid_partition_l2290_229078

-- Define a partition type
def Partition := ℤ → Fin 3

-- Define the property that n, n-50, and n+1987 belong to different subsets
def ValidPartition (p : Partition) : Prop :=
  ∀ n : ℤ, p n ≠ p (n - 50) ∧ p n ≠ p (n + 1987) ∧ p (n - 50) ≠ p (n + 1987)

-- Theorem statement
theorem no_valid_partition : ¬∃ p : Partition, ValidPartition p := by
  sorry

end no_valid_partition_l2290_229078


namespace numerator_greater_than_denominator_l2290_229066

theorem numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ 4 * x - 3 > 9 - 2 * x → 2 < x ∧ x ≤ 3 := by
  sorry

end numerator_greater_than_denominator_l2290_229066


namespace work_completion_time_l2290_229099

/-- Given workers A, B, and C, where A can complete a job in 6 days,
    B can complete it in 5 days, and together they complete it in 2 days with C's help,
    prove that C alone can complete the job in 7.5 days. -/
theorem work_completion_time (a b c : ℝ) 
  (ha : a = 6) 
  (hb : b = 5) 
  (hab : 1 / a + 1 / b + 1 / c = 1 / 2) : 
  c = 15 / 2 := by
sorry

end work_completion_time_l2290_229099


namespace modular_inverse_of_two_mod_187_l2290_229086

theorem modular_inverse_of_two_mod_187 : ∃ x : ℤ, 0 ≤ x ∧ x < 187 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end modular_inverse_of_two_mod_187_l2290_229086


namespace parabola_vertex_l2290_229013

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = x^2 - 2x + 3 is (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, f x = (x - vertex.1)^2 + vertex.2) ∧ 
  (∀ x : ℝ, f x ≥ f vertex.1) :=
sorry

end parabola_vertex_l2290_229013


namespace largest_triangle_perimeter_l2290_229020

theorem largest_triangle_perimeter :
  ∀ x : ℤ,
  (8 : ℝ) + 11 > (x : ℝ) →
  (8 : ℝ) + (x : ℝ) > 11 →
  (11 : ℝ) + (x : ℝ) > 8 →
  (8 : ℝ) + 11 + (x : ℝ) ≤ 37 :=
by sorry

end largest_triangle_perimeter_l2290_229020


namespace w_over_y_value_l2290_229025

theorem w_over_y_value (w x y : ℝ) 
  (h1 : w / x = 1 / 3) 
  (h2 : (x + y) / y = 3.25) : 
  w / y = 0.75 := by
sorry

end w_over_y_value_l2290_229025
