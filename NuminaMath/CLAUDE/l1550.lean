import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1550_155088

theorem sufficient_not_necessary (a b : ℝ) :
  (0 < a ∧ a < b) → (1/4 : ℝ)^a > (1/4 : ℝ)^b ∧
  ∃ a' b' : ℝ, (1/4 : ℝ)^a' > (1/4 : ℝ)^b' ∧ ¬(0 < a' ∧ a' < b') :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1550_155088


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l1550_155024

/-- Given a flagpole and a building under similar conditions, 
    prove that the flagpole's shadow length is 45 meters. -/
theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 22)
  (h3 : building_shadow = 55)
  (h4 : flagpole_height / building_height = building_shadow / building_shadow) :
  flagpole_height * building_shadow / building_height = 45 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l1550_155024


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l1550_155027

-- Define the function f
def f (x : ℝ) : ℝ := |3*x + 2|

-- Theorem for part 1
theorem inequality_solution_set :
  {x : ℝ | f x < 4 - |x - 1|} = {x : ℝ | -5/4 < x ∧ x < 1/2} :=
sorry

-- Theorem for part 2
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ |x - a| - f x ≤ 1/m + 1/n) →
  ∃ a : ℝ, 0 < a ∧ a ≤ 10/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_of_a_l1550_155027


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1550_155056

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_with_current = 12)
  (h2 : current_speed = 2) :
  speed_with_current - 2 * current_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l1550_155056


namespace NUMINAMATH_CALUDE_evaluate_expression_l1550_155055

theorem evaluate_expression (a x : ℝ) (h : x = a + 10) : (x - a + 3) * (x - a - 2) = 104 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1550_155055


namespace NUMINAMATH_CALUDE_cookie_difference_l1550_155003

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 19

/-- The number of raisin cookies Helen baked this morning -/
def raisin_cookies : ℕ := 231

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_today : ℕ := 237

/-- The total number of chocolate chip cookies Helen baked -/
def total_choc_cookies : ℕ := cookies_yesterday + cookies_today

/-- The difference between chocolate chip cookies and raisin cookies -/
theorem cookie_difference : total_choc_cookies - raisin_cookies = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1550_155003


namespace NUMINAMATH_CALUDE_carl_driving_hours_l1550_155031

/-- The number of hours Carl drives per day before promotion -/
def hours_per_day : ℝ :=
  2

/-- The number of additional hours Carl drives per week after promotion -/
def additional_hours_per_week : ℝ :=
  6

/-- The number of hours Carl drives in two weeks after promotion -/
def hours_in_two_weeks_after : ℝ :=
  40

/-- The number of days in two weeks -/
def days_in_two_weeks : ℝ :=
  14

theorem carl_driving_hours :
  hours_per_day * days_in_two_weeks + additional_hours_per_week * 2 = hours_in_two_weeks_after :=
by sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l1550_155031


namespace NUMINAMATH_CALUDE_muffins_for_sale_is_108_l1550_155029

/-- Calculate the number of muffins for sale given the following conditions:
  * 3 boys each make 12 muffins
  * 2 girls each make 20 muffins
  * 1 girl makes 15 muffins
  * 2 boys each make 18 muffins
  * 15% of all muffins will not make it to the sale
-/
def muffinsForSale : ℕ :=
  let boys_group1 := 3 * 12
  let boys_group2 := 2 * 18
  let girls_group1 := 2 * 20
  let girls_group2 := 1 * 15
  let total_muffins := boys_group1 + boys_group2 + girls_group1 + girls_group2
  let muffins_not_for_sale := (total_muffins : ℚ) * (15 : ℚ) / (100 : ℚ)
  ⌊(total_muffins : ℚ) - muffins_not_for_sale⌋.toNat

/-- Theorem stating that the number of muffins for sale is 108 -/
theorem muffins_for_sale_is_108 : muffinsForSale = 108 := by
  sorry

end NUMINAMATH_CALUDE_muffins_for_sale_is_108_l1550_155029


namespace NUMINAMATH_CALUDE_regular_polygon_with_90_degree_difference_has_8_sides_l1550_155014

-- Define a regular polygon
structure RegularPolygon where
  n : ℕ  -- number of sides
  n_ge_3 : n ≥ 3  -- a polygon has at least 3 sides

-- Define the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℚ :=
  (p.n - 2) * 180 / p.n

-- Define the exterior angle of a regular polygon
def exteriorAngle (p : RegularPolygon) : ℚ :=
  360 / p.n

-- Theorem: A regular polygon where each interior angle is 90° larger than each exterior angle has 8 sides
theorem regular_polygon_with_90_degree_difference_has_8_sides :
  ∃ (p : RegularPolygon), interiorAngle p - exteriorAngle p = 90 → p.n = 8 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_90_degree_difference_has_8_sides_l1550_155014


namespace NUMINAMATH_CALUDE_equation_solution_l1550_155028

theorem equation_solution (k : ℝ) : 
  (∀ x : ℝ, -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4)) ↔ k = -13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1550_155028


namespace NUMINAMATH_CALUDE_surface_is_one_sheet_hyperboloid_l1550_155008

/-- The equation of the surface -/
def surface_equation (x y z : ℝ) : Prop :=
  x^2 - 2*x - 3*y^2 + 12*y + 2*z^2 + 12*z - 11 = 0

/-- The standard form of a one-sheet hyperboloid -/
def one_sheet_hyperboloid (a b c : ℝ) (x y z : ℝ) : Prop :=
  (x - a)^2 / 18 - (y - b)^2 / 6 + (z - c)^2 / 9 = 1

/-- Theorem stating that the surface equation represents a one-sheet hyperboloid -/
theorem surface_is_one_sheet_hyperboloid :
  ∀ x y z : ℝ, surface_equation x y z ↔ one_sheet_hyperboloid 1 2 (-3) x y z :=
by sorry

end NUMINAMATH_CALUDE_surface_is_one_sheet_hyperboloid_l1550_155008


namespace NUMINAMATH_CALUDE_star_equation_solution_l1550_155053

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

/-- Theorem stating that if 6 ★ x = 45, then x = 19/3 -/
theorem star_equation_solution :
  (star 6 x = 45) → x = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1550_155053


namespace NUMINAMATH_CALUDE_water_needed_for_mixture_l1550_155020

/-- Given the initial mixture composition and the desired total volume, 
    prove that the amount of water needed is 0.24 liters. -/
theorem water_needed_for_mixture (initial_chemical_b : ℝ) (initial_water : ℝ) 
  (initial_mixture : ℝ) (desired_volume : ℝ) 
  (h1 : initial_chemical_b = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : initial_mixture = 0.08)
  (h4 : desired_volume = 0.64)
  (h5 : initial_chemical_b + initial_water = initial_mixture) : 
  desired_volume * (initial_water / initial_mixture) = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_mixture_l1550_155020


namespace NUMINAMATH_CALUDE_jones_elementary_population_l1550_155010

theorem jones_elementary_population :
  let total_students : ℕ := 360
  let boy_percentage : ℚ := 1/2
  let representative_boys : ℕ := 90
  (representative_boys : ℚ) / (boy_percentage * total_students) = boy_percentage →
  total_students = 360 := by
sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l1550_155010


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1550_155011

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_3 + a_9 = 15 - a_6, prove that a_6 = 5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_equation : a 3 + a 9 = 15 - a 6) : 
  a 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1550_155011


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1550_155022

/-- Given a principal sum and a time period of 2 years, if the simple interest
    is one-fifth of the principal sum, then the rate of interest per annum is 10%. -/
theorem interest_rate_proof (P : ℝ) (P_pos : P > 0) : 
  (P * 2 * 10 / 100 = P / 5) → 10 = (P / 5) / P * 100 / 2 := by
  sorry

#check interest_rate_proof

end NUMINAMATH_CALUDE_interest_rate_proof_l1550_155022


namespace NUMINAMATH_CALUDE_trays_per_trip_is_eight_l1550_155099

-- Define the problem parameters
def trays_table1 : ℕ := 27
def trays_table2 : ℕ := 5
def total_trips : ℕ := 4

-- Define the total number of trays
def total_trays : ℕ := trays_table1 + trays_table2

-- Theorem statement
theorem trays_per_trip_is_eight :
  total_trays / total_trips = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_trays_per_trip_is_eight_l1550_155099


namespace NUMINAMATH_CALUDE_twenty_matches_exist_l1550_155093

/-- Represents the number of matches played up to each day -/
def MatchSequence : Type := Fin 71 → ℕ

/-- The sequence is non-decreasing -/
def IsNonDecreasing (s : MatchSequence) : Prop :=
  ∀ i j : Fin 71, i < j → s i ≤ s j

/-- The difference between consecutive days is at least 1 and at most 12 -/
def HasValidDifference (s : MatchSequence) : Prop :=
  ∀ i : Fin 70, 1 ≤ s (i + 1) - s i ∧ s (i + 1) - s i ≤ 12

/-- The total number of matches played in 70 days does not exceed 120 -/
def HasValidTotal (s : MatchSequence) : Prop :=
  s ⟨70, by norm_num⟩ ≤ 120

theorem twenty_matches_exist (s : MatchSequence)
  (h1 : IsNonDecreasing s)
  (h2 : HasValidDifference s)
  (h3 : HasValidTotal s)
  (h4 : s 0 = 0) :
  ∃ i j : Fin 71, i < j ∧ s j - s i = 20 := by
  sorry


end NUMINAMATH_CALUDE_twenty_matches_exist_l1550_155093


namespace NUMINAMATH_CALUDE_inequality_proof_l1550_155038

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1550_155038


namespace NUMINAMATH_CALUDE_school_garden_flowers_l1550_155035

theorem school_garden_flowers :
  let total_flowers : ℕ := 96
  let green_flowers : ℕ := 9
  let red_flowers : ℕ := 3 * green_flowers
  let blue_flowers : ℕ := total_flowers / 2
  let yellow_flowers : ℕ := total_flowers - (green_flowers + red_flowers + blue_flowers)
  yellow_flowers = 12 := by
sorry

end NUMINAMATH_CALUDE_school_garden_flowers_l1550_155035


namespace NUMINAMATH_CALUDE_library_book_count_l1550_155087

/-- The number of books in the library after a series of transactions --/
def books_in_library (initial : ℕ) (taken_tuesday : ℕ) (returned_wednesday : ℕ) (taken_thursday : ℕ) : ℕ :=
  initial - taken_tuesday + returned_wednesday - taken_thursday

/-- Theorem: The number of books in the library after the given transactions is 150 --/
theorem library_book_count : 
  books_in_library 250 120 35 15 = 150 := by
  sorry

#eval books_in_library 250 120 35 15

end NUMINAMATH_CALUDE_library_book_count_l1550_155087


namespace NUMINAMATH_CALUDE_nested_expression_sum_l1550_155018

def nested_expression : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_expression n)

theorem nested_expression_sum : nested_expression 8 = 1022 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_sum_l1550_155018


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_nonpositive_l1550_155060

theorem negation_of_existence_squared_nonpositive :
  (¬ ∃ x : ℝ, x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_nonpositive_l1550_155060


namespace NUMINAMATH_CALUDE_divisibility_by_203_l1550_155062

theorem divisibility_by_203 (n : ℕ+) : 
  (2013^n.val - 1803^n.val - 1781^n.val + 1774^n.val) % 203 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_203_l1550_155062


namespace NUMINAMATH_CALUDE_complement_A_in_U_l1550_155048

def U : Set ℕ := {x | x ≥ 3}
def A : Set ℕ := {x | x^2 ≥ 10}

theorem complement_A_in_U : (U \ A) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l1550_155048


namespace NUMINAMATH_CALUDE_wedding_guests_fraction_l1550_155019

theorem wedding_guests_fraction (total_guests : ℚ) : 
  let children_fraction : ℚ := 1/8
  let adult_fraction : ℚ := 1 - children_fraction
  let men_fraction_of_adults : ℚ := 3/7
  let women_fraction_of_adults : ℚ := 1 - men_fraction_of_adults
  let adult_women_fraction : ℚ := adult_fraction * women_fraction_of_adults
  adult_women_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_wedding_guests_fraction_l1550_155019


namespace NUMINAMATH_CALUDE_fixed_point_implies_sqrt_two_l1550_155047

noncomputable section

-- Define the logarithmic function
def log_func (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the power function
def power_func (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem fixed_point_implies_sqrt_two 
  (a : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_one : a ≠ 1) 
  (A : ℝ × ℝ) 
  (α : ℝ) 
  (h_log_point : log_func a (A.1 - 3) + 2 = A.2)
  (h_power_point : power_func α A.1 = A.2) :
  power_func α 2 = Real.sqrt 2 :=
sorry

end

end NUMINAMATH_CALUDE_fixed_point_implies_sqrt_two_l1550_155047


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l1550_155012

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l1550_155012


namespace NUMINAMATH_CALUDE_sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one_l1550_155007

theorem sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one :
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one_l1550_155007


namespace NUMINAMATH_CALUDE_dan_limes_remaining_l1550_155054

theorem dan_limes_remaining (initial_limes given_limes : ℕ) : 
  initial_limes = 9 → given_limes = 4 → initial_limes - given_limes = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_limes_remaining_l1550_155054


namespace NUMINAMATH_CALUDE_jennifer_fruits_left_l1550_155057

def fruits_left (pears oranges apples cherries grapes : ℕ) 
  (pears_given oranges_given apples_given cherries_given grapes_given : ℕ) : ℕ :=
  (pears - pears_given) + (oranges - oranges_given) + (apples - apples_given) + 
  (cherries - cherries_given) + (grapes - grapes_given)

theorem jennifer_fruits_left : 
  let pears : ℕ := 15
  let oranges : ℕ := 30
  let apples : ℕ := 2 * pears
  let cherries : ℕ := oranges / 2
  let grapes : ℕ := 3 * apples
  fruits_left pears oranges apples cherries grapes 3 5 5 7 3 = 157 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_fruits_left_l1550_155057


namespace NUMINAMATH_CALUDE_remainder_9053_div_98_l1550_155039

theorem remainder_9053_div_98 : 9053 % 98 = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9053_div_98_l1550_155039


namespace NUMINAMATH_CALUDE_X_is_greatest_l1550_155096

def X : ℚ := 2010/2009 + 2010/2011
def Y : ℚ := 2010/2011 + 2012/2011
def Z : ℚ := 2011/2010 + 2011/2012

theorem X_is_greatest : X > Y ∧ X > Z := by
  sorry

end NUMINAMATH_CALUDE_X_is_greatest_l1550_155096


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1550_155066

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1550_155066


namespace NUMINAMATH_CALUDE_expand_expression_l1550_155090

theorem expand_expression (x y z : ℝ) : 
  (x + 12) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 36 * y + 24 * z + 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1550_155090


namespace NUMINAMATH_CALUDE_inequality_implies_sum_nonpositive_l1550_155006

theorem inequality_implies_sum_nonpositive 
  {a b x y : ℝ} 
  (h1 : 1 < a) 
  (h2 : a < b) 
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) : 
  x + y ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_sum_nonpositive_l1550_155006


namespace NUMINAMATH_CALUDE_linear_correlation_proof_l1550_155081

/-- Determines if two variables are linearly correlated based on the correlation coefficient and critical value -/
def are_linearly_correlated (r : ℝ) (r_critical : ℝ) : Prop :=
  |r| > r_critical

/-- Theorem stating that given conditions lead to linear correlation -/
theorem linear_correlation_proof (r r_critical : ℝ) 
  (h1 : r = -0.9362)
  (h2 : r_critical = 0.8013) :
  are_linearly_correlated r r_critical :=
by
  sorry

#check linear_correlation_proof

end NUMINAMATH_CALUDE_linear_correlation_proof_l1550_155081


namespace NUMINAMATH_CALUDE_nine_oclock_right_angle_l1550_155068

/-- The angle between clock hands at a given hour -/
def clock_angle (hour : ℕ) : ℝ :=
  sorry

/-- A right angle is 90 degrees -/
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

theorem nine_oclock_right_angle : is_right_angle (clock_angle 9) := by
  sorry

end NUMINAMATH_CALUDE_nine_oclock_right_angle_l1550_155068


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1550_155033

theorem perfect_square_condition (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 - 6*x + a^2 = y^2) → (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1550_155033


namespace NUMINAMATH_CALUDE_divisible_by_four_even_equivalence_l1550_155080

theorem divisible_by_four_even_equivalence :
  (∀ n : ℤ, 4 ∣ n → Even n) ↔ (∀ n : ℤ, ¬Even n → ¬(4 ∣ n)) := by sorry

end NUMINAMATH_CALUDE_divisible_by_four_even_equivalence_l1550_155080


namespace NUMINAMATH_CALUDE_calculate_face_value_l1550_155094

/-- The relationship between banker's discount, true discount, and face value -/
def bankers_discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + td^2 / fv

/-- Given the banker's discount and true discount, calculate the face value -/
theorem calculate_face_value (bd td : ℚ) (h : bankers_discount_relation bd td 300) :
  bd = 72 ∧ td = 60 → 300 = 300 := by sorry

end NUMINAMATH_CALUDE_calculate_face_value_l1550_155094


namespace NUMINAMATH_CALUDE_common_chord_equation_l1550_155036

theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 = 4) ∧ (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
  (x - y + 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1550_155036


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l1550_155059

/-- The number of posts in the fence -/
def num_posts : ℕ := 36

/-- The width of each post in inches -/
def post_width_inches : ℕ := 6

/-- The space between adjacent posts in feet -/
def space_between_posts : ℕ := 4

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- Conversion factor from inches to feet -/
def inches_to_feet : ℚ := 1 / 12

theorem square_fence_perimeter :
  let posts_per_side : ℕ := num_posts / num_sides
  let post_width_feet : ℚ := post_width_inches * inches_to_feet
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℚ := gaps_per_side * space_between_posts + posts_per_side * post_width_feet
  num_sides * side_length = 130 := by sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l1550_155059


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l1550_155032

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents two concentric circles -/
structure ConcentricCircles where
  inner : Circle
  outer : Circle
  h : inner.radius < outer.radius

/-- Represents three circles tangent to two concentric circles and to each other -/
structure TangentCircles (cc : ConcentricCircles) where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  tangent_to_concentric : 
    c1.radius = cc.outer.radius - cc.inner.radius ∧
    c2.radius = cc.outer.radius - cc.inner.radius ∧
    c3.radius = cc.outer.radius - cc.inner.radius
  tangent_to_each_other : True  -- This is a simplification, as we can't easily express tangency

/-- The main theorem: If three circles are tangent to two concentric circles and to each other,
    then the ratio of the radii of the concentric circles is 3 -/
theorem concentric_circles_ratio 
  (cc : ConcentricCircles) 
  (tc : TangentCircles cc) : 
  cc.outer.radius / cc.inner.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l1550_155032


namespace NUMINAMATH_CALUDE_committee_selection_l1550_155075

theorem committee_selection (n : ℕ) : 
  (n.choose 3 = 20) → (n.choose 4 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l1550_155075


namespace NUMINAMATH_CALUDE_not_all_data_has_regression_equation_l1550_155017

-- Define the basic concepts
def DataSet : Type := Set (ℝ × ℝ)
def RegressionEquation : Type := ℝ → ℝ

-- Define the properties mentioned in the problem
def hasCorrelation (d : DataSet) : Prop := sorry
def hasCausalRelationship (d : DataSet) : Prop := sorry
def canBeRepresentedByScatterPlot (d : DataSet) : Prop := sorry
def hasLinearCorrelation (d : DataSet) : Prop := sorry
def hasRegressionEquation (d : DataSet) : Prop := sorry

-- Define the statements from the problem
axiom correlation_not_causation : 
  ∀ d : DataSet, hasCorrelation d → ¬ (hasCausalRelationship d)

axiom scatter_plot_reflects_correlation : 
  ∀ d : DataSet, hasCorrelation d → canBeRepresentedByScatterPlot d

axiom regression_line_best_represents : 
  ∀ d : DataSet, hasLinearCorrelation d → hasRegressionEquation d

-- The theorem to be proved
theorem not_all_data_has_regression_equation :
  ¬ (∀ d : DataSet, hasRegressionEquation d) := by
  sorry

end NUMINAMATH_CALUDE_not_all_data_has_regression_equation_l1550_155017


namespace NUMINAMATH_CALUDE_moral_education_story_time_l1550_155061

/-- Proves that telling a 7-minute "Moral Education Story" every week for 20 weeks equals 2 hours and 20 minutes -/
theorem moral_education_story_time :
  let story_duration : ℕ := 7  -- Duration of one story in minutes
  let weeks : ℕ := 20  -- Number of weeks
  let total_minutes : ℕ := story_duration * weeks
  let hours : ℕ := total_minutes / 60
  let remaining_minutes : ℕ := total_minutes % 60
  (hours = 2 ∧ remaining_minutes = 20) := by
  sorry


end NUMINAMATH_CALUDE_moral_education_story_time_l1550_155061


namespace NUMINAMATH_CALUDE_coefficient_a4_value_l1550_155043

theorem coefficient_a4_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5) →
  a₄ = -5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a4_value_l1550_155043


namespace NUMINAMATH_CALUDE_evaluate_expression_l1550_155023

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1550_155023


namespace NUMINAMATH_CALUDE_inequality_proof_l1550_155067

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1 / (1/a + 1/b) + 1 / (1/c + 1/d) ≤ 1 / (1/(a+c) + 1/(b+d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1550_155067


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1550_155072

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 1110 ∧ profit_percentage = 20 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 925 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1550_155072


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l1550_155095

theorem product_remainder_by_10 : (2456 * 7294 * 91803) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l1550_155095


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1550_155034

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 4 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1550_155034


namespace NUMINAMATH_CALUDE_camping_hike_distance_l1550_155089

/-- Hiking distances on a camping trip -/
theorem camping_hike_distance 
  (total_distance : ℝ)
  (car_to_stream : ℝ)
  (stream_to_meadow : ℝ)
  (h_total : total_distance = 0.7)
  (h_car_stream : car_to_stream = 0.2)
  (h_stream_meadow : stream_to_meadow = 0.4) :
  total_distance - (car_to_stream + stream_to_meadow) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l1550_155089


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1550_155037

theorem perfect_square_quadratic (n : ℤ) : ∃ m : ℤ, 4 * n^2 + 12 * n + 9 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1550_155037


namespace NUMINAMATH_CALUDE_equation_solution_l1550_155004

theorem equation_solution : 
  {x : ℝ | x + 45 / (x - 4) = -10} = {-1, -5} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1550_155004


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1550_155085

/-- The function f(x) = x(x-1)(x-2) -/
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2)

/-- The theorem stating that the derivative of f at x=0 is 2 -/
theorem derivative_f_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1550_155085


namespace NUMINAMATH_CALUDE_circle_tangents_theorem_l1550_155083

/-- Given two circles with radii x and y touching a circle with radius R,
    and the distance between points of contact a, this theorem proves
    the squared lengths of their common tangents. -/
theorem circle_tangents_theorem
  (R x y a : ℝ)
  (h_pos_R : R > 0)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_pos_a : a > 0) :
  (∃ (l_ext : ℝ), l_ext^2 = (a/R)^2 * (R+x)*(R+y) ∨ l_ext^2 = (a/R)^2 * (R-x)*(R-y)) ∧
  (∃ (l_int : ℝ), l_int^2 = (a/R)^2 * (R+y)*(R-x)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_theorem_l1550_155083


namespace NUMINAMATH_CALUDE_cylinder_max_lateral_area_l1550_155065

theorem cylinder_max_lateral_area (sphere_area : ℝ) (h_sphere_area : sphere_area = 20 * Real.pi) :
  let R := (sphere_area / (4 * Real.pi)) ^ (1/2)
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 
    r^2 + (l/2)^2 = R^2 ∧ 
    ∀ (r' l' : ℝ), r' > 0 → l' > 0 → r'^2 + (l'/2)^2 = R^2 → 
      2 * Real.pi * r * l ≤ 2 * Real.pi * r' * l' :=
by sorry

end NUMINAMATH_CALUDE_cylinder_max_lateral_area_l1550_155065


namespace NUMINAMATH_CALUDE_brick_height_proof_l1550_155079

/-- Proves that the height of each brick is 6 cm given the wall and brick dimensions --/
theorem brick_height_proof (wall_length wall_width wall_height : ℝ)
                           (brick_length brick_width : ℝ)
                           (num_bricks : ℕ) :
  wall_length = 700 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 5600 →
  ∃ (h : ℝ), h = 6 ∧ 
    wall_length * wall_width * wall_height = 
    num_bricks * brick_length * brick_width * h :=
by
  sorry

end NUMINAMATH_CALUDE_brick_height_proof_l1550_155079


namespace NUMINAMATH_CALUDE_scores_mode_is_37_l1550_155044

def scores : List Nat := [35, 37, 39, 37, 38, 38, 37]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem scores_mode_is_37 : mode scores = 37 := by
  sorry

end NUMINAMATH_CALUDE_scores_mode_is_37_l1550_155044


namespace NUMINAMATH_CALUDE_inverse_sum_mod_31_l1550_155021

theorem inverse_sum_mod_31 : ∃ (a b : ℤ), (5 * a) % 31 = 1 ∧ (5 * 5 * 5 * b) % 31 = 1 ∧ (a + b) % 31 = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_31_l1550_155021


namespace NUMINAMATH_CALUDE_triangle_area_bound_l1550_155086

theorem triangle_area_bound (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) / 4 ≤ Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_bound_l1550_155086


namespace NUMINAMATH_CALUDE_expression_value_l1550_155073

theorem expression_value : 
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1550_155073


namespace NUMINAMATH_CALUDE_rectangle_area_l1550_155002

/-- The area of a rectangle given its perimeter and width -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 56) (h2 : width = 16) :
  let length := (perimeter - 2 * width) / 2
  width * length = 192 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1550_155002


namespace NUMINAMATH_CALUDE_parallelogram_area_equals_rectangle_area_l1550_155071

/-- Represents a rectangle with a given base and area -/
structure Rectangle where
  base : ℝ
  area : ℝ

/-- Represents a parallelogram with a given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Theorem: Given a rectangle with base 6 and area 24, and a parallelogram sharing the same base and height,
    the area of the parallelogram is 24 -/
theorem parallelogram_area_equals_rectangle_area 
  (rect : Rectangle) 
  (para : Parallelogram) 
  (h1 : rect.base = 6) 
  (h2 : rect.area = 24) 
  (h3 : para.base = rect.base) 
  (h4 : para.height = rect.area / rect.base) : 
  para.base * para.height = 24 := by
  sorry

#check parallelogram_area_equals_rectangle_area

end NUMINAMATH_CALUDE_parallelogram_area_equals_rectangle_area_l1550_155071


namespace NUMINAMATH_CALUDE_binomial_variance_example_l1550_155063

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a random variable -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Given a random variable X following the binomial distribution B(6, 1/3), its variance D(X) is 4/3 -/
theorem binomial_variance_example :
  let X : BinomialDistribution := ⟨6, 1/3, by norm_num⟩
  variance X = 4/3 := by sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l1550_155063


namespace NUMINAMATH_CALUDE_quadratic_always_positive_range_l1550_155084

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_range_l1550_155084


namespace NUMINAMATH_CALUDE_remaining_score_is_80_l1550_155013

/-- Given 5 students with 4 known scores and an average score, 
    calculate the remaining score -/
def remaining_score (s1 s2 s3 s4 : ℕ) (avg : ℚ) : ℚ :=
  5 * avg - (s1 + s2 + s3 + s4)

/-- Theorem: The remaining score is 80 -/
theorem remaining_score_is_80 :
  remaining_score 85 95 75 65 80 = 80 := by
  sorry

#eval remaining_score 85 95 75 65 80

end NUMINAMATH_CALUDE_remaining_score_is_80_l1550_155013


namespace NUMINAMATH_CALUDE_triangle_area_l1550_155064

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Vectors m and n
  let m := (Real.sin C, Real.sin B * Real.cos A)
  let n := (b, 2 * c)
  -- m · n = 0
  m.1 * n.1 + m.2 * n.2 = 0 →
  -- a = 2√3
  a = 2 * Real.sqrt 3 →
  -- sin B + sin C = 1
  Real.sin B + Real.sin C = 1 →
  -- Area of triangle ABC is √3
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1550_155064


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_effect_l1550_155000

theorem price_reduction_sales_increase_effect 
  (original_price original_sales : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 30)
  (h2 : sales_increase_percent = 80) :
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_sales := original_sales * (1 + sales_increase_percent / 100)
  let original_revenue := original_price * original_sales
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.26 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_effect_l1550_155000


namespace NUMINAMATH_CALUDE_hash_difference_l1550_155001

def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 6 5) - (hash 5 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1550_155001


namespace NUMINAMATH_CALUDE_total_cars_produced_l1550_155049

theorem total_cars_produced (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_produced_l1550_155049


namespace NUMINAMATH_CALUDE_range_of_r_l1550_155078

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ r x = y) ↔ y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l1550_155078


namespace NUMINAMATH_CALUDE_division_problem_addition_problem_multiplication_problem_l1550_155026

-- Problem 1
theorem division_problem : 246 / 73 = 3 + 27 / 73 := by sorry

-- Problem 2
theorem addition_problem : 9999 + 999 + 99 + 9 = 11106 := by sorry

-- Problem 3
theorem multiplication_problem : 25 * 29 * 4 = 2900 := by sorry

end NUMINAMATH_CALUDE_division_problem_addition_problem_multiplication_problem_l1550_155026


namespace NUMINAMATH_CALUDE_right_triangle_max_sin_product_l1550_155058

theorem right_triangle_max_sin_product (A B C : Real) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ -- Angles are non-negative
  A + B + C = π ∧ -- Sum of angles in a triangle
  C = π / 2 → -- Right angle condition
  ∀ (x y : Real), 0 ≤ x ∧ 0 ≤ y ∧ x + y = π / 2 → 
    Real.sin x * Real.sin y ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_sin_product_l1550_155058


namespace NUMINAMATH_CALUDE_angle_ABF_measure_l1550_155040

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle ABF in a regular octagon -/
def angle_ABF (octagon : RegularOctagon) : ℝ := 22.5

theorem angle_ABF_measure (octagon : RegularOctagon) :
  angle_ABF octagon = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABF_measure_l1550_155040


namespace NUMINAMATH_CALUDE_perpendicular_segments_sum_maximum_l1550_155050

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point within the circle's disk
structure PointInDisk (c : Circle) where
  point : ℝ × ℝ
  in_disk : Real.sqrt ((point.1 - c.center.1)^2 + (point.2 - c.center.2)^2) ≤ c.radius

-- Define two perpendicular line segments from a point to the circle's boundary
structure PerpendicularSegments (c : Circle) (p : PointInDisk c) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_circle1 : Real.sqrt ((endpoint1.1 - c.center.1)^2 + (endpoint1.2 - c.center.2)^2) = c.radius
  on_circle2 : Real.sqrt ((endpoint2.1 - c.center.1)^2 + (endpoint2.2 - c.center.2)^2) = c.radius
  perpendicular : (endpoint1.1 - p.point.1) * (endpoint2.1 - p.point.1) + 
                  (endpoint1.2 - p.point.2) * (endpoint2.2 - p.point.2) = 0

-- Theorem statement
theorem perpendicular_segments_sum_maximum (c : Circle) (p : PointInDisk c) 
  (segments : PerpendicularSegments c p) :
  ∃ (max_segments : PerpendicularSegments c p),
    (Real.sqrt ((max_segments.endpoint1.1 - p.point.1)^2 + (max_segments.endpoint1.2 - p.point.2)^2) +
     Real.sqrt ((max_segments.endpoint2.1 - p.point.1)^2 + (max_segments.endpoint2.2 - p.point.2)^2)) ≥
    (Real.sqrt ((segments.endpoint1.1 - p.point.1)^2 + (segments.endpoint1.2 - p.point.2)^2) +
     Real.sqrt ((segments.endpoint2.1 - p.point.1)^2 + (segments.endpoint2.2 - p.point.2)^2)) ∧
    (Real.sqrt ((max_segments.endpoint1.1 - p.point.1)^2 + (max_segments.endpoint1.2 - p.point.2)^2) =
     Real.sqrt ((max_segments.endpoint2.1 - p.point.1)^2 + (max_segments.endpoint2.2 - p.point.2)^2)) ∧
    (Real.sqrt ((max_segments.endpoint1.1 - max_segments.endpoint2.1)^2 + 
                (max_segments.endpoint1.2 - max_segments.endpoint2.2)^2) = 2 * c.radius) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_segments_sum_maximum_l1550_155050


namespace NUMINAMATH_CALUDE_levels_beaten_l1550_155097

theorem levels_beaten (total_levels : ℕ) (ratio : ℚ) : total_levels = 32 ∧ ratio = 3 / 1 → 
  ∃ (beaten : ℕ), beaten = 24 ∧ beaten * (1 + 1 / ratio) = total_levels := by
sorry

end NUMINAMATH_CALUDE_levels_beaten_l1550_155097


namespace NUMINAMATH_CALUDE_three_primes_sum_to_86_l1550_155045

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem three_primes_sum_to_86 :
  ∃ (a b c : ℕ), isPrime a ∧ isPrime b ∧ isPrime c ∧ a + b + c = 86 ∧
  (∀ (x y z : ℕ), isPrime x ∧ isPrime y ∧ isPrime z ∧ x + y + z = 86 →
    (x = 2 ∧ y = 5 ∧ z = 79) ∨
    (x = 2 ∧ y = 11 ∧ z = 73) ∨
    (x = 2 ∧ y = 13 ∧ z = 71) ∨
    (x = 2 ∧ y = 17 ∧ z = 67) ∨
    (x = 2 ∧ y = 23 ∧ z = 61) ∨
    (x = 2 ∧ y = 31 ∧ z = 53) ∨
    (x = 2 ∧ y = 37 ∧ z = 47) ∨
    (x = 2 ∧ y = 41 ∧ z = 43) ∨
    (x = 5 ∧ y = 2 ∧ z = 79) ∨
    (x = 11 ∧ y = 2 ∧ z = 73) ∨
    (x = 13 ∧ y = 2 ∧ z = 71) ∨
    (x = 17 ∧ y = 2 ∧ z = 67) ∨
    (x = 23 ∧ y = 2 ∧ z = 61) ∨
    (x = 31 ∧ y = 2 ∧ z = 53) ∨
    (x = 37 ∧ y = 2 ∧ z = 47) ∨
    (x = 41 ∧ y = 2 ∧ z = 43) ∨
    (x = 79 ∧ y = 2 ∧ z = 5) ∨
    (x = 73 ∧ y = 2 ∧ z = 11) ∨
    (x = 71 ∧ y = 2 ∧ z = 13) ∨
    (x = 67 ∧ y = 2 ∧ z = 17) ∨
    (x = 61 ∧ y = 2 ∧ z = 23) ∨
    (x = 53 ∧ y = 2 ∧ z = 31) ∨
    (x = 47 ∧ y = 2 ∧ z = 37) ∨
    (x = 43 ∧ y = 2 ∧ z = 41)) :=
by sorry


end NUMINAMATH_CALUDE_three_primes_sum_to_86_l1550_155045


namespace NUMINAMATH_CALUDE_progression_product_exceeds_100000_l1550_155046

theorem progression_product_exceeds_100000 (n : ℕ) : 
  (n ≥ 11 ∧ ∀ k < 11, k > 0 → 10^((k * (k + 1)) / 22) ≤ 10^5) ↔ 
  (∀ k ≤ n, 10^((k * (k + 1)) / 22) > 10^5 ↔ k ≥ 11) := by
  sorry

end NUMINAMATH_CALUDE_progression_product_exceeds_100000_l1550_155046


namespace NUMINAMATH_CALUDE_ethan_candle_coconut_oil_l1550_155051

/-- The amount of coconut oil used in each candle, given the total weight of candles,
    the number of candles, and the amount of beeswax per candle. -/
def coconut_oil_per_candle (total_weight : ℕ) (num_candles : ℕ) (beeswax_per_candle : ℕ) : ℚ :=
  (total_weight - num_candles * beeswax_per_candle) / num_candles

theorem ethan_candle_coconut_oil :
  coconut_oil_per_candle 63 (10 - 3) 8 = 1 := by sorry

end NUMINAMATH_CALUDE_ethan_candle_coconut_oil_l1550_155051


namespace NUMINAMATH_CALUDE_calculation_proof_l1550_155005

theorem calculation_proof : (1/2)⁻¹ - Real.sqrt 3 * Real.tan (30 * π / 180) + (π - 2023)^0 + |-2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1550_155005


namespace NUMINAMATH_CALUDE_potato_cooking_time_l1550_155009

theorem potato_cooking_time 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_potatoes = 16) 
  (h2 : cooked_potatoes = 7) 
  (h3 : remaining_time = 45) :
  remaining_time / (total_potatoes - cooked_potatoes) = 5 := by
  sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l1550_155009


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1550_155082

theorem imaginary_unit_power (i : ℂ) : i * i = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1550_155082


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l1550_155030

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : d ≠ 0)
  (h' : ∀ n, a (n + 1) = a n + d)

/-- A geometric sequence -/
structure GeometricSequence :=
  (b : ℕ → ℝ)
  (r : ℝ)
  (h : ∀ n, b (n + 1) = r * b n)

theorem arithmetic_geometric_sequence_property
  (as : ArithmeticSequence)
  (gs : GeometricSequence)
  (h1 : 2 * as.a 3 - (as.a 7)^2 + 2 * as.a 11 = 0)
  (h2 : gs.b 7 = as.a 7) :
  gs.b 6 * gs.b 8 = 16 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l1550_155030


namespace NUMINAMATH_CALUDE_subtraction_multiplication_theorem_l1550_155069

theorem subtraction_multiplication_theorem : ((3.65 - 1.27) * 2) = 4.76 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_theorem_l1550_155069


namespace NUMINAMATH_CALUDE_max_area_triangle_OAB_l1550_155074

/-- The maximum area of triangle OAB in the complex plane -/
theorem max_area_triangle_OAB :
  ∀ (α β : ℂ),
  β = (1 + Complex.I) * α →
  Complex.abs (α - 2) = 1 →
  (∀ (S : ℝ),
    S = (Complex.abs α * Complex.abs β * Real.sin (Real.pi / 4)) / 2 →
    S ≤ 9 / 2) ∧
  ∃ (α₀ β₀ : ℂ),
    β₀ = (1 + Complex.I) * α₀ ∧
    Complex.abs (α₀ - 2) = 1 ∧
    (Complex.abs α₀ * Complex.abs β₀ * Real.sin (Real.pi / 4)) / 2 = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_OAB_l1550_155074


namespace NUMINAMATH_CALUDE_correct_total_amount_paid_l1550_155070

/-- Calculates the total amount paid for fruits with discounts --/
def totalAmountPaid (
  peachPrice peachCount peachDiscountThreshold peachDiscount : ℚ)
  (applePrice appleCount appleDiscountThreshold appleDiscount : ℚ)
  (orangePrice orangeCount orangeDiscountThreshold orangeDiscount : ℚ)
  (grapefruitPrice grapefruitCount grapefruitDiscountThreshold grapefruitDiscount : ℚ)
  (bundleDiscountThreshold1 bundleDiscountThreshold2 bundleDiscountThreshold3 bundleDiscount : ℚ) : ℚ :=
  let peachTotal := peachPrice * peachCount
  let appleTotal := applePrice * appleCount
  let orangeTotal := orangePrice * orangeCount
  let grapefruitTotal := grapefruitPrice * grapefruitCount
  let peachDiscountTimes := (peachTotal / peachDiscountThreshold).floor
  let appleDiscountTimes := (appleTotal / appleDiscountThreshold).floor
  let orangeDiscountTimes := (orangeTotal / orangeDiscountThreshold).floor
  let grapefruitDiscountTimes := (grapefruitTotal / grapefruitDiscountThreshold).floor
  let totalBeforeDiscount := peachTotal + appleTotal + orangeTotal + grapefruitTotal
  let individualDiscounts := peachDiscountTimes * peachDiscount + 
                             appleDiscountTimes * appleDiscount + 
                             orangeDiscountTimes * orangeDiscount + 
                             grapefruitDiscountTimes * grapefruitDiscount
  let bundleDiscountApplied := if peachCount ≥ bundleDiscountThreshold1 ∧ 
                                  appleCount ≥ bundleDiscountThreshold2 ∧ 
                                  orangeCount ≥ bundleDiscountThreshold3 
                               then bundleDiscount else 0
  totalBeforeDiscount - individualDiscounts - bundleDiscountApplied

theorem correct_total_amount_paid : 
  totalAmountPaid 0.4 400 10 2 0.6 150 15 3 0.5 200 7 1.5 1 80 20 4 100 50 100 10 = 333 := by
  sorry


end NUMINAMATH_CALUDE_correct_total_amount_paid_l1550_155070


namespace NUMINAMATH_CALUDE_problem_solution_l1550_155042

theorem problem_solution (x y : ℝ) (h : |x + 5| + (y - 4)^2 = 0) : (x + y)^99 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1550_155042


namespace NUMINAMATH_CALUDE_equation_solution_l1550_155025

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1550_155025


namespace NUMINAMATH_CALUDE_factor_calculation_l1550_155016

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 9 →
  factor * (2 * initial_number + 13) = 93 →
  factor = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l1550_155016


namespace NUMINAMATH_CALUDE_triangle_side_length_l1550_155092

/-- Given a triangle ABC where sin A, sin B, sin C form an arithmetic sequence,
    B = 30°, and the area is 3/2, prove that the length of side b is √3 + 1. -/
theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c →
  -- sin A, sin B, sin C form an arithmetic sequence
  2 * Real.sin B = Real.sin A + Real.sin C →
  -- B = 30°
  B = π / 6 →
  -- Area of triangle ABC is 3/2
  1/2 * a * c * Real.sin B = 3/2 →
  -- b is opposite to angle B
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  -- Conclusion: length of side b is √3 + 1
  b = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1550_155092


namespace NUMINAMATH_CALUDE_thomas_monthly_pay_l1550_155077

/-- The amount paid to a worker after one month, given their weekly rate and the number of weeks in a month -/
def monthly_pay (weekly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_rate * weeks_per_month

theorem thomas_monthly_pay :
  monthly_pay 4550 4 = 18200 := by
  sorry

end NUMINAMATH_CALUDE_thomas_monthly_pay_l1550_155077


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1550_155091

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : is_positive_geometric_sequence a)
  (h_2 : a 2 = 3)
  (h_8 : a 8 = 27) :
  a 5 = 9 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l1550_155091


namespace NUMINAMATH_CALUDE_hyperbola_parabola_property_l1550_155052

/-- Given a hyperbola and a parabola with specific properties, prove that 2e - b² = 4 -/
theorem hyperbola_parabola_property (a b : ℝ) (e : ℝ) :
  a > 0 →
  b > 0 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y^2 = 4*x) →  -- Common point exists
  (∃ (x₀ y₀ : ℝ), x₀^2 / a^2 - y₀^2 / b^2 = 1 ∧ y₀^2 = 4*x₀ ∧ x₀ + 1 = 2) →  -- Distance to directrix is 2
  e = Real.sqrt (1 + b^2 / a^2) →  -- Definition of hyperbola eccentricity
  2*e - b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_property_l1550_155052


namespace NUMINAMATH_CALUDE_inverse_sum_equals_root_difference_l1550_155015

-- Define the function g
def g (x : ℝ) : ℝ := x^3 * |x|

-- State the theorem
theorem inverse_sum_equals_root_difference :
  (∃ y₁ : ℝ, g y₁ = 8) ∧ (∃ y₂ : ℝ, g y₂ = -125) →
  (∃ y₁ y₂ : ℝ, g y₁ = 8 ∧ g y₂ = -125 ∧ y₁ + y₂ = 2^(1/2) - 5^(3/4)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_root_difference_l1550_155015


namespace NUMINAMATH_CALUDE_graph_transformation_l1550_155041

/-- Given a function f, prove that (1/3)f(x) + 2 is equivalent to scaling f(x) vertically by 1/3 and shifting up by 2 -/
theorem graph_transformation (f : ℝ → ℝ) (x : ℝ) :
  (1/3) * (f x) + 2 = ((1/3) * f x) + 2 := by sorry

end NUMINAMATH_CALUDE_graph_transformation_l1550_155041


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1550_155098

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : (total_players - throwers) % 3 = 0)
  (h4 : throwers ≤ total_players) :
  throwers + ((total_players - throwers) * 2 / 3) = 57 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l1550_155098


namespace NUMINAMATH_CALUDE_book_pages_count_l1550_155076

/-- Count the occurrences of digit 1 in a number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Count the occurrences of digit 1 in page numbers from 1 to n -/
def countOnesInPages (n : ℕ) : ℕ := sorry

/-- The number of pages in the book -/
def numPages : ℕ := 318

/-- The total count of digit 1 in the book's page numbers -/
def totalOnes : ℕ := 171

theorem book_pages_count :
  (countOnesInPages numPages = totalOnes) ∧ 
  (∀ m : ℕ, m < numPages → countOnesInPages m < totalOnes) := by sorry

end NUMINAMATH_CALUDE_book_pages_count_l1550_155076
