import Mathlib

namespace cost_of_300_pencils_l536_53610

/-- The cost of pencils in dollars -/
def cost_in_dollars (num_pencils : ℕ) (cost_per_pencil_cents : ℕ) (cents_per_dollar : ℕ) : ℚ :=
  (num_pencils * cost_per_pencil_cents : ℚ) / cents_per_dollar

/-- Theorem: The cost of 300 pencils is 7.5 dollars -/
theorem cost_of_300_pencils :
  cost_in_dollars 300 5 200 = 7.5 := by
  sorry

end cost_of_300_pencils_l536_53610


namespace parallel_vectors_solution_l536_53605

def vector_a : ℝ × ℝ := (2, -3)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x^2 - 5*x)

theorem parallel_vectors_solution :
  ∀ x : ℝ, (∃ k : ℝ, vector_a = k • vector_b x) → x = 2 ∨ x = 3 := by
  sorry

end parallel_vectors_solution_l536_53605


namespace cyclist_wait_time_correct_l536_53684

/-- The time (in minutes) the cyclist stops to wait after passing the hiker -/
def cyclist_wait_time : ℝ := 3.6667

/-- The hiker's speed in miles per hour -/
def hiker_speed : ℝ := 4

/-- The cyclist's speed in miles per hour -/
def cyclist_speed : ℝ := 15

/-- The time (in minutes) the cyclist waits for the hiker to catch up -/
def catch_up_time : ℝ := 13.75

theorem cyclist_wait_time_correct :
  cyclist_wait_time * (cyclist_speed / 60) = catch_up_time * (hiker_speed / 60) := by
  sorry

#check cyclist_wait_time_correct

end cyclist_wait_time_correct_l536_53684


namespace book_reading_fraction_l536_53618

theorem book_reading_fraction (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 60 →
  pages_read = (total_pages - pages_read) + 20 →
  (pages_read : ℚ) / total_pages = 2 / 3 := by
  sorry

end book_reading_fraction_l536_53618


namespace parallel_line_through_point_l536_53612

/-- A line is represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Convert an equation of the form ax + by = c to slope-intercept form -/
def to_slope_intercept (a b c : ℚ) : Line :=
  { slope := -a / b, intercept := c / b }

theorem parallel_line_through_point 
  (l1 : Line) (p : Point) :
  ∃ (l2 : Line), 
    parallel l1 l2 ∧ 
    l2.passes_through p ∧
    l2.slope = 1/2 ∧ 
    l2.intercept = -2 :=
  sorry

#check parallel_line_through_point

end parallel_line_through_point_l536_53612


namespace f_is_quadratic_l536_53622

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l536_53622


namespace masons_grandmother_age_l536_53679

theorem masons_grandmother_age (mason_age sydney_age father_age grandmother_age : ℕ) :
  mason_age = 20 →
  sydney_age = 3 * mason_age →
  father_age = sydney_age + 6 →
  grandmother_age = 2 * father_age →
  grandmother_age = 132 := by
sorry

end masons_grandmother_age_l536_53679


namespace max_missed_problems_l536_53644

/-- Given a test with 50 problems and a passing score of at least 85%,
    the maximum number of problems a student can miss and still pass is 7. -/
theorem max_missed_problems (total_problems : Nat) (passing_percentage : Rat) :
  total_problems = 50 →
  passing_percentage = 85 / 100 →
  (↑(total_problems - 7) : Rat) / total_problems ≥ passing_percentage ∧
  ∀ n : Nat, n > 7 → (↑(total_problems - n) : Rat) / total_problems < passing_percentage :=
by sorry

end max_missed_problems_l536_53644


namespace shaded_area_of_circle_with_rectangles_l536_53666

/-- The shaded area of a circle with two inscribed rectangles -/
theorem shaded_area_of_circle_with_rectangles :
  let rectangle_width : ℝ := 10
  let rectangle_length : ℝ := 24
  let overlap_side : ℝ := 10
  let circle_radius : ℝ := (rectangle_width ^ 2 + rectangle_length ^ 2).sqrt / 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let total_rectangle_area : ℝ := 2 * rectangle_area
  let overlap_area : ℝ := overlap_side ^ 2
  circle_area - total_rectangle_area + overlap_area = 169 * π - 380 := by
  sorry

end shaded_area_of_circle_with_rectangles_l536_53666


namespace sector_area_sexagesimal_l536_53636

/-- The area of a sector with radius 4 and central angle 625/6000 of a full circle is 5π/3 -/
theorem sector_area_sexagesimal (π : ℝ) (h : π > 0) : 
  let r : ℝ := 4
  let angle_fraction : ℝ := 625 / 6000
  let sector_area := (1/2) * (angle_fraction * 2 * π) * r^2
  sector_area = 5 * π / 3 := by
sorry

end sector_area_sexagesimal_l536_53636


namespace workshop_transfer_l536_53614

theorem workshop_transfer (w : ℕ) (n : ℕ) (x : ℕ) : 
  w ≥ 63 →
  w ≤ 64 →
  31 * w + n * (n + 1) / 2 = 1994 →
  (n = 4 ∧ x = 4) ∨ (n = 2 ∧ x = 21) :=
sorry

end workshop_transfer_l536_53614


namespace winnie_lollipop_distribution_l536_53615

/-- The number of lollipops left after equal distribution --/
def lollipops_left (cherry wintergreen grape shrimp_cocktail friends : ℕ) : ℕ :=
  (cherry + wintergreen + grape + shrimp_cocktail) % friends

theorem winnie_lollipop_distribution :
  lollipops_left 55 134 12 265 15 = 1 := by sorry

end winnie_lollipop_distribution_l536_53615


namespace evaluate_expression_l536_53662

theorem evaluate_expression : (24^36) / (72^18) = 8^18 := by
  sorry

end evaluate_expression_l536_53662


namespace event_arrangements_eq_60_l536_53651

/-- The number of ways to select 4 students from 5 for a three-day event --/
def event_arrangements (total_students : ℕ) (selected_students : ℕ) (days : ℕ) 
  (first_day_attendees : ℕ) : ℕ :=
  Nat.choose total_students first_day_attendees * 
  (Nat.factorial (total_students - first_day_attendees) / 
   Nat.factorial (total_students - selected_students))

/-- Proof that the number of arrangements for the given conditions is 60 --/
theorem event_arrangements_eq_60 : 
  event_arrangements 5 4 3 2 = 60 := by
  sorry

end event_arrangements_eq_60_l536_53651


namespace athlete_seating_arrangements_l536_53637

def number_of_arrangements (n : ℕ) (team_sizes : List ℕ) : ℕ :=
  (Nat.factorial n) * (team_sizes.map Nat.factorial).prod

theorem athlete_seating_arrangements :
  number_of_arrangements 4 [4, 3, 2, 3] = 20736 := by
  sorry

end athlete_seating_arrangements_l536_53637


namespace solution_set_reciprocal_inequality_l536_53629

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (1 / x > 3) ↔ (0 < x ∧ x < 1/3) :=
by sorry

end solution_set_reciprocal_inequality_l536_53629


namespace percent_to_decimal_three_percent_to_decimal_l536_53608

theorem percent_to_decimal (p : ℝ) : p / 100 = p * 0.01 := by sorry

theorem three_percent_to_decimal : (3 : ℝ) / 100 = 0.03 := by sorry

end percent_to_decimal_three_percent_to_decimal_l536_53608


namespace product_units_digit_base_6_l536_53673

/-- The units digit of a positive integer in base-6 is the remainder when the integer is divided by 6 -/
def units_digit_base_6 (n : ℕ) : ℕ := n % 6

/-- The product of the given numbers -/
def product : ℕ := 123 * 57 * 29

theorem product_units_digit_base_6 :
  units_digit_base_6 product = 3 := by sorry

end product_units_digit_base_6_l536_53673


namespace max_product_decomposition_l536_53609

theorem max_product_decomposition :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y = 100 → x * y ≤ 50 * 50 :=
by sorry

end max_product_decomposition_l536_53609


namespace hyperbola_sum_theorem_l536_53606

-- Define the hyperbola equation
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

-- Define the theorem
theorem hyperbola_sum_theorem (h k a b : ℝ) :
  -- Given conditions
  hyperbola_equation h k h k a b ∧
  (h = 3 ∧ k = -5) ∧
  (2 * a = 10) ∧
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ 2 * c = 14) →
  -- Conclusion
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  sorry

end hyperbola_sum_theorem_l536_53606


namespace proposition_truth_l536_53663

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - 2*x + 1 > 0

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

-- Theorem to prove
theorem proposition_truth : ¬p ∧ q := by sorry

end proposition_truth_l536_53663


namespace smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l536_53678

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 11 < y / 17 → y ≥ 13 := by
  sorry

theorem thirteen_satisfies : (8 : ℚ) / 11 < 13 / 17 := by
  sorry

theorem smallest_integer_is_thirteen : ∃ y : ℤ, ((8 : ℚ) / 11 < y / 17) ∧ (∀ z : ℤ, (8 : ℚ) / 11 < z / 17 → z ≥ y) ∧ y = 13 := by
  sorry

end smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l536_53678


namespace sidney_kittens_l536_53611

/-- The number of kittens Sidney has -/
def num_kittens : ℕ := sorry

/-- The number of adult cats Sidney has -/
def num_adult_cats : ℕ := 3

/-- The number of cans Sidney already has -/
def initial_cans : ℕ := 7

/-- The number of additional cans Sidney needs to buy -/
def additional_cans : ℕ := 35

/-- The number of days Sidney needs to feed the cats -/
def num_days : ℕ := 7

/-- The amount of food (in cans) an adult cat eats per day -/
def adult_cat_food_per_day : ℚ := 1

/-- The amount of food (in cans) a kitten eats per day -/
def kitten_food_per_day : ℚ := 3/4

theorem sidney_kittens : 
  num_kittens = 4 ∧
  (num_kittens : ℚ) * kitten_food_per_day * num_days + 
  (num_adult_cats : ℚ) * adult_cat_food_per_day * num_days = 
  initial_cans + additional_cans :=
sorry

end sidney_kittens_l536_53611


namespace discount_percentage_l536_53676

/-- Proves that given a cost price of 66.5, a marked price of 87.5, and a profit of 25% on the cost price, the percentage deducted from the list price is 5%. -/
theorem discount_percentage (cost_price : ℝ) (marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 66.5 →
  marked_price = 87.5 →
  profit_percentage = 25 →
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount_percentage := (marked_price - selling_price) / marked_price * 100
  discount_percentage = 5 := by
sorry

end discount_percentage_l536_53676


namespace ninas_pet_insect_eyes_l536_53686

/-- The number of eyes among Nina's pet insects -/
def total_eyes (num_spiders : ℕ) (spider_eyes : ℕ) (num_ants : ℕ) (ant_eyes : ℕ) : ℕ :=
  num_spiders * spider_eyes + num_ants * ant_eyes

/-- Theorem stating the total number of eyes among Nina's pet insects -/
theorem ninas_pet_insect_eyes :
  total_eyes 3 8 50 2 = 124 := by
  sorry

end ninas_pet_insect_eyes_l536_53686


namespace horner_method_a1_l536_53623

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 0.5x^5 + 4x^4 - 3x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-1, 1, 0, -3, 4, 0.5]

theorem horner_method_a1 : 
  let x := 3
  let result := horner_eval coeffs x
  result = f x ∧ result = 1 := by
  sorry

#eval horner_eval coeffs 3

end horner_method_a1_l536_53623


namespace rectangular_field_path_area_and_cost_l536_53683

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem rectangular_field_path_area_and_cost 
  (field_length field_width path_width cost_per_unit : ℝ)
  (h_length : field_length = 75)
  (h_width : field_width = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_unit : cost_per_unit = 2) :
  let area := path_area field_length field_width path_width
  let cost := construction_cost area cost_per_unit
  area = 675 ∧ cost = 1350 := by
  sorry

end rectangular_field_path_area_and_cost_l536_53683


namespace integer_solutions_yk_eq_x2_plus_x_l536_53650

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  (∃ (x y : ℤ), y^k = x^2 + x) ↔ (k = 2 ∧ (∃ x : ℤ, x = 0 ∨ x = -1)) :=
sorry

end integer_solutions_yk_eq_x2_plus_x_l536_53650


namespace a2_value_l536_53625

def sequence_sum (n : ℕ) (k : ℕ) : ℚ := -1/2 * n^2 + k*n

theorem a2_value (k : ℕ) (h1 : k > 0) 
  (h2 : ∃ (n : ℕ), ∀ (m : ℕ), sequence_sum m k ≤ sequence_sum n k)
  (h3 : ∃ (n : ℕ), sequence_sum n k = 8) :
  sequence_sum 2 k - sequence_sum 1 k = 5/2 := by
sorry

end a2_value_l536_53625


namespace bank_teller_bills_count_l536_53682

theorem bank_teller_bills_count :
  ∀ (num_20_dollar_bills : ℕ),
  (20 * 5 + num_20_dollar_bills * 20 = 780) →
  (20 + num_20_dollar_bills = 54) :=
by
  sorry

end bank_teller_bills_count_l536_53682


namespace grid_polygon_segment_sums_equal_l536_53659

-- Define a type for grid points
structure GridPoint where
  x : ℤ
  y : ℤ

-- Define a type for polygons on a grid
structure GridPolygon where
  vertices : List GridPoint
  convex : Bool
  verticesOnGrid : Bool
  sidesNotAligned : Bool

-- Define a function to calculate the sum of vertical segment lengths
def sumVerticalSegments (p : GridPolygon) : ℝ :=
  sorry

-- Define a function to calculate the sum of horizontal segment lengths
def sumHorizontalSegments (p : GridPolygon) : ℝ :=
  sorry

-- Theorem statement
theorem grid_polygon_segment_sums_equal (p : GridPolygon) :
  p.convex ∧ p.verticesOnGrid ∧ p.sidesNotAligned →
  sumVerticalSegments p = sumHorizontalSegments p :=
sorry

end grid_polygon_segment_sums_equal_l536_53659


namespace house_sale_profit_percentage_l536_53604

/-- Calculates the profit percentage for a house sale --/
theorem house_sale_profit_percentage
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (commission_rate : ℝ)
  (h1 : purchase_price = 80000)
  (h2 : selling_price = 100000)
  (h3 : commission_rate = 0.05)
  : (selling_price - commission_rate * purchase_price - purchase_price) / purchase_price = 0.2 := by
  sorry

end house_sale_profit_percentage_l536_53604


namespace additional_cars_needed_danica_car_arrangement_l536_53664

theorem additional_cars_needed (initial_cars : ℕ) (cars_per_row : ℕ) : ℕ :=
  cars_per_row - (initial_cars % cars_per_row) % cars_per_row

theorem danica_car_arrangement : additional_cars_needed 39 7 = 3 := by
  sorry

end additional_cars_needed_danica_car_arrangement_l536_53664


namespace boat_speed_in_still_water_l536_53698

/-- The speed of a boat in still water, given its downstream travel information. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 45)
  (h3 : downstream_time = 1)
  : ∃ (boat_speed : ℝ), boat_speed = 40 ∧ 
    downstream_distance = downstream_time * (boat_speed + stream_speed) :=
sorry

end boat_speed_in_still_water_l536_53698


namespace x_fourth_coefficient_is_20th_term_l536_53632

def binomial_sum (n : ℕ) : ℕ := (n.choose 4) + ((n + 1).choose 4) + ((n + 2).choose 4)

def arithmetic_sequence (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem x_fourth_coefficient_is_20th_term :
  ∃ n : ℕ, n = 5 ∧ 
  binomial_sum n = arithmetic_sequence (-2) 3 20 := by
  sorry

end x_fourth_coefficient_is_20th_term_l536_53632


namespace x_plus_y_values_l536_53602

theorem x_plus_y_values (x y : ℝ) (hx : x = y * (3 - y)^2) (hy : y = x * (3 - x)^2) :
  x + y ∈ ({0, 3, 4, 5, 8} : Set ℝ) := by
  sorry

end x_plus_y_values_l536_53602


namespace count_perfect_square_factors_of_5400_l536_53669

/-- The number of perfect square factors of 5400 -/
def perfect_square_factors_of_5400 : ℕ :=
  let n := 5400
  let prime_factorization := (2, 2) :: (3, 3) :: (5, 2) :: []
  (prime_factorization.map (fun (p, e) => (e / 2 + 1))).prod

/-- Theorem stating that the number of perfect square factors of 5400 is 8 -/
theorem count_perfect_square_factors_of_5400 :
  perfect_square_factors_of_5400 = 8 := by sorry

end count_perfect_square_factors_of_5400_l536_53669


namespace estimate_passing_papers_l536_53658

theorem estimate_passing_papers (total_papers : ℕ) (sample_size : ℕ) (passing_in_sample : ℕ) 
  (h1 : total_papers = 5000)
  (h2 : sample_size = 400)
  (h3 : passing_in_sample = 360) :
  ⌊(total_papers : ℝ) * (passing_in_sample : ℝ) / (sample_size : ℝ)⌋ = 4500 := by
sorry

end estimate_passing_papers_l536_53658


namespace matrix_transformation_proof_l536_53675

theorem matrix_transformation_proof : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (a b c d : ℝ),
    N * !![a, b; c, d] = !![3*a, b; 3*c, d] :=
by
  sorry

end matrix_transformation_proof_l536_53675


namespace last_four_digits_of_2_to_15000_l536_53693

theorem last_four_digits_of_2_to_15000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^15000 ≡ 1 [ZMOD 1250] := by
  sorry

end last_four_digits_of_2_to_15000_l536_53693


namespace class_average_problem_l536_53643

theorem class_average_problem (first_group_percentage : Real) 
                               (second_group_percentage : Real) 
                               (first_group_average : Real) 
                               (second_group_average : Real) 
                               (overall_average : Real) :
  first_group_percentage = 0.25 →
  second_group_percentage = 0.50 →
  first_group_average = 0.80 →
  second_group_average = 0.65 →
  overall_average = 0.75 →
  let remainder_percentage := 1 - first_group_percentage - second_group_percentage
  let remainder_average := (overall_average - first_group_percentage * first_group_average - 
                            second_group_percentage * second_group_average) / remainder_percentage
  remainder_average = 0.90 := by
sorry

end class_average_problem_l536_53643


namespace greatest_integer_with_gcd_condition_l536_53624

theorem greatest_integer_with_gcd_condition :
  ∃ n : ℕ, n < 200 ∧ n.gcd 30 = 10 ∧ ∀ m : ℕ, m < 200 → m.gcd 30 = 10 → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_integer_with_gcd_condition_l536_53624


namespace least_froods_for_more_points_l536_53694

/-- Sum of first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Points earned from eating n Froods -/
def eating_points (n : ℕ) : ℕ := 5 * n

/-- Proposition: 10 is the least positive integer n for which 
    dropping n Froods earns more points than eating them -/
theorem least_froods_for_more_points : 
  ∀ n : ℕ, n > 0 → (
    (n < 10 → sum_first_n n ≤ eating_points n) ∧
    (sum_first_n 10 > eating_points 10)
  ) := by sorry

end least_froods_for_more_points_l536_53694


namespace five_to_fifth_sum_five_times_l536_53628

theorem five_to_fifth_sum_five_times (n : ℕ) : 5^5 + 5^5 + 5^5 + 5^5 + 5^5 = 5^6 := by
  sorry

end five_to_fifth_sum_five_times_l536_53628


namespace chord_intersections_count_l536_53603

/-- The number of intersection points of chords on a circle -/
def chord_intersections (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of intersection points of chords drawn between n vertices 
    on a circle, excluding the vertices themselves, is equal to binom(n, 4), 
    given that no three chords are concurrent except at a vertex. -/
theorem chord_intersections_count (n : ℕ) (h : n ≥ 4) :
  chord_intersections n = Nat.choose n 4 := by
  sorry

end chord_intersections_count_l536_53603


namespace female_managers_count_l536_53690

theorem female_managers_count (total_employees : ℕ) (female_employees : ℕ) (total_managers : ℕ) (male_associates : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : male_associates = 160) :
  total_managers = total_employees - female_employees - male_associates :=
by
  sorry

end female_managers_count_l536_53690


namespace equal_roots_condition_l536_53601

/-- 
For a quadratic equation ax^2 + bx + c = 0, 
the discriminant is defined as b^2 - 4ac
-/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- 
A quadratic equation has two equal real roots 
if and only if its discriminant is zero
-/
axiom equal_roots_iff_zero_discriminant (a b c : ℝ) : 
  a ≠ 0 → (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ (∀ y : ℝ, a*y^2 + b*y + c = 0 → y = x)) ↔ 
    discriminant a b c = 0

/-- 
For the quadratic equation x^2 + 6x + m = 0 to have two equal real roots, 
m must equal 9
-/
theorem equal_roots_condition : 
  (∃ x : ℝ, x^2 + 6*x + m = 0 ∧ (∀ y : ℝ, y^2 + 6*y + m = 0 → y = x)) → m = 9 := by
sorry

end equal_roots_condition_l536_53601


namespace solution_pairs_l536_53668

theorem solution_pairs : 
  {p : ℕ × ℕ | let (m, n) := p; m^2 + 2 * 3^n = m * (2^(n+1) - 1)} = 
  {(9, 3), (6, 3), (9, 5), (54, 5)} :=
by sorry

end solution_pairs_l536_53668


namespace smallest_number_l536_53671

theorem smallest_number (a b c d : ℚ) 
  (ha : a = 0) 
  (hb : b = -3) 
  (hc : c = 1/3) 
  (hd : d = 1) : 
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end smallest_number_l536_53671


namespace average_difference_l536_53657

theorem average_difference (a c x : ℝ) 
  (h1 : (a + x) / 2 = 40)
  (h2 : (x + c) / 2 = 60) : 
  c - a = 40 := by
sorry

end average_difference_l536_53657


namespace apple_purchase_l536_53667

theorem apple_purchase (cecile_apples diane_apples : ℕ) : 
  diane_apples = cecile_apples + 20 →
  cecile_apples + diane_apples = 50 →
  cecile_apples = 15 := by
sorry

end apple_purchase_l536_53667


namespace decimal_to_fraction_l536_53600

theorem decimal_to_fraction : 
  (0.32 : ℚ) = 8 / 25 := by sorry

end decimal_to_fraction_l536_53600


namespace circle_construction_theorem_circle_line_construction_theorem_l536_53616

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define tangency between circles
def CircleTangent (c1 c2 : Circle) : Prop := sorry

-- Define tangency between a circle and a line
def CircleLineTangent (c : Circle) (l : Line) : Prop := sorry

-- Define a circle passing through a point
def CirclePassesThrough (c : Circle) (p : Point) : Prop := sorry

theorem circle_construction_theorem 
  (P : Point) 
  (S1 S2 : Circle) : 
  ∃ (C : Circle), 
    CirclePassesThrough C P ∧ 
    CircleTangent C S1 ∧ 
    CircleTangent C S2 := by sorry

theorem circle_line_construction_theorem 
  (P : Point) 
  (S : Circle) 
  (L : Line) : 
  ∃ (C : Circle), 
    CirclePassesThrough C P ∧ 
    CircleTangent C S ∧ 
    CircleLineTangent C L := by sorry

end circle_construction_theorem_circle_line_construction_theorem_l536_53616


namespace remainder_problem_l536_53613

theorem remainder_problem (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6)
  (hsum : (f + y) % 15 = 7) :
  (f + y + z) % 35 = 3 ∧ (f + y + z) % 105 = 3 := by
  sorry

end remainder_problem_l536_53613


namespace angle_B_measure_l536_53627

-- Define the hexagon PROBLEMS
structure Hexagon where
  P : ℝ
  R : ℝ
  O : ℝ
  B : ℝ
  L : ℝ
  S : ℝ

-- Define the conditions
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.P = h.R ∧ h.P = h.B ∧ 
  h.O + h.S = 180 ∧ 
  h.L = 90 ∧
  h.P + h.R + h.O + h.B + h.L + h.S = 720

-- State the theorem
theorem angle_B_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.B = 150 := by
  sorry

end angle_B_measure_l536_53627


namespace find_t_l536_53652

-- Define vectors in R²
def AB : Fin 2 → ℝ := ![2, 3]
def AC : ℝ → Fin 2 → ℝ := λ t => ![3, t]

-- Define the dot product of two vectors in R²
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the perpendicular condition
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w = 0

-- Theorem statement
theorem find_t : ∃ t : ℝ, 
  perpendicular AB (λ i => AC t i - AB i) ∧ t = 7/3 := by
  sorry

end find_t_l536_53652


namespace system_of_inequalities_l536_53687

theorem system_of_inequalities (x : ℝ) : 
  (-3 * x^2 + 7 * x + 6 > 0 ∧ 4 * x - 4 * x^2 > -3) ↔ (-1/2 < x ∧ x < 3/2) := by
sorry

end system_of_inequalities_l536_53687


namespace division_problem_l536_53621

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 12 := by
sorry

end division_problem_l536_53621


namespace odometer_puzzle_l536_53685

theorem odometer_puzzle (a b c d : ℕ) : 
  a ≥ 1 →
  a + b + c + d ≤ 10 →
  (1000 * d + 100 * c + 10 * b + a) - (1000 * a + 100 * b + 10 * c + d) % 60 = 0 →
  a^2 + b^2 + c^2 + d^2 = 83 := by
  sorry

end odometer_puzzle_l536_53685


namespace diamond_value_l536_53672

/-- Given that ◇3 in base 5 equals ◇2 in base 6, where ◇ is a digit, prove that ◇ = 1 -/
theorem diamond_value (diamond : ℕ) (h1 : diamond < 10) :
  5 * diamond + 3 = 6 * diamond + 2 → diamond = 1 := by
  sorry

end diamond_value_l536_53672


namespace triangle_ratio_theorem_l536_53660

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two sides
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def pointOnSegment (p1 p2 : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := sorry

-- Define the intersection of two lines
def lineIntersection (l1p1 l1p2 l2p1 l2p2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the median of a triangle
def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem triangle_ratio_theorem (t : Triangle) :
  area t = 2 * Real.sqrt 3 →
  sideLength t.B t.C = 1 →
  angle t.B t.C t.A = π / 3 →
  let D := pointOnSegment t.A t.B 3
  let E := median t t.C
  let M := lineIntersection t.C D t.B E
  ∃ (r : ℝ), r = 3 / 5 ∧ sideLength t.B M = r * sideLength M E :=
by sorry

end triangle_ratio_theorem_l536_53660


namespace mike_picked_limes_l536_53647

/-- The number of limes Alyssa ate -/
def limes_eaten : ℝ := 25.0

/-- The number of limes left -/
def limes_left : ℕ := 7

/-- The number of limes Mike picked -/
def mikes_limes : ℝ := limes_eaten + limes_left

theorem mike_picked_limes : mikes_limes = 32 := by
  sorry

end mike_picked_limes_l536_53647


namespace fraction_equality_solution_l536_53674

theorem fraction_equality_solution :
  let f₁ (x : ℝ) := (5 + 2*x) / (7 + 3*x)
  let f₂ (x : ℝ) := (4 + 3*x) / (9 + 4*x)
  let x₁ := (-5 + Real.sqrt 93) / 2
  let x₂ := (-5 - Real.sqrt 93) / 2
  (f₁ x₁ = f₂ x₁) ∧ (f₁ x₂ = f₂ x₂) :=
by sorry

end fraction_equality_solution_l536_53674


namespace total_beads_needed_l536_53633

/-- The number of green beads in one pattern repeat -/
def green_beads : ℕ := 3

/-- The number of purple beads in one pattern repeat -/
def purple_beads : ℕ := 5

/-- The number of red beads in one pattern repeat -/
def red_beads : ℕ := 2 * green_beads

/-- The total number of beads in one pattern repeat -/
def beads_per_repeat : ℕ := green_beads + purple_beads + red_beads

/-- The number of pattern repeats in one bracelet -/
def repeats_per_bracelet : ℕ := 3

/-- The number of pattern repeats in one necklace -/
def repeats_per_necklace : ℕ := 5

/-- The number of bracelets to make -/
def num_bracelets : ℕ := 1

/-- The number of necklaces to make -/
def num_necklaces : ℕ := 10

theorem total_beads_needed : 
  beads_per_repeat * (repeats_per_bracelet * num_bracelets + 
  repeats_per_necklace * num_necklaces) = 742 := by
  sorry

end total_beads_needed_l536_53633


namespace defective_shipped_percentage_l536_53661

theorem defective_shipped_percentage 
  (total_units : ℕ) 
  (defective_percentage : ℝ) 
  (shipped_percentage : ℝ) 
  (h1 : defective_percentage = 7) 
  (h2 : shipped_percentage = 5) : 
  (defective_percentage / 100) * (shipped_percentage / 100) * 100 = 0.35 := by
  sorry

end defective_shipped_percentage_l536_53661


namespace smallest_k_with_remainder_one_l536_53638

theorem smallest_k_with_remainder_one (k : ℕ) : k = 103 ↔ 
  (k > 1) ∧ 
  (∀ n ∈ ({17, 6, 2} : Set ℕ), k % n = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ n ∈ ({17, 6, 2} : Set ℕ), m % n = 1) → m ≥ k) :=
by sorry

end smallest_k_with_remainder_one_l536_53638


namespace cube_plus_minus_one_divisible_by_seven_l536_53639

theorem cube_plus_minus_one_divisible_by_seven (a : ℤ) (h : ¬ 7 ∣ a) :
  7 ∣ (a^3 + 1) ∨ 7 ∣ (a^3 - 1) :=
by sorry

end cube_plus_minus_one_divisible_by_seven_l536_53639


namespace sausage_problem_l536_53689

/-- Calculates the total pounds of spicy meat mix used to make sausages -/
def total_meat_mix (initial_links : ℕ) (eaten_links : ℕ) (remaining_ounces : ℕ) : ℚ :=
  let remaining_links := initial_links - eaten_links
  let ounces_per_link := remaining_ounces / remaining_links
  let total_ounces := initial_links * ounces_per_link
  total_ounces / 16

/-- Theorem stating that given the conditions, the total meat mix used was 10 pounds -/
theorem sausage_problem (initial_links : ℕ) (eaten_links : ℕ) (remaining_ounces : ℕ) 
  (h1 : initial_links = 40)
  (h2 : eaten_links = 12)
  (h3 : remaining_ounces = 112) :
  total_meat_mix initial_links eaten_links remaining_ounces = 10 := by
  sorry

#eval total_meat_mix 40 12 112

end sausage_problem_l536_53689


namespace expression_equals_one_l536_53640

theorem expression_equals_one 
  (m n k : ℝ) 
  (h : m = 1 / (n * k)) : 
  1 / (1 + m + m * n) + 1 / (1 + n + n * k) + 1 / (1 + k + k * m) = 1 := by
  sorry

end expression_equals_one_l536_53640


namespace lowest_temperature_record_l536_53697

/-- The lowest temperature ever recorded in the world -/
def lowest_temperature : ℝ := -89.2

/-- The location where the lowest temperature was recorded -/
def record_location : String := "Vostok Station, Antarctica"

/-- How the temperature is written -/
def temperature_written : String := "-89.2 °C"

/-- How the temperature is read -/
def temperature_read : String := "negative eighty-nine point two degrees Celsius"

/-- Theorem stating the lowest recorded temperature and its representation -/
theorem lowest_temperature_record :
  lowest_temperature = -89.2 ∧
  record_location = "Vostok Station, Antarctica" ∧
  temperature_written = "-89.2 °C" ∧
  temperature_read = "negative eighty-nine point two degrees Celsius" :=
by sorry

end lowest_temperature_record_l536_53697


namespace tub_fill_time_is_24_minutes_l536_53631

/-- Represents the tub filling problem -/
structure TubFilling where
  capacity : ℕ             -- Tub capacity in liters
  flow_rate : ℕ            -- Tap flow rate in liters per minute
  leak_rate : ℕ            -- Leak rate in liters per minute
  cycle_time : ℕ           -- Time for one on-off cycle in minutes

/-- Calculates the time needed to fill the tub -/
def fill_time (tf : TubFilling) : ℕ :=
  let net_gain_per_cycle := tf.flow_rate - tf.leak_rate * tf.cycle_time
  (tf.capacity + net_gain_per_cycle - 1) / net_gain_per_cycle * tf.cycle_time

/-- Theorem stating that the time to fill the tub is 24 minutes -/
theorem tub_fill_time_is_24_minutes :
  let tf : TubFilling := {
    capacity := 120,
    flow_rate := 12,
    leak_rate := 1,
    cycle_time := 2
  }
  fill_time tf = 24 := by sorry

end tub_fill_time_is_24_minutes_l536_53631


namespace smallest_three_digit_with_digit_product_24_l536_53695

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem smallest_three_digit_with_digit_product_24 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 24 → 146 ≤ n :=
sorry

end smallest_three_digit_with_digit_product_24_l536_53695


namespace gcd_of_three_numbers_l536_53655

theorem gcd_of_three_numbers : Nat.gcd 13680 (Nat.gcd 20400 47600) = 80 := by
  sorry

end gcd_of_three_numbers_l536_53655


namespace coprime_power_sum_not_divisible_by_11_l536_53619

theorem coprime_power_sum_not_divisible_by_11 (a b : ℤ) (h : Int.gcd a b = 1) :
  ¬(11 ∣ (a^5 + 2*b^5)) ∧ ¬(11 ∣ (a^5 - 2*b^5)) := by
  sorry

end coprime_power_sum_not_divisible_by_11_l536_53619


namespace fifth_term_of_sequence_l536_53630

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9 * y
  | 2 => 27 * y^2
  | 3 => 81 * y^3
  | n + 4 => geometric_sequence y n * 3 * y

/-- The fifth term of the geometric sequence is 243y^4 -/
theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 243 * y^4 := by
  sorry

end fifth_term_of_sequence_l536_53630


namespace quadratic_propositions_l536_53617

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_propositions (a b c : ℝ) (ha : a ≠ 0) :
  -- Proposition 1
  (a + b + c = 0 → discriminant a b c ≥ 0) ∧
  -- Proposition 2
  (∃ x y : ℝ, x = -1 ∧ y = 2 ∧ quadratic a b c x ∧ quadratic a b c y → 2*a + c = 0) ∧
  -- Proposition 3
  ((∃ x y : ℝ, x ≠ y ∧ quadratic a 0 c x ∧ quadratic a 0 c y) →
   ∃ u v : ℝ, u ≠ v ∧ quadratic a b c u ∧ quadratic a b c v) :=
by sorry

end quadratic_propositions_l536_53617


namespace isosceles_triangle_perimeter_l536_53646

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 → 
  (a = b) →  -- isosceles condition
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- triangle inequality
  a + b + c = 12 := by
  sorry

end isosceles_triangle_perimeter_l536_53646


namespace square_root_division_l536_53688

theorem square_root_division (x : ℝ) : (Real.sqrt 1936) / x = 4 → x = 11 := by
  sorry

end square_root_division_l536_53688


namespace inequality_proof_l536_53692

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2)) + (y / (x^2 + y^4)) ≤ 1 / (x * y) := by
  sorry

end inequality_proof_l536_53692


namespace james_toy_ratio_l536_53653

/-- Given that James buys toy soldiers and toy cars, with 20 toy cars and a total of 60 toys,
    prove that the ratio of toy soldiers to toy cars is 2:1. -/
theorem james_toy_ratio :
  let total_toys : ℕ := 60
  let toy_cars : ℕ := 20
  let toy_soldiers : ℕ := total_toys - toy_cars
  (toy_soldiers : ℚ) / toy_cars = 2 / 1 :=
by sorry

end james_toy_ratio_l536_53653


namespace jelly_bean_ratio_l536_53691

theorem jelly_bean_ratio : 
  let napoleon_beans : ℕ := 17
  let sedrich_beans : ℕ := napoleon_beans + 4
  let mikey_beans : ℕ := 19
  let total_beans : ℕ := napoleon_beans + sedrich_beans
  2 * total_beans = 4 * mikey_beans := by sorry

end jelly_bean_ratio_l536_53691


namespace problem_statement_l536_53645

theorem problem_statement (a b : ℝ) (h : 4 * a^2 - a * b + b^2 = 1) :
  (abs a ≤ 2 * Real.sqrt 15 / 15) ∧
  (4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3) ∧
  (abs (2 * a - b) ≤ 2 * Real.sqrt 10 / 5) := by
sorry

end problem_statement_l536_53645


namespace camping_trip_percentage_l536_53641

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : total_students > 0)
  (students_more_than_100 : ℕ) 
  (h2 : students_more_than_100 = (18 * total_students) / 100)
  (h3 : (75 * (students_more_than_100 * 100 / 18)) / 100 + students_more_than_100 = 
        (72 * total_students) / 100) :
  (72 * total_students) / 100 = total_students - 
    ((75 * (students_more_than_100 * 100 / 18)) / 100 + students_more_than_100) :=
by sorry

end camping_trip_percentage_l536_53641


namespace special_triangle_area_l536_53642

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- The height of the triangle -/
  height : ℝ
  /-- The smaller part of the base -/
  small_base : ℝ
  /-- The ratio of the divided angle -/
  angle_ratio : ℝ
  /-- The height divides the angle in the given ratio -/
  height_divides_angle : angle_ratio = 2
  /-- The height is 2 cm -/
  height_is_two : height = 2
  /-- The smaller part of the base is 1 cm -/
  small_base_is_one : small_base = 1

/-- The theorem stating the area of the special triangle -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1 / 2 : ℝ) * t.height * (t.small_base + 5 / 3) = 8 / 3 := by
  sorry

end special_triangle_area_l536_53642


namespace different_color_number_probability_l536_53654

/-- Represents the total number of balls -/
def total_balls : ℕ := 9

/-- Represents the number of balls to be drawn -/
def drawn_balls : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of balls per color -/
def balls_per_color : ℕ := 3

/-- Represents the number of possible numbers on each ball -/
def num_numbers : ℕ := 3

/-- The probability of drawing 3 balls with different colors and numbers -/
def probability_different : ℚ := 1 / 14

theorem different_color_number_probability :
  (Nat.factorial num_colors) / (Nat.choose total_balls drawn_balls) = probability_different :=
sorry

end different_color_number_probability_l536_53654


namespace systematic_sampling_proof_l536_53607

theorem systematic_sampling_proof (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 883) (h2 : sample_size = 80) :
  ∃ (sampling_interval : ℕ) (n : ℕ),
    sampling_interval = 11 ∧ 
    n = 3 ∧ 
    total_students = sample_size * sampling_interval + n :=
by sorry

end systematic_sampling_proof_l536_53607


namespace sum_distances_constant_l536_53656

/-- A regular tetrahedron in 3D space -/
structure RegularTetrahedron where
  -- Add necessary fields for a regular tetrahedron

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance from a point to a plane in 3D space -/
def distanceToPlane (p : Point3D) (plane : Set Point3D) : ℝ :=
  sorry

/-- Predicate to check if a point is inside a regular tetrahedron -/
def isInside (p : Point3D) (t : RegularTetrahedron) : Prop :=
  sorry

/-- The faces of a regular tetrahedron -/
def faces (t : RegularTetrahedron) : Finset (Set Point3D) :=
  sorry

/-- Theorem: The sum of distances from any point inside a regular tetrahedron to all its faces is constant -/
theorem sum_distances_constant (t : RegularTetrahedron) :
  ∃ c : ℝ, ∀ p : Point3D, isInside p t →
    (faces t).sum (λ face => distanceToPlane p face) = c :=
  sorry

end sum_distances_constant_l536_53656


namespace area_increase_when_perimeter_increased_l536_53680

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the set of possible area increases. -/
def possibleAreaIncreases : Set ℕ := {2, 4, 21, 36, 38}

/-- Theorem stating the possible area increases when the perimeter is increased by 4 cm. -/
theorem area_increase_when_perimeter_increased
  (r : Rectangle)
  (h_perimeter : perimeter r = 40)
  (h_area : area r ≤ 40)
  (r_new : Rectangle)
  (h_new_perimeter : perimeter r_new = 44)
  : (area r_new - area r) ∈ possibleAreaIncreases := by
  sorry

end area_increase_when_perimeter_increased_l536_53680


namespace inequality_proof_l536_53696

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_3 : a + b + c + d = 3) :
  1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a*b*c*d)^3 := by
  sorry

end inequality_proof_l536_53696


namespace decimal_sum_to_fraction_l536_53699

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.03 + 0.004 + 0.0006 + 0.00007 = 13467 / 100000 := by
  sorry

end decimal_sum_to_fraction_l536_53699


namespace eighteen_binary_l536_53670

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem eighteen_binary : decimal_to_binary 18 = [1, 0, 0, 1, 0] := by
  sorry

end eighteen_binary_l536_53670


namespace min_value_theorem_l536_53626

/-- Given a function y = a^x + b where b > 0 and the graph passes through point (1,3),
    the minimum value of 4/(a-1) + 1/b is 9/2 -/
theorem min_value_theorem (a b : ℝ) (h1 : b > 0) (h2 : a^1 + b = 3) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x^1 + y = 3 → 4/(x-1) + 1/y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 1 ∧ y > 0 ∧ x^1 + y = 3 ∧ 4/(x-1) + 1/y = 9/2) := by
  sorry

end min_value_theorem_l536_53626


namespace parabola_properties_l536_53635

-- Define the parabola and its properties
def parabola (a b c m n t x₀ : ℝ) : Prop :=
  a > 0 ∧
  m = a + b + c ∧
  n = 16*a + 4*b + c ∧
  t = -b / (2*a) ∧
  3*a + b = 0 ∧
  m < c ∧ c < n ∧
  x₀ ≠ 1 ∧
  m = a * x₀^2 + b * x₀ + c

-- State the theorem
theorem parabola_properties (a b c m n t x₀ : ℝ) 
  (h : parabola a b c m n t x₀) : 
  m < n ∧ 1/2 < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 := by
  sorry

end parabola_properties_l536_53635


namespace sum_of_four_consecutive_integers_divisible_by_two_l536_53649

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k ∧ Nat.Prime 2 := by
  sorry

end sum_of_four_consecutive_integers_divisible_by_two_l536_53649


namespace system_solution_l536_53681

theorem system_solution : ∃ (x y : ℝ), 
  (x^2 - 6 * Real.sqrt (3 - 2*x) - y + 11 = 0) ∧ 
  (y^2 - 4 * Real.sqrt (3*y - 2) + 4*x + 16 = 0) ∧
  (x = -3) ∧ (y = 2) := by
  sorry

end system_solution_l536_53681


namespace base7_multiplication_addition_l536_53634

/-- Converts a base 7 number represented as a list of digits to a natural number -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 7 * acc + d) 0

/-- Converts a natural number to its base 7 representation as a list of digits -/
def natToBase7 (n : Nat) : List Nat :=
  if n < 7 then [n]
  else (n % 7) :: natToBase7 (n / 7)

theorem base7_multiplication_addition :
  (base7ToNat [5, 2]) * (base7ToNat [3]) + (base7ToNat [4, 4, 1]) =
  base7ToNat [3, 0, 3] := by
  sorry

end base7_multiplication_addition_l536_53634


namespace common_chord_equation_l536_53620

/-- The equation of the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := by
  sorry

end common_chord_equation_l536_53620


namespace option_a_false_option_b_true_option_c_false_option_d_true_l536_53648

-- Option A
theorem option_a_false (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ¬(1 / a > 1 / b) := by
sorry

-- Option B
theorem option_b_true (a b : ℝ) (h : a < b) (h2 : b < 0) : 
  a^2 > a * b := by
sorry

-- Option C
theorem option_c_false : 
  ∃ a b : ℝ, a > b ∧ ¬(|a| > |b|) := by
sorry

-- Option D
theorem option_d_true (a : ℝ) (h : a > 2) : 
  a + 4 / (a - 2) ≥ 6 := by
sorry

-- Final answer
def correct_options : List Char := ['B', 'D']

end option_a_false_option_b_true_option_c_false_option_d_true_l536_53648


namespace no_common_solution_l536_53677

theorem no_common_solution :
  ¬ ∃ x : ℝ, (8 * x^2 + 6 * x = 5) ∧ (3 * x + 2 = 0) := by
  sorry

end no_common_solution_l536_53677


namespace wall_bricks_count_l536_53665

theorem wall_bricks_count :
  -- Define the variables
  -- x: total number of bricks in the wall
  -- r1: rate of first bricklayer (bricks per hour)
  -- r2: rate of second bricklayer (bricks per hour)
  -- rc: combined rate after reduction (bricks per hour)
  ∀ (x r1 r2 rc : ℚ),
  -- Conditions
  (r1 = x / 7) →  -- First bricklayer's rate
  (r2 = x / 11) →  -- Second bricklayer's rate
  (rc = r1 + r2 - 12) →  -- Combined rate after reduction
  (6 * rc = x) →  -- Time to complete the wall after planning
  -- Conclusion
  x = 179 := by
sorry

end wall_bricks_count_l536_53665
