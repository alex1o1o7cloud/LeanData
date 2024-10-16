import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l2368_236863

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2

def point : ℝ × ℝ := (1, 2)

def tangent_slope : ℝ := 4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 9 = 0

theorem perpendicular_line_proof :
  perpendicular_line point.1 point.2 ∧
  (∃ k : ℝ, k * tangent_slope = -1 ∧
    ∀ x y : ℝ, perpendicular_line x y ↔ y - point.2 = k * (x - point.1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l2368_236863


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l2368_236861

-- Define the circle and square
def circle_diameter : ℝ := 6
def angle_ABE : ℝ := 45

-- Define π as a constant (since we're using an approximation)
def π : ℝ := 3.14

-- Define the theorem
theorem circle_square_area_difference :
  let r : ℝ := circle_diameter / 2
  let square_side : ℝ := r * Real.sqrt 2
  let circle_area : ℝ := π * r^2
  let square_area : ℝ := square_side^2
  abs (circle_area - square_area - 10.26) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_circle_square_area_difference_l2368_236861


namespace NUMINAMATH_CALUDE_incorrect_height_proof_l2368_236817

/-- Given a class of boys with an incorrect average height and one boy's height
    recorded incorrectly, prove the value of the incorrectly recorded height. -/
theorem incorrect_height_proof (n : ℕ) (incorrect_avg real_avg actual_height : ℝ) 
    (hn : n = 35)
    (hi : incorrect_avg = 182)
    (hr : real_avg = 180)
    (ha : actual_height = 106) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = 176 ∧
    n * real_avg = (n - 1) * incorrect_avg + actual_height - incorrect_height :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_proof_l2368_236817


namespace NUMINAMATH_CALUDE_max_value_of_a_min_value_of_expression_l2368_236888

-- Problem I
theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = |x - 5/2| + |x - a|)
  (h2 : ∀ x, f x ≥ a) :
  a ≤ 5/4 ∧ ∃ x, f x = 5/4 := by
sorry

-- Problem II
theorem min_value_of_expression (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 2*y + 3*z = 1) :
  3/x + 2/y + 1/z ≥ 16 + 8*Real.sqrt 3 ∧ 
  ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x + 2*y + 3*z = 1 ∧ 
    3/x + 2/y + 1/z = 16 + 8*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_a_min_value_of_expression_l2368_236888


namespace NUMINAMATH_CALUDE_cards_traded_count_l2368_236864

/-- The total number of cards traded between Padma and Robert -/
def total_cards_traded (padma_initial : ℕ) (robert_initial : ℕ) 
  (padma_first_trade : ℕ) (robert_first_trade : ℕ)
  (robert_second_trade : ℕ) (padma_second_trade : ℕ) : ℕ :=
  (padma_first_trade + robert_first_trade) + (robert_second_trade + padma_second_trade)

/-- Theorem stating the total number of cards traded between Padma and Robert -/
theorem cards_traded_count :
  total_cards_traded 75 88 2 10 8 15 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cards_traded_count_l2368_236864


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l2368_236860

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) :
  num_clients = 9 →
  cars_per_client = 4 →
  selections_per_car = 3 →
  num_clients * cars_per_client = selections_per_car * (num_clients * cars_per_client / selections_per_car) :=
by sorry

end NUMINAMATH_CALUDE_used_car_seller_problem_l2368_236860


namespace NUMINAMATH_CALUDE_shirt_price_shopping_scenario_l2368_236818

/-- The price of a shirt given the shopping scenario --/
theorem shirt_price (total_paid : ℝ) (num_shorts : ℕ) (price_per_short : ℝ) 
  (num_shirts : ℕ) (senior_discount : ℝ) : ℝ :=
  let shorts_cost := num_shorts * price_per_short
  let discounted_shorts_cost := shorts_cost * (1 - senior_discount)
  let shirts_cost := total_paid - discounted_shorts_cost
  shirts_cost / num_shirts

/-- The price of each shirt in the given shopping scenario is $15.30 --/
theorem shopping_scenario : 
  shirt_price 117 3 15 5 0.1 = 15.3 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_shopping_scenario_l2368_236818


namespace NUMINAMATH_CALUDE_junior_score_l2368_236872

theorem junior_score (n : ℝ) (h_n_pos : n > 0) : 
  let junior_percent : ℝ := 0.2
  let senior_percent : ℝ := 0.8
  let overall_avg : ℝ := 86
  let senior_avg : ℝ := 85
  let junior_count : ℝ := junior_percent * n
  let senior_count : ℝ := senior_percent * n
  let total_score : ℝ := overall_avg * n
  let senior_total_score : ℝ := senior_avg * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_l2368_236872


namespace NUMINAMATH_CALUDE_equation_holds_l2368_236847

theorem equation_holds (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c ↔ b + c = 12 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_l2368_236847


namespace NUMINAMATH_CALUDE_defective_units_percentage_l2368_236871

/-- The percentage of defective units that are shipped for sale -/
def defective_shipped_percent : ℝ := 5

/-- The percentage of all units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.4

/-- The percentage of all units that are defective -/
def defective_percent : ℝ := 8

theorem defective_units_percentage :
  defective_shipped_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l2368_236871


namespace NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l2368_236894

/-- Represents the percentage of green ducks in the larger pond -/
def larger_pond_green_percentage : ℝ := 12

theorem green_ducks_percentage_in_larger_pond :
  let smaller_pond_ducks : ℕ := 30
  let larger_pond_ducks : ℕ := 50
  let smaller_pond_green_percentage : ℝ := 20
  let total_green_percentage : ℝ := 15
  (smaller_pond_green_percentage / 100 * smaller_pond_ducks +
   larger_pond_green_percentage / 100 * larger_pond_ducks) /
  (smaller_pond_ducks + larger_pond_ducks) * 100 = total_green_percentage :=
by sorry

end NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l2368_236894


namespace NUMINAMATH_CALUDE_intersection_theorem_l2368_236819

-- Define the sets A and B
def A : Set ℝ := {x | x < -3 ∨ x > 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the expected result
def expected_result : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- State the theorem
theorem intersection_theorem : A_intersect_B = expected_result := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2368_236819


namespace NUMINAMATH_CALUDE_star_equation_solution_l2368_236889

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b + b - a - 1

/-- Theorem: If 3 star x = 20, then x = 6 -/
theorem star_equation_solution :
  (∃ x : ℝ, star 3 x = 20) → (∃ x : ℝ, star 3 x = 20 ∧ x = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2368_236889


namespace NUMINAMATH_CALUDE_silver_cube_gold_coating_value_l2368_236856

/-- Calculate the combined value of a silver cube with gold coating and markup -/
theorem silver_cube_gold_coating_value
  (cube_side : ℝ)
  (silver_density : ℝ)
  (silver_price : ℝ)
  (gold_coating_coverage : ℝ)
  (gold_coating_weight : ℝ)
  (gold_price : ℝ)
  (markup : ℝ)
  (h_cube_side : cube_side = 3)
  (h_silver_density : silver_density = 6)
  (h_silver_price : silver_price = 25)
  (h_gold_coating_coverage : gold_coating_coverage = 1/2)
  (h_gold_coating_weight : gold_coating_weight = 0.1)
  (h_gold_price : gold_price = 1800)
  (h_markup : markup = 1.1)
  : ∃ (total_value : ℝ), total_value = 18711 :=
by
  sorry

end NUMINAMATH_CALUDE_silver_cube_gold_coating_value_l2368_236856


namespace NUMINAMATH_CALUDE_dataset_reduction_fraction_l2368_236895

theorem dataset_reduction_fraction (initial : ℕ) (increase_percent : ℚ) (final : ℕ) : 
  initial = 200 →
  increase_percent = 1/5 →
  final = 180 →
  (initial + initial * increase_percent - final) / (initial + initial * increase_percent) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_dataset_reduction_fraction_l2368_236895


namespace NUMINAMATH_CALUDE_product_xy_equals_25_l2368_236820

theorem product_xy_equals_25 (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(5*y) = 1024) :
  x * y = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_25_l2368_236820


namespace NUMINAMATH_CALUDE_time_is_point_eight_hours_l2368_236869

/-- The number of unique letters in the name --/
def name_length : ℕ := 6

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 15

/-- Calculates the time in hours required to write all possible rearrangements of a name --/
def time_to_write_all_rearrangements : ℚ :=
  (Nat.factorial name_length : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Theorem stating that the time to write all rearrangements is 0.8 hours --/
theorem time_is_point_eight_hours :
  time_to_write_all_rearrangements = 4/5 := by sorry

end NUMINAMATH_CALUDE_time_is_point_eight_hours_l2368_236869


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2368_236825

theorem min_value_of_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 3 * y + 5 * z ≥ 14 * (40 / 3) ^ (1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2368_236825


namespace NUMINAMATH_CALUDE_star_value_l2368_236841

theorem star_value : ∃ (x : ℚ), 
  45 - ((28 * 3) - (37 - (15 / (x - 2)))) = 57 ∧ x = 103/59 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l2368_236841


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l2368_236803

def sum_of_divisors (i j k : ℕ) : ℕ :=
  (2^(i+1) - 1) * (3^(j+1) - 1) * (5^(k+1) - 1) / ((2-1) * (3-1) * (5-1))

theorem divisor_sum_theorem (i j k : ℕ) :
  sum_of_divisors i j k = 1200 → i + j + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l2368_236803


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2368_236882

/-- An isosceles triangle with two sides of lengths 3 and 4 has a perimeter of either 10 or 11. -/
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 3 ∧ c = 4 ∨ a = 4 ∧ c = 3)) ∨ (a = c ∧ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) ∨ (b = c ∧ (b = 3 ∧ a = 4 ∨ b = 4 ∧ a = 3)) →
  a + b + c = 10 ∨ a + b + c = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2368_236882


namespace NUMINAMATH_CALUDE_narrowest_strip_for_specific_figure_l2368_236839

/-- Represents a plane figure composed of an equilateral triangle and circular arcs --/
structure TriangleWithArcs where
  side_length : ℝ
  small_radius : ℝ
  large_radius : ℝ

/-- Calculates the narrowest strip width for a given TriangleWithArcs --/
def narrowest_strip_width (figure : TriangleWithArcs) : ℝ :=
  figure.small_radius + figure.large_radius

/-- Theorem stating that for the specific figure described, the narrowest strip width is 6 units --/
theorem narrowest_strip_for_specific_figure :
  let figure : TriangleWithArcs := {
    side_length := 4,
    small_radius := 1,
    large_radius := 5
  }
  narrowest_strip_width figure = 6 := by
  sorry

end NUMINAMATH_CALUDE_narrowest_strip_for_specific_figure_l2368_236839


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2368_236898

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2368_236898


namespace NUMINAMATH_CALUDE_inequality_proof_l2368_236854

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2368_236854


namespace NUMINAMATH_CALUDE_jason_read_all_books_l2368_236826

theorem jason_read_all_books 
  (jason_books : ℕ) 
  (mary_books : ℕ) 
  (total_books : ℕ) 
  (h1 : jason_books = 18) 
  (h2 : mary_books = 42) 
  (h3 : total_books = 60) 
  (h4 : jason_books + mary_books = total_books) : 
  jason_books = 18 := by
sorry

end NUMINAMATH_CALUDE_jason_read_all_books_l2368_236826


namespace NUMINAMATH_CALUDE_stream_speed_l2368_236884

theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) :
  downstream_distance = 250 →
  downstream_time = 7 →
  upstream_distance = 150 →
  upstream_time = 21 →
  ∃ s : ℝ, abs (s - 14.28) < 0.01 ∧ 
  (∃ b : ℝ, downstream_distance / downstream_time = b + s ∧
            upstream_distance / upstream_time = b - s) :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l2368_236884


namespace NUMINAMATH_CALUDE_angle_bisector_m_abs_z_over_one_plus_i_l2368_236874

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m - 6) (m^2 - 2*m - 15)

-- Theorem 1: When z is on the angle bisector of the first and third quadrants, m = -3
theorem angle_bisector_m (m : ℝ) : z m = Complex.mk (z m).re (z m).re → m = -3 := by
  sorry

-- Theorem 2: When m = -1, |z/(1+i)| = √74
theorem abs_z_over_one_plus_i : Complex.abs (z (-1) / (1 + Complex.I)) = Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_m_abs_z_over_one_plus_i_l2368_236874


namespace NUMINAMATH_CALUDE_scientific_notation_of_1230000_l2368_236870

theorem scientific_notation_of_1230000 :
  (1230000 : ℝ) = 1.23 * (10 : ℝ) ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1230000_l2368_236870


namespace NUMINAMATH_CALUDE_initial_blue_balls_l2368_236848

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 15 → removed = 3 → prob = 1/3 → 
  ∃ (initial_blue : ℕ), 
    initial_blue = 7 ∧ 
    (initial_blue - removed : ℚ) / (total - removed) = prob :=
by sorry

end NUMINAMATH_CALUDE_initial_blue_balls_l2368_236848


namespace NUMINAMATH_CALUDE_new_difference_greater_than_original_l2368_236816

theorem new_difference_greater_than_original
  (x y a b : ℝ)
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_x_gt_y : x > y)
  (h_a_neq_b : a ≠ b) :
  (x + a) - (y - b) > x - y :=
sorry

end NUMINAMATH_CALUDE_new_difference_greater_than_original_l2368_236816


namespace NUMINAMATH_CALUDE_initial_gathering_size_l2368_236873

theorem initial_gathering_size (initial_snackers : ℕ)
  (h1 : initial_snackers = 100)
  (h2 : ∃ (a b c d e : ℕ),
    a = initial_snackers + 20 ∧
    b = a / 2 + 10 ∧
    c = b - 30 ∧
    d = c / 2 ∧
    d = 20) :
  initial_snackers = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_gathering_size_l2368_236873


namespace NUMINAMATH_CALUDE_travel_ways_theorem_l2368_236822

/-- Represents the number of transportation options between two cities -/
structure TransportOptions where
  buses : Nat
  trains : Nat
  ferries : Nat

/-- The total number of ways to travel between two cities -/
def total_ways (options : TransportOptions) : Nat :=
  options.buses + options.trains + options.ferries

theorem travel_ways_theorem (ab_morning : TransportOptions) (bc_afternoon : TransportOptions)
  (h1 : ab_morning.buses = 5)
  (h2 : ab_morning.trains = 2)
  (h3 : ab_morning.ferries = 0)
  (h4 : bc_afternoon.buses = 3)
  (h5 : bc_afternoon.trains = 0)
  (h6 : bc_afternoon.ferries = 2) :
  (total_ways ab_morning) * (total_ways bc_afternoon) = 35 := by
  sorry

#check travel_ways_theorem

end NUMINAMATH_CALUDE_travel_ways_theorem_l2368_236822


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l2368_236890

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 + ab = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => 
  p.1^2 + p.2^2 + p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 21 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m < 21, g m ≠ 4) ∧ g 21 = 4 := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l2368_236890


namespace NUMINAMATH_CALUDE_ratio_of_special_means_l2368_236886

theorem ratio_of_special_means (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h5 : a + b = 36) :
  a / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_special_means_l2368_236886


namespace NUMINAMATH_CALUDE_two_numbers_equation_l2368_236857

theorem two_numbers_equation (α β : ℝ) : 
  (α + β) / 2 = 8 → 
  Real.sqrt (α * β) = 15 → 
  ∃ (x : ℝ), x^2 - 16*x + 225 = 0 ∧ (x = α ∨ x = β) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_equation_l2368_236857


namespace NUMINAMATH_CALUDE_min_knights_in_tournament_l2368_236835

def knight_tournament (total_knights : ℕ) : Prop :=
  ∃ (lancelot_not_dueled : ℕ),
    lancelot_not_dueled = total_knights / 4 ∧
    ∃ (tristan_dueled : ℕ),
      tristan_dueled = (total_knights - lancelot_not_dueled - 1) / 7 ∧
      (total_knights - lancelot_not_dueled - 1) % 7 = 0

theorem min_knights_in_tournament :
  ∀ n : ℕ, knight_tournament n → n ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_min_knights_in_tournament_l2368_236835


namespace NUMINAMATH_CALUDE_sixth_number_divisible_by_45_and_6_l2368_236880

/-- The least common multiple of 45 and 6 -/
def lcm_45_6 : ℕ := 90

/-- The first multiple of lcm_45_6 greater than 190 -/
def first_multiple : ℕ := 270

/-- The ending number we want to prove -/
def ending_number : ℕ := 720

/-- The theorem to prove -/
theorem sixth_number_divisible_by_45_and_6 : 
  ending_number = first_multiple + 5 * lcm_45_6 ∧ 
  ending_number % 45 = 0 ∧ 
  ending_number % 6 = 0 ∧
  ∀ n : ℕ, first_multiple ≤ n ∧ n < ending_number ∧ n % 45 = 0 ∧ n % 6 = 0 → 
    ∃ k : ℕ, k < 6 ∧ n = first_multiple + k * lcm_45_6 :=
by sorry

end NUMINAMATH_CALUDE_sixth_number_divisible_by_45_and_6_l2368_236880


namespace NUMINAMATH_CALUDE_circle_area_with_radius_4_l2368_236813

/-- The area of a circle with radius 4 cm is 16π cm^2. -/
theorem circle_area_with_radius_4 :
  let r : ℝ := 4
  let area := π * r^2
  area = 16 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_radius_4_l2368_236813


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l2368_236814

/-- A triangle with angles satisfying specific ratios is right-angled -/
theorem triangle_is_right_angled (angle1 angle2 angle3 : ℝ) : 
  angle1 + angle2 + angle3 = 180 →
  angle1 = 3 * angle2 →
  angle3 = 2 * angle2 →
  angle1 = 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l2368_236814


namespace NUMINAMATH_CALUDE_tetrahedron_volume_not_determined_by_face_areas_l2368_236891

/-- A tetrahedron with four faces --/
structure Tetrahedron where
  faces : Fin 4 → Real
  volume : Real

/-- Theorem stating that the volume of a tetrahedron is not uniquely determined by its face areas --/
theorem tetrahedron_volume_not_determined_by_face_areas :
  ∃ (t1 t2 : Tetrahedron), (∀ i : Fin 4, t1.faces i = t2.faces i) ∧ t1.volume ≠ t2.volume :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_not_determined_by_face_areas_l2368_236891


namespace NUMINAMATH_CALUDE_total_instruments_is_21_instrument_group_equality_l2368_236836

-- Define the number of body parts
def num_fingers : Nat := 10
def num_hands : Nat := 2
def num_heads : Nat := 1

-- Define the number of each instrument based on the conditions
def num_trumpets : Nat := num_fingers - 3
def num_guitars : Nat := num_hands + 2
def num_trombones : Nat := num_heads + 2
def num_french_horns : Nat := num_guitars - 1
def num_violins : Nat := num_trumpets / 2
def num_saxophones : Nat := num_trombones / 3

-- State the theorem
theorem total_instruments_is_21 :
  num_trumpets + num_guitars + num_trombones + num_french_horns + num_violins + num_saxophones = 21 :=
by sorry

-- Additional condition: equality of instrument groups
theorem instrument_group_equality :
  num_trumpets + num_guitars = num_trombones + num_violins + num_saxophones :=
by sorry

end NUMINAMATH_CALUDE_total_instruments_is_21_instrument_group_equality_l2368_236836


namespace NUMINAMATH_CALUDE_farm_area_theorem_l2368_236834

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  short_side : ℝ
  long_side : ℝ
  diagonal : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the area of a rectangular farm -/
def farm_area (farm : RectangularFarm) : ℝ :=
  farm.short_side * farm.long_side

/-- Theorem: If a rectangular farm has a short side of 30 meters, and the cost of fencing
    one long side, one short side, and the diagonal at Rs. 15 per meter totals Rs. 1800,
    then the area of the farm is 1200 square meters -/
theorem farm_area_theorem (farm : RectangularFarm)
    (h1 : farm.short_side = 30)
    (h2 : farm.fencing_cost_per_meter = 15)
    (h3 : farm.total_fencing_cost = 1800)
    (h4 : farm.long_side + farm.short_side + farm.diagonal = farm.total_fencing_cost / farm.fencing_cost_per_meter)
    (h5 : farm.diagonal^2 = farm.long_side^2 + farm.short_side^2) :
    farm_area farm = 1200 := by
  sorry


end NUMINAMATH_CALUDE_farm_area_theorem_l2368_236834


namespace NUMINAMATH_CALUDE_no_factors_l2368_236853

/-- The main polynomial -/
def f (x : ℝ) : ℝ := x^4 + 3*x^2 + 8

/-- Potential factors -/
def g₁ (x : ℝ) : ℝ := x^2 + 4
def g₂ (x : ℝ) : ℝ := x + 2
def g₃ (x : ℝ) : ℝ := x^2 - 4
def g₄ (x : ℝ) : ℝ := x^2 - x - 2

theorem no_factors : 
  (¬ ∃ (h : ℝ → ℝ), f = g₁ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₂ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₃ * h) ∧
  (¬ ∃ (h : ℝ → ℝ), f = g₄ * h) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l2368_236853


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2368_236844

theorem inequality_solution_set (a b : ℝ) (h1 : a < 0) (h2 : b = a) 
  (h3 : ∀ x, ax + b ≤ 0 ↔ x ≥ -1) :
  ∀ x, (a*x + b) / (x - 2) > 0 ↔ -1 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2368_236844


namespace NUMINAMATH_CALUDE_union_C_R_A_B_eq_expected_result_l2368_236807

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

-- Define the complement of A with respect to ℝ
def C_R_A : Set ℝ := { x | x ∉ A }

-- Define the union of C_R A and B
def union_C_R_A_B : Set ℝ := C_R_A ∪ B

-- Define the expected result
def expected_result : Set ℝ := { x | x < -1 ∨ 1 < x }

-- Theorem statement
theorem union_C_R_A_B_eq_expected_result : union_C_R_A_B = expected_result := by
  sorry

end NUMINAMATH_CALUDE_union_C_R_A_B_eq_expected_result_l2368_236807


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l2368_236877

/-- Defines the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 
  1 + (n - 1) * n / 2

/-- Defines the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := 
  first_element n + n - 1

/-- Defines the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := 
  n * (first_element n + last_element n) / 2

theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l2368_236877


namespace NUMINAMATH_CALUDE_furniture_legs_l2368_236833

theorem furniture_legs 
  (total_tables : ℕ) 
  (total_legs : ℕ) 
  (four_leg_tables : ℕ) 
  (h1 : total_tables = 36)
  (h2 : total_legs = 124)
  (h3 : four_leg_tables = 16) :
  (total_legs - 4 * four_leg_tables) / (total_tables - four_leg_tables) = 3 := by
sorry

end NUMINAMATH_CALUDE_furniture_legs_l2368_236833


namespace NUMINAMATH_CALUDE_rounding_and_scientific_notation_l2368_236801

-- Define rounding to significant figures
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define rounding to decimal places
def roundToDecimalPlaces (x : ℝ) (n : ℕ) : ℝ := sorry

-- Define scientific notation
def scientificNotation (x : ℝ) : ℝ × ℤ := sorry

theorem rounding_and_scientific_notation :
  (roundToSignificantFigures 12.349 2 = 12) ∧
  (roundToDecimalPlaces 0.12349 3 = 0.123) ∧
  (scientificNotation 201200 = (2.012, 5)) ∧
  (scientificNotation 0.0002012 = (2.012, -4)) := by sorry

end NUMINAMATH_CALUDE_rounding_and_scientific_notation_l2368_236801


namespace NUMINAMATH_CALUDE_tape_pieces_for_cube_l2368_236883

/-- Represents a cube with side length n -/
structure Cube where
  sideLength : ℕ

/-- Represents a piece of tape with width 1 cm -/
structure Tape where
  length : ℕ

/-- Function to calculate the number of tape pieces needed to cover a cube -/
def tapePiecesNeeded (c : Cube) : ℕ :=
  2 * c.sideLength

/-- Theorem stating that the number of tape pieces needed is 2n -/
theorem tape_pieces_for_cube (c : Cube) :
  tapePiecesNeeded c = 2 * c.sideLength := by
  sorry

#check tape_pieces_for_cube

end NUMINAMATH_CALUDE_tape_pieces_for_cube_l2368_236883


namespace NUMINAMATH_CALUDE_class_artworks_l2368_236846

/-- Represents the number of artworks created by a class of students -/
def total_artworks (num_students : ℕ) (artworks_group1 : ℕ) (artworks_group2 : ℕ) : ℕ :=
  (num_students / 2) * artworks_group1 + (num_students / 2) * artworks_group2

/-- Theorem stating that a class of 10 students, where half make 3 artworks and half make 4, creates 35 artworks in total -/
theorem class_artworks : total_artworks 10 3 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_artworks_l2368_236846


namespace NUMINAMATH_CALUDE_sqrt_inequality_square_sum_inequality_l2368_236893

-- Theorem 1
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

-- Theorem 2
theorem square_sum_inequality (a b : ℝ) : 
  a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_square_sum_inequality_l2368_236893


namespace NUMINAMATH_CALUDE_shelter_cats_l2368_236845

theorem shelter_cats (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 15 / 7 ∧ 
  cats / (dogs + 12) = 15 / 11 → 
  cats = 45 := by
sorry

end NUMINAMATH_CALUDE_shelter_cats_l2368_236845


namespace NUMINAMATH_CALUDE_problem_solution_l2368_236805

-- Define the propositions
def P (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def Q (x : ℝ) : Prop := x = 1
def R (x : ℝ) : Prop := x^2 + x + 1 < 0
def S (x : ℝ) : Prop := x > 2
def T (x : ℝ) : Prop := x^2 - 3*x + 2 > 0

-- Theorem statement
theorem problem_solution :
  (∀ x, (¬Q x → ¬P x) ↔ (P x → Q x)) ∧
  (¬(∃ x, R x) ↔ (∀ x, x^2 + x + 1 ≥ 0)) ∧
  ((∀ x, S x → T x) ∧ ¬(∀ x, T x → S x)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2368_236805


namespace NUMINAMATH_CALUDE_F_10_squares_l2368_236830

/-- Represents the number of squares in figure F_n -/
def num_squares (n : ℕ) : ℕ :=
  1 + 3 * (n - 1) * n

/-- The theorem stating that F_10 contains 271 squares -/
theorem F_10_squares : num_squares 10 = 271 := by
  sorry

end NUMINAMATH_CALUDE_F_10_squares_l2368_236830


namespace NUMINAMATH_CALUDE_f_cos_10_deg_l2368_236862

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_cos_10_deg : 
  (∀ x, f (Real.sin x) = Real.cos (3 * x)) → 
  f (Real.cos (10 * π / 180)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_cos_10_deg_l2368_236862


namespace NUMINAMATH_CALUDE_sin_negative_nine_half_pi_l2368_236808

theorem sin_negative_nine_half_pi : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_nine_half_pi_l2368_236808


namespace NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l2368_236850

theorem coefficient_x2y3_in_binomial_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k) * x^k * y^(5-k)) =
  10 * x^2 * y^3 + (Finset.range 6).sum (fun k => if k ≠ 2 then (Nat.choose 5 k) * x^k * y^(5-k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l2368_236850


namespace NUMINAMATH_CALUDE_dannys_age_l2368_236855

/-- Proves Danny's current age given Jane's age and their age relationship 19 years ago -/
theorem dannys_age (jane_age : ℕ) (h1 : jane_age = 26) 
  (h2 : ∃ (danny_age : ℕ), danny_age - 19 = 3 * (jane_age - 19)) : 
  ∃ (danny_age : ℕ), danny_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_dannys_age_l2368_236855


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_sum_and_product_l2368_236878

theorem sum_reciprocals_of_sum_and_product (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 10) (hprod : x * y = 20) : 
  1 / x + 1 / y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_sum_and_product_l2368_236878


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2368_236815

/-- Given that a is inversely proportional to b, prove that b₁/b₂ = 5/4 when a₁/a₂ = 4/5 -/
theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (ha₁ : a₁ ≠ 0) (ha₂ : a₂ ≠ 0) (hb₁ : b₁ ≠ 0) (hb₂ : b₂ ≠ 0)
    (h_inverse : ∃ k : ℝ, a₁ * b₁ = k ∧ a₂ * b₂ = k) (h_ratio : a₁ / a₂ = 4 / 5) :
  b₁ / b₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2368_236815


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_three_l2368_236828

theorem cube_root_sum_equals_three :
  ∃ (a b : ℝ), 
    a^3 = 9 + 4 * Real.sqrt 5 ∧ 
    b^3 = 9 - 4 * Real.sqrt 5 ∧ 
    (∃ (k : ℤ), a + b = k) → 
    a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_three_l2368_236828


namespace NUMINAMATH_CALUDE_odot_problem_l2368_236896

-- Define the custom operation
def odot (a b : ℚ) : ℚ := a + (5 * a) / (3 * b)

-- State the theorem
theorem odot_problem : (odot 12 9) + 3 = 155 / 9 := by
  sorry

end NUMINAMATH_CALUDE_odot_problem_l2368_236896


namespace NUMINAMATH_CALUDE_complement_of_union_l2368_236849

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {3, 4, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2368_236849


namespace NUMINAMATH_CALUDE_ribbon_gifts_l2368_236838

theorem ribbon_gifts (total_ribbon : ℕ) (ribbon_per_gift : ℕ) (remaining_ribbon : ℕ) : 
  total_ribbon = 18 ∧ ribbon_per_gift = 2 ∧ remaining_ribbon = 6 →
  (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_gifts_l2368_236838


namespace NUMINAMATH_CALUDE_flywheel_power_l2368_236829

/-- Calculates the power of a flywheel's driving machine in horsepower -/
theorem flywheel_power (r : ℝ) (m : ℝ) (n : ℝ) (t : ℝ) : 
  r = 3 →
  m = 6000 →
  n = 800 →
  t = 3 →
  ∃ (p : ℝ), abs (p - 1431) < 1 ∧ 
  p = (m * (r * n * 2 * Real.pi / 60)^2) / (2 * t * 60 * 746) := by
  sorry

end NUMINAMATH_CALUDE_flywheel_power_l2368_236829


namespace NUMINAMATH_CALUDE_total_price_calculation_l2368_236868

theorem total_price_calculation (refrigerator_price washing_machine_price total_price : ℕ) : 
  refrigerator_price = 4275 →
  washing_machine_price = refrigerator_price - 1490 →
  total_price = refrigerator_price + washing_machine_price →
  total_price = 7060 := by
sorry

end NUMINAMATH_CALUDE_total_price_calculation_l2368_236868


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2368_236810

-- Define the approximate value of the upper bound
def upperBound : ℝ := 0.54

-- State the theorem
theorem arccos_gt_arctan_iff (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1 : ℝ) upperBound :=
by sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2368_236810


namespace NUMINAMATH_CALUDE_multiple_of_seven_in_range_l2368_236840

theorem multiple_of_seven_in_range (y : ℕ) (h1 : ∃ k : ℕ, y = 7 * k)
    (h2 : y * y > 225) (h3 : y < 30) : y = 21 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_seven_in_range_l2368_236840


namespace NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l2368_236832

theorem mixed_doubles_pairing_methods (total_players : Nat) (male_players : Nat) (female_players : Nat) 
  (selected_male : Nat) (selected_female : Nat) :
  total_players = male_players + female_players →
  male_players = 5 →
  female_players = 4 →
  selected_male = 2 →
  selected_female = 2 →
  (Nat.choose male_players selected_male) * (Nat.choose female_players selected_female) * 
  (Nat.factorial selected_male) = 120 := by
sorry

end NUMINAMATH_CALUDE_mixed_doubles_pairing_methods_l2368_236832


namespace NUMINAMATH_CALUDE_first_load_pieces_l2368_236876

/-- The number of pieces of clothing in the first load -/
def first_load (total : ℕ) (num_small_loads : ℕ) (pieces_per_small_load : ℕ) : ℕ :=
  total - (num_small_loads * pieces_per_small_load)

/-- Theorem stating that the number of pieces of clothing in the first load is 17 -/
theorem first_load_pieces : first_load 47 5 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_first_load_pieces_l2368_236876


namespace NUMINAMATH_CALUDE_platform_length_l2368_236823

/-- Given a train and platform with specific properties, prove the length of the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300)
  (h2 : time_platform = 51)
  (h3 : time_pole = 18) :
  ∃ (platform_length : ℝ), platform_length = 550 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2368_236823


namespace NUMINAMATH_CALUDE_game_probability_theorem_l2368_236879

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℕ) 
                     (alex_wins mel_wins chelsea_wins : ℕ) : Prop :=
  let mel_prob := (1 - alex_prob) * (mel_chelsea_ratio / (mel_chelsea_ratio + 1 : ℚ))
  let chelsea_prob := (1 - alex_prob) * (1 / (mel_chelsea_ratio + 1 : ℚ))
  let specific_outcome_prob := alex_prob ^ alex_wins * mel_prob ^ mel_wins * chelsea_prob ^ chelsea_wins
  let arrangements := Nat.choose total_rounds alex_wins * Nat.choose (total_rounds - alex_wins) mel_wins
  (specific_outcome_prob * arrangements : ℚ) = 76545 / 823543

theorem game_probability_theorem : 
  game_probability 7 (3/7) 3 4 2 1 :=
sorry

end NUMINAMATH_CALUDE_game_probability_theorem_l2368_236879


namespace NUMINAMATH_CALUDE_first_day_exceeding_200_chocolates_l2368_236812

def chocolate_count (n : ℕ) : ℕ := 3 * 3^(n - 1)

theorem first_day_exceeding_200_chocolates :
  (∃ n : ℕ, n > 0 ∧ chocolate_count n > 200) ∧
  (∀ m : ℕ, m > 0 ∧ m < 6 → chocolate_count m ≤ 200) ∧
  chocolate_count 6 > 200 :=
sorry

end NUMINAMATH_CALUDE_first_day_exceeding_200_chocolates_l2368_236812


namespace NUMINAMATH_CALUDE_dawn_cd_count_l2368_236897

theorem dawn_cd_count (dawn kristine : ℕ) 
  (h1 : kristine = dawn + 7)
  (h2 : dawn + kristine = 27) : 
  dawn = 10 := by
sorry

end NUMINAMATH_CALUDE_dawn_cd_count_l2368_236897


namespace NUMINAMATH_CALUDE_q_value_l2368_236831

/-- The coordinates of point A -/
def A : ℝ × ℝ := (0, 12)

/-- The coordinates of point Q -/
def Q : ℝ × ℝ := (3, 12)

/-- The coordinates of point B -/
def B : ℝ × ℝ := (15, 0)

/-- The coordinates of point O -/
def O : ℝ × ℝ := (0, 0)

/-- The coordinates of point C -/
def C (q : ℝ) : ℝ × ℝ := (0, q)

/-- The area of triangle ABC -/
def area_ABC : ℝ := 36

/-- Theorem: If the area of triangle ABC is 36 and the points have the given coordinates, then q = 9 -/
theorem q_value : ∃ q : ℝ, C q = (0, q) ∧ area_ABC = 36 → q = 9 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l2368_236831


namespace NUMINAMATH_CALUDE_option_D_is_false_l2368_236800

-- Define the proposition p and q
variable (p q : Prop)

-- Define the statement for option D
def option_D : Prop := (p ∨ q) → (p ∧ q)

-- Theorem stating that option D is false
theorem option_D_is_false : ¬ (∀ p q, option_D p q) := by
  sorry

-- Note: We don't need to prove the other options are correct in this statement,
-- as the question only asks for the incorrect option.

end NUMINAMATH_CALUDE_option_D_is_false_l2368_236800


namespace NUMINAMATH_CALUDE_equation_equivalence_l2368_236811

theorem equation_equivalence (x : ℝ) : x^2 - 10*x - 1 = 0 ↔ (x-5)^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2368_236811


namespace NUMINAMATH_CALUDE_power_three_mod_eleven_l2368_236851

theorem power_three_mod_eleven : 3^221 % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_three_mod_eleven_l2368_236851


namespace NUMINAMATH_CALUDE_like_terms_imply_abs_diff_l2368_236837

/-- 
If -5x^3y^(n-2) and 3x^(2m+5)y are like terms, then |n-5m| = 8.
-/
theorem like_terms_imply_abs_diff (n m : ℤ) : 
  (2 * m + 5 = 3 ∧ n - 2 = 1) → |n - 5 * m| = 8 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_abs_diff_l2368_236837


namespace NUMINAMATH_CALUDE_divisor_difference_two_l2368_236821

theorem divisor_difference_two (k : ℕ+) :
  ∃ (m : ℕ) (d : Fin (m + 1) → ℕ),
    (∀ i, d i ∣ (4 * k)) ∧
    (d 0 = 1) ∧
    (d (Fin.last m) = 4 * k) ∧
    (∀ i j, i < j → d i < d j) ∧
    (∃ i : Fin m, d i.succ - d i = 2) :=
by sorry

end NUMINAMATH_CALUDE_divisor_difference_two_l2368_236821


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2368_236858

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2368_236858


namespace NUMINAMATH_CALUDE_expression_simplification_l2368_236804

theorem expression_simplification (x : ℝ) :
  14 * (150 / 3 + 35 / 7 + 16 / 32 + x) = 777 + 14 * x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2368_236804


namespace NUMINAMATH_CALUDE_inequality_proof_l2368_236852

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2) ∧
  ((a * b * c = 1) → (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 3 / 2)) ∧
  ((a * b * c = 1) → (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2368_236852


namespace NUMINAMATH_CALUDE_complex_exponential_form_theta_l2368_236842

theorem complex_exponential_form_theta (z : ℂ) : 
  z = -1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (2 * Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_form_theta_l2368_236842


namespace NUMINAMATH_CALUDE_inequality_proof_l2368_236802

theorem inequality_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  a + b < b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2368_236802


namespace NUMINAMATH_CALUDE_gift_cost_theorem_l2368_236865

/-- Calculates the total cost of gifts for all workers in a company -/
def total_gift_cost (workers_per_block : ℕ) (num_blocks : ℕ) (gift_worth : ℕ) : ℕ :=
  workers_per_block * num_blocks * gift_worth

/-- The total cost of gifts for all workers in the company is $6000 -/
theorem gift_cost_theorem :
  total_gift_cost 200 15 2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_theorem_l2368_236865


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l2368_236881

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x^2 - 15*x + 1 = 0) : 
  x^4 + 1/x^4 = 49727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l2368_236881


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l2368_236866

def alternating_sequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then (n : ℤ) else -((n + 1) : ℤ)

def sequence_sum (n : ℕ) : ℤ := 
  (List.range n).map alternating_sequence |>.sum

theorem alternating_sequence_sum : sequence_sum 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l2368_236866


namespace NUMINAMATH_CALUDE_phone_repair_amount_is_10_l2368_236899

/-- The amount earned from repairing a phone -/
def phone_repair_amount : ℝ := sorry

/-- The amount earned from repairing a laptop -/
def laptop_repair_amount : ℝ := 20

/-- The total number of phones repaired -/
def total_phones : ℕ := 3 + 5

/-- The total number of laptops repaired -/
def total_laptops : ℕ := 2 + 4

/-- The total amount earned -/
def total_earned : ℝ := 200

theorem phone_repair_amount_is_10 :
  phone_repair_amount * total_phones + laptop_repair_amount * total_laptops = total_earned ∧
  phone_repair_amount = 10 := by sorry

end NUMINAMATH_CALUDE_phone_repair_amount_is_10_l2368_236899


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2368_236843

theorem cos_x_plus_2y_equals_one (x y a : ℝ) 
  (x_in_range : x ∈ Set.Icc (-π/4) (π/4))
  (y_in_range : y ∈ Set.Icc (-π/4) (π/4))
  (eq1 : x^3 + Real.sin x - 2*a = 0)
  (eq2 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry


end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2368_236843


namespace NUMINAMATH_CALUDE_new_customers_count_l2368_236827

-- Define the given conditions
def initial_customers : ℕ := 13
def final_customers : ℕ := 9
def total_left : ℕ := 8

-- Theorem to prove
theorem new_customers_count :
  ∃ (new_customers : ℕ),
    new_customers = total_left + (final_customers - (initial_customers - total_left)) :=
by
  sorry

end NUMINAMATH_CALUDE_new_customers_count_l2368_236827


namespace NUMINAMATH_CALUDE_right_triangle_sides_exist_l2368_236892

/-- A right triangle with perimeter k and incircle radius ρ --/
structure RightTriangle (k ρ : ℝ) where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq : a + b + c = k
  incircle_eq : a * b = 2 * ρ * (k / 2)
  pythagorean : a^2 + b^2 = c^2

/-- The side lengths of a right triangle satisfy the given conditions --/
theorem right_triangle_sides_exist (k ρ : ℝ) (hk : k > 0) (hρ : ρ > 0) :
  ∃ (t : RightTriangle k ρ), t.a > 0 ∧ t.b > 0 ∧ t.c > 0 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sides_exist_l2368_236892


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2368_236887

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 4 →
  c * d = x - 3 * Complex.I →
  x > 0 →
  x = 3 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2368_236887


namespace NUMINAMATH_CALUDE_regression_equation_change_l2368_236867

theorem regression_equation_change (x y : ℝ) :
  y = 3 - 5 * x →
  (3 - 5 * (x + 1)) = y - 5 := by
sorry

end NUMINAMATH_CALUDE_regression_equation_change_l2368_236867


namespace NUMINAMATH_CALUDE_prob_answer_within_four_rings_l2368_236885

/-- The probability of answering a phone call at a specific ring. -/
def prob_answer_at_ring : Fin 4 → ℝ
  | 0 => 0.1  -- First ring
  | 1 => 0.3  -- Second ring
  | 2 => 0.4  -- Third ring
  | 3 => 0.1  -- Fourth ring

/-- Theorem: The probability of answering the phone within the first four rings is 0.9. -/
theorem prob_answer_within_four_rings :
  (Finset.sum Finset.univ prob_answer_at_ring) = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_answer_within_four_rings_l2368_236885


namespace NUMINAMATH_CALUDE_liar_paradox_l2368_236875

/-- Represents the types of people in the land -/
inductive Person
  | Knight
  | Liar
  | Outsider

/-- Represents the statement "I am a liar" -/
def liarStatement : Prop := True

/-- A function that determines if a person tells the truth -/
def tellsTruth (p : Person) : Prop :=
  match p with
  | Person.Knight => True
  | Person.Liar => False
  | Person.Outsider => True

/-- A function that determines if a person's statement matches their nature -/
def statementMatches (p : Person) : Prop :=
  (p = Person.Liar) = tellsTruth p

theorem liar_paradox :
  ∀ p : Person, (p = Person.Knight ∨ p = Person.Liar) →
    (tellsTruth p = (p = Person.Liar)) → p = Person.Outsider := by
  sorry

end NUMINAMATH_CALUDE_liar_paradox_l2368_236875


namespace NUMINAMATH_CALUDE_height_in_meters_l2368_236806

-- Define Xiaochao's height in meters and centimeters
def height_m : ℝ := 1
def height_cm : ℝ := 36

-- Theorem to prove
theorem height_in_meters : height_m + height_cm / 100 = 1.36 := by
  sorry

end NUMINAMATH_CALUDE_height_in_meters_l2368_236806


namespace NUMINAMATH_CALUDE_equation_solution_l2368_236809

theorem equation_solution (k : ℝ) : 
  (∃ x : ℝ, x = -5 ∧ (1 : ℝ) / 2023 * x - 2 = 3 * x + k) →
  (∃ y : ℝ, y = -3 ∧ (1 : ℝ) / 2023 * (2 * y + 1) - 5 = 6 * y + k) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2368_236809


namespace NUMINAMATH_CALUDE_complement_P_in_U_l2368_236824

-- Define the sets U and P
def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

-- State the theorem
theorem complement_P_in_U : 
  (U \ P) = {y | y ∈ Set.Ici (1/2)} := by sorry

end NUMINAMATH_CALUDE_complement_P_in_U_l2368_236824


namespace NUMINAMATH_CALUDE_min_omega_for_max_values_l2368_236859

theorem min_omega_for_max_values (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Set.Icc 0 1, ∃ (n : ℕ), n ≥ 50 ∧ 
    (∀ y ∈ Set.Icc 0 1, Real.sin (ω * x) ≥ Real.sin (ω * y))) →
  ω ≥ 197 * Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_for_max_values_l2368_236859
