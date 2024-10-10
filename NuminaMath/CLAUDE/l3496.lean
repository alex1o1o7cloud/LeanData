import Mathlib

namespace emily_candies_l3496_349619

theorem emily_candies (bob_candies : ℕ) (jennifer_candies : ℕ) (emily_candies : ℕ)
  (h1 : jennifer_candies = 2 * emily_candies)
  (h2 : jennifer_candies = 3 * bob_candies)
  (h3 : bob_candies = 4) :
  emily_candies = 6 := by
sorry

end emily_candies_l3496_349619


namespace least_five_digit_square_cube_l3496_349678

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  n = 15625 ∧                 -- the specific number
  (∀ m : ℕ, 
    (10000 ≤ m ∧ m < 100000) ∧ 
    (∃ x : ℕ, m = x^2) ∧ 
    (∃ y : ℕ, m = y^3) → 
    n ≤ m) :=                 -- least such number
by sorry

end least_five_digit_square_cube_l3496_349678


namespace toffee_cost_l3496_349603

/-- The cost of 1 kg of toffees in rubles -/
def cost_per_kg : ℝ := 1.11

/-- The cost of 9 kg of toffees is less than 10 rubles -/
axiom nine_kg_cost : cost_per_kg * 9 < 10

/-- The cost of 10 kg of toffees is more than 11 rubles -/
axiom ten_kg_cost : cost_per_kg * 10 > 11

/-- Theorem: The cost of 1 kg of toffees is 1.11 rubles -/
theorem toffee_cost : cost_per_kg = 1.11 := by
  sorry

end toffee_cost_l3496_349603


namespace fraction_difference_equals_two_l3496_349622

theorem fraction_difference_equals_two 
  (a b : ℝ) 
  (h1 : 2 * b = 1 + a * b) 
  (h2 : a ≠ 1) 
  (h3 : b ≠ 1) : 
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 := by
  sorry

end fraction_difference_equals_two_l3496_349622


namespace marathon_remainder_yards_l3496_349649

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents the total distance run in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def yardsPerMile : ℕ := 1760

def marathonLength : Marathon := { miles := 25, yards := 500 }

def numberOfMarathons : ℕ := 12

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yardsPerMile ∧
    TotalDistance.yards (
      { miles := m
      , yards := y
      } : TotalDistance
    ) = 720 ∧
    numberOfMarathons * (marathonLength.miles * yardsPerMile + marathonLength.yards) =
      m * yardsPerMile + y :=
by sorry

end marathon_remainder_yards_l3496_349649


namespace triangle_area_from_medians_l3496_349690

theorem triangle_area_from_medians (a b : ℝ) (cos_angle : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 7) (h3 : cos_angle = -3/4) :
  let sin_angle := Real.sqrt (1 - cos_angle^2)
  let sub_triangle_area := 1/2 * (2/3 * a) * (1/3 * b) * sin_angle
  6 * sub_triangle_area = 7 :=
sorry

end triangle_area_from_medians_l3496_349690


namespace a_used_car_for_seven_hours_l3496_349602

/-- Represents the car hire scenario -/
structure CarHire where
  totalCost : ℕ
  bHours : ℕ
  bCost : ℕ
  cHours : ℕ

/-- Calculates the number of hours A used the car -/
def aHours (hire : CarHire) : ℕ :=
  (hire.totalCost - hire.bCost - (hire.cHours * hire.bCost / hire.bHours)) / (hire.bCost / hire.bHours)

/-- Theorem stating that A used the car for 7 hours given the conditions -/
theorem a_used_car_for_seven_hours :
  let hire := CarHire.mk 520 8 160 11
  aHours hire = 7 := by
  sorry


end a_used_car_for_seven_hours_l3496_349602


namespace isosceles_triangle_perimeter_l3496_349694

/-- An isosceles triangle with sides of length 4 and 8 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 4 ∧ b = 8 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 8) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 20 := by
  sorry

end isosceles_triangle_perimeter_l3496_349694


namespace inscribed_circle_distance_l3496_349621

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific coordinates for this problem

-- Define the inscribed circle with center O in triangle ABC
def InscribedCircle (O : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  true  -- We don't need to define the specific properties of the inscribed circle

-- Define points M and N where the circle touches sides AB and AC
def TouchPoints (M N : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  true  -- We don't need to define the specific properties of these points

-- Define the inscribed circle with center Q in triangle AMN
def InscribedCircleAMN (Q : ℝ × ℝ) (A M N : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific properties of this inscribed circle

-- Define the distances between points
def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry  -- We don't need to implement this function for the statement

theorem inscribed_circle_distance
  (A B C O Q M N : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : InscribedCircle O (A, B, C))
  (h3 : TouchPoints M N (A, B, C))
  (h4 : InscribedCircleAMN Q A M N)
  (h5 : Distance A B = 13)
  (h6 : Distance B C = 15)
  (h7 : Distance A C = 14) :
  Distance O Q = 4 :=
sorry

end inscribed_circle_distance_l3496_349621


namespace one_intersection_point_l3496_349646

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

-- Define a point of intersection
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem one_intersection_point :
  ∃! p : ℝ × ℝ, is_intersection p.1 p.2 :=
sorry

end one_intersection_point_l3496_349646


namespace quadratic_rewrite_l3496_349669

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 48 * x - 72 = (a * x + b)^2 + c) →
  a * b = -24 := by
sorry

end quadratic_rewrite_l3496_349669


namespace abs_sum_minimum_l3496_349640

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 7| ≥ 10 ∧ ∃ y : ℝ, |y + 3| + |y + 6| + |y + 7| = 10 :=
sorry

end abs_sum_minimum_l3496_349640


namespace eighteen_digit_divisible_by_99_l3496_349630

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

def is_single_digit (d : ℕ) : Prop := d ≤ 9

def construct_number (x y : ℕ) : ℕ :=
  x * 10^17 + 3640548981270644 + y

theorem eighteen_digit_divisible_by_99 (x y : ℕ) :
  is_single_digit x ∧ is_single_digit y →
  (is_divisible_by_99 (construct_number x y) ↔ x = 9 ∧ y = 1) := by
  sorry

end eighteen_digit_divisible_by_99_l3496_349630


namespace calculation_one_calculation_two_l3496_349666

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem for the first calculation
theorem calculation_one :
  (1 / Real.sqrt 0.04) + (1 / Real.sqrt 27) ^ (1/3) + (Real.sqrt 2 + 1)⁻¹ - 2 ^ (1/2) + (-2) ^ 0 = 8 := by sorry

-- Theorem for the second calculation
theorem calculation_two :
  (2/5) * lg 32 + lg 50 + Real.sqrt ((lg 3)^2 - lg 9 + 1) - lg (2/3) = 3 := by sorry

end calculation_one_calculation_two_l3496_349666


namespace y₁_less_than_y₂_l3496_349656

/-- Linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (2, f 2)

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (-1, f (-1))

/-- y₁ coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ coordinate of point B -/
def y₂ : ℝ := B.2

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l3496_349656


namespace polygon_exterior_angles_l3496_349655

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  n > 2 → exterior_angle = 24 → n * exterior_angle = 360 → n = 15 := by
  sorry

end polygon_exterior_angles_l3496_349655


namespace geometric_sequence_sum_range_l3496_349645

theorem geometric_sequence_sum_range (m : ℝ) (hm : m > 0) :
  ∃ (a b c : ℝ), (a ≠ 0 ∧ b / a = c / b) ∧ (a + b + c = m) →
  b ∈ Set.Icc (-m) 0 ∪ Set.Ioc 0 (m / 3) :=
sorry

end geometric_sequence_sum_range_l3496_349645


namespace classroom_notebooks_l3496_349696

theorem classroom_notebooks (total_students : ℕ) 
  (notebooks_group1 : ℕ) (notebooks_group2 : ℕ) : 
  total_students = 28 → 
  notebooks_group1 = 5 → 
  notebooks_group2 = 3 → 
  (total_students / 2) * notebooks_group1 + (total_students / 2) * notebooks_group2 = 112 := by
  sorry

end classroom_notebooks_l3496_349696


namespace lemon_head_problem_l3496_349624

/-- Given a package size and a total number of Lemon Heads eaten, 
    calculate the number of whole boxes eaten and Lemon Heads left over. -/
def lemonHeadBoxes (packageSize : ℕ) (totalEaten : ℕ) : ℕ × ℕ :=
  (totalEaten / packageSize, totalEaten % packageSize)

/-- Theorem: Given a package size of 6 Lemon Heads and 54 Lemon Heads eaten,
    prove that 9 whole boxes were eaten with 0 Lemon Heads left over. -/
theorem lemon_head_problem : lemonHeadBoxes 6 54 = (9, 0) := by
  sorry

end lemon_head_problem_l3496_349624


namespace extra_boxes_calculation_l3496_349635

/-- Proves that given an order of 3 dozen boxes with extra free boxes equivalent to a 25% discount, the number of extra boxes received is 9 -/
theorem extra_boxes_calculation (dozen : ℕ) (order_size : ℕ) (discount_percent : ℚ) : 
  dozen = 12 →
  order_size = 3 →
  discount_percent = 25 / 100 →
  (dozen * order_size : ℚ) * (1 - discount_percent) = dozen * order_size - 9 :=
by sorry

end extra_boxes_calculation_l3496_349635


namespace sum_reciprocal_n_n_plus_three_l3496_349629

/-- The sum of the infinite series ∑(1 / (n(n+3))) for n from 1 to infinity is equal to 7/9. -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 3)) = 7 / 9 := by sorry

end sum_reciprocal_n_n_plus_three_l3496_349629


namespace price_increase_percentage_l3496_349698

theorem price_increase_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 420) :
  (new_price - original_price) / original_price * 100 = 40 := by
  sorry

end price_increase_percentage_l3496_349698


namespace circular_garden_radius_l3496_349675

/-- 
Given a circular garden with radius r, if the length of the fence (circumference) 
is 1/8 of the area of the garden, then r = 16.
-/
theorem circular_garden_radius (r : ℝ) (h : r > 0) : 
  2 * π * r = (1 / 8) * π * r^2 → r = 16 := by sorry

end circular_garden_radius_l3496_349675


namespace fourth_and_fifth_hexagons_sum_l3496_349653

/-- Represents the number of dots in a hexagonal layer -/
def hexDots : ℕ → ℕ
| 0 => 1  -- central dot
| 1 => 6  -- first layer around central dot
| 2 => 12 -- second layer
| n + 3 => 6 * (n + 1) + 2 * hexDots n  -- new pattern from 4th hexagon onwards

/-- Total dots up to and including the nth hexagon -/
def totalDots : ℕ → ℕ
| 0 => 1
| n + 1 => totalDots n + hexDots (n + 1)

theorem fourth_and_fifth_hexagons_sum :
  totalDots 5 - totalDots 3 = 138 := by
  sorry

end fourth_and_fifth_hexagons_sum_l3496_349653


namespace round_trip_distance_solve_specific_problem_l3496_349600

/-- Calculates the one-way distance of a round trip given the speeds and total time -/
theorem round_trip_distance 
  (speed_to : ℝ) 
  (speed_from : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_to > 0)
  (h2 : speed_from > 0)
  (h3 : total_time > 0) :
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance / speed_to + distance / speed_from = total_time) := by
  sorry

/-- Solves the specific problem with given values -/
theorem solve_specific_problem :
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance / 50 + distance / 75 = 10) ∧
    distance = 300 := by
  sorry

end round_trip_distance_solve_specific_problem_l3496_349600


namespace shelves_used_l3496_349642

def initial_stock : ℕ := 5
def new_shipment : ℕ := 7
def bears_per_shelf : ℕ := 6

theorem shelves_used (initial_stock new_shipment bears_per_shelf : ℕ) :
  initial_stock = 5 →
  new_shipment = 7 →
  bears_per_shelf = 6 →
  (initial_stock + new_shipment) / bears_per_shelf = 2 := by
sorry

end shelves_used_l3496_349642


namespace car_speed_problem_l3496_349639

/-- Proves that given a 15-hour trip where a car travels at 30 mph for the first 5 hours
    and the overall average speed is 38 mph, the average speed for the remaining 10 hours is 42 mph. -/
theorem car_speed_problem (v : ℝ) : 
  (5 * 30 + 10 * v) / 15 = 38 → v = 42 := by
  sorry

end car_speed_problem_l3496_349639


namespace cubic_function_derivative_l3496_349643

/-- Given a function f(x) = ax³ + 4x² + 3x, prove that if f'(1) = 2, then a = -3 -/
theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + 4 * x^2 + 3 * x
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 8 * x + 3
  f' 1 = 2 → a = -3 := by
  sorry

end cubic_function_derivative_l3496_349643


namespace tangent_slope_at_point_A_l3496_349672

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the point A
def point_A : ℝ × ℝ := (2, 10)

-- Theorem statement
theorem tangent_slope_at_point_A :
  (deriv f) point_A.1 = 7 :=
sorry

end tangent_slope_at_point_A_l3496_349672


namespace displacement_increment_l3496_349608

/-- Given an object with equation of motion s = 2t^2, 
    prove that the increment of displacement from time t = 2 to t = 2 + d 
    is equal to 8d + 2d^2 -/
theorem displacement_increment (d : ℝ) : 
  let s (t : ℝ) := 2 * t^2
  (s (2 + d) - s 2) = 8*d + 2*d^2 := by
sorry

end displacement_increment_l3496_349608


namespace max_value_of_expression_l3496_349665

theorem max_value_of_expression (t : ℝ) : 
  ∃ (max : ℝ), max = (1 / 16) ∧ ∀ t, ((3^t - 4*t) * t) / (9^t) ≤ max :=
sorry

end max_value_of_expression_l3496_349665


namespace boat_speed_in_still_water_l3496_349638

/-- The speed of a boat in still water, given its downstream travel information and current rate. -/
theorem boat_speed_in_still_water 
  (current_rate : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (h1 : current_rate = 5)
  (h2 : distance_downstream = 11.25)
  (h3 : time_minutes = 27) :
  ∃ (speed_still_water : ℝ), 
    speed_still_water = 20 ∧ 
    distance_downstream = (speed_still_water + current_rate) * (time_minutes / 60) :=
by sorry

end boat_speed_in_still_water_l3496_349638


namespace shirt_price_proof_l3496_349637

theorem shirt_price_proof (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : final_price = 105)
  (h2 : discount1 = 19.954259576901087)
  (h3 : discount2 = 12.55) :
  ∃ (list_price : ℝ), 
    list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = final_price ∧ 
    list_price = 150 := by
  sorry

end shirt_price_proof_l3496_349637


namespace line_segment_length_l3496_349699

theorem line_segment_length : Real.sqrt ((8 - 3)^2 + (16 - 4)^2) = 13 := by
  sorry

end line_segment_length_l3496_349699


namespace square_floor_tiles_l3496_349681

theorem square_floor_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n ^ 2 = 169 := by
  sorry

end square_floor_tiles_l3496_349681


namespace election_winner_votes_l3496_349677

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 62 / 100) 
  (h2 : vote_difference = 336) 
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 868 :=
sorry

end election_winner_votes_l3496_349677


namespace equation_impossible_l3496_349667

-- Define the set of digits from 1 to 9
def Digits : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the property that all variables are distinct
def AllDistinct (K T U Ch O H : Nat) : Prop :=
  K ≠ T ∧ K ≠ U ∧ K ≠ Ch ∧ K ≠ O ∧ K ≠ H ∧
  T ≠ U ∧ T ≠ Ch ∧ T ≠ O ∧ T ≠ H ∧
  U ≠ Ch ∧ U ≠ O ∧ U ≠ H ∧
  Ch ≠ O ∧ Ch ≠ H ∧
  O ≠ H

theorem equation_impossible :
  ∀ (K T U Ch O H : Nat),
    K ∈ Digits → T ∈ Digits → U ∈ Digits → Ch ∈ Digits → O ∈ Digits → H ∈ Digits →
    AllDistinct K T U Ch O H →
    K * 0 * T ≠ U * Ch * O * H * H * U :=
by sorry

end equation_impossible_l3496_349667


namespace bakery_roll_combinations_l3496_349663

theorem bakery_roll_combinations :
  let n : ℕ := 4  -- number of remaining rolls
  let k : ℕ := 4  -- number of kinds of rolls
  let total : ℕ := 8  -- total number of rolls
  Nat.choose (n + k - 1) (k - 1) = 35 :=
by
  sorry

end bakery_roll_combinations_l3496_349663


namespace hyperbola_vertices_distance_l3496_349686

/-- The distance between the vertices of a hyperbola with equation (y^2 / 27) - (x^2 / 11) = 1 is 6√3. -/
theorem hyperbola_vertices_distance :
  let hyperbola := {p : ℝ × ℝ | (p.2^2 / 27) - (p.1^2 / 11) = 1}
  ∃ v₁ v₂ : ℝ × ℝ, v₁ ∈ hyperbola ∧ v₂ ∈ hyperbola ∧ 
    ∀ p ∈ hyperbola, dist p v₁ ≤ dist v₁ v₂ ∧ dist p v₂ ≤ dist v₁ v₂ ∧
    dist v₁ v₂ = 6 * Real.sqrt 3 :=
sorry

end hyperbola_vertices_distance_l3496_349686


namespace maglev_train_speed_l3496_349679

/-- Proves that the average speed of a maglev train is 225 km/h given specific conditions --/
theorem maglev_train_speed :
  ∀ (subway_speed : ℝ),
    subway_speed > 0 →
    let maglev_speed := 6.25 * subway_speed
    let distance := 30
    let subway_time := distance / subway_speed
    let maglev_time := distance / maglev_speed
    subway_time - maglev_time = 0.7 →
    maglev_speed = 225 := by
  sorry

#check maglev_train_speed

end maglev_train_speed_l3496_349679


namespace smallest_natural_ending_2012_l3496_349670

theorem smallest_natural_ending_2012 : 
  ∃ (n : ℕ), n = 1716 ∧ 
  (∀ (m : ℕ), m < n → (m * 7) % 10000 ≠ 2012) ∧ 
  (n * 7) % 10000 = 2012 := by
sorry

end smallest_natural_ending_2012_l3496_349670


namespace continuity_at_4_l3496_349671

def f (x : ℝ) : ℝ := 2 * x^2 - 3

theorem continuity_at_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |f x - f 4| < ε :=
by sorry

end continuity_at_4_l3496_349671


namespace product_repeating_decimal_and_fraction_l3496_349657

theorem product_repeating_decimal_and_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^n - x.floor) * 10 ≥ 6 ∧ (x * 10^n - x.floor) * 10 < 7) →
  x * (7/3) = 14/9 := by
sorry

end product_repeating_decimal_and_fraction_l3496_349657


namespace percentage_of_x_pay_to_y_l3496_349693

/-- The percentage of X's pay compared to Y's, given their total pay and Y's pay -/
theorem percentage_of_x_pay_to_y (total_pay y_pay x_pay : ℚ) : 
  total_pay = 528 →
  y_pay = 240 →
  x_pay + y_pay = total_pay →
  (x_pay / y_pay) * 100 = 120 := by
  sorry

end percentage_of_x_pay_to_y_l3496_349693


namespace test_questions_count_l3496_349618

/-- Calculates the total number of questions on a test given the time spent answering,
    time per question, and number of unanswered questions. -/
def totalQuestions (hoursSpent : ℕ) (minutesPerQuestion : ℕ) (unansweredQuestions : ℕ) : ℕ :=
  (hoursSpent * 60 / minutesPerQuestion) + unansweredQuestions

/-- Proves that the total number of questions on the test is 100 -/
theorem test_questions_count :
  totalQuestions 2 2 40 = 100 := by sorry

end test_questions_count_l3496_349618


namespace solve_missed_questions_l3496_349660

def missed_questions_problem (your_missed : ℕ) (friend_ratio : ℕ) : Prop :=
  let friend_missed := (your_missed / friend_ratio : ℕ)
  your_missed = 36 ∧ friend_ratio = 5 →
  your_missed + friend_missed = 43

theorem solve_missed_questions : missed_questions_problem 36 5 := by
  sorry

end solve_missed_questions_l3496_349660


namespace polynomial_equality_l3496_349613

-- Define the polynomials P and Q
def P (x y z w : ℝ) : ℝ := x * y + x^2 - z + w
def Q (x y z w : ℝ) : ℝ := x + y

-- State the theorem
theorem polynomial_equality (x y z w : ℝ) :
  (x * y + z + w)^2 - (x^2 - 2*z)*(y^2 - 2*w) = 
  (P x y z w)^2 - (x^2 - 2*z)*(Q x y z w)^2 :=
by sorry

end polynomial_equality_l3496_349613


namespace family_milk_consumption_l3496_349689

/-- Represents the milk consumption of a family member -/
structure MilkConsumption where
  regular : ℝ
  soy : ℝ
  almond : ℝ
  cashew : ℝ
  oat : ℝ
  coconut : ℝ
  lactoseFree : ℝ

/-- Calculates the total milk consumption excluding lactose-free milk -/
def totalConsumption (c : MilkConsumption) : ℝ :=
  c.regular + c.soy + c.almond + c.cashew + c.oat + c.coconut

/-- Represents the family's milk consumption -/
structure FamilyConsumption where
  mitch : MilkConsumption
  sister : MilkConsumption
  mother : MilkConsumption
  father : MilkConsumption
  extraSoyMilk : ℝ

theorem family_milk_consumption (family : FamilyConsumption)
    (h_mitch : family.mitch = ⟨3, 2, 1, 0, 0, 0, 0⟩)
    (h_sister : family.sister = ⟨1.5, 3, 1.5, 1, 0, 0, 0⟩)
    (h_mother : family.mother = ⟨0.5, 2.5, 0, 0, 1, 0, 0.5⟩)
    (h_father : family.father = ⟨2, 1, 3, 0, 0, 1, 0⟩)
    (h_extra_soy : family.extraSoyMilk = 7.5) :
    totalConsumption family.mitch +
    totalConsumption family.sister +
    totalConsumption family.mother +
    totalConsumption family.father +
    family.extraSoyMilk = 31.5 := by
  sorry


end family_milk_consumption_l3496_349689


namespace total_amount_is_175_l3496_349662

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total amount from a given share distribution -/
def totalAmount (s : ShareDistribution) : ℝ :=
  s.x + s.y + s.z

/-- Theorem stating that given the conditions, the total amount is 175 -/
theorem total_amount_is_175 :
  ∀ (s : ShareDistribution),
    s.y = 45 →                -- y's share is 45
    s.y = 0.45 * s.x →        -- y gets 45 paisa for each rupee x gets
    s.z = 0.30 * s.x →        -- z gets 30 paisa for each rupee x gets
    totalAmount s = 175 :=
by
  sorry

#check total_amount_is_175

end total_amount_is_175_l3496_349662


namespace cylinder_lateral_surface_area_l3496_349683

/-- Given a cylinder formed by rotating a square around one of its sides,
    if the volume of the cylinder is 27π cm³,
    then its lateral surface area is 18π cm². -/
theorem cylinder_lateral_surface_area 
  (side : ℝ) 
  (h_cylinder : side > 0) 
  (h_volume : π * side^2 * side = 27 * π) : 
  2 * π * side * side = 18 * π :=
sorry

end cylinder_lateral_surface_area_l3496_349683


namespace students_on_sports_teams_l3496_349614

theorem students_on_sports_teams 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (both_activities : ℕ) 
  (either_activity : ℕ) 
  (h1 : total_students = 320)
  (h2 : band_students = 85)
  (h3 : both_activities = 60)
  (h4 : either_activity = 225)
  (h5 : either_activity = band_students + sports_students - both_activities) :
  sports_students = 200 :=
by
  sorry

end students_on_sports_teams_l3496_349614


namespace symmetric_line_equation_l3496_349634

/-- Given two lines l₁ and l in the plane, this theorem states that the line l₂ 
    which is symmetric to l₁ with respect to l has a specific equation. -/
theorem symmetric_line_equation (x y : ℝ) : 
  let l₁ : ℝ → ℝ := λ x => 2 * x
  let l : ℝ → ℝ := λ x => 3 * x + 3
  let l₂ : ℝ → ℝ := λ x => (11 * x - 21) / 2
  (∀ x, l₂ x = y ↔ 11 * x - 2 * y + 21 = 0) ∧
  (∀ p : ℝ × ℝ, 
    let p₁ := (p.1, l₁ p.1)
    let m := ((p.1 + p₁.1) / 2, (p.2 + p₁.2) / 2)
    m.2 = l m.1 → p.2 = l₂ p.1) :=
by sorry


end symmetric_line_equation_l3496_349634


namespace equation_solution_l3496_349632

theorem equation_solution (y : ℚ) : 
  (∃ x : ℚ, 19 * (x + y) + 17 = 19 * (-x + y) - 21) → 
  (∀ x : ℚ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 → x = -1) :=
by sorry

end equation_solution_l3496_349632


namespace number_of_males_l3496_349654

def town_population : ℕ := 500
def male_percentage : ℚ := 2/5

theorem number_of_males :
  (town_population : ℚ) * male_percentage = 200 := by sorry

end number_of_males_l3496_349654


namespace antonieta_initial_tickets_l3496_349664

/-- The number of tickets required for the Ferris wheel -/
def ferris_wheel_tickets : ℕ := 6

/-- The number of tickets required for the roller coaster -/
def roller_coaster_tickets : ℕ := 5

/-- The number of tickets required for the log ride -/
def log_ride_tickets : ℕ := 7

/-- The number of additional tickets Antonieta needs to buy -/
def additional_tickets_needed : ℕ := 16

/-- The initial number of tickets Antonieta has -/
def initial_tickets : ℕ := 2

theorem antonieta_initial_tickets :
  initial_tickets + additional_tickets_needed =
  ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets :=
by sorry

end antonieta_initial_tickets_l3496_349664


namespace prob_three_correct_is_one_twelfth_l3496_349674

def number_of_houses : ℕ := 5

def probability_three_correct_deliveries : ℚ :=
  (number_of_houses.choose 3 * 1) / number_of_houses.factorial

theorem prob_three_correct_is_one_twelfth :
  probability_three_correct_deliveries = 1 / 12 := by
  sorry

end prob_three_correct_is_one_twelfth_l3496_349674


namespace no_valid_x_l3496_349625

theorem no_valid_x : ¬ ∃ (x : ℕ+), 
  (x : ℝ) - 7 > 0 ∧ 
  (x + 5) * (x - 7) * (x^2 + x + 30) < 800 :=
sorry

end no_valid_x_l3496_349625


namespace certain_number_problem_l3496_349617

theorem certain_number_problem (x : ℤ) : x + 3 = 226 → 3 * x = 669 := by
  sorry

end certain_number_problem_l3496_349617


namespace diamond_digit_equality_l3496_349607

theorem diamond_digit_equality (diamond : ℕ) : 
  diamond < 10 →  -- diamond is a digit
  (9 * diamond + 6 = 10 * diamond + 3) →  -- diamond6₉ = diamond3₁₀
  diamond = 3 :=
by sorry

end diamond_digit_equality_l3496_349607


namespace min_sum_of_squares_l3496_349620

theorem min_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 2*a + 3*b + 4*c = 120) :
  a^2 + b^2 + c^2 ≥ 14400/29 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    2*a₀ + 3*b₀ + 4*c₀ = 120 ∧ a₀^2 + b₀^2 + c₀^2 = 14400/29 :=
by
  sorry

end min_sum_of_squares_l3496_349620


namespace class_size_calculation_l3496_349623

theorem class_size_calculation (female_students : ℕ) (male_students : ℕ) : 
  female_students = 13 → 
  male_students = 3 * female_students → 
  female_students + male_students = 52 := by
sorry

end class_size_calculation_l3496_349623


namespace holly_pill_ratio_l3496_349641

/-- Represents the daily pill intake for Holly --/
structure DailyPillIntake where
  insulin : ℕ
  blood_pressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills taken in a week --/
def weekly_total (d : DailyPillIntake) : ℕ :=
  7 * (d.insulin + d.blood_pressure + d.anticonvulsant)

/-- Represents the ratio of two numbers --/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem holly_pill_ratio :
  ∀ (d : DailyPillIntake),
    d.insulin = 2 →
    d.blood_pressure = 3 →
    weekly_total d = 77 →
    ∃ (r : Ratio), r.numerator = 2 ∧ r.denominator = 1 ∧
      r.numerator * d.blood_pressure = r.denominator * d.anticonvulsant :=
by sorry

end holly_pill_ratio_l3496_349641


namespace chocolate_division_l3496_349644

theorem chocolate_division (total : ℝ) (total_positive : 0 < total) : 
  let al_share := (4 / 10) * total
  let bert_share := (3 / 10) * total
  let carl_share := (2 / 10) * total
  let dana_share := (1 / 10) * total
  al_share + bert_share + carl_share + dana_share = total :=
by sorry

end chocolate_division_l3496_349644


namespace eight_people_lineup_l3496_349631

theorem eight_people_lineup : Nat.factorial 8 = 40320 := by
  sorry

end eight_people_lineup_l3496_349631


namespace vector_equation_solution_l3496_349652

theorem vector_equation_solution :
  let a : ℚ := -491/342
  let b : ℚ := 233/342
  let c : ℚ := 49/38
  let v₁ : Fin 3 → ℚ := ![1, -2, 3]
  let v₂ : Fin 3 → ℚ := ![4, 1, -1]
  let v₃ : Fin 3 → ℚ := ![-3, 2, 1]
  let result : Fin 3 → ℚ := ![0, 1, 4]
  (a • v₁) + (b • v₂) + (c • v₃) = result := by
  sorry

end vector_equation_solution_l3496_349652


namespace exactly_seven_numbers_satisfy_condition_l3496_349697

/-- A two-digit number is a natural number between 10 and 99 inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number. -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of digits of a two-digit number. -/
def digitSum (n : ℕ) : ℕ := tensDigit n + unitsDigit n

/-- The condition specified in the problem. -/
def satisfiesCondition (n : ℕ) : Prop :=
  TwoDigitNumber n ∧ unitsDigit (n - 2 * digitSum n) = 4

/-- The main theorem stating that exactly 7 two-digit numbers satisfy the condition. -/
theorem exactly_seven_numbers_satisfy_condition :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfiesCondition n) ∧ s.card = 7 := by
  sorry

end exactly_seven_numbers_satisfy_condition_l3496_349697


namespace daisy_monday_toys_l3496_349636

/-- The number of dog toys Daisy had on Monday -/
def monday_toys : ℕ := sorry

/-- The number of dog toys Daisy had on Tuesday after losing some -/
def tuesday_toys : ℕ := 3

/-- The number of new toys Daisy's owner bought on Tuesday -/
def tuesday_new_toys : ℕ := 3

/-- The number of new toys Daisy's owner bought on Wednesday -/
def wednesday_new_toys : ℕ := 5

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_toys : ℕ := 13

theorem daisy_monday_toys : 
  monday_toys = 5 :=
by sorry

end daisy_monday_toys_l3496_349636


namespace ben_savings_days_l3496_349680

/-- Calculates the number of days elapsed given Ben's savings scenario --/
def days_elapsed (daily_start : ℕ) (daily_spend : ℕ) (final_amount : ℕ) : ℕ :=
  let daily_save := daily_start - daily_spend
  let d : ℕ := (final_amount - 10) / (2 * daily_save)
  d

/-- Theorem stating that the number of days elapsed is 7 --/
theorem ben_savings_days : days_elapsed 50 15 500 = 7 := by
  sorry

end ben_savings_days_l3496_349680


namespace magnitude_relationship_l3496_349610

theorem magnitude_relationship (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b ∧ a * b < a * b^2 := by
  sorry

end magnitude_relationship_l3496_349610


namespace floor_of_5_7_l3496_349687

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end floor_of_5_7_l3496_349687


namespace bouquet_cost_60_l3496_349688

/-- The cost of a bouquet of tulips at Tony's Tulip Tower -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_rate := 36 / 18
  let threshold := 40
  let extra_rate := base_rate * (3/2)
  if n ≤ threshold then
    n * base_rate
  else
    threshold * base_rate + (n - threshold) * extra_rate

/-- The theorem stating the cost of a bouquet of 60 tulips -/
theorem bouquet_cost_60 : bouquet_cost 60 = 140 := by
  sorry

#eval bouquet_cost 60

end bouquet_cost_60_l3496_349688


namespace system_of_equations_solutions_l3496_349647

theorem system_of_equations_solutions :
  (∃ x y : ℝ, x - 2*y = 0 ∧ 3*x - y = 5 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 3*(x - 1) - 4*(y + 1) = -1 ∧ x/2 + y/3 = -2 ∧ x = -2 ∧ y = -3) :=
by sorry

end system_of_equations_solutions_l3496_349647


namespace cone_lateral_surface_area_l3496_349604

/-- Given a cone with base radius 2 and lateral surface forming a semicircle,
    prove that its lateral surface area is 8π. -/
theorem cone_lateral_surface_area (r : ℝ) (h : r = 2) :
  let l := 2 * r  -- slant height is twice the base radius for a semicircle lateral surface
  π * r * l = 8 * π := by
  sorry

end cone_lateral_surface_area_l3496_349604


namespace cistern_length_is_eight_l3496_349668

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that a cistern with given dimensions has a length of 8 meters --/
theorem cistern_length_is_eight (c : Cistern) 
    (h1 : c.width = 4)
    (h2 : c.depth = 1.25)
    (h3 : c.wetSurfaceArea = 62)
    (h4 : wetSurfaceArea c = c.wetSurfaceArea) : 
    c.length = 8 := by
  sorry


end cistern_length_is_eight_l3496_349668


namespace zoes_bottles_l3496_349651

/-- Given the initial number of bottles, the number of bottles drunk, and the number of bottles bought,
    calculate the final number of bottles. -/
def finalBottles (initial drunk bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Prove that for Zoe's specific case, the final number of bottles is 47. -/
theorem zoes_bottles : finalBottles 42 25 30 = 47 := by
  sorry

end zoes_bottles_l3496_349651


namespace line_equation_proof_l3496_349648

/-- Given a line passing through the point (√3, -3) with an inclination angle of 30°,
    prove that its equation is y = (√3/3)x - 4 -/
theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (Real.sqrt 3, -3)
  let angle : ℝ := 30 * π / 180  -- Convert 30° to radians
  let slope : ℝ := Real.tan angle
  slope * (x - point.1) = y - point.2 →
  y = (Real.sqrt 3 / 3) * x - 4 := by
sorry

end line_equation_proof_l3496_349648


namespace annual_population_increase_rounded_l3496_349676

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours between births -/
def hours_per_birth : ℕ := 6

/-- The number of hours between deaths -/
def hours_per_death : ℕ := 10

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- Calculate the annual population increase -/
def annual_population_increase : ℕ :=
  (hours_per_day / hours_per_birth - hours_per_day / hours_per_death) * days_per_year

/-- Round to the nearest hundred -/
def round_to_hundred (n : ℕ) : ℕ :=
  ((n + 50) / 100) * 100

/-- Theorem stating the annual population increase rounded to the nearest hundred -/
theorem annual_population_increase_rounded :
  round_to_hundred annual_population_increase = 700 := by
  sorry

end annual_population_increase_rounded_l3496_349676


namespace binomial_coefficient_problem_l3496_349609

theorem binomial_coefficient_problem (m : ℕ) :
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m) →
  Nat.choose 8 m = 28 := by
  sorry

end binomial_coefficient_problem_l3496_349609


namespace greatest_integer_radius_l3496_349650

theorem greatest_integer_radius (A : ℝ) (h : A < 60 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r ∧ r = 7 :=
sorry

end greatest_integer_radius_l3496_349650


namespace vector_problem_l3496_349605

/-- Given vectors a and b, if |a| = 6 and a ∥ b, then x = 4 and x + y = 8 -/
theorem vector_problem (x y : ℝ) : 
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (‖a‖ = 6 ∧ ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = 4 ∧ x + y = 8 := by
  sorry


end vector_problem_l3496_349605


namespace complex_equation_solution_l3496_349692

theorem complex_equation_solution :
  ∃ (z : ℂ), z = -3/4 * I ∧ (2 : ℂ) - I * z = -1 + 3 * I * z :=
by
  sorry

end complex_equation_solution_l3496_349692


namespace quadratic_inequality_range_l3496_349661

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_inequality_range_l3496_349661


namespace difference_of_numbers_l3496_349684

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares : x^2 - y^2 = 80) : 
  x - y = 8 := by
sorry

end difference_of_numbers_l3496_349684


namespace inequality_proof_l3496_349673

theorem inequality_proof (a b c d e f : ℝ) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c ≥ |d * x^2 + e * x + f|) : 
  4 * a * c - b^2 ≥ |4 * d * f - e^2| := by
  sorry

end inequality_proof_l3496_349673


namespace april_rainfall_calculation_l3496_349691

/-- Given the March rainfall and the difference between March and April rainfall,
    calculate the April rainfall. -/
def april_rainfall (march_rainfall : ℝ) (rainfall_difference : ℝ) : ℝ :=
  march_rainfall - rainfall_difference

/-- Theorem stating that given the specific March rainfall and difference,
    the April rainfall is 0.46 inches. -/
theorem april_rainfall_calculation :
  april_rainfall 0.81 0.35 = 0.46 := by
  sorry

end april_rainfall_calculation_l3496_349691


namespace min_value_at_seven_l3496_349627

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem min_value_at_seven :
  ∀ x : ℝ, f 7 ≤ f x :=
by sorry

end min_value_at_seven_l3496_349627


namespace lamplighter_monkey_distance_l3496_349685

/-- Represents the speed and time of a monkey's movement --/
structure MonkeyMovement where
  speed : ℝ
  time : ℝ

/-- Calculates the total distance traveled by a Lamplighter monkey --/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.time + swinging.speed * swinging.time

/-- Theorem stating the total distance traveled by the Lamplighter monkey --/
theorem lamplighter_monkey_distance :
  let running := MonkeyMovement.mk 15 5
  let swinging := MonkeyMovement.mk 10 10
  totalDistance running swinging = 175 := by sorry

end lamplighter_monkey_distance_l3496_349685


namespace cubic_sum_theorem_l3496_349616

theorem cubic_sum_theorem (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := by
sorry

end cubic_sum_theorem_l3496_349616


namespace factorization_proof_l3496_349695

theorem factorization_proof (x : ℝ) : 4*x*(x-5) + 7*(x-5) + 12*(x-5) = (4*x + 19)*(x-5) := by
  sorry

end factorization_proof_l3496_349695


namespace children_group_size_l3496_349633

theorem children_group_size (adults_per_group : ℕ) (total_adults : ℕ) (total_children : ℕ) :
  adults_per_group = 17 →
  total_adults = 255 →
  total_children = total_adults →
  total_adults % adults_per_group = 0 →
  ∃ (children_per_group : ℕ),
    children_per_group > 0 ∧
    total_children % children_per_group = 0 ∧
    total_children / children_per_group = total_adults / adults_per_group ∧
    children_per_group = 17 := by
  sorry

end children_group_size_l3496_349633


namespace solution_set_min_value_min_value_ab_equality_condition_l3496_349626

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Part 1: Solution set of f(x) ≤ 4
theorem solution_set (x : ℝ) : f x ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4/3 := by sorry

-- Part 2: Minimum value of f(x)
theorem min_value : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 3 := by sorry

-- Part 3: Minimum value of 1/(a-1) + 2/b given conditions
theorem min_value_ab (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + 2*b = 3) :
  1/(a-1) + 2/b ≥ 9/2 := by sorry

-- Part 4: Equality condition for the minimum value
theorem equality_condition (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + 2*b = 3) :
  1/(a-1) + 2/b = 9/2 ↔ a = 5/3 ∧ b = 2/3 := by sorry

end solution_set_min_value_min_value_ab_equality_condition_l3496_349626


namespace min_value_x2_plus_y2_l3496_349606

theorem min_value_x2_plus_y2 (x y : ℝ) (h : (x + 1)^2 + y^2 = 1/4) :
  ∃ (min : ℝ), min = 1/4 ∧ ∀ (a b : ℝ), (a + 1)^2 + b^2 = 1/4 → a^2 + b^2 ≥ min :=
by sorry

end min_value_x2_plus_y2_l3496_349606


namespace savings_after_four_weeks_l3496_349612

/-- Calculates the total savings after a given number of weeks, 
    with an initial saving amount and a fixed weekly increase. -/
def totalSavings (initialSaving : ℕ) (weeklyIncrease : ℕ) (weeks : ℕ) : ℕ :=
  initialSaving + weeklyIncrease * (weeks - 1)

/-- Theorem: Given an initial saving of $20 and a weekly increase of $10,
    the total savings after 4 weeks is $60. -/
theorem savings_after_four_weeks :
  totalSavings 20 10 4 = 60 := by
  sorry

end savings_after_four_weeks_l3496_349612


namespace triangle_area_l3496_349601

/-- A triangle with integral sides and perimeter 12 has an area of 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → c + a > b → 
  (a * b : ℚ) / 2 = 6 :=
sorry

end triangle_area_l3496_349601


namespace b_can_complete_in_27_days_l3496_349658

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℕ := 15

/-- The number of days A actually works -/
def a_worked_days : ℕ := 5

/-- The number of days B needs to complete the remaining work after A leaves -/
def b_remaining_days : ℕ := 18

/-- The fraction of work completed by A -/
def a_work_fraction : ℚ := a_worked_days / a_total_days

/-- The fraction of work completed by B -/
def b_work_fraction : ℚ := 1 - a_work_fraction

/-- The number of days B needs to complete the entire work alone -/
def b_total_days : ℚ := b_remaining_days / b_work_fraction

theorem b_can_complete_in_27_days : b_total_days = 27 := by
  sorry

end b_can_complete_in_27_days_l3496_349658


namespace loan_principal_is_1200_l3496_349682

/-- Calculates the principal amount of a loan given the interest rate, time period, and total interest paid. -/
def calculate_principal (rate : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that under the given conditions, the loan principal is $1200. -/
theorem loan_principal_is_1200 :
  let rate : ℚ := 4
  let time : ℚ := rate
  let interest : ℚ := 192
  calculate_principal rate time interest = 1200 := by sorry

end loan_principal_is_1200_l3496_349682


namespace orange_count_l3496_349628

theorem orange_count (b t o : ℕ) : 
  (b + t) / 2 = 89 →
  (b + t + o) / 3 = 91 →
  o = 95 := by
sorry

end orange_count_l3496_349628


namespace simplify_expression_l3496_349659

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end simplify_expression_l3496_349659


namespace root_form_sum_l3496_349611

/-- The cubic polynomial 2x^3 + 3x^2 - 5x - 2 = 0 has a real root of the form (∛p + ∛q + 2)/r 
    where p, q, and r are positive integers. -/
def has_root_of_form (p q r : ℕ+) : Prop :=
  ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧
           x = (Real.rpow p (1/3 : ℝ) + Real.rpow q (1/3 : ℝ) + 2) / r

/-- If the cubic polynomial has a root of the specified form, then p + q + r = 10. -/
theorem root_form_sum (p q r : ℕ+) : has_root_of_form p q r → p + q + r = 10 := by
  sorry

end root_form_sum_l3496_349611


namespace underlined_numbers_are_correct_l3496_349615

def sequence_term (n : ℕ) : ℕ := 3 * n - 2

def has_same_digits (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 ∧ d₂ ∈ n.digits 10 → d₁ = d₂

def underlined_numbers : Set ℕ :=
  {n | ∃ k, sequence_term k = n ∧ 10 < n ∧ n < 100000 ∧ has_same_digits n}

theorem underlined_numbers_are_correct : underlined_numbers = 
  {22, 55, 88, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 
   11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999} := by
  sorry

end underlined_numbers_are_correct_l3496_349615
