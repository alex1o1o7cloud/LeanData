import Mathlib

namespace largest_consecutive_integers_sum_sixty_consecutive_integers_sum_largest_n_is_sixty_l726_72694

theorem largest_consecutive_integers_sum (n : ℕ) : 
  (∃ a : ℕ, a > 0 ∧ n * (2 * a + n - 1) = 4020) → n ≤ 60 :=
by
  sorry

theorem sixty_consecutive_integers_sum : 
  ∃ a : ℕ, a > 0 ∧ 60 * (2 * a + 60 - 1) = 4020 :=
by
  sorry

theorem largest_n_is_sixty : 
  ∀ n : ℕ, (∃ a : ℕ, a > 0 ∧ n * (2 * a + n - 1) = 4020) → n ≤ 60 :=
by
  sorry

end largest_consecutive_integers_sum_sixty_consecutive_integers_sum_largest_n_is_sixty_l726_72694


namespace problem_1_l726_72676

theorem problem_1 : (1 : ℝ) - 1^4 - 1/2 * (3 - (-3)^2) = 2 := by sorry

end problem_1_l726_72676


namespace sparse_characterization_l726_72603

/-- A number s grows to r if there exists some integer n > 0 such that s^n = r -/
def GrowsTo (s r : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ s^n = r

/-- A real number r is sparse if there are only finitely many real numbers s that grow to r -/
def Sparse (r : ℝ) : Prop :=
  Set.Finite {s : ℝ | GrowsTo s r}

/-- The characterization of sparse real numbers -/
theorem sparse_characterization (r : ℝ) : Sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := by
  sorry

end sparse_characterization_l726_72603


namespace polynomial_coefficients_l726_72626

/-- The polynomial f(x) = ax^4 - 7x^3 + bx^2 - 12x - 8 -/
def f (a b x : ℝ) : ℝ := a * x^4 - 7 * x^3 + b * x^2 - 12 * x - 8

/-- Theorem stating that if f(2) = -7 and f(-3) = -80, then a = -9/4 and b = 29.25 -/
theorem polynomial_coefficients (a b : ℝ) :
  f a b 2 = -7 ∧ f a b (-3) = -80 → a = -9/4 ∧ b = 29.25 := by
  sorry

#check polynomial_coefficients

end polynomial_coefficients_l726_72626


namespace quadratic_inequality_l726_72624

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 8 * x + 5 > 0 ↔ x < -1/3 := by sorry

end quadratic_inequality_l726_72624


namespace min_product_given_sum_l726_72689

theorem min_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 8 → a * b ≥ 16 := by
  sorry

end min_product_given_sum_l726_72689


namespace flight_cost_B_to_C_l726_72601

/-- Represents a city in a triangular configuration -/
inductive City
| A
| B
| C

/-- Represents the distance between two cities in kilometers -/
def distance (x y : City) : ℝ :=
  match x, y with
  | City.A, City.C => 3000
  | City.B, City.C => 1000
  | _, _ => 0  -- We don't need other distances for this problem

/-- The booking fee for a flight in dollars -/
def bookingFee : ℝ := 100

/-- The cost per kilometer for a flight in dollars -/
def costPerKm : ℝ := 0.1

/-- Calculates the cost of a flight between two cities -/
def flightCost (x y : City) : ℝ :=
  bookingFee + costPerKm * distance x y

/-- States that cities A, B, and C form a right-angled triangle with C as the right angle -/
axiom right_angle_at_C : distance City.A City.B ^ 2 = distance City.A City.C ^ 2 + distance City.B City.C ^ 2

theorem flight_cost_B_to_C :
  flightCost City.B City.C = 200 := by
  sorry

end flight_cost_B_to_C_l726_72601


namespace easter_egg_distribution_l726_72628

theorem easter_egg_distribution (baskets : ℕ) (eggs_per_basket : ℕ) (people : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  people = 20 →
  (baskets * eggs_per_basket) / people = 9 := by
sorry

end easter_egg_distribution_l726_72628


namespace product_of_roots_l726_72619

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 - 12 * a + 9 = 0) →
  (3 * b^3 - 4 * b^2 - 12 * b + 9 = 0) →
  (3 * c^3 - 4 * c^2 - 12 * c + 9 = 0) →
  a * b * c = -3 := by
sorry

end product_of_roots_l726_72619


namespace count_divisible_numbers_count_divisible_numbers_proof_l726_72684

theorem count_divisible_numbers : ℕ → Prop :=
  fun n => 
    (∃ (S : Finset ℕ), 
      (∀ x ∈ S, 1000 ≤ x ∧ x ≤ 3000 ∧ 12 ∣ x ∧ 18 ∣ x ∧ 24 ∣ x) ∧
      (∀ x, 1000 ≤ x ∧ x ≤ 3000 ∧ 12 ∣ x ∧ 18 ∣ x ∧ 24 ∣ x → x ∈ S) ∧
      S.card = n) →
    n = 28

theorem count_divisible_numbers_proof : count_divisible_numbers 28 := by
  sorry

end count_divisible_numbers_count_divisible_numbers_proof_l726_72684


namespace potato_chips_count_l726_72671

/-- The number of potato chips one potato can make -/
def potato_chips_per_potato (total_potatoes wedge_potatoes wedges_per_potato : ℕ) 
  (chip_wedge_difference : ℕ) : ℕ :=
let remaining_potatoes := total_potatoes - wedge_potatoes
let chip_potatoes := remaining_potatoes / 2
let total_wedges := wedge_potatoes * wedges_per_potato
let total_chips := total_wedges + chip_wedge_difference
total_chips / chip_potatoes

/-- Theorem stating that one potato can make 20 potato chips under given conditions -/
theorem potato_chips_count : 
  potato_chips_per_potato 67 13 8 436 = 20 := by
  sorry

end potato_chips_count_l726_72671


namespace cubic_roots_expression_l726_72661

theorem cubic_roots_expression (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  p^3 + q^3 + r^3 - 3*p*q*r = 18 := by
sorry

end cubic_roots_expression_l726_72661


namespace x_equals_five_l726_72670

theorem x_equals_five (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 := by
  sorry

end x_equals_five_l726_72670


namespace complex_fraction_simplification_l726_72666

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31 / 13 - (1 / 13) * Complex.I :=
by sorry

end complex_fraction_simplification_l726_72666


namespace arithmetic_sequence_sum_l726_72691

theorem arithmetic_sequence_sum : 
  ∀ (a l : ℤ) (d : ℤ) (n : ℕ),
    a = 162 →
    d = -6 →
    l = 48 →
    n > 0 →
    l = a + (n - 1) * d →
    (n : ℤ) * (a + l) / 2 = 2100 :=
by sorry

end arithmetic_sequence_sum_l726_72691


namespace pizza_order_l726_72610

theorem pizza_order (cost_per_box : ℚ) (tip_ratio : ℚ) (total_paid : ℚ) : 
  cost_per_box = 7 →
  tip_ratio = 1 / 7 →
  total_paid = 40 →
  ∃ (num_boxes : ℕ), 
    (↑num_boxes * cost_per_box) * (1 + tip_ratio) = total_paid ∧
    num_boxes = 5 := by
  sorry

end pizza_order_l726_72610


namespace foggy_day_walk_l726_72642

/-- Represents a person walking on a straight road -/
structure Walker where
  speed : ℝ
  position : ℝ

/-- The problem setup and solution -/
theorem foggy_day_walk (visibility : ℝ) (alex ben : Walker) (initial_time : ℝ) :
  visibility = 100 →
  alex.speed = 4 →
  ben.speed = 6 →
  initial_time = 60 →
  alex.position = alex.speed * initial_time →
  ben.position = ben.speed * initial_time →
  ∃ (meeting_time : ℝ),
    meeting_time = 50 ∧
    abs (alex.position - alex.speed * meeting_time - (ben.position - ben.speed * meeting_time)) = visibility ∧
    abs (alex.position - alex.speed * meeting_time) = 40 ∧
    abs (ben.position - ben.speed * meeting_time) = 60 :=
by sorry

end foggy_day_walk_l726_72642


namespace freddy_age_l726_72621

/-- Represents the ages of three children --/
structure ChildrenAges where
  matthew : ℕ
  rebecca : ℕ
  freddy : ℕ

/-- The conditions of the problem --/
def problem_conditions (ages : ChildrenAges) : Prop :=
  ages.matthew + ages.rebecca + ages.freddy = 35 ∧
  ages.matthew = ages.rebecca + 2 ∧
  ages.freddy = ages.matthew + 4

/-- The theorem stating that under the given conditions, Freddy is 15 years old --/
theorem freddy_age (ages : ChildrenAges) : 
  problem_conditions ages → ages.freddy = 15 := by
  sorry


end freddy_age_l726_72621


namespace angle_A_measure_l726_72609

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  true

-- Define the measure of an angle
def angle_measure (A B C : ℝ × ℝ) : ℝ :=
  sorry

-- Define the length of a side
def side_length (A B : ℝ × ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem angle_A_measure 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_acute : angle_measure A B C < π / 2 ∧ 
             angle_measure B C A < π / 2 ∧ 
             angle_measure C A B < π / 2)
  (h_BC : side_length B C = 3)
  (h_AB : side_length A B = Real.sqrt 6)
  (h_angle_C : angle_measure B C A = π / 4) :
  angle_measure C A B = π / 3 :=
sorry

end angle_A_measure_l726_72609


namespace tonya_initial_stamps_proof_l726_72652

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- The number of stamps Tonya has left after the trade -/
def tonya_stamps_left : ℕ := 3

/-- The initial number of stamps Tonya had -/
def tonya_initial_stamps : ℕ := 13

theorem tonya_initial_stamps_proof :
  tonya_initial_stamps = 
    (jimmy_matchbooks * matches_per_matchbook / matches_per_stamp) + tonya_stamps_left :=
by sorry

end tonya_initial_stamps_proof_l726_72652


namespace geometric_sequence_sum_l726_72665

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_two : a 1 + a 2 = 324
  sum_third_fourth : a 3 + a 4 = 36

/-- The theorem to be proved -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 4 := by
  sorry

end geometric_sequence_sum_l726_72665


namespace rectangle_perimeter_area_equality_l726_72688

theorem rectangle_perimeter_area_equality (k : ℝ) (h : k > 0) :
  (∃ w : ℝ, w > 0 ∧ 
    8 * w = k ∧  -- Perimeter equals k
    3 * w^2 = k) -- Area equals k
  → k = 64 / 3 := by
sorry

end rectangle_perimeter_area_equality_l726_72688


namespace sum_of_coefficients_l726_72673

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + 2*a₁*x + 4*a₂*x^2 + 8*a₃*x^3 + 16*a₄*x^4 + 32*a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end sum_of_coefficients_l726_72673


namespace triangle_base_length_l726_72607

/-- Given a square with perimeter 40 and a triangle with height 40 that share a side and have equal areas, 
    the base of the triangle is 5. -/
theorem triangle_base_length : 
  ∀ (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ),
    square_perimeter = 40 →
    triangle_height = 40 →
    (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
    triangle_base = 5 := by
  sorry

end triangle_base_length_l726_72607


namespace sqrt_fraction_sum_equals_sqrt_865_over_21_l726_72681

theorem sqrt_fraction_sum_equals_sqrt_865_over_21 :
  Real.sqrt (9 / 49 + 16 / 9) = Real.sqrt 865 / 21 := by
  sorry

end sqrt_fraction_sum_equals_sqrt_865_over_21_l726_72681


namespace hotel_towels_l726_72638

/-- Calculates the total number of towels handed out in a hotel --/
def total_towels (num_rooms : ℕ) (people_per_room : ℕ) (towels_per_person : ℕ) : ℕ :=
  num_rooms * people_per_room * towels_per_person

/-- Proves that a hotel with 10 full rooms, 3 people per room, and 2 towels per person hands out 60 towels --/
theorem hotel_towels : total_towels 10 3 2 = 60 := by
  sorry

end hotel_towels_l726_72638


namespace union_of_sets_l726_72639

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {1, 3, 5, 7}
  A ∪ B = {1, 2, 3, 4, 5, 7} := by
  sorry

end union_of_sets_l726_72639


namespace bridge_length_calculation_l726_72637

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 400 →
  crossing_time = 45 →
  train_speed = 55.99999999999999 →
  ∃ (bridge_length : ℝ), bridge_length = train_speed * crossing_time - train_length ∧
                         bridge_length = 2120 := by
  sorry

end bridge_length_calculation_l726_72637


namespace object_speed_mph_l726_72682

-- Define the distance traveled in feet
def distance_feet : ℝ := 400

-- Define the time traveled in seconds
def time_seconds : ℝ := 4

-- Define the conversion factor from feet to miles
def feet_per_mile : ℝ := 5280

-- Define the conversion factor from seconds to hours
def seconds_per_hour : ℝ := 3600

-- Theorem statement
theorem object_speed_mph :
  let distance_miles := distance_feet / feet_per_mile
  let time_hours := time_seconds / seconds_per_hour
  let speed_mph := distance_miles / time_hours
  ∃ ε > 0, |speed_mph - 68.18| < ε :=
sorry

end object_speed_mph_l726_72682


namespace mr_green_potato_yield_l726_72678

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (length_steps : ℝ) * feet_per_step * (width_steps : ℝ) * feet_per_step * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 3 (3/4) = 3037.5 := by
  sorry

end mr_green_potato_yield_l726_72678


namespace parabola_vertex_y_coordinate_l726_72663

/-- The y-coordinate of the vertex of the parabola y = -2x^2 + 16x + 72 is 104 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := -2 * x^2 + 16 * x + 72
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = 104 :=
by sorry

end parabola_vertex_y_coordinate_l726_72663


namespace min_value_expression_l726_72656

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 ≥ (4 : ℝ)^(1/3) / 2 := by
  sorry

end min_value_expression_l726_72656


namespace tan_four_theta_l726_72635

theorem tan_four_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 := by
  sorry

end tan_four_theta_l726_72635


namespace fraction_sign_change_l726_72657

theorem fraction_sign_change (a b : ℝ) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  sorry

end fraction_sign_change_l726_72657


namespace inverse_variation_problems_l726_72683

/-- Two real numbers vary inversely if their product is constant -/
def VaryInversely (r s : ℝ) : Prop :=
  ∃ k : ℝ, ∀ r' s', r' * s' = k

theorem inverse_variation_problems
  (h : VaryInversely r s)
  (h1 : r = 1500 ↔ s = 0.25) :
  (r = 3000 → s = 0.125) ∧ (s = 0.15 → r = 2500) := by
  sorry

end inverse_variation_problems_l726_72683


namespace celenes_borrowed_books_l726_72699

/-- Represents the problem of determining the number of books Celine borrowed -/
theorem celenes_borrowed_books :
  let daily_charge : ℚ := 0.5
  let days_for_first_book : ℕ := 20
  let days_in_may : ℕ := 31
  let total_paid : ℚ := 41
  let num_books_whole_month : ℕ := 2

  let charge_first_book : ℚ := daily_charge * days_for_first_book
  let charge_per_whole_month_book : ℚ := daily_charge * days_in_may
  let charge_whole_month_books : ℚ := num_books_whole_month * charge_per_whole_month_book
  let total_charge : ℚ := charge_first_book + charge_whole_month_books

  total_charge = total_paid →
  num_books_whole_month + 1 = 3 :=
by sorry

end celenes_borrowed_books_l726_72699


namespace no_valid_base_6_digit_for_divisibility_by_7_l726_72674

theorem no_valid_base_6_digit_for_divisibility_by_7 :
  ∀ d : ℕ, d ≤ 5 → ¬(∃ k : ℤ, 652 + 42 * d = 7 * k) := by
  sorry

end no_valid_base_6_digit_for_divisibility_by_7_l726_72674


namespace positive_operation_on_negative_two_l726_72633

theorem positive_operation_on_negative_two (op : ℝ → ℝ → ℝ) : 
  (op 1 (-2) > 0) → (1 - (-2) > 0) :=
by sorry

end positive_operation_on_negative_two_l726_72633


namespace smallest_colors_l726_72654

/-- A coloring of an infinite table -/
def InfiniteColoring (n : ℕ) := ℤ → ℤ → Fin n

/-- Predicate to check if a 2x3 or 3x2 rectangle has all different colors -/
def ValidRectangle (c : InfiniteColoring n) : Prop :=
  ∀ i j : ℤ, (
    (c i j ≠ c i (j+1) ∧ c i j ≠ c i (j+2) ∧ c i j ≠ c (i+1) j ∧ c i j ≠ c (i+1) (j+1) ∧ c i j ≠ c (i+1) (j+2)) ∧
    (c i (j+1) ≠ c i (j+2) ∧ c i (j+1) ≠ c (i+1) j ∧ c i (j+1) ≠ c (i+1) (j+1) ∧ c i (j+1) ≠ c (i+1) (j+2)) ∧
    (c i (j+2) ≠ c (i+1) j ∧ c i (j+2) ≠ c (i+1) (j+1) ∧ c i (j+2) ≠ c (i+1) (j+2)) ∧
    (c (i+1) j ≠ c (i+1) (j+1) ∧ c (i+1) j ≠ c (i+1) (j+2)) ∧
    (c (i+1) (j+1) ≠ c (i+1) (j+2))
  ) ∧ (
    (c i j ≠ c i (j+1) ∧ c i j ≠ c (i+1) j ∧ c i j ≠ c (i+2) j ∧ c i j ≠ c (i+1) (j+1) ∧ c i j ≠ c (i+2) (j+1)) ∧
    (c i (j+1) ≠ c (i+1) j ∧ c i (j+1) ≠ c (i+2) j ∧ c i (j+1) ≠ c (i+1) (j+1) ∧ c i (j+1) ≠ c (i+2) (j+1)) ∧
    (c (i+1) j ≠ c (i+2) j ∧ c (i+1) j ≠ c (i+1) (j+1) ∧ c (i+1) j ≠ c (i+2) (j+1)) ∧
    (c (i+2) j ≠ c (i+1) (j+1) ∧ c (i+2) j ≠ c (i+2) (j+1)) ∧
    (c (i+1) (j+1) ≠ c (i+2) (j+1))
  )

/-- The smallest number of colors needed is 8 -/
theorem smallest_colors : (∃ c : InfiniteColoring 8, ValidRectangle c) ∧ 
  (∀ n < 8, ¬∃ c : InfiniteColoring n, ValidRectangle c) :=
sorry

end smallest_colors_l726_72654


namespace oplus_properties_l726_72645

def oplus (a b : ℚ) : ℚ := a * b + 2 * a

theorem oplus_properties :
  (oplus 2 (-1) = 2) ∧
  (oplus (-3) (oplus (-4) (1/2)) = 24) := by
  sorry

end oplus_properties_l726_72645


namespace taxi_charge_calculation_l726_72692

/-- Calculates the additional charge per 2/5 of a mile for a taxi service -/
theorem taxi_charge_calculation (initial_fee : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (total_distance / (2/5)) = 0.35 := by
  sorry

end taxi_charge_calculation_l726_72692


namespace line_parametric_to_cartesian_l726_72677

/-- Given a line with parametric equations x = 1 + t/2 and y = 2 + (√3/2)t,
    its Cartesian equation is √3x - y + 2 - √3 = 0 --/
theorem line_parametric_to_cartesian :
  ∀ (x y t : ℝ),
  (x = 1 + t / 2 ∧ y = 2 + (Real.sqrt 3 / 2) * t) ↔
  (Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0) :=
by sorry

end line_parametric_to_cartesian_l726_72677


namespace complex_number_in_first_quadrant_l726_72644

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / (1 + Complex.I) + Complex.I) = Complex.mk a b := by
  sorry

end complex_number_in_first_quadrant_l726_72644


namespace quadratic_radical_simplification_l726_72660

theorem quadratic_radical_simplification (a m n : ℕ+) :
  (a : ℝ) + 2 * Real.sqrt 21 = (Real.sqrt (m : ℝ) + Real.sqrt (n : ℝ))^2 →
  a = 10 ∨ a = 22 := by
  sorry

end quadratic_radical_simplification_l726_72660


namespace barrel_capacity_l726_72606

/-- Prove that given the conditions, each barrel stores 2 gallons less than twice the capacity of a cask. -/
theorem barrel_capacity (num_barrels : ℕ) (cask_capacity : ℕ) (total_capacity : ℕ) :
  num_barrels = 4 →
  cask_capacity = 20 →
  total_capacity = 172 →
  (total_capacity - cask_capacity) / num_barrels = 2 * cask_capacity - 2 := by
  sorry


end barrel_capacity_l726_72606


namespace canoe_row_probability_value_l726_72693

def oar_probability : ℚ := 3/5

/-- The probability of being able to row a canoe with two independent oars -/
def canoe_row_probability : ℚ :=
  oar_probability * oar_probability +  -- Both oars work
  oar_probability * (1 - oar_probability) +  -- Left works, right breaks
  (1 - oar_probability) * oar_probability  -- Left breaks, right works

theorem canoe_row_probability_value :
  canoe_row_probability = 21/25 := by
  sorry

end canoe_row_probability_value_l726_72693


namespace loan_amounts_correct_l726_72698

-- Define the total loan amount in tens of thousands of yuan
def total_loan : ℝ := 68

-- Define the total annual interest in tens of thousands of yuan
def total_interest : ℝ := 8.42

-- Define the annual interest rate for Type A loan
def rate_A : ℝ := 0.12

-- Define the annual interest rate for Type B loan
def rate_B : ℝ := 0.13

-- Define the amount of Type A loan in tens of thousands of yuan
def loan_A : ℝ := 42

-- Define the amount of Type B loan in tens of thousands of yuan
def loan_B : ℝ := 26

theorem loan_amounts_correct : 
  loan_A + loan_B = total_loan ∧ 
  rate_A * loan_A + rate_B * loan_B = total_interest := by
  sorry

end loan_amounts_correct_l726_72698


namespace m_range_when_M_in_fourth_quadrant_l726_72662

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point M with coordinates dependent on m -/
def M (m : ℝ) : Point :=
  { x := m + 3, y := m - 1 }

/-- Theorem: If M(m) is in the fourth quadrant, then -3 < m < 1 -/
theorem m_range_when_M_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant (M m) → -3 < m ∧ m < 1 := by
  sorry

end m_range_when_M_in_fourth_quadrant_l726_72662


namespace geometric_sequence_third_term_l726_72667

/-- A geometric sequence of positive integers with first term 2 and fourth term 162 has third term 18 -/
theorem geometric_sequence_third_term : 
  ∀ (a : ℕ → ℕ) (r : ℕ),
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 1 = 2 →                     -- first term is 2
  a 4 = 162 →                   -- fourth term is 162
  a 3 = 18 :=                   -- third term is 18
by sorry

end geometric_sequence_third_term_l726_72667


namespace newcomer_weight_l726_72623

/-- Represents the weight of a group of people -/
structure GroupWeight where
  initial : ℝ
  new : ℝ

/-- The problem setup -/
def weightProblem (g : GroupWeight) : Prop :=
  -- Initial weight is between 400 kg and 420 kg
  400 ≤ g.initial ∧ g.initial ≤ 420 ∧
  -- The average weight increase is 3.5 kg
  g.new = g.initial - 47 + 68 ∧
  -- The average weight increases by 3.5 kg
  (g.new / 6) - (g.initial / 6) = 3.5

/-- The theorem to prove -/
theorem newcomer_weight (g : GroupWeight) : 
  weightProblem g → 68 = g.new - g.initial + 47 := by
  sorry


end newcomer_weight_l726_72623


namespace arithmetic_sequence_properties_l726_72651

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 51)
    (h2 : seq.a 1 + seq.a 9 = 26) :
  seq.d = 3 ∧ ∀ n, seq.a n = 3 * n - 2 := by
  sorry


end arithmetic_sequence_properties_l726_72651


namespace eight_by_eight_diagonal_shaded_count_l726_72647

/-- Represents a square grid with a diagonal shading pattern -/
structure DiagonalGrid where
  size : Nat
  shaded_rows : Nat
  shaded_per_row : Nat

/-- Calculates the total number of shaded squares in a DiagonalGrid -/
def total_shaded (grid : DiagonalGrid) : Nat :=
  grid.shaded_rows * grid.shaded_per_row

/-- Theorem stating that an 8×8 grid with 7 shaded rows and 7 shaded squares per row has 49 total shaded squares -/
theorem eight_by_eight_diagonal_shaded_count :
  ∀ (grid : DiagonalGrid),
    grid.size = 8 →
    grid.shaded_rows = 7 →
    grid.shaded_per_row = 7 →
    total_shaded grid = 49 := by
  sorry

end eight_by_eight_diagonal_shaded_count_l726_72647


namespace translation_downward_3_units_l726_72600

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Translates a linear function vertically -/
def translate_vertical (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + units }

theorem translation_downward_3_units :
  let original := LinearFunction.mk 3 2
  let translated := translate_vertical original (-3)
  translated = LinearFunction.mk 3 (-1) := by sorry

end translation_downward_3_units_l726_72600


namespace largest_palindrome_divisible_by_15_l726_72653

/-- A function that checks if a number is a 4-digit palindrome --/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- The largest 4-digit palindromic number divisible by 15 --/
def largest_palindrome : ℕ := 5775

/-- Sum of digits of a natural number --/
def digit_sum (n : ℕ) : ℕ :=
  let rec sum_digits (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc else sum_digits (m / 10) (acc + m % 10)
  sum_digits n 0

theorem largest_palindrome_divisible_by_15 :
  is_four_digit_palindrome largest_palindrome ∧
  largest_palindrome % 15 = 0 ∧
  (∀ n : ℕ, is_four_digit_palindrome n → n % 15 = 0 → n ≤ largest_palindrome) ∧
  digit_sum largest_palindrome = 24 := by
  sorry

end largest_palindrome_divisible_by_15_l726_72653


namespace relationship_one_l726_72617

theorem relationship_one (a b : ℝ) : (a - b)^2 + (a * b + 1)^2 = (a^2 + 1) * (b^2 + 1) := by
  sorry

end relationship_one_l726_72617


namespace fraction_evaluation_l726_72675

theorem fraction_evaluation : (3 : ℚ) / (1 - 2/5) = 5 := by sorry

end fraction_evaluation_l726_72675


namespace midpoint_chain_l726_72685

/-- Given a line segment AB with several midpoints, prove that AB = 96 -/
theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 3) →        -- AG = 3
  (B - A = 96) :=      -- AB = 96
by sorry

end midpoint_chain_l726_72685


namespace square_tiles_count_l726_72616

theorem square_tiles_count (total_tiles : ℕ) (total_edges : ℕ) (square_tiles : ℕ) (pentagonal_tiles : ℕ) :
  total_tiles = 30 →
  total_edges = 110 →
  total_tiles = square_tiles + pentagonal_tiles →
  4 * square_tiles + 5 * pentagonal_tiles = total_edges →
  square_tiles = 20 := by
  sorry

end square_tiles_count_l726_72616


namespace marks_spending_l726_72631

-- Define constants for item quantities
def notebooks : ℕ := 4
def pens : ℕ := 3
def books : ℕ := 1
def magazines : ℕ := 2

-- Define prices
def notebook_price : ℚ := 2
def pen_price : ℚ := 1.5
def book_price : ℚ := 12
def magazine_original_price : ℚ := 3

-- Define discount and coupon
def magazine_discount : ℚ := 0.25
def coupon_value : ℚ := 3
def coupon_threshold : ℚ := 20

-- Calculate discounted magazine price
def discounted_magazine_price : ℚ := magazine_original_price * (1 - magazine_discount)

-- Calculate total cost before coupon
def total_before_coupon : ℚ :=
  notebooks * notebook_price +
  pens * pen_price +
  books * book_price +
  magazines * discounted_magazine_price

-- Apply coupon if total is over the threshold
def final_total : ℚ :=
  if total_before_coupon ≥ coupon_threshold
  then total_before_coupon - coupon_value
  else total_before_coupon

-- Theorem to prove
theorem marks_spending :
  final_total = 26 := by sorry

end marks_spending_l726_72631


namespace correct_conclusions_l726_72613

theorem correct_conclusions :
  (∀ a b : ℝ, a + b < 0 ∧ b / a > 0 → |a + 2*b| = -a - 2*b) ∧
  (∀ m : ℚ, |m| + m ≥ 0) ∧
  (∀ a b c : ℝ, c < 0 ∧ 0 < a ∧ a < b → (a - b)*(b - c)*(c - a) > 0) :=
by sorry

end correct_conclusions_l726_72613


namespace oblique_triangular_prism_volume_l726_72690

/-- The volume of an oblique triangular prism -/
theorem oblique_triangular_prism_volume 
  (S d : ℝ) 
  (h_S : S > 0) 
  (h_d : d > 0) : 
  ∃ V : ℝ, V = (1/2) * d * S ∧ V > 0 := by
  sorry

end oblique_triangular_prism_volume_l726_72690


namespace inverse_proposition_l726_72696

theorem inverse_proposition : 
  (∀ a b : ℝ, a^2 + b^2 ≠ 0 → a = 0 ∧ b = 0) ↔ 
  (∀ a b : ℝ, a = 0 ∧ b = 0 → a^2 + b^2 ≠ 0) :=
by sorry

end inverse_proposition_l726_72696


namespace intersection_sum_l726_72630

/-- Two circles intersecting at (1, 3) and (m, n) with centers on x - y - 2 = 0 --/
structure IntersectingCircles where
  m : ℝ
  n : ℝ
  centers_on_line : ∀ (x y : ℝ), (x - y - 2 = 0) → (∃ (r : ℝ), (x - 1)^2 + (y - 3)^2 = r^2 ∧ (x - m)^2 + (y - n)^2 = r^2)

/-- The sum of coordinates of the second intersection point is 4 --/
theorem intersection_sum (c : IntersectingCircles) : c.m + c.n = 4 := by
  sorry

end intersection_sum_l726_72630


namespace eddie_earnings_l726_72620

-- Define the work hours for each day
def monday_hours : ℚ := 5/2
def tuesday_hours : ℚ := 7/6
def wednesday_hours : ℚ := 7/4
def saturday_hours : ℚ := 3/4

-- Define the pay rates
def weekday_rate : ℚ := 4
def saturday_rate : ℚ := 6

-- Define the total earnings
def total_earnings : ℚ := 
  monday_hours * weekday_rate + 
  tuesday_hours * weekday_rate + 
  wednesday_hours * weekday_rate + 
  saturday_hours * saturday_rate

-- Theorem to prove
theorem eddie_earnings : total_earnings = 26.17 := by
  sorry

end eddie_earnings_l726_72620


namespace gcf_of_lcms_l726_72680

-- Define LCM function
def LCM (a b : ℕ) : ℕ := sorry

-- Define GCF function
def GCF (a b : ℕ) : ℕ := sorry

-- Theorem statement
theorem gcf_of_lcms : GCF (LCM 9 15) (LCM 14 25) = 5 := by sorry

end gcf_of_lcms_l726_72680


namespace probability_four_primes_in_six_rolls_l726_72668

/-- The probability of getting exactly 4 prime numbers in 6 rolls of a fair 8-sided die -/
theorem probability_four_primes_in_six_rolls (die : Finset ℕ) 
  (h_die : die = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (h_prime : {n ∈ die | Nat.Prime n} = {2, 3, 5, 7}) : 
  (Nat.choose 6 4 * (4 / 8)^4 * (4 / 8)^2) = 15 / 64 := by
  sorry

end probability_four_primes_in_six_rolls_l726_72668


namespace parallel_vectors_y_value_l726_72632

/-- Given two parallel vectors a and b, prove that y = 4 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) :
  a = (2, 6) →
  b = (1, -1 + y) →
  ∃ (k : ℝ), a = k • b →
  y = 4 := by
sorry

end parallel_vectors_y_value_l726_72632


namespace unique_point_exists_l726_72634

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}

-- Define the diameter endpoints
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the conditions for point P
def IsValidP (p : ℝ × ℝ) : Prop :=
  p ∈ Circle ∧
  (p.1 - A.1)^2 + (p.2 - A.2)^2 + (p.1 - B.1)^2 + (p.2 - B.2)^2 = 10 ∧
  Real.cos (Real.arccos ((p.1 - A.1) * (p.1 - B.1) + (p.2 - A.2) * (p.2 - B.2)) /
    (Real.sqrt ((p.1 - A.1)^2 + (p.2 - A.2)^2) * Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2))) = 1/2

theorem unique_point_exists : ∃! p, IsValidP p :=
  sorry

end unique_point_exists_l726_72634


namespace characterization_of_function_l726_72605

theorem characterization_of_function (f : ℤ → ℝ) 
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a : ℝ, ∃ t : ℤ, a > 0 ∧ ∀ n : ℤ, f n = a * (n + t) := by
sorry

end characterization_of_function_l726_72605


namespace bisecting_line_min_value_l726_72618

/-- A line that bisects the circumference of a circle -/
structure BisetingLine where
  a : ℝ
  b : ℝ
  h1 : a ≥ b
  h2 : b > 0
  h3 : ∀ (x y : ℝ), a * x + 2 * b * y - 2 = 0 → x^2 + y^2 - 4*x - 2*y - 8 = 0

/-- The minimum value of 1/a + 2/b for a bisecting line is 6 -/
theorem bisecting_line_min_value (l : BisetingLine) : 
  (∀ (a' b' : ℝ), a' ≥ b' ∧ b' > 0 → 1 / a' + 2 / b' ≥ 1 / l.a + 2 / l.b) ∧
  1 / l.a + 2 / l.b = 6 :=
sorry

end bisecting_line_min_value_l726_72618


namespace complex_modulus_range_l726_72608

theorem complex_modulus_range (a : ℝ) :
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) →
  -Real.sqrt 5 / 5 ≤ a ∧ a ≤ Real.sqrt 5 / 5 := by
  sorry

end complex_modulus_range_l726_72608


namespace model_parameters_l726_72664

/-- Given a model y = c * e^(k * x) where c > 0, and its logarithmic transformation
    z = ln y resulting in the linear regression equation z = 2x - 1,
    prove that k = 2 and c = 1/e. -/
theorem model_parameters (c : ℝ) (k : ℝ) :
  c > 0 →
  (∀ x y z : ℝ, y = c * Real.exp (k * x) → z = Real.log y → z = 2 * x - 1) →
  k = 2 ∧ c = 1 / Real.exp 1 := by
sorry

end model_parameters_l726_72664


namespace complex_unit_circle_ab_range_l726_72687

theorem complex_unit_circle_ab_range (a b : ℝ) : 
  (Complex.abs (Complex.mk a b) = 1) → 
  (a * b ≥ -1/2 ∧ a * b ≤ 1/2) :=
by sorry

end complex_unit_circle_ab_range_l726_72687


namespace quadratic_equation_coefficients_l726_72604

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x * (x - 1) = 2 * (x + 2) + 8 ↔ a * x^2 + b * x + c = 0) →
  a = 3 ∧ b = -5 := by
  sorry

end quadratic_equation_coefficients_l726_72604


namespace sum_m_2n_3k_l726_72612

theorem sum_m_2n_3k (m n k : ℕ+) 
  (sum_mn : m + n = 2021)
  (prime_m_3k : Nat.Prime (m - 3*k))
  (prime_n_k : Nat.Prime (n + k)) :
  m + 2*n + 3*k = 2025 ∨ m + 2*n + 3*k = 4040 := by
  sorry

end sum_m_2n_3k_l726_72612


namespace quadratic_equation_solution_l726_72636

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 6*x + 5 = 0 ↔ x = -1 ∨ x = -5 := by
  sorry

end quadratic_equation_solution_l726_72636


namespace solution_comparison_l726_72611

theorem solution_comparison (p p' q q' : ℕ+) (hp : p ≠ p') (hq : q ≠ q') :
  (-q : ℚ) / p > (-q' : ℚ) / p' ↔ q * p' < p * q' :=
sorry

end solution_comparison_l726_72611


namespace sector_area_special_case_l726_72655

/-- The area of a sector with arc length and central angle both equal to 5 is 5/2 -/
theorem sector_area_special_case :
  ∀ (l α : ℝ), l = 5 → α = 5 → (1/2) * l * (l / α) = 5/2 := by
  sorry

end sector_area_special_case_l726_72655


namespace average_after_17th_is_40_l726_72672

/-- Represents a batsman's performance -/
structure Batsman where
  totalRunsBefore : ℕ  -- Total runs before the 17th inning
  inningsBefore : ℕ    -- Number of innings before the 17th inning (16)
  runsIn17th : ℕ       -- Runs scored in the 17th inning (88)
  averageIncrease : ℕ  -- Increase in average after 17th inning (3)

/-- Calculate the average score after the 17th inning -/
def averageAfter17th (b : Batsman) : ℚ :=
  (b.totalRunsBefore + b.runsIn17th) / (b.inningsBefore + 1)

/-- The main theorem to prove -/
theorem average_after_17th_is_40 (b : Batsman) 
    (h1 : b.inningsBefore = 16)
    (h2 : b.runsIn17th = 88) 
    (h3 : b.averageIncrease = 3)
    (h4 : averageAfter17th b = (b.totalRunsBefore / b.inningsBefore) + b.averageIncrease) :
  averageAfter17th b = 40 := by
  sorry


end average_after_17th_is_40_l726_72672


namespace square_area_percent_difference_l726_72622

theorem square_area_percent_difference (A B : ℝ) (h : A > B) :
  (A^2 - B^2) / B^2 * 100 = 100 * (A^2 - B^2) / B^2 := by
  sorry

end square_area_percent_difference_l726_72622


namespace tutors_next_meeting_l726_72629

/-- Anthony's work schedule in days -/
def anthony : ℕ := 5

/-- Beth's work schedule in days -/
def beth : ℕ := 6

/-- Carlos's work schedule in days -/
def carlos : ℕ := 8

/-- Diana's work schedule in days -/
def diana : ℕ := 10

/-- The number of days until all tutors work together again -/
def next_meeting : ℕ := 120

theorem tutors_next_meeting :
  Nat.lcm anthony (Nat.lcm beth (Nat.lcm carlos diana)) = next_meeting := by
  sorry

end tutors_next_meeting_l726_72629


namespace small_sphere_radius_l726_72650

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Checks if two spheres are externally tangent -/
def are_externally_tangent (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- The configuration of 5 spheres as described in the problem -/
structure SpheresConfiguration where
  s1 : Sphere
  s2 : Sphere
  s3 : Sphere
  s4 : Sphere
  small : Sphere
  h1 : s1.radius = 2
  h2 : s2.radius = 2
  h3 : s3.radius = 3
  h4 : s4.radius = 3
  h5 : are_externally_tangent s1 s2
  h6 : are_externally_tangent s1 s3
  h7 : are_externally_tangent s1 s4
  h8 : are_externally_tangent s2 s3
  h9 : are_externally_tangent s2 s4
  h10 : are_externally_tangent s3 s4
  h11 : are_externally_tangent s1 small
  h12 : are_externally_tangent s2 small
  h13 : are_externally_tangent s3 small
  h14 : are_externally_tangent s4 small

/-- The main theorem stating that the radius of the small sphere is 6/11 -/
theorem small_sphere_radius (config : SpheresConfiguration) : config.small.radius = 6/11 := by
  sorry

end small_sphere_radius_l726_72650


namespace total_cost_to_fill_displays_l726_72643

-- Define the jewelry types
inductive JewelryType
| Necklace
| Ring
| Bracelet

-- Define the structure for jewelry information
structure JewelryInfo where
  capacity : Nat
  current : Nat
  price : Nat
  discountRules : List (Nat × Nat)

-- Define the jewelry store inventory
def inventory : JewelryType → JewelryInfo
| JewelryType.Necklace => ⟨12, 5, 4, [(4, 10), (6, 15)]⟩
| JewelryType.Ring => ⟨30, 18, 10, [(10, 5), (20, 10)]⟩
| JewelryType.Bracelet => ⟨15, 8, 5, [(7, 8), (10, 12)]⟩

-- Function to calculate the discounted price
def calculateDiscountedPrice (info : JewelryInfo) (quantity : Nat) : Nat :=
  let totalPrice := quantity * info.price
  let applicableDiscount := info.discountRules.foldl
    (fun acc (threshold, discount) => if quantity ≥ threshold then max acc discount else acc)
    0
  totalPrice - totalPrice * applicableDiscount / 100

-- Theorem statement
theorem total_cost_to_fill_displays :
  (calculateDiscountedPrice (inventory JewelryType.Necklace) (12 - 5)) +
  (calculateDiscountedPrice (inventory JewelryType.Ring) (30 - 18)) +
  (calculateDiscountedPrice (inventory JewelryType.Bracelet) (15 - 8)) = 170 := by
  sorry


end total_cost_to_fill_displays_l726_72643


namespace sequence_properties_and_sum_l726_72627

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

def geometric_sequence (b₁ : ℕ) (q : ℕ) : ℕ → ℕ
  | n => b₁ * q^(n - 1)

def merge_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  sorry

theorem sequence_properties_and_sum :
  ∀ (a b : ℕ → ℕ),
    (a 1 = 1) →
    (∀ n, a (b n) = 2^(n+1) - 1) →
    (∀ n, a n = 2*n - 1) →
    (∀ n, b n = 2^n) →
    merge_and_sum a b 100 = 8903 :=
sorry

end sequence_properties_and_sum_l726_72627


namespace number_of_women_is_six_l726_72602

/-- The number of women in a group that can color 360 meters of cloth in 3 days,
    given that 5 women can color 100 meters of cloth in 1 day. -/
def number_of_women : ℕ :=
  let meters_per_day := 360 / 3
  let meters_per_woman_per_day := 100 / 5
  meters_per_day / meters_per_woman_per_day

theorem number_of_women_is_six : number_of_women = 6 := by
  sorry

end number_of_women_is_six_l726_72602


namespace complex_arithmetic_equality_l726_72695

theorem complex_arithmetic_equality : (469157 * 9999)^2 / 53264 + 3758491 = 413303758491 := by
  sorry

end complex_arithmetic_equality_l726_72695


namespace arithmetic_and_geometric_sequences_existence_l726_72679

theorem arithmetic_and_geometric_sequences_existence :
  ∃ (a b c : ℝ) (d r : ℝ),
    d ≠ 0 ∧ r ≠ 0 ∧ r ≠ 1 ∧
    (b - a = d ∧ c - b = d) ∧
    (∃ (x y : ℝ), x * r = y ∧ y * r = a ∧ a * r = b ∧ b * r = c) ∧
    ((a * r = b ∧ b * r = c) ∨ (b * r = a ∧ a * r = c) ∨ (c * r = a ∧ a * r = b)) :=
by sorry

end arithmetic_and_geometric_sequences_existence_l726_72679


namespace midpoint_triangle_area_l726_72640

/-- The area of a triangle formed by midpoints in a square --/
theorem midpoint_triangle_area (s : ℝ) (h : s = 12) :
  let square_area := s^2
  let midpoint_triangle_area := s^2 / 8
  midpoint_triangle_area = 18 := by
  sorry

end midpoint_triangle_area_l726_72640


namespace range_of_m_satisfying_condition_l726_72697

theorem range_of_m_satisfying_condition :
  {m : ℝ | ∀ x : ℝ, m * x^2 - (3 - m) * x + 1 > 0 ∨ m * x > 0} = {m : ℝ | 1/9 < m ∧ m < 1} := by
  sorry

end range_of_m_satisfying_condition_l726_72697


namespace smallest_dual_palindrome_l726_72641

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome :
  ∃ (n : ℕ), n > 6 ∧ 
    isPalindrome n 2 ∧ 
    isPalindrome n 4 ∧ 
    (∀ m : ℕ, m > 6 ∧ m < n → ¬(isPalindrome m 2 ∧ isPalindrome m 4)) ∧
    n = 15 := by
  sorry

end smallest_dual_palindrome_l726_72641


namespace set_relations_l726_72646

open Set

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

theorem set_relations (m : ℝ) :
  (A m ⊆ B ↔ m < 2 ∨ m > 4) ∧
  (A m ∩ B = ∅ ↔ m ≤ 3) := by
  sorry

end set_relations_l726_72646


namespace second_quadrant_points_characterization_l726_72669

def second_quadrant_points : Set (ℤ × ℤ) :=
  {p | p.1 < 0 ∧ p.2 > 0 ∧ p.2 ≤ p.1 + 4}

theorem second_quadrant_points_characterization :
  second_quadrant_points = {(-1, 1), (-1, 2), (-1, 3), (-2, 1), (-2, 2), (-3, 1)} := by
  sorry

end second_quadrant_points_characterization_l726_72669


namespace solve_equation_1_solve_system_2_l726_72658

-- Equation (1)
theorem solve_equation_1 : 
  ∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 ↔ x = -1/4 := by sorry

-- System of equations (2)
theorem solve_system_2 : 
  ∃ x y : ℚ, (3 * x - 2 * y = 9 ∧ 2 * x + 3 * y = 19) ↔ (x = 5 ∧ y = 3) := by sorry

end solve_equation_1_solve_system_2_l726_72658


namespace largest_divisor_of_five_consecutive_integers_l726_72615

theorem largest_divisor_of_five_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℤ), (k * (k+1) * (k+2) * (k+3) * (k+4)) % n = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (l : ℤ), (l * (l+1) * (l+2) * (l+3) * (l+4)) % m ≠ 0) :=
by
  -- The proof goes here
  sorry

end largest_divisor_of_five_consecutive_integers_l726_72615


namespace quadratic_equation_roots_l726_72614

theorem quadratic_equation_roots (x : ℝ) : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ (x^2 - 4*x - 3 = 0 ↔ x = r₁ ∨ x = r₂) := by
  sorry

end quadratic_equation_roots_l726_72614


namespace hyperbola_focus_smaller_x_l726_72649

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  center : Point

/-- Returns true if the given point is a focus of the hyperbola -/
def isFocus (h : Hyperbola) (p : Point) : Prop :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  (p.x = h.center.x - c ∧ p.y = h.center.y) ∨
  (p.x = h.center.x + c ∧ p.y = h.center.y)

/-- Returns true if p1 has a smaller x-coordinate than p2 -/
def hasSmaller_x (p1 p2 : Point) : Prop :=
  p1.x < p2.x

theorem hyperbola_focus_smaller_x (h : Hyperbola) :
  h.a = 7 ∧ h.b = 3 ∧ h.center = { x := 1, y := -8 } →
  ∃ (f : Point), isFocus h f ∧ ∀ (f' : Point), isFocus h f' → hasSmaller_x f f' ∨ f = f' →
  f = { x := 1 - Real.sqrt 58, y := -8 } := by
  sorry

end hyperbola_focus_smaller_x_l726_72649


namespace mini_bank_withdrawal_l726_72648

theorem mini_bank_withdrawal (d c : ℕ) : 
  (0 < c) → (c < 100) →
  (100 * c + d - 350 = 2 * (100 * d + c)) →
  (d = 14 ∧ c = 32) := by
sorry

end mini_bank_withdrawal_l726_72648


namespace max_a_for_integer_solutions_l726_72659

theorem max_a_for_integer_solutions : 
  (∃ (a : ℕ+), ∀ (x : ℤ), x^2 + a*x = -30 → 
    (∀ (b : ℕ+), (∀ (y : ℤ), y^2 + b*y = -30 → b ≤ a))) ∧
  (∃ (x : ℤ), x^2 + 31*x = -30) :=
by sorry

end max_a_for_integer_solutions_l726_72659


namespace quadratic_discriminant_relationship_l726_72625

/-- The discriminant of a quadratic equation ax^2 + 2bx + c = 0 is 1 -/
def discriminant_is_one (a b c : ℝ) : Prop :=
  (2 * b)^2 - 4 * a * c = 1

/-- The relationship between a, b, and c -/
def relationship (a b c : ℝ) : Prop :=
  b^2 - a * c = 1/4

/-- Theorem: If the discriminant of ax^2 + 2bx + c = 0 is 1, 
    then b^2 - ac = 1/4 -/
theorem quadratic_discriminant_relationship 
  (a b c : ℝ) : discriminant_is_one a b c → relationship a b c := by
  sorry

end quadratic_discriminant_relationship_l726_72625


namespace more_stable_scores_lower_variance_problem_solution_l726_72686

/-- Represents an athlete with their test score variance -/
structure Athlete where
  name : String
  variance : ℝ

/-- Determines if an athlete has more stable test scores than another -/
def has_more_stable_scores (a b : Athlete) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with equal average scores, 
    the athlete with lower variance has more stable test scores -/
theorem more_stable_scores_lower_variance 
  (a b : Athlete) 
  (h_avg : ℝ) -- average score of both athletes
  (h_equal_avg : True) -- assumption that both athletes have equal average scores
  : has_more_stable_scores a b ↔ a.variance < b.variance :=
by sorry

/-- Application to the specific problem -/
def athlete_A : Athlete := ⟨"A", 0.024⟩
def athlete_B : Athlete := ⟨"B", 0.008⟩

theorem problem_solution : has_more_stable_scores athlete_B athlete_A :=
by sorry

end more_stable_scores_lower_variance_problem_solution_l726_72686
