import Mathlib

namespace sum_of_polynomials_l234_23464

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem sum_of_polynomials :
  ∀ x : ℝ, p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end sum_of_polynomials_l234_23464


namespace alyssa_total_games_l234_23435

/-- The total number of soccer games Alyssa attends over three years -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Theorem stating that Alyssa will attend 39 games in total -/
theorem alyssa_total_games :
  total_games 11 13 15 = 39 := by
  sorry

end alyssa_total_games_l234_23435


namespace telescope_visual_range_increase_l234_23425

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 80)
  (h2 : new_range = 150) :
  ((new_range - original_range) / original_range) * 100 = 87.5 := by
  sorry

end telescope_visual_range_increase_l234_23425


namespace problem_1_problem_2_problem_3_problem_4_l234_23446

-- Problem 1
theorem problem_1 (m n : ℝ) (hm : m ≠ 0) :
  (2 * m * n) / (3 * m^2) * (6 * m * n) / (5 * n) = 4 * n / 5 :=
sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x ≠ 0) (hy : x ≠ y) :
  (5 * x - 5 * y) / (3 * x^2 * y) * (9 * x * y^2) / (x^2 - y^2) = 15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem problem_3 (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  ((x^3 * y^2) / z)^2 * ((y * z) / x^2)^3 = y^7 * z :=
sorry

-- Problem 4
theorem problem_4 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 2*x + y ≠ 0) (hxy2 : 4*x^2 - y^2 ≠ 0) :
  (4 * x^2 * y^2) / (2*x + y) * (4*x^2 + 4*x*y + y^2) / (2*x + y) / ((2*x*y * (2*x - y)) / (4*x^2 - y^2)) = 4*x^2*y + 2*x*y^2 :=
sorry

end problem_1_problem_2_problem_3_problem_4_l234_23446


namespace book_reading_theorem_l234_23449

def book_reading_problem (total_pages : ℕ) (reading_rate : ℕ) (monday_hours : ℕ) (tuesday_hours : ℚ) : ℚ :=
  let pages_read := monday_hours * reading_rate + tuesday_hours * reading_rate
  let pages_left := total_pages - pages_read
  pages_left / reading_rate

theorem book_reading_theorem :
  book_reading_problem 248 16 3 (13/2) = 6 := by
  sorry

end book_reading_theorem_l234_23449


namespace logarithm_square_sum_l234_23485

theorem logarithm_square_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a * b * c = 10^11) 
  (h2 : Real.log a * Real.log (b * c) + Real.log b * Real.log (c * a) + Real.log c * Real.log (a * b) = 40 * Real.log 10) : 
  Real.sqrt ((Real.log a)^2 + (Real.log b)^2 + (Real.log c)^2) = 9 * Real.log 10 := by
sorry

end logarithm_square_sum_l234_23485


namespace prob_even_and_greater_than_10_l234_23444

/-- Represents a wheel with even and odd numbers -/
structure Wheel where
  evenCount : ℕ
  oddCount : ℕ

/-- Calculates the probability of selecting an even number from a wheel -/
def probEven (w : Wheel) : ℚ :=
  w.evenCount / (w.evenCount + w.oddCount)

/-- Calculates the probability of selecting an odd number from a wheel -/
def probOdd (w : Wheel) : ℚ :=
  w.oddCount / (w.evenCount + w.oddCount)

/-- The wheels used in the problem -/
def wheelA : Wheel := ⟨3, 5⟩
def wheelB : Wheel := ⟨2, 6⟩

/-- The probability that the sum of selected numbers is even -/
def probEvenSum : ℚ :=
  probEven wheelA * probEven wheelB + probOdd wheelA * probOdd wheelB

/-- The conditional probability that an even sum is greater than 10 -/
def probGreaterThan10GivenEven : ℚ := 1/3

/-- The main theorem to prove -/
theorem prob_even_and_greater_than_10 :
  probEvenSum * probGreaterThan10GivenEven = 3/16 := by
  sorry


end prob_even_and_greater_than_10_l234_23444


namespace route_down_length_is_15_l234_23419

/-- Represents a hiking trip up and down a mountain -/
structure HikingTrip where
  rateUp : ℝ        -- Rate of hiking up the mountain in miles per day
  timeUp : ℝ        -- Time taken to hike up in days
  rateDownFactor : ℝ -- Factor by which the rate down is faster than the rate up

/-- Calculates the length of the route down the mountain -/
def routeDownLength (trip : HikingTrip) : ℝ :=
  trip.rateUp * trip.rateDownFactor * trip.timeUp

/-- Theorem stating that for the given conditions, the route down is 15 miles long -/
theorem route_down_length_is_15 : 
  ∀ (trip : HikingTrip), 
  trip.rateUp = 5 ∧ 
  trip.timeUp = 2 ∧ 
  trip.rateDownFactor = 1.5 → 
  routeDownLength trip = 15 := by
  sorry


end route_down_length_is_15_l234_23419


namespace max_y_value_l234_23434

theorem max_y_value (x y : ℝ) (h : (x + y)^4 = x - y) :
  y ≤ 3 * Real.rpow 2 (1/3) / 16 := by
  sorry

end max_y_value_l234_23434


namespace complex_number_coordinates_l234_23410

theorem complex_number_coordinates : 
  let i : ℂ := Complex.I
  let z : ℂ := i * (1 + i)
  z = -1 + i := by sorry

end complex_number_coordinates_l234_23410


namespace investment_growth_l234_23498

/-- Calculates the final amount of an investment after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that $1500 invested at 3% interest for 21 years results in approximately $2709.17 -/
theorem investment_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.03
  let years : ℕ := 21
  let final_amount := compound_interest principal rate years
  ∃ ε > 0, |final_amount - 2709.17| < ε :=
by sorry

end investment_growth_l234_23498


namespace toys_gained_example_l234_23417

/-- Calculates the number of toys' cost price gained in a sale -/
def toys_cost_price_gained (num_toys : ℕ) (selling_price : ℕ) (cost_price_per_toy : ℕ) : ℕ :=
  (selling_price - num_toys * cost_price_per_toy) / cost_price_per_toy

/-- The number of toys' cost price gained when selling 18 toys for Rs. 21000 with a cost price of Rs. 1000 per toy is 3 -/
theorem toys_gained_example : toys_cost_price_gained 18 21000 1000 = 3 := by
  sorry

end toys_gained_example_l234_23417


namespace point_B_coordinates_l234_23442

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem point_B_coordinates :
  let A : Point2D := ⟨2, -3⟩
  let AB_length : ℝ := 4
  let B_parallel_to_x_axis : ℝ → Prop := λ y => y = A.y
  ∃ (B : Point2D), (B.x = -2 ∨ B.x = 6) ∧ 
                   B_parallel_to_x_axis B.y ∧ 
                   ((B.x - A.x)^2 + (B.y - A.y)^2 = AB_length^2) :=
by sorry

end point_B_coordinates_l234_23442


namespace complex_number_problem_l234_23488

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = (Complex.I * (2 + a * Complex.I)) / (1 - Complex.I) →
  (∃ (b : ℝ), z = b * Complex.I) →
  z = 2 * Complex.I :=
by sorry

end complex_number_problem_l234_23488


namespace ages_sum_after_three_years_l234_23429

theorem ages_sum_after_three_years 
  (ava_age bob_age carlo_age : ℕ) 
  (h : ava_age + bob_age + carlo_age = 31) : 
  (ava_age + 3) + (bob_age + 3) + (carlo_age + 3) = 40 := by
  sorry

end ages_sum_after_three_years_l234_23429


namespace price_reduction_equality_l234_23460

theorem price_reduction_equality (z : ℝ) (h : z > 0) : 
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧ 
  (z * (1 - 15/100) * (1 - 15/100) = z * (1 - x/100)) ∧
  x = 27.75 := by
sorry

end price_reduction_equality_l234_23460


namespace rental_fee_calculation_l234_23470

/-- Rental fee calculation for comic books -/
theorem rental_fee_calculation 
  (rental_fee_per_30min : ℕ) 
  (num_students : ℕ) 
  (num_books : ℕ) 
  (rental_duration_hours : ℕ) 
  (h1 : rental_fee_per_30min = 4000)
  (h2 : num_students = 6)
  (h3 : num_books = 4)
  (h4 : rental_duration_hours = 3)
  : (rental_fee_per_30min * (rental_duration_hours * 2) * num_books) / num_students = 16000 := by
  sorry

#check rental_fee_calculation

end rental_fee_calculation_l234_23470


namespace divisor_and_totient_properties_l234_23413

/-- Sum of divisors function -/
def τ (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ (n : ℕ) : ℕ := sorry

theorem divisor_and_totient_properties (n : ℕ) :
  (n > 1 → φ n * τ n < n^2) ∧
  (φ n * τ n + 1 = n^2 ↔ Nat.Prime n) ∧
  ¬∃ (m : ℕ), φ m * τ m + 2023 = m^2 := by
  sorry

end divisor_and_totient_properties_l234_23413


namespace largest_three_digit_number_with_conditions_l234_23414

theorem largest_three_digit_number_with_conditions :
  ∃ n : ℕ,
    n = 960 ∧
    100 ≤ n ∧ n ≤ 999 ∧
    ∃ k : ℕ, n = 7 * k + 1 ∧
    ∃ m : ℕ, n = 8 * m + 4 ∧
    ∀ x : ℕ,
      (100 ≤ x ∧ x ≤ 999 ∧
       ∃ k' : ℕ, x = 7 * k' + 1 ∧
       ∃ m' : ℕ, x = 8 * m' + 4) →
      x ≤ n :=
by sorry

end largest_three_digit_number_with_conditions_l234_23414


namespace stirring_ensures_representativeness_l234_23474

/-- Represents the lottery method for sampling -/
structure LotteryMethod where
  /-- The action of stirring the lots -/
  stir : Bool

/-- Represents the representativeness of a sample -/
def representative (method : LotteryMethod) : Prop :=
  method.stir

/-- Theorem stating that stirring evenly is key to representativeness in the lottery method -/
theorem stirring_ensures_representativeness (method : LotteryMethod) :
  representative method ↔ method.stir :=
sorry

end stirring_ensures_representativeness_l234_23474


namespace intersection_with_complement_l234_23482

def U : Set Nat := {1, 2, 4, 6, 8}
def A : Set Nat := {1, 2, 4}
def B : Set Nat := {2, 4, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end intersection_with_complement_l234_23482


namespace smallest_integer_in_set_l234_23469

theorem smallest_integer_in_set (n : ℤ) : 
  (7 * n + 21 > 4 * n) → (∀ m : ℤ, m < n → ¬(7 * m + 21 > 4 * m)) → n = -6 :=
by sorry

end smallest_integer_in_set_l234_23469


namespace oldest_turner_child_age_l234_23479

theorem oldest_turner_child_age 
  (num_children : ℕ) 
  (average_age : ℕ) 
  (younger_children_ages : List ℕ) :
  num_children = 4 →
  average_age = 9 →
  younger_children_ages = [6, 8, 11] →
  (List.sum younger_children_ages + 11) / num_children = average_age :=
by sorry

end oldest_turner_child_age_l234_23479


namespace helga_shopping_items_l234_23445

def shopping_trip (store1_shoes store1_bags : ℕ) : Prop :=
  let store2_shoes := 2 * store1_shoes
  let store2_bags := store1_bags + 6
  let store3_shoes := 0
  let store3_bags := 0
  let store4_shoes := store1_bags + store2_bags
  let store4_bags := 0
  let store5_shoes := store4_shoes / 2
  let store5_bags := 8
  let store6_shoes := Int.floor (Real.sqrt (store2_shoes + store5_shoes))
  let store6_bags := store1_bags + store2_bags + store5_bags + 5
  let total_shoes := store1_shoes + store2_shoes + store3_shoes + store4_shoes + store5_shoes + store6_shoes
  let total_bags := store1_bags + store2_bags + store3_bags + store4_bags + store5_bags + store6_bags
  total_shoes + total_bags = 95

theorem helga_shopping_items :
  shopping_trip 7 4 := by
  sorry

end helga_shopping_items_l234_23445


namespace diane_gingerbreads_l234_23462

/-- The number of trays with 25 gingerbreads each -/
def trays_25 : ℕ := 4

/-- The number of gingerbreads in each of the first type of tray -/
def gingerbreads_per_tray_25 : ℕ := 25

/-- The number of trays with 20 gingerbreads each -/
def trays_20 : ℕ := 3

/-- The number of gingerbreads in each of the second type of tray -/
def gingerbreads_per_tray_20 : ℕ := 20

/-- The total number of gingerbreads Diane bakes -/
def total_gingerbreads : ℕ := trays_25 * gingerbreads_per_tray_25 + trays_20 * gingerbreads_per_tray_20

theorem diane_gingerbreads : total_gingerbreads = 160 := by
  sorry

end diane_gingerbreads_l234_23462


namespace logarithm_properties_l234_23475

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_properties (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (log b 1 = 0) ∧
  (log b b = 1) ∧
  (log b (1/b) = -1) ∧
  (∀ x : ℝ, 0 < x → x < 1 → log b x < 0) :=
by sorry

end logarithm_properties_l234_23475


namespace binary_addition_theorem_l234_23486

/-- Converts a binary number (represented as a list of bits) to decimal -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to binary (represented as a list of bits) -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec to_binary_aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
    to_binary_aux n

/-- The main theorem: 1010₂ + 10₂ = 1100₂ -/
theorem binary_addition_theorem : 
  decimal_to_binary (binary_to_decimal [false, true, false, true] + 
                     binary_to_decimal [false, true]) =
  [false, false, true, true] := by sorry

end binary_addition_theorem_l234_23486


namespace cubic_equation_solutions_l234_23420

theorem cubic_equation_solutions : 
  ∃! (s : Finset Int), 
    (∀ x ∈ s, (x^3 - x - 1)^2015 = 1) ∧ 
    (∀ x : Int, (x^3 - x - 1)^2015 = 1 → x ∈ s) ∧ 
    Finset.card s = 3 := by
  sorry

end cubic_equation_solutions_l234_23420


namespace janet_ticket_problem_l234_23427

/-- The number of tickets needed for one ride on the roller coaster -/
def roller_coaster_tickets : ℕ := 5

/-- The total number of tickets needed for 7 rides on the roller coaster and 4 rides on the giant slide -/
def total_tickets : ℕ := 47

/-- The number of roller coaster rides -/
def roller_coaster_rides : ℕ := 7

/-- The number of giant slide rides -/
def giant_slide_rides : ℕ := 4

/-- The number of tickets needed for one ride on the giant slide -/
def giant_slide_tickets : ℕ := 3

theorem janet_ticket_problem :
  roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides = total_tickets :=
sorry

end janet_ticket_problem_l234_23427


namespace rug_overlap_problem_l234_23409

theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (two_layer_area : ℝ)
  (h1 : total_area = 350)
  (h2 : covered_area = 250)
  (h3 : two_layer_area = 45) :
  total_area = covered_area + two_layer_area + 55 :=
by sorry

end rug_overlap_problem_l234_23409


namespace sector_to_cone_sector_forms_cone_l234_23404

theorem sector_to_cone (sector_angle : Real) (sector_radius : Real) 
  (base_radius : Real) (slant_height : Real) : Prop :=
  sector_angle = 240 ∧ 
  sector_radius = 12 ∧
  base_radius = 8 ∧
  slant_height = 12 ∧
  sector_angle / 360 * (2 * Real.pi * sector_radius) = 2 * Real.pi * base_radius ∧
  slant_height = sector_radius

theorem sector_forms_cone : 
  ∃ (sector_angle : Real) (sector_radius : Real) 
     (base_radius : Real) (slant_height : Real),
  sector_to_cone sector_angle sector_radius base_radius slant_height := by
  sorry

end sector_to_cone_sector_forms_cone_l234_23404


namespace first_act_clown_mobiles_l234_23433

/-- The number of clowns in each clown mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all clown mobiles -/
def total_clowns : ℕ := 140

/-- The number of clown mobiles -/
def num_clown_mobiles : ℕ := total_clowns / clowns_per_mobile

theorem first_act_clown_mobiles : num_clown_mobiles = 5 := by
  sorry

end first_act_clown_mobiles_l234_23433


namespace intersection_distance_l234_23437

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  Real.sqrt 3 * x - y + 3 = 0

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*y - 2*Real.sqrt 3*x = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ curve_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ curve_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 15 :=
sorry

end intersection_distance_l234_23437


namespace cylinder_volume_ratio_l234_23480

theorem cylinder_volume_ratio : 
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 10
  let cylinder_a_height : ℝ := rectangle_height
  let cylinder_a_circumference : ℝ := rectangle_width
  let cylinder_b_height : ℝ := rectangle_width
  let cylinder_b_circumference : ℝ := rectangle_height
  let cylinder_volume (h : ℝ) (c : ℝ) : ℝ := h * (c / (2 * π))^2 * π
  let volume_a := cylinder_volume cylinder_a_height cylinder_a_circumference
  let volume_b := cylinder_volume cylinder_b_height cylinder_b_circumference
  max volume_a volume_b / min volume_a volume_b = 5 / 3 := by
sorry

end cylinder_volume_ratio_l234_23480


namespace incorrect_statement_l234_23418

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) :
  ¬(∀ (α β : Plane) (m n : Line),
    perp m n → perpPlane n α → parallel n β → perpPlanes α β) :=
by sorry

end incorrect_statement_l234_23418


namespace pizza_and_burgers_theorem_l234_23447

/-- The number of pupils who like both pizza and burgers -/
def both_pizza_and_burgers (total : ℕ) (pizza : ℕ) (burgers : ℕ) : ℕ :=
  pizza + burgers - total

/-- Theorem: Given 200 total pupils, 125 who like pizza, and 115 who like burgers,
    40 pupils like both pizza and burgers. -/
theorem pizza_and_burgers_theorem :
  both_pizza_and_burgers 200 125 115 = 40 := by
  sorry

end pizza_and_burgers_theorem_l234_23447


namespace jason_car_count_l234_23406

/-- The number of red cars counted by Jason -/
def red_cars : ℕ := sorry

/-- The number of green cars counted by Jason -/
def green_cars : ℕ := sorry

/-- The number of purple cars counted by Jason -/
def purple_cars : ℕ := 47

theorem jason_car_count :
  (green_cars = 4 * red_cars) ∧
  (red_cars > purple_cars) ∧
  (green_cars + red_cars + purple_cars = 312) ∧
  (red_cars - purple_cars = 6) :=
by sorry

end jason_car_count_l234_23406


namespace simplify_expression_l234_23455

theorem simplify_expression : 
  1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
sorry

end simplify_expression_l234_23455


namespace root_sum_quotient_l234_23468

theorem root_sum_quotient (p p₁ p₂ a b : ℝ) : 
  (a^2 - a) * p + 2 * a + 7 = 0 →
  (b^2 - b) * p + 2 * b + 7 = 0 →
  a / b + b / a = 7 / 10 →
  (p₁^2 - p₁) * a + 2 * a + 7 = 0 →
  (p₁^2 - p₁) * b + 2 * b + 7 = 0 →
  (p₂^2 - p₂) * a + 2 * a + 7 = 0 →
  (p₂^2 - p₂) * b + 2 * b + 7 = 0 →
  p₁ / p₂ + p₂ / p₁ = 9.2225 := by
sorry

end root_sum_quotient_l234_23468


namespace regular_polygon_27_diagonals_has_9_sides_l234_23454

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon with 27 diagonals has 9 sides -/
theorem regular_polygon_27_diagonals_has_9_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 27 → n = 9 := by
  sorry

end regular_polygon_27_diagonals_has_9_sides_l234_23454


namespace sum_of_integers_l234_23493

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 2)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 10 := by
  sorry

end sum_of_integers_l234_23493


namespace q_satisfies_conditions_l234_23448

/-- The cubic polynomial q(x) that satisfies specific conditions -/
def q (x : ℚ) : ℚ := (51/13) * x^3 - (31/13) * x^2 + (16/13) * x + (3/13)

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions :
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 := by
  sorry

end q_satisfies_conditions_l234_23448


namespace function_properties_l234_23494

noncomputable def f (A ω φ B x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + B

theorem function_properties :
  ∀ (A ω φ B : ℝ),
  A > 0 → ω > 0 → 0 < φ → φ < π →
  f A ω φ B (π / 3) = 1 →
  f A ω φ B (π / 2 / ω - φ / ω) = 3 →
  ∃ (x : ℝ), ω * x + φ = 0 ∧
  ∃ (y : ℝ), ω * y + φ = π ∧
  ω * (7 * π / 12) + φ = 2 * π →
  A = 1 ∧ B = 2 ∧ ω = 2 ∧ φ = 5 * π / 6 ∧
  (∀ (x : ℝ), f A ω φ B x = f A ω φ B (-4 * π / 3 - x)) ∧
  (∀ (x : ℝ), f A ω φ B (x - 5 * π / 12) = 4 - f A ω φ B (x + 5 * π / 12)) :=
by sorry

end function_properties_l234_23494


namespace fencing_length_l234_23415

theorem fencing_length (area : ℝ) (uncovered_side : ℝ) (fencing_length : ℝ) : 
  area = 600 →
  uncovered_side = 20 →
  fencing_length = uncovered_side + 2 * (area / uncovered_side) →
  fencing_length = 80 := by
sorry

end fencing_length_l234_23415


namespace partial_fraction_decomposition_l234_23459

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 9 → x ≠ -4 →
    (7 * x + 3) / (x^2 - 5*x - 36) = (66/13) / (x - 9) + (25/13) / (x + 4) := by
  sorry

end partial_fraction_decomposition_l234_23459


namespace units_digit_of_p_l234_23461

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  p > 0 → 
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  units_digit (p + 1) = 7 →
  units_digit p = 6 := by sorry

end units_digit_of_p_l234_23461


namespace z_in_fourth_quadrant_l234_23441

-- Define the complex number z
variable (z : ℂ)

-- Define the equation z ⋅ (1+2i)² = 3+4i
def equation (z : ℂ) : Prop := z * (1 + 2*Complex.I)^2 = 3 + 4*Complex.I

-- Theorem statement
theorem z_in_fourth_quadrant (h : equation z) : 
  0 < z.re ∧ z.im < 0 := by sorry

end z_in_fourth_quadrant_l234_23441


namespace delivery_ratio_l234_23473

theorem delivery_ratio : 
  let meals : ℕ := 3
  let total : ℕ := 27
  let packages : ℕ := total - meals
  packages / meals = 8 := by
sorry

end delivery_ratio_l234_23473


namespace certain_fraction_proof_l234_23491

theorem certain_fraction_proof (x : ℚ) : 
  (2 / 5) / x = (7 / 15) / (1 / 2) → x = 3 / 7 := by
  sorry

end certain_fraction_proof_l234_23491


namespace power_sum_zero_l234_23492

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2^(3^2) = 0 := by
  sorry

end power_sum_zero_l234_23492


namespace delta_curve_circumscribed_triangle_height_l234_23457

/-- A Δ-curve is a curve with the property that all equilateral triangles circumscribing it have the same height -/
class DeltaCurve (α : Type*) [MetricSpace α] where
  is_delta_curve : α → Prop

variable {α : Type*} [MetricSpace α]

/-- An equilateral triangle -/
structure EquilateralTriangle (α : Type*) [MetricSpace α] where
  points : Fin 3 → α
  is_equilateral : ∀ i j : Fin 3, dist (points i) (points j) = dist (points 0) (points 1)

/-- A point lies on a line -/
def PointOnLine (p : α) (l : Set α) : Prop := p ∈ l

/-- A triangle circumscribes a curve if each side of the triangle touches the curve at exactly one point -/
def Circumscribes (t : EquilateralTriangle α) (k : Set α) : Prop :=
  ∃ a b c : α, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧
    PointOnLine a {x | dist x (t.points 0) = dist x (t.points 1)} ∧
    PointOnLine b {x | dist x (t.points 1) = dist x (t.points 2)} ∧
    PointOnLine c {x | dist x (t.points 2) = dist x (t.points 0)}

/-- The height of an equilateral triangle -/
def Height (t : EquilateralTriangle α) : ℝ := sorry

/-- The main theorem -/
theorem delta_curve_circumscribed_triangle_height 
  (k : Set α) [DeltaCurve α] (t : EquilateralTriangle α) 
  (h_circumscribes : Circumscribes t k) :
  ∀ (t₁ : EquilateralTriangle α),
    (∃ a b c : α, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧
      PointOnLine a {x | dist x (t₁.points 0) = dist x (t₁.points 1)} ∧
      PointOnLine b {x | dist x (t₁.points 1) = dist x (t₁.points 2)} ∧
      PointOnLine c {x | dist x (t₁.points 2) = dist x (t₁.points 0)}) →
    Height t₁ ≤ Height t :=
sorry

end delta_curve_circumscribed_triangle_height_l234_23457


namespace simplify_expression_l234_23476

theorem simplify_expression (x y : ℝ) :
  3 * x + 7 * x^2 + 4 * y - (5 - 3 * x - 7 * x^2 + 2 * y) = 14 * x^2 + 6 * x + 2 * y - 5 := by
  sorry

end simplify_expression_l234_23476


namespace inequality_proof_l234_23463

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) :
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 := by
sorry

end inequality_proof_l234_23463


namespace diophantine_equation_solvable_l234_23497

theorem diophantine_equation_solvable (a : ℕ+) :
  ∃ (x y : ℤ), x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end diophantine_equation_solvable_l234_23497


namespace function_highest_points_omega_range_l234_23407

/-- Given a function f(x) = 2sin(ωx + π/4) with ω > 0, if the graph of f(x) has exactly 3 highest points
    in the interval [0,1], then ω is in the range [17π/4, 25π/4). -/
theorem function_highest_points_omega_range (ω : ℝ) (h1 : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + π / 4)
  (∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 1) ∧
    (∀ y ∈ Set.Icc 0 1, ∃ x ∈ s, f y ≤ f x) ∧
    (∀ z ∉ s, z ∈ Set.Icc 0 1 → ∃ x ∈ s, f z < f x)) →
  17 * π / 4 ≤ ω ∧ ω < 25 * π / 4 := by
  sorry

end function_highest_points_omega_range_l234_23407


namespace flywheel_rotation_l234_23443

-- Define the angular displacement function
def φ (t : ℝ) : ℝ := 8 * t - 0.5 * t^2

-- Define the angular velocity function
def ω (t : ℝ) : ℝ := 8 - t

theorem flywheel_rotation (t : ℝ) :
  -- 1. The angular velocity is the derivative of the angular displacement
  (deriv φ) t = ω t ∧
  -- 2. The angular velocity at t = 3 seconds is 5 rad/s
  ω 3 = 5 ∧
  -- 3. The flywheel stops rotating at t = 8 seconds
  ω 8 = 0 := by
  sorry


end flywheel_rotation_l234_23443


namespace expression_value_l234_23401

theorem expression_value :
  ∀ (a b c d : ℤ),
    (∀ n : ℤ, n < 0 → a ≥ n) →  -- a is the largest negative integer
    (a < 0) →                   -- ensure a is negative
    (b = -c) →                  -- b and c are opposite numbers
    (d < 0) →                   -- d is negative
    (abs d = 2) →               -- absolute value of d is 2
    4*a + (b + c) - abs (3*d) = -10 :=
by
  sorry

end expression_value_l234_23401


namespace last_two_digits_of_product_l234_23402

theorem last_two_digits_of_product (k : ℕ) (h : k ≥ 5) :
  ∃ m : ℕ, (k + 1) * (k + 2) * (k + 3) * (k + 4) ≡ 24 [ZMOD 100] :=
sorry

end last_two_digits_of_product_l234_23402


namespace unique_three_digit_number_l234_23432

/-- A function that returns the digits of a three-digit number -/
def digits (n : ℕ) : Fin 3 → ℕ :=
  fun i => (n / (100 / 10^i.val)) % 10

/-- Check if three numbers form a geometric progression -/
def isGeometricProgression (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Check if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℕ) : Prop :=
  2 * b = a + c

/-- The main theorem -/
theorem unique_three_digit_number : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (let d := digits n
   isGeometricProgression (d 0) (d 1) (d 2) ∧
   d 0 ≠ d 1 ∧ d 1 ≠ d 2 ∧ d 0 ≠ d 2) ∧
  (let m := n - 200
   100 ≤ m ∧ m < 1000 ∧
   let d := digits m
   isArithmeticProgression (d 0) (d 1) (d 2)) ∧
  n = 842 :=
sorry

end unique_three_digit_number_l234_23432


namespace axis_of_symmetry_y₂_greater_y₁_l234_23428

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  t : ℝ
  y₁ : ℝ
  y₂ : ℝ
  h_a_pos : a > 0
  h_point : m = a * 2^2 + b * 2 + c
  h_axis : t = -b / (2 * a)
  h_y₁ : y₁ = a * (-1)^2 + b * (-1) + c
  h_y₂ : y₂ = a * 3^2 + b * 3 + c

/-- When m = c, the axis of symmetry is at x = 1 -/
theorem axis_of_symmetry (p : Parabola) (h : p.m = p.c) : p.t = 1 := by sorry

/-- When c < m, y₂ > y₁ -/
theorem y₂_greater_y₁ (p : Parabola) (h : p.c < p.m) : p.y₂ > p.y₁ := by sorry

end axis_of_symmetry_y₂_greater_y₁_l234_23428


namespace total_instruments_eq_113_l234_23487

/-- The total number of musical instruments owned by Charlie, Carli, Nick, and Daisy -/
def total_instruments (charlie_flutes charlie_horns charlie_harps charlie_drums : ℕ)
  (carli_flute_ratio carli_horn_ratio carli_drum_ratio : ℕ)
  (nick_flute_offset nick_horn_offset nick_drum_ratio nick_drum_offset : ℕ)
  (daisy_horn_denominator : ℕ) : ℕ :=
  let carli_flutes := charlie_flutes * carli_flute_ratio
  let carli_horns := charlie_horns / carli_horn_ratio
  let carli_drums := charlie_drums * carli_drum_ratio

  let nick_flutes := carli_flutes * 2 - nick_flute_offset
  let nick_horns := charlie_horns + carli_horns
  let nick_drums := carli_drums * nick_drum_ratio - nick_drum_offset

  let daisy_flutes := nick_flutes ^ 2
  let daisy_horns := (nick_horns - carli_horns) / daisy_horn_denominator
  let daisy_harps := charlie_harps
  let daisy_drums := (charlie_drums + carli_drums + nick_drums) / 3

  charlie_flutes + charlie_horns + charlie_harps + charlie_drums +
  carli_flutes + carli_horns + carli_drums +
  nick_flutes + nick_horns + nick_drums +
  daisy_flutes + daisy_horns + daisy_harps + daisy_drums

theorem total_instruments_eq_113 :
  total_instruments 1 2 1 5 3 2 2 1 0 4 2 2 = 113 := by
  sorry

end total_instruments_eq_113_l234_23487


namespace fifteen_percent_of_800_is_120_l234_23421

theorem fifteen_percent_of_800_is_120 :
  ∀ x : ℝ, (15 / 100) * x = 120 → x = 800 := by
  sorry

end fifteen_percent_of_800_is_120_l234_23421


namespace gcd_of_4536_13440_216_l234_23439

theorem gcd_of_4536_13440_216 : Nat.gcd 4536 (Nat.gcd 13440 216) = 216 := by
  sorry

end gcd_of_4536_13440_216_l234_23439


namespace substring_012_occurrences_l234_23495

/-- Base-3 representation of an integer without leading zeroes -/
def base3Repr (n : ℕ) : List ℕ := sorry

/-- Continuous string formed by joining base-3 representations of integers from 1 to 729 -/
def continuousString : List ℕ := sorry

/-- Count occurrences of a substring in a list -/
def countSubstring (list : List ℕ) (substring : List ℕ) : ℕ := sorry

theorem substring_012_occurrences :
  countSubstring continuousString [0, 1, 2] = 148 := by sorry

end substring_012_occurrences_l234_23495


namespace remainder_3456_div_23_l234_23453

theorem remainder_3456_div_23 : 3456 % 23 = 6 := by
  sorry

end remainder_3456_div_23_l234_23453


namespace range_of_a_l234_23451

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end range_of_a_l234_23451


namespace triangulation_theorem_l234_23484

/-- A triangulation of a convex polygon with interior points. -/
structure Triangulation where
  /-- The number of vertices in the original polygon. -/
  polygon_vertices : ℕ
  /-- The number of additional interior points. -/
  interior_points : ℕ
  /-- The property that no three interior points are collinear. -/
  no_collinear_interior : Prop

/-- The number of triangles in a triangulation. -/
def num_triangles (t : Triangulation) : ℕ :=
  2 * (t.polygon_vertices + t.interior_points) - 2

/-- The main theorem about the number of triangles in the specific triangulation. -/
theorem triangulation_theorem (t : Triangulation) 
  (h1 : t.polygon_vertices = 1000)
  (h2 : t.interior_points = 500)
  (h3 : t.no_collinear_interior) :
  num_triangles t = 2998 := by
  sorry

end triangulation_theorem_l234_23484


namespace parabola_properties_l234_23489

/-- The quadratic function f(x) = x^2 - 8x + 12 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 12

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 4

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -4

theorem parabola_properties :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ 
  f vertex_x = vertex_y ∧
  f 3 = -3 := by sorry

end parabola_properties_l234_23489


namespace starting_lineup_count_l234_23465

-- Define the total number of players
def total_players : ℕ := 12

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the size of the starting lineup
def lineup_size : ℕ := 5

-- Theorem statement
theorem starting_lineup_count :
  (num_twins * Nat.choose (total_players - 1) (lineup_size - 1)) = 660 := by
  sorry

end starting_lineup_count_l234_23465


namespace smallest_lucky_integer_l234_23438

/-- An integer is lucky if there exist several consecutive integers, including itself, that add up to 2023. -/
def IsLucky (n : ℤ) : Prop :=
  ∃ k : ℕ, ∃ m : ℤ, (m + k : ℤ) = n ∧ (k + 1) * (2 * m + k) / 2 = 2023

/-- The smallest lucky integer -/
def SmallestLuckyInteger : ℤ := -2022

theorem smallest_lucky_integer :
  IsLucky SmallestLuckyInteger ∧
  ∀ n : ℤ, n < SmallestLuckyInteger → ¬IsLucky n :=
by sorry

end smallest_lucky_integer_l234_23438


namespace internet_service_upgrade_l234_23400

/-- Represents the internet service with speed and price -/
structure InternetService where
  speed : ℕ  -- Speed in Mbps
  price : ℕ  -- Price in dollars
  deriving Repr

/-- Calculates the yearly price difference between two services -/
def yearlyPriceDifference (s1 s2 : InternetService) : ℕ :=
  (s2.price - s1.price) * 12

/-- The problem statement -/
theorem internet_service_upgrade (current : InternetService)
    (upgrade20 upgrade30 : InternetService)
    (h1 : current.speed = 10 ∧ current.price = 20)
    (h2 : upgrade20.speed = 20 ∧ upgrade20.price = current.price + 10)
    (h3 : upgrade30.speed = 30)
    (h4 : yearlyPriceDifference upgrade20 upgrade30 = 120) :
    upgrade30.price / current.price = 2 := by
  sorry

end internet_service_upgrade_l234_23400


namespace theresa_video_games_l234_23452

theorem theresa_video_games (tory julia theresa : ℕ) : 
  tory = 6 → 
  julia = tory / 3 → 
  theresa = 3 * julia + 5 → 
  theresa = 11 := by
sorry

end theresa_video_games_l234_23452


namespace larger_cuboid_width_l234_23458

theorem larger_cuboid_width
  (small_length small_width small_height : ℝ)
  (large_length large_height : ℝ)
  (num_small_cuboids : ℕ)
  (h1 : small_length = 5)
  (h2 : small_width = 4)
  (h3 : small_height = 3)
  (h4 : large_length = 16)
  (h5 : large_height = 12)
  (h6 : num_small_cuboids = 32)
  (h7 : small_length * small_width * small_height * num_small_cuboids = large_length * large_height * (large_length * large_height / (small_length * small_width * small_height * num_small_cuboids))) :
  large_length * large_height / (small_length * small_width * small_height * num_small_cuboids) = 10 := by
sorry

end larger_cuboid_width_l234_23458


namespace nap_hours_in_70_days_l234_23467

/-- Calculates the total hours of naps taken in a given number of days -/
def total_nap_hours (days : ℕ) (naps_per_week : ℕ) (hours_per_nap : ℕ) : ℕ :=
  let weeks : ℕ := days / 7
  let total_naps : ℕ := weeks * naps_per_week
  total_naps * hours_per_nap

/-- Theorem stating that 70 days of naps results in 60 hours of nap time -/
theorem nap_hours_in_70_days :
  total_nap_hours 70 3 2 = 60 := by
  sorry

#eval total_nap_hours 70 3 2

end nap_hours_in_70_days_l234_23467


namespace xiaojun_pen_refills_l234_23472

/-- The number of pen refills Xiaojun bought -/
def num_pen_refills : ℕ := 2

/-- The cost of each pen refill in yuan -/
def pen_refill_cost : ℕ := 2

/-- The cost of each eraser in yuan (positive integer) -/
def eraser_cost : ℕ := 2

/-- The total amount spent in yuan -/
def total_spent : ℕ := 6

/-- The number of erasers Xiaojun bought -/
def num_erasers : ℕ := 1

theorem xiaojun_pen_refills :
  num_pen_refills = 2 ∧
  pen_refill_cost = 2 ∧
  eraser_cost > 0 ∧
  total_spent = 6 ∧
  num_pen_refills = 2 * num_erasers ∧
  total_spent = num_pen_refills * pen_refill_cost + num_erasers * eraser_cost :=
by sorry

#check xiaojun_pen_refills

end xiaojun_pen_refills_l234_23472


namespace james_bought_three_dirt_bikes_l234_23423

/-- Calculates the number of dirt bikes James bought given the costs and total spent -/
def number_of_dirt_bikes (dirt_bike_cost off_road_cost registration_cost total_cost : ℕ) 
  (num_off_road : ℕ) : ℕ :=
  let total_off_road_cost := num_off_road * (off_road_cost + registration_cost)
  let remaining_cost := total_cost - total_off_road_cost
  remaining_cost / (dirt_bike_cost + registration_cost)

/-- Proves that James bought 3 dirt bikes given the problem conditions -/
theorem james_bought_three_dirt_bikes : 
  number_of_dirt_bikes 150 300 25 1825 4 = 3 := by
  sorry

end james_bought_three_dirt_bikes_l234_23423


namespace nine_sided_polygon_diagonals_l234_23431

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 9 sides has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by sorry

end nine_sided_polygon_diagonals_l234_23431


namespace wendy_packaging_chocolates_l234_23422

/-- The number of chocolates Wendy can package in one hour -/
def chocolates_per_hour : ℕ := 1152 / 4

/-- The number of chocolates Wendy can package in h hours -/
def chocolates_in_h_hours (h : ℕ) : ℕ := chocolates_per_hour * h

theorem wendy_packaging_chocolates (h : ℕ) : 
  chocolates_in_h_hours h = 288 * h := by
  sorry

#check wendy_packaging_chocolates

end wendy_packaging_chocolates_l234_23422


namespace number_sum_15_equals_96_l234_23405

theorem number_sum_15_equals_96 : ∃ x : ℝ, x + 15 = 96 ∧ x = 81 := by
  sorry

end number_sum_15_equals_96_l234_23405


namespace polynomial_factorization_l234_23411

theorem polynomial_factorization (m : ℤ) : 
  (∃ (a b c d e f : ℤ), ∀ (x y : ℤ), 
    x^2 + 2*x*y + 2*x + m*y + 2*m = (a*x + b*y + c) * (d*x + e*y + f)) ↔ m = 2 := by
  sorry

end polynomial_factorization_l234_23411


namespace imaginary_real_sum_imaginary_real_sum_proof_l234_23424

theorem imaginary_real_sum : ℂ → Prop :=
  fun z : ℂ => 
    let a : ℝ := Complex.im (z⁻¹)
    let b : ℝ := Complex.re ((1 + Complex.I) ^ 2)
    a + b = -1

theorem imaginary_real_sum_proof : imaginary_real_sum Complex.I := by
  sorry

end imaginary_real_sum_imaginary_real_sum_proof_l234_23424


namespace selling_price_calculation_l234_23483

theorem selling_price_calculation (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
  cost_price = 540 →
  markup_percentage = 15 →
  discount_percentage = 26.570048309178745 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discount_amount := marked_price * (discount_percentage / 100)
  let selling_price := marked_price - discount_amount
  selling_price = 456 := by
sorry

end selling_price_calculation_l234_23483


namespace division_result_l234_23481

theorem division_result : ∃ (result : ℚ), 
  (40 / 2 = result) ∧ 
  (40 + result + 2 = 62) ∧ 
  (result = 20) := by
  sorry

end division_result_l234_23481


namespace squirrel_nut_division_l234_23403

theorem squirrel_nut_division (n : ℕ) : ¬(5 ∣ (2022 + n * (n + 1))) := by
  sorry

end squirrel_nut_division_l234_23403


namespace quotient_digits_l234_23478

def dividend (n : ℕ) : ℕ := 100 * n + 38

theorem quotient_digits :
  (∀ n : ℕ, n ≤ 7 → (dividend n) / 8 < 100) ∧
  (dividend 7) / 8 ≥ 10 ∧
  (∀ n : ℕ, n ≥ 8 → (dividend n) / 8 ≥ 100) ∧
  (dividend 8) / 8 < 1000 :=
sorry

end quotient_digits_l234_23478


namespace regular_18gon_symmetry_sum_l234_23450

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Add any necessary fields here

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationalSymmetryAngle (p : RegularPolygon n) : ℝ := sorry

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℝ) + smallestRotationalSymmetryAngle p = 38 := by sorry

end regular_18gon_symmetry_sum_l234_23450


namespace product_from_hcf_lcm_l234_23408

theorem product_from_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 16 → Nat.lcm a b = 160 → a * b = 2560 := by
  sorry

end product_from_hcf_lcm_l234_23408


namespace sum_of_monomials_l234_23436

-- Define the monomials
def monomial1 (x y : ℝ) (m : ℕ) := x^2 * y^m
def monomial2 (x y : ℝ) (n : ℕ) := x^n * y^3

-- Define the condition that the sum is a monomial
def sum_is_monomial (x y : ℝ) (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), ∀ x y, monomial1 x y m + monomial2 x y n = x^a * y^b

-- State the theorem
theorem sum_of_monomials (m n : ℕ) :
  (∀ x y : ℝ, sum_is_monomial x y m n) → m + n = 5 := by
  sorry

end sum_of_monomials_l234_23436


namespace bags_filled_on_saturday_l234_23456

theorem bags_filled_on_saturday (bags_sunday : ℕ) (cans_per_bag : ℕ) (total_cans : ℕ) : 
  bags_sunday = 4 →
  cans_per_bag = 9 →
  total_cans = 63 →
  ∃ (bags_saturday : ℕ), 
    bags_saturday * cans_per_bag + bags_sunday * cans_per_bag = total_cans ∧
    bags_saturday = 3 := by
  sorry

end bags_filled_on_saturday_l234_23456


namespace maximize_annual_average_profit_l234_23466

/-- Represents the problem of maximizing annual average profit for equipment purchase --/
theorem maximize_annual_average_profit :
  let initial_cost : ℕ := 90000
  let first_year_cost : ℕ := 20000
  let annual_cost_increase : ℕ := 20000
  let annual_revenue : ℕ := 110000
  let total_cost (n : ℕ) : ℕ := initial_cost + n * first_year_cost + (n * (n - 1) * annual_cost_increase) / 2
  let total_revenue (n : ℕ) : ℕ := n * annual_revenue
  let total_profit (n : ℕ) : ℤ := (total_revenue n : ℤ) - (total_cost n : ℤ)
  let annual_average_profit (n : ℕ) : ℚ := (total_profit n : ℚ) / n
  ∀ m : ℕ, m > 0 → annual_average_profit 3 ≥ annual_average_profit m :=
by
  sorry


end maximize_annual_average_profit_l234_23466


namespace f_passes_through_point_f_has_max_at_one_f_is_unique_l234_23471

/-- A quadratic function that passes through (2, -6) and has a maximum of -4 at x = 1 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 6

/-- The function f passes through the point (2, -6) -/
theorem f_passes_through_point : f 2 = -6 := by sorry

/-- The function f has a maximum value of -4 when x = 1 -/
theorem f_has_max_at_one :
  (∀ x, f x ≤ f 1) ∧ f 1 = -4 := by sorry

/-- The function f is the unique quadratic function satisfying the given conditions -/
theorem f_is_unique (g : ℝ → ℝ) :
  (g 2 = -6) →
  ((∀ x, g x ≤ g 1) ∧ g 1 = -4) →
  (∃ a b c, ∀ x, g x = a * x^2 + b * x + c) →
  (∀ x, g x = f x) := by sorry

end f_passes_through_point_f_has_max_at_one_f_is_unique_l234_23471


namespace annual_rent_per_square_foot_l234_23416

-- Define the shop dimensions
def shop_length : ℝ := 20
def shop_width : ℝ := 15

-- Define the monthly rent
def monthly_rent : ℝ := 3600

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem annual_rent_per_square_foot :
  let shop_area := shop_length * shop_width
  let annual_rent := monthly_rent * months_per_year
  annual_rent / shop_area = 144 := by
  sorry

end annual_rent_per_square_foot_l234_23416


namespace corn_growth_first_week_l234_23430

/-- Represents the growth of corn over three weeks -/
structure CornGrowth where
  week1 : ℝ
  week2 : ℝ
  week3 : ℝ

/-- The conditions of corn growth as described in the problem -/
def valid_growth (g : CornGrowth) : Prop :=
  g.week2 = 2 * g.week1 ∧
  g.week3 = 4 * g.week2 ∧
  g.week1 + g.week2 + g.week3 = 22

/-- The theorem stating that the corn grew 2 inches in the first week -/
theorem corn_growth_first_week :
  ∀ g : CornGrowth, valid_growth g → g.week1 = 2 :=
by
  sorry

end corn_growth_first_week_l234_23430


namespace exactly_two_correct_l234_23426

-- Define the propositions
def prop1 : Prop := ∃ n : ℤ, ∀ m : ℤ, m < 0 → m ≤ n
def prop2 : Prop := ∃ n : ℤ, ∀ m : ℤ, n ≤ m
def prop3 : Prop := ∀ n : ℤ, n < 0 → n ≤ -1
def prop4 : Prop := ∀ n : ℤ, n > 0 → 1 ≤ n

-- Theorem stating that exactly two propositions are correct
theorem exactly_two_correct : 
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 :=
sorry

end exactly_two_correct_l234_23426


namespace arithmetic_sequence_sum_l234_23490

/-- Arithmetic sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Theorem: For an arithmetic sequence with a_1 = 1 and d = 2,
    if S_{k+2} - S_k = 24, then k = 5 -/
theorem arithmetic_sequence_sum (k : ℕ) :
  S (k + 2) - S k = 24 → k = 5 := by
  sorry

end arithmetic_sequence_sum_l234_23490


namespace sin_alpha_minus_2pi_over_3_l234_23477

theorem sin_alpha_minus_2pi_over_3 (α : ℝ) (h : Real.cos (π / 6 - α) = 2 / 3) :
  Real.sin (α - 2 * π / 3) = -2 / 3 := by
  sorry

end sin_alpha_minus_2pi_over_3_l234_23477


namespace total_white_pieces_l234_23496

/-- The total number of pieces -/
def total_pieces : ℕ := 300

/-- The number of piles -/
def num_piles : ℕ := 100

/-- The number of pieces in each pile -/
def pieces_per_pile : ℕ := 3

/-- The number of piles with exactly one white piece -/
def piles_with_one_white : ℕ := 27

/-- The number of piles with 2 or 3 black pieces -/
def piles_with_two_or_three_black : ℕ := 42

theorem total_white_pieces :
  ∃ (piles_with_three_white : ℕ) 
    (piles_with_two_white : ℕ)
    (total_white : ℕ),
  piles_with_three_white = num_piles - piles_with_one_white - piles_with_two_or_three_black + piles_with_one_white ∧
  piles_with_two_white = num_piles - piles_with_one_white - 2 * piles_with_three_white ∧
  total_white = piles_with_one_white * 1 + piles_with_three_white * 3 + piles_with_two_white * 2 ∧
  total_white = 158 :=
by sorry

end total_white_pieces_l234_23496


namespace triangle_problem_l234_23412

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (Real.tan t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B)) →
  (Real.sin (t.B - t.A) = Real.cos t.C) →
  (t.A = π/4 ∧ t.C = π/3) ∧
  (((1/2) * t.a * t.c * Real.sin t.B = 3 + Real.sqrt 3) →
   (t.a = 2 * Real.sqrt 2 ∧ t.c = 2 * Real.sqrt 3)) :=
by sorry


end triangle_problem_l234_23412


namespace parabola_min_area_sum_l234_23440

/-- A parabola in the Cartesian plane -/
structure Parabola where
  eqn : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- A point lies on a parabola -/
def lies_on (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem parabola_min_area_sum (p : Parabola) (A B : ℝ × ℝ) :
  p.eqn = fun x y ↦ y^2 = 2*x →
  lies_on p A →
  lies_on p B →
  dot_product A B = -1 →
  let F := focus p
  let O := (0, 0)
  ∃ (min : ℝ), min = Real.sqrt 2 / 2 ∧
    ∀ (X Y : ℝ × ℝ), lies_on p X → lies_on p Y → dot_product X Y = -1 →
      triangle_area O F X + triangle_area O F Y ≥ min :=
sorry

end parabola_min_area_sum_l234_23440


namespace dice_probability_theorem_l234_23499

def num_dice : ℕ := 15
def num_sides : ℕ := 6
def target_count : ℕ := 4

theorem dice_probability_theorem :
  (Nat.choose num_dice target_count * 5^(num_dice - target_count)) / 6^num_dice =
  (1365 * 5^11) / 6^15 := by sorry

end dice_probability_theorem_l234_23499
