import Mathlib

namespace quadratic_real_roots_l384_38462

theorem quadratic_real_roots (k m : ℝ) (hm : m ≠ 0) :
  (∃ x : ℝ, x^2 + k*x + m = 0) ↔ m ≤ k^2/4 := by
  sorry

end quadratic_real_roots_l384_38462


namespace target_hit_probability_l384_38496

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.8) (h2 : p2 = 0.7) :
  1 - (1 - p1) * (1 - p2) = 0.94 := by
  sorry

end target_hit_probability_l384_38496


namespace rectangle_area_l384_38404

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := by
  sorry

end rectangle_area_l384_38404


namespace base_6_addition_subtraction_l384_38478

/-- Converts a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Theorem: The sum of 555₆ and 65₆ minus 11₆ equals 1053₆ in base 6 --/
theorem base_6_addition_subtraction :
  to_base_6 (to_base_10 [5, 5, 5] + to_base_10 [5, 6] - to_base_10 [1, 1]) = [3, 5, 0, 1] := by
  sorry

end base_6_addition_subtraction_l384_38478


namespace disk_with_hole_moment_of_inertia_l384_38464

/-- The moment of inertia of a disk with a hole -/
theorem disk_with_hole_moment_of_inertia
  (R M : ℝ)
  (h_R : R > 0)
  (h_M : M > 0) :
  let I₀ : ℝ := (1 / 2) * M * R^2
  let m_hole : ℝ := M / 4
  let R_hole : ℝ := R / 2
  let I_center_hole : ℝ := (1 / 2) * m_hole * R_hole^2
  let d : ℝ := R / 2
  let I_hole : ℝ := I_center_hole + m_hole * d^2
  I₀ - I_hole = (13 / 32) * M * R^2 :=
sorry

end disk_with_hole_moment_of_inertia_l384_38464


namespace janet_lives_count_janet_final_lives_l384_38459

theorem janet_lives_count (initial_lives lost_lives gained_lives : ℕ) : 
  initial_lives - lost_lives + gained_lives = (initial_lives - lost_lives) + gained_lives :=
by sorry

theorem janet_final_lives : 
  38 - 16 + 32 = 54 :=
by sorry

end janet_lives_count_janet_final_lives_l384_38459


namespace mistaken_operation_l384_38440

/-- Given an operation O on real numbers that results in a 99% error
    compared to multiplying by 10, prove that O(x) = 0.1 * x for all x. -/
theorem mistaken_operation (O : ℝ → ℝ) (h : ∀ x : ℝ, O x = 0.01 * (10 * x)) :
  ∀ x : ℝ, O x = 0.1 * x := by
sorry

end mistaken_operation_l384_38440


namespace attendance_difference_l384_38425

/-- The number of games Tara played each year -/
def games_per_year : ℕ := 20

/-- The percentage of games Tara's dad attended in the first year -/
def first_year_attendance_percentage : ℚ := 90 / 100

/-- The number of games Tara's dad attended in the second year -/
def second_year_attendance : ℕ := 14

/-- Theorem stating the difference in games attended between the first and second year -/
theorem attendance_difference :
  ⌊(first_year_attendance_percentage * games_per_year : ℚ)⌋ - second_year_attendance = 4 := by
  sorry

end attendance_difference_l384_38425


namespace discounted_price_theorem_l384_38463

/-- Given a bag marked at $200 with a 40% discount, prove that the discounted price is $120. -/
theorem discounted_price_theorem (marked_price : ℝ) (discount_percentage : ℝ) 
  (h1 : marked_price = 200)
  (h2 : discount_percentage = 40) :
  marked_price * (1 - discount_percentage / 100) = 120 := by
  sorry

end discounted_price_theorem_l384_38463


namespace prime_equation_value_l384_38497

theorem prime_equation_value (p q : ℕ) : 
  Prime p → Prime q → (∃ x : ℤ, p * x + 5 * q = 97) → (40 * p + 101 * q + 4 = 2003) := by
  sorry

end prime_equation_value_l384_38497


namespace total_results_l384_38454

theorem total_results (avg : ℚ) (first_12_avg : ℚ) (last_12_avg : ℚ) (result_13 : ℚ) :
  avg = 50 →
  first_12_avg = 14 →
  last_12_avg = 17 →
  result_13 = 878 →
  ∃ N : ℕ, (N : ℚ) * avg = 12 * first_12_avg + 12 * last_12_avg + result_13 ∧ N = 25 :=
by sorry

end total_results_l384_38454


namespace polyhedron_inequality_l384_38484

/-- A convex polyhedron bounded by quadrilateral faces -/
class ConvexPolyhedron where
  /-- The surface area of the polyhedron -/
  area : ℝ
  /-- The sum of the squares of the polyhedron's edges -/
  edge_sum_squares : ℝ
  /-- The polyhedron is bounded by quadrilateral faces -/
  quad_faces : Prop

/-- 
For a convex polyhedron bounded by quadrilateral faces, 
the sum of the squares of its edges is greater than or equal to twice its surface area 
-/
theorem polyhedron_inequality (p : ConvexPolyhedron) : p.edge_sum_squares ≥ 2 * p.area := by
  sorry

end polyhedron_inequality_l384_38484


namespace unique_solution_equation_l384_38402

theorem unique_solution_equation :
  ∃! y : ℝ, (3 * y^2 - 12 * y) / (y^2 - 4 * y) = y - 2 ∧
             y ≠ 2 ∧
             y^2 - 4 * y ≠ 0 := by
  sorry

end unique_solution_equation_l384_38402


namespace equation_solution_l384_38412

theorem equation_solution : ∃ x : ℝ, x + 1 - 2 * (x - 1) = 1 - 3 * x ∧ x = 0 := by
  sorry

end equation_solution_l384_38412


namespace factorization_equality_l384_38451

theorem factorization_equality (x : ℝ) : 84 * x^5 - 210 * x^9 = -42 * x^5 * (5 * x^4 - 2) := by
  sorry

end factorization_equality_l384_38451


namespace complement_union_problem_l384_38491

theorem complement_union_problem (U A B : Set Nat) : 
  U = {1, 2, 3, 4} →
  A = {1, 2} →
  B = {2, 3} →
  (Aᶜ ∪ B) = {2, 3, 4} := by
  sorry

end complement_union_problem_l384_38491


namespace max_area_rectangle_in_circle_max_area_is_8_l384_38445

theorem max_area_rectangle_in_circle (x y : ℝ) : 
  x > 0 → y > 0 → x^2 + y^2 = 16 → x * y ≤ 8 := by
  sorry

theorem max_area_is_8 : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 16 ∧ x * y = 8 := by
  sorry

end max_area_rectangle_in_circle_max_area_is_8_l384_38445


namespace cubic_equation_roots_l384_38411

theorem cubic_equation_roots : 
  {x : ℝ | x^9 + (9/8)*x^6 + (27/64)*x^3 - x + 219/512 = 0} = 
  {1/2, (-1 - Real.sqrt 13)/4, (-1 + Real.sqrt 13)/4} := by
  sorry

end cubic_equation_roots_l384_38411


namespace dolly_dresses_shipment_l384_38485

theorem dolly_dresses_shipment (total : ℕ) : 
  (70 : ℕ) * total = 140 * 100 → total = 200 := by
  sorry

end dolly_dresses_shipment_l384_38485


namespace holiday_customers_l384_38442

def normal_rate : ℕ := 175
def holiday_multiplier : ℕ := 2
def hours : ℕ := 8

theorem holiday_customers :
  normal_rate * holiday_multiplier * hours = 2800 :=
by sorry

end holiday_customers_l384_38442


namespace crushing_load_calculation_l384_38405

theorem crushing_load_calculation (T H K : ℚ) (L : ℚ) 
  (h1 : T = 5)
  (h2 : H = 10)
  (h3 : K = 2)
  (h4 : L = (30 * T^3 * K) / H^3) :
  L = 15/2 := by
  sorry

end crushing_load_calculation_l384_38405


namespace arithmetic_mean_fractions_l384_38430

theorem arithmetic_mean_fractions (b c x : ℝ) (hbc : b ≠ c) (hx : x ≠ 0) :
  ((x + b) / x + (x - c) / x) / 2 = 1 + (b - c) / (2 * x) := by
  sorry

end arithmetic_mean_fractions_l384_38430


namespace johns_final_push_time_l384_38410

/-- The time of John's final push in a race, given the initial and final distances between John and Steve, and their respective speeds. -/
theorem johns_final_push_time 
  (initial_distance : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 12)
  (h2 : john_speed = 4.2)
  (h3 : steve_speed = 3.7)
  (h4 : final_distance = 2) :
  ∃ t : ℝ, t = 28 ∧ john_speed * t = steve_speed * t + initial_distance + final_distance :=
by sorry

end johns_final_push_time_l384_38410


namespace math_city_intersections_l384_38461

/-- Represents a city with straight streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  (c.num_streets.choose 2)

/-- Theorem: A city with 10 streets meeting the given conditions has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel → c.no_triple_intersections →
  num_intersections c = 45 := by
  sorry

#check math_city_intersections

end math_city_intersections_l384_38461


namespace rock_max_height_l384_38407

/-- The height function of the rock -/
def h (t : ℝ) : ℝ := 150 * t - 15 * t^2

/-- The maximum height reached by the rock -/
theorem rock_max_height : ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 375 := by
  sorry

end rock_max_height_l384_38407


namespace fourth_power_sum_l384_38426

theorem fourth_power_sum (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = 2) :
  x^4 + y^4 = 112 := by
sorry

end fourth_power_sum_l384_38426


namespace inequality_proof_l384_38449

theorem inequality_proof (a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) :
  (a₁ + a₃) / (a₁ + a₂) + (a₂ + a₄) / (a₂ + a₃) + (a₃ + a₁) / (a₃ + a₄) + (a₄ + a₂) / (a₄ + a₁) ≥ 4 :=
by sorry

end inequality_proof_l384_38449


namespace frustum_volume_l384_38489

/-- Given a square pyramid and a smaller pyramid cut from it parallel to the base,
    calculate the volume of the resulting frustum. -/
theorem frustum_volume
  (base_edge : ℝ)
  (altitude : ℝ)
  (small_base_edge : ℝ)
  (small_altitude : ℝ)
  (h_base : base_edge = 15)
  (h_altitude : altitude = 10)
  (h_small_base : small_base_edge = 7.5)
  (h_small_altitude : small_altitude = 5) :
  (1 / 3 * base_edge^2 * altitude) - (1 / 3 * small_base_edge^2 * small_altitude) = 656.25 :=
by sorry

end frustum_volume_l384_38489


namespace expression_equals_half_y_l384_38498

theorem expression_equals_half_y (y d : ℝ) (hy : y > 0) : 
  (4 * y) / 20 + (3 * y) / d = 0.5 * y → d = 10 := by
  sorry

end expression_equals_half_y_l384_38498


namespace olaf_water_requirement_l384_38477

/-- Calculates the total water needed for a sailing trip -/
def water_needed_for_trip (crew_size : ℕ) (water_per_man_per_day : ℚ) 
  (boat_speed : ℕ) (total_distance : ℕ) : ℚ :=
  let trip_duration := total_distance / boat_speed
  let daily_water_requirement := crew_size * water_per_man_per_day
  daily_water_requirement * trip_duration

/-- Theorem: The total water needed for Olaf's sailing trip is 250 gallons -/
theorem olaf_water_requirement : 
  water_needed_for_trip 25 (1/2) 200 4000 = 250 := by
  sorry

end olaf_water_requirement_l384_38477


namespace total_difference_is_122_l384_38447

/-- The total difference in the number of apples and peaches for Mia, Steven, and Jake -/
def total_difference (steven_apples steven_peaches : ℕ) : ℕ :=
  let mia_apples := 2 * steven_apples
  let jake_apples := steven_apples + 4
  let jake_peaches := steven_peaches - 3
  let mia_peaches := jake_peaches + 3
  (mia_apples + mia_peaches) + (steven_apples + steven_peaches) + (jake_apples + jake_peaches)

/-- Theorem stating the total difference in fruits for Mia, Steven, and Jake -/
theorem total_difference_is_122 :
  total_difference 19 15 = 122 :=
by sorry

end total_difference_is_122_l384_38447


namespace joe_cars_count_l384_38456

theorem joe_cars_count (initial_cars new_cars : ℕ) 
  (h1 : initial_cars = 50) 
  (h2 : new_cars = 12) : 
  initial_cars + new_cars = 62 := by
  sorry

end joe_cars_count_l384_38456


namespace square_side_increase_l384_38435

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let b := 2 * a
  let c := b * (1 + 80 / 100)
  c^2 = (a^2 + b^2) * (1 + 159.20000000000002 / 100) := by
  sorry

end square_side_increase_l384_38435


namespace factorization_problems_l384_38499

theorem factorization_problems :
  (∀ a b : ℝ, a^2 * b - a * b^2 = a * b * (a - b)) ∧
  (∀ x : ℝ, 2 * x^2 - 8 = 2 * (x + 2) * (x - 2)) :=
by sorry

end factorization_problems_l384_38499


namespace expression_value_l384_38433

theorem expression_value (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 := by
  sorry

end expression_value_l384_38433


namespace malcolm_red_lights_l384_38446

def malcolm_lights (red : ℕ) (blue : ℕ) (green : ℕ) (left_to_buy : ℕ) (total_white : ℕ) : Prop :=
  blue = 3 * red ∧
  green = 6 ∧
  left_to_buy = 5 ∧
  total_white = 59 ∧
  red + blue + green + left_to_buy = total_white

theorem malcolm_red_lights :
  ∃ (red : ℕ), malcolm_lights red (3 * red) 6 5 59 ∧ red = 12 := by
  sorry

end malcolm_red_lights_l384_38446


namespace asterisk_replacement_l384_38490

theorem asterisk_replacement : ∃ x : ℝ, (x / 20) * (x / 180) = 1 ∧ x = 60 := by
  sorry

end asterisk_replacement_l384_38490


namespace theater_ticket_difference_l384_38460

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold. -/
def TicketSales.total (s : TicketSales) : ℕ :=
  s.orchestra + s.balcony

/-- Calculates the total revenue from ticket sales. -/
def TicketSales.revenue (s : TicketSales) : ℕ :=
  12 * s.orchestra + 8 * s.balcony

theorem theater_ticket_difference (s : TicketSales) :
  s.total = 355 → s.revenue = 3320 → s.balcony - s.orchestra = 115 := by
  sorry

end theater_ticket_difference_l384_38460


namespace quadratic_integer_roots_l384_38444

theorem quadratic_integer_roots (p : ℕ) : 
  Prime p ∧ 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + p*x - 512*p = 0 ∧ y^2 + p*y - 512*p = 0) ↔ 
  p = 2 := by
sorry

end quadratic_integer_roots_l384_38444


namespace chef_used_one_apple_l384_38416

/-- The number of apples used by a chef when making pies -/
def apples_used (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the chef used 1 apple -/
theorem chef_used_one_apple :
  apples_used 40 39 = 1 := by
  sorry

end chef_used_one_apple_l384_38416


namespace andy_restrung_seven_racquets_l384_38486

/-- Calculates the number of racquets Andy restrung during his shift -/
def racquets_restrung (hourly_rate : ℤ) (restring_rate : ℤ) (grommet_rate : ℤ) (stencil_rate : ℤ)
                      (hours_worked : ℤ) (grommets_changed : ℤ) (stencils_painted : ℤ)
                      (total_earnings : ℤ) : ℤ :=
  let hourly_earnings := hourly_rate * hours_worked
  let grommet_earnings := grommet_rate * grommets_changed
  let stencil_earnings := stencil_rate * stencils_painted
  let restring_earnings := total_earnings - hourly_earnings - grommet_earnings - stencil_earnings
  restring_earnings / restring_rate

theorem andy_restrung_seven_racquets :
  racquets_restrung 9 15 10 1 8 2 5 202 = 7 := by
  sorry


end andy_restrung_seven_racquets_l384_38486


namespace f_properties_l384_38400

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

theorem f_properties :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x > f y) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≥ -7) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = -7) :=
sorry

end f_properties_l384_38400


namespace weeks_to_buy_iphone_l384_38455

def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_buy_iphone :
  (iphone_cost - trade_in_value) / weekly_earnings = 7 := by
  sorry

end weeks_to_buy_iphone_l384_38455


namespace product_set_sum_l384_38487

theorem product_set_sum (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Set ℚ) = {-24, -2, -3/2, -1/8, 1, 3} →
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end product_set_sum_l384_38487


namespace amount_of_c_l384_38401

/-- Given four people a, b, c, and d with monetary amounts, prove that c has 500 units of currency. -/
theorem amount_of_c (a b c d : ℕ) : 
  a + b + c + d = 1800 →
  a + c = 500 →
  b + c = 900 →
  a + d = 700 →
  a + b + d = 1300 →
  c = 500 := by
  sorry

end amount_of_c_l384_38401


namespace siblings_combined_weight_l384_38427

/-- Given Antonio's weight and the difference between his and his sister's weight,
    calculate their combined weight. -/
theorem siblings_combined_weight (antonio_weight sister_weight_diff : ℕ) :
  antonio_weight = 50 →
  sister_weight_diff = 12 →
  antonio_weight + (antonio_weight - sister_weight_diff) = 88 := by
  sorry

#check siblings_combined_weight

end siblings_combined_weight_l384_38427


namespace exponent_rules_l384_38470

theorem exponent_rules (a b : ℝ) : 
  ((-b)^2 * (-b)^3 * (-b)^5 = b^10) ∧ ((2*a*b^2)^3 = 8*a^3*b^6) := by
  sorry

end exponent_rules_l384_38470


namespace no_m_exists_for_equality_m_range_for_subset_l384_38415

-- Define the set P
def P : Set ℝ := {x | x^2 - 8*x - 20 ≤ 0}

-- Define the set S parameterized by m
def S (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem 1: There does not exist an m such that P = S(m)
theorem no_m_exists_for_equality : ¬∃ m : ℝ, P = S m := by
  sorry

-- Theorem 2: The set of m such that P ⊆ S(m) is {m | m ≤ 3}
theorem m_range_for_subset : {m : ℝ | P ⊆ S m} = {m : ℝ | m ≤ 3} := by
  sorry

end no_m_exists_for_equality_m_range_for_subset_l384_38415


namespace ladder_slip_distance_l384_38480

/-- The distance the top of a ladder slips down when its bottom moves from 5 feet to 10.658966865741546 feet away from a wall. -/
theorem ladder_slip_distance (ladder_length : Real) (initial_distance : Real) (final_distance : Real) :
  ladder_length = 14 →
  initial_distance = 5 →
  final_distance = 10.658966865741546 →
  let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
  let final_height := Real.sqrt (ladder_length^2 - final_distance^2)
  abs ((initial_height - final_height) - 4.00392512594753) < 0.000001 := by
  sorry

end ladder_slip_distance_l384_38480


namespace x_value_l384_38432

theorem x_value : ∃ x : ℝ, (0.5 * x = 0.05 * 500 - 20) ∧ (x = 10) := by
  sorry

end x_value_l384_38432


namespace second_number_value_l384_38423

theorem second_number_value (a b : ℝ) 
  (eq1 : a * (a - 6) = 7)
  (eq2 : b * (b - 6) = 7)
  (neq : a ≠ b)
  (sum : a + b = 6) :
  b = 7 := by
  sorry

end second_number_value_l384_38423


namespace coin_value_proof_l384_38495

theorem coin_value_proof (total_coins : ℕ) (penny_value : ℕ) (nickel_value : ℕ) :
  total_coins = 16 ∧ 
  penny_value = 1 ∧ 
  nickel_value = 5 →
  ∃ (pennies nickels : ℕ),
    pennies + nickels = total_coins ∧
    nickels = pennies + 2 ∧
    pennies * penny_value + nickels * nickel_value = 52 := by
  sorry

end coin_value_proof_l384_38495


namespace cone_sphere_volume_ratio_l384_38457

theorem cone_sphere_volume_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h) = (1 / 3) * (4 / 3 * π * r^3) → h / r = 4 / 3 := by
  sorry

end cone_sphere_volume_ratio_l384_38457


namespace sphere_diameter_equal_volume_cone_l384_38483

/-- The diameter of a sphere with the same volume as a cone -/
theorem sphere_diameter_equal_volume_cone (r h : ℝ) (hr : r = 2) (hh : h = 8) :
  let cone_volume := (1/3) * Real.pi * r^2 * h
  let sphere_radius := (cone_volume * 3 / (4 * Real.pi))^(1/3)
  2 * sphere_radius = 4 := by sorry

end sphere_diameter_equal_volume_cone_l384_38483


namespace root_sum_fraction_l384_38479

theorem root_sum_fraction (a b c : ℝ) : 
  a^3 - 8*a^2 + 7*a - 3 = 0 → 
  b^3 - 8*b^2 + 7*b - 3 = 0 → 
  c^3 - 8*c^2 + 7*c - 3 = 0 → 
  a / (b*c + 1) + b / (a*c + 1) + c / (a*b + 1) = 17/2 := by
sorry

end root_sum_fraction_l384_38479


namespace orange_purchase_total_l384_38476

/-- The total quantity of oranges bought over three weeks -/
def totalOranges (initialPurchase additionalPurchase : ℕ) : ℕ :=
  let week1Total := initialPurchase + additionalPurchase
  let weeklyPurchaseAfter := 2 * week1Total
  week1Total + weeklyPurchaseAfter + weeklyPurchaseAfter

/-- Proof that the total quantity of oranges bought after three weeks is 75 kgs -/
theorem orange_purchase_total :
  totalOranges 10 5 = 75 := by
  sorry


end orange_purchase_total_l384_38476


namespace largest_number_with_seven_front_l384_38428

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100 = 7) ∧ (n % 100 / 10 ≠ 7) ∧ (n % 10 ≠ 7)

theorem largest_number_with_seven_front :
  ∀ n : ℕ, is_valid_number n → n ≤ 743 :=
by sorry

end largest_number_with_seven_front_l384_38428


namespace min_value_expression_l384_38420

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end min_value_expression_l384_38420


namespace no_positive_integer_solution_l384_38438

theorem no_positive_integer_solution :
  ¬ ∃ (x y : ℕ+), x^2006 - 4*y^2006 - 2006 = 4*y^2007 + 2007*y := by
  sorry

end no_positive_integer_solution_l384_38438


namespace complex_cube_root_unity_l384_38467

theorem complex_cube_root_unity (i : ℂ) (x : ℂ) : 
  i^2 = -1 → 
  x = (-1 + i * Real.sqrt 3) / 2 → 
  1 / (x^3 - x) = -1/2 + (i * Real.sqrt 3) / 2 := by
  sorry

end complex_cube_root_unity_l384_38467


namespace smallest_palindrome_l384_38414

/-- A number is a palindrome in a given base if it reads the same forwards and backwards when represented in that base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in a given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- The number of digits in the representation of a natural number in a given base. -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- 10101₂ in decimal -/
def target : ℕ := 21

theorem smallest_palindrome :
  ∀ n : ℕ,
  (numDigits n 2 = 5 ∧ isPalindrome n 2) →
  (∃ base : ℕ, base > 4 ∧ numDigits n base = 3 ∧ isPalindrome n base) →
  n ≥ target :=
sorry

end smallest_palindrome_l384_38414


namespace triangle_side_and_area_l384_38406

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 1, b = 2, and C = 60°, then c = √3 and the area is √3/2 -/
theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 1) 
  (h2 : b = 2) 
  (h3 : C = Real.pi / 3) -- 60° in radians
  (h4 : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) -- Law of cosines
  (h5 : (a*b*(Real.sin C))/2 = area_triangle) : 
  c = Real.sqrt 3 ∧ area_triangle = Real.sqrt 3 / 2 := by
  sorry


end triangle_side_and_area_l384_38406


namespace arithmetic_sequence_common_difference_l384_38439

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℝ) -- Sum function for the arithmetic sequence
  (h1 : S 2 = 4) -- Given S_2 = 4
  (h2 : S 4 = 20) -- Given S_4 = 20
  : ∃ (a₁ d : ℝ), 
    (∀ n : ℕ, S n = n * (2 * a₁ + (n - 1) * d) / 2) ∧ 
    d = 3 :=
by sorry

end arithmetic_sequence_common_difference_l384_38439


namespace gold_coins_count_l384_38434

theorem gold_coins_count (gold_value : ℕ) (silver_value : ℕ) (silver_count : ℕ) (cash : ℕ) (total : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  silver_count = 5 →
  cash = 30 →
  total = 305 →
  ∃ (gold_count : ℕ), gold_count * gold_value + silver_count * silver_value + cash = total ∧ gold_count = 3 :=
by sorry

end gold_coins_count_l384_38434


namespace digit_subtraction_problem_l384_38409

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem digit_subtraction_problem :
  ∃ (F G D E H I : ℕ),
    is_digit F ∧ is_digit G ∧ is_digit D ∧ is_digit E ∧ is_digit H ∧ is_digit I ∧
    F ≠ G ∧ F ≠ D ∧ F ≠ E ∧ F ≠ H ∧ F ≠ I ∧
    G ≠ D ∧ G ≠ E ∧ G ≠ H ∧ G ≠ I ∧
    D ≠ E ∧ D ≠ H ∧ D ≠ I ∧
    E ≠ H ∧ E ≠ I ∧
    H ≠ I ∧
    F * 10 + G = 93 ∧
    D * 10 + E = 68 ∧
    H * 10 + I = 25 ∧
    (F * 10 + G) - (D * 10 + E) = H * 10 + I :=
by
  sorry

end digit_subtraction_problem_l384_38409


namespace remainder_3005_div_98_l384_38408

theorem remainder_3005_div_98 : 3005 % 98 = 65 := by
  sorry

end remainder_3005_div_98_l384_38408


namespace sequence_properties_l384_38422

-- Define a sequence as a function from natural numbers to real numbers
def Sequence := ℕ → ℝ

-- Statement 1: Sequences appear as isolated points when graphed
def isolated_points (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| ≥ ε

-- Statement 2: All sequences have infinite terms
def infinite_terms (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n

-- Statement 3: The general term formula of a sequence is unique
def unique_formula (s : Sequence) : Prop :=
  ∀ f g : Sequence, (∀ n : ℕ, f n = s n) → (∀ n : ℕ, g n = s n) → f = g

-- Theorem stating that only the first statement is correct
theorem sequence_properties :
  (∀ s : Sequence, isolated_points s) ∧
  (∃ s : Sequence, ¬infinite_terms s) ∧
  (∃ s : Sequence, ¬unique_formula s) :=
sorry

end sequence_properties_l384_38422


namespace sum_of_square_roots_lower_bound_l384_38466

theorem sum_of_square_roots_lower_bound
  (a b c d e : ℝ)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) ≥ 2 :=
sorry

end sum_of_square_roots_lower_bound_l384_38466


namespace gardener_work_days_l384_38429

/-- Calculates the number of days a gardener works on a rose bush replanting project. -/
theorem gardener_work_days
  (num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℚ)
  (gardener_hourly_wage : ℚ)
  (gardener_hours_per_day : ℕ)
  (soil_cubic_feet : ℕ)
  (soil_cost_per_cubic_foot : ℚ)
  (total_project_cost : ℚ)
  (h1 : num_rose_bushes = 20)
  (h2 : cost_per_rose_bush = 150)
  (h3 : gardener_hourly_wage = 30)
  (h4 : gardener_hours_per_day = 5)
  (h5 : soil_cubic_feet = 100)
  (h6 : soil_cost_per_cubic_foot = 5)
  (h7 : total_project_cost = 4100) :
  (total_project_cost - (num_rose_bushes * cost_per_rose_bush + soil_cubic_feet * soil_cost_per_cubic_foot)) / (gardener_hourly_wage * gardener_hours_per_day) = 4 := by
  sorry


end gardener_work_days_l384_38429


namespace population_theorem_l384_38494

/-- The combined population of Pirajussaraí and Tucupira three years ago -/
def combined_population_three_years_ago (pirajussarai_population : ℕ) (tucupira_population : ℕ) : ℕ :=
  pirajussarai_population + tucupira_population

/-- The current combined population of Pirajussaraí and Tucupira -/
def current_combined_population (pirajussarai_population : ℕ) (tucupira_population : ℕ) : ℕ :=
  pirajussarai_population + tucupira_population * 3 / 2

theorem population_theorem (pirajussarai_population : ℕ) (tucupira_population : ℕ) :
  current_combined_population pirajussarai_population tucupira_population = 9000 →
  combined_population_three_years_ago pirajussarai_population tucupira_population = 7200 :=
by
  sorry

#check population_theorem

end population_theorem_l384_38494


namespace paint_time_theorem_l384_38473

/-- The time required to paint a square wall using a cylindrical paint roller -/
theorem paint_time_theorem (roller_length roller_diameter wall_side_length roller_speed : ℝ) :
  roller_length = 20 →
  roller_diameter = 15 →
  wall_side_length = 300 →
  roller_speed = 2 →
  (wall_side_length ^ 2) / (2 * π * (roller_diameter / 2) * roller_length * roller_speed) = 90000 / (600 * π) :=
by sorry

end paint_time_theorem_l384_38473


namespace ellipse_focal_length_l384_38403

/-- The focal length of an ellipse with given properties -/
theorem ellipse_focal_length (k : ℝ) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 - y^2 / k = 1}
  let focus_on_x_axis := true  -- This is a simplification, as we can't directly represent this in Lean
  let eccentricity := (1 : ℝ) / 2
  let focal_length := 1
  (∀ (x y : ℝ), (x, y) ∈ ellipse → x^2 - y^2 / k = 1) ∧ 
  focus_on_x_axis ∧ 
  eccentricity = 1 / 2 →
  focal_length = 1 := by
sorry


end ellipse_focal_length_l384_38403


namespace stone_transport_impossible_l384_38437

/-- The number of stone blocks -/
def n : ℕ := 50

/-- The weight of the first stone block in kg -/
def first_weight : ℕ := 370

/-- The weight increase for each subsequent block in kg -/
def weight_increase : ℕ := 2

/-- The number of available trucks -/
def num_trucks : ℕ := 7

/-- The capacity of each truck in kg -/
def truck_capacity : ℕ := 3000

/-- The total weight of n stone blocks -/
def total_weight (n : ℕ) : ℕ :=
  n * first_weight + (n * (n - 1) / 2) * weight_increase

/-- The total capacity of all trucks -/
def total_capacity : ℕ := num_trucks * truck_capacity

theorem stone_transport_impossible : total_weight n > total_capacity := by
  sorry

end stone_transport_impossible_l384_38437


namespace increase_by_percentage_seventy_increased_by_150_percent_l384_38472

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem seventy_increased_by_150_percent :
  70 * (1 + 150 / 100) = 175 := by sorry

end increase_by_percentage_seventy_increased_by_150_percent_l384_38472


namespace mary_marbles_l384_38448

/-- Given that Joan has 3 yellow marbles and the total number of yellow marbles between Mary and Joan is 12, prove that Mary has 9 yellow marbles. -/
theorem mary_marbles (joan_marbles : ℕ) (total_marbles : ℕ) (h1 : joan_marbles = 3) (h2 : total_marbles = 12) :
  total_marbles - joan_marbles = 9 := by
  sorry

end mary_marbles_l384_38448


namespace prob_all_suits_in_five_draws_l384_38424

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of suits
def num_suits : ℕ := 4

-- Define the number of cards drawn
def cards_drawn : ℕ := 5

-- Define the probability of drawing a card from a specific suit
def prob_suit : ℚ := 1 / 4

-- Theorem statement
theorem prob_all_suits_in_five_draws :
  let prob_sequence := (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4)
  let num_sequences := 24
  (prob_sequence * num_sequences : ℚ) = 9 / 16 := by
  sorry

end prob_all_suits_in_five_draws_l384_38424


namespace cubic_roots_sum_squares_l384_38431

theorem cubic_roots_sum_squares (a b c : ℝ) : 
  (3 * a^3 - 4 * a^2 + 100 * a - 3 = 0) →
  (3 * b^3 - 4 * b^2 + 100 * b - 3 = 0) →
  (3 * c^3 - 4 * c^2 + 100 * c - 3 = 0) →
  (a + b + 2)^2 + (b + c + 2)^2 + (c + a + 2)^2 = 1079/9 := by
  sorry

end cubic_roots_sum_squares_l384_38431


namespace addition_of_like_terms_l384_38421

theorem addition_of_like_terms (a : ℝ) : a + 2*a = 3*a := by
  sorry

end addition_of_like_terms_l384_38421


namespace equivalence_condition_l384_38458

theorem equivalence_condition (a b : ℝ) (h : a * b > 0) :
  a > b ↔ 1 / a < 1 / b := by
  sorry

end equivalence_condition_l384_38458


namespace blueberry_jelly_amount_l384_38443

theorem blueberry_jelly_amount (total_jelly strawberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : strawberry_jelly = 1792) :
  total_jelly - strawberry_jelly = 4518 := by
  sorry

end blueberry_jelly_amount_l384_38443


namespace cricketer_average_score_l384_38492

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_part_matches : ℕ) 
  (overall_average : ℝ) 
  (first_part_average : ℝ) 
  (h1 : total_matches = 12) 
  (h2 : first_part_matches = 8) 
  (h3 : overall_average = 48) 
  (h4 : first_part_average = 40) :
  let last_part_matches := total_matches - first_part_matches
  let total_runs := total_matches * overall_average
  let first_part_runs := first_part_matches * first_part_average
  let last_part_runs := total_runs - first_part_runs
  last_part_runs / last_part_matches = 64 := by
sorry

end cricketer_average_score_l384_38492


namespace no_solution_exists_l384_38441

/-- Sum of digits of a natural number in decimal notation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number n such that n * s(n) = 20222022 -/
theorem no_solution_exists : ¬ ∃ n : ℕ, n * sum_of_digits n = 20222022 := by
  sorry

end no_solution_exists_l384_38441


namespace sugar_package_weight_l384_38450

theorem sugar_package_weight (x : ℝ) 
  (h1 : x > 0)
  (h2 : (4 * x - 10) / (x + 10) = 7 / 8) :
  4 * x + x = 30 := by
  sorry

end sugar_package_weight_l384_38450


namespace law_of_sines_extended_l384_38419

theorem law_of_sines_extended 
  {a b c α β γ : ℝ} 
  (law_of_sines : a / Real.sin α = b / Real.sin β ∧ 
                  b / Real.sin β = c / Real.sin γ)
  (angle_sum : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α := by
sorry

end law_of_sines_extended_l384_38419


namespace imaginary_part_of_complex_number_l384_38471

theorem imaginary_part_of_complex_number :
  let z : ℂ := 1 - 2*I
  Complex.im z = -2 := by
sorry

end imaginary_part_of_complex_number_l384_38471


namespace equation_solutions_l384_38418

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1/((x-2)*(x-3)) + 1/((x-3)*(x-4)) + 1/((x-4)*(x-5))
  ∀ x : ℝ, f x = 1/12 ↔ x = (7 + Real.sqrt 153)/2 ∨ x = (7 - Real.sqrt 153)/2 := by
  sorry

end equation_solutions_l384_38418


namespace max_triangles_from_lines_l384_38468

/-- Given 2017 lines separated into three sets such that lines in the same set are parallel to each other,
    prove that the largest possible number of triangles that can be formed with vertices on these lines
    is 673 * 672^2. -/
theorem max_triangles_from_lines (total_lines : ℕ) (set1 set2 set3 : ℕ) :
  total_lines = 2017 →
  set1 + set2 + set3 = total_lines →
  set1 ≥ set2 →
  set2 ≥ set3 →
  set1 * set2 * set3 ≤ 673 * 672 * 672 :=
by sorry

end max_triangles_from_lines_l384_38468


namespace complex_multiplication_l384_38469

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (2 - i) = 1 + 2*i := by
  sorry

end complex_multiplication_l384_38469


namespace function_formula_l384_38475

theorem function_formula (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2) :
  ∀ x : ℝ, f x = (x + 1)^2 := by
  sorry

end function_formula_l384_38475


namespace rectangle_perimeter_l384_38482

/-- Given a rectangle EFGH where:
  * EF is twice as long as FG
  * FG = 10 units
  * Diagonal EH = 26 units
Prove that the perimeter of EFGH is 60 units -/
theorem rectangle_perimeter (EF FG EH : ℝ) : 
  EF = 2 * FG →
  FG = 10 →
  EH = 26 →
  EH^2 = EF^2 + FG^2 →
  2 * (EF + FG) = 60 := by
  sorry


end rectangle_perimeter_l384_38482


namespace apple_vendor_waste_percentage_l384_38417

/-- Calculates the percentage of apples thrown away given the selling and discarding percentages -/
theorem apple_vendor_waste_percentage
  (initial_apples : ℝ)
  (day1_sell_percentage : ℝ)
  (day1_discard_percentage : ℝ)
  (day2_sell_percentage : ℝ)
  (h1 : initial_apples > 0)
  (h2 : day1_sell_percentage = 0.5)
  (h3 : day1_discard_percentage = 0.2)
  (h4 : day2_sell_percentage = 0.5)
  : (day1_discard_percentage * (1 - day1_sell_percentage) +
     (1 - day2_sell_percentage) * (1 - day1_sell_percentage) * (1 - day1_discard_percentage)) = 0.3 := by
  sorry

#check apple_vendor_waste_percentage

end apple_vendor_waste_percentage_l384_38417


namespace largest_digit_sum_l384_38465

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, and c are digits
  (10 ≤ y ∧ y ≤ 20) →  -- 10 ≤ y ≤ 20
  ((a * 100 + b * 10 + c) : ℚ) / 1000 = 1 / y →  -- 0.abc = 1/y
  a + b + c ≤ 5 :=  -- The sum is at most 5
by sorry

end largest_digit_sum_l384_38465


namespace unique_triple_solution_l384_38481

theorem unique_triple_solution : 
  ∃! (a b c : ℕ+), a * b + b * c = 72 ∧ a * c + b * c = 35 :=
by sorry

end unique_triple_solution_l384_38481


namespace ellipse_equation_l384_38453

/-- Given an ellipse with equation x²/a² + y²/b² = 1, where a > b > 0,
    if its right focus is at (3, 0) and the point (0, -3) is on the ellipse,
    then a² = 18 and b² = 9. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (c : ℝ), c^2 = a^2 - b^2 ∧ c = 3) →
  (0^2 / a^2 + (-3)^2 / b^2 = 1) →
  a^2 = 18 ∧ b^2 = 9 := by sorry

end ellipse_equation_l384_38453


namespace geometric_sequence_a5_l384_38452

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 3) ^ 2 - 3 * (a 3) + 2 = 0 →
  (a 7) ^ 2 - 3 * (a 7) + 2 = 0 →
  a 5 = Real.sqrt 2 := by
sorry

end geometric_sequence_a5_l384_38452


namespace savings_percentage_l384_38474

theorem savings_percentage (salary : ℝ) (savings_after_increase : ℝ) 
  (expense_increase_rate : ℝ) :
  salary = 5500 →
  savings_after_increase = 220 →
  expense_increase_rate = 0.2 →
  ∃ (original_savings_percentage : ℝ),
    original_savings_percentage = 20 ∧
    savings_after_increase = salary - (1 + expense_increase_rate) * 
      (salary - (original_savings_percentage / 100) * salary) :=
by sorry

end savings_percentage_l384_38474


namespace polynomial_infinite_solutions_l384_38436

theorem polynomial_infinite_solutions (P : ℤ → ℤ) (d : ℤ) :
  (∃ (a b : ℤ), ∀ x, P x = a * x + b) ∨ (∀ x, P x = P 0) ↔
  (∃ (S : Set (ℤ × ℤ)), (∀ (x y : ℤ), (x, y) ∈ S → x ≠ y) ∧ 
                         Set.Infinite S ∧
                         (∀ (x y : ℤ), (x, y) ∈ S → P x - P y = d)) :=
by sorry

end polynomial_infinite_solutions_l384_38436


namespace intersection_of_M_and_N_l384_38413

def M : Set ℤ := {x | x < 3}
def N : Set ℤ := {x | 0 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end intersection_of_M_and_N_l384_38413


namespace prime_sum_equality_l384_38493

theorem prime_sum_equality (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p + q = r → p < q → q < r → p = 2 := by
  sorry

end prime_sum_equality_l384_38493


namespace rectangle_area_l384_38488

theorem rectangle_area (length width : ℝ) (h1 : length = 2 * Real.sqrt 6) (h2 : width = 2 * Real.sqrt 3) :
  length * width = 12 * Real.sqrt 2 := by
  sorry

end rectangle_area_l384_38488
