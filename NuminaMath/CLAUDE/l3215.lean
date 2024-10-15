import Mathlib

namespace NUMINAMATH_CALUDE_evaluate_expression_l3215_321508

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3*x^3 - 5*x^2 + 9*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3215_321508


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3215_321520

theorem quadratic_roots_sum_of_squares : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 9*x₁ + 9 = 0) ∧
  (x₂^2 - 9*x₂ + 9 = 0) ∧
  (x₁^2 + x₂^2 = 63) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3215_321520


namespace NUMINAMATH_CALUDE_glass_volume_l3215_321594

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = volume_pessimist)
  (h2 : 0.6 * V = volume_optimist)
  (h3 : volume_optimist - volume_pessimist = 46) :
  V = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l3215_321594


namespace NUMINAMATH_CALUDE_willam_farm_tax_l3215_321529

/-- Farm tax calculation for Mr. Willam -/
theorem willam_farm_tax (total_tax : ℝ) (willam_percentage : ℝ) :
  let willam_tax := total_tax * (willam_percentage / 100)
  willam_tax = total_tax * (willam_percentage / 100) :=
by sorry

#check willam_farm_tax 3840 27.77777777777778

end NUMINAMATH_CALUDE_willam_farm_tax_l3215_321529


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3215_321580

theorem root_sum_reciprocal (p q r : ℂ) : 
  p^3 - p + 1 = 0 → q^3 - q + 1 = 0 → r^3 - r + 1 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) : ℂ) = -10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3215_321580


namespace NUMINAMATH_CALUDE_jack_baseball_cards_l3215_321544

theorem jack_baseball_cards :
  ∀ (total_cards baseball_cards football_cards : ℕ),
    total_cards = 125 →
    baseball_cards = 3 * football_cards + 5 →
    total_cards = baseball_cards + football_cards →
    baseball_cards = 95 := by
  sorry

end NUMINAMATH_CALUDE_jack_baseball_cards_l3215_321544


namespace NUMINAMATH_CALUDE_remainder_of_3_19_times_5_7_mod_100_l3215_321528

theorem remainder_of_3_19_times_5_7_mod_100 : (3^19 * 5^7) % 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_19_times_5_7_mod_100_l3215_321528


namespace NUMINAMATH_CALUDE_price_difference_l3215_321549

/-- Given the total cost of a shirt and sweater, and the price of the shirt,
    calculate the difference in price between the sweater and the shirt. -/
theorem price_difference (total_cost shirt_price : ℚ) 
  (h1 : total_cost = 80.34)
  (h2 : shirt_price = 36.46)
  (h3 : shirt_price < total_cost - shirt_price) :
  total_cost - shirt_price - shirt_price = 7.42 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_l3215_321549


namespace NUMINAMATH_CALUDE_phone_package_comparison_l3215_321599

/-- Represents the monthly bill for a phone package as a function of call duration. -/
structure PhonePackage where
  monthly_fee : ℝ
  call_fee : ℝ
  bill : ℝ → ℝ

/-- Package A with a monthly fee of 15 yuan and a call fee of 0.1 yuan per minute. -/
def package_a : PhonePackage :=
  { monthly_fee := 15
    call_fee := 0.1
    bill := λ x => 0.1 * x + 15 }

/-- Package B with no monthly fee and a call fee of 0.15 yuan per minute. -/
def package_b : PhonePackage :=
  { monthly_fee := 0
    call_fee := 0.15
    bill := λ x => 0.15 * x }

theorem phone_package_comparison :
  ∃ (x : ℝ),
    (x > 0) ∧
    (package_a.bill x = package_b.bill x) ∧
    (x = 300) ∧
    (∀ y : ℝ, y > x → package_a.bill y < package_b.bill y) :=
by sorry

end NUMINAMATH_CALUDE_phone_package_comparison_l3215_321599


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3215_321535

theorem adult_ticket_cost (num_students : Nat) (num_adults : Nat) (student_ticket_cost : Nat) (total_cost : Nat) :
  num_students = 12 →
  num_adults = 4 →
  student_ticket_cost = 1 →
  total_cost = 24 →
  (total_cost - num_students * student_ticket_cost) / num_adults = 3 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3215_321535


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l3215_321567

/-- The total number of tissues brought by three kindergartner groups -/
def total_tissues (group1 group2 group3 tissues_per_box : ℕ) : ℕ :=
  (group1 + group2 + group3) * tissues_per_box

/-- Theorem: The total number of tissues brought by the kindergartner groups is 1200 -/
theorem kindergarten_tissues :
  total_tissues 9 10 11 40 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l3215_321567


namespace NUMINAMATH_CALUDE_sin_negative_1020_degrees_l3215_321523

theorem sin_negative_1020_degrees : Real.sin ((-1020 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1020_degrees_l3215_321523


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l3215_321583

/-- The area of a regular hexagon with side length 8 inches is 96√3 square inches. -/
theorem regular_hexagon_area :
  let side_length : ℝ := 8
  let area : ℝ := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  area = 96 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_area_l3215_321583


namespace NUMINAMATH_CALUDE_prob_rain_weekend_l3215_321550

/-- Probability of rain on Saturday -/
def prob_rain_saturday : ℝ := 0.6

/-- Probability of rain on Sunday given it rained on Saturday -/
def prob_rain_sunday_given_rain_saturday : ℝ := 0.7

/-- Probability of rain on Sunday given it didn't rain on Saturday -/
def prob_rain_sunday_given_no_rain_saturday : ℝ := 0.4

/-- Theorem: The probability of rain over the weekend (at least one day) is 76% -/
theorem prob_rain_weekend : 
  1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday_given_no_rain_saturday) = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_weekend_l3215_321550


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3215_321595

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0) → 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 2*y - 8 = 0) → 
  (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → 
    (x - 2)^2 + (y - 1)^2 = 9) → 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3215_321595


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solve_for_y_l3215_321539

/-- Given an arithmetic sequence with the first three terms as specified,
    prove that the value of y is 5/3 -/
theorem arithmetic_sequence_solve_for_y :
  ∀ (seq : ℕ → ℚ),
  (seq 0 = 2/3) →
  (seq 1 = y + 2) →
  (seq 2 = 4*y) →
  (∀ n, seq (n+1) - seq n = seq (n+2) - seq (n+1)) →
  (y = 5/3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solve_for_y_l3215_321539


namespace NUMINAMATH_CALUDE_sine_inequality_l3215_321536

theorem sine_inequality (n : ℕ+) (θ : ℝ) : |Real.sin (n * θ)| ≤ n * |Real.sin θ| := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l3215_321536


namespace NUMINAMATH_CALUDE_smallest_block_size_l3215_321514

/-- Given a rectangular block made of N identical 1-cm cubes, where 378 cubes are not visible
    when three faces are visible, the smallest possible value of N is 560. -/
theorem smallest_block_size (N : ℕ) : 
  (∃ l m n : ℕ, (l - 1) * (m - 1) * (n - 1) = 378 ∧ N = l * m * n) →
  (∀ N' : ℕ, (∃ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 378 ∧ N' = l' * m' * n') → N' ≥ N) →
  N = 560 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_l3215_321514


namespace NUMINAMATH_CALUDE_group_messages_in_week_l3215_321530

/-- Calculates the total number of messages sent in a week by remaining members of a group -/
theorem group_messages_in_week 
  (initial_members : ℕ) 
  (removed_members : ℕ) 
  (messages_per_day : ℕ) 
  (days_in_week : ℕ) 
  (h1 : initial_members = 150) 
  (h2 : removed_members = 20) 
  (h3 : messages_per_day = 50) 
  (h4 : days_in_week = 7) :
  (initial_members - removed_members) * messages_per_day * days_in_week = 45500 :=
by sorry

end NUMINAMATH_CALUDE_group_messages_in_week_l3215_321530


namespace NUMINAMATH_CALUDE_tan_37_5_deg_identity_l3215_321545

theorem tan_37_5_deg_identity : 
  (Real.tan (37.5 * π / 180)) / (1 - (Real.tan (37.5 * π / 180))^2) = 1 + (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_37_5_deg_identity_l3215_321545


namespace NUMINAMATH_CALUDE_books_in_fiction_section_l3215_321534

theorem books_in_fiction_section 
  (initial_books : ℕ) 
  (books_left : ℕ) 
  (history_books : ℕ) 
  (children_books : ℕ) 
  (wrong_place_books : ℕ) 
  (h1 : initial_books = 51) 
  (h2 : books_left = 16) 
  (h3 : history_books = 12) 
  (h4 : children_books = 8) 
  (h5 : wrong_place_books = 4) : 
  initial_books - books_left - history_books - (children_books - wrong_place_books) = 19 := by
  sorry

end NUMINAMATH_CALUDE_books_in_fiction_section_l3215_321534


namespace NUMINAMATH_CALUDE_psychology_majors_percentage_l3215_321537

theorem psychology_majors_percentage (total_students : ℝ) (h1 : total_students > 0) : 
  let freshmen := 0.60 * total_students
  let liberal_arts_freshmen := 0.40 * freshmen
  let psych_majors := 0.048 * total_students
  psych_majors / liberal_arts_freshmen = 0.20 := by
sorry

end NUMINAMATH_CALUDE_psychology_majors_percentage_l3215_321537


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3215_321564

theorem mans_rowing_speed 
  (v : ℝ) -- Man's rowing speed in still water
  (c : ℝ) -- Speed of the current
  (h1 : c = 1.5) -- The current speed is 1.5 km/hr
  (h2 : (v + c) * 1 = (v - c) * 2) -- It takes twice as long to row upstream as downstream
  : v = 4.5 := by
sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l3215_321564


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3215_321560

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3215_321560


namespace NUMINAMATH_CALUDE_hacker_guarantee_l3215_321513

/-- A computer network with the given properties -/
structure ComputerNetwork where
  num_computers : ℕ
  is_connected : Bool
  no_shared_cycle_vertices : Bool

/-- The game state -/
structure GameState where
  network : ComputerNetwork
  hacked_computers : ℕ
  protected_computers : ℕ

/-- The game rules -/
def game_rules (state : GameState) : Bool :=
  state.hacked_computers + state.protected_computers ≤ state.network.num_computers

/-- The theorem statement -/
theorem hacker_guarantee (network : ComputerNetwork) 
  (h1 : network.num_computers = 2008)
  (h2 : network.is_connected = true)
  (h3 : network.no_shared_cycle_vertices = true) :
  ∃ (final_state : GameState), 
    final_state.network = network ∧ 
    game_rules final_state ∧ 
    final_state.hacked_computers ≥ 671 :=
sorry

end NUMINAMATH_CALUDE_hacker_guarantee_l3215_321513


namespace NUMINAMATH_CALUDE_prob_ratio_l3215_321558

/- Define the total number of cards -/
def total_cards : ℕ := 50

/- Define the number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/- Define the number of cards for each number -/
def cards_per_number : ℕ := 5

/- Define the number of cards drawn -/
def cards_drawn : ℕ := 5

/- Function to calculate the probability of drawing 5 cards of the same number -/
def prob_same_number : ℚ :=
  (distinct_numbers : ℚ) / Nat.choose total_cards cards_drawn

/- Function to calculate the probability of drawing 4 cards of one number and 1 of another -/
def prob_four_and_one : ℚ :=
  (distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number : ℚ) / 
  Nat.choose total_cards cards_drawn

/- Theorem stating the ratio of probabilities -/
theorem prob_ratio : 
  prob_four_and_one / prob_same_number = 225 := by sorry

end NUMINAMATH_CALUDE_prob_ratio_l3215_321558


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_curve_l3215_321540

def curve (x : ℝ) : ℝ := -2 * x^2

theorem equilateral_triangle_on_curve :
  ∃ (P Q : ℝ × ℝ),
    (P.2 = curve P.1) ∧
    (Q.2 = curve Q.1) ∧
    (P.1 = -Q.1) ∧
    (P.2 = Q.2) ∧
    (dist P (0, 0) = dist Q (0, 0)) ∧
    (dist P Q = dist P (0, 0)) ∧
    (dist P (0, 0) = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_curve_l3215_321540


namespace NUMINAMATH_CALUDE_q_value_l3215_321593

theorem q_value (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1/p + 1/q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_q_value_l3215_321593


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3215_321525

/-- Given a line with equation 2x + 4y = -17, prove that its slope (and the slope of any parallel line) is -1/2 -/
theorem parallel_line_slope (x y : ℝ) (h : 2 * x + 4 * y = -17) :
  ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3215_321525


namespace NUMINAMATH_CALUDE_shortest_side_length_l3215_321556

theorem shortest_side_length (A B C : Real) (a b c : Real) : 
  B = π/4 → C = π/3 → c = 1 → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  b / (Real.sin B) = c / (Real.sin C) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  b ≤ a ∧ b ≤ c → 
  b = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_shortest_side_length_l3215_321556


namespace NUMINAMATH_CALUDE_coconut_grove_average_yield_l3215_321557

/-- The yield of coconuts per year for a group of trees -/
structure CoconutYield where
  trees : ℕ
  nuts_per_year : ℕ

/-- The total yield of coconuts from multiple groups of trees -/
def total_yield (yields : List CoconutYield) : ℕ :=
  yields.map (λ y => y.trees * y.nuts_per_year) |>.sum

/-- The total number of trees from multiple groups -/
def total_trees (yields : List CoconutYield) : ℕ :=
  yields.map (λ y => y.trees) |>.sum

/-- The average yield per tree per year -/
def average_yield (yields : List CoconutYield) : ℚ :=
  (total_yield yields : ℚ) / (total_trees yields : ℚ)

theorem coconut_grove_average_yield : 
  let yields : List CoconutYield := [
    { trees := 3, nuts_per_year := 60 },
    { trees := 2, nuts_per_year := 120 },
    { trees := 1, nuts_per_year := 180 }
  ]
  average_yield yields = 100 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_average_yield_l3215_321557


namespace NUMINAMATH_CALUDE_distance_from_origin_implies_k_range_l3215_321515

theorem distance_from_origin_implies_k_range (k : ℝ) (h1 : k > 0) :
  (∃ x : ℝ, x ≠ 0 ∧ x^2 + (k/x)^2 = 1) → 0 < k ∧ k ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_implies_k_range_l3215_321515


namespace NUMINAMATH_CALUDE_optimal_large_trucks_for_fruit_loading_l3215_321588

/-- Represents the problem of loading fruits onto trucks -/
structure FruitLoading where
  total_fruits : ℕ
  large_truck_capacity : ℕ
  small_truck_capacity : ℕ

/-- Checks if a given number of large trucks is optimal for the fruit loading problem -/
def is_optimal_large_trucks (problem : FruitLoading) (num_large_trucks : ℕ) : Prop :=
  let remaining_fruits := problem.total_fruits - num_large_trucks * problem.large_truck_capacity
  -- The remaining fruits can be loaded onto small trucks without leftovers
  remaining_fruits % problem.small_truck_capacity = 0 ∧
  -- Using one more large truck would exceed the total fruits
  (num_large_trucks + 1) * problem.large_truck_capacity > problem.total_fruits

/-- Theorem stating that 8 large trucks is the optimal solution for the given problem -/
theorem optimal_large_trucks_for_fruit_loading :
  let problem : FruitLoading := ⟨134, 15, 7⟩
  is_optimal_large_trucks problem 8 :=
by sorry

end NUMINAMATH_CALUDE_optimal_large_trucks_for_fruit_loading_l3215_321588


namespace NUMINAMATH_CALUDE_quadratic_root_product_l3215_321587

theorem quadratic_root_product (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I) ^ 2 + p * (1 - Complex.I) + q = 0 →
  p * q = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l3215_321587


namespace NUMINAMATH_CALUDE_handshake_count_l3215_321579

theorem handshake_count (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l3215_321579


namespace NUMINAMATH_CALUDE_additional_courses_is_two_l3215_321524

/-- Represents the wall construction problem --/
structure WallProblem where
  initial_courses : ℕ
  bricks_per_course : ℕ
  total_bricks : ℕ

/-- Calculates the number of additional courses added to the wall --/
def additional_courses (w : WallProblem) : ℕ :=
  let initial_bricks := w.initial_courses * w.bricks_per_course
  let remaining_bricks := w.total_bricks - initial_bricks + (w.bricks_per_course / 2)
  remaining_bricks / w.bricks_per_course

/-- Theorem stating that the number of additional courses is 2 --/
theorem additional_courses_is_two (w : WallProblem) 
    (h1 : w.initial_courses = 3)
    (h2 : w.bricks_per_course = 400)
    (h3 : w.total_bricks = 1800) : 
  additional_courses w = 2 := by
  sorry

#eval additional_courses { initial_courses := 3, bricks_per_course := 400, total_bricks := 1800 }

end NUMINAMATH_CALUDE_additional_courses_is_two_l3215_321524


namespace NUMINAMATH_CALUDE_car_distance_l3215_321578

theorem car_distance (total_distance : ℝ) (foot_fraction : ℝ) (bus_fraction : ℝ) :
  total_distance = 90 →
  foot_fraction = 1/5 →
  bus_fraction = 2/3 →
  total_distance * (1 - foot_fraction - bus_fraction) = 12 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l3215_321578


namespace NUMINAMATH_CALUDE_hyperbola_circle_max_radius_l3215_321569

/-- Given a hyperbola and a circle with specific properties, prove that the maximum radius of the circle is √3 -/
theorem hyperbola_circle_max_radius (a b r : ℝ) (e : ℝ) :
  a > 0 →
  b > 0 →
  r > 0 →
  e ≤ 2 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, (x - 2)^2 + y^2 = r^2) →
  (∃ x y : ℝ, b * x + a * y = 0 ∨ b * x - a * y = 0) →
  (∀ x y : ℝ, (b * x + a * y = 0 ∨ b * x - a * y = 0) → 
    ((x - 2)^2 + y^2 = r^2 → (x - 2)^2 + y^2 ≥ r^2)) →
  r ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_max_radius_l3215_321569


namespace NUMINAMATH_CALUDE_common_factor_of_2a2_and_4ab_l3215_321543

theorem common_factor_of_2a2_and_4ab :
  ∀ (a b : ℤ), ∃ (k₁ k₂ : ℤ), 2 * a^2 = (2 * a) * k₁ ∧ 4 * a * b = (2 * a) * k₂ ∧
  (∀ (d : ℤ), (∃ (m₁ m₂ : ℤ), 2 * a^2 = d * m₁ ∧ 4 * a * b = d * m₂) → d ∣ (2 * a)) :=
by sorry

end NUMINAMATH_CALUDE_common_factor_of_2a2_and_4ab_l3215_321543


namespace NUMINAMATH_CALUDE_lcm_equality_implies_equal_no_lcm_equality_with_shift_l3215_321552

theorem lcm_equality_implies_equal (a b : ℕ+) :
  Nat.lcm a (a + 5) = Nat.lcm b (b + 5) → a = b := by sorry

theorem no_lcm_equality_with_shift :
  ¬ ∃ (a b c : ℕ+), Nat.lcm a b = Nat.lcm (a + c) (b + c) := by sorry

end NUMINAMATH_CALUDE_lcm_equality_implies_equal_no_lcm_equality_with_shift_l3215_321552


namespace NUMINAMATH_CALUDE_gloria_pencils_l3215_321533

theorem gloria_pencils (G : ℕ) (h : G + 99 = 101) : G = 2 := by
  sorry

end NUMINAMATH_CALUDE_gloria_pencils_l3215_321533


namespace NUMINAMATH_CALUDE_basketball_league_female_fraction_l3215_321502

theorem basketball_league_female_fraction :
  -- Define variables
  let last_year_males : ℕ := 30
  let last_year_females : ℕ := 15  -- Derived from the solution
  let male_increase_rate : ℚ := 11/10
  let female_increase_rate : ℚ := 5/4
  let total_increase_rate : ℚ := 23/20

  -- Define this year's participants
  let this_year_males : ℚ := last_year_males * male_increase_rate
  let this_year_females : ℚ := last_year_females * female_increase_rate
  let this_year_total : ℚ := (last_year_males + last_year_females) * total_increase_rate

  -- The fraction of female participants this year
  this_year_females / this_year_total = 75 / 207 := by
sorry

end NUMINAMATH_CALUDE_basketball_league_female_fraction_l3215_321502


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3215_321584

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) :
  x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3215_321584


namespace NUMINAMATH_CALUDE_xy_squared_equals_one_l3215_321590

theorem xy_squared_equals_one (x y : ℝ) (h : |x - 2| + (3 + y)^2 = 0) : (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_equals_one_l3215_321590


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_cubic_equation_solutions_l3215_321589

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 2*x - 4 = 0) ↔ (x = Real.sqrt 5 - 1 ∨ x = -Real.sqrt 5 - 1) :=
sorry

theorem cubic_equation_solutions (x : ℝ) :
  (3*x*(x-5) = 5-x) ↔ (x = 5 ∨ x = -1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_cubic_equation_solutions_l3215_321589


namespace NUMINAMATH_CALUDE_six_digit_multiply_rearrange_l3215_321546

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 2

def rearranged (n m : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 200000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    m = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 2

def digit_sum (n : ℕ) : ℕ :=
  (n / 100000) + ((n / 10000) % 10) + ((n / 1000) % 10) +
  ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem six_digit_multiply_rearrange (n : ℕ) :
  is_valid_number n → rearranged n (3 * n) → digit_sum n = 27 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_multiply_rearrange_l3215_321546


namespace NUMINAMATH_CALUDE_book_profit_rate_l3215_321517

/-- Calculate the rate of profit given the cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 80 is 60% -/
theorem book_profit_rate :
  let cost_price : ℚ := 50
  let selling_price : ℚ := 80
  rate_of_profit cost_price selling_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_profit_rate_l3215_321517


namespace NUMINAMATH_CALUDE_kates_retirement_fund_l3215_321568

/-- 
Given an initial retirement fund value and a decrease amount, 
calculate the current value of the retirement fund.
-/
def current_fund_value (initial_value decrease : ℕ) : ℕ :=
  initial_value - decrease

/-- 
Theorem: Given Kate's initial retirement fund value of $1472 and a decrease of $12, 
the current value of her retirement fund is $1460.
-/
theorem kates_retirement_fund : 
  current_fund_value 1472 12 = 1460 := by
  sorry

end NUMINAMATH_CALUDE_kates_retirement_fund_l3215_321568


namespace NUMINAMATH_CALUDE_expected_hypertension_cases_l3215_321509

/-- Given a population where 1 out of 3 individuals has a condition,
    prove that the expected number of individuals with the condition
    in a sample of 450 is 150. -/
theorem expected_hypertension_cases (
  total_sample : ℕ
  ) (h1 : total_sample = 450)
  (probability : ℚ)
  (h2 : probability = 1 / 3) :
  ↑total_sample * probability = 150 :=
sorry

end NUMINAMATH_CALUDE_expected_hypertension_cases_l3215_321509


namespace NUMINAMATH_CALUDE_quadratic_roots_in_unit_interval_l3215_321522

theorem quadratic_roots_in_unit_interval (a b c : ℤ) (ha : a > 0) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ∧
    (a : ℝ) * y^2 + (b : ℝ) * y + (c : ℝ) = 0) : 
  a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_in_unit_interval_l3215_321522


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_linear_expression_factorization_l3215_321518

-- Problem 1
theorem difference_of_squares_factorization (x : ℝ) :
  4 * x^2 - 9 = (2*x + 3) * (2*x - 3) := by sorry

-- Problem 2
theorem linear_expression_factorization (a b x y : ℝ) :
  2*a*(x - y) - 3*b*(y - x) = (x - y)*(2*a + 3*b) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_linear_expression_factorization_l3215_321518


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3215_321538

/-- A quadratic trinomial in x and y -/
def QuadraticTrinomial (a b c : ℝ) := fun (x y : ℝ) ↦ a*x^2 + b*x*y + c*y^2

/-- Predicate for a quadratic trinomial being a perfect square -/
def IsPerfectSquare (q : (ℝ → ℝ → ℝ)) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), q x y = (a*x + b*y)^2

theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquare (QuadraticTrinomial 4 m 9) → (m = 12 ∨ m = -12) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3215_321538


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3215_321562

theorem ceiling_floor_difference : ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3215_321562


namespace NUMINAMATH_CALUDE_courtyard_length_l3215_321521

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) :
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 14400 →
  width * (num_bricks * brick_length * brick_width / width) = 18 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_length_l3215_321521


namespace NUMINAMATH_CALUDE_lemonade_proportion_lemons_for_lemonade_l3215_321554

theorem lemonade_proportion (lemons_initial : ℝ) (gallons_initial : ℝ) (gallons_target : ℝ) :
  lemons_initial > 0 ∧ gallons_initial > 0 ∧ gallons_target > 0 →
  let lemons_target := (lemons_initial * gallons_target) / gallons_initial
  lemons_initial / gallons_initial = lemons_target / gallons_target :=
by
  sorry

theorem lemons_for_lemonade :
  let lemons_initial : ℝ := 36
  let gallons_initial : ℝ := 48
  let gallons_target : ℝ := 10
  (lemons_initial * gallons_target) / gallons_initial = 7.5 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_proportion_lemons_for_lemonade_l3215_321554


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3215_321585

/-- The speed of a boat in still water, given its downstream and upstream distances in one hour -/
theorem boat_speed_in_still_water (downstream upstream : ℝ) 
  (h_downstream : downstream = 11) 
  (h_upstream : upstream = 5) : 
  (downstream + upstream) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3215_321585


namespace NUMINAMATH_CALUDE_smallest_positive_angle_for_negative_2015_l3215_321531

-- Define the concept of angle equivalence
def angle_equivalent (a b : ℤ) : Prop := ∃ k : ℤ, b = a + 360 * k

-- Define the function to find the smallest positive equivalent angle
def smallest_positive_equivalent (a : ℤ) : ℤ :=
  (a % 360 + 360) % 360

-- Theorem statement
theorem smallest_positive_angle_for_negative_2015 :
  smallest_positive_equivalent (-2015) = 145 ∧
  angle_equivalent (-2015) 145 ∧
  ∀ x : ℤ, 0 < x ∧ x < 145 → ¬(angle_equivalent (-2015) x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_for_negative_2015_l3215_321531


namespace NUMINAMATH_CALUDE_john_has_14_burritos_left_l3215_321574

/-- The number of burritos John has left after buying, receiving a free box, giving away some, and eating for 10 days. -/
def burritos_left : ℕ :=
  let total_burritos : ℕ := 15 + 20 + 25 + 5
  let given_away : ℕ := (total_burritos / 3 : ℕ)
  let after_giving : ℕ := total_burritos - given_away
  let eaten : ℕ := 3 * 10
  after_giving - eaten

/-- Theorem stating that John has 14 burritos left -/
theorem john_has_14_burritos_left : burritos_left = 14 := by
  sorry

end NUMINAMATH_CALUDE_john_has_14_burritos_left_l3215_321574


namespace NUMINAMATH_CALUDE_cos_sum_specific_values_l3215_321553

theorem cos_sum_specific_values (α β : ℝ) :
  Complex.exp (α * Complex.I) = (8 : ℝ) / 17 + (15 : ℝ) / 17 * Complex.I →
  Complex.exp (β * Complex.I) = -(5 : ℝ) / 13 + (12 : ℝ) / 13 * Complex.I →
  Real.cos (α + β) = -(220 : ℝ) / 221 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_specific_values_l3215_321553


namespace NUMINAMATH_CALUDE_product_of_one_plus_roots_l3215_321500

theorem product_of_one_plus_roots (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 10 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 51 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_roots_l3215_321500


namespace NUMINAMATH_CALUDE_box_ratio_proof_l3215_321591

def box_problem (total_balls white_balls : ℕ) (blue_white_diff : ℕ) : Prop :=
  let blue_balls : ℕ := white_balls + blue_white_diff
  let red_balls : ℕ := total_balls - (white_balls + blue_balls)
  (red_balls : ℚ) / blue_balls = 2 / 1

theorem box_ratio_proof :
  box_problem 100 16 12 := by
  sorry

end NUMINAMATH_CALUDE_box_ratio_proof_l3215_321591


namespace NUMINAMATH_CALUDE_sixth_term_value_l3215_321596

/-- Represents a geometric sequence --/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- Properties of the geometric sequence --/
def GeometricSequence.properties (seq : GeometricSequence) : Prop :=
  -- Sum of first four terms is 40
  seq.a * (1 + seq.r + seq.r^2 + seq.r^3) = 40 ∧
  -- Fifth term is 32
  seq.a * seq.r^4 = 32

/-- Sixth term of the geometric sequence --/
def GeometricSequence.sixthTerm (seq : GeometricSequence) : ℝ :=
  seq.a * seq.r^5

/-- Theorem stating that the sixth term is 1280/15 --/
theorem sixth_term_value (seq : GeometricSequence) 
  (h : seq.properties) : seq.sixthTerm = 1280/15 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l3215_321596


namespace NUMINAMATH_CALUDE_jorge_ticket_cost_l3215_321512

def number_of_tickets : ℕ := 24
def price_per_ticket : ℚ := 7
def discount_percentage : ℚ := 50 / 100

def total_cost_with_discount : ℚ :=
  number_of_tickets * price_per_ticket * (1 - discount_percentage)

theorem jorge_ticket_cost : total_cost_with_discount = 84 := by
  sorry

end NUMINAMATH_CALUDE_jorge_ticket_cost_l3215_321512


namespace NUMINAMATH_CALUDE_square_binomial_simplification_l3215_321503

theorem square_binomial_simplification (x : ℝ) (h : 3 * x^2 - 12 ≥ 0) :
  (7 - Real.sqrt (3 * x^2 - 12))^2 = 3 * x^2 + 37 - 14 * Real.sqrt (3 * x^2 - 12) := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_simplification_l3215_321503


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_one_or_six_l3215_321577

/-- The number of three-digit whole numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of choices for the first digit (excluding 1 and 6) -/
def first_digit_choices : ℕ := 7

/-- The number of choices for the second and third digits (excluding 1 and 6) -/
def other_digit_choices : ℕ := 8

/-- The number of three-digit numbers without 1 or 6 -/
def numbers_without_one_or_six : ℕ := first_digit_choices * other_digit_choices * other_digit_choices

theorem three_digit_numbers_with_one_or_six : 
  total_three_digit_numbers - numbers_without_one_or_six = 452 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_one_or_six_l3215_321577


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3215_321572

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3215_321572


namespace NUMINAMATH_CALUDE_point_location_l3215_321510

theorem point_location (α : Real) (h : α = 5 * Real.pi / 8) :
  let P : Real × Real := (Real.sin α, Real.tan α)
  P.1 > 0 ∧ P.2 < 0 :=
sorry

end NUMINAMATH_CALUDE_point_location_l3215_321510


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_equations_l3215_321555

-- Define the foci
def F₁ : ℝ × ℝ := (0, -5)
def F₂ : ℝ × ℝ := (0, 5)

-- Define the intersection point
def P : ℝ × ℝ := (3, 4)

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop :=
  y^2 / 40 + x^2 / 15 = 1

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

-- Define the asymptote equation
def is_on_asymptote (x y : ℝ) : Prop :=
  y = (4/3) * x

-- Theorem statement
theorem hyperbola_ellipse_equations :
  (is_on_ellipse P.1 P.2) ∧
  (is_on_hyperbola P.1 P.2) ∧
  (is_on_asymptote P.1 P.2) ∧
  (F₁.2 = -F₂.2) ∧
  (F₁.1 = F₂.1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_equations_l3215_321555


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l3215_321565

theorem lcm_gcf_problem (n : ℕ) :
  Nat.lcm n 12 = 54 ∧ Nat.gcd n 12 = 8 → n = 36 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l3215_321565


namespace NUMINAMATH_CALUDE_max_value_sum_of_inverses_l3215_321542

theorem max_value_sum_of_inverses (a b : ℝ) (h : a + b = 4) :
  (∀ x y : ℝ, x + y = 4 → 1 / (x^2 + 1) + 1 / (y^2 + 1) ≤ 1 / (a^2 + 1) + 1 / (b^2 + 1)) →
  1 / (a^2 + 1) + 1 / (b^2 + 1) = (Real.sqrt 5 + 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_inverses_l3215_321542


namespace NUMINAMATH_CALUDE_supplementary_angle_difference_l3215_321575

theorem supplementary_angle_difference : 
  let angle1 : ℝ := 99
  let angle2 : ℝ := 81
  -- Supplementary angles sum to 180°
  angle1 + angle2 = 180 →
  -- The difference between the larger and smaller angle is 18°
  max angle1 angle2 - min angle1 angle2 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_difference_l3215_321575


namespace NUMINAMATH_CALUDE_certain_number_problem_l3215_321511

theorem certain_number_problem (x y z : ℝ) : 
  x + y = 15 →
  y = 7 →
  3 * x = z * y - 11 →
  z = 5 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3215_321511


namespace NUMINAMATH_CALUDE_tearing_process_l3215_321597

/-- Represents the number of parts after a series of tearing operations -/
def NumParts : ℕ → ℕ
  | 0 => 1  -- Start with one piece
  | n + 1 => NumParts n + 2  -- Each tear adds 2 parts

theorem tearing_process (n : ℕ) :
  ∀ k, Odd (NumParts k) ∧ 
    (¬∃ m, NumParts m = 100) ∧
    (∃ m, NumParts m = 2017) := by
  sorry

#eval NumParts 1008  -- Should evaluate to 2017

end NUMINAMATH_CALUDE_tearing_process_l3215_321597


namespace NUMINAMATH_CALUDE_minimum_cost_purchase_l3215_321501

/-- Represents the unit price and quantity of an ingredient -/
structure Ingredient where
  price : ℝ
  quantity : ℝ

/-- Represents the purchase of two ingredients -/
structure Purchase where
  A : Ingredient
  B : Ingredient

def total_cost (p : Purchase) : ℝ := p.A.price * p.A.quantity + p.B.price * p.B.quantity

def total_quantity (p : Purchase) : ℝ := p.A.quantity + p.B.quantity

theorem minimum_cost_purchase :
  ∀ (p : Purchase),
    p.A.price + p.B.price = 68 →
    5 * p.A.price + 3 * p.B.price = 280 →
    total_quantity p = 36 →
    p.A.quantity ≥ 2 * p.B.quantity →
    total_cost p ≥ 1272 ∧
    (total_cost p = 1272 ↔ p.A.quantity = 24 ∧ p.B.quantity = 12) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cost_purchase_l3215_321501


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l3215_321581

theorem lcm_factor_problem (A B : ℕ) (H : ℕ) (X Y : ℕ) :
  H = 23 →
  Y = 14 →
  max A B = 322 →
  H = Nat.gcd A B →
  Nat.lcm A B = H * X * Y →
  X = 23 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l3215_321581


namespace NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l3215_321566

theorem complex_subtraction_and_multiplication :
  (7 - 3*I) - 3*(2 + 4*I) = 1 - 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l3215_321566


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l3215_321519

/-- Proves that a man walking at 6 km/hr takes 15 minutes to cross a bridge of 1500 meters in length. -/
theorem bridge_crossing_time : 
  let walking_speed : ℝ := 6 -- km/hr
  let bridge_length : ℝ := 1.5 -- km (1500 meters)
  let crossing_time : ℝ := bridge_length / walking_speed * 60 -- in minutes
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l3215_321519


namespace NUMINAMATH_CALUDE_blue_pens_count_l3215_321561

/-- Given a total number of pens and a number of black pens, 
    calculate the number of blue pens. -/
def blue_pens (total : ℕ) (black : ℕ) : ℕ :=
  total - black

/-- Theorem: When the total number of pens is 8 and the number of black pens is 4,
    the number of blue pens is 4. -/
theorem blue_pens_count : blue_pens 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l3215_321561


namespace NUMINAMATH_CALUDE_carls_yard_area_l3215_321527

/-- Represents a rectangular yard with fence posts. -/
structure FencedYard where
  short_posts : ℕ  -- Number of posts on the shorter side
  long_posts : ℕ   -- Number of posts on the longer side
  post_spacing : ℕ -- Distance between adjacent posts in yards

/-- Calculates the total number of fence posts. -/
def total_posts (yard : FencedYard) : ℕ :=
  2 * (yard.short_posts + yard.long_posts) - 4

/-- Calculates the area of the fenced yard in square yards. -/
def yard_area (yard : FencedYard) : ℕ :=
  (yard.short_posts - 1) * (yard.long_posts - 1) * yard.post_spacing^2

/-- Theorem stating the area of Carl's yard. -/
theorem carls_yard_area :
  ∃ (yard : FencedYard),
    yard.short_posts = 4 ∧
    yard.long_posts = 12 ∧
    yard.post_spacing = 5 ∧
    total_posts yard = 24 ∧
    yard.long_posts = 3 * yard.short_posts ∧
    yard_area yard = 825 :=
by sorry

end NUMINAMATH_CALUDE_carls_yard_area_l3215_321527


namespace NUMINAMATH_CALUDE_annie_ride_distance_l3215_321570

/-- Taxi fare calculation --/
def taxi_fare (start_fee : ℚ) (toll : ℚ) (per_mile : ℚ) (miles : ℚ) : ℚ :=
  start_fee + toll + per_mile * miles

theorem annie_ride_distance :
  let mike_start_fee : ℚ := 25/10
  let annie_start_fee : ℚ := 25/10
  let mike_toll : ℚ := 0
  let annie_toll : ℚ := 5
  let per_mile : ℚ := 1/4
  let mike_miles : ℚ := 34
  let annie_miles : ℚ := 14

  taxi_fare mike_start_fee mike_toll per_mile mike_miles =
  taxi_fare annie_start_fee annie_toll per_mile annie_miles :=
by
  sorry


end NUMINAMATH_CALUDE_annie_ride_distance_l3215_321570


namespace NUMINAMATH_CALUDE_cubes_with_le_four_neighbors_eq_144_l3215_321548

/-- Represents a parallelepiped constructed from unit cubes. -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  sides_gt_four : min a (min b c) > 4
  internal_cubes : (a - 2) * (b - 2) * (c - 2) = 836

/-- The number of cubes with no more than four neighbors in the parallelepiped. -/
def cubes_with_le_four_neighbors (p : Parallelepiped) : ℕ :=
  4 * (p.a - 2 + p.b - 2 + p.c - 2) + 8

/-- Theorem stating that the number of cubes with no more than four neighbors is 144. -/
theorem cubes_with_le_four_neighbors_eq_144 (p : Parallelepiped) :
  cubes_with_le_four_neighbors p = 144 := by
  sorry

end NUMINAMATH_CALUDE_cubes_with_le_four_neighbors_eq_144_l3215_321548


namespace NUMINAMATH_CALUDE_matrix_cube_equals_negative_identity_l3215_321582

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

theorem matrix_cube_equals_negative_identity :
  A ^ 3 = !![(-1 : ℤ), 0; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_equals_negative_identity_l3215_321582


namespace NUMINAMATH_CALUDE_part_one_part_two_l3215_321505

-- Define the conditions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := x^2 - 6*x + 8 < 0 ∧ x^2 - 8*x + 15 > 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, q x → (∃ a : ℝ, a > 0 ∧ p x a)) ∧
  (∃ x : ℝ, (∃ a : ℝ, a > 0 ∧ p x a) ∧ ¬q x) ↔
  (∃ a : ℝ, 1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3215_321505


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3215_321547

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^k : ℤ) ∣ (12^500 - 6^500) ∧ 
  ∀ (m : ℕ), (2^m : ℤ) ∣ (12^500 - 6^500) → m ≤ k :=
by
  use 501
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3215_321547


namespace NUMINAMATH_CALUDE_yellow_marbles_total_l3215_321526

/-- The total number of yellow marbles after redistribution -/
def total_marbles_after_redistribution : ℕ → ℕ → ℕ → ℕ → ℕ
  | mary_initial, joan, john, mary_to_tim =>
    (mary_initial - mary_to_tim) + joan + john + mary_to_tim

/-- Theorem stating the total number of yellow marbles after redistribution -/
theorem yellow_marbles_total
  (mary_initial : ℕ)
  (joan : ℕ)
  (john : ℕ)
  (mary_to_tim : ℕ)
  (h1 : mary_initial = 9)
  (h2 : joan = 3)
  (h3 : john = 7)
  (h4 : mary_to_tim = 4)
  (h5 : mary_initial ≥ mary_to_tim) :
  total_marbles_after_redistribution mary_initial joan john mary_to_tim = 19 :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_total_l3215_321526


namespace NUMINAMATH_CALUDE_larger_number_problem_l3215_321586

theorem larger_number_problem (x y : ℝ) : 
  x + y = 84 → y = 3 * x → max x y = 63 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3215_321586


namespace NUMINAMATH_CALUDE_positive_integer_solution_of_equation_l3215_321532

theorem positive_integer_solution_of_equation (x : ℕ+) :
  (4 * x.val^2 - 16 * x.val - 60 = 0) → x.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_of_equation_l3215_321532


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l3215_321598

/-- For a line with equation 2x + y + 1 = 0, its slope is -2 and y-intercept is -1 -/
theorem line_slope_and_intercept :
  ∀ (x y : ℝ), 2*x + y + 1 = 0 → 
  ∃ (k b : ℝ), k = -2 ∧ b = -1 ∧ y = k*x + b := by
sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l3215_321598


namespace NUMINAMATH_CALUDE_initial_rulers_l3215_321541

theorem initial_rulers (taken : ℕ) (remaining : ℕ) : taken = 25 → remaining = 21 → taken + remaining = 46 := by
  sorry

end NUMINAMATH_CALUDE_initial_rulers_l3215_321541


namespace NUMINAMATH_CALUDE_product_digits_sum_l3215_321592

/-- Converts a base-7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The product of 24₇ and 35₇ in base-7 --/
def productBase7 : ℕ := decimalToBase7 (base7ToDecimal 24 * base7ToDecimal 35)

theorem product_digits_sum :
  sumOfDigitsBase7 productBase7 = 15 :=
sorry

end NUMINAMATH_CALUDE_product_digits_sum_l3215_321592


namespace NUMINAMATH_CALUDE_compute_b_l3215_321506

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 21

-- State the theorem
theorem compute_b (a b : ℚ) :
  (f a b (3 + Real.sqrt 5) = 0) → b = -27.5 := by
  sorry

end NUMINAMATH_CALUDE_compute_b_l3215_321506


namespace NUMINAMATH_CALUDE_inequality_proof_l3215_321576

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9/16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3215_321576


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l3215_321507

/-- Represents the profit function for helmet sales -/
def profit_function (x : ℝ) : ℝ :=
  -20 * x^2 + 1400 * x - 60000

/-- The optimal selling price for helmets -/
def optimal_price : ℝ := 70

theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function optimal_price ≥ profit_function x :=
sorry

#check optimal_price_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l3215_321507


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l3215_321551

theorem distinct_prime_factors_of_30_factorial :
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l3215_321551


namespace NUMINAMATH_CALUDE_girls_in_sample_l3215_321573

/-- Calculates the number of girls in a stratified sample -/
def stratified_sample_girls (total_students : ℕ) (total_girls : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * total_girls) / total_students

/-- Proves that the number of girls in the stratified sample is 2 -/
theorem girls_in_sample (total_boys : ℕ) (total_girls : ℕ) (sample_size : ℕ) 
  (h1 : total_boys = 36)
  (h2 : total_girls = 18)
  (h3 : sample_size = 6) :
  stratified_sample_girls (total_boys + total_girls) total_girls sample_size = 2 := by
  sorry

#eval stratified_sample_girls 54 18 6

end NUMINAMATH_CALUDE_girls_in_sample_l3215_321573


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l3215_321504

theorem greatest_integer_satisfying_conditions : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k : ℕ), n = 11 * k - 1) ∧ 
  (∃ (l : ℕ), n = 9 * l + 2) ∧
  (∀ (m : ℕ), m < 150 → 
    (∃ (k' : ℕ), m = 11 * k' - 1) → 
    (∃ (l' : ℕ), m = 9 * l' + 2) → 
    m ≤ n) ∧
  n = 65 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l3215_321504


namespace NUMINAMATH_CALUDE_race_distance_theorem_l3215_321563

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  (speed_pos : speed > 0)

/-- Calculates the distance covered by a runner in a given time -/
def distance (r : Runner) (t : ℝ) : ℝ := r.speed * t

theorem race_distance_theorem 
  (A B C : Runner) 
  (race_length : ℝ)
  (AB_difference : ℝ)
  (BC_difference : ℝ)
  (h1 : race_length = 100)
  (h2 : AB_difference = 10)
  (h3 : BC_difference = 10)
  (h4 : distance A (race_length / A.speed) = race_length)
  (h5 : distance B (race_length / A.speed) = race_length - AB_difference)
  (h6 : distance C (race_length / B.speed) = race_length - BC_difference) :
  distance C (race_length / A.speed) = race_length - 19 := by
  sorry


end NUMINAMATH_CALUDE_race_distance_theorem_l3215_321563


namespace NUMINAMATH_CALUDE_fraction_value_l3215_321516

theorem fraction_value (x : ℝ) : (3 * x^2 + 9 * x + 15) / (3 * x^2 + 9 * x + 5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3215_321516


namespace NUMINAMATH_CALUDE_adams_total_school_time_l3215_321571

/-- The time Adam spent at school on each day of the week --/
structure SchoolWeek where
  monday : Float
  tuesday : Float
  wednesday : Float
  thursday : Float
  friday : Float

/-- Calculate the total time Adam spent at school during the week --/
def totalSchoolTime (week : SchoolWeek) : Float :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Adam's actual school week --/
def adamsWeek : SchoolWeek := {
  monday := 7.75,
  tuesday := 5.75,
  wednesday := 13.5,
  thursday := 8,
  friday := 6.75
}

/-- Theorem stating that Adam's total school time for the week is 41.75 hours --/
theorem adams_total_school_time :
  totalSchoolTime adamsWeek = 41.75 := by
  sorry


end NUMINAMATH_CALUDE_adams_total_school_time_l3215_321571


namespace NUMINAMATH_CALUDE_quadratic_roots_midpoint_l3215_321559

theorem quadratic_roots_midpoint (a b : ℝ) (x₁ x₂ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f 2014 = f 2016) →
  (x₁^2 + a*x₁ + b = 0) →
  (x₂^2 + a*x₂ + b = 0) →
  (x₁ + x₂) / 2 = 2015 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_midpoint_l3215_321559
