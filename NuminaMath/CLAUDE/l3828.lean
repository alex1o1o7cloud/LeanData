import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l3828_382810

theorem solve_equation (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x + 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3828_382810


namespace NUMINAMATH_CALUDE_three_digit_prime_discriminant_not_square_l3828_382806

theorem three_digit_prime_discriminant_not_square (A B C : ℕ) : 
  (100 * A + 10 * B + C).Prime → 
  ¬∃ (n : ℤ), B^2 - 4*A*C = n^2 := by
sorry

end NUMINAMATH_CALUDE_three_digit_prime_discriminant_not_square_l3828_382806


namespace NUMINAMATH_CALUDE_sum_of_divisors_30_l3828_382894

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_30_l3828_382894


namespace NUMINAMATH_CALUDE_books_on_third_shelf_l3828_382880

/-- Represents the number of books on each shelf of a bookcase -/
structure Bookcase where
  shelf1 : ℕ
  shelf2 : ℕ
  shelf3 : ℕ

/-- Defines the properties of the bookcase in the problem -/
def ProblemBookcase (b : Bookcase) : Prop :=
  b.shelf1 + b.shelf2 + b.shelf3 = 275 ∧
  b.shelf3 = 3 * b.shelf2 + 8 ∧
  b.shelf1 = 2 * b.shelf2 - 3

theorem books_on_third_shelf :
  ∀ b : Bookcase, ProblemBookcase b → b.shelf3 = 188 :=
by
  sorry


end NUMINAMATH_CALUDE_books_on_third_shelf_l3828_382880


namespace NUMINAMATH_CALUDE_pet_store_cats_l3828_382879

theorem pet_store_cats (white_cats black_cats total_cats : ℕ) 
  (h1 : white_cats = 2)
  (h2 : black_cats = 10)
  (h3 : total_cats = 15)
  : total_cats - (white_cats + black_cats) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_l3828_382879


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3828_382869

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x + y = 3 ∧ x - y = 1 → x = 2 ∧ y = 1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) : 
  x/2 - (y+1)/3 = 1 ∧ 3*x + 2*y = 10 → x = 3 ∧ y = 1/2 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3828_382869


namespace NUMINAMATH_CALUDE_cubic_extremum_difference_l3828_382833

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_extremum_difference (a b c : ℝ) :
  f' a b 2 = 0 → f' a b 1 = -3 →
  ∃ (min_val : ℝ), ∀ (x : ℝ), f a b c x ≥ min_val ∧ 
  ∀ (M : ℝ), ∃ (y : ℝ), f a b c y > M :=
by sorry

end NUMINAMATH_CALUDE_cubic_extremum_difference_l3828_382833


namespace NUMINAMATH_CALUDE_complex_roots_circle_l3828_382845

theorem complex_roots_circle (z : ℂ) : 
  (z + 2)^6 = 64 * z^6 → Complex.abs (z - (-2/3)) = 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_circle_l3828_382845


namespace NUMINAMATH_CALUDE_henry_final_distance_l3828_382837

-- Define the conversion factor from meters to feet
def metersToFeet : ℝ := 3.28084

-- Define Henry's movements
def northDistance : ℝ := 10 -- in meters
def eastDistance : ℝ := 30 -- in feet
def southDistance : ℝ := 10 * metersToFeet + 40 -- in feet

-- Calculate net southward movement
def netSouthDistance : ℝ := southDistance - (northDistance * metersToFeet)

-- Theorem to prove
theorem henry_final_distance :
  Real.sqrt (eastDistance ^ 2 + netSouthDistance ^ 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_henry_final_distance_l3828_382837


namespace NUMINAMATH_CALUDE_triangle_cosC_l3828_382873

theorem triangle_cosC (A B C : Real) (a b c : Real) : 
  -- Conditions
  (a = 2) →
  (b = 3) →
  (C = 2 * A) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Law of cosines
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  -- Conclusion
  Real.cos C = 1/4 := by sorry

end NUMINAMATH_CALUDE_triangle_cosC_l3828_382873


namespace NUMINAMATH_CALUDE_total_broadcasting_period_l3828_382898

/-- Given a music station that played commercials for a certain duration and maintained a specific ratio of music to commercials, this theorem proves the total broadcasting period. -/
theorem total_broadcasting_period 
  (commercial_duration : ℕ) 
  (music_ratio : ℕ) 
  (commercial_ratio : ℕ) 
  (h1 : commercial_duration = 40)
  (h2 : music_ratio = 9)
  (h3 : commercial_ratio = 5) :
  commercial_duration + (commercial_duration * music_ratio) / commercial_ratio = 112 :=
by sorry

end NUMINAMATH_CALUDE_total_broadcasting_period_l3828_382898


namespace NUMINAMATH_CALUDE_sequence_general_term_l3828_382802

/-- Given a sequence {aₙ} where a₁ = 1 and aₙ₊₁ - aₙ = 2ⁿ for all n ≥ 1,
    prove that the general term is given by aₙ = 2ⁿ - 1 -/
theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n) : 
    ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3828_382802


namespace NUMINAMATH_CALUDE_original_class_strength_l3828_382808

theorem original_class_strength (original_average : ℝ) (new_students : ℕ) 
  (new_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 12 →
  new_average = 32 →
  average_decrease = 4 →
  ∃ x : ℕ, x = 12 ∧ 
    (x + new_students : ℝ) * (original_average - average_decrease) = 
    x * original_average + (new_students : ℝ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_original_class_strength_l3828_382808


namespace NUMINAMATH_CALUDE_bug_return_probability_l3828_382822

/-- Probability of the bug being at the starting corner after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting corner on the eighth move -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3828_382822


namespace NUMINAMATH_CALUDE_buying_100_tickets_may_not_win_l3828_382839

/-- Represents a lottery with a given number of tickets and winning probability per ticket -/
structure Lottery where
  totalTickets : ℕ
  winningProbability : ℝ
  winningProbability_nonneg : 0 ≤ winningProbability
  winningProbability_le_one : winningProbability ≤ 1

/-- The probability of not winning when buying a certain number of tickets -/
def probNotWinning (lottery : Lottery) (ticketsBought : ℕ) : ℝ :=
  (1 - lottery.winningProbability) ^ ticketsBought

/-- Theorem stating that buying 100 tickets in the given lottery may not result in a win -/
theorem buying_100_tickets_may_not_win (lottery : Lottery)
  (h1 : lottery.totalTickets = 100000)
  (h2 : lottery.winningProbability = 0.01) :
  probNotWinning lottery 100 > 0 := by
  sorry

#check buying_100_tickets_may_not_win

end NUMINAMATH_CALUDE_buying_100_tickets_may_not_win_l3828_382839


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l3828_382811

theorem smaller_circle_radius (R : ℝ) (r : ℝ) :
  R = 10 → -- Radius of the larger circle is 10 meters
  (4 * (2 * r) = 2 * R) → -- Four diameters of smaller circles span the diameter of the larger circle
  r = 2.5 := by sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l3828_382811


namespace NUMINAMATH_CALUDE_geese_in_marsh_l3828_382805

theorem geese_in_marsh (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_in_marsh_l3828_382805


namespace NUMINAMATH_CALUDE_solve_for_q_l3828_382814

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/60)
  (h2 : 5/6 = (m+n)/90)
  (h3 : 5/6 = (q-m)/150) : q = 150 := by sorry

end NUMINAMATH_CALUDE_solve_for_q_l3828_382814


namespace NUMINAMATH_CALUDE_max_area_of_three_rectangles_l3828_382892

/-- Given two rectangles with dimensions 9x12 and 10x15, 
    prove that the maximum area of a rectangle that can be formed 
    by arranging these two rectangles along with a third rectangle is 330. -/
theorem max_area_of_three_rectangles : 
  let rect1_width : ℝ := 9
  let rect1_height : ℝ := 12
  let rect2_width : ℝ := 10
  let rect2_height : ℝ := 15
  ∃ (rect3_width rect3_height : ℝ),
    (max 
      (max rect1_width rect2_width * (rect1_height + rect2_height))
      (max rect1_height rect2_height * (rect1_width + rect2_width))
    ) = 330 := by
  sorry

end NUMINAMATH_CALUDE_max_area_of_three_rectangles_l3828_382892


namespace NUMINAMATH_CALUDE_other_focus_coordinates_l3828_382885

/-- A hyperbola with given axes of symmetry and one focus on the y-axis -/
structure Hyperbola where
  x_axis : ℝ
  y_axis : ℝ
  focus_on_y_axis : ℝ × ℝ

/-- The other focus of the hyperbola -/
def other_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Theorem stating that the other focus has coordinates (-2, 2) -/
theorem other_focus_coordinates (h : Hyperbola) 
  (hx : h.x_axis = -1)
  (hy : h.y_axis = 2)
  (hf : h.focus_on_y_axis.1 = 0 ∧ h.focus_on_y_axis.2 = 2) :
  other_focus h = (-2, 2) := by sorry

end NUMINAMATH_CALUDE_other_focus_coordinates_l3828_382885


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3828_382862

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3828_382862


namespace NUMINAMATH_CALUDE_books_combination_l3828_382868

/- Given conditions -/
def totalBooks : ℕ := 13
def booksToSelect : ℕ := 3

/- Theorem to prove -/
theorem books_combination : Nat.choose totalBooks booksToSelect = 286 := by
  sorry

end NUMINAMATH_CALUDE_books_combination_l3828_382868


namespace NUMINAMATH_CALUDE_school_boys_count_l3828_382836

theorem school_boys_count (girls : ℕ) (difference : ℕ) (boys : ℕ) : 
  girls = 635 → difference = 510 → boys = girls + difference → boys = 1145 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l3828_382836


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l3828_382896

/-- Proves that the cost of an orchestra seat is $12 given the conditions of the theater ticket sales --/
theorem theater_ticket_cost (balcony_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (balcony_excess : ℕ) :
  balcony_cost = 8 →
  total_tickets = 350 →
  total_revenue = 3320 →
  balcony_excess = 90 →
  ∃ (orchestra_cost : ℕ), 
    orchestra_cost = 12 ∧
    (total_tickets - balcony_excess) / 2 * orchestra_cost + 
    (total_tickets + balcony_excess) / 2 * balcony_cost = total_revenue :=
by
  sorry

#check theater_ticket_cost

end NUMINAMATH_CALUDE_theater_ticket_cost_l3828_382896


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3828_382849

/-- A rhombus with given perimeter and one diagonal -/
structure Rhombus where
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem: In a rhombus with perimeter 52 and one diagonal 10, the other diagonal is 24 -/
theorem rhombus_diagonal (r : Rhombus) (h1 : r.perimeter = 52) (h2 : r.diagonal2 = 10) :
  ∃ (diagonal1 : ℝ), diagonal1 = 24 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_diagonal_l3828_382849


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3828_382897

def total_cars : ℕ := 300

def audi_percent : ℚ := 10 / 100
def toyota_percent : ℚ := 15 / 100
def acura_percent : ℚ := 20 / 100
def honda_percent : ℚ := 18 / 100

def other_brands_percent : ℚ := audi_percent + toyota_percent + acura_percent + honda_percent

def bmw_percent : ℚ := 1 - other_brands_percent

theorem bmw_sales_count : ⌊(bmw_percent * total_cars : ℚ)⌋ = 111 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3828_382897


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3828_382888

theorem larger_integer_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 198) : 
  x.val = 18 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3828_382888


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3828_382829

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^15 + 11^21) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^15 + 11^21) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3828_382829


namespace NUMINAMATH_CALUDE_soccer_players_count_l3828_382826

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) : 
  total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l3828_382826


namespace NUMINAMATH_CALUDE_binomial_26_6_l3828_382858

theorem binomial_26_6 (h1 : Nat.choose 23 5 = 33649) 
                       (h2 : Nat.choose 23 6 = 33649)
                       (h3 : Nat.choose 25 5 = 53130) : 
  Nat.choose 26 6 = 163032 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l3828_382858


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l3828_382803

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ (x - 1)^2 ≥ 9) →
  a < -4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l3828_382803


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l3828_382809

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (total_jump : ℕ) 
  (h1 : frog_jump = 35)
  (h2 : total_jump = 66) :
  total_jump - frog_jump = 31 := by
sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l3828_382809


namespace NUMINAMATH_CALUDE_first_week_daily_rate_l3828_382891

def daily_rate_first_week (x : ℚ) : Prop :=
  ∃ (total_cost : ℚ),
    total_cost = 7 * x + 16 * 11 ∧
    total_cost = 302

theorem first_week_daily_rate :
  ∀ x : ℚ, daily_rate_first_week x → x = 18 :=
by sorry

end NUMINAMATH_CALUDE_first_week_daily_rate_l3828_382891


namespace NUMINAMATH_CALUDE_water_depth_difference_l3828_382890

theorem water_depth_difference (dean_height : ℝ) (water_depth_factor : ℝ) : 
  dean_height = 9 →
  water_depth_factor = 10 →
  water_depth_factor * dean_height - dean_height = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_water_depth_difference_l3828_382890


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3828_382801

/-- The speed of a boat in still water, given its speed with and against the stream -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ)
  (h1 : along_stream = 9)
  (h2 : against_stream = 5) :
  (along_stream + against_stream) / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3828_382801


namespace NUMINAMATH_CALUDE_distance_at_speed1_proof_l3828_382863

-- Define the total distance
def total_distance : ℝ := 250

-- Define the two speeds
def speed1 : ℝ := 40
def speed2 : ℝ := 60

-- Define the total time
def total_time : ℝ := 5.2

-- Define the distance covered at speed1 (40 kmph)
def distance_at_speed1 : ℝ := 124

-- Theorem statement
theorem distance_at_speed1_proof :
  let distance_at_speed2 := total_distance - distance_at_speed1
  (distance_at_speed1 / speed1) + (distance_at_speed2 / speed2) = total_time :=
by sorry

end NUMINAMATH_CALUDE_distance_at_speed1_proof_l3828_382863


namespace NUMINAMATH_CALUDE_probability_two_slate_rocks_l3828_382864

/-- The probability of selecting two slate rocks from a field with given rock counts -/
theorem probability_two_slate_rocks (slate_count pumice_count granite_count : ℕ) :
  slate_count = 12 →
  pumice_count = 16 →
  granite_count = 8 →
  let total_count := slate_count + pumice_count + granite_count
  (slate_count : ℚ) / total_count * ((slate_count - 1) : ℚ) / (total_count - 1) = 11 / 105 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_slate_rocks_l3828_382864


namespace NUMINAMATH_CALUDE_trivia_team_groups_l3828_382886

theorem trivia_team_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 58)
  (h2 : not_picked = 10)
  (h3 : students_per_group = 6) :
  (total_students - not_picked) / students_per_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l3828_382886


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l3828_382835

theorem sqrt_expression_equals_sqrt_three : 
  Real.sqrt 48 - 6 * Real.sqrt (1/3) - Real.sqrt 18 / Real.sqrt 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l3828_382835


namespace NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l3828_382872

theorem limit_sequence_equals_one_over_e :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((2*n - 1) / (2*n + 1))^(n + 1) - 1/Real.exp 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l3828_382872


namespace NUMINAMATH_CALUDE_shirt_cost_l3828_382856

def flat_rate_shipping : ℝ := 5
def shipping_rate : ℝ := 0.2
def shipping_threshold : ℝ := 50
def socks_price : ℝ := 5
def shorts_price : ℝ := 15
def swim_trunks_price : ℝ := 14
def total_bill : ℝ := 102
def shorts_quantity : ℕ := 2

theorem shirt_cost (shirt_price : ℝ) : 
  (shirt_price + socks_price + shorts_quantity * shorts_price + swim_trunks_price > shipping_threshold) →
  (shirt_price + socks_price + shorts_quantity * shorts_price + swim_trunks_price) * 
    (1 + shipping_rate) = total_bill →
  shirt_price = 36 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l3828_382856


namespace NUMINAMATH_CALUDE_positive_X_value_l3828_382875

-- Define the # operation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- State the theorem
theorem positive_X_value :
  ∃ X : ℝ, X > 0 ∧ hash X 7 = 85 ∧ X = 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_X_value_l3828_382875


namespace NUMINAMATH_CALUDE_probability_red_or_white_is_five_sixths_l3828_382847

def total_marbles : ℕ := 30
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9

def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

def probability_red_or_white : ℚ :=
  (red_marbles + white_marbles : ℚ) / total_marbles

theorem probability_red_or_white_is_five_sixths :
  probability_red_or_white = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_is_five_sixths_l3828_382847


namespace NUMINAMATH_CALUDE_evening_campers_count_l3828_382820

def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def total_campers : ℕ := 98

theorem evening_campers_count : 
  total_campers - (morning_campers + afternoon_campers) = 49 := by
  sorry

end NUMINAMATH_CALUDE_evening_campers_count_l3828_382820


namespace NUMINAMATH_CALUDE_eight_routes_A_to_B_l3828_382842

/-- The number of different routes from A to B, given that all routes must pass through C -/
def routes_A_to_B (roads_A_to_C roads_C_to_B : ℕ) : ℕ :=
  roads_A_to_C * roads_C_to_B

/-- Theorem stating that there are 8 different routes from A to B -/
theorem eight_routes_A_to_B :
  routes_A_to_B 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_routes_A_to_B_l3828_382842


namespace NUMINAMATH_CALUDE_solution_value_l3828_382854

theorem solution_value (x a : ℝ) (h : x = 3 ∧ 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3828_382854


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l3828_382887

theorem cos_pi_4_plus_alpha (α : Real) 
  (h : Real.sin (α - π/4) = 1/3) : 
  Real.cos (π/4 + α) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l3828_382887


namespace NUMINAMATH_CALUDE_intersection_complement_equals_half_open_interval_l3828_382819

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_complement_equals_half_open_interval :
  M ∩ (Set.compl N) = Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_half_open_interval_l3828_382819


namespace NUMINAMATH_CALUDE_system_three_solutions_l3828_382884

/-- The system of equations has exactly three solutions if and only if a = 9 or a = 23 + 4√15 -/
theorem system_three_solutions (a : ℝ) :
  (∃! x y z : ℝ × ℝ, 
    ((abs (y.2 + 9) + abs (x.1 + 2) - 2) * (x.1^2 + x.2^2 - 3) = 0 ∧
     (x.1 + 2)^2 + (x.2 + 4)^2 = a) ∧
    ((abs (y.2 + 9) + abs (y.1 + 2) - 2) * (y.1^2 + y.2^2 - 3) = 0 ∧
     (y.1 + 2)^2 + (y.2 + 4)^2 = a) ∧
    ((abs (z.2 + 9) + abs (z.1 + 2) - 2) * (z.1^2 + z.2^2 - 3) = 0 ∧
     (z.1 + 2)^2 + (z.2 + 4)^2 = a) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔
  (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_system_three_solutions_l3828_382884


namespace NUMINAMATH_CALUDE_simon_kabob_cost_l3828_382889

/-- Represents the cost of making kabob sticks -/
def cost_of_kabobs (cubes_per_stick : ℕ) (cubes_per_slab : ℕ) (cost_per_slab : ℕ) (num_sticks : ℕ) : ℕ :=
  let slabs_needed := (num_sticks * cubes_per_stick + cubes_per_slab - 1) / cubes_per_slab
  slabs_needed * cost_per_slab

/-- Proves that the cost for Simon to make 40 kabob sticks is $50 -/
theorem simon_kabob_cost : cost_of_kabobs 4 80 25 40 = 50 := by
  sorry

end NUMINAMATH_CALUDE_simon_kabob_cost_l3828_382889


namespace NUMINAMATH_CALUDE_point_above_line_l3828_382818

/-- A point (x, y) is above a line ax + by + c = 0 if by < -ax - c -/
def IsAboveLine (x y a b c : ℝ) : Prop := b * y < -a * x - c

/-- The range of t for which (-2, t) is above the line 2x - 3y + 6 = 0 -/
theorem point_above_line (t : ℝ) : 
  IsAboveLine (-2) t 2 (-3) 6 → t > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_point_above_line_l3828_382818


namespace NUMINAMATH_CALUDE_product_of_parts_l3828_382876

theorem product_of_parts (z : ℂ) : z = 1 - I → (z.re * z.im = -1) := by
  sorry

end NUMINAMATH_CALUDE_product_of_parts_l3828_382876


namespace NUMINAMATH_CALUDE_central_high_teachers_central_high_teachers_count_l3828_382867

/-- Calculates the number of teachers required at Central High School -/
theorem central_high_teachers (total_students : ℕ) (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  let total_class_occurrences := total_students * classes_per_student
  let unique_classes := total_class_occurrences / students_per_class
  let required_teachers := unique_classes / classes_per_teacher
  required_teachers

/-- Proves that the number of teachers required at Central High School is 120 -/
theorem central_high_teachers_count : 
  central_high_teachers 1500 6 3 25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_central_high_teachers_central_high_teachers_count_l3828_382867


namespace NUMINAMATH_CALUDE_fruit_store_inventory_l3828_382895

/-- Represents the fruit store inventory and gift basket composition. -/
structure FruitStore where
  cantaloupes : ℕ
  dragonFruits : ℕ
  kiwis : ℕ
  basketCantaloupes : ℕ
  basketDragonFruits : ℕ
  basketKiwis : ℕ

/-- Theorem stating the original number of dragon fruits and remaining kiwis. -/
theorem fruit_store_inventory (store : FruitStore)
  (h1 : store.basketCantaloupes = 2)
  (h2 : store.basketDragonFruits = 4)
  (h3 : store.basketKiwis = 10)
  (h4 : store.dragonFruits = 3 * store.cantaloupes + 10)
  (h5 : store.kiwis = 2 * store.dragonFruits)
  (h6 : store.dragonFruits - store.basketDragonFruits * store.cantaloupes = 130) :
  store.dragonFruits = 370 ∧ 
  store.kiwis - store.basketKiwis * store.cantaloupes = 140 := by
  sorry


end NUMINAMATH_CALUDE_fruit_store_inventory_l3828_382895


namespace NUMINAMATH_CALUDE_adam_apples_proof_l3828_382800

def monday_apples : ℕ := 15
def tuesday_multiplier : ℕ := 3
def wednesday_multiplier : ℕ := 4

def total_apples : ℕ := 
  monday_apples + 
  (tuesday_multiplier * monday_apples) + 
  (wednesday_multiplier * tuesday_multiplier * monday_apples)

theorem adam_apples_proof : total_apples = 240 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_proof_l3828_382800


namespace NUMINAMATH_CALUDE_class_average_weight_l3828_382830

theorem class_average_weight (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ) :
  students_a = 50 →
  students_b = 50 →
  avg_weight_a = 60 →
  avg_weight_b = 80 →
  (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = 70 :=
by sorry

end NUMINAMATH_CALUDE_class_average_weight_l3828_382830


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l3828_382838

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  -- The unique solution is x = 8/3
  use 8/3
  constructor
  · -- Prove that 8/3 satisfies the equation
    sorry
  · -- Prove that any solution must equal 8/3
    sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l3828_382838


namespace NUMINAMATH_CALUDE_quentavious_gum_pieces_l3828_382874

/-- Calculates the number of gum pieces received in an exchange. -/
def gum_pieces_received (initial_nickels : ℕ) (gum_per_nickel : ℕ) (remaining_nickels : ℕ) : ℕ :=
  (initial_nickels - remaining_nickels) * gum_per_nickel

/-- Proves that Quentavious received 6 pieces of gum. -/
theorem quentavious_gum_pieces :
  gum_pieces_received 5 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quentavious_gum_pieces_l3828_382874


namespace NUMINAMATH_CALUDE_athlete_c_most_suitable_l3828_382871

/-- Represents an athlete with their mean jump distance and variance --/
structure Athlete where
  name : String
  mean : ℝ
  variance : ℝ

/-- Determines if one athlete is more suitable than another --/
def moreSuitable (a b : Athlete) : Prop :=
  (a.mean > b.mean) ∨ (a.mean = b.mean ∧ a.variance < b.variance)

/-- Determines if an athlete is the most suitable among a list of athletes --/
def mostSuitable (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, a ≠ b → moreSuitable a b

theorem athlete_c_most_suitable :
  let athletes := [
    Athlete.mk "A" 380 12.5,
    Athlete.mk "B" 360 13.5,
    Athlete.mk "C" 380 2.4,
    Athlete.mk "D" 350 2.7
  ]
  let c := Athlete.mk "C" 380 2.4
  mostSuitable c athletes := by
  sorry

end NUMINAMATH_CALUDE_athlete_c_most_suitable_l3828_382871


namespace NUMINAMATH_CALUDE_carpet_fits_rooms_l3828_382878

/-- Represents a rectangular room --/
structure Room where
  width : ℕ
  length : ℕ

/-- Represents a rectangular carpet --/
structure Carpet where
  width : ℕ
  length : ℕ

/-- Checks if a carpet fits perfectly in a room --/
def fitsPerectly (c : Carpet) (r : Room) : Prop :=
  c.width ^ 2 + c.length ^ 2 = r.width ^ 2 + r.length ^ 2

theorem carpet_fits_rooms :
  ∃ (c : Carpet) (r1 r2 : Room),
    c.width = 25 ∧
    c.length = 50 ∧
    r1.width = 38 ∧
    r2.width = 50 ∧
    r1.length = r2.length ∧
    fitsPerectly c r1 ∧
    fitsPerectly c r2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_fits_rooms_l3828_382878


namespace NUMINAMATH_CALUDE_period_length_l3828_382840

theorem period_length 
  (total_duration : ℕ) 
  (num_periods : ℕ) 
  (break_duration : ℕ) 
  (num_breaks : ℕ) :
  total_duration = 220 →
  num_periods = 5 →
  break_duration = 5 →
  num_breaks = 4 →
  (total_duration - num_breaks * break_duration) / num_periods = 40 :=
by sorry

end NUMINAMATH_CALUDE_period_length_l3828_382840


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3828_382893

/-- Given two vectors a and b in ℝ², prove that if |a| = 1, |b| = 2, and a + b = (2√2, 1), then |3a + b| = 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  norm a = 1 →
  norm b = 2 →
  a + b = (2 * Real.sqrt 2, 1) →
  norm (3 • a + b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3828_382893


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3828_382823

theorem simplify_polynomial (x : ℝ) :
  2 - 4*x - 6*x^2 + 8 + 10*x - 12*x^2 - 14 + 16*x + 18*x^2 = 22*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3828_382823


namespace NUMINAMATH_CALUDE_money_distribution_l3828_382827

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (AC_sum : A + C = 300)
  (C_amount : C = 50) : 
  B + C = 150 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3828_382827


namespace NUMINAMATH_CALUDE_expression_evaluation_l3828_382882

theorem expression_evaluation (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (2*x + y)^2 + (x + y)*(x - y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3828_382882


namespace NUMINAMATH_CALUDE_fraction_equality_l3828_382851

theorem fraction_equality (p q : ℝ) (h : (p⁻¹ + q⁻¹) / (p⁻¹ - q⁻¹) = 1009) :
  (p + q) / (p - q) = -1009 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3828_382851


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3828_382816

/-- Given a system of linear equations with a specific k value, 
    prove that xz/y^2 equals a specific constant --/
theorem system_solution_ratio (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (eq1 : x + (16/5)*y + 4*z = 0)
  (eq2 : 3*x + (16/5)*y + z = 0)
  (eq3 : 2*x + 4*y + 3*z = 0) :
  ∃ (c : ℝ), x*z/y^2 = c :=
by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3828_382816


namespace NUMINAMATH_CALUDE_double_negation_2023_l3828_382899

theorem double_negation_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_double_negation_2023_l3828_382899


namespace NUMINAMATH_CALUDE_garden_yield_calculation_l3828_382812

/-- Represents the dimensions of a garden section in steps -/
structure GardenSection where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield for an L-shaped garden -/
def expected_potato_yield (section1 : GardenSection) (section2 : GardenSection) 
    (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let area1 := (section1.length * section1.width * step_length ^ 2 : ℝ)
  let area2 := (section2.length * section2.width * step_length ^ 2 : ℝ)
  (area1 + area2) * yield_per_sqft

/-- Theorem stating the expected potato yield for the given garden -/
theorem garden_yield_calculation :
  let section1 : GardenSection := { length := 10, width := 25 }
  let section2 : GardenSection := { length := 10, width := 10 }
  let step_length : ℝ := 1.5
  let yield_per_sqft : ℝ := 0.75
  expected_potato_yield section1 section2 step_length yield_per_sqft = 590.625 := by
  sorry

end NUMINAMATH_CALUDE_garden_yield_calculation_l3828_382812


namespace NUMINAMATH_CALUDE_at_least_one_equal_to_a_l3828_382846

theorem at_least_one_equal_to_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end NUMINAMATH_CALUDE_at_least_one_equal_to_a_l3828_382846


namespace NUMINAMATH_CALUDE_derivative_at_one_l3828_382824

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) :
  f' 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3828_382824


namespace NUMINAMATH_CALUDE_black_marble_probability_l3828_382815

theorem black_marble_probability :
  let yellow : ℕ := 24
  let blue : ℕ := 18
  let green : ℕ := 12
  let red : ℕ := 8
  let white : ℕ := 7
  let black : ℕ := 3
  let purple : ℕ := 2
  let total : ℕ := yellow + blue + green + red + white + black + purple
  (black : ℚ) / total = 3 / 74 := by sorry

end NUMINAMATH_CALUDE_black_marble_probability_l3828_382815


namespace NUMINAMATH_CALUDE_paige_mp3_songs_l3828_382852

/-- Calculates the final number of songs on an mp3 player after deleting and adding songs. -/
def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

/-- Theorem: The final number of songs on Paige's mp3 player is 10. -/
theorem paige_mp3_songs : final_song_count 11 9 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paige_mp3_songs_l3828_382852


namespace NUMINAMATH_CALUDE_modulus_of_complex_l3828_382861

theorem modulus_of_complex (m : ℝ) : 
  let z : ℂ := Complex.mk (m - 2) (m + 1)
  Complex.abs z = Real.sqrt (2 * m^2 - 2 * m + 5) := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l3828_382861


namespace NUMINAMATH_CALUDE_flooring_rate_calculation_l3828_382855

/-- Given a rectangular room with specified dimensions and total flooring cost,
    calculate the rate per square meter. -/
theorem flooring_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 10)
    (h_width : width = 4.75)
    (h_total_cost : total_cost = 42750) : 
    total_cost / (length * width) = 900 := by
  sorry

#check flooring_rate_calculation

end NUMINAMATH_CALUDE_flooring_rate_calculation_l3828_382855


namespace NUMINAMATH_CALUDE_discarded_fruit_percentages_l3828_382870

/-- Represents the percentages of fruit sold and discarded over two days -/
structure FruitPercentages where
  pear_sold_day1 : ℝ
  pear_discarded_day1 : ℝ
  pear_sold_day2 : ℝ
  pear_discarded_day2 : ℝ
  apple_sold_day1 : ℝ
  apple_discarded_day1 : ℝ
  apple_sold_day2 : ℝ
  apple_discarded_day2 : ℝ
  orange_sold_day1 : ℝ
  orange_discarded_day1 : ℝ
  orange_sold_day2 : ℝ
  orange_discarded_day2 : ℝ

/-- Calculates the total percentage of fruit discarded over two days -/
def totalDiscardedPercentage (fp : FruitPercentages) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the correct percentages of discarded fruit -/
theorem discarded_fruit_percentages (fp : FruitPercentages) 
  (h1 : fp.pear_sold_day1 = 20)
  (h2 : fp.pear_discarded_day1 = 30)
  (h3 : fp.pear_sold_day2 = 10)
  (h4 : fp.pear_discarded_day2 = 20)
  (h5 : fp.apple_sold_day1 = 25)
  (h6 : fp.apple_discarded_day1 = 15)
  (h7 : fp.apple_sold_day2 = 15)
  (h8 : fp.apple_discarded_day2 = 10)
  (h9 : fp.orange_sold_day1 = 30)
  (h10 : fp.orange_discarded_day1 = 35)
  (h11 : fp.orange_sold_day2 = 20)
  (h12 : fp.orange_discarded_day2 = 30) :
  totalDiscardedPercentage fp = (34.08, 16.66875, 35.42) := by
  sorry

end NUMINAMATH_CALUDE_discarded_fruit_percentages_l3828_382870


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3828_382832

/-- Proves that given a journey of 448 km completed in 20 hours, where the first half is traveled at 21 km/hr, the speed for the second half must be 24 km/hr. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  first_half_speed = 21 →
  (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time →
  second_half_speed = 24 := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_journey_speed_calculation_l3828_382832


namespace NUMINAMATH_CALUDE_kids_total_savings_l3828_382841

-- Define the conversion rate
def pound_to_dollar : ℝ := 1.38

-- Define the savings for each child
def teagan_savings : ℝ := 200 * 0.01 + 15 * 1.00
def rex_savings : ℝ := 100 * 0.05 + 45 * 0.25 + 8 * pound_to_dollar
def toni_savings : ℝ := 330 * 0.10 + 12 * 5.00

-- Define the total savings
def total_savings : ℝ := teagan_savings + rex_savings + toni_savings

-- Theorem statement
theorem kids_total_savings : total_savings = 137.29 := by
  sorry

end NUMINAMATH_CALUDE_kids_total_savings_l3828_382841


namespace NUMINAMATH_CALUDE_shoe_pairs_count_l3828_382865

theorem shoe_pairs_count (total_shoes : ℕ) (prob_same_color : ℚ) : 
  total_shoes = 14 →
  prob_same_color = 1 / 13 →
  (∃ (n : ℕ), n * 2 = total_shoes ∧ 
    prob_same_color = 1 / (2 * n - 1)) →
  ∃ (pairs : ℕ), pairs = 7 ∧ pairs * 2 = total_shoes :=
by sorry

end NUMINAMATH_CALUDE_shoe_pairs_count_l3828_382865


namespace NUMINAMATH_CALUDE_complex_number_properties_l3828_382825

theorem complex_number_properties (z : ℂ) (h : z * (1 + Complex.I) = 2) :
  (Complex.abs z = Real.sqrt 2) ∧
  (∀ p : ℝ, z^2 - p*z + 2 = 0 → p = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3828_382825


namespace NUMINAMATH_CALUDE_bobs_small_gate_width_l3828_382881

/-- Represents a rectangular garden with gates and fencing -/
structure Garden where
  length : ℝ
  width : ℝ
  large_gate_width : ℝ
  total_fencing : ℝ

/-- Calculates the perimeter of a rectangle -/
def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

/-- Calculates the width of the small gate -/
def small_gate_width (g : Garden) : ℝ :=
  g.total_fencing - rectangle_perimeter g.length g.width + g.large_gate_width

/-- Theorem stating the width of the small gate in Bob's garden -/
theorem bobs_small_gate_width :
  let g : Garden := {
    length := 225,
    width := 125,
    large_gate_width := 10,
    total_fencing := 687
  }
  small_gate_width g = 3 := by
  sorry


end NUMINAMATH_CALUDE_bobs_small_gate_width_l3828_382881


namespace NUMINAMATH_CALUDE_relationship_between_a_b_l3828_382866

theorem relationship_between_a_b (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) :
  a < -b ∧ -b < b ∧ b < -a := by sorry

end NUMINAMATH_CALUDE_relationship_between_a_b_l3828_382866


namespace NUMINAMATH_CALUDE_probability_10_heads_in_12_flips_l3828_382859

/-- The probability of getting exactly 10 heads in 12 flips of a fair coin -/
theorem probability_10_heads_in_12_flips : 
  (Nat.choose 12 10 : ℚ) / 2^12 = 66 / 4096 := by sorry

end NUMINAMATH_CALUDE_probability_10_heads_in_12_flips_l3828_382859


namespace NUMINAMATH_CALUDE_complementary_angles_equal_l3828_382831

/-- Two angles that are complementary to the same angle are equal. -/
theorem complementary_angles_equal (α β γ : Real) (h1 : α + γ = 90) (h2 : β + γ = 90) : α = β := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_equal_l3828_382831


namespace NUMINAMATH_CALUDE_A_power_100_eq_A_l3828_382828

/-- The matrix A -/
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]]

/-- Theorem stating that A^100 = A -/
theorem A_power_100_eq_A : A ^ 100 = A := by sorry

end NUMINAMATH_CALUDE_A_power_100_eq_A_l3828_382828


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l3828_382813

/-- The x-intercept of the line 2x + y - 2 = 0 is at x = 1 -/
theorem x_intercept_of_line (x y : ℝ) : 2*x + y - 2 = 0 → y = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l3828_382813


namespace NUMINAMATH_CALUDE_rain_probability_l3828_382821

theorem rain_probability (p_rain : ℝ) (p_consecutive : ℝ) 
  (h1 : p_rain = 1/3)
  (h2 : p_consecutive = 1/5) :
  p_consecutive / p_rain = 3/5 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_l3828_382821


namespace NUMINAMATH_CALUDE_min_coefficient_value_l3828_382834

theorem min_coefficient_value (a b c d : ℤ) :
  (∃ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40) →
  (∃ (min_box : ℤ), 
    (∃ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40 ∧ box ≥ min_box) ∧
    (∀ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40 → box ≥ min_box) ∧
    min_box = 89) :=
by sorry


end NUMINAMATH_CALUDE_min_coefficient_value_l3828_382834


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3828_382804

theorem simplify_trigonometric_expression (α : Real) (h : π < α ∧ α < 2*π) : 
  ((1 + Real.sin α + Real.cos α) * (Real.sin (α/2) - Real.cos (α/2))) / 
  Real.sqrt (2 + 2 * Real.cos α) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l3828_382804


namespace NUMINAMATH_CALUDE_z_properties_l3828_382850

/-- Complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 2)

/-- Condition for z to be purely imaginary -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Condition for z to lie on the line x + 2y + 1 = 0 -/
def on_line (z : ℂ) : Prop := z.re + 2 * z.im + 1 = 0

theorem z_properties (a : ℝ) :
  (is_purely_imaginary (z a) → a = 2) ∧
  (on_line (z a) → a = -1) := by sorry

end NUMINAMATH_CALUDE_z_properties_l3828_382850


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3828_382857

theorem largest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = 105 →      -- Sum of two angles is 7/6 of a right angle (90° * 7/6 = 105°)
  β = α + 20 →       -- One angle is 20° larger than the other
  max α (max β γ) = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3828_382857


namespace NUMINAMATH_CALUDE_matrix_power_four_l3828_382817

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_four : 
  A ^ 4 = !![(-4 : ℝ), 0; 0, -4] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3828_382817


namespace NUMINAMATH_CALUDE_smallest_portion_is_ten_l3828_382843

/-- Represents the distribution of bread loaves -/
structure BreadDistribution where
  a : ℕ  -- smallest portion (first term of arithmetic sequence)
  d : ℕ  -- common difference of arithmetic sequence

/-- The problem of distributing bread loaves -/
def breadProblem (bd : BreadDistribution) : Prop :=
  -- Total sum is 100
  (5 * bd.a + 10 * bd.d = 100) ∧
  -- Sum of larger three portions is 1/3 of sum of smaller two portions
  (3 * bd.a + 9 * bd.d = (2 * bd.a + bd.d) / 3)

/-- Theorem stating the smallest portion is 10 -/
theorem smallest_portion_is_ten :
  ∃ (bd : BreadDistribution), breadProblem bd ∧ bd.a = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_portion_is_ten_l3828_382843


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3828_382883

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1 / 4 : ℚ) + (n : ℚ) / 5 < 7 / 4 ↔ n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3828_382883


namespace NUMINAMATH_CALUDE_each_score_is_individual_l3828_382807

/-- Represents a student in the study -/
structure Student where
  id : Nat
  score : ℝ

/-- Represents the statistical study -/
structure CivilizationKnowledgeStudy where
  population : Finset Student
  sample : Finset Student
  pop_size : Nat
  sample_size : Nat

/-- Properties of the study -/
def valid_study (study : CivilizationKnowledgeStudy) : Prop :=
  study.pop_size = 1200 ∧
  study.sample_size = 100 ∧
  study.sample ⊆ study.population ∧
  study.population.card = study.pop_size ∧
  study.sample.card = study.sample_size

/-- Theorem stating that each student's score is an individual observation -/
theorem each_score_is_individual (study : CivilizationKnowledgeStudy) 
  (h : valid_study study) : 
  ∀ s ∈ study.population, ∃! x : ℝ, x = s.score :=
sorry

end NUMINAMATH_CALUDE_each_score_is_individual_l3828_382807


namespace NUMINAMATH_CALUDE_m_range_l3828_382853

theorem m_range (m : ℝ) 
  (h1 : |m + 1| ≤ 2)
  (h2 : ¬(¬p))
  (h3 : ¬(p ∧ q))
  (p : Prop)
  (q : Prop) :
  -2 < m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3828_382853


namespace NUMINAMATH_CALUDE_volleyball_practice_start_time_l3828_382860

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define addition of minutes to Time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

theorem volleyball_practice_start_time 
  (start_time : Time) 
  (homework_duration : Nat) 
  (break_duration : Nat) : 
  start_time = { hour := 13, minute := 59 } → 
  homework_duration = 96 → 
  break_duration = 25 → 
  addMinutes (addMinutes start_time homework_duration) break_duration = { hour := 16, minute := 0 } :=
by
  sorry


end NUMINAMATH_CALUDE_volleyball_practice_start_time_l3828_382860


namespace NUMINAMATH_CALUDE_alicia_tax_payment_l3828_382848

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def local_tax_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate / 100)

/-- Proves that Alicia's local tax payment is 50 cents per hour. -/
theorem alicia_tax_payment :
  local_tax_cents 25 2 = 50 := by
  sorry

#eval local_tax_cents 25 2

end NUMINAMATH_CALUDE_alicia_tax_payment_l3828_382848


namespace NUMINAMATH_CALUDE_farmland_area_l3828_382844

theorem farmland_area (lizzie_group_area other_group_area remaining_area : ℕ) 
  (h1 : lizzie_group_area = 250)
  (h2 : other_group_area = 265)
  (h3 : remaining_area = 385) :
  lizzie_group_area + other_group_area + remaining_area = 900 := by
  sorry

end NUMINAMATH_CALUDE_farmland_area_l3828_382844


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3828_382877

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  ((2 / (x - 1) - 1 / x) / ((x^2 - 1) / (x^2 - 2*x + 1))) = 1 / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3828_382877
