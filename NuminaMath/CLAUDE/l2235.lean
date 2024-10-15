import Mathlib

namespace NUMINAMATH_CALUDE_mod_seven_power_difference_l2235_223557

theorem mod_seven_power_difference : 47^2023 - 28^2023 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_power_difference_l2235_223557


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2235_223599

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (2 * y - 5)) = Real.sqrt 7 → y = 20.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2235_223599


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l2235_223549

theorem fraction_ratio_equality : ∃ x : ℚ, (3 / 7) / (6 / 5) = x / (2 / 5) ∧ x = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l2235_223549


namespace NUMINAMATH_CALUDE_series_sum_equals_one_six_hundredth_l2235_223504

/-- The sum of the series Σ(6n + 2) / ((6n - 1)^2 * (6n + 5)^2) from n=1 to infinity equals 1/600. -/
theorem series_sum_equals_one_six_hundredth :
  ∑' n : ℕ, (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_six_hundredth_l2235_223504


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2235_223558

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 720 →
  margin = 240 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2235_223558


namespace NUMINAMATH_CALUDE_find_coefficient_l2235_223514

/-- Given a polynomial equation and a sum condition, prove the value of a specific coefficient. -/
theorem find_coefficient (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (x - a) * (x + 2)^5 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + a₄*(x + 1)^4 + a₅*(x + 1)^5 + a₆*(x + 1)^6) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -96) →
  a₄ = -10 :=
by sorry

end NUMINAMATH_CALUDE_find_coefficient_l2235_223514


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l2235_223512

def num_white_socks : ℕ := 4
def num_brown_socks : ℕ := 4
def num_blue_socks : ℕ := 4
def num_red_socks : ℕ := 4

def total_socks : ℕ := num_white_socks + num_brown_socks + num_blue_socks + num_red_socks

theorem sock_pair_combinations :
  (num_red_socks * num_white_socks) + 
  (num_red_socks * num_brown_socks) + 
  (num_red_socks * num_blue_socks) = 48 :=
by sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l2235_223512


namespace NUMINAMATH_CALUDE_building_height_l2235_223576

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 65)
  : (flagpole_height / flagpole_shadow) * building_shadow = 26 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l2235_223576


namespace NUMINAMATH_CALUDE_flea_can_reach_all_points_l2235_223550

/-- The length of the k-th jump for the flea -/
def jumpLength (k : ℕ) : ℕ := 2^k + 1

/-- A jump is represented by its length and direction -/
structure Jump where
  length : ℕ
  direction : Bool  -- true for right, false for left

/-- The final position after a sequence of jumps -/
def finalPosition (jumps : List Jump) : ℤ :=
  jumps.foldl (fun pos jump => 
    if jump.direction then pos + jump.length else pos - jump.length) 0

/-- Theorem: For any natural number n, there exists a sequence of jumps
    that allows the flea to move from point 0 to point n -/
theorem flea_can_reach_all_points (n : ℕ) : 
  ∃ (jumps : List Jump), finalPosition jumps = n := by
  sorry

end NUMINAMATH_CALUDE_flea_can_reach_all_points_l2235_223550


namespace NUMINAMATH_CALUDE_lcm_problem_l2235_223571

theorem lcm_problem (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2235_223571


namespace NUMINAMATH_CALUDE_long_jump_competition_l2235_223569

/-- The long jump competition problem -/
theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second = first + 1 →
  third < second →
  fourth = third + 3 →
  fourth = 24 →
  second - third = 2 :=
by sorry

end NUMINAMATH_CALUDE_long_jump_competition_l2235_223569


namespace NUMINAMATH_CALUDE_smaller_cube_volume_l2235_223502

theorem smaller_cube_volume 
  (large_cube_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (surface_area_diff : ℝ) : ℝ :=
by
  have h1 : large_cube_volume = 343 := by sorry
  have h2 : num_small_cubes = 343 := by sorry
  have h3 : surface_area_diff = 1764 := by sorry
  
  -- Define the side length of the large cube
  let large_side : ℝ := large_cube_volume ^ (1/3)
  
  -- Define the surface area of the large cube
  let large_surface_area : ℝ := 6 * large_side^2
  
  -- Define the volume of each small cube
  let small_cube_volume : ℝ := large_cube_volume / num_small_cubes
  
  -- Define the side length of each small cube
  let small_side : ℝ := small_cube_volume ^ (1/3)
  
  -- Define the total surface area of all small cubes
  let total_small_surface_area : ℝ := 6 * small_side^2 * num_small_cubes
  
  -- The main theorem
  have : small_cube_volume = 1 := by sorry

  exact small_cube_volume

end NUMINAMATH_CALUDE_smaller_cube_volume_l2235_223502


namespace NUMINAMATH_CALUDE_smallest_prime_with_condition_l2235_223581

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem smallest_prime_with_condition : 
  ∃ (p : ℕ), 
    Prime p ∧ 
    is_two_digit p ∧ 
    tens_digit p = 3 ∧ 
    ¬(Prime (reverse_digits p + 5)) ∧
    ∀ (q : ℕ), Prime q ∧ is_two_digit q ∧ tens_digit q = 3 ∧ ¬(Prime (reverse_digits q + 5)) → p ≤ q ∧
    p = 31 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_with_condition_l2235_223581


namespace NUMINAMATH_CALUDE_jerry_piercing_pricing_l2235_223519

theorem jerry_piercing_pricing (nose_price : ℝ) (total_revenue : ℝ) (num_noses : ℕ) (num_ears : ℕ) :
  nose_price = 20 →
  total_revenue = 390 →
  num_noses = 6 →
  num_ears = 9 →
  let ear_price := (total_revenue - nose_price * num_noses) / num_ears
  let percentage_increase := (ear_price - nose_price) / nose_price * 100
  percentage_increase = 50 := by
sorry


end NUMINAMATH_CALUDE_jerry_piercing_pricing_l2235_223519


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l2235_223574

def euler_family_ages : List ℝ := [5, 5, 10, 15, 8, 12, 16]

theorem euler_family_mean_age :
  let ages := euler_family_ages
  let sum_ages := ages.sum
  let num_children := ages.length
  sum_ages / num_children = 10.14 := by
sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l2235_223574


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2235_223560

/-- An isosceles triangle with two interior angles summing to 100° has a vertex angle of either 20° or 80°. -/
theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  (a = b ∨ b = c ∨ a = c) →  -- Triangle is isosceles
  ((a + b = 100 ∧ c ≠ b) ∨ (b + c = 100 ∧ a ≠ b) ∨ (a + c = 100 ∧ a ≠ b)) →  -- Two angles sum to 100°
  (c = 20 ∨ c = 80) ∨ (a = 20 ∨ a = 80) ∨ (b = 20 ∨ b = 80) :=  -- Vertex angle is 20° or 80°
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2235_223560


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2235_223529

/-- The ellipse defined by 2x^2 + 3y^2 = 12 -/
def Ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 12

/-- The function to be maximized -/
def f (x y : ℝ) : ℝ := x + 2 * y

/-- Theorem stating that the maximum value of x + 2y on the given ellipse is √22 -/
theorem max_value_on_ellipse :
  ∃ (max : ℝ), max = Real.sqrt 22 ∧
  (∀ x y : ℝ, Ellipse x y → f x y ≤ max) ∧
  (∃ x y : ℝ, Ellipse x y ∧ f x y = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2235_223529


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2235_223566

theorem election_votes_calculation (total_votes : ℕ) : 
  (∃ (losing_candidate_votes winning_candidate_votes : ℕ),
    losing_candidate_votes = (32 * total_votes) / 100 ∧
    winning_candidate_votes = losing_candidate_votes + 1908 ∧
    winning_candidate_votes + losing_candidate_votes = total_votes) →
  total_votes = 5300 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2235_223566


namespace NUMINAMATH_CALUDE_expression_equals_one_l2235_223511

theorem expression_equals_one : 
  (150^2 - 12^2) / (90^2 - 18^2) * ((90 - 18)*(90 + 18)) / ((150 - 12)*(150 + 12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2235_223511


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2235_223578

theorem sum_of_two_numbers (a b : ℤ) : 
  (a = 2 * b - 43) → (min a b = 19) → (a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2235_223578


namespace NUMINAMATH_CALUDE_division_theorem_l2235_223591

-- Define the dividend polynomial
def dividend (x : ℝ) : ℝ := x^5 + 2*x^3 + x^2 + 3

-- Define the divisor polynomial
def divisor (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Define the quotient polynomial
def quotient (x : ℝ) : ℝ := x^3 + 4*x^2 + 12*x

-- Define the remainder polynomial
def remainder (x : ℝ) : ℝ := 25*x^2 - 72*x + 3

-- Theorem statement
theorem division_theorem :
  ∀ x : ℝ, dividend x = divisor x * quotient x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_division_theorem_l2235_223591


namespace NUMINAMATH_CALUDE_barrelCapacitiesSolution_l2235_223563

/-- Represents the capacities of three barrels --/
structure BarrelCapacities where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given capacities satisfy the problem conditions --/
def satisfiesConditions (c : BarrelCapacities) : Prop :=
  -- After first transfer, 1/4 remains in first barrel
  c.second = (3 * c.first) / 4 ∧
  -- After second transfer, 2/9 remains in second barrel
  c.third = (7 * c.first) / 12 ∧
  -- After third transfer, 50 more units needed to fill first barrel
  c.third + 50 = c.first

/-- The theorem to prove --/
theorem barrelCapacitiesSolution :
  ∃ (c : BarrelCapacities), satisfiesConditions c ∧ c.first = 120 ∧ c.second = 90 ∧ c.third = 70 := by
  sorry

end NUMINAMATH_CALUDE_barrelCapacitiesSolution_l2235_223563


namespace NUMINAMATH_CALUDE_sandbox_capacity_doubled_l2235_223593

/-- Represents the dimensions and capacity of a sandbox -/
structure Sandbox where
  length : ℝ
  width : ℝ
  height : ℝ
  capacity : ℝ

/-- Theorem: Doubling the dimensions of a sandbox increases its capacity by a factor of 8 -/
theorem sandbox_capacity_doubled (original : Sandbox) 
  (h_original_capacity : original.capacity = 10) :
  let new_sandbox := Sandbox.mk 
    (2 * original.length) 
    (2 * original.width) 
    (2 * original.height) 
    ((2 * original.length) * (2 * original.width) * (2 * original.height))
  new_sandbox.capacity = 80 := by
  sorry


end NUMINAMATH_CALUDE_sandbox_capacity_doubled_l2235_223593


namespace NUMINAMATH_CALUDE_age_problem_l2235_223597

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 47 →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2235_223597


namespace NUMINAMATH_CALUDE_seventh_term_is_twenty_l2235_223580

/-- An arithmetic sequence with first term 2 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- Theorem stating that the 7th term of the arithmetic sequence is 20 -/
theorem seventh_term_is_twenty : arithmeticSequence 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_twenty_l2235_223580


namespace NUMINAMATH_CALUDE_rene_received_300_l2235_223503

-- Define the amounts given to each person
def rene_amount : ℝ := sorry
def florence_amount : ℝ := sorry
def isha_amount : ℝ := sorry

-- Define the theorem
theorem rene_received_300 
  (h1 : florence_amount = 3 * rene_amount)
  (h2 : isha_amount = florence_amount / 2)
  (h3 : rene_amount + florence_amount + isha_amount = 1650)
  : rene_amount = 300 := by
  sorry

end NUMINAMATH_CALUDE_rene_received_300_l2235_223503


namespace NUMINAMATH_CALUDE_parabola_min_distance_l2235_223526

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The minimum distance from any point on the parabola to its focus is 1 -/
def min_distance_to_focus (para : Parabola) : Prop :=
  ∃ (x y : ℝ), y^2 = 2 * para.p * x ∧ 1 ≤ Real.sqrt ((x - para.p/2)^2 + y^2)

/-- If the minimum distance from any point on the parabola to the focus is 1, then p = 2 -/
theorem parabola_min_distance (para : Parabola) 
    (h_min : min_distance_to_focus para) : para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_min_distance_l2235_223526


namespace NUMINAMATH_CALUDE_total_distance_covered_l2235_223554

-- Define the given conditions
def cycling_time : ℚ := 30 / 60  -- 30 minutes in hours
def cycling_rate : ℚ := 12       -- 12 mph
def skating_time : ℚ := 45 / 60  -- 45 minutes in hours
def skating_rate : ℚ := 8        -- 8 mph
def total_time : ℚ := 75 / 60    -- 1 hour and 15 minutes in hours

-- State the theorem
theorem total_distance_covered : 
  cycling_time * cycling_rate + skating_time * skating_rate = 12 := by
  sorry -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_total_distance_covered_l2235_223554


namespace NUMINAMATH_CALUDE_fourth_student_id_l2235_223537

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  first_id : ℕ
  step : ℕ

/-- Creates a systematic sample given the total number of students and sample size -/
def create_systematic_sample (total : ℕ) (size : ℕ) : SystematicSample :=
  { total_students := total
  , sample_size := size
  , first_id := 3  -- Given in the problem
  , step := (total - 2) / size }

/-- Checks if a given ID is in the systematic sample -/
def is_in_sample (sample : SystematicSample) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = sample.first_id + k * sample.step ∧ k < sample.sample_size

/-- Main theorem: If 3, 29, and 42 are in the sample, then 16 is also in the sample -/
theorem fourth_student_id (sample : SystematicSample)
    (h_total : sample.total_students = 54)
    (h_size : sample.sample_size = 4)
    (h_3 : is_in_sample sample 3)
    (h_29 : is_in_sample sample 29)
    (h_42 : is_in_sample sample 42) :
    is_in_sample sample 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_id_l2235_223537


namespace NUMINAMATH_CALUDE_correct_number_of_fills_l2235_223588

/-- The number of times Alice must fill her measuring cup -/
def number_of_fills : ℕ := 12

/-- The amount of sugar Alice needs in cups -/
def sugar_needed : ℚ := 15/4

/-- The capacity of Alice's measuring cup in cups -/
def cup_capacity : ℚ := 1/3

/-- Theorem stating that the number of fills is correct -/
theorem correct_number_of_fills :
  (↑number_of_fills : ℚ) * cup_capacity ≥ sugar_needed ∧
  ((↑number_of_fills - 1 : ℚ) * cup_capacity < sugar_needed) :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_fills_l2235_223588


namespace NUMINAMATH_CALUDE_legos_set_cost_l2235_223507

theorem legos_set_cost (total_earnings : ℕ) (num_cars : ℕ) (car_price : ℕ) (legos_price : ℕ) :
  total_earnings = 45 →
  num_cars = 3 →
  car_price = 5 →
  total_earnings = num_cars * car_price + legos_price →
  legos_price = 30 := by
sorry

end NUMINAMATH_CALUDE_legos_set_cost_l2235_223507


namespace NUMINAMATH_CALUDE_special_rectangle_perimeter_l2235_223553

/-- A rectangle with integer dimensions where the area equals the perimeter minus 4 -/
structure SpecialRectangle where
  length : ℕ
  width : ℕ
  not_square : length ≠ width
  area_perimeter_relation : length * width = 2 * (length + width) - 4

/-- The perimeter of a SpecialRectangle is 26 -/
theorem special_rectangle_perimeter (r : SpecialRectangle) : 2 * (r.length + r.width) = 26 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_perimeter_l2235_223553


namespace NUMINAMATH_CALUDE_range_of_a_l2235_223585

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x > 0, (1 / a) - (1 / x) ≤ 2 * x) : 
  a ≥ Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2235_223585


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2235_223531

theorem quadratic_two_roots (a b c : ℝ) (h1 : b > a + c) (h2 : a + c > 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2235_223531


namespace NUMINAMATH_CALUDE_vector_BC_l2235_223584

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

theorem vector_BC : 
  let C : ℝ × ℝ := (A.1 + AC.1, A.2 + AC.2)
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by sorry

end NUMINAMATH_CALUDE_vector_BC_l2235_223584


namespace NUMINAMATH_CALUDE_truck_trailer_weights_l2235_223534

/-- Given the weights of trucks and trailers, prove their specific values -/
theorem truck_trailer_weights :
  ∀ (W_A W_B W_A' W_B' : ℝ),
    W_A + W_A' = 9000 →
    W_B + W_B' = 11000 →
    W_A' = 0.5 * W_A - 400 →
    W_B' = 0.4 * W_B + 500 →
    W_B = W_A + 2000 →
    W_A = 5500 ∧ W_B = 7500 ∧ W_A' = 2350 ∧ W_B' = 3500 := by
  sorry

end NUMINAMATH_CALUDE_truck_trailer_weights_l2235_223534


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l2235_223564

theorem apple_picking_ratio :
  ∀ (first_hour second_hour third_hour : ℕ),
    first_hour = 66 →
    second_hour = 2 * first_hour →
    first_hour + second_hour + third_hour = 220 →
    third_hour * 3 = first_hour :=
by
  sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l2235_223564


namespace NUMINAMATH_CALUDE_total_tiles_from_black_tiles_total_tiles_is_2601_l2235_223567

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  size : ℕ
  blackTilesCount : ℕ

/-- Theorem stating the relationship between the number of black tiles and total tiles -/
theorem total_tiles_from_black_tiles (floor : TiledFloor) 
  (h1 : floor.blackTilesCount = 101) : 
  floor.size * floor.size = 2601 := by
  sorry

/-- Main theorem proving the total number of tiles -/
theorem total_tiles_is_2601 (floor : TiledFloor) 
  (h1 : floor.blackTilesCount = 101) 
  (h2 : floor.blackTilesCount = 2 * floor.size - 1) : 
  floor.size * floor.size = 2601 := by
  sorry

end NUMINAMATH_CALUDE_total_tiles_from_black_tiles_total_tiles_is_2601_l2235_223567


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_of_14400_l2235_223583

/-- The number of perfect square factors of 14400 -/
def perfect_square_factors_of_14400 : ℕ :=
  let n := 14400
  let prime_factorization := (2, 4) :: (3, 2) :: (5, 2) :: []
  sorry

/-- Theorem stating that the number of perfect square factors of 14400 is 12 -/
theorem count_perfect_square_factors_of_14400 :
  perfect_square_factors_of_14400 = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_of_14400_l2235_223583


namespace NUMINAMATH_CALUDE_eggs_to_examine_l2235_223573

def number_of_trays : ℕ := 7
def eggs_per_tray : ℕ := 10
def percentage_to_examine : ℚ := 70 / 100

theorem eggs_to_examine :
  (number_of_trays * (eggs_per_tray * percentage_to_examine).floor) = 49 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_examine_l2235_223573


namespace NUMINAMATH_CALUDE_wade_friday_customers_l2235_223500

/-- Represents the number of customers Wade served on Friday -/
def F : ℕ := by sorry

/-- Wade's tip per customer in dollars -/
def tip_per_customer : ℚ := 2

/-- Total tips Wade made over the three days in dollars -/
def total_tips : ℚ := 296

theorem wade_friday_customers :
  F = 28 ∧
  tip_per_customer * (F + 3 * F + 36) = total_tips :=
by sorry

end NUMINAMATH_CALUDE_wade_friday_customers_l2235_223500


namespace NUMINAMATH_CALUDE_tiles_count_l2235_223532

/-- Represents a square floor tiled with white tiles on the perimeter and black tiles in the center -/
structure TiledSquare where
  side_length : ℕ
  white_tiles : ℕ
  black_tiles : ℕ

/-- The number of white tiles on the perimeter of a square floor -/
def perimeter_tiles (s : TiledSquare) : ℕ := 4 * (s.side_length - 1)

/-- The number of black tiles in the center of a square floor -/
def center_tiles (s : TiledSquare) : ℕ := (s.side_length - 2)^2

/-- Theorem stating that if there are 80 white tiles on the perimeter, there are 361 black tiles in the center -/
theorem tiles_count (s : TiledSquare) :
  perimeter_tiles s = 80 → center_tiles s = 361 := by
  sorry

end NUMINAMATH_CALUDE_tiles_count_l2235_223532


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2235_223596

theorem trigonometric_identity (α β γ x : ℝ) : 
  (Real.sin (x - β) * Real.sin (x - γ)) / (Real.sin (α - β) * Real.sin (α - γ)) +
  (Real.sin (x - γ) * Real.sin (x - α)) / (Real.sin (β - γ) * Real.sin (β - α)) +
  (Real.sin (x - α) * Real.sin (x - β)) / (Real.sin (γ - α) * Real.sin (γ - β)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2235_223596


namespace NUMINAMATH_CALUDE_alberts_to_marys_age_ratio_l2235_223542

theorem alberts_to_marys_age_ratio (albert_age mary_age betty_age : ℕ) : 
  betty_age = 4 → 
  albert_age = 4 * betty_age → 
  mary_age = albert_age - 8 → 
  (albert_age : ℚ) / mary_age = 2 := by
sorry

end NUMINAMATH_CALUDE_alberts_to_marys_age_ratio_l2235_223542


namespace NUMINAMATH_CALUDE_correct_average_l2235_223543

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 25 →
  correct_num = 55 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 19 := by
  sorry

#check correct_average

end NUMINAMATH_CALUDE_correct_average_l2235_223543


namespace NUMINAMATH_CALUDE_square_equation_solution_l2235_223508

theorem square_equation_solution :
  ∀ x : ℝ, x^2 = 16 ↔ x = -4 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2235_223508


namespace NUMINAMATH_CALUDE_a_b_reciprocals_l2235_223520

theorem a_b_reciprocals (a b : ℝ) : 
  a = 1 / (2 - Real.sqrt 3) → 
  b = 1 / (2 + Real.sqrt 3) → 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_a_b_reciprocals_l2235_223520


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l2235_223510

/-- 
Given a non-zero vector a = (m^2 - 1, m + 1) that is parallel to vector b = (1, -2),
prove that m = 1/2.
-/
theorem parallel_vectors_imply_m_value (m : ℝ) :
  (m^2 - 1 ≠ 0 ∨ m + 1 ≠ 0) →  -- Condition 1: Vector a is non-zero
  ∃ (k : ℝ), k ≠ 0 ∧ k * (m^2 - 1) = 1 ∧ k * (m + 1) = -2 →  -- Condition 2 and 3: Parallel vectors
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_value_l2235_223510


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2235_223552

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 16 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2235_223552


namespace NUMINAMATH_CALUDE_cos_BAD_equals_sqrt_13_45_l2235_223522

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the lengths of the sides
def AB (A B : ℝ × ℝ) : ℝ := sorry
def AC (A C : ℝ × ℝ) : ℝ := sorry
def BC (B C : ℝ × ℝ) : ℝ := sorry

-- Define a point D on BC
def D_on_BC (B C D : ℝ × ℝ) : Prop := sorry

-- Define the angle bisector property
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the cosine of an angle
def cos_angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem cos_BAD_equals_sqrt_13_45 
  (A B C D : ℝ × ℝ) 
  (h1 : Triangle A B C)
  (h2 : AB A B = 5)
  (h3 : AC A C = 9)
  (h4 : BC B C = 12)
  (h5 : D_on_BC B C D)
  (h6 : is_angle_bisector A B C D) :
  cos_angle B A D = Real.sqrt (13 / 45) := by
  sorry

end NUMINAMATH_CALUDE_cos_BAD_equals_sqrt_13_45_l2235_223522


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2235_223570

theorem polynomial_factorization (a b : ℝ) (x : ℝ) :
  a + (a+b)*x + (a+2*b)*x^2 + (a+3*b)*x^3 + 3*b*x^4 + 2*b*x^5 + b*x^6 = 
  (1 + x) * (1 + x^2) * (a + b*x + b*x^2 + b*x^3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2235_223570


namespace NUMINAMATH_CALUDE_sin_30_degrees_l2235_223544

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l2235_223544


namespace NUMINAMATH_CALUDE_recurrence_sequence_a1_zero_l2235_223545

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (c : ℝ) (a : ℕ → ℝ) : Prop :=
  c > 2 ∧ ∀ n : ℕ, a n = (a (n - 1))^2 - a (n - 1) ∧ a n < 1 / Real.sqrt (c * n)

/-- The main theorem stating that a₁ = 0 for any sequence satisfying the recurrence relation -/
theorem recurrence_sequence_a1_zero (c : ℝ) (a : ℕ → ℝ) (h : RecurrenceSequence c a) : a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_a1_zero_l2235_223545


namespace NUMINAMATH_CALUDE_polygon_sides_l2235_223528

/-- The number of sides of a polygon given the difference between its interior and exterior angle sums -/
theorem polygon_sides (interior_exterior_diff : ℝ) : interior_exterior_diff = 540 → ∃ n : ℕ, n = 7 ∧ 
  (n - 2) * 180 = 360 + interior_exterior_diff ∧ 
  (∀ m : ℕ, (m - 2) * 180 = 360 + interior_exterior_diff → m = n) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2235_223528


namespace NUMINAMATH_CALUDE_greatest_negative_root_of_sine_cosine_equation_l2235_223501

theorem greatest_negative_root_of_sine_cosine_equation :
  let α : ℝ := Real.arctan (1 / 8)
  let β : ℝ := Real.arctan (4 / 7)
  let root : ℝ := (α + β - 2 * Real.pi) / 9
  (∀ x : ℝ, x < 0 → Real.sin x + 8 * Real.cos x = 4 * Real.sin (8 * x) + 7 * Real.cos (8 * x) → x ≤ root) ∧
  Real.sin root + 8 * Real.cos root = 4 * Real.sin (8 * root) + 7 * Real.cos (8 * root) ∧
  root < 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_negative_root_of_sine_cosine_equation_l2235_223501


namespace NUMINAMATH_CALUDE_sock_pair_count_l2235_223582

/-- Given 8 pairs of socks, calculates the number of different pairs that can be formed
    by selecting 2 socks that are not from the same original pair -/
def sockPairs (totalPairs : Nat) : Nat :=
  let totalSocks := 2 * totalPairs
  let firstChoice := totalSocks
  let secondChoice := totalSocks - 2
  (firstChoice * secondChoice) / 2

/-- Theorem stating that with 8 pairs of socks, the number of different pairs
    that can be formed by selecting 2 socks not from the same original pair is 112 -/
theorem sock_pair_count : sockPairs 8 = 112 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2235_223582


namespace NUMINAMATH_CALUDE_postal_stamps_theorem_l2235_223572

/-- The number of color stamps sold -/
def color_stamps : ℕ := 578833

/-- The total number of stamps sold -/
def total_stamps : ℕ := 1102609

/-- The number of black-and-white stamps sold -/
def bw_stamps : ℕ := total_stamps - color_stamps

theorem postal_stamps_theorem : 
  bw_stamps = 523776 := by sorry

end NUMINAMATH_CALUDE_postal_stamps_theorem_l2235_223572


namespace NUMINAMATH_CALUDE_redbirds_count_l2235_223521

theorem redbirds_count (total : ℕ) (bluebird_fraction : ℚ) (h1 : total = 120) (h2 : bluebird_fraction = 5/6) :
  (1 - bluebird_fraction) * total = 20 := by
  sorry

end NUMINAMATH_CALUDE_redbirds_count_l2235_223521


namespace NUMINAMATH_CALUDE_binomial_coefficient_sequence_periodic_l2235_223598

/-- 
Given positive integers k and m, the sequence of binomial coefficients (n choose k) mod m,
where n ≥ k, is periodic.
-/
theorem binomial_coefficient_sequence_periodic (k m : ℕ+) :
  ∃ (p : ℕ+), ∀ (n : ℕ), n ≥ k →
    (n.choose k : ZMod m) = ((n + p) : ℕ).choose k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sequence_periodic_l2235_223598


namespace NUMINAMATH_CALUDE_dress_cost_theorem_l2235_223559

/-- The cost of a dress given the initial and remaining number of quarters -/
def dress_cost (initial_quarters remaining_quarters : ℕ) : ℚ :=
  (initial_quarters - remaining_quarters) * (1 / 4)

/-- Theorem stating that the dress cost $35 given the initial and remaining quarters -/
theorem dress_cost_theorem (initial_quarters remaining_quarters : ℕ) 
  (h1 : initial_quarters = 160)
  (h2 : remaining_quarters = 20) :
  dress_cost initial_quarters remaining_quarters = 35 := by
  sorry

#eval dress_cost 160 20

end NUMINAMATH_CALUDE_dress_cost_theorem_l2235_223559


namespace NUMINAMATH_CALUDE_chord_line_equation_l2235_223555

/-- The equation of a line passing through a chord of an ellipse, given the chord's midpoint -/
theorem chord_line_equation (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  M = (4, 2) →
  M.1 = (A.1 + B.1) / 2 →
  M.2 = (A.2 + B.2) / 2 →
  A.1^2 + 4 * A.2^2 = 36 →
  B.1^2 + 4 * B.2^2 = 36 →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → x + 2*y - 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2235_223555


namespace NUMINAMATH_CALUDE_original_books_l2235_223505

/-- The number of books person A has -/
def books_A : ℕ := sorry

/-- The number of books person B has -/
def books_B : ℕ := sorry

/-- If A gives 10 books to B, they have an equal number of books -/
axiom equal_books : books_A - 10 = books_B + 10

/-- If B gives 10 books to A, A has twice the number of books B has left -/
axiom double_books : books_A + 10 = 2 * (books_B - 10)

theorem original_books : books_A = 70 ∧ books_B = 50 := by sorry

end NUMINAMATH_CALUDE_original_books_l2235_223505


namespace NUMINAMATH_CALUDE_inequality_solution_l2235_223561

theorem inequality_solution (p q : ℝ) :
  q > 0 →
  (3 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 2 * p^2 * q ↔
  p ≥ 0 ∧ p < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2235_223561


namespace NUMINAMATH_CALUDE_train_length_calculation_l2235_223527

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 12 → 
  ∃ (length_m : ℝ), abs (length_m - 200.04) < 0.01 ∧ length_m = (speed_kmh * 1000 / 3600) * time_s :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2235_223527


namespace NUMINAMATH_CALUDE_max_additional_pens_is_four_l2235_223568

def initial_amount : ℕ := 100
def remaining_amount : ℕ := 61
def pens_bought : ℕ := 3

def cost_per_pen : ℕ := (initial_amount - remaining_amount) / pens_bought

def max_additional_pens : ℕ := remaining_amount / cost_per_pen

theorem max_additional_pens_is_four :
  max_additional_pens = 4 := by sorry

end NUMINAMATH_CALUDE_max_additional_pens_is_four_l2235_223568


namespace NUMINAMATH_CALUDE_sequence_properties_l2235_223536

theorem sequence_properties (a : ℕ+ → ℤ) (S : ℕ+ → ℤ) :
  (∀ n : ℕ+, S n = -n.val^2 + 24*n.val) →
  (∀ n : ℕ+, a n = S n - S (n-1)) →
  (∀ n : ℕ+, a n = -2*n.val + 25) ∧
  (∀ n : ℕ+, S n ≤ S 12) ∧
  (S 12 = 144) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2235_223536


namespace NUMINAMATH_CALUDE_common_rational_root_exists_l2235_223595

theorem common_rational_root_exists :
  ∃ (k : ℚ) (a b c d e f g : ℚ),
    k = -1/3 ∧
    k < 0 ∧
    ¬(∃ n : ℤ, k = n) ∧
    90 * k^4 + a * k^3 + b * k^2 + c * k + 18 = 0 ∧
    18 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 90 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_rational_root_exists_l2235_223595


namespace NUMINAMATH_CALUDE_two_correct_conclusions_l2235_223541

theorem two_correct_conclusions : ∃ (S : Finset (Prop)), S.card = 2 ∧ S ⊆ 
  {∀ (k b x₁ x₂ y₁ y₂ : ℝ), k < 0 → y₁ = k * x₁ + b → y₂ = k * x₂ + b → x₁ > x₂ → y₁ > y₂,
   ∀ (k b : ℝ), (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = k * x + b) ∧ 
                (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ y = k * x + b) ∧ 
                (∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = k * x + b) → 
                k > 0 ∧ b > 0,
   ∀ (m : ℝ), (m - 1) * 0 + m^2 + 2 = 3 → m = 1 ∨ m = -1} ∧ 
  (∀ p ∈ S, p) := by
sorry

end NUMINAMATH_CALUDE_two_correct_conclusions_l2235_223541


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l2235_223577

theorem greatest_integer_satisfying_conditions : ∃ n : ℕ, 
  n < 200 ∧ 
  ∃ k : ℕ, n + 2 = 9 * k ∧
  ∃ l : ℕ, n + 4 = 10 * l ∧
  ∀ m : ℕ, m < 200 → 
    (∃ p : ℕ, m + 2 = 9 * p) → 
    (∃ q : ℕ, m + 4 = 10 * q) → 
    m ≤ n ∧
  n = 166 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l2235_223577


namespace NUMINAMATH_CALUDE_cups_per_girl_l2235_223513

theorem cups_per_girl (total_students : ℕ) (boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  total_students = 30 →
  boys = 10 →
  cups_per_boy = 5 →
  total_cups = 90 →
  2 * boys = total_students - boys →
  (total_students - boys) * (total_cups - boys * cups_per_boy) / (total_students - boys) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cups_per_girl_l2235_223513


namespace NUMINAMATH_CALUDE_dye_making_water_amount_l2235_223589

/-- Given a dye-making process where:
  * The total mixture is 27 liters
  * 5/6 of 18 liters of vinegar is used
  * The water used is 3/5 of the total water available
  Prove that the amount of water used is 12 liters -/
theorem dye_making_water_amount (total_mixture : ℝ) (vinegar_amount : ℝ) (water_fraction : ℝ) :
  total_mixture = 27 →
  vinegar_amount = 5 / 6 * 18 →
  water_fraction = 3 / 5 →
  total_mixture - vinegar_amount = 12 :=
by sorry

end NUMINAMATH_CALUDE_dye_making_water_amount_l2235_223589


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2235_223525

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ

-- Define the given conditions
def given_triangle (x : ℝ) : Prop :=
  ∃ (t : EquilateralTriangle), t.side_length = 2*x ∧ t.side_length = x + 15

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

-- Theorem statement
theorem equilateral_triangle_perimeter :
  ∀ x : ℝ, given_triangle x → ∃ (t : EquilateralTriangle), perimeter t = 90 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2235_223525


namespace NUMINAMATH_CALUDE_omega_circle_l2235_223517

open Complex

/-- Given a complex number z satisfying |z - i| = 1, z ≠ 0, z ≠ 2i, and a complex number ω
    such that (ω / (ω - 2i)) * ((z - 2i) / z) is real, prove that ω lies on the circle
    centered at (0, 1) with radius 1, excluding the points (0, 0) and (0, 2). -/
theorem omega_circle (z ω : ℂ) 
  (h1 : abs (z - I) = 1)
  (h2 : z ≠ 0)
  (h3 : z ≠ 2 * I)
  (h4 : ∃ (r : ℝ), ω / (ω - 2 * I) * ((z - 2 * I) / z) = r) :
  abs (ω - I) = 1 ∧ ω ≠ 0 ∧ ω ≠ 2 * I :=
sorry

end NUMINAMATH_CALUDE_omega_circle_l2235_223517


namespace NUMINAMATH_CALUDE_field_width_l2235_223546

/-- The width of a rectangular field satisfying specific conditions -/
theorem field_width : ∃ (W : ℝ), W = 10 ∧ 
  20 * W * 0.5 - 40 * 0.5 = 8 * 5 * 2 := by
  sorry

end NUMINAMATH_CALUDE_field_width_l2235_223546


namespace NUMINAMATH_CALUDE_quadratic_equation_linear_coefficient_l2235_223551

/-- The exponent of x in the first term of the equation -/
def exponent (m : ℝ) : ℝ := m^2 - 2*m - 1

/-- The equation is quadratic when the exponent equals 2 -/
def is_quadratic (m : ℝ) : Prop := exponent m = 2

/-- The coefficient of x in the equation -/
def linear_coefficient (m : ℝ) : ℝ := -m

theorem quadratic_equation_linear_coefficient :
  ∀ m : ℝ, (m ≠ 3) → is_quadratic m → linear_coefficient m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_linear_coefficient_l2235_223551


namespace NUMINAMATH_CALUDE_min_type_b_workers_l2235_223579

/-- The number of workers in the workshop -/
def total_workers : ℕ := 20

/-- The number of Type A parts a worker can produce per day -/
def type_a_production : ℕ := 6

/-- The number of Type B parts a worker can produce per day -/
def type_b_production : ℕ := 5

/-- The profit (in yuan) from producing one Type A part -/
def type_a_profit : ℕ := 150

/-- The profit (in yuan) from producing one Type B part -/
def type_b_profit : ℕ := 260

/-- The daily profit function (in yuan) based on the number of workers producing Type A parts -/
def daily_profit (x : ℝ) : ℝ :=
  type_a_profit * type_a_production * x + type_b_profit * type_b_production * (total_workers - x)

/-- The minimum required daily profit (in yuan) -/
def min_profit : ℝ := 24000

theorem min_type_b_workers :
  ∀ x : ℝ, 0 ≤ x → x ≤ total_workers →
  (∀ y : ℝ, y ≥ min_profit → daily_profit x ≥ y) →
  total_workers - x ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_type_b_workers_l2235_223579


namespace NUMINAMATH_CALUDE_power_of_power_l2235_223548

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2235_223548


namespace NUMINAMATH_CALUDE_repeating_decimal_reciprocal_l2235_223524

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the repeating decimal 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem repeating_decimal_reciprocal :
  (repeating_decimal)⁻¹ = reciprocal := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_reciprocal_l2235_223524


namespace NUMINAMATH_CALUDE_lawn_care_supplies_l2235_223539

theorem lawn_care_supplies (blade_cost : ℕ) (string_cost : ℕ) (total_cost : ℕ) (num_blades : ℕ) :
  blade_cost = 8 →
  string_cost = 7 →
  total_cost = 39 →
  blade_cost * num_blades + string_cost = total_cost →
  num_blades = 4 := by
sorry

end NUMINAMATH_CALUDE_lawn_care_supplies_l2235_223539


namespace NUMINAMATH_CALUDE_distance_calculation_l2235_223535

def boat_speed : ℝ := 9
def stream_speed : ℝ := 6
def total_time : ℝ := 84

theorem distance_calculation (distance : ℝ) : 
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = total_time) → 
  distance = 210 := by
sorry

end NUMINAMATH_CALUDE_distance_calculation_l2235_223535


namespace NUMINAMATH_CALUDE_auction_price_increase_l2235_223509

/-- Represents an auction with a starting price, ending price, number of bidders, and bids per bidder -/
structure Auction where
  start_price : ℕ
  end_price : ℕ
  num_bidders : ℕ
  bids_per_bidder : ℕ

/-- Calculates the price increase per bid in an auction -/
def price_increase_per_bid (a : Auction) : ℚ :=
  (a.end_price - a.start_price : ℚ) / (a.num_bidders * a.bids_per_bidder : ℚ)

/-- Theorem stating that for the given auction conditions, the price increase per bid is $5 -/
theorem auction_price_increase (a : Auction)
  (h1 : a.start_price = 15)
  (h2 : a.end_price = 65)
  (h3 : a.num_bidders = 2)
  (h4 : a.bids_per_bidder = 5) :
  price_increase_per_bid a = 5 := by
  sorry

end NUMINAMATH_CALUDE_auction_price_increase_l2235_223509


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_15_l2235_223533

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_four_digit_sum_15 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 15 → n ≥ 1009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_15_l2235_223533


namespace NUMINAMATH_CALUDE_not_right_triangle_with_angle_ratio_l2235_223506

theorem not_right_triangle_with_angle_ratio (A B C : ℝ) (h : A + B + C = 180) 
  (ratio : A / 3 = B / 4 ∧ A / 3 = C / 5) : 
  ¬(A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_with_angle_ratio_l2235_223506


namespace NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l2235_223556

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | PentagonalPrism
  | Cube

-- Define a function that determines if a solid can have a rectangular cross-section
def canHaveRectangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | GeometricSolid.Cone => False
  | GeometricSolid.PentagonalPrism => True
  | GeometricSolid.Cube => True

-- Theorem stating that only the cone cannot have a rectangular cross-section
theorem only_cone_cannot_have_rectangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveRectangularCrossSection solid) ↔ solid = GeometricSolid.Cone :=
by sorry

end NUMINAMATH_CALUDE_only_cone_cannot_have_rectangular_cross_section_l2235_223556


namespace NUMINAMATH_CALUDE_marble_197_is_red_l2235_223562

/-- Represents the color of a marble -/
inductive MarbleColor
| Red
| Blue
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cycleLength := 15
  let position := n % cycleLength
  if position ≤ 6 then MarbleColor.Red
  else if position ≤ 11 then MarbleColor.Blue
  else MarbleColor.Green

/-- Theorem stating that the 197th marble is red -/
theorem marble_197_is_red : marbleColor 197 = MarbleColor.Red :=
sorry

end NUMINAMATH_CALUDE_marble_197_is_red_l2235_223562


namespace NUMINAMATH_CALUDE_special_function_period_l2235_223547

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, f (x + a) = (1 + f x) / (1 - f x)

/-- The period of a real function -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- The main theorem: if f is a SpecialFunction with parameter a, then it has period 4|a| -/
theorem special_function_period (f : ℝ → ℝ) (a : ℝ) 
    (hf : SpecialFunction f a) : HasPeriod f (4 * |a|) := by
  sorry

end NUMINAMATH_CALUDE_special_function_period_l2235_223547


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2235_223590

/-- 
Theorem: For a circular garden with radius r, if the length of the fence (circumference) 
is 1/4 of the area of the garden, then r = 8.
-/
theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1/4) * π * r^2 → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2235_223590


namespace NUMINAMATH_CALUDE_carmen_jethro_ratio_l2235_223575

-- Define the amounts of money for each person
def jethro_money : ℚ := 20
def patricia_money : ℚ := 60
def carmen_money : ℚ := 113 - jethro_money - patricia_money

-- Define the conditions
axiom patricia_triple_jethro : patricia_money = 3 * jethro_money
axiom total_money : carmen_money + jethro_money + patricia_money = 113
axiom carmen_multiple_after : ∃ (m : ℚ), carmen_money + 7 = m * jethro_money

-- Theorem to prove
theorem carmen_jethro_ratio :
  (carmen_money + 7) / jethro_money = 2 := by sorry

end NUMINAMATH_CALUDE_carmen_jethro_ratio_l2235_223575


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2235_223515

theorem square_perimeter_ratio (d1 d11 s1 s11 P1 P11 : ℝ) : 
  d1 > 0 → 
  d11 = 11 * d1 → 
  d1 = s1 * Real.sqrt 2 → 
  d11 = s11 * Real.sqrt 2 → 
  P1 = 4 * s1 → 
  P11 = 4 * s11 → 
  P11 / P1 = 11 := by
sorry


end NUMINAMATH_CALUDE_square_perimeter_ratio_l2235_223515


namespace NUMINAMATH_CALUDE_squirrel_acorns_l2235_223538

theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (additional_acorns : ℕ) :
  num_squirrels = 5 →
  total_acorns = 575 →
  additional_acorns = 15 →
  ∃ (acorns_per_squirrel : ℕ),
    acorns_per_squirrel = 130 ∧
    num_squirrels * (acorns_per_squirrel - additional_acorns) = total_acorns :=
by sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l2235_223538


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l2235_223530

def hotel_loss (expenses : ℝ) (payment_ratio : ℝ) : ℝ :=
  expenses - (payment_ratio * expenses)

theorem hotel_loss_calculation (expenses : ℝ) (payment_ratio : ℝ) 
  (h1 : expenses = 100)
  (h2 : payment_ratio = 3/4) :
  hotel_loss expenses payment_ratio = 25 := by
sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l2235_223530


namespace NUMINAMATH_CALUDE_brocard_point_characterization_l2235_223516

open Real

/-- Triangle structure with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Point structure with coordinates x, y -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given side lengths -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the area of a triangle given three points -/
def areaFromPoints (p1 p2 p3 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Definition of Brocard point -/
def isBrocardPoint (p : Point) (t : Triangle) : Prop :=
  let s_abc := triangleArea t
  let s_pbc := areaFromPoints p (Point.mk 0 0) (Point.mk t.c 0)
  let s_pca := areaFromPoints p (Point.mk t.c 0) (Point.mk 0 t.a)
  let s_pab := areaFromPoints p (Point.mk 0 0) (Point.mk 0 t.a)
  isInside p t ∧
  (s_pbc / (t.c^2 * t.a^2) = s_pca / (t.a^2 * t.b^2)) ∧
  (s_pca / (t.a^2 * t.b^2) = s_pab / (t.b^2 * t.c^2)) ∧
  (s_pab / (t.b^2 * t.c^2) = s_abc / (t.a^2 * t.b^2 + t.b^2 * t.c^2 + t.c^2 * t.a^2))

/-- Theorem: Characterization of Brocard point -/
theorem brocard_point_characterization (t : Triangle) (p : Point) :
  isBrocardPoint p t ↔
  (let s_abc := triangleArea t
   let s_pbc := areaFromPoints p (Point.mk 0 0) (Point.mk t.c 0)
   let s_pca := areaFromPoints p (Point.mk t.c 0) (Point.mk 0 t.a)
   let s_pab := areaFromPoints p (Point.mk 0 0) (Point.mk 0 t.a)
   isInside p t ∧
   (s_pbc / (t.c^2 * t.a^2) = s_pca / (t.a^2 * t.b^2)) ∧
   (s_pca / (t.a^2 * t.b^2) = s_pab / (t.b^2 * t.c^2)) ∧
   (s_pab / (t.b^2 * t.c^2) = s_abc / (t.a^2 * t.b^2 + t.b^2 * t.c^2 + t.c^2 * t.a^2))) :=
by sorry

end NUMINAMATH_CALUDE_brocard_point_characterization_l2235_223516


namespace NUMINAMATH_CALUDE_power_equation_solution_l2235_223540

theorem power_equation_solution (K : ℕ) : 32^4 * 4^6 = 2^K → K = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2235_223540


namespace NUMINAMATH_CALUDE_equation_solution_l2235_223518

theorem equation_solution : ∃ x : ℚ, 5 * (x - 8) + 6 = 3 * (3 - 3 * x) + 15 ∧ x = 29 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2235_223518


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2235_223523

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 7 * p^2 + 2 * p - 4 = 0) →
  (3 * q^3 - 7 * q^2 + 2 * q - 4 = 0) →
  (3 * r^3 - 7 * r^2 + 2 * r - 4 = 0) →
  p^2 + q^2 + r^2 = 37/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2235_223523


namespace NUMINAMATH_CALUDE_vanya_more_heads_probability_vanya_more_heads_probability_is_half_l2235_223565

/-- The probability that Vanya gets more heads than Tanya when Vanya flips a coin n+1 times and Tanya flips a coin n times. -/
theorem vanya_more_heads_probability (n : ℕ) : ℝ :=
  let vanya_flips := n + 1
  let tanya_flips := n
  let prob_vanya_more_heads := (1 : ℝ) / 2
  prob_vanya_more_heads

/-- Proof that the probability of Vanya getting more heads than Tanya is 1/2. -/
theorem vanya_more_heads_probability_is_half (n : ℕ) :
  vanya_more_heads_probability n = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vanya_more_heads_probability_vanya_more_heads_probability_is_half_l2235_223565


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2235_223586

/-- The line equation passes through a fixed point for all values of k -/
theorem fixed_point_on_line (k : ℝ) : 
  (2 * k - 1) * 2 - (k - 2) * 3 - (k + 4) = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2235_223586


namespace NUMINAMATH_CALUDE_fixed_point_transformation_l2235_223594

theorem fixed_point_transformation (f : ℝ → ℝ) (h : f 1 = 1) : f (4 - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_transformation_l2235_223594


namespace NUMINAMATH_CALUDE_vector_dot_product_zero_l2235_223592

theorem vector_dot_product_zero (a b : ℝ × ℝ) (h1 : a = (2, 0)) (h2 : b = (1/2, Real.sqrt 3 / 2)) :
  b • (a - b) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_zero_l2235_223592


namespace NUMINAMATH_CALUDE_complement_of_M_l2235_223587

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2235_223587
