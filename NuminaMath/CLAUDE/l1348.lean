import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_B_l1348_134815

def A : Set ℕ := {0,1,2,3,4,5,6}

def B : Set ℕ := {x | ∃ n ∈ A, x = 2 * n}

theorem intersection_A_B : A ∩ B = {0,2,4,6} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1348_134815


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l1348_134877

theorem largest_square_tile_size (wall_width wall_length : ℕ) 
  (hw : wall_width = 24) (hl : wall_length = 18) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    wall_width % tile_size = 0 ∧ 
    wall_length % tile_size = 0 ∧
    ∀ (other_size : ℕ), 
      (wall_width % other_size = 0 ∧ wall_length % other_size = 0) → 
      other_size ≤ tile_size :=
by
  -- The proof would go here
  sorry

#check largest_square_tile_size

end NUMINAMATH_CALUDE_largest_square_tile_size_l1348_134877


namespace NUMINAMATH_CALUDE_zongzi_purchase_l1348_134861

/-- Represents the unit price and quantity of zongzi -/
structure Zongzi where
  price : ℝ
  quantity : ℝ

/-- Represents the purchase of zongzi -/
def Purchase (a b : Zongzi) : Prop :=
  a.price * a.quantity = 1500 ∧
  b.price * b.quantity = 1000 ∧
  b.quantity = a.quantity + 50 ∧
  a.price = 2 * b.price

/-- Represents the additional purchase constraint -/
def AdditionalPurchase (a b : Zongzi) (x : ℝ) : Prop :=
  x + (200 - x) = 200 ∧
  a.price * x + b.price * (200 - x) ≤ 1450

/-- Main theorem -/
theorem zongzi_purchase (a b : Zongzi) (x : ℝ) 
  (h1 : Purchase a b) (h2 : AdditionalPurchase a b x) : 
  b.price = 5 ∧ a.price = 10 ∧ x ≤ 90 := by
  sorry

#check zongzi_purchase

end NUMINAMATH_CALUDE_zongzi_purchase_l1348_134861


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1348_134802

/-- A quadratic equation x^2 + mx + 9 has two distinct real roots if and only if m < -6 or m > 6 -/
theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ m < -6 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1348_134802


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1348_134873

/-- Proves that the average salary of all workers in a workshop is 8000 rupees. -/
theorem workshop_average_salary :
  let total_workers : ℕ := 21
  let technicians : ℕ := 7
  let technician_salary : ℕ := 12000
  let non_technician_salary : ℕ := 6000
  (total_workers * (technicians * technician_salary + (total_workers - technicians) * non_technician_salary)) / (total_workers * total_workers) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1348_134873


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1348_134820

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^2 + y^2) = x + y) : x - y ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1348_134820


namespace NUMINAMATH_CALUDE_era_burger_slices_l1348_134859

/-- Given the distribution of burger slices among Era and her friends, prove that Era is left with 1 slice. -/
theorem era_burger_slices (total_burgers : ℕ) (friends : ℕ) (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) (friend4_slices : ℕ) : 
  total_burgers = 5 →
  friends = 4 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  total_burgers * 2 - (friend1_slices + friend2_slices + friend3_slices + friend4_slices) = 1 := by
  sorry

end NUMINAMATH_CALUDE_era_burger_slices_l1348_134859


namespace NUMINAMATH_CALUDE_shekar_average_marks_l1348_134878

def shekar_scores : List ℕ := [76, 65, 82, 67, 55]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℚ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l1348_134878


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1348_134863

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional innings -/
def newAverage (bp : BatsmanPerformance) (newScore : ℕ) : ℚ :=
  (bp.totalRuns + newScore) / (bp.innings + 1)

theorem batsman_average_after_12th_innings
  (bp : BatsmanPerformance)
  (h1 : bp.innings = 11)
  (h2 : newAverage bp 48 = bp.average + 2)
  : newAverage bp 48 = 26 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1348_134863


namespace NUMINAMATH_CALUDE_skill_players_count_l1348_134872

/-- Represents the football team's water consumption scenario -/
structure FootballTeam where
  cooler_capacity : ℕ
  num_linemen : ℕ
  lineman_consumption : ℕ
  skill_player_consumption : ℕ
  waiting_skill_players : ℕ

/-- Calculates the number of skill position players on the team -/
def num_skill_players (team : FootballTeam) : ℕ :=
  let remaining_water := team.cooler_capacity - team.num_linemen * team.lineman_consumption
  let drinking_skill_players := remaining_water / team.skill_player_consumption
  drinking_skill_players + team.waiting_skill_players

/-- Theorem stating the number of skill position players on the team -/
theorem skill_players_count (team : FootballTeam) 
  (h1 : team.cooler_capacity = 126)
  (h2 : team.num_linemen = 12)
  (h3 : team.lineman_consumption = 8)
  (h4 : team.skill_player_consumption = 6)
  (h5 : team.waiting_skill_players = 5) :
  num_skill_players team = 10 := by
  sorry

#eval num_skill_players {
  cooler_capacity := 126,
  num_linemen := 12,
  lineman_consumption := 8,
  skill_player_consumption := 6,
  waiting_skill_players := 5
}

end NUMINAMATH_CALUDE_skill_players_count_l1348_134872


namespace NUMINAMATH_CALUDE_min_unique_points_is_eight_l1348_134832

/-- A square with points marked on its sides -/
structure MarkedSquare where
  /-- The number of points marked on each side of the square -/
  pointsPerSide : ℕ
  /-- Condition: Each side has exactly 3 points -/
  threePointsPerSide : pointsPerSide = 3

/-- The minimum number of unique points marked on the square -/
def minUniquePoints (s : MarkedSquare) : ℕ :=
  s.pointsPerSide * 4 - 4

/-- Theorem: The minimum number of unique points marked on the square is 8 -/
theorem min_unique_points_is_eight (s : MarkedSquare) : 
  minUniquePoints s = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_unique_points_is_eight_l1348_134832


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l1348_134895

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4851 → 
  min a b = 49 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l1348_134895


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1348_134857

/-- Given f(x) = 3x + 2, prove that there exists a positive integer m 
    such that f^(100)(m) is divisible by 1988 -/
theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, (3^100 : ℤ) * m.val + (3^100 - 1) ∣ 1988 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l1348_134857


namespace NUMINAMATH_CALUDE_middle_angle_range_l1348_134884

theorem middle_angle_range (β : Real) (h1 : 0 < β) (h2 : β < 90) : 
  ∃ (α γ : Real), 
    0 < α ∧ 0 < γ ∧ 
    α + β + γ = 180 ∧ 
    α ≤ β ∧ β ≤ γ :=
by sorry

end NUMINAMATH_CALUDE_middle_angle_range_l1348_134884


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l1348_134848

def shorts_cost : ℚ := 13.99
def jacket_cost : ℚ := 7.43
def total_cost : ℚ := 33.56

theorem shirt_cost_calculation :
  total_cost - (shorts_cost + jacket_cost) = 12.14 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l1348_134848


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1348_134838

theorem sum_of_cubes (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : (x + y)^2 = 2500) (h2 : x * y = 500) : x^3 + y^3 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1348_134838


namespace NUMINAMATH_CALUDE_sqrt_sum_power_equality_l1348_134818

theorem sqrt_sum_power_equality (m n : ℕ) : 
  ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_power_equality_l1348_134818


namespace NUMINAMATH_CALUDE_solution_set_of_f_minimum_value_of_sum_l1348_134883

-- Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem solution_set_of_f (x : ℝ) : f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) := by sorry

-- Part II
theorem minimum_value_of_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 ∧ ∃ (p' q' r' : ℝ), p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ 
    1/(3*p') + 1/(2*q') + 1/r' = 4 ∧ 3*p' + 2*q' + r' = 9/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_minimum_value_of_sum_l1348_134883


namespace NUMINAMATH_CALUDE_complex_magnitude_l1348_134821

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1348_134821


namespace NUMINAMATH_CALUDE_books_added_by_marta_l1348_134885

theorem books_added_by_marta (initial_books final_books : ℕ) 
  (h1 : initial_books = 38)
  (h2 : final_books = 48)
  (h3 : final_books ≥ initial_books) :
  final_books - initial_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_added_by_marta_l1348_134885


namespace NUMINAMATH_CALUDE_chameleon_color_change_l1348_134853

theorem chameleon_color_change (total : ℕ) (blue_factor red_factor : ℕ) : 
  total = 140 → blue_factor = 5 → red_factor = 3 →
  ∃ (initial_blue : ℕ),
    initial_blue > 0 ∧
    initial_blue * blue_factor ≤ total ∧
    (total - initial_blue * blue_factor) * red_factor + initial_blue = total →
    initial_blue * blue_factor - initial_blue = 80 := by
  sorry

#check chameleon_color_change

end NUMINAMATH_CALUDE_chameleon_color_change_l1348_134853


namespace NUMINAMATH_CALUDE_bag_selling_price_l1348_134823

/-- The selling price of a discounted item -/
def selling_price (marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

/-- Theorem: The selling price of a bag marked at $80 with a 15% discount is $68 -/
theorem bag_selling_price :
  selling_price 80 0.15 = 68 := by
  sorry

end NUMINAMATH_CALUDE_bag_selling_price_l1348_134823


namespace NUMINAMATH_CALUDE_expected_same_color_edges_l1348_134894

/-- Represents a 3x3 board -/
def Board := Fin 3 → Fin 3 → Bool

/-- The number of squares in the board -/
def boardSize : Nat := 9

/-- The number of squares to be blackened -/
def blackSquares : Nat := 5

/-- The total number of pairs of adjacent squares -/
def totalAdjacentPairs : Nat := 12

/-- Calculates the probability that two adjacent squares have the same color -/
noncomputable def probSameColor : ℚ := 4 / 9

/-- Theorem: The expected number of edges between two squares of the same color 
    in a 3x3 board with 5 randomly blackened squares is 16/3 -/
theorem expected_same_color_edges :
  (totalAdjacentPairs : ℚ) * probSameColor = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_same_color_edges_l1348_134894


namespace NUMINAMATH_CALUDE_shark_teeth_problem_l1348_134899

theorem shark_teeth_problem (S : ℚ) 
  (hammerhead : ℚ → ℚ)
  (great_white : ℚ → ℚ)
  (h1 : hammerhead S = S / 6)
  (h2 : great_white S = 2 * (S + hammerhead S))
  (h3 : great_white S = 420) : 
  S = 180 := by sorry

end NUMINAMATH_CALUDE_shark_teeth_problem_l1348_134899


namespace NUMINAMATH_CALUDE_toms_blue_marbles_l1348_134880

theorem toms_blue_marbles (jason_blue : ℕ) (total_blue : ℕ) 
  (h1 : jason_blue = 44)
  (h2 : total_blue = 68) :
  total_blue - jason_blue = 24 := by
sorry

end NUMINAMATH_CALUDE_toms_blue_marbles_l1348_134880


namespace NUMINAMATH_CALUDE_chucks_team_score_final_score_proof_l1348_134836

theorem chucks_team_score (red_team_score : ℕ) (score_difference : ℕ) : ℕ :=
  red_team_score + score_difference

theorem final_score_proof (red_team_score : ℕ) (score_difference : ℕ) 
  (h1 : red_team_score = 76)
  (h2 : score_difference = 19) :
  chucks_team_score red_team_score score_difference = 95 := by
  sorry

end NUMINAMATH_CALUDE_chucks_team_score_final_score_proof_l1348_134836


namespace NUMINAMATH_CALUDE_red_permutations_l1348_134830

theorem red_permutations : 
  let n : ℕ := 1
  let total_letters : ℕ := 3 * n
  let permutations : ℕ := Nat.factorial total_letters / (Nat.factorial n)^3
  permutations = 6 := by sorry

end NUMINAMATH_CALUDE_red_permutations_l1348_134830


namespace NUMINAMATH_CALUDE_books_left_unpacked_l1348_134840

theorem books_left_unpacked (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1485 →
  books_per_initial_box = 42 →
  books_per_new_box = 45 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 30 := by
  sorry

end NUMINAMATH_CALUDE_books_left_unpacked_l1348_134840


namespace NUMINAMATH_CALUDE_modulus_of_z_l1348_134870

theorem modulus_of_z (z : ℂ) (h : z / (1 + 2 * I) = 1) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1348_134870


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1348_134893

def is_solution (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 - 3*x*y*z = 2003

theorem cubic_equation_solutions :
  ∀ x y z : ℤ, is_solution x y z ↔ 
    ((x = 668 ∧ y = 668 ∧ z = 667) ∨
     (x = 668 ∧ y = 667 ∧ z = 668) ∨
     (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1348_134893


namespace NUMINAMATH_CALUDE_solution_is_correct_l1348_134876

/-- The equation to be solved -/
def equation (y : ℝ) : Prop :=
  1/6 + 6/y = 14/y - 1/14

/-- The theorem stating that y = 168/5 is the solution to the equation -/
theorem solution_is_correct : equation (168/5) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_correct_l1348_134876


namespace NUMINAMATH_CALUDE_chipped_marbles_possibilities_l1348_134842

def marble_counts : List Nat := [15, 18, 20, 22, 24, 27, 30, 32, 35, 37]

def total_marbles : Nat := marble_counts.sum

theorem chipped_marbles_possibilities :
  ∀ n : Nat, n ∈ marble_counts →
  (total_marbles - n) % 5 = 0 →
  n % 5 = 0 →
  n ∈ [15, 20, 30, 35] :=
by sorry

end NUMINAMATH_CALUDE_chipped_marbles_possibilities_l1348_134842


namespace NUMINAMATH_CALUDE_mistaken_calculation_correction_l1348_134831

theorem mistaken_calculation_correction (x : ℝ) :
  5.46 - x = 3.97 → 5.46 + x = 6.95 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_correction_l1348_134831


namespace NUMINAMATH_CALUDE_range_of_a_l1348_134847

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : is_even_function f)
  (h_incr : increasing_on_nonpositive f)
  (h_ineq : f a ≥ f 2) :
  a ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1348_134847


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l1348_134866

theorem ice_cream_sundaes (total_flavors : ℕ) (h_total : total_flavors = 8) :
  let required_flavor := 1
  let sundae_size := 2
  let max_sundaes := total_flavors - required_flavor
  max_sundaes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l1348_134866


namespace NUMINAMATH_CALUDE_total_chocolate_pieces_l1348_134833

theorem total_chocolate_pieces (num_boxes : ℕ) (pieces_per_box : ℕ) 
  (h1 : num_boxes = 6) 
  (h2 : pieces_per_box = 500) : 
  num_boxes * pieces_per_box = 3000 := by
  sorry

end NUMINAMATH_CALUDE_total_chocolate_pieces_l1348_134833


namespace NUMINAMATH_CALUDE_winnie_min_checks_l1348_134888

/-- Represents the arrangement of jars in Winnie the Pooh's closet -/
structure JarArrangement where
  total : Nat
  jam : Nat
  honey : Nat
  honey_consecutive : Bool

/-- Defines the minimum number of jars to check to find honey -/
def min_checks (arrangement : JarArrangement) : Nat :=
  1

/-- Theorem stating that for Winnie's specific arrangement, the minimum number of checks is 1 -/
theorem winnie_min_checks :
  ∀ (arrangement : JarArrangement),
    arrangement.total = 11 →
    arrangement.jam = 7 →
    arrangement.honey = 4 →
    arrangement.honey_consecutive = true →
    min_checks arrangement = 1 := by
  sorry

end NUMINAMATH_CALUDE_winnie_min_checks_l1348_134888


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1348_134862

theorem car_fuel_efficiency 
  (x : ℝ) 
  (h1 : x > 0) 
  (tank_capacity : ℝ) 
  (h2 : tank_capacity = 12) 
  (efficiency_improvement : ℝ) 
  (h3 : efficiency_improvement = 0.8) 
  (distance_increase : ℝ) 
  (h4 : distance_increase = 96) :
  (tank_capacity * x / efficiency_improvement) - (tank_capacity * x) = distance_increase →
  x = 32 :=
by sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1348_134862


namespace NUMINAMATH_CALUDE_jane_age_ratio_l1348_134855

/-- Represents the ages of Jane and her children at two different times -/
structure FamilyAges where
  J : ℝ  -- Jane's current age
  M : ℝ  -- Years ago
  younger_sum : ℝ  -- Sum of ages of two younger children
  oldest : ℝ  -- Age of oldest child

/-- The conditions given in the problem -/
def satisfies_conditions (ages : FamilyAges) : Prop :=
  ages.J > 0 ∧ 
  ages.M > 0 ∧
  ages.J = 2 * ages.younger_sum ∧
  ages.J = ages.oldest / 2 ∧
  ages.J - ages.M = 3 * (ages.younger_sum - 2 * ages.M) ∧
  ages.J - ages.M = ages.oldest - ages.M

theorem jane_age_ratio (ages : FamilyAges) 
  (h : satisfies_conditions ages) : ages.J / ages.M = 10 := by
  sorry

end NUMINAMATH_CALUDE_jane_age_ratio_l1348_134855


namespace NUMINAMATH_CALUDE_six_term_sequence_count_l1348_134826

def sequence_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n.choose c * (n - c).choose b

theorem six_term_sequence_count : sequence_count 6 3 2 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_term_sequence_count_l1348_134826


namespace NUMINAMATH_CALUDE_remainder_theorem_l1348_134805

-- Define the polynomial P
variable (P : ℝ → ℝ)

-- Define the conditions
axiom P_19 : P 19 = 99
axiom P_99 : P 99 = 19

-- Define the remainder function
def remainder (P : ℝ → ℝ) : ℝ → ℝ :=
  λ x => -x + 118

-- Theorem statement
theorem remainder_theorem :
  ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = (x - 19) * (x - 99) * Q x + remainder P x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1348_134805


namespace NUMINAMATH_CALUDE_complex_number_value_l1348_134837

theorem complex_number_value (z : ℂ) (h : (z - 1) * Complex.I = Complex.abs (Complex.I + 1)) :
  z = 1 - Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_value_l1348_134837


namespace NUMINAMATH_CALUDE_goats_sold_proof_l1348_134892

/-- Represents the number of animals sold -/
def total_animals : ℕ := 80

/-- Represents the total reduction in legs -/
def total_leg_reduction : ℕ := 200

/-- Represents the number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- Represents the number of legs a goat has -/
def goat_legs : ℕ := 4

/-- Represents the number of goats sold -/
def goats_sold : ℕ := 20

/-- Represents the number of chickens sold -/
def chickens_sold : ℕ := total_animals - goats_sold

theorem goats_sold_proof :
  goats_sold * goat_legs + chickens_sold * chicken_legs = total_leg_reduction ∧
  goats_sold + chickens_sold = total_animals :=
by sorry

end NUMINAMATH_CALUDE_goats_sold_proof_l1348_134892


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1348_134824

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_equation :
  let r₁ := (6 + Real.sqrt (36 - 32)) / 2
  let r₂ := (6 - Real.sqrt (36 - 32)) / 2
  r₁ + r₂ = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1348_134824


namespace NUMINAMATH_CALUDE_cherry_popsicles_count_l1348_134810

theorem cherry_popsicles_count (total : ℕ) (grape : ℕ) (banana : ℕ) (cherry : ℕ) :
  total = 17 → grape = 2 → banana = 2 → cherry = total - (grape + banana) → cherry = 13 := by
  sorry

end NUMINAMATH_CALUDE_cherry_popsicles_count_l1348_134810


namespace NUMINAMATH_CALUDE_defective_probability_l1348_134803

/-- The probability that both selected products are defective given that one is defective -/
theorem defective_probability (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ) 
  (h1 : total = genuine + defective)
  (h2 : total = 10)
  (h3 : genuine = 6)
  (h4 : defective = 4)
  (h5 : selected = 2) :
  (defective.choose 2 : ℚ) / (total.choose 2) / 
  ((defective.choose 1 * genuine.choose 1 + defective.choose 2 : ℚ) / total.choose 2) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_defective_probability_l1348_134803


namespace NUMINAMATH_CALUDE_hospital_bill_breakdown_l1348_134835

theorem hospital_bill_breakdown (total_bill : ℝ) (medication_percentage : ℝ) 
  (overnight_percentage : ℝ) (food_cost : ℝ) (h1 : total_bill = 5000) 
  (h2 : medication_percentage = 0.5) (h3 : overnight_percentage = 0.25) 
  (h4 : food_cost = 175) : 
  total_bill * (1 - medication_percentage) * (1 - overnight_percentage) - food_cost = 1700 := by
  sorry

end NUMINAMATH_CALUDE_hospital_bill_breakdown_l1348_134835


namespace NUMINAMATH_CALUDE_union_definition_l1348_134860

theorem union_definition (A B : Set α) : 
  A ∪ B = {x | x ∈ A ∨ x ∈ B} := by
sorry

end NUMINAMATH_CALUDE_union_definition_l1348_134860


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l1348_134843

/-- Given a mixture of two types of candy, prove the cost of the second type. -/
theorem candy_mixture_cost
  (total_mixture : ℝ)
  (first_candy_weight : ℝ)
  (first_candy_cost : ℝ)
  (second_candy_weight : ℝ)
  (mixture_cost : ℝ)
  (h1 : total_mixture = first_candy_weight + second_candy_weight)
  (h2 : total_mixture = 45)
  (h3 : first_candy_weight = 15)
  (h4 : first_candy_cost = 8)
  (h5 : second_candy_weight = 30)
  (h6 : mixture_cost = 6) :
  ∃ (second_candy_cost : ℝ),
    second_candy_cost = 5 ∧
    total_mixture * mixture_cost =
      first_candy_weight * first_candy_cost +
      second_candy_weight * second_candy_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l1348_134843


namespace NUMINAMATH_CALUDE_factorial_equation_unique_solution_l1348_134867

theorem factorial_equation_unique_solution :
  ∃! m : ℕ+, (Nat.factorial 6) * (Nat.factorial 11) = 18 * (Nat.factorial m.val) * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_unique_solution_l1348_134867


namespace NUMINAMATH_CALUDE_train_speed_l1348_134864

/-- Given a bridge and a train, calculate the train's speed in km/h -/
theorem train_speed (bridge_length train_length : ℝ) (crossing_time : ℝ) : 
  bridge_length = 200 →
  train_length = 100 →
  crossing_time = 60 →
  (bridge_length + train_length) / crossing_time * 3.6 = 18 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1348_134864


namespace NUMINAMATH_CALUDE_baseAngle_eq_pi_div_k_l1348_134886

/-- An isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoidAroundCircle where
  /-- The ratio of the parallel sides -/
  k : ℝ
  /-- The angle at the base -/
  baseAngle : ℝ

/-- Theorem: The angle at the base of an isosceles trapezoid inscribed around a circle
    is equal to π/k, where k is the ratio of the parallel sides -/
theorem baseAngle_eq_pi_div_k (t : IsoscelesTrapezoidAroundCircle) :
  t.baseAngle = π / t.k :=
sorry

end NUMINAMATH_CALUDE_baseAngle_eq_pi_div_k_l1348_134886


namespace NUMINAMATH_CALUDE_parallelepiped_coverage_l1348_134812

/-- A parallelepiped with dimensions a, b, and c can have three faces sharing a common vertex
    covered by five-cell strips without overlaps or gaps if and only if at least two of a, b,
    and c are divisible by 5. -/
theorem parallelepiped_coverage (a b c : ℕ) :
  (∃ (faces : Fin 3 → ℕ × ℕ), 
    (faces 0 = (a, b) ∨ faces 0 = (b, c) ∨ faces 0 = (c, a)) ∧
    (faces 1 = (a, b) ∨ faces 1 = (b, c) ∨ faces 1 = (c, a)) ∧
    (faces 2 = (a, b) ∨ faces 2 = (b, c) ∨ faces 2 = (c, a)) ∧
    faces 0 ≠ faces 1 ∧ faces 1 ≠ faces 2 ∧ faces 0 ≠ faces 2 ∧
    ∀ i : Fin 3, ∃ k : ℕ, (faces i).1 * (faces i).2 = 5 * k) ↔
  (a % 5 = 0 ∧ b % 5 = 0) ∨ (b % 5 = 0 ∧ c % 5 = 0) ∨ (c % 5 = 0 ∧ a % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_coverage_l1348_134812


namespace NUMINAMATH_CALUDE_race_course_length_correct_l1348_134816

/-- The length of a race course where two runners finish at the same time -/
def race_course_length : ℝ :=
  let speed_ratio : ℝ := 7
  let head_start : ℝ := 120
  140

theorem race_course_length_correct :
  let speed_ratio : ℝ := 7  -- A is 7 times faster than B
  let head_start : ℝ := 120 -- B starts 120 meters ahead
  let course_length := race_course_length
  course_length / speed_ratio = (course_length - head_start) / 1 :=
by sorry

end NUMINAMATH_CALUDE_race_course_length_correct_l1348_134816


namespace NUMINAMATH_CALUDE_extremum_values_l1348_134814

theorem extremum_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ Real.sqrt 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/(2*y + 1) ≥ (3 + 2*Real.sqrt 2)/3) := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l1348_134814


namespace NUMINAMATH_CALUDE_rock_collection_total_l1348_134850

theorem rock_collection_total (igneous sedimentary : ℕ) : 
  igneous = sedimentary / 2 →
  (2 : ℕ) * igneous / 3 = 40 →
  igneous + sedimentary = 180 :=
by
  sorry

#check rock_collection_total

end NUMINAMATH_CALUDE_rock_collection_total_l1348_134850


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1348_134819

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x^2 + 4*x) = 9 ∧ x^2 + 4*x ≥ 0 :=
by
  -- The unique solution is x = -2 + √85
  use -2 + Real.sqrt 85
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1348_134819


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1348_134844

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 2 = 1 →
  a 5 * a 6 = 4 →
  a 3 * a 4 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1348_134844


namespace NUMINAMATH_CALUDE_cindy_paint_area_l1348_134849

/-- Given that Allen, Ben, and Cindy are painting a fence, where:
    1) The ratio of work done by Allen : Ben : Cindy is 3 : 5 : 2
    2) The total fence area to be painted is 300 square feet
    Prove that Cindy paints 60 square feet of the fence. -/
theorem cindy_paint_area (total_area : ℝ) (allen_ratio ben_ratio cindy_ratio : ℕ) :
  total_area = 300 ∧ 
  allen_ratio = 3 ∧ 
  ben_ratio = 5 ∧ 
  cindy_ratio = 2 →
  cindy_ratio * (total_area / (allen_ratio + ben_ratio + cindy_ratio)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_cindy_paint_area_l1348_134849


namespace NUMINAMATH_CALUDE_juan_and_maria_distance_l1348_134828

/-- The combined distance covered by two runners given their speeds and times -/
def combined_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem stating that the combined distance of Juan and Maria is 658.5 miles -/
theorem juan_and_maria_distance :
  combined_distance 9.5 30 8.3 45 = 658.5 := by
  sorry

end NUMINAMATH_CALUDE_juan_and_maria_distance_l1348_134828


namespace NUMINAMATH_CALUDE_playground_area_ratio_l1348_134839

theorem playground_area_ratio (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (3*r)^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_ratio_l1348_134839


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_l1348_134897

-- Define the inequalities
def inequality1 (x : ℝ) := 2 * x^2 - 3 * x + 1 ≥ 0
def inequality2 (x : ℝ) := x^2 - 2 * x - 3 < 0
def inequality3 (x : ℝ) := -3 * x^2 + 5 * x - 2 > 0

-- Define the solution sets
def solution1 : Set ℝ := {x | x ≤ 1/2 ∨ x ≥ 1}
def solution2 : Set ℝ := {x | -1 < x ∧ x < 3}
def solution3 : Set ℝ := {x | 2/3 < x ∧ x < 1}

-- Theorem statements
theorem inequality1_solution : ∀ x : ℝ, x ∈ solution1 ↔ inequality1 x := by sorry

theorem inequality2_solution : ∀ x : ℝ, x ∈ solution2 ↔ inequality2 x := by sorry

theorem inequality3_solution : ∀ x : ℝ, x ∈ solution3 ↔ inequality3 x := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_inequality3_solution_l1348_134897


namespace NUMINAMATH_CALUDE_kylie_piggy_bank_coins_kylie_piggy_bank_coins_value_l1348_134808

/-- The number of coins Kylie got from her piggy bank -/
def coins_from_piggy_bank : ℕ := sorry

/-- The number of coins Kylie got from her brother -/
def coins_from_brother : ℕ := 13

/-- The number of coins Kylie got from her father -/
def coins_from_father : ℕ := 8

/-- The number of coins Kylie gave to Laura -/
def coins_given_to_laura : ℕ := 21

/-- The number of coins Kylie had left -/
def coins_left : ℕ := 15

theorem kylie_piggy_bank_coins :
  coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_laura = coins_left :=
by sorry

theorem kylie_piggy_bank_coins_value : coins_from_piggy_bank = 15 :=
by sorry

end NUMINAMATH_CALUDE_kylie_piggy_bank_coins_kylie_piggy_bank_coins_value_l1348_134808


namespace NUMINAMATH_CALUDE_square_diff_equals_32_l1348_134822

theorem square_diff_equals_32 (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) :
  a^2 - b^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_diff_equals_32_l1348_134822


namespace NUMINAMATH_CALUDE_expression_evaluation_l1348_134881

theorem expression_evaluation :
  (3^1010 + 4^1012)^2 - (3^1010 - 4^1012)^2 = 10^2630 * 10^1012 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1348_134881


namespace NUMINAMATH_CALUDE_latus_rectum_for_parabola_l1348_134896

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = -1/6 * x^2

/-- The equation of the latus rectum -/
def latus_rectum_equation (y : ℝ) : Prop := y = 3/2

/-- Theorem: The latus rectum equation for the given parabola -/
theorem latus_rectum_for_parabola :
  ∀ x y : ℝ, parabola_equation x y → latus_rectum_equation y :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_for_parabola_l1348_134896


namespace NUMINAMATH_CALUDE_basketball_shot_minimum_l1348_134817

theorem basketball_shot_minimum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < 1) (hbb : b < 1) 
  (h_expected : 3 * a + 2 * b = 2) : 
  (2 / a + 1 / (3 * b)) ≥ 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_basketball_shot_minimum_l1348_134817


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1348_134809

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_second_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 3 = 2) : 
  a 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1348_134809


namespace NUMINAMATH_CALUDE_min_side_length_triangle_l1348_134879

theorem min_side_length_triangle (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (h1 h2 h3 : ℝ), h1 > 0 ∧ h2 > 0 ∧ h3 > 0 ∧
    h1 * a = h2 * b ∧ h2 * b = h3 * c ∧
    h1 = 3 ∧ h2 = 4 ∧ h3 = 5) →
  min a (min b c) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_triangle_l1348_134879


namespace NUMINAMATH_CALUDE_markers_leftover_l1348_134890

theorem markers_leftover (total_markers : ℕ) (num_packages : ℕ) (h1 : total_markers = 154) (h2 : num_packages = 13) :
  total_markers % num_packages = 11 := by
sorry

end NUMINAMATH_CALUDE_markers_leftover_l1348_134890


namespace NUMINAMATH_CALUDE_twenty_four_game_solution_l1348_134874

theorem twenty_four_game_solution : 
  let a : ℝ := 5
  let b : ℝ := 5
  let c : ℝ := 5
  let d : ℝ := 1
  (a - d / b) * c = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_game_solution_l1348_134874


namespace NUMINAMATH_CALUDE_even_odd_product_sum_zero_l1348_134871

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

/-- For even function f and odd function g, f(-x)g(-x) + f(x)g(x) = 0 for all x ∈ ℝ -/
theorem even_odd_product_sum_zero (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
    ∀ x : ℝ, f (-x) * g (-x) + f x * g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_product_sum_zero_l1348_134871


namespace NUMINAMATH_CALUDE_line_equation_l1348_134869

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- The main theorem
theorem line_equation (l : Line2D) :
  passesThrough l ⟨1, 4⟩ ∧ hasEqualIntercepts l →
  (l.a = 4 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1348_134869


namespace NUMINAMATH_CALUDE_greatest_BAABC_div_11_l1348_134834

def is_valid_BAABC (n : ℕ) : Prop :=
  ∃ (A B C : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    n = 10000 * B + 1000 * A + 100 * A + 10 * B + C

theorem greatest_BAABC_div_11 :
  ∀ n : ℕ,
    is_valid_BAABC n →
    n ≤ 96619 ∧
    is_valid_BAABC 96619 ∧
    96619 % 11 = 0 ∧
    (n % 11 = 0 → n ≤ 96619) :=
by sorry

end NUMINAMATH_CALUDE_greatest_BAABC_div_11_l1348_134834


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l1348_134801

-- Define the quadratic polynomial
def q (x : ℚ) : ℚ := (6/5) * x^2 - (18/5) * x - (108/5)

-- Theorem stating the conditions
theorem quadratic_polynomial_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -24 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l1348_134801


namespace NUMINAMATH_CALUDE_fraction_sequence_2012th_term_l1348_134875

/-- Represents the sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → ℚ :=
  sorry

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem fraction_sequence_2012th_term :
  ∃ (n : ℕ), 
    triangle_number n ≤ 2012 ∧ 
    triangle_number (n + 1) > 2012 ∧
    63 * 64 / 2 = 2016 ∧
    fraction_sequence 2012 = 5 / 59 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_sequence_2012th_term_l1348_134875


namespace NUMINAMATH_CALUDE_average_playing_time_l1348_134856

/-- Calculates the average playing time given the hours played on three days,
    where the third day is 3 hours more than each of the first two days. -/
theorem average_playing_time (hours_day1 hours_day2 : ℕ) 
    (h1 : hours_day1 = hours_day2)
    (h2 : hours_day1 > 0) : 
  (hours_day1 + hours_day2 + (hours_day1 + 3)) / 3 = hours_day1 + 1 :=
by sorry

#check average_playing_time

end NUMINAMATH_CALUDE_average_playing_time_l1348_134856


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1348_134882

/-- The number of ways to place distinguishable balls into indistinguishable boxes -/
def place_balls_in_boxes (n_balls : ℕ) (n_boxes : ℕ) : ℕ :=
  ((n_boxes ^ n_balls - n_boxes.choose 1 * (n_boxes - 1) ^ n_balls + n_boxes.choose 2) / n_boxes.factorial)

/-- Theorem: There are 25 ways to place 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : place_balls_in_boxes 5 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1348_134882


namespace NUMINAMATH_CALUDE_company_capital_expenditure_l1348_134846

theorem company_capital_expenditure (C : ℝ) (C_pos : C > 0) :
  let raw_material := (1 / 4 : ℝ) * C
  let remaining_after_raw := C - raw_material
  let machinery := (1 / 10 : ℝ) * remaining_after_raw
  let capital_left := C - raw_material - machinery
  capital_left = (27 / 40 : ℝ) * C :=
by sorry

end NUMINAMATH_CALUDE_company_capital_expenditure_l1348_134846


namespace NUMINAMATH_CALUDE_linear_function_translation_l1348_134825

/-- A linear function passing through a specific point -/
def passes_through (b : ℝ) : Prop :=
  3 = 2 + b

/-- The correct value of b for the translated line -/
def correct_b : ℝ := 1

/-- Theorem stating that the linear function passes through (2, 3) iff b = 1 -/
theorem linear_function_translation :
  passes_through correct_b ∧ 
  (∀ b : ℝ, passes_through b → b = correct_b) :=
sorry

end NUMINAMATH_CALUDE_linear_function_translation_l1348_134825


namespace NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l1348_134829

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- A number m is "good" if there exists a positive integer n such that m = n / d(n) -/
def is_good (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ m = n / num_divisors n

theorem good_numbers_up_to_17_and_18_not_good :
  (∀ m : ℕ, m > 0 ∧ m ≤ 17 → is_good m) ∧ ¬ is_good 18 := by
  sorry

end NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l1348_134829


namespace NUMINAMATH_CALUDE_valid_diagonals_150_sided_polygon_l1348_134851

/-- The number of sides in the polygon -/
def n : ℕ := 150

/-- The total number of diagonals in an n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals to be excluded (those connecting vertices whose indices differ by a multiple of 4) -/
def excluded_diagonals (n : ℕ) : ℕ := n * (n / 4)

/-- The number of valid diagonals in the polygon -/
def valid_diagonals (n : ℕ) : ℕ := total_diagonals n - excluded_diagonals n

theorem valid_diagonals_150_sided_polygon :
  valid_diagonals n = 5400 := by
  sorry


end NUMINAMATH_CALUDE_valid_diagonals_150_sided_polygon_l1348_134851


namespace NUMINAMATH_CALUDE_jeans_pricing_markup_l1348_134804

theorem jeans_pricing_markup (manufacturing_cost : ℝ) (customer_price : ℝ) (retailer_price : ℝ)
  (h1 : customer_price = manufacturing_cost * 1.54)
  (h2 : customer_price = retailer_price * 1.1) :
  (retailer_price - manufacturing_cost) / manufacturing_cost * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_jeans_pricing_markup_l1348_134804


namespace NUMINAMATH_CALUDE_trajectory_equation_l1348_134891

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being tangent internally to C₁ and tangent to C₂
def is_tangent_to_circles (x y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (∀ p q : ℝ, C₁ p q → (x - p)^2 + (y - q)^2 = (r - 1)^2) ∧
    (∀ p q : ℝ, C₂ p q → (x - p)^2 + (y - q)^2 = (r + 9)^2)

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, is_tangent_to_circles x y ↔ x^2/16 + y^2/7 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1348_134891


namespace NUMINAMATH_CALUDE_triangle_problem_l1348_134852

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  (b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2) →
  (a + c = Real.sqrt 10) →
  (b = 2) →
  -- Conclusions
  (B = π / 3) ∧
  (1/2 * a * c * Real.sin B = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1348_134852


namespace NUMINAMATH_CALUDE_plot_length_is_80_l1348_134800

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ
  h_length : length = breadth + length_breadth_difference
  h_perimeter : 2 * (length + breadth) = total_fencing_cost / fencing_cost_per_meter

/-- The length of the rectangular plot is 80 meters given the specified conditions. -/
theorem plot_length_is_80 (plot : RectangularPlot)
  (h_length_diff : plot.length_breadth_difference = 60)
  (h_fencing_cost : plot.fencing_cost_per_meter = 26.5)
  (h_total_cost : plot.total_fencing_cost = 5300) :
  plot.length = 80 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_80_l1348_134800


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l1348_134898

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 1250]) :
  5^9000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l1348_134898


namespace NUMINAMATH_CALUDE_deceased_member_income_l1348_134858

/-- Calculates the income of a deceased family member given the initial and final family situations. -/
theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average_income : ℚ)
  (final_members : ℕ)
  (final_average_income : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = 3)
  (h3 : initial_average_income = 735)
  (h4 : final_average_income = 650) :
  (initial_members : ℚ) * initial_average_income - (final_members : ℚ) * final_average_income = 990 :=
by sorry

end NUMINAMATH_CALUDE_deceased_member_income_l1348_134858


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1348_134868

/-- 
A circle in the xy-plane can be represented by an equation of the form
(x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius.
This theorem proves that for the equation x^2 + 14x + y^2 + 8y - k = 0 to
represent a circle of radius 8, k must equal 1.
-/
theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 64) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1348_134868


namespace NUMINAMATH_CALUDE_power_function_through_point_l1348_134827

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_through_point :
  ∀ f : ℝ → ℝ, isPowerFunction f → f 2 = Real.sqrt 2 →
  ∀ x : ℝ, x ≥ 0 → f x = Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1348_134827


namespace NUMINAMATH_CALUDE_emma_chocolates_l1348_134806

theorem emma_chocolates (emma liam : ℕ) : 
  emma = liam + 10 →
  liam = emma / 3 →
  emma = 15 := by
sorry

end NUMINAMATH_CALUDE_emma_chocolates_l1348_134806


namespace NUMINAMATH_CALUDE_smallest_equal_packages_l1348_134811

theorem smallest_equal_packages (n m : ℕ) : 
  (∀ k l : ℕ, k > 0 ∧ l > 0 ∧ 9 * k = 12 * l → n ≤ k) ∧ 
  (∃ m : ℕ, m > 0 ∧ 9 * n = 12 * m) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_packages_l1348_134811


namespace NUMINAMATH_CALUDE_subset_of_complement_iff_a_in_range_l1348_134845

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M (a : ℝ) : Set ℝ := {x | 3 * a - 1 < x ∧ x < 2 * a}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem subset_of_complement_iff_a_in_range (a : ℝ) :
  N ⊆ (U \ M a) ↔ a ≤ -1/2 ∨ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_subset_of_complement_iff_a_in_range_l1348_134845


namespace NUMINAMATH_CALUDE_longest_common_length_l1348_134841

theorem longest_common_length (wood_lengths : List Nat) : 
  wood_lengths = [90, 72, 120, 150, 108] → 
  Nat.gcd 90 (Nat.gcd 72 (Nat.gcd 120 (Nat.gcd 150 108))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_longest_common_length_l1348_134841


namespace NUMINAMATH_CALUDE_michaels_currency_problem_l1348_134865

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Represents the problem of Michael's currency exchange and spending -/
theorem michaels_currency_problem :
  ∃ (d : ℕ),
    (5 / 8 : ℚ) * d - 75 = d ∧
    d = 200 ∧
    sum_of_digits d = 2 := by sorry

end NUMINAMATH_CALUDE_michaels_currency_problem_l1348_134865


namespace NUMINAMATH_CALUDE_sine_function_midline_l1348_134813

theorem sine_function_midline (A B C D : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h4 : D > 0) :
  (∀ x, 1 ≤ A * Real.sin (B * x + C) + D ∧ A * Real.sin (B * x + C) + D ≤ 5) → D = 3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_midline_l1348_134813


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1348_134889

theorem circle_area_ratio (d : ℝ) (h : d > 0) : 
  (π * ((7 * d) / 2)^2) / (π * (d / 2)^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1348_134889


namespace NUMINAMATH_CALUDE_customer_outreach_time_calculation_l1348_134807

/-- Represents the daily work schedule of a social media account manager --/
structure WorkSchedule where
  total_time : ℝ
  marketing_time : ℝ
  customer_outreach_time : ℝ
  advertisement_time : ℝ

/-- Theorem stating the correct time spent on customer outreach posts --/
theorem customer_outreach_time_calculation (schedule : WorkSchedule) 
  (h1 : schedule.total_time = 8)
  (h2 : schedule.marketing_time = 2)
  (h3 : schedule.advertisement_time = schedule.customer_outreach_time / 2)
  (h4 : schedule.total_time = schedule.marketing_time + schedule.customer_outreach_time + schedule.advertisement_time) :
  schedule.customer_outreach_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_customer_outreach_time_calculation_l1348_134807


namespace NUMINAMATH_CALUDE_inequality_proof_l1348_134854

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_condition : a * b * c * (a + b + c) = a * b + b * c + c * a) :
  5 * (a + b + c) ≥ 7 + 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1348_134854


namespace NUMINAMATH_CALUDE_james_payment_is_18_l1348_134887

/-- James's meal cost -/
def james_meal : ℚ := 16

/-- Friend's meal cost -/
def friend_meal : ℚ := 14

/-- Tip percentage -/
def tip_percent : ℚ := 20 / 100

/-- Calculate James's payment given the meal costs and tip percentage -/
def calculate_james_payment (james_meal friend_meal tip_percent : ℚ) : ℚ :=
  let total_before_tip := james_meal + friend_meal
  let tip := tip_percent * total_before_tip
  let total_with_tip := total_before_tip + tip
  total_with_tip / 2

/-- Theorem stating that James's payment is $18 -/
theorem james_payment_is_18 :
  calculate_james_payment james_meal friend_meal tip_percent = 18 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_is_18_l1348_134887
