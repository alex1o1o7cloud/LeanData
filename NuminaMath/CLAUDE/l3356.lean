import Mathlib

namespace NUMINAMATH_CALUDE_division_with_special_remainder_l3356_335635

theorem division_with_special_remainder :
  ∃! (n : ℕ), n > 0 ∧ 
    ∃ (k m : ℕ), 
      180 = n * k + m ∧ 
      4 * m = k ∧ 
      m < n ∧ 
      n = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_with_special_remainder_l3356_335635


namespace NUMINAMATH_CALUDE_solve_for_y_l3356_335684

theorem solve_for_y (x y : ℝ) (h1 : x^2 - 3*x + 7 = y + 2) (h2 : x = -5) : y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3356_335684


namespace NUMINAMATH_CALUDE_power_of_power_l3356_335689

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3356_335689


namespace NUMINAMATH_CALUDE_f_monotone_increasing_when_a_eq_1_f_range_of_a_l3356_335627

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 3 * |a - 1| * x^2 + 2 * a * x - a

-- Theorem 1: Monotonicity when a = 1
theorem f_monotone_increasing_when_a_eq_1 :
  ∀ x y : ℝ, x < y → (f 1 x) < (f 1 y) :=
sorry

-- Theorem 2: Range of a
theorem f_range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 0 1, |f a x| ≤ f a 1} = Set.Ici (-3/4) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_when_a_eq_1_f_range_of_a_l3356_335627


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3356_335642

theorem boys_to_girls_ratio (B G : ℕ) (h_positive : B > 0 ∧ G > 0) : 
  (1/3 : ℚ) * B + (2/3 : ℚ) * G = (192/360 : ℚ) * (B + G) → 
  (B : ℚ) / G = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3356_335642


namespace NUMINAMATH_CALUDE_quadratic_sum_of_squares_l3356_335639

theorem quadratic_sum_of_squares (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) →
  (∃! y : ℝ, y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0) →
  (∃! z : ℝ, z^2 + c*z + a = 0 ∧ z^2 + a*z + b = 0) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_squares_l3356_335639


namespace NUMINAMATH_CALUDE_only_prop2_is_true_l3356_335673

-- Define the propositions
def prop1 : Prop := ∀ x : ℝ, (∃ y : ℝ, y^2 + 1 > 3*y) ↔ ¬(x^2 + 1 < 3*x)

def prop2 : Prop := ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def prop3 : Prop := ∃ a : ℝ, (a > 2 → a > 5) ∧ ¬(a > 5 → a > 2)

def prop4 : Prop := ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → (x*y ≠ 0)

-- Theorem stating that only prop2 is true
theorem only_prop2_is_true : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by
  sorry

end NUMINAMATH_CALUDE_only_prop2_is_true_l3356_335673


namespace NUMINAMATH_CALUDE_combination_equality_l3356_335611

theorem combination_equality (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) → (x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l3356_335611


namespace NUMINAMATH_CALUDE_event_B_more_likely_l3356_335629

/-- Represents the number of sides on a fair die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 3

/-- Probability of event B: three different numbers appear in three throws -/
def probB : ℚ := 5 / 9

/-- Probability of event A: at least one number appears at least twice in three throws -/
def probA : ℚ := 4 / 9

/-- Theorem stating that event B is more likely than event A -/
theorem event_B_more_likely : probB > probA := by sorry

end NUMINAMATH_CALUDE_event_B_more_likely_l3356_335629


namespace NUMINAMATH_CALUDE_A_has_min_l3356_335672

/-- The function f_{a,b} from R^2 to R^2 -/
def f (a b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (a - b * y - x^2, x)

/-- The n-th iteration of f_{a,b} -/
def f_iter (a b : ℝ) : ℕ → (ℝ × ℝ → ℝ × ℝ)
  | 0 => id
  | n + 1 => f a b ∘ f_iter a b n

/-- The set of periodic points of f_{a,b} -/
def per (a b : ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ n : ℕ+, f_iter a b n P = P}

/-- The set A_b -/
def A (b : ℝ) : Set ℝ :=
  {a | per a b ≠ ∅}

/-- The theorem stating that A_b has a minimum equal to -(b+1)^2/4 -/
theorem A_has_min (b : ℝ) : 
  ∃ min : ℝ, IsGLB (A b) min ∧ min = -(b + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_A_has_min_l3356_335672


namespace NUMINAMATH_CALUDE_abs_value_problem_l3356_335643

theorem abs_value_problem (x p : ℝ) : 
  |x - 3| = p ∧ x > 3 → x - p = 3 := by
sorry

end NUMINAMATH_CALUDE_abs_value_problem_l3356_335643


namespace NUMINAMATH_CALUDE_discount_difference_l3356_335618

/-- Proves that the difference between the claimed and actual discount is 3.75% -/
theorem discount_difference : 
  let first_discount : ℝ := 0.25
  let second_discount : ℝ := 0.15
  let claimed_total_discount : ℝ := 0.40
  let actual_total_discount : ℝ := 1 - (1 - first_discount) * (1 - second_discount)
  claimed_total_discount - actual_total_discount = 0.0375 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3356_335618


namespace NUMINAMATH_CALUDE_orange_calories_l3356_335696

/-- Proves that the number of calories per orange is 80 given the problem conditions -/
theorem orange_calories (orange_cost : ℚ) (initial_amount : ℚ) (required_calories : ℕ) (remaining_amount : ℚ) :
  orange_cost = 6/5 ∧ 
  initial_amount = 10 ∧ 
  required_calories = 400 ∧ 
  remaining_amount = 4 →
  (initial_amount - remaining_amount) / orange_cost * required_calories / ((initial_amount - remaining_amount) / orange_cost) = 80 := by
sorry

end NUMINAMATH_CALUDE_orange_calories_l3356_335696


namespace NUMINAMATH_CALUDE_cubic_minus_four_xy_squared_factorization_l3356_335648

theorem cubic_minus_four_xy_squared_factorization (x y : ℝ) :
  x^3 - 4*x*y^2 = x*(x+2*y)*(x-2*y) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_four_xy_squared_factorization_l3356_335648


namespace NUMINAMATH_CALUDE_equation_has_four_real_solutions_l3356_335656

theorem equation_has_four_real_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, x^2 + 1/x^2 = 2006 + 1/2006) ∧ s.card = 4 ∧
  (∀ y : ℝ, y^2 + 1/y^2 = 2006 + 1/2006 → y ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_four_real_solutions_l3356_335656


namespace NUMINAMATH_CALUDE_number_difference_proof_l3356_335649

theorem number_difference_proof (L S : ℕ) (h1 : L = 1636) (h2 : L = 6 * S + 10) : 
  L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l3356_335649


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3356_335631

theorem arithmetic_sequence_terms (a d : ℝ) (n : ℕ) : 
  (n / 2 : ℝ) * (2 * a + (n - 1 : ℝ) * 2 * d) = 24 →
  (n / 2 : ℝ) * (2 * (a + d) + (n - 1 : ℝ) * 2 * d) = 30 →
  a + ((2 * n - 1 : ℝ) * d) - a = 10.5 →
  2 * n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3356_335631


namespace NUMINAMATH_CALUDE_chocolate_problem_l3356_335617

theorem chocolate_problem :
  ∃ n : ℕ, n = 151 ∧ 
  (∀ m : ℕ, m ≥ 150 ∧ m % 17 = 15 → m ≥ n) ∧
  n ≥ 150 ∧ n % 17 = 15 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_problem_l3356_335617


namespace NUMINAMATH_CALUDE_quadratic_root_divisibility_l3356_335674

theorem quadratic_root_divisibility (a b c n : ℤ) 
  (h : a * n^2 + b * n + c = 0) : 
  c ∣ n := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_divisibility_l3356_335674


namespace NUMINAMATH_CALUDE_square_cutout_l3356_335691

theorem square_cutout (N M : ℕ) (h : N^2 - M^2 = 79) : M = N - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_cutout_l3356_335691


namespace NUMINAMATH_CALUDE_pure_imaginary_quadratic_l3356_335612

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The theorem statement -/
theorem pure_imaginary_quadratic (m : ℝ) :
  IsPureImaginary (Complex.mk (m^2 + m - 2) (m^2 - 1)) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_quadratic_l3356_335612


namespace NUMINAMATH_CALUDE_inequality1_solution_system_solution_integer_system_solution_l3356_335694

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 3 * x - 5 > 5 * x + 3
def inequality2 (x : ℝ) : Prop := x - 1 ≥ 1 - x
def inequality3 (x : ℝ) : Prop := x + 8 > 4 * x - 1

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | x < -4}
def solution_set2 : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

-- Define the integer solutions
def integer_solutions : Set ℤ := {1, 2}

-- Theorem statements
theorem inequality1_solution : 
  {x : ℝ | inequality1 x} = solution_set1 :=
sorry

theorem system_solution : 
  {x : ℝ | inequality2 x ∧ inequality3 x} = solution_set2 :=
sorry

theorem integer_system_solution : 
  {x : ℤ | (x : ℝ) ∈ solution_set2} = integer_solutions :=
sorry

end NUMINAMATH_CALUDE_inequality1_solution_system_solution_integer_system_solution_l3356_335694


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l3356_335622

theorem difference_of_squares_division : (204^2 - 196^2) / 16 = 200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l3356_335622


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l3356_335676

/-- A line in 2D space, represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space, represented by the equation (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  (c.h * l.a + c.k * l.b + l.c)^2 = (l.a^2 + l.b^2) * c.r^2

/-- The main theorem -/
theorem tangent_lines_to_circle (l : Line) (c : Circle) :
  (l.a = 2 ∧ l.b = -1 ∧ c.h = 0 ∧ c.k = 0 ∧ c.r^2 = 5) →
  (∃ l1 l2 : Line,
    (are_parallel l l1 ∧ is_tangent l1 c) ∧
    (are_parallel l l2 ∧ is_tangent l2 c) ∧
    (l1.c = 5 ∨ l1.c = -5) ∧
    (l2.c = 5 ∨ l2.c = -5) ∧
    (l1.c + l2.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l3356_335676


namespace NUMINAMATH_CALUDE_tan_sum_problem_l3356_335638

theorem tan_sum_problem (α β : ℝ) 
  (h1 : Real.tan (α + 2 * β) = 2) 
  (h2 : Real.tan β = -3) : 
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_problem_l3356_335638


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3356_335690

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := Finset.sum (Finset.range (bounces + 1)) (fun i => initialHeight * reboundFactor^i)
  let ascendDistances := Finset.sum (Finset.range bounces) (fun i => initialHeight * reboundFactor^(i+1))
  descendDistances + ascendDistances

/-- Theorem: The total distance traveled by a ball dropped from 25 meters,
    rebounding to 2/3 of its previous height for four bounces, is 1900/27 meters -/
theorem ball_bounce_distance :
  totalDistance 25 (2/3) 4 = 1900/27 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3356_335690


namespace NUMINAMATH_CALUDE_andrea_reach_time_l3356_335668

/-- The time it takes Andrea to reach Lauren's stop location -/
def time_to_reach (initial_distance : ℝ) (speed_ratio : ℝ) (distance_decrease_rate : ℝ) (lauren_stop_time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time it takes Andrea to reach Lauren's stop location -/
theorem andrea_reach_time :
  let initial_distance : ℝ := 30
  let speed_ratio : ℝ := 2
  let distance_decrease_rate : ℝ := 90
  let lauren_stop_time : ℝ := 1/6 -- 10 minutes in hours
  time_to_reach initial_distance speed_ratio distance_decrease_rate lauren_stop_time = 25/60 := by
  sorry

end NUMINAMATH_CALUDE_andrea_reach_time_l3356_335668


namespace NUMINAMATH_CALUDE_team_selection_ways_l3356_335636

def boys : ℕ := 10
def girls : ℕ := 10
def team_size : ℕ := 8
def boys_in_team : ℕ := team_size / 2
def girls_in_team : ℕ := team_size / 2

theorem team_selection_ways : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 44100 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_l3356_335636


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3356_335659

theorem square_ratio_side_length_sum (s1 s2 : ℝ) (h : s1^2 / s2^2 = 32 / 63) :
  ∃ (a b c : ℕ), (s1 / s2 = a * Real.sqrt b / c) ∧ (a + b + c = 39) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l3356_335659


namespace NUMINAMATH_CALUDE_unique_root_is_half_l3356_335603

/-- Given real numbers a, b, c forming an arithmetic sequence with a ≥ b ≥ c ≥ 0,
    and the quadratic equation ax^2 - bx + c = 0 having exactly one root,
    prove that this root is 1/2. -/
theorem unique_root_is_half (a b c : ℝ) 
    (arith_seq : ∃ (d : ℝ), b = a - d ∧ c = a - 2*d)
    (ordered : a ≥ b ∧ b ≥ c ∧ c ≥ 0)
    (one_root : ∃! x, a*x^2 - b*x + c = 0) :
    ∃ x, a*x^2 - b*x + c = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_is_half_l3356_335603


namespace NUMINAMATH_CALUDE_tallest_player_height_l3356_335646

/-- Given a basketball team where the tallest player is 9.5 inches taller than
    the shortest player, and the shortest player is 68.25 inches tall,
    prove that the tallest player is 77.75 inches tall. -/
theorem tallest_player_height :
  let shortest_player_height : ℝ := 68.25
  let height_difference : ℝ := 9.5
  let tallest_player_height : ℝ := shortest_player_height + height_difference
  tallest_player_height = 77.75 := by sorry

end NUMINAMATH_CALUDE_tallest_player_height_l3356_335646


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3356_335647

theorem regular_polygon_sides : ∃ (n : ℕ), n > 2 ∧ (2 * n - n * (n - 3) / 2 = 0) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3356_335647


namespace NUMINAMATH_CALUDE_simple_interest_principal_l3356_335609

/-- Simple interest calculation --/
theorem simple_interest_principal (interest_rate : ℚ) (time_months : ℕ) (interest_earned : ℕ) (principal : ℕ) : 
  interest_rate = 50 / 3 → 
  time_months = 9 → 
  interest_earned = 8625 →
  principal = 69000 →
  interest_earned * 1200 = principal * interest_rate * time_months := by
  sorry

#check simple_interest_principal

end NUMINAMATH_CALUDE_simple_interest_principal_l3356_335609


namespace NUMINAMATH_CALUDE_kaleb_games_proof_l3356_335652

theorem kaleb_games_proof (sold : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h1 : sold = 46)
  (h2 : boxes = 6)
  (h3 : games_per_box = 5) :
  sold + boxes * games_per_box = 76 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_games_proof_l3356_335652


namespace NUMINAMATH_CALUDE_square_area_ratio_l3356_335651

theorem square_area_ratio (y : ℝ) (hy : y > 0) :
  (y^2) / ((3*y)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3356_335651


namespace NUMINAMATH_CALUDE_starters_count_theorem_l3356_335637

def number_of_players : ℕ := 15
def number_of_starters : ℕ := 5

-- Define a function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (choose : ℕ) : ℕ := Nat.choose total choose

-- Define a function to calculate the number of ways to choose starters excluding both twins
def choose_starters_excluding_twins (total : ℕ) (choose : ℕ) : ℕ :=
  choose_starters total choose - choose_starters (total - 2) (choose - 2)

theorem starters_count_theorem : 
  choose_starters_excluding_twins number_of_players number_of_starters = 2717 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_theorem_l3356_335637


namespace NUMINAMATH_CALUDE_jeans_cost_thirty_l3356_335625

/-- The price of socks in dollars -/
def socks_price : ℕ := 5

/-- The price difference between t-shirt and socks in dollars -/
def tshirt_socks_diff : ℕ := 10

/-- The price of a t-shirt in dollars -/
def tshirt_price : ℕ := socks_price + tshirt_socks_diff

/-- The price of jeans in dollars -/
def jeans_price : ℕ := 2 * tshirt_price

theorem jeans_cost_thirty : jeans_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_thirty_l3356_335625


namespace NUMINAMATH_CALUDE_hyperbola_equation_special_case_l3356_335687

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The distance from the foci to the asymptotes of a hyperbola -/
def foci_to_asymptote_distance (h : Hyperbola a b) : ℝ :=
  b

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola a b) : ℝ :=
  2 * a

/-- Theorem: If the distance from the foci to the asymptotes equals the length of the real axis
    and the point (2,2) lies on the hyperbola, then the equation of the hyperbola is x^2/3 - y^2/12 = 1 -/
theorem hyperbola_equation_special_case (h : Hyperbola a b) :
  foci_to_asymptote_distance h = real_axis_length h →
  hyperbola_equation h 2 2 →
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 / 12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_special_case_l3356_335687


namespace NUMINAMATH_CALUDE_rectangle_polygon_perimeter_l3356_335607

theorem rectangle_polygon_perimeter : 
  let n : ℕ := 20
  let rectangle_dimensions : ℕ → ℕ × ℕ := λ i => (i, i + 1)
  let perimeter : ℕ := 2 * (List.range (n + 1)).sum
  perimeter = 462 := by sorry

end NUMINAMATH_CALUDE_rectangle_polygon_perimeter_l3356_335607


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3356_335601

theorem cube_sum_theorem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : 
  x^3 + y^3 = 85/2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3356_335601


namespace NUMINAMATH_CALUDE_same_color_probability_is_two_twentyfifths_l3356_335692

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (purple : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (silver : ℕ)
  (total : ℕ)
  (h_total : purple + green + blue + silver = total)

/-- The probability of getting the same color on all three dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.purple : ℚ) ^ 3 / d.total ^ 3 +
  (d.green : ℚ) ^ 3 / d.total ^ 3 +
  (d.blue : ℚ) ^ 3 / d.total ^ 3 +
  (d.silver : ℚ) ^ 3 / d.total ^ 3

/-- The specific die configuration in the problem -/
def problem_die : ColoredDie :=
  { purple := 6
  , green := 8
  , blue := 10
  , silver := 6
  , total := 30
  , h_total := by simp }

/-- Theorem stating the probability of getting the same color on all three dice -/
theorem same_color_probability_is_two_twentyfifths :
  same_color_probability problem_die = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_two_twentyfifths_l3356_335692


namespace NUMINAMATH_CALUDE_celine_change_l3356_335671

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine buys -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine buys -/
def smartphones_bought : ℕ := 4

/-- The amount of money Celine has in dollars -/
def money_available : ℕ := 3000

/-- The change Celine receives after her purchase -/
def change : ℕ := money_available - (laptop_price * laptops_bought + smartphone_price * smartphones_bought)

theorem celine_change : change = 200 := by
  sorry

end NUMINAMATH_CALUDE_celine_change_l3356_335671


namespace NUMINAMATH_CALUDE_root_sum_equals_square_sum_l3356_335653

theorem root_sum_equals_square_sum (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*a*(x₁-1) - 1 = 0 ∧ 
                x₂^2 - 2*a*(x₂-1) - 1 = 0 ∧ 
                x₁ + x₂ = x₁^2 + x₂^2) ↔ 
  (a = 1 ∨ a = 1/2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_equals_square_sum_l3356_335653


namespace NUMINAMATH_CALUDE_two_stage_discount_l3356_335681

/-- Calculate the actual discount and difference from claimed discount in a two-stage discount scenario -/
theorem two_stage_discount (initial_discount additional_discount claimed_discount : ℝ) :
  initial_discount = 0.4 →
  additional_discount = 0.25 →
  claimed_discount = 0.6 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_additional
  actual_discount = 0.55 ∧ claimed_discount - actual_discount = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_two_stage_discount_l3356_335681


namespace NUMINAMATH_CALUDE_sylvie_turtle_weight_l3356_335655

/-- The weight of turtles Sylvie has, given the feeding conditions -/
theorem sylvie_turtle_weight :
  let food_per_half_pound : ℚ := 1 -- 1 ounce of food per 1/2 pound of body weight
  let ounces_per_jar : ℚ := 15 -- Each jar contains 15 ounces
  let cost_per_jar : ℚ := 2 -- Each jar costs $2
  let total_cost : ℚ := 8 -- It costs $8 to feed the turtles
  
  (total_cost / cost_per_jar) * ounces_per_jar / food_per_half_pound / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sylvie_turtle_weight_l3356_335655


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3356_335686

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x ∨ y = -(Real.sqrt 5 / 2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(√5/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3356_335686


namespace NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l3356_335634

-- Define what it means for an angle to be in the first quadrant
def is_first_quadrant (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the condition we're interested in
def condition (α : Real) : Prop := Real.sin α * Real.cos α > 0

-- Theorem statement
theorem first_quadrant_sufficient_not_necessary :
  (∀ α : Real, is_first_quadrant α → condition α) ∧
  (∃ α : Real, condition α ∧ ¬is_first_quadrant α) := by sorry

end NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l3356_335634


namespace NUMINAMATH_CALUDE_problem_statement_l3356_335626

theorem problem_statement : (-1 : ℤ) ^ 53 + 2 ^ (4^3 + 3^2 - 7^2) = 16777215 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3356_335626


namespace NUMINAMATH_CALUDE_quadrilateral_theorem_l3356_335660

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  P : Point
  Q : Point
  R : Point
  S : Point

def area (q : Quadrilateral) : ℝ := sorry

def diagonalsPerpendicular (q : Quadrilateral) : Prop := sorry

theorem quadrilateral_theorem (a b : ℤ) (h1 : a > b) (h2 : b > 0) :
  let q := Quadrilateral.mk
    (Point.mk a b)
    (Point.mk (b + 2) a)
    (Point.mk (-a) (-b))
    (Point.mk (-(b + 2)) (-a))
  area q = 18 ∧ diagonalsPerpendicular q → a + b = 9 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_theorem_l3356_335660


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3356_335602

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 - 8 = (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 8*x^2 + 16*x + 32) + 56 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3356_335602


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l3356_335666

/-- The number of dogwood trees planted tomorrow to reach the desired total -/
def trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_trees + planted_today)

/-- Theorem stating the number of trees planted tomorrow -/
theorem dogwood_trees_planted_tomorrow :
  trees_planted_tomorrow 39 41 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l3356_335666


namespace NUMINAMATH_CALUDE_similar_point_coordinates_l3356_335616

/-- Given a point A and a similarity ratio, find the coordinates of the similar point A' --/
theorem similar_point_coordinates (A : ℝ × ℝ) (ratio : ℝ) :
  A = (2, 3) ∧ ratio = 2 →
  let A' := (ratio * A.1, ratio * A.2)
  A' = (4, 6) ∨ A' = (-4, -6) := by
  sorry

end NUMINAMATH_CALUDE_similar_point_coordinates_l3356_335616


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l3356_335677

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l3356_335677


namespace NUMINAMATH_CALUDE_amys_tomato_soup_cans_l3356_335678

/-- Amy's soup purchase problem -/
theorem amys_tomato_soup_cans (total_soups chicken_soups tomato_soups : ℕ) : 
  total_soups = 9 →
  chicken_soups = 6 →
  total_soups = chicken_soups + tomato_soups →
  tomato_soups = 3 := by
sorry

end NUMINAMATH_CALUDE_amys_tomato_soup_cans_l3356_335678


namespace NUMINAMATH_CALUDE_min_value_and_valid_a4_l3356_335645

def is_valid_sequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i j : Fin 10, i < j → a i < a j

def lcm_of_sequence (a : Fin 10 → ℕ) : ℕ :=
  Finset.lcm (Finset.range 10) (fun i => a i)

theorem min_value_and_valid_a4 (a : Fin 10 → ℕ) (h : is_valid_sequence a) :
  (∀ b : Fin 10 → ℕ, is_valid_sequence b → lcm_of_sequence a / a 3 ≤ lcm_of_sequence b / b 3) ∧
  (lcm_of_sequence a / a 0 = lcm_of_sequence a / a 3) →
  (lcm_of_sequence a / a 3 = 630) ∧
  (a 3 = 360 ∨ a 3 = 720 ∨ a 3 = 1080) ∧
  (1 ≤ a 3) ∧ (a 3 ≤ 1300) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_valid_a4_l3356_335645


namespace NUMINAMATH_CALUDE_triangle_angle_value_l3356_335641

theorem triangle_angle_value (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  Real.sqrt 3 * c * Real.sin A = a * Real.cos C →
  C = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l3356_335641


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l3356_335657

/-- The radius of the circle with equation 16x^2 + 32x + 16y^2 - 48y + 68 = 0 is 1 -/
theorem circle_radius_is_one :
  ∃ (h k r : ℝ), r = 1 ∧
  ∀ (x y : ℝ), 16*x^2 + 32*x + 16*y^2 - 48*y + 68 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l3356_335657


namespace NUMINAMATH_CALUDE_solution_set_l3356_335670

def system_solution (x y : ℝ) : Prop :=
  5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8

theorem solution_set : 
  {p : ℝ × ℝ | system_solution p.1 p.2} = {(-1, 2), (11, -7), (-11, 7), (1, -2)} := by
sorry

end NUMINAMATH_CALUDE_solution_set_l3356_335670


namespace NUMINAMATH_CALUDE_cube_vertex_distance_to_plane_l3356_335688

/-- Given a cube with side length 15 and three vertices adjacent to vertex A
    at heights 15, 17, and 18 above a plane, the distance from vertex A to the plane is 28/3 -/
theorem cube_vertex_distance_to_plane :
  ∀ (a b c d : ℝ),
  a^2 + b^2 + c^2 = 1 →
  15 * a + d = 15 →
  15 * b + d = 17 →
  15 * c + d = 18 →
  d = 28 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_vertex_distance_to_plane_l3356_335688


namespace NUMINAMATH_CALUDE_mr_li_age_is_25_l3356_335664

-- Define Xiaofang's age this year
def xiaofang_age : ℕ := 5

-- Define the number of years in the future
def years_in_future : ℕ := 3

-- Define the age difference between Mr. Li and Xiaofang in the future
def future_age_difference : ℕ := 20

-- Define Mr. Li's age this year
def mr_li_age : ℕ := xiaofang_age + future_age_difference

-- Theorem to prove
theorem mr_li_age_is_25 : mr_li_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_mr_li_age_is_25_l3356_335664


namespace NUMINAMATH_CALUDE_perfect_square_characterization_l3356_335658

theorem perfect_square_characterization (A : ℕ+) :
  (∃ (d : ℕ+), A = d ^ 2) ↔
  (∀ (n : ℕ+), ∃ (j : ℕ+), j ≤ n ∧ (n ∣ ((A + j) ^ 2 - A))) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_characterization_l3356_335658


namespace NUMINAMATH_CALUDE_ratio_change_l3356_335683

theorem ratio_change (x y : ℤ) (n : ℤ) : 
  y = 72 → x / y = 1 / 4 → (x + n) / y = 1 / 3 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_l3356_335683


namespace NUMINAMATH_CALUDE_function_periodicity_l3356_335633

/-- A function f: ℝ → ℝ satisfying f(x-1) + f(x+1) = √2 f(x) for all x ∈ ℝ is periodic with period 8. -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1) + f (x + 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 8) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l3356_335633


namespace NUMINAMATH_CALUDE_five_students_arrangement_l3356_335669

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangements_without_adjacent (n : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * 2)

theorem five_students_arrangement :
  arrangements_without_adjacent 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_students_arrangement_l3356_335669


namespace NUMINAMATH_CALUDE_university_packaging_cost_l3356_335685

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the pricing scheme for boxes -/
structure BoxPricing where
  initialPrice : ℝ
  initialQuantity : ℕ
  additionalPrice : ℝ

/-- Calculates the minimum cost for packaging a given number of items -/
def minimumPackagingCost (boxDim : BoxDimensions) (pricing : BoxPricing) (itemCount : ℕ) : ℝ :=
  let initialCost := pricing.initialPrice * pricing.initialQuantity
  let additionalBoxes := max (itemCount - pricing.initialQuantity) 0
  let additionalCost := pricing.additionalPrice * additionalBoxes
  initialCost + additionalCost

/-- Theorem stating the minimum packaging cost for the university's collection -/
theorem university_packaging_cost :
  let boxDim : BoxDimensions := { length := 18, width := 22, height := 15 }
  let pricing : BoxPricing := { initialPrice := 0.60, initialQuantity := 100, additionalPrice := 0.55 }
  let itemCount : ℕ := 127
  minimumPackagingCost boxDim pricing itemCount = 74.85 := by
  sorry


end NUMINAMATH_CALUDE_university_packaging_cost_l3356_335685


namespace NUMINAMATH_CALUDE_modulus_z_l3356_335697

/-- Given complex numbers w and z such that wz = 20 - 15i and |w| = √34, prove that |z| = (25√34) / 34 -/
theorem modulus_z (w z : ℂ) (h1 : w * z = 20 - 15 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (25 * Real.sqrt 34) / 34 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l3356_335697


namespace NUMINAMATH_CALUDE_ratio_equality_l3356_335619

theorem ratio_equality (a b : ℝ) (h : a / b = 4 / 7) : 7 * a = 4 * b := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3356_335619


namespace NUMINAMATH_CALUDE_max_cylinder_lateral_area_l3356_335615

/-- Given a rectangle with perimeter 36, prove that when rotated around one of its edges
    to form a cylinder, the maximum lateral surface area of the cylinder is 81. -/
theorem max_cylinder_lateral_area (l w : ℝ) : 
  (l + w = 18) →  -- Perimeter condition: 2(l + w) = 36, simplified to l + w = 18
  (∃ (h r : ℝ), h = w ∧ 2 * π * r = l ∧ 2 * π * r * h ≤ 81) ∧ 
  (∃ (h r : ℝ), h = w ∧ 2 * π * r = l ∧ 2 * π * r * h = 81) :=
sorry

end NUMINAMATH_CALUDE_max_cylinder_lateral_area_l3356_335615


namespace NUMINAMATH_CALUDE_lily_bought_ten_geese_l3356_335698

/-- The number of geese Lily bought -/
def lily_geese : ℕ := sorry

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := 20

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

theorem lily_bought_ten_geese :
  lily_geese = 10 ∧
  rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70 :=
by sorry

end NUMINAMATH_CALUDE_lily_bought_ten_geese_l3356_335698


namespace NUMINAMATH_CALUDE_equation_solution_l3356_335621

theorem equation_solution :
  ∃ (x y z u : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧
    -1/x + 1/y + 1/z + 1/u = 2 ∧
    x = 1 ∧ y = 2 ∧ z = 3 ∧ u = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3356_335621


namespace NUMINAMATH_CALUDE_episodes_per_day_l3356_335620

/-- Given a TV series with 3 seasons of 20 episodes each, watched over 30 days,
    the number of episodes watched per day is 2. -/
theorem episodes_per_day (seasons : ℕ) (episodes_per_season : ℕ) (total_days : ℕ)
    (h1 : seasons = 3)
    (h2 : episodes_per_season = 20)
    (h3 : total_days = 30) :
    (seasons * episodes_per_season) / total_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_episodes_per_day_l3356_335620


namespace NUMINAMATH_CALUDE_line_intersects_segment_m_range_l3356_335699

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by the equation x + my + m = 0 -/
structure Line where
  m : ℝ

def intersectsSegment (l : Line) (a b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    (1 - t) * a.x + t * b.x + l.m * ((1 - t) * a.y + t * b.y) + l.m = 0

theorem line_intersects_segment_m_range (l : Line) :
  let a : Point := ⟨-1, 1⟩
  let b : Point := ⟨2, -2⟩
  intersectsSegment l a b → 1/2 ≤ l.m ∧ l.m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_line_intersects_segment_m_range_l3356_335699


namespace NUMINAMATH_CALUDE_sum_of_roots_l3356_335644

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 18*c^2 + 27*c - 100 = 0)
  (hd : 9*d^3 - 81*d^2 - 324*d + 3969 = 0) : 
  c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3356_335644


namespace NUMINAMATH_CALUDE_boris_fudge_amount_l3356_335608

-- Define the conversion rate from pounds to ounces
def poundsToOunces (pounds : ℝ) : ℝ := pounds * 16

-- Define the amount of fudge eaten by each person
def tomasFudge : ℝ := 1.5
def katyaFudge : ℝ := 0.5

-- Define the total amount of fudge eaten by all three friends in ounces
def totalFudgeOunces : ℝ := 64

-- Theorem to prove
theorem boris_fudge_amount :
  let borisFudgeOunces := totalFudgeOunces - (poundsToOunces tomasFudge + poundsToOunces katyaFudge)
  borisFudgeOunces / 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_boris_fudge_amount_l3356_335608


namespace NUMINAMATH_CALUDE_stability_promotion_criterion_l3356_335679

/-- Represents a rice variety with its yield statistics -/
structure RiceVariety where
  name : String
  average_yield : ℝ
  variance : ℝ

/-- Determines if a rice variety is more stable than another -/
def is_more_stable (a b : RiceVariety) : Prop :=
  a.variance < b.variance

/-- Determines if a rice variety is suitable for promotion based on stability -/
def suitable_for_promotion (a b : RiceVariety) : Prop :=
  is_more_stable a b

theorem stability_promotion_criterion 
  (a b : RiceVariety) 
  (h1 : a.average_yield = b.average_yield) 
  (h2 : a.variance < b.variance) : 
  suitable_for_promotion a b :=
sorry

end NUMINAMATH_CALUDE_stability_promotion_criterion_l3356_335679


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3356_335600

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)
  product_345 : a 3 * a 4 * a 5 = 3
  product_678 : a 6 * a 7 * a 8 = 24

/-- The theorem statement -/
theorem geometric_sequence_property (seq : GeometricSequence) :
  seq.a 9 * seq.a 10 * seq.a 11 = 192 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3356_335600


namespace NUMINAMATH_CALUDE_power_function_through_point_l3356_335650

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

theorem power_function_through_point :
  ∀ f : ℝ → ℝ, is_power_function f →
  f 2 = (1/4 : ℝ) →
  ∃ a : ℝ, (∀ x : ℝ, f x = x^a) ∧ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3356_335650


namespace NUMINAMATH_CALUDE_always_not_three_l3356_335661

def is_single_digit (n : ℕ) : Prop := n < 10

def statement_I (n : ℕ) : Prop := n = 2
def statement_II (n : ℕ) : Prop := n ≠ 3
def statement_III (n : ℕ) : Prop := n = 5
def statement_IV (n : ℕ) : Prop := Even n

theorem always_not_three (n : ℕ) (h_single_digit : is_single_digit n) 
  (h_three_true : ∃ (a b c : Prop) (ha : a) (hb : b) (hc : c), 
    (a = statement_I n ∨ a = statement_II n ∨ a = statement_III n ∨ a = statement_IV n) ∧
    (b = statement_I n ∨ b = statement_II n ∨ b = statement_III n ∨ b = statement_IV n) ∧
    (c = statement_I n ∨ c = statement_II n ∨ c = statement_III n ∨ c = statement_IV n) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  statement_II n := by
  sorry

end NUMINAMATH_CALUDE_always_not_three_l3356_335661


namespace NUMINAMATH_CALUDE_interest_equality_problem_l3356_335610

theorem interest_equality_problem (total : ℝ) (first_part : ℝ) (second_part : ℝ)
  (h1 : total = 2717)
  (h2 : total = first_part + second_part)
  (h3 : first_part * (3/100) * 8 = second_part * (5/100) * 3) :
  second_part = 2449 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_problem_l3356_335610


namespace NUMINAMATH_CALUDE_business_partnership_timing_l3356_335640

/-- Proves that B joined the business 8 months after A started, given the conditions of the problem -/
theorem business_partnership_timing (a_initial_capital b_capital : ℕ) (x : ℕ) : 
  a_initial_capital = 3500 →
  b_capital = 15750 →
  (a_initial_capital * 12) / (b_capital * (12 - x)) = 2 / 3 →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_business_partnership_timing_l3356_335640


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2015_l3356_335680

/-- The function that returns the last two digits of 8^n -/
def lastTwoDigits (n : ℕ) : ℕ := (8^n) % 100

/-- The cycle length of the last two digits of 8^n -/
def cycleLengthOfLastTwoDigits : ℕ := 20

theorem tens_digit_of_8_pow_2015 :
  ∃ (f : ℕ → ℕ),
    (∀ n, f n = lastTwoDigits n) ∧
    (∀ n, f (n + cycleLengthOfLastTwoDigits) = f n) ∧
    (f 15 = 32) →
    (8^2015 / 10) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2015_l3356_335680


namespace NUMINAMATH_CALUDE_bulb_selection_problem_l3356_335665

theorem bulb_selection_problem (total_bulbs : ℕ) (defective_bulbs : ℕ) (probability : ℚ) :
  total_bulbs = 10 →
  defective_bulbs = 4 →
  probability = 1 / 15 →
  ∃ n : ℕ, (((total_bulbs - defective_bulbs : ℚ) / total_bulbs) ^ n = probability) ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_bulb_selection_problem_l3356_335665


namespace NUMINAMATH_CALUDE_gcd_120_4_l3356_335632

/-- The greatest common divisor of 120 and 4 is 4, given they share exactly three positive divisors -/
theorem gcd_120_4 : 
  (∃ (S : Finset Nat), S = {d : Nat | d ∣ 120 ∧ d ∣ 4} ∧ Finset.card S = 3) →
  Nat.gcd 120 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_gcd_120_4_l3356_335632


namespace NUMINAMATH_CALUDE_hilltop_volleyball_club_members_l3356_335613

/-- Represents the Hilltop Volleyball Club inventory problem -/
theorem hilltop_volleyball_club_members :
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 7
  let items_per_member : ℕ := 3
  let cost_per_member : ℕ := items_per_member * (sock_cost + tshirt_cost)
  let total_cost : ℕ := 4026
  total_cost / cost_per_member = 71 :=
by sorry

end NUMINAMATH_CALUDE_hilltop_volleyball_club_members_l3356_335613


namespace NUMINAMATH_CALUDE_angle_aoc_in_regular_octagon_l3356_335630

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The center of a regular octagon -/
def center (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- Angle between two points and the center -/
def angle_with_center (octagon : RegularOctagon) (p1 p2 : Fin 8) : ℝ := sorry

theorem angle_aoc_in_regular_octagon (octagon : RegularOctagon) :
  angle_with_center octagon 0 2 = 45 := by sorry

end NUMINAMATH_CALUDE_angle_aoc_in_regular_octagon_l3356_335630


namespace NUMINAMATH_CALUDE_plan_y_cheaper_at_min_mb_l3356_335663

/-- Represents the cost of a data plan in cents -/
def PlanCost (initialFee : ℕ) (ratePerMB : ℕ) (dataUsage : ℕ) : ℕ :=
  initialFee * 100 + ratePerMB * dataUsage

/-- The minimum whole number of MBs for Plan Y to be cheaper than Plan X -/
def minMBForPlanYCheaper : ℕ := 501

theorem plan_y_cheaper_at_min_mb :
  PlanCost 25 10 minMBForPlanYCheaper < PlanCost 0 15 minMBForPlanYCheaper ∧
  ∀ m : ℕ, m < minMBForPlanYCheaper →
    PlanCost 0 15 m ≤ PlanCost 25 10 m :=
by sorry

end NUMINAMATH_CALUDE_plan_y_cheaper_at_min_mb_l3356_335663


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3356_335624

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), (35 : ℤ) * x ^ 2 + b * x + 35 = (c * x + d) * (e * x + f)) →
  (∃ (k : ℤ), b = 2 * k) ∧ 
  ¬(∀ (k : ℤ), ∃ (c d e f : ℤ), (35 : ℤ) * x ^ 2 + (2 * k) * x + 35 = (c * x + d) * (e * x + f)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3356_335624


namespace NUMINAMATH_CALUDE_rectangle_area_l3356_335628

/-- Given a rectangle where the length is 3 times the width and the width is 5 inches,
    prove that its area is 75 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 → length = 3 * width → area = length * width → area = 75 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3356_335628


namespace NUMINAMATH_CALUDE_train_speed_problem_l3356_335675

theorem train_speed_problem (length_train1 length_train2 distance_between speed_train2 time_to_cross : ℝ)
  (h1 : length_train1 = 100)
  (h2 : length_train2 = 150)
  (h3 : distance_between = 50)
  (h4 : speed_train2 = 15)
  (h5 : time_to_cross = 60)
  : ∃ speed_train1 : ℝ,
    speed_train1 = 10 ∧
    (length_train1 + length_train2 + distance_between) / time_to_cross = speed_train2 - speed_train1 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3356_335675


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l3356_335682

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimming_speed_in_still_water 
  (water_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : water_speed = 2)
  (h2 : distance = 8)
  (h3 : time = 4)
  (h4 : (swimming_speed - water_speed) * time = distance) :
  swimming_speed = 4 :=
by
  sorry

#check swimming_speed_in_still_water

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l3356_335682


namespace NUMINAMATH_CALUDE_rhea_children_eggs_l3356_335614

/-- The number of eggs eaten by Rhea's son and daughter every morning -/
def eggs_eaten_by_children (
  trays_per_week : ℕ)  -- Number of trays bought per week
  (eggs_per_tray : ℕ)  -- Number of eggs per tray
  (eggs_eaten_by_parents : ℕ)  -- Number of eggs eaten by parents per night
  (eggs_not_eaten : ℕ)  -- Number of eggs not eaten per week
  : ℕ :=
  trays_per_week * eggs_per_tray - 7 * eggs_eaten_by_parents - eggs_not_eaten

/-- Theorem stating that Rhea's son and daughter eat 14 eggs every morning -/
theorem rhea_children_eggs : 
  eggs_eaten_by_children 2 24 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_rhea_children_eggs_l3356_335614


namespace NUMINAMATH_CALUDE_rectangle_width_l3356_335667

theorem rectangle_width (square_perimeter : ℝ) (rectangle_length : ℝ) (rectangle_width : ℝ) : 
  square_perimeter = 160 →
  rectangle_length = 32 →
  (square_perimeter / 4) ^ 2 = 5 * (rectangle_length * rectangle_width) →
  rectangle_width = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l3356_335667


namespace NUMINAMATH_CALUDE_largest_tile_size_378_525_l3356_335604

/-- The largest square tile size that can exactly pave a rectangular courtyard -/
def largest_tile_size (length width : ℕ) : ℕ :=
  Nat.gcd length width

/-- Theorem: The largest square tile size for a 378 cm by 525 cm courtyard is 21 cm -/
theorem largest_tile_size_378_525 :
  largest_tile_size 378 525 = 21 := by
  sorry

#eval largest_tile_size 378 525

end NUMINAMATH_CALUDE_largest_tile_size_378_525_l3356_335604


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l3356_335662

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : (a 2)^2 + 12*(a 2) - 8 = 0 ∧ (a 10)^2 + 12*(a 10) - 8 = 0) :
  a 6 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l3356_335662


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_proof_l3356_335654

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_9 : ℕ := 8820

theorem largest_even_digit_multiple_of_9_proof :
  (has_only_even_digits largest_even_digit_multiple_of_9) ∧
  (largest_even_digit_multiple_of_9 < 10000) ∧
  (largest_even_digit_multiple_of_9 % 9 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_9 →
    ¬(has_only_even_digits m ∧ m < 10000 ∧ m % 9 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_proof_l3356_335654


namespace NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l3356_335623

def fibonacci_factorial_series : List Nat :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def sum_last_two_digits (series : List Nat) : Nat :=
  (series.map (λ x => last_two_digits (Nat.factorial x))).sum

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l3356_335623


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3356_335695

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define the set A
def A : Set ℝ := {x | x > 2}

-- Define the complement of A with respect to U
def complement_U_A : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem stating the complement of A with respect to U
theorem complement_of_A_wrt_U :
  complement_U_A = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l3356_335695


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3356_335605

theorem trigonometric_identities :
  (∃ (x y : ℝ), 
    x = Real.sin (-1395 * π / 180) * Real.cos (1140 * π / 180) + 
        Real.cos (-1020 * π / 180) * Real.sin (750 * π / 180) ∧
    y = Real.sin (-11 * π / 6) + Real.cos (3 * π / 4) * Real.tan (4 * π) ∧
    x = (Real.sqrt 2 + 1) / 4 ∧
    y = 1 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3356_335605


namespace NUMINAMATH_CALUDE_find_A_l3356_335606

theorem find_A (A B : ℕ) : 
  A ≤ 9 →
  B ≤ 9 →
  100 ≤ A * 100 + 78 →
  A * 100 + 78 < 1000 →
  100 ≤ 200 + B →
  200 + B < 1000 →
  A * 100 + 78 - (200 + B) = 364 →
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_find_A_l3356_335606


namespace NUMINAMATH_CALUDE_biographies_shelved_l3356_335693

def total_books : ℕ := 46
def top_section_books : ℕ := 24
def western_novels : ℕ := 5

def bottom_section_books : ℕ := total_books - top_section_books

def mystery_books : ℕ := bottom_section_books / 2

theorem biographies_shelved :
  total_books - top_section_books - mystery_books - western_novels = 6 :=
by sorry

end NUMINAMATH_CALUDE_biographies_shelved_l3356_335693
