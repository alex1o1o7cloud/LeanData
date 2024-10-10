import Mathlib

namespace train_length_l672_67261

/-- The length of a train given its speed and time to cross a pole --/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 18 → ∃ (length_m : ℝ), abs (length_m - 300) < 1 := by
  sorry

end train_length_l672_67261


namespace largest_prime_factor_of_3328_l672_67272

theorem largest_prime_factor_of_3328 : 
  (Nat.factors 3328).maximum? = some 13 := by
  sorry

end largest_prime_factor_of_3328_l672_67272


namespace total_dry_grapes_weight_l672_67225

/-- Calculates the total weight of Dry Grapes after dehydrating Fresh Grapes Type A and B -/
theorem total_dry_grapes_weight 
  (water_content_A : Real) 
  (water_content_B : Real)
  (weight_A : Real) 
  (weight_B : Real) :
  water_content_A = 0.92 →
  water_content_B = 0.88 →
  weight_A = 30 →
  weight_B = 50 →
  (1 - water_content_A) * weight_A + (1 - water_content_B) * weight_B = 8.4 :=
by sorry

end total_dry_grapes_weight_l672_67225


namespace store_a_highest_capacity_l672_67291

/-- Represents a store with its CD storage capacity -/
structure Store where
  shelves : ℕ
  racks_per_shelf : ℕ
  cds_per_rack : ℕ

/-- Calculates the total CD capacity of a store -/
def total_capacity (s : Store) : ℕ :=
  s.shelves * s.racks_per_shelf * s.cds_per_rack

/-- The three stores with their respective capacities -/
def store_a : Store := ⟨5, 6, 9⟩
def store_b : Store := ⟨8, 4, 7⟩
def store_c : Store := ⟨10, 3, 8⟩

/-- Theorem stating that Store A has the highest total CD capacity -/
theorem store_a_highest_capacity :
  total_capacity store_a > total_capacity store_b ∧
  total_capacity store_a > total_capacity store_c :=
by
  sorry


end store_a_highest_capacity_l672_67291


namespace train_average_speed_l672_67221

/-- Calculates the average speed of a train journey with a stop -/
theorem train_average_speed 
  (distance1 : ℝ) 
  (time1 : ℝ) 
  (stop_time : ℝ) 
  (distance2 : ℝ) 
  (time2 : ℝ) 
  (h1 : distance1 = 240) 
  (h2 : time1 = 3) 
  (h3 : stop_time = 0.5) 
  (h4 : distance2 = 450) 
  (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = (240 + 450) / (3 + 0.5 + 5) :=
by sorry

end train_average_speed_l672_67221


namespace digit_multiplication_l672_67265

theorem digit_multiplication (A B : ℕ) : 
  A < 10 ∧ B < 10 ∧ A ≠ B ∧ A * (10 * A + B) = 100 * B + 11 * A → A = 8 ∧ B = 6 := by
  sorry

end digit_multiplication_l672_67265


namespace solution_conditions_l672_67285

-- Define the variables
variable (a b x y z : ℝ)

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  (a > 0) ∧ (abs b < a) ∧ (a < Real.sqrt 2 * abs b) ∧ ((3 * a^2 - b^2) * (3 * b^2 - a^2) > 0)

-- Define the equations
def equations (a b x y z : ℝ) : Prop :=
  (x + y + z = a) ∧ (x^2 + y^2 + z^2 = b^2) ∧ (x * y = z^2)

-- Define the property of distinct positive numbers
def distinct_positive (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

-- Theorem statement
theorem solution_conditions (a b x y z : ℝ) :
  equations a b x y z → (conditions a b ↔ distinct_positive x y z) := by
  sorry

end solution_conditions_l672_67285


namespace not_order_preserving_isomorphic_Z_Q_l672_67220

theorem not_order_preserving_isomorphic_Z_Q :
  ¬∃ f : ℤ → ℚ, (∀ q : ℚ, ∃ z : ℤ, f z = q) ∧
    (∀ z₁ z₂ : ℤ, z₁ < z₂ → f z₁ < f z₂) := by
  sorry

end not_order_preserving_isomorphic_Z_Q_l672_67220


namespace largest_number_with_digit_sum_23_l672_67234

def digit_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def all_digits_different (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem largest_number_with_digit_sum_23 :
  ∀ n : Nat, n ≤ 999 →
    (digit_sum n = 23 ∧ all_digits_different n) →
    n ≤ 986 :=
by
  sorry

end largest_number_with_digit_sum_23_l672_67234


namespace opposite_of_negative_six_l672_67231

theorem opposite_of_negative_six : -(-(6)) = 6 := by
  sorry

end opposite_of_negative_six_l672_67231


namespace repeating_decimal_proof_l672_67274

/-- The repeating decimal 0.4̅67̅ as a rational number -/
def repeating_decimal : ℚ := 463 / 990

/-- Proof that 0.4̅67̅ is equal to 463/990 and is in lowest terms -/
theorem repeating_decimal_proof :
  repeating_decimal = 463 / 990 ∧
  (∀ n d : ℤ, n / d = 463 / 990 → d ≠ 0 → d.natAbs ≤ 990 → d = 990) :=
sorry

end repeating_decimal_proof_l672_67274


namespace x_wins_probability_l672_67224

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : ℚ
  
/-- Represents the outcome of the tournament for two specific teams -/
structure TournamentOutcome where
  team_x_points : Nat
  team_y_points : Nat

/-- Calculates the probability of team X finishing with more points than team Y -/
def probability_x_wins (t : SoccerTournament) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given conditions -/
theorem x_wins_probability (t : SoccerTournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_team = 7)
  (h3 : t.win_probability = 1/2) :
  probability_x_wins t = 561/1024 := by
  sorry

end x_wins_probability_l672_67224


namespace tetrahedron_inequality_l672_67200

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The minimum distance between opposite edges -/
  d : ℝ
  /-- The length of the shortest height -/
  h : ℝ
  /-- d is positive -/
  d_pos : d > 0
  /-- h is positive -/
  h_pos : h > 0

/-- 
For any tetrahedron, twice the minimum distance between 
opposite edges is greater than the length of the shortest height
-/
theorem tetrahedron_inequality (t : Tetrahedron) : 2 * t.d > t.h := by
  sorry

end tetrahedron_inequality_l672_67200


namespace carpenter_theorem_l672_67276

def carpenter_problem (total_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) : ℕ :=
  let current_woodblocks := current_logs * woodblocks_per_log
  let remaining_woodblocks := total_woodblocks - current_woodblocks
  remaining_woodblocks / woodblocks_per_log

theorem carpenter_theorem :
  carpenter_problem 80 8 5 = 8 := by
  sorry

end carpenter_theorem_l672_67276


namespace cos_forty_five_degrees_l672_67241

theorem cos_forty_five_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_forty_five_degrees_l672_67241


namespace car_speed_second_hour_l672_67278

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed of the car in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 80) 
  (h2 : average_speed = 70) : 
  (2 * average_speed - speed_first_hour) = 60 := by
  sorry

#check car_speed_second_hour

end car_speed_second_hour_l672_67278


namespace largest_common_term_l672_67257

def is_in_arithmetic_sequence (a : ℕ) (first d : ℕ) : Prop :=
  ∃ k : ℕ, a = first + k * d

theorem largest_common_term (a₁ d₁ a₂ d₂ : ℕ) (h₁ : a₁ = 3) (h₂ : d₁ = 8) (h₃ : a₂ = 5) (h₄ : d₂ = 9) :
  (∀ n : ℕ, n > 131 ∧ n ≤ 150 → ¬(is_in_arithmetic_sequence n a₁ d₁ ∧ is_in_arithmetic_sequence n a₂ d₂)) ∧
  (is_in_arithmetic_sequence 131 a₁ d₁ ∧ is_in_arithmetic_sequence 131 a₂ d₂) :=
sorry

end largest_common_term_l672_67257


namespace josephs_speed_josephs_speed_proof_l672_67298

/-- Joseph's driving problem -/
theorem josephs_speed : ℝ → Prop :=
  fun speed : ℝ =>
    let kyle_distance : ℝ := 62 * 2
    let joseph_distance : ℝ := kyle_distance + 1
    let joseph_time : ℝ := 2.5
    speed * joseph_time = joseph_distance → speed = 50

/-- Proof of Joseph's speed -/
theorem josephs_speed_proof : ∃ (speed : ℝ), josephs_speed speed := by
  sorry

end josephs_speed_josephs_speed_proof_l672_67298


namespace not_p_and_q_implies_at_most_one_true_l672_67247

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

end not_p_and_q_implies_at_most_one_true_l672_67247


namespace y_congruence_l672_67280

theorem y_congruence (y : ℤ) 
  (h1 : (2 + y) % (2^3) = (2 * 2) % (2^3))
  (h2 : (4 + y) % (4^3) = (4 * 2) % (4^3))
  (h3 : (6 + y) % (6^3) = (6 * 2) % (6^3)) :
  y % 24 = 2 := by
  sorry

end y_congruence_l672_67280


namespace triangle_side_length_l672_67206

/-- Given a triangle DEF with side lengths and median as specified, prove that DF = √130 -/
theorem triangle_side_length (DE EF DN : ℝ) (h1 : DE = 7) (h2 : EF = 9) (h3 : DN = 9/2) : 
  ∃ (DF : ℝ), DF = Real.sqrt 130 := by
  sorry

end triangle_side_length_l672_67206


namespace power_equality_implies_y_equals_four_l672_67283

theorem power_equality_implies_y_equals_four :
  ∀ y : ℝ, (4 : ℝ)^12 = 64^y → y = 4 := by
sorry

end power_equality_implies_y_equals_four_l672_67283


namespace alcohol_solution_proof_l672_67207

/-- Proves that adding 3 liters of pure alcohol to a 6-liter solution
    that is 25% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.25
  let added_alcohol : ℝ := 3
  let final_concentration : ℝ := 0.5
  let final_volume : ℝ := initial_volume + added_alcohol
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_alcohol : ℝ := initial_alcohol + added_alcohol
  final_alcohol / final_volume = final_concentration := by
  sorry

end alcohol_solution_proof_l672_67207


namespace string_measurement_l672_67208

theorem string_measurement (string_length : ℚ) (h : string_length = 2/3) :
  let folded_length := string_length / 4
  string_length - folded_length = 1/2 := by sorry

end string_measurement_l672_67208


namespace expression_evaluation_l672_67222

theorem expression_evaluation :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 :=
by sorry

end expression_evaluation_l672_67222


namespace kenny_jumping_jacks_wednesday_l672_67218

/-- Represents the number of jumping jacks Kenny did on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

theorem kenny_jumping_jacks_wednesday (lastWeek : ℕ) (thisWeek : WeeklyJumpingJacks) 
    (h1 : lastWeek = 324)
    (h2 : thisWeek.sunday = 34)
    (h3 : thisWeek.monday = 20)
    (h4 : thisWeek.tuesday = 0)
    (h5 : thisWeek.thursday = 64 ∨ thisWeek.wednesday = 64)
    (h6 : thisWeek.friday = 23)
    (h7 : thisWeek.saturday = 61)
    (h8 : totalJumpingJacks thisWeek > lastWeek) :
  thisWeek.wednesday = 59 := by
  sorry

#check kenny_jumping_jacks_wednesday

end kenny_jumping_jacks_wednesday_l672_67218


namespace oil_leak_calculation_l672_67252

/-- The total amount of oil leaked into the water -/
def total_oil_leaked (pre_repair_leak : ℕ) (during_repair_leak : ℕ) : ℕ :=
  pre_repair_leak + during_repair_leak

/-- Theorem stating the total amount of oil leaked -/
theorem oil_leak_calculation :
  total_oil_leaked 6522 5165 = 11687 := by
  sorry

end oil_leak_calculation_l672_67252


namespace series_sum_equals_half_l672_67251

noncomputable def series_sum (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

theorem series_sum_equals_half :
  ∑' n, series_sum n = 1/2 := by sorry

end series_sum_equals_half_l672_67251


namespace trig_identity_l672_67282

theorem trig_identity (α : Real) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := by
  sorry

end trig_identity_l672_67282


namespace monotone_decreasing_implies_k_bound_l672_67262

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 + 2 * k * x - 8

theorem monotone_decreasing_implies_k_bound :
  (∀ x₁ x₂, -5 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ -1 → f k x₁ > f k x₂) →
  k ≤ 2 := by
  sorry

end monotone_decreasing_implies_k_bound_l672_67262


namespace sqrt_inequality_l672_67229

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) :
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) :=
sorry

end sqrt_inequality_l672_67229


namespace fraction_sum_l672_67249

theorem fraction_sum : (3 : ℚ) / 9 + (7 : ℚ) / 14 = (5 : ℚ) / 6 := by sorry

end fraction_sum_l672_67249


namespace problem_solution_l672_67275

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -6)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 17 / 2 := by
  sorry

end problem_solution_l672_67275


namespace rectangular_hall_area_l672_67211

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1 / 2) * length →
  length - width = 17 →
  length * width = 578 := by
sorry

end rectangular_hall_area_l672_67211


namespace range_of_f_l672_67259

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x < 5 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -4 ≤ y ∧ y < 5 } := by sorry

end range_of_f_l672_67259


namespace sum_of_coefficients_l672_67232

def polynomial (x : ℝ) : ℝ := 3 * (3 * x^7 + 8 * x^4 - 7) + 7 * (x^5 - 7 * x^2 + 5)

theorem sum_of_coefficients : polynomial 1 = 5 := by
  sorry

end sum_of_coefficients_l672_67232


namespace point_transformation_l672_67236

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Transformation from x-axis coordinates to y-axis coordinates -/
def transformToYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

/-- Theorem stating the transformation of point P -/
theorem point_transformation :
  ∃ (P : Point2D), P.x = 1 ∧ P.y = -2 → (transformToYAxis P).x = -1 ∧ (transformToYAxis P).y = 2 := by
  sorry

end point_transformation_l672_67236


namespace smallest_divisible_by_first_five_primes_l672_67299

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem smallest_divisible_by_first_five_primes :
  (∀ p ∈ first_five_primes, 2310 % p = 0) ∧
  (∀ n < 2310, ∃ p ∈ first_five_primes, n % p ≠ 0) :=
sorry

end smallest_divisible_by_first_five_primes_l672_67299


namespace smallest_x_y_sum_l672_67242

/-- The smallest positive integer x such that 720x is a square number -/
def x : ℕ+ := sorry

/-- The smallest positive integer y such that 720y is a fourth power -/
def y : ℕ+ := sorry

theorem smallest_x_y_sum : 
  (∀ x' : ℕ+, x' < x → ¬∃ n : ℕ+, 720 * x' = n^2) ∧
  (∀ y' : ℕ+, y' < y → ¬∃ n : ℕ+, 720 * y' = n^4) ∧
  (∃ n : ℕ+, 720 * x = n^2) ∧
  (∃ n : ℕ+, 720 * y = n^4) ∧
  (x : ℕ) + (y : ℕ) = 1130 := by sorry

end smallest_x_y_sum_l672_67242


namespace equilateral_triangle_splitting_l672_67238

/-- An equilateral triangle with side length 111 -/
def EquilateralTriangle : ℕ := 111

/-- The number of marked points in the triangle -/
def MarkedPoints : ℕ := 6216

/-- The number of linear sets -/
def LinearSets : ℕ := 111

/-- The number of ways to split the marked points into linear sets -/
def SplittingWays : ℕ := 2^4107

theorem equilateral_triangle_splitting (T : ℕ) (points : ℕ) (sets : ℕ) (ways : ℕ) :
  T = EquilateralTriangle →
  points = MarkedPoints →
  sets = LinearSets →
  ways = SplittingWays →
  ways = 2^(points / 3 * 2) :=
by sorry

end equilateral_triangle_splitting_l672_67238


namespace book_pages_count_l672_67240

/-- Given a book with pages numbered consecutively starting from 1,
    this function calculates the total number of digits used to number the pages. -/
def totalDigits (n : ℕ) : ℕ :=
  (n.min 9) + 
  (n - 9).max 0 * 2 + 
  (n - 99).max 0 * 3

/-- Theorem stating that a book has 369 pages if the total number of digits
    used in numbering is 999. -/
theorem book_pages_count : totalDigits 369 = 999 := by
  sorry

end book_pages_count_l672_67240


namespace sin_five_pi_sixths_l672_67212

theorem sin_five_pi_sixths : Real.sin (5 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end sin_five_pi_sixths_l672_67212


namespace union_equals_reals_subset_of_complement_l672_67289

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

-- Theorem for part (1)
theorem union_equals_reals (a : ℝ) : 
  A ∪ B a = Set.univ ↔ a ∈ Set.Iic 0 :=
sorry

-- Theorem for part (2)
theorem subset_of_complement (a : ℝ) :
  B a ⊆ (Set.univ \ A) ↔ a ∈ Set.Ici (1/2) :=
sorry

end union_equals_reals_subset_of_complement_l672_67289


namespace smallest_base_10_integer_l672_67294

def is_valid_base_6_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 5

def is_valid_base_8_digit (y : ℕ) : Prop := y ≥ 0 ∧ y ≤ 7

def base_6_to_decimal (x : ℕ) : ℕ := 6 * x + x

def base_8_to_decimal (y : ℕ) : ℕ := 8 * y + y

theorem smallest_base_10_integer : 
  ∃ (x y : ℕ), 
    is_valid_base_6_digit x ∧ 
    is_valid_base_8_digit y ∧ 
    base_6_to_decimal x = 63 ∧ 
    base_8_to_decimal y = 63 ∧ 
    (∀ (x' y' : ℕ), 
      is_valid_base_6_digit x' ∧ 
      is_valid_base_8_digit y' ∧ 
      base_6_to_decimal x' = base_8_to_decimal y' → 
      base_6_to_decimal x' ≥ 63) :=
by sorry

end smallest_base_10_integer_l672_67294


namespace three_digit_number_sum_l672_67243

theorem three_digit_number_sum (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a ≠ 0 →
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 3194 →
  100 * a + 10 * b + c = 358 := by
  sorry

end three_digit_number_sum_l672_67243


namespace janelle_has_72_marbles_l672_67226

/-- The number of marbles Janelle has after buying blue marbles and giving some away as a gift. -/
def janelles_marbles : ℕ :=
  let initial_green : ℕ := 26
  let blue_bags : ℕ := 6
  let marbles_per_bag : ℕ := 10
  let gift_green : ℕ := 6
  let gift_blue : ℕ := 8
  let total_blue : ℕ := blue_bags * marbles_per_bag
  let total_before_gift : ℕ := initial_green + total_blue
  let total_gift : ℕ := gift_green + gift_blue
  total_before_gift - total_gift

/-- Theorem stating that Janelle has 72 marbles after the transactions. -/
theorem janelle_has_72_marbles : janelles_marbles = 72 := by
  sorry

end janelle_has_72_marbles_l672_67226


namespace pet_store_cages_l672_67273

/-- Represents the number of birds in each cage -/
def birds_per_cage : ℕ := 10

/-- Represents the total number of birds in the store -/
def total_birds : ℕ := 40

/-- Represents the number of bird cages in the store -/
def num_cages : ℕ := total_birds / birds_per_cage

theorem pet_store_cages : num_cages = 4 := by
  sorry

end pet_store_cages_l672_67273


namespace quadratic_complete_square_l672_67295

theorem quadratic_complete_square (x : ℝ) :
  25 * x^2 + 20 * x - 1000 = 0 →
  ∃ (p t : ℝ), (x + p)^2 = t ∧ t = 104/25 := by
  sorry

end quadratic_complete_square_l672_67295


namespace dime_count_in_collection_l672_67237

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of a coin collection in cents --/
def totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter

/-- Calculates the total number of coins in a collection --/
def totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters

theorem dime_count_in_collection (c : CoinCollection) :
  totalCoins c = 13 ∧
  totalValue c = 141 ∧
  c.pennies ≥ 2 ∧
  c.nickels ≥ 2 ∧
  c.dimes ≥ 2 ∧
  c.quarters ≥ 2 →
  c.dimes = 3 := by
  sorry

end dime_count_in_collection_l672_67237


namespace seven_balls_four_boxes_l672_67277

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes with no empty boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 7 indistinguishable balls into 4 indistinguishable boxes with no empty boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 3 := by
  sorry

end seven_balls_four_boxes_l672_67277


namespace final_result_l672_67228

def alternateOperations (start : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => start
  | n + 1 => if n % 2 = 0 
             then alternateOperations start n * 3 
             else alternateOperations start n / 2

theorem final_result : alternateOperations 1458 5 = 3^9 / 2 := by
  sorry

end final_result_l672_67228


namespace sally_sunday_sandwiches_l672_67263

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of pieces of bread used per sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

theorem sally_sunday_sandwiches :
  sunday_sandwiches = total_bread / bread_per_sandwich - saturday_sandwiches :=
sorry

end sally_sunday_sandwiches_l672_67263


namespace complex_magnitude_equation_l672_67202

theorem complex_magnitude_equation : 
  ∃ (t : ℝ), t > 0 ∧ Complex.abs (9 + t * Complex.I) = 15 ↔ t = 12 := by sorry

end complex_magnitude_equation_l672_67202


namespace existence_of_irrational_shifts_l672_67213

theorem existence_of_irrational_shifts (n : ℕ) (a : Fin n → ℝ) :
  ∃ b : ℝ, ∀ i : Fin n, Irrational (a i + b) := by
  sorry

end existence_of_irrational_shifts_l672_67213


namespace part_one_part_two_part_three_l672_67250

/-- Definition of "equation number pair" -/
def is_equation_number_pair (a b : ℝ) : Prop :=
  ∃ x : ℝ, (a / x) + 1 = b ∧ x = 1 / (a + b)

/-- Part 1: Prove [3,-5] is an "equation number pair" and [-2,4] is not -/
theorem part_one :
  is_equation_number_pair 3 (-5) ∧ ¬is_equation_number_pair (-2) 4 := by sorry

/-- Part 2: If [n,3-n] is an "equation number pair", then n = 1/2 -/
theorem part_two (n : ℝ) :
  is_equation_number_pair n (3 - n) → n = 1/2 := by sorry

/-- Part 3: If [m-k,k] is an "equation number pair" (m ≠ -1, m ≠ 0, k ≠ 1), then k = (m^2 + 1) / (m + 1) -/
theorem part_three (m k : ℝ) (hm1 : m ≠ -1) (hm2 : m ≠ 0) (hk : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) := by sorry

end part_one_part_two_part_three_l672_67250


namespace investment_difference_theorem_l672_67281

/-- Calculates the difference in total amounts between two investment schemes after one year -/
def investment_difference (initial_a : ℝ) (initial_b : ℝ) (yield_a : ℝ) (yield_b : ℝ) : ℝ :=
  (initial_a * (1 + yield_a)) - (initial_b * (1 + yield_b))

/-- Theorem stating the difference in total amounts between schemes A and B after one year -/
theorem investment_difference_theorem :
  investment_difference 300 200 0.3 0.5 = 90 := by
  sorry

end investment_difference_theorem_l672_67281


namespace b_contribution_is_16200_l672_67258

/-- Calculates the partner's contribution given the investment details and profit ratio -/
def calculate_partner_contribution (a_investment : ℕ) (total_months : ℕ) (b_join_month : ℕ) (a_profit_share : ℕ) (b_profit_share : ℕ) : ℕ :=
  let a_months := total_months
  let b_months := total_months - b_join_month
  (a_investment * a_months * b_profit_share) / (a_profit_share * b_months)

/-- Proves that B's contribution to the capital is 16200 rs given the problem conditions -/
theorem b_contribution_is_16200 :
  let a_investment := 4500
  let total_months := 12
  let b_join_month := 7
  let a_profit_share := 2
  let b_profit_share := 3
  calculate_partner_contribution a_investment total_months b_join_month a_profit_share b_profit_share = 16200 := by
  sorry

end b_contribution_is_16200_l672_67258


namespace avery_donation_l672_67204

theorem avery_donation (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 4 → 
  pants = 2 * shirts → 
  shorts = pants / 2 → 
  shirts + pants + shorts = 16 := by
sorry

end avery_donation_l672_67204


namespace functional_equation_solution_l672_67227

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  ((∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1)) := by
  sorry

end functional_equation_solution_l672_67227


namespace children_age_sum_l672_67203

/-- Given 5 children with an age difference of 2 years between each, 
    and the eldest being 12 years old, the sum of their ages is 40 years. -/
theorem children_age_sum : 
  let num_children : ℕ := 5
  let age_diff : ℕ := 2
  let eldest_age : ℕ := 12
  let ages : List ℕ := List.range num_children |>.map (λ i => eldest_age - i * age_diff)
  ages.sum = 40 := by sorry

end children_age_sum_l672_67203


namespace equal_share_of_tea_l672_67269

-- Define the total number of cups of tea
def total_cups : ℕ := 10

-- Define the number of people sharing the tea
def num_people : ℕ := 5

-- Define the number of cups each person receives
def cups_per_person : ℚ := total_cups / num_people

-- Theorem to prove
theorem equal_share_of_tea :
  cups_per_person = 2 := by sorry

end equal_share_of_tea_l672_67269


namespace base7_sum_theorem_l672_67239

/-- Represents a single digit in base 7 --/
def Base7Digit := Fin 7

/-- Converts a base 7 number to base 10 --/
def toBase10 (x : Base7Digit) : Nat := x.val

/-- The equation 5XY₇ + 32₇ = 62X₇ in base 7 --/
def base7Equation (X Y : Base7Digit) : Prop :=
  (5 * 7 + toBase10 X) * 7 + toBase10 Y + 32 = (6 * 7 + 2) * 7 + toBase10 X

/-- Theorem stating that if X and Y satisfy the base 7 equation, then X + Y = 10 in base 10 --/
theorem base7_sum_theorem (X Y : Base7Digit) : 
  base7Equation X Y → toBase10 X + toBase10 Y = 10 := by
  sorry

end base7_sum_theorem_l672_67239


namespace nested_sqrt_bounds_l672_67209

theorem nested_sqrt_bounds (x : ℝ) (h : x = Real.sqrt (3 + x)) : 1 < x ∧ x < 3 := by
  sorry

end nested_sqrt_bounds_l672_67209


namespace sine_sum_problem_l672_67268

theorem sine_sum_problem (α : ℝ) (h : Real.sin (π / 3 + α) + Real.sin α = (4 * Real.sqrt 3) / 5) :
  Real.sin (α + 7 * π / 6) = -4 / 5 := by sorry

end sine_sum_problem_l672_67268


namespace inverse_proportion_ratio_l672_67205

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : b₁ ≠ 0) (h₄ : b₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ a b : ℝ, a * b = k) →
  a₁ / a₂ = 3 / 5 →
  b₁ / b₂ = 5 / 3 := by
sorry

end inverse_proportion_ratio_l672_67205


namespace fraction_simplification_l672_67215

theorem fraction_simplification :
  (156 + 72 : ℚ) / 9000 = 19 / 750 := by sorry

end fraction_simplification_l672_67215


namespace doberman_schnauzer_relationship_num_dobermans_proof_l672_67270

/-- The number of Doberman puppies -/
def num_dobermans : ℝ := 37.5

/-- The number of Schnauzers -/
def num_schnauzers : ℕ := 55

/-- Theorem stating the relationship between Doberman puppies and Schnauzers -/
theorem doberman_schnauzer_relationship : 
  3 * num_dobermans - 5 + (num_dobermans - num_schnauzers) = 90 :=
by sorry

/-- Theorem proving the number of Doberman puppies -/
theorem num_dobermans_proof : num_dobermans = 37.5 :=
by sorry

end doberman_schnauzer_relationship_num_dobermans_proof_l672_67270


namespace rotation_volume_of_specific_trapezoid_l672_67230

/-- A trapezoid with given properties -/
structure Trapezoid where
  larger_base : ℝ
  smaller_base : ℝ
  adjacent_angle : ℝ

/-- The volume of the solid formed by rotating the trapezoid about its larger base -/
def rotation_volume (t : Trapezoid) : ℝ := sorry

/-- The theorem stating the volume of the rotated trapezoid -/
theorem rotation_volume_of_specific_trapezoid :
  let t : Trapezoid := {
    larger_base := 8,
    smaller_base := 2,
    adjacent_angle := Real.pi / 4  -- 45° in radians
  }
  rotation_volume t = 36 * Real.pi := by sorry

end rotation_volume_of_specific_trapezoid_l672_67230


namespace f_properties_l672_67245

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

theorem f_properties :
  let T := Real.pi
  let φ := 7 * Real.pi / 12
  (∀ x, f (x + T) = f x) ∧
  (∀ y, T ≤ y → (∀ x, f (x + y) = f x) → y = T) ∧
  (Real.pi / 2 < φ ∧ φ < Real.pi) ∧
  (∀ x, f (x + φ) = f (-x + φ)) ∧
  (∀ ψ, Real.pi / 2 < ψ ∧ ψ < Real.pi → (∀ x, f (x + ψ) = f (-x + ψ)) → ψ = φ) :=
by sorry

end f_properties_l672_67245


namespace brokerage_percentage_approx_l672_67292

/-- Calculates the brokerage percentage given the cash realized and total amount --/
def brokerage_percentage (cash_realized : ℚ) (total_amount : ℚ) : ℚ :=
  ((cash_realized - total_amount) / total_amount) * 100

/-- Theorem stating that the brokerage percentage is approximately 0.24% --/
theorem brokerage_percentage_approx :
  let cash_realized : ℚ := 10425 / 100
  let total_amount : ℚ := 104
  abs (brokerage_percentage cash_realized total_amount - 24 / 100) < 1 / 1000 := by
  sorry

#eval brokerage_percentage (10425 / 100) 104

end brokerage_percentage_approx_l672_67292


namespace unknown_blanket_rate_l672_67271

/-- The unknown rate of two blankets given specific conditions -/
theorem unknown_blanket_rate :
  let num_blankets_1 : ℕ := 5
  let price_1 : ℕ := 100
  let num_blankets_2 : ℕ := 5
  let price_2 : ℕ := 150
  let num_blankets_unknown : ℕ := 2
  let average_price : ℕ := 150
  let total_blankets : ℕ := num_blankets_1 + num_blankets_2 + num_blankets_unknown
  let known_cost : ℕ := num_blankets_1 * price_1 + num_blankets_2 * price_2
  ∃ (unknown_rate : ℕ),
    (known_cost + num_blankets_unknown * unknown_rate) / total_blankets = average_price ∧
    unknown_rate = 275 :=
by sorry

end unknown_blanket_rate_l672_67271


namespace hit_at_least_once_and_not_hit_single_time_are_mutually_exclusive_and_complementary_l672_67210

-- Define the sample space
def SampleSpace := Fin 3 → Bool

-- Define the event of hitting the target at least once
def HitAtLeastOnce (outcome : SampleSpace) : Prop :=
  ∃ i : Fin 3, outcome i = true

-- Define the event of not hitting the target a single time
def NotHitSingleTime (outcome : SampleSpace) : Prop :=
  ∀ i : Fin 3, outcome i = false

-- Theorem statement
theorem hit_at_least_once_and_not_hit_single_time_are_mutually_exclusive_and_complementary :
  (∀ outcome : SampleSpace, ¬(HitAtLeastOnce outcome ∧ NotHitSingleTime outcome)) ∧
  (∀ outcome : SampleSpace, HitAtLeastOnce outcome ↔ ¬NotHitSingleTime outcome) :=
sorry

end hit_at_least_once_and_not_hit_single_time_are_mutually_exclusive_and_complementary_l672_67210


namespace triangle_problem_geometric_sequence_problem_l672_67284

-- Triangle problem
theorem triangle_problem (a b : ℝ) (B : ℝ) 
  (ha : a = Real.sqrt 3) 
  (hb : b = Real.sqrt 2) 
  (hB : B = 45 * π / 180) :
  (∃ (A C c : ℝ),
    (A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
    (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) :=
by sorry

-- Geometric sequence problem
theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 7)
  (hS6 : S 6 = 91)
  (h_geom : ∀ n, S (n+1) - S n = (S 2 - S 1) * (S 2 / S 1) ^ (n-1)) :
  S 4 = 28 :=
by sorry

end triangle_problem_geometric_sequence_problem_l672_67284


namespace max_product_sum_300_l672_67297

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end max_product_sum_300_l672_67297


namespace second_fraction_in_compound_ratio_l672_67267

theorem second_fraction_in_compound_ratio
  (compound_ratio : ℝ)
  (h_ratio : compound_ratio = 0.07142857142857142)
  (f1 : ℝ) (h_f1 : f1 = 2/3)
  (f3 : ℝ) (h_f3 : f3 = 1/3)
  (f4 : ℝ) (h_f4 : f4 = 3/8) :
  ∃ x : ℝ, x * f1 * f3 * f4 = compound_ratio ∧ x = 0.8571428571428571 := by
  sorry

end second_fraction_in_compound_ratio_l672_67267


namespace division_problem_l672_67235

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 12401 → 
  divisor = 163 → 
  remainder = 13 → 
  dividend = divisor * quotient + remainder → 
  quotient = 76 := by
sorry

end division_problem_l672_67235


namespace weight_of_b_l672_67286

/-- Given the average weights of different combinations of people, prove the weight of person b. -/
theorem weight_of_b (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : (b + c + d) / 3 = 47) :
  b = 31 := by
  sorry


end weight_of_b_l672_67286


namespace simplify_expression_l672_67256

theorem simplify_expression (a x y : ℝ) : a^2 * x^2 - a^2 * y^2 = a^2 * (x + y) * (x - y) := by
  sorry

end simplify_expression_l672_67256


namespace xingguang_pass_rate_l672_67214

/-- Calculates the pass rate for a physical fitness test -/
def pass_rate (total_students : ℕ) (failed_students : ℕ) : ℚ :=
  (total_students - failed_students : ℚ) / total_students * 100

/-- Theorem: The pass rate for Xingguang Primary School's physical fitness test is 92% -/
theorem xingguang_pass_rate :
  pass_rate 500 40 = 92 := by
  sorry

end xingguang_pass_rate_l672_67214


namespace x_minus_y_equals_40_l672_67233

theorem x_minus_y_equals_40 (x y : ℤ) (h1 : x + y = 24) (h2 : x = 32) : x - y = 40 := by
  sorry

end x_minus_y_equals_40_l672_67233


namespace min_value_xy_l672_67201

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x * y ≥ 18 := by
  sorry

end min_value_xy_l672_67201


namespace twenty_bees_honey_production_l672_67264

/-- The amount of honey (in grams) produced by a given number of bees in 20 days. -/
def honey_production (num_bees : ℕ) : ℝ :=
  num_bees * 1

/-- Theorem stating that 20 honey bees produce 20 grams of honey in 20 days. -/
theorem twenty_bees_honey_production :
  honey_production 20 = 20 := by
  sorry

end twenty_bees_honey_production_l672_67264


namespace isosceles_triangle_perimeter_l672_67253

/-- An isosceles triangle with side lengths 6 and 7 has a perimeter of either 19 or 20 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) →  -- Given side lengths
  ((a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)) →  -- Isosceles condition
  a + b + c = 19 ∨ a + b + c = 20 :=
by sorry

end isosceles_triangle_perimeter_l672_67253


namespace eulers_formula_l672_67279

/-- A convex polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
structure ConvexPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Euler's formula for convex polyhedra states that the number of faces minus the number of edges plus the number of vertices equals two. -/
theorem eulers_formula (p : ConvexPolyhedron) : p.faces - p.edges + p.vertices = 2 := by
  sorry

end eulers_formula_l672_67279


namespace triangle_angle_B_l672_67246

theorem triangle_angle_B (A B C : Real) (a b c : Real) : 
  A = 2 * Real.pi / 3 →  -- 120° in radians
  a = 2 →
  b = 2 * Real.sqrt 3 / 3 →
  B = Real.pi / 6  -- 30° in radians
:= by sorry

end triangle_angle_B_l672_67246


namespace mark_sold_eight_less_l672_67260

theorem mark_sold_eight_less (total : ℕ) (mark_sold : ℕ) (ann_sold : ℕ) 
  (h_total : total = 9)
  (h_mark : mark_sold < total)
  (h_ann : ann_sold = total - 2)
  (h_mark_positive : mark_sold ≥ 1)
  (h_ann_positive : ann_sold ≥ 1)
  (h_total_greater : mark_sold + ann_sold < total) :
  total - mark_sold = 8 := by
sorry

end mark_sold_eight_less_l672_67260


namespace unique_right_triangle_from_medians_l672_67288

/-- Given two positive real numbers representing the lengths of medians from the endpoints of a hypotenuse,
    there exists at most one right triangle with these medians. -/
theorem unique_right_triangle_from_medians (s_a s_b : ℝ) (h_sa : s_a > 0) (h_sb : s_b > 0) :
  ∃! (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
    s_a^2 = (b/2)^2 + (c/2)^2 ∧ s_b^2 = (a/2)^2 + (c/2)^2 :=
by sorry

end unique_right_triangle_from_medians_l672_67288


namespace sin_theta_plus_2phi_l672_67216

theorem sin_theta_plus_2phi (θ φ : ℝ) (h1 : Complex.exp (Complex.I * θ) = (1/5) - (2/5) * Complex.I)
  (h2 : Complex.exp (Complex.I * φ) = (3/5) + (4/5) * Complex.I) :
  Real.sin (θ + 2*φ) = 62/125 := by
  sorry

end sin_theta_plus_2phi_l672_67216


namespace horner_method_f_2_l672_67219

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem horner_method_f_2 :
  f 2 = horner_eval [4, 0, -3, 2, 5, 1] 2 ∧ horner_eval [4, 0, -3, 2, 5, 1] 2 = 123 := by
  sorry

end horner_method_f_2_l672_67219


namespace marble_problem_l672_67287

/-- The number of marbles Doug lost at the playground -/
def marbles_lost (ed_initial : ℕ) (doug_initial : ℕ) (ed_final : ℕ) (doug_final : ℕ) : ℕ :=
  doug_initial - doug_final

theorem marble_problem (ed_initial : ℕ) (doug_initial : ℕ) (ed_final : ℕ) (doug_final : ℕ) :
  ed_initial = doug_initial + 10 →
  ed_initial = 45 →
  ed_final = doug_final + 21 →
  ed_initial = ed_final →
  marbles_lost ed_initial doug_initial ed_final doug_final = 11 := by
  sorry

#check marble_problem

end marble_problem_l672_67287


namespace max_dot_product_on_circle_l672_67244

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points M and N
def M : ℝ × ℝ := (2, 0)
def N : ℝ × ℝ := (0, -2)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ), Circle P.1 P.2 →
    dot_product (P.1 - M.1, P.2 - M.2) (P.1 - N.1, P.2 - N.2) ≤ 4 + 4 * Real.sqrt 2 :=
by sorry

end max_dot_product_on_circle_l672_67244


namespace simplify_expression_l672_67223

theorem simplify_expression (x : ℝ) : 114 * x - 69 * x + 15 = 45 * x + 15 := by
  sorry

end simplify_expression_l672_67223


namespace complex_number_quadrant_l672_67254

theorem complex_number_quadrant : ∀ z : ℂ, 
  z = (3 + 4*I) / I → (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l672_67254


namespace cost_per_pill_is_five_l672_67248

/-- Represents the annual costs and medication details for Tom --/
structure AnnualMedication where
  pillsPerDay : ℕ
  doctorVisitsPerYear : ℕ
  doctorVisitCost : ℕ
  insuranceCoveragePercent : ℚ
  totalAnnualCost : ℕ

/-- Calculates the cost per pill before insurance coverage --/
def costPerPillBeforeInsurance (am : AnnualMedication) : ℚ :=
  let totalPillsPerYear := am.pillsPerDay * 365
  let annualDoctorVisitsCost := am.doctorVisitsPerYear * am.doctorVisitCost
  let annualMedicationCost := am.totalAnnualCost - annualDoctorVisitsCost
  let totalMedicationCostBeforeInsurance := annualMedicationCost / (1 - am.insuranceCoveragePercent)
  totalMedicationCostBeforeInsurance / totalPillsPerYear

/-- Theorem stating that the cost per pill before insurance is $5 --/
theorem cost_per_pill_is_five (am : AnnualMedication) 
    (h1 : am.pillsPerDay = 2)
    (h2 : am.doctorVisitsPerYear = 2)
    (h3 : am.doctorVisitCost = 400)
    (h4 : am.insuranceCoveragePercent = 4/5)
    (h5 : am.totalAnnualCost = 1530) : 
  costPerPillBeforeInsurance am = 5 := by
  sorry

#eval costPerPillBeforeInsurance {
  pillsPerDay := 2,
  doctorVisitsPerYear := 2,
  doctorVisitCost := 400,
  insuranceCoveragePercent := 4/5,
  totalAnnualCost := 1530
}

end cost_per_pill_is_five_l672_67248


namespace principal_calculation_l672_67217

/-- Simple interest calculation --/
def simple_interest (principal rate time : ℝ) : ℝ := principal * (1 + rate * time)

/-- Theorem: Principal calculation given two-year and three-year amounts --/
theorem principal_calculation (amount_2_years amount_3_years : ℝ) 
  (h1 : amount_2_years = 3450)
  (h2 : amount_3_years = 3655)
  (h3 : ∃ (p r : ℝ), simple_interest p r 2 = amount_2_years ∧ simple_interest p r 3 = amount_3_years) :
  ∃ (p r : ℝ), p = 3245 ∧ simple_interest p r 2 = amount_2_years ∧ simple_interest p r 3 = amount_3_years := by
  sorry

end principal_calculation_l672_67217


namespace min_keystrokes_for_2018_l672_67266

/-- Represents the state of the screen and copy buffer -/
structure ScreenState where
  screen : ℕ  -- number of 'a's on screen
  buffer : ℕ  -- number of 'a's in copy buffer

/-- Represents the possible operations -/
inductive Operation
  | Copy
  | Paste

/-- Applies an operation to the screen state -/
def applyOperation (state : ScreenState) (op : Operation) : ScreenState :=
  match op with
  | Operation.Copy => { state with buffer := state.screen }
  | Operation.Paste => { state with screen := state.screen + state.buffer }

/-- Applies a sequence of operations to the initial screen state -/
def applyOperations (ops : List Operation) : ScreenState :=
  ops.foldl applyOperation { screen := 1, buffer := 0 }

/-- Checks if a sequence of operations achieves the goal -/
def achievesGoal (ops : List Operation) : Prop :=
  (applyOperations ops).screen ≥ 2018

theorem min_keystrokes_for_2018 :
  ∃ (ops : List Operation), achievesGoal ops ∧ ops.length = 21 ∧
  (∀ (other_ops : List Operation), achievesGoal other_ops → other_ops.length ≥ 21) :=
sorry

end min_keystrokes_for_2018_l672_67266


namespace positive_solution_form_l672_67293

theorem positive_solution_form (x : ℝ) : 
  x^2 - 18*x = 80 → 
  x > 0 → 
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ x = Real.sqrt c - d ∧ c = 161 ∧ d = 9 := by
  sorry

end positive_solution_form_l672_67293


namespace couch_cost_is_750_l672_67255

/-- The cost of the couch Daria bought -/
def couch_cost : ℕ := sorry

/-- The amount Daria has saved -/
def savings : ℕ := 500

/-- The cost of the table Daria bought -/
def table_cost : ℕ := 100

/-- The cost of the lamp Daria bought -/
def lamp_cost : ℕ := 50

/-- The amount Daria still owes after paying her savings -/
def remaining_debt : ℕ := 400

/-- Theorem stating that the couch cost is $750 -/
theorem couch_cost_is_750 :
  couch_cost = 750 ∧
  couch_cost + table_cost + lamp_cost = savings + remaining_debt :=
sorry

end couch_cost_is_750_l672_67255


namespace first_place_beats_joe_by_two_l672_67290

/-- Calculates the total points for a team based on their match results -/
def calculate_points (wins : ℕ) (ties : ℕ) : ℕ :=
  3 * wins + ties

/-- Represents the scoring system and match results for the soccer tournament -/
structure TournamentResults where
  win_points : ℕ := 3
  tie_points : ℕ := 1
  joe_wins : ℕ := 1
  joe_ties : ℕ := 3
  first_place_wins : ℕ := 2
  first_place_ties : ℕ := 2

/-- Theorem stating that the first-place team beat Joe's team by 2 points -/
theorem first_place_beats_joe_by_two (results : TournamentResults) :
  calculate_points results.first_place_wins results.first_place_ties -
  calculate_points results.joe_wins results.joe_ties = 2 :=
by
  sorry


end first_place_beats_joe_by_two_l672_67290


namespace problem_statement_l672_67296

theorem problem_statement (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x*y + x*z + y*z)) = -7 := by sorry

end problem_statement_l672_67296
