import Mathlib

namespace NUMINAMATH_CALUDE_min_value_product_l1757_175771

theorem min_value_product (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 37 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l1757_175771


namespace NUMINAMATH_CALUDE_seminar_duration_is_428_l1757_175759

/-- Represents the duration of a seminar session in minutes -/
def seminar_duration (first_part_hours : ℕ) (first_part_minutes : ℕ) (second_part_minutes : ℕ) (closing_event_seconds : ℕ) : ℕ :=
  (first_part_hours * 60 + first_part_minutes) + second_part_minutes + (closing_event_seconds / 60)

/-- Theorem stating that the seminar duration is 428 minutes given the specified conditions -/
theorem seminar_duration_is_428 :
  seminar_duration 4 45 135 500 = 428 := by
  sorry

end NUMINAMATH_CALUDE_seminar_duration_is_428_l1757_175759


namespace NUMINAMATH_CALUDE_specific_prism_volume_l1757_175717

/-- Represents a triangular prism -/
structure TriangularPrism :=
  (lateral_face_area : ℝ)
  (distance_to_face : ℝ)

/-- The volume of a triangular prism -/
def volume (prism : TriangularPrism) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific triangular prism -/
theorem specific_prism_volume :
  ∀ (prism : TriangularPrism),
    prism.lateral_face_area = 4 →
    prism.distance_to_face = 2 →
    volume prism = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_specific_prism_volume_l1757_175717


namespace NUMINAMATH_CALUDE_f_of_five_equals_ln_five_l1757_175753

-- Define the function f
noncomputable def f : ℝ → ℝ := fun y ↦ Real.log y

-- State the theorem
theorem f_of_five_equals_ln_five :
  (∀ x : ℝ, f (Real.exp x) = x) → f 5 = Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_ln_five_l1757_175753


namespace NUMINAMATH_CALUDE_square_root_real_range_l1757_175738

theorem square_root_real_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 3 + x) → x ≥ -3 := by
sorry

end NUMINAMATH_CALUDE_square_root_real_range_l1757_175738


namespace NUMINAMATH_CALUDE_calculation_proof_l1757_175794

theorem calculation_proof : Real.sqrt 4 + |3 - π| + (1/3)⁻¹ = 2 + π := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1757_175794


namespace NUMINAMATH_CALUDE_first_player_wins_l1757_175708

/-- Represents the state of the game -/
structure GameState where
  chips : Nat
  lastMove : Option Nat

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Nat) : Prop :=
  1 ≤ move ∧ move ≤ 9 ∧ move ≤ state.chips ∧ state.lastMove ≠ some move

/-- Represents a winning strategy for the first player -/
def hasWinningStrategy (initialChips : Nat) : Prop :=
  ∃ (strategy : GameState → Nat),
    ∀ (state : GameState),
      state.chips ≤ initialChips →
      (isValidMove state (strategy state) →
        ¬∃ (opponentMove : Nat), isValidMove { chips := state.chips - strategy state, lastMove := some (strategy state) } opponentMove)

/-- The main theorem stating that the first player has a winning strategy for 110 chips -/
theorem first_player_wins : hasWinningStrategy 110 := by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l1757_175708


namespace NUMINAMATH_CALUDE_sequence_formula_smallest_m_bound_l1757_175799

def S (n : ℕ) : ℚ := 3/2 * n^2 - 1/2 * n

def a (n : ℕ+) : ℚ := 3 * n - 2

def T (n : ℕ+) : ℚ := 1 - 1 / (3 * n + 1)

theorem sequence_formula (n : ℕ+) : a n = 3 * n - 2 :=
sorry

theorem smallest_m_bound : 
  ∃ m : ℕ, (∀ n : ℕ+, T n < m / 20) ∧ (∀ k : ℕ, k < m → ∃ n : ℕ+, T n ≥ k / 20) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_smallest_m_bound_l1757_175799


namespace NUMINAMATH_CALUDE_fibonacci_sum_convergence_l1757_175718

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of F_n / 2^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (2 ^ n)

theorem fibonacci_sum_convergence : fibSum = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_convergence_l1757_175718


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1757_175715

/-- The number of fish initially tagged and returned to the pond -/
def tagged_fish : ℕ := 50

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish found in the second catch -/
def tagged_in_second_catch : ℕ := 2

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1250

/-- Theorem stating that the given conditions lead to the correct total number of fish -/
theorem fish_population_estimate :
  (tagged_in_second_catch : ℚ) / second_catch = tagged_fish / total_fish :=
by sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l1757_175715


namespace NUMINAMATH_CALUDE_range_of_m_l1757_175757

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≠ 0) →  -- negation of q
  (abs (m + 1) ≤ 2) →               -- p
  (-1 < m ∧ m < 1) := by            -- conclusion
sorry

end NUMINAMATH_CALUDE_range_of_m_l1757_175757


namespace NUMINAMATH_CALUDE_f_m_plus_n_eq_zero_l1757_175772

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (abs x) + Real.log ((2019 - x) / (2019 + x))

theorem f_m_plus_n_eq_zero 
  (m n : ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-2018 : ℝ) 2018, m < f x ∧ f x < n) 
  (h2 : ∀ x : ℝ, x ∉ Set.Icc (-2018 : ℝ) 2018 → ¬(m < f x ∧ f x < n)) :
  f (m + n) = 0 := by
sorry

end NUMINAMATH_CALUDE_f_m_plus_n_eq_zero_l1757_175772


namespace NUMINAMATH_CALUDE_max_sum_of_squared_distances_l1757_175765

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def sum_of_squared_distances (a b c d : E) : ℝ :=
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2

theorem max_sum_of_squared_distances (a b c d : E) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) : 
  sum_of_squared_distances a b c d ≤ 16 ∧ 
  ∃ (a' b' c' d' : E), sum_of_squared_distances a' b' c' d' = 16 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_squared_distances_l1757_175765


namespace NUMINAMATH_CALUDE_probability_both_red_correct_l1757_175774

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def blue_balls : ℕ := 4
def green_balls : ℕ := 2
def balls_picked : ℕ := 2

def probability_both_red : ℚ := 2 / 15

theorem probability_both_red_correct :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked : ℚ) = probability_both_red :=
sorry

end NUMINAMATH_CALUDE_probability_both_red_correct_l1757_175774


namespace NUMINAMATH_CALUDE_smoothie_ratio_l1757_175776

/-- Given two juices P and V, and two smoothies A and Y, prove that the ratio of P to V in smoothie A is 4:1 -/
theorem smoothie_ratio :
  -- Total amounts of juices
  ∀ (total_p total_v : ℚ),
  -- Amounts in smoothie A
  ∀ (a_p a_v : ℚ),
  -- Amounts in smoothie Y
  ∀ (y_p y_v : ℚ),
  -- Conditions
  total_p = 24 →
  total_v = 25 →
  a_p = 20 →
  total_p = a_p + y_p →
  total_v = a_v + y_v →
  y_p * 5 = y_v →
  -- Conclusion
  a_p * 1 = a_v * 4 :=
by sorry

end NUMINAMATH_CALUDE_smoothie_ratio_l1757_175776


namespace NUMINAMATH_CALUDE_gym_income_calculation_l1757_175732

/-- A gym charges its members a certain amount twice a month and has a fixed number of members. -/
structure Gym where
  charge_per_half_month : ℕ
  number_of_members : ℕ

/-- Calculate the monthly income of a gym -/
def monthly_income (g : Gym) : ℕ :=
  g.charge_per_half_month * 2 * g.number_of_members

/-- Theorem: A gym that charges $18 twice a month and has 300 members makes $10,800 per month -/
theorem gym_income_calculation :
  let g : Gym := { charge_per_half_month := 18, number_of_members := 300 }
  monthly_income g = 10800 := by
  sorry

end NUMINAMATH_CALUDE_gym_income_calculation_l1757_175732


namespace NUMINAMATH_CALUDE_color_film_fraction_l1757_175728

theorem color_film_fraction (x y : ℝ) (h1 : x ≠ 0) : 
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (1 / 100) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected : ℝ) = 6 / 31 := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l1757_175728


namespace NUMINAMATH_CALUDE_no_quadratic_with_discriminant_2019_l1757_175741

theorem no_quadratic_with_discriminant_2019 : ¬ ∃ (a b c : ℤ), b^2 - 4*a*c = 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_with_discriminant_2019_l1757_175741


namespace NUMINAMATH_CALUDE_min_students_in_class_l1757_175760

theorem min_students_in_class (n b g : ℕ) : 
  n ≡ 2 [MOD 5] →
  (3 * g : ℕ) = (2 * b : ℕ) →
  n = b + g →
  n ≥ 57 ∧ (∀ m : ℕ, m < 57 → ¬(m ≡ 2 [MOD 5] ∧ ∃ b' g' : ℕ, (3 * g' : ℕ) = (2 * b' : ℕ) ∧ m = b' + g')) :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_class_l1757_175760


namespace NUMINAMATH_CALUDE_largest_zero_S_l1757_175758

def S : ℕ → ℤ
  | 0 => 0
  | n + 1 => S n + (n + 1) * (if S n < n + 1 then 1 else -1)

theorem largest_zero_S : ∃ k : ℕ, k ≤ 2010 ∧ S k = 0 ∧ ∀ m : ℕ, m ≤ 2010 ∧ m > k → S m ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_zero_S_l1757_175758


namespace NUMINAMATH_CALUDE_quiz_goal_achievement_l1757_175767

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 3/4 →
  completed_quizzes = 40 →
  current_as = 27 →
  (total_quizzes - completed_quizzes) - 
    (↑(total_quizzes) * goal_percentage - current_as).ceil = 2 := by
  sorry

end NUMINAMATH_CALUDE_quiz_goal_achievement_l1757_175767


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1757_175747

/-- Represents the time for a train to cross a bridge given its parameters -/
theorem train_bridge_crossing_time 
  (L : ℝ) -- Length of the train
  (u : ℝ) -- Initial speed of the train
  (a : ℝ) -- Constant acceleration of the train
  (t : ℝ) -- Time to cross the signal post
  (B : ℝ) -- Length of the bridge
  (h1 : L > 0) -- Train has positive length
  (h2 : u ≥ 0) -- Initial speed is non-negative
  (h3 : a > 0) -- Acceleration is positive
  (h4 : t > 0) -- Time to cross signal post is positive
  (h5 : B > 0) -- Bridge has positive length
  (h6 : L = u * t + (1/2) * a * t^2) -- Equation for crossing signal post
  : ∃ T, T > 0 ∧ B + L = u * T + (1/2) * a * T^2 :=
sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1757_175747


namespace NUMINAMATH_CALUDE_binomial_difference_divisibility_l1757_175777

theorem binomial_difference_divisibility (p k : ℕ) (h_prime : Nat.Prime p) (h_k_lower : 2 ≤ k) (h_k_upper : k ≤ p - 2) :
  ∃ m : ℤ, (Nat.choose (p - k + 1) k : ℤ) - (Nat.choose (p - k - 1) (k - 2) : ℤ) = m * p := by
  sorry

end NUMINAMATH_CALUDE_binomial_difference_divisibility_l1757_175777


namespace NUMINAMATH_CALUDE_no_two_right_angles_l1757_175780

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of a triangle
def Triangle.sumOfAngles (t : Triangle) : ℝ := t.angle1 + t.angle2 + t.angle3
def Triangle.isRight (t : Triangle) : Prop := t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Theorem: A triangle cannot have two right angles
theorem no_two_right_angles (t : Triangle) : 
  t.sumOfAngles = 180 → ¬(t.angle1 = 90 ∧ t.angle2 = 90) :=
by
  sorry


end NUMINAMATH_CALUDE_no_two_right_angles_l1757_175780


namespace NUMINAMATH_CALUDE_milk_production_per_cow_l1757_175736

theorem milk_production_per_cow 
  (num_cows : ℕ) 
  (milk_price : ℚ) 
  (butter_ratio : ℕ) 
  (butter_price : ℚ) 
  (num_customers : ℕ) 
  (milk_per_customer : ℕ) 
  (total_earnings : ℚ) 
  (h1 : num_cows = 12)
  (h2 : milk_price = 3)
  (h3 : butter_ratio = 2)
  (h4 : butter_price = 3/2)
  (h5 : num_customers = 6)
  (h6 : milk_per_customer = 6)
  (h7 : total_earnings = 144) :
  (total_earnings / num_cows : ℚ) / milk_price = 4 := by
sorry

end NUMINAMATH_CALUDE_milk_production_per_cow_l1757_175736


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1757_175762

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (3 * X^5 + 15 * X^4 - 42 * X^3 - 60 * X^2 + 48 * X - 47) = 
  (X^3 + 7 * X^2 + 5 * X - 5) * q + (3 * X - 47) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1757_175762


namespace NUMINAMATH_CALUDE_difference_of_squares_l1757_175781

theorem difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1757_175781


namespace NUMINAMATH_CALUDE_bruce_shopping_theorem_l1757_175710

/-- Calculates the remaining money after Bruce's shopping trip. -/
def remaining_money (initial_amount : ℕ) (shirt_price : ℕ) (num_shirts : ℕ) (pants_price : ℕ) : ℕ :=
  initial_amount - (shirt_price * num_shirts + pants_price)

/-- Theorem stating that Bruce has $20 left after his shopping trip. -/
theorem bruce_shopping_theorem :
  remaining_money 71 5 5 26 = 20 := by
  sorry

#eval remaining_money 71 5 5 26

end NUMINAMATH_CALUDE_bruce_shopping_theorem_l1757_175710


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1757_175763

theorem sqrt_expression_equality : 
  Real.sqrt 8 - 2 * Real.sqrt 18 + Real.sqrt 24 = -4 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1757_175763


namespace NUMINAMATH_CALUDE_complete_square_sum_l1757_175764

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 8*x + 8 = 0 ↔ (x + b)^2 = c) → b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1757_175764


namespace NUMINAMATH_CALUDE_middle_term_value_l1757_175714

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → a j - a i = a k - a j

-- Define our specific sequence
def our_sequence (n : ℕ) : ℝ :=
  match n with
  | 0 => 23
  | 1 => 0  -- x (unknown)
  | 2 => 0  -- y (to be proven)
  | 3 => 0  -- z (unknown)
  | 4 => 47
  | _ => 0  -- other terms are not relevant

-- State the theorem
theorem middle_term_value :
  is_arithmetic_sequence our_sequence →
  our_sequence 2 = 35 :=
by sorry

end NUMINAMATH_CALUDE_middle_term_value_l1757_175714


namespace NUMINAMATH_CALUDE_justin_tim_games_l1757_175755

theorem justin_tim_games (n : ℕ) (k : ℕ) (total_players : ℕ) (justin tim : Fin total_players) :
  n = 12 →
  k = 6 →
  total_players = n →
  justin ≠ tim →
  (Nat.choose n k : ℚ) * k / n = 210 :=
sorry

end NUMINAMATH_CALUDE_justin_tim_games_l1757_175755


namespace NUMINAMATH_CALUDE_constant_expression_theorem_l1757_175734

theorem constant_expression_theorem (x y m n : ℝ) :
  (∀ y, (x + y) * (x - 2*y) - m*y*(n*x - y) = 25) →
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) := by sorry

end NUMINAMATH_CALUDE_constant_expression_theorem_l1757_175734


namespace NUMINAMATH_CALUDE_three_true_inequalities_l1757_175743

theorem three_true_inequalities
  (x y a b : ℝ)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hx : x^2 < a^2)
  (hy : y^2 < b^2) :
  (x^2 + y^2 < a^2 + b^2) ∧
  (x^2 * y^2 < a^2 * b^2) ∧
  (x^2 / y^2 < a^2 / b^2) ∧
  ¬(∀ x y a b, x > 0 → y > 0 → a > 0 → b > 0 → x^2 < a^2 → y^2 < b^2 → x^2 - y^2 < a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_three_true_inequalities_l1757_175743


namespace NUMINAMATH_CALUDE_store_holiday_customers_l1757_175796

/-- The number of customers a store sees during holiday season -/
def holiday_customers (regular_rate : ℕ) (hours : ℕ) : ℕ :=
  2 * regular_rate * hours

/-- Theorem: Given the regular customer rate and time period, 
    the store will see 2800 customers during the holiday season -/
theorem store_holiday_customers :
  holiday_customers 175 8 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_store_holiday_customers_l1757_175796


namespace NUMINAMATH_CALUDE_range_of_a_l1757_175749

theorem range_of_a (x y a : ℝ) (h1 : x + y + 3 = x * y) (h2 : x > 0) (h3 : y > 0)
  (h4 : ∀ x y : ℝ, x > 0 → y > 0 → x + y + 3 = x * y → (x + y)^2 - a*(x + y) + 1 ≥ 0) :
  a ≤ 37/6 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1757_175749


namespace NUMINAMATH_CALUDE_river_flow_speed_l1757_175792

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey --/
theorem river_flow_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 6)
  (h2 : distance = 64)
  (h3 : total_time = 24) :
  ∃ (v : ℝ), v = 2 ∧ 
  (distance / (boat_speed - v) + distance / (boat_speed + v) = total_time) :=
by
  sorry

#check river_flow_speed

end NUMINAMATH_CALUDE_river_flow_speed_l1757_175792


namespace NUMINAMATH_CALUDE_science_quiz_passing_requirement_l1757_175727

theorem science_quiz_passing_requirement (total_questions physics_questions chemistry_questions biology_questions : ℕ)
  (physics_correct_percent chemistry_correct_percent biology_correct_percent passing_percent : ℚ) :
  total_questions = 100 →
  physics_questions = 20 →
  chemistry_questions = 40 →
  biology_questions = 40 →
  physics_correct_percent = 80 / 100 →
  chemistry_correct_percent = 50 / 100 →
  biology_correct_percent = 70 / 100 →
  passing_percent = 65 / 100 →
  (passing_percent * total_questions).ceil -
    (physics_correct_percent * physics_questions +
     chemistry_correct_percent * chemistry_questions +
     biology_correct_percent * biology_questions) = 1 := by
  sorry

end NUMINAMATH_CALUDE_science_quiz_passing_requirement_l1757_175727


namespace NUMINAMATH_CALUDE_problem_statement_l1757_175797

theorem problem_statement (a b c d : ℤ) 
  (h1 : a - b - c + d = 13) 
  (h2 : a + b - c - d = 5) : 
  (b - d)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1757_175797


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1757_175733

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The ratio of the angles is 4:1
  a = 72 :=     -- The larger angle is 72°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1757_175733


namespace NUMINAMATH_CALUDE_power_of_81_l1757_175724

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_power_of_81_l1757_175724


namespace NUMINAMATH_CALUDE_concert_attendance_l1757_175782

-- Define the total number of people at the concert
variable (P : ℕ)

-- Define the conditions
def second_band_audience : ℚ := 2/3
def first_band_audience : ℚ := 1/3
def under_30_second_band : ℚ := 1/2
def women_under_30_second_band : ℚ := 3/5
def men_under_30_second_band : ℕ := 20

-- Theorem statement
theorem concert_attendance : 
  second_band_audience + first_band_audience = 1 →
  (second_band_audience * under_30_second_band * (1 - women_under_30_second_band)) * P = men_under_30_second_band →
  P = 150 :=
by sorry

end NUMINAMATH_CALUDE_concert_attendance_l1757_175782


namespace NUMINAMATH_CALUDE_susans_chairs_l1757_175778

theorem susans_chairs (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = 4 * red)
  (h2 : blue = yellow - 2)
  (h3 : red + yellow + blue = 43) :
  red = 5 := by
  sorry

end NUMINAMATH_CALUDE_susans_chairs_l1757_175778


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1757_175779

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2))))

-- State the theorem
theorem bowtie_equation_solution :
  ∀ x : ℝ, bowtie 3 x = 15 → x = 2 * Real.sqrt 33 ∨ x = -2 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1757_175779


namespace NUMINAMATH_CALUDE_forall_op_example_l1757_175731

-- Define the new operation ∀
def forall_op (a b : ℚ) : ℚ := -a - b^2

-- Theorem statement
theorem forall_op_example : forall_op (forall_op 2022 1) 2 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_forall_op_example_l1757_175731


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l1757_175739

theorem trigonometric_inequalities (x : Real) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (1 - Real.cos x ≤ x^2 / 2) ∧
  (x * Real.cos x ≤ Real.sin x ∧ Real.sin x ≤ x * Real.cos (x / 2)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l1757_175739


namespace NUMINAMATH_CALUDE_connie_markers_total_l1757_175725

/-- The total number of markers Connie has is 3343, given that she has 2315 red markers and 1028 blue markers. -/
theorem connie_markers_total : 
  let red_markers : ℕ := 2315
  let blue_markers : ℕ := 1028
  red_markers + blue_markers = 3343 := by sorry

end NUMINAMATH_CALUDE_connie_markers_total_l1757_175725


namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l1757_175769

/-- Prove that the fraction of books sold is 2/3, given the conditions -/
theorem fraction_of_books_sold (price : ℝ) (unsold : ℕ) (total_received : ℝ) :
  price = 4.25 →
  unsold = 30 →
  total_received = 255 →
  (total_received / price) / ((total_received / price) + unsold) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_books_sold_l1757_175769


namespace NUMINAMATH_CALUDE_kenneth_to_micah_ratio_l1757_175702

/-- The number of fish Micah has -/
def micah_fish : ℕ := 7

/-- The number of fish Kenneth has -/
def kenneth_fish : ℕ := 21

/-- The number of fish Matthias has -/
def matthias_fish : ℕ := kenneth_fish - 15

/-- The total number of fish the boys have -/
def total_fish : ℕ := 34

/-- Theorem stating that the ratio of Kenneth's fish to Micah's fish is 3:1 -/
theorem kenneth_to_micah_ratio :
  micah_fish + kenneth_fish + matthias_fish = total_fish →
  kenneth_fish / micah_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_to_micah_ratio_l1757_175702


namespace NUMINAMATH_CALUDE_shorter_train_length_l1757_175775

/-- Calculates the length of the shorter train given the speeds of two trains,
    the time they take to cross each other, and the length of the longer train. -/
theorem shorter_train_length
  (speed1 : ℝ) (speed2 : ℝ) (crossing_time : ℝ) (longer_train_length : ℝ)
  (h1 : speed1 = 60) -- km/hr
  (h2 : speed2 = 40) -- km/hr
  (h3 : crossing_time = 10.799136069114471) -- seconds
  (h4 : longer_train_length = 160) -- meters
  : ∃ (shorter_train_length : ℝ),
    shorter_train_length = 140 ∧ 
    shorter_train_length = 
      (speed1 + speed2) * (5 / 18) * crossing_time - longer_train_length :=
by
  sorry

end NUMINAMATH_CALUDE_shorter_train_length_l1757_175775


namespace NUMINAMATH_CALUDE_ticket_price_is_three_l1757_175752

/-- Represents an amusement park's weekly operations and revenue -/
structure AmusementPark where
  regularDays : Nat
  regularVisitors : Nat
  specialDay1Visitors : Nat
  specialDay2Visitors : Nat
  weeklyRevenue : Nat

/-- Calculates the ticket price given the park's weekly data -/
def calculateTicketPrice (park : AmusementPark) : Rat :=
  park.weeklyRevenue / (park.regularDays * park.regularVisitors + park.specialDay1Visitors + park.specialDay2Visitors)

/-- Theorem stating that the ticket price is $3 given the specific conditions -/
theorem ticket_price_is_three (park : AmusementPark) 
  (h1 : park.regularDays = 5)
  (h2 : park.regularVisitors = 100)
  (h3 : park.specialDay1Visitors = 200)
  (h4 : park.specialDay2Visitors = 300)
  (h5 : park.weeklyRevenue = 3000) :
  calculateTicketPrice park = 3 := by
  sorry

#eval calculateTicketPrice { 
  regularDays := 5, 
  regularVisitors := 100, 
  specialDay1Visitors := 200, 
  specialDay2Visitors := 300, 
  weeklyRevenue := 3000 
}

end NUMINAMATH_CALUDE_ticket_price_is_three_l1757_175752


namespace NUMINAMATH_CALUDE_outfit_count_l1757_175790

/-- The number of distinct outfits that can be made -/
def number_of_outfits (red_shirts green_shirts pants blue_hats red_hats : ℕ) : ℕ :=
  (red_shirts * pants * blue_hats) + (green_shirts * pants * red_hats)

/-- Theorem stating the number of outfits under the given conditions -/
theorem outfit_count :
  number_of_outfits 6 7 9 10 10 = 1170 :=
by sorry

end NUMINAMATH_CALUDE_outfit_count_l1757_175790


namespace NUMINAMATH_CALUDE_intersection_A_B_l1757_175748

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1757_175748


namespace NUMINAMATH_CALUDE_perimeter_ABF₂_is_24_l1757_175787

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 25 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse that intersect with F₁
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume A and B are on the ellipse
axiom A_on_ellipse : ellipse A.1 A.2
axiom B_on_ellipse : ellipse B.1 B.2

-- Assume F₁ intersects the ellipse at A and B
axiom F₁_intersect_A : F₁ = A
axiom F₁_intersect_B : F₁ = B

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of triangle ABF₂
def perimeter_ABF₂ : ℝ := distance A F₂ + distance B F₂ + distance A B

-- Theorem: The perimeter of triangle ABF₂ is 24
theorem perimeter_ABF₂_is_24 : perimeter_ABF₂ = 24 := by sorry

end NUMINAMATH_CALUDE_perimeter_ABF₂_is_24_l1757_175787


namespace NUMINAMATH_CALUDE_loom_weaving_rate_l1757_175716

/-- The rate at which an industrial loom weaves cloth, given the amount of cloth woven and the time taken. -/
theorem loom_weaving_rate (cloth_woven : Real) (time_taken : Real) (h : cloth_woven = 25 ∧ time_taken = 195.3125) :
  cloth_woven / time_taken = 0.128 := by
  sorry

end NUMINAMATH_CALUDE_loom_weaving_rate_l1757_175716


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l1757_175729

theorem consecutive_odd_numbers_problem :
  ∀ x y z : ℤ,
  (y = x + 2) →
  (z = x + 4) →
  (8 * x = 3 * z + 2 * y + 5) →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l1757_175729


namespace NUMINAMATH_CALUDE_inflation_time_is_20_l1757_175773

/-- The time it takes to inflate one soccer ball -/
def inflation_time : ℕ := sorry

/-- The number of balls Alexia inflates -/
def alexia_balls : ℕ := 20

/-- The number of balls Ermias inflates -/
def ermias_balls : ℕ := alexia_balls + 5

/-- The total time taken to inflate all balls -/
def total_time : ℕ := 900

/-- Theorem stating that the inflation time for one ball is 20 minutes -/
theorem inflation_time_is_20 : inflation_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_inflation_time_is_20_l1757_175773


namespace NUMINAMATH_CALUDE_apple_distribution_l1757_175703

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of additional apples He has compared to Adam and Jackie together -/
def he_additional : ℕ := 9

/-- The number of additional apples Adam has compared to Jackie -/
def adam_additional : ℕ := 8

/-- The number of apples Adam has -/
def adam : ℕ := sorry

/-- The number of apples Jackie has -/
def jackie : ℕ := sorry

/-- The number of apples He has -/
def he : ℕ := sorry

theorem apple_distribution :
  (adam + jackie = total_adam_jackie) ∧
  (he = adam + jackie + he_additional) ∧
  (adam = jackie + adam_additional) →
  he = 21 := by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1757_175703


namespace NUMINAMATH_CALUDE_robins_gum_problem_l1757_175788

/-- Given that Robin initially had 18 pieces of gum and now has 44 pieces in total,
    prove that Robin's brother gave her 26 pieces of gum. -/
theorem robins_gum_problem (initial : ℕ) (total : ℕ) (h1 : initial = 18) (h2 : total = 44) :
  total - initial = 26 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_problem_l1757_175788


namespace NUMINAMATH_CALUDE_function_periodicity_l1757_175707

def is_periodic (f : ℕ → ℕ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, f (n + p) = f n

theorem function_periodicity 
  (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f (n + f n) = f n) 
  (h2 : Set.Finite (Set.range f)) : 
  is_periodic f := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l1757_175707


namespace NUMINAMATH_CALUDE_range_m_when_p_necessary_for_q_range_m_when_not_p_necessary_not_sufficient_for_not_q_l1757_175700

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := 1 - m^2 ≤ x ∧ x ≤ 1 + m^2

-- Theorem 1: If p is a necessary condition for q, then the range of m is [-√3, √3]
theorem range_m_when_p_necessary_for_q :
  (∀ x m : ℝ, q x m → p x) →
  ∀ m : ℝ, -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

-- Theorem 2: If ¬p is a necessary but not sufficient condition for ¬q, 
-- then the range of m is (-∞, -3] ∪ [3, +∞)
theorem range_m_when_not_p_necessary_not_sufficient_for_not_q :
  (∀ x m : ℝ, ¬(q x m) → ¬(p x)) ∧ 
  (∃ x m : ℝ, ¬(p x) ∧ q x m) →
  ∀ m : ℝ, m ≤ -3 ∨ m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_m_when_p_necessary_for_q_range_m_when_not_p_necessary_not_sufficient_for_not_q_l1757_175700


namespace NUMINAMATH_CALUDE_weeks_to_afford_bike_l1757_175701

/-- The cost of the bike in dollars -/
def bike_cost : ℕ := 600

/-- The amount of birthday money Chandler received in dollars -/
def birthday_money : ℕ := 150

/-- Chandler's weekly earnings from tutoring in dollars -/
def weekly_earnings : ℕ := 14

/-- The function that calculates the total money Chandler has after working for a given number of weeks -/
def total_money (weeks : ℕ) : ℕ := birthday_money + weekly_earnings * weeks

/-- The theorem stating that 33 is the smallest number of weeks Chandler needs to work to afford the bike -/
theorem weeks_to_afford_bike : 
  (∀ w : ℕ, w < 33 → total_money w < bike_cost) ∧ 
  total_money 33 ≥ bike_cost := by
sorry

end NUMINAMATH_CALUDE_weeks_to_afford_bike_l1757_175701


namespace NUMINAMATH_CALUDE_chewing_gum_revenue_projection_l1757_175793

theorem chewing_gum_revenue_projection (R : ℝ) (h : R > 0) :
  let projected_revenue := 1.40 * R
  let actual_revenue := 0.70 * R
  actual_revenue / projected_revenue = 0.50 := by
sorry

end NUMINAMATH_CALUDE_chewing_gum_revenue_projection_l1757_175793


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1757_175721

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0) ↔ 
  (m < (1 : ℝ) / 5 ∧ m ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1757_175721


namespace NUMINAMATH_CALUDE_inequality_proof_l1757_175720

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b / 3 + c ≥ Real.sqrt ((a + b) * b * (c + a)) ∧
  Real.sqrt ((a + b) * b * (c + a)) ≥ Real.sqrt (a * b) + (Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1757_175720


namespace NUMINAMATH_CALUDE_u_equals_fib_squared_l1757_175726

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def u : ℕ → ℤ
  | 0 => 4
  | 1 => 9
  | n + 2 => 3 * u (n + 1) - u n - 2 * (-1 : ℤ) ^ (n + 2)

theorem u_equals_fib_squared (n : ℕ) : u n = (fib (n + 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_u_equals_fib_squared_l1757_175726


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l1757_175783

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (-1, 2, -3)
  A₂ : ℝ × ℝ × ℝ := (4, -1, 0)
  A₃ : ℝ × ℝ × ℝ := (2, 1, -2)
  A₄ : ℝ × ℝ × ℝ := (3, 4, 5)

/-- Calculate the volume of the tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := by sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedronHeight (t : Tetrahedron) : ℝ := by sorry

/-- Main theorem: Volume and height of the tetrahedron -/
theorem tetrahedron_volume_and_height (t : Tetrahedron) :
  tetrahedronVolume t = 20 / 3 ∧ tetrahedronHeight t = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l1757_175783


namespace NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l1757_175704

theorem complex_exp_13pi_div_2 : Complex.exp (13 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l1757_175704


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1757_175711

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ 6) ∧ (∃ x ∈ Set.Icc 1 3, f a x = 6) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l1757_175711


namespace NUMINAMATH_CALUDE_ratio_x_to_w_l1757_175713

/-- Given the relationships between x, y, z, and w, prove that the ratio of x to w is 0.486 -/
theorem ratio_x_to_w (x y z w : ℝ) 
  (h1 : x = 1.20 * y)
  (h2 : y = 0.30 * z)
  (h3 : z = 1.35 * w) :
  x / w = 0.486 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_w_l1757_175713


namespace NUMINAMATH_CALUDE_swimming_running_speed_ratio_l1757_175789

/-- Proves that given the specified conditions, the ratio of running speed to swimming speed is 4 -/
theorem swimming_running_speed_ratio :
  ∀ (swimming_speed swimming_time running_time total_distance : ℝ),
  swimming_speed = 2 →
  swimming_time = 2 →
  running_time = swimming_time / 2 →
  total_distance = 12 →
  total_distance = swimming_speed * swimming_time + running_time * (total_distance - swimming_speed * swimming_time) / running_time →
  (total_distance - swimming_speed * swimming_time) / running_time / swimming_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_swimming_running_speed_ratio_l1757_175789


namespace NUMINAMATH_CALUDE_average_b_c_l1757_175730

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45) 
  (h2 : c - a = 30) : 
  (b + c) / 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l1757_175730


namespace NUMINAMATH_CALUDE_elena_garden_lilies_l1757_175768

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of petals in Elena's garden -/
def total_petals : ℕ := 63

theorem elena_garden_lilies :
  num_lilies * petals_per_lily + num_tulips * petals_per_tulip = total_petals :=
by sorry

end NUMINAMATH_CALUDE_elena_garden_lilies_l1757_175768


namespace NUMINAMATH_CALUDE_min_value_of_f_l1757_175705

/-- The quadratic function we're minimizing -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 9

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ -15) ∧ (∃ x y : ℝ, f x y = -15) := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1757_175705


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_perimeter_range_l1757_175751

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given equation
def given_equation (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

-- Theorem 1: Prove that A = 60°
theorem angle_A_is_60_degrees (t : Triangle) (h : given_equation t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the range of perimeters when a = 7
theorem perimeter_range (t : Triangle) (h1 : given_equation t) (h2 : t.a = 7) :
  14 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_perimeter_range_l1757_175751


namespace NUMINAMATH_CALUDE_sector_central_angle_l1757_175706

theorem sector_central_angle (perimeter area : ℝ) (h1 : perimeter = 10) (h2 : area = 4) :
  ∃ (r l θ : ℝ), r > 0 ∧ l > 0 ∧ θ > 0 ∧ 
  2 * r + l = perimeter ∧ 
  1/2 * r * l = area ∧ 
  θ = l / r ∧ 
  θ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1757_175706


namespace NUMINAMATH_CALUDE_complement_M_in_U_l1757_175756

-- Define the universal set U
def U : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the set M
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem complement_M_in_U : 
  (U \ M) = {x | 1 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l1757_175756


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l1757_175744

def base_three_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_ten [2, 0, 1, 2, 1] = 178 := by sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l1757_175744


namespace NUMINAMATH_CALUDE_inequality_always_holds_l1757_175750

theorem inequality_always_holds (α : ℝ) : 4 * Real.sin (3 * α) + 5 ≥ 4 * Real.cos (2 * α) + 5 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l1757_175750


namespace NUMINAMATH_CALUDE_symmetric_points_m_value_l1757_175766

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the origin -/
def symmetricAboutOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetric_points_m_value :
  let p : Point := ⟨2, -1⟩
  let q : Point := ⟨-2, m⟩
  symmetricAboutOrigin p q → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_m_value_l1757_175766


namespace NUMINAMATH_CALUDE_max_regions_theorem_l1757_175761

/-- Maximum number of regions formed by n lines in R^2 -/
def max_regions_lines (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Maximum number of regions formed by n planes in R^3 -/
def max_regions_planes (n : ℕ) : ℕ := (n^3 + 5*n) / 6 + 1

/-- Maximum number of regions formed by n circles in R^2 -/
def max_regions_circles (n : ℕ) : ℕ := (n - 1) * n + 2

/-- Maximum number of regions formed by n spheres in R^3 -/
def max_regions_spheres (n : ℕ) : ℕ := n * (n^2 - 3*n + 8) / 3

theorem max_regions_theorem :
  ∀ n : ℕ,
  (max_regions_lines n = n * (n + 1) / 2 + 1) ∧
  (max_regions_planes n = (n^3 + 5*n) / 6 + 1) ∧
  (max_regions_circles n = (n - 1) * n + 2) ∧
  (max_regions_spheres n = n * (n^2 - 3*n + 8) / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_regions_theorem_l1757_175761


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1757_175795

/-- A sequence (a, b, c) is geometric if there exists a non-zero real number r such that b = a * r and c = b * r. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- The condition ac = b^2 is necessary but not sufficient for a, b, c to form a geometric sequence. -/
theorem geometric_sequence_condition (a b c : ℝ) :
  (IsGeometricSequence a b c → a * c = b ^ 2) ∧
  ¬(a * c = b ^ 2 → IsGeometricSequence a b c) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1757_175795


namespace NUMINAMATH_CALUDE_competition_participants_l1757_175737

theorem competition_participants (initial : ℕ) 
  (h1 : initial * 40 / 100 * 50 / 100 * 25 / 100 = 15) : 
  initial = 300 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l1757_175737


namespace NUMINAMATH_CALUDE_functional_equation_defined_everywhere_l1757_175770

noncomputable def f (c : ℝ) : ℝ → ℝ :=
  λ x => if x = 0 then c
         else if x = 1 then 3 - 2*c
         else (-x^3 + 3*x^2 + 2) / (3*x*(1-x))

theorem functional_equation (c : ℝ) :
  ∀ x : ℝ, x ≠ 0 → f c x + 2 * f c ((x - 1) / x) = 3 * x :=
by sorry

theorem defined_everywhere (c : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, f c x = y :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_defined_everywhere_l1757_175770


namespace NUMINAMATH_CALUDE_initial_withdrawal_l1757_175786

theorem initial_withdrawal (initial_balance : ℚ) : 
  let remaining_balance := initial_balance - (2/5) * initial_balance
  let final_balance := remaining_balance + (1/4) * remaining_balance
  final_balance = 750 →
  (2/5) * initial_balance = 400 := by
sorry

end NUMINAMATH_CALUDE_initial_withdrawal_l1757_175786


namespace NUMINAMATH_CALUDE_rectangular_yard_area_l1757_175798

theorem rectangular_yard_area (L W : ℝ) : 
  L = 40 →  -- One full side (length) is 40 feet
  2 * W + L = 52 →  -- Total fencing for three sides is 52 feet
  L * W = 240 :=  -- Area of the yard is 240 square feet
by
  sorry

end NUMINAMATH_CALUDE_rectangular_yard_area_l1757_175798


namespace NUMINAMATH_CALUDE_mike_work_hours_l1757_175712

theorem mike_work_hours : 
  let wash_time : ℕ := 10  -- minutes to wash a car
  let oil_change_time : ℕ := 15  -- minutes to change oil
  let tire_change_time : ℕ := 30  -- minutes to change a set of tires
  let cars_washed : ℕ := 9  -- number of cars Mike washed
  let oil_changes : ℕ := 6  -- number of oil changes Mike performed
  let tire_changes : ℕ := 2  -- number of tire sets Mike changed
  
  let total_minutes : ℕ := 
    wash_time * cars_washed + 
    oil_change_time * oil_changes + 
    tire_change_time * tire_changes
  
  let total_hours : ℕ := total_minutes / 60

  total_hours = 4 := by sorry

end NUMINAMATH_CALUDE_mike_work_hours_l1757_175712


namespace NUMINAMATH_CALUDE_reporter_average_words_per_minute_l1757_175791

/-- Calculates the average words per minute for a reporter given their pay structure and work conditions. -/
theorem reporter_average_words_per_minute 
  (word_pay : ℝ)
  (article_pay : ℝ)
  (num_articles : ℕ)
  (total_hours : ℝ)
  (hourly_earnings : ℝ)
  (h1 : word_pay = 0.1)
  (h2 : article_pay = 60)
  (h3 : num_articles = 3)
  (h4 : total_hours = 4)
  (h5 : hourly_earnings = 105) :
  (((hourly_earnings * total_hours - article_pay * num_articles) / word_pay) / (total_hours * 60)) = 10 := by
  sorry

#check reporter_average_words_per_minute

end NUMINAMATH_CALUDE_reporter_average_words_per_minute_l1757_175791


namespace NUMINAMATH_CALUDE_next_sunday_rest_l1757_175740

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the length of the work-rest cycle in days -/
def cycleDays : ℕ := 10

/-- Represents the number of consecutive work days -/
def workDays : ℕ := 8

/-- Represents the number of consecutive rest days -/
def restDays : ℕ := 2

/-- Theorem stating that the next Sunday rest day occurs after 7 weeks -/
theorem next_sunday_rest (n : ℕ) : 
  cycleDays * n + restDays - 1 = daysInWeek * 7 := by sorry

end NUMINAMATH_CALUDE_next_sunday_rest_l1757_175740


namespace NUMINAMATH_CALUDE_condition_relations_l1757_175735

-- Define the propositions
variable (A B C D : Prop)

-- Define the given conditions
axiom A_sufficient_D : A → D
axiom B_sufficient_C : B → C
axiom D_necessary_C : C → D
axiom D_sufficient_B : D → B

-- Theorem to prove
theorem condition_relations :
  (C → D) ∧ (A → B) := by sorry

end NUMINAMATH_CALUDE_condition_relations_l1757_175735


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1757_175784

/-- Given a quadratic function g(x) = x^2 + cx + d, prove that if g(g(x) + x) / g(x) = x^2 + 44x + 50, then g(x) = x^2 + 44x + 50 -/
theorem quadratic_function_proof (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = x^2 + c*x + d)
  (h2 : ∀ x, g (g x + x) / g x = x^2 + 44*x + 50) :
  ∀ x, g x = x^2 + 44*x + 50 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1757_175784


namespace NUMINAMATH_CALUDE_green_ball_packs_l1757_175742

/-- Given the number of packs of red and yellow balls, the number of balls per pack,
    and the total number of balls, calculate the number of packs of green balls. -/
theorem green_ball_packs (red_packs yellow_packs : ℕ) (balls_per_pack : ℕ) (total_balls : ℕ) : 
  red_packs = 3 → 
  yellow_packs = 10 → 
  balls_per_pack = 19 → 
  total_balls = 399 → 
  (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack = 8 :=
by sorry

end NUMINAMATH_CALUDE_green_ball_packs_l1757_175742


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1757_175754

theorem gcd_lcm_sum : Nat.gcd 42 56 + Nat.lcm 24 18 = 86 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1757_175754


namespace NUMINAMATH_CALUDE_first_player_seeds_l1757_175785

/-- Given a sunflower seed eating contest with three players, where:
  * The second player eats 53 seeds
  * The third player eats 30 more seeds than the second
  * The total number of seeds eaten is 214
  This theorem proves that the first player eats 78 seeds. -/
theorem first_player_seeds (second_player : ℕ) (third_player : ℕ) (total_seeds : ℕ) :
  second_player = 53 →
  third_player = second_player + 30 →
  total_seeds = 214 →
  total_seeds = second_player + third_player + 78 :=
by sorry

end NUMINAMATH_CALUDE_first_player_seeds_l1757_175785


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1757_175722

theorem modulus_of_complex_fraction (z : ℂ) : 
  z = (2.2 * Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1757_175722


namespace NUMINAMATH_CALUDE_unique_cubic_function_l1757_175719

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

theorem unique_cubic_function (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃ (x : ℝ), f a b x = 5 ∧ ∀ (y : ℝ), f a b y ≤ 5)
  (h3 : ∃ (x : ℝ), f a b x = 1 ∧ ∀ (y : ℝ), f a b y ≥ 1) :
  ∀ (x : ℝ), f a b x = x^3 + 3*x^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_unique_cubic_function_l1757_175719


namespace NUMINAMATH_CALUDE_exp_15pi_over_2_l1757_175709

theorem exp_15pi_over_2 : Complex.exp (15 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_15pi_over_2_l1757_175709


namespace NUMINAMATH_CALUDE_triangle_side_equations_l1757_175746

/-- Given a triangle ABC with point A at (1,3) and medians from A satisfying specific equations,
    prove that the sides of the triangle have the given equations. -/
theorem triangle_side_equations (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let median_to_BC (x y : ℝ) := x - 2*y + 1 = 0
  let median_to_AC (x y : ℝ) := y = 1
  (∃ t : ℝ, median_to_BC (C.1 + t*(B.1 - C.1)) (C.2 + t*(B.2 - C.2))) ∧ 
  (∃ s : ℝ, median_to_AC ((1 + C.1)/2) ((3 + C.2)/2)) →
  (B.2 - 3 = (3 - 1)/(1 - B.1) * (B.1 - 1)) ∧ 
  (C.2 - B.2 = (C.2 - B.2)/(C.1 - B.1) * (C.1 - B.1)) ∧ 
  (C.2 - 3 = (C.2 - 3)/(C.1 - 1) * (C.1 - 1)) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l1757_175746


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1757_175723

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
    (∀ x : ℝ, x ∈ s ↔ |x - 1| = |x - 2| + |x - 3| + |x - 4|) ∧
    (2 ∈ s ∧ 4 ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1757_175723


namespace NUMINAMATH_CALUDE_f_value_at_2_l1757_175745

/-- A function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

/-- Theorem: If f(-2) = 10, then f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1757_175745
