import Mathlib

namespace altitude_of_triangle_on_rectangle_diagonal_l1605_160592

theorem altitude_of_triangle_on_rectangle_diagonal (l : ℝ) (h : l > 0) :
  let w := l * Real.sqrt 2 / 2
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := diagonal * altitude / 2
  triangle_area = rectangle_area →
  altitude = l * Real.sqrt 3 / 3 :=
by sorry

end altitude_of_triangle_on_rectangle_diagonal_l1605_160592


namespace cube_sum_equals_linear_sum_l1605_160557

theorem cube_sum_equals_linear_sum (a b : ℝ) 
  (h : (a / (1 + b)) + (b / (1 + a)) = 1) : 
  a^3 + b^3 = a + b := by sorry

end cube_sum_equals_linear_sum_l1605_160557


namespace outfit_combinations_l1605_160591

def num_shirts : ℕ := 5
def num_pants : ℕ := 6
def restricted_combinations : ℕ := 2

theorem outfit_combinations :
  let total_combinations := num_shirts * num_pants
  let restricted_shirt_combinations := num_pants - restricted_combinations
  let unrestricted_combinations := (num_shirts - 1) * num_pants
  unrestricted_combinations + restricted_shirt_combinations = 28 := by
  sorry

end outfit_combinations_l1605_160591


namespace mortgage_payment_sum_l1605_160544

theorem mortgage_payment_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) (h1 : a₁ = 400) (h2 : r = 2) (h3 : n = 11) :
  a₁ * (1 - r^n) / (1 - r) = 819400 := by
  sorry

end mortgage_payment_sum_l1605_160544


namespace extra_workers_for_deeper_hole_extra_workers_needed_l1605_160599

/-- Represents the number of workers needed for a digging task. -/
def workers_needed (initial_workers : ℕ) (initial_depth : ℕ) (initial_hours : ℕ) 
                   (target_depth : ℕ) (target_hours : ℕ) : ℕ :=
  (initial_workers * initial_hours * target_depth) / (initial_depth * target_hours)

/-- Theorem stating the number of workers needed for the new digging task. -/
theorem extra_workers_for_deeper_hole 
  (initial_workers : ℕ) (initial_depth : ℕ) (initial_hours : ℕ)
  (target_depth : ℕ) (target_hours : ℕ) :
  initial_workers = 45 → 
  initial_depth = 30 → 
  initial_hours = 8 → 
  target_depth = 70 → 
  target_hours = 5 → 
  workers_needed initial_workers initial_depth initial_hours target_depth target_hours = 168 :=
by
  sorry

/-- Calculates the extra workers needed based on the initial and required number of workers. -/
def extra_workers (initial : ℕ) (required : ℕ) : ℕ :=
  required - initial

/-- Theorem stating the number of extra workers needed for the new digging task. -/
theorem extra_workers_needed 
  (initial_workers : ℕ) (required_workers : ℕ) :
  initial_workers = 45 →
  required_workers = 168 →
  extra_workers initial_workers required_workers = 123 :=
by
  sorry

end extra_workers_for_deeper_hole_extra_workers_needed_l1605_160599


namespace lcm_1320_924_l1605_160563

theorem lcm_1320_924 : Nat.lcm 1320 924 = 9240 := by
  sorry

end lcm_1320_924_l1605_160563


namespace apple_purchase_theorem_l1605_160533

/-- Represents the cost in cents for a pack of apples --/
structure ApplePack where
  count : ℕ
  cost : ℕ

/-- Represents a purchase of apple packs --/
structure Purchase where
  pack : ApplePack
  quantity : ℕ

def total_apples (purchases : List Purchase) : ℕ :=
  purchases.foldl (fun acc p => acc + p.pack.count * p.quantity) 0

def total_cost (purchases : List Purchase) : ℕ :=
  purchases.foldl (fun acc p => acc + p.pack.cost * p.quantity) 0

def average_cost (purchases : List Purchase) : ℚ :=
  (total_cost purchases : ℚ) / (total_apples purchases : ℚ)

theorem apple_purchase_theorem (scheme1 scheme2 : ApplePack) 
  (purchase1 purchase2 : Purchase) : 
  scheme1.count = 4 → 
  scheme1.cost = 15 → 
  scheme2.count = 7 → 
  scheme2.cost = 28 → 
  purchase1.pack = scheme2 → 
  purchase1.quantity = 4 → 
  purchase2.pack = scheme1 → 
  purchase2.quantity = 2 → 
  total_cost [purchase1, purchase2] = 142 ∧ 
  average_cost [purchase1, purchase2] = 5.0714 := by
  sorry

end apple_purchase_theorem_l1605_160533


namespace smallest_prime_with_digit_sum_28_l1605_160550

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_28 :
  (is_prime 1999) ∧ 
  (digit_sum 1999 = 28) ∧ 
  (∀ m : ℕ, m < 1999 → (is_prime m ∧ digit_sum m = 28) → False) :=
sorry

end smallest_prime_with_digit_sum_28_l1605_160550


namespace problem_statement_l1605_160532

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x * y ≤ a * b) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (4/a + 1/b ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y < Real.sqrt 2) :=
by sorry

end problem_statement_l1605_160532


namespace union_complement_equality_l1605_160569

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 4}
def N : Finset Nat := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_equality_l1605_160569


namespace scholarship_problem_l1605_160552

theorem scholarship_problem (total_students : ℕ) 
  (full_merit_percent half_merit_percent sports_percent need_based_percent : ℚ)
  (full_merit_and_sports_percent half_merit_and_need_based_percent : ℚ)
  (h1 : total_students = 300)
  (h2 : full_merit_percent = 5 / 100)
  (h3 : half_merit_percent = 10 / 100)
  (h4 : sports_percent = 3 / 100)
  (h5 : need_based_percent = 7 / 100)
  (h6 : full_merit_and_sports_percent = 1 / 100)
  (h7 : half_merit_and_need_based_percent = 2 / 100) :
  ↑total_students - 
  (↑total_students * (full_merit_percent + half_merit_percent + sports_percent + need_based_percent) -
   ↑total_students * (full_merit_and_sports_percent + half_merit_and_need_based_percent)) = 234 := by
  sorry

end scholarship_problem_l1605_160552


namespace shirt_price_change_l1605_160500

theorem shirt_price_change (original_price : ℝ) (decrease_percent : ℝ) : 
  original_price > 0 →
  decrease_percent ≥ 0 →
  (1.15 * original_price) * (1 - decrease_percent / 100) = 97.75 →
  decrease_percent = 0 := by
sorry

end shirt_price_change_l1605_160500


namespace same_function_shifted_possible_same_function_different_variable_power_zero_not_always_one_same_domain_range_not_same_function_l1605_160516

-- Define a function type
def RealFunction := ℝ → ℝ

-- Statement 1
theorem same_function_shifted_possible : 
  ∃ (f : RealFunction), ∀ x : ℝ, f x = f (x + 1) :=
sorry

-- Statement 2
theorem same_function_different_variable (f : RealFunction) :
  ∀ x t : ℝ, f x = f t :=
sorry

-- Statement 3
theorem power_zero_not_always_one :
  ∃ x : ℝ, x^0 ≠ 1 :=
sorry

-- Statement 4
theorem same_domain_range_not_same_function :
  ∃ (f g : RealFunction), (∀ x : ℝ, ∃ y : ℝ, f x = y ∧ g x = y) ∧ f ≠ g :=
sorry

end same_function_shifted_possible_same_function_different_variable_power_zero_not_always_one_same_domain_range_not_same_function_l1605_160516


namespace rectangle_y_value_l1605_160536

theorem rectangle_y_value (y : ℝ) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (6, y), (-2, 2), (6, 2)]
  let length : ℝ := 6 - (-2)
  let height : ℝ := y - 2
  let area : ℝ := 64
  (length * height = area) → y = 10 := by
sorry

end rectangle_y_value_l1605_160536


namespace fraction_to_decimal_l1605_160578

theorem fraction_to_decimal :
  (7 : ℚ) / 12 = 0.5833333333333333333333333333333333 :=
sorry

end fraction_to_decimal_l1605_160578


namespace stratified_sampling_problem_l1605_160590

theorem stratified_sampling_problem (total_students : ℕ) 
  (group1_students : ℕ) (selected_from_group1 : ℕ) (n : ℕ) : 
  total_students = 1230 → 
  group1_students = 480 → 
  selected_from_group1 = 16 → 
  (n : ℚ) / total_students = selected_from_group1 / group1_students → 
  n = 41 := by
sorry

end stratified_sampling_problem_l1605_160590


namespace inequality_proof_l1605_160545

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x*y + y*z + x*z) :=
by sorry

end inequality_proof_l1605_160545


namespace abc_sum_product_l1605_160531

theorem abc_sum_product (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c > 0) :
  a * b + b * c + c * a < 0 := by
  sorry

end abc_sum_product_l1605_160531


namespace intersection_condition_l1605_160570

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → a > -1 := by
  sorry

end intersection_condition_l1605_160570


namespace abs_value_condition_l1605_160530

theorem abs_value_condition (a b : ℝ) (h : ((1 + a * b) / (a + b))^2 < 1) :
  (abs a > 1 ∧ abs b < 1) ∨ (abs a < 1 ∧ abs b > 1) := by
  sorry

end abs_value_condition_l1605_160530


namespace james_oreos_count_l1605_160580

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := sorry

/-- The number of Oreos James has -/
def james_oreos : ℕ := 4 * jordan_oreos + 7

/-- The total number of Oreos -/
def total_oreos : ℕ := 52

theorem james_oreos_count : james_oreos = 43 := by
  sorry

end james_oreos_count_l1605_160580


namespace max_value_f_on_interval_l1605_160507

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x ≤ f c ∧
  f c = 2 :=
sorry

end max_value_f_on_interval_l1605_160507


namespace right_triangle_third_side_l1605_160529

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a = 3 → b = 4 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = Real.sqrt 7 ∨ c = 5 := by
  sorry

end right_triangle_third_side_l1605_160529


namespace cubic_square_fraction_inequality_l1605_160541

theorem cubic_square_fraction_inequality (s r : ℝ) (hs : s > 0) (hr : r > 0) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end cubic_square_fraction_inequality_l1605_160541


namespace sue_answer_for_ben_partner_answer_formula_l1605_160506

/-- Given an initial number, calculate the partner's final answer according to the instructions -/
def partnerAnswer (x : ℤ) : ℤ :=
  (((x + 2) * 3 - 2) * 3)

/-- Theorem stating that for Ben's initial number 6, Sue's answer should be 66 -/
theorem sue_answer_for_ben :
  partnerAnswer 6 = 66 := by sorry

/-- Theorem proving the general formula for the partner's answer -/
theorem partner_answer_formula (x : ℤ) :
  partnerAnswer x = (((x + 2) * 3 - 2) * 3) := by sorry

end sue_answer_for_ben_partner_answer_formula_l1605_160506


namespace balloon_count_l1605_160501

theorem balloon_count (initial_balloons : Real) (friend_balloons : Real) 
  (h1 : initial_balloons = 7.0) 
  (h2 : friend_balloons = 5.0) : 
  initial_balloons + friend_balloons = 12.0 := by
  sorry

end balloon_count_l1605_160501


namespace intersection_implies_a_value_l1605_160566

theorem intersection_implies_a_value (P Q : Set ℕ) (a : ℕ) :
  P = {0, a} →
  Q = {1, 2} →
  (P ∩ Q).Nonempty →
  a = 1 ∨ a = 2 := by
sorry

end intersection_implies_a_value_l1605_160566


namespace cos_x_plus_2y_equals_one_l1605_160587

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
  sorry

end cos_x_plus_2y_equals_one_l1605_160587


namespace square_sum_equation_l1605_160583

theorem square_sum_equation (n m : ℕ) (h : n ^ 2 = (Finset.range (m - 99)).sum (λ i => i + 100)) : n + m = 497 := by
  sorry

end square_sum_equation_l1605_160583


namespace euler_totient_equation_solutions_l1605_160594

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem euler_totient_equation_solutions (a b : ℕ) :
  (a > 0 ∧ b > 0 ∧ 14 * (phi a)^2 - phi (a * b) + 22 * (phi b)^2 = a^2 + b^2) ↔
  (∃ x y : ℕ, a = 30 * 2^x * 3^y ∧ b = 6 * 2^x * 3^y) :=
sorry

end euler_totient_equation_solutions_l1605_160594


namespace toms_climbing_time_l1605_160525

/-- Proves that Tom's climbing time is 2 hours given the conditions -/
theorem toms_climbing_time (elizabeth_time : ℕ) (tom_factor : ℕ) :
  elizabeth_time = 30 →
  tom_factor = 4 →
  (elizabeth_time * tom_factor : ℚ) / 60 = 2 :=
by
  sorry

end toms_climbing_time_l1605_160525


namespace distilled_water_remaining_l1605_160556

/-- Represents a mixed number as a pair of integers (whole, fraction) -/
structure MixedNumber where
  whole : Int
  numerator : Int
  denominator : Int
  denom_pos : denominator > 0

/-- Converts a MixedNumber to a rational number -/
def mixedToRational (m : MixedNumber) : Rat :=
  m.whole + (m.numerator : Rat) / (m.denominator : Rat)

theorem distilled_water_remaining
  (initial : MixedNumber)
  (used : MixedNumber)
  (h_initial : initial = ⟨3, 1, 2, by norm_num⟩)
  (h_used : used = ⟨1, 3, 4, by norm_num⟩) :
  mixedToRational initial - mixedToRational used = 7/4 := by
  sorry

#check distilled_water_remaining

end distilled_water_remaining_l1605_160556


namespace peter_has_winning_strategy_l1605_160515

/-- Represents the possible moves in the game -/
inductive Move
  | Single : Nat → Nat → Move  -- 1x1
  | HorizontalRect : Nat → Nat → Move  -- 1x2
  | VerticalRect : Nat → Nat → Move  -- 2x1
  | Square : Nat → Nat → Move  -- 2x2

/-- Represents the game state -/
structure GameState where
  board : Matrix (Fin 8) (Fin 8) Bool
  currentPlayer : Bool  -- true for Peter, false for Victor

/-- Checks if a move is valid in the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- The symmetry strategy for Peter -/
def symmetryStrategy : Strategy :=
  sorry

/-- Theorem: Peter has a winning strategy -/
theorem peter_has_winning_strategy :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.currentPlayer = true →  -- Peter's turn
      ¬(isGameOver game) →
      ∃ (move : Move),
        isValidMove game move ∧
        ¬(isGameOver (applyMove game move)) ∧
        ∀ (victor_move : Move),
          isValidMove (applyMove game move) victor_move →
          ¬(isGameOver (applyMove (applyMove game move) victor_move)) →
          ∃ (peter_response : Move),
            isValidMove (applyMove (applyMove game move) victor_move) peter_response ∧
            strategy (applyMove (applyMove game move) victor_move) = peter_response :=
  sorry

end peter_has_winning_strategy_l1605_160515


namespace absolute_value_equation_solution_l1605_160518

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 15 :=
by
  -- The unique solution is y = 4.75
  use 4.75
  sorry

end absolute_value_equation_solution_l1605_160518


namespace fraction_simplification_l1605_160588

theorem fraction_simplification :
  5 / (Real.sqrt 75 + 3 * Real.sqrt 5 + 2 * Real.sqrt 45) = (25 * Real.sqrt 3 - 45 * Real.sqrt 5) / 330 := by
  sorry

end fraction_simplification_l1605_160588


namespace lottery_winnings_l1605_160577

/-- Calculates the total winnings for lottery tickets -/
theorem lottery_winnings
  (num_tickets : ℕ)
  (winning_numbers_per_ticket : ℕ)
  (value_per_winning_number : ℕ)
  (h1 : num_tickets = 3)
  (h2 : winning_numbers_per_ticket = 5)
  (h3 : value_per_winning_number = 20) :
  num_tickets * winning_numbers_per_ticket * value_per_winning_number = 300 :=
by sorry

end lottery_winnings_l1605_160577


namespace joel_peppers_l1605_160559

/-- Represents the number of peppers picked each day of the week -/
structure WeeklyPeppers where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the number of non-hot peppers given the weekly pepper count -/
def nonHotPeppers (w : WeeklyPeppers) : ℕ :=
  let total := w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday
  (total * 4) / 5

theorem joel_peppers :
  let w : WeeklyPeppers := {
    sunday := 7,
    monday := 12,
    tuesday := 14,
    wednesday := 12,
    thursday := 5,
    friday := 18,
    saturday := 12
  }
  nonHotPeppers w = 64 := by
  sorry

end joel_peppers_l1605_160559


namespace dividend_calculation_l1605_160526

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 13)
  (h_quotient : quotient = 17)
  (h_remainder : remainder = 1) :
  divisor * quotient + remainder = 222 := by
  sorry

end dividend_calculation_l1605_160526


namespace coefficient_of_3x2y_l1605_160597

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℕ → ℕ → ℚ) : ℚ := m 0 0

/-- A monomial is represented as a function from ℕ × ℕ to ℚ, where m i j represents the coefficient of x^i * y^j. -/
def monomial_3x2y : ℕ → ℕ → ℚ := fun i j => if i = 2 ∧ j = 1 then 3 else 0

theorem coefficient_of_3x2y :
  coefficient monomial_3x2y = 3 := by sorry

end coefficient_of_3x2y_l1605_160597


namespace total_games_played_l1605_160535

theorem total_games_played (total_teams : Nat) (rivalry_groups : Nat) (teams_per_group : Nat) (additional_games_per_team : Nat) : 
  total_teams = 50 → 
  rivalry_groups = 10 → 
  teams_per_group = 5 → 
  additional_games_per_team = 2 → 
  (total_teams * (total_teams - 1) / 2) + (rivalry_groups * teams_per_group * additional_games_per_team / 2) = 1325 := by
  sorry

end total_games_played_l1605_160535


namespace divisors_121_divisors_1000_divisors_1000000000_l1605_160510

-- Define a function to calculate the number of divisors given prime factorization
def num_divisors (factorization : List (Nat × Nat)) : Nat :=
  factorization.foldl (fun acc (_, exp) => acc * (exp + 1)) 1

-- Theorem for 121
theorem divisors_121 :
  num_divisors [(11, 2)] = 3 := by sorry

-- Theorem for 1000
theorem divisors_1000 :
  num_divisors [(2, 3), (5, 3)] = 16 := by sorry

-- Theorem for 1000000000
theorem divisors_1000000000 :
  num_divisors [(2, 9), (5, 9)] = 100 := by sorry

end divisors_121_divisors_1000_divisors_1000000000_l1605_160510


namespace f_properties_l1605_160508

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem f_properties :
  (∀ x > 1, f x > 0) ∧
  (∀ x, 0 < x → x < 1 → f x < 0) ∧
  (∀ x > 0, f x ≥ -1 / (2 * Real.exp 1)) ∧
  (∀ x > 0, f x ≥ x - 1) :=
sorry

end f_properties_l1605_160508


namespace mickeys_jaydens_difference_l1605_160521

theorem mickeys_jaydens_difference (mickey jayden coraline : ℕ) : 
  (∃ d : ℕ, mickey = jayden + d) →
  jayden = coraline - 40 →
  coraline = 80 →
  mickey + jayden + coraline = 180 →
  ∃ d : ℕ, mickey = jayden + d ∧ d = 20 := by
  sorry

end mickeys_jaydens_difference_l1605_160521


namespace set_intersection_difference_l1605_160551

theorem set_intersection_difference (S T : Set ℕ) (a b : ℕ) :
  S = {1, 2, a} →
  T = {2, 3, 4, b} →
  S ∩ T = {1, 2, 3} →
  a - b = 2 := by
sorry

end set_intersection_difference_l1605_160551


namespace maria_cookies_l1605_160589

theorem maria_cookies (cookies_per_bag : ℕ) (chocolate_chip : ℕ) (baggies : ℕ) 
  (h1 : cookies_per_bag = 8)
  (h2 : chocolate_chip = 5)
  (h3 : baggies = 3) :
  cookies_per_bag * baggies - chocolate_chip = 19 := by
  sorry

end maria_cookies_l1605_160589


namespace bob_age_proof_l1605_160505

theorem bob_age_proof (alice_age bob_age charlie_age : ℕ) : 
  (alice_age + 10 = 2 * (bob_age - 10)) →
  (alice_age = bob_age + 7) →
  (charlie_age = (alice_age + bob_age) / 2) →
  bob_age = 37 := by
  sorry

end bob_age_proof_l1605_160505


namespace circular_arrangement_problem_l1605_160539

/-- Represents a circular arrangement of 6 numbers -/
structure CircularArrangement where
  numbers : Fin 6 → ℕ
  sum_rule : ∀ i : Fin 6, numbers i + numbers (i + 1) = 2 * numbers ((i + 2) % 6)

theorem circular_arrangement_problem 
  (arr : CircularArrangement)
  (h1 : ∃ i : Fin 6, arr.numbers i = 15 ∧ arr.numbers ((i + 1) % 6) + arr.numbers ((i + 5) % 6) = 16)
  (h2 : ∃ j : Fin 6, arr.numbers j + arr.numbers ((j + 2) % 6) = 10) :
  ∃ k : Fin 6, arr.numbers k = 7 ∧ arr.numbers ((k + 1) % 6) + arr.numbers ((k + 5) % 6) = 10 :=
sorry

end circular_arrangement_problem_l1605_160539


namespace problem_statement_l1605_160523

theorem problem_statement (a b c : ℝ) (h : a^3 + a*b + a*c < 0) : b^5 - 4*a*c > 0 := by
  sorry

end problem_statement_l1605_160523


namespace diner_menu_problem_l1605_160513

theorem diner_menu_problem (n : ℕ) (h1 : n > 0) : 
  let vegan_dishes : ℕ := 6
  let vegan_fraction : ℚ := 1 / 6
  let nut_containing_vegan : ℕ := 5
  (vegan_dishes : ℚ) / n = vegan_fraction →
  (vegan_dishes - nut_containing_vegan : ℚ) / n = 1 / 36 := by
  sorry

end diner_menu_problem_l1605_160513


namespace cube_of_sqrt_three_l1605_160568

theorem cube_of_sqrt_three (x : ℝ) : 
  Real.sqrt (x + 3) = 3 → (x + 3)^3 = 729 := by
sorry

end cube_of_sqrt_three_l1605_160568


namespace line_in_plane_if_points_in_plane_l1605_160596

-- Define the types for our geometric objects
variable (α : Type) [LinearOrderedField α]
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (on_line : Point → Line → Prop)
variable (on_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_in_plane_if_points_in_plane 
  (a b l : Line) (α : Plane) (M N : Point) :
  line_in_plane a α →
  line_in_plane b α →
  on_line M a →
  on_line N b →
  on_line M l →
  on_line N l →
  line_in_plane l α :=
sorry

end line_in_plane_if_points_in_plane_l1605_160596


namespace frog_arrangement_problem_l1605_160511

theorem frog_arrangement_problem :
  ∃! (N : ℕ), 
    N > 0 ∧
    N % 2 = 1 ∧
    N % 3 = 1 ∧
    N % 4 = 1 ∧
    N % 5 = 0 ∧
    N < 50 ∧
    N = 25 := by sorry

end frog_arrangement_problem_l1605_160511


namespace arithmetic_sequence_general_term_l1605_160560

/-- Given an arithmetic sequence {a_n} where a₁₀ = 30 and a₂₀ = 50,
    the general term is a_n = 2n + 10 -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)  -- The sequence
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_10 : a 10 = 30)  -- Given condition
  (h_20 : a 20 = 50)  -- Given condition
  : ∀ n : ℕ, a n = 2 * n + 10 := by
  sorry

end arithmetic_sequence_general_term_l1605_160560


namespace regular_triangular_pyramid_volume_regular_triangular_pyramid_volume_is_correct_l1605_160575

/-- The volume of a regular triangular pyramid with specific properties -/
theorem regular_triangular_pyramid_volume 
  (r : ℝ) -- Length of the perpendicular from the base of the height to a lateral edge
  (α : ℝ) -- Dihedral angle between the lateral face and the base of the pyramid
  (h1 : 0 < r) -- r is positive
  (h2 : 0 < α) -- α is positive
  (h3 : α < π / 2) -- α is less than 90 degrees
  : ℝ :=
  let volume := (Real.sqrt 3 * r^3 * Real.sqrt ((4 + Real.tan α ^ 2) ^ 3)) / (8 * Real.tan α ^ 2)
  volume

#check regular_triangular_pyramid_volume

theorem regular_triangular_pyramid_volume_is_correct
  (r : ℝ) -- Length of the perpendicular from the base of the height to a lateral edge
  (α : ℝ) -- Dihedral angle between the lateral face and the base of the pyramid
  (h1 : 0 < r) -- r is positive
  (h2 : 0 < α) -- α is positive
  (h3 : α < π / 2) -- α is less than 90 degrees
  : regular_triangular_pyramid_volume r α h1 h2 h3 = 
    (Real.sqrt 3 * r^3 * Real.sqrt ((4 + Real.tan α ^ 2) ^ 3)) / (8 * Real.tan α ^ 2) :=
by sorry

end regular_triangular_pyramid_volume_regular_triangular_pyramid_volume_is_correct_l1605_160575


namespace last_locker_opened_l1605_160564

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Toggles the state of a locker -/
def toggleLocker (state : LockerState) : LockerState :=
  match state with
  | LockerState.Open => LockerState.Closed
  | LockerState.Closed => LockerState.Open

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

/-- The main theorem stating that the last locker opened is 509 -/
theorem last_locker_opened :
  ∀ (n : Nat), n ≤ 512 →
    (isPerfectSquare n ↔ (
      ∀ (k : Nat), k ≤ 512 →
        (n % k = 0 → toggleLocker (
          if k = 1 then LockerState.Closed
          else if k < n then toggleLocker LockerState.Closed
          else LockerState.Closed
        ) = LockerState.Open)
    )) →
  (∀ m : Nat, m > 509 ∧ m ≤ 512 →
    ¬(∀ (k : Nat), k ≤ 512 →
      (m % k = 0 → toggleLocker (
        if k = 1 then LockerState.Closed
        else if k < m then toggleLocker LockerState.Closed
        else LockerState.Closed
      ) = LockerState.Open))) →
  isPerfectSquare 509 :=
by
  sorry


end last_locker_opened_l1605_160564


namespace sasha_guessing_game_l1605_160598

theorem sasha_guessing_game (X : ℕ) (hX : X ≤ 100) :
  ∃ (questions : List (ℕ × ℕ)),
    questions.length ≤ 7 ∧
    (∀ (M N : ℕ), (M, N) ∈ questions → M < 100 ∧ N < 100) ∧
    ∀ (Y : ℕ), Y ≤ 100 →
      (∀ (M N : ℕ), (M, N) ∈ questions →
        Nat.gcd (X + M) N = Nat.gcd (Y + M) N) →
      X = Y :=
by sorry

end sasha_guessing_game_l1605_160598


namespace nell_baseball_cards_l1605_160504

theorem nell_baseball_cards (initial_cards given_to_john given_to_jeff : ℕ) 
  (h1 : initial_cards = 573)
  (h2 : given_to_john = 195)
  (h3 : given_to_jeff = 168) :
  initial_cards - (given_to_john + given_to_jeff) = 210 := by
  sorry

end nell_baseball_cards_l1605_160504


namespace quadratic_roots_nature_l1605_160562

theorem quadratic_roots_nature (x : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -4 * Real.sqrt 5
  let c : ℝ := 20
  let discriminant := b^2 - 4*a*c
  discriminant = 0 ∧ ∃ (root : ℝ), x^2 - 4*x*(Real.sqrt 5) + 20 = 0 → x = root :=
by sorry

end quadratic_roots_nature_l1605_160562


namespace equilateral_triangle_intersections_l1605_160519

/-- Represents a point on a side of the triangle -/
structure DivisionPoint where
  side : Fin 3
  position : Fin 11

/-- Represents a line segment from a vertex to a division point -/
structure Segment where
  vertex : Fin 3
  endpoint : DivisionPoint

/-- The number of intersection points in the described configuration -/
def intersection_points : ℕ := 301

/-- States that the number of intersection points in the described triangle configuration is 301 -/
theorem equilateral_triangle_intersections :
  ∀ (triangle : Type) (is_equilateral : triangle → Prop) 
    (divide_sides : triangle → Fin 3 → Fin 12 → DivisionPoint)
    (connect_vertices : triangle → Segment → Prop),
  (∃ (t : triangle), is_equilateral t ∧ 
    (∀ (s : Fin 3) (p : Fin 12), ∃ (dp : DivisionPoint), divide_sides t s p = dp) ∧
    (∀ (v : Fin 3) (dp : DivisionPoint), v ≠ dp.side → connect_vertices t ⟨v, dp⟩)) →
  (∃ (intersection_count : ℕ), intersection_count = intersection_points) :=
by sorry

end equilateral_triangle_intersections_l1605_160519


namespace trapezoid_mn_length_l1605_160524

/-- Represents a trapezoid ABCD with point M on diagonal AC and point N on diagonal BD -/
structure Trapezoid where
  /-- Length of base AD -/
  ad : ℝ
  /-- Length of base BC -/
  bc : ℝ
  /-- Ratio of AM to MC on diagonal AC -/
  am_mc_ratio : ℝ × ℝ
  /-- Length of segment MN -/
  mn : ℝ

/-- Theorem stating the length of MN in the given trapezoid configuration -/
theorem trapezoid_mn_length (t : Trapezoid) :
  t.ad = 3 ∧ t.bc = 18 ∧ t.am_mc_ratio = (1, 2) → t.mn = 4 := by
  sorry

end trapezoid_mn_length_l1605_160524


namespace min_cost_22_bottles_l1605_160520

/-- Calculates the minimum cost to buy a given number of bottles -/
def min_cost (single_price : ℚ) (box_price : ℚ) (bottles_needed : ℕ) : ℚ :=
  let box_size := 6
  let full_boxes := bottles_needed / box_size
  let remaining_bottles := bottles_needed % box_size
  full_boxes * box_price + remaining_bottles * single_price

/-- The minimum cost to buy 22 bottles is R$ 56.20 -/
theorem min_cost_22_bottles :
  min_cost (280 / 100) (1500 / 100) 22 = 5620 / 100 := by
  sorry

end min_cost_22_bottles_l1605_160520


namespace power_sums_l1605_160522

variable (x y p q : ℝ)

def sum_condition : Prop := x + y = -p
def product_condition : Prop := x * y = q

theorem power_sums (h1 : sum_condition x y p) (h2 : product_condition x y q) :
  (x^2 + y^2 = p^2 - 2*q) ∧
  (x^3 + y^3 = -p^3 + 3*p*q) ∧
  (x^4 + y^4 = p^4 - 4*p^2*q + 2*q^2) := by
  sorry

end power_sums_l1605_160522


namespace largest_M_has_property_l1605_160561

/-- The property that for any 10 distinct real numbers in [1, M], 
    there exist three that form a quadratic with no real roots -/
def has_property (M : ℝ) : Prop :=
  ∀ (a : Fin 10 → ℝ), (∀ i j, i ≠ j → a i ≠ a j) → 
  (∀ i, 1 ≤ a i ∧ a i ≤ M) →
  ∃ i j k, i < j ∧ j < k ∧ a i < a j ∧ a j < a k ∧
  (a j)^2 < 4 * (a i) * (a k)

/-- The largest integer M > 1 with the property -/
def largest_M : ℕ := 4^255

theorem largest_M_has_property :
  (has_property (largest_M : ℝ)) ∧
  ∀ n : ℕ, n > largest_M → ¬(has_property (n : ℝ)) :=
by sorry


end largest_M_has_property_l1605_160561


namespace son_is_eighteen_l1605_160542

theorem son_is_eighteen (father_age son_age : ℕ) : 
  father_age + son_age = 55 →
  ∃ (y : ℕ), father_age + y + (son_age + y) = 93 ∧ son_age + y = father_age →
  (father_age = 18 ∨ son_age = 18) →
  son_age = 18 := by
sorry

end son_is_eighteen_l1605_160542


namespace lyceum_students_count_l1605_160548

theorem lyceum_students_count :
  ∀ n : ℕ,
  (1000 < n ∧ n < 2000) →
  (n * 76 % 100 = 0) →
  (n * 5 % 37 = 0) →
  n = 1850 :=
by
  sorry

end lyceum_students_count_l1605_160548


namespace basketball_win_rate_l1605_160576

theorem basketball_win_rate (games_won : ℕ) (first_games : ℕ) (total_games : ℕ) (remaining_games : ℕ) (win_rate : ℚ) : 
  games_won = 25 ∧ 
  first_games = 35 ∧ 
  total_games = 60 ∧ 
  remaining_games = 25 ∧ 
  win_rate = 4/5 →
  (games_won + remaining_games : ℚ) / total_games = win_rate ↔ 
  remaining_games = 23 := by
sorry

end basketball_win_rate_l1605_160576


namespace two_numbers_difference_l1605_160546

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 20000 →
  b % 5 = 0 →
  b / 10 = a →
  (b % 10 = 0 ∨ b % 10 = 5) →
  b - a = 16358 := by
sorry

end two_numbers_difference_l1605_160546


namespace probability_of_double_l1605_160572

-- Define the range of integers for the mini-domino set
def dominoRange : ℕ := 7

-- Define a function to calculate the total number of pairings
def totalPairings (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the number of doubles in the set
def numDoubles : ℕ := dominoRange

-- Theorem statement
theorem probability_of_double :
  (numDoubles : ℚ) / (totalPairings dominoRange : ℚ) = 1 / 4 := by
  sorry

end probability_of_double_l1605_160572


namespace average_difference_l1605_160512

def average (a b : Int) : ℚ := (a + b) / 2

theorem average_difference : 
  average 500 1000 - average 100 500 = 450 := by sorry

end average_difference_l1605_160512


namespace lanas_final_pages_l1605_160565

def lanas_pages (initial_pages : ℕ) (duanes_pages : ℕ) : ℕ :=
  initial_pages + duanes_pages / 2

theorem lanas_final_pages :
  lanas_pages 8 42 = 29 := by sorry

end lanas_final_pages_l1605_160565


namespace hyperbola_asymptotes_l1605_160502

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun (x y : ℝ) => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  e = 2 → (∀ x y, hyperbola x y ↔ asymptotes x y) :=
by sorry

end hyperbola_asymptotes_l1605_160502


namespace ferry_speed_proof_l1605_160586

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 6

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 3

/-- The time taken by ferry P in hours -/
def time_P : ℝ := 3

/-- The time taken by ferry Q in hours -/
def time_Q : ℝ := time_P + 3

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := 3 * distance_P

theorem ferry_speed_proof :
  speed_P = 6 ∧
  speed_Q = speed_P + 3 ∧
  time_Q = time_P + 3 ∧
  distance_Q = 3 * distance_P ∧
  distance_P = speed_P * time_P ∧
  distance_Q = speed_Q * time_Q :=
by sorry

end ferry_speed_proof_l1605_160586


namespace infinite_sum_solution_l1605_160571

theorem infinite_sum_solution (k : ℝ) (h1 : k > 2) 
  (h2 : (∑' n, (6 * n + 2) / k^n) = 15) : 
  k = (38 + 2 * Real.sqrt 46) / 30 := by
sorry

end infinite_sum_solution_l1605_160571


namespace unique_solution_for_x_l1605_160528

theorem unique_solution_for_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 15)
  (h2 : y + 1 / x = 7 / 20)
  (h3 : x * y = 2) :
  x = 10 := by
  sorry

end unique_solution_for_x_l1605_160528


namespace angle_sixty_degrees_l1605_160509

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem angle_sixty_degrees (t : Triangle) 
  (h : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) : 
  t.A = 60 * π / 180 := by
  sorry


end angle_sixty_degrees_l1605_160509


namespace smallest_non_prime_non_square_no_small_factors_l1605_160540

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_prime_factor_less_than_60 (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p < 60 ∧ p ∣ n

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ n : ℕ, n < 4087 → is_prime n ∨ is_square n ∨ has_prime_factor_less_than_60 n) ∧
  ¬is_prime 4087 ∧
  ¬is_square 4087 ∧
  ¬has_prime_factor_less_than_60 4087 :=
sorry

end smallest_non_prime_non_square_no_small_factors_l1605_160540


namespace family_trip_eggs_l1605_160547

/-- Calculates the total number of boiled eggs prepared for a family trip -/
def total_eggs (num_adults num_girls num_boys : ℕ) (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) : ℕ :=
  num_adults * eggs_per_adult + num_girls * eggs_per_girl + num_boys * (eggs_per_girl + 1)

/-- Theorem stating that the total number of boiled eggs for the given family trip is 36 -/
theorem family_trip_eggs :
  total_eggs 3 7 10 3 1 = 36 := by
  sorry

end family_trip_eggs_l1605_160547


namespace abs_product_of_neg_two_and_four_l1605_160534

theorem abs_product_of_neg_two_and_four :
  ∀ x y : ℤ, x = -2 → y = 4 → |x * y| = 8 := by
  sorry

end abs_product_of_neg_two_and_four_l1605_160534


namespace ellipse_intersection_theorem_l1605_160574

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- Definition of the line that intersects C -/
def L (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Definition of points A and B as intersections of C and L -/
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂

/-- Condition for OA ⊥ OB -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- Theorem stating the conditions for perpendicularity and the length of AB -/
theorem ellipse_intersection_theorem :
  ∀ k : ℝ, intersectionPoints k →
    (k = 1/2 ∨ k = -1/2) ↔
      (∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂ ∧
        perpendicular x₁ y₁ x₂ y₂ ∧
        ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) = 4*(65^(1/2))/17) :=
by sorry


end ellipse_intersection_theorem_l1605_160574


namespace isosceles_triangle_l1605_160553

/-- A triangle with sides a, b, c exists -/
def triangle_exists (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Triangle PQR with sides p, q, r -/
structure Triangle (p q r : ℝ) : Type :=
  (exists_triangle : triangle_exists p q r)

/-- For any positive integer n, a triangle with sides p^n, q^n, r^n exists -/
def power_triangle_exists (p q r : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → triangle_exists (p^n) (q^n) (r^n)

/-- Main theorem: If a triangle PQR with sides p, q, r exists, and for any positive integer n,
    a triangle with sides p^n, q^n, r^n also exists, then at least two sides of triangle PQR are equal -/
theorem isosceles_triangle (p q r : ℝ) (tr : Triangle p q r) 
    (h : power_triangle_exists p q r) : 
    p = q ∨ q = r ∨ r = p :=
sorry

end isosceles_triangle_l1605_160553


namespace gift_arrangement_count_gift_arrangement_proof_l1605_160538

theorem gift_arrangement_count : ℕ → ℕ → ℕ
  | 5, 4 => 120
  | _, _ => 0

theorem gift_arrangement_proof (n m : ℕ) (hn : n = 5) (hm : m = 4) :
  gift_arrangement_count n m = (n.choose 1) * m.factorial :=
by sorry

end gift_arrangement_count_gift_arrangement_proof_l1605_160538


namespace infinite_primes_with_solutions_l1605_160503

theorem infinite_primes_with_solutions : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ x y : ℤ, x^2 + x + 1 = p * y} := by
  sorry

end infinite_primes_with_solutions_l1605_160503


namespace cake_recipe_flour_l1605_160554

/-- The number of cups of flour in a cake recipe -/
def recipe_flour (sugar_cups : ℕ) (flour_added : ℕ) (flour_remaining : ℕ) : ℕ :=
  flour_added + flour_remaining

theorem cake_recipe_flour :
  ∀ (sugar_cups : ℕ) (flour_added : ℕ) (flour_remaining : ℕ),
    sugar_cups = 2 →
    flour_added = 7 →
    flour_remaining = sugar_cups + 1 →
    recipe_flour sugar_cups flour_added flour_remaining = 10 :=
by
  sorry

end cake_recipe_flour_l1605_160554


namespace cyclic_sum_inequality_l1605_160593

theorem cyclic_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + 
   c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end cyclic_sum_inequality_l1605_160593


namespace coffee_shop_weekly_total_l1605_160527

/-- Represents a coffee shop with its brewing characteristics -/
structure CoffeeShop where
  weekday_rate : ℕ  -- Cups brewed per hour on weekdays
  weekend_total : ℕ  -- Total cups brewed over the weekend
  daily_hours : ℕ  -- Hours open per day

/-- Calculates the total number of coffee cups brewed in one week -/
def weekly_total (shop : CoffeeShop) : ℕ :=
  (shop.weekday_rate * shop.daily_hours * 5) + shop.weekend_total

/-- Theorem stating that a coffee shop with given characteristics brews 370 cups per week -/
theorem coffee_shop_weekly_total :
  ∀ (shop : CoffeeShop),
    shop.weekday_rate = 10 ∧
    shop.weekend_total = 120 ∧
    shop.daily_hours = 5 →
    weekly_total shop = 370 := by
  sorry


end coffee_shop_weekly_total_l1605_160527


namespace inequality_proof_l1605_160567

theorem inequality_proof (a b c d : ℝ) :
  (a^8 + b^3 + c^8 + d^3)^2 ≤ 4 * (a^4 + b^8 + c^8 + d^4) := by
  sorry

end inequality_proof_l1605_160567


namespace overlapping_squares_area_l1605_160514

/-- Represents a square sheet of paper -/
structure Square :=
  (side : ℝ)

/-- Represents the rotation of a square -/
inductive Rotation
  | NoRotation
  | Rotation45
  | Rotation90

/-- Represents a stack of rotated squares -/
structure RotatedSquares :=
  (bottom : Square)
  (middle : Square)
  (top : Square)
  (middleRotation : Rotation)
  (topRotation : Rotation)

/-- Calculates the area of the resulting shape formed by overlapping rotated squares -/
def resultingArea (rs : RotatedSquares) : ℝ :=
  sorry

theorem overlapping_squares_area :
  ∀ (rs : RotatedSquares),
    rs.bottom.side = 8 ∧
    rs.middle.side = 8 ∧
    rs.top.side = 8 ∧
    rs.middleRotation = Rotation.Rotation45 ∧
    rs.topRotation = Rotation.Rotation90 →
    resultingArea rs = 192 :=
  sorry

end overlapping_squares_area_l1605_160514


namespace cos_angle_POQ_l1605_160595

/-- Given two points P and Q on the unit circle centered at the origin O,
    where P is in the first quadrant with x-coordinate 4/5,
    and Q is in the fourth quadrant with x-coordinate 5/13,
    prove that the cosine of angle POQ is 56/65. -/
theorem cos_angle_POQ (P Q : ℝ × ℝ) : 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (P.1 = 4/5) →          -- x-coordinate of P is 4/5
  (P.2 ≥ 0) →            -- P is in the first quadrant
  (Q.1 = 5/13) →         -- x-coordinate of Q is 5/13
  (Q.2 ≤ 0) →            -- Q is in the fourth quadrant
  Real.cos (Real.arccos P.1 + Real.arccos Q.1) = 56/65 := by
  sorry


end cos_angle_POQ_l1605_160595


namespace weighted_average_price_approximation_l1605_160584

def large_bottles : ℕ := 1365
def small_bottles : ℕ := 720
def medium_bottles : ℕ := 450
def extra_large_bottles : ℕ := 275

def large_price : ℚ := 189 / 100
def small_price : ℚ := 142 / 100
def medium_price : ℚ := 162 / 100
def extra_large_price : ℚ := 209 / 100

def total_bottles : ℕ := large_bottles + small_bottles + medium_bottles + extra_large_bottles

def total_cost : ℚ := 
  large_bottles * large_price + 
  small_bottles * small_price + 
  medium_bottles * medium_price + 
  extra_large_bottles * extra_large_price

def weighted_average_price : ℚ := total_cost / total_bottles

theorem weighted_average_price_approximation : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |weighted_average_price - 175/100| < ε :=
sorry

end weighted_average_price_approximation_l1605_160584


namespace find_divisor_l1605_160581

theorem find_divisor : ∃ (d : ℕ), d > 0 ∧ 
  (13603 - 31) % d = 0 ∧
  (∀ (n : ℕ), n < 31 → (13603 - n) % d ≠ 0) ∧
  d = 13572 := by
  sorry

end find_divisor_l1605_160581


namespace even_decreasing_inequality_l1605_160537

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define a decreasing function on [0, +∞)
def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f y < f x

-- Theorem statement
theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_decreasing : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 :=
sorry

end even_decreasing_inequality_l1605_160537


namespace total_wings_is_14_l1605_160573

/-- Represents the types of birds available for purchase. -/
inductive BirdType
| Parrot
| Pigeon
| Canary

/-- Represents the money received from each grandparent. -/
def grandparentMoney : List ℕ := [45, 60, 55, 50]

/-- Represents the cost of each bird type. -/
def birdCost : BirdType → ℕ
| BirdType.Parrot => 35
| BirdType.Pigeon => 25
| BirdType.Canary => 20

/-- Represents the number of birds in a discounted set for each bird type. -/
def discountSet : BirdType → ℕ
| BirdType.Parrot => 3
| BirdType.Pigeon => 4
| BirdType.Canary => 5

/-- Represents the cost of a discounted set for each bird type. -/
def discountSetCost : BirdType → ℕ
| BirdType.Parrot => 35 * 2 + 35 / 2
| BirdType.Pigeon => 25 * 3
| BirdType.Canary => 20 * 4

/-- Represents the number of wings each bird has. -/
def wingsPerBird : ℕ := 2

/-- Represents the total money John has to spend. -/
def totalMoney : ℕ := grandparentMoney.sum

/-- Theorem stating that the total number of wings of all birds John bought is 14. -/
theorem total_wings_is_14 :
  ∃ (parrot pigeon canary : ℕ),
    parrot > 0 ∧ pigeon > 0 ∧ canary > 0 ∧
    parrot * birdCost BirdType.Parrot +
    pigeon * birdCost BirdType.Pigeon +
    canary * birdCost BirdType.Canary = totalMoney ∧
    (parrot + pigeon + canary) * wingsPerBird = 14 :=
  sorry

end total_wings_is_14_l1605_160573


namespace linear_functions_intersection_l1605_160579

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Evaluate a linear function at a given x -/
def LinearFunction.eval (f : LinearFunction) (x : ℝ) : ℝ :=
  f.slope * x + f.intercept

theorem linear_functions_intersection (f₁ f₂ : LinearFunction) :
  (f₁.eval 2 = f₂.eval 2) →
  (|f₁.eval 8 - f₂.eval 8| = 8) →
  ((f₁.eval 20 = 100) ∨ (f₂.eval 20 = 100)) →
  ((f₁.eval 20 = 76 ∧ f₂.eval 20 = 100) ∨ (f₁.eval 20 = 100 ∧ f₂.eval 20 = 124) ∨
   (f₁.eval 20 = 100 ∧ f₂.eval 20 = 76) ∨ (f₁.eval 20 = 124 ∧ f₂.eval 20 = 100)) := by
  sorry

end linear_functions_intersection_l1605_160579


namespace carrie_leftover_money_l1605_160558

/-- Calculates the amount of money Carrie has left after purchasing a bike, helmet, and accessories --/
theorem carrie_leftover_money 
  (hourly_rate : ℝ)
  (hours_per_week : ℝ)
  (weeks_worked : ℝ)
  (bike_cost : ℝ)
  (sales_tax_rate : ℝ)
  (helmet_cost : ℝ)
  (accessories_cost : ℝ)
  (h1 : hourly_rate = 8)
  (h2 : hours_per_week = 35)
  (h3 : weeks_worked = 4)
  (h4 : bike_cost = 400)
  (h5 : sales_tax_rate = 0.06)
  (h6 : helmet_cost = 50)
  (h7 : accessories_cost = 30) :
  hourly_rate * hours_per_week * weeks_worked - 
  (bike_cost * (1 + sales_tax_rate) + helmet_cost + accessories_cost) = 616 :=
by sorry

end carrie_leftover_money_l1605_160558


namespace wade_sandwich_cost_l1605_160582

def sandwich_cost (total_spent : ℚ) (num_sandwiches : ℕ) (num_drinks : ℕ) (drink_cost : ℚ) : ℚ :=
  (total_spent - (num_drinks : ℚ) * drink_cost) / (num_sandwiches : ℚ)

theorem wade_sandwich_cost :
  sandwich_cost 26 3 2 4 = 6 :=
by
  sorry

end wade_sandwich_cost_l1605_160582


namespace additional_gas_needed_l1605_160549

/-- Calculates the additional gallons of gas needed for a truck to reach its destination. -/
theorem additional_gas_needed
  (miles_per_gallon : ℝ)
  (total_distance : ℝ)
  (current_gas : ℝ)
  (h1 : miles_per_gallon = 3)
  (h2 : total_distance = 90)
  (h3 : current_gas = 12) :
  (total_distance - current_gas * miles_per_gallon) / miles_per_gallon = 18 := by
  sorry

end additional_gas_needed_l1605_160549


namespace slope_angle_range_l1605_160543

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + (3 - Real.sqrt 3)*x + 3/4

def is_on_curve (p : ℝ × ℝ) : Prop := p.2 = f p.1

theorem slope_angle_range (p q : ℝ × ℝ) (hp : is_on_curve p) (hq : is_on_curve q) :
  let α := Real.arctan ((q.2 - p.2) / (q.1 - p.1))
  α ∈ Set.union (Set.Ico 0 (Real.pi / 2)) (Set.Icc (2 * Real.pi / 3) Real.pi) :=
sorry

end slope_angle_range_l1605_160543


namespace max_braking_distance_l1605_160517

/-- The braking distance function for a car -/
def s (t : ℝ) : ℝ := 15 * t - 6 * t^2

/-- The maximum distance traveled by the car before stopping -/
theorem max_braking_distance :
  (∃ t : ℝ, ∀ u : ℝ, s u ≤ s t) ∧ (∃ t : ℝ, s t = 75/8) :=
sorry

end max_braking_distance_l1605_160517


namespace lindsay_doll_difference_l1605_160585

/-- The number of dolls Lindsay has with different hair colors -/
structure DollCounts where
  blonde : ℕ
  brown : ℕ
  black : ℕ

/-- Lindsay's doll collection satisfying the given conditions -/
def lindsay_dolls : DollCounts where
  blonde := 4
  brown := 4 * 4
  black := 4 * 4 - 2

/-- The difference between the number of dolls with black and brown hair combined
    and the number of dolls with blonde hair -/
def hair_color_difference (d : DollCounts) : ℕ :=
  d.brown + d.black - d.blonde

theorem lindsay_doll_difference :
  hair_color_difference lindsay_dolls = 26 := by
  sorry

end lindsay_doll_difference_l1605_160585


namespace A_intersect_B_range_of_a_l1605_160555

-- Define the sets A, B, and C
def A : Set ℝ := {x | x < -2 ∨ (3 < x ∧ x < 4)}
def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem 1: Intersection of A and B
theorem A_intersect_B : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 5} := by sorry

-- Theorem 2: Range of a given B ∩ C = B
theorem range_of_a (h : B ∩ C a = B) : a ≤ -3 := by sorry

end A_intersect_B_range_of_a_l1605_160555
