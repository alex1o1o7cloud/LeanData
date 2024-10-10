import Mathlib

namespace scientific_notation_conversion_l397_39768

theorem scientific_notation_conversion :
  (2.61 * 10^(-5) = 0.0000261) ∧ (0.00068 = 6.8 * 10^(-4)) := by
  sorry

end scientific_notation_conversion_l397_39768


namespace star_equation_solutions_l397_39754

-- Define the * operation
def star (a b : ℝ) : ℝ := a * (a + b) + b

-- Theorem statement
theorem star_equation_solutions :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  star a₁ 2.5 = 28.5 ∧ 
  star a₂ 2.5 = 28.5 ∧
  (a₁ = 4 ∨ a₁ = -13/2) ∧
  (a₂ = 4 ∨ a₂ = -13/2) :=
sorry

end star_equation_solutions_l397_39754


namespace value_of_fraction_l397_39782

-- Define the real numbers
variable (a₁ a₂ b₁ b₂ : ℝ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence : Prop :=
  ∃ d : ℝ, a₁ - (-1) = d ∧ a₂ - a₁ = d ∧ (-4) - a₂ = d

-- Define the geometric sequence condition
def is_geometric_sequence : Prop :=
  ∃ r : ℝ, b₁ / (-1) = r ∧ b₂ / b₁ = r ∧ (-8) / b₂ = r

-- Theorem statement
theorem value_of_fraction (h1 : is_arithmetic_sequence a₁ a₂)
                          (h2 : is_geometric_sequence b₁ b₂) :
  (a₂ - a₁) / b₂ = 1 / 4 := by
  sorry

end value_of_fraction_l397_39782


namespace distribution_count_l397_39713

/-- Represents a distribution of tickets to people -/
structure TicketDistribution where
  /-- The number of tickets -/
  num_tickets : Nat
  /-- The number of people -/
  num_people : Nat
  /-- Condition that each person receives at least one ticket -/
  at_least_one_ticket : num_tickets ≥ num_people
  /-- Condition that the number of tickets is 5 -/
  five_tickets : num_tickets = 5
  /-- Condition that the number of people is 4 -/
  four_people : num_people = 4

/-- Counts the number of valid distributions -/
def count_distributions (d : TicketDistribution) : Nat :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating that the number of valid distributions is 96 -/
theorem distribution_count (d : TicketDistribution) : count_distributions d = 96 := by
  sorry

end distribution_count_l397_39713


namespace man_speed_man_speed_result_l397_39785

/-- Calculates the speed of a man given a train passing him in the opposite direction -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * (3600 / 1000)
  man_speed_kmh

/-- The speed of the man is approximately 6 km/h -/
theorem man_speed_result :
  ∃ ε > 0, |man_speed 200 60 10.909090909090908 - 6| < ε :=
by
  sorry

end man_speed_man_speed_result_l397_39785


namespace petya_wins_l397_39729

/-- Represents the game board -/
def Board (n : ℕ) := Fin n → Fin n → Bool

/-- Represents a position on the board -/
structure Position (n : ℕ) where
  row : Fin n
  col : Fin n

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- Represents the game state -/
structure GameState (n : ℕ) where
  board : Board n
  rook_position : Position n
  current_player : Player

/-- Checks if a move is valid -/
def is_valid_move (n : ℕ) (state : GameState n) (new_pos : Position n) : Bool :=
  sorry

/-- Applies a move to the game state -/
def apply_move (n : ℕ) (state : GameState n) (new_pos : Position n) : GameState n :=
  sorry

/-- Checks if the game is over -/
def is_game_over (n : ℕ) (state : GameState n) : Bool :=
  sorry

/-- The main theorem stating Petya has a winning strategy -/
theorem petya_wins (n : ℕ) (h : n ≥ 2) :
  ∃ (strategy : GameState n → Position n),
    ∀ (game : GameState n),
      game.current_player = Player.Petya →
      ¬(is_game_over n game) →
      is_valid_move n game (strategy game) ∧
      (∀ (vasya_move : Position n),
        is_valid_move n (apply_move n game (strategy game)) vasya_move →
        ∃ (petya_next_move : Position n),
          is_valid_move n (apply_move n (apply_move n game (strategy game)) vasya_move) petya_next_move) :=
sorry

end petya_wins_l397_39729


namespace complex_fraction_equals_i_l397_39774

theorem complex_fraction_equals_i : 
  let i : ℂ := Complex.I
  (1 + i^2017) / (1 - i) = i := by sorry

end complex_fraction_equals_i_l397_39774


namespace divisibility_condition_l397_39772

theorem divisibility_condition (m n : ℕ) : 
  m ≥ 1 → n ≥ 1 → 
  (m * n) ∣ (3^m + 1) → 
  (m * n) ∣ (3^n + 1) → 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) := by
  sorry

end divisibility_condition_l397_39772


namespace heathers_remaining_blocks_l397_39742

theorem heathers_remaining_blocks
  (initial_blocks : ℕ)
  (shared_with_jose : ℕ)
  (shared_with_emily : ℕ)
  (h1 : initial_blocks = 86)
  (h2 : shared_with_jose = 41)
  (h3 : shared_with_emily = 15) :
  initial_blocks - (shared_with_jose + shared_with_emily) = 30 :=
by sorry

end heathers_remaining_blocks_l397_39742


namespace candy_bar_ratio_l397_39744

/-- Proves the ratio of candy bars given the second time to the first time -/
theorem candy_bar_ratio (initial_bars : ℕ) (initial_given : ℕ) (bought_bars : ℕ) (kept_bars : ℕ) :
  initial_bars = 7 →
  initial_given = 3 →
  bought_bars = 30 →
  kept_bars = 22 →
  ∃ (second_given : ℕ), 
    second_given = initial_bars + bought_bars - kept_bars - initial_given ∧
    second_given = 4 * initial_given :=
by sorry

end candy_bar_ratio_l397_39744


namespace max_ratio_squared_l397_39705

theorem max_ratio_squared (c d x y : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c ≥ d)
  (heq : c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x)^2 + (d - y)^2)
  (hx : 0 ≤ x ∧ x < c) (hy : 0 ≤ y ∧ y < d) :
  (c / d)^2 ≤ 4/3 :=
sorry

end max_ratio_squared_l397_39705


namespace welders_proof_l397_39707

/-- Represents the initial number of welders -/
def initial_welders : ℕ := 12

/-- Represents the number of days initially needed to complete the order -/
def initial_days : ℕ := 3

/-- Represents the number of welders that leave after the first day -/
def welders_left : ℕ := 9

/-- Represents the additional days needed by remaining welders to complete the order -/
def additional_days : ℕ := 8

/-- Proves that the initial number of welders is correct given the conditions -/
theorem welders_proof :
  (initial_welders - welders_left) * additional_days = initial_welders * (initial_days - 1) :=
by sorry

end welders_proof_l397_39707


namespace largest_circle_area_l397_39787

theorem largest_circle_area (playground_area : Real) (π : Real) : 
  playground_area = 400 → π = 3.1 → 
  (π * (Real.sqrt playground_area / 2)^2 : Real) = 310 := by
  sorry

end largest_circle_area_l397_39787


namespace income_calculation_l397_39715

theorem income_calculation (income expenditure savings : ℕ) : 
  income = 7 * expenditure / 6 →
  savings = income - expenditure →
  savings = 2000 →
  income = 14000 := by
sorry

end income_calculation_l397_39715


namespace first_term_of_arithmetic_sequence_l397_39749

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 3)
  (h_d : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = -1 := by
sorry

end first_term_of_arithmetic_sequence_l397_39749


namespace dogs_with_spots_l397_39781

theorem dogs_with_spots (total_dogs : ℚ) (pointy_ears : ℚ) : ℚ :=
  by
  have h1 : pointy_ears = total_dogs / 5 := by sorry
  have h2 : total_dogs = pointy_ears * 5 := by sorry
  have h3 : total_dogs / 2 = (pointy_ears * 5) / 2 := by sorry
  exact (pointy_ears * 5) / 2

#check dogs_with_spots

end dogs_with_spots_l397_39781


namespace largest_consecutive_sum_l397_39776

theorem largest_consecutive_sum (n : ℕ) : n = 14141 ↔ 
  (∀ k : ℕ, k ≤ n → (k * (k + 1)) / 2 ≤ 100000000) ∧
  (∀ m : ℕ, m > n → (m * (m + 1)) / 2 > 100000000) := by
sorry

end largest_consecutive_sum_l397_39776


namespace power_inequality_l397_39740

theorem power_inequality (n : ℕ) (h : n > 2) : n^(n+1) > (n+1)^n := by
  sorry

end power_inequality_l397_39740


namespace points_five_units_from_negative_three_l397_39700

theorem points_five_units_from_negative_three (x : ℝ) : 
  (|x - (-3)| = 5) ↔ (x = -8 ∨ x = 2) := by
  sorry

end points_five_units_from_negative_three_l397_39700


namespace inequality_proof_l397_39733

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 1) : 
  (x * Real.sqrt x) / (y + z) + (y * Real.sqrt y) / (z + x) + (z * Real.sqrt z) / (x + y) ≥ Real.sqrt 3 / 2 := by
sorry

end inequality_proof_l397_39733


namespace solution_set_quadratic_inequality_l397_39778

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end solution_set_quadratic_inequality_l397_39778


namespace lunks_for_apples_l397_39712

/-- The number of lunks that can be traded for a given number of kunks -/
def lunks_per_kunks : ℚ := 4 / 2

/-- The number of kunks that can be traded for a given number of apples -/
def kunks_per_apples : ℚ := 3 / 5

/-- The number of apples we want to purchase -/
def target_apples : ℕ := 20

/-- Theorem: The number of lunks needed to purchase 20 apples is 24 -/
theorem lunks_for_apples : 
  (target_apples : ℚ) * kunks_per_apples * lunks_per_kunks = 24 := by sorry

end lunks_for_apples_l397_39712


namespace unique_conjugate_pair_l397_39755

/-- A quadratic trinomial function -/
def QuadraticTrinomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Conjugate numbers for a function -/
def Conjugate (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y ∧ f y = x

theorem unique_conjugate_pair (a b c : ℝ) (x y : ℝ) :
  x ≠ y →
  let f := QuadraticTrinomial a b c
  Conjugate f x y →
  ∀ u v : ℝ, Conjugate f u v → (u = x ∧ v = y) ∨ (u = y ∧ v = x) := by
  sorry

end unique_conjugate_pair_l397_39755


namespace parabola_properties_l397_39739

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 22

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (3, 5)

-- Define a point that the parabola passes through
def point : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through the given point
  parabola_equation point.1 = point.2 ∧
  -- The vertex of the parabola is at (3, 5)
  (∀ x, parabola_equation x ≤ parabola_equation vertex.1) ∧
  -- The axis of symmetry is vertical (x = 3)
  (∀ x, parabola_equation (2 * vertex.1 - x) = parabola_equation x) :=
by sorry

end parabola_properties_l397_39739


namespace kody_age_proof_l397_39799

/-- Kody's current age -/
def kody_age : ℕ := 32

/-- Mohamed's current age -/
def mohamed_age : ℕ := 60

/-- The time difference between now and the past reference point -/
def years_passed : ℕ := 4

theorem kody_age_proof :
  (∃ (kody_past mohamed_past : ℕ),
    kody_past = mohamed_past / 2 ∧
    kody_past + years_passed = kody_age ∧
    mohamed_past + years_passed = mohamed_age) ∧
  mohamed_age = 2 * 30 →
  kody_age = 32 := by sorry

end kody_age_proof_l397_39799


namespace line_tangent_to_parabola_l397_39741

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ x, a*x^2 + b*x + c = 0 ∧ ∀ y, a*y^2 + b*y + c = 0 → y = x

/-- The problem statement -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, y^2 = 32*x → (4*x + 6*y + k = 0 ↔ 
    ∃! t, 4*t + 6*(32*t)^(1/2) + k = 0 ∨ 4*t + 6*(-32*t)^(1/2) + k = 0)) →
  k = 72 := by
  sorry

end line_tangent_to_parabola_l397_39741


namespace complex_on_imaginary_axis_l397_39791

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (a^2 - 2*a) + (a^2 - a - 2)*I
  (z.re = 0) → (a = 0 ∨ a = 2) := by
  sorry

end complex_on_imaginary_axis_l397_39791


namespace box_height_minimum_l397_39714

theorem box_height_minimum (x : ℝ) : 
  x > 0 →                           -- side length is positive
  2 * x^2 + 4 * x * (2 * x) ≥ 120 → -- surface area is at least 120
  2 * x ≥ 4 * Real.sqrt 3 :=        -- height (2x) is at least 4√3
by
  sorry

end box_height_minimum_l397_39714


namespace complement_of_A_l397_39726

def A : Set ℝ := {x | |x - 1| ≤ 2}

theorem complement_of_A :
  Aᶜ = {x : ℝ | x < -1 ∨ x > 3} :=
by sorry

end complement_of_A_l397_39726


namespace extremum_implies_a_equals_one_l397_39745

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x*a

-- State the theorem
theorem extremum_implies_a_equals_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 1 := by
  sorry

end extremum_implies_a_equals_one_l397_39745


namespace frog_jump_parity_l397_39701

def frog_jump (n : ℕ) (t : ℕ) : ℕ :=
  (t * (t + 1) / 2 - 1) % n

theorem frog_jump_parity (n : ℕ) (h1 : n > 1) :
  (∀ r : ℕ, r < n → ∃ t : ℕ, frog_jump n t = r) →
  Even n :=
sorry

end frog_jump_parity_l397_39701


namespace sufficient_condition_for_collinearity_l397_39704

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem sufficient_condition_for_collinearity (x : ℝ) :
  let a : ℝ × ℝ := (1, 2 - x)
  let b : ℝ × ℝ := (2 + x, 3)
  b = (1, 3) → collinear a b :=
by
  sorry

end sufficient_condition_for_collinearity_l397_39704


namespace no_real_a_with_unique_solution_l397_39735

-- Define the function f(x) = x^2 + ax + 2a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2*a

-- Define the property that |f(x)| ≤ 5 has exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, |f a x| ≤ 5

-- Theorem statement
theorem no_real_a_with_unique_solution :
  ¬∃ a : ℝ, has_unique_solution a :=
sorry

end no_real_a_with_unique_solution_l397_39735


namespace imaginary_part_of_z_l397_39743

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z : ℂ := (x + Complex.I) / (y - Complex.I)
  Complex.im z = 1 := by sorry

end imaginary_part_of_z_l397_39743


namespace expression_equals_two_l397_39795

theorem expression_equals_two :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end expression_equals_two_l397_39795


namespace class_size_calculation_l397_39709

theorem class_size_calculation (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 13 → average_increase = 1/2 → 
  (mark_increase : ℚ) / average_increase = 26 := by
  sorry

end class_size_calculation_l397_39709


namespace circle_equation_l397_39711

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
theorem circle_equation (x y : ℝ) : 
  let h : ℝ := 1
  let k : ℝ := 2
  let r : ℝ := 5
  (x - h)^2 + (y - k)^2 = r^2 := by sorry

end circle_equation_l397_39711


namespace wheat_mixture_problem_arun_wheat_problem_l397_39747

/-- Calculates the rate of the second wheat purchase given the conditions of Arun's wheat mixture problem -/
theorem wheat_mixture_problem (first_quantity : ℝ) (first_rate : ℝ) (second_quantity : ℝ) (selling_rate : ℝ) (profit_percentage : ℝ) : ℝ :=
  let total_quantity := first_quantity + second_quantity
  let first_cost := first_quantity * first_rate
  let total_selling_price := total_quantity * selling_rate
  let total_cost := total_selling_price / (1 + profit_percentage / 100)
  (total_cost - first_cost) / second_quantity

/-- The rate of the second wheat purchase in Arun's problem is 14.25 -/
theorem arun_wheat_problem : 
  wheat_mixture_problem 30 11.50 20 15.75 25 = 14.25 := by
  sorry

end wheat_mixture_problem_arun_wheat_problem_l397_39747


namespace prime_square_in_A_implies_prime_in_A_l397_39773

def A : Set ℕ := {n : ℕ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a^2 + 2*b^2}

theorem prime_square_in_A_implies_prime_in_A (p : ℕ) (hp : Nat.Prime p) (hp2 : p^2 ∈ A) : p ∈ A := by
  sorry

end prime_square_in_A_implies_prime_in_A_l397_39773


namespace max_value_of_g_l397_39752

/-- The function g(x) = 4x - x^4 --/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The theorem stating that the maximum value of g(x) on [0, √4] is 3 --/
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  g x = 3 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ g x :=
sorry

end max_value_of_g_l397_39752


namespace flower_position_l397_39756

/-- Represents the number of students in the circle -/
def n : ℕ := 7

/-- Represents the number of times the drum is beaten -/
def k : ℕ := 50

/-- Function to calculate the final position after k rotations in a circle of n elements -/
def finalPosition (n k : ℕ) : ℕ := 
  (k % n) + 1

theorem flower_position : 
  finalPosition n k = 2 := by sorry

end flower_position_l397_39756


namespace photo_arrangements_l397_39797

/-- The number of ways 7 students can stand in a line for a photo, 
    given specific constraints on their positions. -/
theorem photo_arrangements (n : Nat) (h1 : n = 7) : 
  (∃ (arrangement_count : Nat), 
    (∀ (A B C : Nat) (others : Finset Nat), 
      A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
      others.card = n - 3 ∧
      (∀ x ∈ others, x ≠ A ∧ x ≠ B ∧ x ≠ C) ∧
      (∀ perm : List Nat, perm.length = n →
        (perm.indexOf A).succ ≠ perm.indexOf B ∧
        (perm.indexOf A).pred ≠ perm.indexOf B ∧
        ((perm.indexOf B).succ = perm.indexOf C ∨
         (perm.indexOf B).pred = perm.indexOf C)) →
    arrangement_count = 1200)) :=
by sorry

end photo_arrangements_l397_39797


namespace composite_expression_l397_39757

theorem composite_expression (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b := by
  sorry

end composite_expression_l397_39757


namespace necklace_price_l397_39717

theorem necklace_price (bracelet_price earring_price ensemble_price : ℕ)
                       (necklaces bracelets earrings ensembles : ℕ)
                       (total_revenue : ℕ) :
  bracelet_price = 15 →
  earring_price = 10 →
  ensemble_price = 45 →
  necklaces = 5 →
  bracelets = 10 →
  earrings = 20 →
  ensembles = 2 →
  total_revenue = 565 →
  ∃ (necklace_price : ℕ),
    necklace_price = 25 ∧
    necklace_price * necklaces + bracelet_price * bracelets + 
    earring_price * earrings + ensemble_price * ensembles = total_revenue :=
by
  sorry

end necklace_price_l397_39717


namespace sufficient_not_necessary_condition_l397_39753

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by
  sorry

end sufficient_not_necessary_condition_l397_39753


namespace simplify_square_root_difference_l397_39767

theorem simplify_square_root_difference : (Real.sqrt 8 - Real.sqrt (4 + 1/2))^2 = 1/2 := by
  sorry

end simplify_square_root_difference_l397_39767


namespace max_value_theorem_l397_39775

theorem max_value_theorem (a b : ℝ) 
  (h1 : 0 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 1 ≤ a + b ∧ a + b ≤ 4) 
  (h3 : ∀ x y : ℝ, 0 ≤ x - y ∧ x - y ≤ 1 → 1 ≤ x + y ∧ x + y ≤ 4 → x - 2*y ≤ a - 2*b) :
  8*a + 2002*b = 8 :=
sorry

end max_value_theorem_l397_39775


namespace solution_set_theorem_g_zero_range_l397_39702

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (x a : ℝ) : ℝ := f x a - |3 + a|

-- Theorem for the solution set of |x-1| + |x+3| > 6
theorem solution_set_theorem :
  {x : ℝ | |x - 1| + |x + 3| > 6} = {x | x < -4} ∪ {x | -3 < x ∧ x < 1} ∪ {x | x > 2} :=
sorry

-- Theorem for the range of a when g has a zero
theorem g_zero_range (a : ℝ) :
  (∃ x, g x a = 0) ↔ a ≥ -2 :=
sorry

end solution_set_theorem_g_zero_range_l397_39702


namespace modified_rectangle_area_l397_39706

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem about the area of a modified rectangle --/
theorem modified_rectangle_area 
  (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (modified : Rectangle), 
    (modified.length = original.length ∧ modified.width = original.width - 2) ∨
    (modified.length = original.length - 2 ∧ modified.width = original.width) ∧
    area modified = 15) :
  ∃ (final : Rectangle), 
    ((h2.choose.length = original.length ∧ h2.choose.width = original.width - 2) →
      final.length = original.length - 2 ∧ final.width = original.width) ∧
    ((h2.choose.length = original.length - 2 ∧ h2.choose.width = original.width) →
      final.length = original.length ∧ final.width = original.width - 2) ∧
    area final = 7 := by
  sorry

end modified_rectangle_area_l397_39706


namespace area_of_fourth_square_l397_39730

/-- Given two right triangles PQR and PRS sharing a common hypotenuse PR,
    where the squares on PQ, QR, and RS have areas 25, 49, and 64 square units respectively,
    prove that the area of the square on PS is 10 square units. -/
theorem area_of_fourth_square (P Q R S : ℝ × ℝ) : 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 25 →
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 49 →
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = 64 →
  (P.1 - S.1)^2 + (P.2 - S.2)^2 = 10 := by
  sorry


end area_of_fourth_square_l397_39730


namespace infimum_of_expression_l397_39779

theorem infimum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a)) + (2 / b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ (1 / (2 * a₀)) + (2 / b₀) = 9/2 :=
by sorry

end infimum_of_expression_l397_39779


namespace arithmetic_calculations_l397_39748

theorem arithmetic_calculations :
  (-(1/8 : ℚ) + 3/4 - (-(1/4)) - 5/8 = 1/4) ∧
  (-3^2 + 5 * (-6) - (-4)^2 / (-8) = -37) := by
  sorry

end arithmetic_calculations_l397_39748


namespace composition_equation_solution_l397_39792

def α : ℝ → ℝ := λ x ↦ 4 * x + 9
def β : ℝ → ℝ := λ x ↦ 9 * x + 6

theorem composition_equation_solution :
  ∃! x : ℝ, α (β x) = 8 ∧ x = -25/36 := by sorry

end composition_equation_solution_l397_39792


namespace incorrect_elimination_process_l397_39750

/-- Given a system of two linear equations in two variables, 
    prove that a specific elimination process is incorrect. -/
theorem incorrect_elimination_process 
  (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  ¬ (∃ (k : ℝ), 2 * a + b + 2 * (a - b) = 7 + 2 * k ∧ k ≠ 0) :=
sorry

end incorrect_elimination_process_l397_39750


namespace intersection_complement_when_m_3_find_m_for_given_intersection_l397_39716

-- Define set A
def A : Set ℝ := {x | |x - 2| < 3}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x < 5} :=
sorry

-- Theorem 2
theorem find_m_for_given_intersection :
  A ∩ B 8 = {x | -1 < x ∧ x < 4} :=
sorry

end intersection_complement_when_m_3_find_m_for_given_intersection_l397_39716


namespace square_area_problem_l397_39769

theorem square_area_problem (x : ℝ) (h : 4 * x^2 = 240) : x^2 + (2*x)^2 + x^2 = 360 := by
  sorry

end square_area_problem_l397_39769


namespace soda_bottle_difference_l397_39780

/-- The number of regular soda bottles in the grocery store. -/
def regular_soda : ℕ := 67

/-- The number of diet soda bottles in the grocery store. -/
def diet_soda : ℕ := 9

/-- The difference between the number of regular soda bottles and diet soda bottles. -/
def soda_difference : ℕ := regular_soda - diet_soda

theorem soda_bottle_difference : soda_difference = 58 := by
  sorry

end soda_bottle_difference_l397_39780


namespace collinear_vectors_l397_39723

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem collinear_vectors (h1 : ¬ Collinear ℝ ({0, a, b} : Set V))
    (h2 : Collinear ℝ ({0, 2 • a + k • b, a - b} : Set V)) :
  k = -2 := by
  sorry

end collinear_vectors_l397_39723


namespace odd_reciprocal_sum_diverges_exists_rearrangement_alternating_harmonic_diverges_l397_39751

open Set
open Function
open BigOperators
open Filter

def diverges_to_infinity (s : ℕ → ℝ) : Prop :=
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n > N → s n > M

theorem odd_reciprocal_sum_diverges :
  diverges_to_infinity (λ n : ℕ => ∑ k in Finset.range n, 1 / (2 * k + 1 : ℝ)) :=
sorry

theorem exists_rearrangement_alternating_harmonic_diverges :
  ∃ f : ℕ → ℕ, Bijective f ∧
    diverges_to_infinity (λ n : ℕ => ∑ k in Finset.range n, (-1 : ℝ)^(f.invFun k - 1) / f.invFun k) :=
sorry

end odd_reciprocal_sum_diverges_exists_rearrangement_alternating_harmonic_diverges_l397_39751


namespace point_not_on_line_l397_39784

theorem point_not_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  ¬ (∃ (x y : ℝ), y = m * x + b ∧ x = 0 ∧ y = -2023) :=
by sorry

end point_not_on_line_l397_39784


namespace breakfast_omelet_eggs_l397_39728

/-- The number of eggs Gus ate in total -/
def total_eggs : ℕ := 6

/-- The number of eggs in Gus's lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs in Gus's dinner -/
def dinner_eggs : ℕ := 1

/-- The number of eggs in Gus's breakfast omelet -/
def breakfast_eggs : ℕ := total_eggs - lunch_eggs - dinner_eggs

theorem breakfast_omelet_eggs :
  breakfast_eggs = 2 := by
  sorry

end breakfast_omelet_eggs_l397_39728


namespace survey_probability_l397_39761

theorem survey_probability : 
  let n : ℕ := 14  -- Total number of questions
  let k : ℕ := 10  -- Number of correct answers
  let m : ℕ := 4   -- Number of possible answers per question
  (n.choose k * (m - 1)^(n - k)) / m^n = 1001 * 3^4 / 4^14 := by
  sorry

end survey_probability_l397_39761


namespace tutor_schedule_lcm_l397_39736

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 8)) = 120 := by
  sorry

end tutor_schedule_lcm_l397_39736


namespace quadratic_range_theorem_l397_39703

/-- A quadratic function passing through specific points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The range of a quadratic function within a specific interval -/
def range_in_interval (f : ℝ → ℝ) (l u : ℝ) : Set ℝ :=
  {y | ∃ x, l < x ∧ x < u ∧ f x = y}

theorem quadratic_range_theorem (a b c : ℝ) (h : a ≠ 0) :
  quadratic_function a b c (-1) = -5 →
  quadratic_function a b c 0 = -8 →
  quadratic_function a b c 1 = -9 →
  quadratic_function a b c 3 = -5 →
  quadratic_function a b c 5 = 7 →
  range_in_interval (quadratic_function a b c) 0 5 = {y | -9 ≤ y ∧ y < 7} := by
  sorry

end quadratic_range_theorem_l397_39703


namespace multiple_properties_l397_39718

theorem multiple_properties (x y : ℤ) 
  (hx : ∃ m : ℤ, x = 6 * m) 
  (hy : ∃ n : ℤ, y = 9 * n) : 
  (∃ k : ℤ, x - y = 3 * k) ∧ 
  (∃ a b : ℤ, (∃ m : ℤ, a = 6 * m) ∧ (∃ n : ℤ, b = 9 * n) ∧ (∃ l : ℤ, a - b = 9 * l)) :=
by sorry

end multiple_properties_l397_39718


namespace motorbike_time_difference_l397_39764

theorem motorbike_time_difference :
  let distance : ℝ := 960
  let speed_slow : ℝ := 60
  let speed_fast : ℝ := 64
  let time_slow : ℝ := distance / speed_slow
  let time_fast : ℝ := distance / speed_fast
  time_slow - time_fast = 1 := by
  sorry

end motorbike_time_difference_l397_39764


namespace polynomial_simplification_l397_39762

variable (a : ℝ)

theorem polynomial_simplification :
  ((-a^3)^2 * a^3 - 4*a^2 * a^7 = -3*a^9) ∧
  ((2*a + 1) * (-2*a + 1) = 4*a^2 - 1) := by
sorry

end polynomial_simplification_l397_39762


namespace fourth_vertex_of_rectangle_l397_39738

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if four points form a rectangle --/
def is_rectangle (r : Rectangle) : Prop :=
  let (x1, y1) := r.v1
  let (x2, y2) := r.v2
  let (x3, y3) := r.v3
  let (x4, y4) := r.v4
  ((x1 = x3 ∧ x2 = x4) ∨ (x1 = x2 ∧ x3 = x4)) ∧
  ((y1 = y2 ∧ y3 = y4) ∨ (y1 = y4 ∧ y2 = y3))

/-- Theorem stating that given three vertices of a rectangle, the fourth vertex is determined --/
theorem fourth_vertex_of_rectangle (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1 = 2 ∧ y1 = 1)
  (h2 : x2 = 4 ∧ y2 = 1)
  (h3 : x3 = 2 ∧ y3 = 5) :
  ∃ (r : Rectangle), is_rectangle r ∧ 
    r.v1 = (x1, y1) ∧ r.v2 = (x2, y2) ∧ r.v3 = (x3, y3) ∧ r.v4 = (4, 5) := by
  sorry

#check fourth_vertex_of_rectangle

end fourth_vertex_of_rectangle_l397_39738


namespace whistle_search_bound_l397_39777

/-- Represents a football field -/
structure FootballField where
  length : ℝ
  width : ℝ

/-- Represents the position of an object on the field -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the referee's search process -/
def search (field : FootballField) (start : Position) (whistle : Position) : ℕ :=
  sorry

/-- Theorem stating the upper bound on the number of steps needed to find the whistle -/
theorem whistle_search_bound 
  (field : FootballField)
  (start : Position)
  (whistle : Position)
  (h_field_size : field.length = 100 ∧ field.width = 70)
  (h_start_corner : start.x = 0 ∧ start.y = 0)
  (h_whistle_on_field : whistle.x ≥ 0 ∧ whistle.x ≤ field.length ∧ whistle.y ≥ 0 ∧ whistle.y ≤ field.width)
  (d : ℝ)
  (h_initial_distance : d = Real.sqrt ((whistle.x - start.x)^2 + (whistle.y - start.y)^2)) :
  (search field start whistle) ≤ ⌊Real.sqrt 2 * (d + 1)⌋ + 4 :=
sorry

end whistle_search_bound_l397_39777


namespace expression_evaluation_l397_39794

theorem expression_evaluation :
  let x : ℚ := -3
  let y : ℚ := 1/5
  (2*x + y)^2 - (x + 2*y)*(x - 2*y) - (3*x - y)*(x - 5*y) = -12 := by
  sorry

end expression_evaluation_l397_39794


namespace greatest_multiple_of_four_with_fifth_power_less_than_2000_l397_39760

theorem greatest_multiple_of_four_with_fifth_power_less_than_2000 :
  ∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^5 < 2000 ∧ ∀ y : ℕ, y > 0 → 4 ∣ y → y^5 < 2000 → y ≤ x :=
by
  sorry

end greatest_multiple_of_four_with_fifth_power_less_than_2000_l397_39760


namespace hobby_gender_independence_l397_39763

/-- Represents the contingency table data -/
structure ContingencyTable where
  total : ℕ
  male_hobby : ℕ
  female_no_hobby : ℕ

/-- Calculates the chi-square value for the independence test -/
def chi_square (ct : ContingencyTable) : ℝ :=
  sorry

/-- Calculates the probability of selecting k males from those without a hobby -/
def prob_select_males (ct : ContingencyTable) (k : ℕ) : ℚ :=
  sorry

/-- Calculates the expected number of males selected -/
def expected_males (ct : ContingencyTable) : ℚ :=
  sorry

/-- Main theorem encompassing all parts of the problem -/
theorem hobby_gender_independence (ct : ContingencyTable) 
  (h1 : ct.total = 100) 
  (h2 : ct.male_hobby = 30) 
  (h3 : ct.female_no_hobby = 10) : 
  chi_square ct < 6.635 ∧ 
  prob_select_males ct 0 = 3/29 ∧ 
  prob_select_males ct 1 = 40/87 ∧ 
  prob_select_males ct 2 = 38/87 ∧
  expected_males ct = 4/3 :=
sorry

end hobby_gender_independence_l397_39763


namespace intersection_M_N_l397_39793

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l397_39793


namespace unique_zero_implies_a_range_l397_39708

/-- A function f(x) = 2ax² - x - 1 has exactly one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ (Set.Ioo 0 1) ∧ 2 * a * x^2 - x - 1 = 0

/-- The theorem stating that if f(x) = 2ax² - x - 1 has exactly one zero in (0, 1), 
    then a is in the interval (1, +∞) -/
theorem unique_zero_implies_a_range :
  ∀ a : ℝ, has_unique_zero_in_interval a → a ∈ Set.Ioi 1 :=
by sorry

end unique_zero_implies_a_range_l397_39708


namespace largest_of_three_l397_39798

theorem largest_of_three (a b c : ℝ) : 
  let x₁ := a
  let x₂ := if b > x₁ then b else x₁
  let x₃ := if c > x₂ then c else x₂
  x₃ = max a (max b c) := by
sorry

end largest_of_three_l397_39798


namespace area_of_triangle_OBA_l397_39758

/-- Given two points A and B in polar coordinates, prove that the area of triangle OBA is 6 --/
theorem area_of_triangle_OBA (A B : ℝ × ℝ) (h_A : A = (3, π/3)) (h_B : B = (4, π/6)) : 
  let O : ℝ × ℝ := (0, 0)
  let area := (1/2) * (A.1 * B.1) * Real.sin (B.2 - A.2)
  area = 6 := by sorry

end area_of_triangle_OBA_l397_39758


namespace building_height_ratio_l397_39710

/-- Given a flagpole and two buildings under similar shadow conditions, 
    this theorem proves that the ratio of the heights of Building A to Building B is 5:6. -/
theorem building_height_ratio 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_a_shadow : ℝ) 
  (building_b_shadow : ℝ) 
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_a_shadow_pos : 0 < building_a_shadow)
  (building_b_shadow_pos : 0 < building_b_shadow)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_a_shadow : building_a_shadow = 60)
  (h_building_b_shadow : building_b_shadow = 72) :
  (flagpole_height / flagpole_shadow * building_a_shadow) / 
  (flagpole_height / flagpole_shadow * building_b_shadow) = 5 / 6 := by
  sorry

#check building_height_ratio

end building_height_ratio_l397_39710


namespace product_equality_l397_39766

theorem product_equality (h : 213 * 16 = 3408) : 1.6 * 21.3 = 34.08 := by
  sorry

end product_equality_l397_39766


namespace thankYouCards_count_l397_39737

/-- Represents the number of items to be mailed --/
structure MailItems where
  thankYouCards : ℕ
  bills : ℕ
  rebates : ℕ
  jobApplications : ℕ

/-- Calculates the total number of stamps required --/
def totalStamps (items : MailItems) : ℕ :=
  items.thankYouCards + items.bills + 1 + items.rebates + items.jobApplications

/-- Theorem stating the number of thank you cards --/
theorem thankYouCards_count (items : MailItems) : 
  items.bills = 2 ∧ 
  items.rebates = items.bills + 3 ∧ 
  items.jobApplications = 2 * items.rebates ∧
  totalStamps items = 21 →
  items.thankYouCards = 3 := by
  sorry

end thankYouCards_count_l397_39737


namespace quadratic_real_roots_l397_39765

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - (k - 1) = 0) ↔ k ≥ 0 := by sorry

end quadratic_real_roots_l397_39765


namespace toy_car_cost_price_l397_39783

/-- The cost price of a toy car given specific pricing conditions --/
theorem toy_car_cost_price :
  ∀ (cost_price : ℝ),
  let initial_price := 2 * cost_price
  let second_day_price := 0.9 * initial_price
  let final_price := second_day_price - 360
  (final_price = 1.44 * cost_price) →
  cost_price = 1000 := by
sorry

end toy_car_cost_price_l397_39783


namespace prob_neither_red_nor_white_l397_39759

/-- The probability of drawing a ball that is neither red nor white from a bag containing
    2 red balls, 3 white balls, and 5 yellow balls. -/
theorem prob_neither_red_nor_white :
  let total_balls : ℕ := 2 + 3 + 5
  let yellow_balls : ℕ := 5
  (yellow_balls : ℚ) / total_balls = 1 / 2 := by sorry

end prob_neither_red_nor_white_l397_39759


namespace lifeguard_swim_time_l397_39796

/-- Proves the time spent swimming front crawl given total distance, speeds, and total time -/
theorem lifeguard_swim_time 
  (total_distance : ℝ) 
  (front_crawl_speed : ℝ) 
  (breaststroke_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 500)
  (h2 : front_crawl_speed = 45)
  (h3 : breaststroke_speed = 35)
  (h4 : total_time = 12) :
  ∃ (front_crawl_time : ℝ), 
    front_crawl_time * front_crawl_speed + 
    (total_time - front_crawl_time) * breaststroke_speed = total_distance ∧ 
    front_crawl_time = 8 := by
  sorry


end lifeguard_swim_time_l397_39796


namespace max_constant_inequality_l397_39790

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ a) ∧
  (∀ (b : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ b) → b ≥ Real.sqrt 2) :=
by sorry

end max_constant_inequality_l397_39790


namespace customers_left_l397_39746

theorem customers_left (initial : Nat) (remaining : Nat) : initial - remaining = 11 :=
  by
  -- Proof goes here
  sorry

end customers_left_l397_39746


namespace cricket_team_age_difference_l397_39727

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) :
  team_size = 11 →
  captain_age = 26 →
  team_avg_age = 23 →
  ∃ (wicket_keeper_age : ℕ),
    wicket_keeper_age > captain_age ∧
    (team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) + 1 = team_avg_age ∧
    wicket_keeper_age - captain_age = 3 :=
by sorry

end cricket_team_age_difference_l397_39727


namespace function_properties_l397_39719

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 4) / x

-- State the theorem
theorem function_properties (a : ℝ) :
  f a 1 = 5 →
  (a = 1 ∧
   (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) ∧
   (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end

end function_properties_l397_39719


namespace simplify_trig_expression_l397_39725

theorem simplify_trig_expression :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end simplify_trig_expression_l397_39725


namespace forgotten_item_distance_l397_39722

/-- Calculates the total distance walked when a person forgets an item halfway to school -/
def total_distance_walked (home_to_school : ℕ) : ℕ :=
  let halfway := home_to_school / 2
  halfway + halfway + home_to_school

/-- Proves that the total distance walked is 1500 meters given the conditions -/
theorem forgotten_item_distance :
  total_distance_walked 750 = 1500 := by
  sorry

#eval total_distance_walked 750

end forgotten_item_distance_l397_39722


namespace largest_integer_cube_less_than_triple_square_l397_39786

theorem largest_integer_cube_less_than_triple_square :
  ∀ n : ℤ, n > 2 → n^3 ≥ 3*n^2 ∧ 2^3 < 3*2^2 := by
  sorry

end largest_integer_cube_less_than_triple_square_l397_39786


namespace max_segments_theorem_l397_39731

/-- Represents an equilateral triangle divided into smaller equilateral triangles --/
structure DividedTriangle where
  n : ℕ  -- number of parts each side is divided into

/-- The maximum number of segments that can be marked without forming a complete smaller triangle --/
def max_marked_segments (t : DividedTriangle) : ℕ := t.n * (t.n + 1)

/-- Theorem stating the maximum number of segments that can be marked --/
theorem max_segments_theorem (t : DividedTriangle) :
  max_marked_segments t = t.n * (t.n + 1) :=
by sorry

end max_segments_theorem_l397_39731


namespace johns_age_l397_39789

def johns_age_problem (j d : ℕ) : Prop :=
  (j = d - 30) ∧ (j + d = 80)

theorem johns_age : ∃ j d : ℕ, johns_age_problem j d ∧ j = 25 := by
  sorry

end johns_age_l397_39789


namespace x_squared_coefficient_l397_39771

-- Define the polynomial expression
def poly (x : ℝ) : ℝ := 5 * (x - 2 * x^3) - 4 * (2 * x^2 - x^3 + 3 * x^6) + 3 * (5 * x^2 - 2 * x^8)

-- Theorem stating that the coefficient of x^2 in the polynomial is 7
theorem x_squared_coefficient : (deriv (deriv poly)) 0 / 2 = 7 := by
  sorry

end x_squared_coefficient_l397_39771


namespace solution_implies_result_l397_39721

theorem solution_implies_result (a b x y : ℝ) 
  (h1 : x = 1)
  (h2 : y = -2)
  (h3 : 2*a*x - 3*y = 10 - b)
  (h4 : a*x - b*y = -1) :
  (b - a)^3 = -125 := by
sorry

end solution_implies_result_l397_39721


namespace students_in_other_communities_l397_39720

theorem students_in_other_communities 
  (total_students : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) :
  total_students = 1520 →
  muslim_percent = 41/100 →
  hindu_percent = 32/100 →
  sikh_percent = 12/100 →
  (total_students : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 228 := by
  sorry

end students_in_other_communities_l397_39720


namespace first_option_cost_is_68_l397_39732

/-- Represents the car rental problem with given conditions -/
def CarRentalProblem (trip_distance : ℝ) (second_option_cost : ℝ) 
  (gas_efficiency : ℝ) (gas_cost_per_liter : ℝ) (savings : ℝ) : Prop :=
  let total_distance := 2 * trip_distance
  let gas_needed := total_distance / gas_efficiency
  let gas_cost := gas_needed * gas_cost_per_liter
  let first_option_cost := second_option_cost - savings
  first_option_cost = 68

/-- Theorem stating that the first option costs $68 per day -/
theorem first_option_cost_is_68 :
  CarRentalProblem 150 90 15 0.9 22 := by
  sorry

#check first_option_cost_is_68

end first_option_cost_is_68_l397_39732


namespace orange_shirt_cost_l397_39770

-- Define the number of students in each grade
def kindergartners : ℕ := 101
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

-- Define the cost of shirts for each grade (in cents to avoid floating-point issues)
def yellow_shirt_cost : ℕ := 500  -- $5.00
def blue_shirt_cost : ℕ := 560    -- $5.60
def green_shirt_cost : ℕ := 525   -- $5.25

-- Define the total amount spent by P.T.O. (in cents)
def total_spent : ℕ := 231700  -- $2,317.00

-- Theorem to prove
theorem orange_shirt_cost :
  (total_spent
    - (first_graders * yellow_shirt_cost
    + second_graders * blue_shirt_cost
    + third_graders * green_shirt_cost))
  / kindergartners = 580 := by
  sorry

end orange_shirt_cost_l397_39770


namespace quadratic_vertex_form_h_l397_39788

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3

/-- The scaled quadratic function -/
def g (x : ℝ) : ℝ := 4 * f x

/-- The vertex form of a quadratic function -/
def vertex_form (m h p : ℝ) (x : ℝ) : ℝ := m * (x - h)^2 + p

theorem quadratic_vertex_form_h :
  ∃ (m p : ℝ), ∀ x, g x = vertex_form m (-5/4) p x :=
sorry

end quadratic_vertex_form_h_l397_39788


namespace factorize_xm_minus_xn_l397_39724

theorem factorize_xm_minus_xn (x m n : ℝ) : x * m - x * n = x * (m - n) := by
  sorry

end factorize_xm_minus_xn_l397_39724


namespace roses_to_mother_l397_39734

def roses_problem (total_roses grandmother_roses sister_roses kept_roses : ℕ) : ℕ :=
  total_roses - (grandmother_roses + sister_roses + kept_roses)

theorem roses_to_mother :
  roses_problem 20 9 4 1 = 6 := by
  sorry

end roses_to_mother_l397_39734
