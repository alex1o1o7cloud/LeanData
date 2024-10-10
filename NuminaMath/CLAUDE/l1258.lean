import Mathlib

namespace subset_condition_l1258_125851

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem subset_condition (a : ℝ) : A a ⊆ (A a ∩ B) ↔ 6 ≤ a ∧ a ≤ 9 := by
  sorry

end subset_condition_l1258_125851


namespace arithmetic_sequence_sum_l1258_125888

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1)
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: If S_10 = S_20 in an arithmetic sequence, then S_30 = 0 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (h : seq.S 10 = seq.S 20) :
  seq.S 30 = 0 := by
  sorry


end arithmetic_sequence_sum_l1258_125888


namespace xyz_value_l1258_125864

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15 - x * y * z) :
  x * y * z = 15 := by
  sorry

end xyz_value_l1258_125864


namespace no_real_roots_quadratic_l1258_125885

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end no_real_roots_quadratic_l1258_125885


namespace vote_difference_is_42_l1258_125838

/-- Proves that the difference in votes for the bill between re-vote and original vote is 42 -/
theorem vote_difference_is_42 
  (total_members : ℕ) 
  (original_for original_against : ℕ) 
  (revote_for revote_against : ℕ) :
  total_members = 400 →
  original_for + original_against = total_members →
  original_against > original_for →
  revote_for + revote_against = total_members →
  revote_for > revote_against →
  (revote_for - revote_against) = 3 * (original_against - original_for) →
  revote_for = (11 * original_against) / 10 →
  revote_for - original_for = 42 := by
sorry


end vote_difference_is_42_l1258_125838


namespace problem_statement_l1258_125863

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) :
  x * y ≤ 9/8 ∧ 
  4^x + 2^y ≥ 4 * Real.sqrt 2 ∧ 
  x / y + 1 / x ≥ 2/3 + 2 * Real.sqrt 3 / 3 := by
  sorry

end problem_statement_l1258_125863


namespace all_chameleons_green_chameleon_color_convergence_l1258_125858

/-- Represents the colors of chameleons --/
inductive Color
| Yellow
| Red
| Green

/-- Represents the state of chameleons on the island --/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons --/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons --/
def totalChameleons : Nat := 34

/-- Function to model the color change when two different colored chameleons meet --/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Red => Color.Green
  | Color.Red, Color.Yellow => Color.Green
  | Color.Yellow, Color.Green => Color.Red
  | Color.Green, Color.Yellow => Color.Red
  | Color.Red, Color.Green => Color.Yellow
  | Color.Green, Color.Red => Color.Yellow
  | _, _ => c1  -- No change if same color

/-- Theorem stating that all chameleons will eventually be green --/
theorem all_chameleons_green (finalState : ChameleonState) : 
  (finalState.yellow + finalState.red + finalState.green = totalChameleons) →
  (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
sorry

/-- Main theorem to prove --/
theorem chameleon_color_convergence :
  ∃ (finalState : ChameleonState),
    (finalState.yellow + finalState.red + finalState.green = totalChameleons) ∧
    (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
sorry

end all_chameleons_green_chameleon_color_convergence_l1258_125858


namespace union_of_A_and_B_l1258_125844

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end union_of_A_and_B_l1258_125844


namespace otimes_inequality_solution_set_l1258_125825

-- Define the custom operation ⊗
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem otimes_inequality_solution_set :
  ∀ x : ℝ, (otimes (x - 2) (x + 2) < 2) ↔ (x < 0 ∨ x > 1) :=
by sorry

end otimes_inequality_solution_set_l1258_125825


namespace f_positive_all_reals_f_positive_interval_l1258_125822

/-- The quadratic function f(x) = x^2 + 2(a-2)x + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 4

/-- Theorem 1: f(x) > 0 for all x ∈ ℝ if and only if 0 < a < 4 -/
theorem f_positive_all_reals (a : ℝ) :
  (∀ x : ℝ, f a x > 0) ↔ (0 < a ∧ a < 4) :=
sorry

/-- Theorem 2: f(x) > 0 for x ∈ [-3, 1] if and only if a ∈ (-1/2, 4) -/
theorem f_positive_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 1 → f a x > 0) ↔ (a > -1/2 ∧ a < 4) :=
sorry

end f_positive_all_reals_f_positive_interval_l1258_125822


namespace no_line_exists_l1258_125859

-- Define the points and curve
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)
def trajectory (x y : ℝ) : Prop := y^2 = -4*x

-- Define the line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 8)

-- Define the dot product of vectors QM and QN
def dot_product_QM_QN (M N : ℝ × ℝ) : ℝ :=
  (M.1 + 1) * (N.1 + 1) + M.2 * N.2

-- Theorem statement
theorem no_line_exists : ¬ ∃ (k : ℝ) (M N : ℝ × ℝ),
  M ≠ N ∧
  trajectory M.1 M.2 ∧
  trajectory N.1 N.2 ∧
  line_through_A k M.1 M.2 ∧
  line_through_A k N.1 N.2 ∧
  dot_product_QM_QN M N = 97 :=
by sorry

end no_line_exists_l1258_125859


namespace book_distribution_theorem_l1258_125875

/-- The number of ways to choose 3 books from 5 different books for 3 students -/
def choose_books : ℕ := 60

/-- The number of ways to buy 3 books from 5 different books for 3 students -/
def buy_books : ℕ := 125

/-- The number of different books available -/
def num_books : ℕ := 5

/-- The number of students receiving books -/
def num_students : ℕ := 3

theorem book_distribution_theorem :
  (choose_books = num_books * (num_books - 1) * (num_books - 2)) ∧
  (buy_books = num_books * num_books * num_books) := by
  sorry

end book_distribution_theorem_l1258_125875


namespace quiche_volume_l1258_125833

/-- Calculates the total volume of a quiche given the ingredients' volumes and spinach reduction factor. -/
theorem quiche_volume 
  (raw_spinach : ℝ) 
  (reduction_factor : ℝ) 
  (cream_cheese : ℝ) 
  (eggs : ℝ) 
  (h1 : 0 < reduction_factor) 
  (h2 : reduction_factor < 1) :
  raw_spinach * reduction_factor + cream_cheese + eggs = 
  (raw_spinach * reduction_factor + cream_cheese + eggs) := by
  sorry

end quiche_volume_l1258_125833


namespace ferris_wheel_cost_calculation_l1258_125836

/-- The cost of a Ferris wheel ride in tickets -/
def ferris_wheel_cost : ℕ := sorry

/-- The number of Ferris wheel rides -/
def ferris_wheel_rides : ℕ := 2

/-- The cost of a roller coaster ride in tickets -/
def roller_coaster_cost : ℕ := 5

/-- The number of roller coaster rides -/
def roller_coaster_rides : ℕ := 3

/-- The cost of a log ride in tickets -/
def log_ride_cost : ℕ := 1

/-- The number of log rides -/
def log_ride_rides : ℕ := 7

/-- The initial number of tickets Dolly has -/
def initial_tickets : ℕ := 20

/-- The number of additional tickets Dolly buys -/
def additional_tickets : ℕ := 6

theorem ferris_wheel_cost_calculation :
  ferris_wheel_cost = 2 :=
sorry

end ferris_wheel_cost_calculation_l1258_125836


namespace polynomial_properties_l1258_125884

/-- Definition of the polynomial -/
def p (x y : ℝ) : ℝ := -5*x^2 - x*y^4 + 2^6*x*y + 3

/-- The number of terms in the polynomial -/
def num_terms : ℕ := 4

/-- The degree of the polynomial -/
def degree : ℕ := 5

/-- The coefficient of the highest degree term -/
def highest_coeff : ℝ := -1

/-- Theorem stating the properties of the polynomial -/
theorem polynomial_properties :
  (num_terms = 4) ∧ 
  (degree = 5) ∧ 
  (highest_coeff = -1) := by sorry

end polynomial_properties_l1258_125884


namespace base_seven_234567_equals_41483_l1258_125818

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_234567_equals_41483 :
  base_seven_to_decimal [7, 6, 5, 4, 3, 2] = 41483 := by
  sorry

end base_seven_234567_equals_41483_l1258_125818


namespace bottles_lasted_74_days_l1258_125813

/-- The number of bottles Debby bought -/
def total_bottles : ℕ := 8066

/-- The number of bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := total_bottles / bottles_per_day

theorem bottles_lasted_74_days : days_lasted = 74 := by
  sorry

end bottles_lasted_74_days_l1258_125813


namespace quadratic_root_implies_m_l1258_125840

theorem quadratic_root_implies_m (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 3 * x - 5 * m + 4 = 0 → x = 2 ∨ x ≠ 2) →
  ((m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) →
  m = 6 := by
sorry

end quadratic_root_implies_m_l1258_125840


namespace pasture_rent_problem_l1258_125897

/-- Represents the number of oxen each person puts in the pasture -/
structure OxenCount where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of months each person's oxen graze -/
structure GrazingMonths where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total oxen-months for all three people -/
def totalOxenMonths (oxen : OxenCount) (months : GrazingMonths) : ℕ :=
  oxen.a * months.a + oxen.b * months.b + oxen.c * months.c

/-- Calculates a person's share of the rent based on their oxen-months -/
def rentShare (totalRent : ℚ) (oxenMonths : ℕ) (totalOxenMonths : ℕ) : ℚ :=
  totalRent * (oxenMonths : ℚ) / (totalOxenMonths : ℚ)

theorem pasture_rent_problem (totalRent : ℚ) (oxen : OxenCount) (months : GrazingMonths) 
    (h1 : totalRent = 175)
    (h2 : oxen.a = 10 ∧ oxen.b = 12 ∧ oxen.c = 15)
    (h3 : months.a = 7 ∧ months.b = 5)
    (h4 : rentShare totalRent (oxen.c * months.c) (totalOxenMonths oxen months) = 45) :
    months.c = 3 := by
  sorry

end pasture_rent_problem_l1258_125897


namespace sin_2x_equals_plus_minus_one_l1258_125869

/-- Given vectors a and b, if a is a non-zero scalar multiple of b, then sin(2x) = ±1 -/
theorem sin_2x_equals_plus_minus_one (x : ℝ) :
  let a : ℝ × ℝ := (Real.cos x, -Real.sin x)
  let b : ℝ × ℝ := (-Real.cos (π/2 - x), Real.cos x)
  ∀ t : ℝ, t ≠ 0 → a = t • b → Real.sin (2*x) = 1 ∨ Real.sin (2*x) = -1 :=
by sorry

end sin_2x_equals_plus_minus_one_l1258_125869


namespace gcd_of_powers_of_47_l1258_125868

theorem gcd_of_powers_of_47 : Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_47_l1258_125868


namespace square_plus_reciprocal_square_l1258_125870

theorem square_plus_reciprocal_square (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 6 = 104 := by
sorry

end square_plus_reciprocal_square_l1258_125870


namespace helen_raisin_cookies_l1258_125895

/-- The number of raisin cookies Helen baked yesterday -/
def raisin_cookies_yesterday : ℕ := sorry

/-- The number of chocolate chip cookies Helen baked yesterday -/
def choc_chip_cookies_yesterday : ℕ := 519

/-- The number of raisin cookies Helen baked today -/
def raisin_cookies_today : ℕ := 280

/-- The number of chocolate chip cookies Helen baked today -/
def choc_chip_cookies_today : ℕ := 359

/-- Helen baked 20 more raisin cookies yesterday compared to today -/
axiom raisin_cookies_difference : raisin_cookies_yesterday = raisin_cookies_today + 20

theorem helen_raisin_cookies : raisin_cookies_yesterday = 300 := by sorry

end helen_raisin_cookies_l1258_125895


namespace ice_cream_theorem_l1258_125857

theorem ice_cream_theorem (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end ice_cream_theorem_l1258_125857


namespace inequality_proof_l1258_125824

theorem inequality_proof (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hx₃ : x₃ > 0)
  (hy₁ : y₁ > 0) (hy₂ : y₂ > 0) (hy₃ : y₃ > 0)
  (hz₁ : z₁ > 0) (hz₂ : z₂ > 0) (hz₃ : z₃ > 0) :
  (x₁^3 + x₂^3 + x₃^3 + 1) * (y₁^3 + y₂^3 + y₃^3 + 1) * (z₁^3 + z₂^3 + z₃^3 + 1) ≥ 
  (9/2) * (x₁ + y₁ + z₁) * (x₂ + y₂ + z₂) * (x₃ + y₃ + z₃) :=
by sorry

end inequality_proof_l1258_125824


namespace fibonacci_like_sequence_b9_l1258_125814

def fibonacci_like_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) = b (n + 1) + b n

theorem fibonacci_like_sequence_b9 (b : ℕ → ℕ) :
  fibonacci_like_sequence b →
  (∀ n m : ℕ, n < m → b n < b m) →
  b 8 = 100 →
  b 9 = 194 := by sorry

end fibonacci_like_sequence_b9_l1258_125814


namespace triangle_angle_calculation_l1258_125804

theorem triangle_angle_calculation (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hC : 0 < B) (hD : B < π) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = π / 3) :
  ∃ (A : ℝ), 
    0 < A ∧ A < π / 2 ∧ 
    Real.sin A = (a * Real.sin B) / b ∧
    A = π / 4 :=
sorry

end triangle_angle_calculation_l1258_125804


namespace quadratic_equation_proof_l1258_125820

theorem quadratic_equation_proof (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 + (2*k - 1)*x + k^2 - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end quadratic_equation_proof_l1258_125820


namespace intersection_of_P_and_Q_l1258_125880

-- Define the sets P and Q
def P : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Theorem statement
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end intersection_of_P_and_Q_l1258_125880


namespace mayor_approval_probability_l1258_125856

/-- The probability of a voter approving the mayor's work -/
def p_approve : ℝ := 0.6

/-- The number of voters selected -/
def n_voters : ℕ := 4

/-- The number of approvals we're interested in -/
def k_approvals : ℕ := 2

/-- The probability of exactly k_approvals in n_voters independent trials -/
def prob_k_approvals (p : ℝ) (n k : ℕ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem mayor_approval_probability :
  prob_k_approvals p_approve n_voters k_approvals = 0.3456 := by
  sorry

end mayor_approval_probability_l1258_125856


namespace units_digit_3_pow_20_l1258_125809

def units_digit_pattern : ℕ → ℕ
| 0 => 3
| 1 => 9
| 2 => 7
| 3 => 1
| n + 4 => units_digit_pattern n

theorem units_digit_3_pow_20 :
  units_digit_pattern 19 = 1 :=
by sorry

end units_digit_3_pow_20_l1258_125809


namespace max_n_for_L_perfect_square_l1258_125879

/-- Definition of L(n): the number of permutations of {1,2,...,n} with exactly one landmark point -/
def L (n : ℕ) : ℕ := 4 * (2^(n-2) - 1)

/-- Theorem stating that 3 is the maximum n ≥ 3 for which L(n) is a perfect square -/
theorem max_n_for_L_perfect_square :
  ∀ n : ℕ, n ≥ 3 → (∃ k : ℕ, L n = k^2) → n ≤ 3 :=
sorry

end max_n_for_L_perfect_square_l1258_125879


namespace circular_window_panes_l1258_125811

theorem circular_window_panes (r : ℝ) (x : ℝ) : 
  r = 20 → 
  (9 : ℝ) * (π * r^2) = π * (r + x)^2 → 
  x = 40 :=
by sorry

end circular_window_panes_l1258_125811


namespace missy_watch_time_l1258_125865

/-- The total time Missy spends watching TV -/
def total_watch_time (num_reality_shows : ℕ) (reality_show_duration : ℕ) (num_cartoons : ℕ) (cartoon_duration : ℕ) : ℕ :=
  num_reality_shows * reality_show_duration + num_cartoons * cartoon_duration

/-- Theorem stating that Missy spends 150 minutes watching TV -/
theorem missy_watch_time :
  total_watch_time 5 28 1 10 = 150 := by
  sorry

end missy_watch_time_l1258_125865


namespace notebook_cost_l1258_125849

theorem notebook_cost (total_students : ℕ) (total_cost : ℕ) 
  (h1 : total_students = 36)
  (h2 : ∃ (buyers : ℕ) (notebooks_per_student : ℕ) (cost_per_notebook : ℕ),
    buyers > total_students / 2 ∧
    notebooks_per_student > 1 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost)
  (h3 : total_cost = 2310) :
  ∃ (cost_per_notebook : ℕ), cost_per_notebook = 11 ∧
    ∃ (buyers : ℕ) (notebooks_per_student : ℕ),
      buyers > total_students / 2 ∧
      notebooks_per_student > 1 ∧
      cost_per_notebook > notebooks_per_student ∧
      buyers * notebooks_per_student * cost_per_notebook = total_cost :=
by sorry

end notebook_cost_l1258_125849


namespace chess_tournament_games_l1258_125815

/-- Proves that in a group of 9 players, if each player plays every other player
    the same number of times, and a total of 36 games are played, then each
    player must play every other player exactly once. -/
theorem chess_tournament_games (n : ℕ) (total_games : ℕ) 
    (h1 : n = 9)
    (h2 : total_games = 36)
    (h3 : ∀ i j : Fin n, i ≠ j → ∃ k : ℕ, k > 0) :
  ∀ i j : Fin n, i ≠ j → ∃ k : ℕ, k = 1 := by
  sorry

#check chess_tournament_games

end chess_tournament_games_l1258_125815


namespace smallest_crate_dimension_l1258_125854

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.radius * 2 ≤ crate.height)

theorem smallest_crate_dimension (x : ℝ) :
  let crate := CrateDimensions.mk x 8 12
  let cylinder := Cylinder.mk 6 (max x (max 8 12))
  cylinderFitsInCrate crate cylinder →
  min x (min 8 12) = 8 := by
  sorry

#check smallest_crate_dimension

end smallest_crate_dimension_l1258_125854


namespace four_digit_divisible_by_12_l1258_125891

def is_valid_number (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ a + b = 11 ∧ (a * 1000 + 520 + b) % 12 = 0

theorem four_digit_divisible_by_12 :
  ∀ a b : ℕ, is_valid_number a b → (a = 7 ∧ b = 4) ∨ (a = 3 ∧ b = 8) :=
by sorry

end four_digit_divisible_by_12_l1258_125891


namespace hannah_unique_number_l1258_125817

/-- Represents a student's counting sequence -/
structure StudentSequence where
  start : Nat
  step : Nat

/-- The set of all numbers from 1 to 1200 -/
def allNumbers : Set Nat := {n | 1 ≤ n ∧ n ≤ 1200}

/-- Generate a sequence for a student -/
def generateSequence (s : StudentSequence) : Set Nat :=
  {n ∈ allNumbers | ∃ k, n = s.start + k * s.step}

/-- Alice's sequence -/
def aliceSeq : Set Nat := allNumbers \ (generateSequence ⟨4, 4⟩)

/-- Barbara's sequence -/
def barbaraSeq : Set Nat := (allNumbers \ aliceSeq) \ (generateSequence ⟨5, 5⟩)

/-- Candice's sequence -/
def candiceSeq : Set Nat := (allNumbers \ (aliceSeq ∪ barbaraSeq)) \ (generateSequence ⟨6, 6⟩)

/-- Debbie, Eliza, and Fatima's combined sequence -/
def defSeq : Set Nat := 
  (allNumbers \ (aliceSeq ∪ barbaraSeq ∪ candiceSeq)) \ 
  (generateSequence ⟨7, 7⟩ ∪ generateSequence ⟨14, 7⟩ ∪ generateSequence ⟨21, 7⟩)

/-- George's sequence -/
def georgeSeq : Set Nat := allNumbers \ (aliceSeq ∪ barbaraSeq ∪ candiceSeq ∪ defSeq)

/-- Hannah's number -/
def hannahNumber : Nat := 1189

/-- Theorem: Hannah's number is the only number not spoken by any other student -/
theorem hannah_unique_number : 
  hannahNumber ∈ allNumbers ∧ 
  hannahNumber ∉ aliceSeq ∧ 
  hannahNumber ∉ barbaraSeq ∧ 
  hannahNumber ∉ candiceSeq ∧ 
  hannahNumber ∉ defSeq ∧ 
  hannahNumber ∉ georgeSeq ∧
  ∀ n ∈ allNumbers, n ≠ hannahNumber → 
    n ∈ aliceSeq ∨ n ∈ barbaraSeq ∨ n ∈ candiceSeq ∨ n ∈ defSeq ∨ n ∈ georgeSeq := by
  sorry

end hannah_unique_number_l1258_125817


namespace free_younger_son_time_l1258_125898

/-- The time required to cut all strands of duct tape -/
def cut_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  total_strands / (hannah_rate + son_rate)

/-- Theorem stating that it takes 2 minutes to cut 22 strands of duct tape -/
theorem free_younger_son_time :
  cut_time 22 8 3 = 2 := by sorry

end free_younger_son_time_l1258_125898


namespace symmetric_point_coordinates_l1258_125899

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point2D) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetric_point_coordinates :
  let M : Point2D := ⟨1, -2⟩
  let N : Point2D := ⟨-1, 2⟩
  symmetricToOrigin M N → N = ⟨-1, 2⟩ := by
  sorry

end symmetric_point_coordinates_l1258_125899


namespace sphere_radius_ratio_l1258_125890

theorem sphere_radius_ratio (v_large v_small : ℝ) (h1 : v_large = 432 * Real.pi) (h2 : v_small = 0.25 * v_large) :
  (∃ r_small r_large : ℝ, 
    v_small = (4/3) * Real.pi * r_small^3 ∧ 
    v_large = (4/3) * Real.pi * r_large^3 ∧ 
    r_small / r_large = 1 / (2^(2/3))) := by
  sorry

end sphere_radius_ratio_l1258_125890


namespace walking_distance_l1258_125839

theorem walking_distance (original_speed original_distance increased_speed additional_distance : ℝ) 
  (h1 : original_speed = 4)
  (h2 : increased_speed = 5)
  (h3 : additional_distance = 6)
  (h4 : original_distance / original_speed = (original_distance + additional_distance) / increased_speed) :
  original_distance = 24 := by
sorry

end walking_distance_l1258_125839


namespace paint_calculation_l1258_125878

/-- The amount of white paint needed in ounces -/
def white_paint : ℕ := 20

/-- The amount of green paint needed in ounces -/
def green_paint : ℕ := 15

/-- The amount of brown paint needed in ounces -/
def brown_paint : ℕ := 34

/-- The total amount of paint needed in ounces -/
def total_paint : ℕ := white_paint + green_paint + brown_paint

theorem paint_calculation : total_paint = 69 := by
  sorry

end paint_calculation_l1258_125878


namespace line_equation_with_opposite_intercepts_l1258_125808

/-- A line passing through a given point with opposite intercepts on the coordinate axes -/
structure LineWithOppositeIntercepts where
  -- The x-coordinate of the point
  x : ℝ
  -- The y-coordinate of the point
  y : ℝ
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the point (x, y)
  point_on_line : a * x + b * y + c = 0
  -- The intercepts are opposite in value
  opposite_intercepts : a * c = -b * c ∨ a = 0 ∧ b = 0 ∧ c = 0

/-- The equation of a line with opposite intercepts passing through (3, -2) -/
theorem line_equation_with_opposite_intercepts :
  ∀ (l : LineWithOppositeIntercepts),
  l.x = 3 ∧ l.y = -2 →
  (l.a = 2 ∧ l.b = 3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = -5) := by
  sorry

end line_equation_with_opposite_intercepts_l1258_125808


namespace min_value_of_fraction_l1258_125842

theorem min_value_of_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ m : ℝ, m = 1 - Real.sqrt 2 ∧ ∀ z : ℝ, z = x*y/(x+y-2) → m ≤ z :=
sorry

end min_value_of_fraction_l1258_125842


namespace boat_upstream_time_l1258_125807

theorem boat_upstream_time (B C : ℝ) (h1 : B = 4 * C) (h2 : B > 0) (h3 : C > 0) : 
  (10 : ℝ) * (B + C) / (B - C) = 50 / 3 := by
  sorry

end boat_upstream_time_l1258_125807


namespace ratio_from_mean_ratio_l1258_125892

theorem ratio_from_mean_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a + b) / 2 / Real.sqrt (a * b) = 25 / 24 →
  a / b = 16 / 9 ∨ a / b = 9 / 16 := by
sorry

end ratio_from_mean_ratio_l1258_125892


namespace min_xy_value_least_xy_value_l1258_125834

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) :
  ∀ (a b : ℕ+), ((1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6) → (x : ℕ) * y ≤ (a : ℕ) * b :=
by
  sorry

theorem least_xy_value :
  ∃ (x y : ℕ+), ((1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 6) ∧ (x : ℕ) * y = 90 :=
by
  sorry

end min_xy_value_least_xy_value_l1258_125834


namespace probability_sum_greater_than_9_l1258_125847

def number_set : Finset ℕ := {1, 3, 5, 7, 9}

def sum_greater_than_9 (a b : ℕ) : Prop := a + b > 9

def valid_pair (a b : ℕ) : Prop := a ∈ number_set ∧ b ∈ number_set ∧ a ≠ b

theorem probability_sum_greater_than_9 :
  Nat.card {p : ℕ × ℕ | p.1 < p.2 ∧ valid_pair p.1 p.2 ∧ sum_greater_than_9 p.1 p.2} /
  Nat.card {p : ℕ × ℕ | p.1 < p.2 ∧ valid_pair p.1 p.2} = 3 / 5 := by
  sorry

end probability_sum_greater_than_9_l1258_125847


namespace rectangle_area_lower_bound_l1258_125812

theorem rectangle_area_lower_bound 
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a * x = 1) 
  (eq2 : c * x = 3) 
  (eq3 : b * y = 10) 
  (eq4 : a * z = 9) : 
  (a + b + c) * (x + y + z) ≥ 90 := by
  sorry

end rectangle_area_lower_bound_l1258_125812


namespace smallest_n_congruence_l1258_125871

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 13 * m ≡ 456 [ZMOD 5] → n ≤ m) ∧ 
  13 * n ≡ 456 [ZMOD 5] := by
  sorry

end smallest_n_congruence_l1258_125871


namespace intersection_A_complementB_l1258_125883

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Define the complement of B with respect to U
def complementB : Set ℝ := U \ B

-- Theorem statement
theorem intersection_A_complementB : A ∩ complementB = {x | -1 < x ∧ x < 2} := by
  sorry

end intersection_A_complementB_l1258_125883


namespace intersection_condition_union_condition_l1258_125853

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a-1)*x + (a^2-5) = 0}

-- Theorem for part (1)
theorem intersection_condition (a : ℝ) : A ∩ B a = {2} → a = -5 ∨ a = 1 := by sorry

-- Theorem for part (2)
theorem union_condition (a : ℝ) : A ∪ B a = A → a > 3 := by sorry

end intersection_condition_union_condition_l1258_125853


namespace red_pigment_in_brown_l1258_125843

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : Real
  red : Real
  yellow : Real
  weight : Real

/-- The sky blue paint composition -/
def skyBlue : PaintMixture := {
  blue := 0.1
  red := 0.9
  yellow := 0
  weight := 1
}

/-- The green paint composition -/
def green : PaintMixture := {
  blue := 0.7
  red := 0
  yellow := 0.3
  weight := 1
}

/-- The resulting brown paint composition -/
def brown : PaintMixture := {
  blue := 0.4
  red := 0
  yellow := 0
  weight := 10
}

/-- Theorem stating the amount of red pigment in the brown paint -/
theorem red_pigment_in_brown :
  ∃ (x y : Real),
    x + y = brown.weight ∧
    x * skyBlue.blue + y * green.blue = brown.blue * brown.weight ∧
    x * skyBlue.red = 4.5 := by
  sorry


end red_pigment_in_brown_l1258_125843


namespace frank_lamp_purchase_l1258_125821

/-- Frank's lamp purchase problem -/
theorem frank_lamp_purchase (frank_money : ℕ) (cheapest_lamp : ℕ) (expensive_factor : ℕ) :
  frank_money = 90 →
  cheapest_lamp = 20 →
  expensive_factor = 3 →
  frank_money - (cheapest_lamp * expensive_factor) = 30 := by
  sorry

end frank_lamp_purchase_l1258_125821


namespace smallest_number_with_conditions_l1258_125886

def is_prime (n : ℕ) : Prop := sorry

def is_cube (n : ℕ) : Prop := sorry

def ends_with (a b : ℕ) : Prop := sorry

def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_number_with_conditions (p : ℕ) (hp_prime : is_prime p) (hp_cube : is_cube p) :
  ∃ (A : ℕ), 
    A = 11713 ∧ 
    p = 13 ∧
    p ∣ A ∧ 
    ends_with A p ∧ 
    digit_sum A = p ∧ 
    ∀ (B : ℕ), (p ∣ B ∧ ends_with B p ∧ digit_sum B = p) → A ≤ B :=
by sorry

end smallest_number_with_conditions_l1258_125886


namespace rosys_age_l1258_125828

/-- Proves Rosy's current age given the conditions of the problem -/
theorem rosys_age :
  ∀ (rosy_age : ℕ),
  (∃ (david_age : ℕ),
    david_age = rosy_age + 18 ∧
    david_age + 6 = 2 * (rosy_age + 6)) →
  rosy_age = 12 :=
by
  sorry

end rosys_age_l1258_125828


namespace profit_at_35_selling_price_for_600_profit_no_900_profit_l1258_125850

/-- Represents the daily sales and profit model for a product in a shopping mall. -/
structure SalesModel where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_volume : ℝ → ℝ
  profit : ℝ → ℝ

/-- The specific sales model for the given problem. -/
def mall_model : SalesModel :=
  { purchase_price := 30
    min_selling_price := 30
    max_selling_price := 55
    sales_volume := fun x => -2 * x + 140
    profit := fun x => (x - 30) * (-2 * x + 140) }

/-- Theorem 1: The daily profit when the selling price is 35 yuan is 350 yuan. -/
theorem profit_at_35 (model : SalesModel := mall_model) :
    model.profit 35 = 350 := by sorry

/-- Theorem 2: The selling price that yields a daily profit of 600 yuan is 40 yuan. -/
theorem selling_price_for_600_profit (model : SalesModel := mall_model) :
    ∃ x, model.min_selling_price ≤ x ∧ x ≤ model.max_selling_price ∧ model.profit x = 600 ∧ x = 40 := by sorry

/-- Theorem 3: There is no selling price within the given range that can yield a daily profit of 900 yuan. -/
theorem no_900_profit (model : SalesModel := mall_model) :
    ¬∃ x, model.min_selling_price ≤ x ∧ x ≤ model.max_selling_price ∧ model.profit x = 900 := by sorry

end profit_at_35_selling_price_for_600_profit_no_900_profit_l1258_125850


namespace total_votes_is_129_l1258_125832

/-- The number of votes for each cake type in a baking contest. -/
structure CakeVotes where
  witch : ℕ
  unicorn : ℕ
  dragon : ℕ
  mermaid : ℕ
  fairy : ℕ
  phoenix : ℕ

/-- The conditions for the cake voting contest. -/
def contestConditions (votes : CakeVotes) : Prop :=
  votes.witch = 15 ∧
  votes.unicorn = 3 * votes.witch ∧
  votes.dragon = votes.witch + 7 ∧
  votes.dragon = (votes.mermaid * 5) / 4 ∧
  votes.mermaid = votes.dragon - 3 ∧
  votes.mermaid = 2 * votes.fairy ∧
  votes.fairy = votes.witch - 5 ∧
  votes.phoenix = votes.dragon - (votes.dragon / 5) ∧
  votes.phoenix = votes.fairy + 15

/-- The theorem stating that given the contest conditions, the total number of votes is 129. -/
theorem total_votes_is_129 (votes : CakeVotes) :
  contestConditions votes → votes.witch + votes.unicorn + votes.dragon + votes.mermaid + votes.fairy + votes.phoenix = 129 := by
  sorry


end total_votes_is_129_l1258_125832


namespace festival_attendance_l1258_125835

theorem festival_attendance (total : ℕ) (first_day : ℕ) : 
  total = 2700 →
  first_day + (first_day / 2) + (3 * first_day) = total →
  first_day / 2 = 300 :=
by sorry

end festival_attendance_l1258_125835


namespace decimal_24_equals_binary_11000_l1258_125802

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_24_equals_binary_11000 : 
  to_binary 24 = [false, false, false, true, true] ∧ 
  from_binary [false, false, false, true, true] = 24 := by
  sorry

end decimal_24_equals_binary_11000_l1258_125802


namespace paradise_park_large_seats_l1258_125873

/-- Represents a Ferris wheel with small and large seats. -/
structure FerrisWheel where
  smallSeats : Nat
  largeSeats : Nat
  smallSeatCapacity : Nat
  largeSeatCapacity : Nat
  totalLargeSeatCapacity : Nat

/-- The Ferris wheel in paradise park -/
def paradiseParkFerrisWheel : FerrisWheel := {
  smallSeats := 3
  largeSeats := 0  -- We don't know this value yet
  smallSeatCapacity := 16
  largeSeatCapacity := 12
  totalLargeSeatCapacity := 84
}

/-- Theorem: The number of large seats on the paradise park Ferris wheel is 7 -/
theorem paradise_park_large_seats : 
  paradiseParkFerrisWheel.largeSeats = 
    paradiseParkFerrisWheel.totalLargeSeatCapacity / paradiseParkFerrisWheel.largeSeatCapacity := by
  sorry

end paradise_park_large_seats_l1258_125873


namespace function_zero_implies_a_range_l1258_125805

theorem function_zero_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1) 1 ∧ 2 * a * x₀ - a + 3 = 0) →
  a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
sorry

end function_zero_implies_a_range_l1258_125805


namespace city_mpg_is_14_l1258_125837

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
def calculate_city_mpg (car : CarFuelEfficiency) : ℝ :=
  sorry

/-- Theorem stating that for a car with given fuel efficiency data, 
    the city miles per gallon is 14 -/
theorem city_mpg_is_14 (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 480)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.city_mpg_difference = 6) :
  calculate_city_mpg car = 14 := by
  sorry

end city_mpg_is_14_l1258_125837


namespace investment_theorem_l1258_125830

/-- Calculates the total investment with interest after one year -/
def total_investment_with_interest (total_investment : ℝ) (amount_at_3_percent : ℝ) : ℝ :=
  let amount_at_5_percent := total_investment - amount_at_3_percent
  let interest_at_3_percent := amount_at_3_percent * 0.03
  let interest_at_5_percent := amount_at_5_percent * 0.05
  total_investment + interest_at_3_percent + interest_at_5_percent

/-- Theorem stating that the total investment with interest is $1,046 -/
theorem investment_theorem :
  total_investment_with_interest 1000 199.99999999999983 = 1046 := by
  sorry

end investment_theorem_l1258_125830


namespace largest_three_digit_geometric_sequence_l1258_125894

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem largest_three_digit_geometric_sequence :
  ∀ n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    (n / 100 = 8) ∧
    digits_are_distinct n ∧
    is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) →
    n ≤ 842 :=
sorry

end largest_three_digit_geometric_sequence_l1258_125894


namespace shoes_cost_proof_l1258_125831

def budget : ℕ := 200
def shirt_cost : ℕ := 30
def pants_cost : ℕ := 46
def coat_cost : ℕ := 38
def socks_cost : ℕ := 11
def belt_cost : ℕ := 18
def remaining : ℕ := 16

theorem shoes_cost_proof :
  budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost) - remaining = 41 := by
  sorry

end shoes_cost_proof_l1258_125831


namespace inequality_implications_l1258_125889

theorem inequality_implications (a b : ℝ) (h : a + 1 > b + 1) :
  (a > b) ∧ (a + 2 > b + 2) ∧ (-a < -b) ∧ ¬(∀ a b : ℝ, a + 1 > b + 1 → 2*a > 3*b) :=
by sorry

end inequality_implications_l1258_125889


namespace remainder_of_2745_base12_div_5_l1258_125826

/-- Converts a base 12 number to base 10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base 12 representation of 2745 --/
def number_base12 : List Nat := [2, 7, 4, 5]

theorem remainder_of_2745_base12_div_5 :
  (base12ToBase10 number_base12) % 5 = 2 := by
  sorry

end remainder_of_2745_base12_div_5_l1258_125826


namespace necessary_condition_for_greater_than_five_l1258_125816

theorem necessary_condition_for_greater_than_five (x : ℝ) :
  x > 5 → x > 3 := by sorry

end necessary_condition_for_greater_than_five_l1258_125816


namespace complex_exponential_form_l1258_125881

/-- Given a complex number z = e^a(cos b + i sin b), its exponential form is e^(a + ib) -/
theorem complex_exponential_form (a b : ℝ) :
  let z : ℂ := Complex.exp a * (Complex.cos b + Complex.I * Complex.sin b)
  z = Complex.exp (a + Complex.I * b) := by
  sorry

end complex_exponential_form_l1258_125881


namespace seating_theorem_l1258_125861

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange 10 people in a row with seating restrictions -/
def seating_arrangements : ℕ :=
  factorial 10 - 2 * (factorial 8 * factorial 3) + factorial 8 * factorial 3 * factorial 3

/-- Theorem stating the number of valid seating arrangements -/
theorem seating_theorem :
  seating_arrangements = 4596480 :=
sorry

end seating_theorem_l1258_125861


namespace equation_solution_l1258_125800

theorem equation_solution :
  ∃! y : ℝ, (4 * y - 2) / (5 * y - 5) = 3 / 4 ∧ y = -7 := by
  sorry

end equation_solution_l1258_125800


namespace a_values_l1258_125876

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define the set of possible values for a
def possible_a : Set ℝ := {-1, 0, 2/3}

-- Statement to prove
theorem a_values (a : ℝ) : (N a ⊆ M) ↔ a ∈ possible_a := by sorry

end a_values_l1258_125876


namespace binomial_expansion_constant_term_l1258_125852

theorem binomial_expansion_constant_term (n : ℕ) : 
  (∃ k : ℚ, 2 * (n.choose 2) = (n.choose 1) + k ∧ (n.choose 3) = (n.choose 2) + k) →
  (∃ r : ℕ, r ≤ n ∧ 21 = 7 * r ∧ n.choose r = 35) := by
sorry

end binomial_expansion_constant_term_l1258_125852


namespace sum_remainder_mod_13_l1258_125846

theorem sum_remainder_mod_13 : (9123 + 9124 + 9125 + 9126) % 13 = 4 := by
  sorry

end sum_remainder_mod_13_l1258_125846


namespace square_plus_four_equals_54_l1258_125848

theorem square_plus_four_equals_54 (x : ℝ) (h : x = 5) : 2 * x^2 + 4 = 54 := by
  sorry

end square_plus_four_equals_54_l1258_125848


namespace population_growth_over_three_years_l1258_125823

/-- Represents the demographic rates for a given year -/
structure YearlyRates where
  birth_rate : ℝ
  death_rate : ℝ
  in_migration : ℝ
  out_migration : ℝ

/-- Calculates the net growth rate for a given year -/
def net_growth_rate (rates : YearlyRates) : ℝ :=
  rates.birth_rate + rates.in_migration - rates.death_rate - rates.out_migration

/-- Theorem stating the net percentage increase in population over three years -/
theorem population_growth_over_three_years 
  (year1 : YearlyRates)
  (year2 : YearlyRates)
  (year3 : YearlyRates)
  (h1 : year1 = { birth_rate := 0.025, death_rate := 0.01, in_migration := 0.03, out_migration := 0.02 })
  (h2 : year2 = { birth_rate := 0.02, death_rate := 0.015, in_migration := 0.04, out_migration := 0.035 })
  (h3 : year3 = { birth_rate := 0.022, death_rate := 0.008, in_migration := 0.025, out_migration := 0.01 })
  : ∃ (ε : ℝ), abs ((1 + net_growth_rate year1) * (1 + net_growth_rate year2) * (1 + net_growth_rate year3) - 1 - 0.065675) < ε ∧ ε > 0 := by
  sorry

end population_growth_over_three_years_l1258_125823


namespace square_area_error_l1258_125893

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x + 0.38 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let percentage_error := (area_error / actual_area) * 100
  percentage_error = 90.44 := by
sorry

end square_area_error_l1258_125893


namespace wire_length_proof_l1258_125896

theorem wire_length_proof (shorter_piece : ℝ) (longer_piece : ℝ) : 
  shorter_piece = 14.285714285714285 →
  shorter_piece = (2/5) * longer_piece →
  shorter_piece + longer_piece = 50 := by
sorry

end wire_length_proof_l1258_125896


namespace equation_roots_l1258_125845

theorem equation_roots (m : ℝ) :
  ((m - 2) ≠ 0) →  -- Condition for linear equation
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ * (x₁ + 2*m) + m * (1 - x₁) - 1 = 0 ∧ 
    x₂ * (x₂ + 2*m) + m * (1 - x₂) - 1 = 0) :=
by sorry

end equation_roots_l1258_125845


namespace total_money_collected_l1258_125874

/-- Calculates the total money collected from ticket sales given the prices and attendance. -/
theorem total_money_collected 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_attendance : ℕ) 
  (children_attendance : ℕ) 
  (h1 : adult_price = 60 / 100) 
  (h2 : child_price = 25 / 100) 
  (h3 : total_attendance = 280) 
  (h4 : children_attendance = 80) :
  (total_attendance - children_attendance) * adult_price + children_attendance * child_price = 140 / 100 := by
sorry

end total_money_collected_l1258_125874


namespace k_value_theorem_l1258_125872

theorem k_value_theorem (x y z k : ℝ) 
  (h1 : (y + z) / x = k)
  (h2 : (z + x) / y = k)
  (h3 : (x + y) / z = k)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0) :
  k = 2 ∨ k = -1 := by
sorry

end k_value_theorem_l1258_125872


namespace parabola_through_points_with_parallel_tangent_l1258_125855

/-- A parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.tangent_slope (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

/-- Theorem stating the conditions and the result to be proved -/
theorem parabola_through_points_with_parallel_tangent 
  (p : Parabola) 
  (h1 : p.y_coord 1 = 1) 
  (h2 : p.y_coord 2 = -1) 
  (h3 : p.tangent_slope 2 = 1) : 
  p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry


end parabola_through_points_with_parallel_tangent_l1258_125855


namespace river_road_cars_l1258_125829

/-- Proves that the number of cars on River Road is 60 -/
theorem river_road_cars :
  ∀ (buses cars motorcycles : ℕ),
    (buses : ℚ) / cars = 1 / 3 →
    cars = buses + 40 →
    buses + cars + motorcycles = 720 →
    cars = 60 := by
  sorry

end river_road_cars_l1258_125829


namespace paige_finished_problems_l1258_125841

/-- Given that Paige had 43 math problems, 12 science problems, and 11 problems left to do for homework,
    prove that she finished 44 problems at school. -/
theorem paige_finished_problems (math_problems : ℕ) (science_problems : ℕ) (problems_left : ℕ)
  (h1 : math_problems = 43)
  (h2 : science_problems = 12)
  (h3 : problems_left = 11) :
  math_problems + science_problems - problems_left = 44 := by
  sorry

end paige_finished_problems_l1258_125841


namespace square_perimeter_sum_l1258_125806

theorem square_perimeter_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 65) (h4 : a^2 - b^2 = 33) : 
  4*a + 4*b = 44 := by
sorry

end square_perimeter_sum_l1258_125806


namespace vector_operation_l1258_125882

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_operation :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end vector_operation_l1258_125882


namespace complex_equation_l1258_125801

theorem complex_equation : Complex.I ^ 3 + 2 * Complex.I = Complex.I := by
  sorry

end complex_equation_l1258_125801


namespace circle_segment_distance_squared_l1258_125862

theorem circle_segment_distance_squared (r AB BC : ℝ) (angle_ABC : ℝ) : 
  r = Real.sqrt 75 →
  AB = 7 →
  BC = 3 →
  angle_ABC = 2 * Real.pi / 3 →
  ∃ (O B : ℝ × ℝ), 
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = 61 :=
by sorry

end circle_segment_distance_squared_l1258_125862


namespace percentage_commutation_l1258_125887

theorem percentage_commutation (x : ℝ) (h : 0.30 * 0.15 * x = 18) :
  0.15 * 0.30 * x = 18 := by
  sorry

end percentage_commutation_l1258_125887


namespace mixed_number_calculation_l1258_125867

theorem mixed_number_calculation : 
  53 * ((3 + 1/5) - (4 + 1/2)) / ((2 + 3/4) + (1 + 2/3)) = -(15 + 3/5) := by sorry

end mixed_number_calculation_l1258_125867


namespace no_two_right_angles_l1258_125866

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : a + b + c = 180

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = 90

-- Theorem statement
theorem no_two_right_angles (t : Triangle) : 
  ¬(is_right_angle t.a ∧ is_right_angle t.b) ∧ 
  ¬(is_right_angle t.b ∧ is_right_angle t.c) ∧ 
  ¬(is_right_angle t.c ∧ is_right_angle t.a) :=
sorry

end no_two_right_angles_l1258_125866


namespace hot_pot_restaurant_problem_l1258_125803

-- Define variables for set prices
variable (price_A price_B : ℚ)

-- Define variables for daily quantities and income
variable (day1_A day1_B day2_A day2_B : ℕ)
variable (income1 income2 : ℚ)

-- Define variables for costs and constraints
variable (cost_A cost_B : ℚ)
variable (max_sets : ℕ)
variable (set_A_ratio : ℚ)

-- Define variables for extra ingredients
variable (extra_cost : ℚ)

-- Define variables for Xiaoming's spending
variable (xiaoming_total : ℚ)
variable (xiaoming_set_A_ratio : ℚ)

-- Theorem statement
theorem hot_pot_restaurant_problem 
  (h1 : day1_A * price_A + day1_B * price_B = income1)
  (h2 : day2_A * price_A + day2_B * price_B = income2)
  (h3 : day1_A = 20 ∧ day1_B = 10 ∧ income1 = 2800)
  (h4 : day2_A = 15 ∧ day2_B = 20 ∧ income2 = 3350)
  (h5 : cost_A = 45 ∧ cost_B = 50)
  (h6 : max_sets = 50)
  (h7 : set_A_ratio = 1/5)
  (h8 : extra_cost = 10)
  (h9 : xiaoming_total = 1610)
  (h10 : xiaoming_set_A_ratio = 1/4) :
  price_A = 90 ∧ 
  price_B = 100 ∧ 
  (∃ (m : ℕ), m ≥ max_sets * set_A_ratio ∧ 
              m ≤ max_sets ∧ 
              (price_A - cost_A) * m + (price_B - cost_B) * (max_sets - m) = 2455) ∧
  (∃ (x y : ℕ), x = xiaoming_set_A_ratio * (x + y) ∧
                90 * x + 100 * y + 110 * (3 * x - y) = xiaoming_total ∧
                3 * x - y = 5) := by
  sorry

end hot_pot_restaurant_problem_l1258_125803


namespace combined_mean_of_two_sets_l1258_125810

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 16 →
  set2_count = 9 →
  set2_mean = 20 →
  let total_count := set1_count + set2_count
  let combined_sum := set1_count * set1_mean + set2_count * set2_mean
  combined_sum / total_count = 18.25 := by
  sorry

end combined_mean_of_two_sets_l1258_125810


namespace regression_result_l1258_125877

/-- The regression equation -/
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 2.57

/-- Theorem: For the given regression equation, when x = 28, y = 135.57 -/
theorem regression_result : regression_equation 28 = 135.57 := by
  sorry

end regression_result_l1258_125877


namespace expression_percentage_of_y_l1258_125819

theorem expression_percentage_of_y (y : ℝ) (z : ℂ) (h : y > 0) :
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10) / y = 0.6 := by
  sorry

end expression_percentage_of_y_l1258_125819


namespace volunteer_members_count_l1258_125827

/-- The number of sheets of cookies baked by each member -/
def sheets_per_member : ℕ := 10

/-- The number of cookies on each sheet -/
def cookies_per_sheet : ℕ := 16

/-- The total number of cookies baked -/
def total_cookies : ℕ := 16000

/-- The number of members who volunteered to bake cookies -/
def num_members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

theorem volunteer_members_count : num_members = 100 := by
  sorry

end volunteer_members_count_l1258_125827


namespace expenditure_problem_l1258_125860

theorem expenditure_problem (first_avg : ℝ) (second_avg : ℝ) (total_avg : ℝ) 
  (second_days : ℕ) (h1 : first_avg = 350) (h2 : second_avg = 420) 
  (h3 : total_avg = 390) (h4 : second_days = 4) : 
  ∃ (first_days : ℕ), first_days + second_days = 7 ∧ 
  (first_avg * first_days + second_avg * second_days) / (first_days + second_days) = total_avg :=
by
  sorry

#check expenditure_problem

end expenditure_problem_l1258_125860
