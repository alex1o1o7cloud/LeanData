import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_of_products_l4046_404648

/-- A permutation of the numbers 1 to 12 -/
def Permutation12 := Fin 12 → Fin 12

/-- The sum of products for a given permutation -/
def sumOfProducts (p : Permutation12) : ℕ :=
  (p 0 + 1) * (p 1 + 1) * (p 2 + 1) +
  (p 3 + 1) * (p 4 + 1) * (p 5 + 1) +
  (p 6 + 1) * (p 7 + 1) * (p 8 + 1) +
  (p 9 + 1) * (p 10 + 1) * (p 11 + 1)

theorem min_sum_of_products :
  (∀ p : Permutation12, Function.Bijective p → sumOfProducts p ≥ 646) ∧
  (∃ p : Permutation12, Function.Bijective p ∧ sumOfProducts p = 646) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_products_l4046_404648


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4046_404683

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_1 + a_3 + a_5 = 21, 
    prove that a_3 + a_5 + a_7 = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term condition
  a 1 + a 3 + a 5 = 21 →        -- sum of odd terms condition
  a 3 + a 5 + a 7 = 42 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4046_404683


namespace NUMINAMATH_CALUDE_angle_2013_in_third_quadrant_l4046_404657

-- Define the quadrants
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to determine the quadrant of an angle
def angleQuadrant (angle : ℝ) : Quadrant := sorry

-- Theorem stating that 2013° is in the third quadrant
theorem angle_2013_in_third_quadrant :
  angleQuadrant 2013 = Quadrant.Third :=
by
  -- Define the relationship between 2013° and 213°
  have h1 : 2013 = 5 * 360 + 213 := by sorry
  
  -- State that 213° is in the third quadrant
  have h2 : angleQuadrant 213 = Quadrant.Third := by sorry
  
  -- State that angles with the same terminal side are in the same quadrant
  have h3 : ∀ (a b : ℝ), (a - b) % 360 = 0 → angleQuadrant a = angleQuadrant b := by sorry
  
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_angle_2013_in_third_quadrant_l4046_404657


namespace NUMINAMATH_CALUDE_min_value_theorem_l4046_404631

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4046_404631


namespace NUMINAMATH_CALUDE_rectangle_area_and_diagonal_l4046_404679

/-- Represents a rectangle with length, width, perimeter, area, and diagonal --/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter : ℝ
  area : ℝ
  diagonal : ℝ

/-- Theorem about the area and diagonal of a specific rectangle --/
theorem rectangle_area_and_diagonal (r : Rectangle) 
  (h1 : r.length = 4 * r.width)
  (h2 : r.perimeter = 200) :
  r.area = 1600 ∧ r.diagonal = Real.sqrt 6800 := by
  sorry

#check rectangle_area_and_diagonal

end NUMINAMATH_CALUDE_rectangle_area_and_diagonal_l4046_404679


namespace NUMINAMATH_CALUDE_spring_deformation_l4046_404676

/-- A uniform spring with two attached weights -/
structure Spring :=
  (k : ℝ)  -- Spring constant
  (m₁ : ℝ) -- Mass of the top weight
  (m₂ : ℝ) -- Mass of the bottom weight

/-- The gravitational acceleration constant -/
def g : ℝ := 9.81

/-- Deformation when the spring is held vertically at its midpoint -/
def vertical_deformation (s : Spring) (x₁ x₂ : ℝ) : Prop :=
  2 * s.k * x₁ = s.m₁ * g ∧ x₁ = 0.08 ∧ x₂ = 0.15

/-- Deformation when the spring is laid horizontally -/
def horizontal_deformation (s : Spring) (x : ℝ) : Prop :=
  s.k * x = s.m₁ * g

/-- Theorem stating the relationship between vertical and horizontal deformations -/
theorem spring_deformation (s : Spring) (x₁ x₂ x : ℝ) :
  vertical_deformation s x₁ x₂ → horizontal_deformation s x → x = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_spring_deformation_l4046_404676


namespace NUMINAMATH_CALUDE_diminished_value_proof_l4046_404636

theorem diminished_value_proof : 
  let numbers := [12, 16, 18, 21, 28]
  let smallest_number := 1015
  let diminished_value := 7
  (∀ n ∈ numbers, (smallest_number - diminished_value) % n = 0) ∧
  (∀ m < smallest_number, ∃ n ∈ numbers, ∀ k : ℕ, m - k ≠ 0 ∨ (m - k) % n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_diminished_value_proof_l4046_404636


namespace NUMINAMATH_CALUDE_system_solution_l4046_404602

theorem system_solution : ∃ (x y : ℝ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 :=
by
  use 4, -1
  sorry

end NUMINAMATH_CALUDE_system_solution_l4046_404602


namespace NUMINAMATH_CALUDE_probability_threshold_min_probability_value_l4046_404630

/-- The probability that Alex and Dylan are on the same team given their card picks -/
def probability_same_team (a : ℕ) : ℚ :=
  let total_outcomes := (50 : ℚ) * 49 / 2
  let favorable_outcomes := ((a - 1 : ℚ) * (a - 2) / 2) + ((43 - a : ℚ) * (42 - a) / 2)
  favorable_outcomes / total_outcomes

/-- The minimum value of a for which the probability is at least 1/2 -/
def min_a : ℕ := 8

theorem probability_threshold :
  probability_same_team min_a ≥ 1/2 ∧
  ∀ a < min_a, probability_same_team a < 1/2 :=
sorry

theorem min_probability_value :
  probability_same_team min_a = 88/175 :=
sorry

end NUMINAMATH_CALUDE_probability_threshold_min_probability_value_l4046_404630


namespace NUMINAMATH_CALUDE_library_bookshelf_selection_l4046_404634

/-- Represents a bookshelf with three tiers -/
structure Bookshelf :=
  (tier1 : ℕ)
  (tier2 : ℕ)
  (tier3 : ℕ)

/-- The number of ways to select a book from a bookshelf -/
def selectBook (b : Bookshelf) : ℕ := b.tier1 + b.tier2 + b.tier3

/-- Theorem: The number of ways to select a book from the given bookshelf is 16 -/
theorem library_bookshelf_selection :
  ∃ (b : Bookshelf), b.tier1 = 3 ∧ b.tier2 = 5 ∧ b.tier3 = 8 ∧ selectBook b = 16 := by
  sorry


end NUMINAMATH_CALUDE_library_bookshelf_selection_l4046_404634


namespace NUMINAMATH_CALUDE_n_value_for_specific_x_y_l4046_404682

theorem n_value_for_specific_x_y : ∀ (x y n : ℝ), 
  x = 3 → y = 1 → n = x - y^(x-y) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_n_value_for_specific_x_y_l4046_404682


namespace NUMINAMATH_CALUDE_tetrahedrons_count_l4046_404619

/-- The number of tetrahedrons formed by choosing 4 vertices from a triangular prism -/
def tetrahedrons_from_prism : ℕ :=
  Nat.choose 6 4 - 3

/-- Theorem stating that the number of tetrahedrons is 12 -/
theorem tetrahedrons_count : tetrahedrons_from_prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedrons_count_l4046_404619


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l4046_404605

/-- Given that x and y are inversely proportional, prove that y = -27 when x = -9,
    given that x = 3y when x + y = 36. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
    (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 36 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) : 
    x = -9 → y = -27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l4046_404605


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_smallest_a_for_inequality_l4046_404615

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3| + |x + 2|

-- Theorem 1: The solution set of f(x) ≤ 5 is [0, 2]
theorem solution_set_f_leq_5 : 
  {x : ℝ | f x ≤ 5} = Set.Icc 0 2 := by sorry

-- Theorem 2: The smallest value of a such that f(x) ≤ a - |x| for all x in [-1, 2] is 7
theorem smallest_a_for_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → f x ≤ a - |x|) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → f x ≤ a - |x|) → a ≥ 7) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_smallest_a_for_inequality_l4046_404615


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l4046_404607

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 1 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l4046_404607


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l4046_404638

theorem max_area_right_triangle (c : ℝ) (h : c = 8) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + y^2 = c^2 →
  (1/2) * x * y ≤ (1/2) * a * b ∧
  (1/2) * a * b = 16 := by
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l4046_404638


namespace NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l4046_404695

theorem no_real_roots_of_quadratic : 
  ¬∃ (x : ℝ), x^2 - 4*x + 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_of_quadratic_l4046_404695


namespace NUMINAMATH_CALUDE_computer_price_increase_l4046_404692

theorem computer_price_increase (d : ℝ) (h1 : d * 1.3 = 338) (h2 : ∃ x : ℝ, x * d = 520) : 
  ∃ x : ℝ, x * d = 520 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l4046_404692


namespace NUMINAMATH_CALUDE_sum_of_roots_l4046_404680

theorem sum_of_roots (m n : ℝ) : 
  m^2 - 3*m - 2 = 0 → n^2 - 3*n - 2 = 0 → m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4046_404680


namespace NUMINAMATH_CALUDE_distance_AB_l4046_404675

def A : ℝ × ℝ := (-3, 2)
def B : ℝ × ℝ := (1, -1)

theorem distance_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_l4046_404675


namespace NUMINAMATH_CALUDE_range_of_a_l4046_404674

theorem range_of_a (a : ℝ) : 
  (∅ : Set ℝ) ⊂ {x : ℝ | x^2 ≤ a} → a ∈ Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4046_404674


namespace NUMINAMATH_CALUDE_number_difference_l4046_404678

theorem number_difference (A B : ℕ) (h1 : A + B = 1812) (h2 : A = 7 * B + 4) : A - B = 1360 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l4046_404678


namespace NUMINAMATH_CALUDE_complement_determines_set_l4046_404660

def U : Set Nat := {1, 2, 3, 4}

theorem complement_determines_set (B : Set Nat) (h : Set.compl B = {2, 3}) : B = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_determines_set_l4046_404660


namespace NUMINAMATH_CALUDE_max_value_of_m_over_n_l4046_404687

theorem max_value_of_m_over_n (n : ℝ) (m : ℝ) (h_n : n > 0) :
  (∀ x > 0, Real.log x + 1 ≥ m - n / x) →
  m / n ≤ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_over_n_l4046_404687


namespace NUMINAMATH_CALUDE_remainder_problem_l4046_404632

theorem remainder_problem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 38) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l4046_404632


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l4046_404665

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/3) 10 (5*π/6)
  r = 5 * Real.sqrt 2 ∧ θ = 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l4046_404665


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l4046_404609

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A function to check if a number is five-digit -/
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function to check if a number is divisible by all numbers in a list -/
def divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ m ∈ list, n % m = 0

theorem smallest_five_digit_divisible_by_smallest_primes :
  (is_five_digit 11550) ∧ 
  (divisible_by_all 11550 smallest_primes) ∧ 
  (∀ n : Nat, is_five_digit n ∧ divisible_by_all n smallest_primes → 11550 ≤ n) := by
  sorry

#check smallest_five_digit_divisible_by_smallest_primes

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l4046_404609


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l4046_404658

theorem real_part_of_complex_product : Complex.re ((1 + 3 * Complex.I) * Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l4046_404658


namespace NUMINAMATH_CALUDE_factors_of_135_l4046_404629

theorem factors_of_135 : Nat.card (Nat.divisors 135) = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_135_l4046_404629


namespace NUMINAMATH_CALUDE_kates_wand_cost_l4046_404621

/-- Proves that the original cost of each wand is $60 given the conditions of Kate's wand purchase and sale. -/
theorem kates_wand_cost (total_wands : ℕ) (kept_wands : ℕ) (sold_wands : ℕ) 
  (price_increase : ℕ) (total_collected : ℕ) : ℕ :=
  by
  have h1 : total_wands = 3 := by sorry
  have h2 : kept_wands = 1 := by sorry
  have h3 : sold_wands = 2 := by sorry
  have h4 : price_increase = 5 := by sorry
  have h5 : total_collected = 130 := by sorry
  
  have h6 : sold_wands = total_wands - kept_wands := by sorry
  
  have h7 : total_collected / sold_wands - price_increase = 60 := by sorry
  
  exact 60

end NUMINAMATH_CALUDE_kates_wand_cost_l4046_404621


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l4046_404669

theorem smallest_three_digit_congruence :
  ∃ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧ 
    (60 * n ≡ 180 [MOD 300]) ∧ 
    (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (60 * m ≡ 180 [MOD 300]) → n ≤ m) ∧
    n = 103 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l4046_404669


namespace NUMINAMATH_CALUDE_papa_carlo_solution_l4046_404606

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60

/-- Represents a clock with its displayed time and offset -/
structure Clock where
  displayed_time : Time
  offset : Int

def Papa_Carlo_problem (clocks : Vector Clock 4) : Prop :=
  ∃ (correct_time : Time),
    (clocks.get 0).offset = -2 ∧
    (clocks.get 1).offset = -3 ∧
    (clocks.get 2).offset = 4 ∧
    (clocks.get 3).offset = 5 ∧
    (clocks.get 0).displayed_time = Time.mk 14 54 (by norm_num) ∧
    (clocks.get 1).displayed_time = Time.mk 14 57 (by norm_num) ∧
    (clocks.get 2).displayed_time = Time.mk 15 2 (by norm_num) ∧
    (clocks.get 3).displayed_time = Time.mk 15 3 (by norm_num) ∧
    correct_time = Time.mk 14 59 (by norm_num)

theorem papa_carlo_solution (clocks : Vector Clock 4) 
  (h : Papa_Carlo_problem clocks) : 
  ∃ (correct_time : Time), correct_time = Time.mk 14 59 (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_papa_carlo_solution_l4046_404606


namespace NUMINAMATH_CALUDE_merchant_bought_15_keyboards_l4046_404601

/-- The number of keyboards bought by a merchant -/
def num_keyboards : ℕ := 15

/-- The number of printers bought by the merchant -/
def num_printers : ℕ := 25

/-- The cost of one keyboard in dollars -/
def cost_keyboard : ℕ := 20

/-- The cost of one printer in dollars -/
def cost_printer : ℕ := 70

/-- The total cost of all items bought by the merchant in dollars -/
def total_cost : ℕ := 2050

/-- Theorem stating that the number of keyboards bought is 15 -/
theorem merchant_bought_15_keyboards :
  num_keyboards * cost_keyboard + num_printers * cost_printer = total_cost :=
sorry

end NUMINAMATH_CALUDE_merchant_bought_15_keyboards_l4046_404601


namespace NUMINAMATH_CALUDE_battery_factory_robots_l4046_404654

/-- The number of robots working simultaneously in a battery factory -/
def num_robots : ℕ :=
  let time_per_battery : ℕ := 15  -- 6 minutes for materials + 9 minutes for creation
  let total_time : ℕ := 300       -- 5 hours * 60 minutes
  let total_batteries : ℕ := 200
  total_batteries * time_per_battery / total_time

theorem battery_factory_robots :
  num_robots = 10 :=
sorry

end NUMINAMATH_CALUDE_battery_factory_robots_l4046_404654


namespace NUMINAMATH_CALUDE_problem_statement_l4046_404656

theorem problem_statement (x : ℝ) (h : x * (x + 3) = 154) : (x + 1) * (x + 2) = 156 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4046_404656


namespace NUMINAMATH_CALUDE_pen_profit_calculation_l4046_404689

theorem pen_profit_calculation (total_pens : ℕ) (buy_price sell_price : ℚ) (target_profit : ℚ) :
  total_pens = 2000 →
  buy_price = 15/100 →
  sell_price = 30/100 →
  target_profit = 120 →
  ∃ (sold_pens : ℕ), 
    sold_pens ≤ total_pens ∧ 
    (↑sold_pens * sell_price) - (↑total_pens * buy_price) = target_profit ∧
    sold_pens = 1400 :=
by sorry

end NUMINAMATH_CALUDE_pen_profit_calculation_l4046_404689


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l4046_404681

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 48) : 
  ∃ (x y : ℕ), x * y = 48 ∧ x + y ≤ a + b ∧ x + y = 49 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l4046_404681


namespace NUMINAMATH_CALUDE_lines_intersection_l4046_404652

-- Define the two lines
def line1 (s : ℚ) : ℚ × ℚ := (1 + 2*s, 4 - 3*s)
def line2 (v : ℚ) : ℚ × ℚ := (-2 + 3*v, 6 - v)

-- Define the intersection point
def intersection_point : ℚ × ℚ := (17/11, 35/11)

-- Theorem statement
theorem lines_intersection :
  ∃ (s v : ℚ), line1 s = line2 v ∧ line1 s = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_lines_intersection_l4046_404652


namespace NUMINAMATH_CALUDE_second_player_wins_l4046_404627

/-- Represents the possible moves in the game -/
inductive Move where
  | two : Move
  | four : Move
  | five : Move

/-- Defines the game state -/
structure GameState where
  chips : Nat
  player_turn : Bool  -- True for first player, False for second player

/-- Determines if a position is winning for the current player -/
def is_winning_position (state : GameState) : Bool :=
  match state.chips % 7 with
  | 0 | 1 | 3 => false
  | _ => true

/-- Theorem stating that the second player has a winning strategy when starting with 2016 chips -/
theorem second_player_wins :
  let initial_state : GameState := { chips := 2016, player_turn := true }
  ¬(is_winning_position initial_state) := by
  sorry

end NUMINAMATH_CALUDE_second_player_wins_l4046_404627


namespace NUMINAMATH_CALUDE_truth_values_equivalence_l4046_404622

theorem truth_values_equivalence (p q : Prop) 
  (h1 : p ∨ q) (h2 : ¬(p ∧ q)) : p ↔ ¬q := by
  sorry

end NUMINAMATH_CALUDE_truth_values_equivalence_l4046_404622


namespace NUMINAMATH_CALUDE_lucky_number_2015_l4046_404611

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- A function that returns true if a positive integer is a "lucky number" (sum of digits is 8) -/
def isLuckyNumber (n : ℕ+) : Prop := sumOfDigits n = 8

/-- A function that returns the nth "lucky number" -/
def nthLuckyNumber (n : ℕ+) : ℕ+ := sorry

theorem lucky_number_2015 : nthLuckyNumber 106 = 2015 := by sorry

end NUMINAMATH_CALUDE_lucky_number_2015_l4046_404611


namespace NUMINAMATH_CALUDE_max_value_theorem_l4046_404628

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 ∧ ∃ (a' b' c' : ℝ), a' + b'^3 + c'^4 = 2 ∧ 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4046_404628


namespace NUMINAMATH_CALUDE_right_triangle_angle_measure_l4046_404696

theorem right_triangle_angle_measure (A B C : ℝ) : 
  A = 90 →  -- A is the right angle (90 degrees)
  C = 3 * B →  -- C is three times B
  A + B + C = 180 →  -- Sum of angles in a triangle
  B = 22.5 :=  -- B is 22.5 degrees
by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_measure_l4046_404696


namespace NUMINAMATH_CALUDE_total_toothpicks_needed_l4046_404667

/-- The number of small triangles in the base row of the large equilateral triangle. -/
def base_triangles : ℕ := 2004

/-- The total number of small triangles in the large equilateral triangle. -/
def total_triangles : ℕ := base_triangles * (base_triangles + 1) / 2

/-- The number of toothpicks needed if each side of each small triangle was unique. -/
def total_sides : ℕ := 3 * total_triangles

/-- The number of toothpicks on the boundary of the large triangle. -/
def boundary_toothpicks : ℕ := 3 * base_triangles

/-- Theorem: The total number of toothpicks needed to construct the large equilateral triangle. -/
theorem total_toothpicks_needed : 
  (total_sides / 2) + boundary_toothpicks = 3021042 := by
  sorry

end NUMINAMATH_CALUDE_total_toothpicks_needed_l4046_404667


namespace NUMINAMATH_CALUDE_first_day_rainfall_is_26_l4046_404624

/-- Rainfall data for May -/
structure RainfallData where
  day2 : ℝ
  day3_diff : ℝ
  normal_average : ℝ
  less_than_average : ℝ

/-- Calculate the rainfall on the first day -/
def calculate_first_day_rainfall (data : RainfallData) : ℝ :=
  3 * data.normal_average - data.less_than_average - data.day2 - (data.day2 - data.day3_diff)

/-- Theorem stating that the rainfall on the first day is 26 cm -/
theorem first_day_rainfall_is_26 (data : RainfallData)
  (h1 : data.day2 = 34)
  (h2 : data.day3_diff = 12)
  (h3 : data.normal_average = 140)
  (h4 : data.less_than_average = 58) :
  calculate_first_day_rainfall data = 26 := by
  sorry

#eval calculate_first_day_rainfall ⟨34, 12, 140, 58⟩

end NUMINAMATH_CALUDE_first_day_rainfall_is_26_l4046_404624


namespace NUMINAMATH_CALUDE_pepperoni_coverage_l4046_404662

theorem pepperoni_coverage (pizza_diameter : ℝ) (pepperoni_count : ℕ) (pepperoni_across : ℕ) :
  pizza_diameter = 18 →
  pepperoni_count = 36 →
  pepperoni_across = 9 →
  (pepperoni_count * (pizza_diameter / pepperoni_across / 2)^2 * Real.pi) / (pizza_diameter / 2)^2 / Real.pi = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_l4046_404662


namespace NUMINAMATH_CALUDE_janet_action_figures_l4046_404644

theorem janet_action_figures (x : ℕ) : 
  (x - 2 : ℤ) + 2 * (x - 2 : ℤ) = 24 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l4046_404644


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4046_404633

theorem inequality_equivalence (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x ↔ x < -3 ∨ x > 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4046_404633


namespace NUMINAMATH_CALUDE_obtuse_triangle_k_range_l4046_404661

/-- An obtuse triangle ABC with sides a = k, b = k + 2, and c = k + 4 -/
structure ObtuseTriangle (k : ℝ) where
  a : ℝ := k
  b : ℝ := k + 2
  c : ℝ := k + 4
  is_obtuse : c^2 > a^2 + b^2

/-- The range of possible values for k in an obtuse triangle with sides k, k+2, k+4 -/
theorem obtuse_triangle_k_range (k : ℝ) :
  (∃ t : ObtuseTriangle k, True) ↔ 2 < k ∧ k < 6 := by
  sorry

#check obtuse_triangle_k_range

end NUMINAMATH_CALUDE_obtuse_triangle_k_range_l4046_404661


namespace NUMINAMATH_CALUDE_four_digit_repeat_count_l4046_404612

theorem four_digit_repeat_count : ∀ n : ℕ, (20 ≤ n ∧ n ≤ 99) → (Finset.range 100 \ Finset.range 20).card = 80 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_repeat_count_l4046_404612


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l4046_404620

/-- Given an arithmetic sequence with first term 3² and last term 3⁴, 
    the middle term y is equal to 45. -/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (a 0 = 3^2) → 
    (a 2 = 3^4) → 
    (∀ n : ℕ, n < 2 → a (n + 1) - a n = a 1 - a 0) → 
    a 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l4046_404620


namespace NUMINAMATH_CALUDE_keith_purchases_cost_l4046_404691

/-- The total cost of Keith's purchases -/
def total_cost (rabbit_toy pet_food cage water_bottle bedding found_money : ℝ)
  (rabbit_discount cage_tax : ℝ) : ℝ :=
  let rabbit_toy_original := rabbit_toy / (1 - rabbit_discount)
  let cage_with_tax := cage * (1 + cage_tax)
  rabbit_toy + pet_food + cage_with_tax + water_bottle + bedding - found_money

/-- Theorem stating the total cost of Keith's purchases -/
theorem keith_purchases_cost :
  total_cost 6.51 5.79 12.51 4.99 7.65 1 0.1 0.08 = 37.454 := by
  sorry

end NUMINAMATH_CALUDE_keith_purchases_cost_l4046_404691


namespace NUMINAMATH_CALUDE_fourth_degree_reduction_l4046_404643

theorem fourth_degree_reduction (a b c d : ℝ) :
  ∃ (A B C k : ℝ), ∀ (t x : ℝ),
    (t^4 + a*t^3 + b*t^2 + c*t + d = 0) ↔
    (t = x + k ∧ x^4 = A*x^2 + B*x + C) :=
sorry

end NUMINAMATH_CALUDE_fourth_degree_reduction_l4046_404643


namespace NUMINAMATH_CALUDE_common_external_tangents_exist_l4046_404684

/-- Two circles with centers O₁ and O₂ and radii r₁ and r₂ respectively -/
structure TwoCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  r₁ : ℝ
  r₂ : ℝ
  h₁ : r₁ > r₂
  h₂ : r₁ > 0
  h₃ : r₂ > 0
  h₄ : dist O₁ O₂ > r₁ + r₂  -- circles lie outside each other

/-- A line in 2D space represented by its normal vector and a point on the line -/
structure Line where
  normal : ℝ × ℝ
  point : ℝ × ℝ

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent_to (l : Line) (c : TwoCircles) (i : Fin 2) : Prop :=
  let center := if i = 0 then c.O₁ else c.O₂
  let radius := if i = 0 then c.r₁ else c.r₂
  dist l.point center = radius ∧ 
  (l.normal.1 * (center.1 - l.point.1) + l.normal.2 * (center.2 - l.point.2) = 0)

/-- Theorem: There exist two common external tangent lines for two circles lying outside each other -/
theorem common_external_tangents_exist (c : TwoCircles) : 
  ∃ l₁ l₂ : Line, (is_tangent_to l₁ c 0 ∧ is_tangent_to l₁ c 1) ∧ 
                  (is_tangent_to l₂ c 0 ∧ is_tangent_to l₂ c 1) ∧
                  l₁ ≠ l₂ := by
  sorry

end NUMINAMATH_CALUDE_common_external_tangents_exist_l4046_404684


namespace NUMINAMATH_CALUDE_simplified_expression_l4046_404646

theorem simplified_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6) / ((x - 3) * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l4046_404646


namespace NUMINAMATH_CALUDE_expression_simplification_l4046_404655

theorem expression_simplification (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a ≠ -2) :
  (((a^2 + 1) / a - 2) / ((a + 2) * (a - 1) / (a^2 + 2*a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4046_404655


namespace NUMINAMATH_CALUDE_circle_area_through_DEF_l4046_404603

-- Define the triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  let d_e := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let d_f := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  d_e = d_f ∧ d_e = 5 * Real.sqrt 3

-- Define the tangent circle
def tangent_circle (D E F : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  let g_e := Real.sqrt ((E.1 - G.1)^2 + (E.2 - G.2)^2)
  let g_f := Real.sqrt ((F.1 - G.1)^2 + (F.2 - G.2)^2)
  g_e = 6 ∧ g_f = 6

-- Define the altitude condition
def altitude_condition (D E F : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  let m_ef := (F.2 - E.2) / (F.1 - E.1)
  let m_dg := (G.2 - D.2) / (G.1 - D.1)
  m_ef * m_dg = -1

-- Theorem statement
theorem circle_area_through_DEF 
  (D E F : ℝ × ℝ) 
  (G : ℝ × ℝ) 
  (h1 : triangle_DEF D E F) 
  (h2 : tangent_circle D E F G) 
  (h3 : altitude_condition D E F G) :
  let R := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / 2
  Real.pi * R^2 = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_through_DEF_l4046_404603


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l4046_404640

theorem largest_multiple_of_15_under_500 : ∃ (n : ℕ), n = 495 ∧ 
  (∀ m : ℕ, m % 15 = 0 → m < 500 → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l4046_404640


namespace NUMINAMATH_CALUDE_clock_advance_proof_l4046_404697

def clock_hours : ℕ := 12

def start_time : ℕ := 3

def hours_elapsed : ℕ := 2500

def end_time : ℕ := 7

theorem clock_advance_proof :
  (start_time + hours_elapsed) % clock_hours = end_time :=
by sorry

end NUMINAMATH_CALUDE_clock_advance_proof_l4046_404697


namespace NUMINAMATH_CALUDE_combined_friends_list_l4046_404659

theorem combined_friends_list (james_friends : ℕ) (susan_friends : ℕ) (maria_friends : ℕ)
  (james_john_shared : ℕ) (james_john_maria_shared : ℕ)
  (h1 : james_friends = 90)
  (h2 : susan_friends = 50)
  (h3 : maria_friends = 80)
  (h4 : james_john_shared = 35)
  (h5 : james_john_maria_shared = 10) :
  james_friends + 4 * susan_friends - james_john_shared + maria_friends - james_john_maria_shared = 325 := by
  sorry

end NUMINAMATH_CALUDE_combined_friends_list_l4046_404659


namespace NUMINAMATH_CALUDE_reciprocal_sum_range_l4046_404637

theorem reciprocal_sum_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 1 / y ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1 / a + 1 / b = 4 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_range_l4046_404637


namespace NUMINAMATH_CALUDE_ryan_chinese_hours_l4046_404699

/-- Ryan's daily study hours -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  more_chinese : chinese = english + 1

/-- Theorem: Ryan spends 7 hours on learning Chinese -/
theorem ryan_chinese_hours (ryan : StudyHours) (h : ryan.english = 6) : ryan.chinese = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_hours_l4046_404699


namespace NUMINAMATH_CALUDE_order_of_numbers_l4046_404613

theorem order_of_numbers : Real.log 0.76 < 0.76 ∧ 0.76 < 60.7 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l4046_404613


namespace NUMINAMATH_CALUDE_trapezoid_height_l4046_404625

/-- The height of a trapezoid with specific properties -/
theorem trapezoid_height (a b : ℝ) (h_ab : a < b) : ∃ h : ℝ,
  h = a * b / (b - a) ∧
  ∃ (AB CD : ℝ) (angle_diagonals angle_sides : ℝ),
    AB = a ∧
    CD = b ∧
    angle_diagonals = 90 ∧
    angle_sides = 45 ∧
    h > 0 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_height_l4046_404625


namespace NUMINAMATH_CALUDE_book_cost_problem_l4046_404617

/-- Given two books with a total cost of 300 Rs, if one is sold at a 15% loss
    and the other at a 19% gain, and both are sold at the same price,
    then the cost of the book sold at a loss is 175 Rs. -/
theorem book_cost_problem (C₁ C₂ SP : ℝ) : 
  C₁ + C₂ = 300 →
  SP = 0.85 * C₁ →
  SP = 1.19 * C₂ →
  C₁ = 175 := by
sorry

end NUMINAMATH_CALUDE_book_cost_problem_l4046_404617


namespace NUMINAMATH_CALUDE_contrapositive_truth_l4046_404664

/-- The function f(x) = x^2 - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- The theorem statement -/
theorem contrapositive_truth (m : ℝ) 
  (h : ∀ x > 0, f m x ≥ 0) :
  ∀ a b, a > 0 → b > 0 → 
    (a + b ≤ 1 → 1/a + 2/b ≥ 3 + Real.sqrt 2 * m) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_l4046_404664


namespace NUMINAMATH_CALUDE_no_coprime_natural_solution_l4046_404649

theorem no_coprime_natural_solution :
  ¬ ∃ (x y : ℕ), 
    (x ≠ 0) ∧ (y ≠ 0) ∧ 
    (Nat.gcd x y = 1) ∧ 
    (y^2 + y = x^3 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_coprime_natural_solution_l4046_404649


namespace NUMINAMATH_CALUDE_employee_discount_percentage_l4046_404604

theorem employee_discount_percentage
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 180) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_employee_discount_percentage_l4046_404604


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l4046_404663

/-- Given two parallel vectors a and b, prove that tan(α) = -1 -/
theorem parallel_vectors_tan_alpha (a b : ℝ × ℝ) (α : ℝ) :
  a = (Real.sqrt 2, -Real.sqrt 2) →
  b = (Real.cos α, Real.sin α) →
  (∃ (k : ℝ), a = k • b) →
  Real.tan α = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_alpha_l4046_404663


namespace NUMINAMATH_CALUDE_profit_doubling_l4046_404610

theorem profit_doubling (cost : ℝ) (price : ℝ) (h1 : price = 1.5 * cost) :
  let double_price := 2 * price
  (double_price - cost) / cost * 100 = 200 :=
by sorry

end NUMINAMATH_CALUDE_profit_doubling_l4046_404610


namespace NUMINAMATH_CALUDE_water_scooped_out_l4046_404623

theorem water_scooped_out (total_weight : ℝ) (alcohol_concentration : ℝ) :
  total_weight = 10 ∧ alcohol_concentration = 0.75 →
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 10 ∧ x / total_weight = alcohol_concentration ∧ x = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_water_scooped_out_l4046_404623


namespace NUMINAMATH_CALUDE_max_crates_on_trip_l4046_404693

theorem max_crates_on_trip (crate_weight : ℝ) (max_weight : ℝ) (h1 : crate_weight ≥ 1250) (h2 : max_weight = 6250) :
  ⌊max_weight / crate_weight⌋ = 5 :=
sorry

end NUMINAMATH_CALUDE_max_crates_on_trip_l4046_404693


namespace NUMINAMATH_CALUDE_new_person_weight_l4046_404647

/-- Given a group of 7 people, if replacing one person weighing 95 kg with a new person
    increases the average weight by 12.3 kg, then the weight of the new person is 181.1 kg. -/
theorem new_person_weight (group_size : ℕ) (weight_increase : ℝ) (old_weight : ℝ) :
  group_size = 7 →
  weight_increase = 12.3 →
  old_weight = 95 →
  (group_size : ℝ) * weight_increase + old_weight = 181.1 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l4046_404647


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_range_l4046_404650

/-- Given f(x) = |x+a| + |x-1/a| where a ≠ 0, if for all x ∈ ℝ, f(x) ≥ |m-1|, then m ∈ [-1, 3] -/
theorem function_inequality_implies_m_range (a m : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, |x + a| + |x - 1/a| ≥ |m - 1|) → m ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_range_l4046_404650


namespace NUMINAMATH_CALUDE_gardner_brownies_l4046_404616

theorem gardner_brownies :
  ∀ (students cookies cupcakes brownies total_treats : ℕ),
    students = 20 →
    cookies = 20 →
    cupcakes = 25 →
    total_treats = students * 4 →
    total_treats = cookies + cupcakes + brownies →
    brownies = 35 := by
  sorry

end NUMINAMATH_CALUDE_gardner_brownies_l4046_404616


namespace NUMINAMATH_CALUDE_solution_proof_l4046_404672

/-- Custom operation for 2x2 matrices -/
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating the solution to the given equation -/
theorem solution_proof :
  ∃ x : ℝ, matrix_op (x + 1) (x + 2) (x - 3) (x - 1) = 27 ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_solution_proof_l4046_404672


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4046_404671

theorem simplify_and_evaluate (a b : ℝ) : 
  a = Real.tan (π / 3) → 
  b = Real.sin (π / 3) → 
  ((b^2 + a^2) / a - 2 * b) / (1 - b / a) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4046_404671


namespace NUMINAMATH_CALUDE_cafeteria_seats_unseated_fraction_l4046_404639

theorem cafeteria_seats_unseated_fraction :
  let total_tables : ℕ := 15
  let seats_per_table : ℕ := 10
  let seats_taken : ℕ := 135
  let total_seats : ℕ := total_tables * seats_per_table
  let seats_unseated : ℕ := total_seats - seats_taken
  (seats_unseated : ℚ) / total_seats = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_seats_unseated_fraction_l4046_404639


namespace NUMINAMATH_CALUDE_a_10_value_l4046_404694

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_10_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 = 4 →
  a 6 = 6 →
  a 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_a_10_value_l4046_404694


namespace NUMINAMATH_CALUDE_son_work_time_l4046_404670

theorem son_work_time (man_time son_father_time : ℝ) 
  (h1 : man_time = 5)
  (h2 : son_father_time = 3) : 
  let man_rate := 1 / man_time
  let combined_rate := 1 / son_father_time
  let son_rate := combined_rate - man_rate
  1 / son_rate = 7.5 := by sorry

end NUMINAMATH_CALUDE_son_work_time_l4046_404670


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l4046_404686

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := x > 2

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  (∃ x : ℝ, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l4046_404686


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l4046_404653

theorem contrapositive_equivalence (f : ℝ → ℝ) (a : ℝ) :
  (a ≥ (1/2) → ∀ x ≥ 0, f x ≥ 0) ↔
  (∃ x ≥ 0, f x < 0 → a < (1/2)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l4046_404653


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4046_404600

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4046_404600


namespace NUMINAMATH_CALUDE_amount_owed_l4046_404673

-- Define the rate per room
def rate_per_room : ℚ := 11 / 2

-- Define the number of rooms cleaned
def rooms_cleaned : ℚ := 7 / 3

-- Theorem statement
theorem amount_owed : rate_per_room * rooms_cleaned = 77 / 6 := by
  sorry

end NUMINAMATH_CALUDE_amount_owed_l4046_404673


namespace NUMINAMATH_CALUDE_trigonometric_identities_l4046_404690

theorem trigonometric_identities :
  (((Real.tan (10 * π / 180)) * (Real.tan (70 * π / 180))) /
   ((Real.tan (70 * π / 180)) - (Real.tan (10 * π / 180)) + (Real.tan (120 * π / 180))) = Real.sqrt 3 / 3) ∧
  ((2 * (Real.cos (40 * π / 180)) + (Real.cos (10 * π / 180)) * (1 + Real.sqrt 3 * (Real.tan (10 * π / 180)))) /
   (Real.sqrt (1 + Real.cos (10 * π / 180))) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l4046_404690


namespace NUMINAMATH_CALUDE_largest_class_size_l4046_404698

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (class_diff : ℕ) :
  total_students = 95 →
  num_classes = 5 →
  class_diff = 2 →
  ∃ (x : ℕ), x = 23 ∧ 
    (x + (x - class_diff) + (x - 2*class_diff) + (x - 3*class_diff) + (x - 4*class_diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l4046_404698


namespace NUMINAMATH_CALUDE_no_quadratic_term_in_polynomial_difference_l4046_404614

theorem no_quadratic_term_in_polynomial_difference (x : ℝ) :
  let p₁ := 2 * x^3 - 8 * x^2 + x - 1
  let p₂ := 3 * x^3 + 2 * m * x^2 - 5 * x + 3
  (∃ m : ℝ, ∀ a b c d : ℝ, p₁ - p₂ = a * x^3 + c * x + d) → m = -4 :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_term_in_polynomial_difference_l4046_404614


namespace NUMINAMATH_CALUDE_andy_problem_solving_l4046_404642

theorem andy_problem_solving (last_problem : ℕ) (total_solved : ℕ) (h1 : last_problem = 125) (h2 : total_solved = 51) : 
  last_problem - total_solved + 1 = 75 := by
sorry

end NUMINAMATH_CALUDE_andy_problem_solving_l4046_404642


namespace NUMINAMATH_CALUDE_reading_book_cost_is_12_l4046_404618

/-- The cost of a reading book given the total amount available and number of students -/
def reading_book_cost (total_amount : ℕ) (num_students : ℕ) : ℚ :=
  (total_amount : ℚ) / (num_students : ℚ)

/-- Theorem: The cost of each reading book is $12 -/
theorem reading_book_cost_is_12 :
  reading_book_cost 360 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_reading_book_cost_is_12_l4046_404618


namespace NUMINAMATH_CALUDE_sum_and_equality_problem_l4046_404635

theorem sum_and_equality_problem (x y z : ℚ) : 
  x + y + z = 150 ∧ x + 10 = y - 10 ∧ y - 10 = 6 * z → y = 1030 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_equality_problem_l4046_404635


namespace NUMINAMATH_CALUDE_teresas_age_at_birth_l4046_404608

/-- Given the current ages of Teresa and Morio, and Morio's age when their daughter Michiko was born,
    prove that Teresa's age when Michiko was born is 26. -/
theorem teresas_age_at_birth (teresa_current_age morio_current_age morio_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59)
  (h2 : morio_current_age = 71)
  (h3 : morio_age_at_birth = 38) :
  teresa_current_age - (morio_current_age - morio_age_at_birth) = 26 := by
  sorry

end NUMINAMATH_CALUDE_teresas_age_at_birth_l4046_404608


namespace NUMINAMATH_CALUDE_f_monotone_increasing_f_extreme_values_l4046_404685

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Theorem for monotonically increasing intervals
theorem f_monotone_increasing :
  (∀ x y, x < y ∧ x < -2/3 → f x < f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) :=
sorry

-- Theorem for extreme values on [-1, 3]
theorem f_extreme_values :
  (∀ x, -1 ≤ x ∧ x ≤ 3 → -6 ≤ f x ∧ f x ≤ 94/27) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x = -6) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x = 94/27) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_f_extreme_values_l4046_404685


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_30_l4046_404666

def consecutive_sum (start : ℕ) (count : ℕ) : ℕ :=
  count * start + count * (count - 1) / 2

theorem largest_consecutive_sum_30 :
  (∃ (n : ℕ), n > 0 ∧ ∃ (start : ℕ), start > 0 ∧ consecutive_sum start n = 30) ∧
  (∀ (m : ℕ), m > 5 → ¬∃ (start : ℕ), start > 0 ∧ consecutive_sum start m = 30) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_30_l4046_404666


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4046_404645

theorem trigonometric_identity (x y : ℝ) : 
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4046_404645


namespace NUMINAMATH_CALUDE_specific_hill_ground_depth_l4046_404641

/-- Represents a cone-shaped hill -/
structure ConeHill where
  height : ℝ
  aboveGroundVolumeFraction : ℝ

/-- Calculates the depth of the ground at the base of a cone-shaped hill -/
def groundDepth (hill : ConeHill) : ℝ :=
  hill.height * (1 - (hill.aboveGroundVolumeFraction ^ (1/3)))

/-- Theorem stating that for a specific cone-shaped hill, the ground depth is 355 feet -/
theorem specific_hill_ground_depth :
  let hill : ConeHill := { height := 5000, aboveGroundVolumeFraction := 1/5 }
  groundDepth hill = 355 := by
  sorry

end NUMINAMATH_CALUDE_specific_hill_ground_depth_l4046_404641


namespace NUMINAMATH_CALUDE_russian_football_championship_l4046_404651

/-- Represents a football championship. -/
structure Championship where
  teams : Nat
  matches_per_pair : Nat

/-- Calculate the number of matches a single team plays in a season. -/
def matches_per_team (c : Championship) : Nat :=
  (c.teams - 1) * c.matches_per_pair

/-- Calculate the total number of matches in a season. -/
def total_matches (c : Championship) : Nat :=
  (c.teams * (c.teams - 1) * c.matches_per_pair) / 2

/-- Theorem stating the number of matches for a single team and total matches in the championship. -/
theorem russian_football_championship 
  (c : Championship) 
  (h1 : c.teams = 16) 
  (h2 : c.matches_per_pair = 2) : 
  matches_per_team c = 30 ∧ total_matches c = 240 := by
  sorry

#eval matches_per_team ⟨16, 2⟩
#eval total_matches ⟨16, 2⟩

end NUMINAMATH_CALUDE_russian_football_championship_l4046_404651


namespace NUMINAMATH_CALUDE_remaining_popsicle_sticks_l4046_404626

theorem remaining_popsicle_sticks 
  (initial : ℝ) 
  (given_to_lisa : ℝ) 
  (given_to_peter : ℝ) 
  (given_to_you : ℝ) 
  (h1 : initial = 63.5) 
  (h2 : given_to_lisa = 18.2) 
  (h3 : given_to_peter = 21.7) 
  (h4 : given_to_you = 10.1) : 
  initial - (given_to_lisa + given_to_peter + given_to_you) = 13.5 := by
sorry

end NUMINAMATH_CALUDE_remaining_popsicle_sticks_l4046_404626


namespace NUMINAMATH_CALUDE_smaller_cuboid_height_l4046_404668

/-- Given a large cuboid and smaller cuboids with specified dimensions,
    prove that the height of each smaller cuboid is 3 meters. -/
theorem smaller_cuboid_height
  (large_length large_width large_height : ℝ)
  (small_length small_width : ℝ)
  (num_small_cuboids : ℝ)
  (h_large_length : large_length = 18)
  (h_large_width : large_width = 15)
  (h_large_height : large_height = 2)
  (h_small_length : small_length = 6)
  (h_small_width : small_width = 4)
  (h_num_small_cuboids : num_small_cuboids = 7.5)
  (h_volume_conservation : large_length * large_width * large_height =
    num_small_cuboids * small_length * small_width * (large_length * large_width * large_height / (num_small_cuboids * small_length * small_width))) :
  large_length * large_width * large_height / (num_small_cuboids * small_length * small_width) = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cuboid_height_l4046_404668


namespace NUMINAMATH_CALUDE_inverse_implies_negation_l4046_404677

theorem inverse_implies_negation (p : Prop) : 
  (¬p → p) → ¬p :=
sorry

end NUMINAMATH_CALUDE_inverse_implies_negation_l4046_404677


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l4046_404688

/-- 
Given an arithmetic progression with sum of n terms equal to 220,
common difference 3, first term an integer, and n > 1,
prove that the sum of the first 10 terms is 215.
-/
theorem arithmetic_progression_sum (n : ℕ) (a : ℤ) :
  n > 1 →
  (n : ℝ) * (a + (n - 1) * 3 / 2) = 220 →
  10 * (a + (10 - 1) * 3 / 2) = 215 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l4046_404688
