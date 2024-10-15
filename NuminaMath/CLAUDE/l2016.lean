import Mathlib

namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2016_201611

def M : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 2}

theorem intersection_complement_theorem :
  M ∩ (Nᶜ) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2016_201611


namespace NUMINAMATH_CALUDE_max_coins_ali_baba_l2016_201639

/-- Represents the coin distribution game --/
structure CoinGame where
  totalPiles : Nat
  initialCoinsPerPile : Nat
  totalCoins : Nat
  selectablePiles : Nat
  takablePiles : Nat

/-- Defines the specific game instance --/
def aliBabaGame : CoinGame :=
  { totalPiles := 10
  , initialCoinsPerPile := 10
  , totalCoins := 100
  , selectablePiles := 4
  , takablePiles := 3 
  }

/-- Theorem stating the maximum number of coins Ali Baba can take --/
theorem max_coins_ali_baba (game : CoinGame) (h1 : game = aliBabaGame) : 
  ∃ (maxCoins : Nat), maxCoins = 72 ∧ 
  (∀ (strategy : CoinGame → Nat), strategy game ≤ maxCoins) := by
  sorry

end NUMINAMATH_CALUDE_max_coins_ali_baba_l2016_201639


namespace NUMINAMATH_CALUDE_number_of_arrangements_l2016_201661

/-- Represents the number of students of each gender -/
def num_students : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := 2 * num_students

/-- Represents the number of positions where male student A can stand -/
def positions_for_A : ℕ := total_students - 2

/-- Represents the number of ways to arrange the two adjacent female students -/
def adjacent_female_arrangements : ℕ := 2

/-- Represents the number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := 3 * 2

/-- The theorem stating the number of different arrangements -/
theorem number_of_arrangements :
  positions_for_A * adjacent_female_arrangements * remaining_arrangements * Nat.factorial 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l2016_201661


namespace NUMINAMATH_CALUDE_product_repeating_decimal_one_third_and_eight_l2016_201650

def repeating_decimal_one_third : ℚ := 1/3

theorem product_repeating_decimal_one_third_and_eight :
  repeating_decimal_one_third * 8 = 8/3 := by sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_one_third_and_eight_l2016_201650


namespace NUMINAMATH_CALUDE_staircase_steps_l2016_201668

/-- The number of toothpicks used in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := 2 * (n * (n + 1) * (2 * n + 1)) / 3

/-- Theorem stating that a staircase with 630 toothpicks has 9 steps -/
theorem staircase_steps : ∃ (n : ℕ), toothpicks n = 630 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_staircase_steps_l2016_201668


namespace NUMINAMATH_CALUDE_system_solution_l2016_201677

theorem system_solution (x y : ℝ) (h1 : 2*x + y = 7) (h2 : x + 2*y = 10) : 
  (x + y) / 3 = 17/9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2016_201677


namespace NUMINAMATH_CALUDE_vlads_height_in_feet_l2016_201644

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_inches_lt_12 : inches < 12

/-- Converts a height to total inches -/
def Height.to_inches (h : Height) : ℕ :=
  h.feet * 12 + h.inches

/-- The height of Vlad's sister -/
def sister_height : Height :=
  { feet := 2, inches := 10, h_inches_lt_12 := by sorry }

/-- The difference in height between Vlad and his sister in inches -/
def height_difference : ℕ := 41

/-- Theorem: Vlad's height in feet is 6 -/
theorem vlads_height_in_feet :
  (Height.to_inches sister_height + height_difference) / 12 = 6 := by sorry

end NUMINAMATH_CALUDE_vlads_height_in_feet_l2016_201644


namespace NUMINAMATH_CALUDE_equation_solutions_l2016_201654

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), (x₁ = 3/5 ∧ x₂ = -3) ∧ 
    (2*x₁ - 3)^2 = 9*x₁^2 ∧ (2*x₂ - 3)^2 = 9*x₂^2) ∧
  (∃ (y₁ y₂ : ℝ), (y₁ = 2 ∧ y₂ = -1/2) ∧ 
    2*y₁*(y₁-2) + y₁ = 2 ∧ 2*y₂*(y₂-2) + y₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2016_201654


namespace NUMINAMATH_CALUDE_jones_elementary_population_l2016_201610

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ),
    (90 : ℚ) = boys_percentage * ((20 : ℚ) / 100) * total_students →
    total_students = 450 := by
  sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l2016_201610


namespace NUMINAMATH_CALUDE_f_derivative_l2016_201647

noncomputable def f (x : ℝ) : ℝ := (1 - x) / ((1 + x^2) * Real.cos x)

theorem f_derivative :
  deriv f = λ x => ((x^2 - 2*x - 1) * Real.cos x + (1 - x) * (1 + x^2) * Real.sin x) / ((1 + x^2)^2 * (Real.cos x)^2) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l2016_201647


namespace NUMINAMATH_CALUDE_ab_multiplier_l2016_201613

theorem ab_multiplier (a b m : ℝ) : 
  4 * a = 30 ∧ 5 * b = 30 ∧ m * (a * b) = 1800 → m = 40 := by
  sorry

end NUMINAMATH_CALUDE_ab_multiplier_l2016_201613


namespace NUMINAMATH_CALUDE_multiplicative_inverse_144_mod_941_l2016_201682

theorem multiplicative_inverse_144_mod_941 : ∃ n : ℤ, 
  0 ≤ n ∧ n < 941 ∧ (144 * n) % 941 = 1 := by
  use 364
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_144_mod_941_l2016_201682


namespace NUMINAMATH_CALUDE_binary_arithmetic_theorem_l2016_201671

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ ⟨i, bi⟩ acc => acc + if bi then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

def binary_add (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a + binary_to_decimal b)

def binary_sub (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a - binary_to_decimal b)

theorem binary_arithmetic_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [false, true, false, true] -- 1010₂
  let d := [true, false, false, true] -- 1001₂
  binary_add (binary_sub (binary_add a b) c) d = [true, false, false, false, true] -- 10001₂
  := by sorry

end NUMINAMATH_CALUDE_binary_arithmetic_theorem_l2016_201671


namespace NUMINAMATH_CALUDE_hiker_rate_ratio_l2016_201603

/-- Proves that the ratio of the rate down to the rate up is 1.5 given the hiking conditions --/
theorem hiker_rate_ratio 
  (rate_up : ℝ) 
  (time_up : ℝ) 
  (distance_down : ℝ) 
  (h1 : rate_up = 3) 
  (h2 : time_up = 2) 
  (h3 : distance_down = 9) 
  (h4 : time_up = distance_down / (distance_down / time_up)) : 
  (distance_down / time_up) / rate_up = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_hiker_rate_ratio_l2016_201603


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2016_201683

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h : is_arithmetic_sequence a) (h2 : a 2 = 2) (h3 : a 3 = -4) :
  ∃ d : ℤ, d = -6 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2016_201683


namespace NUMINAMATH_CALUDE_fencing_cost_140m_perimeter_l2016_201664

/-- The cost of fencing a rectangular plot -/
def fencing_cost (width : ℝ) (rate : ℝ) : ℝ :=
  let length : ℝ := width + 10
  let perimeter : ℝ := 2 * (length + width)
  rate * perimeter

theorem fencing_cost_140m_perimeter :
  ∃ (width : ℝ),
    (2 * (width + (width + 10)) = 140) ∧
    (fencing_cost width 6.5 = 910) := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_140m_perimeter_l2016_201664


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l2016_201616

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l2016_201616


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2016_201676

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l2016_201676


namespace NUMINAMATH_CALUDE_f_is_power_and_increasing_l2016_201606

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x > 0, f x = x^a

-- Define an increasing function on (0, +∞)
def isIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- Define the function f(x) = x^(1/2)
def f (x : ℝ) : ℝ := x^(1/2)

-- Theorem statement
theorem f_is_power_and_increasing :
  isPowerFunction f ∧ isIncreasing f :=
sorry

end NUMINAMATH_CALUDE_f_is_power_and_increasing_l2016_201606


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l2016_201694

theorem sum_of_squares_theorem (a d : ℤ) : 
  ∃ (x y z w : ℤ), 
    a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (x*a + y*d)^2 + (z*a + w*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l2016_201694


namespace NUMINAMATH_CALUDE_ram_gopal_ratio_l2016_201612

theorem ram_gopal_ratio (ram_money : ℕ) (krishan_money : ℕ) (gopal_krishan_ratio : Rat) :
  ram_money = 735 →
  krishan_money = 4335 →
  gopal_krishan_ratio = 7 / 17 →
  (ram_money : Rat) / ((gopal_krishan_ratio * krishan_money) : Rat) = 7 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ram_gopal_ratio_l2016_201612


namespace NUMINAMATH_CALUDE_no_guaranteed_win_strategy_l2016_201662

/-- Represents a game state with the current number on the board -/
structure GameState where
  number : ℕ

/-- Represents a player's move, adding a digit to the number -/
inductive Move
| PrependDigit (d : ℕ) : Move
| AppendDigit (d : ℕ) : Move
| InsertDigit (d : ℕ) (pos : ℕ) : Move

/-- Apply a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Bool :=
  sorry

/-- Theorem stating that no player can guarantee a win -/
theorem no_guaranteed_win_strategy :
  ∀ (strategy : GameState → Move),
  ∃ (opponent_moves : List Move),
  let final_state := opponent_moves.foldl applyMove ⟨7⟩
  ¬ isPerfectSquare final_state.number :=
sorry

end NUMINAMATH_CALUDE_no_guaranteed_win_strategy_l2016_201662


namespace NUMINAMATH_CALUDE_probability_through_C_l2016_201691

theorem probability_through_C (total_paths : ℕ) (paths_A_to_C : ℕ) (paths_C_to_B : ℕ) :
  total_paths = Nat.choose 6 3 →
  paths_A_to_C = Nat.choose 3 2 →
  paths_C_to_B = Nat.choose 3 1 →
  (paths_A_to_C * paths_C_to_B : ℚ) / total_paths = 21 / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_through_C_l2016_201691


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2016_201674

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 9 → 1/a < 1/9) ∧ 
  (∃ a, 1/a < 1/9 ∧ ¬(a > 9)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2016_201674


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2016_201628

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the number of intersection points between three circles
def intersectionPoints (c1 c2 c3 : Circle) : ℕ := sorry

theorem circle_intersection_theorem :
  -- There exist three circles that intersect at exactly one point
  (∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 1) ∧
  -- There exist three circles that intersect at exactly two points
  (∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 2) ∧
  -- There do not exist three circles that intersect at exactly three points
  (¬ ∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 3) ∧
  -- There do not exist three circles that intersect at exactly four points
  (¬ ∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2016_201628


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2016_201695

/-- Given an ellipse with equation x²/m² + y²/n² = 1, where m > 0 and n > 0,
    whose right focus coincides with the focus of the parabola y² = 8x,
    and has an eccentricity of 1/2, prove that its standard equation is
    x²/16 + y²/12 = 1. -/
theorem ellipse_standard_equation
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_focus : m^2 - n^2 = 4)  -- Right focus coincides with parabola focus (2, 0)
  (h_eccentricity : 2 / m = 1 / 2)  -- Eccentricity is 1/2
  : ∃ (x y : ℝ), x^2/16 + y^2/12 = 1 ∧ x^2/m^2 + y^2/n^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2016_201695


namespace NUMINAMATH_CALUDE_ladybug_leaves_l2016_201634

theorem ladybug_leaves (ladybugs_per_leaf : ℕ) (total_ladybugs : ℕ) (h1 : ladybugs_per_leaf = 139) (h2 : total_ladybugs = 11676) :
  total_ladybugs / ladybugs_per_leaf = 84 := by
sorry

end NUMINAMATH_CALUDE_ladybug_leaves_l2016_201634


namespace NUMINAMATH_CALUDE_tan_half_angle_problem_l2016_201669

theorem tan_half_angle_problem (α : Real) (h : Real.tan (α / 2) = 2) :
  (Real.tan (α + Real.pi / 4) = -1 / 7) ∧
  ((6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_problem_l2016_201669


namespace NUMINAMATH_CALUDE_jimmy_matchbooks_count_l2016_201673

/-- The number of matches in one matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of stamps Tonya initially had -/
def tonya_initial_stamps : ℕ := 13

/-- The number of stamps Tonya had left after trading -/
def tonya_final_stamps : ℕ := 3

/-- The number of matchbooks Jimmy had -/
def jimmy_matchbooks : ℕ := 5

theorem jimmy_matchbooks_count :
  jimmy_matchbooks * matches_per_matchbook = 
    (tonya_initial_stamps - tonya_final_stamps) * matches_per_stamp :=
by sorry

end NUMINAMATH_CALUDE_jimmy_matchbooks_count_l2016_201673


namespace NUMINAMATH_CALUDE_neighborhood_cleanup_weight_l2016_201658

/-- The total weight of litter collected during a neighborhood clean-up. -/
def total_litter_weight (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (bag_weight : ℕ) : ℕ :=
  (gina_bags + neighborhood_multiplier * gina_bags) * bag_weight

/-- Theorem stating that the total weight of litter collected is 664 pounds. -/
theorem neighborhood_cleanup_weight :
  total_litter_weight 2 82 4 = 664 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cleanup_weight_l2016_201658


namespace NUMINAMATH_CALUDE_net_profit_is_107_70_l2016_201657

/-- Laundry shop rates and quantities for a three-day period --/
structure LaundryData where
  regular_rate : ℝ
  delicate_rate : ℝ
  business_rate : ℝ
  bulky_rate : ℝ
  discount_rate : ℝ
  day1_regular : ℝ
  day1_delicate : ℝ
  day1_business : ℝ
  day1_bulky : ℝ
  day2_regular : ℝ
  day2_delicate : ℝ
  day2_business : ℝ
  day2_bulky : ℝ
  day3_regular : ℝ
  day3_delicate : ℝ
  day3_business : ℝ
  day3_bulky : ℝ
  overhead_costs : ℝ

/-- Calculate the net profit for a three-day period in a laundry shop --/
def calculate_net_profit (data : LaundryData) : ℝ :=
  let day1_total := data.regular_rate * data.day1_regular +
                    data.delicate_rate * data.day1_delicate +
                    data.business_rate * data.day1_business +
                    data.bulky_rate * data.day1_bulky
  let day2_total := data.regular_rate * data.day2_regular +
                    data.delicate_rate * data.day2_delicate +
                    data.business_rate * data.day2_business +
                    data.bulky_rate * data.day2_bulky
  let day3_total := (data.regular_rate * data.day3_regular +
                    data.delicate_rate * data.day3_delicate +
                    data.business_rate * data.day3_business +
                    data.bulky_rate * data.day3_bulky) * (1 - data.discount_rate)
  day1_total + day2_total + day3_total - data.overhead_costs

/-- Theorem: The net profit for the given three-day period is $107.70 --/
theorem net_profit_is_107_70 :
  let data := LaundryData.mk 3 4 5 6 0.1 7 4 3 2 10 6 4 3 20 4 5 2 150
  calculate_net_profit data = 107.7 := by
  sorry

end NUMINAMATH_CALUDE_net_profit_is_107_70_l2016_201657


namespace NUMINAMATH_CALUDE_solution_mixture_proof_l2016_201631

theorem solution_mixture_proof (x : ℝ) 
  (h1 : x + 20 = 100) -- First solution is x% carbonated water and 20% lemonade
  (h2 : 0.6799999999999997 * x + 0.32000000000000003 * 55 = 72) -- Mixture equation
  : x = 80 := by
  sorry

end NUMINAMATH_CALUDE_solution_mixture_proof_l2016_201631


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2016_201655

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 4, 3]
def den1 : List Nat := [1, 3]
def num2 : List Nat := [2, 0, 4]
def den2 : List Nat := [2, 3]

-- Convert the numbers to base 10
def num1_base10 : Nat := to_base_10 num1 8
def den1_base10 : Nat := to_base_10 den1 4
def num2_base10 : Nat := to_base_10 num2 7
def den2_base10 : Nat := to_base_10 den2 5

-- Define the theorem
theorem base_conversion_theorem :
  (num1_base10 : ℚ) / den1_base10 + (num2_base10 : ℚ) / den2_base10 = 31 + 51 / 91 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2016_201655


namespace NUMINAMATH_CALUDE_inequality_proof_l2016_201659

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + 
  (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2016_201659


namespace NUMINAMATH_CALUDE_initial_interest_rate_l2016_201637

/-- Given interest conditions, prove the initial interest rate -/
theorem initial_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Initial interest rate in percentage
  (h1 : P * r / 100 = 202.50)  -- Interest at initial rate
  (h2 : P * (r + 5) / 100 = 225)  -- Interest at increased rate
  : r = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_interest_rate_l2016_201637


namespace NUMINAMATH_CALUDE_ellipse_properties_l2016_201679

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eccentricity : ℝ
  eccentricity_eq : eccentricity = Real.sqrt 6 / 3
  equation : ℝ → ℝ → Prop
  equation_def : equation = λ x y => x^2 / a^2 + y^2 / b^2 = 1
  focal_line_length : ℝ
  focal_line_length_eq : focal_line_length = 2 * Real.sqrt 3 / 3

/-- The main theorem about the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  e.equation = λ x y => x^2 / 3 + y^2 = 1 ∧
  ∃ k : ℝ, k = 7 / 6 ∧
    ∀ C D : ℝ × ℝ,
      (e.equation C.1 C.2 ∧ e.equation D.1 D.2) →
      (C.2 = k * C.1 + 2 ∧ D.2 = k * D.1 + 2) →
      (C.1 - (-1))^2 + (C.2 - 0)^2 = (D.1 - (-1))^2 + (D.2 - 0)^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2016_201679


namespace NUMINAMATH_CALUDE_least_common_denominator_l2016_201666

theorem least_common_denominator : 
  let denominators := [3, 4, 5, 6, 8, 9, 10]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 6) 8) 9) 10 = 360 :=
by sorry

end NUMINAMATH_CALUDE_least_common_denominator_l2016_201666


namespace NUMINAMATH_CALUDE_product_of_real_parts_quadratic_complex_l2016_201619

theorem product_of_real_parts_quadratic_complex (x : ℂ) :
  x^2 + 3*x = -2 + 2*I →
  ∃ (s₁ s₂ : ℂ), (s₁^2 + 3*s₁ = -2 + 2*I) ∧ 
                 (s₂^2 + 3*s₂ = -2 + 2*I) ∧
                 (s₁.re * s₂.re = (5 - 2*Real.sqrt 5) / 4) :=
by sorry

end NUMINAMATH_CALUDE_product_of_real_parts_quadratic_complex_l2016_201619


namespace NUMINAMATH_CALUDE_emilia_valentin_numbers_l2016_201645

theorem emilia_valentin_numbers (x : ℝ) : 
  (5 + 9) / 2 = 7 ∧ 
  (5 + x) / 2 = 10 ∧ 
  (x + 9) / 2 = 12 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_emilia_valentin_numbers_l2016_201645


namespace NUMINAMATH_CALUDE_factorization_4a_squared_minus_2a_l2016_201656

-- Define what it means for an expression to be a factorization from left to right
def is_factorization_left_to_right (f g : ℝ → ℝ) : Prop :=
  ∃ (h k : ℝ → ℝ), (∀ x, f x = h x * k x) ∧ (∀ x, g x = h x * k x) ∧ (f ≠ g)

-- Define the left side of the equation
def left_side (a : ℝ) : ℝ := 4 * a^2 - 2 * a

-- Define the right side of the equation
def right_side (a : ℝ) : ℝ := 2 * a * (2 * a - 1)

-- Theorem statement
theorem factorization_4a_squared_minus_2a :
  is_factorization_left_to_right left_side right_side :=
sorry

end NUMINAMATH_CALUDE_factorization_4a_squared_minus_2a_l2016_201656


namespace NUMINAMATH_CALUDE_intersection_A_B_l2016_201687

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2016_201687


namespace NUMINAMATH_CALUDE_probability_point_closer_to_center_l2016_201681

theorem probability_point_closer_to_center (R : ℝ) (r : ℝ) : R > 0 → r > 0 → R = 3 * r →
  (π * (2 * r)^2) / (π * R^2) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_closer_to_center_l2016_201681


namespace NUMINAMATH_CALUDE_crayon_count_initial_crayon_count_l2016_201624

theorem crayon_count (crayons_taken : ℕ) (crayons_left : ℕ) : ℕ :=
  crayons_taken + crayons_left

theorem initial_crayon_count : crayon_count 3 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_initial_crayon_count_l2016_201624


namespace NUMINAMATH_CALUDE_volume_of_special_prism_l2016_201625

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  base : Set Point3D
  height : ℝ

/-- Given a cube, returns the midpoints of edges AB, AD, and AA₁ -/
def getMidpoints (c : Cube) : Set Point3D :=
  { Point3D.mk (c.edgeLength / 2) 0 0,
    Point3D.mk 0 (c.edgeLength / 2) 0,
    Point3D.mk 0 0 (c.edgeLength / 2) }

/-- Constructs a triangular prism from given midpoints -/
def constructPrism (midpoints : Set Point3D) (c : Cube) : TriangularPrism :=
  sorry

/-- Calculates the volume of a triangular prism -/
def prismVolume (p : TriangularPrism) : ℝ :=
  sorry

theorem volume_of_special_prism (c : Cube) :
  c.edgeLength = 1 →
  let midpoints := getMidpoints c
  let prism := constructPrism midpoints c
  prismVolume prism = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_special_prism_l2016_201625


namespace NUMINAMATH_CALUDE_intersection_angle_l2016_201665

/-- A regular hexagonal pyramid with lateral faces at 45° to the base -/
structure RegularHexagonalPyramid :=
  (base : Set (ℝ × ℝ))
  (apex : ℝ × ℝ × ℝ)
  (lateral_angle : Real)
  (is_regular : Bool)
  (lateral_angle_eq : lateral_angle = Real.pi / 4)

/-- A plane intersecting the pyramid -/
structure IntersectingPlane :=
  (base_edge : Set (ℝ × ℝ))
  (intersections : Set (ℝ × ℝ × ℝ))
  (is_parallel : Bool)

/-- The theorem to be proved -/
theorem intersection_angle (p : RegularHexagonalPyramid) (s : IntersectingPlane) :
  p.is_regular ∧ s.is_parallel →
  ∃ α : Real, α = Real.arctan (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_angle_l2016_201665


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l2016_201653

-- Define a function to count the number of positive factors of a natural number
def countFactors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number has exactly 12 factors
def has12Factors (n : ℕ) : Prop := countFactors n = 12

-- Theorem statement
theorem least_integer_with_12_factors :
  ∃ (k : ℕ), k > 0 ∧ has12Factors k ∧ ∀ (m : ℕ), m > 0 → has12Factors m → k ≤ m :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l2016_201653


namespace NUMINAMATH_CALUDE_birthday_800th_day_l2016_201633

/-- Given a person born on a Tuesday, their 800th day of life will fall on a Thursday. -/
theorem birthday_800th_day (birth_day : Nat) (days_passed : Nat) : 
  birth_day = 2 → days_passed = 800 → (birth_day + days_passed) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_birthday_800th_day_l2016_201633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2016_201636

/-- Given an arithmetic sequence where the 5th term is 23 and the 7th term is 37, 
    prove that the 9th term is 51. -/
theorem arithmetic_sequence_ninth_term 
  (a : ℤ) -- First term of the sequence
  (d : ℤ) -- Common difference
  (h1 : a + 4 * d = 23) -- 5th term is 23
  (h2 : a + 6 * d = 37) -- 7th term is 37
  : a + 8 * d = 51 := by -- 9th term is 51
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2016_201636


namespace NUMINAMATH_CALUDE_number_of_sodas_bought_l2016_201627

/-- Given the total cost, sandwich cost, and soda cost, calculate the number of sodas bought -/
theorem number_of_sodas_bought (total_cost sandwich_cost soda_cost : ℚ) 
  (h_total : total_cost = 8.36)
  (h_sandwich : sandwich_cost = 2.44)
  (h_soda : soda_cost = 0.87) :
  (total_cost - 2 * sandwich_cost) / soda_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_number_of_sodas_bought_l2016_201627


namespace NUMINAMATH_CALUDE_board_numbers_l2016_201620

theorem board_numbers (a b : ℕ+) : 
  (a.val - b.val)^2 = a.val^2 - b.val^2 - 4038 →
  ((a.val = 2020 ∧ b.val = 1) ∨
   (a.val = 2020 ∧ b.val = 2019) ∨
   (a.val = 676 ∧ b.val = 3) ∨
   (a.val = 676 ∧ b.val = 673)) :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_l2016_201620


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l2016_201615

/-- Given an arithmetic sequence with first term 5/8 and eleventh term 3/4,
    the sixth term is 11/16. -/
theorem sixth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), 
    (∀ n m, a (n + m) - a n = m * (a 2 - a 1)) →  -- arithmetic sequence condition
    a 1 = 5/8 →                                   -- first term
    a 11 = 3/4 →                                  -- eleventh term
    a 6 = 11/16 :=                                -- sixth term
by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l2016_201615


namespace NUMINAMATH_CALUDE_small_planter_capacity_l2016_201629

/-- Given the total number of seeds, the number and capacity of large planters,
    and the number of small planters, prove that each small planter can hold 4 seeds. -/
theorem small_planter_capacity
  (total_seeds : ℕ)
  (large_planters : ℕ)
  (large_planter_capacity : ℕ)
  (small_planters : ℕ)
  (h1 : total_seeds = 200)
  (h2 : large_planters = 4)
  (h3 : large_planter_capacity = 20)
  (h4 : small_planters = 30)
  : (total_seeds - large_planters * large_planter_capacity) / small_planters = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_planter_capacity_l2016_201629


namespace NUMINAMATH_CALUDE_vector_addition_l2016_201622

/-- Given two vectors AB and BC in 2D space, prove that AC is their sum. -/
theorem vector_addition (AB BC : ℝ × ℝ) (h1 : AB = (2, 3)) (h2 : BC = (1, -4)) :
  AB.1 + BC.1 = 3 ∧ AB.2 + BC.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l2016_201622


namespace NUMINAMATH_CALUDE_sqrt_problem_l2016_201680

theorem sqrt_problem (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_problem_l2016_201680


namespace NUMINAMATH_CALUDE_chain_breaking_theorem_l2016_201604

/-- Represents a chain with n links -/
structure Chain (n : ℕ) where
  links : Fin n → ℕ
  all_links_one : ∀ i, links i = 1

/-- Represents a set of chain segments after breaking -/
structure Segments (n : ℕ) where
  pieces : List ℕ
  sum_pieces : pieces.sum = n

/-- Function to break a chain into segments -/
def break_chain (n : ℕ) (k : ℕ) (break_points : Fin (k-1) → ℕ) : Segments n :=
  sorry

/-- Function to check if a weight can be measured using given segments -/
def can_measure (segments : List ℕ) (weight : ℕ) : Prop :=
  sorry

theorem chain_breaking_theorem (k : ℕ) :
  let n := k * 2^k - 1
  ∃ (break_points : Fin (k-1) → ℕ),
    let segments := (break_chain n k break_points).pieces
    ∀ w : ℕ, w ≤ n → can_measure segments w :=
  sorry

end NUMINAMATH_CALUDE_chain_breaking_theorem_l2016_201604


namespace NUMINAMATH_CALUDE_tennis_tournament_n_is_five_l2016_201643

/-- Represents a tennis tournament with the given conditions --/
structure TennisTournament where
  n : ℕ
  total_players : ℕ := 5 * n
  total_matches : ℕ := (total_players * (total_players - 1)) / 2
  women_wins : ℕ
  men_wins : ℕ
  no_ties : women_wins + men_wins = total_matches
  win_ratio : women_wins * 2 = men_wins * 3

/-- The theorem stating that n must be 5 for the given conditions --/
theorem tennis_tournament_n_is_five :
  ∀ t : TennisTournament, t.n = 5 := by sorry

end NUMINAMATH_CALUDE_tennis_tournament_n_is_five_l2016_201643


namespace NUMINAMATH_CALUDE_no_defective_products_exactly_two_defective_products_at_least_two_defective_products_l2016_201697

-- Define the total number of items
def total_items : ℕ := 100

-- Define the number of defective items
def defective_items : ℕ := 3

-- Define the number of items to be selected
def selected_items : ℕ := 5

-- Theorem for scenario (I): No defective product
theorem no_defective_products : 
  Nat.choose (total_items - defective_items) selected_items = 64446024 := by sorry

-- Theorem for scenario (II): Exactly two defective products
theorem exactly_two_defective_products :
  Nat.choose defective_items 2 * Nat.choose (total_items - defective_items) (selected_items - 2) = 442320 := by sorry

-- Theorem for scenario (III): At least two defective products
theorem at_least_two_defective_products :
  Nat.choose defective_items 2 * Nat.choose (total_items - defective_items) (selected_items - 2) +
  Nat.choose defective_items 3 * Nat.choose (total_items - defective_items) (selected_items - 3) = 446886 := by sorry

end NUMINAMATH_CALUDE_no_defective_products_exactly_two_defective_products_at_least_two_defective_products_l2016_201697


namespace NUMINAMATH_CALUDE_y_percentage_more_than_z_l2016_201648

/-- Given that x gets 25% more than y, the total amount is 370, and z's share is 100,
    prove that y gets 20% more than z. -/
theorem y_percentage_more_than_z (x y z : ℝ) : 
  x = 1.25 * y →  -- x gets 25% more than y
  x + y + z = 370 →  -- total amount is 370
  z = 100 →  -- z's share is 100
  y = 1.2 * z  -- y gets 20% more than z
  := by sorry

end NUMINAMATH_CALUDE_y_percentage_more_than_z_l2016_201648


namespace NUMINAMATH_CALUDE_unique_prime_103207_l2016_201675

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_prime_103207 :
  (¬ is_prime 103201) ∧
  (¬ is_prime 103202) ∧
  (¬ is_prime 103203) ∧
  (is_prime 103207) ∧
  (¬ is_prime 103209) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_103207_l2016_201675


namespace NUMINAMATH_CALUDE_solution_pairs_l2016_201692

theorem solution_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_solution_pairs_l2016_201692


namespace NUMINAMATH_CALUDE_girls_to_boys_fraction_l2016_201693

theorem girls_to_boys_fraction (total : ℕ) (girls : ℕ) (h1 : total = 35) (h2 : girls = 10) :
  (girls : ℚ) / ((total - girls) : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_girls_to_boys_fraction_l2016_201693


namespace NUMINAMATH_CALUDE_factorization_identity_l2016_201601

theorem factorization_identity (x y : ℝ) : x^2 - 2*x*y + y^2 - 1 = (x - y + 1) * (x - y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l2016_201601


namespace NUMINAMATH_CALUDE_marble_jar_problem_l2016_201602

theorem marble_jar_problem (num_marbles : ℕ) : 
  (∀ (x : ℚ), x = num_marbles / 20 → 
    x - 1 = num_marbles / 22) → 
  num_marbles = 220 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l2016_201602


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2016_201646

/-- Calculates the price of a stock given the investment amount, stock percentage, and annual income. -/
theorem stock_price_calculation (investment : ℝ) (stock_percentage : ℝ) (annual_income : ℝ) :
  investment = 6800 ∧ 
  stock_percentage = 0.6 ∧ 
  annual_income = 3000 →
  ∃ (stock_price : ℝ), stock_price = 136 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2016_201646


namespace NUMINAMATH_CALUDE_folk_song_competition_probability_l2016_201667

theorem folk_song_competition_probability : 
  ∀ (n m k : ℕ),
  n = 6 →  -- number of provinces
  m = 2 →  -- number of singers per province
  k = 4 →  -- number of winners selected
  (Nat.choose n 1 * Nat.choose (n - 1) 2 * Nat.choose m 1 * Nat.choose m 1) / 
  (Nat.choose (n * m) k) = 16 / 33 := by
  sorry

end NUMINAMATH_CALUDE_folk_song_competition_probability_l2016_201667


namespace NUMINAMATH_CALUDE_similar_squares_side_length_l2016_201689

theorem similar_squares_side_length (s1 s2 : ℝ) (h1 : s1 > 0) (h2 : s2 > 0) : 
  (s1 ^ 2 : ℝ) / (s2 ^ 2) = 9 → s2 = 5 → s1 = 15 := by sorry

end NUMINAMATH_CALUDE_similar_squares_side_length_l2016_201689


namespace NUMINAMATH_CALUDE_max_expression_value_l2016_201663

def expression (a b c d : ℕ) : ℕ := c * a^b - d

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 5 ∧
    ∀ (a' b' c' d' : ℕ),
      a' ∈ ({1, 2, 3, 4} : Set ℕ) →
      b' ∈ ({1, 2, 3, 4} : Set ℕ) →
      c' ∈ ({1, 2, 3, 4} : Set ℕ) →
      d' ∈ ({1, 2, 3, 4} : Set ℕ) →
      a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ b' ≠ c' ∧ b' ≠ d' ∧ c' ≠ d' →
      expression a' b' c' d' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_expression_value_l2016_201663


namespace NUMINAMATH_CALUDE_fourth_game_shots_l2016_201638

/-- Given a basketball player's performance over four games, calculate the number of successful shots in the fourth game. -/
theorem fourth_game_shots (initial_shots initial_made fourth_game_shots : ℕ) 
  (h1 : initial_shots = 30)
  (h2 : initial_made = 12)
  (h3 : fourth_game_shots = 10)
  (h4 : (initial_made : ℚ) / initial_shots = 2/5)
  (h5 : ((initial_made + x) : ℚ) / (initial_shots + fourth_game_shots) = 1/2) :
  x = 8 :=
sorry

end NUMINAMATH_CALUDE_fourth_game_shots_l2016_201638


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l2016_201641

theorem min_value_sum_of_squares (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l2016_201641


namespace NUMINAMATH_CALUDE_right_triangle_point_distance_l2016_201618

theorem right_triangle_point_distance (h d x : ℝ) : 
  h > 0 → d > 0 → x > 0 →
  x + Real.sqrt ((x + h)^2 + d^2) = h + d →
  x = h * d / (2 * h + d) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_point_distance_l2016_201618


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2016_201652

theorem simplify_fraction_product : 8 * (15 / 9) * (-45 / 40) = -1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2016_201652


namespace NUMINAMATH_CALUDE_work_completion_time_l2016_201626

/-- The efficiency of worker q -/
def q_efficiency : ℝ := 1

/-- The efficiency of worker p relative to q -/
def p_efficiency : ℝ := 1.6

/-- The efficiency of worker r relative to q -/
def r_efficiency : ℝ := 1.4

/-- The time taken by p alone to complete the work -/
def p_time : ℝ := 26

/-- The total amount of work to be done -/
def total_work : ℝ := p_efficiency * p_time

/-- The combined efficiency of p, q, and r -/
def combined_efficiency : ℝ := p_efficiency + q_efficiency + r_efficiency

/-- The theorem stating the time taken for p, q, and r to complete the work together -/
theorem work_completion_time : 
  total_work / combined_efficiency = 10.4 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2016_201626


namespace NUMINAMATH_CALUDE_abs_neg_five_equals_five_l2016_201608

theorem abs_neg_five_equals_five :
  abs (-5 : ℤ) = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_five_equals_five_l2016_201608


namespace NUMINAMATH_CALUDE_dehydrated_men_fraction_l2016_201600

theorem dehydrated_men_fraction (total_men : ℕ) (finished_men : ℕ) 
  (h1 : total_men = 80)
  (h2 : finished_men = 52)
  (h3 : (1 : ℚ) / 4 * total_men = total_men - (total_men - (1 : ℚ) / 4 * total_men))
  (h4 : ∃ x : ℚ, x * (total_men - (1 : ℚ) / 4 * total_men) * (1 : ℚ) / 5 = 
    total_men - finished_men - (1 : ℚ) / 4 * total_men) :
  ∃ x : ℚ, x = 2 / 3 ∧ 
    x * (total_men - (1 : ℚ) / 4 * total_men) * (1 : ℚ) / 5 = 
    total_men - finished_men - (1 : ℚ) / 4 * total_men :=
by sorry


end NUMINAMATH_CALUDE_dehydrated_men_fraction_l2016_201600


namespace NUMINAMATH_CALUDE_fixed_costs_calculation_l2016_201660

/-- The fixed monthly costs for a computer manufacturer producing electronic components -/
def fixed_monthly_costs : ℝ := 16699.50

/-- The production cost per component -/
def production_cost : ℝ := 80

/-- The shipping cost per component -/
def shipping_cost : ℝ := 7

/-- The number of components produced and sold per month -/
def monthly_units : ℕ := 150

/-- The lowest selling price per component for break-even -/
def selling_price : ℝ := 198.33

theorem fixed_costs_calculation :
  fixed_monthly_costs = 
    selling_price * monthly_units - 
    (production_cost + shipping_cost) * monthly_units :=
by sorry

end NUMINAMATH_CALUDE_fixed_costs_calculation_l2016_201660


namespace NUMINAMATH_CALUDE_number_of_spinsters_l2016_201623

-- Define the number of spinsters and cats
def spinsters : ℕ := sorry
def cats : ℕ := sorry

-- State the theorem
theorem number_of_spinsters :
  -- Condition 1: The ratio of spinsters to cats is 2:7
  (spinsters : ℚ) / cats = 2 / 7 →
  -- Condition 2: There are 55 more cats than spinsters
  cats = spinsters + 55 →
  -- Conclusion: The number of spinsters is 22
  spinsters = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_spinsters_l2016_201623


namespace NUMINAMATH_CALUDE_sequence_length_l2016_201609

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Theorem statement
theorem sequence_length :
  let a₁ : ℝ := 2.5
  let d : ℝ := 5
  let aₙ : ℝ := 62.5
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence a₁ d n = aₙ ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_l2016_201609


namespace NUMINAMATH_CALUDE_return_trip_time_l2016_201621

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  d : ℝ  -- distance between cities
  v : ℝ  -- speed of plane in still air
  u : ℝ  -- speed of wind
  outboundTime : ℝ  -- time from A to B against wind
  returnTimeDifference : ℝ  -- difference in return time compared to calm air

/-- Conditions for the flight scenario --/
def flightConditions (s : FlightScenario) : Prop :=
  s.v > 0 ∧ s.u > 0 ∧ s.d > 0 ∧
  s.outboundTime = 60 ∧
  s.returnTimeDifference = 10 ∧
  s.d = s.outboundTime * (s.v - s.u) ∧
  s.d / (s.v + s.u) = s.d / s.v - s.returnTimeDifference

/-- The theorem stating that the return trip takes 20 minutes --/
theorem return_trip_time (s : FlightScenario) 
  (h : flightConditions s) : s.d / (s.v + s.u) = 20 := by
  sorry


end NUMINAMATH_CALUDE_return_trip_time_l2016_201621


namespace NUMINAMATH_CALUDE_bookmark_position_l2016_201642

/-- Represents a book with pages and a bookmark --/
structure Book where
  pages : ℕ
  coverThickness : ℕ
  bookmarkPosition : ℕ

/-- Calculates the total thickness of a book in page-equivalent units --/
def bookThickness (b : Book) : ℕ := b.pages + 2 * b.coverThickness

/-- The problem setup --/
def bookshelfProblem (book1 book2 : Book) : Prop :=
  book1.pages = 250 ∧
  book2.pages = 250 ∧
  book1.coverThickness = 10 ∧
  book2.coverThickness = 10 ∧
  book1.bookmarkPosition = 125 ∧
  (bookThickness book1 + bookThickness book2) / 3 = book1.bookmarkPosition + book1.coverThickness + book2.bookmarkPosition

theorem bookmark_position (book1 book2 : Book) :
  bookshelfProblem book1 book2 → book2.bookmarkPosition = 35 :=
by sorry

end NUMINAMATH_CALUDE_bookmark_position_l2016_201642


namespace NUMINAMATH_CALUDE_three_rug_overlap_l2016_201649

theorem three_rug_overlap (total_rug_area floor_area double_layer_area : ℝ) 
  (h1 : total_rug_area = 90)
  (h2 : floor_area = 60)
  (h3 : double_layer_area = 12) : 
  ∃ (triple_layer_area : ℝ),
    triple_layer_area = 9 ∧
    ∃ (single_layer_area : ℝ),
      single_layer_area + double_layer_area + triple_layer_area = floor_area ∧
      single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = total_rug_area :=
by sorry

end NUMINAMATH_CALUDE_three_rug_overlap_l2016_201649


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l2016_201670

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.log x / Real.log a

-- Theorem statement
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  (1 / 7 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_a_range_l2016_201670


namespace NUMINAMATH_CALUDE_roses_kept_l2016_201699

theorem roses_kept (initial : ℕ) (mother grandmother sister : ℕ) 
  (h1 : initial = 20)
  (h2 : mother = 6)
  (h3 : grandmother = 9)
  (h4 : sister = 4) :
  initial - (mother + grandmother + sister) = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_kept_l2016_201699


namespace NUMINAMATH_CALUDE_pregnant_cow_percentage_l2016_201614

theorem pregnant_cow_percentage (total_cows : ℕ) (female_percentage : ℚ) (pregnant_cows : ℕ) : 
  total_cows = 44 →
  female_percentage = 1/2 →
  pregnant_cows = 11 →
  (pregnant_cows : ℚ) / (female_percentage * total_cows) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pregnant_cow_percentage_l2016_201614


namespace NUMINAMATH_CALUDE_school_chairs_problem_l2016_201688

theorem school_chairs_problem (initial_chairs : ℕ) : 
  initial_chairs < 35 →
  ∃ (k : ℕ), initial_chairs + 27 = 35 * k →
  initial_chairs = 8 := by
sorry

end NUMINAMATH_CALUDE_school_chairs_problem_l2016_201688


namespace NUMINAMATH_CALUDE_intersection_values_l2016_201672

/-- Definition of the circle M -/
def circle_M (x y : ℝ) : Prop := x^2 - 2*x + y^2 + 4*y - 10 = 0

/-- Definition of the intersecting line -/
def intersecting_line (x y : ℝ) (C : ℝ) : Prop := x + 3*y + C = 0

/-- Theorem stating the possible values of C -/
theorem intersection_values (C : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    circle_M A.1 A.2 ∧ 
    circle_M B.1 B.2 ∧
    intersecting_line A.1 A.2 C ∧
    intersecting_line B.1 B.2 C ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 20) →
  C = 15 ∨ C = -5 := by
sorry

end NUMINAMATH_CALUDE_intersection_values_l2016_201672


namespace NUMINAMATH_CALUDE_terminal_side_angle_expression_l2016_201635

theorem terminal_side_angle_expression (α : Real) :
  let P : Real × Real := (1, 3)
  let r : Real := Real.sqrt (P.1^2 + P.2^2)
  (P.1 / r = Real.cos α) ∧ (P.2 / r = Real.sin α) →
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_angle_expression_l2016_201635


namespace NUMINAMATH_CALUDE_cloth_cutting_l2016_201678

theorem cloth_cutting (S : ℝ) : 
  S / 2 + S / 4 = 75 → S = 100 := by
sorry

end NUMINAMATH_CALUDE_cloth_cutting_l2016_201678


namespace NUMINAMATH_CALUDE_unique_partition_count_l2016_201630

/-- The number of ways to partition n into three distinct positive integers -/
def partition_count (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2 - 3 * ((n / 2) - 2) - 1

/-- Theorem stating that 18 is the only positive integer satisfying the condition -/
theorem unique_partition_count :
  ∀ n : ℕ, n > 0 → (partition_count n = n + 1 ↔ n = 18) := by sorry

end NUMINAMATH_CALUDE_unique_partition_count_l2016_201630


namespace NUMINAMATH_CALUDE_point_inside_circle_l2016_201617

theorem point_inside_circle (r d : ℝ) (hr : r = 6) (hd : d = 4) :
  d < r → ∃ (P : ℝ × ℝ) (O : ℝ × ℝ), ‖P - O‖ = d ∧ P ∈ interior {x | ‖x - O‖ ≤ r} :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2016_201617


namespace NUMINAMATH_CALUDE_binary_sum_to_octal_to_decimal_l2016_201605

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 8^(digits.length - 1 - i)) 0

/-- The main theorem to be proved -/
theorem binary_sum_to_octal_to_decimal : 
  let a := binary_to_decimal [true, true, true, true, true, true, true, true]
  let b := binary_to_decimal [true, true, true, true, true]
  let sum := a + b
  let octal := decimal_to_octal sum
  octal_to_decimal octal = 286 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_to_octal_to_decimal_l2016_201605


namespace NUMINAMATH_CALUDE_lines_non_intersecting_l2016_201640

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the property of a line being contained within a plane
variable (contained_in_plane : Line → Plane → Prop)

-- Define the property of two lines being non-intersecting
variable (non_intersecting : Line → Line → Prop)

-- State the theorem
theorem lines_non_intersecting
  (l a : Line) (α : Plane)
  (h1 : parallel_to_plane l α)
  (h2 : contained_in_plane a α) :
  non_intersecting l a :=
sorry

end NUMINAMATH_CALUDE_lines_non_intersecting_l2016_201640


namespace NUMINAMATH_CALUDE_diego_fruit_problem_l2016_201696

/-- Given a bag with capacity for fruit and some fruits already in the bag,
    calculate the remaining capacity for additional fruit. -/
def remaining_capacity (bag_capacity : ℕ) (occupied_capacity : ℕ) : ℕ :=
  bag_capacity - occupied_capacity

/-- Diego's fruit buying problem -/
theorem diego_fruit_problem (bag_capacity : ℕ) (watermelon_weight : ℕ) (grapes_weight : ℕ) (oranges_weight : ℕ) 
  (h1 : bag_capacity = 20)
  (h2 : watermelon_weight = 1)
  (h3 : grapes_weight = 1)
  (h4 : oranges_weight = 1) :
  remaining_capacity bag_capacity (watermelon_weight + grapes_weight + oranges_weight) = 17 :=
sorry

end NUMINAMATH_CALUDE_diego_fruit_problem_l2016_201696


namespace NUMINAMATH_CALUDE_theater_population_l2016_201690

theorem theater_population :
  ∀ (total : ℕ),
  (19 : ℕ) + (total / 2) + (total / 4) + 6 = total →
  total = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_population_l2016_201690


namespace NUMINAMATH_CALUDE_group_size_l2016_201684

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  average_increase = 2 →
  old_weight = 65 →
  new_weight = 81 →
  (new_weight - old_weight) / average_increase = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_l2016_201684


namespace NUMINAMATH_CALUDE_inequality_solution_l2016_201685

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 23 / 10) ↔ (x > -2 ∧ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2016_201685


namespace NUMINAMATH_CALUDE_factorization_equality_l2016_201686

theorem factorization_equality (x : ℝ) : 16 * x^3 + 8 * x^2 = 8 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2016_201686


namespace NUMINAMATH_CALUDE_tuesday_pages_l2016_201698

/-- Represents the number of pages read on each day of the week --/
structure PagesRead where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Represents the reading plan for the week --/
def ReadingPlan (total_pages : ℕ) (pages : PagesRead) : Prop :=
  pages.monday = 23 ∧
  pages.wednesday = 61 ∧
  pages.thursday = 12 ∧
  pages.friday = 2 * pages.thursday ∧
  total_pages = pages.monday + pages.tuesday + pages.wednesday + pages.thursday + pages.friday

theorem tuesday_pages (total_pages : ℕ) (pages : PagesRead) 
  (h : ReadingPlan total_pages pages) (h_total : total_pages = 158) : 
  pages.tuesday = 38 := by
  sorry

#check tuesday_pages

end NUMINAMATH_CALUDE_tuesday_pages_l2016_201698


namespace NUMINAMATH_CALUDE_extreme_values_depend_on_a_consistent_monotonicity_implies_b_bound_max_ab_difference_l2016_201632

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

-- Define the derivatives of f and g
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a
def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + b

-- Define consistent monotonicity
def consistent_monotonicity (a b : ℝ) (l : Set ℝ) : Prop :=
  ∀ x ∈ l, f' a x * g' b x ≥ 0

theorem extreme_values_depend_on_a (a : ℝ) : 
  (a ≥ 0 → ∀ x : ℝ, f' a x ≥ 0) ∧ 
  (a < 0 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f' a x₁ < 0 ∧ f' a x₂ > 0) :=
sorry

theorem consistent_monotonicity_implies_b_bound (a b : ℝ) :
  a > 0 → consistent_monotonicity a b { x | x ≥ -2 } → b ≥ 4 :=
sorry

theorem max_ab_difference (a b : ℝ) :
  a < 0 → a ≠ b → consistent_monotonicity a b (Set.Ioo a b) → |a - b| ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_depend_on_a_consistent_monotonicity_implies_b_bound_max_ab_difference_l2016_201632


namespace NUMINAMATH_CALUDE_contradiction_assumption_l2016_201651

theorem contradiction_assumption (x y : ℝ) (h : x > y) : 
  ¬(x^3 > y^3) ↔ x^3 ≤ y^3 := by
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l2016_201651


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l2016_201607

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

/-- Theorem: The equation of the circle with diameter endpoints A(1,4) and B(3,-2) -/
theorem circle_equation_from_diameter_endpoints :
  let A : Point := ⟨1, 4⟩
  let B : Point := ⟨3, -2⟩
  let center : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
  let radius : ℝ := Real.sqrt ((B.x - center.x)^2 + (B.y - center.y)^2)
  CircleEquation center radius = fun x y ↦ (x - 2)^2 + (y - 1)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l2016_201607
