import Mathlib

namespace NUMINAMATH_CALUDE_complex_and_trig_problem_l1717_171794

/-- Given a complex number z and an angle θ, prove the magnitude of z and a trigonometric expression -/
theorem complex_and_trig_problem (z : ℂ) (θ : ℝ) : 
  θ = 4 * π / 3 →
  (∃ (x y : ℝ), z = x + y * I ∧ x + 3 * y = 0) →
  Complex.abs z = Real.sqrt 21 / 2 ∧
  (2 * Real.cos (θ / 2) ^ 2 - 1) / (Real.sqrt 2 * Real.sin (θ + π / 4)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_and_trig_problem_l1717_171794


namespace NUMINAMATH_CALUDE_paper_number_sum_paper_number_sum_proof_l1717_171774

/-- Given n pieces of paper, each containing 3 different positive integers no greater than n,
    and any two pieces sharing exactly one common number, prove that the sum of all numbers
    written on these pieces of paper is equal to 3 * n(n+1)/2. -/
theorem paper_number_sum (n : ℕ) : ℕ :=
  let paper_count := n
  let max_number := n
  let numbers_per_paper := 3
  let shared_number_count := 1
  3 * (n * (n + 1) / 2)

-- The proof is omitted as per instructions
theorem paper_number_sum_proof (n : ℕ) :
  paper_number_sum n = 3 * (n * (n + 1) / 2) := by sorry

end NUMINAMATH_CALUDE_paper_number_sum_paper_number_sum_proof_l1717_171774


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l1717_171746

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l1717_171746


namespace NUMINAMATH_CALUDE_sports_suits_cost_prices_l1717_171737

/-- The cost price of one set of type A sports suits -/
def cost_A : ℝ := 180

/-- The cost price of one set of type B sports suits -/
def cost_B : ℝ := 200

/-- The total cost of purchasing one set of each type -/
def total_cost : ℝ := 380

/-- The total amount spent on type A sports suits -/
def total_A : ℝ := 8100

/-- The total amount spent on type B sports suits -/
def total_B : ℝ := 9000

theorem sports_suits_cost_prices :
  (total_A / cost_A = total_B / cost_B) ∧
  (cost_A + cost_B = total_cost) ∧
  (cost_A = 180) ∧
  (cost_B = 200) := by
  sorry

end NUMINAMATH_CALUDE_sports_suits_cost_prices_l1717_171737


namespace NUMINAMATH_CALUDE_bobby_candy_theorem_l1717_171771

/-- The number of candy pieces Bobby ate -/
def pieces_eaten : ℕ := 23

/-- The number of candy pieces Bobby has left -/
def pieces_left : ℕ := 7

/-- The initial number of candy pieces Bobby had -/
def initial_pieces : ℕ := pieces_eaten + pieces_left

theorem bobby_candy_theorem : initial_pieces = 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_theorem_l1717_171771


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1717_171702

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 5*x + 4 = 0 ↔ x = 4 ∨ x = 1) ∧
  (∃ x : ℝ, x^2 = 4 - 2*x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1717_171702


namespace NUMINAMATH_CALUDE_remaining_amount_is_15_60_l1717_171768

/-- Calculate the remaining amount for a trip given expenses and gifts --/
def calculate_remaining_amount (initial_amount gas_cost lunch_cost gift_cost_per_person num_people extra_gift_cost grandma_gift toll_fee ice_cream_cost : ℚ) : ℚ :=
  let total_spent := gas_cost + lunch_cost + (gift_cost_per_person * num_people) + extra_gift_cost
  let total_received := initial_amount + (grandma_gift * num_people)
  let amount_before_return := total_received - total_spent
  amount_before_return - (toll_fee + ice_cream_cost)

/-- Theorem stating that the remaining amount for the return trip is $15.60 --/
theorem remaining_amount_is_15_60 :
  calculate_remaining_amount 60 12 23.40 5 3 7 10 8 9 = 15.60 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_is_15_60_l1717_171768


namespace NUMINAMATH_CALUDE_solution_set_a_range_l1717_171724

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Part 1
theorem solution_set (x : ℝ) :
  (f x 3 > g 3 + 2) ↔ (x < -4 ∨ x > 2) :=
sorry

-- Part 2
theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) → a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_range_l1717_171724


namespace NUMINAMATH_CALUDE_sin_negative_150_degrees_l1717_171751

theorem sin_negative_150_degrees : Real.sin (-(150 * π / 180)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_150_degrees_l1717_171751


namespace NUMINAMATH_CALUDE_sum_of_fractions_and_decimal_l1717_171767

theorem sum_of_fractions_and_decimal : 
  (1 : ℚ) / 3 + 5 / 24 + (816 : ℚ) / 100 + 1 / 8 = 5296 / 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_and_decimal_l1717_171767


namespace NUMINAMATH_CALUDE_intersection_A_B_subset_A_C_iff_a_in_0_2_l1717_171720

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def B : Set ℝ := {x | (x-1)/(x-3) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (2*a+4)*x + a^2 + 4*a ≤ 0}

-- Define the interval (3, 4]
def interval_3_4 : Set ℝ := {x | 3 < x ∧ x ≤ 4}

-- Theorem statements
theorem intersection_A_B : A ∩ B = interval_3_4 := by sorry

theorem subset_A_C_iff_a_in_0_2 :
  ∀ a : ℝ, A ⊆ C a ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_subset_A_C_iff_a_in_0_2_l1717_171720


namespace NUMINAMATH_CALUDE_stamp_collection_value_l1717_171732

theorem stamp_collection_value
  (total_stamps : ℕ)
  (sample_stamps : ℕ)
  (sample_value : ℚ)
  (h1 : total_stamps = 18)
  (h2 : sample_stamps = 6)
  (h3 : sample_value = 15)
  : ℚ :=
by
  -- The total value of the stamp collection is 45 dollars
  sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l1717_171732


namespace NUMINAMATH_CALUDE_bobs_age_l1717_171729

theorem bobs_age (alice : ℝ) (bob : ℝ) 
  (h1 : bob = 3 * alice - 20) 
  (h2 : bob + alice = 70) : 
  bob = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_bobs_age_l1717_171729


namespace NUMINAMATH_CALUDE_simplify_expression_l1717_171798

theorem simplify_expression (a b : ℝ) (h : a = -b) :
  (2 * a * b * (a^3 - b^3)) / (a^2 + a*b + b^2) - 
  ((a - b) * (a^4 - b^4)) / (a^2 - b^2) = -8 * a^3 := by
  sorry

#check simplify_expression

end NUMINAMATH_CALUDE_simplify_expression_l1717_171798


namespace NUMINAMATH_CALUDE_cards_left_l1717_171787

def basketball_boxes : ℕ := 4
def basketball_cards_per_box : ℕ := 10
def baseball_boxes : ℕ := 5
def baseball_cards_per_box : ℕ := 8
def cards_given_away : ℕ := 58

theorem cards_left : 
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box - 
  cards_given_away = 22 := by sorry

end NUMINAMATH_CALUDE_cards_left_l1717_171787


namespace NUMINAMATH_CALUDE_problem_solution_l1717_171764

theorem problem_solution (x y : ℝ) 
  (h1 : x / 2 + 5 = 11) 
  (h2 : Real.sqrt y = x) : 
  x = 12 ∧ y = 144 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1717_171764


namespace NUMINAMATH_CALUDE_at_most_one_square_l1717_171749

theorem at_most_one_square (a : ℕ → ℤ) (h : ∀ n : ℕ, a (n + 1) = (a n)^3 + 1999) :
  ∃! n : ℕ, ∃ k : ℤ, a n = k^2 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_square_l1717_171749


namespace NUMINAMATH_CALUDE_cos_negative_300_degrees_l1717_171704

theorem cos_negative_300_degrees : Real.cos (-(300 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_300_degrees_l1717_171704


namespace NUMINAMATH_CALUDE_pancake_max_pieces_l1717_171753

/-- The maximum number of pieces a circle can be divided into with n straight cuts -/
def maxPieces (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- A round pancake can be divided into at most 7 pieces with three straight cuts -/
theorem pancake_max_pieces :
  maxPieces 3 = 7 :=
sorry

end NUMINAMATH_CALUDE_pancake_max_pieces_l1717_171753


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1717_171744

/-- A geometric sequence with a_m = 3 and a_{m+6} = 24 -/
def GeometricSequence (a : ℕ → ℝ) (m : ℕ) : Prop :=
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) ∧ 
  a m = 3 ∧ 
  a (m + 6) = 24

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) 
  (h : GeometricSequence a m) : 
  a (m + 18) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1717_171744


namespace NUMINAMATH_CALUDE_jungkook_balls_count_l1717_171797

/-- The number of boxes Jungkook has -/
def num_boxes : ℕ := 3

/-- The number of balls in each box -/
def balls_per_box : ℕ := 2

/-- The total number of balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_balls_count : total_balls = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_balls_count_l1717_171797


namespace NUMINAMATH_CALUDE_fourth_child_age_l1717_171772

theorem fourth_child_age (ages : Fin 4 → ℕ) 
  (avg_age : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (known_ages : ages 0 = 6 ∧ ages 1 = 8 ∧ ages 2 = 11) :
  ages 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_child_age_l1717_171772


namespace NUMINAMATH_CALUDE_optimal_playground_max_area_l1717_171719

/-- Represents a rectangular playground with given constraints -/
structure Playground where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 190
  length_constraint : length ≥ 100
  width_constraint : width ≥ 60

/-- The area of a playground -/
def area (p : Playground) : ℝ := p.length * p.width

/-- The optimal playground dimensions -/
def optimal_playground : Playground := {
  length := 100,
  width := 90,
  perimeter_constraint := by sorry,
  length_constraint := by sorry,
  width_constraint := by sorry
}

/-- Theorem stating that the optimal playground has the maximum area -/
theorem optimal_playground_max_area :
  ∀ p : Playground, area p ≤ area optimal_playground := by sorry

end NUMINAMATH_CALUDE_optimal_playground_max_area_l1717_171719


namespace NUMINAMATH_CALUDE_problem_solution_l1717_171750

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1717_171750


namespace NUMINAMATH_CALUDE_sum_of_f_values_l1717_171703

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 2/3 → f x + f ((x - 1) / (3 * x - 2)) = x

/-- The main theorem stating the sum of f(0), f(1), and f(2) -/
theorem sum_of_f_values (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  f 0 + f 1 + f 2 = 87/40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l1717_171703


namespace NUMINAMATH_CALUDE_function_constant_l1717_171799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) - Real.log (1 + x) + a

theorem function_constant (a : ℝ) :
  (∃ (M N : ℝ), (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f a x ≤ M ∧ N ≤ f a x) ∧ M + N = 1) →
  a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_constant_l1717_171799


namespace NUMINAMATH_CALUDE_inequality_proof_l1717_171754

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a - b / a > b - a / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1717_171754


namespace NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l1717_171793

/-- A function f: ℝ → ℝ is odd -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period p -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem min_zeros_odd_periodic_function
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_period : HasPeriod f 3)
  (h_f2 : f 2 = 0) :
  ∃ (S : Finset ℝ), S.card ≥ 7 ∧ (∀ x ∈ S, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l1717_171793


namespace NUMINAMATH_CALUDE_number_division_problem_l1717_171742

theorem number_division_problem (x : ℚ) : x / 5 = 80 + x / 6 → x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1717_171742


namespace NUMINAMATH_CALUDE_correct_number_of_values_l1717_171775

theorem correct_number_of_values 
  (original_mean : ℝ) 
  (incorrect_value : ℝ) 
  (correct_value : ℝ) 
  (correct_mean : ℝ) 
  (h1 : original_mean = 190) 
  (h2 : incorrect_value = 130) 
  (h3 : correct_value = 165) 
  (h4 : correct_mean = 191.4) : 
  ∃ n : ℕ, n > 0 ∧ 
    n * original_mean + (correct_value - incorrect_value) = n * correct_mean ∧ 
    n = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_values_l1717_171775


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1717_171734

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 3*x - 5 = 2) : 
  2*x^2 + 6*x - 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1717_171734


namespace NUMINAMATH_CALUDE_circle_radius_l1717_171777

theorem circle_radius (max_distance min_distance : ℝ) 
  (h_max : max_distance = 11)
  (h_min : min_distance = 5) :
  ∃ (r : ℝ), (r = 3 ∨ r = 8) ∧ 
  ((max_distance - min_distance = 2 * r) ∨ (max_distance + min_distance = 4 * r)) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l1717_171777


namespace NUMINAMATH_CALUDE_tiffany_green_buckets_l1717_171701

/-- Carnival ring toss game -/
structure CarnivalGame where
  total_money : ℕ
  cost_per_play : ℕ
  rings_per_play : ℕ
  red_bucket_points : ℕ
  green_bucket_points : ℕ
  games_played : ℕ
  red_buckets_hit : ℕ
  total_points : ℕ

/-- Calculate the number of green buckets hit -/
def green_buckets_hit (game : CarnivalGame) : ℕ :=
  (game.total_points - game.red_buckets_hit * game.red_bucket_points) / game.green_bucket_points

/-- Theorem: Tiffany hit 10 green buckets -/
theorem tiffany_green_buckets :
  let game : CarnivalGame := {
    total_money := 3,
    cost_per_play := 1,
    rings_per_play := 5,
    red_bucket_points := 2,
    green_bucket_points := 3,
    games_played := 2,
    red_buckets_hit := 4,
    total_points := 38
  }
  green_buckets_hit game = 10 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_green_buckets_l1717_171701


namespace NUMINAMATH_CALUDE_aquarium_length_l1717_171707

/-- The length of an aquarium given its volume, breadth, and water height -/
theorem aquarium_length (volume : ℝ) (breadth height : ℝ) (h1 : volume = 10000)
  (h2 : breadth = 20) (h3 : height = 10) : volume / (breadth * height) = 50 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_length_l1717_171707


namespace NUMINAMATH_CALUDE_certain_number_value_l1717_171730

theorem certain_number_value : ∃! x : ℝ,
  (28 + x + 42 + 78 + 104) / 5 = 90 ∧
  (128 + 255 + 511 + 1023 + x) / 5 = 423 ∧
  x = 198 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1717_171730


namespace NUMINAMATH_CALUDE_total_meals_sold_l1717_171785

/-- Given the ratio of kids meals to adult meals to seniors' meals and the number of kids meals sold,
    calculate the total number of meals sold. -/
theorem total_meals_sold (kids_ratio adult_ratio seniors_ratio kids_meals : ℕ) : 
  kids_ratio > 0 → 
  adult_ratio > 0 → 
  seniors_ratio > 0 → 
  kids_ratio = 3 → 
  adult_ratio = 2 → 
  seniors_ratio = 1 → 
  kids_meals = 12 → 
  kids_meals + (adult_ratio * kids_meals / kids_ratio) + (seniors_ratio * kids_meals / kids_ratio) = 24 := by
sorry

end NUMINAMATH_CALUDE_total_meals_sold_l1717_171785


namespace NUMINAMATH_CALUDE_gemma_change_is_five_l1717_171708

-- Define the given conditions
def number_of_pizzas : ℕ := 4
def price_per_pizza : ℕ := 10
def tip_amount : ℕ := 5
def payment_amount : ℕ := 50

-- Define the function to calculate the change
def calculate_change (pizzas : ℕ) (price : ℕ) (tip : ℕ) (payment : ℕ) : ℕ :=
  payment - (pizzas * price + tip)

-- Theorem statement
theorem gemma_change_is_five :
  calculate_change number_of_pizzas price_per_pizza tip_amount payment_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_gemma_change_is_five_l1717_171708


namespace NUMINAMATH_CALUDE_ladybugs_without_spots_l1717_171722

theorem ladybugs_without_spots (total : Nat) (with_spots : Nat) (without_spots : Nat) : 
  total = 67082 → with_spots = 12170 → without_spots = total - with_spots → without_spots = 54912 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_without_spots_l1717_171722


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l1717_171733

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 6) :
  ∃ (min : ℝ), min = 16 ∧ ∀ (a b : ℝ), a + 2 * b = 6 → 2^a + 4^b ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l1717_171733


namespace NUMINAMATH_CALUDE_area_of_M_l1717_171786

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (abs y + abs (4 + y) ≤ 4) ∧
               ((x - y^2 - 4*y - 3) / (2*y - x + 3) ≥ 0) ∧
               (-4 ≤ y) ∧ (y ≤ 0)}

-- Define the area function
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_M : area M = 8 := by sorry

end NUMINAMATH_CALUDE_area_of_M_l1717_171786


namespace NUMINAMATH_CALUDE_mrs_franklin_valentines_l1717_171765

theorem mrs_franklin_valentines (given_away : ℕ) (left : ℕ) 
  (h1 : given_away = 42) (h2 : left = 16) : 
  given_away + left = 58 := by
  sorry

end NUMINAMATH_CALUDE_mrs_franklin_valentines_l1717_171765


namespace NUMINAMATH_CALUDE_triangle_prime_angles_l1717_171710

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem triangle_prime_angles 
  (a b c : ℕ) 
  (sum_180 : a + b + c = 180) 
  (all_prime : is_prime a ∧ is_prime b ∧ is_prime c) 
  (all_less_120 : a < 120 ∧ b < 120 ∧ c < 120) : 
  ((a = 2 ∧ b = 71 ∧ c = 107) ∨ (a = 2 ∧ b = 89 ∧ c = 89)) ∨
  ((a = 71 ∧ b = 2 ∧ c = 107) ∨ (a = 89 ∧ b = 2 ∧ c = 89)) ∨
  ((a = 71 ∧ b = 107 ∧ c = 2) ∨ (a = 89 ∧ b = 89 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_prime_angles_l1717_171710


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l1717_171780

theorem circle_equation_m_range (m : ℝ) :
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0 ↔ (x - 1)^2 + (y - 2)^2 = r^2) →
  m < 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l1717_171780


namespace NUMINAMATH_CALUDE_wall_tiling_impossible_l1717_171700

/-- Represents the dimensions of a rectangular cuboid -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if a smaller cuboid can tile a larger cuboid -/
def can_tile (wall : Dimensions) (brick : Dimensions) : Prop :=
  ∃ (a b c : ℕ), 
    (a * brick.length = wall.length ∧ 
     b * brick.width = wall.width ∧ 
     c * brick.height = wall.height) ∨
    (a * brick.length = wall.length ∧ 
     b * brick.width = wall.height ∧ 
     c * brick.height = wall.width) ∨
    (a * brick.length = wall.width ∧ 
     b * brick.width = wall.length ∧ 
     c * brick.height = wall.height) ∨
    (a * brick.length = wall.width ∧ 
     b * brick.width = wall.height ∧ 
     c * brick.height = wall.length) ∨
    (a * brick.length = wall.height ∧ 
     b * brick.width = wall.length ∧ 
     c * brick.height = wall.width) ∨
    (a * brick.length = wall.height ∧ 
     b * brick.width = wall.width ∧ 
     c * brick.height = wall.length)

theorem wall_tiling_impossible (wall : Dimensions) 
  (brick1 : Dimensions) (brick2 : Dimensions) : 
  wall.length = 27 ∧ wall.width = 16 ∧ wall.height = 15 →
  brick1.length = 3 ∧ brick1.width = 5 ∧ brick1.height = 7 →
  brick2.length = 2 ∧ brick2.width = 5 ∧ brick2.height = 6 →
  ¬(can_tile wall brick1 ∨ can_tile wall brick2) :=
sorry

end NUMINAMATH_CALUDE_wall_tiling_impossible_l1717_171700


namespace NUMINAMATH_CALUDE_two_students_per_section_l1717_171758

/-- Represents a school bus with a given number of rows and total capacity. -/
structure SchoolBus where
  rows : ℕ
  capacity : ℕ

/-- Calculates the number of students allowed per section in a school bus. -/
def studentsPerSection (bus : SchoolBus) : ℚ :=
  bus.capacity / (2 * bus.rows)

/-- Theorem stating that for a bus with 13 rows and capacity of 52 students,
    the number of students per section is 2. -/
theorem two_students_per_section :
  let bus : SchoolBus := { rows := 13, capacity := 52 }
  studentsPerSection bus = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_students_per_section_l1717_171758


namespace NUMINAMATH_CALUDE_mask_price_problem_l1717_171712

theorem mask_price_problem (first_total second_total : ℚ) 
  (price_increase : ℚ) (quantity_increase : ℕ) :
  first_total = 500000 →
  second_total = 770000 →
  price_increase = 1.4 →
  quantity_increase = 10000 →
  ∃ (first_price first_quantity : ℚ),
    first_price * first_quantity = first_total ∧
    price_increase * first_price * (first_quantity + quantity_increase) = second_total ∧
    first_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_mask_price_problem_l1717_171712


namespace NUMINAMATH_CALUDE_total_rice_weight_l1717_171763

-- Define the number of containers
def num_containers : ℕ := 4

-- Define the weight of rice in each container (in ounces)
def rice_per_container : ℚ := 25

-- Define the conversion rate from ounces to pounds
def ounces_per_pound : ℚ := 16

-- Theorem to prove
theorem total_rice_weight :
  (num_containers : ℚ) * rice_per_container / ounces_per_pound = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_total_rice_weight_l1717_171763


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l1717_171789

theorem fraction_equality_solution :
  ∀ m n : ℕ+, 
  (m : ℚ) / ((n : ℚ) + m) = (n : ℚ) / ((n : ℚ) - m) →
  (∃ h : ℕ, m = (2*h + 1)*h ∧ n = (2*h + 1)*(h + 1)) ∨
  (∃ h : ℕ+, m = 2*h*(4*h^2 - 1) ∧ n = 2*h*(4*h^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l1717_171789


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l1717_171795

theorem quadratic_root_k_value (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0 ∧ x = 7) → k = 119 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l1717_171795


namespace NUMINAMATH_CALUDE_number_of_diagonals_sum_of_interior_angles_l1717_171723

-- Define the number of sides
def n : ℕ := 150

-- Theorem for the number of diagonals
theorem number_of_diagonals : 
  n * (n - 3) / 2 = 11025 :=
sorry

-- Theorem for the sum of interior angles
theorem sum_of_interior_angles : 
  180 * (n - 2) = 26640 :=
sorry

end NUMINAMATH_CALUDE_number_of_diagonals_sum_of_interior_angles_l1717_171723


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1717_171726

theorem complex_equation_solution (x : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (1 - 2*i) * (x + i) = 4 - 3*i) : 
  x = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1717_171726


namespace NUMINAMATH_CALUDE_max_value_implies_m_l1717_171716

/-- Given that the maximum value of f(x) = sin(x + π/2) + cos(x - π/2) + m is 2√2, prove that m = √2 -/
theorem max_value_implies_m (f : ℝ → ℝ) (m : ℝ) 
  (h : ∀ x, f x = Real.sin (x + π/2) + Real.cos (x - π/2) + m) 
  (h_max : ∃ x₀, ∀ x, f x ≤ f x₀ ∧ f x₀ = 2 * Real.sqrt 2) : 
  m = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l1717_171716


namespace NUMINAMATH_CALUDE_M_is_graph_of_square_function_l1717_171743

def M : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

theorem M_is_graph_of_square_function :
  M = {p : ℝ × ℝ | p.2 = p.1^2} := by sorry

end NUMINAMATH_CALUDE_M_is_graph_of_square_function_l1717_171743


namespace NUMINAMATH_CALUDE_vote_intersection_l1717_171759

theorem vote_intersection (U A B : Finset Nat) : 
  Finset.card U = 250 →
  Finset.card A = 172 →
  Finset.card B = 143 →
  Finset.card (U \ (A ∪ B)) = 37 →
  Finset.card (A ∩ B) = 102 := by
sorry

end NUMINAMATH_CALUDE_vote_intersection_l1717_171759


namespace NUMINAMATH_CALUDE_truncated_prism_edges_l1717_171781

/-- Represents a truncated rectangular prism -/
structure TruncatedPrism where
  originalEdges : ℕ
  normalTruncations : ℕ
  intersectingTruncations : ℕ

/-- Calculates the number of edges after truncation -/
def edgesAfterTruncation (p : TruncatedPrism) : ℕ :=
  p.originalEdges - p.intersectingTruncations +
  p.normalTruncations * 3 + p.intersectingTruncations * 4

/-- Theorem stating that the specific truncation scenario results in 33 edges -/
theorem truncated_prism_edges :
  ∀ p : TruncatedPrism,
  p.originalEdges = 12 ∧
  p.normalTruncations = 6 ∧
  p.intersectingTruncations = 1 →
  edgesAfterTruncation p = 33 :=
by
  sorry


end NUMINAMATH_CALUDE_truncated_prism_edges_l1717_171781


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1717_171748

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1717_171748


namespace NUMINAMATH_CALUDE_min_value_expression_l1717_171727

theorem min_value_expression (k n : ℝ) (h1 : k ≥ 0) (h2 : n ≥ 0) (h3 : 2 * k + n = 2) :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 2 * x + y = 2 → 2 * k^2 - 4 * n ≤ 2 * x^2 - 4 * y ∧
  ∃ k₀ n₀ : ℝ, k₀ ≥ 0 ∧ n₀ ≥ 0 ∧ 2 * k₀ + n₀ = 2 ∧ 2 * k₀^2 - 4 * n₀ = -8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1717_171727


namespace NUMINAMATH_CALUDE_sergeant_travel_distance_l1717_171747

/-- Proves that given an infantry column of length 1 km, if the infantry walks 4/3 km
    during the time it takes for someone to travel from the end to the beginning of
    the column and back at twice the speed of the infantry, then the total distance
    traveled by that person is 8/3 km. -/
theorem sergeant_travel_distance
  (column_length : ℝ)
  (infantry_distance : ℝ)
  (sergeant_speed_ratio : ℝ)
  (h1 : column_length = 1)
  (h2 : infantry_distance = 4/3)
  (h3 : sergeant_speed_ratio = 2) :
  2 * infantry_distance = 8/3 := by
  sorry

#check sergeant_travel_distance

end NUMINAMATH_CALUDE_sergeant_travel_distance_l1717_171747


namespace NUMINAMATH_CALUDE_first_month_sale_l1717_171735

theorem first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) 
  (h1 : sale2 = 6927)
  (h2 : sale3 = 6855)
  (h3 : sale4 = 7230)
  (h4 : sale5 = 6562)
  (h5 : sale6 = 7391)
  (h6 : average = 6900) :
  ∃ (sale1 : ℕ), 
    sale1 = 6435 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average :=
sorry

end NUMINAMATH_CALUDE_first_month_sale_l1717_171735


namespace NUMINAMATH_CALUDE_conditional_statement_else_branch_l1717_171770

/-- Represents a conditional statement structure -/
inductive ConditionalStatement
  | ifThenElse (condition : Prop) (thenBranch : Prop) (elseBranch : Prop)

/-- Represents the execution of a conditional statement -/
def executeConditional (stmt : ConditionalStatement) (conditionMet : Bool) : Prop :=
  match stmt with
  | ConditionalStatement.ifThenElse _ thenBranch elseBranch => 
      if conditionMet then thenBranch else elseBranch

theorem conditional_statement_else_branch 
  (stmt : ConditionalStatement) (conditionMet : Bool) :
  ¬conditionMet → 
  executeConditional stmt conditionMet = 
    match stmt with
    | ConditionalStatement.ifThenElse _ _ elseBranch => elseBranch :=
by
  sorry

end NUMINAMATH_CALUDE_conditional_statement_else_branch_l1717_171770


namespace NUMINAMATH_CALUDE_line_properties_l1717_171762

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + 2 * m * y + 6 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

/-- Two lines are parallel if their slopes are equal -/
def parallel (m : ℝ) : Prop := (-1 / (2 * m)) = (-(m - 2) / (3 * m))

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop := (-1 / (2 * m)) * (-(m - 2) / (3 * m)) = -1

theorem line_properties (m : ℝ) :
  (parallel m ↔ m = 0 ∨ m = 7/2) ∧
  (perpendicular m ↔ m = -1/2 ∨ m = 2/3) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1717_171762


namespace NUMINAMATH_CALUDE_quadratic_equation_set_l1717_171715

theorem quadratic_equation_set (a : ℝ) : 
  (∃! x, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_set_l1717_171715


namespace NUMINAMATH_CALUDE_problem_statement_l1717_171717

theorem problem_statement (w x y : ℝ) 
  (h1 : 6/w + 6/x = 6/y) 
  (h2 : w*x = y) 
  (h3 : (w + x)/2 = 0.5) : 
  y = 0.25 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1717_171717


namespace NUMINAMATH_CALUDE_five_solutions_l1717_171776

/-- The system of equations has exactly 5 distinct real solutions -/
theorem five_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ × ℝ)),
    Finset.card solutions = 5 ∧
    ∀ (x y z w : ℝ), (x, y, z, w) ∈ solutions ↔
      x = z + w + 2*z*w*x ∧
      y = w + x + w*x*y ∧
      z = x + y + x*y*z ∧
      w = y + z + 2*y*z*w := by
  sorry

end NUMINAMATH_CALUDE_five_solutions_l1717_171776


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1717_171736

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating that if a₁ + a₉ = 10 in an arithmetic sequence, then a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 1 + a 9 = 10) : 
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1717_171736


namespace NUMINAMATH_CALUDE_exponent_division_23_l1717_171757

theorem exponent_division_23 : (23 ^ 11) / (23 ^ 8) = 12167 := by sorry

end NUMINAMATH_CALUDE_exponent_division_23_l1717_171757


namespace NUMINAMATH_CALUDE_typist_salary_problem_l1717_171709

/-- Proves that if a salary S is increased by 10% and then decreased by 5%,
    resulting in Rs. 6270, then the original salary S was Rs. 6000. -/
theorem typist_salary_problem (S : ℝ) : 
  (S * 1.1 * 0.95 = 6270) → S = 6000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l1717_171709


namespace NUMINAMATH_CALUDE_x_20_digits_l1717_171783

theorem x_20_digits (x : ℝ) (h1 : x > 0) (h2 : 10^7 ≤ x^4) (h3 : x^5 < 10^9) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 := by
  sorry

end NUMINAMATH_CALUDE_x_20_digits_l1717_171783


namespace NUMINAMATH_CALUDE_min_h_12_l1717_171706

/-- A function h : ℕ+ → ℤ is quibbling if h(x) + h(y) ≥ x^2 + 10*y for all positive integers x and y -/
def IsQuibbling (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y ≥ x^2 + 10*y

/-- The sum of h(1) to h(15) -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem min_h_12 (h : ℕ+ → ℤ) (hQuib : IsQuibbling h) (hMin : ∀ g : ℕ+ → ℤ, IsQuibbling g → SumH g ≥ SumH h) :
  h ⟨12, by norm_num⟩ ≥ 144 := by
  sorry


end NUMINAMATH_CALUDE_min_h_12_l1717_171706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1717_171792

/-- An arithmetic sequence with first term 2 and the sum of the second and third terms equal to 13 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ 
  a 2 + a 3 = 13 ∧ 
  ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1717_171792


namespace NUMINAMATH_CALUDE_inequality_proof_l1717_171769

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1717_171769


namespace NUMINAMATH_CALUDE_count_triples_eq_30787_l1717_171705

/-- 
Counts the number of ordered triples (x,y,z) of non-negative integers 
satisfying x ≤ y ≤ z and x + y + z ≤ 100
-/
def count_triples : ℕ := 
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (x, y, z) := t
    x ≤ y ∧ y ≤ z ∧ x + y + z ≤ 100
  ) (Finset.product (Finset.range 101) (Finset.product (Finset.range 101) (Finset.range 101)))).card

theorem count_triples_eq_30787 : count_triples = 30787 := by
  sorry


end NUMINAMATH_CALUDE_count_triples_eq_30787_l1717_171705


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l1717_171784

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a sampling scenario -/
structure SamplingScenario where
  total_items : ℕ
  sample_size : ℕ
  is_homogeneous : Bool
  has_structure : Bool
  has_strata : Bool

/-- Determines the most appropriate sampling method for a given scenario -/
def appropriate_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.is_homogeneous then SamplingMethod.SimpleRandom
  else if scenario.has_structure then SamplingMethod.Systematic
  else if scenario.has_strata then SamplingMethod.Stratified
  else SamplingMethod.SimpleRandom

theorem sampling_methods_correct :
  (appropriate_sampling_method ⟨10, 3, true, false, false⟩ = SamplingMethod.SimpleRandom) ∧
  (appropriate_sampling_method ⟨1280, 32, false, true, false⟩ = SamplingMethod.Systematic) ∧
  (appropriate_sampling_method ⟨12, 50, false, false, true⟩ = SamplingMethod.Stratified) :=
by sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l1717_171784


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1717_171756

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum : a 2 + (a 1 + a 2 + a 3) = 0) : 
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1717_171756


namespace NUMINAMATH_CALUDE_smallest_product_l1717_171782

def digits : List Nat := [4, 5, 6, 7]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
    product a b c d ≥ 2622 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l1717_171782


namespace NUMINAMATH_CALUDE_one_third_of_six_y_plus_three_l1717_171760

theorem one_third_of_six_y_plus_three (y : ℝ) : (1 / 3) * (6 * y + 3) = 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_six_y_plus_three_l1717_171760


namespace NUMINAMATH_CALUDE_union_of_S_and_T_l1717_171745

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_S_and_T_l1717_171745


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1717_171766

theorem probability_of_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = 5)
  (h2 : red_balls = 3)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls : ℚ) / total_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1717_171766


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1717_171714

theorem geometric_sequence_sum (a₁ a₂ a₃ a₆ a₇ a₈ : ℚ) :
  a₁ = 4096 →
  a₂ = 1024 →
  a₃ = 256 →
  a₆ = 4 →
  a₇ = 1 →
  a₈ = 1/4 →
  ∃ r : ℚ, r ≠ 0 ∧
    (∀ n : ℕ, n ≥ 1 → a₁ * r^(n-1) = a₁ * (a₂ / a₁)^(n-1)) →
    a₄ + a₅ = 80 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1717_171714


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l1717_171728

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  m_valid : minutes < 60

def add_time_and_duration (t : Time) (d : Duration) : Time :=
  sorry

theorem sunset_time_calculation (sunrise : Time) (daylight : Duration) :
  sunrise.hours = 6 ∧ sunrise.minutes = 45 ∧
  daylight.hours = 11 ∧ daylight.minutes = 36 →
  let sunset := add_time_and_duration sunrise daylight
  sunset.hours = 18 ∧ sunset.minutes = 21 :=
sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l1717_171728


namespace NUMINAMATH_CALUDE_card_game_result_l1717_171752

/-- Represents the money distribution in a card game -/
structure CardGame where
  initialRatio : Fin 3 → ℕ
  finalRatio : Fin 3 → ℕ
  winAmount : ℕ

/-- Calculate the final amounts for each player -/
def finalAmounts (game : CardGame) : Fin 3 → ℕ :=
  fun i => game.finalRatio i * (game.winAmount * 90) / (game.finalRatio 0 + game.finalRatio 1 + game.finalRatio 2)

/-- The theorem statement -/
theorem card_game_result (game : CardGame) 
  (h_initial : game.initialRatio = ![7, 6, 5])
  (h_final : game.finalRatio = ![6, 5, 4])
  (h_win : game.winAmount = 1200) :
  finalAmounts game = ![43200, 36000, 28800] := by
  sorry

#eval finalAmounts { initialRatio := ![7, 6, 5], finalRatio := ![6, 5, 4], winAmount := 1200 }

end NUMINAMATH_CALUDE_card_game_result_l1717_171752


namespace NUMINAMATH_CALUDE_min_players_team_l1717_171725

theorem min_players_team (n : ℕ) : 
  (n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 7920 :=
by sorry

end NUMINAMATH_CALUDE_min_players_team_l1717_171725


namespace NUMINAMATH_CALUDE_base6_addition_sum_l1717_171755

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- Converts a Base6Digit to its natural number representation -/
def to_nat (d : Base6Digit) : Nat := d.val

/-- Represents the base-6 addition problem 5CD₆ + 32₆ = 61C₆ -/
def base6_addition_problem (C D : Base6Digit) : Prop :=
  (5 * 6 * 6 + to_nat C * 6 + to_nat D) + (3 * 6 + 2) = 
  (6 * 6 + 1 * 6 + to_nat C)

theorem base6_addition_sum (C D : Base6Digit) :
  base6_addition_problem C D → to_nat C + to_nat D = 6 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_sum_l1717_171755


namespace NUMINAMATH_CALUDE_square_plus_one_eq_empty_l1717_171779

theorem square_plus_one_eq_empty : {x : ℝ | x^2 + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_square_plus_one_eq_empty_l1717_171779


namespace NUMINAMATH_CALUDE_odds_against_third_horse_l1717_171711

/-- Represents the probability of a horse winning a race -/
def probability (p q : ℚ) : ℚ := q / (p + q)

/-- Given three horses in a race with no ties, calculates the odds against the third horse winning -/
theorem odds_against_third_horse 
  (prob_x prob_y : ℚ) 
  (hx : prob_x = probability 3 1) 
  (hy : prob_y = probability 2 3) 
  (h_sum : prob_x + prob_y < 1) :
  ∃ (p q : ℚ), p / q = 17 / 3 ∧ probability p q = 1 - prob_x - prob_y := by
sorry


end NUMINAMATH_CALUDE_odds_against_third_horse_l1717_171711


namespace NUMINAMATH_CALUDE_integral_of_f_l1717_171778

theorem integral_of_f (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 + 2 * ∫ x in (0:ℝ)..1, f x) → 
  ∫ x in (0:ℝ)..1, f x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_l1717_171778


namespace NUMINAMATH_CALUDE_marbles_started_with_l1717_171788

def marbles_bought : Real := 489.0
def total_marbles : Real := 2778.0

theorem marbles_started_with : total_marbles - marbles_bought = 2289.0 := by
  sorry

end NUMINAMATH_CALUDE_marbles_started_with_l1717_171788


namespace NUMINAMATH_CALUDE_max_total_pieces_l1717_171738

/-- Represents a chessboard configuration -/
structure ChessboardConfig where
  white_pieces : ℕ
  black_pieces : ℕ

/-- The size of the chessboard -/
def board_size : ℕ := 8

/-- Condition: In each row and column, the number of white pieces is twice the number of black pieces -/
def valid_distribution (config : ChessboardConfig) : Prop :=
  config.white_pieces = 2 * config.black_pieces

/-- The total number of pieces on the board -/
def total_pieces (config : ChessboardConfig) : ℕ :=
  config.white_pieces + config.black_pieces

/-- The maximum number of pieces that can be placed on the board -/
def max_pieces : ℕ := board_size * board_size

theorem max_total_pieces :
  ∃ (config : ChessboardConfig),
    valid_distribution config ∧
    (∀ (other : ChessboardConfig),
      valid_distribution other →
      total_pieces other ≤ total_pieces config) ∧
    total_pieces config = 48 :=
  sorry

end NUMINAMATH_CALUDE_max_total_pieces_l1717_171738


namespace NUMINAMATH_CALUDE_matthews_cakes_l1717_171790

theorem matthews_cakes (initial_crackers : ℕ) (friends : ℕ) (crackers_eaten : ℕ)
  (h1 : initial_crackers = 22)
  (h2 : friends = 11)
  (h3 : crackers_eaten = 2)
  (h4 : initial_crackers = friends * crackers_eaten) :
  ∃ initial_cakes : ℕ, initial_cakes = 22 ∧ initial_cakes = friends * crackers_eaten :=
by sorry

end NUMINAMATH_CALUDE_matthews_cakes_l1717_171790


namespace NUMINAMATH_CALUDE_student_sample_size_l1717_171761

/-- Represents the frequency distribution of student weights --/
structure WeightDistribution where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  remaining : ℕ

/-- The total number of students in the sample --/
def total_students (w : WeightDistribution) : ℕ :=
  w.group1 + w.group2 + w.group3 + w.remaining

/-- The given conditions for the weight distribution --/
def weight_distribution_conditions (w : WeightDistribution) : Prop :=
  w.group1 + w.group2 + w.group3 > 0 ∧
  w.group2 = 12 ∧
  w.group2 = 2 * w.group1 ∧
  w.group3 = 3 * w.group1

theorem student_sample_size :
  ∃ w : WeightDistribution, weight_distribution_conditions w ∧ total_students w = 48 :=
sorry

end NUMINAMATH_CALUDE_student_sample_size_l1717_171761


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1717_171739

theorem arithmetic_expression_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1717_171739


namespace NUMINAMATH_CALUDE_store_earnings_calculation_l1717_171773

/-- Represents the earnings calculation for a store selling bottled drinks -/
theorem store_earnings_calculation (cola_price juice_price water_price sports_price : ℚ)
                                   (cola_sold juice_sold water_sold sports_sold : ℕ) :
  cola_price = 3 →
  juice_price = 3/2 →
  water_price = 1 →
  sports_price = 5/2 →
  cola_sold = 18 →
  juice_sold = 15 →
  water_sold = 30 →
  sports_sold = 22 →
  cola_price * cola_sold + juice_price * juice_sold + 
  water_price * water_sold + sports_price * sports_sold = 161.5 := by
sorry

end NUMINAMATH_CALUDE_store_earnings_calculation_l1717_171773


namespace NUMINAMATH_CALUDE_tenth_prime_is_29_l1717_171796

/-- Definition of natural numbers -/
def NaturalNumber (n : ℕ) : Prop := n ≥ 0

/-- Definition of prime numbers -/
def PrimeNumber (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < p → (p % m = 0 → m = 1)

/-- Function to get the nth prime number -/
def nthPrime (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The 10th prime number is 29 -/
theorem tenth_prime_is_29 : nthPrime 10 = 29 := by
  sorry

end NUMINAMATH_CALUDE_tenth_prime_is_29_l1717_171796


namespace NUMINAMATH_CALUDE_expression_value_l1717_171718

theorem expression_value (x y : ℝ) (h : x - 2*y = 1) : 3 - 4*y + 2*x = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1717_171718


namespace NUMINAMATH_CALUDE_max_non_empty_intersection_l1717_171721

-- Define the set A_n
def A (n : ℕ) : Set ℝ := {x : ℝ | n < x^n ∧ x^n < n + 1}

-- Define the intersection of sets A_1 to A_n
def intersection_up_to (n : ℕ) : Set ℝ := ⋂ i ∈ Finset.range n, A (i + 1)

-- State the theorem
theorem max_non_empty_intersection :
  (∃ (n : ℕ), intersection_up_to n ≠ ∅ ∧
    ∀ (m : ℕ), m > n → intersection_up_to m = ∅) ∧
  (∀ (n : ℕ), intersection_up_to n ≠ ∅ → n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_non_empty_intersection_l1717_171721


namespace NUMINAMATH_CALUDE_odot_ten_five_l1717_171741

-- Define the ⊙ operation
def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

-- Theorem statement
theorem odot_ten_five : odot 10 5 = 38 / 3 := by
  sorry

end NUMINAMATH_CALUDE_odot_ten_five_l1717_171741


namespace NUMINAMATH_CALUDE_wig_cost_calculation_l1717_171740

theorem wig_cost_calculation (plays : ℕ) (acts_per_play : ℕ) (wigs_per_act : ℕ) (cost_per_wig : ℕ) :
  plays = 2 →
  acts_per_play = 5 →
  wigs_per_act = 2 →
  cost_per_wig = 5 →
  plays * acts_per_play * wigs_per_act * cost_per_wig = 100 :=
by sorry

end NUMINAMATH_CALUDE_wig_cost_calculation_l1717_171740


namespace NUMINAMATH_CALUDE_smallest_a_for_equation_l1717_171713

theorem smallest_a_for_equation : 
  ∀ a : ℕ, a ≥ 2 → 
  (∃ (p : ℕ) (b : ℕ), Prime p ∧ b ≥ 2 ∧ (a^p - a) / p = b^2) → 
  a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_equation_l1717_171713


namespace NUMINAMATH_CALUDE_difference_max_min_both_subjects_l1717_171791

/-- The number of students studying both Mathematics and Science --/
def studyingBoth (m s : ℕ) : ℕ := m + s - 3000

theorem difference_max_min_both_subjects : 
  ∃ (m_min m_max s_min s_max : ℕ),
    2100 ≤ m_min ∧ m_min ≤ m_max ∧ m_max ≤ 2250 ∧
    1050 ≤ s_min ∧ s_min ≤ s_max ∧ s_max ≤ 1350 ∧
    (∀ m s, 2100 ≤ m ∧ m ≤ 2250 ∧ 1050 ≤ s ∧ s ≤ 1350 →
      studyingBoth m_max s_max ≥ studyingBoth m s ∧
      studyingBoth m_min s_min ≤ studyingBoth m s) ∧
    studyingBoth m_max s_max - studyingBoth m_min s_min = 450 := by
  sorry

end NUMINAMATH_CALUDE_difference_max_min_both_subjects_l1717_171791


namespace NUMINAMATH_CALUDE_contribution_increase_l1717_171731

theorem contribution_increase (initial_contributions : ℕ) (initial_average : ℚ) (new_contribution : ℚ) :
  initial_contributions = 3 →
  initial_average = 75 →
  new_contribution = 150 →
  let total_initial := initial_contributions * initial_average
  let new_total := total_initial + new_contribution
  let new_average := new_total / (initial_contributions + 1)
  let increase := new_average - initial_average
  let percentage_increase := (increase / initial_average) * 100
  percentage_increase = 25 := by
  sorry

end NUMINAMATH_CALUDE_contribution_increase_l1717_171731
