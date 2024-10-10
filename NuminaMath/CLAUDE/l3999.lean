import Mathlib

namespace oranges_remaining_l3999_399972

def initial_oranges : ℕ := 60
def percentage_taken : ℚ := 45 / 100

theorem oranges_remaining : 
  initial_oranges - (percentage_taken * initial_oranges).floor = 33 := by
  sorry

end oranges_remaining_l3999_399972


namespace magnitude_of_parallel_vector_difference_l3999_399999

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of their difference is 2√5. -/
theorem magnitude_of_parallel_vector_difference :
  ∀ (x : ℝ), 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 6]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →  -- Parallel condition
  ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end magnitude_of_parallel_vector_difference_l3999_399999


namespace hannah_strawberries_l3999_399936

theorem hannah_strawberries (daily_harvest : ℕ) (days : ℕ) (stolen : ℕ) (remaining : ℕ) :
  daily_harvest = 5 →
  days = 30 →
  stolen = 30 →
  remaining = 100 →
  daily_harvest * days - stolen - remaining = 20 :=
by sorry

end hannah_strawberries_l3999_399936


namespace second_player_wins_l3999_399931

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a move in the game --/
inductive Move
  | Up : Nat → Move
  | Left : Nat → Move

/-- Applies a move to a position --/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.Up n => ⟨pos.x, pos.y + n⟩
  | Move.Left n => ⟨pos.x - n, pos.y⟩

/-- Checks if a position is valid on the 8x8 board --/
def isValidPosition (pos : Position) : Prop :=
  1 ≤ pos.x ∧ pos.x ≤ 8 ∧ 1 ≤ pos.y ∧ pos.y ≤ 8

/-- Checks if a move is valid from a given position --/
def isValidMove (pos : Position) (move : Move) : Prop :=
  isValidPosition (applyMove pos move)

/-- Represents the game state --/
structure GameState :=
  (position : Position)
  (currentPlayer : Bool)  -- True for first player, False for second player

/-- The winning strategy for the second player --/
def secondPlayerWinningStrategy : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.position = ⟨1, 1⟩ →
      initialState.currentPlayer = true →
      ∀ (game : ℕ → GameState),
        game 0 = initialState →
        (∀ n : ℕ, 
          (game (n+1)).position = 
            if (game n).currentPlayer
            then applyMove (game n).position (strategy (game n))
            else applyMove (game n).position (strategy (game n))) →
        ∃ (n : ℕ), ¬isValidMove (game n).position (strategy (game n))

theorem second_player_wins : secondPlayerWinningStrategy :=
  sorry

end second_player_wins_l3999_399931


namespace twenty_three_in_base_two_l3999_399915

theorem twenty_three_in_base_two : 23 = 1*2^4 + 0*2^3 + 1*2^2 + 1*2^1 + 1*2^0 := by
  sorry

end twenty_three_in_base_two_l3999_399915


namespace equation_solution_l3999_399947

theorem equation_solution (x : ℝ) : 
  Real.sqrt (x + 15) - 9 / Real.sqrt (x + 15) = 3 → x = 18 * Real.sqrt 5 / 4 - 6 := by
  sorry

end equation_solution_l3999_399947


namespace set_A_enumeration_l3999_399938

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_enumeration : A = {(-1, 0), (0, -1), (1, 0)} := by sorry

end set_A_enumeration_l3999_399938


namespace chore_division_proof_l3999_399926

/-- Time to sweep one room in minutes -/
def sweep_time_per_room : ℕ := 3

/-- Time to wash one dish in minutes -/
def wash_dish_time : ℕ := 2

/-- Number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- Number of laundry loads Billy does -/
def billy_laundry_loads : ℕ := 2

/-- Number of dishes Billy washes -/
def billy_dishes : ℕ := 6

/-- Time to do one load of laundry in minutes -/
def laundry_time : ℕ := 9

theorem chore_division_proof :
  anna_rooms * sweep_time_per_room = 
  billy_laundry_loads * laundry_time + billy_dishes * wash_dish_time :=
by sorry

end chore_division_proof_l3999_399926


namespace revenue_is_90_dollars_l3999_399987

def total_bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def bags_with_10_percent_rotten : ℕ := 4
def bags_with_20_percent_rotten : ℕ := 3
def bags_with_5_percent_rotten : ℕ := 3
def oranges_for_juice : ℕ := 70
def oranges_for_jams : ℕ := 15
def selling_price_per_orange : ℚ := 0.50

def total_oranges : ℕ := total_bags * oranges_per_bag

def rotten_oranges : ℕ := 
  bags_with_10_percent_rotten * oranges_per_bag / 10 +
  bags_with_20_percent_rotten * oranges_per_bag / 5 +
  bags_with_5_percent_rotten * oranges_per_bag / 20

def good_oranges : ℕ := total_oranges - rotten_oranges

def oranges_for_sale : ℕ := good_oranges - oranges_for_juice - oranges_for_jams

def total_revenue : ℚ := oranges_for_sale * selling_price_per_orange

theorem revenue_is_90_dollars : total_revenue = 90 := by
  sorry

end revenue_is_90_dollars_l3999_399987


namespace new_men_average_age_l3999_399969

/-- Given a group of 8 men, when two men aged 21 and 23 are replaced by two new men,
    and the average age of the group increases by 2 years,
    prove that the average age of the two new men is 30 years. -/
theorem new_men_average_age
  (initial_count : Nat)
  (replaced_age1 replaced_age2 : Nat)
  (age_increase : ℝ)
  (h_count : initial_count = 8)
  (h_replaced1 : replaced_age1 = 21)
  (h_replaced2 : replaced_age2 = 23)
  (h_increase : age_increase = 2)
  : (↑initial_count * age_increase + ↑replaced_age1 + ↑replaced_age2) / 2 = 30 :=
by sorry

end new_men_average_age_l3999_399969


namespace total_savings_is_150_l3999_399901

/-- Calculates the total savings for the year based on the given savings pattern. -/
def total_savings (savings_jan_to_jul : ℕ) (savings_aug_to_nov : ℕ) (savings_dec : ℕ) : ℕ :=
  7 * savings_jan_to_jul + 4 * savings_aug_to_nov + savings_dec

/-- Proves that the total savings for the year is $150 given the specified savings pattern. -/
theorem total_savings_is_150 :
  total_savings 10 15 20 = 150 := by sorry

end total_savings_is_150_l3999_399901


namespace factorial_squared_greater_than_power_l3999_399912

theorem factorial_squared_greater_than_power (n : ℕ) (h : n > 2) :
  (Nat.factorial n)^2 > n^n := by
  sorry

end factorial_squared_greater_than_power_l3999_399912


namespace arrangements_count_is_2880_l3999_399963

/-- The number of arrangements of 4 students and 3 teachers in a row,
    where exactly two teachers are standing next to each other. -/
def arrangements_count : ℕ :=
  let num_students : ℕ := 4
  let num_teachers : ℕ := 3
  let num_units : ℕ := num_students + 1  -- 4 students + 1 teacher pair
  let teacher_pair_permutations : ℕ := 2  -- 2! ways to arrange 2 teachers in a pair
  let remaining_teacher_positions : ℕ := num_students + 1  -- positions for the remaining teacher
  let teacher_pair_combinations : ℕ := 3  -- number of ways to choose 2 teachers out of 3
  (Nat.factorial num_units) * teacher_pair_permutations * remaining_teacher_positions * teacher_pair_combinations

/-- Theorem stating that the number of arrangements is 2880 -/
theorem arrangements_count_is_2880 : arrangements_count = 2880 := by
  sorry

end arrangements_count_is_2880_l3999_399963


namespace rationalize_denominator_l3999_399970

/-- Proves that the rationalization of 1/(√5 + √7 + √11) is equal to (-√5 - √7 + √11 + 2√385)/139 -/
theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) =
    (A * Real.sqrt 5 + B * Real.sqrt 7 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = -1 ∧
    B = -1 ∧
    C = 1 ∧
    D = 2 ∧
    E = 385 ∧
    F = 139 ∧
    F > 0 :=
by sorry

end rationalize_denominator_l3999_399970


namespace subset_implies_max_a_max_a_is_negative_three_l3999_399981

theorem subset_implies_max_a (a : ℝ) : 
  let A : Set ℝ := {x | |x| ≥ 3}
  let B : Set ℝ := {x | x ≥ a}
  A ⊆ B → a ≤ -3 :=
by
  sorry

theorem max_a_is_negative_three :
  ∃ c, c = -3 ∧ 
  (∀ a : ℝ, (let A : Set ℝ := {x | |x| ≥ 3}; let B : Set ℝ := {x | x ≥ a}; A ⊆ B) → a ≤ c) ∧
  (let A : Set ℝ := {x | |x| ≥ 3}; let B : Set ℝ := {x | x ≥ c}; A ⊆ B) :=
by
  sorry

end subset_implies_max_a_max_a_is_negative_three_l3999_399981


namespace second_largest_of_five_consecutive_odds_l3999_399965

theorem second_largest_of_five_consecutive_odds (a b c d e : ℕ) : 
  (∀ n : ℕ, n ∈ [a, b, c, d, e] → n % 2 = 1) →  -- all numbers are odd
  (b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2) →  -- consecutive
  a + b + c + d + e = 195 →  -- sum is 195
  d = 41 :=  -- 2nd largest (4th in sequence) is 41
by
  sorry

end second_largest_of_five_consecutive_odds_l3999_399965


namespace star_polygon_points_l3999_399957

/-- A regular star polygon with ℓ points -/
structure StarPolygon where
  ℓ : ℕ
  x_angle : Real
  y_angle : Real
  h_x_less_y : x_angle = y_angle - 15
  h_external_sum : ℓ * (x_angle + y_angle) = 360
  h_internal_sum : ℓ * (180 - x_angle - y_angle) = 2 * 360

/-- Theorem: The number of points in the star polygon is 24 -/
theorem star_polygon_points (s : StarPolygon) : s.ℓ = 24 := by
  sorry

end star_polygon_points_l3999_399957


namespace collinear_points_k_value_l3999_399935

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define basis vectors
variable (e₁ e₂ : V)

-- Define points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- State the theorem
theorem collinear_points_k_value
  (h_basis : LinearIndependent ℝ ![e₁, e₂])
  (h_AB : B - A = e₁ - k • e₂)
  (h_CB : B - C = 2 • e₁ - e₂)
  (h_CD : D - C = 3 • e₁ - 3 • e₂)
  (h_collinear : ∃ (t : ℝ), D - A = t • (B - A)) :
  k = 2 := by
  sorry

end collinear_points_k_value_l3999_399935


namespace petya_win_probability_l3999_399927

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initialStones : Nat
  maxTake : Nat
  minTake : Nat

/-- A player in the game -/
inductive Player
  | Petya
  | Computer

/-- The strategy used by a player -/
inductive Strategy
  | Random
  | Optimal

/-- The result of the game -/
inductive GameResult
  | PetyaWins
  | ComputerWins

/-- The probability of Petya winning the game -/
def winProbability (game : HeapOfStones) (firstPlayer : Player) 
    (petyaStrategy : Strategy) (computerStrategy : Strategy) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem petya_win_probability :
  let game : HeapOfStones := ⟨16, 4, 1⟩
  winProbability game Player.Petya Strategy.Random Strategy.Optimal = 1 / 256 :=
sorry

end petya_win_probability_l3999_399927


namespace max_product_of_functions_l3999_399923

/-- Given two real-valued functions f and g with specified ranges,
    prove that the maximum value of their product is 14 -/
theorem max_product_of_functions (f g : ℝ → ℝ)
  (hf : Set.range f = Set.Icc (-7) 4)
  (hg : Set.range g = Set.Icc 0 2) :
  ∃ x : ℝ, f x * g x = 14 ∧ ∀ y : ℝ, f y * g y ≤ 14 := by
  sorry


end max_product_of_functions_l3999_399923


namespace quadratic_real_roots_l3999_399933

/-- The quadratic equation (a-5)x^2 - 4x - 1 = 0 has real roots if and only if a ≥ 1 and a ≠ 5. -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a - 5) * x^2 - 4*x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by sorry

end quadratic_real_roots_l3999_399933


namespace cows_sold_l3999_399971

/-- The number of cows sold by a man last year, given the following conditions:
  * He initially had 39 cows
  * 25 cows died last year
  * The number of cows increased by 24 this year
  * He bought 43 more cows
  * His friend gave him 8 cows as a gift
  * He now has 83 cows -/
theorem cows_sold (initial : ℕ) (died : ℕ) (increased : ℕ) (bought : ℕ) (gifted : ℕ) (current : ℕ)
  (h_initial : initial = 39)
  (h_died : died = 25)
  (h_increased : increased = 24)
  (h_bought : bought = 43)
  (h_gifted : gifted = 8)
  (h_current : current = 83)
  (h_equation : current = initial - died - (initial - died - increased - bought - gifted)) :
  initial - died - increased - bought - gifted = 6 := by
  sorry

end cows_sold_l3999_399971


namespace remainder_product_mod_75_l3999_399960

theorem remainder_product_mod_75 : (3203 * 4507 * 9929) % 75 = 34 := by
  sorry

end remainder_product_mod_75_l3999_399960


namespace f_2_equals_5_l3999_399948

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem f_2_equals_5 : f 2 = 5 := by sorry

end f_2_equals_5_l3999_399948


namespace annas_lemonade_sales_l3999_399925

/-- Anna's lemonade sales problem -/
theorem annas_lemonade_sales 
  (plain_glasses : ℕ) 
  (plain_price : ℚ) 
  (plain_strawberry_difference : ℚ) 
  (h1 : plain_glasses = 36)
  (h2 : plain_price = 3/4)
  (h3 : plain_strawberry_difference = 11) :
  (plain_glasses : ℚ) * plain_price - plain_strawberry_difference = 16 := by
  sorry


end annas_lemonade_sales_l3999_399925


namespace sqrt_two_plus_sqrt_three_root_l3999_399906

theorem sqrt_two_plus_sqrt_three_root : ∃ x : ℝ, x = Real.sqrt 2 + Real.sqrt 3 ∧ x^4 - 10*x^2 + 1 = 0 := by
  sorry

end sqrt_two_plus_sqrt_three_root_l3999_399906


namespace infinitely_many_a_without_solution_l3999_399937

-- Define τ(n) as the number of positive divisors of n
def tau (n : ℕ+) : ℕ := sorry

-- Statement of the theorem
theorem infinitely_many_a_without_solution :
  ∃ (S : Set ℕ+), (Set.Infinite S) ∧ 
  (∀ (a : ℕ+), a ∈ S → ∀ (n : ℕ+), tau (a * n) ≠ n) :=
sorry

end infinitely_many_a_without_solution_l3999_399937


namespace sum_of_squares_of_coefficients_l3999_399994

def polynomial (x : ℝ) : ℝ := 4 * (x^4 + 3*x^2 + 1)

theorem sum_of_squares_of_coefficients :
  (4^2) + (12^2) + (4^2) = 176 :=
by sorry

end sum_of_squares_of_coefficients_l3999_399994


namespace unit_digit_of_7_to_500_l3999_399959

theorem unit_digit_of_7_to_500 : 7^500 % 10 = 1 := by
  sorry

end unit_digit_of_7_to_500_l3999_399959


namespace max_candy_leftover_l3999_399962

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 11 * q + r ∧ r ≤ 10 ∧ ∀ (r' : ℕ), x = 11 * q + r' → r' ≤ r := by
  sorry

end max_candy_leftover_l3999_399962


namespace monotonically_decreasing_cubic_l3999_399943

/-- A function f is monotonically decreasing on an interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem monotonically_decreasing_cubic (a d : ℝ) :
  MonotonicallyDecreasing (fun x => x^3 - a*x^2 + 4*d) 0 2 → a ≥ 3 := by
  sorry

end monotonically_decreasing_cubic_l3999_399943


namespace triangle_equal_area_division_l3999_399982

theorem triangle_equal_area_division :
  let triangle := [(0, 0), (1, 1), (9, 1)]
  let total_area := 4
  let dividing_line := 3
  let left_area := (1/2) * dividing_line * (dividing_line/9)
  let right_area := (1/2) * (1 - dividing_line/9) * (9 - dividing_line)
  left_area = right_area ∧ left_area = total_area/2 := by sorry

end triangle_equal_area_division_l3999_399982


namespace min_sum_squares_l3999_399954

theorem min_sum_squares (x y z : ℝ) (h : 2*x + y + 2*z = 6) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), 2*a + b + 2*c = 6 → a^2 + b^2 + c^2 ≥ m :=
sorry

end min_sum_squares_l3999_399954


namespace distance_between_trees_l3999_399949

def yard_length : ℝ := 300
def num_trees : ℕ := 26

theorem distance_between_trees :
  let num_intervals : ℕ := num_trees - 1
  let distance : ℝ := yard_length / num_intervals
  distance = 12 := by sorry

end distance_between_trees_l3999_399949


namespace condition_sufficient_not_necessary_l3999_399903

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem condition_sufficient_not_necessary (a : ℕ → ℝ) :
  (∀ n, a (n + 1) > |a n|) → is_increasing a ∧
  ¬(is_increasing a → ∀ n, a (n + 1) > |a n|) :=
by
  sorry

end condition_sufficient_not_necessary_l3999_399903


namespace two_possible_values_l3999_399920

def triangle (a b : ℕ) : ℕ := min a b

def nabla (a b : ℕ) : ℕ := max a b

theorem two_possible_values (x : ℕ) : 
  ∃ (s : Finset ℕ), (s.card = 2) ∧ 
  (triangle 6 (nabla 4 (triangle x 5)) ∈ s) := by
  sorry

end two_possible_values_l3999_399920


namespace largest_fraction_l3999_399978

theorem largest_fraction : (5 : ℚ) / 6 > 3 / 4 ∧ (5 : ℚ) / 6 > 4 / 5 := by
  sorry

end largest_fraction_l3999_399978


namespace sum_of_digits_5N_plus_2013_l3999_399984

/-- Sum of digits function in base 10 -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The smallest positive integer with sum of digits 2013 -/
def N : ℕ := sorry

/-- Theorem stating the sum of digits of (5N + 2013) is 18 -/
theorem sum_of_digits_5N_plus_2013 :
  sum_of_digits (5 * N + 2013) = 18 ∧ 
  sum_of_digits N = 2013 ∧
  ∀ m : ℕ, m < N → sum_of_digits m ≠ 2013 := by sorry

end sum_of_digits_5N_plus_2013_l3999_399984


namespace remainder_fifty_pow_2019_plus_one_mod_seven_l3999_399996

theorem remainder_fifty_pow_2019_plus_one_mod_seven (n : ℕ) : (50^2019 + 1) % 7 = 2 := by
  sorry

end remainder_fifty_pow_2019_plus_one_mod_seven_l3999_399996


namespace number_difference_proof_l3999_399930

theorem number_difference_proof (x : ℚ) : x - (3/5) * x = 50 → x = 125 := by
  sorry

end number_difference_proof_l3999_399930


namespace quadratic_roots_inequality_l3999_399907

theorem quadratic_roots_inequality (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - a*x₁ + a = 0 → x₂^2 - a*x₂ + a = 0 → x₁ ≠ x₂ → x₁^2 + x₂^2 ≥ 2*(x₁ + x₂) := by
  sorry

end quadratic_roots_inequality_l3999_399907


namespace product_evaluation_l3999_399950

theorem product_evaluation (n : ℕ) (h : n = 2) :
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 120 := by
  sorry

end product_evaluation_l3999_399950


namespace quadratic_intersects_x_axis_l3999_399902

/-- The quadratic function y = (k-1)x^2 + 2x - 1 intersects the x-axis if and only if k ≥ 0 and k ≠ 1 -/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) := by
  sorry

end quadratic_intersects_x_axis_l3999_399902


namespace train_length_l3999_399942

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 50 →
  man_speed = 5 →
  passing_time = 7.2 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 :=
by
  sorry

#check train_length

end train_length_l3999_399942


namespace imaginary_part_of_complex_fraction_l3999_399988

theorem imaginary_part_of_complex_fraction : Complex.im ((2 + 4 * Complex.I) / (1 + Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l3999_399988


namespace ab_length_in_two_isosceles_triangles_l3999_399993

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2 * t.leg

theorem ab_length_in_two_isosceles_triangles 
  (abc cde : IsoscelesTriangle)
  (h1 : perimeter cde = 22)
  (h2 : perimeter abc = 24)
  (h3 : cde.base = 8)
  (h4 : abc.leg = cde.leg) : 
  abc.base = 10 := by sorry

end ab_length_in_two_isosceles_triangles_l3999_399993


namespace unique_negative_zero_l3999_399921

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- State the theorem
theorem unique_negative_zero (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) ↔ a > 3/2 := by sorry

end unique_negative_zero_l3999_399921


namespace square_difference_equals_400_l3999_399956

theorem square_difference_equals_400 : (25 + 8)^2 - (8^2 + 25^2) = 400 := by
  sorry

end square_difference_equals_400_l3999_399956


namespace art_museum_exhibits_l3999_399908

/-- The number of exhibits in an art museum --/
def num_exhibits : ℕ := 4

/-- The number of pictures the museum currently has --/
def current_pictures : ℕ := 15

/-- The number of additional pictures needed for equal distribution --/
def additional_pictures : ℕ := 1

theorem art_museum_exhibits :
  (current_pictures + additional_pictures) % num_exhibits = 0 ∧
  current_pictures % num_exhibits ≠ 0 ∧
  num_exhibits > 1 :=
sorry

end art_museum_exhibits_l3999_399908


namespace intersection_line_equation_l3999_399961

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the line passing through the intersection points of two circles --/
def intersectionLine (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = -143/117

/-- The main theorem statement --/
theorem intersection_line_equation (c1 c2 : Circle) 
  (h1 : c1 = ⟨(-5, -6), 10⟩) 
  (h2 : c2 = ⟨(4, 7), Real.sqrt 85⟩) : 
  ∀ x y, (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧ 
         (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2 → 
  intersectionLine c1 c2 x y := by
  sorry

end intersection_line_equation_l3999_399961


namespace line_passes_through_fixed_point_l3999_399900

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p (1, 0) = |p.1 + 1|}

-- Define the property of line l intersecting C at M and N
def intersects_at_MN (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop :=
  M ∈ C ∧ N ∈ C ∧ M ∈ l ∧ N ∈ l ∧ M ≠ N ∧ M ≠ (0, 0) ∧ N ≠ (0, 0)

-- Define the perpendicularity of OM and ON
def OM_perp_ON (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

-- Theorem statement
theorem line_passes_through_fixed_point :
  ∀ l : Set (ℝ × ℝ), ∀ M N : ℝ × ℝ,
  intersects_at_MN l M N → OM_perp_ON M N →
  (4, 0) ∈ l :=
sorry

end line_passes_through_fixed_point_l3999_399900


namespace diameter_endpoint_coordinates_l3999_399928

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space --/
def Point := ℝ × ℝ

/-- Checks if two points are endpoints of a diameter in a circle --/
def are_diameter_endpoints (c : Circle) (p1 p2 : Point) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = c.center

theorem diameter_endpoint_coordinates (c : Circle) (p1 p2 : Point) :
  c.center = (3, 4) →
  p1 = (1, -2) →
  are_diameter_endpoints c p1 p2 →
  p2 = (5, 10) := by
  sorry

end diameter_endpoint_coordinates_l3999_399928


namespace missing_figure_proof_l3999_399918

theorem missing_figure_proof (x : ℝ) (h : (0.50 / 100) * x = 0.12) : x = 24 := by
  sorry

end missing_figure_proof_l3999_399918


namespace bowling_ball_weight_proof_l3999_399917

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 35

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 28

theorem bowling_ball_weight_proof :
  (5 * bowling_ball_weight = 4 * kayak_weight) →
  bowling_ball_weight = 28 := by
  sorry

end bowling_ball_weight_proof_l3999_399917


namespace electricity_gasoline_ratio_l3999_399964

theorem electricity_gasoline_ratio (total : ℕ) (both : ℕ) (gas_only : ℕ) (neither : ℕ)
  (h_total : total = 300)
  (h_both : both = 120)
  (h_gas_only : gas_only = 60)
  (h_neither : neither = 24)
  (h_sum : total = both + gas_only + (total - both - gas_only - neither) + neither) :
  (total - both - gas_only - neither) / neither = 4 := by
sorry

end electricity_gasoline_ratio_l3999_399964


namespace sandwich_problem_solution_l3999_399910

/-- Represents the sandwich shop problem -/
def sandwich_problem (sandwich_price : ℝ) (delivery_fee : ℝ) (tip_percentage : ℝ) (total_received : ℝ) : Prop :=
  ∃ (num_sandwiches : ℝ),
    sandwich_price * num_sandwiches + delivery_fee + 
    (sandwich_price * num_sandwiches + delivery_fee) * tip_percentage = total_received ∧
    num_sandwiches = 18

/-- Theorem stating the solution to the sandwich problem -/
theorem sandwich_problem_solution :
  sandwich_problem 5 20 0.1 121 := by
  sorry

end sandwich_problem_solution_l3999_399910


namespace chinese_chess_probability_l3999_399998

/-- The probability of player A winning a game of Chinese chess -/
def prob_A_win : ℝ := 0.2

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := 0.5

/-- The probability of player B winning a game of Chinese chess -/
def prob_B_win : ℝ := 1 - (prob_A_win + prob_draw)

theorem chinese_chess_probability :
  prob_B_win = 0.3 := by sorry

end chinese_chess_probability_l3999_399998


namespace distance_probability_l3999_399945

-- Define the points and distances
def A : ℝ × ℝ := (0, -10)
def B : ℝ × ℝ := (0, 0)
def AB : ℝ := 10
def BC : ℝ := 6
def AC_max : ℝ := 8

-- Define the angle range
def angle_range : Set ℝ := Set.Ioo 0 Real.pi

-- Define the probability function
noncomputable def probability_AC_less_than_8 : ℝ :=
  (30 : ℝ) / 180

-- State the theorem
theorem distance_probability :
  probability_AC_less_than_8 = 1/6 := by sorry

end distance_probability_l3999_399945


namespace number_problem_l3999_399924

theorem number_problem : ∃ x : ℝ, x + 3 * x = 20 ∧ x = 5 := by sorry

end number_problem_l3999_399924


namespace max_b_over_a_l3999_399990

theorem max_b_over_a (a b : ℝ) (h_a : a > 0) : 
  (∀ x : ℝ, a * Real.exp x ≥ 2 * x + b) → b / a ≤ 1 := by
  sorry

end max_b_over_a_l3999_399990


namespace mushroom_collection_l3999_399955

theorem mushroom_collection (total_mushrooms : ℕ) (h1 : total_mushrooms = 289) :
  ∃ (num_children : ℕ) (mushrooms_per_child : ℕ),
    num_children > 0 ∧
    mushrooms_per_child > 0 ∧
    num_children * mushrooms_per_child = total_mushrooms ∧
    num_children = 17 := by
  sorry

end mushroom_collection_l3999_399955


namespace jack_classic_authors_l3999_399980

/-- The number of books each classic author has in Jack's collection -/
def books_per_author : ℕ := 33

/-- The total number of books in Jack's classics section -/
def total_books : ℕ := 198

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := total_books / books_per_author

theorem jack_classic_authors :
  num_authors = 6 :=
by sorry

end jack_classic_authors_l3999_399980


namespace students_passed_l3999_399941

theorem students_passed (total : ℕ) (failure_rate : ℚ) : 
  total = 1000 → failure_rate = 0.4 → (total : ℚ) * (1 - failure_rate) = 600 := by
  sorry

end students_passed_l3999_399941


namespace pine_cone_weight_l3999_399958

/-- The weight of each pine cone given the conditions in Alan's backyard scenario -/
theorem pine_cone_weight (
  trees : ℕ)
  (cones_per_tree : ℕ)
  (roof_percentage : ℚ)
  (total_roof_weight : ℕ)
  (h1 : trees = 8)
  (h2 : cones_per_tree = 200)
  (h3 : roof_percentage = 3/10)
  (h4 : total_roof_weight = 1920)
  : (total_roof_weight : ℚ) / ((trees * cones_per_tree : ℕ) * roof_percentage) = 4 := by
  sorry

end pine_cone_weight_l3999_399958


namespace adult_elephant_weekly_bananas_eq_630_l3999_399983

/-- The number of bananas an adult elephant eats per day -/
def adult_elephant_daily_bananas : ℕ := 90

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of bananas an adult elephant eats in a week -/
def adult_elephant_weekly_bananas : ℕ := adult_elephant_daily_bananas * days_in_week

theorem adult_elephant_weekly_bananas_eq_630 :
  adult_elephant_weekly_bananas = 630 := by
  sorry

end adult_elephant_weekly_bananas_eq_630_l3999_399983


namespace cylinder_height_l3999_399986

/-- A cylinder with base diameter equal to height and volume 16π has height 4 -/
theorem cylinder_height (r h : ℝ) (h_positive : 0 < h) (r_positive : 0 < r) : 
  h = 2 * r → π * r^2 * h = 16 * π → h = 4 := by
  sorry

end cylinder_height_l3999_399986


namespace cistern_filling_problem_l3999_399914

/-- The time taken for pipe A to fill the cistern -/
def time_A : ℝ := 16

/-- The time taken for pipe B to empty the cistern -/
def time_B : ℝ := 20

/-- The time taken to fill the cistern when both pipes are open -/
def time_both : ℝ := 80

/-- Theorem stating that the given times satisfy the cistern filling problem -/
theorem cistern_filling_problem :
  1 / time_A - 1 / time_B = 1 / time_both := by sorry

end cistern_filling_problem_l3999_399914


namespace determine_english_marks_l3999_399939

/-- Represents a student's marks in 5 subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculate the average marks -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

/-- Theorem: Given 4 subject marks and the average, the 5th subject mark is uniquely determined -/
theorem determine_english_marks (marks : StudentMarks) (avg : ℚ) 
    (h1 : marks.mathematics = 65)
    (h2 : marks.physics = 82)
    (h3 : marks.chemistry = 67)
    (h4 : marks.biology = 85)
    (h5 : average marks = avg)
    (h6 : avg = 75) :
  marks.english = 76 := by
  sorry


end determine_english_marks_l3999_399939


namespace fourth_to_third_grade_ratio_l3999_399992

/-- Given the number of students in each grade, prove the ratio of 4th to 3rd grade students -/
theorem fourth_to_third_grade_ratio 
  (third_grade : ℕ) 
  (second_grade : ℕ) 
  (total_students : ℕ) 
  (h1 : third_grade = 19) 
  (h2 : second_grade = 29) 
  (h3 : total_students = 86) :
  (total_students - second_grade - third_grade) / third_grade = 2 := by
sorry

end fourth_to_third_grade_ratio_l3999_399992


namespace stratified_sampling_most_appropriate_l3999_399976

/-- Represents the different sampling methods --/
inductive SamplingMethod
  | Lottery
  | Systematic
  | Stratified
  | RandomNumber

/-- Represents a grade level --/
inductive Grade
  | Third
  | Sixth
  | Ninth

/-- Represents the characteristics of the sampling problem --/
structure SamplingProblem where
  grades : List Grade
  proportionalSampling : Bool
  distinctGroups : Bool

/-- Determines the most appropriate sampling method for a given problem --/
def mostAppropriateMethod (problem : SamplingProblem) : SamplingMethod :=
  if problem.distinctGroups && problem.proportionalSampling then
    SamplingMethod.Stratified
  else
    SamplingMethod.Lottery  -- Default to Lottery for simplicity

/-- The specific problem described in the question --/
def schoolEyesightProblem : SamplingProblem :=
  { grades := [Grade.Third, Grade.Sixth, Grade.Ninth]
  , proportionalSampling := true
  , distinctGroups := true }

theorem stratified_sampling_most_appropriate :
  mostAppropriateMethod schoolEyesightProblem = SamplingMethod.Stratified := by
  sorry


end stratified_sampling_most_appropriate_l3999_399976


namespace horner_first_step_value_l3999_399968

/-- Horner's Method first step for polynomial evaluation -/
def horner_first_step (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  a 4 * x + a 3

/-- Polynomial coefficients -/
def f_coeff : ℕ → ℝ
  | 4 => 3
  | 3 => 0
  | 2 => 2
  | 1 => 1
  | 0 => 4
  | _ => 0

theorem horner_first_step_value :
  horner_first_step f_coeff 10 = 30 := by sorry

end horner_first_step_value_l3999_399968


namespace expression_simplification_l3999_399991

theorem expression_simplification (w v : ℝ) :
  3 * w + 5 * w + 7 * v + 9 * w + 11 * v + 15 = 17 * w + 18 * v + 15 := by
  sorry

end expression_simplification_l3999_399991


namespace system_solution_l3999_399975

theorem system_solution (x y : ℝ) (k n : ℤ) : 
  (2 * (Real.cos x)^2 - 2 * Real.sqrt 2 * Real.cos x * (Real.cos (8 * x))^2 + (Real.cos (8 * x))^2 = 0 ∧
   Real.sin x = Real.cos y) ↔ 
  ((x = π/4 + 2*π*↑k ∧ (y = π/4 + 2*π*↑n ∨ y = -π/4 + 2*π*↑n)) ∨
   (x = -π/4 + 2*π*↑k ∧ (y = 3*π/4 + 2*π*↑n ∨ y = -3*π/4 + 2*π*↑n))) := by
sorry

end system_solution_l3999_399975


namespace direct_proportion_problem_l3999_399922

/-- A direct proportion function -/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

theorem direct_proportion_problem (f : ℝ → ℝ) 
  (h1 : DirectProportion f) 
  (h2 : f (-2) = 4) : 
  f 3 = -6 := by
  sorry

end direct_proportion_problem_l3999_399922


namespace parabola_intercept_minimum_l3999_399966

/-- Parabola defined by x^2 = 8y -/
def Parabola (x y : ℝ) : Prop := x^2 = 8*y

/-- Line with slope k passing through point (x, y) -/
def Line (k x y : ℝ) : Prop := y = k*x + 2

/-- Length of line segment intercepted by the parabola for a line with slope k -/
def InterceptLength (k : ℝ) : ℝ := 8*k^2 + 8

/-- The condition given in the problem relating k1 and k2 -/
def SlopeCondition (k1 k2 : ℝ) : Prop := 1/k1^2 + 4/k2^2 = 1

theorem parabola_intercept_minimum :
  ∀ k1 k2 : ℝ, 
  SlopeCondition k1 k2 →
  InterceptLength k1 + InterceptLength k2 ≥ 88 :=
sorry

end parabola_intercept_minimum_l3999_399966


namespace soda_price_proof_l3999_399995

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.85

/-- The total price for 100 cans purchased in 24-can cases -/
def total_price : ℝ := 34

theorem soda_price_proof :
  discounted_price * 100 = total_price ∧ regular_price = 0.40 :=
sorry

end soda_price_proof_l3999_399995


namespace candy_distribution_l3999_399916

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 15 →
  num_bags = 5 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 3 :=
by sorry

end candy_distribution_l3999_399916


namespace binary_sum_equals_136_l3999_399979

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary1 : List Bool := [true, false, true, false, true, false, true]
def binary2 : List Bool := [true, true, false, false, true, true]

theorem binary_sum_equals_136 :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 136 := by
  sorry

end binary_sum_equals_136_l3999_399979


namespace sum_of_solutions_quadratic_l3999_399905

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (63 - 21*x - x^2 = 0) → 
  (∃ r s : ℝ, (63 - 21*r - r^2 = 0) ∧ (63 - 21*s - s^2 = 0) ∧ (r + s = -21)) :=
by sorry

end sum_of_solutions_quadratic_l3999_399905


namespace min_cos_B_angle_A_values_l3999_399944

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.c = 6 * Real.sqrt 3 ∧ t.b = 6

-- Theorem for the minimum value of cos B
theorem min_cos_B (t : Triangle) (h : triangle_conditions t) :
  ∃ (min_cos_B : ℝ), min_cos_B = 1/3 ∧ ∀ (cos_B : ℝ), cos_B = (t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c) → cos_B ≥ min_cos_B :=
sorry

-- Theorem for the possible values of angle A
theorem angle_A_values (t : Triangle) (h1 : triangle_conditions t) (h2 : t.a * t.b * Real.cos t.C = 12) :
  t.A = π/2 ∨ t.A = π/6 :=
sorry

end min_cos_B_angle_A_values_l3999_399944


namespace primitive_points_polynomial_theorem_l3999_399973

/-- A primitive point is an ordered pair of integers with greatest common divisor 1. -/
def PrimitivePoint : Type := { p : ℤ × ℤ // Int.gcd p.1 p.2 = 1 }

/-- The theorem statement -/
theorem primitive_points_polynomial_theorem (S : Finset PrimitivePoint) :
  ∃ (n : ℕ+) (a : Fin (n + 1) → ℤ),
    ∀ (p : PrimitivePoint), p ∈ S →
      (Finset.range (n + 1)).sum (fun i => a i * p.val.1^(n - i) * p.val.2^i) = 1 := by
  sorry

end primitive_points_polynomial_theorem_l3999_399973


namespace x_investment_value_l3999_399913

/-- Represents the investment and profit scenario of a business partnership --/
structure BusinessPartnership where
  x_investment : ℕ  -- X's investment
  y_investment : ℕ  -- Y's investment
  z_investment : ℕ  -- Z's investment
  total_profit : ℕ  -- Total profit
  z_profit : ℕ      -- Z's share of the profit
  x_months : ℕ      -- Months X and Y were in business before Z joined
  z_months : ℕ      -- Months Z was in business

/-- The main theorem stating that X's investment was 35700 given the conditions --/
theorem x_investment_value (bp : BusinessPartnership) : 
  bp.y_investment = 42000 ∧ 
  bp.z_investment = 48000 ∧ 
  bp.total_profit = 14300 ∧ 
  bp.z_profit = 4160 ∧
  bp.x_months = 12 ∧
  bp.z_months = 8 →
  bp.x_investment = 35700 := by
  sorry

#check x_investment_value

end x_investment_value_l3999_399913


namespace cost_price_of_ball_l3999_399934

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls_sold = 17)
  (h3 : num_balls_loss = 5) : 
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧ 
    cost_price = 60 := by
  sorry

end cost_price_of_ball_l3999_399934


namespace jacob_winning_strategy_l3999_399929

/-- Represents the game board --/
structure Board :=
  (m : ℕ) -- number of rows
  (n : ℕ) -- number of columns

/-- Represents a position on the board --/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Defines a valid move on the board --/
def ValidMove (b : Board) (start finish : Position) : Prop :=
  (finish.row ≥ start.row ∧ finish.col = start.col) ∨
  (finish.col ≥ start.col ∧ finish.row = start.row)

/-- Defines the winning position --/
def IsWinningPosition (b : Board) (p : Position) : Prop :=
  p.row = b.m ∧ p.col = b.n

/-- Jacob's winning strategy exists --/
def JacobHasWinningStrategy (b : Board) : Prop :=
  ∃ (strategy : Position → Position),
    ∀ (p : Position),
      ValidMove b p (strategy p) ∧
      (∀ (q : Position), ValidMove b (strategy p) q →
        (IsWinningPosition b q ∨ ∃ (r : Position), ValidMove b q r ∧ IsWinningPosition b (strategy r)))

/-- The main theorem: Jacob has a winning strategy iff m ≠ n --/
theorem jacob_winning_strategy (b : Board) :
  JacobHasWinningStrategy b ↔ b.m ≠ b.n :=
sorry

end jacob_winning_strategy_l3999_399929


namespace distance_between_vertices_l3999_399967

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 5
def g (x : ℝ) : ℝ := x^2 + 6*x + 13

-- Define the vertices of the two parabolas
def vertex_f : ℝ × ℝ := (2, f 2)
def vertex_g : ℝ × ℝ := (-3, g (-3))

-- State the theorem
theorem distance_between_vertices : 
  Real.sqrt ((vertex_f.1 - vertex_g.1)^2 + (vertex_f.2 - vertex_g.2)^2) = Real.sqrt 34 :=
by sorry

end distance_between_vertices_l3999_399967


namespace plane_perpendicularity_l3999_399951

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularP : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : parallel m n) 
  (h4 : parallelLP m α) 
  (h5 : perpendicular n β) : 
  perpendicularP α β := by sorry

end plane_perpendicularity_l3999_399951


namespace monotonic_increasing_condition_l3999_399953

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f a x ≤ f a y) →
  a ≥ -1 :=
by sorry

end monotonic_increasing_condition_l3999_399953


namespace equation_one_solutions_l3999_399952

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by sorry

end equation_one_solutions_l3999_399952


namespace pamphlet_cost_is_correct_l3999_399989

/-- The cost of one pamphlet in dollars -/
def pamphlet_cost : ℝ := 1.11

/-- Condition 1: Nine copies cost less than $10.00 -/
axiom condition1 : 9 * pamphlet_cost < 10

/-- Condition 2: Ten copies cost more than $11.00 -/
axiom condition2 : 10 * pamphlet_cost > 11

/-- Theorem: The cost of one pamphlet is $1.11 -/
theorem pamphlet_cost_is_correct : pamphlet_cost = 1.11 := by
  sorry


end pamphlet_cost_is_correct_l3999_399989


namespace no_common_root_l3999_399919

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x₀ : ℝ, (x₀^2 + b*x₀ + c = 0) ∧ (x₀^2 + a*x₀ + d = 0) := by
  sorry

end no_common_root_l3999_399919


namespace students_in_two_classes_l3999_399904

theorem students_in_two_classes
  (total_students : ℕ)
  (history_students : ℕ)
  (math_students : ℕ)
  (english_students : ℕ)
  (all_three_classes : ℕ)
  (h_total : total_students = 68)
  (h_history : history_students = 19)
  (h_math : math_students = 14)
  (h_english : english_students = 26)
  (h_all_three : all_three_classes = 3)
  (h_at_least_one : total_students = history_students + math_students + english_students
    - (history_students + math_students - all_three_classes
    + history_students + english_students - all_three_classes
    + math_students + english_students - all_three_classes)
    + all_three_classes) :
  history_students + math_students - all_three_classes
  + history_students + english_students - all_three_classes
  + math_students + english_students - all_three_classes
  - 3 * all_three_classes = 6 :=
sorry

end students_in_two_classes_l3999_399904


namespace chord_length_problem_l3999_399940

/-- The chord length cut by a line on a circle -/
def chord_length (circle_center : ℝ × ℝ) (circle_radius : ℝ) (line : ℝ → ℝ → ℝ) : ℝ :=
  2 * circle_radius

/-- The problem statement -/
theorem chord_length_problem :
  let circle_center := (3, 0)
  let circle_radius := 3
  let line := fun x y => 3 * x - 4 * y - 9
  chord_length circle_center circle_radius line = 6 := by
  sorry


end chord_length_problem_l3999_399940


namespace six_digit_nondecreasing_remainder_l3999_399977

theorem six_digit_nondecreasing_remainder (n : Nat) (k : Nat) : 
  n = 6 → k = 9 → (Nat.choose (n + k - 1) n) % 1000 = 3 := by
  sorry

end six_digit_nondecreasing_remainder_l3999_399977


namespace equation_solutions_l3999_399997

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 6*x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ 2*x*(x-1) = 3-3*x
  let sol1 : Set ℝ := {3 + Real.sqrt 6, 3 - Real.sqrt 6}
  let sol2 : Set ℝ := {1, -3/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y, eq1 y → y ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y, eq2 y → y ∈ sol2) :=
by sorry


end equation_solutions_l3999_399997


namespace distribute_5_4_l3999_399909

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_4 : distribute 5 4 = 1024 := by
  sorry

end distribute_5_4_l3999_399909


namespace not_product_of_consecutive_numbers_l3999_399946

theorem not_product_of_consecutive_numbers (n k : ℕ) :
  ¬ ∃ x : ℕ, x * (x + 1) = 2 * n^(3*k) + 4 * n^k + 10 := by
  sorry

end not_product_of_consecutive_numbers_l3999_399946


namespace landscape_breadth_l3999_399932

/-- Represents a rectangular landscape with specific features -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ
  walking_path_ratio : ℝ
  water_body_ratio : ℝ

/-- Theorem stating the breadth of the landscape given specific conditions -/
theorem landscape_breadth (l : Landscape) 
  (h1 : l.breadth = 8 * l.length)
  (h2 : l.playground_area = 3200)
  (h3 : l.playground_area = (l.length * l.breadth) / 9)
  (h4 : l.walking_path_ratio = 1 / 18)
  (h5 : l.water_body_ratio = 1 / 6)
  : l.breadth = 480 := by
  sorry

end landscape_breadth_l3999_399932


namespace bicycle_distance_l3999_399974

/-- Given a bicycle traveling b/2 feet in t seconds, prove it travels 50b/t yards in 5 minutes -/
theorem bicycle_distance (b t : ℝ) (h : b > 0) (h' : t > 0) : 
  (b / 2) / t * (5 * 60) / 3 = 50 * b / t := by
  sorry

end bicycle_distance_l3999_399974


namespace mass_equivalence_l3999_399911

-- Define symbols as real numbers representing their masses
variable (circle square triangle zero : ℝ)

-- Define the balanced scales conditions
axiom scale1 : 3 * circle = 2 * triangle
axiom scale2 : square + circle + triangle = 2 * square

-- Define the mass of the left side of the equation to prove
def left_side : ℝ := circle + 3 * triangle

-- Define the mass of the right side of the equation to prove
def right_side : ℝ := 3 * zero + square

-- Theorem to prove
theorem mass_equivalence : left_side = right_side :=
sorry

end mass_equivalence_l3999_399911


namespace x_fourth_plus_reciprocal_l3999_399985

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end x_fourth_plus_reciprocal_l3999_399985
