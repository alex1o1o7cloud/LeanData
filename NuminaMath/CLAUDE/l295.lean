import Mathlib

namespace intersection_of_A_and_B_l295_29519

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l295_29519


namespace initial_ratio_proof_l295_29531

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (numerator : ℚ)
  (denominator : ℚ)

/-- Represents the contents of a bucket with two liquids -/
structure Bucket :=
  (liquidA : ℚ)
  (liquidB : ℚ)

def replace_mixture (b : Bucket) (amount : ℚ) : Bucket :=
  { liquidA := b.liquidA,
    liquidB := b.liquidB + amount }

def ratio (b : Bucket) : Ratio :=
  { numerator := b.liquidA,
    denominator := b.liquidB }

theorem initial_ratio_proof (initial : Bucket) 
  (h1 : initial.liquidA = 21)
  (h2 : ratio (replace_mixture initial 9) = Ratio.mk 7 9) :
  ratio initial = Ratio.mk 7 6 := by
  sorry

end initial_ratio_proof_l295_29531


namespace dave_walking_probability_l295_29586

/-- Represents the number of gates in the airport terminal -/
def total_gates : ℕ := 15

/-- Represents the number of gates Dave can be assigned to -/
def dave_gates : ℕ := 10

/-- Represents the distance between adjacent gates in feet -/
def gate_distance : ℕ := 100

/-- Represents the maximum walking distance in feet -/
def max_walk_distance : ℕ := 300

/-- Calculates the number of valid gate combinations for Dave's initial and new gates -/
def total_combinations : ℕ := dave_gates * (dave_gates - 1)

/-- Calculates the number of valid gate combinations where Dave walks 300 feet or less -/
def valid_combinations : ℕ := 58

/-- The probability of Dave walking 300 feet or fewer to his new gate -/
def probability : ℚ := valid_combinations / total_combinations

theorem dave_walking_probability :
  probability = 29 / 45 := by sorry

end dave_walking_probability_l295_29586


namespace polynomial_value_l295_29523

theorem polynomial_value (a b : ℝ) : 
  (a * 2^3 + b * 2 + 3 = 5) → 
  (a * (-2)^2 - 1/2 * b * (-2) - 3 = -2) :=
by sorry

end polynomial_value_l295_29523


namespace total_points_l295_29524

def game1_mike : ℕ := 5
def game1_john : ℕ := game1_mike + 2

def game2_mike : ℕ := 7
def game2_john : ℕ := game2_mike - 3

def game3_mike : ℕ := 10
def game3_john : ℕ := game3_mike / 2

def game4_mike : ℕ := 12
def game4_john : ℕ := game4_mike * 2

def game5_mike : ℕ := 6
def game5_john : ℕ := game5_mike

def game6_john : ℕ := 8
def game6_mike : ℕ := game6_john + 4

def mike_total : ℕ := game1_mike + game2_mike + game3_mike + game4_mike + game5_mike + game6_mike
def john_total : ℕ := game1_john + game2_john + game3_john + game4_john + game5_john + game6_john

theorem total_points : mike_total + john_total = 106 := by
  sorry

end total_points_l295_29524


namespace jordans_rectangle_width_l295_29588

-- Define the rectangle type
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the area function for rectangles
def area (r : Rectangle) : ℝ := r.length * r.width

-- State the theorem
theorem jordans_rectangle_width 
  (carol_rect : Rectangle)
  (jordan_rect : Rectangle)
  (h1 : carol_rect.length = 5)
  (h2 : carol_rect.width = 24)
  (h3 : jordan_rect.length = 2)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 60 := by
  sorry

end jordans_rectangle_width_l295_29588


namespace rectangle_dimension_l295_29518

/-- A rectangle with vertices at (0, 0), (0, 6), (x, 6), and (x, 0) has a perimeter of 40 units. -/
def rectangle_perimeter (x : ℝ) : Prop :=
  x > 0 ∧ 2 * (x + 6) = 40

/-- The value of x for which the rectangle has a perimeter of 40 units is 14. -/
theorem rectangle_dimension : ∃ x : ℝ, rectangle_perimeter x ∧ x = 14 := by
  sorry

end rectangle_dimension_l295_29518


namespace dice_roll_probability_l295_29593

-- Define a dice roll
def DiceRoll : Type := Fin 6

-- Define a point as a pair of dice rolls
def Point : Type := DiceRoll × DiceRoll

-- Define the condition for a point to be inside the circle
def InsideCircle (p : Point) : Prop :=
  (p.1.val + 1)^2 + (p.2.val + 1)^2 < 17

-- Define the total number of possible outcomes
def TotalOutcomes : Nat := 36

-- Define the number of favorable outcomes
def FavorableOutcomes : Nat := 8

-- Theorem statement
theorem dice_roll_probability :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 2 / 9 :=
sorry

end dice_roll_probability_l295_29593


namespace staircase_extension_l295_29506

/-- Calculates the number of toothpicks needed for a staircase of n steps -/
def toothpicks (n : ℕ) : ℕ := 
  if n = 0 then 0
  else if n = 1 then 4
  else 4 + (n - 1) * 3 + ((n - 1) * (n - 2)) / 2

/-- The number of additional toothpicks needed to extend an n-step staircase to an m-step staircase -/
def additional_toothpicks (n m : ℕ) : ℕ := toothpicks m - toothpicks n

theorem staircase_extension :
  additional_toothpicks 3 6 = 36 :=
sorry

end staircase_extension_l295_29506


namespace not_divisible_and_only_prime_l295_29565

theorem not_divisible_and_only_prime (n : ℕ) : 
  (n > 1 → ¬(n ∣ (2^n - 1))) ∧ 
  (n.Prime ∧ n^2 ∣ (2^n + 1) ↔ n = 3) := by
  sorry

end not_divisible_and_only_prime_l295_29565


namespace quadratic_roots_sum_and_product_l295_29558

theorem quadratic_roots_sum_and_product (x₁ x₂ : ℝ) :
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → x₁ + x₂ + x₁*x₂ = 4 := by
  sorry

end quadratic_roots_sum_and_product_l295_29558


namespace number_of_paths_in_grid_l295_29579

-- Define the grid dimensions
def grid_width : ℕ := 7
def grid_height : ℕ := 6

-- Define the total number of steps
def total_steps : ℕ := grid_width + grid_height

-- Theorem statement
theorem number_of_paths_in_grid : 
  (Nat.choose total_steps grid_height : ℕ) = 1716 := by
  sorry

end number_of_paths_in_grid_l295_29579


namespace solve_for_y_l295_29591

theorem solve_for_y (x y : ℤ) (h1 : x + y = 290) (h2 : x - y = 200) : y = 45 := by
  sorry

end solve_for_y_l295_29591


namespace percentage_difference_l295_29540

theorem percentage_difference (x y : ℝ) (h : x = 18 * y) :
  (x - y) / x * 100 = 94.44 := by
  sorry

end percentage_difference_l295_29540


namespace max_quarters_sasha_l295_29546

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Represents the total amount Sasha has in dollars -/
def total_amount : ℚ := 480 / 100

theorem max_quarters_sasha : 
  ∀ q : ℕ, 
    (q : ℚ) * quarter_value + 
    (2 * q : ℚ) * nickel_value + 
    (q : ℚ) * dime_value ≤ total_amount → 
    q ≤ 10 := by
  sorry

end max_quarters_sasha_l295_29546


namespace fish_distribution_l295_29526

theorem fish_distribution (bodies_of_water : ℕ) (total_fish : ℕ) 
  (h1 : bodies_of_water = 6) 
  (h2 : total_fish = 1050) : 
  total_fish / bodies_of_water = 175 := by
sorry

end fish_distribution_l295_29526


namespace multiplication_table_odd_fraction_l295_29549

theorem multiplication_table_odd_fraction :
  let n : ℕ := 16
  let total_products : ℕ := n * n
  let odd_numbers : ℕ := (n + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 :=
by sorry

end multiplication_table_odd_fraction_l295_29549


namespace arithmetic_sequence_a4_l295_29597

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_a4 (seq : ArithmeticSequence) 
    (h1 : seq.S 6 = 24) (h2 : seq.S 9 = 63) : seq.a 4 = 5 := by
  sorry

end arithmetic_sequence_a4_l295_29597


namespace quadratic_form_equivalence_l295_29511

theorem quadratic_form_equivalence (y : ℝ) : y^2 - 8*y = (y - 4)^2 - 16 := by
  sorry

end quadratic_form_equivalence_l295_29511


namespace raisin_problem_l295_29533

theorem raisin_problem (x : ℕ) : 
  (x / 3 : ℚ) + 4 + ((2 * x / 3 - 4) / 2 : ℚ) + 16 = x → x = 54 := by
  sorry

end raisin_problem_l295_29533


namespace tangent_point_coordinates_l295_29573

theorem tangent_point_coordinates (x y : ℝ) :
  y = x^2 →  -- curve equation
  (2 * x = -3) →  -- slope condition
  (x = -3/2 ∧ y = 9/4)  -- coordinates of point P
  := by sorry

end tangent_point_coordinates_l295_29573


namespace existence_of_special_multiple_l295_29552

theorem existence_of_special_multiple : ∃ n : ℕ,
  (n % 2020 = 0) ∧
  (∀ d : Fin 10, ∃! pos : ℕ, 
    (n / 10^pos % 10 : Fin 10) = d) :=
sorry

end existence_of_special_multiple_l295_29552


namespace probability_divisor_of_twelve_l295_29520

def divisors_of_twelve : Finset ℕ := {1, 2, 3, 4, 6, 12}

theorem probability_divisor_of_twelve (die : Finset ℕ) 
  (h1 : die = Finset.range 12) 
  (h2 : die.card = 12) : 
  (divisors_of_twelve.card : ℚ) / (die.card : ℚ) = 1/2 := by
  sorry

end probability_divisor_of_twelve_l295_29520


namespace impossible_average_weight_problem_l295_29537

theorem impossible_average_weight_problem :
  ¬ ∃ (n : ℕ), n > 0 ∧ (n * 55 + 50) / (n + 1) = 50 := by
  sorry

end impossible_average_weight_problem_l295_29537


namespace trigonometric_identities_l295_29529

theorem trigonometric_identities :
  (∃ (x y : Real),
    x = Real.tan (20 * π / 180) ∧
    y = Real.tan (40 * π / 180) ∧
    x + y + Real.sqrt 3 * x * y = Real.sqrt 3) ∧
  (∃ (z w : Real),
    z = Real.sin (50 * π / 180) ∧
    w = Real.tan (10 * π / 180) ∧
    z * (1 + Real.sqrt 3 * w) = 1) := by
  sorry

end trigonometric_identities_l295_29529


namespace inequality_proof_l295_29507

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 3) : 
  (x + 3) / (y + z) + (y + 3) / (z + x) + (z + 3) / (x + y) + 3 ≥ 
  27 * ((Real.sqrt x + Real.sqrt y + Real.sqrt z)^2) / ((x + y + z)^3) := by
  sorry

end inequality_proof_l295_29507


namespace parabola_vertex_l295_29516

/-- The parabola defined by the equation y = 2(x-3)^2 - 7 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 7

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 - 7 is at the point (3, -7) -/
theorem parabola_vertex : 
  ∃ (v : Vertex), v.x = 3 ∧ v.y = -7 ∧ 
  (∀ (x y : ℝ), parabola x y → (x - v.x)^2 ≤ (y - v.y) / 2) :=
sorry

end parabola_vertex_l295_29516


namespace linda_needs_two_more_batches_l295_29517

/-- The number of additional batches of cookies Linda needs to bake --/
def additional_batches (classmates : ℕ) (cookies_per_student : ℕ) (dozens_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : ℕ :=
  let total_cookies_needed := classmates * cookies_per_student
  let cookies_per_batch := dozens_per_batch * 12
  let cookies_already_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let cookies_still_needed := total_cookies_needed - cookies_already_made
  (cookies_still_needed + cookies_per_batch - 1) / cookies_per_batch

/-- Theorem stating that Linda needs to bake 2 more batches of cookies --/
theorem linda_needs_two_more_batches :
  additional_batches 24 10 4 2 1 = 2 := by
  sorry

end linda_needs_two_more_batches_l295_29517


namespace arthur_dinner_cost_theorem_l295_29574

/-- Calculates the total cost of Arthur's dinner, including tips --/
def arthurDinnerCost (appetizer_cost dessert_cost entree_cost wine_cost : ℝ)
  (entree_discount appetizer_discount dessert_discount bill_discount tax_rate waiter_tip_rate busser_tip_rate : ℝ) : ℝ :=
  let discounted_entree := entree_cost * (1 - entree_discount)
  let subtotal := discounted_entree + 2 * wine_cost
  let discounted_subtotal := subtotal * (1 - bill_discount)
  let tax := discounted_subtotal * tax_rate
  let total_with_tax := discounted_subtotal + tax
  let original_cost := appetizer_cost + entree_cost + 2 * wine_cost + dessert_cost
  let original_with_tax := original_cost * (1 + tax_rate)
  let waiter_tip := original_with_tax * waiter_tip_rate
  let total_with_waiter_tip := total_with_tax + waiter_tip
  let busser_tip := total_with_waiter_tip * busser_tip_rate
  total_with_waiter_tip + busser_tip

/-- Theorem stating that Arthur's dinner cost is $38.556 --/
theorem arthur_dinner_cost_theorem :
  arthurDinnerCost 8 7 30 4 0.4 1 1 0.1 0.08 0.2 0.05 = 38.556 := by
  sorry


end arthur_dinner_cost_theorem_l295_29574


namespace square_mod_32_l295_29535

theorem square_mod_32 (n : ℕ) (h : n % 8 = 6) : n^2 % 32 = 4 := by
  sorry

end square_mod_32_l295_29535


namespace solution_sets_l295_29528

def f (a x : ℝ) : ℝ := x^2 - 3*a*x + 2*a^2

theorem solution_sets :
  (∀ x, f 1 x ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) ∧
  (∀ a, (a = 0 → ∀ x, ¬(f a x < 0)) ∧
        (a > 0 → ∀ x, f a x < 0 ↔ a < x ∧ x < 2*a) ∧
        (a < 0 → ∀ x, f a x < 0 ↔ 2*a < x ∧ x < a)) :=
by sorry

end solution_sets_l295_29528


namespace double_counted_integer_l295_29525

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem double_counted_integer (n : ℕ) (x : ℕ) :
  sum_of_first_n n + x = 5053 → x = 3 := by
  sorry

end double_counted_integer_l295_29525


namespace students_playing_soccer_l295_29548

-- Define the total number of students
def total_students : ℕ := 450

-- Define the number of boys
def boys : ℕ := 320

-- Define the percentage of boys playing soccer
def boys_soccer_percentage : ℚ := 86 / 100

-- Define the number of girls not playing soccer
def girls_not_soccer : ℕ := 95

-- Theorem to prove
theorem students_playing_soccer : 
  ∃ (soccer_players : ℕ), 
    soccer_players = 250 ∧ 
    soccer_players ≤ total_students ∧
    (total_students - boys) - girls_not_soccer = 
      (1 - boys_soccer_percentage) * soccer_players :=
sorry

end students_playing_soccer_l295_29548


namespace sum_of_abc_l295_29564

theorem sum_of_abc (a b c : ℕ+) (h1 : a * b + c = 31)
                   (h2 : b * c + a = 31) (h3 : a * c + b = 31) :
  (a : ℕ) + b + c = 32 := by
  sorry

end sum_of_abc_l295_29564


namespace max_value_sqrt_x2_y2_l295_29599

theorem max_value_sqrt_x2_y2 (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 6 * x) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x' y' : ℝ), 3 * x'^2 + 2 * y'^2 = 6 * x' → Real.sqrt (x'^2 + y'^2) ≤ max :=
by sorry

end max_value_sqrt_x2_y2_l295_29599


namespace oliver_workout_total_l295_29572

/-- Oliver's workout schedule over four days -/
def workout_schedule (monday tuesday wednesday thursday : ℕ) : Prop :=
  monday = 4 ∧ 
  tuesday = monday - 2 ∧ 
  wednesday = 2 * monday ∧ 
  thursday = 2 * tuesday

/-- The total workout hours over four days -/
def total_hours (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem: Given Oliver's workout schedule, the total hours worked out is 18 -/
theorem oliver_workout_total :
  ∀ (monday tuesday wednesday thursday : ℕ),
  workout_schedule monday tuesday wednesday thursday →
  total_hours monday tuesday wednesday thursday = 18 :=
by
  sorry

end oliver_workout_total_l295_29572


namespace trinomial_binomial_product_l295_29538

theorem trinomial_binomial_product : 
  ∀ x : ℝ, (2 * x^2 + 3 * x + 1) * (x - 4) = 2 * x^3 - 5 * x^2 - 11 * x - 4 := by
  sorry

end trinomial_binomial_product_l295_29538


namespace eva_process_terminates_l295_29589

/-- Represents a deck of cards -/
def Deck := List Nat

/-- Flips the first n cards in the deck -/
def flipCards (n : Nat) (deck : Deck) : Deck :=
  (deck.take n).reverse ++ deck.drop n

/-- Performs one step of Eva's operation -/
def evaStep (deck : Deck) : Deck :=
  match deck with
  | [] => []
  | k :: rest => flipCards k deck

/-- Predicate to check if the process has terminated -/
def isTerminated (deck : Deck) : Prop :=
  match deck with
  | 1 :: _ => True
  | _ => False

/-- Theorem stating that Eva's process always terminates -/
theorem eva_process_terminates (initial_deck : Deck) 
  (h_valid : initial_deck.length = 100 ∧ initial_deck.toFinset = Finset.range 100) :
  ∃ (n : Nat), isTerminated (n.iterate evaStep initial_deck) := by
  sorry

end eva_process_terminates_l295_29589


namespace geometric_sequence_sum_l295_29598

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 2) →
  (a 9 + a 10 = 16) := by
  sorry

end geometric_sequence_sum_l295_29598


namespace M_intersect_N_eq_M_l295_29566

def M : Set ℤ := {0, 1}
def N : Set ℤ := {x | ∃ y, y = Real.sqrt (1 - x)}

theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end M_intersect_N_eq_M_l295_29566


namespace inequality_solution_l295_29568

def solution_set : Set ℤ := {-3, 2}

def inequality (x : ℤ) : Prop :=
  (x^2 + 6*x + 8) * (x^2 - 4*x + 3) < 0

theorem inequality_solution :
  ∀ x : ℤ, inequality x ↔ x ∈ solution_set :=
by sorry

end inequality_solution_l295_29568


namespace quadratic_factorization_l295_29557

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l295_29557


namespace minimum_cologne_drops_l295_29514

theorem minimum_cologne_drops (f : ℕ) (n : ℕ) : 
  f > 0 →  -- number of boys is positive
  n > 0 →  -- number of drops is positive
  (∀ g : ℕ, g ≤ 4 → (3 * n : ℝ) ≥ (f * (n / 2 + 15) : ℝ)) →  -- no girl receives more than 3 bottles worth
  (f * ((n / 2 : ℝ) - 15) > (3 * n : ℝ)) →  -- mother receives more than any girl
  n ≥ 53 :=
by sorry

end minimum_cologne_drops_l295_29514


namespace solutions_difference_squared_l295_29561

theorem solutions_difference_squared (α β : ℝ) : 
  α ≠ β ∧ α^2 = 2*α + 1 ∧ β^2 = 2*β + 1 → (α - β)^2 = 8 := by sorry

end solutions_difference_squared_l295_29561


namespace pipe_fill_time_l295_29502

/-- Given two pipes that can fill a pool, where one takes T hours and the other takes 12 hours,
    prove that if both pipes together take 4.8 hours to fill the pool, then T = 8. -/
theorem pipe_fill_time (T : ℝ) :
  T > 0 →
  1 / T + 1 / 12 = 1 / 4.8 →
  T = 8 :=
by sorry

end pipe_fill_time_l295_29502


namespace probability_two_red_correct_l295_29584

def bag_red_balls : ℕ := 9
def bag_white_balls : ℕ := 3
def total_balls : ℕ := bag_red_balls + bag_white_balls
def drawn_balls : ℕ := 4

def probability_two_red : ℚ :=
  (Nat.choose bag_red_balls 2 * Nat.choose bag_white_balls 2) / Nat.choose total_balls drawn_balls

theorem probability_two_red_correct :
  probability_two_red = (Nat.choose bag_red_balls 2 * Nat.choose bag_white_balls 2) / Nat.choose total_balls drawn_balls :=
by sorry

end probability_two_red_correct_l295_29584


namespace water_composition_ratio_l295_29595

theorem water_composition_ratio :
  ∀ (total_mass : ℝ) (hydrogen_mass : ℝ),
    total_mass = 117 →
    hydrogen_mass = 13 →
    (hydrogen_mass / (total_mass - hydrogen_mass) = 1 / 8) :=
by
  sorry

end water_composition_ratio_l295_29595


namespace consecutive_integers_sum_46_l295_29567

theorem consecutive_integers_sum_46 :
  ∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  y = x + 1 ∧ z = y + 1 ∧ w = z + 1 ∧
  x + y + z + w = 46 :=
by sorry

end consecutive_integers_sum_46_l295_29567


namespace domino_count_for_0_to_12_l295_29521

/-- The number of tiles in a standard set of dominoes -/
def standard_domino_count : ℕ := 28

/-- The lowest value on a domino tile -/
def min_value : ℕ := 0

/-- The highest value on a domino tile in the new set -/
def max_value : ℕ := 12

/-- The number of tiles in a domino set with values from min_value to max_value -/
def domino_count (min : ℕ) (max : ℕ) : ℕ :=
  let n := max - min + 1
  (n * (n + 1)) / 2

theorem domino_count_for_0_to_12 :
  domino_count min_value max_value = 91 :=
sorry

end domino_count_for_0_to_12_l295_29521


namespace magnitude_of_sum_l295_29547

/-- Given two vectors a and b in ℝ², prove that the magnitude of a + 3b is 5√5 when a is parallel to b -/
theorem magnitude_of_sum (a b : ℝ × ℝ) (h_parallel : ∃ (k : ℝ), b = k • a) : 
  a.1 = 1 → a.2 = 2 → b.1 = -2 → 
  ‖(a.1 + 3 * b.1, a.2 + 3 * b.2)‖ = 5 * Real.sqrt 5 := by
  sorry

end magnitude_of_sum_l295_29547


namespace triangle_area_l295_29544

/-- Given a triangle ABC with sides a, b, and c, prove that its area is 3√2/4
    when sinA = √3/3 and b² + c² - a² = 6 -/
theorem triangle_area (a b c : ℝ) (h1 : Real.sin A = Real.sqrt 3 / 3) 
  (h2 : b^2 + c^2 - a^2 = 6) : 
  (1/2 : ℝ) * b * c * Real.sin A = 3 * Real.sqrt 2 / 4 := by
  sorry

end triangle_area_l295_29544


namespace bucket_weight_l295_29508

theorem bucket_weight (p q : ℝ) 
  (h1 : ∃ x y : ℝ, x + 3/4 * y = p ∧ x + 1/3 * y = q) : 
  ∃ w : ℝ, w = (5*q - p)/5 ∧ 
  ∀ x y : ℝ, (x + 3/4 * y = p ∧ x + 1/3 * y = q) → x + 1/4 * y = w :=
sorry

end bucket_weight_l295_29508


namespace recipe_flour_calculation_l295_29556

theorem recipe_flour_calculation :
  let full_recipe : ℚ := 7 + 3/4
  let one_third_recipe : ℚ := (1/3) * full_recipe
  one_third_recipe = 2 + 7/12 := by sorry

end recipe_flour_calculation_l295_29556


namespace bounded_sequence_with_recurrence_is_constant_two_l295_29509

def is_bounded_sequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n, a n ≤ M

def satisfies_recurrence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

theorem bounded_sequence_with_recurrence_is_constant_two (a : ℕ → ℕ) 
  (h_bounded : is_bounded_sequence a)
  (h_recurrence : satisfies_recurrence a) :
  ∀ n, a n = 2 :=
by sorry

end bounded_sequence_with_recurrence_is_constant_two_l295_29509


namespace tourist_cyclist_speed_problem_l295_29585

/-- Represents the problem of finding the maximum speed of a tourist and the corresponding speed of a cyclist --/
theorem tourist_cyclist_speed_problem 
  (distance : ℝ) 
  (min_cyclist_time : ℝ) 
  (cyclist_speed_increase : ℝ) 
  (meet_time : ℝ) :
  distance = 8 ∧ 
  min_cyclist_time = 0.5 ∧ 
  cyclist_speed_increase = 0.25 ∧
  meet_time = 1/6 →
  ∃ (tourist_speed cyclist_speed : ℝ),
    tourist_speed = 7 ∧
    cyclist_speed = 16 ∧
    (∀ x : ℕ, x > tourist_speed → 
      ¬(∃ y : ℝ, 
        distance / y ≥ min_cyclist_time ∧
        x * (distance / y + meet_time) + y * meet_time * (1 + cyclist_speed_increase) = distance)) :=
by sorry

end tourist_cyclist_speed_problem_l295_29585


namespace symmetrical_function_is_two_minus_ln_l295_29532

/-- A function whose graph is symmetrical to y = e^(2-x) with respect to y = x -/
def SymmetricalToExp (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f y = x

/-- The main theorem stating that f(x) = 2 - ln(x) -/
theorem symmetrical_function_is_two_minus_ln (f : ℝ → ℝ) 
    (h : SymmetricalToExp f) : 
    ∀ x > 0, f x = 2 - Real.log x := by
  sorry

end symmetrical_function_is_two_minus_ln_l295_29532


namespace arithmetic_sequence_sum_theorem_l295_29545

/-- The sum of an arithmetic sequence with first term 3, common difference 4, and last term not exceeding 47 -/
def arithmetic_sequence_sum : ℕ → ℕ := λ n => n * (3 + (4 * n - 1)) / 2

/-- The number of terms in the sequence -/
def n : ℕ := 12

theorem arithmetic_sequence_sum_theorem :
  (∀ k : ℕ, k ≤ n → 3 + 4 * (k - 1) ≤ 47) ∧ 
  3 + 4 * (n - 1) = 47 ∧
  arithmetic_sequence_sum n = 300 := by
sorry

end arithmetic_sequence_sum_theorem_l295_29545


namespace logarithm_expression_equality_l295_29504

theorem logarithm_expression_equality : 
  (Real.log (27^(1/2)) + Real.log 8 - 3 * Real.log (10^(1/2))) / Real.log 1.2 = 3/2 := by
  sorry

end logarithm_expression_equality_l295_29504


namespace ryan_bus_meet_once_l295_29587

/-- Represents the movement of Ryan and the bus on a linear trail --/
structure TrailMovement where
  ryan_speed : ℝ
  bus_speed : ℝ
  bench_distance : ℝ
  regular_stop_time : ℝ
  extra_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Ryan and the bus meet --/
def number_of_meetings (movement : TrailMovement) : ℕ :=
  sorry

/-- The specific trail movement scenario described in the problem --/
def problem_scenario : TrailMovement :=
  { ryan_speed := 6
  , bus_speed := 15
  , bench_distance := 300
  , regular_stop_time := 45
  , extra_stop_time := 90
  , initial_distance := 300 }

/-- Theorem stating that Ryan and the bus meet exactly once --/
theorem ryan_bus_meet_once :
  number_of_meetings problem_scenario = 1 := by
  sorry

end ryan_bus_meet_once_l295_29587


namespace f_1_eq_0_f_increasing_f_inequality_solution_l295_29553

noncomputable section

variable (f : ℝ → ℝ)

axiom domain : ∀ x, x > 0 → f x ≠ 0 → True

axiom f_4 : f 4 = 1

axiom f_product : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

axiom f_neg_on_unit : ∀ x, 0 < x → x < 1 → f x < 0

theorem f_1_eq_0 : f 1 = 0 := by sorry

theorem f_increasing : ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

theorem f_inequality_solution : 
  ∀ x, x > 0 → (f (3 * x + 1) + f (2 * x - 6) ≤ 3 ↔ 3 < x ∧ x ≤ 5) := by sorry

end

end f_1_eq_0_f_increasing_f_inequality_solution_l295_29553


namespace absolute_value_inequalities_l295_29522

theorem absolute_value_inequalities (x y : ℝ) : 
  (abs (x + y) ≤ abs x + abs y) ∧ 
  (abs (x - y) ≥ abs x - abs y) ∧ 
  (abs (x - y) ≥ abs (abs x - abs y)) := by
  sorry

end absolute_value_inequalities_l295_29522


namespace min_three_digit_quotient_l295_29505

theorem min_three_digit_quotient :
  ∃ (a b c : ℕ), 
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    (∀ (x y z : ℕ), x ≤ 9 → y ≤ 9 → z ≤ 9 →
      (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ) ≤ (100 * x + 10 * y + z : ℚ) / (x + y + z : ℚ)) ∧
    (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ) = 50.5 := by
  sorry

end min_three_digit_quotient_l295_29505


namespace grape_juice_percentage_l295_29578

-- Define the initial mixture volume
def initial_volume : ℝ := 30

-- Define the initial percentage of grape juice
def initial_grape_percentage : ℝ := 0.1

-- Define the volume of grape juice added
def added_grape_volume : ℝ := 10

-- Define the resulting percentage of grape juice
def resulting_grape_percentage : ℝ := 0.325

theorem grape_juice_percentage :
  let initial_grape_volume := initial_volume * initial_grape_percentage
  let total_grape_volume := initial_grape_volume + added_grape_volume
  let final_volume := initial_volume + added_grape_volume
  (total_grape_volume / final_volume) = resulting_grape_percentage := by
sorry

end grape_juice_percentage_l295_29578


namespace apple_bag_weight_l295_29560

/-- Given a bag of apples costing 3.50 dollars, and knowing that 7 pounds of apples
    at the same rate would cost 4.9 dollars, prove that the bag contains 5 pounds of apples. -/
theorem apple_bag_weight (bag_cost : ℝ) (rate_pounds : ℝ) (rate_cost : ℝ) :
  bag_cost = 3.50 →
  rate_pounds = 7 →
  rate_cost = 4.9 →
  (rate_cost / rate_pounds) * (bag_cost / (rate_cost / rate_pounds)) = 5 :=
by sorry

end apple_bag_weight_l295_29560


namespace cube_inequality_l295_29536

theorem cube_inequality (a b : ℝ) : a < b → a^3 < b^3 := by
  sorry

end cube_inequality_l295_29536


namespace range_of_a_l295_29571

theorem range_of_a (p q : Prop) (h_p : ∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ a : ℝ, a ≥ Real.exp x) 
  (h_q : ∃ (a : ℝ) (x : ℝ), x^2 + 4*x + a = 0) (h_pq : p ∧ q) :
  ∃ a : ℝ, a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end range_of_a_l295_29571


namespace no_six_numbers_exist_l295_29503

/-- Represents a six-digit number composed of digits 1 to 6 without repetitions -/
def SixDigitNumber := Fin 6 → Fin 6

/-- Represents a three-digit number composed of digits 1 to 6 without repetitions -/
def ThreeDigitNumber := Fin 3 → Fin 6

/-- Checks if a ThreeDigitNumber can be obtained from a SixDigitNumber by deleting three digits -/
def canBeObtained (six : SixDigitNumber) (three : ThreeDigitNumber) : Prop :=
  ∃ (i j k : Fin 6), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    (∀ m : Fin 3, three m = six (if m < i then m else if m < j then m + 1 else m + 2))

/-- The main theorem stating that the required set of six numbers does not exist -/
theorem no_six_numbers_exist : 
  ¬ ∃ (numbers : Fin 6 → SixDigitNumber),
    (∀ i : Fin 6, Function.Injective (numbers i)) ∧
    (∀ three : ThreeDigitNumber, Function.Injective three → 
      ∃ (i : Fin 6), canBeObtained (numbers i) three) :=
by sorry


end no_six_numbers_exist_l295_29503


namespace pentadecagon_triangles_l295_29530

/-- The number of sides in a regular pentadecagon -/
def n : ℕ := 15

/-- The total number of triangles that can be formed using any three vertices of a regular pentadecagon -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles formed by three consecutive vertices in a regular pentadecagon -/
def consecutive_triangles : ℕ := n

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon, 
    where no triangle is formed by three consecutive vertices -/
def valid_triangles : ℕ := total_triangles - consecutive_triangles

theorem pentadecagon_triangles : valid_triangles = 440 := by
  sorry

end pentadecagon_triangles_l295_29530


namespace sports_club_intersection_l295_29577

/-- Given a sports club with the following properties:
  - There are 30 total members
  - 16 members play badminton
  - 19 members play tennis
  - 2 members play neither badminton nor tennis
  Prove that 7 members play both badminton and tennis -/
theorem sports_club_intersection (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 16 →
  tennis = 19 →
  neither = 2 →
  badminton + tennis - (total - neither) = 7 := by
  sorry

end sports_club_intersection_l295_29577


namespace equal_intercept_line_equation_l295_29583

/-- A line passing through (2, 1) with equal intercepts on x and y axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 1)
  point_condition : 1 = 2 * m + b
  -- The line has equal intercepts on x and y axes
  equal_intercepts : (m ≠ -1 → -b / (1 + m) = -b / m) ∧ (m = -1 → b = 0)

/-- The equation of the line is either x+y-3=0 or y = 1/2x -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -1 ∧ l.b = 3) ∨ (l.m = 1/2 ∧ l.b = 0) :=
sorry

end equal_intercept_line_equation_l295_29583


namespace fixed_salary_is_1000_l295_29575

/-- Represents the earnings structure and goal of a sales executive -/
structure SalesExecutive where
  commissionRate : Float
  targetEarnings : Float
  targetSales : Float

/-- Calculates the fixed salary for a sales executive -/
def calculateFixedSalary (exec : SalesExecutive) : Float :=
  exec.targetEarnings - exec.commissionRate * exec.targetSales

/-- Theorem: The fixed salary for the given sales executive is $1000 -/
theorem fixed_salary_is_1000 :
  let exec : SalesExecutive := {
    commissionRate := 0.05,
    targetEarnings := 5000,
    targetSales := 80000
  }
  calculateFixedSalary exec = 1000 := by
  sorry

#eval calculateFixedSalary {
  commissionRate := 0.05,
  targetEarnings := 5000,
  targetSales := 80000
}

end fixed_salary_is_1000_l295_29575


namespace range_of_t_l295_29527

noncomputable def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 2)^2 - 2 * (Real.log x / Real.log 2) + t - 4

noncomputable def F (t : ℝ) (x : ℝ) : ℝ := f (g t x) - 1

theorem range_of_t (t : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 1 (2 * Real.sqrt 2) ∧ y ∈ Set.Icc 1 (2 * Real.sqrt 2) ∧
    F t x = 0 ∧ F t y = 0 ∧
    ∀ z ∈ Set.Icc 1 (2 * Real.sqrt 2), F t z = 0 → z = x ∨ z = y) →
  t ∈ Set.Icc 4 (9/2) :=
sorry

end range_of_t_l295_29527


namespace product_sequence_sum_l295_29543

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 12) (h2 : b = a - 1) : a + b = 71 := by
  sorry

end product_sequence_sum_l295_29543


namespace circle_diameter_l295_29515

/-- Given a circle with area M and circumference N, if M/N = 15, then the diameter is 60 -/
theorem circle_diameter (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  let r := N / (2 * Real.pi)
  let d := 2 * r
  d = 60 := by sorry

end circle_diameter_l295_29515


namespace right_triangle_hypotenuse_l295_29542

theorem right_triangle_hypotenuse : 
  ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg = 2 * short_leg + 3 →
  (1 / 2) * short_leg * long_leg = 84 →
  short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2 →
  hypotenuse = Real.sqrt 261 :=
by sorry

end right_triangle_hypotenuse_l295_29542


namespace algebraic_expression_value_l295_29594

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b + 3 = 0) :
  5 + 2*b - a = 8 := by
  sorry

end algebraic_expression_value_l295_29594


namespace negation_of_all_exponential_are_monotonic_l295_29570

-- Define exponential function
def ExponentialFunction (f : ℝ → ℝ) : Prop := sorry

-- Define monotonic function
def MonotonicFunction (f : ℝ → ℝ) : Prop := sorry

-- Theorem statement
theorem negation_of_all_exponential_are_monotonic :
  (¬ ∀ f : ℝ → ℝ, ExponentialFunction f → MonotonicFunction f) ↔
  (∃ f : ℝ → ℝ, ExponentialFunction f ∧ ¬MonotonicFunction f) :=
by sorry

end negation_of_all_exponential_are_monotonic_l295_29570


namespace player_one_wins_with_2023_coins_l295_29580

/-- Represents the possible moves for each player -/
inductive Move
| three : Move
| five : Move
| two : Move
| four : Move

/-- Represents a player in the game -/
inductive Player
| one : Player
| two : Player

/-- The game state -/
structure GameState where
  coins : ℕ
  currentPlayer : Player

/-- Determines if a move is valid for a given player -/
def validMove (player : Player) (move : Move) : Bool :=
  match player, move with
  | Player.one, Move.three => true
  | Player.one, Move.five => true
  | Player.two, Move.two => true
  | Player.two, Move.four => true
  | _, _ => false

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : Option GameState :=
  if validMove state.currentPlayer move then
    let newCoins := match move with
      | Move.three => state.coins - 3
      | Move.five => state.coins - 5
      | Move.two => state.coins - 2
      | Move.four => state.coins - 4
    let newPlayer := match state.currentPlayer with
      | Player.one => Player.two
      | Player.two => Player.one
    some { coins := newCoins, currentPlayer := newPlayer }
  else
    none

/-- Determines if a player has a winning strategy from a given game state -/
def hasWinningStrategy (state : GameState) : Prop :=
  sorry

/-- The main theorem: Player 1 has a winning strategy when starting with 2023 coins -/
theorem player_one_wins_with_2023_coins :
  hasWinningStrategy { coins := 2023, currentPlayer := Player.one } :=
  sorry

end player_one_wins_with_2023_coins_l295_29580


namespace flowers_left_in_peters_garden_l295_29512

/-- The number of flowers in Amanda's garden -/
def amanda_flowers : ℕ := 20

/-- The number of flowers in Peter's garden before giving away -/
def peter_flowers : ℕ := 3 * amanda_flowers

/-- The number of flowers Peter gave away -/
def flowers_given_away : ℕ := 15

/-- Theorem: The number of flowers left in Peter's garden is 45 -/
theorem flowers_left_in_peters_garden :
  peter_flowers - flowers_given_away = 45 := by
  sorry

end flowers_left_in_peters_garden_l295_29512


namespace parabola_translation_l295_29501

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 2 0 0  -- y = 2x^2
  let p_translated := translate p 1 3  -- Translate 1 right, 3 up
  y = 2 * x^2 → y = 2 * (x - 1)^2 + 3 :=
by
  sorry

#check parabola_translation

end parabola_translation_l295_29501


namespace calculation_proof_l295_29534

theorem calculation_proof : (8 * 5.4 - 0.6 * 10 / 1.2)^2 = 1459.24 := by
  sorry

end calculation_proof_l295_29534


namespace negation_of_existence_negation_of_square_equals_one_l295_29592

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_square_equals_one :
  (¬ ∃ x : ℝ, x^2 = 1) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
by sorry

end negation_of_existence_negation_of_square_equals_one_l295_29592


namespace same_color_shoe_probability_l295_29555

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 7

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
def same_color_prob : ℚ := 1 / 13

theorem same_color_shoe_probability :
  (num_pairs : ℚ) / (total_shoes.choose selected_shoes) = same_color_prob :=
sorry

end same_color_shoe_probability_l295_29555


namespace triangle_altitude_segment_l295_29596

theorem triangle_altitude_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 82.5 := by
sorry

end triangle_altitude_segment_l295_29596


namespace pairball_play_time_l295_29513

theorem pairball_play_time (total_duration : ℕ) (num_children : ℕ) (children_per_game : ℕ) :
  total_duration = 120 →
  num_children = 6 →
  children_per_game = 2 →
  (total_duration * children_per_game) / num_children = 40 :=
by
  sorry

end pairball_play_time_l295_29513


namespace complex_product_pure_imaginary_l295_29590

theorem complex_product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 2 + Complex.I
  (z₁ * z₂).re = 0 → a = 1 := by
  sorry

end complex_product_pure_imaginary_l295_29590


namespace blackboard_sum_divisibility_l295_29581

theorem blackboard_sum_divisibility (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ x ∈ Finset.range n, ¬ ∃ y ∈ Finset.range n,
    (((n * (3 * n - 1)) / 2 - (n + x)) % (n + y) = 0) := by
  sorry

end blackboard_sum_divisibility_l295_29581


namespace polynomial_factorization_l295_29551

theorem polynomial_factorization :
  ∀ (x y a b : ℝ),
    (12 * x^3 * y - 3 * x * y^2 = 3 * x * y * (4 * x^2 - y)) ∧
    (x - 9 * x^3 = x * (1 + 3 * x) * (1 - 3 * x)) ∧
    (3 * a^2 - 12 * a * b * (a - b) = 3 * (a - 2 * b)^2) :=
by sorry

end polynomial_factorization_l295_29551


namespace bus_tour_sales_l295_29554

theorem bus_tour_sales (total_tickets : ℕ) (senior_price regular_price : ℕ) (regular_tickets : ℕ)
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : regular_tickets = 41) :
  (total_tickets - regular_tickets) * senior_price + regular_tickets * regular_price = 855 := by
  sorry

end bus_tour_sales_l295_29554


namespace arctan_sum_equals_pi_half_l295_29563

theorem arctan_sum_equals_pi_half (a b : ℝ) (h1 : a = 1/3) (h2 : (a+1)*(b+1) = 3) :
  Real.arctan a + Real.arctan b = π/2 := by
  sorry

end arctan_sum_equals_pi_half_l295_29563


namespace sunday_school_average_class_size_l295_29576

/-- Represents the number of students in each age group -/
structure AgeGroups where
  three_year_olds : Nat
  four_year_olds : Nat
  five_year_olds : Nat
  six_year_olds : Nat
  seven_year_olds : Nat
  eight_year_olds : Nat

/-- Calculates the average class size given the age groups -/
def averageClassSize (groups : AgeGroups) : Rat :=
  let class1 := groups.three_year_olds + groups.four_year_olds
  let class2 := groups.five_year_olds + groups.six_year_olds
  let class3 := groups.seven_year_olds + groups.eight_year_olds
  let totalStudents := class1 + class2 + class3
  (totalStudents : Rat) / 3

/-- The specific age groups given in the problem -/
def sundaySchoolGroups : AgeGroups := {
  three_year_olds := 13,
  four_year_olds := 20,
  five_year_olds := 15,
  six_year_olds := 22,
  seven_year_olds := 18,
  eight_year_olds := 25
}

theorem sunday_school_average_class_size :
  averageClassSize sundaySchoolGroups = 113 / 3 := by
  sorry

#eval averageClassSize sundaySchoolGroups

end sunday_school_average_class_size_l295_29576


namespace intersection_condition_l295_29550

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a = 0 ∨ a = 1/2 ∨ a = 1/3 := by
  sorry

end intersection_condition_l295_29550


namespace polynomial_property_l295_29539

def P (a b c : ℝ) (x : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + c

theorem polynomial_property (a b c : ℝ) :
  P a b c 0 = 8 →
  (∃ m : ℝ, m = (-(c / 2)) ∧ 
             m = -((a / 2) / 3) ∧ 
             m = 2 + a + b + c) →
  b = -38 := by sorry

end polynomial_property_l295_29539


namespace sphere_surface_area_l295_29500

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem sphere_surface_area (π : ℝ) (h : π > 0) : 
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * π * radius^2
  surface_area = 81 * π :=
by sorry

end sphere_surface_area_l295_29500


namespace sum_interior_angles_hexagon_l295_29541

/-- The sum of interior angles of a regular hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  ∀ (n : ℕ) (sum : ℝ),
  n = 6 →
  sum = (n - 2) * 180 →
  sum = 720 := by
  sorry

end sum_interior_angles_hexagon_l295_29541


namespace mans_rate_in_still_water_l295_29559

/-- Given a man's rowing speeds with and against a stream, calculates his rate in still water. -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 20) 
  (h2 : speed_against_stream = 8) : 
  (speed_with_stream + speed_against_stream) / 2 = 14 := by
  sorry

#check mans_rate_in_still_water

end mans_rate_in_still_water_l295_29559


namespace fourth_term_is_eight_l295_29569

/-- An arithmetic progression with the given property -/
def ArithmeticProgression (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = S n + 1

theorem fourth_term_is_eight
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : ArithmeticProgression a S) :
  a 4 = 8 := by
  sorry

end fourth_term_is_eight_l295_29569


namespace hyperbola_eccentricity_l295_29582

/-- A hyperbola with foci on the y-axis and asymptotes y = ±4x has eccentricity √17/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a = 4*b) : 
  let e := (Real.sqrt (a^2 + b^2)) / a
  e = Real.sqrt 17 / 4 := by sorry

end hyperbola_eccentricity_l295_29582


namespace sequence_properties_l295_29562

def S (n : ℕ+) : ℤ := -n^2 + 24*n

def a (n : ℕ+) : ℤ := -2*n + 25

theorem sequence_properties :
  (∀ n : ℕ+, S n - S (n-1) = a n) ∧
  (∀ n : ℕ+, n ≤ 12 → S n ≤ S 12) ∧
  (S 12 = 144) := by sorry

end sequence_properties_l295_29562


namespace goods_train_speed_l295_29510

/-- The speed of a goods train passing a man in another train -/
theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (passing_time : ℝ) 
  (goods_train_length : ℝ) 
  (h1 : man_train_speed = 50) 
  (h2 : passing_time = 9 / 3600) 
  (h3 : goods_train_length = 0.28) : 
  ∃ (goods_train_speed : ℝ), goods_train_speed = 62 := by
sorry

end goods_train_speed_l295_29510
