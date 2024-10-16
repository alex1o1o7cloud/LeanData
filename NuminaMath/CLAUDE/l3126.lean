import Mathlib

namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l3126_312638

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_squares_per_face : Nat
  num_faces : Nat

/-- The number of unpainted cubes in a painted cube -/
def num_unpainted_cubes (c : PaintedCube) : Nat :=
  c.total_cubes - (c.painted_squares_per_face * c.num_faces / 2)

/-- Theorem: In a 6x6x6 cube with 216 unit cubes and 4 painted squares on each of 6 faces,
    the number of unpainted cubes is 208 -/
theorem unpainted_cubes_in_6x6x6 :
  let c : PaintedCube := {
    size := 6,
    total_cubes := 216,
    painted_squares_per_face := 4,
    num_faces := 6
  }
  num_unpainted_cubes c = 208 := by sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l3126_312638


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l3126_312661

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_one
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = 2^x + 2*x + m)
  (m : ℝ) :
  f (-1) = -3 :=
sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_one_l3126_312661


namespace NUMINAMATH_CALUDE_patricia_lemon_heads_l3126_312695

/-- The number of Lemon Heads Patricia ate -/
def eaten : ℕ := 15

/-- The number of Lemon Heads Patricia gave to her friend -/
def given : ℕ := 5

/-- The number of Lemon Heads in each package -/
def per_package : ℕ := 3

/-- The function to calculate the number of packages -/
def calculate_packages (total : ℕ) : ℕ :=
  (total + per_package - 1) / per_package

/-- Theorem stating that Patricia originally had 7 packages of Lemon Heads -/
theorem patricia_lemon_heads : calculate_packages (eaten + given) = 7 := by
  sorry

end NUMINAMATH_CALUDE_patricia_lemon_heads_l3126_312695


namespace NUMINAMATH_CALUDE_prime_p_cube_condition_l3126_312605

theorem prime_p_cube_condition (p : ℕ) : 
  Prime p → (∃ n : ℕ, 13 * p + 1 = n^3) → p = 2 ∨ p = 211 := by
sorry

end NUMINAMATH_CALUDE_prime_p_cube_condition_l3126_312605


namespace NUMINAMATH_CALUDE_system_properties_l3126_312693

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x + 3 * y = 4 - a) ∧ (x - y = 3 * a)

-- Theorem statement
theorem system_properties :
  ∀ (x y a : ℝ), system x y a →
    ((x + y = 0) → (a = -2)) ∧
    (x + 2 * y = 3) ∧
    (y = -x / 2 + 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_system_properties_l3126_312693


namespace NUMINAMATH_CALUDE_circle_equation_l3126_312669

/-- Given a circle with center at (-2, 3) and tangent to the y-axis, 
    its equation is (x+2)^2+(y-3)^2=4 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (-2, 3)
  let tangent_to_y_axis : ℝ → Prop := λ r => r = 2
  tangent_to_y_axis (abs center.1) →
  (x + 2)^2 + (y - 3)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l3126_312669


namespace NUMINAMATH_CALUDE_rounding_shift_l3126_312667

/-- Rounding function that rounds to the nearest integer -/
noncomputable def f (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

/-- Theorem stating that adding an integer to the input of f
    is equivalent to adding the same integer to the output of f -/
theorem rounding_shift (x : ℝ) (m : ℤ) : f (x + m) = f x + m := by
  sorry

end NUMINAMATH_CALUDE_rounding_shift_l3126_312667


namespace NUMINAMATH_CALUDE_real_roots_quadratic_range_l3126_312671

theorem real_roots_quadratic_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_range_l3126_312671


namespace NUMINAMATH_CALUDE_thursday_productivity_l3126_312655

/-- Represents the relationship between cups of coffee and lines of code written --/
structure CoffeeProductivity where
  k : ℝ  -- Proportionality constant
  coffee_to_code : ℝ → ℝ  -- Function that converts cups of coffee to lines of code

/-- Given the conditions from the problem, prove that the programmer wrote 250 lines of code on Thursday --/
theorem thursday_productivity (cp : CoffeeProductivity) 
  (h1 : cp.coffee_to_code 3 = 150)  -- Wednesday's data
  (h2 : ∀ c, cp.coffee_to_code c = cp.k * c)  -- Direct proportionality
  : cp.coffee_to_code 5 = 250 := by
  sorry

#check thursday_productivity

end NUMINAMATH_CALUDE_thursday_productivity_l3126_312655


namespace NUMINAMATH_CALUDE_division_problem_l3126_312615

theorem division_problem (a : ℝ) : a / 0.3 = 0.6 → a = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3126_312615


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_l3126_312657

/-- The probability of selecting at least one girl when randomly choosing 2 people from a group of 3 boys and 2 girls is 7/10 -/
theorem prob_at_least_one_girl (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 2) :
  let total := boys + girls
  let prob_at_least_one_girl := 1 - (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ)
  prob_at_least_one_girl = 7/10 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_l3126_312657


namespace NUMINAMATH_CALUDE_andy_work_hours_l3126_312603

-- Define the variables and constants
def hourly_rate : ℝ := 9
def restring_fee : ℝ := 15
def grommet_fee : ℝ := 10
def stencil_fee : ℝ := 1
def total_earnings : ℝ := 202
def racquets_strung : ℕ := 7
def grommets_changed : ℕ := 2
def stencils_painted : ℕ := 5

-- State the theorem
theorem andy_work_hours :
  ∃ (hours : ℝ),
    hours * hourly_rate +
    racquets_strung * restring_fee +
    grommets_changed * grommet_fee +
    stencils_painted * stencil_fee = total_earnings ∧
    hours = 8 := by sorry

end NUMINAMATH_CALUDE_andy_work_hours_l3126_312603


namespace NUMINAMATH_CALUDE_smallest_even_five_digit_number_tens_place_l3126_312660

def Digits : Finset ℕ := {1, 2, 3, 5, 8}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  (∀ d : ℕ, d ∈ Digits → (n.digits 10).count d = 1) ∧
  (∀ d : ℕ, d ∉ Digits → (n.digits 10).count d = 0)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem smallest_even_five_digit_number_tens_place :
  ∃ n : ℕ, is_valid_number n ∧ is_even n ∧
    (∀ m : ℕ, is_valid_number m ∧ is_even m → n ≤ m) ∧
    tens_digit n = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_even_five_digit_number_tens_place_l3126_312660


namespace NUMINAMATH_CALUDE_matrix_power_2018_l3126_312622

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 1, 1]

theorem matrix_power_2018 :
  A ^ 2018 = !![1, 0; 2018, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2018_l3126_312622


namespace NUMINAMATH_CALUDE_original_rectangle_area_l3126_312675

theorem original_rectangle_area (new_area : ℝ) (h1 : new_area = 32) : ∃ original_area : ℝ,
  (original_area * 4 = new_area) ∧ original_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l3126_312675


namespace NUMINAMATH_CALUDE_max_min_f_on_I_l3126_312663

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval
def I : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem max_min_f_on_I :
  ∃ (a b : ℝ), a ∈ I ∧ b ∈ I ∧
  (∀ x ∈ I, f x ≤ f a) ∧
  (∀ x ∈ I, f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 := by
sorry

end NUMINAMATH_CALUDE_max_min_f_on_I_l3126_312663


namespace NUMINAMATH_CALUDE_no_four_integers_l3126_312649

theorem no_four_integers (n : ℕ) (hn : n ≥ 1) :
  ¬ ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    n^2 ≤ a ∧ a < (n+1)^2 ∧
    n^2 ≤ b ∧ b < (n+1)^2 ∧
    n^2 ≤ c ∧ c < (n+1)^2 ∧
    n^2 ≤ d ∧ d < (n+1)^2 ∧
    a * d = b * c :=
by sorry

end NUMINAMATH_CALUDE_no_four_integers_l3126_312649


namespace NUMINAMATH_CALUDE_vicente_spent_2475_l3126_312672

/-- Calculates the total amount spent by Vicente on rice and meat --/
def total_spent (rice_kg : ℕ) (rice_price : ℚ) (rice_discount : ℚ)
                (meat_lbs : ℕ) (meat_price : ℚ) (meat_tax : ℚ) : ℚ :=
  let rice_cost := rice_kg * rice_price * (1 - rice_discount)
  let meat_cost := meat_lbs * meat_price * (1 + meat_tax)
  rice_cost + meat_cost

/-- Theorem stating that Vicente's total spent is $24.75 --/
theorem vicente_spent_2475 :
  total_spent 5 2 (1/10) 3 5 (1/20) = 2475/100 := by
  sorry

end NUMINAMATH_CALUDE_vicente_spent_2475_l3126_312672


namespace NUMINAMATH_CALUDE_simplify_fraction_l3126_312618

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3126_312618


namespace NUMINAMATH_CALUDE_M_equals_six_eight_C_U_A_inter_C_U_B_equals_five_nine_ten_l3126_312641

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6, 8}

-- Define set B
def B : Set ℕ := {x ∈ U | x < 5}

-- Define set M
def M : Set ℕ := {x ∈ A | x ∉ B}

-- Define the complement of A in U
def C_U_A : Set ℕ := U \ A

-- Define the complement of B in U
def C_U_B : Set ℕ := U \ B

-- Theorem for part (1)
theorem M_equals_six_eight : M = {6, 8} := by sorry

-- Theorem for part (2)
theorem C_U_A_inter_C_U_B_equals_five_nine_ten : C_U_A ∩ C_U_B = {5, 9, 10} := by sorry

end NUMINAMATH_CALUDE_M_equals_six_eight_C_U_A_inter_C_U_B_equals_five_nine_ten_l3126_312641


namespace NUMINAMATH_CALUDE_peter_wins_iff_n_odd_l3126_312628

/-- Represents the state of a cup (empty or filled) -/
inductive CupState
| Empty : CupState
| Filled : CupState

/-- Represents a player in the game -/
inductive Player
| Peter : Player
| Vasya : Player

/-- The game state on a 2n-gon -/
structure GameState (n : ℕ) where
  cups : Fin (2 * n) → CupState
  currentPlayer : Player

/-- Checks if two positions are symmetric with respect to the center of the 2n-gon -/
def isSymmetric (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + j.val) % (2 * n) = 0

/-- A valid move in the game -/
inductive Move (n : ℕ)
| Single : Fin (2 * n) → Move n
| Double : (i j : Fin (2 * n)) → isSymmetric n i j → Move n

/-- Applies a move to the game state -/
def applyMove (n : ℕ) (state : GameState n) (move : Move n) : GameState n :=
  sorry

/-- Checks if a player has a winning strategy -/
def hasWinningStrategy (n : ℕ) (player : Player) : Prop :=
  sorry

/-- The main theorem: Peter has a winning strategy if and only if n is odd -/
theorem peter_wins_iff_n_odd (n : ℕ) :
  hasWinningStrategy n Player.Peter ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_peter_wins_iff_n_odd_l3126_312628


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3126_312681

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 3) (h2 : p 3 = 9) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * (x - 3) * q x + (6 * x - 9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3126_312681


namespace NUMINAMATH_CALUDE_sum_of_30th_set_l3126_312616

/-- Defines the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + (n * (n - 1)) / 2

/-- Defines the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Defines the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

theorem sum_of_30th_set : S 30 = 13515 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_30th_set_l3126_312616


namespace NUMINAMATH_CALUDE_function_roots_imply_a_range_l3126_312624

/-- The function f(x) = 2ln(x) - x^2 + a has two roots in [1/e, e] iff a ∈ (1, 2 + 1/e^2] -/
theorem function_roots_imply_a_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 2 * Real.log x - x^2 + a) →
  (∃ x y, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
          y ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ 
          x ≠ y ∧ f x = 0 ∧ f y = 0) →
  a ∈ Set.Ioo 1 (2 + 1 / (Real.exp 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_function_roots_imply_a_range_l3126_312624


namespace NUMINAMATH_CALUDE_coin_flip_frequency_l3126_312620

/-- The frequency of an event is the ratio of the number of times the event occurs to the total number of trials. -/
def frequency (occurrences : ℕ) (trials : ℕ) : ℚ :=
  occurrences / trials

/-- In an experiment of flipping a coin 100 times, the frequency of getting "heads" is 49. -/
theorem coin_flip_frequency :
  frequency 49 100 = 49/100 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_frequency_l3126_312620


namespace NUMINAMATH_CALUDE_prob_second_day_A_l3126_312642

-- Define the probabilities
def prob_A_given_A : ℝ := 0.7
def prob_A_given_B : ℝ := 0.5
def prob_first_day_A : ℝ := 0.5
def prob_first_day_B : ℝ := 0.5

-- State the theorem
theorem prob_second_day_A :
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_day_A_l3126_312642


namespace NUMINAMATH_CALUDE_movie_profit_l3126_312676

def movie_production (main_actor_fee supporting_actor_fee extra_fee : ℕ)
                     (main_actor_food supporting_actor_food crew_food : ℕ)
                     (post_production_cost revenue : ℕ) : Prop :=
  let main_actors := 2
  let supporting_actors := 3
  let extras := 1
  let total_people := 50
  let actor_fees := main_actors * main_actor_fee + 
                    supporting_actors * supporting_actor_fee + 
                    extras * extra_fee
  let food_cost := main_actors * main_actor_food + 
                   (supporting_actors + extras) * supporting_actor_food + 
                   (total_people - main_actors - supporting_actors - extras) * crew_food
  let equipment_rental := 2 * (actor_fees + food_cost)
  let total_cost := actor_fees + food_cost + equipment_rental + post_production_cost
  let profit := revenue - total_cost
  profit = 4584

theorem movie_profit :
  movie_production 500 100 50 10 5 3 850 10000 :=
by sorry

end NUMINAMATH_CALUDE_movie_profit_l3126_312676


namespace NUMINAMATH_CALUDE_smoking_lung_cancer_relationship_l3126_312692

-- Define the confidence level in the smoking-lung cancer relationship
def confidence_level : ℝ := 0.99

-- Define the probability of making a mistake in the conclusion
def error_probability : ℝ := 0.01

-- Define a sample size
def sample_size : ℕ := 100

-- Define a predicate for having lung cancer
def has_lung_cancer : (ℕ → Prop) := sorry

-- Define a predicate for being a smoker
def is_smoker : (ℕ → Prop) := sorry

-- Theorem stating that high confidence in the smoking-lung cancer relationship
-- does not preclude the possibility of a sample with no lung cancer cases
theorem smoking_lung_cancer_relationship 
  (h1 : confidence_level > 0.99) 
  (h2 : error_probability ≤ 0.01) :
  ∃ (sample : Finset ℕ), 
    (∀ i ∈ sample, is_smoker i) ∧ 
    (Finset.card sample = sample_size) ∧
    (∀ i ∈ sample, ¬has_lung_cancer i) := by
  sorry

end NUMINAMATH_CALUDE_smoking_lung_cancer_relationship_l3126_312692


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3126_312607

theorem exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3126_312607


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l3126_312656

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : subset l α)
  (h2 : perp l β) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l3126_312656


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3126_312677

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (2*x - 3)^6 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 ∧
    a₄ = 240 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3126_312677


namespace NUMINAMATH_CALUDE_ripe_peaches_initially_l3126_312635

/-- Represents the state of the fruit bowl on a given day -/
structure FruitBowl :=
  (day : ℕ)
  (ripe : ℕ)
  (unripe : ℕ)

/-- Updates the fruit bowl state for the next day -/
def nextDay (bowl : FruitBowl) : FruitBowl :=
  let newRipe := bowl.ripe + 2
  let newUnripe := bowl.unripe - 2
  let (finalRipe, finalUnripe) := 
    if bowl.day = 2 then (newRipe - 3, newUnripe) else (newRipe, newUnripe)
  { day := bowl.day + 1, ripe := finalRipe, unripe := finalUnripe }

def initialBowl (initialRipe : ℕ) : FruitBowl :=
  { day := 0, ripe := initialRipe, unripe := 18 - initialRipe }

theorem ripe_peaches_initially (initialRipe : ℕ) : 
  (initialRipe = 1) ↔ 
  (let finalBowl := (initialBowl initialRipe) |> nextDay |> nextDay |> nextDay |> nextDay |> nextDay
   finalBowl.ripe = finalBowl.unripe + 7 ∧ finalBowl.ripe + finalBowl.unripe = 15) :=
sorry

end NUMINAMATH_CALUDE_ripe_peaches_initially_l3126_312635


namespace NUMINAMATH_CALUDE_third_shot_probability_l3126_312682

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.9

-- Define the number of shots
def num_shots : ℕ := 4

-- Define the event of hitting the target on the nth shot
def hit_on_nth_shot (n : ℕ) : ℝ := hit_probability

-- Theorem statement
theorem third_shot_probability :
  hit_on_nth_shot 3 = hit_probability :=
by sorry

end NUMINAMATH_CALUDE_third_shot_probability_l3126_312682


namespace NUMINAMATH_CALUDE_smallest_multiple_with_remainder_three_l3126_312690

theorem smallest_multiple_with_remainder_three : 
  (∀ n : ℕ, n > 1 ∧ n < 843 → 
    ¬(n % 4 = 3 ∧ n % 5 = 3 ∧ n % 6 = 3 ∧ n % 7 = 3 ∧ n % 8 = 3)) ∧ 
  (843 % 4 = 3 ∧ 843 % 5 = 3 ∧ 843 % 6 = 3 ∧ 843 % 7 = 3 ∧ 843 % 8 = 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_remainder_three_l3126_312690


namespace NUMINAMATH_CALUDE_no_spiky_two_digit_integers_l3126_312680

/-- A two-digit positive integer is spiky if it equals the sum of its tens digit and 
    the cube of its units digit subtracted by twice the tens digit. -/
def IsSpiky (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ 
  ∃ a b : ℕ, n = 10 * a + b ∧ 
             n = a + b^3 - 2*a

/-- There are no spiky two-digit positive integers. -/
theorem no_spiky_two_digit_integers : ¬∃ n : ℕ, IsSpiky n := by
  sorry

#check no_spiky_two_digit_integers

end NUMINAMATH_CALUDE_no_spiky_two_digit_integers_l3126_312680


namespace NUMINAMATH_CALUDE_sum_of_digits_l3126_312647

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    b = a + c ∧
    100 * c + 10 * b + a = n + 99 ∧
    n = 253

theorem sum_of_digits (n : ℕ) (h : is_valid_number n) : 
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3126_312647


namespace NUMINAMATH_CALUDE_married_men_fraction_l3126_312619

theorem married_men_fraction (total_women : ℕ) (h_pos : 0 < total_women) :
  let single_women := (3 * total_women : ℕ) / 5
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  (married_men : ℚ) / total_people = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l3126_312619


namespace NUMINAMATH_CALUDE_carpet_width_in_cm_l3126_312621

/-- Proves that the width of the carpet is 1000 centimeters given the room dimensions and carpeting costs. -/
theorem carpet_width_in_cm (room_length room_breadth carpet_cost_per_meter total_cost : ℝ) 
  (h1 : room_length = 18)
  (h2 : room_breadth = 7.5)
  (h3 : carpet_cost_per_meter = 4.5)
  (h4 : total_cost = 810) : 
  (total_cost / carpet_cost_per_meter) / room_length * 100 = 1000 := by
  sorry

#check carpet_width_in_cm

end NUMINAMATH_CALUDE_carpet_width_in_cm_l3126_312621


namespace NUMINAMATH_CALUDE_sports_equipment_purchase_l3126_312606

/-- Represents the cost function for Scheme A -/
def cost_scheme_a (x : ℕ) : ℝ := 25 * x + 550

/-- Represents the cost function for Scheme B -/
def cost_scheme_b (x : ℕ) : ℝ := 22.5 * x + 720

theorem sports_equipment_purchase :
  /- Cost functions are correct -/
  (∀ x : ℕ, x ≥ 10 → cost_scheme_a x = 25 * x + 550 ∧ cost_scheme_b x = 22.5 * x + 720) ∧
  /- Scheme A is more cost-effective for 15 boxes -/
  cost_scheme_a 15 < cost_scheme_b 15 ∧
  /- Scheme A allows purchasing more balls with 1800 yuan budget -/
  (∃ x_a x_b : ℕ, cost_scheme_a x_a ≤ 1800 ∧ cost_scheme_b x_b ≤ 1800 ∧ x_a > x_b) :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_purchase_l3126_312606


namespace NUMINAMATH_CALUDE_bobbys_shoes_cost_l3126_312639

/-- The total cost for Bobby's handmade shoes -/
def total_cost (mold_cost labor_rate hours discount : ℝ) : ℝ :=
  mold_cost + discount * labor_rate * hours

/-- Theorem stating the total cost for Bobby's handmade shoes is $730 -/
theorem bobbys_shoes_cost :
  total_cost 250 75 8 0.8 = 730 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_shoes_cost_l3126_312639


namespace NUMINAMATH_CALUDE_min_workers_theorem_l3126_312691

-- Define the problem parameters
def total_days : ℕ := 40
def days_worked : ℕ := 10
def initial_workers : ℕ := 10
def work_completed : ℚ := 1/4

-- Define the function to calculate the minimum number of workers
def min_workers_needed (total_days : ℕ) (days_worked : ℕ) (initial_workers : ℕ) (work_completed : ℚ) : ℕ :=
  -- Implementation details are not provided in the statement
  sorry

-- Theorem statement
theorem min_workers_theorem :
  min_workers_needed total_days days_worked initial_workers work_completed = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_workers_theorem_l3126_312691


namespace NUMINAMATH_CALUDE_simplest_form_sqrt_l3126_312632

/-- A number is a perfect square if it's the product of an integer with itself -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A square root is in simplest form if it cannot be simplified further -/
def is_simplest_form (n : ℕ) : Prop :=
  ¬(∃ a b : ℕ, n = a * b ∧ is_perfect_square a ∧ b > 1)

/-- The square root of a fraction is in simplest form if it cannot be simplified further -/
def is_simplest_form_frac (n d : ℕ) : Prop :=
  ¬(∃ a b c : ℕ, n = a * b ∧ d = a * c ∧ is_perfect_square a ∧ (b > 1 ∨ c > 1))

theorem simplest_form_sqrt :
  is_simplest_form 14 ∧
  ¬is_simplest_form 12 ∧
  ¬is_simplest_form 8 ∧
  ¬is_simplest_form_frac 1 3 :=
sorry

end NUMINAMATH_CALUDE_simplest_form_sqrt_l3126_312632


namespace NUMINAMATH_CALUDE_remaining_distance_calculation_l3126_312662

/-- The remaining distance to travel after four people have traveled part of the way. -/
def remaining_distance (total_distance : ℝ) 
  (amoli_speed1 amoli_time1 amoli_speed2 amoli_time2 : ℝ)
  (anayet_speed1 anayet_time1 anayet_speed2 anayet_time2 : ℝ)
  (bimal_speed1 bimal_time1 bimal_speed2 bimal_time2 : ℝ)
  (chandni_distance : ℝ) : ℝ :=
  total_distance - 
  (amoli_speed1 * amoli_time1 + amoli_speed2 * amoli_time2 +
   anayet_speed1 * anayet_time1 + anayet_speed2 * anayet_time2 +
   bimal_speed1 * bimal_time1 + bimal_speed2 * bimal_time2 +
   chandni_distance)

/-- Theorem stating the remaining distance to travel. -/
theorem remaining_distance_calculation : 
  remaining_distance 1475 42 3.5 38 2 61 2.5 75 1.5 55 4 30 2 35 = 672 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_calculation_l3126_312662


namespace NUMINAMATH_CALUDE_no_integer_solution_l3126_312694

theorem no_integer_solution :
  ∀ (x y z : ℤ), x ≠ 0 → 2 * x^4 + 2 * x^2 * y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3126_312694


namespace NUMINAMATH_CALUDE_spinner_probability_l3126_312630

/- Define an isosceles triangle with the given angle property -/
structure IsoscelesTriangle where
  baseAngle : ℝ
  vertexAngle : ℝ
  isIsosceles : baseAngle = 2 * vertexAngle

/- Define the division of the triangle into regions by altitudes -/
def triangleRegions : ℕ := 6

/- Define the number of shaded regions -/
def shadedRegions : ℕ := 4

/- Define the probability of landing in a shaded region -/
def shadedProbability (t : IsoscelesTriangle) : ℚ :=
  shadedRegions / triangleRegions

/- Theorem statement -/
theorem spinner_probability (t : IsoscelesTriangle) :
  shadedProbability t = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3126_312630


namespace NUMINAMATH_CALUDE_program_flowchart_components_l3126_312652

-- Define a program flowchart
structure ProgramFlowchart where
  is_diagram : Bool
  represents_algorithm : Bool
  uses_specified_shapes : Bool
  uses_directional_lines : Bool
  uses_textual_explanations : Bool

-- Define the components of a program flowchart
structure FlowchartComponents where
  has_operation_boxes : Bool
  has_flow_lines_with_arrows : Bool
  has_textual_explanations : Bool

-- Theorem statement
theorem program_flowchart_components 
  (pf : ProgramFlowchart) 
  (h1 : pf.is_diagram = true)
  (h2 : pf.represents_algorithm = true)
  (h3 : pf.uses_specified_shapes = true)
  (h4 : pf.uses_directional_lines = true)
  (h5 : pf.uses_textual_explanations = true) :
  ∃ (fc : FlowchartComponents), 
    fc.has_operation_boxes = true ∧ 
    fc.has_flow_lines_with_arrows = true ∧ 
    fc.has_textual_explanations = true :=
  sorry

end NUMINAMATH_CALUDE_program_flowchart_components_l3126_312652


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l3126_312664

/-- A real polynomial of degree 3 -/
structure Polynomial3 where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of the polynomial at a point x -/
def Polynomial3.eval (p : Polynomial3) (x : ℝ) : ℝ :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d

/-- The condition that |p(x)| ≤ 1 for all x such that |x| ≤ 1 -/
def BoundedOnUnitInterval (p : Polynomial3) : Prop :=
  ∀ x : ℝ, |x| ≤ 1 → |p.eval x| ≤ 1

/-- The theorem statement -/
theorem polynomial_coefficient_bound (p : Polynomial3) 
  (h : BoundedOnUnitInterval p) : 
  |p.a| + |p.b| + |p.c| + |p.d| ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l3126_312664


namespace NUMINAMATH_CALUDE_bob_distance_at_meeting_l3126_312611

/-- The distance between point X and point Y in miles -/
def total_distance : ℝ := 52

/-- Yolanda's walking speed in miles per hour -/
def yolanda_speed : ℝ := 3

/-- Bob's walking speed in miles per hour -/
def bob_speed : ℝ := 4

/-- The time difference between Yolanda's and Bob's start in hours -/
def time_difference : ℝ := 1

/-- The theorem stating that Bob walked 28 miles when they met -/
theorem bob_distance_at_meeting : 
  ∃ (t : ℝ), t > 0 ∧ yolanda_speed * (t + time_difference) + bob_speed * t = total_distance ∧ 
  bob_speed * t = 28 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_at_meeting_l3126_312611


namespace NUMINAMATH_CALUDE_otimes_composition_l3126_312626

-- Define the new operation
def otimes (x y : ℝ) : ℝ := x^2 + y^2

-- State the theorem
theorem otimes_composition (x : ℝ) : otimes x (otimes x x) = x^2 + 4*x^4 := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l3126_312626


namespace NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l3126_312678

theorem no_positive_a_satisfies_inequality : 
  ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l3126_312678


namespace NUMINAMATH_CALUDE_bean_sprouts_and_dried_tofu_problem_l3126_312698

/-- Bean sprouts and dried tofu problem -/
theorem bean_sprouts_and_dried_tofu_problem 
  (bean_sprouts_price dried_tofu_price : ℚ)
  (bean_sprouts_sell_price dried_tofu_sell_price : ℚ)
  (total_units : ℕ)
  (max_cost : ℚ) :
  bean_sprouts_price = 60 →
  dried_tofu_price = 40 →
  bean_sprouts_sell_price = 80 →
  dried_tofu_sell_price = 55 →
  total_units = 200 →
  max_cost = 10440 →
  2 * bean_sprouts_price + 3 * dried_tofu_price = 240 →
  3 * bean_sprouts_price + 4 * dried_tofu_price = 340 →
  ∃ (bean_sprouts_units dried_tofu_units : ℕ),
    bean_sprouts_units + dried_tofu_units = total_units ∧
    bean_sprouts_price * bean_sprouts_units + dried_tofu_price * dried_tofu_units ≤ max_cost ∧
    (bean_sprouts_units : ℚ) ≥ (3/2) * dried_tofu_units ∧
    bean_sprouts_units = 122 ∧
    dried_tofu_units = 78 ∧
    (bean_sprouts_sell_price - bean_sprouts_price) * bean_sprouts_units +
    (dried_tofu_sell_price - dried_tofu_price) * dried_tofu_units = 3610 ∧
    ∀ (other_bean_sprouts_units other_dried_tofu_units : ℕ),
      other_bean_sprouts_units + other_dried_tofu_units = total_units →
      bean_sprouts_price * other_bean_sprouts_units + dried_tofu_price * other_dried_tofu_units ≤ max_cost →
      (other_bean_sprouts_units : ℚ) ≥ (3/2) * other_dried_tofu_units →
      (bean_sprouts_sell_price - bean_sprouts_price) * other_bean_sprouts_units +
      (dried_tofu_sell_price - dried_tofu_price) * other_dried_tofu_units ≤ 3610 :=
by
  sorry

end NUMINAMATH_CALUDE_bean_sprouts_and_dried_tofu_problem_l3126_312698


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3126_312608

theorem rectangular_box_surface_area
  (x y z : ℝ)
  (h1 : 4 * x + 4 * y + 4 * z = 240)
  (h2 : Real.sqrt (x^2 + y^2 + z^2) = 31) :
  2 * (x * y + y * z + z * x) = 2639 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3126_312608


namespace NUMINAMATH_CALUDE_haley_facebook_pictures_l3126_312634

/-- The number of pictures Haley uploaded to Facebook -/
def total_pictures : ℕ := 65

/-- The number of pictures in the first album -/
def first_album_pictures : ℕ := 17

/-- The number of additional albums -/
def additional_albums : ℕ := 6

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 8

/-- Theorem stating the total number of pictures uploaded to Facebook -/
theorem haley_facebook_pictures :
  total_pictures = first_album_pictures + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_haley_facebook_pictures_l3126_312634


namespace NUMINAMATH_CALUDE_mark_chocolates_proof_l3126_312653

/-- The number of chocolates Mark started with --/
def initial_chocolates : ℕ := 104

/-- The number of chocolates Mark's sister took --/
def sister_chocolates : ℕ → Prop := λ x => 5 ≤ x ∧ x ≤ 10

theorem mark_chocolates_proof :
  ∃ (sister_took : ℕ),
    sister_chocolates sister_took ∧
    (initial_chocolates / 4 : ℚ) * 3 / 3 * 2 - 40 - sister_took = 4 ∧
    initial_chocolates % 4 = 0 ∧
    initial_chocolates % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_mark_chocolates_proof_l3126_312653


namespace NUMINAMATH_CALUDE_readers_of_both_l3126_312625

theorem readers_of_both (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 150) 
  (h2 : science_fiction = 120) 
  (h3 : literary = 90) :
  science_fiction + literary - total = 60 := by
  sorry

end NUMINAMATH_CALUDE_readers_of_both_l3126_312625


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3126_312685

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3126_312685


namespace NUMINAMATH_CALUDE_largest_common_value_proof_l3126_312665

/-- First arithmetic progression with initial term 4 and common difference 5 -/
def seq1 (n : ℕ) : ℕ := 4 + 5 * n

/-- Second arithmetic progression with initial term 5 and common difference 8 -/
def seq2 (m : ℕ) : ℕ := 5 + 8 * m

/-- The largest common value less than 1000 in both sequences -/
def largest_common_value : ℕ := 989

theorem largest_common_value_proof :
  (∃ n m : ℕ, seq1 n = largest_common_value ∧ seq2 m = largest_common_value) ∧
  (∀ k : ℕ, k < 1000 → (∃ n m : ℕ, seq1 n = k ∧ seq2 m = k) → k ≤ largest_common_value) :=
sorry

end NUMINAMATH_CALUDE_largest_common_value_proof_l3126_312665


namespace NUMINAMATH_CALUDE_emily_calculation_l3126_312627

theorem emily_calculation (n : ℕ) (h : n = 50) : n^2 - 99 = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_emily_calculation_l3126_312627


namespace NUMINAMATH_CALUDE_definite_integral_3x_minus_sinx_l3126_312646

theorem definite_integral_3x_minus_sinx : 
  ∫ x in (0)..(π/2), (3*x - Real.sin x) = 3*π^2/8 - 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_3x_minus_sinx_l3126_312646


namespace NUMINAMATH_CALUDE_test_score_proof_l3126_312617

theorem test_score_proof (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 30 →
  correct_points = 20 →
  incorrect_points = 5 →
  total_score = 325 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 19 :=
by sorry

end NUMINAMATH_CALUDE_test_score_proof_l3126_312617


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3126_312613

/-- Given two polynomials (7x^2 + 5) and (3x^3 + 2x + 1), their product is equal to 21x^5 + 29x^3 + 7x^2 + 10x + 5 -/
theorem polynomial_product_expansion (x : ℝ) : 
  (7 * x^2 + 5) * (3 * x^3 + 2 * x + 1) = 21 * x^5 + 29 * x^3 + 7 * x^2 + 10 * x + 5 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_product_expansion_l3126_312613


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3126_312600

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (1, x) (-2, 3) → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3126_312600


namespace NUMINAMATH_CALUDE_max_consecutive_special_is_correct_l3126_312659

/-- A number is special if it's a 20-digit number that cannot be represented
    as a product of a 10-digit number and an 11-digit number. -/
def IsSpecial (n : ℕ) : Prop :=
  10^19 ≤ n ∧ n < 10^20 ∧
  ∀ a b : ℕ, 10^9 ≤ a ∧ a < 10^10 → 10^10 ≤ b ∧ b < 10^11 → n ≠ a * b

/-- The maximum quantity of consecutive special numbers -/
def MaxConsecutiveSpecial : ℕ := 10^9 - 1

/-- Theorem stating that MaxConsecutiveSpecial is indeed the maximum
    quantity of consecutive special numbers -/
theorem max_consecutive_special_is_correct :
  (∀ k : ℕ, k < MaxConsecutiveSpecial →
    ∀ i : ℕ, i < k → IsSpecial (10^19 + i + 1)) ∧
  (∀ k : ℕ, k > MaxConsecutiveSpecial →
    ∃ i j : ℕ, i < j ∧ j - i = k ∧ ¬IsSpecial j) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_special_is_correct_l3126_312659


namespace NUMINAMATH_CALUDE_book_discount_percentage_l3126_312601

theorem book_discount_percentage (marked_price : ℝ) (cost_price : ℝ) (selling_price : ℝ) :
  cost_price = 0.64 * marked_price →
  (selling_price - cost_price) / cost_price = 0.375 →
  (marked_price - selling_price) / marked_price = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_book_discount_percentage_l3126_312601


namespace NUMINAMATH_CALUDE_brothers_identity_l3126_312696

-- Define the two brothers
inductive Brother
| Tweedledum
| Tweedledee

-- Define a function to represent a brother's statement
def statement (b : Brother) : Brother :=
  match b with
  | Brother.Tweedledum => Brother.Tweedledum
  | Brother.Tweedledee => Brother.Tweedledee

-- Define the consistency of statements
def consistent (first second : Brother) : Prop :=
  (statement first = Brother.Tweedledum) ∧ (statement second = Brother.Tweedledee)

-- Theorem: The only consistent scenario is when both brothers tell the truth
theorem brothers_identity :
  ∀ (first second : Brother),
    consistent first second →
    (first = Brother.Tweedledum ∧ second = Brother.Tweedledee) :=
by sorry


end NUMINAMATH_CALUDE_brothers_identity_l3126_312696


namespace NUMINAMATH_CALUDE_parallelogram_area_l3126_312644

theorem parallelogram_area (base height : ℝ) (h1 : base = 26) (h2 : height = 16) : 
  base * height = 416 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3126_312644


namespace NUMINAMATH_CALUDE_sin_two_x_value_l3126_312648

theorem sin_two_x_value (x : ℝ) 
  (h : Real.sin (Real.pi + x) + Real.sin ((3 * Real.pi) / 2 + x) = 1/2) : 
  Real.sin (2 * x) = -(3/4) := by
  sorry

end NUMINAMATH_CALUDE_sin_two_x_value_l3126_312648


namespace NUMINAMATH_CALUDE_expression_evaluation_l3126_312612

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 8)
  (h2 : b = a + 3)
  (h3 : a = 2)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 5 ≠ 0) :
  ((a + 3) / (a + 1)) * ((b - 1) / (b - 3)) * ((c + 7) / (c + 5)) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3126_312612


namespace NUMINAMATH_CALUDE_f_5_equals_102_l3126_312674

def f (x y : ℝ) : ℝ := 2 * x^2 + y

theorem f_5_equals_102 (y : ℝ) (some_value : ℝ) :
  f some_value y = 60 →
  f 5 y = 102 →
  f 5 y = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_102_l3126_312674


namespace NUMINAMATH_CALUDE_race_distance_proof_l3126_312668

/-- The total distance of a race where:
  * A covers the distance in 20 seconds
  * B covers the distance in 25 seconds
  * A beats B by 14 meters
-/
def race_distance : ℝ := 56

/-- A's time to complete the race in seconds -/
def time_A : ℝ := 20

/-- B's time to complete the race in seconds -/
def time_B : ℝ := 25

/-- The distance by which A beats B in meters -/
def beat_distance : ℝ := 14

theorem race_distance_proof :
  race_distance = (time_B * beat_distance) / (time_B / time_A - 1) :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l3126_312668


namespace NUMINAMATH_CALUDE_remainder_7531_mod_11_l3126_312629

def digit_sum (n : ℕ) : ℕ := sorry

theorem remainder_7531_mod_11 :
  ∃ k : ℤ, 7531 = 11 * k + 5 :=
by
  have h1 : ∀ n : ℕ, ∃ k : ℤ, n = 11 * k + (digit_sum n % 11) := sorry
  sorry

end NUMINAMATH_CALUDE_remainder_7531_mod_11_l3126_312629


namespace NUMINAMATH_CALUDE_simplify_expressions_l3126_312633

variable (a b : ℝ)

theorem simplify_expressions :
  (-2 * a * b - a^2 + 3 * a * b - 5 * a^2 = a * b - 6 * a^2) ∧
  ((4 * a * b - b^2) - 2 * (a^2 + 2 * a * b - b^2) = b^2 - 2 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3126_312633


namespace NUMINAMATH_CALUDE_max_perimeter_of_rectangle_with_area_36_exists_rectangle_with_area_36_and_perimeter_74_l3126_312658

-- Define a rectangle with integer side lengths
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.length * r.width

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Theorem: The maximum perimeter of a rectangle with integer side lengths and area 36 is 74
theorem max_perimeter_of_rectangle_with_area_36 :
  ∀ r : Rectangle, area r = 36 → perimeter r ≤ 74 :=
by
  sorry

-- Theorem: There exists a rectangle with integer side lengths, area 36, and perimeter 74
theorem exists_rectangle_with_area_36_and_perimeter_74 :
  ∃ r : Rectangle, area r = 36 ∧ perimeter r = 74 :=
by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_rectangle_with_area_36_exists_rectangle_with_area_36_and_perimeter_74_l3126_312658


namespace NUMINAMATH_CALUDE_number_division_theorem_l3126_312645

theorem number_division_theorem : 
  ∃ (n : ℕ), (n : ℝ) / 189 = 18.444444444444443 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_number_division_theorem_l3126_312645


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3126_312688

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 - x) ↔ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3126_312688


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3126_312610

/-- A function f: ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The condition p: f(x) = x³ + 2x² + mx + 1 is monotonically increasing -/
def p (m : ℝ) : Prop :=
  MonotonicallyIncreasing (fun x => x^3 + 2*x^2 + m*x + 1)

/-- The condition q: m ≥ 8x / (x² + 4) holds for any x > 0 -/
def q (m : ℝ) : Prop :=
  ∀ x, x > 0 → m ≥ 8*x / (x^2 + 4)

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ m, q m → p m) ∧ (∃ m, p m ∧ ¬q m) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3126_312610


namespace NUMINAMATH_CALUDE_remaining_files_count_l3126_312654

def initial_music_files : ℕ := 4
def initial_video_files : ℕ := 21
def initial_document_files : ℕ := 12
def initial_photo_files : ℕ := 30
def initial_app_files : ℕ := 7

def deleted_video_files : ℕ := 15
def deleted_document_files : ℕ := 10
def deleted_photo_files : ℕ := 18
def deleted_app_files : ℕ := 3

theorem remaining_files_count :
  initial_music_files +
  (initial_video_files - deleted_video_files) +
  (initial_document_files - deleted_document_files) +
  (initial_photo_files - deleted_photo_files) +
  (initial_app_files - deleted_app_files) = 28 := by
  sorry

end NUMINAMATH_CALUDE_remaining_files_count_l3126_312654


namespace NUMINAMATH_CALUDE_spinner_probability_l3126_312687

structure GameBoard where
  regions : Nat
  shaded_regions : Nat
  is_square : Bool
  equal_probability : Bool

def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.regions

theorem spinner_probability (board : GameBoard) 
  (h1 : board.regions = 4)
  (h2 : board.shaded_regions = 3)
  (h3 : board.is_square = true)
  (h4 : board.equal_probability = true) :
  probability_shaded board = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3126_312687


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3126_312679

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3126_312679


namespace NUMINAMATH_CALUDE_fraction_equality_l3126_312636

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * b - b^2

-- Define the # operation
def hash_op (a b : ℝ) : ℝ := a + b - a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 6 2) / (hash_op 6 2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3126_312636


namespace NUMINAMATH_CALUDE_exists_number_satisfying_condition_l3126_312697

theorem exists_number_satisfying_condition : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_condition_l3126_312697


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l3126_312623

theorem stratified_sampling_male_count :
  let total_athletes : ℕ := 32 + 24
  let male_athletes : ℕ := 32
  let sample_size : ℕ := 14
  let male_sample : ℕ := (male_athletes * sample_size) / total_athletes
  male_sample = 8 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l3126_312623


namespace NUMINAMATH_CALUDE_min_distance_a_c_l3126_312614

/-- Given vectors a and b in ℝ² satisfying the specified conditions,
    prove that the minimum distance between a and c is (√7 - √2) / 2 -/
theorem min_distance_a_c (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = 1)
  (h3 : a • b = 1)
  : (∀ c : ℝ × ℝ, (a - 2 • c) • (b - c) = 0 → 
    ‖a - c‖ ≥ (Real.sqrt 7 - Real.sqrt 2) / 2) ∧ 
    (∃ c : ℝ × ℝ, (a - 2 • c) • (b - c) = 0 ∧ 
    ‖a - c‖ = (Real.sqrt 7 - Real.sqrt 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_min_distance_a_c_l3126_312614


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3126_312699

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3126_312699


namespace NUMINAMATH_CALUDE_square_root_problem_l3126_312640

theorem square_root_problem (a : ℝ) (n : ℝ) (hn : n > 0) :
  (2 * a - 3)^2 = n ∧ (3 * a - 22)^2 = n → n = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3126_312640


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3126_312683

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) -- a is the sequence
  (S : ℕ → ℝ) -- S is the sum function
  (h1 : ∀ n, S n = 3^n - 1) -- Given condition
  (h2 : ∀ n, S n = S (n-1) + a n) -- Property of sum of sequences
  : ∀ n, a n = 2 * 3^(n-1) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3126_312683


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l3126_312643

/-- Represents the number of available colors for triangles -/
def total_colors : ℕ := 8

/-- Represents the number of colors available for corner triangles -/
def corner_colors : ℕ := total_colors - 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def distinguishable_triangles : ℕ :=
  corner_colors +  -- All corners same color
  (corner_colors * (corner_colors - 1)) +  -- Two corners same color
  choose corner_colors 3  -- All corners different colors

theorem distinguishable_triangles_count :
  distinguishable_triangles = 84 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l3126_312643


namespace NUMINAMATH_CALUDE_hexagon_area_lower_bound_l3126_312666

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Area of the hexagon formed by extending the sides of the triangle -/
def hexagon_area (t : Triangle) : ℝ := sorry

/-- The area of the hexagon is at least 13 times the area of the triangle -/
theorem hexagon_area_lower_bound (t : Triangle) :
  hexagon_area t ≥ 13 * area t := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_lower_bound_l3126_312666


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3126_312604

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| ≥ 1} = {x : ℝ | x ≤ -2 ∨ x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3126_312604


namespace NUMINAMATH_CALUDE_initial_candies_count_l3126_312650

-- Define the given conditions
def candies_given_to_chloe : ℝ := 28.0
def candies_left : ℕ := 6

-- Define the theorem to prove
theorem initial_candies_count : 
  candies_given_to_chloe + candies_left = 34.0 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_count_l3126_312650


namespace NUMINAMATH_CALUDE_buses_in_five_days_l3126_312609

/-- Represents the number of buses leaving a station over multiple days -/
def buses_over_days (buses_per_half_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  buses_per_half_hour * 2 * hours_per_day * days

/-- Theorem stating that 120 buses leave the station over 5 days -/
theorem buses_in_five_days :
  buses_over_days 1 12 5 = 120 := by
  sorry

#eval buses_over_days 1 12 5

end NUMINAMATH_CALUDE_buses_in_five_days_l3126_312609


namespace NUMINAMATH_CALUDE_percentage_calculation_l3126_312686

def total_population : ℕ := 40000
def part_population : ℕ := 36000

theorem percentage_calculation : 
  (part_population : ℚ) / (total_population : ℚ) * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3126_312686


namespace NUMINAMATH_CALUDE_no_valid_coloring_l3126_312689

theorem no_valid_coloring : ¬∃ (f : ℕ+ → Bool), 
  (∀ n : ℕ+, f n ≠ f (n + 5)) ∧ 
  (∀ n : ℕ+, f n ≠ f (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_coloring_l3126_312689


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l3126_312631

theorem no_solutions_for_equation :
  ∀ (x y : ℝ), x^2 + y^2 - 2*y + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l3126_312631


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l3126_312673

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Theorem for the minimum value when a = 1
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 4 ∧ ∃ y : ℝ, f 1 y = 4 :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4/a + 1) ↔ (a < 0 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_l3126_312673


namespace NUMINAMATH_CALUDE_new_average_production_l3126_312602

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) 
  (h1 : n = 10)
  (h2 : past_avg = 50)
  (h3 : today_prod = 105) :
  (n * past_avg + today_prod) / (n + 1) = 55 := by
sorry

end NUMINAMATH_CALUDE_new_average_production_l3126_312602


namespace NUMINAMATH_CALUDE_erased_odd_number_l3126_312684

theorem erased_odd_number (n : ℕ) (erased : ℕ) :
  (∃ k, n = k^2) ∧
  (∃ m, erased = 2*m - 1) ∧
  (n^2 - erased = 2008) →
  erased = 17 := by
sorry

end NUMINAMATH_CALUDE_erased_odd_number_l3126_312684


namespace NUMINAMATH_CALUDE_hexagon_triangles_l3126_312670

/-- The number of triangles that can be formed from a regular hexagon and its center -/
def num_triangles_hexagon : ℕ :=
  let total_points : ℕ := 7
  let total_combinations : ℕ := Nat.choose total_points 3
  let invalid_triangles : ℕ := 3
  total_combinations - invalid_triangles

theorem hexagon_triangles :
  num_triangles_hexagon = 32 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_triangles_l3126_312670


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3126_312637

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ 2*x^2 - 3*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 17) / 4 ∧ 
             x₂ = (3 - Real.sqrt 17) / 4 ∧ 
             f x₁ = 0 ∧ f x₂ = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3126_312637


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l3126_312651

/-- Hyperbola C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 2 - y^2 / 8 = 1

/-- Hyperbola C₂ -/
def C₂ (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Asymptote of C₁ -/
def asymptote_C₁ (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Asymptote of C₂ -/
def asymptote_C₂ (x y a b : ℝ) : Prop := y = (b / a) * x ∨ y = -(b / a) * x

theorem hyperbola_b_value (a b : ℝ) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_same_asymptotes : ∀ x y, asymptote_C₁ x y ↔ asymptote_C₂ x y a b)
  (h_focal_length : 4 * Real.sqrt 5 = 2 * Real.sqrt (a^2 + b^2)) :
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l3126_312651
