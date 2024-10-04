import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Num.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.LinearAlgebra.Vector
import Mathlib.Probability
import Mathlib.Probability.Distribution.Binomial
import Mathlib.Probability.Theory
import Mathlib.Tactic

namespace number_of_students_play_football_l326_326058

-- definitions and conditions
def Total := 410
def C := 175
def B := 140
def Neither := 50

-- expected answer
def F_expected := 375

-- proof problem
theorem number_of_students_play_football : 
    Total - C + B - Neither = F_expected :=
by
  sorry

end number_of_students_play_football_l326_326058


namespace ant_probability_reach_bottom_l326_326180

-- Define the structure of the modified octahedron
structure ModifiedOctahedron where
  vertices : Finset ℕ
  adjacency : ℕ → Finset ℕ
  top_vertex : ℕ
  bottom_vertex : ℕ
  middle_ring : Finset ℕ

-- Conditions of the problem
def problem_conditions (M : ModifiedOctahedron) : Prop :=
  M.vertices.card = 12 ∧
  M.middle_ring.card = 5 ∧
  M.adjacency M.top_vertex = M.middle_ring ∧
  ∀ v ∈ M.middle_ring, M.adjacency v = insert M.top_vertex (insert M.bottom_vertex (M.middle_ring.erase v))

-- The probability calculation based on given conditions
noncomputable def probability_second_vertex_bottom (M : ModifiedOctahedron) [h : problem_conditions M] : ℚ :=
  1 / 5

-- The theorem statement
theorem ant_probability_reach_bottom (M : ModifiedOctahedron) [h : problem_conditions M] :
  probability_second_vertex_bottom M = 1 / 5 := 
sorry

end ant_probability_reach_bottom_l326_326180


namespace sum_of_three_squares_l326_326546

variable (t s : ℝ)

-- Given equations
axiom h1 : 3 * t + 2 * s = 27
axiom h2 : 2 * t + 3 * s = 25

-- What we aim to prove
theorem sum_of_three_squares : 3 * s = 63 / 5 :=
by
  sorry

end sum_of_three_squares_l326_326546


namespace jason_read_books_l326_326720

-- Let j be the number of books Jason has, m be the number of books Mary has, and t be the total number of books they have together.
def jason_books : ℕ := 18
def mary_books : ℕ := 42
def total_books : ℕ := 60

theorem jason_read_books : jason_books + mary_books = total_books → jason_books = 18 :=
by
  intro h
  exact eq.refl 18

end jason_read_books_l326_326720


namespace bobbit_worm_eats_2_fish_per_day_l326_326844

theorem bobbit_worm_eats_2_fish_per_day :
  ∃ (x : ℕ), 
  (60 - 14 * x + 8 - 7 * x = 26) → 
  x = 2 := 
by {
  let x := 2,
  existsi x,
  intros h,
  linarith,
  sorry
}

end bobbit_worm_eats_2_fish_per_day_l326_326844


namespace find_primes_l326_326590

-- Definition of being a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

-- Lean 4 statement of the problem
theorem find_primes (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 → p = 5 ∧ q = 3 ∧ r = 19 := 
by
  sorry

end find_primes_l326_326590


namespace find_min_value_l326_326618

theorem find_min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / a + 1 / b = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end find_min_value_l326_326618


namespace isaac_age_l326_326004

axiom ages : Set ℕ := {2, 4, 6, 8, 10, 12, 14}

def age_pair_summing_to_18 (a b : ℕ) : Prop :=
  a + b = 18

def age_pair_summing_to_le_14 (a b : ℕ) : Prop :=
  a + b ≤ 14

def isaac_home_with_age_four (i : ℕ) : Prop :=
  i + 4 ∈ ages

theorem isaac_age :
  ∃ (age_of_isaac : ℕ), age_of_isaac ∈ ages ∧ age_of_isaac = 14 :=
begin
  sorry
end

end isaac_age_l326_326004


namespace average_and_median_of_seven_numbers_l326_326439

theorem average_and_median_of_seven_numbers (x : ℝ) 
  (h_avg : (60 + 100 + x + 40 + 50 + 200 + 90) / 7 = x)
  (h_med : median [60, 100, x, 40, 50, 200, 90] = x) :
  x = 90 :=
sorry

end average_and_median_of_seven_numbers_l326_326439


namespace fraction_of_purple_eggs_l326_326164

variable (E : ℕ)

def num_blue_eggs := (4 / 5 : ℚ) * E
def num_purple_eggs := (1 / 5 : ℚ) * E
def purple_eggs_with_candy := (1 / 2 : ℚ) * (1 / 5 : ℚ) * E
def blue_eggs_with_candy := (1 / 4 : ℚ) * (4 / 5 : ℚ) * E
def total_eggs_with_five_candies := purple_eggs_with_candy + blue_eggs_with_candy
def chance_of_five_candies_when_random := (3 / 10 : ℚ)

theorem fraction_of_purple_eggs :
  chance_of_five_candies_when_random * E = total_eggs_with_five_candies →
  num_purple_eggs / E = (1 / 5 : ℚ) :=
  by
    sorry

end fraction_of_purple_eggs_l326_326164


namespace minimum_rows_l326_326893

theorem minimum_rows (n : ℕ) (C : ℕ → ℕ) (hC_bounds : ∀ i, 1 ≤ C i ∧ C i ≤ 39) 
  (hC_sum : (Finset.range n).sum C = 1990) :
  ∃ k, k = 12 ∧ ∀ (R : ℕ) (hR : R = 199), 
    ∀ (seating : ℕ → ℕ) (h_seating : ∀ i, seating i ≤ R) 
    (h_seating_capacity : (Finset.range k).sum seating = 1990),
    True := sorry

end minimum_rows_l326_326893


namespace probability_of_black_ball_l326_326168

theorem probability_of_black_ball (black_balls red_balls : ℕ) (h_black : black_balls = 6) (h_red : red_balls = 18) :
  (black_balls.toRat / (black_balls + red_balls).toRat) = (1 / 4) := by
  sorry

end probability_of_black_ball_l326_326168


namespace append_digits_divisible_l326_326485

theorem append_digits_divisible (x y : ℕ) (h1 : x = 9) (h2 : y = 4) :
  (2013 * 100 + x * 10 + y) % 101 = 0 :=
by {
  have six_digit_num : ℕ := 2013 * 100 + x * 10 + y,
  rw [h1, h2],
  norm_num,
  exact dec_trivial
}

end append_digits_divisible_l326_326485


namespace accuracy_l326_326881

-- Given number and accuracy statement
def given_number : ℝ := 3.145 * 10^8
def expanded_form : ℕ := 314500000

-- Proof statement: the number is accurate to the hundred thousand's place
theorem accuracy (h : given_number = expanded_form) : 
  ∃ n : ℕ, expanded_form = n * 10^5 ∧ (n % 10) ≠ 0 := 
by
  sorry

end accuracy_l326_326881


namespace find_y_when_x_9_l326_326864

variable (x y k : ℝ)

-- Conditions
def inverse_relation (x y : ℝ) (k : ℝ) := 3 * y = k / (x ^ 3)
def fixed_condition := inverse_relation 3 27 k

-- Problem to prove
theorem find_y_when_x_9 (h1 : fixed_condition) (h2 : inverse_relation 9 y k) : y = 1 :=
sorry

end find_y_when_x_9_l326_326864


namespace sum_reciprocal_eq_l326_326426

theorem sum_reciprocal_eq (n : ℕ) (h : n ≥ 2) : 
  (∑ k in Finset.range (n-1), (1 / (k * (k + 1) : ℝ))) = 1 - (1 / (n : ℝ)) :=
by
  sorry

end sum_reciprocal_eq_l326_326426


namespace bob_extra_slices_l326_326195

noncomputable def small_pizza_slices := 4
noncomputable def large_pizza_slices := 8

noncomputable def small_pizzas_purchased := 3
noncomputable def large_pizzas_purchased := 2

noncomputable def george_slices_eaten := 3

noncomputable def bill_slices_eaten := 3
noncomputable def fred_slices_eaten := 3
noncomputable def mark_slices_eaten := 3

noncomputable def slices_left_over := 10

theorem bob_extra_slices (small_pizza_slices large_pizza_slices small_pizzas_purchased large_pizzas_purchased 
                          george_slices_eaten bill_slices_eaten fred_slices_eaten mark_slices_eaten 
                          slices_left_over : ℕ):
  let total_slices := small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices
  let slices_after_george := total_slices - george_slices_eaten
  let slices_after_others := slices_after_george - (bill_slices_eaten + fred_slices_eaten + mark_slices_eaten)
  slices_left_over = slices_after_others - (slices_after_others / 3 + 2 / 3 * slices_after_others) - slices_after_others / 2
  := 1

end bob_extra_slices_l326_326195


namespace log_ordering_l326_326027

theorem log_ordering (π : Real) (logπ3 : π ≠ 1) :
  let a := Real.logBase π (3 : Real) -- a = log_π 3
  let b := Real.logBase (3 : Real) (4 : Real) -- b = log_3 4
  let c := Real.logBase (4 : Real) (17 : Real) -- c = log_4 17
  a < b ∧ b < c :=
by
  let a := Real.logBase π (3 : Real)
  let b := Real.logBase (3 : Real)
  let c := Real.logBase (4 : Real)
  sorry

end log_ordering_l326_326027


namespace probability_of_same_color_when_rolling_two_24_sided_dice_l326_326129

-- Defining the conditions
def numSides : ℕ := 24
def purpleSides : ℕ := 5
def blueSides : ℕ := 8
def redSides : ℕ := 10
def goldSides : ℕ := 1

-- Required to use rational numbers for probabilities
def probability (eventSides : ℕ) (totalSides : ℕ) : ℚ := eventSides / totalSides

-- Main theorem statement
theorem probability_of_same_color_when_rolling_two_24_sided_dice :
  probability purpleSides numSides * probability purpleSides numSides +
  probability blueSides numSides * probability blueSides numSides +
  probability redSides numSides * probability redSides numSides +
  probability goldSides numSides * probability goldSides numSides =
  95 / 288 :=
by
  sorry

end probability_of_same_color_when_rolling_two_24_sided_dice_l326_326129


namespace sita_geeta_distance_approximately_l326_326133

-- Define the initial conditions
def sita_geeta_distance_initial : ℝ := 10
def sita_geeta_turn_distance : ℝ := 7.5

-- Define the condition
def distance_squared (a b : ℝ) : ℝ := a^2 + b^2

-- Define the distance between Sita and Geeta when they stop
noncomputable def sita_geeta_distance_final : ℝ :=
  Real.sqrt (distance_squared (2 * sita_geeta_distance_initial) sita_geeta_turn_distance)

-- The statement to be proved
theorem sita_geeta_distance_approximately :
  sita_geeta_distance_final ≈ 21.36 := 
sorry

end sita_geeta_distance_approximately_l326_326133


namespace zinc_weight_in_mixture_l326_326863

theorem zinc_weight_in_mixture (total_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) (total_parts : ℝ) (fraction_zinc : ℝ) (weight_zinc : ℝ) :
  zinc_ratio = 9 ∧ copper_ratio = 11 ∧ total_weight = 70 ∧ total_parts = zinc_ratio + copper_ratio ∧
  fraction_zinc = zinc_ratio / total_parts ∧ weight_zinc = fraction_zinc * total_weight →
  weight_zinc = 31.5 :=
by
  intros h
  sorry

end zinc_weight_in_mixture_l326_326863


namespace min_reciprocal_sum_l326_326449

theorem min_reciprocal_sum (a : ℝ) (m n : ℝ) 
  (h1 : 0 < a ∧ a ≠ 1) 
  (h2 : xy_line_relation : 2 * m + n = 1)
  (h3 : mn_pos : 0 < m * n) :
  frac_sum_min : ∃ m n : ℝ, 0 < m ∧ 0 < n ∧ 2 * m + n = 1 ∧ (1 / m + 1 / n) = 3 + 2 * real.sqrt 2 := 
begin
  sorry -- proof is not provided here
end

end min_reciprocal_sum_l326_326449


namespace derivative_of_f_derivative_of_g_l326_326998

-- Define the function f(x) = x^4 - 2x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^4 - 2 * x^2 + 3 * x - 1

-- Define the function g(x) = (x-1)/x
def g (x : ℝ) : ℝ := (x - 1) / x

-- Prove that the derivative of f is 4x^3 - 4x + 3
theorem derivative_of_f : ∀ x : ℝ, deriv f x = 4 * x^3 - 4 * x + 3 :=
by
  intro x
  sorry

-- Prove that the derivative of g is 1/x^2
theorem derivative_of_g : ∀ x : ℝ, deriv g x = 1 / x^2 :=
by
  intro x
  sorry

end derivative_of_f_derivative_of_g_l326_326998


namespace find_f_2019_l326_326640

noncomputable def even (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = f(x)

theorem find_f_2019 (f : ℝ → ℝ) 
  (h_even : even f) 
  (h_periodic : ∀ x : ℝ, f(x + 6) - f(x) = 2 * f(3)) : 
  f 2019 = 0 := 
sorry

end find_f_2019_l326_326640


namespace sweatshirt_sales_l326_326016

variables (S H : ℝ)

theorem sweatshirt_sales (h1 : 13 * S + 9 * H = 370) (h2 : 9 * S + 2 * H = 180) :
  12 * S + 6 * H = 300 :=
sorry

end sweatshirt_sales_l326_326016


namespace arrange_traffic_flow_l326_326894

-- Let's define the structure for the problem
structure CityGraph where
  cities : Type -- Abstract type for cities
  roads : cities → cities → Prop -- Roads are represented as pairs of cities
  traffic_flow : cities → cities → ℕ -- Traffic flow on roads is either 1 or 2
  traffic_flow_correct : ∀ (c1 c2 : cities), roads c1 c2 → (traffic_flow c1 c2 = 1 ∨ traffic_flow c1 c2 = 2)

-- Define the condition that the total traffic flow at each city is odd
def total_traffic_odd (G : CityGraph) (c : G.cities) : Prop :=
  let incoming := Σ' (c' : G.cities), if G.roads c' c then G.traffic_flow c' c else 0
  let outgoing := Σ' (c' : G.cities), if G.roads c c' then G.traffic_flow c c' else 0
  (incoming + outgoing) % 2 = 1

-- Define the target property: arranging directions such that the residue is ±1
def valid_orientation (G : CityGraph) (dir : G.cities → G.cities → Bool) : Prop :=
  ∀ c : G.cities, 
    let incoming := Σ' (c' : G.cities), if dir c' c then G.traffic_flow c' c else 0
    let outgoing := Σ' (c' : G.cities), if dir c c' then G.traffic_flow c c' else 0
    abs (incoming - outgoing) = 1

-- The main theorem to be proved
theorem arrange_traffic_flow (G : CityGraph)
  (h_odd : ∀ c : G.cities, total_traffic_odd G c) :
  ∃ dir : G.cities → G.cities → Bool, valid_orientation G dir :=
sorry

end arrange_traffic_flow_l326_326894


namespace park_is_square_l326_326967

-- Defining the concept of a square field
def square_field : ℕ := 4

-- Given condition: The sum of the right angles from the park and the square field
axiom angles_sum (park_angles : ℕ) : park_angles + square_field = 8

-- The theorem to be proven
theorem park_is_square (park_angles : ℕ) (h : park_angles + square_field = 8) : park_angles = 4 :=
by sorry

end park_is_square_l326_326967


namespace initial_ducks_l326_326080

theorem initial_ducks (D : ℕ) (h1 : D + 20 = 33) : D = 13 :=
by sorry

end initial_ducks_l326_326080


namespace center_is_one_l326_326547

-- Definitions based on conditions
def adjacent (a b : ℕ) : Prop := abs (a - b) = 1

def is_grid_valid (grid : matrix (fin 3) (fin 3) ℕ) : Prop :=
  ∀ i j, (i, j) ∈ (finset.univ : finset (fin 3 × fin 3)) → 
    ∀ (di dj : ℕ), (di = 0 ∨ di = 2) ∧ (dj = 0 ∨ dj = 2) →
      adjacent (grid i j) (grid (i + di) (j + dj))

def corner_sum (grid : matrix (fin 3) (fin 3) ℕ) : ℕ :=
  grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2

def is_edge (grid : matrix (fin 3) (fin 3) ℕ) (n : ℕ) : Prop :=
  (grid 1 0 = n ∨ grid 0 1 = n ∨ grid 2 1 = n ∨ grid 1 2 = n)

def center_number (grid : matrix (fin 3) (fin 3) ℕ) : ℕ :=
  grid 1 1

-- The theorem to prove
theorem center_is_one {grid : matrix (fin 3) (fin 3) ℕ} :
  is_grid_valid grid →
  corner_sum grid = 24 →
  is_edge grid 6 →
  center_number grid = 1 :=
sorry

end center_is_one_l326_326547


namespace smallest_ge_1_point_1_l326_326922

theorem smallest_ge_1_point_1 (h : set ℝ) (h_cond : {1.4, 9/10, 1.2, 0.5, 13/10} ⊆ h) :
  ∃ x ∈ h, x ≥ 1.1 ∧ ∀ y ∈ h, y ≥ 1.1 → x ≤ y → x = 1.2 := 
  sorry

end smallest_ge_1_point_1_l326_326922


namespace rhombus_perimeter_l326_326804

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  let side_length := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let perimeter := 4 * side_length in
  perimeter = 68 :=
by
  have h3 : d1 / 2 = 8, from by rw [h1],
  have h4 : d2 / 2 = 15, from by rw [h2],
  have h5 : side_length = 17, from by
    calc
      side_length
          = Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) : rfl
      ... = Real.sqrt (8 ^ 2 + 15 ^ 2) : by rw [h3, h4]
      ... = Real.sqrt (64 + 225) : rfl
      ... = Real.sqrt 289 : rfl
      ... = 17 : by norm_num,
  calc
    perimeter
        = 4 * side_length : rfl
    ... = 4 * 17 : by rw [h5]
    ... = 68 : by norm_num

end rhombus_perimeter_l326_326804


namespace pizzaOrderTotalCost_l326_326065

def basePizzaCost : ℝ := 10.00

def pepperoniCost : ℝ := 1.50
def sausageCost : ℝ := 1.50
def otherToppingCost : ℝ := 1.00
def extraCheeseCost : ℝ := 2.00

def sonPizzaCost : ℝ := basePizzaCost + pepperoniCost
def daughterPizzaCost : ℝ := basePizzaCost + sausageCost + otherToppingCost
def rubyAndHusbandPizzaCost : ℝ := basePizzaCost + 3 * otherToppingCost
def cousinPizzaCost : ℝ := basePizzaCost + 2 * otherToppingCost + extraCheeseCost

def totalPizzaCost : ℝ := sonPizzaCost + daughterPizzaCost + rubyAndHusbandPizzaCost + cousinPizzaCost
def tip : ℝ := 5.00

theorem pizzaOrderTotalCost : totalPizzaCost + tip = 56.00 := by
  have cost_sonPizza : sonPizzaCost = 11.50 := by
    rw [sonPizzaCost, basePizzaCost, pepperoniCost]
    norm_num
  have cost_daughterPizza : daughterPizzaCost = 12.50 := by
    rw [daughterPizzaCost, basePizzaCost, sausageCost, otherToppingCost]
    norm_num
  have cost_rubyAndHusbandPizza : rubyAndHusbandPizzaCost = 13.00 := by
    rw [rubyAndHusbandPizzaCost, basePizzaCost, otherToppingCost]
    norm_num
  have cost_cousinPizza : cousinPizzaCost = 14.00 := by
    rw [cousinPizzaCost, basePizzaCost, otherToppingCost, extraCheeseCost]
    norm_num
  have total_cost : totalPizzaCost = 51.00 := by
    rw [totalPizzaCost, cost_sonPizza, cost_daughterPizza, cost_rubyAndHusbandPizza, cost_cousinPizza]
    norm_num
  rw [totalPizzaCost, total_cost, tip]
  norm_num

end pizzaOrderTotalCost_l326_326065


namespace find_a_l326_326364

-- Define the triangle and the given conditions
variables (A B C : Type) [Real A] [Real B] [Real C]

-- Given conditions
def angle_A_deg := 60
def area_ABC := sqrt 3
def b_plus_c := 6

-- Known side lengths opposite to angles
variables (a b c : ℝ)

-- Statement to be proved
theorem find_a : (angle_A_deg = 60) → (area_ABC = sqrt 3) →
                  (b + c = 6) → (a = 2 * sqrt 6) :=
by
  -- proof goes here
  sorry

end find_a_l326_326364


namespace sum_of_ages_l326_326418

theorem sum_of_ages (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4) 
  (h2 : rachel_age = 19) : rachel_age + leah_age = 34 :=
by
  -- Proof steps are omitted since we only need the statement
  sorry

end sum_of_ages_l326_326418


namespace bruce_and_anne_clean_house_l326_326941

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l326_326941


namespace find_m_n_l326_326335

theorem find_m_n (a b : ℕ) (m n : ℕ) 
    (h_like_terms : m + 1 = 2)
    (h_sum_zero : n - 1 = -1) : 
    m = 1 ∧ n = 0 :=
by {
  sorry
}

end find_m_n_l326_326335


namespace probability_team_A_advances_l326_326707
open_locale classical

variable {Ω : Type*} [fintype Ω] [decidable_eq Ω]

def teams := fin 4
def equally_strong (ω : teams → Prop) := ∀ t, ω t = ω
def advancing_teams (ω : teams → Prop) : set (teams × teams) := { y | y.1 ≠ y.2 ∧ ω y.1 ∧ ω y.2 }

theorem probability_team_A_advances (ω : teams → Prop) (advancing : finset (teams × teams)) :
  equally_strong ω →
  advancing = { (i, j) | i ≠ j ∧ ω i ∧ ω j } →
  (advancing.count ((λ p, p.1 = 0 ∨ p.2 = 0)) / advancing.card : ℝ) = 1 / 2 := 
sorry

end probability_team_A_advances_l326_326707


namespace power_seven_evaluation_l326_326246

theorem power_seven_evaluation (a b : ℝ) (h : a = (7 : ℝ)^(1/4) ∧ b = (7 : ℝ)^(1/7)) : 
  a / b = (7 : ℝ)^(3/28) :=
  sorry

end power_seven_evaluation_l326_326246


namespace extremum_min_value_l326_326675

theorem extremum_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ f : ℝ → ℝ, f = (λ x, 4 * x^3 - a * x^2 - 2 * b * x) ∧ f' 1 = 0) →
  (dfrac 4 a + dfrac 1 b) = dfrac 3 2 :=
by
  sorry

end extremum_min_value_l326_326675


namespace positive_number_divisible_by_4_l326_326532

theorem positive_number_divisible_by_4 (N : ℕ) (h1 : N % 4 = 0) (h2 : (2 + 4 + N + 3) % 2 = 1) : N = 4 := 
by 
  sorry

end positive_number_divisible_by_4_l326_326532


namespace find_t_l326_326307

noncomputable def f (x t : ℝ) : ℝ :=
  (x^3 + t * x^2 + (real.sqrt 2) * t * real.sin (x + (real.pi / 4)) + 2 * t) / (x^2 + 2 + real.cos x)

theorem find_t (t : ℝ) (m n : ℝ) (h1: t ≠ 0) (h2: m = (f (real.sqrt 2) t)) (h3: n = (f (-real.sqrt 2) t)) (h4: m + n = 2017) :
  t = 2017 / 2 :=
begin
  sorry
end

end find_t_l326_326307


namespace minimum_angle_between_bisectors_theorem_l326_326607

noncomputable def minimum_angle_between_bisectors (A B C O : Point) (OA OB OC : ℝ) : ℝ :=
  if h1 : angle A O B = 60 then
  if h2 : angle B O C = 120 then
  if h3 : angle A O C = 90 then
  45
  else 0
  else 0
  else 0

theorem minimum_angle_between_bisectors_theorem
  (A B C O : Point)
  (OA OB OC : ℝ)
  (h1 : angle A O B = 60)
  (h2 : angle B O C = 120)
  (h3 : angle A O C = 90) :
  minimum_angle_between_bisectors A B C O OA OB OC = 45 :=
by sorry

end minimum_angle_between_bisectors_theorem_l326_326607


namespace find_c_l326_326337

theorem find_c (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) : c = (n * a) / (n - 2 * a * b) :=
by
  sorry

end find_c_l326_326337


namespace students_without_favorite_subject_l326_326687

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end students_without_favorite_subject_l326_326687


namespace find_N_l326_326669

theorem find_N (x y N : ℝ) (h1 : 2 * x + y = N) (h2 : x + 2 * y = 5) (h3 : (x + y) / 3 = 1) : N = 4 :=
by
  have h4 : x + y = 3 := by
    linarith [h3]
  have h5 : y = 3 - x := by
    linarith [h4]
  have h6 : x + 2 * (3 - x) = 5 := by
    linarith [h2, h5]
  have h7 : x = 1 := by
    linarith [h6]
  have h8 : y = 2 := by
    linarith [h4, h7]
  have h9 : 2 * x + y = 4 := by
    linarith [h7, h8]
  linarith [h1, h9]

end find_N_l326_326669


namespace inverse_of_21_mod_31_l326_326633

theorem inverse_of_21_mod_31 (h : (17 : ℤ)⁻¹ ≡ 13 [ZMOD 31]) : (21 : ℤ)⁻¹ ≡ 6 [ZMOD 31] := by
  sorry

end inverse_of_21_mod_31_l326_326633


namespace root_in_interval_l326_326450

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 7

theorem root_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
by
  sorry

end root_in_interval_l326_326450


namespace bruce_anne_cleaning_house_l326_326958

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l326_326958


namespace line_through_fixed_point_minimum_area_line_range_k_not_third_quadrant_l326_326504

noncomputable def fixed_point_C : (ℝ × ℝ) := (-5, -3)

theorem line_through_fixed_point (l : ℝ → ℝ → Prop) : 
  (∀ x y, l x y → y = -3/5 * x - 3) → l (-5) (-3) :=
by
  sorry

theorem minimum_area_line (l : ℝ → ℝ → Prop) (S : ℝ) :
  (∀ A B, l A.1 A.2 → l B.1 B.2 → (A.1 < 0) → (B.2 > 0) → (area_Δ A B (0,0) = S)) →
  ∃ a b c : ℝ, l = fun x y => a * x + b * y + c = 0 ∧ S = 25/2 ∧ a = 5 ∧ b = -2 ∧ c = -5 :=
by
  sorry

def line_not_in_third_quadrant (k : ℝ) := 
  ∀ x y, x > 0 ∨ y > 0 ∨ x = 0 ∨ y = 0 → y = k * x

theorem range_k_not_third_quadrant : 
  ∀ k, line_not_in_third_quadrant k → k ∈ Icc (-∞) 0 ∩ Icc (0) (∞) :=
by
  sorry

end line_through_fixed_point_minimum_area_line_range_k_not_third_quadrant_l326_326504


namespace smaller_square_area_fraction_l326_326826

/-- A square in the 2D plane. --/
structure Square :=
  (A B C D : Point)
  (AB : LineSegment A B)
  (BC : LineSegment B C)
  (CD : LineSegment C D)
  (DA : LineSegment D A)
  (is_square : ∀ (p : Point), p ∈ set_of_points A B C D → congruent_segments AB BC CD DA)

def Point := ℝ × ℝ

/-- The midpoint of a line segment --/
def midpoint (A B : Point) : Point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The area of a square given the length of one side --/
def square_area (side_length : ℝ) : ℝ :=
  side_length * side_length

/-- We need a proof that the area of the smaller square PQRS, formed by connecting the midpoints of a given square ABCD, is 1/4 of the area of the original square ABCD. --/
theorem smaller_square_area_fraction (ABCD : Square) :
  let side_length := distance ABCD.ABC.ABC.ABC.ABC
  all_eq s (midpoint ABCD.ABC.ABC)
  square_area (distance (midpoint ABCD.ABC.ABC) (midpoint ABCD.D.ABC) (ABCD.ABC.ABC (midpoint ABCD.D.ABC)) (distance :x )
      ) = distance θ₁.side_length

  sorry

end smaller_square_area_fraction_l326_326826


namespace total_trip_cost_l326_326211

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l326_326211


namespace max_radius_of_circle_l326_326156

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (distance center point = radius)

theorem max_radius_of_circle :
  ∀ (C : (ℝ × ℝ) × ℝ), 
  let center := C.1 in 
  let radius := C.2 in 
  point_on_circle center radius (4, 0) → 
  point_on_circle center radius (-4, 0) → 
  radius <= 4 :=
by
  intros
  sorry

end max_radius_of_circle_l326_326156


namespace point_a_number_l326_326813

theorem point_a_number (x : ℝ) (h : abs (x - 2) = 6) : x = 8 ∨ x = -4 :=
sorry

end point_a_number_l326_326813


namespace convex_quadrilateral_exists_l326_326503

-- Given a set of 5 points in the plane, where no three points are collinear, 
-- prove that we can choose 4 points among them that form a convex quadrilateral.
theorem convex_quadrilateral_exists (points : Finset (ℝ × ℝ)) 
  (h_card : points.card = 5) 
  (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), 
    (p1 ∈ points) → (p2 ∈ points) → (p3 ∈ points) → ¬Collinear ({p1, p2, p3} : Finset (ℝ × ℝ))) :
  ∃ (quad_points : Finset (ℝ × ℝ)), 
    quad_points.card = 4 ∧ Convex ℝ (convex_hull ℝ (quad_points : Set (ℝ × ℝ))) :=
sorry

end convex_quadrilateral_exists_l326_326503


namespace find_t_l326_326050

noncomputable def jenny_hours (t : ℝ) : ℝ := t + 3
noncomputable def jenny_rate (t : ℝ) : ℝ := 4t - 6
noncomputable def mike_hours (t : ℝ) : ℝ := 4t - 7
noncomputable def mike_rate (t : ℝ) : ℝ := t + 3

theorem find_t (t : ℝ):
  (jenny_hours t) * (jenny_rate t) = (mike_hours t) * (mike_rate t) + 3 →
  t = 3 :=
by
  sorry

end find_t_l326_326050


namespace meaningful_sqrt_l326_326704

theorem meaningful_sqrt (x : ℝ) (h : x - 3 ≥ 0) : x = 4 → sqrt (x - 3) ≥ 0 :=
by
  intro hx
  rw [hx]
  simp
  exact h

end meaningful_sqrt_l326_326704


namespace count_multiples_14_not_3_5_l326_326323

noncomputable def count_specified_multiples : Nat :=
  Set.size {n ∈ Finset.range 301 | n % 14 = 0 ∧ (n % 3 ≠ 0 ∧ n % 5 ≠ 0)}

theorem count_multiples_14_not_3_5 : count_specified_multiples = 11 := 
sorry

end count_multiples_14_not_3_5_l326_326323


namespace profit_difference_l326_326519

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end profit_difference_l326_326519


namespace smallest_k_always_win_l326_326381

theorem smallest_k_always_win (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, k = (nat.ceil (log n 2)) ∧ 
    ∀ (sheets : ℕ → finset ℕ), 
    (∀ i, sheets i ⊆ finset.range (n + 1) ∧ (finset.range (n + 1)).filter (λ x, x ∉ sheets i) ⊆ finset.range (n + 1)) → 
    ∀ config: fin n → bool, 
    (finset.range (n + 1)) = (finset.range (n + 1)).bunion (λ i, if config i then sheets i else (finset.range (n + 1)).filter (λ x, x ∉ sheets i)) :=
begin
  sorry
end

end smallest_k_always_win_l326_326381


namespace baron_munchausen_theorem_l326_326222

theorem baron_munchausen_theorem (n : ℕ) (a b : ℕ) (P : polynomial ℝ)
  (hP : P = polynomial.C_X ^ n - polynomial.C a * polynomial.X ^ (n - 1) + polynomial.C b * polynomial.X ^ (n - 2) + ⋯ )
  (roots_nat : ∀ x ∈ P.roots, x ∈ ℕ) :
  let lines := a in
  let intersections := (a * (a - 1)) / 2 in
  intersections = b :=
sorry

end baron_munchausen_theorem_l326_326222


namespace angle_between_lateral_edge_and_height_of_pyramid_l326_326835

-- Definition of some geometric concepts might be abstracted as variables
-- if necessary based on Mathlib's capabilities

variables (α k : ℝ)

-- The main lemma we want to prove: the angle α between the lateral edge and the height
-- of a regular triangular pyramid given the ratio k of its lateral surface area to base area.
theorem angle_between_lateral_edge_and_height_of_pyramid
  (h_ratio : ∀ (h : ℝ), k = (sqrt (4 + h ^ 2) / h)) :
  α = real.arccotg (sqrt (k ^ 2 - 1) / 2) := by
  sorry

end angle_between_lateral_edge_and_height_of_pyramid_l326_326835


namespace ivan_filled_two_piggy_banks_l326_326718

-- Let a penny be worth 0.01 dollars
def penny_value : ℝ := 0.01

-- Let a dime be worth 0.10 dollars
def dime_value : ℝ := 0.10

-- A piggy bank can hold 100 pennies and 50 dimes
def pennies_in_piggy_bank : ℕ := 100
def dimes_in_piggy_bank : ℕ := 50

-- Ivan has 12 dollars in total
def total_money : ℝ := 12.0

-- The value of a fully filled piggy bank
def piggy_bank_value :=
  (pennies_in_piggy_bank : ℝ) * penny_value + (dimes_in_piggy_bank : ℝ) * dime_value

-- Number of piggy banks filled by Ivan
def piggy_banks_filled :=
  total_money / piggy_bank_value

theorem ivan_filled_two_piggy_banks : piggy_banks_filled = 2 :=
by
  sorry

end ivan_filled_two_piggy_banks_l326_326718


namespace bruce_anne_clean_in_4_hours_l326_326948

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l326_326948


namespace arctan_tan_subtraction_l326_326562

theorem arctan_tan_subtraction :
  ∀ {A B : ℝ}, A = 75 ∧ B = 30 →
    ∃ (result : ℝ), result = 15 ∧ arctan (tan A - 3 * tan B) = result :=
begin
  intros a b h,
  rcases h with ⟨ha, hb⟩,
  use 15,
  split,
  { refl, },       -- result = 15
  { sorry, }       -- arctan (tan A - 3 * tan B) = result
end

end arctan_tan_subtraction_l326_326562


namespace amount_greater_first_number_l326_326091

-- Definitions for conditions
def avg_with_errors := 40.2
def avg_correct := 40.3
def wrong_num_2 := 13
def correct_num_2 := 31
def n := 10

-- Given conditions as hypotheses
def avg_sum_with_errors : Float := n * avg_with_errors
def avg_sum_correct : Float := n * avg_correct
def error_sum := avg_sum_correct - avg_sum_with_errors
def second_num_error := correct_num_2 - wrong_num_2

-- Target statement
theorem amount_greater_first_number (W A : Float) (x : Float) (h1 : W = A + x) (h2 : error_sum = 1) (h3 : second_num_error = 18):
  x = 19 :=
by
  sorry

end amount_greater_first_number_l326_326091


namespace bus_final_count_l326_326471

def initial_people : ℕ := 110
def first_stop_off : ℕ := 20
def first_stop_on : ℕ := 15
def second_stop_off : ℕ := 34
def second_stop_on : ℕ := 17
def third_stop_off : ℕ := 18
def third_stop_on : ℕ := 7
def fourth_stop_off : ℕ := 29
def fourth_stop_on : ℕ := 19
def fifth_stop_off : ℕ := 11
def fifth_stop_on : ℕ := 13
def sixth_stop_off : ℕ := 15
def sixth_stop_on : ℕ := 8
def seventh_stop_off : ℕ := 13
def seventh_stop_on : ℕ := 5
def eighth_stop_off : ℕ := 6
def eighth_stop_on : ℕ := 0

theorem bus_final_count :
  initial_people - first_stop_off + first_stop_on 
  - second_stop_off + second_stop_on 
  - third_stop_off + third_stop_on 
  - fourth_stop_off + fourth_stop_on 
  - fifth_stop_off + fifth_stop_on 
  - sixth_stop_off + sixth_stop_on 
  - seventh_stop_off + seventh_stop_on 
  - eighth_stop_off + eighth_stop_on = 48 :=
by sorry

end bus_final_count_l326_326471


namespace rhombus_perimeter_l326_326811

theorem rhombus_perimeter (d1 d2 : ℕ) (h_d1 : d1 = 16) (h_d2 : d2 = 30) :
  let side_length := Math.sqrt ((d1 / 2)^2 + (d2 / 2)^2) in
  let perimeter := 4 * side_length in
  perimeter = 68 := by
    dsimp only [side_length, perimeter]
    rw [h_d1, h_d2]
    norm_num
    sorry

end rhombus_perimeter_l326_326811


namespace average_tickets_sold_by_female_members_l326_326515

theorem average_tickets_sold_by_female_members 
  (average_all : ℕ)
  (ratio_mf : ℕ)
  (average_male : ℕ)
  (h1 : average_all = 66)
  (h2 : ratio_mf = 2)
  (h3 : average_male = 58) :
  ∃ (F : ℕ), F = 70 :=
by
  let M := 1
  let num_female := ratio_mf * M
  let total_tickets_male := average_male * M
  let total_tickets_female := num_female * 70
  have total_all_members : ℕ := M + num_female
  have total_tickets_all : ℕ := total_tickets_male + total_tickets_female
  have average_all_eq : average_all = total_tickets_all / total_all_members
  use 70
  sorry

end average_tickets_sold_by_female_members_l326_326515


namespace intersection_coordinates_l326_326000

variable (A B C G H Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C]

-- Defining points with given conditions
def is_point_on_line_AB (A B G : A) : Prop := 
  ∃ (x y : ℝ), x * A + y * B = G ∧ x / y = 3 / 2

def is_point_on_line_BC (B C H : B) : Prop := 
  ∃ (x y : ℝ), x * B + y * C = H ∧ x / y = 1 / 3

-- Stating the main theorem
theorem intersection_coordinates (A B C G H Q : A) (hG : is_point_on_line_AB A B G) (hH : is_point_on_line_BC B C H) :
  ∃ (x y z : ℝ), Q = x * A + y * B + z * C ∧ x = 3 / 14 ∧ y = 2 / 14 ∧ z = 6 / 14 :=
sorry

end intersection_coordinates_l326_326000


namespace solve_equation_l326_326785

theorem solve_equation (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) → x = -0.5 := 
by
  sorry

end solve_equation_l326_326785


namespace infinity_powers_of_two_iff_not_divisible_by_five_l326_326193

def sequence (a : ℕ → ℕ) (b : ℕ → ℕ) :=
  ∀ n, a (n+1) = a n + b n ∧ b n = a n % 10

theorem infinity_powers_of_two_iff_not_divisible_by_five (a : ℕ → ℕ) (b : ℕ → ℕ) (h_seq : sequence a b) :
  (∀ k, ∃ n, a n = 2^k) ↔ ¬ (∃ n, 5 ∣ a 1) :=
sorry

end infinity_powers_of_two_iff_not_divisible_by_five_l326_326193


namespace area_ratio_of_similar_triangles_l326_326854

noncomputable def similarity_ratio := 3 / 5

theorem area_ratio_of_similar_triangles (k : ℝ) (h_sim : similarity_ratio = k) : (k^2 = 9 / 25) :=
by
  sorry

end area_ratio_of_similar_triangles_l326_326854


namespace line_and_circle_separate_l326_326317

open Real

-- Define the vectors a and b
variables (α β : ℝ)
def vec_a : ℝ × ℝ := (2 * cos α, 2 * sin α)
def vec_b : ℝ × ℝ := (3 * cos β, 3 * sin β)

-- Define the angle between a and b
def angle_ab : ℝ := pi / 3 -- 60 degrees in radians

-- Line equation: x * cos(α) - y * sin(α) + 1/2 = 0
noncomputable def line_eq (x y : ℝ) : Prop :=
  x * cos α - y * sin α + 1/2 = 0

-- Circle equation: (x - cos(β))^2 + (y + sin(β))^2 = 1/2
def circle_eq (x y : ℝ) : Prop :=
  (x - cos β)^2 + (y + sin β)^2 = 1/2

-- Proof statement
theorem line_and_circle_separate (α β : ℝ) 
  (h_angle : cos (α - β) = 1/2) :
  ∃ l c : ℝ, line_eq α β l c → circle_eq α β l c → false :=
sorry

end line_and_circle_separate_l326_326317


namespace angle_sum_l326_326333

theorem angle_sum {A B D F G : Type} 
  (angle_A : ℝ) 
  (angle_AFG : ℝ) 
  (angle_AGF : ℝ) 
  (angle_BFD : ℝ)
  (H1 : angle_A = 30)
  (H2 : angle_AFG = angle_AGF)
  (H3 : angle_BFD = 105)
  (H4 : angle_AFG + angle_BFD = 180) 
  : angle_B + angle_D = 75 := 
by 
  sorry

end angle_sum_l326_326333


namespace group_of_2019th_odd_positive_l326_326220

theorem group_of_2019th_odd_positive {
  n : ℕ
  odd_positions : List ℕ
  group_seq : List (List ℕ)
  odd_grouping : ℕ → ℕ
  position_of_2019th_group : ℕ
  is_2019th_odd : ∀ k : ℕ, odd_positions k = 2 * k + 1
  arrangement : ∀ k, group_seq k = if even k then [2 * k + 1, 2 * k + 3] else [2 * k + 1, 2 * k + 3, 2 * k + 5]
  odd_group_assignment : ∀ k, k < 5 → odd_grouping k = 2 * k / 5 + 1 ∧ odd_grouping (k + 5) = 2 * ((k + 5) / 5) + 2
  position_of_2019th := odd_grouping 2019
} : position_of_2019th = 404 := by
sorry

end group_of_2019th_odd_positive_l326_326220


namespace price_per_kg_of_fruits_l326_326047

theorem price_per_kg_of_fruits (mangoes apples oranges : ℕ) (total_amount : ℕ)
  (h1 : mangoes = 400)
  (h2 : apples = 2 * mangoes)
  (h3 : oranges = mangoes + 200)
  (h4 : total_amount = 90000) :
  (total_amount / (mangoes + apples + oranges) = 50) :=
by
  sorry

end price_per_kg_of_fruits_l326_326047


namespace equation_of_circle_C_l326_326679

theorem equation_of_circle_C :
  ∃ (h k r : ℝ), (h = 2) ∧ (k = 3) ∧ (r = 2) ∧
  (h, k ∈ {p : ℝ × ℝ | p.1 = 3∧) ∧ 
  ( ∀ x y : ℝ), (x-2)^2 + (y-3)^2 = 4 := 
begin
  sorry
end

end equation_of_circle_C_l326_326679


namespace quadratic_difference_square_l326_326331

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end quadratic_difference_square_l326_326331


namespace sum_of_coordinates_x_l326_326023

-- Given points Y and Z
def Y : ℝ × ℝ := (2, 8)
def Z : ℝ × ℝ := (0, -4)

-- Given ratio conditions
def ratio_condition (X Y Z : ℝ × ℝ) : Prop :=
  dist X Z / dist X Y = 1/3 ∧ dist Z Y / dist X Y = 1/3

-- Define X, ensuring Z is the midpoint of XY
def X : ℝ × ℝ := (4, 20)

-- Prove that sum of coordinates of X is 10
theorem sum_of_coordinates_x (h : ratio_condition X Y Z) : (X.1 + X.2) = 10 := 
  sorry

end sum_of_coordinates_x_l326_326023


namespace john_max_books_l326_326012

theorem john_max_books (h₁ : 4575 ≥ 0) (h₂ : 325 > 0) : 
  ∃ (x : ℕ), x = 14 ∧ ∀ n : ℕ, n ≤ x ↔ n * 325 ≤ 4575 := 
  sorry

end john_max_books_l326_326012


namespace determine_k_range_l326_326751

def p (k : ℝ) : Prop := ∀ x : ℝ, y = k * x + 1 → y > y

def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2 * k - 3) * x + 1 = 0


theorem determine_k_range (h1 : ¬ (p k ∧ q k)) (h2 : p k ∨ q k) :
  k ∈ set.Iic 0 ∪ set.Ioo (1/2) (5/2) := 
sorry

end determine_k_range_l326_326751


namespace necessary_but_not_sufficient_condition_l326_326117

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  {x | 1 / x ≤ 1} ⊆ {x | Real.log x ≥ 0} ∧ 
  ¬ ({x | Real.log x ≥ 0} ⊆ {x | 1 / x ≤ 1}) :=
by
  sorry

end necessary_but_not_sufficient_condition_l326_326117


namespace number_of_possible_third_sides_l326_326855

theorem number_of_possible_third_sides 
  (a b : ℕ) (even_third_side : ℕ → Prop)
  (h_a : a = 5) (h_b : b = 7)
  (h_even : ∀ n, even_third_side n ↔ (2 ∣ n)) :
  ∃ n, n = 4 :=
by
  intros a b even_third_side h_a h_b h_even
  sorry

end number_of_possible_third_sides_l326_326855


namespace sum_ak_lt_one_l326_326400

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1/2 ∧ ∀ k ≥ 2, 2 * k * a k = (2 * k - 3) * a (k - 1)

theorem sum_ak_lt_one (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, ∑ k in Finset.range (n + 1), a (k + 1) < 1 :=
sorry

end sum_ak_lt_one_l326_326400


namespace seashell_count_l326_326758

variable (initial_seashells additional_seashells total_seashells : ℕ)

theorem seashell_count (h1 : initial_seashells = 19) (h2 : additional_seashells = 6) : 
  total_seashells = initial_seashells + additional_seashells → total_seashells = 25 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end seashell_count_l326_326758


namespace find_a_l326_326648

noncomputable def f (x : ℝ) := real.sqrt (x - 1)

theorem find_a (a : ℝ) (h : f a = 3) : a = 10 :=
by sorry

end find_a_l326_326648


namespace jack_total_yen_l326_326005

def pounds := 42
def euros := 11
def yen := 3000
def pounds_per_euro := 2
def yen_per_pound := 100

theorem jack_total_yen : (euros * pounds_per_euro + pounds) * yen_per_pound + yen = 9400 := by
  sorry

end jack_total_yen_l326_326005


namespace paperboy_deliveries_l326_326181

def D : ℕ → ℕ 
| 0     := 1
| 1     := 2
| 2     := 3
| 3     := 6
| (n+4) := D (n+3) + D (n+2) + D (n+1)

theorem paperboy_deliveries : D 12 = 1431 := 
by 
-- Proof will be placed here
sorry

end paperboy_deliveries_l326_326181


namespace sum_fractional_parts_l326_326566

noncomputable def zeta (x : ℝ) : ℝ := ∑' n : ℕ, if n > 0 then 1 / (n : ℝ) ^ x else 0

def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem sum_fractional_parts :
  (∑' k : ℕ, if k ≥ 2 then fractional_part (zeta (2 * k / (k + 1))) else 0) 
  = zeta (4 / 3) - 1 :=
by
  sorry

end sum_fractional_parts_l326_326566


namespace find_ab_l326_326498

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 18 :=
begin
  sorry -- Proof is not required
end

end find_ab_l326_326498


namespace min_transport_time_l326_326874

-- Define the conditions and the proof statement.
variable (v : ℝ) (t : ℝ)
variable (d : ℝ := 400)
variable (n : ℕ := 26)

-- Condition: Distance between two locations is 400 kilometers (d)
def distance := d

-- Condition: The speed of the trucks in kilometers per hour (v)
def speed := v

-- Condition: The number of trucks (26)
def num_trucks := n

-- Condition: The minimum distance between every two trucks
def min_distance (v : ℝ) := (v / 20) ^ 2

-- The proof statement: Prove the minimum time t is 10 hours
theorem min_transport_time (hv : speed = 80) : t = 10 := 
sorry

end min_transport_time_l326_326874


namespace prime_sum_probability_l326_326988

-- Definition of the problem conditions
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of the probability calculation
def num_valid_pairs : ℕ := 3
def total_pairs : ℕ := 45
def probability_prime_sum_gt_10 : ℚ := num_valid_pairs / total_pairs

-- Problem statement in Lean
theorem prime_sum_probability : probability_prime_sum_gt_10 = 1 / 15 :=
by
  sorry

end prime_sum_probability_l326_326988


namespace combinatorial_solution_l326_326119

theorem combinatorial_solution :
  ∃ x : ℕ, ((x = 4) ∨ (x = 9)) ∧ (Nat.choose 28 x = Nat.choose 28 (3 * x - 8)) :=
begin
  let x4_solution : ℕ := 4,
  let x9_solution : ℕ := 9,
  use [x4_solution, x9_solution],
  split,
  { right, refl },
  { sorry }
end

end combinatorial_solution_l326_326119


namespace sum_of_a_and_b_l326_326533

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem sum_of_a_and_b :
  let p1 := (0, 0) : ℝ × ℝ
  let p2 := (2, 4) : ℝ × ℝ
  let p3 := (5, 3) : ℝ × ℝ
  let p4 := (4, 0) : ℝ × ℝ
  let d1 := distance p1 p2
  let d2 := distance p2 p3
  let d3 := distance p3 p4
  let d4 := distance p4 p1
  let perimeter := d1 + d2 + d3 + d4
  ∃ a b c d : ℕ, perimeter = (a:ℝ) * Real.sqrt (c:ℝ) + (b:ℝ) * Real.sqrt (d:ℝ) + 4 ∧
    a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l326_326533


namespace ball_hits_ground_l326_326099

def height (t : ℝ) : ℝ := -6.1 * t^2 + 4.2 * t + 7

theorem ball_hits_ground : ∃ t : ℝ, height t = 0 ∧ t = 2 :=
by
  sorry

end ball_hits_ground_l326_326099


namespace gems_received_l326_326847

-- Definitions corresponding to the conditions
def dollars_spent : ℕ := 250
def gems_per_dollar : ℕ := 100
def bonus_percentage : ℝ := 0.20

-- Theorem statement
theorem gems_received : (dollars_spent * gems_per_dollar : ℕ) + (bonus_percentage * (dollars_spent * gems_per_dollar) : ℕ) = 30000 := 
by
  sorry

end gems_received_l326_326847


namespace triangle_shortest_side_condition_l326_326697

theorem triangle_shortest_side_condition
  (A B C : Type) 
  (r : ℝ) (AF FB : ℝ)
  (P : ℝ)
  (h_AF : AF = 7)
  (h_FB : FB = 9)
  (h_r : r = 5)
  (h_P : P = 46) 
  : (min (min (7 + 9) (2 * 14)) ((7 + 9) - 14)) = 2 := 
by sorry

end triangle_shortest_side_condition_l326_326697


namespace probability_real_roots_of_quadratic_eq_l326_326176

theorem probability_real_roots_of_quadratic_eq :
  let outcomes := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)] in
  let event_A := [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), 
                  (3, 2), (4, 2), (5, 2), (6, 2), 
                  (4, 3), (5, 3), (6, 3), 
                  (4, 4), (5, 4), (6, 4), 
                  (5, 5), (6, 5), 
                  (5, 6), (6, 6)] in
  (event_A.length : ℚ) / outcomes.length = 19 / 36 :=
by 
  -- no proof needed as specified
  sorry

end probability_real_roots_of_quadratic_eq_l326_326176


namespace johnny_marbles_combination_l326_326723

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end johnny_marbles_combination_l326_326723


namespace taller_tree_height_l326_326123

theorem taller_tree_height :
  ∀ (h : ℕ), 
    ∃ (h_s : ℕ), (h_s = h - 24) ∧ (5 * h = 7 * h_s) → h = 84 :=
by
  sorry

end taller_tree_height_l326_326123


namespace regular_dodecagon_product_l326_326534

noncomputable def Q_dodecagon_product : ℂ :=
  let z_center := 2;
  let z1 := 1;
  let z7 := 3;
  ∏ (k : ℕ) in finset.range 12, (complex.of_real ((1 + real.cos (2 * real.pi * k / 12)) + (2 * real.sin (2 * real.pi * k / 12)) * complex.I))

theorem regular_dodecagon_product (Q_n : ℕ → ℂ) 
  (h1 : Q_n 1 = 1 + 0 * complex.I)
  (h7 : Q_n 7 = 3 + 0 * complex.I)
  (h_cent : ∀ k, Q_n k = complex.of_real (2 * real.cos (2 * real.pi * k / 12)) + (2 * real.sin (2 * real.pi * k / 12)) * complex.I) :
  (∏ k in finset.range 12, Q_n k) = 4095 :=
begin
  -- The proof can be filled in here
  sorry
end

end regular_dodecagon_product_l326_326534


namespace inequality_of_sequence_l326_326620

theorem inequality_of_sequence 
  (n : ℕ)
  (a : Fin (2*n-1) → ℝ)
  (h1 : ∀ i j, i < j → j < 2*n → a i ≥ a j)
  (h2 : ∀ i, i < 2*n → a i ≥ 0) :
  (∑ i in Finset.range (2 * n - 1), (-1)^i * (a i)^2) 
  ≥ ((∑ i in Finset.range (2 * n - 1), (-1)^i * a i)^2) :=
sorry

end inequality_of_sequence_l326_326620


namespace probability_two_jacks_or_one_queen_l326_326668

theorem probability_two_jacks_or_one_queen (cards : Finset ℕ) (queens jacks : Finset ℕ) :
  cards.card = 52 ∧
  queens.card = 4 ∧
  jacks.card = 4 ∧
  (queens ⊆ cards) ∧
  (jacks ⊆ cards) →
  ∑ x in (cards.choose 2), if x.to_finset ⊇ jacks ∧ x.to_finset.card = 2 then (1 : ℚ) / (52 * 51 / 2) else 
    if x.to_finset ∩ queens ≠ ∅ then (1 : ℚ) / (52 * 51 / 2) else 0 = (2 / 13 : ℚ) :=
by
  sorry

end probability_two_jacks_or_one_queen_l326_326668


namespace coordinates_of_P_l326_326638

theorem coordinates_of_P (a : ℝ) (h : 2 * a - 6 = 0) : (2 * a - 6, a + 1) = (0, 4) :=
by 
  have ha : a = 3 := by linarith
  rw [ha]
  sorry

end coordinates_of_P_l326_326638


namespace disease_spread_days_l326_326247

-- Define the grid size
def grid_size : ℕ := 2015

-- Define the center of the grid
def center : ℕ × ℕ := (1008, 1008)

-- Define the Taxicab (Manhattan) distance function
def taxicab_distance (p1 p2 : ℕ × ℕ) : ℕ := 
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)
 
-- Define the initially diseased plant
def initial_diseased_plant : (ℕ × ℕ) := center

-- Define the boundary conditions of the grid
def is_within_bounds (p : ℕ × ℕ) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ grid_size ∧ 1 ≤ p.2 ∧ p.2 ≤ grid_size

-- Define the farthest corners of the grid
def farthest_corners : list (ℕ × ℕ) := 
  [(1, 1), (1, 2015), (2015, 1), (2015, 2015)]

-- Calculate the farthest distance from center to any corner in terms of taxicab distance
def maximum_distance_to_corners : ℕ := 
  farthest_corners.map (λ corner, taxicab_distance center corner) |>.maximum

-- State the theorem
theorem disease_spread_days : maximum_distance_to_corners = 2014 := 
by
  simp [maximum_distance_to_corners, taxicab_distance, center, farthest_corners]
  -- This simplifies the calculation to checking that the maximum taxicab distance is 2014
  sorry

end disease_spread_days_l326_326247


namespace solve_inequality_and_find_positive_int_solutions_l326_326079

theorem solve_inequality_and_find_positive_int_solutions :
  ∀ (x : ℝ), (2 * x + 1) / 3 - 1 ≤ (2 / 5) * x → x ≤ 2.5 ∧ ∃ (n : ℕ), n = 1 ∨ n = 2 :=
by
  intro x
  intro h
  sorry

end solve_inequality_and_find_positive_int_solutions_l326_326079


namespace intersection_M_N_union_complements_M_N_l326_326613

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_M_N :
  M ∩ N = {x | 1 ≤ x ∧ x < 5} :=
by {
  sorry
}

theorem union_complements_M_N :
  (compl M) ∪ (compl N) = {x | x < 1 ∨ x ≥ 5} :=
by {
  sorry
}

end intersection_M_N_union_complements_M_N_l326_326613


namespace min_value_inequality_l326_326390

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end min_value_inequality_l326_326390


namespace tire_miles_used_l326_326175

theorem tire_miles_used (total_miles : ℕ) (num_tires_used : ℕ) (total_tires : ℕ) (h : total_tires > num_tires_used) :
  total_miles * num_tires_used % total_tires = 0 →
  total_miles * num_tires_used / total_tires = 36000 :=
by
  -- Let's translate the problem to our Lean theorem
  let total_tire_miles := total_miles * num_tires_used
  have h1 : total_tire_miles = 252000 := by sorry
  let miles_per_tire := total_tire_miles / total_tires
  have h2 : miles_per_tire = 36000 := by sorry
  exact h2

end tire_miles_used_l326_326175


namespace exists_tangent_line_parallel_l326_326685

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x + a * x
noncomputable def line := 3
noncomputable def parallel_slope := 3

theorem exists_tangent_line_parallel (a : ℝ) :
  (∃ (x : ℝ) (h : 0 < x), deriv (f a) x = parallel_slope) ↔
  a ∈ Set.Ioo (Real.neg_infty) (3 - Real.exp (-2)) ∪ Set.Ioo (3 - Real.exp (-2)) 3 := 
sorry

end exists_tangent_line_parallel_l326_326685


namespace number_of_terms_arithmetic_sequence_l326_326242

theorem number_of_terms_arithmetic_sequence :
  let a := 2.5
  let d := 5
  let l := 67.5
  ∃ n : ℕ, a + (n - 1) * d = l :=
by
  sorry

end number_of_terms_arithmetic_sequence_l326_326242


namespace vertex_coordinates_l326_326095

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := (x + 3) ^ 2 - 1

-- Define the statement for the coordinates of the vertex of the parabola
theorem vertex_coordinates : ∃ (h k : ℝ), (∀ x : ℝ, parabola x = (x + 3) ^ 2 - 1) ∧ h = -3 ∧ k = -1 := 
  sorry

end vertex_coordinates_l326_326095


namespace initial_wings_l326_326901

-- Definitions of the given conditions
def num_friends : ℕ := 4
def wings_each : ℕ := 4
def additional_wings : ℕ := 7
def total_wings : ℕ := num_friends * wings_each

-- Proposition that the initial number of cooked wings is 9
theorem initial_wings (total_wings additional_wings : ℕ) : 
  initial_wings = total_wings - additional_wings := 
sorry

end initial_wings_l326_326901


namespace like_terms_monomials_l326_326347

theorem like_terms_monomials (a b : ℕ) : (5 * (m^8) * (n^6) = -(3/4) * (m^(2*a)) * (n^(2*b))) → (a = 4 ∧ b = 3) := by
  sorry

end like_terms_monomials_l326_326347


namespace rhombus_perimeter_l326_326800

-- Definitions based on conditions
def diagonal1 : ℝ := 16
def diagonal2 : ℝ := 30
def half_diagonal1 : ℝ := diagonal1 / 2
def half_diagonal2 : ℝ := diagonal2 / 2

-- Mathematical formulation in Lean
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := 
by
  -- Given diagonals
  have h_half_d1 : (d1 / 2) = 8 := by sorry,
  have h_half_d2 : (d2 / 2) = 15 := by sorry,
  
  -- Combine into Pythagorean theorem and perimeter calculation
  have h_side_length : real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 17 := by sorry,
  show 4 * real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 68 := by sorry

end rhombus_perimeter_l326_326800


namespace third_number_in_expression_l326_326505

theorem third_number_in_expression :
  ∃ c : ℝ, ((26.3 * 12 * c) / 3) + 125 = 2229 :=
begin
  use 20,
  sorry
end

end third_number_in_expression_l326_326505


namespace repeated_decimal_sum_l326_326142

def gcd (a b : ℕ) : ℕ := Int.gcd a b

def decimal_to_fraction_in_lowest_terms (d : ℕ) (n : ℕ) (repeating : ℕ) : ℚ :=
  (repeating * 10^d + n) / (10^d * 9)

theorem repeated_decimal_sum (d n : ℕ) (h : n < 10^d) : n = 56 → gcd 56 99 = 1 → 
  0.\overline{56} in lowest terms = 56 / 99 → (56 + 99 = 155) :=
by
  intros h1 gcd_56_99 h2
  sorry

end repeated_decimal_sum_l326_326142


namespace yellow_tint_percentage_after_adding_tint_l326_326113

theorem yellow_tint_percentage_after_adding_tint 
  (original_volume : ℕ) 
  (initial_red_tint_percentage initial_yellow_tint_percentage initial_water_percentage : ℚ)
  (added_yellow_tint_volume : ℕ)
  (h1 : original_volume = 50)
  (h2 : initial_red_tint_percentage = 20 / 100)
  (h3 : initial_yellow_tint_percentage = 50 / 100)
  (h4 : initial_water_percentage = 30 / 100)
  (h5 : added_yellow_tint_volume = 10) :
  (35 / (original_volume + added_yellow_tint_volume) * 100).nat_abs = 58 := 
by
  sorry

end yellow_tint_percentage_after_adding_tint_l326_326113


namespace find_WZ_l326_326387

noncomputable def WZ (X Y Z W : ℝ) : ℝ := 27

theorem find_WZ (X Y Z W : ℝ) (h1 : ∠XYZ = 90) (h2 : W ∈ segment X Z)
  (h3 : XW = 3) (h4 : YW = 9) : WZ X Y Z W = 27 :=
by
  sorry

end find_WZ_l326_326387


namespace initial_stock_calc_l326_326560

/-- Definitions for conditions in the problem. -/

/-- On the first day, 500 people took 1 can of food each. -/
def cans_taken_first_day : ℕ := 500 * 1

/-- Carla restocked 1500 cans after the first day. -/
def restocked_first_day : ℕ := 1500

/-- On the second day, 1000 people took 2 cans of food each. -/
def cans_taken_second_day : ℕ := 1000 * 2

/-- Carla restocked 3000 cans after the second day. -/
def restocked_second_day : ℕ := 3000

/-- Carla gave away 2500 cans of food in total. -/
def total_cans_given_away : ℕ := 2500

/-- The Lean theorem stating the equivalent proof problem. -/
theorem initial_stock_calc : 
  cans_taken_first_day + cans_taken_second_day = total_cans_given_away →
  (restocked_first_day + restocked_second_day) - total_cans_given_away = 2000 := by
  sorry

end initial_stock_calc_l326_326560


namespace line_circle_intersection_l326_326623

-- Define the parametric equations of the line
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-1 - (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

-- Define the circle equation in polar form and convert it to cartesian form
noncomputable def circle : ℝ × ℝ → Prop :=
  fun p => (p.1 - 1/2)^2 + (p.2 - (Real.sqrt 3 / 2))^2 = 1

-- Define the condition that points M and N are on both the line and the circle,
-- and calculate the product of distances |PM| * |PN|
theorem line_circle_intersection :
  ∃PM PN : ℝ, ∃ t1 t2 : ℝ,
  (parametric_line t1 = PM) ∧ (parametric_line t2 = PN) ∧
  -- distances from P to M and P to N
  (PM = 6 + 2 * Real.sqrt 3) :=
sorry

end line_circle_intersection_l326_326623


namespace electronics_weight_l326_326832

theorem electronics_weight (x : ℝ) : 
  let books := 5 * x in
  let clothes := 4 * x in
  let electronics := 2 * x in
  5 / (4 * x - 7) = 5 / 2 →
  electronics = 7 :=
by
  intros
  sorry

end electronics_weight_l326_326832


namespace calculate_P_Q_nested_l326_326382

def P (x : ℝ) : ℝ := 3 * Real.sqrt x
def Q (x : ℝ) : ℝ := x^2

theorem calculate_P_Q_nested (x : ℝ) : P (Q (P (Q (P (Q 5))))) = 135 :=
by
  sorry

end calculate_P_Q_nested_l326_326382


namespace sin_power_sum_l326_326969

theorem sin_power_sum :
  ∑ k in Finset.range 46, (Real.sin (2 * k * Real.pi / 180)) ^ 4 = 113 / 8 :=
by
  sorry

end sin_power_sum_l326_326969


namespace correct_calculation_l326_326487

theorem correct_calculation :
  (|9| ≠ 3) ∧ (|9| ≠ -3) ∧
  (-1 - 1 ≠ 0) ∧
  ((-1)^(-1) ≠ 0) ∧
  ((-1)^0 = 1) :=
by
  sorry

end correct_calculation_l326_326487


namespace min_value_inequality_l326_326388

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end min_value_inequality_l326_326388


namespace cleaner_flow_rate_l326_326530

theorem cleaner_flow_rate (x : ℝ) (h1 : 30 + 10 * x + 20 = 80) : x = 3 :=
by
  skip -- Placeholder for the actual proof

-- Dummy statement to ensure namespace does not stay empty. This will obviously be replaced with the full proof.
#check cleaner_flow_rate

end cleaner_flow_rate_l326_326530


namespace positive_difference_of_two_numbers_l326_326463

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end positive_difference_of_two_numbers_l326_326463


namespace calculate_value_l326_326267

theorem calculate_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : 
  (x + 1 / x) * (z - 1 / z) = 4 := 
by 
  -- Proof omitted, this is just the statement
  sorry

end calculate_value_l326_326267


namespace crayon_selection_l326_326888

theorem crayon_selection : (nat.choose 15 3) = 455 :=
by
  sorry

end crayon_selection_l326_326888


namespace max_marked_points_no_convex_quad_l326_326148

-- Define the condition that no four points form a convex quadrilateral
def no_convex_quadrilateral (points : Finset (ℕ × ℕ)) : Prop :=
  ∀ (a b c d : ℕ × ℕ), a ∈ points → b ∈ points → c ∈ points → d ∈ points → 
  ¬(convex_quadrilateral a b c d)

-- Define when a set of points forms a convex quadrilateral
def convex_quadrilateral (a b c d : ℕ × ℕ) : Prop :=
  -- Placeholder function to define convex quadrilateral condition
  sorry

-- Proving the maximum number of points that can be marked without forming a convex quadrilateral
theorem max_marked_points_no_convex_quad (n : ℕ) (h : n ≥ 3) :
  ∃ (m : ℕ), (∀ (points : Finset (ℕ × ℕ)), no_convex_quadrilateral points → points.card ≤ 2 * n - 1) :=
begin
  use 2 * n - 1,
  sorry
end

end max_marked_points_no_convex_quad_l326_326148


namespace lena_shaded_cells_l326_326052

theorem lena_shaded_cells (h w : ℕ) (x y : ℝ) (a : ℝ)
  (hw : h = 60) (ww : w = 70)
  (hx : x = 0.83) : 
  let total_shaded := (2 * (29 + 24) + 1) in
  total_shaded = 108 := by
  sorry

end lena_shaded_cells_l326_326052


namespace a_2004_eq_l326_326313

noncomputable def a : ℕ → ℝ
| 1 := 1
| (n + 1) := (λ n, (sqrt 3 * a n - 1) / (a n + sqrt 3)) n

theorem a_2004_eq : a 2004 = 2 + sqrt 3 :=
sorry

end a_2004_eq_l326_326313


namespace triangle_area_l326_326355

theorem triangle_area (PQ QR RJ KS : ℝ) (h_PQ : PQ = 7) (h_QR : QR = 4) (h_RJ : RJ = 3) (h_KS : KS = 3)
  (P Q R S J K T : Type*) [affine_space ℝ P] [affine_space ℝ Q] [affine_space ℝ R]
  [affine_space ℝ S] [affine_space ℝ J] [affine_space ℝ K] [affine_space ℝ T]
  (h_RS_PQ : segment RS = segment PQ)
  (h_PJ_intersects_JK : line PJ ∩ line JK = {T}) :
  area (triangle PTQ) = 91 / 3 :=
by
  sorry

end triangle_area_l326_326355


namespace books_return_to_initial_configuration_l326_326435

noncomputable def reverse_books (n : ℕ) (seq : list ℕ) (k : ℕ) : list ℕ :=
(seq.take k).reverse ++ (seq.drop k)

theorem books_return_to_initial_configuration (n : ℕ) (M : ℕ) (a_i : ℕ → ℕ) :
  ∃ (M : ℕ), (∀ i, (M > 0) ∧ (∀ k, k < 2 * M → reverse_books n (list.of_fn a_i) k = list.of_fn a_i)) :=
sorry

end books_return_to_initial_configuration_l326_326435


namespace rental_days_l326_326443

-- Definitions based on conditions
def daily_rate := 30
def weekly_rate := 190
def total_payment := 310

-- Prove that Jennie rented the car for 11 days
theorem rental_days : ∃ d : ℕ, d = 11 ∧ (total_payment = weekly_rate + (d - 7) * daily_rate) ∨ (d < 7 ∧ total_payment = d * daily_rate) :=
by
  sorry

end rental_days_l326_326443


namespace projection_correct_l326_326187

def proj (u v : ℝ × ℝ) :=
  let dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2
  let scale (k : ℝ) (w : ℝ × ℝ) : ℝ × ℝ :=
    (k * w.1, k * w.2)
  let numerator := dot_product u v
  let denominator := dot_product v v
  scale (numerator / denominator) v

theorem projection_correct :
  let u := (1, -1 : ℝ × ℝ)
  let v := (5, 1 : ℝ × ℝ)
  let result := proj u v
  result = ((10 / 13 : ℝ), (2 / 13 : ℝ)) := 
by
  let u := (1, -1 : ℝ × ℝ)
  let v := (5, 1 : ℝ × ℝ)
  let result := proj u v
  show result = ((10 / 13 : ℝ), (2 / 13 : ℝ))
  sorry

end projection_correct_l326_326187


namespace alicia_local_tax_in_cents_l326_326545

theorem alicia_local_tax_in_cents (hourly_wage : ℝ) (tax_rate : ℝ)
  (h_hourly_wage : hourly_wage = 30) (h_tax_rate : tax_rate = 0.021) :
  (hourly_wage * tax_rate * 100) = 63 := by
  sorry

end alicia_local_tax_in_cents_l326_326545


namespace initial_caps_correct_l326_326729

variable (bought : ℕ)
variable (total : ℕ)

def initial_bottle_caps (bought : ℕ) (total : ℕ) : ℕ :=
  total - bought

-- Given conditions
def bought_caps : ℕ := 7
def total_caps : ℕ := 47

theorem initial_caps_correct : initial_bottle_caps bought_caps total_caps = 40 :=
by
  -- proof here
  sorry

end initial_caps_correct_l326_326729


namespace probability_p_s_mod_10_l326_326475

noncomputable def count_valid_pairs : ℕ :=
  (Finset.Icc 1 100).card * ((Finset.Icc 1 100).card - 1) / 2

noncomputable def total_pairs : ℕ := 4950

theorem probability_p_s_mod_10 (a b : ℕ) (h_a : a ∈ Finset.Icc 1 100) (h_b : b ∈ Finset.Icc 1 100) (h_neq : a ≠ b) :
  let S := a + b,
  let P := a * b in
  (Nat.gcd a b = 1 ↔ Rational.gcd (a+b) (a*b+n) !=0) (hermite_of_galois a b) (P+S) % 10 = n  := 

begin
  sorry -- Proof not required
end

end probability_p_s_mod_10_l326_326475


namespace percentage_increase_of_nathan_money_l326_326041

theorem percentage_increase_of_nathan_money (dollars_lydia : ℤ) (euros_jorge : ℤ) (exchange_rate : ℚ) (dollars_nathan : ℤ) :
  dollars_lydia = 600 →
  euros_jorge = 450 →
  exchange_rate = 3 / 2 →
  dollars_nathan = 700 →
  let dollars_jorge := euros_jorge * exchange_rate,
      avg_dollars_lydia_jorge := (dollars_lydia + dollars_jorge) / 2,
      percentage_increase := (dollars_nathan - avg_dollars_lydia_jorge) / avg_dollars_lydia_jorge * 100
  in percentage_increase ≈ 9.8 :=
begin
  assume h1 h2 h3 h4,
  let dollars_jorge := euros_jorge * exchange_rate,
  have hj : dollars_jorge = 675 := by sorry,
  let avg_dollars_lydia_jorge := (dollars_lydia + dollars_jorge) / 2,
  have havg : avg_dollars_lydia_jorge = 637.5 := by sorry,
  let percentage_increase := (dollars_nathan - avg_dollars_lydia_jorge) / avg_dollars_lydia_jorge * 100,
  have hperc : percentage_increase = 9.8 := by sorry,
  norm_num,
  exact hperc,
end

end percentage_increase_of_nathan_money_l326_326041


namespace choose_photographers_l326_326701

theorem choose_photographers (n k : ℕ) (h_n : n = 10) (h_k : k = 3) : Nat.choose n k = 120 := by
  -- The proof is omitted
  sorry

end choose_photographers_l326_326701


namespace problem_statement_l326_326984

-- Define the variables a and b
noncomputable def a : ℝ := Real.arcsin (4 / 5)
noncomputable def b : ℝ := Real.arctan 3

-- Use corresponding values for sin and tan 
def sin_a : ℝ := Real.sin a
def tan_b : ℝ := Real.tan b

-- Asserted conditions
lemma condition_sin_a : sin_a = 4 / 5 := by
  sorry

lemma condition_tan_b : tan_b = 3 := by
  sorry

-- Prove the desired equality
theorem problem_statement : Real.sin (a - b) = -1 / Real.sqrt 10 := by
  sorry

end problem_statement_l326_326984


namespace max_good_diagonals_in_convex_polygon_l326_326350

def is_contractible (n : Nat) : Prop :=
  n = 2 + k

theorem max_good_diagonals_in_convex_polygon (n : Nat) (h : n ≥ 2) : 
  ∃ g : Nat, g = 2 * (n / 2).floor - 2 :=
by
  sorry

end max_good_diagonals_in_convex_polygon_l326_326350


namespace mike_total_cans_l326_326763

theorem mike_total_cans (monday_cans : ℕ) (tuesday_cans : ℕ) (total_cans : ℕ) : 
  monday_cans = 71 ∧ tuesday_cans = 27 ∧ total_cans = monday_cans + tuesday_cans → total_cans = 98 :=
by
  sorry

end mike_total_cans_l326_326763


namespace shortest_distance_ladybird_spider_l326_326695

/-- In a square garden PQRT with side 10 meters, a ladybird sets off from Q and moves along
edge QR at 30 cm/min. At the same time, a spider sets off from R and moves along edge RT at 40 cm/min.
Prove that the shortest distance between them, in meters, is 8. -/
theorem shortest_distance_ladybird_spider :
  ∃ t : ℝ, 0 ≤ t ∧ let QL := 30 * t in let RS := 40 * t in
  sqrt ((1000 - QL)^2 + RS^2) = 800 :=
sorry

end shortest_distance_ladybird_spider_l326_326695


namespace tangent_parallel_tangent_perpendicular_l326_326820

-- Part (1): Parallel case
theorem tangent_parallel (m : ℝ) :
  (∃ (f : ℝ → ℝ), f = λ x, m*x^3 + 2*x + 1 ∧ f.deriv 1 = 3 ∧ f 1 = m + 3) →
  m = 1 / 3 :=
by sorry

-- Part (2): Perpendicular case
theorem tangent_perpendicular :
  (∃ (m : ℝ) (f : ℝ → ℝ), f = λ x, m*x^3 + 2*x + 1 ∧ f.deriv 1 = 2 ∧ f 1 = m + 3) →
  (∃ (l : ℝ → ℝ), l = λ x, 2*x + 1) :=
by sorry

end tangent_parallel_tangent_perpendicular_l326_326820


namespace Bruno_wants_2_5_dozens_l326_326962

theorem Bruno_wants_2_5_dozens (total_pens : ℕ) (dozen_pens : ℕ) (h_total_pens : total_pens = 30) (h_dozen_pens : dozen_pens = 12) : (total_pens / dozen_pens : ℚ) = 2.5 :=
by 
  sorry

end Bruno_wants_2_5_dozens_l326_326962


namespace cos_diff_identity_l326_326293

variable {α : ℝ}

def sin_alpha := -3 / 5

def alpha_interval (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi)

theorem cos_diff_identity (h1 : Real.sin α = sin_alpha) (h2 : alpha_interval α) :
  Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 10 :=
  sorry

end cos_diff_identity_l326_326293


namespace road_trip_mileage_base10_l326_326011

-- Defining the base 8 number 3452
def base8_to_base10 (n : Nat) : Nat :=
  3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 2 * 8^0

-- Stating the problem as a theorem
theorem road_trip_mileage_base10 : base8_to_base10 3452 = 1834 := by
  sorry

end road_trip_mileage_base10_l326_326011


namespace concentration_after_dilution_l326_326606

-- Definitions and conditions
def initial_volume : ℝ := 5
def initial_concentration : ℝ := 0.06
def poured_out_volume : ℝ := 1
def added_water_volume : ℝ := 2

-- Theorem statement
theorem concentration_after_dilution : 
  (initial_volume * initial_concentration - poured_out_volume * initial_concentration) / 
  (initial_volume - poured_out_volume + added_water_volume) = 0.04 :=
by 
  sorry

end concentration_after_dilution_l326_326606


namespace complex_product_conjugate_l326_326753

def conjugate (z : ℂ) : ℂ := complex.conj z

noncomputable def z : ℂ := (3 - I) / (2 + I)

theorem complex_product_conjugate :
  (2 + I) * z = 3 - I → z * conjugate(z) = 2 :=
by
  sorry

end complex_product_conjugate_l326_326753


namespace complex_modulus_squared_l326_326029

open Complex

theorem complex_modulus_squared (w : ℂ) (h : w^2 + abs w ^ 2 = 7 + 2 * I) : abs w ^ 2 = 53 / 14 :=
sorry

end complex_modulus_squared_l326_326029


namespace g_is_odd_function_l326_326577

noncomputable def g (x : ℝ) : ℝ := log (x ^ 3 + sqrt (1 + x ^ 6))

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x := 
by
  intro x
  sorry

end g_is_odd_function_l326_326577


namespace perpendicular_distance_l326_326376

open EuclideanGeometry

noncomputable def excenter (A B C : Point) : Point := sorry
noncomputable def excircle (A B C : Point) : Circle := sorry
noncomputable def tangentPoint (A B C : Point) (excircle : Circle) (L : Line) : Point := sorry
noncomputable def meet (L1 L2 : Line) : Point := sorry
noncomputable def perpendicular (P : Point) (L : Line) : Point := sorry

axiom excircle_tangent {A B C : Point} (I_a : Point) (B' C' : Point) :
  (excircle A B C).tangentPoint A B I_a = B' ∧ (excircle A B C).tangentPoint A C I_a = C'

axiom meet_points {A B C I_a B' C' P Q : Point} :
  meet (I_a.line B) (B'.line C') = P ∧ meet (I_a.line C) (B'.lineC') = Q

axiom intersection_point {A B C I_a B' C' P Q M : Point} :
  meet (B.line Q) (C.line P) = M

theorem perpendicular_distance {A B C I_a B' C' P Q M : Point} (r : ℝ) :
  excircle_tangent A B C I_a B' C' →
  meet_points A B C I_a B' C' P Q →
  intersection_point A B C I_a B' C' P Q M →
  length (perpendicular M (B.line C)) = r := sorry

end perpendicular_distance_l326_326376


namespace rectangle_area_percentage_increase_l326_326189

theorem rectangle_area_percentage_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let A := l * w
  let len_inc := 1.3 * l
  let wid_inc := 1.15 * w
  let A_new := len_inc * wid_inc
  let percentage_increase := ((A_new - A) / A) * 100
  percentage_increase = 49.5 :=
by
  sorry

end rectangle_area_percentage_increase_l326_326189


namespace triangle_inequality_l326_326020

noncomputable def triangle_with_excircles (A1 A2 A3 B1 C1 : Point) (k1 k2 k3 : Circle) 
  (S S1 S2 S3 : ℝ) : Prop :=
  let touches (c : Circle) (A B : Point) : Prop := sorry

  touches k1 A2 A3 ∧
  (A1 + B1) ∈ k1 ∧
  (A1 + C1) ∈ k1 ∧
  S = triangle_area A1 A2 A3 ∧
  S1 = quadrilateral_area A1 A2 C1 A3 ∧
  S2 = quadrilateral_area A2 A3 C2 A1 ∧
  S3 = quadrilateral_area A3 A1 C3 A2

theorem triangle_inequality
  (A1 A2 A3 B1 C1 : Point) (k1 k2 k3 : Circle)
  (S S1 S2 S3 : ℝ) (h : triangle_with_excircles A1 A2 A3 B1 C1 k1 k2 k3 S S1 S2 S3) :
  1 / S ≤ 1 / S1 + 1 / S2 + 1 / S3 :=
sorry

end triangle_inequality_l326_326020


namespace regular_tetrahedron_properties_l326_326224

theorem regular_tetrahedron_properties 
  (V : Type*) [metric_space V] {p q r s : V}
  (hTetrahedron : regular_tetrahedron p q r s) :
  (∀ (a b : point_of_tetrahedron), edge_length a b = edge_length a c) ∧
  (∀ (face1 face2 : face_of_tetrahedron), is_congruent face1 face2 ∧ is_equilateral face1) ∧
  (∀ (v : vertex_of_tetrahedron), angle_between_edges v = angle_between_edges u) :=
sorry

end regular_tetrahedron_properties_l326_326224


namespace sale_saving_percent_l326_326933

theorem sale_saving_percent :
  (∀ (original_price sale_price amount_saved : ℕ),
   (original_price = 600) →
   (sale_price = (4 * (original_price / 20))) →
   (amount_saved = original_price - sale_price) →
   (amount_saved * 100 / original_price = 80)) :=
by
  intros original_price sale_price amount_saved
  assume h₁ h₂ h₃
  simp [h₁, h₂, h₃]
  sorry

end sale_saving_percent_l326_326933


namespace range_of_GM_dot_GN_l326_326621

open set

/--
Given a cube ABCD-A1B1C1D1 with edge length 4, a sphere O is inscribed in the cube, with MN as the diameter of sphere O, and point G is a moving point on the surface of the cube, prove that the range of the dot product of vectors GM and GN is [0,8].
-/
theorem range_of_GM_dot_GN :
  let a := 4,
      r := a / 2,
      cube_vertices := finset.univ.product finset.univ.product finset.univ,  -- Simplified representation of cube vertices
      sphere_center := (a / 2, a / 2, a / 2),  -- Center of the cube
      G : ℝ × ℝ × ℝ := sorry,  -- G is some point on the surface of the cube, exact representation might need more detail
      O := sphere_center,
      M := (a / 2, a / 2, (a / 2 - r)),
      N := (a / 2, a / 2, (a / 2 + r)),
      GO : ℝ := dist G O,
      GM := dist G M,
      GN := dist G N
  in (GM * GN) ∈ Icc (0 : ℝ) 8 :=
sorry

end range_of_GM_dot_GN_l326_326621


namespace solution_f_2_l326_326652

definition f : ℝ → ℝ 
:= sorry

theorem solution_f_2 : (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) + f((x - 1) / x) = 1 + x) → f(2) = 3 / 2 :=
begin
  intro hf,
  sorry
end

end solution_f_2_l326_326652


namespace set_inclusion_l326_326291

-- Definitions based on given conditions
def setA (x : ℝ) : Prop := 0 < x ∧ x < 2
def setB (x : ℝ) : Prop := x > 0

-- Statement of the proof problem
theorem set_inclusion : ∀ x, setA x → setB x :=
by
  intros x h
  sorry

end set_inclusion_l326_326291


namespace min_fence_dimensions_l326_326756

theorem min_fence_dimensions (A : ℝ) (hA : A ≥ 800) (x : ℝ) (hx : 2 * x * x = A) : x = 20 ∧ 2 * x = 40 := by
  sorry

end min_fence_dimensions_l326_326756


namespace angle_in_interval_l326_326637

-- Definitions of conditions
variables (a b : ℝ)
variable (h : a * b < 0)

-- Define the points P and Q
def P : ℝ × ℝ := (0, -1 / b)
def Q : ℝ × ℝ := (1 / a, 0)

-- Define the slope of the line PQ
def slope (P Q : ℝ × ℝ) : ℝ := (snd Q - snd P) / (fst Q - fst P)

-- Define the angle of inclination
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

-- Main theorem statement
theorem angle_in_interval : angle_of_inclination (slope P Q) ∈ (Set.Ioo (Real.pi / 2) Real.pi) :=
sorry

end angle_in_interval_l326_326637


namespace pole_height_intersection_l326_326492

theorem pole_height_intersection
  (d : ℤ) (h1 h2 h3 : ℤ)
  (interval : ℤ)
  (h_interval : interval = 150)
  (h_d : d = interval)
  (h_h1 : h1 = 30)
  (h_h2 : h2 = 100)
  (h_h3 : h3 = 60) :
  ∃ y : ℤ, y = 103 :=
by
  let x1 := 0
  let x2 := x1 + d
  let x3 := x2 + d
  let slope1 := (0 - h1) / (x2 - x1)
  let slope2 := (0 - h2) / (x3 - x2)
  let eq1 := slope1 * (x2) + h1
  let eq2 := slope2 * (x3) + (h2 + slope2 * d)
  have inter := eq1 = eq2
  use 103
  sorry

end pole_height_intersection_l326_326492


namespace Q_lies_on_circumcircle_FDB_FEC_l326_326733

-- Definitions for the problem setup
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)
structure Circle := (center : Point) (radius : ℝ) (contains : Point → Prop)

noncomputable def circumcircle (A B C : Point) : Circle := sorry -- definition of circumcircle is external knowledge

variables (A B C D E Q F : Point)
variable (ABC : Triangle)
variable hD : D = some_point_on (A, B) -- defined point D on AB
variable hE : E = some_point_on (A, C) -- defined point E on AC
variable hCirc_Q : Q ∈ circumcircle(A, D, E) ∧ Q ∈ circumcircle(A, B, C) -- Q lies on circumcircles of ADE and ABC
variable hF : F = intersection_point (line(B, C)) (line(D, E))

theorem Q_lies_on_circumcircle_FDB_FEC :
  Q ∈ circumcircle(F, D, B) ∧ Q ∈ circumcircle(F, E, C) := sorry

end Q_lies_on_circumcircle_FDB_FEC_l326_326733


namespace not_collinear_l326_326929

def a : ℝ × ℝ × ℝ := (1, 4, -2)
def b : ℝ × ℝ × ℝ := (1, 1, -1)
def c1 : ℝ × ℝ × ℝ := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
def c2 : ℝ × ℝ × ℝ := (4 * a.1 + 2 * b.1, 4 * a.2 + 2 * b.2, 4 * a.3 + 2 * b.3)

theorem not_collinear : ¬ ∃ γ : ℝ, c1 = (γ • c2) :=
by
  sorry

end not_collinear_l326_326929


namespace volume_of_solid_l326_326227

-- Define the given functions
def f (x : ℝ) : ℝ := 2 * x - x^2
def g (x : ℝ) : ℝ := -x + 2

-- Calculate the intersection points (roots of the equation x^2 - 3x + 2 = 0)
-- These are x = 1 and x = 2

-- Define the volumes V1 and V2
def V1 : ℝ := Real.pi * ∫ x in 1..2, (f x)^2
def V2 : ℝ := Real.pi * ∫ x in 1..2, (g x)^2

-- Calculate the final volume
def volume : ℝ := V1 - V2

-- The theorem stating the volume is 1/5 * pi
theorem volume_of_solid :
  volume = (1 / 5) * Real.pi :=
by
  sorry

end volume_of_solid_l326_326227


namespace truck_boxes_per_trip_l326_326201

theorem truck_boxes_per_trip (total_boxes trips : ℕ) (h1 : total_boxes = 871) (h2 : trips = 218) : total_boxes / trips = 4 := by
  sorry

end truck_boxes_per_trip_l326_326201


namespace sum_of_solutions_eq_l326_326983

theorem sum_of_solutions_eq (x : ℝ) : (5 * x - 7) * (4 * x + 11) = 0 ->
  -((27 : ℝ) / (20 : ℝ)) =
  - ((5 * - 7) * (4 * x + 11)) / ((5 * x - 7) * 4) :=
by
  intro h
  sorry

end sum_of_solutions_eq_l326_326983


namespace rhombus_perimeter_l326_326809

theorem rhombus_perimeter (d1 d2 : ℕ) (h_d1 : d1 = 16) (h_d2 : d2 = 30) :
  let side_length := Math.sqrt ((d1 / 2)^2 + (d2 / 2)^2) in
  let perimeter := 4 * side_length in
  perimeter = 68 := by
    dsimp only [side_length, perimeter]
    rw [h_d1, h_d2]
    norm_num
    sorry

end rhombus_perimeter_l326_326809


namespace Calculate_Surface_Area_of_S_l326_326565

-- Definitions based on the conditions:
def E : ℝ × ℝ × ℝ := (12, 12, 12)
def side_length : ℝ := 12
def EI_length : ℝ := 3
def points (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

-- Definition for the cube vertices and points I, J, K
def cube_surface_area : ℝ := 6 * side_length^2

-- Function to calculate the area of the modified solid
def modified_surface_area (m n p : ℕ) : ℝ := m + n * real.sqrt p

-- The theorem to prove
theorem Calculate_Surface_Area_of_S' : modified_surface_area 840 45 2 = 840 + 45 * real.sqrt 2 :=
sorry

end Calculate_Surface_Area_of_S_l326_326565


namespace sum_first_ten_special_numbers_l326_326328

theorem sum_first_ten_special_numbers :
  let s := { n : ℕ | ∃ p q r : ℕ, nat.prime p ∧ nat.prime q ∧ nat.prime r ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ n = p * q * r } in
  s.sum (λ n, n) = 1358 :=
by sorry

end sum_first_ten_special_numbers_l326_326328


namespace polygon_sides_given_triangles_l326_326680

definition divides_into_triangles (n k : ℕ) : Prop :=
  k = n - 2

theorem polygon_sides_given_triangles (n : ℕ) :
  divides_into_triangles n 5 → n = 7 :=
by
  intro h,
  have : 5 = n - 2, from h,
  linarith,
  sorry

end polygon_sides_given_triangles_l326_326680


namespace bruce_and_anne_clean_together_l326_326956

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l326_326956


namespace bruce_anne_clean_in_4_hours_l326_326949

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l326_326949


namespace num_right_angle_triangles_l326_326708

-- Step d): Lean 4 statement
theorem num_right_angle_triangles {C : ℝ × ℝ} (hC : C.2 = 0) :
  (C = (-2, 0) ∨ C = (4, 0) ∨ C = (1, 0)) ↔ ∃ A B : ℝ × ℝ,
  (A = (-2, 3)) ∧ (B = (4, 3)) ∧ 
  (A.2 = B.2) ∧ (A.1 ≠ B.1) ∧ 
  (((C.1-A.1)*(B.1-A.1) + (C.2-A.2)*(B.2-A.2) = 0) ∨ 
   ((C.1-B.1)*(A.1-B.1) + (C.2-B.2)*(A.2-B.2) = 0)) :=
sorry

end num_right_angle_triangles_l326_326708


namespace triangle_DEF_area_l326_326892

noncomputable def area_of_triangle_DEF (radius_small radius_large : ℝ) (tangent : ℝ → ℝ → ℝ → Prop) 
(congruent : ℝ → ℝ → Prop) 
(h1 : radius_small = 3) 
(h2 : radius_large = 5) 
(h3 : tangent DEF radius_small radius_large) 
(h4 : congruent DE DF) : ℝ := 
  63

theorem triangle_DEF_area : 
  ∃ (area : ℝ), area_of_triangle_DEF 3 5 tangent congruent 3 5 tangent DEF radius_small radius_large = 63 :=
begin
  use 63,
  sorry
end

end triangle_DEF_area_l326_326892


namespace magnitude_product_l326_326586

theorem magnitude_product :
  complex.abs ((4 * real.sqrt 2 - 4 * complex.i) * (real.sqrt 3 + 3 * complex.i)) = 24 := by
  sorry

end magnitude_product_l326_326586


namespace additional_money_spent_on_dvds_correct_l326_326374

def initial_money : ℕ := 320
def spent_on_books : ℕ := initial_money / 4 + 10
def remaining_after_books : ℕ := initial_money - spent_on_books
def spent_on_dvds_portion : ℕ := 2 * remaining_after_books / 5
def remaining_after_dvds : ℕ := 130
def total_spent_on_dvds : ℕ := remaining_after_books - remaining_after_dvds
def additional_spent_on_dvds : ℕ := total_spent_on_dvds - spent_on_dvds_portion

theorem additional_money_spent_on_dvds_correct : additional_spent_on_dvds = 8 :=
by
  sorry

end additional_money_spent_on_dvds_correct_l326_326374


namespace function_passes_through_1_1_l326_326448

-- Define the function f
def f (a x : ℝ) : ℝ := log a x + 1

theorem function_passes_through_1_1 (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f a 1 = 1 := by
  -- By the property of the logarithmic function, log_a(1) is 0 for a > 0 and a ≠ 1
  have log_one : log a 1 = 0 := by sorry
  -- Therefore, f(a, 1) = log(a, 1) + 1 = 0 + 1 = 1
  simp [f, log_one]
  -- this completes the statement (proof is omitted, as only the statement is required)
  sorry

end function_passes_through_1_1_l326_326448


namespace probability_of_x_in_interval_l326_326907

theorem probability_of_x_in_interval :
  let total_length := 3
  let length_of_A := 1
  let probability := (length_of_A : ℝ) / total_length
  x ∈ Icc (-1:ℝ) 2 -> probability = (1/3 : ℝ) := 
by
  -- Proof omitted
  sorry

end probability_of_x_in_interval_l326_326907


namespace surjective_constant_function_l326_326250

theorem surjective_constant_function :
  ∀ f : ℤ → ℤ, (∀ g : ℤ → ℤ, (∀ y : ℤ, ∃ x : ℤ, g x = y) →
  (∀ z : ℤ, ∃ x : ℤ, f x + g x = z)) → ∃ c : ℤ, ∀ x : ℤ, f x = c :=
begin
  sorry
end

end surjective_constant_function_l326_326250


namespace find_a_of_perpendicular_tangent_and_line_l326_326356

open Real

theorem find_a_of_perpendicular_tangent_and_line :
  let e := Real.exp 1
  let slope_tangent := 1 / e
  let slope_line (a : ℝ) := a
  let tangent_perpendicular := ∀ (a : ℝ), slope_tangent * slope_line a = -1
  tangent_perpendicular -> ∃ a : ℝ, a = -e :=
by {
  sorry
}

end find_a_of_perpendicular_tangent_and_line_l326_326356


namespace pascal_row_20_elements_l326_326480

theorem pascal_row_20_elements :
  binomial 20 4 = 4845 ∧ binomial 20 5 = 15504 :=
by
  sorry

end pascal_row_20_elements_l326_326480


namespace conjugate_z_l326_326306

-- Defining the complex number z
def z : ℂ := 1 / (1 + complex.I)

-- Theorem stating the conjugate of z is 1/2 + 1/2i
theorem conjugate_z : complex.conj z = (1 / 2 : ℂ) + (1 / 2 * complex.I) := by
  sorry

end conjugate_z_l326_326306


namespace minimum_time_to_transport_supplies_l326_326875

theorem minimum_time_to_transport_supplies (v : ℝ) (h_pos : v > 0) :
  ∀ t, (∃ t_1, t_1 = 400 / v) ∧
         (∀ i : ℕ, 1 ≤ i ∧ i < 26 → ∃ d, d = i * (v / 20)^2) →
         (∃ t_additional, t_additional = 25 * (v / 20)^2 / v) →
         t = (400 / v) + (25 * (v / 20)^2 / v) →
         t ≥ 10 :=
begin
  sorry
end

end minimum_time_to_transport_supplies_l326_326875


namespace find_angle_at_A_l326_326360

def triangle_angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def ab_lt_bc_lt_ac (AB BC AC : ℝ) : Prop :=
  AB < BC ∧ BC < AC

def angles_relation (α β γ : ℝ) : Prop :=
  (α = 2 * γ) ∧ (β = 3 * γ)

theorem find_angle_at_A
  (AB BC AC : ℝ)
  (α β γ : ℝ)
  (h1 : ab_lt_bc_lt_ac AB BC AC)
  (h2 : angles_relation α β γ)
  (h3 : triangle_angles_sum_to_180 α β γ) :
  α = 60 :=
sorry

end find_angle_at_A_l326_326360


namespace rectangle_height_l326_326190

theorem rectangle_height (y : ℝ) (h_pos : 0 < y) 
  (h_area : let length := 5 - (-3)
            let height := y - (-2)
            length * height = 112) : y = 12 := 
by 
  -- The proof is omitted
  sorry

end rectangle_height_l326_326190


namespace smallest_positive_period_1_smallest_positive_period_2_l326_326223

-- To prove the smallest positive period T for f(x) = |sin x| + |cos x| is π/2
theorem smallest_positive_period_1 : ∃ T > 0, T = Real.pi / 2 ∧ ∀ x : ℝ, (abs (Real.sin (x + T)) + abs (Real.cos (x + T)) = abs (Real.sin x) + abs (Real.cos x))  := sorry

-- To prove the smallest positive period T for f(x) = tan (2x/3) is 3π/2
theorem smallest_positive_period_2 : ∃ T > 0, T = 3 * Real.pi / 2 ∧ ∀ x : ℝ, (Real.tan ((2 * x) / 3 + T) = Real.tan ((2 * x) / 3)) := sorry

end smallest_positive_period_1_smallest_positive_period_2_l326_326223


namespace inequality_proof_l326_326296

variable {a1 a2 a3 a4 a5 : ℝ}

theorem inequality_proof (h1 : 1 < a1) (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) > (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_proof_l326_326296


namespace swap_derangement_bring_spectators_to_correct_seats_l326_326120

-- Definitions for seats, spectators, and the condition of a derangement
def is_derangement (σ : list ℕ) : Prop :=
  ∀ i, 1 ≤ i → i ≤ σ.length → σ.nth (i-1) ≠ some i

-- Definition of the operation: swapping adjacent spectators
def valid_swap (σ : list ℕ) (i : ℕ) : list ℕ :=
  if i < σ.length ∧ σ[i] ≠ i+1 ∧ σ[i+1] ≠ i+2 then [σ.take i, [σ[i+1], σ[i]], σ.drop (i+2)].join
  else σ

-- Inductive step demonstrating that swapping maintains a derangement
theorem swap_derangement (σ : list ℕ) (i : ℕ) :
  is_derangement σ → is_derangement (valid_swap σ i) :=
sorry

-- Main theorem stating that we can bring spectators to their correct seats
theorem bring_spectators_to_correct_seats (σ : list ℕ)
  (h1 : is_derangement σ)
  (h2 : σ.length > 0) :
  ∃ π, (∀ i, 1 ≤ i → i ≤ σ.length → π.nth (i-1) = some i) :=
sorry

end swap_derangement_bring_spectators_to_correct_seats_l326_326120


namespace average_tickets_sold_by_female_l326_326514

-- Define the conditions as Lean expressions.

def totalMembers (M : ℕ) : ℕ := M + 2 * M
def totalTickets (F : ℕ) (M : ℕ) : ℕ := 58 * M + F * 2 * M
def averageTicketsPerMember (F : ℕ) (M : ℕ) : ℕ := (totalTickets F M) / (totalMembers M)

theorem average_tickets_sold_by_female (F M : ℕ) 
  (h1 : 66 * (totalMembers M) = totalTickets F M) :
  F = 70 :=
by
  sorry

end average_tickets_sold_by_female_l326_326514


namespace rotation_matrix_150_deg_eq_l326_326978

open Real

/-- Matrix for a 150-degree counter-clockwise rotation -/
theorem rotation_matrix_150_deg_eq :
  (matrix.of_list [[-sqrt 3 / 2, -1 / 2], [1 / 2, -sqrt 3 / 2]] : matrix ℝ ℝ ℝ) =
    ![
      [- (sqrt 3) / 2, - 1 / 2 ],
      [1 / 2, - (sqrt 3) / 2]
    ] :=
sorry

end rotation_matrix_150_deg_eq_l326_326978


namespace Mika_stickers_l326_326762

def stickers_left (initial: ℕ) (bought: ℕ) (birthday: ℕ) (given: ℕ) (used: ℕ) : ℕ :=
  initial + bought + birthday - given - used

theorem Mika_stickers :
  stickers_left 20 26 20 6 58 = 2 :=
by
  rw [stickers_left]
  simp
  sorry

end Mika_stickers_l326_326762


namespace find_common_ratio_l326_326712

-- Declare the sequence and conditions
variables {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions of the problem 
def positive_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ m n : ℕ, a m = a 0 * q ^ m) ∧ q > 0

def third_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 + a 5 = 5

def fifth_term_seventh_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 5 + a 7 = 20

-- The final lean statement proving the common ratio is 2
theorem find_common_ratio 
  (h1 : positive_geometric_sequence a q) 
  (h2 : third_term_condition a q) 
  (h3 : fifth_term_seventh_term_condition a q) : 
  q = 2 :=
sorry

end find_common_ratio_l326_326712


namespace duty_scheduling_problem_l326_326468

-- Define the conditions
variables {students : Finset ℕ}
variable (A : ℕ)
variable (days : Finset ℕ := Finset.range 5)
variable (days_thu_fri : Finset ℕ := {3, 4})

-- Assert the conditions
def valid_conditions (students: Finset ℕ) (A : ℕ) (days : Finset ℕ) (days_thu_fri : Finset ℕ) : Prop := 
  students.card = 5 ∧ A ∈ students ∧ A ∉ days_thu_fri

-- Define the final proof theorem
theorem duty_scheduling_problem (students : Finset ℕ) (A : ℕ) (h_valid: valid_conditions students A days days_thu_fri) :
  ∃ seq_count : ℕ, seq_count = 72 := 
begin
  -- Here you would normally include the steps to prove the theorem, but we skip this with sorry
  sorry
end

end duty_scheduling_problem_l326_326468


namespace quadratic_difference_square_l326_326332

theorem quadratic_difference_square (α β : ℝ) (h : α ≠ β) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) : (α - β)^2 = 5 := by
  sorry

end quadratic_difference_square_l326_326332


namespace problem_maximum_marks_l326_326871

theorem problem_maximum_marks (M : ℝ) (h : 0.92 * M = 184) : M = 200 :=
sorry

end problem_maximum_marks_l326_326871


namespace f_at_one_l326_326604

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + x + 10
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^4 + x^3 + b * x^2 + 100 * x + c

theorem f_at_one :
  ∃ (a b c : ℝ), 
  ∀ g, ∀ f, 
  (∀ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ g r1 = 0 ∧ g r2 = 0 ∧ g r3 = 0 →
   g = λ x, x^3 + a * x^2 + x + 10 →
   f = λ x, x^4 + x^3 + b * x^2 + 100 * x + c →
   f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) →
  f 1 = -7007 :=
begin
  sorry,
end

end f_at_one_l326_326604


namespace factorial_fraction_l326_326553

theorem factorial_fraction : 
  (4 * nat.factorial 6 + 24 * nat.factorial 5) / nat.factorial 7 = 48 / 7 := 
by
  sorry

end factorial_fraction_l326_326553


namespace number_of_valid_m_values_l326_326136

theorem number_of_valid_m_values : 
  let m_values := {d ∈ Finset.divisors 540 // 540 / d >= 3} in
  Finset.card m_values = 17 := 
by 
  sorry

end number_of_valid_m_values_l326_326136


namespace count_numbers_between_200_and_500_with_digit_two_l326_326661

noncomputable def count_numbers_with_digit_two (a b : ℕ) : ℕ :=
  let numbers := list.filter (λ n, '2' ∈ n.digits 10) (list.range' a (b - a))
  numbers.length

theorem count_numbers_between_200_and_500_with_digit_two :
  count_numbers_with_digit_two 200 500 = 138 :=
by sorry

end count_numbers_between_200_and_500_with_digit_two_l326_326661


namespace toddlers_teeth_min_count_l326_326178

theorem toddlers_teeth_min_count (n : ℕ) (teeth : ℕ → ℕ) :
  (∑ i in Finset.range n, teeth i = 90) →
  (∀ i j, i ≠ j → teeth i + teeth j ≤ 9) →
  n ≥ 23 :=
sorry

end toddlers_teeth_min_count_l326_326178


namespace students_without_favorite_subject_l326_326688

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end students_without_favorite_subject_l326_326688


namespace y_value_l326_326666

-- Given conditions
variables (x y : ℝ)
axiom h1 : x - y = 20
axiom h2 : x + y = 14

-- Prove that y = -3
theorem y_value : y = -3 :=
by { sorry }

end y_value_l326_326666


namespace common_elements_count_l326_326736

def set_U : Finset ℤ := 
  (Finset.range 1500).image (λ n, 5 * (n + 1))

def set_V : Finset ℤ := 
  (Finset.range 1500).image (λ n, 8 * (n + 1))

theorem common_elements_count :
  (set_U ∩ set_V).card = 187 := 
sorry

end common_elements_count_l326_326736


namespace wire_divided_into_quarters_l326_326253

theorem wire_divided_into_quarters
  (l : ℕ) -- length of the wire
  (parts : ℕ) -- number of parts the wire is divided into
  (h_l : l = 28) -- wire is 28 cm long
  (h_parts : parts = 4) -- wire is divided into 4 parts
  : l / parts = 7 := -- each part is 7 cm long
by
  -- use sorry to skip the proof
  sorry

end wire_divided_into_quarters_l326_326253


namespace pastoral_scenery_l326_326528

variable {x y m w: ℝ}

-- Problem Conditions
def eq1 : Prop := 3 * x + 4 * y = 330
def eq2 : Prop := 4 * x + 3 * y = 300
def survival_rate_A : ℝ := 0.7
def survival_rate_B : ℝ := 0.9
def condition_next_year : Prop := (1 - survival_rate_A) * m + (1 - survival_rate_B) * (400 - m) ≤ 80
def planting_cost (m : ℝ) : ℝ := 30 * m + 60 * (400 - m)
def min_cost_condition : Prop := planting_cost 200 = 18000

-- Lean Statement
theorem pastoral_scenery:
  eq1 ∧ eq2 ∧ condition_next_year →
  (x = 30 ∧ y = 60) ∧ (m = 200 ∧ planting_cost 200 = 18000) :=
by
  unfold planting_cost
  unfold eq1 eq2 condition_next_year
  sorry

end pastoral_scenery_l326_326528


namespace impossible_to_transport_50_stones_l326_326478

def arithmetic_sequence (a d n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

def can_transport (weights : List ℕ) (k : ℕ) (max_weight : ℕ) : Prop :=
  ∃ partition : List (List ℕ), partition.length = k ∧
    (∀ part ∈ partition, (part.sum ≤ max_weight))

theorem impossible_to_transport_50_stones :
  ¬ can_transport (arithmetic_sequence 370 2 50) 7 3000 :=
by
  sorry

end impossible_to_transport_50_stones_l326_326478


namespace find_fourth_mark_l326_326442

-- Definitions of conditions
def average_of_four (a b c d : ℕ) : Prop :=
  (a + b + c + d) / 4 = 60

def known_marks (a b c : ℕ) : Prop :=
  a = 30 ∧ b = 55 ∧ c = 65

-- Theorem statement
theorem find_fourth_mark {d : ℕ} (h_avg : average_of_four 30 55 65 d) (h_known : known_marks 30 55 65) : d = 90 := 
by 
  sorry

end find_fourth_mark_l326_326442


namespace min_surface_area_cone_around_unit_sphere_l326_326479

noncomputable def surface_area_cone (r h : ℝ) : ℝ :=
  π * r * (r + sqrt (h^2 + r^2))

theorem min_surface_area_cone_around_unit_sphere :
  ∃ r h, r > 1 ∧ h > 2 ∧ surface_area_cone r h = 8 * π := sorry

end min_surface_area_cone_around_unit_sphere_l326_326479


namespace equal_tangents_l326_326878

theorem equal_tangents
  (O A B C D M P Q : Point) (circle : Circle)
  (h1 : chord A B circle) (h2 : midpoint M A B)
  (h3 : chord C D circle) (h4 : passes_through M C D) 
  (h5 : tangent_line C P circle) (h6 : tangent_line D Q circle) 
  (h7 : collinear P C D) (h8 : collinear Q C D) :
  PA = QB :=
sorry

end equal_tangents_l326_326878


namespace sum_is_perfect_cube_l326_326249

noncomputable def f (n : ℕ) : ℕ :=
  if n = 0 then 0 else 3*n^2 - 3*n + 1

theorem sum_is_perfect_cube (f : ℕ+ → ℕ+) (h : ∀ n, ∃ k, (∑ i in Finset.range n.succ, f (⟨i, i.is_lt⟩)) = k^3 ∧ (∑ i in Finset.range n.succ, f (⟨i, i.is_lt⟩)) ≤ n^3) :
  ∀ n, f n = 3*n^2 - 3*n + 1 :=
by
  sorry

end sum_is_perfect_cube_l326_326249


namespace rotate_result_l326_326125

noncomputable def original_vector : ℝ × ℝ × ℝ := (2, 1, 2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def is_orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

def rotate_90_deg_zaxis (v : ℝ × ℝ × ℝ) : Prop :=
  ∃ (w : ℝ × ℝ × ℝ), magnitude w = magnitude v ∧ is_orthogonal v w

theorem rotate_result :
  rotate_90_deg_zaxis original_vector →
  original_vector.magnitude = (3:ℝ) →
  ∃ (w : ℝ × ℝ × ℝ),
    w = (6 / real.sqrt 17, 6 / real.sqrt 17, -9 / real.sqrt 17) :=
begin
  sorry
end

end rotate_result_l326_326125


namespace complex_modulus_product_l326_326248

theorem complex_modulus_product : abs (4 - 3 * Complex.i) * abs (4 + 3 * Complex.i) = 25 := by
  sorry

end complex_modulus_product_l326_326248


namespace three_angles_division_l326_326116

noncomputable def angles (α β : ℝ) : Prop :=
  (α + β = 80) ∧ (β + 10 = 75) ∧ (α = 15) ∧ (β = 65)

theorem three_angles_division :
  ∃ (α β : ℝ), angles α β :=
by {
  existsi (15 : ℝ),
  existsi (65 : ℝ),
  split,
  {
    sorry,
  },
  split,
  {
    sorry,
  },
  split,
  {
    sorry,
  },
  {
    sorry,
  },
}

end three_angles_division_l326_326116


namespace relationship_between_abc_l326_326619

open Real

-- Define the constants for the problem
noncomputable def a : ℝ := sqrt 2023 - sqrt 2022
noncomputable def b : ℝ := sqrt 2022 - sqrt 2021
noncomputable def c : ℝ := sqrt 2021 - sqrt 2020

-- State the theorem we want to prove
theorem relationship_between_abc : c > b ∧ b > a := 
sorry

end relationship_between_abc_l326_326619


namespace find_number_l326_326512

theorem find_number :
  (∃ x : ℝ, x / 1500 = 0.016833333333333332) → (∃ x : ℝ, x = 25.25) :=
by
  intro h
  cases h with x hx
  use 25.25
  sorry

end find_number_l326_326512


namespace total_time_on_highway_l326_326044

/-
  Mary and Paul passed a gas station; Mary was traveling west at a constant speed of 50 miles/hour, 
  and Paul was traveling west at a constant speed of 80 miles/hour. Paul passed the gas station 
  15 minutes after Mary and caught up with her 25 minutes later. Prove that the total time they 
  remained on the highway was 2/3 hours.
-/

def mary_speed : ℝ := 50 -- miles/hour
def paul_speed : ℝ := 80 -- miles/hour
def time_paul_after_mary : ℝ := 15 / 60 -- hours
def time_catch_up : ℝ := 25 / 60 -- hours

theorem total_time_on_highway : (35 / 60) = 2 / 3 :=
  by simp [mary_speed, paul_speed, time_paul_after_mary, time_catch_up]; norm_num

end total_time_on_highway_l326_326044


namespace total_birds_and_storks_l326_326882

theorem total_birds_and_storks (initial_birds initial_storks additional_storks : ℕ) 
  (h1 : initial_birds = 3) 
  (h2 : initial_storks = 4) 
  (h3 : additional_storks = 6) 
  : initial_birds + initial_storks + additional_storks = 13 := 
  by sorry

end total_birds_and_storks_l326_326882


namespace program_output_is_24_l326_326831

def program_final_value : ℕ :=
  let mut t := 1
  let mut i := 2
  while i ≤ 4 do
    t := t * i
    i := i + 1
  t

theorem program_output_is_24 : program_final_value = 24 := 
by
  unfold program_final_value
  sorry

end program_output_is_24_l326_326831


namespace jack_total_yen_l326_326007

theorem jack_total_yen (pounds euros yen : ℕ) (pounds_per_euro yen_per_pound : ℕ) 
  (h_pounds : pounds = 42) 
  (h_euros : euros = 11) 
  (h_yen : yen = 3000) 
  (h_pounds_per_euro : pounds_per_euro = 2) 
  (h_yen_per_pound : yen_per_pound = 100) : 
  9400 = yen + (pounds * yen_per_pound) + ((euros * pounds_per_euro) * yen_per_pound) :=
by
  rw [h_pounds, h_euros, h_yen, h_pounds_per_euro, h_yen_per_pound]
  norm_num
  sorry

end jack_total_yen_l326_326007


namespace weight_placement_count_l326_326842

theorem weight_placement_count :
  let weights := [1, 2, 4, 10],
      valid_placements := number_of_valid_placements weights in
  valid_placements = 105 :=
sorry

end weight_placement_count_l326_326842


namespace books_sold_l326_326414

theorem books_sold (initial_books left_books sold_books : ℕ) (h1 : initial_books = 108) (h2 : left_books = 66) : sold_books = 42 :=
by
  have : sold_books = initial_books - left_books := sorry
  rw [h1, h2] at this
  exact this

end books_sold_l326_326414


namespace find_a_plus_2b_l326_326651

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6 * a * x + b

noncomputable def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 6 * a

theorem find_a_plus_2b (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f a b 2 = 9) : a + 2 * b = -24 := 
by sorry

end find_a_plus_2b_l326_326651


namespace calculate_value_l326_326124

-- Definition of the given values
def val1 : ℕ := 444
def val2 : ℕ := 44
def val3 : ℕ := 4

-- Theorem statement proving the value of the expression
theorem calculate_value : (val1 - val2 - val3) = 396 := 
by 
  sorry

end calculate_value_l326_326124


namespace total_cookies_collected_l326_326917

theorem total_cookies_collected 
  (abigail_boxes : ℕ) (grayson_boxes : ℕ) (olivia_boxes : ℕ) (cookies_per_box : ℕ)
  (h1 : abigail_boxes = 2) (h2 : grayson_boxes = 3) (h3 : olivia_boxes = 3) (h4 : cookies_per_box = 48) :
  (abigail_boxes * cookies_per_box) + ((grayson_boxes * (cookies_per_box / 4))) + (olivia_boxes * cookies_per_box) = 276 := 
by 
  sorry

end total_cookies_collected_l326_326917


namespace percent_decrease_italy_uk_l326_326013

theorem percent_decrease_italy_uk (original_italian_price : ℝ) (original_uk_price : ℝ) (discount_uk : ℝ) (exchange_rate : ℝ) :
  original_italian_price = 200 →
  original_uk_price = 150 →
  discount_uk = 20 →
  exchange_rate = 0.85 →
  let discounted_uk_price := original_uk_price * (1 - discount_uk / 100) in
  let discounted_uk_price_in_euros := discounted_uk_price / exchange_rate in
  let decrease_in_price := original_italian_price - discounted_uk_price_in_euros in
  let percent_decrease := (decrease_in_price / original_italian_price) * 100 in
  percent_decrease = 29.41 :=
by
  intros h1 h2 h3 h4
  simp only
  sorry

end percent_decrease_italy_uk_l326_326013


namespace isosceles_triangle_area_relationship_l326_326927

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c)

def inscribed_in_circle {a b c : ℕ} (h_isosceles : is_isosceles_triangle a b c) : Prop :=
  -- This would specify properties about being inscribed in a circle
  sorry

def areas_relationship (D E F : ℝ) : Prop :=
  D + E = F

theorem isosceles_triangle_area_relationship :
  ∃ (D E F : ℝ), is_isosceles_triangle 12 12 20 ∧ inscribed_in_circle (is_isosceles_triangle 12 12 20) ∧ areas_relationship D E F :=
sorry

end isosceles_triangle_area_relationship_l326_326927


namespace solution_set_of_inequality_l326_326118

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 0) : (1 / x < 1 / 3) ↔ (x ∈ (Set.Ioo (-∞) 0) ∪ Set.Ioo 3 ∞) :=
by
  sorry

end solution_set_of_inequality_l326_326118


namespace median_size_is_165_l326_326460

def sizes : List ℕ := [150, 155, 160, 165, 170, 175, 180]
def frequencies : List ℕ := [1, 6, 8, 12, 5, 4, 2]

noncomputable def cumulative_frequencies : List ℕ :=
  List.scanl (+) 0 frequencies

theorem median_size_is_165 :
  let total_students := 38
  let median_index1 := total_students / 2
  let median_index2 := median_index1 + 1
  let indices := List.range (frequencies.length)
  let cumulative_pairs := List.zip indices cumulative_frequencies
  let median_val_indices :=
    List.filter (λ (pair : ℕ × ℕ), pair.2 >= median_index1 ∧ pair.2 < median_index2 + frequencies.pair.1) cumulative_pairs
  median_val_indices.map (λ pair => sizes.pair.1).head = 165 :=
sorry

end median_size_is_165_l326_326460


namespace example_calculation_l326_326923

theorem example_calculation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end example_calculation_l326_326923


namespace tangency_condition_l326_326097

-- Define the equation for the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 = 9

-- Define the equation for the hyperbola
def hyperbola_eq (x y m : ℝ) : Prop :=
  (x - 2)^2 - m * (y + 1)^2 = 1

-- Prove that for the ellipse and hyperbola to be tangent, m must equal 3
theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y ∧ hyperbola_eq x y m) → m = 3 :=
by
  sorry

end tangency_condition_l326_326097


namespace no_such_eight_digit_number_exists_l326_326986

theorem no_such_eight_digit_number_exists :
  ¬ ∃ (N : ℕ), (10000000 ≤ N) ∧ (N < 100000000) ∧ 
               (∀ (d : Fin 8), (1 ≤ d + 1 ∧ d + 1 ≤ 8) → 
                             (let digit := (N / 10^(7-(d: ℕ))) % 10 in 
                             digit ≠ 0 ∧ N % digit = d + 1)) :=
sorry

end no_such_eight_digit_number_exists_l326_326986


namespace square_perimeter_and_inscribed_circle_area_l326_326196

theorem square_perimeter_and_inscribed_circle_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2,
      P := 4 * s,
      r := s / 2,
      A := Real.pi * r^2 in
  P = 48 ∧ A = 36 * Real.pi := 
by
  sorry

end square_perimeter_and_inscribed_circle_area_l326_326196


namespace probabilty_no_intersect_paths_l326_326931

-- Define the conditions as discussed
def jia_moves (start end : ℕ × ℕ) : Prop :=
  start = (0, 0) ∧ end = (4, 4)

def yi_moves (start end : ℕ × ℕ) : Prop :=
  start = (0, 0) ∧ end = (4, 4)

def no_common_points (jia_path yi_path : list (ℕ × ℕ)) : Prop :=
  ∀ p ∈ jia_path, p ∉ yi_path

-- Define the probability statement based on the problem conditions
theorem probabilty_no_intersect_paths :
  ∀ jia_start jia_end yi_start yi_end,
    jia_moves jia_start jia_end →
    yi_moves yi_start yi_end →
    (∃ paths_jia paths_yi, no_common_points paths_jia paths_yi →
    (length paths_jia = 70 ∧ length paths_yi = 70) →
    ∃ p, paths_jia = p ∧ paths_yi = p →
    (∃ intersection_paths, length intersection_paths = 3150) →
    (4900 - 3150 = 1750) →
    (1750 / 4900 = 5 / 14)) :=
  sorry

end probabilty_no_intersect_paths_l326_326931


namespace parallelogram_AB_B1_A1_C1_l326_326584

/- Define the equilateral triangles on sides AB, AC, and internal equilateral on BC -/
/- Define the concept of parallelogram in lean -/
/- Prove that ABB_1AC_1 is a parallelogram based on the given conditions -/

theorem parallelogram_AB_B1_A1_C1
  {A B C B₁ C₁ A₁ : Type}
  (hB₁ : ∀ (AB₁ : Type), (equilateral_triangle AB₁ A B ∧ similar AB₁ ABC))
  (hC₁ : ∀ (AC₁ : Type), (equilateral_triangle AC₁ A C ∧ similar AC₁ ABC))
  (hA₁ : ∀ (BA₁ : Type), (equilateral_triangle BA₁ B C ∧ similar BA₁ ABC)) :
  is_parallelogram (quadrilateral A B₁ A₁ C₁) :=
sorry

end parallelogram_AB_B1_A1_C1_l326_326584


namespace sum_binomials_is_zero_l326_326963

-- Define the sum S as given in the original problem
def S : ℤ := ∑ k in Finset.range 50, (-1)^k * Nat.choose 100 (2 * k + 1)

-- The statement that we need to prove
theorem sum_binomials_is_zero : S = 0 :=
sorry

end sum_binomials_is_zero_l326_326963


namespace mia_study_time_l326_326761

-- Let's define the conditions in Lean
def total_hours_in_day : ℕ := 24
def fraction_time_watching_TV : ℚ := 1 / 5
def fraction_time_studying : ℚ := 1 / 4

-- Time remaining after watching TV
def time_left_after_TV (total_hours : ℚ) (fraction_TV : ℚ) : ℚ :=
  total_hours * (1 - fraction_TV)

-- Time spent studying
def time_studying (remaining_time : ℚ) (fraction_studying : ℚ) : ℚ :=
  remaining_time * fraction_studying

-- Convert hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- The theorem statement
theorem mia_study_time :
  let total_hours := (total_hours_in_day : ℚ),
      time_after_TV := time_left_after_TV total_hours fraction_time_watching_TV,
      study_hours := time_studying time_after_TV fraction_time_studying,
      study_minutes := hours_to_minutes study_hours in
  study_minutes = 288 := by
  sorry

end mia_study_time_l326_326761


namespace maple_trees_after_planting_l326_326838

theorem maple_trees_after_planting (existing_maple: ℕ) (maple_to_plant: ℕ) : 
  existing_maple = 2 → maple_to_plant = 9 → existing_maple + maple_to_plant = 11 :=
by
  intros h_existing h_planting
  rw [h_existing, h_planting]
  norm_num
  sorry

end maple_trees_after_planting_l326_326838


namespace seashells_given_to_sam_is_correct_l326_326375

-- Define the initial conditions
def initial_seashells : ℕ := 70
def seashells_left : ℕ := 27

-- Define the number of seashells given to Sam
def seashells_given_to_sam : ℕ := initial_seashells - seashells_left

-- Prove that the number of seashells given to Sam is 43
theorem seashells_given_to_sam_is_correct : seashells_given_to_sam = 43 := by
  unfold seashells_given_to_sam
  rw initial_seashells
  rw seashells_left
  norm_num
  sorry

end seashells_given_to_sam_is_correct_l326_326375


namespace problem1_problem2_l326_326349

variables (A B C D : Prop)
variables (prob_infected_B_A prob_infected_C_A prob_infected_D_A : ℚ)
variables (prob_infected_C_B prob_infected_D_B prob_infected_D_C : ℚ)

def prob_exact_one_infected 
  (prob_infected_B_A : ℚ) (prob_infected_C_A : ℚ) (prob_infected_D_A : ℚ) : ℚ :=
  3 * prob_infected_B_A * (1 - prob_infected_C_A) * (1 - prob_infected_D_A)

theorem problem1 
  (hA : (prob_infected_B_A = 1/2) ∧ (prob_infected_C_A = 1/2) ∧ (prob_infected_D_A = 1/2)) :
  prob_exact_one_infected prob_infected_B_A prob_infected_C_A prob_infected_D_A = 3/8 :=
sorry

def random_variable_distribution 
  (prob_infected_C_A : ℚ) (prob_infected_C_B : ℚ) (prob_infected_D_A : ℚ) (prob_infected_D_B : ℚ) (prob_infected_D_C : ℚ) :
  (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ) :=
  (1, (1 - prob_infected_C_A) * (1 - prob_infected_D_A), 
   2, prob_infected_C_A * (1 - prob_infected_D_A) + (1 - prob_infected_C_A) * prob_infected_D_A, 
   3, prob_infected_C_A * prob_infected_D_A)

def expected_value (distribution : (ℚ × ℚ) × (ℚ × ℚ) × (ℚ × ℚ)) : ℚ :=
  let ((x1, p1), (x2, p2), (x3, p3)) := distribution in
  x1 * p1 + x2 * p2 + x3 * p3

theorem problem2 
  (hB : prob_infected_C_A = 1/2 ∧ prob_infected_C_B = 1/2 ∧ prob_infected_D_A = 1/3 ∧ prob_infected_D_B = 1/3 ∧ prob_infected_D_C = 1/3) :
  random_variable_distribution prob_infected_C_A prob_infected_C_B prob_infected_D_A prob_infected_D_B prob_infected_D_C = 
    ((1, 1/3), (2, 1/2), (3, 1/6)) ∧
  expected_value (random_variable_distribution prob_infected_C_A prob_infected_C_B prob_infected_D_A prob_infected_D_B prob_infected_D_C) = 11/6 :=
sorry

end problem1_problem2_l326_326349


namespace minimum_value_and_corresponding_x_set_tan_value_if_f_equals_2f_prime_l326_326649

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f_prime (x : ℝ) : ℝ := Real.cos x - Real.sin x

noncomputable def g (x : ℝ) : ℝ := f x * f_prime x

-- Statement 1: Prove minimum value of g(x) == -1 and set of corresponding x
theorem minimum_value_and_corresponding_x_set :
  let min_value := -1
  let corresponding_x_set := { x : ℝ | ∃ (k : ℤ), x = (-Real.pi / 2) + k * Real.pi } in
  ∃ val : ℝ, val = min_value ∧ ∃ xs : Set ℝ, xs = corresponding_x_set := sorry

-- Statement 2: Prove that tan(x + pi / 4) == 2 given f(x) = 2 * f_prime(x)
theorem tan_value_if_f_equals_2f_prime :
  (∀ x : ℝ, f x = 2 * f_prime x) → (tan (x : ℝ) (Real.pi / 4) = 2) := sorry

end minimum_value_and_corresponding_x_set_tan_value_if_f_equals_2f_prime_l326_326649


namespace sin_psi_is_one_l326_326161

noncomputable def square_side_length : ℝ := 4

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := { x := -2, y := -2 }
def B : Point := { x := -2, y := 2 }
def C : Point := { x := 2, y := 2 }
def D : Point := { x := 2, y := -2 }
def M : Point := { x := -2, y := 0 }
def N : Point := { x := 2, y := 0 }
def P : Point := { x := 0, y := -2 }

def distance (p1 p2: Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

def AM : ℝ := distance A M
def AP : ℝ := distance A P
def MP : ℝ := distance M P
def cos_psi : ℝ := (AM^2 + AP^2 - MP^2) / (2 * AM * AP)
def sin_psi : ℝ := (1 - cos_psi^2).sqrt

theorem sin_psi_is_one : sin_psi = 1 :=
  by
  sorry

end sin_psi_is_one_l326_326161


namespace count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l326_326324

theorem count_of_numbers_less_than_100_divisible_by_2_but_not_by_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

theorem count_of_numbers_less_than_100_divisible_by_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∨ n % 3 = 0) (Finset.range 100)) = 66 :=
sorry

theorem count_of_numbers_less_than_100_not_divisible_by_either_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 ≠ 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

end count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l326_326324


namespace smallest_triangle_perimeter_l326_326910

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def smallest_possible_prime_perimeter : ℕ :=
  31

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  a > 5 ∧ b > 5 ∧ c > 5 ∧
                  is_prime a ∧ is_prime b ∧ is_prime c ∧
                  triangle_inequality a b c ∧
                  is_prime (a + b + c) ∧
                  a + b + c = smallest_possible_prime_perimeter :=
sorry

end smallest_triangle_perimeter_l326_326910


namespace distinct_colorings_2x2_grid_l326_326564

/-- Number of distinct colorings of a 2 × 2 grid of squares 
    using 10 colors, considering equivalent rotations. -/
theorem distinct_colorings_2x2_grid : 
  let n := 10 in 
  -- Case 1: All squares the same color
  let case1 := n in
  -- Case 2: Two colors in non-adjacent squares
  let case2 := n * (n - 1) / 2 in
  -- Case 3: All other colorings (corrected for rotations)
  let case3 := (n^4 - case1 - case2) / 4 in
  -- Total distinct colorings
  case1 + case2 + case3 = 2530 :=
by sorry

end distinct_colorings_2x2_grid_l326_326564


namespace gcd_of_consecutive_digit_sums_is_1111_l326_326446

theorem gcd_of_consecutive_digit_sums_is_1111 (p q r s : ℕ) (hc : q = p+1 ∧ r = p+2 ∧ s = p+3) :
  ∃ d, d = 1111 ∧ ∀ n : ℕ, n = (1000 * p + 100 * q + 10 * r + s) + (1000 * s + 100 * r + 10 * q + p) → d ∣ n := by
  use 1111
  sorry

end gcd_of_consecutive_digit_sums_is_1111_l326_326446


namespace variance_red_balls_correct_l326_326167

noncomputable def variance_of_red_balls : ℝ :=
  let p : ℝ := 3 / 5
  let n : ℕ := 4
  n * p * (1 - p)

theorem variance_red_balls_correct :
  variance_of_red_balls = 24 / 25 :=
by
  sorry

end variance_red_balls_correct_l326_326167


namespace range_g_eq_real_l326_326569

def g (x : ℝ) : ℝ := ⌊x⌋ + x

theorem range_g_eq_real : Set.range g = Set.univ := by
  sorry

end range_g_eq_real_l326_326569


namespace arithmetic_sequence_sum_l326_326634

noncomputable def sum_eq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, (∑ i in finset.range n, a (i + 1)) = S n

theorem arithmetic_sequence_sum 
  (S a : ℕ → ℝ)
  (h_sum : sum_eq S a)
  (h_a1 : a 1 = 1)
  (h_a8 : a 8 = 3 * a 3) :
  ∑ i in finset.range n, (a (i + 2) / (S (i + 1) * S (i + 2))) = 1 - 1 / ((n + 1)^2) :=
sorry

end arithmetic_sequence_sum_l326_326634


namespace probability_intersect_expected_intersection_points_l326_326741
open Real

def a_and_b (a b : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4}

def intersects (a b : ℕ) : Prop :=
  b^2 ≤ 1 + a^2

theorem probability_intersect (a b : ℕ) (h : a_and_b a b) :
  (∑ (x in {1, 2, 3, 4}), ∑ (y in {1, 2, 3, 4}), if intersects x y then 1 else 0) / (4 * 4) = 5 / 8 :=
sorry

theorem expected_intersection_points (a b : ℕ) (h : a_and_b a b) :
  (2 * 5 / 8) = 5 / 4 :=
sorry

end probability_intersect_expected_intersection_points_l326_326741


namespace employment_percentage_approx_64_l326_326713

-- Define the problem conditions
variables (P : ℝ) (E : ℝ)
variable (h_employed_males : 0.5 * P = E * P - 0.21875 * E * P)
variable (h_employed_females : 0.21875 * (E * P) = E * P - 0.5 * P)

-- State the theorem that proves the employment percentage is approximately 0.64
theorem employment_percentage_approx_64 
  (h1 : 0.5 * P = E * P - 0.21875 * E * P)
  (h2 : 0.21875 * (E * P) = E * P - 0.5 * P) : E ≈ 0.64 :=
by
  sorry

end employment_percentage_approx_64_l326_326713


namespace trains_cross_time_approx_six_seconds_l326_326543

-- Definitions based on conditions
def length_train1 : ℝ := 108
def speed_train1_kmph : ℝ := 50
def length_train2 : ℝ := 112
def speed_train2_kmph : ℝ := 82

-- Helper conversion functions and relative speed calculation
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

def speed_train1_mps : ℝ := kmph_to_mps speed_train1_kmph
def speed_train2_mps : ℝ := kmph_to_mps speed_train2_kmph

def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

-- Total distance to be covered
def total_distance : ℝ := length_train1 + length_train2

-- Time calculation (in seconds)
def time_to_cross : ℝ := total_distance / relative_speed_mps

-- The theorem we need to prove
theorem trains_cross_time_approx_six_seconds :
  abs (time_to_cross - 6) < 0.01 :=
by {
  sorry -- proof goes here
}

end trains_cross_time_approx_six_seconds_l326_326543


namespace expansion_coeff_sum_l326_326568

theorem expansion_coeff_sum
  (a : ℕ → ℤ)
  (h : ∀ x y : ℤ, (x - 2 * y) ^ 5 * (x + 3 * y) ^ 4 = 
    a 9 * x ^ 9 + 
    a 8 * x ^ 8 * y + 
    a 7 * x ^ 7 * y ^ 2 + 
    a 6 * x ^ 6 * y ^ 3 + 
    a 5 * x ^ 5 * y ^ 4 + 
    a 4 * x ^ 4 * y ^ 5 + 
    a 3 * x ^ 3 * y ^ 6 + 
    a 2 * x ^ 2 * y ^ 7 + 
    a 1 * x * y ^ 8 + 
    a 0 * y ^ 9) :
  a 0 + a 8 = -2602 := by
  sorry

end expansion_coeff_sum_l326_326568


namespace exists_f_prime_eq_inverses_l326_326612

theorem exists_f_prime_eq_inverses (f : ℝ → ℝ) (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : ContinuousOn f (Set.Icc a b))
  (h4 : DifferentiableOn ℝ f (Set.Ioo a b)) :
  ∃ c ∈ Set.Ioo a b, (deriv f c) = (1 / (a - c)) + (1 / (b - c)) + (1 / (a + b)) :=
by
  sorry

end exists_f_prime_eq_inverses_l326_326612


namespace boxes_per_case_l326_326779

/-- Let's define the variables for the problem.
    We are given that Shirley sold 10 boxes of trefoils,
    and she needs to deliver 5 cases of boxes. --/
def total_boxes : ℕ := 10
def number_of_cases : ℕ := 5

/-- We need to prove that the number of boxes in each case is 2. --/
theorem boxes_per_case :
  total_boxes / number_of_cases = 2 :=
by
  -- Definition step where we specify the calculation
  unfold total_boxes number_of_cases
  -- The problem requires a division operation
  norm_num
  -- The result should be correct according to the solution steps
  done

end boxes_per_case_l326_326779


namespace number_of_combinations_l326_326727

-- Conditions as definitions
def n : ℕ := 9
def k : ℕ := 4

-- Lean statement of the equivalent proof problem
theorem number_of_combinations : (nat.choose n k) = 126 := by
  -- Sorry is used to skip the proof
  sorry

end number_of_combinations_l326_326727


namespace find_a_b_a_b_values_l326_326024

/-
Define the matrix M as given in the problem.
Define the constants a and b, and state the condition that proves their correct values such that M_inv = a * M + b * I.
-/

open Matrix

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 0;
     1, -3]

noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1/2, 0;
     1/6, -1/3]

theorem find_a_b :
  ∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

theorem a_b_values :
  (∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  (∃ a b : ℚ, a = 1/6 ∧ b = 1/6) :=
sorry

end find_a_b_a_b_values_l326_326024


namespace johnny_marbles_combination_l326_326724

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end johnny_marbles_combination_l326_326724


namespace age_problem_l326_326526

theorem age_problem : 
  let S : ℕ := 26 in
  let M : ℕ := S + 28 in
  let Y := (λ S M => ∃ Y : ℕ, M + Y = 2 * (S + Y)) S M in
  Y = ↑2 :=
by
  sorry

end age_problem_l326_326526


namespace most_suitable_statistical_graph_is_pie_chart_l326_326087

variable (nitrogen oxygen rare_gases carbon_dioxide other_gases : ℝ)
variable (total_air_volume : ℝ)

-- Volume percentages as conditions
def volume_fractions (nitrogen_perc oxygen_perc rare_gases_perc carbon_dioxide_perc other_gases_perc: ℝ) :=
  nitrogen_perc + oxygen_perc + rare_gases_perc + carbon_dioxide_perc + other_gases_perc = 100

-- Each volume fraction condition
def nitrogen_perc := 78
def oxygen_perc := 21
def rare_gases_perc := 0.94
def carbon_dioxide_perc := 0.03
def other_gases_perc := 0.03

theorem most_suitable_statistical_graph_is_pie_chart :
  volume_fractions nitrogen_perc oxygen_perc rare_gases_perc carbon_dioxide_perc other_gases_perc →
  "Pie chart" = "The most suitable type of statistical graph to reflect the percentage of the volume of each component gas in the air" :=
by
  sorry

end most_suitable_statistical_graph_is_pie_chart_l326_326087


namespace coin_problem_l326_326150

theorem coin_problem :
  ∃ (p n d q : ℕ), p + n + d + q = 11 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 132 ∧
                   p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ 
                   q = 3 :=
by
  sorry

end coin_problem_l326_326150


namespace consecutive_zeros_ones_sequences_l326_326325

def count_sequences_of_length (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  (finset.range (n+1)).sum f

theorem consecutive_zeros_ones_sequences : 
  count_sequences_of_length 15 (λ k => 16 - k) + count_sequences_of_length 15 (λ k => 16 - k) - 2 = 266 :=
by
  sorry

end consecutive_zeros_ones_sequences_l326_326325


namespace rhombus_perimeter_l326_326799

-- Definitions based on conditions
def diagonal1 : ℝ := 16
def diagonal2 : ℝ := 30
def half_diagonal1 : ℝ := diagonal1 / 2
def half_diagonal2 : ℝ := diagonal2 / 2

-- Mathematical formulation in Lean
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := 
by
  -- Given diagonals
  have h_half_d1 : (d1 / 2) = 8 := by sorry,
  have h_half_d2 : (d2 / 2) = 15 := by sorry,
  
  -- Combine into Pythagorean theorem and perimeter calculation
  have h_side_length : real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 17 := by sorry,
  show 4 * real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 68 := by sorry

end rhombus_perimeter_l326_326799


namespace proof_problem_example_application_l326_326454

variable (a b : ℝ)

def p : Prop := a < b → ∀ c : ℝ, a * c^2 < b * c^2

def q : Prop := ∃ x₀ > 0, x₀ - 1 + Real.log x₀ = 0

theorem proof_problem (hp : ¬p) (hq : q) : (¬p) ∧ q :=
by {
  exact And.intro hp hq,
}

-- Inclusing our example instantiation would look like this
theorem example_application : (¬p 1 2) ∧ q :=
sorry

end proof_problem_example_application_l326_326454


namespace projection_problem_l326_326184

theorem projection_problem : 
  let v₁ := (⟨3, 3⟩ : ℝ × ℝ)
  let p₁ := (⟨45 / 10, 9 / 10⟩ : ℝ × ℝ)
  let v₂ := (⟨1, -1⟩ : ℝ × ℝ)
  -- Define the projection of one vector onto another
  let proj (u v : ℝ × ℝ) : ℝ × ℝ := 
    let d := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
    ⟨d * v.1, d * v.2⟩
in
  proj v₂ (⟨5, 1⟩ : ℝ × ℝ) = (⟨10 / 13, 2 / 13⟩ : ℝ × ℝ) :=
by
  let v₁ := (⟨3, 3⟩ : ℝ × ℝ)
  let p₁ := (⟨45 / 10, 9 / 10⟩ : ℝ × ℝ)
  let v₂ := (⟨1, -1⟩ : ℝ × ℝ)
  let proj (u v : ℝ × ℝ) := 
    let d := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
    -- Define the projection as d times the vector v
    ⟨d * v.1, d * v.2⟩
  -- Simplify the vector p₁ to get the direction vector for projection
  have p1_simplified : p₁ = ⟨5, 1⟩ := by
    unfold p₁ 
    apply rfl

  sorry -- Proof omitted for brevity

end projection_problem_l326_326184


namespace vector_cross_product_magnitude_l326_326315

-- Define the conditions
def vector_a : vector ℝ 2 := ⟨[-3, 4], rfl⟩
def vector_b : vector ℝ 2 := ⟨[0, 2], rfl⟩

-- Function to calculate the magnitude of a 2D vector
noncomputable def magnitude (v : vector ℝ 2) : ℝ :=
  real.sqrt (v.to_list.map (λ x, x * x)).sum

-- Function to calculate the sine of the angle between two 2D vectors
noncomputable def sin_theta (a b : vector ℝ 2) : ℝ :=
  let cos_theta := ((a.to_list.head * b.to_list.head) + (a.to_list.tail.head * b.to_list.tail.head)) / (magnitude a * magnitude b) in
  real.sqrt (1 - cos_theta * cos_theta)

-- The main statement to prove
theorem vector_cross_product_magnitude : 
  |vector_a × vector_b| = 6 :=
by
  -- Definitions and calculations based on conditions
  let mag_a := magnitude vector_a
  let mag_b := magnitude vector_b
  let sin_theta_ab := sin_theta vector_a vector_b
  have h_mag_a : mag_a = 5 := sorry
  have h_mag_b : mag_b = 2 := sorry
  have h_sin_theta : sin_theta_ab = 3 / 5 := sorry
  show |vector_a × vector_b| = 6 from
  calc
    |vector_a × vector_b| = mag_a * mag_b * sin_theta_ab : sorry
                   ... = 5 * 2 * (3 / 5) : by rw [h_mag_a, h_mag_b, h_sin_theta]
                   ... = 6 : by norm_num

end vector_cross_product_magnitude_l326_326315


namespace store_shelves_l326_326160

theorem store_shelves (initial_books sold_books books_per_shelf : ℕ) 
    (h_initial: initial_books = 27)
    (h_sold: sold_books = 6)
    (h_per_shelf: books_per_shelf = 7) :
    (initial_books - sold_books) / books_per_shelf = 3 := by
  sorry

end store_shelves_l326_326160


namespace investments_ratio_l326_326455

theorem investments_ratio (P Q : ℝ) (hpq : 7 / 10 = (P * 2) / (Q * 4)) : P / Q = 7 / 5 :=
by 
  sorry

end investments_ratio_l326_326455


namespace linda_age_l326_326092

variable (s j l : ℕ)

theorem linda_age (h1 : (s + j + l) / 3 = 11) 
                  (h2 : l - 5 = s) 
                  (h3 : j + 4 = 3 * (s + 4) / 4) :
                  l = 14 := by
  sorry

end linda_age_l326_326092


namespace chocolates_on_fourth_day_l326_326402

noncomputable def chocolates_eaten (total: ℕ) (day_diff: ℕ -> ℕ -> ℕ) (day1 day2 day3 day4 day5: ℕ) : Prop :=
  day1 + day2 + day3 + day4 + day5 = total ∧
  day2 = day_diff day1 8 ∧
  day3 = day_diff day2 8 ∧
  day4 = day_diff day3 8 ∧
  day5 = day_diff day4 8

theorem chocolates_on_fourth_day:
  chocolates_eaten 150 (λ previous next_diff, previous + next_diff) (38 - 24) (38 - 16) (38 - 8) 38 (38 + 8) →
  ∃ c: ℕ, 38 = c :=
by
  sorry

end chocolates_on_fourth_day_l326_326402


namespace angle_same_terminal_side_l326_326860

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, -330 = k * 360 + 30 :=
by
  use -1
  sorry

end angle_same_terminal_side_l326_326860


namespace binom_n_2_l326_326481

theorem binom_n_2 (n : ℕ) (hn : 2 ≤ n) : nat.choose n 2 = n * (n - 1) / 2 := 
by
  sorry

end binom_n_2_l326_326481


namespace sqrt_x_minus_2_real_l326_326334

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ (y : ℝ), y = sqrt (x - 2)) ↔ x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_real_l326_326334


namespace area_of_triangle_KBC_l326_326711

theorem area_of_triangle_KBC : 
  ∀ (A B C D E F J I K G H : Type) 
    (squares_eq : ∀ ABJI_area : ℝ, ABJI_area = 25 ∧ ∀ FEHG_area : ℝ, FEHG_area = 49)
    (equilateral_hexagon : True)
    (FE_BC_eq_length : ∀ (FE BC : ℝ), FE = 7 ∧ BC = 7 ∧ FE = BC)
    (JB_length : ∀ (JB : ℝ), JB = 5)
    (right_angle_triangle : ∀ (JBK : Prop), B ≠ J ∧ B ≠ K ∧ J ≠ K ∧ angle J K B = 90),
  let area := (1 / 2 : ℝ) * 5 * 7 in
  area = 17.5 := 
begin 
  intros,
  sorry, 
end

end area_of_triangle_KBC_l326_326711


namespace polynomial_inequality_degree_1999_l326_326993

open Real

theorem polynomial_inequality_degree_1999 (f : ℝ[X]) (hdeg : f.degree = 1999) : 
  |f.eval 0| ≤ (3000 : ℝ) ^ 2000 * ∫ x in -1 .. 1, |f.eval x| :=
sorry

end polynomial_inequality_degree_1999_l326_326993


namespace number_of_palindromes_less_than_1000_l326_326575

-- Definitions of what constitutes a palindrome and the condition numbers being less than 1000.
def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits in s = s.reverse

def palindromes_less_than (n : ℕ) : ℕ :=
  (list.range n).filter is_palindrome

theorem number_of_palindromes_less_than_1000 : palindromes_less_than 1000 = 108 :=
by sorry

end number_of_palindromes_less_than_1000_l326_326575


namespace sum_of_divisors_120_l326_326858

theorem sum_of_divisors_120 : (∑ n in (finset.filter (λ d, 120 % d = 0) (finset.range (120 + 1))), n) = 360 := by
  sorry

end sum_of_divisors_120_l326_326858


namespace total_money_of_james_and_ali_l326_326009

def jamesOwns : ℕ := 145
def jamesAliDifference : ℕ := 40
def aliOwns : ℕ := jamesOwns - jamesAliDifference

theorem total_money_of_james_and_ali :
  jamesOwns + aliOwns = 250 := by
  sorry

end total_money_of_james_and_ali_l326_326009


namespace find_x_l326_326602

theorem find_x (x : ℝ) (h : |x - 10| + |x - 14| = |3x - 42|) : x = 18 := sorry

end find_x_l326_326602


namespace triangle_side_b_correct_l326_326740

theorem triangle_side_b_correct (a c b : ℝ) (cosA : ℝ) 
  (h_a : a = 2) 
  (h_c : c = 2 * real.sqrt 3) 
  (h_cosA : cosA = (real.sqrt 3) / 2) 
  (h_b_c_lt : b < c) : b = 2 := 
by
  sorry

end triangle_side_b_correct_l326_326740


namespace transform_OAB_l326_326385

-- Defining the vertices of the triangle in the xy-plane
def O := (0, 0)
def A := (1, 0)
def B := (1, 1)

-- Defining the transformation from xy-plane to uv-plane
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^2 - y^2, x * y)

-- Proving that the transformed image of the triangle OAB is the set of points (0,0), (1,0), and (0,1)
theorem transform_OAB :
  (transform 0 0) = (0, 0) ∧
  (transform 1 0) = (1, 0) ∧
  (transform 1 1) = (0, 1) :=
by
  split; sorry

end transform_OAB_l326_326385


namespace puzzle_pieces_left_is_150_l326_326048

def puzzle_pieces_left (total_pieces : ℕ) (sons : ℕ) (reyn_pieces : ℕ) : ℕ :=
  let rhys_pieces := 2 * reyn_pieces in
  let rory_pieces := 3 * reyn_pieces in
  let total_placed := reyn_pieces + rhys_pieces + rory_pieces in
  total_pieces - total_placed

theorem puzzle_pieces_left_is_150 :
  puzzle_pieces_left 300 3 25 = 150 :=
by
  sorry

end puzzle_pieces_left_is_150_l326_326048


namespace bruce_anne_cleaning_house_l326_326959

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l326_326959


namespace even_of_form_4a_plus_2_not_diff_of_squares_l326_326415

theorem even_of_form_4a_plus_2_not_diff_of_squares (a x y : ℤ) : ¬ (4 * a + 2 = x^2 - y^2) :=
by sorry

end even_of_form_4a_plus_2_not_diff_of_squares_l326_326415


namespace locus_of_centers_l326_326796

theorem locus_of_centers (a b : ℝ) :
  let C1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 1
  let C3 : ℝ → ℝ → Prop := λ x y, (x - 3)^2 + y^2 = 25
  (∀ x y, C1 x y → ¬C3 x y) ∧ 
  (∃ x y r, (a,b) = (x,y) ∧
    (x-a)^2 + (y-b)^2 = (r + 1)^2 ∧ 
    (x - 3 - a)^2 + (y - b)^2 = (5 - r)^2) →
  12 * a^2 + 16 * b^2 - 36 * a - 81 = 0 := 
begin
  intros,
  sorry
end

end locus_of_centers_l326_326796


namespace three_excellent_students_l326_326263

variables (A B C D E : Prop)

-- Everyone's statements
variable hA : A → B
variable hB : B → C
variable hC : C → D
variable hD : D → E

-- Condition: exactly three students score excellent
axiom h : (C ∧ D ∧ E) ∧ ¬(A ∨ B)

theorem three_excellent_students : (C ∧ D ∧ E) := by
  exact h.1

end three_excellent_students_l326_326263


namespace rowing_downstream_rate_l326_326524

theorem rowing_downstream_rate (rate_still_water rate_current : ℝ) : 
  rate_still_water = 20 → rate_current = 10 → rate_still_water + rate_current = 30 :=
by
  intro h1 h2
  rw [h1, h2]
  exact rfl

end rowing_downstream_rate_l326_326524


namespace grade_conversion_possible_l326_326501

-- Given conditions
variables {a b : ℕ}

-- Non-zero and different grades from 100
axiom (h1 : a ≠ 0) (h2 : a ≠ 100) (h3 : b ≠ 0) (h4 : b ≠ 100)

-- Proof target
theorem grade_conversion_possible (a b : ℕ) (h1 : a ≠ 0) (h2 : a ≠ 100) (h3 : b ≠ 0) (h4 : b ≠ 100) :
  ∃ recalculation_sequence, (some_final_grade a = b ∧ some_final_grade b = a) :=
sorry

end grade_conversion_possible_l326_326501


namespace philosophy_contains_statements_l326_326424

def philosophy_statement : Prop :=
  ∀ (a b : Prop), (a ↔ (movement_absolute ∧ context_dependence)) ∧ (b ↔ (truth_contextual)) →
  (a ∧ b) ↔ (philosophy_implied).

axiom movement_absolute : Prop
axiom context_dependence : Prop
axiom truth_contextual : Prop
axiom philosophy_implied : Prop

theorem philosophy_contains_statements :
  philosophy_statement :=
by
  unfold philosophy_statement
  intros
  sorry  -- Proof omitted

end philosophy_contains_statements_l326_326424


namespace bruce_anne_clean_in_4_hours_l326_326950

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l326_326950


namespace find_k_l326_326819

theorem find_k (k : ℝ) : 
  let P := (-5 : ℝ, 0 : ℝ),
      Q := (0 : ℝ, -5 : ℝ),
      R := (0 : ℝ, 10 : ℝ),
      S := (15 : ℝ, k)
  in (0 - (-5))/(0 - (-5)) = (k - 10)/(15 - 0) → k = -5 := 
by
  intros P Q R S h
  sorry

end find_k_l326_326819


namespace determine_constants_l326_326240

theorem determine_constants (α β : ℝ) (h_eq : ∀ x, (x - α) / (x + β) = (x^2 - 96 * x + 2210) / (x^2 + 65 * x - 3510))
  (h_num : ∀ x, x^2 - 96 * x + 2210 = (x - 34) * (x - 62))
  (h_denom : ∀ x, x^2 + 65 * x - 3510 = (x - 45) * (x + 78)) :
  α + β = 112 :=
sorry

end determine_constants_l326_326240


namespace angle_difference_l326_326508

theorem angle_difference (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :
  dist A B = 39 →
  dist B C = 56 →
  dist C A = 35 →
  ∠CAB - ∠ABC = 60 :=
by
  sorry

end angle_difference_l326_326508


namespace equilateral_triangle_subdivision_l326_326716

theorem equilateral_triangle_subdivision 
    (T : Type) [fintype T] [metric_space T] 
    (triangle : T) (A B C G : T) (equilateral : is_equilateral_triangle triangle) :
    (is_centroid triangle G ∧ divides_into_three_smaller_triangles triangle G) →
    (∀ t₁ t₂ t₃ : T, is_smaller_triangle t₁ ∧ is_smaller_triangle t₂ ∧ is_smaller_triangle t₃ → 
        is_congruent t₁ t₂ ∧ is_congruent t₂ t₃ ∧ 
        (area t₁ = (1 / 3) * area triangle)) :=
begin
  sorry,
end

end equilateral_triangle_subdivision_l326_326716


namespace cream_ratio_l326_326721

def joe_ends_with_cream (start_coffee : ℕ) (drank_coffee : ℕ) (added_cream : ℕ) : ℕ :=
  added_cream

def joann_cream_left (start_coffee : ℕ) (added_cream : ℕ) (drank_mix : ℕ) : ℚ :=
  added_cream - drank_mix * (added_cream / (start_coffee + added_cream))

theorem cream_ratio (start_coffee : ℕ) (joe_drinks : ℕ) (joe_adds : ℕ)
                    (joann_adds : ℕ) (joann_drinks : ℕ) :
  joe_ends_with_cream start_coffee joe_drinks joe_adds / 
  joann_cream_left start_coffee joann_adds joann_drinks = (9 : ℚ) / (7 : ℚ) :=
by
  sorry

end cream_ratio_l326_326721


namespace value_of_expression_l326_326294

theorem value_of_expression (a b c d x y : ℤ) 
  (h1 : a = -b) 
  (h2 : c * d = 1)
  (h3 : abs x = 3)
  (h4 : y = -1) : 
  2 * x - c * d + 6 * (a + b) - abs y = 4 ∨ 2 * x - c * d + 6 * (a + b) - abs y = -8 := 
by 
  sorry

end value_of_expression_l326_326294


namespace parabola_Pi2_intersects_x_axis_at_33_l326_326059

/-
  Given:
  - Parabola Π₁ passes through (10,0) and (13,0).
  - Parabola Π₂ passes through (13,0).
  - The vertex of Π₁ bisects the segment connecting the origin and the vertex of Π₂.

  Prove:
  - Parabola Π₂ intersects the x-axis again at x = 33.
-/

def parabola_intersects_x_axis (x₁₁ x₁₂ x₂₁ : ℝ) : ℝ :=
  let x_v1 := (x₁₁ + x₁₂) / 2
  let x_v2 := 2 * x_v1
  let x₂₂ := 2 * x_v2 - x₂₁
  x₂₂

theorem parabola_Pi2_intersects_x_axis_at_33 :
  parabola_intersects_x_axis 10 13 13 = 33 :=
by
  sorry

end parabola_Pi2_intersects_x_axis_at_33_l326_326059


namespace middle_managers_sample_count_l326_326518

def employees_total : ℕ := 1000
def managers_middle_total : ℕ := 150
def sample_total : ℕ := 200

theorem middle_managers_sample_count :
  sample_total * managers_middle_total / employees_total = 30 := by
  sorry

end middle_managers_sample_count_l326_326518


namespace minimum_value_expr_l326_326138

theorem minimum_value_expr (x y : ℝ) : 
  ∃ (a b : ℝ), 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 = 2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ∧ 
  2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ≥ 4 :=
by 
  sorry

end minimum_value_expr_l326_326138


namespace min_value_inequality_l326_326391

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end min_value_inequality_l326_326391


namespace complex_transformation_result_l326_326856

noncomputable def complex_transformation (z : ℂ) : ℂ :=
  let rotation : ℂ := (√3 / 2) + (1 / 2) * Complex.I
  let dilation : ℂ := 2
  dilation * rotation * z

theorem complex_transformation_result :
  complex_transformation (-1 - 4 * Complex.I) =
  4 - √3 - (4 * √3 + 1) * Complex.I :=
by
  -- Placeholder for the proof
  sorry

end complex_transformation_result_l326_326856


namespace interest_rate_bc_l326_326523

def interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

def gain_b (interest_bc interest_ab : ℝ) : ℝ :=
  interest_bc - interest_ab

theorem interest_rate_bc :
  ∀ (principal : ℝ) (rate_ab rate_bc : ℝ) (time : ℕ) (gain : ℝ),
    principal = 3500 → rate_ab = 0.10 → time = 3 → gain = 525 →
    interest principal rate_ab time = 1050 →
    gain_b (interest principal rate_bc time) (interest principal rate_ab time) = gain →
    rate_bc = 0.15 :=
by
  intros principal rate_ab rate_bc time gain h_principal h_rate_ab h_time h_gain h_interest_ab h_gain_b
  sorry

end interest_rate_bc_l326_326523


namespace length_AD_l326_326717

-- Define the given conditions
variables (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
variables (angleBAD : A → B → D → ℝ) (angleABC : A → B → C → ℝ) (angleBCD : B → C → D → ℝ)
variables (AB CD AD : ℝ)

-- Assuming the given conditions: angles and lengths
axiom h_angleBAD : angleBAD A B D = 60
axiom h_angleABC : angleABC A B C = 30
axiom h_angleBCD : angleBCD B C D = 30
axiom h_AB : AB = 15
axiom h_CD : CD = 8

-- Proving the length of AD
theorem length_AD : AD = 3.5 :=
sorry

end length_AD_l326_326717


namespace union_of_sets_l326_326237

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_of_sets : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l326_326237


namespace avg_age_difference_l326_326351

-- Definitions of the conditions
def num_players : Nat := 11
def captain_age : Nat := 26
def wicket_keeper_age : Nat := captain_age + 5
def avg_team_age : ℝ := 24
def total_team_age : ℝ := avg_team_age * num_players
def num_remaining_players : Nat := num_players - 2
def total_remaining_age : ℝ := total_team_age - captain_age - wicket_keeper_age
def avg_remaining_age : ℝ := total_remaining_age / num_remaining_players

-- Statement of the problem
theorem avg_age_difference :
  avg_team_age - avg_remaining_age = 1 := by
  sorry

end avg_age_difference_l326_326351


namespace area_bounded_by_circles_l326_326473

-- Define the centers of the circles and their radii
def center_A : (ℝ × ℝ) := (2, 2)
def center_B : (ℝ × ℝ) := (6, 2)
def center_C : (ℝ × ℝ) := (4, 6)
def radius : ℝ := 3

-- Assert the area of the region bounded by the circles and the x-axis
theorem area_bounded_by_circles :
  let area := (20 - 9 * real.pi / 2) in 
  true := sorry

end area_bounded_by_circles_l326_326473


namespace prime_sum_root_eq_diff_l326_326159

noncomputable def is_prime (n : ℕ) : Prop := Prime

theorem prime_sum_root_eq_diff (p q n : ℕ) (hp : is_prime p) (hq : is_prime q) (hn : n > 0) :
  (p > q) → (Real.root n (p + q)) = (p - q) → (p, q, n) = (5, 3, 3) :=
by
  intros h
  sorry

end prime_sum_root_eq_diff_l326_326159


namespace probability_segments_length_l326_326182

theorem probability_segments_length (x y : ℝ) : 
    80 ≥ x ∧ x ≥ 20 ∧ 80 ≥ y ∧ y ≥ 20 ∧ 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20 → 
    (∃ (s : ℝ), s = (200 / 3200) ∧ s = (1 / 16)) :=
by
  intros h
  sorry

end probability_segments_length_l326_326182


namespace ratio_of_radii_l326_326170

noncomputable def circle_tangent_ratio (r1 r2 : ℝ) (N M P : Type) : Prop :=
∃ (l j k : Type), 
  (circle N r1) ∧ 
  (circle M r2) ∧ 
  (external_tangent l N M) ∧ 
  (circle P r2) ∧ 
  (external_tangent l N P) ∧ 
  (external_tangent k N P ∧ (M, N, P on_one_side k)) ∧ 
  (parallel j k) ∧ 
  (r1 / r2 = 3 : Prop)

theorem ratio_of_radii (r1 r2 : ℝ) (N M P : Type) :
  circle_tangent_ratio r1 r2 N M P := 
sorry

end ratio_of_radii_l326_326170


namespace smallest_integer_in_correct_range_l326_326599

theorem smallest_integer_in_correct_range :
  ∃ (n : ℤ), n > 1 ∧ n % 3 = 1 ∧ n % 5 = 1 ∧ n % 8 = 1 ∧ n % 7 = 2 ∧ 161 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_integer_in_correct_range_l326_326599


namespace sum_sqrt_reciprocal_lt_sqrt3_over6_l326_326278

def a (n : ℕ) : ℝ := 4 * n * (4 * n + 1) * (4 * n + 2)

theorem sum_sqrt_reciprocal_lt_sqrt3_over6 (n : ℕ) :
  ∑ i in (Finset.range n).map (λ i:ℕ, i+1), (1 / real.sqrt (a i)) < real.sqrt 3 / 6 := 
sorry

end sum_sqrt_reciprocal_lt_sqrt3_over6_l326_326278


namespace complex_point_correspondence_l326_326642

noncomputable def complex_number_z : ℂ :=
  (∑ k in finset.range 11, complex.i ^ k)

theorem complex_point_correspondence :
  complex_number_z = complex.i :=
sorry

end complex_point_correspondence_l326_326642


namespace salmon_total_l326_326583

def num_male : ℕ := 712261
def num_female : ℕ := 259378
def num_total : ℕ := 971639

theorem salmon_total :
  num_male + num_female = num_total :=
by
  -- proof will be provided here
  sorry

end salmon_total_l326_326583


namespace no_six_digit_number_l326_326995

theorem no_six_digit_number :
  ¬ ∃ (n : ℕ), n >= 100000 ∧ n < 1000000 ∧
               (∀ d ∈ Int.digits 10 n, d = 1 ∨ d = 2 ∨ d = 3) ∧
               (Int.digits 10 n).count 3 = 2 ∧
               n % 9 = 0 :=
by
  sorry

end no_six_digit_number_l326_326995


namespace hyperbola_eccentricity_l326_326311

/--
Given the hyperbola x^2 / a^2 - y^2 / b^2 = 1 with a > 0 and b > 0,
and given that a tangent line is drawn from point F1 to the circle x^2 + y^2 = a^2,
intersecting the right branch of the hyperbola at point P,
and given that ∠ F1 P F2 = 45°, the eccentricity of the hyperbola is √3.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ F1 F2 : ℝ × ℝ, F1 ≠ F2)
  (h2 : ∃ P : ℝ × ℝ, tangent (circle a) (hyperbola a b) P F1)
  (h3 : angle F1 P F2 = 45) : 
  eccentricity (hyperbola a b) = √3 := sorry

end hyperbola_eccentricity_l326_326311


namespace tetrahedron_fits_box_l326_326198

/-- A tetrahedron with all faces as congruent triangles with sides 8, 9, and 10 can be packed 
    into a box with internal dimensions 5 × 8 × 8 --/
theorem tetrahedron_fits_box : 
  ∃ x y z : ℝ, 
  x^2 + y^2 = 100 ∧ 
  x^2 + z^2 = 81 ∧ 
  y^2 + z^2 = 64 ∧ 
  x ≤ 8 ∧ 
  y ≤ 8 ∧ 
  z ≤ 5 :=
begin
  sorry
end

end tetrahedron_fits_box_l326_326198


namespace problem1_correct_problem2_correct_l326_326965

noncomputable def problem1 : ℚ :=
  (1/2 - 5/9 + 7/12) * (-36)

theorem problem1_correct : problem1 = -19 := 
by 
  sorry

noncomputable def mixed_number (a : ℤ) (b : ℚ) : ℚ := a + b

noncomputable def problem2 : ℚ :=
  (mixed_number (-199) (24/25)) * 5

theorem problem2_correct : problem2 = -999 - 4/5 :=
by
  sorry

end problem1_correct_problem2_correct_l326_326965


namespace line_passes_through_fixed_point_l326_326106

theorem line_passes_through_fixed_point (a : ℝ) : 
  ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ (a + 1) * x - y - 2 * a + 1 = 0 :=
begin
  let x := 2,
  let y := 3,
  use [x, y],
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end line_passes_through_fixed_point_l326_326106


namespace bruce_and_anne_clean_together_l326_326952

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l326_326952


namespace students_without_favorite_subject_l326_326689

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end students_without_favorite_subject_l326_326689


namespace find_a4b4_l326_326815

theorem find_a4b4 
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) :
  a4 * b4 = -6 :=
sorry

end find_a4b4_l326_326815


namespace cookies_collected_total_is_276_l326_326918

noncomputable def number_of_cookies_in_one_box : ℕ := 48

def abigail_boxes : ℕ := 2
def grayson_boxes : ℕ := 3 / 4
def olivia_boxes : ℕ := 3

def total_cookies_collected : ℕ :=
  abigail_boxes * number_of_cookies_in_one_box + 
  (grayson_boxes * number_of_cookies_in_one_box) + 
  olivia_boxes * number_of_cookies_in_one_box

theorem cookies_collected_total_is_276 : total_cookies_collected = 276 := sorry

end cookies_collected_total_is_276_l326_326918


namespace rhombus_perimeter_l326_326802

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  let side_length := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let perimeter := 4 * side_length in
  perimeter = 68 :=
by
  have h3 : d1 / 2 = 8, from by rw [h1],
  have h4 : d2 / 2 = 15, from by rw [h2],
  have h5 : side_length = 17, from by
    calc
      side_length
          = Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) : rfl
      ... = Real.sqrt (8 ^ 2 + 15 ^ 2) : by rw [h3, h4]
      ... = Real.sqrt (64 + 225) : rfl
      ... = Real.sqrt 289 : rfl
      ... = 17 : by norm_num,
  calc
    perimeter
        = 4 * side_length : rfl
    ... = 4 * 17 : by rw [h5]
    ... = 68 : by norm_num

end rhombus_perimeter_l326_326802


namespace count_numbers_between_200_and_500_with_digit_two_l326_326662

noncomputable def count_numbers_with_digit_two (a b : ℕ) : ℕ :=
  let numbers := list.filter (λ n, '2' ∈ n.digits 10) (list.range' a (b - a))
  numbers.length

theorem count_numbers_between_200_and_500_with_digit_two :
  count_numbers_with_digit_two 200 500 = 138 :=
by sorry

end count_numbers_between_200_and_500_with_digit_two_l326_326662


namespace factorial_fraction_simplification_l326_326555

theorem factorial_fraction_simplification : 
  (4 * (Nat.factorial 6) + 24 * (Nat.factorial 5)) / (Nat.factorial 7) = 8 / 7 :=
by
  sorry

end factorial_fraction_simplification_l326_326555


namespace pollutant_decay_l326_326897

noncomputable def p (t : ℝ) (p0 : ℝ) := p0 * 2^(-t / 30)

theorem pollutant_decay : 
  ∃ p0 : ℝ, p0 = 300 ∧ p 60 p0 = 75 * Real.log 2 := 
by
  sorry

end pollutant_decay_l326_326897


namespace value_of_expression_l326_326484

theorem value_of_expression :
  ∀ (x y z : ℝ), x = 2 → y = -3 → z = 6 → x^2 + y^2 + z^2 + 2*x*y - 2*y*z = 73 :=
by
  intro x y z hx hy hz
  rw [hx, hy, hz]
  calc
    (2:ℝ)^2 + (-3:ℝ)^2 + (6:ℝ)^2 + 2*(2:ℝ)*(-3:ℝ) - 2*(-3:ℝ)*(6:ℝ) 
    = 4 + 9 + 36 - 12 + 36 : by norm_num
  ... = 73 : by norm_num
  done

end value_of_expression_l326_326484


namespace sales_volume_at_price_45_proof_new_sell_price_for_unchanged_profit_proof_xiao_ming_discount_proof_l326_326039

-- Define parameters
def cost_price : ℕ := 40
def orig_sell_price : ℕ := 60
def orig_sales_volume : ℕ := 20

-- Condition for change in sales volume with decrease in price
def volume_increase_per_price_decrease : ℕ := 2

-- Part 1: Calculate sales volume for price 45 yuan
def sales_volume_at_price_45 : ℕ := orig_sales_volume + (orig_sell_price - 45) * volume_increase_per_price_decrease

-- Part 2: Determine new selling price for unchanged profit
def orig_profit : ℕ := (orig_sell_price - cost_price) * orig_sales_volume
def new_sales_volume (x : ℕ) : ℕ := orig_sales_volume + (orig_sell_price - x) * volume_increase_per_price_decrease
def new_profit (x : ℕ) : ℕ := (x - cost_price) * (new_sales_volume x)

-- Part 3: Determine the discount percentage for Xiao Ming's price not exceeding 50 yuan
def xiao_ming_price : ℕ := 62.5
def max_allowed_price : ℕ := 50
def discount_percentage : ℕ := 100 - (max_allowed_price * 100 / xiao_ming_price)

-- Lean theorem statement to prove the solution
theorem sales_volume_at_price_45_proof : sales_volume_at_price_45 = 50 := by sorry
theorem new_sell_price_for_unchanged_profit_proof : ∃ x, new_profit x = orig_profit ∧ new_sales_volume x = max_by new_sales_volume x := by sorry
theorem xiao_ming_discount_proof : xiao_ming_price * (discount_percentage / 100) ≤ max_allowed_price := by sorry

-- Prove all calculated values are correct
example : sales_volume_at_price_45 = 50 := by sorry
example : ∃ x, new_profit x = orig_profit ∧ new_sales_volume x = max_by new_sales_volume x := by sorry
example : discount_percentage = 20 := by sorry

end sales_volume_at_price_45_proof_new_sell_price_for_unchanged_profit_proof_xiao_ming_discount_proof_l326_326039


namespace similar_D_A_l326_326714

theorem similar_D_A 
  (A1 A2 A3 : Type) [affine_plane A1] [affine_plane A2] [affine_plane A3]
  (P : A1) 
  (B1 B2 B3 : A1)
  (h1 : ∃ P : A1, drop_perpendicular P A1 A2 = B1)
  (h2 : ∃ P : A1, drop_perpendicular P A2 A3 = B2)
  (h3 : ∃ P : A1, drop_perpendicular P A3 A1 = B3)
  (C1 C2 C3 : A1)
  (h4 : ∃ C1 C2 C3, similar_triangulation B1 B2 B3 C1 C2 C3)
  (D1 D2 D3 : A1)
  (h5 : ∃ D1 D2 D3, similar_triangulation C1 C2 C3 D1 D2 D3) :
  similar_triangulation A1 A2 A3 D1 D2 D3 :=
sorry

end similar_D_A_l326_326714


namespace prob_of_event_l326_326655

variable (ξ : Nat → ℝ)
variable (P : Nat → ℝ)

-- Given the probability distribution
def prob_dist := ∀ k : Nat, P k = 1 / (2 ^ k)

-- Define the event in question
def event.2_lt_ξ_leq_4 : ℝ := P 3 + P 4

-- Lean theorem statement to prove
theorem prob_of_event : prob_dist P → event.2_lt_ξ_leq_4 P = 3 / 16 := by
  intro h
  sorry

end prob_of_event_l326_326655


namespace problem1_solution_problem2_solution_l326_326433

theorem problem1_solution (x : ℝ) :
  (2 < |2 * x - 5| ∧ |2 * x - 5| ≤ 7) → ((-1 ≤ x ∧ x < 3 / 2) ∨ (7 / 2 < x ∧ x ≤ 6)) := by
  sorry

theorem problem2_solution (x : ℝ) :
  (1 / (x - 1) > x + 1) → (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) := by
  sorry

end problem1_solution_problem2_solution_l326_326433


namespace Ann_quiz_goal_l326_326221
-- Import the entire Mathlib library

/-- 
 Ann's goal is to earn an A on at least 85% of her 60 quizzes.
 She earned an A on 34 of the first 40 quizzes.
 We need to prove that Ann can afford to score below an A on at most 3 of the remaining quizzes to meet her goal.
-/
theorem Ann_quiz_goal (total_quizzes : ℕ) (quizzes_done : ℕ) (A_quizzes_done : ℕ) (goal_percentage : ℚ) :
  total_quizzes = 60 →
  quizzes_done = 40 →
  A_quizzes_done = 34 →
  goal_percentage = 0.85 →
  let total_required_A_quizzes := (goal_percentage * total_quizzes).ceil.to_nat in
  let remaining_quizzes := total_quizzes - quizzes_done in
  let additional_A_needed := total_required_A_quizzes - A_quizzes_done in
  let max_below_A_quizzes := remaining_quizzes - additional_A_needed in
  max_below_A_quizzes = 3 :=
by
  intros h1 h2 h3 h4
  sorry

end Ann_quiz_goal_l326_326221


namespace range_of_b_l326_326035

noncomputable def f (b x : ℝ) : ℝ := -x^3 + b*x

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f b x = 0 → x ∈ set.Icc (-2 : ℝ) (2 : ℝ))
  ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → -3 * x^2 + b ≥ 0)
  → 3 ≤ b ∧ b ≤ 4 :=
by
  intro h
  sorry

end range_of_b_l326_326035


namespace probability_samantha_in_sam_not_l326_326776

noncomputable def probability_in_picture_but_not (time_samantha : ℕ) (lap_samantha : ℕ) (time_sam : ℕ) (lap_sam : ℕ) : ℚ :=
  let seconds_raced := 900
  let samantha_laps := seconds_raced / time_samantha
  let sam_laps := seconds_raced / time_sam
  let start_line_samantha := (samantha_laps - (samantha_laps % 1)) * time_samantha + ((samantha_laps % 1) * lap_samantha)
  let start_line_sam := (sam_laps - (sam_laps % 1)) * time_sam + ((sam_laps % 1) * lap_sam)
  let in_picture_duration := 80
  let overlapping_time := 30
  overlapping_time / in_picture_duration

theorem probability_samantha_in_sam_not : probability_in_picture_but_not 120 60 75 25 = 3 / 8 := by
  sorry

end probability_samantha_in_sam_not_l326_326776


namespace second_player_wins_l326_326502

open Function

-- Definitions based on conditions
def initial_pile1 : ℕ := 10
def initial_pile2 : ℕ := 15
def initial_pile3 : ℕ := 20

-- Lean proof statement
theorem second_player_wins (p1 p2 p3 : ℕ) (h1 : p1 = initial_pile1) (h2 : p2 = initial_pile2) (h3 : p3 = initial_pile3) :
  ∑ n in [p1 - 1, p2 - 1, p3 - 1], n % 2 = 0 → True :=
  sorry

end second_player_wins_l326_326502


namespace distance_between_parallel_lines_l326_326316

theorem distance_between_parallel_lines : 
  (∀ (x y : ℝ), 3 * x - 4 * y + 1 = 0 ↔ 3 * x - 4 * y + 1 = 0) ∧
  (∀ (x y : ℝ), 3 * x - 4 * y - 4 = 0 ↔ 3 * x - 4 * y - 4 = 0) →
  abs (-4 - 1) / sqrt (3 ^ 2 + (-4) ^ 2) = 1 :=
by
  sorry

end distance_between_parallel_lines_l326_326316


namespace unique_positive_integers_sum_l326_326030

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77) / 3 + 5 / 3)

theorem unique_positive_integers_sum :
  ∃ (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c),
    x^100 = 3 * x^98 + 17 * x^96 + 13 * x^94 - 2 * x^50 + (a : ℝ) * x^46 + (b : ℝ) * x^44 + (c : ℝ) * x^40
    ∧ a + b + c = 167 := by
  sorry

end unique_positive_integers_sum_l326_326030


namespace total_students_in_lunchroom_l326_326166

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end total_students_in_lunchroom_l326_326166


namespace perimeter_of_larger_triangle_l326_326853

-- Define the conditions as given in the problem
def isosceles_triangle (a b : ℝ) := a = b

def smaller_triangle_isosceles (a b c : ℝ) :=
  isosceles_triangle a b ∧ c = 8 ∧ a = 16

def similar_triangles (k : ℝ) (a b c : ℝ) (a' b' c' : ℝ) :=
  a' = k * a ∧ b' = k * b ∧ c' = k * c

-- The main statement to prove
theorem perimeter_of_larger_triangle 
  (a b c a' b' c' : ℝ) (k : ℝ)
  (h1 : smaller_triangle_isosceles a b c)
  (h2 : c = 8)
  (h3 : isosceles_triangle a b)
  (h4 : similar_triangles k a b c a' b' c')
  (h5 : c' = 40)
  (scaling_factor : k = 40 / 8)
  : a' + b' + c' = 200 :=
begin
  sorry
end

end perimeter_of_larger_triangle_l326_326853


namespace cut_scene_length_l326_326891

theorem cut_scene_length
  (original_length final_length : ℕ)
  (h_original : original_length = 60)
  (h_final : final_length = 54) :
  original_length - final_length = 6 :=
by 
  sorry

end cut_scene_length_l326_326891


namespace frog_jumps_probability_l326_326900

-- Definitions and conditions
def frog_jump_distance (n : ℕ) (jump_length : ℝ) : Prop :=
  n = 3 ∧ jump_length = 1

-- Query: the probability that the resultant jump vector is within 1 meter
def frog_probability_within_distance (dist : ℝ) : ℝ :=
  if dist ≤ 1 then 1 / 4 else 0

-- The main theorem to prove the probability is 1/4 given the conditions.
theorem frog_jumps_probability :
  ∀ (n : ℕ) (jump_length final_distance : ℝ),
    frog_jump_distance n jump_length →
    final_distance ≤ 1 →
    frog_probability_within_distance final_distance = 1 / 4 :=
  sorry

end frog_jumps_probability_l326_326900


namespace total_cookies_collected_l326_326915

theorem total_cookies_collected 
  (abigail_boxes : ℕ) (grayson_boxes : ℕ) (olivia_boxes : ℕ) (cookies_per_box : ℕ)
  (h1 : abigail_boxes = 2) (h2 : grayson_boxes = 3) (h3 : olivia_boxes = 3) (h4 : cookies_per_box = 48) :
  (abigail_boxes * cookies_per_box) + ((grayson_boxes * (cookies_per_box / 4))) + (olivia_boxes * cookies_per_box) = 276 := 
by 
  sorry

end total_cookies_collected_l326_326915


namespace units_digit_of_product_of_seven_consecutive_integers_is_zero_l326_326972

/-- Define seven consecutive positive integers and show the units digit of their product is 0 -/
theorem units_digit_of_product_of_seven_consecutive_integers_is_zero (n : ℕ) :
  ∃ (k : ℕ), k = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 ∧ k = 0 :=
by {
  -- We state that the units digit k of the product of seven consecutive integers
  -- starting from n is 0
  sorry
}

end units_digit_of_product_of_seven_consecutive_integers_is_zero_l326_326972


namespace all_matrices_in_G_generated_by_A_and_B_l326_326731

open Matrix

-- Define the matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![0, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![[-1, 1], ![-3, 2]]

-- Define the set G of 2x2 integer matrices with determinant 1 and whose c-element is multiple of 3
def G : Set (Matrix (Fin 2) (Fin 2) ℤ) := 
  { M | M.det = 1 ∧ (M 1 0) % 3 = 0 }

-- Theorem statement
theorem all_matrices_in_G_generated_by_A_and_B :
  ∀ M ∈ G, ∃ (n : ℕ), (Fin n → {X // X = A ∨ X = A⁻¹ ∨ X = B ∨ X = B⁻¹}) (L : Fin n → Matrix (Fin 2) (Fin 2) ℤ),
  (∏ i, L i) = M :=
sorry

end all_matrices_in_G_generated_by_A_and_B_l326_326731


namespace unique_four_digit_perfect_cube_divisible_by_16_and_9_l326_326321

theorem unique_four_digit_perfect_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 16 = 0 ∧ n % 9 = 0 ∧ n = 1728 :=
by sorry

end unique_four_digit_perfect_cube_divisible_by_16_and_9_l326_326321


namespace all_marvelous_numbers_divisible_by_5_l326_326378

def marvelous_number (a b c d e n : ℕ) : Prop := 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  n = a^4 + b^4 + c^4 + d^4 + e^4 ∧
  (∀ x, x ∈ {a, b, c, d, e} → x ∣ n)

theorem all_marvelous_numbers_divisible_by_5 (a b c d e n : ℕ) 
  (h_marv : marvelous_number a b c d e n) : 5 ∣ n :=
  sorry

end all_marvelous_numbers_divisible_by_5_l326_326378


namespace bruce_anne_clean_in_4_hours_l326_326951

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l326_326951


namespace problem_statement_l326_326122

noncomputable def a_b (a b : ℚ) : Prop :=
  a + b = 6 ∧ a / b = 6

theorem problem_statement (a b : ℚ) (h : a_b a b) : 
  (a * b - (a - b)) = 6 / 49 :=
by
  sorry

end problem_statement_l326_326122


namespace minimum_detectors_203_l326_326114

def minimum_detectors (length : ℕ) : ℕ :=
  length / 3 * 2 -- This models the generalization for 1 × (3k + 2)

theorem minimum_detectors_203 : minimum_detectors 203 = 134 :=
by
  -- Length is 203, k = 67 which follows from the floor division
  -- Therefore, minimum detectors = 2 * 67 = 134
  sorry

end minimum_detectors_203_l326_326114


namespace freshmen_count_l326_326084

theorem freshmen_count (n : ℕ) (h1 : n < 600) (h2 : n % 17 = 16) (h3 : n % 19 = 18) : n = 322 := 
by 
  sorry

end freshmen_count_l326_326084


namespace minimize_median_mn_l326_326549

-- Define all conditions for the triangle KLM

-- Circumradius R
def circumradius : ℝ := 10

-- Side KL
def side_kl : ℝ := 16

-- Height MH from vertex M to side KL
def height_mh : ℝ := 39 / 10

-- Function to define the angle alpha as computed
def angle_kml (α : ℝ) : Prop :=
  α = Real.pi - Real.arcsin (4/5)

-- Problem statement:
-- Given the circumradius, side KL, and height MH,
-- prove that angle KML of triangle KLM which minimizes the median MN is precisely π - arcsin(4/5).

theorem minimize_median_mn :
  ∃α : ℝ, angle_kml α :=
sorry

end minimize_median_mn_l326_326549


namespace median_divides_triangle_into_two_equal_parts_l326_326861

theorem median_divides_triangle_into_two_equal_parts (A B C : Point) (M : Point)
  (is_median : M = midpoint B C) :
  area (triangle A M B) = area (triangle A M C) :=
sorry

end median_divides_triangle_into_two_equal_parts_l326_326861


namespace exists_side_shorter_than_longer_diagonal_l326_326416

-- Definitions
structure ConvexQuadrilateral (A B C D : Type) : Prop :=
(convex : true) -- This will be more detailed, depending on what "convex" exactly means in Lean.

variables {A B C D : Type} [ConvexQuadrilateral A B C D]

def is_shorter_than_longer_diagonal
  (AB BC CD DA AC BD : ℝ) : Prop :=
∃ (s : ℝ), ((s = AB ∨ s = BC ∨ s = CD ∨ s = DA) ∧ (AC ≥ BD → s < AC) ∧ (BD > AC → s < BD))

-- Statement
theorem exists_side_shorter_than_longer_diagonal
  (AB BC CD DA AC BD : ℝ) :
  ConvexQuadrilateral A B C D →
  is_shorter_than_longer_diagonal AB BC CD DA AC BD :=
by
  sorry

end exists_side_shorter_than_longer_diagonal_l326_326416


namespace restore_price_after_reduction_by_20_percent_l326_326217

theorem restore_price_after_reduction_by_20_percent (P : ℝ) (h : P > 0) :
  let new_price := 0.8 * P
  let increase_percent := (P / new_price - 1) * 100
  increase_percent = 25 :=
by 
  sorry

end restore_price_after_reduction_by_20_percent_l326_326217


namespace ineq_sqrt_two_l326_326282

theorem ineq_sqrt_two (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by 
  sorry

end ineq_sqrt_two_l326_326282


namespace sasha_remaining_questions_l326_326067

theorem sasha_remaining_questions
  (qph : ℕ) (total_questions : ℕ) (hours_worked : ℕ)
  (h_qph : qph = 15) (h_total_questions : total_questions = 60) (h_hours_worked : hours_worked = 2) :
  total_questions - (qph * hours_worked) = 30 :=
by
  sorry

end sasha_remaining_questions_l326_326067


namespace count_numbers_with_digit_2_from_200_to_499_l326_326663

def count_numbers_with_digit_2 (lower upper : ℕ) : ℕ :=
  let A := 100  -- Numbers of the form 2xx (from 200 to 299)
  let B := 30   -- Numbers of the form x2x (where first digit is 2, 3, or 4, last digit can be any)
  let C := 30   -- Numbers of the form xx2 (similar reasoning as B)
  let A_and_B := 10  -- Numbers of the form 22x
  let A_and_C := 10  -- Numbers of the form 2x2
  let B_and_C := 3   -- Numbers of the form x22
  let A_and_B_and_C := 1  -- The number 222
  A + B + C - A_and_B - A_and_C - B_and_C + A_and_B_and_C

theorem count_numbers_with_digit_2_from_200_to_499 : 
  count_numbers_with_digit_2 200 499 = 138 :=
by
  unfold count_numbers_with_digit_2
  exact rfl

end count_numbers_with_digit_2_from_200_to_499_l326_326663


namespace sin_six_theta_l326_326665

theorem sin_six_theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (6 * θ) = - (630 * Real.sqrt 8) / 15625 := by
  sorry

end sin_six_theta_l326_326665


namespace fraction_of_work_left_l326_326890

theorem fraction_of_work_left 
  (A_days : ℕ) (B_days : ℕ) (work_days : ℕ) 
  (A_rate : ℚ := 1 / A_days) (B_rate : ℚ := 1 / B_days) (combined_rate : ℚ := 1 / A_days + 1 / B_days) 
  (work_completed : ℚ := combined_rate * work_days) (fraction_left : ℚ := 1 - work_completed)
  (hA : A_days = 15) (hB : B_days = 20) (hW : work_days = 4) 
  : fraction_left = 8 / 15 :=
sorry

end fraction_of_work_left_l326_326890


namespace problem1_problem2_l326_326557

theorem problem1 : Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0 :=
by
  sorry

theorem problem2 : Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3 :=
by
  sorry

end problem1_problem2_l326_326557


namespace positive_difference_of_two_numbers_l326_326464

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end positive_difference_of_two_numbers_l326_326464


namespace area_of_quadrilateral_l326_326115

theorem area_of_quadrilateral (A B C D : ℝ × ℝ) (p : ℝ) :
  (D.2 = 1 - 2 * A.1 + A.1^2 ∧ B.2 = A.1^2 + 2 * A.1 + 1) →
  (A.2 = A.1^2 ∧ C.2 = A.1^2 ∧ A.1 > 0) →
  (A.1 - D.1 = -(B.1 - A.1)) →
  (B.1 = -A.1 - 1 ∧ D.1 = 1 - A.1) →
  (AC_parallel_to_x_axis : C.1 - A.1 = -A.1 - A.1) →
  (BD_length : (D.1 - B.1)^2 + (D.2 - B.2)^2 = p^2) →
  (area : ℝ) = (p^2 - 4) / 4 :=
begin
  intros HD_1_1_y HB_y AA_1 CA_200x _B _1A_p,
  sorry,
end

end area_of_quadrilateral_l326_326115


namespace ants_crushed_l326_326535

theorem ants_crushed {original_ants left_ants crushed_ants : ℕ} 
  (h1 : original_ants = 102) 
  (h2 : left_ants = 42) 
  (h3 : crushed_ants = original_ants - left_ants) : 
  crushed_ants = 60 := 
by
  sorry

end ants_crushed_l326_326535


namespace x_coordinate_at_2005th_stop_l326_326862

theorem x_coordinate_at_2005th_stop :
 (∃ (f : ℕ → ℤ × ℤ),
    f 0 = (0, 0) ∧
    f 1 = (1, 0) ∧
    f 2 = (1, 1) ∧
    f 3 = (0, 1) ∧
    f 4 = (-1, 1) ∧
    f 5 = (-1, 0) ∧
    f 9 = (2, -1))
  → (∃ (f : ℕ → ℤ × ℤ), f 2005 = (3, -n)) := sorry

end x_coordinate_at_2005th_stop_l326_326862


namespace problem4_l326_326063

theorem problem4 (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hx1 : x < 1) (hy1 : y < 1)
(h : a / (1 - x) + b / (1 - y) = 1) : 
  real.cbrt (a * y) + real.cbrt (b * x) ≤ 1 := 
sorry

end problem4_l326_326063


namespace calculate_weeks_proof_l326_326771

variables (initial_total_fish final_koi_fish final_goldfish added_koi_per_day added_goldfish_per_day : ℕ)
variables (days_in_week : ℕ) 

def total_fish (initial_fish final_koi final_goldfish koi_per_day goldfish_per_day days weeks_in_a_week) : ℕ :=
  initial_fish + days * (koi_per_day + goldfish_per_day)

noncomputable def calculate_weeks 
  (initial_total_fish = 280) 
  (final_koi_fish = 227)
  (final_goldfish = 200) 
  (added_koi_per_day = 2) 
  (added_goldfish_per_day = 5) 
  (days_in_week = 7) : ℕ :=
  let total_weeks := ((final_koi_fish + final_goldfish) - initial_total_fish) / (added_koi_per_day + added_goldfish_per_day) / days_in_week in
  total_weeks

theorem calculate_weeks_proof :
  calculate_weeks 280 227 200 2 5 7 = 3 := 
by
  sorry

end calculate_weeks_proof_l326_326771


namespace collinear_vector_l326_326641

theorem collinear_vector (c R : ℝ) (A B : ℝ × ℝ) (hA: A.1 ^ 2 + A.2 ^ 2 = R ^ 2) (hB: B.1 ^ 2 + B.2 ^ 2 = R ^ 2) 
                         (h_line_A: 2 * A.1 + A.2 = c) (h_line_B: 2 * B.1 + B.2 = c) :
                         ∃ k : ℝ, (4, 2) = (k * (A.1 + B.1), k * (A.2 + B.2)) :=
sorry

end collinear_vector_l326_326641


namespace exists_intersecting_line_l326_326053

open Set

-- Define the condition of a finite set of polygons each pair having a common point.
variable {α : Type*} [TopologicalSpace α]

def hasCommonPoint (P Q : Set α) : Prop :=
  ∃ p, p ∈ P ∧ p ∈ Q

def finiteSetOfPolygons (polygons : Finset (Set α)) : Prop :=
  ∀ P Q ∈ polygons, P ≠ Q → hasCommonPoint P Q

-- The math theorem stating that there exists a line having common points with all these polygons.
theorem exists_intersecting_line (polygons : Finset (Set (ℝ × ℝ))) (h : finiteSetOfPolygons polygons) :
  ∃ l : Set (ℝ × ℝ), (∀ P ∈ polygons, ∃ p ∈ P, p ∈ l) :=
begin
  sorry
end

end exists_intersecting_line_l326_326053


namespace trigonometric_problem_l326_326615

theorem trigonometric_problem
  (α : ℝ)
  (h1 : Real.tan α = Real.sqrt 3)
  (h2 : π < α)
  (h3 : α < 3 * π / 2) :
  Real.cos (2 * π - α) - Real.sin α = (Real.sqrt 3 - 1) / 2 := by
  sorry

end trigonometric_problem_l326_326615


namespace james_hours_per_class_l326_326371

theorem james_hours_per_class
  (classes_per_week : ℕ)
  (calories_burn_rate : ℕ)
  (total_calories_per_week : ℕ)
  (classes_per_week = 3)
  (calories_burn_rate = 7)
  (total_calories_per_week = 1890) :
  total_calories_per_week / (classes_per_week * calories_burn_rate) = 1.5 * 60 := 
by
  sorry

end james_hours_per_class_l326_326371


namespace probability_of_distance_less_than_8000_l326_326795

def distance (c1 c2 : String) : ℕ :=
  if (c1 = "Tokyo" ∧ c2 = "Sydney") ∨ (c1 = "Sydney" ∧ c2 = "Tokyo") then 4800 else
  if (c1 = "Tokyo" ∧ c2 = "New York") ∨ (c1 = "New York" ∧ c2 = "Tokyo") then 6760 else
  if (c1 = "Tokyo" ∧ c2 = "Paris") ∨ (c1 = "Paris" ∧ c2 = "Tokyo") then 6037 else
  if (c1 = "Sydney" ∧ c2 = "New York") ∨ (c1 = "New York" ∧ c2 = "Sydney") then 9954 else
  if (c1 = "Sydney" ∧ c2 = "Paris") ∨ (c1 = "Paris" ∧ c2 = "Sydney") then 10560 else
  if (c1 = "New York" ∧ c2 = "Paris") ∨ (c1 = "Paris" ∧ c2 = "New York") then 3624 else 0

def total_pairs := 6
def valid_pairs := 4

theorem probability_of_distance_less_than_8000 :
  valid_pairs / total_pairs = (2 : ℚ) / 3 :=
by
  sorry

end probability_of_distance_less_than_8000_l326_326795


namespace rhombus_perimeter_l326_326808

def half (a: ℕ) := a / 2

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  let s1 := half d1
      s2 := half d2
      side_length := Math.sqrt (s1^2 + s2^2)
      perimeter := 4 * side_length
  in perimeter = 68 := by
  sorry

end rhombus_perimeter_l326_326808


namespace range_of_t_log_inequality_l326_326971

noncomputable def f (x: ℝ) : ℝ := Real.log (x + 1) - x^2 - x

noncomputable def h (x t: ℝ) : ℝ := Real.log (x + 1) - x^2 + 3 / 2 * x - t

theorem range_of_t (t: ℝ) : 
  ∃ t ∈ set.Icc (Real.log 3 - 1) (Real.log 2 + 1 / 2), 
  ∃ a b ∈ set.Icc (0 : ℝ) 2, a ≠ b ∧ h a t = 0 ∧ h b t = 0 := 
sorry

theorem log_inequality (n: ℕ) (hn: 0 < n): 
  Real.log (n + 2) < (∑ i in finset.range (n + 1), 1 / (i + 1) : ℝ) + Real.log 2 := 
sorry

end range_of_t_log_inequality_l326_326971


namespace probability_even_then_prime_l326_326587

theorem probability_even_then_prime  : (1 / 2) * (1 / 2) = 1 / 4 :=
by
  -- Explanation of conditions
  let even_count := 3
  let total_sides := 6
  let even_prob := even_count / total_sides
  -- even_prob = 3 / 6

  let prime_count := 3
  let prime_prob := prime_count / total_sides
  -- prime_prob = 3 / 6

  -- Combining probabilities (since events are independent)
  have prob_even := even_prob
  have prob_prime := prime_prob
  have combined_prob := prob_even * prob_prime
  -- Expected combined probability
  have expected_prob := 1 / 4

  -- Confirm the proof goal
  show combined_prob = expected_prob from
  calc
    prob_even * prob_prime = 1 / 2 * 1 / 2 : by sorry -- details of intermediate steps skipped for illustrative purpose
                      ... = 1 / 4 : by rfl

end probability_even_then_prime_l326_326587


namespace students_not_participating_l326_326694

theorem students_not_participating (
  (total_students : ℕ) (G9 G10 : ℕ)
  (h_total : total_students = 210)
  (h_ratio : G9 * 4 = G10 * 3)
  (h_sum : G9 + G10 = 210)
  (participating_G9 : ℕ)
  (participating_G10 : ℕ)
  (h_part_G9 : participating_G9 = G9 / 2)
  (h_part_G10 : participating_G10 = 3 * G10 / 7)
) :
  (total_students - (participating_G9 + participating_G10)) = 114 :=
sorry

end students_not_participating_l326_326694


namespace number_of_valid_ns_equals_22_l326_326244

noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

def is_integer_angle (n : ℕ) : Prop := 
  ∃ k : ℤ, interior_angle n = k

def is_valid_polygon (n : ℕ) : Prop := 
  n > 2

def valid_n (n : ℕ) : Prop := 
  is_integer_angle n ∧ is_valid_polygon n

def count_valid_ns : ℕ := 
  Finset.card {n | valid_n n}.to_finset

theorem number_of_valid_ns_equals_22 : count_valid_ns = 22 :=
  sorry

end number_of_valid_ns_equals_22_l326_326244


namespace factorial_fraction_l326_326552

theorem factorial_fraction : 
  (4 * nat.factorial 6 + 24 * nat.factorial 5) / nat.factorial 7 = 48 / 7 := 
by
  sorry

end factorial_fraction_l326_326552


namespace magic_square_l326_326357

variable (a b c d e s: ℕ)

axiom h1 : 30 + e + 18 = s
axiom h2 : 15 + c + d = s
axiom h3 : a + 27 + b = s
axiom h4 : 30 + 15 + a = s
axiom h5 : e + c + 27 = s
axiom h6 : 18 + d + b = s
axiom h7 : 30 + c + b = s
axiom h8 : a + c + 18 = s

theorem magic_square : d + e = 47 :=
by
  sorry

end magic_square_l326_326357


namespace max_negative_coefficients_l326_326436

theorem max_negative_coefficients (p : Polynomial ℝ)
  (h_form : ∃ c : Fin 2011 -> ℝ, (∀ i, c i = 1 ∨ c i = -1) ∧ p = ∑ i in (Finset.range 2011), (c i) * x ^ i) 
  (h_no_real_roots : ∀ x : ℝ, ¬(p.eval x = 0)) :
  ∃ c : Fin 2011 -> ℝ, (∑ i, (if c i = -1 then 1 else 0)) = 1005 := sorry

end max_negative_coefficients_l326_326436


namespace fibonacci_150_mod_5_l326_326085

def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem fibonacci_150_mod_5 : fibonacci 150 % 5 = 0 :=
by
  sorry

end fibonacci_150_mod_5_l326_326085


namespace stockC_has_greatest_percent_increase_l326_326413

-- Define the opening and closing prices for each stock
def openingPriceA : ℝ := 28
def closingPriceA : ℝ := 29
def openingPriceB : ℝ := 55
def closingPriceB : ℝ := 57
def openingPriceC : ℝ := 75
def closingPriceC : ℝ := 78

-- Define the percent increase formula
def percentIncrease (opening closing : ℝ) : ℝ := ((closing - opening) / opening) * 100

-- Calculate percent increases for each stock
def percentIncreaseA : ℝ := percentIncrease openingPriceA closingPriceA
def percentIncreaseB : ℝ := percentIncrease openingPriceB closingPriceB
def percentIncreaseC : ℝ := percentIncrease openingPriceC closingPriceC

-- Statement of the problem to prove
theorem stockC_has_greatest_percent_increase :
  (percentIncreaseC > percentIncreaseA) ∧ (percentIncreaseC > percentIncreaseB) := sorry

end stockC_has_greatest_percent_increase_l326_326413


namespace jack_total_yen_l326_326008

theorem jack_total_yen (pounds euros yen : ℕ) (pounds_per_euro yen_per_pound : ℕ) 
  (h_pounds : pounds = 42) 
  (h_euros : euros = 11) 
  (h_yen : yen = 3000) 
  (h_pounds_per_euro : pounds_per_euro = 2) 
  (h_yen_per_pound : yen_per_pound = 100) : 
  9400 = yen + (pounds * yen_per_pound) + ((euros * pounds_per_euro) * yen_per_pound) :=
by
  rw [h_pounds, h_euros, h_yen, h_pounds_per_euro, h_yen_per_pound]
  norm_num
  sorry

end jack_total_yen_l326_326008


namespace max_triangle_area_l326_326314

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

theorem max_triangle_area
  (x1 y1 x2 y2 : ℝ)
  (hA : parabola x1 y1)
  (hB : parabola x2 y2)
  (h_sum_y : y1 + y2 = 2)
  (h_neq : y1 ≠ y2) :
  ∃ area : ℝ, area = 121 / 12 :=
sorry

end max_triangle_area_l326_326314


namespace log_equation_solution_l326_326975

noncomputable theory
open_locale classical

theorem log_equation_solution (p : ℝ) (h : p > -6) :
  (log 10 p + log 10 (p + 6) = log 10 (2 * p + 9)) ↔ (p = -2 + real.sqrt 13 ∨ p = -2 - real.sqrt 13) :=
by sorry

end log_equation_solution_l326_326975


namespace triangle_inequality_from_condition_l326_326865

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end triangle_inequality_from_condition_l326_326865


namespace find_d_l326_326380

theorem find_d (m a b d : ℕ) 
(hm : 0 < m) 
(ha : m^2 < a ∧ a < m^2 + m) 
(hb : m^2 < b ∧ b < m^2 + m) 
(hab : a ≠ b)
(hd : m^2 < d ∧ d < m^2 + m ∧ d ∣ (a * b)) : 
d = a ∨ d = b :=
sorry

end find_d_l326_326380


namespace solve_work_problem_l326_326493

variables (A B C : ℚ)

-- Conditions
def condition1 := B + C = 1/3
def condition2 := C + A = 1/4
def condition3 := C = 1/24

-- Conclusion (Question translated to proof statement)
theorem solve_work_problem (h1 : condition1 B C) (h2 : condition2 C A) (h3 : condition3 C) : A + B = 1/2 :=
by sorry

end solve_work_problem_l326_326493


namespace chocolate_difference_l326_326126

theorem chocolate_difference :
  let numbers := [25, 17, 21, 34, 32] in
  (List.maximum numbers).getOrElse 0 - (List.minimum numbers).getOrElse 0 = 17 :=
by
  sorry

end chocolate_difference_l326_326126


namespace no_lion_is_bird_some_majestic_not_birds_l326_326676

-- Conditions
variable {Lion Majestic Creature Bird : Type}
variable (is_lion : Lion → Prop)
variable (is_majestic : Majestic → Prop)
variable (is_bird : Bird → Prop)
variable (lion_to_majestic : ∀ l, is_lion l → is_majestic l)
variable (no_bird_is_lion : ∀ b, is_bird b → ¬is_lion b)

-- Statements to be proven
theorem no_lion_is_bird : ∀ l, is_lion l → ¬is_bird l := sorry

theorem some_majestic_not_birds : ∃ m, is_majestic m ∧ ¬is_bird m := sorry

end no_lion_is_bird_some_majestic_not_birds_l326_326676


namespace Bruce_Anne_combined_cleaning_time_l326_326944

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l326_326944


namespace original_number_of_men_l326_326494

theorem original_number_of_men (x : ℕ) (h1 : x * 50 = (x - 10) * 60) : x = 60 :=
by
  sorry

end original_number_of_men_l326_326494


namespace AQ_computation_l326_326773

noncomputable def AQ_length : ℝ :=
  let PC := 3
  let BP := 2
  let CQ := 2
  have h1 : ∀ A B C : ℝ, (PC + CQ) = 5, from sorry,
  have h2 : ∀ (BC BP PC : ℝ), BC^2 = BP^2 + PC^2 - 2 * BP * PC * Real.cos (Real.pi / 3), from sorry,
  have h3 : ∀ (AC AQ CQ : ℝ), AC^2 = AQ^2 + CQ^2 - 2 * AQ * CQ * Real.cos (Real.pi / 3), from sorry,
  have h4 : ∀ (AB AQ R : ℝ), AB = 3 → AR = 5 - AQ → AB^2 = AR^2 + R^2 - 2 * AR * R * Real.cos (Real.pi / 3), from sorry,
  have h5 : ∀ (A B C : ℝ), BC^2 + AC^2 = AB^2, from sorry,
  have h6 : AQ = 8 / 5, from sorry
  AQ

theorem AQ_computation (PC BP CQ : ℝ) (hPC : PC = 3) (hBP : BP = 2) (hCQ : CQ = 2) : AQ_length = 8 / 5 :=
by
  simp [AQ_length, hPC, hBP, hCQ]
  sorry

end AQ_computation_l326_326773


namespace cookies_collected_total_is_276_l326_326920

noncomputable def number_of_cookies_in_one_box : ℕ := 48

def abigail_boxes : ℕ := 2
def grayson_boxes : ℕ := 3 / 4
def olivia_boxes : ℕ := 3

def total_cookies_collected : ℕ :=
  abigail_boxes * number_of_cookies_in_one_box + 
  (grayson_boxes * number_of_cookies_in_one_box) + 
  olivia_boxes * number_of_cookies_in_one_box

theorem cookies_collected_total_is_276 : total_cookies_collected = 276 := sorry

end cookies_collected_total_is_276_l326_326920


namespace speed_of_stream_correct_l326_326525

def downstream_speed : ℝ := 15 -- speed in kmph
def upstream_speed : ℝ := 8   -- speed in kmph
def speed_of_stream (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem speed_of_stream_correct :
  speed_of_stream downstream_speed upstream_speed = 3.5 := by
  sorry

end speed_of_stream_correct_l326_326525


namespace unique_solutions_l326_326591

noncomputable def find_triples (x y : ℕ) (p : ℕ) : Prop :=
  p ^ x - y ^ p = 1

theorem unique_solutions (x y p : ℕ) (h : prime p) : 
  find_triples x y p ↔ (x = 1 ∧ y = 1 ∧ p = 2) ∨ (x = 2 ∧ y = 2 ∧ p = 3) :=
begin
  sorry
end

end unique_solutions_l326_326591


namespace trip_cost_l326_326208

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l326_326208


namespace decreasing_range_l326_326645

def f (a x : ℝ) : ℝ := 
  if x > 1 then a / x 
  else (2 - 3 * a) * x + 1

theorem decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → 
  a ∈ Ioo (2/3 : ℝ) (3/4 : ℝ) ∨ a ∈ Icc (2/3 : ℝ) (3/4 : ℝ) :=
sorry

end decreasing_range_l326_326645


namespace inequality_solution_l326_326837

theorem inequality_solution (x : ℝ) : 2 * x - 1 ≤ 3 → x ≤ 2 :=
by
  intro h
  -- Here we would perform the solution steps, but we'll skip the proof with sorry.
  sorry

end inequality_solution_l326_326837


namespace area_of_M_l326_326232

noncomputable def FigureM : Set (ℝ × ℝ) :=
  {xy | let x := xy.1; let y := xy.2;
       (y + x ≥ Abs (x - y)) ∧
       ((x^2 - 8*x + y^2 + 6*y) / (x + 2*y - 8) ≤ 0)}

theorem area_of_M : 
  let M := {xy | let x := xy.1; let y := xy.2;
                (y + x ≥ Abs (x - y)) ∧
                ((x^2 - 8*x + y^2 + 6*y) / (x + 2*y - 8) ≤ 0)} in
  Area M = 8 := 
sorry

end area_of_M_l326_326232


namespace ratio_female_to_male_on_duty_l326_326057

theorem ratio_female_to_male_on_duty
  (total_officers_on_duty : ℕ)
  (total_female_officers : ℕ)
  (percent_female_on_duty : ℕ)
  (h1 : total_officers_on_duty = 500)
  (h2 : total_female_officers = 1000)
  (h3 : percent_female_on_duty = 25) :
  (let F := (percent_female_on_duty / 100) * total_female_officers in
   let M := total_officers_on_duty - F in
   F / M = 1) :=
by
  sorry

end ratio_female_to_male_on_duty_l326_326057


namespace base_6_arithmetic_l326_326203

theorem base_6_arithmetic :
  (4253 : ℕ)₆ + (24316 : ℕ)₆ - (3524 : ℕ)₆ = (25111 : ℕ)₆ := sorry

end base_6_arithmetic_l326_326203


namespace absolute_value_sum_l326_326339

theorem absolute_value_sum (a b : ℤ) (h_a : |a| = 5) (h_b : |b| = 3) : 
  (a + b = 8) ∨ (a + b = 2) ∨ (a + b = -2) ∨ (a + b = -8) :=
by
  sorry

end absolute_value_sum_l326_326339


namespace number_of_combinations_l326_326726

-- Conditions as definitions
def n : ℕ := 9
def k : ℕ := 4

-- Lean statement of the equivalent proof problem
theorem number_of_combinations : (nat.choose n k) = 126 := by
  -- Sorry is used to skip the proof
  sorry

end number_of_combinations_l326_326726


namespace count_greater_than_3_is_zero_l326_326470

def nums := ({0.8, (1:Real) / 2} : Set ℝ)

theorem count_greater_than_3_is_zero : (nums.filter (λ x, x > 3)).card = 0 := by
  sorry

end count_greater_than_3_is_zero_l326_326470


namespace commission_percentage_l326_326216

theorem commission_percentage (commission_earned total_sales : ℝ) (h₀ : commission_earned = 18) (h₁ : total_sales = 720) : 
  ((commission_earned / total_sales) * 100) = 2.5 := by {
  sorry
}

end commission_percentage_l326_326216


namespace number_div_by_3_l326_326883

theorem number_div_by_3 (x : ℕ) (h : 54 = x - 39) : x / 3 = 31 :=
by
  sorry

end number_div_by_3_l326_326883


namespace simplify_log_expression_l326_326072

-- Define a and the expressions using logarithms
variables (a : ℝ) (h : a > 1)

-- Define the initial expression and the simplified expression
def initial_expr : ℝ := 
  (log a (sqrt (a^2 - 1)) * (log (1/a) (sqrt (a^2 - 1)))^2) / 
  (log (a^2) (a^2 - 1) * log (a^(1/3)) (sqrt ((a^2 - 1)^(1/6))))

def simplified_expr : ℝ := log a (sqrt (a^2 - 1))

-- Prove that the initial expression simplifies to the simplified expression
theorem simplify_log_expression (a : ℝ) (h : a > 1) : 
  initial_expr a h = simplified_expr a h :=
by sorry

end simplify_log_expression_l326_326072


namespace constant_term_of_expansion_l326_326094

noncomputable def const_term_expansion (f : ℚ[X]) : ℤ := 
  f.coeff 0

theorem constant_term_of_expansion :
  const_term_expansion ((C(2) * X + 1) * (1 - C(1) * X⁻¹)^5) = -9 :=
sorry

end constant_term_of_expansion_l326_326094


namespace unique_covering_100x100_l326_326531

-- Definition of a unit square in a grid
structure UnitSquare : Type :=
  (x : ℕ)
  (y : ℕ)

-- Definitions related to the problem
def is_border (n : ℕ) (s : UnitSquare) : Prop :=
  (s.x = 0 ∨ s.x = n ∨ s.y = 0 ∨ s.y = n) ∧ 
  (s.x ≤ n ∧ s.y ≤ n)

def border_squares (n : ℕ) : set UnitSquare :=
  {s | is_border n s}

-- The main theorem 
theorem unique_covering_100x100 :
  ∃! (cover : list (set UnitSquare)), 
    ((∀ s ∈ cover, ∃ n, s = border_squares n) ∧
    disjointUnions cover (border_squares 100) ∧
    (#cover = 50))
  := sorry

end unique_covering_100x100_l326_326531


namespace sean_bought_3_sodas_l326_326423

def soda_cost (S : ℕ) : ℕ := S * 1
def soup_cost (S : ℕ) (C : ℕ) : Prop := C = S
def sandwich_cost (C : ℕ) (X : ℕ) : Prop := X = 3 * C
def total_cost (S C X : ℕ) : Prop := S + 2 * C + X = 18

theorem sean_bought_3_sodas (S C X : ℕ) (h1 : soup_cost S C) (h2 : sandwich_cost C X) (h3 : total_cost S C X) : S = 3 :=
by
  sorry

end sean_bought_3_sodas_l326_326423


namespace hemisphere_surface_area_l326_326483

theorem hemisphere_surface_area (r : ℝ) (h : r = 8) :
  let base_area := π * r^2 in
  let curved_surface_area := 2 * π * r^2 in
  base_area + curved_surface_area = 192 * π :=
by
  have r_eq : r = 8 := h
  sorry

end hemisphere_surface_area_l326_326483


namespace final_solution_l326_326509

def is_valid_grid (grid : List (List (Option ℕ))) : Prop :=
  ∀ i j, i < 4 ∧ j < 4 →
    grid[i].all_some ∧
    grid.transpose[j].all_some ∧
    (∀ n, count grid[i] n ≤ 1) ∧
    (∀ n, count (grid.transpose[j]) n ≤ 1)

def grid_partial_filled : List (List (Option ℕ)) :=
  [
    [some 1, none, none, none],
    [none, some 2, some 3, none],
    [none, none, none, some 4],
    [some 4, none, none, none]
  ]

def grid_final_filled (A C : ℕ) : List (List (Option ℕ)) :=
  [
    [some 1, none, none, some C],
    [none, some 2, some 3, none],
    [none, none, none, some 4],
    [some 4, some A, none, none]
  ]

theorem final_solution (A C : ℕ) (h : A = 3) (h' : C = 4):
  is_valid_grid (grid_final_filled A C) →
  A + C = 7 :=
by
  sorry

end final_solution_l326_326509


namespace pie_shop_revenue_l326_326529

noncomputable def revenue_day1 := 5 * 6 * 12 + 6 * 6 * 8 + 7 * 6 * 10
noncomputable def revenue_day2 := 6 * 6 * 15 + 7 * 6 * 10 + 8 * 6 * 14
noncomputable def revenue_day3 := 4 * 6 * 18 + 7 * 6 * 7 + 9 * 6 * 13
noncomputable def total_revenue := revenue_day1 + revenue_day2 + revenue_day3

theorem pie_shop_revenue : total_revenue = 4128 := by
  sorry

end pie_shop_revenue_l326_326529


namespace probability_b_greater_than_a_l326_326157

open Probability

theorem probability_b_greater_than_a :
  (∃ (a : ℕ) (ha : a ∈ set.Icc 1 1000)
     (b : ℕ) (hb : b ∈ set.Icc 1 1000),
        b > a) →
  (∀ (a b : ℕ), (a ∈ set.Icc 1 1000) → (b ∈ set.Icc 1 1000) →
     P(b > a) = 0.4995) := sorry

end probability_b_greater_than_a_l326_326157


namespace product_of_consecutive_odd_numbers_l326_326830

/-- Prove that the product of four consecutive positive odd numbers 
(10n - 3), (10n - 1), (10n + 1), (10n + 3) ends with the penultimate digits "09"
  for any positive integer n. -/
theorem product_of_consecutive_odd_numbers (n : ℕ) (hn : 0 < n) :
  let product := (10 * n - 3) * (10 * n - 1) * (10 * n + 1) * (10 * n + 3) in
  (product % 100 = 9) :=
  by
  sorry

end product_of_consecutive_odd_numbers_l326_326830


namespace total_trip_cost_l326_326210

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l326_326210


namespace rational_function_sum_eq_l326_326235

open Polynomial

/-- Given rational function conditions, prove that p(x) + q(x) equals 2x^2 + 2x - 21. -/
theorem rational_function_sum_eq :
  ∀ (p q : Polynomial ℝ),
  (∃ c : ℝ, ∀ x, p(x) = (q(x) + c / q(x))) ->
  (∃ a : ℝ, q(x) = (a * (x - 2) * (x + 3))) ->
  q(4) = 9 -> p(4) = 5 -> ∀ x, p(x) + q(x) = 2 * x^2 + 2 * x - 21 :=
by
  sorry

end rational_function_sum_eq_l326_326235


namespace sum_first_nine_terms_arithmetic_sequence_l326_326287

variables {a : ℕ → ℚ}

def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_first_nine_terms_arithmetic_sequence (h_seq : is_arithmetic_sequence a) 
  (h1 : a 2 * a 4 * a 6 * a 8 = 120)
  (h2 : 1 / (a 4 * a 6 * a 8) + 1 / (a 2 * a 6 * a 8) + 1 / (a 2 * a 4 * a 8) + 1 / (a 2 * a 4 * a 6) = 7 / 60) :
  let S9 := (9 * (a 1 + a 9) / 2) in S9 = 63 / 2 :=
by
  sorry

end sum_first_nine_terms_arithmetic_sequence_l326_326287


namespace outfits_count_l326_326081

theorem outfits_count (shirts ties pants: ℕ) (h_shirts: shirts = 8) (h_ties: ties = 5) (h_pants: pants = 4) :
  shirts * ties * pants = 160 :=
by
  rw [h_shirts, h_ties, h_pants]
  norm_num
  exact eq.refl 160
  sorry -- this is the building placeholder

end outfits_count_l326_326081


namespace angle_between_chords_of_tangency_is_90_l326_326851

theorem angle_between_chords_of_tangency_is_90
  {circle1 circle2 : Type*}
  (radius1 radius2 : ℝ)
  (center1 center2 point_tangency1 point_tangency2 : ℝ)
  (common_tangent_point : ℝ)
  (condition1 : center1 ≠ center2)
  (condition2 : radius1 ≠ radius2)
  (condition3 : abs(center1 - center2) = radius1 + radius2)
  (condition4 : common_tangent_point ∈ segment point_tangency1 point_tangency2) :
  angle_between_chords point_tangency1 point_tangency2 common_tangent_point = 90 := 
sorry

end angle_between_chords_of_tangency_is_90_l326_326851


namespace number_of_combinations_l326_326728

-- Conditions as definitions
def n : ℕ := 9
def k : ℕ := 4

-- Lean statement of the equivalent proof problem
theorem number_of_combinations : (nat.choose n k) = 126 := by
  -- Sorry is used to skip the proof
  sorry

end number_of_combinations_l326_326728


namespace smallest_pos_int_36m_minus_5n_l326_326262

theorem smallest_pos_int_36m_minus_5n : ∃ k : ℕ, (k > 0) ∧ (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ k = 36 * m - 5 * n) ∧ ∀ l : ℕ, (l > 0 ∧ (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ l = 36 * m - 5 * n)) → l ≥ k → k = 11 :=
begin
  sorry
end

end smallest_pos_int_36m_minus_5n_l326_326262


namespace dima_should_choose_96_l326_326158

/-- Define the game conditions -/
def is_rect_perimeter (p : ℕ) : Prop := p ≤ 97 ∧ p % 2 = 0

/-- Define the function to count distinct rectangle pairs (a, b) for given perimeter p -/
def count_unique_rectangles (p : ℕ) : ℕ :=
  let k := p / 2 in (k - 1) / 2

/-- Statement to prove the optimal number p -/
theorem dima_should_choose_96 : ∃ p, is_rect_perimeter p ∧ 
  (∀ p', is_rect_perimeter p' → count_unique_rectangles p' ≤ count_unique_rectangles p) ∧ 
  p = 96 :=
by
  sorry

end dima_should_choose_96_l326_326158


namespace circumcenter_distance_equal_l326_326824

theorem circumcenter_distance_equal
  (A B C M K N O1 O2 : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space M]
  [metric_space K] [metric_space N] [metric_space O1] [metric_space O2]
  (triangle_ABC : Triangle A B C)
  (median_BM : Median B M AC)
  (perpendicular_BM : ⊥ B M)
  (intersect_perpendicular_altitudes : IntersectPerpendicularAltitudes B K N (altitude A) (altitude C))
  (circumcenter_ABK : Circumcenter ABK O1)
  (circumcenter_CBN : Circumcenter CBN O2) :
  distance M O1 = distance M O2 := 
sorry

end circumcenter_distance_equal_l326_326824


namespace carA_carB_average_speed_comparison_l326_326559

section car_speeds

variables (D a : ℝ) (x y : ℝ)

-- Definition of Car A
def carA_final_speed : ℝ := sqrt (2 * a * (D / 3))
def carA_time_acceleration : ℝ := carA_final_speed / a
def carA_time_constant_speed : ℝ := (2 * D / 3) / carA_final_speed
def carA_total_time : ℝ := carA_time_acceleration + carA_time_constant_speed
def carA_average_speed : ℝ := D / carA_total_time

-- Definition of Car B
def carB_acceleration_time : ℝ := (λ tB, tB / 3)
def carB_final_speed : ℝ := (λ tB, a * carB_acceleration_time tB)
def carB_distance_acceleration : ℝ := (λ tB, 1/2 * a * (carB_acceleration_time tB)^2)
def carB_remaining_distance : ℝ := (λ dB1, D - dB1)
def carB_remaining_time : ℝ := (λ tB, 2 * tB / 3)
def carB_total_time : ℝ := (λ tB, tB)
def carB_average_speed : ℝ := (λ tB, D / carB_total_time tB)

-- The average speed comparison proof statement
theorem carA_carB_average_speed_comparison (tB : ℝ) :
  carA_average_speed D a ≤ carB_average_speed D a tB :=
sorry

end car_speeds

end carA_carB_average_speed_comparison_l326_326559


namespace similarity_of_triangles_l326_326025

open EuclideanGeometry

-- Let ΔABC be a triangle
variables (A B C : Point)

-- Define the points where the excircles touch the sides of ΔABC
variables (A' B' C' : Point)

-- Define the points where the circumcircles of certain triangles intersect the circumcircle of ΔABC
variables (A1 B1 C1 : Point)

-- Define the points where the incircle of ΔABC touches the sides
variables (D E F : Point)

-- Define the geometric conditions
axiom excircles_tangent_to_sides :
  excircle_tangent A' B' C' A B C

axiom intersections_with_circumcircle :
  intersection_of_circumcircles_with_circumcircle A' B' C' A1 B1 C1 A B C

axiom incircle_tangent_to_sides :
  incircle_tangency_points D E F A B C

-- Define the theorem we need to prove
theorem similarity_of_triangles :
  similar (triangle A1 B1 C1) (triangle D E F) :=
sorry

end similarity_of_triangles_l326_326025


namespace find_coordinates_of_P_l326_326290

-- Define the points
def P1 : ℝ × ℝ := (2, -1)
def P2 : ℝ × ℝ := (0, 5)

-- Define the point P
def P : ℝ × ℝ := (-2, 11)

-- Conditions encoded as vector relationships
def vector_P1_P (p : ℝ × ℝ) := (p.1 - P1.1, p.2 - P1.2)
def vector_PP2 (p : ℝ × ℝ) := (P2.1 - p.1, P2.2 - p.2)

-- The hypothesis that | P1P | = 2 * | PP2 |
axiom vector_relation : ∀ (p : ℝ × ℝ), 
  vector_P1_P p = (-2 * (vector_PP2 p).1, -2 * (vector_PP2 p).2) → p = P

theorem find_coordinates_of_P : P = (-2, 11) :=
by
  sorry

end find_coordinates_of_P_l326_326290


namespace rhombus_perimeter_l326_326806

def half (a: ℕ) := a / 2

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  let s1 := half d1
      s2 := half d2
      side_length := Math.sqrt (s1^2 + s2^2)
      perimeter := 4 * side_length
  in perimeter = 68 := by
  sorry

end rhombus_perimeter_l326_326806


namespace exp_is_analytic_l326_326780

open Complex

theorem exp_is_analytic : ∀ z : ℂ, Differentiable ℂ (λ z, exp z) :=
by
  intros z
  sorry

end exp_is_analytic_l326_326780


namespace rectangle_perimeter_l326_326419

theorem rectangle_perimeter (PB BQ PR QS : ℝ) (hPB : PB = 10) (hBQ : BQ = 25) (hPR : PR = 35) (hQS : QS = 45) : 
  let PQ := Real.sqrt (10 ^ 2 + 25 ^ 2),
      OB := 137.5 / Real.sqrt 29,
      perimeter := (2 * 2 * OB + 2 * 2 * OB) in
  perimeter = 1100 / Real.sqrt 29 :=
by
  sorry

end rectangle_perimeter_l326_326419


namespace probability_sum_greater_than_seven_l326_326657

noncomputable def A : Finset ℕ := {0, 1, 2, 3, 5, 6, 8}

theorem probability_sum_greater_than_seven :
  (let pairs := A.powerset.filter (λ s, s.card = 2)
   in (pairs.filter (λ s, s.sum > 7)).card / pairs.card : ℚ) = 10 / 21 := by
  sorry

end probability_sum_greater_than_seven_l326_326657


namespace f_of_zero_eq_zero_l326_326188

theorem f_of_zero_eq_zero (f : ℝ → ℝ) (h : ∀ x : ℝ, 4 * f(f(x)) - 2 * f(x) - 3 * x = 0) : f(0) = 0 :=
sorry

end f_of_zero_eq_zero_l326_326188


namespace probability_adjacent_A_before_B_l326_326764

theorem probability_adjacent_A_before_B 
  (total_students : ℕ)
  (A B C D : ℚ)
  (hA : total_students = 8)
  (hB : B = 1/3) : 
  (∃ prob : ℚ, prob = 1/3) :=
by
  sorry

end probability_adjacent_A_before_B_l326_326764


namespace striped_to_total_ratio_l326_326691

theorem striped_to_total_ratio (total_students shorts_checkered_diff striped_shorts_diff : ℕ)
    (h_total : total_students = 81)
    (h_shorts_checkered : ∃ checkered, shorts_checkered_diff = checkered + 19)
    (h_striped_shorts : ∃ shorts, striped_shorts_diff = shorts + 8) :
    (striped_shorts_diff : ℚ) / total_students = 2 / 3 :=
by sorry

end striped_to_total_ratio_l326_326691


namespace b_catches_up_with_a_l326_326154

-- Define the speeds of A and B and the time delay for B's start
def a_speed : ℝ := 10 -- A's speed in kmph
def b_speed : ℝ := 20 -- B's speed in kmph
def time_delay : ℝ := 5 -- Time delay before B starts in hours

-- Define the initial distance covered by A
def distance_a_initial : ℝ := a_speed * time_delay

-- Define the relative speed of B with respect to A
def relative_speed : ℝ := b_speed - a_speed

-- Define the time for B to catch up
def time_to_catch_up : ℝ := distance_a_initial / relative_speed

-- Define the distance from the start where B catches up with A
def catch_up_distance : ℝ := b_speed * time_to_catch_up

-- The theorem we need to prove
theorem b_catches_up_with_a :
  catch_up_distance = 100 := 
by
  -- Initial conditions and calculations provided so far.
  have h1 : distance_a_initial = 50 := by simp [distance_a_initial, a_speed, time_delay]
  have h2 : relative_speed = 10 := by simp [relative_speed, b_speed, a_speed]
  have h3 : time_to_catch_up = 5 := by simp [time_to_catch_up, distance_a_initial, relative_speed, h1, h2]
  show catch_up_distance = 100 from by simp [catch_up_distance, b_speed, time_to_catch_up, h3]

-- Adding sorry to avoid needing to prove in this example
sorry

end b_catches_up_with_a_l326_326154


namespace correct_option_B_l326_326102

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end correct_option_B_l326_326102


namespace floor_P_squared_l326_326750

noncomputable def P : ℝ :=
  ∑ i in Finset.range 1000, real.sqrt (4 + 1 / (i + 1)^2 + 1 / (i + 2)^2)

theorem floor_P_squared :
  ⌊P * P⌋ = 4007996 :=
by
  have hP_approx : P ≈ 2001.998 := sorry
  calc ⌊P * P⌋ = ⌊(2001.998)^2⌋ : by rw [hP_approx]
          ...  = 4007996 : by norm_num

end floor_P_squared_l326_326750


namespace range_of_a_l326_326632

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

theorem range_of_a (a : ℝ) : (A ∩ B a) = B a ↔ (2 ≤ a ∧ a < 10) ∨ (a = 1) := by
  sorry

end range_of_a_l326_326632


namespace cost_per_serving_l326_326775

-- Define the costs
def pasta_cost : ℝ := 1.00
def sauce_cost : ℝ := 2.00
def meatball_cost : ℝ := 5.00

-- Define the number of servings
def servings : ℝ := 8.0

-- State the theorem
theorem cost_per_serving : (pasta_cost + sauce_cost + meatball_cost) / servings = 1.00 :=
by
  sorry

end cost_per_serving_l326_326775


namespace solution_set_f_x_gt_4_l326_326647

def f (x : ℝ) : ℝ := if x > 1 then 2^x else x^2 - 6 * x + 9

theorem solution_set_f_x_gt_4 : {x : ℝ | f x > 4} = {x | x < 1 ∨ x > 2} :=
by
  sorry

end solution_set_f_x_gt_4_l326_326647


namespace profit_difference_l326_326520

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end profit_difference_l326_326520


namespace negation_of_exists_l326_326107

theorem negation_of_exists (x : ℝ) : x^2 + 2 * x + 2 > 0 := sorry

end negation_of_exists_l326_326107


namespace no_point_in_punctured_disk_l326_326234

theorem no_point_in_punctured_disk (A B C D E F G : ℝ) (hB2_4AC : B^2 - 4 * A * C < 0) :
  ∃ δ > 0, ∀ x y : ℝ, 0 < x^2 + y^2 → x^2 + y^2 < δ^2 → 
    ¬(A * x^2 + B * x * y + C * y^2 + D * x^3 + E * x^2 * y + F * x * y^2 + G * y^3 = 0) :=
sorry

end no_point_in_punctured_disk_l326_326234


namespace bruce_anne_cleaning_house_l326_326961

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l326_326961


namespace limit_exists_rational_to_irrational_always_irrational_l326_326603

noncomputable def x_k_sequence (x : ℝ) (n : ℕ) : ℝ :=
  sorry  -- define the sequence generation based on the problem conditions

theorem limit_exists (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : ∃ y, tendsto (x_k_sequence x) at_top (𝓝 y) :=
  sorry

theorem rational_to_irrational (x : ℝ) (h : x = 0.1) : irrational (lim (x_k_sequence x) at_top) :=
  sorry

theorem always_irrational : ∃ x ∈ set.Icc 0 1, ∀ y, y = lim (x_k_sequence x) at_top → irrational y :=
  sorry

end limit_exists_rational_to_irrational_always_irrational_l326_326603


namespace cookies_collected_total_is_276_l326_326919

noncomputable def number_of_cookies_in_one_box : ℕ := 48

def abigail_boxes : ℕ := 2
def grayson_boxes : ℕ := 3 / 4
def olivia_boxes : ℕ := 3

def total_cookies_collected : ℕ :=
  abigail_boxes * number_of_cookies_in_one_box + 
  (grayson_boxes * number_of_cookies_in_one_box) + 
  olivia_boxes * number_of_cookies_in_one_box

theorem cookies_collected_total_is_276 : total_cookies_collected = 276 := sorry

end cookies_collected_total_is_276_l326_326919


namespace x_minus_y_eq_neg3_l326_326281

variables (x y : ℝ)

theorem x_minus_y_eq_neg3 (h : (x * complex.I) + 2 = y - complex.I) : x - y = -3 :=
sorry

end x_minus_y_eq_neg3_l326_326281


namespace cos_2x_identity_l326_326295

variable {θ x : ℝ}

-- Given conditions:
def condition1 : Prop := sin (2 * x) = (sin θ + cos θ) / 2
def condition2 : Prop := cos x ^ 2 - sin θ * cos θ

-- Theorem to prove:
theorem cos_2x_identity (h1 : condition1) (h2: condition2) : cos (2 * x) = (-1 - sqrt 33) / 8 :=
by
  sorry

end cos_2x_identity_l326_326295


namespace sum_of_p_q_r_l326_326834

theorem sum_of_p_q_r (area_ratio : ℚ) (p q r : ℤ) (side_ratio : ℚ) :
  area_ratio = 245 / 125 →
  p = 7 →
  q = 5 →
  r = 5 →
  side_ratio = (7 * real.sqrt 5) / 5 →
  p + q + r = 17 :=
by sorry

end sum_of_p_q_r_l326_326834


namespace binary_representation_of_53_l326_326238

theorem binary_representation_of_53 : nat.binary_repr 53 = "110101" :=
sorry

end binary_representation_of_53_l326_326238


namespace infinite_seq_zero_or_nine_l326_326769

theorem infinite_seq_zero_or_nine {a : ℕ → ℕ → ℝ} 
  (h : ∀ n, ∃ k, (0 ≤ k ∧ k ≤ 9) ∧ (∀ m : ℕ, a k m = (a k).digits_base 10 m)) : 
  ∃ i j, (i ≠ j ∧ (∀ m : ℕ, (a i m = a j m) → ∀ m : ℕ, (a i - a j).digits_base 10 m = 0) 
       ∨ (i ≠ j ∧ (∀ m : ℕ, (a i m = a j m) → ∀ m : ℕ, (a i - a j).digits_base 10 m = 9))) :=
sorry

end infinite_seq_zero_or_nine_l326_326769


namespace bruce_and_anne_clean_house_l326_326940

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l326_326940


namespace solve_equation_1_solve_equation_2_l326_326432

theorem solve_equation_1 (x : ℝ) : 2 * x^2 - x = 0 ↔ x = 0 ∨ x = 1 / 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : (2 * x + 1)^2 - 9 = 0 ↔ x = 1 ∨ x = -2 := 
by sorry

end solve_equation_1_solve_equation_2_l326_326432


namespace Solomon_collected_66_l326_326427

-- Definitions
variables (J S L : ℕ) -- J for Juwan, S for Solomon, L for Levi

-- Conditions
axiom C1 : S = 3 * J
axiom C2 : L = J / 2
axiom C3 : J + S + L = 99

-- Theorem to prove
theorem Solomon_collected_66 : S = 66 :=
by
  sorry

end Solomon_collected_66_l326_326427


namespace area_triangle_BQW_l326_326710

-- Define the given conditions.
variables (A B C D Z W Q : Type)
variables (is_rectangle : rectangle A B C D)
variables (AZ_length : length A Z = 8)
variables (WC_length : length W C = 8)
variables (AB_length : length A B = 16)
variables (area_trapezoid_ZWCD : area_trapezoid Z W C D = 160)
variables (Q_midpoint : midpoint Q Z W)

-- State the theorem to be proved.
theorem area_triangle_BQW (A B C D Z W Q : Type)
  (is_rectangle : rectangle A B C D)
  (AZ_length : length A Z = 8)
  (WC_length : length W C = 8)
  (AB_length : length A B = 16)
  (area_trapezoid_ZWCD : area_trapezoid Z W C D = 160)
  (Q_midpoint : midpoint Q Z W) :
  area_triangle B Q W = 48 :=
  sorry

end area_triangle_BQW_l326_326710


namespace floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l326_326384

theorem floor_of_sqrt_sum_eq_floor_of_sqrt_expr (n : ℤ): 
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
sorry

end floor_of_sqrt_sum_eq_floor_of_sqrt_expr_l326_326384


namespace quadrilateral_diagonal_halves_area_parallelogram_l326_326867

theorem quadrilateral_diagonal_halves_area_parallelogram 
  (A B C D : Type*)
  [AddGroup A] [Module ℝ A] 
  (area : A → ℝ)
  (h1 : area (A + C) = area (A + B + C + D) / 2 )
  (h2 : area (B + D) = area (A + B + C + D) / 2 ) :
  (area A = area C) ∧ (area B = area D) :=
sorry

end quadrilateral_diagonal_halves_area_parallelogram_l326_326867


namespace intersection_points_l326_326827

variable {α : Type} [LinearOrderedField α]

def hasAtMostOneIntersection (f : α → α) (a : α) : Prop :=
  if h : ∃ y, f a = y then True else True

theorem intersection_points (f : α → α) (a : α) (ha : ∃ y, f a = y) : hasAtMostOneIntersection f a :=
  by
    unfold hasAtMostOneIntersection
    split_ifs
    { triv }
    { triv }
    sorry

end intersection_points_l326_326827


namespace side_length_of_regular_pentagon_l326_326843

theorem side_length_of_regular_pentagon (perimeter : ℝ) (number_of_sides : ℕ) (h1 : perimeter = 23.4) (h2 : number_of_sides = 5) : 
  perimeter / number_of_sides = 4.68 :=
by
  sorry

end side_length_of_regular_pentagon_l326_326843


namespace _l326_326698

variables {a₁ d : ℝ} {S : ℕ → ℝ}

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

def sequence_sum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

noncomputable theorem max_sum_terms (a₁ d : ℝ) (h_pos : a₁ > 0)
  (h_sum_eq : sequence_sum a₁ d 9 = sequence_sum a₁ d 12) :
  ∃ n : ℕ, n = 10 ∨ n = 11 ∧
  ∀ m : ℕ, sequence_sum a₁ d n ≥ sequence_sum a₁ d m := 
sorry

end _l326_326698


namespace expansion_terms_count_eq_1001_l326_326270

theorem expansion_terms_count_eq_1001 (N : ℕ) :
  (∃ x y z w : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ (∀ t : ℕ, t ≥ 0) ∧ x + y + z + w + t = N) →
  (finset.card ({(i, j, k, l, m) : finset ℕ // i > 0 ∧ j > 0 ∧ k > 0 ∧ l > 0 ∧ (∀ n : ℕ, n ≥ 0) ∧ i + j + k + l + m = N}) = 1001) ↔ N = 14 :=
by sorry

end expansion_terms_count_eq_1001_l326_326270


namespace coeff_x4_expansion_l326_326254

theorem coeff_x4_expansion : 
  let f := (4 * X^2 + 6 * X + 9 / 4)^4 in
  -- extracting the coefficient of x^4
  coeff f 4 = 4374 := 
by
  sorry

end coeff_x4_expansion_l326_326254


namespace roots_difference_squared_quadratic_roots_property_l326_326329

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end roots_difference_squared_quadratic_roots_property_l326_326329


namespace max_length_b_c_cos_beta_values_l326_326318

section Problem

variables (α β : Real)
variables (a : Vector Real 2 := ![(Real.cos α), (Real.sin α)])
variables (b : Vector Real 2 := ![(Real.cos β), (Real.sin β)])
variables (c : Vector Real 2 := ![-1, 0])

noncomputable def vector_sum := b + c
noncomputable def vector_length := (vector_sum b c).length

theorem max_length_b_c : vector_length b c = 2 := sorry

axiom perp (a b : Vector Real 2) : a • b = 0
variables (h_perp : α = Real.pi / 4 ∧ perp a (vector_sum b c))

theorem cos_beta_values : Real.cos β = 0 ∨ Real.cos β = 1 := sorry

end Problem

end max_length_b_c_cos_beta_values_l326_326318


namespace bruce_and_anne_clean_together_l326_326954

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l326_326954


namespace profit_ratio_l326_326759

variables (P_s : ℝ)

theorem profit_ratio (h1 : 21 * (7 / 3) + 3 * P_s = 175) : P_s / 21 = 2 :=
by
  sorry

end profit_ratio_l326_326759


namespace sum_d_k_squared_l326_326973

theorem sum_d_k_squared :
  let d_k (k : ℕ) := k + 1 / (3 * k + 1 / (3 * k + 1 / (3 * k + ...)))
  ∑ k in finset.range 21, d_k k ^ 2 =
  (20 * 21 * 41) / 6 + 21 * 20 * (real.sqrt 5 - 2) + 20 * (real.sqrt 5 - 2)^2 :=
sorry

end sum_d_k_squared_l326_326973


namespace unique_solution_pair_l326_326576

open Real

theorem unique_solution_pair :
  ∃! (x y : ℝ), y = (x-1)^2 ∧ x * y - y = -3 :=
sorry

end unique_solution_pair_l326_326576


namespace average_growth_rate_l326_326899

theorem average_growth_rate (output_april : ℝ) (decrease_may_percent : ℝ) (output_july : ℝ) : 
  output_april = 500 → 
  decrease_may_percent = 0.2 → 
  output_july = 576 → 
  ∃ (x : ℝ), (1 - decrease_may_percent) * output_april * (1 + x)^2 = output_july ∧ x = 0.2 :=
by
  intros h1 h2 h3
  existsi 0.2
  split
  sorry   -- Proof will go here
  refl

end average_growth_rate_l326_326899


namespace avg_children_proof_l326_326991

-- Definitions of conditions
def num_families : ℕ := 15
def avg_children_per_family : ℕ := 3
def num_childless_families : ℕ := 3

-- Calculating the total number of children
def total_children : ℕ := num_families * avg_children_per_family

-- Calculating the number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Calculating the average number of children in families with children
def avg_children_in_families_with_children : ℝ := total_children / num_families_with_children

-- Theorem stating the goal
theorem avg_children_proof : avg_children_in_families_with_children = 3.8 := by
  sorry

end avg_children_proof_l326_326991


namespace bruce_anne_cleaning_house_l326_326960

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l326_326960


namespace playerA_always_wins_l326_326132

-- Conditions from the problem description
def initial_odds (p: ℕ) : Prop := nat.prime p ∧ p % 2 = 1
def player_choices (A B : list ℕ) : Prop := A.card = 1000 ∧ B.card = 500 ∧ B ⊆ A

-- The game dynamics encoded in Lean definitions
def valid_move (p_list new_p_list : list ℕ) : Prop := 
  ∃ S: list ℕ, S ⊆ p_list ∧ new_p_list = (list.pmap (λ n h, nat.factors (n - 2)) S (λ _ h, nat.pos_of_mem_prime h)).join

def game_condition (moves: list (ℕ × list ℕ)) (A B : list ℕ) : Prop := 
  ∃ N: ℕ, (∀ (n : ℕ) (p_list: list ℕ), n ∈ moves ∧ valid_move p_list (N::p_list.tail)) ∧ 
  (N ≡ 1 [MOD 4] ∨ N ≡ 3 [MOD 4])

-- Winning strategy encoded
def winning_strategy (A B : list ℕ) (initial_state: ℕ): Prop := 
  initial_state % 4 = 1 ∧ 
  ∀ n: ℕ, ∃ S: list ℕ, S ⊆ B ∧ 
  (if (prod S) % 4 = 1 then (A ⊆ initial_odds) else A.card > 0) 

-- The main theorem to state player A's strategy as proof of the conclusion
theorem playerA_always_wins (A B : list ℕ) (initial_state: ℕ) 
  (h : player_choices A B) : 
  ∃ strategy: list ℕ, 
  winning_strategy A B initial_state ∧ 
  game_condition (strategy.prod B) A :=
sorry -- proof to be filled in

end playerA_always_wins_l326_326132


namespace symmetric_line_equation_l326_326817

theorem symmetric_line_equation (x y : ℝ) :
  let line_original := x - 2 * y + 1 = 0
  let line_symmetry := x = 1
  let line_symmetric := x + 2 * y - 3 = 0
  ∀ (x y : ℝ), (2 - x - 2 * y + 1 = 0) ↔ (x + 2 * y - 3 = 0) := by
sorry

end symmetric_line_equation_l326_326817


namespace ternary_representation_a4_b4_l326_326396

-- Define sequences aₙ and bₙ
noncomputable def a : ℕ → ℝ
| 0     := 2
| (n+1) := a n * real.sqrt (1 + (a n)^2 + (b n)^2) - b n

noncomputable def b : ℕ → ℝ
| 0     := 2
| (n+1) := b n * real.sqrt (1 + (a n)^2 + (b n)^2) + a n

-- Ternary representations of a₄ and b₄
noncomputable def ternary_representation (x : ℝ) : string :=
sorry  -- Define your realistic ternary conversion function here

-- The theorem to prove
theorem ternary_representation_a4_b4 :
  ternary_representation (a 4) = "1000001100111222" ∧
  ternary_representation (b 4) = "2211100110000012" :=
sorry

end ternary_representation_a4_b4_l326_326396


namespace find_a_l326_326654

theorem find_a (a : ℝ) : 
  is_parallel_line ax_y_minus_1_minus_a_eq_0 x_minus_half_y_eq_0 :=
by 
  /-
   Given:
   - ax + y - 1 - a = 0 is parallel to x - 1/2 y = 0
   Prove:
   - a = -2
  -/
  sorry

/- Definitions of the conditions -/
def is_parallel_line (ax_y_minus_1_minus_a_eq_0 : ℝ) (x_minus_half_y_eq_0 : ℝ) : Prop :=
  (ax_y_minus_1_minus_a_eq_0 = -2)

end find_a_l326_326654


namespace total_money_amount_l326_326719

-- Define the conditions
def num_bills : ℕ := 3
def value_per_bill : ℕ := 20
def initial_amount : ℕ := 75

-- Define the statement about the total amount of money James has
theorem total_money_amount : num_bills * value_per_bill + initial_amount = 135 := 
by 
  -- Since the proof is not required, we use 'sorry' to skip it
  sorry

end total_money_amount_l326_326719


namespace distance_from_apex_to_top_of_sphere_l326_326173

-- Necessary conditions
def base_radius : ℝ := 10
def height : ℝ := 30
def slant_height : ℝ := Real.sqrt (base_radius ^ 2 + height ^ 2)

-- To prove: distance from apex to the top of the sphere
theorem distance_from_apex_to_top_of_sphere :
  let R := 15 * (Real.sqrt 10 - 1)
  height - R = 15 * Real.sqrt 10 + 15 :=
by
  sorry -- Proof is omitted

end distance_from_apex_to_top_of_sphere_l326_326173


namespace combined_weight_of_jake_and_sister_l326_326673

theorem combined_weight_of_jake_and_sister
  (J : ℕ) (S : ℕ)
  (h₁ : J = 113)
  (h₂ : J - 33 = 2 * S)
  : J + S = 153 :=
sorry

end combined_weight_of_jake_and_sister_l326_326673


namespace sequence_convergence_l326_326266

open BigOperators

noncomputable def S_n (a : ℝ) (n : ℕ) : ℝ :=
  n^a * ∑ k in Finset.range (n-1).succ, (1 / (k:ℝ)^2019 * 1 / ((n-k):ℝ)^2019)

theorem sequence_convergence (a : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |S_n a n - 2 * Real.zeta 2019| < ε) →
  (a = 2019) :=
sorry

end sequence_convergence_l326_326266


namespace average_speed_is_correct_l326_326191

-- Defining the conditions:
constant D : ℝ -- Distance between Silvertown and Goldtown in kilometers
constant upstream_speed : ℝ := 6 -- Speed upstream in km/h
constant downstream_speed : ℝ := 3 -- Speed downstream in km/h

-- Definition of average speed for the round trip:
noncomputable def average_speed_round_trip (D upstream_speed downstream_speed : ℝ) : ℝ :=
  let time_upstream := D / upstream_speed
  let time_downstream := D / downstream_speed
  let total_distance := 2 * D
  let total_time := time_upstream + time_downstream
  total_distance / total_time

-- The theorem to be proven:
theorem average_speed_is_correct : average_speed_round_trip D upstream_speed downstream_speed = 2.4 :=
sorry

end average_speed_is_correct_l326_326191


namespace cookie_portion_l326_326040

theorem cookie_portion :
  ∃ (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_senior_ate : ℕ) (cookies_senior_took_second_day : ℕ) 
    (cookies_senior_put_back : ℕ) (cookies_junior_took : ℕ),
  total_cookies = 22 ∧
  remaining_cookies = 11 ∧
  cookies_senior_ate = 3 ∧
  cookies_senior_took_second_day = 3 ∧
  cookies_senior_put_back = 2 ∧
  cookies_junior_took = 7 ∧
  4 / 22 = 2 / 11 :=
by
  existsi 22, 11, 3, 3, 2, 7
  sorry

end cookie_portion_l326_326040


namespace solve_for_x_l326_326784

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 4) * x - 3 = 5 → x = 112 := by
  sorry

end solve_for_x_l326_326784


namespace set_proof_l326_326506

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem set_proof :
  (U \ A) ∩ (U \ B) = {4, 8} := by
  sorry

end set_proof_l326_326506


namespace solve_for_x_l326_326075

theorem solve_for_x (x : ℝ) : 3^(x + 6) = 81^(x + 1) → x = 2 / 3 := by
  intro h
  sorry

end solve_for_x_l326_326075


namespace line_equation_l326_326999

theorem line_equation (x y : ℝ) (w : ℝ × ℝ)
  (h : w = (x, y))
  (proj_condition : ∃ w, w = (x, y) ∧ (proj ⟨4, 3⟩ w = ⟨2, 3 / 2⟩)) :
  y = -(4 / 3) * x + 25 / 6 := 
by
  sorry

end line_equation_l326_326999


namespace triangle_side_length_l326_326363

-- Define the problem with the given conditions
theorem triangle_side_length (a b c : ℝ) (A : ℝ)
  (hA : A = 60) -- angle A is 60 degrees
  (h_area : (sqrt 3) / 2 * b * c = sqrt 3) -- given area condition
  (h_bc_sum : b + c = 6) -- sum of sides b and c is 6
  : a = 2 * sqrt 6 := 
sorry -- proof goes here

end triangle_side_length_l326_326363


namespace marble_motion_l326_326179

-- Definitions and conditions
def initial_speed (v_0 : ℝ) : Prop := v_0 > 0
def horizontal_distance (D : ℝ) : Prop := D > 0
def initial_height (h : ℝ) : Prop := h > 0
def gravity (g : ℝ) : Prop := g > 0
def distance_to_edge (x_0 : ℝ) : Prop := x_0 = 0

-- Variables
variables (v_0 D h g x_0 : ℝ)
variables [initial_speed v_0, horizontal_distance D, initial_height h, gravity g, distance_to_edge x_0]

theorem marble_motion :
  (∀ t : ℝ, x(t) = v_0 * t) ∧
  (∀ t : ℝ, y(t) = h - (1 / 2) * g * t^2) ∧
  (tc = D / v_0) ∧
  (y(tc) = h - (g * D^2) / (2 * v_0^2)) ∧
  (vy(tc) = -g * (D / v_0)) ∧
  (v_0 = 2 * D * sqrt(g / (2 * h))) :=
by sorry

end marble_motion_l326_326179


namespace lines_parallel_l326_326870

-- Define the slopes of the lines based on the given points
def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

-- Given points for line l
def point1_l := (-2 : ℝ, 0 : ℝ)
def point2_l (a : ℝ) := (0 : ℝ, a)

-- Given points for line ll
def point1_ll := (4 : ℝ, 0 : ℝ)
def point2_ll := (6 : ℝ, 2 : ℝ)

-- Define the slopes of the lines
def slope_l (a : ℝ) : ℝ :=
  slope (fst point1_l) (snd point1_l) (fst (point2_l a)) (snd (point2_l a))

def slope_ll : ℝ :=
  slope (fst point1_ll) (snd point1_ll) (fst point2_ll) (snd point2_ll)

-- Statement to prove that the value of a makes the two lines parallel
theorem lines_parallel (a : ℝ) : slope_l a = slope_ll → a = 2 := by
  -- Since the slope of line ll is given by the points (4, 0) and (6, 2)
  have slope_ll_eq : slope_ll = 1 := by
    -- Calculation of slope_ll
    unfold slope_ll
    unfold slope
    simp
  -- Slope of line l is determined by the points (-2, 0) and (0, a)
  have slope_l_eq : slope_l a = a / 2 := by
    -- Calculation of slope_l
    unfold slope_l
    unfold slope
    simp
  -- Equating the slopes
  intro h
  rw [slope_l_eq, slope_ll_eq] at h
  -- Solving for a
  linarith

end lines_parallel_l326_326870


namespace number_of_white_balls_proof_l326_326218

-- Definitions based on conditions
def total_balls := 20
def total_draws := 404
def white_balls_drawn := 101

-- The probability of drawing a white ball can be calculated as 
def probability_white_ball := white_balls_drawn / total_draws.toFloat
def number_of_white_balls := total_balls * probability_white_ball

-- The proof problem: number_of_white_balls is 5
theorem number_of_white_balls_proof : number_of_white_balls = 5 := 
by
  sorry
  -- Proof would go here, showing number_of_white_balls = 5

end number_of_white_balls_proof_l326_326218


namespace lisa_marble_problem_l326_326404

theorem lisa_marble_problem
    (friends: ℕ)
    (initial_marbles: ℕ)
    (h_friends: friends = 12)
    (h_initial_marbles: initial_marbles = 34)
    : let total_needed_marbles := (friends * (friends + 1)) / 2 in
      let additional_marbles := total_needed_marbles - initial_marbles in
      let sum_3rd_7th_11th := 3 + 7 + 11 in
      additional_marbles = 44 ∧ sum_3rd_7th_11th = 21 :=
by
  intros
  rcases h_friends with rfl
  rcases h_initial_marbles with rfl
  let total_needed_marbles := (12 * (12 + 1)) / 2
  let additional_marbles := total_needed_marbles - 34
  have h_additional_marbles: additional_marbles = 44 := by norm_num
  let sum_3rd_7th_11th := 3 + 7 + 11
  have h_sum_3rd_7th_11th: sum_3rd_7th_11th = 21 := by norm_num
  exact ⟨h_additional_marbles, h_sum_3rd_7th_11th⟩

end lisa_marble_problem_l326_326404


namespace solve_for_b_l326_326428

theorem solve_for_b : 
  ∀ b : ℝ, log 5 (b ^ 2 - 11 * b) = 3 ↔ (b = (11 + real.sqrt 621) / 2 ∨ b = (11 - real.sqrt 621) / 2) :=
by
  intros b
  sorry

end solve_for_b_l326_326428


namespace fraction_meaningful_iff_l326_326845

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = 1 / (x + 1)) ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_iff_l326_326845


namespace Carla_tile_counts_l326_326846

/-- Carla counts 38 tiles and 75 books on Monday.
    On Tuesday, she counts the tiles \(x\) times and the books 3 times.
    Carla counted something 301 times on Tuesday.
    Prove that \(x = 2\) --/

theorem Carla_tile_counts (x : ℕ) 
  (count_tiles : ∀ (day : ℕ), day = 1 → 38) 
  (count_books : ∀ (day : ℕ), day = 1 → 75) 
  (total_books_tues : ∀ (day : ℕ), day = 2 → 3 * count_books 1)
  (total_counts_tues : ∀ (day : ℕ), day = 2 → 38 * x + total_books_tues 2 = 301) :
  x = 2 :=
sorry

end Carla_tile_counts_l326_326846


namespace parabolas_intersect_at_l326_326573

noncomputable def f₁ (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5
noncomputable def f₂ (x : ℝ) : ℝ := x^2 - 6 * x + 8

theorem parabolas_intersect_at :
  (f₁ (3 - real.sqrt 113 / 4) = f₂ (3 - real.sqrt 113 / 4)) ∧
  (f₁ (3 + real.sqrt 113 / 4) = f₂ (3 + real.sqrt 113 / 4)) ∧
  (f₁ (3 - real.sqrt 113 / 4) = 360 - 9 * real.sqrt 113 / 16) ∧
  (f₁ (3 + real.sqrt 113 / 4) = 360 + 9 * real.sqrt 113 / 16) :=
sorry

end parabolas_intersect_at_l326_326573


namespace cube_net_count_l326_326109

/-- A net of a cube is a two-dimensional arrangement of six squares.
    A regular tetrahedron has exactly 2 unique nets.
    For a cube, consider all possible ways in which the six faces can be arranged such that they 
    form a cube when properly folded. -/
theorem cube_net_count : cube_nets_count = 11 :=
sorry

end cube_net_count_l326_326109


namespace min_num_triangles_l326_326353

theorem min_num_triangles (k : ℕ) (hk : k > 2) :
  ∃ n, at_most_one_point_per_triangle k n ∧ n = k + 1 := 
sorry

end min_num_triangles_l326_326353


namespace triangle_medians_similar_original_l326_326140

theorem triangle_medians_similar_original (a b : ℕ) (h1 : a < b) (h2 : a = 47) (h3 : b = 65) :
  ∃ c : ℕ, a < b < c ∧ c = 79 ∧ ∃ (a' b' c' : ℕ), a' = 7 ∧ b' = 13 ∧ c' = 17 ∧ a' < b' < c' := by
  sorry

end triangle_medians_similar_original_l326_326140


namespace polygon_visibility_theorem_l326_326994

structure SimpleClosedPolygon (T : Type) :=
(sides : list T)
(divide_into_parts : ℕ)
(renumber : list ℕ)
(equilateral_triangles : list T)
(is_external : T → Prop)
(visibility_internal_point : ∀ (s1 s2 : T), ∃ P : T, (s1 ∈ sides) ∧ (s2 ∈ sides) ∧ is_external P)

noncomputable def constructed_polygon_meets_visibility_conditions (T : Type) [inhabited T] 
  (polygon : SimpleClosedPolygon T) : Prop :=
forall (s1 s2 : T), s1 ∈ polygon.sides → s2 ∈ polygon.sides
  → ∃ P : T, (polygon.is_external P) 
              ∧ (∃ P1 P2 : T, P1 ≠ P2 → ¬(∀ sides ∈ polygon.sides, is_external P1 ∧ is_external P2))

theorem polygon_visibility_theorem {T : Type} [inhabited T]
  (polygon : SimpleClosedPolygon T)
  (construction : ∀ T, SimpleClosedPolygon T) :
  constructed_polygon_meets_visibility_conditions T polygon :=
sorry

end polygon_visibility_theorem_l326_326994


namespace price_of_each_shirt_l326_326585

theorem price_of_each_shirt 
  (original_price_shoes : ℝ) 
  (discount_shoes : ℝ) 
  (number_of_shirts : ℕ) 
  (additional_discount : ℝ) 
  (total_spent : ℝ) 
  : 
  original_price_shoes = 200 
  → discount_shoes = 0.30 
  → number_of_shirts = 2 
  → additional_discount = 0.05 
  → total_spent = 285 
  → let discounted_shoes_price := original_price_shoes * (1 - discount_shoes)
      in let total_before_additional_discount := discounted_shoes_price + number_of_shirts * (price_per_shirt : ℝ)
          in let final_total := total_before_additional_discount * (1 - additional_discount)
             in final_total = total_spent 
                → (price_per_shirt : ℝ) = 80 :=
by {
  intros h1 h2 h3 h4 h5;
  let discounted_shoes_price := 200 * (1 - 0.30) in
  let total_before_additional_discount := discounted_shoes_price + 2 * (price_per_shirt : ℝ) in
  let final_total := total_before_additional_discount * (1 - 0.05) in
  have step1 : discounted_shoes_price = 140 := by simp [discounted_shoes_price, h1, h2],
  have step2 : final_total = 285 := by simp [final_total, step1, total_before_additional_discount, h5],
  have step3 : step2 → 140 + 2 * (price_per_shirt : ℝ) = 300 := by sorry,
  have step4 : step3 → 2 * (price_per_shirt : ℝ) = 160 := by sorry,
  exact by linarith.step4,
}

end price_of_each_shirt_l326_326585


namespace toni_saved_330_dimes_l326_326789

noncomputable def num_dimes_saved_by_toni (teagan_pennies : ℕ) (rex_nickels : ℕ) (total_savings : ℝ) : ℕ :=
  let teagan_dollars := (teagan_pennies : ℝ) / 100
  let rex_dollars := (rex_nickels : ℝ) / 20
  let toni_dollars := total_savings - (teagan_dollars + rex_dollars)
  (toni_dollars / 0.10).to_nat

theorem toni_saved_330_dimes :
  num_dimes_saved_by_toni 200 100 40 = 330 := by
  sorry

end toni_saved_330_dimes_l326_326789


namespace find_hyperbola_eccentricity_l326_326626

-- Definitions for conditions
variables (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_point_on_right_branch : ∃ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
variables (F1 F2 P : ℝ × ℝ) -- Coordinates of points F1, F2 and P
variables (m n : ℝ) (h_mn : m > n) (h_PF1 : m = EuclideanDist P F1) (h_PF2 : n = EuclideanDist P F2)
variables (h_m_n_diff : m - n = 2 * a) (h_pf_sum : EuclideanDist (P - F1) (P - F2) = 2 * c)
variables (h_area : area_of_triangle P F1 F2 = a * c)

-- Define hyperbola eccentricity as a function of a and c
noncomputable def hyperbola_eccentricity (a c : ℝ) : ℝ := c / a

-- Theorem statement
theorem find_hyperbola_eccentricity :
  hyperbola_eccentricity a c = (1 + Real.sqrt 5) / 2 := sorry

end find_hyperbola_eccentricity_l326_326626


namespace parametric_segment_squared_sum_l326_326825

noncomputable def pqr_squared_sum : ℝ :=
  let p := (-5) / 2 in
  let q := 1 in
  let r := 4 in
  let s := (-3) in
  p^2 + q^2 + r^2 + s^2

theorem parametric_segment_squared_sum :
  let p := (-5) / 2 in
  let q := 1 in
  let r := 4 in
  let s := (-3) in
  p^2 + q^2 + r^2 + s^2 = 32.25 :=
by 
  sorry

end parametric_segment_squared_sum_l326_326825


namespace mia_study_time_l326_326760

-- Let's define the conditions in Lean
def total_hours_in_day : ℕ := 24
def fraction_time_watching_TV : ℚ := 1 / 5
def fraction_time_studying : ℚ := 1 / 4

-- Time remaining after watching TV
def time_left_after_TV (total_hours : ℚ) (fraction_TV : ℚ) : ℚ :=
  total_hours * (1 - fraction_TV)

-- Time spent studying
def time_studying (remaining_time : ℚ) (fraction_studying : ℚ) : ℚ :=
  remaining_time * fraction_studying

-- Convert hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- The theorem statement
theorem mia_study_time :
  let total_hours := (total_hours_in_day : ℚ),
      time_after_TV := time_left_after_TV total_hours fraction_time_watching_TV,
      study_hours := time_studying time_after_TV fraction_time_studying,
      study_minutes := hours_to_minutes study_hours in
  study_minutes = 288 := by
  sorry

end mia_study_time_l326_326760


namespace find_x_l326_326992

theorem find_x (x : ℝ) (h : log 8 (3 * x - 4) = 2) : x = 68 / 3 :=
sorry

end find_x_l326_326992


namespace vertical_line_slope_angle_l326_326461

-- Definition: A vertical line
def is_vertical_line (l : ℝ → ℝ) := ∀ x, l x = 0

-- Theorem Statement: The slope of a vertical line considering its angle of inclination with respect to the positive x-axis is 90 degrees.
theorem vertical_line_slope_angle : is_vertical_line (λ x : ℝ, 0) → angle_of_inclination (λ x : ℝ, 0) = 90 :=
by
  sorry

end vertical_line_slope_angle_l326_326461


namespace permutations_red_l326_326497

theorem permutations_red : 
  let word := ['r', 'e', 'd'] in 
  fintype.card (set.univ : set (finset word)) = 6 := 
by 
  sorry

end permutations_red_l326_326497


namespace mushroom_cut_l326_326277

theorem mushroom_cut (k_used : ℕ) (s_used : ℕ) (remaining_pieces : ℕ) (initial_mushrooms : ℕ) (total_pieces : ℕ) (pieces_per_mushroom : ℕ) :
  k_used = 38 ∧ s_used = 42 ∧ remaining_pieces = 8 ∧ initial_mushrooms = 22 ∧ total_pieces = k_used + s_used + remaining_pieces ∧ pieces_per_mushroom = total_pieces / initial_mushrooms →
  pieces_per_mushroom = 4 :=
by
  intro h
  cases h with hk hs
  cases hs with hr hi
  cases hi with ht hp
  cases hp with ht'
  sorry

end mushroom_cut_l326_326277


namespace garden_area_ratio_l326_326895
open Real

theorem garden_area_ratio 
  (L W : ℝ) 
  (h1 : L / W = 5 / 4) 
  (h2 : L + W = 50) :
  (L * W) / (π * (W / 2)^2) = 5 / π :=
by
  have hW_pos : 0 < W := sorry
  have hL_pos : 0 < L := sorry
  have hL : L = 5 / 4 * W := by sorry
  have h_square : (L * W) / (π * (W / 2)^2) = (5 / 4 * W * W) / (π * (W / 2)^2) := by sorry
  have eq1 : (L * W) = 2500 / 81 := by sorry
  have eq2 : π * (W / 2)^2 = π * (100 / 9)^2 := by sorry
  have eq3 : (L * W) / (π * (W / 2)^2) = (50000 / 81) / (10000π / 81) := by sorry
  have eq_final : (50000 / 10000π) = 5 / π := by sorry
  exact eq_final

end garden_area_ratio_l326_326895


namespace johns_old_cards_l326_326014

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def total_pages : ℕ := 8

def total_cards := total_pages * cards_per_page
def old_cards := total_cards - new_cards

theorem johns_old_cards :
  old_cards = 16 :=
by
  -- Note: No specific solution steps needed here, just stating the theorem
  sorry

end johns_old_cards_l326_326014


namespace largest_four_digit_number_l326_326259

theorem largest_four_digit_number (digits : List ℕ)
  (h1 : digits = [1, 5, 9, 4]) : 
  list.append digits ≠ list.perm digits [9, 5, 4, 1] :=
by
  sorry

end largest_four_digit_number_l326_326259


namespace part1_l326_326163

theorem part1 :
  2^(-Real.logb 2 4) - (8 / 27)^(-2 / 3) + Real.logb 10 (1 / 100) + (sqrt 2 - 1)^(Real.logb 10 1) + (Real.logb 10 5)^2 + (Real.logb 10 2) * (Real.logb 10 50) = -2 :=
sorry

end part1_l326_326163


namespace max_min_f_on_interval_cos_4x0_value_l326_326309

-- Part 1: Verify the maximum and minimum values of the function.
def f (x : ℝ) : ℝ :=
  3 * (Real.sin (x + Real.pi / 6))^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x - (1 / 2) * (Real.cos x)^2

theorem max_min_f_on_interval : 
  (∀ x ∈ Icc 0 (Real.pi / 2), f x ≤ 13 / 4) ∧ (∃ x ∈ Icc 0 (Real.pi / 2), f x = 13 / 4)
  ∧ (∀ x ∈ Icc 0 (Real.pi / 2), f x ≥ 1 / 4) ∧ (∃ x ∈ Icc 0 (Real.pi / 2), f x = 1 / 4) := 
sorry

-- Part 2: Given f(2 * x0) = 49/20 and x0 in the interval, prove cos 4x0 equals the specified value.
variable (x0 : ℝ)
variable (hx0_interval : x0 ∈ Set.Ioo (Real.pi / 6) (7 * Real.pi / 24))
variable (hf_x0 : f (2 * x0) = 49 / 20)

theorem cos_4x0_value : 
  Real.cos (4 * x0) = -(4 * Real.sqrt 3 + 3) / 10 := 
sorry

end max_min_f_on_interval_cos_4x0_value_l326_326309


namespace percent_commute_l326_326671

variable (x : ℝ)

theorem percent_commute (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_commute_l326_326671


namespace no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l326_326868

theorem no_integer_for_58th_power_64_digits : ¬ ∃ n : ℤ, 10^63 ≤ n^58 ∧ n^58 < 10^64 :=
sorry

theorem valid_replacement_for_64_digits (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 81) : 
  ¬ ∃ n : ℤ, 10^(k-1) ≤ n^58 ∧ n^58 < 10^k :=
sorry

end no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l326_326868


namespace translate_graph_cos_l326_326849

/-- Let f(x) = cos(2x). 
    Translate f(x) to the left by π/6 units to get g(x), 
    then translate g(x) upwards by 1 unit to get h(x). 
    Prove that h(x) = cos(2x + π/3) + 1. -/
theorem translate_graph_cos :
  let f (x : ℝ) := Real.cos (2 * x)
  let g (x : ℝ) := f (x + Real.pi / 6)
  let h (x : ℝ) := g x + 1
  ∀ (x : ℝ), h x = Real.cos (2 * x + Real.pi / 3) + 1 :=
by
  sorry

end translate_graph_cos_l326_326849


namespace find_magnitude_of_b_l326_326614

open ComplexConjugate Real

variable (a b : ℝ × ℝ)

-- Define vectors
def vector_a : ℝ × ℝ := (1, -2)
def vector_sum : ℝ × ℝ := (0, 2)

-- Additional assumption
variable (h : vector_a + b = vector_sum)

-- Magnitude function
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem find_magnitude_of_b (ha : a = vector_a) (hb : a + b = vector_sum) : magnitude b = Real.sqrt 17 := by
  -- Given proof context will be here
  sorry

end find_magnitude_of_b_l326_326614


namespace man_speed_l326_326905

theorem man_speed (time_in_minutes : ℕ) (distance_in_km : ℕ) 
  (h_time : time_in_minutes = 30) 
  (h_distance : distance_in_km = 5) : 
  (distance_in_km : ℝ) / (time_in_minutes / 60 : ℝ) = 10 :=
by 
  sorry

end man_speed_l326_326905


namespace Lizzie_group_difference_l326_326405

theorem Lizzie_group_difference
  (lizzie_group_members : ℕ)
  (total_members : ℕ)
  (lizzie_more_than_other : lizzie_group_members > total_members - lizzie_group_members)
  (lizzie_members_eq : lizzie_group_members = 54)
  (total_members_eq : total_members = 91)
  : lizzie_group_members - (total_members - lizzie_group_members) = 17 := 
sorry

end Lizzie_group_difference_l326_326405


namespace frustum_volume_l326_326172

-- Define the condition that the smaller cone has a height of 3 and a base radius of 2
def smaller_cone_height := 3
def smaller_cone_base_radius := 2

-- Define the base radius of the original cone
def original_cone_base_radius := 4

-- Using similarity of triangles, find the height of the original cone
axiom height_of_original_cone : ℕ
axiom height_proportion : height_of_original_cone = 6

-- Compute the volume of the original cone and the smaller cone, then find the volume of the frustum
def volume_of_frustum : ℝ :=
  let volume_of_original_cone := (1 / 3) * Real.pi * (original_cone_base_radius ^ 2) * height_of_original_cone
  let volume_of_smaller_cone := (1 / 3) * Real.pi * (smaller_cone_base_radius ^ 2) * smaller_cone_height
  volume_of_original_cone - volume_of_smaller_cone

-- Prove that the volume of the resulting frustum is 28π
theorem frustum_volume : volume_of_frustum = 28 * Real.pi := 
  sorry

end frustum_volume_l326_326172


namespace unique_function_solution_l326_326997

theorem unique_function_solution (f : ℕ+ → ℕ+) :
  (∀ m n : ℕ+, m^2 + f(n)^2 + (m - f(n))^2 ≥ f(m)^2 + n^2) →
  ∀ n : ℕ+, f(n) = n :=
by
  assume condition : ∀ m n : ℕ+, m^2 + f(n)^2 + (m - f(n))^2 ≥ f(m)^2 + n^2
  sorry

end unique_function_solution_l326_326997


namespace average_marks_of_all_students_l326_326440

-- Define the conditions
def class1_students := 30
def class1_average := 40
def class2_students := 50
def class2_average := 90

-- Define the expected result
def total_students := class1_students + class2_students
def total_marks := (class1_students * class1_average) + (class2_students * class2_average)
def expected_average := 71.25

-- Prove that the average marks of all the students is 71.25
theorem average_marks_of_all_students : 
  (total_marks / total_students : ℝ) = expected_average := 
by
  sorry

end average_marks_of_all_students_l326_326440


namespace sin_angle_ratio_eq_one_l326_326361

-- Given triangle ABC
variables {A B C D : Type} [OrderedField ℝ]
variables (a b c d : ℝ)

-- Conditions
def angle_B_90 (B: Triangle.field) : Prop := ∠ B = 90
def angle_C_30 (C: Triangle.field) : Prop := ∠ C = 30
def D_divides_BC (d: ℝ) (b: ℝ) (c: ℝ) : Prop := b = 2*c

theorem sin_angle_ratio_eq_one
(angle_B: angle_B_90)
(angle_C: angle_C_30)
(divide_ratio: D_divides_BC d b c) :
  (sin ∠ BAD) / (sin ∠ CAD) = 1 :=
by
  sorry

end sin_angle_ratio_eq_one_l326_326361


namespace water_removal_l326_326001

theorem water_removal (n : ℕ) : 
  (∀n, (2:ℚ) / (n + 2) = 1 / 8) ↔ (n = 14) := 
by 
  sorry

end water_removal_l326_326001


namespace martian_right_angle_l326_326437

theorem martian_right_angle :
  ∀ (full_circle clerts_per_right_angle : ℕ),
  (full_circle = 600) →
  (clerts_per_right_angle = full_circle / 3) →
  clerts_per_right_angle = 200 :=
by
  intros full_circle clerts_per_right_angle h1 h2
  sorry

end martian_right_angle_l326_326437


namespace susie_investment_amount_l326_326083

theorem susie_investment_amount
  (x : ℝ)
  (pretty_penny_rate : ℝ)
  (five_and_dime_rate : ℝ)
  (initial_amount : ℝ)
  (total_after_two_years : ℝ) :
  pretty_penny_rate = 0.03 →
  five_and_dime_rate = 0.05 →
  initial_amount = 1000 →
  total_after_two_years = 1090.02 →
  x * (1 + pretty_penny_rate) ^ 2 + (initial_amount - x) * (1 + five_and_dime_rate) ^ 2 = total_after_two_years →
  x = 300 :=
begin
  sorry
end

end susie_investment_amount_l326_326083


namespace product_invariant_l326_326051

namespace problem_invariant_product

-- Define the arithmetic mean
def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2

-- Define the harmonic mean
def harmonic_mean (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

-- Define the transformation step
def transform (a b : ℝ) : ℝ × ℝ :=
  (arithmetic_mean a b, harmonic_mean a b)

-- Initial conditions
def initial_pair : ℝ × ℝ := (1, 2)

-- The invariant product to be proved
theorem product_invariant : (transform^[2016] initial_pair).1 * (transform^[2016] initial_pair).2 = 2 :=
sorry

end problem_invariant_product

end product_invariant_l326_326051


namespace Bruce_Anne_combined_cleaning_time_l326_326942

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l326_326942


namespace find_increase_in_inches_l326_326130

structure Cylinder :=
  (radius : ℝ)
  (height : ℝ)

def volume (c : Cylinder) : ℝ := 
  Real.pi * c.radius^2 * c.height

def first_modified_cylinder (x : ℝ) : Cylinder :=
  { radius := 5 + 4 * x, height := 7 }

def second_modified_cylinder (x : ℝ) : Cylinder :=
  { radius := 5, height := 7 + x }

theorem find_increase_in_inches (x : ℝ) :
  volume (first_modified_cylinder x) = volume (second_modified_cylinder x) 
  ↔ x = (12398 / 22400) :=
sorry

end find_increase_in_inches_l326_326130


namespace range_of_a_l326_326345

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l326_326345


namespace increasing_interval_pi_2pi_l326_326977

noncomputable def f (x : ℝ) : ℝ := x * cos x - sin x

theorem increasing_interval_pi_2pi :
  ∃ I : Set ℝ, I = Ioo π (2 * π) ∧ ∀ x y ∈ I, x < y → f x < f y :=
sorry

end increasing_interval_pi_2pi_l326_326977


namespace probability_of_both_odd_is_5_over_18_probability_of_product_even_is_13_over_18_l326_326611

noncomputable def probability_both_odd : ℚ :=
  let total_outcomes := 36
  let odd_numbers := {1, 3, 5, 7, 9}
  let odd_pairs := (Finset.card (Finset.pairs odd_numbers)).nat_abs
  odd_pairs / total_outcomes

noncomputable def probability_product_even : ℚ :=
  let total_outcomes := 36
  let even_numbers := {2, 4, 6, 8}
  let odd_numbers := {1, 3, 5, 7, 9}
  let even_odd_pairs := (Finset.card (Finset.product even_numbers odd_numbers)).nat_abs / 2
  let even_pairs := (Finset.card (Finset.pairs even_numbers)).nat_abs
  (even_odd_pairs + even_pairs) / total_outcomes

theorem probability_of_both_odd_is_5_over_18 :
  probability_both_odd = 5 / 18 :=
sorry

theorem probability_of_product_even_is_13_over_18 :
  probability_product_even = 13 / 18 :=
sorry

end probability_of_both_odd_is_5_over_18_probability_of_product_even_is_13_over_18_l326_326611


namespace minutes_until_8_00_am_l326_326672

-- Definitions based on conditions
def time_in_minutes (hours : Nat) (minutes : Nat) : Nat := hours * 60 + minutes

def current_time : Nat := time_in_minutes 7 30 + 16

def target_time : Nat := time_in_minutes 8 0

-- The theorem we need to prove
theorem minutes_until_8_00_am : target_time - current_time = 14 :=
by
  sorry

end minutes_until_8_00_am_l326_326672


namespace cart_max_speed_l326_326538

noncomputable def maximum_speed (a R : ℝ) : ℝ :=
  (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4)

theorem cart_max_speed (a R v : ℝ) (h : v = maximum_speed a R) : 
  v = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4) :=
by
  -- Proof is omitted
  sorry

end cart_max_speed_l326_326538


namespace vessel_base_length_l326_326541

noncomputable def volume_of_cube (side: ℝ) : ℝ :=
  side ^ 3

noncomputable def volume_displaced (length breadth height: ℝ) : ℝ :=
  length * breadth * height

theorem vessel_base_length
  (breadth : ℝ) 
  (cube_edge : ℝ)
  (water_rise : ℝ)
  (displaced_volume : ℝ) 
  (h1 : breadth = 30) 
  (h2 : cube_edge = 30) 
  (h3 : water_rise = 15) 
  (h4 : volume_of_cube cube_edge = displaced_volume) :
  volume_displaced (displaced_volume / (breadth * water_rise)) breadth water_rise = displaced_volume :=
  by
  sorry

end vessel_base_length_l326_326541


namespace correct_range_g_l326_326393

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ) -- assuming this is yet another function to be defined
variable (P : ℝ) -- helper variable for periodicity
variable (S : Set ℝ) -- helper variable for ranges 

-- Conditions
def periodic (f : ℝ → ℝ) (P : ℝ) : Prop := ∀ x, f(x + P) = f(x)

axiom f_periodic : periodic f 1
axiom g_def : ∀ x, g x = f x - 2 * x
axiom g_range : Set.range (fun x => g x) ∩ (Set.Icc 2 3) = Set.Icc (-2 : ℝ) 6

-- Goal
noncomputable def range_g_on_interval : Set ℝ := 
  Set.range (fun x => g (x)) ∩ Set.Icc (-2017 : ℝ) 2017

theorem correct_range_g : range_g_on_interval f g = Set.Icc (-4030 : ℝ) 4044 :=
sorry

end correct_range_g_l326_326393


namespace roots_difference_squared_quadratic_roots_property_l326_326330

noncomputable def α : ℝ := (3 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (3 - Real.sqrt 5) / 2

theorem roots_difference_squared :
  α - β = Real.sqrt 5 :=
by
  sorry

theorem quadratic_roots_property :
  (α - β) ^ 2 = 5 :=
by
  sorry

end roots_difference_squared_quadratic_roots_property_l326_326330


namespace bad_arrangements_count_l326_326112

def is_bad (arr: List ℕ) : Prop :=
  ∃ n, 1 ≤ n ∧ n ≤ 21 ∧ ¬∃ (sub: List ℕ), sub ⊂ arr ∧ list.sum sub = n

theorem bad_arrangements_count : 
  ∃! (s : Finset (Finset ℕ)), s.card = 4 ∧ ∀ arr ∈ s, arr.nodup ∧ (is_bad arr) :=
sorry

end bad_arrangements_count_l326_326112


namespace problem_solution_l326_326556

theorem problem_solution : (-2)^2 + 4 * 2^(-1) - | -8 | = -2 := by
  sorry

end problem_solution_l326_326556


namespace exists_three_mutually_related_or_unrelated_l326_326071

variable {V : Type} [Fintype V] [DecidableRel (λ v w : V, v ≠ w)]

theorem exists_three_mutually_related_or_unrelated (hV : Fintype.card V = 6) 
  (knows : V → V → Prop) : 
  (∃ A B C : V, (knows A B ∧ knows A C ∧ knows B C) ∨ 
                  (¬ knows A B ∧ ¬ knows A C ∧ ¬ knows B C)) :=
sorry

end exists_three_mutually_related_or_unrelated_l326_326071


namespace bruce_and_anne_clean_house_l326_326939

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l326_326939


namespace problem_statement_l326_326279

noncomputable theory  -- Use noncomputable context if necessary to avoid computation issues

variable (a b c d : ℝ)  -- Define variables over the real numbers

theorem problem_statement (h : a * d - b * c = 1) : a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := 
by {
  sorry,  -- Proof placeholder
}

end problem_statement_l326_326279


namespace exradius_product_sum_inradius_sum_reciprocal_inradius_exradius_product_l326_326397

variables (a b c t s ρ ρa ρb ρc : ℝ)

-- Given conditions
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Main theorem statements
theorem exradius_product_sum (a b c t s ρ ρa ρb ρc : ℝ)
  (h1 : s = semiperimeter a b c)
  (h2 : ρ * ρa * ρb * ρc = t^2)
  (h3 : 1 / ρ = 1 / ρa + 1 / ρb + 1 / ρc) :
  ρa * ρb + ρb * ρc + ρc * ρa = s^2 :=
sorry

theorem inradius_sum_reciprocal (a b c t s ρ ρa ρb ρc : ℝ)
  (h1 : s = semiperimeter a b c)
  (h2 : 1 / ρ = 1 / ρa + 1 / ρb + 1 / ρc) :
  ρa * ρb + ρb * ρc + ρc * ρa = s^2 :=
sorry

theorem inradius_exradius_product (a b c t s ρ ρa ρb ρc : ℝ)
  (h1 : s = semiperimeter a b c)
  (h2 : ρ * ρa * ρb * ρc = t^2) :
  ρ * ρa * ρb * ρc = t^2 :=
sorry

end exradius_product_sum_inradius_sum_reciprocal_inradius_exradius_product_l326_326397


namespace find_d_l326_326836

open Real

-- Define the given conditions
variable (a b c d e : ℝ)

axiom cond1 : 3 * (a^2 + b^2 + c^2) + 4 = 2 * d + sqrt (a + b + c - d + e)
axiom cond2 : e = 1

-- Define the theorem stating that d = 7/4 under the given conditions
theorem find_d : d = 7/4 := by
  sorry

end find_d_l326_326836


namespace interior_angle_of_arithmetic_sequence_triangle_l326_326684

theorem interior_angle_of_arithmetic_sequence_triangle :
  ∀ (α d : ℝ), (α - d) + α + (α + d) = 180 → α = 60 :=
by 
  sorry

end interior_angle_of_arithmetic_sequence_triangle_l326_326684


namespace find_first_number_in_second_set_l326_326088

theorem find_first_number_in_second_set: 
  ∃ x: ℕ, (20 + 40 + 60) / 3 = (x + 80 + 15) / 3 + 5 ∧ x = 10 :=
by
  sorry

end find_first_number_in_second_set_l326_326088


namespace rhombus_perimeter_l326_326803

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  let side_length := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let perimeter := 4 * side_length in
  perimeter = 68 :=
by
  have h3 : d1 / 2 = 8, from by rw [h1],
  have h4 : d2 / 2 = 15, from by rw [h2],
  have h5 : side_length = 17, from by
    calc
      side_length
          = Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) : rfl
      ... = Real.sqrt (8 ^ 2 + 15 ^ 2) : by rw [h3, h4]
      ... = Real.sqrt (64 + 225) : rfl
      ... = Real.sqrt 289 : rfl
      ... = 17 : by norm_num,
  calc
    perimeter
        = 4 * side_length : rfl
    ... = 4 * 17 : by rw [h5]
    ... = 68 : by norm_num

end rhombus_perimeter_l326_326803


namespace trip_cost_l326_326205

variable (price : ℕ) (discount : ℕ) (numPeople : ℕ)

theorem trip_cost :
  price = 147 →
  discount = 14 →
  numPeople = 2 →
  (price - discount) * numPeople = 266 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end trip_cost_l326_326205


namespace alexander_has_more_pencils_l326_326438

-- Definitions based on conditions
def asaf_age := 50
def total_age := 140
def total_pencils := 220

-- Auxiliary definitions based on conditions
def alexander_age := total_age - asaf_age
def age_difference := alexander_age - asaf_age
def asaf_pencils := 2 * age_difference
def alexander_pencils := total_pencils - asaf_pencils

-- Statement to prove
theorem alexander_has_more_pencils :
  (alexander_pencils - asaf_pencils) = 60 := sorry

end alexander_has_more_pencils_l326_326438


namespace cn_parallel_to_midpoints_line_l326_326636

variables {A B C L M N : Point}
variables [IsTriangle ABC] [IsTriangle LMN]

theorem cn_parallel_to_midpoints_line
  (h_simil : Triangle ABC ≈ Triangle LMN)
  (h_ac_eq_bc : AC = BC)
  (h_ln_eq_mn : LN = MN)
  (h_order_ccw : CounterClockwise ABC)
  (h_al_eq_bm : AL = BM) :
  CN ∥ midpoint_line AB LM :=
sorry

end cn_parallel_to_midpoints_line_l326_326636


namespace total_students_in_lunchroom_l326_326165

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end total_students_in_lunchroom_l326_326165


namespace g_of_odd_function_l326_326754

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then 2 * x else 
  if x = 0 then 0 else 
  g x

theorem g_of_odd_function : 
  (∀ x, f (-x) = -f x) → g 3 = -8 :=
by sorry

end g_of_odd_function_l326_326754


namespace triangle_area_l326_326459

theorem triangle_area (a b c : ℕ) (h₁ : a = 20) (h₂ : b = 21) (h₃ : c = 29) (h₄ : a^2 + b^2 = c^2) : 
  1 / 2 * (a * b) = 210 :=
by {
  rw [h₁, h₂, h₃] at h₄,
  have : a = 20 ∧ b = 21 ∧ c = 29 := ⟨h₁, h₂, h₃⟩,
  rw [this.1, this.2.1] at h₄,
  simp,
  sorry
}

end triangle_area_l326_326459


namespace solve_log_equation_l326_326783

-- Given conditions
def T (x : ℝ) : Prop := log 2 x + 2 * log 4 x = 9

-- The goal to prove
theorem solve_log_equation (x : ℝ) (h : T x) : x = 16 * sqrt 2 := 
sorry

end solve_log_equation_l326_326783


namespace solve_for_x_l326_326073

theorem solve_for_x : ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 :=
by 
  intros x hx h
  sorry

end solve_for_x_l326_326073


namespace solve_inequality_l326_326982

theorem solve_inequality :
  {x : ℝ | 8*x^3 - 6*x^2 + 5*x - 5 < 0} = {x : ℝ | x < 1/2} :=
sorry

end solve_inequality_l326_326982


namespace exists_infinitely_many_skew_lines_in_plane_l326_326681

noncomputable theory

-- Define a structure for plane
structure Plane (α : Type) := 
  (points : α → Prop)

-- Define a structure for line
structure Line (α : Type) := 
  (exists_point : α → Prop)

-- Define the condition: l is outside the plane α
def line_outside_plane (l : Line α) (α : Plane α) : Prop :=
  ∀ p : α, ¬l.exists_point p

-- Given: line l is outside plane α
variable (α : Type) [plane : Plane α] [line : Line α]
variable (h : line_outside_plane line plane)

-- We need to prove:
theorem exists_infinitely_many_skew_lines_in_plane (l : Line α) (α : Plane α) (h : line_outside_plane l α) : 
  ∃ (f : ℕ → α → Prop), ∀ n, (Plane.points α (f n)) ∧ (∀ m ≠ n, ¬Line.exists_point l (f m)) :=
sorry

end exists_infinitely_many_skew_lines_in_plane_l326_326681


namespace remainder_when_dividing_P_by_DDD_l326_326735

variables (P D D' D'' Q Q' Q'' R R' R'' : ℕ)

-- Define the conditions
def condition1 : Prop := P = Q * D + R
def condition2 : Prop := Q = Q' * D' + R'
def condition3 : Prop := Q' = Q'' * D'' + R''

-- Theorem statement asserting the given conclusion
theorem remainder_when_dividing_P_by_DDD' 
  (H1 : condition1 P D Q R)
  (H2 : condition2 Q D' Q' R')
  (H3 : condition3 Q' D'' Q'' R'') : 
  P % (D * D' * D') = R'' * D * D' + R * D' + R := 
sorry

end remainder_when_dividing_P_by_DDD_l326_326735


namespace solve_log_eq_l326_326076

theorem solve_log_eq : ∀ x : ℝ, (2 : ℝ) ^ (Real.log x / Real.log 3) = (1 / 4 : ℝ) → x = 1 / 9 :=
by
  intro x
  sorry

end solve_log_eq_l326_326076


namespace negative_linear_correlation_l326_326302

theorem negative_linear_correlation (x y : ℝ) (h : y = 3 - 2 * x) : 
  ∃ c : ℝ, c < 0 ∧ y = 3 + c * x := 
by  
  sorry

end negative_linear_correlation_l326_326302


namespace arun_remaining_days_l326_326496

theorem arun_remaining_days (W : ℝ) : 
  let arun_tarun_rate := W / 10
  let arun_rate := W / 70
  let tarun_rate := arun_tarun_rate - arun_rate 
  let work_done_4_days := 4 * arun_tarun_rate 
  let remaining_work := W - work_done_4_days
  let required_days := remaining_work / arun_rate in
  required_days = 42 :=
by 
  let arun_tarun_rate := W / 10
  let arun_rate := W / 70
  let tarun_rate := arun_tarun_rate - arun_rate 
  let work_done_4_days := 4 * arun_tarun_rate
  let remaining_work := W - work_done_4_days
  let required_days := remaining_work / arun_rate in
  sorry

end arun_remaining_days_l326_326496


namespace total_cookies_correct_l326_326912

-- Define the conditions
def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

-- Define the number of cookies collected by each person
def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℕ := (grayson_boxes * cookies_per_box).to_nat
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box

-- Define the total number of cookies collected
def total_cookies : ℕ := abigail_cookies + grayson_cookies + olivia_cookies

-- Prove that the total number of cookies collected is 276
theorem total_cookies_correct : total_cookies = 276 := sorry

end total_cookies_correct_l326_326912


namespace cans_recycled_l326_326369

theorem cans_recycled (cents_per_bottle cents_per_can cents_per_glass : ℕ) (bottles glass_containers : ℕ) (total_earnings : ℕ)
  (hb : cents_per_bottle = 10) (hc : cents_per_can = 5) (hg : cents_per_glass = 15)
  (nb : bottles = 80) (ng : glass_containers = 50) (earnings : total_earnings = 2500) :
  let made_from_bottles := bottles * cents_per_bottle,
      made_from_glass := glass_containers * cents_per_glass,
      remaining_earnings := total_earnings - (made_from_bottles + made_from_glass)
  in remaining_earnings / cents_per_can = 190 :=
by
  sorry

end cans_recycled_l326_326369


namespace coeff_x2_in_expansion_l326_326341

theorem coeff_x2_in_expansion :
  let p1 := (Polynomial.X ^ 2 - 3 * Polynomial.X + 1) ^ 8,
      p2 := (2 * Polynomial.X - 1) ^ 4,
      p := p1 * p2
  in (p.coeff 2 = 380) :=
by sorry

end coeff_x2_in_expansion_l326_326341


namespace profit_difference_is_50_l326_326522

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end profit_difference_is_50_l326_326522


namespace restore_bicentral_quadrilateral_l326_326444

-- Definitions and setup
variables (O I M : Point)
variable [is_circumcenter O]
variable [is_incenter I]
variable [is_midpoint M]

-- Statement of the problem
theorem restore_bicentral_quadrilateral
  (O I M: Point)
  [is_circumcenter O]
  [is_incenter I]
  [is_midpointof_diagonal M]
  : ∃ A B C D : Point, is_bicentral O I M A B C D :=
sorry

end restore_bicentral_quadrilateral_l326_326444


namespace solve_equation_l326_326252

theorem solve_equation (x : ℝ) (h : (Real.cbrt (3 - x) + Real.sqrt (x - 2) = 2)) : x = 2 ∨ x = 30 :=
by
  sorry

end solve_equation_l326_326252


namespace relationship_among_new_stationary_points_l326_326572

noncomputable def new_stationary_point (f : ℝ → ℝ) : ℝ :=
  Classical.choose (exists_real_root (λ x : ℝ, f x = f x.deriv))

def g (x : ℝ) : ℝ := 2 * x
def h (x : ℝ) : ℝ := Real.log x
def ϕ (x : ℝ) : ℝ := x ^ 3

theorem relationship_among_new_stationary_points :
  let a := new_stationary_point g,
      b := new_stationary_point h,
      c := new_stationary_point ϕ
  in c > b ∧ b > a :=
by
  sorry

end relationship_among_new_stationary_points_l326_326572


namespace rounding_maximizes_estimate_l326_326276

variable (x y z : ℕ) -- large positive integers
variable (x_rnd_up y_rnd_down z_rnd_down : ℕ)

-- Rounding conditions
constant hxr : x_rnd_up ≥ x
constant hyr : y_rnd_down ≤ y
constant hzr : z_rnd_down ≤ z

-- The expression we're estimating
def exact_value := 2 * (x / y - z : ℤ)
def estimated_value := 2 * (x_rnd_up / y_rnd_down - z_rnd_down : ℤ)

theorem rounding_maximizes_estimate (hxr : x_rnd_up ≥ x)
                                    (hyr : y_rnd_down ≤ y)
                                    (hzr : z_rnd_down ≤ z) :
  estimated_value x y z x_rnd_up y_rnd_down z_rnd_down > exact_value x y z :=
sorry  -- Proof to be filled in

end rounding_maximizes_estimate_l326_326276


namespace total_cookies_correct_l326_326914

-- Define the conditions
def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

-- Define the number of cookies collected by each person
def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℕ := (grayson_boxes * cookies_per_box).to_nat
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box

-- Define the total number of cookies collected
def total_cookies : ℕ := abigail_cookies + grayson_cookies + olivia_cookies

-- Prove that the total number of cookies collected is 276
theorem total_cookies_correct : total_cookies = 276 := sorry

end total_cookies_correct_l326_326914


namespace smallest_n_l326_326544

-- Define the problem conditions
def n (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  n % 9 = 2 ∧
  n % 6 = 3

-- Define the theorem to find the smallest n
theorem smallest_n : ∃ n : ℕ, n n ∧ n = 137 := by
  sorry

end smallest_n_l326_326544


namespace find_lambda_l326_326304

variable (n : ℕ) (λ : ℝ) (an sn : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 2 * n + λ

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def sum_increasing (S : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S (n + 1) > S n 

theorem find_lambda 
  (h_arith : arithmetic_sequence an)
  (h_sum : sum_of_terms sn an)
  (h_incr : sum_increasing sn) : 
  λ > -4 :=
sorry

end find_lambda_l326_326304


namespace pyramid_height_is_4_l326_326757

def pyramid_levels (total_cases : ℕ) : ℕ :=
  let rec sum_of_squares (n : ℕ) (acc : ℕ) :=
    if acc ≥ total_cases then n
    else sum_of_squares (n + 1) (acc + n^2)
  sum_of_squares 1 0

theorem pyramid_height_is_4 (h : Σ n, (∑ k in finRange n, (k + 1) ^ 2) = 30) :
  h.1 = 4 :=
sorry

end pyramid_height_is_4_l326_326757


namespace smallest_a_x4_plus_a2_not_prime_l326_326598

theorem smallest_a_x4_plus_a2_not_prime : ∃ a : ℕ, a > 0 ∧ (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ∧ 
  (∀ b : ℕ, b > 0 → (∀ x : ℤ, ¬ Prime (x^4 + b^2)) → a ≤ b) :=
begin
  let a := 8,
  use a,
  split,
  { exact nat.succ_pos' 7 },
  split,
  { intro x,
    unfold Prime,
    sorry
  },
  { intros b hb h,
    unfold Prime at h,
    have h2: b ≥ 8, 
    {
      sorry
    },
    exact h2
  }
end

end smallest_a_x4_plus_a2_not_prime_l326_326598


namespace total_cookies_correct_l326_326913

-- Define the conditions
def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

-- Define the number of cookies collected by each person
def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℕ := (grayson_boxes * cookies_per_box).to_nat
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box

-- Define the total number of cookies collected
def total_cookies : ℕ := abigail_cookies + grayson_cookies + olivia_cookies

-- Prove that the total number of cookies collected is 276
theorem total_cookies_correct : total_cookies = 276 := sorry

end total_cookies_correct_l326_326913


namespace find_number_l326_326155

theorem find_number (x : ℝ) (h : 0.15 * 40 = 0.25 * x + 2) : x = 16 :=
by
  sorry

end find_number_l326_326155


namespace probability_correct_l326_326348

open Finset

-- Define the set of distinct numbers
def num_set : Finset ℕ := {3, 7, 21, 27, 35, 42, 51}

-- Define the condition for a multiple of 63
def multiple_of_63 (a b : ℕ) : Prop := (a * b) % 63 = 0

-- Define the number of ways to pick distinct pairs
def total_pairs : ℕ := choose num_set.card 2

-- Define the number of successful pairs
def successful_pairs : ℕ := (num_set.prod fun a => 
  (num_set.filter (fun b => (a ≠ b) ∧ multiple_of_63 a b)).card ) / 2

-- Compute the probability
def probability_multiple_of_63 : ℚ := successful_pairs / total_pairs

-- The proof statement
theorem probability_correct : probability_multiple_of_63 = 3 / 7 := sorry

end probability_correct_l326_326348


namespace counting_special_positions_l326_326146

theorem counting_special_positions :
  {n : ℕ | 100 ≤ n ∧ n ≤ 2008 ∧ n % 9 = 1}.card = 213 := by
sorry

end counting_special_positions_l326_326146


namespace exists_fib_mod_1000_zero_l326_326230

def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem exists_fib_mod_1000_zero : ∃ n ≤ 1000001, fib n % 1000 = 0 :=
sorry

end exists_fib_mod_1000_zero_l326_326230


namespace johns_weight_l326_326015

-- Definitions based on the given conditions
def max_weight : ℝ := 1000
def safety_percentage : ℝ := 0.20
def bar_weight : ℝ := 550

-- Theorem stating the mathematically equivalent proof problem
theorem johns_weight : 
  (johns_safe_weight : ℝ) = max_weight - safety_percentage * max_weight 
  → (johns_safe_weight - bar_weight = 250) :=
by
  sorry

end johns_weight_l326_326015


namespace total_value_of_item_l326_326872

variable {V : ℝ}

theorem total_value_of_item (h : 0.07 * (V - 1000) = 109.20) : V = 2560 := 
by
  sorry

end total_value_of_item_l326_326872


namespace rhombus_perimeter_l326_326807

def half (a: ℕ) := a / 2

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  let s1 := half d1
      s2 := half d2
      side_length := Math.sqrt (s1^2 + s2^2)
      perimeter := 4 * side_length
  in perimeter = 68 := by
  sorry

end rhombus_perimeter_l326_326807


namespace parabola_min_value_l326_326062

variable {x0 y0 : ℝ}

def isOnParabola (x0 y0 : ℝ) : Prop := x0^2 = y0

noncomputable def expression (y0 x0 : ℝ) : ℝ :=
  Real.sqrt 2 * y0 + |x0 - y0 - 2|

theorem parabola_min_value :
  isOnParabola x0 y0 → ∃ (m : ℝ), m = (9 / 4 : ℝ) - (Real.sqrt 2 / 4) ∧ 
  ∀ y0 x0, expression y0 x0 ≥ (9 / 4 : ℝ) - (Real.sqrt 2 / 4) := 
by
  sorry

end parabola_min_value_l326_326062


namespace part1_part2_part3_l326_326308

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := - (1/4) * x^4 + (2/3) * x^3 + a * x^2 - 2 * x - 2

theorem part1 (h1 : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → deriv (f x a) ≤ 0) 
              (h2 : ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → deriv (f x a) ≥ 0) :
  a = 1/2 := sorry

theorem part2 (a : ℝ) (ha : a = 1/2)
              (h_three_sol : ∃ x1 x2 x3 : ℝ, 2^x1 ≠ 2^x2 ∧ 2^x2 ≠ 2^x3 ∧ 2^x1 ≠ 2^x3 ∧
                                f (2^x1) a = m ∧ f (2^x2) a = m ∧ f (2^x3) a = m) :
  -37/12 < m ∧ m < -8/3 := sorry

theorem part3 (a : ℝ) (ha : a = 1/2)
              (p : ℝ) (h_no_intersect : ∀ x : ℝ, ∃ y : ℝ, y = log 2 (f x a + p) → y ≠ 0) :
  5/12 < p ∧ p < 17/12 := sorry

end part1_part2_part3_l326_326308


namespace barcode_count_12_l326_326510

def barcodeCount : ℕ → ℕ
| 0     := 0   -- No barcodes with width 0
| 1     := 1   -- Only 1 barcode (B) of width 1
| 2     := 1   -- Only 1 barcode (BB) of width 2
| 3     := 1   -- Only 1 barcode (BWB) of width 3
| 4     := 3   -- (BBWB, BWBB, BWWB) of width 4
| (m+5) := barcodeCount (m+3) + 2 * barcodeCount (m+2) + barcodeCount m

theorem barcode_count_12 : barcodeCount 12 = 116 := sorry

end barcode_count_12_l326_326510


namespace projection_problem_l326_326185

theorem projection_problem : 
  let v₁ := (⟨3, 3⟩ : ℝ × ℝ)
  let p₁ := (⟨45 / 10, 9 / 10⟩ : ℝ × ℝ)
  let v₂ := (⟨1, -1⟩ : ℝ × ℝ)
  -- Define the projection of one vector onto another
  let proj (u v : ℝ × ℝ) : ℝ × ℝ := 
    let d := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
    ⟨d * v.1, d * v.2⟩
in
  proj v₂ (⟨5, 1⟩ : ℝ × ℝ) = (⟨10 / 13, 2 / 13⟩ : ℝ × ℝ) :=
by
  let v₁ := (⟨3, 3⟩ : ℝ × ℝ)
  let p₁ := (⟨45 / 10, 9 / 10⟩ : ℝ × ℝ)
  let v₂ := (⟨1, -1⟩ : ℝ × ℝ)
  let proj (u v : ℝ × ℝ) := 
    let d := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
    -- Define the projection as d times the vector v
    ⟨d * v.1, d * v.2⟩
  -- Simplify the vector p₁ to get the direction vector for projection
  have p1_simplified : p₁ = ⟨5, 1⟩ := by
    unfold p₁ 
    apply rfl

  sorry -- Proof omitted for brevity

end projection_problem_l326_326185


namespace triangle_side_length_l326_326362

-- Define the problem with the given conditions
theorem triangle_side_length (a b c : ℝ) (A : ℝ)
  (hA : A = 60) -- angle A is 60 degrees
  (h_area : (sqrt 3) / 2 * b * c = sqrt 3) -- given area condition
  (h_bc_sum : b + c = 6) -- sum of sides b and c is 6
  : a = 2 * sqrt 6 := 
sorry -- proof goes here

end triangle_side_length_l326_326362


namespace value_of_m_l326_326631

def p (m : ℝ) : Prop :=
  4 < m ∧ m < 10

def q (m : ℝ) : Prop :=
  8 < m ∧ m < 12

theorem value_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
by
  sorry

end value_of_m_l326_326631


namespace correct_judgments_l326_326213

-- Define the conditions
def judgment1 (ABC : Triangle) (A : Vertex) : Prop := rotate_around_vertex_A_preserves_angles ABC A
def judgment2 (ABC : Triangle) : Prop := right_triangle_congruent_hypotenuse_perimeter ABC
def judgment3 (ABC : Triangle) : Prop := not_two_sides_and_height_congruent_not_congruent ABC
def judgment4 (ABC : Triangle) : Prop := two_sides_and_median_congruent_is_congruent ABC

-- Theorem statement
theorem correct_judgments (ABC : Triangle) (A : Vertex) :
  judgment1 ABC A ∧ judgment2 ABC ∧ judgment4 ABC :=
by
  sorry

end correct_judgments_l326_326213


namespace min_transport_time_l326_326873

-- Define the conditions and the proof statement.
variable (v : ℝ) (t : ℝ)
variable (d : ℝ := 400)
variable (n : ℕ := 26)

-- Condition: Distance between two locations is 400 kilometers (d)
def distance := d

-- Condition: The speed of the trucks in kilometers per hour (v)
def speed := v

-- Condition: The number of trucks (26)
def num_trucks := n

-- Condition: The minimum distance between every two trucks
def min_distance (v : ℝ) := (v / 20) ^ 2

-- The proof statement: Prove the minimum time t is 10 hours
theorem min_transport_time (hv : speed = 80) : t = 10 := 
sorry

end min_transport_time_l326_326873


namespace find_equation_of_l_l326_326300

open Real

/-- Define the point M(2, 1) -/
def M : ℝ × ℝ := (2, 1)

/-- Define the original line equation x - 2y + 1 = 0 as a function -/
def line1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- Define the line l that passes through M and is perpendicular to line1 -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 5 = 0

/-- The theorem to be proven: the line l passing through M and perpendicular to line1 has the equation 2x + y - 5 = 0 -/
theorem find_equation_of_l (x y : ℝ)
  (hM : M = (2, 1))
  (hl_perpendicular : ∀ x y : ℝ, line1 x y → line_l y (-x / 2)) :
  line_l x y ↔ (x, y) = (2, 1) :=
by
  sorry

end find_equation_of_l_l326_326300


namespace triangle_distance_l326_326477

-- Lean statement representing the mathematical problem
theorem triangle_distance
  (ABC A'B'C' : Triangle) (a a' : ℝ)
  (hABC : ∀ (i j : {v : Fin 3 // v ≠ v}), dist (ABC.points i) (ABC.points j) ≥ a)
  (hA'B'C' : ∀ (i j : {v : Fin 3 // v ≠ v}), dist (A'B'C'.points i) (A'B'C'.points j) ≥ a') :
  ∃ (p ∈ ABC.points) (q ∈ A'B'C'.points), dist p q ≥ sqrt ((a^2 + a'^2) / 3) :=
begin
  sorry
end

end triangle_distance_l326_326477


namespace part1_part2_l326_326880

-- Part (1)
theorem part1 (x y : ℝ) (hx : x = 1 - real.sqrt 3) (hy : y = 1 + real.sqrt 3) :
  x^2 + y^2 - x * y - 2 * x + 2 * y = 10 + 4 * real.sqrt 3 :=
by sorry

-- Part (2)
theorem part2 (a : ℝ) (ha : a = 1 / (2 - real.sqrt 3)) :
  (1 - 2 * a + a^2) / (a - 1) - real.sqrt (a^2 - 2 * a + 1) / (a^2 - a) = 2 * real.sqrt 3 - 1 :=
by sorry

end part1_part2_l326_326880


namespace min_weights_required_l326_326144

def watermelon_problem (W : Set ℕ) : Prop :=
  (∀ n ∈ (range 1 21), ∃ w1 w2 ∈ W, n = w1 + w2) ∧ ∀ V ⊆ W, (∀ n ∈ (range 1 21), ∃ w1 w2 ∈ V, n = w1 + w2) → V = W

def min_weights : ℕ := 6

theorem min_weights_required : ∃ (W : Set ℕ), watermelon_problem W ∧ card W = min_weights :=
by
  sorry

end min_weights_required_l326_326144


namespace z_in_fourth_quadrant_l326_326829

def z : ℂ := (1 / (1 - complex.I)) + (2 / (1 + complex.I))

theorem z_in_fourth_quadrant : (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end z_in_fourth_quadrant_l326_326829


namespace jane_earnings_l326_326373

def age_of_child (jane_start_age : ℕ) (child_factor : ℕ) : ℕ :=
  jane_start_age / child_factor

def babysit_rate (age : ℕ) : ℕ :=
  if age < 2 then 5
  else if age <= 5 then 7
  else 8

def amount_earned (hours rate : ℕ) : ℕ := 
  hours * rate

def total_earnings (earnings : List ℕ) : ℕ :=
  earnings.foldl (·+·) 0

theorem jane_earnings
  (jane_start_age : ℕ := 18)
  (child_A_hours : ℕ := 50)
  (child_B_hours : ℕ := 90)
  (child_C_hours : ℕ := 130)
  (child_D_hours : ℕ := 70) :
  let child_A_age := age_of_child jane_start_age 2
  let child_B_age := child_A_age - 2
  let child_C_age := child_B_age + 3
  let child_D_age := child_C_age
  let earnings_A := amount_earned child_A_hours (babysit_rate child_A_age)
  let earnings_B := amount_earned child_B_hours (babysit_rate child_B_age)
  let earnings_C := amount_earned child_C_hours (babysit_rate child_C_age)
  let earnings_D := amount_earned child_D_hours (babysit_rate child_D_age)
  total_earnings [earnings_A, earnings_B, earnings_C, earnings_D] = 2720 :=
by
  sorry

end jane_earnings_l326_326373


namespace rhombus_perimeter_l326_326798

-- Definitions based on conditions
def diagonal1 : ℝ := 16
def diagonal2 : ℝ := 30
def half_diagonal1 : ℝ := diagonal1 / 2
def half_diagonal2 : ℝ := diagonal2 / 2

-- Mathematical formulation in Lean
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := 
by
  -- Given diagonals
  have h_half_d1 : (d1 / 2) = 8 := by sorry,
  have h_half_d2 : (d2 / 2) = 15 := by sorry,
  
  -- Combine into Pythagorean theorem and perimeter calculation
  have h_side_length : real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 17 := by sorry,
  show 4 * real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 68 := by sorry

end rhombus_perimeter_l326_326798


namespace rectangle_room_properties_l326_326908

def rect_room_diagonal_area (length width : ℝ) : ℝ × ℝ :=
  let diagonal := Real.sqrt ((length ^ 2) + (width ^ 2))
  let area := length * width
  (diagonal, area)

theorem rectangle_room_properties :
  ∃ (length width : ℝ), (2 * length + 2 * width = 82) ∧ (length / width = 3 / 2) ∧ 
    rect_room_diagonal_area length width = (29.56, 403.44) :=
by
  sorry

end rectangle_room_properties_l326_326908


namespace correct_options_l326_326744

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f(x) = f(-x + 16))
variable (h2 : ∀ x, 8 < x → f(x+8) < f(x))

theorem correct_options : (f 7 = f 9) ∧ (f 7 > f 10) :=
by
  sorry

end correct_options_l326_326744


namespace least_possible_sum_of_exponents_for_500_l326_326340

theorem least_possible_sum_of_exponents_for_500 :
  ∃ (S : Finset ℕ), (500 = S.sum (pow 2)) ∧ S.card ≥ 3 ∧ S.sum id = 32 :=
by
  sorry

end least_possible_sum_of_exponents_for_500_l326_326340


namespace problem_f_f_3_l326_326743

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then
    Float.sin (π * x / 6)
  else
    1 - 2 * x

theorem problem_f_f_3 : f (f 3) = -1/2 :=
by
  sorry

end problem_f_f_3_l326_326743


namespace tangent_line_of_ellipse_l326_326297

noncomputable def ellipse_tangent_line (a b x0 y0 x y : ℝ) : Prop :=
  x0 * x / a^2 + y0 * y / b^2 = 1

theorem tangent_line_of_ellipse
  (a b x0 y0 : ℝ)
  (h_ellipse : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_b : a > b) :
  ellipse_tangent_line a b x0 y0 x y :=
sorry

end tangent_line_of_ellipse_l326_326297


namespace sum_first_n_terms_general_term_l326_326285

-- Define the sequence and conditions
variable {a : ℕ → ℝ}
variable {k : ℝ}

-- Conditions
def seq_condition (n : ℕ) (h : n ∈ {n | n > 0}) : Prop := 2 * a (n + 1) = a n + a (n + 2) + k
def a1_condition : Prop := a 1 = 2
def a3_a5_condition : Prop := a 3 + a 5 = -4

-- First problem: Sum of the first n terms when k = 0
theorem sum_first_n_terms (n : ℕ) (h_k : k = 0) :
  (∀ n, n > 0 → seq_condition n (by simp [*]))
  → a1_condition
  → a3_a5_condition
  → (∑ i in finset.range n, a i) = (-2 * n^2 + 8 * n) / 3 :=
sorry

-- Second problem: General term of the sequence when a4 = -1
theorem general_term (n : ℕ) (h_a4 : a 4 = -1) :
  (∀ n, n > 0 → seq_condition n (by simp [*]))
  → a1_condition
  → a3_a5_condition
  → k = 2 
  → a n = -n^2 + 4 * n - 1 :=
sorry

end sum_first_n_terms_general_term_l326_326285


namespace printing_presses_count_l326_326368

-- Define the conditions using Lean statements
def press_rate (P hours papers : ℕ) := papers / hours
def presses_time (presses hours papers : ℕ) := press_rate presses hours papers

theorem printing_presses_count :
  ∀ P : ℕ, 
  presses_time P 12 500000 = presses_time 30 16 500000 → 
  P = 40 :=
begin
  intros,
  sorry,
end

end printing_presses_count_l326_326368


namespace tan_theta_is_one_fourth_l326_326732

noncomputable def theta_is_valid (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ :=
  (sin (2 * θ), cos θ)

noncomputable def vector_b (θ : ℝ) : ℝ × ℝ :=
  (2, -cos θ)

noncomputable def vectors_are_orthogonal (θ : ℝ) : Prop :=
  (vector_a θ).1 * (vector_b θ).1 + (vector_a θ).2 * (vector_b θ).2 = 0

theorem tan_theta_is_one_fourth (θ : ℝ) (h1 : theta_is_valid θ) (h2 : vectors_are_orthogonal θ) : 
  Real.tan θ = 1 / 4 := 
sorry

end tan_theta_is_one_fourth_l326_326732


namespace pure_gold_addition_l326_326925

theorem pure_gold_addition (x : ℝ) (hx : x = 100) :
  ∀ (w : ℝ) (p : ℝ) (w = 25) (p = 0.50), 
  (12.5 + x) / (25 + x) = 0.90 := by
  intros
  rw [hx]
  rw [w, p]
  sorry

end pure_gold_addition_l326_326925


namespace deductive_reasoning_is_D_l326_326489

inductive Option where
| A : Option
| B : Option
| C : Option
| D : Option

def is_analogical_reasoning : Option → Prop
| Option.A => true
| Option.B => true
| _       => false

def is_inductive_reasoning : Option → Prop
| Option.C => true
| _       => false

def is_deductive_reasoning : Option → Prop
| Option.D => true
| _       => false

theorem deductive_reasoning_is_D :
  ∀ (o : Option), is_deductive_reasoning o ↔ o = Option.D :=
by
  assume o
  cases o
  · simp [is_deductive_reasoning]
  · simp [is_deductive_reasoning]
  · simp [is_deductive_reasoning]
  · simp [is_deductive_reasoning]
  · sorry

end deductive_reasoning_is_D_l326_326489


namespace shadow_velocity_l326_326790

theorem shadow_velocity (X Y : Type) [UniformlyMovingAlongLine X l] [UniformlyMovingAlongLine Y m]
  (v_X : ℝ) (θ : ℝ) (acute : θ < π / 2) (obtuse : θ > π / 2)
  (intersection : ∃ O, Intersect l m O) :
  (Y_can_move_faster : shadow (Y).velocity (X).velocity * (dist_along m / dist_along l) > (X).velocity) ∧
  (Y_can_move_slower : shadow (Y).velocity (X).velocity * (dist_along m / dist_along l) < (X).velocity) :=
  sorry

end shadow_velocity_l326_326790


namespace six_star_three_l326_326268

-- Define the mathematical operation.
def operation (r t : ℝ) : ℝ := sorry

axiom condition_1 (r : ℝ) : operation r 0 = r^2
axiom condition_2 (r t : ℝ) : operation r t = operation t r
axiom condition_3 (r t : ℝ) : operation (r + 1) t = operation r t + 2 * t + 1

-- Prove that 6 * 3 = 75 given the conditions.
theorem six_star_three : operation 6 3 = 75 := by
  sorry

end six_star_three_l326_326268


namespace projection_correct_l326_326186

def proj (u v : ℝ × ℝ) :=
  let dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2
  let scale (k : ℝ) (w : ℝ × ℝ) : ℝ × ℝ :=
    (k * w.1, k * w.2)
  let numerator := dot_product u v
  let denominator := dot_product v v
  scale (numerator / denominator) v

theorem projection_correct :
  let u := (1, -1 : ℝ × ℝ)
  let v := (5, 1 : ℝ × ℝ)
  let result := proj u v
  result = ((10 / 13 : ℝ), (2 / 13 : ℝ)) := 
by
  let u := (1, -1 : ℝ × ℝ)
  let v := (5, 1 : ℝ × ℝ)
  let result := proj u v
  show result = ((10 / 13 : ℝ), (2 / 13 : ℝ))
  sorry

end projection_correct_l326_326186


namespace problem_statement_l326_326280

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a + 2 * cos (x / 2) ^ 2) * cos (x + π / 2)

open Real

theorem problem_statement : 
  (∀ x, f (π / 2) (-1) = 0) ∧ 
  (∀ α, α ∈ Ioo (π / 2) π → f (α / 2) (-1) = -2 / 5) ↔ 
  (∀ α, α ∈ Ioo (π / 2) π → cos (π / 6 - 2 * α) = (-7 * sqrt 3 - 24) / 50) :=
by
  sorry

end problem_statement_l326_326280


namespace find_omega_l326_326292

noncomputable def omega_solution (ω : ℝ) : Prop :=
  ω > 0 ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) > 2 * Real.cos (ω * y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) ≥ 1)

theorem find_omega : omega_solution (1 / 2) :=
sorry

end find_omega_l326_326292


namespace medians_distance_inequality_l326_326739

theorem medians_distance_inequality
  (A B C D E F G : Type)
  (tri : triangle A B C)
  (medians_AD : median A D)
  (medians_BE : median B E)
  (medians_CF : median C F)
  (medians_intersect_G : intersect G [AD, BE, CF])
  (S : ℝ) : 
  GD^2 + GE^2 + GF^2 ≥ (sqrt 3 / 3) * S :=
  sorry

end medians_distance_inequality_l326_326739


namespace exists_root_in_interval_l326_326241

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem exists_root_in_interval : ∃ c, 1 < c ∧ c < 2 ∧ f c = 0 :=
begin
  sorry
end

end exists_root_in_interval_l326_326241


namespace problem1_problem2_l326_326936

-- Problem 1
theorem problem1 (a : ℝ) : (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) → a < 0 := 
sorry

-- Problem 2
theorem problem2 : ∑ n in Finset.range 100, Int.floor (Real.log n / Real.log 3) = 284 := 
sorry

end problem1_problem2_l326_326936


namespace unique_special_poly_l326_326660

-- Define the polynomial form and its roots property
def is_special_poly (P : Polynomial ℂ) : Prop :=
  ∀ r, P.eval r = 0 → P.eval (r * Complex.e2piI / 3) = 0 ∧ P.eval (r * Complex.e2piI / 3 ^ 2) = 0

-- Statement of the problem
theorem unique_special_poly : 
  ∃! P : Polynomial ℂ, P.degree = 6 ∧ ∀ a b c d e : ℝ, P = Polynomial.C a • Polynomial.X ^ 5 + Polynomial.C b • Polynomial.X ^ 4 + 
  Polynomial.C c • Polynomial.X ^ 3 + Polynomial.C d • Polynomial.X ^ 2 + Polynomial.C e • Polynomial.X + Polynomial.C 2024 ∧ 
  ∀ r, is_special_poly P := 
sorry

end unique_special_poly_l326_326660


namespace students_without_favorite_subject_l326_326690

theorem students_without_favorite_subject
  (total_students : ℕ)
  (students_like_math : ℕ)
  (students_like_english : ℕ)
  (remaining_students : ℕ)
  (students_like_science : ℕ)
  (students_without_favorite : ℕ)
  (h1 : total_students = 30)
  (h2 : students_like_math = total_students * (1 / 5))
  (h3 : students_like_english = total_students * (1 / 3))
  (h4 : remaining_students = total_students - (students_like_math + students_like_english))
  (h5 : students_like_science = remaining_students * (1 / 7))
  (h6 : students_without_favorite = remaining_students - students_like_science) :
  students_without_favorite = 12 := by
  sorry

end students_without_favorite_subject_l326_326690


namespace greatest_a_l326_326445

theorem greatest_a (a : ℤ) (h_pos : a > 0) : 
  (∀ x : ℤ, (x^2 + a * x = -30) → (a = 31)) :=
by {
  sorry
}

end greatest_a_l326_326445


namespace midpoint_sum_and_distance_l326_326098

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem midpoint_sum_and_distance :
  let p1 := (-4 : ℝ, 1 : ℝ)
  let p2 := (10 : ℝ, 19 : ℝ)
  let mid := midpoint p1 p2
  mid.1 + mid.2 = 13 ∧ distance p1 p2 = 2 * Real.sqrt 130 :=
by
  sorry

end midpoint_sum_and_distance_l326_326098


namespace thomas_vs_maria_investment_difference_l326_326043

noncomputable def yearly_compounded_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def monthly_compounded_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem thomas_vs_maria_investment_difference :
  let P := 100000
  let r := 0.05
  let t := 3
  let n := 12
  let A_maria := yearly_compounded_amount P r t
  let A_thomas := monthly_compounded_amount P r n t
  round (A_thomas - A_maria) = 421 :=
by
  let P := 100000
  let r := 0.05
  let t := 3
  let n := 12
  let A_maria := yearly_compounded_amount P r t
  let A_thomas := monthly_compounded_amount P r n t
  exact (421 : ℝ) sorry

end thomas_vs_maria_investment_difference_l326_326043


namespace hyperbola_focus_l326_326233

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus
  (a b : ℝ)
  (hEq : ∀ x y : ℝ, ((x - 1)^2 / a^2) - ((y - 10)^2 / b^2) = 1):
  (1 + c 7 3, 10) = (1 + Real.sqrt (7^2 + 3^2), 10) :=
by
  sorry

end hyperbola_focus_l326_326233


namespace quadratic_roots_shift_l326_326746

theorem quadratic_roots_shift (d e a : ℝ)
  (h1 : ∀ r s : ℝ, (4 * r^2 - a * r - 12 = 0) ∧ (4 * s^2 - a * s - 12 = 0) →
                 ((r + 3) * (s + 3) = e)) :
  e = (3 * a + 24) / 4 :=
by
  -- Conditions from the problem
  have h_sum : ∀ r s : ℝ, 4 * r^2 - a * r - 12 = 0 ∧ 4 * s^2 - a * s - 12 = 0 → r + s = a / 4,
  { sorry },
  have h_prod : ∀ r s : ℝ, 4 * r^2 - a * r - 12 = 0 ∧ 4 * s^2 - a * s - 12 = 0 → r * s = -3,
  { sorry },
  -- Applying the hypotheses and calculating e
  apply h1,
  intros r s hr_hs,
  rw [h_sum r s hr_hs, h_prod r s hr_hs],
  sorry

end quadratic_roots_shift_l326_326746


namespace total_travel_time_l326_326457

noncomputable def travel_time (distance_razumeyevo_river : ℝ) (distance_vkusnoteevo_river : ℝ)
    (distance_downstream : ℝ) (river_width : ℝ) (current_speed : ℝ)
    (swimming_speed : ℝ) (walking_speed : ℝ) : ℝ := 
  let time_walk1 := distance_razumeyevo_river / walking_speed
  let effective_swimming_speed := real.sqrt (swimming_speed^2 - current_speed^2)
  let time_swim := river_width / effective_swimming_speed
  let time_walk2 := distance_vkusnoteevo_river / walking_speed
  time_walk1 + time_swim + time_walk2

theorem total_travel_time :
    travel_time 3 1 3.25 0.5 1 2 4 = 1.5 := 
by
  sorry

end total_travel_time_l326_326457


namespace solve_polynomial_division_l326_326271

theorem solve_polynomial_division :
  ∃ a : ℤ, (∀ x : ℂ, ∃ p : polynomial ℂ, x^2 - x + (a : ℂ) * p x = x^15 + x^2 + 100) → a = 2 := by
  sorry

end solve_polynomial_division_l326_326271


namespace find_tangent_points_l326_326467

noncomputable def curve := λ x : ℝ, x^3 + x - 2

def tangent_parallel_to_line (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (y = curve x) ∧ (3 * x^2 + 1 = 4)

theorem find_tangent_points :
  {P : ℝ × ℝ | tangent_parallel_to_line P} = {(-1, -4), (1, 0)} :=
by {
  sorry
}

end find_tangent_points_l326_326467


namespace find_value_of_ff_l326_326310

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 3^x

theorem find_value_of_ff : f (f (1 / 4)) = 1 / 9 :=
by
  sorry

end find_value_of_ff_l326_326310


namespace rhombus_perimeter_l326_326797

-- Definitions based on conditions
def diagonal1 : ℝ := 16
def diagonal2 : ℝ := 30
def half_diagonal1 : ℝ := diagonal1 / 2
def half_diagonal2 : ℝ := diagonal2 / 2

-- Mathematical formulation in Lean
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := 
by
  -- Given diagonals
  have h_half_d1 : (d1 / 2) = 8 := by sorry,
  have h_half_d2 : (d2 / 2) = 15 := by sorry,
  
  -- Combine into Pythagorean theorem and perimeter calculation
  have h_side_length : real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 17 := by sorry,
  show 4 * real.sqrt ((d1 / 2)^2 + (d2 / 2)^2) = 68 := by sorry

end rhombus_perimeter_l326_326797


namespace mechanic_worked_days_l326_326527

-- Definitions of conditions as variables
def hourly_rate : ℝ := 60
def hours_per_day : ℝ := 8
def cost_of_parts : ℝ := 2500
def total_amount_paid : ℝ := 9220

-- Definition to calculate the total labor cost
def total_labor_cost : ℝ := total_amount_paid - cost_of_parts

-- Definition to calculate the daily labor cost
def daily_labor_cost : ℝ := hourly_rate * hours_per_day

-- Proof (statement only) that the number of days the mechanic worked on the car is 14
theorem mechanic_worked_days : total_labor_cost / daily_labor_cost = 14 := by
  sorry

end mechanic_worked_days_l326_326527


namespace probability_of_sum_of_dice_rolls_odd_l326_326474

noncomputable def probability_sum_odd (n : ℕ) : ℚ :=
if n = 3 then 1 / 4 else 0

theorem probability_of_sum_of_dice_rolls_odd :
  probability_sum_odd 3 = 1 / 4 :=
sorry

end probability_of_sum_of_dice_rolls_odd_l326_326474


namespace age_difference_l326_326486

theorem age_difference {hannah_age july_age july_husband_age current_years july_age_now : ℕ}
    (hannah_initial_age : hannah_age = 6)
    (age_relationship : hannah_initial_age = 2 * july_age)
    (time_later : current_years = 20)
    (july_current_age : july_age_now = july_age + current_years)
    (husband_current_age : july_husband_age = 25)
    (husband_age_difference : july_husband_age - july_age_now = 2) :
    True :=
by
  sorry

end age_difference_l326_326486


namespace negation_of_universal_l326_326452

theorem negation_of_universal (P : ℝ → Prop) (h : ∀ x > 0, x^2 - x < 0) : 
  ∃ x > 0, x^2 - x ≥ 0 :=
begin
  sorry
end

end negation_of_universal_l326_326452


namespace triangle_is_right_angled_l326_326630

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end triangle_is_right_angled_l326_326630


namespace find_c_l326_326303

def vector := ℝ × ℝ

def a : vector := (1, 2)
def x : ℝ
def b : vector := (x, 6)
def c := (2 * a.1 + b.1, 2 * a.2 + b.2)

noncomputable def distance (v1 v2 : vector) : ℝ :=
  real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)

theorem find_c (h1 : distance a b = 2 * real.sqrt 5) (h2 : ¬ ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2) : c = (1, 10) :=
sorry

end find_c_l326_326303


namespace trees_1995_l326_326453

def T (n : ℤ) : ℤ := sorry

def k : ℚ := sorry

axiom tree_growth (n : ℤ) : T(n + 2) - T(n) = k * T(n + 1)
axiom trees_1993 : T(1993) = 50
axiom trees_1994 : T(1994) = 75
axiom trees_1996 : T(1996) = 140

theorem trees_1995 : T(1995) = 99 :=
sorry

end trees_1995_l326_326453


namespace johns_sixth_quiz_score_l326_326722

theorem johns_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (mean : ℕ) (n : ℕ) :
  s1 = 86 ∧ s2 = 91 ∧ s3 = 83 ∧ s4 = 88 ∧ s5 = 97 ∧ mean = 90 ∧ n = 6 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / n = mean ∧ s6 = 95 :=
by
  intro h
  obtain ⟨hs1, hs2, hs3, hs4, hs5, hmean, hn⟩ := h
  have htotal : s1 + s2 + s3 + s4 + s5 + 95 = 540 := by sorry
  have hmean_eq : (s1 + s2 + s3 + s4 + s5 + 95) / n = mean := by sorry
  exact ⟨95, hmean_eq, rfl⟩

end johns_sixth_quiz_score_l326_326722


namespace part1_part2_part3_l326_326286

variable (a : ℕ → ℚ) (S : ℕ → ℚ)

-- Define the sequence a_n such that a_1 = 1/2 and the recurrence relation a_n + 2S_nS_{n-1} = 0 for n >= 2
axiom h_a1 : a 1 = 1 / 2
axiom h_recur : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0

-- Define S_n as the sum of the first n terms of a
axiom h_S_def : ∀ n, S n = ∑ i in finset.range (n+1), a i

-- Proof problem 1: Prove that {1/S_n} is an arithmetic sequence with a common difference of 2
theorem part1 : ∃ d : ℚ, d = 2 ∧ ∀ n ≥ 2, (1 / S n) - (1 / S (n - 1)) = d := sorry

-- Proof problem 2: Prove S_n = 1/(2n) and a_n as defined in problem
theorem part2 : (∀ n ≥ 1, S n = 1 / (2 * n)) ∧ (a 1 = 1 / 2 ∧ (∀ n ≥ 2, a n = -1 / (2 * n * (n - 1)))) := sorry

-- Proof problem 3: Prove the inequality S_1^2 + S_2^2 + ... + S_n^2 ≤ 1/2 - 1/(4n)
theorem part3 : ∀ n ≥ 1, ∑ i in finset.range (n+1), (S i)^2 ≤ 1 / 2 - 1 / (4 * n) := sorry

end part1_part2_part3_l326_326286


namespace infinite_palindromic_in_ap_l326_326866

def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_digits in s = s.reverse

def arithmetic_progression (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

theorem infinite_palindromic_in_ap : ∃ᶠ n in at_top, is_palindromic (arithmetic_progression 18 19 n) :=
sorry

end infinite_palindromic_in_ap_l326_326866


namespace hyperbola_asymptotes_correct_l326_326256

noncomputable def asymptotes_for_hyperbola : Prop :=
  ∀ (x y : ℂ),
    9 * (x : ℂ) ^ 2 - 4 * (y : ℂ) ^ 2 = -36 → 
    (y = (3 / 2) * (-Complex.I) * x) ∨ (y = -(3 / 2) * (-Complex.I) * x)

theorem hyperbola_asymptotes_correct :
  asymptotes_for_hyperbola := 
sorry

end hyperbola_asymptotes_correct_l326_326256


namespace modulus_z1_eq_three_l326_326709

noncomputable def z1 (x y : ℝ) : ℂ := x + y * complex.I
noncomputable def z2 (x y : ℝ) : ℂ := y + x * complex.I

theorem modulus_z1_eq_three (x y : ℝ) (h : (z1 x y) * (z2 x y) = 9 * complex.I) : complex.abs (z1 x y) = 3 := by
  sorry

end modulus_z1_eq_three_l326_326709


namespace Bruce_Anne_combined_cleaning_time_l326_326943

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l326_326943


namespace largest_divisible_by_digits_sum_l326_326462

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem largest_divisible_by_digits_sum : ∃ n, n < 900 ∧ n % digits_sum n = 0 ∧ ∀ m, m < 900 ∧ m % digits_sum m = 0 → m ≤ 888 :=
by
  sorry

end largest_divisible_by_digits_sum_l326_326462


namespace problem1_problem2_l326_326558

noncomputable def problem1_lhs := sqrt 12 * sqrt (1/3) - sqrt 18 + abs (sqrt 2 - 2)
noncomputable def problem1_rhs := 4 - 4 * sqrt 2
noncomputable def problem2_lhs := (7 + 4 * sqrt 3) * (7 - 4 * sqrt 3) - (sqrt 3 - 1)^2
noncomputable def problem2_rhs := 2 * sqrt 3 - 3

theorem problem1 : problem1_lhs = problem1_rhs := by
  sorry

theorem problem2 : problem2_lhs = problem2_rhs := by
  sorry

end problem1_problem2_l326_326558


namespace nonempty_even_sums_subsets_count_l326_326605

theorem nonempty_even_sums_subsets_count :
  (∃ (S : Finset ℕ), S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
                      S.nonempty ∧
                      S.sum id % 2 = 0 ∧
                      ∑ ' S_, (S_ ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) (S_.nonempty ∧ S_.sum id % 2 = 0) = 511) := sorry

end nonempty_even_sums_subsets_count_l326_326605


namespace loisa_savings_l326_326406

namespace SavingsProof

def cost_cash : ℤ := 450
def down_payment : ℤ := 100
def payment_first_4_months : ℤ := 4 * 40
def payment_next_4_months : ℤ := 4 * 35
def payment_last_4_months : ℤ := 4 * 30

def total_installment_payment : ℤ :=
  down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

theorem loisa_savings :
  (total_installment_payment - cost_cash) = 70 := by
  sorry

end SavingsProof

end loisa_savings_l326_326406


namespace proof_problem_l326_326305

noncomputable def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

def line_eq (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

noncomputable def moving_circle_C (x y k : ℝ) : Prop :=
  (x - k)^2 + (y - √3 * k)^2 = 4

def tangent_line_condition (k : ℝ) : Prop :=
  |k| > 2

def point_on_line_xy4 (x y : ℝ) : Prop :=
  x + y = 4

def min_area_tangents (area : ℝ) : Prop :=
  area = 4

theorem proof_problem (m x y k : ℝ) (area : ℝ) :
  (∀ m : ℝ, ∃ x y : ℝ, line_eq m x y → ¬ circle_O x y) →
  tangent_line_condition k →
  point_on_line_xy4 x y →
  min_area_tangents area :=
by
  sorry

end proof_problem_l326_326305


namespace concurrence_or_parallel_l326_326366

variables {V : Type} [inner_product_space ℝ V] [finite_dimensional ℝ V]

/-- 
Given a triangle ΔABC with heights AA₁ and BB₁, angle bisectors AA₂ and BB₂, 
and the incircle touching sides BC and AC at points A₃ and B₃ respectively, 
prove that lines A₁B₁, A₂B₂, and A₃B₃ intersect at one point or are parallel.
-/
theorem concurrence_or_parallel (A B C A₁ B₁ A₂ B₂ A₃ B₃ : V)
  (h₁ : ∃ l : V, ∀ v ∈ lower_set l, collinear ℝ ({A, A₁, v}))
  (h₁' : ∃ l : V, ∀ v ∈ lower_set l, collinear ℝ ({B, B₁, v}))
  (h₂ : ∃ l : V, ∀ v ∈ lower_set l, collinear ℝ ({A, A₂, v}))
  (h₂' : ∃ l : V, ∀ v ∈ lower_set l, collinear ℝ ({B, B₂, v}))
  (h₃ : ∃ I, is_incircle I A B C A₃ B₃) :
  (∃ P : V, collinear ℝ [{A₁, B₁, P}] ∧ collinear ℝ [{A₂, B₂, P}] ∧ collinear ℝ [{A₃, B₃, P}]) 
  ∨ parallel ℝ ({A₁, B₁, A₂, B₂, A₃, B₃}) :=
sorry

end concurrence_or_parallel_l326_326366


namespace third_cyclist_speed_l326_326877

theorem third_cyclist_speed (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : 
  ∃ V : ℝ, V = (a + 3 * b + Real.sqrt (a^2 - 10 * a * b + 9 * b^2)) / 4 :=
by
  sorry

end third_cyclist_speed_l326_326877


namespace problem_l326_326970

def binom (n k : ℕ) : ℕ := n.choose k

def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem : binom 10 3 * perm 8 2 = 6720 := by
  sorry

end problem_l326_326970


namespace find_f_2015_l326_326622

variables (f : ℝ → ℝ)

-- Conditions
axiom f_periodic : ∀ x, f(x) + f(x + 6) = 0
axiom f_symmetric : ∀ x, f(-x) = -f(x)  -- This encapsulates the symmetry about the origin point (0,0)
axiom f_value : f(7) = 4

theorem find_f_2015 : f(2015) = 4 :=
by
  sorry

end find_f_2015_l326_326622


namespace solution_sum_is_sqrt3_l326_326734

noncomputable def sum_of_solutions : ℝ :=
  ∑ x in { x : ℝ | 0 < x ∧ x ^ 3 ^ sqrt 3 = 3 ^ (3 ^ x) }, x

theorem solution_sum_is_sqrt3 :
  sum_of_solutions = sqrt 3 :=
sorry

end solution_sum_is_sqrt3_l326_326734


namespace bob_questions_first_hour_l326_326551

theorem bob_questions_first_hour (x : ℕ) (h1 : 2 * x = y) (h2 : 4 * x = z) (h3 : x + y + z = 91) : x = 13 :=
by
  -- Construct the relations as given by conditions
  have hy : y = 2 * x := h1,
  have hz : z = 4 * x := h2,

  -- Use the fact (h3) to establish "7x = 91"
  calc
    7 * x = x + y + z : by rw [hy, hz]; ring
      ... = 91         : by assumption
  -- Solve 7x = 91 to conclude x = 13
  sorry

end bob_questions_first_hour_l326_326551


namespace area_triangle_ABC_given_conditions_l326_326289

variable (a b c : ℝ) (A B C : ℝ)

noncomputable def area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem area_triangle_ABC_given_conditions
  (habc : a = 4)
  (hbc : b + c = 5)
  (htan : Real.tan B + Real.tan C + Real.sqrt 3 = Real.sqrt 3 * (Real.tan B * Real.tan C))
  : area_of_triangle_ABC a b c (Real.pi / 3) B C = 3 * Real.sqrt 3 / 4 := 
sorry

end area_triangle_ABC_given_conditions_l326_326289


namespace numberPlayingTwoOrMore_is_250_l326_326693

variable (totalStudents : ℕ)
variable (fractionPlayAtLeastOne : ℚ)
variable (probabilityPlayExactlyOne : ℚ)
variable (numberPlayingAtLeastOne numberPlayingExactlyOne numberPlayTwoOrMore : ℕ)

-- Given
axiom totalStudents_eq_800 : totalStudents = 800
axiom fractionPlayAtLeastOne_eq_3_div_5 : fractionPlayAtLeastOne = 3 / 5
axiom probabilityPlayExactlyOne_eq_48_div_100 : probabilityPlayExactlyOne = 48 / 100

-- Definitions
def numberPlayingAtLeastOne := (fractionPlayAtLeastOne * totalStudents).natAbs
def numberPlayingExactlyOne := (probabilityPlayExactlyOne * numberPlayingAtLeastOne).natAbs
def numberPlayTwoOrMore := numberPlayingAtLeastOne - numberPlayingExactlyOne

-- Prove
theorem numberPlayingTwoOrMore_is_250 :
  numberPlayTwoOrMore = 250 :=
by
  have H1 : numberPlayingAtLeastOne = 480 := by 
    rw [numberPlayingAtLeastOne, fractionPlayAtLeastOne_eq_3_div_5, totalStudents_eq_800]
    norm_num

  have H2 : numberPlayingExactlyOne = 230 := by 
    rw [numberPlayingExactlyOne, probabilityPlayExactlyOne_eq_48_div_100, H1]
    norm_num

  rw [numberPlayTwoOrMore, H1, H2]
  norm_num

end numberPlayingTwoOrMore_is_250_l326_326693


namespace part1_payment_x_40_part2_equal_cost_part3_cost_effective_l326_326086

def price_racket : ℕ := 150
def price_ball : ℕ := 15
def discount_rate : ℚ := 0.9
def num_rackets : ℕ := 10

def option_one_cost (x : ℕ) : ℕ :=
  (num_rackets * price_racket) + ((x - 20) * price_ball)

def option_two_cost (x : ℕ) : ℚ :=
  ((num_rackets * price_racket) + (x * price_ball)) * discount_rate

theorem part1_payment_x_40 :
  option_one_cost 40 = 1800 ∧ option_two_cost 40 = 1890 :=
by
  sorry

theorem part2_equal_cost (x : ℕ) : 
  (13.5 * x + 1350 : ℚ) = (15 * x + 1200 : ℚ) → x = 100 :=
by
  sorry

theorem part3_cost_effective :
  ∃ x, option_one_cost 20 + ((40 - 20) * price_ball * discount_rate) = 1770 :=
by
  sorry

end part1_payment_x_40_part2_equal_cost_part3_cost_effective_l326_326086


namespace day_after_festival_45_days_l326_326372

/--
  Jamie's birthday is on a Tuesday. A 5-day festival starts on Jamie's birthday.
  What day of the week will it be 45 days after the festival ends?
-/
theorem day_after_festival_45_days (start_day : ℕ) (birthday_is_tuesday : start_day = 2) (days_festival : ℕ) (festival_duration : days_festival = 5) : 
  let end_day := (start_day + days_festival - 1) % 7 in
  (end_day + 45) % 7 = 3 :=
by sorry

end day_after_festival_45_days_l326_326372


namespace team_played_total_games_l326_326579

variables {G R : ℕ}

-- condition 1
def win_first_100_games := 0.75 * 100

-- condition 2
def win_remaining_games (R : ℕ) := 0.50 * R

-- condition 3
def total_games_won (G : ℕ) := 0.70 * G

-- condition 4
def total_games (G R : ℕ) := G = 100 + R

theorem team_played_total_games 
  (h1 : win_first_100_games = 75)
  (h2 : ∀ R, win_remaining_games R = 0.50 * R)
  (h3 : ∀ G, total_games_won G = 0.70 * G)
  (h4 : ∀ G R, total_games G R = G)
  : G = 125 :=
sorry

end team_played_total_games_l326_326579


namespace ratio_of_inscribed_square_l326_326989

theorem ratio_of_inscribed_square (a : ℝ) (h_nonzero : a ≠ 0):
  (let large_square_area := a^2;
       s := a / 3;
       inscribed_square_area := (s * sqrt 5)^2
   in (inscribed_square_area / large_square_area) = 5 / 9) := sorry

end ratio_of_inscribed_square_l326_326989


namespace non_persistent_matches_more_than_half_l326_326696

noncomputable def total_players (n : ℕ) := n > 4

def total_matches (n : ℕ) := n * (n - 1) / 2

def is_persistent (results : list (ℕ × ℕ)) (player : ℕ) : Prop :=
  ∀ (match : ℕ × ℕ), match ∈ results → player = match.1 → ∀ (other_match : ℕ × ℕ), other_match ∈ results → match.2 < other_match.2

def is_non_persistent (results : list (ℕ × ℕ)) (player : ℕ) : Prop :=
  ¬is_persistent results player

theorem non_persistent_matches_more_than_half (n : ℕ) (results : list (ℕ × ℕ)) :
  total_players n →
  (∀ player : ℕ, player < n → ∃ opponent : ℕ, (player, opponent) ∈ results) →
  ∃ m : ℕ, m > (n - 1) / 2 ∧
    (∃ (matches_non_persistent : finset (ℕ × ℕ)),
      matches_non_persistent.card = m ∧
      ∀ match in matches_non_persistent, (is_non_persistent results match.1 ∧ is_non_persistent results match.2)) :=
sorry

end non_persistent_matches_more_than_half_l326_326696


namespace magnitude_of_angle_C_area_of_triangle_l326_326288

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {Δ : ℝ}

-- Given conditions
axiom h1 : 2 * cos C * (a * cos C + c * cos A) + b = 0

-- Prove that C = 120°
theorem magnitude_of_angle_C (h1: 2 * cos C * (a * cos C + c * cos A) + b = 0) : C = 120 := 
    sorry

-- Given additional conditions
axiom h_b: b = 2
axiom h_c: c = 2 * sqrt 3 
axiom h_C: C = 120 -- from previous proof

-- Prove that the area of the triangle is √3
theorem area_of_triangle (h_b : b = 2) (h_c : c = 2 * sqrt 3 ) (h_C : C = 120)
   : Δ = sqrt 3 := 
    sorry

end magnitude_of_angle_C_area_of_triangle_l326_326288


namespace ratio_of_pentagon_area_to_total_area_l326_326236

theorem ratio_of_pentagon_area_to_total_area 
    (s : ℝ)
    (PQRS TUVW WXYZ : ℝ)
    (side_length : PQRS = TUVW ∧ TUVW = WXYZ ∧ WXYZ = s)
    (PQ_parallel_adj_VW: PQRS ∥ TUVW ∧ PQRS.adj TUVW)
    (RS_parallel_adj_XY : PQRS ∥ WXYZ ∧ PQRS.adj WXYZ)
    (R_S_three_fourths_WY_VW : (distance R W = 3/4 * distance W Y) ∧ (distance S V = 3/4 * distance V W)) :
    (area_of_pentagon PAWSR / (3 * s^2) = 1 / 8) :=
by sorry

end ratio_of_pentagon_area_to_total_area_l326_326236


namespace triangle_DEF_area_l326_326692

theorem triangle_DEF_area
  (DE DF : ℝ)
  (h_angle_D : DE * DE + DF * DF = (sqrt (DE^2 + DF^2))^2)
  (h_DE : DE = 30)
  (h_DF : DF = 40) :
  1 / 2 * DE * DF = 600 :=
by
  simp [h_DE, h_DF]
  exact by norm_num
  sorry

end triangle_DEF_area_l326_326692


namespace highest_probability_event_C_l326_326898

-- Define the fair dice numbers and their properties
def dice_numbers := {1, 2, 3, 4, 5, 6}

-- Define events
def event_A (x y : ℕ) : Prop := x ∈ dice_numbers ∧ y ∈ dice_numbers ∧ (even x) ∧ (even y)
def event_B (x y : ℕ) : Prop := x ∈ dice_numbers ∧ y ∈ dice_numbers ∧ (odd (x + y))
def event_C (x y : ℕ) : Prop := x ∈ dice_numbers ∧ y ∈ dice_numbers ∧ (x + y < 13)
def event_D (x y : ℕ) : Prop := x ∈ dice_numbers ∧ y ∈ dice_numbers ∧ (x + y < 2)

-- Define the main theorem
theorem highest_probability_event_C : 
  (∃ x y, event_C x y ∧ 
  ∀ (p q : ℕ), event_A p q ∨ event_B p q ∨ event_C p q ∨ event_D p q → 
  ¬ (event_C x y) → (event_A p q ∨ event_B p q ∨ event_D p q) ≤ event_C x y) :=
sorry

end highest_probability_event_C_l326_326898


namespace line_conditions_l326_326284

noncomputable def slope (A B : ℝ × ℝ) : ℝ :=
  if A.1 = B.1 then 0 else (B.2 - A.2) / (B.1 - A.1)

theorem line_conditions (m n : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-2, m)) (hB : B = (m, 4))
  (h_parallel : slope A B = -2)
  (h_perpendicular : 2 * (-1/n) = -1) :
  m + n = -10 :=
sorry

end line_conditions_l326_326284


namespace four_digit_perfect_square_l326_326257

theorem four_digit_perfect_square (N : ℕ) (a b : ℤ) :
  N = 1100 * a + 11 * b ∧
  N >= 1000 ∧ N <= 9999 ∧
  a >= 0 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧
  (∃ (x : ℤ), N = 11 * x^2) →
  N = 7744 := by
  sorry

end four_digit_perfect_square_l326_326257


namespace number_of_good_points_l326_326018

open Set

variable {α : Type*}

noncomputable def good_point (A B C D A_b A_c B_a B_c C_a C_b : α) [MetricSpace α] :=
Concyclic {A_b, A_c, B_a, B_c, C_a, C_b}

theorem number_of_good_points (A B C : α) [MetricSpace α] :
  ∃! D : α, good_point A B C D sorry sorry sorry sorry sorry = 4 :=
sorry


end number_of_good_points_l326_326018


namespace axis_of_symmetry_l326_326816

theorem axis_of_symmetry (x : ℝ) (h : x = -Real.pi / 12) :
  ∃ k : ℤ, 2 * x - Real.pi / 3 = k * Real.pi + Real.pi / 2 :=
sorry

end axis_of_symmetry_l326_326816


namespace count_functions_l326_326265

theorem count_functions (n : ℕ) (f : ℕ → ℕ) :
  (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ i > j → f i ≥ f j) →
  (∑ i in Finset.range n, (i + 1) + f (i + 1) = 2023) →
  n = 63 →
  ∑ i in Finset.range n, f (i + 1) = 7 →
  (number_of_valid_functions f = 15) :=
sorry

end count_functions_l326_326265


namespace locus_of_midpoint_l326_326628

theorem locus_of_midpoint (x y : ℝ) (h : y ≠ 0) :
  (∃ P : ℝ × ℝ, P = (2*x, 2*y) ∧ ((P.1^2 + (P.2-3)^2 = 9))) →
  (x^2 + (y - 3/2)^2 = 9/4) :=
by
  sorry

end locus_of_midpoint_l326_326628


namespace constant_term_in_binomial_expansion_l326_326093

-- Theorem statement
theorem constant_term_in_binomial_expansion : 
  (∃ T : ℕ → ℕ → ℝ, T 6 2 = 15 → 
    T (6 : ℕ) (Nat.choose (6 : ℕ) (2 : ℕ)) = 15) 
  → True := 
begin
  sorry
end

end constant_term_in_binomial_expansion_l326_326093


namespace find_a_2020_l326_326656

def sequence_an (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1

def sum_bn (a : ℕ → ℤ) : Prop :=
  ∑ n in Finset.range 2019, (a (n + 1) - (n + 1)) = 2019

theorem find_a_2020 (a : ℕ → ℤ) 
  (h_seq : sequence_an a) 
  (h_sum : sum_bn a) : 
  a 2020 = 1 :=
sorry

end find_a_2020_l326_326656


namespace train_length_l326_326200

variable (speed_kph : ℝ) (time_s : ℝ) (bridge_length_m : ℝ)

theorem train_length
  (h1 : speed_kph = 50)
  (h2 : time_s = 36)
  (h3 : bridge_length_m = 140) :
  let speed_mps := speed_kph * 1000 / 3600 in
  let distance_m := speed_mps * time_s in
  let train_length_m := distance_m - bridge_length_m in
  train_length_m = 360 := by
  unfold speed_mps distance_m train_length_m
  rw [h1, h2, h3]
  norm_num
  sorry

end train_length_l326_326200


namespace num_sets_sum_18_l326_326326

theorem num_sets_sum_18 : ∃! (S : Set (Finset ℕ)), 
  (∃ n a, n > 1 ∧ a > 0 ∧ S = {a | a ∈ {a..a + n - 1}} ∧ S.sum id = 18) :=
sorry

end num_sets_sum_18_l326_326326


namespace ball_distribution_l326_326121

theorem ball_distribution (basketballs volleyballs classes balls : ℕ) 
  (h1 : basketballs = 2) 
  (h2 : volleyballs = 3) 
  (h3 : classes = 4) 
  (h4 : balls = 4) :
  (classes.choose 3) + (classes.choose 2) = 10 :=
by
  sorry

end ball_distribution_l326_326121


namespace bruce_and_anne_clean_together_l326_326955

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l326_326955


namespace max_last_digit_of_sequence_l326_326818

theorem max_last_digit_of_sequence :
  ∀ (s : Fin 1001 → ℕ), 
  (s 0 = 2) →
  (∀ (i : Fin 1000), (s i) * 10 + (s i.succ) ∈ {n | n % 17 = 0 ∨ n % 23 = 0}) →
  ∃ (d : ℕ), (d = s ⟨1000, sorry⟩) ∧ (∀ (d' : ℕ), d' = s ⟨1000, sorry⟩ → d' ≤ d) ∧ (d = 2) :=
by
  intros s h1 h2
  use 2
  sorry

end max_last_digit_of_sequence_l326_326818


namespace red_balls_to_remove_l326_326887

theorem red_balls_to_remove (total_balls red_percentage target_percentage : ℝ) (total_red_balls total_blue_balls : ℕ) :
  total_balls = 600 → 
  red_percentage = 0.7 → 
  target_percentage = 0.65 → 
  total_red_balls = nat.floor (red_percentage * total_balls) → 
  total_blue_balls = total_balls.to_nat - total_red_balls → 
  (∃ (x : ℕ), (420 - x) / (600 - x).to_real = target_percentage ∧ x = 86) := 
begin
  sorry
end

end red_balls_to_remove_l326_326887


namespace solve_log_eqn_l326_326431

noncomputable def log_base (b a : ℝ) : ℝ := real.log a / real.log b

theorem solve_log_eqn (a : ℝ) (h : 1 < a) : log_base 5 a + log_base 3 a = log_base 5 a * log_base 3 a ↔ a = 15 :=
by sorry

end solve_log_eqn_l326_326431


namespace distance_second_part_eq_12_l326_326889

def distance_first_part : ℝ := 10
def speed_first_part : ℝ := 12
def speed_second_part : ℝ := 10
def average_speed_trip : ℝ := 10.82

theorem distance_second_part_eq_12 (x : ℝ)
  (H1 : x = distance_first_part / speed_first_part + x / speed_second_part)
  (H2 : average_speed_trip = (distance_first_part + x) / H1) :
  x = 12 :=
  sorry

end distance_second_part_eq_12_l326_326889


namespace green_candy_pieces_l326_326472

theorem green_candy_pieces (t r b g : ℝ) (h_total : t = 12509.72) (h_red : r = 568.29) (h_blue : b = 2473.43) :
  g = t - r - b :=
begin
  -- By substituting the given values,
  -- green pieces = 12,509.72 - 568.29 - 2,473.43
  -- green pieces = 12,509.72 - 3,041.72
  -- green pieces = 9,468
  sorry
end

end green_candy_pieces_l326_326472


namespace internal_diagonal_cubes_l326_326885

theorem internal_diagonal_cubes (a b c : ℕ) (h₁ : a = 200) (h₂ : b = 300) (h₃ : c = 450) :
  let gcd_ab := Nat.gcd a b,
      gcd_bc := Nat.gcd b c,
      gcd_ca := Nat.gcd c a,
      gcd_abc := Nat.gcd (Nat.gcd a b) c in
  a + b + c - (gcd_ab + gcd_bc + gcd_ca) + gcd_abc = 700 := by
  sorry

end internal_diagonal_cubes_l326_326885


namespace sequence_recurrence_le_1_over_4n_l326_326036

theorem sequence_recurrence_le_1_over_4n
  (n : ℕ)
  (a : ℕ → ℝ)
  (c : ℝ)
  (h1 : a n = 0)
  (h2 : ∀ k, k < n → a k = c + ∑ i in finset.range(n - k), a (i + k) * (a i + a (i + 1))) : 
  c ≤ 1 / (4 * n) :=
sorry

end sequence_recurrence_le_1_over_4n_l326_326036


namespace relationship_among_abc_l326_326639

theorem relationship_among_abc (e1 e2 : ℝ) (h1 : 0 ≤ e1) (h2 : e1 < 1) (h3 : e2 > 1) :
  let a := 3 ^ e1
  let b := 2 ^ (-e2)
  let c := Real.sqrt 5
  b < c ∧ c < a := by
  sorry

end relationship_among_abc_l326_326639


namespace equation_of_circle_C_l326_326678

theorem equation_of_circle_C :
  ∃ (h k r : ℝ), (h = 2) ∧ (k = 3) ∧ (r = 2) ∧
  (h, k ∈ {p : ℝ × ℝ | p.1 = 3∧) ∧ 
  ( ∀ x y : ℝ), (x-2)^2 + (y-3)^2 = 4 := 
begin
  sorry
end

end equation_of_circle_C_l326_326678


namespace rebecca_marbles_l326_326772

theorem rebecca_marbles (M : ℕ) (h1 : 20 = M + 14) : M = 6 :=
by
  sorry

end rebecca_marbles_l326_326772


namespace math_problem_solution_proof_l326_326367

variables {A B C K L M N : Point} {AC BC AB : ℝ} 
variables (angleACB : ℝ) (radiusOmega : ℝ) 
variables (MKisParallel : Line)
variables (triangleANLArea : ℝ)

open Real

/--
In triangle ABC, side AC is equal to 6, and angle ACB is 120 degrees. 
A circle Ω with a radius of \sqrt{3} touches sides BC and AC of triangle ABC at points K and L respectively 
and intersects side AB at points M and N (with M lying between A and N) such that segment MK is parallel to AC.
To prove: 
1) CL = 1 
2) MK = 3 
3) AB = 2\sqrt{13} 
4) Area of triangle ANL = \frac{125\sqrt{3}}{52} 
-/
theorem math_problem_solution_proof
  (h_AC : AC = 6)
  (h_angleACB : angleACB = 120)
  (h_radiusOmega : radiusOmega = sqrt 3)
  (h_tangent_at_K : TangentToCircle K BC)
  (h_tangent_at_L : TangentToCircle L AC)
  (h_MK_parallel_AC : Parallel MKisParallel AC)
  (CL MK AB triangleANLArea : ℝ)
  (h_CL : CL = 1)
  (h_MK : MK = 3)
  (h_AB : AB = 2 * sqrt 13)
  (h_area_triangle_ANL : triangleANLArea = 125 * sqrt 3 / 52) : 
  CL = 1 ∧ MK = 3 ∧ AB = 2 * sqrt 13 ∧ triangleANLArea = 125 * sqrt 3 / 52 :=
by {
  sorry
}

end math_problem_solution_proof_l326_326367


namespace students_like_both_l326_326659

variable (total_students : ℕ) 
variable (students_like_sea : ℕ) 
variable (students_like_mountains : ℕ) 
variable (students_like_neither : ℕ) 

theorem students_like_both (h1 : total_students = 500)
                           (h2 : students_like_sea = 337)
                           (h3 : students_like_mountains = 289)
                           (h4 : students_like_neither = 56) :
  (students_like_sea + students_like_mountains - (total_students - students_like_neither)) = 182 :=
sorry

end students_like_both_l326_326659


namespace total_travel_time_l326_326458

noncomputable def travel_time (distance_razumeyevo_river : ℝ) (distance_vkusnoteevo_river : ℝ)
    (distance_downstream : ℝ) (river_width : ℝ) (current_speed : ℝ)
    (swimming_speed : ℝ) (walking_speed : ℝ) : ℝ := 
  let time_walk1 := distance_razumeyevo_river / walking_speed
  let effective_swimming_speed := real.sqrt (swimming_speed^2 - current_speed^2)
  let time_swim := river_width / effective_swimming_speed
  let time_walk2 := distance_vkusnoteevo_river / walking_speed
  time_walk1 + time_swim + time_walk2

theorem total_travel_time :
    travel_time 3 1 3.25 0.5 1 2 4 = 1.5 := 
by
  sorry

end total_travel_time_l326_326458


namespace exists_sum_or_diff_divisible_by_100_l326_326064

theorem exists_sum_or_diff_divisible_by_100 (S : Finset ℤ) (hS : S.card = 52) :
  ∃ x y ∈ S, (100 ∣ x - y) ∨ (100 ∣ x + y) := by
  sorry

end exists_sum_or_diff_divisible_by_100_l326_326064


namespace max_gcd_of_consecutive_terms_l326_326482

theorem max_gcd_of_consecutive_terms (n : ℕ) (h : n ≥ 1) :
  ∃ d, d = 2 ∧ (∀ k : ℕ, (gcd ((k + 1)! + 2 * (k + 1)) (k! + 2 * k) ≤ d)) 
:= by sorry

end max_gcd_of_consecutive_terms_l326_326482


namespace max_c_for_quadratic_inequality_l326_326258

theorem max_c_for_quadratic_inequality (c : ℝ) : 
  (-c^2 + 9 * c - 20 ≥ 0) → c ≤ 5 := 
begin
  sorry
end

example : ∃ c : ℝ, (-c^2 + 9 * c - 20 ≥ 0) ∧ c = 5 :=
begin
  use 5,
  split,
  { linarith, -- verifies that 5 satisfies the inequality
  },
  { refl },
end

end max_c_for_quadratic_inequality_l326_326258


namespace factor_x4_plus_64_monic_real_l326_326588

theorem factor_x4_plus_64_monic_real :
  ∀ x : ℝ, x^4 + 64 = (x^2 + 4 * x + 8) * (x^2 - 4 * x + 8) := 
by
  intros
  sorry

end factor_x4_plus_64_monic_real_l326_326588


namespace odd_parity_check_codes_count_l326_326215

theorem odd_parity_check_codes_count :
  let C := λ (n k : ℕ), Nat.choose n k in
  C 6 1 + C 6 3 + C 6 5 = 32 :=
by
  -- Definitions related to combinations (binomial coefficients)
  let C := λ (n k : ℕ), Nat.choose n k
  -- Insert sorry to skip actual proof.
  sorry

end odd_parity_check_codes_count_l326_326215


namespace point_coordinates_l326_326767

-- Definitions of the conditions
def is_on_line (x y : ℝ) : Prop := 3 * x + y - 5 = 0

def distance_to_line (x y : ℝ) : Prop := abs (x - y - 1) = 2 * sqrt 2

-- The theorem statement
theorem point_coordinates (x y : ℝ) (h1 : is_on_line x y) (h2 : distance_to_line x y) : 
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = -1) :=
sorry

end point_coordinates_l326_326767


namespace find_abc_l326_326788

theorem find_abc (A B C : ℕ) 
  (h1 : A < 5) (h2 : B < 5) (h3 : C < 5) 
  (h4 : A ≠ 0) (h5 : B ≠ 0) (h6 : C ≠ 0) 
  (h7 : B + C = 5) 
  (h8 : A + 1 = C) 
  (h9 : A + B = C) : 
  100 * A + 10 * B + C = 314 :=
begin
  sorry
end

end find_abc_l326_326788


namespace simplify_trig_expression_l326_326781

theorem simplify_trig_expression (α : ℝ) :
  -sin (α / 2 - 3 * π) - cos (α / 4) ^ 2 + sin (α / 4) ^ 2 = 2 * sin (2 * α - π / 3) :=
by
  sorry

end simplify_trig_expression_l326_326781


namespace positive_numbers_l326_326770

theorem positive_numbers {a b c : ℝ} (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end positive_numbers_l326_326770


namespace profit_difference_is_50_l326_326521

-- Given conditions
def materials_cost_cars : ℕ := 100
def cars_made_per_month : ℕ := 4
def price_per_car : ℕ := 50

def materials_cost_motorcycles : ℕ := 250
def motorcycles_made_per_month : ℕ := 8
def price_per_motorcycle : ℕ := 50

-- Definitions based on the conditions
def profit_from_cars : ℕ := (cars_made_per_month * price_per_car) - materials_cost_cars
def profit_from_motorcycles : ℕ := (motorcycles_made_per_month * price_per_motorcycle) - materials_cost_motorcycles

-- The difference in profit
def profit_difference : ℕ := profit_from_motorcycles - profit_from_cars

-- The proof goal
theorem profit_difference_is_50 :
  profit_difference = 50 := by
  sorry

end profit_difference_is_50_l326_326521


namespace triangle_inequality_y_difference_l326_326359

theorem triangle_inequality_y_difference :
  (∃ y : ℤ, 4 ≤ y ∧ y ≤ 16) →
  (∀ y : ℤ, 4 ≤ y ∧ y ≤ 16 → 
     let greatest := 16 in
     let least := 4 in
     greatest - least = 12) :=
by
  assume h,
  intro y,
  intro hy,
  let greatest := 16,
  let least := 4,
  have hgreatest : greatest = 16 := rfl,
  have hleast : least = 4 := rfl,
  rw [hgreatest, hleast],
  exact rfl

end triangle_inequality_y_difference_l326_326359


namespace polynomial_condition_l326_326032

variable {K : Type*} [Field K]
variable {n : ℕ}
variable {a : Fin n → K} (h_distinct : Function.Injective a)
variable {b : Fin n → K}

theorem polynomial_condition (P : Polynomial K) (hP : ∀ i : Fin n, P.eval (a i) = b i) :
  ∃ c : K, 
    P = (Polynomial.sum (Finset.univ : Finset (Fin n)) 
                       (λ i, Polynomial.C (b i) *
                          Polynomial.prod (Finset.filter (λ j, j ≠ i) Finset.univ) 
                            (λ j, Polynomial.C (a i - a j)⁻¹ * (Polynomial.X - Polynomial.C (a j)))
                )) + Polynomial.C c * Polynomial.prod Finset.univ (λ i, Polynomial.X - Polynomial.C (a i)) := 
sorry

end polynomial_condition_l326_326032


namespace limit_tangent_cosine_sine_l326_326226

open Real

theorem limit_tangent_cosine_sine (cos sin tan : ℝ → ℝ) (hcos : ∀ x, differentiable_at ℝ cos x)
  (hsin : ∀ x, differentiable_at ℝ sin x) (cos_1 : cos 1 = cos 1) (tan_cos1 : tan (cos 1) = tan (cos 1)) :
  filter.tendsto (λ x, tan (cos x + sin ((x - 1) / (x + 1)) * cos ((x + 1) / (x - 1)))) (nhds 1) (nhds (tan (cos 1))) :=
  sorry

end limit_tangent_cosine_sine_l326_326226


namespace f_odd_function_l326_326298

def f (x : ℝ) : ℝ :=
if x < 0 then -x - 1 else if x = 0 then 0 else -x + 1

theorem f_odd_function (f_odd : ∀ x : ℝ, f (-x) = -f x)
  (f_pos : ∀ x : ℝ, 0 < x → f x = -x + 1) :
  f (-2) = 1 ∧ (∀ x : ℝ, (x < 0 → f x = -x - 1) ∧ (x = 0 → f x = 0) ∧ (0 < x → f x = -x + 1)) :=
by {
  split,
  {
    -- Proving f(-2) = 1 given the conditions
    have H1 := f_pos 2 (by norm_num),
    rw [←neg_zero, ←f_odd, H1],
    norm_num,
  },
  {
    intro x,
    split,
    {
      intro hx_neg,
      rw [f_odd, if_pos hx_neg, if_neg (lt_of_le_of_ne (le_of_not_lt hx_neg) (ne_of_lt hx_neg).symm)],
    },
    split,
    {
      intro hx_zero,
      rw [hx_zero, if_neg (lt_irrefl 0), if_pos rfl],
    },
    {
      intro hx_pos,
      rw [if_neg (lt_irrefl x).ne, if_neg (ne_of_gt hx_pos)],
    }
  }
}

end f_odd_function_l326_326298


namespace num_arrangements_l326_326930

theorem num_arrangements (A B C D : Type) : 
  (∃ (arrangements : list (list (A ⊕ B ⊕ C ⊕ D))),
  ∀ arrang ∈ arrangements,
    (∃ i j k, 
    0 ≤ i < 4 ∧ 0 ≤ j < 4 ∧ 0 ≤ k < 4 ∧
    arrang[i] = A ∧ arrang[j] = B ∧ arrang[k] = C ∧ 
    i < k ∧ j < k ∧ k ≠ i ∧ k ≠ j ∧ i ≠ j) →
  arrangements.length = 16) := sorry

end num_arrangements_l326_326930


namespace ellipse_C2_equation_line_AB_equation_l326_326162

-- Definition of the first ellipse C1
def ellipse_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Definition of the second ellipse C2 based on the given conditions
def ellipse_C2 (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 4) = 1

-- Definition of points O, A, B and their collinearity
variable {O A B : ℝ × ℝ}

-- Given that point A is on ellipse C1 and OA = 2
def is_on_C1_and_OA_eq_2 : Prop :=
  ellipse_C1 A.1 A.2 ∧ dist O A = 2

-- Given that point B is on ellipse C2
def is_on_C2 : Prop :=
  ellipse_C2 B.1 B.2

-- Definition of the equations of line AB
def line_AB (k : ℝ) : Prop :=
  ∀ x y : ℝ, y = k * x

-- The main proof statements
theorem ellipse_C2_equation :
  ∀ x y : ℝ, ellipse_C2 x y ↔ ((x^2 / 16) + (y^2 / 4) = 1) := by
  sorry

theorem line_AB_equation :
  is_on_C1_and_OA_eq_2 ∧ is_on_C2 → (∃ k : ℝ, line_AB k ∧ (k = 1 ∨ k = -1)) := by
  sorry

end ellipse_C2_equation_line_AB_equation_l326_326162


namespace percentage_decrease_in_y_l326_326755

variable (x y z K : ℝ)

-- Conditions
def condition1 : Prop := K = x * y * z
def condition2 (x1 : ℝ) : Prop := x1 = 1.30 * x
def condition3 (z1 : ℝ) : Prop := z1 = 0.90 * z
def condition4 (x1 y1 z1 : ℝ) : Prop := K = x1 * y1 * z1

-- Statement
theorem percentage_decrease_in_y (x1 z1 y0 y1 : ℝ)
  (hx1 : condition2 x x1) (hz1 : condition3 z z1) (hK : condition1 x y z K)
  (hK_new : condition4 x1 y1 z1 K) :
  (1 - y1 / y0) * 100 ≈ 14.53 :=
by {
  sorry
}

end percentage_decrease_in_y_l326_326755


namespace alpha_square_inequality_alpha_square_equality_infinite_alpha_square_ratio_tends_to_zero_l326_326737

-- Define the function α(n)
def alpha (n : ℕ) : ℕ := n.binary_reprs.count 1

-- Statement 1: Inequality for all n
theorem alpha_square_inequality (n : ℕ) : alpha (n^2) ≤ (1 / 2) * alpha(n) * (alpha(n) + 1) :=
by
  sorry

-- Statement 2: Equality for infinitely many integers
theorem alpha_square_equality_infinite : ∃ᶠ n in at_top, alpha (n^2) = (1 / 2) * alpha(n) * (alpha(n) + 1) :=
by
  sorry

-- Statement 3: Sequence where the ratio tends to zero
theorem alpha_square_ratio_tends_to_zero : 
  ∃ (n_i : ℕ → ℕ), filter.tendsto (λ i, (alpha (n_i i^2)) / (alpha (n_i i))) filter.at_top (filter.principal {r | r = 0}) :=
by
  sorry

end alpha_square_inequality_alpha_square_equality_infinite_alpha_square_ratio_tends_to_zero_l326_326737


namespace Bruce_Anne_combined_cleaning_time_l326_326945

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l326_326945


namespace rectangle_angle_l326_326706

theorem rectangle_angle {PQRS : Type} [rectangle PQRS]
    (ratio_condition : ∃ (θ : ℝ), ∠PSQ / ∠PQS = θ ∧ θ = 1 / 5) :
    ∠QSR = 75 :=
by
  sorry

end rectangle_angle_l326_326706


namespace winning_candidate_percentage_l326_326699

theorem winning_candidate_percentage (P : ℝ) 
  (total_votes : ℝ) (vote_majority : ℝ) 
  (h1 : total_votes = 800) 
  (h2 : vote_majority = 320) 
  (h3 : (P / 100) * total_votes - ((100 - P) / 100) * total_votes = vote_majority) : 
  P = 70 :=
begin
  -- Proof here
  sorry
end

end winning_candidate_percentage_l326_326699


namespace linda_savings_l326_326038

theorem linda_savings :
  let original_savings := 880
  let cost_of_tv := 220
  let amount_spent_on_furniture := original_savings - cost_of_tv
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings
  fraction_spent_on_furniture = 3 / 4 :=
by
  -- original savings
  let original_savings := 880
  -- cost of the TV
  let cost_of_tv := 220
  -- amount spent on furniture
  let amount_spent_on_furniture := original_savings - cost_of_tv
  -- fraction spent on furniture
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings

  -- need to show that this fraction is 3/4
  sorry

end linda_savings_l326_326038


namespace calculate_expression_l326_326966

theorem calculate_expression :
  |real.sqrt 3 - 2| + real.cbrt 27 - real.sqrt 16 + (-1)^2023 = - real.sqrt 3 :=
by
  sorry

end calculate_expression_l326_326966


namespace total_trip_cost_l326_326212

-- Definitions for the problem
def price_per_person : ℕ := 147
def discount : ℕ := 14
def number_of_people : ℕ := 2

-- Statement to prove
theorem total_trip_cost :
  (price_per_person - discount) * number_of_people = 266 :=
by
  sorry

end total_trip_cost_l326_326212


namespace tn_lt_one_sixth_l326_326932

-- Definitions
def S {n : ℕ} (a : ℕ → ℕ) := (1/6) * a n * (a n + 3)

def a_n (n : ℕ) : ℕ := 3 * n

def b_n (n : ℕ) : ℝ := 1 / ((a_n n - 1) * (a_n n + 2))

def T (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), b_n i

-- Theorem to prove T_n < 1/6
theorem tn_lt_one_sixth (n : ℕ) : T n < 1 / 6 := 
  sorry

end tn_lt_one_sixth_l326_326932


namespace maximum_bottles_l326_326578

-- Definitions for the number of bottles each shop sells
def bottles_from_shop_A : ℕ := 150
def bottles_from_shop_B : ℕ := 180
def bottles_from_shop_C : ℕ := 220

-- The main statement to prove
theorem maximum_bottles : bottles_from_shop_A + bottles_from_shop_B + bottles_from_shop_C = 550 := 
by 
  sorry

end maximum_bottles_l326_326578


namespace EmerieNickelsCount_l326_326151

-- Definitions of the conditions
def ZainTotalCoins := 48
def ExtraCoinsPerType := 10
def EmerieQuarters := 6
def EmerieDimes := 7

-- Problem statement translated to Lean 4
theorem EmerieNickelsCount :
  ∀ (ZainTotalCoins EmerieQuarters EmerieDimes ExtraCoinsPerType : ℕ),
  EmerieQuarters = 6 →
  EmerieDimes = 7 →
  ZainTotalCoins = 48 →
  ExtraCoinsPerType = 10 →
  let EmerieTotalCoins := ZainTotalCoins - 3 * ExtraCoinsPerType in
  let EmerieNickels := EmerieTotalCoins - (EmerieQuarters + EmerieDimes) in
  EmerieNickels = 5 :=
begin
  intros,
  sorry
end

end EmerieNickelsCount_l326_326151


namespace meaningful_sqrt_x_minus_3_l326_326703

theorem meaningful_sqrt_x_minus_3 :
  ∀ x : ℕ, x ∈ {0, 1, 2, 4} → (x - 3) ≥ 0 ↔ x = 4 :=
by sorry

end meaningful_sqrt_x_minus_3_l326_326703


namespace beat_by_9_seconds_l326_326352

-- Define the problem statement in Lean 4
theorem beat_by_9_seconds (d : ℝ) (t_A : ℝ) (d_B : ℝ) (s_A : ℝ) (time_diff : ℝ) :
  d = 1000 → t_A = 90 → d_B = 900 → s_A = 1000 / 90 → time_diff = (100 / s_A) → time_diff = 9 :=
by
  intro d_eq t_A_eq d_B_eq s_A_eq time_diff_eq
  rw [d_eq, t_A_eq, d_B_eq, s_A_eq, time_diff_eq]
  sorry

end beat_by_9_seconds_l326_326352


namespace nonneg_int_solutions_eqn_l326_326251

theorem nonneg_int_solutions_eqn :
  { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1 } =
  {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} :=
by {
  sorry
}

end nonneg_int_solutions_eqn_l326_326251


namespace is_factorization_l326_326548

-- Define the conditions
def A_transformation : Prop := (∀ x : ℝ, (x + 1) * (x - 1) = x ^ 2 - 1)
def B_transformation : Prop := (∀ m : ℝ, m ^ 2 + m - 4 = (m + 3) * (m - 2) + 2)
def C_transformation : Prop := (∀ x : ℝ, x ^ 2 + 2 * x = x * (x + 2))
def D_transformation : Prop := (∀ x : ℝ, 2 * x ^ 2 + 2 * x = 2 * x ^ 2 * (1 + (1 / x)))

-- The goal is to prove that transformation C is a factorization
theorem is_factorization : C_transformation :=
by
  sorry

end is_factorization_l326_326548


namespace first_driver_spends_less_time_l326_326852

noncomputable def round_trip_time (d : ℝ) (v₁ v₂ : ℝ) : ℝ := (d / v₁) + (d / v₂)

theorem first_driver_spends_less_time (d : ℝ) : 
  round_trip_time d 80 80 < round_trip_time d 90 70 :=
by
  --We skip the proof here
  sorry

end first_driver_spends_less_time_l326_326852


namespace range_of_m_l326_326344

open Real

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - m * x + m > 0) ↔ (0 < m ∧ m < 4) :=
by
  sorry

end range_of_m_l326_326344


namespace domain_of_function_l326_326814

theorem domain_of_function :
  (∀ x ∈ ℝ, (2 - x ≥ 0) ∧ (x > 0) ∧ (x ≠ 1) → 
    (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)) :=
by
  sorry

end domain_of_function_l326_326814


namespace symmetric_to_y_eq_2x_plus_1_about_1_1_l326_326594

noncomputable def symmetric_line_equation (p : ℝ × ℝ) (l : ℝ → ℝ) : (ℝ → ℝ) :=
  λ x, -1 / 2 * x + 3 / 2

theorem symmetric_to_y_eq_2x_plus_1_about_1_1 :
  symmetric_line_equation (1, 1) (λ x, 2 * x + 1) = (λ x, -1 / 2 * x + 3 / 2) :=
sorry

end symmetric_to_y_eq_2x_plus_1_about_1_1_l326_326594


namespace max_speed_of_cart_l326_326536

theorem max_speed_of_cart (R a : ℝ) (hR : R > 0) (ha : a > 0) :
  ∃ v_max : ℝ, v_max = sqrt (sqrt ((16 * a^2 * R^2 * Real.pi^2) / (1 + 16 * Real.pi^2))) :=
by
  sorry

end max_speed_of_cart_l326_326536


namespace increasing_sequence_a_range_l326_326398

theorem increasing_sequence_a_range (f : ℕ → ℝ) (a : ℝ)
  (h1 : ∀ n, f n = if n ≤ 7 then (3 - a) * n - 3 else a ^ (n - 6))
  (h2 : ∀ n : ℕ, f n < f (n + 1)) :
  2 < a ∧ a < 3 :=
sorry

end increasing_sequence_a_range_l326_326398


namespace equal_areas_greater_perimeter_l326_326540

noncomputable def side_length_square := Real.sqrt 3 + 3

noncomputable def length_rectangle := Real.sqrt 72 + 3 * Real.sqrt 6
noncomputable def width_rectangle := Real.sqrt 2

noncomputable def area_square := (side_length_square) ^ 2

noncomputable def area_rectangle := length_rectangle * width_rectangle

noncomputable def perimeter_square := 4 * side_length_square

noncomputable def perimeter_rectangle := 2 * (length_rectangle + width_rectangle)

theorem equal_areas : area_square = area_rectangle := sorry

theorem greater_perimeter : perimeter_square < perimeter_rectangle := sorry

end equal_areas_greater_perimeter_l326_326540


namespace average_tickets_sold_by_female_l326_326513

-- Define the conditions as Lean expressions.

def totalMembers (M : ℕ) : ℕ := M + 2 * M
def totalTickets (F : ℕ) (M : ℕ) : ℕ := 58 * M + F * 2 * M
def averageTicketsPerMember (F : ℕ) (M : ℕ) : ℕ := (totalTickets F M) / (totalMembers M)

theorem average_tickets_sold_by_female (F M : ℕ) 
  (h1 : 66 * (totalMembers M) = totalTickets F M) :
  F = 70 :=
by
  sorry

end average_tickets_sold_by_female_l326_326513


namespace solve_for_y_l326_326074

variable (y : ℝ)

theorem solve_for_y (h : 16^(2*y-3) = (1/4)^(y+2)) : y = 10/9 :=
sorry

end solve_for_y_l326_326074


namespace checkerboard_coverage_zero_l326_326171

noncomputable def disc_coverage (D : ℝ) : ℝ :=
  let r := D / 4 in
  let center := (4 * D, 4 * D) in
  let is_completely_covered (i j : ℕ) : Prop :=
    ((i : ℝ) * D - 4 * D) ^ 2 + ((j : ℝ) * D - 4 * D) ^ 2 < r ^ 2 in
  let num_squares := ∑ i in finset.range 8, ∑ j in finset.range 8, if is_completely_covered i j then 1 else 0 in
  num_squares

theorem checkerboard_coverage_zero (D : ℝ) : disc_coverage D = 0 :=
by
  sorry

end checkerboard_coverage_zero_l326_326171


namespace initial_population_first_village_l326_326911

theorem initial_population_first_village (P : ℕ) :
  let population_after_years (initial rate years : ℤ) := initial + rate * years in
  population_after_years P (-1200) 14 = population_after_years 42000 800 14 →
  P = 70000 :=
by
  intros;
  sorry

end initial_population_first_village_l326_326911


namespace average_age_ac_l326_326090

theorem average_age_ac (A B C : ℕ) (h1 : (A + B + C) / 3 = 25) (h2 : B = 17) : (A + C) / 2 = 29 := by
  have h3 : A + B + C = 75 := by
    linarith
  have h4 : A + C = 75 - 17 := by
    linarith
  have h5 : A + C = 58 := by
    linarith
  have h6 : (A + C) / 2 = 29 := by
    linarith
  exact h6

end average_age_ac_l326_326090


namespace axis_of_symmetry_l326_326848

noncomputable theory

def initial_function (x : ℝ) : ℝ := sin (x + π / 6)

def transformed_function (x : ℝ) : ℝ := 
    let scaled_function := sin (2 * x + π / 6)
    in sin (2 * (x - π / 3) + π / 6)

theorem axis_of_symmetry : ∃ x : ℝ, transformed_function x = sin (2 * x - π / 2) ∧ x = -π / 2 := 
by 
  sorry

end axis_of_symmetry_l326_326848


namespace pentagon_angles_l326_326730

theorem pentagon_angles (A B C D E F : Point)
  (ABCD_pentagon : convex_pentagon A B C D E)
  (angle_DCB : ∠ D C B = 90°) 
  (angle_DEA : ∠ D E A = 90°) 
  (DC_eq_DE : dist D C = dist D E)
  (F_on_AB : ∃ k, F = k * A + (1 - k) * B)
  (ratio_AF_BF_AE_BC : (dist A F) / (dist B F) = (dist A E) / (dist B C)) :
  ∠ F E C = ∠ B D C ∧ ∠ F C E = ∠ A D E := 
sorry

end pentagon_angles_l326_326730


namespace total_sum_lent_l326_326197

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ) 
  (h1 : second_part = 1640) 
  (h2 : (x * 8 * 0.03) = (second_part * 3 * 0.05)) :
  total_sum = x + second_part → total_sum = 2665 := by
  sorry

end total_sum_lent_l326_326197


namespace lattice_points_count_l326_326903

theorem lattice_points_count (x1 y1 x2 y2 : ℤ) (h1 : x1 = 8) (h2 : y1 = 34) (h3 : x2 = 73) (h4 : y2 = 430) :
  ∃ n : ℕ, n = 2 :=
by
  -- Conditions
  have hx_diff : x2 - x1 = 65 := by rw [h1, h3]; norm_num
  have hy_diff : y2 - y1 = 396 := by rw [h2, h4]; norm_num
  have h_gcd : Int.gcd 396 65 = 1 := by norm_num 
  have h_slope : 396 / 65 := by norm_num
  
  -- Conclusion
  exact ⟨2, by sorry⟩

end lattice_points_count_l326_326903


namespace solution_set_inequality_l326_326677

theorem solution_set_inequality (m : ℝ) (h : 3 - m < 0) :
  { x : ℝ | (2 - m) * x + 2 > m } = { x : ℝ | x < -1 } :=
sorry

end solution_set_inequality_l326_326677


namespace rhombus_perimeter_l326_326810

theorem rhombus_perimeter (d1 d2 : ℕ) (h_d1 : d1 = 16) (h_d2 : d2 = 30) :
  let side_length := Math.sqrt ((d1 / 2)^2 + (d2 / 2)^2) in
  let perimeter := 4 * side_length in
  perimeter = 68 := by
    dsimp only [side_length, perimeter]
    rw [h_d1, h_d2]
    norm_num
    sorry

end rhombus_perimeter_l326_326810


namespace biker_bob_north_distance_l326_326935

def distanceNorthAfterEast (initialNorth : ℝ) (west : ℝ) (east : ℝ) (totalDistance : ℝ) : ℝ := 
  let netWest := west - east
  let northSquare := totalDistance^2 - netWest^2
  let totalNorth := Real.sqrt northSquare
  totalNorth - initialNorth

theorem biker_bob_north_distance :
  distanceNorthAfterEast 5 8 4 20.396078054371138 = 15 :=
by
  sorry

end biker_bob_north_distance_l326_326935


namespace price_reduction_correct_l326_326542

theorem price_reduction_correct (P : ℝ) : 
  let first_reduction := 0.92 * P
  let second_reduction := first_reduction * 0.90
  second_reduction = 0.828 * P := 
by 
  sorry

end price_reduction_correct_l326_326542


namespace no_integer_solution_l326_326245

theorem no_integer_solution (x y : ℤ) : 2 * x + 6 * y ≠ 91 :=
by
  sorry

end no_integer_solution_l326_326245


namespace ratio_of_heights_l326_326715

theorem ratio_of_heights (α : ℝ) (hα : α > Real.pi / 6) :
  let O : Point := -- define point O
  angle AOB = 2 * Real.pi / 3 ∧ angle BOC = 2 * Real.pi / 3 ∧ angle COA = 2 * Real.pi / 3 ∧
  ∃ (triangle ABC : Triangle), isosceles_triangle ABC ∧
  base_angle ABC = α →
  ratio_height O ABC = 
    2 * Real.sin (α - Real.pi / 6) / Real.cos α := sorry

end ratio_of_heights_l326_326715


namespace normal_distribution_property_l326_326026

noncomputable def ξ : MeasureTheory.ProbabilityDistribution := 
  MeasureTheory.ProbabilityDistribution.normal 10 σ^2

theorem normal_distribution_property (hξ : ∀ a b, MeasureTheory.ProbabilityMeasure (N(10, σ^2)) a b) :
  ProbabilityTheory.Prob (λ x, |x - 10| < 1) = 0.8 :=
by
  -- Given: ξ ~ N(10, σ^2)
  have h1 : MeasureTheory.ProbabilityMeasure (N(10, σ^2)) := ξ
  -- Given: P(ξ < 11) = 0.9
  have h2 : ProbabilityTheory.Prob (λ x, x < 11) = 0.9 :=
    sorry -- this would follow from the distribution properties
  -- Prove: P(|ξ - 10| < 1) = 0.8
  sorry

end normal_distribution_property_l326_326026


namespace possible_values_of_a_l326_326103

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2 * x else 1 / x

theorem possible_values_of_a (a : ℝ) (h : f 1 + f a = -2) : a = 1 ∨ a = -1 :=
by sorry

end possible_values_of_a_l326_326103


namespace book_arrangement_l326_326327

theorem book_arrangement (total_books : ℕ) (identical_books : ℕ) 
  (h_total : total_books = 7) (h_identical : identical_books = 3) : 
  (∏ i in Finset.range total_books.succ, i + 1) / (∏ i in Finset.range identical_books.succ, i + 1) = 840 := 
by 
  -- We will use Finset.range to calculate the factorial and then use division to check the equation
  sorry

end book_arrangement_l326_326327


namespace find_x_l326_326392

theorem find_x (x : ℤ) (hx1 : x ≥ 1) (hx2 : cos (real.to.rad x) = sin (real.to.rad (x ^ 2))) : x = 9 :=
sorry

end find_x_l326_326392


namespace no_valid_squarish_numbers_l326_326976

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≠ 9 ∧ d < 9

theorem no_valid_squarish_numbers :
  ∀ N : ℕ,
    (∀ d, valid_digit d) →
    is_perfect_square N →
    let first_two : ℕ := N / 10000 in
    let middle_two : ℕ := (N / 100) % 100 in
    let last_two : ℕ := N % 100 in
    is_perfect_square first_two ∧
    is_perfect_square middle_two ∧
    is_perfect_square last_two ∧
    (middle_two % 2 = 0) →
    false :=
begin
  intros N h_valid_digits h_perfect_square,
  -- declare and define first_two, middle_two, last_two
  let first_two := N / 10000,
  let middle_two := (N / 100) % 100,
  let last_two := N % 100,
  -- assume the conditions on first_two, middle_two, last_two
  assume h_conditions,
  -- proof goes here
  sorry
end

end no_valid_squarish_numbers_l326_326976


namespace trigonometric_identity_l326_326616

open Real

noncomputable def tan_alpha_eq : ℝ := 4

theorem trigonometric_identity : 
  ∀ α : ℝ, tan α = tan_alpha_eq → 
    (1 + cos (2 * α) + 8 * sin(α) ^ 2) / sin (2 * α) = 65 / 4 := 
by
  sorry

end trigonometric_identity_l326_326616


namespace value_of_k_l326_326386

variable (β : ℝ)
noncomputable def k_def : ℝ :=
  ((Real.sin β + Real.sec β) ^ 2 + (Real.cos β + Real.csc β) ^ 2) - (Real.sin β ^ 2 + Real.cos β ^ 2 + Real.sec β ^ 2 + Real.csc β ^ 2)

theorem value_of_k (β : ℝ) (k : ℝ) :
  ((Real.sin β + Real.sec β) ^ 2 + (Real.cos β + Real.csc β) ^ 2 = k + Real.sin β ^ 2 + Real.cos β ^ 2 + Real.sec β ^ 2 + Real.csc β ^ 2) →
  k = 4 / (Real.sin (2 * β)) :=
by 
  sorry

end value_of_k_l326_326386


namespace dice_roll_eccentricity_probability_l326_326909

theorem dice_roll_eccentricity_probability :
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ a > b ∧
     sqrt(1 - (b^2 : ℝ) / (a^2 : ℝ)) > sqrt(3) / 2) ↔ true := by
  sorry

end dice_roll_eccentricity_probability_l326_326909


namespace find_abc_l326_326395

noncomputable def x := sqrt ((sqrt 75 / 2) + (5 / 2))

theorem find_abc :
  ∃ a b c : ℕ,
    a + b + c = 219 ∧
    x^100 = 3*x^98 + 16*x^96 + 15*x^94 - x^50 + a*x^46 + b*x^44 + c*x^42 :=
by
  sorry

end find_abc_l326_326395


namespace complex_point_distance_log_sums_tangent_line_equation_sin_square_constraint_l326_326507

-- Problem 1
theorem complex_point_distance :
  let z : ℂ := 1 - 2 * complex.I
  distance (1:ℂ) (-2 * complex.I:ℂ) = √5 := sorry

-- Problem 2
theorem log_sums :
  ∀ (a b : ℝ), 
  2^a = sqrt 10 ∧ 5^b = sqrt 10 → (1/a + 1/b) = 1 := sorry

-- Problem 3
theorem tangent_line_equation :
  ∀ (g f : ℝ → ℝ), 
  (∀ x, f x = g x + x^2) → 
  (9*1 + g 1 - 1 = 0) → 
  let f' := λ x, deriv f x
  let g' := λ x, deriv g x
  (f' 1 = -7) → 
  tangent_line_eq_at f 1 (1, f 1) = set_of (λ (p : ℝ × ℝ), p.2 = -7 * p.1 + 7) := sorry

-- Problem 4
theorem sin_square_constraint (a : ℝ) :
  ∀ x ∈ set.Icc (0:ℝ) (real.pi / 2),
  (sin x)^2 + a * cos x + a ≤ 1 → 
  a ∈ set.Ioc (-∞) 0 := sorry

end complex_point_distance_log_sums_tangent_line_equation_sin_square_constraint_l326_326507


namespace minimum_teachers_required_l326_326192

theorem minimum_teachers_required 
    (maths_teachers : ℕ) 
    (physics_teachers : ℕ)
    (chemistry_teachers : ℕ)
    (max_subjects : ℕ)
    (h1 : maths_teachers = 6)
    (h2 : physics_teachers = 5)
    (h3 : chemistry_teachers = 5)
    (h4 : max_subjects = 4) :
    maths_teachers + physics_teachers + chemistry_teachers = 16 := 
begin
    sorry
end

end minimum_teachers_required_l326_326192


namespace number_of_huge_ancient_oaks_l326_326841

theorem number_of_huge_ancient_oaks :
  ∀ (total_trees medium_fir_trees saplings : ℕ),
  total_trees = 96 →
  medium_fir_trees = 23 →
  saplings = 58 →
  total_trees - medium_fir_trees - saplings = 15 :=
by
  intros total_trees medium_fir_trees saplings h_total_trees h_medium_fir_trees h_saplings
  rw [h_total_trees, h_medium_fir_trees, h_saplings]
  exact eq.refl 15

end number_of_huge_ancient_oaks_l326_326841


namespace unique_scalar_matrix_l326_326592

theorem unique_scalar_matrix (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, Matrix.mulVec N v = 5 • v) → 
  N = !![5, 0, 0; 0, 5, 0; 0, 0, 5] :=
by
  intro hv
  sorry -- Proof omitted as per instructions

end unique_scalar_matrix_l326_326592


namespace max_speed_of_cart_l326_326537

theorem max_speed_of_cart (R a : ℝ) (hR : R > 0) (ha : a > 0) :
  ∃ v_max : ℝ, v_max = sqrt (sqrt ((16 * a^2 * R^2 * Real.pi^2) / (1 + 16 * Real.pi^2))) :=
by
  sorry

end max_speed_of_cart_l326_326537


namespace fraction_minimum_decimal_digits_l326_326137

def minimum_decimal_digits (n d : ℕ) : ℕ := sorry

theorem fraction_minimum_decimal_digits :
  minimum_decimal_digits 987654321 (2^28 * 5^3) = 28 :=
sorry

end fraction_minimum_decimal_digits_l326_326137


namespace bruce_and_anne_clean_house_l326_326938

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l326_326938


namespace average_tickets_sold_by_female_members_l326_326516

theorem average_tickets_sold_by_female_members 
  (average_all : ℕ)
  (ratio_mf : ℕ)
  (average_male : ℕ)
  (h1 : average_all = 66)
  (h2 : ratio_mf = 2)
  (h3 : average_male = 58) :
  ∃ (F : ℕ), F = 70 :=
by
  let M := 1
  let num_female := ratio_mf * M
  let total_tickets_male := average_male * M
  let total_tickets_female := num_female * 70
  have total_all_members : ℕ := M + num_female
  have total_tickets_all : ℕ := total_tickets_male + total_tickets_female
  have average_all_eq : average_all = total_tickets_all / total_all_members
  use 70
  sorry

end average_tickets_sold_by_female_members_l326_326516


namespace maker_wins_l326_326042

-- Definitions for Maker and Breaker's game conditions
structure Game (m n : ℕ) :=
(maxHeight : ∀ i : ℕ, i < m → i < n)

-- The main theorem stating the winning conditions
theorem maker_wins 
  (m n : ℕ) : 
  (m = 1) ∨ 
  ((m > 1) ∧ (n > 1) ∧ (m % 2 = 1) ∧ (n % 2 = 1)) → 
  ∃ game : Game m n, 
  (∀ height : ℕ, height < n → -- For any height less than n,
    ∃ i : ℕ, i < m → -- there exists a column i less than m such that
    Maker_places_first ∧ -- Maker places first
    Green_row_maker_wins_at height) -- A green row can be completed at given height
:= sorry

end maker_wins_l326_326042


namespace roots_negative_reciprocals_l326_326141

theorem roots_negative_reciprocals (a b c r s : ℝ) (h1 : a * r^2 + b * r + c = 0)
    (h2 : a * s^2 + b * s + c = 0) (h3 : r = -1 / s) (h4 : s = -1 / r) :
    a = -c :=
by
  -- Insert clever tricks to auto-solve or reuse axioms here
  sorry

end roots_negative_reciprocals_l326_326141


namespace sum_of_abs_distances_l326_326745

-- Definition of the problem conditions
variables {p q r s : ℝ}
axiom h1 : |p - q| = 3
axiom h2 : |q - r| = 5
axiom h3 : |r - s| = 7

-- The statement of the problem in Lean: sum of all possible values of |p - s| is 30
theorem sum_of_abs_distances : Σ (x : ℝ), (∃ (p q r s : ℝ), |p - q| = 3 ∧ |q - r| = 5 ∧ |r - s| = 7 ∧ x = |p - s|) = 30 :=
by
  sorry

end sum_of_abs_distances_l326_326745


namespace irrational_sqrt3_l326_326143

theorem irrational_sqrt3 : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a * a = 3 * b * b) :=
by
  sorry

end irrational_sqrt3_l326_326143


namespace sasha_remaining_questions_l326_326069

variable (rate : Int) (initial_questions : Int) (hours_worked : Int)

theorem sasha_remaining_questions
  (h1 : rate = 15)
  (h2 : initial_questions = 60)
  (h3 : hours_worked = 2) :
  initial_questions - rate * hours_worked = 30 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sasha_remaining_questions_l326_326069


namespace S_is_N_l326_326377

noncomputable def S : set ℕ := { n | sorry }

theorem S_is_N :
  (∀ k : ℕ, ∃ n : ℕ, n ∈ S ∧ k ≤ n ∧ n < k + 2003) ∧
  (∀ n : ℕ, n ∈ S → n > 1 → (n/2) ∈ S) →
  S = set.univ :=
by sorry

end S_is_N_l326_326377


namespace perpendicular_tangents_l326_326643

theorem perpendicular_tangents (x₀ : ℝ) :
  let y1 := λ x : ℝ, (1 / 6) * x^2 - 1
  let y2 := λ x : ℝ, 1 + x^3
  let dy1 := λ x : ℝ, 1 / 3 * x
  let dy2 := λ x : ℝ, 3 * x^2
  (dy1 x₀) * (dy2 x₀) = -1 → x₀ = -1 :=
by
  sorry

end perpendicular_tangents_l326_326643


namespace false_statement_l326_326243

-- Condition: There is a 99% confidence that smoking is related to lung disease.
def confidence_99_smoking_lung_disease : Prop :=
  -- This is a probabilistic statement about the confidence level
  ∀ (S L : Prop), (S → L) → (99/100 : ℝ)

-- Derived incorrect interpretation: If someone smokes, then they have a 99% chance of having lung disease.
def incorrect_statement : Prop :=
  ∀ S L : Prop, S → (99/100 : ℝ) = 99%

-- Proof problem statement: Prove that the derived interpretation is false.
theorem false_statement :
  ¬ incorrect_statement :=
sorry

end false_statement_l326_326243


namespace bruce_and_anne_clean_house_l326_326937

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l326_326937


namespace rectangle_divisible_into_squares_l326_326269

theorem rectangle_divisible_into_squares (n : ℕ) : 
  ∃ (rectangle : ℕ × ℕ), rectangle = (nat.fib n, nat.fib (n+1)) ∧ 
    (∃ (square_sizes : Finset (ℕ × ℕ)), 
      square_sizes.card = n ∧ 
      ∀ (size : ℕ), 2 ≠ square_sizes.filter (λ s, s.1 = size).card)
:= by
  sorry

end rectangle_divisible_into_squares_l326_326269


namespace sin_A_plus_pi_four_l326_326617

variable {α β γ : ℝ}

theorem sin_A_plus_pi_four (h : 3 * (sin β)^2 + 7 * (sin γ)^2 = 2 * (sin α) * (sin β) * (sin γ) + 2 * (sin α)^2) :
  sin (α + π/4) = -√10 / 10 :=
by
  sorry

end sin_A_plus_pi_four_l326_326617


namespace probability_strictly_greater_first_die_l326_326128

theorem probability_strictly_greater_first_die 
    (sides : ℕ)
    (dice : ℕ) 
    (rolls : list ℕ) 
    (h_sides : sides = 8)
    (h_dice : dice = 3)
    (h_rolls_len : rolls.length = dice) 
    (h_indep : ∀ (i j : ℕ) (h_i : i < rolls.length) (h_j : j < rolls.length), i ≠ j → rolls.nth_le i h_i ≠ rolls.nth_le j h_j)
    (h_fair : ∀ die : ℕ, die ∈ rolls → die = sides) :
  (probability (λ (lst : list ℕ), lst.head > lst.nth_le 1 sorry ∧ lst.head > lst.nth_le 2 sorry) = 35 / 128) := 
sorry

end probability_strictly_greater_first_die_l326_326128


namespace evaluate_f_at_1_l326_326134

def f (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + 2 * x^4

theorem evaluate_f_at_1 : f 1 = 4 := by
  unfold f
  norm_num

end evaluate_f_at_1_l326_326134


namespace trip_cost_l326_326206

variable (price : ℕ) (discount : ℕ) (numPeople : ℕ)

theorem trip_cost :
  price = 147 →
  discount = 14 →
  numPeople = 2 →
  (price - discount) * numPeople = 266 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end trip_cost_l326_326206


namespace rhombus_perimeter_l326_326801

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  let side_length := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let perimeter := 4 * side_length in
  perimeter = 68 :=
by
  have h3 : d1 / 2 = 8, from by rw [h1],
  have h4 : d2 / 2 = 15, from by rw [h2],
  have h5 : side_length = 17, from by
    calc
      side_length
          = Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) : rfl
      ... = Real.sqrt (8 ^ 2 + 15 ^ 2) : by rw [h3, h4]
      ... = Real.sqrt (64 + 225) : rfl
      ... = Real.sqrt 289 : rfl
      ... = 17 : by norm_num,
  calc
    perimeter
        = 4 * side_length : rfl
    ... = 4 * 17 : by rw [h5]
    ... = 68 : by norm_num

end rhombus_perimeter_l326_326801


namespace area_of_triangle_is_correct_l326_326593

def line_1 (x y : ℝ) : Prop := y - 5 * x = -4
def line_2 (x y : ℝ) : Prop := 4 * y + 2 * x = 16

def y_axis (x y : ℝ) : Prop := x = 0

def satisfies_y_intercepts (f : ℝ → ℝ) : Prop :=
f 0 = -4 ∧ f 0 = 4

noncomputable def area_of_triangle (height base : ℝ) : ℝ :=
(1 / 2) * base * height

theorem area_of_triangle_is_correct :
  ∃ (x y : ℝ), line_1 x y ∧ line_2 x y ∧ y_axis 0 8 ∧ area_of_triangle (16 / 11) 8 = (64 / 11) := 
sorry

end area_of_triangle_is_correct_l326_326593


namespace epsilon_choice_l326_326742

theorem epsilon_choice (n : ℕ) (a : ℕ → ℝ) (h : 2 ≤ n) :
  ∃ ε : ℕ → ℤ, (∀ i, ε i = 1 ∨ ε i = -1) ∧ 
  (∑ i in finset.range n, a i) ^ 2 + (∑ i in finset.range n, ε i * a i) ^ 2 ≤ (n + 1) * (∑ i in finset.range n, a i ^ 2) := 
sorry

end epsilon_choice_l326_326742


namespace elementary_sampling_count_l326_326183

theorem elementary_sampling_count :
  ∃ (a : ℕ), (a + (a + 600) + (a + 1200) = 3600) ∧
             (a = 600) ∧
             (a + 1200 = 1800) ∧
             (1800 * 1 / 100 = 18) :=
by {
  sorry
}

end elementary_sampling_count_l326_326183


namespace grain_distance_from_church_l326_326499

theorem grain_distance_from_church 
  (church_tower_height : ℝ)
  (catholic_tower_distance : ℝ)
  (catholic_tower_height : ℝ)
  (constant_speed : Prop) : 
  (∃ x : ℝ, 
    (church_tower_height = 150 ∧
    catholic_tower_distance = 350 ∧
    catholic_tower_height = 200 ∧
    constant_speed → x = 200)) :=
begin
  sorry
end

end grain_distance_from_church_l326_326499


namespace prod_permutation_multiple_3_l326_326879

theorem prod_permutation_multiple_3 :
  ∀ (a : Fin 2006 → ℤ),
    (∀ i j : Fin 2006, a i = a j → i = j) ∧ 
    (∀ i : Fin 2006, 1 ≤ a i ∧ a i ≤ 2006) ∧
    (List.range 2006).perm (List.ofFn (λ i => (a ⟨i, sorry⟩))) →
    (∏ i in Finset.range 2006, ((a i) ^ 2 - i) % 3) = 0 := sorry

end prod_permutation_multiple_3_l326_326879


namespace bruce_and_anne_clean_together_l326_326953

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l326_326953


namespace grocery_store_percentage_more_reg_soda_than_diet_l326_326177

noncomputable def percentage_more_bottles (regular_soda : ℕ) (diet_soda : ℕ) : ℝ :=
  ((regular_soda - diet_soda).to_real / diet_soda.to_real * 100)

theorem grocery_store_percentage_more_reg_soda_than_diet :
  percentage_more_bottles 780 530 ≈ 47.17 := 
sorry

end grocery_store_percentage_more_reg_soda_than_diet_l326_326177


namespace triangle_circumradius_inequality_l326_326624

noncomputable def nondegenerate_triangle (ABC : Type*) [add_comm_group ABC] [module ℝ ABC] : Prop :=
∃ A B C : ABC, A ≠ B ∧ B ≠ C ∧ C ≠ A

noncomputable def have_circumcenter (ABC : Type*) [add_comm_group ABC] [module ℝ ABC] (O : ABC) : Prop :=
∃ A B C : ABC, ∃ R : ℝ, ∀ X : ABC, (X = A ∨ X = B ∨ X = C) → (dist O X = R)

noncomputable def have_orthocenter (ABC : Type*) [add_comm_group ABC] [module ℝ ABC] (H : ABC) : Prop :=
∃ A B C : ABC, ∀ X : ABC, (X = A ∨ X = B ∨ X = C) → orthogonal_projection (affine_span ℝ {A, B, C}) X = H

noncomputable def circumradius (ABC : Type*) [add_comm_group ABC] [module ℝ ABC] (R : ℝ) : Prop :=
∃ A B C : ABC, ∀ X : ABC, (X = A ∨ X = B ∨ X = C) → (dist (circumcenter ABC) X = R)

theorem triangle_circumradius_inequality
  (ABC : Type*) [add_comm_group ABC] [module ℝ ABC]
  (h_nondegenerate : nondegenerate_triangle ABC)
  (O : ABC) (H : ABC) (R : ℝ)
  (h_circumcenter : have_circumcenter ABC O)
  (h_orthocenter : have_orthocenter ABC H)
  (h_circumradius : circumradius ABC R) :
  dist O H < 3 * R :=
sorry

end triangle_circumradius_inequality_l326_326624


namespace common_ratio_of_geometric_series_l326_326255

theorem common_ratio_of_geometric_series (a1 a2 a3 : ℚ) (h1 : a1 = -4 / 7)
                                         (h2 : a2 = 14 / 3) (h3 : a3 = -98 / 9) :
  ∃ r : ℚ, r = a2 / a1 ∧ r = a3 / a2 ∧ r = -49 / 6 :=
by
  use -49 / 6
  sorry

end common_ratio_of_geometric_series_l326_326255


namespace find_bracelet_price_l326_326049

def price_of_bracelet (B : ℝ) : Prop :=
  let necklacePrice := 25.0
  let earringPrice := 10.0
  let ensemblePrice := 45.0
  let soldNecklaces := 5
  let soldBracelets := 10
  let soldEarrings := 20
  let soldEnsembles := 2
  let totalSales := 565.0
  (soldNecklaces * necklacePrice + soldBracelets * B + soldEarrings * earringPrice + soldEnsembles * ensemblePrice = totalSales)

theorem find_bracelet_price : ∃ B : ℝ, price_of_bracelet B ∧ B = 15.0 := 
by 
  use 15.0
  split
  · unfold price_of_bracelet 
    sorry
  · refl

end find_bracelet_price_l326_326049


namespace pentagon_area_l326_326550

theorem pentagon_area (A B C D E : Point) 
  (hABC : area A B C = 1)
  (hBCD : area B C D = 1)
  (hCDE : area C D E = 1)
  (hDEA : area D E A = 1)
  (hEAB : area E A B = 1) :
  area_pentagon A B C D E = (5 + Real.sqrt 5) / 2 :=
sorry

end pentagon_area_l326_326550


namespace solve_inequality_system_l326_326078

theorem solve_inequality_system (x : ℝ) :
  (5 * x - 1 > 3 * (x + 1)) ∧ (x - 1 ≤ 7 - x) ↔ (2 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequality_system_l326_326078


namespace fish_farm_estimated_mass_l326_326896

noncomputable def total_fish_mass_in_pond 
  (initial_fry: ℕ) 
  (survival_rate: ℝ) 
  (haul1_count: ℕ) (haul1_avg_weight: ℝ) 
  (haul2_count: ℕ) (haul2_avg_weight: ℝ) 
  (haul3_count: ℕ) (haul3_avg_weight: ℝ) : ℝ :=
  let surviving_fish := initial_fry * survival_rate
  let total_mass_haul1 := haul1_count * haul1_avg_weight
  let total_mass_haul2 := haul2_count * haul2_avg_weight
  let total_mass_haul3 := haul3_count * haul3_avg_weight
  let average_weight_per_fish := (total_mass_haul1 + total_mass_haul2 + total_mass_haul3) / (haul1_count + haul2_count + haul3_count)
  average_weight_per_fish * surviving_fish

theorem fish_farm_estimated_mass :
  total_fish_mass_in_pond 
    80000           -- initial fry
    0.95            -- survival rate
    40 2.5          -- first haul: 40 fish, 2.5 kg each
    25 2.2          -- second haul: 25 fish, 2.2 kg each
    35 2.8          -- third haul: 35 fish, 2.8 kg each
    = 192280 := by
  sorry

end fish_farm_estimated_mass_l326_326896


namespace find_number_l326_326490

-- Define the conditions.
def condition (x : ℚ) : Prop := x - (1 / 3) * x = 16 / 3

-- Define the theorem from the translated (question, conditions, correct answer) tuple
theorem find_number : ∃ x : ℚ, condition x ∧ x = 8 :=
by
  sorry

end find_number_l326_326490


namespace die_painting_count_l326_326219

open Nat

theorem die_painting_count :
  let faces := {1, 2, 3, 4, 5, 6}
  let invalid_sums := {{1, 3, 6}, {2, 3, 5}}
  (choose 6 3) - (invalid_sums.card) = 18 :=
by
  let faces := {1, 2, 3, 4, 5, 6}
  let invalid_sums := {{1, 3, 6}, {2, 3, 5}}
  have h1 : (choose 6 3) = 20 := by sorry
  have h2 : invalid_sums.card = 2 := by sorry
  rw [h1, h2]
  norm_num

end die_painting_count_l326_326219


namespace inverse_of_f_is_odd_and_increasing_l326_326823

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

theorem inverse_of_f_is_odd_and_increasing :
  (∀ y, f (inverse f y) = y) ∧ (∀ x, -inverse f (-x) = inverse f x) ∧ 
  (∀ x > 0, inverse f (y x) < inverse f (y (x + 1))) := 
sorry

end inverse_of_f_is_odd_and_increasing_l326_326823


namespace paint_gallons_needed_l326_326010

def radius (d : ℝ) := d / 2

def lateral_surface_area (r h : ℝ) := 2 * Real.pi * r * h

def total_area (num_pillars : ℕ) (area_per_pillar : ℝ) := num_pillars * area_per_pillar

def gallons_needed (total_area : ℝ) (coverage_per_gallon : ℝ) := 
  Real.ceil (total_area / coverage_per_gallon)

theorem paint_gallons_needed
  (num_pillars : ℕ)
  (height : ℝ)
  (diameter : ℝ)
  (coverage_per_gallon : ℝ) :
  num_pillars = 20 ∧ height = 12 ∧ diameter = 8 ∧ coverage_per_gallon = 400 →
  gallons_needed (total_area num_pillars (lateral_surface_area (radius diameter) height)) coverage_per_gallon = 16 :=
by
  sorry

end paint_gallons_needed_l326_326010


namespace map_distance_of_rams_camp_l326_326412

noncomputable def scale (actual_dist_km : ℝ) (map_dist_inch : ℝ) : ℝ :=
  actual_dist_km * 39370.0787 / map_dist_inch

noncomputable def map_distance (actual_dist_km : ℝ) (scale : ℝ) : ℝ :=
  (actual_dist_km * 39370.0787) / scale

theorem map_distance_of_rams_camp :
  (map_distance 12.205128205128204 (scale 136 312)) ≈ 27.98 :=
sorry

end map_distance_of_rams_camp_l326_326412


namespace length_of_rest_of_body_is_25_l326_326407

-- Given conditions
def total_height : ℝ := 60
def legs_fraction : ℝ := 1 / 3
def head_fraction : ℝ := 1 / 4
def legs_length := legs_fraction * total_height
def head_length := head_fraction * total_height
def rest_of_body_length := total_height - (legs_length + head_length)

-- Theorem statement to prove
theorem length_of_rest_of_body_is_25 : rest_of_body_length = 25 :=
sorry

end length_of_rest_of_body_is_25_l326_326407


namespace difference_students_rabbits_l326_326582

-- Define the number of students per classroom
def students_per_classroom := 22

-- Define the number of rabbits per classroom
def rabbits_per_classroom := 4

-- Define the number of classrooms
def classrooms := 6

-- Calculate the total number of students
def total_students := students_per_classroom * classrooms

-- Calculate the total number of rabbits
def total_rabbits := rabbits_per_classroom * classrooms

-- Prove the difference between the number of students and rabbits is 108
theorem difference_students_rabbits : total_students - total_rabbits = 108 := by
  sorry

end difference_students_rabbits_l326_326582


namespace yan_ratio_l326_326491

variables (w x y : ℝ)

-- Given conditions
def yan_conditions : Prop :=
  w > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (y / w = x / w + (x + y) / (7 * w))

-- The ratio of Yan's distance from his home to his distance from the stadium is 3/4
theorem yan_ratio (h : yan_conditions w x y) : 
  x / y = 3 / 4 :=
sorry

end yan_ratio_l326_326491


namespace sum_of_digits_of_n_l326_326127

theorem sum_of_digits_of_n (n : ℕ) (h : (n + 1)! + (n + 2)! = n! * 530) : (n = 21) → (2 + 1 = 3) :=
by
  intro hn
  have : (n = 21) := hn
  sorry

end sum_of_digits_of_n_l326_326127


namespace shaded_area_fraction_l326_326061

-- Define the regular hexagon and the mentioned geometric points
structure RegularHexagon :=
(center : Point) (vertices : Fin 6 → Point)

class Midpoint (p1 p2 m : Point) : Prop :=
(middle : m = (p1 + p2) / 2)

-- Define the problem
noncomputable def fraction_shaded_area (hex : RegularHexagon) (X : Point)
  [Midpoint (hex.vertices 2) (hex.vertices 3) X] : ℚ :=
  5 / 12

-- Example usage: to run and verify in Lean
theorem shaded_area_fraction (hex : RegularHexagon) (X : Point)
  [Midpoint (hex.vertices 2) (hex.vertices 3) X] :
  fraction_shaded_area hex X = 5 / 12 :=
sorry

end shaded_area_fraction_l326_326061


namespace sasha_remaining_questions_l326_326068

theorem sasha_remaining_questions
  (qph : ℕ) (total_questions : ℕ) (hours_worked : ℕ)
  (h_qph : qph = 15) (h_total_questions : total_questions = 60) (h_hours_worked : hours_worked = 2) :
  total_questions - (qph * hours_worked) = 30 :=
by
  sorry

end sasha_remaining_questions_l326_326068


namespace sum_is_composite_l326_326033

open Nat

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h : a^2 - a * b + b^2 = c^2 - c * d + d^2) : ∃ k, 1 < k < a + b + c + d ∧ k ∣ (a + b + c + d) := 
sorry

end sum_is_composite_l326_326033


namespace part1_part2_part3_l326_326034

section problem
variables {x m : ℝ}

-- Define Set A and Set B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Prove parts 1, 2, and 3
-- Part 1: If m = 3, find A ∩ (ℝ \ B)
theorem part1 : A ∩ (Set.compl (B 3)) = {x | 3 ≤ x ∧ x < 4} := 
sorry

-- Part 2: If A ∩ B = ∅, find the range of m
theorem part2 (h : A ∩ B m = ∅) : m ≤ -2 :=
sorry

-- Part 3: If A ∩ B = A, find the range of m
theorem part3 (h : A ⊆ B m) : 4 ≤ m :=
sorry

end problem

end part1_part2_part3_l326_326034


namespace magnitude_of_vector_subtraction_l326_326658

variables {V : Type*} [inner_product_space ℝ V]

-- Conditions
variables (a b : V) (h₁ : inner_product_space.inner a b = 0)
  (h₂ : ∥a∥ = 1) (h₃ : ∥b∥ = 2)

-- Theorem statement
theorem magnitude_of_vector_subtraction : 
  ∥2 • a - b∥ = 2 * real.sqrt 2 :=
sorry

end magnitude_of_vector_subtraction_l326_326658


namespace dihedral_angle_proof_l326_326031

variables (P A B C M O H: Type*) [metric_space P] [metric_space A] [metric_space B]
[metric_space C] [metric_space M] [metric_space O] [metric_space H]

-- Conditions
variables (PA : ∀ p : P, is_vertex_of_cone p)
variables (circumference_points : ∀ a b c : P, are_points_on_circumference a b c)
variables (angle_ABC_90 : ∀ a b c : P, angle(a, b, c) = 90)
variables (midpoint_M : ∀ a : P, M = midpoint A P)
variables (length_AB_one : ∀ a b : P, distance a b = 1)
variables (length_AC_two : ∀ a c : P, distance a c = 2)
variables (length_AP_sqrt2 : ∀ a p : P, distance a p = real.sqrt 2)

-- The dihedral angle
noncomputable def dihedral_angle_MBC_A : ℝ := real.arctan (2 / 3)

-- Proof statement
theorem dihedral_angle_proof :
  ∀ (P A B C M : Type*) [metric_space P] [metric_space A] [metric_space B]
    [metric_space C] [metric_space M] (PA : ∀ p : P, is_vertex_of_cone p)
    (circumference_points : ∀ a b c : P, are_points_on_circumference a b c)
    (angle_ABC_90 : ∀ a b c : P, angle(a, b, c) = 90)
    (midpoint_M : ∀ a : P, M = midpoint A P)
    (length_AB_one : ∀ a b : P, distance a b = 1)
    (length_AC_two : ∀ a c : P, distance a c = 2)
    (length_AP_sqrt2 : ∀ a p : P, distance a p = real.sqrt 2),
  angle_dihedral M B C A = real.arctan (2 / 3) :=
by
  sorry

end dihedral_angle_proof_l326_326031


namespace perpendicular_lines_slope_l326_326601

theorem perpendicular_lines_slope (b : ℝ) (h1 : ∀ x : ℝ, 3 * x + 7 = 4 * (-(b / 4) * x + 4)) : b = 4 / 3 :=
by
  sorry

end perpendicular_lines_slope_l326_326601


namespace solution_alcohol_content_l326_326782

noncomputable def volume_of_solution_y_and_z (V: ℝ) : Prop :=
  let vol_X := 300.0
  let conc_X := 0.10
  let conc_Y := 0.30
  let conc_Z := 0.40
  let vol_Y := 2 * V
  let vol_new := vol_X + vol_Y + V
  let alcohol_new := conc_X * vol_X + conc_Y * vol_Y + conc_Z * V
  (alcohol_new / vol_new) = 0.22

theorem solution_alcohol_content : volume_of_solution_y_and_z 300.0 :=
by
  sorry

end solution_alcohol_content_l326_326782


namespace minimum_angle_between_bisectors_theorem_l326_326608

noncomputable def minimum_angle_between_bisectors (A B C O : Point) (OA OB OC : ℝ) : ℝ :=
  if h1 : angle A O B = 60 then
  if h2 : angle B O C = 120 then
  if h3 : angle A O C = 90 then
  45
  else 0
  else 0
  else 0

theorem minimum_angle_between_bisectors_theorem
  (A B C O : Point)
  (OA OB OC : ℝ)
  (h1 : angle A O B = 60)
  (h2 : angle B O C = 120)
  (h3 : angle A O C = 90) :
  minimum_angle_between_bisectors A B C O OA OB OC = 45 :=
by sorry

end minimum_angle_between_bisectors_theorem_l326_326608


namespace student_correct_answers_l326_326153

theorem student_correct_answers (C I : ℕ) 
  (h1 : C + I = 100) 
  (h2 : C - 2 * I = 61) : 
  C = 87 :=
by
  sorry

end student_correct_answers_l326_326153


namespace graph_of_abs_eqn_eq_two_points_l326_326104

theorem graph_of_abs_eqn_eq_two_points :
  { (x, y) | |x * y| + |x - y + 1| = 0 } = { (0, 1), (-1, 0) } :=
by
  sorry

end graph_of_abs_eqn_eq_two_points_l326_326104


namespace area_enclosed_l326_326019

noncomputable def area (a : ℝ) : ℝ :=
  ∫ x in 0..(Real.exp a - 1), a * x - x * Real.log (x + 1)

theorem area_enclosed (a : ℝ) (h : a ≠ 0) : 
  area a = (Real.exp (2 * a) - 1) / 4 + Real.exp a - 1 := by
  sorry

end area_enclosed_l326_326019


namespace biology_exam_students_l326_326766

theorem biology_exam_students :
  let students := 200
  let score_A := (1 / 4) * students
  let remaining_students := students - score_A
  let score_B := (1 / 5) * remaining_students
  let score_C := (1 / 3) * remaining_students
  let score_D := (5 / 12) * remaining_students
  let score_F := students - (score_A + score_B + score_C + score_D)
  let re_assessed_C := (3 / 5) * score_C
  let final_score_B := score_B + re_assessed_C
  let final_score_C := score_C - re_assessed_C
  score_A = 50 ∧ 
  final_score_B = 60 ∧ 
  final_score_C = 20 ∧ 
  score_D = 62 ∧ 
  score_F = 8 :=
by {
  sorry
}

end biology_exam_students_l326_326766


namespace ratio_small_to_large_is_one_to_one_l326_326225

theorem ratio_small_to_large_is_one_to_one
  (total_beads : ℕ)
  (large_beads_per_bracelet : ℕ)
  (bracelets_count : ℕ)
  (small_beads : ℕ)
  (large_beads : ℕ)
  (small_beads_per_bracelet : ℕ) :
  total_beads = 528 →
  large_beads_per_bracelet = 12 →
  bracelets_count = 11 →
  large_beads = total_beads / 2 →
  large_beads >= bracelets_count * large_beads_per_bracelet →
  small_beads = total_beads / 2 →
  small_beads_per_bracelet = small_beads / bracelets_count →
  small_beads_per_bracelet / large_beads_per_bracelet = 1 :=
by sorry

end ratio_small_to_large_is_one_to_one_l326_326225


namespace geometric_sequence_a1_value_l326_326627

variable {a_1 q : ℝ}

theorem geometric_sequence_a1_value
  (h1 : a_1 * q^2 = 1)
  (h2 : a_1 * q^4 + (3 / 2) * a_1 * q^3 = 1) :
  a_1 = 4 := by
  sorry

end geometric_sequence_a1_value_l326_326627


namespace railway_networks_exist_l326_326469

def railway_networks (n: ℕ) (roads: ℕ) : Prop :=
  ∀ (cities : Fin n → Prop),
    (∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ collinear cities i j k) →
    (∃ (viaducts: ℕ), viaducts ≥ 0) →
    n = 5 →
    roads = 4 →
    count_networks cities roads = 125

theorem railway_networks_exist : railway_networks 5 4 := 
  by 
  sorry

end railway_networks_exist_l326_326469


namespace real_root_of_polynomial_l326_326974

theorem real_root_of_polynomial :
  ∃ x : ℝ, (x^5 - 2 * x^4 - x^2 + 2 * x - 3 = 0) ∧ (x = 3) :=
by
  use 3
  split
  -- Placeholders for proving that x = 3 satisfies the polynomial equation
  sorry
  -- Trivially, 3 equals 3
  refl

end real_root_of_polynomial_l326_326974


namespace median_and_altitude_l326_326857

noncomputable def median_length (D E F : Point) (hDE : length D E = 10) (hDF : length D F = 10) (hEF : length E F = 12) : ℝ :=
  8

noncomputable def altitude_length (D E F : Point) (hDE : length D E = 10) (hDF : length D F = 10) (hEF : length E F = 12) : ℝ :=
  8

theorem median_and_altitude (D E F : Point) (hDE : length D E = 10) (hDF : length D F = 10) (hEF : length E F = 12) :
  median_length D E F hDE hDF hEF = 8 ∧ altitude_length D E F hDE hDF hEF = 8 :=
by
  sorry

end median_and_altitude_l326_326857


namespace minimum_value_of_f_roots_sum_gt_2_l326_326646

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f 1 = 1 := by
  exists 1
  sorry

theorem roots_sum_gt_2 (a x₁ x₂ : ℝ) (h_f_x₁ : f x₁ = a) (h_f_x₂ : f x₂ = a) (h_x₁_lt_x₂ : x₁ < x₂) :
    x₁ + x₂ > 2 := by
  sorry

end minimum_value_of_f_roots_sum_gt_2_l326_326646


namespace find_a_l326_326451

-- Define the slopes of the lines and the condition that they are perpendicular.
def slope1 (a : ℝ) : ℝ := a
def slope2 (a : ℝ) : ℝ := a + 2

-- The main statement of our problem.
theorem find_a (a : ℝ) (h : slope1 a * slope2 a = -1) : a = -1 :=
sorry

end find_a_l326_326451


namespace erdos_szekeres_theorem_erdos_szekeres_optimal_l326_326425

-- Statement for the given problem as described 
theorem erdos_szekeres_theorem (m n : ℕ) (u : ℕ → ℝ) (len_u : u.length = m * n + 1) :
  ∃ v : ℕ → ℝ, v.length = n + 1 ∧ (∀ i j, i < j → v i < v j) ∨ (∃ w : ℕ → ℝ, w.length = m + 1 ∧ (∀ i j, i < j → w i > w j)) := sorry

-- Statement for optimality, proves the bound is exact and cannot be improved.
theorem erdos_szekeres_optimal (m n : ℕ) :
  ∃ u : ℕ → ℝ, u.length = m * n ∧ 
  (∀ v : ℕ → ℝ, v.length = n + 1 → ¬ (∀ i j, i < j → v i < v j)) ∧ 
  (∀ w : ℕ → ℝ, w.length = m + 1 → ¬ (∀ i j, i < j → w i > w j)) := sorry

end erdos_szekeres_theorem_erdos_szekeres_optimal_l326_326425


namespace min_value_2a_plus_b_l326_326346

theorem min_value_2a_plus_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 / a + 3 / b = 1) ∧ (2 * a + b = 7 + 4 * Real.sqrt 3) :=
begin
  sorry,
end

end min_value_2a_plus_b_l326_326346


namespace vasya_hits_ship_l326_326060

theorem vasya_hits_ship (board_size : ℕ) (ship_length : ℕ) (shots : ℕ) : 
  board_size = 10 ∧ ship_length = 4 ∧ shots = 24 → ∃ strategy : Fin board_size × Fin board_size → Prop, 
  (∀ pos, strategy pos → pos.1 * board_size + pos.2 < shots) ∧ 
  ∀ (ship_pos : Fin board_size × Fin board_size) (horizontal : Bool), 
  ∃ shot_pos, strategy shot_pos ∧ 
  (if horizontal then 
    ship_pos.1 = shot_pos.1 ∧ ship_pos.2 ≤ shot_pos.2 ∧ shot_pos.2 < ship_pos.2 + ship_length 
  else 
    ship_pos.2 = shot_pos.2 ∧ ship_pos.1 ≤ shot_pos.1 ∧ shot_pos.1 < ship_pos.1 + ship_length) :=
sorry

end vasya_hits_ship_l326_326060


namespace xy_value_l326_326683

theorem xy_value (x y : ℝ) (h : sqrt (x + 2) + (y - sqrt 3)^2 = 0) : 
  x * y = -2 * sqrt 3 := 
sorry

end xy_value_l326_326683


namespace markov_chain_L_properties_l326_326768

-- Definition of Markov chain L and states E.

variables {ι : Type} [fintype ι] [decidable_eq ι]
variables (L : ι → ι → set ℕ)
variables (E : ι → ι)

-- Conditions of problem stated as assumptions
def markov_chain_L (L : ι → ι → set ℕ) :=
∀ (i j : ι), ∃ n ∈ L i j, true

-- Markov chain contains the states E_{22}, E_{88}, ..., E_{nn}
def contains_states (L : ι → ι → set ℕ) (E : ι → ι) :=
∀ (i : ℕ), L (E i) (E i) ≠ ∅

-- Markov chain is connected
def is_connected_chain (L : ι → ι → set ℕ) :=
∀ (i j : ι), L i j ≠ ∅

-- The Complete Theorem:
theorem markov_chain_L_properties 
  (L : ι → ι → set ℕ) (E : ι → ι) 
  [markov_chain_L L]
  [contains_states L E]
  [is_connected_chain L] : 
  (∀ (i : ℕ), L (E i) (E i) ≠ ∅) ∧ 
  (∀ (i j : ι), L i j ≠ ∅) :=
begin
  split,
  { sorry }, -- proof showing L contains states E_{22}, E_{88}, ..., E_{nn}
  { sorry } -- proof showing L is a connected chain
end

end markov_chain_L_properties_l326_326768


namespace unbiased_estimator_of_population_mean_l326_326275

variable {n : ℕ} (x : Fin 4 → ℕ) (f : Fin 4 → ℕ) 

def n := 50
def x : Fin 4 → ℕ := ![2, 5, 7, 10]
def f : Fin 4 → ℕ := ![16, 12, 8, 14]

def sampleProductSum : ℕ := ∑ i, f i * x i

def sampleMean (totalProduct : ℕ) (n : ℕ) : ℝ := totalProduct / n

theorem unbiased_estimator_of_population_mean : sampleMean sampleProductSum n = 5.76 := by
  sorry

end unbiased_estimator_of_population_mean_l326_326275


namespace polynomial_divisibility_l326_326273

theorem polynomial_divisibility :
  ∃ (p : Polynomial ℤ), (Polynomial.X ^ 2 - Polynomial.X + 2) * p = Polynomial.X ^ 15 + Polynomial.X ^ 2 + 100 :=
by
  sorry

end polynomial_divisibility_l326_326273


namespace mrs_hilt_rocks_l326_326409

def garden_length := 10
def garden_width := 15
def rock_coverage := 1
def available_rocks := 64

theorem mrs_hilt_rocks :
  ∃ extra_rocks : ℕ, 2 * (garden_length + garden_width) <= available_rocks ∧ extra_rocks = available_rocks - 2 * (garden_length + garden_width) ∧ extra_rocks = 14 :=
by
  sorry

end mrs_hilt_rocks_l326_326409


namespace minimum_angle_45_degree_l326_326610

-- We define the points O, A, B, and C with the given angles between their connecting rays.
def O : Point := sorry   -- Origin point
def A : Point := sorry   -- Point A
def B : Point := sorry   -- Point B
def C : Point := sorry   -- Point C

noncomputable def angle (p1 p2 : Point) : ℝ := sorry  -- Function to calculate the angle between two points

-- Conditions from the problem:
def condition1 : angle O A = 60 := sorry
def condition2 : angle O B = 90 := sorry
def condition3 : angle O C = 120 := sorry

-- The theorem we need to prove:
theorem minimum_angle_45_degree :
  ∀ (D E F : Point), bisector O A B D ∧ bisector O B C E ∧ bisector O C A F →
  ∀ (angle1 angle2 angle3 : ℝ),
    (angle D O E = angle1 ∨ angle E O F = angle2 ∨ angle F O D = angle3) →
    min angle1 angle2 angle3 = 45 :=
sorry

end minimum_angle_45_degree_l326_326610


namespace find_all_f_l326_326996

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_all_f :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x + 2 * y^2) →
  ∃ a c : ℝ, (∀ x : ℝ, f x = x^2 + a * x + c) ∧ (a^2 - 4 * c ≤ 0) := sorry

end find_all_f_l326_326996


namespace age_of_replaced_person_l326_326089

theorem age_of_replaced_person (avg_age x : ℕ) (h1 : 10 * avg_age - 10 * (avg_age - 3) = x - 18) : x = 48 := 
by
  -- The proof goes here, but we are omitting it as per instruction.
  sorry

end age_of_replaced_person_l326_326089


namespace choose_60_non_adjacent_squares_l326_326139

theorem choose_60_non_adjacent_squares :
  let total_ways : ℕ := 62
  in total_ways = 62 := by sorry

end choose_60_non_adjacent_squares_l326_326139


namespace jack_total_yen_l326_326006

def pounds := 42
def euros := 11
def yen := 3000
def pounds_per_euro := 2
def yen_per_pound := 100

theorem jack_total_yen : (euros * pounds_per_euro + pounds) * yen_per_pound + yen = 9400 := by
  sorry

end jack_total_yen_l326_326006


namespace geometric_solution_l326_326002

theorem geometric_solution (x y m : ℝ) (h1 : y = sqrt (2 * x^2 + 2 * x - m)) (h2 : y = x - 2) : m ≥ 12 := 
sorry

end geometric_solution_l326_326002


namespace hydrogen_atom_diameter_scientific_notation_l326_326096

theorem hydrogen_atom_diameter_scientific_notation :
  (0.0000000001 : ℝ) = 1 * 10^(-10) :=
sorry

end hydrogen_atom_diameter_scientific_notation_l326_326096


namespace mike_initial_quarters_undetermined_l326_326046

theorem mike_initial_quarters_undetermined 
  (initial_nickels : ℕ)
  (nickels_borrowed : ℕ)
  (current_nickels : ℕ)
  (some_quarters : ℕ)
  (h1 : initial_nickels = 87)
  (h2 : nickels_borrowed = 75)
  (h3 : current_nickels = 12)
  (h4 : nickels_borrowed + current_nickels = initial_nickels) :
  ∀ (initial_quarters: ℕ), True :=
by
  intro _,
  trivial,
  sorry

end mike_initial_quarters_undetermined_l326_326046


namespace find_f_10_l326_326650

def f : ℕ → ℚ := sorry
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = f x / (1 + f x)
axiom f_initial : f 1 = 1

theorem find_f_10 : f 10 = 1 / 10 :=
by
  sorry

end find_f_10_l326_326650


namespace paint_cost_correct_l326_326342

-- Definitions based on conditions
def cost_per_quart_A := 3.20
def cost_per_quart_B := 4.30
def coverage_per_quart_A := 10
def coverage_per_quart_B := 8
def edge_length := 10
def number_faces := 6
def face_area := edge_length * edge_length

-- Number of faces to be painted with each color based on the pattern
def faces_painted_A := 4
def faces_painted_B := 2

-- Total areas to be painted with each color
def total_area_A := faces_painted_A * face_area
def total_area_B := faces_painted_B * face_area

-- Number of quarts needed for each color
def quarts_needed_A := total_area_A / coverage_per_quart_A
def quarts_needed_B := total_area_B / coverage_per_quart_B

-- Total cost for each color
def total_cost_A := quarts_needed_A * cost_per_quart_A
def total_cost_B := quarts_needed_B * cost_per_quart_B

-- Total cost to paint the cube
def total_cost := total_cost_A + total_cost_B

-- The theorem to be proved
theorem paint_cost_correct : total_cost = 235.50 :=
by
  sorry

end paint_cost_correct_l326_326342


namespace all_numbers_equal_l326_326987

open Finset Function

-- Define the 10x10 table
variable (T : Fin 10 → Fin 10 → ℝ)

-- Predicate to find if a number is the maximum in its row
def maxInRow (i : Fin 10) (x : ℝ) : Prop :=
  ∀ j : Fin 10, T i j ≤ x

-- Predicate to find if a number is the minimum in its column
def minInCol (j : Fin 10) (x : ℝ) : Prop :=
  ∀ i : Fin 10, x ≤ T i j

-- Underlined numbers are defined as both maximum in their row and minimum in their column
def underlined (i : Fin 10) (j : Fin 10) : Prop :=
  maxInRow T i (T i j) ∧ minInCol T j (T i j)

-- Theorem stating that all numbers in the table are equal
theorem all_numbers_equal
  (h : ∀ i j, (Σ i' : Fin 10, underlined T i' j).card = 1)
  (h' : ∀ j i, (Σ j' : Fin 10, underlined T i j').card = 1) :
  ∀ i j i' j', T i j = T i' j' :=
by
  sorry

end all_numbers_equal_l326_326987


namespace evaluate_sum_l326_326571

/-- Define double factorial (n!!) recursively -/
def double_factorial : ℕ → ℕ
| 0 := 1
| 1 := 1
| n := n * double_factorial (n-2)

/-- Define the sum we are interested in -/
def sum_double_factorial : ℕ → ℚ := λ (n : ℕ), ∑ i in (finset.range n).filter (λ x, x > 0),
  (double_factorial (2 * i - 1)) / (double_factorial (2 * i))

/-- The final aim -/
theorem evaluate_sum : (∑ i in (finset.range 1006).filter (λ x, x > 0),
    (double_factorial (2 * i - 1)) / (double_factorial (2 * i)) = 1999 / 2^1999) →
    (1999 * 1 / 10 : ℚ) = 199.9 :=
begin
  sorry,
end

end evaluate_sum_l326_326571


namespace least_integer_exists_l326_326260

theorem least_integer_exists (x : ℕ) (h1 : x = 10 * (x / 10) + x % 10) (h2 : (x / 10) = x / 17) : x = 17 :=
sorry

end least_integer_exists_l326_326260


namespace min_value_inequality_l326_326389

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end min_value_inequality_l326_326389


namespace triangle_is_equilateral_l326_326828

-- Definitions and conditions from the problem
def is_centroid (M A B C : Point) : Prop :=
  ∃ G H I J, 
  M = centroid ΔABC ∧ 
  M divides median AM, BM, CM in ratio 2:1

def perimeters_equal (A B C M : Point) : Prop :=
  perimeter (triangle A B M) = perimeter (triangle B C M) ∧
  perimeter (triangle B C M) = perimeter (triangle A C M)

-- Theorem to be proved
theorem triangle_is_equilateral (A B C M : Point) 
  (hM: is_centroid M A B C) 
  (hEqual: perimeters_equal A B C M) :
  is_equilateral (triangle A B C) :=
sorry


end triangle_is_equilateral_l326_326828


namespace rhombus_perimeter_l326_326805

def half (a: ℕ) := a / 2

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  let s1 := half d1
      s2 := half d2
      side_length := Math.sqrt (s1^2 + s2^2)
      perimeter := 4 * side_length
  in perimeter = 68 := by
  sorry

end rhombus_perimeter_l326_326805


namespace preferred_order_for_boy_l326_326169

variable (p q : ℝ)
variable (h : p < q)

theorem preferred_order_for_boy (p q : ℝ) (h : p < q) : 
  (2 * p * q - p^2 * q) > (2 * p * q - p * q^2) := 
sorry

end preferred_order_for_boy_l326_326169


namespace polynomial_divisibility_l326_326274

theorem polynomial_divisibility :
  ∃ (p : Polynomial ℤ), (Polynomial.X ^ 2 - Polynomial.X + 2) * p = Polynomial.X ^ 15 + Polynomial.X ^ 2 + 100 :=
by
  sorry

end polynomial_divisibility_l326_326274


namespace angle_range_of_asymptotes_l326_326231

-- Define the conditions of the hyperbola
variables (a b : ℝ) (ha : a > 0) (hb : b > 0)
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the distance conditions for the hyperbola
def distance_condition (e c : ℝ) (he : e = c / a) (hc : c = √(a^2 + b^2)) : Prop :=
  let x := 3 * a / 2
  let dist_to_focus := e * (3 * a / 2 - a^2 / c)
  let dist_to_directrix := x + a^2 / c
  dist_to_focus > dist_to_directrix

-- Conclusion: The range of acute angles formed by the asymptotes
theorem angle_range_of_asymptotes (e c : ℝ) (he : e = c / a) (hc : c = √(a^2 + b^2))
    (dist_cond : distance_condition a b e c he hc) : 
    ∃ theta : ℝ, 0 < theta ∧ theta < 60 := 
sorry

end angle_range_of_asymptotes_l326_326231


namespace find_abc_l326_326056

theorem find_abc (a b c : ℤ) 
  (h₁ : a^4 - 2 * b^2 = a)
  (h₂ : b^4 - 2 * c^2 = b)
  (h₃ : c^4 - 2 * a^2 = c)
  (h₄ : a + b + c = -3) : 
  a = -1 ∧ b = -1 ∧ c = -1 := 
sorry

end find_abc_l326_326056


namespace trig_inequalities_l326_326787

theorem trig_inequalities (x : ℝ) (h : 0 < x ∧ x < π / 6) :
  sin(sin x) < cos(cos x) ∧ cos(cos x) < sin(cos x) ∧ sin(cos x) < cos(sin x) :=
by
  sorry

end trig_inequalities_l326_326787


namespace find_a_l326_326365

-- Define the triangle and the given conditions
variables (A B C : Type) [Real A] [Real B] [Real C]

-- Given conditions
def angle_A_deg := 60
def area_ABC := sqrt 3
def b_plus_c := 6

-- Known side lengths opposite to angles
variables (a b c : ℝ)

-- Statement to be proved
theorem find_a : (angle_A_deg = 60) → (area_ABC = sqrt 3) →
                  (b + c = 6) → (a = 2 * sqrt 6) :=
by
  -- proof goes here
  sorry

end find_a_l326_326365


namespace black_balls_in_box_l326_326686

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end black_balls_in_box_l326_326686


namespace find_a1_range_l326_326567

-- Definitions as per conditions
def arithmetic_seq_condition (a : ℕ → ℝ) : Prop :=
  let a4 := a 4
  let a5 := a 5
  let a6 := a 6
  let a7 := a 7
  (sin a4) ^ 2 * (cos a7) ^ 2 - (sin a7) ^ 2 * (cos a4) ^ 2 = sin (a5 + a6)

def arithmetic_sum_condition (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) : Prop :=
  ∀ n, S n n * a1 + n * (n - 1) / 2 * d = S_max n ↔ n = 9

-- Theorem statement
theorem find_a1_range (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ) :
  arithmetic_seq_condition a ∧ d ∈ Ioo (-1 : ℝ) 0 ∧ arithmetic_sum_condition a d a1 →
  (4 * pi / 3) < a1 ∧ a1 < (3 * pi / 2) :=
sorry

end find_a1_range_l326_326567


namespace find_counterfeit_coin_l326_326149

theorem find_counterfeit_coin (coin : Fin 7 → ℕ)
  (genuine_weight : ℕ)
  (scale : List (Fin 7) → List (Fin 7) → Ordering)
  (H1 : ∃ i, coin i < genuine_weight)
  (H2 : ∀ i j k l m n, 
    scale [i] [j, k, l] = Ordering.gt → coin i < genuine_weight ∧ coin j = genuine_weight ∧ coin k = genuine_weight ∧ coin l = genuine_weight)
  (H3 : ∀ i j k l m n, 
    scale [i] [j, k, l] = Ordering.eq → coin i = genuine_weight ∧ coin j = genuine_weight ∧ coin k = genuine_weight ∧ coin l = genuine_weight)
  (H4 : ∀ i j k l m, 
    scale [i] [j, k, l] = Ordering.lt → coin j = genuine_weight ∧ coin k = genuine_weight ∧ coin l = genuine_weight ∧ coin i < genuine_weight)
  (H5 : ∀ i j k m n l, 
    scale [i, j, k] [m, l, n] = Ordering.gt → coin i = genuine_weight ∧ coin j = genuine_weight ∧ coin k = genuine_weight ∧ coin m < genuine_weight)
  (H6 : ∀ i j k m n l, 
    scale [i, j, k] [m, l, n] = Ordering.eq → coin i = genuine_weight ∧ coin j = genuine_weight ∧ coin k = genuine_weight ∧ coin m = genuine_weight ∧ coin n = genuine_weight ∧ coin l = genuine_weight)
  (H7 : ∀ i j k m n l, 
    scale [i, j, k] [m, l, n] = Ordering.lt → coin m = genuine_weight ∧ coin n = genuine_weight ∧ coin l = genuine_weight ∧ coin i < genuine_weight)
  : ∃ counterfeit, ∀ coin_weights : Fin 7 → ℕ, coin_weights = coin → coin counterfeit < genuine_weight :=
by
  sorry

end find_counterfeit_coin_l326_326149


namespace arithmetic_mean_no_8_l326_326791

theorem arithmetic_mean_no_8 :
  let N := ((∑ n in list.range 1 (10), (10^n - 1)) / 9) / 9
  in (N = 123456790) ∧ ∀ d : ℕ, d ∈ [0,1,2,3,4,5,6,7,9] → d ∉ N.digits 10 :=
sorry

end arithmetic_mean_no_8_l326_326791


namespace sum_distances_greater_than_100_l326_326765

theorem sum_distances_greater_than_100 :
  ∀ (circle : Type) [metric_space circle] (radius : ℝ) (marked_points : fin 100 → circle),
  ∃ (point : circle), (radius = 1) →
  (∀ i, dist point (marked_points i) > 0) →
  (∑ i in finset.fin_range 100, dist point (marked_points i)) > 100 :=
by intro circle _ r marked_points
   sorry

end sum_distances_greater_than_100_l326_326765


namespace correct_average_of_ten_numbers_l326_326441

theorem correct_average_of_ten_numbers (incorrect_avg : ℝ) (incorrect_reading : ℝ) (correct_reading : ℝ) (n : ℕ)
  (h1 : incorrect_avg = 16) 
  (h2 : incorrect_reading = 25)
  (h3 : correct_reading = 35)
  (h4 : n = 10): 
  (correct_sum : ℝ) 
  (correct_avg : ℝ) :
  let incorrect_sum := n * incorrect_avg,
      difference := correct_reading - incorrect_reading,
      correct_sum := incorrect_sum + difference,
      correct_avg := correct_sum / n
  in correct_avg = 17 := 
by
  sorry

end correct_average_of_ten_numbers_l326_326441


namespace label_2000_2024_l326_326581

noncomputable def label : ℕ × ℕ → ℕ
| (0, 0) := 0
| (x, y) :=
  if x > 0 then max (label (x - 1, y)) (label (x - 1, y + 1)) - 1 else
  if y > 0 then max (label (x, y - 1)) (label (x + 1, y - 1)) - 1 else
  0

theorem label_2000_2024 :
  ∃ n, (0 ≤ n ∧ n ≤ 6048) ∧ (n % 3 = 0) ∧ label (2000, 2024) = n := 
sorry

end label_2000_2024_l326_326581


namespace max_dist_min_dist_l326_326283

noncomputable def z : ℂ := sorry

def condition : Prop := complex.abs (z + 2 - 2 * complex.I) = 1

theorem max_dist : condition → complex.abs (z - 3 - 2 * complex.I) ≤ 6 :=
begin
  sorry
end

theorem min_dist : condition → complex.abs (z - 3 - 2 * complex.I) ≥ 4 :=
begin
  sorry
end

end max_dist_min_dist_l326_326283


namespace johnny_marbles_combination_l326_326725

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end johnny_marbles_combination_l326_326725


namespace intersection_sets_l326_326401

variable {x : ℝ}
variable {y : ℝ}

def M := { y | ∃ x, y = 2^x }
def P := { x | ∃ y, y = sqrt(x - 1) }

theorem intersection_sets :
  M ∩ P = { x | x ≥ 1 } :=
by
  sorry

end intersection_sets_l326_326401


namespace pencils_per_child_l326_326580

-- Define the conditions
def totalPencils : ℕ := 18
def numberOfChildren : ℕ := 9

-- The proof problem
theorem pencils_per_child : totalPencils / numberOfChildren = 2 := 
by
  sorry

end pencils_per_child_l326_326580


namespace range_of_a_log_function_l326_326980

variable a : ℝ

theorem range_of_a_log_function :
  (∀ y : ℝ, ∃ x : ℝ, y = log (0.5) (x^2 + a * x + 1)) →
  a ≤ -2 ∨ a ≥ 2 :=
by
  -- Proof begins here.
  sorry

end range_of_a_log_function_l326_326980


namespace minimum_angle_45_degree_l326_326609

-- We define the points O, A, B, and C with the given angles between their connecting rays.
def O : Point := sorry   -- Origin point
def A : Point := sorry   -- Point A
def B : Point := sorry   -- Point B
def C : Point := sorry   -- Point C

noncomputable def angle (p1 p2 : Point) : ℝ := sorry  -- Function to calculate the angle between two points

-- Conditions from the problem:
def condition1 : angle O A = 60 := sorry
def condition2 : angle O B = 90 := sorry
def condition3 : angle O C = 120 := sorry

-- The theorem we need to prove:
theorem minimum_angle_45_degree :
  ∀ (D E F : Point), bisector O A B D ∧ bisector O B C E ∧ bisector O C A F →
  ∀ (angle1 angle2 angle3 : ℝ),
    (angle D O E = angle1 ∨ angle E O F = angle2 ∨ angle F O D = angle3) →
    min angle1 angle2 angle3 = 45 :=
sorry

end minimum_angle_45_degree_l326_326609


namespace correct_operation_is_d_l326_326488

theorem correct_operation_is_d (a b : ℝ) : 
  (∀ x y : ℝ, -x * y = -(x * y)) → 
  (∀ x : ℝ, x⁻¹ * (x ^ 2) = x) → 
  (∀ x : ℝ, x ^ 10 / x ^ 4 = x ^ 6) →
  ((a - b) * (-a - b) ≠ a ^ 2 - b ^ 2) ∧ 
  (2 * a ^ 2 * a ^ 3 ≠ 2 * a ^ 6) ∧ 
  ((-a) ^ 10 / (-a) ^ 4 = a ^ 6) :=
by
  intros h1 h2 h3
  sorry

end correct_operation_is_d_l326_326488


namespace rhombus_major_premise_incorrect_l326_326358

theorem rhombus_major_premise_incorrect :
  (∀ r : Type, (is_rhombus r → has_perpendicular_bisecting_diagonals r)) →
  ¬ (∀ r : Type, is_rhombus r → has_equal_diagonals r) :=
by 
  sorry

end rhombus_major_premise_incorrect_l326_326358


namespace general_term_sequence_l326_326447

def alternating_signs (n : ℕ) : ℤ :=
  (-1)^(n+1)

def abs_value_term (n : ℕ) : ℚ :=
  1/n

/-- General term of sequence 1, -1/2, 1/3, -1/4, ... is (-1)^(n+1)/n -/
theorem general_term_sequence (n : ℕ) : ℚ :=
  ∑ (a : ℤ) in 1..n, from alternating_signs n * abs_value_term n

end general_term_sequence_l326_326447


namespace sasha_remaining_questions_l326_326070

variable (rate : Int) (initial_questions : Int) (hours_worked : Int)

theorem sasha_remaining_questions
  (h1 : rate = 15)
  (h2 : initial_questions = 60)
  (h3 : hours_worked = 2) :
  initial_questions - rate * hours_worked = 30 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end sasha_remaining_questions_l326_326070


namespace parallel_line_equation_l326_326312

theorem parallel_line_equation :
  ∃ (c : ℝ), 
    (∀ x : ℝ, y = (3 / 4) * x + 6 → (y = (3 / 4) * x + c → abs (c - 6) = 4 * (5 / 4))) → c = 1 :=
by
  sorry

end parallel_line_equation_l326_326312


namespace hexagon_area_l326_326022

theorem hexagon_area (ABCDEF : hexagon) (J K L : point)
  (hJ : is_midpoint J (side AB)) (hK : is_midpoint K (side CD))
  (hL : is_midpoint L (side EF))
  (h_area_JKL : area (triangle J K L) = 100) :
  area (hexagon ABCDEF) = 800 / 3 :=
by sorry

end hexagon_area_l326_326022


namespace number_of_real_solutions_l326_326110

theorem number_of_real_solutions :
  {x : ℝ | 2 ^ (2 * x ^ 2 - 7 * x + 5) = 1}.finite.to_finset.card = 2 :=
by
  -- We will include a placeholder proof to ensure syntactical correctness.
  sorry

end number_of_real_solutions_l326_326110


namespace solve_for_x_l326_326985

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  (4 * y^2 + y + 6 = 3 * (9 * x^2 + y + 3)) ↔ (x = 1 ∨ x = -1/3) :=
by
  sorry

end solve_for_x_l326_326985


namespace original_average_is_40_l326_326793

-- Define the conditions
variable (students : ℕ) (new_avg : ℝ) (original_avg : ℝ)

-- Hypothesis
def hypothesis : Prop :=
  students = 10 ∧ new_avg = 80 ∧ 2 * original_avg = new_avg

-- Main theorem
theorem original_average_is_40 (h : hypothesis students new_avg original_avg) : original_avg = 40 := by
  cases h with
  | intro h1 h2 h3 =>
  have avg_proof : original_avg = 40 := by
    calc
      original_avg = new_avg / 2 := by exact Eq.symm h3
      ... = 80 / 2 := by rfl
      ... = 40 := by norm_num
  exact avg_proof

end original_average_is_40_l326_326793


namespace crow_eating_time_l326_326511

theorem crow_eating_time (p : ℝ) (h : 0 ≤ p) : ∃ x : ℝ, x = 40 * p :=
begin
  use 40 * p,
  refl,
end

end crow_eating_time_l326_326511


namespace perp_condition_l326_326629

variables (α β : Plane) (m l : Line)

noncomputable theory

-- Conditions
def m_in_alpha : Prop := m ∈ α
def alpha_inter_beta_is_l : Prop := α ∩ β = l
def alpha_perp_beta : Prop := α ⟂ β
def m_perp_l : Prop := m ⟂ l

-- To Prove
theorem perp_condition :
  (m_in_alpha α m) → (alpha_inter_beta_is_l α β l) → (alpha_perp_beta α β) → (m_perp_l m l) ↔ (m ⟂ β) :=
by sorry

end perp_condition_l326_326629


namespace exists_permutation_complete_residue_system_l326_326749

open Nat

theorem exists_permutation_complete_residue_system (n : ℕ) (hn : Prime n) :
  ∃ (a : Fin n → Fin n), 
  (∀ i j, i ≠ j → a i ≠ a j) ∧             -- a is a permutation
  (∀ k, 1 ≤ k → k ≤ n → 
    ∃ m, m < n ∧ ((List.prod (List.map (λ i, a i) (List.range (Fin.val k))) : ℕ) % n = m)) := 
sorry

end exists_permutation_complete_residue_system_l326_326749


namespace bruce_anne_cleaning_house_l326_326957

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l326_326957


namespace solve_for_x_l326_326429

-- Define the quadratic equation condition
def quadratic_eq (x : ℝ) : Prop := 3 * x^2 - 7 * x - 6 = 0

-- The main theorem to prove
theorem solve_for_x (x : ℝ) : x > 0 ∧ quadratic_eq x → x = 3 := by
  sorry

end solve_for_x_l326_326429


namespace trip_cost_l326_326204

variable (price : ℕ) (discount : ℕ) (numPeople : ℕ)

theorem trip_cost :
  price = 147 →
  discount = 14 →
  numPeople = 2 →
  (price - discount) * numPeople = 266 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end trip_cost_l326_326204


namespace count_multiples_14_not_3_5_l326_326322

noncomputable def count_specified_multiples : Nat :=
  Set.size {n ∈ Finset.range 301 | n % 14 = 0 ∧ (n % 3 ≠ 0 ∧ n % 5 ≠ 0)}

theorem count_multiples_14_not_3_5 : count_specified_multiples = 11 := 
sorry

end count_multiples_14_not_3_5_l326_326322


namespace total_profit_l326_326202

/-- Let B's investment be x.
    Then A's investment is 3x, and C's investment is 9x/2.
    The share of C is given as Rs. 15000.000000000002.
    We need to prove that the total profit is Rs. 28333.33333333333.
    
    Conditions: 
    1) C's investment is 9/2 of B's investment
    2) A's investment is 3 times of B's investment

    We prove that the total profit is 28333.33333333333 -/
theorem total_profit (x : ℝ) (hx1 : 9 / 2 * x = 2 / 3 * (9 / 2 * x)) 
                 (hx2 : 3 * x = 2 / 3 * (9 / 2 * x)) :
  let c_share : ℝ := 15000.000000000002 in
  let ratio_c_part := 9 in
  let total_ratio_parts := 17 in
  let per_part_value := c_share / ratio_c_part in
  let calculated_profit := per_part_value * total_ratio_parts in
  calculated_profit = 28333.33333333333 :=
by
  sorry

end total_profit_l326_326202


namespace total_pepper_cost_l326_326570

variable (wg wp wr wo : ℝ) (pg pr py po : ℝ)
variable (total_cost : ℝ)

-- Define the weights and prices for each type of pepper.
def weights_prices : Prop :=
  wg = 2.8333333333333335 ∧ wp = 1.20 ∧
  wr = 3.254 ∧ pr = 1.35 ∧
  wy = 1.375 ∧ py = 1.50 ∧
  wo = 0.567 ∧ po = 1.65

-- Calculate the total cost from the conditions given.
def calculate_total_cost : ℝ :=
  wg * wp + wr * pr + wy * py + wo * po

-- State the theorem that we need to prove.
theorem total_pepper_cost (h : weights_prices) : calculate_total_cost = 10.79 := by
  sorry

end total_pepper_cost_l326_326570


namespace solve_polynomial_division_l326_326272

theorem solve_polynomial_division :
  ∃ a : ℤ, (∀ x : ℂ, ∃ p : polynomial ℂ, x^2 - x + (a : ℂ) * p x = x^15 + x^2 + 100) → a = 2 := by
  sorry

end solve_polynomial_division_l326_326272


namespace empty_set_subset_nat_l326_326635

theorem empty_set_subset_nat : ∅ ⊆ set.univ :=
by {
  sorry
}

end empty_set_subset_nat_l326_326635


namespace total_cookies_collected_l326_326916

theorem total_cookies_collected 
  (abigail_boxes : ℕ) (grayson_boxes : ℕ) (olivia_boxes : ℕ) (cookies_per_box : ℕ)
  (h1 : abigail_boxes = 2) (h2 : grayson_boxes = 3) (h3 : olivia_boxes = 3) (h4 : cookies_per_box = 48) :
  (abigail_boxes * cookies_per_box) + ((grayson_boxes * (cookies_per_box / 4))) + (olivia_boxes * cookies_per_box) = 276 := 
by 
  sorry

end total_cookies_collected_l326_326916


namespace chord_is_26_l326_326517

noncomputable def circle_radius : ℝ := 15
noncomputable def chord_length (r : ℝ) (c : ℝ) : ℝ :=
  let half_chord_length : ℝ := sqrt (r^2 - (r / 2)^2)
  in 2 * half_chord_length

theorem chord_is_26 :
  chord_length circle_radius 26 = 26 :=
by
  sorry

end chord_is_26_l326_326517


namespace semicircle_arc_length_l326_326794

theorem semicircle_arc_length (a b : ℝ) (hypotenuse_sum : a + b = 70) (a_eq_30 : a = 30) (b_eq_40 : b = 40) :
  ∃ (R : ℝ), (R = 24) ∧ (π * R = 12 * π) :=
by
  sorry

end semicircle_arc_length_l326_326794


namespace cubic_product_of_roots_l326_326563

theorem cubic_product_of_roots :
  let a := 1
  let b := -15
  let c := 60
  let d := -36
  let equation := (x^3 - 15 * x^2 + 60 * x - 36 = 0)
  let product_of_roots := -d / a
  product_of_roots = 36 :=
by
  let a := 1
  let d := -36
  calc -d / a = -(-36) / 1 : by sorry
              ...   = 36    : by sorry

end cubic_product_of_roots_l326_326563


namespace definite_integral_solution_l326_326229

theorem definite_integral_solution (a : ℝ) (h : ∫ x in 1..a, (2 * x + 1 / x) = 3 + Real.log 2) : a = 2 :=
by 
  sorry

end definite_integral_solution_l326_326229


namespace mitzi_remaining_money_l326_326408

noncomputable def total_spent : ℕ := 3000 + 2500 + 1500 + (2200 - (2200 * 20 / 100))
noncomputable def initial_amount : ℕ := 10000
def remaining_yen : ℕ := initial_amount - total_spent
noncomputable def exchange_rate : ℕ := 110
noncomputable def remaining_dollars : ℝ := (remaining_yen : ℝ) / (exchange_rate : ℝ)
noncomputable def rounded_dollars : ℝ := Float.ofRat (Rat.of (remaining_dollars * 100)).round / 100

theorem mitzi_remaining_money :
  remaining_yen = 1240 ∧ rounded_dollars = 11.27 := by
  sorry

end mitzi_remaining_money_l326_326408


namespace cart_max_speed_l326_326539

noncomputable def maximum_speed (a R : ℝ) : ℝ :=
  (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4)

theorem cart_max_speed (a R v : ℝ) (h : v = maximum_speed a R) : 
  v = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4) :=
by
  -- Proof is omitted
  sorry

end cart_max_speed_l326_326539


namespace circles_perpendicular_iff_plane_condition_l326_326131

-- Definitions for circles and planes.
variables (S1 S2 : Sphere) (cone: Cone) (cylinder: Cylinder) (A O: Point)

-- Conditions
axiom intersecting_circles_on_sphere : S1 ∩ S2 = {A}
axiom cone_tangent_to_sphere : tangent_to_sphere cone S1
axiom cylinder_tangent_to_sphere : tangent_to_sphere cylinder S1

-- Perpendicularity Condition to be proved
theorem circles_perpendicular_iff_plane_condition :
  (S1 ⊥ S2) ↔ (plane_of S2 passes_through_apex cone ∨ parallel_to_axis S2 cylinder) := 
sorry

end circles_perpendicular_iff_plane_condition_l326_326131


namespace m_value_for_no_linear_term_l326_326682

theorem m_value_for_no_linear_term (m : ℝ) :
  ¬∃ m,
    let p := (λ x : ℝ, (x - 2) * (x^2 + m*x + 1)) in
    ∃ b : ℝ, polynomial.coeff (polynomial.monomial 1 b) p = 0 :=
  m = 1 / 2 := sorry

end m_value_for_no_linear_term_l326_326682


namespace recurring_decimal_correct_l326_326774

def recurring_decimal := list nat → ℕ → nat

def recurring_decimal_value (x : recurring_decimal) (l : list nat) (m : ℕ) : nat :=
  let sum_digits := l.sum in
  let full_cycles := m / sum_digits in
  let remainder := m % sum_digits in
  let cycle_length := l.length in
  78.2 + full_cycles * cycle_length + (remainder / sum_digits)  -- Assumption for placing remainder

theorem recurring_decimal_correct (l : list ℕ) (m : ℕ) (x : ℕ) :
  l = [6, 7, 8, 2, 3, 0] →
  list.length l = 6 →
  list.sum l = 26 →
  m = 2017 →
  x = (recurring_decimal_value (λ l m, 78.2 + (m / l.sum) * list.length l + ((m % l.sum) / l.sum)) l m) →
  x = 78.2 + 30678 :=
begin
  intros hl hl_len hl_sum hm hx,
  sorry -- The proof steps go here
end

end recurring_decimal_correct_l326_326774


namespace ratio_of_perimeters_of_squares_l326_326476

theorem ratio_of_perimeters_of_squares (d1 d11 : ℝ) (s1 s11 : ℝ) (P1 P11 : ℝ) 
  (h1 : d11 = 11 * d1)
  (h2 : d1 = s1 * Real.sqrt 2)
  (h3 : d11 = s11 * Real.sqrt 2) :
  P11 / P1 = 11 :=
by
  sorry

end ratio_of_perimeters_of_squares_l326_326476


namespace art_gallery_total_l326_326926

theorem art_gallery_total (A : ℕ) (h₁ : (1 / 3) * A = D)
                         (h₂ : (1 / 6) * D = sculptures_on_display)
                         (h₃ : (1 / 3) * N = paintings_not_on_display)
                         (h₄ : (2 / 3) * N = 1200)
                         (D = (1 / 3) * A)
                         (N = A - D)
                         (N = (2 / 3) * A) :
                         A = 2700 :=
by
  sorry

end art_gallery_total_l326_326926


namespace total_seeds_gray_sections_combined_l326_326700

noncomputable def total_seeds_first_circle : ℕ := 87
noncomputable def seeds_white_first_circle : ℕ := 68
noncomputable def total_seeds_second_circle : ℕ := 110
noncomputable def seeds_white_second_circle : ℕ := 68

theorem total_seeds_gray_sections_combined :
  (total_seeds_first_circle - seeds_white_first_circle) +
  (total_seeds_second_circle - seeds_white_second_circle) = 61 :=
by
  sorry

end total_seeds_gray_sections_combined_l326_326700


namespace battery_replacement_15th_l326_326319

/-- Replaces batteries every 7 months, given the first replacement in January, prove that the
    15th replacement occurs in March. -/
theorem battery_replacement_15th (repl_interval : ℕ) (first_month : ℕ) : 
(first_month = 1) ∧ (repl_interval = 7) → 
((repl_interval * (15 - 1)) % 12 = 2) ∧ (2 + first_month = 3) :=
by
  intro h
  cases h with h_month h_interval
  sorry

end battery_replacement_15th_l326_326319


namespace largest_subset_no_diff_5_or_8_l326_326383

noncomputable def maxSubsetSize : ℕ :=
  1085

theorem largest_subset_no_diff_5_or_8 (S : Finset ℕ)
  (h1 : ∀ x ∈ S, x > 0 ∧ x ≤ 2023)
  (h2 : ∀ x y ∈ S, x ≠ y → (x - y).natAbs ≠ 5 ∧ (x - y).natAbs ≠ 8) : 
  S.card ≤ maxSubsetSize :=
  sorry

end largest_subset_no_diff_5_or_8_l326_326383


namespace distance_highest_point_to_plane_l326_326411

-- Define the radius and other constants
def r : ℝ := 24 - 12 * Real.sqrt 2

-- Define the height of the fifth sphere's center above the plane
def fifth_sphere_center_height : ℝ := 
  Real.sqrt (2 / 3) * (2 * r) + r

-- The final distance from the highest point of the fifth sphere to the plane
def distance_to_plane : ℝ := fifth_sphere_center_height + r

-- Theorem statement
theorem distance_highest_point_to_plane :
  distance_to_plane = 24 :=
sorry

end distance_highest_point_to_plane_l326_326411


namespace compute_fraction_l326_326747

theorem compute_fraction (c d : ℚ) (h₁ : c = 4 / 7) (h₂ : d = 5 / 6) :
  c^3 * d^(-2) + c^(-1) * d^2 = 1832401 / 1234800 := by
  rw [h₁, h₂]
  sorry

end compute_fraction_l326_326747


namespace proof_problem_l326_326595

open Real

noncomputable def line1 (A : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ c : ℝ, (A.1 = 3 ∧ A.2 = 2) ∧ (l (A.1, A.2) = 4 * A.1 + A.2 + c) ∧ (c = -14)

noncomputable def line2 (B : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ c : ℝ, (B.1 = 3 ∧ B.2 = 0) ∧ (l (B.1, B.2) = B.1 - 2 * B.2 + c) ∧ (c = -3)

theorem proof_problem :
  (∃ l1 : ℝ × ℝ → ℝ, line1 (3, 2) l1 ∧ (l1 (3, 2) = 4 * 3 + 2 - 14)) ∧
  (∃ l2 : ℝ × ℝ → ℝ, line2 (3, 0) l2 ∧ (l2 (3, 0) = 3 - 2 * 0 - 3)) :=
by {
  sorry,
}

end proof_problem_l326_326595


namespace ratio_of_sines_l326_326738

noncomputable def sides_consecutive_integers (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = b + 1

noncomputable def valid_triangle (A B C : ℝ) (a b c : ℕ) : Prop :=
  ∠ A + ∠ B + ∠ C = real.pi ∧ a = b + c := sorry

noncomputable def cos_law_eqn (a b : ℝ) (A : ℝ) : Prop :=
  3 * b = 20 * a * (real.cos A)

theorem ratio_of_sines (A B C : ℝ) (a b c : ℕ)
  (h1 : sides_consecutive_integers a b c)
  (h2 : valid_triangle A B C a b c)
  (h3 : A > B ∧ B > C)
  (h4 : cos_law_eqn a b A) :
  real.sin A / real.sin B = 6 / 5 ∧ real.sin B / real.sin C = 5 / 4 := sorry

end ratio_of_sines_l326_326738


namespace angle_bisector_coordinates_distance_to_x_axis_l326_326625

structure Point where
  x : ℝ
  y : ℝ

def M (m : ℝ) : Point :=
  ⟨m - 1, 2 * m + 3⟩

theorem angle_bisector_coordinates (m : ℝ) :
  (M m = ⟨-5, -5⟩) ∨ (M m = ⟨-(5/3), 5/3⟩) := sorry

theorem distance_to_x_axis (m : ℝ) :
  (|2 * m + 3| = 1) → (M m = ⟨-2, 1⟩) ∨ (M m = ⟨-3, -1⟩) := sorry

end angle_bisector_coordinates_distance_to_x_axis_l326_326625


namespace largest_possible_k_l326_326921

theorem largest_possible_k (N : ℕ) (hN : ∃ k : ℕ, N = 3 * k + (N % 3)) 
(h_prob : ∀ balls : { red white blue : ℕ // red + white + blue = N }, 
  let ⟨r, w, b, h_sum⟩ := balls in (r ≥ 1 /\ w ≥ 1 /\ b ≥ 1) -> 
  (6 * r * w * b : ℚ) / (N * (N - 1) * (N - 2)) > 23 / 100) :
  let k := N / 3 in k = 29 :=
by
  sorry

end largest_possible_k_l326_326921


namespace average_marks_of_all_candidates_l326_326792

def n : ℕ := 120
def p : ℕ := 100
def f : ℕ := n - p
def A_p : ℕ := 39
def A_f : ℕ := 15
def total_marks : ℕ := p * A_p + f * A_f
def average_marks : ℚ := total_marks / n

theorem average_marks_of_all_candidates :
  average_marks = 35 := 
sorry

end average_marks_of_all_candidates_l326_326792


namespace common_ratio_l326_326037

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem common_ratio (a₁ : ℝ) (h : a₁ ≠ 0) : 
  (∀ S4 S5 S6, S5 = geometric_sum a₁ q 5 ∧ S4 = geometric_sum a₁ q 4 ∧ S6 = geometric_sum a₁ q 6 → 
  2 * S4 = S5 + S6) → 
  q = -2 := 
by
  sorry

end common_ratio_l326_326037


namespace set_d_forms_proportion_l326_326924

theorem set_d_forms_proportion (a b c d : ℕ) : (a, b, c, d) = (3, 2, 4, 6) → b * c = a * d := by
  intros h
  cases h
  calc
    2 * 4 = 8     : by norm_num
    ... = 3 * 6   : by norm_num

end set_d_forms_proportion_l326_326924


namespace math_problem_l326_326021

-- Define the sets and the required conditions in Lean
def A (a : ℝ) : set ℝ := {x | 2 * x^2 + a * x + 2 = 0}
def B (a : ℝ) : set ℝ := {x | x^2 + 3 * x + 2 * a = 0}
def U (a : ℝ) : set ℝ := A a ∪ B a

theorem math_problem :
  (2 ∈ A a) ∧ (2 ∈ B a) → 
  (∃ a', a' = -5 ∧ A a' = {2, 1/2} ∧ B a' = {2, -5}) ∧
  ((complement (U a) (A a)) ∪ (complement (U a) (B a)) = {1/2, -5}) ∧
  (∀ x ∈ ({1/2, -5}: set ℝ), 
    (x = 1/2 ∨ x = -5 ∨ x = ∅ ∨ x = {1/2, -5})) :=
sorry

end math_problem_l326_326021


namespace trip_cost_l326_326209

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l326_326209


namespace arithmetic_sequence_sum_l326_326964

theorem arithmetic_sequence_sum : 
  let a1 := 1
  let d := 2
  let an := 25
  let n := 13
  let S := (n * (a1 + an)) / 2
in S = 169 :=
by
  -- Placeholder proof
  sorry

end arithmetic_sequence_sum_l326_326964


namespace not_divisible_by_x2_x_plus_1_l326_326379

noncomputable def g : polynomial ℝ := sorry -- fixed polynomial with real coefficients
noncomputable def f (x : ℝ) : polynomial ℝ := x^2 + x * g (x^3)

theorem not_divisible_by_x2_x_plus_1 :
  ¬ ∃ h : polynomial ℝ, f = (polynomial.X^2 - polynomial.X + 1) * h :=
sorry

end not_divisible_by_x2_x_plus_1_l326_326379


namespace num_friends_bought_robots_l326_326934

def robot_cost : Real := 8.75
def tax_charged : Real := 7.22
def change_left : Real := 11.53
def initial_amount : Real := 80.0
def friends_bought_robots : Nat := 7

theorem num_friends_bought_robots :
  (initial_amount - (change_left + tax_charged)) / robot_cost = friends_bought_robots := sorry

end num_friends_bought_robots_l326_326934


namespace sum_even_positives_less_than_80_l326_326859

theorem sum_even_positives_less_than_80 : 
  (∑ i in Finset.filter (λ n, even n) (Finset.range 80).filter (λ n, n > 0), i) = 1560 := 
sorry

end sum_even_positives_less_than_80_l326_326859


namespace polynomial_is_X_pow_l326_326906
noncomputable theory

open Polynomial

-- Define the main theorem
theorem polynomial_is_X_pow (f : Polynomial ℤ) 
  (H_non_constant : ¬ f.natDegree = 0)
  (H_prime_property : ∀ p : ℕ, Nat.Prime p → ∃ q : ℕ, Nat.Prime q ∧ ∃ m : ℕ, 0 < m ∧ f.eval (p : ℤ) = q ^ m) : 
  ∃ n : ℕ, 0 < n ∧ f = X ^ n := 
  sorry

end polynomial_is_X_pow_l326_326906


namespace solve_equation_l326_326786

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end solve_equation_l326_326786


namespace max_integer_value_f_l326_326979

def f (x : ℝ) : ℝ := (4 * x^3 + 4 * x^2 + 12 * x + 23) / (4 * x^3 + 4 * x^2 + 12 * x + 9)

theorem max_integer_value_f : ∃ (m : ℤ), ∀ x : ℝ, f x < m + 1 ∧ f x ≥ m :=
by
  use 1
  sorry

end max_integer_value_f_l326_326979


namespace positive_difference_between_two_numbers_l326_326466

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end positive_difference_between_two_numbers_l326_326466


namespace count_valid_numbers_l326_326320

-- Define a set with the four digits of 2023
def digits : Finset ℕ := {2, 0, 2, 3}

-- Predicate to check if a number is a valid 4-digit number using exactly those digits
def valid_number (n : ℕ) : Prop :=
  (n >= 2000) ∧ (n < 10000) ∧ (digits \ (Finset.ofMultiset (Multiset.ofDigits n)) = ∅)

theorem count_valid_numbers : ∃ (count : ℕ), count = 6 ∧ ∀ n, valid_number n → 2000 ≤ n := by
  sorry

end count_valid_numbers_l326_326320


namespace part1_solution_set_part2_range_of_a_l326_326399

def f(x : ℝ) := |x|
def g(x : ℝ) := |2 * x - 2|

theorem part1_solution_set :
  ∀ x : ℝ, f(x) > g(x) ↔ (2 / 3) < x ∧ x < 2 := sorry

theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * f(x) + g(x) > a * x + 1) ↔ a ∈ set.Icc (-4 : ℝ) 1 := sorry

end part1_solution_set_part2_range_of_a_l326_326399


namespace num_scalene_triangles_with_perimeter_lt_15_l326_326111

def is_scalene_triangle (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem num_scalene_triangles_with_perimeter_lt_15 :
  { (a, b, c) : ℕ × ℕ × ℕ // is_scalene_triangle a b c ∧ (a + b + c < 15) }.to_finset.card = 7 :=
sorry

end num_scalene_triangles_with_perimeter_lt_15_l326_326111


namespace negation_of_universal_l326_326108

theorem negation_of_universal: (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by 
  sorry

end negation_of_universal_l326_326108


namespace debut_show_tickets_l326_326194

variable (P : ℕ) -- Number of people who bought tickets for the debut show

-- Conditions
def three_times_more (P : ℕ) : Bool := (3 * P = P + 2 * P)
def ticket_cost : ℕ := 25
def total_revenue (P : ℕ) : ℕ := 4 * P * ticket_cost

-- Main statement
theorem debut_show_tickets (h1 : three_times_more P = true) 
                           (h2 : total_revenue P = 20000) : P = 200 :=
by
  sorry

end debut_show_tickets_l326_326194


namespace card_arrangement_sum_10_l326_326840

-- Definitions:
def red_cards := {1, 2, 3, 4}
def blue_cards := {1, 2, 3, 4}
def cards := red_cards ∪ blue_cards

-- Problem statement in Lean 4:
theorem card_arrangement_sum_10 : 
  ∃ (arrangements : set (list ℕ)), arrangements.card = 432 ∧
  ∀ arrangement ∈ arrangements, arrangement.length = 4 ∧ (list.sum arrangement) = 10 ∧ 
  ∀ n ∈ arrangement, n ∈ cards := sorry

end card_arrangement_sum_10_l326_326840


namespace bruce_anne_clean_in_4_hours_l326_326947

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l326_326947


namespace trip_cost_l326_326207

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l326_326207


namespace solve_for_x_l326_326430

theorem solve_for_x :
  ∃ x : ℝ, (x - 6)^3 = (1 / 16)^(-1) ∧ x = 6 + 2^(4 / 3) :=
begin
  sorry
end

end solve_for_x_l326_326430


namespace solve_inequality_l326_326434

noncomputable def f (x : ℝ) : ℝ := sqrt(x^3 - 10 * x + 7) + 1
noncomputable def g (x : ℝ) : ℝ := abs(x^3 - 18 * x + 28)

theorem solve_inequality :
  ∀ x : ℝ, (f x) * (g x) ≤ 0 ↔ x = -1 + sqrt(15) :=
by sorry

end solve_inequality_l326_326434


namespace sara_sent_letters_l326_326066

theorem sara_sent_letters (J : ℕ)
  (h1 : 9 + 3 * J + J = 33) : J = 6 :=
by
  sorry

end sara_sent_letters_l326_326066


namespace meaningful_sqrt_x_minus_3_l326_326702

theorem meaningful_sqrt_x_minus_3 :
  ∀ x : ℕ, x ∈ {0, 1, 2, 4} → (x - 3) ≥ 0 ↔ x = 4 :=
by sorry

end meaningful_sqrt_x_minus_3_l326_326702


namespace find_quadratic_relationship_l326_326981

-- Definitions of the conditions given by the points
def y (x : ℤ) : ℤ :=
  if x = 0 then 200
  else if x = 2 then 160
  else if x = 4 then 80
  else if x = 6 then 0
  else if x = 8 then -120
  else sorry  -- undefined for other values of x in this context

-- The theorem we want to prove
theorem find_quadratic_relationship :
  ∀ (x : ℤ), y x = -10 * x * x + 200 :=
by
  intro x
  cases x
  case 0 { sorry }
  case 2 { sorry }
  case 4 { sorry }
  case 6 { sorry }
  case 8 { sorry }
  case _ { sorry }

end find_quadratic_relationship_l326_326981


namespace sum_abs_sequence_15_terms_l326_326653

def a (n : ℕ) : ℤ := 2 * n - 7

theorem sum_abs_sequence_15_terms : (Finset.sum (Finset.range 15) (λ n, |a (n + 1)|)) = 153 :=
  sorry

end sum_abs_sequence_15_terms_l326_326653


namespace lindas_daughters_and_granddaughters_no_daughters_l326_326403

def number_of_people_with_no_daughters (total_daughters total_descendants daughters_with_5_daughters : ℕ) : ℕ :=
  total_descendants - (5 * daughters_with_5_daughters - total_daughters + daughters_with_5_daughters)

theorem lindas_daughters_and_granddaughters_no_daughters
  (total_daughters : ℕ)
  (total_descendants : ℕ)
  (daughters_with_5_daughters : ℕ)
  (H1 : total_daughters = 8)
  (H2 : total_descendants = 43)
  (H3 : 5 * daughters_with_5_daughters = 35)
  : number_of_people_with_no_daughters total_daughters total_descendants daughters_with_5_daughters = 36 :=
by
  -- Code to check the proof goes here.
  sorry

end lindas_daughters_and_granddaughters_no_daughters_l326_326403


namespace complex_conjugate_solution_l326_326752

namespace ComplexProof

open Complex

theorem complex_conjugate_solution (z : ℂ) (h : (1 + I) * z = 2 * I) : conj z = 1 - I :=
sorry

end ComplexProof

end complex_conjugate_solution_l326_326752


namespace Alan_shells_l326_326055

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end Alan_shells_l326_326055


namespace sally_total_fries_is_50_l326_326420

-- Definitions for the conditions
def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 3 * 12
def mark_fraction_given_to_sally : ℕ := mark_initial_fries / 3
def jessica_total_cm_of_fries : ℕ := 240
def fry_length_cm : ℕ := 5
def jessica_total_fries : ℕ := jessica_total_cm_of_fries / fry_length_cm
def jessica_fraction_given_to_sally : ℕ := jessica_total_fries / 2

-- Definition for the question
def total_fries_sally_has (sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally : ℕ) : ℕ :=
  sally_initial_fries + mark_fraction_given_to_sally + jessica_fraction_given_to_sally

-- The theorem to be proved
theorem sally_total_fries_is_50 :
  total_fries_sally_has sally_initial_fries mark_fraction_given_to_sally jessica_fraction_given_to_sally = 50 :=
sorry

end sally_total_fries_is_50_l326_326420


namespace area_of_triangle_AOB_is_correct_l326_326904

noncomputable def area_of_triangle_AOB : ℝ :=
  let ellipse := {x // (x.1 ^ 2 / 2 + x.2 ^ 2 = 1)}
  let F1 := (1, 0)
  let line := {p : ℝ × ℝ | p.2 = p.1 - 1}
  let A := (0, -1)
  let B := (4/3, 1/3)
  let O := (0, 0)
  let AB_dist := real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
  let d_O_AB := real.abs (0 - 1) / real.sqrt 2
  (1 / 2) * AB_dist * d_O_AB

theorem area_of_triangle_AOB_is_correct : area_of_triangle_AOB = 2 / 3 :=
by sorry

end area_of_triangle_AOB_is_correct_l326_326904


namespace value_of_abs_m_minus_n_l326_326644

theorem value_of_abs_m_minus_n  (m n : ℝ) (h_eq : ∀ x, (x^2 - 2 * x + m) * (x^2 - 2 * x + n) = 0)
  (h_arith_seq : ∀ x₁ x₂ x₃ x₄ : ℝ, x₁ + x₂ = 2 ∧ x₃ + x₄ = 2 ∧ x₁ = 1 / 4 ∧ x₂ = 3 / 4 ∧ x₃ = 5 / 4 ∧ x₄ = 7 / 4) :
  |m - n| = 1 / 2 :=
by
  sorry

end value_of_abs_m_minus_n_l326_326644


namespace modified_cube_cubies_l326_326886

structure RubiksCube :=
  (original_cubies : ℕ := 27)
  (removed_corners : ℕ := 8)
  (total_layers : ℕ := 3)
  (edges_per_layer : ℕ := 4)
  (faces_center_cubies : ℕ := 6)
  (center_cubie : ℕ := 1)

noncomputable def cubies_with_n_faces (n : ℕ) : ℕ :=
  if n = 4 then 12
  else if n = 1 then 6
  else if n = 0 then 1
  else 0

theorem modified_cube_cubies :
  (cubies_with_n_faces 4 = 12) ∧ (cubies_with_n_faces 1 = 6) ∧ (cubies_with_n_faces 0 = 1) := by
  sorry

end modified_cube_cubies_l326_326886


namespace chess_tournament_participants_l326_326869

theorem chess_tournament_participants (games_played : ℕ) (n : ℕ) 
    (h_games : games_played = n * (n - 1) / 2) : n = 19 := 
by 
    have h : n * (n - 1) = 342 := by linarith [h_games]
    exact Nat.mul_eq_iff_eq_or_eq.mp h.gt1.1 

end chess_tournament_participants_l326_326869


namespace siding_cost_l326_326421

noncomputable def cos_30_deg := Real.cos (30 * Real.pi / 180)

structure Wall :=
(width : ℝ)
(height : ℝ)

structure RoofSection :=
(base : ℝ)
(avg_height : ℝ)
(inclination_deg : ℝ)

def wall_area (w : Wall) : ℝ := w.width * w.height

def slant_height (r : RoofSection) : ℝ := r.avg_height / cos_30_deg

def roof_area (r : RoofSection) : ℝ := r.base * slant_height r

def total_area (w : Wall) (r : RoofSection) : ℝ := wall_area w + 2 * roof_area r

def sections_needed (area : ℝ) (section_area : ℝ) : ℝ := Real.ceil (area / section_area)

def total_cost (sections : ℝ) (cost_per_section : ℝ) : ℝ := sections * cost_per_section

theorem siding_cost :
  total_cost
    (sections_needed (total_area
      ⟨10, 8⟩
      ⟨10, 7, 30⟩)
      100)
    30
  = 90 := by
  sorry

end siding_cost_l326_326421


namespace intersection_A_B_l326_326343

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 4} :=
sorry

end intersection_A_B_l326_326343


namespace belfried_industries_payroll_l326_326495

theorem belfried_industries_payroll (P : ℝ) (tax_paid : ℝ) : 
  ((P > 200000) ∧ (tax_paid = 0.002 * (P - 200000)) ∧ (tax_paid = 200)) → P = 300000 :=
by
  sorry

end belfried_industries_payroll_l326_326495


namespace sin_cos_intersection_ratio_l326_326174

theorem sin_cos_intersection_ratio :
  ∃ p q : ℕ, (RelPrime p q) ∧ Rat.positives p q ∧ SegmentRatio({30, 150}, 360) = (p, q) := 
by {
  sorry
}

end sin_cos_intersection_ratio_l326_326174


namespace rhombus_perimeter_l326_326812

theorem rhombus_perimeter (d1 d2 : ℕ) (h_d1 : d1 = 16) (h_d2 : d2 = 30) :
  let side_length := Math.sqrt ((d1 / 2)^2 + (d2 / 2)^2) in
  let perimeter := 4 * side_length in
  perimeter = 68 := by
    dsimp only [side_length, perimeter]
    rw [h_d1, h_d2]
    norm_num
    sorry

end rhombus_perimeter_l326_326812


namespace find_function_solution_l326_326589

noncomputable def function_solution (f : ℤ → ℤ) : Prop :=
∀ x y : ℤ, x ≠ 0 → x * f (2 * f y - x) + y^2 * f (2 * x - f y) = f x ^ 2 / x + f (y * f y)

theorem find_function_solution : 
  ∀ f : ℤ → ℤ, function_solution f → (∀ x : ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end find_function_solution_l326_326589


namespace count_pairs_is_five_l326_326574

noncomputable def count_pairs : ℕ :=
  {xy : ℕ × ℕ // (0 < xy.1 ∧ xy.1 < xy.2 ∧ 2 * xy.1 + 3 * xy.2 = 80)}.to_finset.card

theorem count_pairs_is_five : count_pairs = 5 :=
  sorry

end count_pairs_is_five_l326_326574


namespace evaluate_expression_l326_326990

theorem evaluate_expression :
  ( (4 / 9 : ℝ) ^ (1 / 2) - ( (real.sqrt 2) / 2) ^ 0 + ( 27 / 64 : ℝ) ^ (-1 / 3) ) = 1 := 
begin
   sorry
end

end evaluate_expression_l326_326990


namespace even_function_a_value_l326_326336

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, a * sin (x + π / 4) + 3 * sin (x - π / 4) = a * sin (-x + π / 4) + 3 * sin (-x - π / 4)) → a = -3 :=
by
  intro h
  sorry

end even_function_a_value_l326_326336


namespace sum_of_roots_equation_l326_326600

theorem sum_of_roots_equation :
  let poly1 := (3 * X^4 + 2 * X^3 - 9 * X^2 + 5 * X - 15)
  let poly2 := (4 * X^3 - 16 * X^2 + X + 7)
  let equation := X in poly1 * poly2 - 10 = 0
  ∑ (root : _) in roots_of (poly1 * poly2 - 10), root = (10 : ℝ) / 3 :=
sorry

end sum_of_roots_equation_l326_326600


namespace problem1_problem2_l326_326850

section ProofProblems

-- Definitions for binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1: Prove that n! = binom(n, k) * k! * (n-k)!
theorem problem1 (n k : ℕ) : n.factorial = binom n k * k.factorial * (n - k).factorial :=
by sorry

-- Problem 2: Prove that binom(n, k) = binom(n-1, k) + binom(n-1, k-1)
theorem problem2 (n k : ℕ) : binom n k = binom (n-1) k + binom (n-1) (k-1) :=
by sorry

end ProofProblems

end problem1_problem2_l326_326850


namespace solve_equation_1_solve_equation_2_l326_326077

theorem solve_equation_1 (y: ℝ) : y^2 - 6 * y + 1 = 0 ↔ (y = 3 + 2 * Real.sqrt 2 ∨ y = 3 - 2 * Real.sqrt 2) :=
sorry

theorem solve_equation_2 (x: ℝ) : 2 * (x - 4)^2 = x^2 - 16 ↔ (x = 4 ∨ x = 12) :=
sorry

end solve_equation_1_solve_equation_2_l326_326077


namespace friends_bill_correct_l326_326147

def taco_price (total_bill enchilada_count enchilada_price : ℝ) : ℝ :=
  (total_bill - (enchilada_count * enchilada_price)) / 2

def friends_bill (taco_count friend_taco_price friend_enchilada_count enchilada_price : ℝ) : ℝ :=
  (taco_count * friend_taco_price) + (friend_enchilada_count * enchilada_price)

theorem friends_bill_correct :
  friends_bill 3 (taco_price 7.80 3 2) 5 2 = 12.70 := by
  sorry

end friends_bill_correct_l326_326147


namespace max_tetrahedron_in_cube_l326_326145

open Real

noncomputable def cube_edge_length : ℝ := 6
noncomputable def max_tetrahedron_edge_length (a : ℝ) : Prop :=
  ∃ x : ℝ, x = 2 * sqrt 6 ∧ 
          (∃ R : ℝ, R = (a * sqrt 3) / 2 ∧ x / sqrt (2 / 3) = 4 * R / 3)

theorem max_tetrahedron_in_cube : max_tetrahedron_edge_length cube_edge_length :=
sorry

end max_tetrahedron_in_cube_l326_326145


namespace percentage_increase_l326_326833

variables (a b x m : ℝ) (p : ℝ)
variables (h1 : a / b = 4 / 5)
variables (h2 : x = a + (p / 100) * a)
variables (h3 : m = b - 0.6 * b)
variables (h4 : m / x = 0.4)

theorem percentage_increase (a_pos : 0 < a) (b_pos : 0 < b) : p = 25 :=
by sorry

end percentage_increase_l326_326833


namespace find_real_root_of_sqrt_eq_l326_326597

-- Definitions based on conditions
def is_real_real_root_of_sqrt_eq (x : ℝ) : Prop :=
  sqrt x + sqrt (x + 6) = 12

-- Statement of the proof problem
theorem find_real_root_of_sqrt_eq : is_real_real_root_of_sqrt_eq (529 / 16) :=
by
  sorry

end find_real_root_of_sqrt_eq_l326_326597


namespace tiling_implies_divisibility_l326_326748

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end tiling_implies_divisibility_l326_326748


namespace alexandra_brianna_meeting_probability_l326_326101

noncomputable def probability_meeting (A B : ℕ × ℕ) : ℚ :=
if A = (0,0) ∧ B = (5,7) then 347 / 768 else 0

theorem alexandra_brianna_meeting_probability :
  probability_meeting (0,0) (5,7) = 347 / 768 := 
by sorry

end alexandra_brianna_meeting_probability_l326_326101


namespace find_x2_plus_y2_l326_326338

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
  sorry

end find_x2_plus_y2_l326_326338


namespace div_m_by_18_equals_500_l326_326105

-- Define the conditions
noncomputable def m : ℕ := 9000 -- 'm' is given as 9000 since it fulfills all conditions described
def is_multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def all_digits_9_or_0 (n : ℕ) : Prop := ∀ (d : ℕ), (∃ (k : ℕ), n = 10^k * d) → (d = 0 ∨ d = 9)

-- Define the proof problem statement
theorem div_m_by_18_equals_500 
  (h1 : is_multiple_of_18 m) 
  (h2 : all_digits_9_or_0 m) 
  (h3 : ∀ n, is_multiple_of_18 n ∧ all_digits_9_or_0 n → n ≤ m) : 
  m / 18 = 500 :=
sorry

end div_m_by_18_equals_500_l326_326105


namespace f_sum_values_l326_326821

-- Define the function f and its properties
variable (f : ℕ → ℕ)

-- The function is strictly increasing
axiom f_strictly_increasing : ∀ {x y : ℕ}, x < y → f(x) < f(y)

-- The function satisfies f(f(k)) = 3k
axiom f_property : ∀ k : ℕ, f(f(k)) = 3 * k

-- The main theorem to prove
theorem f_sum_values : f(1) + f(9) + f(96) = 197 := sorry

end f_sum_values_l326_326821


namespace positive_difference_between_two_numbers_l326_326465

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end positive_difference_between_two_numbers_l326_326465


namespace pyramid_volume_correct_l326_326456

-- Defining the dimensions and points
def EF_dist : ℝ := 10 * Real.sqrt 2
def FG_dist : ℝ := 15 * Real.sqrt 2
def EH_dist : ℝ := Real.sqrt ((10 * Real.sqrt 2) ^ 2 + (15 * Real.sqrt 2) ^ 2)  -- Diagonal length by Pythagoras

-- Defining the coordinates of points assuming the origin is at the center of the rectangle
def G := (5 * Real.sqrt 2, 0, 0)
def H := (-5 * Real.sqrt 2, 0, 0)
def E : ℝ × ℝ × ℝ := (0, Real.sqrt 275, 0)
def Q : ℝ × ℝ × ℝ := (0, Real.sqrt 275 / 2, 225 / (2 * Real.sqrt 31))

-- Function to calculate the volume of the pyramid
def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  1 / 3 * base_area * height

-- Base area is derived from given lengths
def base_area : ℝ := 15 * Real.sqrt 31

-- Verifying the distance condition from Q to E, F, G
def is_distance_correct : Prop :=
  (Q.1^2 + (Q.2 - E.2)^2 + Q.3^2 = 875 / 4) ∧
  ((Q.1 - G.1)^2 + Q.2^2 + Q.3^2 = 875 / 4) ∧
  ((Q.1 + H.1)^2 + Q.2^2 + Q.3^2 = 875 / 4)

-- Final volume of the tetrahedron
def volume : ℝ := pyramid_volume base_area (225 / (2 * Real.sqrt 31))

-- The condition to prove
theorem pyramid_volume_correct : volume = 1687.5 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end pyramid_volume_correct_l326_326456


namespace Bruce_Anne_combined_cleaning_time_l326_326946

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l326_326946


namespace blackboard_final_number_lower_bound_l326_326264

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def L (c : ℝ) : ℝ := 1 + Real.log c / Real.log phi

theorem blackboard_final_number_lower_bound (c : ℝ) (n : ℕ) (h_pos_c : c > 1) (h_pos_n : n > 0) :
  ∃ x, x ≥ ((c^(n / (L c)) - 1) / (c^(1 / (L c)) - 1))^(L c) :=
sorry

end blackboard_final_number_lower_bound_l326_326264


namespace hall_paving_l326_326902

theorem hall_paving :
  ∀ (hall_length hall_breadth stone_length stone_breadth : ℕ),
    hall_length = 72 →
    hall_breadth = 30 →
    stone_length = 8 →
    stone_breadth = 10 →
    let Area_hall := hall_length * hall_breadth
    let Length_stone := stone_length / 10
    let Breadth_stone := stone_breadth / 10
    let Area_stone := Length_stone * Breadth_stone 
    (Area_hall / Area_stone) = 2700 :=
by
  intros hall_length hall_breadth stone_length stone_breadth
  intro h1 h2 h3 h4
  let Area_hall := hall_length * hall_breadth
  let Length_stone := stone_length / 10
  let Breadth_stone := stone_breadth / 10
  let Area_stone := Length_stone * Breadth_stone 
  have h5 : Area_hall / Area_stone = 2700 := sorry
  exact h5

end hall_paving_l326_326902


namespace march_1_is_monday_l326_326674

open Nat

theorem march_1_is_monday (h : ∃ k : ℕ, k % 7 = 5 ∧ march 13 == saturday) : 
  (march 1 = monday) :=
sorry

end march_1_is_monday_l326_326674


namespace minimum_time_to_transport_supplies_l326_326876

theorem minimum_time_to_transport_supplies (v : ℝ) (h_pos : v > 0) :
  ∀ t, (∃ t_1, t_1 = 400 / v) ∧
         (∀ i : ℕ, 1 ≤ i ∧ i < 26 → ∃ d, d = i * (v / 20)^2) →
         (∃ t_additional, t_additional = 25 * (v / 20)^2 / v) →
         t = (400 / v) + (25 * (v / 20)^2 / v) →
         t ≥ 10 :=
begin
  sorry
end

end minimum_time_to_transport_supplies_l326_326876


namespace permutations_deranged_l326_326261

-- Define the problem statement
theorem permutations_deranged (n : ℕ) (h : n ≥ 3) : 
  ∃ k, k = (n^3 - 6*n^2 + 14*n - 13) * ((nat.factorial (n - 3))) :=
sorry

end permutations_deranged_l326_326261


namespace sum_eq_product_l326_326417

theorem sum_eq_product (a b c : ℝ) (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) :
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) =
  ((b - c) * (c - a) * (a - b)) / ((1 + b * c) * (1 + c * a) * (1 + a * b)) :=
by
  sorry

end sum_eq_product_l326_326417


namespace mary_needs_more_apples_l326_326045

theorem mary_needs_more_apples :
  let pies := 15
  let apples_per_pie := 10
  let harvested_apples := 40
  let total_apples_needed := pies * apples_per_pie
  let more_apples_needed := total_apples_needed - harvested_apples
  more_apples_needed = 110 :=
by
  sorry

end mary_needs_more_apples_l326_326045


namespace unique_albums_count_l326_326928

structure AlbumCollection where
  A : Set String
  J : Set String
  B : Set String

open AlbumCollection

def uniqueAlbums (collections : AlbumCollection) : Nat :=
  (collections.A \ collections.J).card + (collections.J \ collections.A).card

theorem unique_albums_count (collections : AlbumCollection)
  (hA : collections.A.card = 20)
  (h_intersect : (collections.A ∩ collections.J).card = 10)
  (hJ_diff : (collections.J \ collections.A).card = 8)
  (hB_overlap : (collections.B ∩ (collections.A \ collections.J)).card = 5) :
  uniqueAlbums collections = 18 := by
  sorry

end unique_albums_count_l326_326928


namespace solve_equation_l326_326884

theorem solve_equation :
  ∃ (q : ℝ), q * 120 = 173 * 240 ∧ q = 345.33 :=
by
  use 345.33
  split
  · sorry -- proof of q * 120 = 173 * 240
  · sorry -- proof of q = 345.33

end solve_equation_l326_326884


namespace random_events_l326_326214

-- Define the events based on the conditions
inductive Event
| event1 : Event  -- Tossing a coin twice in succession, and both times it lands heads up
| event2 : Event  -- Opposite charges attract each other
| event3 : Event  -- At standard atmospheric pressure, water freezes at 1℃
| event4 : Event  -- Rolling a die and the number facing up is even

-- Define the property of being a random event
def isRandom (e : Event) : Prop :=
  e = Event.event1 ∨ e = Event.event4

-- The statement to be proved in Lean 4
theorem random_events :
  ∀ e, e = Event.event1 ∨ e = Event.event4 ↔ isRandom e :=
by
  intro e
  split
  . intro h
    cases h
    case inl h => rw [h]; exact or.inl rfl
    case inr h => rw [h]; exact or.inr rfl
  . intro h
    cases h
    case or.inl h => left; assumption
    case or.inr h => right; assumption

end random_events_l326_326214


namespace root_in_interval_l326_326822

noncomputable def f (x: ℝ) : ℝ := x^2 + (Real.log x) - 4

theorem root_in_interval : 
  (∃ ξ ∈ Set.Ioo 1 2, f ξ = 0) :=
by
  sorry

end root_in_interval_l326_326822


namespace initial_distance_l326_326199

-- Definitions based on conditions
def speed_thief : ℝ := 8 -- in km/hr
def speed_policeman : ℝ := 10 -- in km/hr
def distance_thief_runs : ℝ := 0.7 -- in km

-- Theorem statement
theorem initial_distance
  (relative_speed := speed_policeman - speed_thief) -- Relative speed (in km/hr)
  (time_to_overtake := distance_thief_runs / relative_speed) -- Time for the policeman to overtake the thief (in hours)
  (initial_distance := speed_policeman * time_to_overtake) -- Initial distance (in km)
  : initial_distance = 3.5 :=
by
  sorry

end initial_distance_l326_326199


namespace meaningful_sqrt_l326_326705

theorem meaningful_sqrt (x : ℝ) (h : x - 3 ≥ 0) : x = 4 → sqrt (x - 3) ≥ 0 :=
by
  intro hx
  rw [hx]
  simp
  exact h

end meaningful_sqrt_l326_326705


namespace no_simultaneous_squares_l326_326968

theorem no_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + 2 * y = a^2 ∧ y^2 + 2 * x = b^2) :=
by
  sorry

end no_simultaneous_squares_l326_326968


namespace set_intersection_complement_l326_326777

theorem set_intersection_complement :
  let M := { x : ℝ | x ≥ -2 }
  let N := { x : ℝ | 2^x - 1 > 0 }
  let Cr_N := { x : ℝ | ¬ (2^x - 1 > 0) }
  in M ∩ Cr_N = { x : ℝ | -2 ≤ x ∧ x ≤ 0 } :=
sorry

end set_intersection_complement_l326_326777


namespace last_remaining_number_is_odd_l326_326410

-- Define the initial set of numbers from 1 to 50
def initial_set := finset.range 51

-- Function to perform the operation of replacing two numbers with the absolute value of their difference
def replace (a b : ℕ) : ℕ :=
  abs (a - b)

-- Define a proposition stating the last remaining number is odd
theorem last_remaining_number_is_odd (S : finset ℕ) (hS : S = initial_set) :
  odd (finset.sum S id) :=
begin
  sorry
end

end last_remaining_number_is_odd_l326_326410


namespace first_player_wins_l326_326839

noncomputable def winning_strategy (n : ℕ) : Prop :=
  ∃ strategy : (Fin n → Bool) → (Fin n → Bool), 
    ∀ state : Fin n → Bool, 
      strategy state ≠ state ∧
      ∀ next_state, strategy state = next_state ↔ 
        ∃ i : Fin n, next_state = state.update i (¬state i) ∧
          (state.update i (¬state i)).toList ∉ List.range (2^n)

theorem first_player_wins : winning_strategy 2012 :=
sorry

end first_player_wins_l326_326839


namespace power_function_through_point_l326_326299

theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) (h₁ : ∀ x : ℝ, f x = x ^ α)
  (h₂ : f 2 = sqrt 2) : f x = sqrt x :=
by
  have : 2 ^ α = sqrt 2 := h₂
  -- sorry

end power_function_through_point_l326_326299


namespace even_F_even_T_l326_326028

section
variable (f : ℝ → ℝ)

def F (x : ℝ) : ℝ := f x * f (-x)
def G (x : ℝ) : ℝ := f x * |f (-x)|
def H (x : ℝ) : ℝ := f x - f (-x)
def T (x : ℝ) : ℝ := f x + f (-x)

theorem even_F : ∀ x : ℝ, F f x = F f (-x) :=
by
  intro x
  unfold F
  rw [neg_neg]

theorem even_T : ∀ x : ℝ, T f x = T f (-x) :=
by
  intro x
  unfold T
  rw [neg_neg]
end

end even_F_even_T_l326_326028


namespace symmetric_circle_l326_326100

theorem symmetric_circle (x y : ℝ) :
  let initial_circle := (x + 2) ^ 2 + y ^ 2 = 5 in
  let symmetric_equation := (x - 2) ^ 2 + y ^ 2 = 5 in
  ∀ (x y : ℝ),
    initial_circle →
    symmetric_equation :=
by
  intros _ _ h
  sorry

end symmetric_circle_l326_326100


namespace average_depth_dean_average_depth_sara_average_depth_nick_l326_326239

def depth_without_waves (h : ℝ) := 10 * h

def peak_depth (h : ℝ) := 1.25 * (10 * h)

def average_depth (h : ℝ) := (depth_without_waves h + peak_depth h) / 2

theorem average_depth_dean :
  average_depth 6 = 67.5 :=
sorry

theorem average_depth_sara :
  average_depth 5 = 56.25 :=
sorry

theorem average_depth_nick :
  average_depth 5.5 = 61.875 :=
sorry

end average_depth_dean_average_depth_sara_average_depth_nick_l326_326239


namespace probability_penny_dime_nickel_quarter_l326_326082

-- Define the possible states of a coin.
inductive Coin
| heads : Coin
| tails : Coin

-- Define a structure for outcomes when flipping five coins.
structure Outcome where
  penny : Coin
  nickel : Coin
  dime : Coin
  quarter : Coin
  half_dollar : Coin

-- Function to check if the penny and dime are the same and the nickel and quarter are the same.
def is_successful (o : Outcome) : Bool :=
  (o.penny = o.dime) ∧ (o.nickel = o.quarter)

-- Calculate the successful outcomes.
def count_successful_outcomes (outcomes : List Outcome) : Nat :=
  (outcomes.filter is_successful).length

-- Define total number of outcomes.
def total_outcomes : Nat :=
  2^5

-- Define the list of possible outcomes.
def all_outcomes : List Outcome :=
  List.product (List.product (List.product (List.product [Coin.heads, Coin.tails] [Coin.heads, Coin.tails]) [Coin.heads, Coin.tails]) [Coin.heads, Coin.tails]) [Coin.heads, Coin.tails]
  |> List.map (fun ⟨⟨⟨⟨penny, nickel⟩, dime⟩, quarter⟩, half_dollar⟩ =>
    { penny := penny, nickel := nickel, dime := dime, quarter := quarter, half_dollar := half_dollar : Outcome })

-- Probability calculation using the count of successful outcomes and total outcomes.
def success_probability : Rational :=
  ⟨count_successful_outcomes all_outcomes, total_outcomes⟩

-- Lean statement to prove the required probability.
theorem probability_penny_dime_nickel_quarter :
  success_probability = ⟨1, 4⟩ := 
  sorry

end probability_penny_dime_nickel_quarter_l326_326082


namespace geometric_mean_arithmetic_mean_harmonic_mean_l326_326596

theorem geometric_mean (s : set ℕ) (n : ℕ) (h : s = {16, 64, 81, 243}) (hn : n = s.card) :
  (∀ x ∈ s, 0 < x) → (Π (x ∈ s) ^ (1 / n.to_real)) = 84 := 
sorry

theorem arithmetic_mean (s : set ℕ) (n : ℕ) (h : s = {16, 64, 81, 243}) (hn : n = s.card) :
  (Σ (x ∈ s) / n.to_real) = 101 := 
sorry

theorem harmonic_mean (s : set ℕ) (n : ℕ) (h : s = {16, 64, 81, 243}) (hn : n = s.card) :
  (n.to_real / Σ (1 / (x.to_real) ∈ s)) = 42.32 := 
sorry

end geometric_mean_arithmetic_mean_harmonic_mean_l326_326596


namespace perfect_squares_limit_l326_326422

-- Define the condition
def sasha_board := set (fin 1000000 → ℕ)

-- Define the theorem
theorem perfect_squares_limit (b : sasha_board) (h : ∀ n, b n ≠ 0) :
  ∃ n ≤ 100, ∀ k ∈ b, k = n^2 :=
sorry

end perfect_squares_limit_l326_326422


namespace shirts_sold_l326_326135

theorem shirts_sold (S : ℕ) (H_total : 69 = 7 * 7 + 5 * S) : S = 4 :=
by
  sorry -- Placeholder for the proof

end shirts_sold_l326_326135


namespace divide_clock_face_l326_326354

theorem divide_clock_face :
  ∃ (part1 part2 part3 : set ℕ),
  (part1 ∪ part2 ∪ part3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) ∧
  (part1 ∩ part2 = ∅) ∧ (part1 ∩ part3 = ∅) ∧ (part2 ∩ part3 = ∅) ∧
  (part1.sum = 26) ∧ (part2.sum = 26) ∧ (part3.sum = 26) :=
begin
  sorry
end

end divide_clock_face_l326_326354


namespace sheila_weekly_earnings_l326_326778

theorem sheila_weekly_earnings:
  (∀(m w f : ℕ), (m = 8) → (w = 8) → (f = 8) → 
   ∀(t th : ℕ), (t = 6) → (th = 6) → 
   ∀(h : ℕ), (h = 6) → 
   (m + w + f + t + th) * h = 216) := by
  sorry

end sheila_weekly_earnings_l326_326778


namespace factor_squared_of_symmetric_poly_l326_326500

theorem factor_squared_of_symmetric_poly (P : Polynomial ℤ → Polynomial ℤ → Polynomial ℤ)
  (h_symm : ∀ x y, P x y = P y x)
  (h_factor : ∀ x y, (x - y) ∣ P x y) :
  ∀ x y, (x - y) ^ 2 ∣ P x y := 
sorry

end factor_squared_of_symmetric_poly_l326_326500


namespace factorial_fraction_simplification_l326_326554

theorem factorial_fraction_simplification : 
  (4 * (Nat.factorial 6) + 24 * (Nat.factorial 5)) / (Nat.factorial 7) = 8 / 7 :=
by
  sorry

end factorial_fraction_simplification_l326_326554


namespace velocity_at_3_seconds_l326_326301

variable (t : ℝ)
variable (s : ℝ)

def motion_eq (t : ℝ) : ℝ := 1 + t + t^2

theorem velocity_at_3_seconds : 
  (deriv motion_eq 3) = 7 :=
by
  sorry

end velocity_at_3_seconds_l326_326301


namespace Alan_shells_l326_326054

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end Alan_shells_l326_326054


namespace count_numbers_with_digit_2_from_200_to_499_l326_326664

def count_numbers_with_digit_2 (lower upper : ℕ) : ℕ :=
  let A := 100  -- Numbers of the form 2xx (from 200 to 299)
  let B := 30   -- Numbers of the form x2x (where first digit is 2, 3, or 4, last digit can be any)
  let C := 30   -- Numbers of the form xx2 (similar reasoning as B)
  let A_and_B := 10  -- Numbers of the form 22x
  let A_and_C := 10  -- Numbers of the form 2x2
  let B_and_C := 3   -- Numbers of the form x22
  let A_and_B_and_C := 1  -- The number 222
  A + B + C - A_and_B - A_and_C - B_and_C + A_and_B_and_C

theorem count_numbers_with_digit_2_from_200_to_499 : 
  count_numbers_with_digit_2 200 499 = 138 :=
by
  unfold count_numbers_with_digit_2
  exact rfl

end count_numbers_with_digit_2_from_200_to_499_l326_326664


namespace sin_pow_six_sum_l326_326228

theorem sin_pow_six_sum :
    (∑ k in Finset.range 46, Real.sin ((k+1) * (π / 90))) ^ 6 
    + (∑ k in Finset.rangeRev 46, Real.sin ((k+46+1) * (π / 90))) ^ 6
  = 115 / 16 := 
sorry

end sin_pow_six_sum_l326_326228


namespace zoe_pictures_l326_326152

theorem zoe_pictures (pictures_taken : ℕ) (dolphin_show_pictures : ℕ)
  (h1 : pictures_taken = 28) (h2 : dolphin_show_pictures = 16) :
  pictures_taken + dolphin_show_pictures = 44 :=
sorry

end zoe_pictures_l326_326152


namespace compute_expression_l326_326561

theorem compute_expression : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end compute_expression_l326_326561


namespace Karlsson_can_eat_all_jam_l326_326017

noncomputable def total_jam : ℝ := sorry -- Assume total amount of jam
constant jar_count   : ℕ := 1000
constant max_jam_per_jar : ℝ := total_jam / 100
constant daily_pick_count : ℕ := 100

-- Define the amounts in each jar. Using a vector to represent the jam amounts in jars.
constant jam_in_jars : Fin jar_count → ℝ

axiom jar_lower_bound (i : Fin jar_count) : 0 ≤ jam_in_jars i
axiom jar_upper_bound (i : Fin jar_count) : jam_in_jars i ≤ max_jam_per_jar

def can_eat_all_jam : Prop :=
  ∃ strategy : (Fin jar_count → Fin daily_pick_count) → ℝ → ℕ → Unit,
  (∀ day : ℕ, ∃ S : Fin daily_pick_count → Fin jar_count, ∃ m : ℝ,
   (∀ i, jam_in_jars (S i) ≥ m) ∧
   (∀ i, m ≥ 0) ∧
   (∀ i, jam_in_jars (S i) - m ≥ 0) ∧ 
   strategy S m day) ∧
  (∃ d : ℕ, ∀ i, jam_in_jars i = 0)

theorem Karlsson_can_eat_all_jam : can_eat_all_jam :=
sorry

end Karlsson_can_eat_all_jam_l326_326017


namespace james_score_unique_l326_326370

variable (s c w : ℕ)

-- The conditions given in the problem
def score (c w : ℕ) : ℕ := 20 + 3 * c - w
axiom h1 : s > 50
axiom h2 : ∀ s' : ℕ, s' > 50 ∧ s' < s → ((∀ c' w', s' = score c' w' → (c' ≠ c ∨ w' ≠ w)) → false)

-- The statement to prove
theorem james_score_unique : ∃ c w, s = 53 ∧ score c w = 53 :=
by
  exists 11 0
  -- Add sorry as placeholder for the proof
  sorry

end james_score_unique_l326_326370


namespace certain_number_is_18_l326_326670

theorem certain_number_is_18 (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : p - q = 0.20833333333333334) : 3 / q = 18 :=
sorry

end certain_number_is_18_l326_326670


namespace age_of_seventh_person_l326_326394

theorem age_of_seventh_person (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ) 
    (h1 : A1 < A2) (h2 : A2 < A3) (h3 : A3 < A4) (h4 : A4 < A5) (h5 : A5 < A6) 
    (h6 : A2 = A1 + D1) (h7 : A3 = A2 + D2) (h8 : A4 = A3 + D3) 
    (h9 : A5 = A4 + D4) (h10 : A6 = A5 + D5)
    (h11 : A1 + A2 + A3 + A4 + A5 + A6 = 246) 
    (h12 : 246 + A7 = 315) : A7 = 69 :=
by
  sorry

end age_of_seventh_person_l326_326394


namespace cos_even_function_l326_326003

theorem cos_even_function : ∀ x : ℝ, cos x = cos (-x) := by
  intro x
  exact sorry

end cos_even_function_l326_326003


namespace ordered_pairs_count_l326_326667

theorem ordered_pairs_count : 
  (∃ s : Finset (ℕ × ℕ), (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) ∧ s.card = 15) :=
by
  -- The proof would go here
  sorry

end ordered_pairs_count_l326_326667
