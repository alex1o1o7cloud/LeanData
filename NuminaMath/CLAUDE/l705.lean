import Mathlib

namespace shorter_stick_length_l705_70538

theorem shorter_stick_length (longer shorter : ℝ) 
  (h1 : longer - shorter = 12)
  (h2 : (2/3) * longer = shorter) : 
  shorter = 24 := by
  sorry

end shorter_stick_length_l705_70538


namespace absolute_value_simplification_l705_70521

theorem absolute_value_simplification : |-4^2 + 6| = 10 := by
  sorry

end absolute_value_simplification_l705_70521


namespace candy_problem_l705_70556

theorem candy_problem (initial_candy : ℕ) (num_bowls : ℕ) (removed_per_bowl : ℕ) (remaining_in_bowl : ℕ)
  (h1 : initial_candy = 100)
  (h2 : num_bowls = 4)
  (h3 : removed_per_bowl = 3)
  (h4 : remaining_in_bowl = 20) :
  initial_candy - (num_bowls * (remaining_in_bowl + removed_per_bowl)) = 8 :=
sorry

end candy_problem_l705_70556


namespace complex_square_l705_70554

theorem complex_square (z : ℂ) : z = 2 + 3*I → z^2 = -5 + 12*I := by
  sorry

end complex_square_l705_70554


namespace total_players_on_ground_l705_70519

theorem total_players_on_ground (cricket hockey football softball basketball volleyball netball rugby : ℕ) 
  (h1 : cricket = 35)
  (h2 : hockey = 28)
  (h3 : football = 33)
  (h4 : softball = 35)
  (h5 : basketball = 29)
  (h6 : volleyball = 32)
  (h7 : netball = 34)
  (h8 : rugby = 37) :
  cricket + hockey + football + softball + basketball + volleyball + netball + rugby = 263 := by
  sorry

end total_players_on_ground_l705_70519


namespace consecutive_sum_100_l705_70503

theorem consecutive_sum_100 (n : ℕ) :
  (∃ (m : ℕ), m = n ∧ 
    n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) →
  n = 18 := by
sorry

end consecutive_sum_100_l705_70503


namespace consecutive_points_theorem_l705_70562

/-- Represents a point on a straight line -/
structure Point where
  x : ℝ

/-- Represents the distance between two points -/
def distance (p q : Point) : ℝ := q.x - p.x

theorem consecutive_points_theorem (a b c d e : Point)
  (consecutive : a.x < b.x ∧ b.x < c.x ∧ c.x < d.x ∧ d.x < e.x)
  (bc_eq_2cd : distance b c = 2 * distance c d)
  (de_eq_8 : distance d e = 8)
  (ac_eq_11 : distance a c = 11)
  (ae_eq_22 : distance a e = 22) :
  distance a b = 5 := by
  sorry

end consecutive_points_theorem_l705_70562


namespace inheritance_investment_percentage_l705_70587

/-- Given an inheritance and investment scenario, prove the unknown investment percentage --/
theorem inheritance_investment_percentage 
  (total_inheritance : ℝ) 
  (known_investment : ℝ) 
  (known_rate : ℝ) 
  (total_interest : ℝ) 
  (h1 : total_inheritance = 4000)
  (h2 : known_investment = 1800)
  (h3 : known_rate = 0.065)
  (h4 : total_interest = 227)
  : ∃ (unknown_rate : ℝ), 
    known_investment * known_rate + (total_inheritance - known_investment) * unknown_rate = total_interest ∧ 
    unknown_rate = 0.05 := by
  sorry


end inheritance_investment_percentage_l705_70587


namespace rope_cost_minimum_l705_70527

/-- The cost of one foot of rope in dollars -/
def cost_per_foot : ℚ := 5 / 4

/-- The length of rope needed in feet -/
def rope_length_needed : ℚ := 5

/-- The minimum cost to buy the required length of rope -/
def min_cost : ℚ := rope_length_needed * cost_per_foot

theorem rope_cost_minimum :
  min_cost = 25 / 4 := by sorry

end rope_cost_minimum_l705_70527


namespace sufficient_unnecessary_condition_l705_70563

theorem sufficient_unnecessary_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-4) 2 → (1/2) * x^2 - a ≥ 0) ↔ a ≤ 0 :=
by sorry

end sufficient_unnecessary_condition_l705_70563


namespace garden_ratio_l705_70586

theorem garden_ratio (area width length : ℝ) : 
  area = 768 →
  width = 16 →
  area = length * width →
  length / width = 3 := by
sorry

end garden_ratio_l705_70586


namespace parabola_directrix_l705_70500

/-- The directrix of a parabola with equation y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/8 * x^2) → (∃ (k : ℝ), k = 2 ∧ k = y + 1/(4 * (1/8))) :=
by sorry

end parabola_directrix_l705_70500


namespace meeting_point_divides_segment_l705_70528

/-- The meeting point of two people moving towards each other on a line --/
def meeting_point (x₁ y₁ x₂ y₂ : ℚ) (m n : ℕ) : ℚ × ℚ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Theorem stating that the meeting point divides the line segment in the correct ratio --/
theorem meeting_point_divides_segment : 
  let mark_start : ℚ × ℚ := (2, 6)
  let sandy_start : ℚ × ℚ := (4, -2)
  let speed_ratio : ℕ × ℕ := (2, 1)
  let meet_point := meeting_point mark_start.1 mark_start.2 sandy_start.1 sandy_start.2 speed_ratio.1 speed_ratio.2
  meet_point = (8/3, 10/3) :=
by sorry

end meeting_point_divides_segment_l705_70528


namespace nora_game_probability_l705_70524

theorem nora_game_probability (p_lose : ℚ) (h1 : p_lose = 5/8) (h2 : ¬ ∃ p_tie : ℚ, p_tie > 0) :
  ∃ p_win : ℚ, p_win = 3/8 ∧ p_win + p_lose = 1 := by
  sorry

end nora_game_probability_l705_70524


namespace max_different_sums_l705_70597

def penny : ℚ := 1 / 100
def nickel : ℚ := 5 / 100
def dime : ℚ := 10 / 100
def half_dollar : ℚ := 50 / 100

def coin_set : Finset ℚ := {penny, nickel, nickel, dime, dime, half_dollar}

def sum_pairs (s : Finset ℚ) : Finset ℚ :=
  (s.product s).image (λ (x, y) => x + y)

theorem max_different_sums :
  (sum_pairs coin_set).card = 8 := by sorry

end max_different_sums_l705_70597


namespace fraction_sum_cube_l705_70588

theorem fraction_sum_cube (a b : ℝ) (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 5/2 := by
  sorry

end fraction_sum_cube_l705_70588


namespace mary_baseball_cards_l705_70580

theorem mary_baseball_cards 
  (initial_cards : ℕ) 
  (torn_cards : ℕ) 
  (cards_from_fred : ℕ) 
  (total_cards : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : torn_cards = 8) 
  (h3 : cards_from_fred = 26) 
  (h4 : total_cards = 84) : 
  total_cards - (initial_cards - torn_cards + cards_from_fred) = 48 := by
  sorry

end mary_baseball_cards_l705_70580


namespace cubic_polynomial_condition_l705_70523

/-- A polynomial is cubic if its highest degree term is of degree 3 -/
def IsCubicPolynomial (p : Polynomial ℝ) : Prop :=
  p.degree = 3

theorem cubic_polynomial_condition (m n : ℕ) :
  IsCubicPolynomial (X * Y^(m-n) + (n-2) * X^2 * Y^2 + 1) →
  m + 2*n = 8 :=
by sorry

end cubic_polynomial_condition_l705_70523


namespace power_multiplication_l705_70582

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l705_70582


namespace impossible_to_flip_all_l705_70522

/-- Represents the state of a coin: true if facing up, false if facing down -/
def Coin := Bool

/-- Represents the state of 5 coins -/
def CoinState := Fin 5 → Coin

/-- An operation that flips exactly 4 coins -/
def FlipFour (state : CoinState) : CoinState :=
  sorry

/-- The initial state where all coins are facing up -/
def initialState : CoinState := fun _ => true

/-- The target state where all coins are facing down -/
def targetState : CoinState := fun _ => false

/-- Predicate to check if a state can be reached from the initial state -/
def Reachable (state : CoinState) : Prop :=
  sorry

theorem impossible_to_flip_all :
  ¬ Reachable targetState :=
sorry

end impossible_to_flip_all_l705_70522


namespace fraction_simplification_l705_70585

theorem fraction_simplification (a b c d : ℝ) 
  (ha : a = Real.sqrt 125)
  (hb : b = 3 * Real.sqrt 45)
  (hc : c = 4 * Real.sqrt 20)
  (hd : d = Real.sqrt 75) :
  5 / (a + b + c + d) = Real.sqrt 5 / 27 := by
  sorry

end fraction_simplification_l705_70585


namespace reflect_M_x_axis_l705_70512

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point M -/
def M : ℝ × ℝ := (3, -4)

/-- Theorem stating that reflecting M across the x-axis results in (3, 4) -/
theorem reflect_M_x_axis : reflect_x M = (3, 4) := by
  sorry

end reflect_M_x_axis_l705_70512


namespace balloon_arrangements_eq_36_l705_70564

/-- The number of distinguishable arrangements of letters in "BALLOON" with vowels first -/
def balloon_arrangements : ℕ :=
  let vowels := ['A', 'O', 'O']
  let consonants := ['B', 'L', 'L', 'N']
  let vowel_arrangements := Nat.factorial 3 / Nat.factorial 2
  let consonant_arrangements := Nat.factorial 4 / Nat.factorial 2
  vowel_arrangements * consonant_arrangements

/-- Theorem stating that the number of distinguishable arrangements of "BALLOON" with vowels first is 36 -/
theorem balloon_arrangements_eq_36 : balloon_arrangements = 36 := by
  sorry

end balloon_arrangements_eq_36_l705_70564


namespace price_reduction_proof_l705_70567

/-- The original selling price in yuan -/
def original_price : ℝ := 40

/-- The cost price in yuan -/
def cost_price : ℝ := 30

/-- The initial daily sales volume -/
def initial_sales : ℕ := 48

/-- The price after two consecutive reductions in yuan -/
def reduced_price : ℝ := 32.4

/-- The increase in daily sales for every 0.5 yuan reduction in price -/
def sales_increase_rate : ℝ := 8

/-- The desired daily profit in yuan -/
def desired_profit : ℝ := 504

/-- The percentage reduction that results in the reduced price after two consecutive reductions -/
def percentage_reduction : ℝ := 0.1

/-- The price reduction that achieves the desired daily profit -/
def price_reduction : ℝ := 3

theorem price_reduction_proof :
  (∃ x : ℝ, original_price * (1 - x)^2 = reduced_price ∧ 0 < x ∧ x < 1 ∧ x = percentage_reduction) ∧
  (∃ y : ℝ, (original_price - cost_price - y) * (initial_sales + sales_increase_rate * y) = desired_profit ∧ y = price_reduction) :=
sorry

end price_reduction_proof_l705_70567


namespace louis_age_l705_70592

/-- Given that Carla will be 30 years old in 6 years and the sum of Carla and Louis's current ages is 55, prove that Louis is currently 31 years old. -/
theorem louis_age (carla_future_age : ℕ) (years_until_future : ℕ) (sum_of_ages : ℕ) :
  carla_future_age = 30 →
  years_until_future = 6 →
  sum_of_ages = 55 →
  sum_of_ages - (carla_future_age - years_until_future) = 31 := by
  sorry

end louis_age_l705_70592


namespace min_value_theorem_l705_70571

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  6 * x + 1 / x^6 ≥ 7 ∧ ∃ y > 0, 6 * y + 1 / y^6 = 7 :=
sorry

end min_value_theorem_l705_70571


namespace min_value_theorem_l705_70513

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (2 / x) + (3 / y) ≥ 1 ∧ ((2 / x) + (3 / y) = 1 ↔ x = 12 ∧ y = 8) := by
  sorry

end min_value_theorem_l705_70513


namespace decreasing_iff_a_in_range_l705_70544

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x else (2 - 3 * a) * x + 1

/-- The property that f is a decreasing function on ℝ -/
def is_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → f a x > f a y

/-- The main theorem stating the equivalence between f being decreasing and a being in (2/3, 3/4] -/
theorem decreasing_iff_a_in_range (a : ℝ) :
  is_decreasing a ↔ 2/3 < a ∧ a ≤ 3/4 := by sorry

end decreasing_iff_a_in_range_l705_70544


namespace min_cakes_to_recover_investment_l705_70511

def investment : ℕ := 8000
def revenue_per_cake : ℕ := 15
def expense_per_cake : ℕ := 5

theorem min_cakes_to_recover_investment :
  ∀ n : ℕ, n * (revenue_per_cake - expense_per_cake) ≥ investment → n ≥ 800 :=
by sorry

end min_cakes_to_recover_investment_l705_70511


namespace jessie_weight_loss_l705_70576

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (current_weight weight_lost : ℕ) 
  (h1 : current_weight = 27)
  (h2 : weight_lost = 101) :
  current_weight + weight_lost = 128 := by
  sorry

end jessie_weight_loss_l705_70576


namespace logarithm_equation_solution_l705_70525

theorem logarithm_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) 
  (h_eq : (Real.log x) / (3 * Real.log b) + (Real.log b) / (3 * Real.log x) = 1) :
  x = b ^ ((3 + Real.sqrt 5) / 2) ∨ x = b ^ ((3 - Real.sqrt 5) / 2) := by
  sorry

end logarithm_equation_solution_l705_70525


namespace min_value_of_sum_of_fractions_l705_70570

theorem min_value_of_sum_of_fractions (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 ∧ 
  ((a / b + b / c + c / d + d / a) = 4 ↔ a = b ∧ b = c ∧ c = d) :=
sorry

end min_value_of_sum_of_fractions_l705_70570


namespace marble_difference_l705_70515

/-- The number of bags Mara has -/
def mara_bags : ℕ := 12

/-- The number of marbles in each of Mara's bags -/
def mara_marbles_per_bag : ℕ := 2

/-- The number of bags Markus has -/
def markus_bags : ℕ := 2

/-- The number of marbles in each of Markus's bags -/
def markus_marbles_per_bag : ℕ := 13

/-- The difference in the total number of marbles between Markus and Mara -/
theorem marble_difference : 
  markus_bags * markus_marbles_per_bag - mara_bags * mara_marbles_per_bag = 2 := by
  sorry

end marble_difference_l705_70515


namespace team_ages_mode_l705_70549

def team_ages : List Nat := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem team_ages_mode :
  mode team_ages = 18 := by
  sorry

end team_ages_mode_l705_70549


namespace max_daily_sales_amount_l705_70537

def f (t : ℕ) : ℝ := -t + 30

def g (t : ℕ) : ℝ :=
  if t ≤ 10 then 2 * t + 40 else 15

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales_amount (t : ℕ) (h1 : 1 ≤ t) (h2 : t ≤ 20) :
  ∃ (max_t : ℕ) (max_value : ℝ), 
    (∀ t', 1 ≤ t' → t' ≤ 20 → S t' ≤ S max_t) ∧ 
    S max_t = max_value ∧ 
    max_t = 5 ∧ 
    max_value = 1250 :=
  sorry

end max_daily_sales_amount_l705_70537


namespace lucy_fish_purchase_l705_70504

/-- The number of fish Lucy bought -/
def fish_bought (initial final : ℝ) : ℝ := final - initial

/-- Proof that Lucy bought 280 fish -/
theorem lucy_fish_purchase : fish_bought 212.0 492 = 280 := by
  sorry

end lucy_fish_purchase_l705_70504


namespace solutions_of_quartic_equation_l705_70545

theorem solutions_of_quartic_equation :
  ∀ x : ℂ, x^4 - 16 = 0 ↔ x ∈ ({2, -2, 2*I, -2*I} : Set ℂ) :=
by sorry

end solutions_of_quartic_equation_l705_70545


namespace car_trip_distance_l705_70531

theorem car_trip_distance (D : ℝ) 
  (h1 : D > 0)
  (h2 : (1/2) * D + (1/4) * ((1/2) * D) + (1/3) * ((1/2) * D - (1/4) * ((1/2) * D)) + 270 = D) :
  (1/4) * D = 270 := by
  sorry

end car_trip_distance_l705_70531


namespace simplify_fraction_l705_70572

theorem simplify_fraction : 48 / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l705_70572


namespace total_fish_l705_70506

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 8) : 
  lilly_fish + rosy_fish = 18 := by
  sorry

end total_fish_l705_70506


namespace sum_of_even_integers_2_to_2022_l705_70536

theorem sum_of_even_integers_2_to_2022 : 
  (Finset.range 1011).sum (fun i => 2 * (i + 1)) = 1023112 := by
  sorry

end sum_of_even_integers_2_to_2022_l705_70536


namespace frog_corner_prob_four_hops_l705_70558

/-- Represents a position on the 4x4 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the state of the frog's movement -/
structure FrogState where
  position : Position
  hops : Nat

/-- The probability of moving to a corner in one hop from a given position -/
def cornerProbFromPosition (pos : Position) : ℚ :=
  match pos with
  | Position.Center => 0
  | Position.Edge => 1/8
  | Position.Corner => 1

/-- The probability of the frog being in a corner after n hops -/
def cornerProbAfterNHops (n : Nat) : ℚ :=
  sorry

theorem frog_corner_prob_four_hops :
  cornerProbAfterNHops 4 = 3/8 := by sorry

end frog_corner_prob_four_hops_l705_70558


namespace xyz_product_l705_70590

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end xyz_product_l705_70590


namespace race_distance_l705_70532

theorem race_distance (t1 t2 combined_time : ℝ) (h1 : t1 = 21) (h2 : t2 = 24) 
  (h3 : combined_time = 75) : 
  let d := (5 * t1 + 5 * t2) / combined_time
  d = 3 := by sorry

end race_distance_l705_70532


namespace modular_inverse_17_mod_1001_l705_70543

theorem modular_inverse_17_mod_1001 : ∃ x : ℕ, x ≤ 1000 ∧ (17 * x) % 1001 = 1 :=
by
  use 530
  sorry

end modular_inverse_17_mod_1001_l705_70543


namespace largest_divisor_of_five_consecutive_integers_l705_70533

theorem largest_divisor_of_five_consecutive_integers (a : ℤ) :
  ∃ (k : ℤ), (a - 2) + (a - 1) + a + (a + 1) + (a + 2) = 5 * k :=
sorry

end largest_divisor_of_five_consecutive_integers_l705_70533


namespace triangle_is_isosceles_l705_70529

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a = 2 * b * Real.cos C →
  b = c :=
by sorry

end triangle_is_isosceles_l705_70529


namespace water_layer_thickness_l705_70555

/-- Thickness of water layer after removing a sphere from a cylindrical vessel -/
theorem water_layer_thickness (R r : ℝ) (h_R : R = 4) (h_r : r = 3) :
  let V := π * R^2 * (2 * r)
  let V_sphere := (4/3) * π * r^3
  let V_water := V - V_sphere
  V_water / (π * R^2) = 15/4 := by
  sorry

end water_layer_thickness_l705_70555


namespace slope_range_from_angle_of_inclination_l705_70540

theorem slope_range_from_angle_of_inclination :
  ∀ a k : ℝ, 
    (π / 4 ≤ a ∧ a ≤ π / 2) →
    k = Real.tan a →
    (1 ≤ k ∧ ∀ y : ℝ, y ≥ 1 → ∃ x : ℝ, π / 4 ≤ x ∧ x ≤ π / 2 ∧ y = Real.tan x) :=
by sorry

end slope_range_from_angle_of_inclination_l705_70540


namespace units_digit_factorial_product_squared_l705_70565

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The main theorem -/
theorem units_digit_factorial_product_squared :
  unitsDigit ((factorial 1 * factorial 2 * factorial 3 * factorial 4) ^ 2) = 4 := by
  sorry

end units_digit_factorial_product_squared_l705_70565


namespace bike_distance_l705_70569

/-- Proves that the distance covered by a bike is 88 miles given the conditions -/
theorem bike_distance (time : ℝ) (truck_distance : ℝ) (speed_difference : ℝ) : 
  time = 8 → 
  truck_distance = 112 → 
  speed_difference = 3 → 
  (truck_distance / time - speed_difference) * time = 88 := by
  sorry

#check bike_distance

end bike_distance_l705_70569


namespace number_of_divisors_36_l705_70530

theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end number_of_divisors_36_l705_70530


namespace adams_students_l705_70581

/-- The number of students Adam teaches in 10 years -/
def students_in_ten_years (normal_students_per_year : ℕ) (first_year_students : ℕ) (total_years : ℕ) : ℕ :=
  first_year_students + (total_years - 1) * normal_students_per_year

/-- Theorem stating the total number of students Adam will teach in 10 years -/
theorem adams_students : 
  students_in_ten_years 50 40 10 = 490 := by
  sorry

end adams_students_l705_70581


namespace fraction_exceeding_by_20_l705_70559

theorem fraction_exceeding_by_20 (N : ℚ) (F : ℚ) : 
  N = 32 → N = F * N + 20 → F = 3/8 := by
  sorry

end fraction_exceeding_by_20_l705_70559


namespace black_tshirt_cost_black_tshirt_cost_is_30_l705_70548

/-- The cost of black t-shirts given the sale conditions -/
theorem black_tshirt_cost (total_tshirts : ℕ) (sale_duration : ℕ) 
  (white_tshirt_cost : ℕ) (revenue_per_minute : ℕ) : ℕ :=
  let total_revenue := sale_duration * revenue_per_minute
  let num_black_tshirts := total_tshirts / 2
  let num_white_tshirts := total_tshirts / 2
  let white_tshirt_revenue := num_white_tshirts * white_tshirt_cost
  let black_tshirt_revenue := total_revenue - white_tshirt_revenue
  black_tshirt_revenue / num_black_tshirts

/-- The cost of black t-shirts is $30 given the specific sale conditions -/
theorem black_tshirt_cost_is_30 : 
  black_tshirt_cost 200 25 25 220 = 30 := by
  sorry

end black_tshirt_cost_black_tshirt_cost_is_30_l705_70548


namespace trig_expression_equality_l705_70595

theorem trig_expression_equality : 
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  2 * (Real.sin (50 * π / 180) - 1) / Real.sin (40 * π / 180) := by sorry

end trig_expression_equality_l705_70595


namespace square_root_equation_solution_l705_70594

theorem square_root_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end square_root_equation_solution_l705_70594


namespace students_per_van_l705_70541

theorem students_per_van (num_vans : ℕ) (num_minibusses : ℕ) (students_per_minibus : ℕ) (total_students : ℕ) :
  num_vans = 6 →
  num_minibusses = 4 →
  students_per_minibus = 24 →
  total_students = 156 →
  (total_students - num_minibusses * students_per_minibus) / num_vans = 10 :=
by sorry

end students_per_van_l705_70541


namespace boat_speed_in_still_water_l705_70550

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 5 →
  downstream_distance = 34.47 →
  downstream_time = 44 / 60 →
  ∃ (boat_speed : ℝ), abs (boat_speed - 42.01) < 0.01 :=
by
  sorry


end boat_speed_in_still_water_l705_70550


namespace fifth_decimal_place_of_1_0025_pow_10_l705_70577

theorem fifth_decimal_place_of_1_0025_pow_10 :
  ∃ (n : ℕ) (r : ℚ), 
    (1 + 1/400)^10 = n + r ∧ 
    n < (1 + 1/400)^10 ∧
    (1 + 1/400)^10 < n + 1 ∧
    (r * 100000).floor = 8 :=
by sorry

end fifth_decimal_place_of_1_0025_pow_10_l705_70577


namespace complex_equation_solution_l705_70561

theorem complex_equation_solution (x y : ℝ) : 
  (x / (1 + Complex.I)) + (y / (1 + 2 * Complex.I)) = 5 / (1 + Complex.I) → y = 5 := by
sorry

end complex_equation_solution_l705_70561


namespace negation_of_universal_inequality_l705_70505

theorem negation_of_universal_inequality :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end negation_of_universal_inequality_l705_70505


namespace complement_of_intersection_l705_70598

def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {3, 4, 7, 8}
def U : Set ℕ := A ∪ B

theorem complement_of_intersection (A B : Set ℕ) (U : Set ℕ) (h : U = A ∪ B) :
  (A ∩ B)ᶜ = {3, 5, 8} := by
  sorry

end complement_of_intersection_l705_70598


namespace p_sufficient_not_necessary_for_q_r_necessary_not_sufficient_for_p_l705_70526

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | |3*x - 4| > 2}
def B : Set ℝ := {x : ℝ | 1 / (x^2 - x - 2) > 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | (x - a) * (x - a - 1) ≥ 0}

-- Define the propositions p, q, and r
def p (x : ℝ) : Prop := x ∉ A
def q (x : ℝ) : Prop := x ∉ B
def r (a x : ℝ) : Prop := x ∈ C a

-- Theorem 1: p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, (2/3 ≤ x ∧ x ≤ 2) → (-1 ≤ x ∧ x ≤ 2)) ∧
  (∃ x : ℝ, (-1 ≤ x ∧ x ≤ 2) ∧ ¬(2/3 ≤ x ∧ x ≤ 2)) :=
sorry

-- Theorem 2: r is a necessary but not sufficient condition for p
--            if and only if a ≥ 2 or a ≤ -1/3
theorem r_necessary_not_sufficient_for_p (a : ℝ) :
  ((∀ x : ℝ, p x → r a x) ∧ (∃ x : ℝ, r a x ∧ ¬(p x))) ↔ (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end p_sufficient_not_necessary_for_q_r_necessary_not_sufficient_for_p_l705_70526


namespace bicycle_business_loss_percentage_l705_70553

/-- Calculates the overall loss percentage for a bicycle business -/
def overall_loss_percentage (cp1 sp1 cp2 sp2 cp3 sp3 : ℚ) : ℚ :=
  let tcp := cp1 + cp2 + cp3
  let tsp := sp1 + sp2 + sp3
  let loss := tcp - tsp
  (loss / tcp) * 100

/-- Theorem stating the overall loss percentage for the given bicycle business -/
theorem bicycle_business_loss_percentage :
  let cp1 := 1000
  let sp1 := 1080
  let cp2 := 1500
  let sp2 := 1100
  let cp3 := 2000
  let sp3 := 2200
  overall_loss_percentage cp1 sp1 cp2 sp2 cp3 sp3 = 2.67 := by
  sorry


end bicycle_business_loss_percentage_l705_70553


namespace no_perfect_square_E_l705_70566

-- Define E(x) as the integer closest to x on the number line
noncomputable def E (x : ℝ) : ℤ :=
  round x

-- Theorem statement
theorem no_perfect_square_E (n : ℕ+) : ¬∃ (k : ℕ), E (n + Real.sqrt n) = k^2 := by
  sorry

end no_perfect_square_E_l705_70566


namespace special_vector_exists_l705_70568

/-- Define a new operation * for 2D vectors -/
def vec_mult (m n : Fin 2 → ℝ) : Fin 2 → ℝ := 
  λ i => if i = 0 then m 0 * n 0 + m 1 * n 1 else m 0 * n 1 + m 1 * n 0

/-- Theorem: If m * p = m for all m, then p = (1, 0) -/
theorem special_vector_exists :
  ∃ p : Fin 2 → ℝ, (∀ m : Fin 2 → ℝ, vec_mult m p = m) → p 0 = 1 ∧ p 1 = 0 :=
by
  sorry

end special_vector_exists_l705_70568


namespace vector_combination_l705_70589

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (e₁ e₂ : V)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := e₁ + 2 • e₂
def b (e₁ e₂ : V) : V := 3 • e₁ - e₂

-- State the theorem
theorem vector_combination (e₁ e₂ : V) :
  3 • (a e₁ e₂) - 2 • (b e₁ e₂) = -3 • e₁ + 8 • e₂ := by
  sorry

end vector_combination_l705_70589


namespace hall_length_width_difference_l705_70579

/-- Represents a rectangular hall with given properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  area : ℝ
  width_half_length : width = length / 2
  area_eq : area = length * width

/-- Theorem stating the difference between length and width of the hall -/
theorem hall_length_width_difference (hall : RectangularHall) 
  (h_area : hall.area = 578) : hall.length - hall.width = 17 := by
  sorry

end hall_length_width_difference_l705_70579


namespace exercise_gender_relation_l705_70552

/-- Represents the contingency table data -/
structure ContingencyTable where
  male_regular : ℕ
  female_regular : ℕ
  male_not_regular : ℕ
  female_not_regular : ℕ

/-- Calculates the chi-square value -/
def chi_square (table : ContingencyTable) : ℚ :=
  let n := table.male_regular + table.female_regular + table.male_not_regular + table.female_not_regular
  let a := table.male_regular
  let b := table.female_regular
  let c := table.male_not_regular
  let d := table.female_not_regular
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The theorem to be proved -/
theorem exercise_gender_relation (total_students : ℕ) (prob_regular : ℚ) (male_regular : ℕ) (female_not_regular : ℕ) 
    (h_total : total_students = 100)
    (h_prob : prob_regular = 1/2)
    (h_male_regular : male_regular = 35)
    (h_female_not_regular : female_not_regular = 25)
    (h_critical_value : (2706 : ℚ)/1000 < (3841 : ℚ)/1000) :
  let table := ContingencyTable.mk 
    male_regular
    (total_students / 2 - male_regular)
    (total_students / 2 - female_not_regular)
    female_not_regular
  chi_square table > (2706 : ℚ)/1000 := by
  sorry

end exercise_gender_relation_l705_70552


namespace xyz_and_fourth_power_sum_l705_70520

theorem xyz_and_fourth_power_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 1)
  (sum_sq_eq : x^2 + y^2 + z^2 = 2)
  (sum_cube_eq : x^3 + y^3 + z^3 = 3) :
  x * y * z = 1/6 ∧ x^4 + y^4 + z^4 = 25/6 := by
  sorry

end xyz_and_fourth_power_sum_l705_70520


namespace parabola_specific_point_l705_70547

def parabola_point (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = (y + 2)^2

theorem parabola_specific_point :
  let x : ℝ := Real.sqrt 704
  let y : ℝ := 88
  parabola_point x y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  Real.sqrt (x^2 + (y - 2)^2) = 90 := by sorry

end parabola_specific_point_l705_70547


namespace sum_of_roots_equals_negative_one_l705_70516

theorem sum_of_roots_equals_negative_one :
  ∀ x y : ℝ, (x - 4) * (x + 5) = 33 ∧ (y - 4) * (y + 5) = 33 → x + y = -1 := by
sorry

end sum_of_roots_equals_negative_one_l705_70516


namespace juggler_count_l705_70501

theorem juggler_count (balls_per_juggler : ℕ) (total_balls : ℕ) (h1 : balls_per_juggler = 6) (h2 : total_balls = 2268) :
  total_balls / balls_per_juggler = 378 := by
  sorry

end juggler_count_l705_70501


namespace largest_B_term_l705_70578

def B (k : ℕ) : ℝ := (Nat.choose 2000 k) * (0.1 ^ k)

theorem largest_B_term : 
  ∀ j ∈ Finset.range 2001, B 181 ≥ B j :=
sorry

end largest_B_term_l705_70578


namespace sisters_ages_l705_70574

theorem sisters_ages (s g : ℕ) : 
  (s > 0) → 
  (g > 0) → 
  (1000 ≤ g * 100 + s) → 
  (g * 100 + s < 10000) → 
  (∃ a : ℕ, g * 100 + s = a * a) →
  (∃ b : ℕ, (g + 13) * 100 + (s + 13) = b * b) →
  s + g = 55 := by
sorry

end sisters_ages_l705_70574


namespace division_inequality_l705_70591

theorem division_inequality (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end division_inequality_l705_70591


namespace equation_solutions_l705_70584

theorem equation_solutions :
  (∃ x : ℚ, 3 + 2 * x = 6 ∧ x = 3 / 2) ∧
  (∃ x : ℚ, 3 - 1 / 2 * x = 3 * x + 1 ∧ x = 4 / 7) := by
  sorry

end equation_solutions_l705_70584


namespace walnut_trees_planted_park_walnut_trees_l705_70573

/-- The number of walnut trees planted in a park -/
def trees_planted (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that the number of trees planted is the difference between final and initial counts -/
theorem walnut_trees_planted (initial : ℕ) (final : ℕ) (h : initial ≤ final) :
  trees_planted initial final = final - initial :=
by sorry

/-- The specific problem instance -/
theorem park_walnut_trees :
  trees_planted 22 55 = 33 :=
by sorry

end walnut_trees_planted_park_walnut_trees_l705_70573


namespace bryden_receives_45_dollars_l705_70510

/-- The face value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of quarters Bryden has -/
def bryden_quarters : ℕ := 6

/-- The collector's offer as a percentage of face value -/
def collector_offer_percent : ℕ := 3000

/-- Calculate the amount Bryden will receive for his quarters -/
def bryden_received : ℚ :=
  (quarter_value * bryden_quarters) * (collector_offer_percent / 100)

theorem bryden_receives_45_dollars :
  bryden_received = 45 := by sorry

end bryden_receives_45_dollars_l705_70510


namespace points_from_lines_l705_70583

/-- The number of lines formed by n points on a plane, where no three points are collinear -/
def num_lines (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that if 45 lines are formed by n points on a plane where no three are collinear, then n = 10 -/
theorem points_from_lines (n : ℕ) (h : num_lines n = 45) : n = 10 := by
  sorry

end points_from_lines_l705_70583


namespace shaded_area_is_48_l705_70508

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  area : ℝ
  small_triangle_count : ℕ
  small_triangle_area : ℝ

/-- The shaded area in an isosceles right triangle -/
def shaded_area (t : IsoscelesRightTriangle) (shaded_count : ℕ) : ℝ :=
  shaded_count * t.small_triangle_area

/-- Theorem: The shaded area of 12 small triangles in the given isosceles right triangle is 48 -/
theorem shaded_area_is_48 (t : IsoscelesRightTriangle) 
    (h1 : t.leg_length = 12)
    (h2 : t.area = 1/2 * t.leg_length * t.leg_length)
    (h3 : t.small_triangle_count = 18)
    (h4 : t.small_triangle_area = t.area / t.small_triangle_count) :
  shaded_area t 12 = 48 := by
  sorry

end shaded_area_is_48_l705_70508


namespace women_per_table_l705_70542

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 7 →
  men_per_table = 2 →
  total_customers = 63 →
  ∃ women_per_table : ℕ, women_per_table * num_tables + men_per_table * num_tables = total_customers ∧ women_per_table = 7 :=
by sorry

end women_per_table_l705_70542


namespace cow_count_l705_70509

/-- Represents the number of cows in a field with given conditions -/
def total_cows (male_cows female_cows : ℕ) : Prop :=
  female_cows = 2 * male_cows ∧
  female_cows / 2 = male_cows / 2 + 50

/-- Proves that the total number of cows is 300 given the conditions -/
theorem cow_count : ∃ (male_cows female_cows : ℕ), 
  total_cows male_cows female_cows ∧ 
  male_cows + female_cows = 300 :=
sorry

end cow_count_l705_70509


namespace farm_spiders_l705_70534

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  ducks : ℕ
  cows : ℕ
  spiders : ℕ

/-- Conditions of the farm problem --/
def farm_conditions (animals : FarmAnimals) : Prop :=
  2 * animals.ducks = 3 * animals.cows ∧
  2 * animals.ducks = 60 ∧
  2 * animals.ducks + 4 * animals.cows + 8 * animals.spiders = 270 ∧
  animals.ducks + animals.cows + animals.spiders = 70

/-- Theorem stating that under the given conditions, there are 20 spiders on the farm --/
theorem farm_spiders (animals : FarmAnimals) :
  farm_conditions animals → animals.spiders = 20 := by
  sorry

end farm_spiders_l705_70534


namespace arcsin_one_half_equals_pi_sixth_l705_70551

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_equals_pi_sixth_l705_70551


namespace max_y_value_max_y_value_achievable_l705_70546

theorem max_y_value (x y : ℤ) (h : x * y + 5 * x + 4 * y = -5) : y ≤ 10 := by
  sorry

theorem max_y_value_achievable : ∃ x y : ℤ, x * y + 5 * x + 4 * y = -5 ∧ y = 10 := by
  sorry

end max_y_value_max_y_value_achievable_l705_70546


namespace loss_fraction_for_apple_l705_70599

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem stating that for given cost price 17 and selling price 16, 
    the fraction of loss is 1/17 -/
theorem loss_fraction_for_apple : 
  fractionOfLoss 17 16 = 1 / 17 := by
  sorry

end loss_fraction_for_apple_l705_70599


namespace cubic_and_quadratic_equations_l705_70539

theorem cubic_and_quadratic_equations :
  (∃ x : ℝ, x^3 + 64 = 0 ↔ x = -4) ∧
  (∃ x : ℝ, (x - 2)^2 = 81 ↔ x = 11 ∨ x = -7) := by
  sorry

end cubic_and_quadratic_equations_l705_70539


namespace triangle_perimeter_l705_70514

theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) :
  r = 3.5 →
  A = 56 →
  A = r * (p / 2) →
  p = 32 := by
sorry

end triangle_perimeter_l705_70514


namespace unique_operation_assignment_l705_70596

-- Define the type for arithmetic operations
inductive ArithOp
| Add
| Sub
| Mul
| Div
| Eq

-- Define a function to apply an arithmetic operation
def apply_op (op : ArithOp) (x y : ℤ) : Prop :=
  match op with
  | ArithOp.Add => x + y = 0
  | ArithOp.Sub => x - y = 0
  | ArithOp.Mul => x * y = 0
  | ArithOp.Div => y ≠ 0 ∧ x / y = 0
  | ArithOp.Eq => x = y

-- Define the theorem
theorem unique_operation_assignment :
  ∃! (A B C D E : ArithOp),
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧
    (C ≠ D) ∧ (C ≠ E) ∧
    (D ≠ E) ∧
    apply_op A 4 2 ∧ apply_op B 2 2 ∧
    apply_op B 8 0 ∧ apply_op C 4 2 ∧
    apply_op D 2 3 ∧ apply_op B 5 5 ∧
    apply_op B 4 0 ∧ apply_op E 5 1 :=
sorry

end unique_operation_assignment_l705_70596


namespace german_french_fraction_l705_70593

/-- Conference language distribution -/
structure ConferenceLanguages where
  total : ℝ
  english : ℝ
  french : ℝ
  german : ℝ
  english_french : ℝ
  english_german : ℝ
  french_german : ℝ
  all_three : ℝ

/-- Language distribution satisfies the given conditions -/
def ValidDistribution (c : ConferenceLanguages) : Prop :=
  c.english_french = (1/5) * c.english ∧
  c.english_german = (1/3) * c.english ∧
  c.english_french = (1/8) * c.french ∧
  c.french_german = (1/2) * c.french ∧
  c.english_german = (1/6) * c.german

/-- The fraction of German speakers who also speak French is 2/5 -/
theorem german_french_fraction (c : ConferenceLanguages) 
  (h : ValidDistribution c) : 
  c.french_german / c.german = 2/5 := by
  sorry

end german_french_fraction_l705_70593


namespace johns_yearly_oil_change_cost_l705_70507

/-- Calculates the yearly cost of oil changes for a driver. -/
def yearly_oil_change_cost (miles_per_month : ℕ) (miles_per_oil_change : ℕ) (free_changes_per_year : ℕ) (cost_per_change : ℕ) : ℕ :=
  let changes_per_year := 12 * miles_per_month / miles_per_oil_change
  let paid_changes := changes_per_year - free_changes_per_year
  paid_changes * cost_per_change

/-- Theorem stating that John's yearly oil change cost is $150. -/
theorem johns_yearly_oil_change_cost :
  yearly_oil_change_cost 1000 3000 1 50 = 150 := by
  sorry

#eval yearly_oil_change_cost 1000 3000 1 50

end johns_yearly_oil_change_cost_l705_70507


namespace quadratic_equation_roots_l705_70575

theorem quadratic_equation_roots (k : ℝ) (h : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, k * x₁^2 + (k + 3) * x₁ + 3 = 0 ∧ k * x₂^2 + (k + 3) * x₂ + 3 = 0) ∧
  (∀ x : ℤ, k * x^2 + (k + 3) * x + 3 = 0 → k = 1 ∨ k = 3) :=
by sorry

end quadratic_equation_roots_l705_70575


namespace right_triangles_in_18gon_l705_70560

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- A right-angled triangle formed by three vertices of a regular polygon --/
structure RightTriangle (p : RegularPolygon n) where
  vertices : Fin 3 → Fin n
  is_right_angled : sorry

/-- The number of right-angled triangles in a regular polygon --/
def num_right_triangles (p : RegularPolygon n) : ℕ :=
  sorry

theorem right_triangles_in_18gon :
  ∀ (p : RegularPolygon 18), num_right_triangles p = 144 :=
sorry

end right_triangles_in_18gon_l705_70560


namespace actual_speed_is_30_l705_70557

/-- Given that increasing the speed by 10 miles per hour reduces travel time by 1/4,
    prove that the actual average speed is 30 miles per hour. -/
theorem actual_speed_is_30 (v : ℝ) (h : v / (v + 10) = 3 / 4) : v = 30 := by
  sorry

end actual_speed_is_30_l705_70557


namespace r_fourth_plus_inverse_r_fourth_l705_70502

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by
  sorry

end r_fourth_plus_inverse_r_fourth_l705_70502


namespace expression_simplification_and_evaluation_l705_70517

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 2 - 2
  ((x + 3) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 + x)) = Real.sqrt 2 - 1 :=
by sorry

end expression_simplification_and_evaluation_l705_70517


namespace power_of_128_equals_32_l705_70535

theorem power_of_128_equals_32 : (128 : ℝ) ^ (5/7 : ℝ) = 32 := by
  have h : 128 = 2^7 := by sorry
  sorry

end power_of_128_equals_32_l705_70535


namespace pollys_age_equals_sum_of_children_ages_l705_70518

/-- Represents Polly's age when it equals the sum of her three children's ages -/
def pollys_age : ℕ := 33

/-- Represents the age of Polly's first child -/
def first_child_age (x : ℕ) : ℕ := x - 20

/-- Represents the age of Polly's second child -/
def second_child_age (x : ℕ) : ℕ := x - 22

/-- Represents the age of Polly's third child -/
def third_child_age (x : ℕ) : ℕ := x - 24

/-- Theorem stating that Polly's age equals the sum of her three children's ages -/
theorem pollys_age_equals_sum_of_children_ages :
  pollys_age = first_child_age pollys_age + second_child_age pollys_age + third_child_age pollys_age :=
by sorry

end pollys_age_equals_sum_of_children_ages_l705_70518
