import Mathlib

namespace extra_birds_calculation_l3840_384090

structure BirdPopulation where
  totalBirds : Nat
  sparrows : Nat
  robins : Nat
  bluebirds : Nat
  totalNests : Nat
  sparrowNests : Nat
  robinNests : Nat
  bluebirdNests : Nat

def extraBirds (bp : BirdPopulation) : Nat :=
  (bp.sparrows - bp.sparrowNests) + (bp.robins - bp.robinNests) + (bp.bluebirds - bp.bluebirdNests)

theorem extra_birds_calculation (bp : BirdPopulation) 
  (h1 : bp.totalBirds = bp.sparrows + bp.robins + bp.bluebirds)
  (h2 : bp.totalNests = bp.sparrowNests + bp.robinNests + bp.bluebirdNests)
  (h3 : bp.totalBirds = 18) (h4 : bp.sparrows = 10) (h5 : bp.robins = 5) (h6 : bp.bluebirds = 3)
  (h7 : bp.totalNests = 8) (h8 : bp.sparrowNests = 4) (h9 : bp.robinNests = 2) (h10 : bp.bluebirdNests = 2) :
  extraBirds bp = 10 := by
  sorry

end extra_birds_calculation_l3840_384090


namespace print_shop_price_differences_l3840_384092

/-- Represents a print shop with its pricing structure -/
structure PrintShop where
  base_price : ℝ
  discount_threshold : ℕ
  discount_rate : ℝ
  flat_discount : ℝ

/-- Calculates the price for a given number of copies at a print shop -/
def calculate_price (shop : PrintShop) (copies : ℕ) : ℝ :=
  let base_total := shop.base_price * copies
  if copies ≥ shop.discount_threshold then
    base_total * (1 - shop.discount_rate) - shop.flat_discount
  else
    base_total

/-- Theorem stating the price differences between print shops for 60 copies -/
theorem print_shop_price_differences
  (shop_x shop_y shop_z shop_w : PrintShop)
  (hx : shop_x = { base_price := 1.25, discount_threshold := 0, discount_rate := 0, flat_discount := 0 })
  (hy : shop_y = { base_price := 2.75, discount_threshold := 0, discount_rate := 0, flat_discount := 0 })
  (hz : shop_z = { base_price := 3.00, discount_threshold := 50, discount_rate := 0.1, flat_discount := 0 })
  (hw : shop_w = { base_price := 2.00, discount_threshold := 60, discount_rate := 0, flat_discount := 5 }) :
  let copies := 60
  let min_price := min (min (min (calculate_price shop_x copies) (calculate_price shop_y copies))
                            (calculate_price shop_z copies))
                       (calculate_price shop_w copies)
  (calculate_price shop_y copies - min_price = 90) ∧
  (calculate_price shop_z copies - min_price = 87) ∧
  (calculate_price shop_w copies - min_price = 40) := by
  sorry

end print_shop_price_differences_l3840_384092


namespace equation_negative_roots_a_range_l3840_384062

theorem equation_negative_roots_a_range :
  ∀ a : ℝ,
  (∀ x : ℝ, x < 0 → 4^x - 2^(x-1) + a = 0) →
  (-1/2 < a ∧ a ≤ 1/16) :=
by sorry

end equation_negative_roots_a_range_l3840_384062


namespace axe_sharpening_cost_l3840_384066

theorem axe_sharpening_cost
  (trees_per_sharpening : ℕ)
  (total_sharpening_cost : ℚ)
  (min_trees_chopped : ℕ)
  (h1 : trees_per_sharpening = 13)
  (h2 : total_sharpening_cost = 35)
  (h3 : min_trees_chopped ≥ 91) :
  let sharpenings := min_trees_chopped / trees_per_sharpening
  total_sharpening_cost / sharpenings = 5 := by
sorry

end axe_sharpening_cost_l3840_384066


namespace remaining_money_l3840_384087

/-- Calculates the remaining money after spending on sweets and giving to friends -/
theorem remaining_money 
  (initial_amount : ℚ)
  (spent_on_sweets : ℚ)
  (given_to_each_friend : ℚ)
  (number_of_friends : ℕ)
  (h1 : initial_amount = 7.1)
  (h2 : spent_on_sweets = 1.05)
  (h3 : given_to_each_friend = 1)
  (h4 : number_of_friends = 2) :
  initial_amount - spent_on_sweets - (given_to_each_friend * number_of_friends) = 4.05 := by
  sorry

#eval (7.1 : ℚ) - 1.05 - (1 * 2)  -- This should evaluate to 4.05

end remaining_money_l3840_384087


namespace yi_jianlian_shots_l3840_384027

/-- Given the basketball game statistics of Yi Jianlian, prove the number of two-point shots and free throws --/
theorem yi_jianlian_shots (total_shots : ℕ) (total_points : ℕ) (three_pointers : ℕ) 
  (h1 : total_shots = 16)
  (h2 : total_points = 28)
  (h3 : three_pointers = 3) :
  ∃ (two_pointers free_throws : ℕ),
    two_pointers + free_throws + three_pointers = total_shots ∧
    2 * two_pointers + free_throws + 3 * three_pointers = total_points ∧
    two_pointers = 6 ∧
    free_throws = 7 := by
  sorry

end yi_jianlian_shots_l3840_384027


namespace first_number_is_five_l3840_384022

/-- A sequence where each sum is 1 less than the actual sum of two numbers -/
def SpecialSequence (seq : List (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (a b c : ℕ), (a, b, c) ∈ seq → a + b = c + 1

/-- The first equation in the sequence is x + 7 = 12 -/
def FirstEquation (x : ℕ) : Prop :=
  x + 7 = 12

theorem first_number_is_five (seq : List (ℕ × ℕ × ℕ)) (x : ℕ) 
  (h1 : SpecialSequence seq) (h2 : FirstEquation x) : x = 5 := by
  sorry

end first_number_is_five_l3840_384022


namespace min_value_x2_2xy_y2_l3840_384036

theorem min_value_x2_2xy_y2 :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) := by
  sorry

end min_value_x2_2xy_y2_l3840_384036


namespace blake_grocery_change_l3840_384009

/-- Calculates the change Blake receives after purchasing groceries with discounts and sales tax. -/
theorem blake_grocery_change (oranges apples mangoes strawberries bananas : ℚ)
  (strawberry_discount banana_discount sales_tax : ℚ)
  (blake_money : ℚ)
  (h1 : oranges = 40)
  (h2 : apples = 50)
  (h3 : mangoes = 60)
  (h4 : strawberries = 30)
  (h5 : bananas = 20)
  (h6 : strawberry_discount = 10 / 100)
  (h7 : banana_discount = 5 / 100)
  (h8 : sales_tax = 7 / 100)
  (h9 : blake_money = 300) :
  let discounted_strawberries := strawberries * (1 - strawberry_discount)
  let discounted_bananas := bananas * (1 - banana_discount)
  let total_cost := oranges + apples + mangoes + discounted_strawberries + discounted_bananas
  let total_with_tax := total_cost * (1 + sales_tax)
  blake_money - total_with_tax = 90.28 := by
sorry


end blake_grocery_change_l3840_384009


namespace similar_triangles_leg_length_l3840_384015

/-- Given two similar right triangles, where one has legs 12 and 9, and the other has legs x and 6, prove that x = 8 -/
theorem similar_triangles_leg_length : 
  ∀ x : ℝ, 
  (12 : ℝ) / x = (9 : ℝ) / 6 → 
  x = 8 := by sorry

end similar_triangles_leg_length_l3840_384015


namespace sixth_score_achieves_target_mean_l3840_384038

def existing_scores : List ℝ := [76, 82, 79, 84, 91]
def target_mean : ℝ := 85
def sixth_score : ℝ := 98

theorem sixth_score_achieves_target_mean :
  let all_scores := existing_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by sorry

end sixth_score_achieves_target_mean_l3840_384038


namespace equation_is_quadratic_l3840_384097

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 - 5 * x^2 - x

theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry

end equation_is_quadratic_l3840_384097


namespace expressway_lengths_l3840_384075

theorem expressway_lengths (total : ℕ) (difference : ℕ) 
  (h1 : total = 519)
  (h2 : difference = 45) : 
  ∃ (new expanded : ℕ), 
    new + expanded = total ∧ 
    new = 2 * expanded - difference ∧
    new = 331 ∧ 
    expanded = 188 := by
  sorry

end expressway_lengths_l3840_384075


namespace complex_sum_simplification_l3840_384093

theorem complex_sum_simplification :
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 7) / 2
  z₁^8 + z₂^8 = -7.375 := by
sorry

end complex_sum_simplification_l3840_384093


namespace certain_number_proof_l3840_384057

theorem certain_number_proof (x : ℝ) : 
  0.8 * 170 - 0.35 * x = 31 → x = 300 := by
  sorry

end certain_number_proof_l3840_384057


namespace zoo_ticket_price_l3840_384072

theorem zoo_ticket_price (total_people : ℕ) (num_children : ℕ) (child_price : ℕ) (total_bill : ℕ) :
  total_people = 201 →
  num_children = 161 →
  child_price = 4 →
  total_bill = 964 →
  (total_people - num_children) * 8 + num_children * child_price = total_bill :=
by sorry

end zoo_ticket_price_l3840_384072


namespace car_average_speed_l3840_384060

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 145) (h2 : speed2 = 60) :
  (speed1 + speed2) / 2 = 102.5 := by
  sorry

end car_average_speed_l3840_384060


namespace probability_no_adjacent_red_in_ring_l3840_384019

/-- The number of red marbles -/
def num_red : ℕ := 4

/-- The number of blue marbles -/
def num_blue : ℕ := 8

/-- The total number of marbles -/
def total_marbles : ℕ := num_red + num_blue

/-- The probability of no two red marbles being adjacent when arranged in a ring -/
def probability_no_adjacent_red : ℚ := 7 / 33

/-- Theorem: The probability of no two red marbles being adjacent when 4 red marbles
    and 8 blue marbles are randomly arranged in a ring is 7/33 -/
theorem probability_no_adjacent_red_in_ring :
  probability_no_adjacent_red = 7 / 33 :=
by sorry

end probability_no_adjacent_red_in_ring_l3840_384019


namespace state_return_cost_l3840_384091

/-- The cost of a federal tax return -/
def federal_cost : ℕ := 50

/-- The cost of quarterly business taxes -/
def quarterly_cost : ℕ := 80

/-- The number of federal returns sold -/
def federal_sold : ℕ := 60

/-- The number of state returns sold -/
def state_sold : ℕ := 20

/-- The number of quarterly returns sold -/
def quarterly_sold : ℕ := 10

/-- The total revenue -/
def total_revenue : ℕ := 4400

/-- The cost of a state return -/
def state_cost : ℕ := 30

theorem state_return_cost :
  federal_cost * federal_sold + state_cost * state_sold + quarterly_cost * quarterly_sold = total_revenue :=
by sorry

end state_return_cost_l3840_384091


namespace initial_lambs_correct_l3840_384094

/-- The number of lambs Mary initially had -/
def initial_lambs : ℕ := 6

/-- The number of lambs that had babies -/
def lambs_with_babies : ℕ := 2

/-- The number of babies each lamb had -/
def babies_per_lamb : ℕ := 2

/-- The number of lambs Mary traded -/
def traded_lambs : ℕ := 3

/-- The number of extra lambs Mary found -/
def found_lambs : ℕ := 7

/-- The total number of lambs Mary has now -/
def total_lambs : ℕ := 14

/-- Theorem stating that the initial number of lambs is correct given the conditions -/
theorem initial_lambs_correct : 
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs = total_lambs :=
by sorry

end initial_lambs_correct_l3840_384094


namespace max_value_sqrt_sum_l3840_384021

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -3) :
  ∃ (max : ℝ), max = 6 * Real.sqrt 3 ∧
  ∀ (a b c : ℝ), a + b + c = 3 → a ≥ -1 → b ≥ -2 → c ≥ -3 →
  Real.sqrt (4 * a + 4) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 12) ≤ max :=
by sorry

end max_value_sqrt_sum_l3840_384021


namespace exists_permutation_with_many_swaps_l3840_384018

/-- 
Represents a permutation of cards numbered from 1 to n.
-/
def Permutation (n : ℕ) := Fin n → Fin n

/-- 
Counts the number of adjacent swaps needed to sort a permutation into descending order.
-/
def countSwaps (n : ℕ) (p : Permutation n) : ℕ := sorry

/-- 
Theorem: There exists a permutation of n cards that requires at least n(n-1)/2 adjacent swaps
to sort into descending order.
-/
theorem exists_permutation_with_many_swaps (n : ℕ) :
  ∃ (p : Permutation n), countSwaps n p ≥ n * (n - 1) / 2 := by sorry

end exists_permutation_with_many_swaps_l3840_384018


namespace age_difference_l3840_384008

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 15) : A - C = 15 := by
  sorry

end age_difference_l3840_384008


namespace total_games_proof_l3840_384043

/-- The number of baseball games Benny's high school played -/
def total_games (games_attended games_missed : ℕ) : ℕ :=
  games_attended + games_missed

/-- Theorem stating that the total number of games is the sum of attended and missed games -/
theorem total_games_proof (games_attended games_missed : ℕ) :
  total_games games_attended games_missed = games_attended + games_missed :=
by sorry

end total_games_proof_l3840_384043


namespace largest_prime_factors_difference_l3840_384070

def n : Nat := 483045

theorem largest_prime_factors_difference (p q : Nat) :
  Nat.Prime p ∧ Nat.Prime q ∧
  p ∣ n ∧ q ∣ n ∧
  (∀ r, Nat.Prime r → r ∣ n → r ≤ p ∧ r ≤ q) →
  p ≠ q →
  (max p q) - (min p q) = 8 := by
sorry

end largest_prime_factors_difference_l3840_384070


namespace isabel_albums_l3840_384054

theorem isabel_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 2)
  (h2 : camera_pics = 4)
  (h3 : pics_per_album = 2)
  : (phone_pics + camera_pics) / pics_per_album = 3 := by
  sorry

end isabel_albums_l3840_384054


namespace geometric_series_common_ratio_l3840_384033

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/144
  let r : ℚ := -10/21
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) :=
by sorry

end geometric_series_common_ratio_l3840_384033


namespace smallest_mu_inequality_l3840_384004

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∃ (μ : ℝ), ∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + b*c + μ*c*d) ∧
  (∀ (μ : ℝ), (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + b*c + μ*c*d) → μ ≥ 1) :=
by sorry

end smallest_mu_inequality_l3840_384004


namespace exists_tetrahedron_all_obtuse_dihedral_angles_l3840_384084

/-- A tetrahedron is represented by its four vertices in 3D space -/
def Tetrahedron := Fin 4 → ℝ × ℝ × ℝ

/-- The dihedral angle between two faces of a tetrahedron -/
def dihedralAngle (t : Tetrahedron) (i j : Fin 4) : ℝ :=
  sorry  -- Definition of dihedral angle calculation

/-- A dihedral angle is obtuse if it's greater than π/2 -/
def isObtuse (angle : ℝ) : Prop := angle > Real.pi / 2

/-- Theorem: There exists a tetrahedron where all dihedral angles are obtuse -/
theorem exists_tetrahedron_all_obtuse_dihedral_angles :
  ∃ t : Tetrahedron, ∀ i j : Fin 4, i ≠ j → isObtuse (dihedralAngle t i j) :=
sorry

end exists_tetrahedron_all_obtuse_dihedral_angles_l3840_384084


namespace different_suit_card_selection_l3840_384082

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def num_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 4

/-- Theorem: The number of ways to choose 4 cards from a standard deck of 52 cards,
    where all four cards must be of different suits, is equal to 28561. -/
theorem different_suit_card_selection :
  (cards_per_suit ^ cards_to_choose : ℕ) = 28561 := by
  sorry

end different_suit_card_selection_l3840_384082


namespace distinct_centroids_count_l3840_384051

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- Represents the distribution of points on the perimeter of a rectangle -/
structure PerimeterPoints where
  total : ℕ
  long_side : ℕ
  short_side : ℕ

/-- Calculates the number of distinct centroid positions for triangles formed by
    any three non-collinear points from the specified points on the rectangle's perimeter -/
def count_distinct_centroids (rect : Rectangle) (points : PerimeterPoints) : ℕ :=
  sorry

/-- The main theorem stating that for a 12x8 rectangle with 48 equally spaced points
    on its perimeter, there are 925 distinct centroid positions -/
theorem distinct_centroids_count :
  let rect := Rectangle.mk 12 8
  let points := PerimeterPoints.mk 48 16 8
  count_distinct_centroids rect points = 925 := by
  sorry

end distinct_centroids_count_l3840_384051


namespace product_of_complex_sets_l3840_384032

theorem product_of_complex_sets : ∃ (z₁ z₂ : ℂ), 
  (Complex.I * z₁ = 1) ∧ 
  (z₂ + Complex.I = 1) ∧ 
  (z₁ * z₂ = -1 - Complex.I) := by sorry

end product_of_complex_sets_l3840_384032


namespace complex_number_problem_l3840_384059

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  ∃ (r : ℝ), z₁ * z₂ = r →
  z₂ = 4 + 2 * Complex.I :=
by sorry

end complex_number_problem_l3840_384059


namespace expected_deviation_10_gt_100_l3840_384025

/-- Represents the outcome of a coin toss experiment -/
structure CoinTossExperiment where
  n : ℕ  -- number of tosses
  m : ℕ  -- number of heads
  h_m_le_n : m ≤ n  -- ensure m is not greater than n

/-- The frequency of heads in a coin toss experiment -/
def frequency (e : CoinTossExperiment) : ℚ :=
  e.m / e.n

/-- The deviation of the frequency from the probability of a fair coin (0.5) -/
def deviation (e : CoinTossExperiment) : ℚ :=
  frequency e - 1/2

/-- The absolute deviation of the frequency from the probability of a fair coin (0.5) -/
def absoluteDeviation (e : CoinTossExperiment) : ℚ :=
  |deviation e|

/-- The expected value of the absolute deviation for n coin tosses -/
noncomputable def expectedAbsoluteDeviation (n : ℕ) : ℝ :=
  sorry  -- Definition not provided in the problem, so we leave it as sorry

/-- Theorem stating that the expected absolute deviation for 10 tosses
    is greater than for 100 tosses -/
theorem expected_deviation_10_gt_100 :
  expectedAbsoluteDeviation 10 > expectedAbsoluteDeviation 100 :=
by sorry

end expected_deviation_10_gt_100_l3840_384025


namespace mia_money_l3840_384071

/-- Given that Darwin has $45 and Mia has $20 more than twice as much money as Darwin,
    prove that Mia has $110. -/
theorem mia_money (darwin_money : ℕ) (mia_money : ℕ) : 
  darwin_money = 45 → 
  mia_money = 2 * darwin_money + 20 → 
  mia_money = 110 := by
sorry

end mia_money_l3840_384071


namespace markers_per_box_l3840_384012

theorem markers_per_box (initial_markers : ℕ) (new_boxes : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32)
  (h2 : new_boxes = 6)
  (h3 : total_markers = 86) :
  (total_markers - initial_markers) / new_boxes = 9 :=
by sorry

end markers_per_box_l3840_384012


namespace officer_selection_count_l3840_384013

/-- The number of members in the club -/
def club_members : ℕ := 12

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 4

/-- The number of ways to choose officers from the club members -/
def ways_to_choose_officers : ℕ := club_members * (club_members - 1) * (club_members - 2) * (club_members - 3)

theorem officer_selection_count :
  ways_to_choose_officers = 11880 :=
sorry

end officer_selection_count_l3840_384013


namespace properties_of_A_l3840_384023

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem properties_of_A :
  (∀ a ∈ A, ∀ b : ℕ, b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a : ℕ, a ∉ A → a ≠ 1 → ∃ b : ℕ, b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
sorry

end properties_of_A_l3840_384023


namespace physics_marks_l3840_384040

/-- Represents the marks obtained in each subject --/
structure Marks where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ
  biology : ℝ
  computerScience : ℝ

/-- The conditions of the problem --/
def ProblemConditions (m : Marks) : Prop :=
  -- Average score across all subjects is 75
  (m.physics + m.chemistry + m.mathematics + m.biology + m.computerScience) / 5 = 75 ∧
  -- Average score in Physics, Mathematics, and Biology is 85
  (m.physics + m.mathematics + m.biology) / 3 = 85 ∧
  -- Average score in Physics, Chemistry, and Computer Science is 70
  (m.physics + m.chemistry + m.computerScience) / 3 = 70 ∧
  -- Weightages sum to 100%
  0.20 + 0.25 + 0.20 + 0.15 + 0.20 = 1

theorem physics_marks (m : Marks) (h : ProblemConditions m) : m.physics = 90 := by
  sorry

end physics_marks_l3840_384040


namespace infiniteSeriesSum_l3840_384029

/-- The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

/-- Theorem: The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ is equal to 3/4 -/
theorem infiniteSeriesSum : infiniteSeries = 3/4 := by
  sorry

end infiniteSeriesSum_l3840_384029


namespace ln_abs_even_and_increasing_l3840_384003

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem ln_abs_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end ln_abs_even_and_increasing_l3840_384003


namespace gaussland_olympics_l3840_384058

theorem gaussland_olympics (total_students : ℕ) (events_per_student : ℕ) (students_per_event : ℕ) (total_coaches : ℕ) 
  (h1 : total_students = 480)
  (h2 : events_per_student = 4)
  (h3 : students_per_event = 20)
  (h4 : total_coaches = 16)
  : (total_students * events_per_student) / (students_per_event * total_coaches) = 6 := by
  sorry

#check gaussland_olympics

end gaussland_olympics_l3840_384058


namespace sum_is_non_horizontal_line_l3840_384096

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure Quadratic where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original_parabola : Quadratic → ℝ → ℝ := λ q x => q.a * x^2 + q.b * x + q.c

/-- Reflection of the parabola about the x-axis -/
def reflected_parabola : Quadratic → ℝ → ℝ := λ q x => -q.a * x^2 - q.b * x - q.c

/-- Horizontal translation of a function -/
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x => f (x - h)

/-- The sum of the translated original and reflected parabolas -/
def sum_of_translated_parabolas (q : Quadratic) : ℝ → ℝ :=
  λ x => translate (original_parabola q) 3 x + translate (reflected_parabola q) (-3) x

/-- Theorem stating that the sum of translated parabolas is a non-horizontal line -/
theorem sum_is_non_horizontal_line (q : Quadratic) (h : q.a ≠ 0) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, sum_of_translated_parabolas q x = m * x + k :=
sorry

end sum_is_non_horizontal_line_l3840_384096


namespace square_and_cube_roots_l3840_384047

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 81 → x = 3 ∨ x = -3) ∧ 
  (∀ y : ℝ, y^3 = -64/125 → y = -4/5) := by
  sorry

end square_and_cube_roots_l3840_384047


namespace cosine_is_periodic_l3840_384055

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the cosine function
def cos : ℝ → ℝ := sorry

theorem cosine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric cos →
  IsPeriodic cos := by
  sorry

end cosine_is_periodic_l3840_384055


namespace equilateral_triangle_area_decrease_l3840_384088

/-- The decrease in area of an equilateral triangle when its sides are shortened --/
theorem equilateral_triangle_area_decrease :
  ∀ (s : ℝ),
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 100 * Real.sqrt 3 →
  let new_s := s - 3
  let original_area := (s^2 * Real.sqrt 3) / 4
  let new_area := (new_s^2 * Real.sqrt 3) / 4
  original_area - new_area = 27.75 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_area_decrease_l3840_384088


namespace largest_quantity_l3840_384099

theorem largest_quantity : 
  let A := (3010 : ℚ) / 3009 + 3010 / 3011
  let B := (3010 : ℚ) / 3011 + 3012 / 3011
  let C := (3011 : ℚ) / 3010 + 3011 / 3012
  A > B ∧ A > C := by
  sorry

end largest_quantity_l3840_384099


namespace inequality_solution_set_l3840_384046

theorem inequality_solution_set (x : ℝ) :
  (1 / (x^2 + 2) > 5 / x + 21 / 10) ↔ (-2 < x ∧ x < 0) :=
by sorry

end inequality_solution_set_l3840_384046


namespace triangle_theorem_l3840_384034

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific acute triangle -/
theorem triangle_theorem (t : Triangle) 
  (h_acute : 0 < t.A ∧ t.A < π/2 ∧ 0 < t.B ∧ t.B < π/2 ∧ 0 < t.C ∧ t.C < π/2)
  (h_cos : Real.cos t.A / Real.cos t.C = t.a / (2 * t.b - t.c))
  (h_a : t.a = Real.sqrt 7)
  (h_c : t.c = 3)
  (h_D : ∃ D : ℝ × ℝ, D = ((t.b + t.c)/2, 0)) :
  t.A = π/3 ∧ Real.sqrt ((t.b^2 + t.c^2 + 2*t.b*t.c*Real.cos t.A) / 4) = Real.sqrt 19 / 2 := by
  sorry

end triangle_theorem_l3840_384034


namespace isosceles_triangle_base_length_l3840_384044

theorem isosceles_triangle_base_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- triangle sides are positive
  a = b →                  -- isosceles triangle condition
  a = 5 →                  -- given leg length
  a + b > c →              -- triangle inequality
  c + a > b →              -- triangle inequality
  c ≠ 11                   -- base cannot be 11
  := by sorry

end isosceles_triangle_base_length_l3840_384044


namespace f_neg_two_eq_eleven_l3840_384028

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem: f(-2) = 11 -/
theorem f_neg_two_eq_eleven : f (-2) = 11 := by
  sorry

end f_neg_two_eq_eleven_l3840_384028


namespace smallest_prime_factor_of_5_5_minus_5_3_l3840_384006

theorem smallest_prime_factor_of_5_5_minus_5_3 :
  Nat.minFac (5^5 - 5^3) = 2 := by sorry

end smallest_prime_factor_of_5_5_minus_5_3_l3840_384006


namespace smallest_n_value_l3840_384074

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) : 
  (∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N) → N = 80 := by
  sorry

end smallest_n_value_l3840_384074


namespace initial_girls_count_l3840_384069

theorem initial_girls_count (p : ℕ) : 
  (60 : ℚ) / 100 * p = 18 ∧ 
  ((60 : ℚ) / 100 * p - 3) / p = 1 / 2 := by
  sorry

end initial_girls_count_l3840_384069


namespace max_sum_squared_distances_l3840_384001

-- Define the space
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [Finite E] [Fact (finrank ℝ E = 3)]

-- Define unit vectors
variable (a b c d : E)

-- State the theorem
theorem max_sum_squared_distances (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 :=
by sorry

end max_sum_squared_distances_l3840_384001


namespace pages_read_sunday_l3840_384052

def average_pages_per_day : ℕ := 50
def days_in_week : ℕ := 7
def pages_monday : ℕ := 65
def pages_tuesday : ℕ := 28
def pages_wednesday : ℕ := 0
def pages_thursday : ℕ := 70
def pages_friday : ℕ := 56
def pages_saturday : ℕ := 88

def total_pages_week : ℕ := average_pages_per_day * days_in_week
def pages_monday_to_friday : ℕ := pages_monday + pages_tuesday + pages_wednesday + pages_thursday + pages_friday
def pages_monday_to_saturday : ℕ := pages_monday_to_friday + pages_saturday

theorem pages_read_sunday : 
  total_pages_week - pages_monday_to_saturday = 43 := by sorry

end pages_read_sunday_l3840_384052


namespace line_quadrant_theorem_l3840_384016

/-- Represents a line in the 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line passes through a given quadrant -/
def passes_through_quadrant (l : Line) (q : ℕ) : Prop :=
  match q with
  | 1 => ∃ x > 0, l.slope * x + l.intercept > 0
  | 2 => ∃ x < 0, l.slope * x + l.intercept > 0
  | 3 => ∃ x < 0, l.slope * x + l.intercept < 0
  | 4 => ∃ x > 0, l.slope * x + l.intercept < 0
  | _ => False

/-- The main theorem -/
theorem line_quadrant_theorem (a b : ℝ) (h1 : a < 0) (h2 : b > 0) 
  (h3 : passes_through_quadrant (Line.mk a b) 1)
  (h4 : passes_through_quadrant (Line.mk a b) 2)
  (h5 : passes_through_quadrant (Line.mk a b) 4) :
  ¬ passes_through_quadrant (Line.mk b a) 2 := by
  sorry

end line_quadrant_theorem_l3840_384016


namespace negation_of_all_dogs_are_playful_l3840_384081

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a dog and being playful
variable (dog : U → Prop)
variable (playful : U → Prop)

-- State the theorem
theorem negation_of_all_dogs_are_playful :
  (¬∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬playful x) :=
sorry

end negation_of_all_dogs_are_playful_l3840_384081


namespace inequality_proof_l3840_384063

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/a^2 + 1/b^2 + 1/c^2 + 1/d^2) ≥ 36 := by
  sorry

end inequality_proof_l3840_384063


namespace equation_root_implies_a_value_l3840_384042

theorem equation_root_implies_a_value (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x - 3) = a / (3 - x) - 1) →
  a = -8 := by
  sorry

end equation_root_implies_a_value_l3840_384042


namespace library_shelves_l3840_384049

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
sorry

end library_shelves_l3840_384049


namespace count_distinct_sums_l3840_384065

def S : Finset ℕ := {2, 5, 8, 11, 14, 17, 20, 23}

def sumOfFourDistinct (s : Finset ℕ) : Finset ℕ :=
  (s.powerset.filter (fun t => t.card = 4)).image (fun t => t.sum id)

theorem count_distinct_sums : (sumOfFourDistinct S).card = 49 := by
  sorry

end count_distinct_sums_l3840_384065


namespace pear_count_l3840_384017

theorem pear_count (initial_apples : ℕ) (apple_removal_rate : ℚ) (pear_removal_rate : ℚ) :
  initial_apples = 160 →
  apple_removal_rate = 3/4 →
  pear_removal_rate = 1/3 →
  (initial_apples * (1 - apple_removal_rate) : ℚ) = (1/2 : ℚ) * (initial_pears * (1 - pear_removal_rate) : ℚ) →
  initial_pears = 120 :=
by
  sorry

end pear_count_l3840_384017


namespace smallest_integer_satisfying_conditions_l3840_384077

theorem smallest_integer_satisfying_conditions : ∃ (N x y : ℕ), 
  N > 0 ∧ 
  (N : ℚ) = 1.2 * x ∧ 
  (N : ℚ) = 0.81 * y ∧ 
  (∀ (M z w : ℕ), M > 0 → (M : ℚ) = 1.2 * z → (M : ℚ) = 0.81 * w → M ≥ N) ∧
  N = 162 :=
by sorry

end smallest_integer_satisfying_conditions_l3840_384077


namespace sales_solution_l3840_384002

def sales_problem (sales : List ℕ) (average : ℕ) : Prop :=
  sales.length = 4 ∧ 
  (sales.sum + (average * 5 - sales.sum)) / 5 = average

theorem sales_solution (sales : List ℕ) (average : ℕ) 
  (h : sales_problem sales average) : 
  average * 5 - sales.sum = (average * 5 - sales.sum) := by sorry

end sales_solution_l3840_384002


namespace sqrt_1_minus_x_real_l3840_384083

theorem sqrt_1_minus_x_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by sorry

end sqrt_1_minus_x_real_l3840_384083


namespace point_transformation_sum_l3840_384031

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90° counterclockwise around (2, 3) -/
def rotate90 (p : Point) : Point :=
  { x := -p.y + 5, y := p.x + 1 }

/-- Reflects a point about the line y = -x -/
def reflectAboutNegativeX (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- The final transformation applied to point P -/
def finalTransform (p : Point) : Point :=
  reflectAboutNegativeX (rotate90 p)

/-- Theorem statement -/
theorem point_transformation_sum (a b : ℝ) :
  let p := Point.mk a b
  finalTransform p = Point.mk (-3) 2 → a + b = 11 := by
  sorry

end point_transformation_sum_l3840_384031


namespace sum_of_perpendiculars_l3840_384010

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the inscribed circle
structure InscribedCircle (t : EquilateralTriangle) where
  center : ℝ × ℝ  -- Representing the center point P
  radius : ℝ
  radius_pos : radius > 0
  touches_sides : True  -- Assumption that the circle touches all sides

-- Define the perpendicular distances
def perpendicular_distances (t : EquilateralTriangle) (c : InscribedCircle t) : ℝ × ℝ × ℝ :=
  (c.radius, c.radius, c.radius)

-- Theorem statement
theorem sum_of_perpendiculars (t : EquilateralTriangle) (c : InscribedCircle t) :
  let (d1, d2, d3) := perpendicular_distances t c
  d1 + d2 + d3 = (Real.sqrt 3 * t.side_length) / 2 :=
sorry

end sum_of_perpendiculars_l3840_384010


namespace max_pencils_theorem_l3840_384076

/-- Represents the discount rules for pencil purchases -/
structure DiscountRules where
  large_set : Nat
  large_discount : Rat
  small_set : Nat
  small_discount : Rat

/-- Calculates the maximum number of pencils that can be purchased given initial funds and discount rules -/
def max_pencils (initial_funds : Nat) (rules : DiscountRules) : Nat :=
  sorry

/-- The theorem stating that given the specific initial funds and discount rules, the maximum number of pencils that can be purchased is 36 -/
theorem max_pencils_theorem (initial_funds : Nat) (rules : DiscountRules) :
  initial_funds = 30 ∧
  rules.large_set = 20 ∧
  rules.large_discount = 1/4 ∧
  rules.small_set = 5 ∧
  rules.small_discount = 1/10
  → max_pencils initial_funds rules = 36 := by
  sorry

end max_pencils_theorem_l3840_384076


namespace distance_Cara_approx_l3840_384053

/-- The distance between two skaters on a frozen lake --/
def distance_CD : ℝ := 100

/-- Cara's skating speed in meters per second --/
def speed_Cara : ℝ := 9

/-- Danny's skating speed in meters per second --/
def speed_Danny : ℝ := 6

/-- The angle between Cara's path and the line CD in degrees --/
def angle_Cara : ℝ := 75

/-- The time it takes for Cara and Danny to meet --/
noncomputable def meeting_time : ℝ := 
  let a : ℝ := 45
  let b : ℝ := -1800 * Real.cos (angle_Cara * Real.pi / 180)
  let c : ℝ := 10000
  (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

/-- The distance Cara skates before meeting Danny --/
noncomputable def distance_Cara : ℝ := speed_Cara * meeting_time

/-- Theorem stating that the distance Cara skates is approximately 27.36144 meters --/
theorem distance_Cara_approx : 
  ∃ ε > 0, abs (distance_Cara - 27.36144) < ε :=
by sorry

end distance_Cara_approx_l3840_384053


namespace simplify_expression_l3840_384098

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (6 * b^5) = 144 * b^11 := by
  sorry

end simplify_expression_l3840_384098


namespace complex_number_in_fourth_quadrant_l3840_384014

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (1 : ℂ) / ((1 + Complex.I)^2 + 1) + Complex.I^4
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l3840_384014


namespace largest_divisor_five_consecutive_integers_l3840_384073

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end largest_divisor_five_consecutive_integers_l3840_384073


namespace cubic_roots_theorem_l3840_384039

open Complex

-- Define the cubic equation
def cubic_equation (p q x : ℂ) : Prop := x^3 + p*x + q = 0

-- Define the condition for roots forming an equilateral triangle
def roots_form_equilateral_triangle (r₁ r₂ r₃ : ℂ) : Prop :=
  abs (r₁ - r₂) = Real.sqrt 3 ∧
  abs (r₂ - r₃) = Real.sqrt 3 ∧
  abs (r₃ - r₁) = Real.sqrt 3

theorem cubic_roots_theorem (p q : ℂ) :
  (∃ r₁ r₂ r₃ : ℂ, 
    cubic_equation p q r₁ ∧
    cubic_equation p q r₂ ∧
    cubic_equation p q r₃ ∧
    roots_form_equilateral_triangle r₁ r₂ r₃) →
  arg q = 2 * Real.pi / 3 →
  p + q = -1/2 + (Real.sqrt 3 / 2) * I :=
by sorry

end cubic_roots_theorem_l3840_384039


namespace power_multiplication_l3840_384045

theorem power_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end power_multiplication_l3840_384045


namespace expression_evaluation_l3840_384041

theorem expression_evaluation : 
  ∃ ε > 0, |((10 * 1.8 - 2 * 1.5) / 0.3 + Real.rpow 3 (2/3) - Real.log 4) - 50.6938| < ε :=
by sorry

end expression_evaluation_l3840_384041


namespace f_monotone_increasing_and_g_bound_l3840_384020

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (2 * x) / (x + 2)

noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

theorem f_monotone_increasing_and_g_bound (a : ℝ) :
  (∀ x > 0, Monotone f) ∧
  (∀ x > 0, g x < x + a ↔ a > -3) := by sorry

end f_monotone_increasing_and_g_bound_l3840_384020


namespace binomial_distribution_parameters_l3840_384086

variable (ξ : ℕ → ℝ)
variable (n : ℕ)
variable (p : ℝ)

-- ξ follows a binomial distribution B(n, p)
def is_binomial (ξ : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop := sorry

-- Expected value of ξ
def expectation (ξ : ℕ → ℝ) : ℝ := sorry

-- Variance of ξ
def variance (ξ : ℕ → ℝ) : ℝ := sorry

theorem binomial_distribution_parameters 
  (h1 : is_binomial ξ n p)
  (h2 : expectation ξ = 5/3)
  (h3 : variance ξ = 10/9) :
  n = 5 ∧ p = 1/3 := by sorry

end binomial_distribution_parameters_l3840_384086


namespace binomial_coefficient_equation_l3840_384080

theorem binomial_coefficient_equation : 
  ∀ n : ℤ, (Nat.choose 25 n.toNat + Nat.choose 25 12 = Nat.choose 26 13) ↔ (n = 11 ∨ n = 13) := by
  sorry

end binomial_coefficient_equation_l3840_384080


namespace inequality_solution_l3840_384085

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
by sorry

end inequality_solution_l3840_384085


namespace opposite_number_l3840_384030

theorem opposite_number (x : ℤ) : (- x = 2016) → (x = -2016) := by
  sorry

end opposite_number_l3840_384030


namespace a_range_l3840_384064

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) → 
  a > -1 ∧ a < 1 :=
sorry

end a_range_l3840_384064


namespace class_composition_l3840_384089

/-- Represents the percentage of men in a college class -/
def percentage_men : ℝ := 40

/-- Represents the percentage of women in a college class -/
def percentage_women : ℝ := 100 - percentage_men

/-- Represents the percentage of women who are science majors -/
def women_science_percentage : ℝ := 20

/-- Represents the percentage of non-science majors in the class -/
def non_science_percentage : ℝ := 60

/-- Represents the percentage of men who are science majors -/
def men_science_percentage : ℝ := 70

theorem class_composition :
  percentage_men + percentage_women = 100 ∧
  women_science_percentage / 100 * percentage_women +
    men_science_percentage / 100 * percentage_men = 100 - non_science_percentage :=
by sorry

end class_composition_l3840_384089


namespace right_triangle_hypotenuse_hypotenuse_length_l3840_384050

theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a > 0 ∧ b > 0) :
  (∃ (m_a m_b : ℝ), 
    m_a^2 = b^2 + (a/2)^2 ∧
    m_b^2 = a^2 + (b/2)^2 ∧
    m_a = 6 ∧
    m_b = Real.sqrt 34) →
  a^2 + b^2 = 56 :=
by sorry

theorem hypotenuse_length (a b : ℝ) (h_right : a > 0 ∧ b > 0) :
  a^2 + b^2 = 56 →
  Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 14 :=
by sorry

end right_triangle_hypotenuse_hypotenuse_length_l3840_384050


namespace functional_equation_solution_l3840_384024

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x + 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∀ x : ℝ, f x = x + 1 := by
  sorry

end functional_equation_solution_l3840_384024


namespace polynomial_expansion_l3840_384056

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 1) * (-2 * t^2 + 3 * t - 5) =
  -6 * t^5 + 5 * t^4 - t^3 - 24 * t^2 + 23 * t - 5 := by
  sorry

end polynomial_expansion_l3840_384056


namespace residual_plot_ordinate_l3840_384048

/-- Represents a residual plot used in residual analysis -/
structure ResidualPlot where
  /-- The ordinate of the residual plot -/
  ordinate : ℝ
  /-- The abscissa of the residual plot (could be sample number, height data, or estimated weight) -/
  abscissa : ℝ

/-- Represents a residual in statistical analysis -/
def Residual : Type := ℝ

/-- Theorem stating that the ordinate of a residual plot represents the residual -/
theorem residual_plot_ordinate (plot : ResidualPlot) : 
  ∃ (r : Residual), plot.ordinate = r :=
sorry

end residual_plot_ordinate_l3840_384048


namespace wang_speed_inequality_l3840_384095

theorem wang_speed_inequality (a b v : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) 
  (hv : v = 2 * a * b / (a + b)) : a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end wang_speed_inequality_l3840_384095


namespace girls_percentage_after_adding_boy_l3840_384000

def initial_boys : ℕ := 11
def initial_girls : ℕ := 13
def added_boys : ℕ := 1

def total_students : ℕ := initial_boys + initial_girls + added_boys

def girls_percentage : ℚ := (initial_girls : ℚ) / (total_students : ℚ) * 100

theorem girls_percentage_after_adding_boy :
  girls_percentage = 52 := by sorry

end girls_percentage_after_adding_boy_l3840_384000


namespace quadratic_solution_sum_l3840_384035

theorem quadratic_solution_sum (a b : ℝ) : 
  (a^2 - 6*a + 11 = 23) → 
  (b^2 - 6*b + 11 = 23) → 
  (a ≥ b) → 
  (a + 3*b = 12 - 2*Real.sqrt 21) := by
sorry

end quadratic_solution_sum_l3840_384035


namespace stratified_by_educational_stage_is_most_reasonable_l3840_384007

-- Define the different sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define the educational stages
inductive EducationalStage
| Primary
| JuniorHigh
| HighSchool

-- Define the conditions
def significantDifferencesInEducationalStages : Prop := True
def noSignificantDifferencesBetweenGenders : Prop := True
def goalIsUnderstandVisionConditions : Prop := True

-- Define the most reasonable sampling method
def mostReasonableSamplingMethod : SamplingMethod := SamplingMethod.StratifiedByEducationalStage

-- Theorem statement
theorem stratified_by_educational_stage_is_most_reasonable :
  significantDifferencesInEducationalStages →
  noSignificantDifferencesBetweenGenders →
  goalIsUnderstandVisionConditions →
  mostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by
  sorry

end stratified_by_educational_stage_is_most_reasonable_l3840_384007


namespace matrix_property_implies_k_one_and_n_even_l3840_384005

open Matrix

theorem matrix_property_implies_k_one_and_n_even 
  (k n : ℕ) 
  (hk : k ≥ 1) 
  (hn : n ≥ 2) 
  (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : A ^ 3 = 0)
  (h2 : A ^ k * B + B * A = 1) :
  k = 1 ∧ Even n :=
sorry

end matrix_property_implies_k_one_and_n_even_l3840_384005


namespace sum_of_digits_M_l3840_384078

-- Define a function to represent the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define a predicate to check if a number only uses allowed digits
def uses_allowed_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_M (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_allowed_digits M)
  (h_double : sum_of_digits (2 * M) = 31)
  (h_half : sum_of_digits (M / 2) = 28) :
  sum_of_digits M = 29 := by sorry

end sum_of_digits_M_l3840_384078


namespace arithmetic_sequence_k_value_l3840_384061

/-- An arithmetic sequence with specified terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  a2_eq_3 : a 2 = 3
  a4_eq_7 : a 4 = 7

/-- The theorem stating that k = 8 for the given conditions -/
theorem arithmetic_sequence_k_value (seq : ArithmeticSequence) :
  ∃ k : ℕ, seq.a k = 15 ∧ k = 8 := by
  sorry


end arithmetic_sequence_k_value_l3840_384061


namespace curve_equation_represents_quadrants_l3840_384026

-- Define the circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the right quadrant of the circle
def right_quadrant (x y : ℝ) : Prop := x = Real.sqrt (1 - y^2) ∧ x ≥ 0

-- Define the lower quadrant of the circle
def lower_quadrant (x y : ℝ) : Prop := y = -Real.sqrt (1 - x^2) ∧ y ≤ 0

-- Theorem stating the equation represents the right and lower quadrants of the unit circle
theorem curve_equation_represents_quadrants :
  ∀ x y : ℝ, unit_circle x y →
  ((x - Real.sqrt (1 - y^2)) * (y + Real.sqrt (1 - x^2)) = 0) ↔
  (right_quadrant x y ∨ lower_quadrant x y) :=
sorry

end curve_equation_represents_quadrants_l3840_384026


namespace inequality_solution_l3840_384068

theorem inequality_solution (x : ℝ) : (x^2 - 4) / (x^2 - 9) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end inequality_solution_l3840_384068


namespace abs_sum_inequality_l3840_384011

theorem abs_sum_inequality (x : ℝ) : |x + 3| + |x - 4| < 8 ↔ 4 ≤ x ∧ x < 4.5 := by
  sorry

end abs_sum_inequality_l3840_384011


namespace two_zeros_iff_a_positive_l3840_384067

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

theorem two_zeros_iff_a_positive (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ 
   ∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ↔ 
  a > 0 :=
sorry

end two_zeros_iff_a_positive_l3840_384067


namespace oil_depth_in_cylindrical_tank_l3840_384079

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : Real
  diameter : Real

/-- Represents the oil in the tank --/
structure Oil where
  depth : Real
  surface_area : Real

theorem oil_depth_in_cylindrical_tank
  (tank : HorizontalCylindricalTank)
  (oil : Oil)
  (h_length : tank.length = 12)
  (h_diameter : tank.diameter = 4)
  (h_surface_area : oil.surface_area = 24) :
  oil.depth = 2 - Real.sqrt 3 ∨ oil.depth = 2 + Real.sqrt 3 := by
  sorry

end oil_depth_in_cylindrical_tank_l3840_384079


namespace f_negative_five_equals_negative_five_l3840_384037

/-- Given a function f(x) = a * sin(x) + b * tan(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem f_negative_five_equals_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 1)
  (h2 : f 5 = 7) :
  f (-5) = -5 := by
  sorry

end f_negative_five_equals_negative_five_l3840_384037
