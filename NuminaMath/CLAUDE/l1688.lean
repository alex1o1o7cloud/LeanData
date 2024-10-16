import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_for_system_l1688_168882

/-- The system of inequalities has a unique solution for specific values of a -/
theorem unique_solution_for_system (a : ℝ) :
  (∃! x y : ℝ, x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ↔ 
  (a = 1 ∧ ∃! x y : ℝ, x = -1 ∧ y = 0 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) ∨
  (a = -3 ∧ ∃! x y : ℝ, x = 1 ∧ y = 2 ∧ x^2 + y^2 - 2*y ≤ 1 ∧ x + y + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_system_l1688_168882


namespace NUMINAMATH_CALUDE_g_of_8_equals_69_l1688_168807

-- Define the function g
def g (n : ℤ) : ℤ := n^2 - 3*n + 29

-- State the theorem
theorem g_of_8_equals_69 : g 8 = 69 := by
  sorry

end NUMINAMATH_CALUDE_g_of_8_equals_69_l1688_168807


namespace NUMINAMATH_CALUDE_sports_club_overlapping_members_l1688_168849

theorem sports_club_overlapping_members 
  (total_members : ℕ) 
  (badminton_players : ℕ) 
  (tennis_players : ℕ) 
  (neither_players : ℕ) 
  (h1 : total_members = 30)
  (h2 : badminton_players = 17)
  (h3 : tennis_players = 21)
  (h4 : neither_players = 2) :
  badminton_players + tennis_players - total_members + neither_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlapping_members_l1688_168849


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1688_168875

theorem polynomial_coefficient_sum :
  ∀ (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ),
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1688_168875


namespace NUMINAMATH_CALUDE_spider_return_probability_l1688_168871

/-- Probability of the spider being at the starting corner after n moves -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => (1 - P n) / 3

/-- The probability of returning to the starting corner on the eighth move -/
theorem spider_return_probability : P 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_spider_return_probability_l1688_168871


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l1688_168828

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 3 * x + 2 < 5 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l1688_168828


namespace NUMINAMATH_CALUDE_tangent_circles_distance_l1688_168877

/-- The distance between the centers of two tangent circles with radii 1 and 7 is either 6 or 8 -/
theorem tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 1 → r₂ = 7 → (d = r₁ + r₂ ∨ d = |r₂ - r₁|) → d = 6 ∨ d = 8 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_distance_l1688_168877


namespace NUMINAMATH_CALUDE_prob_three_heads_before_two_tails_l1688_168892

/-- The probability of getting a specific outcome when flipping a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- The state space for the coin flipping process -/
inductive CoinState
| H0  -- No heads or tails flipped yet
| H1  -- 1 consecutive head flipped
| H2  -- 2 consecutive heads flipped
| T1  -- 1 tail flipped
| HHH -- 3 consecutive heads (win state)
| TT  -- 2 consecutive tails (lose state)

/-- The probability of reaching the HHH state from a given state -/
noncomputable def prob_reach_HHH : CoinState → ℚ
| CoinState.H0 => sorry
| CoinState.H1 => sorry
| CoinState.H2 => sorry
| CoinState.T1 => sorry
| CoinState.HHH => 1
| CoinState.TT => 0

/-- The main theorem: probability of reaching HHH from the initial state is 3/8 -/
theorem prob_three_heads_before_two_tails : prob_reach_HHH CoinState.H0 = 3/8 := by sorry

end NUMINAMATH_CALUDE_prob_three_heads_before_two_tails_l1688_168892


namespace NUMINAMATH_CALUDE_alternating_number_divisibility_l1688_168839

/-- Represents a number in the form 1010101...0101 -/
def AlternatingNumber (n : ℕ) : ℕ := sorry

/-- The count of ones in an AlternatingNumber -/
def CountOnes (n : ℕ) : ℕ := sorry

theorem alternating_number_divisibility (n : ℕ) :
  (∃ k : ℕ, CountOnes n = 198 * k) ↔ AlternatingNumber n % 9999 = 0 := by sorry

end NUMINAMATH_CALUDE_alternating_number_divisibility_l1688_168839


namespace NUMINAMATH_CALUDE_discount_card_saves_money_l1688_168857

-- Define the cost of the discount card
def discount_card_cost : ℝ := 100

-- Define the discount percentage
def discount_percentage : ℝ := 0.03

-- Define the cost of cakes
def cake_cost : ℝ := 500

-- Define the number of cakes
def num_cakes : ℕ := 4

-- Define the cost of fruits
def fruit_cost : ℝ := 1600

-- Calculate the total cost without discount
def total_cost_without_discount : ℝ := cake_cost * num_cakes + fruit_cost

-- Calculate the discounted amount
def discounted_amount : ℝ := total_cost_without_discount * discount_percentage

-- Calculate the total cost with discount
def total_cost_with_discount : ℝ := 
  total_cost_without_discount - discounted_amount + discount_card_cost

-- Theorem to prove that buying the discount card saves money
theorem discount_card_saves_money : 
  total_cost_with_discount < total_cost_without_discount :=
by sorry

end NUMINAMATH_CALUDE_discount_card_saves_money_l1688_168857


namespace NUMINAMATH_CALUDE_no_three_naturals_with_pairwise_sums_as_power_of_three_l1688_168883

theorem no_three_naturals_with_pairwise_sums_as_power_of_three :
  ¬ ∃ (a b c : ℕ), 
    (∃ m : ℕ, a + b = 3^m) ∧ 
    (∃ n : ℕ, b + c = 3^n) ∧ 
    (∃ p : ℕ, c + a = 3^p) :=
sorry

end NUMINAMATH_CALUDE_no_three_naturals_with_pairwise_sums_as_power_of_three_l1688_168883


namespace NUMINAMATH_CALUDE_inequality_proof_l1688_168816

theorem inequality_proof (t : Real) (h : 0 ≤ t ∧ t ≤ π / 2) :
  Real.sqrt 2 * (Real.sin t + Real.cos t) ≥ 2 * (Real.sin (2 * t))^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1688_168816


namespace NUMINAMATH_CALUDE_revenue_calculation_l1688_168826

/-- The total revenue from selling apples and oranges -/
def total_revenue (z t : ℕ) (a b : ℚ) : ℚ :=
  z * a + t * b

/-- Theorem: The total revenue from selling 200 apples at $0.50 each and 75 oranges at $0.75 each is $156.25 -/
theorem revenue_calculation :
  total_revenue 200 75 (1/2) (3/4) = 156.25 := by
  sorry

end NUMINAMATH_CALUDE_revenue_calculation_l1688_168826


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1688_168802

theorem quadratic_root_implies_m (m : ℝ) : 
  (3^2 : ℝ) - m*3 - 6 = 0 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1688_168802


namespace NUMINAMATH_CALUDE_solve_average_age_problem_l1688_168878

def average_age_problem (T : ℝ) (original_size : ℕ) (replaced_age : ℝ) (age_decrease : ℝ) : Prop :=
  let new_size : ℕ := original_size
  let new_average : ℝ := (T - replaced_age + (T / original_size - age_decrease)) / new_size
  (T / original_size) - age_decrease = new_average

theorem solve_average_age_problem :
  ∀ (T : ℝ) (original_size : ℕ) (replaced_age : ℝ) (age_decrease : ℝ),
  original_size = 20 →
  replaced_age = 60 →
  age_decrease = 4 →
  average_age_problem T original_size replaced_age age_decrease →
  (T / original_size - age_decrease) = 40 :=
sorry

end NUMINAMATH_CALUDE_solve_average_age_problem_l1688_168878


namespace NUMINAMATH_CALUDE_log_equation_solution_l1688_168859

theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → (Real.log x / Real.log 2 = -1/2 ↔ x = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1688_168859


namespace NUMINAMATH_CALUDE_probability_x_plus_y_le_five_l1688_168860

/-- The probability of randomly selecting a point (x,y) from the rectangle [0,4] × [0,7] such that x + y ≤ 5 is equal to 5/14. -/
theorem probability_x_plus_y_le_five : 
  let total_area : ℝ := 4 * 7
  let favorable_area : ℝ := (1 / 2) * 5 * 4
  favorable_area / total_area = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_le_five_l1688_168860


namespace NUMINAMATH_CALUDE_donut_distribution_count_l1688_168824

/-- The number of ways to distribute items into bins -/
def distribute_items (total_items : ℕ) (num_bins : ℕ) (items_to_distribute : ℕ) : ℕ :=
  Nat.choose (items_to_distribute + num_bins - 1) (num_bins - 1)

/-- Theorem stating the number of ways to distribute donuts -/
theorem donut_distribution_count :
  let total_donuts : ℕ := 10
  let donut_types : ℕ := 5
  let donuts_to_distribute : ℕ := total_donuts - donut_types
  distribute_items total_donuts donut_types donuts_to_distribute = 126 := by
  sorry

#eval distribute_items 10 5 5

end NUMINAMATH_CALUDE_donut_distribution_count_l1688_168824


namespace NUMINAMATH_CALUDE_cube_root_function_l1688_168846

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, 
    prove that y = 2√3 when x = 8 -/
theorem cube_root_function (k : ℝ) :
  (∀ x : ℝ, x > 0 → k * x^(1/3) = 4 * Real.sqrt 3 → x = 64) →
  k * 8^(1/3) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_function_l1688_168846


namespace NUMINAMATH_CALUDE_sector_perimeter_l1688_168840

theorem sector_perimeter (r c θ : ℝ) (hr : r = 10) (hc : c = 10) (hθ : θ = 120 * π / 180) :
  let s := r * θ
  let p := s + c
  p = (20 * π / 3) + 10 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l1688_168840


namespace NUMINAMATH_CALUDE_special_polygon_exists_l1688_168821

/-- A polygon with the specified properties --/
structure SpecialPolygon where
  vertices : Finset (ℝ × ℝ)
  inside_square : ∀ (v : ℝ × ℝ), v ∈ vertices → v.1 ∈ [-1, 1] ∧ v.2 ∈ [-1, 1]
  side_count : vertices.card = 12
  side_length : ∀ (v w : ℝ × ℝ), v ∈ vertices → w ∈ vertices → v ≠ w →
    Real.sqrt ((v.1 - w.1)^2 + (v.2 - w.2)^2) = 1
  angle_multiples : ∀ (u v w : ℝ × ℝ), u ∈ vertices → v ∈ vertices → w ∈ vertices →
    u ≠ v → v ≠ w → u ≠ w →
    ∃ (n : ℕ), Real.cos (n * (Real.pi / 4)) = 
      ((u.1 - v.1) * (w.1 - v.1) + (u.2 - v.2) * (w.2 - v.2)) /
      (Real.sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2) * Real.sqrt ((w.1 - v.1)^2 + (w.2 - v.2)^2))

/-- The main theorem stating the existence of the special polygon --/
theorem special_polygon_exists : ∃ (p : SpecialPolygon), True := by
  sorry


end NUMINAMATH_CALUDE_special_polygon_exists_l1688_168821


namespace NUMINAMATH_CALUDE_wooden_box_height_is_6_meters_l1688_168889

def wooden_box_length : ℝ := 8
def wooden_box_width : ℝ := 10
def small_box_length : ℝ := 0.04
def small_box_width : ℝ := 0.05
def small_box_height : ℝ := 0.06
def max_small_boxes : ℕ := 4000000

theorem wooden_box_height_is_6_meters :
  let small_box_volume := small_box_length * small_box_width * small_box_height
  let total_volume := small_box_volume * max_small_boxes
  let wooden_box_height := total_volume / (wooden_box_length * wooden_box_width)
  wooden_box_height = 6 := by sorry

end NUMINAMATH_CALUDE_wooden_box_height_is_6_meters_l1688_168889


namespace NUMINAMATH_CALUDE_dans_remaining_money_l1688_168830

def remaining_money (initial_amount spending : ℚ) : ℚ :=
  initial_amount - spending

theorem dans_remaining_money :
  remaining_money 4 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l1688_168830


namespace NUMINAMATH_CALUDE_sequence_properties_l1688_168884

/-- Sequence of integers defined by a recursive formula -/
def a : ℕ → ℕ
  | 0 => 4
  | 1 => 11
  | (n + 2) => 3 * a (n + 1) - a n

/-- Theorem stating the properties of the sequence -/
theorem sequence_properties :
  ∀ n : ℕ,
    a (n + 1) > a n ∧
    Nat.gcd (a n) (a (n + 1)) = 1 ∧
    (a n ∣ a (n + 1)^2 - 5) ∧
    (a (n + 1) ∣ a n^2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1688_168884


namespace NUMINAMATH_CALUDE_triangle_sine_cosine_equality_l1688_168896

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Add conditions for a valid triangle
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π
  -- Add triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_sine_cosine_equality (t : Triangle) :
  t.b * Real.sin t.β + t.a * Real.cos t.β * Real.sin t.γ =
  t.c * Real.sin t.γ + t.a * Real.cos t.γ * Real.sin t.β := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_cosine_equality_l1688_168896


namespace NUMINAMATH_CALUDE_power_equation_solution_l1688_168815

theorem power_equation_solution :
  ∃ y : ℝ, ((1/8 : ℝ) * 2^36 = 4^y) → y = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1688_168815


namespace NUMINAMATH_CALUDE_three_pairs_probability_l1688_168866

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (cards_per_rank : Nat)
  (h1 : cards = ranks * cards_per_rank)

/-- A poker hand -/
structure PokerHand :=
  (size : Nat)

/-- The probability of drawing a specific hand -/
def probability (deck : Deck) (hand : PokerHand) (valid_hands : Nat) : Rat :=
  valid_hands / (Nat.choose deck.cards hand.size)

/-- Theorem: Probability of drawing exactly three pairs in a 6-card hand -/
theorem three_pairs_probability (d : Deck) (h : PokerHand) : 
  d.cards = 52 → d.ranks = 13 → d.cards_per_rank = 4 → h.size = 6 → 
  probability d h ((Nat.choose d.ranks 3) * (Nat.choose d.cards_per_rank 2)^3) = 154/51845 := by
  sorry


end NUMINAMATH_CALUDE_three_pairs_probability_l1688_168866


namespace NUMINAMATH_CALUDE_tan_sum_pi_third_l1688_168895

theorem tan_sum_pi_third (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_third_l1688_168895


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1688_168897

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 → 
  (a = b) →  -- isosceles condition
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- triangle inequality
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1688_168897


namespace NUMINAMATH_CALUDE_fencing_problem_l1688_168829

theorem fencing_problem (area : ℝ) (uncovered_side : ℝ) :
  area = 600 ∧ uncovered_side = 10 →
  ∃ width : ℝ, 
    area = uncovered_side * width ∧
    uncovered_side + 2 * width = 130 :=
by sorry

end NUMINAMATH_CALUDE_fencing_problem_l1688_168829


namespace NUMINAMATH_CALUDE_overestimation_correct_l1688_168834

/-- The overestimation in cents when y quarters are miscounted as half-dollars and y pennies are miscounted as nickels -/
def overestimation (y : ℕ) : ℕ := 29 * y

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a half-dollar in cents -/
def half_dollar_value : ℕ := 50

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

theorem overestimation_correct (y : ℕ) : 
  overestimation y = 
    y * (half_dollar_value - quarter_value) + 
    y * (nickel_value - penny_value) := by
  sorry

end NUMINAMATH_CALUDE_overestimation_correct_l1688_168834


namespace NUMINAMATH_CALUDE_cube_preserves_order_l1688_168870

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l1688_168870


namespace NUMINAMATH_CALUDE_cuboids_painted_l1688_168862

theorem cuboids_painted (faces_per_cuboid : ℕ) (total_faces : ℕ) (h1 : faces_per_cuboid = 6) (h2 : total_faces = 36) :
  total_faces / faces_per_cuboid = 6 :=
by sorry

end NUMINAMATH_CALUDE_cuboids_painted_l1688_168862


namespace NUMINAMATH_CALUDE_triangle_projection_similarity_l1688_168806

/-- For any triangle, there exist perpendicular distances that make the projected triangle similar to the original -/
theorem triangle_projection_similarity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
    x^2 + b^2 = y^2 + a^2 ∧
    (x - y)^2 + c^2 = y^2 + a^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_projection_similarity_l1688_168806


namespace NUMINAMATH_CALUDE_total_heads_eq_97_l1688_168822

/-- Represents the number of Lumix aliens -/
def l : ℕ := 23

/-- Represents the number of Obscra aliens -/
def o : ℕ := 37

/-- The total number of aliens -/
def total_aliens : ℕ := 60

/-- The total number of legs -/
def total_legs : ℕ := 129

/-- Lumix aliens have 1 head and 4 legs -/
axiom lumix_anatomy : l * 1 + l * 4 = l + 4 * l

/-- Obscra aliens have 2 heads and 1 leg -/
axiom obscra_anatomy : o * 2 + o * 1 = 2 * o + o

/-- The total number of aliens is 60 -/
axiom total_aliens_eq : l + o = total_aliens

/-- The total number of legs is 129 -/
axiom total_legs_eq : 4 * l + o = total_legs

/-- The theorem to be proved -/
theorem total_heads_eq_97 : l + 2 * o = 97 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_eq_97_l1688_168822


namespace NUMINAMATH_CALUDE_probability_all_red_in_hat_l1688_168888

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
  | AllRed
  | TwoGreen

/-- The probability of drawing all red chips before two green chips -/
def probability_all_red (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of drawing all red chips -/
theorem probability_all_red_in_hat :
  probability_all_red 7 4 3 = 1/7 :=
sorry

end NUMINAMATH_CALUDE_probability_all_red_in_hat_l1688_168888


namespace NUMINAMATH_CALUDE_max_square_plots_l1688_168808

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Represents the available internal fencing -/
def availableFencing : ℝ := 2400

/-- Calculates the number of square plots given the number of plots along the width -/
def numPlots (n : ℕ) : ℕ := n * n * 2

/-- Calculates the amount of internal fencing needed for a given number of plots along the width -/
def fencingNeeded (n : ℕ) (field : FieldDimensions) : ℝ :=
  (2 * n - 1) * field.width + (n - 1) * field.length

/-- The main theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
    (h_length : field.length = 60) 
    (h_width : field.width = 30) :
    ∃ (n : ℕ), 
      numPlots n = 400 ∧ 
      fencingNeeded n field ≤ availableFencing ∧
      ∀ (m : ℕ), fencingNeeded m field ≤ availableFencing → numPlots m ≤ numPlots n := by
  sorry


end NUMINAMATH_CALUDE_max_square_plots_l1688_168808


namespace NUMINAMATH_CALUDE_fishing_trip_cost_l1688_168874

/-- The fishing trip cost problem -/
theorem fishing_trip_cost (alice_paid bob_paid chris_paid : ℝ) 
  (h1 : alice_paid = 135)
  (h2 : bob_paid = 165)
  (h3 : chris_paid = 225)
  (x y : ℝ) 
  (h4 : x = (alice_paid + bob_paid + chris_paid) / 3 - alice_paid)
  (h5 : y = (alice_paid + bob_paid + chris_paid) / 3 - bob_paid) :
  x - y = 30 := by
sorry

end NUMINAMATH_CALUDE_fishing_trip_cost_l1688_168874


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1688_168804

/-- The area of a circle with circumference 12π meters is 36π square meters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 12 * π → π * r^2 = 36 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1688_168804


namespace NUMINAMATH_CALUDE_parallelogram_area_l1688_168841

/-- The area of a parallelogram with given base, slant height, and angle --/
theorem parallelogram_area (base slant_height : ℝ) (angle : ℝ) : 
  base = 24 → slant_height = 26 → angle = 40 * π / 180 →
  abs (base * (slant_height * Real.cos angle) - 478) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1688_168841


namespace NUMINAMATH_CALUDE_volleyball_team_selection_16_6_2_1_l1688_168880

def volleyball_team_selection (n : ℕ) (k : ℕ) (t : ℕ) (c : ℕ) : ℕ :=
  Nat.choose (n - t - c) (k - c) + t * Nat.choose (n - t - c) (k - c - 1)

theorem volleyball_team_selection_16_6_2_1 :
  volleyball_team_selection 16 6 2 1 = 2717 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_16_6_2_1_l1688_168880


namespace NUMINAMATH_CALUDE_equal_money_distribution_l1688_168858

/-- Represents the money distribution problem with Carmela and her cousins -/
def money_distribution (carmela_initial : ℕ) (cousin_initial : ℕ) (num_cousins : ℕ) (amount_given : ℕ) : Prop :=
  let total_money := carmela_initial + num_cousins * cousin_initial
  let people_count := num_cousins + 1
  let carmela_final := carmela_initial - num_cousins * amount_given
  let cousin_final := cousin_initial + amount_given
  (carmela_final = cousin_final) ∧ (total_money = people_count * carmela_final)

/-- Theorem stating that giving $1 to each cousin results in equal distribution -/
theorem equal_money_distribution :
  money_distribution 7 2 4 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_distribution_l1688_168858


namespace NUMINAMATH_CALUDE_water_needed_to_fill_glasses_l1688_168809

theorem water_needed_to_fill_glasses (num_glasses : ℕ) (glass_capacity : ℚ) (current_fullness : ℚ) :
  num_glasses = 10 →
  glass_capacity = 6 →
  current_fullness = 4/5 →
  (num_glasses : ℚ) * glass_capacity * (1 - current_fullness) = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_to_fill_glasses_l1688_168809


namespace NUMINAMATH_CALUDE_matrix_fourth_power_l1688_168852

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_fourth_power :
  A ^ 4 = !![(-4 : ℤ), 6; -6, 5] := by sorry

end NUMINAMATH_CALUDE_matrix_fourth_power_l1688_168852


namespace NUMINAMATH_CALUDE_weight_of_b_l1688_168845

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l1688_168845


namespace NUMINAMATH_CALUDE_no_correlation_iff_deterministic_l1688_168894

/-- Represents a pair of variables -/
inductive VariablePair
  | HeightEyesight
  | PointCoordinates
  | RiceYieldFertilizer
  | ExamScoreReviewTime
  | DistanceTime
  | IncomeHouseholdTax
  | SalesAdvertising

/-- Defines whether a pair of variables has a deterministic relationship -/
def isDeterministic (pair : VariablePair) : Prop :=
  match pair with
  | VariablePair.HeightEyesight => true
  | VariablePair.PointCoordinates => true
  | VariablePair.RiceYieldFertilizer => false
  | VariablePair.ExamScoreReviewTime => false
  | VariablePair.DistanceTime => true
  | VariablePair.IncomeHouseholdTax => false
  | VariablePair.SalesAdvertising => false

/-- Defines whether a pair of variables exhibits a correlation relationship -/
def isCorrelated (pair : VariablePair) : Prop :=
  ¬(isDeterministic pair)

/-- The main theorem stating that the pairs without correlation are exactly those with deterministic relationships -/
theorem no_correlation_iff_deterministic :
  ∀ (pair : VariablePair), ¬(isCorrelated pair) ↔ isDeterministic pair := by
  sorry

end NUMINAMATH_CALUDE_no_correlation_iff_deterministic_l1688_168894


namespace NUMINAMATH_CALUDE_second_floor_bedrooms_l1688_168801

theorem second_floor_bedrooms (total_bedrooms first_floor_bedrooms : ℕ) 
  (h1 : total_bedrooms = 10)
  (h2 : first_floor_bedrooms = 8) :
  total_bedrooms - first_floor_bedrooms = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_floor_bedrooms_l1688_168801


namespace NUMINAMATH_CALUDE_greatest_gcd_triangular_number_l1688_168885

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

/-- The greatest possible value of gcd(6T_n, n-2) is 12 -/
theorem greatest_gcd_triangular_number :
  ∃ (n : ℕ+), Nat.gcd (6 * T n) (n.val - 2) = 12 ∧
  ∀ (m : ℕ+), Nat.gcd (6 * T m) (m.val - 2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_greatest_gcd_triangular_number_l1688_168885


namespace NUMINAMATH_CALUDE_inequality_proof_l1688_168869

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + d))) + (1 / (d * (1 + a))) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1688_168869


namespace NUMINAMATH_CALUDE_stratified_optimal_survey1_simple_random_optimal_survey2_l1688_168813

/-- Represents the income level of a family -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing the conditions of Survey 1 -/
structure Survey1 where
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

/-- Structure representing the conditions of Survey 2 -/
structure Survey2 where
  totalStudents : Nat
  sampleSize : Nat

/-- Function to determine the optimal sampling method for Survey 1 -/
def optimalMethodSurvey1 (s : Survey1) : SamplingMethod := sorry

/-- Function to determine the optimal sampling method for Survey 2 -/
def optimalMethodSurvey2 (s : Survey2) : SamplingMethod := sorry

/-- Theorem stating that stratified sampling is optimal for Survey 1 -/
theorem stratified_optimal_survey1 (s : Survey1) :
  s.highIncomeFamilies = 125 →
  s.middleIncomeFamilies = 200 →
  s.lowIncomeFamilies = 95 →
  s.sampleSize = 100 →
  optimalMethodSurvey1 s = SamplingMethod.Stratified :=
by sorry

/-- Theorem stating that simple random sampling is optimal for Survey 2 -/
theorem simple_random_optimal_survey2 (s : Survey2) :
  s.totalStudents = 5 →
  s.sampleSize = 3 →
  optimalMethodSurvey2 s = SamplingMethod.SimpleRandom :=
by sorry

end NUMINAMATH_CALUDE_stratified_optimal_survey1_simple_random_optimal_survey2_l1688_168813


namespace NUMINAMATH_CALUDE_subsidy_scheme_2_maximizes_profit_l1688_168836

-- Define the daily processing capacity
def x : ℝ := 100

-- Define the constraints on x
axiom x_lower_bound : 70 ≤ x
axiom x_upper_bound : x ≤ 100

-- Define the total daily processing cost function
def total_cost (x : ℝ) : ℝ := 0.5 * x^2 + 40 * x + 3200

-- Define the selling price per ton
def selling_price : ℝ := 110

-- Define the two subsidy schemes
def subsidy_scheme_1 : ℝ := 2300
def subsidy_scheme_2 (x : ℝ) : ℝ := 30 * x

-- Define the profit functions for each subsidy scheme
def profit_scheme_1 (x : ℝ) : ℝ := selling_price * x - total_cost x + subsidy_scheme_1
def profit_scheme_2 (x : ℝ) : ℝ := selling_price * x - total_cost x + subsidy_scheme_2 x

-- Theorem: Subsidy scheme 2 maximizes profit
theorem subsidy_scheme_2_maximizes_profit :
  profit_scheme_2 x > profit_scheme_1 x :=
sorry

end NUMINAMATH_CALUDE_subsidy_scheme_2_maximizes_profit_l1688_168836


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1688_168853

def f (x : ℝ) := x^3 + 1.1*x^2 + 0.9*x - 1.4

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 0.625 0.6875, f c = 0 :=
by
  have h1 : f 0.625 < 0 := by sorry
  have h2 : f 0.6875 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l1688_168853


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l1688_168873

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 36 * x + 1

theorem f_extrema_on_interval :
  let a : ℝ := 1
  let b : ℝ := 11
  ∀ x ∈ Set.Icc a b, 
    f x ≥ -43 ∧ f x ≤ 2630 ∧ 
    (∃ x₁ ∈ Set.Icc a b, f x₁ = -43) ∧ 
    (∃ x₂ ∈ Set.Icc a b, f x₂ = 2630) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l1688_168873


namespace NUMINAMATH_CALUDE_union_of_nonnegative_and_less_than_one_is_real_l1688_168811

theorem union_of_nonnegative_and_less_than_one_is_real : 
  ({x : ℝ | x ≥ 0} ∪ {x : ℝ | x < 1}) = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_of_nonnegative_and_less_than_one_is_real_l1688_168811


namespace NUMINAMATH_CALUDE_coffee_bread_combinations_l1688_168819

theorem coffee_bread_combinations (coffee_types bread_types : ℕ) 
  (h1 : coffee_types = 2) (h2 : bread_types = 3) : 
  coffee_types * bread_types = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_bread_combinations_l1688_168819


namespace NUMINAMATH_CALUDE_point_order_on_line_l1688_168848

/-- Proves that for points (-3, y₁), (1, y₂), (-1, y₃) lying on the line y = 3x - b, 
    the relationship y₁ < y₃ < y₂ holds. -/
theorem point_order_on_line (b y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 3 * (-3) - b)
  (h₂ : y₂ = 3 * 1 - b)
  (h₃ : y₃ = 3 * (-1) - b) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_point_order_on_line_l1688_168848


namespace NUMINAMATH_CALUDE_factor_expression_1_l1688_168823

theorem factor_expression_1 (m n : ℝ) :
  4/9 * m^2 + 4/3 * m * n + n^2 = (2/3 * m + n)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_1_l1688_168823


namespace NUMINAMATH_CALUDE_next_event_occurrence_l1688_168867

/-- Represents the periodic event occurrence pattern -/
structure EventPattern where
  x : ℕ  -- Number of consecutive years the event occurs
  y : ℕ  -- Number of consecutive years of break

/-- Checks if the event occurs in a given year based on the pattern and a reference year -/
def eventOccurs (pattern : EventPattern) (referenceYear : ℕ) (year : ℕ) : Prop :=
  (year - referenceYear) % (pattern.x + pattern.y) < pattern.x

/-- The main theorem stating the next occurrence of the event after 2013 -/
theorem next_event_occurrence (pattern : EventPattern) : 
  (eventOccurs pattern 1964 1964) ∧
  (eventOccurs pattern 1964 1986) ∧
  (eventOccurs pattern 1964 1996) ∧
  (eventOccurs pattern 1964 2008) ∧
  (¬ eventOccurs pattern 1964 1976) ∧
  (¬ eventOccurs pattern 1964 1993) ∧
  (¬ eventOccurs pattern 1964 2006) ∧
  (¬ eventOccurs pattern 1964 2013) →
  ∀ year : ℕ, year > 2013 → eventOccurs pattern 1964 year → year ≥ 2018 :=
by
  sorry

#check next_event_occurrence

end NUMINAMATH_CALUDE_next_event_occurrence_l1688_168867


namespace NUMINAMATH_CALUDE_tiling_theorem_l1688_168851

/-- Represents a tile on the board -/
inductive Tile
  | SmallTile : Tile  -- 1 x 3 tile
  | LargeTile : Tile  -- 2 x 2 tile

/-- Represents the position of the 2 x 2 tile -/
inductive LargeTilePosition
  | Central : LargeTilePosition
  | Corner : LargeTilePosition

/-- Represents a board configuration -/
structure Board :=
  (size : Nat)
  (largeTilePos : LargeTilePosition)

/-- Checks if a board can be tiled -/
def canBeTiled (b : Board) : Prop :=
  match b.largeTilePos with
  | LargeTilePosition.Central => true
  | LargeTilePosition.Corner => false

/-- The main theorem to be proved -/
theorem tiling_theorem (b : Board) (h : b.size = 10000) :
  canBeTiled b ↔ b.largeTilePos = LargeTilePosition.Central :=
sorry

end NUMINAMATH_CALUDE_tiling_theorem_l1688_168851


namespace NUMINAMATH_CALUDE_comic_book_collections_l1688_168847

def kymbrea_initial : ℕ := 50
def kymbrea_rate : ℕ := 1
def lashawn_initial : ℕ := 20
def lashawn_rate : ℕ := 7
def months : ℕ := 33

theorem comic_book_collections : 
  (lashawn_initial + lashawn_rate * months) = 
  3 * (kymbrea_initial + kymbrea_rate * months) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_collections_l1688_168847


namespace NUMINAMATH_CALUDE_sector_area_for_unit_radian_l1688_168876

/-- Given a circle where the arc length corresponding to a central angle of 1 radian is 2,
    prove that the area of the sector corresponding to this central angle is 2. -/
theorem sector_area_for_unit_radian (r : ℝ) (l : ℝ) (α : ℝ) : 
  α = 1 → l = 2 → α = l / r → (1 / 2) * r * l = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_for_unit_radian_l1688_168876


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l1688_168891

/-- Given a hyperbola and a circle with specific intersection points, 
    prove that the fourth intersection point has specific coordinates. -/
theorem fourth_intersection_point 
  (C : Set (ℝ × ℝ)) -- The circle
  (h : C.Nonempty) -- The circle is not empty
  (hyp : Set (ℝ × ℝ)) -- The hyperbola
  (hyp_eq : ∀ p ∈ hyp, p.1 * p.2 = 2) -- Equation of the hyperbola
  (intersect : C ∩ hyp = {(4, 1/2), (-2, -1), (2/3, 3), (-1/2, -4)}) -- Intersection points
  : (-1/2, -4) ∈ C ∩ hyp := by
  sorry


end NUMINAMATH_CALUDE_fourth_intersection_point_l1688_168891


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1688_168835

/-- Represents a rectangular grid --/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a rectangular shaded region within the grid --/
structure ShadedRegion :=
  (start_x : ℕ)
  (start_y : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a shaded region --/
def area_of_region (region : ShadedRegion) : ℕ :=
  region.width * region.height

/-- Calculates the total area of multiple shaded regions --/
def total_shaded_area (regions : List ShadedRegion) : ℕ :=
  regions.map area_of_region |>.sum

theorem shaded_area_theorem (grid : Grid) (regions : List ShadedRegion) : 
  grid.width = 15 → 
  grid.height = 5 → 
  regions = [
    { start_x := 0, start_y := 0, width := 6, height := 3 },
    { start_x := 6, start_y := 3, width := 9, height := 2 }
  ] → 
  total_shaded_area regions = 36 := by
  sorry

#check shaded_area_theorem

end NUMINAMATH_CALUDE_shaded_area_theorem_l1688_168835


namespace NUMINAMATH_CALUDE_derivative_at_pi_over_two_l1688_168856

open Real

theorem derivative_at_pi_over_two (f : ℝ → ℝ) (hf : ∀ x, f x = sin x + 2 * x * (deriv f 0)) :
  deriv f (π / 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_over_two_l1688_168856


namespace NUMINAMATH_CALUDE_cube_volume_scaling_l1688_168831

theorem cube_volume_scaling (v : ℝ) (s : ℝ) :
  v > 0 →
  s > 0 →
  let original_side := v ^ (1/3)
  let scaled_side := s * original_side
  let scaled_volume := scaled_side ^ 3
  v = 64 ∧ s = 2 → scaled_volume = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_scaling_l1688_168831


namespace NUMINAMATH_CALUDE_power_of_power_l1688_168818

-- Define the problem statement
theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1688_168818


namespace NUMINAMATH_CALUDE_constant_term_proof_l1688_168833

theorem constant_term_proof (x y z : ℤ) (k : ℤ) : 
  x = 20 → 
  4 * x + y + z = k → 
  2 * x - y - z = 40 → 
  3 * x + y - z = 20 → 
  k = 80 := by
sorry

end NUMINAMATH_CALUDE_constant_term_proof_l1688_168833


namespace NUMINAMATH_CALUDE_g_composition_theorem_l1688_168887

/-- The function g defined as g(x) = bx^3 - 1 --/
def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - 1

/-- Theorem stating that if g(g(1)) = -1 and b is positive, then b = 1 --/
theorem g_composition_theorem (b : ℝ) (h1 : b > 0) (h2 : g b (g b 1) = -1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_theorem_l1688_168887


namespace NUMINAMATH_CALUDE_least_value_quadratic_l1688_168886

theorem least_value_quadratic (x : ℝ) : 
  (5 * x^2 + 7 * x + 3 = 6) → x ≥ (-7 - Real.sqrt 109) / 10 := by
  sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l1688_168886


namespace NUMINAMATH_CALUDE_sum_equals_negative_six_l1688_168868

theorem sum_equals_negative_six (a b c d : ℤ) :
  (∃ x : ℤ, a + 2 = x ∧ b + 3 = x ∧ c + 4 = x ∧ d + 5 = x ∧ a + b + c + d + 8 = x) →
  a + b + c + d = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_six_l1688_168868


namespace NUMINAMATH_CALUDE_green_ball_probability_l1688_168855

/-- Represents a container with balls -/
structure Container where
  green : ℕ
  red : ℕ

/-- The probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.green + c.red)

/-- The containers given in the problem -/
def containers : List Container := [
  ⟨8, 2⟩,  -- Container A
  ⟨6, 4⟩,  -- Container B
  ⟨5, 5⟩,  -- Container C
  ⟨8, 2⟩   -- Container D
]

/-- The number of containers -/
def numContainers : ℕ := containers.length

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability : 
  (1 / numContainers) * (containers.map greenProbability).sum = 43 / 160 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l1688_168855


namespace NUMINAMATH_CALUDE_petra_beads_removal_l1688_168837

/-- Represents the number of blue beads Petra has initially -/
def initial_blue_beads : ℕ := 49

/-- Represents the number of red beads Petra has initially -/
def initial_red_beads : ℕ := 1

/-- Represents the total number of beads Petra has initially -/
def initial_total_beads : ℕ := initial_blue_beads + initial_red_beads

/-- Represents the number of beads Petra needs to remove -/
def beads_to_remove : ℕ := 40

/-- Represents the desired percentage of blue beads after removal -/
def desired_blue_percentage : ℚ := 90 / 100

theorem petra_beads_removal :
  let remaining_beads := initial_total_beads - beads_to_remove
  let remaining_blue_beads := initial_blue_beads - (beads_to_remove - initial_red_beads)
  (remaining_blue_beads : ℚ) / remaining_beads = desired_blue_percentage :=
sorry

end NUMINAMATH_CALUDE_petra_beads_removal_l1688_168837


namespace NUMINAMATH_CALUDE_parabola_focus_l1688_168820

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), p = (a, b) ∧
  ∀ (x y : ℝ), parabola x y → (x - a)^2 + (y - b)^2 = (y - b + 1/4)^2

-- Theorem statement
theorem parabola_focus :
  focus (0, -1) parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1688_168820


namespace NUMINAMATH_CALUDE_number_difference_l1688_168899

theorem number_difference (x y : ℝ) (h1 : x + y = 147) (h2 : x - 0.375 * y = 4) (h3 : x ≥ y) : x - 0.375 * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1688_168899


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1688_168893

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 4 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1688_168893


namespace NUMINAMATH_CALUDE_bug_on_square_probability_l1688_168861

/-- Probability of returning to the starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 2/3 - (1/3) * P n

/-- The problem statement -/
theorem bug_on_square_probability : P 8 = 3248/6561 := by
  sorry

end NUMINAMATH_CALUDE_bug_on_square_probability_l1688_168861


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l1688_168825

theorem product_xyz_equals_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (h3 : z + 1/x = 2) : 
  x * y * z = -1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l1688_168825


namespace NUMINAMATH_CALUDE_expression_value_l1688_168879

theorem expression_value (x y z : ℝ) (hx : x = 1) (hy : y = 1) (hz : z = 3) :
  x^2 * y * z - x * y * z^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1688_168879


namespace NUMINAMATH_CALUDE_candy_total_l1688_168844

def candy_problem (tabitha stan : ℕ) : Prop :=
  ∃ (julie carlos veronica benjamin : ℕ),
    tabitha = 22 ∧
    stan = 16 ∧
    julie = tabitha / 2 ∧
    carlos = 2 * stan ∧
    veronica = julie + stan ∧
    benjamin = (tabitha + carlos) / 2 + 9 ∧
    tabitha + stan + julie + carlos + veronica + benjamin = 144

theorem candy_total : candy_problem 22 16 := by
  sorry

end NUMINAMATH_CALUDE_candy_total_l1688_168844


namespace NUMINAMATH_CALUDE_valid_galaxish_words_remainder_l1688_168872

/-- Represents the set of letters in Galaxish --/
inductive GalaxishLetter
| S
| T
| U

/-- Represents a Galaxish word as a list of letters --/
def GalaxishWord := List GalaxishLetter

/-- Checks if a letter is a consonant --/
def is_consonant (l : GalaxishLetter) : Bool :=
  match l with
  | GalaxishLetter.S => true
  | GalaxishLetter.T => true
  | GalaxishLetter.U => false

/-- Checks if a Galaxish word is valid --/
def is_valid_galaxish_word (word : GalaxishWord) : Bool :=
  let rec check (w : GalaxishWord) (consonant_count : Nat) : Bool :=
    match w with
    | [] => true
    | l::ls => 
      if is_consonant l then
        check ls (consonant_count + 1)
      else if consonant_count >= 3 then
        check ls 0
      else
        false
  check word 0

/-- Counts the number of valid 8-letter Galaxish words --/
def count_valid_galaxish_words : Nat :=
  sorry

theorem valid_galaxish_words_remainder :
  count_valid_galaxish_words % 1000 = 56 := by sorry

end NUMINAMATH_CALUDE_valid_galaxish_words_remainder_l1688_168872


namespace NUMINAMATH_CALUDE_product_simplification_l1688_168803

theorem product_simplification (y : ℝ) : 
  (y^4 + 9*y^2 + 81) * (y^2 - 9) = y^6 - 729 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l1688_168803


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1688_168805

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1688_168805


namespace NUMINAMATH_CALUDE_sum_of_powers_l1688_168814

theorem sum_of_powers : -1^2008 + (-1)^2009 + 1^2010 - 1^2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1688_168814


namespace NUMINAMATH_CALUDE_target_avg_income_l1688_168865

def past_income : List ℝ := [406, 413, 420, 436, 395]
def next_weeks : ℕ := 5
def total_weeks : ℕ := 10
def next_avg_income : ℝ := 586

theorem target_avg_income :
  let past_total := past_income.sum
  let next_total := next_avg_income * next_weeks
  let total_income := past_total + next_total
  (total_income / total_weeks : ℝ) = 500 := by sorry

end NUMINAMATH_CALUDE_target_avg_income_l1688_168865


namespace NUMINAMATH_CALUDE_lindsey_final_balance_l1688_168832

def september_savings : ℕ := 50
def october_savings : ℕ := 37
def november_savings : ℕ := 11
def mom_bonus_threshold : ℕ := 75
def mom_bonus : ℕ := 25
def video_game_cost : ℕ := 87

def total_savings : ℕ := september_savings + october_savings + november_savings

def final_balance : ℕ :=
  if total_savings > mom_bonus_threshold
  then total_savings + mom_bonus - video_game_cost
  else total_savings - video_game_cost

theorem lindsey_final_balance : final_balance = 36 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_final_balance_l1688_168832


namespace NUMINAMATH_CALUDE_rihanna_remaining_money_l1688_168817

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount : ℚ) 
  (mango_price : ℚ) (mango_count : ℕ)
  (juice_price : ℚ) (juice_count : ℕ)
  (chips_price : ℚ) (chips_count : ℕ)
  (chocolate_price : ℚ) (chocolate_count : ℕ) : ℚ :=
  initial_amount - 
  (mango_price * mango_count + 
   juice_price * juice_count + 
   chips_price * chips_count + 
   chocolate_price * chocolate_count)

/-- Theorem: Rihanna's remaining money after shopping --/
theorem rihanna_remaining_money : 
  remaining_money 50 3 6 3.5 4 2.25 2 1.75 3 = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_rihanna_remaining_money_l1688_168817


namespace NUMINAMATH_CALUDE_opposite_sides_parameter_set_is_correct_l1688_168890

/-- The set of parameter values for which points A and B lie on opposite sides of a line -/
def opposite_sides_parameter_set : Set ℝ :=
  {a | a < -2 ∨ (0 < a ∧ a < 2/3) ∨ a > 8/7}

/-- Equation of point A -/
def point_A_eq (a x y : ℝ) : Prop :=
  5 * a^2 + 12 * a * x + 4 * a * y + 8 * x^2 + 8 * x * y + 4 * y^2 = 0

/-- Equation of parabola with vertex at point B -/
def parabola_B_eq (a x y : ℝ) : Prop :=
  a * x^2 - 2 * a^2 * x - a * y + a^3 + 4 = 0

/-- Equation of the line -/
def line_eq (x y : ℝ) : Prop :=
  y - 3 * x = 4

/-- Theorem stating that the set of parameter values is correct -/
theorem opposite_sides_parameter_set_is_correct :
  ∀ a : ℝ, a ∈ opposite_sides_parameter_set ↔
    ∃ (x_A y_A x_B y_B : ℝ),
      point_A_eq a x_A y_A ∧
      parabola_B_eq a x_B y_B ∧
      ¬line_eq x_A y_A ∧
      ¬line_eq x_B y_B ∧
      (line_eq x_A y_A ↔ ¬line_eq x_B y_B) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_parameter_set_is_correct_l1688_168890


namespace NUMINAMATH_CALUDE_annular_sector_area_l1688_168812

/-- An annulus is the region between two concentric circles. -/
structure Annulus where
  R : ℝ
  r : ℝ
  h : R > r

/-- A point on the larger circle of the annulus. -/
structure PointOnLargerCircle (A : Annulus) where
  P : ℝ × ℝ
  h : (P.1 - 0)^2 + (P.2 - 0)^2 = A.R^2

/-- A point on the smaller circle of the annulus. -/
structure PointOnSmallerCircle (A : Annulus) where
  Q : ℝ × ℝ
  h : (Q.1 - 0)^2 + (Q.2 - 0)^2 = A.r^2

/-- A tangent line to the smaller circle. -/
def IsTangent (A : Annulus) (P : PointOnLargerCircle A) (Q : PointOnSmallerCircle A) : Prop :=
  (P.P.1 - Q.Q.1)^2 + (P.P.2 - Q.Q.2)^2 = A.R^2 - A.r^2

/-- The theorem stating the area of the annular sector. -/
theorem annular_sector_area (A : Annulus) (P : PointOnLargerCircle A) (Q : PointOnSmallerCircle A)
    (θ : ℝ) (t : ℝ) (h_tangent : IsTangent A P Q) (h_t : t^2 = A.R^2 - A.r^2) :
    (θ/2 - π) * A.r^2 + θ * t^2 / 2 = θ * A.R^2 / 2 - π * A.r^2 := by
  sorry

#check annular_sector_area

end NUMINAMATH_CALUDE_annular_sector_area_l1688_168812


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l1688_168864

/-- Represents the employee categories --/
inductive EmployeeCategory
  | MiddleAged
  | Young
  | Senior

/-- Represents the company's employee distribution --/
structure CompanyEmployees where
  total : ℕ
  ratio : EmployeeCategory → ℕ

/-- Represents the sampling results --/
structure SamplingResult where
  sampleSize : ℕ
  sampledEmployees : EmployeeCategory → ℕ

/-- Calculates the number of employees to be sampled for each category --/
def stratifiedSampling (company : CompanyEmployees) (sampleSize : ℕ) : SamplingResult :=
  let totalRatio := (company.ratio EmployeeCategory.MiddleAged) + 
                    (company.ratio EmployeeCategory.Young) + 
                    (company.ratio EmployeeCategory.Senior)
  let sampledEmployees (category : EmployeeCategory) :=
    (sampleSize * company.ratio category) / totalRatio
  { sampleSize := sampleSize,
    sampledEmployees := sampledEmployees }

/-- The main theorem to prove --/
theorem stratified_sampling_correct 
  (company : CompanyEmployees) 
  (sample : SamplingResult) : 
  company.total = 3200 ∧ 
  company.ratio EmployeeCategory.MiddleAged = 5 ∧
  company.ratio EmployeeCategory.Young = 3 ∧
  company.ratio EmployeeCategory.Senior = 2 ∧
  sample.sampleSize = 400 ∧
  sample = stratifiedSampling company sample.sampleSize →
  sample.sampledEmployees EmployeeCategory.MiddleAged = 200 ∧
  sample.sampledEmployees EmployeeCategory.Young = 120 ∧
  sample.sampledEmployees EmployeeCategory.Senior = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l1688_168864


namespace NUMINAMATH_CALUDE_tank_fill_time_l1688_168898

-- Define the rates of the pipes
def input_pipe_rate : ℚ := 1 / 15
def outlet_pipe_rate : ℚ := 1 / 45

-- Define the combined rate of all pipes
def combined_rate : ℚ := 2 * input_pipe_rate - outlet_pipe_rate

-- State the theorem
theorem tank_fill_time :
  (1 : ℚ) / combined_rate = 9 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1688_168898


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1688_168827

/-- Given the complex equation (1+2i)a + b = 2i, where a and b are real numbers, prove that a = 1 and b = -1. -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * 2 + 1 * (a : ℂ) + (b : ℂ) = (Complex.I : ℂ) * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1688_168827


namespace NUMINAMATH_CALUDE_initial_girls_count_l1688_168863

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 3/5 →
  ((initial_girls : ℚ) - 3) / total = 1/2 →
  initial_girls = 18 := by
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l1688_168863


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1688_168854

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1688_168854


namespace NUMINAMATH_CALUDE_worker_time_relationship_l1688_168843

/-- Given a batch of parts and a production rate, this theorem establishes
    the relationship between the number of workers and the time needed to complete the task. -/
theorem worker_time_relationship 
  (total_parts : ℕ) 
  (production_rate : ℕ) 
  (h1 : total_parts = 200)
  (h2 : production_rate = 10) :
  ∀ x y : ℝ, x > 0 → (y = (total_parts : ℝ) / (production_rate * x)) ↔ y = 20 / x :=
by sorry

end NUMINAMATH_CALUDE_worker_time_relationship_l1688_168843


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1688_168842

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1688_168842


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1688_168800

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x - 3

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1688_168800


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1688_168850

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a ≤ 1) ∧ 
  ¬(0 < a ∧ a ≤ 1 → ∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1688_168850


namespace NUMINAMATH_CALUDE_sqrt_meaningful_condition_l1688_168838

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x + 6) ↔ x ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_condition_l1688_168838


namespace NUMINAMATH_CALUDE_equation_proof_l1688_168881

theorem equation_proof : 484 + 2 * 22 * 7 + 49 = 841 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1688_168881


namespace NUMINAMATH_CALUDE_cubic_value_l1688_168810

theorem cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2010 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_l1688_168810
