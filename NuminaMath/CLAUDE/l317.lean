import Mathlib

namespace NUMINAMATH_CALUDE_average_speed_calculation_l317_31745

def initial_reading : ℕ := 3223
def final_reading : ℕ := 3443
def total_time : ℕ := 12

def distance : ℕ := final_reading - initial_reading

def average_speed : ℚ := distance / total_time

theorem average_speed_calculation : 
  (average_speed : ℚ) = 55/3 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l317_31745


namespace NUMINAMATH_CALUDE_regular_price_of_bread_l317_31789

/-- The regular price of a full pound of bread, given sale conditions -/
theorem regular_price_of_bread (sale_price : ℝ) (discount_rate : ℝ) : 
  sale_price = 2 →
  discount_rate = 0.6 →
  ∃ (regular_price : ℝ), 
    regular_price = 20 ∧ 
    sale_price = (1 - discount_rate) * (regular_price / 4) :=
by sorry

end NUMINAMATH_CALUDE_regular_price_of_bread_l317_31789


namespace NUMINAMATH_CALUDE_average_pqr_l317_31746

theorem average_pqr (p q r : ℝ) (h : (5 / 4) * (p + q + r) = 15) :
  (p + q + r) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_pqr_l317_31746


namespace NUMINAMATH_CALUDE_card_sorting_theorem_l317_31748

/-- A function that represents the cost of sorting n cards -/
def sortingCost (n : ℕ) : ℕ := sorry

/-- The theorem states that 365 cards can be sorted within 2000 comparisons -/
theorem card_sorting_theorem :
  ∃ (f : ℕ → ℕ), 
    (∀ n ≤ 365, f n ≤ sortingCost n) ∧ 
    (f 365 ≤ 2000) := by
  sorry

/-- The cost of sorting 3 cards is 1 -/
axiom sort_three_cost : sortingCost 3 = 1

/-- The cost of sorting n+1 cards is at most k+1 if n ≤ 3^k -/
axiom sort_cost_bound (n k : ℕ) :
  n ≤ 3^k → sortingCost (n + 1) ≤ sortingCost n + k + 1

/-- There are 365 cards -/
def total_cards : ℕ := 365

/-- The maximum allowed cost is 2000 -/
def max_cost : ℕ := 2000

end NUMINAMATH_CALUDE_card_sorting_theorem_l317_31748


namespace NUMINAMATH_CALUDE_binomial_10_5_l317_31784

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_5_l317_31784


namespace NUMINAMATH_CALUDE_crease_length_is_twenty_thirds_l317_31719

/-- Represents a right triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- Represents the crease formed when point A is folded onto the midpoint of side BC -/
def crease_length (t : RightTriangle) : ℝ := sorry

/-- Theorem stating that the length of the crease is 20/3 inches -/
theorem crease_length_is_twenty_thirds (t : RightTriangle) :
  crease_length t = 20/3 := by sorry

end NUMINAMATH_CALUDE_crease_length_is_twenty_thirds_l317_31719


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_sum_product_l317_31712

theorem cubic_polynomial_root_sum_product (α β γ : ℂ) :
  (α + β + γ = -4) →
  (α * β * γ = 14) →
  (α^3 + 4*α^2 + 5*α - 14 = 0) →
  (β^3 + 4*β^2 + 5*β - 14 = 0) →
  (γ^3 + 4*γ^2 + 5*γ - 14 = 0) →
  ∃ p q : ℂ, (α+β)^3 + p*(α+β)^2 + q*(α+β) + 34 = 0 ∧
            (β+γ)^3 + p*(β+γ)^2 + q*(β+γ) + 34 = 0 ∧
            (γ+α)^3 + p*(γ+α)^2 + q*(γ+α) + 34 = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_sum_product_l317_31712


namespace NUMINAMATH_CALUDE_blue_chips_count_l317_31729

theorem blue_chips_count (total : ℚ) 
  (h1 : total * (1 / 10) + total * (1 / 2) + 12 = total) : 
  total * (1 / 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_chips_count_l317_31729


namespace NUMINAMATH_CALUDE_matches_played_is_ten_l317_31737

/-- The number of matches a player has played, given their current average and the effect of a future match on that average. -/
def matches_played (current_average : ℚ) (future_score : ℚ) (average_increase : ℚ) : ℕ :=
  let n : ℕ := sorry
  n

/-- Theorem stating that the number of matches played is 10 under the given conditions. -/
theorem matches_played_is_ten :
  matches_played 32 76 4 = 10 := by sorry

end NUMINAMATH_CALUDE_matches_played_is_ten_l317_31737


namespace NUMINAMATH_CALUDE_wooden_stick_sawing_theorem_l317_31783

/-- Represents the sawing of a wooden stick into segments -/
structure WoodenStickSawing where
  num_segments : ℕ
  total_time : ℕ
  
/-- Calculates the average time per cut for a wooden stick sawing -/
def average_time_per_cut (sawing : WoodenStickSawing) : ℚ :=
  sawing.total_time / (sawing.num_segments - 1)

/-- Theorem stating that for a wooden stick sawed into 5 segments in 20 minutes,
    the average time per cut is 5 minutes -/
theorem wooden_stick_sawing_theorem (sawing : WoodenStickSawing) 
    (h1 : sawing.num_segments = 5) 
    (h2 : sawing.total_time = 20) : 
    average_time_per_cut sawing = 5 := by
  sorry

end NUMINAMATH_CALUDE_wooden_stick_sawing_theorem_l317_31783


namespace NUMINAMATH_CALUDE_flag_run_time_l317_31741

/-- The time taken to run between equally spaced flags -/
def run_time (start_flag end_flag : ℕ) (time : ℚ) : Prop :=
  start_flag < end_flag ∧ time > 0 ∧
  ∀ (i j : ℕ), start_flag ≤ i ∧ i < j ∧ j ≤ end_flag →
    (time * (j - i : ℚ)) / (end_flag - start_flag : ℚ) =
    time * ((j - start_flag : ℚ) / (end_flag - start_flag : ℚ) - (i - start_flag : ℚ) / (end_flag - start_flag : ℚ))

theorem flag_run_time :
  run_time 1 8 8 → run_time 1 12 (88/7 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_flag_run_time_l317_31741


namespace NUMINAMATH_CALUDE_matrix_N_property_l317_31760

theorem matrix_N_property (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ w : Fin 3 → ℝ, N.mulVec w = (3 : ℝ) • w) ↔
  N = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] :=
by sorry

end NUMINAMATH_CALUDE_matrix_N_property_l317_31760


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l317_31771

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → (∀ a b : ℕ, a^2 - b^2 = 187 → x^2 + y^2 ≤ a^2 + b^2) → 
  x^2 + y^2 = 205 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l317_31771


namespace NUMINAMATH_CALUDE_westerville_gnomes_l317_31718

theorem westerville_gnomes (ravenswood westerville : ℕ) : 
  ravenswood = 4 * westerville →
  (60 * ravenswood) / 100 = 48 →
  westerville = 20 := by
sorry

end NUMINAMATH_CALUDE_westerville_gnomes_l317_31718


namespace NUMINAMATH_CALUDE_inverse_proportion_l317_31731

/-- Given that x is inversely proportional to y, prove that if x = 4 when y = 2, then x = 4/5 when y = 10 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  x * 10 = k → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l317_31731


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l317_31751

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a1 : a 1 = 3)
  (h_a3 : a 3 = 12) :
  a 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l317_31751


namespace NUMINAMATH_CALUDE_outfit_count_is_18900_l317_31799

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def blue_shirts : ℕ := 8
def pants : ℕ := 9
def green_hats : ℕ := 10
def red_hats : ℕ := 10
def blue_hats : ℕ := 10
def ties_per_color : ℕ := 5

def valid_outfit (shirt_color hat_color : String) : Bool :=
  shirt_color ≠ hat_color

def count_outfits_for_hat_color (hat_color : String) : ℕ :=
  match hat_color with
  | "green" => (red_shirts + blue_shirts) * pants * green_hats * ties_per_color
  | "red" => (green_shirts + blue_shirts) * pants * red_hats * ties_per_color
  | "blue" => (red_shirts + green_shirts) * pants * blue_hats * ties_per_color
  | _ => 0

def total_outfits : ℕ :=
  count_outfits_for_hat_color "green" +
  count_outfits_for_hat_color "red" +
  count_outfits_for_hat_color "blue"

theorem outfit_count_is_18900 : total_outfits = 18900 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_is_18900_l317_31799


namespace NUMINAMATH_CALUDE_number_problem_l317_31766

theorem number_problem (x : ℝ) : 0.5 * x = 0.8 * 150 + 80 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l317_31766


namespace NUMINAMATH_CALUDE_contains_quadrilateral_l317_31723

/-- A plane graph with n vertices and m edges, where no three points are collinear -/
structure PlaneGraph where
  n : ℕ
  m : ℕ
  no_collinear_triple : True  -- Placeholder for the condition that no three points are collinear

/-- Theorem: If m > (1/4)n(1 + √(4n - 3)) in a plane graph, then it contains a quadrilateral -/
theorem contains_quadrilateral (G : PlaneGraph) :
  G.m > (1/4 : ℝ) * G.n * (1 + Real.sqrt (4 * G.n - 3)) →
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
    (∃ (e1 e2 e3 e4 : Set ℕ), 
      e1 = {a, b} ∧ e2 = {b, c} ∧ e3 = {c, d} ∧ e4 = {d, a}) :=
by sorry

end NUMINAMATH_CALUDE_contains_quadrilateral_l317_31723


namespace NUMINAMATH_CALUDE_sqrt_twenty_minus_sqrt_five_l317_31757

theorem sqrt_twenty_minus_sqrt_five : Real.sqrt 20 - Real.sqrt 5 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twenty_minus_sqrt_five_l317_31757


namespace NUMINAMATH_CALUDE_sundae_price_l317_31716

/-- Proves that given the specified conditions, the price of each sundae is $1.40 -/
theorem sundae_price : 
  ∀ (ice_cream_bars sundaes : ℕ) 
    (total_price ice_cream_price sundae_price : ℚ),
  ice_cream_bars = 125 →
  sundaes = 125 →
  total_price = 250 →
  ice_cream_price = 0.60 →
  total_price = ice_cream_bars * ice_cream_price + sundaes * sundae_price →
  sundae_price = 1.40 := by
sorry

end NUMINAMATH_CALUDE_sundae_price_l317_31716


namespace NUMINAMATH_CALUDE_g_difference_at_3_and_neg_3_l317_31708

def g (x : ℝ) : ℝ := x^6 + 5*x^2 + 3*x

theorem g_difference_at_3_and_neg_3 : g 3 - g (-3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_at_3_and_neg_3_l317_31708


namespace NUMINAMATH_CALUDE_fraction_meaningful_l317_31727

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l317_31727


namespace NUMINAMATH_CALUDE_peters_erasers_l317_31703

theorem peters_erasers (x : ℕ) : x + 3 = 11 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_peters_erasers_l317_31703


namespace NUMINAMATH_CALUDE_gcd_7800_360_minus_20_l317_31738

theorem gcd_7800_360_minus_20 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7800_360_minus_20_l317_31738


namespace NUMINAMATH_CALUDE_expression_evaluation_l317_31747

theorem expression_evaluation (a : ℝ) (h : a^2 + a = 6) :
  (a^2 - 2*a) / (a^2 - 1) / (a - 1 - (2*a - 1) / (a + 1)) = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l317_31747


namespace NUMINAMATH_CALUDE_partition_characterization_l317_31763

/-- The set V_p for a prime p -/
def V_p (p : ℕ) : Set ℕ :=
  {k | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

/-- A partition of the set {1,2,...,k} into p subsets -/
def IsValidPartition (p k : ℕ) (partition : List (List ℕ)) : Prop :=
  (partition.length = p) ∧
  (partition.join.toFinset = Finset.range k) ∧
  (∀ s ∈ partition, s.sum = (partition.head!).sum)

theorem partition_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ k : ℕ, (∃ partition : List (List ℕ), IsValidPartition p k partition) ↔ k ∈ V_p p :=
sorry

end NUMINAMATH_CALUDE_partition_characterization_l317_31763


namespace NUMINAMATH_CALUDE_original_number_proof_l317_31711

theorem original_number_proof :
  ∃ x : ℝ, (x - x / 3 = x - 48) ∧ (x = 144) := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l317_31711


namespace NUMINAMATH_CALUDE_range_of_a_l317_31785

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := (1 + a)^2 + (1 - a)^2 < 4

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 ≥ 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (¬(p a ∧ q a) ∧ (p a ∨ q a)) → (a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l317_31785


namespace NUMINAMATH_CALUDE_sandwich_count_l317_31707

def num_meat : ℕ := 12
def num_cheese : ℕ := 11
def num_toppings : ℕ := 8

def sandwich_combinations : ℕ := (num_meat.choose 2) * (num_cheese.choose 2) * (num_toppings.choose 2)

theorem sandwich_count : sandwich_combinations = 101640 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l317_31707


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l317_31779

/-- The time when a ball hits the ground given its height equation -/
theorem ball_hitting_ground_time : ∃ t : ℚ, t > 0 ∧ -4.9 * t^2 + 4 * t + 6 = 0 ∧ t = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l317_31779


namespace NUMINAMATH_CALUDE_integral_bounds_l317_31715

theorem integral_bounds : 
  let f : ℝ → ℝ := λ x => 1 / (1 + 3 * Real.sin x ^ 2)
  let a : ℝ := 0
  let b : ℝ := Real.pi / 6
  (2 * Real.pi) / 21 ≤ ∫ x in a..b, f x ∧ ∫ x in a..b, f x ≤ Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_bounds_l317_31715


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l317_31742

theorem smallest_m_for_integral_solutions :
  ∀ m : ℕ+,
  (∃ x : ℤ, 12 * x^2 - m * x + 504 = 0) →
  m ≥ 156 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l317_31742


namespace NUMINAMATH_CALUDE_light_switch_correspondence_l317_31743

/-- Represents a room in the house -/
structure Room (n : ℕ) where
  id : Fin (2^n)

/-- Represents a light switch in the house -/
structure Switch (n : ℕ) where
  id : Fin (2^n)

/-- A function that represents a check of switches -/
def Check (n : ℕ) := Fin (2^n) → Bool

/-- A sequence of checks -/
def CheckSequence (n : ℕ) (m : ℕ) := Fin m → Check n

/-- A bijection between rooms and switches -/
def Correspondence (n : ℕ) := {f : Room n → Switch n // Function.Bijective f}

/-- The main theorem stating that 2n checks are sufficient and 2n-1 checks are not -/
theorem light_switch_correspondence (n : ℕ) :
  (∃ (cs : CheckSequence n (2*n)), ∃ (c : Correspondence n), 
    ∀ (r : Room n), ∃ (s : Switch n), c.val r = s ∧ 
      ∀ (i : Fin (2*n)), cs i (r.id) = cs i (s.id)) ∧
  (∀ (cs : CheckSequence n (2*n - 1)), ¬∃ (c : Correspondence n), 
    ∀ (r : Room n), ∃ (s : Switch n), c.val r = s ∧ 
      ∀ (i : Fin (2*n - 1)), cs i (r.id) = cs i (s.id)) :=
sorry

end NUMINAMATH_CALUDE_light_switch_correspondence_l317_31743


namespace NUMINAMATH_CALUDE_a_exp_a_inequality_l317_31701

theorem a_exp_a_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_a_exp_a_inequality_l317_31701


namespace NUMINAMATH_CALUDE_circle_graph_fractions_l317_31756

/-- Represents the fractions of a circle graph split into three colors -/
structure CircleGraph :=
  (black : ℚ)
  (gray : ℚ)
  (white : ℚ)

/-- The conditions of the circle graph -/
def valid_circle_graph (g : CircleGraph) : Prop :=
  g.black = 2 * g.gray ∧
  g.white = g.gray / 2 ∧
  g.black + g.gray + g.white = 1

/-- The theorem to prove -/
theorem circle_graph_fractions :
  ∃ (g : CircleGraph), valid_circle_graph g ∧
    g.black = 4/7 ∧ g.gray = 2/7 ∧ g.white = 1/7 :=
sorry

end NUMINAMATH_CALUDE_circle_graph_fractions_l317_31756


namespace NUMINAMATH_CALUDE_trefoils_per_case_l317_31762

theorem trefoils_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) :
  total_boxes = 24 →
  total_cases = 3 →
  total_boxes = boxes_per_case * total_cases →
  boxes_per_case = 8 := by
  sorry

end NUMINAMATH_CALUDE_trefoils_per_case_l317_31762


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l317_31774

/-- A geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_fifth_term
  (seq : GeometricSequence)
  (h1 : seq.a 1 * seq.a 3 = 4)
  (h2 : seq.a 7 * seq.a 9 = 25) :
  seq.a 5 = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l317_31774


namespace NUMINAMATH_CALUDE_egg_difference_solution_l317_31709

/-- Represents the problem of calculating the difference between eggs in perfect condition
    in undropped trays and cracked eggs in dropped trays. -/
def egg_difference_problem (total_eggs : ℕ) (num_trays : ℕ) (dropped_trays : ℕ)
  (first_tray_capacity : ℕ) (second_tray_capacity : ℕ) (third_tray_capacity : ℕ)
  (first_tray_cracked : ℕ) (second_tray_cracked : ℕ) (third_tray_cracked : ℕ) : Prop :=
  let total_dropped_capacity := first_tray_capacity + second_tray_capacity + third_tray_capacity
  let undropped_eggs := total_eggs - total_dropped_capacity
  let total_cracked := first_tray_cracked + second_tray_cracked + third_tray_cracked
  undropped_eggs - total_cracked = 8

/-- The main theorem stating the solution to the egg problem. -/
theorem egg_difference_solution :
  egg_difference_problem 60 5 3 15 12 10 7 5 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_solution_l317_31709


namespace NUMINAMATH_CALUDE_symmetric_function_periodic_l317_31780

/-- A function f: ℝ → ℝ is symmetric with respect to the point (a, y₀) if for all x, f(a + x) - y₀ = y₀ - f(a - x) -/
def SymmetricPoint (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f (a + x) - y₀ = y₀ - f (a - x)

/-- A function f: ℝ → ℝ is symmetric with respect to the line x = b if for all x, f(b + x) = f(b - x) -/
def SymmetricLine (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (b + x) = f (b - x)

/-- The main theorem: if f is symmetric with respect to a point (a, y₀) and a line x = b where b > a,
    then f is periodic with period 4(b-a) -/
theorem symmetric_function_periodic (f : ℝ → ℝ) (a b y₀ : ℝ) 
    (h_point : SymmetricPoint f a y₀) (h_line : SymmetricLine f b) (h_order : b > a) :
    ∀ x, f (x + 4*(b - a)) = f x := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_periodic_l317_31780


namespace NUMINAMATH_CALUDE_same_suit_bottom_probability_l317_31778

def deck_size : Nat := 6
def black_cards : Nat := 3
def red_cards : Nat := 3

theorem same_suit_bottom_probability :
  let total_arrangements := Nat.factorial deck_size
  let favorable_outcomes := 2 * (Nat.factorial black_cards * Nat.factorial red_cards)
  (favorable_outcomes : ℚ) / total_arrangements = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_same_suit_bottom_probability_l317_31778


namespace NUMINAMATH_CALUDE_S_13_equals_3510_l317_31788

/-- The sequence S defined for natural numbers -/
def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

/-- Theorem stating that S(13) equals 3510 -/
theorem S_13_equals_3510 : S 13 = 3510 := by
  sorry

end NUMINAMATH_CALUDE_S_13_equals_3510_l317_31788


namespace NUMINAMATH_CALUDE_variance_scaled_sample_l317_31782

variable (s : ℝ) (x : Fin 5 → ℝ)

def variance (x : Fin 5 → ℝ) : ℝ := sorry

def scaled_sample (x : Fin 5 → ℝ) : Fin 5 → ℝ := fun i => 2 * x i

theorem variance_scaled_sample (h : variance x = 3) : 
  variance (scaled_sample x) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_scaled_sample_l317_31782


namespace NUMINAMATH_CALUDE_leftover_value_is_9_65_l317_31790

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of quarters in a full roll -/
def quarters_per_roll : ℕ := 42

/-- The number of dimes in a full roll -/
def dimes_per_roll : ℕ := 48

/-- Gary's quarters -/
def gary_quarters : ℕ := 127

/-- Gary's dimes -/
def gary_dimes : ℕ := 212

/-- Kim's quarters -/
def kim_quarters : ℕ := 158

/-- Kim's dimes -/
def kim_dimes : ℕ := 297

/-- Theorem: The value of leftover quarters and dimes is $9.65 -/
theorem leftover_value_is_9_65 :
  let total_quarters := gary_quarters + kim_quarters
  let total_dimes := gary_dimes + kim_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 9.65 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_9_65_l317_31790


namespace NUMINAMATH_CALUDE_carrie_harvest_l317_31733

/-- Represents the number of carrots Carrie harvested -/
def num_carrots : ℕ := 350

/-- Represents the number of tomatoes Carrie harvested -/
def num_tomatoes : ℕ := 200

/-- Represents the price of a tomato in cents -/
def tomato_price : ℕ := 100

/-- Represents the price of a carrot in cents -/
def carrot_price : ℕ := 150

/-- Represents the total revenue in cents -/
def total_revenue : ℕ := 72500

theorem carrie_harvest :
  num_tomatoes * tomato_price + num_carrots * carrot_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_carrie_harvest_l317_31733


namespace NUMINAMATH_CALUDE_joseph_driving_time_l317_31704

theorem joseph_driving_time :
  let joseph_speed : ℝ := 50
  let kyle_speed : ℝ := 62
  let kyle_time : ℝ := 2
  let distance_difference : ℝ := 1
  let joseph_distance : ℝ := kyle_speed * kyle_time + distance_difference
  joseph_distance / joseph_speed = 2.5 := by sorry

end NUMINAMATH_CALUDE_joseph_driving_time_l317_31704


namespace NUMINAMATH_CALUDE_bricklayer_problem_l317_31776

theorem bricklayer_problem (time1 time2 reduction_rate joint_time : ℝ) 
  (h1 : time1 = 8)
  (h2 : time2 = 12)
  (h3 : reduction_rate = 12)
  (h4 : joint_time = 6) :
  ∃ (total_bricks : ℝ),
    total_bricks = 288 ∧
    joint_time * ((total_bricks / time1) + (total_bricks / time2) - reduction_rate) = total_bricks :=
  sorry

end NUMINAMATH_CALUDE_bricklayer_problem_l317_31776


namespace NUMINAMATH_CALUDE_machinery_expenditure_l317_31769

theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (machinery : ℝ) :
  total = 93750 →
  raw_materials = 35000 →
  machinery + raw_materials + (0.2 * total) = total →
  machinery = 40000 := by
sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l317_31769


namespace NUMINAMATH_CALUDE_special_polyhedron_ratio_l317_31767

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : Nat
  x : ℝ
  y : ℝ
  isIsosceles : Bool
  vertexDegrees : Finset Nat
  dihedralAnglesEqual : Bool

/-- The conditions for our special polyhedron -/
def specialPolyhedronConditions (p : SpecialPolyhedron) : Prop :=
  p.faces = 12 ∧
  p.isIsosceles = true ∧
  p.vertexDegrees = {3, 6} ∧
  p.dihedralAnglesEqual = true

/-- The theorem stating the ratio of x to y for our special polyhedron -/
theorem special_polyhedron_ratio (p : SpecialPolyhedron) 
  (h : specialPolyhedronConditions p) : p.x / p.y = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_special_polyhedron_ratio_l317_31767


namespace NUMINAMATH_CALUDE_fixed_points_sum_zero_l317_31710

open Real

/-- The sum of fixed points of natural logarithm and exponential functions is zero -/
theorem fixed_points_sum_zero :
  ∃ t₁ t₂ : ℝ, 
    (exp t₁ = -t₁) ∧ 
    (log t₂ = -t₂) ∧ 
    (t₁ + t₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_sum_zero_l317_31710


namespace NUMINAMATH_CALUDE_jellybean_probability_l317_31787

/-- The probability of drawing 3 blue jellybeans in succession without replacement from a bag containing 10 red and 10 blue jellybeans -/
theorem jellybean_probability : 
  let total_jellybeans : ℕ := 10 + 10
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  (blue_jellybeans : ℚ) / total_jellybeans *
  ((blue_jellybeans - 1) : ℚ) / (total_jellybeans - 1) *
  ((blue_jellybeans - 2) : ℚ) / (total_jellybeans - 2) = 2 / 19 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_probability_l317_31787


namespace NUMINAMATH_CALUDE_perpendicular_sum_implies_zero_l317_31750

/-- Given vectors a and b in ℝ², if a is perpendicular to (a + b), then the second component of b is 0. -/
theorem perpendicular_sum_implies_zero (a b : ℝ × ℝ) (h : a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 2) :
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) → b.2 = 0 := by
  sorry

#check perpendicular_sum_implies_zero

end NUMINAMATH_CALUDE_perpendicular_sum_implies_zero_l317_31750


namespace NUMINAMATH_CALUDE_parallelogram_area_from_side_and_diagonals_l317_31713

/-- The area of a parallelogram given one side and two diagonals -/
theorem parallelogram_area_from_side_and_diagonals
  (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ)
  (h_side : side = 51)
  (h_diag1 : diagonal1 = 40)
  (h_diag2 : diagonal2 = 74) :
  let s := (side + diagonal1 / 2 + diagonal2 / 2) / 2
  4 * Real.sqrt (s * (s - side) * (s - diagonal1 / 2) * (s - diagonal2 / 2)) = 1224 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_from_side_and_diagonals_l317_31713


namespace NUMINAMATH_CALUDE_characterize_M_l317_31773

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x | (m-1)*x - 1 = 0}

def M : Set ℝ := {m | A ∩ B m = B m}

theorem characterize_M : M = {3/2, 4/3, 1} := by sorry

end NUMINAMATH_CALUDE_characterize_M_l317_31773


namespace NUMINAMATH_CALUDE_song_difference_main_result_l317_31725

/-- Represents the number of songs in various categories for a composer --/
structure SongCounts where
  total : ℕ
  top10 : ℕ
  top100 : ℕ
  unreleased : ℕ

/-- The difference between top100 and top10 songs is 10 --/
theorem song_difference (s : SongCounts) : s.top100 - s.top10 = 10 :=
  by
  have h1 : s.total = 80 := by sorry
  have h2 : s.top10 = 25 := by sorry
  have h3 : s.unreleased = s.top10 - 5 := by sorry
  have h4 : s.total = s.top10 + s.top100 + s.unreleased := by sorry
  sorry

/-- Main theorem stating the result --/
theorem main_result : ∃ s : SongCounts, s.top100 - s.top10 = 10 :=
  by
  sorry

end NUMINAMATH_CALUDE_song_difference_main_result_l317_31725


namespace NUMINAMATH_CALUDE_collectiveEarnings_l317_31735

-- Define the workers and their properties
structure Worker where
  name : String
  normalHours : Float
  hourlyRate : Float
  overtimeMultiplier : Float
  actualHours : Float

-- Calculate earnings for a worker
def calculateEarnings (w : Worker) : Float :=
  let regularPay := min w.normalHours w.actualHours * w.hourlyRate
  let overtimeHours := max (w.actualHours - w.normalHours) 0
  let overtimePay := overtimeHours * w.hourlyRate * w.overtimeMultiplier
  regularPay + overtimePay

-- Define Lloyd and Casey
def lloyd : Worker := {
  name := "Lloyd"
  normalHours := 7.5
  hourlyRate := 4.50
  overtimeMultiplier := 2.0
  actualHours := 10.5
}

def casey : Worker := {
  name := "Casey"
  normalHours := 8.0
  hourlyRate := 5.00
  overtimeMultiplier := 1.5
  actualHours := 9.5
}

-- Theorem: Lloyd and Casey's collective earnings equal $112.00
theorem collectiveEarnings : calculateEarnings lloyd + calculateEarnings casey = 112.00 := by
  sorry

end NUMINAMATH_CALUDE_collectiveEarnings_l317_31735


namespace NUMINAMATH_CALUDE_identical_views_solids_l317_31772

-- Define the set of all possible solids
inductive Solid
  | Sphere
  | TriangularPyramid
  | Cube
  | Cylinder

-- Define a predicate for solids with identical views
def has_identical_views (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => true
  | Solid.TriangularPyramid => true
  | Solid.Cube => true
  | Solid.Cylinder => false

-- Theorem stating that the set of solids with identical views
-- is equal to the set containing Sphere, Triangular Pyramid, and Cube
theorem identical_views_solids :
  {s : Solid | has_identical_views s} =
  {Solid.Sphere, Solid.TriangularPyramid, Solid.Cube} :=
by sorry

end NUMINAMATH_CALUDE_identical_views_solids_l317_31772


namespace NUMINAMATH_CALUDE_base_8_to_10_reversal_exists_l317_31786

theorem base_8_to_10_reversal_exists : ∃ (a b c : Nat), 
  a < 8 ∧ b < 8 ∧ c < 8 ∧
  (512 * a + 64 * b + 8 * c + 6 : Nat) = 
  (1000 * 6 + 100 * c + 10 * b + a : Nat) :=
sorry

end NUMINAMATH_CALUDE_base_8_to_10_reversal_exists_l317_31786


namespace NUMINAMATH_CALUDE_banana_consumption_l317_31792

theorem banana_consumption (n : ℕ) (a : ℝ) (h1 : n = 7) (h2 : a > 0) : 
  (a * (2^(n-1))) = 128 ∧ 
  (a * (2^n - 1)) / (2 - 1) = 254 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_consumption_l317_31792


namespace NUMINAMATH_CALUDE_georgia_green_buttons_l317_31705

/-- The number of green buttons Georgia has -/
def green_buttons : ℕ := sorry

/-- The number of yellow buttons Georgia has -/
def yellow_buttons : ℕ := 4

/-- The number of black buttons Georgia has -/
def black_buttons : ℕ := 2

/-- The number of buttons Georgia gave away -/
def buttons_given_away : ℕ := 4

/-- The number of buttons Georgia has left -/
def buttons_left : ℕ := 5

theorem georgia_green_buttons :
  yellow_buttons + black_buttons + green_buttons = buttons_given_away + buttons_left :=
sorry

end NUMINAMATH_CALUDE_georgia_green_buttons_l317_31705


namespace NUMINAMATH_CALUDE_unique_prime_sum_of_squares_and_divisibility_l317_31775

theorem unique_prime_sum_of_squares_and_divisibility :
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (m n : ℕ+), 
      p = m^2 + n^2 ∧ 
      (m^3 + n^3 + 8*m*n) % p = 0) ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_of_squares_and_divisibility_l317_31775


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l317_31781

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The median of a sequence is the middle value when the sequence is ordered. -/
def hasMedian (a : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ n : ℕ, a n = m ∧ (∀ i j : ℕ, i ≤ n ∧ n ≤ j → a i ≤ m ∧ m ≤ a j)

theorem arithmetic_sequence_first_term
  (a : ℕ → ℕ)
  (h1 : isArithmeticSequence a)
  (h2 : hasMedian a 1010)
  (h3 : ∃ n : ℕ, a n = 2015 ∧ ∀ m : ℕ, m > n → a m > 2015) :
  a 0 = 5 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l317_31781


namespace NUMINAMATH_CALUDE_shadow_relation_sets_l317_31794

def is_shadow_relation (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (1 / x) ∈ A

def set_A : Set ℝ := {-1, 1}
def set_B : Set ℝ := {1/2, 2}
def set_C : Set ℝ := {x : ℝ | x^2 > 1}
def set_D : Set ℝ := {x : ℝ | x > 0}

theorem shadow_relation_sets :
  is_shadow_relation set_A ∧
  is_shadow_relation set_B ∧
  is_shadow_relation set_D ∧
  ¬is_shadow_relation set_C :=
sorry

end NUMINAMATH_CALUDE_shadow_relation_sets_l317_31794


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l317_31700

theorem complex_number_in_first_quadrant :
  let z : ℂ := (2 + Complex.I) / 3
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l317_31700


namespace NUMINAMATH_CALUDE_system_solution_unique_l317_31770

theorem system_solution_unique :
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x + 2 * y = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l317_31770


namespace NUMINAMATH_CALUDE_train_length_l317_31736

/-- Train crossing a bridge problem -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (man_speed : ℝ) :
  train_speed = 80 →
  bridge_length = 1 →
  man_speed = 5 →
  (bridge_length / train_speed) * man_speed = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l317_31736


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l317_31777

-- Define the circle C₁
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 8*y + 19 = 0

-- Define the line l
def line_l (x y a : ℝ) : Prop :=
  x + 2*y - a = 0

-- Theorem statement
theorem circle_symmetry_line (a : ℝ) :
  (∃ (x y : ℝ), circle_C1 x y ∧ line_l x y a) →
  (∀ (x y : ℝ), circle_C1 x y → 
    ∃ (x' y' : ℝ), circle_C1 x' y' ∧ 
      ((x + x')/2, (y + y')/2) ∈ {(x, y) | line_l x y a}) →
  a = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l317_31777


namespace NUMINAMATH_CALUDE_equal_coin_count_theorem_l317_31765

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | HalfDollar
  | OneDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .HalfDollar => 50
  | .OneDollar => 100

/-- The total value of coins in cents --/
def totalValue : ℕ := 332

/-- The number of different coin types --/
def numCoinTypes : ℕ := 5

theorem equal_coin_count_theorem :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n * (coinValue CoinType.Penny + coinValue CoinType.Nickel + 
         coinValue CoinType.Dime + coinValue CoinType.HalfDollar + 
         coinValue CoinType.OneDollar) = totalValue ∧
    n * numCoinTypes = 10 := by
  sorry

end NUMINAMATH_CALUDE_equal_coin_count_theorem_l317_31765


namespace NUMINAMATH_CALUDE_vector_problem_l317_31722

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- Two vectors are in opposite directions if their dot product is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

theorem vector_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (9, x)
  collinear a b → opposite_directions a b → x = -3 := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l317_31722


namespace NUMINAMATH_CALUDE_correct_equation_is_fourth_l317_31761

theorem correct_equation_is_fourth : 
  ∃ (a b : ℝ), 
    (2*a + 3*b ≠ 5*a*b) ∧ 
    ((3*a^3)^2 ≠ 6*a^6) ∧ 
    (a^6 / a^2 ≠ a^3) ∧ 
    (a^2 * a^3 = a^5) := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_is_fourth_l317_31761


namespace NUMINAMATH_CALUDE_perimeter_difference_l317_31706

/-- The perimeter of a rectangle --/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- The perimeter of the cross-shaped figure --/
def cross_perimeter (center_side : ℕ) : ℕ :=
  4 * center_side

/-- The positive difference between two natural numbers --/
def positive_difference (a b : ℕ) : ℕ :=
  max a b - min a b

theorem perimeter_difference :
  positive_difference (rectangle_perimeter 3 2) (cross_perimeter 3) = 2 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_difference_l317_31706


namespace NUMINAMATH_CALUDE_find_x_l317_31764

-- Define the relationship between y, x, and a
def relationship (y x a k : ℝ) : Prop := y^4 * Real.sqrt x = k / a

-- Theorem statement
theorem find_x (k : ℝ) : 
  (∃ y x a, relationship y x a k ∧ y = 1 ∧ x = 16 ∧ a = 2) →
  (∀ y x a, relationship y x a k → y = 2 → a = 4 → x = 1/64) :=
by sorry

end NUMINAMATH_CALUDE_find_x_l317_31764


namespace NUMINAMATH_CALUDE_sqrt_product_minus_one_equals_546_l317_31791

theorem sqrt_product_minus_one_equals_546 : 
  Real.sqrt ((25 : ℝ) * 24 * 23 * 22 - 1) = 546 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_minus_one_equals_546_l317_31791


namespace NUMINAMATH_CALUDE_log_equation_solution_l317_31702

-- Define the logarithm function for base 16
noncomputable def log16 (x : ℝ) : ℝ := Real.log x / Real.log 16

-- State the theorem
theorem log_equation_solution :
  ∀ y : ℝ, log16 (3 * y - 4) = 2 → y = 260 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l317_31702


namespace NUMINAMATH_CALUDE_acyclic_orientations_not_div_three_l317_31793

/-- A bipartite graph representing airline connections between Russian and Ukrainian cities -/
structure AirlineGraph where
  vertices : Type
  edges : Set (vertices × vertices)
  is_bipartite : ∃ (A B : Set vertices), A ∪ B = univ ∧ A ∩ B = ∅ ∧
    ∀ e ∈ edges, (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A)

/-- The number of acyclic orientations of a graph -/
def num_acyclic_orientations (G : AirlineGraph) : ℕ :=
  sorry

/-- Theorem: The number of acyclic orientations of the airline graph is not divisible by 3 -/
theorem acyclic_orientations_not_div_three (G : AirlineGraph) :
  ¬(3 ∣ num_acyclic_orientations G) :=
sorry

end NUMINAMATH_CALUDE_acyclic_orientations_not_div_three_l317_31793


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_3_8_9_l317_31798

theorem smallest_five_digit_divisible_by_2_3_8_9 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 → 
    (m % 2 = 0 ∧ m % 3 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0) → 
    n ≤ m) ∧  -- smallest such number
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧  -- divisible by 2, 3, 8, and 9
  n = 10008  -- the specific value
:= by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_2_3_8_9_l317_31798


namespace NUMINAMATH_CALUDE_problem_solution_l317_31720

def proposition_p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def proposition_q (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1) 1, x^2 - x - 1 + m ≤ 0

theorem problem_solution (m : ℝ) :
  (proposition_p m ↔ (1 ≤ m ∧ m ≤ 2)) ∧
  ((proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m) ↔
    (m < 1 ∨ (5/4 < m ∧ m ≤ 2))) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l317_31720


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l317_31752

def original_price : ℝ := 103.5
def sale_price : ℝ := 78.2
def price_increase_percentage : ℝ := 25
def price_difference : ℝ := 5.75

theorem discount_percentage_proof :
  ∃ (discount_percentage : ℝ),
    sale_price = original_price - (discount_percentage / 100) * original_price ∧
    original_price - (sale_price + price_increase_percentage / 100 * sale_price) = price_difference ∧
    (discount_percentage ≥ 24.43 ∧ discount_percentage ≤ 24.45) := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l317_31752


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l317_31796

theorem linear_equation_exponent (a : ℝ) : 
  (∀ x, ∃ k m : ℝ, 3 * x^(2*a - 1) - 4 = k * x + m) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l317_31796


namespace NUMINAMATH_CALUDE_gadget_production_proof_l317_31753

/-- Represents the production rate of gadgets per worker per hour -/
def gadget_rate (workers : ℕ) (hours : ℕ) (gadgets : ℕ) : ℚ :=
  (gadgets : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Calculates the number of gadgets produced given workers, hours, and rate -/
def gadgets_produced (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours : ℚ) * rate

theorem gadget_production_proof :
  let rate1 := gadget_rate 150 3 600
  let rate2 := gadget_rate 100 4 800
  let final_rate := max rate1 rate2
  gadgets_produced 75 5 final_rate = 750 := by
  sorry

end NUMINAMATH_CALUDE_gadget_production_proof_l317_31753


namespace NUMINAMATH_CALUDE_triangle_shape_determination_l317_31730

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The shape of a triangle is determined by its side lengths and angles -/
def triangle_shape (t : Triangle) : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := sorry

/-- Two angles and the side between them -/
def sas_data (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Ratio of two angle bisectors -/
def angle_bisector_ratio (t : Triangle) : ℝ := sorry

/-- Ratio of circumradius to inradius -/
def radii_ratio (t : Triangle) : ℝ := sorry

/-- Ratio of area to perimeter -/
def area_perimeter_ratio (t : Triangle) : ℝ := sorry

/-- A function is shape-determining if it uniquely determines the triangle's shape -/
def is_shape_determining (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → triangle_shape t1 = triangle_shape t2

theorem triangle_shape_determination :
  is_shape_determining sas_data ∧
  ¬ is_shape_determining angle_bisector_ratio ∧
  is_shape_determining radii_ratio ∧
  is_shape_determining area_perimeter_ratio :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_determination_l317_31730


namespace NUMINAMATH_CALUDE_every_tomcat_has_thinner_queen_l317_31726

/-- Represents a cat in the exhibition -/
inductive Cat
| Tomcat : Cat
| Queen : Cat

/-- The total number of cats in the row -/
def total_cats : Nat := 29

/-- The number of tomcats in the row -/
def num_tomcats : Nat := 10

/-- The number of queens in the row -/
def num_queens : Nat := 19

/-- Represents the row of cats at the exhibition -/
def cat_row : Fin total_cats → Cat := sorry

/-- Predicate to check if a cat is fatter than another -/
def is_fatter (c1 c2 : Cat) : Prop := sorry

/-- Two cats are adjacent if their positions differ by 1 -/
def adjacent (i j : Fin total_cats) : Prop :=
  (i.val + 1 = j.val) ∨ (j.val + 1 = i.val)

/-- Each queen has a fatter tomcat next to her -/
axiom queen_has_fatter_tomcat :
  ∀ (i : Fin total_cats), cat_row i = Cat.Queen →
    ∃ (j : Fin total_cats), adjacent i j ∧ cat_row j = Cat.Tomcat ∧ is_fatter (cat_row j) (cat_row i)

/-- The main theorem to be proved -/
theorem every_tomcat_has_thinner_queen :
  ∀ (i : Fin total_cats), cat_row i = Cat.Tomcat →
    ∃ (j : Fin total_cats), adjacent i j ∧ cat_row j = Cat.Queen ∧ is_fatter (cat_row i) (cat_row j) := by
  sorry

end NUMINAMATH_CALUDE_every_tomcat_has_thinner_queen_l317_31726


namespace NUMINAMATH_CALUDE_breakfast_cost_is_correct_l317_31714

/-- Calculates the total cost of breakfast for Francis, Kiera, and David --/
def total_breakfast_cost (muffin_price fruit_cup_price coffee_price : ℚ)
  (discount_rate : ℚ) (voucher : ℚ) : ℚ :=
  let francis_cost := 2 * muffin_price + 2 * fruit_cup_price + coffee_price -
    discount_rate * (2 * muffin_price + fruit_cup_price)
  let kiera_cost := 2 * muffin_price + fruit_cup_price + coffee_price -
    discount_rate * (2 * muffin_price + fruit_cup_price)
  let david_cost := 3 * muffin_price + fruit_cup_price + coffee_price - voucher
  francis_cost + kiera_cost + david_cost

/-- Theorem stating that the total breakfast cost is $27.10 --/
theorem breakfast_cost_is_correct :
  total_breakfast_cost 2 3 1.5 0.1 2 = 27.1 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_correct_l317_31714


namespace NUMINAMATH_CALUDE_non_green_car_probability_l317_31724

/-- The probability of selecting a non-green car from a set of 60 cars with 30 green cars is 1/2 -/
theorem non_green_car_probability (total_cars : ℕ) (green_cars : ℕ) 
  (h1 : total_cars = 60) 
  (h2 : green_cars = 30) : 
  (total_cars - green_cars : ℚ) / total_cars = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_non_green_car_probability_l317_31724


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l317_31740

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 + (3 - x)^3
  ∃ (x₁ x₂ : ℝ), (x₁ = 1.5 + Real.sqrt 5 / 2) ∧ 
                 (x₂ = 1.5 - Real.sqrt 5 / 2) ∧ 
                 (f x₁ = 18) ∧ 
                 (f x₂ = 18) ∧ 
                 (∀ x : ℝ, f x = 18 → (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l317_31740


namespace NUMINAMATH_CALUDE_scarves_per_box_l317_31739

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_items : ℕ) : 
  num_boxes = 7 → 
  mittens_per_box = 4 → 
  total_items = 49 → 
  (total_items - num_boxes * mittens_per_box) / num_boxes = 3 := by
sorry

end NUMINAMATH_CALUDE_scarves_per_box_l317_31739


namespace NUMINAMATH_CALUDE_solution_is_five_l317_31754

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := 
  x > 3 ∧ log10 (x - 3) + log10 x = 1

-- State the theorem
theorem solution_is_five : 
  ∃ (x : ℝ), equation x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_is_five_l317_31754


namespace NUMINAMATH_CALUDE_sequence_sum_property_l317_31797

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = n^2 + 2*n + 5) →
  (∀ n : ℕ, S (n+1) - S n = a (n+1)) →
  a 2 + a 3 + a 4 + a 4 + a 5 = 41 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l317_31797


namespace NUMINAMATH_CALUDE_system_solution_l317_31717

theorem system_solution : 
  ∃ (x y : ℚ), (3 * x - 4 * y = -7) ∧ (4 * x - 3 * y = 5) ∧ (x = 41/7) ∧ (y = 43/7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l317_31717


namespace NUMINAMATH_CALUDE_cone_volume_l317_31795

/-- The volume of a cone with height h, whose lateral surface unfolds into a sector with a central angle of 120°, is πh³/24. -/
theorem cone_volume (h : ℝ) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (π * h^3) / 24 ∧ 
  V = (1/3) * π * (h^2 / 8) * h ∧
  ∃ (R : ℝ), R > 0 ∧ R^2 = h^2 / 8 ∧
  ∃ (l : ℝ), l > 0 ∧ l = 3 * R ∧
  2 * π * R = (2 * π * l) / 3 :=
sorry

end NUMINAMATH_CALUDE_cone_volume_l317_31795


namespace NUMINAMATH_CALUDE_facebook_bonus_calculation_l317_31744

/-- Calculates the bonus amount for each female mother employee at Facebook --/
theorem facebook_bonus_calculation (total_employees : ℕ) (non_mother_females : ℕ) 
  (annual_earnings : ℚ) (bonus_percentage : ℚ) :
  total_employees = 3300 →
  non_mother_females = 1200 →
  annual_earnings = 5000000 →
  bonus_percentage = 1/4 →
  ∃ (bonus_per_employee : ℚ),
    bonus_per_employee = 1250 ∧
    bonus_per_employee = (annual_earnings * bonus_percentage) / 
      (total_employees - (total_employees / 3) - non_mother_females) :=
by
  sorry


end NUMINAMATH_CALUDE_facebook_bonus_calculation_l317_31744


namespace NUMINAMATH_CALUDE_probability_product_multiple_of_four_l317_31755

def range_start : ℕ := 5
def range_end : ℕ := 25

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def count_in_range : ℕ := range_end - range_start + 1

def count_multiples_of_four : ℕ := (range_end / 4) - ((range_start - 1) / 4)

def total_combinations : ℕ := count_in_range * (count_in_range - 1) / 2

def favorable_combinations : ℕ := count_multiples_of_four * (count_multiples_of_four - 1) / 2

theorem probability_product_multiple_of_four :
  (favorable_combinations : ℚ) / total_combinations = 1 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_product_multiple_of_four_l317_31755


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l317_31758

/-- Given that M(4,7) is the midpoint of line segment AB and A(5,3) is one endpoint,
    the product of the coordinates of point B is 33. -/
theorem midpoint_coordinate_product : 
  let A : ℝ × ℝ := (5, 3)
  let M : ℝ × ℝ := (4, 7)
  ∃ B : ℝ × ℝ, 
    (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → 
    B.1 * B.2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l317_31758


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l317_31734

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 4

/-- The logarithm base 1/2 -/
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem quadratic_function_properties :
  (∀ x, f x ≥ -4) ∧  -- Minimum value is -4
  (f 2 = -4) ∧  -- Minimum occurs at x = 2
  (f 0 = 0) ∧  -- Passes through origin
  (∀ x, x ∈ Set.Icc (1/8 : ℝ) 2 → f (log_half x) ≥ -4) ∧  -- Minimum in the interval
  (∃ x, x ∈ Set.Icc (1/8 : ℝ) 2 ∧ f (log_half x) = -4) ∧  -- Minimum is attained
  (∀ x, x ∈ Set.Icc (1/8 : ℝ) 2 → f (log_half x) ≤ 5) ∧  -- Maximum in the interval
  (∃ x, x ∈ Set.Icc (1/8 : ℝ) 2 ∧ f (log_half x) = 5)  -- Maximum is attained
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l317_31734


namespace NUMINAMATH_CALUDE_max_value_of_g_l317_31749

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 * x - 2 * x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (x_max : ℝ), x_max ∈ Set.Icc (-2) 2 ∧
  (∀ x ∈ Set.Icc (-2) 2, g x ≤ g x_max) ∧
  g x_max = 6 ∧ x_max = -2 := by
  sorry


end NUMINAMATH_CALUDE_max_value_of_g_l317_31749


namespace NUMINAMATH_CALUDE_two_plus_insertion_theorem_l317_31732

/-- Represents a way to split a number into three parts by inserting two plus signs -/
structure ThreePartSplit (n : ℕ) :=
  (first second third : ℕ)
  (split_valid : n = first * 100000 + second * 100 + third)
  (no_rearrange : first < 100 ∧ second < 1000 ∧ third < 100)

/-- The problem statement -/
theorem two_plus_insertion_theorem :
  ∃ (split : ThreePartSplit 8789924),
    split.first + split.second + split.third = 1010 := by
  sorry

end NUMINAMATH_CALUDE_two_plus_insertion_theorem_l317_31732


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l317_31768

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def drawnWinnerBallCount : ℕ := 6

theorem lottery_winning_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / (winnerBallCount.choose drawnWinnerBallCount) = 1 / 476721000 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l317_31768


namespace NUMINAMATH_CALUDE_range_of_sum_l317_31721

theorem range_of_sum (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) :
  0 ≤ a + b ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l317_31721


namespace NUMINAMATH_CALUDE_valid_pairs_l317_31728

def is_valid_pair (A B : Nat) : Prop :=
  A ≠ B ∧
  A ≥ 10 ∧ A ≤ 99 ∧
  B ≥ 10 ∧ B ≤ 99 ∧
  A % 10 = B % 10 ∧
  A / 9 = B % 9 ∧
  B / 9 = A % 9

theorem valid_pairs : 
  (∀ A B : Nat, is_valid_pair A B → 
    ((A = 85 ∧ B = 75) ∨ (A = 25 ∧ B = 65) ∨ (A = 15 ∧ B = 55))) ∧
  (is_valid_pair 85 75 ∧ is_valid_pair 25 65 ∧ is_valid_pair 15 55) := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_l317_31728


namespace NUMINAMATH_CALUDE_inequality_proof_l317_31759

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h : abs (Real.sqrt (a * b) - Real.sqrt (c * d)) ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l317_31759
