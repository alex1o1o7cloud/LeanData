import Mathlib

namespace cos_A_minus_B_l4010_401095

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -3/8 := by
sorry

end cos_A_minus_B_l4010_401095


namespace merry_go_round_area_l4010_401008

theorem merry_go_round_area (diameter : Real) (h : diameter = 2) :
  let radius : Real := diameter / 2
  let area : Real := π * radius ^ 2
  area = π := by
  sorry

end merry_go_round_area_l4010_401008


namespace dusty_change_l4010_401028

def single_layer_cost : ℝ := 4
def single_layer_tax_rate : ℝ := 0.05
def double_layer_cost : ℝ := 7
def double_layer_tax_rate : ℝ := 0.10
def fruit_tart_cost : ℝ := 5
def fruit_tart_tax_rate : ℝ := 0.08

def single_layer_quantity : ℕ := 7
def double_layer_quantity : ℕ := 5
def fruit_tart_quantity : ℕ := 3

def payment_amount : ℝ := 200

theorem dusty_change :
  let single_layer_total := single_layer_quantity * (single_layer_cost * (1 + single_layer_tax_rate))
  let double_layer_total := double_layer_quantity * (double_layer_cost * (1 + double_layer_tax_rate))
  let fruit_tart_total := fruit_tart_quantity * (fruit_tart_cost * (1 + fruit_tart_tax_rate))
  let total_cost := single_layer_total + double_layer_total + fruit_tart_total
  payment_amount - total_cost = 115.90 := by
  sorry

end dusty_change_l4010_401028


namespace algebraic_identities_l4010_401094

theorem algebraic_identities (a b c : ℝ) : 
  (a^4 * (a^2)^3 = a^10) ∧ 
  (2*a^3*b^2*c / ((1/3)*a^2*b) = 6*a*b*c) ∧ 
  (6*a*((1/3)*a*b - b) - (2*a*b + b)*(a - 1) = -5*a*b + b) ∧ 
  ((a - 2)^2 - (3*a + 2*b)*(3*a - 2*b) = -8*a^2 - 4*a + 4 + 4*b^2) := by
  sorry

end algebraic_identities_l4010_401094


namespace colored_tape_length_l4010_401056

theorem colored_tape_length : 
  ∀ (original_length : ℝ),
  (1 / 5 : ℝ) * original_length + -- Used for art
  (3 / 4 : ℝ) * (4 / 5 : ℝ) * original_length + -- Given away
  1.5 = original_length → -- Remaining length
  original_length = 7.5 := by
sorry

end colored_tape_length_l4010_401056


namespace two_vectors_basis_iff_linearly_independent_not_any_two_vectors_form_basis_l4010_401098

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

/-- Two vectors form a basis for a 2-dimensional vector space if and only if they are linearly independent. -/
theorem two_vectors_basis_iff_linearly_independent (v w : V) :
  Submodule.span ℝ {v, w} = ⊤ ↔ LinearIndependent ℝ ![v, w] :=
sorry

/-- It is not true that any two vectors in a 2-dimensional vector space form a basis. -/
theorem not_any_two_vectors_form_basis :
  ¬ ∀ (v w : V), Submodule.span ℝ {v, w} = ⊤ :=
sorry

end two_vectors_basis_iff_linearly_independent_not_any_two_vectors_form_basis_l4010_401098


namespace park_walk_time_l4010_401037

/-- Represents the time in minutes for various walks in the park. -/
structure ParkWalks where
  office_to_hidden_lake : ℕ
  hidden_lake_to_office : ℕ
  total_time : ℕ

/-- Calculates the time from Park Office to Lake Park restaurant. -/
def time_to_restaurant (w : ParkWalks) : ℕ :=
  w.total_time - (w.office_to_hidden_lake + w.hidden_lake_to_office)

/-- Theorem stating the time from Park Office to Lake Park restaurant is 10 minutes. -/
theorem park_walk_time (w : ParkWalks) 
  (h1 : w.office_to_hidden_lake = 15)
  (h2 : w.hidden_lake_to_office = 7)
  (h3 : w.total_time = 32) : 
  time_to_restaurant w = 10 := by
  sorry

#eval time_to_restaurant { office_to_hidden_lake := 15, hidden_lake_to_office := 7, total_time := 32 }

end park_walk_time_l4010_401037


namespace stating_initial_order_correct_l4010_401059

/-- Represents the colors of the notebooks -/
inductive Color
  | Blue
  | Grey
  | Brown
  | Red
  | Yellow

/-- Represents a stack of notebooks -/
def Stack := List Color

/-- The first arrangement of notebooks -/
def first_arrangement : (Stack × Stack) :=
  ([Color.Red, Color.Yellow, Color.Grey], [Color.Brown, Color.Blue])

/-- The second arrangement of notebooks -/
def second_arrangement : (Stack × Stack) :=
  ([Color.Brown, Color.Red], [Color.Yellow, Color.Grey, Color.Blue])

/-- The hypothesized initial order of notebooks -/
def initial_order : Stack :=
  [Color.Brown, Color.Red, Color.Yellow, Color.Grey, Color.Blue]

/-- 
Theorem stating that the initial_order is correct given the two arrangements
-/
theorem initial_order_correct :
  ∃ (process : Stack → (Stack × Stack)),
    process initial_order = first_arrangement ∧
    process (initial_order.reverse.reverse) = second_arrangement :=
sorry

end stating_initial_order_correct_l4010_401059


namespace complex_multiplication_l4010_401046

/-- Given that i is the imaginary unit, prove that (2+i)(1-3i) = 5-5i -/
theorem complex_multiplication (i : ℂ) (hi : i * i = -1) :
  (2 + i) * (1 - 3*i) = 5 - 5*i := by
  sorry

end complex_multiplication_l4010_401046


namespace divide_by_approximate_700_l4010_401001

-- Define the approximation tolerance
def tolerance : ℝ := 0.001

-- Define the condition from the problem
def condition (x : ℝ) : Prop :=
  abs (49 / x - 700) < tolerance

-- State the theorem
theorem divide_by_approximate_700 :
  ∃ x : ℝ, condition x ∧ abs (x - 0.07) < tolerance :=
sorry

end divide_by_approximate_700_l4010_401001


namespace sunTzu_nests_count_l4010_401030

/-- Geometric sequence with first term a and common ratio r -/
def geometricSeq (a : ℕ) (r : ℕ) : ℕ → ℕ := fun n => a * r ^ (n - 1)

/-- The number of nests in Sun Tzu's Arithmetic problem -/
def sunTzuNests : ℕ := geometricSeq 9 9 4

theorem sunTzu_nests_count : sunTzuNests = 6561 := by
  sorry

end sunTzu_nests_count_l4010_401030


namespace shortest_distance_parabola_to_line_l4010_401055

/-- The shortest distance between a point on the parabola y = x^2 - 6x + 11 and the line y = 2x - 5 -/
theorem shortest_distance_parabola_to_line :
  let parabola := fun x : ℝ => x^2 - 6*x + 11
  let line := fun x : ℝ => 2*x - 5
  let distance := fun a : ℝ => |2*a - (a^2 - 6*a + 11) - 5| / Real.sqrt 5
  ∃ (min_dist : ℝ), min_dist = 16 * Real.sqrt 5 / 5 ∧
    ∀ a : ℝ, distance a ≥ min_dist :=
by sorry


end shortest_distance_parabola_to_line_l4010_401055


namespace parallel_lines_k_equals_3_l4010_401036

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = m₁ * x + b₁) ↔ (y = m₂ * x + b₂)) ↔ m₁ = m₂

/-- If the line y = kx - 1 is parallel to the line y = 3x, then k = 3 -/
theorem parallel_lines_k_equals_3 (k : ℝ) :
  (∃ x y : ℝ, y = k * x - 1 ∧ y = 3 * x) → k = 3 :=
by sorry

end parallel_lines_k_equals_3_l4010_401036


namespace norma_cards_l4010_401089

/-- 
Given that Norma loses 70 cards and has 18 cards left,
prove that she initially had 88 cards.
-/
theorem norma_cards : 
  ∀ (initial_cards : ℕ),
  (initial_cards - 70 = 18) → initial_cards = 88 := by
  sorry

end norma_cards_l4010_401089


namespace haircut_cost_l4010_401045

-- Define the constants
def hair_growth_rate : ℝ := 1.5
def max_hair_length : ℝ := 9
def min_hair_length : ℝ := 6
def tip_percentage : ℝ := 0.2
def annual_haircut_cost : ℝ := 324

-- Define the theorem
theorem haircut_cost (haircut_cost : ℝ) : 
  hair_growth_rate * 12 / (max_hair_length - min_hair_length) * 
  (haircut_cost * (1 + tip_percentage)) = annual_haircut_cost → 
  haircut_cost = 45 := by
sorry

end haircut_cost_l4010_401045


namespace geometric_series_sum_l4010_401017

theorem geometric_series_sum : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let series_sum : ℚ := (a * (1 - r^n)) / (1 - r)
  series_sum = 341/1024 := by sorry

end geometric_series_sum_l4010_401017


namespace junk_mail_distribution_l4010_401053

theorem junk_mail_distribution (total : ℕ) (blocks : ℕ) (first : ℕ) (second : ℕ) 
  (h1 : total = 2758)
  (h2 : blocks = 5)
  (h3 : first = 365)
  (h4 : second = 421) :
  (total - first - second) / (blocks - 2) = 657 := by
  sorry

end junk_mail_distribution_l4010_401053


namespace complex_magnitude_one_l4010_401006

theorem complex_magnitude_one (n : ℕ) (a : ℝ) (z : ℂ)
  (h_n : n ≥ 2)
  (h_a : 0 < a ∧ a < (n + 1 : ℝ) / (n - 1 : ℝ))
  (h_z : z^(n+1) - a * z^n + a * z - 1 = 0) :
  Complex.abs z = 1 := by
  sorry

end complex_magnitude_one_l4010_401006


namespace factorization_equality_l4010_401016

theorem factorization_equality (a b c : ℝ) : a^2 - 2*a*b + b^2 - c^2 = (a - b + c) * (a - b - c) := by
  sorry

end factorization_equality_l4010_401016


namespace imaginary_part_of_z_l4010_401010

theorem imaginary_part_of_z (z : ℂ) (h : z + (3 - 4*I) = 1) : z.im = 4 := by
  sorry

end imaginary_part_of_z_l4010_401010


namespace sixth_triangular_number_l4010_401057

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

/-- The 6th triangular number is 21 -/
theorem sixth_triangular_number : triangular 6 = 21 := by
  sorry

end sixth_triangular_number_l4010_401057


namespace opera_ticket_price_increase_l4010_401049

theorem opera_ticket_price_increase (last_year_price this_year_price : ℝ) 
  (h1 : last_year_price = 85)
  (h2 : this_year_price = 102) :
  (this_year_price - last_year_price) / last_year_price * 100 = 20 := by
sorry

end opera_ticket_price_increase_l4010_401049


namespace number_of_partitions_l4010_401034

-- Define the set A
def A : Set Nat := {1, 2}

-- Define what a partition is
def is_partition (A₁ A₂ : Set Nat) : Prop :=
  A₁ ∪ A₂ = A

-- Define when two partitions are considered the same
def same_partition (A₁ A₂ : Set Nat) : Prop :=
  A₁ = A₂

-- Define a function to count the number of different partitions
def count_partitions : Nat :=
  sorry

-- The theorem to prove
theorem number_of_partitions :
  count_partitions = 9 :=
sorry

end number_of_partitions_l4010_401034


namespace mixed_committee_probability_l4010_401077

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 6

def probability_mixed_committee : ℚ :=
  1 - (2 * Nat.choose boys committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 33187 / 33649 :=
sorry

end mixed_committee_probability_l4010_401077


namespace acme_vowel_soup_combinations_l4010_401084

def vowel_count : ℕ := 20
def word_length : ℕ := 5

theorem acme_vowel_soup_combinations :
  vowel_count ^ word_length = 3200000 := by
  sorry

end acme_vowel_soup_combinations_l4010_401084


namespace geometric_sequence_common_ratio_l4010_401044

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a2 : a 2 = 18) 
  (h_a4 : a 4 = 8) :
  ∃ q : ℝ, (q = 2/3 ∨ q = -2/3) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end geometric_sequence_common_ratio_l4010_401044


namespace relationship_abc_l4010_401033

theorem relationship_abc : 
  let a : ℝ := (1/3 : ℝ)^(2/3 : ℝ)
  let b : ℝ := (1/3 : ℝ)^(1/3 : ℝ)
  let c : ℝ := (2/3 : ℝ)^(1/3 : ℝ)
  c > b ∧ b > a := by sorry

end relationship_abc_l4010_401033


namespace f_inequality_solutions_l4010_401047

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m^2 + 1) * x + m

theorem f_inequality_solutions :
  (∀ x, f 2 x ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ m, m > 0 →
    (0 < m ∧ m < 1 →
      (∀ x, f m x > 0 ↔ x < m ∨ x > 1/m)) ∧
    (m = 1 →
      (∀ x, f m x > 0 ↔ x ≠ 1)) ∧
    (m > 1 →
      (∀ x, f m x > 0 ↔ x < 1/m ∨ x > m))) :=
by sorry

end f_inequality_solutions_l4010_401047


namespace ellipse_focal_length_l4010_401048

/-- The focal length of an ellipse with equation 2x^2 + 3y^2 = 1 is √6/3 -/
theorem ellipse_focal_length : 
  let a : ℝ := 1 / Real.sqrt 2
  let b : ℝ := 1 / Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c = Real.sqrt 6 / 3 := by sorry

end ellipse_focal_length_l4010_401048


namespace logistics_problem_l4010_401015

/-- Represents the problem of transporting goods using two types of trucks -/
theorem logistics_problem (total_goods : ℕ) (type_a_capacity : ℕ) (type_b_capacity : ℕ) (num_type_a : ℕ) :
  total_goods = 300 →
  type_a_capacity = 20 →
  type_b_capacity = 15 →
  num_type_a = 7 →
  ∃ (num_type_b : ℕ),
    num_type_b ≥ 11 ∧
    num_type_a * type_a_capacity + num_type_b * type_b_capacity ≥ total_goods ∧
    ∀ (m : ℕ), m < num_type_b →
      num_type_a * type_a_capacity + m * type_b_capacity < total_goods :=
by
  sorry


end logistics_problem_l4010_401015


namespace oliver_stickers_l4010_401011

theorem oliver_stickers (initial_stickers : ℕ) (remaining_stickers : ℕ) 
  (h1 : initial_stickers = 135)
  (h2 : remaining_stickers = 54)
  (h3 : ∃ x : ℚ, 0 ≤ x ∧ x < 1 ∧ 
    remaining_stickers = initial_stickers - (x * initial_stickers).floor - 
    ((2/5 : ℚ) * (initial_stickers - (x * initial_stickers).floor)).floor) :
  ∃ x : ℚ, x = 1/3 ∧ 
    remaining_stickers = initial_stickers - (x * initial_stickers).floor - 
    ((2/5 : ℚ) * (initial_stickers - (x * initial_stickers).floor)).floor :=
sorry

end oliver_stickers_l4010_401011


namespace sequence_properties_l4010_401069

def arithmetic_sequence (n : ℕ) : ℕ := n

def geometric_sequence (n : ℕ) : ℕ := 2^n

def S (n : ℕ) : ℚ := (n^2 + n) / 2

def T (n : ℕ) : ℕ := 2 * (2^n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → arithmetic_sequence n = n) ∧
  (∀ n : ℕ, n ≥ 1 → geometric_sequence n = 2^n) ∧
  (∀ n : ℕ, n < 8 → T n + arithmetic_sequence n ≤ 300) ∧
  (T 8 + arithmetic_sequence 8 > 300) :=
sorry

end sequence_properties_l4010_401069


namespace scientific_notation_of_189100_l4010_401085

/-- The scientific notation of 189100 is 1.891 × 10^5 -/
theorem scientific_notation_of_189100 :
  (189100 : ℝ) = 1.891 * (10 : ℝ)^5 := by sorry

end scientific_notation_of_189100_l4010_401085


namespace faculty_reduction_l4010_401038

theorem faculty_reduction (original : ℕ) (reduced : ℕ) (reduction_rate : ℚ) : 
  reduced = (1 - reduction_rate) * original ∧ 
  reduced = 195 ∧ 
  reduction_rate = 1/4 → 
  original = 260 :=
by sorry

end faculty_reduction_l4010_401038


namespace irreducible_fractions_l4010_401081

theorem irreducible_fractions (n : ℕ) : 
  (Nat.gcd (2*n + 13) (n + 7) = 1) ∧ 
  (Nat.gcd (2*n^2 - 1) (n + 1) = 1) ∧ 
  (Nat.gcd (n^2 - n + 1) (n^2 + 1) = 1) := by
  sorry

end irreducible_fractions_l4010_401081


namespace total_commission_proof_l4010_401039

def commission_rate : ℚ := 2 / 100

def house_prices : List ℚ := [157000, 499000, 125000]

def calculate_commission (price : ℚ) : ℚ :=
  price * commission_rate

theorem total_commission_proof :
  (house_prices.map calculate_commission).sum = 15620 := by
  sorry

end total_commission_proof_l4010_401039


namespace minimum_value_problem_l4010_401093

theorem minimum_value_problem (x y : ℝ) (h1 : 5 * x - x * y - y = -6) (h2 : x > -1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≥ z := by
  sorry

end minimum_value_problem_l4010_401093


namespace apple_rate_problem_l4010_401054

theorem apple_rate_problem (apple_rate : ℕ) : 
  (8 * apple_rate + 9 * 75 = 1235) → apple_rate = 70 := by
  sorry

end apple_rate_problem_l4010_401054


namespace visitor_and_revenue_properties_l4010_401027

/-- Represents the daily change in visitors (in 10,000 people) --/
def visitor_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

/-- The ticket price per person in yuan --/
def ticket_price : ℝ := 15

/-- Theorem stating the properties of visitor numbers and revenue --/
theorem visitor_and_revenue_properties (a : ℝ) : 
  let visitors_day3 := a + visitor_changes[0] + visitor_changes[1]
  let max_visitors := (List.map (λ i => a + (List.take i visitor_changes).sum) (List.range 7)).maximum?
  let total_visitors := a * 7 + visitor_changes.sum
  (visitors_day3 = a + 2.4) ∧ 
  (max_visitors = some (a + 2.8)) ∧
  (a = 2 → total_visitors * ticket_price * 10000 = 4.08 * 10^6) := by sorry

end visitor_and_revenue_properties_l4010_401027


namespace fraction_of_powers_equals_three_fifths_l4010_401099

theorem fraction_of_powers_equals_three_fifths :
  (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3/5 := by
  sorry

end fraction_of_powers_equals_three_fifths_l4010_401099


namespace triangle_value_l4010_401029

theorem triangle_value (triangle p : ℤ) 
  (h1 : triangle + p = 85)
  (h2 : (triangle + p) + 3 * p = 154) : 
  triangle = 62 := by
  sorry

end triangle_value_l4010_401029


namespace square_area_is_400_l4010_401002

/-- A square is cut into five rectangles of equal area, with one rectangle having a width of 5. -/
structure CutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of the rectangle with known width -/
  known_width : ℝ
  /-- The area of each rectangle -/
  rectangle_area : ℝ
  /-- The known width is 5 -/
  known_width_is_5 : known_width = 5
  /-- The square is divided into 5 rectangles of equal area -/
  five_equal_rectangles : side * side = 5 * rectangle_area

/-- The area of the square is 400 -/
theorem square_area_is_400 (s : CutSquare) : s.side * s.side = 400 := by
  sorry

end square_area_is_400_l4010_401002


namespace min_extractions_to_reverse_l4010_401007

/-- Represents a stack of cards -/
def CardStack := List Nat

/-- Represents an extraction operation on a card stack -/
def Extraction := CardStack → CardStack

/-- Checks if a card stack is in reverse order -/
def is_reversed (stack : CardStack) : Prop :=
  stack = List.reverse (List.range stack.length)

/-- Theorem: Minimum number of extractions to reverse a card stack -/
theorem min_extractions_to_reverse (n : Nat) :
  ∃ (k : Nat) (extractions : List Extraction),
    k = n / 2 + 1 ∧
    extractions.length = k ∧
    is_reversed (extractions.foldl (λ acc f => f acc) (List.range n)) :=
  sorry

end min_extractions_to_reverse_l4010_401007


namespace logger_productivity_l4010_401090

/-- Represents the number of trees one logger can cut down per day -/
def trees_per_logger_per_day (forest_length : ℕ) (forest_width : ℕ) (trees_per_square_mile : ℕ) 
  (days_per_month : ℕ) (num_loggers : ℕ) (num_months : ℕ) : ℕ :=
  let total_trees := forest_length * forest_width * trees_per_square_mile
  let total_days := num_months * days_per_month
  total_trees / (num_loggers * total_days)

theorem logger_productivity : 
  trees_per_logger_per_day 4 6 600 30 8 10 = 6 := by
  sorry

#eval trees_per_logger_per_day 4 6 600 30 8 10

end logger_productivity_l4010_401090


namespace magic_forest_coin_difference_l4010_401014

theorem magic_forest_coin_difference :
  ∀ (x y : ℕ),
  let trees_with_no_coins := 2 * x
  let trees_with_one_coin := y
  let trees_with_two_coins := 3
  let trees_with_three_coins := x
  let trees_with_four_coins := 4
  let total_coins := y + 3 * x + 22
  let total_trees := 3 * x + y + 7
  total_coins - total_trees = 15 :=
by
  sorry

end magic_forest_coin_difference_l4010_401014


namespace sum_of_powers_implies_sum_power_l4010_401012

theorem sum_of_powers_implies_sum_power (a b : ℝ) : 
  a^2009 + b^2009 = 0 → (a + b)^2009 = 0 := by sorry

end sum_of_powers_implies_sum_power_l4010_401012


namespace arithmetic_mean_of_fractions_l4010_401088

theorem arithmetic_mean_of_fractions :
  let a := 3 / 5
  let b := 5 / 7
  let c := 9 / 14
  let arithmetic_mean := (a + b) / 2
  arithmetic_mean = 23 / 35 ∧ arithmetic_mean ≠ c := by
  sorry

end arithmetic_mean_of_fractions_l4010_401088


namespace range_of_a_l4010_401087

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1)^2 > 4
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a))) →
  (∀ a : ℝ, a ≥ 1 ↔ (∀ x : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a)))) :=
by sorry

end range_of_a_l4010_401087


namespace base_three_five_digits_l4010_401074

theorem base_three_five_digits : ∃! b : ℕ, b ≥ 2 ∧ b^4 ≤ 200 ∧ 200 < b^5 := by sorry

end base_three_five_digits_l4010_401074


namespace quadratic_inequality_result_l4010_401031

theorem quadratic_inequality_result (y : ℝ) (h : y^2 - 7*y + 12 < 0) :
  44 < y^2 + 7*y + 14 ∧ y^2 + 7*y + 14 < 58 := by
  sorry

end quadratic_inequality_result_l4010_401031


namespace noemi_initial_money_l4010_401096

/-- The amount of money Noemi lost on roulette -/
def roulette_loss : ℕ := 400

/-- The amount of money Noemi lost on blackjack -/
def blackjack_loss : ℕ := 500

/-- The amount of money Noemi still has in her purse -/
def remaining_money : ℕ := 800

/-- The initial amount of money Noemi had -/
def initial_money : ℕ := roulette_loss + blackjack_loss + remaining_money

theorem noemi_initial_money : 
  initial_money = roulette_loss + blackjack_loss + remaining_money := by
  sorry

end noemi_initial_money_l4010_401096


namespace stating_special_numeral_satisfies_condition_l4010_401022

/-- 
A numeral with two 1's where the difference between their place values is 99.99.
-/
def special_numeral : ℝ := 1.11

/-- 
The difference between the place values of the two 1's in the special numeral.
-/
def place_value_difference : ℝ := 99.99

/-- 
Theorem stating that the special_numeral satisfies the required condition.
-/
theorem special_numeral_satisfies_condition : 
  (100 : ℝ) - (1 / 100 : ℝ) = place_value_difference :=
by sorry

end stating_special_numeral_satisfies_condition_l4010_401022


namespace speed_calculation_l4010_401067

theorem speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 50 → time = 2.5 → speed = distance / time → speed = 20 := by
sorry

end speed_calculation_l4010_401067


namespace equation_solutions_l4010_401063

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = -11 ∧ x₂ = 1 ∧ 
  (∀ x : ℝ, 4 * (2 * x + 1)^2 = 9 * (x - 3)^2 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solutions_l4010_401063


namespace ellipse_minor_axis_length_l4010_401065

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse defined by its center and semi-axes lengths -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis length
  b : ℝ  -- semi-minor axis length

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- The given points -/
def points : List Point := [
  { x := 1, y := 1 },
  { x := 0, y := 0 },
  { x := 0, y := 3 },
  { x := 4, y := 0 },
  { x := 4, y := 3 }
]

theorem ellipse_minor_axis_length :
  ∃ (e : Ellipse),
    (∀ p ∈ points, pointOnEllipse p e) ∧
    (e.center.x = 2 ∧ e.center.y = 1.5) ∧
    (e.a = 2) ∧
    (e.b * 2 = 2 * Real.sqrt 3) :=
by sorry

#check ellipse_minor_axis_length

end ellipse_minor_axis_length_l4010_401065


namespace f_has_three_zeros_l4010_401092

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := (m * (x^2 - 1)) / x - 2 * Real.log x

theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧ 
  (∀ x, x > 0 → f (1/2) x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry


end f_has_three_zeros_l4010_401092


namespace inequality_addition_l4010_401023

theorem inequality_addition (x y : ℝ) (h : x < y) : x + 5 < y + 5 := by
  sorry

end inequality_addition_l4010_401023


namespace line_translation_coincidence_l4010_401003

/-- 
Given a line y = kx + 2 in the Cartesian plane,
prove that if the line is translated upward by 3 units
and then rightward by 2 units, and the resulting line
coincides with the original line, then k = 3/2.
-/
theorem line_translation_coincidence (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 2 ↔ y = k * (x - 2) + 5) → k = 3/2 := by
  sorry

end line_translation_coincidence_l4010_401003


namespace enrique_commission_l4010_401064

/-- Calculates the total commission for a salesperson given their sales and commission rate. -/
def calculate_commission (suit_price : ℚ) (suit_count : ℕ) 
                         (shirt_price : ℚ) (shirt_count : ℕ) 
                         (loafer_price : ℚ) (loafer_count : ℕ) 
                         (commission_rate : ℚ) : ℚ :=
  let total_sales := suit_price * suit_count + 
                     shirt_price * shirt_count + 
                     loafer_price * loafer_count
  total_sales * commission_rate

/-- Theorem stating that Enrique's commission is $300.00 given his sales and commission rate. -/
theorem enrique_commission :
  calculate_commission 700 2 50 6 150 2 (15/100) = 300 := by
  sorry

end enrique_commission_l4010_401064


namespace water_transfer_theorem_l4010_401091

/-- Represents a water canister with a given capacity and current water level. -/
structure Canister where
  capacity : ℝ
  water : ℝ
  h_water_nonneg : 0 ≤ water
  h_water_le_capacity : water ≤ capacity

/-- The result of pouring water from one canister to another. -/
structure PourResult where
  source : Canister
  target : Canister

theorem water_transfer_theorem (c d : Canister) 
  (h_c_half_full : c.water = c.capacity / 2)
  (h_d_capacity : d.capacity = 2 * c.capacity)
  (h_d_third_full : d.water = d.capacity / 3)
  : ∃ (result : PourResult), 
    result.target.water = result.target.capacity ∧ 
    result.source.water = result.source.capacity / 12 := by
  sorry

end water_transfer_theorem_l4010_401091


namespace unique_perpendicular_to_skew_lines_l4010_401061

/-- A line in three-dimensional space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- A line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicular lines
  sorry

theorem unique_perpendicular_to_skew_lines 
  (p : ℝ × ℝ × ℝ) (l1 l2 : Line3D) (h : are_skew l1 l2) :
  ∃! l : Line3D, l.point = p ∧ is_perpendicular l l1 ∧ is_perpendicular l l2 :=
sorry

end unique_perpendicular_to_skew_lines_l4010_401061


namespace circular_well_volume_l4010_401078

/-- The volume of a circular cylinder with diameter 2 metres and height 14 metres is 14π cubic metres. -/
theorem circular_well_volume :
  let diameter : ℝ := 2
  let depth : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = 14 * π := by
  sorry

end circular_well_volume_l4010_401078


namespace sufficient_not_necessary_l4010_401062

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, (1 < x ∧ x < 2) → x * (x - 3) < 0) ∧
  (∃ x, x * (x - 3) < 0 ∧ ¬(1 < x ∧ x < 2)) :=
sorry

end sufficient_not_necessary_l4010_401062


namespace discounted_shirt_price_l4010_401068

/-- Given a shirt sold at a 30% discount for 560 units of currency,
    prove that the original price was 800 units of currency. -/
theorem discounted_shirt_price (discount_percent : ℝ) (discounted_price : ℝ) :
  discount_percent = 30 →
  discounted_price = 560 →
  (1 - discount_percent / 100) * 800 = discounted_price := by
sorry

end discounted_shirt_price_l4010_401068


namespace sin_cos_difference_equals_half_l4010_401080

theorem sin_cos_difference_equals_half : 
  Real.sin (-(10 * π / 180)) * Real.cos (160 * π / 180) - 
  Real.sin (80 * π / 180) * Real.sin (200 * π / 180) = 1/2 := by
  sorry

end sin_cos_difference_equals_half_l4010_401080


namespace degree_three_polynomial_l4010_401097

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 6*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 10*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The theorem stating that c = -3/5 makes h(x) a polynomial of degree 3 -/
theorem degree_three_polynomial :
  ∃ (c : ℝ), c = -3/5 ∧ 
  (∀ (x : ℝ), h c x = 2 + (-15 - 3*c)*x + (4 - 0*c)*x^2 + (-5 - 7*c)*x^3) :=
sorry

end degree_three_polynomial_l4010_401097


namespace parabola_coeff_sum_l4010_401013

/-- A parabola with equation y = px^2 + qx + r, vertex (-3, 7), and passing through (-6, 4) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := -3
  vertex_y : ℝ := 7
  point_x : ℝ := -6
  point_y : ℝ := 4
  eq_at_vertex : 7 = p * (-3)^2 + q * (-3) + r
  eq_at_point : 4 = p * (-6)^2 + q * (-6) + r

/-- The sum of coefficients p, q, and r for the parabola is 7/3 -/
theorem parabola_coeff_sum (par : Parabola) : par.p + par.q + par.r = 7/3 := by
  sorry

end parabola_coeff_sum_l4010_401013


namespace polygon_angle_sum_l4010_401025

theorem polygon_angle_sum (n : ℕ) : 
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 := by
  sorry

end polygon_angle_sum_l4010_401025


namespace salmon_trip_count_l4010_401009

theorem salmon_trip_count (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : female_salmon = 259378) : 
  male_salmon + female_salmon = 971639 := by
  sorry

end salmon_trip_count_l4010_401009


namespace distinct_arrangements_of_three_letters_l4010_401041

/-- The number of distinct arrangements of 3 unique letters -/
def arrangements_of_three_letters : ℕ := 6

/-- The word consists of 3 distinct letters -/
def number_of_letters : ℕ := 3

theorem distinct_arrangements_of_three_letters : 
  arrangements_of_three_letters = Nat.factorial number_of_letters := by
  sorry

end distinct_arrangements_of_three_letters_l4010_401041


namespace partnership_investment_l4010_401018

/-- Represents a partnership with three partners -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  duration : ℝ
  a_share : ℝ
  b_share : ℝ

/-- Theorem stating the conditions and the result to be proven -/
theorem partnership_investment (p : Partnership)
  (ha : p.a_investment = 11000)
  (hc : p.c_investment = 23000)
  (hd : p.duration = 8)
  (hsa : p.a_share = 2431)
  (hsb : p.b_share = 3315) :
  p.b_investment = 15000 := by
  sorry


end partnership_investment_l4010_401018


namespace plastic_for_one_ruler_l4010_401073

/-- The amount of plastic needed to make one ruler, given the total amount of plastic and the number of rulers that can be made. -/
def plastic_per_ruler (total_plastic : ℕ) (num_rulers : ℕ) : ℚ :=
  (total_plastic : ℚ) / (num_rulers : ℚ)

/-- Theorem stating that 8 grams of plastic are needed to make one ruler. -/
theorem plastic_for_one_ruler :
  plastic_per_ruler 828 103 = 8 := by
  sorry

end plastic_for_one_ruler_l4010_401073


namespace range_of_m_l4010_401058

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := by
sorry

end range_of_m_l4010_401058


namespace probability_black_white_balls_l4010_401005

/-- The probability of picking one black ball and one white ball from a jar -/
theorem probability_black_white_balls (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = black_balls + white_balls + green_balls)
  (h2 : black_balls = 3)
  (h3 : white_balls = 3)
  (h4 : green_balls = 1) :
  (black_balls * white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3 / 7 := by
sorry

end probability_black_white_balls_l4010_401005


namespace min_tan_half_angle_l4010_401051

theorem min_tan_half_angle (A B C : Real) (h1 : A + B + C = π) 
  (h2 : Real.tan (A/2) + Real.tan (B/2) = 1) :
  ∃ (m : Real), m = 3/4 ∧ ∀ x, x = Real.tan (C/2) → x ≥ m := by
  sorry

end min_tan_half_angle_l4010_401051


namespace range_equality_odd_decreasing_function_l4010_401052

-- Statement 1
theorem range_equality (f : ℝ → ℝ) : Set.range f = Set.range (fun x ↦ f (x + 1)) := by sorry

-- Statement 3
theorem odd_decreasing_function (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing_neg : ∀ x y, x < y → y < 0 → f y < f x) : 
  ∀ x y, 0 < x → x < y → f y < f x := by sorry

end range_equality_odd_decreasing_function_l4010_401052


namespace two_digit_reverse_diff_64_l4010_401019

/-- Given a two-digit number, return the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_reverse_diff_64 (N : ℕ) :
  is_two_digit N →
  N - reverse_digits N = 64 →
  N = 90 :=
by sorry

end two_digit_reverse_diff_64_l4010_401019


namespace books_rearrangement_l4010_401060

theorem books_rearrangement (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1500 → 
  books_per_initial_box = 42 → 
  books_per_new_box = 45 → 
  (initial_boxes * books_per_initial_box) % books_per_new_box = 0 :=
by sorry

end books_rearrangement_l4010_401060


namespace benny_picked_two_l4010_401035

-- Define the total number of apples picked
def total_apples : ℕ := 11

-- Define the number of apples Dan picked
def dan_apples : ℕ := 9

-- Define Benny's apples as the difference between total and Dan's
def benny_apples : ℕ := total_apples - dan_apples

-- Theorem stating that Benny picked 2 apples
theorem benny_picked_two : benny_apples = 2 := by
  sorry

end benny_picked_two_l4010_401035


namespace overtime_rate_calculation_l4010_401040

/-- Calculate the overtime rate given the following conditions:
  * Regular hourly rate
  * Total weekly pay
  * Total hours worked
  * Overtime hours worked
-/
def calculate_overtime_rate (regular_rate : ℚ) (total_pay : ℚ) (total_hours : ℕ) (overtime_hours : ℕ) : ℚ :=
  let regular_hours := total_hours - overtime_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_hours

theorem overtime_rate_calculation :
  let regular_rate : ℚ := 60 / 100  -- 60 cents per hour
  let total_pay : ℚ := 3240 / 100   -- $32.40
  let total_hours : ℕ := 50
  let overtime_hours : ℕ := 8
  calculate_overtime_rate regular_rate total_pay total_hours overtime_hours = 90 / 100 := by
    sorry

#eval calculate_overtime_rate (60 / 100) (3240 / 100) 50 8

end overtime_rate_calculation_l4010_401040


namespace similar_triangle_shortest_side_l4010_401050

/-- Given two similar right triangles, where the first triangle has a side of 15 units and a hypotenuse of 34 units, and the second triangle has a hypotenuse of 68 units, the shortest side of the second triangle is 2√931 units. -/
theorem similar_triangle_shortest_side :
  ∀ (a b c d e : ℝ),
  a^2 + 15^2 = 34^2 →  -- Pythagorean theorem for the first triangle
  a ≤ 15 →  -- a is the shortest side of the first triangle
  c^2 + d^2 = 68^2 →  -- Pythagorean theorem for the second triangle
  c / a = d / 15 →  -- triangles are similar
  c / a = 68 / 34 →  -- ratio of hypotenuses
  c = 2 * Real.sqrt 931 :=
by sorry

end similar_triangle_shortest_side_l4010_401050


namespace oranges_bought_l4010_401076

/-- Proves the number of oranges bought given the conditions of the problem -/
theorem oranges_bought (total_cost : ℚ) (apple_cost : ℚ) (orange_cost : ℚ) (apple_count : ℕ) :
  total_cost = 4.56 →
  apple_count = 3 →
  orange_cost = apple_cost + 0.28 →
  apple_cost = 0.26 →
  (total_cost - apple_count * apple_cost) / orange_cost = 7 := by
  sorry

end oranges_bought_l4010_401076


namespace max_xy_value_l4010_401079

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 3*x + 8*y = 48) :
  x*y ≤ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 8*y₀ = 48 ∧ x₀*y₀ = 18 :=
sorry

end max_xy_value_l4010_401079


namespace expand_and_simplify_l4010_401026

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * (4 / y - 7 * y^3) = 3 / y - 21 * y^3 / 4 := by sorry

end expand_and_simplify_l4010_401026


namespace arithmetic_equality_l4010_401032

theorem arithmetic_equality : 2^2 * 7 + 5 * 12 + 7^2 * 2 + 6 * 3 = 212 := by
  sorry

end arithmetic_equality_l4010_401032


namespace arithmetic_expression_equals_24_l4010_401070

theorem arithmetic_expression_equals_24 : (2 + 4 / 10) * 10 = 24 := by
  sorry

#check arithmetic_expression_equals_24

end arithmetic_expression_equals_24_l4010_401070


namespace line_up_ways_l4010_401000

def number_of_people : ℕ := 5

theorem line_up_ways (n : ℕ) (h : n = number_of_people) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 :=
sorry

end line_up_ways_l4010_401000


namespace arc_problem_l4010_401024

theorem arc_problem (X Y Z : ℝ × ℝ) (d : ℝ) : 
  X.1 = 0 ∧ X.2 = 0 ∧  -- Assume X is at origin
  Y.1 = 15 ∧ Y.2 = 0 ∧  -- Assume Y is on x-axis
  Z.1^2 + Z.2^2 = (3 + d)^2 ∧  -- XZ = 3 + d
  (Z.1 - 15)^2 + Z.2^2 = (12 + d)^2 →  -- YZ = 12 + d
  d = 5 := by
sorry

end arc_problem_l4010_401024


namespace additional_charge_per_segment_l4010_401083

/-- Proves that the additional charge per 2/5 of a mile is $0.40 --/
theorem additional_charge_per_segment (initial_fee : ℚ) (trip_distance : ℚ) (total_charge : ℚ) 
  (h1 : initial_fee = 9/4)  -- $2.25
  (h2 : trip_distance = 18/5)  -- 3.6 miles
  (h3 : total_charge = 117/20)  -- $5.85
  : (total_charge - initial_fee) / (trip_distance / (2/5)) = 2/5 := by
  sorry

end additional_charge_per_segment_l4010_401083


namespace sachin_age_l4010_401082

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age * 9 = rahul_age * 6) :
  sachin_age = 14 := by sorry

end sachin_age_l4010_401082


namespace student_pet_difference_l4010_401072

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 24

/-- The number of rabbits in each classroom -/
def rabbits_per_classroom : ℕ := 2

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 3

/-- Theorem: The difference between the total number of students and the total number of pets
    in all fourth-grade classrooms is 95 -/
theorem student_pet_difference :
  num_classrooms * students_per_classroom - 
  (num_classrooms * rabbits_per_classroom + num_classrooms * hamsters_per_classroom) = 95 := by
  sorry

end student_pet_difference_l4010_401072


namespace quadratic_roots_relation_l4010_401042

/-- Given two quadratic equations y² + dy + e = 0 and 4x² - ax - 12 = 0,
    where the roots of the first equation are each three more than 
    the roots of the second equation, prove that e = (3a + 24) / 4 -/
theorem quadratic_roots_relation (a d e : ℝ) : 
  (∀ x y : ℝ, (4 * x^2 - a * x - 12 = 0 → y^2 + d * y + e = 0 → y = x + 3)) →
  e = (3 * a + 24) / 4 := by
  sorry

end quadratic_roots_relation_l4010_401042


namespace angle_measure_l4010_401071

theorem angle_measure (PQR PQS : ℝ) (h1 : PQR = 40) (h2 : PQS = 15) : PQR - PQS = 25 := by
  sorry

end angle_measure_l4010_401071


namespace largest_valid_number_l4010_401066

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 100 = (n / 100 % 10 + n / 1000) % 10) ∧
  (n % 10 = (n / 10 % 10 + n / 100 % 10) % 10)

theorem largest_valid_number : 
  (∀ m : ℕ, is_valid m → m ≤ 9099) ∧ is_valid 9099 :=
sorry

end largest_valid_number_l4010_401066


namespace divisible_by_117_and_2_less_than_2011_l4010_401086

theorem divisible_by_117_and_2_less_than_2011 : 
  (Finset.filter (fun n => n < 2011 ∧ n % 117 = 0 ∧ n % 2 = 0) (Finset.range 2011)).card = 8 := by
  sorry

end divisible_by_117_and_2_less_than_2011_l4010_401086


namespace remainder_problem_l4010_401021

theorem remainder_problem (N : ℤ) : N % 357 = 36 → N % 17 = 2 := by
  sorry

end remainder_problem_l4010_401021


namespace engineering_exam_pass_percentage_l4010_401020

theorem engineering_exam_pass_percentage
  (total_male : ℕ)
  (total_female : ℕ)
  (male_eng_percent : ℚ)
  (female_eng_percent : ℚ)
  (male_pass_percent : ℚ)
  (female_pass_percent : ℚ)
  (h1 : total_male = 120)
  (h2 : total_female = 100)
  (h3 : male_eng_percent = 25 / 100)
  (h4 : female_eng_percent = 20 / 100)
  (h5 : male_pass_percent = 20 / 100)
  (h6 : female_pass_percent = 25 / 100)
  : (↑(Nat.floor ((male_eng_percent * male_pass_percent * total_male + female_eng_percent * female_pass_percent * total_female) / (male_eng_percent * total_male + female_eng_percent * total_female) * 100)) : ℚ) = 22 := by
  sorry

end engineering_exam_pass_percentage_l4010_401020


namespace A_intersect_B_empty_l4010_401004

/-- The set A defined by the equation (y-3)/(x-2) = a+1 -/
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = a + 1}

/-- The set B defined by the equation (a^2-1)x + (a-1)y = 15 -/
def B (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a^2 - 1) * p.1 + (a - 1) * p.2 = 15}

/-- The theorem stating that A ∩ B is empty if and only if a is in the set {-1, -4, 1, 5/2} -/
theorem A_intersect_B_empty (a : ℝ) :
  A a ∩ B a = ∅ ↔ a ∈ ({-1, -4, 1, (5:ℝ)/2} : Set ℝ) := by
  sorry

end A_intersect_B_empty_l4010_401004


namespace complex_equation_sum_l4010_401043

theorem complex_equation_sum (a b : ℝ) (h : (3 * b : ℂ) + (2 * a - 2) * Complex.I = 1 - Complex.I) : 
  a + b = 5/6 := by
  sorry

end complex_equation_sum_l4010_401043


namespace matrix_multiplication_result_l4010_401075

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 0, -1; 0, 3, -2; -2, 3, 2]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, -1, 0; 2, 0, -1; 3, 0, 0]
def C : Matrix (Fin 3) (Fin 3) ℝ := !![-1, -2, 0; 0, 0, -3; 10, 2, -3]

theorem matrix_multiplication_result : A * B = C := by sorry

end matrix_multiplication_result_l4010_401075
