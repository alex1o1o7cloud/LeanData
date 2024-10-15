import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_l3678_367839

theorem sum_of_odd_numbers (N : ℕ) : 
  1001 + 1003 + 1005 + 1007 + 1009 = 5100 - N → N = 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_l3678_367839


namespace NUMINAMATH_CALUDE_tank_capacity_l3678_367844

/-- Proves that the capacity of a tank is 21600 litres given specific inlet and outlet conditions. -/
theorem tank_capacity : 
  ∀ (outlet_time inlet_rate extended_time : ℝ),
  outlet_time = 10 →
  inlet_rate = 16 * 60 →
  extended_time = outlet_time + 8 →
  ∃ (capacity : ℝ),
  capacity / outlet_time - inlet_rate = capacity / extended_time ∧
  capacity = 21600 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l3678_367844


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3678_367850

theorem polynomial_value_theorem (x : ℂ) (h : x^2 + x + 1 = 0) :
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3678_367850


namespace NUMINAMATH_CALUDE_final_net_worth_l3678_367867

/-- Represents a person's assets --/
structure Assets where
  cash : Int
  has_house : Bool
  has_vehicle : Bool

/-- Represents a transaction between two people --/
inductive Transaction
  | sell_house (price : Int)
  | sell_vehicle (price : Int)

/-- Performs a transaction and updates the assets of both parties --/
def perform_transaction (a b : Assets) (t : Transaction) : Assets × Assets :=
  match t with
  | Transaction.sell_house price => 
    ({ cash := a.cash + price, has_house := false, has_vehicle := a.has_vehicle },
     { cash := b.cash - price, has_house := true, has_vehicle := b.has_vehicle })
  | Transaction.sell_vehicle price => 
    ({ cash := a.cash - price, has_house := a.has_house, has_vehicle := true },
     { cash := b.cash + price, has_house := b.has_house, has_vehicle := false })

/-- Calculates the net worth of a person given their assets and the values of the house and vehicle --/
def net_worth (a : Assets) (house_value vehicle_value : Int) : Int :=
  a.cash + (if a.has_house then house_value else 0) + (if a.has_vehicle then vehicle_value else 0)

/-- The main theorem stating the final net worth of Mr. A and Mr. B --/
theorem final_net_worth (initial_a initial_b : Assets) 
  (house_value vehicle_value : Int) (transactions : List Transaction) : 
  initial_a.cash = 20000 → initial_a.has_house = true → initial_a.has_vehicle = false →
  initial_b.cash = 22000 → initial_b.has_house = false → initial_b.has_vehicle = true →
  house_value = 20000 → vehicle_value = 10000 →
  transactions = [
    Transaction.sell_house 25000,
    Transaction.sell_vehicle 12000,
    Transaction.sell_house 18000,
    Transaction.sell_vehicle 9000
  ] →
  let (final_a, final_b) := transactions.foldl 
    (fun (acc : Assets × Assets) (t : Transaction) => perform_transaction acc.1 acc.2 t) 
    (initial_a, initial_b)
  net_worth final_a house_value vehicle_value = 40000 ∧ 
  net_worth final_b house_value vehicle_value = 8000 := by
  sorry


end NUMINAMATH_CALUDE_final_net_worth_l3678_367867


namespace NUMINAMATH_CALUDE_product_increase_by_2016_l3678_367838

theorem product_increase_by_2016 : ∃ (a b c : ℕ), 
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
sorry

end NUMINAMATH_CALUDE_product_increase_by_2016_l3678_367838


namespace NUMINAMATH_CALUDE_apps_deletion_ways_l3678_367829

-- Define the total number of applications
def total_apps : ℕ := 21

-- Define the number of applications to be deleted
def apps_to_delete : ℕ := 6

-- Define the number of special applications
def special_apps : ℕ := 6

-- Define the number of special apps to be selected
def special_apps_to_select : ℕ := 3

-- Define the number of pairs of special apps
def special_pairs : ℕ := 3

-- Theorem statement
theorem apps_deletion_ways :
  (2^special_pairs) * (Nat.choose (total_apps - special_apps) (apps_to_delete - special_apps_to_select)) = 3640 :=
sorry

end NUMINAMATH_CALUDE_apps_deletion_ways_l3678_367829


namespace NUMINAMATH_CALUDE_marble_probability_theorem_l3678_367883

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing 4 marbles of the same color -/
def probSameColor (counts : MarbleCounts) : ℚ :=
  let total := counts.red + counts.white + counts.blue + counts.green
  let probRed := Nat.choose counts.red 4 / Nat.choose total 4
  let probWhite := Nat.choose counts.white 4 / Nat.choose total 4
  let probBlue := Nat.choose counts.blue 4 / Nat.choose total 4
  let probGreen := Nat.choose counts.green 4 / Nat.choose total 4
  probRed + probWhite + probBlue + probGreen

theorem marble_probability_theorem (counts : MarbleCounts) 
    (h : counts = ⟨6, 7, 8, 9⟩) : 
    probSameColor counts = 82 / 9135 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_theorem_l3678_367883


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3678_367840

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4*x + 3*y + k = 0 → y^2 = 16*x) →
  (∃! p : ℝ × ℝ, (4*(p.1) + 3*(p.2) + k = 0) ∧ (p.2)^2 = 16*(p.1)) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3678_367840


namespace NUMINAMATH_CALUDE_platonic_self_coincidences_l3678_367827

/-- Represents a regular polyhedron -/
structure RegularPolyhedron where
  n : ℕ  -- number of sides of each face
  F : ℕ  -- number of faces
  is_regular : n ≥ 3 ∧ F ≥ 4  -- conditions for regularity

/-- Calculates the number of self-coincidences for a regular polyhedron -/
def self_coincidences (p : RegularPolyhedron) : ℕ :=
  2 * p.n * p.F

/-- Theorem stating the number of self-coincidences for each Platonic solid -/
theorem platonic_self_coincidences :
  ∃ (tetrahedron cube octahedron dodecahedron icosahedron : RegularPolyhedron),
    (self_coincidences tetrahedron = 24) ∧
    (self_coincidences cube = 48) ∧
    (self_coincidences octahedron = 48) ∧
    (self_coincidences dodecahedron = 120) ∧
    (self_coincidences icosahedron = 120) :=
by sorry

end NUMINAMATH_CALUDE_platonic_self_coincidences_l3678_367827


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l3678_367816

/-- If a circle with equation (x-1)^2 + (y-2)^2 = 1 is symmetric about the line y = x + b, then b = 1 -/
theorem circle_symmetry_line (b : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 ↔ (x - 1)^2 + ((x + b) - 2)^2 = 1) → 
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l3678_367816


namespace NUMINAMATH_CALUDE_a_10_value_l3678_367851

def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value (a : ℕ → ℤ) 
  (h_seq : arithmeticSequence a) 
  (h_a7 : a 7 = 4) 
  (h_a8 : a 8 = 1) : 
  a 10 = -5 := by
sorry

end NUMINAMATH_CALUDE_a_10_value_l3678_367851


namespace NUMINAMATH_CALUDE_jar_water_problem_l3678_367835

theorem jar_water_problem (s l w : ℝ) (hs : s > 0) (hl : l > 0) (hw : w > 0)
  (h1 : w = l / 2)  -- Larger jar is 1/2 full
  (h2 : w + w = 2 * l / 3)  -- When combined, 2/3 of larger jar is filled
  (h3 : s < l)  -- Smaller jar has less capacity
  : w = 3 * s / 4  -- Smaller jar was 3/4 full
  := by sorry

end NUMINAMATH_CALUDE_jar_water_problem_l3678_367835


namespace NUMINAMATH_CALUDE_range_of_a_l3678_367899

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (A ∪ B a = {x | x < 1}) ↔ (-1 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3678_367899


namespace NUMINAMATH_CALUDE_penguin_count_l3678_367828

/-- The number of penguins in a zoo can be determined by adding the number of penguins 
    already fed and the number of penguins still to be fed. -/
theorem penguin_count (total_fish : ℕ) (fed_penguins : ℕ) (to_be_fed : ℕ) :
  total_fish ≥ fed_penguins + to_be_fed →
  fed_penguins + to_be_fed = fed_penguins + to_be_fed :=
by
  sorry

#check penguin_count

end NUMINAMATH_CALUDE_penguin_count_l3678_367828


namespace NUMINAMATH_CALUDE_maxim_method_correct_only_for_24_l3678_367891

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9 ∧ ones ≥ 1

/-- Maxim's division method -/
def maximMethod (A : ℚ) (N : TwoDigitNumber) : ℚ :=
  A / (N.tens + N.ones : ℚ) - A / (N.tens * N.ones : ℚ)

/-- The theorem stating that 24 is the only two-digit number for which Maxim's method works -/
theorem maxim_method_correct_only_for_24 :
  ∀ (N : TwoDigitNumber),
    (∀ (A : ℚ), maximMethod A N = A / (10 * N.tens + N.ones : ℚ)) ↔
    (N.tens = 2 ∧ N.ones = 4) :=
sorry

end NUMINAMATH_CALUDE_maxim_method_correct_only_for_24_l3678_367891


namespace NUMINAMATH_CALUDE_girl_multiplication_mistake_l3678_367812

theorem girl_multiplication_mistake (x : ℝ) : 43 * x - 34 * x = 1215 → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_girl_multiplication_mistake_l3678_367812


namespace NUMINAMATH_CALUDE_simplify_expression_l3678_367849

theorem simplify_expression (x : ℝ) : 5 * x + 2 * (4 + x) = 7 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3678_367849


namespace NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l3678_367864

theorem smallest_divisor_after_subtraction (n : ℕ) (m : ℕ) (d : ℕ) : 
  n = 378461 →
  m = 5 →
  d = 47307 →
  (n - m) % d = 0 ∧
  ∀ k : ℕ, 5 < k → k < d → (n - m) % k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_subtraction_l3678_367864


namespace NUMINAMATH_CALUDE_inequality_theorem_l3678_367857

theorem inequality_theorem (a b : ℝ) : 
  (a * b > 0 → b / a + a / b ≥ 2) ∧ 
  (a + 2 * b = 1 → 3^a + 9^b ≥ 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3678_367857


namespace NUMINAMATH_CALUDE_function_characterization_l3678_367890

theorem function_characterization (f : ℕ → ℕ) :
  (∀ m n : ℕ, (m^2 + f n) ∣ (m * f m + n)) →
  (∀ n : ℕ, f n = n) := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l3678_367890


namespace NUMINAMATH_CALUDE_division_equation_solution_l3678_367860

theorem division_equation_solution :
  ∃ x : ℝ, (0.009 / x = 0.1) ∧ (x = 0.09) := by
  sorry

end NUMINAMATH_CALUDE_division_equation_solution_l3678_367860


namespace NUMINAMATH_CALUDE_milk_needed_for_cookies_l3678_367872

/-- The number of cups in a quart -/
def cups_per_quart : ℚ := 4

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℕ := 24

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 6

/-- The number of cups of milk needed for the target number of cookies -/
def milk_needed : ℚ := 3

theorem milk_needed_for_cookies :
  milk_needed = (target_cookies : ℚ) / cookies_per_three_quarts * (3 * cups_per_quart) :=
sorry

end NUMINAMATH_CALUDE_milk_needed_for_cookies_l3678_367872


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3678_367889

theorem framed_painting_ratio :
  ∀ (x : ℝ),
    x > 0 →
    (20 + 2*x) * (30 + 6*x) - 20 * 30 = 20 * 30 * (3/4) →
    (min (20 + 2*x) (30 + 6*x)) / (max (20 + 2*x) (30 + 6*x)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3678_367889


namespace NUMINAMATH_CALUDE_eight_power_x_equals_one_eighth_of_two_power_thirty_l3678_367807

theorem eight_power_x_equals_one_eighth_of_two_power_thirty (x : ℝ) : 
  (1/8 : ℝ) * (2^30) = 8^x → x = 9 := by
sorry

end NUMINAMATH_CALUDE_eight_power_x_equals_one_eighth_of_two_power_thirty_l3678_367807


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_eq_2_l3678_367831

-- Define the hyperbola equation
def hyperbola_eq (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote equation
def asymptote_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 0

-- Theorem statement
theorem hyperbola_asymptote_implies_a_eq_2 :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, hyperbola_eq x y a ↔ asymptote_eq x y) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_eq_2_l3678_367831


namespace NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l3678_367856

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l3678_367856


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3678_367863

theorem sqrt_product_plus_one : 
  Real.sqrt ((41:ℝ) * 40 * 39 * 38 + 1) = 1559 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3678_367863


namespace NUMINAMATH_CALUDE_luncheon_tables_l3678_367847

theorem luncheon_tables (invited : ℕ) (no_show : ℕ) (per_table : ℕ) 
  (h1 : invited = 18) 
  (h2 : no_show = 12) 
  (h3 : per_table = 3) : 
  (invited - no_show) / per_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_tables_l3678_367847


namespace NUMINAMATH_CALUDE_choose_4_from_10_l3678_367866

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_4_from_10_l3678_367866


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3678_367811

-- Define atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define number of atoms in the compound
def num_Ca : ℕ := 1
def num_O : ℕ := 2
def num_H : ℕ := 2

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_Ca : ℝ) * atomic_weight_Ca + 
  (num_O : ℝ) * atomic_weight_O + 
  (num_H : ℝ) * atomic_weight_H

-- Theorem statement
theorem compound_molecular_weight : 
  molecular_weight = 74.10 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3678_367811


namespace NUMINAMATH_CALUDE_pauls_diner_cost_l3678_367897

/-- Represents the pricing and discount policy at Paul's Diner -/
structure PaulsDiner where
  sandwich_price : ℕ
  soda_price : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ

/-- Calculates the total cost for a purchase at Paul's Diner -/
def total_cost (diner : PaulsDiner) (sandwiches : ℕ) (sodas : ℕ) : ℕ :=
  let sandwich_cost := diner.sandwich_price * sandwiches
  let soda_cost := diner.soda_price * sodas
  let subtotal := sandwich_cost + soda_cost
  if sandwiches > diner.discount_threshold then
    subtotal - diner.discount_amount
  else
    subtotal

/-- Theorem stating that the total cost for 6 sandwiches and 3 sodas is 29 -/
theorem pauls_diner_cost :
  ∃ (d : PaulsDiner), total_cost d 6 3 = 29 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_pauls_diner_cost_l3678_367897


namespace NUMINAMATH_CALUDE_periodic_function_theorem_l3678_367841

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem periodic_function_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f :=
sorry

end NUMINAMATH_CALUDE_periodic_function_theorem_l3678_367841


namespace NUMINAMATH_CALUDE_sequence_characterization_l3678_367874

theorem sequence_characterization (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_characterization_l3678_367874


namespace NUMINAMATH_CALUDE_marble_distribution_l3678_367859

/-- The minimum number of additional marbles needed and the sum of marbles for specific friends -/
theorem marble_distribution (n : Nat) (initial_marbles : Nat) 
  (h1 : n = 12) (h2 : initial_marbles = 34) : 
  let additional_marbles := (n * (n + 1)) / 2 - initial_marbles
  let third_friend := 3
  let seventh_friend := 7
  let eleventh_friend := 11
  (additional_marbles = 44) ∧ 
  (third_friend + seventh_friend + eleventh_friend = 21) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3678_367859


namespace NUMINAMATH_CALUDE_rachel_picked_apples_l3678_367842

/-- Represents the number of apples Rachel picked from her tree -/
def apples_picked : ℕ := 2

/-- The initial number of apples on Rachel's tree -/
def initial_apples : ℕ := 4

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 3

/-- The final number of apples on the tree -/
def final_apples : ℕ := 5

/-- Theorem stating that the number of apples Rachel picked is correct -/
theorem rachel_picked_apples :
  initial_apples - apples_picked + new_apples = final_apples :=
by sorry

end NUMINAMATH_CALUDE_rachel_picked_apples_l3678_367842


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3678_367898

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the first line y = 5x + 3 -/
def slope1 : ℝ := 5

/-- The slope of the second line y = (3k)x + 7 -/
def slope2 (k : ℝ) : ℝ := 3 * k

/-- Theorem: If the lines y = 5x + 3 and y = (3k)x + 7 are parallel, then k = 5/3 -/
theorem parallel_lines_k_value (k : ℝ) :
  parallel slope1 (slope2 k) → k = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3678_367898


namespace NUMINAMATH_CALUDE_shoe_price_ratio_l3678_367846

theorem shoe_price_ratio (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := (2/3) * selling_price
  cost_price / marked_price = 1/2 := by sorry

end NUMINAMATH_CALUDE_shoe_price_ratio_l3678_367846


namespace NUMINAMATH_CALUDE_prob_three_out_of_five_l3678_367880

def prob_single_win : ℚ := 2/3

theorem prob_three_out_of_five :
  let n : ℕ := 5
  let k : ℕ := 3
  let p : ℚ := prob_single_win
  (n.choose k) * p^k * (1-p)^(n-k) = 80/243 := by
sorry

end NUMINAMATH_CALUDE_prob_three_out_of_five_l3678_367880


namespace NUMINAMATH_CALUDE_cyclic_symmetric_count_l3678_367808

-- Definition of cyclic symmetric expression
def is_cyclic_symmetric (σ : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, σ a b c = σ b c a ∧ σ a b c = σ c a b

-- Define the three expressions
def σ₁ (a b c : ℝ) : ℝ := a * b * c
def σ₂ (a b c : ℝ) : ℝ := a^2 - b^2 + c^2
noncomputable def σ₃ (A B C : ℝ) : ℝ := Real.cos C * Real.cos (A - B) - (Real.cos C)^2

-- Theorem statement
theorem cyclic_symmetric_count :
  (is_cyclic_symmetric σ₁) ∧
  ¬(is_cyclic_symmetric σ₂) ∧
  (is_cyclic_symmetric σ₃) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_symmetric_count_l3678_367808


namespace NUMINAMATH_CALUDE_all_2k_trips_use_both_modes_all_2k_plus_1_trips_use_both_modes_l3678_367865

/-- A type representing cities in a country. -/
structure City where
  id : Nat

/-- A type representing transportation modes. -/
inductive TransportMode
  | Bus
  | Flight

/-- A function that determines if two cities are directly connected by a given transport mode. -/
def directConnection (c1 c2 : City) (mode : TransportMode) : Prop :=
  sorry

/-- A proposition stating that any two cities are connected by either a direct flight or a direct bus route. -/
axiom connected_cities (c1 c2 : City) :
  directConnection c1 c2 TransportMode.Bus ∨ directConnection c1 c2 TransportMode.Flight

/-- A type representing a round trip as a list of cities. -/
def RoundTrip := List City

/-- A function that checks if a round trip uses both bus and flight. -/
def usesBothModes (trip : RoundTrip) : Prop :=
  sorry

/-- A theorem stating that all round trips touching 2k cities (k > 3) must use both bus and flight. -/
theorem all_2k_trips_use_both_modes (k : Nat) (h : k > 3) :
  ∀ (trip : RoundTrip), trip.length = 2 * k → usesBothModes trip :=
  sorry

/-- The main theorem to prove: if all round trips touching 2k cities (k > 3) must use both bus and flight,
    then all round trips touching 2k+1 cities must also use both bus and flight. -/
theorem all_2k_plus_1_trips_use_both_modes (k : Nat) (h : k > 3) :
  (∀ (trip : RoundTrip), trip.length = 2 * k → usesBothModes trip) →
  (∀ (trip : RoundTrip), trip.length = 2 * k + 1 → usesBothModes trip) :=
  sorry

end NUMINAMATH_CALUDE_all_2k_trips_use_both_modes_all_2k_plus_1_trips_use_both_modes_l3678_367865


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l3678_367826

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.cos x

theorem derivative_f_at_pi : 
  deriv f π = -2 * π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l3678_367826


namespace NUMINAMATH_CALUDE_lindas_age_multiple_l3678_367815

/-- Given:
  - Linda's age (L) is 3 more than a certain multiple (M) of Jane's age (J)
  - In five years, the sum of their ages will be 28
  - Linda's current age is 13
Prove that the multiple M is equal to 2 -/
theorem lindas_age_multiple (J L M : ℕ) : 
  L = M * J + 3 →
  L = 13 →
  L + 5 + J + 5 = 28 →
  M = 2 := by
sorry

end NUMINAMATH_CALUDE_lindas_age_multiple_l3678_367815


namespace NUMINAMATH_CALUDE_all_statements_false_l3678_367822

-- Define prime and composite numbers
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

-- Define the four statements
def statement1 : Prop := ∀ p q : ℕ, isPrime p → isPrime q → isComposite (p + q)
def statement2 : Prop := ∀ a b : ℕ, isComposite a → isComposite b → isComposite (a + b)
def statement3 : Prop := ∀ p c : ℕ, isPrime p → isComposite c → isComposite (p + c)
def statement4 : Prop := ∀ p c : ℕ, isPrime p → isComposite c → ¬(isComposite (p + c))

-- Theorem stating that all four statements are false
theorem all_statements_false : ¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 :=
sorry

end NUMINAMATH_CALUDE_all_statements_false_l3678_367822


namespace NUMINAMATH_CALUDE_students_guinea_pigs_difference_l3678_367873

theorem students_guinea_pigs_difference : 
  let students_per_class : ℕ := 25
  let guinea_pigs_per_class : ℕ := 3
  let num_classes : ℕ := 6
  let total_students : ℕ := students_per_class * num_classes
  let total_guinea_pigs : ℕ := guinea_pigs_per_class * num_classes
  total_students - total_guinea_pigs = 132 :=
by
  sorry


end NUMINAMATH_CALUDE_students_guinea_pigs_difference_l3678_367873


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3678_367833

/-- The surface area of a cuboid -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 8, width 10, and height 12 is 592 -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 8 10 12 = 592 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3678_367833


namespace NUMINAMATH_CALUDE_student_rank_from_left_l3678_367814

/-- Given a total number of students and a student's rank from the right,
    calculate the student's rank from the left. -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem: Given 20 students in total and a student ranked 13th from the right,
    prove that the student's rank from the left is 8th. -/
theorem student_rank_from_left :
  let totalStudents : ℕ := 20
  let rankFromRight : ℕ := 13
  rankFromLeft totalStudents rankFromRight = 8 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_from_left_l3678_367814


namespace NUMINAMATH_CALUDE_vector_param_validity_l3678_367885

/-- A vector parameterization of a line -/
structure VectorParam where
  x0 : ℝ
  y0 : ℝ
  dx : ℝ
  dy : ℝ

/-- The line equation y = -3/2x - 5 -/
def line_equation (x y : ℝ) : Prop := y = -3/2 * x - 5

/-- Predicate for a valid vector parameterization -/
def is_valid_param (p : VectorParam) : Prop :=
  line_equation p.x0 p.y0 ∧ p.dy = -3/2 * p.dx

theorem vector_param_validity (p : VectorParam) :
  is_valid_param p ↔ ∀ t : ℝ, line_equation (p.x0 + t * p.dx) (p.y0 + t * p.dy) :=
sorry

end NUMINAMATH_CALUDE_vector_param_validity_l3678_367885


namespace NUMINAMATH_CALUDE_bobs_income_changes_l3678_367834

def initial_income : ℝ := 2750
def february_increase : ℝ := 0.15
def march_decrease : ℝ := 0.10

theorem bobs_income_changes (initial : ℝ) (increase : ℝ) (decrease : ℝ) :
  initial = initial_income →
  increase = february_increase →
  decrease = march_decrease →
  initial * (1 + increase) * (1 - decrease) = 2846.25 :=
by sorry

end NUMINAMATH_CALUDE_bobs_income_changes_l3678_367834


namespace NUMINAMATH_CALUDE_palmer_photos_before_trip_l3678_367810

def photos_before_trip (first_week : ℕ) (second_week_multiplier : ℕ) (third_fourth_week : ℕ) (total_after_trip : ℕ) : ℕ :=
  total_after_trip - (first_week + second_week_multiplier * first_week + third_fourth_week)

theorem palmer_photos_before_trip :
  photos_before_trip 50 2 80 380 = 150 :=
by sorry

end NUMINAMATH_CALUDE_palmer_photos_before_trip_l3678_367810


namespace NUMINAMATH_CALUDE_tangent_line_slope_range_l3678_367854

open Real Set

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem tangent_line_slope_range :
  let symmetry_axis : ℝ := π / 3
  let tangent_line (m c : ℝ) (x y : ℝ) : Prop := x + m * y + c = 0
  ∃ (c : ℝ), ∃ (x : ℝ), tangent_line m c x (f x) ↔ 
    m ∈ Iic (-1/4) ∪ Ici (1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_range_l3678_367854


namespace NUMINAMATH_CALUDE_building_heights_l3678_367801

/-- Given three buildings with specified height relationships, calculate their total height. -/
theorem building_heights (height_1 : ℝ) : 
  height_1 = 600 →
  let height_2 := 2 * height_1
  let height_3 := 3 * (height_1 + height_2)
  height_1 + height_2 + height_3 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_l3678_367801


namespace NUMINAMATH_CALUDE_triangle_height_l3678_367800

/-- Given a triangle with area 3 square meters and base 2 meters, its height is 3 meters -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 3 → base = 2 → area = (base * height) / 2 → height = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l3678_367800


namespace NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l3678_367818

theorem acute_triangle_sine_cosine_inequality (α β γ : Real) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = π) : 
  Real.sin α * Real.sin β * Real.sin γ > 5 * Real.cos α * Real.cos β * Real.cos γ := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_cosine_inequality_l3678_367818


namespace NUMINAMATH_CALUDE_probability_both_classes_l3678_367804

-- Define the total number of students
def total_students : ℕ := 40

-- Define the number of students in Mandarin
def mandarin_students : ℕ := 30

-- Define the number of students in German
def german_students : ℕ := 35

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the theorem
theorem probability_both_classes : 
  let students_both := mandarin_students + german_students - total_students
  let students_only_mandarin := mandarin_students - students_both
  let students_only_german := german_students - students_both
  let total_ways := choose total_students 2
  let ways_not_both := choose students_only_mandarin 2 + choose students_only_german 2
  (total_ways - ways_not_both) / total_ways = 145 / 156 := by
sorry

end NUMINAMATH_CALUDE_probability_both_classes_l3678_367804


namespace NUMINAMATH_CALUDE_simplify_fraction_l3678_367868

theorem simplify_fraction (x : ℝ) (h : x = 2) : 15 * x^5 / (45 * x^3) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3678_367868


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l3678_367878

theorem cubic_polynomial_root (b c : ℚ) :
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 5 - 2*Real.sqrt 2) →
  (∃ (y : ℤ), y^3 + b*y + c = 0 ∧ y = -10) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l3678_367878


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3678_367802

-- Problem 1
theorem problem_1 : 13 + (-24) - (-40) = 29 := by sorry

-- Problem 2
theorem problem_2 : 3 * (-2) + (-36) / 4 = -15 := by sorry

-- Problem 3
theorem problem_3 : (1 + 3/4 - 7/8 - 7/16) / (-7/8) = -1/2 := by sorry

-- Problem 4
theorem problem_4 : (-2)^3 / 4 - (10 - (-1)^10 * 2) = -10 := by sorry

-- Problem 5
theorem problem_5 (x y : ℝ) : 7*x*y + 2 - 3*x*y - 5 = 4*x*y - 3 := by sorry

-- Problem 6
theorem problem_6 (x : ℝ) : 4*x^2 - (5*x + x^2) + 6*x - 2*x^2 = x^2 + x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3678_367802


namespace NUMINAMATH_CALUDE_vector_problem_l3678_367892

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-1, -1)

theorem vector_problem :
  let magnitude := Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2)
  magnitude = 3 * Real.sqrt 2 ∧
  (let angle := Real.arccos ((a.1 + b.1) * (2 * a.1 - b.1) + (a.2 + b.2) * (2 * a.2 - b.2)) /
    (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) * Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2))
   angle = π / 4 → a.1 * b.1 + a.2 * b.2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3678_367892


namespace NUMINAMATH_CALUDE_system_solution_l3678_367862

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3678_367862


namespace NUMINAMATH_CALUDE_circle_equation_l3678_367803

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ  -- (ρ, θ)

-- Define the line l
def line_l (p : PolarCoordinate) : Prop :=
  p.1 * Real.cos p.2 + p.1 * Real.sin p.2 = 2

-- Define the point M where line l intersects the polar axis
def point_M : ℝ × ℝ := (2, 0)  -- Cartesian coordinates

-- Define the circle with OM as diameter
def circle_OM (p : PolarCoordinate) : Prop :=
  p.1 = 2 * Real.cos p.2

-- Theorem statement
theorem circle_equation (p : PolarCoordinate) :
  line_l p → circle_OM p :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3678_367803


namespace NUMINAMATH_CALUDE_unique_root_condition_l3678_367895

/-- The equation ln(x+a) - 4(x+a)^2 + a = 0 has a unique root if and only if a = (3 ln 2 + 1) / 2 -/
theorem unique_root_condition (a : ℝ) :
  (∃! x : ℝ, Real.log (x + a) - 4 * (x + a)^2 + a = 0) ↔ 
  a = (3 * Real.log 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l3678_367895


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_square_root_9_l3678_367852

theorem cube_root_27_times_fourth_root_81_times_square_root_9 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_square_root_9_l3678_367852


namespace NUMINAMATH_CALUDE_smaller_integer_proof_l3678_367861

theorem smaller_integer_proof (x y : ℤ) (h1 : x + y = -9) (h2 : y - x = 1) : x = -5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_proof_l3678_367861


namespace NUMINAMATH_CALUDE_function_properties_l3678_367896

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.sin (ω * x))^2 + Real.sqrt 3

def is_symmetry_axis (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_symmetry : ∃ x₁ x₂, is_symmetry_axis (f ω) x₁ ∧ is_symmetry_axis (f ω) x₂)
  (h_min_dist : ∃ x₁ x₂, is_symmetry_axis (f ω) x₁ ∧ is_symmetry_axis (f ω) x₂ ∧ |x₁ - x₂| ≥ π/2 ∧ 
    ∀ y₁ y₂, is_symmetry_axis (f ω) y₁ ∧ is_symmetry_axis (f ω) y₂ → |y₁ - y₂| ≥ |x₁ - x₂|) :
  (ω = 1) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-5*π/12 + k*π) (π/12 + k*π), 
    ∀ y ∈ Set.Icc (-5*π/12 + k*π) (π/12 + k*π), x < y → (f ω x) < (f ω y)) ∧
  (∀ α : ℝ, f ω α = 2/3 → Real.sin (5*π/6 - 4*α) = -7/9) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3678_367896


namespace NUMINAMATH_CALUDE_x_value_when_derivative_is_three_l3678_367824

def f (x : ℝ) := x^3

theorem x_value_when_derivative_is_three (x : ℝ) (h1 : x > 0) (h2 : (deriv f) x = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_derivative_is_three_l3678_367824


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3678_367821

/-- For a parabola with equation y^2 = 4x, the distance between its focus and directrix is 2. -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 4*x → ∃ (f d : ℝ × ℝ),
    (f.1 = 1 ∧ f.2 = 0) ∧  -- focus
    (d.1 = -1 ∧ ∀ t, d.2 = t) ∧  -- directrix
    (f.1 - d.1 = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3678_367821


namespace NUMINAMATH_CALUDE_vector_problem_l3678_367817

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (-1, 2)
def c (m : ℝ) : ℝ × ℝ := (2, m)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_problem (m : ℝ) :
  (dot_product a (c m) < m^2 → m > 4 ∨ m < -2) ∧
  (parallel (a.1 + (c m).1, a.2 + (c m).2) b → m = -14) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l3678_367817


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3678_367823

theorem inequality_not_always_true (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  ∃ a b, 0 < b ∧ b < a ∧ ¬((1 / (a - b)) > (1 / b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3678_367823


namespace NUMINAMATH_CALUDE_root_equations_l3678_367894

/-- Given two constants c and d, prove that they satisfy the given conditions -/
theorem root_equations (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x + c) * (x + d) * (x + 8) = 0 ∧
    (y + c) * (y + d) * (y + 8) = 0 ∧
    (z + c) * (z + d) * (z + 8) = 0 ∧
    (x + 2) ≠ 0 ∧ (y + 2) ≠ 0 ∧ (z + 2) ≠ 0) ∧
  (∃! w : ℝ, (w + 3*c) * (w + 2) * (w + 4) = 0 ∧
    (w + d) ≠ 0 ∧ (w + 8) ≠ 0) →
  c = 2/3 ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_root_equations_l3678_367894


namespace NUMINAMATH_CALUDE_max_ones_on_board_l3678_367893

/-- The operation that replaces two numbers with their GCD and LCM -/
def replace_with_gcd_lcm (a b : ℕ) : List ℕ :=
  [Nat.gcd a b, Nat.lcm a b]

/-- The set of numbers on the board -/
def board : Set ℕ := Finset.range 2014

/-- A sequence of operations on the board -/
def operation_sequence := List (ℕ × ℕ)

/-- Apply a sequence of operations to the board -/
def apply_operations (ops : operation_sequence) (b : Set ℕ) : Set ℕ :=
  sorry

/-- Count the number of 1's in a set of natural numbers -/
def count_ones (s : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating the maximum number of 1's obtainable -/
theorem max_ones_on_board :
  ∃ (ops : operation_sequence),
    ∀ (ops' : operation_sequence),
      count_ones (apply_operations ops board) ≥ count_ones (apply_operations ops' board) ∧
      count_ones (apply_operations ops board) = 1007 :=
  sorry

end NUMINAMATH_CALUDE_max_ones_on_board_l3678_367893


namespace NUMINAMATH_CALUDE_positive_t_value_l3678_367887

theorem positive_t_value (a b : ℂ) (t : ℝ) (h1 : a * b = t - 3 * Complex.I) 
  (h2 : Complex.abs a = 3) (h3 : Complex.abs b = 5) : 
  t > 0 → t = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_t_value_l3678_367887


namespace NUMINAMATH_CALUDE_proposition_q_false_l3678_367830

theorem proposition_q_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_proposition_q_false_l3678_367830


namespace NUMINAMATH_CALUDE_equation_solution_l3678_367805

theorem equation_solution : ∃ x : ℝ, 2 * x - 3 = 6 - x :=
  by
    use 3
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3678_367805


namespace NUMINAMATH_CALUDE_product_modulo_25_l3678_367855

theorem product_modulo_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (68 * 95 * 113) % 25 = m ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_25_l3678_367855


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3678_367843

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor where
  side_length : ℕ
  is_valid : side_length > 0

/-- Calculates the number of black tiles on the boundary of the floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.side_length - 4

/-- Calculates the total number of tiles on the floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem stating that a square floor with 100 black boundary tiles has 676 total tiles. -/
theorem square_floor_tiles (floor : SquareFloor) :
  black_tiles floor = 100 → total_tiles floor = 676 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3678_367843


namespace NUMINAMATH_CALUDE_odd_factorials_equal_sum_factorial_l3678_367884

def product_of_odd_factorials (m : ℕ) : ℕ :=
  (List.range m).foldl (λ acc i => acc * Nat.factorial (2 * i + 1)) 1

def sum_of_first_n (m : ℕ) : ℕ :=
  m * (m + 1) / 2

theorem odd_factorials_equal_sum_factorial (m : ℕ) :
  (product_of_odd_factorials m = Nat.factorial (sum_of_first_n m)) ↔ (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4) := by
  sorry

end NUMINAMATH_CALUDE_odd_factorials_equal_sum_factorial_l3678_367884


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l3678_367806

theorem trigonometric_inequalities (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  (Real.sqrt (y * z / x) + Real.sqrt (z * x / y) + Real.sqrt (x * y / z) ≥ Real.sqrt 3) ∧
  (Real.sqrt (x * y / (z + x * y)) + Real.sqrt (y * z / (x + y * z)) + Real.sqrt (z * x / (y + z * x)) ≤ 3 / 2) ∧
  (x / (x + y * z) + y / (y + z * x) + z / (z + x * y) ≤ 9 / 4) ∧
  ((x - y * z) / (x + y * z) + (y - z * x) / (y + z * x) + (z - x * y) / (z + x * y) ≤ 3 / 2) ∧
  ((x - y * z) / (x + y * z) * (y - z * x) / (y + z * x) * (z - x * y) / (z + x * y) ≤ 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l3678_367806


namespace NUMINAMATH_CALUDE_problem_solution_l3678_367825

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3678_367825


namespace NUMINAMATH_CALUDE_prob_spade_heart_king_l3678_367819

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a spade, then a heart, then a King from a standard 52-card deck -/
theorem prob_spade_heart_king :
  (NumSpades * NumHearts * NumKings) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 17 / 3683 := by
  sorry


end NUMINAMATH_CALUDE_prob_spade_heart_king_l3678_367819


namespace NUMINAMATH_CALUDE_factorization_proof_l3678_367837

theorem factorization_proof (z : ℂ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3678_367837


namespace NUMINAMATH_CALUDE_equation_root_conditions_l3678_367882

theorem equation_root_conditions (a : ℝ) : 
  (∃ x > 0, |x| = a*x - a) ∧ 
  (∀ x < 0, |x| ≠ a*x - a) → 
  a > 1 ∨ a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_conditions_l3678_367882


namespace NUMINAMATH_CALUDE_shems_earnings_proof_l3678_367876

/-- Calculates Shem's earnings for a workday given Kem's hourly rate, Shem's rate multiplier, and hours worked. -/
def shems_daily_earnings (kems_hourly_rate : ℝ) (shems_rate_multiplier : ℝ) (hours_worked : ℝ) : ℝ :=
  kems_hourly_rate * shems_rate_multiplier * hours_worked

/-- Proves that Shem's earnings for an 8-hour workday is $80, given the conditions. -/
theorem shems_earnings_proof (kems_hourly_rate : ℝ) (shems_rate_multiplier : ℝ) (hours_worked : ℝ) 
    (h1 : kems_hourly_rate = 4)
    (h2 : shems_rate_multiplier = 2.5)
    (h3 : hours_worked = 8) :
    shems_daily_earnings kems_hourly_rate shems_rate_multiplier hours_worked = 80 := by
  sorry

end NUMINAMATH_CALUDE_shems_earnings_proof_l3678_367876


namespace NUMINAMATH_CALUDE_face_value_calculation_l3678_367886

/-- Given a company's dividend rate, an investor's return on investment, and the purchase price of shares, 
    calculate the face value of the shares. -/
theorem face_value_calculation (dividend_rate : ℝ) (roi : ℝ) (purchase_price : ℝ) :
  dividend_rate = 0.185 →
  roi = 0.25 →
  purchase_price = 37 →
  ∃ (face_value : ℝ), face_value * dividend_rate = purchase_price * roi ∧ face_value = 50 := by
  sorry

end NUMINAMATH_CALUDE_face_value_calculation_l3678_367886


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l3678_367813

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row -/
def arrangementCount : ℕ :=
  let volunteerCount : ℕ := 5
  let elderlyCount : ℕ := 2
  let totalCount : ℕ := volunteerCount + elderlyCount
  let endPositions : ℕ := 2  -- number of end positions
  let intermediatePositions : ℕ := totalCount - endPositions - 1  -- -1 for elderly pair

  -- Choose volunteers for end positions
  let endArrangements : ℕ := volunteerCount * (volunteerCount - 1)
  
  -- Arrange remaining volunteers and elderly pair
  let intermediateArrangements : ℕ := Nat.factorial intermediatePositions
  
  -- Arrange elderly within their pair
  let elderlyArrangements : ℕ := Nat.factorial elderlyCount

  endArrangements * intermediateArrangements * elderlyArrangements

theorem arrangement_count_correct :
  arrangementCount = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_correct_l3678_367813


namespace NUMINAMATH_CALUDE_system_solution_l3678_367877

/-- 
Given a system of equations x - y = a and xy = b, 
this theorem proves that the solutions are 
(x, y) = ((a + √(a² + 4b))/2, (-a + √(a² + 4b))/2) and 
(x, y) = ((a - √(a² + 4b))/2, (-a - √(a² + 4b))/2).
-/
theorem system_solution (a b : ℝ) :
  let x₁ := (a + Real.sqrt (a^2 + 4*b)) / 2
  let y₁ := (-a + Real.sqrt (a^2 + 4*b)) / 2
  let x₂ := (a - Real.sqrt (a^2 + 4*b)) / 2
  let y₂ := (-a - Real.sqrt (a^2 + 4*b)) / 2
  (x₁ - y₁ = a ∧ x₁ * y₁ = b) ∧ 
  (x₂ - y₂ = a ∧ x₂ * y₂ = b) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3678_367877


namespace NUMINAMATH_CALUDE_G_equals_2F_l3678_367879

noncomputable section

variable (x : ℝ)

def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def G (x : ℝ) : ℝ := F ((x * (1 + x^2)) / (1 + x^4))

theorem G_equals_2F : G x = 2 * F x := by sorry

end NUMINAMATH_CALUDE_G_equals_2F_l3678_367879


namespace NUMINAMATH_CALUDE_range_of_a_l3678_367871

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3678_367871


namespace NUMINAMATH_CALUDE_fourth_root_of_25000000_l3678_367853

theorem fourth_root_of_25000000 : (70.7 : ℝ)^4 = 25000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_25000000_l3678_367853


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l3678_367836

/-- The function f(x) = (3 - x^2) * e^x is monotonically increasing on the interval (-3, 1) -/
theorem monotone_increasing_interval (x : ℝ) :
  StrictMonoOn (fun x => (3 - x^2) * Real.exp x) (Set.Ioo (-3) 1) := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l3678_367836


namespace NUMINAMATH_CALUDE_third_circle_radius_l3678_367869

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) : 
  r₁ = 25 → r₂ = 40 → 
  π * r₃^2 = (π * r₂^2 - π * r₁^2) / 2 →
  r₃ = 15 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l3678_367869


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l3678_367888

/-- Represents the tax calculation for a citizen in Country X --/
def TaxCalculation (income : ℝ) (totalTax : ℝ) (baseRate : ℝ) (baseIncome : ℝ) : Prop :=
  income > baseIncome ∧
  totalTax = baseRate * baseIncome + 
    ((income - baseIncome) * (totalTax - baseRate * baseIncome) / (income - baseIncome))

/-- Theorem statement for the tax calculation problem --/
theorem tax_rate_calculation :
  ∀ (income : ℝ) (totalTax : ℝ),
  TaxCalculation income totalTax 0.15 40000 →
  income = 50000 →
  totalTax = 8000 →
  (totalTax - 0.15 * 40000) / (income - 40000) = 0.20 := by
  sorry


end NUMINAMATH_CALUDE_tax_rate_calculation_l3678_367888


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3678_367809

theorem book_purchase_problem (total_A total_B only_B both : ℕ) 
  (h1 : total_A = 2 * total_B)
  (h2 : both = 500)
  (h3 : both = 2 * only_B) :
  total_A - both = 1000 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3678_367809


namespace NUMINAMATH_CALUDE_quadratic_solution_l3678_367820

theorem quadratic_solution : 
  ∀ x : ℝ, x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3678_367820


namespace NUMINAMATH_CALUDE_sum_of_roots_l3678_367848

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3678_367848


namespace NUMINAMATH_CALUDE_overall_profit_calculation_l3678_367875

/-- Calculate the overall profit from selling a refrigerator and a mobile phone -/
theorem overall_profit_calculation (refrigerator_cost mobile_cost : ℕ) 
  (refrigerator_loss_percent mobile_profit_percent : ℚ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 5 / 100 →
  mobile_profit_percent = 10 / 100 →
  (refrigerator_cost * (1 - refrigerator_loss_percent) + 
   mobile_cost * (1 + mobile_profit_percent)) - 
  (refrigerator_cost + mobile_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_calculation_l3678_367875


namespace NUMINAMATH_CALUDE_relay_race_time_reduction_l3678_367845

theorem relay_race_time_reduction (T T₁ T₂ T₃ T₄ T₅ : ℝ) 
  (total_time : T = T₁ + T₂ + T₃ + T₄ + T₅)
  (first_runner : T₁ / 2 + T₂ + T₃ + T₄ + T₅ = 0.95 * T)
  (second_runner : T₁ + T₂ / 2 + T₃ + T₄ + T₅ = 0.90 * T)
  (third_runner : T₁ + T₂ + T₃ / 2 + T₄ + T₅ = 0.88 * T)
  (fourth_runner : T₁ + T₂ + T₃ + T₄ / 2 + T₅ = 0.85 * T) :
  T₁ + T₂ + T₃ + T₄ + T₅ / 2 = 0.92 * T := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_reduction_l3678_367845


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l3678_367870

theorem quadratic_root_implies_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a = 0) ∧ (2^2 - a = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l3678_367870


namespace NUMINAMATH_CALUDE_max_value_of_f_l3678_367832

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 12 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3678_367832


namespace NUMINAMATH_CALUDE_equation_solutions_l3678_367881

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1) ∧
  (∀ x : ℝ, 2*x^3 = -16 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3678_367881


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l3678_367858

theorem bus_seating_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let people_per_seat : ℕ := 3
  let back_seat_capacity : ℕ := 7
  let total_capacity : ℕ := left_seats * people_per_seat + right_seats * people_per_seat + back_seat_capacity
  total_capacity = 88 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l3678_367858
