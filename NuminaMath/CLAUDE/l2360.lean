import Mathlib

namespace NUMINAMATH_CALUDE_bert_grocery_spending_l2360_236030

/-- Represents Bert's spending scenario -/
structure BertSpending where
  initial_amount : ℚ
  hardware_fraction : ℚ
  dry_cleaning_amount : ℚ
  final_amount : ℚ

/-- Calculates the fraction spent at the grocery store -/
def grocery_fraction (b : BertSpending) : ℚ :=
  let remaining_before_grocery := b.initial_amount - (b.hardware_fraction * b.initial_amount) - b.dry_cleaning_amount
  let spent_at_grocery := remaining_before_grocery - b.final_amount
  spent_at_grocery / remaining_before_grocery

/-- Theorem stating that Bert spent 1/2 of his remaining money at the grocery store -/
theorem bert_grocery_spending (b : BertSpending) 
  (h1 : b.initial_amount = 52)
  (h2 : b.hardware_fraction = 1/4)
  (h3 : b.dry_cleaning_amount = 9)
  (h4 : b.final_amount = 15) :
  grocery_fraction b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_bert_grocery_spending_l2360_236030


namespace NUMINAMATH_CALUDE_lyndees_friends_l2360_236015

theorem lyndees_friends (total_chicken : ℕ) (lyndee_ate : ℕ) (friend_ate : ℕ) 
  (h1 : total_chicken = 11)
  (h2 : lyndee_ate = 1)
  (h3 : friend_ate = 2)
  (h4 : total_chicken = lyndee_ate + friend_ate * (total_chicken - lyndee_ate) / friend_ate) :
  (total_chicken - lyndee_ate) / friend_ate = 5 := by
  sorry

end NUMINAMATH_CALUDE_lyndees_friends_l2360_236015


namespace NUMINAMATH_CALUDE_smallest_N_proof_l2360_236032

/-- The smallest natural number N such that N × 999 consists entirely of the digit seven in its decimal representation -/
def smallest_N : ℕ := 778556334111889667445223

/-- Predicate to check if a natural number consists entirely of the digit seven -/
def all_sevens (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

theorem smallest_N_proof :
  (smallest_N * 999).digits 10 = List.replicate 27 7 ∧
  ∀ m : ℕ, m < smallest_N → ¬(all_sevens (m * 999)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_proof_l2360_236032


namespace NUMINAMATH_CALUDE_altitude_equation_l2360_236078

/-- Given a triangle ABC with side equations:
    AB: 3x + 4y + 12 = 0
    BC: 4x - 3y + 16 = 0
    CA: 2x + y - 2 = 0
    The altitude from A to BC has the equation x - 2y + 4 = 0 -/
theorem altitude_equation (x y : ℝ) :
  (3 * x + 4 * y + 12 = 0) →  -- AB
  (4 * x - 3 * y + 16 = 0) →  -- BC
  (2 * x + y - 2 = 0) →       -- CA
  (x - 2 * y + 4 = 0)         -- Altitude from A to BC
:= by sorry

end NUMINAMATH_CALUDE_altitude_equation_l2360_236078


namespace NUMINAMATH_CALUDE_initial_apples_count_l2360_236088

def cafeteria_apples (apples_handed_out : ℕ) (apples_per_pie : ℕ) (pies_made : ℕ) : ℕ :=
  apples_handed_out + apples_per_pie * pies_made

theorem initial_apples_count :
  cafeteria_apples 41 5 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2360_236088


namespace NUMINAMATH_CALUDE_dice_sum_not_22_l2360_236065

theorem dice_sum_not_22 (a b c d e : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  1 ≤ e ∧ e ≤ 6 →
  a * b * c * d * e = 432 →
  a + b + c + d + e ≠ 22 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_not_22_l2360_236065


namespace NUMINAMATH_CALUDE_quadratic_one_root_l2360_236009

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem quadratic_one_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l2360_236009


namespace NUMINAMATH_CALUDE_total_cost_price_calculation_l2360_236067

theorem total_cost_price_calculation (sp1 sp2 sp3 : ℚ) (profit1 loss2 profit3 : ℚ) :
  sp1 = 600 ∧ profit1 = 25/100 ∧
  sp2 = 800 ∧ loss2 = 20/100 ∧
  sp3 = 1000 ∧ profit3 = 30/100 →
  ∃ (cp1 cp2 cp3 : ℚ),
    cp1 = sp1 / (1 + profit1) ∧
    cp2 = sp2 / (1 - loss2) ∧
    cp3 = sp3 / (1 + profit3) ∧
    cp1 + cp2 + cp3 = 2249.23 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_price_calculation_l2360_236067


namespace NUMINAMATH_CALUDE_dan_picked_nine_apples_l2360_236046

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The difference between Dan's and Benny's apple count -/
def difference : ℕ := 7

/-- The number of apples Dan picked -/
def dan_apples : ℕ := benny_apples + difference

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_apples_l2360_236046


namespace NUMINAMATH_CALUDE_relief_supplies_total_l2360_236018

/-- The total amount of relief supplies in tons -/
def total_supplies : ℝ := 644

/-- Team A's daily transport capacity in tons -/
def team_a_capacity : ℝ := 64.4

/-- The percentage by which Team A's capacity exceeds Team B's -/
def capacity_difference_percentage : ℝ := 75

/-- The additional amount Team A has transported when it reaches half the total supplies -/
def additional_transport : ℝ := 138

/-- Theorem stating the total amount of relief supplies -/
theorem relief_supplies_total : 
  ∃ (team_b_capacity : ℝ),
    team_a_capacity = team_b_capacity * (1 + capacity_difference_percentage / 100) ∧
    (total_supplies / 2) - (total_supplies / 2 - additional_transport) = 
      (team_a_capacity - team_b_capacity) * (total_supplies / (2 * team_a_capacity)) ∧
    total_supplies = 644 :=
by sorry

end NUMINAMATH_CALUDE_relief_supplies_total_l2360_236018


namespace NUMINAMATH_CALUDE_max_factors_theorem_l2360_236025

def is_valid_pair (b n : ℕ) : Prop :=
  0 < b ∧ b ≤ 20 ∧ 0 < n ∧ n ≤ 20 ∧ b ≠ n

def num_factors (m : ℕ) : ℕ := (Nat.factors m).length + 1

def max_factors : ℕ := 81

theorem max_factors_theorem :
  ∀ b n : ℕ, is_valid_pair b n →
    num_factors (b^n) ≤ max_factors :=
by sorry

end NUMINAMATH_CALUDE_max_factors_theorem_l2360_236025


namespace NUMINAMATH_CALUDE_m_value_proof_l2360_236079

theorem m_value_proof (m : ℝ) : 
  (m > 0) ∧ 
  (∀ x : ℝ, (x / (x - 1) < 0 → 0 < x ∧ x < m)) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < m ∧ x / (x - 1) ≥ 0) →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_m_value_proof_l2360_236079


namespace NUMINAMATH_CALUDE_max_value_expression_l2360_236006

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8 ≤ a ∧ a ≤ 8) 
  (hb : -8 ≤ b ∧ b ≤ 8) 
  (hc : -8 ≤ c ∧ c ≤ 8) 
  (hd : -8 ≤ d ∧ d ≤ 8) : 
  (∀ x y z w, -8 ≤ x ∧ x ≤ 8 → -8 ≤ y ∧ y ≤ 8 → -8 ≤ z ∧ z ≤ 8 → -8 ≤ w ∧ w ≤ 8 → 
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 272) ∧ 
  (∃ x y z w, -8 ≤ x ∧ x ≤ 8 ∧ -8 ≤ y ∧ y ≤ 8 ∧ -8 ≤ z ∧ z ≤ 8 ∧ -8 ≤ w ∧ w ≤ 8 ∧
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 272) := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2360_236006


namespace NUMINAMATH_CALUDE_bus_rental_solution_l2360_236059

/-- Represents the bus rental problem for a school study tour. -/
structure BusRentalProblem where
  capacityA : Nat  -- Capacity of A type bus
  capacityB : Nat  -- Capacity of B type bus
  extraPeople : Nat  -- People without seats in original plan
  fewerBusesB : Nat  -- Number of fewer B type buses needed
  totalBuses : Nat  -- Total number of buses to be rented
  maxTypeB : Nat  -- Maximum number of B type buses
  feeA : Nat  -- Rental fee for A type bus
  feeB : Nat  -- Rental fee for B type bus

/-- Represents a bus rental scheme. -/
structure RentalScheme where
  numA : Nat  -- Number of A type buses
  numB : Nat  -- Number of B type buses

/-- The main theorem about the bus rental problem. -/
theorem bus_rental_solution (p : BusRentalProblem)
  (h1 : p.capacityA = 45)
  (h2 : p.capacityB = 60)
  (h3 : p.extraPeople = 30)
  (h4 : p.fewerBusesB = 6)
  (h5 : p.totalBuses = 25)
  (h6 : p.maxTypeB = 7)
  (h7 : p.feeA = 220)
  (h8 : p.feeB = 300) :
  ∃ (originalA totalPeople : Nat) (schemes : List RentalScheme) (bestScheme : RentalScheme),
    originalA = 26 ∧
    totalPeople = 1200 ∧
    schemes = [⟨20, 5⟩, ⟨19, 6⟩, ⟨18, 7⟩] ∧
    bestScheme = ⟨20, 5⟩ ∧
    (∀ scheme ∈ schemes, 
      scheme.numA + scheme.numB = p.totalBuses ∧
      scheme.numB ≤ p.maxTypeB ∧
      scheme.numA * p.capacityA + scheme.numB * p.capacityB ≥ totalPeople) ∧
    (∀ scheme ∈ schemes,
      scheme.numA * p.feeA + scheme.numB * p.feeB ≥ 
      bestScheme.numA * p.feeA + bestScheme.numB * p.feeB) := by
  sorry

end NUMINAMATH_CALUDE_bus_rental_solution_l2360_236059


namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l2360_236000

-- System of equations
theorem system_of_equations_solution (x y : ℝ) :
  (3 * x + 4 * y = 2) ∧ (2 * x - y = 5) ↔ (x = 2 ∧ y = -1) := by sorry

-- System of inequalities
theorem system_of_inequalities_solution (x : ℝ) :
  (x - 3 * (x - 1) < 7) ∧ (x - 2 ≤ (2 * x - 3) / 3) ↔ (-2 < x ∧ x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l2360_236000


namespace NUMINAMATH_CALUDE_quadratic_convergence_l2360_236010

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the property that |f(x)| ≤ 1/2 for all x in [3, 5]
def bounded_on_interval (p q : ℝ) : Prop :=
  ∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → |f p q x| ≤ 1/2

-- Define the repeated application of f
def f_iterate (p q : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f p q (f_iterate p q n x)

-- State the theorem
theorem quadratic_convergence (p q : ℝ) (h : bounded_on_interval p q) :
  f_iterate p q 2017 ((7 + Real.sqrt 15) / 2) = (7 - Real.sqrt 15) / 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_convergence_l2360_236010


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2360_236029

theorem inequality_equivalence (x : ℝ) : (x + 2) / (x - 1) > 3 ↔ 1 < x ∧ x < 5/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2360_236029


namespace NUMINAMATH_CALUDE_change_difference_is_thirty_percent_l2360_236013

-- Define the initial and final percentages
def initial_yes : ℚ := 40 / 100
def initial_no : ℚ := 30 / 100
def initial_undecided : ℚ := 30 / 100
def final_yes : ℚ := 60 / 100
def final_no : ℚ := 20 / 100
def final_undecided : ℚ := 20 / 100

-- Define the minimum and maximum change percentages
def min_change : ℚ := max 0 (final_yes - initial_yes)
def max_change : ℚ := min 1 (initial_no + initial_undecided + abs (final_yes - initial_yes))

-- Theorem statement
theorem change_difference_is_thirty_percent :
  max_change - min_change = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_change_difference_is_thirty_percent_l2360_236013


namespace NUMINAMATH_CALUDE_extracurricular_materials_selection_l2360_236076

theorem extracurricular_materials_selection (n : Nat) (k : Nat) (m : Nat) : 
  n = 6 → k = 2 → m = 1 → 
  (Nat.choose n m) * (m * (n - m) * (n - m - 1)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_extracurricular_materials_selection_l2360_236076


namespace NUMINAMATH_CALUDE_fold_point_sum_l2360_236033

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Determines if a line folds one point onto another -/
def folds (l : Line) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  midpoint.y = l.slope * midpoint.x + l.intercept

/-- The main theorem -/
theorem fold_point_sum (l : Line) :
  folds l ⟨1, 3⟩ ⟨5, 1⟩ →
  folds l ⟨8, 4⟩ ⟨m, n⟩ →
  m + n = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fold_point_sum_l2360_236033


namespace NUMINAMATH_CALUDE_m_range_l2360_236024

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x^2 - 4| + x^2 + m*x

/-- The condition that f has two distinct zero points in (0, 3) -/
def has_two_distinct_zeros (m : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ x < 3 ∧ 0 < y ∧ y < 3 ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0

/-- The theorem stating the range of m -/
theorem m_range (m : ℝ) :
  has_two_distinct_zeros m → -14/3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2360_236024


namespace NUMINAMATH_CALUDE_solve_for_n_l2360_236072

def first_seven_multiples_of_four : List ℕ := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem solve_for_n (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 8 := by sorry

end NUMINAMATH_CALUDE_solve_for_n_l2360_236072


namespace NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2360_236047

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  -- Add any necessary fields

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

theorem pyramid_cross_section_distance
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h : ℝ) :
  cs1.area = 125 * Real.sqrt 3 →
  cs2.area = 500 * Real.sqrt 3 →
  cs2.distance_from_apex - cs1.distance_from_apex = 10 →
  cs2.distance_from_apex = h →
  h = 20 := by
  sorry

#check pyramid_cross_section_distance

end NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2360_236047


namespace NUMINAMATH_CALUDE_profit_calculation_l2360_236090

/-- The number of pencils bought by the store owner -/
def total_pencils : ℕ := 2000

/-- The cost price of each pencil in dollars -/
def cost_price : ℚ := 15 / 100

/-- The selling price of each pencil in dollars -/
def selling_price : ℚ := 30 / 100

/-- The desired profit in dollars -/
def desired_profit : ℚ := 150

/-- The number of pencils that must be sold to make the desired profit -/
def pencils_to_sell : ℕ := 1500

theorem profit_calculation :
  (pencils_to_sell : ℚ) * selling_price - (total_pencils : ℚ) * cost_price = desired_profit :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l2360_236090


namespace NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l2360_236045

theorem product_one_sum_greater_than_reciprocals 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) : 
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ 
  (a < 1 ∧ b > 1 ∧ c < 1) ∨ 
  (a < 1 ∧ b < 1 ∧ c > 1) :=
sorry

end NUMINAMATH_CALUDE_product_one_sum_greater_than_reciprocals_l2360_236045


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2360_236085

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2360_236085


namespace NUMINAMATH_CALUDE_harriets_age_l2360_236004

/-- Given information about Peter and Harriet's ages, prove Harriet's current age -/
theorem harriets_age (peter_mother_age : ℕ) (peter_age : ℕ) (harriet_age : ℕ) : 
  peter_mother_age = 60 →
  peter_age = peter_mother_age / 2 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  harriet_age = 13 := by sorry

end NUMINAMATH_CALUDE_harriets_age_l2360_236004


namespace NUMINAMATH_CALUDE_cake_mix_buyers_l2360_236021

theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) 
  (h1 : total = 100)
  (h2 : muffin = 40)
  (h3 : both = 19)
  (h4 : neither_prob = 29/100) :
  ∃ cake : ℕ, cake = 50 ∧ 
    cake + muffin - both = total - (neither_prob * total).num := by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_l2360_236021


namespace NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l2360_236066

/-- The area of a regular hexagon with the same perimeter as a square of area 16 -/
theorem hexagon_area_equal_perimeter (square_area : ℝ) (square_side : ℝ) (hex_side : ℝ) :
  square_area = 16 →
  square_side^2 = square_area →
  4 * square_side = 6 * hex_side →
  (3 * hex_side^2 * Real.sqrt 3) / 2 = (32 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_equal_perimeter_l2360_236066


namespace NUMINAMATH_CALUDE_cone_surface_area_l2360_236061

/-- Given a cone with base radius 3 and lateral surface that unfolds into a sector
    with central angle 2π/3, its surface area is 36π. -/
theorem cone_surface_area (r : ℝ) (θ : ℝ) (S : ℝ) : 
  r = 3 → 
  θ = 2 * Real.pi / 3 →
  S = r * r * Real.pi + r * (r * θ) →
  S = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l2360_236061


namespace NUMINAMATH_CALUDE_gain_amount_calculation_l2360_236028

theorem gain_amount_calculation (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 110)
  (h2 : gain_percentage = 0.10) : 
  let cost_price := selling_price / (1 + gain_percentage)
  selling_price - cost_price = 10 := by
sorry

end NUMINAMATH_CALUDE_gain_amount_calculation_l2360_236028


namespace NUMINAMATH_CALUDE_hana_stamp_collection_l2360_236011

/-- The value of Hana's entire stamp collection -/
def total_value : ℚ := 980 / 27

/-- The fraction of the collection Hana sold -/
def sold_fraction : ℚ := 4/7 + 1/3 * (3/7) + 1/5 * (2/7)

/-- The amount Hana earned from her sales -/
def earned_amount : ℚ := 28

theorem hana_stamp_collection :
  sold_fraction * total_value = earned_amount :=
sorry

end NUMINAMATH_CALUDE_hana_stamp_collection_l2360_236011


namespace NUMINAMATH_CALUDE_band_sections_sum_l2360_236048

theorem band_sections_sum (total : ℕ) (trumpet_frac trombone_frac clarinet_frac flute_frac : ℚ) : 
  total = 500 →
  trumpet_frac = 1/2 →
  trombone_frac = 3/25 →
  clarinet_frac = 23/100 →
  flute_frac = 2/25 →
  ⌊total * trumpet_frac⌋ + ⌊total * trombone_frac⌋ + ⌊total * clarinet_frac⌋ + ⌊total * flute_frac⌋ = 465 :=
by sorry

end NUMINAMATH_CALUDE_band_sections_sum_l2360_236048


namespace NUMINAMATH_CALUDE_k_value_l2360_236062

theorem k_value (a b k : ℝ) 
  (h1 : 4^a = k) 
  (h2 : 9^b = k) 
  (h3 : 1/a + 1/b = 2) : k = 6 := by
sorry

end NUMINAMATH_CALUDE_k_value_l2360_236062


namespace NUMINAMATH_CALUDE_valid_sequence_count_l2360_236056

/-- Represents a sequence of coin tosses -/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a specific subsequence in a coin sequence -/
def countSubsequence (seq : CoinSequence) (subseq : CoinSequence) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions -/
def isValidSequence (seq : CoinSequence) : Prop :=
  (countSubsequence seq [true, true] = 3) ∧
  (countSubsequence seq [true, false] = 4) ∧
  (countSubsequence seq [false, true] = 5) ∧
  (countSubsequence seq [false, false] = 6)

/-- The number of valid coin sequences -/
def validSequenceCount : Nat :=
  sorry

theorem valid_sequence_count :
  validSequenceCount = 16170 :=
sorry

end NUMINAMATH_CALUDE_valid_sequence_count_l2360_236056


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2360_236098

/-- The line 4x + 3y + k = 0 is tangent to the parabola y^2 = 16x if and only if k = 9 -/
theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, 4 * x + 3 * y + k = 0 → y^2 = 16 * x) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2360_236098


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2360_236071

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2360_236071


namespace NUMINAMATH_CALUDE_magic_sum_values_l2360_236016

/-- Represents a triangle configuration with 6 numbers -/
structure TriangleConfig where
  vertices : Fin 6 → Nat
  distinct : ∀ i j, i ≠ j → vertices i ≠ vertices j
  range : ∀ i, vertices i ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)

/-- The sum of numbers on one side of the triangle -/
def sideSum (config : TriangleConfig) : Nat :=
  config.vertices 0 + config.vertices 1 + config.vertices 2

/-- All sides have the same sum -/
def validConfig (config : TriangleConfig) : Prop :=
  sideSum config = config.vertices 0 + config.vertices 3 + config.vertices 5 ∧
  sideSum config = config.vertices 2 + config.vertices 4 + config.vertices 5

theorem magic_sum_values :
  ∃ (config : TriangleConfig), validConfig config ∧
  sideSum config ∈ ({9, 10, 11, 12} : Set Nat) ∧
  ∀ (otherConfig : TriangleConfig),
    validConfig otherConfig →
    sideSum otherConfig ∈ ({9, 10, 11, 12} : Set Nat) := by
  sorry

end NUMINAMATH_CALUDE_magic_sum_values_l2360_236016


namespace NUMINAMATH_CALUDE_line_not_in_first_quadrant_l2360_236034

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a line passes through the first quadrant -/
def passesFirstQuadrant (l : Line) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ l.a * x + l.b * y + l.c = 0

theorem line_not_in_first_quadrant (l : Line) 
  (h1 : ¬passesFirstQuadrant l) 
  (h2 : l.a * l.b > 0) : 
  l.a * l.c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_first_quadrant_l2360_236034


namespace NUMINAMATH_CALUDE_lindsay_dolls_l2360_236055

theorem lindsay_dolls (blonde : ℕ) (brown black red : ℕ) : 
  blonde = 6 →
  brown = 3 * blonde →
  black = brown / 2 →
  red = 2 * black →
  (black + brown + red) - blonde = 39 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_dolls_l2360_236055


namespace NUMINAMATH_CALUDE_equation_solutions_l2360_236020

-- Define the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos (2 * x)

-- State the theorem
theorem equation_solutions :
  ∃! (solutions : Finset ℝ), solutions.card = 60 ∧ ∀ x, x ∈ solutions ↔ equation x :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2360_236020


namespace NUMINAMATH_CALUDE_smallest_perfect_square_with_perfect_square_factors_l2360_236022

/-- A function that returns the number of positive integer factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square_num (n : ℕ) : Prop := sorry

theorem smallest_perfect_square_with_perfect_square_factors : 
  ∀ n : ℕ, n > 1 → is_perfect_square n → is_perfect_square_num (num_factors n) → n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_with_perfect_square_factors_l2360_236022


namespace NUMINAMATH_CALUDE_actual_distance_travelled_l2360_236041

theorem actual_distance_travelled (speed1 speed2 : ℝ) (extra_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : extra_distance = 20)
  (h4 : ∀ d : ℝ, d / speed1 = (d + extra_distance) / speed2) :
  ∃ d : ℝ, d = 50 ∧ d / speed1 = (d + extra_distance) / speed2 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_travelled_l2360_236041


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2360_236086

def is_composite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = n

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → sum_of_two_composites n) ∧
  ¬sum_of_two_composites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l2360_236086


namespace NUMINAMATH_CALUDE_cell_division_problem_l2360_236035

/-- The number of cells after a given time, starting with one cell -/
def num_cells (division_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  2^(elapsed_time / division_time)

/-- The time between cell divisions in minutes -/
def division_time : ℕ := 30

/-- The total elapsed time in minutes -/
def total_time : ℕ := 4 * 60 + 30

theorem cell_division_problem :
  num_cells division_time total_time = 512 := by
  sorry

end NUMINAMATH_CALUDE_cell_division_problem_l2360_236035


namespace NUMINAMATH_CALUDE_cab_driver_fifth_day_income_verify_cab_driver_income_l2360_236027

/-- Calculates the cab driver's income on the fifth day given the income for the first four days and the average income for five days. -/
theorem cab_driver_fifth_day_income 
  (income_day1 income_day2 income_day3 income_day4 : ℚ) 
  (average_income : ℚ) : ℚ :=
  let total_income := 5 * average_income
  let sum_four_days := income_day1 + income_day2 + income_day3 + income_day4
  total_income - sum_four_days

/-- Verifies that the calculated fifth day income is correct given the specific values from the problem. -/
theorem verify_cab_driver_income : 
  cab_driver_fifth_day_income 300 150 750 400 420 = 500 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_fifth_day_income_verify_cab_driver_income_l2360_236027


namespace NUMINAMATH_CALUDE_luke_stars_made_l2360_236075

-- Define the given conditions
def stars_per_jar : ℕ := 85
def bottles_to_fill : ℕ := 4
def additional_stars_needed : ℕ := 307

-- Define the theorem
theorem luke_stars_made : 
  (stars_per_jar * bottles_to_fill) - additional_stars_needed = 33 := by
  sorry

end NUMINAMATH_CALUDE_luke_stars_made_l2360_236075


namespace NUMINAMATH_CALUDE_janet_lives_gained_l2360_236097

theorem janet_lives_gained (initial_lives : ℕ) (lives_lost : ℕ) (final_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : lives_lost = 23)
  (h3 : final_lives = 70) :
  final_lives - (initial_lives - lives_lost) = 46 := by
  sorry

end NUMINAMATH_CALUDE_janet_lives_gained_l2360_236097


namespace NUMINAMATH_CALUDE_num_true_props_l2360_236077

-- Define the propositions as boolean variables
def prop1 : Bool := true  -- All lateral edges of a regular pyramid are equal
def prop2 : Bool := false -- The lateral faces of a right prism are all congruent rectangles
def prop3 : Bool := true  -- The generatrix of a cylinder is perpendicular to the base
def prop4 : Bool := true  -- The section obtained by cutting a cone with a plane passing through the axis of rotation is always a congruent isosceles triangle

-- Define a function to count true propositions
def countTrueProps (p1 p2 p3 p4 : Bool) : Nat :=
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0)

-- Theorem stating that the number of true propositions is 3
theorem num_true_props : countTrueProps prop1 prop2 prop3 prop4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_true_props_l2360_236077


namespace NUMINAMATH_CALUDE_four_digit_reverse_pairs_l2360_236096

def reverse_digits (n : ℕ) : ℕ :=
  let digits := String.toList (toString n)
  String.toNat! (String.mk (List.reverse digits))

def ends_with_three_zeros (n : ℕ) : Prop :=
  n % 1000 = 0

theorem four_digit_reverse_pairs : 
  ∀ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    b = reverse_digits a ∧
    ends_with_three_zeros (a * b) →
    ((a = 5216 ∧ b = 6125) ∨
     (a = 5736 ∧ b = 6375) ∨
     (a = 5264 ∧ b = 4625) ∨
     (a = 5784 ∧ b = 4875))
  := by sorry

end NUMINAMATH_CALUDE_four_digit_reverse_pairs_l2360_236096


namespace NUMINAMATH_CALUDE_disinfectant_problem_l2360_236087

/-- Represents the price and volume of a disinfectant brand -/
structure DisinfectantBrand where
  price : ℝ
  volume : ℝ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  brand1_bottles : ℕ
  brand2_bottles : ℕ

/-- Calculates the total cost of a purchase plan -/
def totalCost (brand1 brand2 : DisinfectantBrand) (plan : PurchasePlan) : ℝ :=
  brand1.price * plan.brand1_bottles + brand2.price * plan.brand2_bottles

/-- Calculates the total volume of a purchase plan -/
def totalVolume (brand1 brand2 : DisinfectantBrand) (plan : PurchasePlan) : ℝ :=
  brand1.volume * plan.brand1_bottles + brand2.volume * plan.brand2_bottles

/-- Theorem stating the properties of the disinfectant purchase problem -/
theorem disinfectant_problem (brand1 brand2 : DisinfectantBrand) 
  (h1 : brand1.volume = 200)
  (h2 : brand2.volume = 500)
  (h3 : totalCost brand1 brand2 { brand1_bottles := 3, brand2_bottles := 2 } = 80)
  (h4 : totalCost brand1 brand2 { brand1_bottles := 1, brand2_bottles := 4 } = 110)
  (h5 : ∃ (plan : PurchasePlan), totalVolume brand1 brand2 plan = 4000 ∧ 
        plan.brand1_bottles > 0 ∧ plan.brand2_bottles > 0)
  (h6 : ∃ (plan : PurchasePlan), totalCost brand1 brand2 plan = 2500)
  (h7 : (2500 / (1000 * 10 : ℝ)) * (brand1.volume * brand1.price + brand2.volume * brand2.price) = 5000) :
  brand1.price = 10 ∧ brand2.price = 25 ∧ 
  (2500 / (1000 * 10 : ℝ)) * (brand1.volume * brand1.price + brand2.volume * brand2.price) / 1000 = 5 := by
  sorry


end NUMINAMATH_CALUDE_disinfectant_problem_l2360_236087


namespace NUMINAMATH_CALUDE_trapezoid_square_area_equality_l2360_236012

/-- Given a trapezoid with upper side 15 cm, lower side 9 cm, and height 12 cm,
    the side length of a square with the same area as the trapezoid is 12 cm. -/
theorem trapezoid_square_area_equality (upper_side lower_side height : ℝ) 
    (h1 : upper_side = 15)
    (h2 : lower_side = 9)
    (h3 : height = 12) :
    ∃ (square_side : ℝ), 
      (1/2 * (upper_side + lower_side) * height = square_side^2) ∧ 
      square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_square_area_equality_l2360_236012


namespace NUMINAMATH_CALUDE_carlos_cookie_count_l2360_236081

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Rectangle
  | Square

/-- Represents a cookie with its shape and area -/
structure Cookie where
  shape : CookieShape
  area : ℝ

/-- Represents a batch of cookies -/
structure CookieBatch where
  shape : CookieShape
  totalArea : ℝ
  count : ℕ

/-- The theorem to be proved -/
theorem carlos_cookie_count 
  (anne_batch : CookieBatch)
  (carlos_batch : CookieBatch)
  (h1 : anne_batch.shape = CookieShape.Rectangle)
  (h2 : carlos_batch.shape = CookieShape.Square)
  (h3 : anne_batch.totalArea = 180)
  (h4 : anne_batch.count = 15)
  (h5 : anne_batch.totalArea = carlos_batch.totalArea) :
  carlos_batch.count = 20 := by
  sorry


end NUMINAMATH_CALUDE_carlos_cookie_count_l2360_236081


namespace NUMINAMATH_CALUDE_no_preimage_range_l2360_236093

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := Set.univ

def f (x : ℝ) : ℝ := -x^2 + 2*x - 1

theorem no_preimage_range :
  {p : ℝ | p ∈ B ∧ ∀ x ∈ A, f x ≠ p} = {p : ℝ | p ≥ -1} := by
  sorry

end NUMINAMATH_CALUDE_no_preimage_range_l2360_236093


namespace NUMINAMATH_CALUDE_cement_theft_proof_l2360_236040

/-- Represents the weight of cement bags in kilograms -/
structure BagWeight where
  small : Nat
  large : Nat

/-- Represents the number of cement bags -/
structure BagCount where
  small : Nat
  large : Nat

/-- Calculates the total weight of cement given bag weights and counts -/
def totalWeight (w : BagWeight) (c : BagCount) : Nat :=
  w.small * c.small + w.large * c.large

/-- Represents the manager's assumption of bag weight -/
def managerAssumedWeight : Nat := 25

theorem cement_theft_proof (w : BagWeight) (c : BagCount) 
  (h1 : w.small = 25)
  (h2 : w.large = 40)
  (h3 : c.small = 2 * c.large)
  (h4 : totalWeight w c - w.large * 60 = managerAssumedWeight * (c.small + c.large)) :
  totalWeight w c - w.large * 60 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_cement_theft_proof_l2360_236040


namespace NUMINAMATH_CALUDE_gcf_of_4140_and_9920_l2360_236082

theorem gcf_of_4140_and_9920 : Nat.gcd 4140 9920 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_4140_and_9920_l2360_236082


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l2360_236099

theorem systematic_sampling_removal (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1252) 
  (h2 : sample_size = 50) : 
  ∃ (removed : ℕ), removed = total_students % sample_size ∧ removed = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l2360_236099


namespace NUMINAMATH_CALUDE_prob_sum_four_twice_l2360_236042

/-- A die with 3 sides --/
def ThreeSidedDie : Type := Fin 3

/-- The sum of two dice rolls --/
def diceSum (d1 d2 : ThreeSidedDie) : Nat :=
  d1.val + d2.val + 2

/-- The probability of rolling a sum of 4 with two 3-sided dice --/
def probSumFour : ℚ :=
  3 / 9

/-- The probability of rolling a sum of 4 twice in a row with two 3-sided dice --/
theorem prob_sum_four_twice : probSumFour * probSumFour = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_four_twice_l2360_236042


namespace NUMINAMATH_CALUDE_remainder_sum_l2360_236002

theorem remainder_sum (x : ℤ) (h : x % 6 = 3) : 
  (x^2 % 30) + (x^3 % 11) = 14 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l2360_236002


namespace NUMINAMATH_CALUDE_bobs_grade_l2360_236052

theorem bobs_grade (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = jason_grade / 2 →
  bob_grade = 35 := by
sorry

end NUMINAMATH_CALUDE_bobs_grade_l2360_236052


namespace NUMINAMATH_CALUDE_parallelogram_base_l2360_236083

theorem parallelogram_base (area height : ℝ) (h1 : area = 375) (h2 : height = 15) :
  area / height = 25 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2360_236083


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_five_fourths_l2360_236043

theorem trigonometric_expression_equals_five_fourths :
  Real.sqrt 2 * Real.cos (π / 4) - Real.sin (π / 3) ^ 2 + Real.tan (π / 4) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_five_fourths_l2360_236043


namespace NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l2360_236069

/-- The width of the rectangle. -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle. -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle. -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the square that can be covered exactly by the rectangles. -/
def square_side : ℕ := rectangle_area

/-- The area of the square. -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square. -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem min_rectangles_to_cover_square : 
  num_rectangles = 12 ∧ 
  square_area % rectangle_area = 0 ∧
  ∀ n : ℕ, n < square_side → n * n % rectangle_area ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l2360_236069


namespace NUMINAMATH_CALUDE_fifth_degree_polynomial_existence_l2360_236005

theorem fifth_degree_polynomial_existence : ∃ (P : ℝ → ℝ),
  (∀ x : ℝ, P x = 0 → x < 0) ∧
  (∀ x : ℝ, (deriv P) x = 0 → x > 0) ∧
  (∃ x : ℝ, P x = 0 ∧ ∀ y : ℝ, y ≠ x → P y ≠ 0) ∧
  (∃ x : ℝ, (deriv P) x = 0 ∧ ∀ y : ℝ, y ≠ x → (deriv P) y ≠ 0) ∧
  (∃ a b c d e f : ℝ, ∀ x : ℝ, P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) :=
by sorry

end NUMINAMATH_CALUDE_fifth_degree_polynomial_existence_l2360_236005


namespace NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_fourth_l2360_236023

theorem power_of_fraction_five_sevenths_fourth : (5 / 7 : ℚ) ^ 4 = 625 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_fourth_l2360_236023


namespace NUMINAMATH_CALUDE_power_five_2023_mod_11_l2360_236084

theorem power_five_2023_mod_11 : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_five_2023_mod_11_l2360_236084


namespace NUMINAMATH_CALUDE_jim_total_cars_l2360_236094

/-- The number of model cars Jim has -/
structure ModelCars where
  buicks : ℕ
  fords : ℕ
  chevys : ℕ

/-- Jim's collection of model cars satisfying the given conditions -/
def jim_collection : ModelCars :=
  { buicks := 220,
    fords := 55,
    chevys := 26 }

/-- Theorem stating the total number of model cars Jim has -/
theorem jim_total_cars :
  jim_collection.buicks = 220 ∧
  jim_collection.buicks = 4 * jim_collection.fords ∧
  jim_collection.fords = 2 * jim_collection.chevys + 3 →
  jim_collection.buicks + jim_collection.fords + jim_collection.chevys = 301 := by
  sorry

#eval jim_collection.buicks + jim_collection.fords + jim_collection.chevys

end NUMINAMATH_CALUDE_jim_total_cars_l2360_236094


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l2360_236049

theorem max_value_of_sum_of_square_roots (a b c : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 8 → 
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 9 ∧ 
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
    Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l2360_236049


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2360_236026

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2360_236026


namespace NUMINAMATH_CALUDE_sum_divisible_by_three_combinations_l2360_236031

/-- The number of integers from 1 to 300 that give remainder 0, 1, or 2 when divided by 3 -/
def count_mod_3 : ℕ := 100

/-- The total number of ways to select 3 numbers from 1 to 300 such that their sum is divisible by 3 -/
def total_combinations : ℕ := 1485100

/-- The number of ways to choose 3 elements from a set of size n -/
def choose (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

theorem sum_divisible_by_three_combinations :
  3 * choose count_mod_3 + count_mod_3^3 = total_combinations :=
sorry

end NUMINAMATH_CALUDE_sum_divisible_by_three_combinations_l2360_236031


namespace NUMINAMATH_CALUDE_number_of_boys_l2360_236007

theorem number_of_boys (total_pupils : ℕ) (number_of_girls : ℕ) 
  (h1 : total_pupils = 929) 
  (h2 : number_of_girls = 542) : 
  total_pupils - number_of_girls = 387 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l2360_236007


namespace NUMINAMATH_CALUDE_fraction_inequality_l2360_236036

theorem fraction_inequality (a b c x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) : 
  min (x / (a*b + a*c)) (y / (a*c + b*c)) < (x + y) / (a*b + b*c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2360_236036


namespace NUMINAMATH_CALUDE_stork_bird_difference_l2360_236089

/-- Given initial birds, storks, and additional birds, calculate the difference between storks and total birds -/
theorem stork_bird_difference (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 → storks = 6 → additional_birds = 3 →
  storks - (initial_birds + additional_birds) = 1 := by
  sorry

end NUMINAMATH_CALUDE_stork_bird_difference_l2360_236089


namespace NUMINAMATH_CALUDE_octal_to_decimal_l2360_236038

theorem octal_to_decimal : (3 * 8^2 + 6 * 8^1 + 7 * 8^0) = 247 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l2360_236038


namespace NUMINAMATH_CALUDE_rectangle_area_18_l2360_236068

/-- A rectangle with base twice the height and area equal to perimeter has area 18 -/
theorem rectangle_area_18 (h : ℝ) (b : ℝ) (area : ℝ) (perimeter : ℝ) : 
  b = 2 * h →                        -- base is twice the height
  area = b * h →                     -- area formula
  perimeter = 2 * (b + h) →          -- perimeter formula
  area = perimeter →                 -- area is numerically equal to perimeter
  area = 18 :=                       -- prove that area is 18
by sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l2360_236068


namespace NUMINAMATH_CALUDE_value_of_b_is_two_l2360_236037

def f (x : ℝ) := x^2 - 2*x + 2

theorem value_of_b_is_two :
  ∃ b : ℝ, b > 1 ∧
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_value_of_b_is_two_l2360_236037


namespace NUMINAMATH_CALUDE_largest_prefix_for_two_digit_quotient_l2360_236001

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem largest_prefix_for_two_digit_quotient :
  ∀ n : ℕ, n ≤ 9 →
    (is_two_digit ((n * 100 + 72) / 6) ↔ n ≤ 5) ∧
    (∀ m : ℕ, m ≤ 9 ∧ m > 5 → ¬(is_two_digit ((m * 100 + 72) / 6))) :=
by sorry

end NUMINAMATH_CALUDE_largest_prefix_for_two_digit_quotient_l2360_236001


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2360_236060

theorem complex_equation_sum (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + 2 * i) / i = b + i →
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2360_236060


namespace NUMINAMATH_CALUDE_optimal_partition_l2360_236051

def minimizeAbsoluteErrors (diameters : List ℝ) : 
  ℝ × ℝ × ℝ := sorry

theorem optimal_partition (diameters : List ℝ) 
  (h1 : diameters.length = 120) 
  (h2 : List.Sorted (· ≤ ·) diameters) :
  let (d, a, b) := minimizeAbsoluteErrors diameters
  d = (diameters.get! 59 + diameters.get! 60) / 2 ∧
  a = (diameters.take 60).sum / 60 ∧
  b = (diameters.drop 60).sum / 60 :=
sorry

end NUMINAMATH_CALUDE_optimal_partition_l2360_236051


namespace NUMINAMATH_CALUDE_right_triangle_altitude_l2360_236014

theorem right_triangle_altitude (A h c : ℝ) : 
  A > 0 → c > 0 → h > 0 →
  A = 540 → c = 36 →
  A = (1/2) * c * h →
  h = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_l2360_236014


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2360_236008

def A (m : ℝ) : Set ℝ := {-1, 2, 2*m-1}
def B (m : ℝ) : Set ℝ := {2, m^2}

theorem subset_implies_m_equals_one (m : ℝ) :
  B m ⊆ A m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2360_236008


namespace NUMINAMATH_CALUDE_line_plane_intersection_l2360_236017

/-- Given a plane α and two intersecting planes that form a line l, 
    prove the direction vector of l and the sine of the angle between l and α. -/
theorem line_plane_intersection (x y z : ℝ) : 
  let α : ℝ → ℝ → ℝ → Prop := λ x y z => x + 2*y - 2*z + 1 = 0
  let plane1 : ℝ → ℝ → ℝ → Prop := λ x y z => x - y + 3 = 0
  let plane2 : ℝ → ℝ → ℝ → Prop := λ x y z => x - 2*z - 1 = 0
  let l : Set (ℝ × ℝ × ℝ) := {p | plane1 p.1 p.2.1 p.2.2 ∧ plane2 p.1 p.2.1 p.2.2}
  let direction_vector : ℝ × ℝ × ℝ := (2, 2, 1)
  let normal_vector : ℝ × ℝ × ℝ := (1, 2, -2)
  let angle_sine : ℝ := 4/9
  (∀ p ∈ l, ∃ t : ℝ, p = (t * direction_vector.1, t * direction_vector.2.1, t * direction_vector.2.2)) ∧
  (|normal_vector.1 * direction_vector.1 + normal_vector.2.1 * direction_vector.2.1 + normal_vector.2.2 * direction_vector.2.2| / 
   (Real.sqrt (normal_vector.1^2 + normal_vector.2.1^2 + normal_vector.2.2^2) * 
    Real.sqrt (direction_vector.1^2 + direction_vector.2.1^2 + direction_vector.2.2^2)) = angle_sine) :=
by sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l2360_236017


namespace NUMINAMATH_CALUDE_star_three_six_eq_seven_l2360_236073

/-- The ☆ operation on rational numbers -/
def star (a : ℚ) (x y : ℚ) : ℚ := a^2 * x + a * y + 1

/-- Theorem: If 1 ☆ 2 = 3, then 3 ☆ 6 = 7 -/
theorem star_three_six_eq_seven (a : ℚ) (h : star a 1 2 = 3) : star a 3 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_star_three_six_eq_seven_l2360_236073


namespace NUMINAMATH_CALUDE_multiples_of_seven_between_20_and_150_l2360_236070

theorem multiples_of_seven_between_20_and_150 : 
  (Finset.filter (fun n => n % 7 = 0 ∧ n > 20 ∧ n < 150) (Finset.range 150)).card = 19 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_between_20_and_150_l2360_236070


namespace NUMINAMATH_CALUDE_meshed_gears_speed_proportion_l2360_236039

/-- Represents a gear with number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Proves that for four meshed gears, their angular speeds are proportional to yzw : xzw : xyw : xyz -/
theorem meshed_gears_speed_proportion
  (A B C D : Gear)
  (h_mesh : A.teeth * A.speed = B.teeth * B.speed ∧
            B.teeth * B.speed = C.teeth * C.speed ∧
            C.teeth * C.speed = D.teeth * D.speed) :
  ∃ (k : ℝ), k ≠ 0 ∧
    A.speed = k * (B.teeth * C.teeth * D.teeth) ∧
    B.speed = k * (A.teeth * C.teeth * D.teeth) ∧
    C.speed = k * (A.teeth * B.teeth * D.teeth) ∧
    D.speed = k * (A.teeth * B.teeth * C.teeth) :=
sorry

end NUMINAMATH_CALUDE_meshed_gears_speed_proportion_l2360_236039


namespace NUMINAMATH_CALUDE_wire_cutting_l2360_236053

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (longer_piece : ℝ) : 
  total_length = 30 →
  difference = 2 →
  longer_piece = total_length / 2 + difference / 2 →
  longer_piece = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2360_236053


namespace NUMINAMATH_CALUDE_james_yearly_pages_l2360_236074

/-- Calculates the number of pages James writes in a year -/
def pages_per_year (pages_per_letter : ℕ) (num_friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * num_friends * times_per_week * weeks_per_year

/-- Proves that James writes 624 pages in a year -/
theorem james_yearly_pages :
  pages_per_year 3 2 2 52 = 624 := by
  sorry

end NUMINAMATH_CALUDE_james_yearly_pages_l2360_236074


namespace NUMINAMATH_CALUDE_x_fifth_plus_inverse_l2360_236057

theorem x_fifth_plus_inverse (x : ℝ) (h_pos : x > 0) (h_eq : x^2 + 1/x^2 = 7) :
  x^5 + 1/x^5 = 123 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_plus_inverse_l2360_236057


namespace NUMINAMATH_CALUDE_differential_savings_example_l2360_236054

/-- Calculates the differential savings when tax rate is reduced -/
def differential_savings (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  income * old_rate - income * new_rate

/-- Proves that the differential savings for a specific case is correct -/
theorem differential_savings_example : 
  differential_savings 34500 0.42 0.28 = 4830 := by
  sorry

end NUMINAMATH_CALUDE_differential_savings_example_l2360_236054


namespace NUMINAMATH_CALUDE_divisors_multiple_of_five_3780_l2360_236064

/-- The number of positive divisors of 3780 that are multiples of 5 -/
def divisors_multiple_of_five (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d % 5 = 0) (Nat.divisors n)).card

/-- Theorem stating that the number of positive divisors of 3780 that are multiples of 5 is 24 -/
theorem divisors_multiple_of_five_3780 :
  divisors_multiple_of_five 3780 = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisors_multiple_of_five_3780_l2360_236064


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2360_236095

theorem root_equation_implies_expression_value (a : ℝ) :
  a^2 - 2*a - 2 = 0 →
  (1 - 1/(a + 1)) / (a^3 / (a^2 + 2*a + 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l2360_236095


namespace NUMINAMATH_CALUDE_four_propositions_true_l2360_236003

theorem four_propositions_true : 
  (∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (¬ ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
  (¬ ∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_true_l2360_236003


namespace NUMINAMATH_CALUDE_beaver_leaves_count_l2360_236050

theorem beaver_leaves_count :
  ∀ (beaver_dens raccoon_dens : ℕ),
    beaver_dens = raccoon_dens + 3 →
    5 * beaver_dens = 6 * raccoon_dens →
    5 * beaver_dens = 90 :=
by
  sorry

#check beaver_leaves_count

end NUMINAMATH_CALUDE_beaver_leaves_count_l2360_236050


namespace NUMINAMATH_CALUDE_function_max_min_range_l2360_236080

open Real

theorem function_max_min_range (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = m * sin (x + π/4) - Real.sqrt 2 * sin x) → 
  (∃ max min : ℝ, ∀ x ∈ Set.Ioo 0 (7*π/6), f x ≤ max ∧ min ≤ f x) →
  2 < m ∧ m < 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_function_max_min_range_l2360_236080


namespace NUMINAMATH_CALUDE_blocks_required_for_specified_wall_l2360_236092

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length₁ : ℕ
  length₂ : ℕ

/-- Calculates the number of blocks required for a wall with given specifications --/
def calculateBlocksRequired (wall : WallDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating that the number of blocks required for the specified wall is 404 --/
theorem blocks_required_for_specified_wall :
  let wall := WallDimensions.mk 150 8
  let block := BlockDimensions.mk 1 3 2
  calculateBlocksRequired wall block = 404 :=
by sorry

end NUMINAMATH_CALUDE_blocks_required_for_specified_wall_l2360_236092


namespace NUMINAMATH_CALUDE_special_triangle_sum_l2360_236063

/-- Triangle ABC with given side lengths and circles P and Q with specific properties -/
structure SpecialTriangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Radius of circle P
  radiusP : ℝ
  -- Radius of circle Q (to be determined)
  radiusQ : ℝ
  -- Conditions
  isosceles : AB = AC
  tangentP : radiusP < AB ∧ radiusP < BC
  tangentQ : radiusQ < AB ∧ radiusQ < BC
  externalTangent : radiusQ + radiusP < BC
  -- Representation of radiusQ
  m : ℕ
  n : ℕ
  k : ℕ
  radiusQForm : radiusQ = m - n * Real.sqrt k
  kPrime : Nat.Prime k

/-- The main theorem stating the sum of m and nk for the special triangle -/
theorem special_triangle_sum (t : SpecialTriangle) 
  (h1 : t.AB = 130)
  (h2 : t.BC = 150)
  (h3 : t.radiusP = 20) :
  t.m + t.n * t.k = 386 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sum_l2360_236063


namespace NUMINAMATH_CALUDE_bridget_apples_proof_l2360_236058

/-- Represents the number of apples Bridget initially bought -/
def initial_apples : ℕ := 20

/-- Represents the number of apples Bridget ate -/
def apples_eaten : ℕ := 2

/-- Represents the number of apples Bridget gave to Cassie -/
def apples_to_cassie : ℕ := 5

/-- Represents the number of apples Bridget kept for herself -/
def apples_kept : ℕ := 6

theorem bridget_apples_proof :
  let remaining_after_eating := initial_apples - apples_eaten
  let remaining_after_ann := remaining_after_eating - (remaining_after_eating / 3)
  let final_remaining := remaining_after_ann - apples_to_cassie
  final_remaining = apples_kept :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_proof_l2360_236058


namespace NUMINAMATH_CALUDE_function_property_l2360_236044

/-- Given a function f(x) = ax^5 + bx^3 + 3 where f(2023) = 16, prove that f(-2023) = -10 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 + 3
  f 2023 = 16 → f (-2023) = -10 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2360_236044


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2360_236091

theorem fourteenth_root_of_unity : 
  ∃ n : ℕ, n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 14)) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2360_236091


namespace NUMINAMATH_CALUDE_nine_digit_divisibility_l2360_236019

theorem nine_digit_divisibility (A : ℕ) : 
  A < 10 →
  (457319808 * 10 + A) % 2 = 0 →
  (457319808 * 10 + A) % 5 = 0 →
  (457319808 * 10 + A) % 8 = 0 →
  (457319808 * 10 + A) % 10 = 0 →
  (457319808 * 10 + A) % 16 = 0 →
  A = 0 := by
sorry

end NUMINAMATH_CALUDE_nine_digit_divisibility_l2360_236019
