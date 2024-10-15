import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l2619_261918

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem sum_of_factorials_perfect_square (n : ℕ) :
  isPerfectSquare (sumOfFactorials n) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorials_perfect_square_l2619_261918


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2619_261914

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2619_261914


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2619_261974

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 678 [ZMOD 11] ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬(19 * m ≡ 678 [ZMOD 11])) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2619_261974


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2619_261910

theorem adult_tickets_sold (adult_price student_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 6)
  (h2 : student_price = 3)
  (h3 : total_tickets = 846)
  (h4 : total_revenue = 3846) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * student_price = total_revenue ∧
    adult_tickets = 436 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2619_261910


namespace NUMINAMATH_CALUDE_juvenile_female_percentage_l2619_261905

/-- Represents the population of alligators on Lagoon Island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  adult_females : ℕ
  juvenile_females : ℕ

/-- Conditions for the Lagoon Island alligator population -/
def lagoon_conditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.total / 2 ∧
  pop.males = 25 ∧
  pop.adult_females = 15 ∧
  pop.juvenile_females = pop.total / 2 - pop.adult_females

/-- Theorem: The percentage of juvenile female alligators is 40% -/
theorem juvenile_female_percentage (pop : AlligatorPopulation) 
  (h : lagoon_conditions pop) : 
  (pop.juvenile_females : ℚ) / (pop.total / 2 : ℚ) = 2/5 := by
  sorry

#check juvenile_female_percentage

end NUMINAMATH_CALUDE_juvenile_female_percentage_l2619_261905


namespace NUMINAMATH_CALUDE_slope_of_MN_constant_sum_of_reciprocals_l2619_261916

/- Ellipse C₁ -/
def C₁ (b : ℝ) (x y : ℝ) : Prop := x^2/8 + y^2/b^2 = 1 ∧ b > 0

/- Parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 8*x

/- Right focus F₂ -/
def F₂ : ℝ × ℝ := (2, 0)

/- Theorem for the slope of line MN -/
theorem slope_of_MN (b : ℝ) (M N : ℝ × ℝ) :
  C₁ b M.1 M.2 → C₁ b N.1 N.2 → ((M.1 + N.1)/2, (M.2 + N.2)/2) = (1, 1) →
  (N.2 - M.2) / (N.1 - M.1) = -1/2 :=
sorry

/- Theorem for the constant sum of reciprocals -/
theorem constant_sum_of_reciprocals (b : ℝ) (A B C D : ℝ × ℝ) (m n : ℝ) :
  C₁ b A.1 A.2 → C₁ b B.1 B.2 → C₁ b C.1 C.2 → C₁ b D.1 D.2 →
  ((A.1 - F₂.1) * (B.1 - F₂.1) + (A.2 - F₂.2) * (B.2 - F₂.2) = 0) →
  ((C.1 - F₂.1) * (D.1 - F₂.1) + (C.2 - F₂.2) * (D.2 - F₂.2) = 0) →
  m = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) →
  n = Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) →
  1/m + 1/n = 3 * Real.sqrt 2 / 8 :=
sorry

end NUMINAMATH_CALUDE_slope_of_MN_constant_sum_of_reciprocals_l2619_261916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2619_261986

/-- 
Given an arithmetic sequence of consecutive integers where:
- k is a natural number
- The first term is k^2 - 1
- The number of terms is 2k - 1

The sum of all terms in this sequence is equal to 2k^3 + k^2 - 5k + 2
-/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let first_term := k^2 - 1
  let num_terms := 2*k - 1
  let last_term := first_term + (num_terms - 1)
  (num_terms : ℝ) * (first_term + last_term) / 2 = 2*k^3 + k^2 - 5*k + 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2619_261986


namespace NUMINAMATH_CALUDE_puffy_muffy_weight_l2619_261994

/-- The weight of Scruffy in ounces -/
def scruffy_weight : ℕ := 12

/-- The weight difference between Scruffy and Muffy in ounces -/
def muffy_scruffy_diff : ℕ := 3

/-- The weight difference between Puffy and Muffy in ounces -/
def puffy_muffy_diff : ℕ := 5

/-- The combined weight of Puffy and Muffy in ounces -/
def combined_weight : ℕ := scruffy_weight - muffy_scruffy_diff + (scruffy_weight - muffy_scruffy_diff + puffy_muffy_diff)

theorem puffy_muffy_weight : combined_weight = 23 := by
  sorry

end NUMINAMATH_CALUDE_puffy_muffy_weight_l2619_261994


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2619_261906

/-- An isosceles triangle with two sides of lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) →
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2619_261906


namespace NUMINAMATH_CALUDE_parallel_condition_l2619_261924

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem parallel_condition (l m : Line) (α : Plane)
  (h1 : ¬ subset l α)
  (h2 : subset m α) :
  (∀ l m, parallel_lines l m → parallel_line_plane l α) ∧
  (∃ l m, parallel_line_plane l α ∧ ¬ parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l2619_261924


namespace NUMINAMATH_CALUDE_duplicate_page_number_l2619_261917

theorem duplicate_page_number (n : ℕ) (p : ℕ) : 
  (n ≥ 70) →
  (n ≤ 71) →
  (n * (n + 1)) / 2 + p = 2550 →
  p = 80 := by
sorry

end NUMINAMATH_CALUDE_duplicate_page_number_l2619_261917


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l2619_261900

/-- Represents a circular table with chairs -/
structure CircularTable :=
  (num_chairs : ℕ)

/-- Represents a group of married couples -/
structure MarriedCouples :=
  (num_couples : ℕ)

/-- Represents the constraints for seating arrangements -/
structure SeatingConstraints :=
  (alternate_gender : Bool)
  (no_adjacent_spouses : Bool)
  (no_opposite_spouses : Bool)

/-- Calculates the number of valid seating arrangements -/
noncomputable def count_seating_arrangements (table : CircularTable) (couples : MarriedCouples) (constraints : SeatingConstraints) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem seating_arrangement_count :
  ∀ (table : CircularTable) (couples : MarriedCouples) (constraints : SeatingConstraints),
    table.num_chairs = 10 →
    couples.num_couples = 5 →
    constraints.alternate_gender = true →
    constraints.no_adjacent_spouses = true →
    constraints.no_opposite_spouses = true →
    count_seating_arrangements table couples constraints = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l2619_261900


namespace NUMINAMATH_CALUDE_great_8_teams_l2619_261936

-- Define the number of teams
def n : ℕ := sorry

-- Define the total number of games
def total_games : ℕ := 36

-- Theorem stating the conditions and the result to be proven
theorem great_8_teams :
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < total_games) ∧
  (n * (n - 1) / 2 = total_games) →
  n = 9 := by sorry

end NUMINAMATH_CALUDE_great_8_teams_l2619_261936


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_3780_l2619_261941

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_3780 :
  let factorization := prime_factorization 3780
  (factorization = [(2, 2), (3, 3), (5, 1), (7, 2)]) →
  count_perfect_square_factors 3780 = 8 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_3780_l2619_261941


namespace NUMINAMATH_CALUDE_newspaper_circulation_estimate_l2619_261947

/-- Estimated circulation of a newspaper given survey results -/
theorem newspaper_circulation_estimate 
  (city_population : ℕ) 
  (survey_size : ℕ) 
  (buyers_in_survey : ℕ) 
  (h1 : city_population = 8000000)
  (h2 : survey_size = 2500)
  (h3 : buyers_in_survey = 500) :
  (buyers_in_survey : ℚ) / survey_size * (city_population / 10000) = 160 := by
  sorry

#check newspaper_circulation_estimate

end NUMINAMATH_CALUDE_newspaper_circulation_estimate_l2619_261947


namespace NUMINAMATH_CALUDE_shed_length_calculation_l2619_261990

theorem shed_length_calculation (backyard_length backyard_width shed_width sod_area : ℝ) :
  backyard_length = 20 ∧
  backyard_width = 13 ∧
  shed_width = 5 ∧
  sod_area = 245 →
  backyard_length * backyard_width - sod_area = shed_width * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shed_length_calculation_l2619_261990


namespace NUMINAMATH_CALUDE_fraction_equality_l2619_261942

theorem fraction_equality (w x y : ℚ) 
  (h1 : w / y = 3 / 4)
  (h2 : (x + y) / y = 13 / 4) :
  w / x = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2619_261942


namespace NUMINAMATH_CALUDE_additional_money_needed_l2619_261943

/-- The amount of money Cory has initially -/
def initial_money : ℚ := 20

/-- The cost of one pack of candies -/
def candy_pack_cost : ℚ := 49

/-- The number of candy packs Cory wants to buy -/
def num_packs : ℕ := 2

/-- Theorem: Given Cory's initial money and the cost of candy packs,
    the additional amount needed to buy two packs is $78.00 -/
theorem additional_money_needed :
  (candy_pack_cost * num_packs : ℚ) - initial_money = 78 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l2619_261943


namespace NUMINAMATH_CALUDE_fraction_order_l2619_261930

theorem fraction_order (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hab : a > b) : 
  (b / a < (b + c) / (a + c)) ∧ 
  ((b + c) / (a + c) < (a + d) / (b + d)) ∧ 
  ((a + d) / (b + d) < a / b) := by
sorry

end NUMINAMATH_CALUDE_fraction_order_l2619_261930


namespace NUMINAMATH_CALUDE_train_length_l2619_261944

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 100 → time_s = 9 → 
  ∃ (length_m : ℝ), abs (length_m - 250.02) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2619_261944


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2619_261939

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2619_261939


namespace NUMINAMATH_CALUDE_fencing_cost_distribution_impossible_equal_distribution_impossible_l2619_261992

/-- Represents the dimensions of the cottage settlement. -/
structure Settlement where
  n : ℕ
  m : ℕ

/-- Calculates the total cost of fencing for the entire settlement. -/
def totalFencingCost (s : Settlement) : ℕ :=
  10000 * (2 * s.n * s.m + s.n + s.m - 4)

/-- Calculates the sum of costs if equal numbers of residents spent 0, 10000, 30000, 40000,
    and the rest spent 20000 rubles. -/
def proposedCostSum (s : Settlement) : ℕ :=
  100000 + 20000 * (s.n * s.m - 4)

/-- Theorem stating that the proposed cost distribution is impossible. -/
theorem fencing_cost_distribution_impossible (s : Settlement) :
  totalFencingCost s ≠ proposedCostSum s :=
sorry

/-- Theorem stating that it's impossible to have equal numbers of residents spending
    0, 10000, 30000, 40000 rubles with the rest spending 20000 rubles. -/
theorem equal_distribution_impossible (s : Settlement) :
  ¬ ∃ (k : ℕ), k > 0 ∧ 
    s.n * s.m = 4 * k + (s.n * s.m - 4 * k) ∧
    totalFencingCost s = k * (0 + 10000 + 30000 + 40000) + (s.n * s.m - 4 * k) * 20000 :=
sorry

end NUMINAMATH_CALUDE_fencing_cost_distribution_impossible_equal_distribution_impossible_l2619_261992


namespace NUMINAMATH_CALUDE_inequality_implies_range_l2619_261938

theorem inequality_implies_range (a : ℝ) : 
  (∀ x : ℝ, |2*x + 1| + |x - 2| ≥ a^2 - a + 1/2) → 
  -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l2619_261938


namespace NUMINAMATH_CALUDE_b_completion_time_l2619_261958

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 2
def work_rate_B : ℚ := 1 / 6

-- Define the total work as 1
def total_work : ℚ := 1

-- Define the work done in one day by both A and B
def work_done_together : ℚ := work_rate_A + work_rate_B

-- Define the remaining work after one day
def remaining_work : ℚ := total_work - work_done_together

-- Theorem to prove
theorem b_completion_time :
  remaining_work / work_rate_B = 2 := by sorry

end NUMINAMATH_CALUDE_b_completion_time_l2619_261958


namespace NUMINAMATH_CALUDE_expression_simplification_l2619_261979

theorem expression_simplification (x : ℤ) (hx : x ≠ 0) :
  (x^3 - 3*x^2*(x+2) + 4*x*(x+2)^2 - (x+2)^3 + 2) / (x * (x+2)) = 2 / (x * (x+2)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2619_261979


namespace NUMINAMATH_CALUDE_sum_of_xy_l2619_261902

theorem sum_of_xy (x y : ℕ) : 
  x > 0 → y > 0 → x < 25 → y < 25 → x + y + x * y = 118 → x + y = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2619_261902


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2619_261966

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) :
  Real.sqrt (a - 2) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2619_261966


namespace NUMINAMATH_CALUDE_xsin2x_necessary_not_sufficient_l2619_261978

theorem xsin2x_necessary_not_sufficient (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (∀ x, (0 < x ∧ x < π/2) → (x * Real.sin x < 1 → x * Real.sin x * Real.sin x < 1)) ∧
  (∃ x, (0 < x ∧ x < π/2) ∧ x * Real.sin x * Real.sin x < 1 ∧ x * Real.sin x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_xsin2x_necessary_not_sufficient_l2619_261978


namespace NUMINAMATH_CALUDE_vector_perpendicular_l2619_261904

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, -3)

theorem vector_perpendicular : 
  let diff := (a.1 - b.1, a.2 - b.2)
  a.1 * diff.1 + a.2 * diff.2 = 0 := by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l2619_261904


namespace NUMINAMATH_CALUDE_magic_square_x_value_l2619_261953

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  entries : Matrix (Fin 3) (Fin 3) ℤ
  is_magic : ∀ (i j : Fin 3), 
    (entries i 0 + entries i 1 + entries i 2 = entries 0 0 + entries 0 1 + entries 0 2) ∧ 
    (entries 0 j + entries 1 j + entries 2 j = entries 0 0 + entries 0 1 + entries 0 2) ∧
    (entries 0 0 + entries 1 1 + entries 2 2 = entries 0 0 + entries 0 1 + entries 0 2) ∧
    (entries 0 2 + entries 1 1 + entries 2 0 = entries 0 0 + entries 0 1 + entries 0 2)

/-- The main theorem stating the value of x in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.entries 0 0 = x)
  (h2 : ms.entries 0 1 = 21)
  (h3 : ms.entries 0 2 = 70)
  (h4 : ms.entries 1 0 = 7) :
  x = 133 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l2619_261953


namespace NUMINAMATH_CALUDE_base8_563_to_base3_l2619_261912

/-- Converts a base 8 number to base 10 --/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Converts a base 10 number to base 3 --/
def base10ToBase3 (n : Nat) : List Nat :=
  sorry  -- Implementation details omitted

theorem base8_563_to_base3 :
  base10ToBase3 (base8ToBase10 563) = [1, 1, 1, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_base8_563_to_base3_l2619_261912


namespace NUMINAMATH_CALUDE_x_plus_y_equals_eight_l2619_261907

theorem x_plus_y_equals_eight (x y : ℝ) 
  (h1 : |x| - x + y = 8) 
  (h2 : x + |y| + y = 16) : 
  x + y = 8 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_eight_l2619_261907


namespace NUMINAMATH_CALUDE_contest_prize_money_l2619_261971

/-- The total prize money for a novel contest -/
def total_prize_money (
  total_novels : ℕ
  ) (first_prize second_prize third_prize remaining_prize : ℕ
  ) : ℕ :=
  first_prize + second_prize + third_prize + (total_novels - 3) * remaining_prize

/-- Theorem stating that the total prize money for the given contest is $800 -/
theorem contest_prize_money :
  total_prize_money 18 200 150 120 22 = 800 := by
  sorry

end NUMINAMATH_CALUDE_contest_prize_money_l2619_261971


namespace NUMINAMATH_CALUDE_smallest_number_l2619_261975

def number_set : Set ℤ := {-1, 0, 1, 2}

theorem smallest_number : ∀ x ∈ number_set, -1 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2619_261975


namespace NUMINAMATH_CALUDE_lily_of_valley_cost_price_l2619_261946

/-- The cost price of a pot of lily of the valley -/
def cost_price : ℝ := 2.4

/-- The selling price of a pot of lily of the valley -/
def selling_price : ℝ := cost_price * 1.25

/-- The number of pots sold -/
def num_pots : ℕ := 150

/-- The total revenue from selling the pots -/
def total_revenue : ℝ := 450

theorem lily_of_valley_cost_price :
  cost_price = 2.4 ∧
  selling_price = cost_price * 1.25 ∧
  (num_pots : ℝ) * selling_price = total_revenue :=
sorry

end NUMINAMATH_CALUDE_lily_of_valley_cost_price_l2619_261946


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l2619_261908

theorem rectangle_longer_side (r : ℝ) (h1 : r = 6) : ∃ L : ℝ,
  (L * (2 * r) = 3 * (π * r^2)) ∧ L = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l2619_261908


namespace NUMINAMATH_CALUDE_m_le_n_l2619_261982

theorem m_le_n (a b : ℝ) : 
  let m := (6^a) / (36^(a+1) + 1)
  let n := (1/3) * b^2 - b + 5/6
  m ≤ n := by sorry

end NUMINAMATH_CALUDE_m_le_n_l2619_261982


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2619_261967

theorem max_value_of_expression (x y : ℝ) : 
  2 * x^2 + 3 * y^2 = 22 * x + 18 * y + 20 →
  4 * x + 5 * y ≤ 110 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2619_261967


namespace NUMINAMATH_CALUDE_muffin_combinations_l2619_261980

/-- Given four kinds of muffins, when purchasing eight muffins with at least one of each kind,
    there are 23 different possible combinations. -/
theorem muffin_combinations : ℕ :=
  let num_muffin_types : ℕ := 4
  let total_muffins : ℕ := 8
  let min_of_each_type : ℕ := 1
  23

#check muffin_combinations

end NUMINAMATH_CALUDE_muffin_combinations_l2619_261980


namespace NUMINAMATH_CALUDE_set_operations_l2619_261901

def U : Set ℤ := {1, 2, 3, 4, 5, 6, 7, 8}

def A : Set ℤ := {x | x^2 - 3*x + 2 = 0}

def B : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}

def C : Set ℤ := {x | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5}) ∧
  ((U.compl ∩ B) ∪ (U.compl ∩ C) = {1, 2, 6, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2619_261901


namespace NUMINAMATH_CALUDE_sqrt_calculation_l2619_261931

theorem sqrt_calculation : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 - 3 * Real.sqrt 2 = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l2619_261931


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_y_l2619_261964

theorem max_value_of_x_plus_y : ∃ (max : ℤ),
  (max = 13) ∧
  (∀ x y : ℤ, 3 * x^2 + 5 * y^2 = 345 → x + y ≤ max) ∧
  (∃ x y : ℤ, 3 * x^2 + 5 * y^2 = 345 ∧ x + y = max) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_y_l2619_261964


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l2619_261913

theorem necessary_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b - a - b + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l2619_261913


namespace NUMINAMATH_CALUDE_g_monotonically_decreasing_l2619_261977

/-- The function g(x) defined in terms of parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the conditions for g(x) to be monotonically decreasing -/
theorem g_monotonically_decreasing (a : ℝ) :
  (∀ x < a / 3, g_derivative a x ≤ 0) ↔ -1 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_g_monotonically_decreasing_l2619_261977


namespace NUMINAMATH_CALUDE_x_value_l2619_261915

theorem x_value (x : ℚ) (h : 1/4 - 1/6 = 4/x) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2619_261915


namespace NUMINAMATH_CALUDE_valentine_treats_l2619_261956

/-- Represents the number of heart biscuits Mrs. Heine buys for each dog -/
def heart_biscuits_per_dog : ℕ := sorry

/-- Represents the total number of items Mrs. Heine buys -/
def total_items : ℕ := 12

/-- Represents the number of dogs -/
def num_dogs : ℕ := 2

/-- Represents the number of sets of puppy boots per dog -/
def puppy_boots_per_dog : ℕ := 1

theorem valentine_treats :
  heart_biscuits_per_dog * num_dogs + puppy_boots_per_dog * num_dogs = total_items ∧
  heart_biscuits_per_dog = 4 := by sorry

end NUMINAMATH_CALUDE_valentine_treats_l2619_261956


namespace NUMINAMATH_CALUDE_intersection_forms_hyperbola_l2619_261987

/-- The equation of the first line -/
def line1 (t x y : ℝ) : Prop := t * x - 2 * y - 3 * t = 0

/-- The equation of the second line -/
def line2 (t x y : ℝ) : Prop := x - 2 * t * y + 3 = 0

/-- The equation of a hyperbola -/
def is_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / (9/4) = 1

/-- Theorem stating that the intersection points form a hyperbola -/
theorem intersection_forms_hyperbola :
  ∀ t x y : ℝ, line1 t x y → line2 t x y → is_hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_intersection_forms_hyperbola_l2619_261987


namespace NUMINAMATH_CALUDE_china_coal_production_2003_l2619_261969

/-- Represents a large number in different formats -/
structure LargeNumber where
  value : Nat
  word_representation : String
  billion_representation : String

/-- Converts a natural number to its word representation -/
def nat_to_words (n : Nat) : String :=
  sorry

/-- Converts a natural number to its billion representation -/
def nat_to_billions (n : Nat) : String :=
  sorry

/-- Theorem stating the correct representations of China's coal production in 2003 -/
theorem china_coal_production_2003 :
  let production : LargeNumber := {
    value := 15500000000,
    word_representation := nat_to_words 15500000000,
    billion_representation := nat_to_billions 15500000000
  }
  production.word_representation = "one hundred and fifty-five billion" ∧
  production.billion_representation = "155 billion" :=
sorry

end NUMINAMATH_CALUDE_china_coal_production_2003_l2619_261969


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l2619_261965

theorem complex_subtraction_simplification :
  (5 - 3*I) - (2 + 7*I) = 3 - 10*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l2619_261965


namespace NUMINAMATH_CALUDE_triangle_area_upper_bound_l2619_261954

/-- Given a triangle ABC with BC = 2 and AB · AC = 1, prove that its area is at most √2 -/
theorem triangle_area_upper_bound (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let dot_product (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2
  let triangle_area := Real.sqrt (((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))^2 / 4)
  Real.sqrt ((BC.1^2 + BC.2^2) / 4) = 1 →
  dot_product AB AC = 1 →
  triangle_area ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_upper_bound_l2619_261954


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2619_261955

theorem no_natural_square_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2619_261955


namespace NUMINAMATH_CALUDE_smallest_c_in_arithmetic_progression_l2619_261959

theorem smallest_c_in_arithmetic_progression (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →
  a * b * c * d = 256 →
  ∀ x : ℝ, (∃ a' b' d' : ℝ, 
    0 < a' ∧ 0 < b' ∧ 0 < x ∧ 0 < d' ∧
    (∃ r' : ℝ, b' = a' + r' ∧ x = b' + r' ∧ d' = x + r') ∧
    a' * b' * x * d' = 256) →
  x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_in_arithmetic_progression_l2619_261959


namespace NUMINAMATH_CALUDE_cucumber_water_percentage_l2619_261949

/-- Calculates the new water percentage in cucumbers after evaporation -/
theorem cucumber_water_percentage
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 99)
  (h3 : final_weight = 20)
  : (final_weight - (initial_weight * (1 - initial_water_percentage / 100))) / final_weight * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_water_percentage_l2619_261949


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2619_261927

/-- Given a train that crosses a pole in a certain time, calculate its speed in kmph. -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) :
  train_length = 800.064 →
  crossing_time = 18 →
  (train_length / 1000) / (crossing_time / 3600) = 160.0128 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2619_261927


namespace NUMINAMATH_CALUDE_jeff_new_cabinet_counters_l2619_261929

/-- Calculates the number of counters over which new cabinets were installed --/
def counters_with_new_cabinets (initial_cabinets : ℕ) (cabinets_per_new_counter : ℕ) (additional_cabinets : ℕ) (total_cabinets : ℕ) : ℕ :=
  (total_cabinets - initial_cabinets - additional_cabinets) / cabinets_per_new_counter

/-- Proves that Jeff installed new cabinets over 9 counters --/
theorem jeff_new_cabinet_counters :
  let initial_cabinets := 3
  let cabinets_per_new_counter := 2
  let additional_cabinets := 5
  let total_cabinets := 26
  counters_with_new_cabinets initial_cabinets cabinets_per_new_counter additional_cabinets total_cabinets = 9 := by
  sorry

end NUMINAMATH_CALUDE_jeff_new_cabinet_counters_l2619_261929


namespace NUMINAMATH_CALUDE_train_length_l2619_261988

/-- The length of a train given its speed, time to pass a platform, and platform length. -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (platform_length : ℝ) :
  train_speed = 60 →
  time_to_pass = 23.998080153587715 →
  platform_length = 260 →
  (train_speed * 1000 / 3600) * time_to_pass - platform_length = 140 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2619_261988


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l2619_261995

def M : Set Nat := {1, 2, 3, 4, 5}
def N : Set Nat := {2, 5}

theorem complement_of_N_in_M :
  M \ N = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l2619_261995


namespace NUMINAMATH_CALUDE_function_max_abs_bound_l2619_261935

theorem function_max_abs_bound (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f (x : ℝ) := a * x^3 + (1 - 4*a) * x^2 + (5*a - 1) * x - 5*a + 3
  let g (x : ℝ) := (1 - a) * x^3 - x^2 + (2 - a) * x - 3*a - 1
  ∀ x : ℝ, max (|f x|) (|g x|) ≥ a + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_max_abs_bound_l2619_261935


namespace NUMINAMATH_CALUDE_A_inter_B_eq_A_l2619_261903

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x ≤ 3}

-- Theorem statement
theorem A_inter_B_eq_A : A ∩ B = A := by sorry

end NUMINAMATH_CALUDE_A_inter_B_eq_A_l2619_261903


namespace NUMINAMATH_CALUDE_minyoung_position_l2619_261973

/-- Given a line of people, calculates the position from the front given the total number of people and the position from the back. -/
def position_from_front (total : ℕ) (from_back : ℕ) : ℕ :=
  total - from_back + 1

/-- Proves that in a line of 27 people, if a person is 13th from the back, they are 15th from the front. -/
theorem minyoung_position :
  position_from_front 27 13 = 15 := by
  sorry

end NUMINAMATH_CALUDE_minyoung_position_l2619_261973


namespace NUMINAMATH_CALUDE_variance_implies_fluctuation_l2619_261970

-- Define a type for our data set
def DataSet := List ℝ

-- Define variance
def variance (data : DataSet) : ℝ := sorry

-- Define a measure of fluctuation
def fluctuation (data : DataSet) : ℝ := sorry

-- Theorem statement
theorem variance_implies_fluctuation (data1 data2 : DataSet) :
  variance data1 > variance data2 → fluctuation data1 > fluctuation data2 := by
  sorry

end NUMINAMATH_CALUDE_variance_implies_fluctuation_l2619_261970


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l2619_261960

/-- The parabola y = ax^2 + 10 is tangent to the line y = 2x + 3 if and only if a = 1/7 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 10 = 2 * x + 3) ↔ a = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l2619_261960


namespace NUMINAMATH_CALUDE_eraser_pencil_price_ratio_l2619_261976

/-- Represents the store's sales and pricing structure -/
structure StoreSales where
  pencils_sold : ℕ
  total_earnings : ℕ
  eraser_price : ℕ
  pencil_eraser_ratio : ℕ

/-- Theorem stating the ratio of eraser price to pencil price -/
theorem eraser_pencil_price_ratio 
  (s : StoreSales) 
  (h1 : s.pencils_sold = 20)
  (h2 : s.total_earnings = 80)
  (h3 : s.eraser_price = 1)
  (h4 : s.pencil_eraser_ratio = 2) : 
  (s.eraser_price : ℚ) / ((s.total_earnings - s.eraser_price * s.pencils_sold * s.pencil_eraser_ratio) / s.pencils_sold : ℚ) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_eraser_pencil_price_ratio_l2619_261976


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l2619_261996

theorem students_liking_both_desserts
  (total_students : ℕ)
  (ice_cream_fans : ℕ)
  (cookie_fans : ℕ)
  (neither_fans : ℕ)
  (h1 : total_students = 50)
  (h2 : ice_cream_fans = 28)
  (h3 : cookie_fans = 20)
  (h4 : neither_fans = 14) :
  total_students - neither_fans - (ice_cream_fans + cookie_fans - total_students + neither_fans) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l2619_261996


namespace NUMINAMATH_CALUDE_incorrect_translation_l2619_261926

/-- Represents a parabola of the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Checks if a parabola passes through the origin -/
def passes_through_origin (p : Parabola) : Prop :=
  0 = (0 + p.a)^2 + p.b

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - d }

theorem incorrect_translation :
  let original := Parabola.mk 3 (-4)
  let translated := translate_vertical original 4
  ¬ passes_through_origin translated :=
by sorry

end NUMINAMATH_CALUDE_incorrect_translation_l2619_261926


namespace NUMINAMATH_CALUDE_standard_form_of_negative_r_l2619_261951

/-- Converts a polar coordinate point to its standard form where r > 0 and 0 ≤ θ < 2π -/
def standardPolarForm (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  sorry

theorem standard_form_of_negative_r :
  let original : ℝ × ℝ := (-3, π/6)
  let standard : ℝ × ℝ := standardPolarForm original.1 original.2
  standard = (3, 7*π/6) ∧ standard.1 > 0 ∧ 0 ≤ standard.2 ∧ standard.2 < 2*π :=
by sorry

end NUMINAMATH_CALUDE_standard_form_of_negative_r_l2619_261951


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2619_261961

theorem unique_solution_quadratic_system :
  ∃! x : ℚ, (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) ∧ (x = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l2619_261961


namespace NUMINAMATH_CALUDE_max_min_diff_abs_sum_ratio_l2619_261981

/-- The difference between the maximum and minimum values of |a + b| / (|a| + |b|) for nonzero real numbers a and b is 1. -/
theorem max_min_diff_abs_sum_ratio : ∃ (m' M' : ℝ),
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → m' ≤ |a + b| / (|a| + |b|) ∧ |a + b| / (|a| + |b|) ≤ M') ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ |a + b| / (|a| + |b|) = m') ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ |a + b| / (|a| + |b|) = M') ∧
  M' - m' = 1 := by
sorry

end NUMINAMATH_CALUDE_max_min_diff_abs_sum_ratio_l2619_261981


namespace NUMINAMATH_CALUDE_stratified_sampling_young_representatives_l2619_261934

/-- Represents the number of young representatives to be selected in a stratified sampling scenario. -/
def young_representatives (total_population : ℕ) (young_population : ℕ) (total_representatives : ℕ) : ℕ :=
  (young_population * total_representatives) / total_population

/-- Theorem stating that for the given population numbers and sampling size, 
    the number of young representatives to be selected is 7. -/
theorem stratified_sampling_young_representatives :
  young_representatives 1000 350 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_young_representatives_l2619_261934


namespace NUMINAMATH_CALUDE_no_k_exists_for_not_in_second_quadrant_l2619_261972

/-- A linear function that does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x y : ℝ, y = (k - 1) * x + k → (x < 0 → y ≤ 0)

/-- Theorem stating that there is no k for which the linear function y=(k-1)x+k does not pass through the second quadrant -/
theorem no_k_exists_for_not_in_second_quadrant :
  ¬ ∃ k : ℝ, not_in_second_quadrant k :=
sorry

end NUMINAMATH_CALUDE_no_k_exists_for_not_in_second_quadrant_l2619_261972


namespace NUMINAMATH_CALUDE_probability_exactly_two_eights_value_l2619_261985

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def target_value : ℕ := 8
def num_target : ℕ := 2

def probability_exactly_two_eights : ℚ :=
  (Nat.choose num_dice num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem probability_exactly_two_eights_value :
  probability_exactly_two_eights = 28 * 117649 / 16777216 :=
sorry

end NUMINAMATH_CALUDE_probability_exactly_two_eights_value_l2619_261985


namespace NUMINAMATH_CALUDE_base_conversion_sum_equality_l2619_261922

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_conversion_sum_equality : 
  let num1 := base_to_decimal [2, 5, 3] 8
  let den1 := base_to_decimal [1, 3] 4
  let num2 := base_to_decimal [1, 4, 4] 5
  let den2 := base_to_decimal [3, 3] 3
  (num1 : ℚ) / den1 + (num2 : ℚ) / den2 = 28.511904 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_equality_l2619_261922


namespace NUMINAMATH_CALUDE_daughter_least_intelligent_l2619_261952

-- Define the types for people and intelligence levels
inductive Person : Type
| Father : Person
| Sister : Person
| Son : Person
| Daughter : Person

inductive IntelligenceLevel : Type
| Least : IntelligenceLevel
| Smartest : IntelligenceLevel

-- Define the properties
def isTwin (p1 p2 : Person) : Prop := sorry

def sex (p : Person) : Bool := sorry

def age (p : Person) : ℕ := sorry

def intelligenceLevel (p : Person) : IntelligenceLevel := sorry

-- Define the theorem
theorem daughter_least_intelligent 
  (h1 : ∀ p1 p2 : Person, intelligenceLevel p1 = IntelligenceLevel.Least → 
        intelligenceLevel p2 = IntelligenceLevel.Smartest → 
        (∃ p3 : Person, isTwin p1 p3 ∧ sex p3 ≠ sex p2))
  (h2 : ∀ p1 p2 : Person, intelligenceLevel p1 = IntelligenceLevel.Least → 
        intelligenceLevel p2 = IntelligenceLevel.Smartest → 
        age p1 = age p2)
  : intelligenceLevel Person.Daughter = IntelligenceLevel.Least := by
  sorry

end NUMINAMATH_CALUDE_daughter_least_intelligent_l2619_261952


namespace NUMINAMATH_CALUDE_inequality_proof_l2619_261948

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2619_261948


namespace NUMINAMATH_CALUDE_shaded_area_is_twelve_l2619_261984

-- Define the rectangle
structure Rectangle where
  base : ℝ
  height : ℝ

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem setup
def problemSetup (rect : Rectangle) (tri : IsoscelesTriangle) : Prop :=
  rect.height = tri.height ∧
  rect.base = 12 ∧
  rect.height = 8 ∧
  tri.base = 12

-- Define the intersection point
def intersectionPoint : Point :=
  { x := 18, y := 2 }

-- Theorem statement
theorem shaded_area_is_twelve (rect : Rectangle) (tri : IsoscelesTriangle) 
  (h : problemSetup rect tri) : 
  (1/2 : ℝ) * tri.base * intersectionPoint.y = 12 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_twelve_l2619_261984


namespace NUMINAMATH_CALUDE_defective_pens_probability_l2619_261925

theorem defective_pens_probability (total_pens : Nat) (defective_pens : Nat) (bought_pens : Nat) :
  total_pens = 10 →
  defective_pens = 2 →
  bought_pens = 2 →
  (((total_pens - defective_pens : ℚ) / total_pens) * 
   ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1))) = 0.6222222222222222 := by
  sorry

end NUMINAMATH_CALUDE_defective_pens_probability_l2619_261925


namespace NUMINAMATH_CALUDE_fraction_as_power_series_l2619_261919

theorem fraction_as_power_series :
  ∃ (a : ℕ → ℚ), (9 : ℚ) / 10 = (5 : ℚ) / 6 + ∑' n, a n / (6 ^ (n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_as_power_series_l2619_261919


namespace NUMINAMATH_CALUDE_two_bags_below_threshold_probability_l2619_261932

-- Define the normal distribution parameters
def μ : ℝ := 500
def σ : ℝ := 5

-- Define the threshold weight
def threshold : ℝ := 485

-- Define the probability of selecting one bag below the threshold
def prob_one_bag : ℝ := 0.0013

-- Theorem statement
theorem two_bags_below_threshold_probability :
  let prob_two_bags := prob_one_bag * prob_one_bag
  prob_two_bags < 2e-6 := by sorry

end NUMINAMATH_CALUDE_two_bags_below_threshold_probability_l2619_261932


namespace NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2619_261950

/-- The circle with center (2, -1) and radius √2 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 2}

/-- A line passing through the origin with slope k -/
def tangentLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

/-- The set of slopes of lines passing through the origin and tangent to C -/
def tangentSlopes : Set ℝ :=
  {k : ℝ | ∃ p ∈ C, p ∈ tangentLine k ∧ (0, 0) ∈ tangentLine k}

theorem sum_of_tangent_slopes :
  ∃ (k₁ k₂ : ℝ), k₁ ∈ tangentSlopes ∧ k₂ ∈ tangentSlopes ∧ k₁ + k₂ = -2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_tangent_slopes_l2619_261950


namespace NUMINAMATH_CALUDE_total_balloons_l2619_261968

theorem total_balloons (allan_balloons jake_balloons maria_balloons : ℕ) 
  (h1 : allan_balloons = 5)
  (h2 : jake_balloons = 7)
  (h3 : maria_balloons = 3) :
  allan_balloons + jake_balloons + maria_balloons = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l2619_261968


namespace NUMINAMATH_CALUDE_max_missed_questions_correct_l2619_261991

/-- The number of questions in the test -/
def total_questions : ℕ := 50

/-- The minimum passing percentage -/
def passing_percentage : ℚ := 85 / 100

/-- The greatest number of questions a student can miss and still pass -/
def max_missed_questions : ℕ := 7

theorem max_missed_questions_correct :
  max_missed_questions = ⌊(1 - passing_percentage) * total_questions⌋ := by
  sorry

end NUMINAMATH_CALUDE_max_missed_questions_correct_l2619_261991


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2619_261909

-- Define the hyperbola
def hyperbola (x y : ℝ) := y^2 - x^2 = 2

-- Define the foci
def foci : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Define the asymptotes
def asymptotes (x y : ℝ) := x^2/3 - y^2/3 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ (x y : ℝ),
  (∃ (f : ℝ × ℝ), f ∈ foci) →
  (∀ (x' y' : ℝ), asymptotes x' y' ↔ asymptotes x y) →
  hyperbola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2619_261909


namespace NUMINAMATH_CALUDE_exactly_two_pairs_exist_l2619_261928

-- Define the type for a pair of real numbers
def RealPair := ℝ × ℝ

-- Define a function to check if two lines are identical
def are_lines_identical (b c : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (2 = k * c) ∧ 
    (3 * b = k * 4) ∧ 
    (c = k * 16)

-- Define the set of pairs (b, c) that make the lines identical
def identical_line_pairs : Set RealPair :=
  {p : RealPair | are_lines_identical p.1 p.2}

-- Theorem statement
theorem exactly_two_pairs_exist : 
  ∃ (p₁ p₂ : RealPair), p₁ ≠ p₂ ∧ 
    p₁ ∈ identical_line_pairs ∧ 
    p₂ ∈ identical_line_pairs ∧ 
    ∀ (p : RealPair), p ∈ identical_line_pairs → p = p₁ ∨ p = p₂ :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_pairs_exist_l2619_261928


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2619_261921

def p (x : ℝ) : ℝ := 5*x^9 - 3*x^7 + 4*x^6 - 8*x^4 + 3*x^3 - 6*x + 5

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, p = λ x => (3*x - 6) * q x + 2321 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2619_261921


namespace NUMINAMATH_CALUDE_triangle_area_16_triangle_AED_area_l2619_261957

/-- The area of a triangle with base 8 and height 4 is 16 square units. -/
theorem triangle_area_16 (base height : ℝ) (h1 : base = 8) (h2 : height = 4) :
  (1 / 2) * base * height = 16 := by sorry

/-- Given a triangle AED where AE = 8, height = 4, and ED = DA = 5,
    the area of triangle AED is 16 square units. -/
theorem triangle_AED_area (AE ED DA height : ℝ)
  (h1 : AE = 8)
  (h2 : height = 4)
  (h3 : ED = 5)
  (h4 : DA = 5) :
  (1 / 2) * AE * height = 16 := by sorry

end NUMINAMATH_CALUDE_triangle_area_16_triangle_AED_area_l2619_261957


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l2619_261962

/-- Proves that the cost of an adult ticket is 8 dollars given the specified conditions -/
theorem adult_ticket_cost
  (total_attendees : ℕ)
  (num_children : ℕ)
  (child_ticket_cost : ℕ)
  (total_revenue : ℕ)
  (h1 : total_attendees = 22)
  (h2 : num_children = 18)
  (h3 : child_ticket_cost = 1)
  (h4 : total_revenue = 50) :
  (total_revenue - num_children * child_ticket_cost) / (total_attendees - num_children) = 8 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l2619_261962


namespace NUMINAMATH_CALUDE_shirt_final_price_l2619_261920

/-- The final price of a shirt after two successive discounts --/
theorem shirt_final_price (list_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  list_price = 150 → 
  discount1 = 19.954259576901087 →
  discount2 = 12.55 →
  list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = 105 := by
sorry

end NUMINAMATH_CALUDE_shirt_final_price_l2619_261920


namespace NUMINAMATH_CALUDE_no_solution_exists_l2619_261993

theorem no_solution_exists : ¬∃ n : ℤ,
  50 ≤ n ∧ n ≤ 150 ∧
  8 ∣ n ∧
  n % 10 = 6 ∧
  n % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2619_261993


namespace NUMINAMATH_CALUDE_sqrt_calculation_l2619_261997

theorem sqrt_calculation : Real.sqrt (1/2) * Real.sqrt 8 - (Real.sqrt 3)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l2619_261997


namespace NUMINAMATH_CALUDE_cos_x_plus_7pi_12_l2619_261999

theorem cos_x_plus_7pi_12 (x : ℝ) (h : Real.sin (x + π / 12) = 1 / 3) :
  Real.cos (x + 7 * π / 12) = - 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_7pi_12_l2619_261999


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2619_261911

/-- A three-digit number composed of distinct digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ tens ≠ ones ∧ hundreds ≠ ones
  valid_range : hundreds ∈ Finset.range 10 ∧ tens ∈ Finset.range 10 ∧ ones ∈ Finset.range 10

/-- Check if one digit is the average of the other two -/
def has_average_digit (n : ThreeDigitNumber) : Prop :=
  2 * n.hundreds = n.tens + n.ones ∨
  2 * n.tens = n.hundreds + n.ones ∨
  2 * n.ones = n.hundreds + n.tens

/-- Check if the sum of digits is divisible by 3 -/
def sum_divisible_by_three (n : ThreeDigitNumber) : Prop :=
  (n.hundreds + n.tens + n.ones) % 3 = 0

/-- The set of all valid three-digit numbers satisfying the conditions -/
def valid_numbers : Finset ThreeDigitNumber :=
  sorry

theorem count_valid_numbers : valid_numbers.card = 160 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2619_261911


namespace NUMINAMATH_CALUDE_integer_root_values_l2619_261923

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 2*x^2 + b*x + 8 = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-81, -26, -19, -12, -11, 4, 9, 47} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l2619_261923


namespace NUMINAMATH_CALUDE_product_of_complements_bound_l2619_261983

theorem product_of_complements_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≥ 0) (h₂ : x₂ ≥ 0) (h₃ : x₃ ≥ 0) 
  (h_sum : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_complements_bound_l2619_261983


namespace NUMINAMATH_CALUDE_mirror_area_l2619_261989

theorem mirror_area (frame_width frame_height frame_thickness : ℕ) : 
  frame_width = 100 ∧ frame_height = 120 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 6300 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l2619_261989


namespace NUMINAMATH_CALUDE_no_real_roots_iff_range_m_range_when_necessary_not_sufficient_l2619_261940

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop :=
  x^2 - 2*a*x + 2*a^2 - a - 6 = 0

-- Define the range of a for no real roots
def no_real_roots (a : ℝ) : Prop :=
  a < -2 ∨ a > 3

-- Define the necessary condition
def necessary_condition (a : ℝ) : Prop :=
  -2 ≤ a ∧ a ≤ 3

-- Define the condition q
def condition_q (m a : ℝ) : Prop :=
  m - 1 ≤ a ∧ a ≤ m + 3

-- Theorem 1: The equation has no real roots iff a is in the specified range
theorem no_real_roots_iff_range (a : ℝ) :
  (∀ x : ℝ, ¬(quadratic_equation a x)) ↔ no_real_roots a :=
sorry

-- Theorem 2: If the necessary condition is true but not sufficient for condition q,
-- then m is in the range [-1, 0]
theorem m_range_when_necessary_not_sufficient :
  (∀ a : ℝ, condition_q m a → necessary_condition a) ∧
  (∃ a : ℝ, necessary_condition a ∧ ¬(condition_q m a)) →
  -1 ≤ m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_range_m_range_when_necessary_not_sufficient_l2619_261940


namespace NUMINAMATH_CALUDE_count_triples_eq_200_l2619_261945

/-- Counts the number of ways to partition a positive integer into two positive integers -/
def partitionCount (n : ℕ) : ℕ := if n ≤ 1 then 0 else n - 1

/-- Counts the number of ordered triples (a,b,c) satisfying the given conditions -/
def countTriples : ℕ :=
  (partitionCount 3) + (partitionCount 4) + (partitionCount 9) +
  (partitionCount 19) + (partitionCount 24) + (partitionCount 49) +
  (partitionCount 99)

theorem count_triples_eq_200 :
  countTriples = 200 :=
sorry

end NUMINAMATH_CALUDE_count_triples_eq_200_l2619_261945


namespace NUMINAMATH_CALUDE_clock_angle_at_7_proof_l2619_261963

/-- The smaller angle formed by the hands of a clock at 7 o'clock -/
def clock_angle_at_7 : ℝ := 150

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- The number of hour points on a clock -/
def clock_hour_points : ℕ := 12

/-- The position of the hour hand at 7 o'clock -/
def hour_hand_position : ℕ := 7

/-- The position of the minute hand at 7 o'clock -/
def minute_hand_position : ℕ := 12

theorem clock_angle_at_7_proof :
  clock_angle_at_7 = (minute_hand_position - hour_hand_position) * (full_circle_degrees / clock_hour_points) :=
by sorry

end NUMINAMATH_CALUDE_clock_angle_at_7_proof_l2619_261963


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2619_261933

theorem fractional_equation_solution_range (x m : ℝ) : 
  ((2 * x - m) / (x + 1) = 3) → 
  (x < 0) → 
  (m > -3 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2619_261933


namespace NUMINAMATH_CALUDE_max_green_cards_achievable_green_cards_l2619_261937

/-- Represents the number of cards of each color in the box -/
structure CardCount where
  green : ℕ
  yellow : ℕ

/-- The probability of selecting three cards of the same color -/
def prob_same_color (cc : CardCount) : ℚ :=
  let total := cc.green + cc.yellow
  (cc.green.choose 3 + cc.yellow.choose 3) / total.choose 3

/-- The main theorem stating the maximum number of green cards possible -/
theorem max_green_cards (cc : CardCount) : 
  cc.green + cc.yellow ≤ 2209 →
  prob_same_color cc = 1/3 →
  cc.green ≤ 1092 := by
  sorry

/-- The theorem stating that 1092 green cards is achievable -/
theorem achievable_green_cards : 
  ∃ (cc : CardCount), cc.green + cc.yellow ≤ 2209 ∧ 
  prob_same_color cc = 1/3 ∧ 
  cc.green = 1092 := by
  sorry

end NUMINAMATH_CALUDE_max_green_cards_achievable_green_cards_l2619_261937


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l2619_261998

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_vertex_x_coordinate
  (a b c : ℝ)
  (h1 : f a b c 0 = 0)
  (h2 : f a b c 4 = 0)
  (h3 : f a b c 3 = 9) :
  -b / (2 * a) = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l2619_261998
